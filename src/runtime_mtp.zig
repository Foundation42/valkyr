//! GPU recorder for the Multi-Token Prediction head.
//!
//! Pairs with the CPU oracle in `cpu/mtp.zig`. Single-position decode
//! at `pos`: takes a previous-slot hidden state + an input token, runs
//! it through the MTP head module, writes next-slot hidden + logits
//! into caller-owned buffers.
//!
//! Reuses `runtime_hybrid.ChatKernels` + `Scratch` + `computeForwardPushes`
//! so the MTP path picks up every kernel/precision the chat path uses.
//! The MTP-specific buffers (e_norm, h_norm, concat, h_out + per-MTP-
//! layer KV caches) live in `MtpRuntimeState`. The transformer-block
//! dispatch chain is inlined (mirrors the `full_attention` branch of
//! `runtime_hybrid.recordOneLayer`) — copy-paste rather than refactor
//! to keep the existing chat path unmodified.
//!
//! Used by:
//!   - the MTP-1b-β-2 GPU/CPU parity smoke (single-step gate);
//!   - the MTP-1c draft+verify chain (next chunk).

const std = @import("std");
const vk = @import("gpu/vk.zig");
const buffer = @import("gpu/buffer.zig");
const recorder = @import("gpu/recorder.zig");
const config_mod = @import("config.zig");
const gpu_model = @import("gpu/model.zig");
const runtime = @import("runtime.zig");
const runtime_hybrid = @import("runtime_hybrid.zig");

/// Per-call MTP runtime buffers. Owned. `init` zeroes the KV caches
/// (fresh start). Caller reuses across slots / draft positions; rewind
/// for verify-rejected slots will land in MTP-1d.
pub const MtpRuntimeState = struct {
    /// Per-MTP-layer K/V caches, sized `[max_pos × n_kv_heads × head_dim]`
    /// fp32 each. One pair per `cfg.mtp_num_hidden_layers` block.
    kv_k: []buffer.Buffer,
    kv_v: []buffer.Buffer,
    /// `[hidden]` fp32. Holds `embed_tokens[token_id]` before the
    /// pre_fc_norm_embedding pass.
    embed_tmp: buffer.Buffer,
    /// `[hidden]` fp32. RMSNormed embedding side.
    e_norm: buffer.Buffer,
    /// `[hidden]` fp32. RMSNormed hidden-prev side.
    h_norm: buffer.Buffer,
    /// `[2 * hidden]` fp32. `[e_norm; h_norm]`, fed into `mtp.fc`.
    concat: buffer.Buffer,
    /// `[hidden]` fp32. Pre-final-norm MTP output — fed into the next
    /// slot as `h_prev` (matches DeepSeek-V3 §2.3).
    h_out: buffer.Buffer,
    allocator: std.mem.Allocator,

    pub fn init(
        gpa: std.mem.Allocator,
        ctx: *const vk.Context,
        cfg: config_mod.Config,
        max_pos: u32,
    ) !MtpRuntimeState {
        const f = @sizeOf(f32);
        const hidden = cfg.hidden_size;
        const kv_dim = cfg.num_key_value_heads * cfg.head_dim;
        const n_layers = cfg.mtp_num_hidden_layers;

        var kv_k = try gpa.alloc(buffer.Buffer, n_layers);
        errdefer gpa.free(kv_k);
        var kv_v = try gpa.alloc(buffer.Buffer, n_layers);
        errdefer gpa.free(kv_v);

        var k_init: usize = 0;
        var v_init: usize = 0;
        errdefer {
            for (kv_k[0..k_init]) |*b| b.deinit(ctx.device);
            for (kv_v[0..v_init]) |*b| b.deinit(ctx.device);
        }
        for (kv_k) |*b| {
            b.* = try buffer.Buffer.initDeviceOnly(ctx, max_pos * kv_dim * f);
            try b.fillZero(ctx);
            k_init += 1;
        }
        for (kv_v) |*b| {
            b.* = try buffer.Buffer.initDeviceOnly(ctx, max_pos * kv_dim * f);
            try b.fillZero(ctx);
            v_init += 1;
        }

        var embed_tmp = try buffer.Buffer.initDeviceOnly(ctx, hidden * f);
        errdefer embed_tmp.deinit(ctx.device);
        var e_norm = try buffer.Buffer.initDeviceOnly(ctx, hidden * f);
        errdefer e_norm.deinit(ctx.device);
        var h_norm = try buffer.Buffer.initDeviceOnly(ctx, hidden * f);
        errdefer h_norm.deinit(ctx.device);
        var concat = try buffer.Buffer.initDeviceOnly(ctx, 2 * hidden * f);
        errdefer concat.deinit(ctx.device);
        var h_out = try buffer.Buffer.initDeviceOnly(ctx, hidden * f);
        errdefer h_out.deinit(ctx.device);

        return .{
            .kv_k = kv_k,
            .kv_v = kv_v,
            .embed_tmp = embed_tmp,
            .e_norm = e_norm,
            .h_norm = h_norm,
            .concat = concat,
            .h_out = h_out,
            .allocator = gpa,
        };
    }

    pub fn deinit(self: *MtpRuntimeState, device: vk.c.VkDevice) void {
        for (self.kv_k) |*b| b.deinit(device);
        for (self.kv_v) |*b| b.deinit(device);
        self.allocator.free(self.kv_k);
        self.allocator.free(self.kv_v);
        self.embed_tmp.deinit(device);
        self.e_norm.deinit(device);
        self.h_norm.deinit(device);
        self.concat.deinit(device);
        self.h_out.deinit(device);
    }
};

/// Record one MTP step's dispatches.
///
///   `h_prev_buf` — `[hidden]` fp32, caller-owned. Previous slot's
///                  hidden (or main forward's last hidden for slot 0).
///   `token_id`   — input token whose embedding kicks off the step.
///   `pos`        — sequence position (KV write row + RoPE position).
///   `logits_buf` — `[vocab_size]` fp32, caller-owned. Output target.
///
/// Mutates `mtp_state.h_out` to carry the pre-final-norm hidden into
/// the next slot. `sc.stream` is also clobbered; callers running both a
/// main forward and an MTP step must order them sequentially.
pub fn recordMtpStep(
    rec: *recorder.Recorder,
    sc: *const runtime_hybrid.Scratch,
    gm: *const gpu_model.GpuModel,
    cfg: config_mod.Config,
    k: *const runtime_hybrid.ChatKernels,
    mtp_state: *const MtpRuntimeState,
    h_prev_buf: *const buffer.Buffer,
    token_id: u32,
    pos: usize,
    max_pos: u32,
    logits_buf: *const buffer.Buffer,
) !void {
    const mtp = gm.mtp_head orelse return error.NoMtpHead;
    if (mtp_state.kv_k.len != mtp.layers.len) return error.MtpStateLayerCountMismatch;

    const hidden: u32 = @intCast(cfg.hidden_size);
    const inter: u32 = @intCast(cfg.intermediate_size);
    const vocab: u32 = @intCast(cfg.vocab_size);

    const p = runtime_hybrid.computeForwardPushes(cfg, pos, max_pos);

    // ── 1. embed lookup → mtp_state.embed_tmp ──────────────────────
    const embed_push = runtime_hybrid.EmbedLookupPush{
        .token_id = token_id,
        .dim = hidden,
        .scale = if (cfg.family.embedScalesByDim()) @sqrt(@as(f32, @floatFromInt(hidden))) else 1.0,
    };
    try runtime.recDispatch1D(rec, &k.embed, &.{ &gm.embed_tokens, &mtp_state.embed_tmp }, &embed_push, hidden);

    // ── 2. rmsnorm both inputs ─────────────────────────────────────
    try runtime.recDispatchPerRow(rec, &k.rmsnorm, &.{ &mtp_state.embed_tmp, &mtp.pre_fc_norm_embedding, &mtp_state.e_norm }, &p.rms_push, 1);
    try runtime.recDispatchPerRow(rec, &k.rmsnorm, &.{ h_prev_buf, &mtp.pre_fc_norm_hidden, &mtp_state.h_norm }, &p.rms_push, 1);

    // ── 3. concat: [e_norm; h_norm] → mtp_state.concat ─────────────
    const sc_e = runtime_hybrid.SliceCopyPush{ .src_off = 0, .dst_off = 0,      .n_elem = hidden };
    const sc_h = runtime_hybrid.SliceCopyPush{ .src_off = 0, .dst_off = hidden, .n_elem = hidden };
    try runtime.recDispatch1D(rec, &k.slice_copy, &.{ &mtp_state.e_norm, &mtp_state.concat }, &sc_e, hidden);
    try runtime.recDispatch1D(rec, &k.slice_copy, &.{ &mtp_state.h_norm, &mtp_state.concat }, &sc_h, hidden);

    // ── 4. fc projection: [hidden, 2*hidden] · [2*hidden] → sc.stream ─
    try runtime.recDispatchMatmul(rec, &k.matmul, &.{ &mtp_state.concat, &mtp.fc, &sc.stream }, 1, hidden, 2 * hidden);

    // ── 5. transformer block(s) ────────────────────────────────────
    // Mirrors the `.full_attention` branch of
    // `runtime_hybrid.recordOneLayer` exactly, with two substitutions:
    // `gm.layers[layer_idx]` → `mtp.layers[i]`,
    // `state.kv_{k,v}[layer_idx].?` → `mtp_state.kv_{k,v}[i]`.
    // No TQ4 V — the MTP block is too small to benefit and the chat
    // path's TQ4 hooks live on the main KV cache, not the MTP one.
    for (mtp.layers, 0..) |layer, i| {
        try runtime.recDispatchPerRow(rec, &k.rmsnorm, &.{ &sc.stream, &layer.input_layernorm, &sc.x_norm }, &p.rms_push, 1);

        try runtime.recDispatchMatmul(rec, &k.matmul, &.{ &sc.x_norm, &layer.q_proj.?, &sc.q_gate }, 1, p.q_proj_rows, hidden);
        try runtime.recDispatch1D(rec, &k.split_q_gate, &.{ &sc.q_gate, &sc.q, &sc.gate_attn }, &p.split_push, p.q_dim);
        try runtime.recDispatchMatmul(rec, &k.matmul, &.{ &sc.x_norm, &layer.k_proj.?, &sc.k }, 1, p.kv_dim, hidden);
        try runtime.recDispatchMatmul(rec, &k.matmul, &.{ &sc.x_norm, &layer.v_proj.?, &sc.v }, 1, p.kv_dim, hidden);

        try runtime.recDispatchPerRow(rec, &k.rmsnorm, &.{ &sc.q, &layer.q_norm.?, &sc.q }, &p.qkn_push, p.n_q_heads);
        try runtime.recDispatchPerRow(rec, &k.rmsnorm, &.{ &sc.k, &layer.k_norm.?, &sc.k }, &p.qkn_push, p.n_kv_heads);

        try runtime.recDispatch1D(rec, &k.rope_partial, &.{ &sc.q, &sc.qrot }, &p.rope_q_push, p.n_q_heads * p.head_dim);
        try runtime.recDispatch1D(rec, &k.rope_partial, &.{ &sc.k, &sc.krot }, &p.rope_k_push, p.n_kv_heads * p.head_dim);

        try runtime.recDispatch1D(rec, &k.kv_write, &.{ &sc.krot, &mtp_state.kv_k[i] }, &p.kv_write_push, p.kv_dim);
        try runtime.recDispatch1D(rec, &k.kv_write, &.{ &sc.v,    &mtp_state.kv_v[i] }, &p.kv_write_push, p.kv_dim);

        try rec.dispatch(&k.scores, &.{ &sc.qrot, &mtp_state.kv_k[i], &sc.scores }, &p.scores_push, p.n_q_heads * p.n_pos, 1, 1);
        try runtime.recDispatchPerRow(rec, &k.softmax, &.{ &sc.scores, &sc.scores }, &p.softmax_push, p.n_q_heads);
        try rec.dispatch(&k.attn_out, &.{ &sc.scores, &mtp_state.kv_v[i], &sc.head_out }, &p.attn_out_push, p.n_q_heads * p.head_dim, 1, 1);

        try runtime.recDispatch1D(rec, &k.sigmoid_mul, &.{ &sc.head_out, &sc.gate_attn, &sc.head_out_gated }, &p.sigmul_push, p.q_dim);
        try runtime.recDispatchMatmul(rec, &k.matmul, &.{ &sc.head_out_gated, &layer.o_proj.?, &sc.attn_out }, 1, hidden, p.q_dim);

        try runtime.recDispatch1D(rec, &k.add, &.{ &sc.stream, &sc.attn_out }, &p.add_push, hidden);

        try runtime.recDispatchPerRow(rec, &k.rmsnorm, &.{ &sc.stream, &layer.post_attention_layernorm, &sc.mid_norm }, &p.rms_push, 1);
        try runtime.recDispatchMatmul(rec, &k.matmul, &.{ &sc.mid_norm, &layer.gate_proj, &sc.gate }, 1, inter, hidden);
        try runtime.recDispatchMatmul(rec, &k.matmul, &.{ &sc.mid_norm, &layer.up_proj,   &sc.up },   1, inter, hidden);
        try runtime.recDispatch1D(rec, &k.swiglu, &.{ &sc.gate, &sc.up, &sc.fused }, &p.swiglu_push, inter);
        try runtime.recDispatchMatmul(rec, &k.matmul, &.{ &sc.fused, &layer.down_proj, &sc.ffn_out }, 1, hidden, inter);

        try runtime.recDispatch1D(rec, &k.add, &.{ &sc.stream, &sc.ffn_out }, &p.add_push, hidden);
    }

    // ── 6. h_out (pre-final-norm) for the next slot ────────────────
    const sc_h_out = runtime_hybrid.SliceCopyPush{ .src_off = 0, .dst_off = 0, .n_elem = hidden };
    try runtime.recDispatch1D(rec, &k.slice_copy, &.{ &sc.stream, &mtp_state.h_out }, &sc_h_out, hidden);

    // ── 7. final norm + (shared) lm_head ───────────────────────────
    try runtime.recDispatchPerRow(rec, &k.rmsnorm, &.{ &sc.stream, &mtp.norm, &sc.final_norm_out }, &p.rms_push, 1);
    try runtime.recDispatchMatmul(rec, &k.matmul_lm_head, &.{ &sc.final_norm_out, &gm.lm_head, logits_buf }, 1, vocab, hidden);
}
