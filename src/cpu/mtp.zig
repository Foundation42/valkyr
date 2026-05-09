//! CPU reference for the Multi-Token Prediction (MTP) head.
//!
//! Architecture (Qwen3.5/3.6, DeepSeek-V3 et al, with
//! `mtp_num_hidden_layers = 1` typical):
//!
//!   1. RMSNorm the input embedding (`mtp.pre_fc_norm_embedding`) and
//!      the previous-slot hidden state (`mtp.pre_fc_norm_hidden`).
//!   2. Concat channel-wise → `[2 * hidden]`.
//!   3. Project back to `hidden` via `mtp.fc.weight`.
//!   4. Run `mtp_num_hidden_layers` normal full-attention transformer
//!      blocks (q_norm/k_norm and attn_output_gate per the base config —
//!      same shape contract as the base model's full-attn layers).
//!   5. Final RMSNorm via `mtp.norm.weight`.
//!   6. Project to logits via the SHARED base `model.lm_head`.
//!
//! At inference the same module is reused recursively across draft slots
//! (slot k+1 reads slot k's hidden_out and previous predicted token).
//! This file ships the single-step kernel; chain logic lands in MTP-1c.

const std = @import("std");
const config_mod = @import("../config.zig");
const model_mod = @import("../model.zig");
const cpu_math = @import("math.zig");
const full_attn = @import("full_attn.zig");

const Config = config_mod.Config;

/// Per-call state for `forwardMtpStep`. Owns one KV cache per MTP
/// transformer block. Caller constructs once, reuses across slots /
/// positions; `init` zeroes the caches (fresh start). Cache rows are
/// indexed by the same `pos` the caller passes to `forwardMtpStep`.
pub const MtpState = struct {
    kv: []full_attn.KvCache,
    allocator: std.mem.Allocator,

    pub fn init(gpa: std.mem.Allocator, model: *const model_mod.Model, max_pos: usize) !MtpState {
        const cfg = model.config;
        const mtp = model.mtp_head orelse return error.NoMtpHead;
        const kv = try gpa.alloc(full_attn.KvCache, mtp.layers.len);
        errdefer gpa.free(kv);
        var n_init: usize = 0;
        errdefer for (kv[0..n_init]) |*c| c.deinit(gpa);
        for (kv) |*c| {
            c.* = try full_attn.KvCache.init(gpa, cfg, max_pos);
            n_init += 1;
        }
        return .{ .kv = kv, .allocator = gpa };
    }

    pub fn deinit(self: *MtpState) void {
        for (self.kv) |*c| c.deinit(self.allocator);
        self.allocator.free(self.kv);
    }
};

/// One step through the MTP head. Single-position decode at `pos`.
///
///   - `token_id` — the input token whose embedding is fed in (for slot
///     0 this is the actual next token from the main path; for slot k≥1
///     it's the previous slot's predicted token).
///   - `h_prev` — `[hidden]` previous-slot hidden state (for slot 0 this
///     is the main model's last hidden state at position `pos`).
///   - `pos` — position in the sequence; the MTP layer reads/writes its
///     KV cache at row `pos` and applies RoPE at that position.
///   - `h_out` — `[hidden]` written; used as `h_prev` for slot k+1
///     (NB: pre-norm — before `mtp.norm`, matching DeepSeek-V3 §2.3).
///   - `logits` — `[vocab_size]` written; next-token distribution for
///     the slot AFTER this one.
pub fn forwardMtpStep(
    model: *const model_mod.Model,
    state: *MtpState,
    token_id: u32,
    h_prev: []const f32,
    pos: usize,
    scratch: std.mem.Allocator,
    h_out: []f32,
    logits: []f32,
) !void {
    const cfg = model.config;
    const mtp = model.mtp_head orelse return error.NoMtpHead;
    if (h_prev.len != cfg.hidden_size) return error.LengthMismatch;
    if (h_out.len != cfg.hidden_size) return error.LengthMismatch;
    if (logits.len != cfg.vocab_size) return error.LogitsSizeMismatch;
    if (token_id >= cfg.vocab_size) return error.TokenOutOfRange;
    if (state.kv.len != mtp.layers.len) return error.MtpStateLayerCountMismatch;

    const hidden = cfg.hidden_size;
    const inter = cfg.intermediate_size;

    // ── 1+2+3: norms + concat + fc projection ───────────────────────
    const embed = try scratch.alloc(f32, hidden);
    try cpu_math.embedRowAsF32(embed, model.embed_tokens, token_id);
    if (cfg.family.embedScalesByDim()) {
        const s: f32 = @sqrt(@as(f32, @floatFromInt(hidden)));
        for (embed) |*xi| xi.* *= s;
    }

    const e_norm = try scratch.alloc(f32, hidden);
    const h_norm = try scratch.alloc(f32, hidden);
    try cpu_math.rmsnorm(e_norm, embed, mtp.pre_fc_norm_embedding, cfg.rms_norm_eps, cfg.family);
    try cpu_math.rmsnorm(h_norm, h_prev, mtp.pre_fc_norm_hidden, cfg.rms_norm_eps, cfg.family);

    // Channel-wise concat: [e_norm; h_norm] → [2*hidden]. Order matches
    // Qwen3 reference: embedding first, hidden second.
    const concat = try scratch.alloc(f32, 2 * hidden);
    @memcpy(concat[0..hidden], e_norm);
    @memcpy(concat[hidden .. 2 * hidden], h_norm);

    // mtp.fc shape is [hidden, 2*hidden]; matmul_nt computes
    // `stream = concat · fc^T` so the output is [hidden].
    const stream = try scratch.alloc(f32, hidden);
    try cpu_math.matmul_nt(stream, concat, mtp.fc, 1, hidden, 2 * hidden);

    // ── 4: transformer block(s) ─────────────────────────────────────
    // Each MTP layer is a normal full-attention block; full_attn.decodeStep
    // already handles attn_output_gate (q_proj 2× width) and partial RoPE
    // via cfg, so we just call it.
    const x_norm = try scratch.alloc(f32, hidden);
    const attn_out = try scratch.alloc(f32, hidden);
    const mid_norm = try scratch.alloc(f32, hidden);
    const gate = try scratch.alloc(f32, inter);
    const up = try scratch.alloc(f32, inter);
    const fused = try scratch.alloc(f32, inter);
    const ffn_out = try scratch.alloc(f32, hidden);

    for (mtp.layers, 0..) |layer, i| {
        try cpu_math.rmsnorm(x_norm, stream, layer.input_layernorm, cfg.rms_norm_eps, cfg.family);
        try full_attn.decodeStep(scratch, cfg, layer, &state.kv[i], x_norm, attn_out, pos);
        for (stream, attn_out) |*si, ai| si.* += ai;

        try cpu_math.rmsnorm(mid_norm, stream, layer.post_attention_layernorm, cfg.rms_norm_eps, cfg.family);
        try cpu_math.matmul_nt(gate, mid_norm, layer.gate_proj, 1, inter, hidden);
        try cpu_math.matmul_nt(up, mid_norm, layer.up_proj, 1, inter, hidden);
        try cpu_math.gatedFfn(fused, gate, up, cfg.family);
        try cpu_math.matmul_nt(ffn_out, fused, layer.down_proj, 1, hidden, inter);
        for (stream, ffn_out) |*si, fi| si.* += fi;
    }

    // h_out is the pre-norm hidden — that's what slot k+1 wants as its
    // `h_prev` (it'll re-norm via pre_fc_norm_hidden on the next step).
    @memcpy(h_out, stream);

    // ── 5+6: final norm + shared lm_head ────────────────────────────
    const final = try scratch.alloc(f32, hidden);
    try cpu_math.rmsnorm(final, stream, mtp.norm, cfg.rms_norm_eps, cfg.family);
    try cpu_math.matmul_nt(logits, final, model.lm_head, 1, cfg.vocab_size, hidden);
}
