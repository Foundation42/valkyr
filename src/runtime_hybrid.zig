//! Library-side forward-step + sampling primitives for hybrid models
//! (Qwen3.5 family — Gated DeltaNet linear-attention layers
//! interleaved with full-attention layers, plus an attention-output
//! gate and partial RoPE).
//!
//! Mirror of `runtime.zig`'s role for dense models. The split exists
//! because the hybrid path needs:
//!   - 19 distinct kernels vs the dense path's 11 (gated_delta_step,
//!     conv1d_update, rmsnorm_gated, l2norm_per_head, split_q_gate,
//!     sigmoid_mul, slice_copy, scale on top of the dense set);
//!   - per-layer SSM state (conv + recurrent) that is reset between
//!     prompts, alongside the standard KV cache;
//!   - a per-layer dispatch that branches on `layer.layer_type`
//!     (linear vs full attention).
//!
//! Generation output is bit-identical to the CLI hybrid path. Hosts
//! that just want text-out should use `Session` (chunk D will teach
//! it to pick this module when `cfg.family.isHybrid()` is true). For
//! bespoke orchestration, callers can drive `recordOneLayer`,
//! `recordForwardStep`, etc. directly.

const std = @import("std");
const vk = @import("gpu/vk.zig");
const buffer = @import("gpu/buffer.zig");
const pipeline = @import("gpu/pipeline.zig");
const recorder = @import("gpu/recorder.zig");
const gpu_model = @import("gpu/model.zig");
const config_mod = @import("config.zig");
const shaders = @import("shaders");
const runtime = @import("runtime.zig");

// ── Push structs ──────────────────────────────────────────────────
//
// Some structs are reused from `runtime` (RmsnormPush, RopePartialPush,
// AttnScoresPush, SoftmaxPush, AttnOutputPush, KvWritePush, MatmulPush,
// AddInPlacePush, GegluPush, EmbedLookupPush, Tq4PackPush) — pulled
// in via re-exports below. The structs unique to hybrid are declared
// fresh here so callers can reference everything off one module.

pub const RmsnormPush = runtime.RmsnormPush;
pub const RopePartialPush = runtime.RopePartialPush;
pub const KvWritePush = runtime.KvWritePush;
pub const AttnScoresPush = runtime.AttnScoresPush;
pub const SoftmaxPush = runtime.SoftmaxPush;
pub const AttnOutputPush = runtime.AttnOutputPush;
pub const AddInPlacePush = runtime.AddInPlacePush;
pub const GegluPush = runtime.GegluPush;
pub const EmbedLookupPush = runtime.EmbedLookupPush;
pub const MatmulPush = runtime.MatmulPush;
pub const Tq4PackPush = runtime.Tq4PackPush;

pub const SplitQGatePush = extern struct { num_heads: u32, head_dim: u32 };
pub const SigmoidMulPush = extern struct { n_elem: u32 };
pub const L2normPush = extern struct { head_dim: u32, eps: f32 };
pub const Conv1dUpdatePush = extern struct { conv_dim: u32, kernel_size: u32 };
pub const RmsnormGatedPush = extern struct { head_dim: u32, eps: f32 };
pub const GatedDeltaStepPush = extern struct {
    num_k_heads: u32,
    num_v_heads: u32,
    head_k: u32,
    head_v: u32,
};
pub const ScalePush = extern struct { n: u32, scale: f32 };
pub const SliceCopyPush = extern struct { src_off: u32, dst_off: u32, n_elem: u32 };

// ── ChatKernels: pre-built compute pipelines ──────────────────────

/// Bundle of compiled compute pipelines a hybrid forward pass needs.
/// Built once per model + reused across every recorded layer.
pub const ChatKernels = struct {
    embed: pipeline.Kernel,
    rmsnorm: pipeline.Kernel,
    matmul: pipeline.Kernel,
    matmul_lm_head: pipeline.Kernel,
    add: pipeline.Kernel,
    swiglu: pipeline.Kernel,
    rope_partial: pipeline.Kernel,
    split_q_gate: pipeline.Kernel,
    sigmoid_mul: pipeline.Kernel,
    l2norm_per_head: pipeline.Kernel,
    conv1d_update: pipeline.Kernel,
    rmsnorm_gated: pipeline.Kernel,
    gated_delta_step: pipeline.Kernel,
    kv_write: pipeline.Kernel,
    scores: pipeline.Kernel,
    softmax: pipeline.Kernel,
    attn_out: pipeline.Kernel,
    /// FlashDecoding phase 1 — split-K decode kernel for full-attention
    /// layers. Replaces the `scores → softmax → attn_out` trio when
    /// `cfg.head_dim ≤ FA_HEAD_DIM_MAX`. SPIR-V variant picked by
    /// `runtime.faDecodeSplitSpv(head_dim)` at init.
    fa_decode_split: pipeline.Kernel,
    /// FlashDecoding phase 2 — merge per-split (O, m, l) partials.
    /// Paired with `fa_decode_split`.
    fa_decode_merge: pipeline.Kernel,
    /// Fused FlashDecoding + TQ4-V dequant (T-arc, 2026-05-09). Mirrors
    /// the dense `runtime.ChatKernels.fa_decode_split_tq4v`. Used when
    /// `cfg.head_dim == 256` and the host provides `Tq4VHooks` —
    /// reads the packed V cache directly and dequants inline.
    fa_decode_split_tq4v: pipeline.Kernel,
    /// Fused FlashAttention forward (prefill). Used by the n_q>1
    /// batched-prefill path in `recordFullAttnLayerBatched`. Same
    /// d=128/d=256 SPIR-V variant selector as the dense path
    /// (`runtime.faForwardSpv`).
    fa_forward: pipeline.Kernel,
    /// Batched partial RoPE — covers n_q query rows in one dispatch
    /// with absolute positions `[pos_offset .. pos_offset + n_q)`.
    /// Used by the prefill path; n_q=1 decode keeps the single-position
    /// `rope_partial`.
    rope_partial_batched: pipeline.Kernel,
    slice_copy: pipeline.Kernel,
    scale: pipeline.Kernel,

    pub fn init(ctx: *const vk.Context, precision: gpu_model.Precision, head_dim: u32) !ChatKernels {
        // Mirrors the dense `ChatKernels.init`: precision-aware matmul
        // for the per-layer projections, separate selector for the LM
        // head — bf16 when any non-fp32 path is active. lm_head + embed
        // never go Q4_0/Q4_K (argmax sensitivity).
        const matmul_spv: []align(4) const u8 = switch (precision) {
            .fp32_all => &shaders.matmul_nt_v2,
            .bf16_matmul => &shaders.matmul_nt_v2_bf16,
            .q4_0_matmul => &shaders.matmul_nt_v2_q4_0,
            .q4_k_matmul => &shaders.matmul_nt_v2_q4_k,
        };
        const lm_head_spv: []align(4) const u8 = switch (precision) {
            .fp32_all => &shaders.matmul_nt_v2,
            .bf16_matmul, .q4_0_matmul, .q4_k_matmul => &shaders.matmul_nt_v2_bf16,
        };
        const embed_spv: []align(4) const u8 = switch (precision) {
            .fp32_all => &shaders.embed_lookup,
            .bf16_matmul, .q4_0_matmul, .q4_k_matmul => &shaders.embed_lookup_bf16,
        };
        return .{
            .embed = try pipeline.Kernel.init(ctx, embed_spv, 2, @sizeOf(EmbedLookupPush)),
            .rmsnorm = try pipeline.Kernel.init(ctx, &shaders.rmsnorm, 3, @sizeOf(RmsnormPush)),
            .matmul = try pipeline.Kernel.init(ctx, matmul_spv, 3, @sizeOf(MatmulPush)),
            .matmul_lm_head = try pipeline.Kernel.init(ctx, lm_head_spv, 3, @sizeOf(MatmulPush)),
            .add = try pipeline.Kernel.init(ctx, &shaders.add_in_place, 2, @sizeOf(AddInPlacePush)),
            .swiglu = try pipeline.Kernel.init(ctx, &shaders.swiglu, 3, @sizeOf(GegluPush)),
            .rope_partial = try pipeline.Kernel.init(ctx, &shaders.rope_partial, 2, @sizeOf(RopePartialPush)),
            .split_q_gate = try pipeline.Kernel.init(ctx, &shaders.split_q_gate, 3, @sizeOf(SplitQGatePush)),
            .sigmoid_mul = try pipeline.Kernel.init(ctx, &shaders.sigmoid_mul, 3, @sizeOf(SigmoidMulPush)),
            .l2norm_per_head = try pipeline.Kernel.init(ctx, &shaders.l2norm_per_head, 2, @sizeOf(L2normPush)),
            .conv1d_update = try pipeline.Kernel.init(ctx, &shaders.conv1d_update, 4, @sizeOf(Conv1dUpdatePush)),
            .rmsnorm_gated = try pipeline.Kernel.init(ctx, &shaders.rmsnorm_gated, 4, @sizeOf(RmsnormGatedPush)),
            .gated_delta_step = try pipeline.Kernel.init(ctx, &shaders.gated_delta_step, 9, @sizeOf(GatedDeltaStepPush)),
            .kv_write = try pipeline.Kernel.init(ctx, &shaders.kv_write, 2, @sizeOf(KvWritePush)),
            .scores = try pipeline.Kernel.init(ctx, &shaders.attn_scores, 3, @sizeOf(AttnScoresPush)),
            .softmax = try pipeline.Kernel.init(ctx, &shaders.softmax, 2, @sizeOf(SoftmaxPush)),
            .attn_out = try pipeline.Kernel.init(ctx, &shaders.attn_output, 3, @sizeOf(AttnOutputPush)),
            .fa_decode_split = try pipeline.Kernel.init(ctx, runtime.faDecodeSplitSpv(head_dim), 6, @sizeOf(runtime.FaDecodeSplitPush)),
            .fa_decode_merge = try pipeline.Kernel.init(ctx, &shaders.fa_decode_merge, 4, @sizeOf(runtime.FaDecodeMergePush)),
            .fa_decode_split_tq4v = try pipeline.Kernel.init(ctx, &shaders.fa_decode_split_tq4v, 6, @sizeOf(runtime.FaDecodeSplitPush)),
            .fa_forward = try pipeline.Kernel.init(ctx, runtime.faForwardSpv(head_dim), 5, @sizeOf(runtime.FaForwardPush)),
            .rope_partial_batched = try pipeline.Kernel.init(ctx, &shaders.rope_partial_batched, 2, @sizeOf(runtime.RopeBatchedPush)),
            .slice_copy = try pipeline.Kernel.init(ctx, &shaders.slice_copy, 2, @sizeOf(SliceCopyPush)),
            .scale = try pipeline.Kernel.init(ctx, &shaders.scale, 2, @sizeOf(ScalePush)),
        };
    }

    pub fn deinit(self: *ChatKernels) void {
        inline for (.{
            "embed",            "rmsnorm",        "matmul",           "matmul_lm_head", "add",
            "swiglu",           "rope_partial",   "split_q_gate",     "sigmoid_mul",    "l2norm_per_head",
            "conv1d_update",    "rmsnorm_gated",  "gated_delta_step", "kv_write",       "scores",
            "softmax",          "attn_out",       "fa_decode_split",  "fa_decode_merge",
            "fa_decode_split_tq4v",
            "fa_forward",       "rope_partial_batched",
            "slice_copy",       "scale",
        }) |fname| {
            @field(self, fname).deinit();
        }
    }
};

// ── Per-pass scratch buffers ──────────────────────────────────────

/// Per-step intermediate buffers for the hybrid forward pass. Sized
/// once at init from `cfg + max_pos`; reused across every layer of
/// every step in the chat session. The `dequant_v` slot is non-null
/// only when the host will pass `Tq4VHooks` (asymmetric K=fp/V=TQ4).
pub const Scratch = struct {
    stream: buffer.Buffer,
    x_norm: buffer.Buffer,
    mid_norm: buffer.Buffer,
    attn_out: buffer.Buffer,
    gate: buffer.Buffer,
    up: buffer.Buffer,
    fused: buffer.Buffer,
    ffn_out: buffer.Buffer,
    final_norm_out: buffer.Buffer,
    logits: buffer.Buffer,
    // Linear-attn scratch
    /// Single-row x_norm feeder for the linear-attn batched-prefill
    /// path. Linear attention is sequential per row (each step mutates
    /// the SSM state), so β-4 dispatches `gated_delta_step` n_q times;
    /// each iteration `slice_copy`s row t of `x_norm` into here, runs
    /// the existing single-row chain, then `slice_copy`s the row's
    /// output back. Sized at the n_q=1 footprint regardless of
    /// `max_n_q` — only one row is live at a time.
    x_norm_lin: buffer.Buffer,
    /// Single-row attn_out catcher for the linear-attn batched-prefill
    /// path. Mirrors `x_norm_lin` on the output side.
    attn_out_lin: buffer.Buffer,
    mixed_qkv: buffer.Buffer,
    mixed_qkv_post: buffer.Buffer,
    z: buffer.Buffer,
    b_raw: buffer.Buffer,
    a_raw: buffer.Buffer,
    q_lin: buffer.Buffer,
    k_lin: buffer.Buffer,
    v_lin: buffer.Buffer,
    q_lin_n: buffer.Buffer,
    k_lin_n: buffer.Buffer,
    y: buffer.Buffer,
    post_norm: buffer.Buffer,
    // Full-attn scratch
    q_gate: buffer.Buffer,
    q: buffer.Buffer,
    gate_attn: buffer.Buffer,
    k: buffer.Buffer,
    v: buffer.Buffer,
    qrot: buffer.Buffer,
    krot: buffer.Buffer,
    head_out: buffer.Buffer,
    head_out_gated: buffer.Buffer,
    scores: buffer.Buffer,
    /// FlashDecoding phase-1 partials. Sized for the worst-case
    /// `n_q_heads × max_n_splits × head_dim` (and `n_q_heads ×
    /// max_n_splits` for the m/l pair) `chooseFaDecodeSplit(max_pos)`
    /// can produce. Only used on the FA path; the 3-pass fallback
    /// leaves them untouched.
    fa_o_partial: buffer.Buffer,
    fa_m_partial: buffer.Buffer,
    fa_l_partial: buffer.Buffer,
    /// LSE side-output for `fa_forward` (prefill path). Inference-time
    /// callers pass `write_lse = 0` and the kernel never writes here,
    /// but the binding still has to resolve — Vulkan's pipeline layout
    /// matches the shader's declared `set=0, binding=4`. Sized for
    /// `[max_n_q, n_q_heads]`.
    fa_lse: buffer.Buffer,
    /// Shared TQ4-V dequant scratch — sized for one full V history at
    /// `max_pos`, reused across all full-attn layers. Only allocated
    /// when the host is using TQ4 V-cache; otherwise `null`.
    dequant_v: ?buffer.Buffer,

    /// `max_n_q` is the maximum number of query rows (positions) any
    /// single forward step records into this scratch. The decode path
    /// sets it to 1; batched-prefill / MTP-verify paths set it to the
    /// largest n_q they intend to record (e.g. prompt length, or k+1
    /// for a k-draft MTP verify). Buffers downstream of attention scale
    /// linearly with this dim; linear-attn intermediates do not (β-4
    /// dispatches gated_delta sequentially per row).
    pub fn init(ctx: *const vk.Context, cfg: config_mod.Config, max_pos: u32, max_n_q: u32, tq4v: bool) !Scratch {
        if (max_n_q == 0) return error.MaxNqMustBePositive;
        const f = @sizeOf(f32);
        const hidden = cfg.hidden_size;
        const inter = cfg.intermediate_size;
        const head_dim = cfg.head_dim;
        const n_q = cfg.num_attention_heads;
        const n_kv = cfg.num_key_value_heads;
        const q_dim = n_q * head_dim;
        const kv_dim = n_kv * head_dim;
        const q_proj_rows = if (cfg.attn_output_gate) 2 * q_dim else q_dim;
        const conv_dim = cfg.linearAttnConvDim();
        const value_dim = cfg.linear_num_value_heads * cfg.linear_value_head_dim;
        const key_dim = cfg.linear_num_key_heads * cfg.linear_key_head_dim;
        const max_n_q_us: usize = max_n_q;

        // Worst-case FlashDecoding split count — mirrors the heuristic
        // in `runtime.chooseFaDecodeSplit` applied to `max_pos`. Inlined
        // (rather than calling through) to keep this scratch allocator
        // independent of the per-step push-compute path.
        const max_n_splits: u32 = if (max_pos <= 4)
            1
        else if (max_pos < 1024)
            4
        else
            (max_pos + 255) / 256;
        const o_partial_elems: usize = n_q * @as(usize, max_n_splits) * head_dim;
        const ml_partial_elems: usize = n_q * @as(usize, max_n_splits);
        return .{
            .stream          = try buffer.Buffer.initDeviceOnly(ctx, max_n_q_us * hidden * f),
            .x_norm          = try buffer.Buffer.initDeviceOnly(ctx, max_n_q_us * hidden * f),
            .mid_norm        = try buffer.Buffer.initDeviceOnly(ctx, max_n_q_us * hidden * f),
            .attn_out        = try buffer.Buffer.initDeviceOnly(ctx, max_n_q_us * hidden * f),
            .gate            = try buffer.Buffer.initDeviceOnly(ctx, max_n_q_us * inter * f),
            .up              = try buffer.Buffer.initDeviceOnly(ctx, max_n_q_us * inter * f),
            .fused           = try buffer.Buffer.initDeviceOnly(ctx, max_n_q_us * inter * f),
            .ffn_out         = try buffer.Buffer.initDeviceOnly(ctx, max_n_q_us * hidden * f),
            .final_norm_out  = try buffer.Buffer.initDeviceOnly(ctx, max_n_q_us * hidden * f),
            .logits          = try buffer.Buffer.initDeviceOnly(ctx, max_n_q_us * cfg.vocab_size * f),
            // Linear-attn intermediates: kept at n_q=1 footprint. β-4
            // walks `gated_delta_step` sequentially per row, so we only
            // ever hold one row of linear-attn state at a time. The
            // two row-feeders bridge the multi-row x_norm / attn_out
            // buffers to the single-row chain.
            .x_norm_lin      = try buffer.Buffer.initDeviceOnly(ctx, hidden * f),
            .attn_out_lin    = try buffer.Buffer.initDeviceOnly(ctx, hidden * f),
            .mixed_qkv       = try buffer.Buffer.initDeviceOnly(ctx, conv_dim * f),
            .mixed_qkv_post  = try buffer.Buffer.initDeviceOnly(ctx, conv_dim * f),
            .z               = try buffer.Buffer.initDeviceOnly(ctx, value_dim * f),
            .b_raw           = try buffer.Buffer.initDeviceOnly(ctx, cfg.linear_num_value_heads * f),
            .a_raw           = try buffer.Buffer.initDeviceOnly(ctx, cfg.linear_num_value_heads * f),
            .q_lin           = try buffer.Buffer.initDeviceOnly(ctx, key_dim * f),
            .k_lin           = try buffer.Buffer.initDeviceOnly(ctx, key_dim * f),
            .v_lin           = try buffer.Buffer.initDeviceOnly(ctx, value_dim * f),
            .q_lin_n         = try buffer.Buffer.initDeviceOnly(ctx, key_dim * f),
            .k_lin_n         = try buffer.Buffer.initDeviceOnly(ctx, key_dim * f),
            .y               = try buffer.Buffer.initDeviceOnly(ctx, value_dim * f),
            .post_norm       = try buffer.Buffer.initDeviceOnly(ctx, value_dim * f),
            // Full-attn intermediates: scaled to `max_n_q` rows so the
            // batched-prefill path can carry n_q queries through one
            // dispatch chain.
            .q_gate          = try buffer.Buffer.initDeviceOnly(ctx, max_n_q_us * q_proj_rows * f),
            .q               = try buffer.Buffer.initDeviceOnly(ctx, max_n_q_us * q_dim * f),
            .gate_attn       = try buffer.Buffer.initDeviceOnly(ctx, max_n_q_us * q_dim * f),
            .k               = try buffer.Buffer.initDeviceOnly(ctx, max_n_q_us * kv_dim * f),
            .v               = try buffer.Buffer.initDeviceOnly(ctx, max_n_q_us * kv_dim * f),
            .qrot            = try buffer.Buffer.initDeviceOnly(ctx, max_n_q_us * q_dim * f),
            .krot            = try buffer.Buffer.initDeviceOnly(ctx, max_n_q_us * kv_dim * f),
            .head_out        = try buffer.Buffer.initDeviceOnly(ctx, max_n_q_us * q_dim * f),
            .head_out_gated  = try buffer.Buffer.initDeviceOnly(ctx, max_n_q_us * q_dim * f),
            // 3-pass scores: `[max_n_q, n_q_heads, max_pos]`. FA path
            // doesn't read this; only the `head_dim > FA_HEAD_DIM_MAX`
            // fallback would.
            .scores          = try buffer.Buffer.initDeviceOnly(ctx, max_n_q_us * n_q * max_pos * f),
            // FA partials: `[max_n_q, n_q_heads, max_n_splits, head_dim]`.
            .fa_o_partial    = try buffer.Buffer.initDeviceOnly(ctx, max_n_q_us * o_partial_elems * f),
            .fa_m_partial    = try buffer.Buffer.initDeviceOnly(ctx, max_n_q_us * ml_partial_elems * f),
            .fa_l_partial    = try buffer.Buffer.initDeviceOnly(ctx, max_n_q_us * ml_partial_elems * f),
            .fa_lse          = try buffer.Buffer.initDeviceOnly(ctx, max_n_q_us * n_q * f),
            // Shared V dequant scratch — V history depth is `max_pos`
            // independent of how many queries read it. No max_n_q scale.
            .dequant_v       = if (tq4v) try buffer.Buffer.initDeviceOnly(ctx, max_pos * kv_dim * f) else null,
        };
    }

    pub fn deinit(self: *Scratch, device: vk.c.VkDevice) void {
        inline for (.{
            "stream", "x_norm", "mid_norm", "attn_out",
            "gate", "up", "fused", "ffn_out",
            "final_norm_out", "logits",
            "x_norm_lin", "attn_out_lin",
            "mixed_qkv", "mixed_qkv_post", "z", "b_raw", "a_raw",
            "q_lin", "k_lin", "v_lin", "q_lin_n", "k_lin_n",
            "y", "post_norm",
            "q_gate", "q", "gate_attn", "k", "v", "qrot", "krot",
            "head_out", "head_out_gated", "scores",
            "fa_o_partial", "fa_m_partial", "fa_l_partial", "fa_lse",
        }) |fname| {
            @field(self, fname).deinit(device);
        }
        if (self.dequant_v) |*b| b.deinit(device);
    }
};

// ── Per-layer SSM + KV state ──────────────────────────────────────

/// Per-layer Gated DeltaNet state (conv + recurrent) for linear-attn
/// layers, plus per-layer KV cache for full-attn layers. Lives the
/// whole chat session — only `reset` (zeroes the SSM state) is
/// per-prompt; KV is overwritten in place by prefill.
pub const State = struct {
    /// `[num_layers]` per-layer Gated DeltaNet conv state. Slots for
    /// full-attention layers are `null`.
    ssm_conv: []?buffer.Buffer,
    /// `[num_layers]` per-layer Gated DeltaNet recurrent state.
    ssm_rec: []?buffer.Buffer,
    /// `[num_layers]` per-layer KV-cache K. Slots for linear layers
    /// are `null`. Sized for `max_pos` positions.
    kv_k: []?buffer.Buffer,
    /// `[num_layers]` per-layer KV-cache V (fp32 path). When TQ4-V
    /// is enabled this stays `null` for every layer and `kv_v_tq4`
    /// is used instead.
    kv_v: []?buffer.Buffer,
    /// `[num_layers]` per-layer TQ4-packed V cache. Allocated only
    /// for `.full_attention` slots when TQ4-V is enabled; `null`
    /// otherwise.
    kv_v_tq4: []?buffer.Buffer,
    allocator: std.mem.Allocator,

    pub fn init(gpa: std.mem.Allocator, ctx: *const vk.Context, cfg: config_mod.Config, max_pos: u32, tq4v: bool) !State {
        const f = @sizeOf(f32);
        const u = @sizeOf(u32);
        const conv_dim = cfg.linearAttnConvDim();
        const conv_kernel = cfg.linear_conv_kernel_dim;
        const head_k = cfg.linear_key_head_dim;
        const head_v = cfg.linear_value_head_dim;
        const n_v_heads = cfg.linear_num_value_heads;
        const kv_dim = cfg.num_key_value_heads * cfg.head_dim;
        // TQ4 sizing — same convention as `GpuKvCacheTq4`: block_size
        // = head_dim (must be 128 or 256 to hit the existing shader
        // pair), n_blocks_per_pos = num_kv_heads, each block is one γ
        // word plus head_dim/8 packed-index words.
        const block_size = cfg.head_dim;
        const n_blocks_per_pos = cfg.num_key_value_heads;
        const u32s_per_block = 1 + block_size / 8;

        var ssm_conv = try gpa.alloc(?buffer.Buffer, cfg.num_hidden_layers);
        @memset(ssm_conv, null);
        var ssm_rec = try gpa.alloc(?buffer.Buffer, cfg.num_hidden_layers);
        @memset(ssm_rec, null);
        var kv_k = try gpa.alloc(?buffer.Buffer, cfg.num_hidden_layers);
        @memset(kv_k, null);
        var kv_v = try gpa.alloc(?buffer.Buffer, cfg.num_hidden_layers);
        @memset(kv_v, null);
        var kv_v_tq4 = try gpa.alloc(?buffer.Buffer, cfg.num_hidden_layers);
        @memset(kv_v_tq4, null);

        for (cfg.layer_types[0..cfg.num_hidden_layers], 0..) |lt, i| switch (lt) {
            .linear_attention => {
                ssm_conv[i] = try buffer.Buffer.initDeviceOnly(ctx, conv_dim * conv_kernel * f);
                ssm_rec[i] = try buffer.Buffer.initDeviceOnly(ctx, n_v_heads * head_k * head_v * f);
            },
            .full_attention => {
                kv_k[i] = try buffer.Buffer.initDeviceOnly(ctx, max_pos * kv_dim * f);
                if (tq4v) {
                    kv_v_tq4[i] = try buffer.Buffer.initDeviceOnly(ctx, max_pos * n_blocks_per_pos * u32s_per_block * u);
                } else {
                    kv_v[i] = try buffer.Buffer.initDeviceOnly(ctx, max_pos * kv_dim * f);
                }
            },
        };
        return .{
            .ssm_conv = ssm_conv,
            .ssm_rec = ssm_rec,
            .kv_k = kv_k,
            .kv_v = kv_v,
            .kv_v_tq4 = kv_v_tq4,
            .allocator = gpa,
        };
    }

    pub fn deinit(self: *State, device: vk.c.VkDevice) void {
        for (self.ssm_conv) |*b| if (b.*) |*v| v.deinit(device);
        for (self.ssm_rec) |*b| if (b.*) |*v| v.deinit(device);
        for (self.kv_k) |*b| if (b.*) |*v| v.deinit(device);
        for (self.kv_v) |*b| if (b.*) |*v| v.deinit(device);
        for (self.kv_v_tq4) |*b| if (b.*) |*v| v.deinit(device);
        self.allocator.free(self.ssm_conv);
        self.allocator.free(self.ssm_rec);
        self.allocator.free(self.kv_k);
        self.allocator.free(self.kv_v);
        self.allocator.free(self.kv_v_tq4);
    }

    /// Zero the per-layer SSM state (conv + recurrent) so a fresh
    /// prompt can run from a clean Gated DeltaNet state. KV slots are
    /// left untouched: they are unconditionally overwritten when the
    /// new prompt's prefill writes to positions [0, n_pos), and only
    /// positions in that range are read by attention. Used by the
    /// `--prompts` batch path where a single model load runs many
    /// independent prompts.
    pub fn reset(self: *State, ctx: *const vk.Context) !void {
        for (self.ssm_conv) |*b| if (b.*) |*v| try v.fillZero(ctx);
        for (self.ssm_rec) |*b| if (b.*) |*v| try v.fillZero(ctx);
    }
};

/// TQ4-V hooks for the hybrid forward step. Differs from the dense
/// `runtime.Tq4VHooks` because the hybrid `State` already owns its
/// per-layer V buffers (no separate `GpuKvCacheTq4`); only the pack
/// + unpack kernels and the per-pos block count are needed.
pub const Tq4VHooks = struct {
    pack: *const pipeline.Kernel,
    unpack: *const pipeline.Kernel,
    n_blocks_per_pos: u32,
};

// ── Per-step push computation ─────────────────────────────────────

pub const ForwardPushes = struct {
    rms_push: RmsnormPush,
    qkn_push: RmsnormPush,
    add_push: AddInPlacePush,
    swiglu_push: GegluPush,
    conv1d_push: Conv1dUpdatePush,
    l2_push: L2normPush,
    rms_gated_push: RmsnormGatedPush,
    gds_push: GatedDeltaStepPush,
    split_push: SplitQGatePush,
    sigmul_push: SigmoidMulPush,
    rope_q_push: RopePartialPush,
    rope_k_push: RopePartialPush,
    kv_write_push: KvWritePush,
    scores_push: AttnScoresPush,
    softmax_push: SoftmaxPush,
    attn_out_push: AttnOutputPush,
    /// Whether the full-attention layer takes the FA path. Set when
    /// `head_dim ≤ runtime.FA_HEAD_DIM_MAX`; otherwise the
    /// `recordOneLayer` falls through to the 3-pass chain. Decided once
    /// per session at config-load time (head_dim is constant per model).
    attn_use_fa: bool,
    fa_decode_split_push: runtime.FaDecodeSplitPush,
    fa_decode_merge_push: runtime.FaDecodeMergePush,
    n_pos: u32,
    q_scale: f32,
    key_dim: u32,
    value_dim: u32,
    n_v_heads: u32,
    n_k_heads_lin: u32,
    n_q_heads: u32,
    n_kv_heads: u32,
    head_dim: u32,
    q_dim: u32,
    q_proj_rows: u32,
    kv_dim: u32,
    conv_dim: u32,
    /// Starting position of the first query in this batch. The i-th
    /// query occupies absolute position `pos_start + i`. For the
    /// single-position decode path this matches the existing `pos`
    /// argument and `n_q == 1`. β-3+ uses it to drive per-row RoPE +
    /// KV-write at dispatch time.
    pos_start: u32,
    /// Number of query rows recorded in this forward step. Decode
    /// path = 1; batched-prefill / MTP-verify > 1. Per-row pushes
    /// (`rope_q_push`, `rope_k_push`, `kv_write_push`) and 3-pass
    /// attention pushes carry the row-zero values; `recordOneLayer`
    /// recomputes per-row variants when `n_q > 1`.
    n_q: u32,
    /// Batched-RoPE pushes for the n_q>1 prefill path. Cover
    /// positions `[pos_start, pos_start + n_q)` via `pos_offset` set
    /// to `pos_start`. Decode path (n_q=1) ignores these.
    rope_q_batched_push: runtime.RopeBatchedPush,
    rope_k_batched_push: runtime.RopeBatchedPush,
    /// Fused FlashAttention forward push for the n_q>1 prefill path.
    /// Causal=1, write_lse=0 (we don't need LSE at inference). Decode
    /// path uses `fa_decode_split_push` + `fa_decode_merge_push`.
    fa_forward_push: runtime.FaForwardPush,
};

/// Compute forward-step push constants for a hybrid model. `pos_start`
/// is the absolute position of the first query in this batch; queries
/// occupy `[pos_start, pos_start + n_q)`. For decode `(pos, 1)` is
/// equivalent to the previous single-position signature.
pub fn computeForwardPushes(cfg: config_mod.Config, pos_start: usize, n_q: u32, max_pos: u32) ForwardPushes {
    const hidden: u32 = @intCast(cfg.hidden_size);
    const inter: u32 = @intCast(cfg.intermediate_size);
    const head_dim: u32 = @intCast(cfg.head_dim);
    const n_q_heads: u32 = @intCast(cfg.num_attention_heads);
    const n_kv_heads: u32 = @intCast(cfg.num_key_value_heads);
    const heads_per_kv: u32 = n_q_heads / n_kv_heads;
    const q_dim: u32 = n_q_heads * head_dim;
    const q_proj_rows: u32 = if (cfg.attn_output_gate) 2 * q_dim else q_dim;
    const kv_dim: u32 = n_kv_heads * head_dim;
    const rotary_dim: u32 = @intFromFloat(@as(f32, @floatFromInt(head_dim)) * cfg.partial_rotary_factor);

    const conv_dim: u32 = @intCast(cfg.linearAttnConvDim());
    const value_dim: u32 = @intCast(cfg.linear_num_value_heads * cfg.linear_value_head_dim);
    const key_dim: u32 = @intCast(cfg.linear_num_key_heads * cfg.linear_key_head_dim);
    const n_v_heads: u32 = @intCast(cfg.linear_num_value_heads);
    const n_k_heads_lin: u32 = @intCast(cfg.linear_num_key_heads);
    const head_k: u32 = @intCast(cfg.linear_key_head_dim);
    const head_v: u32 = @intCast(cfg.linear_value_head_dim);
    const conv_kernel: u32 = @intCast(cfg.linear_conv_kernel_dim);
    const q_scale: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(cfg.linear_key_head_dim)));

    const gemma_quirk: u32 = if (cfg.family.rmsnormAddOne()) 1 else 0;
    // `n_pos` here = the largest KV history length any query in this
    // batch will see (i.e. `pos_start + n_q`). Per-row pushes below use
    // row-zero's `pos_start` for now; β-3 recomputes per-row variants
    // inside `recordOneLayer` when `n_q > 1`.
    const n_pos: u32 = @intCast(pos_start + n_q);

    return .{
        .rms_push = .{ .dim = hidden, .eps = cfg.rms_norm_eps, .gemma_quirk = gemma_quirk },
        .qkn_push = .{ .dim = head_dim, .eps = cfg.rms_norm_eps, .gemma_quirk = gemma_quirk },
        .add_push = .{ .n = hidden },
        .swiglu_push = .{ .n = inter },
        .conv1d_push = .{ .conv_dim = conv_dim, .kernel_size = conv_kernel },
        .l2_push = .{ .head_dim = head_k, .eps = 1e-6 },
        .rms_gated_push = .{ .head_dim = head_v, .eps = cfg.rms_norm_eps },
        .gds_push = .{
            .num_k_heads = n_k_heads_lin,
            .num_v_heads = n_v_heads,
            .head_k = head_k,
            .head_v = head_v,
        },
        .split_push = .{ .num_heads = n_q_heads, .head_dim = head_dim },
        .sigmul_push = .{ .n_elem = q_dim },
        .rope_q_push = .{
            .n_heads = n_q_heads,
            .head_dim = head_dim,
            .rotary_dim = rotary_dim,
            .pos = @intCast(pos_start),
            .theta_base = cfg.rope_theta,
        },
        .rope_k_push = .{
            .n_heads = n_kv_heads,
            .head_dim = head_dim,
            .rotary_dim = rotary_dim,
            .pos = @intCast(pos_start),
            .theta_base = cfg.rope_theta,
        },
        .kv_write_push = .{ .n = kv_dim, .dst_off = @as(u32, @intCast(pos_start)) * kv_dim },
        .scores_push = .{
            .n_heads = n_q_heads,
            .heads_per_kv = heads_per_kv,
            .head_dim = head_dim,
            .n_pos = n_pos,
            .kv_stride = kv_dim,
            .scores_stride = max_pos,
            .inv_sqrt_dim = 1.0 / @sqrt(@as(f32, @floatFromInt(head_dim))),
        },
        .softmax_push = .{ .dim = n_pos, .stride = max_pos },
        .attn_out_push = .{
            .n_heads = n_q_heads,
            .heads_per_kv = heads_per_kv,
            .head_dim = head_dim,
            .n_pos = n_pos,
            .kv_stride = kv_dim,
            .scores_stride = max_pos,
        },
        .attn_use_fa = head_dim <= runtime.FA_HEAD_DIM_MAX,
        .fa_decode_split_push = blk: {
            const ch = runtime.chooseFaDecodeSplit(n_pos);
            break :blk .{
                .n_heads = n_q_heads,
                .heads_per_kv = heads_per_kv,
                .head_dim = head_dim,
                .n_kv = n_pos,
                .kv_stride = kv_dim,
                .n_splits = ch.n_splits,
                .split_size = ch.split_size,
                .inv_sqrt_dim = 1.0 / @sqrt(@as(f32, @floatFromInt(head_dim))),
            };
        },
        .fa_decode_merge_push = .{
            .n_heads = n_q_heads,
            .head_dim = head_dim,
            .n_splits = runtime.chooseFaDecodeSplit(n_pos).n_splits,
        },
        .n_pos = n_pos,
        .q_scale = q_scale,
        .key_dim = key_dim,
        .value_dim = value_dim,
        .n_v_heads = n_v_heads,
        .n_k_heads_lin = n_k_heads_lin,
        .n_q_heads = n_q_heads,
        .n_kv_heads = n_kv_heads,
        .head_dim = head_dim,
        .q_dim = q_dim,
        .q_proj_rows = q_proj_rows,
        .kv_dim = kv_dim,
        .conv_dim = conv_dim,
        .pos_start = @intCast(pos_start),
        .n_q = n_q,
        .rope_q_batched_push = .{
            .n_pos = n_q,
            .n_heads = n_q_heads,
            .head_dim = head_dim,
            .rotary_dim = rotary_dim,
            .theta_base = cfg.rope_theta,
            .pos_offset = @intCast(pos_start),
        },
        .rope_k_batched_push = .{
            .n_pos = n_q,
            .n_heads = n_kv_heads,
            .head_dim = head_dim,
            .rotary_dim = rotary_dim,
            .theta_base = cfg.rope_theta,
            .pos_offset = @intCast(pos_start),
        },
        .fa_forward_push = .{
            .n_q = n_q,
            .n_heads = n_q_heads,
            .heads_per_kv = heads_per_kv,
            .head_dim = head_dim,
            .n_kv = n_pos,
            .kv_stride = kv_dim,
            .causal = 1,
            .write_lse = 0,
            .inv_sqrt_dim = 1.0 / @sqrt(@as(f32, @floatFromInt(head_dim))),
        },
    };
}

// ── Per-layer + full-step recording ───────────────────────────────

/// Record the dispatches for a single hybrid transformer block. The
/// layer may be either a linear-attention (Gated DeltaNet) layer or
/// a full-attention layer; the layer body branches on
/// `layer.layer_type`. Caller manages the recorder's begin /
/// endAndSubmit cycle.
pub fn recordOneLayer(
    rec: *recorder.Recorder,
    sc: *const Scratch,
    state: *const State,
    gm: *const gpu_model.GpuModel,
    cfg: config_mod.Config,
    k: *const ChatKernels,
    layer_idx: usize,
    pos: usize,
    p: *const ForwardPushes,
    tq4_v: ?Tq4VHooks,
) !void {
    const hidden: u32 = @intCast(cfg.hidden_size);
    const inter: u32 = @intCast(cfg.intermediate_size);
    const layer = &gm.layers[layer_idx];

    try runtime.recDispatchPerRow(rec, &k.rmsnorm, &.{ &sc.stream, &layer.input_layernorm, &sc.x_norm }, &p.rms_push, 1);

    switch (layer.layer_type) {
        .linear_attention => {
            try runtime.recDispatchMatmul(rec, &k.matmul, &.{ &sc.x_norm, &layer.in_proj_qkv.?, &sc.mixed_qkv }, 1, p.conv_dim, hidden);
            try runtime.recDispatchMatmul(rec, &k.matmul, &.{ &sc.x_norm, &layer.in_proj_z.?, &sc.z }, 1, p.value_dim, hidden);
            try runtime.recDispatchMatmul(rec, &k.matmul, &.{ &sc.x_norm, &layer.in_proj_b.?, &sc.b_raw }, 1, p.n_v_heads, hidden);
            try runtime.recDispatchMatmul(rec, &k.matmul, &.{ &sc.x_norm, &layer.in_proj_a.?, &sc.a_raw }, 1, p.n_v_heads, hidden);
            try runtime.recDispatch1D(rec, &k.conv1d_update, &.{ &sc.mixed_qkv, &layer.conv1d_weight.?, &state.ssm_conv[layer_idx].?, &sc.mixed_qkv_post }, &p.conv1d_push, p.conv_dim);

            const slice_q_push = SliceCopyPush{ .src_off = 0,             .dst_off = 0, .n_elem = p.key_dim };
            const slice_k_push = SliceCopyPush{ .src_off = p.key_dim,     .dst_off = 0, .n_elem = p.key_dim };
            const slice_v_push = SliceCopyPush{ .src_off = 2 * p.key_dim, .dst_off = 0, .n_elem = p.value_dim };
            try runtime.recDispatch1D(rec, &k.slice_copy, &.{ &sc.mixed_qkv_post, &sc.q_lin }, &slice_q_push, p.key_dim);
            try runtime.recDispatch1D(rec, &k.slice_copy, &.{ &sc.mixed_qkv_post, &sc.k_lin }, &slice_k_push, p.key_dim);
            try runtime.recDispatch1D(rec, &k.slice_copy, &.{ &sc.mixed_qkv_post, &sc.v_lin }, &slice_v_push, p.value_dim);

            try rec.dispatch(&k.l2norm_per_head, &.{ &sc.q_lin, &sc.q_lin_n }, &p.l2_push, p.n_k_heads_lin, 1, 1);
            try rec.dispatch(&k.l2norm_per_head, &.{ &sc.k_lin, &sc.k_lin_n }, &p.l2_push, p.n_k_heads_lin, 1, 1);
            const scale_push = ScalePush{ .n = p.key_dim, .scale = p.q_scale };
            try runtime.recDispatch1D(rec, &k.scale, &.{ &sc.q_lin_n, &sc.q_lin }, &scale_push, p.key_dim);

            try rec.dispatch(&k.gated_delta_step, &.{
                &state.ssm_rec[layer_idx].?, &sc.q_lin, &sc.k_lin_n, &sc.v_lin,
                &sc.b_raw, &sc.a_raw, &layer.A_log.?, &layer.dt_bias.?,
                &sc.y,
            }, &p.gds_push, p.n_v_heads, 1, 1);
            try rec.dispatch(&k.rmsnorm_gated, &.{ &sc.y, &sc.z, &layer.ssm_norm_weight.?, &sc.post_norm }, &p.rms_gated_push, p.n_v_heads, 1, 1);
            try runtime.recDispatchMatmul(rec, &k.matmul, &.{ &sc.post_norm, &layer.out_proj.?, &sc.attn_out }, 1, hidden, p.value_dim);
        },
        .full_attention => {
            try runtime.recDispatchMatmul(rec, &k.matmul, &.{ &sc.x_norm, &layer.q_proj.?, &sc.q_gate }, 1, p.q_proj_rows, hidden);
            try runtime.recDispatch1D(rec, &k.split_q_gate, &.{ &sc.q_gate, &sc.q, &sc.gate_attn }, &p.split_push, p.q_dim);
            try runtime.recDispatchMatmul(rec, &k.matmul, &.{ &sc.x_norm, &layer.k_proj.?, &sc.k }, 1, p.kv_dim, hidden);
            try runtime.recDispatchMatmul(rec, &k.matmul, &.{ &sc.x_norm, &layer.v_proj.?, &sc.v }, 1, p.kv_dim, hidden);
            try runtime.recDispatchPerRow(rec, &k.rmsnorm, &.{ &sc.q, &layer.q_norm.?, &sc.q }, &p.qkn_push, p.n_q_heads);
            try runtime.recDispatchPerRow(rec, &k.rmsnorm, &.{ &sc.k, &layer.k_norm.?, &sc.k }, &p.qkn_push, p.n_kv_heads);
            try runtime.recDispatch1D(rec, &k.rope_partial, &.{ &sc.q, &sc.qrot }, &p.rope_q_push, p.n_q_heads * p.head_dim);
            try runtime.recDispatch1D(rec, &k.rope_partial, &.{ &sc.k, &sc.krot }, &p.rope_k_push, p.n_kv_heads * p.head_dim);
            try runtime.recDispatch1D(rec, &k.kv_write, &.{ &sc.krot, &state.kv_k[layer_idx].? }, &p.kv_write_push, p.kv_dim);

            // V write: legacy fp32 raw copy, or TQ4 quantising
            // pack-to-cache when tq4_v is supplied.
            if (tq4_v) |t| {
                const pack_push = Tq4PackPush{ .dst_block_idx = @as(u32, @intCast(pos)) * t.n_blocks_per_pos };
                try rec.dispatch(t.pack, &.{ &sc.v, &state.kv_v_tq4[layer_idx].? }, &pack_push, t.n_blocks_per_pos, 1, 1);
            } else {
                try runtime.recDispatch1D(rec, &k.kv_write, &.{ &sc.v, &state.kv_v[layer_idx].? }, &p.kv_write_push, p.kv_dim);
            }

            // V buffer for attention: TQ4-V dequants here on-demand
            // (lifted above the FA / 3-pass branch so both paths read
            // the same binding). When the host isn't using TQ4-V the
            // fp32 cache is fed straight in.
            //
            // The T-arc fused `fa_decode_split_tq4v` kernel exists and
            // parity-matches (see `--fa-decode-tq4v-smoke`) but is *not*
            // dispatched here — see the matching note in
            // `runtime.recordOneLayer` and `docs/perf.md` §"Fused TQ4-V"
            // for the bench data showing why the unfused parallel
            // unpack wins on RTX 3090.
            const v_for_attn: *const buffer.Buffer = if (tq4_v) |t| blk: {
                const total_blocks: u32 = p.n_pos * t.n_blocks_per_pos;
                try rec.dispatch(t.unpack, &.{ &state.kv_v_tq4[layer_idx].?, &sc.dequant_v.? }, null, total_blocks, 1, 1);
                break :blk &sc.dequant_v.?;
            } else &state.kv_v[layer_idx].?;

            if (p.attn_use_fa) {
                // FlashDecoding (split-K + merge). `head_dim ≤
                // FA_HEAD_DIM_MAX` is enforced at `computeForwardPushes`
                // time. Same kernel pair the dense path uses; the only
                // difference is Qwen3.5/3.6 has the q-gate branch downstream
                // (`sigmoid_mul` after `head_out` → `head_out_gated`).
                const split = p.fa_decode_split_push;
                try rec.dispatch(
                    &k.fa_decode_split,
                    &.{ &sc.qrot, &state.kv_k[layer_idx].?, v_for_attn, &sc.fa_o_partial, &sc.fa_m_partial, &sc.fa_l_partial },
                    &split,
                    p.n_q_heads * split.n_splits,
                    1,
                    1,
                );
                try rec.dispatch(
                    &k.fa_decode_merge,
                    &.{ &sc.fa_o_partial, &sc.fa_m_partial, &sc.fa_l_partial, &sc.head_out },
                    &p.fa_decode_merge_push,
                    p.n_q_heads,
                    1,
                    1,
                );
            } else {
                // 3-pass fallback. Used when head_dim > FA_HEAD_DIM_MAX
                // — currently no Qwen3.5/3.6 hybrid model triggers this,
                // but it's the same shape-agnostic chain the dense path
                // uses for non-FA heads.
                try rec.dispatch(&k.scores, &.{ &sc.qrot, &state.kv_k[layer_idx].?, &sc.scores }, &p.scores_push, p.n_q_heads * p.n_pos, 1, 1);
                try runtime.recDispatchPerRow(rec, &k.softmax, &.{ &sc.scores, &sc.scores }, &p.softmax_push, p.n_q_heads);
                try rec.dispatch(&k.attn_out, &.{ &sc.scores, v_for_attn, &sc.head_out }, &p.attn_out_push, p.n_q_heads * p.head_dim, 1, 1);
            }
            try runtime.recDispatch1D(rec, &k.sigmoid_mul, &.{ &sc.head_out, &sc.gate_attn, &sc.head_out_gated }, &p.sigmul_push, p.q_dim);
            try runtime.recDispatchMatmul(rec, &k.matmul, &.{ &sc.head_out_gated, &layer.o_proj.?, &sc.attn_out }, 1, hidden, p.q_dim);
        },
    }

    try runtime.recDispatch1D(rec, &k.add, &.{ &sc.stream, &sc.attn_out }, &p.add_push, hidden);
    try runtime.recDispatchPerRow(rec, &k.rmsnorm, &.{ &sc.stream, &layer.post_attention_layernorm, &sc.mid_norm }, &p.rms_push, 1);
    try runtime.recDispatchMatmul(rec, &k.matmul, &.{ &sc.mid_norm, &layer.gate_proj, &sc.gate }, 1, inter, hidden);
    try runtime.recDispatchMatmul(rec, &k.matmul, &.{ &sc.mid_norm, &layer.up_proj, &sc.up }, 1, inter, hidden);
    try runtime.recDispatch1D(rec, &k.swiglu, &.{ &sc.gate, &sc.up, &sc.fused }, &p.swiglu_push, inter);
    try runtime.recDispatchMatmul(rec, &k.matmul, &.{ &sc.fused, &layer.down_proj, &sc.ffn_out }, 1, hidden, inter);
    try runtime.recDispatch1D(rec, &k.add, &.{ &sc.stream, &sc.ffn_out }, &p.add_push, hidden);
}

/// Record a single full-attention layer's forward at n_q query rows
/// (batched-prefill primitive). Mirrors the `.full_attention` arm of
/// `recordOneLayer` but consumes a pre-RMSNormed `x_norm` (n_q × hidden)
/// and produces the layer's residual contribution `attn_out` (n_q ×
/// hidden) — caller does the residual add. Uses `fa_forward` (D-arc-
/// covered head_dim ≤ 256) with `causal = 1` so query t only sees
/// keys at positions `[0, pos_start + t]`. KV cache rows
/// `[pos_start, pos_start + n_q)` are appended.
///
/// β-3 building block: GPU companion to `cpu/full_attn.zig::prefillStep`.
/// β-5 will call it from `recordOneLayer` when `p.n_q > 1`. Decode
/// (n_q=1) keeps using the existing `fa_decode_split + merge` chain.
///
/// Limitations (β-3 scope):
///   * fp32 V cache only (no TQ4-V on the prefill branch yet).
///   * No 3-pass fallback — head_dim must be ≤ FA_HEAD_DIM_MAX.
///   * Linear-attention layers must be widened separately (β-4).
///
/// Inputs:
///   `x_norm_in`  — `[n_q, hidden_size]` fp32, pre-attn RMSNorm output
///                  (caller computes this).
///   `attn_out`   — `[n_q, hidden_size]` fp32, written by this call.
pub fn recordFullAttnLayerBatched(
    rec: *recorder.Recorder,
    sc: *const Scratch,
    state: *const State,
    gm: *const gpu_model.GpuModel,
    cfg: config_mod.Config,
    k: *const ChatKernels,
    layer_idx: usize,
    p: *const ForwardPushes,
    x_norm_in: *const buffer.Buffer,
    attn_out: *const buffer.Buffer,
) !void {
    if (p.n_q < 1) return error.InvalidNq;
    if (p.head_dim > runtime.FA_HEAD_DIM_MAX) return error.HeadDimTooLargeForFa;
    const layer = &gm.layers[layer_idx];
    if (layer.layer_type != .full_attention) return error.NotFullLayer;

    const hidden: u32 = @intCast(cfg.hidden_size);
    const n_q_rows = p.n_q;

    // 1. Q-projection (2× wide if attn_output_gate). M = n_q.
    try runtime.recDispatchMatmul(rec, &k.matmul, &.{ x_norm_in, &layer.q_proj.?, &sc.q_gate }, n_q_rows, p.q_proj_rows, hidden);

    // 2. split_q_gate per row. The shader's per-element math is
    //    independent across rows, so we can dispatch
    //    `n_q × n_q_heads × head_dim` threads in one call by passing
    //    an inflated `num_heads = n_q × n_q_heads`. Layout works
    //    because `[n_q, n_q_heads, 2 × head_dim]` flat is identical
    //    to `[n_q × n_q_heads, 2 × head_dim]`.
    const split_push_batched = SplitQGatePush{
        .num_heads = n_q_rows * p.n_q_heads,
        .head_dim = p.head_dim,
    };
    try runtime.recDispatch1D(rec, &k.split_q_gate, &.{ &sc.q_gate, &sc.q, &sc.gate_attn }, &split_push_batched, n_q_rows * p.q_dim);

    // 3. K and V projections. M = n_q.
    try runtime.recDispatchMatmul(rec, &k.matmul, &.{ x_norm_in, &layer.k_proj.?, &sc.k }, n_q_rows, p.kv_dim, hidden);
    try runtime.recDispatchMatmul(rec, &k.matmul, &.{ x_norm_in, &layer.v_proj.?, &sc.v }, n_q_rows, p.kv_dim, hidden);

    // 4. Per-head q_norm / k_norm — `n_q × n_heads` rows.
    try runtime.recDispatchPerRow(rec, &k.rmsnorm, &.{ &sc.q, &layer.q_norm.?, &sc.q }, &p.qkn_push, n_q_rows * p.n_q_heads);
    try runtime.recDispatchPerRow(rec, &k.rmsnorm, &.{ &sc.k, &layer.k_norm.?, &sc.k }, &p.qkn_push, n_q_rows * p.n_kv_heads);

    // 5. Batched RoPE — covers [pos_start, pos_start + n_q) per row.
    try runtime.recDispatch1D(rec, &k.rope_partial_batched, &.{ &sc.q, &sc.qrot }, &p.rope_q_batched_push, n_q_rows * p.n_q_heads * p.head_dim);
    try runtime.recDispatch1D(rec, &k.rope_partial_batched, &.{ &sc.k, &sc.krot }, &p.rope_k_batched_push, n_q_rows * p.n_kv_heads * p.head_dim);

    // 6. Append n_q rows of K, V to the layer's KV cache. krot/v are
    //    contiguous `[n_q × kv_dim]`, the cache rows
    //    `[pos_start..pos_start+n_q)` are also contiguous, so a single
    //    `kv_write` per buffer suffices.
    const kv_write_batched = KvWritePush{
        .n = n_q_rows * p.kv_dim,
        .dst_off = p.pos_start * p.kv_dim,
    };
    try runtime.recDispatch1D(rec, &k.kv_write, &.{ &sc.krot, &state.kv_k[layer_idx].? }, &kv_write_batched, n_q_rows * p.kv_dim);
    try runtime.recDispatch1D(rec, &k.kv_write, &.{ &sc.v, &state.kv_v[layer_idx].? }, &kv_write_batched, n_q_rows * p.kv_dim);

    // 7. fa_forward — n_q × n_heads workgroups, causal mask honours
    //    `[0..pos_start + t]` per query row t. Reads K/V from the
    //    full cache (n_kv = pos_start + n_q).
    try rec.dispatch(
        &k.fa_forward,
        &.{ &sc.qrot, &state.kv_k[layer_idx].?, &state.kv_v[layer_idx].?, &sc.head_out, &sc.fa_lse },
        &p.fa_forward_push,
        n_q_rows * p.n_q_heads,
        1,
        1,
    );

    // 8. attn_output_gate — head_out *= sigmoid(gate). Pointwise.
    const sigmul_batched = SigmoidMulPush{ .n_elem = n_q_rows * p.q_dim };
    try runtime.recDispatch1D(rec, &k.sigmoid_mul, &.{ &sc.head_out, &sc.gate_attn, &sc.head_out_gated }, &sigmul_batched, n_q_rows * p.q_dim);

    // 9. o_proj — M = n_q.
    try runtime.recDispatchMatmul(rec, &k.matmul, &.{ &sc.head_out_gated, &layer.o_proj.?, attn_out }, n_q_rows, hidden, p.q_dim);
}

/// Record a single Gated-DeltaNet (linear-attention) layer's forward
/// at n_q query rows. Linear-attention recurrence is fundamentally
/// sequential — each row mutates `state.ssm_{conv,rec}` in place — so
/// this loops the existing single-row chain n_q times, slicing one
/// row of `x_norm_in` into `sc.x_norm_lin` per iteration and slicing
/// the per-row output from `sc.attn_out_lin` into the right offset of
/// `attn_out`.
///
/// β-4 building block: GPU companion to a sequential
/// `gated_delta.decodeStep` loop on the CPU side. β-5 will dispatch
/// it from `recordOneLayer` when `p.n_q > 1`.
///
/// Inputs:
///   `x_norm_in`  — `[n_q, hidden_size]` fp32, pre-attn RMSNorm output.
///   `attn_out`   — `[n_q, hidden_size]` fp32, written row-by-row.
pub fn recordLinearAttnLayerBatched(
    rec: *recorder.Recorder,
    sc: *const Scratch,
    state: *const State,
    gm: *const gpu_model.GpuModel,
    cfg: config_mod.Config,
    k: *const ChatKernels,
    layer_idx: usize,
    p: *const ForwardPushes,
    x_norm_in: *const buffer.Buffer,
    attn_out: *const buffer.Buffer,
) !void {
    if (p.n_q < 1) return error.InvalidNq;
    const layer = &gm.layers[layer_idx];
    if (layer.layer_type != .linear_attention) return error.NotLinearLayer;

    const hidden: u32 = @intCast(cfg.hidden_size);

    var t: u32 = 0;
    while (t < p.n_q) : (t += 1) {
        // 1. slice_copy x_norm[t] → x_norm_lin
        const slice_in = SliceCopyPush{
            .src_off = t * hidden,
            .dst_off = 0,
            .n_elem = hidden,
        };
        try runtime.recDispatch1D(rec, &k.slice_copy, &.{ x_norm_in, &sc.x_norm_lin }, &slice_in, hidden);

        // 2. Single-row chain on x_norm_lin → attn_out_lin. Mirrors
        //    the existing `.linear_attention` arm of `recordOneLayer`,
        //    swapping `&sc.x_norm` for `&sc.x_norm_lin` and
        //    `&sc.attn_out` for `&sc.attn_out_lin` so this iteration
        //    doesn't smear across rows.
        try runtime.recDispatchMatmul(rec, &k.matmul, &.{ &sc.x_norm_lin, &layer.in_proj_qkv.?, &sc.mixed_qkv }, 1, p.conv_dim, hidden);
        try runtime.recDispatchMatmul(rec, &k.matmul, &.{ &sc.x_norm_lin, &layer.in_proj_z.?, &sc.z }, 1, p.value_dim, hidden);
        try runtime.recDispatchMatmul(rec, &k.matmul, &.{ &sc.x_norm_lin, &layer.in_proj_b.?, &sc.b_raw }, 1, p.n_v_heads, hidden);
        try runtime.recDispatchMatmul(rec, &k.matmul, &.{ &sc.x_norm_lin, &layer.in_proj_a.?, &sc.a_raw }, 1, p.n_v_heads, hidden);
        try runtime.recDispatch1D(rec, &k.conv1d_update, &.{ &sc.mixed_qkv, &layer.conv1d_weight.?, &state.ssm_conv[layer_idx].?, &sc.mixed_qkv_post }, &p.conv1d_push, p.conv_dim);

        const slice_q_push = SliceCopyPush{ .src_off = 0, .dst_off = 0, .n_elem = p.key_dim };
        const slice_k_push = SliceCopyPush{ .src_off = p.key_dim, .dst_off = 0, .n_elem = p.key_dim };
        const slice_v_push = SliceCopyPush{ .src_off = 2 * p.key_dim, .dst_off = 0, .n_elem = p.value_dim };
        try runtime.recDispatch1D(rec, &k.slice_copy, &.{ &sc.mixed_qkv_post, &sc.q_lin }, &slice_q_push, p.key_dim);
        try runtime.recDispatch1D(rec, &k.slice_copy, &.{ &sc.mixed_qkv_post, &sc.k_lin }, &slice_k_push, p.key_dim);
        try runtime.recDispatch1D(rec, &k.slice_copy, &.{ &sc.mixed_qkv_post, &sc.v_lin }, &slice_v_push, p.value_dim);

        try rec.dispatch(&k.l2norm_per_head, &.{ &sc.q_lin, &sc.q_lin_n }, &p.l2_push, p.n_k_heads_lin, 1, 1);
        try rec.dispatch(&k.l2norm_per_head, &.{ &sc.k_lin, &sc.k_lin_n }, &p.l2_push, p.n_k_heads_lin, 1, 1);
        const scale_push = ScalePush{ .n = p.key_dim, .scale = p.q_scale };
        try runtime.recDispatch1D(rec, &k.scale, &.{ &sc.q_lin_n, &sc.q_lin }, &scale_push, p.key_dim);

        try rec.dispatch(&k.gated_delta_step, &.{
            &state.ssm_rec[layer_idx].?, &sc.q_lin, &sc.k_lin_n, &sc.v_lin,
            &sc.b_raw, &sc.a_raw, &layer.A_log.?, &layer.dt_bias.?,
            &sc.y,
        }, &p.gds_push, p.n_v_heads, 1, 1);
        try rec.dispatch(&k.rmsnorm_gated, &.{ &sc.y, &sc.z, &layer.ssm_norm_weight.?, &sc.post_norm }, &p.rms_gated_push, p.n_v_heads, 1, 1);
        try runtime.recDispatchMatmul(rec, &k.matmul, &.{ &sc.post_norm, &layer.out_proj.?, &sc.attn_out_lin }, 1, hidden, p.value_dim);

        // 3. slice_copy attn_out_lin → attn_out[t]
        const slice_out = SliceCopyPush{
            .src_off = 0,
            .dst_off = t * hidden,
            .n_elem = hidden,
        };
        try runtime.recDispatch1D(rec, &k.slice_copy, &.{ &sc.attn_out_lin, attn_out }, &slice_out, hidden);
    }
}

/// One-call full hybrid forward: embed → all layers → optional
/// sample-step. Identical dispatch order to valkyr's CLI hybrid path,
/// so generation output is bit-identical.
pub fn recordForwardStep(
    rec: *recorder.Recorder,
    sc: *const Scratch,
    state: *const State,
    gm: *const gpu_model.GpuModel,
    cfg: config_mod.Config,
    k: *const ChatKernels,
    pos: usize,
    token_id: u32,
    max_pos: u32,
    tq4_v: ?Tq4VHooks,
    compute_logits: bool,
) !void {
    const hidden: u32 = @intCast(cfg.hidden_size);
    const vocab: u32 = @intCast(cfg.vocab_size);

    const pushes = computeForwardPushes(cfg, pos, 1, max_pos);
    const embed_push = EmbedLookupPush{ .token_id = token_id, .dim = hidden, .scale = 1.0 };

    try runtime.recDispatch1D(rec, &k.embed, &.{ &gm.embed_tokens, &sc.stream }, &embed_push, hidden);

    for (0..cfg.num_hidden_layers) |layer_idx| {
        try recordOneLayer(rec, sc, state, gm, cfg, k, layer_idx, pos, &pushes, tq4_v);
    }

    if (compute_logits) {
        try runtime.recDispatchPerRow(rec, &k.rmsnorm, &.{ &sc.stream, &gm.final_norm, &sc.final_norm_out }, &pushes.rms_push, 1);
        try runtime.recDispatchMatmul(rec, &k.matmul_lm_head, &.{ &sc.final_norm_out, &gm.lm_head, &sc.logits }, 1, vocab, hidden);
    }
}
