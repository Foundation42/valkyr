//! Library-side forward-step + sampling primitives for embedded callers.
//!
//! `loader.zig` (chunk 7a) handles the static "model on disk → model on
//! GPU" half. This file handles the runtime half: per-token forward
//! recording + CPU-side sampling. Together they are everything chunk
//! 7c's `Session` needs to wrap a default state machine around.
//!
//! Scope for chunk 7b
//! ──────────────────
//! - Dense Llama / Gemma full-attention only (no hybrid linear-attn,
//!   no TQ4 V-cache). Hybrid + TQ4 will get exposed if/when a host
//!   actually wants them; today's first integration target is
//!   straightforward dense models.
//! - One-shot full-forward (`Forward.recordStep`) is enough for ai_demo's
//!   "load + tokenize + forward + print predicted token" verification.
//!   The per-layer split needed for frame-budget chunking is exposed
//!   too via `Forward.recordOneLayer` so chunk 7c's Session can build
//!   the state machine on top.
//!
//! Why this duplicates structs from main.zig
//! ─────────────────────────────────────────
//! valkyr's CLI (`main.zig`) declares its own copies of these push
//! structs + helpers as file-private consts; ~325 references inside
//! main.zig touch them. Lifting all of those to runtime.zig would be
//! a big-blast-radius refactor for chunk 7b.
//!
//! Both definitions are layout-compatible (extern structs with
//! identical field order), so they marshal the same bytes into the
//! same SPIR-V shaders. Drift risk is bounded — these structs change
//! infrequently (stable since phase 1) and any divergence would show
//! up as immediate validation-layer / output-correctness failures.
//! When chunk 7c lands the Session API, we can DRY main.zig to
//! re-import these from runtime as a follow-up cleanup.

const std = @import("std");
const vk = @import("gpu/vk.zig");
const buffer = @import("gpu/buffer.zig");
const pipeline = @import("gpu/pipeline.zig");
const recorder = @import("gpu/recorder.zig");
const gpu_model = @import("gpu/model.zig");
const gpu_scratch = @import("gpu/scratch.zig");
const config_mod = @import("config.zig");
const shaders = @import("shaders");

// ── Push structs ──────────────────────────────────────────────────

pub const RmsnormPush = extern struct {
    dim: u32,
    eps: f32,
    gemma_quirk: u32,
};

/// LayerNorm has neither the gemma_quirk gain offset nor a third
/// configurable knob — `(dim, eps)` is enough for both forward and
/// backward variants.
pub const LayernormPush = extern struct {
    dim: u32,
    eps: f32,
};

/// Embedding-table gradient scatter. `vocab_size` is included for
/// shader-side bounds checks but the actual workgroup count is set
/// by the caller's dispatch and must equal vocab_size.
pub const EmbeddingBackwardPush = extern struct {
    dim: u32,
    n_pos: u32,
    vocab_size: u32,
};

pub const RopePush = extern struct {
    n_heads: u32,
    head_dim: u32,
    pos: u32,
    theta_base: f32,
};

pub const RopePartialPush = extern struct {
    n_heads: u32,
    head_dim: u32,
    rotary_dim: u32,
    pos: u32,
    theta_base: f32,
};

pub const KvWritePush = extern struct {
    n: u32,
    dst_off: u32,
};

pub const Tq4PackPush = extern struct { dst_block_idx: u32 };

/// Optional TQ4 V-cache hooks. When supplied to `recordOneLayer`,
/// V is packed into the TQ4 cache instead of the fp32 V cache
/// passed via `kv`, and the whole V history is dequantised into a
/// scratch buffer just before attention. K stays in `kv` (K=fp /
/// V=TQ4 asymmetric).
pub const Tq4VHooks = struct {
    pack: *const pipeline.Kernel,
    unpack: *const pipeline.Kernel,
    cache: *const gpu_scratch.GpuKvCacheTq4,
};

pub const AttnScoresPush = extern struct {
    n_heads: u32,
    heads_per_kv: u32,
    head_dim: u32,
    n_pos: u32,
    kv_stride: u32,
    scores_stride: u32,
    inv_sqrt_dim: f32,
};

pub const SoftmaxPush = extern struct { dim: u32, stride: u32 };

pub const AttnOutputPush = extern struct {
    n_heads: u32,
    heads_per_kv: u32,
    head_dim: u32,
    n_pos: u32,
    kv_stride: u32,
    scores_stride: u32,
};

pub const AttnBackwardDattnPush = extern struct {
    n_q: u32,
    n_heads: u32,
    heads_per_kv: u32,
    head_dim: u32,
    n_kv: u32,
    kv_stride: u32,       // n_kv_heads * head_dim
    attn_stride: u32,     // row stride per (q, h) in d_attn (== n_kv typically)
};

pub const AttnBackwardDvPush = extern struct {
    n_q: u32,
    n_heads: u32,
    heads_per_kv: u32,
    n_kv_heads: u32,
    head_dim: u32,
    n_kv: u32,
    attn_stride: u32,
};

pub const AttnBackwardDqPush = extern struct {
    n_q: u32,
    n_heads: u32,
    heads_per_kv: u32,
    head_dim: u32,
    n_kv: u32,
    kv_stride: u32,
    scores_stride: u32,
    inv_sqrt_dim: f32,
};

pub const AttnBackwardDkPush = extern struct {
    n_q: u32,
    n_heads: u32,
    heads_per_kv: u32,
    n_kv_heads: u32,
    head_dim: u32,
    n_kv: u32,
    scores_stride: u32,
    inv_sqrt_dim: f32,
};

pub const AddInPlacePush = extern struct { n: u32 };

pub const ReluPush = extern struct { n: u32 };

pub const ReluBackwardPush = extern struct { n: u32 };

pub const LinearBackwardDxPush = extern struct { dim_out: u32, dim_in: u32 };

/// Batched linear-layer backward (dx and dW variants share this layout).
/// Forward: out = x @ Wᵀ where x[M, K], W[N, K], out[M, N].
pub const LinearBatchedPush = extern struct { M: u32, N: u32, K: u32 };

pub const OuterProductPush = extern struct { dim_out: u32, dim_in: u32 };

pub const SgdStepPush = extern struct { n: u32, lr: f32, weight_decay: f32 = 0 };

pub const AdamStepPush = extern struct {
    n: u32,
    lr: f32,
    beta1: f32,
    beta2: f32,
    eps: f32,
    t: u32,
    weight_decay: f32 = 0,
};

pub const MseLossGradPush = extern struct { n: u32 };

/// SwiGLU FFN nonlinearity. Both forward and backward kernels are
/// elementwise over `n` output values; same push struct serves both.
pub const SwigluPush = extern struct { n: u32 };

/// Batched RoPE (forward + backward share this struct). One dispatch
/// covers `n_pos` rows of `[n_heads, head_dim]`. Setting `rotary_dim
/// = head_dim` gives full RoPE; smaller gives Qwen3.5-style partial.
pub const RopeBatchedPush = extern struct {
    n_pos: u32,
    n_heads: u32,
    head_dim: u32,
    rotary_dim: u32,
    theta_base: f32,
};

/// Push struct for the fused QK-RoPE shaders (qk_rope_partial_batched +
/// qk_rope_backward_batched). Both Q and K are processed in one
/// dispatch — the kernel routes the first n_q_heads heads to Q's
/// buffer and the next n_kv_heads heads to K's buffer. Otherwise the
/// rotation math is identical to the unfused rope_partial_batched
/// pair, so the parity is bit-equal modulo subgroup ordering.
pub const QkRopeBatchedPush = extern struct {
    n_pos: u32,
    n_q_heads: u32,
    n_kv_heads: u32,
    head_dim: u32,
    rotary_dim: u32,
    theta_base: f32,
};

/// Scaled MSE loss gradient. Bakes the (2/N) factor into the kernel
/// so the transformer-training side doesn't need a follow-up scale.
pub const MseLossGradScaledPush = extern struct { n: u32, scale: f32 };

/// Multi-query attention forward (training-style). Push fields mirror
/// the cpu_train_transformer.attentionForward signature; setting
/// `causal != 0` enables the `k > q + (n_kv − n_q)` mask.
pub const AttnScoresTrainPush = extern struct {
    n_q: u32,
    n_heads: u32,
    heads_per_kv: u32,
    head_dim: u32,
    n_kv: u32,
    kv_stride: u32,
    scores_stride: u32,
    causal: u32,
    inv_sqrt_dim: f32,
};

pub const AttnOutputTrainPush = extern struct {
    n_q: u32,
    n_heads: u32,
    heads_per_kv: u32,
    head_dim: u32,
    n_kv: u32,
    kv_stride: u32,
    attn_stride: u32,
};

pub const Mlp2ForwardBatchedPush = extern struct {
    dim_in: u32,
    dim_hidden: u32,
    dim_out: u32,
    n_samples: u32,
};

pub const Mlp2ForwardTrainBatchedPush = extern struct {
    dim_in: u32,
    dim_hidden: u32,
    dim_out: u32,
    n_samples: u32,
};

pub const Mlp2DyBatchedPush = extern struct {
    dim_out: u32,
    n_samples: u32,
};

pub const Mlp2DhPreBatchedPush = extern struct {
    dim_hidden: u32,
    dim_out: u32,
    n_samples: u32,
};

pub const Mlp2DwAccumPush = extern struct {
    dim_i: u32,
    dim_j: u32,
    n_samples: u32,
};

pub const Mlp2DbAccumPush = extern struct {
    dim_i: u32,
    n_samples: u32,
};

pub const SoftmaxCeLossGradPush = extern struct {
    dim_out: u32,
    n_samples: u32,
};

/// Cut Cross-Entropy forward — fused LM-head matmul + online-softmax CE.
/// One workgroup per row (n_samples WGs). The shader hardcodes
/// CHUNK == local_size_x == 256, so the only runtime parameters are the
/// problem shape: `n_samples` rows of `dim`-wide hidden states against a
/// `vocab`-row LM-head weight matrix. CPU oracle is in src/cpu/cce.zig.
pub const CceForwardPush = extern struct {
    n_samples: u32,
    vocab: u32,
    dim: u32,
    /// Optional z-loss scale (Chronicals §"Z-Loss"). Adds λ_z · lse² to
    /// the per-row loss. Default 0 ⇒ plain CE; typical training value
    /// is 1e-4. Cost is one scalar add per row in the shader.
    z_loss_scale: f32 = 0.0,
};

/// CCE backward — d_h component (one WG per row) and dW component (one
/// WG per vocab entry, mirrors embedding_backward.comp's vocab-major
/// layout to avoid VK_EXT_shader_atomic_float). Both kernels share the
/// same problem shape so they reuse the same push-constant struct.
pub const CceBackwardPush = extern struct {
    n_samples: u32,
    vocab: u32,
    dim: u32,
    /// Must match the value used in the matching cce_forward dispatch:
    /// the gradient picks up a (1 + 2·λ_z·lse) factor on the softmax
    /// part of dz when λ_z > 0. Default 0 ⇒ plain CE backward.
    z_loss_scale: f32 = 0.0,
};

pub const Mlp2LossBatchedPush = extern struct {
    dim_out: u32,
    n_samples: u32,
};

pub const GegluPush = extern struct { n: u32 };

pub const EmbedLookupPush = extern struct {
    token_id: u32,
    dim: u32,
    scale: f32,
};

/// Batched embedding lookup over n_pos token positions. Distinct from
/// `EmbedLookupPush` (decode-style single-token); used by the
/// transformer-training forward pass.
pub const EmbedLookupBatchedPush = extern struct {
    dim: u32,
    n_pos: u32,
    scale: f32,
};

pub const MatmulPush = extern struct { m: u32, n: u32, k: u32 };

// ── ChatKernels: pre-built compute pipelines ──────────────────────

/// Bundle of compiled compute pipelines a forward pass needs. Built
/// once per `Forward` and re-used across every recorded layer.
pub const ChatKernels = struct {
    embed: pipeline.Kernel,
    rmsnorm: pipeline.Kernel,
    matmul: pipeline.Kernel,
    matmul_lm_head: pipeline.Kernel,
    rope: pipeline.Kernel,
    rope_partial: pipeline.Kernel,
    kv_write: pipeline.Kernel,
    scores: pipeline.Kernel,
    softmax: pipeline.Kernel,
    attn_out: pipeline.Kernel,
    add: pipeline.Kernel,
    geglu: pipeline.Kernel,

    pub fn init(
        ctx: *const vk.Context,
        precision: gpu_model.Precision,
        family: config_mod.Family,
    ) !ChatKernels {
        const matmul_spv: []align(4) const u8 = switch (precision) {
            .fp32_all => &shaders.matmul_nt_v2,
            .bf16_matmul => &shaders.matmul_nt_v2_bf16,
            .q4_0_matmul => &shaders.matmul_nt_v2_q4_0,
            .q4_k_matmul => &shaders.matmul_nt_v2_q4_k,
        };
        // LM head + embeddings stay fp32/bf16 even when layer matmuls
        // are quantised — argmax-shifting risk on logits, see
        // project_q4_k.md.
        const lm_head_spv: []align(4) const u8 = switch (precision) {
            .fp32_all => &shaders.matmul_nt_v2,
            .bf16_matmul, .q4_0_matmul, .q4_k_matmul => &shaders.matmul_nt_v2_bf16,
        };
        const embed_spv: []align(4) const u8 = switch (precision) {
            .fp32_all => &shaders.embed_lookup,
            .bf16_matmul, .q4_0_matmul, .q4_k_matmul => &shaders.embed_lookup_bf16,
        };
        const ffn_spv: []align(4) const u8 = switch (family.activation()) {
            .gelu => &shaders.geglu,
            .silu => &shaders.swiglu,
        };
        return .{
            .embed = try pipeline.Kernel.init(ctx, embed_spv, 2, @sizeOf(EmbedLookupPush)),
            .rmsnorm = try pipeline.Kernel.init(ctx, &shaders.rmsnorm, 3, @sizeOf(RmsnormPush)),
            .matmul = try pipeline.Kernel.init(ctx, matmul_spv, 3, @sizeOf(MatmulPush)),
            .matmul_lm_head = try pipeline.Kernel.init(ctx, lm_head_spv, 3, @sizeOf(MatmulPush)),
            .rope = try pipeline.Kernel.init(ctx, &shaders.rope, 2, @sizeOf(RopePush)),
            .rope_partial = try pipeline.Kernel.init(ctx, &shaders.rope_partial, 2, @sizeOf(RopePartialPush)),
            .kv_write = try pipeline.Kernel.init(ctx, &shaders.kv_write, 2, @sizeOf(KvWritePush)),
            .scores = try pipeline.Kernel.init(ctx, &shaders.attn_scores, 3, @sizeOf(AttnScoresPush)),
            .softmax = try pipeline.Kernel.init(ctx, &shaders.softmax, 2, @sizeOf(SoftmaxPush)),
            .attn_out = try pipeline.Kernel.init(ctx, &shaders.attn_output, 3, @sizeOf(AttnOutputPush)),
            .add = try pipeline.Kernel.init(ctx, &shaders.add_in_place, 2, @sizeOf(AddInPlacePush)),
            .geglu = try pipeline.Kernel.init(ctx, ffn_spv, 3, @sizeOf(GegluPush)),
        };
    }

    pub fn deinit(self: *ChatKernels) void {
        self.embed.deinit();
        self.rmsnorm.deinit();
        self.matmul.deinit();
        self.matmul_lm_head.deinit();
        self.rope.deinit();
        self.rope_partial.deinit();
        self.kv_write.deinit();
        self.scores.deinit();
        self.softmax.deinit();
        self.attn_out.deinit();
        self.add.deinit();
        self.geglu.deinit();
    }
};

// ── Per-step push computation ─────────────────────────────────────

pub const ForwardPushes = struct {
    rms_push: RmsnormPush,
    qkn_push: RmsnormPush,
    add_push: AddInPlacePush,
    rope_q_push: RopePush,
    rope_k_push: RopePush,
    rope_q_partial_push: RopePartialPush,
    rope_k_partial_push: RopePartialPush,
    use_partial_rope: bool,
    kv_write_push: KvWritePush,
    scores_push: AttnScoresPush,
    softmax_push: SoftmaxPush,
    attn_out_push: AttnOutputPush,
    geglu_push: GegluPush,
    n_pos: u32,
};

pub fn computeForwardPushes(
    cfg: config_mod.Config,
    sc: *const gpu_scratch.GpuScratch,
    pos: usize,
) ForwardPushes {
    const hidden: u32 = @intCast(cfg.hidden_size);
    const inter: u32 = @intCast(cfg.intermediate_size);
    const kv_dim: u32 = @intCast(cfg.num_key_value_heads * cfg.head_dim);
    const gemma_quirk: u32 = if (cfg.family == .gemma) 1 else 0;
    const heads_per_kv: u32 = @intCast(cfg.num_attention_heads / cfg.num_key_value_heads);
    const inv_sqrt_dim: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(cfg.head_dim)));
    const max_pos_u32: u32 = @intCast(sc.max_pos);
    const n_pos: u32 = @intCast(pos + 1);
    const head_dim_u32: u32 = @intCast(cfg.head_dim);
    const rotary_dim: u32 = @intFromFloat(@as(f32, @floatFromInt(head_dim_u32)) * cfg.partial_rotary_factor);
    const use_partial_rope: bool = cfg.partial_rotary_factor < 1.0;

    return .{
        .rms_push = .{ .dim = hidden, .eps = cfg.rms_norm_eps, .gemma_quirk = gemma_quirk },
        .qkn_push = .{ .dim = head_dim_u32, .eps = cfg.rms_norm_eps, .gemma_quirk = 0 },
        .add_push = .{ .n = hidden },
        .rope_q_push = .{
            .n_heads = @intCast(cfg.num_attention_heads),
            .head_dim = head_dim_u32,
            .pos = @intCast(pos),
            .theta_base = cfg.rope_theta,
        },
        .rope_k_push = .{
            .n_heads = @intCast(cfg.num_key_value_heads),
            .head_dim = head_dim_u32,
            .pos = @intCast(pos),
            .theta_base = cfg.rope_theta,
        },
        .rope_q_partial_push = .{
            .n_heads = @intCast(cfg.num_attention_heads),
            .head_dim = head_dim_u32,
            .rotary_dim = rotary_dim,
            .pos = @intCast(pos),
            .theta_base = cfg.rope_theta,
        },
        .rope_k_partial_push = .{
            .n_heads = @intCast(cfg.num_key_value_heads),
            .head_dim = head_dim_u32,
            .rotary_dim = rotary_dim,
            .pos = @intCast(pos),
            .theta_base = cfg.rope_theta,
        },
        .use_partial_rope = use_partial_rope,
        .kv_write_push = .{ .n = kv_dim, .dst_off = @intCast(pos * @as(usize, kv_dim)) },
        .scores_push = .{
            .n_heads = @intCast(cfg.num_attention_heads),
            .heads_per_kv = heads_per_kv,
            .head_dim = @intCast(cfg.head_dim),
            .n_pos = n_pos,
            .kv_stride = kv_dim,
            .scores_stride = max_pos_u32,
            .inv_sqrt_dim = inv_sqrt_dim,
        },
        .softmax_push = .{ .dim = n_pos, .stride = max_pos_u32 },
        .attn_out_push = .{
            .n_heads = @intCast(cfg.num_attention_heads),
            .heads_per_kv = heads_per_kv,
            .head_dim = @intCast(cfg.head_dim),
            .n_pos = n_pos,
            .kv_stride = kv_dim,
            .scores_stride = max_pos_u32,
        },
        .geglu_push = .{ .n = inter },
        .n_pos = n_pos,
    };
}

// ── Dispatch helpers ──────────────────────────────────────────────

pub fn recDispatch1D(
    rec: *recorder.Recorder,
    kern: *const pipeline.Kernel,
    bufs: []const *const buffer.Buffer,
    push: anytype,
    n: u32,
) !void {
    const local: u32 = 256;
    const groups: u32 = (n + local - 1) / local;
    try rec.dispatch(kern, bufs, push, groups, 1, 1);
}

pub fn recDispatchPerRow(
    rec: *recorder.Recorder,
    kern: *const pipeline.Kernel,
    bufs: []const *const buffer.Buffer,
    push: anytype,
    n_rows: u32,
) !void {
    try rec.dispatch(kern, bufs, push, n_rows, 1, 1);
}

pub fn recDispatchMatmul(
    rec: *recorder.Recorder,
    kern: *const pipeline.Kernel,
    bufs: []const *const buffer.Buffer,
    m: u32,
    n: u32,
    k: u32,
) !void {
    const push = MatmulPush{ .m = m, .n = n, .k = k };
    try rec.dispatch(kern, bufs, &push, m * n, 1, 1);
}

pub fn recDispatchRope(
    rec: *recorder.Recorder,
    kern: *const pipeline.Kernel,
    bufs: []const *const buffer.Buffer,
    push: *const RopePush,
    n_heads: usize,
    head_dim: usize,
) !void {
    const local: u32 = 256;
    const pairs: u32 = @intCast(n_heads * (head_dim / 2));
    const groups: u32 = (pairs + local - 1) / local;
    try rec.dispatch(kern, bufs, push, groups, 1, 1);
}

// ── Per-layer + full-step recording ───────────────────────────────

/// Record one transformer block's dispatches (input_layernorm → Q/K/V
/// → optional q_norm/k_norm → RoPE (full or partial) → KV write →
/// attention → o_proj → residual → post_attention_layernorm → gated
/// FFN → residual). Used by the full-forward `recordStep` AND
/// directly by chunk 7c's Session for frame-budget chunking.
///
/// `tq4_v` switches V into a TQ4-packed cache (asymmetric K=fp/V=TQ4).
/// `pos` is only consulted when `tq4_v` is non-null (for the V-pack
/// destination block index); pass `p.kv_write_push.dst_off /
/// p.kv_write_push.n` if you don't have it handy.
pub fn recordOneLayer(
    rec: *recorder.Recorder,
    sc: *const gpu_scratch.GpuScratch,
    gm: *const gpu_model.GpuModel,
    kv: *const gpu_scratch.GpuKvCache,
    cfg: config_mod.Config,
    k: *const ChatKernels,
    layer_idx: usize,
    pos: usize,
    p: *const ForwardPushes,
    tq4_v: ?Tq4VHooks,
) !void {
    const hidden: u32 = @intCast(cfg.hidden_size);
    const inter: u32 = @intCast(cfg.intermediate_size);
    const q_dim: u32 = @intCast(cfg.num_attention_heads * cfg.head_dim);
    const kv_dim: u32 = @intCast(cfg.num_key_value_heads * cfg.head_dim);

    const layer = &gm.layers[layer_idx];

    try recDispatchPerRow(rec, &k.rmsnorm, &.{ &sc.stream, &layer.input_layernorm, &sc.x_norm }, &p.rms_push, 1);

    try recDispatchMatmul(rec, &k.matmul, &.{ &sc.x_norm, &layer.q_proj.?, &sc.q }, 1, q_dim, hidden);
    try recDispatchMatmul(rec, &k.matmul, &.{ &sc.x_norm, &layer.k_proj.?, &sc.k }, 1, kv_dim, hidden);
    try recDispatchMatmul(rec, &k.matmul, &.{ &sc.x_norm, &layer.v_proj.?, &sc.v }, 1, kv_dim, hidden);

    if (layer.q_norm) |*qn| {
        try recDispatchPerRow(rec, &k.rmsnorm, &.{ &sc.q, qn, &sc.q }, &p.qkn_push, @intCast(cfg.num_attention_heads));
    }
    if (layer.k_norm) |*kn| {
        try recDispatchPerRow(rec, &k.rmsnorm, &.{ &sc.k, kn, &sc.k }, &p.qkn_push, @intCast(cfg.num_key_value_heads));
    }

    if (p.use_partial_rope) {
        try recDispatch1D(rec, &k.rope_partial, &.{ &sc.q, &sc.q_rot }, &p.rope_q_partial_push, @intCast(cfg.num_attention_heads * cfg.head_dim));
        try recDispatch1D(rec, &k.rope_partial, &.{ &sc.k, &sc.k_rot }, &p.rope_k_partial_push, @intCast(cfg.num_key_value_heads * cfg.head_dim));
    } else {
        try recDispatchRope(rec, &k.rope, &.{ &sc.q, &sc.q_rot }, &p.rope_q_push, cfg.num_attention_heads, cfg.head_dim);
        try recDispatchRope(rec, &k.rope, &.{ &sc.k, &sc.k_rot }, &p.rope_k_push, cfg.num_key_value_heads, cfg.head_dim);
    }

    const kv_layer = &kv.layers[layer_idx];
    try recDispatch1D(rec, &k.kv_write, &.{ &sc.k_rot, &kv_layer.k_cache }, &p.kv_write_push, kv_dim);

    if (tq4_v) |t| {
        const tq_layer = &t.cache.layers[layer_idx];
        const n_blocks: u32 = @intCast(t.cache.n_blocks_per_pos);
        const pack_push = Tq4PackPush{ .dst_block_idx = @intCast(pos * t.cache.n_blocks_per_pos) };
        try rec.dispatch(t.pack, &.{ &sc.v, &tq_layer.v_cache }, &pack_push, n_blocks, 1, 1);
    } else {
        try recDispatch1D(rec, &k.kv_write, &.{ &sc.v, &kv_layer.v_cache }, &p.kv_write_push, kv_dim);
    }

    try rec.dispatch(
        &k.scores,
        &.{ &sc.q_rot, &kv_layer.k_cache, &sc.scores },
        &p.scores_push,
        @as(u32, @intCast(cfg.num_attention_heads)) * p.n_pos,
        1,
        1,
    );
    try recDispatchPerRow(rec, &k.softmax, &.{ &sc.scores, &sc.scores }, &p.softmax_push, @intCast(cfg.num_attention_heads));

    const v_for_attn: *const buffer.Buffer = if (tq4_v) |t| blk: {
        const tq_layer = &t.cache.layers[layer_idx];
        const total_blocks: u32 = p.n_pos * @as(u32, @intCast(t.cache.n_blocks_per_pos));
        try rec.dispatch(t.unpack, &.{ &tq_layer.v_cache, &t.cache.dequant_v }, null, total_blocks, 1, 1);
        break :blk &t.cache.dequant_v;
    } else &kv_layer.v_cache;

    try rec.dispatch(
        &k.attn_out,
        &.{ &sc.scores, v_for_attn, &sc.head_out },
        &p.attn_out_push,
        @as(u32, @intCast(cfg.num_attention_heads)) * @as(u32, @intCast(cfg.head_dim)),
        1,
        1,
    );

    try recDispatchMatmul(rec, &k.matmul, &.{ &sc.head_out, &layer.o_proj.?, &sc.attn_out }, 1, hidden, q_dim);
    try recDispatch1D(rec, &k.add, &.{ &sc.stream, &sc.attn_out }, &p.add_push, hidden);

    try recDispatchPerRow(rec, &k.rmsnorm, &.{ &sc.stream, &layer.post_attention_layernorm, &sc.mid_norm }, &p.rms_push, 1);

    try recDispatchMatmul(rec, &k.matmul, &.{ &sc.mid_norm, &layer.gate_proj, &sc.gate }, 1, inter, hidden);
    try recDispatchMatmul(rec, &k.matmul, &.{ &sc.mid_norm, &layer.up_proj, &sc.up }, 1, inter, hidden);
    try recDispatch1D(rec, &k.geglu, &.{ &sc.gate, &sc.up, &sc.fused }, &p.geglu_push, inter);
    try recDispatchMatmul(rec, &k.matmul, &.{ &sc.fused, &layer.down_proj, &sc.ffn_out }, 1, hidden, inter);

    try recDispatch1D(rec, &k.add, &.{ &sc.stream, &sc.ffn_out }, &p.add_push, hidden);
}

/// Embedding lookup → scratch.stream. Called once per token at the
/// start of a forward.
pub fn recordEmbedding(
    rec: *recorder.Recorder,
    sc: *const gpu_scratch.GpuScratch,
    gm: *const gpu_model.GpuModel,
    cfg: config_mod.Config,
    k: *const ChatKernels,
    token_id: u32,
) !void {
    const hidden: u32 = @intCast(cfg.hidden_size);
    const embed_push = EmbedLookupPush{
        .token_id = token_id,
        .dim = hidden,
        .scale = if (cfg.family.embedScalesByDim()) @sqrt(@as(f32, @floatFromInt(hidden))) else 1.0,
    };
    try recDispatch1D(rec, &k.embed, &.{ &gm.embed_tokens, &sc.stream }, &embed_push, hidden);
}

/// One-call full forward: embed → all layers → optional sample-step.
/// Equivalent to `Forward.recordStep` but free-standing so callers
/// that already have an owned `ChatKernels` (e.g. valkyr's CLI) don't
/// need to wrap it in a `Forward`. Identical dispatch order, so
/// generation output is bit-identical to `Forward.recordStep`.
pub fn recordForwardStep(
    rec: *recorder.Recorder,
    sc: *const gpu_scratch.GpuScratch,
    gm: *const gpu_model.GpuModel,
    kv: *const gpu_scratch.GpuKvCache,
    cfg: config_mod.Config,
    k: *const ChatKernels,
    pos: usize,
    token_id: u32,
    tq4_v: ?Tq4VHooks,
    compute_logits: bool,
) !void {
    const pushes = computeForwardPushes(cfg, sc, pos);
    try recordEmbedding(rec, sc, gm, cfg, k, token_id);
    for (0..cfg.num_hidden_layers) |layer_idx| {
        try recordOneLayer(rec, sc, gm, kv, cfg, k, layer_idx, pos, &pushes, tq4_v);
    }
    if (compute_logits) {
        try recordSampleStep(rec, sc, gm, cfg, k, &pushes);
    }
}

/// Final norm + LM head matmul → scratch.logits. Skip on prefill
/// tokens (only the LAST prompt token needs logits to sample).
pub fn recordSampleStep(
    rec: *recorder.Recorder,
    sc: *const gpu_scratch.GpuScratch,
    gm: *const gpu_model.GpuModel,
    cfg: config_mod.Config,
    k: *const ChatKernels,
    p: *const ForwardPushes,
) !void {
    const hidden: u32 = @intCast(cfg.hidden_size);
    const vocab: u32 = @intCast(cfg.vocab_size);
    try recDispatchPerRow(rec, &k.rmsnorm, &.{ &sc.stream, &gm.final_norm, &sc.final_norm_out }, &p.rms_push, 1);
    try recDispatchMatmul(rec, &k.matmul_lm_head, &.{ &sc.final_norm_out, &gm.lm_head, &sc.logits }, 1, vocab, hidden);
}

// ── High-level Forward facade ─────────────────────────────────────

/// Convenience wrapper: builds the kernel set once and exposes a
/// one-call recordStep that does embedding + all layers + (optional)
/// sample-step. Layout of dispatches into the recorder is identical
/// to valkyr's CLI chat path.
///
/// For frame-budget chunking, callers should bypass `recordStep` and
/// drive `recordEmbedding` + `recordOneLayer` + `recordSampleStep`
/// themselves at their own pace — that's what chunk 7c's Session does.
pub const Forward = struct {
    kernels: ChatKernels,
    cfg: config_mod.Config,

    pub fn init(
        ctx: *const vk.Context,
        gm: *const gpu_model.GpuModel,
    ) !Forward {
        return .{
            .kernels = try ChatKernels.init(ctx, gm.precision, gm.config.family),
            .cfg = gm.config,
        };
    }

    pub fn deinit(self: *Forward) void {
        self.kernels.deinit();
    }

    pub fn recordStep(
        self: *const Forward,
        rec: *recorder.Recorder,
        sc: *const gpu_scratch.GpuScratch,
        gm: *const gpu_model.GpuModel,
        kv: *const gpu_scratch.GpuKvCache,
        pos: usize,
        token_id: u32,
        compute_logits: bool,
    ) !void {
        const pushes = computeForwardPushes(self.cfg, sc, pos);
        try recordEmbedding(rec, sc, gm, self.cfg, &self.kernels, token_id);
        for (0..self.cfg.num_hidden_layers) |layer_idx| {
            try recordOneLayer(rec, sc, gm, kv, self.cfg, &self.kernels, layer_idx, pos, &pushes, null);
        }
        if (compute_logits) {
            try recordSampleStep(rec, sc, gm, self.cfg, &self.kernels, &pushes);
        }
    }
};

// ── Sampling ──────────────────────────────────────────────────────

/// Greedy argmax over a logits vector. Returns the token id with the
/// largest logit. Ties are broken toward lower index (the first max
/// wins) — this matches valkyr's CLI sampler so library-driven
/// generation reproduces CLI-driven generation bit-for-bit.
pub fn sampleArgmax(logits: []const f32) u32 {
    var best: u32 = 0;
    var best_v: f32 = logits[0];
    var i: u32 = 1;
    while (i < logits.len) : (i += 1) {
        if (logits[i] > best_v) {
            best_v = logits[i];
            best = i;
        }
    }
    return best;
}
