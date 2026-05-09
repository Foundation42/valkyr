//! TransformerTrainerRunner — multi-layer decoder fine-tune on the GPU.
//!
//! Wraps the chunk-8c-α-3 forward + softmax-CE loss-grad + backward +
//! Adam pipeline into a single object. Pipelines, weight buffers,
//! Adam m/v state, per-layer activations + grads, and shared scratch
//! all live for the lifetime of the Runner. One `step(token_ids,
//! target_one_hot)` runs three recorder cycles (forward+backward,
//! embedding_backward after dE_embed fillZero, Adam), plus a host-side
//! row-reduce of the RMSNorm dw_partials.
//!
//! Topology is: embed → N · {rms→Q/K/V→SDPA→O+residual→rms→FF1→ReLU→
//! FF2+residual} → final_rms → lm_head. Same shape the chunk-8c CPU
//! oracle (`cpu/train_decoder.zig`) validates. Init weights are taken
//! by-borrow from the caller; the Runner uploads its own copies.
//!
//! Built for chunk 8c-β-1: lift the ~1k-line `runDecoderStackTrainGpuSmoke`
//! buffer mountain into a reusable type so β-2 can instantiate it at
//! real Qwen dims without copying everything again.

const std = @import("std");
const util = @import("../util.zig");
const vk = @import("../gpu/vk.zig");
const buffer = @import("../gpu/buffer.zig");
const pipeline = @import("../gpu/pipeline.zig");
const recorder_mod = @import("../gpu/recorder.zig");
const runtime = @import("../runtime.zig");
const runtime_hybrid = @import("../runtime_hybrid.zig");
const lora_helpers = @import("lora.zig");
const shaders = @import("shaders");

pub const Config = struct {
    dim: u32,
    n_heads: u32,
    n_kv_heads: u32,
    head_dim: u32,
    ff_dim: u32,
    n_pos: u32,
    n_layers: u32,
    vocab_size: u32,
    rms_eps: f32 = 1e-5,
    causal: bool = true,
    /// RoPE rotation dim. 0 disables RoPE; `head_dim` gives full RoPE;
    /// smaller gives partial (Qwen3.5-style). Forward-pass RoPE applies
    /// to Q + K post-projection; backward inverts the rotation on the
    /// gradients before the linear-backward of W_q / W_k.
    rotary_dim: u32 = 0,
    rope_theta: f32 = 10_000.0,
    /// Per-head Q/K RMSNorm (Qwen3 architectural detail). When true,
    /// RMSNorm is applied to Q + K after projection but before RoPE.
    /// Two new learnable [head_dim] gains per layer (`w_q_norm`,
    /// `w_k_norm`).
    qk_norm: bool = false,
    /// Adam learning rate. Sensible default for the toy 8c-α-3 demo:
    /// 1e-2. Mutable across steps; write `runner.lr` between ticks
    /// for schedules.
    lr: f32 = 1e-2,
    beta1: f32 = 0.9,
    beta2: f32 = 0.999,
    eps_adam: f32 = 1e-8,

    /// LoRA target bitmask (A4-3). Each bit selects one projection
    /// to be replaced by `y = x·Wᵀ + (α/r)·(x·Aᵀ)·Bᵀ` with W frozen
    /// and (A, B) trained. Use `LoraTarget.q | LoraTarget.k | ...` to
    /// build, or `LoraTarget.all` / `LoraTarget.all_attn` shortcuts.
    /// 0 ⇒ no LoRA (default), every dense matmul stays in the plain
    /// path. The same `lora_rank` / `lora_alpha` / `lora_lr_b_scale`
    /// apply to every enabled projection (per-projection rank is on
    /// the A4-3+ board if a workload needs it).
    lora_targets: u32 = 0,
    lora_rank: u32 = 0,
    lora_alpha: f32 = 0.0,
    /// LoRA+ differential learning rate for B (Chronicals Theorem 1).
    /// Effective Adam lr for B is `lr · lora_lr_b_scale`; 1.0 is
    /// classical LoRA, ~16 is the Chronicals default. Untouched when
    /// `lora_targets = 0`.
    lora_lr_b_scale: f32 = 1.0,
};

/// Bitmask constants for `Config.lora_targets`. Use bitwise-or to
/// compose multiple projections; e.g. `LoraTarget.q | LoraTarget.v`
/// for "LoRA on Q + V only".
pub const LoraTarget = struct {
    pub const q: u32 = 1 << 0;
    pub const k: u32 = 1 << 1;
    pub const v: u32 = 1 << 2;
    pub const o: u32 = 1 << 3;
    pub const gate: u32 = 1 << 4;
    pub const up: u32 = 1 << 5;
    pub const down: u32 = 1 << 6;
    /// All four attention projections (Q + K + V + O). The most common
    /// LoRA config in the literature.
    pub const all_attn: u32 = q | k | v | o;
    /// FFN-only LoRA: gate + up + down.
    pub const all_ffn: u32 = gate | up | down;
    /// Full LoRA on every dense matmul in the layer (max optimizer-state
    /// reduction; ~30-40% step-time penalty on Qwen3-0.6B).
    pub const all: u32 = all_attn | all_ffn;
};

/// Internal enum mirroring `LoraTarget` bits, used for indexing into
/// `LoraState.slots`. The bit layout `1 << @intFromEnum(p)` matches
/// the public `LoraTarget` constants exactly.
const Proj = enum(u3) { q = 0, k, v, o, gate, up, down };
const proj_count: comptime_int = 7;

fn projBit(p: Proj) u32 {
    return @as(u32, 1) << @intFromEnum(p);
}
fn projEnabled(targets: u32, p: Proj) bool {
    return (targets & projBit(p)) != 0;
}

pub const LayerWeights = struct {
    w_n1: []const f32, // [dim]
    w_q: []const f32, // [n_heads*head_dim, dim]
    w_k: []const f32, // [n_kv_heads*head_dim, dim]
    w_v: []const f32, // [n_kv_heads*head_dim, dim]
    w_o: []const f32, // [dim, n_heads*head_dim]
    w_n2: []const f32, // [dim]
    w_gate: []const f32, // [ff_dim, dim]
    w_up: []const f32, // [ff_dim, dim]
    w_down: []const f32, // [dim, ff_dim]
    /// [head_dim] when Config.qk_norm is true, else empty.
    w_q_norm: []const f32 = &.{},
    /// [head_dim] when Config.qk_norm is true, else empty.
    w_k_norm: []const f32 = &.{},
};

pub const InitWeights = struct {
    embed: []const f32, // [vocab, dim]
    final_norm: []const f32, // [dim]
    lm_head: []const f32, // [vocab, dim]
    layers: []const LayerWeights, // length cfg.n_layers

    /// Per-layer LoRA initial weights, one slice per target projection.
    /// When empty for an enabled projection, the Runner default-init's
    /// A from a fixed-seed scaled normal (σ = 1/sqrt(rank)) and B from
    /// zeros. When non-empty, length must equal cfg.n_layers and each
    /// entry's shapes must match `[r, K_proj]` for A and `[N_proj, r]`
    /// for B. Disabled projections (per `cfg.lora_targets`) ignore the
    /// corresponding slice entirely — leaving it empty is fine even
    /// when `cfg.lora_targets = 0`.
    lora_q: []const LoraInitWeights = &.{},
    lora_k: []const LoraInitWeights = &.{},
    lora_v: []const LoraInitWeights = &.{},
    lora_o: []const LoraInitWeights = &.{},
    lora_gate: []const LoraInitWeights = &.{},
    lora_up: []const LoraInitWeights = &.{},
    lora_down: []const LoraInitWeights = &.{},
};

pub const LoraInitWeights = struct {
    a: []const f32, // [r, K_proj]
    b: []const f32, // [N_proj, r]
};

// ── Checkpoint format (chunk 8c-β-6b) ────────────────────────────────
// File layout:
//   [CheckpointHeader: 16 bytes]
//   [Config: @sizeOf(Config) bytes, std.mem.asBytes(&cfg)]
//   [body: positional fp32 blocks per `Runner.collectBlobs`]
// Body order: stack-level (embed, final_norm, lm_head) × (param, m, v),
// then per-layer × N: (w_n1..w_down + optional w_q_norm/w_k_norm) ×
// (param, m, v). Sizes are implied by the Config — no per-block names.

const checkpoint_magic: [4]u8 = .{ 'V', 'K', 'P', 'T' };
const checkpoint_version: u32 = 1;
/// `.lvkpt` (A4-4) — LoRA-only checkpoint. Stores per-projection-per-
/// layer A / B / Adam m/v for every enabled target plus the cfg
/// snapshot + step_t. Tiny compared to `.vkpt` (kilobytes-megabytes vs
/// gigabytes) — base W stays in safetensors. Magic intentionally
/// distinct from `.vkpt`'s "VKPT" so the loader fails fast on swap.
const lora_checkpoint_magic: [4]u8 = .{ 'V', 'L', 'K', 'P' };
const lora_checkpoint_version: u32 = 1;
/// Buffered-IO chunk size for checkpoint save/load. 4 MiB is large
/// enough to amortise syscall overhead across the many small per-layer
/// blobs and small enough that the staging-side memcpy still fits in
/// L2 on most CPUs.
const checkpoint_io_buf_bytes: comptime_int = 4 * 1024 * 1024;

const CheckpointHeader = extern struct {
    magic: [4]u8,
    version: u32,
    step_t: u32,
    cfg_size: u32,
};

const Blob = struct {
    buf: *buffer.Buffer,
    numel: usize,
};

/// Compare two Configs by structural fields only — leaves lr / beta /
/// eps_adam to the loader. Padding bytes inside Config are unspecified
/// (it isn't extern), so byte-equality of asBytes(cfg) wouldn't be safe;
/// field-by-field is.
fn cfgShapeMatches(a: Config, b: Config) bool {
    return a.dim == b.dim and a.n_heads == b.n_heads and a.n_kv_heads == b.n_kv_heads and
        a.head_dim == b.head_dim and a.ff_dim == b.ff_dim and a.n_pos == b.n_pos and
        a.n_layers == b.n_layers and a.vocab_size == b.vocab_size and
        a.rms_eps == b.rms_eps and a.causal == b.causal and
        a.rotary_dim == b.rotary_dim and a.rope_theta == b.rope_theta and
        a.qk_norm == b.qk_norm and
        a.lora_targets == b.lora_targets and a.lora_rank == b.lora_rank;
}

pub const Runner = struct {
    ctx: *const vk.Context,
    allocator: std.mem.Allocator,
    cfg: Config,

    /// Adam timestep counter. Starts at 1 (first step uses t=1 for bias
    /// correction). Auto-increments at the end of each `step()`.
    step_t: u32,

    // ── Pipelines (one per shader; reused across all dispatches).
    k_embed: pipeline.Kernel,
    k_rms: pipeline.Kernel,
    k_rms_bw: pipeline.Kernel,
    k_matmul: pipeline.Kernel,
    k_attn_scores: pipeline.Kernel,
    k_attn_output: pipeline.Kernel,
    k_softmax: pipeline.Kernel,
    k_softmax_bw: pipeline.Kernel,
    /// FlashAttention forward (single fused kernel). Used by both
    /// `Runner.step` (training, with `write_lse = 1` so the FA-2
    /// backward can recompute the softmax inline) and
    /// `forwardLogits` (inference, `write_lse = 0`) when
    /// `attn_use_fa == true`. Capped at head_dim ≤ 128 (shader
    /// compile-time limit); falls back to the 3-pass chain for
    /// larger heads.
    k_fa_forward: pipeline.Kernel,
    /// FlashAttention-2 backward chain — replaces the 5-kernel 3-pass
    /// (`attn_dattn → attn_dv → softmax_bw → attn_dq → attn_dk`) when
    /// `attn_use_fa == true`. Phase 1 reduces D[q, h] = Σ_d O · dO,
    /// phase 2 accumulates dQ tile-on-K, phase 3 accumulates dK + dV
    /// tile-on-Q with the GQA fold inside.
    k_fa_bw_d: pipeline.Kernel,
    k_fa_bw_dq: pipeline.Kernel,
    k_fa_bw_dkv: pipeline.Kernel,
    k_attn_dattn: pipeline.Kernel,
    k_attn_dv: pipeline.Kernel,
    k_attn_dq: pipeline.Kernel,
    k_attn_dk: pipeline.Kernel,
    k_swiglu_fwd: pipeline.Kernel,
    k_swiglu_bw: pipeline.Kernel,
    // Fused QK-RoPE: one dispatch processes both Q (n_q_heads) and K
    // (n_kv_heads) — replaces a back-to-back rope_partial_batched pair.
    // Saves command-processor launch overhead and lets cos/sin be
    // co-issued for adjacent (Q, K) head positions sharing the same
    // (p, i) — see shaders/qk_rope_*_batched.comp.
    k_rope_fwd: pipeline.Kernel,
    k_rope_bw: pipeline.Kernel,
    k_vec_add: pipeline.Kernel,
    k_add: pipeline.Kernel,
    k_lin_dx: pipeline.Kernel,
    k_lin_dw: pipeline.Kernel,
    // CCE forward + backward replace the
    // matmul(lm_head) → softmax_ce_loss_grad_batched_v2 → linear_backward_dx + linear_backward_dw
    // chain. Memory wise: drops the [n_pos, vocab] logits + d_logits + one-hot
    // target tensors, replacing them with a [n_pos] f32 lse cache + [n_pos]
    // u32 target id buffer.
    k_cce_forward: pipeline.Kernel,
    k_cce_backward_dh: pipeline.Kernel,
    k_cce_backward_dw: pipeline.Kernel,
    k_embed_bw: pipeline.Kernel,
    k_adam: pipeline.Kernel,

    // ── Recorder (re-used across the 3 phases of each step).
    rec: recorder_mod.Recorder,

    // ── Stack-level weight buffers (DEVICE_LOCAL, mutated by Adam).
    buf_w_embed: buffer.Buffer,
    buf_w_final_norm: buffer.Buffer,
    buf_w_lm_head: buffer.Buffer,

    // ── Per-step host-uploadable inputs.
    buf_token_ids: buffer.Buffer, // dynamic, [n_pos] u32
    buf_target_ids: buffer.Buffer, // dynamic, [n_pos] u32 — CCE replacement for target_oh

    // ── Stack-level activations + gradients.
    buf_x_emb: buffer.Buffer,
    buf_final_norm_out: buffer.Buffer,
    // `buf_logits` is kept for `forwardLogits` (inference / CE measurement)
    // but is *not* allocated/written in step()'s gradient path — CCE fuses
    // the LM-head matmul into the loss kernel so logits never materialize.
    buf_logits: buffer.Buffer,
    buf_lse: buffer.Buffer, // device-only, [n_pos] f32 — bridges cce_forward → cce_backward_*
    buf_loss_per_row: buffer.Buffer, // device-only, [n_pos] f32 — host averages for total loss
    buf_d_final_norm_out: buffer.Buffer,
    buf_d_last_y: buffer.Buffer,
    buf_dw_lm_head: buffer.Buffer,
    buf_dw_final_norm_partial: buffer.Buffer, // [n_pos, dim] partial; row-reduce host-side
    buf_dw_final_norm: buffer.Buffer, // dynamic, [dim] reduced
    buf_dE_embed: buffer.Buffer,

    // ── Stack-level Adam m/v (one each per stack-level weight).
    buf_m_embed: buffer.Buffer,
    buf_v_embed: buffer.Buffer,
    buf_m_final_norm: buffer.Buffer,
    buf_v_final_norm: buffer.Buffer,
    buf_m_lm_head: buffer.Buffer,
    buf_v_lm_head: buffer.Buffer,

    // ── Per-layer arrays (length cfg.n_layers).
    buf_w_n1: []buffer.Buffer,
    buf_w_q: []buffer.Buffer,
    buf_w_k: []buffer.Buffer,
    buf_w_v: []buffer.Buffer,
    buf_w_o: []buffer.Buffer,
    buf_w_n2: []buffer.Buffer,
    buf_w_gate: []buffer.Buffer,
    buf_w_up: []buffer.Buffer,
    buf_w_down: []buffer.Buffer,

    buf_n1: []buffer.Buffer,
    buf_q: []buffer.Buffer,
    buf_k: []buffer.Buffer,
    buf_v: []buffer.Buffer,
    buf_attn: []buffer.Buffer,
    buf_attn_out: []buffer.Buffer,
    buf_mid: []buffer.Buffer,
    buf_n2: []buffer.Buffer,
    buf_pre_gate: []buffer.Buffer,
    buf_up: []buffer.Buffer,
    buf_gated: []buffer.Buffer,
    buf_y: []buffer.Buffer,

    // Q/K-norm per-layer buffers. Empty slices (length 0) when
    // `cfg.qk_norm = false` — the recordLayer{Forward,Backward} paths
    // skip the rmsnorm dispatches entirely.
    buf_w_q_norm: []buffer.Buffer,
    buf_w_k_norm: []buffer.Buffer,
    buf_q_pre_norm: []buffer.Buffer, // saved per-layer for rmsnorm_bw input
    buf_k_pre_norm: []buffer.Buffer,
    buf_dw_q_norm_partial: []buffer.Buffer, // [n_pos*n_heads*head_dim] per layer
    buf_dw_k_norm_partial: []buffer.Buffer,
    buf_dw_q_norm: []buffer.Buffer, // dynamic, [head_dim]
    buf_dw_k_norm: []buffer.Buffer,
    buf_m_q_norm: []buffer.Buffer,
    buf_v_q_norm: []buffer.Buffer,
    buf_m_k_norm: []buffer.Buffer,
    buf_v_k_norm: []buffer.Buffer,

    buf_dw_n1_partial: []buffer.Buffer,
    buf_dw_q: []buffer.Buffer,
    buf_dw_k: []buffer.Buffer,
    buf_dw_v: []buffer.Buffer,
    buf_dw_o: []buffer.Buffer,
    buf_dw_n2_partial: []buffer.Buffer,
    buf_dw_gate: []buffer.Buffer,
    buf_dw_up: []buffer.Buffer,
    buf_dw_down: []buffer.Buffer,

    // Dynamic; host-reduced from the corresponding _partial each step.
    buf_dw_n1: []buffer.Buffer,
    buf_dw_n2: []buffer.Buffer,

    buf_m_n1: []buffer.Buffer,
    buf_v_n1: []buffer.Buffer,
    buf_m_q: []buffer.Buffer,
    buf_v_q: []buffer.Buffer,
    buf_m_k: []buffer.Buffer,
    buf_v_k: []buffer.Buffer,
    buf_m_v: []buffer.Buffer,
    buf_v_v: []buffer.Buffer,
    buf_m_o: []buffer.Buffer,
    buf_v_o: []buffer.Buffer,
    buf_m_n2: []buffer.Buffer,
    buf_v_n2: []buffer.Buffer,
    buf_m_gate: []buffer.Buffer,
    buf_v_gate: []buffer.Buffer,
    buf_m_up: []buffer.Buffer,
    buf_v_up: []buffer.Buffer,
    buf_m_down: []buffer.Buffer,
    buf_v_down: []buffer.Buffer,

    /// Per-layer d_x_in: written by layer L's backward, consumed as
    /// d_y_in by layer L-1's backward (or fed to embedding_backward
    /// when L == 0).
    buf_d_x_in: []buffer.Buffer,

    // ── Shared scratch (forward + backward).
    sc_scores: buffer.Buffer,
    /// Per-layer FA forward LSE buffer [n_pos × n_heads]. Populated
    /// by forward (`write_lse = 1` in training so the FA-2 backward
    /// can recompute softmax inline; `write_lse = 0` for inference).
    /// Per-layer because Runner.step records every forward layer
    /// before any backward — a single buffer would be overwritten
    /// by later layers.
    buf_fa_lse: []buffer.Buffer,
    /// Per-layer FA backward phase-1 output [n_pos × n_heads]. Holds
    /// `D[q, h] = Σ_d O · dO`; consumed by phases 2 and 3 of the
    /// same layer's backward.
    buf_fa_d: []buffer.Buffer,
    sc_o: buffer.Buffer,
    sc_ff_out: buffer.Buffer,
    sc_q_pre: buffer.Buffer, // pre-RoPE Q (matmul/rmsnorm output, RoPE input)
    sc_k_pre: buffer.Buffer, // pre-RoPE K
    sc_dQ_pre: buffer.Buffer, // pre-RoPE dQ (RoPE-bw output, lin-bw or rms-bw input)
    sc_dK_pre: buffer.Buffer, // pre-RoPE dK
    sc_dQ_pre_norm: buffer.Buffer, // pre-norm dQ (rms-bw output, lin-bw input). Allocated empty when qk_norm=false.
    sc_dK_pre_norm: buffer.Buffer,
    sc_d_gated: buffer.Buffer,
    sc_d_pre_gate: buffer.Buffer,
    sc_d_up_grad: buffer.Buffer,
    sc_d_n2: buffer.Buffer,
    sc_d_n2_up: buffer.Buffer,
    sc_d_mid_norm: buffer.Buffer,
    sc_d_attn_out: buffer.Buffer,
    sc_d_attn: buffer.Buffer,
    sc_d_scores: buffer.Buffer,
    sc_dQ: buffer.Buffer,
    sc_dK: buffer.Buffer,
    sc_dV: buffer.Buffer,
    sc_d_n1: buffer.Buffer,
    sc_d_n1_k: buffer.Buffer,
    sc_d_n1_v: buffer.Buffer,

    // ── Host-side scratch for the dw_partial row-reduce.
    dw_partial_host: []f32, // [n_pos, dim] readback target
    dw_reduced: []f32, // [dim] reduced result, then update()'d into the dynamic buf

    // ── Cached push constants. All depend only on Config so we build
    //    them once in init() and reuse forever. Stored verbatim by value
    //    (not pointers — no GC concerns since we own them).
    push_embed: runtime.EmbedLookupBatchedPush,
    push_rms: runtime.RmsnormPush,
    push_n_pos_dim: runtime.AddInPlacePush,
    push_swiglu: runtime.SwigluPush,
    push_softmax: runtime.SoftmaxPush,
    push_cce: runtime.CceForwardPush, // shape struct shared by cce_forward + cce_backward_{dh,dw}
    push_mm_q: runtime.MatmulPush,
    push_mm_k: runtime.MatmulPush,
    push_mm_v: runtime.MatmulPush,
    push_mm_o: runtime.MatmulPush,
    push_mm_gate: runtime.MatmulPush,
    push_mm_up: runtime.MatmulPush,
    push_mm_down: runtime.MatmulPush,
    push_mm_lm_head: runtime.MatmulPush,
    push_attn_scores: runtime.AttnScoresTrainPush,
    push_attn_output: runtime.AttnOutputTrainPush,
    /// FA forward push for inference (`forwardLogits` /
    /// `forward_only=true`); `write_lse = 0`.
    push_fa_forward: runtime.FaForwardPush,
    /// FA forward push for training (`step` / `forward_only=false`);
    /// `write_lse = 1` so phase-1/2/3 backward can recompute softmax
    /// from saved LSE.
    push_fa_forward_train: runtime.FaForwardPush,
    push_fa_bw_d: runtime.FaBwDPush,
    push_fa_bw_dq: runtime.FaBwDqPush,
    push_fa_bw_dkv: runtime.FaBwDkvPush,
    /// True when `head_dim ≤ HEAD_DIM_MAX` (128 — shader compile-time
    /// cap). When false, the FA path falls back to the 3-pass chain
    /// even in `forwardLogits`. Computed once at init from `cfg`.
    attn_use_fa: bool,
    push_rope_qk: runtime.QkRopeBatchedPush,
    push_rms_qk: runtime.RmsnormPush, // dim = head_dim; same for q-norm + k-norm
    push_lin_lm_head: runtime.LinearBatchedPush,
    push_lin_down: runtime.LinearBatchedPush,
    push_lin_gate: runtime.LinearBatchedPush,
    push_lin_up: runtime.LinearBatchedPush,
    push_lin_o: runtime.LinearBatchedPush,
    push_lin_q: runtime.LinearBatchedPush,
    push_lin_k: runtime.LinearBatchedPush,
    push_lin_v: runtime.LinearBatchedPush,
    push_dattn: runtime.AttnBackwardDattnPush,
    push_dv: runtime.AttnBackwardDvPush,
    push_dq: runtime.AttnBackwardDqPush,
    push_dk: runtime.AttnBackwardDkPush,
    push_embed_bw: runtime.EmbeddingBackwardPush,

    /// Optional LoRA state (A4-2 → A4-3). Populated iff
    /// `cfg.lora_targets != 0`. Owns the only LoRA-specific kernel
    /// (k_scale), per-projection A/B adapters + Adam m/v + dW, the
    /// per-projection [n_pos, r] intermediate cache that bridges LoRA
    /// forward → backward, and the shared scratch buffers (sized to
    /// the largest enabled projection). All other kernels (matmul,
    /// lin_dx, lin_dw, add_in_place) are reused from the Runner.
    lora: ?LoraState,

    pub fn init(
        allocator: std.mem.Allocator,
        ctx: *const vk.Context,
        cfg: Config,
        weights: InitWeights,
    ) !Runner {
        // Validation: weight slice lengths must match cfg.
        const dim: usize = @intCast(cfg.dim);
        const vocab: usize = @intCast(cfg.vocab_size);
        const q_dim: usize = @intCast(cfg.n_heads * cfg.head_dim);
        const kv_dim: usize = @intCast(cfg.n_kv_heads * cfg.head_dim);
        const ff_dim: usize = @intCast(cfg.ff_dim);
        const n_pos: usize = @intCast(cfg.n_pos);
        const n_layers: usize = @intCast(cfg.n_layers);

        if (weights.embed.len != vocab * dim) return error.EmbedShape;
        if (weights.final_norm.len != dim) return error.FinalNormShape;
        if (weights.lm_head.len != vocab * dim) return error.LmHeadShape;
        if (weights.layers.len != n_layers) return error.LayerCount;
        for (weights.layers) |lw| {
            if (lw.w_n1.len != dim) return error.LayerN1Shape;
            if (lw.w_q.len != q_dim * dim) return error.LayerQShape;
            if (lw.w_k.len != kv_dim * dim) return error.LayerKShape;
            if (lw.w_v.len != kv_dim * dim) return error.LayerVShape;
            if (lw.w_o.len != dim * q_dim) return error.LayerOShape;
            if (lw.w_n2.len != dim) return error.LayerN2Shape;
            if (lw.w_gate.len != ff_dim * dim) return error.LayerGateShape;
            if (lw.w_up.len != ff_dim * dim) return error.LayerUpShape;
            if (lw.w_down.len != dim * ff_dim) return error.LayerDownShape;
            if (cfg.qk_norm) {
                if (lw.w_q_norm.len != cfg.head_dim) return error.LayerQNormShape;
                if (lw.w_k_norm.len != cfg.head_dim) return error.LayerKNormShape;
            }
        }

        const f32sz = @sizeOf(f32);
        const inv_sqrt_d: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(cfg.head_dim)));
        const heads_per_kv: u32 = cfg.n_heads / cfg.n_kv_heads;
        const scores_total: usize = n_pos * @as(usize, @intCast(cfg.n_heads)) * n_pos;

        // ── Pipelines.
        var k_embed = try pipeline.Kernel.init(ctx, &shaders.embed_lookup_batched, 3, @sizeOf(runtime.EmbedLookupBatchedPush));
        errdefer k_embed.deinit();
        var k_rms = try pipeline.Kernel.init(ctx, &shaders.rmsnorm, 3, @sizeOf(runtime.RmsnormPush));
        errdefer k_rms.deinit();
        var k_rms_bw = try pipeline.Kernel.init(ctx, &shaders.rmsnorm_backward, 5, @sizeOf(runtime.RmsnormPush));
        errdefer k_rms_bw.deinit();
        var k_matmul = try pipeline.Kernel.init(ctx, &shaders.matmul_nt_v2, 3, @sizeOf(runtime.MatmulPush));
        errdefer k_matmul.deinit();
        var k_attn_scores = try pipeline.Kernel.init(ctx, &shaders.attn_scores_train, 3, @sizeOf(runtime.AttnScoresTrainPush));
        errdefer k_attn_scores.deinit();
        var k_attn_output = try pipeline.Kernel.init(ctx, &shaders.attn_output_train, 3, @sizeOf(runtime.AttnOutputTrainPush));
        errdefer k_attn_output.deinit();
        var k_softmax = try pipeline.Kernel.init(ctx, &shaders.softmax, 2, @sizeOf(runtime.SoftmaxPush));
        errdefer k_softmax.deinit();
        var k_softmax_bw = try pipeline.Kernel.init(ctx, &shaders.softmax_backward, 3, @sizeOf(runtime.SoftmaxPush));
        errdefer k_softmax_bw.deinit();
        // FlashAttention shader picks: d=128 default, _d256 variant when
        // cfg.head_dim > 128 (Qwen3.5 family). fa_bw_d has no head_dim-
        // sized shared mem so it shares the d=128 build at d=256.
        const head_dim_u32: u32 = @intCast(cfg.head_dim);
        var k_fa_forward = try pipeline.Kernel.init(ctx, runtime.faForwardSpv(head_dim_u32), 5, @sizeOf(runtime.FaForwardPush));
        errdefer k_fa_forward.deinit();
        var k_fa_bw_d = try pipeline.Kernel.init(ctx, &shaders.fa_bw_d, 3, @sizeOf(runtime.FaBwDPush));
        errdefer k_fa_bw_d.deinit();
        var k_fa_bw_dq = try pipeline.Kernel.init(ctx, runtime.faBwDqSpv(head_dim_u32), 7, @sizeOf(runtime.FaBwDqPush));
        errdefer k_fa_bw_dq.deinit();
        var k_fa_bw_dkv = try pipeline.Kernel.init(ctx, runtime.faBwDkvSpv(head_dim_u32), 8, @sizeOf(runtime.FaBwDkvPush));
        errdefer k_fa_bw_dkv.deinit();
        var k_attn_dattn = try pipeline.Kernel.init(ctx, &shaders.attn_backward_dattn, 3, @sizeOf(runtime.AttnBackwardDattnPush));
        errdefer k_attn_dattn.deinit();
        var k_attn_dv = try pipeline.Kernel.init(ctx, &shaders.attn_backward_dv, 3, @sizeOf(runtime.AttnBackwardDvPush));
        errdefer k_attn_dv.deinit();
        var k_attn_dq = try pipeline.Kernel.init(ctx, &shaders.attn_backward_dq, 3, @sizeOf(runtime.AttnBackwardDqPush));
        errdefer k_attn_dq.deinit();
        var k_attn_dk = try pipeline.Kernel.init(ctx, &shaders.attn_backward_dk, 3, @sizeOf(runtime.AttnBackwardDkPush));
        errdefer k_attn_dk.deinit();
        var k_swiglu_fwd = try pipeline.Kernel.init(ctx, &shaders.swiglu_forward, 3, @sizeOf(runtime.SwigluPush));
        errdefer k_swiglu_fwd.deinit();
        var k_swiglu_bw = try pipeline.Kernel.init(ctx, &shaders.swiglu_backward, 5, @sizeOf(runtime.SwigluPush));
        errdefer k_swiglu_bw.deinit();
        var k_rope_fwd = try pipeline.Kernel.init(ctx, &shaders.qk_rope_partial_batched, 4, @sizeOf(runtime.QkRopeBatchedPush));
        errdefer k_rope_fwd.deinit();
        var k_rope_bw = try pipeline.Kernel.init(ctx, &shaders.qk_rope_backward_batched, 4, @sizeOf(runtime.QkRopeBatchedPush));
        errdefer k_rope_bw.deinit();
        var k_vec_add = try pipeline.Kernel.init(ctx, &shaders.vec_add, 3, @sizeOf(runtime.AddInPlacePush));
        errdefer k_vec_add.deinit();
        var k_add = try pipeline.Kernel.init(ctx, &shaders.add_in_place, 2, @sizeOf(runtime.AddInPlacePush));
        errdefer k_add.deinit();
        var k_lin_dx = try pipeline.Kernel.init(ctx, &shaders.linear_backward_dx_batched, 3, @sizeOf(runtime.LinearBatchedPush));
        errdefer k_lin_dx.deinit();
        var k_lin_dw = try pipeline.Kernel.init(ctx, &shaders.linear_backward_dw_batched, 3, @sizeOf(runtime.LinearBatchedPush));
        errdefer k_lin_dw.deinit();
        // CCE: fused LM-head matmul + online-softmax CE forward, plus split
        // backward (d_h is row-major, dW is vocab-major — both atomic-free
        // and bound to a [n_pos] u32 target buffer + [n_pos] f32 lse cache).
        // See shaders/cce_*.comp + src/cpu/cce.zig for the math + parity.
        var k_cce_forward = try pipeline.Kernel.init(ctx, &shaders.cce_forward, 5, @sizeOf(runtime.CceForwardPush));
        errdefer k_cce_forward.deinit();
        var k_cce_backward_dh = try pipeline.Kernel.init(ctx, &shaders.cce_backward_dh, 5, @sizeOf(runtime.CceBackwardPush));
        errdefer k_cce_backward_dh.deinit();
        var k_cce_backward_dw = try pipeline.Kernel.init(ctx, &shaders.cce_backward_dw, 5, @sizeOf(runtime.CceBackwardPush));
        errdefer k_cce_backward_dw.deinit();
        var k_embed_bw = try pipeline.Kernel.init(ctx, &shaders.embedding_backward, 3, @sizeOf(runtime.EmbeddingBackwardPush));
        errdefer k_embed_bw.deinit();
        var k_adam = try pipeline.Kernel.init(ctx, &shaders.adam_step, 4, @sizeOf(runtime.AdamStepPush));
        errdefer k_adam.deinit();

        // ── Recorder. Phase-1 dispatch count (forward + loss-grad +
        //    backward) is the largest: 9 + 38·N (3 stack-level forward
        //    + 4 stack-level backward + 38 per-layer + 2 reused). Adam
        //    phase is 8N+3, embed_bw phase is 1. Size for phase-1 with
        //    headroom; descriptors ≤ 5 per dispatch.
        // Phase-1 (forward+loss-grad+backward) per-layer dispatch is
        //   forward 15 (+2 RoPE if rotary_dim>0, +2 if qk_norm)
        //   + backward 27 (+2 RoPE, +2 qk_norm) = 42 to 50 per layer.
        // Plus 9 stack-level (embed + final_norm + lm_head + ce + lm_dx +
        // lm_dw + final_norm_bw + 2 spare).
        // LoRA: per enabled projection, forward matmul (1) →
        // recordLoraForward (5) net +4; backward lin_dx+lin_dw (2) →
        // recordLoraBackward (9) net +7. Total +11 per layer per
        // enabled projection. Adam phase goes +1 per enabled projection
        // (skip frozen W's Adam, add A and B). Bump per-layer cap by
        // 12 × popcount(targets) to absorb both phases with headroom.
        const lora_count: u32 = @popCount(cfg.lora_targets);
        const per_layer_max: u32 = 50 + 12 * lora_count;
        const phase1_dispatches: u32 = 32 + per_layer_max * cfg.n_layers;
        var rec = try recorder_mod.Recorder.init(ctx, phase1_dispatches, 8 * phase1_dispatches);
        errdefer rec.deinit();

        // ── Stack-level weight buffers.
        const buf_w_embed = try buffer.Buffer.initStatic(ctx, f32, weights.embed);
        errdefer @constCast(&buf_w_embed).deinit(ctx.device);
        const buf_w_final_norm = try buffer.Buffer.initStatic(ctx, f32, weights.final_norm);
        errdefer @constCast(&buf_w_final_norm).deinit(ctx.device);
        const buf_w_lm_head = try buffer.Buffer.initStatic(ctx, f32, weights.lm_head);
        errdefer @constCast(&buf_w_lm_head).deinit(ctx.device);

        // ── Dynamic input buffers.
        const buf_token_ids = try buffer.Buffer.initDynamic(ctx, n_pos * @sizeOf(u32));
        errdefer @constCast(&buf_token_ids).deinit(ctx.device);
        const buf_target_ids = try buffer.Buffer.initDynamic(ctx, n_pos * @sizeOf(u32));
        errdefer @constCast(&buf_target_ids).deinit(ctx.device);

        // ── Stack-level activation + grad buffers.
        const buf_x_emb = try buffer.Buffer.initDeviceOnly(ctx, n_pos * dim * f32sz);
        errdefer @constCast(&buf_x_emb).deinit(ctx.device);
        const buf_final_norm_out = try buffer.Buffer.initDeviceOnly(ctx, n_pos * dim * f32sz);
        errdefer @constCast(&buf_final_norm_out).deinit(ctx.device);
        // `buf_logits` only used by `forwardLogits` — step()'s gradient
        // path goes through cce_forward and never materializes logits.
        const buf_logits = try buffer.Buffer.initDeviceOnly(ctx, n_pos * vocab * f32sz);
        errdefer @constCast(&buf_logits).deinit(ctx.device);
        const buf_lse = try buffer.Buffer.initDeviceOnly(ctx, n_pos * f32sz);
        errdefer @constCast(&buf_lse).deinit(ctx.device);
        const buf_loss_per_row = try buffer.Buffer.initDeviceOnly(ctx, n_pos * f32sz);
        errdefer @constCast(&buf_loss_per_row).deinit(ctx.device);
        const buf_d_final_norm_out = try buffer.Buffer.initDeviceOnly(ctx, n_pos * dim * f32sz);
        errdefer @constCast(&buf_d_final_norm_out).deinit(ctx.device);
        const buf_d_last_y = try buffer.Buffer.initDeviceOnly(ctx, n_pos * dim * f32sz);
        errdefer @constCast(&buf_d_last_y).deinit(ctx.device);
        const buf_dw_lm_head = try buffer.Buffer.initDeviceOnly(ctx, weights.lm_head.len * f32sz);
        errdefer @constCast(&buf_dw_lm_head).deinit(ctx.device);
        const buf_dw_final_norm_partial = try buffer.Buffer.initDeviceOnly(ctx, n_pos * dim * f32sz);
        errdefer @constCast(&buf_dw_final_norm_partial).deinit(ctx.device);
        const buf_dw_final_norm = try buffer.Buffer.initDynamic(ctx, dim * f32sz);
        errdefer @constCast(&buf_dw_final_norm).deinit(ctx.device);
        const buf_dE_embed = try buffer.Buffer.initDeviceOnly(ctx, weights.embed.len * f32sz);
        errdefer @constCast(&buf_dE_embed).deinit(ctx.device);

        // ── Stack-level Adam m/v.
        const buf_m_embed = try buffer.Buffer.initDeviceOnly(ctx, weights.embed.len * f32sz);
        errdefer @constCast(&buf_m_embed).deinit(ctx.device);
        const buf_v_embed = try buffer.Buffer.initDeviceOnly(ctx, weights.embed.len * f32sz);
        errdefer @constCast(&buf_v_embed).deinit(ctx.device);
        const buf_m_final_norm = try buffer.Buffer.initDeviceOnly(ctx, dim * f32sz);
        errdefer @constCast(&buf_m_final_norm).deinit(ctx.device);
        const buf_v_final_norm = try buffer.Buffer.initDeviceOnly(ctx, dim * f32sz);
        errdefer @constCast(&buf_v_final_norm).deinit(ctx.device);
        const buf_m_lm_head = try buffer.Buffer.initDeviceOnly(ctx, weights.lm_head.len * f32sz);
        errdefer @constCast(&buf_m_lm_head).deinit(ctx.device);
        const buf_v_lm_head = try buffer.Buffer.initDeviceOnly(ctx, weights.lm_head.len * f32sz);
        errdefer @constCast(&buf_v_lm_head).deinit(ctx.device);

        // ── Per-layer arrays. Track populated counts so partial-init
        //    failures roll back correctly. We use slice errdefers that
        //    only deinit the populated prefix.
        var per_layer = try PerLayerArrays.alloc(allocator, n_layers, cfg);
        errdefer per_layer.deinitOnError(ctx.device, allocator);

        for (0..n_layers) |li| {
            const lw = weights.layers[li];
            try per_layer.populateLayer(ctx, li, cfg, lw);
        }

        // ── LoRA state (A4-2 → A4-3). Allocated only when
        //    cfg.lora_targets != 0; the shaders + per-projection slots
        //    are LoRA-specific so we don't pay any allocation cost in
        //    the no-LoRA path.
        var lora_state: ?LoraState = if (cfg.lora_targets != 0)
            try LoraState.allocAndInit(allocator, ctx, cfg, weights)
        else
            null;
        errdefer if (lora_state) |*ls| ls.deinit(ctx.device, allocator);

        // ── Shared scratch.
        const sc_scores = try buffer.Buffer.initDeviceOnly(ctx, scores_total * f32sz);
        errdefer @constCast(&sc_scores).deinit(ctx.device);
        // Per-layer FA LSE + D buffers — sized [n_pos × n_heads] each.
        // Total at Qwen3-0.6B max_pos=2048: 28 layers × 2048 × 16 × 4B
        // × 2 = 7.3 MB. Cheap relative to weights.
        const fa_lse_total: usize = @as(usize, cfg.n_pos) * cfg.n_heads;
        const buf_fa_lse = try allocator.alloc(buffer.Buffer, n_layers);
        errdefer allocator.free(buf_fa_lse);
        const buf_fa_d = try allocator.alloc(buffer.Buffer, n_layers);
        errdefer allocator.free(buf_fa_d);
        var fa_alloc_count: usize = 0;
        errdefer {
            var ii: usize = 0;
            while (ii < fa_alloc_count) : (ii += 1) {
                buf_fa_lse[ii].deinit(ctx.device);
                buf_fa_d[ii].deinit(ctx.device);
            }
        }
        for (0..n_layers) |li| {
            buf_fa_lse[li] = try buffer.Buffer.initDeviceOnly(ctx, fa_lse_total * f32sz);
            buf_fa_d[li]   = try buffer.Buffer.initDeviceOnly(ctx, fa_lse_total * f32sz);
            fa_alloc_count = li + 1;
        }
        const sc_o = try buffer.Buffer.initDeviceOnly(ctx, n_pos * dim * f32sz);
        errdefer @constCast(&sc_o).deinit(ctx.device);
        const sc_ff_out = try buffer.Buffer.initDeviceOnly(ctx, n_pos * dim * f32sz);
        errdefer @constCast(&sc_ff_out).deinit(ctx.device);
        const sc_q_pre = try buffer.Buffer.initDeviceOnly(ctx, n_pos * q_dim * f32sz);
        errdefer @constCast(&sc_q_pre).deinit(ctx.device);
        const sc_k_pre = try buffer.Buffer.initDeviceOnly(ctx, n_pos * kv_dim * f32sz);
        errdefer @constCast(&sc_k_pre).deinit(ctx.device);
        const sc_dQ_pre = try buffer.Buffer.initDeviceOnly(ctx, n_pos * q_dim * f32sz);
        errdefer @constCast(&sc_dQ_pre).deinit(ctx.device);
        const sc_dK_pre = try buffer.Buffer.initDeviceOnly(ctx, n_pos * kv_dim * f32sz);
        errdefer @constCast(&sc_dK_pre).deinit(ctx.device);
        // sc_dQ_pre_norm/sc_dK_pre_norm: pre-norm dQ/dK (rmsnorm-bw output).
        // 1-byte placeholder when qk_norm=false; never bound by any
        // dispatch in that case.
        const qkn_q_bytes: usize = if (cfg.qk_norm) n_pos * q_dim * f32sz else 4;
        const qkn_k_bytes: usize = if (cfg.qk_norm) n_pos * kv_dim * f32sz else 4;
        const sc_dQ_pre_norm = try buffer.Buffer.initDeviceOnly(ctx, qkn_q_bytes);
        errdefer @constCast(&sc_dQ_pre_norm).deinit(ctx.device);
        const sc_dK_pre_norm = try buffer.Buffer.initDeviceOnly(ctx, qkn_k_bytes);
        errdefer @constCast(&sc_dK_pre_norm).deinit(ctx.device);
        const sc_d_gated = try buffer.Buffer.initDeviceOnly(ctx, n_pos * ff_dim * f32sz);
        errdefer @constCast(&sc_d_gated).deinit(ctx.device);
        const sc_d_pre_gate = try buffer.Buffer.initDeviceOnly(ctx, n_pos * ff_dim * f32sz);
        errdefer @constCast(&sc_d_pre_gate).deinit(ctx.device);
        const sc_d_up_grad = try buffer.Buffer.initDeviceOnly(ctx, n_pos * ff_dim * f32sz);
        errdefer @constCast(&sc_d_up_grad).deinit(ctx.device);
        const sc_d_n2 = try buffer.Buffer.initDeviceOnly(ctx, n_pos * dim * f32sz);
        errdefer @constCast(&sc_d_n2).deinit(ctx.device);
        const sc_d_n2_up = try buffer.Buffer.initDeviceOnly(ctx, n_pos * dim * f32sz);
        errdefer @constCast(&sc_d_n2_up).deinit(ctx.device);
        const sc_d_mid_norm = try buffer.Buffer.initDeviceOnly(ctx, n_pos * dim * f32sz);
        errdefer @constCast(&sc_d_mid_norm).deinit(ctx.device);
        const sc_d_attn_out = try buffer.Buffer.initDeviceOnly(ctx, n_pos * q_dim * f32sz);
        errdefer @constCast(&sc_d_attn_out).deinit(ctx.device);
        const sc_d_attn = try buffer.Buffer.initDeviceOnly(ctx, scores_total * f32sz);
        errdefer @constCast(&sc_d_attn).deinit(ctx.device);
        const sc_d_scores = try buffer.Buffer.initDeviceOnly(ctx, scores_total * f32sz);
        errdefer @constCast(&sc_d_scores).deinit(ctx.device);
        const sc_dQ = try buffer.Buffer.initDeviceOnly(ctx, n_pos * q_dim * f32sz);
        errdefer @constCast(&sc_dQ).deinit(ctx.device);
        const sc_dK = try buffer.Buffer.initDeviceOnly(ctx, n_pos * kv_dim * f32sz);
        errdefer @constCast(&sc_dK).deinit(ctx.device);
        const sc_dV = try buffer.Buffer.initDeviceOnly(ctx, n_pos * kv_dim * f32sz);
        errdefer @constCast(&sc_dV).deinit(ctx.device);
        const sc_d_n1 = try buffer.Buffer.initDeviceOnly(ctx, n_pos * dim * f32sz);
        errdefer @constCast(&sc_d_n1).deinit(ctx.device);
        const sc_d_n1_k = try buffer.Buffer.initDeviceOnly(ctx, n_pos * dim * f32sz);
        errdefer @constCast(&sc_d_n1_k).deinit(ctx.device);
        const sc_d_n1_v = try buffer.Buffer.initDeviceOnly(ctx, n_pos * dim * f32sz);
        errdefer @constCast(&sc_d_n1_v).deinit(ctx.device);

        // ── Host scratch for dw_partial reduce.
        // Sized for the largest (n_rows × dim_per_row) shape any of the
        // partial buffers can have:
        //   final_norm / n1 / n2 partials → n_pos · dim
        //   q-norm partial               → n_pos · n_heads · head_dim
        //   k-norm partial               → n_pos · n_kv_heads · head_dim
        // For most LLMs n_heads · head_dim == dim, but Qwen3-0.6B has
        // n_heads · head_dim = 16 · 128 = 2048 vs dim = 1024 (q-side
        // expanded for non-GQA path), so the q-norm partial is the
        // limiting factor there. Pre-CCE this overflowed silently in
        // ReleaseFast (assertion compiled out) and panicked in Debug.
        const head_dim_usz: usize = @intCast(cfg.head_dim);
        const max_partial_row: usize = @max(
            dim,
            @max(
                @as(usize, cfg.n_heads) * head_dim_usz,
                @as(usize, cfg.n_kv_heads) * head_dim_usz,
            ),
        );
        const dw_partial_host = try allocator.alloc(f32, n_pos * max_partial_row);
        errdefer allocator.free(dw_partial_host);
        // dw_reduced takes the per-row width (dim, or head_dim for q/k-norm).
        // dim ≥ head_dim always, so sizing to dim is sufficient.
        const dw_reduced = try allocator.alloc(f32, dim);
        errdefer allocator.free(dw_reduced);

        // ── Cached pushes (build once).
        const push_embed = runtime.EmbedLookupBatchedPush{ .dim = cfg.dim, .n_pos = cfg.n_pos, .scale = 1.0 };
        const push_rms = runtime.RmsnormPush{ .dim = cfg.dim, .eps = cfg.rms_eps, .gemma_quirk = 0 };
        const push_n_pos_dim = runtime.AddInPlacePush{ .n = cfg.n_pos * cfg.dim };
        const push_swiglu = runtime.SwigluPush{ .n = cfg.n_pos * cfg.ff_dim };
        const push_softmax = runtime.SoftmaxPush{ .dim = cfg.n_pos, .stride = cfg.n_pos };
        const push_cce = runtime.CceForwardPush{ .n_samples = cfg.n_pos, .vocab = cfg.vocab_size, .dim = cfg.dim };
        const push_mm_q = runtime.MatmulPush{ .m = cfg.n_pos, .n = @intCast(q_dim), .k = cfg.dim };
        const push_mm_k = runtime.MatmulPush{ .m = cfg.n_pos, .n = @intCast(kv_dim), .k = cfg.dim };
        const push_mm_v = runtime.MatmulPush{ .m = cfg.n_pos, .n = @intCast(kv_dim), .k = cfg.dim };
        const push_mm_o = runtime.MatmulPush{ .m = cfg.n_pos, .n = cfg.dim, .k = @intCast(q_dim) };
        const push_mm_gate = runtime.MatmulPush{ .m = cfg.n_pos, .n = cfg.ff_dim, .k = cfg.dim };
        const push_mm_up = runtime.MatmulPush{ .m = cfg.n_pos, .n = cfg.ff_dim, .k = cfg.dim };
        const push_mm_down = runtime.MatmulPush{ .m = cfg.n_pos, .n = cfg.dim, .k = cfg.ff_dim };
        const push_mm_lm_head = runtime.MatmulPush{ .m = cfg.n_pos, .n = cfg.vocab_size, .k = cfg.dim };
        const push_attn_scores = runtime.AttnScoresTrainPush{
            .n_q = cfg.n_pos,
            .n_heads = cfg.n_heads,
            .heads_per_kv = heads_per_kv,
            .head_dim = cfg.head_dim,
            .n_kv = cfg.n_pos,
            .kv_stride = @intCast(kv_dim),
            .scores_stride = cfg.n_pos,
            .causal = if (cfg.causal) 1 else 0,
            .inv_sqrt_dim = inv_sqrt_d,
        };
        const push_attn_output = runtime.AttnOutputTrainPush{
            .n_q = cfg.n_pos,
            .n_heads = cfg.n_heads,
            .heads_per_kv = heads_per_kv,
            .head_dim = cfg.head_dim,
            .n_kv = cfg.n_pos,
            .kv_stride = @intCast(kv_dim),
            .attn_stride = cfg.n_pos,
        };
        const push_fa_forward = runtime.FaForwardPush{
            .n_q = cfg.n_pos,
            .n_heads = cfg.n_heads,
            .heads_per_kv = heads_per_kv,
            .head_dim = cfg.head_dim,
            .n_kv = cfg.n_pos,
            .kv_stride = @intCast(kv_dim),
            .causal = if (cfg.causal) 1 else 0,
            .write_lse = 0,
            .inv_sqrt_dim = inv_sqrt_d,
        };
        // Same as `push_fa_forward` but with `write_lse = 1` so the FA-2
        // backward can recompute softmax from the saved LSE row. Used by
        // `step` (training) only.
        var push_fa_forward_train = push_fa_forward;
        push_fa_forward_train.write_lse = 1;
        const push_fa_bw_d = runtime.FaBwDPush{
            .n_q = cfg.n_pos,
            .n_heads = cfg.n_heads,
            .head_dim = cfg.head_dim,
        };
        const push_fa_bw_dq = runtime.FaBwDqPush{
            .n_q = cfg.n_pos,
            .n_heads = cfg.n_heads,
            .heads_per_kv = heads_per_kv,
            .head_dim = cfg.head_dim,
            .n_kv = cfg.n_pos,
            .kv_stride = @intCast(kv_dim),
            .causal = if (cfg.causal) 1 else 0,
            .inv_sqrt_dim = inv_sqrt_d,
        };
        const push_fa_bw_dkv = runtime.FaBwDkvPush{
            .n_q = cfg.n_pos,
            .n_kv = cfg.n_pos,
            .n_heads = cfg.n_heads,
            .n_kv_heads = cfg.n_kv_heads,
            .heads_per_kv = heads_per_kv,
            .head_dim = cfg.head_dim,
            .kv_stride = @intCast(kv_dim),
            .causal = if (cfg.causal) 1 else 0,
            .inv_sqrt_dim = inv_sqrt_d,
        };
        // FA shaders ship in two HEAD_DIM_MAX variants (128 default
        // BC=16; _d256 BC=8) — both fit AMD RDNA's 32 KB/WG ceiling.
        // The pipeline pick above selected the right SPIR-V; here we
        // gate the FA dispatch off for anything still above the cap.
        const attn_use_fa: bool = head_dim_u32 <= runtime.FA_HEAD_DIM_MAX;
        const push_rope_qk = runtime.QkRopeBatchedPush{
            .n_pos = cfg.n_pos,
            .n_q_heads = cfg.n_heads,
            .n_kv_heads = cfg.n_kv_heads,
            .head_dim = cfg.head_dim,
            .rotary_dim = cfg.rotary_dim,
            .theta_base = cfg.rope_theta,
        };
        const push_rms_qk = runtime.RmsnormPush{
            .dim = cfg.head_dim,
            .eps = cfg.rms_eps,
            .gemma_quirk = 0,
        };
        const push_lin_lm_head = runtime.LinearBatchedPush{ .M = cfg.n_pos, .N = cfg.vocab_size, .K = cfg.dim };
        const push_lin_down = runtime.LinearBatchedPush{ .M = cfg.n_pos, .N = cfg.dim, .K = cfg.ff_dim };
        const push_lin_gate = runtime.LinearBatchedPush{ .M = cfg.n_pos, .N = cfg.ff_dim, .K = cfg.dim };
        const push_lin_up = runtime.LinearBatchedPush{ .M = cfg.n_pos, .N = cfg.ff_dim, .K = cfg.dim };
        const push_lin_o = runtime.LinearBatchedPush{ .M = cfg.n_pos, .N = cfg.dim, .K = @intCast(q_dim) };
        const push_lin_q = runtime.LinearBatchedPush{ .M = cfg.n_pos, .N = @intCast(q_dim), .K = cfg.dim };
        const push_lin_k = runtime.LinearBatchedPush{ .M = cfg.n_pos, .N = @intCast(kv_dim), .K = cfg.dim };
        const push_lin_v = runtime.LinearBatchedPush{ .M = cfg.n_pos, .N = @intCast(kv_dim), .K = cfg.dim };
        const push_dattn = runtime.AttnBackwardDattnPush{
            .n_q = cfg.n_pos,
            .n_heads = cfg.n_heads,
            .heads_per_kv = heads_per_kv,
            .head_dim = cfg.head_dim,
            .n_kv = cfg.n_pos,
            .kv_stride = @intCast(kv_dim),
            .attn_stride = cfg.n_pos,
        };
        const push_dv = runtime.AttnBackwardDvPush{
            .n_q = cfg.n_pos,
            .n_heads = cfg.n_heads,
            .heads_per_kv = heads_per_kv,
            .n_kv_heads = cfg.n_kv_heads,
            .head_dim = cfg.head_dim,
            .n_kv = cfg.n_pos,
            .attn_stride = cfg.n_pos,
        };
        const push_dq = runtime.AttnBackwardDqPush{
            .n_q = cfg.n_pos,
            .n_heads = cfg.n_heads,
            .heads_per_kv = heads_per_kv,
            .head_dim = cfg.head_dim,
            .n_kv = cfg.n_pos,
            .kv_stride = @intCast(kv_dim),
            .scores_stride = cfg.n_pos,
            .inv_sqrt_dim = inv_sqrt_d,
        };
        const push_dk = runtime.AttnBackwardDkPush{
            .n_q = cfg.n_pos,
            .n_heads = cfg.n_heads,
            .heads_per_kv = heads_per_kv,
            .n_kv_heads = cfg.n_kv_heads,
            .head_dim = cfg.head_dim,
            .n_kv = cfg.n_pos,
            .scores_stride = cfg.n_pos,
            .inv_sqrt_dim = inv_sqrt_d,
        };
        const push_embed_bw = runtime.EmbeddingBackwardPush{
            .dim = cfg.dim,
            .n_pos = cfg.n_pos,
            .vocab_size = cfg.vocab_size,
        };

        return Runner{
            .ctx = ctx,
            .allocator = allocator,
            .cfg = cfg,
            .step_t = 1,
            .k_embed = k_embed,
            .k_rms = k_rms,
            .k_rms_bw = k_rms_bw,
            .k_matmul = k_matmul,
            .k_attn_scores = k_attn_scores,
            .k_attn_output = k_attn_output,
            .k_softmax = k_softmax,
            .k_softmax_bw = k_softmax_bw,
            .k_fa_forward = k_fa_forward,
            .k_fa_bw_d = k_fa_bw_d,
            .k_fa_bw_dq = k_fa_bw_dq,
            .k_fa_bw_dkv = k_fa_bw_dkv,
            .k_attn_dattn = k_attn_dattn,
            .k_attn_dv = k_attn_dv,
            .k_attn_dq = k_attn_dq,
            .k_attn_dk = k_attn_dk,
            .k_swiglu_fwd = k_swiglu_fwd,
            .k_swiglu_bw = k_swiglu_bw,
            .k_rope_fwd = k_rope_fwd,
            .k_rope_bw = k_rope_bw,
            .k_vec_add = k_vec_add,
            .k_add = k_add,
            .k_lin_dx = k_lin_dx,
            .k_lin_dw = k_lin_dw,
            .k_cce_forward = k_cce_forward,
            .k_cce_backward_dh = k_cce_backward_dh,
            .k_cce_backward_dw = k_cce_backward_dw,
            .k_embed_bw = k_embed_bw,
            .k_adam = k_adam,
            .rec = rec,
            .buf_w_embed = buf_w_embed,
            .buf_w_final_norm = buf_w_final_norm,
            .buf_w_lm_head = buf_w_lm_head,
            .buf_token_ids = buf_token_ids,
            .buf_target_ids = buf_target_ids,
            .buf_x_emb = buf_x_emb,
            .buf_final_norm_out = buf_final_norm_out,
            .buf_logits = buf_logits,
            .buf_lse = buf_lse,
            .buf_loss_per_row = buf_loss_per_row,
            .buf_d_final_norm_out = buf_d_final_norm_out,
            .buf_d_last_y = buf_d_last_y,
            .buf_dw_lm_head = buf_dw_lm_head,
            .buf_dw_final_norm_partial = buf_dw_final_norm_partial,
            .buf_dw_final_norm = buf_dw_final_norm,
            .buf_dE_embed = buf_dE_embed,
            .buf_m_embed = buf_m_embed,
            .buf_v_embed = buf_v_embed,
            .buf_m_final_norm = buf_m_final_norm,
            .buf_v_final_norm = buf_v_final_norm,
            .buf_m_lm_head = buf_m_lm_head,
            .buf_v_lm_head = buf_v_lm_head,
            .buf_w_n1 = per_layer.w_n1,
            .buf_w_q = per_layer.w_q,
            .buf_w_k = per_layer.w_k,
            .buf_w_v = per_layer.w_v,
            .buf_w_o = per_layer.w_o,
            .buf_w_n2 = per_layer.w_n2,
            .buf_w_gate = per_layer.w_gate,
            .buf_w_up = per_layer.w_up,
            .buf_w_down = per_layer.w_down,
            .buf_n1 = per_layer.a_n1,
            .buf_q = per_layer.a_q,
            .buf_k = per_layer.a_k,
            .buf_v = per_layer.a_v,
            .buf_attn = per_layer.a_attn,
            .buf_attn_out = per_layer.a_attn_out,
            .buf_mid = per_layer.a_mid,
            .buf_n2 = per_layer.a_n2,
            .buf_pre_gate = per_layer.a_pre_gate,
            .buf_up = per_layer.a_up,
            .buf_gated = per_layer.a_gated,
            .buf_y = per_layer.a_y,
            .buf_dw_n1_partial = per_layer.dw_n1_partial,
            .buf_dw_q = per_layer.dw_q,
            .buf_dw_k = per_layer.dw_k,
            .buf_dw_v = per_layer.dw_v,
            .buf_dw_o = per_layer.dw_o,
            .buf_dw_n2_partial = per_layer.dw_n2_partial,
            .buf_dw_gate = per_layer.dw_gate,
            .buf_dw_up = per_layer.dw_up,
            .buf_dw_down = per_layer.dw_down,
            .buf_dw_n1 = per_layer.dw_n1,
            .buf_dw_n2 = per_layer.dw_n2,
            .buf_m_n1 = per_layer.m_n1,
            .buf_v_n1 = per_layer.v_n1,
            .buf_m_q = per_layer.m_q,
            .buf_v_q = per_layer.v_q,
            .buf_m_k = per_layer.m_k,
            .buf_v_k = per_layer.v_k,
            .buf_m_v = per_layer.m_v,
            .buf_v_v = per_layer.v_v,
            .buf_m_o = per_layer.m_o,
            .buf_v_o = per_layer.v_o,
            .buf_m_n2 = per_layer.m_n2,
            .buf_v_n2 = per_layer.v_n2,
            .buf_m_gate = per_layer.m_gate,
            .buf_v_gate = per_layer.v_gate,
            .buf_m_up = per_layer.m_up,
            .buf_v_up = per_layer.v_up,
            .buf_m_down = per_layer.m_down,
            .buf_v_down = per_layer.v_down,
            .buf_w_q_norm = per_layer.w_q_norm,
            .buf_w_k_norm = per_layer.w_k_norm,
            .buf_q_pre_norm = per_layer.a_q_pre_norm,
            .buf_k_pre_norm = per_layer.a_k_pre_norm,
            .buf_dw_q_norm_partial = per_layer.dw_q_norm_partial,
            .buf_dw_k_norm_partial = per_layer.dw_k_norm_partial,
            .buf_dw_q_norm = per_layer.dw_q_norm,
            .buf_dw_k_norm = per_layer.dw_k_norm,
            .buf_m_q_norm = per_layer.m_q_norm,
            .buf_v_q_norm = per_layer.v_q_norm,
            .buf_m_k_norm = per_layer.m_k_norm,
            .buf_v_k_norm = per_layer.v_k_norm,
            .buf_d_x_in = per_layer.d_x_in,
            .sc_scores = sc_scores,
            .buf_fa_lse = buf_fa_lse,
            .buf_fa_d = buf_fa_d,
            .sc_o = sc_o,
            .sc_ff_out = sc_ff_out,
            .sc_q_pre = sc_q_pre,
            .sc_k_pre = sc_k_pre,
            .sc_dQ_pre = sc_dQ_pre,
            .sc_dK_pre = sc_dK_pre,
            .sc_dQ_pre_norm = sc_dQ_pre_norm,
            .sc_dK_pre_norm = sc_dK_pre_norm,
            .sc_d_gated = sc_d_gated,
            .sc_d_pre_gate = sc_d_pre_gate,
            .sc_d_up_grad = sc_d_up_grad,
            .sc_d_n2 = sc_d_n2,
            .sc_d_n2_up = sc_d_n2_up,
            .sc_d_mid_norm = sc_d_mid_norm,
            .sc_d_attn_out = sc_d_attn_out,
            .sc_d_attn = sc_d_attn,
            .sc_d_scores = sc_d_scores,
            .sc_dQ = sc_dQ,
            .sc_dK = sc_dK,
            .sc_dV = sc_dV,
            .sc_d_n1 = sc_d_n1,
            .sc_d_n1_k = sc_d_n1_k,
            .sc_d_n1_v = sc_d_n1_v,
            .dw_partial_host = dw_partial_host,
            .dw_reduced = dw_reduced,
            .push_embed = push_embed,
            .push_rms = push_rms,
            .push_n_pos_dim = push_n_pos_dim,
            .push_swiglu = push_swiglu,
            .push_softmax = push_softmax,
            .push_cce = push_cce,
            .push_mm_q = push_mm_q,
            .push_mm_k = push_mm_k,
            .push_mm_v = push_mm_v,
            .push_mm_o = push_mm_o,
            .push_mm_gate = push_mm_gate,
            .push_mm_up = push_mm_up,
            .push_mm_down = push_mm_down,
            .push_mm_lm_head = push_mm_lm_head,
            .push_attn_scores = push_attn_scores,
            .push_attn_output = push_attn_output,
            .push_fa_forward = push_fa_forward,
            .push_fa_forward_train = push_fa_forward_train,
            .push_fa_bw_d = push_fa_bw_d,
            .push_fa_bw_dq = push_fa_bw_dq,
            .push_fa_bw_dkv = push_fa_bw_dkv,
            .attn_use_fa = attn_use_fa,
            .push_rope_qk = push_rope_qk,
            .push_rms_qk = push_rms_qk,
            .push_lin_lm_head = push_lin_lm_head,
            .push_lin_down = push_lin_down,
            .push_lin_gate = push_lin_gate,
            .push_lin_up = push_lin_up,
            .push_lin_o = push_lin_o,
            .push_lin_q = push_lin_q,
            .push_lin_k = push_lin_k,
            .push_lin_v = push_lin_v,
            .push_dattn = push_dattn,
            .push_dv = push_dv,
            .push_dq = push_dq,
            .push_dk = push_dk,
            .push_embed_bw = push_embed_bw,
            .lora = lora_state,
        };
    }

    pub fn deinit(self: *Runner) void {
        const dev = self.ctx.device;
        const alloc = self.allocator;

        // Stack-level buffers.
        self.buf_w_embed.deinit(dev);
        self.buf_w_final_norm.deinit(dev);
        self.buf_w_lm_head.deinit(dev);
        self.buf_token_ids.deinit(dev);
        self.buf_target_ids.deinit(dev);
        self.buf_x_emb.deinit(dev);
        self.buf_final_norm_out.deinit(dev);
        self.buf_logits.deinit(dev);
        self.buf_lse.deinit(dev);
        self.buf_loss_per_row.deinit(dev);
        self.buf_d_final_norm_out.deinit(dev);
        self.buf_d_last_y.deinit(dev);
        self.buf_dw_lm_head.deinit(dev);
        self.buf_dw_final_norm_partial.deinit(dev);
        self.buf_dw_final_norm.deinit(dev);
        self.buf_dE_embed.deinit(dev);
        self.buf_m_embed.deinit(dev);
        self.buf_v_embed.deinit(dev);
        self.buf_m_final_norm.deinit(dev);
        self.buf_v_final_norm.deinit(dev);
        self.buf_m_lm_head.deinit(dev);
        self.buf_v_lm_head.deinit(dev);

        // Per-layer buffers.
        const arrs = [_][]buffer.Buffer{
            self.buf_w_n1,         self.buf_w_q,          self.buf_w_k,
            self.buf_w_v,          self.buf_w_o,          self.buf_w_n2,
            self.buf_w_gate,       self.buf_w_up,         self.buf_w_down,
            self.buf_n1,           self.buf_q,            self.buf_k,
            self.buf_v,            self.buf_attn,         self.buf_attn_out,
            self.buf_mid,          self.buf_n2,           self.buf_pre_gate,
            self.buf_up,           self.buf_gated,        self.buf_y,
            self.buf_dw_n1_partial, self.buf_dw_q,        self.buf_dw_k,
            self.buf_dw_v,         self.buf_dw_o,         self.buf_dw_n2_partial,
            self.buf_dw_gate,      self.buf_dw_up,        self.buf_dw_down,
            self.buf_dw_n1,        self.buf_dw_n2,        self.buf_m_n1,
            self.buf_v_n1,         self.buf_m_q,          self.buf_v_q,
            self.buf_m_k,          self.buf_v_k,          self.buf_m_v,
            self.buf_v_v,          self.buf_m_o,          self.buf_v_o,
            self.buf_m_n2,         self.buf_v_n2,         self.buf_m_gate,
            self.buf_v_gate,       self.buf_m_up,         self.buf_v_up,
            self.buf_m_down,       self.buf_v_down,       self.buf_w_q_norm,
            self.buf_w_k_norm,     self.buf_q_pre_norm,   self.buf_k_pre_norm,
            self.buf_dw_q_norm_partial, self.buf_dw_k_norm_partial,
            self.buf_dw_q_norm,    self.buf_dw_k_norm,    self.buf_m_q_norm,
            self.buf_v_q_norm,     self.buf_m_k_norm,     self.buf_v_k_norm,
            self.buf_d_x_in,
        };
        for (arrs) |arr| {
            for (arr) |*b| b.deinit(dev);
            alloc.free(arr);
        }

        // Shared scratch.
        self.sc_scores.deinit(dev);
        for (self.buf_fa_lse) |*b| b.deinit(dev);
        for (self.buf_fa_d) |*b| b.deinit(dev);
        self.allocator.free(self.buf_fa_lse);
        self.allocator.free(self.buf_fa_d);
        self.sc_o.deinit(dev);
        self.sc_ff_out.deinit(dev);
        self.sc_q_pre.deinit(dev);
        self.sc_k_pre.deinit(dev);
        self.sc_dQ_pre.deinit(dev);
        self.sc_dK_pre.deinit(dev);
        self.sc_dQ_pre_norm.deinit(dev);
        self.sc_dK_pre_norm.deinit(dev);
        self.sc_d_gated.deinit(dev);
        self.sc_d_pre_gate.deinit(dev);
        self.sc_d_up_grad.deinit(dev);
        self.sc_d_n2.deinit(dev);
        self.sc_d_n2_up.deinit(dev);
        self.sc_d_mid_norm.deinit(dev);
        self.sc_d_attn_out.deinit(dev);
        self.sc_d_attn.deinit(dev);
        self.sc_d_scores.deinit(dev);
        self.sc_dQ.deinit(dev);
        self.sc_dK.deinit(dev);
        self.sc_dV.deinit(dev);
        self.sc_d_n1.deinit(dev);
        self.sc_d_n1_k.deinit(dev);
        self.sc_d_n1_v.deinit(dev);

        alloc.free(self.dw_partial_host);
        alloc.free(self.dw_reduced);

        // LoRA state (its k_scale + per-projection slots + scratches),
        // if present.
        if (self.lora) |*ls| ls.deinit(dev, alloc);

        // Pipelines + recorder.
        self.k_embed.deinit();
        self.k_rms.deinit();
        self.k_rms_bw.deinit();
        self.k_matmul.deinit();
        self.k_attn_scores.deinit();
        self.k_attn_output.deinit();
        self.k_softmax.deinit();
        self.k_softmax_bw.deinit();
        self.k_fa_forward.deinit();
        self.k_fa_bw_d.deinit();
        self.k_fa_bw_dq.deinit();
        self.k_fa_bw_dkv.deinit();
        self.k_attn_dattn.deinit();
        self.k_attn_dv.deinit();
        self.k_attn_dq.deinit();
        self.k_attn_dk.deinit();
        self.k_swiglu_fwd.deinit();
        self.k_swiglu_bw.deinit();
        self.k_rope_fwd.deinit();
        self.k_rope_bw.deinit();
        self.k_vec_add.deinit();
        self.k_add.deinit();
        self.k_lin_dx.deinit();
        self.k_lin_dw.deinit();
        self.k_cce_forward.deinit();
        self.k_cce_backward_dh.deinit();
        self.k_cce_backward_dw.deinit();
        self.k_embed_bw.deinit();
        self.k_adam.deinit();
        self.rec.deinit();
    }

    /// One Adam training step. `token_ids` and `target_ids` length must
    /// each equal cfg.n_pos. `target_ids[p]` is the gold next-token id at
    /// position p — the CCE forward kernel handles the implicit one-hot
    /// without ever materializing it. Increments `step_t`.
    pub fn step(self: *Runner, token_ids: []const u32, target_ids: []const u32) !void {
        const cfg = self.cfg;
        if (token_ids.len != cfg.n_pos) return error.TokenIdsLen;
        if (target_ids.len != cfg.n_pos) return error.TargetLen;

        self.buf_token_ids.update(u32, token_ids);
        self.buf_target_ids.update(u32, target_ids);

        // CCE backward dW accumulates (matches embedding_backward, linearBackward
        // dW), so reset between steps. The other accumulating buffer
        // (buf_dE_embed) is zeroed in Phase 2 below.
        try self.buf_dw_lm_head.fillZero(self.ctx);

        // ── Phase 1: forward + loss-grad + per-layer backward.
        try self.rec.reset();
        try self.rec.begin();
        try self.recordEmbedLookup();
        // step() runs the full forward-backward loop; backward consumes
        // buf_attn[i] saved by the 3-pass chain, so forward_only=false.
        for (0..cfg.n_layers) |li| try self.recordLayerForward(@intCast(li), false);
        try self.recordHeadForwardAndLossGrad();
        try self.recordHeadBackward();
        var li: usize = cfg.n_layers;
        while (li > 0) {
            li -= 1;
            try self.recordLayerBackward(@intCast(li));
        }
        try self.rec.endAndSubmit();

        // ── Phase 2: zero dE_embed, then embedding_backward (additive scatter).
        try self.buf_dE_embed.fillZero(self.ctx);
        try self.rec.reset();
        try self.rec.begin();
        try self.rec.dispatch(
            &self.k_embed_bw,
            &.{ &self.buf_d_x_in[0], &self.buf_token_ids, &self.buf_dE_embed },
            &self.push_embed_bw,
            cfg.vocab_size,
            1,
            1,
        );
        try self.rec.endAndSubmit();

        // ── Phase 3: host-side row-reduce of all dw_partials → dynamic bufs.
        try self.reduceDwPartial(&self.buf_dw_final_norm_partial, &self.buf_dw_final_norm);
        for (0..cfg.n_layers) |i| {
            try self.reduceDwPartial(&self.buf_dw_n1_partial[i], &self.buf_dw_n1[i]);
            try self.reduceDwPartial(&self.buf_dw_n2_partial[i], &self.buf_dw_n2[i]);
            if (cfg.qk_norm) {
                const hd: usize = @intCast(cfg.head_dim);
                try self.reduceDwPartialN(&self.buf_dw_q_norm_partial[i], &self.buf_dw_q_norm[i], cfg.n_pos * cfg.n_heads, hd);
                try self.reduceDwPartialN(&self.buf_dw_k_norm_partial[i], &self.buf_dw_k_norm[i], cfg.n_pos * cfg.n_kv_heads, hd);
            }
        }

        // ── Phase 4: Adam update on every parameter.
        try self.rec.reset();
        try self.rec.begin();
        try self.recordAdamAll();
        try self.rec.endAndSubmit();

        self.step_t += 1;
    }

    /// Forward-only: reads token_ids, runs the full forward pass, copies
    /// final logits out to `out_logits`. Does not modify any weights.
    /// Used for measuring CE loss without taking a gradient step.
    pub fn forwardLogits(self: *Runner, token_ids: []const u32, out_logits: []f32) !void {
        const cfg = self.cfg;
        if (token_ids.len != cfg.n_pos) return error.TokenIdsLen;
        if (out_logits.len != @as(usize, cfg.n_pos) * cfg.vocab_size) return error.OutLogitsLen;

        self.buf_token_ids.update(u32, token_ids);

        try self.rec.reset();
        try self.rec.begin();
        try self.recordEmbedLookup();
        // forwardLogits is forward-only (no backward), so when
        // attn_use_fa the per-layer attention takes the single
        // fa_forward dispatch instead of the 3-pass chain.
        for (0..cfg.n_layers) |li| try self.recordLayerForward(@intCast(li), true);
        // final RMSNorm + lm_head matmul (without the loss-grad).
        const last_idx = cfg.n_layers - 1;
        try self.rec.dispatch(
            &self.k_rms,
            &.{ &self.buf_y[last_idx], &self.buf_w_final_norm, &self.buf_final_norm_out },
            &self.push_rms,
            cfg.n_pos,
            1,
            1,
        );
        try self.rec.dispatch(
            &self.k_matmul,
            &.{ &self.buf_final_norm_out, &self.buf_w_lm_head, &self.buf_logits },
            &self.push_mm_lm_head,
            cfg.n_pos * cfg.vocab_size,
            1,
            1,
        );
        try self.rec.endAndSubmit();
        try self.buf_logits.readBack(self.ctx, f32, out_logits);
    }

    // ── Checkpoint save/load (chunk 8c-β-6b) ──────────────────────────
    //
    // Persist + restore complete training state: all params, all Adam
    // m/v, the step_t counter, and the cfg snapshot. File format is
    // positional fp32 (no per-tensor names) — see `collectBlobs` for
    // the canonical ordering. Skips activations, dW partials, scratch,
    // pipelines, and the recorder — all rebuilt from params + the next
    // forward/backward pass.

    // PERF NOTE — checkpoint save floor (measured 2026-05-08):
    // Save at Qwen3-0.6B (8.40 GiB fp32) takes ~46-52 s on an RTX 3090.
    // The bottleneck is reading from HOST_VISIBLE staging memory back
    // to host RAM during the per-blob copy: on systems without ReBAR /
    // SAM enabled, NVIDIA host-visible mappings above ~256 MiB fall
    // into a slow PCIe-BAR fallback path. We tried pooling all
    // transfers through one 622 MiB persistent staging buffer (which
    // would amortise allocation cost) but it specifically *hits* the
    // slow path and got a wash. The current per-blob `readBack` /
    // `uploadFromHost` shape keeps individual allocations small enough
    // to stay in the fast BAR1 window for most blobs. A buffered
    // writer/reader wraps the file to coalesce per-blob writes into
    // 4 MiB-aligned sequential I/O, which is a small unambiguous win
    // (~3-5 s shaved across the 933 blobs).
    //
    // Real fixes would be: (a) async overlap of GPU copy and file
    // write; (b) save in bf16 / quantized form (changes the format);
    // (c) chunked staging pool tuned to fit in BAR1. Defer until a
    // real workflow makes save time hurt — single-checkpoint cost is
    // currently a one-time tax per fine-tune session.

    pub fn saveCheckpoint(self: *Runner, allocator: std.mem.Allocator, path: []const u8) !void {
        const blobs = try self.collectBlobs(allocator);
        defer allocator.free(blobs);

        const file = try std.fs.cwd().createFile(path, .{ .truncate = true });
        defer file.close();

        const SaveWriter = std.io.BufferedWriter(checkpoint_io_buf_bytes, std.fs.File.Writer);
        var bw = SaveWriter{ .unbuffered_writer = file.writer() };
        const w = bw.writer();

        const header = CheckpointHeader{
            .magic = checkpoint_magic,
            .version = checkpoint_version,
            .step_t = self.step_t,
            .cfg_size = @sizeOf(Config),
        };
        try w.writeAll(std.mem.asBytes(&header));
        try w.writeAll(std.mem.asBytes(&self.cfg));

        var scratch = std.ArrayList(f32).init(allocator);
        defer scratch.deinit();

        for (blobs) |b| {
            try scratch.resize(b.numel);
            try b.buf.readBack(self.ctx, f32, scratch.items);
            try w.writeAll(std.mem.sliceAsBytes(scratch.items));
        }
        try bw.flush();
    }

    pub fn loadCheckpoint(self: *Runner, allocator: std.mem.Allocator, path: []const u8) !void {
        const blobs = try self.collectBlobs(allocator);
        defer allocator.free(blobs);

        const file = try std.fs.cwd().openFile(path, .{ .mode = .read_only });
        defer file.close();

        const LoadReader = std.io.BufferedReader(checkpoint_io_buf_bytes, std.fs.File.Reader);
        var br = LoadReader{ .unbuffered_reader = file.reader() };
        const r = br.reader();

        var header: CheckpointHeader = undefined;
        try r.readNoEof(std.mem.asBytes(&header));
        if (!std.mem.eql(u8, &header.magic, &checkpoint_magic)) return error.InvalidCheckpointMagic;
        if (header.version != checkpoint_version) return error.UnsupportedCheckpointVersion;
        if (header.cfg_size != @sizeOf(Config)) return error.CheckpointConfigSizeMismatch;

        var saved_cfg: Config = undefined;
        try r.readNoEof(std.mem.asBytes(&saved_cfg));
        if (!cfgShapeMatches(saved_cfg, self.cfg)) return error.CheckpointConfigMismatch;

        // Restore optimizer state. The saved cfg's lr/beta/eps_adam are
        // the snapshot's source of truth; caller can override post-load.
        self.step_t = header.step_t;
        self.cfg.lr = saved_cfg.lr;
        self.cfg.beta1 = saved_cfg.beta1;
        self.cfg.beta2 = saved_cfg.beta2;
        self.cfg.eps_adam = saved_cfg.eps_adam;

        var scratch = std.ArrayList(f32).init(allocator);
        defer scratch.deinit();

        for (blobs) |b| {
            try scratch.resize(b.numel);
            try r.readNoEof(std.mem.sliceAsBytes(scratch.items));
            try b.buf.uploadFromHost(self.ctx, f32, scratch.items);
        }
    }

    // ── LoRA-only checkpoint (chunk A4-4) ─────────────────────────────
    //
    // Persists per-projection-per-layer A / B / Adam m/v for every
    // enabled target. Skips base W's (still on disk in the source
    // safetensors) and skips RMSNorm gains, embed, lm_head, final_norm
    // (not LoRA targets — those are full-weight params even in LoRA
    // mode, so saving them would defeat the LoRA-checkpoint-is-tiny
    // story; if a workload wants them too, use the full `.vkpt` save).
    //
    // Body order is identical to the in-memory iteration order: outer
    // loop over Proj.q / .k / .v / .o / .gate / .up / .down (matches
    // `LoraTarget`'s bit layout), inner loop over layers 0..n_layers,
    // each inner step writes (A, m_A, v_A, B, m_B, v_B). Disabled
    // projections contribute zero bytes and are skipped on both ends.
    //
    // Returns error.LoraCheckpointEmpty if the Runner has no LoRA
    // state (cfg.lora_targets = 0); calling save here is meaningless
    // and almost certainly a caller bug.

    pub fn saveLoraCheckpoint(self: *Runner, allocator: std.mem.Allocator, path: []const u8) !void {
        if (self.lora == null) return error.LoraCheckpointEmpty;

        const blobs = try self.collectLoraBlobs(allocator);
        defer allocator.free(blobs);

        const file = try std.fs.cwd().createFile(path, .{ .truncate = true });
        defer file.close();

        const SaveWriter = std.io.BufferedWriter(checkpoint_io_buf_bytes, std.fs.File.Writer);
        var bw = SaveWriter{ .unbuffered_writer = file.writer() };
        const w = bw.writer();

        const header = CheckpointHeader{
            .magic = lora_checkpoint_magic,
            .version = lora_checkpoint_version,
            .step_t = self.step_t,
            .cfg_size = @sizeOf(Config),
        };
        try w.writeAll(std.mem.asBytes(&header));
        try w.writeAll(std.mem.asBytes(&self.cfg));

        var scratch = std.ArrayList(f32).init(allocator);
        defer scratch.deinit();

        for (blobs) |b| {
            try scratch.resize(b.numel);
            try b.buf.readBack(self.ctx, f32, scratch.items);
            try w.writeAll(std.mem.sliceAsBytes(scratch.items));
        }
        try bw.flush();
    }

    pub fn loadLoraCheckpoint(self: *Runner, allocator: std.mem.Allocator, path: []const u8) !void {
        if (self.lora == null) return error.LoraCheckpointEmpty;

        const blobs = try self.collectLoraBlobs(allocator);
        defer allocator.free(blobs);

        const file = try std.fs.cwd().openFile(path, .{ .mode = .read_only });
        defer file.close();

        const LoadReader = std.io.BufferedReader(checkpoint_io_buf_bytes, std.fs.File.Reader);
        var br = LoadReader{ .unbuffered_reader = file.reader() };
        const r = br.reader();

        var header: CheckpointHeader = undefined;
        try r.readNoEof(std.mem.asBytes(&header));
        if (!std.mem.eql(u8, &header.magic, &lora_checkpoint_magic)) return error.InvalidLoraCheckpointMagic;
        if (header.version != lora_checkpoint_version) return error.UnsupportedLoraCheckpointVersion;
        if (header.cfg_size != @sizeOf(Config)) return error.LoraCheckpointConfigSizeMismatch;

        var saved_cfg: Config = undefined;
        try r.readNoEof(std.mem.asBytes(&saved_cfg));
        if (!cfgShapeMatches(saved_cfg, self.cfg)) return error.LoraCheckpointConfigMismatch;

        // step_t + Adam hyperparameters round-trip from the snapshot;
        // base-arch lr/beta/eps already match cfgShapeMatches's
        // structural fields. Caller can override lr post-load.
        self.step_t = header.step_t;
        self.cfg.lr = saved_cfg.lr;
        self.cfg.beta1 = saved_cfg.beta1;
        self.cfg.beta2 = saved_cfg.beta2;
        self.cfg.eps_adam = saved_cfg.eps_adam;
        self.cfg.lora_alpha = saved_cfg.lora_alpha;
        self.cfg.lora_lr_b_scale = saved_cfg.lora_lr_b_scale;

        var scratch = std.ArrayList(f32).init(allocator);
        defer scratch.deinit();

        for (blobs) |b| {
            try scratch.resize(b.numel);
            try r.readNoEof(std.mem.sliceAsBytes(scratch.items));
            try b.buf.uploadFromHost(self.ctx, f32, scratch.items);
        }
    }

    fn collectLoraBlobs(self: *Runner, allocator: std.mem.Allocator) ![]Blob {
        var out = std.ArrayList(Blob).init(allocator);
        errdefer out.deinit();

        if (self.lora) |*ls| {
            inline for (@typeInfo(Proj).@"enum".fields) |f| {
                const p: Proj = @enumFromInt(f.value);
                if (ls.slots[@intFromEnum(p)]) |*slot| {
                    const a_numel: usize = @as(usize, slot.shape.r) * @as(usize, slot.shape.K);
                    const b_numel: usize = @as(usize, slot.shape.N) * @as(usize, slot.shape.r);
                    for (0..self.cfg.n_layers) |i| {
                        try out.appendSlice(&[_]Blob{
                            .{ .buf = &slot.a[i], .numel = a_numel },
                            .{ .buf = &slot.m_a[i], .numel = a_numel },
                            .{ .buf = &slot.v_a[i], .numel = a_numel },
                            .{ .buf = &slot.b[i], .numel = b_numel },
                            .{ .buf = &slot.m_b[i], .numel = b_numel },
                            .{ .buf = &slot.v_b[i], .numel = b_numel },
                        });
                    }
                }
            }
        }

        return try out.toOwnedSlice();
    }

    /// Build the canonical (param, m, v) ordered list of every device
    /// buffer that needs to round-trip through a checkpoint. Both save
    /// and load walk this list in the same order.
    fn collectBlobs(self: *Runner, allocator: std.mem.Allocator) ![]Blob {
        var out = std.ArrayList(Blob).init(allocator);
        errdefer out.deinit();

        const cfg = self.cfg;
        const vocab: usize = @intCast(cfg.vocab_size);
        const dim: usize = @intCast(cfg.dim);
        const ff_dim: usize = @intCast(cfg.ff_dim);
        const head_dim: usize = @intCast(cfg.head_dim);
        const q_dim: usize = @as(usize, @intCast(cfg.n_heads)) * @as(usize, @intCast(cfg.head_dim));
        const kv_dim: usize = @as(usize, @intCast(cfg.n_kv_heads)) * @as(usize, @intCast(cfg.head_dim));

        // Stack-level (param, m, v).
        try out.appendSlice(&[_]Blob{
            .{ .buf = &self.buf_w_embed, .numel = vocab * dim },
            .{ .buf = &self.buf_m_embed, .numel = vocab * dim },
            .{ .buf = &self.buf_v_embed, .numel = vocab * dim },
            .{ .buf = &self.buf_w_final_norm, .numel = dim },
            .{ .buf = &self.buf_m_final_norm, .numel = dim },
            .{ .buf = &self.buf_v_final_norm, .numel = dim },
            .{ .buf = &self.buf_w_lm_head, .numel = vocab * dim },
            .{ .buf = &self.buf_m_lm_head, .numel = vocab * dim },
            .{ .buf = &self.buf_v_lm_head, .numel = vocab * dim },
        });

        // Per-layer.
        for (0..cfg.n_layers) |i| {
            try out.appendSlice(&[_]Blob{
                .{ .buf = &self.buf_w_n1[i], .numel = dim },
                .{ .buf = &self.buf_m_n1[i], .numel = dim },
                .{ .buf = &self.buf_v_n1[i], .numel = dim },
                .{ .buf = &self.buf_w_q[i], .numel = q_dim * dim },
                .{ .buf = &self.buf_m_q[i], .numel = q_dim * dim },
                .{ .buf = &self.buf_v_q[i], .numel = q_dim * dim },
                .{ .buf = &self.buf_w_k[i], .numel = kv_dim * dim },
                .{ .buf = &self.buf_m_k[i], .numel = kv_dim * dim },
                .{ .buf = &self.buf_v_k[i], .numel = kv_dim * dim },
                .{ .buf = &self.buf_w_v[i], .numel = kv_dim * dim },
                .{ .buf = &self.buf_m_v[i], .numel = kv_dim * dim },
                .{ .buf = &self.buf_v_v[i], .numel = kv_dim * dim },
                .{ .buf = &self.buf_w_o[i], .numel = dim * q_dim },
                .{ .buf = &self.buf_m_o[i], .numel = dim * q_dim },
                .{ .buf = &self.buf_v_o[i], .numel = dim * q_dim },
                .{ .buf = &self.buf_w_n2[i], .numel = dim },
                .{ .buf = &self.buf_m_n2[i], .numel = dim },
                .{ .buf = &self.buf_v_n2[i], .numel = dim },
                .{ .buf = &self.buf_w_gate[i], .numel = ff_dim * dim },
                .{ .buf = &self.buf_m_gate[i], .numel = ff_dim * dim },
                .{ .buf = &self.buf_v_gate[i], .numel = ff_dim * dim },
                .{ .buf = &self.buf_w_up[i], .numel = ff_dim * dim },
                .{ .buf = &self.buf_m_up[i], .numel = ff_dim * dim },
                .{ .buf = &self.buf_v_up[i], .numel = ff_dim * dim },
                .{ .buf = &self.buf_w_down[i], .numel = dim * ff_dim },
                .{ .buf = &self.buf_m_down[i], .numel = dim * ff_dim },
                .{ .buf = &self.buf_v_down[i], .numel = dim * ff_dim },
            });
            if (cfg.qk_norm) {
                try out.appendSlice(&[_]Blob{
                    .{ .buf = &self.buf_w_q_norm[i], .numel = head_dim },
                    .{ .buf = &self.buf_m_q_norm[i], .numel = head_dim },
                    .{ .buf = &self.buf_v_q_norm[i], .numel = head_dim },
                    .{ .buf = &self.buf_w_k_norm[i], .numel = head_dim },
                    .{ .buf = &self.buf_m_k_norm[i], .numel = head_dim },
                    .{ .buf = &self.buf_v_k_norm[i], .numel = head_dim },
                });
            }
        }

        return try out.toOwnedSlice();
    }

    // ── Internal recording helpers ────────────────────────────────────

    // ── Per-projection record helpers (A4-3) ──────────────────────────
    //
    // A projection's path is one of two shapes: plain dense matmul +
    // backward (as the wired-in path always was), or LoRA-augmented
    // (W frozen, A/B trained — see `LoraState.allocAndInit` for the
    // shape derivation). These three helpers decide per call.
    //
    // The plain branches mirror the original inline dispatches exactly,
    // so when `cfg.lora_targets = 0` the recorded cmdbuf is bit-equal
    // to the pre-A4-2 path. The LoRA branches consume the matching
    // `LoraState.slots[Proj]` and the shared scratches.

    fn recordProjForward(
        self: *Runner,
        proj: Proj,
        li: u32,
        x: *const buffer.Buffer,
        w: *const buffer.Buffer,
        y: *const buffer.Buffer,
        push: *const runtime.MatmulPush,
    ) !void {
        const i: usize = @intCast(li);
        if (self.lora) |*ls| {
            if (ls.slots[@intFromEnum(proj)]) |*slot| {
                try lora_helpers.recordLoraForward(
                    &self.rec,
                    ls.kernels(self),
                    .{
                        .x = x,
                        .w = w,
                        .a = &slot.a[i],
                        .b = &slot.b[i],
                        .y = y,
                        .intermediate_out = &slot.intermediate[i],
                        .sc_y_lora = &ls.sc_y_lora,
                        .sc_y_lora_scaled = &ls.sc_y_lora_scaled,
                    },
                    slot.shape,
                );
                return;
            }
        }
        try self.rec.dispatch(&self.k_matmul, &.{ x, w, y }, push, self.cfg.n_pos * push.n, 1, 1);
    }

    fn recordProjBackward(
        self: *Runner,
        proj: Proj,
        li: u32,
        dy: *const buffer.Buffer,
        x: *const buffer.Buffer,
        w: *const buffer.Buffer,
        dx: *const buffer.Buffer,
        dw: *const buffer.Buffer,
        push: *const runtime.LinearBatchedPush,
    ) !void {
        const i: usize = @intCast(li);
        if (self.lora) |*ls| {
            if (ls.slots[@intFromEnum(proj)]) |*slot| {
                try lora_helpers.recordLoraBackward(
                    &self.rec,
                    ls.kernels(self),
                    .{
                        .dy = dy,
                        .x = x,
                        .w = w,
                        .a = &slot.a[i],
                        .b = &slot.b[i],
                        .intermediate = &slot.intermediate[i],
                        .dx = dx,
                        .dA = &slot.dw_a[i],
                        .dB = &slot.dw_b[i],
                        .sc_dy_B = &ls.sc_dy_B,
                        .sc_dx_lora = &ls.sc_dx_lora,
                        .sc_dx_lora_scaled = &ls.sc_dx_lora_scaled,
                        .sc_dA_unscaled = &ls.sc_dA_unscaled,
                        .sc_dB_unscaled = &ls.sc_dB_unscaled,
                    },
                    slot.shape,
                );
                return;
            }
        }
        try self.rec.dispatch(&self.k_lin_dx, &.{ dy, w, dx }, push, util.ceilDiv(push.M, group_lwg), util.ceilDiv(push.K, group_lwg), 1);
        try self.rec.dispatch(&self.k_lin_dw, &.{ dy, x, dw }, push, util.ceilDiv(push.N, group_lwg), util.ceilDiv(push.K, group_lwg), 1);
    }

    fn recordProjAdam(
        self: *Runner,
        proj: Proj,
        li: u32,
        w: *const buffer.Buffer,
        dw: *const buffer.Buffer,
        m: *const buffer.Buffer,
        v: *const buffer.Buffer,
        push: *const runtime.AdamStepPush,
    ) !void {
        const i: usize = @intCast(li);
        if (self.lora) |*ls| {
            if (ls.slots[@intFromEnum(proj)]) |*slot| {
                const n_a: u32 = @intCast(slot.a[i].bytes / @sizeOf(f32));
                const n_b: u32 = @intCast(slot.b[i].bytes / @sizeOf(f32));
                const adam_a = runtime.AdamStepPush{ .n = n_a, .lr = push.lr, .beta1 = push.beta1, .beta2 = push.beta2, .eps = push.eps, .t = push.t };
                const adam_b = runtime.AdamStepPush{ .n = n_b, .lr = push.lr * self.cfg.lora_lr_b_scale, .beta1 = push.beta1, .beta2 = push.beta2, .eps = push.eps, .t = push.t };
                try self.rec.dispatch(&self.k_adam, &.{ &slot.a[i], &slot.dw_a[i], &slot.m_a[i], &slot.v_a[i] }, &adam_a, util.ceilDiv(n_a, group_lin), 1, 1);
                try self.rec.dispatch(&self.k_adam, &.{ &slot.b[i], &slot.dw_b[i], &slot.m_b[i], &slot.v_b[i] }, &adam_b, util.ceilDiv(n_b, group_lin), 1, 1);
                return;
            }
            // LoRA is globally active but this projection isn't in
            // `lora_targets`. Skip Adam entirely — `.lvkpt` doesn't
            // persist this W's state, so any drift wouldn't survive
            // a checkpoint round-trip. Net behaviour: every dense W
            // stays at its safetensors value across the LoRA fine-tune.
            return;
        }
        try self.rec.dispatch(&self.k_adam, &.{ w, dw, m, v }, push, util.ceilDiv(push.n, group_lin), 1, 1);
    }

    fn recordEmbedLookup(self: *Runner) !void {
        const total: u32 = self.cfg.n_pos * self.cfg.dim;
        try self.rec.dispatch(
            &self.k_embed,
            &.{ &self.buf_w_embed, &self.buf_token_ids, &self.buf_x_emb },
            &self.push_embed,
            util.ceilDiv(total, group_lin),
            1,
            1,
        );
    }

    fn recordLayerForward(self: *Runner, li: u32, forward_only: bool) !void {
        const cfg = self.cfg;
        const x_in_buf: *const buffer.Buffer = if (li == 0) &self.buf_x_emb else &self.buf_y[li - 1];
        const i: usize = @intCast(li);

        // 1. RMSNorm n1.
        try self.rec.dispatch(&self.k_rms, &.{ x_in_buf, &self.buf_w_n1[i], &self.buf_n1[i] }, &self.push_rms, cfg.n_pos, 1, 1);
        // 2-4. Q/K/V matmuls. The Q/K matmul destination depends on
        // which transforms come after:
        //   qk_norm + rope: matmul → q_pre_norm, rmsnorm → sc_q_pre, rope → buf_q
        //   qk_norm only:   matmul → q_pre_norm, rmsnorm → buf_q
        //   rope only:      matmul → sc_q_pre,                       rope → buf_q
        //   neither:        matmul → buf_q
        const q_matmul_dst: *const buffer.Buffer = if (cfg.qk_norm)
            &self.buf_q_pre_norm[i]
        else if (cfg.rotary_dim > 0)
            &self.sc_q_pre
        else
            &self.buf_q[i];
        const k_matmul_dst: *const buffer.Buffer = if (cfg.qk_norm)
            &self.buf_k_pre_norm[i]
        else if (cfg.rotary_dim > 0)
            &self.sc_k_pre
        else
            &self.buf_k[i];
        // Each of Q/K/V/O/gate/up/down may be LoRA-augmented or plain
        // dense matmul depending on `cfg.lora_targets`. The LoRA chain
        // writes the same destination buffer so downstream paths
        // (qk_norm, RoPE, attention, swiglu, etc.) see an identical
        // buffer layout — only the per-element value picks up the
        // `(α/r)·B·A·x` term when LoRA is on.
        try self.recordProjForward(.q, li, &self.buf_n1[i], &self.buf_w_q[i], q_matmul_dst, &self.push_mm_q);
        try self.recordProjForward(.k, li, &self.buf_n1[i], &self.buf_w_k[i], k_matmul_dst, &self.push_mm_k);
        try self.recordProjForward(.v, li, &self.buf_n1[i], &self.buf_w_v[i], &self.buf_v[i], &self.push_mm_v);
        if (cfg.qk_norm) {
            // rmsnorm dispatches: one workgroup per row, n_rows = n_pos * n_heads
            // (Q) or n_pos * n_kv_heads (K), dim = head_dim. The rmsnorm
            // shader takes n_rows as its X dispatch count.
            const q_dst: *const buffer.Buffer = if (cfg.rotary_dim > 0) &self.sc_q_pre else &self.buf_q[i];
            const k_dst: *const buffer.Buffer = if (cfg.rotary_dim > 0) &self.sc_k_pre else &self.buf_k[i];
            try self.rec.dispatch(&self.k_rms, &.{ &self.buf_q_pre_norm[i], &self.buf_w_q_norm[i], q_dst }, &self.push_rms_qk, cfg.n_pos * cfg.n_heads, 1, 1);
            try self.rec.dispatch(&self.k_rms, &.{ &self.buf_k_pre_norm[i], &self.buf_w_k_norm[i], k_dst }, &self.push_rms_qk, cfg.n_pos * cfg.n_kv_heads, 1, 1);
        }
        if (cfg.rotary_dim > 0) {
            // Fused QK-RoPE: one dispatch over n_pos × (n_heads + n_kv_heads) × head_dim.
            const qk_total: u32 = cfg.n_pos * (cfg.n_heads + cfg.n_kv_heads) * cfg.head_dim;
            try self.rec.dispatch(&self.k_rope_fwd, &.{ &self.sc_q_pre, &self.buf_q[i], &self.sc_k_pre, &self.buf_k[i] }, &self.push_rope_qk, util.ceilDiv(qk_total, group_lin), 1, 1);
        }
        // 5-7. Attention. Two paths:
        //   - attn_use_fa : single fa_forward dispatch. Writes
        //     `buf_attn_out[i]`; emits `buf_fa_lse` when
        //     `forward_only == false` (training) so the FA-2 backward
        //     can recompute the softmax inline. Skips materialising
        //     the [n_pos × n_heads × n_pos] `buf_attn[i]` entirely.
        //   - otherwise : the 3-pass chain (attn_scores → softmax →
        //     attn_output) that saves `buf_attn[i]` for backward —
        //     the head_dim > 128 fallback.
        if (self.attn_use_fa) {
            const fa_push: *const runtime.FaForwardPush = if (forward_only)
                &self.push_fa_forward
            else
                &self.push_fa_forward_train;
            try self.rec.dispatch(
                &self.k_fa_forward,
                &.{ &self.buf_q[i], &self.buf_k[i], &self.buf_v[i], &self.buf_attn_out[i], &self.buf_fa_lse[i] },
                fa_push,
                cfg.n_pos * cfg.n_heads,
                1,
                1,
            );
        } else {
            // 5. attention scores (causal mask via -inf).
            try self.rec.dispatch(&self.k_attn_scores, &.{ &self.buf_q[i], &self.buf_k[i], &self.sc_scores }, &self.push_attn_scores, cfg.n_pos * cfg.n_heads * cfg.n_pos, 1, 1);
            // 6. softmax.
            try self.rec.dispatch(&self.k_softmax, &.{ &self.sc_scores, &self.buf_attn[i] }, &self.push_softmax, cfg.n_pos * cfg.n_heads, 1, 1);
            // 7. attention output.
            try self.rec.dispatch(&self.k_attn_output, &.{ &self.buf_attn[i], &self.buf_v[i], &self.buf_attn_out[i] }, &self.push_attn_output, cfg.n_pos * cfg.n_heads * cfg.head_dim, 1, 1);
        }
        // 8. O projection (LoRA-aware).
        try self.recordProjForward(.o, li, &self.buf_attn_out[i], &self.buf_w_o[i], &self.sc_o, &self.push_mm_o);
        // 9. mid = x_in + o (residual).
        const add_groups: u32 = util.ceilDiv(self.push_n_pos_dim.n, group_lin);
        try self.rec.dispatch(&self.k_vec_add, &.{ x_in_buf, &self.sc_o, &self.buf_mid[i] }, &self.push_n_pos_dim, add_groups, 1, 1);
        // 10. RMSNorm n2.
        try self.rec.dispatch(&self.k_rms, &.{ &self.buf_mid[i], &self.buf_w_n2[i], &self.buf_n2[i] }, &self.push_rms, cfg.n_pos, 1, 1);
        // 11. W_gate matmul (LoRA-aware).
        try self.recordProjForward(.gate, li, &self.buf_n2[i], &self.buf_w_gate[i], &self.buf_pre_gate[i], &self.push_mm_gate);
        // 12. W_up matmul (LoRA-aware).
        try self.recordProjForward(.up, li, &self.buf_n2[i], &self.buf_w_up[i], &self.buf_up[i], &self.push_mm_up);
        // 13. SwiGLU fwd: gated = silu(pre_gate) · up.
        const swiglu_groups: u32 = util.ceilDiv(self.push_swiglu.n, group_lin);
        try self.rec.dispatch(&self.k_swiglu_fwd, &.{ &self.buf_pre_gate[i], &self.buf_up[i], &self.buf_gated[i] }, &self.push_swiglu, swiglu_groups, 1, 1);
        // 14. W_down matmul → ff_out (LoRA-aware).
        try self.recordProjForward(.down, li, &self.buf_gated[i], &self.buf_w_down[i], &self.sc_ff_out, &self.push_mm_down);
        // 15. y = mid + ff_out.
        try self.rec.dispatch(&self.k_vec_add, &.{ &self.buf_mid[i], &self.sc_ff_out, &self.buf_y[i] }, &self.push_n_pos_dim, add_groups, 1, 1);
    }

    fn recordHeadForwardAndLossGrad(self: *Runner) !void {
        const cfg = self.cfg;
        const last_idx = cfg.n_layers - 1;
        try self.rec.dispatch(
            &self.k_rms,
            &.{ &self.buf_y[last_idx], &self.buf_w_final_norm, &self.buf_final_norm_out },
            &self.push_rms,
            cfg.n_pos,
            1,
            1,
        );
        // CCE forward: one workgroup per row, fuses h · W_lm^T with
        // chunked online-softmax CE. No [n_pos, vocab] logit tensor;
        // outputs are [n_pos] f32 lse (cached for backward) and
        // [n_pos] f32 per-row loss (host averages for total loss).
        try self.rec.dispatch(
            &self.k_cce_forward,
            &.{
                &self.buf_final_norm_out,
                &self.buf_w_lm_head,
                &self.buf_target_ids,
                &self.buf_lse,
                &self.buf_loss_per_row,
            },
            &self.push_cce,
            cfg.n_pos,
            1,
            1,
        );
    }

    fn recordHeadBackward(self: *Runner) !void {
        const cfg = self.cfg;
        // CCE backward d_h: one WG per row, recomputes z_chunk on the fly,
        // derives dz from cached lse, accumulates d_final_norm_out without
        // ever materializing d_logits. Output is overwritten (no fillZero
        // required).
        try self.rec.dispatch(
            &self.k_cce_backward_dh,
            &.{
                &self.buf_final_norm_out,
                &self.buf_w_lm_head,
                &self.buf_target_ids,
                &self.buf_lse,
                &self.buf_d_final_norm_out,
            },
            &self.push_cce,
            cfg.n_pos,
            1,
            1,
        );
        // CCE backward dW: one WG per vocab entry (mirrors embedding_backward's
        // vocab-major no-atomic layout). Loops over rows internally, accumulates
        // into buf_dw_lm_head — caller zeroes via fillZero at top of step().
        try self.rec.dispatch(
            &self.k_cce_backward_dw,
            &.{
                &self.buf_final_norm_out,
                &self.buf_w_lm_head,
                &self.buf_target_ids,
                &self.buf_lse,
                &self.buf_dw_lm_head,
            },
            &self.push_cce,
            cfg.vocab_size,
            1,
            1,
        );
        // final RMSNorm backward.
        const last_idx = self.cfg.n_layers - 1;
        try self.rec.dispatch(
            &self.k_rms_bw,
            &.{ &self.buf_d_final_norm_out, &self.buf_y[last_idx], &self.buf_w_final_norm, &self.buf_d_last_y, &self.buf_dw_final_norm_partial },
            &self.push_rms,
            self.cfg.n_pos,
            1,
            1,
        );
    }

    fn recordLayerBackward(self: *Runner, li: u32) !void {
        const cfg = self.cfg;
        const i: usize = @intCast(li);
        const last = cfg.n_layers - 1;
        const d_y_in: *const buffer.Buffer = if (li == last) &self.buf_d_last_y else &self.buf_d_x_in[i + 1];
        const d_x_in_out = &self.buf_d_x_in[i];
        const x_in_buf: *const buffer.Buffer = if (li == 0) &self.buf_x_emb else &self.buf_y[i - 1];

        const add_groups: u32 = util.ceilDiv(self.push_n_pos_dim.n, group_lin);
        const swiglu_groups: u32 = util.ceilDiv(self.push_swiglu.n, group_lin);

        // W_down backward (LoRA-aware: when LoRA is on .down, dW_down
        // is replaced by ∇A_down / ∇B_down and dx still flows through
        // the (α/r)·dy_B·A path).
        try self.recordProjBackward(.down, li, d_y_in, &self.buf_gated[i], &self.buf_w_down[i], &self.sc_d_gated, &self.buf_dw_down[i], &self.push_lin_down);
        // SwiGLU bw: (d_gated, pre_gate, up) → (d_pre_gate, d_up).
        try self.rec.dispatch(&self.k_swiglu_bw, &.{ &self.sc_d_gated, &self.buf_pre_gate[i], &self.buf_up[i], &self.sc_d_pre_gate, &self.sc_d_up_grad }, &self.push_swiglu, swiglu_groups, 1, 1);
        // W_gate backward (writes d_n2).
        try self.recordProjBackward(.gate, li, &self.sc_d_pre_gate, &self.buf_n2[i], &self.buf_w_gate[i], &self.sc_d_n2, &self.buf_dw_gate[i], &self.push_lin_gate);
        // W_up backward + accumulate into d_n2.
        try self.recordProjBackward(.up, li, &self.sc_d_up_grad, &self.buf_n2[i], &self.buf_w_up[i], &self.sc_d_n2_up, &self.buf_dw_up[i], &self.push_lin_up);
        try self.rec.dispatch(&self.k_add, &.{ &self.sc_d_n2, &self.sc_d_n2_up }, &self.push_n_pos_dim, add_groups, 1, 1);
        // RMSNorm n2 bw → d_mid_norm + dw_n2_partial.
        try self.rec.dispatch(&self.k_rms_bw, &.{ &self.sc_d_n2, &self.buf_mid[i], &self.buf_w_n2[i], &self.sc_d_mid_norm, &self.buf_dw_n2_partial[i] }, &self.push_rms, cfg.n_pos, 1, 1);
        // d_y_in += d_mid_norm. From here, d_y_in holds d_mid_total.
        try self.rec.dispatch(&self.k_add, &.{ d_y_in, &self.sc_d_mid_norm }, &self.push_n_pos_dim, add_groups, 1, 1);
        // O projection backward (LoRA-aware).
        try self.recordProjBackward(.o, li, d_y_in, &self.buf_attn_out[i], &self.buf_w_o[i], &self.sc_d_attn_out, &self.buf_dw_o[i], &self.push_lin_o);
        // SDPA backward. Two paths matching the forward gate:
        //   - attn_use_fa : 3-kernel FA-2 backward (D + dQ + dKV).
        //     Recomputes the softmax inline from the saved LSE, so
        //     `buf_attn[i]` is never read on this path.
        //   - otherwise   : 5-kernel 3-pass chain (head_dim > 128).
        if (self.attn_use_fa) {
            // Phase 1: D[q, h] = Σ_d O · dO. Reads buf_attn_out (the
            // FA-forward `out`) and sc_d_attn_out (the gradient flowing
            // in from the O-projection backward).
            try self.rec.dispatch(&self.k_fa_bw_d, &.{ &self.buf_attn_out[i], &self.sc_d_attn_out, &self.buf_fa_d[i] }, &self.push_fa_bw_d, cfg.n_pos * cfg.n_heads, 1, 1);
            // Phase 2: per-(q, h) dQ accumulation, tile-on-K.
            try self.rec.dispatch(&self.k_fa_bw_dq, &.{ &self.buf_q[i], &self.buf_k[i], &self.buf_v[i], &self.sc_d_attn_out, &self.buf_fa_lse[i], &self.buf_fa_d[i], &self.sc_dQ }, &self.push_fa_bw_dq, cfg.n_pos * cfg.n_heads, 1, 1);
            // Phase 3: per-(k, kv_h) dK + dV accumulation, tile-on-Q
            // with the GQA fold over heads_per_kv inside the WG.
            try self.rec.dispatch(&self.k_fa_bw_dkv, &.{ &self.buf_q[i], &self.buf_k[i], &self.buf_v[i], &self.sc_d_attn_out, &self.buf_fa_lse[i], &self.buf_fa_d[i], &self.sc_dK, &self.sc_dV }, &self.push_fa_bw_dkv, cfg.n_pos * cfg.n_kv_heads, 1, 1);
        } else {
            try self.rec.dispatch(&self.k_attn_dattn, &.{ &self.sc_d_attn_out, &self.buf_v[i], &self.sc_d_attn }, &self.push_dattn, cfg.n_pos * cfg.n_heads * cfg.n_pos, 1, 1);
            try self.rec.dispatch(&self.k_attn_dv, &.{ &self.buf_attn[i], &self.sc_d_attn_out, &self.sc_dV }, &self.push_dv, cfg.n_pos * cfg.n_kv_heads * cfg.head_dim, 1, 1);
            try self.rec.dispatch(&self.k_softmax_bw, &.{ &self.sc_d_attn, &self.buf_attn[i], &self.sc_d_scores }, &self.push_softmax, cfg.n_pos * cfg.n_heads, 1, 1);
            try self.rec.dispatch(&self.k_attn_dq, &.{ &self.sc_d_scores, &self.buf_k[i], &self.sc_dQ }, &self.push_dq, cfg.n_pos * cfg.n_heads * cfg.head_dim, 1, 1);
            try self.rec.dispatch(&self.k_attn_dk, &.{ &self.sc_d_scores, &self.buf_q[i], &self.sc_dK }, &self.push_dk, cfg.n_pos * cfg.n_kv_heads * cfg.head_dim, 1, 1);
        }

        // RoPE backward: the dQ / dK from SDPA are gradients of the
        // post-RoPE Q / K. Inverting the rotation lands them as
        // gradients of the pre-RoPE Q / K (i.e. of the rmsnorm
        // output when qk_norm is on, or of the matmul output when
        // it's off).
        if (cfg.rotary_dim > 0) {
            // Fused QK-RoPE backward — one dispatch covers both dQ and dK.
            const qk_total: u32 = self.cfg.n_pos * (self.cfg.n_heads + self.cfg.n_kv_heads) * self.cfg.head_dim;
            try self.rec.dispatch(&self.k_rope_bw, &.{ &self.sc_dQ, &self.sc_dQ_pre, &self.sc_dK, &self.sc_dK_pre }, &self.push_rope_qk, util.ceilDiv(qk_total, group_lin), 1, 1);
        }

        // Q/K-norm backward: rmsnorm_bw needs the pre-norm input
        // (saved in buf_q_pre_norm[i]) plus the upstream gradient
        // (post-RoPE-bw if RoPE is on, else post-attn dQ/dK).
        // Writes a per-row dw_partial that the host reduces in step().
        if (cfg.qk_norm) {
            const dQ_norm_in: *const buffer.Buffer = if (cfg.rotary_dim > 0) &self.sc_dQ_pre else &self.sc_dQ;
            const dK_norm_in: *const buffer.Buffer = if (cfg.rotary_dim > 0) &self.sc_dK_pre else &self.sc_dK;
            try self.rec.dispatch(&self.k_rms_bw, &.{ dQ_norm_in, &self.buf_q_pre_norm[i], &self.buf_w_q_norm[i], &self.sc_dQ_pre_norm, &self.buf_dw_q_norm_partial[i] }, &self.push_rms_qk, cfg.n_pos * cfg.n_heads, 1, 1);
            try self.rec.dispatch(&self.k_rms_bw, &.{ dK_norm_in, &self.buf_k_pre_norm[i], &self.buf_w_k_norm[i], &self.sc_dK_pre_norm, &self.buf_dw_k_norm_partial[i] }, &self.push_rms_qk, cfg.n_pos * cfg.n_kv_heads, 1, 1);
        }

        // Final input to lin_dx/lin_dw of W_q / W_k: the pre-everything
        // dQ / dK. Decision tree:
        //   qk_norm enabled (with or without RoPE): sc_dQ_pre_norm / sc_dK_pre_norm
        //   RoPE only:                              sc_dQ_pre / sc_dK_pre
        //   neither:                                sc_dQ / sc_dK
        const dQ_lin: *const buffer.Buffer = if (cfg.qk_norm)
            &self.sc_dQ_pre_norm
        else if (cfg.rotary_dim > 0)
            &self.sc_dQ_pre
        else
            &self.sc_dQ;
        const dK_lin: *const buffer.Buffer = if (cfg.qk_norm)
            &self.sc_dK_pre_norm
        else if (cfg.rotary_dim > 0)
            &self.sc_dK_pre
        else
            &self.sc_dK;

        // Q/K/V backward (LoRA-aware). Q writes directly into sc_d_n1
        // (saves an add_in_place); K/V write to scratch and add into
        // sc_d_n1.
        try self.recordProjBackward(.q, li, dQ_lin, &self.buf_n1[i], &self.buf_w_q[i], &self.sc_d_n1, &self.buf_dw_q[i], &self.push_lin_q);
        try self.recordProjBackward(.k, li, dK_lin, &self.buf_n1[i], &self.buf_w_k[i], &self.sc_d_n1_k, &self.buf_dw_k[i], &self.push_lin_k);
        try self.rec.dispatch(&self.k_add, &.{ &self.sc_d_n1, &self.sc_d_n1_k }, &self.push_n_pos_dim, add_groups, 1, 1);
        try self.recordProjBackward(.v, li, &self.sc_dV, &self.buf_n1[i], &self.buf_w_v[i], &self.sc_d_n1_v, &self.buf_dw_v[i], &self.push_lin_v);
        try self.rec.dispatch(&self.k_add, &.{ &self.sc_d_n1, &self.sc_d_n1_v }, &self.push_n_pos_dim, add_groups, 1, 1);
        // RMSNorm n1 bw → writes the rmsnorm.dx contribution to d_x_in_out.
        try self.rec.dispatch(&self.k_rms_bw, &.{ &self.sc_d_n1, x_in_buf, &self.buf_w_n1[i], d_x_in_out, &self.buf_dw_n1_partial[i] }, &self.push_rms, cfg.n_pos, 1, 1);
        // d_x_in_out += d_y_in (residual contribution of `mid = x_in + o`).
        try self.rec.dispatch(&self.k_add, &.{ d_x_in_out, d_y_in }, &self.push_n_pos_dim, add_groups, 1, 1);
    }

    fn recordAdamAll(self: *Runner) !void {
        const cfg = self.cfg;
        const dim_n: u32 = cfg.dim;
        const lr = cfg.lr;
        const beta1 = cfg.beta1;
        const beta2 = cfg.beta2;
        const eps = cfg.eps_adam;
        const t = self.step_t;

        const n_embed: u32 = cfg.vocab_size * cfg.dim;
        const n_lm_head: u32 = cfg.vocab_size * cfg.dim;
        const n_q_w: u32 = cfg.n_heads * cfg.head_dim * cfg.dim;
        const n_k_w: u32 = cfg.n_kv_heads * cfg.head_dim * cfg.dim;
        const n_v_w: u32 = cfg.n_kv_heads * cfg.head_dim * cfg.dim;
        const n_o_w: u32 = cfg.dim * cfg.n_heads * cfg.head_dim;
        const n_gate_w: u32 = cfg.ff_dim * cfg.dim;
        const n_up_w: u32 = cfg.ff_dim * cfg.dim;
        const n_down_w: u32 = cfg.dim * cfg.ff_dim;

        // ── In LoRA mode (`lora_targets != 0`), the standard semantics
        //    is to freeze every non-LoRA parameter — only A,B for each
        //    enabled projection move. Without this, .lvkpt round-trip
        //    breaks: a fresh Runner loads only LoRA params, so any
        //    drift in the non-LoRA params (embed, lm_head, RMSNorm
        //    gains) doesn't survive the load. Skipping the Adam
        //    dispatches here also saves a chunk of compute per step.
        const lora_active: bool = (cfg.lora_targets != 0);

        if (!lora_active) {
            const adam_embed = runtime.AdamStepPush{ .n = n_embed, .lr = lr, .beta1 = beta1, .beta2 = beta2, .eps = eps, .t = t };
            try self.rec.dispatch(&self.k_adam, &.{ &self.buf_w_embed, &self.buf_dE_embed, &self.buf_m_embed, &self.buf_v_embed }, &adam_embed, util.ceilDiv(n_embed, group_lin), 1, 1);
        }

        for (0..cfg.n_layers) |i| {
            const adam_n1 = runtime.AdamStepPush{ .n = dim_n, .lr = lr, .beta1 = beta1, .beta2 = beta2, .eps = eps, .t = t };
            const adam_q = runtime.AdamStepPush{ .n = n_q_w, .lr = lr, .beta1 = beta1, .beta2 = beta2, .eps = eps, .t = t };
            const adam_k = runtime.AdamStepPush{ .n = n_k_w, .lr = lr, .beta1 = beta1, .beta2 = beta2, .eps = eps, .t = t };
            const adam_v = runtime.AdamStepPush{ .n = n_v_w, .lr = lr, .beta1 = beta1, .beta2 = beta2, .eps = eps, .t = t };
            const adam_o = runtime.AdamStepPush{ .n = n_o_w, .lr = lr, .beta1 = beta1, .beta2 = beta2, .eps = eps, .t = t };
            const adam_n2 = runtime.AdamStepPush{ .n = dim_n, .lr = lr, .beta1 = beta1, .beta2 = beta2, .eps = eps, .t = t };
            const adam_gate = runtime.AdamStepPush{ .n = n_gate_w, .lr = lr, .beta1 = beta1, .beta2 = beta2, .eps = eps, .t = t };
            const adam_up = runtime.AdamStepPush{ .n = n_up_w, .lr = lr, .beta1 = beta1, .beta2 = beta2, .eps = eps, .t = t };
            const adam_down = runtime.AdamStepPush{ .n = n_down_w, .lr = lr, .beta1 = beta1, .beta2 = beta2, .eps = eps, .t = t };

            if (!lora_active) {
                try self.rec.dispatch(&self.k_adam, &.{ &self.buf_w_n1[i], &self.buf_dw_n1[i], &self.buf_m_n1[i], &self.buf_v_n1[i] }, &adam_n1, util.ceilDiv(dim_n, group_lin), 1, 1);
            }
            // For each of the 7 dense projections: when LoRA is on for
            // that target, recordProjAdam skips the frozen W's Adam and
            // runs Adam on A and B (with B's lr scaled by lora_lr_b_scale,
            // i.e. LoRA+ Theorem 1). Otherwise — and only when not in
            // LoRA-active mode — it runs the plain W Adam step. The
            // RMSNorm gains (n1, n2, q_norm, k_norm) are never LoRA'd
            // and are frozen along with everything else when LoRA is on.
            try self.recordProjAdam(.q, @intCast(i), &self.buf_w_q[i], &self.buf_dw_q[i], &self.buf_m_q[i], &self.buf_v_q[i], &adam_q);
            try self.recordProjAdam(.k, @intCast(i), &self.buf_w_k[i], &self.buf_dw_k[i], &self.buf_m_k[i], &self.buf_v_k[i], &adam_k);
            try self.recordProjAdam(.v, @intCast(i), &self.buf_w_v[i], &self.buf_dw_v[i], &self.buf_m_v[i], &self.buf_v_v[i], &adam_v);
            try self.recordProjAdam(.o, @intCast(i), &self.buf_w_o[i], &self.buf_dw_o[i], &self.buf_m_o[i], &self.buf_v_o[i], &adam_o);
            if (!lora_active) {
                try self.rec.dispatch(&self.k_adam, &.{ &self.buf_w_n2[i], &self.buf_dw_n2[i], &self.buf_m_n2[i], &self.buf_v_n2[i] }, &adam_n2, util.ceilDiv(dim_n, group_lin), 1, 1);
            }
            try self.recordProjAdam(.gate, @intCast(i), &self.buf_w_gate[i], &self.buf_dw_gate[i], &self.buf_m_gate[i], &self.buf_v_gate[i], &adam_gate);
            try self.recordProjAdam(.up, @intCast(i), &self.buf_w_up[i], &self.buf_dw_up[i], &self.buf_m_up[i], &self.buf_v_up[i], &adam_up);
            try self.recordProjAdam(.down, @intCast(i), &self.buf_w_down[i], &self.buf_dw_down[i], &self.buf_m_down[i], &self.buf_v_down[i], &adam_down);
            if (cfg.qk_norm and !lora_active) {
                const adam_qn = runtime.AdamStepPush{ .n = cfg.head_dim, .lr = lr, .beta1 = beta1, .beta2 = beta2, .eps = eps, .t = t };
                try self.rec.dispatch(&self.k_adam, &.{ &self.buf_w_q_norm[i], &self.buf_dw_q_norm[i], &self.buf_m_q_norm[i], &self.buf_v_q_norm[i] }, &adam_qn, util.ceilDiv(cfg.head_dim, group_lin), 1, 1);
                try self.rec.dispatch(&self.k_adam, &.{ &self.buf_w_k_norm[i], &self.buf_dw_k_norm[i], &self.buf_m_k_norm[i], &self.buf_v_k_norm[i] }, &adam_qn, util.ceilDiv(cfg.head_dim, group_lin), 1, 1);
            }
        }

        if (!lora_active) {
            const adam_final_norm = runtime.AdamStepPush{ .n = dim_n, .lr = lr, .beta1 = beta1, .beta2 = beta2, .eps = eps, .t = t };
            const adam_lm_head = runtime.AdamStepPush{ .n = n_lm_head, .lr = lr, .beta1 = beta1, .beta2 = beta2, .eps = eps, .t = t };
            try self.rec.dispatch(&self.k_adam, &.{ &self.buf_w_final_norm, &self.buf_dw_final_norm, &self.buf_m_final_norm, &self.buf_v_final_norm }, &adam_final_norm, util.ceilDiv(dim_n, group_lin), 1, 1);
            try self.rec.dispatch(&self.k_adam, &.{ &self.buf_w_lm_head, &self.buf_dw_lm_head, &self.buf_m_lm_head, &self.buf_v_lm_head }, &adam_lm_head, util.ceilDiv(n_lm_head, group_lin), 1, 1);
        }
    }

    fn reduceDwPartial(self: *Runner, partial: *const buffer.Buffer, dst_dynamic: *buffer.Buffer) !void {
        const dim: usize = @intCast(self.cfg.dim);
        const n_pos: usize = @intCast(self.cfg.n_pos);
        // dw_partial_host is sized for the *largest* partial shape (n_pos *
        // max(dim, n_heads * head_dim, n_kv_heads * head_dim)). Slice it
        // to the actual readback size (n_pos * dim) so we don't try to
        // read past the partial buffer's end.
        const total = n_pos * dim;
        try partial.readBack(self.ctx, f32, self.dw_partial_host[0..total]);
        @memset(self.dw_reduced[0..dim], 0);
        for (0..n_pos) |row| {
            const off = row * dim;
            for (0..dim) |idx| self.dw_reduced[idx] += self.dw_partial_host[off + idx];
        }
        dst_dynamic.update(f32, self.dw_reduced[0..dim]);
    }

    /// Generic per-row reduce: partial is [n_rows × dim_per_row], dst
    /// is [dim_per_row]. Used by Q/K-norm where rows are
    /// (n_pos × n_heads) and dim_per_row is head_dim — different shape
    /// than the main RMSNorm dw partial reducer above.
    fn reduceDwPartialN(
        self: *Runner,
        partial: *const buffer.Buffer,
        dst_dynamic: *buffer.Buffer,
        n_rows: usize,
        dim_per_row: usize,
    ) !void {
        const total = n_rows * dim_per_row;
        // dw_partial_host is sized for the largest partial shape we hit
        // — n_pos * dim. Verify q/k-norm fits there too.
        std.debug.assert(total <= self.dw_partial_host.len);
        std.debug.assert(dim_per_row <= self.dw_reduced.len);
        try partial.readBack(self.ctx, f32, self.dw_partial_host[0..total]);
        @memset(self.dw_reduced[0..dim_per_row], 0);
        for (0..n_rows) |row| {
            const off = row * dim_per_row;
            for (0..dim_per_row) |idx| self.dw_reduced[idx] += self.dw_partial_host[off + idx];
        }
        dst_dynamic.update(f32, self.dw_reduced[0..dim_per_row]);
    }
};

// ── Internal helpers ──────────────────────────────────────────────────

const group_lin: u32 = 256;
const group_lwg: u32 = 16;


/// Bag of per-layer slice arrays used during init. Keeps `init()`
/// readable by isolating the per-layer alloc-and-rollback logic.
const PerLayerArrays = struct {
    n_layers: usize,
    populated: usize,

    w_n1: []buffer.Buffer,
    w_q: []buffer.Buffer,
    w_k: []buffer.Buffer,
    w_v: []buffer.Buffer,
    w_o: []buffer.Buffer,
    w_n2: []buffer.Buffer,
    w_gate: []buffer.Buffer,
    w_up: []buffer.Buffer,
    w_down: []buffer.Buffer,

    a_n1: []buffer.Buffer,
    a_q: []buffer.Buffer,
    a_k: []buffer.Buffer,
    a_v: []buffer.Buffer,
    a_attn: []buffer.Buffer,
    a_attn_out: []buffer.Buffer,
    a_mid: []buffer.Buffer,
    a_n2: []buffer.Buffer,
    a_pre_gate: []buffer.Buffer,
    a_up: []buffer.Buffer,
    a_gated: []buffer.Buffer,
    a_y: []buffer.Buffer,

    dw_n1_partial: []buffer.Buffer,
    dw_q: []buffer.Buffer,
    dw_k: []buffer.Buffer,
    dw_v: []buffer.Buffer,
    dw_o: []buffer.Buffer,
    dw_n2_partial: []buffer.Buffer,
    dw_gate: []buffer.Buffer,
    dw_up: []buffer.Buffer,
    dw_down: []buffer.Buffer,

    dw_n1: []buffer.Buffer,
    dw_n2: []buffer.Buffer,

    m_n1: []buffer.Buffer,
    v_n1: []buffer.Buffer,
    m_q: []buffer.Buffer,
    v_q: []buffer.Buffer,
    m_k: []buffer.Buffer,
    v_k: []buffer.Buffer,
    m_v: []buffer.Buffer,
    v_v: []buffer.Buffer,
    m_o: []buffer.Buffer,
    v_o: []buffer.Buffer,
    m_n2: []buffer.Buffer,
    v_n2: []buffer.Buffer,
    m_gate: []buffer.Buffer,
    v_gate: []buffer.Buffer,
    m_up: []buffer.Buffer,
    v_up: []buffer.Buffer,
    m_down: []buffer.Buffer,
    v_down: []buffer.Buffer,

    // Q/K-norm per-layer (always alloc'd as a slice of `n_layers`
    // entries; the entries themselves stay zero-sized when
    // cfg.qk_norm is false, so nothing else changes shape-wise).
    w_q_norm: []buffer.Buffer,
    w_k_norm: []buffer.Buffer,
    a_q_pre_norm: []buffer.Buffer,
    a_k_pre_norm: []buffer.Buffer,
    dw_q_norm_partial: []buffer.Buffer,
    dw_k_norm_partial: []buffer.Buffer,
    dw_q_norm: []buffer.Buffer,
    dw_k_norm: []buffer.Buffer,
    m_q_norm: []buffer.Buffer,
    v_q_norm: []buffer.Buffer,
    m_k_norm: []buffer.Buffer,
    v_k_norm: []buffer.Buffer,

    d_x_in: []buffer.Buffer,

    fn alloc(allocator: std.mem.Allocator, n_layers: usize, cfg: Config) !PerLayerArrays {
        var pl: PerLayerArrays = undefined;
        pl.n_layers = n_layers;
        pl.populated = 0;
        pl.w_n1 = try allocator.alloc(buffer.Buffer, n_layers);
        pl.w_q = try allocator.alloc(buffer.Buffer, n_layers);
        pl.w_k = try allocator.alloc(buffer.Buffer, n_layers);
        pl.w_v = try allocator.alloc(buffer.Buffer, n_layers);
        pl.w_o = try allocator.alloc(buffer.Buffer, n_layers);
        pl.w_n2 = try allocator.alloc(buffer.Buffer, n_layers);
        pl.w_gate = try allocator.alloc(buffer.Buffer, n_layers);
        pl.w_up = try allocator.alloc(buffer.Buffer, n_layers);
        pl.w_down = try allocator.alloc(buffer.Buffer, n_layers);
        pl.a_n1 = try allocator.alloc(buffer.Buffer, n_layers);
        pl.a_q = try allocator.alloc(buffer.Buffer, n_layers);
        pl.a_k = try allocator.alloc(buffer.Buffer, n_layers);
        pl.a_v = try allocator.alloc(buffer.Buffer, n_layers);
        pl.a_attn = try allocator.alloc(buffer.Buffer, n_layers);
        pl.a_attn_out = try allocator.alloc(buffer.Buffer, n_layers);
        pl.a_mid = try allocator.alloc(buffer.Buffer, n_layers);
        pl.a_n2 = try allocator.alloc(buffer.Buffer, n_layers);
        pl.a_pre_gate = try allocator.alloc(buffer.Buffer, n_layers);
        pl.a_up = try allocator.alloc(buffer.Buffer, n_layers);
        pl.a_gated = try allocator.alloc(buffer.Buffer, n_layers);
        pl.a_y = try allocator.alloc(buffer.Buffer, n_layers);
        pl.dw_n1_partial = try allocator.alloc(buffer.Buffer, n_layers);
        pl.dw_q = try allocator.alloc(buffer.Buffer, n_layers);
        pl.dw_k = try allocator.alloc(buffer.Buffer, n_layers);
        pl.dw_v = try allocator.alloc(buffer.Buffer, n_layers);
        pl.dw_o = try allocator.alloc(buffer.Buffer, n_layers);
        pl.dw_n2_partial = try allocator.alloc(buffer.Buffer, n_layers);
        pl.dw_gate = try allocator.alloc(buffer.Buffer, n_layers);
        pl.dw_up = try allocator.alloc(buffer.Buffer, n_layers);
        pl.dw_down = try allocator.alloc(buffer.Buffer, n_layers);
        pl.dw_n1 = try allocator.alloc(buffer.Buffer, n_layers);
        pl.dw_n2 = try allocator.alloc(buffer.Buffer, n_layers);
        pl.m_n1 = try allocator.alloc(buffer.Buffer, n_layers);
        pl.v_n1 = try allocator.alloc(buffer.Buffer, n_layers);
        pl.m_q = try allocator.alloc(buffer.Buffer, n_layers);
        pl.v_q = try allocator.alloc(buffer.Buffer, n_layers);
        pl.m_k = try allocator.alloc(buffer.Buffer, n_layers);
        pl.v_k = try allocator.alloc(buffer.Buffer, n_layers);
        pl.m_v = try allocator.alloc(buffer.Buffer, n_layers);
        pl.v_v = try allocator.alloc(buffer.Buffer, n_layers);
        pl.m_o = try allocator.alloc(buffer.Buffer, n_layers);
        pl.v_o = try allocator.alloc(buffer.Buffer, n_layers);
        pl.m_n2 = try allocator.alloc(buffer.Buffer, n_layers);
        pl.v_n2 = try allocator.alloc(buffer.Buffer, n_layers);
        pl.m_gate = try allocator.alloc(buffer.Buffer, n_layers);
        pl.v_gate = try allocator.alloc(buffer.Buffer, n_layers);
        pl.m_up = try allocator.alloc(buffer.Buffer, n_layers);
        pl.v_up = try allocator.alloc(buffer.Buffer, n_layers);
        pl.m_down = try allocator.alloc(buffer.Buffer, n_layers);
        pl.v_down = try allocator.alloc(buffer.Buffer, n_layers);
        // Q/K-norm slots: allocated only when enabled. Empty slices
        // when disabled — the per-layer dispatch paths key off
        // `cfg.qk_norm`, so empty slices are never indexed.
        const qkn = cfg.qk_norm;
        pl.w_q_norm = if (qkn) try allocator.alloc(buffer.Buffer, n_layers) else &.{};
        pl.w_k_norm = if (qkn) try allocator.alloc(buffer.Buffer, n_layers) else &.{};
        pl.a_q_pre_norm = if (qkn) try allocator.alloc(buffer.Buffer, n_layers) else &.{};
        pl.a_k_pre_norm = if (qkn) try allocator.alloc(buffer.Buffer, n_layers) else &.{};
        pl.dw_q_norm_partial = if (qkn) try allocator.alloc(buffer.Buffer, n_layers) else &.{};
        pl.dw_k_norm_partial = if (qkn) try allocator.alloc(buffer.Buffer, n_layers) else &.{};
        pl.dw_q_norm = if (qkn) try allocator.alloc(buffer.Buffer, n_layers) else &.{};
        pl.dw_k_norm = if (qkn) try allocator.alloc(buffer.Buffer, n_layers) else &.{};
        pl.m_q_norm = if (qkn) try allocator.alloc(buffer.Buffer, n_layers) else &.{};
        pl.v_q_norm = if (qkn) try allocator.alloc(buffer.Buffer, n_layers) else &.{};
        pl.m_k_norm = if (qkn) try allocator.alloc(buffer.Buffer, n_layers) else &.{};
        pl.v_k_norm = if (qkn) try allocator.alloc(buffer.Buffer, n_layers) else &.{};
        pl.d_x_in = try allocator.alloc(buffer.Buffer, n_layers);
        return pl;
    }

    fn populateLayer(self: *PerLayerArrays, ctx: *const vk.Context, li: usize, cfg: Config, lw: LayerWeights) !void {
        const f32sz = @sizeOf(f32);
        const dim: usize = @intCast(cfg.dim);
        const n_pos: usize = @intCast(cfg.n_pos);
        const q_dim: usize = @intCast(cfg.n_heads * cfg.head_dim);
        const kv_dim: usize = @intCast(cfg.n_kv_heads * cfg.head_dim);
        const ff_dim: usize = @intCast(cfg.ff_dim);
        const scores_total: usize = n_pos * @as(usize, @intCast(cfg.n_heads)) * n_pos;

        self.w_n1[li] = try buffer.Buffer.initStatic(ctx, f32, lw.w_n1);
        self.w_q[li] = try buffer.Buffer.initStatic(ctx, f32, lw.w_q);
        self.w_k[li] = try buffer.Buffer.initStatic(ctx, f32, lw.w_k);
        self.w_v[li] = try buffer.Buffer.initStatic(ctx, f32, lw.w_v);
        self.w_o[li] = try buffer.Buffer.initStatic(ctx, f32, lw.w_o);
        self.w_n2[li] = try buffer.Buffer.initStatic(ctx, f32, lw.w_n2);
        self.w_gate[li] = try buffer.Buffer.initStatic(ctx, f32, lw.w_gate);
        self.w_up[li] = try buffer.Buffer.initStatic(ctx, f32, lw.w_up);
        self.w_down[li] = try buffer.Buffer.initStatic(ctx, f32, lw.w_down);

        self.a_n1[li] = try buffer.Buffer.initDeviceOnly(ctx, n_pos * dim * f32sz);
        self.a_q[li] = try buffer.Buffer.initDeviceOnly(ctx, n_pos * q_dim * f32sz);
        self.a_k[li] = try buffer.Buffer.initDeviceOnly(ctx, n_pos * kv_dim * f32sz);
        self.a_v[li] = try buffer.Buffer.initDeviceOnly(ctx, n_pos * kv_dim * f32sz);
        self.a_attn[li] = try buffer.Buffer.initDeviceOnly(ctx, scores_total * f32sz);
        self.a_attn_out[li] = try buffer.Buffer.initDeviceOnly(ctx, n_pos * q_dim * f32sz);
        self.a_mid[li] = try buffer.Buffer.initDeviceOnly(ctx, n_pos * dim * f32sz);
        self.a_n2[li] = try buffer.Buffer.initDeviceOnly(ctx, n_pos * dim * f32sz);
        self.a_pre_gate[li] = try buffer.Buffer.initDeviceOnly(ctx, n_pos * ff_dim * f32sz);
        self.a_up[li] = try buffer.Buffer.initDeviceOnly(ctx, n_pos * ff_dim * f32sz);
        self.a_gated[li] = try buffer.Buffer.initDeviceOnly(ctx, n_pos * ff_dim * f32sz);
        self.a_y[li] = try buffer.Buffer.initDeviceOnly(ctx, n_pos * dim * f32sz);

        self.dw_n1_partial[li] = try buffer.Buffer.initDeviceOnly(ctx, n_pos * dim * f32sz);
        self.dw_q[li] = try buffer.Buffer.initDeviceOnly(ctx, lw.w_q.len * f32sz);
        self.dw_k[li] = try buffer.Buffer.initDeviceOnly(ctx, lw.w_k.len * f32sz);
        self.dw_v[li] = try buffer.Buffer.initDeviceOnly(ctx, lw.w_v.len * f32sz);
        self.dw_o[li] = try buffer.Buffer.initDeviceOnly(ctx, lw.w_o.len * f32sz);
        self.dw_n2_partial[li] = try buffer.Buffer.initDeviceOnly(ctx, n_pos * dim * f32sz);
        self.dw_gate[li] = try buffer.Buffer.initDeviceOnly(ctx, lw.w_gate.len * f32sz);
        self.dw_up[li] = try buffer.Buffer.initDeviceOnly(ctx, lw.w_up.len * f32sz);
        self.dw_down[li] = try buffer.Buffer.initDeviceOnly(ctx, lw.w_down.len * f32sz);

        self.dw_n1[li] = try buffer.Buffer.initDynamic(ctx, dim * f32sz);
        self.dw_n2[li] = try buffer.Buffer.initDynamic(ctx, dim * f32sz);

        self.m_n1[li] = try buffer.Buffer.initDeviceOnly(ctx, dim * f32sz);
        self.v_n1[li] = try buffer.Buffer.initDeviceOnly(ctx, dim * f32sz);
        self.m_q[li] = try buffer.Buffer.initDeviceOnly(ctx, lw.w_q.len * f32sz);
        self.v_q[li] = try buffer.Buffer.initDeviceOnly(ctx, lw.w_q.len * f32sz);
        self.m_k[li] = try buffer.Buffer.initDeviceOnly(ctx, lw.w_k.len * f32sz);
        self.v_k[li] = try buffer.Buffer.initDeviceOnly(ctx, lw.w_k.len * f32sz);
        self.m_v[li] = try buffer.Buffer.initDeviceOnly(ctx, lw.w_v.len * f32sz);
        self.v_v[li] = try buffer.Buffer.initDeviceOnly(ctx, lw.w_v.len * f32sz);
        self.m_o[li] = try buffer.Buffer.initDeviceOnly(ctx, lw.w_o.len * f32sz);
        self.v_o[li] = try buffer.Buffer.initDeviceOnly(ctx, lw.w_o.len * f32sz);
        self.m_n2[li] = try buffer.Buffer.initDeviceOnly(ctx, dim * f32sz);
        self.v_n2[li] = try buffer.Buffer.initDeviceOnly(ctx, dim * f32sz);
        self.m_gate[li] = try buffer.Buffer.initDeviceOnly(ctx, lw.w_gate.len * f32sz);
        self.v_gate[li] = try buffer.Buffer.initDeviceOnly(ctx, lw.w_gate.len * f32sz);
        self.m_up[li] = try buffer.Buffer.initDeviceOnly(ctx, lw.w_up.len * f32sz);
        self.v_up[li] = try buffer.Buffer.initDeviceOnly(ctx, lw.w_up.len * f32sz);
        self.m_down[li] = try buffer.Buffer.initDeviceOnly(ctx, lw.w_down.len * f32sz);
        self.v_down[li] = try buffer.Buffer.initDeviceOnly(ctx, lw.w_down.len * f32sz);

        self.d_x_in[li] = try buffer.Buffer.initDeviceOnly(ctx, n_pos * dim * f32sz);

        if (cfg.qk_norm) {
            const hd: usize = @intCast(cfg.head_dim);
            self.w_q_norm[li] = try buffer.Buffer.initStatic(ctx, f32, lw.w_q_norm);
            self.w_k_norm[li] = try buffer.Buffer.initStatic(ctx, f32, lw.w_k_norm);
            self.a_q_pre_norm[li] = try buffer.Buffer.initDeviceOnly(ctx, n_pos * q_dim * f32sz);
            self.a_k_pre_norm[li] = try buffer.Buffer.initDeviceOnly(ctx, n_pos * kv_dim * f32sz);
            self.dw_q_norm_partial[li] = try buffer.Buffer.initDeviceOnly(ctx, n_pos * q_dim * f32sz);
            self.dw_k_norm_partial[li] = try buffer.Buffer.initDeviceOnly(ctx, n_pos * kv_dim * f32sz);
            self.dw_q_norm[li] = try buffer.Buffer.initDynamic(ctx, hd * f32sz);
            self.dw_k_norm[li] = try buffer.Buffer.initDynamic(ctx, hd * f32sz);
            self.m_q_norm[li] = try buffer.Buffer.initDeviceOnly(ctx, hd * f32sz);
            self.v_q_norm[li] = try buffer.Buffer.initDeviceOnly(ctx, hd * f32sz);
            self.m_k_norm[li] = try buffer.Buffer.initDeviceOnly(ctx, hd * f32sz);
            self.v_k_norm[li] = try buffer.Buffer.initDeviceOnly(ctx, hd * f32sz);
        }

        self.populated = li + 1;
    }

    /// Roll back partial init: deinit the populated prefix of every
    /// per-layer buffer slot, then free all the slice allocations.
    fn deinitOnError(self: *PerLayerArrays, dev: anytype, allocator: std.mem.Allocator) void {
        const arrs = [_][]buffer.Buffer{
            self.w_n1,        self.w_q,          self.w_k,
            self.w_v,         self.w_o,          self.w_n2,
            self.w_gate,      self.w_up,         self.w_down,
            self.a_n1,        self.a_q,          self.a_k,
            self.a_v,         self.a_attn,      self.a_attn_out,
            self.a_mid,       self.a_n2,        self.a_pre_gate,
            self.a_up,        self.a_gated,     self.a_y,
            self.dw_n1_partial, self.dw_q,      self.dw_k,
            self.dw_v,        self.dw_o,        self.dw_n2_partial,
            self.dw_gate,     self.dw_up,       self.dw_down,
            self.dw_n1,       self.dw_n2,       self.m_n1,
            self.v_n1,        self.m_q,         self.v_q,
            self.m_k,         self.v_k,         self.m_v,
            self.v_v,         self.m_o,         self.v_o,
            self.m_n2,        self.v_n2,        self.m_gate,
            self.v_gate,      self.m_up,        self.v_up,
            self.m_down,      self.v_down,      self.d_x_in,
        };
        for (arrs) |arr| {
            for (arr[0..self.populated]) |*b| b.deinit(dev);
            allocator.free(arr);
        }
    }
};

// ── LoRA state (A4-2 → A4-3) ─────────────────────────────────────────
//
// Lives on the Runner as `lora: ?LoraState` and is populated only when
// `cfg.lora_targets != 0`. One `ProjLoraSlot` per enabled projection
// (Q / K / V / O / gate / up / down) holds that projection's per-layer
// adapters + Adam state; disabled projections sit at `null` in the
// `slots` array and pay no allocation cost. Shared scratches are sized
// to the largest enabled projection along each axis (max N for the
// y-side scratches, max K for the dx-side, both for ∇A/∇B).
//
// Default init (when InitWeights.lora_<proj> is empty):
//   A ~ N(0, 1/sqrt(rank))  — Hu et al. LoRA paper.
//   B = 0                   — guarantees the LoRA delta is zero on the
//                             first forward, so loss(LoRA on, B=0) ==
//                             loss(LoRA off). The first backward kicks
//                             B into motion (∇B is driven by `inter
//                             = x·Aᵀ` which is non-zero); ∇A starts
//                             non-zero from step 2 once B has moved.

const ProjLoraSlot = struct {
    a: []buffer.Buffer,
    b: []buffer.Buffer,
    intermediate: []buffer.Buffer,
    dw_a: []buffer.Buffer,
    dw_b: []buffer.Buffer,
    m_a: []buffer.Buffer,
    v_a: []buffer.Buffer,
    m_b: []buffer.Buffer,
    v_b: []buffer.Buffer,
    shape: lora_helpers.LoraShape,

    fn deinit(self: *ProjLoraSlot, dev: anytype, allocator: std.mem.Allocator) void {
        const arrs = [_][]buffer.Buffer{
            self.a, self.b, self.intermediate, self.dw_a, self.dw_b,
            self.m_a, self.v_a, self.m_b, self.v_b,
        };
        for (arrs) |arr| {
            for (arr) |*buf| buf.deinit(dev);
            allocator.free(arr);
        }
    }
};

const LoraState = struct {
    k_scale: pipeline.Kernel,
    targets: u32,
    slots: [proj_count]?ProjLoraSlot,

    sc_y_lora: buffer.Buffer,
    sc_y_lora_scaled: buffer.Buffer,
    sc_dy_B: buffer.Buffer,
    sc_dx_lora: buffer.Buffer,
    sc_dx_lora_scaled: buffer.Buffer,
    sc_dA_unscaled: buffer.Buffer,
    sc_dB_unscaled: buffer.Buffer,

    fn projShape(p: Proj, cfg: Config) lora_helpers.LoraShape {
        const r = cfg.lora_rank;
        const aor: f32 = cfg.lora_alpha / @as(f32, @floatFromInt(r));
        const q_dim: u32 = cfg.n_heads * cfg.head_dim;
        const kv_dim: u32 = cfg.n_kv_heads * cfg.head_dim;
        return switch (p) {
            .q => .{ .M = cfg.n_pos, .N = q_dim, .K = cfg.dim, .r = r, .alpha_over_r = aor },
            .k => .{ .M = cfg.n_pos, .N = kv_dim, .K = cfg.dim, .r = r, .alpha_over_r = aor },
            .v => .{ .M = cfg.n_pos, .N = kv_dim, .K = cfg.dim, .r = r, .alpha_over_r = aor },
            .o => .{ .M = cfg.n_pos, .N = cfg.dim, .K = q_dim, .r = r, .alpha_over_r = aor },
            .gate => .{ .M = cfg.n_pos, .N = cfg.ff_dim, .K = cfg.dim, .r = r, .alpha_over_r = aor },
            .up => .{ .M = cfg.n_pos, .N = cfg.ff_dim, .K = cfg.dim, .r = r, .alpha_over_r = aor },
            .down => .{ .M = cfg.n_pos, .N = cfg.dim, .K = cfg.ff_dim, .r = r, .alpha_over_r = aor },
        };
    }

    fn projInitSlice(p: Proj, weights: InitWeights) []const LoraInitWeights {
        return switch (p) {
            .q => weights.lora_q,
            .k => weights.lora_k,
            .v => weights.lora_v,
            .o => weights.lora_o,
            .gate => weights.lora_gate,
            .up => weights.lora_up,
            .down => weights.lora_down,
        };
    }

    fn allocAndInit(
        allocator: std.mem.Allocator,
        ctx: *const vk.Context,
        cfg: Config,
        weights: InitWeights,
    ) !LoraState {
        std.debug.assert(cfg.lora_targets != 0);
        if (cfg.lora_rank == 0) return error.LoraRankZero;

        const n_layers: usize = @intCast(cfg.n_layers);
        const n_pos: usize = @intCast(cfg.n_pos);
        const r: usize = @intCast(cfg.lora_rank);
        const f32sz = @sizeOf(f32);

        var k_scale = try pipeline.Kernel.init(ctx, &shaders.scale, 2, @sizeOf(runtime_hybrid.ScalePush));
        errdefer k_scale.deinit();

        // ── Default-init RNG (fixed seed so smokes are deterministic).
        //    σ = 1/sqrt(r) is the same scale Hu et al. use in the LoRA
        //    paper; the seed is shared across projections so tests can
        //    re-derive A from the seed if needed.
        var prng = std.Random.DefaultPrng.init(0x10AAFADE);
        const rng = prng.random();
        const sigma: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(r)));

        // ── Track maxima for scratch sizing.
        var max_n: u32 = 0;
        var max_k: u32 = 0;

        var slots: [proj_count]?ProjLoraSlot = .{null} ** proj_count;
        errdefer {
            for (&slots) |*opt| if (opt.*) |*s| s.deinit(ctx.device, allocator);
        }

        // ── Allocate one slot per enabled projection.
        inline for (@typeInfo(Proj).@"enum".fields) |f| {
            const p: Proj = @enumFromInt(f.value);
            if (projEnabled(cfg.lora_targets, p)) {
                const shape = projShape(p, cfg);
                const provided = projInitSlice(p, weights);
                if (provided.len != 0 and provided.len != cfg.n_layers) return error.LoraInitLen;
                if (shape.N > max_n) max_n = shape.N;
                if (shape.K > max_k) max_k = shape.K;

                const a_numel: usize = @as(usize, shape.r) * @as(usize, shape.K);
                const b_numel: usize = @as(usize, shape.N) * @as(usize, shape.r);
                const inter_numel: usize = n_pos * @as(usize, shape.r);

                const a_slice = try allocator.alloc(buffer.Buffer, n_layers);
                errdefer allocator.free(a_slice);
                const b_slice = try allocator.alloc(buffer.Buffer, n_layers);
                errdefer allocator.free(b_slice);
                const intermediate_slice = try allocator.alloc(buffer.Buffer, n_layers);
                errdefer allocator.free(intermediate_slice);
                const dw_a_slice = try allocator.alloc(buffer.Buffer, n_layers);
                errdefer allocator.free(dw_a_slice);
                const dw_b_slice = try allocator.alloc(buffer.Buffer, n_layers);
                errdefer allocator.free(dw_b_slice);
                const m_a_slice = try allocator.alloc(buffer.Buffer, n_layers);
                errdefer allocator.free(m_a_slice);
                const v_a_slice = try allocator.alloc(buffer.Buffer, n_layers);
                errdefer allocator.free(v_a_slice);
                const m_b_slice = try allocator.alloc(buffer.Buffer, n_layers);
                errdefer allocator.free(m_b_slice);
                const v_b_slice = try allocator.alloc(buffer.Buffer, n_layers);
                errdefer allocator.free(v_b_slice);

                // Per-projection host staging — reused across layers.
                const a_host = try allocator.alloc(f32, a_numel);
                defer allocator.free(a_host);
                const b_host = try allocator.alloc(f32, b_numel);
                defer allocator.free(b_host);
                @memset(b_host, 0);

                var populated: usize = 0;
                errdefer {
                    for (0..populated) |i| {
                        a_slice[i].deinit(ctx.device);
                        b_slice[i].deinit(ctx.device);
                        intermediate_slice[i].deinit(ctx.device);
                        dw_a_slice[i].deinit(ctx.device);
                        dw_b_slice[i].deinit(ctx.device);
                        m_a_slice[i].deinit(ctx.device);
                        v_a_slice[i].deinit(ctx.device);
                        m_b_slice[i].deinit(ctx.device);
                        v_b_slice[i].deinit(ctx.device);
                    }
                }

                for (0..n_layers) |li| {
                    const a_src: []const f32 = blk: {
                        if (provided.len != 0) {
                            if (provided[li].a.len != a_numel) return error.LoraAShape;
                            break :blk provided[li].a;
                        }
                        for (a_host) |*v| v.* = rng.floatNorm(f32) * sigma;
                        break :blk a_host;
                    };
                    const b_src: []const f32 = blk: {
                        if (provided.len != 0) {
                            if (provided[li].b.len != b_numel) return error.LoraBShape;
                            break :blk provided[li].b;
                        }
                        break :blk b_host;
                    };
                    a_slice[li] = try buffer.Buffer.initStatic(ctx, f32, a_src);
                    b_slice[li] = try buffer.Buffer.initStatic(ctx, f32, b_src);
                    intermediate_slice[li] = try buffer.Buffer.initDeviceOnly(ctx, inter_numel * f32sz);
                    dw_a_slice[li] = try buffer.Buffer.initDeviceOnly(ctx, a_numel * f32sz);
                    dw_b_slice[li] = try buffer.Buffer.initDeviceOnly(ctx, b_numel * f32sz);
                    m_a_slice[li] = try buffer.Buffer.initDeviceOnly(ctx, a_numel * f32sz);
                    v_a_slice[li] = try buffer.Buffer.initDeviceOnly(ctx, a_numel * f32sz);
                    m_b_slice[li] = try buffer.Buffer.initDeviceOnly(ctx, b_numel * f32sz);
                    v_b_slice[li] = try buffer.Buffer.initDeviceOnly(ctx, b_numel * f32sz);
                    populated = li + 1;
                }

                slots[@intFromEnum(p)] = ProjLoraSlot{
                    .a = a_slice,
                    .b = b_slice,
                    .intermediate = intermediate_slice,
                    .dw_a = dw_a_slice,
                    .dw_b = dw_b_slice,
                    .m_a = m_a_slice,
                    .v_a = v_a_slice,
                    .m_b = m_b_slice,
                    .v_b = v_b_slice,
                    .shape = shape,
                };
            }
        }

        // ── Shared scratches sized to the largest enabled projection
        //    along each axis. The Recorder issues a global memory
        //    barrier between every dispatch (see gpu/recorder.zig), so
        //    reusing a scratch across consecutive LoRA chains within
        //    the same layer (and across layers) is safe.
        const max_n_us: usize = max_n;
        const max_k_us: usize = max_k;
        const sc_y_lora = try buffer.Buffer.initDeviceOnly(ctx, n_pos * max_n_us * f32sz);
        errdefer @constCast(&sc_y_lora).deinit(ctx.device);
        const sc_y_lora_scaled = try buffer.Buffer.initDeviceOnly(ctx, n_pos * max_n_us * f32sz);
        errdefer @constCast(&sc_y_lora_scaled).deinit(ctx.device);
        const sc_dy_B = try buffer.Buffer.initDeviceOnly(ctx, n_pos * r * f32sz);
        errdefer @constCast(&sc_dy_B).deinit(ctx.device);
        const sc_dx_lora = try buffer.Buffer.initDeviceOnly(ctx, n_pos * max_k_us * f32sz);
        errdefer @constCast(&sc_dx_lora).deinit(ctx.device);
        const sc_dx_lora_scaled = try buffer.Buffer.initDeviceOnly(ctx, n_pos * max_k_us * f32sz);
        errdefer @constCast(&sc_dx_lora_scaled).deinit(ctx.device);
        const sc_dA_unscaled = try buffer.Buffer.initDeviceOnly(ctx, r * max_k_us * f32sz);
        errdefer @constCast(&sc_dA_unscaled).deinit(ctx.device);
        const sc_dB_unscaled = try buffer.Buffer.initDeviceOnly(ctx, max_n_us * r * f32sz);
        errdefer @constCast(&sc_dB_unscaled).deinit(ctx.device);

        return LoraState{
            .k_scale = k_scale,
            .targets = cfg.lora_targets,
            .slots = slots,
            .sc_y_lora = sc_y_lora,
            .sc_y_lora_scaled = sc_y_lora_scaled,
            .sc_dy_B = sc_dy_B,
            .sc_dx_lora = sc_dx_lora,
            .sc_dx_lora_scaled = sc_dx_lora_scaled,
            .sc_dA_unscaled = sc_dA_unscaled,
            .sc_dB_unscaled = sc_dB_unscaled,
        };
    }

    fn deinit(self: *LoraState, dev: anytype, allocator: std.mem.Allocator) void {
        self.k_scale.deinit();
        for (&self.slots) |*opt| if (opt.*) |*s| s.deinit(dev, allocator);
        self.sc_y_lora.deinit(dev);
        self.sc_y_lora_scaled.deinit(dev);
        self.sc_dy_B.deinit(dev);
        self.sc_dx_lora.deinit(dev);
        self.sc_dx_lora_scaled.deinit(dev);
        self.sc_dA_unscaled.deinit(dev);
        self.sc_dB_unscaled.deinit(dev);
    }

    fn kernels(self: *const LoraState, runner: *const Runner) lora_helpers.LoraKernels {
        return .{
            .matmul = &runner.k_matmul,
            .lin_dx = &runner.k_lin_dx,
            .lin_dw = &runner.k_lin_dw,
            .scale = &self.k_scale,
            .add_in_place = &runner.k_add,
        };
    }
};
