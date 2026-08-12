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
    /// Plain RMSNorm with no learned gain — the W binding is ignored.
    /// Gemma 4 normalises V this way on every attention layer. Defaults
    /// to 0 so every existing call site is unchanged.
    weightless: u32 = 0,
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

/// tanh soft-cap on the final logits: `out = tanh(in / cap) * cap`.
/// Gemma 4 uses cap = 30.0. Applied after the LM-head matmul and before
/// any sampling.
pub const SoftcapPush = extern struct {
    n_elem: u32,
    cap: f32,
};

pub const RopePartialPush = extern struct {
    n_heads: u32,
    head_dim: u32,
    rotary_dim: u32,
    pos: u32,
    theta_base: f32,
    /// Distance to a rotating pair's partner within the head. Zero
    /// selects the legacy Qwen3.5 value (`rotary_dim / 2`), which keeps
    /// every existing call site bit-identical. Gemma 4's "proportional"
    /// RoPE sets this to `head_dim / 2`.
    pair_stride: u32 = 0,
    /// Denominator used to build inv_freq. Zero selects the legacy
    /// Qwen3.5 value (`rotary_dim`). Gemma 4 sets this to the full
    /// `head_dim` — that is what makes its scheme "proportional", and
    /// getting it wrong shifts every angle by a constant factor.
    freq_dim: u32 = 0,
};

/// Elementwise `out[i] = in[i] * scale`. Shared with the hybrid
/// runtime's `ScalePush`; Gemma 4 uses it for the per-layer output
/// scalar.
pub const ScalePush = extern struct { n: u32, scale: f32 };

/// Copy `n_elem` floats between buffers at the given offsets. Gemma 4
/// uses it to fork V off the k_proj output on `attention_k_eq_v` layers.
pub const SliceCopyPush = extern struct { src_off: u32, dst_off: u32, n_elem: u32 };

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

/// Broadcast a bias vector across rows of a batched matmul result.
pub const AddBiasRowsPush = extern struct { n_rows: u32, dim: u32 };

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
/// `pos_offset` shifts the absolute position used for the rotation
/// angles — row `p` rotates as if it were at position `pos_offset + p`.
/// Training-style prefill from the start of the sequence sets it to 0;
/// chat-side batched prefill that picks up after a decoded prefix sets
/// it to the current cursor position.
pub const RopeBatchedPush = extern struct {
    n_pos: u32,
    n_heads: u32,
    head_dim: u32,
    rotary_dim: u32,
    theta_base: f32,
    pos_offset: u32,
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

/// FlashAttention forward (Dao Algorithm 1; FA-2-style outer-Q loop).
/// One workgroup per (q, h) pair — gx = n_q × n_heads. Subsumes both
/// decode (n_q=1, no mask) and prefill (n_q ≥ 1, optional causal) by
/// runtime parameters; no n_q-specific shader variant required.
///
/// `write_lse != 0` writes `lse[q, h] = m_final + log(l_final)` into
/// the LSE binding for FA backward to recompute softmax later. When
/// `write_lse == 0`, the LSE binding is unused but must still resolve
/// (callers can bind the same buffer as Q to keep descriptors happy).
///
/// Compile-time caps in `shaders/fa_forward.comp`: BC=16 keys per
/// tile, HEAD_DIM ≤ 128. Larger head_dim or BC needs a separate
/// shader variant.
pub const FaForwardPush = extern struct {
    n_q: u32,
    n_heads: u32,
    heads_per_kv: u32,
    head_dim: u32,
    n_kv: u32,
    kv_stride: u32,
    causal: u32,
    write_lse: u32,
    inv_sqrt_dim: f32,
    /// Sliding-window span. Keys older than `window` positions before
    /// the query are masked exactly like keys past the causal cutoff.
    /// Zero = unbounded, which is every family except Gemma 4 (whose
    /// sliding layers use 1024). Defaulted so existing call sites —
    /// and every parity test — keep their current semantics.
    window: u32 = 0,
};

/// FlashDecoding phase 1 (Tri Dao 2023). Decode-only split-K kernel:
/// implicit n_q = 1, no causal mask, K-dimension sharded across
/// `n_splits` workgroups per query head. Dispatch `n_heads × n_splits`
/// WGs; each WG processes keys [split_id · split_size, (split_id+1)
/// · split_size) and emits an unnormalised partial (O, m, l) triple.
///
/// Bindings (5+ writeable): Q, K, V, O_partial, M_partial, L_partial.
/// Buffer shapes:
///   Q          [n_heads, head_dim]
///   K, V       [n_kv, n_kv_heads, head_dim]
///   O_partial  [n_heads, n_splits, head_dim]
///   M_partial  [n_heads, n_splits]
///   L_partial  [n_heads, n_splits]
pub const FaDecodeSplitPush = extern struct {
    n_heads: u32,
    heads_per_kv: u32,
    head_dim: u32,
    n_kv: u32,
    kv_stride: u32,
    n_splits: u32,
    split_size: u32,
    inv_sqrt_dim: f32,
    /// Sliding-window span; 0 = unbounded. At decode the query sits at
    /// n_kv − 1, so this admits keys [n_kv − window, n_kv − 1]. Splits
    /// entirely below the floor emit the neutral partial the merge
    /// already understands. Defaulted so existing call sites keep their
    /// current semantics.
    window: u32 = 0,
};

/// FlashDecoding phase 2 — merge per-split partials into the final
/// attention output. Dispatch `n_heads` WGs; each combines the
/// `n_splits` partials for its head via running max + rescaled sum.
/// `n_splits` is bounded by the shader's coef-cache size (1024).
pub const FaDecodeMergePush = extern struct {
    n_heads: u32,
    head_dim: u32,
    n_splits: u32,
};

/// Pick (n_splits, split_size) for FlashDecoding given the current
/// decode `n_kv`. Targets ≥ 4 splits for typical chat lengths so
/// `n_heads × n_splits` saturates a modern GPU's SM count, and caps
/// `split_size` at 256 (the shape `runFlashDecodingGpuSmoke` exercises).
///
///   n_kv ≤ 4         → 1 split × split_size = max(n_kv, 1)
///   4  < n_kv < 1024 → 4 splits × ceilDiv(n_kv, 4) (≤ 256)
///   n_kv ≥ 1024      → split_size = 256, n_splits = ceilDiv(n_kv, 256)
///
/// The returned `n_splits × split_size ≥ n_kv` always; the kernel's
/// per-WG bounds check handles the partial tail when split_size doesn't
/// divide n_kv. Buffer sizing in `GpuScratch` mirrors this heuristic
/// applied to `max_pos`.
pub fn chooseFaDecodeSplit(n_kv: u32) struct { n_splits: u32, split_size: u32 } {
    if (n_kv <= 4) {
        return .{ .n_splits = 1, .split_size = if (n_kv == 0) 1 else n_kv };
    }
    if (n_kv < 1024) {
        const split_size: u32 = (n_kv + 3) / 4;
        return .{ .n_splits = 4, .split_size = split_size };
    }
    const split_size: u32 = 256;
    const n_splits: u32 = (n_kv + split_size - 1) / split_size;
    return .{ .n_splits = n_splits, .split_size = split_size };
}

/// Maximum head_dim the FlashAttention / FlashDecoding shaders accept.
/// Three SPIR-V variants ship per FA forward/decode shader: the default
/// d=128 build (BC=16), a `_d256` build (BC=8) for the Qwen3.5 family,
/// and a `_d512` build (BC=4) for Gemma 4's global layers. Halving BC
/// as the head grows keeps shared memory roughly constant — ~20 KB at
/// d=512 — which clears AMD RDNA's 32 KB/WG ceiling.
///
/// The dispatcher picks at pipeline-init time via `faForwardSpv` /
/// `faDecodeSplitSpv` / `faBwDqSpv` / `faBwDkvSpv` below.
/// `fa_decode_merge` and `fa_bw_d` share their d=128 build at any head
/// dim (no head_dim-sized shared mem). The backward shaders stop at
/// 256 — training doesn't run Gemma 4's geometry — so callers that
/// need FA *backward* must still gate on 256 themselves.
/// Heads above 512 take the 3-pass fallback.
pub const FA_HEAD_DIM_MAX: u32 = 512;

/// FA SPIR-V variant selector. `head_dim` ≤ 128 picks the BC=16 build;
/// (128, 256] picks the BC=8 `_d256` build. Caller is responsible for
/// gating with `head_dim ≤ FA_HEAD_DIM_MAX` first — anything larger
/// would silently get the d=256 build and overflow shared mem.
pub fn faForwardSpv(head_dim: u32) []const u8 {
    if (head_dim <= 128) return shaders.fa_forward[0..];
    if (head_dim <= 256) return shaders.fa_forward_d256[0..];
    return shaders.fa_forward_d512[0..];
}
pub fn faDecodeSplitSpv(head_dim: u32) []const u8 {
    if (head_dim <= 128) return shaders.fa_decode_split[0..];
    if (head_dim <= 256) return shaders.fa_decode_split_d256[0..];
    return shaders.fa_decode_split_d512[0..];
}
pub fn faBwDqSpv(head_dim: u32) []const u8 {
    return if (head_dim <= 128) shaders.fa_bw_dq[0..] else shaders.fa_bw_dq_d256[0..];
}
pub fn faBwDkvSpv(head_dim: u32) []const u8 {
    return if (head_dim <= 128) shaders.fa_bw_dkv[0..] else shaders.fa_bw_dkv_d256[0..];
}

/// FlashAttention-2 backward, phase 1 — per-row D reduction
/// (`shaders/fa_bw_d.comp`). Computes `D[q, h] = Σ_d O · dO`. Dispatch
/// `n_q × n_heads` workgroups; each cooperatively reduces `head_dim`
/// products via subgroup ops.
///
/// Bindings (3 readonly+writeable): O, dO, D.
pub const FaBwDPush = extern struct {
    n_q: u32,
    n_heads: u32,
    head_dim: u32,
};

/// FlashAttention-2 backward, phase 2 — per-(q, h) dQ accumulation
/// (`shaders/fa_bw_dq.comp`). Recomputes the softmax inline from saved
/// LSE; never materialises the [n_q × n_heads × n_kv] attn matrix.
/// Dispatch `n_q × n_heads` workgroups; each owns its dQ row.
///
/// Bindings (7): Q, K, V, dO, LSE, D, dQ.
pub const FaBwDqPush = extern struct {
    n_q: u32,
    n_heads: u32,
    heads_per_kv: u32,
    head_dim: u32,
    n_kv: u32,
    kv_stride: u32,        // n_kv_heads * head_dim
    causal: u32,
    inv_sqrt_dim: f32,
};

/// FlashAttention-2 backward, phase 3 — per-(k, kv_h) dK + dV
/// accumulation (`shaders/fa_bw_dkv.comp`). Symmetric to phase 2:
/// tile-on-Q outer loop, GQA fold over heads_per_kv inside the WG.
/// Dispatch `n_kv × n_kv_heads` workgroups; each owns its dK + dV row.
///
/// Bindings (8): Q, K, V, dO, LSE, D, dK, dV.
pub const FaBwDkvPush = extern struct {
    n_q: u32,
    n_kv: u32,
    n_heads: u32,
    n_kv_heads: u32,
    heads_per_kv: u32,
    head_dim: u32,
    kv_stride: u32,        // n_kv_heads * head_dim
    causal: u32,
    inv_sqrt_dim: f32,
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
    /// Optional label smoothing ε (Chronicals §"Label Smoothing").
    /// Softens the one-hot target to (1−ε)·δ_{v,t} + ε/V; the per-row
    /// loss becomes lse − (1−ε)·z_target − ε·z_mean. Default 0 ⇒
    /// hard CE; typical training value is 0.1. Adds one cooperative
    /// reduction per chunk in the forward shader.
    label_smoothing_eps: f32 = 0.0,
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
    /// Must match the value used in the matching cce_forward dispatch.
    /// Replaces the target indicator (1.0) with (1−ε) and subtracts
    /// ε/V from every dz_v. Default 0 ⇒ hard-CE backward.
    label_smoothing_eps: f32 = 0.0,
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
    /// Elementwise scale — Gemma 4's per-layer output scalar.
    scale: pipeline.Kernel,
    /// Buffer-to-buffer copy — forks V off the k_proj output where
    /// `attention_k_eq_v` removes the v_proj weight.
    slice_copy: pipeline.Kernel,
    /// Gemma 2+ tanh soft-cap on the final logits. Built unconditionally
    /// (it is a trivial elementwise kernel); the forward path only
    /// dispatches it when `cfg.final_logit_softcapping != 0`.
    softcap: pipeline.Kernel,
    rope: pipeline.Kernel,
    rope_partial: pipeline.Kernel,
    kv_write: pipeline.Kernel,
    scores: pipeline.Kernel,
    softmax: pipeline.Kernel,
    attn_out: pipeline.Kernel,
    /// FlashDecoding phase 1 — split-K decode kernel. Replaces the
    /// `scores → softmax → attn_out` trio when `cfg.head_dim ≤ 128`
    /// (see `FA_HEAD_DIM_MAX`); falls through to 3-pass otherwise.
    /// FA decode kernel built for `fa_base_head_dim`. Gemma 4 varies
    /// head_dim per layer (256 sliding / 512 global) and the SPIR-V
    /// variant is fixed at pipeline-creation time, so one kernel cannot
    /// serve both: dispatching the d256 build at head_dim 512 overruns
    /// its shared-memory tiles and silently corrupts the output.
    fa_decode_split: pipeline.Kernel,
    /// Built for `cfg.maxHeadDim()` when that exceeds the base — null
    /// for every uniform-geometry family. Selected per layer by
    /// `faDecodeSplitFor`.
    fa_decode_split_wide: ?pipeline.Kernel,
    fa_base_head_dim: u32,
    /// FlashDecoding phase 2 — merge per-split (O, m, l) partials into
    /// final attention output. Paired with `fa_decode_split`.
    fa_decode_merge: pipeline.Kernel,
    /// Fused FlashDecoding + TQ4 V dequant. Used when `cfg.head_dim ==
    /// 256` and `--tq4v` is active — the kernel reads the packed V
    /// cache directly and dequants inline per K-tile, skipping the
    /// `tq4_unpack` dispatch and `dequant_v` HBM scratch entirely.
    /// Always built (single d=256 SPIR-V variant); the dispatcher
    /// gates by head_dim + tq4_v presence.
    fa_decode_split_tq4v: pipeline.Kernel,
    add: pipeline.Kernel,
    geglu: pipeline.Kernel,

    /// Pick the FA decode kernel whose SPIR-V variant covers
    /// `head_dim`. Callers pass the *per-layer* head dim.
    pub fn faDecodeSplitFor(self: *const ChatKernels, head_dim: u32) *const pipeline.Kernel {
        if (head_dim > self.fa_base_head_dim) {
            if (self.fa_decode_split_wide) |*w| return w;
        }
        return &self.fa_decode_split;
    }

    /// `head_dim` is the model's base (smallest) per-layer head dim;
    /// `max_head_dim` the largest. They differ only on Gemma 4, whose
    /// sliding layers are 256 and global layers 512.
    pub fn initWide(
        ctx: *const vk.Context,
        precision: gpu_model.Precision,
        family: config_mod.Family,
        head_dim: u32,
        max_head_dim: u32,
    ) !ChatKernels {
        var k = try init(ctx, precision, family, head_dim);
        errdefer k.deinit();
        if (max_head_dim > head_dim) {
            k.fa_decode_split_wide = try pipeline.Kernel.init(ctx, faDecodeSplitSpv(max_head_dim), 6, @sizeOf(FaDecodeSplitPush));
        }
        return k;
    }

    pub fn init(
        ctx: *const vk.Context,
        precision: gpu_model.Precision,
        family: config_mod.Family,
        head_dim: u32,
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
            .softcap = try pipeline.Kernel.init(ctx, &shaders.softcap, 2, @sizeOf(SoftcapPush)),
            .scale = try pipeline.Kernel.init(ctx, &shaders.scale, 2, @sizeOf(ScalePush)),
            .slice_copy = try pipeline.Kernel.init(ctx, &shaders.slice_copy, 2, @sizeOf(SliceCopyPush)),
            .rope = try pipeline.Kernel.init(ctx, &shaders.rope, 2, @sizeOf(RopePush)),
            .rope_partial = try pipeline.Kernel.init(ctx, &shaders.rope_partial, 2, @sizeOf(RopePartialPush)),
            .kv_write = try pipeline.Kernel.init(ctx, &shaders.kv_write, 2, @sizeOf(KvWritePush)),
            .scores = try pipeline.Kernel.init(ctx, &shaders.attn_scores, 3, @sizeOf(AttnScoresPush)),
            .softmax = try pipeline.Kernel.init(ctx, &shaders.softmax, 2, @sizeOf(SoftmaxPush)),
            .attn_out = try pipeline.Kernel.init(ctx, &shaders.attn_output, 3, @sizeOf(AttnOutputPush)),
            .fa_decode_split = try pipeline.Kernel.init(ctx, faDecodeSplitSpv(head_dim), 6, @sizeOf(FaDecodeSplitPush)),
            .fa_decode_split_wide = null,
            .fa_base_head_dim = head_dim,
            .fa_decode_merge = try pipeline.Kernel.init(ctx, &shaders.fa_decode_merge, 4, @sizeOf(FaDecodeMergePush)),
            .fa_decode_split_tq4v = try pipeline.Kernel.init(ctx, &shaders.fa_decode_split_tq4v, 6, @sizeOf(FaDecodeSplitPush)),
            .add = try pipeline.Kernel.init(ctx, &shaders.add_in_place, 2, @sizeOf(AddInPlacePush)),
            .geglu = try pipeline.Kernel.init(ctx, ffn_spv, 3, @sizeOf(GegluPush)),
        };
    }

    pub fn deinit(self: *ChatKernels) void {
        self.embed.deinit();
        self.rmsnorm.deinit();
        self.matmul.deinit();
        self.matmul_lm_head.deinit();
        self.softcap.deinit();
        self.scale.deinit();
        self.slice_copy.deinit();
        self.rope.deinit();
        self.rope_partial.deinit();
        self.kv_write.deinit();
        self.scores.deinit();
        self.softmax.deinit();
        self.attn_out.deinit();
        self.fa_decode_split.deinit();
        if (self.fa_decode_split_wide) |*w| w.deinit();
        self.fa_decode_merge.deinit();
        self.fa_decode_split_tq4v.deinit();
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
    /// Gemma 4 applies a plain, gain-free per-head RMSNorm to V on every
    /// attention layer. No tensor exists for it in the checkpoint, so it
    /// rides on the config rather than on the presence of a weight.
    v_norm_weightless: bool,
    v_norm_push: RmsnormPush,
    kv_write_push: KvWritePush,
    scores_push: AttnScoresPush,
    softmax_push: SoftmaxPush,
    attn_out_push: AttnOutputPush,
    /// Set when `cfg.head_dim ≤ FA_HEAD_DIM_MAX`. `recordOneLayer`
    /// then dispatches the FlashDecoding split + merge pair instead
    /// of the 3-pass `scores → softmax → attn_out` chain.
    attn_use_fa: bool,
    fa_decode_split_push: FaDecodeSplitPush,
    fa_decode_merge_push: FaDecodeMergePush,
    geglu_push: GegluPush,
    n_pos: u32,
};

pub fn computeForwardPushes(
    cfg: config_mod.Config,
    sc: *const gpu_scratch.GpuScratch,
    pos: usize,
    /// Which layer these pushes are for. Only Gemma 4 varies anything by
    /// layer (head dim, KV head count, RoPE base, rotary fraction,
    /// sliding window); for every other family the result is identical
    /// for all layers and callers may pass 0. This is a pure arithmetic
    /// function, so recomputing it per layer costs nothing.
    layer_idx: usize,
) ForwardPushes {
    const hidden: u32 = @intCast(cfg.hidden_size);
    const inter: u32 = @intCast(cfg.intermediate_size);
    const gemma_quirk: u32 = if (cfg.family.rmsnormAddOne()) 1 else 0;
    const max_pos_u32: u32 = @intCast(sc.max_pos);
    const n_pos: u32 = @intCast(pos + 1);

    // ── Per-layer attention geometry ────────────────────────────────
    const head_dim_i = cfg.headDimAt(layer_idx);
    const n_kv_heads_i = cfg.numKvHeadsAt(layer_idx);
    const head_dim_u32: u32 = @intCast(head_dim_i);
    const kv_dim: u32 = @intCast(n_kv_heads_i * head_dim_i);
    const heads_per_kv: u32 = @intCast(cfg.num_attention_heads / n_kv_heads_i);
    const window: u32 = @intCast(cfg.attnWindowAt(layer_idx));

    // Gemma 4 folds the 1/sqrt(head_dim) into its trained weights and
    // uses a scale of exactly 1.0. Everyone else takes the textbook
    // value. Named `inv_sqrt_dim` in the push structs for historical
    // reasons — it is the pre-softmax QK scale, whatever its value.
    const inv_sqrt_dim: f32 = cfg.attnScaleAt(layer_idx);

    const rope_theta_i: f32 = cfg.ropeThetaAt(layer_idx);
    const partial_i: f32 = cfg.partialRotaryAt(layer_idx);
    const rotary_dim: u32 = @intFromFloat(@as(f32, @floatFromInt(head_dim_u32)) * partial_i);
    const use_partial_rope: bool = partial_i < 1.0;

    // Gemma 4's global layers use rope_type "proportional": the pair
    // partner sits head_dim/2 away rather than rotary_dim/2, and the
    // frequency schedule is built over the full head_dim. Zero selects
    // the legacy Qwen3.5 behaviour in the shared kernel.
    const rope_is_proportional = cfg.family == .gemma4 and use_partial_rope;
    const rope_pair_stride: u32 = if (rope_is_proportional) head_dim_u32 / 2 else 0;
    const rope_freq_dim: u32 = if (rope_is_proportional) head_dim_u32 else 0;

    return .{
        .rms_push = .{ .dim = hidden, .eps = cfg.rms_norm_eps, .gemma_quirk = gemma_quirk },
        .qkn_push = .{ .dim = head_dim_u32, .eps = cfg.rms_norm_eps, .gemma_quirk = 0 },
        .add_push = .{ .n = hidden },
        .rope_q_push = .{
            .n_heads = @intCast(cfg.num_attention_heads),
            .head_dim = head_dim_u32,
            .pos = @intCast(pos),
            .theta_base = rope_theta_i,
        },
        .rope_k_push = .{
            .n_heads = @intCast(n_kv_heads_i),
            .head_dim = head_dim_u32,
            .pos = @intCast(pos),
            .theta_base = rope_theta_i,
        },
        .rope_q_partial_push = .{
            .n_heads = @intCast(cfg.num_attention_heads),
            .head_dim = head_dim_u32,
            .rotary_dim = rotary_dim,
            .pos = @intCast(pos),
            .theta_base = rope_theta_i,
            .pair_stride = rope_pair_stride,
            .freq_dim = rope_freq_dim,
        },
        .rope_k_partial_push = .{
            .n_heads = @intCast(n_kv_heads_i),
            .head_dim = head_dim_u32,
            .rotary_dim = rotary_dim,
            .pos = @intCast(pos),
            .theta_base = rope_theta_i,
            .pair_stride = rope_pair_stride,
            .freq_dim = rope_freq_dim,
        },
        .use_partial_rope = use_partial_rope,
        .v_norm_weightless = cfg.family == .gemma4,
        .v_norm_push = .{
            .dim = head_dim_u32,
            .eps = cfg.rms_norm_eps,
            .gemma_quirk = 0,
            .weightless = 1,
        },
        .kv_write_push = .{ .n = kv_dim, .dst_off = @intCast(pos * @as(usize, kv_dim)) },
        .scores_push = .{
            .n_heads = @intCast(cfg.num_attention_heads),
            .heads_per_kv = heads_per_kv,
            .head_dim = head_dim_u32,
            .n_pos = n_pos,
            .kv_stride = kv_dim,
            .scores_stride = max_pos_u32,
            .inv_sqrt_dim = inv_sqrt_dim,
        },
        .softmax_push = .{ .dim = n_pos, .stride = max_pos_u32 },
        .attn_out_push = .{
            .n_heads = @intCast(cfg.num_attention_heads),
            .heads_per_kv = heads_per_kv,
            .head_dim = head_dim_u32,
            .n_pos = n_pos,
            .kv_stride = kv_dim,
            .scores_stride = max_pos_u32,
        },
        .attn_use_fa = head_dim_u32 <= FA_HEAD_DIM_MAX,
        .fa_decode_split_push = blk: {
            const ch = chooseFaDecodeSplit(n_pos);
            break :blk .{
                .n_heads = @intCast(cfg.num_attention_heads),
                .heads_per_kv = heads_per_kv,
                .head_dim = head_dim_u32,
                .n_kv = n_pos,
                .kv_stride = kv_dim,
                .n_splits = ch.n_splits,
                .split_size = ch.split_size,
                .inv_sqrt_dim = inv_sqrt_dim,
                .window = window,
            };
        },
        .fa_decode_merge_push = blk: {
            const ch = chooseFaDecodeSplit(n_pos);
            break :blk .{
                .n_heads = @intCast(cfg.num_attention_heads),
                .head_dim = head_dim_u32,
                .n_splits = ch.n_splits,
            };
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

/// `MAX_M` baked into `matmul_nt_v2_q4_k_mcol.comp`. The shader's
/// per-thread `acc[MAX_M]` is a fixed array; pc.M ≤ MAX_M is the
/// caller's job. MTP-verify at n_q=4 fits comfortably; batched
/// chat prefill of typical chat-template prompts (>>8 tokens) does
/// not — the dispatcher falls back to the row-major path for those.
pub const Q4K_MCOL_MAX_M: u32 = 8;

/// Q4_K-only column-major matmul fast path. When `k_mcol_opt` is
/// non-null AND `m > 1` AND `m ≤ Q4K_MCOL_MAX_M`, dispatches the
/// mcol variant (one WG per output column, weight dequant
/// amortized across M activation rows). Otherwise falls back to
/// `recDispatchMatmul`'s row-major kernel.
///
/// The mcol kernel is bit-numerics-equivalent to the row-major
/// kernel at all valid M (parity-tested in
/// `runGpuMatmulQ4_KMColSmoke`); for M=1 they're algorithmically
/// identical anyway, so the dispatcher can safely route either
/// way at M=1 — we keep the row-major path for M=1 because that's
/// the path that's been battle-tested in production.
pub fn recDispatchMatmulPreferMCol(
    rec: *recorder.Recorder,
    k_orig: *const pipeline.Kernel,
    k_mcol_opt: ?*const pipeline.Kernel,
    bufs: []const *const buffer.Buffer,
    m: u32,
    n: u32,
    k: u32,
) !void {
    const push = MatmulPush{ .m = m, .n = n, .k = k };
    if (m > 1 and m <= Q4K_MCOL_MAX_M) {
        if (k_mcol_opt) |k_mcol| {
            try rec.dispatch(k_mcol, bufs, &push, n, 1, 1);
            return;
        }
    }
    try rec.dispatch(k_orig, bufs, &push, m * n, 1, 1);
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
    // Per-layer geometry. Uniform for every family except Gemma 4,
    // whose global layers run 16x512 against a single 512-wide KV head
    // while its sliding layers run 16x256 against 8.
    const head_dim_i = cfg.headDimAt(layer_idx);
    const n_kv_heads_i = cfg.numKvHeadsAt(layer_idx);
    const q_dim: u32 = @intCast(cfg.num_attention_heads * head_dim_i);
    const kv_dim: u32 = @intCast(n_kv_heads_i * head_dim_i);

    const layer = &gm.layers[layer_idx];

    try recDispatchPerRow(rec, &k.rmsnorm, &.{ &sc.stream, &layer.input_layernorm, &sc.x_norm }, &p.rms_push, 1);

    try recDispatchMatmul(rec, &k.matmul, &.{ &sc.x_norm, &layer.q_proj.?, &sc.q }, 1, q_dim, hidden);
    try recDispatchMatmul(rec, &k.matmul, &.{ &sc.x_norm, &layer.k_proj.?, &sc.k }, 1, kv_dim, hidden);
    if (layer.v_proj) |*vp| {
        try recDispatchMatmul(rec, &k.matmul, &.{ &sc.x_norm, vp, &sc.v }, 1, kv_dim, hidden);
    } else {
        // `attention_k_eq_v`: V shares the k_proj OUTPUT, taken here —
        // before k_norm and before RoPE. K and V diverge immediately
        // after (K gets the learned k_norm plus RoPE; V gets a plain
        // weightless norm and no RoPE), so this is a copy, not an alias.
        // Re-running the k_proj matmul would give the same values for
        // more work.
        const copy_push = SliceCopyPush{ .src_off = 0, .dst_off = 0, .n_elem = kv_dim };
        try recDispatch1D(rec, &k.slice_copy, &.{ &sc.k, &sc.v }, &copy_push, kv_dim);
    }

    if (layer.q_norm) |*qn| {
        try recDispatchPerRow(rec, &k.rmsnorm, &.{ &sc.q, qn, &sc.q }, &p.qkn_push, @intCast(cfg.num_attention_heads));
    }
    if (layer.k_norm) |*kn| {
        try recDispatchPerRow(rec, &k.rmsnorm, &.{ &sc.k, kn, &sc.k }, &p.qkn_push, @intCast(n_kv_heads_i));
    }
    if (p.v_norm_weightless) {
        // Gemma 4 normalises V per head with NO learned gain, on every
        // attention layer — aliased or not. The W binding is ignored by
        // the shader but must still resolve, so it re-binds sc.v.
        try recDispatchPerRow(rec, &k.rmsnorm, &.{ &sc.v, &sc.v, &sc.v }, &p.v_norm_push, @intCast(n_kv_heads_i));
    }

    if (p.use_partial_rope) {
        try recDispatch1D(rec, &k.rope_partial, &.{ &sc.q, &sc.q_rot }, &p.rope_q_partial_push, q_dim);
        try recDispatch1D(rec, &k.rope_partial, &.{ &sc.k, &sc.k_rot }, &p.rope_k_partial_push, kv_dim);
    } else {
        try recDispatchRope(rec, &k.rope, &.{ &sc.q, &sc.q_rot }, &p.rope_q_push, cfg.num_attention_heads, head_dim_i);
        try recDispatchRope(rec, &k.rope, &.{ &sc.k, &sc.k_rot }, &p.rope_k_push, n_kv_heads_i, head_dim_i);
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

    // V cache for attention. With `--tq4v` we unpack the whole V cache
    // into `dequant_v` first; the attention kernel(s) below see the
    // same fp32 layout `[n_pos, n_kv_heads, head_dim]` either way.
    //
    // The T-arc fused `fa_decode_split_tq4v` kernel exists and parity-
    // matches (see `--fa-decode-tq4v-smoke`) but is *not* dispatched
    // here — it's slower than this path on RTX 3090 / Gemma 2B because
    // tq4_unpack saturates many parallel WGs (n_pos × n_kv_heads,
    // 16k+ at long ctx) while the fused kernel crams the same dequant
    // work into the FA path's n_heads × n_splits WGs. Bandwidth
    // savings don't pay for the lost parallelism. See `docs/perf.md`
    // §"Fused TQ4-V (T-arc, investigated)" for full bench data.
    const v_for_attn: *const buffer.Buffer = if (tq4_v) |t| blk: {
        const tq_layer = &t.cache.layers[layer_idx];
        const total_blocks: u32 = p.n_pos * @as(u32, @intCast(t.cache.n_blocks_per_pos));
        try rec.dispatch(t.unpack, &.{ &tq_layer.v_cache, &t.cache.dequant_v }, null, total_blocks, 1, 1);
        break :blk &t.cache.dequant_v;
    } else &kv_layer.v_cache;

    if (p.attn_use_fa) {
        // FlashDecoding (split-K + merge). `head_dim ≤ FA_HEAD_DIM_MAX`
        // is enforced at `computeForwardPushes` time. Phase 1 emits
        // unnormalised (O, m, l) partials for each (head, split) pair;
        // phase 2 combines them into `head_out` with running-max +
        // rescaled-sum.
        const split = p.fa_decode_split_push;
        try rec.dispatch(
            k.faDecodeSplitFor(p.fa_decode_split_push.head_dim),
            &.{ &sc.q_rot, &kv_layer.k_cache, v_for_attn, &sc.fa_o_partial, &sc.fa_m_partial, &sc.fa_l_partial },
            &split,
            @as(u32, @intCast(cfg.num_attention_heads)) * split.n_splits,
            1,
            1,
        );
        try rec.dispatch(
            &k.fa_decode_merge,
            &.{ &sc.fa_o_partial, &sc.fa_m_partial, &sc.fa_l_partial, &sc.head_out },
            &p.fa_decode_merge_push,
            @as(u32, @intCast(cfg.num_attention_heads)),
            1,
            1,
        );
    } else {
        // 3-pass fallback: materialises the [n_heads × n_pos] scores
        // tensor in HBM. Used when head_dim > FA_HEAD_DIM_MAX
        // (e.g. Qwen3.5 d=256).
        try rec.dispatch(
            &k.scores,
            &.{ &sc.q_rot, &kv_layer.k_cache, &sc.scores },
            &p.scores_push,
            @as(u32, @intCast(cfg.num_attention_heads)) * p.n_pos,
            1,
            1,
        );
        try recDispatchPerRow(rec, &k.softmax, &.{ &sc.scores, &sc.scores }, &p.softmax_push, @intCast(cfg.num_attention_heads));
        try rec.dispatch(
            &k.attn_out,
            &.{ &sc.scores, v_for_attn, &sc.head_out },
            &p.attn_out_push,
            @as(u32, @intCast(cfg.num_attention_heads)) * @as(u32, @intCast(head_dim_i)),
            1,
            1,
        );
    }

    try recDispatchMatmul(rec, &k.matmul, &.{ &sc.head_out, &layer.o_proj.?, &sc.attn_out }, 1, hidden, q_dim);

    // ── Attention output → residual ─────────────────────────────────
    //
    // Two different block shapes hang off the same norm tensors, and the
    // NAMES do not mean the same thing in each:
    //
    //   Llama / Qwen:  h = x + attn(...)
    //                  h = h + mlp(post_attention_layernorm(h))
    //     — `post_attention_layernorm` IS the pre-FFN norm.
    //
    //   Gemma 2+ / 4:  h = x + post_attention_layernorm(attn(...))
    //                  h = h + post_feedforward_layernorm(
    //                            mlp(pre_feedforward_layernorm(h)))
    //                  h = h * layer_scalar
    //     — `post_attention_layernorm` normalises the ATTENTION OUTPUT
    //       before its residual, and the pre-FFN role belongs to
    //       `pre_feedforward_layernorm`.
    //
    // Wiring the sandwich form by position rather than by role gives a
    // model that runs and produces fluent-looking text while being
    // quietly wrong, so the two paths are kept explicitly separate.
    const sandwich = layer.pre_feedforward_layernorm != null;

    if (sandwich) {
        try recDispatchPerRow(rec, &k.rmsnorm, &.{ &sc.attn_out, &layer.post_attention_layernorm, &sc.attn_out }, &p.rms_push, 1);
        try recDispatch1D(rec, &k.add, &.{ &sc.stream, &sc.attn_out }, &p.add_push, hidden);
        try recDispatchPerRow(rec, &k.rmsnorm, &.{ &sc.stream, &layer.pre_feedforward_layernorm.?, &sc.mid_norm }, &p.rms_push, 1);
    } else {
        try recDispatch1D(rec, &k.add, &.{ &sc.stream, &sc.attn_out }, &p.add_push, hidden);
        try recDispatchPerRow(rec, &k.rmsnorm, &.{ &sc.stream, &layer.post_attention_layernorm, &sc.mid_norm }, &p.rms_push, 1);
    }

    try recDispatchMatmul(rec, &k.matmul, &.{ &sc.mid_norm, &layer.gate_proj, &sc.gate }, 1, inter, hidden);
    try recDispatchMatmul(rec, &k.matmul, &.{ &sc.mid_norm, &layer.up_proj, &sc.up }, 1, inter, hidden);
    try recDispatch1D(rec, &k.geglu, &.{ &sc.gate, &sc.up, &sc.fused }, &p.geglu_push, inter);
    try recDispatchMatmul(rec, &k.matmul, &.{ &sc.fused, &layer.down_proj, &sc.ffn_out }, 1, hidden, inter);

    if (layer.post_feedforward_layernorm) |*pfn| {
        try recDispatchPerRow(rec, &k.rmsnorm, &.{ &sc.ffn_out, pfn, &sc.ffn_out }, &p.rms_push, 1);
    }

    try recDispatch1D(rec, &k.add, &.{ &sc.stream, &sc.ffn_out }, &p.add_push, hidden);

    // Per-layer output scalar, applied to the whole block output after
    // the FFN residual. Skipped entirely when the checkpoint has none.
    if (layer.layer_scalar) |ls| {
        const scale_push = ScalePush{ .n = hidden, .scale = ls };
        try recDispatch1D(rec, &k.scale, &.{ &sc.stream, &sc.stream }, &scale_push, hidden);
    }
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
    try recordEmbedding(rec, sc, gm, cfg, k, token_id);
    // Pushes are recomputed per layer rather than hoisted: on Gemma 4
    // the head dim, KV head count, RoPE base, rotary fraction and
    // sliding window all vary by layer. It is pure arithmetic, so the
    // per-layer call is free, and for uniform families every iteration
    // produces the same struct.
    for (0..cfg.num_hidden_layers) |layer_idx| {
        const pushes = computeForwardPushes(cfg, sc, pos, layer_idx);
        try recordOneLayer(rec, sc, gm, kv, cfg, k, layer_idx, pos, &pushes, tq4_v);
    }
    if (compute_logits) {
        // The sample step only touches layer-independent fields
        // (final-norm dims, vocab, softcap), so layer 0 is as good as
        // any — but pass the last layer's index to keep the intent
        // "whatever the stream just came out of" rather than arbitrary.
        const tail = computeForwardPushes(cfg, sc, pos, cfg.num_hidden_layers - 1);
        try recordSampleStep(rec, sc, gm, cfg, k, &tail);
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

    // Gemma 2+ squash the logits through tanh before sampling. Skipped
    // entirely when the config doesn't ask for it, so no other family
    // pays a dispatch. In place over the logits buffer — the shader
    // reads and writes the same binding, which is safe because every
    // thread touches exactly one element.
    if (cfg.final_logit_softcapping != 0) {
        const softcap_push = SoftcapPush{
            .n_elem = vocab,
            .cap = cfg.final_logit_softcapping,
        };
        try recDispatch1D(rec, &k.softcap, &.{ &sc.logits, &sc.logits }, &softcap_push, vocab);
    }
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
            .kernels = try ChatKernels.initWide(ctx, gm.precision, gm.config.family, @intCast(gm.config.head_dim), @intCast(gm.config.maxHeadDim())),
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
        // Delegate rather than re-implement: this used to duplicate the
        // embedding / layer-loop / sample-step sequence, which meant it
        // also duplicated the assumption that one ForwardPushes covers
        // every layer. That is false on Gemma 4.
        try recordForwardStep(
            rec,
            sc,
            gm,
            kv,
            self.cfg,
            &self.kernels,
            pos,
            token_id,
            null,
            compute_logits,
        );
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

// ── Gemma 4 vision embedder ───────────────────────────────────────
//
// Runs the whole image at once: every dispatch below is batched over
// `n_patches` rows, so a 280-token image costs the same number of
// dispatches as a single-token one. See `vision.zig` for the pipeline
// and for why the norms are what they are.

/// Kernels for the vision embedder. Built separately from ChatKernels
/// because a text-only checkpoint never needs them.
pub const VisionKernels = struct {
    layernorm: pipeline.Kernel,
    rmsnorm: pipeline.Kernel,
    matmul: pipeline.Kernel,
    add: pipeline.Kernel,
    add_bias_rows: pipeline.Kernel,

    pub fn init(ctx: *const vk.Context) !VisionKernels {
        return .{
            .layernorm = try pipeline.Kernel.init(ctx, &shaders.layernorm, 4, @sizeOf(LayernormPush)),
            .rmsnorm = try pipeline.Kernel.init(ctx, &shaders.rmsnorm, 3, @sizeOf(RmsnormPush)),
            // fp32 matmul: the vision weights are uploaded fp32, so the
            // quantized variants would be reading the wrong layout.
            .matmul = try pipeline.Kernel.init(ctx, &shaders.matmul_nt_v2, 3, @sizeOf(MatmulPush)),
            .add = try pipeline.Kernel.init(ctx, &shaders.add_in_place, 2, @sizeOf(AddInPlacePush)),
            .add_bias_rows = try pipeline.Kernel.init(ctx, &shaders.add_bias_rows, 2, @sizeOf(AddBiasRowsPush)),
        };
    }

    pub fn deinit(self: *VisionKernels) void {
        self.layernorm.deinit();
        self.rmsnorm.deinit();
        self.matmul.deinit();
        self.add.deinit();
        self.add_bias_rows.deinit();
    }
};

/// Device-side scratch for one image. Sized for `max_patches`; a
/// smaller image just uses a prefix of each buffer.
/// Intermediates only. The two INPUTS (patches, pos_sum) are supplied
/// by the caller at record time rather than owned here: they change per
/// image and, in an embedded host, arrive from wherever the framebuffer
/// was staged.
pub const VisionScratch = struct {
    a: buffer.Buffer,
    b: buffer.Buffer,
    /// [max_patches, embed] — the soft tokens, ready to splice.
    out: buffer.Buffer,
    max_patches: usize,

    pub fn init(ctx: *const vk.Context, max_patches: usize, patch_elems: usize, embed: usize) !VisionScratch {
        const f = @sizeOf(f32);
        return .{
            .a = try buffer.Buffer.initDeviceOnly(ctx, max_patches * @max(patch_elems, embed) * f),
            .b = try buffer.Buffer.initDeviceOnly(ctx, max_patches * embed * f),
            .out = try buffer.Buffer.initDeviceOnly(ctx, max_patches * embed * f),
            .max_patches = max_patches,
        };
    }

    pub fn deinit(self: *VisionScratch, device: vk.c.VkDevice) void {
        self.a.deinit(device);
        self.b.deinit(device);
        self.out.deinit(device);
    }
};

/// Record the patch embedder.
///
///   `patches`  [n_patches, patch_elems] preprocessed image
///   `pos_sum`  [n_patches, embed] per-patch sum of the two positional
///              rows (axis 0 by column, axis 1 by row), summed host-side
///
/// Result lands in `sc.out` as [n_patches, embed].
pub fn recordVisionEmbed(
    rec: *recorder.Recorder,
    sc: *const VisionScratch,
    patches: *const buffer.Buffer,
    pos_sum: *const buffer.Buffer,
    gv: *const gpu_model.GpuVision,
    k: *const VisionKernels,
    n_patches: u32,
    embed_dim: u32,
) !void {
    const pe: u32 = @intCast(gv.patch_elems);
    // PyTorch LayerNorm default, deliberately not the model's rms eps.
    const ln_push = LayernormPush{ .dim = pe, .eps = 1e-5 };
    const ln_embed_push = LayernormPush{ .dim = embed_dim, .eps = 1e-5 };

    // LayerNorm over the raw patch, one workgroup per patch.
    try recDispatchPerRow(rec, &k.layernorm, &.{ patches, &gv.patch_ln1_w, &gv.patch_ln1_b, &sc.a }, &ln_push, n_patches);

    // [n_patches, 6912] x [embed, 6912]^T -> [n_patches, embed], + bias.
    try recDispatchMatmul(rec, &k.matmul, &.{ &sc.a, &gv.patch_dense_w, &sc.b }, n_patches, embed_dim, pe);
    const bias_push = AddBiasRowsPush{ .n_rows = n_patches, .dim = embed_dim };
    try recDispatch1D(rec, &k.add_bias_rows, &.{ &gv.patch_dense_b, &sc.b }, &bias_push, n_patches * embed_dim);

    try recDispatchPerRow(rec, &k.layernorm, &.{ &sc.b, &gv.patch_ln2_w, &gv.patch_ln2_b, &sc.a }, &ln_embed_push, n_patches);

    // + posemb(col) + posemb(row), pre-summed on the host.
    try recDispatch1D(rec, &k.add, &.{ &sc.a, pos_sum }, &AddInPlacePush{ .n = n_patches * embed_dim }, n_patches * embed_dim);

    try recDispatchPerRow(rec, &k.layernorm, &.{ &sc.a, &gv.pos_norm_w, &gv.pos_norm_b, &sc.b }, &ln_embed_push, n_patches);

    // Weightless RMSNorm (no gain tensor exists); W binding is ignored
    // by the shader but must still resolve, so it re-binds the input.
    const rms_push = RmsnormPush{ .dim = embed_dim, .eps = 1e-6, .gemma_quirk = 0, .weightless = 1 };
    try recDispatchPerRow(rec, &k.rmsnorm, &.{ &sc.b, &sc.b, &sc.a }, &rms_push, n_patches);

    try recDispatchMatmul(rec, &k.matmul, &.{ &sc.a, &gv.embedding_projection, &sc.out }, n_patches, embed_dim, embed_dim);
}
