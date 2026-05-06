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
const vk = @import("../gpu/vk.zig");
const buffer = @import("../gpu/buffer.zig");
const pipeline = @import("../gpu/pipeline.zig");
const recorder_mod = @import("../gpu/recorder.zig");
const runtime = @import("../runtime.zig");
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
};

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
};

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
    k_attn_dattn: pipeline.Kernel,
    k_attn_dv: pipeline.Kernel,
    k_attn_dq: pipeline.Kernel,
    k_attn_dk: pipeline.Kernel,
    k_swiglu_fwd: pipeline.Kernel,
    k_swiglu_bw: pipeline.Kernel,
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
    push_rope_q: runtime.RopeBatchedPush,
    push_rope_k: runtime.RopeBatchedPush,
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
        var k_rope_fwd = try pipeline.Kernel.init(ctx, &shaders.rope_partial_batched, 2, @sizeOf(runtime.RopeBatchedPush));
        errdefer k_rope_fwd.deinit();
        var k_rope_bw = try pipeline.Kernel.init(ctx, &shaders.rope_backward_batched, 2, @sizeOf(runtime.RopeBatchedPush));
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
        // lm_dw + final_norm_bw + 2 spare). Headroom: 32 + 50·N.
        const phase1_dispatches: u32 = 32 + 50 * cfg.n_layers;
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

        // ── Shared scratch.
        const sc_scores = try buffer.Buffer.initDeviceOnly(ctx, scores_total * f32sz);
        errdefer @constCast(&sc_scores).deinit(ctx.device);
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
        const dw_partial_host = try allocator.alloc(f32, n_pos * dim);
        errdefer allocator.free(dw_partial_host);
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
        const push_rope_q = runtime.RopeBatchedPush{
            .n_pos = cfg.n_pos,
            .n_heads = cfg.n_heads,
            .head_dim = cfg.head_dim,
            .rotary_dim = cfg.rotary_dim,
            .theta_base = cfg.rope_theta,
        };
        const push_rope_k = runtime.RopeBatchedPush{
            .n_pos = cfg.n_pos,
            .n_heads = cfg.n_kv_heads,
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
            .push_rope_q = push_rope_q,
            .push_rope_k = push_rope_k,
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

        // Pipelines + recorder.
        self.k_embed.deinit();
        self.k_rms.deinit();
        self.k_rms_bw.deinit();
        self.k_matmul.deinit();
        self.k_attn_scores.deinit();
        self.k_attn_output.deinit();
        self.k_softmax.deinit();
        self.k_softmax_bw.deinit();
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
        for (0..cfg.n_layers) |li| try self.recordLayerForward(@intCast(li));
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
        for (0..cfg.n_layers) |li| try self.recordLayerForward(@intCast(li));
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

    // ── Internal recording helpers ────────────────────────────────────

    fn recordEmbedLookup(self: *Runner) !void {
        const total: u32 = self.cfg.n_pos * self.cfg.dim;
        try self.rec.dispatch(
            &self.k_embed,
            &.{ &self.buf_w_embed, &self.buf_token_ids, &self.buf_x_emb },
            &self.push_embed,
            ceilDiv(total, group_lin),
            1,
            1,
        );
    }

    fn recordLayerForward(self: *Runner, li: u32) !void {
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
        try self.rec.dispatch(&self.k_matmul, &.{ &self.buf_n1[i], &self.buf_w_q[i], q_matmul_dst }, &self.push_mm_q, cfg.n_pos * self.push_mm_q.n, 1, 1);
        try self.rec.dispatch(&self.k_matmul, &.{ &self.buf_n1[i], &self.buf_w_k[i], k_matmul_dst }, &self.push_mm_k, cfg.n_pos * self.push_mm_k.n, 1, 1);
        try self.rec.dispatch(&self.k_matmul, &.{ &self.buf_n1[i], &self.buf_w_v[i], &self.buf_v[i] }, &self.push_mm_v, cfg.n_pos * self.push_mm_v.n, 1, 1);
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
            const q_total: u32 = cfg.n_pos * cfg.n_heads * cfg.head_dim;
            const k_total: u32 = cfg.n_pos * cfg.n_kv_heads * cfg.head_dim;
            try self.rec.dispatch(&self.k_rope_fwd, &.{ &self.sc_q_pre, &self.buf_q[i] }, &self.push_rope_q, ceilDiv(q_total, group_lin), 1, 1);
            try self.rec.dispatch(&self.k_rope_fwd, &.{ &self.sc_k_pre, &self.buf_k[i] }, &self.push_rope_k, ceilDiv(k_total, group_lin), 1, 1);
        }
        // 5. attention scores (causal mask via -inf).
        try self.rec.dispatch(&self.k_attn_scores, &.{ &self.buf_q[i], &self.buf_k[i], &self.sc_scores }, &self.push_attn_scores, cfg.n_pos * cfg.n_heads * cfg.n_pos, 1, 1);
        // 6. softmax.
        try self.rec.dispatch(&self.k_softmax, &.{ &self.sc_scores, &self.buf_attn[i] }, &self.push_softmax, cfg.n_pos * cfg.n_heads, 1, 1);
        // 7. attention output.
        try self.rec.dispatch(&self.k_attn_output, &.{ &self.buf_attn[i], &self.buf_v[i], &self.buf_attn_out[i] }, &self.push_attn_output, cfg.n_pos * cfg.n_heads * cfg.head_dim, 1, 1);
        // 8. O projection.
        try self.rec.dispatch(&self.k_matmul, &.{ &self.buf_attn_out[i], &self.buf_w_o[i], &self.sc_o }, &self.push_mm_o, cfg.n_pos * self.push_mm_o.n, 1, 1);
        // 9. mid = x_in + o (residual).
        const add_groups: u32 = ceilDiv(self.push_n_pos_dim.n, group_lin);
        try self.rec.dispatch(&self.k_vec_add, &.{ x_in_buf, &self.sc_o, &self.buf_mid[i] }, &self.push_n_pos_dim, add_groups, 1, 1);
        // 10. RMSNorm n2.
        try self.rec.dispatch(&self.k_rms, &.{ &self.buf_mid[i], &self.buf_w_n2[i], &self.buf_n2[i] }, &self.push_rms, cfg.n_pos, 1, 1);
        // 11. W_gate matmul.
        try self.rec.dispatch(&self.k_matmul, &.{ &self.buf_n2[i], &self.buf_w_gate[i], &self.buf_pre_gate[i] }, &self.push_mm_gate, cfg.n_pos * self.push_mm_gate.n, 1, 1);
        // 12. W_up matmul.
        try self.rec.dispatch(&self.k_matmul, &.{ &self.buf_n2[i], &self.buf_w_up[i], &self.buf_up[i] }, &self.push_mm_up, cfg.n_pos * self.push_mm_up.n, 1, 1);
        // 13. SwiGLU fwd: gated = silu(pre_gate) · up.
        const swiglu_groups: u32 = ceilDiv(self.push_swiglu.n, group_lin);
        try self.rec.dispatch(&self.k_swiglu_fwd, &.{ &self.buf_pre_gate[i], &self.buf_up[i], &self.buf_gated[i] }, &self.push_swiglu, swiglu_groups, 1, 1);
        // 14. W_down matmul → ff_out.
        try self.rec.dispatch(&self.k_matmul, &.{ &self.buf_gated[i], &self.buf_w_down[i], &self.sc_ff_out }, &self.push_mm_down, cfg.n_pos * self.push_mm_down.n, 1, 1);
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

        const add_groups: u32 = ceilDiv(self.push_n_pos_dim.n, group_lin);
        const swiglu_groups: u32 = ceilDiv(self.push_swiglu.n, group_lin);

        // W_down dx + dW (treats d_y_in as d_ff_out).
        try self.rec.dispatch(&self.k_lin_dx, &.{ d_y_in, &self.buf_w_down[i], &self.sc_d_gated }, &self.push_lin_down, ceilDiv(self.push_lin_down.M, group_lwg), ceilDiv(self.push_lin_down.K, group_lwg), 1);
        try self.rec.dispatch(&self.k_lin_dw, &.{ d_y_in, &self.buf_gated[i], &self.buf_dw_down[i] }, &self.push_lin_down, ceilDiv(self.push_lin_down.N, group_lwg), ceilDiv(self.push_lin_down.K, group_lwg), 1);
        // SwiGLU bw: (d_gated, pre_gate, up) → (d_pre_gate, d_up).
        try self.rec.dispatch(&self.k_swiglu_bw, &.{ &self.sc_d_gated, &self.buf_pre_gate[i], &self.buf_up[i], &self.sc_d_pre_gate, &self.sc_d_up_grad }, &self.push_swiglu, swiglu_groups, 1, 1);
        // W_gate dx + dW (writes d_n2).
        try self.rec.dispatch(&self.k_lin_dx, &.{ &self.sc_d_pre_gate, &self.buf_w_gate[i], &self.sc_d_n2 }, &self.push_lin_gate, ceilDiv(self.push_lin_gate.M, group_lwg), ceilDiv(self.push_lin_gate.K, group_lwg), 1);
        try self.rec.dispatch(&self.k_lin_dw, &.{ &self.sc_d_pre_gate, &self.buf_n2[i], &self.buf_dw_gate[i] }, &self.push_lin_gate, ceilDiv(self.push_lin_gate.N, group_lwg), ceilDiv(self.push_lin_gate.K, group_lwg), 1);
        // W_up dx + dW + accumulate into d_n2.
        try self.rec.dispatch(&self.k_lin_dx, &.{ &self.sc_d_up_grad, &self.buf_w_up[i], &self.sc_d_n2_up }, &self.push_lin_up, ceilDiv(self.push_lin_up.M, group_lwg), ceilDiv(self.push_lin_up.K, group_lwg), 1);
        try self.rec.dispatch(&self.k_lin_dw, &.{ &self.sc_d_up_grad, &self.buf_n2[i], &self.buf_dw_up[i] }, &self.push_lin_up, ceilDiv(self.push_lin_up.N, group_lwg), ceilDiv(self.push_lin_up.K, group_lwg), 1);
        try self.rec.dispatch(&self.k_add, &.{ &self.sc_d_n2, &self.sc_d_n2_up }, &self.push_n_pos_dim, add_groups, 1, 1);
        // RMSNorm n2 bw → d_mid_norm + dw_n2_partial.
        try self.rec.dispatch(&self.k_rms_bw, &.{ &self.sc_d_n2, &self.buf_mid[i], &self.buf_w_n2[i], &self.sc_d_mid_norm, &self.buf_dw_n2_partial[i] }, &self.push_rms, cfg.n_pos, 1, 1);
        // d_y_in += d_mid_norm. From here, d_y_in holds d_mid_total.
        try self.rec.dispatch(&self.k_add, &.{ d_y_in, &self.sc_d_mid_norm }, &self.push_n_pos_dim, add_groups, 1, 1);
        // O projection dx + dW.
        try self.rec.dispatch(&self.k_lin_dx, &.{ d_y_in, &self.buf_w_o[i], &self.sc_d_attn_out }, &self.push_lin_o, ceilDiv(self.push_lin_o.M, group_lwg), ceilDiv(self.push_lin_o.K, group_lwg), 1);
        try self.rec.dispatch(&self.k_lin_dw, &.{ d_y_in, &self.buf_attn_out[i], &self.buf_dw_o[i] }, &self.push_lin_o, ceilDiv(self.push_lin_o.N, group_lwg), ceilDiv(self.push_lin_o.K, group_lwg), 1);
        // SDPA backward.
        try self.rec.dispatch(&self.k_attn_dattn, &.{ &self.sc_d_attn_out, &self.buf_v[i], &self.sc_d_attn }, &self.push_dattn, cfg.n_pos * cfg.n_heads * cfg.n_pos, 1, 1);
        try self.rec.dispatch(&self.k_attn_dv, &.{ &self.buf_attn[i], &self.sc_d_attn_out, &self.sc_dV }, &self.push_dv, cfg.n_pos * cfg.n_kv_heads * cfg.head_dim, 1, 1);
        try self.rec.dispatch(&self.k_softmax_bw, &.{ &self.sc_d_attn, &self.buf_attn[i], &self.sc_d_scores }, &self.push_softmax, cfg.n_pos * cfg.n_heads, 1, 1);
        try self.rec.dispatch(&self.k_attn_dq, &.{ &self.sc_d_scores, &self.buf_k[i], &self.sc_dQ }, &self.push_dq, cfg.n_pos * cfg.n_heads * cfg.head_dim, 1, 1);
        try self.rec.dispatch(&self.k_attn_dk, &.{ &self.sc_d_scores, &self.buf_q[i], &self.sc_dK }, &self.push_dk, cfg.n_pos * cfg.n_kv_heads * cfg.head_dim, 1, 1);

        // RoPE backward: the dQ / dK from SDPA are gradients of the
        // post-RoPE Q / K. Inverting the rotation lands them as
        // gradients of the pre-RoPE Q / K (i.e. of the rmsnorm
        // output when qk_norm is on, or of the matmul output when
        // it's off).
        if (cfg.rotary_dim > 0) {
            const q_total: u32 = cfg.n_pos * cfg.n_heads * cfg.head_dim;
            const k_total: u32 = cfg.n_pos * cfg.n_kv_heads * cfg.head_dim;
            try self.rec.dispatch(&self.k_rope_bw, &.{ &self.sc_dQ, &self.sc_dQ_pre }, &self.push_rope_q, ceilDiv(q_total, group_lin), 1, 1);
            try self.rec.dispatch(&self.k_rope_bw, &.{ &self.sc_dK, &self.sc_dK_pre }, &self.push_rope_k, ceilDiv(k_total, group_lin), 1, 1);
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

        // Q proj. Writes directly into sc_d_n1 (saves an add_in_place).
        try self.rec.dispatch(&self.k_lin_dx, &.{ dQ_lin, &self.buf_w_q[i], &self.sc_d_n1 }, &self.push_lin_q, ceilDiv(self.push_lin_q.M, group_lwg), ceilDiv(self.push_lin_q.K, group_lwg), 1);
        try self.rec.dispatch(&self.k_lin_dw, &.{ dQ_lin, &self.buf_n1[i], &self.buf_dw_q[i] }, &self.push_lin_q, ceilDiv(self.push_lin_q.N, group_lwg), ceilDiv(self.push_lin_q.K, group_lwg), 1);
        // K proj + accumulate into d_n1.
        try self.rec.dispatch(&self.k_lin_dx, &.{ dK_lin, &self.buf_w_k[i], &self.sc_d_n1_k }, &self.push_lin_k, ceilDiv(self.push_lin_k.M, group_lwg), ceilDiv(self.push_lin_k.K, group_lwg), 1);
        try self.rec.dispatch(&self.k_lin_dw, &.{ dK_lin, &self.buf_n1[i], &self.buf_dw_k[i] }, &self.push_lin_k, ceilDiv(self.push_lin_k.N, group_lwg), ceilDiv(self.push_lin_k.K, group_lwg), 1);
        try self.rec.dispatch(&self.k_add, &.{ &self.sc_d_n1, &self.sc_d_n1_k }, &self.push_n_pos_dim, add_groups, 1, 1);
        // V proj + accumulate into d_n1.
        try self.rec.dispatch(&self.k_lin_dx, &.{ &self.sc_dV, &self.buf_w_v[i], &self.sc_d_n1_v }, &self.push_lin_v, ceilDiv(self.push_lin_v.M, group_lwg), ceilDiv(self.push_lin_v.K, group_lwg), 1);
        try self.rec.dispatch(&self.k_lin_dw, &.{ &self.sc_dV, &self.buf_n1[i], &self.buf_dw_v[i] }, &self.push_lin_v, ceilDiv(self.push_lin_v.N, group_lwg), ceilDiv(self.push_lin_v.K, group_lwg), 1);
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

        const adam_embed = runtime.AdamStepPush{ .n = n_embed, .lr = lr, .beta1 = beta1, .beta2 = beta2, .eps = eps, .t = t };
        try self.rec.dispatch(&self.k_adam, &.{ &self.buf_w_embed, &self.buf_dE_embed, &self.buf_m_embed, &self.buf_v_embed }, &adam_embed, ceilDiv(n_embed, group_lin), 1, 1);

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

            try self.rec.dispatch(&self.k_adam, &.{ &self.buf_w_n1[i], &self.buf_dw_n1[i], &self.buf_m_n1[i], &self.buf_v_n1[i] }, &adam_n1, ceilDiv(dim_n, group_lin), 1, 1);
            try self.rec.dispatch(&self.k_adam, &.{ &self.buf_w_q[i], &self.buf_dw_q[i], &self.buf_m_q[i], &self.buf_v_q[i] }, &adam_q, ceilDiv(n_q_w, group_lin), 1, 1);
            try self.rec.dispatch(&self.k_adam, &.{ &self.buf_w_k[i], &self.buf_dw_k[i], &self.buf_m_k[i], &self.buf_v_k[i] }, &adam_k, ceilDiv(n_k_w, group_lin), 1, 1);
            try self.rec.dispatch(&self.k_adam, &.{ &self.buf_w_v[i], &self.buf_dw_v[i], &self.buf_m_v[i], &self.buf_v_v[i] }, &adam_v, ceilDiv(n_v_w, group_lin), 1, 1);
            try self.rec.dispatch(&self.k_adam, &.{ &self.buf_w_o[i], &self.buf_dw_o[i], &self.buf_m_o[i], &self.buf_v_o[i] }, &adam_o, ceilDiv(n_o_w, group_lin), 1, 1);
            try self.rec.dispatch(&self.k_adam, &.{ &self.buf_w_n2[i], &self.buf_dw_n2[i], &self.buf_m_n2[i], &self.buf_v_n2[i] }, &adam_n2, ceilDiv(dim_n, group_lin), 1, 1);
            try self.rec.dispatch(&self.k_adam, &.{ &self.buf_w_gate[i], &self.buf_dw_gate[i], &self.buf_m_gate[i], &self.buf_v_gate[i] }, &adam_gate, ceilDiv(n_gate_w, group_lin), 1, 1);
            try self.rec.dispatch(&self.k_adam, &.{ &self.buf_w_up[i], &self.buf_dw_up[i], &self.buf_m_up[i], &self.buf_v_up[i] }, &adam_up, ceilDiv(n_up_w, group_lin), 1, 1);
            try self.rec.dispatch(&self.k_adam, &.{ &self.buf_w_down[i], &self.buf_dw_down[i], &self.buf_m_down[i], &self.buf_v_down[i] }, &adam_down, ceilDiv(n_down_w, group_lin), 1, 1);
            if (cfg.qk_norm) {
                const adam_qn = runtime.AdamStepPush{ .n = cfg.head_dim, .lr = lr, .beta1 = beta1, .beta2 = beta2, .eps = eps, .t = t };
                try self.rec.dispatch(&self.k_adam, &.{ &self.buf_w_q_norm[i], &self.buf_dw_q_norm[i], &self.buf_m_q_norm[i], &self.buf_v_q_norm[i] }, &adam_qn, ceilDiv(cfg.head_dim, group_lin), 1, 1);
                try self.rec.dispatch(&self.k_adam, &.{ &self.buf_w_k_norm[i], &self.buf_dw_k_norm[i], &self.buf_m_k_norm[i], &self.buf_v_k_norm[i] }, &adam_qn, ceilDiv(cfg.head_dim, group_lin), 1, 1);
            }
        }

        const adam_final_norm = runtime.AdamStepPush{ .n = dim_n, .lr = lr, .beta1 = beta1, .beta2 = beta2, .eps = eps, .t = t };
        const adam_lm_head = runtime.AdamStepPush{ .n = n_lm_head, .lr = lr, .beta1 = beta1, .beta2 = beta2, .eps = eps, .t = t };
        try self.rec.dispatch(&self.k_adam, &.{ &self.buf_w_final_norm, &self.buf_dw_final_norm, &self.buf_m_final_norm, &self.buf_v_final_norm }, &adam_final_norm, ceilDiv(dim_n, group_lin), 1, 1);
        try self.rec.dispatch(&self.k_adam, &.{ &self.buf_w_lm_head, &self.buf_dw_lm_head, &self.buf_m_lm_head, &self.buf_v_lm_head }, &adam_lm_head, ceilDiv(n_lm_head, group_lin), 1, 1);
    }

    fn reduceDwPartial(self: *Runner, partial: *const buffer.Buffer, dst_dynamic: *buffer.Buffer) !void {
        const dim: usize = @intCast(self.cfg.dim);
        const n_pos: usize = @intCast(self.cfg.n_pos);
        try partial.readBack(self.ctx, f32, self.dw_partial_host);
        @memset(self.dw_reduced, 0);
        for (0..n_pos) |row| {
            const off = row * dim;
            for (0..dim) |idx| self.dw_reduced[idx] += self.dw_partial_host[off + idx];
        }
        dst_dynamic.update(f32, self.dw_reduced);
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

fn ceilDiv(num: u32, den: u32) u32 {
    return (num + den - 1) / den;
}

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
