//! Decoder + Real-model training smokes invoked via:
//! - `--decoder-stack-train-smoke` (synthetic stack, 200-step convergence)
//! - `--real-train-step-smoke` (Qwen3-0.6B, one Adam step asserts CE_after < CE_before)
//! - `--real-multi-step-smoke` (Qwen3-0.6B, 30-step overfit gate; β-6a)
//! - `--checkpoint-smoke` (toy stack save/load round-trip; β-6b)
//! - `--real-sampling-smoke` (Qwen3-0.6B, sampled-text-shift; β-6c)
//! - plus the no-arg fallthrough's CPU/GPU decoder fine-tune chain.
//! Extracted from main.zig.

const std = @import("std");
const vk = @import("../gpu/vk.zig");
const buffer = @import("../gpu/buffer.zig");
const pipeline = @import("../gpu/pipeline.zig");
const gpu_recorder = @import("../gpu/recorder.zig");
const cpu_train_decoder = @import("../cpu/train_decoder.zig");
const train_transformer = @import("../train/transformer.zig");
const train_dataset = @import("../train/dataset.zig");
const train_load_real = @import("../train/load_real.zig");
const train_sampling = @import("../train/sampling.zig");
const model_mod = @import("../model.zig");
const tokenizer_mod = @import("../tokenizer.zig");
const hf_cache = @import("../hf_cache.zig");
const runtime = @import("../runtime.zig");
const shaders = @import("shaders");

// ── End-to-end toy-decoder fine-tune smoke (Tier-2 chunk 8a) ────────
//
// Wires the chunk 1-7 backward primitives into a single decoder layer
// and trains it with Adam against a synthetic target. The target is
// the layer's *own initial output* with a small random perturbation,
// so loss must reach near zero for any working backward chain — proves
// gradients flow correctly through:
//
//   x → RMSNorm → Q/K/V projections → SDPA(causal) → o-proj → +residual
//     → RMSNorm → FF1 → ReLU → FF2 → +residual → y
//
// If any single piece's gradient is wrong, loss plateaus or diverges.
// 100 steps of Adam is plenty to drive a fresh-initialized layer onto
// a single fixed (input, target) pair when the chain is correct.

pub fn runDecoderFineTuneCpuSmoke(allocator: std.mem.Allocator) !void {
    const cfg = cpu_train_decoder.Config{
        .dim = 16,
        .n_heads = 2,
        .n_kv_heads = 2, // no GQA — exercise the simpler path here
        .head_dim = 8,
        .ff_dim = 32,
        .n_pos = 4,
        .rms_eps = 1e-5,
        .causal = true,
    };
    const dim = cfg.dim;
    const q_dim = cfg.n_heads * cfg.head_dim;
    const kv_dim = cfg.n_kv_heads * cfg.head_dim;

    // ── Initialize layer weights with small Gaussian-ish noise.
    var prng = std.Random.DefaultPrng.init(0xDEC0_DE01);
    const rng = prng.random();
    const initScale: f32 = 0.1;

    const w_n1 = try allocator.alloc(f32, dim);
    defer allocator.free(w_n1);
    const w_q = try allocator.alloc(f32, q_dim * dim);
    defer allocator.free(w_q);
    const w_k = try allocator.alloc(f32, kv_dim * dim);
    defer allocator.free(w_k);
    const w_v = try allocator.alloc(f32, kv_dim * dim);
    defer allocator.free(w_v);
    const w_o = try allocator.alloc(f32, dim * q_dim);
    defer allocator.free(w_o);
    const w_n2 = try allocator.alloc(f32, dim);
    defer allocator.free(w_n2);
    const w_gate = try allocator.alloc(f32, cfg.ff_dim * dim);
    defer allocator.free(w_gate);
    const w_up = try allocator.alloc(f32, cfg.ff_dim * dim);
    defer allocator.free(w_up);
    const w_down = try allocator.alloc(f32, dim * cfg.ff_dim);
    defer allocator.free(w_down);

    for (w_n1) |*v| v.* = 1.0; // RMSNorm gain init = 1
    for (w_n2) |*v| v.* = 1.0;
    for (w_q) |*v| v.* = (rng.float(f32) * 2.0 - 1.0) * initScale;
    for (w_k) |*v| v.* = (rng.float(f32) * 2.0 - 1.0) * initScale;
    for (w_v) |*v| v.* = (rng.float(f32) * 2.0 - 1.0) * initScale;
    for (w_o) |*v| v.* = (rng.float(f32) * 2.0 - 1.0) * initScale;
    for (w_gate) |*v| v.* = (rng.float(f32) * 2.0 - 1.0) * initScale;
    for (w_up) |*v| v.* = (rng.float(f32) * 2.0 - 1.0) * initScale;
    for (w_down) |*v| v.* = (rng.float(f32) * 2.0 - 1.0) * initScale;

    var layer = cpu_train_decoder.Layer{
        .cfg = cfg,
        .w_n1 = w_n1,
        .w_q = w_q,
        .w_k = w_k,
        .w_v = w_v,
        .w_o = w_o,
        .w_n2 = w_n2,
        .w_gate = w_gate,
        .w_up = w_up,
        .w_down = w_down,
        .w_q_norm = &.{}, // qk_norm disabled in this smoke
        .w_k_norm = &.{},
    };

    var acts = try cpu_train_decoder.allocActs(allocator, cfg);
    defer cpu_train_decoder.freeActs(allocator, &acts);

    // Random fixed input.
    for (acts.x_in) |*v| v.* = (rng.float(f32) * 2.0 - 1.0);

    // Run a forward pass with the fresh weights to capture y_init,
    // then perturb to define the target. This guarantees the target
    // is reachable in principle (it's a small step away from the
    // initial output).
    cpu_train_decoder.forward(&layer, &acts);
    const target = try allocator.alloc(f32, cfg.n_pos * dim);
    defer allocator.free(target);
    for (target, acts.y) |*tv, yv| tv.* = yv + (rng.float(f32) * 2.0 - 1.0) * 0.5;

    const initial_loss = cpu_train_decoder.mseLoss(acts.y, target);

    // ── Allocate grads + Adam state.
    var grads = cpu_train_decoder.Grads{
        .dw_n1 = try allocator.alloc(f32, w_n1.len),
        .dw_q = try allocator.alloc(f32, w_q.len),
        .dw_k = try allocator.alloc(f32, w_k.len),
        .dw_v = try allocator.alloc(f32, w_v.len),
        .dw_o = try allocator.alloc(f32, w_o.len),
        .dw_n2 = try allocator.alloc(f32, w_n2.len),
        .dw_gate = try allocator.alloc(f32, w_gate.len),
        .dw_up = try allocator.alloc(f32, w_up.len),
        .dw_down = try allocator.alloc(f32, w_down.len),
        .dw_q_norm = &.{},
        .dw_k_norm = &.{},
    };
    defer {
        allocator.free(grads.dw_n1);
        allocator.free(grads.dw_q);
        allocator.free(grads.dw_k);
        allocator.free(grads.dw_v);
        allocator.free(grads.dw_o);
        allocator.free(grads.dw_n2);
        allocator.free(grads.dw_gate);
        allocator.free(grads.dw_up);
        allocator.free(grads.dw_down);
    }

    var adam = try cpu_train_decoder.AdamState.init(allocator, &layer, 1e-2);
    defer adam.deinit(allocator);

    // ── Train.
    const n_steps: usize = 100;
    var final_loss = initial_loss;
    for (1..n_steps + 1) |_| {
        cpu_train_decoder.forward(&layer, &acts);
        final_loss = cpu_train_decoder.mseLoss(acts.y, target);
        grads.zero();
        try cpu_train_decoder.backward(allocator, &layer, &acts, target, &grads);
        cpu_train_decoder.adamStep(&adam, &layer, &grads);
    }

    // Loss must drop by at least 100× over 100 Adam steps on a
    // reachable target. If the gradient chain is broken, loss either
    // plateaus or diverges. Adam's mid-descent momentum bumps don't
    // matter — only the start-vs-end ratio does.
    const ratio = final_loss / initial_loss;
    if (ratio > 1e-2) {
        std.debug.print(
            "decoder fine-tune: initial_loss={d:.6} final_loss={d:.6} ratio={d:.4}\n",
            .{ initial_loss, final_loss, ratio },
        );
        return error.LossDidNotDecrease;
    }

    std.debug.print(
        "PASS decoder fine-tune CPU (dim={d} heads={d} ff_dim={d} n_pos={d}; loss {d:.6} → {d:.6} ({e:.2}× drop) over {d} Adam steps)\n",
        .{ dim, cfg.n_heads, cfg.ff_dim, cfg.n_pos, initial_loss, final_loss, 1.0 / ratio, n_steps },
    );
}

// ── chunk 8c-α-1: stack of decoder layers + lm_head + softmax-CE,
// CPU oracle. Validates that the stack-level gradient chain composes
// correctly: chunk-8a single-layer `backwardFromDy` chains via
// `d_x_in → next layer's d_y`, plus the new pieces (embedding + final
// rmsnorm + lm_head linear + softmax-CE). Same convergence-on-a-
// reachable-target shape as 8a — only the architecture and loss
// changed.
pub fn runDecoderStackFineTuneCpuSmoke(allocator: std.mem.Allocator) !void {
    const cfg = cpu_train_decoder.StackConfig{
        .base = .{
            .dim = 16,
            .n_heads = 2,
            .n_kv_heads = 2,
            .head_dim = 8,
            .ff_dim = 32,
            .n_pos = 4,
            .rms_eps = 1e-5,
            .causal = true,
        },
        .n_layers = 2,
        .vocab_size = 8,
    };
    const dim = cfg.base.dim;
    const n_pos = cfg.base.n_pos;
    const vocab = cfg.vocab_size;
    const q_dim = cfg.base.n_heads * cfg.base.head_dim;
    const kv_dim = cfg.base.n_kv_heads * cfg.base.head_dim;

    var prng = std.Random.DefaultPrng.init(0xC0_DE_AC_01);
    const rng = prng.random();
    const initScale: f32 = 0.1;

    // ── Embedding + final norm + lm_head.
    const w_embed = try allocator.alloc(f32, vocab * dim);
    defer allocator.free(w_embed);
    for (w_embed) |*v| v.* = (rng.float(f32) * 2.0 - 1.0) * initScale;
    const w_final_norm = try allocator.alloc(f32, dim);
    defer allocator.free(w_final_norm);
    for (w_final_norm) |*v| v.* = 1.0;
    const w_lm_head = try allocator.alloc(f32, vocab * dim);
    defer allocator.free(w_lm_head);
    for (w_lm_head) |*v| v.* = (rng.float(f32) * 2.0 - 1.0) * initScale;

    // ── Per-layer weight buffers.
    const layers = try allocator.alloc(cpu_train_decoder.Layer, cfg.n_layers);
    defer allocator.free(layers);
    const layer_weight_buffers = try allocator.alloc([9][]f32, cfg.n_layers);
    defer allocator.free(layer_weight_buffers);
    defer for (layer_weight_buffers) |slots| for (slots) |s| allocator.free(s);

    for (layers, layer_weight_buffers) |*layer, *slots| {
        const w_n1 = try allocator.alloc(f32, dim);
        const w_q = try allocator.alloc(f32, q_dim * dim);
        const w_k = try allocator.alloc(f32, kv_dim * dim);
        const w_v = try allocator.alloc(f32, kv_dim * dim);
        const w_o = try allocator.alloc(f32, dim * q_dim);
        const w_n2 = try allocator.alloc(f32, dim);
        const w_gate = try allocator.alloc(f32, cfg.base.ff_dim * dim);
        const w_up = try allocator.alloc(f32, cfg.base.ff_dim * dim);
        const w_down = try allocator.alloc(f32, dim * cfg.base.ff_dim);
        for (w_n1) |*v| v.* = 1.0;
        for (w_n2) |*v| v.* = 1.0;
        for (w_q) |*v| v.* = (rng.float(f32) * 2.0 - 1.0) * initScale;
        for (w_k) |*v| v.* = (rng.float(f32) * 2.0 - 1.0) * initScale;
        for (w_v) |*v| v.* = (rng.float(f32) * 2.0 - 1.0) * initScale;
        for (w_o) |*v| v.* = (rng.float(f32) * 2.0 - 1.0) * initScale;
        for (w_gate) |*v| v.* = (rng.float(f32) * 2.0 - 1.0) * initScale;
        for (w_up) |*v| v.* = (rng.float(f32) * 2.0 - 1.0) * initScale;
        for (w_down) |*v| v.* = (rng.float(f32) * 2.0 - 1.0) * initScale;
        slots.* = .{ w_n1, w_q, w_k, w_v, w_o, w_n2, w_gate, w_up, w_down };
        layer.* = .{
            .cfg = cfg.base,
            .w_n1 = w_n1,
            .w_q = w_q,
            .w_k = w_k,
            .w_v = w_v,
            .w_o = w_o,
            .w_n2 = w_n2,
            .w_gate = w_gate,
            .w_up = w_up,
            .w_down = w_down,
            .w_q_norm = &.{}, // qk_norm disabled in this smoke
            .w_k_norm = &.{},
        };
    }

    var stack = cpu_train_decoder.Stack{
        .cfg = cfg,
        .embed = w_embed,
        .layers = layers,
        .final_norm = w_final_norm,
        .lm_head = w_lm_head,
    };

    var acts = try cpu_train_decoder.allocStackActs(allocator, cfg);
    defer cpu_train_decoder.freeStackActs(allocator, &acts);

    // ── Synthetic input + reachable target.
    // Random token IDs in [0, vocab); target IDs likewise.
    const token_ids = try allocator.alloc(u32, n_pos);
    defer allocator.free(token_ids);
    for (token_ids) |*tid| tid.* = rng.intRangeLessThan(u32, 0, @intCast(vocab));
    const target_ids = try allocator.alloc(u32, n_pos);
    defer allocator.free(target_ids);
    for (target_ids) |*tid| tid.* = rng.intRangeLessThan(u32, 0, @intCast(vocab));

    acts.token_ids = token_ids;
    cpu_train_decoder.stackForward(&stack, &acts);
    const initial_loss = cpu_train_decoder.softmaxCeLoss(acts.logits, target_ids, n_pos, vocab);

    // ── Grads + Adam.
    var grads = try cpu_train_decoder.allocStackGrads(allocator, &stack);
    defer cpu_train_decoder.freeStackGrads(allocator, &grads);

    var adam = try cpu_train_decoder.stackAdamInit(allocator, &stack, 1e-2);
    defer cpu_train_decoder.stackAdamDeinit(&adam, allocator);

    // ── Train.
    const n_steps: usize = 200;
    var final_loss = initial_loss;
    for (1..n_steps + 1) |_| {
        cpu_train_decoder.stackForward(&stack, &acts);
        final_loss = cpu_train_decoder.softmaxCeLoss(acts.logits, target_ids, n_pos, vocab);
        grads.zero();
        try cpu_train_decoder.stackBackward(allocator, &stack, &acts, target_ids, &grads);
        cpu_train_decoder.stackAdamStep(&adam, &stack, &grads);
    }

    // Cross-entropy on a tiny vocab with a reachable target should
    // collapse loss by orders of magnitude. The single-layer 8a smoke
    // hit 5+ orders of magnitude in 100 steps; CE through a 2-layer
    // stack is somewhat slower but should still clear 1e-2 inside 200.
    const ratio = final_loss / initial_loss;
    if (ratio > 1e-2) {
        std.debug.print(
            "stack fine-tune: initial_loss={d:.6} final_loss={d:.6} ratio={d:.4}\n",
            .{ initial_loss, final_loss, ratio },
        );
        return error.LossDidNotDecrease;
    }

    std.debug.print(
        "PASS decoder stack fine-tune CPU (n_layers={d} dim={d} vocab={d} n_pos={d}; CE loss {d:.6} → {d:.6} ({e:.2}× drop) over {d} Adam steps)\n",
        .{ cfg.n_layers, dim, vocab, n_pos, initial_loss, final_loss, 1.0 / ratio, n_steps },
    );
}

// ── chunk 8c-α-2: stack GPU backward parity vs CPU oracle ──────────
//
// Same role as the chunk-8b stage-A parity smoke, scaled up to the
// stack: CPU forward + CPU backward give an oracle for every
// gradient (embedding, per-layer × N, final_norm, lm_head); the GPU
// recorder replays the stack-level backward chain over the saved
// activations and we compare each slice. The per-layer 24-dispatch
// chain (the 8b stage-A 23 dispatches plus an `add_in_place` to
// fold in the residual `mid = x_in + o` contribution into d_x_in)
// repeats N times, so we factor it into a helper.

const StackBwKernels = struct {
    lin_dx: *const pipeline.Kernel,
    lin_dw: *const pipeline.Kernel,
    swiglu_bw: *const pipeline.Kernel,
    rms_bw: *const pipeline.Kernel,
    attn_dattn: *const pipeline.Kernel,
    attn_dv: *const pipeline.Kernel,
    attn_dq: *const pipeline.Kernel,
    attn_dk: *const pipeline.Kernel,
    softmax_bw: *const pipeline.Kernel,
    add: *const pipeline.Kernel,
};

const StackLayerBufs = struct {
    // Saved activations (read).
    x_in: *const buffer.Buffer,
    n1: *const buffer.Buffer,
    q: *const buffer.Buffer,
    k: *const buffer.Buffer,
    v: *const buffer.Buffer,
    attn: *const buffer.Buffer,
    attn_out: *const buffer.Buffer,
    mid: *const buffer.Buffer,
    n2: *const buffer.Buffer,
    pre_gate: *const buffer.Buffer,
    up: *const buffer.Buffer,
    gated: *const buffer.Buffer,
    // Weights (read).
    w_n1: *const buffer.Buffer,
    w_q: *const buffer.Buffer,
    w_k: *const buffer.Buffer,
    w_v: *const buffer.Buffer,
    w_o: *const buffer.Buffer,
    w_n2: *const buffer.Buffer,
    w_gate: *const buffer.Buffer,
    w_up: *const buffer.Buffer,
    w_down: *const buffer.Buffer,
    // Grad outputs (write — RMSNorm gains come back as per-row partials).
    dw_n1_partial: *const buffer.Buffer,
    dw_q: *const buffer.Buffer,
    dw_k: *const buffer.Buffer,
    dw_v: *const buffer.Buffer,
    dw_o: *const buffer.Buffer,
    dw_n2_partial: *const buffer.Buffer,
    dw_gate: *const buffer.Buffer,
    dw_up: *const buffer.Buffer,
    dw_down: *const buffer.Buffer,
};

const StackBwScratch = struct {
    d_gated: *const buffer.Buffer,
    d_pre_gate: *const buffer.Buffer,
    d_up: *const buffer.Buffer,
    d_n2: *const buffer.Buffer,
    d_n2_up: *const buffer.Buffer,
    d_mid_norm: *const buffer.Buffer,
    d_attn_out: *const buffer.Buffer,
    d_attn: *const buffer.Buffer,
    d_scores: *const buffer.Buffer,
    dQ: *const buffer.Buffer,
    dK: *const buffer.Buffer,
    dV: *const buffer.Buffer,
    d_n1: *const buffer.Buffer,
    d_n1_k: *const buffer.Buffer,
    d_n1_v: *const buffer.Buffer,
};

/// Records the 24-dispatch per-layer backward chain into the recorder.
/// `d_y_in` is *mutated in place* into `d_mid_total` by `add_in_place`
/// after RMSNorm-n2 backward — same aliasing as the chunk-8a CPU oracle.
/// `d_x_in_out` receives the layer's input gradient (used as the next
/// layer's `d_y_in` for the iteration that follows, or as the
/// embedding-backward input for layer 0).
fn recordStackLayerBackward(
    rec: *gpu_recorder.Recorder,
    kernels: StackBwKernels,
    bufs: StackLayerBufs,
    scratch: StackBwScratch,
    pushes: struct {
        lin_down: *const runtime.LinearBatchedPush,
        lin_gate: *const runtime.LinearBatchedPush,
        lin_up: *const runtime.LinearBatchedPush,
        lin_o: *const runtime.LinearBatchedPush,
        lin_q: *const runtime.LinearBatchedPush,
        lin_k: *const runtime.LinearBatchedPush,
        lin_v: *const runtime.LinearBatchedPush,
        swiglu: *const runtime.SwigluPush,
        rms: *const runtime.RmsnormPush,
        add_dim: *const runtime.AddInPlacePush,
        softmax: *const runtime.SoftmaxPush,
        dattn: *const runtime.AttnBackwardDattnPush,
        dv: *const runtime.AttnBackwardDvPush,
        dq: *const runtime.AttnBackwardDqPush,
        dk: *const runtime.AttnBackwardDkPush,
    },
    shape: struct { n_pos: u32, n_heads: u32, n_kv_heads: u32, head_dim: u32 },
    d_y_in: *const buffer.Buffer,
    d_x_in_out: *const buffer.Buffer,
) !void {
    const lwg: u32 = 16;
    const groupsLin: u32 = 256;
    const groupsCeil = struct {
        fn f(n: u32) u32 {
            return (n + lwg - 1) / lwg;
        }
    }.f;
    const addGroups: u32 = (pushes.add_dim.n + groupsLin - 1) / groupsLin;
    const swigluGroups: u32 = (pushes.swiglu.n + groupsLin - 1) / groupsLin;

    // W_down dx + dW.
    try rec.dispatch(kernels.lin_dx, &.{ d_y_in, bufs.w_down, scratch.d_gated }, pushes.lin_down, groupsCeil(pushes.lin_down.M), groupsCeil(pushes.lin_down.K), 1);
    try rec.dispatch(kernels.lin_dw, &.{ d_y_in, bufs.gated, bufs.dw_down }, pushes.lin_down, groupsCeil(pushes.lin_down.N), groupsCeil(pushes.lin_down.K), 1);

    // SwiGLU backward: (d_gated, pre_gate, up) → (d_pre_gate, d_up).
    try rec.dispatch(kernels.swiglu_bw, &.{ scratch.d_gated, bufs.pre_gate, bufs.up, scratch.d_pre_gate, scratch.d_up }, pushes.swiglu, swigluGroups, 1, 1);

    // W_gate dx + dW (writes d_n2; W_up's dx accumulates into d_n2_up next).
    try rec.dispatch(kernels.lin_dx, &.{ scratch.d_pre_gate, bufs.w_gate, scratch.d_n2 }, pushes.lin_gate, groupsCeil(pushes.lin_gate.M), groupsCeil(pushes.lin_gate.K), 1);
    try rec.dispatch(kernels.lin_dw, &.{ scratch.d_pre_gate, bufs.n2, bufs.dw_gate }, pushes.lin_gate, groupsCeil(pushes.lin_gate.N), groupsCeil(pushes.lin_gate.K), 1);

    // W_up dx + dW + accumulate into d_n2.
    try rec.dispatch(kernels.lin_dx, &.{ scratch.d_up, bufs.w_up, scratch.d_n2_up }, pushes.lin_up, groupsCeil(pushes.lin_up.M), groupsCeil(pushes.lin_up.K), 1);
    try rec.dispatch(kernels.lin_dw, &.{ scratch.d_up, bufs.n2, bufs.dw_up }, pushes.lin_up, groupsCeil(pushes.lin_up.N), groupsCeil(pushes.lin_up.K), 1);
    try rec.dispatch(kernels.add, &.{ scratch.d_n2, scratch.d_n2_up }, pushes.add_dim, addGroups, 1, 1);

    // RMSNorm n2 backward → d_mid_norm + dw_n2_partial.
    try rec.dispatch(kernels.rms_bw, &.{ scratch.d_n2, bufs.mid, bufs.w_n2, scratch.d_mid_norm, bufs.dw_n2_partial }, pushes.rms, shape.n_pos, 1, 1);

    // d_y_in += d_mid_norm. From here, d_y_in holds d_mid_total —
    // the gradient flowing into both the o-projection AND (via the
    // residual) back to x_in.
    try rec.dispatch(kernels.add, &.{ d_y_in, scratch.d_mid_norm }, pushes.add_dim, addGroups, 1, 1);

    // O projection dx + dW (treats d_y_in as d_o).
    try rec.dispatch(kernels.lin_dx, &.{ d_y_in, bufs.w_o, scratch.d_attn_out }, pushes.lin_o, groupsCeil(pushes.lin_o.M), groupsCeil(pushes.lin_o.K), 1);
    try rec.dispatch(kernels.lin_dw, &.{ d_y_in, bufs.attn_out, bufs.dw_o }, pushes.lin_o, groupsCeil(pushes.lin_o.N), groupsCeil(pushes.lin_o.K), 1);

    // SDPA backward.
    try rec.dispatch(kernels.attn_dattn, &.{ scratch.d_attn_out, bufs.v, scratch.d_attn }, pushes.dattn, shape.n_pos * shape.n_heads * shape.n_pos, 1, 1);
    try rec.dispatch(kernels.attn_dv, &.{ bufs.attn, scratch.d_attn_out, scratch.dV }, pushes.dv, shape.n_pos * shape.n_kv_heads * shape.head_dim, 1, 1);
    try rec.dispatch(kernels.softmax_bw, &.{ scratch.d_attn, bufs.attn, scratch.d_scores }, pushes.softmax, shape.n_pos * shape.n_heads, 1, 1);
    try rec.dispatch(kernels.attn_dq, &.{ scratch.d_scores, bufs.k, scratch.dQ }, pushes.dq, shape.n_pos * shape.n_heads * shape.head_dim, 1, 1);
    try rec.dispatch(kernels.attn_dk, &.{ scratch.d_scores, bufs.q, scratch.dK }, pushes.dk, shape.n_pos * shape.n_kv_heads * shape.head_dim, 1, 1);

    // Q proj. Writes directly into scratch.d_n1 (rather than a separate
    // d_n1_q buffer) to save one add_in_place — K and V then accumulate
    // into d_n1.
    try rec.dispatch(kernels.lin_dx, &.{ scratch.dQ, bufs.w_q, scratch.d_n1 }, pushes.lin_q, groupsCeil(pushes.lin_q.M), groupsCeil(pushes.lin_q.K), 1);
    try rec.dispatch(kernels.lin_dw, &.{ scratch.dQ, bufs.n1, bufs.dw_q }, pushes.lin_q, groupsCeil(pushes.lin_q.N), groupsCeil(pushes.lin_q.K), 1);

    // K proj + accumulate into d_n1.
    try rec.dispatch(kernels.lin_dx, &.{ scratch.dK, bufs.w_k, scratch.d_n1_k }, pushes.lin_k, groupsCeil(pushes.lin_k.M), groupsCeil(pushes.lin_k.K), 1);
    try rec.dispatch(kernels.lin_dw, &.{ scratch.dK, bufs.n1, bufs.dw_k }, pushes.lin_k, groupsCeil(pushes.lin_k.N), groupsCeil(pushes.lin_k.K), 1);
    try rec.dispatch(kernels.add, &.{ scratch.d_n1, scratch.d_n1_k }, pushes.add_dim, addGroups, 1, 1);

    // V proj + accumulate into d_n1.
    try rec.dispatch(kernels.lin_dx, &.{ scratch.dV, bufs.w_v, scratch.d_n1_v }, pushes.lin_v, groupsCeil(pushes.lin_v.M), groupsCeil(pushes.lin_v.K), 1);
    try rec.dispatch(kernels.lin_dw, &.{ scratch.dV, bufs.n1, bufs.dw_v }, pushes.lin_v, groupsCeil(pushes.lin_v.N), groupsCeil(pushes.lin_v.K), 1);
    try rec.dispatch(kernels.add, &.{ scratch.d_n1, scratch.d_n1_v }, pushes.add_dim, addGroups, 1, 1);

    // RMSNorm n1 backward → writes the rmsnorm.dx contribution to
    // d_x_in_out + dw_n1_partial.
    try rec.dispatch(kernels.rms_bw, &.{ scratch.d_n1, bufs.x_in, bufs.w_n1, d_x_in_out, bufs.dw_n1_partial }, pushes.rms, shape.n_pos, 1, 1);

    // d_x_in_out += d_y_in (the residual contribution of `mid = x_in + o`,
    // currently held in d_y_in's mutated form = d_mid_total). This
    // is the second residual path that the chunk-8a backward
    // discarded — must be present for stack training to flow grads
    // through residuals correctly.
    try rec.dispatch(kernels.add, &.{ d_x_in_out, d_y_in }, pushes.add_dim, addGroups, 1, 1);
}

pub fn runDecoderStackBackwardGpuParitySmoke(allocator: std.mem.Allocator) !void {
    const cfg = cpu_train_decoder.StackConfig{
        .base = .{
            .dim = 16,
            .n_heads = 2,
            .n_kv_heads = 2,
            .head_dim = 8,
            .ff_dim = 32,
            .n_pos = 4,
            .rms_eps = 1e-5,
            .causal = true,
        },
        .n_layers = 2,
        .vocab_size = 8,
    };
    const dim = cfg.base.dim;
    const n_pos = cfg.base.n_pos;
    const n_heads = cfg.base.n_heads;
    const n_kv_heads = cfg.base.n_kv_heads;
    const head_dim = cfg.base.head_dim;
    const ff_dim = cfg.base.ff_dim;
    const vocab = cfg.vocab_size;
    const q_dim = n_heads * head_dim;
    const kv_dim = n_kv_heads * head_dim;
    const heads_per_kv: u32 = @intCast(n_heads / n_kv_heads);
    const inv_sqrt_d: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(head_dim)));
    const scores_total = n_pos * n_heads * n_pos;

    // ── Identical RNG to runDecoderStackFineTuneCpuSmoke so weights +
    //    inputs match.
    var prng = std.Random.DefaultPrng.init(0xC0_DE_AC_01);
    const rng = prng.random();
    const initScale: f32 = 0.1;

    const w_embed = try allocator.alloc(f32, vocab * dim);
    defer allocator.free(w_embed);
    for (w_embed) |*v| v.* = (rng.float(f32) * 2.0 - 1.0) * initScale;
    const w_final_norm = try allocator.alloc(f32, dim);
    defer allocator.free(w_final_norm);
    for (w_final_norm) |*v| v.* = 1.0;
    const w_lm_head = try allocator.alloc(f32, vocab * dim);
    defer allocator.free(w_lm_head);
    for (w_lm_head) |*v| v.* = (rng.float(f32) * 2.0 - 1.0) * initScale;

    const layers = try allocator.alloc(cpu_train_decoder.Layer, cfg.n_layers);
    defer allocator.free(layers);
    const layer_weight_buffers = try allocator.alloc([9][]f32, cfg.n_layers);
    defer allocator.free(layer_weight_buffers);
    defer for (layer_weight_buffers) |slots| for (slots) |s| allocator.free(s);

    for (layers, layer_weight_buffers) |*layer, *slots| {
        const w_n1 = try allocator.alloc(f32, dim);
        const w_q = try allocator.alloc(f32, q_dim * dim);
        const w_k = try allocator.alloc(f32, kv_dim * dim);
        const w_v = try allocator.alloc(f32, kv_dim * dim);
        const w_o = try allocator.alloc(f32, dim * q_dim);
        const w_n2 = try allocator.alloc(f32, dim);
        const w_gate = try allocator.alloc(f32, ff_dim * dim);
        const w_up = try allocator.alloc(f32, ff_dim * dim);
        const w_down = try allocator.alloc(f32, dim * ff_dim);
        for (w_n1) |*v| v.* = 1.0;
        for (w_n2) |*v| v.* = 1.0;
        for (w_q) |*v| v.* = (rng.float(f32) * 2.0 - 1.0) * initScale;
        for (w_k) |*v| v.* = (rng.float(f32) * 2.0 - 1.0) * initScale;
        for (w_v) |*v| v.* = (rng.float(f32) * 2.0 - 1.0) * initScale;
        for (w_o) |*v| v.* = (rng.float(f32) * 2.0 - 1.0) * initScale;
        for (w_gate) |*v| v.* = (rng.float(f32) * 2.0 - 1.0) * initScale;
        for (w_up) |*v| v.* = (rng.float(f32) * 2.0 - 1.0) * initScale;
        for (w_down) |*v| v.* = (rng.float(f32) * 2.0 - 1.0) * initScale;
        slots.* = .{ w_n1, w_q, w_k, w_v, w_o, w_n2, w_gate, w_up, w_down };
        layer.* = .{
            .cfg = cfg.base,
            .w_n1 = w_n1,
            .w_q = w_q,
            .w_k = w_k,
            .w_v = w_v,
            .w_o = w_o,
            .w_n2 = w_n2,
            .w_gate = w_gate,
            .w_up = w_up,
            .w_down = w_down,
            .w_q_norm = &.{}, // qk_norm disabled in this smoke
            .w_k_norm = &.{},
        };
    }

    var stack = cpu_train_decoder.Stack{
        .cfg = cfg,
        .embed = w_embed,
        .layers = layers,
        .final_norm = w_final_norm,
        .lm_head = w_lm_head,
    };

    var acts = try cpu_train_decoder.allocStackActs(allocator, cfg);
    defer cpu_train_decoder.freeStackActs(allocator, &acts);

    const token_ids = try allocator.alloc(u32, n_pos);
    defer allocator.free(token_ids);
    for (token_ids) |*tid| tid.* = rng.intRangeLessThan(u32, 0, @intCast(vocab));
    const target_ids = try allocator.alloc(u32, n_pos);
    defer allocator.free(target_ids);
    for (target_ids) |*tid| tid.* = rng.intRangeLessThan(u32, 0, @intCast(vocab));

    acts.token_ids = token_ids;
    cpu_train_decoder.stackForward(&stack, &acts);

    // ── CPU oracle: full backward.
    var grads_cpu = try cpu_train_decoder.allocStackGrads(allocator, &stack);
    defer cpu_train_decoder.freeStackGrads(allocator, &grads_cpu);
    grads_cpu.zero();
    try cpu_train_decoder.stackBackward(allocator, &stack, &acts, target_ids, &grads_cpu);

    // ── One-hot target tensor for the GPU softmax_ce_loss_grad shader.
    const target_one_hot = try allocator.alloc(f32, n_pos * vocab);
    defer allocator.free(target_one_hot);
    @memset(target_one_hot, 0);
    for (target_ids, 0..) |tid, p| target_one_hot[p * vocab + @as(usize, tid)] = 1.0;

    // ── GPU bring-up.
    var ctx = try vk.Context.init(allocator);
    defer ctx.deinit();

    var k_lin_dx = try pipeline.Kernel.init(&ctx, &shaders.linear_backward_dx_batched, 3, @sizeOf(runtime.LinearBatchedPush));
    defer k_lin_dx.deinit();
    var k_lin_dw = try pipeline.Kernel.init(&ctx, &shaders.linear_backward_dw_batched, 3, @sizeOf(runtime.LinearBatchedPush));
    defer k_lin_dw.deinit();
    var k_swiglu_bw = try pipeline.Kernel.init(&ctx, &shaders.swiglu_backward, 5, @sizeOf(runtime.SwigluPush));
    defer k_swiglu_bw.deinit();
    var k_rms_bw = try pipeline.Kernel.init(&ctx, &shaders.rmsnorm_backward, 5, @sizeOf(runtime.RmsnormPush));
    defer k_rms_bw.deinit();
    var k_attn_dattn = try pipeline.Kernel.init(&ctx, &shaders.attn_backward_dattn, 3, @sizeOf(runtime.AttnBackwardDattnPush));
    defer k_attn_dattn.deinit();
    var k_attn_dv = try pipeline.Kernel.init(&ctx, &shaders.attn_backward_dv, 3, @sizeOf(runtime.AttnBackwardDvPush));
    defer k_attn_dv.deinit();
    var k_attn_dq = try pipeline.Kernel.init(&ctx, &shaders.attn_backward_dq, 3, @sizeOf(runtime.AttnBackwardDqPush));
    defer k_attn_dq.deinit();
    var k_attn_dk = try pipeline.Kernel.init(&ctx, &shaders.attn_backward_dk, 3, @sizeOf(runtime.AttnBackwardDkPush));
    defer k_attn_dk.deinit();
    var k_softmax_bw = try pipeline.Kernel.init(&ctx, &shaders.softmax_backward, 3, @sizeOf(runtime.SoftmaxPush));
    defer k_softmax_bw.deinit();
    var k_add = try pipeline.Kernel.init(&ctx, &shaders.add_in_place, 2, @sizeOf(runtime.AddInPlacePush));
    defer k_add.deinit();
    var k_ce_loss_grad = try pipeline.Kernel.init(&ctx, &shaders.softmax_ce_loss_grad_batched_v2, 3, @sizeOf(runtime.SoftmaxCeLossGradPush));
    defer k_ce_loss_grad.deinit();
    var k_embed_bw = try pipeline.Kernel.init(&ctx, &shaders.embedding_backward, 3, @sizeOf(runtime.EmbeddingBackwardPush));
    defer k_embed_bw.deinit();

    const f32sz = @sizeOf(f32);

    // ── Stack-level inputs.
    var buf_logits = try buffer.Buffer.initStatic(&ctx, f32, acts.logits);
    defer buf_logits.deinit(ctx.device);
    var buf_target_oh = try buffer.Buffer.initStatic(&ctx, f32, target_one_hot);
    defer buf_target_oh.deinit(ctx.device);
    var buf_final_norm_out = try buffer.Buffer.initStatic(&ctx, f32, acts.final_norm_out);
    defer buf_final_norm_out.deinit(ctx.device);
    var buf_w_lm_head = try buffer.Buffer.initStatic(&ctx, f32, w_lm_head);
    defer buf_w_lm_head.deinit(ctx.device);
    var buf_w_final_norm = try buffer.Buffer.initStatic(&ctx, f32, w_final_norm);
    defer buf_w_final_norm.deinit(ctx.device);
    var buf_token_ids = try buffer.Buffer.initStatic(&ctx, u32, token_ids);
    defer buf_token_ids.deinit(ctx.device);

    // ── Per-layer weight + activation buffers (heap-allocated arrays).
    const f32_vk_buf_alloc = struct {
        fn deinitMany(buffers: []buffer.Buffer, dev: vk.c.VkDevice) void {
            for (buffers) |*b| b.deinit(dev);
        }
    };
    _ = f32_vk_buf_alloc;

    const buf_w_n1 = try allocator.alloc(buffer.Buffer, cfg.n_layers);
    defer { for (buf_w_n1) |*b| b.deinit(ctx.device); allocator.free(buf_w_n1); }
    const buf_w_q = try allocator.alloc(buffer.Buffer, cfg.n_layers);
    defer { for (buf_w_q) |*b| b.deinit(ctx.device); allocator.free(buf_w_q); }
    const buf_w_k = try allocator.alloc(buffer.Buffer, cfg.n_layers);
    defer { for (buf_w_k) |*b| b.deinit(ctx.device); allocator.free(buf_w_k); }
    const buf_w_v = try allocator.alloc(buffer.Buffer, cfg.n_layers);
    defer { for (buf_w_v) |*b| b.deinit(ctx.device); allocator.free(buf_w_v); }
    const buf_w_o = try allocator.alloc(buffer.Buffer, cfg.n_layers);
    defer { for (buf_w_o) |*b| b.deinit(ctx.device); allocator.free(buf_w_o); }
    const buf_w_n2 = try allocator.alloc(buffer.Buffer, cfg.n_layers);
    defer { for (buf_w_n2) |*b| b.deinit(ctx.device); allocator.free(buf_w_n2); }
    const buf_w_gate = try allocator.alloc(buffer.Buffer, cfg.n_layers);
    defer { for (buf_w_gate) |*b| b.deinit(ctx.device); allocator.free(buf_w_gate); }
    const buf_w_up = try allocator.alloc(buffer.Buffer, cfg.n_layers);
    defer { for (buf_w_up) |*b| b.deinit(ctx.device); allocator.free(buf_w_up); }
    const buf_w_down = try allocator.alloc(buffer.Buffer, cfg.n_layers);
    defer { for (buf_w_down) |*b| b.deinit(ctx.device); allocator.free(buf_w_down); }

    const buf_x_in = try allocator.alloc(buffer.Buffer, cfg.n_layers);
    defer { for (buf_x_in) |*b| b.deinit(ctx.device); allocator.free(buf_x_in); }
    const buf_n1 = try allocator.alloc(buffer.Buffer, cfg.n_layers);
    defer { for (buf_n1) |*b| b.deinit(ctx.device); allocator.free(buf_n1); }
    const buf_q = try allocator.alloc(buffer.Buffer, cfg.n_layers);
    defer { for (buf_q) |*b| b.deinit(ctx.device); allocator.free(buf_q); }
    const buf_k = try allocator.alloc(buffer.Buffer, cfg.n_layers);
    defer { for (buf_k) |*b| b.deinit(ctx.device); allocator.free(buf_k); }
    const buf_v = try allocator.alloc(buffer.Buffer, cfg.n_layers);
    defer { for (buf_v) |*b| b.deinit(ctx.device); allocator.free(buf_v); }
    const buf_attn = try allocator.alloc(buffer.Buffer, cfg.n_layers);
    defer { for (buf_attn) |*b| b.deinit(ctx.device); allocator.free(buf_attn); }
    const buf_attn_out = try allocator.alloc(buffer.Buffer, cfg.n_layers);
    defer { for (buf_attn_out) |*b| b.deinit(ctx.device); allocator.free(buf_attn_out); }
    const buf_mid = try allocator.alloc(buffer.Buffer, cfg.n_layers);
    defer { for (buf_mid) |*b| b.deinit(ctx.device); allocator.free(buf_mid); }
    const buf_n2 = try allocator.alloc(buffer.Buffer, cfg.n_layers);
    defer { for (buf_n2) |*b| b.deinit(ctx.device); allocator.free(buf_n2); }
    const buf_pre_gate = try allocator.alloc(buffer.Buffer, cfg.n_layers);
    defer { for (buf_pre_gate) |*b| b.deinit(ctx.device); allocator.free(buf_pre_gate); }
    const buf_up = try allocator.alloc(buffer.Buffer, cfg.n_layers);
    defer { for (buf_up) |*b| b.deinit(ctx.device); allocator.free(buf_up); }
    const buf_gated = try allocator.alloc(buffer.Buffer, cfg.n_layers);
    defer { for (buf_gated) |*b| b.deinit(ctx.device); allocator.free(buf_gated); }

    const buf_dw_n1_partial = try allocator.alloc(buffer.Buffer, cfg.n_layers);
    defer { for (buf_dw_n1_partial) |*b| b.deinit(ctx.device); allocator.free(buf_dw_n1_partial); }
    const buf_dw_q = try allocator.alloc(buffer.Buffer, cfg.n_layers);
    defer { for (buf_dw_q) |*b| b.deinit(ctx.device); allocator.free(buf_dw_q); }
    const buf_dw_k = try allocator.alloc(buffer.Buffer, cfg.n_layers);
    defer { for (buf_dw_k) |*b| b.deinit(ctx.device); allocator.free(buf_dw_k); }
    const buf_dw_v = try allocator.alloc(buffer.Buffer, cfg.n_layers);
    defer { for (buf_dw_v) |*b| b.deinit(ctx.device); allocator.free(buf_dw_v); }
    const buf_dw_o = try allocator.alloc(buffer.Buffer, cfg.n_layers);
    defer { for (buf_dw_o) |*b| b.deinit(ctx.device); allocator.free(buf_dw_o); }
    const buf_dw_n2_partial = try allocator.alloc(buffer.Buffer, cfg.n_layers);
    defer { for (buf_dw_n2_partial) |*b| b.deinit(ctx.device); allocator.free(buf_dw_n2_partial); }
    const buf_dw_gate = try allocator.alloc(buffer.Buffer, cfg.n_layers);
    defer { for (buf_dw_gate) |*b| b.deinit(ctx.device); allocator.free(buf_dw_gate); }
    const buf_dw_up = try allocator.alloc(buffer.Buffer, cfg.n_layers);
    defer { for (buf_dw_up) |*b| b.deinit(ctx.device); allocator.free(buf_dw_up); }
    const buf_dw_down = try allocator.alloc(buffer.Buffer, cfg.n_layers);
    defer { for (buf_dw_down) |*b| b.deinit(ctx.device); allocator.free(buf_dw_down); }

    // d_x_in_per_layer: layer i writes into d_x_in[i]; layer i-1 reads
    // it as d_y_in. Layer 0's d_x_in[0] is the input to embedding_bw.
    const buf_d_x_in = try allocator.alloc(buffer.Buffer, cfg.n_layers);
    defer { for (buf_d_x_in) |*b| b.deinit(ctx.device); allocator.free(buf_d_x_in); }

    for (0..cfg.n_layers) |li| {
        const layer = &layers[li];
        const la = &acts.layer_acts[li];
        buf_w_n1[li] = try buffer.Buffer.initStatic(&ctx, f32, layer.w_n1);
        buf_w_q[li] = try buffer.Buffer.initStatic(&ctx, f32, layer.w_q);
        buf_w_k[li] = try buffer.Buffer.initStatic(&ctx, f32, layer.w_k);
        buf_w_v[li] = try buffer.Buffer.initStatic(&ctx, f32, layer.w_v);
        buf_w_o[li] = try buffer.Buffer.initStatic(&ctx, f32, layer.w_o);
        buf_w_n2[li] = try buffer.Buffer.initStatic(&ctx, f32, layer.w_n2);
        buf_w_gate[li] = try buffer.Buffer.initStatic(&ctx, f32, layer.w_gate);
        buf_w_up[li] = try buffer.Buffer.initStatic(&ctx, f32, layer.w_up);
        buf_w_down[li] = try buffer.Buffer.initStatic(&ctx, f32, layer.w_down);

        buf_x_in[li] = try buffer.Buffer.initStatic(&ctx, f32, la.x_in);
        buf_n1[li] = try buffer.Buffer.initStatic(&ctx, f32, la.n1);
        buf_q[li] = try buffer.Buffer.initStatic(&ctx, f32, la.q);
        buf_k[li] = try buffer.Buffer.initStatic(&ctx, f32, la.k);
        buf_v[li] = try buffer.Buffer.initStatic(&ctx, f32, la.v);
        buf_attn[li] = try buffer.Buffer.initStatic(&ctx, f32, la.attn);
        buf_attn_out[li] = try buffer.Buffer.initStatic(&ctx, f32, la.attn_out);
        buf_mid[li] = try buffer.Buffer.initStatic(&ctx, f32, la.mid);
        buf_n2[li] = try buffer.Buffer.initStatic(&ctx, f32, la.n2);
        buf_pre_gate[li] = try buffer.Buffer.initStatic(&ctx, f32, la.pre_gate);
        buf_up[li] = try buffer.Buffer.initStatic(&ctx, f32, la.up);
        buf_gated[li] = try buffer.Buffer.initStatic(&ctx, f32, la.gated);

        buf_dw_n1_partial[li] = try buffer.Buffer.initDeviceOnly(&ctx, n_pos * dim * f32sz);
        buf_dw_q[li] = try buffer.Buffer.initDeviceOnly(&ctx, layer.w_q.len * f32sz);
        buf_dw_k[li] = try buffer.Buffer.initDeviceOnly(&ctx, layer.w_k.len * f32sz);
        buf_dw_v[li] = try buffer.Buffer.initDeviceOnly(&ctx, layer.w_v.len * f32sz);
        buf_dw_o[li] = try buffer.Buffer.initDeviceOnly(&ctx, layer.w_o.len * f32sz);
        buf_dw_n2_partial[li] = try buffer.Buffer.initDeviceOnly(&ctx, n_pos * dim * f32sz);
        buf_dw_gate[li] = try buffer.Buffer.initDeviceOnly(&ctx, layer.w_gate.len * f32sz);
        buf_dw_up[li] = try buffer.Buffer.initDeviceOnly(&ctx, layer.w_up.len * f32sz);
        buf_dw_down[li] = try buffer.Buffer.initDeviceOnly(&ctx, layer.w_down.len * f32sz);

        buf_d_x_in[li] = try buffer.Buffer.initDeviceOnly(&ctx, n_pos * dim * f32sz);
    }

    // ── Stack-level scratch + grad outputs.
    var buf_d_logits = try buffer.Buffer.initDeviceOnly(&ctx, n_pos * vocab * f32sz);
    defer buf_d_logits.deinit(ctx.device);
    var buf_d_final_norm_out = try buffer.Buffer.initDeviceOnly(&ctx, n_pos * dim * f32sz);
    defer buf_d_final_norm_out.deinit(ctx.device);
    var buf_d_last_y = try buffer.Buffer.initDeviceOnly(&ctx, n_pos * dim * f32sz);
    defer buf_d_last_y.deinit(ctx.device);
    var buf_dw_lm_head = try buffer.Buffer.initDeviceOnly(&ctx, w_lm_head.len * f32sz);
    defer buf_dw_lm_head.deinit(ctx.device);
    var buf_dw_final_norm_partial = try buffer.Buffer.initDeviceOnly(&ctx, n_pos * dim * f32sz);
    defer buf_dw_final_norm_partial.deinit(ctx.device);
    var buf_dE_embed = try buffer.Buffer.initDeviceOnly(&ctx, w_embed.len * f32sz);
    defer buf_dE_embed.deinit(ctx.device);

    // ── Per-layer-shared scratch (reused across layer iterations —
    //    barriers between dispatches keep each iteration's reads
    //    after the previous's writes).
    var sc_d_gated = try buffer.Buffer.initDeviceOnly(&ctx, n_pos * ff_dim * f32sz);
    defer sc_d_gated.deinit(ctx.device);
    var sc_d_pre_gate = try buffer.Buffer.initDeviceOnly(&ctx, n_pos * ff_dim * f32sz);
    defer sc_d_pre_gate.deinit(ctx.device);
    var sc_d_up_grad = try buffer.Buffer.initDeviceOnly(&ctx, n_pos * ff_dim * f32sz);
    defer sc_d_up_grad.deinit(ctx.device);
    var sc_d_n2 = try buffer.Buffer.initDeviceOnly(&ctx, n_pos * dim * f32sz);
    defer sc_d_n2.deinit(ctx.device);
    var sc_d_n2_up = try buffer.Buffer.initDeviceOnly(&ctx, n_pos * dim * f32sz);
    defer sc_d_n2_up.deinit(ctx.device);
    var sc_d_mid_norm = try buffer.Buffer.initDeviceOnly(&ctx, n_pos * dim * f32sz);
    defer sc_d_mid_norm.deinit(ctx.device);
    var sc_d_attn_out = try buffer.Buffer.initDeviceOnly(&ctx, n_pos * q_dim * f32sz);
    defer sc_d_attn_out.deinit(ctx.device);
    var sc_d_attn = try buffer.Buffer.initDeviceOnly(&ctx, scores_total * f32sz);
    defer sc_d_attn.deinit(ctx.device);
    var sc_d_scores = try buffer.Buffer.initDeviceOnly(&ctx, scores_total * f32sz);
    defer sc_d_scores.deinit(ctx.device);
    var sc_dQ = try buffer.Buffer.initDeviceOnly(&ctx, n_pos * q_dim * f32sz);
    defer sc_dQ.deinit(ctx.device);
    var sc_dK = try buffer.Buffer.initDeviceOnly(&ctx, n_pos * kv_dim * f32sz);
    defer sc_dK.deinit(ctx.device);
    var sc_dV = try buffer.Buffer.initDeviceOnly(&ctx, n_pos * kv_dim * f32sz);
    defer sc_dV.deinit(ctx.device);
    var sc_d_n1 = try buffer.Buffer.initDeviceOnly(&ctx, n_pos * dim * f32sz);
    defer sc_d_n1.deinit(ctx.device);
    var sc_d_n1_k = try buffer.Buffer.initDeviceOnly(&ctx, n_pos * dim * f32sz);
    defer sc_d_n1_k.deinit(ctx.device);
    var sc_d_n1_v = try buffer.Buffer.initDeviceOnly(&ctx, n_pos * dim * f32sz);
    defer sc_d_n1_v.deinit(ctx.device);

    // ── Pushes.
    const push_rms = runtime.RmsnormPush{ .dim = @intCast(dim), .eps = cfg.base.rms_eps, .gemma_quirk = 0 };
    const push_add_dim = runtime.AddInPlacePush{ .n = @intCast(n_pos * dim) };
    const push_swiglu = runtime.SwigluPush{ .n = @intCast(n_pos * ff_dim) };
    const push_softmax = runtime.SoftmaxPush{ .dim = @intCast(n_pos), .stride = @intCast(n_pos) };
    const push_ce = runtime.SoftmaxCeLossGradPush{ .dim_out = @intCast(vocab), .n_samples = @intCast(n_pos) };
    const push_lm_head = runtime.LinearBatchedPush{ .M = @intCast(n_pos), .N = @intCast(vocab), .K = @intCast(dim) };
    const push_lin_down = runtime.LinearBatchedPush{ .M = @intCast(n_pos), .N = @intCast(dim), .K = @intCast(ff_dim) };
    const push_lin_gate = runtime.LinearBatchedPush{ .M = @intCast(n_pos), .N = @intCast(ff_dim), .K = @intCast(dim) };
    const push_lin_up = runtime.LinearBatchedPush{ .M = @intCast(n_pos), .N = @intCast(ff_dim), .K = @intCast(dim) };
    const push_lin_o = runtime.LinearBatchedPush{ .M = @intCast(n_pos), .N = @intCast(dim), .K = @intCast(q_dim) };
    const push_lin_q = runtime.LinearBatchedPush{ .M = @intCast(n_pos), .N = @intCast(q_dim), .K = @intCast(dim) };
    const push_lin_k = runtime.LinearBatchedPush{ .M = @intCast(n_pos), .N = @intCast(kv_dim), .K = @intCast(dim) };
    const push_lin_v = runtime.LinearBatchedPush{ .M = @intCast(n_pos), .N = @intCast(kv_dim), .K = @intCast(dim) };
    const push_dattn = runtime.AttnBackwardDattnPush{
        .n_q = @intCast(n_pos),
        .n_heads = @intCast(n_heads),
        .heads_per_kv = heads_per_kv,
        .head_dim = @intCast(head_dim),
        .n_kv = @intCast(n_pos),
        .kv_stride = @intCast(kv_dim),
        .attn_stride = @intCast(n_pos),
    };
    const push_dv = runtime.AttnBackwardDvPush{
        .n_q = @intCast(n_pos),
        .n_heads = @intCast(n_heads),
        .heads_per_kv = heads_per_kv,
        .n_kv_heads = @intCast(n_kv_heads),
        .head_dim = @intCast(head_dim),
        .n_kv = @intCast(n_pos),
        .attn_stride = @intCast(n_pos),
    };
    const push_dq = runtime.AttnBackwardDqPush{
        .n_q = @intCast(n_pos),
        .n_heads = @intCast(n_heads),
        .heads_per_kv = heads_per_kv,
        .head_dim = @intCast(head_dim),
        .n_kv = @intCast(n_pos),
        .kv_stride = @intCast(kv_dim),
        .scores_stride = @intCast(n_pos),
        .inv_sqrt_dim = inv_sqrt_d,
    };
    const push_dk = runtime.AttnBackwardDkPush{
        .n_q = @intCast(n_pos),
        .n_heads = @intCast(n_heads),
        .heads_per_kv = heads_per_kv,
        .n_kv_heads = @intCast(n_kv_heads),
        .head_dim = @intCast(head_dim),
        .n_kv = @intCast(n_pos),
        .scores_stride = @intCast(n_pos),
        .inv_sqrt_dim = inv_sqrt_d,
    };
    const push_embed = runtime.EmbeddingBackwardPush{
        .dim = @intCast(dim),
        .n_pos = @intCast(n_pos),
        .vocab_size = @intCast(vocab),
    };

    // ── Recorder. Total dispatches: 1 (CE loss-grad) + 2 (lm_head dx/dW)
    //    + 1 (final_norm rmsnorm bw) + 24·N (per-layer chain) + 1
    //    (embedding_bw) = 5 + 24·N. For N=2: 53.
    var rec = try gpu_recorder.Recorder.init(&ctx, 128, 512);
    defer rec.deinit();
    try rec.begin();

    const lwg: u32 = 16;
    const groupsCeil = struct {
        fn f(n: u32) u32 {
            return (n + lwg - 1) / lwg;
        }
    }.f;

    // ── 1. CE loss-grad: logits + target_one_hot → d_logits.
    //    v2 shader: one workgroup per sample (n_pos workgroups).
    try rec.dispatch(&k_ce_loss_grad, &.{ &buf_logits, &buf_target_oh, &buf_d_logits }, &push_ce, @intCast(n_pos), 1, 1);

    // ── 2. lm_head: dx into d_final_norm_out, dW into dw_lm_head.
    try rec.dispatch(&k_lin_dx, &.{ &buf_d_logits, &buf_w_lm_head, &buf_d_final_norm_out }, &push_lm_head, groupsCeil(push_lm_head.M), groupsCeil(push_lm_head.K), 1);
    try rec.dispatch(&k_lin_dw, &.{ &buf_d_logits, &buf_final_norm_out, &buf_dw_lm_head }, &push_lm_head, groupsCeil(push_lm_head.N), groupsCeil(push_lm_head.K), 1);

    // ── 3. final_norm rmsnorm backward → d_last_y + dw_final_norm_partial.
    const last_idx = cfg.n_layers - 1;
    // Last layer's `y` is held in its `mid` buffer? No — `mid` is the
    // post-attention residual output; `y` is the post-FF residual
    // output. The CPU oracle captures y in `acts.layer_acts[last].y`
    // but on the GPU we didn't upload `y` separately because layer
    // i+1's x_in already holds it. For the last layer we need to upload
    // y explicitly. We do so here as a one-off — small additional buffer.
    var buf_last_y = try buffer.Buffer.initStatic(&ctx, f32, acts.layer_acts[last_idx].y);
    defer buf_last_y.deinit(ctx.device);

    try rec.dispatch(&k_rms_bw, &.{ &buf_d_final_norm_out, &buf_last_y, &buf_w_final_norm, &buf_d_last_y, &buf_dw_final_norm_partial }, &push_rms, @intCast(n_pos), 1, 1);

    // ── 4. Per-layer backward, last → first. d_y_in for layer N-1
    //    is d_last_y; for layer i < N-1 it's d_x_in[i+1].
    var li = cfg.n_layers;
    while (li > 0) {
        li -= 1;
        const d_y_in: *const buffer.Buffer = if (li == cfg.n_layers - 1) &buf_d_last_y else &buf_d_x_in[li + 1];
        try recordStackLayerBackward(
            &rec,
            .{
                .lin_dx = &k_lin_dx,
                .lin_dw = &k_lin_dw,
                .swiglu_bw = &k_swiglu_bw,
                .rms_bw = &k_rms_bw,
                .attn_dattn = &k_attn_dattn,
                .attn_dv = &k_attn_dv,
                .attn_dq = &k_attn_dq,
                .attn_dk = &k_attn_dk,
                .softmax_bw = &k_softmax_bw,
                .add = &k_add,
            },
            .{
                .x_in = &buf_x_in[li],
                .n1 = &buf_n1[li],
                .q = &buf_q[li],
                .k = &buf_k[li],
                .v = &buf_v[li],
                .attn = &buf_attn[li],
                .attn_out = &buf_attn_out[li],
                .mid = &buf_mid[li],
                .n2 = &buf_n2[li],
                .pre_gate = &buf_pre_gate[li],
                .up = &buf_up[li],
                .gated = &buf_gated[li],
                .w_n1 = &buf_w_n1[li],
                .w_q = &buf_w_q[li],
                .w_k = &buf_w_k[li],
                .w_v = &buf_w_v[li],
                .w_o = &buf_w_o[li],
                .w_n2 = &buf_w_n2[li],
                .w_gate = &buf_w_gate[li],
                .w_up = &buf_w_up[li],
                .w_down = &buf_w_down[li],
                .dw_n1_partial = &buf_dw_n1_partial[li],
                .dw_q = &buf_dw_q[li],
                .dw_k = &buf_dw_k[li],
                .dw_v = &buf_dw_v[li],
                .dw_o = &buf_dw_o[li],
                .dw_n2_partial = &buf_dw_n2_partial[li],
                .dw_gate = &buf_dw_gate[li],
                .dw_up = &buf_dw_up[li],
                .dw_down = &buf_dw_down[li],
            },
            .{
                .d_gated = &sc_d_gated,
                .d_pre_gate = &sc_d_pre_gate,
                .d_up = &sc_d_up_grad,
                .d_n2 = &sc_d_n2,
                .d_n2_up = &sc_d_n2_up,
                .d_mid_norm = &sc_d_mid_norm,
                .d_attn_out = &sc_d_attn_out,
                .d_attn = &sc_d_attn,
                .d_scores = &sc_d_scores,
                .dQ = &sc_dQ,
                .dK = &sc_dK,
                .dV = &sc_dV,
                .d_n1 = &sc_d_n1,
                .d_n1_k = &sc_d_n1_k,
                .d_n1_v = &sc_d_n1_v,
            },
            .{
                .lin_down = &push_lin_down,
                .lin_gate = &push_lin_gate,
                .lin_up = &push_lin_up,
                .lin_o = &push_lin_o,
                .lin_q = &push_lin_q,
                .lin_k = &push_lin_k,
                .lin_v = &push_lin_v,
                .swiglu = &push_swiglu,
                .rms = &push_rms,
                .add_dim = &push_add_dim,
                .softmax = &push_softmax,
                .dattn = &push_dattn,
                .dv = &push_dv,
                .dq = &push_dq,
                .dk = &push_dk,
            },
            .{ .n_pos = @intCast(n_pos), .n_heads = @intCast(n_heads), .n_kv_heads = @intCast(n_kv_heads), .head_dim = @intCast(head_dim) },
            d_y_in,
            &buf_d_x_in[li],
        );
    }

    // ── 5. embedding_backward: dE_embed[token_id, :] += d_x_in[0][p, :].
    //    initDeviceOnly already zero-filled the dE buffer.
    try rec.dispatch(&k_embed_bw, &.{ &buf_d_x_in[0], &buf_token_ids, &buf_dE_embed }, &push_embed, @intCast(vocab), 1, 1);

    try rec.endAndSubmit();

    // ── Read back grad buffers + reduce per-row partials.
    const gpu_dE_embed = try allocator.alloc(f32, w_embed.len);
    defer allocator.free(gpu_dE_embed);
    const gpu_dw_final_norm_partial = try allocator.alloc(f32, n_pos * dim);
    defer allocator.free(gpu_dw_final_norm_partial);
    const gpu_dw_lm_head = try allocator.alloc(f32, w_lm_head.len);
    defer allocator.free(gpu_dw_lm_head);
    try buf_dE_embed.readBack(&ctx, f32, gpu_dE_embed);
    try buf_dw_final_norm_partial.readBack(&ctx, f32, gpu_dw_final_norm_partial);
    try buf_dw_lm_head.readBack(&ctx, f32, gpu_dw_lm_head);

    const gpu_dw_final_norm = try allocator.alloc(f32, dim);
    defer allocator.free(gpu_dw_final_norm);
    @memset(gpu_dw_final_norm, 0);
    for (0..n_pos) |row| {
        const off = row * dim;
        for (0..dim) |i| gpu_dw_final_norm[i] += gpu_dw_final_norm_partial[off + i];
    }

    // Per-layer reads + reductions.
    const gpu_layer_grads = try allocator.alloc([9][]f32, cfg.n_layers);
    defer {
        for (gpu_layer_grads) |slots| for (slots) |s| allocator.free(s);
        allocator.free(gpu_layer_grads);
    }
    for (0..cfg.n_layers) |i| {
        const layer = &layers[i];
        const dw_n1_partial = try allocator.alloc(f32, n_pos * dim);
        defer allocator.free(dw_n1_partial);
        const dw_n2_partial = try allocator.alloc(f32, n_pos * dim);
        defer allocator.free(dw_n2_partial);
        try buf_dw_n1_partial[i].readBack(&ctx, f32, dw_n1_partial);
        try buf_dw_n2_partial[i].readBack(&ctx, f32, dw_n2_partial);
        const dw_n1 = try allocator.alloc(f32, dim);
        const dw_q_g = try allocator.alloc(f32, layer.w_q.len);
        const dw_k_g = try allocator.alloc(f32, layer.w_k.len);
        const dw_v_g = try allocator.alloc(f32, layer.w_v.len);
        const dw_o_g = try allocator.alloc(f32, layer.w_o.len);
        const dw_n2 = try allocator.alloc(f32, dim);
        const dw_gate_g = try allocator.alloc(f32, layer.w_gate.len);
        const dw_up_g = try allocator.alloc(f32, layer.w_up.len);
        const dw_down_g = try allocator.alloc(f32, layer.w_down.len);
        @memset(dw_n1, 0);
        @memset(dw_n2, 0);
        for (0..n_pos) |row| {
            const off = row * dim;
            for (0..dim) |idx| {
                dw_n1[idx] += dw_n1_partial[off + idx];
                dw_n2[idx] += dw_n2_partial[off + idx];
            }
        }
        try buf_dw_q[i].readBack(&ctx, f32, dw_q_g);
        try buf_dw_k[i].readBack(&ctx, f32, dw_k_g);
        try buf_dw_v[i].readBack(&ctx, f32, dw_v_g);
        try buf_dw_o[i].readBack(&ctx, f32, dw_o_g);
        try buf_dw_gate[i].readBack(&ctx, f32, dw_gate_g);
        try buf_dw_up[i].readBack(&ctx, f32, dw_up_g);
        try buf_dw_down[i].readBack(&ctx, f32, dw_down_g);
        gpu_layer_grads[i] = .{ dw_n1, dw_q_g, dw_k_g, dw_v_g, dw_o_g, dw_n2, dw_gate_g, dw_up_g, dw_down_g };
    }

    // ── Compare. Tolerance: per-slice rel-err with 1e-5 floor.
    var worst_rel: f32 = 0;
    var worst_label: []const u8 = "";

    const compareSlice = struct {
        fn f(label: []const u8, c_s: []const f32, g_s: []const f32, worst: *f32, worst_lbl: *[]const u8) !void {
            var max_abs_cpu: f32 = 0;
            for (c_s) |v| max_abs_cpu = @max(max_abs_cpu, @abs(v));
            const tol: f32 = @max(1e-5, max_abs_cpu * 1e-3);
            var max_abs_diff: f32 = 0;
            for (c_s, g_s) |a, b| {
                const d = @abs(a - b);
                if (d > max_abs_diff) max_abs_diff = d;
            }
            const rel = if (max_abs_cpu > 0) max_abs_diff / max_abs_cpu else max_abs_diff;
            if (rel > worst.*) {
                worst.* = rel;
                worst_lbl.* = label;
            }
            if (max_abs_diff > tol) {
                std.debug.print(
                    "stack backward parity FAIL: {s} max |Δ|={e:.3} (tol={e:.3}, max_cpu={e:.3})\n",
                    .{ label, max_abs_diff, tol, max_abs_cpu },
                );
                return error.ParityFailed;
            }
        }
    }.f;

    try compareSlice("dE_embed", grads_cpu.dE_embed, gpu_dE_embed, &worst_rel, &worst_label);
    try compareSlice("dw_final_norm", grads_cpu.dw_final_norm, gpu_dw_final_norm, &worst_rel, &worst_label);
    try compareSlice("dw_lm_head", grads_cpu.dw_lm_head, gpu_dw_lm_head, &worst_rel, &worst_label);

    // Heap-allocate each label so the captured worst_label slice
    // doesn't get clobbered by later iterations (the comparator stores
    // the slice by reference for the final pass-line).
    var owned_labels = std.ArrayList([]u8).init(allocator);
    defer {
        for (owned_labels.items) |s| allocator.free(s);
        owned_labels.deinit();
    }
    const layer_field_names = [_][]const u8{ "dw_n1", "dw_q", "dw_k", "dw_v", "dw_o", "dw_n2", "dw_gate", "dw_up", "dw_down" };
    for (0..cfg.n_layers) |i| {
        const cpu_slices = [_][]const f32{
            grads_cpu.layer_grads[i].dw_n1,
            grads_cpu.layer_grads[i].dw_q,
            grads_cpu.layer_grads[i].dw_k,
            grads_cpu.layer_grads[i].dw_v,
            grads_cpu.layer_grads[i].dw_o,
            grads_cpu.layer_grads[i].dw_n2,
            grads_cpu.layer_grads[i].dw_gate,
            grads_cpu.layer_grads[i].dw_up,
            grads_cpu.layer_grads[i].dw_down,
        };
        for (cpu_slices, gpu_layer_grads[i], layer_field_names) |c_s, g_s, fname| {
            const lbl = try std.fmt.allocPrint(allocator, "L{d}.{s}", .{ i, fname });
            try owned_labels.append(lbl);
            try compareSlice(lbl, c_s, g_s, &worst_rel, &worst_label);
        }
    }

    std.debug.print(
        "PASS GPU decoder stack backward parity (n_layers={d} dim={d} vocab={d} n_pos={d}; {d} dispatches, worst rel-err {e:.2} on {s})\n",
        .{ cfg.n_layers, dim, vocab, n_pos, 5 + 27 * cfg.n_layers, worst_rel, worst_label },
    );
}

// ── chunk 8c-α-3: full-GPU stack training, 100-step Adam loop ──────
//
// Self-sustaining transformer trainer: every operation per step
// (embed_lookup → N decoder layers → final RMSNorm → lm_head →
// softmax-CE loss-grad → backward → Adam) lives on the GPU. CPU only
// builds the initial weights + token IDs + one-hot target, then
// reads back final logits to compute the closing loss.
//
// Per-step dispatch count: 1 embed + 14·N forward + 2 (final_norm +
// lm_head) + 1 loss-grad + 5+24·N backward + 8N+3 Adam =
// 11 + 46·N. For N=2: 103 dispatches/step.

const StackFwKernels = struct {
    rms: *const pipeline.Kernel,
    matmul: *const pipeline.Kernel,
    attn_scores: *const pipeline.Kernel,
    attn_output: *const pipeline.Kernel,
    softmax: *const pipeline.Kernel,
    swiglu_fwd: *const pipeline.Kernel,
    vec_add: *const pipeline.Kernel,
};

const StackFwLayerBufs = struct {
    // Read-only.
    x_in: *const buffer.Buffer,
    w_n1: *const buffer.Buffer,
    w_q: *const buffer.Buffer,
    w_k: *const buffer.Buffer,
    w_v: *const buffer.Buffer,
    w_o: *const buffer.Buffer,
    w_n2: *const buffer.Buffer,
    w_gate: *const buffer.Buffer,
    w_up: *const buffer.Buffer,
    w_down: *const buffer.Buffer,
    // Saved-activation outputs (consumed by backward).
    n1: *const buffer.Buffer,
    q: *const buffer.Buffer,
    k: *const buffer.Buffer,
    v: *const buffer.Buffer,
    attn: *const buffer.Buffer,
    attn_out: *const buffer.Buffer,
    mid: *const buffer.Buffer,
    n2: *const buffer.Buffer,
    pre_gate: *const buffer.Buffer,
    up: *const buffer.Buffer,
    gated: *const buffer.Buffer,
    y: *const buffer.Buffer,
};

const StackFwScratch = struct {
    scores: *const buffer.Buffer,
    o: *const buffer.Buffer,
    ff_out: *const buffer.Buffer,
};

fn recordStackLayerForward(
    rec: *gpu_recorder.Recorder,
    kernels: StackFwKernels,
    bufs: StackFwLayerBufs,
    scratch: StackFwScratch,
    pushes: struct {
        rms: *const runtime.RmsnormPush,
        mm_q: *const runtime.MatmulPush,
        mm_k: *const runtime.MatmulPush,
        mm_v: *const runtime.MatmulPush,
        mm_o: *const runtime.MatmulPush,
        mm_gate: *const runtime.MatmulPush,
        mm_up: *const runtime.MatmulPush,
        mm_down: *const runtime.MatmulPush,
        attn_scores: *const runtime.AttnScoresTrainPush,
        attn_output: *const runtime.AttnOutputTrainPush,
        softmax: *const runtime.SoftmaxPush,
        swiglu: *const runtime.SwigluPush,
        add_dim: *const runtime.AddInPlacePush,
    },
    shape: struct {
        n_pos: u32,
        n_heads: u32,
        n_kv_heads: u32,
        head_dim: u32,
        ff_dim: u32,
        dim: u32,
    },
) !void {
    const groupsLin: u32 = 256;
    const addGroups: u32 = (pushes.add_dim.n + groupsLin - 1) / groupsLin;
    const swigluGroups: u32 = (pushes.swiglu.n + groupsLin - 1) / groupsLin;

    // 1. RMSNorm n1.
    try rec.dispatch(kernels.rms, &.{ bufs.x_in, bufs.w_n1, bufs.n1 }, pushes.rms, shape.n_pos, 1, 1);
    // 2-4. Q/K/V projections.
    try rec.dispatch(kernels.matmul, &.{ bufs.n1, bufs.w_q, bufs.q }, pushes.mm_q, shape.n_pos * pushes.mm_q.n, 1, 1);
    try rec.dispatch(kernels.matmul, &.{ bufs.n1, bufs.w_k, bufs.k }, pushes.mm_k, shape.n_pos * pushes.mm_k.n, 1, 1);
    try rec.dispatch(kernels.matmul, &.{ bufs.n1, bufs.w_v, bufs.v }, pushes.mm_v, shape.n_pos * pushes.mm_v.n, 1, 1);
    // 5. attention scores (causal-mask via -inf for keys beyond q).
    try rec.dispatch(kernels.attn_scores, &.{ bufs.q, bufs.k, scratch.scores }, pushes.attn_scores, shape.n_pos * shape.n_heads * shape.n_pos, 1, 1);
    // 6. softmax → attn.
    try rec.dispatch(kernels.softmax, &.{ scratch.scores, bufs.attn }, pushes.softmax, shape.n_pos * shape.n_heads, 1, 1);
    // 7. attention output.
    try rec.dispatch(kernels.attn_output, &.{ bufs.attn, bufs.v, bufs.attn_out }, pushes.attn_output, shape.n_pos * shape.n_heads * shape.head_dim, 1, 1);
    // 8. O projection.
    try rec.dispatch(kernels.matmul, &.{ bufs.attn_out, bufs.w_o, scratch.o }, pushes.mm_o, shape.n_pos * pushes.mm_o.n, 1, 1);
    // 9. mid = x_in + o (residual).
    try rec.dispatch(kernels.vec_add, &.{ bufs.x_in, scratch.o, bufs.mid }, pushes.add_dim, addGroups, 1, 1);
    // 10. RMSNorm n2.
    try rec.dispatch(kernels.rms, &.{ bufs.mid, bufs.w_n2, bufs.n2 }, pushes.rms, shape.n_pos, 1, 1);
    // 11. W_gate matmul.
    try rec.dispatch(kernels.matmul, &.{ bufs.n2, bufs.w_gate, bufs.pre_gate }, pushes.mm_gate, shape.n_pos * pushes.mm_gate.n, 1, 1);
    // 12. W_up matmul.
    try rec.dispatch(kernels.matmul, &.{ bufs.n2, bufs.w_up, bufs.up }, pushes.mm_up, shape.n_pos * pushes.mm_up.n, 1, 1);
    // 13. SwiGLU fwd: gated = silu(pre_gate) · up.
    try rec.dispatch(kernels.swiglu_fwd, &.{ bufs.pre_gate, bufs.up, bufs.gated }, pushes.swiglu, swigluGroups, 1, 1);
    // 14. W_down matmul → ff_out.
    try rec.dispatch(kernels.matmul, &.{ bufs.gated, bufs.w_down, scratch.ff_out }, pushes.mm_down, shape.n_pos * pushes.mm_down.n, 1, 1);
    // 15. y = mid + ff_out (residual).
    try rec.dispatch(kernels.vec_add, &.{ bufs.mid, scratch.ff_out, bufs.y }, pushes.add_dim, addGroups, 1, 1);
}

pub fn runDecoderStackTrainGpuSmoke(allocator: std.mem.Allocator) !void {
    const cfg = cpu_train_decoder.StackConfig{
        .base = .{
            .dim = 16,
            .n_heads = 2,
            .n_kv_heads = 2,
            .head_dim = 8,
            .ff_dim = 32,
            .n_pos = 4,
            .rms_eps = 1e-5,
            .causal = true,
            .rotary_dim = 8, // full RoPE (rotary_dim = head_dim)
            .qk_norm = true, // per-head Q/K-norm (Qwen3 architectural detail)
        },
        .n_layers = 2,
        .vocab_size = 8,
    };
    const dim = cfg.base.dim;
    const n_pos = cfg.base.n_pos;
    const n_heads = cfg.base.n_heads;
    const n_kv_heads = cfg.base.n_kv_heads;
    const head_dim = cfg.base.head_dim;
    const ff_dim = cfg.base.ff_dim;
    const vocab = cfg.vocab_size;
    const q_dim = n_heads * head_dim;
    const kv_dim = n_kv_heads * head_dim;

    // Same RNG sequence as the 8c-α-1 / α-2 smokes so initial weights
    // + token_ids + target_ids are bit-equal — initial loss matches.
    var prng = std.Random.DefaultPrng.init(0xC0_DE_AC_01);
    const rng = prng.random();
    const initScale: f32 = 0.1;

    // ── CPU-side weight + token init.
    const w_embed = try allocator.alloc(f32, vocab * dim);
    defer allocator.free(w_embed);
    for (w_embed) |*v| v.* = (rng.float(f32) * 2.0 - 1.0) * initScale;
    const w_final_norm = try allocator.alloc(f32, dim);
    defer allocator.free(w_final_norm);
    for (w_final_norm) |*v| v.* = 1.0;
    const w_lm_head = try allocator.alloc(f32, vocab * dim);
    defer allocator.free(w_lm_head);
    for (w_lm_head) |*v| v.* = (rng.float(f32) * 2.0 - 1.0) * initScale;

    const layers = try allocator.alloc(cpu_train_decoder.Layer, cfg.n_layers);
    defer allocator.free(layers);
    const layer_weight_buffers = try allocator.alloc([9][]f32, cfg.n_layers);
    defer allocator.free(layer_weight_buffers);
    defer for (layer_weight_buffers) |slots| for (slots) |s| allocator.free(s);

    const layer_qk_buffers = try allocator.alloc([2][]f32, cfg.n_layers);
    defer allocator.free(layer_qk_buffers);
    defer for (layer_qk_buffers) |slots| for (slots) |s| allocator.free(s);

    for (layers, layer_weight_buffers, layer_qk_buffers) |*layer, *slots, *qk_slots| {
        const w_n1 = try allocator.alloc(f32, dim);
        const w_q = try allocator.alloc(f32, q_dim * dim);
        const w_k = try allocator.alloc(f32, kv_dim * dim);
        const w_v = try allocator.alloc(f32, kv_dim * dim);
        const w_o = try allocator.alloc(f32, dim * q_dim);
        const w_n2 = try allocator.alloc(f32, dim);
        const w_gate = try allocator.alloc(f32, ff_dim * dim);
        const w_up = try allocator.alloc(f32, ff_dim * dim);
        const w_down = try allocator.alloc(f32, dim * ff_dim);
        const w_q_norm = try allocator.alloc(f32, head_dim);
        const w_k_norm = try allocator.alloc(f32, head_dim);
        for (w_n1) |*v| v.* = 1.0;
        for (w_n2) |*v| v.* = 1.0;
        for (w_q_norm) |*v| v.* = 1.0;
        for (w_k_norm) |*v| v.* = 1.0;
        for (w_q) |*v| v.* = (rng.float(f32) * 2.0 - 1.0) * initScale;
        for (w_k) |*v| v.* = (rng.float(f32) * 2.0 - 1.0) * initScale;
        for (w_v) |*v| v.* = (rng.float(f32) * 2.0 - 1.0) * initScale;
        for (w_o) |*v| v.* = (rng.float(f32) * 2.0 - 1.0) * initScale;
        for (w_gate) |*v| v.* = (rng.float(f32) * 2.0 - 1.0) * initScale;
        for (w_up) |*v| v.* = (rng.float(f32) * 2.0 - 1.0) * initScale;
        for (w_down) |*v| v.* = (rng.float(f32) * 2.0 - 1.0) * initScale;
        slots.* = .{ w_n1, w_q, w_k, w_v, w_o, w_n2, w_gate, w_up, w_down };
        qk_slots.* = .{ w_q_norm, w_k_norm };
        layer.* = .{
            .cfg = cfg.base,
            .w_n1 = w_n1,
            .w_q = w_q,
            .w_k = w_k,
            .w_v = w_v,
            .w_o = w_o,
            .w_n2 = w_n2,
            .w_gate = w_gate,
            .w_up = w_up,
            .w_down = w_down,
            .w_q_norm = w_q_norm,
            .w_k_norm = w_k_norm,
        };
    }

    var stack = cpu_train_decoder.Stack{
        .cfg = cfg,
        .embed = w_embed,
        .layers = layers,
        .final_norm = w_final_norm,
        .lm_head = w_lm_head,
    };

    // ── Compute initial CPU loss for the convergence assertion.
    var acts_cpu = try cpu_train_decoder.allocStackActs(allocator, cfg);
    defer cpu_train_decoder.freeStackActs(allocator, &acts_cpu);

    const token_ids = try allocator.alloc(u32, n_pos);
    defer allocator.free(token_ids);
    for (token_ids) |*tid| tid.* = rng.intRangeLessThan(u32, 0, @intCast(vocab));
    const target_ids = try allocator.alloc(u32, n_pos);
    defer allocator.free(target_ids);
    for (target_ids) |*tid| tid.* = rng.intRangeLessThan(u32, 0, @intCast(vocab));

    acts_cpu.token_ids = token_ids;
    cpu_train_decoder.stackForward(&stack, &acts_cpu);
    const initial_loss = cpu_train_decoder.softmaxCeLoss(acts_cpu.logits, target_ids, n_pos, vocab);
    // CCE consumes target ids directly — no [n_pos × vocab] one-hot needed.

    // ── GPU bring-up via Runner.
    var ctx = try vk.Context.init(allocator);
    defer ctx.deinit();

    // Build LayerWeights slice borrowed by the Runner. Lifetimes:
    // the Runner uploads into its own DEVICE_LOCAL buffers in init(),
    // so these borrows only need to outlive the init() call.
    const layer_weights = try allocator.alloc(train_transformer.LayerWeights, cfg.n_layers);
    defer allocator.free(layer_weights);
    for (layers, layer_weights) |*layer, *lw| {
        lw.* = .{
            .w_n1 = layer.w_n1,
            .w_q = layer.w_q,
            .w_k = layer.w_k,
            .w_v = layer.w_v,
            .w_o = layer.w_o,
            .w_n2 = layer.w_n2,
            .w_gate = layer.w_gate,
            .w_up = layer.w_up,
            .w_down = layer.w_down,
            .w_q_norm = layer.w_q_norm,
            .w_k_norm = layer.w_k_norm,
        };
    }

    var runner = try train_transformer.Runner.init(
        allocator,
        &ctx,
        .{
            .dim = @intCast(dim),
            .n_heads = @intCast(n_heads),
            .n_kv_heads = @intCast(n_kv_heads),
            .head_dim = @intCast(head_dim),
            .ff_dim = @intCast(ff_dim),
            .n_pos = @intCast(n_pos),
            .n_layers = @intCast(cfg.n_layers),
            .vocab_size = @intCast(vocab),
            .rms_eps = cfg.base.rms_eps,
            .causal = cfg.base.causal,
            .rotary_dim = @intCast(cfg.base.rotary_dim),
            .rope_theta = cfg.base.rope_theta,
            .qk_norm = cfg.base.qk_norm,
            .lr = 1e-2,
        },
        .{
            .embed = w_embed,
            .final_norm = w_final_norm,
            .lm_head = w_lm_head,
            .layers = layer_weights,
        },
    );
    defer runner.deinit();

    const n_steps: u32 = 200;
    var step_t: u32 = 0;
    while (step_t < n_steps) : (step_t += 1) {
        try runner.step(token_ids, target_ids);
    }

    const logits_final = try allocator.alloc(f32, n_pos * vocab);
    defer allocator.free(logits_final);
    try runner.forwardLogits(token_ids, logits_final);

    const final_loss = cpu_train_decoder.softmaxCeLoss(logits_final, target_ids, n_pos, vocab);
    const ratio = final_loss / initial_loss;
    if (ratio > 1e-2) {
        std.debug.print(
            "GPU stack fine-tune: initial_loss={d:.6} final_loss={d:.6} ratio={d:.4}\n",
            .{ initial_loss, final_loss, ratio },
        );
        return error.LossDidNotDecrease;
    }

    // Per-layer dispatches: 51 baseline (β-3a-2 SwiGLU stack) + 4 when
    // RoPE is enabled (2 fwd + 2 bw across Q + K) + 6 when qk_norm is
    // enabled (2 fwd rmsnorm + 2 bw rmsnorm + 2 Adam updates).
    var per_layer_dispatches: usize = 51;
    if (cfg.base.rotary_dim > 0) per_layer_dispatches += 4;
    if (cfg.base.qk_norm) per_layer_dispatches += 6;
    std.debug.print(
        "PASS GPU decoder stack fine-tune via Runner (n_layers={d} dim={d} vocab={d} n_pos={d}; CE loss {d:.6} → {d:.6} ({e:.2}× drop) over {d} Adam steps, {d} dispatches/step)\n",
        .{ cfg.n_layers, dim, vocab, n_pos, initial_loss, final_loss, 1.0 / ratio, n_steps, 11 + per_layer_dispatches * cfg.n_layers },
    );
}

// ── chunk 8c-β-2: Real-shape buffer sizing (synthetic weights) ──────
//
// Instantiates `train_transformer.Runner` at Qwen3-0.6B-class dims with
// random fp32 weights and verifies the runtime envelope holds: init
// succeeds, one Adam step doesn't OOM, loss decreases over ~50 steps,
// and reports ms/step so β-3 can compare once real weights enter.
//
// Architecture is the toy-decoder shape the trainer ships today (no
// SwiGLU, no RoPE, no q/k-norm). Architectural fidelity to real Qwen3
// belongs to a separate β-3a chunk that adds the missing primitives
// (each gets the chunk-1..7 treatment: CPU oracle → GPU shader → smoke
// → fold into Runner). What β-2 establishes is that the *buffer
// sizing* + *dispatch sequencing* the Runner generates scales to
// realistic memory pressure (~9 GB) and depth (28 layers).
//
// Pass criteria:
//   - Runner.init returns without OOM / descriptor exhaustion.
//   - All 50 steps run.
//   - Final CE < initial CE (any ratio < 1; this is a "does it still
//     train" envelope check, not a convergence assertion).
//   - All loss values finite.
//
// All weights heap-allocated via the smoke's allocator; per-layer
// weights live in an arena so the slice lifetime ends cleanly when
// the smoke returns.

pub fn runDecoderStackTrainGpuRealShapeSmoke(allocator: std.mem.Allocator) !void {
    const dim: u32 = 1024;
    const n_layers: u32 = 28;
    const n_heads: u32 = 16;
    const n_kv_heads: u32 = 8;
    const head_dim: u32 = 64;
    const ff_dim: u32 = 3072;
    const vocab: u32 = 151_936;
    const n_pos: u32 = 64;
    const q_dim: u32 = n_heads * head_dim;
    const kv_dim: u32 = n_kv_heads * head_dim;

    var prng = std.Random.DefaultPrng.init(0xBE_BA_2A_01);
    const rng = prng.random();
    // Conservative init scale — wide layers + Adam can blow up at 0.1
    // even on a single-batch overfit. 0.02 keeps initial activations
    // and logits bounded.
    const init_scale: f32 = 0.02;

    const fillRandom = struct {
        fn run(r: std.Random, buf: []f32, scale: f32) void {
            for (buf) |*v| v.* = (r.float(f32) * 2.0 - 1.0) * scale;
        }
    }.run;

    // Stack-level synthetic weights.
    const w_embed = try allocator.alloc(f32, @as(usize, vocab) * dim);
    defer allocator.free(w_embed);
    fillRandom(rng, w_embed, init_scale);
    const w_final_norm = try allocator.alloc(f32, dim);
    defer allocator.free(w_final_norm);
    @memset(w_final_norm, 1.0);
    const w_lm_head = try allocator.alloc(f32, @as(usize, vocab) * dim);
    defer allocator.free(w_lm_head);
    fillRandom(rng, w_lm_head, init_scale);

    // Per-layer weights live in an arena; freed wholesale at smoke end.
    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();
    const aalloc = arena.allocator();
    const layer_weights = try allocator.alloc(train_transformer.LayerWeights, n_layers);
    defer allocator.free(layer_weights);
    for (layer_weights) |*lw| {
        const w_n1 = try aalloc.alloc(f32, dim);
        const w_q = try aalloc.alloc(f32, @as(usize, q_dim) * dim);
        const w_k = try aalloc.alloc(f32, @as(usize, kv_dim) * dim);
        const w_v = try aalloc.alloc(f32, @as(usize, kv_dim) * dim);
        const w_o = try aalloc.alloc(f32, @as(usize, dim) * q_dim);
        const w_n2 = try aalloc.alloc(f32, dim);
        const w_gate = try aalloc.alloc(f32, @as(usize, ff_dim) * dim);
        const w_up = try aalloc.alloc(f32, @as(usize, ff_dim) * dim);
        const w_down = try aalloc.alloc(f32, @as(usize, dim) * ff_dim);
        const w_q_norm = try aalloc.alloc(f32, head_dim);
        const w_k_norm = try aalloc.alloc(f32, head_dim);
        @memset(w_n1, 1.0);
        @memset(w_n2, 1.0);
        @memset(w_q_norm, 1.0);
        @memset(w_k_norm, 1.0);
        fillRandom(rng, w_q, init_scale);
        fillRandom(rng, w_k, init_scale);
        fillRandom(rng, w_v, init_scale);
        fillRandom(rng, w_o, init_scale);
        fillRandom(rng, w_gate, init_scale);
        fillRandom(rng, w_up, init_scale);
        fillRandom(rng, w_down, init_scale);
        lw.* = .{
            .w_n1 = w_n1,
            .w_q = w_q,
            .w_k = w_k,
            .w_v = w_v,
            .w_o = w_o,
            .w_n2 = w_n2,
            .w_gate = w_gate,
            .w_up = w_up,
            .w_down = w_down,
            .w_q_norm = w_q_norm,
            .w_k_norm = w_k_norm,
        };
    }

    // Random batch — overfit a single fixed (token_ids, target_ids) pair
    // so loss should drop monotonically.
    const token_ids = try allocator.alloc(u32, n_pos);
    defer allocator.free(token_ids);
    for (token_ids) |*t| t.* = rng.intRangeLessThan(u32, 0, vocab);
    const target_ids = try allocator.alloc(u32, n_pos);
    defer allocator.free(target_ids);
    for (target_ids) |*t| t.* = rng.intRangeLessThan(u32, 0, vocab);

    // CCE takes target ids directly — no one-hot expansion.

    // ── GPU bring-up.
    var ctx = try vk.Context.init(allocator);
    defer ctx.deinit();

    var runner = try train_transformer.Runner.init(
        allocator,
        &ctx,
        .{
            .dim = dim,
            .n_heads = n_heads,
            .n_kv_heads = n_kv_heads,
            .head_dim = head_dim,
            .ff_dim = ff_dim,
            .n_pos = n_pos,
            .n_layers = n_layers,
            .vocab_size = vocab,
            .rotary_dim = head_dim, // full RoPE
            .qk_norm = true, // per-head Q/K-norm (Qwen3 architectural detail)
            // Same lr as the toy 8c-α-3 smoke (1e-2). Single-batch
            // overfit is the regime we're in, not pre-training.
            .lr = 1e-2,
        },
        .{
            .embed = w_embed,
            .final_norm = w_final_norm,
            .lm_head = w_lm_head,
            .layers = layer_weights,
        },
    );
    defer runner.deinit();

    const logits_buf = try allocator.alloc(f32, @as(usize, n_pos) * vocab);
    defer allocator.free(logits_buf);

    try runner.forwardLogits(token_ids, logits_buf);
    const initial_loss = cpu_train_decoder.softmaxCeLoss(logits_buf, target_ids, n_pos, vocab);
    if (!std.math.isFinite(initial_loss)) {
        std.debug.print("Real-shape envelope: initial_loss not finite ({d})\n", .{initial_loss});
        return error.LossNotFinite;
    }

    const n_steps: u32 = 50;
    const t_start = std.time.nanoTimestamp();
    var step_t: u32 = 0;
    while (step_t < n_steps) : (step_t += 1) {
        try runner.step(token_ids, target_ids);
        if (step_t % 10 == 0 or step_t == n_steps - 1) {
            // Cheap heartbeat: forward + CE so we can watch the curve.
            try runner.forwardLogits(token_ids, logits_buf);
            const heartbeat = cpu_train_decoder.softmaxCeLoss(logits_buf, target_ids, n_pos, vocab);
            std.debug.print("  envelope step {d}: CE={d:.6}\n", .{ step_t + 1, heartbeat });
            if (!std.math.isFinite(heartbeat)) return error.LossNotFinite;
        }
    }
    const t_end = std.time.nanoTimestamp();
    const elapsed_ms: f64 = @as(f64, @floatFromInt(t_end - t_start)) / 1.0e6;
    const ms_per_step: f64 = elapsed_ms / @as(f64, @floatFromInt(n_steps));

    try runner.forwardLogits(token_ids, logits_buf);
    const final_loss = cpu_train_decoder.softmaxCeLoss(logits_buf, target_ids, n_pos, vocab);
    if (!std.math.isFinite(final_loss)) {
        std.debug.print("Real-shape envelope: final_loss not finite ({d}) after initial={d:.6}\n", .{ final_loss, initial_loss });
        return error.LossNotFinite;
    }
    if (final_loss >= initial_loss) {
        std.debug.print("Real-shape envelope: initial_loss={d:.6} final_loss={d:.6} (no decrease)\n", .{ initial_loss, final_loss });
        return error.LossDidNotDecrease;
    }

    std.debug.print(
        "PASS GPU decoder stack real-shape envelope (Qwen3-0.6B-class toy-arch: n_layers={d} dim={d} GQA {d}/{d} ff_dim={d} vocab={d} n_pos={d}; CE {d:.6} → {d:.6} over {d} Adam steps, {d:.1} ms/step)\n",
        .{ n_layers, dim, n_heads, n_kv_heads, ff_dim, vocab, n_pos, initial_loss, final_loss, n_steps, ms_per_step },
    );
}

// ── chunk 8c-β-3a: real Qwen3 weight load (fp32 train tensors) ───────
//
// Loads Qwen3-0.6B from the local HF cache, materialises fp32 training
// weights via `train_load_real.loadTrainWeightsFromId`, and gates the
// loader with three checks:
//
//   1. The derived `train_transformer.Config` matches Qwen3-0.6B's
//      published architecture (28 layers, 16/8 GQA at head_dim=128,
//      ff_dim=3072, vocab=151,936, full-RoPE, qk_norm on, rms_eps=1e-6,
//      rope_theta=1e6).
//   2. Every fp32 buffer has the expected element count.
//   3. A sampled subset of values from each tensor is finite. Cheap
//      vs scanning all ~720M params; catches any byte-pattern errors
//      from a wrong dtype path or shape misread.
//
// SKIP on `error.HfModelNotInCache` — fresh checkouts won't have the
// 1.2 GB safetensors local. PASS proves the bytes-on-disk → fp32 path
// is end-to-end correct; β-3b will run a forward pass to gate layout.

pub fn runRealModelLoadSmoke(allocator: std.mem.Allocator) !void {
    const model_id = "Qwen/Qwen3-0.6B";
    const n_pos: u32 = 64;

    var weights = train_load_real.loadTrainWeightsFromId(allocator, model_id, n_pos) catch |err| switch (err) {
        error.HfModelNotInCache => {
            std.debug.print("SKIP runRealModelLoadSmoke (Qwen3-0.6B not in HF cache)\n", .{});
            return;
        },
        else => return err,
    };
    defer weights.deinit();

    const cfg = weights.cfg;

    // ── 1. Architecture identity check.
    if (cfg.n_layers != 28) return error.UnexpectedLayers;
    if (cfg.dim != 1024) return error.UnexpectedHiddenSize;
    if (cfg.n_heads != 16) return error.UnexpectedNumHeads;
    if (cfg.n_kv_heads != 8) return error.UnexpectedNumKvHeads;
    if (cfg.head_dim != 128) return error.UnexpectedHeadDim;
    if (cfg.ff_dim != 3072) return error.UnexpectedFfDim;
    if (cfg.vocab_size != 151_936) return error.UnexpectedVocabSize;
    if (cfg.n_pos != n_pos) return error.UnexpectedNPos;
    if (cfg.rotary_dim != cfg.head_dim) return error.UnexpectedRotaryDim;
    if (!cfg.qk_norm) return error.QkNormShouldBeOn;
    if (@abs(cfg.rms_eps - 1e-6) > 1e-9) return error.UnexpectedRmsEps;
    if (@abs(cfg.rope_theta - 1_000_000.0) > 1.0) return error.UnexpectedRopeTheta;

    // ── 2. Shape (numel) check. Mirrors transformer.Runner.init's
    //    validation — if these slip past us, init would error first.
    const dim: usize = cfg.dim;
    const q_dim: usize = @as(usize, cfg.n_heads) * cfg.head_dim;
    const kv_dim: usize = @as(usize, cfg.n_kv_heads) * cfg.head_dim;
    const ff_dim: usize = cfg.ff_dim;
    const vocab: usize = cfg.vocab_size;
    const head_dim: usize = cfg.head_dim;

    if (weights.embed.len != vocab * dim) return error.EmbedNumelMismatch;
    if (weights.final_norm.len != dim) return error.FinalNormNumelMismatch;
    if (weights.lm_head.len != vocab * dim) return error.LmHeadNumelMismatch;
    if (weights.layers.len != cfg.n_layers) return error.LayersCountMismatch;

    for (weights.layers, 0..) |lw, li| {
        if (lw.w_n1.len != dim) return error.LayerN1NumelMismatch;
        if (lw.w_q.len != q_dim * dim) return error.LayerQNumelMismatch;
        if (lw.w_k.len != kv_dim * dim) return error.LayerKNumelMismatch;
        if (lw.w_v.len != kv_dim * dim) return error.LayerVNumelMismatch;
        if (lw.w_o.len != dim * q_dim) return error.LayerONumelMismatch;
        if (lw.w_n2.len != dim) return error.LayerN2NumelMismatch;
        if (lw.w_gate.len != ff_dim * dim) return error.LayerGateNumelMismatch;
        if (lw.w_up.len != ff_dim * dim) return error.LayerUpNumelMismatch;
        if (lw.w_down.len != dim * ff_dim) return error.LayerDownNumelMismatch;
        if (lw.w_q_norm.len != head_dim) return error.LayerQNormNumelMismatch;
        if (lw.w_k_norm.len != head_dim) return error.LayerKNormNumelMismatch;
        _ = li; // li available for richer errors if any of the above fire.
    }

    // ── 3. Sampled finiteness scan. We sample ~256 elements per
    //    tensor, evenly spaced — cheap and catches any wholesale-wrong
    //    bytes (dtype mismatch, shape misread, etc).
    const sample = struct {
        fn run(name: []const u8, slice: []const f32) !void {
            const n = slice.len;
            if (n == 0) return;
            const stride: usize = @max(1, n / 256);
            var i: usize = 0;
            while (i < n) : (i += stride) {
                if (!std.math.isFinite(slice[i])) {
                    std.debug.print("non-finite value in {s} at index {d}: {d}\n", .{ name, i, slice[i] });
                    return error.NonFiniteWeight;
                }
            }
        }
    }.run;

    try sample("embed", weights.embed);
    try sample("final_norm", weights.final_norm);
    try sample("lm_head", weights.lm_head);
    for (weights.layers, 0..) |lw, li| {
        // Only sample the big ones — rmsnorm gains are tiny and were
        // already covered by the shape check.
        try sample("w_q", lw.w_q);
        try sample("w_k", lw.w_k);
        try sample("w_v", lw.w_v);
        try sample("w_o", lw.w_o);
        try sample("w_gate", lw.w_gate);
        try sample("w_up", lw.w_up);
        try sample("w_down", lw.w_down);
        _ = li;
    }

    // ── Stat: total fp32 weight bytes (rough: matches what Runner
    //    would upload, modulo embed-vs-lm_head distinct copies).
    var total_f32: usize = weights.embed.len + weights.final_norm.len + weights.lm_head.len;
    for (weights.layers) |lw| {
        total_f32 += lw.w_n1.len + lw.w_q.len + lw.w_k.len + lw.w_v.len + lw.w_o.len;
        total_f32 += lw.w_n2.len + lw.w_gate.len + lw.w_up.len + lw.w_down.len;
        total_f32 += lw.w_q_norm.len + lw.w_k_norm.len;
    }
    const total_mib: f64 = @as(f64, @floatFromInt(total_f32 * @sizeOf(f32))) / (1024.0 * 1024.0);

    std.debug.print(
        "PASS real Qwen3-0.6B weight load (n_layers={d} dim={d} GQA {d}/{d} head_dim={d} ff_dim={d} vocab={d}; rope_theta={d:.0} rms_eps={e}; {d:.1} MiB fp32 host)\n",
        .{ cfg.n_layers, cfg.dim, cfg.n_heads, cfg.n_kv_heads, cfg.head_dim, cfg.ff_dim, cfg.vocab_size, cfg.rope_theta, cfg.rms_eps, total_mib },
    );
}

// ── chunk 8c-β-3b: real Qwen3 forward sanity ─────────────────────────
//
// β-3a proved the bytes-on-disk → fp32 conversion is correct shape-
// and value-wise; this proves they're plumbed through the trainer's
// forward path correctly. Catches anything β-3a's element-count gate
// misses: row/column-major confusion, RoPE convention drift, Q/K-norm
// applied at the wrong point, RMSNorm formula divergence, etc — every
// one of which would silently produce garbage logits.
//
// Method: load Qwen3-0.6B, instantiate `train_transformer.Runner` at
// n_pos=16 (cheap activations), tokenize a short fluent English
// prompt, run `forwardLogits`, then score CE for next-token prediction
// at every real prompt position. A correctly-wired Qwen3-0.6B should
// average ~2-5 nats on coherent English; ≥10 means the architecture
// pipeline is mis-wired (random-uniform CE is ln(151936) ≈ 11.93).
//
// Pass criterion: every logit finite + mean CE < 8.0. The window is
// generous so a small first-iteration RoPE-base/eps mismatch doesn't
// fail the gate; if mean CE comes back near 11-12 we know to dig.
//
// SKIP on `error.HfModelNotInCache`.

pub fn runRealModelForwardSmoke(allocator: std.mem.Allocator) !void {
    const model_id = "Qwen/Qwen3-0.6B";
    const n_pos: u32 = 16;
    const prompt_text = "The capital of France is Paris.";
    const ce_threshold: f32 = 8.0;

    // ── Resolve the model dir (also gives us a place to find the
    //    tokenizer.json without hard-coding the snapshot hash).
    const dir_path = hf_cache.resolveModelArg(allocator, model_id) catch |err| switch (err) {
        error.HfModelNotInCache => {
            std.debug.print("SKIP runRealModelForwardSmoke (Qwen3-0.6B not in HF cache)\n", .{});
            return;
        },
        else => return err,
    };
    defer allocator.free(dir_path);

    // ── Load fp32 train weights. Reuses the β-3a path (already
    //    validated by `runRealModelLoadSmoke`).
    var cpu = try model_mod.Model.load(allocator, dir_path);
    defer cpu.deinit();
    var weights = try train_load_real.loadTrainWeights(allocator, &cpu, n_pos);
    defer weights.deinit();

    // ── Tokenize the prompt. We need this *before* dropping the CPU
    //    model so we have a tokenizer. (Loader is the natural sibling
    //    that bundles tokenizer + weights, but β-3a's TrainWeights
    //    intentionally doesn't carry a tokenizer; β-4 will introduce
    //    a dataset abstraction that owns it.)
    const tok_path = try std.fmt.allocPrint(allocator, "{s}/tokenizer.json", .{dir_path});
    defer allocator.free(tok_path);
    var tok = try tokenizer_mod.Tokenizer.loadFromFile(allocator, tok_path);
    defer tok.deinit();
    const prompt_ids = try tok.encode(allocator, prompt_text);
    defer allocator.free(prompt_ids);
    const real_len: usize = prompt_ids.len;
    if (real_len < 2 or real_len > n_pos) {
        std.debug.print("Forward smoke: prompt tokenized to {d} ids; want 2..={d}\n", .{ real_len, n_pos });
        return error.PromptTokenLenOutOfRange;
    }

    // ── Build the n_pos token window: prompt followed by repeats of
    //    the last real id. Padding choice doesn't affect CE at real
    //    positions (we only score p in [0..real_len-2]), but sticking
    //    to in-vocab ids keeps the embedding lookup well-behaved.
    const token_ids = try allocator.alloc(u32, n_pos);
    defer allocator.free(token_ids);
    for (0..n_pos) |p| {
        token_ids[p] = if (p < real_len) prompt_ids[p] else prompt_ids[real_len - 1];
    }

    // ── GPU bring-up + Runner instantiation.
    var ctx = try vk.Context.init(allocator);
    defer ctx.deinit();

    var runner = try train_transformer.Runner.init(allocator, &ctx, weights.cfg, weights.view());
    defer runner.deinit();

    const logits = try allocator.alloc(f32, @as(usize, n_pos) * weights.cfg.vocab_size);
    defer allocator.free(logits);

    try runner.forwardLogits(token_ids, logits);

    // ── Finite-check (sampled). 16 positions × 151936 vocab is 2.4M
    //    elements — full scan is fine, but we mirror β-3a's stride
    //    sample to keep the smoke under a second.
    {
        const stride: usize = @max(1, logits.len / 4096);
        var i: usize = 0;
        while (i < logits.len) : (i += stride) {
            if (!std.math.isFinite(logits[i])) {
                std.debug.print("Forward smoke: non-finite logit at index {d}: {d}\n", .{ i, logits[i] });
                return error.NonFiniteLogit;
            }
        }
    }

    // ── Per-position CE for next-token prediction. logits[p] is the
    //    distribution over the token at position p+1, so we score
    //    pairs (logits[p], target=token_ids[p+1]) for p in [0..real_len-2].
    const vocab: usize = weights.cfg.vocab_size;
    var ce_sum: f64 = 0;
    var ce_n: usize = 0;
    for (0..real_len - 1) |p| {
        const off = p * vocab;
        // Numerically stable log-softmax: log Z = max + log Σ exp(x - max).
        var m: f32 = -std.math.inf(f32);
        for (0..vocab) |o| m = @max(m, logits[off + o]);
        var sum_e: f64 = 0;
        for (0..vocab) |o| sum_e += @exp(@as(f64, logits[off + o]) - @as(f64, m));
        const log_z: f64 = @as(f64, m) + @log(sum_e);
        const tgt: u32 = token_ids[p + 1];
        ce_sum += log_z - @as(f64, logits[off + tgt]);
        ce_n += 1;
    }
    const ce_mean: f32 = @floatCast(ce_sum / @as(f64, @floatFromInt(ce_n)));

    // ── Argmax at logits[real_len - 1]: the model's prediction for
    //    the token *immediately after* the prompt ends. No ground
    //    truth here (so no CE for this position), but it's the most
    //    illuminating qualitative signal — a sensibly-wired forward
    //    pass on "The capital of France is Paris." should suggest a
    //    plausible continuation (whitespace, a connective, EOS, etc).
    var argmax_after_prompt: u32 = 0;
    var argmax_score: f32 = -std.math.inf(f32);
    {
        const off = (real_len - 1) * vocab;
        for (0..vocab) |o| {
            if (logits[off + o] > argmax_score) {
                argmax_score = logits[off + o];
                argmax_after_prompt = @intCast(o);
            }
        }
    }

    if (!std.math.isFinite(ce_mean)) {
        std.debug.print("Forward smoke: mean CE not finite ({d})\n", .{ce_mean});
        return error.CeNotFinite;
    }
    if (ce_mean >= ce_threshold) {
        std.debug.print(
            "Forward smoke: mean CE {d:.3} ≥ threshold {d:.3} — model output looks random; check architecture wiring\n",
            .{ ce_mean, ce_threshold },
        );
        return error.CeAboveThreshold;
    }

    // Best-effort decode of the predicted post-prompt token for the
    // PASS line. decodeForDisplay handles GPT-2-style byte mapping
    // and special tokens; if it fails (rare), fall back to id only.
    const decoded = tok.decodeForDisplay(allocator, argmax_after_prompt) catch null;
    defer if (decoded) |d| allocator.free(d);

    if (decoded) |d| {
        std.debug.print(
            "PASS real Qwen3-0.6B forward sanity (n_pos={d} prompt_tokens={d} mean_CE={d:.3} nats; argmax-after-prompt={d} \"{s}\")\n",
            .{ n_pos, real_len, ce_mean, argmax_after_prompt, d },
        );
    } else {
        std.debug.print(
            "PASS real Qwen3-0.6B forward sanity (n_pos={d} prompt_tokens={d} mean_CE={d:.3} nats; argmax-after-prompt={d})\n",
            .{ n_pos, real_len, ce_mean, argmax_after_prompt },
        );
    }
}

// ── chunk 8c-β-4: tokenizer + dataset stub ───────────────────────────
//
// β-3a/3b proved the static loader path. β-4 adds the streaming side:
// a `train_dataset.Dataset` packs tokenized examples into one stream
// with EOS separators, and produces sliding (n_pos+1)-windows as
// (input_ids, target_ids) batches for next-token-prediction training.
// β-5 will iterate over these and call `runner.step` once per batch.
//
// The new module supports both in-memory examples and a JSONL file
// (one `{"text": "..."}` object per line). This smoke uses the JSONL
// path to exercise both code paths end-to-end and ships a tiny
// fact-style dataset at `data/train/tiny_facts.jsonl` so future
// chunks have a hermetic corpus to overfit on.
//
// Pass criterion: dataset builds with packed_ids longer than n_pos+1,
// at least one batch is producible, forward CE on the first batch is
// finite + below `pretrained_ce_threshold`. The threshold is loose
// (8 nats) because we're running a *trained* Qwen3-0.6B on factual
// English — actual CE will land in the 1.5-4 nat regime.

pub fn runRealModelDatasetSmoke(allocator: std.mem.Allocator) !void {
    const model_id = "Qwen/Qwen3-0.6B";
    const jsonl_path = "data/train/tiny_facts.jsonl";
    const n_pos: u32 = 16;
    const eos_id: u32 = 151_645; // Qwen3 <|im_end|>; matches config.eos_token_id.
    const ce_threshold: f32 = 8.0;

    const dir_path = hf_cache.resolveModelArg(allocator, model_id) catch |err| switch (err) {
        error.HfModelNotInCache => {
            std.debug.print("SKIP runRealModelDatasetSmoke (Qwen3-0.6B not in HF cache)\n", .{});
            return;
        },
        else => return err,
    };
    defer allocator.free(dir_path);

    var cpu = try model_mod.Model.load(allocator, dir_path);
    defer cpu.deinit();

    var weights = try train_load_real.loadTrainWeights(allocator, &cpu, n_pos);
    defer weights.deinit();

    const tok_path = try std.fmt.allocPrint(allocator, "{s}/tokenizer.json", .{dir_path});
    defer allocator.free(tok_path);
    var tok = try tokenizer_mod.Tokenizer.loadFromFile(allocator, tok_path);
    defer tok.deinit();

    var ds = try train_dataset.buildFromJsonl(allocator, &tok, jsonl_path, n_pos, eos_id);
    defer ds.deinit();

    if (ds.numBatches() == 0) {
        std.debug.print("Dataset smoke: numBatches=0 ({d} packed ids ≤ n_pos={d})\n", .{ ds.packed_ids.len, n_pos });
        return error.DatasetTooShort;
    }

    // ── Read batch 0 and verify the next-token shift is intact.
    const input_ids = try allocator.alloc(u32, n_pos);
    defer allocator.free(input_ids);
    const target_ids = try allocator.alloc(u32, n_pos);
    defer allocator.free(target_ids);
    try ds.batch(0, input_ids, target_ids);

    for (0..n_pos) |p| {
        // packed_ids[p+1] should equal both target_ids[p] and (for p<n_pos-1)
        // input_ids[p+1]. The first guarantees the shift is correct;
        // the second confirms the windowing is consistent.
        if (target_ids[p] != ds.packed_ids[p + 1]) return error.TargetShiftMismatch;
        if (p + 1 < n_pos and input_ids[p + 1] != ds.packed_ids[p + 1]) return error.InputWindowMismatch;
    }

    // ── Forward + CE on the first batch.
    var ctx = try vk.Context.init(allocator);
    defer ctx.deinit();
    var runner = try train_transformer.Runner.init(allocator, &ctx, weights.cfg, weights.view());
    defer runner.deinit();

    const vocab: usize = weights.cfg.vocab_size;
    const logits = try allocator.alloc(f32, @as(usize, n_pos) * vocab);
    defer allocator.free(logits);
    try runner.forwardLogits(input_ids, logits);

    var ce_sum: f64 = 0;
    for (0..n_pos) |p| {
        const off = p * vocab;
        var m: f32 = -std.math.inf(f32);
        for (0..vocab) |o| m = @max(m, logits[off + o]);
        var sum_e: f64 = 0;
        for (0..vocab) |o| sum_e += @exp(@as(f64, logits[off + o]) - @as(f64, m));
        const log_z: f64 = @as(f64, m) + @log(sum_e);
        ce_sum += log_z - @as(f64, logits[off + target_ids[p]]);
    }
    const ce_mean: f32 = @floatCast(ce_sum / @as(f64, @floatFromInt(n_pos)));

    if (!std.math.isFinite(ce_mean)) {
        std.debug.print("Dataset smoke: mean CE not finite ({d})\n", .{ce_mean});
        return error.CeNotFinite;
    }
    if (ce_mean >= ce_threshold) {
        std.debug.print(
            "Dataset smoke: batch-0 mean CE {d:.3} ≥ threshold {d:.3}\n",
            .{ ce_mean, ce_threshold },
        );
        return error.CeAboveThreshold;
    }

    std.debug.print(
        "PASS real Qwen3-0.6B dataset (jsonl→{d} packed ids; {d} batches at n_pos={d}; batch-0 mean_CE={d:.3} nats)\n",
        .{ ds.packed_ids.len, ds.numBatches(), n_pos, ce_mean },
    );
}

// ── chunk 8c-β-5: end-to-end one-step real-model train ──────────────
//
// The "does it actually train" gate. β-3a/3b proved the static loader
// is correct, β-4 proved the streaming side is correct. β-5 closes the
// loop: load Qwen3-0.6B, sample one real batch from the dataset, take
// one Adam step, forward again on the same batch, assert the loss
// decreased.
//
// Why a small lr matters: the β-2 envelope uses lr=1e-2 because it's
// overfitting random-init weights from 0.02 scale, where one update
// of size lr*sign(g) ≈ 1e-2 is a 50% relative perturbation. Pretrained
// weights are at scales of 0.01-0.5 with carefully-tuned magnitudes;
// a 1e-2 update would catastrophically diverge them. Standard real-
// world fine-tune lr is 1e-5 to 5e-5 — we use 1e-5, which gives a
// first-step relative perturbation of 0.01-0.1% per weight (small but
// measurable in the loss).
//
// Pass criterion:
//   - CE_before finite
//   - CE_after finite
//   - CE_after < CE_before (the actual gate — if false, either the
//     gradient is zero, the lr is wrong, or training is broken)
//
// The decrease is expected to be small (a few % of CE_before) because
// (1) lr is conservative and (2) we're stepping on a single small
// batch — overfitting one window with one Adam step will visibly
// shift its CE but won't dent broader Qwen3-0.6B knowledge. β-6 will
// loop this for many steps + checkpoints.

pub fn runRealModelTrainStepSmoke(allocator: std.mem.Allocator) !void {
    const model_id = "Qwen/Qwen3-0.6B";
    const jsonl_path = "data/train/tiny_facts.jsonl";
    const n_pos: u32 = 16;
    const eos_id: u32 = 151_645;
    const lr: f32 = 1e-5;

    const dir_path = hf_cache.resolveModelArg(allocator, model_id) catch |err| switch (err) {
        error.HfModelNotInCache => {
            std.debug.print("SKIP runRealModelTrainStepSmoke (Qwen3-0.6B not in HF cache)\n", .{});
            return;
        },
        else => return err,
    };
    defer allocator.free(dir_path);

    var cpu = try model_mod.Model.load(allocator, dir_path);
    defer cpu.deinit();

    var weights = try train_load_real.loadTrainWeights(allocator, &cpu, n_pos);
    defer weights.deinit();
    // Override the trainer's default lr with a fine-tune-appropriate
    // value before instantiation. Config is value-passed into init,
    // and Runner stores it; setting via .lr field on a copy is fine.
    var cfg = weights.cfg;
    cfg.lr = lr;

    const tok_path = try std.fmt.allocPrint(allocator, "{s}/tokenizer.json", .{dir_path});
    defer allocator.free(tok_path);
    var tok = try tokenizer_mod.Tokenizer.loadFromFile(allocator, tok_path);
    defer tok.deinit();

    var ds = try train_dataset.buildFromJsonl(allocator, &tok, jsonl_path, n_pos, eos_id);
    defer ds.deinit();
    if (ds.numBatches() == 0) return error.DatasetTooShort;

    const input_ids = try allocator.alloc(u32, n_pos);
    defer allocator.free(input_ids);
    const target_ids = try allocator.alloc(u32, n_pos);
    defer allocator.free(target_ids);
    try ds.batch(0, input_ids, target_ids);

    // CCE forward consumes target ids directly — no [n_pos × vocab] one-hot.
    const vocab: usize = cfg.vocab_size;

    // ── GPU bring-up + Runner.
    var ctx = try vk.Context.init(allocator);
    defer ctx.deinit();

    var runner = try train_transformer.Runner.init(allocator, &ctx, cfg, weights.view());
    defer runner.deinit();

    const logits = try allocator.alloc(f32, @as(usize, n_pos) * vocab);
    defer allocator.free(logits);

    // ── CE before.
    try runner.forwardLogits(input_ids, logits);
    const ce_before = computeMeanCe(logits, target_ids, n_pos, vocab);
    if (!std.math.isFinite(ce_before)) {
        std.debug.print("Train-step smoke: CE_before not finite ({d})\n", .{ce_before});
        return error.CeBeforeNotFinite;
    }

    // ── One Adam step on this single batch.
    const t_step_start = std.time.nanoTimestamp();
    try runner.step(input_ids, target_ids);
    const t_step_end = std.time.nanoTimestamp();
    const step_ms: f64 = @as(f64, @floatFromInt(t_step_end - t_step_start)) / 1.0e6;

    // ── CE after.
    try runner.forwardLogits(input_ids, logits);
    const ce_after = computeMeanCe(logits, target_ids, n_pos, vocab);
    if (!std.math.isFinite(ce_after)) {
        std.debug.print("Train-step smoke: CE_after not finite ({d}) after CE_before={d:.6}\n", .{ ce_after, ce_before });
        return error.CeAfterNotFinite;
    }

    if (ce_after >= ce_before) {
        std.debug.print(
            "Train-step smoke: CE did not decrease (before={d:.6} after={d:.6} delta={d:.6}) at lr={e}\n",
            .{ ce_before, ce_after, ce_after - ce_before, lr },
        );
        return error.CeDidNotDecrease;
    }

    const delta = ce_before - ce_after;
    const rel_pct: f64 = 100.0 * @as(f64, delta) / @as(f64, ce_before);
    std.debug.print(
        "PASS real Qwen3-0.6B one-step train (n_pos={d} lr={e}; CE {d:.6} → {d:.6}, Δ={d:.6} ({d:.3}%); step={d:.1} ms)\n",
        .{ n_pos, lr, ce_before, ce_after, delta, rel_pct, step_ms },
    );
}

// ── A4-2: real Qwen3-0.6B β-5 with LoRA-Q wired in ───────────────────
//
// Same shape and dataset as `runRealModelTrainStepSmoke` but with
// `cfg.lora_q_enabled = true` (rank-16, α=32 ⇒ α/r=2). Validates the
// LoRA-Q wiring at production scale (28 Qwen3-0.6B layers × 1024-dim
// W_q each) and reports ms/step so we can compare against the
// no-LoRA baseline (saved in memory: 265 ms post-fused-QK-RoPE).
//
// Pass criteria match β-5: CE_before / CE_after finite + CE_after <
// CE_before. With B = 0 init and only A,B trainable, ∇A = 0 on step 1
// — but ∇B is non-zero, so step 1 still moves B and tilts the LoRA
// delta. CE should still drop visibly even at lr=1e-5 (typical fine-
// tune scale).

pub fn runRealModelTrainStepLoraQSmoke(allocator: std.mem.Allocator) !void {
    return runRealModelTrainStepLoraTargetsSmoke(allocator, train_transformer.LoraTarget.q, "Q-only");
}

pub fn runRealModelTrainStepLoraAllSmoke(allocator: std.mem.Allocator) !void {
    return runRealModelTrainStepLoraTargetsSmoke(allocator, train_transformer.LoraTarget.all, "all-7");
}

fn runRealModelTrainStepLoraTargetsSmoke(
    allocator: std.mem.Allocator,
    targets: u32,
    label: []const u8,
) !void {
    const model_id = "Qwen/Qwen3-0.6B";
    const jsonl_path = "data/train/tiny_facts.jsonl";
    const n_pos: u32 = 16;
    const eos_id: u32 = 151_645;
    const lr: f32 = 1e-5;
    const lora_rank: u32 = 16;
    const lora_alpha: f32 = 32.0;

    const dir_path = hf_cache.resolveModelArg(allocator, model_id) catch |err| switch (err) {
        error.HfModelNotInCache => {
            std.debug.print("SKIP runRealModelTrainStepLora{s}Smoke (Qwen3-0.6B not in HF cache)\n", .{label});
            return;
        },
        else => return err,
    };
    defer allocator.free(dir_path);

    var cpu = try model_mod.Model.load(allocator, dir_path);
    defer cpu.deinit();

    var weights = try train_load_real.loadTrainWeights(allocator, &cpu, n_pos);
    defer weights.deinit();
    var cfg = weights.cfg;
    cfg.lr = lr;
    cfg.lora_targets = targets;
    cfg.lora_rank = lora_rank;
    cfg.lora_alpha = lora_alpha;

    const tok_path = try std.fmt.allocPrint(allocator, "{s}/tokenizer.json", .{dir_path});
    defer allocator.free(tok_path);
    var tok = try tokenizer_mod.Tokenizer.loadFromFile(allocator, tok_path);
    defer tok.deinit();

    var ds = try train_dataset.buildFromJsonl(allocator, &tok, jsonl_path, n_pos, eos_id);
    defer ds.deinit();
    if (ds.numBatches() == 0) return error.DatasetTooShort;

    const input_ids = try allocator.alloc(u32, n_pos);
    defer allocator.free(input_ids);
    const target_ids = try allocator.alloc(u32, n_pos);
    defer allocator.free(target_ids);
    try ds.batch(0, input_ids, target_ids);

    const vocab: usize = cfg.vocab_size;

    var ctx = try vk.Context.init(allocator);
    defer ctx.deinit();

    var runner = try train_transformer.Runner.init(allocator, &ctx, cfg, weights.view());
    defer runner.deinit();

    const logits = try allocator.alloc(f32, @as(usize, n_pos) * vocab);
    defer allocator.free(logits);

    try runner.forwardLogits(input_ids, logits);
    const ce_before = computeMeanCe(logits, target_ids, n_pos, vocab);
    if (!std.math.isFinite(ce_before)) {
        std.debug.print("LoRA train-step smoke ({s}): CE_before not finite ({d})\n", .{ label, ce_before });
        return error.CeBeforeNotFinite;
    }

    const t_step_start = std.time.nanoTimestamp();
    try runner.step(input_ids, target_ids);
    const t_step_end = std.time.nanoTimestamp();
    const step_ms: f64 = @as(f64, @floatFromInt(t_step_end - t_step_start)) / 1.0e6;

    try runner.forwardLogits(input_ids, logits);
    const ce_after = computeMeanCe(logits, target_ids, n_pos, vocab);
    if (!std.math.isFinite(ce_after)) {
        std.debug.print("LoRA train-step smoke ({s}): CE_after not finite ({d}) after CE_before={d:.6}\n", .{ label, ce_after, ce_before });
        return error.CeAfterNotFinite;
    }

    if (ce_after >= ce_before) {
        std.debug.print(
            "LoRA train-step smoke ({s}): CE did not decrease (before={d:.6} after={d:.6} delta={d:.6}) at lr={e}\n",
            .{ label, ce_before, ce_after, ce_after - ce_before, lr },
        );
        return error.CeDidNotDecrease;
    }

    const delta = ce_before - ce_after;
    const rel_pct: f64 = 100.0 * @as(f64, delta) / @as(f64, ce_before);
    std.debug.print(
        "PASS real Qwen3-0.6B LoRA one-step train ({s}, {d}/7 targets, n_pos={d} lr={e} rank={d} α={d:.0} α/r={d:.2}; CE {d:.6} → {d:.6}, Δ={d:.6} ({d:.3}%); step={d:.1} ms)\n",
        .{ label, @popCount(targets), n_pos, lr, lora_rank, lora_alpha, lora_alpha / @as(f32, @floatFromInt(lora_rank)), ce_before, ce_after, delta, rel_pct, step_ms },
    );
}

// ── chunk 8c-β-6b: checkpoint save/load round-trip ──────────────────
//
// Trains a toy 8c-α-3-shape Runner for K steps, saves a checkpoint,
// then trains M more steps to establish the "continuous" trajectory.
// Spins up a *fresh* Runner with identical Config, loads the same
// checkpoint into it, trains M more steps, and asserts the post-load
// trajectory matches the continuous one within fp tolerance.
//
// Three gates:
//   1. Round-trip integrity (no train): CE on Runner_2 immediately
//      after load matches CE on Runner_1 at save time.
//   2. Step-t restored: Runner_2's first post-load Adam step uses the
//      correct bias-corrected update — checked indirectly by the
//      trajectory match below.
//   3. Adam m/v restored: Runner_2's M-step trajectory matches
//      Runner_1's continuous K+M trajectory. If we forgot to save m or
//      v, momentum "resets" and CE diverges.
//
// Tolerance is generous (1e-2 absolute on a CE value of ~1.0) because
// GPU subgroup-sum reductions are non-deterministic across runs (and
// each Adam step compounds the jitter through 51-127 dispatches at
// this shape). If round-trip integrity holds tightly (<1e-5), the
// post-train delta is what shows the m/v + step_t recovery worked.

pub fn runDecoderStackCheckpointSmoke(allocator: std.mem.Allocator) !void {
    const cfg_static = cpu_train_decoder.StackConfig{
        .base = .{
            .dim = 16,
            .n_heads = 2,
            .n_kv_heads = 2,
            .head_dim = 8,
            .ff_dim = 32,
            .n_pos = 4,
            .rms_eps = 1e-5,
            .causal = true,
            .rotary_dim = 8,
            .qk_norm = true,
        },
        .n_layers = 2,
        .vocab_size = 8,
    };
    const dim = cfg_static.base.dim;
    const n_pos = cfg_static.base.n_pos;
    const n_heads = cfg_static.base.n_heads;
    const n_kv_heads = cfg_static.base.n_kv_heads;
    const head_dim = cfg_static.base.head_dim;
    const ff_dim = cfg_static.base.ff_dim;
    const vocab = cfg_static.vocab_size;
    const q_dim = n_heads * head_dim;
    const kv_dim = n_kv_heads * head_dim;

    // Fixed RNG seed so initial weights + token_ids + target_ids are
    // identical across every Runner instantiation in this smoke.
    const seed: u64 = 0xC0_DE_CC_07;
    const init_scale: f32 = 0.1;

    // Build the cpu-side weight buffers we'll feed into both Runners.
    // Two arenas (one per Runner) keep ownership clear.
    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();
    const aa = arena.allocator();

    const w_embed = try aa.alloc(f32, vocab * dim);
    const w_final_norm = try aa.alloc(f32, dim);
    const w_lm_head = try aa.alloc(f32, vocab * dim);

    var prng = std.Random.DefaultPrng.init(seed);
    var rng = prng.random();
    for (w_embed) |*v| v.* = (rng.float(f32) * 2.0 - 1.0) * init_scale;
    for (w_final_norm) |*v| v.* = 1.0;
    for (w_lm_head) |*v| v.* = (rng.float(f32) * 2.0 - 1.0) * init_scale;

    const layer_weights = try aa.alloc(train_transformer.LayerWeights, cfg_static.n_layers);
    for (layer_weights) |*lw| {
        const w_n1 = try aa.alloc(f32, dim);
        const w_q = try aa.alloc(f32, q_dim * dim);
        const w_k = try aa.alloc(f32, kv_dim * dim);
        const w_v = try aa.alloc(f32, kv_dim * dim);
        const w_o = try aa.alloc(f32, dim * q_dim);
        const w_n2 = try aa.alloc(f32, dim);
        const w_gate = try aa.alloc(f32, ff_dim * dim);
        const w_up = try aa.alloc(f32, ff_dim * dim);
        const w_down = try aa.alloc(f32, dim * ff_dim);
        const w_q_norm = try aa.alloc(f32, head_dim);
        const w_k_norm = try aa.alloc(f32, head_dim);
        for (w_n1) |*v| v.* = 1.0;
        for (w_n2) |*v| v.* = 1.0;
        for (w_q_norm) |*v| v.* = 1.0;
        for (w_k_norm) |*v| v.* = 1.0;
        for (w_q) |*v| v.* = (rng.float(f32) * 2.0 - 1.0) * init_scale;
        for (w_k) |*v| v.* = (rng.float(f32) * 2.0 - 1.0) * init_scale;
        for (w_v) |*v| v.* = (rng.float(f32) * 2.0 - 1.0) * init_scale;
        for (w_o) |*v| v.* = (rng.float(f32) * 2.0 - 1.0) * init_scale;
        for (w_gate) |*v| v.* = (rng.float(f32) * 2.0 - 1.0) * init_scale;
        for (w_up) |*v| v.* = (rng.float(f32) * 2.0 - 1.0) * init_scale;
        for (w_down) |*v| v.* = (rng.float(f32) * 2.0 - 1.0) * init_scale;
        lw.* = .{
            .w_n1 = w_n1,
            .w_q = w_q,
            .w_k = w_k,
            .w_v = w_v,
            .w_o = w_o,
            .w_n2 = w_n2,
            .w_gate = w_gate,
            .w_up = w_up,
            .w_down = w_down,
            .w_q_norm = w_q_norm,
            .w_k_norm = w_k_norm,
        };
    }

    const token_ids = try aa.alloc(u32, n_pos);
    const target_ids = try aa.alloc(u32, n_pos);
    for (token_ids) |*tid| tid.* = rng.intRangeLessThan(u32, 0, @intCast(vocab));
    for (target_ids) |*tid| tid.* = rng.intRangeLessThan(u32, 0, @intCast(vocab));

    var ctx = try vk.Context.init(allocator);
    defer ctx.deinit();

    const runner_cfg = train_transformer.Config{
        .dim = @intCast(dim),
        .n_heads = @intCast(n_heads),
        .n_kv_heads = @intCast(n_kv_heads),
        .head_dim = @intCast(head_dim),
        .ff_dim = @intCast(ff_dim),
        .n_pos = @intCast(n_pos),
        .n_layers = @intCast(cfg_static.n_layers),
        .vocab_size = @intCast(vocab),
        .rms_eps = cfg_static.base.rms_eps,
        .causal = cfg_static.base.causal,
        .rotary_dim = @intCast(cfg_static.base.rotary_dim),
        .rope_theta = cfg_static.base.rope_theta,
        .qk_norm = cfg_static.base.qk_norm,
        .lr = 1e-2,
    };
    const init_weights = train_transformer.InitWeights{
        .embed = w_embed,
        .final_norm = w_final_norm,
        .lm_head = w_lm_head,
        .layers = layer_weights,
    };

    const k_steps: u32 = 10;
    const m_steps: u32 = 10;
    const ckpt_path = "/tmp/valkyr_ckpt_smoke.vkpt";
    defer std.fs.cwd().deleteFile(ckpt_path) catch {};

    const logits_buf = try allocator.alloc(f32, n_pos * vocab);
    defer allocator.free(logits_buf);

    // ── Runner A: train K, save, train M more.
    var runner_a = try train_transformer.Runner.init(allocator, &ctx, runner_cfg, init_weights);
    defer runner_a.deinit();

    var i: u32 = 0;
    while (i < k_steps) : (i += 1) try runner_a.step(token_ids, target_ids);

    try runner_a.forwardLogits(token_ids, logits_buf);
    const ce_at_save = cpu_train_decoder.softmaxCeLoss(logits_buf, target_ids, n_pos, vocab);
    if (!std.math.isFinite(ce_at_save)) return error.CeAtSaveNotFinite;

    try runner_a.saveCheckpoint(allocator, ckpt_path);

    i = 0;
    while (i < m_steps) : (i += 1) try runner_a.step(token_ids, target_ids);

    try runner_a.forwardLogits(token_ids, logits_buf);
    const ce_continuous = cpu_train_decoder.softmaxCeLoss(logits_buf, target_ids, n_pos, vocab);
    if (!std.math.isFinite(ce_continuous)) return error.CeContinuousNotFinite;

    // ── Runner B: fresh init, load checkpoint, train M more.
    var runner_b = try train_transformer.Runner.init(allocator, &ctx, runner_cfg, init_weights);
    defer runner_b.deinit();
    try runner_b.loadCheckpoint(allocator, ckpt_path);

    // Gate 1: round-trip integrity. After load (no further training)
    // CE on Runner B should match CE at save on Runner A modulo
    // GPU forward non-determinism (very small at this shape).
    try runner_b.forwardLogits(token_ids, logits_buf);
    const ce_after_load = cpu_train_decoder.softmaxCeLoss(logits_buf, target_ids, n_pos, vocab);
    const roundtrip_delta = @abs(ce_after_load - ce_at_save);
    if (roundtrip_delta > 1e-3) {
        std.debug.print(
            "Checkpoint round-trip FAIL: ce_at_save={d:.6} ce_after_load={d:.6} delta={e:.3}\n",
            .{ ce_at_save, ce_after_load, roundtrip_delta },
        );
        return error.CheckpointRoundTripFailed;
    }

    // Gate 2 + 3: trajectory continuity. Runner B trained M steps
    // post-load should land near Runner A's continuous K+M endpoint.
    // If Adam m/v or step_t weren't restored, momentum resets and the
    // M-step trajectory diverges (typically by 0.1-1.0 in CE).
    i = 0;
    while (i < m_steps) : (i += 1) try runner_b.step(token_ids, target_ids);

    try runner_b.forwardLogits(token_ids, logits_buf);
    const ce_after_resume = cpu_train_decoder.softmaxCeLoss(logits_buf, target_ids, n_pos, vocab);
    if (!std.math.isFinite(ce_after_resume)) return error.CeAfterResumeNotFinite;

    const trajectory_delta = @abs(ce_after_resume - ce_continuous);
    if (trajectory_delta > 1e-2) {
        std.debug.print(
            "Checkpoint trajectory FAIL: ce_continuous={d:.6} ce_after_resume={d:.6} delta={e:.3}\n",
            .{ ce_continuous, ce_after_resume, trajectory_delta },
        );
        return error.CheckpointTrajectoryDivergent;
    }

    std.debug.print(
        "PASS Runner checkpoint round-trip ({d}-layer toy stack qk_norm+RoPE; K={d}+M={d} steps; CE save={d:.6} load={d:.6} (Δ={e:.2}); resume {d:.6} vs continuous {d:.6} (Δ={e:.2}))\n",
        .{ cfg_static.n_layers, k_steps, m_steps, ce_at_save, ce_after_load, roundtrip_delta, ce_after_resume, ce_continuous, trajectory_delta },
    );
}

// ── A4-4: `.lvkpt` (LoRA-only) round-trip smoke ──────────────────────
//
// Mirror of runDecoderStackCheckpointSmoke but with LoRA enabled on
// the attention block (Q+K+V+O, rank-4). Saves a `.lvkpt`; reloads
// into a fresh Runner with identical Config; asserts both
//   1. round-trip CE match (Runner B post-load == Runner A at save),
//   2. trajectory match (Runner B post-load + M more steps == Runner
//      A's continuous K+M endpoint).
// Gate 2 specifically validates that Adam m/v + step_t round-trip too:
// without them, the M-step trajectory after load would diverge by
// 0.1-1.0 in CE since momentum resets to zero at the wrong step.
//
// Fixed seed + same toy shape as the `.vkpt` smoke so the two can run
// side-by-side in CI without parameter drift.

pub fn runDecoderStackLoraCheckpointSmoke(allocator: std.mem.Allocator) !void {
    const cfg_static = cpu_train_decoder.StackConfig{
        .base = .{
            .dim = 16,
            .n_heads = 2,
            .n_kv_heads = 2,
            .head_dim = 8,
            .ff_dim = 32,
            .n_pos = 4,
            .rms_eps = 1e-5,
            .causal = true,
            .rotary_dim = 8,
            .qk_norm = true,
        },
        .n_layers = 2,
        .vocab_size = 8,
    };
    const dim = cfg_static.base.dim;
    const n_pos = cfg_static.base.n_pos;
    const n_heads = cfg_static.base.n_heads;
    const n_kv_heads = cfg_static.base.n_kv_heads;
    const head_dim = cfg_static.base.head_dim;
    const ff_dim = cfg_static.base.ff_dim;
    const vocab = cfg_static.vocab_size;
    const q_dim = n_heads * head_dim;
    const kv_dim = n_kv_heads * head_dim;

    const seed: u64 = 0xC0_DE_CC_07;
    const init_scale: f32 = 0.1;
    const lora_rank: u32 = 4;
    const lora_alpha: f32 = 8.0;

    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();
    const aa = arena.allocator();

    const w_embed = try aa.alloc(f32, vocab * dim);
    const w_final_norm = try aa.alloc(f32, dim);
    const w_lm_head = try aa.alloc(f32, vocab * dim);

    var prng = std.Random.DefaultPrng.init(seed);
    var rng = prng.random();
    for (w_embed) |*v| v.* = (rng.float(f32) * 2.0 - 1.0) * init_scale;
    for (w_final_norm) |*v| v.* = 1.0;
    for (w_lm_head) |*v| v.* = (rng.float(f32) * 2.0 - 1.0) * init_scale;

    const layer_weights = try aa.alloc(train_transformer.LayerWeights, cfg_static.n_layers);
    for (layer_weights) |*lw| {
        const w_n1 = try aa.alloc(f32, dim);
        const w_q = try aa.alloc(f32, q_dim * dim);
        const w_k = try aa.alloc(f32, kv_dim * dim);
        const w_v = try aa.alloc(f32, kv_dim * dim);
        const w_o = try aa.alloc(f32, dim * q_dim);
        const w_n2 = try aa.alloc(f32, dim);
        const w_gate = try aa.alloc(f32, ff_dim * dim);
        const w_up = try aa.alloc(f32, ff_dim * dim);
        const w_down = try aa.alloc(f32, dim * ff_dim);
        const w_q_norm = try aa.alloc(f32, head_dim);
        const w_k_norm = try aa.alloc(f32, head_dim);
        for (w_n1) |*v| v.* = 1.0;
        for (w_n2) |*v| v.* = 1.0;
        for (w_q_norm) |*v| v.* = 1.0;
        for (w_k_norm) |*v| v.* = 1.0;
        for (w_q) |*v| v.* = (rng.float(f32) * 2.0 - 1.0) * init_scale;
        for (w_k) |*v| v.* = (rng.float(f32) * 2.0 - 1.0) * init_scale;
        for (w_v) |*v| v.* = (rng.float(f32) * 2.0 - 1.0) * init_scale;
        for (w_o) |*v| v.* = (rng.float(f32) * 2.0 - 1.0) * init_scale;
        for (w_gate) |*v| v.* = (rng.float(f32) * 2.0 - 1.0) * init_scale;
        for (w_up) |*v| v.* = (rng.float(f32) * 2.0 - 1.0) * init_scale;
        for (w_down) |*v| v.* = (rng.float(f32) * 2.0 - 1.0) * init_scale;
        lw.* = .{
            .w_n1 = w_n1,
            .w_q = w_q,
            .w_k = w_k,
            .w_v = w_v,
            .w_o = w_o,
            .w_n2 = w_n2,
            .w_gate = w_gate,
            .w_up = w_up,
            .w_down = w_down,
            .w_q_norm = w_q_norm,
            .w_k_norm = w_k_norm,
        };
    }

    const token_ids = try aa.alloc(u32, n_pos);
    const target_ids = try aa.alloc(u32, n_pos);
    for (token_ids) |*tid| tid.* = rng.intRangeLessThan(u32, 0, @intCast(vocab));
    for (target_ids) |*tid| tid.* = rng.intRangeLessThan(u32, 0, @intCast(vocab));

    var ctx = try vk.Context.init(allocator);
    defer ctx.deinit();

    const runner_cfg = train_transformer.Config{
        .dim = @intCast(dim),
        .n_heads = @intCast(n_heads),
        .n_kv_heads = @intCast(n_kv_heads),
        .head_dim = @intCast(head_dim),
        .ff_dim = @intCast(ff_dim),
        .n_pos = @intCast(n_pos),
        .n_layers = @intCast(cfg_static.n_layers),
        .vocab_size = @intCast(vocab),
        .rms_eps = cfg_static.base.rms_eps,
        .causal = cfg_static.base.causal,
        .rotary_dim = @intCast(cfg_static.base.rotary_dim),
        .rope_theta = cfg_static.base.rope_theta,
        .qk_norm = cfg_static.base.qk_norm,
        .lr = 1e-2,
        .lora_targets = train_transformer.LoraTarget.all_attn,
        .lora_rank = lora_rank,
        .lora_alpha = lora_alpha,
    };
    const init_weights = train_transformer.InitWeights{
        .embed = w_embed,
        .final_norm = w_final_norm,
        .lm_head = w_lm_head,
        .layers = layer_weights,
    };

    const k_steps: u32 = 10;
    const m_steps: u32 = 10;
    const ckpt_path = "/tmp/valkyr_lora_ckpt_smoke.lvkpt";
    defer std.fs.cwd().deleteFile(ckpt_path) catch {};

    const logits_buf = try allocator.alloc(f32, n_pos * vocab);
    defer allocator.free(logits_buf);

    // ── Runner A: train K, save, train M more.
    var runner_a = try train_transformer.Runner.init(allocator, &ctx, runner_cfg, init_weights);
    defer runner_a.deinit();

    var i: u32 = 0;
    while (i < k_steps) : (i += 1) try runner_a.step(token_ids, target_ids);

    try runner_a.forwardLogits(token_ids, logits_buf);
    const ce_at_save = cpu_train_decoder.softmaxCeLoss(logits_buf, target_ids, n_pos, vocab);
    if (!std.math.isFinite(ce_at_save)) return error.CeAtSaveNotFinite;

    try runner_a.saveLoraCheckpoint(allocator, ckpt_path);

    i = 0;
    while (i < m_steps) : (i += 1) try runner_a.step(token_ids, target_ids);

    try runner_a.forwardLogits(token_ids, logits_buf);
    const ce_continuous = cpu_train_decoder.softmaxCeLoss(logits_buf, target_ids, n_pos, vocab);
    if (!std.math.isFinite(ce_continuous)) return error.CeContinuousNotFinite;

    // ── Runner B: fresh init (same base weights, default LoRA init),
    //    load `.lvkpt`, train M more, compare to continuous endpoint.
    var runner_b = try train_transformer.Runner.init(allocator, &ctx, runner_cfg, init_weights);
    defer runner_b.deinit();
    try runner_b.loadLoraCheckpoint(allocator, ckpt_path);

    // Gate 1: round-trip integrity. CE on Runner B post-load should
    // match Runner A's CE at save modulo small fp non-determinism on
    // the LoRA chain (5 dispatches per projection per layer = many
    // adds and the rounding mode of the GPU's fma may differ from
    // run to run within ~1e-3).
    try runner_b.forwardLogits(token_ids, logits_buf);
    const ce_after_load = cpu_train_decoder.softmaxCeLoss(logits_buf, target_ids, n_pos, vocab);
    const roundtrip_delta = @abs(ce_after_load - ce_at_save);
    if (roundtrip_delta > 1e-3) {
        std.debug.print(
            "LoRA checkpoint round-trip FAIL: ce_at_save={d:.6} ce_after_load={d:.6} delta={e:.3}\n",
            .{ ce_at_save, ce_after_load, roundtrip_delta },
        );
        return error.LoraCheckpointRoundTripFailed;
    }

    // Gate 2: trajectory continuity. Runner B trained M steps post-
    // load should land near Runner A's K+M endpoint. If Adam m/v /
    // step_t weren't restored, the M-step trajectory diverges visibly.
    i = 0;
    while (i < m_steps) : (i += 1) try runner_b.step(token_ids, target_ids);

    try runner_b.forwardLogits(token_ids, logits_buf);
    const ce_after_resume = cpu_train_decoder.softmaxCeLoss(logits_buf, target_ids, n_pos, vocab);
    if (!std.math.isFinite(ce_after_resume)) return error.CeAfterResumeNotFinite;

    const trajectory_delta = @abs(ce_after_resume - ce_continuous);
    if (trajectory_delta > 1e-2) {
        std.debug.print(
            "LoRA checkpoint trajectory FAIL: ce_continuous={d:.6} ce_after_resume={d:.6} delta={e:.3}\n",
            .{ ce_continuous, ce_after_resume, trajectory_delta },
        );
        return error.LoraCheckpointTrajectoryDivergent;
    }

    // Report the on-disk size — the marquee number for the LoRA
    // checkpoint story (kilobytes-ish, vs ~0.6 MiB on the toy
    // for the full `.vkpt` save and ~9 GiB on Qwen3-0.6B).
    const file = try std.fs.cwd().openFile(ckpt_path, .{ .mode = .read_only });
    defer file.close();
    const stat = try file.stat();

    std.debug.print(
        "PASS Runner LoRA checkpoint round-trip ({d}-layer toy stack qk_norm+RoPE; LoRA all_attn rank={d}; K={d}+M={d} steps; CE save={d:.6} load={d:.6} (Δ={e:.2}); resume {d:.6} vs continuous {d:.6} (Δ={e:.2}); .lvkpt {d:.1} KiB)\n",
        .{ cfg_static.n_layers, lora_rank, k_steps, m_steps, ce_at_save, ce_after_load, roundtrip_delta, ce_after_resume, ce_continuous, trajectory_delta, @as(f64, @floatFromInt(stat.size)) / 1024.0 },
    );
}

// ── chunk 8c-β-6c: sampled-text-shift validation ────────────────────
//
// Closes the "did this actually do anything observable" loop. Samples
// tokens from the model before fine-tuning, trains the same K=30 steps
// β-6a does on batch 0 of `tiny_facts.jsonl`, then samples again with
// the same prompt. Gate: post-fine-tune token sequence != pre-fine-tune
// token sequence (training visibly shifted argmax for ≥1 position).
//
// Sampling is greedy (argmax) — deterministic, no temperature/top-k. We
// use Runner.forwardLogits inside an autoregressive loop:
//   window = [prompt..., pad, pad, ...]   (length n_pos, right-padded)
//   gen_pos = prompt.len - 1               (where to read next-token logits)
//   loop n_gen times:
//     forwardLogits(window) → logits[n_pos × vocab]
//     pick argmax(logits[gen_pos])
//     extend in-window (gen_pos++) until full, then slide left
//
// Right-padding works because Qwen3 attention is causal — logits at
// position p depend only on tokens 0..p. Pads at p+1..n_pos-1 are
// invisible to logits[p].
//
// Probe prompt "The capital of France is" matches the start of batch 0
// of tiny_facts.jsonl, so single-batch overfit memorizes the training
// continuation (" Paris. Paris sits on the river Seine..."). The pre-
// fine-tune model probably also says "Paris" (it's a well-known fact in
// pretraining), but the *literal* continuation past 1-2 tokens diverges:
// the training data's exact phrasing is unique enough that overfit
// reproduces it verbatim while pretrained completions are generic.

pub fn runRealModelSamplingSmoke(allocator: std.mem.Allocator) !void {
    const model_id = "Qwen/Qwen3-0.6B";
    const jsonl_path = "data/train/tiny_facts.jsonl";
    const probe_prompt = "The capital of France is";
    const n_pos: u32 = 16;
    const n_gen: u32 = 20;
    const n_train: u32 = 30;
    const eos_id: u32 = 151_645;
    const lr: f32 = 1e-5;

    const dir_path = hf_cache.resolveModelArg(allocator, model_id) catch |err| switch (err) {
        error.HfModelNotInCache => {
            std.debug.print("SKIP runRealModelSamplingSmoke (Qwen3-0.6B not in HF cache)\n", .{});
            return;
        },
        else => return err,
    };
    defer allocator.free(dir_path);

    var cpu = try model_mod.Model.load(allocator, dir_path);
    defer cpu.deinit();

    var weights = try train_load_real.loadTrainWeights(allocator, &cpu, n_pos);
    defer weights.deinit();
    var cfg = weights.cfg;
    cfg.lr = lr;

    const tok_path = try std.fmt.allocPrint(allocator, "{s}/tokenizer.json", .{dir_path});
    defer allocator.free(tok_path);
    var tok = try tokenizer_mod.Tokenizer.loadFromFile(allocator, tok_path);
    defer tok.deinit();

    var ds = try train_dataset.buildFromJsonl(allocator, &tok, jsonl_path, n_pos, eos_id);
    defer ds.deinit();
    if (ds.numBatches() == 0) return error.DatasetTooShort;

    const prompt_ids = try tok.encode(allocator, probe_prompt);
    defer allocator.free(prompt_ids);
    if (prompt_ids.len == 0 or prompt_ids.len >= n_pos) return error.PromptShape;

    const vocab: usize = cfg.vocab_size;

    var ctx = try vk.Context.init(allocator);
    defer ctx.deinit();

    var runner = try train_transformer.Runner.init(allocator, &ctx, cfg, weights.view());
    defer runner.deinit();

    // ── Sample BEFORE fine-tune.
    const sample_pre = try train_sampling.greedyDecode(allocator, &runner, prompt_ids, n_gen, n_pos, vocab, eos_id);
    defer allocator.free(sample_pre);

    // ── Train K steps on batch 0.
    const input_ids = try allocator.alloc(u32, n_pos);
    defer allocator.free(input_ids);
    const target_ids = try allocator.alloc(u32, n_pos);
    defer allocator.free(target_ids);
    try ds.batch(0, input_ids, target_ids);

    var i: u32 = 0;
    while (i < n_train) : (i += 1) {
        try runner.step(input_ids, target_ids);
    }

    // ── Sample AFTER fine-tune.
    const sample_post = try train_sampling.greedyDecode(allocator, &runner, prompt_ids, n_gen, n_pos, vocab, eos_id);
    defer allocator.free(sample_post);

    // ── Decode both to text.
    const pre_text = try train_sampling.decodeIdsToText(allocator, &tok, sample_pre);
    defer allocator.free(pre_text);
    const post_text = try train_sampling.decodeIdsToText(allocator, &tok, sample_post);
    defer allocator.free(post_text);

    std.debug.print("    Probe        : \"{s}\"\n", .{probe_prompt});
    std.debug.print("    Pre-fine-tune: \"{s}\"\n", .{pre_text});
    std.debug.print("    Post-train   : \"{s}\"\n", .{post_text});

    // ── Gate: at least one sampled token differs.
    if (sample_pre.len != sample_post.len) return error.SampleLenMismatch;
    const sampled_offset = prompt_ids.len; // first generated token
    var differed: usize = 0;
    for (sample_pre[sampled_offset..], sample_post[sampled_offset..]) |a, b| {
        if (a != b) differed += 1;
    }
    if (differed == 0) {
        std.debug.print("Sampling smoke: pre and post identical — training did not shift argmax\n", .{});
        return error.SampleDidNotShift;
    }

    std.debug.print(
        "PASS real Qwen3-0.6B sampled-text-shift (prompt=\"{s}\" n_train={d} lr={e}; {d}/{d} generated tokens shifted)\n",
        .{ probe_prompt, n_train, lr, differed, n_gen },
    );
}

// ── chunk 8c-β-6a: multi-step training loop ──────────────────────────
//
// Extends β-5 (one Adam step on real Qwen3-0.6B) to N steps on the same
// batch, validating that the training loop stays healthy past step 1:
//   - Adam state (m, v, t) updates correctly across iterations
//   - no NaN bloom from gradient buildup or overflow
//   - CE drops sharply on a single-batch overfit gate
//
// Same setup as β-5: load Qwen3-0.6B, build the dataset, take batch 0.
// Then loop runner.step n_steps times on that one batch, measure CE
// before and after via forwardLogits. Mid-training CE checks are
// intentionally omitted — the per-call cost (~150 ms forward + 9 MB
// readback) doubles the wallclock for limited diagnostic value when the
// gate is "did final CE drop" not "did each step monotonically drop".
//
// Gate: ce_final < ce_init * 0.1 (90% drop minimum; β-5 already hits 54%
// in one step, so 30 single-batch steps should easily clear this) and
// CE is finite at every check.

pub fn runRealModelMultiStepSmoke(allocator: std.mem.Allocator) !void {
    const model_id = "Qwen/Qwen3-0.6B";
    const jsonl_path = "data/train/tiny_facts.jsonl";
    const n_pos: u32 = 16;
    const eos_id: u32 = 151_645;
    const lr: f32 = 1e-5;
    const n_steps: u32 = 30;

    const dir_path = hf_cache.resolveModelArg(allocator, model_id) catch |err| switch (err) {
        error.HfModelNotInCache => {
            std.debug.print("SKIP runRealModelMultiStepSmoke (Qwen3-0.6B not in HF cache)\n", .{});
            return;
        },
        else => return err,
    };
    defer allocator.free(dir_path);

    var cpu = try model_mod.Model.load(allocator, dir_path);
    defer cpu.deinit();

    var weights = try train_load_real.loadTrainWeights(allocator, &cpu, n_pos);
    defer weights.deinit();
    var cfg = weights.cfg;
    cfg.lr = lr;

    const tok_path = try std.fmt.allocPrint(allocator, "{s}/tokenizer.json", .{dir_path});
    defer allocator.free(tok_path);
    var tok = try tokenizer_mod.Tokenizer.loadFromFile(allocator, tok_path);
    defer tok.deinit();

    var ds = try train_dataset.buildFromJsonl(allocator, &tok, jsonl_path, n_pos, eos_id);
    defer ds.deinit();
    if (ds.numBatches() == 0) return error.DatasetTooShort;

    const input_ids = try allocator.alloc(u32, n_pos);
    defer allocator.free(input_ids);
    const target_ids = try allocator.alloc(u32, n_pos);
    defer allocator.free(target_ids);
    try ds.batch(0, input_ids, target_ids);

    const vocab: usize = cfg.vocab_size;

    var ctx = try vk.Context.init(allocator);
    defer ctx.deinit();

    var runner = try train_transformer.Runner.init(allocator, &ctx, cfg, weights.view());
    defer runner.deinit();

    const logits = try allocator.alloc(f32, @as(usize, n_pos) * vocab);
    defer allocator.free(logits);

    // ── CE before training.
    try runner.forwardLogits(input_ids, logits);
    const ce_init = computeMeanCe(logits, target_ids, n_pos, vocab);
    if (!std.math.isFinite(ce_init)) {
        std.debug.print("Multi-step smoke: CE_init not finite ({d})\n", .{ce_init});
        return error.CeInitNotFinite;
    }

    // ── N Adam steps on the same batch.
    var step_ns_total: i128 = 0;
    for (0..n_steps) |_| {
        const t_start = std.time.nanoTimestamp();
        try runner.step(input_ids, target_ids);
        const t_end = std.time.nanoTimestamp();
        step_ns_total += t_end - t_start;
    }
    const mean_step_ns = @divTrunc(step_ns_total, @as(i128, n_steps));
    const mean_step_ms: f64 = @as(f64, @floatFromInt(mean_step_ns)) / 1.0e6;

    // ── CE after training.
    try runner.forwardLogits(input_ids, logits);
    const ce_final = computeMeanCe(logits, target_ids, n_pos, vocab);
    if (!std.math.isFinite(ce_final)) {
        std.debug.print(
            "Multi-step smoke: CE_final not finite ({d}) after CE_init={d:.6}\n",
            .{ ce_final, ce_init },
        );
        return error.CeFinalNotFinite;
    }
    if (ce_final >= ce_init * 0.1) {
        std.debug.print(
            "Multi-step smoke: CE did not converge enough (init={d:.6} final={d:.6}, ratio={d:.4} expected < 0.1) at lr={e} over {d} steps\n",
            .{ ce_init, ce_final, ce_final / ce_init, lr, n_steps },
        );
        return error.CeDidNotConverge;
    }

    const drop_pct: f64 = 100.0 * @as(f64, ce_init - ce_final) / @as(f64, ce_init);
    std.debug.print(
        "PASS real Qwen3-0.6B multi-step train (n_steps={d} n_pos={d} lr={e}; CE {d:.6} → {d:.6}, drop {d:.2}%; {d:.1} ms/step)\n",
        .{ n_steps, n_pos, lr, ce_init, ce_final, drop_pct, mean_step_ms },
    );
}

/// Mean per-position cross-entropy: −log p(target_ids[p] | logits[p,·])
/// averaged over `n_pos`. Uses fp64 accumulation for the per-position
/// log-Z so n_pos × vocab × magnitude doesn't lose precision.
fn computeMeanCe(logits: []const f32, target_ids: []const u32, n_pos: u32, vocab: usize) f32 {
    var ce_sum: f64 = 0;
    for (0..n_pos) |p| {
        const off = p * vocab;
        var m: f32 = -std.math.inf(f32);
        for (0..vocab) |o| m = @max(m, logits[off + o]);
        var sum_e: f64 = 0;
        for (0..vocab) |o| sum_e += @exp(@as(f64, logits[off + o]) - @as(f64, m));
        const log_z: f64 = @as(f64, m) + @log(sum_e);
        ce_sum += log_z - @as(f64, logits[off + target_ids[p]]);
    }
    return @floatCast(ce_sum / @as(f64, @floatFromInt(n_pos)));
}

// ── chunk 8b stage A: gpu backward chain parity vs cpu oracle ────────
//
// Replays the same toy decoder layer as `runDecoderFineTuneCpuSmoke` for
// one backward pass, but on the GPU. Forward + d_y (MSE-loss-grad) stay
// on CPU; activations + weights + d_y are uploaded as-is and the GPU
// backward chain (linear-dx/-dW batched, RMSNorm backward, SDPA
// backward, ReLU backward, residual add, softmax backward) recomputes
// the eight weight gradients. Each gradient buffer is then compared
// against the CPU oracle's matching slice.
//
// The full chain composes 23 dispatches on a single recorder; the
// recorder injects compute-shader memory barriers between consecutive
// dispatches. dW for the two RMSNorm gains comes back as per-row
// partials and is summed host-side, mirroring `runGpuRmsnormBackwardSmoke`.
//
// CPU oracle uses fp64 accumulators in linear / softmax / attention
// reductions; GPU is fp32 throughout. Tolerance is set against the
// largest absolute value in each slice with a 1e-3 relative cap and a
// small absolute floor to skip near-zero entries.

pub fn runDecoderBackwardGpuParitySmoke(allocator: std.mem.Allocator) !void {
    _ = allocator;
    // chunk-8b stage A (single-layer GPU backward parity vs CPU oracle).
    // Deprecated by the SwiGLU swap in β-3a-2: the chunk-8c-α-2 stack
    // parity smoke covers the same dispatch chain at depth N=2, plus
    // β-3a-1's primitive parity smoke covers the SwiGLU bw shader.
    // Body removed rather than rewritten; reinstate if a per-layer
    // surface needs an independent gate.
    std.debug.print("SKIP runDecoderBackwardGpuParitySmoke (deprecated by 8c-α-2 + β-3a-1)\n", .{});
}

// ── chunk 8b stage B: full-GPU toy decoder fine-tune ────────────────
//
// Runs the same toy decoder as runDecoderFineTuneCpuSmoke / Stage A, but
// every operation lives on the GPU: forward, MSE loss-grad, backward,
// and Adam. The training loop is self-sustaining across 100 steps and
// the assertion is the same as 8a — `final_loss / initial_loss ≤ 1e-2`.
//
// This is the first transformer-trainer pipeline to live entirely on
// the GPU. It's intentionally inlined into the smoke (not a Runner) —
// the eventual TransformerTrainerRunner shape decision lives outside
// chunk 8b's scope; what this proves is that the kernel inventory is
// complete and that one decoder layer composes correctly under
// repeated optimizer steps.
//
// Per-step dispatch count: 14 forward + 1 loss-grad + 23 backward +
// 8 Adam = 46 dispatches. The recorder is reset and re-recorded
// every step. The same kernels are reused across all 100 steps —
// only the buffer contents change.

pub fn runDecoderTrainGpuSmoke(allocator: std.mem.Allocator) !void {
    _ = allocator;
    // chunk-8b stage B (single-layer full GPU MSE training).
    // Deprecated by β-3a-2: the chunk-8c-α-3 stack training smoke
    // exercises the same Runner forward+backward+Adam loop with
    // softmax-CE + multi-layer + lm_head, plus β-2's real-shape
    // envelope covers Qwen-class scale.
    std.debug.print("SKIP runDecoderTrainGpuSmoke (deprecated by 8c-α-3 + β-2)\n", .{});
}
