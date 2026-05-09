//! Pure CPU oracle smokes invoked from main.zig's "run all" fallthrough:
//! matmul, RoPE identity, softmax, MLP forward+backward+SGD, multi-layer
//! MlpN training, and RMSNorm backward parity. Extracted from main.zig.

const std = @import("std");
const cpu_math = @import("../cpu/math.zig");
const cpu_train = @import("../cpu/train.zig");
const cpu_train_transformer = @import("../cpu/train_transformer.zig");
const cpu_lora = @import("../cpu/lora.zig");
const cpu_flash_attn = @import("../cpu/flash_attn.zig");
const cpu_mtp = @import("../cpu/mtp.zig");
const dtype = @import("../dtype.zig");
const lora_merge = @import("../commands/lora_merge.zig");
const safetensors = @import("../safetensors.zig");
const model_mod = @import("../model.zig");
const hf_cache = @import("../hf_cache.zig");

// ── matmul smoke: synthetic A·B^T, hand-checked oracle ──────────────

pub fn runMatmulSmoke(allocator: std.mem.Allocator) !void {
    _ = allocator;
    // A = [[1, 2, 3], [4, 5, 6]]                M=2, K=3
    // B = [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1]]   N=4, K=3
    // A · Bᵀ = [[1, 2, 3, 6], [4, 5, 6, 15]]    M=2, N=4
    const a = [_]f32{ 1, 2, 3, 4, 5, 6 };
    const b_data = [_]f32{ 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1 };
    const want = [_]f32{ 1, 2, 3, 6, 4, 5, 6, 15 };

    const b_shape = [_]usize{ 4, 3 };
    const b: safetensors.Tensor = .{
        .dtype = .f32,
        .shape = &b_shape,
        .bytes = std.mem.sliceAsBytes(b_data[0..]),
    };

    var out: [8]f32 = undefined;
    try cpu_math.matmul_nt(&out, &a, b, 2, 4, 3);

    for (out, want, 0..) |got, w, i| {
        if (got != w) {
            std.debug.print("matmul MISMATCH at {d}: got {d}, expected {d}\n", .{ i, got, w });
            return error.ParityFailed;
        }
    }
    std.debug.print("PASS matmul_nt synthetic (2×3 · (4×3)ᵀ → 2×4)\n", .{});
}

// ── RoPE identity smoke: pos=0 must be a no-op ──────────────────────

pub fn runRopeIdentitySmoke(allocator: std.mem.Allocator) !void {
    // Synthetic Q-shaped input. n_heads=8, head_dim=64 — stays small.
    const n_heads: usize = 8;
    const head_dim: usize = 64;
    const total = n_heads * head_dim;
    const in_v = try allocator.alloc(f32, total);
    defer allocator.free(in_v);
    const out_v = try allocator.alloc(f32, total);
    defer allocator.free(out_v);
    for (in_v, 0..) |*x, i| x.* = @as(f32, @floatFromInt(i)) * 0.001 - 0.5;

    try cpu_math.applyRope(out_v, in_v, n_heads, head_dim, 0, 10000.0);

    for (in_v, out_v, 0..) |a, b, i| {
        if (a != b) {
            std.debug.print("RoPE pos=0 NOT identity at i={d}: in={d} out={d}\n", .{ i, a, b });
            return error.ParityFailed;
        }
    }
    std.debug.print("PASS RoPE pos=0 identity ({d} heads × {d} dim)\n", .{ n_heads, head_dim });
}

// ── softmax smoke: stable form on a hand-checked input ─────────────

pub fn runSoftmaxSmoke(allocator: std.mem.Allocator) !void {
    _ = allocator;
    // Inputs include a large positive value; the naive exp(x) form
    // would overflow but the stable variant should produce normal
    // probabilities. Reference computed with the same shifted form.
    var x = [_]f32{ 1.0, 2.0, 3.0, 100.0 };
    cpu_math.softmax(&x);
    var sum: f32 = 0;
    for (x) |v| sum += v;
    if (@abs(sum - 1.0) > 1e-6) {
        std.debug.print("softmax sum {d} != 1\n", .{sum});
        return error.ParityFailed;
    }
    // The big value should dominate (≈ 1.0); the others should be tiny.
    if (x[3] < 0.99 or x[0] > 1e-30 or x[1] > 1e-30 or x[2] > 1e-30) {
        std.debug.print("softmax distribution wrong: {any}\n", .{x});
        return error.ParityFailed;
    }
    std.debug.print("PASS softmax stable form (handles +100 without overflow)\n", .{});
}

// ── tiny-MLP training smoke: forward + backward + SGD converge ──────
//
// Chunk 1 of training-v0. Drives the CPU oracle in src/cpu/train.zig
// hard enough to catch a sign error or off-by-one anywhere in the
// derivative chain: a 2-layer MLP fits a single (input, target) pair
// and we assert loss drops by ≥ 100×. The numbers are tight on
// purpose — the only way for the loss curve to plateau here is if a
// gradient is wrong.
//
// We also do an explicit numerical-gradient check on one weight in
// each layer; that's the surest test that backward(...) is consistent
// with forward(...), independent of how well SGD converges.

pub fn runTrainCpuSmoke(allocator: std.mem.Allocator) !void {
    // Tiny on purpose — total weights = 4·8 + 8 + 4·8 + 4 = 76. Big
    // enough to need a hidden layer with ReLU active on some neurons,
    // small enough to land convergence in a few hundred steps.
    const dim_in: usize = 4;
    const dim_h: usize = 8;
    const dim_out: usize = 4;

    var mlp = try cpu_train.Mlp.init(allocator, dim_in, dim_h, dim_out, 0.3, 0xC0FFEE);
    defer mlp.deinit(allocator);

    var grads = try cpu_train.Grads.init(allocator, &mlp);
    defer grads.deinit(allocator);

    const x = [_]f32{ 1.0, 0.5, -0.3, 0.2 };
    const target = [_]f32{ 1.0, 0.0, 0.0, 0.0 };

    var h_pre: [dim_h]f32 = undefined;
    var h: [dim_h]f32 = undefined;
    var y: [dim_out]f32 = undefined;
    var act: cpu_train.Activations = .{
        .x = &x,
        .h_pre = &h_pre,
        .h = &h,
        .y = &y,
    };

    // ── Loss-decrease check ────────────────────────────────────────
    cpu_train.forward(&mlp, &act);
    const loss_initial = cpu_train.mseLoss(act.y, &target);

    const lr: f32 = 0.05;
    const n_steps: u32 = 400;
    var loss_final: f32 = 0;
    for (0..n_steps) |_| {
        loss_final = try cpu_train.trainStep(allocator, &mlp, &act, &grads, &target, lr);
    }

    if (!(loss_final < loss_initial / 100.0)) {
        std.debug.print(
            "training did not converge: loss[0] = {d:.6}, loss[{d}] = {d:.6}\n",
            .{ loss_initial, n_steps, loss_final },
        );
        return error.ParityFailed;
    }

    // ── Numerical-gradient parity check ────────────────────────────
    // For a fresh MLP and one (x, target), pick a single weight in W2
    // and a single weight in W1, perturb it by eps in each direction,
    // diff the losses, and compare to the analytic gradient. If the
    // chain rule in backward() is right these match within fp32
    // tolerance. The check is the truest kind — it doesn't trust SGD,
    // doesn't trust the loss being convex, just the calculus.
    var probe_mlp = try cpu_train.Mlp.init(allocator, dim_in, dim_h, dim_out, 0.3, 0xBEEF1234);
    defer probe_mlp.deinit(allocator);
    var probe_grads = try cpu_train.Grads.init(allocator, &probe_mlp);
    defer probe_grads.deinit(allocator);

    var probe_h_pre: [dim_h]f32 = undefined;
    var probe_h: [dim_h]f32 = undefined;
    var probe_y: [dim_out]f32 = undefined;
    var probe_act: cpu_train.Activations = .{
        .x = &x,
        .h_pre = &probe_h_pre,
        .h = &probe_h,
        .y = &probe_y,
    };

    // Analytic gradients at this point.
    cpu_train.forward(&probe_mlp, &probe_act);
    var dL_dy: [dim_out]f32 = undefined;
    cpu_train.mseLossGrad(&dL_dy, probe_act.y, &target);
    try cpu_train.backward(allocator, &probe_mlp, &probe_act, &dL_dy, &probe_grads);

    const eps: f32 = 1e-3;
    // Test one element from each parameter buffer.
    const probe_targets = [_]struct { name: []const u8, buf: []f32, grad: []const f32, idx: usize }{
        .{ .name = "W2[1, 3]", .buf = probe_mlp.w2, .grad = probe_grads.dw2, .idx = 1 * dim_h + 3 },
        .{ .name = "W1[2, 0]", .buf = probe_mlp.w1, .grad = probe_grads.dw1, .idx = 2 * dim_in + 0 },
        .{ .name = "b2[2]", .buf = probe_mlp.b2, .grad = probe_grads.db2, .idx = 2 },
        .{ .name = "b1[5]", .buf = probe_mlp.b1, .grad = probe_grads.db1, .idx = 5 },
    };
    for (probe_targets) |t| {
        const orig = t.buf[t.idx];
        t.buf[t.idx] = orig + eps;
        cpu_train.forward(&probe_mlp, &probe_act);
        const loss_plus = cpu_train.mseLoss(probe_act.y, &target);
        t.buf[t.idx] = orig - eps;
        cpu_train.forward(&probe_mlp, &probe_act);
        const loss_minus = cpu_train.mseLoss(probe_act.y, &target);
        t.buf[t.idx] = orig;

        const numeric = (loss_plus - loss_minus) / (2.0 * eps);
        const analytic = t.grad[t.idx];
        const denom = @max(@abs(numeric), @abs(analytic));
        const rel_err = if (denom > 0) @abs(numeric - analytic) / denom else @abs(numeric - analytic);
        if (rel_err > 1e-2) {
            std.debug.print(
                "grad mismatch on {s}: analytic = {d:.6}, numeric = {d:.6}, rel_err = {d:.4}\n",
                .{ t.name, analytic, numeric, rel_err },
            );
            return error.ParityFailed;
        }
    }

    std.debug.print(
        "PASS train MLP CPU oracle: loss {d:.6} → {d:.6} over {d} SGD steps; numeric grad parity ≤ 1%\n",
        .{ loss_initial, loss_final, n_steps },
    );
}

// ── Multi-layer MLP CPU oracle smoke ──────────────────────────────
//
// Tier-1 of the post-v0 training arc: generalise the 2-layer Mlp to
// any depth. This smoke exercises three things on `MlpN`:
//
//   1. n=2 numerical equivalence with `Mlp` — same dims, same seed →
//      forward output bit-identical (to fp32 rounding noise). This is
//      the strongest cross-check that the new code path doesn't drift
//      from the proven 2-layer reference.
//
//   2. n=3 numeric-gradient parity — perturb a single weight in each
//      of the three layers, diff the loss; the central-difference
//      slope must agree with the analytic gradient to ≤ 1%. This
//      proves the chain rule is right at depth.
//
//   3. n=3 convergence — 500 SGD steps must drop the loss by ≥ 100×
//      on the same toy (x, target) pair the 2-layer test uses.

pub fn runTrainCpuMultiLayerSmoke(allocator: std.mem.Allocator) !void {
    // ── (1) n=2 equivalence with `Mlp` ────────────────────────────
    {
        const dim_in: usize = 4;
        const dim_h: usize = 8;
        const dim_out: usize = 4;
        const seed: u64 = 0xC0FFEE;
        const init_scale: f32 = 0.3;

        var ref = try cpu_train.Mlp.init(allocator, dim_in, dim_h, dim_out, init_scale, seed);
        defer ref.deinit(allocator);

        var gen = try cpu_train.MlpN.init(allocator, &.{ dim_in, dim_h, dim_out }, init_scale, seed);
        defer gen.deinit(allocator);

        // Same RNG order: weights[0] should equal w1, weights[1] equal w2.
        for (ref.w1, gen.weights[0]) |a, b| {
            if (a != b) {
                std.debug.print("MlpN W1 != Mlp W1\n", .{});
                return error.ParityFailed;
            }
        }
        for (ref.w2, gen.weights[1]) |a, b| {
            if (a != b) {
                std.debug.print("MlpN W2 != Mlp W2\n", .{});
                return error.ParityFailed;
            }
        }

        const x = [_]f32{ 1.0, 0.5, -0.3, 0.2 };

        var ref_h_pre: [dim_h]f32 = undefined;
        var ref_h: [dim_h]f32 = undefined;
        var ref_y: [dim_out]f32 = undefined;
        var ref_act: cpu_train.Activations = .{ .x = &x, .h_pre = &ref_h_pre, .h = &ref_h, .y = &ref_y };
        cpu_train.forward(&ref, &ref_act);

        var gen_act = try cpu_train.ActivationsN.init(allocator, &gen);
        defer gen_act.deinit(allocator);
        gen_act.x = &x;
        cpu_train.forwardN(&gen, &gen_act);

        for (ref_y, gen_act.y()) |a, b| {
            if (a != b) {
                std.debug.print("MlpN forward != Mlp forward (n=2)\n", .{});
                return error.ParityFailed;
            }
        }
    }

    // ── (2)/(3) n=3 numeric grad + convergence ────────────────────
    const dim_in: usize = 4;
    const dim_h1: usize = 8;
    const dim_h2: usize = 6;
    const dim_out: usize = 4;
    const layer_dims = [_]usize{ dim_in, dim_h1, dim_h2, dim_out };

    var mlp = try cpu_train.MlpN.init(allocator, &layer_dims, 0.3, 0xCAFE2026);
    defer mlp.deinit(allocator);

    var grads = try cpu_train.GradsN.init(allocator, &mlp);
    defer grads.deinit(allocator);

    const x = [_]f32{ 1.0, 0.5, -0.3, 0.2 };
    const target = [_]f32{ 1.0, 0.0, 0.0, 0.0 };

    var act = try cpu_train.ActivationsN.init(allocator, &mlp);
    defer act.deinit(allocator);
    act.x = &x;

    cpu_train.forwardN(&mlp, &act);
    const loss_initial = cpu_train.mseLoss(act.y(), &target);

    // Numeric-grad probe BEFORE training (so we measure analytic vs
    // numeric on the un-trained network, where weights haven't drifted
    // toward zero gradient).
    var dL_dy: [dim_out]f32 = undefined;
    cpu_train.mseLossGrad(&dL_dy, act.y(), &target);
    try cpu_train.backwardN(allocator, &mlp, &act, &dL_dy, &grads);

    const eps: f32 = 1e-3;
    // One probe per layer: layer 0 hits W[0], layer 1 hits W[1] (and a bias),
    // layer 2 hits W[2] (and a bias on the output).
    const probes = [_]struct { name: []const u8, layer: usize, is_bias: bool, idx: usize }{
        .{ .name = "W[0][3, 0]", .layer = 0, .is_bias = false, .idx = 3 * dim_in + 0 },
        .{ .name = "W[1][2, 5]", .layer = 1, .is_bias = false, .idx = 2 * dim_h1 + 5 },
        .{ .name = "W[2][1, 4]", .layer = 2, .is_bias = false, .idx = 1 * dim_h2 + 4 },
        .{ .name = "b[1][3]", .layer = 1, .is_bias = true, .idx = 3 },
        .{ .name = "b[2][2]", .layer = 2, .is_bias = true, .idx = 2 },
    };
    for (probes) |p| {
        const buf = if (p.is_bias) mlp.biases[p.layer] else mlp.weights[p.layer];
        const grad = if (p.is_bias) grads.db[p.layer] else grads.dw[p.layer];
        const orig = buf[p.idx];

        buf[p.idx] = orig + eps;
        cpu_train.forwardN(&mlp, &act);
        const loss_plus = cpu_train.mseLoss(act.y(), &target);
        buf[p.idx] = orig - eps;
        cpu_train.forwardN(&mlp, &act);
        const loss_minus = cpu_train.mseLoss(act.y(), &target);
        buf[p.idx] = orig;

        const numeric = (loss_plus - loss_minus) / (2.0 * eps);
        const analytic = grad[p.idx];
        const denom = @max(@abs(numeric), @abs(analytic));
        const rel_err = if (denom > 0) @abs(numeric - analytic) / denom else @abs(numeric - analytic);
        if (rel_err > 1e-2) {
            std.debug.print(
                "MlpN grad mismatch on {s}: analytic = {d:.6}, numeric = {d:.6}, rel_err = {d:.4}\n",
                .{ p.name, analytic, numeric, rel_err },
            );
            return error.ParityFailed;
        }
    }

    // Convergence run.
    const lr: f32 = 0.05;
    const n_steps: u32 = 500;
    var loss_final: f32 = 0;
    for (0..n_steps) |_| {
        loss_final = try cpu_train.trainStepN(allocator, &mlp, &act, &grads, &target, lr);
    }
    if (!(loss_final < loss_initial / 100.0)) {
        std.debug.print(
            "MlpN n=3 did not converge: loss[0] = {d:.6}, loss[{d}] = {d:.6}\n",
            .{ loss_initial, n_steps, loss_final },
        );
        return error.ParityFailed;
    }

    std.debug.print(
        "PASS train MLP CPU oracle (multi-layer): n=2 equivalent to Mlp; n=3 loss {d:.6} → {d:.6} over {d} SGD steps; numeric grad parity ≤ 1% across all 3 layers\n",
        .{ loss_initial, loss_final, n_steps },
    );
}

// ── RMSNorm backward CPU oracle smoke ──────────────────────────────
//
// Tier-2 chunk 1 of the post-v0 training arc — first transformer
// primitive backward. RMSNorm forward is a row-wise scaling by
// `γ · rms_inv`; the backward involves a cross-row scalar reduction
// that the analytic derivation has to nail exactly.
//
// Three claims, each verified to ≤ 1% rel-err vs central-difference
// numeric gradient:
//   - dL/dx in the plain case (gemma_quirk=false)
//   - dL/dx with the gemma_quirk offset (γ = w + 1)
//   - dL/dw — same formula in both cases since (w + 1) and w have
//     identical derivatives w.r.t. w
//
// Probes one element from each gradient buffer; that's enough to
// catch any sign/shape mistake in the chain rule. The whole-buffer
// numeric grad is too noisy at fp32 to use as a parity oracle —
// per-element probes with a moderate eps are the right shape.

pub fn runRmsNormBackwardCpuSmoke(allocator: std.mem.Allocator) !void {
    _ = allocator; // reserved for future heap fallbacks at larger dim
    const dim: usize = 16;
    const eps: f32 = 1e-6;

    var x: [dim]f32 = undefined;
    var w: [dim]f32 = undefined;
    var dy: [dim]f32 = undefined;
    var prng = std.Random.DefaultPrng.init(0xCAFE5EED);
    const rng = prng.random();
    for (&x) |*v| v.* = (rng.float(f32) * 2.0 - 1.0) * 1.5;
    for (&w) |*v| v.* = 0.5 + rng.float(f32) * 0.5;
    for (&dy) |*v| v.* = (rng.float(f32) * 2.0 - 1.0);

    var dx: [dim]f32 = undefined;
    var dw: [dim]f32 = undefined;
    var y_buf: [dim]f32 = undefined;
    var probe_y: [dim]f32 = undefined;

    // Cases: (gemma_quirk = false), (gemma_quirk = true).
    const cases = [_]bool{ false, true };
    var max_rel_err_overall: f32 = 0;
    for (cases) |gq| {
        // Analytic gradients.
        @memset(&dw, 0);
        cpu_train_transformer.rmsNormBackwardRow(&dy, &x, &w, eps, gq, &dx, &dw);

        // Loss = Σ dy_i · y_i (so dL/dy_i = dy_i exactly).
        // For numeric grad we re-evaluate the loss with x or w perturbed.
        const eps_h: f32 = 1e-3;

        // Probe several positions in dx and dw.
        const probes_x = [_]usize{ 0, 3, 7, 12 };
        const probes_w = [_]usize{ 1, 5, 9, 14 };

        for (probes_x) |i| {
            const orig = x[i];
            x[i] = orig + eps_h;
            _ = cpu_train_transformer.rmsNormForwardRow(&x, &w, eps, gq, &probe_y);
            var l_plus: f32 = 0;
            for (dy, probe_y) |d, yi| l_plus += d * yi;
            x[i] = orig - eps_h;
            _ = cpu_train_transformer.rmsNormForwardRow(&x, &w, eps, gq, &probe_y);
            var l_minus: f32 = 0;
            for (dy, probe_y) |d, yi| l_minus += d * yi;
            x[i] = orig;

            const numeric = (l_plus - l_minus) / (2.0 * eps_h);
            const analytic = dx[i];
            const denom = @max(@abs(numeric), @abs(analytic));
            const rel_err = if (denom > 0) @abs(numeric - analytic) / denom else @abs(numeric - analytic);
            if (rel_err > 1e-2) {
                std.debug.print(
                    "rmsnorm dx[{d}] (gq={}): analytic={d:.6} numeric={d:.6} rel_err={d:.4}\n",
                    .{ i, gq, analytic, numeric, rel_err },
                );
                return error.ParityFailed;
            }
            if (rel_err > max_rel_err_overall) max_rel_err_overall = rel_err;
        }

        for (probes_w) |i| {
            const orig = w[i];
            w[i] = orig + eps_h;
            _ = cpu_train_transformer.rmsNormForwardRow(&x, &w, eps, gq, &probe_y);
            var l_plus: f32 = 0;
            for (dy, probe_y) |d, yi| l_plus += d * yi;
            w[i] = orig - eps_h;
            _ = cpu_train_transformer.rmsNormForwardRow(&x, &w, eps, gq, &probe_y);
            var l_minus: f32 = 0;
            for (dy, probe_y) |d, yi| l_minus += d * yi;
            w[i] = orig;

            const numeric = (l_plus - l_minus) / (2.0 * eps_h);
            const analytic = dw[i];
            const denom = @max(@abs(numeric), @abs(analytic));
            const rel_err = if (denom > 0) @abs(numeric - analytic) / denom else @abs(numeric - analytic);
            if (rel_err > 1e-2) {
                std.debug.print(
                    "rmsnorm dw[{d}] (gq={}): analytic={d:.6} numeric={d:.6} rel_err={d:.4}\n",
                    .{ i, gq, analytic, numeric, rel_err },
                );
                return error.ParityFailed;
            }
            if (rel_err > max_rel_err_overall) max_rel_err_overall = rel_err;
        }
    }

    // Multi-row sanity: dw must accumulate across rows.
    const n_rows: usize = 3;
    var x_multi: [n_rows * dim]f32 = undefined;
    var dy_multi: [n_rows * dim]f32 = undefined;
    var dx_multi: [n_rows * dim]f32 = undefined;
    var dw_multi: [dim]f32 = undefined;
    for (&x_multi) |*v| v.* = (rng.float(f32) * 2.0 - 1.0);
    for (&dy_multi) |*v| v.* = (rng.float(f32) * 2.0 - 1.0);

    @memset(&dw_multi, 0);
    cpu_train_transformer.rmsNormBackward(&dy_multi, &x_multi, &w, eps, false, n_rows, &dx_multi, &dw_multi);

    // Compare against summing per-row backward calls explicitly.
    var dw_ref: [dim]f32 = undefined;
    @memset(&dw_ref, 0);
    var dx_ref_row: [dim]f32 = undefined;
    for (0..n_rows) |r| {
        const off = r * dim;
        cpu_train_transformer.rmsNormBackwardRow(
            dy_multi[off .. off + dim],
            x_multi[off .. off + dim],
            &w,
            eps,
            false,
            &dx_ref_row,
            &dw_ref,
        );
        for (dx_multi[off .. off + dim], dx_ref_row) |a, b| {
            if (a != b) {
                std.debug.print("multi-row dx mismatch at row {d}\n", .{r});
                return error.ParityFailed;
            }
        }
    }
    for (dw_multi, dw_ref) |a, b| {
        if (a != b) {
            std.debug.print("multi-row dw accumulation off\n", .{});
            return error.ParityFailed;
        }
    }

    // Suppress unused-variable warning for y_buf.
    _ = &y_buf;

    std.debug.print(
        "PASS rmsnorm backward CPU oracle (numeric-grad parity ≤ {e} on dx + dw, gemma_quirk on/off; multi-row dw accum bit-exact)\n",
        .{max_rel_err_overall},
    );
}

// ── lora-merge math: pre-fold W' = W + (α/r)·B·A vs explicit LoRA ────
//
// The chat-path `--lora-ckpt` route folds the LoRA delta into the base
// weight at load time, then runs the unmodified inference matmul. That
// only works if `forward(merged_W) ≡ forward_lora(W, A, B)` to the
// kernel-noise floor. This smoke gates the math identity — both at
// fp32 (exact algebra modulo reduction order) and after the bf16
// round-trip the chat path actually does on `bf16_matmul` precision.
pub fn runLoraMergeMathSmoke(allocator: std.mem.Allocator) !void {
    const M: usize = 4;
    const N: usize = 8;
    const K: usize = 16;
    const r: usize = 4;
    const alpha: f32 = 8.0;
    const aor: f32 = alpha / @as(f32, @floatFromInt(r));

    var rng = std.Random.DefaultPrng.init(0xCAFE_F00D_BABE_C0DE);
    const rand = rng.random();

    const x = try allocator.alloc(f32, M * K);
    defer allocator.free(x);
    const w_base = try allocator.alloc(f32, N * K);
    defer allocator.free(w_base);
    const a_lora = try allocator.alloc(f32, r * K);
    defer allocator.free(a_lora);
    const b_lora = try allocator.alloc(f32, N * r);
    defer allocator.free(b_lora);
    for (x) |*v| v.* = rand.floatNorm(f32) * 0.5;
    for (w_base) |*v| v.* = rand.floatNorm(f32) * 0.1;
    for (a_lora) |*v| v.* = rand.floatNorm(f32) * 0.5;
    for (b_lora) |*v| v.* = rand.floatNorm(f32) * 0.5;

    // ── Path A: explicit LoRA forward (the trainer's math, the truth).
    const y_oracle = try allocator.alloc(f32, M * N);
    defer allocator.free(y_oracle);
    const intermediate = try allocator.alloc(f32, M * r);
    defer allocator.free(intermediate);
    cpu_lora.loraForward(x, w_base, a_lora, b_lora, M, N, K, r, aor, y_oracle, intermediate);

    // ── Path B: merge the delta into W, then plain matmul_nt.
    const w_merged = try allocator.alloc(f32, N * K);
    defer allocator.free(w_merged);
    @memcpy(w_merged, w_base);
    _ = lora_merge.applyLoraDeltaFp32(w_merged, a_lora, b_lora, N, K, r, aor);

    const y_merged = try allocator.alloc(f32, M * N);
    defer allocator.free(y_merged);
    matmulNt(x, w_merged, y_merged, M, N, K);

    var max_abs_fp32: f32 = 0.0;
    var max_y: f32 = 0.0;
    for (y_oracle, y_merged) |a, b| {
        max_abs_fp32 = @max(max_abs_fp32, @abs(a - b));
        max_y = @max(max_y, @abs(a));
    }
    const rel_fp32 = max_abs_fp32 / @max(max_y, 1.0e-9);
    if (rel_fp32 > 1.0e-5) {
        std.debug.print("lora-merge fp32 parity: rel-err {e} > 1e-5\n", .{rel_fp32});
        return error.ParityFailed;
    }

    // ── Path C: bf16 round-trip on the merged weights, then matmul.
    const w_bf16 = try allocator.alloc(u16, N * K);
    defer allocator.free(w_bf16);
    const w_round = try allocator.alloc(f32, N * K);
    defer allocator.free(w_round);
    dtype.f32SliceToBf16(w_merged, w_bf16);
    for (w_bf16, w_round) |s, *d| d.* = dtype.bf16ToF32(s);

    const y_bf16 = try allocator.alloc(f32, M * N);
    defer allocator.free(y_bf16);
    matmulNt(x, w_round, y_bf16, M, N, K);

    var max_abs_bf16: f32 = 0.0;
    for (y_oracle, y_bf16) |a, b| {
        max_abs_bf16 = @max(max_abs_bf16, @abs(a - b));
    }
    const rel_bf16 = max_abs_bf16 / @max(max_y, 1.0e-9);
    // bf16 has ~3 decimal digits / ~1/256 relative precision; a tight
    // gate at 1% catches rounding regressions while leaving head-room
    // for the K=16 reduction noise.
    if (rel_bf16 > 1.0e-2) {
        std.debug.print("lora-merge bf16 round-trip: rel-err {e} > 1e-2\n", .{rel_bf16});
        return error.ParityFailed;
    }

    std.debug.print(
        "PASS lora-merge math (M=4 N=8 K=16 r=4 α=8: fp32 rel-err {e}, bf16 round-trip rel-err {e})\n",
        .{ rel_fp32, rel_bf16 },
    );
}

// ── F2: FlashAttention forward CPU oracle parity ──────────────────────
//
// Gates `cpu/flash_attn.zig:flashAttentionForward` against
// `cpu/train_transformer.zig:attentionForward`. Five shape cases
// covering decode (n_q=1, no mask), prefill causal (n_q == n_kv),
// non-aligned blocks (n_q not divisible by Br), GQA (heads_per_kv > 1),
// and Qwen3-0.6B per-layer dims. One LSE-only case checks the
// log-sum-exp output that FA backward will need to recompute the
// softmax. Tolerance: 1e-5 rel-err — reduction order differs between
// the two paths so bit-equality is not expected.

const FlashCase = struct {
    name: []const u8,
    n_q: usize,
    n_kv: usize,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    causal: bool,
    Br: usize,
    Bc: usize,
};

fn runOneFlashCase(allocator: std.mem.Allocator, case: FlashCase, check_lse: bool) !void {
    const Q = try allocator.alloc(f32, case.n_q * case.n_heads * case.head_dim);
    defer allocator.free(Q);
    const K = try allocator.alloc(f32, case.n_kv * case.n_kv_heads * case.head_dim);
    defer allocator.free(K);
    const V = try allocator.alloc(f32, case.n_kv * case.n_kv_heads * case.head_dim);
    defer allocator.free(V);
    const out_ref = try allocator.alloc(f32, case.n_q * case.n_heads * case.head_dim);
    defer allocator.free(out_ref);
    const out_fa = try allocator.alloc(f32, case.n_q * case.n_heads * case.head_dim);
    defer allocator.free(out_fa);
    const scores = try allocator.alloc(f32, case.n_q * case.n_heads * case.n_kv);
    defer allocator.free(scores);
    const attn = try allocator.alloc(f32, case.n_q * case.n_heads * case.n_kv);
    defer allocator.free(attn);
    const lse = try allocator.alloc(f32, case.n_q * case.n_heads);
    defer allocator.free(lse);

    const s_tile = try allocator.alloc(f32, case.Br * case.Bc);
    defer allocator.free(s_tile);
    const p_tile = try allocator.alloc(f32, case.Br * case.Bc);
    defer allocator.free(p_tile);
    const o_acc = try allocator.alloc(f32, case.Br * case.head_dim);
    defer allocator.free(o_acc);
    const m_acc = try allocator.alloc(f32, case.Br);
    defer allocator.free(m_acc);
    const l_acc = try allocator.alloc(f32, case.Br);
    defer allocator.free(l_acc);

    var prng = std.Random.DefaultPrng.init(0xFA_CE_AB);
    const rng = prng.random();
    for (Q) |*x| x.* = (rng.float(f32) - 0.5) * 0.5;
    for (K) |*x| x.* = (rng.float(f32) - 0.5) * 0.5;
    for (V) |*x| x.* = (rng.float(f32) - 0.5) * 0.5;

    cpu_train_transformer.attentionForward(
        Q, K, V, case.n_q, case.n_kv, case.n_heads, case.n_kv_heads, case.head_dim,
        case.causal, scores, attn, out_ref,
    );
    cpu_flash_attn.flashAttentionForward(
        Q, K, V, case.n_q, case.n_kv, case.n_heads, case.n_kv_heads, case.head_dim,
        case.causal, case.Br, case.Bc,
        out_fa, lse,
        s_tile, p_tile, o_acc, m_acc, l_acc,
    );

    var max_abs: f32 = 0;
    var max_ref: f32 = 0;
    for (out_ref, out_fa) |r, f| {
        const d = @abs(r - f);
        if (d > max_abs) max_abs = d;
        if (@abs(r) > max_ref) max_ref = @abs(r);
    }
    const rel = if (max_ref > 0) max_abs / max_ref else max_abs;
    if (rel > 1e-5) {
        std.debug.print(
            "FA parity FAIL ({s}): max|Δ|={e:.3} max|ref|={e:.3} rel={e:.3}\n",
            .{ case.name, max_abs, max_ref, rel },
        );
        return error.ParityFailed;
    }

    // Optional: cross-check the LSE output against log Σ exp(scores).
    if (check_lse) {
        const NEG_INF: f32 = -std.math.inf(f32);
        for (0..case.n_q) |q| {
            for (0..case.n_heads) |h| {
                const row_off = q * case.n_heads * case.n_kv + h * case.n_kv;
                var max_s: f32 = NEG_INF;
                for (0..case.n_kv) |k| {
                    const v = scores[row_off + k];
                    if (v > max_s) max_s = v;
                }
                var sum: f64 = 0;
                for (0..case.n_kv) |k| {
                    const v = scores[row_off + k];
                    if (!std.math.isInf(v)) sum += @exp(v - max_s);
                }
                const lse_ref: f32 = if (sum > 0)
                    max_s + @as(f32, @floatCast(@log(sum)))
                else
                    NEG_INF;
                const lse_fa = lse[q * case.n_heads + h];
                if (std.math.isInf(lse_ref) and std.math.isInf(lse_fa)) continue;
                const d = @abs(lse_ref - lse_fa);
                const lse_rel = if (@abs(lse_ref) > 1e-6) d / @abs(lse_ref) else d;
                if (lse_rel > 1e-5) {
                    std.debug.print(
                        "FA lse FAIL ({s}) q={d} h={d}: ref={e:.6} fa={e:.6} rel={e:.3}\n",
                        .{ case.name, q, h, lse_ref, lse_fa, lse_rel },
                    );
                    return error.ParityFailed;
                }
            }
        }
    }
}

pub fn runFlashAttentionParitySmoke(allocator: std.mem.Allocator) !void {
    const cases = [_]FlashCase{
        .{ .name = "decode-tiny (n_q=1 n_kv=8 GQA 4:2 d=16)", .n_q = 1, .n_kv = 8, .n_heads = 4, .n_kv_heads = 2, .head_dim = 16, .causal = false, .Br = 1, .Bc = 4 },
        .{ .name = "decode-medium (n_q=1 n_kv=128 GQA 4:2 d=32)", .n_q = 1, .n_kv = 128, .n_heads = 4, .n_kv_heads = 2, .head_dim = 32, .causal = false, .Br = 1, .Bc = 32 },
        .{ .name = "prefill-causal (n_q=4 n_kv=4 GQA 4:2 d=16)", .n_q = 4, .n_kv = 4, .n_heads = 4, .n_kv_heads = 2, .head_dim = 16, .causal = true, .Br = 2, .Bc = 2 },
        .{ .name = "prefill-qwen3 (n_q=16 n_kv=16 GQA 16:8 d=128)", .n_q = 16, .n_kv = 16, .n_heads = 16, .n_kv_heads = 8, .head_dim = 128, .causal = true, .Br = 4, .Bc = 4 },
        .{ .name = "decode-qwen35 (n_q=1 n_kv=128 GQA 8:2 d=256)", .n_q = 1, .n_kv = 128, .n_heads = 8, .n_kv_heads = 2, .head_dim = 256, .causal = false, .Br = 1, .Bc = 8 },
        .{ .name = "prefill-qwen35 (n_q=8 n_kv=8 GQA 8:2 d=256)", .n_q = 8, .n_kv = 8, .n_heads = 8, .n_kv_heads = 2, .head_dim = 256, .causal = true, .Br = 4, .Bc = 4 },
        .{ .name = "non-aligned blocks (n_q=10 n_kv=12 Br=4 Bc=5)", .n_q = 10, .n_kv = 12, .n_heads = 4, .n_kv_heads = 2, .head_dim = 16, .causal = true, .Br = 4, .Bc = 5 },
    };

    for (cases) |c| try runOneFlashCase(allocator, c, false);

    // Dedicated LSE-checked case so the per-row log-sum-exp output is
    // gated independently of the bulk parity sweep.
    try runOneFlashCase(allocator, .{
        .name = "lse-output (n_q=4 n_kv=4 d=8)",
        .n_q = 4, .n_kv = 4, .n_heads = 2, .n_kv_heads = 1, .head_dim = 8,
        .causal = true, .Br = 2, .Bc = 2,
    }, true);

    std.debug.print(
        "PASS flash-attention forward parity ({d} cases incl. GQA + causal + non-aligned blocks + LSE)\n",
        .{cases.len + 1},
    );
}

// ── F6a: FlashAttention backward CPU oracle parity ────────────────────
//
// Gates `cpu/flash_attn.zig:flashAttentionBackward` against
// `cpu/train_transformer.zig:attentionBackward`. The two paths consume
// different forward-saved tensors — the 3-pass oracle takes `attn`
// (full softmax matrix), the FA-2 oracle takes `O` + `lse` and
// recomputes P inline — so this smoke runs both forwards on the same
// inputs and confirms dQ/dK/dV match. Reduction order differs, so
// bit-equality is not expected; tolerance: 1e-4 rel-err.
//
// Shapes mirror the forward parity smoke's training-relevant cases
// (n_q == n_kv, causal, GQA), plus a heads_per_kv=4 case to exercise
// the kv_h fold and a non-aligned causal case so masked-out rows are
// covered.

const FlashBackwardCase = struct {
    name: []const u8,
    n_q: usize,
    n_kv: usize,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    causal: bool,
};

fn runOneFlashBackwardCase(allocator: std.mem.Allocator, case: FlashBackwardCase) !struct { dq: f32, dk: f32, dv: f32 } {
    const Q = try allocator.alloc(f32, case.n_q * case.n_heads * case.head_dim);
    defer allocator.free(Q);
    const K = try allocator.alloc(f32, case.n_kv * case.n_kv_heads * case.head_dim);
    defer allocator.free(K);
    const V = try allocator.alloc(f32, case.n_kv * case.n_kv_heads * case.head_dim);
    defer allocator.free(V);
    const dO = try allocator.alloc(f32, case.n_q * case.n_heads * case.head_dim);
    defer allocator.free(dO);

    // 3-pass forward: produces `attn` (consumed by 3-pass backward) and
    // `O_ref` (used as the saved `O` input to FA backward — equivalent
    // to FA forward's output to ~1e-6 per the F2 forward smoke).
    const scores = try allocator.alloc(f32, case.n_q * case.n_heads * case.n_kv);
    defer allocator.free(scores);
    const attn = try allocator.alloc(f32, case.n_q * case.n_heads * case.n_kv);
    defer allocator.free(attn);
    const O_ref = try allocator.alloc(f32, case.n_q * case.n_heads * case.head_dim);
    defer allocator.free(O_ref);

    // FA forward: produces `lse` (consumed by FA backward) and `O_fa`.
    const O_fa = try allocator.alloc(f32, case.n_q * case.n_heads * case.head_dim);
    defer allocator.free(O_fa);
    const lse = try allocator.alloc(f32, case.n_q * case.n_heads);
    defer allocator.free(lse);

    // FA forward tile scratch.
    const Br: usize = @min(case.n_q, 4);
    const Bc: usize = @min(case.n_kv, 4);
    const s_tile = try allocator.alloc(f32, Br * Bc);
    defer allocator.free(s_tile);
    const p_tile = try allocator.alloc(f32, Br * Bc);
    defer allocator.free(p_tile);
    const o_acc = try allocator.alloc(f32, Br * case.head_dim);
    defer allocator.free(o_acc);
    const m_acc = try allocator.alloc(f32, Br);
    defer allocator.free(m_acc);
    const l_acc = try allocator.alloc(f32, Br);
    defer allocator.free(l_acc);

    // Backward outputs.
    const d_scores = try allocator.alloc(f32, case.n_q * case.n_heads * case.n_kv);
    defer allocator.free(d_scores);
    const dQ_ref = try allocator.alloc(f32, case.n_q * case.n_heads * case.head_dim);
    defer allocator.free(dQ_ref);
    const dK_ref = try allocator.alloc(f32, case.n_kv * case.n_kv_heads * case.head_dim);
    defer allocator.free(dK_ref);
    const dV_ref = try allocator.alloc(f32, case.n_kv * case.n_kv_heads * case.head_dim);
    defer allocator.free(dV_ref);
    const dQ_fa = try allocator.alloc(f32, case.n_q * case.n_heads * case.head_dim);
    defer allocator.free(dQ_fa);
    const dK_fa = try allocator.alloc(f32, case.n_kv * case.n_kv_heads * case.head_dim);
    defer allocator.free(dK_fa);
    const dV_fa = try allocator.alloc(f32, case.n_kv * case.n_kv_heads * case.head_dim);
    defer allocator.free(dV_fa);

    var prng = std.Random.DefaultPrng.init(0xF6_BACE);
    const rng = prng.random();
    for (Q) |*x| x.* = (rng.float(f32) - 0.5) * 0.5;
    for (K) |*x| x.* = (rng.float(f32) - 0.5) * 0.5;
    for (V) |*x| x.* = (rng.float(f32) - 0.5) * 0.5;
    for (dO) |*x| x.* = (rng.float(f32) - 0.5) * 0.5;

    cpu_train_transformer.attentionForward(
        Q, K, V, case.n_q, case.n_kv, case.n_heads, case.n_kv_heads, case.head_dim,
        case.causal, scores, attn, O_ref,
    );
    cpu_flash_attn.flashAttentionForward(
        Q, K, V, case.n_q, case.n_kv, case.n_heads, case.n_kv_heads, case.head_dim,
        case.causal, Br, Bc,
        O_fa, lse,
        s_tile, p_tile, o_acc, m_acc, l_acc,
    );

    cpu_train_transformer.attentionBackward(
        dO, Q, K, V, attn, case.n_q, case.n_kv, case.n_heads, case.n_kv_heads, case.head_dim,
        case.causal, d_scores, dQ_ref, dK_ref, dV_ref,
    );
    cpu_flash_attn.flashAttentionBackward(
        Q, K, V, O_fa, dO, lse, case.n_q, case.n_kv, case.n_heads, case.n_kv_heads, case.head_dim,
        case.causal, dQ_fa, dK_fa, dV_fa,
    );

    const checkRel = struct {
        fn rel(ref: []const f32, got: []const f32) f32 {
            var max_abs: f32 = 0;
            var max_ref: f32 = 0;
            for (ref, got) |r, g| {
                const d = @abs(r - g);
                if (d > max_abs) max_abs = d;
                if (@abs(r) > max_ref) max_ref = @abs(r);
            }
            return if (max_ref > 0) max_abs / max_ref else max_abs;
        }
    }.rel;

    const rel_dq = checkRel(dQ_ref, dQ_fa);
    const rel_dk = checkRel(dK_ref, dK_fa);
    const rel_dv = checkRel(dV_ref, dV_fa);

    const tol: f32 = 1e-4;
    if (rel_dq > tol or rel_dk > tol or rel_dv > tol) {
        std.debug.print(
            "FA backward parity FAIL ({s}): rel(dQ/dK/dV)=({e:.3},{e:.3},{e:.3})\n",
            .{ case.name, rel_dq, rel_dk, rel_dv },
        );
        return error.ParityFailed;
    }

    return .{ .dq = rel_dq, .dk = rel_dk, .dv = rel_dv };
}

pub fn runFlashAttentionBackwardParitySmoke(allocator: std.mem.Allocator) !void {
    const cases = [_]FlashBackwardCase{
        .{ .name = "prefill-tiny (n_q=4 n_kv=4 GQA 4:2 d=16)",        .n_q = 4,  .n_kv = 4,  .n_heads = 4,  .n_kv_heads = 2, .head_dim = 16,  .causal = true  },
        .{ .name = "prefill-qwen3 (n_q=16 n_kv=16 GQA 16:8 d=128)",   .n_q = 16, .n_kv = 16, .n_heads = 16, .n_kv_heads = 8, .head_dim = 128, .causal = true  },
        .{ .name = "prefill-qwen35 (n_q=8 n_kv=8 GQA 8:2 d=256)",     .n_q = 8,  .n_kv = 8,  .n_heads = 8,  .n_kv_heads = 2, .head_dim = 256, .causal = true  },
        .{ .name = "non-aligned (n_q=10 n_kv=12 GQA 4:2 d=16 causal)", .n_q = 10, .n_kv = 12, .n_heads = 4,  .n_kv_heads = 2, .head_dim = 16,  .causal = true  },
        .{ .name = "gqa-4to1 (n_q=8 n_kv=8 heads_per_kv=4 d=16)",     .n_q = 8,  .n_kv = 8,  .n_heads = 4,  .n_kv_heads = 1, .head_dim = 16,  .causal = true  },
        .{ .name = "non-causal (n_q=4 n_kv=8 GQA 4:2 d=16)",          .n_q = 4,  .n_kv = 8,  .n_heads = 4,  .n_kv_heads = 2, .head_dim = 16,  .causal = false },
    };

    var max_dq: f32 = 0;
    var max_dk: f32 = 0;
    var max_dv: f32 = 0;
    for (cases) |c| {
        const r = try runOneFlashBackwardCase(allocator, c);
        if (r.dq > max_dq) max_dq = r.dq;
        if (r.dk > max_dk) max_dk = r.dk;
        if (r.dv > max_dv) max_dv = r.dv;
    }

    std.debug.print(
        "PASS flash-attention backward parity ({d} cases incl. GQA + causal + non-aligned + d=128 + d=256, max rel(dQ/dK/dV)=({e:.2},{e:.2},{e:.2}))\n",
        .{ cases.len, max_dq, max_dk, max_dv },
    );
}

/// Naive `y[m, n] = Σ_k x[m, k] · w[n, k]` — the matmul-nt convention.
/// Used only by smokes; production kernels live in src/runtime.zig.
fn matmulNt(x: []const f32, w: []const f32, y: []f32, M: usize, N: usize, K: usize) void {
    std.debug.assert(x.len == M * K);
    std.debug.assert(w.len == N * K);
    std.debug.assert(y.len == M * N);
    for (0..M) |m| {
        for (0..N) |n| {
            var s: f64 = 0.0;
            for (0..K) |k| s += @as(f64, x[m * K + k]) * @as(f64, w[n * K + k]);
            y[m * N + n] = @floatCast(s);
        }
    }
}

// ── MTP forward CPU sanity smoke (chunk MTP-1b-α) ───────────────────
//
// Loads Qwen3.5-0.8B (smallest MTP-equipped checkpoint), runs
// `forwardMtpStep` end-to-end on a synthetic h_prev, and checks:
//
//   1. No NaN / Inf in logits or h_out (numerical health).
//   2. Deterministic across re-runs from a fresh state (no state leak).
//   3. Output depends on input — running with two different tokens from
//      the same h_prev / pos yields different logits.
//   4. Magnitude is sane (max |logit| < 100, otherwise flag warning).
//
// This is the build-and-shape gate for MTP forward; tighter
// CPU/HF-reference parity will land alongside the GPU recorder in
// MTP-1b-β. Real-model smoke pattern (graceful SKIP if Qwen3.5-0.8B
// isn't in the HF cache).

pub fn runMtpForwardCpuSmoke(allocator: std.mem.Allocator) !void {
    const model_id = "Qwen/Qwen3.5-0.8B";
    const dir_path = hf_cache.resolveModelArg(allocator, model_id) catch |err| switch (err) {
        error.FileNotFound => {
            std.debug.print("SKIP runMtpForwardCpuSmoke ({s} not in HF cache)\n", .{model_id});
            return;
        },
        else => return err,
    };
    defer allocator.free(dir_path);

    var model = try model_mod.Model.load(allocator, dir_path);
    defer model.deinit();

    if (model.mtp_head == null) {
        std.debug.print("FAIL runMtpForwardCpuSmoke: {s} has no MTP head\n", .{model_id});
        return error.NoMtpHead;
    }

    const cfg = model.config;

    var prng = std.Random.DefaultPrng.init(0x1B_DEC0DE);
    const rng = prng.random();

    const h_prev = try allocator.alloc(f32, cfg.hidden_size);
    defer allocator.free(h_prev);
    // Small Gaussian-ish noise — avoids the all-zero degenerate path
    // through RMSNorm and keeps q_norm/k_norm well-conditioned.
    for (h_prev) |*v| v.* = rng.floatNorm(f32) * 0.1;

    const h_out_a = try allocator.alloc(f32, cfg.hidden_size);
    defer allocator.free(h_out_a);
    const logits_a = try allocator.alloc(f32, cfg.vocab_size);
    defer allocator.free(logits_a);

    const h_out_a2 = try allocator.alloc(f32, cfg.hidden_size);
    defer allocator.free(h_out_a2);
    const logits_a2 = try allocator.alloc(f32, cfg.vocab_size);
    defer allocator.free(logits_a2);

    const h_out_b = try allocator.alloc(f32, cfg.hidden_size);
    defer allocator.free(h_out_b);
    const logits_b = try allocator.alloc(f32, cfg.vocab_size);
    defer allocator.free(logits_b);

    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();

    const tok_a: u32 = cfg.eos_token_id orelse 0;
    const tok_b: u32 = if (tok_a + 100 < @as(u32, @intCast(cfg.vocab_size))) tok_a + 100 else 1;

    const t0 = std.time.nanoTimestamp();

    // Run 1: tok_a, fresh state.
    {
        var state = try cpu_mtp.MtpState.init(allocator, &model, 8);
        defer state.deinit();
        _ = arena.reset(.retain_capacity);
        try cpu_mtp.forwardMtpStep(&model, &state, tok_a, h_prev, 0, arena.allocator(), h_out_a, logits_a);
    }

    // Run 2: tok_a, fresh state — must match run 1 (determinism).
    {
        var state = try cpu_mtp.MtpState.init(allocator, &model, 8);
        defer state.deinit();
        _ = arena.reset(.retain_capacity);
        try cpu_mtp.forwardMtpStep(&model, &state, tok_a, h_prev, 0, arena.allocator(), h_out_a2, logits_a2);
    }

    // Run 3: tok_b, fresh state — must produce different logits.
    {
        var state = try cpu_mtp.MtpState.init(allocator, &model, 8);
        defer state.deinit();
        _ = arena.reset(.retain_capacity);
        try cpu_mtp.forwardMtpStep(&model, &state, tok_b, h_prev, 0, arena.allocator(), h_out_b, logits_b);
    }

    const t1 = std.time.nanoTimestamp();
    const ms_total: f64 = @as(f64, @floatFromInt(t1 - t0)) / 1_000_000.0;

    // ── Determinism gate ────────────────────────────────────────────
    for (logits_a, logits_a2, 0..) |a, a2, i| {
        if (a != a2) {
            std.debug.print("FAIL mtp-forward-cpu nondeterministic: logit[{d}] {e} vs {e}\n", .{ i, a, a2 });
            return error.Nondeterministic;
        }
    }

    // ── Numerical-health + magnitude + top-1 ────────────────────────
    var max_abs_a: f32 = 0;
    var max_abs_b: f32 = 0;
    var top1_a: usize = 0;
    var top1_b: usize = 0;
    for (logits_a, 0..) |x, i| {
        if (std.math.isNan(x) or std.math.isInf(x)) {
            std.debug.print("FAIL mtp-forward-cpu: logits_a[{d}] is NaN/Inf\n", .{i});
            return error.NonFinite;
        }
        const ax = @abs(x);
        if (ax > max_abs_a) max_abs_a = ax;
        if (x > logits_a[top1_a]) top1_a = i;
    }
    for (logits_b, 0..) |x, i| {
        if (std.math.isNan(x) or std.math.isInf(x)) {
            std.debug.print("FAIL mtp-forward-cpu: logits_b[{d}] is NaN/Inf\n", .{i});
            return error.NonFinite;
        }
        const ax = @abs(x);
        if (ax > max_abs_b) max_abs_b = ax;
        if (x > logits_b[top1_b]) top1_b = i;
    }
    for (h_out_a) |x| {
        if (std.math.isNan(x) or std.math.isInf(x)) {
            std.debug.print("FAIL mtp-forward-cpu: h_out_a NaN/Inf\n", .{});
            return error.NonFinite;
        }
    }
    if (max_abs_a > 100.0 or max_abs_b > 100.0) {
        std.debug.print("WARN mtp-forward-cpu: |logit| max max(a,b)={d:.3} (> 100, possible blow-up)\n", .{@max(max_abs_a, max_abs_b)});
    }

    // ── Input-dependence gate ───────────────────────────────────────
    var any_diff = false;
    for (logits_a, logits_b) |a, b| {
        if (a != b) {
            any_diff = true;
            break;
        }
    }
    if (!any_diff) {
        std.debug.print("FAIL mtp-forward-cpu: identical logits despite tok_a={d} != tok_b={d}\n", .{ tok_a, tok_b });
        return error.NoInputDependence;
    }

    std.debug.print(
        "PASS mtp-forward-cpu ({s}): {d} MTP layer(s), tok_a={d} top1={d} max|logit|={d:.2}, tok_b={d} top1={d} max|logit|={d:.2}, {d:.0} ms (3 steps)\n",
        .{ model_id, model.mtp_head.?.layers.len, tok_a, top1_a, max_abs_a, tok_b, top1_b, max_abs_b, ms_total },
    );
}

// ── T1: fused FA + TQ4-V CPU oracle parity ─────────────────────────────
//
// Gates `cpu/flash_attn.zig:flashAttentionDecodeForwardTq4V` against
// the standard `flashAttentionForward` running on V values that have
// been pre-dequanted via the SAME `dequantizeBlockTQ4` round-trip the
// fused oracle uses. Both paths therefore consume identical V values;
// the only difference is the inner-loop fusion. Bit-exactness expected
// modulo fp32 reduction-order — tolerance 1e-5 rel-err.
//
// What this smoke does NOT test: TQ4 reconstruction quality (that's
// gated by `--turboquant-smoke`). What it gates: the fused inner-loop
// math + the cache-slice ↔ BlockTQ4 round-trip. Cases sweep the two
// shapes the T-arc targets:
//
//   gemma2b      n_kv=64  GQA 8:1  d=256  (Gemma 2B per-layer)
//   qwen35-08b   n_kv=64  GQA 8:2  d=256  (Qwen3.5 0.8B per-layer)
//   gemma2b-256  n_kv=256 GQA 8:1  d=256  (longer ctx, more tile iterations)

const cpu_turboquant = @import("../cpu/turboquant.zig");

const Tq4VCase = struct {
    name: []const u8,
    n_kv: u32,
    n_heads: u32,
    n_kv_heads: u32,
};

fn runOneTq4VCase(allocator: std.mem.Allocator, case: Tq4VCase) !f32 {
    const head_dim: usize = 256;
    const n_kv: usize = case.n_kv;
    const n_heads: usize = case.n_heads;
    const n_kv_heads: usize = case.n_kv_heads;
    const n_blocks_per_pos: usize = n_kv_heads;

    const Q = try allocator.alloc(f32, n_heads * head_dim);
    defer allocator.free(Q);
    const K = try allocator.alloc(f32, n_kv * n_kv_heads * head_dim);
    defer allocator.free(K);
    const V_fp = try allocator.alloc(f32, n_kv * n_kv_heads * head_dim);
    defer allocator.free(V_fp);
    const V_tq4 = try allocator.alloc(u32, n_kv * n_blocks_per_pos * 33);
    defer allocator.free(V_tq4);

    var prng = std.Random.DefaultPrng.init(0xFADEC4DEC0);
    const rng = prng.random();
    for (Q) |*x| x.* = (rng.float(f32) - 0.5) * 0.5;
    for (K) |*x| x.* = (rng.float(f32) - 0.5) * 0.5;
    // V values that look like layer activations — Gaussian-ish.
    var v_raw = try allocator.alloc(f32, n_kv * n_kv_heads * head_dim);
    defer allocator.free(v_raw);
    for (v_raw) |*x| x.* = (rng.float(f32) - 0.5) * 0.5;

    // Pack V into the GPU cache layout, then dequant back into V_fp so
    // both paths consume the identical (post-quantisation) V values.
    var blk_in: [256]f32 = undefined;
    var blk_out: [256]f32 = undefined;
    for (0..n_kv) |k| {
        for (0..n_kv_heads) |kv_h| {
            const v_off = (k * n_kv_heads + kv_h) * head_dim;
            @memcpy(&blk_in, v_raw[v_off..][0..head_dim]);

            var blk_struct: cpu_turboquant.BlockTQ4(256) = undefined;
            cpu_turboquant.quantizeBlockTQ4(256, &blk_in, &blk_struct);

            const cache_off = (k * n_blocks_per_pos + kv_h) * 33;
            cpu_flash_attn.blockTq4ToCacheSlice_256(&blk_struct, V_tq4[cache_off..][0..33]);

            // Dequant via the same struct → V_fp. This is the "reference"
            // V the standard FA path will see.
            cpu_turboquant.dequantizeBlockTQ4(256, &blk_struct, &blk_out);
            @memcpy(V_fp[v_off..][0..head_dim], &blk_out);
        }
    }

    // Reference: standard FA at n_q=1 on the pre-dequanted V.
    const out_ref = try allocator.alloc(f32, n_heads * head_dim);
    defer allocator.free(out_ref);
    const Br: usize = 1;
    const Bc: usize = 8;
    const s_tile = try allocator.alloc(f32, Br * Bc);
    defer allocator.free(s_tile);
    const p_tile = try allocator.alloc(f32, Br * Bc);
    defer allocator.free(p_tile);
    const o_acc = try allocator.alloc(f32, Br * head_dim);
    defer allocator.free(o_acc);
    const m_acc = try allocator.alloc(f32, Br);
    defer allocator.free(m_acc);
    const l_acc = try allocator.alloc(f32, Br);
    defer allocator.free(l_acc);
    cpu_flash_attn.flashAttentionForward(
        Q, K, V_fp,
        1, n_kv, n_heads, n_kv_heads, head_dim,
        false, Br, Bc,
        out_ref, null,
        s_tile, p_tile, o_acc, m_acc, l_acc,
    );

    // Fused: same FA at n_q=1 reading V from the TQ4 cache.
    const out_fused = try allocator.alloc(f32, n_heads * head_dim);
    defer allocator.free(out_fused);
    const v_block = try allocator.alloc(f32, head_dim);
    defer allocator.free(v_block);
    cpu_flash_attn.flashAttentionDecodeForwardTq4V(
        Q, K, V_tq4,
        n_kv, n_heads, n_kv_heads, n_blocks_per_pos, Bc,
        out_fused,
        s_tile, p_tile, o_acc, v_block,
    );

    var max_abs: f32 = 0;
    var max_ref: f32 = 0;
    for (out_ref, out_fused) |r, f| {
        const d = @abs(r - f);
        if (d > max_abs) max_abs = d;
        if (@abs(r) > max_ref) max_ref = @abs(r);
    }
    const rel = if (max_ref > 0) max_abs / max_ref else max_abs;
    if (rel > 1e-5) {
        std.debug.print(
            "FA TQ4-V parity FAIL ({s}): max|Δ|={e:.3} max|ref|={e:.3} rel={e:.3}\n",
            .{ case.name, max_abs, max_ref, rel },
        );
        return error.ParityFailed;
    }
    return rel;
}

pub fn runFlashAttentionTq4VParitySmoke(allocator: std.mem.Allocator) !void {
    const cases = [_]Tq4VCase{
        .{ .name = "gemma2b      (n_kv=64  GQA 8:1 d=256)", .n_kv = 64,  .n_heads = 8, .n_kv_heads = 1 },
        .{ .name = "qwen35-08b   (n_kv=64  GQA 8:2 d=256)", .n_kv = 64,  .n_heads = 8, .n_kv_heads = 2 },
        .{ .name = "gemma2b-256  (n_kv=256 GQA 8:1 d=256)", .n_kv = 256, .n_heads = 8, .n_kv_heads = 1 },
    };
    var max_rel: f32 = 0;
    for (cases) |c| {
        const r = try runOneTq4VCase(allocator, c);
        if (r > max_rel) max_rel = r;
    }
    std.debug.print(
        "PASS flash-attention TQ4-V CPU oracle parity ({d} cases incl. Gemma 2B + Qwen3.5 0.8B per-layer, max rel={e:.2})\n",
        .{ cases.len, max_rel },
    );
}
