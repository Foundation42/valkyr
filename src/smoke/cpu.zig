//! Pure CPU oracle smokes invoked from main.zig's "run all" fallthrough:
//! matmul, RoPE identity, softmax, MLP forward+backward+SGD, multi-layer
//! MlpN training, and RMSNorm backward parity. Extracted from main.zig.

const std = @import("std");
const cpu_math = @import("../cpu/math.zig");
const cpu_train = @import("../cpu/train.zig");
const cpu_train_transformer = @import("../cpu/train_transformer.zig");
const cpu_lora = @import("../cpu/lora.zig");
const dtype = @import("../dtype.zig");
const lora_merge = @import("../commands/lora_merge.zig");
const safetensors = @import("../safetensors.zig");

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
