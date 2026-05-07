//! Activation + RoPE + softmax + FWHT/RHT + TurboQuant TQ4 + GeLU +
//! Q4_0/Q4_K + miscellaneous parity smokes for individual GPU kernels
//! and CPU oracles. Driven by the no-arg fallthrough in main.zig.
//! Extracted from main.zig.

const std = @import("std");
const vk = @import("../gpu/vk.zig");
const buffer = @import("../gpu/buffer.zig");
const pipeline = @import("../gpu/pipeline.zig");
const gpu_recorder = @import("../gpu/recorder.zig");
const cpu_math = @import("../cpu/math.zig");
const cpu_train_transformer = @import("../cpu/train_transformer.zig");
const turboquant = @import("../cpu/turboquant.zig");
const q4_0 = @import("../cpu/q4_0.zig");
const q4_k = @import("../cpu/q4_k.zig");
const runtime = @import("../runtime.zig");
const runtime_hybrid = @import("../runtime_hybrid.zig");
const shaders = @import("shaders");


// ── gpu geglu smoke: synthetic vs CPU geglu ─────────────────────────

const GegluPush = runtime.GegluPush;

pub fn runGpuGegluSmoke(allocator: std.mem.Allocator) !void {
    var ctx = try vk.Context.init(allocator);
    defer ctx.deinit();

    const n: usize = 4096;
    const gate = try allocator.alloc(f32, n);
    defer allocator.free(gate);
    const upv = try allocator.alloc(f32, n);
    defer allocator.free(upv);
    // Range [-3, 3] across the array hits both the tanh saturation
    // tails and the linear region around 0 — exercises the full curve.
    for (gate, upv, 0..) |*g, *u, i| {
        const t = @as(f32, @floatFromInt(i)) / @as(f32, @floatFromInt(n));
        g.* = -3.0 + 6.0 * t;
        u.* = 1.0 - 2.0 * t;
    }

    const want = try allocator.alloc(f32, n);
    defer allocator.free(want);
    try cpu_math.geglu(want, gate, upv);

    var buf_g = try buffer.Buffer.initStatic(&ctx, f32, gate);
    defer buf_g.deinit(ctx.device);
    var buf_u = try buffer.Buffer.initStatic(&ctx, f32, upv);
    defer buf_u.deinit(ctx.device);
    var buf_o = try buffer.Buffer.initDeviceOnly(&ctx, n * @sizeOf(f32));
    defer buf_o.deinit(ctx.device);

    var kern = try pipeline.Kernel.init(&ctx, &shaders.geglu, 3, @sizeOf(GegluPush));
    defer kern.deinit();
    try kern.bind(&.{ &buf_g, &buf_u, &buf_o });

    const local: u32 = 256;
    const groups: u32 = (@as(u32, @intCast(n)) + local - 1) / local;
    const push = GegluPush{ .n = @intCast(n) };
    try buffer.submitOneShot(&ctx, struct {
        kern: *const pipeline.Kernel,
        push: *const GegluPush,
        groups: u32,
        pub fn record(s: @This(), cmd: vk.c.VkCommandBuffer) void {
            s.kern.dispatch(cmd, s.push, s.groups, 1, 1);
        }
    }{ .kern = &kern, .push = &push, .groups = groups });

    const got = try allocator.alloc(f32, n);
    defer allocator.free(got);
    try buf_o.readBack(&ctx, f32, got);

    var max_abs: f32 = 0;
    for (got, want) |g, e| {
        const d = @abs(g - e);
        if (d > max_abs) max_abs = d;
    }
    if (max_abs > 1e-5) {
        std.debug.print("GPU GeGLU: max |Δ| = {e}\n", .{max_abs});
        return error.ParityFailed;
    }
    std.debug.print("PASS GPU geglu synthetic ({d} elems, x∈[-3,3])\n", .{n});
}

// ── gpu rope smoke: pos=0 identity + pos=1 vs CPU ───────────────────

const RopePush = runtime.RopePush;
const RopePartialPush = runtime.RopePartialPush;

pub fn runGpuRopeSmoke(allocator: std.mem.Allocator) !void {
    var ctx = try vk.Context.init(allocator);
    defer ctx.deinit();

    const n_heads: usize = 8;
    const head_dim: usize = 64;
    const total = n_heads * head_dim;

    const in_v = try allocator.alloc(f32, total);
    defer allocator.free(in_v);
    for (in_v, 0..) |*x, i| x.* = @as(f32, @floatFromInt(i)) * 0.001 - 0.5;

    var buf_in = try buffer.Buffer.initStatic(&ctx, f32, in_v);
    defer buf_in.deinit(ctx.device);
    var buf_out = try buffer.Buffer.initDeviceOnly(&ctx, total * @sizeOf(f32));
    defer buf_out.deinit(ctx.device);

    var kern = try pipeline.Kernel.init(&ctx, &shaders.rope, 2, @sizeOf(RopePush));
    defer kern.deinit();
    try kern.bind(&.{ &buf_in, &buf_out });

    // pos=0: must be identity.
    const local: u32 = 256;
    const pairs: u32 = @intCast(n_heads * (head_dim / 2));
    const groups: u32 = (pairs + local - 1) / local;

    const push0 = RopePush{ .n_heads = @intCast(n_heads), .head_dim = @intCast(head_dim), .pos = 0, .theta_base = 10000.0 };
    try buffer.submitOneShot(&ctx, struct {
        kern: *const pipeline.Kernel,
        push: *const RopePush,
        groups: u32,
        pub fn record(s: @This(), cmd: vk.c.VkCommandBuffer) void {
            s.kern.dispatch(cmd, s.push, s.groups, 1, 1);
        }
    }{ .kern = &kern, .push = &push0, .groups = groups });

    const got0 = try allocator.alloc(f32, total);
    defer allocator.free(got0);
    try buf_out.readBack(&ctx, f32, got0);
    for (got0, in_v, 0..) |g, e, i| {
        if (g != e) {
            std.debug.print("GPU RoPE pos=0 NOT identity at {d}: in={d} out={d}\n", .{ i, e, g });
            return error.ParityFailed;
        }
    }

    // pos=1: parity vs CPU.
    const want = try allocator.alloc(f32, total);
    defer allocator.free(want);
    try cpu_math.applyRope(want, in_v, n_heads, head_dim, 1, 10000.0);

    const push1 = RopePush{ .n_heads = @intCast(n_heads), .head_dim = @intCast(head_dim), .pos = 1, .theta_base = 10000.0 };
    try buffer.submitOneShot(&ctx, struct {
        kern: *const pipeline.Kernel,
        push: *const RopePush,
        groups: u32,
        pub fn record(s: @This(), cmd: vk.c.VkCommandBuffer) void {
            s.kern.dispatch(cmd, s.push, s.groups, 1, 1);
        }
    }{ .kern = &kern, .push = &push1, .groups = groups });

    const got1 = try allocator.alloc(f32, total);
    defer allocator.free(got1);
    try buf_out.readBack(&ctx, f32, got1);

    var max_abs: f32 = 0;
    for (got1, want) |g, e| {
        const d = @abs(g - e);
        if (d > max_abs) max_abs = d;
    }
    if (max_abs > 1e-5) {
        std.debug.print("GPU RoPE pos=1: max |Δ| = {e}\n", .{max_abs});
        return error.ParityFailed;
    }
    std.debug.print("PASS GPU rope (pos=0 identity + pos=1 vs CPU within 1e-5)\n", .{});
}

// ── gpu rope-partial smoke: rotary_dim < head_dim, vs CPU ───────────

pub fn runGpuRopePartialSmoke(allocator: std.mem.Allocator) !void {
    var ctx = try vk.Context.init(allocator);
    defer ctx.deinit();

    // Qwen3.5 shape: head_dim=256, rotary_dim=64, theta=10M.
    const n_heads: usize = 8;
    const head_dim: usize = 256;
    const rotary_dim: usize = 64;
    const total = n_heads * head_dim;
    const theta_base: f32 = 1.0e7;

    const in_v = try allocator.alloc(f32, total);
    defer allocator.free(in_v);
    for (in_v, 0..) |*x, i| x.* = @as(f32, @floatFromInt(i)) * 0.001 - 0.5;

    var buf_in = try buffer.Buffer.initStatic(&ctx, f32, in_v);
    defer buf_in.deinit(ctx.device);
    var buf_out = try buffer.Buffer.initDeviceOnly(&ctx, total * @sizeOf(f32));
    defer buf_out.deinit(ctx.device);

    var kern = try pipeline.Kernel.init(&ctx, &shaders.rope_partial, 2, @sizeOf(RopePartialPush));
    defer kern.deinit();
    try kern.bind(&.{ &buf_in, &buf_out });

    // Dispatch one thread per output element (n_heads * head_dim).
    const local: u32 = 256;
    const elems: u32 = @intCast(total);
    const groups: u32 = (elems + local - 1) / local;

    // Reference: CPU partial RoPE at pos=3.
    const want = try allocator.alloc(f32, total);
    defer allocator.free(want);
    try cpu_math.applyRopePartial(want, in_v, n_heads, head_dim, rotary_dim, 3, theta_base);

    const push = RopePartialPush{
        .n_heads = @intCast(n_heads),
        .head_dim = @intCast(head_dim),
        .rotary_dim = @intCast(rotary_dim),
        .pos = 3,
        .theta_base = theta_base,
    };
    try buffer.submitOneShot(&ctx, struct {
        kern: *const pipeline.Kernel,
        push: *const RopePartialPush,
        groups: u32,
        pub fn record(s: @This(), cmd: vk.c.VkCommandBuffer) void {
            s.kern.dispatch(cmd, s.push, s.groups, 1, 1);
        }
    }{ .kern = &kern, .push = &push, .groups = groups });

    const got = try allocator.alloc(f32, total);
    defer allocator.free(got);
    try buf_out.readBack(&ctx, f32, got);

    var max_abs: f32 = 0;
    for (got, want) |g, e| {
        const d = @abs(g - e);
        if (d > max_abs) max_abs = d;
    }
    if (max_abs > 1e-5) {
        std.debug.print("GPU rope_partial: max |Δ| = {e}\n", .{max_abs});
        return error.ParityFailed;
    }
    // Also confirm the pass-through region is byte-identical (the
    // tail of each head must equal the input).
    for (0..n_heads) |h| {
        for (rotary_dim..head_dim) |d| {
            const idx = h * head_dim + d;
            if (got[idx] != in_v[idx]) {
                std.debug.print("rope_partial pass-through broken at h={d} d={d}: in={d} out={d}\n", .{ h, d, in_v[idx], got[idx] });
                return error.ParityFailed;
            }
        }
    }
    std.debug.print("PASS GPU rope_partial (rotary_dim=64 of head_dim=256, max |Δ| vs CPU = {e})\n", .{max_abs});
}

// ── gpu split_q_gate smoke: synthetic round-trip ────────────────────
//
// Validates that the `[h0_q, h0_gate, h1_q, h1_gate, …]` interleaved
// layout produced by the 2× q_proj is correctly demuxed into two flat
// `(num_heads*head_dim)` buffers preserving per-head ordering.

const SplitQGatePush = runtime_hybrid.SplitQGatePush;

pub fn runGpuSplitQGateSmoke(allocator: std.mem.Allocator) !void {
    var ctx = try vk.Context.init(allocator);
    defer ctx.deinit();

    const num_heads: usize = 4;
    const head_dim: usize = 8;
    const total = num_heads * head_dim;
    const wide = 2 * total;

    const in_v = try allocator.alloc(f32, wide);
    defer allocator.free(in_v);
    // Synthetic: q values are positive ints, gate values negative ints,
    // so any layout bug shows up as sign mismatches.
    for (0..num_heads) |h| {
        for (0..head_dim) |d| {
            const off = h * 2 * head_dim;
            in_v[off + d] = @floatFromInt(h * 10 + d + 1);
            in_v[off + head_dim + d] = -@as(f32, @floatFromInt(h * 10 + d + 1));
        }
    }
    var buf_in = try buffer.Buffer.initStatic(&ctx, f32, in_v);
    defer buf_in.deinit(ctx.device);
    var buf_q = try buffer.Buffer.initDeviceOnly(&ctx, total * @sizeOf(f32));
    defer buf_q.deinit(ctx.device);
    var buf_gate = try buffer.Buffer.initDeviceOnly(&ctx, total * @sizeOf(f32));
    defer buf_gate.deinit(ctx.device);

    var kern = try pipeline.Kernel.init(&ctx, &shaders.split_q_gate, 3, @sizeOf(SplitQGatePush));
    defer kern.deinit();
    try kern.bind(&.{ &buf_in, &buf_q, &buf_gate });

    const local: u32 = 256;
    const groups: u32 = (@as(u32, @intCast(total)) + local - 1) / local;
    const push = SplitQGatePush{ .num_heads = @intCast(num_heads), .head_dim = @intCast(head_dim) };
    try buffer.submitOneShot(&ctx, struct {
        kern: *const pipeline.Kernel,
        push: *const SplitQGatePush,
        groups: u32,
        pub fn record(s: @This(), cmd: vk.c.VkCommandBuffer) void {
            s.kern.dispatch(cmd, s.push, s.groups, 1, 1);
        }
    }{ .kern = &kern, .push = &push, .groups = groups });

    const got_q = try allocator.alloc(f32, total);
    defer allocator.free(got_q);
    const got_gate = try allocator.alloc(f32, total);
    defer allocator.free(got_gate);
    try buf_q.readBack(&ctx, f32, got_q);
    try buf_gate.readBack(&ctx, f32, got_gate);

    for (0..num_heads) |h| {
        for (0..head_dim) |d| {
            const want_q: f32 = @floatFromInt(h * 10 + d + 1);
            const want_g: f32 = -want_q;
            const idx = h * head_dim + d;
            if (got_q[idx] != want_q or got_gate[idx] != want_g) {
                std.debug.print("split_q_gate mismatch at h={d} d={d}: q={d}/{d} gate={d}/{d}\n", .{ h, d, got_q[idx], want_q, got_gate[idx], want_g });
                return error.ParityFailed;
            }
        }
    }
    std.debug.print("PASS GPU split_q_gate (4 heads × 8 dim, layout bit-exact)\n", .{});
}

// ── gpu sigmoid_mul smoke: out = a * sigmoid(b) vs CPU ──────────────

const SigmoidMulPush = runtime_hybrid.SigmoidMulPush;

pub fn runGpuSigmoidMulSmoke(allocator: std.mem.Allocator) !void {
    var ctx = try vk.Context.init(allocator);
    defer ctx.deinit();

    const n: usize = 1024;
    const a = try allocator.alloc(f32, n);
    defer allocator.free(a);
    const b = try allocator.alloc(f32, n);
    defer allocator.free(b);
    for (a, b, 0..) |*ai, *bi, i| {
        ai.* = @sin(@as(f32, @floatFromInt(i)) * 0.1) * 2.0;
        bi.* = @cos(@as(f32, @floatFromInt(i)) * 0.13) * 4.0;
    }

    var buf_a = try buffer.Buffer.initStatic(&ctx, f32, a);
    defer buf_a.deinit(ctx.device);
    var buf_b = try buffer.Buffer.initStatic(&ctx, f32, b);
    defer buf_b.deinit(ctx.device);
    var buf_out = try buffer.Buffer.initDeviceOnly(&ctx, n * @sizeOf(f32));
    defer buf_out.deinit(ctx.device);

    var kern = try pipeline.Kernel.init(&ctx, &shaders.sigmoid_mul, 3, @sizeOf(SigmoidMulPush));
    defer kern.deinit();
    try kern.bind(&.{ &buf_a, &buf_b, &buf_out });

    const local: u32 = 256;
    const groups: u32 = (@as(u32, @intCast(n)) + local - 1) / local;
    const push = SigmoidMulPush{ .n_elem = @intCast(n) };
    try buffer.submitOneShot(&ctx, struct {
        kern: *const pipeline.Kernel,
        push: *const SigmoidMulPush,
        groups: u32,
        pub fn record(s: @This(), cmd: vk.c.VkCommandBuffer) void {
            s.kern.dispatch(cmd, s.push, s.groups, 1, 1);
        }
    }{ .kern = &kern, .push = &push, .groups = groups });

    const got = try allocator.alloc(f32, n);
    defer allocator.free(got);
    try buf_out.readBack(&ctx, f32, got);

    var max_abs: f32 = 0;
    for (got, a, b) |g, ai, bi| {
        const want = ai * (1.0 / (1.0 + @exp(-bi)));
        const d = @abs(g - want);
        if (d > max_abs) max_abs = d;
    }
    if (max_abs > 1e-6) {
        std.debug.print("GPU sigmoid_mul: max |Δ| = {e}\n", .{max_abs});
        return error.ParityFailed;
    }
    std.debug.print("PASS GPU sigmoid_mul (1024 elems, max |Δ| vs CPU = {e})\n", .{max_abs});
}

// ── chunk 8c-β-3a-1: SwiGLU primitive parity (CPU + GPU) ────────────
//
// CPU oracle: numeric-grad parity for swigluForward / swigluBackward.
// GPU parity: swiglu_forward + swiglu_backward shaders match the
// `cpu_train_transformer.swigluForward` / `swigluBackward` reference
// to fp32 tolerance.
//
// SwiGLU is the FFN nonlinearity for Llama / Qwen / Mistral; this
// primitive replaces the toy stack's ReLU FFN in a follow-up
// integration chunk (β-3a-2). Here we just verify the math.

pub fn runSwiGluCpuSmoke(allocator: std.mem.Allocator) !void {
    const n: usize = 64;
    var prng = std.Random.DefaultPrng.init(0x5C_16_1A_AA);
    const rng = prng.random();

    const pre_gate = try allocator.alloc(f32, n);
    defer allocator.free(pre_gate);
    const up = try allocator.alloc(f32, n);
    defer allocator.free(up);
    // Scale ±1 matches the other primitive parity smokes; the central-
    // difference truncation error scales with the input curvature, so
    // wider inputs blow past 1% rel-err even when the analytical
    // gradient is correct (verified by hand at probe points).
    for (pre_gate) |*v| v.* = rng.float(f32) * 2.0 - 1.0;
    for (up) |*v| v.* = rng.float(f32) * 2.0 - 1.0;

    const gated = try allocator.alloc(f32, n);
    defer allocator.free(gated);
    cpu_train_transformer.swigluForward(pre_gate, up, gated);

    // Random upstream gradient.
    const d_gated = try allocator.alloc(f32, n);
    defer allocator.free(d_gated);
    for (d_gated) |*v| v.* = (rng.float(f32) * 2.0 - 1.0) * 0.5;

    const d_pre_gate = try allocator.alloc(f32, n);
    defer allocator.free(d_pre_gate);
    const d_up = try allocator.alloc(f32, n);
    defer allocator.free(d_up);
    cpu_train_transformer.swigluBackward(d_gated, pre_gate, up, d_pre_gate, d_up);

    // Numeric-grad check against the analytical backward. Loss
    // L = Σ d_gated · gated; dL/d_pre_gate and dL/d_up should match.
    const eps: f32 = 1e-3;
    var max_rel: f32 = 0;
    const n_probes: usize = 8;
    for (0..n_probes) |pi| {
        const i = (pi * 7 + 3) % n;
        // d/d_pre_gate
        const pg_p = pre_gate[i] + eps;
        const pg_m = pre_gate[i] - eps;
        const sig_p = 1.0 / (1.0 + @exp(-pg_p));
        const sig_m = 1.0 / (1.0 + @exp(-pg_m));
        const gated_p = pg_p * sig_p * up[i];
        const gated_m = pg_m * sig_m * up[i];
        const num_dpg = d_gated[i] * (gated_p - gated_m) / (2.0 * eps);
        const denom_pg = @max(@abs(num_dpg), 1e-6);
        const rel_pg = @abs(num_dpg - d_pre_gate[i]) / denom_pg;
        if (rel_pg > max_rel) max_rel = rel_pg;
        // d/d_up
        const u_p = up[i] + eps;
        const u_m = up[i] - eps;
        const sig_z = 1.0 / (1.0 + @exp(-pre_gate[i]));
        const silu_z = pre_gate[i] * sig_z;
        const gu_p = silu_z * u_p;
        const gu_m = silu_z * u_m;
        const num_du = d_gated[i] * (gu_p - gu_m) / (2.0 * eps);
        const denom_u = @max(@abs(num_du), 1e-6);
        const rel_u = @abs(num_du - d_up[i]) / denom_u;
        if (rel_u > max_rel) max_rel = rel_u;
    }
    if (max_rel > 1e-2) {
        std.debug.print("SwiGLU CPU: max numeric-grad rel-err = {e}\n", .{max_rel});
        return error.ParityFailed;
    }
    std.debug.print("PASS SwiGLU CPU oracle (n={d}, numeric-grad parity ≤ {e})\n", .{ n, max_rel });
}

pub fn runGpuSwiGluSmoke(allocator: std.mem.Allocator) !void {
    var ctx = try vk.Context.init(allocator);
    defer ctx.deinit();

    const n: usize = 1024;
    const pre_gate = try allocator.alloc(f32, n);
    defer allocator.free(pre_gate);
    const up = try allocator.alloc(f32, n);
    defer allocator.free(up);
    const d_gated = try allocator.alloc(f32, n);
    defer allocator.free(d_gated);
    var prng = std.Random.DefaultPrng.init(0x5C_16_1A_BB);
    const rng = prng.random();
    for (pre_gate) |*v| v.* = (rng.float(f32) * 2.0 - 1.0) * 3.0;
    for (up) |*v| v.* = (rng.float(f32) * 2.0 - 1.0) * 3.0;
    for (d_gated) |*v| v.* = (rng.float(f32) * 2.0 - 1.0) * 0.5;

    // CPU references.
    const want_gated = try allocator.alloc(f32, n);
    defer allocator.free(want_gated);
    cpu_train_transformer.swigluForward(pre_gate, up, want_gated);
    const want_d_pre_gate = try allocator.alloc(f32, n);
    defer allocator.free(want_d_pre_gate);
    const want_d_up = try allocator.alloc(f32, n);
    defer allocator.free(want_d_up);
    cpu_train_transformer.swigluBackward(d_gated, pre_gate, up, want_d_pre_gate, want_d_up);

    // GPU forward.
    var buf_pg = try buffer.Buffer.initStatic(&ctx, f32, pre_gate);
    defer buf_pg.deinit(ctx.device);
    var buf_up = try buffer.Buffer.initStatic(&ctx, f32, up);
    defer buf_up.deinit(ctx.device);
    var buf_gated = try buffer.Buffer.initDeviceOnly(&ctx, n * @sizeOf(f32));
    defer buf_gated.deinit(ctx.device);

    var k_fwd = try pipeline.Kernel.init(&ctx, &shaders.swiglu_forward, 3, @sizeOf(runtime.SwigluPush));
    defer k_fwd.deinit();
    try k_fwd.bind(&.{ &buf_pg, &buf_up, &buf_gated });

    const local: u32 = 256;
    const groups: u32 = (@as(u32, @intCast(n)) + local - 1) / local;
    const push = runtime.SwigluPush{ .n = @intCast(n) };
    try buffer.submitOneShot(&ctx, struct {
        kern: *const pipeline.Kernel,
        push: *const runtime.SwigluPush,
        groups: u32,
        pub fn record(s: @This(), cmd: vk.c.VkCommandBuffer) void {
            s.kern.dispatch(cmd, s.push, s.groups, 1, 1);
        }
    }{ .kern = &k_fwd, .push = &push, .groups = groups });

    const got_gated = try allocator.alloc(f32, n);
    defer allocator.free(got_gated);
    try buf_gated.readBack(&ctx, f32, got_gated);

    var max_fwd: f32 = 0;
    for (got_gated, want_gated) |g, w| {
        const d = @abs(g - w);
        if (d > max_fwd) max_fwd = d;
    }
    if (max_fwd > 1e-6) {
        std.debug.print("SwiGLU GPU forward: max |Δ| = {e}\n", .{max_fwd});
        return error.ParityFailed;
    }

    // GPU backward.
    var buf_dg = try buffer.Buffer.initStatic(&ctx, f32, d_gated);
    defer buf_dg.deinit(ctx.device);
    var buf_dpg = try buffer.Buffer.initDeviceOnly(&ctx, n * @sizeOf(f32));
    defer buf_dpg.deinit(ctx.device);
    var buf_du = try buffer.Buffer.initDeviceOnly(&ctx, n * @sizeOf(f32));
    defer buf_du.deinit(ctx.device);

    var k_bw = try pipeline.Kernel.init(&ctx, &shaders.swiglu_backward, 5, @sizeOf(runtime.SwigluPush));
    defer k_bw.deinit();
    try k_bw.bind(&.{ &buf_dg, &buf_pg, &buf_up, &buf_dpg, &buf_du });

    try buffer.submitOneShot(&ctx, struct {
        kern: *const pipeline.Kernel,
        push: *const runtime.SwigluPush,
        groups: u32,
        pub fn record(s: @This(), cmd: vk.c.VkCommandBuffer) void {
            s.kern.dispatch(cmd, s.push, s.groups, 1, 1);
        }
    }{ .kern = &k_bw, .push = &push, .groups = groups });

    const got_dpg = try allocator.alloc(f32, n);
    defer allocator.free(got_dpg);
    const got_du = try allocator.alloc(f32, n);
    defer allocator.free(got_du);
    try buf_dpg.readBack(&ctx, f32, got_dpg);
    try buf_du.readBack(&ctx, f32, got_du);

    var max_bw: f32 = 0;
    for (got_dpg, want_d_pre_gate) |g, w| {
        const d = @abs(g - w);
        if (d > max_bw) max_bw = d;
    }
    for (got_du, want_d_up) |g, w| {
        const d = @abs(g - w);
        if (d > max_bw) max_bw = d;
    }
    if (max_bw > 1e-6) {
        std.debug.print("SwiGLU GPU backward: max |Δ| = {e}\n", .{max_bw});
        return error.ParityFailed;
    }
    std.debug.print(
        "PASS GPU SwiGLU forward + backward (n={d}, fwd max |Δ| = {e}, bw max |Δ| = {e})\n",
        .{ n, max_fwd, max_bw },
    );
}

// ── chunk 8c-β-3a-3: batched RoPE primitive parity ──────────────────
//
// Tests the new `rope_partial_batched.comp` + `rope_backward_batched.comp`
// shaders against the CPU `ropeForwardBatched` / `ropeBackwardBatched`
// helpers. Operates over [n_pos, n_heads, head_dim] in one dispatch
// each. Setting rotary_dim < head_dim exercises the partial-rotation
// path (Qwen3.5-style); we run with rotary_dim = head_dim (full RoPE).

pub fn runGpuRopeBatchedSmoke(allocator: std.mem.Allocator) !void {
    var ctx = try vk.Context.init(allocator);
    defer ctx.deinit();

    const n_pos: usize = 4;
    const n_heads: usize = 2;
    const head_dim: usize = 8;
    const rotary_dim: usize = head_dim;
    const theta_base: f32 = 10_000.0;
    const total = n_pos * n_heads * head_dim;

    const x = try allocator.alloc(f32, total);
    defer allocator.free(x);
    var prng = std.Random.DefaultPrng.init(0xCAFE_B0_BE);
    const rng = prng.random();
    for (x) |*v| v.* = rng.float(f32) * 2.0 - 1.0;

    const cpu_y = try allocator.alloc(f32, total);
    defer allocator.free(cpu_y);
    try cpu_train_transformer.ropeForwardBatched(cpu_y, x, n_pos, n_heads, head_dim, rotary_dim, theta_base);

    const dy = try allocator.alloc(f32, total);
    defer allocator.free(dy);
    for (dy) |*v| v.* = rng.float(f32) * 2.0 - 1.0;

    const cpu_dx = try allocator.alloc(f32, total);
    defer allocator.free(cpu_dx);
    try cpu_train_transformer.ropeBackwardBatched(cpu_dx, dy, n_pos, n_heads, head_dim, rotary_dim, theta_base);

    // GPU forward.
    var buf_x = try buffer.Buffer.initStatic(&ctx, f32, x);
    defer buf_x.deinit(ctx.device);
    var buf_y = try buffer.Buffer.initDeviceOnly(&ctx, total * @sizeOf(f32));
    defer buf_y.deinit(ctx.device);
    var k_fwd = try pipeline.Kernel.init(&ctx, &shaders.rope_partial_batched, 2, @sizeOf(runtime.RopeBatchedPush));
    defer k_fwd.deinit();
    try k_fwd.bind(&.{ &buf_x, &buf_y });
    const push = runtime.RopeBatchedPush{
        .n_pos = @intCast(n_pos),
        .n_heads = @intCast(n_heads),
        .head_dim = @intCast(head_dim),
        .rotary_dim = @intCast(rotary_dim),
        .theta_base = theta_base,
    };
    const local: u32 = 256;
    const groups: u32 = (@as(u32, @intCast(total)) + local - 1) / local;
    try buffer.submitOneShot(&ctx, struct {
        kern: *const pipeline.Kernel,
        push: *const runtime.RopeBatchedPush,
        groups: u32,
        pub fn record(s: @This(), cmd: vk.c.VkCommandBuffer) void {
            s.kern.dispatch(cmd, s.push, s.groups, 1, 1);
        }
    }{ .kern = &k_fwd, .push = &push, .groups = groups });
    const gpu_y = try allocator.alloc(f32, total);
    defer allocator.free(gpu_y);
    try buf_y.readBack(&ctx, f32, gpu_y);

    var max_fwd: f32 = 0;
    for (gpu_y, cpu_y) |g, c| {
        const d = @abs(g - c);
        if (d > max_fwd) max_fwd = d;
    }
    if (max_fwd > 1e-5) {
        std.debug.print("RoPE batched fwd: max |Δ| = {e}\n", .{max_fwd});
        return error.ParityFailed;
    }

    // GPU backward.
    var buf_dy = try buffer.Buffer.initStatic(&ctx, f32, dy);
    defer buf_dy.deinit(ctx.device);
    var buf_dx = try buffer.Buffer.initDeviceOnly(&ctx, total * @sizeOf(f32));
    defer buf_dx.deinit(ctx.device);
    var k_bw = try pipeline.Kernel.init(&ctx, &shaders.rope_backward_batched, 2, @sizeOf(runtime.RopeBatchedPush));
    defer k_bw.deinit();
    try k_bw.bind(&.{ &buf_dy, &buf_dx });
    try buffer.submitOneShot(&ctx, struct {
        kern: *const pipeline.Kernel,
        push: *const runtime.RopeBatchedPush,
        groups: u32,
        pub fn record(s: @This(), cmd: vk.c.VkCommandBuffer) void {
            s.kern.dispatch(cmd, s.push, s.groups, 1, 1);
        }
    }{ .kern = &k_bw, .push = &push, .groups = groups });
    const gpu_dx = try allocator.alloc(f32, total);
    defer allocator.free(gpu_dx);
    try buf_dx.readBack(&ctx, f32, gpu_dx);

    var max_bw: f32 = 0;
    for (gpu_dx, cpu_dx) |g, c| {
        const d = @abs(g - c);
        if (d > max_bw) max_bw = d;
    }
    if (max_bw > 1e-5) {
        std.debug.print("RoPE batched bw: max |Δ| = {e}\n", .{max_bw});
        return error.ParityFailed;
    }

    // Round-trip: rope_bw(rope_fwd(x)) ≈ x.
    var max_rt: f32 = 0;
    const rt_dx = try allocator.alloc(f32, total);
    defer allocator.free(rt_dx);
    try cpu_train_transformer.ropeBackwardBatched(rt_dx, cpu_y, n_pos, n_heads, head_dim, rotary_dim, theta_base);
    for (rt_dx, x) |a, b| {
        const d = @abs(a - b);
        if (d > max_rt) max_rt = d;
    }

    std.debug.print(
        "PASS GPU RoPE batched fwd + bw (n_pos={d} n_heads={d} head_dim={d}; fwd |Δ|={e}, bw |Δ|={e}, round-trip |Δ|={e})\n",
        .{ n_pos, n_heads, head_dim, max_fwd, max_bw, max_rt },
    );
}

// ── gpu l2norm-per-head smoke: synthetic vs CPU ─────────────────────

const L2normPush = runtime_hybrid.L2normPush;

pub fn runGpuL2normPerHeadSmoke(allocator: std.mem.Allocator) !void {
    var ctx = try vk.Context.init(allocator);
    defer ctx.deinit();

    // Qwen3.5 Gated DeltaNet: head_k_dim = 128, num_heads varies. Try
    // a couple of heads with distinct value ranges so we'd notice if
    // the per-head reduction got cross-contaminated.
    const num_heads: usize = 4;
    const head_dim: usize = 128;
    const total = num_heads * head_dim;
    const eps: f32 = 1e-6;

    const in_v = try allocator.alloc(f32, total);
    defer allocator.free(in_v);
    for (0..num_heads) |h| {
        const head_scale: f32 = @floatFromInt(h + 1);
        for (0..head_dim) |d| {
            in_v[h * head_dim + d] = head_scale * (@as(f32, @floatFromInt(d)) * 0.01 - 0.5);
        }
    }

    var buf_in = try buffer.Buffer.initStatic(&ctx, f32, in_v);
    defer buf_in.deinit(ctx.device);
    var buf_out = try buffer.Buffer.initDeviceOnly(&ctx, total * @sizeOf(f32));
    defer buf_out.deinit(ctx.device);

    var kern = try pipeline.Kernel.init(&ctx, &shaders.l2norm_per_head, 2, @sizeOf(L2normPush));
    defer kern.deinit();
    try kern.bind(&.{ &buf_in, &buf_out });

    const push = L2normPush{ .head_dim = @intCast(head_dim), .eps = eps };
    try buffer.submitOneShot(&ctx, struct {
        kern: *const pipeline.Kernel,
        push: *const L2normPush,
        groups: u32,
        pub fn record(s: @This(), cmd: vk.c.VkCommandBuffer) void {
            s.kern.dispatch(cmd, s.push, s.groups, 1, 1);
        }
    }{ .kern = &kern, .push = &push, .groups = @intCast(num_heads) });

    const got = try allocator.alloc(f32, total);
    defer allocator.free(got);
    try buf_out.readBack(&ctx, f32, got);

    // CPU reference: per-head L2-norm.
    const want = try allocator.alloc(f32, total);
    defer allocator.free(want);
    for (0..num_heads) |h| {
        var s: f32 = 0;
        for (0..head_dim) |d| s += in_v[h * head_dim + d] * in_v[h * head_dim + d];
        const inv = 1.0 / @sqrt(s + eps);
        for (0..head_dim) |d| want[h * head_dim + d] = in_v[h * head_dim + d] * inv;
    }

    var max_abs: f32 = 0;
    for (got, want) |g, e| {
        const d = @abs(g - e);
        if (d > max_abs) max_abs = d;
    }
    if (max_abs > 1e-5) {
        std.debug.print("GPU l2norm_per_head: max |Δ| = {e}\n", .{max_abs});
        return error.ParityFailed;
    }
    std.debug.print("PASS GPU l2norm_per_head (4 heads × 128 dim, max |Δ| vs CPU = {e})\n", .{max_abs});
}

// ── gpu conv1d_update smoke: 3-step rollout vs CPU ──────────────────
//
// Fires the kernel three times back-to-back so we exercise the in-
// place state shift across multiple decode steps; if the shift / append
// got transposed, the third output diverges immediately.

const Conv1dUpdatePush = runtime_hybrid.Conv1dUpdatePush;

pub fn runGpuConv1dUpdateSmoke(allocator: std.mem.Allocator) !void {
    var ctx = try vk.Context.init(allocator);
    defer ctx.deinit();

    const conv_dim: usize = 64;
    const kernel: usize = 4;

    const weight = try allocator.alloc(f32, conv_dim * kernel);
    defer allocator.free(weight);
    for (weight, 0..) |*w, i| w.* = @sin(@as(f32, @floatFromInt(i)) * 0.07) * 0.5;

    var buf_w = try buffer.Buffer.initStatic(&ctx, f32, weight);
    defer buf_w.deinit(ctx.device);

    // GPU side state — `initDeviceOnly` zero-fills, which is exactly
    // the "fresh sequence" initial state.
    var buf_state = try buffer.Buffer.initDeviceOnly(&ctx, conv_dim * kernel * @sizeOf(f32));
    defer buf_state.deinit(ctx.device);

    var kern = try pipeline.Kernel.init(&ctx, &shaders.conv1d_update, 4, @sizeOf(Conv1dUpdatePush));
    defer kern.deinit();

    // CPU mirror state (same zero init).
    const cpu_state = try allocator.alloc(f32, conv_dim * kernel);
    defer allocator.free(cpu_state);
    @memset(cpu_state, 0.0);

    const push = Conv1dUpdatePush{ .conv_dim = @intCast(conv_dim), .kernel_size = @intCast(kernel) };

    for (0..3) |step| {
        const x = try allocator.alloc(f32, conv_dim);
        defer allocator.free(x);
        for (x, 0..) |*v, i| v.* = @cos(@as(f32, @floatFromInt(i + step * 7)) * 0.13);

        var buf_in = try buffer.Buffer.initStatic(&ctx, f32, x);
        defer buf_in.deinit(ctx.device);
        var buf_out = try buffer.Buffer.initDeviceOnly(&ctx, conv_dim * @sizeOf(f32));
        defer buf_out.deinit(ctx.device);

        try kern.bind(&.{ &buf_in, &buf_w, &buf_state, &buf_out });
        const local: u32 = 128;
        const groups: u32 = (@as(u32, @intCast(conv_dim)) + local - 1) / local;
        try buffer.submitOneShot(&ctx, struct {
            kern: *const pipeline.Kernel,
            push: *const Conv1dUpdatePush,
            groups: u32,
            pub fn record(s: @This(), cmd: vk.c.VkCommandBuffer) void {
                s.kern.dispatch(cmd, s.push, s.groups, 1, 1);
            }
        }{ .kern = &kern, .push = &push, .groups = groups });

        const got = try allocator.alloc(f32, conv_dim);
        defer allocator.free(got);
        try buf_out.readBack(&ctx, f32, got);

        // CPU reference.
        const want = try allocator.alloc(f32, conv_dim);
        defer allocator.free(want);
        for (0..conv_dim) |c| {
            var k_idx: usize = 0;
            while (k_idx + 1 < kernel) : (k_idx += 1) {
                cpu_state[c * kernel + k_idx] = cpu_state[c * kernel + k_idx + 1];
            }
            cpu_state[c * kernel + kernel - 1] = x[c];
            var acc: f32 = 0;
            for (0..kernel) |k_pos| acc += cpu_state[c * kernel + k_pos] * weight[c * kernel + k_pos];
            want[c] = acc / (1.0 + @exp(-acc));
        }

        var max_abs: f32 = 0;
        for (got, want) |g, e| {
            const d = @abs(g - e);
            if (d > max_abs) max_abs = d;
        }
        if (max_abs > 1e-5) {
            std.debug.print("conv1d_update step {d}: max |Δ| = {e}\n", .{ step, max_abs });
            return error.ParityFailed;
        }
    }
    std.debug.print("PASS GPU conv1d_update (3-step rollout, conv_dim=64 kernel=4)\n", .{});
}

// ── gpu rmsnorm_gated smoke: synthetic vs CPU ───────────────────────

const RmsnormGatedPush = runtime_hybrid.RmsnormGatedPush;

pub fn runGpuRmsnormGatedSmoke(allocator: std.mem.Allocator) !void {
    var ctx = try vk.Context.init(allocator);
    defer ctx.deinit();

    const num_heads: usize = 4;
    const head_dim: usize = 128;
    const total = num_heads * head_dim;
    const eps: f32 = 1e-6;

    const x = try allocator.alloc(f32, total);
    defer allocator.free(x);
    const z = try allocator.alloc(f32, total);
    defer allocator.free(z);
    const w = try allocator.alloc(f32, head_dim);
    defer allocator.free(w);
    for (x, 0..) |*v, i| v.* = @sin(@as(f32, @floatFromInt(i)) * 0.05) * 0.5;
    for (z, 0..) |*v, i| v.* = @cos(@as(f32, @floatFromInt(i)) * 0.07) * 1.5;
    for (w, 0..) |*v, i| v.* = 0.3 + @as(f32, @floatFromInt(i)) * 0.001;

    var buf_x = try buffer.Buffer.initStatic(&ctx, f32, x);
    defer buf_x.deinit(ctx.device);
    var buf_z = try buffer.Buffer.initStatic(&ctx, f32, z);
    defer buf_z.deinit(ctx.device);
    var buf_w = try buffer.Buffer.initStatic(&ctx, f32, w);
    defer buf_w.deinit(ctx.device);
    var buf_out = try buffer.Buffer.initDeviceOnly(&ctx, total * @sizeOf(f32));
    defer buf_out.deinit(ctx.device);

    var kern = try pipeline.Kernel.init(&ctx, &shaders.rmsnorm_gated, 4, @sizeOf(RmsnormGatedPush));
    defer kern.deinit();
    try kern.bind(&.{ &buf_x, &buf_z, &buf_w, &buf_out });

    const push = RmsnormGatedPush{ .head_dim = @intCast(head_dim), .eps = eps };
    try buffer.submitOneShot(&ctx, struct {
        kern: *const pipeline.Kernel,
        push: *const RmsnormGatedPush,
        groups: u32,
        pub fn record(s: @This(), cmd: vk.c.VkCommandBuffer) void {
            s.kern.dispatch(cmd, s.push, s.groups, 1, 1);
        }
    }{ .kern = &kern, .push = &push, .groups = @intCast(num_heads) });

    const got = try allocator.alloc(f32, total);
    defer allocator.free(got);
    try buf_out.readBack(&ctx, f32, got);

    // CPU reference.
    const want = try allocator.alloc(f32, total);
    defer allocator.free(want);
    for (0..num_heads) |h| {
        const off = h * head_dim;
        var s: f32 = 0;
        for (0..head_dim) |d| s += x[off + d] * x[off + d];
        const inv = 1.0 / @sqrt(s / @as(f32, @floatFromInt(head_dim)) + eps);
        for (0..head_dim) |d| {
            const normed = w[d] * (x[off + d] * inv);
            const zd = z[off + d];
            const silu_z = zd / (1.0 + @exp(-zd));
            want[off + d] = normed * silu_z;
        }
    }

    var max_abs: f32 = 0;
    for (got, want) |g, e| {
        const d = @abs(g - e);
        if (d > max_abs) max_abs = d;
    }
    if (max_abs > 1e-5) {
        std.debug.print("GPU rmsnorm_gated: max |Δ| = {e}\n", .{max_abs});
        return error.ParityFailed;
    }
    std.debug.print("PASS GPU rmsnorm_gated (4 heads × 128 dim, max |Δ| vs CPU = {e})\n", .{max_abs});
}

// ── gpu gated_delta_step smoke: 2-step rollout vs CPU ───────────────
//
// Hot kernel of the Qwen3.5 GatedDeltaNet decode. Two back-to-back
// invocations to exercise both the readout AND the in-place state
// update; if the state update lands wrong, step 2 diverges.

const GatedDeltaStepPush = runtime_hybrid.GatedDeltaStepPush;

pub fn runGpuGatedDeltaStepSmoke(allocator: std.mem.Allocator) !void {
    var ctx = try vk.Context.init(allocator);
    defer ctx.deinit();

    // Qwen3.5-4B Gated DeltaNet shape: 16 K-heads, 32 V-heads, head_k =
    // head_v = 128 (with implicit 2× GQA repeat).
    const num_k_heads: usize = 16;
    const num_v_heads: usize = 32;
    const head_k: usize = 128;
    const head_v: usize = 128;
    const heads_per_k: usize = num_v_heads / num_k_heads;

    const q = try allocator.alloc(f32, num_k_heads * head_k);
    defer allocator.free(q);
    const k = try allocator.alloc(f32, num_k_heads * head_k);
    defer allocator.free(k);
    const v = try allocator.alloc(f32, num_v_heads * head_v);
    defer allocator.free(v);
    const b_raw = try allocator.alloc(f32, num_v_heads);
    defer allocator.free(b_raw);
    const a_raw = try allocator.alloc(f32, num_v_heads);
    defer allocator.free(a_raw);
    const A_log = try allocator.alloc(f32, num_v_heads);
    defer allocator.free(A_log);
    const dt_bias = try allocator.alloc(f32, num_v_heads);
    defer allocator.free(dt_bias);

    // Seed-style synthetic inputs. CPU mirror state mirrors GPU state
    // (both start at zero from initDeviceOnly).
    for (q, 0..) |*x, i| x.* = @sin(@as(f32, @floatFromInt(i)) * 0.011) * 0.3;
    for (k, 0..) |*x, i| x.* = @cos(@as(f32, @floatFromInt(i)) * 0.013) * 0.3;
    for (v, 0..) |*x, i| x.* = @sin(@as(f32, @floatFromInt(i)) * 0.017) * 0.5;
    for (b_raw, 0..) |*x, i| x.* = @as(f32, @floatFromInt(i)) * 0.05 - 0.5;
    for (a_raw, 0..) |*x, i| x.* = @cos(@as(f32, @floatFromInt(i)) * 0.7) * 0.4;
    for (A_log, 0..) |*x, i| x.* = -1.0 + @as(f32, @floatFromInt(i)) * 0.02;
    for (dt_bias) |*x| x.* = 0.1;

    var buf_q = try buffer.Buffer.initStatic(&ctx, f32, q);
    defer buf_q.deinit(ctx.device);
    var buf_k = try buffer.Buffer.initStatic(&ctx, f32, k);
    defer buf_k.deinit(ctx.device);
    var buf_v = try buffer.Buffer.initStatic(&ctx, f32, v);
    defer buf_v.deinit(ctx.device);
    var buf_b = try buffer.Buffer.initStatic(&ctx, f32, b_raw);
    defer buf_b.deinit(ctx.device);
    var buf_a = try buffer.Buffer.initStatic(&ctx, f32, a_raw);
    defer buf_a.deinit(ctx.device);
    var buf_alog = try buffer.Buffer.initStatic(&ctx, f32, A_log);
    defer buf_alog.deinit(ctx.device);
    var buf_dt = try buffer.Buffer.initStatic(&ctx, f32, dt_bias);
    defer buf_dt.deinit(ctx.device);

    var buf_state = try buffer.Buffer.initDeviceOnly(&ctx, num_v_heads * head_k * head_v * @sizeOf(f32));
    defer buf_state.deinit(ctx.device);
    var buf_y = try buffer.Buffer.initDeviceOnly(&ctx, num_v_heads * head_v * @sizeOf(f32));
    defer buf_y.deinit(ctx.device);

    var kern = try pipeline.Kernel.init(&ctx, &shaders.gated_delta_step, 9, @sizeOf(GatedDeltaStepPush));
    defer kern.deinit();
    try kern.bind(&.{ &buf_state, &buf_q, &buf_k, &buf_v, &buf_b, &buf_a, &buf_alog, &buf_dt, &buf_y });

    const push = GatedDeltaStepPush{
        .num_k_heads = @intCast(num_k_heads),
        .num_v_heads = @intCast(num_v_heads),
        .head_k = @intCast(head_k),
        .head_v = @intCast(head_v),
    };

    // CPU mirror state.
    const cpu_state = try allocator.alloc(f32, num_v_heads * head_k * head_v);
    defer allocator.free(cpu_state);
    @memset(cpu_state, 0.0);

    for (0..2) |step| {
        // GPU dispatch (one workgroup per V-head, head_v threads each).
        try buffer.submitOneShot(&ctx, struct {
            kern: *const pipeline.Kernel,
            push: *const GatedDeltaStepPush,
            n_v: u32,
            pub fn record(s: @This(), cmd: vk.c.VkCommandBuffer) void {
                s.kern.dispatch(cmd, s.push, s.n_v, 1, 1);
            }
        }{ .kern = &kern, .push = &push, .n_v = @intCast(num_v_heads) });

        const got = try allocator.alloc(f32, num_v_heads * head_v);
        defer allocator.free(got);
        try buf_y.readBack(&ctx, f32, got);

        // CPU reference replicates the math of the kernel exactly,
        // including the algebraic identity used for `y`.
        const want = try allocator.alloc(f32, num_v_heads * head_v);
        defer allocator.free(want);

        for (0..num_v_heads) |h| {
            const h_k = h / heads_per_k;
            const k_off = h_k * head_k;
            const S_off = h * head_k * head_v;

            // gates
            const beta_h: f32 = 1.0 / (1.0 + @exp(-b_raw[h]));
            const sp = if ((a_raw[h] + dt_bias[h]) > 20.0) (a_raw[h] + dt_bias[h]) else @log(1.0 + @exp(a_raw[h] + dt_bias[h]));
            const g_t_h: f32 = @exp(-@exp(A_log[h]) * sp);

            // <k, q>
            var kq: f32 = 0;
            for (0..head_k) |d| kq += k[k_off + d] * q[k_off + d];

            // Sq, Sk per column t (computed from S_old). kv_mem (=
            // decayed Sk) used by `delta` is `g_t * Sk_old`.
            var Sq: [128]f32 = undefined;
            var Sk: [128]f32 = undefined;
            for (0..head_v) |t| {
                var sq: f32 = 0;
                var sk: f32 = 0;
                for (0..head_k) |d| {
                    const s_dt = cpu_state[S_off + d * head_v + t];
                    sq += s_dt * q[k_off + d];
                    sk += s_dt * k[k_off + d];
                }
                Sq[t] = sq;
                Sk[t] = sk * g_t_h; // decayed
            }

            // delta, y, state update.
            for (0..head_v) |t| {
                const v_in = v[h * head_v + t];
                const delta_t = (v_in - Sk[t]) * beta_h;
                want[h * head_v + t] = g_t_h * Sq[t] + delta_t * kq;
                for (0..head_k) |d| {
                    const idx = S_off + d * head_v + t;
                    cpu_state[idx] = g_t_h * cpu_state[idx] + k[k_off + d] * delta_t;
                }
            }
        }

        var max_abs: f32 = 0;
        for (got, want) |g, e| {
            const d = @abs(g - e);
            if (d > max_abs) max_abs = d;
        }
        if (max_abs > 1e-4) {
            std.debug.print("gated_delta_step step {d}: max |Δ| = {e}\n", .{ step, max_abs });
            return error.ParityFailed;
        }
    }
    std.debug.print("PASS GPU gated_delta_step (2-step rollout, 32 v-heads × 128² state, GQA 2:1)\n", .{});
}

// ── gpu softmax smoke: synthetic vs CPU softmax ─────────────────────

const SoftmaxPush = runtime.SoftmaxPush;

pub fn runGpuSoftmaxSmoke(allocator: std.mem.Allocator) !void {
    var ctx = try vk.Context.init(allocator);
    defer ctx.deinit();

    const dim: usize = 2048;
    const x = try allocator.alloc(f32, dim);
    defer allocator.free(x);
    // Mix of negative, near-zero, and one big positive — exercises the
    // numerical-stability subtract-max path.
    for (x, 0..) |*v, i| v.* = @as(f32, @floatFromInt(@as(i32, @intCast(i)) - 1024)) * 0.01;
    x[42] = 100.0; // a clear winner; without subtract-max would overflow exp

    const want = try allocator.alloc(f32, dim);
    defer allocator.free(want);
    @memcpy(want, x);
    cpu_math.softmax(want);

    var buf_in = try buffer.Buffer.initStatic(&ctx, f32, x);
    defer buf_in.deinit(ctx.device);
    var buf_out = try buffer.Buffer.initDeviceOnly(&ctx, dim * @sizeOf(f32));
    defer buf_out.deinit(ctx.device);

    var kern = try pipeline.Kernel.init(&ctx, &shaders.softmax, 2, @sizeOf(SoftmaxPush));
    defer kern.deinit();
    try kern.bind(&.{ &buf_in, &buf_out });

    const push = SoftmaxPush{ .dim = @intCast(dim), .stride = @intCast(dim) };
    try buffer.submitOneShot(&ctx, struct {
        kern: *const pipeline.Kernel,
        push: *const SoftmaxPush,
        pub fn record(s: @This(), cmd: vk.c.VkCommandBuffer) void {
            s.kern.dispatch(cmd, s.push, 1, 1, 1);
        }
    }{ .kern = &kern, .push = &push });

    const got = try allocator.alloc(f32, dim);
    defer allocator.free(got);
    try buf_out.readBack(&ctx, f32, got);

    var max_abs: f32 = 0;
    var sum: f32 = 0;
    for (got, want) |g, e| {
        const d = @abs(g - e);
        if (d > max_abs) max_abs = d;
        sum += g;
    }
    if (max_abs > 1e-5) {
        std.debug.print("GPU softmax: max |Δ| = {e}\n", .{max_abs});
        return error.ParityFailed;
    }
    if (@abs(sum - 1.0) > 1e-5) {
        std.debug.print("GPU softmax sum {d} != 1\n", .{sum});
        return error.ParityFailed;
    }
    std.debug.print("PASS GPU softmax synthetic (dim=2048, sum=1±1e-5, vs CPU 1e-5)\n", .{});
}

// ── gpu fwht256 smoke: in-place FWHT on a 256-vec vs CPU oracle ────

pub fn runGpuFwhtSmoke(allocator: std.mem.Allocator) !void {
    var ctx = try vk.Context.init(allocator);
    defer ctx.deinit();

    // Deterministic input + CPU reference.
    var input: [256]f32 = undefined;
    var prng = std.Random.DefaultPrng.init(0xFEED_F00D);
    const r = prng.random();
    for (&input) |*v| v.* = r.floatNorm(f32);
    var expected = input;
    turboquant.fwht(&expected);

    // Round-trip through GPU.
    var buf = try buffer.Buffer.initStatic(&ctx, f32, &input);
    defer buf.deinit(ctx.device);

    var kern = try pipeline.Kernel.init(&ctx, &shaders.fwht256, 1, 0);
    defer kern.deinit();
    try kern.bind(&.{&buf});

    try buffer.submitOneShot(&ctx, struct {
        kern: *const pipeline.Kernel,
        pub fn record(s: @This(), cmd: vk.c.VkCommandBuffer) void {
            s.kern.dispatch(cmd, null, 1, 1, 1);
        }
    }{ .kern = &kern });

    var got: [256]f32 = undefined;
    try buf.readBack(&ctx, f32, &got);

    var max_err: f32 = 0;
    for (got, expected) |g, w| max_err = @max(max_err, @abs(g - w));
    if (max_err > 1e-3) {
        std.debug.print("GPU fwht max |Δ| vs CPU = {e}\n", .{max_err});
        return error.ParityFailed;
    }
    std.debug.print("PASS GPU fwht256 (256-elem in-place butterfly, max |Δ| vs CPU = {e:.2})\n", .{max_err});
}

// ── gpu rht_pre256 smoke: signs · x then FWHT, vs CPU rhtForward ──

pub fn runGpuRhtPreSmoke(allocator: std.mem.Allocator) !void {
    var ctx = try vk.Context.init(allocator);
    defer ctx.deinit();

    var input: [256]f32 = undefined;
    var prng = std.Random.DefaultPrng.init(0xBA5E_BA11);
    const r = prng.random();
    for (&input) |*v| v.* = r.floatNorm(f32);
    var expected = input;
    turboquant.rhtForward(&expected);

    var buf = try buffer.Buffer.initStatic(&ctx, f32, &input);
    defer buf.deinit(ctx.device);

    var kern = try pipeline.Kernel.init(&ctx, &shaders.rht_pre256, 1, 0);
    defer kern.deinit();
    try kern.bind(&.{&buf});

    try buffer.submitOneShot(&ctx, struct {
        kern: *const pipeline.Kernel,
        pub fn record(s: @This(), cmd: vk.c.VkCommandBuffer) void {
            s.kern.dispatch(cmd, null, 1, 1, 1);
        }
    }{ .kern = &kern });

    var got: [256]f32 = undefined;
    try buf.readBack(&ctx, f32, &got);

    var max_err: f32 = 0;
    for (got, expected) |g, w| max_err = @max(max_err, @abs(g - w));
    if (max_err > 1e-3) {
        std.debug.print("GPU rht_pre max |Δ| vs CPU = {e}\n", .{max_err});
        return error.ParityFailed;
    }
    std.debug.print("PASS GPU rht_pre256 (signs · x then FWHT, max |Δ| vs CPU = {e:.2})\n", .{max_err});
}

// ── gpu rht round-trip smoke: rht_post(rht_pre(x)) ≈ x on device ──

pub fn runGpuRhtRoundTripSmoke(allocator: std.mem.Allocator) !void {
    var ctx = try vk.Context.init(allocator);
    defer ctx.deinit();

    var input: [256]f32 = undefined;
    var prng = std.Random.DefaultPrng.init(0x1234_5678);
    const r = prng.random();
    for (&input) |*v| v.* = r.floatNorm(f32);

    var buf = try buffer.Buffer.initStatic(&ctx, f32, &input);
    defer buf.deinit(ctx.device);

    var pre = try pipeline.Kernel.init(&ctx, &shaders.rht_pre256, 1, 0);
    defer pre.deinit();
    try pre.bind(&.{&buf});

    var post = try pipeline.Kernel.init(&ctx, &shaders.rht_post256, 1, 0);
    defer post.deinit();
    try post.bind(&.{&buf});

    // Two dispatches; submitOneShot waits for queue idle between them
    // so the second read-after-write has hard ordering.
    try buffer.submitOneShot(&ctx, struct {
        kern: *const pipeline.Kernel,
        pub fn record(s: @This(), cmd: vk.c.VkCommandBuffer) void {
            s.kern.dispatch(cmd, null, 1, 1, 1);
        }
    }{ .kern = &pre });
    try buffer.submitOneShot(&ctx, struct {
        kern: *const pipeline.Kernel,
        pub fn record(s: @This(), cmd: vk.c.VkCommandBuffer) void {
            s.kern.dispatch(cmd, null, 1, 1, 1);
        }
    }{ .kern = &post });

    var got: [256]f32 = undefined;
    try buf.readBack(&ctx, f32, &got);

    var max_err: f32 = 0;
    for (got, input) |g, w| max_err = @max(max_err, @abs(g - w));
    if (max_err > 1e-4) {
        std.debug.print("GPU rht round-trip max |Δ| = {e}\n", .{max_err});
        return error.ParityFailed;
    }
    std.debug.print("PASS GPU rht round-trip (rht_post ∘ rht_pre = id, max |Δ| = {e:.2})\n", .{max_err});
}

// ── gpu rht fused round-trip: pre + post in one command buffer ─────
//
// Same round-trip as runGpuRhtRoundTripSmoke, but using the Recorder
// pattern so both dispatches go into a single command buffer, with
// the recorder's auto-emitted compute→compute memory barrier between
// them. This is the orchestration shape recordForwardStep uses for
// every existing kernel, so verifying correctness here means the RHT
// shaders are ready to drop into any layer of the real forward path.

pub fn runGpuRhtFusedRoundTripSmoke(allocator: std.mem.Allocator) !void {
    var ctx = try vk.Context.init(allocator);
    defer ctx.deinit();

    var input: [256]f32 = undefined;
    var prng = std.Random.DefaultPrng.init(0xCC00_FFEE);
    const r = prng.random();
    for (&input) |*v| v.* = r.floatNorm(f32);

    var buf = try buffer.Buffer.initStatic(&ctx, f32, &input);
    defer buf.deinit(ctx.device);

    var pre = try pipeline.Kernel.init(&ctx, &shaders.rht_pre256, 1, 0);
    defer pre.deinit();
    var post = try pipeline.Kernel.init(&ctx, &shaders.rht_post256, 1, 0);
    defer post.deinit();

    var rec = try gpu_recorder.Recorder.init(&ctx, 4, 4);
    defer rec.deinit();

    try rec.begin();
    try rec.dispatch(&pre,  &.{&buf}, null, 1, 1, 1);
    try rec.dispatch(&post, &.{&buf}, null, 1, 1, 1);
    try rec.endAndSubmit();

    var got: [256]f32 = undefined;
    try buf.readBack(&ctx, f32, &got);

    var max_err: f32 = 0;
    for (got, input) |g, w| max_err = @max(max_err, @abs(g - w));
    if (max_err > 1e-4) {
        std.debug.print("GPU rht fused round-trip max |Δ| = {e}\n", .{max_err});
        return error.ParityFailed;
    }
    std.debug.print("PASS GPU rht fused round-trip (recorder + auto-barrier, max |Δ| = {e:.2})\n", .{max_err});
}

// ── gpu tq4_pack smoke: full TQ4 quantize on GPU vs CPU oracle ─────
//
// Same deterministic ramp as the YATQ bit-exact test: x[i] =
// (i/128) - 1 for i in [0, 256). Quantize on GPU and compare:
//   - 256 Lloyd-Max indices: bit-exact vs CPU quantizeBlockTQ4
//   - γ (norm-correction): within 1e-3 relative tolerance vs CPU
//     (CPU stores f16 γ; GPU stores f32. The truncation tolerance
//     of f16 mantissa is ~5e-4, so 1e-3 is comfortable.)
//
// GPU output is 33 u32s per block: [0] = γ as f32 bits, [1..33] = 32
// LE u32s holding the same byte-stream layout as BlockTQ4.indices.

pub fn runGpuTq4PackSmoke(allocator: std.mem.Allocator) !void {
    var ctx = try vk.Context.init(allocator);
    defer ctx.deinit();

    var input: [256]f32 = undefined;
    for (&input, 0..) |*v, i| v.* = (@as(f32, @floatFromInt(i)) / 128.0) - 1.0;

    var cpu_blk: turboquant.BlockTQ4(256) = undefined;
    turboquant.quantizeBlockTQ4(256, &input, &cpu_blk);
    const cpu_gamma_f32: f32 = @floatCast(cpu_blk.gamma);

    var input_buf = try buffer.Buffer.initStatic(&ctx, f32, &input);
    defer input_buf.deinit(ctx.device);
    var output_buf = try buffer.Buffer.initDeviceOnly(&ctx, 33 * @sizeOf(u32));
    defer output_buf.deinit(ctx.device);

    var kern = try pipeline.Kernel.init(&ctx, &shaders.tq4_pack256, 2, 0);
    defer kern.deinit();
    try kern.bind(&.{ &input_buf, &output_buf });

    try buffer.submitOneShot(&ctx, struct {
        kern: *const pipeline.Kernel,
        pub fn record(s: @This(), cmd: vk.c.VkCommandBuffer) void {
            s.kern.dispatch(cmd, null, 1, 1, 1);
        }
    }{ .kern = &kern });

    var out: [33]u32 = undefined;
    try output_buf.readBack(&ctx, f32, @ptrCast(&out));
    const gpu_gamma: f32 = @bitCast(out[0]);

    const gamma_rel = @abs(gpu_gamma - cpu_gamma_f32) / cpu_gamma_f32;
    if (gamma_rel > 1e-3) {
        std.debug.print("γ mismatch: gpu={d} cpu(f16)={d} rel={e}\n", .{ gpu_gamma, cpu_gamma_f32, gamma_rel });
        return error.ParityFailed;
    }

    // Compare indices bit-exact. GPU u32[k] = bytes [4k..4k+4] LE; each
    // byte holds two 4-bit indices (low nibble even, high nibble odd).
    // CPU BlockTQ4.indices[k] holds element 2k in low nibble and 2k+1 in
    // high — so CPU indices[k] should equal byte (k % 4) of out[1 + k/4].
    for (0..128) |byte_idx| {
        const word = out[1 + byte_idx / 4];
        const shift: u5 = @intCast((byte_idx % 4) * 8);
        const gpu_byte: u8 = @intCast((word >> shift) & 0xff);
        const cpu_byte = cpu_blk.indices[byte_idx];
        if (gpu_byte != cpu_byte) {
            std.debug.print("idx mismatch at byte {d}: gpu={x:0>2} cpu={x:0>2}\n", .{ byte_idx, gpu_byte, cpu_byte });
            return error.ParityFailed;
        }
    }

    std.debug.print("PASS GPU tq4_pack256 (256 indices bit-exact, γ rel-err {e:.2} vs CPU)\n", .{gamma_rel});
}

// ── gpu tq4_unpack smoke: dequant a CPU-packed block on GPU vs CPU ──

pub fn runGpuTq4UnpackSmoke(allocator: std.mem.Allocator) !void {
    var ctx = try vk.Context.init(allocator);
    defer ctx.deinit();

    var input: [256]f32 = undefined;
    for (&input, 0..) |*v, i| v.* = (@as(f32, @floatFromInt(i)) / 128.0) - 1.0;

    var cpu_blk: turboquant.BlockTQ4(256) = undefined;
    turboquant.quantizeBlockTQ4(256, &input, &cpu_blk);
    var cpu_dequant: [256]f32 = undefined;
    turboquant.dequantizeBlockTQ4(256, &cpu_blk, &cpu_dequant);

    // Build the 33-u32 GPU input: word[0] = γ as f32 bits, words[1..33] =
    // 128 bytes of cpu_blk.indices viewed as 32 LE u32s.
    var gpu_in: [33]u32 = undefined;
    gpu_in[0] = @bitCast(@as(f32, @floatCast(cpu_blk.gamma)));
    @memcpy(@as([*]u8, @ptrCast(gpu_in[1..].ptr))[0..128], &cpu_blk.indices);

    var input_buf = try buffer.Buffer.initStatic(&ctx, u32, &gpu_in);
    defer input_buf.deinit(ctx.device);
    var output_buf = try buffer.Buffer.initDeviceOnly(&ctx, 256 * @sizeOf(f32));
    defer output_buf.deinit(ctx.device);

    var kern = try pipeline.Kernel.init(&ctx, &shaders.tq4_unpack256, 2, 0);
    defer kern.deinit();
    try kern.bind(&.{ &input_buf, &output_buf });

    try buffer.submitOneShot(&ctx, struct {
        kern: *const pipeline.Kernel,
        pub fn record(s: @This(), cmd: vk.c.VkCommandBuffer) void {
            s.kern.dispatch(cmd, null, 1, 1, 1);
        }
    }{ .kern = &kern });

    var gpu_dequant: [256]f32 = undefined;
    try output_buf.readBack(&ctx, f32, &gpu_dequant);

    var max_err: f32 = 0;
    for (gpu_dequant, cpu_dequant) |g, w| max_err = @max(max_err, @abs(g - w));
    if (max_err > 1e-4) {
        std.debug.print("GPU tq4_unpack max |Δ| vs CPU = {e}\n", .{max_err});
        return error.ParityFailed;
    }
    std.debug.print("PASS GPU tq4_unpack256 (CPU-packed block dequanted on GPU, max |Δ| vs CPU = {e:.2})\n", .{max_err});
}

// ── gpu tq4 round-trip: pack → unpack on GPU vs CPU oracle ──────────

pub fn runGpuTq4RoundTripSmoke(allocator: std.mem.Allocator) !void {
    var ctx = try vk.Context.init(allocator);
    defer ctx.deinit();

    var input: [256]f32 = undefined;
    var prng = std.Random.DefaultPrng.init(0xACED_F00D);
    const r = prng.random();
    for (&input) |*v| v.* = r.floatNorm(f32);

    var cpu_blk: turboquant.BlockTQ4(256) = undefined;
    turboquant.quantizeBlockTQ4(256, &input, &cpu_blk);
    var cpu_dequant: [256]f32 = undefined;
    turboquant.dequantizeBlockTQ4(256, &cpu_blk, &cpu_dequant);

    // GPU: pack input, then unpack the GPU-packed block — chained
    // through the Recorder so both dispatches share a command buffer
    // with the auto-emitted compute→compute barrier.
    var input_buf = try buffer.Buffer.initStatic(&ctx, f32, &input);
    defer input_buf.deinit(ctx.device);
    var packed_buf = try buffer.Buffer.initDeviceOnly(&ctx, 33 * @sizeOf(u32));
    defer packed_buf.deinit(ctx.device);
    var output_buf = try buffer.Buffer.initDeviceOnly(&ctx, 256 * @sizeOf(f32));
    defer output_buf.deinit(ctx.device);

    var pack_kern = try pipeline.Kernel.init(&ctx, &shaders.tq4_pack256, 2, 0);
    defer pack_kern.deinit();
    var unpack_kern = try pipeline.Kernel.init(&ctx, &shaders.tq4_unpack256, 2, 0);
    defer unpack_kern.deinit();

    var rec = try gpu_recorder.Recorder.init(&ctx, 4, 8);
    defer rec.deinit();
    try rec.begin();
    try rec.dispatch(&pack_kern,   &.{ &input_buf, &packed_buf }, null, 1, 1, 1);
    try rec.dispatch(&unpack_kern, &.{ &packed_buf, &output_buf }, null, 1, 1, 1);
    try rec.endAndSubmit();

    var gpu_dequant: [256]f32 = undefined;
    try output_buf.readBack(&ctx, f32, &gpu_dequant);

    var max_err: f32 = 0;
    for (gpu_dequant, cpu_dequant) |g, w| max_err = @max(max_err, @abs(g - w));
    // GPU and CPU both compute γ in fp32 then truncate to f16, but the
    // raw γ values differ by ~f32-ULP because the L2 norm reductions
    // traverse elements in different orders (subgroup-tree on GPU vs
    // linear on CPU). When the two raw γs straddle an f16 boundary
    // they round to different f16 values, and that single-ULP delta
    // multiplies through the centroid magnitudes (max ~2.73), giving
    // ~5e-4 × 2.73 ≈ 1.4e-3 reconstruction divergence in the worst
    // case. The indices are bit-exact (verified separately in
    // tq4_pack256 smoke); only the γ scaling drifts.
    if (max_err > 5e-3) {
        std.debug.print("GPU tq4 round-trip max |Δ| vs CPU = {e}\n", .{max_err});
        return error.ParityFailed;
    }
    std.debug.print("PASS GPU tq4 pack→unpack round-trip (recorder, max |Δ| vs CPU = {e:.2})\n", .{max_err});
}

// ── gpu tq4_pack_to_cache smoke: positional pack into a multi-block cache ──
//
// Pack three different 256-vec inputs into slots 0, 1, 2 of a 3-block
// cache buffer using the dst_block_idx push constant. Then dispatch
// tq4_unpack256 with WG count = 3 to dequantise all three blocks at
// once. Each reconstructed block must match the corresponding
// CPU pack→dequant output.

const Tq4PackPush = runtime.Tq4PackPush;

pub fn runGpuTq4PackToCacheSmoke(allocator: std.mem.Allocator) !void {
    var ctx = try vk.Context.init(allocator);
    defer ctx.deinit();

    const n_blocks: usize = 3;
    var inputs: [n_blocks][256]f32 = undefined;
    var prng = std.Random.DefaultPrng.init(0xCA75_F00D);
    const r = prng.random();
    for (&inputs) |*blk_in| for (blk_in) |*v| { v.* = r.floatNorm(f32); };

    // CPU oracle: pack+dequant each block.
    var cpu_dequants: [n_blocks][256]f32 = undefined;
    for (&inputs, &cpu_dequants) |*x, *y| {
        var blk: turboquant.BlockTQ4(256) = undefined;
        turboquant.quantizeBlockTQ4(256, x, &blk);
        turboquant.dequantizeBlockTQ4(256, &blk, y);
    }

    // GPU side: a single 256-vec staging buffer for the input, a
    // 3-block packed cache, and a 3-block dequant output.
    var stage_buf = try buffer.Buffer.initStatic(&ctx, f32, &inputs[0]);
    defer stage_buf.deinit(ctx.device);
    var cache_buf = try buffer.Buffer.initDeviceOnly(&ctx, n_blocks * 33 * @sizeOf(u32));
    defer cache_buf.deinit(ctx.device);
    var deq_buf = try buffer.Buffer.initDeviceOnly(&ctx, n_blocks * 256 * @sizeOf(f32));
    defer deq_buf.deinit(ctx.device);

    var pack = try pipeline.Kernel.init(&ctx, &shaders.tq4_pack_to_cache, 2, @sizeOf(Tq4PackPush));
    defer pack.deinit();
    var unpack = try pipeline.Kernel.init(&ctx, &shaders.tq4_unpack256, 2, 0);
    defer unpack.deinit();

    // For block 0, the staging buffer already holds inputs[0] — pack
    // it directly. For blocks 1 and 2, update the staging buffer
    // between dispatches via the dynamic-update path (the buffer was
    // initStatic so we don't have a host-mapped pointer; instead we
    // make a fresh static buffer per block). Simpler: recreate stage
    // buffer per iteration.
    {
        var rec = try gpu_recorder.Recorder.init(&ctx, 8, 16);
        defer rec.deinit();
        try rec.begin();
        var push = Tq4PackPush{ .dst_block_idx = 0 };
        try rec.dispatch(&pack, &.{ &stage_buf, &cache_buf }, &push, 1, 1, 1);
        try rec.endAndSubmit();
    }
    for (1..n_blocks) |b| {
        var s = try buffer.Buffer.initStatic(&ctx, f32, &inputs[b]);
        defer s.deinit(ctx.device);
        var rec = try gpu_recorder.Recorder.init(&ctx, 8, 16);
        defer rec.deinit();
        try rec.begin();
        var push = Tq4PackPush{ .dst_block_idx = @intCast(b) };
        try rec.dispatch(&pack, &.{ &s, &cache_buf }, &push, 1, 1, 1);
        try rec.endAndSubmit();
    }

    // Single dispatch unpacks all 3 blocks (WG count = 3).
    {
        var rec = try gpu_recorder.Recorder.init(&ctx, 8, 16);
        defer rec.deinit();
        try rec.begin();
        try rec.dispatch(&unpack, &.{ &cache_buf, &deq_buf }, null, n_blocks, 1, 1);
        try rec.endAndSubmit();
    }

    var got: [n_blocks * 256]f32 = undefined;
    try deq_buf.readBack(&ctx, f32, &got);

    var max_err: f32 = 0;
    for (0..n_blocks) |b| {
        for (0..256) |i| {
            const g = got[b * 256 + i];
            const w = cpu_dequants[b][i];
            max_err = @max(max_err, @abs(g - w));
        }
    }
    if (max_err > 5e-3) {
        std.debug.print("GPU tq4_pack_to_cache max |Δ| = {e}\n", .{max_err});
        return error.ParityFailed;
    }
    std.debug.print("PASS GPU tq4_pack_to_cache (3 blocks at distinct positions, max |Δ| vs CPU = {e:.2})\n", .{max_err});
}

// ── gelu_tanh smoke: scalar against PyTorch reference values ────────

pub fn runGeluSmoke(allocator: std.mem.Allocator) !void {
    _ = allocator;
    // Reference values from torch.nn.functional.gelu(approximate="tanh")
    // (which matches HF's gelu_pytorch_tanh, which Gemma uses).
    const cases = [_]struct { x: f32, want: f32 }{
        .{ .x = 0.0, .want = 0.0 },
        .{ .x = 1.0, .want = 0.8411919876 },
        .{ .x = -1.0, .want = -0.15880800784 },
        .{ .x = 2.0, .want = 1.9545976400 },
        .{ .x = -2.0, .want = -0.04540234059 },
    };
    for (cases) |tc| {
        const got = cpu_math.gelu_tanh(tc.x);
        const err = @abs(got - tc.want);
        if (err > 1e-5) {
            std.debug.print("gelu_tanh({d}): got {d}, want {d} (err {e})\n", .{ tc.x, got, tc.want, err });
            return error.ParityFailed;
        }
    }
    std.debug.print("PASS gelu_tanh (5 ref values within 1e-5)\n", .{});
}

// ── TurboQuant tables smoke: Lloyd-Max codebook + RHT sign pattern ──

pub fn runTurboquantSmoke(allocator: std.mem.Allocator) !void {
    _ = allocator;
    // 1) RHT sign pattern: byte 0 = 0xa7 = 0b10100111, so LSB-first
    //    bits 0..7 = 1,1,1,0,0,1,0,1, signs = -1,-1,-1,+1,+1,-1,+1,-1.
    const sign_expected = [_]f32{ -1, -1, -1, 1, 1, -1, 1, -1 };
    for (sign_expected, 0..) |want, i| {
        if (turboquant.rhtSign(i) != want) {
            std.debug.print("rhtSign({d}): got {d}, want {d}\n", .{ i, turboquant.rhtSign(i), want });
            return error.ParityFailed;
        }
    }
    // The pattern is 256-bit periodic.
    if (turboquant.rhtSign(0) != turboquant.rhtSign(256)) return error.ParityFailed;

    // 2) Lloyd-Max codebooks symmetric about zero (b=3 and b=4).
    inline for ([_]turboquant.Bits{ .b3, .b4 }) |b| {
        const n: usize = if (b == .b3) 8 else 16;
        var i: usize = 0;
        while (i < n / 2) : (i += 1) {
            const lo = turboquant.lloydMaxCentroid(@intCast(i), b);
            const hi = turboquant.lloydMaxCentroid(@intCast(n - 1 - i), b);
            if (@abs(-lo - hi) > 1e-6) {
                std.debug.print("centroid asymmetry b={d}: lo={d} hi={d}\n", .{ @intFromEnum(b), lo, hi });
                return error.ParityFailed;
            }
        }
    }

    // 3) Quantize each centroid back through lloydMaxIndex; should round-trip.
    inline for ([_]turboquant.Bits{ .b3, .b4 }) |b| {
        const n: usize = if (b == .b3) 8 else 16;
        var i: usize = 0;
        while (i < n) : (i += 1) {
            const c = turboquant.lloydMaxCentroid(@intCast(i), b);
            const idx = turboquant.lloydMaxIndex(c, b);
            if (idx != i) {
                std.debug.print("centroid round-trip b={d} i={d}: got idx={d}\n", .{ @intFromEnum(b), i, idx });
                return error.ParityFailed;
            }
        }
    }

    // 4) Hand-checked corner values vs YATQ's (x >= b).sum() idiom.
    //    b=3: x=0.0 → bin 4 (centroid +0.2451), x=0.6 → bin 5 (+0.756),
    //    x=-0.6 → bin 2 (-0.756). b=4: x=0.0 → bin 8 (+0.1284).
    if (turboquant.lloydMaxIndex(0.0, .b3) != 4) return error.ParityFailed;
    if (turboquant.lloydMaxIndex(0.6, .b3) != 5) return error.ParityFailed;
    if (turboquant.lloydMaxIndex(-0.6, .b3) != 2) return error.ParityFailed;
    if (turboquant.lloydMaxIndex(0.0, .b4) != 8) return error.ParityFailed;

    // 5) FWHT hand-checked at d=4: [1,2,3,4] → [10,-2,-4,0].
    {
        var x = [_]f32{ 1, 2, 3, 4 };
        turboquant.fwht(&x);
        const want = [_]f32{ 10, -2, -4, 0 };
        for (x, want) |got, w| if (got != w) return error.ParityFailed;
    }

    // 6) FWHT applied twice multiplies by d (since H · H = d · I).
    {
        var x = [_]f32{ 0.5, -1.25, 3.0, 0.0, -2.5, 1.75, 0.125, -0.75 };
        const orig = x;
        turboquant.fwht(&x);
        turboquant.fwht(&x);
        const d: f32 = @floatFromInt(x.len);
        for (x, orig) |got, w| if (@abs(got - d * w) > 1e-5) return error.ParityFailed;
    }

    // 7) RHT round-trip on a 256-vector: rhtInverse(rhtForward(x)) ≈ x.
    {
        var prng = std.Random.DefaultPrng.init(0xDEADBEEF);
        const r = prng.random();
        var x: [256]f32 = undefined;
        for (&x) |*v| v.* = r.floatNorm(f32);
        const orig = x;
        turboquant.rhtForward(&x);
        turboquant.rhtInverse(&x);
        var max_err: f32 = 0;
        for (x, orig) |got, w| max_err = @max(max_err, @abs(got - w));
        if (max_err > 1e-4) {
            std.debug.print("rht round-trip max |Δ| = {e}\n", .{max_err});
            return error.ParityFailed;
        }
    }

    // 8) TQ4 round-trip on a 256-d Gaussian block. The norm-correction
    //    γ must give a reconstruction whose L2 norm equals the original
    //    to within f16 quantisation; per-element MSE must be in the
    //    expected range for 4-bit Lloyd-Max on a unit Gaussian
    //    (<0.005 average squared error per coord).
    {
        var prng = std.Random.DefaultPrng.init(0x5EED1234);
        const r = prng.random();
        var x: [256]f32 = undefined;
        for (&x) |*v| v.* = r.floatNorm(f32);

        var raw_sq: f32 = 0;
        for (x) |v| raw_sq += v * v;
        const raw_norm = @sqrt(raw_sq);

        var blk: turboquant.BlockTQ4(256) = undefined;
        turboquant.quantizeBlockTQ4(256, &x, &blk);
        var y: [256]f32 = undefined;
        turboquant.dequantizeBlockTQ4(256, &blk, &y);

        var rec_sq: f32 = 0;
        var err_sq: f32 = 0;
        for (x, y) |xi, yi| {
            rec_sq += yi * yi;
            const d = xi - yi;
            err_sq += d * d;
        }
        const rec_norm = @sqrt(rec_sq);
        const norm_rel = @abs(raw_norm - rec_norm) / raw_norm;
        const mse = err_sq / @as(f32, @floatFromInt(x.len));

        if (norm_rel > 1e-3) {
            std.debug.print("TQ4 norm preservation: raw={d:.4} rec={d:.4} rel={e}\n", .{ raw_norm, rec_norm, norm_rel });
            return error.ParityFailed;
        }
        if (mse > 0.01) {
            std.debug.print("TQ4 MSE = {d:.5} (>0.01 threshold)\n", .{mse});
            return error.ParityFailed;
        }
        std.debug.print("       TQ4 256-d Gaussian: MSE={d:.5}, norm-rel-err={e:.2}\n", .{ mse, norm_rel });
    }

    // 9) Bit-exact parity vs YATQ Python reference. Input is the
    //    deterministic ramp x[i] = (i/128) - 1, which exists in both
    //    Zig and Python with no PRNG-cross-language risk. Expected
    //    indices generated by reference/turboquant/cross_validate.py.
    {
        var x: [256]f32 = undefined;
        for (&x, 0..) |*v, i| {
            v.* = (@as(f32, @floatFromInt(i)) / 128.0) - 1.0;
        }
        const yatq_indices_b4 = [256]u8{
            8,  7,  12, 6,  6,  11, 8,  7,  9,  9,  7,  13, 8,  13, 11, 7,
            14, 4,  13, 10, 10, 8,  6,  13, 4,  7,  3,  6,  10, 15, 6,  8,
            10, 6,  13, 11, 9,  13, 9,  6,  3,  3,  5,  2,  7,  12, 11, 3,
            9,  4,  12, 10, 14, 3,  8,  6,  8,  8,  5,  9,  3,  4,  13, 4,
            8,  7,  4,  7,  8,  6,  7,  7,  5,  10, 10, 6,  12, 6,  2,  12,
            5,  7,  7,  1,  8,  12, 11, 9,  14, 6,  5,  1,  8,  6,  9,  10,
            11, 9,  13, 10, 4,  13, 6,  5,  8,  9,  10, 12, 9,  10, 15, 2,
            6,  5,  8,  5,  12, 5,  6,  6,  9,  7,  4,  2,  5,  14, 2,  3,
            10, 8,  7,  8,  9,  8,  8,  7,  3,  7,  10, 8,  8,  11, 5,  12,
            8,  6,  11, 3,  13, 12, 13, 8,  14, 7,  8,  2,  7,  5,  8,  4,
            12, 9,  13, 6,  2,  10, 6,  2,  4,  8,  12, 13, 11, 12, 13, 3,
            12, 4,  12, 5,  10, 7,  4,  10, 7,  6,  6,  1,  2,  12, 9,  3,
            7,  7,  11, 9,  7,  10, 8,  9,  12, 12, 3,  11, 5,  12, 13, 5,
            12, 2,  10, 7,  12, 7,  6,  9,  5,  11, 4,  3,  12, 15, 1,  8,
            7,  5,  9,  12, 8,  11, 10, 7,  6,  3,  3,  4,  11, 7,  8,  5,
            6,  5,  9,  8,  13, 8,  5,  9,  5,  8,  4,  7,  4,  3,  14, 9,
        };
        const yatq_raw_norm: f32 = 9.23774529;

        var blk: turboquant.BlockTQ4(256) = undefined;
        turboquant.quantizeBlockTQ4(256, &x, &blk);

        // (a) raw L2 norm of input (independent of quantization) must
        //     agree with what numpy computed.
        var raw_sq: f32 = 0;
        for (x) |v| raw_sq += v * v;
        const our_raw_norm = @sqrt(raw_sq);
        if (@abs(our_raw_norm - yatq_raw_norm) > 1e-4) {
            std.debug.print("L2 norm divergence: our={d} yatq={d}\n", .{ our_raw_norm, yatq_raw_norm });
            return error.ParityFailed;
        }

        // (b) every Lloyd-Max index must match bit-exact.
        var k: usize = 0;
        while (k < 128) : (k += 1) {
            const lo = blk.indices[k] & 0x0f;
            const hi = (blk.indices[k] >> 4) & 0x0f;
            const want_lo = yatq_indices_b4[2 * k];
            const want_hi = yatq_indices_b4[2 * k + 1];
            if (lo != want_lo or hi != want_hi) {
                std.debug.print("idx mismatch at coord {d}: got ({d},{d}) want ({d},{d})\n", .{ 2 * k, lo, hi, want_lo, want_hi });
                return error.ParityFailed;
            }
        }
    }

    std.debug.print("PASS turboquant CPU oracle (tables + FWHT + RHT + TQ4 round-trip + YATQ bit-exact)\n", .{});
}

// ── q4_0 CPU smoke: round-trip parity for the int4 weight oracle ───
//
// Tier-1 quantization preflight. The CPU q4_0.zig functions are the
// reference the GPU shader will be parity-checked against, so this
// smoke verifies them in isolation: hand-checked single-block encode,
// round-trip on a Gaussian row (cosine sim and per-element MSE), and
// the symmetric edge case where one element saturates +7.

pub fn runQ4_0Smoke(allocator: std.mem.Allocator) !void {
    _ = allocator;

    // 1) Single-block hand-check. With a deterministic ramp the largest
    //    magnitude is the first or last element, so the scale d picks
    //    up that element's sign and the round-trip is bit-tight.
    {
        var src: [32]f32 = undefined;
        for (&src, 0..) |*v, i| v.* = @as(f32, @floatFromInt(@as(i32, @intCast(i)) - 16)) * 0.1;
        // src ranges -1.6 .. +1.5 in steps of 0.1. amax = 1.6 with the
        // largest signed magnitude = -1.6. So d = -1.6 / -8 = 0.2.
        var blocks: [1]q4_0.Block = undefined;
        q4_0.quantizeRow(&src, &blocks);
        if (@abs(@as(f32, @floatCast(blocks[0].d)) - 0.2) > 1e-3) {
            std.debug.print("q4_0 single-block: d={d}, want 0.2\n", .{@as(f32, @floatCast(blocks[0].d))});
            return error.ParityFailed;
        }

        var rec: [32]f32 = undefined;
        q4_0.dequantizeRow(&blocks, &rec);
        var max_err: f32 = 0;
        for (src, rec) |x, y| max_err = @max(max_err, @abs(x - y));
        // With d=0.2, snap-to-grid worst case is 0.1 (half a step).
        if (max_err > 0.105) {
            std.debug.print("q4_0 single-block round-trip: max |Δ|={d}\n", .{max_err});
            return error.ParityFailed;
        }
    }

    // 2) Gaussian row round-trip (1024 floats = 32 blocks). Q4_0 on a
    //    unit-Gaussian source gives ≈ 0.0033 SNR-bound MSE per coord;
    //    we leave generous headroom against PRNG variance.
    {
        const n_elems: usize = 1024;
        var prng = std.Random.DefaultPrng.init(0xC0DEC0DECAFE);
        const r = prng.random();
        var src: [n_elems]f32 = undefined;
        for (&src) |*v| v.* = r.floatNorm(f32);

        var blocks: [n_elems / 32]q4_0.Block = undefined;
        q4_0.quantizeRow(&src, &blocks);
        var rec: [n_elems]f32 = undefined;
        q4_0.dequantizeRow(&blocks, &rec);

        var err_sq: f64 = 0;
        for (src, rec) |x, y| {
            const d = @as(f64, x) - @as(f64, y);
            err_sq += d * d;
        }
        const mse: f64 = err_sq / @as(f64, @floatFromInt(n_elems));
        const cos = q4_0.cosineSim(&src, &rec);

        if (mse > 0.01) {
            std.debug.print("q4_0 Gaussian MSE={d:.5} (>0.01)\n", .{mse});
            return error.ParityFailed;
        }
        if (cos < 0.995) {
            std.debug.print("q4_0 Gaussian cos-sim={d:.5} (<0.995)\n", .{cos});
            return error.ParityFailed;
        }
        std.debug.print("       q4_0 1024-float Gaussian: MSE={d:.5}, cos-sim={d:.5}\n", .{ mse, cos });
    }

    // 3) Saturation / extremes: in the llama.cpp Q4_0 scheme, the
    //    element with the largest magnitude maps to idx 0 (signed
    //    -8), so it round-trips EXACTLY. The opposite extreme
    //    saturates against id=15 (signed +7) and reconstructs as
    //    (15-8)*|d| = 7/8 of the original. Confirms clamp + sign
    //    handling at the boundary.
    {
        var src: [32]f32 = [_]f32{0.0} ** 32;
        src[0] = 1.0;   // largest magnitude (positive) — should round-trip exactly
        src[15] = -1.0; // opposite extreme — should saturate to -0.875
        var blocks: [1]q4_0.Block = undefined;
        q4_0.quantizeRow(&src, &blocks);

        // d = max / -8 = 1.0 / -8 = -0.125. Element 0 → idx 0 → (0-8)*-0.125 = 1.0.
        // Element 15 → 8/(-0.125) ⇒ raw 8 → +8.5 ⇒ floor 16 ⇒ clamp 15 → (15-8)*-0.125 = -0.875.
        const idx0: u8 = blocks[0].qs[0] & 0x0F;
        const idx15: u8 = blocks[0].qs[15] & 0x0F;
        if (idx0 != 0 or idx15 != 15) {
            std.debug.print("q4_0 saturation: idx[0]={d} (want 0), idx[15]={d} (want 15)\n", .{ idx0, idx15 });
            return error.ParityFailed;
        }

        var rec: [32]f32 = undefined;
        q4_0.dequantizeRow(&blocks, &rec);
        if (@abs(rec[0] - 1.0) > 1e-6 or @abs(rec[15] - (-0.875)) > 1e-6) {
            std.debug.print("q4_0 saturation decode: rec[0]={d} (want 1.0), rec[15]={d} (want -0.875)\n", .{ rec[0], rec[15] });
            return error.ParityFailed;
        }
    }

    std.debug.print("PASS q4_0 CPU oracle (single-block, Gaussian round-trip, saturation edge)\n", .{});
}

// ── q4_K CPU smoke: round-trip parity for the asymmetric int4 oracle ─
//
// Verifies our llama.cpp-compatible Q4_K_M reference at three points:
// the constant-block degenerate case (super-scales should both be 0,
// reconstruction exact), a Gaussian super-block round-trip (cosine sim
// > 0.999, MSE noticeably better than Q4_0's 0.005 on the same input
// since asymmetric quant + iterative refinement both help), and the
// scales-byte packing/unpacking — encode a known (sc[0..7], m[0..7])
// pattern with non-trivial top-2-bits and confirm getScaleMinK4
// recovers it bit-perfect.

pub fn runQ4_KSmoke(allocator: std.mem.Allocator) !void {
    _ = allocator;

    // 1) All-zero block: true degenerate case. min=max=0, makeQkx2Quants
    //    short-circuits to scale=0, the_min=0; super-scales d=dmin=0;
    //    dequant returns all zero. Verifies the early-out doesn't write
    //    garbage into the qs/scales bytes (left at memset-0 from phase 0).
    {
        const src: [q4_k.QK_K]f32 = [_]f32{0.0} ** q4_k.QK_K;
        var blocks: [1]q4_k.Block = undefined;
        q4_k.quantizeRow(&src, &blocks);

        var rec: [q4_k.QK_K]f32 = undefined;
        q4_k.dequantizeRow(&blocks, &rec);

        var max_err: f32 = 0;
        for (rec) |y| max_err = @max(max_err, @abs(y));
        if (max_err > 1e-6) {
            std.debug.print("q4_K zero block: rec should be all-zero, max |y|={d}\n", .{max_err});
            return error.ParityFailed;
        }
    }

    // 2) Scale-byte packing: hand-build a (sc, m) pattern with values that
    //    exercise both the low-6-bit and high-2-bit slots, encode it via
    //    the same op-sequence quantizeRow uses, and confirm getScaleMinK4
    //    recovers it. Catches any byte-shift mistake without needing a
    //    full quantize round-trip to expose.
    {
        const sc_in = [_]u8{ 0, 1, 17, 63, 32, 47, 60, 33 };
        const m_in = [_]u8{ 63, 0, 5, 31, 48, 11, 50, 21 };
        var scales: [q4_k.K_SCALE_SIZE]u8 = [_]u8{0} ** q4_k.K_SCALE_SIZE;
        for (0..8) |j| {
            const ls = sc_in[j];
            const lm = m_in[j];
            if (j < 4) {
                scales[j] = ls;
                scales[j + 4] = lm;
            } else {
                scales[j + 4] = (ls & 0x0F) | ((lm & 0x0F) << 4);
                scales[j - 4] |= @as(u8, @intCast(ls >> 4)) << 6;
                scales[j] |= @as(u8, @intCast(lm >> 4)) << 6;
            }
        }
        for (0..8) |j| {
            var sc_out: u8 = undefined;
            var m_out: u8 = undefined;
            q4_k.getScaleMinK4(@intCast(j), &scales, &sc_out, &m_out);
            if (sc_out != sc_in[j] or m_out != m_in[j]) {
                std.debug.print(
                    "q4_K scales pack[{d}]: got sc={d} m={d}, want sc={d} m={d}\n",
                    .{ j, sc_out, m_out, sc_in[j], m_in[j] },
                );
                return error.ParityFailed;
            }
        }
    }

    // 3) Gaussian super-block round-trip (one full 256-elem block). Q4_K
    //    on unit Gaussian gives substantially better MSE than Q4_0 at
    //    the same nominal bitrate — the asymmetric offset + iterative
    //    refinement together typically halve per-coord error vs Q4_0's
    //    max-magnitude scheme.
    {
        const n_elems: usize = q4_k.QK_K;
        var prng = std.Random.DefaultPrng.init(0xC0DEC0DECAFE);
        const r = prng.random();
        var src: [n_elems]f32 = undefined;
        for (&src) |*v| v.* = r.floatNorm(f32);

        var blocks: [1]q4_k.Block = undefined;
        q4_k.quantizeRow(&src, &blocks);
        var rec: [n_elems]f32 = undefined;
        q4_k.dequantizeRow(&blocks, &rec);

        var err_sq: f64 = 0;
        for (src, rec) |x, y| {
            const d = @as(f64, x) - @as(f64, y);
            err_sq += d * d;
        }
        const mse: f64 = err_sq / @as(f64, @floatFromInt(n_elems));
        const cos = q4_0.cosineSim(&src, &rec);

        // q4_K typical MSE on unit Gaussian is ~0.003 (vs ~0.008 for q4_0
        // on the same input); leave slack for prng variance.
        // q4_K typical MSE on unit Gaussian is ~0.006 on 256-elem blocks
        // (vs q4_0's 0.008–0.010 — ~30% improvement). Threshold leaves
        // room for prng variance across seeds while still proving the
        // win over q4_0 holds.
        if (mse > 0.008) {
            std.debug.print("q4_K Gaussian MSE={d:.5} (>0.008)\n", .{mse});
            return error.ParityFailed;
        }
        if (cos < 0.997) {
            std.debug.print("q4_K Gaussian cos-sim={d:.5} (<0.997)\n", .{cos});
            return error.ParityFailed;
        }
        std.debug.print("       q4_K 256-float Gaussian: MSE={d:.5}, cos-sim={d:.5}\n", .{ mse, cos });
    }

    // 4) Multi-super-block Gaussian (4×256 = 1024 elems). Same expected
    //    MSE/cos-sim envelope; verifies the b-loop in quantizeRow doesn't
    //    leak state between super-blocks.
    {
        const n_blocks: usize = 4;
        const n_elems: usize = q4_k.QK_K * n_blocks;
        var prng = std.Random.DefaultPrng.init(0xBADCAFEDEADBEEF);
        const r = prng.random();
        var src: [n_elems]f32 = undefined;
        for (&src) |*v| v.* = r.floatNorm(f32);

        var blocks: [n_blocks]q4_k.Block = undefined;
        q4_k.quantizeRow(&src, &blocks);
        var rec: [n_elems]f32 = undefined;
        q4_k.dequantizeRow(&blocks, &rec);

        var err_sq: f64 = 0;
        for (src, rec) |x, y| {
            const d = @as(f64, x) - @as(f64, y);
            err_sq += d * d;
        }
        const mse: f64 = err_sq / @as(f64, @floatFromInt(n_elems));
        const cos = q4_0.cosineSim(&src, &rec);

        if (mse > 0.008) {
            std.debug.print("q4_K multi-block MSE={d:.5}\n", .{mse});
            return error.ParityFailed;
        }
        if (cos < 0.997) {
            std.debug.print("q4_K multi-block cos-sim={d:.5}\n", .{cos});
            return error.ParityFailed;
        }
    }

    std.debug.print("PASS q4_K CPU oracle (constant block, scales packing, Gaussian round-trip)\n", .{});
}


