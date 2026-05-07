//! GPU MLP smokes + TrainingRunner smokes + the headless `--train-demo`
//! CLI. All exercised by the no-arg fallthrough in main.zig. Extracted
//! verbatim from main.zig.

const std = @import("std");
const vk = @import("../gpu/vk.zig");
const buffer = @import("../gpu/buffer.zig");
const pipeline = @import("../gpu/pipeline.zig");
const gpu_recorder = @import("../gpu/recorder.zig");
const cpu_train = @import("../cpu/train.zig");
const train_runner = @import("../train/runner.zig");
const train_runner_n = @import("../train/runner_n.zig");
const runtime = @import("../runtime.zig");
const shaders = @import("shaders");

const aliases = @import("../runtime_aliases.zig");
const helpers = @import("../smoke/helpers.zig");
const util = @import("../util.zig");

// Aliases for runtime push-constant types referenced inside the moved
// code. aliases.ReluPush + a few others were already declared inline in the
// extracted block; the rest mirror what main.zig still keeps.

// ── tiny-MLP GPU forward smoke: matmul → bias → relu → matmul → bias ──
//
// Chunk 2 of training-v0. Composes existing kernels (matmul_nt,
// add_in_place, plus the new relu.comp) into a 2-layer MLP forward
// pass on the GPU and parity-checks against the CPU oracle in
// `cpu_train.forward`. The MLP is built deterministically so the GPU
// dispatch and the CPU reference both see bit-identical weights.
//
// Why compose existing matmuls instead of writing a fused mlp_forward
// shader: chunk 2 is a parity proof, not a perf milestone. A fused
// shader is a ~30-line port once the composition is verified, but the
// composition exposes the building blocks the upcoming backward
// shaders also need (matmul, transposed matmul, relu).

pub fn runGpuMlpForwardSmoke(allocator: std.mem.Allocator) !void {
    var ctx = try vk.Context.init(allocator);
    defer ctx.deinit();

    const dim_in: usize = 4;
    const dim_h: usize = 8;
    const dim_out: usize = 4;

    // Same construction as the CPU smoke: deterministic weights so we
    // can run forward both places and diff. Different seed than chunk 1
    // — exercising the same shape on a fresh init catches any "smoke
    // happened to pass on lucky weights" failure modes.
    var mlp = try cpu_train.Mlp.init(allocator, dim_in, dim_h, dim_out, 0.3, 0xF0DDA1A);
    defer mlp.deinit(allocator);
    // Bias at zero from init() is too forgiving — set b1 and b2 to
    // distinct nonzero values so a "bias never applied" bug surfaces.
    for (mlp.b1, 0..) |*v, i| v.* = 0.1 - 0.05 * @as(f32, @floatFromInt(i));
    for (mlp.b2, 0..) |*v, i| v.* = -0.2 + 0.07 * @as(f32, @floatFromInt(i));

    const x = [_]f32{ 1.0, 0.5, -0.3, 0.2 };

    // ── CPU oracle ─────────────────────────────────────────────────
    var h_pre_cpu: [dim_h]f32 = undefined;
    var h_cpu: [dim_h]f32 = undefined;
    var y_cpu: [dim_out]f32 = undefined;
    var act: cpu_train.Activations = .{
        .x = &x,
        .h_pre = &h_pre_cpu,
        .h = &h_cpu,
        .y = &y_cpu,
    };
    cpu_train.forward(&mlp, &act);

    // ── GPU buffers ────────────────────────────────────────────────
    var buf_x = try buffer.Buffer.initStatic(&ctx, f32, &x);
    defer buf_x.deinit(ctx.device);
    var buf_w1 = try buffer.Buffer.initStatic(&ctx, f32, mlp.w1);
    defer buf_w1.deinit(ctx.device);
    var buf_b1 = try buffer.Buffer.initStatic(&ctx, f32, mlp.b1);
    defer buf_b1.deinit(ctx.device);
    var buf_w2 = try buffer.Buffer.initStatic(&ctx, f32, mlp.w2);
    defer buf_w2.deinit(ctx.device);
    var buf_b2 = try buffer.Buffer.initStatic(&ctx, f32, mlp.b2);
    defer buf_b2.deinit(ctx.device);
    var buf_h_pre = try buffer.Buffer.initDeviceOnly(&ctx, dim_h * @sizeOf(f32));
    defer buf_h_pre.deinit(ctx.device);
    var buf_h = try buffer.Buffer.initDeviceOnly(&ctx, dim_h * @sizeOf(f32));
    defer buf_h.deinit(ctx.device);
    var buf_y = try buffer.Buffer.initDeviceOnly(&ctx, dim_out * @sizeOf(f32));
    defer buf_y.deinit(ctx.device);

    // ── Pipelines ──────────────────────────────────────────────────
    var k_matmul = try pipeline.Kernel.init(&ctx, &shaders.matmul_nt, 3, @sizeOf(aliases.MatmulPush));
    defer k_matmul.deinit();
    var k_add = try pipeline.Kernel.init(&ctx, &shaders.add_in_place, 2, @sizeOf(runtime.AddInPlacePush));
    defer k_add.deinit();
    var k_relu = try pipeline.Kernel.init(&ctx, &shaders.relu, 2, @sizeOf(aliases.ReluPush));
    defer k_relu.deinit();

    // ── Record + submit ────────────────────────────────────────────
    // Treat x and h as 1×K row matrices so matmul_nt computes
    // y_pre[1, N] = x[1, K] · W[N, K]ᵀ. That's exactly W·x for the
    // row vector x, which is the layer formulation we want.
    var rec = try gpu_recorder.Recorder.init(&ctx, 16, 64);
    defer rec.deinit();
    try rec.begin();

    const matmul1_push = aliases.MatmulPush{ .m = 1, .n = @intCast(dim_h), .k = @intCast(dim_in) };
    const matmul2_push = aliases.MatmulPush{ .m = 1, .n = @intCast(dim_out), .k = @intCast(dim_h) };
    const add1_push = runtime.AddInPlacePush{ .n = @intCast(dim_h) };
    const add2_push = runtime.AddInPlacePush{ .n = @intCast(dim_out) };
    const relu_push = aliases.ReluPush{ .n = @intCast(dim_h) };

    // matmul_nt grid is (ceil(M/16), ceil(N/16)) — for M=1, N≤16 the
    // whole work is one workgroup, which is correct (threads outside
    // bound early-out).
    try rec.dispatch(&k_matmul, &.{ &buf_x, &buf_w1, &buf_h_pre }, &matmul1_push, 1, 1, 1);
    try rec.dispatch(&k_add, &.{ &buf_h_pre, &buf_b1 }, &add1_push, util.ceilDiv(@as(u32, dim_h), 256), 1, 1);
    try rec.dispatch(&k_relu, &.{ &buf_h_pre, &buf_h }, &relu_push, util.ceilDiv(@as(u32, dim_h), 256), 1, 1);
    try rec.dispatch(&k_matmul, &.{ &buf_h, &buf_w2, &buf_y }, &matmul2_push, 1, 1, 1);
    try rec.dispatch(&k_add, &.{ &buf_y, &buf_b2 }, &add2_push, util.ceilDiv(@as(u32, dim_out), 256), 1, 1);

    try rec.endAndSubmit();

    // ── Compare ───────────────────────────────────────────────────
    var h_pre_gpu: [dim_h]f32 = undefined;
    var h_gpu: [dim_h]f32 = undefined;
    var y_gpu: [dim_out]f32 = undefined;
    try buf_h_pre.readBack(&ctx, f32, &h_pre_gpu);
    try buf_h.readBack(&ctx, f32, &h_gpu);
    try buf_y.readBack(&ctx, f32, &y_gpu);

    const tol: f32 = 1e-5;
    const ParityCase = struct { name: []const u8, got: []const f32, want: []const f32 };
    const cases = [_]ParityCase{
        .{ .name = "h_pre", .got = &h_pre_gpu, .want = &h_pre_cpu },
        .{ .name = "h", .got = &h_gpu, .want = &h_cpu },
        .{ .name = "y", .got = &y_gpu, .want = &y_cpu },
    };
    var max_abs: f32 = 0;
    for (cases) |cs| {
        for (cs.got, cs.want, 0..) |g, w, i| {
            const d = @abs(g - w);
            if (d > tol) {
                std.debug.print(
                    "GPU MLP forward MISMATCH on {s}[{d}]: got {d:.7}, expected {d:.7}\n",
                    .{ cs.name, i, g, w },
                );
                return error.ParityFailed;
            }
            if (d > max_abs) max_abs = d;
        }
    }
    std.debug.print(
        "PASS GPU MLP forward (4→8→4, matmul+bias+relu+matmul+bias, max |Δ| vs CPU = {e})\n",
        .{max_abs},
    );
}

// ── tiny-MLP GPU backward smoke: gradients vs CPU oracle ────────────
//
// Chunk 3 of training-v0. Runs the full forward + backward pipeline on
// the GPU and parity-checks every gradient buffer against the CPU
// oracle in `cpu_train.backward`. Three new shaders carry it:
//
//   linear_backward_dx — dL/dh from dL/dy and W2 (transposed matvec)
//   relu_backward      — gates dL/dh by the saved h_pre > 0 mask
//   outer_product      — dL/dW = upstream ⊗ input  (dW2 and dW1 both)
//
// db gradients reuse `slice_copy` (dL/db = dL/dy by definition; pure
// memcpy at the buffer level). Per-step loss-grad (dL/dy = y - target)
// is computed in a single host-side op against `add_in_place`-style
// staging — kept on host because it's one fp32 subtract per step at
// dim_out scale, not worth a shader.

pub fn runGpuMlpBackwardSmoke(allocator: std.mem.Allocator) !void {
    var ctx = try vk.Context.init(allocator);
    defer ctx.deinit();

    const dim_in: usize = 4;
    const dim_h: usize = 8;
    const dim_out: usize = 4;

    var mlp = try cpu_train.Mlp.init(allocator, dim_in, dim_h, dim_out, 0.3, 0xBACC0FF);
    defer mlp.deinit(allocator);
    for (mlp.b1, 0..) |*v, i| v.* = 0.1 - 0.05 * @as(f32, @floatFromInt(i));
    for (mlp.b2, 0..) |*v, i| v.* = -0.2 + 0.07 * @as(f32, @floatFromInt(i));

    const x = [_]f32{ 1.0, 0.5, -0.3, 0.2 };
    const target = [_]f32{ 1.0, 0.0, 0.0, 0.0 };

    // ── CPU oracle: forward + grads ────────────────────────────────
    var h_pre_cpu: [dim_h]f32 = undefined;
    var h_cpu: [dim_h]f32 = undefined;
    var y_cpu: [dim_out]f32 = undefined;
    var act: cpu_train.Activations = .{
        .x = &x,
        .h_pre = &h_pre_cpu,
        .h = &h_cpu,
        .y = &y_cpu,
    };
    cpu_train.forward(&mlp, &act);
    var dL_dy_cpu: [dim_out]f32 = undefined;
    cpu_train.mseLossGrad(&dL_dy_cpu, &y_cpu, &target);
    var grads_cpu = try cpu_train.Grads.init(allocator, &mlp);
    defer grads_cpu.deinit(allocator);
    try cpu_train.backward(allocator, &mlp, &act, &dL_dy_cpu, &grads_cpu);

    // ── GPU buffers ────────────────────────────────────────────────
    var buf_x = try buffer.Buffer.initStatic(&ctx, f32, &x);
    defer buf_x.deinit(ctx.device);
    var buf_w1 = try buffer.Buffer.initStatic(&ctx, f32, mlp.w1);
    defer buf_w1.deinit(ctx.device);
    var buf_b1 = try buffer.Buffer.initStatic(&ctx, f32, mlp.b1);
    defer buf_b1.deinit(ctx.device);
    var buf_w2 = try buffer.Buffer.initStatic(&ctx, f32, mlp.w2);
    defer buf_w2.deinit(ctx.device);
    var buf_b2 = try buffer.Buffer.initStatic(&ctx, f32, mlp.b2);
    defer buf_b2.deinit(ctx.device);
    var buf_h_pre = try buffer.Buffer.initDeviceOnly(&ctx, dim_h * @sizeOf(f32));
    defer buf_h_pre.deinit(ctx.device);
    var buf_h = try buffer.Buffer.initDeviceOnly(&ctx, dim_h * @sizeOf(f32));
    defer buf_h.deinit(ctx.device);
    var buf_y = try buffer.Buffer.initDeviceOnly(&ctx, dim_out * @sizeOf(f32));
    defer buf_y.deinit(ctx.device);

    // dL/dy is staged from host (cheap, dim_out scalars). Same for
    // grads buffers — they get *written* by the GPU.
    var buf_dL_dy = try buffer.Buffer.initStatic(&ctx, f32, &dL_dy_cpu);
    defer buf_dL_dy.deinit(ctx.device);
    var buf_dh = try buffer.Buffer.initDeviceOnly(&ctx, dim_h * @sizeOf(f32));
    defer buf_dh.deinit(ctx.device);
    var buf_dh_pre = try buffer.Buffer.initDeviceOnly(&ctx, dim_h * @sizeOf(f32));
    defer buf_dh_pre.deinit(ctx.device);
    var buf_dw1 = try buffer.Buffer.initDeviceOnly(&ctx, dim_h * dim_in * @sizeOf(f32));
    defer buf_dw1.deinit(ctx.device);
    var buf_db1 = try buffer.Buffer.initDeviceOnly(&ctx, dim_h * @sizeOf(f32));
    defer buf_db1.deinit(ctx.device);
    var buf_dw2 = try buffer.Buffer.initDeviceOnly(&ctx, dim_out * dim_h * @sizeOf(f32));
    defer buf_dw2.deinit(ctx.device);
    var buf_db2 = try buffer.Buffer.initDeviceOnly(&ctx, dim_out * @sizeOf(f32));
    defer buf_db2.deinit(ctx.device);

    // ── Pipelines ──────────────────────────────────────────────────
    var k_matmul = try pipeline.Kernel.init(&ctx, &shaders.matmul_nt, 3, @sizeOf(aliases.MatmulPush));
    defer k_matmul.deinit();
    var k_add = try pipeline.Kernel.init(&ctx, &shaders.add_in_place, 2, @sizeOf(runtime.AddInPlacePush));
    defer k_add.deinit();
    var k_relu = try pipeline.Kernel.init(&ctx, &shaders.relu, 2, @sizeOf(aliases.ReluPush));
    defer k_relu.deinit();
    var k_relu_bw = try pipeline.Kernel.init(&ctx, &shaders.relu_backward, 3, @sizeOf(aliases.ReluBackwardPush));
    defer k_relu_bw.deinit();
    var k_lin_dx = try pipeline.Kernel.init(&ctx, &shaders.linear_backward_dx, 3, @sizeOf(aliases.LinearBackwardDxPush));
    defer k_lin_dx.deinit();
    var k_outer = try pipeline.Kernel.init(&ctx, &shaders.outer_product, 3, @sizeOf(aliases.OuterProductPush));
    defer k_outer.deinit();
    var k_copy = try pipeline.Kernel.init(&ctx, &shaders.slice_copy, 2, @sizeOf(aliases.SliceCopyPush));
    defer k_copy.deinit();

    var rec = try gpu_recorder.Recorder.init(&ctx, 32, 128);
    defer rec.deinit();
    try rec.begin();

    // ── Forward (same as chunk 2) ──────────────────────────────────
    const matmul1_push = aliases.MatmulPush{ .m = 1, .n = @intCast(dim_h), .k = @intCast(dim_in) };
    const matmul2_push = aliases.MatmulPush{ .m = 1, .n = @intCast(dim_out), .k = @intCast(dim_h) };
    const add1_push = runtime.AddInPlacePush{ .n = @intCast(dim_h) };
    const add2_push = runtime.AddInPlacePush{ .n = @intCast(dim_out) };
    const relu_push = aliases.ReluPush{ .n = @intCast(dim_h) };
    try rec.dispatch(&k_matmul, &.{ &buf_x, &buf_w1, &buf_h_pre }, &matmul1_push, 1, 1, 1);
    try rec.dispatch(&k_add, &.{ &buf_h_pre, &buf_b1 }, &add1_push, util.ceilDiv(@as(u32, dim_h), 256), 1, 1);
    try rec.dispatch(&k_relu, &.{ &buf_h_pre, &buf_h }, &relu_push, util.ceilDiv(@as(u32, dim_h), 256), 1, 1);
    try rec.dispatch(&k_matmul, &.{ &buf_h, &buf_w2, &buf_y }, &matmul2_push, 1, 1, 1);
    try rec.dispatch(&k_add, &.{ &buf_y, &buf_b2 }, &add2_push, util.ceilDiv(@as(u32, dim_out), 256), 1, 1);

    // ── Backward ───────────────────────────────────────────────────
    // dL/db2 = dL/dy   (slice_copy 0..dim_out → 0..dim_out).
    const copy_db2 = aliases.SliceCopyPush{ .src_off = 0, .dst_off = 0, .n_elem = @intCast(dim_out) };
    try rec.dispatch(&k_copy, &.{ &buf_dL_dy, &buf_db2 }, &copy_db2, util.ceilDiv(@as(u32, dim_out), 256), 1, 1);

    // dL/dW2[i, j] = dL/dy[i] · h[j]   (outer product, [dim_out, dim_h]).
    const op_dw2 = aliases.OuterProductPush{ .dim_out = @intCast(dim_out), .dim_in = @intCast(dim_h) };
    try rec.dispatch(
        &k_outer,
        &.{ &buf_dL_dy, &buf_h, &buf_dw2 },
        &op_dw2,
        util.ceilDiv(@as(u32, dim_out), 16),
        util.ceilDiv(@as(u32, dim_h), 16),
        1,
    );

    // dL/dh = W2^T · dL/dy  (transposed matvec).
    const lin_dx_push = aliases.LinearBackwardDxPush{ .dim_out = @intCast(dim_out), .dim_in = @intCast(dim_h) };
    try rec.dispatch(
        &k_lin_dx,
        &.{ &buf_dL_dy, &buf_w2, &buf_dh },
        &lin_dx_push,
        util.ceilDiv(@as(u32, dim_h), 256),
        1,
        1,
    );

    // dL/dh_pre = dL/dh · 1[h_pre > 0].
    const relu_bw_push = aliases.ReluBackwardPush{ .n = @intCast(dim_h) };
    try rec.dispatch(
        &k_relu_bw,
        &.{ &buf_dh, &buf_h_pre, &buf_dh_pre },
        &relu_bw_push,
        util.ceilDiv(@as(u32, dim_h), 256),
        1,
        1,
    );

    // dL/db1 = dL/dh_pre.
    const copy_db1 = aliases.SliceCopyPush{ .src_off = 0, .dst_off = 0, .n_elem = @intCast(dim_h) };
    try rec.dispatch(&k_copy, &.{ &buf_dh_pre, &buf_db1 }, &copy_db1, util.ceilDiv(@as(u32, dim_h), 256), 1, 1);

    // dL/dW1[j, k] = dL/dh_pre[j] · x[k].
    const op_dw1 = aliases.OuterProductPush{ .dim_out = @intCast(dim_h), .dim_in = @intCast(dim_in) };
    try rec.dispatch(
        &k_outer,
        &.{ &buf_dh_pre, &buf_x, &buf_dw1 },
        &op_dw1,
        util.ceilDiv(@as(u32, dim_h), 16),
        util.ceilDiv(@as(u32, dim_in), 16),
        1,
    );

    try rec.endAndSubmit();

    // ── Read back + parity ─────────────────────────────────────────
    var dw1_gpu: [dim_h * dim_in]f32 = undefined;
    var db1_gpu: [dim_h]f32 = undefined;
    var dw2_gpu: [dim_out * dim_h]f32 = undefined;
    var db2_gpu: [dim_out]f32 = undefined;
    try buf_dw1.readBack(&ctx, f32, &dw1_gpu);
    try buf_db1.readBack(&ctx, f32, &db1_gpu);
    try buf_dw2.readBack(&ctx, f32, &dw2_gpu);
    try buf_db2.readBack(&ctx, f32, &db2_gpu);

    const tol: f32 = 1e-5;
    const ParityCase = struct { name: []const u8, got: []const f32, want: []const f32 };
    const cases = [_]ParityCase{
        .{ .name = "dW1", .got = &dw1_gpu, .want = grads_cpu.dw1 },
        .{ .name = "db1", .got = &db1_gpu, .want = grads_cpu.db1 },
        .{ .name = "dW2", .got = &dw2_gpu, .want = grads_cpu.dw2 },
        .{ .name = "db2", .got = &db2_gpu, .want = grads_cpu.db2 },
    };
    var max_abs: f32 = 0;
    for (cases) |cs| {
        for (cs.got, cs.want, 0..) |g, w, i| {
            const d = @abs(g - w);
            if (d > tol) {
                std.debug.print(
                    "GPU MLP backward MISMATCH on {s}[{d}]: got {d:.7}, expected {d:.7}\n",
                    .{ cs.name, i, g, w },
                );
                return error.ParityFailed;
            }
            if (d > max_abs) max_abs = d;
        }
    }
    std.debug.print(
        "PASS GPU MLP backward (dW1, db1, dW2, db2 vs CPU; max |Δ| = {e})\n",
        .{max_abs},
    );
}

// ── tiny-MLP GPU full training loop: SGD step + multi-step convergence ──
//
// Chunk 4 of training-v0. Adds the SGD step shader (`param -= lr ·
// grad`) and the MSE loss-grad shader (`dL/dy = pred − target`),
// closing the loop so the entire forward → backward → update cycle
// stays on the GPU with no per-step CPU↔GPU sync. Then runs N steps
// on both CPU oracle and GPU and asserts:
//
//   1. After 1 step, GPU weights == CPU weights within 1e-5
//      (every kernel and the optimizer touched parameters identically).
//   2. After N steps, the GPU loss curve matches the CPU loss curve
//      within 1e-4 — error accumulates very slowly at our dims, but
//      this catches any drift from a subtly-wrong dispatch order.
//
// One sync at the very end (read final weights and final pred for
// loss). Every intermediate step is pure GPU — same shape as the
// in-frame training the engine integration needs.

pub fn runGpuMlpTrainSmoke(allocator: std.mem.Allocator) !void {
    var ctx = try vk.Context.init(allocator);
    defer ctx.deinit();

    const dim_in: usize = 4;
    const dim_h: usize = 8;
    const dim_out: usize = 4;
    const lr: f32 = 0.05;
    const n_steps: u32 = 50;

    // Twin MLPs — same seed, same weights to start. CPU drives oracle,
    // GPU drives the unit under test. After n_steps each they should
    // be bit-close.
    var mlp_cpu = try cpu_train.Mlp.init(allocator, dim_in, dim_h, dim_out, 0.3, 0x57DD57D);
    defer mlp_cpu.deinit(allocator);
    for (mlp_cpu.b1, 0..) |*v, i| v.* = 0.1 - 0.05 * @as(f32, @floatFromInt(i));
    for (mlp_cpu.b2, 0..) |*v, i| v.* = -0.2 + 0.07 * @as(f32, @floatFromInt(i));

    const x = [_]f32{ 1.0, 0.5, -0.3, 0.2 };
    const target = [_]f32{ 1.0, 0.0, 0.0, 0.0 };

    // ── GPU buffers (params live device-side and are mutated in
    // place by the SGD step; activations + grads live device-side
    // and are reused across steps) ──────────────────────────────────
    var buf_x = try buffer.Buffer.initStatic(&ctx, f32, &x);
    defer buf_x.deinit(ctx.device);
    var buf_target = try buffer.Buffer.initStatic(&ctx, f32, &target);
    defer buf_target.deinit(ctx.device);
    var buf_w1 = try buffer.Buffer.initStatic(&ctx, f32, mlp_cpu.w1);
    defer buf_w1.deinit(ctx.device);
    var buf_b1 = try buffer.Buffer.initStatic(&ctx, f32, mlp_cpu.b1);
    defer buf_b1.deinit(ctx.device);
    var buf_w2 = try buffer.Buffer.initStatic(&ctx, f32, mlp_cpu.w2);
    defer buf_w2.deinit(ctx.device);
    var buf_b2 = try buffer.Buffer.initStatic(&ctx, f32, mlp_cpu.b2);
    defer buf_b2.deinit(ctx.device);
    var buf_h_pre = try buffer.Buffer.initDeviceOnly(&ctx, dim_h * @sizeOf(f32));
    defer buf_h_pre.deinit(ctx.device);
    var buf_h = try buffer.Buffer.initDeviceOnly(&ctx, dim_h * @sizeOf(f32));
    defer buf_h.deinit(ctx.device);
    var buf_y = try buffer.Buffer.initDeviceOnly(&ctx, dim_out * @sizeOf(f32));
    defer buf_y.deinit(ctx.device);
    var buf_dL_dy = try buffer.Buffer.initDeviceOnly(&ctx, dim_out * @sizeOf(f32));
    defer buf_dL_dy.deinit(ctx.device);
    var buf_dh = try buffer.Buffer.initDeviceOnly(&ctx, dim_h * @sizeOf(f32));
    defer buf_dh.deinit(ctx.device);
    var buf_dh_pre = try buffer.Buffer.initDeviceOnly(&ctx, dim_h * @sizeOf(f32));
    defer buf_dh_pre.deinit(ctx.device);
    var buf_dw1 = try buffer.Buffer.initDeviceOnly(&ctx, dim_h * dim_in * @sizeOf(f32));
    defer buf_dw1.deinit(ctx.device);
    var buf_db1 = try buffer.Buffer.initDeviceOnly(&ctx, dim_h * @sizeOf(f32));
    defer buf_db1.deinit(ctx.device);
    var buf_dw2 = try buffer.Buffer.initDeviceOnly(&ctx, dim_out * dim_h * @sizeOf(f32));
    defer buf_dw2.deinit(ctx.device);
    var buf_db2 = try buffer.Buffer.initDeviceOnly(&ctx, dim_out * @sizeOf(f32));
    defer buf_db2.deinit(ctx.device);

    // ── Pipelines ──────────────────────────────────────────────────
    var k_matmul = try pipeline.Kernel.init(&ctx, &shaders.matmul_nt, 3, @sizeOf(aliases.MatmulPush));
    defer k_matmul.deinit();
    var k_add = try pipeline.Kernel.init(&ctx, &shaders.add_in_place, 2, @sizeOf(runtime.AddInPlacePush));
    defer k_add.deinit();
    var k_relu = try pipeline.Kernel.init(&ctx, &shaders.relu, 2, @sizeOf(aliases.ReluPush));
    defer k_relu.deinit();
    var k_relu_bw = try pipeline.Kernel.init(&ctx, &shaders.relu_backward, 3, @sizeOf(aliases.ReluBackwardPush));
    defer k_relu_bw.deinit();
    var k_lin_dx = try pipeline.Kernel.init(&ctx, &shaders.linear_backward_dx, 3, @sizeOf(aliases.LinearBackwardDxPush));
    defer k_lin_dx.deinit();
    var k_outer = try pipeline.Kernel.init(&ctx, &shaders.outer_product, 3, @sizeOf(aliases.OuterProductPush));
    defer k_outer.deinit();
    var k_copy = try pipeline.Kernel.init(&ctx, &shaders.slice_copy, 2, @sizeOf(aliases.SliceCopyPush));
    defer k_copy.deinit();
    var k_sgd = try pipeline.Kernel.init(&ctx, &shaders.sgd_step, 2, @sizeOf(aliases.SgdStepPush));
    defer k_sgd.deinit();
    var k_mse_grad = try pipeline.Kernel.init(&ctx, &shaders.mse_loss_grad, 3, @sizeOf(aliases.MseLossGradPush));
    defer k_mse_grad.deinit();

    // Push constants reused across steps.
    const matmul1_push = aliases.MatmulPush{ .m = 1, .n = @intCast(dim_h), .k = @intCast(dim_in) };
    const matmul2_push = aliases.MatmulPush{ .m = 1, .n = @intCast(dim_out), .k = @intCast(dim_h) };
    const add1_push = runtime.AddInPlacePush{ .n = @intCast(dim_h) };
    const add2_push = runtime.AddInPlacePush{ .n = @intCast(dim_out) };
    const relu_push = aliases.ReluPush{ .n = @intCast(dim_h) };
    const mse_grad_push = aliases.MseLossGradPush{ .n = @intCast(dim_out) };
    const op_dw2 = aliases.OuterProductPush{ .dim_out = @intCast(dim_out), .dim_in = @intCast(dim_h) };
    const lin_dx_push = aliases.LinearBackwardDxPush{ .dim_out = @intCast(dim_out), .dim_in = @intCast(dim_h) };
    const relu_bw_push = aliases.ReluBackwardPush{ .n = @intCast(dim_h) };
    const op_dw1 = aliases.OuterProductPush{ .dim_out = @intCast(dim_h), .dim_in = @intCast(dim_in) };
    const copy_db2 = aliases.SliceCopyPush{ .src_off = 0, .dst_off = 0, .n_elem = @intCast(dim_out) };
    const copy_db1 = aliases.SliceCopyPush{ .src_off = 0, .dst_off = 0, .n_elem = @intCast(dim_h) };
    const sgd_w1_push = aliases.SgdStepPush{ .n = @intCast(mlp_cpu.w1.len), .lr = lr };
    const sgd_b1_push = aliases.SgdStepPush{ .n = @intCast(mlp_cpu.b1.len), .lr = lr };
    const sgd_w2_push = aliases.SgdStepPush{ .n = @intCast(mlp_cpu.w2.len), .lr = lr };
    const sgd_b2_push = aliases.SgdStepPush{ .n = @intCast(mlp_cpu.b2.len), .lr = lr };

    // ── Run N steps on each side ──────────────────────────────────
    // Two GPU snapshots taken: after step 1 (single-step parity) and
    // after step n_steps (full-trajectory parity).
    const cpu_h_pre = try allocator.alloc(f32, dim_h);
    defer allocator.free(cpu_h_pre);
    const cpu_h = try allocator.alloc(f32, dim_h);
    defer allocator.free(cpu_h);
    const cpu_y = try allocator.alloc(f32, dim_out);
    defer allocator.free(cpu_y);
    var act_cpu: cpu_train.Activations = .{
        .x = &x,
        .h_pre = cpu_h_pre,
        .h = cpu_h,
        .y = cpu_y,
    };
    var grads_cpu = try cpu_train.Grads.init(allocator, &mlp_cpu);
    defer grads_cpu.deinit(allocator);

    const w1_after_1 = try allocator.alloc(f32, mlp_cpu.w1.len);
    defer allocator.free(w1_after_1);
    const w2_after_1 = try allocator.alloc(f32, mlp_cpu.w2.len);
    defer allocator.free(w2_after_1);
    const b1_after_1 = try allocator.alloc(f32, mlp_cpu.b1.len);
    defer allocator.free(b1_after_1);
    const b2_after_1 = try allocator.alloc(f32, mlp_cpu.b2.len);
    defer allocator.free(b2_after_1);

    // GPU side: drive each step through its own Recorder lifecycle.
    // Re-using one recorder across N steps would need a way to reset
    // n_dispatched + descriptor pool; cleanest for this smoke is one
    // recorder per step. Real trainer uses a frame-style loop with
    // per-frame recorder reset (chunk 5).
    var step: u32 = 0;
    while (step < n_steps) : (step += 1) {
        // CPU step.
        _ = try cpu_train.trainStep(allocator, &mlp_cpu, &act_cpu, &grads_cpu, &target, lr);

        // GPU step.
        var rec = try gpu_recorder.Recorder.init(&ctx, 16, 64);
        defer rec.deinit();
        try rec.begin();

        // Forward.
        try rec.dispatch(&k_matmul, &.{ &buf_x, &buf_w1, &buf_h_pre }, &matmul1_push, 1, 1, 1);
        try rec.dispatch(&k_add, &.{ &buf_h_pre, &buf_b1 }, &add1_push, util.ceilDiv(@as(u32, dim_h), 256), 1, 1);
        try rec.dispatch(&k_relu, &.{ &buf_h_pre, &buf_h }, &relu_push, util.ceilDiv(@as(u32, dim_h), 256), 1, 1);
        try rec.dispatch(&k_matmul, &.{ &buf_h, &buf_w2, &buf_y }, &matmul2_push, 1, 1, 1);
        try rec.dispatch(&k_add, &.{ &buf_y, &buf_b2 }, &add2_push, util.ceilDiv(@as(u32, dim_out), 256), 1, 1);

        // Loss grad.
        try rec.dispatch(&k_mse_grad, &.{ &buf_y, &buf_target, &buf_dL_dy }, &mse_grad_push, util.ceilDiv(@as(u32, dim_out), 256), 1, 1);

        // Backward.
        try rec.dispatch(&k_copy, &.{ &buf_dL_dy, &buf_db2 }, &copy_db2, util.ceilDiv(@as(u32, dim_out), 256), 1, 1);
        try rec.dispatch(&k_outer, &.{ &buf_dL_dy, &buf_h, &buf_dw2 }, &op_dw2, util.ceilDiv(@as(u32, dim_out), 16), util.ceilDiv(@as(u32, dim_h), 16), 1);
        try rec.dispatch(&k_lin_dx, &.{ &buf_dL_dy, &buf_w2, &buf_dh }, &lin_dx_push, util.ceilDiv(@as(u32, dim_h), 256), 1, 1);
        try rec.dispatch(&k_relu_bw, &.{ &buf_dh, &buf_h_pre, &buf_dh_pre }, &relu_bw_push, util.ceilDiv(@as(u32, dim_h), 256), 1, 1);
        try rec.dispatch(&k_copy, &.{ &buf_dh_pre, &buf_db1 }, &copy_db1, util.ceilDiv(@as(u32, dim_h), 256), 1, 1);
        try rec.dispatch(&k_outer, &.{ &buf_dh_pre, &buf_x, &buf_dw1 }, &op_dw1, util.ceilDiv(@as(u32, dim_h), 16), util.ceilDiv(@as(u32, dim_in), 16), 1);

        // SGD step (param -= lr · grad). Note W2 is updated BEFORE the
        // dh = W2^T · dL/dy dispatch reads W2 — which is fine because
        // dh was computed earlier in this same recorder, and the next
        // step's W2-read starts a fresh recorder with a barrier.
        try rec.dispatch(&k_sgd, &.{ &buf_w1, &buf_dw1 }, &sgd_w1_push, util.ceilDiv(@intCast(mlp_cpu.w1.len), 256), 1, 1);
        try rec.dispatch(&k_sgd, &.{ &buf_b1, &buf_db1 }, &sgd_b1_push, util.ceilDiv(@intCast(mlp_cpu.b1.len), 256), 1, 1);
        try rec.dispatch(&k_sgd, &.{ &buf_w2, &buf_dw2 }, &sgd_w2_push, util.ceilDiv(@intCast(mlp_cpu.w2.len), 256), 1, 1);
        try rec.dispatch(&k_sgd, &.{ &buf_b2, &buf_db2 }, &sgd_b2_push, util.ceilDiv(@intCast(mlp_cpu.b2.len), 256), 1, 1);

        try rec.endAndSubmit();

        // Snapshot after step 1 for the single-step parity check.
        if (step == 0) {
            try buf_w1.readBack(&ctx, f32, w1_after_1);
            try buf_b1.readBack(&ctx, f32, b1_after_1);
            try buf_w2.readBack(&ctx, f32, w2_after_1);
            try buf_b2.readBack(&ctx, f32, b2_after_1);
        }
    }

    // ── Single-step parity ────────────────────────────────────────
    // After step 0, the CPU has already mutated mlp_cpu — we need a
    // fresh oracle for the "after one step" state. Easiest: re-run
    // chunk-1's first step from scratch and compare.
    var oracle = try cpu_train.Mlp.init(allocator, dim_in, dim_h, dim_out, 0.3, 0x57DD57D);
    defer oracle.deinit(allocator);
    for (oracle.b1, 0..) |*v, i| v.* = 0.1 - 0.05 * @as(f32, @floatFromInt(i));
    for (oracle.b2, 0..) |*v, i| v.* = -0.2 + 0.07 * @as(f32, @floatFromInt(i));
    const oracle_h_pre = try allocator.alloc(f32, dim_h);
    defer allocator.free(oracle_h_pre);
    const oracle_h = try allocator.alloc(f32, dim_h);
    defer allocator.free(oracle_h);
    const oracle_y = try allocator.alloc(f32, dim_out);
    defer allocator.free(oracle_y);
    var oracle_act: cpu_train.Activations = .{
        .x = &x,
        .h_pre = oracle_h_pre,
        .h = oracle_h,
        .y = oracle_y,
    };
    var oracle_grads = try cpu_train.Grads.init(allocator, &oracle);
    defer oracle_grads.deinit(allocator);
    _ = try cpu_train.trainStep(allocator, &oracle, &oracle_act, &oracle_grads, &target, lr);

    const tol_step1: f32 = 1e-5;
    var max_step1: f32 = 0;
    for (oracle.w1, w1_after_1, 0..) |w, g, i| {
        const d = @abs(g - w);
        if (d > tol_step1) {
            std.debug.print("step-1 W1 MISMATCH[{d}]: gpu={d} cpu={d}\n", .{ i, g, w });
            return error.ParityFailed;
        }
        if (d > max_step1) max_step1 = d;
    }
    for (oracle.b1, b1_after_1, 0..) |w, g, i| {
        const d = @abs(g - w);
        if (d > tol_step1) {
            std.debug.print("step-1 b1 MISMATCH[{d}]: gpu={d} cpu={d}\n", .{ i, g, w });
            return error.ParityFailed;
        }
    }
    for (oracle.w2, w2_after_1, 0..) |w, g, i| {
        const d = @abs(g - w);
        if (d > tol_step1) {
            std.debug.print("step-1 W2 MISMATCH[{d}]: gpu={d} cpu={d}\n", .{ i, g, w });
            return error.ParityFailed;
        }
    }
    for (oracle.b2, b2_after_1, 0..) |w, g, i| {
        const d = @abs(g - w);
        if (d > tol_step1) {
            std.debug.print("step-1 b2 MISMATCH[{d}]: gpu={d} cpu={d}\n", .{ i, g, w });
            return error.ParityFailed;
        }
    }

    // ── Full-trajectory parity ────────────────────────────────────
    // After n_steps, compare the GPU's final weights against the
    // CPU's final weights. Looser tol because rounding at each step
    // accumulates — but at our dims and step count it stays tight.
    const w1_gpu = try allocator.alloc(f32, mlp_cpu.w1.len);
    defer allocator.free(w1_gpu);
    const b1_gpu = try allocator.alloc(f32, mlp_cpu.b1.len);
    defer allocator.free(b1_gpu);
    const w2_gpu = try allocator.alloc(f32, mlp_cpu.w2.len);
    defer allocator.free(w2_gpu);
    const b2_gpu = try allocator.alloc(f32, mlp_cpu.b2.len);
    defer allocator.free(b2_gpu);
    try buf_w1.readBack(&ctx, f32, w1_gpu);
    try buf_b1.readBack(&ctx, f32, b1_gpu);
    try buf_w2.readBack(&ctx, f32, w2_gpu);
    try buf_b2.readBack(&ctx, f32, b2_gpu);

    const tol_traj: f32 = 1e-4;
    var max_traj: f32 = 0;
    const ParamCase = struct { name: []const u8, gpu: []const f32, cpu: []const f32 };
    const traj_cases = [_]ParamCase{
        .{ .name = "W1", .gpu = w1_gpu, .cpu = mlp_cpu.w1 },
        .{ .name = "b1", .gpu = b1_gpu, .cpu = mlp_cpu.b1 },
        .{ .name = "W2", .gpu = w2_gpu, .cpu = mlp_cpu.w2 },
        .{ .name = "b2", .gpu = b2_gpu, .cpu = mlp_cpu.b2 },
    };
    for (traj_cases) |cs| {
        for (cs.gpu, cs.cpu, 0..) |g, c, i| {
            const d = @abs(g - c);
            if (d > tol_traj) {
                std.debug.print("traj {s} MISMATCH[{d}] @ step {d}: gpu={d} cpu={d}\n", .{ cs.name, i, n_steps, g, c });
                return error.ParityFailed;
            }
            if (d > max_traj) max_traj = d;
        }
    }
    std.debug.print(
        "PASS GPU MLP train ({d} SGD steps; step-1 |Δ|={e}, after-{d} |Δ|={e}, all on-device)\n",
        .{ n_steps, max_step1, n_steps, max_traj },
    );
}

// ── Multi-layer GPU MLP train smoke ───────────────────────────────
//
// Tier-1 chunk 2 of the post-v0 training arc. Same primitives as the
// 2-layer GPU train smoke, but orchestrated across n=3 layers in
// generic loops. Bit-exact step-1 parity vs the new MlpN CPU oracle
// proves the dispatch ordering works at depth before we lift it into
// the runner.
//
// Composition: forward is matmul → bias add → ReLU per hidden layer,
// then matmul → bias add (no ReLU) on the output. Backward seeds
// dL/dy via mse_loss_grad on the output, then for each layer L from
// n-1 down to 0 emits db = d_pre, dW = d_pre ⊗ input[L], and (if
// L > 0) propagates d_post[L-1] = W[L]ᵀ · d_pre[L] then
// d_pre[L-1] = d_post[L-1] · 1[pre[L-1] > 0]. SGD updates run after
// all gradients are computed so each W[L] is read at its pre-update
// value during backward.

pub fn runGpuMlpNTrainSmoke(allocator: std.mem.Allocator) !void {
    var ctx = try vk.Context.init(allocator);
    defer ctx.deinit();

    const layer_dims = [_]usize{ 4, 8, 6, 4 };
    const n = layer_dims.len - 1;
    const lr: f32 = 0.05;
    const n_steps: u32 = 50;

    var mlp_cpu = try cpu_train.MlpN.init(allocator, &layer_dims, 0.3, 0xC0FFEE3D);
    defer mlp_cpu.deinit(allocator);
    // Stamp non-zero biases so gradient flow through b is exercised.
    for (mlp_cpu.biases, 0..) |b, layer_idx| {
        for (b, 0..) |*v, i| {
            const li: f32 = @floatFromInt(layer_idx + 1);
            const ii: f32 = @floatFromInt(@as(i32, @intCast(i % 3)));
            v.* = 0.05 * (li - ii);
        }
    }

    const x = [_]f32{ 1.0, 0.5, -0.3, 0.2 };
    const target = [_]f32{ 1.0, 0.0, 0.0, 0.0 };

    // ── GPU buffers ───────────────────────────────────────────────
    var buf_x = try buffer.Buffer.initStatic(&ctx, f32, &x);
    defer buf_x.deinit(ctx.device);
    var buf_target = try buffer.Buffer.initStatic(&ctx, f32, &target);
    defer buf_target.deinit(ctx.device);

    var bufs_w: [n]buffer.Buffer = undefined;
    var bufs_b: [n]buffer.Buffer = undefined;
    var bufs_dw: [n]buffer.Buffer = undefined;
    var bufs_db: [n]buffer.Buffer = undefined;
    var bufs_pre: [n]buffer.Buffer = undefined;
    var bufs_post: [n]buffer.Buffer = undefined;
    var bufs_dpre: [n]buffer.Buffer = undefined;
    // d_post[L-1] for L=1..n-1 — needed only when there's a layer
    // below to back-prop through. Index by L (i.e. dpost_for_below[L]
    // is the d_post going INTO layer L-1's d_pre); slot 0 unused.
    var bufs_dpost: [n]buffer.Buffer = undefined;

    for (0..n) |L| {
        bufs_w[L] = try buffer.Buffer.initStatic(&ctx, f32, mlp_cpu.weights[L]);
        bufs_b[L] = try buffer.Buffer.initStatic(&ctx, f32, mlp_cpu.biases[L]);
        const dim_o = layer_dims[L + 1];
        const dim_i = layer_dims[L];
        bufs_dw[L] = try buffer.Buffer.initDeviceOnly(&ctx, dim_o * dim_i * @sizeOf(f32));
        bufs_db[L] = try buffer.Buffer.initDeviceOnly(&ctx, dim_o * @sizeOf(f32));
        bufs_pre[L] = try buffer.Buffer.initDeviceOnly(&ctx, dim_o * @sizeOf(f32));
        bufs_post[L] = try buffer.Buffer.initDeviceOnly(&ctx, dim_o * @sizeOf(f32));
        bufs_dpre[L] = try buffer.Buffer.initDeviceOnly(&ctx, dim_o * @sizeOf(f32));
    }
    for (1..n) |L| {
        bufs_dpost[L] = try buffer.Buffer.initDeviceOnly(&ctx, layer_dims[L] * @sizeOf(f32));
    }
    defer {
        for (0..n) |L| {
            bufs_w[L].deinit(ctx.device);
            bufs_b[L].deinit(ctx.device);
            bufs_dw[L].deinit(ctx.device);
            bufs_db[L].deinit(ctx.device);
            bufs_pre[L].deinit(ctx.device);
            bufs_post[L].deinit(ctx.device);
            bufs_dpre[L].deinit(ctx.device);
        }
        for (1..n) |L| bufs_dpost[L].deinit(ctx.device);
    }

    // ── Pipelines ──────────────────────────────────────────────────
    var k_matmul = try pipeline.Kernel.init(&ctx, &shaders.matmul_nt, 3, @sizeOf(aliases.MatmulPush));
    defer k_matmul.deinit();
    var k_add = try pipeline.Kernel.init(&ctx, &shaders.add_in_place, 2, @sizeOf(runtime.AddInPlacePush));
    defer k_add.deinit();
    var k_relu = try pipeline.Kernel.init(&ctx, &shaders.relu, 2, @sizeOf(aliases.ReluPush));
    defer k_relu.deinit();
    var k_relu_bw = try pipeline.Kernel.init(&ctx, &shaders.relu_backward, 3, @sizeOf(aliases.ReluBackwardPush));
    defer k_relu_bw.deinit();
    var k_lin_dx = try pipeline.Kernel.init(&ctx, &shaders.linear_backward_dx, 3, @sizeOf(aliases.LinearBackwardDxPush));
    defer k_lin_dx.deinit();
    var k_outer = try pipeline.Kernel.init(&ctx, &shaders.outer_product, 3, @sizeOf(aliases.OuterProductPush));
    defer k_outer.deinit();
    var k_copy = try pipeline.Kernel.init(&ctx, &shaders.slice_copy, 2, @sizeOf(aliases.SliceCopyPush));
    defer k_copy.deinit();
    var k_sgd = try pipeline.Kernel.init(&ctx, &shaders.sgd_step, 2, @sizeOf(aliases.SgdStepPush));
    defer k_sgd.deinit();
    var k_mse_grad = try pipeline.Kernel.init(&ctx, &shaders.mse_loss_grad, 3, @sizeOf(aliases.MseLossGradPush));
    defer k_mse_grad.deinit();

    // ── CPU oracle to compare each step against ────────────────────
    var act_cpu = try cpu_train.ActivationsN.init(allocator, &mlp_cpu);
    defer act_cpu.deinit(allocator);
    act_cpu.x = &x;
    var grads_cpu = try cpu_train.GradsN.init(allocator, &mlp_cpu);
    defer grads_cpu.deinit(allocator);

    // Snapshot buffers for the after-step-1 parity check (per layer).
    var w_after_1: [n][]f32 = undefined;
    var b_after_1: [n][]f32 = undefined;
    for (0..n) |L| {
        w_after_1[L] = try allocator.alloc(f32, mlp_cpu.weights[L].len);
        b_after_1[L] = try allocator.alloc(f32, mlp_cpu.biases[L].len);
    }
    defer for (0..n) |L| {
        allocator.free(w_after_1[L]);
        allocator.free(b_after_1[L]);
    };

    var step: u32 = 0;
    while (step < n_steps) : (step += 1) {
        // CPU step.
        _ = try cpu_train.trainStepN(allocator, &mlp_cpu, &act_cpu, &grads_cpu, &target, lr);

        // GPU step.
        var rec = try gpu_recorder.Recorder.init(&ctx, 32, 256);
        defer rec.deinit();
        try rec.begin();

        // Forward pass: layer L reads input[L] = (L==0 ? x : post[L-1]),
        // writes pre[L] = W[L]·input + b[L], then post[L] = ReLU(pre[L])
        // for hidden layers; for the output layer post[L] = pre[L] (we
        // don't dispatch a relu, and downstream readers use pre as y).
        for (0..n) |L| {
            const dim_o: u32 = @intCast(layer_dims[L + 1]);
            const dim_i: u32 = @intCast(layer_dims[L]);
            const input_buf = if (L == 0) &buf_x else &bufs_post[L - 1];
            const matmul_push = aliases.MatmulPush{ .m = 1, .n = dim_o, .k = dim_i };
            try rec.dispatch(&k_matmul, &.{ input_buf, &bufs_w[L], &bufs_pre[L] }, &matmul_push, 1, 1, 1);
            const add_push = runtime.AddInPlacePush{ .n = dim_o };
            try rec.dispatch(&k_add, &.{ &bufs_pre[L], &bufs_b[L] }, &add_push, util.ceilDiv(dim_o, 256), 1, 1);
            if (L + 1 < n) {
                const relu_push = aliases.ReluPush{ .n = dim_o };
                try rec.dispatch(&k_relu, &.{ &bufs_pre[L], &bufs_post[L] }, &relu_push, util.ceilDiv(dim_o, 256), 1, 1);
            } else {
                // Output layer has no ReLU — copy pre→post so backward
                // and the host can both read the prediction from
                // bufs_post[n-1] uniformly.
                const copy_push = aliases.SliceCopyPush{ .src_off = 0, .dst_off = 0, .n_elem = dim_o };
                try rec.dispatch(&k_copy, &.{ &bufs_pre[L], &bufs_post[L] }, &copy_push, util.ceilDiv(dim_o, 256), 1, 1);
            }
        }

        // Loss-grad seeds d_pre on the output layer (no ReLU there).
        const dim_out_u: u32 = @intCast(layer_dims[n]);
        const mse_grad_push = aliases.MseLossGradPush{ .n = dim_out_u };
        try rec.dispatch(&k_mse_grad, &.{ &bufs_post[n - 1], &buf_target, &bufs_dpre[n - 1] }, &mse_grad_push, util.ceilDiv(dim_out_u, 256), 1, 1);

        // Backward pass per layer, top-down. db = d_pre; dW = d_pre ⊗ input;
        // if not the bottom layer, propagate d_post and apply the ReLU mask.
        var Lp1: usize = n;
        while (Lp1 > 0) : (Lp1 -= 1) {
            const L = Lp1 - 1;
            const dim_o: u32 = @intCast(layer_dims[L + 1]);
            const dim_i: u32 = @intCast(layer_dims[L]);
            const input_buf = if (L == 0) &buf_x else &bufs_post[L - 1];

            // db[L] = d_pre[L]
            const copy_db = aliases.SliceCopyPush{ .src_off = 0, .dst_off = 0, .n_elem = dim_o };
            try rec.dispatch(&k_copy, &.{ &bufs_dpre[L], &bufs_db[L] }, &copy_db, util.ceilDiv(dim_o, 256), 1, 1);
            // dW[L] = d_pre[L] ⊗ input
            const op_push = aliases.OuterProductPush{ .dim_out = dim_o, .dim_in = dim_i };
            try rec.dispatch(&k_outer, &.{ &bufs_dpre[L], input_buf, &bufs_dw[L] }, &op_push, util.ceilDiv(dim_o, 16), util.ceilDiv(dim_i, 16), 1);
            // Propagate to layer L-1's d_pre, with ReLU mask on its pre.
            if (L > 0) {
                const lin_dx_push = aliases.LinearBackwardDxPush{ .dim_out = dim_o, .dim_in = dim_i };
                try rec.dispatch(&k_lin_dx, &.{ &bufs_dpre[L], &bufs_w[L], &bufs_dpost[L] }, &lin_dx_push, util.ceilDiv(dim_i, 256), 1, 1);
                const relu_bw_push = aliases.ReluBackwardPush{ .n = dim_i };
                try rec.dispatch(&k_relu_bw, &.{ &bufs_dpost[L], &bufs_pre[L - 1], &bufs_dpre[L - 1] }, &relu_bw_push, util.ceilDiv(dim_i, 256), 1, 1);
            }
        }

        // SGD updates run after all gradients have been computed using
        // pre-update weights. Order across layers doesn't matter.
        for (0..n) |L| {
            const sgd_w_push = aliases.SgdStepPush{ .n = @intCast(mlp_cpu.weights[L].len), .lr = lr };
            try rec.dispatch(&k_sgd, &.{ &bufs_w[L], &bufs_dw[L] }, &sgd_w_push, util.ceilDiv(@intCast(mlp_cpu.weights[L].len), 256), 1, 1);
            const sgd_b_push = aliases.SgdStepPush{ .n = @intCast(mlp_cpu.biases[L].len), .lr = lr };
            try rec.dispatch(&k_sgd, &.{ &bufs_b[L], &bufs_db[L] }, &sgd_b_push, util.ceilDiv(@intCast(mlp_cpu.biases[L].len), 256), 1, 1);
        }

        try rec.endAndSubmit();

        if (step == 0) {
            for (0..n) |L| {
                try bufs_w[L].readBack(&ctx, f32, w_after_1[L]);
                try bufs_b[L].readBack(&ctx, f32, b_after_1[L]);
            }
        }
    }

    // ── Single-step parity ────────────────────────────────────────
    // Re-run step 1 from a fresh oracle (mlp_cpu has already moved on).
    var oracle = try cpu_train.MlpN.init(allocator, &layer_dims, 0.3, 0xC0FFEE3D);
    defer oracle.deinit(allocator);
    for (oracle.biases, 0..) |b, layer_idx| {
        for (b, 0..) |*v, i| {
            const li: f32 = @floatFromInt(layer_idx + 1);
            const ii: f32 = @floatFromInt(@as(i32, @intCast(i % 3)));
            v.* = 0.05 * (li - ii);
        }
    }
    var oracle_act = try cpu_train.ActivationsN.init(allocator, &oracle);
    defer oracle_act.deinit(allocator);
    oracle_act.x = &x;
    var oracle_grads = try cpu_train.GradsN.init(allocator, &oracle);
    defer oracle_grads.deinit(allocator);
    _ = try cpu_train.trainStepN(allocator, &oracle, &oracle_act, &oracle_grads, &target, lr);

    const tol_step1: f32 = 1e-5;
    var max_step1: f32 = 0;
    for (0..n) |L| {
        for (oracle.weights[L], w_after_1[L], 0..) |c, g, i| {
            const d = @abs(g - c);
            if (d > tol_step1) {
                std.debug.print("step-1 W[{d}] MISMATCH[{d}]: gpu={d} cpu={d}\n", .{ L, i, g, c });
                return error.ParityFailed;
            }
            if (d > max_step1) max_step1 = d;
        }
        for (oracle.biases[L], b_after_1[L], 0..) |c, g, i| {
            const d = @abs(g - c);
            if (d > tol_step1) {
                std.debug.print("step-1 b[{d}] MISMATCH[{d}]: gpu={d} cpu={d}\n", .{ L, i, g, c });
                return error.ParityFailed;
            }
            if (d > max_step1) max_step1 = d;
        }
    }

    // ── Full-trajectory parity ────────────────────────────────────
    const tol_traj: f32 = 1e-4;
    var max_traj: f32 = 0;
    for (0..n) |L| {
        const w_gpu = try allocator.alloc(f32, mlp_cpu.weights[L].len);
        defer allocator.free(w_gpu);
        const b_gpu = try allocator.alloc(f32, mlp_cpu.biases[L].len);
        defer allocator.free(b_gpu);
        try bufs_w[L].readBack(&ctx, f32, w_gpu);
        try bufs_b[L].readBack(&ctx, f32, b_gpu);
        for (mlp_cpu.weights[L], w_gpu, 0..) |c, g, i| {
            const d = @abs(g - c);
            if (d > tol_traj) {
                std.debug.print("traj W[{d}] MISMATCH[{d}] @ step {d}: gpu={d} cpu={d}\n", .{ L, i, n_steps, g, c });
                return error.ParityFailed;
            }
            if (d > max_traj) max_traj = d;
        }
        for (mlp_cpu.biases[L], b_gpu, 0..) |c, g, i| {
            const d = @abs(g - c);
            if (d > tol_traj) {
                std.debug.print("traj b[{d}] MISMATCH[{d}] @ step {d}: gpu={d} cpu={d}\n", .{ L, i, n_steps, g, c });
                return error.ParityFailed;
            }
            if (d > max_traj) max_traj = d;
        }
    }

    std.debug.print(
        "PASS GPU MLP train (n={d} layers, dims [4,8,6,4], {d} SGD steps; step-1 |Δ|={e}, after-{d} |Δ|={e})\n",
        .{ n, n_steps, max_step1, n_steps, max_traj },
    );
}

// ── TrainingRunner smoke: persistent buffers, streamed inputs ───────
//
// Chunk 5 of training-v0. Exercises the public TrainingRunner API:
// init() builds pipelines + buffers once, tickStep() streams a fresh
// (input, target) per call and returns the prediction, tickPredict()
// is forward-only. The smoke runs a small training schedule against
// two distinct (x, target) pairs alternating each step (mimicking a
// streaming task) and asserts loss converges; final readWeights()
// shape-checks the buffers can be pulled back to a CPU `Mlp`.
//
// Single submit per tick: this is the same call shape the engine-
// integration `aiTrain` hook will use, just with a Recorder.attachCmd
// in chunk 7 instead of a standalone Recorder.

pub fn runTrainingRunnerSmoke(allocator: std.mem.Allocator) !void {
    var ctx = try vk.Context.init(allocator);
    defer ctx.deinit();

    const cfg = train_runner.Mlp2Config{
        .dim_in = 4,
        .dim_hidden = 8,
        .dim_out = 4,
        .lr = 0.05,
        .init_seed = 0x70077077,
    };

    var runner = try train_runner.TrainingRunner.init(allocator, &ctx, cfg);
    defer runner.deinit();

    // Two streaming (input, target) pairs — alternating per step.
    // Single-pair convergence already covered in earlier smokes; here
    // we want to see the runner handle a moving target without state
    // smearing between calls.
    const x_a = [_]f32{ 1.0, 0.0, 0.0, 0.0 };
    const t_a = [_]f32{ 1.0, 0.2, 0.0, 0.0 };
    const x_b = [_]f32{ 0.0, 1.0, 0.0, 0.0 };
    const t_b = [_]f32{ 0.0, 0.0, 0.7, 0.3 };

    var pred: [4]f32 = undefined;

    // Initial loss across both pairs.
    try runner.tickPredict(&x_a, &pred);
    var initial_loss = cpu_train.mseLoss(&pred, &t_a);
    try runner.tickPredict(&x_b, &pred);
    initial_loss += cpu_train.mseLoss(&pred, &t_b);

    // Run a handful of steps alternating between the two pairs.
    const n_steps: u32 = 200;
    var s: u32 = 0;
    while (s < n_steps) : (s += 1) {
        if (s & 1 == 0) {
            try runner.tickStep(&x_a, &t_a, &pred);
        } else {
            try runner.tickStep(&x_b, &t_b, &pred);
        }
    }

    // Final loss across both pairs.
    try runner.tickPredict(&x_a, &pred);
    var final_loss = cpu_train.mseLoss(&pred, &t_a);
    try runner.tickPredict(&x_b, &pred);
    final_loss += cpu_train.mseLoss(&pred, &t_b);

    if (!(final_loss < initial_loss * 0.1)) {
        std.debug.print(
            "TrainingRunner did not converge: loss[0] = {d:.6}, loss[{d}] = {d:.6}\n",
            .{ initial_loss, n_steps, final_loss },
        );
        return error.ParityFailed;
    }

    // readWeights round-trip: build a CPU MLP of matching shape and
    // pull weights back. Sanity check that dimensions agree and the
    // staging path works.
    var cpu_mirror = try cpu_train.Mlp.init(allocator, cfg.dim_in, cfg.dim_hidden, cfg.dim_out, 0.0, 0);
    defer cpu_mirror.deinit(allocator);
    try runner.readWeights(&cpu_mirror);
    var nan_count: usize = 0;
    for (cpu_mirror.w1) |v| if (std.math.isNan(v)) {
        nan_count += 1;
    };
    if (nan_count != 0) {
        std.debug.print("TrainingRunner readWeights returned NaNs in W1: {d}\n", .{nan_count});
        return error.ParityFailed;
    }

    std.debug.print(
        "PASS TrainingRunner ({d} alternating steps; loss {d:.6} → {d:.6}, readWeights OK)\n",
        .{ n_steps, initial_loss, final_loss },
    );
}

// ── TrainingRunner attach-mode smoke: host owns submit, valkyr records ─
//
// Chunk 7 (valkyr-side) of training-v0. Demonstrates the attach
// surface that a host engine like Matryoshka uses: the host owns the
// VkContext, the per-frame VkCommandBuffer, and the submit cadence;
// valkyr's TrainingRunner records its forward + loss-grad + backward
// + SGD into the host's command buffer via `tickStepRecord`. No
// per-step submit from valkyr's side — the host bundles everything
// into one render submit.
//
// The smoke runs 60 attached steps (a "second of frames at 60 fps"
// cadence) and asserts loss converges. Visual click-to-red sphere
// demo lives in the matryoshka repo on top of this surface.

pub fn runTrainingRunnerAttachedSmoke(allocator: std.mem.Allocator) !void {
    var ctx = try vk.Context.init(allocator);
    defer ctx.deinit();

    const cfg = train_runner.Mlp2Config{
        .dim_in = 4,
        .dim_hidden = 16,
        .dim_out = 4,
        .lr = 0.05,
        .init_seed = 0xA77ACED,
    };
    var runner = try train_runner.TrainingRunner.init(allocator, &ctx, cfg);
    defer runner.deinit();

    // ── Host-owned cmd buffer + fence (matryoshka-equivalent setup) ──
    var cb_ai = std.mem.zeroes(vk.c.VkCommandBufferAllocateInfo);
    cb_ai.sType = vk.c.VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    cb_ai.commandPool = ctx.cmd_pool;
    cb_ai.level = vk.c.VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cb_ai.commandBufferCount = 1;
    var host_cmd: vk.c.VkCommandBuffer = null;
    try vk.check(vk.c.vkAllocateCommandBuffers(ctx.device, &cb_ai, &host_cmd));
    defer vk.c.vkFreeCommandBuffers(ctx.device, ctx.cmd_pool, 1, &host_cmd);

    var fci = std.mem.zeroes(vk.c.VkFenceCreateInfo);
    fci.sType = vk.c.VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    var host_fence: vk.c.VkFence = null;
    try vk.check(vk.c.vkCreateFence(ctx.device, &fci, null, &host_fence));
    defer vk.c.vkDestroyFence(ctx.device, host_fence, null);

    // Streaming target signal: same shape as the headless demo, just
    // 60 frames worth.
    const dt: f32 = 0.05;
    var pred: [4]f32 = undefined;
    var input: [4]f32 = undefined;
    var target: [4]f32 = undefined;

    // Initial loss for convergence check.
    try runner.tickPredict(&[_]f32{ 1, 0, 0, 0 }, &pred);
    const initial_loss = cpu_train.mseLoss(&pred, &[_]f32{ 0.5, 0.5, 0.5, 0 });

    const n_frames: u32 = 60;
    var f: u32 = 0;
    while (f < n_frames) : (f += 1) {
        const t = @as(f32, @floatFromInt(f)) * dt;
        input[0] = @sin(t);
        input[1] = @cos(t);
        input[2] = @sin(2 * t);
        input[3] = @cos(2 * t);
        target[0] = 0.5 + 0.5 * @sin(t);
        target[1] = 0.5 + 0.5 * @cos(t);
        target[2] = 0.5;
        target[3] = 0.0;

        // Per-frame: reset + begin + attach + record + end + submit + wait.
        try vk.check(vk.c.vkResetCommandBuffer(host_cmd, 0));
        try vk.check(vk.c.vkResetFences(ctx.device, 1, &host_fence));

        var bi = std.mem.zeroes(vk.c.VkCommandBufferBeginInfo);
        bi.sType = vk.c.VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        bi.flags = vk.c.VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        try vk.check(vk.c.vkBeginCommandBuffer(host_cmd, &bi));

        var rec = try gpu_recorder.Recorder.attachCmd(&ctx, host_cmd, 24, 96);
        defer rec.deinit();
        try rec.begin();

        try runner.tickStepRecord(&rec, &input, &target);

        try vk.check(vk.c.vkEndCommandBuffer(host_cmd));
        var submit = std.mem.zeroes(vk.c.VkSubmitInfo);
        submit.sType = vk.c.VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submit.commandBufferCount = 1;
        submit.pCommandBuffers = &host_cmd;
        try vk.check(vk.c.vkQueueSubmit(ctx.queue, 1, &submit, host_fence));
        const timeout_ns: u64 = 10 * 1_000_000_000;
        try vk.check(vk.c.vkWaitForFences(ctx.device, 1, &host_fence, vk.c.VK_TRUE, timeout_ns));
    }

    // Final loss against the last frame's target.
    try runner.tickPredict(&input, &pred);
    const final_loss = cpu_train.mseLoss(&pred, &target);

    if (!(final_loss < initial_loss * 0.5)) {
        std.debug.print(
            "TrainingRunner attached did not converge: loss[0] = {d:.6}, loss[{d}] = {d:.6}\n",
            .{ initial_loss, n_frames, final_loss },
        );
        return error.ParityFailed;
    }

    std.debug.print(
        "PASS TrainingRunner attached ({d} host-submitted frames; loss {d:.6} → {d:.6}, valkyr-record + host-submit OK)\n",
        .{ n_frames, initial_loss, final_loss },
    );
}

// ── MlpNRunner standalone smoke ─────────────────────────────────────
//
// Tier-1 chunk 3a of the post-v0 training arc. Exercises the new
// multi-layer runner on a 4 → 8 → 6 → 4 net (n=3) using the same
// alternating-pair convergence shape as the 2-layer
// `runTrainingRunnerSmoke`. Standalone path: each tick owns its own
// Recorder and submits.

pub fn runTrainingRunnerNSmoke(allocator: std.mem.Allocator) !void {
    var ctx = try vk.Context.init(allocator);
    defer ctx.deinit();

    const layer_dims = [_]u32{ 4, 8, 6, 4 };
    const cfg = train_runner_n.MlpNConfig{
        .layer_dims = &layer_dims,
        .lr = 0.05,
        .init_seed = 0x70077077,
    };

    var runner = try train_runner_n.MlpNRunner.init(allocator, &ctx, cfg);
    defer runner.deinit();

    const x_a = [_]f32{ 1.0, 0.0, 0.0, 0.0 };
    const t_a = [_]f32{ 1.0, 0.2, 0.0, 0.0 };
    const x_b = [_]f32{ 0.0, 1.0, 0.0, 0.0 };
    const t_b = [_]f32{ 0.0, 0.0, 0.7, 0.3 };

    var pred: [4]f32 = undefined;

    try runner.tickPredict(&x_a, &pred);
    var initial_loss = cpu_train.mseLoss(&pred, &t_a);
    try runner.tickPredict(&x_b, &pred);
    initial_loss += cpu_train.mseLoss(&pred, &t_b);

    // Deeper net + vanilla SGD on a 2-pair alternating task is slower
    // to converge than the 2-layer counterpart — gradient attenuation
    // through the extra ReLU stage. 600 steps at lr=0.05 lands an
    // 8× drop comfortably; tightening past that is an Adam discussion
    // (chunk 3b), not a runner correctness one.
    const n_steps: u32 = 600;
    var s: u32 = 0;
    while (s < n_steps) : (s += 1) {
        if (s & 1 == 0) {
            try runner.tickStep(&x_a, &t_a, &pred);
        } else {
            try runner.tickStep(&x_b, &t_b, &pred);
        }
    }

    try runner.tickPredict(&x_a, &pred);
    var final_loss = cpu_train.mseLoss(&pred, &t_a);
    try runner.tickPredict(&x_b, &pred);
    final_loss += cpu_train.mseLoss(&pred, &t_b);

    if (!(final_loss < initial_loss * 0.125)) {
        std.debug.print(
            "MlpNRunner did not converge: loss[0] = {d:.6}, loss[{d}] = {d:.6}\n",
            .{ initial_loss, n_steps, final_loss },
        );
        return error.ParityFailed;
    }

    // readWeights round-trip: build a CPU MlpN of matching shape and
    // pull weights back. Verifies dimensions agree and the staging
    // path runs through cleanly.
    const seed_dims = [_]usize{ 4, 8, 6, 4 };
    var cpu_mirror = try cpu_train.MlpN.init(allocator, &seed_dims, 0.0, 0);
    defer cpu_mirror.deinit(allocator);
    try runner.readWeights(&cpu_mirror);
    var nan_count: usize = 0;
    for (cpu_mirror.weights) |w| for (w) |v| if (std.math.isNan(v)) {
        nan_count += 1;
    };
    if (nan_count != 0) {
        std.debug.print("MlpNRunner readWeights returned NaNs: {d}\n", .{nan_count});
        return error.ParityFailed;
    }

    std.debug.print(
        "PASS MlpNRunner standalone (n={d} layers, dims [4,8,6,4]; {d} alternating steps; loss {d:.6} → {d:.6}, readWeights OK)\n",
        .{ runner.nLayers(), n_steps, initial_loss, final_loss },
    );
}

// ── MlpNRunner attach-mode smoke ────────────────────────────────────
//
// Same setup as `runTrainingRunnerAttachedSmoke` but on the multi-layer
// runner — host owns the cmd buffer + fence, valkyr records into it.

pub fn runTrainingRunnerNAttachedSmoke(allocator: std.mem.Allocator) !void {
    var ctx = try vk.Context.init(allocator);
    defer ctx.deinit();

    const layer_dims = [_]u32{ 4, 12, 8, 4 };
    const cfg = train_runner_n.MlpNConfig{
        .layer_dims = &layer_dims,
        .lr = 0.05,
        .init_seed = 0xA77ACED2,
    };
    var runner = try train_runner_n.MlpNRunner.init(allocator, &ctx, cfg);
    defer runner.deinit();

    var cb_ai = std.mem.zeroes(vk.c.VkCommandBufferAllocateInfo);
    cb_ai.sType = vk.c.VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    cb_ai.commandPool = ctx.cmd_pool;
    cb_ai.level = vk.c.VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cb_ai.commandBufferCount = 1;
    var host_cmd: vk.c.VkCommandBuffer = null;
    try vk.check(vk.c.vkAllocateCommandBuffers(ctx.device, &cb_ai, &host_cmd));
    defer vk.c.vkFreeCommandBuffers(ctx.device, ctx.cmd_pool, 1, &host_cmd);

    var fci = std.mem.zeroes(vk.c.VkFenceCreateInfo);
    fci.sType = vk.c.VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    var host_fence: vk.c.VkFence = null;
    try vk.check(vk.c.vkCreateFence(ctx.device, &fci, null, &host_fence));
    defer vk.c.vkDestroyFence(ctx.device, host_fence, null);

    const dt: f32 = 0.05;
    var pred: [4]f32 = undefined;
    var input: [4]f32 = undefined;
    var target: [4]f32 = undefined;

    try runner.tickPredict(&[_]f32{ 1, 0, 0, 0 }, &pred);
    const initial_loss = cpu_train.mseLoss(&pred, &[_]f32{ 0.5, 0.5, 0.5, 0 });

    const n_frames: u32 = 60;
    var f: u32 = 0;
    while (f < n_frames) : (f += 1) {
        const t = @as(f32, @floatFromInt(f)) * dt;
        input[0] = @sin(t);
        input[1] = @cos(t);
        input[2] = @sin(2 * t);
        input[3] = @cos(2 * t);
        target[0] = 0.5 + 0.5 * @sin(t);
        target[1] = 0.5 + 0.5 * @cos(t);
        target[2] = 0.5;
        target[3] = 0.0;

        try vk.check(vk.c.vkResetCommandBuffer(host_cmd, 0));
        try vk.check(vk.c.vkResetFences(ctx.device, 1, &host_fence));

        var bi = std.mem.zeroes(vk.c.VkCommandBufferBeginInfo);
        bi.sType = vk.c.VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        bi.flags = vk.c.VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        try vk.check(vk.c.vkBeginCommandBuffer(host_cmd, &bi));

        var rec = try gpu_recorder.Recorder.attachCmd(&ctx, host_cmd, 48, 192);
        defer rec.deinit();
        try rec.begin();

        try runner.tickStepRecord(&rec, &input, &target);

        try vk.check(vk.c.vkEndCommandBuffer(host_cmd));
        var submit = std.mem.zeroes(vk.c.VkSubmitInfo);
        submit.sType = vk.c.VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submit.commandBufferCount = 1;
        submit.pCommandBuffers = &host_cmd;
        try vk.check(vk.c.vkQueueSubmit(ctx.queue, 1, &submit, host_fence));
        const timeout_ns: u64 = 10 * 1_000_000_000;
        try vk.check(vk.c.vkWaitForFences(ctx.device, 1, &host_fence, vk.c.VK_TRUE, timeout_ns));
    }

    try runner.tickPredict(&input, &pred);
    const final_loss = cpu_train.mseLoss(&pred, &target);

    if (!(final_loss < initial_loss * 0.5)) {
        std.debug.print(
            "MlpNRunner attached did not converge: loss[0] = {d:.6}, loss[{d}] = {d:.6}\n",
            .{ initial_loss, n_frames, final_loss },
        );
        return error.ParityFailed;
    }

    std.debug.print(
        "PASS MlpNRunner attached (n={d} layers, dims [4,12,8,4]; {d} host-submitted frames; loss {d:.6} → {d:.6})\n",
        .{ runner.nLayers(), n_frames, initial_loss, final_loss },
    );
}

// ── Weight-decay + cosine-LR smoke ─────────────────────────────────
//
// Tier-1 chunk 5 of the post-v0 training arc. Two claims:
//
//   (1) Weight-decay shrinks W toward zero in the absence of a
//       gradient signal. We feed a constant (x=0, target=0) every step;
//       MSE loss is then 0 and the gradient is identically zero, so
//       any change in ‖W‖ comes entirely from the wd shrinkage. After
//       N steps the weight L2 norm should equal ‖W₀‖·(1 − lr·wd)^N.
//
//   (2) `cosineLr` returns lr_max at step 0, lr_min at step ≥ T, and
//       the half-cosine in between. Pure host-side helper — sanity
//       check three sample points.
//
// Both runners share the SGD shader, so testing on MlpNRunner covers
// the TrainingRunner path too. (Adam shader gets the same wd push;
// Adam parity vs CPU is left for chunk 5b — math is straightforward
// AdamW, the more interesting test is a longer-horizon convergence
// run.)

pub fn runWeightDecayCosineLrSmoke(allocator: std.mem.Allocator) !void {
    var ctx = try vk.Context.init(allocator);
    defer ctx.deinit();

    // ── (1) Weight-decay shrinkage with zero gradient ─────────────
    const layer_dims = [_]u32{ 4, 8, 4 };
    const lr: f32 = 0.05;
    const wd: f32 = 0.1;
    const n_steps: u32 = 30;

    var runner = try train_runner_n.MlpNRunner.init(allocator, &ctx, .{
        .layer_dims = &layer_dims,
        .lr = lr,
        .weight_decay = wd,
        .init_seed = 0xCDDECA1,
    });
    defer runner.deinit();

    // Snapshot initial L2 norms per layer.
    const seed_dims = [_]usize{ 4, 8, 4 };
    var mirror = try cpu_train.MlpN.init(allocator, &seed_dims, 0.0, 0);
    defer mirror.deinit(allocator);
    try runner.readWeights(&mirror);

    var w_norm0: [2]f32 = undefined;
    for (mirror.weights, 0..) |w, L| {
        var s: f64 = 0;
        for (w) |v| s += @as(f64, v) * @as(f64, v);
        w_norm0[L] = @floatCast(@sqrt(s));
    }

    // Step with zero (x, target). Loss-grad is zero, so the only
    // change to W is the decoupled shrinkage W ← W·(1 − lr·wd).
    const zero_x = [_]f32{ 0, 0, 0, 0 };
    const zero_t = [_]f32{ 0, 0, 0, 0 };
    var s: u32 = 0;
    while (s < n_steps) : (s += 1) {
        try runner.tickStep(&zero_x, &zero_t, null);
    }

    try runner.readWeights(&mirror);
    const expected_factor: f32 = std.math.pow(f32, 1.0 - lr * wd, @as(f32, @floatFromInt(n_steps)));
    var max_rel_err: f32 = 0;
    for (mirror.weights, 0..) |w, L| {
        var s2: f64 = 0;
        for (w) |v| s2 += @as(f64, v) * @as(f64, v);
        const norm_after: f32 = @floatCast(@sqrt(s2));
        const expected = w_norm0[L] * expected_factor;
        const rel_err = @abs(norm_after - expected) / expected;
        if (rel_err > max_rel_err) max_rel_err = rel_err;
        if (rel_err > 1e-3) {
            std.debug.print(
                "weight-decay shrinkage off on layer W[{d}]: ‖W‖={d:.6}, expected={d:.6}, rel_err={d:.4}\n",
                .{ L, norm_after, expected, rel_err },
            );
            return error.ParityFailed;
        }
    }

    // Biases must be untouched (wd = 0 on bias dispatches).
    for (mirror.biases) |b| {
        for (b) |v| {
            if (v != 0.0) {
                std.debug.print("bias drift under wd: expected 0, got {d:.6}\n", .{v});
                return error.ParityFailed;
            }
        }
    }

    // ── (2) cosineLr endpoints + midpoint ─────────────────────────
    const lr_max: f32 = 0.1;
    const lr_min: f32 = 0.01;
    const total: u32 = 1000;
    const lr0 = train_runner_n.cosineLr(0, total, lr_max, lr_min);
    const lrT = train_runner_n.cosineLr(total, total, lr_max, lr_min);
    const lrMid = train_runner_n.cosineLr(total / 2, total, lr_max, lr_min);

    if (@abs(lr0 - lr_max) > 1e-6) {
        std.debug.print("cosineLr(0) = {d:.6}, expected {d:.6}\n", .{ lr0, lr_max });
        return error.ParityFailed;
    }
    if (@abs(lrT - lr_min) > 1e-6) {
        std.debug.print("cosineLr(T) = {d:.6}, expected {d:.6}\n", .{ lrT, lr_min });
        return error.ParityFailed;
    }
    // At step T/2 the cosine = 0, so lr = lr_min + (lr_max - lr_min)/2.
    const mid_expected = lr_min + (lr_max - lr_min) * 0.5;
    if (@abs(lrMid - mid_expected) > 1e-3) {
        std.debug.print("cosineLr(T/2) = {d:.6}, expected {d:.6}\n", .{ lrMid, mid_expected });
        return error.ParityFailed;
    }

    std.debug.print(
        "PASS weight decay + cosine LR (W shrunk by (1 − lr·wd)^{d} = {d:.4}, max rel-err = {e}; cosine LR endpoints + mid OK)\n",
        .{ n_steps, expected_factor, max_rel_err },
    );
}

// ── TrainingRunner batched-predict parity smoke ─────────────────────
//
// Verifies the new `tickPredictBatch` against N sequential
// `tickPredict` calls. Both should be bit-identical because the
// batched shader is just N independent applications of the same
// forward formula — same fp32 ops, same accumulation order.

pub fn runTrainingRunnerBatchedSmoke(allocator: std.mem.Allocator) !void {
    var ctx = try vk.Context.init(allocator);
    defer ctx.deinit();

    const n_samples: u32 = 256;
    const cfg = train_runner.Mlp2Config{
        .dim_in = 5,
        .dim_hidden = 16,
        .dim_out = 3,
        .lr = 0.0,
        .init_seed = 0xBA7CCED,
        .max_batch_size = n_samples,
    };
    var runner = try train_runner.TrainingRunner.init(allocator, &ctx, cfg);
    defer runner.deinit();

    // Build a dense input grid: U-grid × V-grid plus the Fourier-style
    // features the matryoshka demo will use. Same shape so the smoke
    // exercises the realistic path.
    const grid: u32 = 16;
    if (grid * grid != n_samples) return error.SetupBug;
    var x_batch = try allocator.alloc(f32, n_samples * cfg.dim_in);
    defer allocator.free(x_batch);
    for (0..grid) |gv| {
        for (0..grid) |gu| {
            const idx = (gv * grid + gu) * cfg.dim_in;
            const u = @as(f32, @floatFromInt(gu)) / @as(f32, @floatFromInt(grid - 1));
            const v = @as(f32, @floatFromInt(gv)) / @as(f32, @floatFromInt(grid - 1));
            x_batch[idx + 0] = u;
            x_batch[idx + 1] = v;
            x_batch[idx + 2] = @sin(2 * std.math.pi * u);
            x_batch[idx + 3] = @cos(2 * std.math.pi * u);
            x_batch[idx + 4] = @sin(2 * std.math.pi * v);
        }
    }

    // ── Batched predict ────────────────────────────────────────────
    const y_batched = try allocator.alloc(f32, n_samples * cfg.dim_out);
    defer allocator.free(y_batched);
    try runner.tickPredictBatch(x_batch, y_batched);

    // ── Sequential predict, sample by sample ───────────────────────
    const y_sequential = try allocator.alloc(f32, n_samples * cfg.dim_out);
    defer allocator.free(y_sequential);
    var x_one: [5]f32 = undefined;
    var y_one: [3]f32 = undefined;
    for (0..n_samples) |i| {
        @memcpy(&x_one, x_batch[i * cfg.dim_in ..][0..cfg.dim_in]);
        try runner.tickPredict(&x_one, &y_one);
        @memcpy(y_sequential[i * cfg.dim_out ..][0..cfg.dim_out], &y_one);
    }

    // ── Parity ────────────────────────────────────────────────────
    var max_abs: f32 = 0;
    for (y_batched, y_sequential, 0..) |b, s, i| {
        const d = @abs(b - s);
        if (d > 1e-5) {
            std.debug.print(
                "tickPredictBatch MISMATCH at {d}: batched={d:.7} sequential={d:.7}\n",
                .{ i, b, s },
            );
            return error.ParityFailed;
        }
        if (d > max_abs) max_abs = d;
    }
    std.debug.print(
        "PASS TrainingRunner batched predict ({d} samples, dim 5→16→3; max |Δ| vs sequential = {e})\n",
        .{ n_samples, max_abs },
    );
}

// ── TrainingRunner batched-train parity smoke ───────────────────────
//
// Validates `tickStepBatch` against a CPU oracle that simulates the
// same averaged-batch SGD step. The oracle does:
//   for each sample n: forward + per-sample grads
//   sum grads across samples; divide by N
//   SGD step
// then runs the same cycle GPU-side and asserts final weights match
// within fp32 tolerance over K consecutive steps. K = 8 catches any
// drift that would compound across steps; tol is loose vs single-step
// because outer-product accumulation order across N samples matters.

pub fn runTrainingRunnerBatchedTrainSmoke(allocator: std.mem.Allocator) !void {
    var ctx = try vk.Context.init(allocator);
    defer ctx.deinit();

    const n_samples: u32 = 16;
    const cfg = train_runner.Mlp2Config{
        .dim_in = 4,
        .dim_hidden = 8,
        .dim_out = 3,
        .lr = 0.05,
        .init_seed = 0xBA7C7AA,
        .max_batch_size = n_samples,
    };
    var runner = try train_runner.TrainingRunner.init(allocator, &ctx, cfg);
    defer runner.deinit();

    // CPU mirror — same seed + scale = same starting MLP.
    var cpu_mlp = try cpu_train.Mlp.init(allocator, cfg.dim_in, cfg.dim_hidden, cfg.dim_out, cfg.init_scale, cfg.init_seed);
    defer cpu_mlp.deinit(allocator);

    // Build a fixed batch of synthetic (x, target) pairs.
    var rng = std.Random.DefaultPrng.init(0x5EED5);
    const r = rng.random();
    const x_batch = try allocator.alloc(f32, n_samples * cfg.dim_in);
    defer allocator.free(x_batch);
    const target_batch = try allocator.alloc(f32, n_samples * cfg.dim_out);
    defer allocator.free(target_batch);
    for (x_batch) |*v| v.* = r.float(f32) * 2 - 1;
    for (target_batch) |*v| v.* = r.float(f32);

    // CPU oracle: K averaged-batch SGD steps.
    const K: u32 = 8;
    const cpu_grads = try cpu_train.Grads.init(allocator, &cpu_mlp);
    defer @constCast(&cpu_grads).deinit(allocator);
    const acc_dw1 = try allocator.alloc(f32, cpu_mlp.w1.len);
    defer allocator.free(acc_dw1);
    const acc_db1 = try allocator.alloc(f32, cpu_mlp.b1.len);
    defer allocator.free(acc_db1);
    const acc_dw2 = try allocator.alloc(f32, cpu_mlp.w2.len);
    defer allocator.free(acc_dw2);
    const acc_db2 = try allocator.alloc(f32, cpu_mlp.b2.len);
    defer allocator.free(acc_db2);
    const cpu_h_pre = try allocator.alloc(f32, cfg.dim_hidden);
    defer allocator.free(cpu_h_pre);
    const cpu_h = try allocator.alloc(f32, cfg.dim_hidden);
    defer allocator.free(cpu_h);
    const cpu_y = try allocator.alloc(f32, cfg.dim_out);
    defer allocator.free(cpu_y);
    const cpu_dy = try allocator.alloc(f32, cfg.dim_out);
    defer allocator.free(cpu_dy);

    var step: u32 = 0;
    while (step < K) : (step += 1) {
        @memset(acc_dw1, 0);
        @memset(acc_db1, 0);
        @memset(acc_dw2, 0);
        @memset(acc_db2, 0);
        for (0..n_samples) |n| {
            const x_row = x_batch[n * cfg.dim_in ..][0..cfg.dim_in];
            const t_row = target_batch[n * cfg.dim_out ..][0..cfg.dim_out];
            var act: cpu_train.Activations = .{
                .x = x_row,
                .h_pre = cpu_h_pre,
                .h = cpu_h,
                .y = cpu_y,
            };
            cpu_train.forward(&cpu_mlp, &act);
            cpu_train.mseLossGrad(cpu_dy, cpu_y, t_row);
            var sample_grads: cpu_train.Grads = .{
                .dw1 = cpu_grads.dw1,
                .db1 = cpu_grads.db1,
                .dw2 = cpu_grads.dw2,
                .db2 = cpu_grads.db2,
            };
            try cpu_train.backward(allocator, &cpu_mlp, &act, cpu_dy, &sample_grads);
            for (acc_dw1, sample_grads.dw1) |*a, g| a.* += g;
            for (acc_db1, sample_grads.db1) |*a, g| a.* += g;
            for (acc_dw2, sample_grads.dw2) |*a, g| a.* += g;
            for (acc_db2, sample_grads.db2) |*a, g| a.* += g;
        }
        const inv_n: f32 = 1.0 / @as(f32, @floatFromInt(n_samples));
        for (acc_dw1) |*v| v.* *= inv_n;
        for (acc_db1) |*v| v.* *= inv_n;
        for (acc_dw2) |*v| v.* *= inv_n;
        for (acc_db2) |*v| v.* *= inv_n;
        const final_grads: cpu_train.Grads = .{
            .dw1 = acc_dw1,
            .db1 = acc_db1,
            .dw2 = acc_dw2,
            .db2 = acc_db2,
        };
        cpu_train.sgdStep(&cpu_mlp, &final_grads, cfg.lr);

        // Mirror on GPU.
        try runner.tickStepBatch(x_batch, target_batch);
    }

    // Compare final weights.
    const gpu_w1 = try allocator.alloc(f32, cpu_mlp.w1.len);
    defer allocator.free(gpu_w1);
    const gpu_b1 = try allocator.alloc(f32, cpu_mlp.b1.len);
    defer allocator.free(gpu_b1);
    const gpu_w2 = try allocator.alloc(f32, cpu_mlp.w2.len);
    defer allocator.free(gpu_w2);
    const gpu_b2 = try allocator.alloc(f32, cpu_mlp.b2.len);
    defer allocator.free(gpu_b2);
    // Construct an Mlp view whose slices alias our scratch buffers —
    // bypass Mlp.init so no allocator-owned weights leak when we
    // immediately overwrite them via readWeights.
    var gpu_mlp: cpu_train.Mlp = .{
        .dim_in = cfg.dim_in,
        .dim_hidden = cfg.dim_hidden,
        .dim_out = cfg.dim_out,
        .w1 = gpu_w1,
        .b1 = gpu_b1,
        .w2 = gpu_w2,
        .b2 = gpu_b2,
    };
    try runner.readWeights(&gpu_mlp);

    const tol: f32 = 1e-4;
    var max_abs: f32 = 0;
    const ParamCase = struct { name: []const u8, gpu: []const f32, cpu: []const f32 };
    const cases = [_]ParamCase{
        .{ .name = "W1", .gpu = gpu_w1, .cpu = cpu_mlp.w1 },
        .{ .name = "b1", .gpu = gpu_b1, .cpu = cpu_mlp.b1 },
        .{ .name = "W2", .gpu = gpu_w2, .cpu = cpu_mlp.w2 },
        .{ .name = "b2", .gpu = gpu_b2, .cpu = cpu_mlp.b2 },
    };
    for (cases) |cs| {
        for (cs.gpu, cs.cpu, 0..) |g, c, i| {
            const d = @abs(g - c);
            if (d > tol) {
                std.debug.print(
                    "tickStepBatch MISMATCH on {s}[{d}] @ step {d}: gpu={d:.7} cpu={d:.7}\n",
                    .{ cs.name, i, K, g, c },
                );
                return error.ParityFailed;
            }
            if (d > max_abs) max_abs = d;
        }
    }
    std.debug.print(
        "PASS TrainingRunner batched train ({d} samples × {d} steps, dim 4→8→3; max |Δ| vs CPU oracle = {e})\n",
        .{ n_samples, K, max_abs },
    );
}

// ── TrainingRunner cooperative-tickFrame smoke ──────────────────────
//
// Exercises the host-driven frame lifecycle: host owns the
// VkCommandBuffer, runner records training + predict into it, host
// submits once per frame. Verifies the budget API matches Session's
// shape (.steps / .microseconds / .either) by checking:
//
//   1. .steps = N caps the recorded step count exactly at N
//   2. .microseconds = 0 records 0 steps (early-out path)
//   3. .either = { steps = 1, microseconds = huge } caps at 1 step
//
// Plus end-to-end convergence: train for ~30 host frames, predictions
// at the supervised UVs end up close to their targets — same kind of
// proof we used for chunks 5/7 but routed entirely through the
// cooperative attach API.

pub fn runTrainingRunnerCoopSmoke(allocator: std.mem.Allocator) !void {
    var ctx = try vk.Context.init(allocator);
    defer ctx.deinit();

    const n_train: u32 = 8;
    const n_predict: u32 = 8;
    const cfg = train_runner.Mlp2Config{
        .dim_in = 4,
        .dim_hidden = 16,
        .dim_out = 3,
        .lr = 0.1,
        .init_seed = 0x7177EF00,
        .max_batch_size = @max(n_train, n_predict),
    };
    var runner = try train_runner.TrainingRunner.init(allocator, &ctx, cfg);
    defer runner.deinit();

    // Synthetic training set + predict UVs. Inputs are random, targets
    // are a simple linear function of the input — easy regression for
    // a 16-wide hidden layer.
    var rng = std.Random.DefaultPrng.init(0xCD0CD0);
    const r = rng.random();
    const x_train = try allocator.alloc(f32, n_train * cfg.dim_in);
    defer allocator.free(x_train);
    const t_train = try allocator.alloc(f32, n_train * cfg.dim_out);
    defer allocator.free(t_train);
    for (x_train) |*v| v.* = r.float(f32) * 2 - 1;
    for (0..n_train) |i| {
        const xr = x_train[i * cfg.dim_in ..][0..cfg.dim_in];
        const tr = t_train[i * cfg.dim_out ..][0..cfg.dim_out];
        // Trivial mapping: target[o] = sum_k x[k] * 0.5
        for (0..cfg.dim_out) |o| {
            var acc: f32 = 0;
            for (xr) |xk| acc += xk * 0.5;
            tr[o] = acc * (0.7 + 0.1 * @as(f32, @floatFromInt(o)));
        }
    }
    const x_predict = try allocator.alloc(f32, n_predict * cfg.dim_in);
    defer allocator.free(x_predict);
    @memcpy(x_predict, x_train); // predict on the same UVs the trainer sees

    try runner.uploadTrainBatch(x_train, t_train);
    try runner.uploadPredictInputs(x_predict);

    // ── Host-owned cmd buffer + fence ──
    var cb_ai = std.mem.zeroes(vk.c.VkCommandBufferAllocateInfo);
    cb_ai.sType = vk.c.VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    cb_ai.commandPool = ctx.cmd_pool;
    cb_ai.level = vk.c.VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cb_ai.commandBufferCount = 1;
    var host_cmd: vk.c.VkCommandBuffer = null;
    try vk.check(vk.c.vkAllocateCommandBuffers(ctx.device, &cb_ai, &host_cmd));
    defer vk.c.vkFreeCommandBuffers(ctx.device, ctx.cmd_pool, 1, &host_cmd);

    var fci = std.mem.zeroes(vk.c.VkFenceCreateInfo);
    fci.sType = vk.c.VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    var host_fence: vk.c.VkFence = null;
    try vk.check(vk.c.vkCreateFence(ctx.device, &fci, null, &host_fence));
    defer vk.c.vkDestroyFence(ctx.device, host_fence, null);

    // Helper: run one host frame with a given budget. Returns the
    // tick result so callers can assert step counts.
    const runFrame = struct {
        fn f(
            rn: *train_runner.TrainingRunner,
            ct: *const vk.Context,
            cmd: vk.c.VkCommandBuffer,
            fnc: vk.c.VkFence,
            budget: train_runner.TrainBudget,
            n_tr: u32,
            n_pr: u32,
        ) !train_runner.TrainTickResult {
            try vk.check(vk.c.vkResetCommandBuffer(cmd, 0));
            try vk.check(vk.c.vkResetFences(ct.device, 1, &fnc));
            var bi = std.mem.zeroes(vk.c.VkCommandBufferBeginInfo);
            bi.sType = vk.c.VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
            bi.flags = vk.c.VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
            try vk.check(vk.c.vkBeginCommandBuffer(cmd, &bi));

            var rec = try gpu_recorder.Recorder.attachCmd(ct, cmd, 64, 256);
            defer rec.deinit();
            try rec.begin();

            const result = try rn.tickFrameTrain(&rec, budget, n_tr);
            try rn.tickPredictBatchRecord(&rec, n_pr);

            try vk.check(vk.c.vkEndCommandBuffer(cmd));
            var submit = std.mem.zeroes(vk.c.VkSubmitInfo);
            submit.sType = vk.c.VK_STRUCTURE_TYPE_SUBMIT_INFO;
            submit.commandBufferCount = 1;
            submit.pCommandBuffers = &cmd;
            try vk.check(vk.c.vkQueueSubmit(ct.queue, 1, &submit, fnc));
            const timeout_ns: u64 = 10 * 1_000_000_000;
            try vk.check(vk.c.vkWaitForFences(ct.device, 1, &fnc, vk.c.VK_TRUE, timeout_ns));
            return result;
        }
    }.f;

    // ── Test 1: .steps = 3 caps at exactly 3 ──
    const r3 = try runFrame(&runner, &ctx, host_cmd, host_fence, .{ .steps = 3 }, n_train, n_predict);
    if (r3.steps_completed != 3) {
        std.debug.print(".steps=3 wanted 3 steps, got {d}\n", .{r3.steps_completed});
        return error.ParityFailed;
    }

    // ── Test 2: .microseconds = 0 records 0 steps ──
    const r0 = try runFrame(&runner, &ctx, host_cmd, host_fence, .{ .microseconds = 0 }, n_train, n_predict);
    if (r0.steps_completed != 0) {
        std.debug.print(".microseconds=0 wanted 0 steps, got {d}\n", .{r0.steps_completed});
        return error.ParityFailed;
    }

    // ── Test 3: .either with huge µs cap behaves like .steps ──
    const r1 = try runFrame(
        &runner,
        &ctx,
        host_cmd,
        host_fence,
        .{ .either = .{ .steps = 1, .microseconds = 1_000_000 } },
        n_train,
        n_predict,
    );
    if (r1.steps_completed != 1) {
        std.debug.print(".either wanted 1 step, got {d}\n", .{r1.steps_completed});
        return error.ParityFailed;
    }

    // ── Test 4: convergence over many frames ──
    // Predict initial loss, train for 30 frames at .steps=2 each, then
    // predict again. Loss should drop materially.
    var pred_initial: [n_predict * 3]f32 = undefined;
    try runner.tickPredictBatch(x_predict, &pred_initial);
    var loss_initial: f32 = 0;
    for (0..n_predict) |i| {
        const py = pred_initial[i * cfg.dim_out ..][0..cfg.dim_out];
        const ty = t_train[i * cfg.dim_out ..][0..cfg.dim_out];
        loss_initial += cpu_train.mseLoss(py, ty);
    }
    var n_frames: u32 = 0;
    while (n_frames < 30) : (n_frames += 1) {
        _ = try runFrame(&runner, &ctx, host_cmd, host_fence, .{ .steps = 2 }, n_train, n_predict);
    }
    var pred_final: [n_predict * 3]f32 = undefined;
    try runner.tickPredictBatch(x_predict, &pred_final);
    var loss_final: f32 = 0;
    for (0..n_predict) |i| {
        const py = pred_final[i * cfg.dim_out ..][0..cfg.dim_out];
        const ty = t_train[i * cfg.dim_out ..][0..cfg.dim_out];
        loss_final += cpu_train.mseLoss(py, ty);
    }
    // 4× drop is plenty to demonstrate the cooperative API trains
    // the network end-to-end. Tighter convergence is the parity test
    // in chunk 8's smoke — this one just checks the wiring.
    if (!(loss_final < loss_initial * 0.25)) {
        std.debug.print("coop train didn't converge: initial={d:.6} final={d:.6}\n", .{ loss_initial, loss_final });
        return error.ParityFailed;
    }

    std.debug.print(
        "PASS TrainingRunner cooperative tickFrame (budget shapes ok; loss {d:.6} → {d:.6} over 30 host frames)\n",
        .{ loss_initial, loss_final },
    );
}

// ── TrainingRunner staging-readback parity smoke ────────────────────
//
// Verifies that predict outputs written via the host-mapped staging
// buffer (chunk 10) match the synchronous `tickPredictBatch` readback.
// Cooperative path: host records predict + readback into its own cmd
// buffer, submits, waits on its own fence, then reads the staging
// region directly. Should be bit-identical to the standalone path
// since it's the same dispatch with a vkCmdCopyBuffer appended.

pub fn runTrainingRunnerStagingSmoke(allocator: std.mem.Allocator) !void {
    var ctx = try vk.Context.init(allocator);
    defer ctx.deinit();

    const n_samples: u32 = 64;
    const cfg = train_runner.Mlp2Config{
        .dim_in = 4,
        .dim_hidden = 16,
        .dim_out = 3,
        .lr = 0.0,
        .init_seed = 0x57A661D6,
        .max_batch_size = n_samples,
    };
    var runner = try train_runner.TrainingRunner.init(allocator, &ctx, cfg);
    defer runner.deinit();

    // Random input grid.
    var rng = std.Random.DefaultPrng.init(0xCAFE57AD);
    const r = rng.random();
    const x_batch = try allocator.alloc(f32, n_samples * cfg.dim_in);
    defer allocator.free(x_batch);
    for (x_batch) |*v| v.* = r.float(f32) * 2 - 1;

    // ── Reference: synchronous tickPredictBatch ──
    const y_sync = try allocator.alloc(f32, n_samples * cfg.dim_out);
    defer allocator.free(y_sync);
    try runner.tickPredictBatch(x_batch, y_sync);

    // ── Cooperative: stage inputs, record predict + readback into a
    // host-owned cmd buffer, submit, wait on host fence, read staging.
    try runner.uploadPredictInputs(x_batch);

    var cb_ai = std.mem.zeroes(vk.c.VkCommandBufferAllocateInfo);
    cb_ai.sType = vk.c.VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    cb_ai.commandPool = ctx.cmd_pool;
    cb_ai.level = vk.c.VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cb_ai.commandBufferCount = 1;
    var host_cmd: vk.c.VkCommandBuffer = null;
    try vk.check(vk.c.vkAllocateCommandBuffers(ctx.device, &cb_ai, &host_cmd));
    defer vk.c.vkFreeCommandBuffers(ctx.device, ctx.cmd_pool, 1, &host_cmd);
    var fci = std.mem.zeroes(vk.c.VkFenceCreateInfo);
    fci.sType = vk.c.VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    var host_fence: vk.c.VkFence = null;
    try vk.check(vk.c.vkCreateFence(ctx.device, &fci, null, &host_fence));
    defer vk.c.vkDestroyFence(ctx.device, host_fence, null);

    var bi = std.mem.zeroes(vk.c.VkCommandBufferBeginInfo);
    bi.sType = vk.c.VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    bi.flags = vk.c.VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    try vk.check(vk.c.vkBeginCommandBuffer(host_cmd, &bi));

    var rec = try gpu_recorder.Recorder.attachCmd(&ctx, host_cmd, 8, 32);
    defer rec.deinit();
    try rec.begin();
    try runner.tickPredictBatchRecord(&rec, n_samples);
    try runner.recordPredictReadback(&rec, n_samples);

    try vk.check(vk.c.vkEndCommandBuffer(host_cmd));
    var submit = std.mem.zeroes(vk.c.VkSubmitInfo);
    submit.sType = vk.c.VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submit.commandBufferCount = 1;
    submit.pCommandBuffers = &host_cmd;
    try vk.check(vk.c.vkQueueSubmit(ctx.queue, 1, &submit, host_fence));
    const timeout_ns: u64 = 10 * 1_000_000_000;
    try vk.check(vk.c.vkWaitForFences(ctx.device, 1, &host_fence, vk.c.VK_TRUE, timeout_ns));

    const y_staging = try allocator.alloc(f32, n_samples * cfg.dim_out);
    defer allocator.free(y_staging);
    try runner.readPredictStaging(y_staging);

    var max_abs: f32 = 0;
    for (y_sync, y_staging, 0..) |a, b, i| {
        const d = @abs(a - b);
        if (d > 1e-6) {
            std.debug.print(
                "staging readback MISMATCH at {d}: sync={d:.7} staging={d:.7}\n",
                .{ i, a, b },
            );
            return error.ParityFailed;
        }
        if (d > max_abs) max_abs = d;
    }
    std.debug.print(
        "PASS TrainingRunner staging readback ({d} samples; vs synchronous tickPredictBatch max |Δ| = {e})\n",
        .{ n_samples, max_abs },
    );
}

// ── TrainingRunner Adam-optimizer parity smoke ──────────────────────
//
// CPU oracle does the canonical Adam update step-by-step:
//
//   m ← β₁·m + (1 − β₁)·g
//   v ← β₂·v + (1 − β₂)·g²
//   m̂ = m / (1 − β₁ᵗ)
//   v̂ = v / (1 − β₂ᵗ)
//   param ← param − lr · m̂ / (√v̂ + ε)
//
// Same averaged-batch gradients as the SGD smoke (chunk 8). Verifies
// GPU adam_step.comp matches the standard formulation within fp32
// tolerance over K consecutive steps. Tolerance is looser than SGD
// because the bias-correction terms involve `pow(beta, t)` which
// rounds slightly differently CPU vs GPU.

const adamCpuOracle = struct {
    fn step(
        param: []f32,
        grad: []const f32,
        m: []f32,
        v: []f32,
        lr: f32,
        beta1: f32,
        beta2: f32,
        eps: f32,
        t: u32,
    ) void {
        const tf: f32 = @floatFromInt(t);
        const bc1 = 1.0 - std.math.pow(f32, beta1, tf);
        const bc2 = 1.0 - std.math.pow(f32, beta2, tf);
        for (param, grad, m, v) |*p, g, *mi, *vi| {
            mi.* = beta1 * mi.* + (1.0 - beta1) * g;
            vi.* = beta2 * vi.* + (1.0 - beta2) * g * g;
            const m_hat = mi.* / bc1;
            const v_hat = vi.* / bc2;
            p.* -= lr * m_hat / (@sqrt(v_hat) + eps);
        }
    }
}.step;

pub fn runTrainingRunnerAdamSmoke(allocator: std.mem.Allocator) !void {
    var ctx = try vk.Context.init(allocator);
    defer ctx.deinit();

    const n_samples: u32 = 16;
    const cfg = train_runner.Mlp2Config{
        .dim_in = 4,
        .dim_hidden = 8,
        .dim_out = 3,
        .lr = 0.01,
        .init_seed = 0xADA70A50,
        .max_batch_size = n_samples,
        .optimizer = .adam,
    };
    var runner = try train_runner.TrainingRunner.init(allocator, &ctx, cfg);
    defer runner.deinit();

    var cpu_mlp = try cpu_train.Mlp.init(allocator, cfg.dim_in, cfg.dim_hidden, cfg.dim_out, cfg.init_scale, cfg.init_seed);
    defer cpu_mlp.deinit(allocator);

    // Adam state on CPU.
    const m_w1 = try allocator.alloc(f32, cpu_mlp.w1.len);
    defer allocator.free(m_w1);
    @memset(m_w1, 0);
    const v_w1 = try allocator.alloc(f32, cpu_mlp.w1.len);
    defer allocator.free(v_w1);
    @memset(v_w1, 0);
    const m_b1 = try allocator.alloc(f32, cpu_mlp.b1.len);
    defer allocator.free(m_b1);
    @memset(m_b1, 0);
    const v_b1 = try allocator.alloc(f32, cpu_mlp.b1.len);
    defer allocator.free(v_b1);
    @memset(v_b1, 0);
    const m_w2 = try allocator.alloc(f32, cpu_mlp.w2.len);
    defer allocator.free(m_w2);
    @memset(m_w2, 0);
    const v_w2 = try allocator.alloc(f32, cpu_mlp.w2.len);
    defer allocator.free(v_w2);
    @memset(v_w2, 0);
    const m_b2 = try allocator.alloc(f32, cpu_mlp.b2.len);
    defer allocator.free(m_b2);
    @memset(m_b2, 0);
    const v_b2 = try allocator.alloc(f32, cpu_mlp.b2.len);
    defer allocator.free(v_b2);
    @memset(v_b2, 0);

    var rng = std.Random.DefaultPrng.init(0x5EED5);
    const r = rng.random();
    const x_batch = try allocator.alloc(f32, n_samples * cfg.dim_in);
    defer allocator.free(x_batch);
    const target_batch = try allocator.alloc(f32, n_samples * cfg.dim_out);
    defer allocator.free(target_batch);
    for (x_batch) |*vv| vv.* = r.float(f32) * 2 - 1;
    for (target_batch) |*vv| vv.* = r.float(f32);

    const grads = try cpu_train.Grads.init(allocator, &cpu_mlp);
    defer @constCast(&grads).deinit(allocator);
    const acc_dw1 = try allocator.alloc(f32, cpu_mlp.w1.len);
    defer allocator.free(acc_dw1);
    const acc_db1 = try allocator.alloc(f32, cpu_mlp.b1.len);
    defer allocator.free(acc_db1);
    const acc_dw2 = try allocator.alloc(f32, cpu_mlp.w2.len);
    defer allocator.free(acc_dw2);
    const acc_db2 = try allocator.alloc(f32, cpu_mlp.b2.len);
    defer allocator.free(acc_db2);
    const cpu_h_pre = try allocator.alloc(f32, cfg.dim_hidden);
    defer allocator.free(cpu_h_pre);
    const cpu_h = try allocator.alloc(f32, cfg.dim_hidden);
    defer allocator.free(cpu_h);
    const cpu_y = try allocator.alloc(f32, cfg.dim_out);
    defer allocator.free(cpu_y);
    const cpu_dy = try allocator.alloc(f32, cfg.dim_out);
    defer allocator.free(cpu_dy);

    const K: u32 = 5;
    var step: u32 = 0;
    while (step < K) : (step += 1) {
        @memset(acc_dw1, 0);
        @memset(acc_db1, 0);
        @memset(acc_dw2, 0);
        @memset(acc_db2, 0);
        for (0..n_samples) |i| {
            const x_row = x_batch[i * cfg.dim_in ..][0..cfg.dim_in];
            const t_row = target_batch[i * cfg.dim_out ..][0..cfg.dim_out];
            var act: cpu_train.Activations = .{ .x = x_row, .h_pre = cpu_h_pre, .h = cpu_h, .y = cpu_y };
            cpu_train.forward(&cpu_mlp, &act);
            cpu_train.mseLossGrad(cpu_dy, cpu_y, t_row);
            var sample_grads: cpu_train.Grads = .{
                .dw1 = grads.dw1, .db1 = grads.db1, .dw2 = grads.dw2, .db2 = grads.db2,
            };
            try cpu_train.backward(allocator, &cpu_mlp, &act, cpu_dy, &sample_grads);
            for (acc_dw1, sample_grads.dw1) |*a, gi| a.* += gi;
            for (acc_db1, sample_grads.db1) |*a, gi| a.* += gi;
            for (acc_dw2, sample_grads.dw2) |*a, gi| a.* += gi;
            for (acc_db2, sample_grads.db2) |*a, gi| a.* += gi;
        }
        const inv_n: f32 = 1.0 / @as(f32, @floatFromInt(n_samples));
        for (acc_dw1) |*vv| vv.* *= inv_n;
        for (acc_db1) |*vv| vv.* *= inv_n;
        for (acc_dw2) |*vv| vv.* *= inv_n;
        for (acc_db2) |*vv| vv.* *= inv_n;

        const t_idx = step + 1;
        adamCpuOracle(cpu_mlp.w1, acc_dw1, m_w1, v_w1, cfg.lr, cfg.adam_beta1, cfg.adam_beta2, cfg.adam_eps, t_idx);
        adamCpuOracle(cpu_mlp.b1, acc_db1, m_b1, v_b1, cfg.lr, cfg.adam_beta1, cfg.adam_beta2, cfg.adam_eps, t_idx);
        adamCpuOracle(cpu_mlp.w2, acc_dw2, m_w2, v_w2, cfg.lr, cfg.adam_beta1, cfg.adam_beta2, cfg.adam_eps, t_idx);
        adamCpuOracle(cpu_mlp.b2, acc_db2, m_b2, v_b2, cfg.lr, cfg.adam_beta1, cfg.adam_beta2, cfg.adam_eps, t_idx);

        try runner.tickStepBatch(x_batch, target_batch);
    }

    const gpu_w1 = try allocator.alloc(f32, cpu_mlp.w1.len);
    defer allocator.free(gpu_w1);
    const gpu_b1 = try allocator.alloc(f32, cpu_mlp.b1.len);
    defer allocator.free(gpu_b1);
    const gpu_w2 = try allocator.alloc(f32, cpu_mlp.w2.len);
    defer allocator.free(gpu_w2);
    const gpu_b2 = try allocator.alloc(f32, cpu_mlp.b2.len);
    defer allocator.free(gpu_b2);
    var gpu_mlp: cpu_train.Mlp = .{
        .dim_in = cfg.dim_in,
        .dim_hidden = cfg.dim_hidden,
        .dim_out = cfg.dim_out,
        .w1 = gpu_w1,
        .b1 = gpu_b1,
        .w2 = gpu_w2,
        .b2 = gpu_b2,
    };
    try runner.readWeights(&gpu_mlp);

    const tol: f32 = 1e-4;
    var max_abs: f32 = 0;
    const ParamCase = struct { name: []const u8, gpu: []const f32, cpu: []const f32 };
    const cases = [_]ParamCase{
        .{ .name = "W1", .gpu = gpu_w1, .cpu = cpu_mlp.w1 },
        .{ .name = "b1", .gpu = gpu_b1, .cpu = cpu_mlp.b1 },
        .{ .name = "W2", .gpu = gpu_w2, .cpu = cpu_mlp.w2 },
        .{ .name = "b2", .gpu = gpu_b2, .cpu = cpu_mlp.b2 },
    };
    for (cases) |cs| {
        for (cs.gpu, cs.cpu, 0..) |g, cv, i| {
            const d = @abs(g - cv);
            if (d > tol) {
                std.debug.print("Adam MISMATCH on {s}[{d}] @ step {d}: gpu={d:.7} cpu={d:.7}\n", .{ cs.name, i, K, g, cv });
                return error.ParityFailed;
            }
            if (d > max_abs) max_abs = d;
        }
    }
    std.debug.print(
        "PASS TrainingRunner Adam ({d} samples × {d} steps, dim 4→8→3; max |Δ| vs CPU oracle = {e})\n",
        .{ n_samples, K, max_abs },
    );
}

// ── TrainingRunner cross-entropy loss-grad parity smoke ─────────────
//
// CPU oracle does the canonical stable softmax + CE-grad:
//   m = max(y)
//   p = exp(y - m) / Σ exp(y - m)
//   dy = (p − target) / N
// then compares against the GPU shader's output. Catches any sign or
// off-by-one in the softmax + scale-by-1/N chain. Two scenarios:
// hard-label (one-hot target) and soft-label (mixed distribution), so
// the test exercises both classifier and distillation use cases.

pub fn runTrainingRunnerCrossEntropySmoke(allocator: std.mem.Allocator) !void {
    var ctx = try vk.Context.init(allocator);
    defer ctx.deinit();

    const n_samples: u32 = 32;
    const dim_out: u32 = 5;
    const cfg = train_runner.Mlp2Config{
        .dim_in = 4,
        .dim_hidden = 8,
        .dim_out = dim_out,
        .lr = 0.0,
        .init_seed = 0xC10557E,
        .max_batch_size = n_samples,
        .loss = .cross_entropy,
    };
    var runner = try train_runner.TrainingRunner.init(allocator, &ctx, cfg);
    defer runner.deinit();

    // Build synthetic batch + target.
    var rng = std.Random.DefaultPrng.init(0xCEED57A);
    const r = rng.random();
    const x_batch = try allocator.alloc(f32, n_samples * cfg.dim_in);
    defer allocator.free(x_batch);
    const target_batch = try allocator.alloc(f32, n_samples * cfg.dim_out);
    defer allocator.free(target_batch);
    for (x_batch) |*v| v.* = r.float(f32) * 2 - 1;
    // Half samples one-hot (hard label), half soft (Dirichlet-ish).
    var i: u32 = 0;
    while (i < n_samples) : (i += 1) {
        const off = i * dim_out;
        if (i < n_samples / 2) {
            const cls = r.uintLessThan(u32, dim_out);
            for (0..dim_out) |o| target_batch[off + o] = if (o == cls) 1.0 else 0.0;
        } else {
            var sum: f32 = 0;
            for (0..dim_out) |o| {
                const v = r.float(f32);
                target_batch[off + o] = v;
                sum += v;
            }
            for (0..dim_out) |o| target_batch[off + o] /= sum;
        }
    }

    // Run forward via tickPredictBatch to get logits, then run our
    // CE-grad shader by triggering one batched train step (dy is the
    // first thing recordTrainBatch dispatches).
    //
    // Approach: read y_train_batch directly after a single batched
    // train step. But that requires staging or a peek. Simplest: use
    // a minimal run that lets us readBack y_b. Our recordTrainBatch
    // already writes h_pre/h/y to dedicated buffers; we can readBack
    // y_train_batch after the dispatch.
    try runner.uploadTrainBatch(x_batch, target_batch);

    // Manually drive forward + dy + readback into a recorder. This
    // lets us isolate the dy output for parity-checking without the
    // rest of the train chain mutating weights (lr=0 already takes
    // care of that, but it's cleaner this way).
    try runner.tickStepBatch(x_batch, target_batch);

    // Pull the y_train_batch logits + dy_train_batch back to host.
    const y_gpu = try allocator.alloc(f32, n_samples * dim_out);
    defer allocator.free(y_gpu);
    const dy_gpu = try allocator.alloc(f32, n_samples * dim_out);
    defer allocator.free(dy_gpu);
    try runner.y_train_batch.?.readBack(&ctx, f32, y_gpu);
    try runner.dy_train_batch.?.readBack(&ctx, f32, dy_gpu);

    // CPU oracle: stable softmax + CE-grad, /N pre-divide.
    const dy_cpu = try allocator.alloc(f32, n_samples * dim_out);
    defer allocator.free(dy_cpu);
    const inv_n: f32 = 1.0 / @as(f32, @floatFromInt(n_samples));
    for (0..n_samples) |n| {
        const off = n * dim_out;
        var m: f32 = y_gpu[off];
        for (1..dim_out) |o| m = @max(m, y_gpu[off + o]);
        var sum_e: f32 = 0;
        for (0..dim_out) |o| sum_e += @exp(y_gpu[off + o] - m);
        for (0..dim_out) |o| {
            const p = @exp(y_gpu[off + o] - m) / sum_e;
            dy_cpu[off + o] = (p - target_batch[off + o]) * inv_n;
        }
    }

    var max_abs: f32 = 0;
    for (dy_gpu, dy_cpu, 0..) |g, c, idx| {
        const d = @abs(g - c);
        if (d > 1e-5) {
            std.debug.print(
                "CE loss-grad MISMATCH at {d}: gpu={d:.7} cpu={d:.7}\n",
                .{ idx, g, c },
            );
            return error.ParityFailed;
        }
        if (d > max_abs) max_abs = d;
    }
    std.debug.print(
        "PASS TrainingRunner cross-entropy loss-grad ({d} samples × {d} classes; max |Δ| vs CPU oracle = {e})\n",
        .{ n_samples, dim_out, max_abs },
    );
}

// ── TrainingRunner loss-target decay smoke ──────────────────────────
//
// Verifies the runtime self-throttling logic:
//   1. With decay = .loss_target { target = 0.05 }, training runs
//      until loss < target, then `tickFrameTrain` reports idle = true
//      and steps_completed = 0.
//   2. last_loss caches a real value (> 0, ≤ target by the time idle).
//   3. Resumption: changing target_batch to break the fit causes
//      next frame's tickFrameTrain to report idle = false again.

pub fn runTrainingRunnerDecaySmoke(allocator: std.mem.Allocator) !void {
    var ctx = try vk.Context.init(allocator);
    defer ctx.deinit();

    const n_train: u32 = 8;
    const cfg = train_runner.Mlp2Config{
        .dim_in = 4,
        .dim_hidden = 16,
        .dim_out = 3,
        .lr = 0.05,
        .init_seed = 0xDECA1A77,
        .max_batch_size = n_train,
        .optimizer = .adam,
        .decay = .{ .loss_target = .{ .target = 0.001 } },
    };
    var runner = try train_runner.TrainingRunner.init(allocator, &ctx, cfg);
    defer runner.deinit();

    // Non-trivial supervision: targets ∈ [-2, 2], well outside the
    // initial MLP's near-zero output range. Initial loss is large so
    // we actually exercise the converge-then-idle path; a target of
    // 0.001 is reachable in a few dozen Adam steps but not on frame 1.
    var rng = std.Random.DefaultPrng.init(0xDECAD0);
    const r = rng.random();
    const x_train = try allocator.alloc(f32, n_train * cfg.dim_in);
    defer allocator.free(x_train);
    const t_train = try allocator.alloc(f32, n_train * cfg.dim_out);
    defer allocator.free(t_train);
    for (x_train) |*v| v.* = r.float(f32) * 2 - 1;
    for (t_train) |*v| v.* = r.float(f32) * 4 - 2;

    try runner.uploadTrainBatch(x_train, t_train);

    var cb_ai = std.mem.zeroes(vk.c.VkCommandBufferAllocateInfo);
    cb_ai.sType = vk.c.VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    cb_ai.commandPool = ctx.cmd_pool;
    cb_ai.level = vk.c.VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cb_ai.commandBufferCount = 1;
    var host_cmd: vk.c.VkCommandBuffer = null;
    try vk.check(vk.c.vkAllocateCommandBuffers(ctx.device, &cb_ai, &host_cmd));
    defer vk.c.vkFreeCommandBuffers(ctx.device, ctx.cmd_pool, 1, &host_cmd);
    var fci = std.mem.zeroes(vk.c.VkFenceCreateInfo);
    fci.sType = vk.c.VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    var host_fence: vk.c.VkFence = null;
    try vk.check(vk.c.vkCreateFence(ctx.device, &fci, null, &host_fence));
    defer vk.c.vkDestroyFence(ctx.device, host_fence, null);

    const runFrame = struct {
        fn f(
            rn: *train_runner.TrainingRunner,
            ct: *const vk.Context,
            cmd: vk.c.VkCommandBuffer,
            fnc: vk.c.VkFence,
            n: u32,
        ) !train_runner.TrainTickResult {
            try vk.check(vk.c.vkResetCommandBuffer(cmd, 0));
            try vk.check(vk.c.vkResetFences(ct.device, 1, &fnc));
            var bi = std.mem.zeroes(vk.c.VkCommandBufferBeginInfo);
            bi.sType = vk.c.VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
            bi.flags = vk.c.VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
            try vk.check(vk.c.vkBeginCommandBuffer(cmd, &bi));
            var rec = try gpu_recorder.Recorder.attachCmd(ct, cmd, 64, 256);
            defer rec.deinit();
            try rec.begin();
            const result = try rn.tickFrameTrain(&rec, .{ .steps = 4 }, n);
            try vk.check(vk.c.vkEndCommandBuffer(cmd));
            var submit = std.mem.zeroes(vk.c.VkSubmitInfo);
            submit.sType = vk.c.VK_STRUCTURE_TYPE_SUBMIT_INFO;
            submit.commandBufferCount = 1;
            submit.pCommandBuffers = &cmd;
            try vk.check(vk.c.vkQueueSubmit(ct.queue, 1, &submit, fnc));
            const timeout_ns: u64 = 10 * 1_000_000_000;
            try vk.check(vk.c.vkWaitForFences(ct.device, 1, &fnc, vk.c.VK_TRUE, timeout_ns));
            return result;
        }
    }.f;

    // ── Phase 1: train until idle ──
    var idle_at: ?u32 = null;
    var loss_at_idle: ?f32 = null;
    var f: u32 = 0;
    while (f < 200) : (f += 1) {
        const result = try runFrame(&runner, &ctx, host_cmd, host_fence, n_train);
        if (result.idle) {
            idle_at = f;
            loss_at_idle = result.last_loss;
            break;
        }
    }
    if (idle_at == null) {
        std.debug.print("decay never reached idle within 200 frames; last_loss={?}\n", .{runner.getLastLoss()});
        return error.ParityFailed;
    }
    if (loss_at_idle == null or loss_at_idle.? > 0.001) {
        std.debug.print("idle reported but last_loss = {?} > target=0.001\n", .{loss_at_idle});
        return error.ParityFailed;
    }
    // Convergence (not lucky-init) check: idle should land after at
    // least a handful of training frames.
    if (idle_at.? < 3) {
        std.debug.print("idle reached on frame {d} — too early; init likely already met target. Bump targets.\n", .{idle_at.?});
        return error.ParityFailed;
    }

    // Confirm subsequent frames stay idle (training really paused).
    var f2: u32 = 0;
    while (f2 < 5) : (f2 += 1) {
        const result = try runFrame(&runner, &ctx, host_cmd, host_fence, n_train);
        if (!result.idle or result.steps_completed != 0) {
            std.debug.print("decay didn't stay idle at frame +{d}: idle={any} steps={d}\n", .{ f2, result.idle, result.steps_completed });
            return error.ParityFailed;
        }
    }

    // ── Phase 2: break the fit, training should resume ──
    // Stomp the targets with very different values that the network
    // hasn't seen — loss should jump above target and tickFrameTrain
    // should return idle = false.
    for (t_train) |*v| v.* = 5.0; // wildly different from the trained targets
    try runner.uploadTrainBatch(x_train, t_train);

    // First post-stomp frame: staging still has old loss (low), so
    // runtime gates training off and records loss eval. Need TWO
    // frames for the resumption signal to propagate (eval frame N
    // measures fresh loss; frame N+1 reads it and trains).
    _ = try runFrame(&runner, &ctx, host_cmd, host_fence, n_train);
    const resume_result = try runFrame(&runner, &ctx, host_cmd, host_fence, n_train);
    if (resume_result.idle or resume_result.steps_completed == 0) {
        std.debug.print(
            "decay didn't resume after target stomp: idle={any} steps={d} last_loss={?}\n",
            .{ resume_result.idle, resume_result.steps_completed, resume_result.last_loss },
        );
        return error.ParityFailed;
    }

    std.debug.print(
        "PASS TrainingRunner loss-target decay (idle at frame {?d}, loss={d:.6}; resumed cleanly after target stomp)\n",
        .{ idle_at, loss_at_idle.? },
    );
}

// ── --train-demo: headless on-device training demo ──────────────────
//
// Chunk 6 of training-v0. End-user-facing CLI showing the
// TrainingRunner converge on a non-trivial streaming signal, with no
// model file required. Inputs and targets are derived from a virtual
// "frame number" t — the same shape the engine integration will run,
// just printed to stdout instead of driving a sphere's BRDF. Lets a
// fresh checkout verify the whole training pipeline works end-to-end
// without any external assets, and gives a feel for "what a few
// hundred frames of training does to the loss" before the visual
// demo lands.

pub fn runTrainDemo(allocator: std.mem.Allocator, steps: u32, hidden: u32, lr: f32, print_every: u32) !void {
    var ctx = try vk.Context.init(allocator);
    defer ctx.deinit();

    const cfg = train_runner.Mlp2Config{
        .dim_in = 4,
        .dim_hidden = hidden,
        .dim_out = 4,
        .lr = lr,
        .init_seed = 0xDEDEDED0,
    };

    var runner = try train_runner.TrainingRunner.init(allocator, &ctx, cfg);
    defer runner.deinit();

    std.debug.print(
        "valkyr --train-demo on {s}\n  cfg: dim 4→{d}→4, lr={d}, steps={d}\n",
        .{ ctx.deviceName(), hidden, lr, steps },
    );
    std.debug.print("  task: regress (sin t, cos t, sin 2t, cos 2t) → (½+½ sin t, ½+½ cos t, ½ + ⅓ sin 3t, 0)\n", .{});

    // Input/target factories. Time advances by `dt` per step; the
    // signal is intentionally smooth (so SGD on a single-sample stream
    // sees a moving but locally-flat target) but non-trivial (the
    // network must combine all 4 inputs to predict the 3rd component).
    const dt: f32 = 0.05;
    var pred: [4]f32 = undefined;
    var target: [4]f32 = undefined;
    var input: [4]f32 = undefined;

    var step: u32 = 0;
    var ema_loss: f32 = 0;
    const ema_alpha: f32 = 0.05;
    const t_start = std.time.nanoTimestamp();

    while (step < steps) : (step += 1) {
        const t = @as(f32, @floatFromInt(step)) * dt;
        input[0] = @sin(t);
        input[1] = @cos(t);
        input[2] = @sin(2 * t);
        input[3] = @cos(2 * t);
        target[0] = 0.5 + 0.5 * @sin(t);
        target[1] = 0.5 + 0.5 * @cos(t);
        target[2] = 0.5 + 0.333 * @sin(3 * t);
        target[3] = 0.0;

        try runner.tickStep(&input, &target, &pred);
        const loss = cpu_train.mseLoss(&pred, &target);
        ema_loss = if (step == 0) loss else ema_alpha * loss + (1 - ema_alpha) * ema_loss;

        if ((step + 1) % print_every == 0 or step == 0) {
            std.debug.print(
                "  step {d:>5}  loss={d:.6}  ema={d:.6}  pred=({d:.3},{d:.3},{d:.3},{d:.3})\n",
                .{ step + 1, loss, ema_loss, pred[0], pred[1], pred[2], pred[3] },
            );
        }
    }
    const t_end = std.time.nanoTimestamp();
    const elapsed_us = @divTrunc(t_end - t_start, 1000);
    const us_per_step = @divTrunc(elapsed_us, @as(i128, steps));
    std.debug.print(
        "done: {d} steps in {d} µs ({d} µs/step, {d:.0} steps/s)\n",
        .{ steps, elapsed_us, us_per_step, 1.0e6 / @as(f64, @floatFromInt(us_per_step)) },
    );
}

// ── --train-demo-n: headless multi-layer training demo ──────────────
//
// Mirrors `--train-demo` but routed through MlpNRunner with a
// `--layers a,b,c,d` CSV controlling depth. End-user-facing reach
// for the multi-layer surface introduced in chunk 3a of the post-v0
// training arc. Same streaming (input, target) signal and same
// per-step loss / EMA print pattern as the 2-layer version, so the
// loss curves are visually comparable across depths.

pub fn runTrainDemoN(
    allocator: std.mem.Allocator,
    steps: u32,
    layers_csv: []const u8,
    lr: f32,
    print_every: u32,
) !void {
    var ctx = try vk.Context.init(allocator);
    defer ctx.deinit();

    // Parse `layers_csv` into a u32 slice. Errors on empty / invalid
    // entries; first entry must be 4 and last must be 4 to match the
    // input/target signal shape below.
    var dims = std.ArrayList(u32).init(allocator);
    defer dims.deinit();
    var it = std.mem.splitScalar(u8, layers_csv, ',');
    while (it.next()) |tok| {
        const trimmed = std.mem.trim(u8, tok, " \t");
        if (trimmed.len == 0) continue;
        try dims.append(try std.fmt.parseInt(u32, trimmed, 10));
    }
    if (dims.items.len < 2) {
        std.debug.print("--train-demo-n: --layers needs at least 2 dims, got '{s}'\n", .{layers_csv});
        return error.InvalidArgs;
    }
    if (dims.items[0] != 4 or dims.items[dims.items.len - 1] != 4) {
        std.debug.print("--train-demo-n: signal is fixed at dim_in=4 and dim_out=4; got [{d}, ..., {d}]\n", .{ dims.items[0], dims.items[dims.items.len - 1] });
        return error.InvalidArgs;
    }

    const cfg = train_runner_n.MlpNConfig{
        .layer_dims = dims.items,
        .lr = lr,
        .init_seed = 0xDEDEDED1,
    };
    var runner = try train_runner_n.MlpNRunner.init(allocator, &ctx, cfg);
    defer runner.deinit();

    std.debug.print("valkyr --train-demo-n on {s}\n  cfg: layers", .{ctx.deviceName()});
    for (dims.items, 0..) |d, idx| {
        std.debug.print("{s}{d}", .{ if (idx == 0) " " else "→", d });
    }
    std.debug.print(", lr={d}, steps={d}\n", .{ lr, steps });
    std.debug.print("  task: regress (sin t, cos t, sin 2t, cos 2t) → (½+½ sin t, ½+½ cos t, ½ + ⅓ sin 3t, 0)\n", .{});

    const dt: f32 = 0.05;
    var pred: [4]f32 = undefined;
    var target: [4]f32 = undefined;
    var input: [4]f32 = undefined;

    var step: u32 = 0;
    var ema_loss: f32 = 0;
    const ema_alpha: f32 = 0.05;
    const t_start = std.time.nanoTimestamp();

    while (step < steps) : (step += 1) {
        const t = @as(f32, @floatFromInt(step)) * dt;
        input[0] = @sin(t);
        input[1] = @cos(t);
        input[2] = @sin(2 * t);
        input[3] = @cos(2 * t);
        target[0] = 0.5 + 0.5 * @sin(t);
        target[1] = 0.5 + 0.5 * @cos(t);
        target[2] = 0.5 + 0.333 * @sin(3 * t);
        target[3] = 0.0;

        try runner.tickStep(&input, &target, &pred);
        const loss = cpu_train.mseLoss(&pred, &target);
        ema_loss = if (step == 0) loss else ema_alpha * loss + (1 - ema_alpha) * ema_loss;

        if ((step + 1) % print_every == 0 or step == 0) {
            std.debug.print(
                "  step {d:>5}  loss={d:.6}  ema={d:.6}  pred=({d:.3},{d:.3},{d:.3},{d:.3})\n",
                .{ step + 1, loss, ema_loss, pred[0], pred[1], pred[2], pred[3] },
            );
        }
    }
    const t_end = std.time.nanoTimestamp();
    const elapsed_us = @divTrunc(t_end - t_start, 1000);
    const us_per_step = @divTrunc(elapsed_us, @as(i128, steps));
    std.debug.print(
        "done: {d} steps in {d} µs ({d} µs/step, {d:.0} steps/s)\n",
        .{ steps, elapsed_us, us_per_step, 1.0e6 / @as(f64, @floatFromInt(us_per_step)) },
    );
}
