//! TrainingRunner — per-frame on-device training loop.
//!
//! Wraps the chunk-2/3/4 forward + loss-grad + backward + SGD pipeline
//! into a single object with persistent parameter and activation
//! buffers and a small `tickStep` API. One submit per step. Pipelines
//! and weight buffers live for the lifetime of the Runner; `x` and
//! `target` are HOST_VISIBLE dynamic buffers so the host can stream
//! a fresh (input, target) per frame without staging through a
//! transient buffer.
//!
//! Symmetric in spirit to `inference/runner.zig` but scoped tighter:
//! single-task training, no batching, no SPSC queue, no thread mode.
//! Real-time training inside a host frame budget is the use case.
//! Multi-task / batched / async variants land later if needed.
//!
//! Architecture is a fixed 2-layer MLP — Linear → ReLU → Linear — the
//! same shape the CPU oracle in `cpu/train.zig` validates. Generalising
//! to deeper or different topologies is the next-tier project (the
//! "Unsloth" full-transformer training port in the roadmap).

const std = @import("std");
const vk = @import("../gpu/vk.zig");
const buffer = @import("../gpu/buffer.zig");
const pipeline = @import("../gpu/pipeline.zig");
const recorder_mod = @import("../gpu/recorder.zig");
const runtime = @import("../runtime.zig");
const cpu_train = @import("../cpu/train.zig");
const shaders = @import("shaders");

pub const Mlp2Config = struct {
    dim_in: u32,
    dim_hidden: u32,
    dim_out: u32,
    /// SGD learning rate. Mutable across frames — write directly to
    /// `runner.lr` between `tickStep` calls if you want a schedule.
    lr: f32,
    /// Initial weight scale: weights drawn from U(-init_scale, +init_scale).
    /// 0.3 lands the click-to-red toy task in the well of convergence
    /// without hiding bugs behind clever init; bump up for harder tasks.
    init_scale: f32 = 0.3,
    /// Seed for the initial weight RNG. Same seed → same starting MLP,
    /// matching the CPU oracle's `Mlp.init`.
    init_seed: u64 = 0xCAFE,
    /// Max sample count for `tickPredictBatch`. Sized 0 disables the
    /// batched-predict surface entirely (and skips allocating its
    /// host-mapped X / device-only Y buffers). Visualisation use
    /// cases set this to N×N for a UV-grid overlay.
    max_batch_size: u32 = 0,
};

/// Compile-time cap on `dim_hidden` enforced by the batched forward
/// shader (it stores the hidden vector in a per-thread fp32 array of
/// this size). Mirrors the GLSL `MAX_HIDDEN` constant.
pub const MLP2_MAX_HIDDEN: u32 = 64;

pub const TrainingRunner = struct {
    ctx: *const vk.Context,
    cfg: Mlp2Config,
    lr: f32,

    // ── Pipelines (built once, reused across every step) ──
    k_matmul: pipeline.Kernel,
    k_add: pipeline.Kernel,
    k_relu: pipeline.Kernel,
    k_mse_grad: pipeline.Kernel,
    k_relu_bw: pipeline.Kernel,
    k_lin_dx: pipeline.Kernel,
    k_outer: pipeline.Kernel,
    k_copy: pipeline.Kernel,
    k_sgd: pipeline.Kernel,
    /// Batched forward (only built when cfg.max_batch_size > 0).
    k_predict_batch: ?pipeline.Kernel,

    // ── Parameter buffers (DEVICE_LOCAL, mutated by SGD) ──
    w1: buffer.Buffer,
    b1: buffer.Buffer,
    w2: buffer.Buffer,
    b2: buffer.Buffer,

    // ── Per-step input/target (HOST_VISIBLE, host re-fills each frame) ──
    x_buf: buffer.Buffer,
    target_buf: buffer.Buffer,

    // ── Activation + gradient scratch (DEVICE_LOCAL, reused) ──
    h_pre: buffer.Buffer,
    h: buffer.Buffer,
    y: buffer.Buffer,
    dL_dy: buffer.Buffer,
    dh: buffer.Buffer,
    dh_pre: buffer.Buffer,
    dw1: buffer.Buffer,
    db1: buffer.Buffer,
    dw2: buffer.Buffer,
    db2: buffer.Buffer,

    /// Batched-predict input + output (only allocated when
    /// `cfg.max_batch_size > 0`). `x_batch` is dynamic (host-mapped)
    /// so callers stream a fresh grid per frame; `y_batch` is
    /// device-only and read back to fill the visualisation tiles.
    x_batch: ?buffer.Buffer,
    y_batch: ?buffer.Buffer,

    pub fn init(allocator: std.mem.Allocator, ctx: *const vk.Context, cfg: Mlp2Config) !TrainingRunner {
        if (cfg.dim_hidden > MLP2_MAX_HIDDEN) return error.HiddenDimExceedsMax;
        // ── Initial weights via the CPU oracle ─────────────────────
        // Reusing `cpu_train.Mlp.init` guarantees the GPU runner starts
        // from a known starting point we can also recreate on the CPU
        // side — used by the smoke and by parity tests downstream.
        var seed_mlp = try cpu_train.Mlp.init(
            allocator,
            cfg.dim_in,
            cfg.dim_hidden,
            cfg.dim_out,
            cfg.init_scale,
            cfg.init_seed,
        );
        defer seed_mlp.deinit(allocator);

        // ── Pipelines ──────────────────────────────────────────────
        const k_matmul = try pipeline.Kernel.init(ctx, &shaders.matmul_nt, 3, @sizeOf(runtime.MatmulPush));
        errdefer @constCast(&k_matmul).deinit();
        const k_add = try pipeline.Kernel.init(ctx, &shaders.add_in_place, 2, @sizeOf(runtime.AddInPlacePush));
        errdefer @constCast(&k_add).deinit();
        const k_relu = try pipeline.Kernel.init(ctx, &shaders.relu, 2, @sizeOf(runtime.ReluPush));
        errdefer @constCast(&k_relu).deinit();
        const k_mse_grad = try pipeline.Kernel.init(ctx, &shaders.mse_loss_grad, 3, @sizeOf(runtime.MseLossGradPush));
        errdefer @constCast(&k_mse_grad).deinit();
        const k_relu_bw = try pipeline.Kernel.init(ctx, &shaders.relu_backward, 3, @sizeOf(runtime.ReluBackwardPush));
        errdefer @constCast(&k_relu_bw).deinit();
        const k_lin_dx = try pipeline.Kernel.init(ctx, &shaders.linear_backward_dx, 3, @sizeOf(runtime.LinearBackwardDxPush));
        errdefer @constCast(&k_lin_dx).deinit();
        const k_outer = try pipeline.Kernel.init(ctx, &shaders.outer_product, 3, @sizeOf(runtime.OuterProductPush));
        errdefer @constCast(&k_outer).deinit();
        const k_copy = try pipeline.Kernel.init(ctx, &shaders.slice_copy, 2, 12); // SliceCopyPush is 3*u32
        errdefer @constCast(&k_copy).deinit();
        const k_sgd = try pipeline.Kernel.init(ctx, &shaders.sgd_step, 2, @sizeOf(runtime.SgdStepPush));
        errdefer @constCast(&k_sgd).deinit();
        var k_predict_batch: ?pipeline.Kernel = null;
        if (cfg.max_batch_size > 0) {
            k_predict_batch = try pipeline.Kernel.init(ctx, &shaders.mlp2_forward_batched, 6, @sizeOf(runtime.Mlp2ForwardBatchedPush));
        }
        errdefer if (k_predict_batch) |*k| k.deinit();

        // ── Parameter buffers (start populated from seed_mlp) ──
        const w1 = try buffer.Buffer.initStatic(ctx, f32, seed_mlp.w1);
        errdefer @constCast(&w1).deinit(ctx.device);
        const b1 = try buffer.Buffer.initStatic(ctx, f32, seed_mlp.b1);
        errdefer @constCast(&b1).deinit(ctx.device);
        const w2 = try buffer.Buffer.initStatic(ctx, f32, seed_mlp.w2);
        errdefer @constCast(&w2).deinit(ctx.device);
        const b2 = try buffer.Buffer.initStatic(ctx, f32, seed_mlp.b2);
        errdefer @constCast(&b2).deinit(ctx.device);

        // ── Per-step input/target as dynamic (host-mapped) buffers ──
        // Streaming a new (x, target) every frame is exactly the
        // dynamic-buffer use case from gpu/buffer.zig: HOST_VISIBLE,
        // persistently mapped, written via Buffer.update with no
        // staging round-trip.
        const x_buf = try buffer.Buffer.initDynamic(ctx, cfg.dim_in * @sizeOf(f32));
        errdefer @constCast(&x_buf).deinit(ctx.device);
        const target_buf = try buffer.Buffer.initDynamic(ctx, cfg.dim_out * @sizeOf(f32));
        errdefer @constCast(&target_buf).deinit(ctx.device);

        // ── Activation + gradient scratch ──
        const h_pre = try buffer.Buffer.initDeviceOnly(ctx, cfg.dim_hidden * @sizeOf(f32));
        errdefer @constCast(&h_pre).deinit(ctx.device);
        const h = try buffer.Buffer.initDeviceOnly(ctx, cfg.dim_hidden * @sizeOf(f32));
        errdefer @constCast(&h).deinit(ctx.device);
        const y = try buffer.Buffer.initDeviceOnly(ctx, cfg.dim_out * @sizeOf(f32));
        errdefer @constCast(&y).deinit(ctx.device);
        const dL_dy = try buffer.Buffer.initDeviceOnly(ctx, cfg.dim_out * @sizeOf(f32));
        errdefer @constCast(&dL_dy).deinit(ctx.device);
        const dh = try buffer.Buffer.initDeviceOnly(ctx, cfg.dim_hidden * @sizeOf(f32));
        errdefer @constCast(&dh).deinit(ctx.device);
        const dh_pre = try buffer.Buffer.initDeviceOnly(ctx, cfg.dim_hidden * @sizeOf(f32));
        errdefer @constCast(&dh_pre).deinit(ctx.device);
        const dw1 = try buffer.Buffer.initDeviceOnly(ctx, cfg.dim_hidden * cfg.dim_in * @sizeOf(f32));
        errdefer @constCast(&dw1).deinit(ctx.device);
        const db1 = try buffer.Buffer.initDeviceOnly(ctx, cfg.dim_hidden * @sizeOf(f32));
        errdefer @constCast(&db1).deinit(ctx.device);
        const dw2 = try buffer.Buffer.initDeviceOnly(ctx, cfg.dim_out * cfg.dim_hidden * @sizeOf(f32));
        errdefer @constCast(&dw2).deinit(ctx.device);
        const db2 = try buffer.Buffer.initDeviceOnly(ctx, cfg.dim_out * @sizeOf(f32));
        errdefer @constCast(&db2).deinit(ctx.device);

        var x_batch: ?buffer.Buffer = null;
        var y_batch: ?buffer.Buffer = null;
        if (cfg.max_batch_size > 0) {
            x_batch = try buffer.Buffer.initDynamic(ctx, cfg.max_batch_size * cfg.dim_in * @sizeOf(f32));
            errdefer if (x_batch) |*xb| xb.deinit(ctx.device);
            y_batch = try buffer.Buffer.initDeviceOnly(ctx, cfg.max_batch_size * cfg.dim_out * @sizeOf(f32));
            errdefer if (y_batch) |*yb| yb.deinit(ctx.device);
        }

        return .{
            .ctx = ctx,
            .cfg = cfg,
            .lr = cfg.lr,
            .k_matmul = k_matmul,
            .k_add = k_add,
            .k_relu = k_relu,
            .k_mse_grad = k_mse_grad,
            .k_relu_bw = k_relu_bw,
            .k_lin_dx = k_lin_dx,
            .k_outer = k_outer,
            .k_copy = k_copy,
            .k_sgd = k_sgd,
            .k_predict_batch = k_predict_batch,
            .w1 = w1,
            .b1 = b1,
            .w2 = w2,
            .b2 = b2,
            .x_buf = x_buf,
            .target_buf = target_buf,
            .h_pre = h_pre,
            .h = h,
            .y = y,
            .dL_dy = dL_dy,
            .dh = dh,
            .dh_pre = dh_pre,
            .dw1 = dw1,
            .db1 = db1,
            .dw2 = dw2,
            .db2 = db2,
            .x_batch = x_batch,
            .y_batch = y_batch,
        };
    }

    pub fn deinit(self: *TrainingRunner) void {
        const dev = self.ctx.device;
        self.w1.deinit(dev);
        self.b1.deinit(dev);
        self.w2.deinit(dev);
        self.b2.deinit(dev);
        self.x_buf.deinit(dev);
        self.target_buf.deinit(dev);
        self.h_pre.deinit(dev);
        self.h.deinit(dev);
        self.y.deinit(dev);
        self.dL_dy.deinit(dev);
        self.dh.deinit(dev);
        self.dh_pre.deinit(dev);
        self.dw1.deinit(dev);
        self.db1.deinit(dev);
        self.dw2.deinit(dev);
        self.db2.deinit(dev);
        if (self.x_batch) |*xb| xb.deinit(dev);
        if (self.y_batch) |*yb| yb.deinit(dev);
        self.k_matmul.deinit();
        self.k_add.deinit();
        self.k_relu.deinit();
        self.k_mse_grad.deinit();
        self.k_relu_bw.deinit();
        self.k_lin_dx.deinit();
        self.k_outer.deinit();
        self.k_copy.deinit();
        self.k_sgd.deinit();
        if (self.k_predict_batch) |*k| k.deinit();
    }

    /// Run one full training step: upload x/target, record forward +
    /// loss-grad + backward + SGD, submit, wait. If `out_pred` is
    /// non-null it's filled with the prediction y *before* the SGD
    /// step (i.e. the prediction the loss was computed against —
    /// the natural "what does the model say right now" reading).
    pub fn tickStep(
        self: *TrainingRunner,
        x_in: []const f32,
        target_in: []const f32,
        out_pred: ?[]f32,
    ) !void {
        if (x_in.len != self.cfg.dim_in) return error.XDimMismatch;
        if (target_in.len != self.cfg.dim_out) return error.TargetDimMismatch;
        if (out_pred) |op| if (op.len != self.cfg.dim_out) return error.OutDimMismatch;

        // Stream input/target into the dynamic buffers. No staging,
        // no submit — Buffer.update is a memcpy into HOST_COHERENT
        // memory, visible to the GPU at the next dispatch.
        self.x_buf.update(f32, x_in);
        self.target_buf.update(f32, target_in);

        var rec = try recorder_mod.Recorder.init(self.ctx, 24, 96);
        defer rec.deinit();
        try rec.begin();

        try self.recordStep(&rec);

        try rec.endAndSubmit();

        if (out_pred) |op| try self.y.readBack(self.ctx, f32, op);
    }

    /// Forward-only: predict for `x_in`, fill `out_pred`. Does not
    /// modify weights. Useful when there's no target this frame
    /// (interactive / inference-style use).
    pub fn tickPredict(self: *TrainingRunner, x_in: []const f32, out_pred: []f32) !void {
        if (x_in.len != self.cfg.dim_in) return error.XDimMismatch;
        if (out_pred.len != self.cfg.dim_out) return error.OutDimMismatch;
        self.x_buf.update(f32, x_in);

        var rec = try recorder_mod.Recorder.init(self.ctx, 8, 32);
        defer rec.deinit();
        try rec.begin();
        try self.recordForward(&rec);
        try rec.endAndSubmit();
        try self.y.readBack(self.ctx, f32, out_pred);
    }

    /// Attach-mode step: record forward + loss-grad + backward + SGD
    /// into an existing host-owned Recorder, **without** submitting.
    /// The host (e.g. matryoshka's render thread) owns the command
    /// buffer and the submit cadence; this contributes a small block
    /// of dispatches to the host's per-frame command buffer.
    ///
    /// Buffer.update on the dynamic x/target buffers is safe to call
    /// here because those buffers are HOST_COHERENT; no extra barrier
    /// needed — the next dispatch sees the bytes.
    ///
    /// No `out_pred` parameter: a readback inside attach mode would
    /// need to block on the host's submit, breaking the cooperative
    /// model. Hosts that want the prediction should bind `runner.y`
    /// directly as an input SSBO to a downstream compute pass — keeps
    /// the prediction GPU-resident, which is exactly the engine-
    /// integration story.
    pub fn tickStepRecord(
        self: *TrainingRunner,
        rec: *recorder_mod.Recorder,
        x_in: []const f32,
        target_in: []const f32,
    ) !void {
        if (x_in.len != self.cfg.dim_in) return error.XDimMismatch;
        if (target_in.len != self.cfg.dim_out) return error.TargetDimMismatch;
        self.x_buf.update(f32, x_in);
        self.target_buf.update(f32, target_in);
        try self.recordStep(rec);
    }

    /// Attach-mode forward-only counterpart to `tickStepRecord`.
    /// Records just the forward pass into the host's recorder; weights
    /// are not modified. Pair this with a downstream pass that binds
    /// `runner.y` as input.
    pub fn tickPredictRecord(
        self: *TrainingRunner,
        rec: *recorder_mod.Recorder,
        x_in: []const f32,
    ) !void {
        if (x_in.len != self.cfg.dim_in) return error.XDimMismatch;
        self.x_buf.update(f32, x_in);
        try self.recordForward(rec);
    }

    /// Read back current weights into a CPU-side `cpu_train.Mlp`. Slow
    /// (one staging round-trip per parameter); intended for inspection,
    /// checkpointing, parity tests — not the hot path. Caller must have
    /// allocated the destination MLP with matching shape.
    pub fn readWeights(self: *const TrainingRunner, dst: *cpu_train.Mlp) !void {
        try self.w1.readBack(self.ctx, f32, dst.w1);
        try self.b1.readBack(self.ctx, f32, dst.b1);
        try self.w2.readBack(self.ctx, f32, dst.w2);
        try self.b2.readBack(self.ctx, f32, dst.b2);
    }

    /// Run one forward pass on a batch of `n_samples` inputs in a
    /// single dispatch and read the outputs back. Far cheaper than N
    /// sequential `tickPredict` calls — one submit, one waitIdle, one
    /// readBack regardless of N.
    ///
    /// `x_batch.len` must equal `n * dim_in`; `out_y.len` must equal
    /// `n * dim_out`. `n` may be ≤ `cfg.max_batch_size`. `cfg.max_batch_size`
    /// must have been > 0 at init time.
    ///
    /// Visualisation use case: pass the UV grid for a 16×16 tile
    /// surface and write the resulting RGB into per-tile materials.
    pub fn tickPredictBatch(self: *TrainingRunner, x_batch: []const f32, out_y: []f32) !void {
        const xb = self.x_batch orelse return error.BatchSizeNotConfigured;
        const yb = self.y_batch orelse return error.BatchSizeNotConfigured;
        const k = self.k_predict_batch orelse return error.BatchSizeNotConfigured;
        if (x_batch.len % self.cfg.dim_in != 0) return error.XBatchDimMismatch;
        const n: u32 = @intCast(x_batch.len / self.cfg.dim_in);
        if (n == 0) return;
        if (n > self.cfg.max_batch_size) return error.BatchSizeExceedsMax;
        if (out_y.len != n * self.cfg.dim_out) return error.OutBatchDimMismatch;

        @constCast(&xb).update(f32, x_batch);

        var rec = try recorder_mod.Recorder.init(self.ctx, 8, 32);
        defer rec.deinit();
        try rec.begin();

        const push = runtime.Mlp2ForwardBatchedPush{
            .dim_in = self.cfg.dim_in,
            .dim_hidden = self.cfg.dim_hidden,
            .dim_out = self.cfg.dim_out,
            .n_samples = n,
        };
        try rec.dispatch(
            &k,
            &.{ &self.w1, &self.b1, &self.w2, &self.b2, &xb, &yb },
            &push,
            ceilDiv(n, 64),
            1,
            1,
        );
        try rec.endAndSubmit();

        try yb.readBack(self.ctx, f32, out_y);
    }

    /// Attach-mode batched predict: record the dispatch into an
    /// existing host-owned Recorder. Caller must upload `x_batch`
    /// data into `runner.x_batch` (HOST_VISIBLE) themselves before
    /// the host's submit, and read `runner.y_batch` after submit
    /// completes. Provided for hosts that want zero per-frame submits
    /// from valkyr.
    pub fn tickPredictBatchRecord(self: *TrainingRunner, rec: *recorder_mod.Recorder, n_samples: u32) !void {
        const xb = self.x_batch orelse return error.BatchSizeNotConfigured;
        const yb = self.y_batch orelse return error.BatchSizeNotConfigured;
        const k = self.k_predict_batch orelse return error.BatchSizeNotConfigured;
        if (n_samples == 0) return;
        if (n_samples > self.cfg.max_batch_size) return error.BatchSizeExceedsMax;
        const push = runtime.Mlp2ForwardBatchedPush{
            .dim_in = self.cfg.dim_in,
            .dim_hidden = self.cfg.dim_hidden,
            .dim_out = self.cfg.dim_out,
            .n_samples = n_samples,
        };
        try rec.dispatch(
            &k,
            &.{ &self.w1, &self.b1, &self.w2, &self.b2, &xb, &yb },
            &push,
            ceilDiv(n_samples, 64),
            1,
            1,
        );
    }

    // ── Internal: dispatch sequences ───────────────────────────────

    fn recordForward(self: *const TrainingRunner, rec: *recorder_mod.Recorder) !void {
        const dim_in = self.cfg.dim_in;
        const dim_h = self.cfg.dim_hidden;
        const dim_out = self.cfg.dim_out;

        const matmul1_push = runtime.MatmulPush{ .m = 1, .n = dim_h, .k = dim_in };
        const matmul2_push = runtime.MatmulPush{ .m = 1, .n = dim_out, .k = dim_h };
        const add1_push = runtime.AddInPlacePush{ .n = dim_h };
        const add2_push = runtime.AddInPlacePush{ .n = dim_out };
        const relu_push = runtime.ReluPush{ .n = dim_h };

        try rec.dispatch(&self.k_matmul, &.{ &self.x_buf, &self.w1, &self.h_pre }, &matmul1_push, 1, 1, 1);
        try rec.dispatch(&self.k_add, &.{ &self.h_pre, &self.b1 }, &add1_push, ceilDiv(dim_h, 256), 1, 1);
        try rec.dispatch(&self.k_relu, &.{ &self.h_pre, &self.h }, &relu_push, ceilDiv(dim_h, 256), 1, 1);
        try rec.dispatch(&self.k_matmul, &.{ &self.h, &self.w2, &self.y }, &matmul2_push, 1, 1, 1);
        try rec.dispatch(&self.k_add, &.{ &self.y, &self.b2 }, &add2_push, ceilDiv(dim_out, 256), 1, 1);
    }

    fn recordStep(self: *const TrainingRunner, rec: *recorder_mod.Recorder) !void {
        try self.recordForward(rec);

        const dim_in = self.cfg.dim_in;
        const dim_h = self.cfg.dim_hidden;
        const dim_out = self.cfg.dim_out;

        // Loss grad.
        const mse_grad_push = runtime.MseLossGradPush{ .n = dim_out };
        try rec.dispatch(&self.k_mse_grad, &.{ &self.y, &self.target_buf, &self.dL_dy }, &mse_grad_push, ceilDiv(dim_out, 256), 1, 1);

        // Backward.
        const SliceCopyPush = extern struct { src_off: u32, dst_off: u32, n_elem: u32 };
        const op_dw2 = runtime.OuterProductPush{ .dim_out = dim_out, .dim_in = dim_h };
        const lin_dx_push = runtime.LinearBackwardDxPush{ .dim_out = dim_out, .dim_in = dim_h };
        const relu_bw_push = runtime.ReluBackwardPush{ .n = dim_h };
        const op_dw1 = runtime.OuterProductPush{ .dim_out = dim_h, .dim_in = dim_in };
        const copy_db2 = SliceCopyPush{ .src_off = 0, .dst_off = 0, .n_elem = dim_out };
        const copy_db1 = SliceCopyPush{ .src_off = 0, .dst_off = 0, .n_elem = dim_h };

        try rec.dispatch(&self.k_copy, &.{ &self.dL_dy, &self.db2 }, &copy_db2, ceilDiv(dim_out, 256), 1, 1);
        try rec.dispatch(&self.k_outer, &.{ &self.dL_dy, &self.h, &self.dw2 }, &op_dw2, ceilDiv(dim_out, 16), ceilDiv(dim_h, 16), 1);
        try rec.dispatch(&self.k_lin_dx, &.{ &self.dL_dy, &self.w2, &self.dh }, &lin_dx_push, ceilDiv(dim_h, 256), 1, 1);
        try rec.dispatch(&self.k_relu_bw, &.{ &self.dh, &self.h_pre, &self.dh_pre }, &relu_bw_push, ceilDiv(dim_h, 256), 1, 1);
        try rec.dispatch(&self.k_copy, &.{ &self.dh_pre, &self.db1 }, &copy_db1, ceilDiv(dim_h, 256), 1, 1);
        try rec.dispatch(&self.k_outer, &.{ &self.dh_pre, &self.x_buf, &self.dw1 }, &op_dw1, ceilDiv(dim_h, 16), ceilDiv(dim_in, 16), 1);

        // SGD: param[i] -= lr · grad[i]. Same shader for every buffer.
        const w1_n: u32 = dim_h * dim_in;
        const w2_n: u32 = dim_out * dim_h;
        const sgd_w1_push = runtime.SgdStepPush{ .n = w1_n, .lr = self.lr };
        const sgd_b1_push = runtime.SgdStepPush{ .n = dim_h, .lr = self.lr };
        const sgd_w2_push = runtime.SgdStepPush{ .n = w2_n, .lr = self.lr };
        const sgd_b2_push = runtime.SgdStepPush{ .n = dim_out, .lr = self.lr };

        try rec.dispatch(&self.k_sgd, &.{ &self.w1, &self.dw1 }, &sgd_w1_push, ceilDiv(w1_n, 256), 1, 1);
        try rec.dispatch(&self.k_sgd, &.{ &self.b1, &self.db1 }, &sgd_b1_push, ceilDiv(dim_h, 256), 1, 1);
        try rec.dispatch(&self.k_sgd, &.{ &self.w2, &self.dw2 }, &sgd_w2_push, ceilDiv(w2_n, 256), 1, 1);
        try rec.dispatch(&self.k_sgd, &.{ &self.b2, &self.db2 }, &sgd_b2_push, ceilDiv(dim_out, 256), 1, 1);
    }
};

fn ceilDiv(num: u32, den: u32) u32 {
    return (num + den - 1) / den;
}
