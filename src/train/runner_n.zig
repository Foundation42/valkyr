//! MlpNRunner — multi-layer counterpart to TrainingRunner.
//!
//! Same per-frame on-device training surface as `runner.zig` but
//! generalised over depth via a `layer_dims: []const u32` config. n=2
//! produces the same numeric trajectory as `TrainingRunner` (modulo
//! the 2-layer fast-path's fused batched shader); n=3+ unlocks
//! deeper architectures for the post-v0 training arc.
//!
//! Approach-B sibling (see roadmap.md). MlpNRunner ships first as a
//! single-sample SGD-only path so the multi-layer surface is reachable
//! today; batched + Adam + cross-entropy + loss-target decay parity
//! with the existing 2-layer runner come in follow-up chunks. The
//! 2-layer fast path in `runner.zig` is untouched.
//!
//! Internals match the chunk-2 GPU parity smoke verbatim — same
//! primitive composition (matmul_nt + add_in_place + relu +
//! mse_loss_grad + slice_copy + outer_product + linear_backward_dx +
//! relu_backward + sgd_step), just lifted into a persistent runner
//! with per-layer parameter, activation, and gradient buffers.

const std = @import("std");
const vk = @import("../gpu/vk.zig");
const buffer = @import("../gpu/buffer.zig");
const pipeline = @import("../gpu/pipeline.zig");
const recorder_mod = @import("../gpu/recorder.zig");
const runtime = @import("../runtime.zig");
const cpu_train = @import("../cpu/train.zig");
const shaders = @import("shaders");

/// Configuration for an N-layer MLP runner. `layer_dims` has length n+1
/// and is borrowed by the caller — `init` duplicates it into runner-
/// owned memory, so the slice does not need to outlive the call.
pub const MlpNConfig = struct {
    /// Layer dimensions: [dim_in, h_1, h_2, ..., h_{n-1}, dim_out].
    /// Must have at least 2 entries (a single linear layer). The
    /// number of trainable layers is `layer_dims.len - 1`.
    layer_dims: []const u32,
    /// SGD learning rate. Sensible default for the toy demos: 0.05.
    /// Mutable across frames — write `runner.lr` between ticks for
    /// schedules.
    lr: f32,
    /// Initial weight scale: weights drawn from U(-init_scale, +init_scale).
    init_scale: f32 = 0.3,
    /// RNG seed for initial weights. Same seed → bit-identical starting
    /// MLP, matching `cpu_train.MlpN.init`.
    init_seed: u64 = 0xCAFE_4E_01,
};

/// Multi-layer MLP runner. Persistent parameter, activation, and
/// gradient buffers across the runner's lifetime; one submit per
/// `tickStep` (or zero submits when used via `tickStepRecord` from a
/// host-owned recorder).
pub const MlpNRunner = struct {
    ctx: *const vk.Context,
    allocator: std.mem.Allocator,
    layer_dims: []u32, // owned; length n+1
    lr: f32,
    init_seed: u64,

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

    // ── Per-layer parameter buffers (DEVICE_LOCAL, mutated by SGD) ──
    // Length n. Each weights[i] is shape [layer_dims[i+1], layer_dims[i]],
    // each biases[i] is length layer_dims[i+1].
    weights: []buffer.Buffer,
    biases: []buffer.Buffer,
    dw: []buffer.Buffer,
    db: []buffer.Buffer,

    // ── Per-step input/target (HOST_VISIBLE, host re-fills each frame) ──
    x_buf: buffer.Buffer,
    target_buf: buffer.Buffer,

    // ── Activation + gradient scratch (DEVICE_LOCAL, reused) ──
    // pre[L] is the pre-activation of layer L (used for the ReLU mask
    // on backward). post[L] is the ReLU output, fed as input to layer
    // L+1; for the output layer post[L] = pre[L] (no ReLU). dpre[L] is
    // the gradient flowing through layer L's pre-activation; dpost[L]
    // (only for L > 0) is the gradient feeding into layer L-1's post.
    pre: []buffer.Buffer,
    post: []buffer.Buffer,
    dpre: []buffer.Buffer,
    /// Index 0 unused; dpost[L] for L=1..n-1 carries the upstream
    /// gradient into layer L-1's d_pre via the ReLU mask.
    dpost: []buffer.Buffer,

    pub fn init(
        allocator: std.mem.Allocator,
        ctx: *const vk.Context,
        cfg: MlpNConfig,
    ) !MlpNRunner {
        if (cfg.layer_dims.len < 2) return error.InvalidLayerDims;

        // ── Dup layer_dims into runner-owned memory ───────────────
        const layer_dims = try allocator.alloc(u32, cfg.layer_dims.len);
        errdefer allocator.free(layer_dims);
        @memcpy(layer_dims, cfg.layer_dims);

        const n = layer_dims.len - 1;

        // ── Initial weights via the CPU oracle ───────────────────
        // Reusing `cpu_train.MlpN.init` guarantees the GPU runner
        // starts from a known starting point we can also recreate on
        // the CPU side.
        const seed_dims = try allocator.alloc(usize, layer_dims.len);
        defer allocator.free(seed_dims);
        for (layer_dims, 0..) |d, i| seed_dims[i] = d;
        var seed_mlp = try cpu_train.MlpN.init(
            allocator,
            seed_dims,
            cfg.init_scale,
            cfg.init_seed,
        );
        defer seed_mlp.deinit(allocator);

        // ── Pipelines ─────────────────────────────────────────────
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

        // ── Per-layer buffer arrays ───────────────────────────────
        const weights = try allocator.alloc(buffer.Buffer, n);
        errdefer allocator.free(weights);
        const biases = try allocator.alloc(buffer.Buffer, n);
        errdefer allocator.free(biases);
        const dw = try allocator.alloc(buffer.Buffer, n);
        errdefer allocator.free(dw);
        const db = try allocator.alloc(buffer.Buffer, n);
        errdefer allocator.free(db);
        const pre = try allocator.alloc(buffer.Buffer, n);
        errdefer allocator.free(pre);
        const post = try allocator.alloc(buffer.Buffer, n);
        errdefer allocator.free(post);
        const dpre = try allocator.alloc(buffer.Buffer, n);
        errdefer allocator.free(dpre);
        const dpost = try allocator.alloc(buffer.Buffer, n);
        errdefer allocator.free(dpost);

        // Track how many of each are populated so we can roll back on
        // partial failure mid-init.
        var n_w: usize = 0;
        var n_b: usize = 0;
        var n_dw: usize = 0;
        var n_db: usize = 0;
        var n_pre: usize = 0;
        var n_post: usize = 0;
        var n_dpre: usize = 0;
        var n_dpost: usize = 0; // index 0 unused
        errdefer for (weights[0..n_w]) |*buf| buf.deinit(ctx.device);
        errdefer for (biases[0..n_b]) |*buf| buf.deinit(ctx.device);
        errdefer for (dw[0..n_dw]) |*buf| buf.deinit(ctx.device);
        errdefer for (db[0..n_db]) |*buf| buf.deinit(ctx.device);
        errdefer for (pre[0..n_pre]) |*buf| buf.deinit(ctx.device);
        errdefer for (post[0..n_post]) |*buf| buf.deinit(ctx.device);
        errdefer for (dpre[0..n_dpre]) |*buf| buf.deinit(ctx.device);
        errdefer for (1..n_dpost + 1) |L| dpost[L].deinit(ctx.device);

        for (0..n) |L| {
            const dim_o: usize = @intCast(layer_dims[L + 1]);
            const dim_i: usize = @intCast(layer_dims[L]);

            weights[L] = try buffer.Buffer.initStatic(ctx, f32, seed_mlp.weights[L]);
            n_w += 1;
            biases[L] = try buffer.Buffer.initStatic(ctx, f32, seed_mlp.biases[L]);
            n_b += 1;
            dw[L] = try buffer.Buffer.initDeviceOnly(ctx, dim_o * dim_i * @sizeOf(f32));
            n_dw += 1;
            db[L] = try buffer.Buffer.initDeviceOnly(ctx, dim_o * @sizeOf(f32));
            n_db += 1;
            pre[L] = try buffer.Buffer.initDeviceOnly(ctx, dim_o * @sizeOf(f32));
            n_pre += 1;
            post[L] = try buffer.Buffer.initDeviceOnly(ctx, dim_o * @sizeOf(f32));
            n_post += 1;
            dpre[L] = try buffer.Buffer.initDeviceOnly(ctx, dim_o * @sizeOf(f32));
            n_dpre += 1;
        }
        for (1..n) |L| {
            const dim_below: usize = @intCast(layer_dims[L]);
            dpost[L] = try buffer.Buffer.initDeviceOnly(ctx, dim_below * @sizeOf(f32));
            n_dpost += 1;
        }

        // ── Per-step host-mapped input + target buffers ───────────
        const dim_in_bytes = layer_dims[0] * @sizeOf(f32);
        const dim_out_bytes = layer_dims[layer_dims.len - 1] * @sizeOf(f32);
        const x_buf = try buffer.Buffer.initDynamic(ctx, dim_in_bytes);
        errdefer @constCast(&x_buf).deinit(ctx.device);
        const target_buf = try buffer.Buffer.initDynamic(ctx, dim_out_bytes);
        errdefer @constCast(&target_buf).deinit(ctx.device);

        return .{
            .ctx = ctx,
            .allocator = allocator,
            .layer_dims = layer_dims,
            .lr = cfg.lr,
            .init_seed = cfg.init_seed,
            .k_matmul = k_matmul,
            .k_add = k_add,
            .k_relu = k_relu,
            .k_mse_grad = k_mse_grad,
            .k_relu_bw = k_relu_bw,
            .k_lin_dx = k_lin_dx,
            .k_outer = k_outer,
            .k_copy = k_copy,
            .k_sgd = k_sgd,
            .weights = weights,
            .biases = biases,
            .dw = dw,
            .db = db,
            .x_buf = x_buf,
            .target_buf = target_buf,
            .pre = pre,
            .post = post,
            .dpre = dpre,
            .dpost = dpost,
        };
    }

    pub fn deinit(self: *MlpNRunner) void {
        const dev = self.ctx.device;
        const n = self.nLayers();
        for (0..n) |L| {
            self.weights[L].deinit(dev);
            self.biases[L].deinit(dev);
            self.dw[L].deinit(dev);
            self.db[L].deinit(dev);
            self.pre[L].deinit(dev);
            self.post[L].deinit(dev);
            self.dpre[L].deinit(dev);
        }
        for (1..n) |L| self.dpost[L].deinit(dev);
        self.x_buf.deinit(dev);
        self.target_buf.deinit(dev);
        self.k_matmul.deinit();
        self.k_add.deinit();
        self.k_relu.deinit();
        self.k_mse_grad.deinit();
        self.k_relu_bw.deinit();
        self.k_lin_dx.deinit();
        self.k_outer.deinit();
        self.k_copy.deinit();
        self.k_sgd.deinit();
        self.allocator.free(self.weights);
        self.allocator.free(self.biases);
        self.allocator.free(self.dw);
        self.allocator.free(self.db);
        self.allocator.free(self.pre);
        self.allocator.free(self.post);
        self.allocator.free(self.dpre);
        self.allocator.free(self.dpost);
        self.allocator.free(self.layer_dims);
    }

    pub fn nLayers(self: *const MlpNRunner) usize {
        return self.layer_dims.len - 1;
    }
    pub fn dimIn(self: *const MlpNRunner) u32 {
        return self.layer_dims[0];
    }
    pub fn dimOut(self: *const MlpNRunner) u32 {
        return self.layer_dims[self.layer_dims.len - 1];
    }

    /// Run one full SGD step: upload x/target, record forward + loss-grad
    /// + backward + SGD, submit. If `out_pred` is non-null it's filled
    /// with the prediction y *before* the SGD step (the prediction the
    /// loss was computed against — the natural "what does the model
    /// say right now" reading).
    pub fn tickStep(
        self: *MlpNRunner,
        x_in: []const f32,
        target_in: []const f32,
        out_pred: ?[]f32,
    ) !void {
        if (x_in.len != self.dimIn()) return error.XDimMismatch;
        if (target_in.len != self.dimOut()) return error.TargetDimMismatch;
        if (out_pred) |op| if (op.len != self.dimOut()) return error.OutDimMismatch;

        self.x_buf.update(f32, x_in);
        self.target_buf.update(f32, target_in);

        // Per-layer dispatch budget: forward = 3 dispatches per layer
        // (matmul + add + relu/copy); backward = up to 4 per layer
        // (copy_db, outer_dw, lin_dx, relu_bw); sgd = 2 per layer (W + b);
        // plus 1 for mse_loss_grad. Round up generously.
        const n: u32 = @intCast(self.nLayers());
        const dispatches: u32 = 1 + 9 * n;
        var rec = try recorder_mod.Recorder.init(self.ctx, dispatches, 4 * dispatches);
        defer rec.deinit();
        try rec.begin();

        try self.recordStep(&rec);

        try rec.endAndSubmit();

        if (out_pred) |op| try self.post[n - 1].readBack(self.ctx, f32, op);
    }

    /// Forward-only: predict for `x_in`, fill `out_pred`. Does not modify
    /// weights.
    pub fn tickPredict(self: *MlpNRunner, x_in: []const f32, out_pred: []f32) !void {
        if (x_in.len != self.dimIn()) return error.XDimMismatch;
        if (out_pred.len != self.dimOut()) return error.OutDimMismatch;
        self.x_buf.update(f32, x_in);

        const n: u32 = @intCast(self.nLayers());
        const dispatches: u32 = 3 * n;
        var rec = try recorder_mod.Recorder.init(self.ctx, dispatches, 3 * dispatches);
        defer rec.deinit();
        try rec.begin();
        try self.recordForward(&rec);
        try rec.endAndSubmit();
        try self.post[n - 1].readBack(self.ctx, f32, out_pred);
    }

    /// Attach-mode step: record forward + loss-grad + backward + SGD
    /// into an existing host-owned Recorder, **without** submitting.
    /// Mirrors the contract of `TrainingRunner.tickStepRecord`.
    pub fn tickStepRecord(
        self: *MlpNRunner,
        rec: *recorder_mod.Recorder,
        x_in: []const f32,
        target_in: []const f32,
    ) !void {
        if (x_in.len != self.dimIn()) return error.XDimMismatch;
        if (target_in.len != self.dimOut()) return error.TargetDimMismatch;
        self.x_buf.update(f32, x_in);
        self.target_buf.update(f32, target_in);
        try self.recordStep(rec);
    }

    /// Attach-mode forward-only counterpart to `tickStepRecord`.
    pub fn tickPredictRecord(
        self: *MlpNRunner,
        rec: *recorder_mod.Recorder,
        x_in: []const f32,
    ) !void {
        if (x_in.len != self.dimIn()) return error.XDimMismatch;
        self.x_buf.update(f32, x_in);
        try self.recordForward(rec);
    }

    /// Read back current weights into a CPU-side `cpu_train.MlpN` with
    /// matching shape. Slow (one staging round-trip per parameter);
    /// inspection / parity / checkpointing only.
    pub fn readWeights(self: *const MlpNRunner, dst: *cpu_train.MlpN) !void {
        const n = self.nLayers();
        if (dst.weights.len != n) return error.LayerCountMismatch;
        for (0..n) |L| {
            try self.weights[L].readBack(self.ctx, f32, dst.weights[L]);
            try self.biases[L].readBack(self.ctx, f32, dst.biases[L]);
        }
    }

    fn recordForward(self: *const MlpNRunner, rec: *recorder_mod.Recorder) !void {
        const n = self.nLayers();
        for (0..n) |L| {
            const dim_o: u32 = self.layer_dims[L + 1];
            const dim_i: u32 = self.layer_dims[L];
            const input_buf = if (L == 0) &self.x_buf else &self.post[L - 1];
            const matmul_push = runtime.MatmulPush{ .m = 1, .n = dim_o, .k = dim_i };
            try rec.dispatch(&self.k_matmul, &.{ input_buf, &self.weights[L], &self.pre[L] }, &matmul_push, 1, 1, 1);
            const add_push = runtime.AddInPlacePush{ .n = dim_o };
            try rec.dispatch(&self.k_add, &.{ &self.pre[L], &self.biases[L] }, &add_push, ceilDiv(dim_o, 256), 1, 1);
            if (L + 1 < n) {
                const relu_push = runtime.ReluPush{ .n = dim_o };
                try rec.dispatch(&self.k_relu, &.{ &self.pre[L], &self.post[L] }, &relu_push, ceilDiv(dim_o, 256), 1, 1);
            } else {
                // Output layer has no ReLU — copy pre→post so all
                // downstream readers see a uniform "y is in post[n-1]".
                const SliceCopyPush = extern struct { src_off: u32, dst_off: u32, n_elem: u32 };
                const copy_push = SliceCopyPush{ .src_off = 0, .dst_off = 0, .n_elem = dim_o };
                try rec.dispatch(&self.k_copy, &.{ &self.pre[L], &self.post[L] }, &copy_push, ceilDiv(dim_o, 256), 1, 1);
            }
        }
    }

    fn recordStep(self: *const MlpNRunner, rec: *recorder_mod.Recorder) !void {
        const n = self.nLayers();
        try self.recordForward(rec);

        // Loss grad seeds d_pre on the output layer (no ReLU at output).
        const dim_out: u32 = self.dimOut();
        const mse_grad_push = runtime.MseLossGradPush{ .n = dim_out };
        try rec.dispatch(
            &self.k_mse_grad,
            &.{ &self.post[n - 1], &self.target_buf, &self.dpre[n - 1] },
            &mse_grad_push,
            ceilDiv(dim_out, 256),
            1,
            1,
        );

        // Backward, top-down. Each layer emits db = d_pre, dW = d_pre ⊗ input;
        // and (when there's a layer below) propagates d_post = W^T·d_pre then
        // applies the ReLU mask on the layer-below's pre to get d_pre[L-1].
        const SliceCopyPush = extern struct { src_off: u32, dst_off: u32, n_elem: u32 };
        var Lp1: usize = n;
        while (Lp1 > 0) : (Lp1 -= 1) {
            const L = Lp1 - 1;
            const dim_o: u32 = self.layer_dims[L + 1];
            const dim_i: u32 = self.layer_dims[L];
            const input_buf = if (L == 0) &self.x_buf else &self.post[L - 1];

            const copy_db = SliceCopyPush{ .src_off = 0, .dst_off = 0, .n_elem = dim_o };
            try rec.dispatch(&self.k_copy, &.{ &self.dpre[L], &self.db[L] }, &copy_db, ceilDiv(dim_o, 256), 1, 1);
            const op_push = runtime.OuterProductPush{ .dim_out = dim_o, .dim_in = dim_i };
            try rec.dispatch(&self.k_outer, &.{ &self.dpre[L], input_buf, &self.dw[L] }, &op_push, ceilDiv(dim_o, 16), ceilDiv(dim_i, 16), 1);
            if (L > 0) {
                const lin_dx_push = runtime.LinearBackwardDxPush{ .dim_out = dim_o, .dim_in = dim_i };
                try rec.dispatch(&self.k_lin_dx, &.{ &self.dpre[L], &self.weights[L], &self.dpost[L] }, &lin_dx_push, ceilDiv(dim_i, 256), 1, 1);
                const relu_bw_push = runtime.ReluBackwardPush{ .n = dim_i };
                try rec.dispatch(&self.k_relu_bw, &.{ &self.dpost[L], &self.pre[L - 1], &self.dpre[L - 1] }, &relu_bw_push, ceilDiv(dim_i, 256), 1, 1);
            }
        }

        // SGD updates run after all gradients are computed using the
        // pre-update weights. Order across layers doesn't matter.
        for (0..n) |L| {
            const dim_o: u32 = self.layer_dims[L + 1];
            const dim_i: u32 = self.layer_dims[L];
            const w_n: u32 = dim_o * dim_i;
            const sgd_w_push = runtime.SgdStepPush{ .n = w_n, .lr = self.lr };
            try rec.dispatch(&self.k_sgd, &.{ &self.weights[L], &self.dw[L] }, &sgd_w_push, ceilDiv(w_n, 256), 1, 1);
            const sgd_b_push = runtime.SgdStepPush{ .n = dim_o, .lr = self.lr };
            try rec.dispatch(&self.k_sgd, &.{ &self.biases[L], &self.db[L] }, &sgd_b_push, ceilDiv(dim_o, 256), 1, 1);
        }
    }
};

fn ceilDiv(num: u32, den: u32) u32 {
    return (num + den - 1) / den;
}
