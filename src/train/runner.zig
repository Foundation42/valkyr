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

pub const OptimizerKind = enum { sgd, adam };

/// Loss head selection. The loss-grad shader runs first in the
/// backward chain; everything downstream (`mlp2_dh_pre_batched`,
/// outer-product accumulators, optimizer step) is identical across
/// loss kinds because they all consume `dy = ∂L/∂y` regardless of
/// which L produced it.
pub const LossKind = enum {
    /// L = ½·Σ(y - target)² / N. Continuous regression; target is
    /// the desired output values directly.
    mse,
    /// L = -Σ target·log(softmax(y)) / N. Categorical classification
    /// or token prediction; target is the desired probability
    /// distribution (one-hot for hard labels). Logits y are passed
    /// raw — softmax happens inside the loss-grad shader.
    cross_entropy,
};

pub const Mlp2Config = struct {
    dim_in: u32,
    dim_hidden: u32,
    dim_out: u32,
    /// Optimizer learning rate. Sensible defaults differ by choice:
    /// SGD wants ~0.05 for our toy MLPs, Adam ~1e-3 for the same.
    /// Mutable across frames — write `runner.lr` between ticks for
    /// schedules.
    lr: f32,
    /// Initial weight scale: weights drawn from U(-init_scale, +init_scale).
    /// 0.3 lands the toy task in the well of convergence without hiding
    /// bugs behind clever init; bump up for harder tasks.
    init_scale: f32 = 0.3,
    /// Seed for the initial weight RNG. Same seed → same starting MLP,
    /// matching the CPU oracle's `Mlp.init`.
    init_seed: u64 = 0xCAFE,
    /// Max sample count for `tickPredictBatch` / `tickStepBatch`. Sized
    /// 0 disables the batched surface entirely (and skips its buffers).
    /// Visualisation use cases set this to N × N for a UV-grid overlay.
    max_batch_size: u32 = 0,
    /// Optimizer choice. `.sgd` is the simplest — `param ← param − lr·grad`.
    /// `.adam` adds running mean + running variance (per-parameter
    /// momentum buffers) and is far more sample-efficient for small
    /// MLPs. SGD path stays bit-exact when `.sgd` is selected.
    optimizer: OptimizerKind = .sgd,
    /// Adam exponential-decay rate for the first moment (mean).
    /// Standard ML default. Ignored under `.sgd`.
    adam_beta1: f32 = 0.9,
    /// Adam exponential-decay rate for the second moment (variance).
    /// Standard ML default. Ignored under `.sgd`.
    adam_beta2: f32 = 0.999,
    /// Adam denominator stabiliser. Standard ML default. Ignored under
    /// `.sgd`.
    adam_eps: f32 = 1e-8,
    /// Loss head. `.mse` regresses continuous targets; `.cross_entropy`
    /// fits categorical / token-prediction targets via stable softmax
    /// inside the loss-grad shader.
    loss: LossKind = .mse,
};

/// Compile-time cap on `dim_hidden` enforced by the batched forward
/// shader (it stores the hidden vector in a per-thread fp32 array of
/// this size). Mirrors the GLSL `MAX_HIDDEN` constant.
pub const MLP2_MAX_HIDDEN: u32 = 64;

/// Per-tickFrameTrain work cap. Mirrors `session.Budget` shape one-
/// for-one with `steps` substituted for `layers` — same union variants
/// (steps / microseconds / either), same defaults pattern, same
/// "sample wall-clock between units" semantics.
///
/// Wall-clock here is CPU recording time (the cost of vkCmd* calls
/// inside `recordTrainBatch`), not GPU execution time. That matches
/// `session.Budget` exactly: the inference runner also measures
/// recording-side wall-clock and lets the GPU run async after the
/// host submits. Hosts that want to cap GPU work specifically should
/// pre-calibrate "µs per step" and use `.steps`.
pub const TrainBudget = union(enum) {
    /// Run up to N batched SGD steps. Smallest legal value is 1 — a
    /// 0-step tickFrame is a no-op.
    steps: u32,
    /// Run steps until elapsed wall-clock ≥ µs cap. Sampled once per
    /// step (cheap), so one expensive step can overshoot by one unit.
    /// `microseconds = 0` is valid — tickFrameTrain returns immediately
    /// without recording.
    microseconds: u64,
    /// Both caps active; whichever fires first wins. Most embed
    /// callers want this — coarse step cap as backstop, fine µs cap
    /// as the real budget.
    either: struct { steps: u32, microseconds: u64 },

    pub fn defaults() TrainBudget {
        return .{ .steps = 1 };
    }
};

/// Result of a `tickFrameTrain` call. Lets hosts track schedule
/// stability frame-to-frame without sampling wall-clock themselves.
pub const TrainTickResult = struct {
    /// Number of batched SGD steps actually recorded this tick.
    steps_completed: u32,
    /// Wall-clock spent in `tickFrameTrain` recording, microseconds.
    /// Includes barrier emission and dispatch recording for every
    /// step that ran.
    elapsed_us: u64,
};

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
    /// Batched-training pipelines (only built when cfg.max_batch_size > 0).
    k_train_fwd_batch: ?pipeline.Kernel,
    k_train_dy_batch: ?pipeline.Kernel,
    k_train_dh_pre_batch: ?pipeline.Kernel,
    k_train_dw_accum: ?pipeline.Kernel,
    k_train_db_accum: ?pipeline.Kernel,
    /// Adam optimizer pipeline (only built when `cfg.optimizer == .adam`).
    k_adam: ?pipeline.Kernel,
    /// Cross-entropy loss-grad pipeline (only built when
    /// `cfg.loss == .cross_entropy`). MSE reuses the existing
    /// `k_train_dy_batch`.
    k_train_ce_loss_grad: ?pipeline.Kernel,

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
    /// Host-readback staging for predict outputs. Pair: GPU records
    /// `vkCmdCopyBuffer y_batch → y_batch_staging` into the host's
    /// cmd buffer (via `recordPredictReadback`); after the host's
    /// frame fence signals, CPU reads via `readPredictStaging`. No
    /// runner-side submits, no waitIdle.
    y_batch_staging: ?buffer.Buffer,

    /// Adam optimizer state. One pair (m, v) per parameter buffer.
    /// Allocated only when `cfg.optimizer == .adam`. Step counter
    /// is host-side and 1-indexed at dispatch time so the bias-
    /// correction terms work out the first call after init.
    adam_m_w1: ?buffer.Buffer,
    adam_v_w1: ?buffer.Buffer,
    adam_m_b1: ?buffer.Buffer,
    adam_v_b1: ?buffer.Buffer,
    adam_m_w2: ?buffer.Buffer,
    adam_v_w2: ?buffer.Buffer,
    adam_m_b2: ?buffer.Buffer,
    adam_v_b2: ?buffer.Buffer,
    adam_step: u32,

    /// Batched-training inputs (HOST_VISIBLE) and per-sample
    /// activations / per-sample backward scratch (DEVICE_LOCAL). Sized
    /// for `cfg.max_batch_size` rows. Allocated alongside the
    /// batched-predict surface — same `max_batch_size > 0` gate.
    x_train_batch: ?buffer.Buffer, // [N, dim_in]
    target_batch: ?buffer.Buffer, // [N, dim_out]
    h_pre_train_batch: ?buffer.Buffer, // [N, dim_hidden]
    h_train_batch: ?buffer.Buffer, // [N, dim_hidden]
    y_train_batch: ?buffer.Buffer, // [N, dim_out]
    dy_train_batch: ?buffer.Buffer, // [N, dim_out]
    dh_pre_train_batch: ?buffer.Buffer, // [N, dim_hidden]

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
        var k_train_fwd_batch: ?pipeline.Kernel = null;
        var k_train_dy_batch: ?pipeline.Kernel = null;
        var k_train_dh_pre_batch: ?pipeline.Kernel = null;
        var k_train_dw_accum: ?pipeline.Kernel = null;
        var k_train_db_accum: ?pipeline.Kernel = null;
        if (cfg.max_batch_size > 0) {
            k_predict_batch = try pipeline.Kernel.init(ctx, &shaders.mlp2_forward_batched, 6, @sizeOf(runtime.Mlp2ForwardBatchedPush));
            k_train_fwd_batch = try pipeline.Kernel.init(ctx, &shaders.mlp2_forward_train_batched, 8, @sizeOf(runtime.Mlp2ForwardTrainBatchedPush));
            k_train_dy_batch = try pipeline.Kernel.init(ctx, &shaders.mlp2_dy_batched, 3, @sizeOf(runtime.Mlp2DyBatchedPush));
            k_train_dh_pre_batch = try pipeline.Kernel.init(ctx, &shaders.mlp2_dh_pre_batched, 4, @sizeOf(runtime.Mlp2DhPreBatchedPush));
            k_train_dw_accum = try pipeline.Kernel.init(ctx, &shaders.mlp2_dw_accum, 3, @sizeOf(runtime.Mlp2DwAccumPush));
            k_train_db_accum = try pipeline.Kernel.init(ctx, &shaders.mlp2_db_accum, 2, @sizeOf(runtime.Mlp2DbAccumPush));
        }
        errdefer if (k_predict_batch) |*k| k.deinit();
        errdefer if (k_train_fwd_batch) |*k| k.deinit();
        errdefer if (k_train_dy_batch) |*k| k.deinit();
        errdefer if (k_train_dh_pre_batch) |*k| k.deinit();
        errdefer if (k_train_dw_accum) |*k| k.deinit();
        errdefer if (k_train_db_accum) |*k| k.deinit();
        var k_adam: ?pipeline.Kernel = null;
        if (cfg.optimizer == .adam) {
            k_adam = try pipeline.Kernel.init(ctx, &shaders.adam_step, 4, @sizeOf(runtime.AdamStepPush));
        }
        errdefer if (k_adam) |*k| k.deinit();
        var k_train_ce_loss_grad: ?pipeline.Kernel = null;
        if (cfg.loss == .cross_entropy and cfg.max_batch_size > 0) {
            k_train_ce_loss_grad = try pipeline.Kernel.init(
                ctx,
                &shaders.softmax_ce_loss_grad_batched,
                3,
                @sizeOf(runtime.SoftmaxCeLossGradPush),
            );
        }
        errdefer if (k_train_ce_loss_grad) |*k| k.deinit();

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
        var x_train_batch: ?buffer.Buffer = null;
        var target_batch: ?buffer.Buffer = null;
        var h_pre_train_batch: ?buffer.Buffer = null;
        var h_train_batch: ?buffer.Buffer = null;
        var y_train_batch: ?buffer.Buffer = null;
        var dy_train_batch: ?buffer.Buffer = null;
        var dh_pre_train_batch: ?buffer.Buffer = null;
        var y_batch_staging: ?buffer.Buffer = null;
        if (cfg.max_batch_size > 0) {
            x_batch = try buffer.Buffer.initDynamic(ctx, cfg.max_batch_size * cfg.dim_in * @sizeOf(f32));
            errdefer if (x_batch) |*b| b.deinit(ctx.device);
            y_batch = try buffer.Buffer.initDeviceOnly(ctx, cfg.max_batch_size * cfg.dim_out * @sizeOf(f32));
            errdefer if (y_batch) |*b| b.deinit(ctx.device);
            y_batch_staging = try buffer.Buffer.initHostReadback(ctx, cfg.max_batch_size * cfg.dim_out * @sizeOf(f32));
            errdefer if (y_batch_staging) |*b| b.deinit(ctx.device);
            x_train_batch = try buffer.Buffer.initDynamic(ctx, cfg.max_batch_size * cfg.dim_in * @sizeOf(f32));
            errdefer if (x_train_batch) |*b| b.deinit(ctx.device);
            target_batch = try buffer.Buffer.initDynamic(ctx, cfg.max_batch_size * cfg.dim_out * @sizeOf(f32));
            errdefer if (target_batch) |*b| b.deinit(ctx.device);
            h_pre_train_batch = try buffer.Buffer.initDeviceOnly(ctx, cfg.max_batch_size * cfg.dim_hidden * @sizeOf(f32));
            errdefer if (h_pre_train_batch) |*b| b.deinit(ctx.device);
            h_train_batch = try buffer.Buffer.initDeviceOnly(ctx, cfg.max_batch_size * cfg.dim_hidden * @sizeOf(f32));
            errdefer if (h_train_batch) |*b| b.deinit(ctx.device);
            y_train_batch = try buffer.Buffer.initDeviceOnly(ctx, cfg.max_batch_size * cfg.dim_out * @sizeOf(f32));
            errdefer if (y_train_batch) |*b| b.deinit(ctx.device);
            dy_train_batch = try buffer.Buffer.initDeviceOnly(ctx, cfg.max_batch_size * cfg.dim_out * @sizeOf(f32));
            errdefer if (dy_train_batch) |*b| b.deinit(ctx.device);
            dh_pre_train_batch = try buffer.Buffer.initDeviceOnly(ctx, cfg.max_batch_size * cfg.dim_hidden * @sizeOf(f32));
            errdefer if (dh_pre_train_batch) |*b| b.deinit(ctx.device);
        }

        var adam_m_w1: ?buffer.Buffer = null;
        var adam_v_w1: ?buffer.Buffer = null;
        var adam_m_b1: ?buffer.Buffer = null;
        var adam_v_b1: ?buffer.Buffer = null;
        var adam_m_w2: ?buffer.Buffer = null;
        var adam_v_w2: ?buffer.Buffer = null;
        var adam_m_b2: ?buffer.Buffer = null;
        var adam_v_b2: ?buffer.Buffer = null;
        if (cfg.optimizer == .adam) {
            // Adam moment buffers — same shape as their parameter
            // buffers, zero-initialised (initDeviceOnly does that
            // for us via vkCmdFillBuffer).
            adam_m_w1 = try buffer.Buffer.initDeviceOnly(ctx, seed_mlp.w1.len * @sizeOf(f32));
            errdefer if (adam_m_w1) |*b| b.deinit(ctx.device);
            adam_v_w1 = try buffer.Buffer.initDeviceOnly(ctx, seed_mlp.w1.len * @sizeOf(f32));
            errdefer if (adam_v_w1) |*b| b.deinit(ctx.device);
            adam_m_b1 = try buffer.Buffer.initDeviceOnly(ctx, seed_mlp.b1.len * @sizeOf(f32));
            errdefer if (adam_m_b1) |*b| b.deinit(ctx.device);
            adam_v_b1 = try buffer.Buffer.initDeviceOnly(ctx, seed_mlp.b1.len * @sizeOf(f32));
            errdefer if (adam_v_b1) |*b| b.deinit(ctx.device);
            adam_m_w2 = try buffer.Buffer.initDeviceOnly(ctx, seed_mlp.w2.len * @sizeOf(f32));
            errdefer if (adam_m_w2) |*b| b.deinit(ctx.device);
            adam_v_w2 = try buffer.Buffer.initDeviceOnly(ctx, seed_mlp.w2.len * @sizeOf(f32));
            errdefer if (adam_v_w2) |*b| b.deinit(ctx.device);
            adam_m_b2 = try buffer.Buffer.initDeviceOnly(ctx, seed_mlp.b2.len * @sizeOf(f32));
            errdefer if (adam_m_b2) |*b| b.deinit(ctx.device);
            adam_v_b2 = try buffer.Buffer.initDeviceOnly(ctx, seed_mlp.b2.len * @sizeOf(f32));
            errdefer if (adam_v_b2) |*b| b.deinit(ctx.device);
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
            .k_train_fwd_batch = k_train_fwd_batch,
            .k_train_dy_batch = k_train_dy_batch,
            .k_train_dh_pre_batch = k_train_dh_pre_batch,
            .k_train_dw_accum = k_train_dw_accum,
            .k_train_db_accum = k_train_db_accum,
            .k_adam = k_adam,
            .k_train_ce_loss_grad = k_train_ce_loss_grad,
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
            .y_batch_staging = y_batch_staging,
            .x_train_batch = x_train_batch,
            .target_batch = target_batch,
            .h_pre_train_batch = h_pre_train_batch,
            .h_train_batch = h_train_batch,
            .y_train_batch = y_train_batch,
            .dy_train_batch = dy_train_batch,
            .dh_pre_train_batch = dh_pre_train_batch,
            .adam_m_w1 = adam_m_w1,
            .adam_v_w1 = adam_v_w1,
            .adam_m_b1 = adam_m_b1,
            .adam_v_b1 = adam_v_b1,
            .adam_m_w2 = adam_m_w2,
            .adam_v_w2 = adam_v_w2,
            .adam_m_b2 = adam_m_b2,
            .adam_v_b2 = adam_v_b2,
            .adam_step = 0,
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
        if (self.x_batch) |*b| b.deinit(dev);
        if (self.y_batch) |*b| b.deinit(dev);
        if (self.y_batch_staging) |*b| b.deinit(dev);
        if (self.x_train_batch) |*b| b.deinit(dev);
        if (self.target_batch) |*b| b.deinit(dev);
        if (self.h_pre_train_batch) |*b| b.deinit(dev);
        if (self.h_train_batch) |*b| b.deinit(dev);
        if (self.y_train_batch) |*b| b.deinit(dev);
        if (self.dy_train_batch) |*b| b.deinit(dev);
        if (self.dh_pre_train_batch) |*b| b.deinit(dev);
        if (self.adam_m_w1) |*b| b.deinit(dev);
        if (self.adam_v_w1) |*b| b.deinit(dev);
        if (self.adam_m_b1) |*b| b.deinit(dev);
        if (self.adam_v_b1) |*b| b.deinit(dev);
        if (self.adam_m_w2) |*b| b.deinit(dev);
        if (self.adam_v_w2) |*b| b.deinit(dev);
        if (self.adam_m_b2) |*b| b.deinit(dev);
        if (self.adam_v_b2) |*b| b.deinit(dev);
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
        if (self.k_train_fwd_batch) |*k| k.deinit();
        if (self.k_train_dy_batch) |*k| k.deinit();
        if (self.k_train_dh_pre_batch) |*k| k.deinit();
        if (self.k_train_dw_accum) |*k| k.deinit();
        if (self.k_train_db_accum) |*k| k.deinit();
        if (self.k_adam) |*k| k.deinit();
        if (self.k_train_ce_loss_grad) |*k| k.deinit();
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

    /// Run one batched SGD step over `n_samples` (inputs, targets):
    /// upload into the host-mapped x_train / target buffers, record
    /// the full forward → loss-grad → backward → SGD chain in one
    /// submit, wait. Mean gradient over the batch is automatic —
    /// dy is pre-divided by N inside `mlp2_dy_batched.comp`.
    ///
    /// `x_batch.len = n * dim_in`, `target_batch.len = n * dim_out`,
    /// `n ≤ cfg.max_batch_size`. `cfg.max_batch_size` must have been
    /// > 0 at init.
    ///
    /// Roughly 9× faster than N sequential `tickStep` calls — same
    /// bookkeeping (one submit, one waitIdle) for any batch size.
    pub fn tickStepBatch(self: *TrainingRunner, x_batch: []const f32, target_batch: []const f32) !void {
        if (self.x_train_batch == null) return error.BatchSizeNotConfigured;
        if (x_batch.len % self.cfg.dim_in != 0) return error.XBatchDimMismatch;
        const n: u32 = @intCast(x_batch.len / self.cfg.dim_in);
        if (n == 0) return;
        if (n > self.cfg.max_batch_size) return error.BatchSizeExceedsMax;
        if (target_batch.len != n * self.cfg.dim_out) return error.TargetBatchDimMismatch;

        @constCast(&self.x_train_batch.?).update(f32, x_batch);
        @constCast(&self.target_batch.?).update(f32, target_batch);

        var rec = try recorder_mod.Recorder.init(self.ctx, 16, 64);
        defer rec.deinit();
        try rec.begin();
        try self.recordTrainBatch(&rec, n);
        try rec.endAndSubmit();
    }

    /// Attach-mode batched train: record the chain into an existing
    /// host-owned Recorder, no submit. Caller must have populated
    /// `runner.x_train_batch` and `runner.target_batch` already (via
    /// `runner.uploadTrainBatch`).
    pub fn tickStepBatchRecord(self: *TrainingRunner, rec: *recorder_mod.Recorder, n_samples: u32) !void {
        if (self.x_train_batch == null) return error.BatchSizeNotConfigured;
        if (n_samples == 0) return;
        if (n_samples > self.cfg.max_batch_size) return error.BatchSizeExceedsMax;
        try self.recordTrainBatch(rec, n_samples);
    }

    /// Stage a (x_batch, target_batch) pair into the host-mapped
    /// training buffers. Cheap memcpy — backs onto Buffer.update on
    /// the dynamic buffers; the next training dispatch reads from them
    /// directly.
    pub fn uploadTrainBatch(self: *TrainingRunner, x_batch: []const f32, target_batch: []const f32) !void {
        if (self.x_train_batch == null) return error.BatchSizeNotConfigured;
        if (x_batch.len % self.cfg.dim_in != 0) return error.XBatchDimMismatch;
        const n: u32 = @intCast(x_batch.len / self.cfg.dim_in);
        if (n > self.cfg.max_batch_size) return error.BatchSizeExceedsMax;
        if (target_batch.len != n * self.cfg.dim_out) return error.TargetBatchDimMismatch;
        @constCast(&self.x_train_batch.?).update(f32, x_batch);
        @constCast(&self.target_batch.?).update(f32, target_batch);
    }

    /// Stage a batch of predict inputs into `runner.x_batch`. Mirrors
    /// `uploadTrainBatch` for the predict path. Use before
    /// `tickPredictBatchRecord` in attach mode.
    pub fn uploadPredictInputs(self: *TrainingRunner, x_batch: []const f32) !void {
        if (self.x_batch == null) return error.BatchSizeNotConfigured;
        if (x_batch.len % self.cfg.dim_in != 0) return error.XBatchDimMismatch;
        const n: u32 = @intCast(x_batch.len / self.cfg.dim_in);
        if (n > self.cfg.max_batch_size) return error.BatchSizeExceedsMax;
        @constCast(&self.x_batch.?).update(f32, x_batch);
    }

    /// Record a transfer-stage copy of `y_batch` (device-only) into
    /// `y_batch_staging` (host-mapped). Insert AFTER
    /// `tickPredictBatchRecord`, BEFORE the host ends + submits the
    /// command buffer. Emits the compute→transfer barrier between the
    /// predict dispatch's writes and this copy's reads.
    ///
    /// After the host's frame fence signals, the staging buffer's
    /// mapped region is guaranteed visible to the CPU; read it via
    /// `readPredictStaging`.
    pub fn recordPredictReadback(
        self: *const TrainingRunner,
        rec: *recorder_mod.Recorder,
        n_samples: u32,
    ) !void {
        if (self.y_batch == null or self.y_batch_staging == null) return error.BatchSizeNotConfigured;
        if (n_samples == 0) return;
        if (n_samples > self.cfg.max_batch_size) return error.BatchSizeExceedsMax;

        const cmd = rec.cmd;

        // Compute-write → transfer-read barrier on the predict output.
        var mb = std.mem.zeroes(vk.c.VkMemoryBarrier);
        mb.sType = vk.c.VK_STRUCTURE_TYPE_MEMORY_BARRIER;
        mb.srcAccessMask = vk.c.VK_ACCESS_SHADER_WRITE_BIT;
        mb.dstAccessMask = vk.c.VK_ACCESS_TRANSFER_READ_BIT;
        vk.c.vkCmdPipelineBarrier(
            cmd,
            vk.c.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            vk.c.VK_PIPELINE_STAGE_TRANSFER_BIT,
            0,
            1,
            &mb,
            0,
            null,
            0,
            null,
        );

        const bytes: usize = @as(usize, n_samples) * @as(usize, self.cfg.dim_out) * @sizeOf(f32);
        const region = vk.c.VkBufferCopy{
            .srcOffset = 0,
            .dstOffset = 0,
            .size = @intCast(bytes),
        };
        vk.c.vkCmdCopyBuffer(
            cmd,
            self.y_batch.?.handle,
            self.y_batch_staging.?.handle,
            1,
            &region,
        );
    }

    /// Read predictions from the host-mapped staging buffer. Caller
    /// MUST have waited on the fence covering the submit that issued
    /// the recorded `recordPredictReadback` — otherwise the read may
    /// observe stale or partially-written data.
    pub fn readPredictStaging(self: *const TrainingRunner, out_y: []f32) !void {
        const staging = self.y_batch_staging orelse return error.BatchSizeNotConfigured;
        const mapped = staging.mapped orelse return error.StagingNotMapped;
        const bytes = out_y.len * @sizeOf(f32);
        if (bytes > staging.bytes) return error.OutputTooLarge;
        @memcpy(std.mem.sliceAsBytes(out_y), @as([*]u8, @ptrCast(mapped))[0..bytes]);
    }

    /// Cooperative training tick. Records up to `budget` batched SGD
    /// steps over the staged (x_train_batch, target_batch) into the
    /// host's Recorder, no submit. The host owns the submit cadence.
    ///
    /// Multi-step mode runs the same staged batch through multiple
    /// SGD steps within one frame — useful when the host has CPU-side
    /// supervision data that updates slowly and wants to spend more
    /// GPU budget when there's frame-time headroom.
    ///
    /// Mirrors `Session.tickFrame` shape: caller passes the budget,
    /// runner decides how many units to actually do based on wall-
    /// clock sampling between units. Returns step-count + elapsed for
    /// telemetry / schedule stability.
    pub fn tickFrameTrain(
        self: *TrainingRunner,
        rec: *recorder_mod.Recorder,
        budget: TrainBudget,
        n_samples: u32,
    ) !TrainTickResult {
        if (self.x_train_batch == null) return error.BatchSizeNotConfigured;
        if (n_samples == 0) return .{ .steps_completed = 0, .elapsed_us = 0 };
        if (n_samples > self.cfg.max_batch_size) return error.BatchSizeExceedsMax;

        const max_steps: u32 = switch (budget) {
            .steps => |s| s,
            .either => |e| e.steps,
            .microseconds => std.math.maxInt(u32),
        };
        const us_cap: u64 = switch (budget) {
            .microseconds => |us| us,
            .either => |e| e.microseconds,
            .steps => std.math.maxInt(u64),
        };
        if (max_steps == 0 or us_cap == 0) {
            return .{ .steps_completed = 0, .elapsed_us = 0 };
        }

        const t0 = std.time.nanoTimestamp();
        var steps_done: u32 = 0;
        while (steps_done < max_steps) {
            try self.recordTrainBatch(rec, n_samples);
            steps_done += 1;
            const elapsed_ns = std.time.nanoTimestamp() - t0;
            const elapsed_us: u64 = @intCast(@divTrunc(elapsed_ns, 1000));
            if (elapsed_us >= us_cap) break;
        }
        const final_ns = std.time.nanoTimestamp() - t0;
        return .{
            .steps_completed = steps_done,
            .elapsed_us = @intCast(@divTrunc(final_ns, 1000)),
        };
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

    /// Record a complete batched optimizer step into `rec`. 11 dispatches:
    /// forward, dy, dh_pre, dw2 accum, db2 accum, dw1 accum, db1 accum,
    /// then 4× optimizer step (W1, b1, W2, b2 — SGD or Adam depending
    /// on `cfg.optimizer`). Caller must have populated `x_train_batch`
    /// and `target_batch` already.
    fn recordTrainBatch(self: *TrainingRunner, rec: *recorder_mod.Recorder, n_samples: u32) !void {
        const dim_in = self.cfg.dim_in;
        const dim_h = self.cfg.dim_hidden;
        const dim_out = self.cfg.dim_out;

        const k_fwd = &(self.k_train_fwd_batch.?);
        const k_dy = &(self.k_train_dy_batch.?);
        const k_dhp = &(self.k_train_dh_pre_batch.?);
        const k_dw = &(self.k_train_dw_accum.?);
        const k_db = &(self.k_train_db_accum.?);
        const x_train = &(self.x_train_batch.?);
        const tgt = &(self.target_batch.?);
        const h_pre_b = &(self.h_pre_train_batch.?);
        const h_b = &(self.h_train_batch.?);
        const y_b = &(self.y_train_batch.?);
        const dy_b = &(self.dy_train_batch.?);
        const dhp_b = &(self.dh_pre_train_batch.?);

        // 1. forward → h_pre, h, y
        const fwd_push = runtime.Mlp2ForwardTrainBatchedPush{
            .dim_in = dim_in,
            .dim_hidden = dim_h,
            .dim_out = dim_out,
            .n_samples = n_samples,
        };
        try rec.dispatch(
            k_fwd,
            &.{ &self.w1, &self.b1, &self.w2, &self.b2, x_train, h_pre_b, h_b, y_b },
            &fwd_push,
            ceilDiv(n_samples, 64),
            1,
            1,
        );

        // 2. dy = ∂L/∂y, scaled by 1/N (mean over batch).
        //    MSE: dy = (y − target) / N — `mlp2_dy_batched`
        //    CE:  dy = (softmax(y) − target) / N — `softmax_ce_loss_grad_batched`
        // Downstream backward chain is identical across loss kinds.
        switch (self.cfg.loss) {
            .mse => {
                const dy_push = runtime.Mlp2DyBatchedPush{ .dim_out = dim_out, .n_samples = n_samples };
                try rec.dispatch(
                    k_dy,
                    &.{ y_b, tgt, dy_b },
                    &dy_push,
                    ceilDiv(dim_out * n_samples, 256),
                    1,
                    1,
                );
            },
            .cross_entropy => {
                const k_ce = &(self.k_train_ce_loss_grad.?);
                const ce_push = runtime.SoftmaxCeLossGradPush{ .dim_out = dim_out, .n_samples = n_samples };
                try rec.dispatch(
                    k_ce,
                    &.{ y_b, tgt, dy_b },
                    &ce_push,
                    ceilDiv(n_samples, 64),
                    1,
                    1,
                );
            },
        }

        // 3. dh_pre = (W2^T · dy) · 1[h_pre > 0]
        const dhp_push = runtime.Mlp2DhPreBatchedPush{
            .dim_hidden = dim_h,
            .dim_out = dim_out,
            .n_samples = n_samples,
        };
        try rec.dispatch(
            k_dhp,
            &.{ dy_b, &self.w2, h_pre_b, dhp_b },
            &dhp_push,
            ceilDiv(dim_h * n_samples, 256),
            1,
            1,
        );

        // 4. dW2[o, h] = Σ_n dy[n, o] · h[n, h]
        const dw2_push = runtime.Mlp2DwAccumPush{
            .dim_i = dim_out,
            .dim_j = dim_h,
            .n_samples = n_samples,
        };
        try rec.dispatch(
            k_dw,
            &.{ dy_b, h_b, &self.dw2 },
            &dw2_push,
            ceilDiv(dim_out, 16),
            ceilDiv(dim_h, 16),
            1,
        );

        // 5. db2[o] = Σ_n dy[n, o]
        const db2_push = runtime.Mlp2DbAccumPush{ .dim_i = dim_out, .n_samples = n_samples };
        try rec.dispatch(
            k_db,
            &.{ dy_b, &self.db2 },
            &db2_push,
            ceilDiv(dim_out, 256),
            1,
            1,
        );

        // 6. dW1[h, k] = Σ_n dh_pre[n, h] · x[n, k]
        const dw1_push = runtime.Mlp2DwAccumPush{
            .dim_i = dim_h,
            .dim_j = dim_in,
            .n_samples = n_samples,
        };
        try rec.dispatch(
            k_dw,
            &.{ dhp_b, x_train, &self.dw1 },
            &dw1_push,
            ceilDiv(dim_h, 16),
            ceilDiv(dim_in, 16),
            1,
        );

        // 7. db1[h] = Σ_n dh_pre[n, h]
        const db1_push = runtime.Mlp2DbAccumPush{ .dim_i = dim_h, .n_samples = n_samples };
        try rec.dispatch(
            k_db,
            &.{ dhp_b, &self.db1 },
            &db1_push,
            ceilDiv(dim_h, 256),
            1,
            1,
        );

        // 8. Optimizer step on all four params.
        const w1_n: u32 = dim_h * dim_in;
        const w2_n: u32 = dim_out * dim_h;
        switch (self.cfg.optimizer) {
            .sgd => {
                const sgd_w1 = runtime.SgdStepPush{ .n = w1_n, .lr = self.lr };
                const sgd_b1 = runtime.SgdStepPush{ .n = dim_h, .lr = self.lr };
                const sgd_w2 = runtime.SgdStepPush{ .n = w2_n, .lr = self.lr };
                const sgd_b2 = runtime.SgdStepPush{ .n = dim_out, .lr = self.lr };
                try rec.dispatch(&self.k_sgd, &.{ &self.w1, &self.dw1 }, &sgd_w1, ceilDiv(w1_n, 256), 1, 1);
                try rec.dispatch(&self.k_sgd, &.{ &self.b1, &self.db1 }, &sgd_b1, ceilDiv(dim_h, 256), 1, 1);
                try rec.dispatch(&self.k_sgd, &.{ &self.w2, &self.dw2 }, &sgd_w2, ceilDiv(w2_n, 256), 1, 1);
                try rec.dispatch(&self.k_sgd, &.{ &self.b2, &self.db2 }, &sgd_b2, ceilDiv(dim_out, 256), 1, 1);
            },
            .adam => {
                // Step counter is 1-indexed at dispatch — host bumps it
                // BEFORE scheduling the dispatch so the bias-correction
                // terms work out the first call after init / reset.
                const k_adam = &(self.k_adam.?);
                const m_w1 = &(self.adam_m_w1.?);
                const v_w1 = &(self.adam_v_w1.?);
                const m_b1 = &(self.adam_m_b1.?);
                const v_b1 = &(self.adam_v_b1.?);
                const m_w2 = &(self.adam_m_w2.?);
                const v_w2 = &(self.adam_v_w2.?);
                const m_b2 = &(self.adam_m_b2.?);
                const v_b2 = &(self.adam_v_b2.?);
                self.adam_step +%= 1;
                const t = self.adam_step;
                const adam_w1 = runtime.AdamStepPush{
                    .n = w1_n, .lr = self.lr, .beta1 = self.cfg.adam_beta1,
                    .beta2 = self.cfg.adam_beta2, .eps = self.cfg.adam_eps, .t = t,
                };
                const adam_b1 = runtime.AdamStepPush{
                    .n = dim_h, .lr = self.lr, .beta1 = self.cfg.adam_beta1,
                    .beta2 = self.cfg.adam_beta2, .eps = self.cfg.adam_eps, .t = t,
                };
                const adam_w2 = runtime.AdamStepPush{
                    .n = w2_n, .lr = self.lr, .beta1 = self.cfg.adam_beta1,
                    .beta2 = self.cfg.adam_beta2, .eps = self.cfg.adam_eps, .t = t,
                };
                const adam_b2 = runtime.AdamStepPush{
                    .n = dim_out, .lr = self.lr, .beta1 = self.cfg.adam_beta1,
                    .beta2 = self.cfg.adam_beta2, .eps = self.cfg.adam_eps, .t = t,
                };
                try rec.dispatch(k_adam, &.{ &self.w1, &self.dw1, m_w1, v_w1 }, &adam_w1, ceilDiv(w1_n, 256), 1, 1);
                try rec.dispatch(k_adam, &.{ &self.b1, &self.db1, m_b1, v_b1 }, &adam_b1, ceilDiv(dim_h, 256), 1, 1);
                try rec.dispatch(k_adam, &.{ &self.w2, &self.dw2, m_w2, v_w2 }, &adam_w2, ceilDiv(w2_n, 256), 1, 1);
                try rec.dispatch(k_adam, &.{ &self.b2, &self.db2, m_b2, v_b2 }, &adam_b2, ceilDiv(dim_out, 256), 1, 1);
            },
        }
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
