//! LoRA (Low-Rank Adaptation) Recorder-based dispatch helpers.
//!
//! Composes existing kernels (matmul_nt_v2, linear_backward_dx_batched,
//! linear_backward_dw_batched, scale, add_in_place) into the LoRA
//! forward and backward chains. No new SPIR-V — just dispatch sequences
//! recorded into a `gpu_recorder.Recorder` cmdbuf.
//!
//! Math (matches `cpu/lora.zig`'s parity oracle):
//!
//!   Forward:
//!     intermediate = x · Aᵀ                                  [M, r]
//!     y_base       = x · Wᵀ                                  [M, N]
//!     y_lora       = intermediate · Bᵀ                       [M, N]
//!     y            = y_base + (α/r) · y_lora                 [M, N]
//!
//!   Backward (given dy ∈ [M, N]):
//!     dy_B = dy · B                                          [M, r]   (reused)
//!     dx   = dy · W           + (α/r) · dy_B · A             [M, K]
//!     ∇A  += (α/r) · dy_Bᵀ · x                               [r, K]
//!     ∇B  += (α/r) · dyᵀ · intermediate                      [N, r]
//!
//! Per-LoRA-linear dispatch counts:
//!   forward  = 5 dispatches  (was 1 base matmul)
//!   backward = 9 dispatches  (was 2 — linear_backward_dx + linear_backward_dw)
//!
//! Designed for direct re-use inside `train_transformer.Runner.step`.
//! The existing `runGpuLoraSmoke` uses `submitOneShot` per dispatch; this
//! module's helpers go through `Recorder.dispatch` so all dispatches in
//! a forward+backward step share one cmdbuf and one submit.

const std = @import("std");
const util = @import("../util.zig");
const buffer = @import("../gpu/buffer.zig");
const pipeline = @import("../gpu/pipeline.zig");
const recorder_mod = @import("../gpu/recorder.zig");
const runtime = @import("../runtime.zig");
const runtime_hybrid = @import("../runtime_hybrid.zig");
const aliases = @import("../runtime_aliases.zig");

/// Dispatch group sizes — must match the shaders we compose:
///   matmul_nt_v2: 1D, ceilDiv(total_outputs, 1) — already 1 thread per cell
///   linear_backward_d{x,w}_batched: 2D 16×16 grid
///   scale, add_in_place: 1D, ceilDiv(n, 256)
const group_lwg: u32 = 16;
const group_lin: u32 = 256;

pub const LoraKernels = struct {
    matmul: *const pipeline.Kernel,
    lin_dx: *const pipeline.Kernel,
    lin_dw: *const pipeline.Kernel,
    scale: *const pipeline.Kernel,
    add_in_place: *const pipeline.Kernel,
};

pub const LoraShape = struct {
    M: u32,
    N: u32,
    K: u32,
    r: u32,
    alpha_over_r: f32,
};

pub const LoraForwardBuffers = struct {
    x: *const buffer.Buffer, // [M, K] input
    w: *const buffer.Buffer, // [N, K] frozen base weight
    a: *const buffer.Buffer, // [r, K] LoRA factor A
    b: *const buffer.Buffer, // [N, r] LoRA factor B
    y: *const buffer.Buffer, // [M, N] output (overwritten)
    intermediate_out: *const buffer.Buffer, // [M, r] saved for backward
    sc_y_lora: *const buffer.Buffer, // [M, N] scratch
    sc_y_lora_scaled: *const buffer.Buffer, // [M, N] scratch
};

pub const LoraBackwardBuffers = struct {
    dy: *const buffer.Buffer, // [M, N] upstream gradient
    x: *const buffer.Buffer, // [M, K] input from forward
    w: *const buffer.Buffer, // [N, K] frozen base weight
    a: *const buffer.Buffer, // [r, K] LoRA factor A
    b: *const buffer.Buffer, // [N, r] LoRA factor B
    intermediate: *const buffer.Buffer, // [M, r] from forward
    /// Output dx ∈ [M, K]. May be the same physical buffer as another
    /// dx target if the caller will accumulate elsewhere; we *overwrite*
    /// it here (matches `linear_backward_dx_batched`).
    dx: *const buffer.Buffer,
    /// ∇A ∈ [r, K]. We *overwrite* (linear_backward_dw_batched does too).
    dA: *const buffer.Buffer,
    /// ∇B ∈ [N, r]. Overwritten.
    dB: *const buffer.Buffer,
    sc_dy_B: *const buffer.Buffer, // [M, r]
    sc_dx_lora: *const buffer.Buffer, // [M, K]
    sc_dx_lora_scaled: *const buffer.Buffer, // [M, K]
    sc_dA_unscaled: *const buffer.Buffer, // [r, K]
    sc_dB_unscaled: *const buffer.Buffer, // [N, r]
};

/// Record the LoRA-augmented linear forward (5 dispatches) into the
/// active recorder cmdbuf. Caller is responsible for the recorder's
/// `begin` / `endAndSubmit` envelope.
pub fn recordLoraForward(
    rec: *recorder_mod.Recorder,
    kernels: LoraKernels,
    bufs: LoraForwardBuffers,
    shape: LoraShape,
) !void {
    const M = shape.M;
    const N = shape.N;
    const K = shape.K;
    const r = shape.r;

    // 1. y = x · Wᵀ                                       [M, N]
    const push_y_base = aliases.MatmulPush{ .m = M, .n = N, .k = K };
    try rec.dispatch(kernels.matmul, &.{ bufs.x, bufs.w, bufs.y }, &push_y_base, M * N, 1, 1);

    // 2. intermediate = x · Aᵀ                            [M, r]
    const push_inter = aliases.MatmulPush{ .m = M, .n = r, .k = K };
    try rec.dispatch(kernels.matmul, &.{ bufs.x, bufs.a, bufs.intermediate_out }, &push_inter, M * r, 1, 1);

    // 3. y_lora = intermediate · Bᵀ                       [M, N]
    const push_y_lora = aliases.MatmulPush{ .m = M, .n = N, .k = r };
    try rec.dispatch(kernels.matmul, &.{ bufs.intermediate_out, bufs.b, bufs.sc_y_lora }, &push_y_lora, M * N, 1, 1);

    // 4. sc_y_lora_scaled = y_lora * (α/r)
    const push_scale = runtime_hybrid.ScalePush{ .n = M * N, .scale = shape.alpha_over_r };
    try rec.dispatch(kernels.scale, &.{ bufs.sc_y_lora, bufs.sc_y_lora_scaled }, &push_scale, util.ceilDiv(M * N, group_lin), 1, 1);

    // 5. y += sc_y_lora_scaled
    const push_add = runtime.AddInPlacePush{ .n = M * N };
    try rec.dispatch(kernels.add_in_place, &.{ bufs.y, bufs.sc_y_lora_scaled }, &push_add, util.ceilDiv(M * N, group_lin), 1, 1);
}

/// Record the LoRA-augmented linear backward (9 dispatches). Caller
/// provides the upstream `dy` and is responsible for zero-init of `dA`
/// and `dB` if accumulation across micro-batches is needed (we
/// *overwrite* with the per-call gradients here, matching
/// `linear_backward_dw_batched`'s convention — accumulation is the
/// caller's job in a separate add pass).
pub fn recordLoraBackward(
    rec: *recorder_mod.Recorder,
    kernels: LoraKernels,
    bufs: LoraBackwardBuffers,
    shape: LoraShape,
) !void {
    const M = shape.M;
    const N = shape.N;
    const K = shape.K;
    const r = shape.r;

    // 1. dx = dy · W                                      [M, K]
    const push_dx_base = runtime.LinearBatchedPush{ .M = M, .N = N, .K = K };
    try rec.dispatch(
        kernels.lin_dx,
        &.{ bufs.dy, bufs.w, bufs.dx },
        &push_dx_base,
        util.ceilDiv(M, group_lwg),
        util.ceilDiv(K, group_lwg),
        1,
    );

    // 2. dy_B = dy · B                                    [M, r]
    //   linear_backward_dx semantics: dy[M,N] · W[N,K] → dx[M,K], with
    //   K substituted by r so the result is [M, r].
    const push_dy_B = runtime.LinearBatchedPush{ .M = M, .N = N, .K = r };
    try rec.dispatch(
        kernels.lin_dx,
        &.{ bufs.dy, bufs.b, bufs.sc_dy_B },
        &push_dy_B,
        util.ceilDiv(M, group_lwg),
        util.ceilDiv(r, group_lwg),
        1,
    );

    // 3. dx_lora = dy_B · A                               [M, K]
    //   N=r in the linear_backward_dx sense (it's the contracted dim).
    const push_dx_lora = runtime.LinearBatchedPush{ .M = M, .N = r, .K = K };
    try rec.dispatch(
        kernels.lin_dx,
        &.{ bufs.sc_dy_B, bufs.a, bufs.sc_dx_lora },
        &push_dx_lora,
        util.ceilDiv(M, group_lwg),
        util.ceilDiv(K, group_lwg),
        1,
    );

    // 4. sc_dx_lora_scaled = dx_lora * (α/r)
    const push_scale_dx = runtime_hybrid.ScalePush{ .n = M * K, .scale = shape.alpha_over_r };
    try rec.dispatch(
        kernels.scale,
        &.{ bufs.sc_dx_lora, bufs.sc_dx_lora_scaled },
        &push_scale_dx,
        util.ceilDiv(M * K, group_lin),
        1,
        1,
    );

    // 5. dx += sc_dx_lora_scaled
    const push_add_dx = runtime.AddInPlacePush{ .n = M * K };
    try rec.dispatch(
        kernels.add_in_place,
        &.{ bufs.dx, bufs.sc_dx_lora_scaled },
        &push_add_dx,
        util.ceilDiv(M * K, group_lin),
        1,
        1,
    );

    // 6. ∇A_unscaled = dy_Bᵀ · x                          [r, K]
    //   linear_backward_dw semantics: dyᵀ · x → dW. With N=r here so
    //   dW[r, K].
    const push_dA = runtime.LinearBatchedPush{ .M = M, .N = r, .K = K };
    try rec.dispatch(
        kernels.lin_dw,
        &.{ bufs.sc_dy_B, bufs.x, bufs.sc_dA_unscaled },
        &push_dA,
        util.ceilDiv(r, group_lwg),
        util.ceilDiv(K, group_lwg),
        1,
    );

    // 7. ∇A = ∇A_unscaled * (α/r)
    const push_scale_dA = runtime_hybrid.ScalePush{ .n = r * K, .scale = shape.alpha_over_r };
    try rec.dispatch(
        kernels.scale,
        &.{ bufs.sc_dA_unscaled, bufs.dA },
        &push_scale_dA,
        util.ceilDiv(r * K, group_lin),
        1,
        1,
    );

    // 8. ∇B_unscaled = dyᵀ · intermediate                 [N, r]
    const push_dB = runtime.LinearBatchedPush{ .M = M, .N = N, .K = r };
    try rec.dispatch(
        kernels.lin_dw,
        &.{ bufs.dy, bufs.intermediate, bufs.sc_dB_unscaled },
        &push_dB,
        util.ceilDiv(N, group_lwg),
        util.ceilDiv(r, group_lwg),
        1,
    );

    // 9. ∇B = ∇B_unscaled * (α/r)
    const push_scale_dB = runtime_hybrid.ScalePush{ .n = N * r, .scale = shape.alpha_over_r };
    try rec.dispatch(
        kernels.scale,
        &.{ bufs.sc_dB_unscaled, bufs.dB },
        &push_scale_dB,
        util.ceilDiv(N * r, group_lin),
        1,
        1,
    );
}
