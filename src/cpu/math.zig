//! CPU reference math primitives.
//!
//! These exist for correctness, not speed. Every kernel has a GPU
//! counterpart in `shaders/`; the CPU version here is the oracle the
//! GPU version must match to within fp32 tolerance, and (where the
//! arithmetic order matches) is also what we'll diff against TRiP's
//! C reference for absolute correctness.
//!
//! All operations take fp32 activations; weight tensors retain their
//! storage dtype (bf16/fp16/f32 — whichever the checkpoint shipped) and
//! are converted lazily, scalar-by-scalar, exactly like TRiP does. This
//! matches the C reference's accumulation order, which we want when we
//! eventually diff outputs.

const std = @import("std");
const safetensors = @import("../safetensors.zig");
const config_mod = @import("../config.zig");
const dtype = @import("../dtype.zig");

const Tensor = safetensors.Tensor;
const Family = config_mod.Family;

/// out[i] = (gain[i] / rms) * in[i]
/// where:
///     rms  = sqrt(mean(in[i]²) + eps)
///     gain = (1 + weight[i])  if family == .gemma
///          = weight[i]         if family == .llama
///
/// Length of all three slices is `dim`. `weight` is read straight from
/// the checkpoint; we dispatch on its dtype here rather than ask the
/// caller to pre-materialise an fp32 copy. This both matches TRiP and
/// avoids a pointless 2× memory blowup for Gemma's bf16 weights.
///
/// Accumulation order intentionally mirrors TRiP/math.c rmsnorm so that
/// a later parity test against the C reference compares bit-close at
/// fp32. Specifically: rms is summed sequentially in the input order;
/// the final per-element formula is `(gain * rms_inv) * in[i]` with no
/// intermediate that TRiP doesn't also compute.
pub fn rmsnorm(
    out: []f32,
    in: []const f32,
    weight: Tensor,
    eps: f32,
    family: Family,
) !void {
    const dim = in.len;
    if (out.len != dim) return error.LengthMismatch;
    if (weight.shape.len != 1 or weight.shape[0] != dim) return error.WeightShapeMismatch;

    // ── rms = sqrt(mean(x²) + eps) ─────────────────────────────────
    var sum_sq: f32 = 0.0;
    for (in) |x| sum_sq += x * x;
    const mean_sq = sum_sq / @as(f32, @floatFromInt(dim));
    const rms_inv = 1.0 / @sqrt(mean_sq + eps);

    const apply_gemma_quirk = family == .gemma;

    // ── per-element scale ──────────────────────────────────────────
    switch (weight.dtype) {
        .f32 => {
            const w = @as([*]align(1) const f32, @ptrCast(weight.bytes.ptr))[0..dim];
            if (apply_gemma_quirk) {
                for (out, in, w) |*o, x, wi| o.* = (1.0 + wi) * rms_inv * x;
            } else {
                for (out, in, w) |*o, x, wi| o.* = wi * rms_inv * x;
            }
        },
        .bf16 => {
            const w_u16 = dtype.asU16(weight.bytes);
            if (apply_gemma_quirk) {
                for (out, in, w_u16) |*o, x, wb| {
                    const wi = dtype.bf16ToF32(wb);
                    o.* = (1.0 + wi) * rms_inv * x;
                }
            } else {
                for (out, in, w_u16) |*o, x, wb| {
                    const wi = dtype.bf16ToF32(wb);
                    o.* = wi * rms_inv * x;
                }
            }
        },
        .f16 => {
            const w_u16 = dtype.asU16(weight.bytes);
            if (apply_gemma_quirk) {
                for (out, in, w_u16) |*o, x, wh| {
                    const wi = dtype.f16ToF32(wh);
                    o.* = (1.0 + wi) * rms_inv * x;
                }
            } else {
                for (out, in, w_u16) |*o, x, wh| {
                    const wi = dtype.f16ToF32(wh);
                    o.* = wi * rms_inv * x;
                }
            }
        },
        else => return error.UnsupportedWeightDtype,
    }
}

/// Materialise one row of an [N, D] tensor into `dst` as fp32.
/// `dst.len == D`. Convenience for the per-token embedding lookup that
/// kicks off every forward pass — could live in a more general "tensor
/// helper" module later, but it's tiny and only called from one place.
pub fn embedRowAsF32(dst: []f32, tensor: Tensor, row_idx: usize) !void {
    if (tensor.shape.len != 2) return error.NotMatrix;
    const d = tensor.shape[1];
    if (dst.len != d) return error.LengthMismatch;
    if (row_idx >= tensor.shape[0]) return error.RowOutOfRange;

    const row_bytes_len = d * tensor.dtype.elemSize();
    const row_start = row_idx * row_bytes_len;
    const row = tensor.bytes[row_start .. row_start + row_bytes_len];

    switch (tensor.dtype) {
        .f32 => {
            const src = @as([*]align(1) const f32, @ptrCast(row.ptr))[0..d];
            @memcpy(dst, src);
        },
        .bf16 => dtype.bf16SliceToF32(dtype.asU16(row), dst),
        .f16 => dtype.f16SliceToF32(dtype.asU16(row), dst),
        else => return error.UnsupportedTensorDtype,
    }
}
