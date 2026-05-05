//! CPU reference for transformer-primitive training.
//!
//! Sister to `cpu/train.zig`, holding forward/backward oracles for the
//! transformer building blocks the next training tier needs. Same
//! conventions as `cpu/math.zig`:
//!   - Sequential scalar fp32 accumulation; no SIMD, no Kahan.
//!   - Slice-typed APIs; caller owns all buffers.
//!   - Row-major; per-row operations cleanly factored.
//!
//! First inhabitant: RMSNorm. Forward already lives in
//! `shaders/rmsnorm.comp` and `cpu/forward.zig`'s inference path; we
//! restate it here in oracle form (no fused kernel quirks, just the
//! math) so the backward derivation has a clean reference to diff
//! against.
//!
//! Math reminder. Forward, single row of dim D:
//!     s       = Σⱼ xⱼ²
//!     m       = s / D
//!     ri      = 1 / √(m + ε)            // rms_inv
//!     γᵢ      = wᵢ                      // (or wᵢ + 1 if gemma_quirk)
//!     yᵢ      = γᵢ · ri · xᵢ
//!
//! Backward — given dL/dy, returns dL/dx and dL/dw:
//!
//!     ∂yⱼ/∂wᵢ = δᵢⱼ · ri · xᵢ
//!     dL/dwᵢ  = (dL/dyᵢ) · ri · xᵢ
//!
//!     A       = Σⱼ (dL/dyⱼ) · γⱼ · xⱼ          // scalar across row
//!     dL/dxᵢ  = ri · (γᵢ · (dL/dyᵢ) − xᵢ · ri² · A / D)
//!             = ri · γᵢ · (dL/dyᵢ) − (ri³ · xᵢ / D) · A
//!
//! The gemma_quirk offset (γ = w + 1) drops out of the gradient — the
//! derivative of (w + 1) w.r.t. w is 1, same as plain w. Only the
//! forward `γ` value changes.

const std = @import("std");

/// Forward pass on a single row. Writes `y[i] = γᵢ · ri · xᵢ` and
/// returns `ri` so the caller can stash it for backward (or recompute
/// — backward does not require it as input, by design).
pub fn rmsNormForwardRow(
    x: []const f32,
    w: []const f32,
    eps: f32,
    gemma_quirk: bool,
    y: []f32,
) f32 {
    std.debug.assert(x.len == w.len);
    std.debug.assert(y.len == x.len);
    var s: f64 = 0;
    for (x) |v| s += @as(f64, v) * @as(f64, v);
    const m: f64 = s / @as(f64, @floatFromInt(x.len));
    const ri: f32 = 1.0 / @sqrt(@as(f32, @floatCast(m)) + eps);
    for (x, w, y) |xi, wi, *yi| {
        const gain = if (gemma_quirk) wi + 1.0 else wi;
        yi.* = gain * ri * xi;
    }
    return ri;
}

/// Backward pass on a single row. Caller passes the upstream gradient
/// `dy`, the original `x` and `w`, and writes `dx` and accumulates
/// into `dw` (assumed pre-zeroed by the caller). `dw` is *accumulated*
/// rather than overwritten so multi-row dgain falls out trivially —
/// each row contributes its dgain term and they sum.
///
/// `eps` and `gemma_quirk` must match the forward pass exactly.
pub fn rmsNormBackwardRow(
    dy: []const f32,
    x: []const f32,
    w: []const f32,
    eps: f32,
    gemma_quirk: bool,
    dx: []f32,
    dw_accum: []f32,
) void {
    const D = x.len;
    std.debug.assert(dy.len == D);
    std.debug.assert(w.len == D);
    std.debug.assert(dx.len == D);
    std.debug.assert(dw_accum.len == D);

    // Recompute rms_inv from x. We deliberately don't take `ri` as
    // input: backward should be self-sufficient (matches the GPU
    // shader contract — caller passes raw activations, kernel
    // recomputes the reduction). For perf this could cache `ri` per
    // row in a forward-train variant; chunk-2 question.
    var s: f64 = 0;
    for (x) |v| s += @as(f64, v) * @as(f64, v);
    const m: f64 = s / @as(f64, @floatFromInt(D));
    const ri: f32 = 1.0 / @sqrt(@as(f32, @floatCast(m)) + eps);

    // Cross-row scalar reduction A = Σⱼ dyⱼ · γⱼ · xⱼ (fp64 accumulation
    // for parity stability against the GPU's wider reduction tree).
    var A: f64 = 0;
    for (dy, x, w) |dyi, xi, wi| {
        const gain = if (gemma_quirk) wi + 1.0 else wi;
        A += @as(f64, dyi) * @as(f64, gain) * @as(f64, xi);
    }

    const ri3_over_D: f32 = ri * ri * ri / @as(f32, @floatFromInt(D));
    const A_f32: f32 = @floatCast(A);

    // dL/dxᵢ = ri · γᵢ · dyᵢ − (ri³ / D) · xᵢ · A
    // dL/dwᵢ += dyᵢ · ri · xᵢ
    for (0..D) |i| {
        const gain: f32 = if (gemma_quirk) w[i] + 1.0 else w[i];
        dx[i] = ri * gain * dy[i] - ri3_over_D * x[i] * A_f32;
        dw_accum[i] += dy[i] * ri * x[i];
    }
}

/// Multi-row convenience wrapper. Forward writes `y[r, :]` for r=0..n_rows.
/// Caller supplies `y` of shape [n_rows, dim]. Returns nothing — caller
/// re-runs `rmsNormForwardRow` for ri values per row if backward needs them.
pub fn rmsNormForward(
    x: []const f32,
    w: []const f32,
    eps: f32,
    gemma_quirk: bool,
    n_rows: usize,
    y: []f32,
) void {
    const dim = w.len;
    std.debug.assert(x.len == n_rows * dim);
    std.debug.assert(y.len == n_rows * dim);
    for (0..n_rows) |r| {
        const off = r * dim;
        _ = rmsNormForwardRow(x[off .. off + dim], w, eps, gemma_quirk, y[off .. off + dim]);
    }
}

/// Multi-row backward. `dw` accumulates across rows (the natural
/// "gain is shared across the sequence" convention); caller is
/// responsible for zeroing `dw` before the call if they want a fresh
/// accumulation. `dx` is per-row: each row's dx is independent.
pub fn rmsNormBackward(
    dy: []const f32,
    x: []const f32,
    w: []const f32,
    eps: f32,
    gemma_quirk: bool,
    n_rows: usize,
    dx: []f32,
    dw: []f32,
) void {
    const dim = w.len;
    std.debug.assert(dy.len == n_rows * dim);
    std.debug.assert(x.len == n_rows * dim);
    std.debug.assert(dx.len == n_rows * dim);
    std.debug.assert(dw.len == dim);
    for (0..n_rows) |r| {
        const off = r * dim;
        rmsNormBackwardRow(
            dy[off .. off + dim],
            x[off .. off + dim],
            w,
            eps,
            gemma_quirk,
            dx[off .. off + dim],
            dw,
        );
    }
}
