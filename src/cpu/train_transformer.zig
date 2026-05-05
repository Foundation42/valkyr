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

// ── LayerNorm ──────────────────────────────────────────────────────
//
// LayerNorm forward, single row of dim D:
//     μ        = (1/D) Σⱼ xⱼ
//     zⱼ       = xⱼ − μ
//     v        = (1/D) Σⱼ zⱼ²        (variance)
//     s        = 1 / √(v + ε)        (norm_inv)
//     yᵢ       = γᵢ · s · zᵢ + βᵢ
//
// LayerNorm has both a gain (γ = w) and a bias (β); RMSNorm has gain
// only. The mean subtraction also adds a second cross-row reduction
// in backward — RMSNorm's `A` becomes LN's `(A1, A2)`.
//
// Backward — given dL/dy returns dL/dx, accumulates dL/dw and dL/dβ:
//
//     A1       = Σⱼ dyⱼ · γⱼ            (scalar across row)
//     A2       = Σⱼ dyⱼ · γⱼ · zⱼ       (scalar across row)
//     dL/dxᵢ   = s · (γᵢ · dyᵢ − A1/D − (s² · zᵢ / D) · A2)
//     dL/dwᵢ  += dyᵢ · s · zᵢ          (= dyᵢ · n̂ᵢ)
//     dL/dβᵢ  += dyᵢ                    (scalar identity)
//
// Same fp64 reduction convention as `rmsNormBackwardRow` — the
// forward sums get widened just for the row-wide reductions, then
// the per-element math drops back to fp32.

/// Forward pass on a single row. Returns `s` (norm_inv) so callers
/// caching activations have access; backward recomputes it from x.
pub fn layerNormForwardRow(
    x: []const f32,
    w: []const f32,
    bias: []const f32,
    eps: f32,
    y: []f32,
) f32 {
    std.debug.assert(x.len == w.len);
    std.debug.assert(y.len == x.len);
    std.debug.assert(bias.len == x.len);
    const D: f64 = @floatFromInt(x.len);

    var sum: f64 = 0;
    for (x) |v| sum += v;
    const mu: f64 = sum / D;

    var var_acc: f64 = 0;
    for (x) |v| {
        const z = @as(f64, v) - mu;
        var_acc += z * z;
    }
    const v: f64 = var_acc / D;
    const s: f32 = 1.0 / @sqrt(@as(f32, @floatCast(v)) + eps);

    const mu_f32: f32 = @floatCast(mu);
    for (x, w, bias, y) |xi, wi, bi, *yi| {
        const z = xi - mu_f32;
        yi.* = wi * s * z + bi;
    }
    return s;
}

/// Backward pass on a single row. Same accumulation convention as
/// `rmsNormBackwardRow`: `dw` and `dbias` are accumulated (caller
/// zero-fills); `dx` is overwritten.
pub fn layerNormBackwardRow(
    dy: []const f32,
    x: []const f32,
    w: []const f32,
    eps: f32,
    dx: []f32,
    dw_accum: []f32,
    dbias_accum: []f32,
) void {
    const D = x.len;
    std.debug.assert(dy.len == D);
    std.debug.assert(w.len == D);
    std.debug.assert(dx.len == D);
    std.debug.assert(dw_accum.len == D);
    std.debug.assert(dbias_accum.len == D);
    const D_f: f64 = @floatFromInt(D);

    // Recompute mu, v, s from x.
    var sum: f64 = 0;
    for (x) |v| sum += v;
    const mu: f64 = sum / D_f;
    var var_acc: f64 = 0;
    for (x) |v| {
        const z = @as(f64, v) - mu;
        var_acc += z * z;
    }
    const v_d: f64 = var_acc / D_f;
    const s: f32 = 1.0 / @sqrt(@as(f32, @floatCast(v_d)) + eps);
    const mu_f32: f32 = @floatCast(mu);

    // A1 = Σ dyⱼ · γⱼ        ;   A2 = Σ dyⱼ · γⱼ · zⱼ.
    var A1: f64 = 0;
    var A2: f64 = 0;
    for (dy, x, w) |dyi, xi, wi| {
        const z = @as(f64, xi) - mu;
        const dyg = @as(f64, dyi) * @as(f64, wi);
        A1 += dyg;
        A2 += dyg * z;
    }
    const A1_f: f32 = @floatCast(A1);
    const A2_f: f32 = @floatCast(A2);
    const inv_D: f32 = 1.0 / @as(f32, @floatFromInt(D));
    const s2_over_D: f32 = s * s * inv_D;

    for (0..D) |i| {
        const z = x[i] - mu_f32;
        dx[i] = s * (w[i] * dy[i] - A1_f * inv_D - s2_over_D * z * A2_f);
        dw_accum[i] += dy[i] * s * z;
        dbias_accum[i] += dy[i];
    }
}

/// Multi-row LayerNorm forward. `bias` is broadcast across rows just
/// like `w` — the standard transformer LN convention.
pub fn layerNormForward(
    x: []const f32,
    w: []const f32,
    bias: []const f32,
    eps: f32,
    n_rows: usize,
    y: []f32,
) void {
    const dim = w.len;
    std.debug.assert(bias.len == dim);
    std.debug.assert(x.len == n_rows * dim);
    std.debug.assert(y.len == n_rows * dim);
    for (0..n_rows) |r| {
        const off = r * dim;
        _ = layerNormForwardRow(x[off .. off + dim], w, bias, eps, y[off .. off + dim]);
    }
}

/// Multi-row LayerNorm backward. `dw` and `dbias` accumulated across
/// rows; `dx` per-row.
pub fn layerNormBackward(
    dy: []const f32,
    x: []const f32,
    w: []const f32,
    eps: f32,
    n_rows: usize,
    dx: []f32,
    dw: []f32,
    dbias: []f32,
) void {
    const dim = w.len;
    std.debug.assert(dy.len == n_rows * dim);
    std.debug.assert(x.len == n_rows * dim);
    std.debug.assert(dx.len == n_rows * dim);
    std.debug.assert(dw.len == dim);
    std.debug.assert(dbias.len == dim);
    for (0..n_rows) |r| {
        const off = r * dim;
        layerNormBackwardRow(
            dy[off .. off + dim],
            x[off .. off + dim],
            w,
            eps,
            dx[off .. off + dim],
            dw,
            dbias,
        );
    }
}

// ── Embedding gradient ─────────────────────────────────────────────
//
// Forward (in inference) does `x[t, :] = E[token_ids[t], :]`. Backward
// scatters the per-position upstream gradient back into the embedding
// table at the same indexed rows. Only rows whose vocab id appears in
// `token_ids` get a non-zero contribution; rows not referenced this
// step stay at zero (fresh-zeroed by the caller before the call).
//
// When the same token appears at multiple positions, all those
// position contributions sum into the same `dE[token_id, :]` row.
//
// API mirrors the rmsnorm/layernorm dw_accum pattern: caller
// zero-fills `dE` before the call; we accumulate into it. Lets a
// host stack multiple sequences into one update if it wants.
//
// `dy` shape:        [n_pos, dim]
// `token_ids` shape: [n_pos], values in [0, vocab_size)
// `dE` shape:        [vocab_size, dim]

pub fn embeddingBackward(
    dy: []const f32,
    token_ids: []const u32,
    vocab_size: usize,
    dim: usize,
    dE: []f32,
) void {
    std.debug.assert(dy.len == token_ids.len * dim);
    std.debug.assert(dE.len == vocab_size * dim);
    for (token_ids, 0..) |tok, p| {
        std.debug.assert(tok < vocab_size);
        const src = dy[p * dim .. (p + 1) * dim];
        const dst = dE[@as(usize, tok) * dim .. (@as(usize, tok) + 1) * dim];
        for (dst, src) |*d, s| d.* += s;
    }
}
