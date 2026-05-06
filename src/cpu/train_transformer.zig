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

// ── Softmax backward ───────────────────────────────────────────────
//
// Given the softmax forward yᵢ = exp(xᵢ − max) / Σ exp(xⱼ − max), the
// Jacobian is ∂yⱼ/∂xᵢ = yⱼ (δᵢⱼ − yᵢ), and the backward step
//
//     dxᵢ = Σⱼ dyⱼ · ∂yⱼ/∂xᵢ
//         = dyᵢ · yᵢ − yᵢ · Σⱼ dyⱼ · yⱼ
//         = yᵢ · (dyᵢ − ⟨dy, y⟩)
//
// reduces to a single scalar reduction `S = ⟨dy, y⟩` per row, then a
// per-element write `dxᵢ = yᵢ · (dyᵢ − S)`. Foundation for attention
// backward, where the softmax sits at the centre of the chain.
//
// Saved-activation is `y` (not `x`): forward already computed y, and
// y is sufficient — no need to re-do max/sum/exp.

/// Single-row softmax backward.
pub fn softmaxBackwardRow(
    dy: []const f32,
    y: []const f32,
    dx: []f32,
) void {
    std.debug.assert(dy.len == y.len);
    std.debug.assert(dx.len == y.len);
    var S: f64 = 0;
    for (dy, y) |d, yi| S += @as(f64, d) * @as(f64, yi);
    const S_f: f32 = @floatCast(S);
    for (dy, y, dx) |d, yi, *dxi| dxi.* = yi * (d - S_f);
}

/// Multi-row softmax backward. Per-row independent — no cross-row
/// state to manage.
pub fn softmaxBackward(
    dy: []const f32,
    y: []const f32,
    n_rows: usize,
    dim: usize,
    dx: []f32,
) void {
    std.debug.assert(dy.len == n_rows * dim);
    std.debug.assert(y.len == n_rows * dim);
    std.debug.assert(dx.len == n_rows * dim);
    for (0..n_rows) |r| {
        const off = r * dim;
        softmaxBackwardRow(
            dy[off .. off + dim],
            y[off .. off + dim],
            dx[off .. off + dim],
        );
    }
}

// ── RoPE backward ──────────────────────────────────────────────────
//
// Forward applies a 2D rotation by +θ to each (j, j+R/2) pair within
// the first `rotary_dim = R` channels of every head:
//     out_j         = aⱼ · cos − bⱼ · sin
//     out_{j+R/2}   = aⱼ · sin + bⱼ · cos
// where (aⱼ, bⱼ) = (in_j, in_{j+R/2}). The Jacobian is the rotation
// matrix [[c, −s], [s, c]]; backward is its transpose, i.e. rotation
// by −θ:
//     d_in_j        = cos · d_out_j     + sin · d_out_{j+R/2}
//     d_in_{j+R/2}  = −sin · d_out_j    + cos · d_out_{j+R/2}
//
// Pass-through tail (channels i ≥ rotary_dim) is identity in forward,
// so backward is also identity: d_in_i = d_out_i.
//
// No parameters — RoPE has no learnable weights.

pub fn ropeBackwardPartial(
    d_in: []f32,
    d_out: []const f32,
    n_heads: usize,
    head_dim: usize,
    rotary_dim: usize,
    pos: usize,
    theta_base: f32,
) !void {
    const total = n_heads * head_dim;
    if (d_in.len != total or d_out.len != total) return error.LengthMismatch;
    if (rotary_dim > head_dim) return error.RotaryDimTooLarge;
    if (rotary_dim % 2 != 0) return error.OddRotaryDim;
    const half = rotary_dim / 2;

    const pos_f: f32 = @floatFromInt(pos);
    const rdim_f: f32 = @floatFromInt(rotary_dim);

    for (0..n_heads) |h| {
        const off = h * head_dim;
        for (0..half) |j| {
            const freq = 1.0 / std.math.pow(f32, theta_base, (2.0 * @as(f32, @floatFromInt(j))) / rdim_f);
            const angle = pos_f * freq;
            const cos_a = @cos(angle);
            const sin_a = @sin(angle);
            const da = d_out[off + j];
            const db = d_out[off + j + half];
            d_in[off + j] = cos_a * da + sin_a * db;
            d_in[off + j + half] = -sin_a * da + cos_a * db;
        }
        if (rotary_dim < head_dim) {
            for (rotary_dim..head_dim) |i| d_in[off + i] = d_out[off + i];
        }
    }
}

pub fn ropeBackward(
    d_in: []f32,
    d_out: []const f32,
    n_heads: usize,
    head_dim: usize,
    pos: usize,
    theta_base: f32,
) !void {
    return ropeBackwardPartial(d_in, d_out, n_heads, head_dim, head_dim, pos, theta_base);
}

/// Batched RoPE forward over `n_pos` rows of `[n_heads, head_dim]`.
/// Position for row `p` is just `p`. Setting `rotary_dim = head_dim`
/// gives full RoPE. Sister to `cpu_math.applyRope` but loops the
/// position outside, matching the GPU `rope_partial_batched.comp`
/// dispatch shape.
pub fn ropeForwardBatched(
    out: []f32,
    in: []const f32,
    n_pos: usize,
    n_heads: usize,
    head_dim: usize,
    rotary_dim: usize,
    theta_base: f32,
) !void {
    const row_stride = n_heads * head_dim;
    if (in.len != n_pos * row_stride or out.len != n_pos * row_stride) return error.LengthMismatch;
    if (rotary_dim > head_dim) return error.RotaryDimTooLarge;
    if (rotary_dim % 2 != 0) return error.OddRotaryDim;
    const half = rotary_dim / 2;
    const rdim_f: f32 = @floatFromInt(rotary_dim);
    for (0..n_pos) |p| {
        const row_off = p * row_stride;
        const pos_f: f32 = @floatFromInt(p);
        for (0..n_heads) |h| {
            const off = row_off + h * head_dim;
            for (0..half) |j| {
                const freq = 1.0 / std.math.pow(f32, theta_base, (2.0 * @as(f32, @floatFromInt(j))) / rdim_f);
                const angle = pos_f * freq;
                const cos_a = @cos(angle);
                const sin_a = @sin(angle);
                const a = in[off + j];
                const b = in[off + j + half];
                out[off + j] = a * cos_a - b * sin_a;
                out[off + j + half] = a * sin_a + b * cos_a;
            }
            if (rotary_dim < head_dim) {
                for (rotary_dim..head_dim) |i| out[off + i] = in[off + i];
            }
        }
    }
}

/// Batched RoPE backward over `n_pos` rows of `[n_heads, head_dim]`.
/// Mirrors the forward; output `d_in` is *overwritten* (not accumulated).
pub fn ropeBackwardBatched(
    d_in: []f32,
    d_out: []const f32,
    n_pos: usize,
    n_heads: usize,
    head_dim: usize,
    rotary_dim: usize,
    theta_base: f32,
) !void {
    const row_stride = n_heads * head_dim;
    if (d_in.len != n_pos * row_stride or d_out.len != n_pos * row_stride) return error.LengthMismatch;
    if (rotary_dim > head_dim) return error.RotaryDimTooLarge;
    if (rotary_dim % 2 != 0) return error.OddRotaryDim;
    const half = rotary_dim / 2;
    const rdim_f: f32 = @floatFromInt(rotary_dim);
    for (0..n_pos) |p| {
        const row_off = p * row_stride;
        const pos_f: f32 = @floatFromInt(p);
        for (0..n_heads) |h| {
            const off = row_off + h * head_dim;
            for (0..half) |j| {
                const freq = 1.0 / std.math.pow(f32, theta_base, (2.0 * @as(f32, @floatFromInt(j))) / rdim_f);
                const angle = pos_f * freq;
                const cos_a = @cos(angle);
                const sin_a = @sin(angle);
                const da = d_out[off + j];
                const db = d_out[off + j + half];
                d_in[off + j] = cos_a * da + sin_a * db;
                d_in[off + j + half] = -sin_a * da + cos_a * db;
            }
            if (rotary_dim < head_dim) {
                for (rotary_dim..head_dim) |i| d_in[off + i] = d_out[off + i];
            }
        }
    }
}

// ── Scaled-dot-product attention forward + backward ────────────────
//
// Generic shape covers both training-style multi-query and decode-style
// single-query attention:
//
//     Q       [n_q, n_heads, head_dim]            row-major
//     K       [n_kv, n_kv_heads, head_dim]
//     V       [n_kv, n_kv_heads, head_dim]
//     scores  [n_q, n_heads, n_kv]                pre-softmax
//     attn    [n_q, n_heads, n_kv]                post-softmax (saved for bwd)
//     out     [n_q, n_heads, head_dim]
//
// GQA fold: head h reads K/V from `kv_h(h) = h / heads_per_kv` where
// `heads_per_kv = n_heads / n_kv_heads`.
//
// Causal mask (when `causal = true`): for query at position q (0-indexed
// from the start of the local window), keys at positions k > q are
// masked out — forward sets `attn[q, h, k > q] = 0`. n_q ≤ n_kv is
// assumed for causal attention with q's window aligned to the *end* of
// the K/V window (the typical "decode against full history" layout):
// query position q corresponds to key position `q + (n_kv − n_q)`,
// keys at indices > `q + (n_kv − n_q)` are masked.
//
// For decode-step (n_q = 1, n_kv = pos+1) with causal = true, this
// reduces to "no mask at all" (every key index is ≤ the single query's
// position). For training-step (n_q = n_kv, both = sequence length) it
// reduces to the standard lower-triangular causal mask.
//
// Forward:
//     scores[q, h, k] = (Q[q, h] · K[k, kv_h(h)]) * inv_sqrt_d
//                       (or -inf if causally masked)
//     attn[q, h, :]   = softmax(scores[q, h, :])
//     out[q, h, d]    = Σ_k attn[q, h, k] · V[k, kv_h(h), d]
//
// Backward (saved activations: attn, Q, K, V):
//     dV[k, kv_h, d] = Σ_{h: kv_h(h)=kv_h} Σ_q attn[q, h, k] · dout[q, h, d]
//     d_attn[q, h, k] = Σ_d dout[q, h, d] · V[k, kv_h(h), d]
//     d_scores[q, h, :] = softmax_backward(d_attn[q, h, :], attn[q, h, :])
//     dQ[q, h, d]   = inv_sqrt_d · Σ_k d_scores[q, h, k] · K[k, kv_h(h), d]
//     dK[k, kv_h, d] = inv_sqrt_d · Σ_{h: kv_h(h)=kv_h} Σ_q
//                                d_scores[q, h, k] · Q[q, h, d]
//
// Causal mask handling in backward: masked entries have attn = 0 → no
// dV contribution; softmax_backward yields d_scores = 0 at masked
// entries (since y_i = 0 means dx_i = 0); so dQ/dK pick up zeros from
// those positions naturally — no extra masking logic needed in the
// backward kernels themselves, as long as forward produced a clean
// zero on the attn row for masked keys.

inline fn causalKeyLimit(q: usize, n_q: usize, n_kv: usize) usize {
    // Query q's "absolute" position is q + (n_kv − n_q); inclusive limit.
    return q + (n_kv - n_q);
}

/// Full attention forward. Writes `scores`, `attn`, and `out`. The
/// caller is expected to keep `attn` for backward (or re-run forward).
pub fn attentionForward(
    Q: []const f32,
    K: []const f32,
    V: []const f32,
    n_q: usize,
    n_kv: usize,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    causal: bool,
    scores: []f32,
    attn: []f32,
    out: []f32,
) void {
    std.debug.assert(n_heads % n_kv_heads == 0);
    const heads_per_kv = n_heads / n_kv_heads;
    std.debug.assert(Q.len == n_q * n_heads * head_dim);
    std.debug.assert(K.len == n_kv * n_kv_heads * head_dim);
    std.debug.assert(V.len == n_kv * n_kv_heads * head_dim);
    std.debug.assert(scores.len == n_q * n_heads * n_kv);
    std.debug.assert(attn.len == n_q * n_heads * n_kv);
    std.debug.assert(out.len == n_q * n_heads * head_dim);
    if (causal) std.debug.assert(n_q <= n_kv);

    const inv_sqrt_d: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(head_dim)));

    for (0..n_q) |q| {
        const k_limit: usize = if (causal) causalKeyLimit(q, n_q, n_kv) else (n_kv - 1);
        for (0..n_heads) |h| {
            const kv_h = h / heads_per_kv;
            const q_off = q * n_heads * head_dim + h * head_dim;
            const s_row_off = q * n_heads * n_kv + h * n_kv;

            // ── 1. scores[q, h, k] = Q · K * inv_sqrt_d  (or -inf if masked)
            for (0..n_kv) |k| {
                if (causal and k > k_limit) {
                    scores[s_row_off + k] = -std.math.inf(f32);
                    continue;
                }
                const k_off = k * n_kv_heads * head_dim + kv_h * head_dim;
                var s: f64 = 0;
                for (0..head_dim) |d| {
                    s += @as(f64, Q[q_off + d]) * @as(f64, K[k_off + d]);
                }
                scores[s_row_off + k] = @as(f32, @floatCast(s)) * inv_sqrt_d;
            }

            // ── 2. attn[q, h, :] = softmax(scores[q, h, :])  (stable)
            var max_s: f32 = -std.math.inf(f32);
            for (0..n_kv) |k| {
                const v = scores[s_row_off + k];
                if (v > max_s) max_s = v;
            }
            var sum: f64 = 0;
            for (0..n_kv) |k| {
                const v = scores[s_row_off + k];
                const e: f32 = if (std.math.isInf(v) and v < 0) 0.0 else @exp(v - max_s);
                attn[s_row_off + k] = e;
                sum += e;
            }
            const inv_sum: f32 = if (sum > 0) 1.0 / @as(f32, @floatCast(sum)) else 0.0;
            for (0..n_kv) |k| attn[s_row_off + k] *= inv_sum;

            // ── 3. out[q, h, d] = Σ_k attn · V
            const o_off = q * n_heads * head_dim + h * head_dim;
            for (0..head_dim) |d| out[o_off + d] = 0.0;
            for (0..n_kv) |k| {
                const v_off = k * n_kv_heads * head_dim + kv_h * head_dim;
                const w = attn[s_row_off + k];
                if (w == 0.0) continue;
                for (0..head_dim) |d| out[o_off + d] += w * V[v_off + d];
            }
        }
    }
}

/// Full attention backward. `attn` is the saved softmax output from
/// forward. `dQ`, `dK`, `dV` are *overwritten* (not accumulated) — the
/// caller is responsible for any cross-step accumulation. `d_scores`
/// is a scratch buffer the same shape as `scores` — reused as the
/// staging area for `softmax_backward`.
pub fn attentionBackward(
    d_out: []const f32,
    Q: []const f32,
    K: []const f32,
    V: []const f32,
    attn: []const f32,
    n_q: usize,
    n_kv: usize,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    causal: bool,
    d_scores: []f32,
    dQ: []f32,
    dK: []f32,
    dV: []f32,
) void {
    std.debug.assert(n_heads % n_kv_heads == 0);
    _ = causal; // mask is implicit in `attn` (zeros at masked entries)
    const heads_per_kv = n_heads / n_kv_heads;
    std.debug.assert(d_out.len == n_q * n_heads * head_dim);
    std.debug.assert(Q.len == n_q * n_heads * head_dim);
    std.debug.assert(K.len == n_kv * n_kv_heads * head_dim);
    std.debug.assert(V.len == n_kv * n_kv_heads * head_dim);
    std.debug.assert(attn.len == n_q * n_heads * n_kv);
    std.debug.assert(d_scores.len == n_q * n_heads * n_kv);
    std.debug.assert(dQ.len == n_q * n_heads * head_dim);
    std.debug.assert(dK.len == n_kv * n_kv_heads * head_dim);
    std.debug.assert(dV.len == n_kv * n_kv_heads * head_dim);

    const inv_sqrt_d: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(head_dim)));

    @memset(dQ, 0.0);
    @memset(dK, 0.0);
    @memset(dV, 0.0);

    // ── 1. dV[k, kv_h, d] = Σ_h Σ_q attn[q, h, k] · dout[q, h, d]
    //
    //     and d_attn[q, h, k] = Σ_d dout[q, h, d] · V[k, kv_h(h), d]
    //
    // Fused into one pass over (q, h, k, d) for cache behaviour, fp64
    // accumulation on the d-axis sum to match the row-wide reduction
    // convention used elsewhere.
    for (0..n_q) |q| {
        for (0..n_heads) |h| {
            const kv_h = h / heads_per_kv;
            const dout_off = q * n_heads * head_dim + h * head_dim;
            const a_row_off = q * n_heads * n_kv + h * n_kv;
            const ds_row_off = a_row_off;

            for (0..n_kv) |k| {
                const v_off = k * n_kv_heads * head_dim + kv_h * head_dim;
                const a = attn[a_row_off + k];

                // dV accumulation — only contributes when attn != 0.
                if (a != 0.0) {
                    for (0..head_dim) |d| {
                        dV[v_off + d] += a * d_out[dout_off + d];
                    }
                }

                // d_attn[q, h, k] (staged into d_scores buffer).
                var dot: f64 = 0;
                for (0..head_dim) |d| {
                    dot += @as(f64, d_out[dout_off + d]) * @as(f64, V[v_off + d]);
                }
                d_scores[ds_row_off + k] = @floatCast(dot);
            }
        }
    }

    // ── 2. d_scores[q, h, :] = softmax_backward(d_attn, attn)
    softmaxBackward(d_scores, attn, n_q * n_heads, n_kv, d_scores);

    // ── 3. dQ[q, h, d] = inv_sqrt_d · Σ_k d_scores · K
    //      dK[k, kv_h, d] += inv_sqrt_d · Σ_h Σ_q d_scores · Q
    for (0..n_q) |q| {
        for (0..n_heads) |h| {
            const kv_h = h / heads_per_kv;
            const q_off = q * n_heads * head_dim + h * head_dim;
            const ds_row_off = q * n_heads * n_kv + h * n_kv;

            for (0..n_kv) |k| {
                const k_off = k * n_kv_heads * head_dim + kv_h * head_dim;
                const ds = d_scores[ds_row_off + k];
                if (ds == 0.0) continue;
                const ds_scaled = ds * inv_sqrt_d;
                for (0..head_dim) |d| {
                    dQ[q_off + d] += ds_scaled * K[k_off + d];
                    dK[k_off + d] += ds_scaled * Q[q_off + d];
                }
            }
        }
    }
}

// ── SwiGLU FFN ────────────────────────────────────────────────────────
//
// SwiGLU is the FFN used by Llama / Qwen / Mistral. Replaces the toy
// stack's FF1 → ReLU → FF2 with a gated path:
//
//     gate     = silu(x @ W_gate^T)         silu(z) = z · σ(z)
//     up       = x @ W_up^T
//     gated    = gate · up                  (elementwise)
//     y        = gated @ W_down^T
//
// Three weight tensors instead of two; the gate's nonlinearity is
// SiLU (a.k.a. swish-1). The matmuls themselves reuse the existing
// linear primitive — only the SwiGLU non-linearity (silu plus the
// gate·up multiply) needs new oracle code.
//
// Backward, given d_gated = ∂L/∂gated (the gradient flowing down from
// the W_down dx step):
//
//     d_up         = d_gated · gate
//     d_gate       = d_gated · up
//     d_pre_gate_i = d_gate_i · silu'(pre_gate_i)
//
// where silu'(z) = σ(z) · (1 + z·(1 − σ(z))) — derived via product
// rule from silu(z) = z · σ(z) and σ'(z) = σ(z)·(1 − σ(z)).
//
// Saved activations: `pre_gate` and `up` (sufficient — `gate` and
// `gated` are recomputed in the kernels rather than stored, since
// silu is cheap and the alternative inflates per-layer activation
// memory). The fused-shader implementation will follow the same
// recompute strategy on the GPU.
//
// fp64 accumulator is unnecessary here: every operation is per-element
// — no cross-element reduction.

fn sigmoidf(z: f32) f32 {
    return 1.0 / (1.0 + @exp(-z));
}

/// SwiGLU forward: `gated[i] = silu(pre_gate[i]) · up[i]`. All three
/// slices must have the same length. `gated` is the input that the
/// W_down matmul consumes.
pub fn swigluForward(
    pre_gate: []const f32,
    up: []const f32,
    gated: []f32,
) void {
    std.debug.assert(pre_gate.len == up.len);
    std.debug.assert(gated.len == pre_gate.len);
    for (pre_gate, up, gated) |z, u, *g| {
        const sig = sigmoidf(z);
        const silu_z = z * sig;
        g.* = silu_z * u;
    }
}

/// SwiGLU backward. Given the upstream gradient `d_gated` and the saved
/// pre_gate / up activations, writes the gradients into `d_pre_gate`
/// and `d_up`. Outputs *overwrite* (not accumulate) — caller zeroes if
/// it needs accumulation.
pub fn swigluBackward(
    d_gated: []const f32,
    pre_gate: []const f32,
    up: []const f32,
    d_pre_gate: []f32,
    d_up: []f32,
) void {
    std.debug.assert(d_gated.len == pre_gate.len);
    std.debug.assert(up.len == pre_gate.len);
    std.debug.assert(d_pre_gate.len == pre_gate.len);
    std.debug.assert(d_up.len == pre_gate.len);
    for (d_gated, pre_gate, up, d_pre_gate, d_up) |dg, z, u, *dpg, *du| {
        const sig = sigmoidf(z);
        const silu_z = z * sig;
        const silu_grad = sig + silu_z * (1.0 - sig);
        du.* = dg * silu_z;
        dpg.* = dg * u * silu_grad;
    }
}
