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
///          = weight[i]         if family == .llama or .qwen3
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

    const apply_gemma_quirk = family.rmsnormAddOne();

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

/// Qwen3 per-head q_norm / k_norm: applies an independent RMSNorm to
/// each `head_dim`-sized slice of `in`, all sharing the same gain vector
/// `weight` (shape [head_dim]).
///
/// The mean-of-squares is computed PER HEAD (not across the whole vector
/// — that would be plain rmsnorm). Each head normalises against its own
/// L2; the gain vector is shared across heads because each head is
/// considered an independent unit-variance subspace.
///
/// `family` is forwarded so the existing `(1+w)` Gemma quirk handling
/// stays consistent — but in practice this is only ever called for
/// Qwen3 (no quirk), so the gemma branch is dead code. Kept for API
/// symmetry with `rmsnorm`.
pub fn rmsnormPerHead(
    out: []f32,
    in: []const f32,
    weight: Tensor,
    eps: f32,
    num_heads: usize,
    head_dim: usize,
    family: Family,
) !void {
    if (in.len != num_heads * head_dim) return error.LengthMismatch;
    if (out.len != in.len) return error.LengthMismatch;
    if (weight.shape.len != 1 or weight.shape[0] != head_dim) return error.WeightShapeMismatch;

    for (0..num_heads) |h| {
        const off = h * head_dim;
        try rmsnorm(out[off .. off + head_dim], in[off .. off + head_dim], weight, eps, family);
    }
}

/// out[i, j] = sum_k a[i, k] * b[j, k]
/// Shapes: a is [M, K] row-major, b is [N, K] row-major (HuggingFace
/// linear-layer storage), out is [M, N] row-major.
///
/// "nt" because b is read as if transposed: each output column j comes
/// from row j of b. This is the canonical layout for transformer linear
/// layers — `y = W·x` with W in HF's [out_dim, in_dim] becomes a single
/// matmul_nt call with no extra copy.
///
/// Inner accumulation order matches TRiP/math.c matmulf_nt (sequential
/// over k, scalar fp32 add) so the future C-vs-Zig parity test compares
/// bit-close. Performance is deliberately not optimised; this is the
/// correctness oracle, with the GPU kernel being where speed lives.
pub fn matmul_nt(
    out: []f32,
    a: []const f32,
    b: Tensor,
    m: usize,
    n: usize,
    k: usize,
) !void {
    if (a.len != m * k) return error.LengthMismatch;
    if (out.len != m * n) return error.LengthMismatch;
    if (b.shape.len != 2 or b.shape[0] != n or b.shape[1] != k) return error.WeightShapeMismatch;

    switch (b.dtype) {
        .f32 => {
            const bf = @as([*]align(1) const f32, @ptrCast(b.bytes.ptr))[0 .. n * k];
            for (0..m) |i| {
                const a_row_off = i * k;
                for (0..n) |j| {
                    const b_row_off = j * k;
                    var acc: f32 = 0;
                    for (0..k) |kk| acc += a[a_row_off + kk] * bf[b_row_off + kk];
                    out[i * n + j] = acc;
                }
            }
        },
        .bf16 => {
            const bu = dtype.asU16(b.bytes);
            for (0..m) |i| {
                const a_row_off = i * k;
                for (0..n) |j| {
                    const b_row_off = j * k;
                    var acc: f32 = 0;
                    for (0..k) |kk| {
                        const bk = dtype.bf16ToF32(bu[b_row_off + kk]);
                        acc += a[a_row_off + kk] * bk;
                    }
                    out[i * n + j] = acc;
                }
            }
        },
        .f16 => {
            const bu = dtype.asU16(b.bytes);
            for (0..m) |i| {
                const a_row_off = i * k;
                for (0..n) |j| {
                    const b_row_off = j * k;
                    var acc: f32 = 0;
                    for (0..k) |kk| {
                        const bk = dtype.f16ToF32(bu[b_row_off + kk]);
                        acc += a[a_row_off + kk] * bk;
                    }
                    out[i * n + j] = acc;
                }
            }
        },
        else => return error.UnsupportedWeightDtype,
    }
}

/// Apply rotary positional embeddings (RoPE), HuggingFace half-split
/// convention: pairs are `(j, j + head_dim/2)` for j in [0, head_dim/2).
/// `qk` carries `n_heads` heads of width `head_dim` laid out as one
/// flat slice `[n_heads * head_dim]`; we rotate each head independently
/// so the same routine serves both Q (n_heads = num_attention_heads)
/// and K (n_heads = num_key_value_heads under MQA/GQA).
///
/// Convention choice notes: TRiP uses pairwise-interleaved RoPE and a
/// custom half-split→pairwise permuting matmul (math.c
/// matmulf_nt_interleaved) so its in-memory Q has pairwise layout.
/// We don't do that — our Q is in the HF half-split layout direct from
/// the standard matmul, so the half-split RoPE is the natural match.
/// Final attention scores are identical between the two paths.
///
/// At pos = 0 every angle is zero, so cos = 1, sin = 0, and the output
/// equals the input. That's our sanity check in the smoke test.
pub fn applyRope(
    out: []f32,
    in: []const f32,
    n_heads: usize,
    head_dim: usize,
    pos: usize,
    theta_base: f32,
) !void {
    return applyRopePartial(out, in, n_heads, head_dim, head_dim, pos, theta_base);
}

/// Partial-rotation RoPE: rotate only the first `rotary_dim` of each
/// `head_dim`-sized head; the trailing `head_dim - rotary_dim` entries
/// pass through unchanged. Qwen3.5 uses `rotary_dim = head_dim / 4`
/// (partial_rotary_factor = 0.25). The rotation pattern within
/// `rotary_dim` matches HF's GPT-NeoX form: pair index `j` with index
/// `j + rotary_dim/2`, rotate by `pos * theta_base ^ (-2j/rotary_dim)`.
///
/// Crucially the wavelength denominator is `rotary_dim`, NOT `head_dim`
/// — HF builds its inv_freq table with `dim = int(head_dim *
/// partial_rotary_factor)`. Get that wrong and the cos/sin tables drift
/// by a factor of `partial_rotary_factor` and parity collapses.
pub fn applyRopePartial(
    out: []f32,
    in: []const f32,
    n_heads: usize,
    head_dim: usize,
    rotary_dim: usize,
    pos: usize,
    theta_base: f32,
) !void {
    const total = n_heads * head_dim;
    if (in.len != total or out.len != total) return error.LengthMismatch;
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
            const a = in[off + j];
            const b = in[off + j + half];
            out[off + j] = a * cos_a - b * sin_a;
            out[off + j + half] = a * sin_a + b * cos_a;
        }
        // Pass-through tail.
        if (rotary_dim < head_dim) {
            for (rotary_dim..head_dim) |i| out[off + i] = in[off + i];
        }
    }
}

/// In-place numerically stable softmax over a slice. Subtract the max
/// before exponentiating so the largest exp() argument is 0 and we
/// don't overflow on big positive scores; the resulting probabilities
/// are identical to the naive version in exact arithmetic but tolerate
/// fp32 inputs in [-∞, +∞] without inf/nan.
pub fn softmax(x: []f32) void {
    if (x.len == 0) return;
    var max_v: f32 = x[0];
    for (x[1..]) |v| {
        if (v > max_v) max_v = v;
    }
    var sum: f32 = 0;
    for (x) |*v| {
        const e = @exp(v.* - max_v);
        v.* = e;
        sum += e;
    }
    const inv = 1.0 / sum;
    for (x) |*v| v.* *= inv;
}

/// GELU with the tanh approximation:
///     0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))
///
/// This is the variant Gemma 1 was trained with (TRiP NL_GELU_TANH;
/// HuggingFace `gelu_pytorch_tanh`). Despite Gemma's `hidden_act:
/// "gelu"` config string, the modeling code routes to this approx —
/// using the exact erf form here would silently shift every activation
/// and drift logits.
pub inline fn gelu_tanh(x: f32) f32 {
    const sqrt_2_over_pi: f32 = 0.7978845608028654;
    const c: f32 = 0.044715;
    const inner = sqrt_2_over_pi * (x + c * x * x * x);
    return 0.5 * x * (1.0 + std.math.tanh(inner));
}

/// GeGLU activation for the gated FFN:
///     out[i] = gelu_tanh(gate[i]) * up[i]
///
/// `out` may alias `gate` or `up` for in-place evaluation. Sizes must
/// match. This is the entire "activation" step of Gemma's FFN; the
/// gate/up matmuls and the down matmul live at the call site.
pub fn geglu(out: []f32, gate: []const f32, up: []const f32) !void {
    if (out.len != gate.len or out.len != up.len) return error.LengthMismatch;
    for (out, gate, up) |*o, g, u| o.* = gelu_tanh(g) * u;
}

/// SwiGLU activation for the gated FFN:
///     out[i] = silu(gate[i]) * up[i]
/// where silu(x) = x * sigmoid(x) = x / (1 + exp(-x)).
///
/// This is the Llama / Qwen3 variant. Same shape as GeGLU; only the
/// elementwise nonlinearity differs.
pub fn swiglu(out: []f32, gate: []const f32, up: []const f32) !void {
    if (out.len != gate.len or out.len != up.len) return error.LengthMismatch;
    for (out, gate, up) |*o, g, u| {
        const s = g / (1.0 + std.math.exp(-g));
        o.* = s * u;
    }
}

/// Family-dispatching wrapper: picks GeGLU or SwiGLU based on the
/// declared activation. Call sites stay family-agnostic.
pub fn gatedFfn(out: []f32, gate: []const f32, up: []const f32, family: Family) !void {
    return switch (family.activation()) {
        .gelu => geglu(out, gate, up),
        .silu => swiglu(out, gate, up),
    };
}

/// Materialise one row of an [N, D] tensor into `dst` as fp32.
/// `dst.len == D`. Convenience for the per-token embedding lookup that
/// kicks off every forward pass — could live in a more general "tensor
/// helper" module later, but it's tiny and only called from one place.
/// Materialise an entire tensor as fp32. `dst.len` must equal the
/// tensor's element count. Bf16 / fp16 are bit-pattern-converted; fp32
/// is byte-copied. Used for small Gated-DeltaNet scalars (A_log,
/// dt_bias) that we read once per layer at decode time and operate on
/// in fp32 throughout.
pub fn tensorToF32Slice(dst: []f32, tensor: Tensor) !void {
    var n_elem: usize = 1;
    for (tensor.shape) |d| n_elem *= d;
    if (dst.len != n_elem) return error.LengthMismatch;
    switch (tensor.dtype) {
        .f32 => {
            const src = @as([*]align(1) const f32, @ptrCast(tensor.bytes.ptr))[0..n_elem];
            @memcpy(dst, src);
        },
        .bf16 => dtype.bf16SliceToF32(dtype.asU16(tensor.bytes), dst),
        .f16 => dtype.f16SliceToF32(dtype.asU16(tensor.bytes), dst),
        else => return error.UnsupportedTensorDtype,
    }
}

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
