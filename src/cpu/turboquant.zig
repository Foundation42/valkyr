//! TurboQuant CPU reference primitives.
//!
//! Constants and helpers for the TurboQuant KV-cache compression path
//! (paper: arXiv:2504.19874, Zandieh et al., ICLR 2026 — Algorithm 1
//! "MSE" branch only; we deliberately skip the QJL residual stage since
//! the practitioner consensus is that it explodes attention-score
//! variance and the Lloyd-Max codebook alone gives lower PPL).
//!
//! The Lloyd-Max centroids and seeded sign pattern here are bit-exact
//! matches to arclabs001/YATQ (`turboquant_wht.py`) and llama.cpp's
//! `cpy-utils.cuh` — that's deliberate, so a future cross-check against
//! either reference is byte-clean instead of "close enough."

const std = @import("std");

// ── Lloyd-Max optimal codebooks for a unit-Gaussian source ──────────
//
// `lm_centroids_b{n}` = the 2^n reconstruction levels.
// `lm_boundaries_b{n}` = the 2^n - 1 decision thresholds.
//
// Bin index for value x is the count of boundaries that x meets or
// exceeds (i.e. `x >= b` for each b). Symmetric about zero.

pub const lm_centroids_b3 = [8]f32{
    -2.1519, -1.3439, -0.7560, -0.2451,
     0.2451,  0.7560,  1.3439,  2.1519,
};

pub const lm_boundaries_b3 = [7]f32{
    -1.7479, -1.0500, -0.5005, 0.0000,
     0.5005,  1.0500,  1.7479,
};

pub const lm_centroids_b4 = [16]f32{
    -2.7326, -2.0690, -1.6180, -1.2562,
    -0.9423, -0.6568, -0.3880, -0.1284,
     0.1284,  0.3880,  0.6568,  0.9423,
     1.2562,  1.6180,  2.0690,  2.7326,
};

pub const lm_boundaries_b4 = [15]f32{
    -2.4008, -1.8435, -1.4371, -1.0993,
    -0.7995, -0.5224, -0.2582, 0.0000,
     0.2582,  0.5224,  0.7995,  1.0993,
     1.4371,  1.8435,  2.4008,
};

// ── Randomized Hadamard transform sign pattern ──────────────────────
//
// 32 bytes = 256 bits = one period of seeded sign flips for the RHT
// pre-conditioning step. Bit i lives at byte (i >> 3), bit position
// (i & 7), LSB-indexed. A set bit means a -1 flip; clear means +1.
// This exact pattern lifts straight from llama.cpp `cpy-utils.cuh`
// (TBQ_SIGNS) so that any test we run against llama.cpp's quantizer
// hits the same butterfly signs.

pub const tbq_signs = [32]u8{
    0xa7, 0x3b, 0x91, 0xf4, 0x6d, 0xc2, 0x58, 0x0e,
    0xb3, 0x7f, 0x24, 0xd6, 0x89, 0x45, 0xea, 0x1c,
    0x63, 0xaf, 0xd8, 0x52, 0x97, 0x0b, 0xe1, 0x3d,
    0x76, 0xc4, 0x19, 0xfe, 0x4a, 0x85, 0x2c, 0xdb,
};

/// +1.0 or -1.0 for the i-th sign flip. Indices wrap modulo 256.
pub inline fn rhtSign(i: usize) f32 {
    const idx: u8 = @intCast(i & 0xff);
    const shift: u3 = @intCast(idx & 7);
    const bit = (tbq_signs[idx >> 3] >> shift) & 1;
    return if (bit == 1) -1.0 else 1.0;
}

// ── Fast Walsh-Hadamard transform + sign-flip pre-conditioner ───────
//
// Naturally-ordered, unnormalised butterfly — H · H = d · I, so the
// inverse is FWHT followed by /d. The pre-conditioner multiplies each
// coordinate by `rhtSign(i)` before the forward FWHT (and after the
// inverse), which makes Π·x near-Gaussian for a wide range of input
// distributions and is what lets a single global Lloyd-Max codebook
// suffice for every layer / head / token.

/// In-place forward FWHT. Replaces x with H · x. `x.len` must be a
/// non-zero power of 2. Matches YATQ `serial_wht()` butterfly order.
pub fn fwht(x: []f32) void {
    const d = x.len;
    std.debug.assert(d > 0 and (d & (d - 1)) == 0);
    var length: usize = 1;
    while (length < d) : (length *= 2) {
        var i: usize = 0;
        while (i < d) : (i += 2 * length) {
            var j: usize = 0;
            while (j < length) : (j += 1) {
                const u = x[i + j];
                const v = x[i + j + length];
                x[i + j] = u + v;
                x[i + j + length] = u - v;
            }
        }
    }
}

/// In-place inverse FWHT: H · x / d. Matches YATQ `inverse_wht()`.
pub fn ifwht(x: []f32) void {
    fwht(x);
    const inv_d: f32 = 1.0 / @as(f32, @floatFromInt(x.len));
    for (x) |*v| v.* *= inv_d;
}

/// In-place RHT sign-flip step: x[i] *= rhtSign(i). Cycles through
/// tbq_signs every 256 elements.
pub fn applySignFlips(x: []f32) void {
    for (x, 0..) |*v, i| v.* *= rhtSign(i);
}

/// Forward RHT pre-conditioner: sign-flips, then FWHT.
pub fn rhtForward(x: []f32) void {
    applySignFlips(x);
    fwht(x);
}

/// Inverse RHT: IFWHT, then sign-flips. The composition
/// rhtInverse(rhtForward(x)) recovers x exactly (modulo fp32 rounding).
pub fn rhtInverse(x: []f32) void {
    ifwht(x);
    applySignFlips(x);
}

// ── Lloyd-Max quantize / dequantize ─────────────────────────────────

/// Bit-width parameter for the codebook. Phase 1 supports 3 and 4;
/// extending to 2/5/6/7/8 is just adding more tables.
pub const Bits = enum(u3) {
    b3 = 3,
    b4 = 4,
};

fn boundariesOf(comptime b: Bits) []const f32 {
    return switch (b) {
        .b3 => &lm_boundaries_b3,
        .b4 => &lm_boundaries_b4,
    };
}

fn centroidsOf(comptime b: Bits) []const f32 {
    return switch (b) {
        .b3 => &lm_centroids_b3,
        .b4 => &lm_centroids_b4,
    };
}

/// Map x to the index of the nearest Lloyd-Max centroid for `b`.
/// Boundaries are pre-sorted ascending; linear scan is faster than
/// binary search at 7/15 entries and matches YATQ's `(x >= b).sum()`
/// idiom exactly so off-by-ones are easy to compare.
pub fn lloydMaxIndex(x: f32, comptime b: Bits) u8 {
    var idx: u8 = 0;
    for (boundariesOf(b)) |bnd| {
        if (x >= bnd) idx += 1;
    }
    return idx;
}

/// Reconstruction value for the i-th codebook entry of `b`.
pub inline fn lloydMaxCentroid(i: u8, comptime b: Bits) f32 {
    return centroidsOf(b)[i];
}

// ── TQ4 packed-block format (256 elements / block) ──────────────────
//
// Memory layout matches llama.cpp `block_tbq4_0`: one fp16 norm
// scalar (γ) followed by 128 bytes holding 256 × 4-bit Lloyd-Max
// indices, two per byte — element 2k in the low nibble, 2k+1 in the
// high nibble. 130 bytes per 256-element block, 4× compression vs
// fp32 K-cache, ~2× vs fp16 K-cache.
//
// The γ scalar is the *norm-correction* factor (TheTom / spiritbuun
// trick): it is `original_L2 / ‖spatial-reconstruction‖`, not the raw
// L2 norm. Storing the corrected scalar guarantees the reconstructed
// vector has *exactly* the original L2 norm, which removes one source
// of bias per block at zero decode cost. Practitioner reports show
// this alone takes TQ3/TQ4 from "close to q8_0" to "actually beats
// q8_0" on PPL.

pub const block_size_tq4: usize = 256;

pub const BlockTQ4 = extern struct {
    gamma: f16,
    indices: [block_size_tq4 / 2]u8, // 128 bytes, two 4-bit indices per byte
};

comptime {
    // Pack guard: 2 + 128 = 130 bytes.
    std.debug.assert(@sizeOf(BlockTQ4) == 130);
}

inline fn packNibbles(lo: u8, hi: u8) u8 {
    return (lo & 0x0f) | ((hi & 0x0f) << 4);
}

inline fn unpackLo(byte: u8) u8 {
    return byte & 0x0f;
}

inline fn unpackHi(byte: u8) u8 {
    return (byte >> 4) & 0x0f;
}

/// Quantize a 256-element block into TQ4 packed form. Pipeline:
///   1. raw_norm = ‖x‖
///   2. x_norm = x / raw_norm                       (or all zeros)
///   3. y = FWHT(signs · x_norm)                    (rhtForward, in place)
///   4. indices[i] = lloydMaxIndex(y[i], b=4)
///   5. y_recon[i] = lloydMaxCentroid(indices[i], b=4)
///   6. x_recon = signs · IFWHT(y_recon)            (rhtInverse, in place)
///   7. γ = raw_norm / ‖x_recon‖                    (norm-correction)
pub fn quantizeBlockTQ4(input: *const [block_size_tq4]f32, out: *BlockTQ4) void {
    var raw_norm_sq: f32 = 0;
    for (input) |v| raw_norm_sq += v * v;
    const raw_norm = @sqrt(raw_norm_sq);

    if (raw_norm == 0) {
        out.gamma = 0;
        @memset(&out.indices, 0);
        return;
    }

    // Normalize + rhtForward in one buffer.
    var y: [block_size_tq4]f32 = undefined;
    const inv_norm = 1.0 / raw_norm;
    for (input, 0..) |v, i| y[i] = v * inv_norm;
    rhtForward(&y);

    // Lloyd-Max quantize → indices, and build WHT-domain reconstruction.
    var indices: [block_size_tq4]u8 = undefined;
    var y_recon: [block_size_tq4]f32 = undefined;
    for (y, 0..) |v, i| {
        const idx = lloydMaxIndex(v, .b4);
        indices[i] = idx;
        y_recon[i] = lloydMaxCentroid(idx, .b4);
    }

    // Inverse RHT to get the spatial-domain reconstruction, then take
    // its norm to compute the correction factor.
    rhtInverse(&y_recon);
    var recon_norm_sq: f32 = 0;
    for (y_recon) |v| recon_norm_sq += v * v;
    const recon_norm = @sqrt(recon_norm_sq);
    out.gamma = @floatCast(raw_norm / recon_norm);

    // Pack two 4-bit indices per byte.
    var k: usize = 0;
    while (k < block_size_tq4 / 2) : (k += 1) {
        out.indices[k] = packNibbles(indices[2 * k], indices[2 * k + 1]);
    }
}

/// Dequantize a TQ4 packed block back to fp32.
pub fn dequantizeBlockTQ4(in: *const BlockTQ4, out: *[block_size_tq4]f32) void {
    const gamma: f32 = @floatCast(in.gamma);
    if (gamma == 0) {
        @memset(out, 0);
        return;
    }

    // Unpack indices → centroid lookups in the WHT domain.
    var k: usize = 0;
    while (k < block_size_tq4 / 2) : (k += 1) {
        out[2 * k] = lloydMaxCentroid(unpackLo(in.indices[k]), .b4);
        out[2 * k + 1] = lloydMaxCentroid(unpackHi(in.indices[k]), .b4);
    }

    // Inverse RHT lifts back to the original (normalized) domain, then
    // γ rescales to the original L2 norm exactly.
    rhtInverse(out);
    for (out) |*v| v.* *= gamma;
}

// ── Self-tests ──────────────────────────────────────────────────────

test "rhtSign decodes first byte 0xa7 LSB-first" {
    // 0xa7 = 0b10100111 → bits 0..7 LSB-first = 1,1,1,0,0,1,0,1 →
    // signs = -1,-1,-1,+1,+1,-1,+1,-1.
    const expected = [_]f32{ -1.0, -1.0, -1.0, 1.0, 1.0, -1.0, 1.0, -1.0 };
    for (expected, 0..) |want, i| {
        try std.testing.expectEqual(want, rhtSign(i));
    }
}

test "rhtSign wraps modulo 256" {
    try std.testing.expectEqual(rhtSign(0), rhtSign(256));
    try std.testing.expectEqual(rhtSign(7), rhtSign(7 + 256));
    try std.testing.expectEqual(rhtSign(255), rhtSign(255 + 256));
}

test "lloydMaxIndex matches YATQ's (x >= b).sum() at corner values" {
    // b=3, eight centroids. x=0.0 is at boundary[3]=0.0, so it falls
    // into bin 4 (the first positive centroid, +0.2451).
    try std.testing.expectEqual(@as(u8, 4), lloydMaxIndex(0.0, .b3));
    // Anything well below the most-negative boundary lands in bin 0.
    try std.testing.expectEqual(@as(u8, 0), lloydMaxIndex(-5.0, .b3));
    // Anything well above the most-positive boundary lands in bin 7.
    try std.testing.expectEqual(@as(u8, 7), lloydMaxIndex(5.0, .b3));
    // x=0.6 is between 0.5005 and 1.0500 → bin 5 = centroid +0.756.
    try std.testing.expectEqual(@as(u8, 5), lloydMaxIndex(0.6, .b3));
    // x=-0.6 by symmetry falls into bin 2 = centroid -0.756.
    try std.testing.expectEqual(@as(u8, 2), lloydMaxIndex(-0.6, .b3));

    // b=4, sixteen centroids. x=0.0 → bin 8 (centroid +0.1284).
    try std.testing.expectEqual(@as(u8, 8), lloydMaxIndex(0.0, .b4));
    // x=0.4 lies between boundaries[8]=0.2582 and [9]=0.5224 → bin 9.
    try std.testing.expectEqual(@as(u8, 9), lloydMaxIndex(0.4, .b4));
}

test "lloydMaxCentroid round-trips through lloydMaxIndex on its own outputs" {
    // Quantize each centroid value back; you should recover the same
    // index. This also implicitly checks the centroid/boundary pairing.
    inline for ([_]Bits{ .b3, .b4 }) |b| {
        const cs = centroidsOf(b);
        for (cs, 0..) |c, i| {
            const idx = lloydMaxIndex(c, b);
            try std.testing.expectEqual(@as(u8, @intCast(i)), idx);
        }
    }
}

test "Lloyd-Max codebooks are symmetric about zero" {
    inline for ([_]Bits{ .b3, .b4 }) |b| {
        const cs = centroidsOf(b);
        const n = cs.len;
        var i: usize = 0;
        while (i < n / 2) : (i += 1) {
            const lo = cs[i];
            const hi = cs[n - 1 - i];
            try std.testing.expectApproxEqAbs(-lo, hi, 1e-6);
        }
    }
}

test "fwht hand-checked at d=4" {
    // Hand-computed: WHT([1,2,3,4]) = [10, -2, -4, 0].
    var x = [_]f32{ 1, 2, 3, 4 };
    fwht(&x);
    try std.testing.expectEqual(@as(f32, 10), x[0]);
    try std.testing.expectEqual(@as(f32, -2), x[1]);
    try std.testing.expectEqual(@as(f32, -4), x[2]);
    try std.testing.expectEqual(@as(f32, 0), x[3]);
}

test "fwht of delta_0 is all-ones" {
    // First row of the natural-ordered Hadamard matrix is all +1.
    var x = [_]f32{ 1, 0, 0, 0, 0, 0, 0, 0 };
    fwht(&x);
    for (x) |v| try std.testing.expectEqual(@as(f32, 1), v);
}

test "fwht twice multiplies by d" {
    // H · H = d · I.
    var x = [_]f32{ 0.5, -1.25, 3.0, 0.0, -2.5, 1.75, 0.125, -0.75 };
    const x_orig = x;
    fwht(&x);
    fwht(&x);
    const d: f32 = @floatFromInt(x.len);
    for (x, x_orig) |got, want| {
        try std.testing.expectApproxEqAbs(d * want, got, 1e-5);
    }
}

test "ifwht inverts fwht for an arbitrary 256-vector" {
    var prng = std.Random.DefaultPrng.init(0xCAFEBABE);
    const r = prng.random();
    var x: [256]f32 = undefined;
    for (&x) |*v| v.* = r.floatNorm(f32);
    var y = x;
    fwht(&y);
    ifwht(&y);
    for (x, y) |want, got| try std.testing.expectApproxEqAbs(want, got, 1e-4);
}

test "rht round-trip recovers the original vector" {
    var prng = std.Random.DefaultPrng.init(0xDEADBEEF);
    const r = prng.random();
    var x: [256]f32 = undefined;
    for (&x) |*v| v.* = r.floatNorm(f32);
    var y = x;
    rhtForward(&y);
    rhtInverse(&y);
    for (x, y) |want, got| try std.testing.expectApproxEqAbs(want, got, 1e-4);
}

test "TQ4 round-trip preserves L2 norm exactly" {
    var prng = std.Random.DefaultPrng.init(0x5EED1234);
    const r = prng.random();
    var x: [block_size_tq4]f32 = undefined;
    for (&x) |*v| v.* = r.floatNorm(f32);

    var raw_sq: f32 = 0;
    for (x) |v| raw_sq += v * v;
    const raw_norm = @sqrt(raw_sq);

    var blk: BlockTQ4 = undefined;
    quantizeBlockTQ4(&x, &blk);
    var y: [block_size_tq4]f32 = undefined;
    dequantizeBlockTQ4(&blk, &y);

    var rec_sq: f32 = 0;
    for (y) |v| rec_sq += v * v;
    const rec_norm = @sqrt(rec_sq);

    // Norm-correction trick should give recon_norm == raw_norm to within
    // f16-truncation tolerance on γ.
    try std.testing.expectApproxEqRel(raw_norm, rec_norm, 1e-3);
}

test "TQ4 zero vector round-trips to zero" {
    var x: [block_size_tq4]f32 = .{0} ** block_size_tq4;
    var blk: BlockTQ4 = undefined;
    quantizeBlockTQ4(&x, &blk);
    try std.testing.expectEqual(@as(f16, 0), blk.gamma);
    var y: [block_size_tq4]f32 = undefined;
    dequantizeBlockTQ4(&blk, &y);
    for (y) |v| try std.testing.expectEqual(@as(f32, 0), v);
}
