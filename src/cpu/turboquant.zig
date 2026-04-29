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
