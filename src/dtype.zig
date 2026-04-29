//! Reduced-precision → fp32 conversion.
//!
//! Used to materialise weights for the CPU reference forward pass and
//! (later) to feed the GPU staging path while we're still fp32-only
//! on-device. Both bf16 and fp16 are bit-pattern conversions — no
//! arithmetic, no rounding decisions to make on the way *up* to fp32.
//!
//! Layouts (sign | exponent | mantissa, MSB first):
//!     fp32   : 1 |  8 | 23
//!     bf16   : 1 |  8 |  7    ← same exponent as fp32 → just a shift
//!     fp16   : 1 |  5 | 10    ← different exponent bias → repack
//!
//! Performance note: for the CPU reference we don't bother with SIMD;
//! a scalar loop over `numel` runs at memory bandwidth on any modern
//! CPU, and the kernel is paged-in mmap data so we're DRAM-bound either
//! way. When we move to GPU we won't convert at all — we'll stage bf16
//! into a buffer and let a one-shot conversion kernel run on the GPU.

const std = @import("std");

/// Convert one bf16 value (held in a u16) to f32. bf16's sign+exponent+
/// mantissa layout is the upper 16 bits of fp32, so we just shift the
/// bits into place; the lower 16 bits of the fp32 mantissa are zero
/// (truncation). NaN / Inf / subnormal patterns survive unchanged.
pub inline fn bf16ToF32(bits: u16) f32 {
    const u: u32 = @as(u32, bits) << 16;
    return @bitCast(u);
}

/// Convert one fp16 value (held in a u16) to f32. fp16 has a 5-bit
/// exponent biased by 15; fp32 has an 8-bit exponent biased by 127, so
/// we shift the exponent up by 13 (bit positions of mantissa) and add
/// (127 - 15) << 23 to rebias. Subnormals and Inf/NaN get the special-
/// case branches.
pub inline fn f16ToF32(bits: u16) f32 {
    const sign: u32 = (@as(u32, bits) & 0x8000) << 16;
    const exp: u32 = (@as(u32, bits) & 0x7C00) >> 10;
    const mant: u32 = @as(u32, bits) & 0x03FF;

    if (exp == 0) {
        if (mant == 0) {
            // Signed zero.
            return @bitCast(sign);
        }
        // Subnormal — normalise into fp32 by shifting until the leading
        // 1 falls off the mantissa, decrementing the implicit exponent
        // accordingly. We start at fp32 exponent 127 - 15 + 1 = 113
        // (one less than the smallest fp16 normal exponent in fp32 bias).
        var m = mant;
        var e: u32 = 113;
        while ((m & 0x0400) == 0) {
            m <<= 1;
            e -%= 1;
        }
        m &= 0x03FF;
        const bits_out: u32 = sign | (e << 23) | (m << 13);
        return @bitCast(bits_out);
    }
    if (exp == 0x1F) {
        // Inf or NaN — preserve mantissa nonzero-ness.
        const bits_out: u32 = sign | 0x7F800000 | (mant << 13);
        return @bitCast(bits_out);
    }
    // Normal — rebias and shift.
    const bits_out: u32 = sign | ((exp + 112) << 23) | (mant << 13);
    return @bitCast(bits_out);
}

/// Convert `n` bf16 values from `src` into `dst`. Both slices must
/// have length `n`. `src` is byte-addressed (no alignment requirement
/// — SafeTensors data sections aren't always aligned).
pub fn bf16SliceToF32(src: []align(1) const u16, dst: []f32) void {
    std.debug.assert(src.len == dst.len);
    for (src, dst) |s, *d| d.* = bf16ToF32(s);
}

/// Convert `n` fp16 values from `src` into `dst`.
pub fn f16SliceToF32(src: []align(1) const u16, dst: []f32) void {
    std.debug.assert(src.len == dst.len);
    for (src, dst) |s, *d| d.* = f16ToF32(s);
}

// ── Convenience: view tensor bytes as a typed unaligned u16 slice ────

pub fn asU16(bytes: []const u8) []align(1) const u16 {
    std.debug.assert(bytes.len % 2 == 0);
    return @as([*]align(1) const u16, @ptrCast(bytes.ptr))[0 .. bytes.len / 2];
}
