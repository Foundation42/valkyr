//! Q4_0-style 4-bit weight quantization, CPU reference.
//!
//! Block size = 32 elements, per-block fp16 scale, signed 4-bit
//! indices in [-8, 7] stored as q+8 in [0, 15]. The encoding follows
//! llama.cpp's `quantize_row_q4_0_reference` (block_q4_0 in
//! ggml-quants.c) so a future GGUF reader can pass byte-clean tensors
//! through this oracle, but the **on-device layout differs**: we store
//! one block as 5 `u32`s (1 word for the fp16 scale in the low half,
//! 4 words for 32 packed nibbles, sequential) for cleaner std430
//! alignment in the matmul shader. The repacking happens in
//! `gpu/model.zig` at upload time; here we just produce the canonical
//! [scale, idx0..idx31] tuples.
//!
//! Quantization is the standard "max-magnitude / -8" scheme: pick the
//! signed value with largest absolute magnitude in the block, divide
//! by -8 to get a (possibly negative) scale d, then for each element
//! compute id = round(x * (1/d)) clamped to [0, 15] (which corresponds
//! to signed q ∈ [-8, 7]). Dequant: x ≈ (id - 8) * d.
//!
//! This is symmetric quantization with no zero-point. For most
//! transformer weights it gives ≈0.05–0.10 PPL above Q4_K_M but is
//! dramatically simpler to implement (and bit-clean to match against
//! a CPU reference). We can layer Q4_1-style asymmetric on top later
//! if quality bites — the on-device shader already has a per-block
//! header word free for a min/zero-point.

const std = @import("std");

pub const BLOCK_SIZE: usize = 32;

/// Canonical CPU representation of one quantized block. Mirrors
/// llama.cpp's `block_q4_0` byte-for-byte (`d: f16` + `qs: [16]u8`),
/// where `qs[j]` packs index j in the low nibble and index j+16 in
/// the high nibble — that's the SIMD-friendly llama.cpp layout, kept
/// here for future GGUF interop. The GPU upload path translates this
/// into a sequential 5-u32 layout.
pub const Block = extern struct {
    d: f16,
    qs: [16]u8,
};

comptime {
    std.debug.assert(@sizeOf(Block) == 18);
    std.debug.assert(@alignOf(Block) == 2);
}

/// Quantize one row of `BLOCK_SIZE * n_blocks` floats into `n_blocks`
/// blocks. Caller sizes `dst.len == src.len / BLOCK_SIZE`.
pub fn quantizeRow(src: []const f32, dst: []Block) void {
    std.debug.assert(src.len % BLOCK_SIZE == 0);
    std.debug.assert(dst.len * BLOCK_SIZE == src.len);

    for (dst, 0..) |*blk, b| {
        const base = b * BLOCK_SIZE;
        // Pick the signed value with the largest absolute magnitude.
        // Using `max` (the signed value, not amax) so the scale `d`
        // inherits its sign — gives a slightly better fit when the
        // block is dominated by one extreme negative element.
        var amax: f32 = 0;
        var maxv: f32 = 0;
        for (0..BLOCK_SIZE) |j| {
            const v = src[base + j];
            const av = @abs(v);
            if (av > amax) {
                amax = av;
                maxv = v;
            }
        }
        const d: f32 = maxv / -8.0;
        // Negative scale ⇒ id = clamp(round(x * (1/d)) + 8, 0, 15).
        // The +8 shift turns signed q ∈ [-8, 7] into unsigned [0, 15]
        // so we can pack as nibbles. The MIN(15, …) handles the edge
        // case where the *largest* element rounds to +8 (which would
        // otherwise overflow the 4-bit range — saturate at 15 = +7
        // post-shift, i.e. signed +7).
        const id_inv: f32 = if (d != 0.0) 1.0 / d else 0.0;
        // qs[j] = idx[j] | (idx[j + 16] << 4) — llama.cpp's split layout.
        for (0..BLOCK_SIZE / 2) |j| {
            const x0 = src[base + j] * id_inv;
            const x1 = src[base + BLOCK_SIZE / 2 + j] * id_inv;
            const idx0: i32 = @intFromFloat(@floor(x0 + 8.5));
            const idx1: i32 = @intFromFloat(@floor(x1 + 8.5));
            const lo: u8 = @intCast(@max(@as(i32, 0), @min(@as(i32, 15), idx0)));
            const hi: u8 = @intCast(@max(@as(i32, 0), @min(@as(i32, 15), idx1)));
            blk.qs[j] = lo | (hi << 4);
        }
        blk.d = @floatCast(d);
    }
}

/// Dequantize `n_blocks` blocks back to `BLOCK_SIZE * n_blocks` floats.
pub fn dequantizeRow(src: []const Block, dst: []f32) void {
    std.debug.assert(dst.len % BLOCK_SIZE == 0);
    std.debug.assert(src.len * BLOCK_SIZE == dst.len);

    for (src, 0..) |blk, b| {
        const base = b * BLOCK_SIZE;
        const d: f32 = @floatCast(blk.d);
        for (0..BLOCK_SIZE / 2) |j| {
            const lo: i32 = @intCast(blk.qs[j] & 0x0F);
            const hi: i32 = @intCast((blk.qs[j] >> 4) & 0x0F);
            dst[base + j] = @as(f32, @floatFromInt(lo - 8)) * d;
            dst[base + BLOCK_SIZE / 2 + j] = @as(f32, @floatFromInt(hi - 8)) * d;
        }
    }
}

/// Number of u32 words per block in the on-device GPU layout.
pub const GPU_U32S_PER_BLOCK: usize = 5;

/// Repack canonical `Block` (18 bytes, llama.cpp-compatible split
/// nibble layout) into the GPU's sequential 5-u32-per-block layout.
/// Output layout per block:
///   word 0       = scale: fp16 d in low 16 bits, upper 16 bits zero.
///   words 1..5   = 32 4-bit unsigned indices, 8 per u32, sequential.
///                  Element k of the block lives in qs[k/8] at bit
///                  shift (k%8) * 4.
/// This matches `shaders/matmul_nt_v2_q4_0.comp`'s decoder.
pub fn packForGpu(blocks: []const Block, dst: []u32) void {
    std.debug.assert(dst.len == blocks.len * GPU_U32S_PER_BLOCK);
    for (blocks, 0..) |blk, b| {
        const base = b * GPU_U32S_PER_BLOCK;
        const d_bits: u16 = @bitCast(blk.d);
        dst[base] = @as(u32, d_bits);
        var w0: u32 = 0;
        var w1: u32 = 0;
        var w2: u32 = 0;
        var w3: u32 = 0;
        // CPU layout: qs[j] = idx[j] | (idx[j+16] << 4). Translate to
        // sequential nibbles 0..31 → words 0..3 with 8 nibbles each.
        for (0..16) |j| {
            const lo: u32 = blk.qs[j] & 0x0F;
            const hi: u32 = (blk.qs[j] >> 4) & 0x0F;
            // Element j is at sequential position j; element j+16 at j+16.
            const seq_lo: u32 = lo << @intCast(@as(u32, @intCast(j % 8)) * 4);
            const seq_hi: u32 = hi << @intCast(@as(u32, @intCast((j + 16) % 8)) * 4);
            switch (j / 8) {
                0 => w0 |= seq_lo,
                1 => w1 |= seq_lo,
                else => unreachable,
            }
            switch ((j + 16) / 8) {
                2 => w2 |= seq_hi,
                3 => w3 |= seq_hi,
                else => unreachable,
            }
        }
        dst[base + 1] = w0;
        dst[base + 2] = w1;
        dst[base + 3] = w2;
        dst[base + 4] = w3;
    }
}

/// Cosine similarity between `a` and `b`. Returns 1.0 for identical
/// directions, -1.0 for opposite, NaN if either is the zero vector.
pub fn cosineSim(a: []const f32, b: []const f32) f32 {
    std.debug.assert(a.len == b.len);
    var dot: f64 = 0;
    var na: f64 = 0;
    var nb: f64 = 0;
    for (a, b) |x, y| {
        dot += @as(f64, x) * @as(f64, y);
        na += @as(f64, x) * @as(f64, x);
        nb += @as(f64, y) * @as(f64, y);
    }
    return @floatCast(dot / (@sqrt(na) * @sqrt(nb)));
}
