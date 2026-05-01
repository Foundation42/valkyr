//! Q4_K_M-style 4-bit weight quantization, CPU reference.
//!
//! Super-block size = 256 elements (= 8 sub-blocks of 32). Each sub-block
//! gets a 6-bit unsigned scale and a 6-bit unsigned min; the super-block
//! holds two fp16 super-scales (`d` for the per-sub-block scales, `dmin`
//! for the per-sub-block mins). Per element: 4-bit unsigned quant. The
//! dequant formula is asymmetric:
//!
//!   x ≈ d * sc[j] * q[i] - dmin * m[j]
//!
//! where `j = i / 32` is the sub-block, `sc[j], m[j] ∈ [0,63]`, `q[i] ∈
//! [0,15]`. The asymmetric offset `m[j]` is what beats Q4_0 on quality —
//! Q4_0's symmetric quant has to pay full precision on both sides of zero
//! even when the sub-block is one-sided. The iterative `make_qkx2_quants`
//! also outperforms Q4_0's max-magnitude scheme.
//!
//! Layout matches llama.cpp's `block_q4_K` byte-for-byte (144 bytes per
//! super-block: 2× fp16 + 12 B packed scales + 128 B packed nibbles), so
//! a future GGUF reader can pass tensors straight through this oracle.
//! The on-device GPU layout is a separate concern — see `packForGpu` in
//! a sibling chunk.
//!
//! Effective bit rate: 4.5 bits/elem (vs 5 bits/elem for our padded q4_0
//! GPU layout). Bandwidth win on dominant matmuls is ~10%; the bigger
//! draw is quality (~0.05–0.10 PPL improvement, what llama.cpp ships
//! as default).

const std = @import("std");

pub const QK_K: usize = 256;
pub const K_SCALE_SIZE: usize = 12;

/// Canonical CPU representation of one super-block. Mirrors llama.cpp's
/// `block_q4_K`: two fp16 super-scales, 12 packed 6-bit (scale, min)
/// pairs for 8 sub-blocks, and 128 bytes of 4-bit unsigned weights.
///
/// `qs` packs **pairs** of sub-blocks: bytes 0..31 hold sub-blocks 0+1
/// (low nibble = element of sub-block 0, high nibble = element of
/// sub-block 1, both at the same intra-sub-block index). Bytes 32..63
/// hold pair (2,3); 64..95 hold (4,5); 96..127 hold (6,7). Same SIMD-
/// friendly llama.cpp layout we keep here for GGUF compat.
pub const Block = extern struct {
    d: f16,
    dmin: f16,
    scales: [K_SCALE_SIZE]u8,
    qs: [QK_K / 2]u8,
};

comptime {
    std.debug.assert(@sizeOf(Block) == 144);
    std.debug.assert(@offsetOf(Block, "scales") == 4);
    std.debug.assert(@offsetOf(Block, "qs") == 16);
}

inline fn nearestInt(x: f32) i32 {
    return @as(i32, @intFromFloat(@round(x)));
}

/// Decode the 6-bit (scale, min) pair for sub-block `j` ∈ [0, 8) from
/// the 12-byte packed `q`. Mirrors llama.cpp's `get_scale_min_k4`.
///
/// Layout (encoder must zero `q` before encoding):
///   q[0..3].low6   = sc[0..3]              q[4..7].low6   = m[0..3]
///   q[8..11].low4  = sc[4..7].low4         q[8..11].high4 = m[4..7].low4
///   q[0..3].high2  = sc[4..7].high2        q[4..7].high2  = m[4..7].high2
pub fn getScaleMinK4(j: u32, q: *const [K_SCALE_SIZE]u8, d_out: *u8, m_out: *u8) void {
    if (j < 4) {
        d_out.* = q[j] & 63;
        m_out.* = q[j + 4] & 63;
    } else {
        d_out.* = (q[j + 4] & 0x0F) | ((q[j - 4] >> 6) << 4);
        m_out.* = (q[j + 4] >> 4) | ((q[j] >> 6) << 4);
    }
}

/// llama.cpp's `make_qkx2_quants`. Computes per-sub-block scale (positive)
/// and -min (also positive after the `min ≤ 0` clamp), iteratively
/// refining via least-squares over `nstep+1` candidate iscales drawn from
/// `(rmin + rdelta*is + nmax) / (max-min)`. Writes 4-bit unsigned quant
/// indices into `L`. `weights[i]` lets per-element importance bias the
/// fit — for q4_K they're set to `av_x + |x[i]|`, biasing the fit toward
/// large-magnitude elements.
fn makeQkx2Quants(
    n: usize,
    nmax: i32,
    x: []const f32,
    weights: []const f32,
    L: []u8,
    Laux: []u8,
    rmin: f32,
    rdelta: f32,
    nstep: i32,
    use_mad: bool,
    the_min: *f32,
) f32 {
    std.debug.assert(x.len >= n);
    std.debug.assert(weights.len >= n);
    std.debug.assert(L.len >= n);
    std.debug.assert(Laux.len >= n);

    var min: f32 = x[0];
    var max: f32 = x[0];
    var sum_w: f32 = weights[0];
    var sum_x: f32 = sum_w * x[0];
    for (1..n) |i| {
        if (x[i] < min) min = x[i];
        if (x[i] > max) max = x[i];
        const w = weights[i];
        sum_w += w;
        sum_x += w * x[i];
    }
    if (min > 0) min = 0;
    if (max == min) {
        for (L[0..n]) |*p| p.* = 0;
        the_min.* = -min;
        return 0.0;
    }
    var iscale: f32 = @as(f32, @floatFromInt(nmax)) / (max - min);
    var scale: f32 = 1.0 / iscale;
    var best_error: f32 = 0;
    for (0..n) |i| {
        const l_raw = nearestInt(iscale * (x[i] - min));
        const l: i32 = @max(0, @min(nmax, l_raw));
        L[i] = @intCast(l);
        var diff: f32 = scale * @as(f32, @floatFromInt(l)) + min - x[i];
        diff = if (use_mad) @abs(diff) else diff * diff;
        best_error += weights[i] * diff;
    }
    if (nstep < 1) {
        the_min.* = -min;
        return scale;
    }
    var is: i32 = 0;
    while (is <= nstep) : (is += 1) {
        iscale = (rmin + rdelta * @as(f32, @floatFromInt(is)) + @as(f32, @floatFromInt(nmax))) / (max - min);
        var sum_l: f32 = 0;
        var sum_l2: f32 = 0;
        var sum_xl: f32 = 0;
        for (0..n) |i| {
            const l_raw = nearestInt(iscale * (x[i] - min));
            const l: i32 = @max(0, @min(nmax, l_raw));
            Laux[i] = @intCast(l);
            const w = weights[i];
            const lf: f32 = @floatFromInt(l);
            sum_l += w * lf;
            sum_l2 += w * lf * lf;
            sum_xl += w * lf * x[i];
        }
        const D: f32 = sum_w * sum_l2 - sum_l * sum_l;
        if (D > 0) {
            var this_scale: f32 = (sum_w * sum_xl - sum_x * sum_l) / D;
            var this_min: f32 = (sum_l2 * sum_x - sum_l * sum_xl) / D;
            if (this_min > 0) {
                this_min = 0;
                this_scale = sum_xl / sum_l2;
            }
            var cur_error: f32 = 0;
            for (0..n) |i| {
                var diff: f32 = this_scale * @as(f32, @floatFromInt(Laux[i])) + this_min - x[i];
                diff = if (use_mad) @abs(diff) else diff * diff;
                cur_error += weights[i] * diff;
            }
            if (cur_error < best_error) {
                @memcpy(L[0..n], Laux[0..n]);
                best_error = cur_error;
                scale = this_scale;
                min = this_min;
            }
        }
    }
    the_min.* = -min;
    return scale;
}

/// Quantize `src.len` floats (must be a multiple of 256) into `dst.len`
/// super-blocks (= src.len / 256). Mirrors llama.cpp's
/// `quantize_row_q4_K_ref` — same iterative refinement, same packing.
pub fn quantizeRow(src: []const f32, dst: []Block) void {
    std.debug.assert(src.len % QK_K == 0);
    std.debug.assert(dst.len * QK_K == src.len);

    var L_buf: [QK_K]u8 = undefined;
    var Laux: [32]u8 = undefined;
    var weights: [32]f32 = undefined;
    var mins: [QK_K / 32]f32 = undefined;
    var scales: [QK_K / 32]f32 = undefined;

    for (dst, 0..) |*blk, b| {
        // Phase 0: zero the scales region. Encoding for j ∈ [4,8) uses
        // `|=` into bytes 0..3 and 4..7 (top-2-bit slots) — those slots
        // must start clean. `qs` is fully assigned in phase 4 so doesn't
        // need pre-zeroing.
        @memset(&blk.scales, 0);

        const x_base = b * QK_K;
        var max_scale: f32 = 0;
        var max_min: f32 = 0;

        // Phase 1: solve per-sub-block (scale, min) via the iterative
        // refinement. Weights bias the fit toward large-magnitude
        // elements (av_x + |x[l]|), matching llama.cpp.
        for (0..QK_K / 32) |j| {
            const sub_x = src[x_base + 32 * j .. x_base + 32 * j + 32];
            var sum_x2: f32 = 0;
            for (sub_x) |v| sum_x2 += v * v;
            const av_x: f32 = @sqrt(sum_x2 / 32.0);
            for (sub_x, 0..) |v, l| weights[l] = av_x + @abs(v);
            scales[j] = makeQkx2Quants(
                32,
                15,
                sub_x,
                &weights,
                L_buf[32 * j .. 32 * j + 32],
                &Laux,
                -1.0,
                0.1,
                20,
                false,
                &mins[j],
            );
            if (scales[j] > max_scale) max_scale = scales[j];
            if (mins[j] > max_min) max_min = mins[j];
        }

        const inv_scale: f32 = if (max_scale > 0) 63.0 / max_scale else 0.0;
        const inv_min: f32 = if (max_min > 0) 63.0 / max_min else 0.0;

        // Phase 2: 6-bit-encode per-sub-block (scale, min) into 12 bytes.
        for (0..QK_K / 32) |j| {
            const ls_raw = nearestInt(inv_scale * scales[j]);
            const lm_raw = nearestInt(inv_min * mins[j]);
            const ls: u8 = @intCast(@min(63, @max(0, ls_raw)));
            const lm: u8 = @intCast(@min(63, @max(0, lm_raw)));
            if (j < 4) {
                blk.scales[j] = ls;
                blk.scales[j + 4] = lm;
            } else {
                blk.scales[j + 4] = (ls & 0x0F) | ((lm & 0x0F) << 4);
                blk.scales[j - 4] |= @as(u8, @intCast(ls >> 4)) << 6;
                blk.scales[j] |= @as(u8, @intCast(lm >> 4)) << 6;
            }
        }

        blk.d = @floatCast(max_scale / 63.0);
        blk.dmin = @floatCast(max_min / 63.0);

        // Phase 3: re-quantize values using the *encoded* (lossy) scales.
        // The 6-bit rounding in phase 2 means d*sc and dmin*m no longer
        // exactly match scales[j] and mins[j]; recomputing q against the
        // actually-stored scales preserves the dequant invariant.
        for (0..QK_K / 32) |j| {
            var sc: u8 = undefined;
            var m: u8 = undefined;
            getScaleMinK4(@intCast(j), &blk.scales, &sc, &m);
            const d_eff: f32 = @as(f32, @floatCast(blk.d)) * @as(f32, @floatFromInt(sc));
            if (d_eff == 0.0) continue;
            const dm_eff: f32 = @as(f32, @floatCast(blk.dmin)) * @as(f32, @floatFromInt(m));
            for (0..32) |ii| {
                const l_raw = nearestInt((src[x_base + 32 * j + ii] + dm_eff) / d_eff);
                const l: i32 = @max(0, @min(15, l_raw));
                L_buf[32 * j + ii] = @intCast(l);
            }
        }

        // Phase 4: pack 256 4-bit weights into 128 bytes, paired sub-blocks.
        var q_off: usize = 0;
        var j: usize = 0;
        while (j < QK_K) : (j += 64) {
            for (0..32) |l| {
                blk.qs[q_off + l] = L_buf[j + l] | (L_buf[j + l + 32] << 4);
            }
            q_off += 32;
        }
    }
}

/// Number of u32 words per super-block in the on-device GPU layout.
/// 144 bytes / 4 = 36 words, no padding.
pub const GPU_U32S_PER_SUPERBLOCK: usize = 36;

/// Repack canonical `Block` (144 bytes, llama.cpp-compatible) into the
/// GPU's contiguous 36-u32-per-super-block layout. Output per super-
/// block:
///   word 0     = d (low 16 bits, fp16) | dmin (high 16 bits, fp16)
///   words 1..3 = scales[12], little-endian byte-packed (4 bytes/word)
///   words 4..35= qs[128],     little-endian byte-packed (4 bytes/word)
///
/// This matches `shaders/matmul_nt_v2_q4_k.comp`'s decoder.
pub fn packForGpu(blocks: []const Block, dst: []u32) void {
    std.debug.assert(dst.len == blocks.len * GPU_U32S_PER_SUPERBLOCK);
    for (blocks, 0..) |blk, b| {
        const base = b * GPU_U32S_PER_SUPERBLOCK;
        const d_bits: u32 = @as(u16, @bitCast(blk.d));
        const dmin_bits: u32 = @as(u16, @bitCast(blk.dmin));
        dst[base] = d_bits | (dmin_bits << 16);
        for (0..3) |w| {
            const off = w * 4;
            dst[base + 1 + w] =
                @as(u32, blk.scales[off]) |
                (@as(u32, blk.scales[off + 1]) << 8) |
                (@as(u32, blk.scales[off + 2]) << 16) |
                (@as(u32, blk.scales[off + 3]) << 24);
        }
        for (0..32) |w| {
            const off = w * 4;
            dst[base + 4 + w] =
                @as(u32, blk.qs[off]) |
                (@as(u32, blk.qs[off + 1]) << 8) |
                (@as(u32, blk.qs[off + 2]) << 16) |
                (@as(u32, blk.qs[off + 3]) << 24);
        }
    }
}

/// Dequantize `src.len` super-blocks into `src.len * 256` floats. Mirrors
/// llama.cpp's `dequantize_row_q4_K`.
pub fn dequantizeRow(src: []const Block, dst: []f32) void {
    std.debug.assert(dst.len % QK_K == 0);
    std.debug.assert(src.len * QK_K == dst.len);

    for (src, 0..) |blk, b| {
        const y_base = b * QK_K;
        const d: f32 = @floatCast(blk.d);
        const dmin: f32 = @floatCast(blk.dmin);
        var is: u32 = 0;
        var j: usize = 0;
        var q_off: usize = 0;
        while (j < QK_K) : (j += 64) {
            var sc: u8 = undefined;
            var m: u8 = undefined;
            getScaleMinK4(is, &blk.scales, &sc, &m);
            const d1: f32 = d * @as(f32, @floatFromInt(sc));
            const m1: f32 = dmin * @as(f32, @floatFromInt(m));
            getScaleMinK4(is + 1, &blk.scales, &sc, &m);
            const d2: f32 = d * @as(f32, @floatFromInt(sc));
            const m2: f32 = dmin * @as(f32, @floatFromInt(m));
            for (0..32) |l| {
                dst[y_base + j + l] = d1 * @as(f32, @floatFromInt(blk.qs[q_off + l] & 0x0F)) - m1;
            }
            for (0..32) |l| {
                dst[y_base + j + 32 + l] = d2 * @as(f32, @floatFromInt(blk.qs[q_off + l] >> 4)) - m2;
            }
            q_off += 32;
            is += 2;
        }
    }
}
