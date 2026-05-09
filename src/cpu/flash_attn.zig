//! CPU reference for FlashAttention forward (Dao et al. 2022,
//! Algorithm 1; FlashAttention-2-style outer-Q loop).
//!
//! Computes the same `out` tensor as
//! `cpu_train_transformer.attentionForward` but with tiled
//! accumulation and online softmax — never materialises the full
//! [n_q × n_heads × n_kv] scores tensor. The mathematical identity is
//! exact (modulo fp32 rounding from a different reduction order):
//!
//!     out[q, h, :] = Σ_k softmax(Q · Kᵀ · inv_sqrt_d)[q, h, k] · V[k, kv_h(h), :]
//!
//! Block sizes Br/Bc are tunable; correctness is independent of the
//! choice. The "online" trick maintains running rowmax (m_i) and
//! rowsum (l_i) across K-block iterations, rescaling the partial
//! output every time a new max is seen:
//!
//!     m_new = max(m_old, m_tile)
//!     l_new = exp(m_old − m_new) · l_old + rowsum(P_tile)
//!     O_new = exp(m_old − m_new) · O_old + P_tile · V_tile
//!     P_tile = exp(S_tile − m_new)
//!
//! At the end, divide by l_final to get the softmax-weighted output.
//! Optionally outputs `lse[q, h] = m_final + log(l_final)` — the
//! quantity FlashAttention backward needs to recompute the softmax
//! without saving the full attn matrix.
//!
//! GQA / MQA fold: query head `h` reads K/V slab `kv_h(h) = h /
//! heads_per_kv`, identical to the standard path. Causal mask
//! (when `causal == true`) follows the same `k_limit = q + (n_kv −
//! n_q)` convention as `attentionForward`, applied per-row inside
//! each tile.

const std = @import("std");
const cpu_turboquant = @import("turboquant.zig");

const NEG_INF: f32 = -std.math.inf(f32);

/// Re-assemble a 256-element TQ4 block from the GPU-cache u32 layout
/// (33 u32s per block: word[0] = fp32-bits of fp16-quantised gamma,
/// words[1..33] = 32 LE u32s of packed 4-bit indices) back into the
/// CPU `BlockTQ4(256)` struct that `dequantizeBlockTQ4` consumes. The
/// gamma round-trip is exact because the cache value is the fp32
/// representation of the fp16-rounded gamma — `@floatCast` to f16 is
/// the inverse.
inline fn cacheSliceToBlockTq4_256(cache: *const [33]u32) cpu_turboquant.BlockTQ4(256) {
    var blk: cpu_turboquant.BlockTQ4(256) = undefined;
    const gamma_f32: f32 = @bitCast(cache[0]);
    blk.gamma = @floatCast(gamma_f32);
    inline for (0..32) |w| {
        std.mem.writeInt(u32, blk.indices[w * 4 ..][0..4], cache[1 + w], .little);
    }
    return blk;
}

/// Inverse of `cacheSliceToBlockTq4_256`: pack a `BlockTQ4(256)` into
/// the 33-u32 GPU cache layout. Used by smokes to feed CPU-quantised
/// blocks into the GPU shader.
pub fn blockTq4ToCacheSlice_256(blk: *const cpu_turboquant.BlockTQ4(256), cache: *[33]u32) void {
    const gamma_f32: f32 = @floatCast(blk.gamma);
    cache[0] = @bitCast(gamma_f32);
    inline for (0..32) |w| {
        cache[1 + w] = std.mem.readInt(u32, blk.indices[w * 4 ..][0..4], .little);
    }
}

inline fn causalKeyLimit(q: usize, n_q: usize, n_kv: usize) usize {
    return q + (n_kv - n_q);
}

/// Tiled attention forward with online softmax. Single-pass: writes
/// `out` directly without ever holding the full scores matrix.
///
/// Block sizes Br (queries per tile) and Bc (keys per tile) are
/// independent of correctness; pick whatever fits the eventual GPU
/// shared-memory budget. Caller-allocated tile scratch keeps this
/// function free of heap allocs.
///
/// Shapes:
///   Q   [n_q,  n_heads,    head_dim]
///   K   [n_kv, n_kv_heads, head_dim]
///   V   [n_kv, n_kv_heads, head_dim]
///   out [n_q, n_heads, head_dim]
///   lse [n_q, n_heads]            optional (`null` to skip)
///
/// Tile scratch (caller-allocated):
///   s_tile [Br × Bc]    per-tile scores
///   p_tile [Br × Bc]    per-tile probabilities (exp-shifted)
///   o_acc  [Br × head_dim]   running output accumulator
///   m_acc  [Br]              running rowmax
///   l_acc  [Br]              running rowsum
pub fn flashAttentionForward(
    Q: []const f32,
    K: []const f32,
    V: []const f32,
    n_q: usize,
    n_kv: usize,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    causal: bool,
    Br: usize,
    Bc: usize,
    out: []f32,
    lse: ?[]f32,
    s_tile: []f32,
    p_tile: []f32,
    o_acc: []f32,
    m_acc: []f32,
    l_acc: []f32,
) void {
    std.debug.assert(n_heads % n_kv_heads == 0);
    const heads_per_kv = n_heads / n_kv_heads;
    std.debug.assert(Q.len == n_q * n_heads * head_dim);
    std.debug.assert(K.len == n_kv * n_kv_heads * head_dim);
    std.debug.assert(V.len == n_kv * n_kv_heads * head_dim);
    std.debug.assert(out.len == n_q * n_heads * head_dim);
    std.debug.assert(s_tile.len >= Br * Bc);
    std.debug.assert(p_tile.len >= Br * Bc);
    std.debug.assert(o_acc.len >= Br * head_dim);
    std.debug.assert(m_acc.len >= Br);
    std.debug.assert(l_acc.len >= Br);
    if (lse) |L| std.debug.assert(L.len == n_q * n_heads);
    if (causal) std.debug.assert(n_q <= n_kv);

    const inv_sqrt_d: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(head_dim)));

    var q_block_start: usize = 0;
    while (q_block_start < n_q) : (q_block_start += Br) {
        const q_block_end = @min(q_block_start + Br, n_q);
        const q_block_len = q_block_end - q_block_start;

        for (0..n_heads) |h| {
            const kv_h = h / heads_per_kv;

            // ── Initialise per-row accumulators for this (q_block, h) tile.
            for (0..q_block_len) |qi| m_acc[qi] = NEG_INF;
            for (0..q_block_len) |qi| l_acc[qi] = 0.0;
            for (0..q_block_len * head_dim) |i| o_acc[i] = 0.0;

            var k_block_start: usize = 0;
            while (k_block_start < n_kv) : (k_block_start += Bc) {
                const k_block_end = @min(k_block_start + Bc, n_kv);
                const k_block_len = k_block_end - k_block_start;

                // ── 1. S_tile[Br × Bc] = Q_block · K_block^T · inv_sqrt_d.
                for (0..q_block_len) |qi| {
                    const q = q_block_start + qi;
                    const k_limit: usize = if (causal) causalKeyLimit(q, n_q, n_kv) else n_kv - 1;
                    const q_off = q * n_heads * head_dim + h * head_dim;
                    for (0..k_block_len) |ki| {
                        const k = k_block_start + ki;
                        if (causal and k > k_limit) {
                            s_tile[qi * Bc + ki] = NEG_INF;
                            continue;
                        }
                        const k_off = k * n_kv_heads * head_dim + kv_h * head_dim;
                        var s: f64 = 0;
                        for (0..head_dim) |d| {
                            s += @as(f64, Q[q_off + d]) * @as(f64, K[k_off + d]);
                        }
                        s_tile[qi * Bc + ki] = @as(f32, @floatCast(s)) * inv_sqrt_d;
                    }
                }

                // ── 2. Per-row online softmax update.
                for (0..q_block_len) |qi| {
                    // Tile rowmax.
                    var m_tile: f32 = NEG_INF;
                    for (0..k_block_len) |ki| {
                        const v = s_tile[qi * Bc + ki];
                        if (v > m_tile) m_tile = v;
                    }

                    const m_old = m_acc[qi];
                    const m_new: f32 = if (m_tile > m_old) m_tile else m_old;

                    // P_tile = exp(S_tile - m_new); rowsum gives this tile's contribution to l.
                    var l_tile: f64 = 0;
                    for (0..k_block_len) |ki| {
                        const s = s_tile[qi * Bc + ki];
                        const p: f32 = if (s == NEG_INF) 0.0 else @exp(s - m_new);
                        p_tile[qi * Bc + ki] = p;
                        l_tile += p;
                    }

                    // scale_old = exp(m_old - m_new) for rescaling the running O / l.
                    // When m_old is -inf (first non-empty tile), exp(-inf - m_new) = 0,
                    // which is what we want — the all-zero O accumulator and zero
                    // l_acc survive the rescale untouched.
                    const scale_old: f32 = if (m_old == NEG_INF) 0.0 else @exp(m_old - m_new);

                    // Rescale O_block row, then add P_tile · V_block contribution.
                    for (0..head_dim) |d| {
                        o_acc[qi * head_dim + d] *= scale_old;
                    }
                    for (0..k_block_len) |ki| {
                        const p = p_tile[qi * Bc + ki];
                        if (p == 0.0) continue;
                        const k = k_block_start + ki;
                        const v_off = k * n_kv_heads * head_dim + kv_h * head_dim;
                        for (0..head_dim) |d| {
                            o_acc[qi * head_dim + d] += p * V[v_off + d];
                        }
                    }

                    m_acc[qi] = m_new;
                    l_acc[qi] = scale_old * l_acc[qi] + @as(f32, @floatCast(l_tile));
                }
            }

            // ── 3. Normalise and write out (and optional LSE).
            for (0..q_block_len) |qi| {
                const q = q_block_start + qi;
                const inv_l: f32 = if (l_acc[qi] > 0) 1.0 / l_acc[qi] else 0.0;
                const o_off = q * n_heads * head_dim + h * head_dim;
                for (0..head_dim) |d| {
                    out[o_off + d] = o_acc[qi * head_dim + d] * inv_l;
                }
                if (lse) |L| {
                    L[q * n_heads + h] = if (l_acc[qi] > 0)
                        m_acc[qi] + @log(l_acc[qi])
                    else
                        NEG_INF;
                }
            }
        }
    }
}

/// FlashAttention backward (Dao et al. 2023, Algorithm 4 — the
/// FlashAttention-2 backward). Recomputes the softmax inline from the
/// saved `O` and `lse` buffers, never materialising the full
/// [n_q × n_heads × n_kv] attention matrix that
/// `cpu_train_transformer.attentionBackward` consumes.
///
/// Math identity (exact modulo fp32 rounding from a different reduction
/// order). Given the saved forward results `O = attn · V` (where
/// `attn = softmax(Q · Kᵀ · inv_sqrt_d)`) and `lse = log(Σ_k exp(S))`:
///
///     D[q, h] = Σ_d O[q, h, d] · dO[q, h, d]
///             = Σ_k attn[q, h, k] · (Σ_d dO[q, h, d] · V[k, kv_h, d])
///             = Σ_k attn[q, h, k] · d_attn[q, h, k]                  (eq. ★)
///
///     P[q, h, k] = exp(S[q, h, k] − lse[q, h])    (= attn[q, h, k])
///     dP[q, h, k] = Σ_d dO[q, h, d] · V[k, kv_h, d]
///     dS[q, h, k] = P[q, h, k] · (dP[q, h, k] − D[q, h])
///
/// where the substitution at (★) is what lets the FA-2 backward replace
/// the [n_q × n_heads × n_kv] `Σ_k(attn · d_attn)` term in `softmax_backward`
/// with a [n_q × n_heads] rowwise reduction over `head_dim`. The final
/// gradients are unchanged:
///
///     dV[k, kv_h, d] = Σ_h_in_kv Σ_q P[q, h, k] · dO[q, h, d]
///     dQ[q, h, d]    = Σ_k dS[q, h, k] · K[k, kv_h, d] / √d
///     dK[k, kv_h, d] = Σ_h_in_kv Σ_q dS[q, h, k] · Q[q, h, d] / √d
///
/// Writes (overwrites; does not accumulate) into `dQ`, `dK`, `dV` —
/// matches `attentionBackward`'s contract.
///
/// Causal mask follows the same `k_limit = q + (n_kv − n_q)` convention
/// as the forward; entries beyond `k_limit` are skipped (P stays zero,
/// no contribution to gradients).
///
/// Heap-free: D[q, h] is computed inline once per (q, h) row before
/// the k loop, replacing the [n_q × n_heads × n_kv] `d_scores` scratch
/// that the 3-pass oracle requires.
///
/// Shapes:
///   Q   [n_q,  n_heads,    head_dim]
///   K   [n_kv, n_kv_heads, head_dim]
///   V   [n_kv, n_kv_heads, head_dim]
///   O   [n_q,  n_heads,    head_dim]    saved forward output
///   dO  [n_q,  n_heads,    head_dim]    incoming gradient
///   lse [n_q,  n_heads]                 saved log-sum-exp from forward
///   dQ  [n_q,  n_heads,    head_dim]    output (overwritten)
///   dK  [n_kv, n_kv_heads, head_dim]    output (overwritten)
///   dV  [n_kv, n_kv_heads, head_dim]    output (overwritten)
pub fn flashAttentionBackward(
    Q: []const f32,
    K: []const f32,
    V: []const f32,
    O: []const f32,
    dO: []const f32,
    lse: []const f32,
    n_q: usize,
    n_kv: usize,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    causal: bool,
    dQ: []f32,
    dK: []f32,
    dV: []f32,
) void {
    std.debug.assert(n_heads % n_kv_heads == 0);
    const heads_per_kv = n_heads / n_kv_heads;
    std.debug.assert(Q.len == n_q * n_heads * head_dim);
    std.debug.assert(K.len == n_kv * n_kv_heads * head_dim);
    std.debug.assert(V.len == n_kv * n_kv_heads * head_dim);
    std.debug.assert(O.len == n_q * n_heads * head_dim);
    std.debug.assert(dO.len == n_q * n_heads * head_dim);
    std.debug.assert(lse.len == n_q * n_heads);
    std.debug.assert(dQ.len == n_q * n_heads * head_dim);
    std.debug.assert(dK.len == n_kv * n_kv_heads * head_dim);
    std.debug.assert(dV.len == n_kv * n_kv_heads * head_dim);
    if (causal) std.debug.assert(n_q <= n_kv);

    const inv_sqrt_d: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(head_dim)));

    @memset(dQ, 0.0);
    @memset(dK, 0.0);
    @memset(dV, 0.0);

    for (0..n_q) |q| {
        for (0..n_heads) |h| {
            const kv_h = h / heads_per_kv;
            const o_off = q * n_heads * head_dim + h * head_dim;
            // dQ[q, h, :] / dO[q, h, :] / Q[q, h, :] / O[q, h, :] all
            // share this offset (q-major heads-second head_dim-last).
            const lse_qh = lse[q * n_heads + h];

            // If the entire row was masked at forward time, no gradient
            // flows through this (q, h). Skips the recompute and zero
            // contribution to dQ/dK/dV (already memset).
            if (lse_qh == NEG_INF) continue;

            // D[q, h] = Σ_d O[q, h, d] · dO[q, h, d] — fp64 to match
            // `attentionBackward`'s d-axis reduction precision. This
            // is the FA-2 simplification of `Σ_k attn · d_attn`.
            var D: f64 = 0;
            for (0..head_dim) |d| {
                D += @as(f64, O[o_off + d]) * @as(f64, dO[o_off + d]);
            }
            const D_f32: f32 = @floatCast(D);

            const k_limit: usize = if (causal) causalKeyLimit(q, n_q, n_kv) else n_kv - 1;

            for (0..n_kv) |k| {
                if (causal and k > k_limit) continue;

                const k_off = k * n_kv_heads * head_dim + kv_h * head_dim;

                // Recompute S = Q · Kᵀ · inv_sqrt_d, then P = exp(S − LSE).
                var s: f64 = 0;
                for (0..head_dim) |d| {
                    s += @as(f64, Q[o_off + d]) * @as(f64, K[k_off + d]);
                }
                const S: f32 = @as(f32, @floatCast(s)) * inv_sqrt_d;
                const P: f32 = @exp(S - lse_qh);
                if (P == 0.0) continue;

                // dV[k, kv_h, d] += P · dO[q, h, d]
                for (0..head_dim) |d| {
                    dV[k_off + d] += P * dO[o_off + d];
                }

                // dP = Σ_d dO[q, h, d] · V[k, kv_h, d] — fp64 reduction.
                var dP: f64 = 0;
                for (0..head_dim) |d| {
                    dP += @as(f64, dO[o_off + d]) * @as(f64, V[k_off + d]);
                }

                // dS = P · (dP − D); scale by inv_sqrt_d once for the
                // dQ / dK chain rule.
                const dS_scaled: f32 = P * (@as(f32, @floatCast(dP)) - D_f32) * inv_sqrt_d;

                // dQ[q, h, d] += dS_scaled · K[k, kv_h, d]
                // dK[k, kv_h, d] += dS_scaled · Q[q, h, d]
                for (0..head_dim) |d| {
                    dQ[o_off + d] += dS_scaled * K[k_off + d];
                    dK[k_off + d] += dS_scaled * Q[o_off + d];
                }
            }
        }
    }
}


// ── Fused FlashAttention decode forward with TQ4-packed V cache ──────
//
// Same algorithm as `flashAttentionForward` (decode-only: n_q is
// implicit in `Q.len == n_heads × head_dim`), but reads V from the
// GPU TQ4 cache layout instead of a fp32 V tensor. Each per-(k, kv_h)
// V vector is dequantised inline from a 256-element TQ4 block (33 u32s)
// just before it's consumed by the running-O update.
//
// The mathematical contract matches the GPU `fa_decode_split_tq4v.comp`
// kernel under construction: gamma is stored as the fp32 representation
// of the fp16-rounded value, indices are 4-bit Lloyd-Max picks of the
// L2-normalised RHT-rotated input. Reconstruction error is the
// dequantizeBlockTQ4 norm-rel-err (~1e-4 on Gaussian inputs), so parity
// vs the unfused fp-V FA path holds within that tolerance, NOT bit-equal.
//
// d=256 only — the block size is fixed at 256, matching head_dim for
// every model in this T-arc (Gemma 2B + Qwen3.5 0.8B/4B). A d=128
// variant would need its own helper around `BlockTQ4(128)` (and the
// matching `tq4_unpack128.comp`).

/// Decode-mode (n_q = 1) flash-attention with TQ4-V cache.
///
/// Shapes:
///   Q          [n_heads × 256]
///   K          [n_kv × n_kv_heads × 256]
///   V_tq4      [n_kv × n_blocks_per_pos × 33]    GPU cache layout
///                                                 (n_blocks_per_pos == n_kv_heads for d=256)
///   out        [n_heads × 256]
///   v_block    [256]                              dequant scratch (caller-allocated)
///   s_tile     [Bc]                               score scratch
///   p_tile     [Bc]                               probability scratch
///   o_acc      [256]                              running O accumulator
///
/// Tolerance vs fp-V reference: ~1e-4 norm-rel (TQ4 reconstruction
/// error), NOT bit-equal — same headroom the GPU smoke uses.
pub fn flashAttentionDecodeForwardTq4V(
    Q: []const f32,
    K: []const f32,
    V_tq4: []const u32,
    n_kv: usize,
    n_heads: usize,
    n_kv_heads: usize,
    n_blocks_per_pos: usize,
    Bc: usize,
    out: []f32,
    s_tile: []f32,
    p_tile: []f32,
    o_acc: []f32,
    v_block: []f32,
) void {
    const head_dim: usize = 256;
    std.debug.assert(n_heads % n_kv_heads == 0);
    const heads_per_kv = n_heads / n_kv_heads;
    std.debug.assert(Q.len == n_heads * head_dim);
    std.debug.assert(K.len == n_kv * n_kv_heads * head_dim);
    std.debug.assert(V_tq4.len == n_kv * n_blocks_per_pos * 33);
    std.debug.assert(out.len == n_heads * head_dim);
    std.debug.assert(s_tile.len >= Bc);
    std.debug.assert(p_tile.len >= Bc);
    std.debug.assert(o_acc.len >= head_dim);
    std.debug.assert(v_block.len >= head_dim);
    // d=256 path assumes one TQ4 block per (kv_h, head_dim chunk). For
    // GQA models with kv_dim > head_dim, n_blocks_per_pos == n_kv_heads.
    std.debug.assert(n_blocks_per_pos == n_kv_heads);

    const inv_sqrt_d: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(head_dim)));

    for (0..n_heads) |h| {
        const kv_h = h / heads_per_kv;
        const q_off = h * head_dim;

        var m_state: f32 = NEG_INF;
        var l_state: f32 = 0.0;
        for (0..head_dim) |d| o_acc[d] = 0.0;

        var k_block_start: usize = 0;
        while (k_block_start < n_kv) : (k_block_start += Bc) {
            const k_block_end = @min(k_block_start + Bc, n_kv);
            const k_block_len = k_block_end - k_block_start;

            // S_tile[ki] = Q · K[k_block_start + ki, kv_h, :] · inv_sqrt_d.
            // K stays fp32 — only V is TQ4-packed in this T-arc.
            for (0..k_block_len) |ki| {
                const k = k_block_start + ki;
                const k_off = k * n_kv_heads * head_dim + kv_h * head_dim;
                var s: f64 = 0;
                for (0..head_dim) |d| {
                    s += @as(f64, Q[q_off + d]) * @as(f64, K[k_off + d]);
                }
                s_tile[ki] = @as(f32, @floatCast(s)) * inv_sqrt_d;
            }

            // Tile rowmax + online softmax update.
            var m_tile: f32 = NEG_INF;
            for (0..k_block_len) |ki| {
                if (s_tile[ki] > m_tile) m_tile = s_tile[ki];
            }
            const m_old = m_state;
            const m_new: f32 = if (m_tile > m_old) m_tile else m_old;

            var l_tile: f64 = 0;
            for (0..k_block_len) |ki| {
                const p: f32 = @exp(s_tile[ki] - m_new);
                p_tile[ki] = p;
                l_tile += p;
            }

            const scale_old: f32 = if (m_old == NEG_INF) 0.0 else @exp(m_old - m_new);
            // Rescale running O before adding this tile's contribution.
            for (0..head_dim) |d| o_acc[d] *= scale_old;

            // For each key in this tile, dequant its V block from the
            // TQ4 cache and add P_tile[ki] · V to o_acc.
            for (0..k_block_len) |ki| {
                const k = k_block_start + ki;
                const block_idx: usize = k * n_blocks_per_pos + kv_h;
                const cache_off: usize = block_idx * 33;
                const cache_slice = V_tq4[cache_off..][0..33];
                const blk = cacheSliceToBlockTq4_256(cache_slice);
                var v_arr: [256]f32 = undefined;
                cpu_turboquant.dequantizeBlockTQ4(256, &blk, &v_arr);
                @memcpy(v_block[0..head_dim], v_arr[0..]);

                const p = p_tile[ki];
                if (p == 0.0) continue;
                for (0..head_dim) |d| {
                    o_acc[d] += p * v_block[d];
                }
            }

            m_state = m_new;
            l_state = scale_old * l_state + @as(f32, @floatCast(l_tile));
        }

        // Final normalise + write out.
        const inv_l: f32 = if (l_state > 0) 1.0 / l_state else 0.0;
        for (0..head_dim) |d| {
            out[q_off + d] = o_acc[d] * inv_l;
        }
    }
}
