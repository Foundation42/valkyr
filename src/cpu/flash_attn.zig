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

const NEG_INF: f32 = -std.math.inf(f32);

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

