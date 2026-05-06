//! Cut Cross-Entropy (CCE) — chunked online-softmax cross-entropy that
//! never materializes the full `[N, V]` logit tensor.
//!
//! Mathematically identical to the standard "matmul → softmax → CE" path
//! used by `softmaxCeLoss` + `softmaxCeLossGrad` in `train_decoder.zig`,
//! but with `O(N·C)` working memory instead of `O(N·V)`. For Qwen2.5
//! (V = 151,936) at N = 4096 and C = 4096, that's ~620 MB → ~64 MB
//! transient and the persistent footprint drops to a single `[N]` `lse`
//! buffer (16 KB) + the existing `[N]` `target_ids` (16 KB).
//!
//! Reference: Wijmans et al., "Cut Your Losses in Large-Vocabulary
//! Language Models" (2024); algorithms 1–3 of Nair, "Chronicals" (2026,
//! arXiv:2601.02609v1).
//!
//! ## The two-pass design
//!
//! Forward (`cceForward`):
//!   for each row n:
//!     m, d ← -inf, 0                              -- online softmax stats
//!     z_target ← 0
//!     for each vocab chunk c of width C:
//!       z_chunk[C] ← h[n] · W[c·C : (c+1)·C]ᵀ    -- compute, never store
//!       m_new      ← max(m, max(z_chunk))
//!       d          ← d · exp(m − m_new) + Σ exp(z_chunk − m_new)
//!       m          ← m_new
//!       if target_id[n] ∈ this chunk: z_target ← z_chunk[target_id[n] − c·C]
//!     lse[n]   ← log(d) + m                       -- cached for backward
//!     loss[n]  ← lse[n] − z_target                -- −log p_target
//!   return mean(loss)
//!
//! Backward (`cceBackward`):
//!   for each row n, each chunk c:
//!     z_chunk[C]  ← h[n] · W[c·C : (c+1)·C]ᵀ     -- recompute (same math as fwd)
//!     for o in chunk:
//!       p     ← exp(z_chunk[o] − lse[n])          -- softmax via cached lse
//!       dz_o  ← (p − [c·C+o == target_id[n]]) / N  -- pre-scaled by mean
//!       d_h[n]            += dz_o · W[c·C+o]
//!       dW[c·C+o]         += dz_o · h[n]
//!
//! Compute cost: forward is one `N·V·D` matmul (chunked, never written
//! to HBM). Backward recomputes the matmul (`N·V·D`) and does both grad
//! accumulations in the same chunk pass — total `2·N·V·D`, same as the
//! materialized path's `linearBackward(d_logits → d_h, dW)`. The win is
//! purely on memory traffic: the materialized `[N, V]` logit and
//! d_logits tensors are gone.
//!
//! ## Precision
//!
//! All inner reductions accumulate in f64 to match `softmaxCeLoss` /
//! `softmaxCeLossGrad` / `linearBackward` in `train_decoder.zig`. The
//! parity test in this file asserts agreement to 1e-5 relative against
//! the materialized reference.
//!
//! ## Convention
//!
//! - `h` is `[N, D]` row-major (HF activation layout)
//! - `W` is `[V, D]` row-major (HF linear weight layout: each row is one
//!   output's input weights)
//! - `target_ids` is `[N]` u32 indices into vocab
//! - `d_h` is **overwritten** (matches `linearBackward`'s `dx`)
//! - `dW` is **accumulated** into (matches `linearBackward`'s `dW`)
//! - Loss is mean-reduced over `N`; `dz` is pre-scaled by `1/N` so the
//!   downstream optimizer sees averaged gradients without an extra pass

const std = @import("std");

/// Optional regularizers on the cross-entropy loss. Both default to 0
/// (plain CE) so existing call sites pass `.{}` and get bit-equal
/// behavior to the un-regularized path.
///
/// `z_loss_scale` (Chronicals §"Z-Loss", PaLM eq. 15) adds λ_z · lse²
/// to the loss. Quadratically penalizes large logsumexp values to
/// prevent fp16 overflow. Typical training value: 1e-4.
///
/// `label_smoothing` (Chronicals §"Label Smoothing", eq. 13–14) softens
/// the one-hot target to (1-ε)·δ_{v,target} + ε/V. Equivalent to
/// blending L_CE with L_uniform = lse − mean(z). Typical value: 0.1.
pub const CceLossOpts = struct {
    z_loss_scale: f32 = 0.0,
    label_smoothing: f32 = 0.0,
};

/// Forward pass: online-softmax CE over chunks of vocab, never
/// materializing logits. Writes per-row log-sum-exp into `lse_out` for
/// the backward pass to reuse. `opts` selects the loss form (plain CE,
/// z-loss, label smoothing, or both).
///
/// Asserts (in debug) on shape and chunk-size sanity.
pub fn cceForward(
    h: []const f32,
    w_lm: []const f32,
    target_ids: []const u32,
    n: usize,
    vocab: usize,
    dim: usize,
    chunk: usize,
    opts: CceLossOpts,
    lse_out: []f32,
) f32 {
    std.debug.assert(h.len == n * dim);
    std.debug.assert(w_lm.len == vocab * dim);
    std.debug.assert(target_ids.len == n);
    std.debug.assert(lse_out.len == n);
    std.debug.assert(chunk > 0);

    var total_loss: f64 = 0;
    const eps: f32 = opts.label_smoothing;
    const t_scale: f32 = 1.0 - eps; // (1 − ε) factor on the target logit

    for (0..n) |row| {
        const h_off = row * dim;
        var m: f32 = -std.math.inf(f32);
        var d_acc: f64 = 0;
        var z_sum: f64 = 0; // running Σ_v z_v for label-smoothing's L_uniform
        var z_target: f32 = 0;
        const tgt: usize = @intCast(target_ids[row]);
        std.debug.assert(tgt < vocab);

        var c_start: usize = 0;
        while (c_start < vocab) : (c_start += chunk) {
            const c_end = @min(c_start + chunk, vocab);

            // Compute z_chunk[o] = Σ_k h[row,k] · w_lm[c_start+o, k]
            // and the chunk-local max in the same pass. Bounded fixed
            // stack allocation — caller picks `chunk` knowing that.
            var chunk_max: f32 = -std.math.inf(f32);
            var z_chunk_buf: [4096]f32 = undefined;
            std.debug.assert(c_end - c_start <= z_chunk_buf.len);
            const z = z_chunk_buf[0 .. c_end - c_start];

            for (0..z.len) |o| {
                const w_off = (c_start + o) * dim;
                var s: f64 = 0;
                for (0..dim) |k| {
                    s += @as(f64, h[h_off + k]) * @as(f64, w_lm[w_off + k]);
                }
                const zi: f32 = @floatCast(s);
                z[o] = zi;
                if (zi > chunk_max) chunk_max = zi;
                // Label smoothing's L_uniform term needs Σ_v z_v / V.
                // Accumulate the raw chunk sum here in f64 (independent
                // of the online-softmax rescaling).
                z_sum += @as(f64, zi);
            }

            // Online softmax merge.
            const m_new = @max(m, chunk_max);
            // d_new = d · exp(m − m_new) + Σ exp(z − m_new)
            var chunk_sum: f64 = 0;
            for (z) |zi| chunk_sum += @exp(@as(f64, zi) - @as(f64, m_new));
            const rescale: f64 = if (std.math.isInf(m)) 0.0 else @exp(@as(f64, m) - @as(f64, m_new));
            d_acc = d_acc * rescale + chunk_sum;
            m = m_new;

            // Capture the target logit if it falls in this chunk.
            if (tgt >= c_start and tgt < c_end) {
                z_target = z[tgt - c_start];
            }
        }

        const lse: f32 = @floatCast(@log(d_acc) + @as(f64, m));
        lse_out[row] = lse;
        // Label smoothing: blend hard CE with uniform CE.
        //   L_smoothed = (1−ε)·(lse − z_target) + ε·(lse − z_mean)
        //              = lse − (1−ε)·z_target − ε·z_mean
        const z_mean: f64 = z_sum / @as(f64, @floatFromInt(vocab));
        const ce_smoothed: f64 = @as(f64, lse) - @as(f64, t_scale) * @as(f64, z_target) - @as(f64, eps) * z_mean;
        const z_penalty: f64 = @as(f64, opts.z_loss_scale) * @as(f64, lse) * @as(f64, lse);
        total_loss += ce_smoothed + z_penalty;
    }

    return @floatCast(total_loss / @as(f64, @floatFromInt(n)));
}

/// Backward pass: recompute z_chunk per chunk, derive softmax probs
/// from the cached `lse`, and stream gradients into `d_h` (overwrite)
/// and `dW` (accumulate) without ever instantiating dL/dz.
///
/// `lse` must come from the matching `cceForward` call — same `h`,
/// `w_lm`, `chunk`. Mismatched lse is silent corruption (f64 numerics
/// will look "close" but be wrong); call sites should treat the
/// (lse, h, w_lm) triple as an atomic carry across forward and
/// backward.
/// `opts` selects the regularizer terms (z-loss + label smoothing).
/// Must match the same value passed to `cceForward`. With z-loss the
/// softmax part of dz picks up a (1 + 2·λ_z·lse) factor; with label
/// smoothing the target indicator becomes (1−ε) and every dz_v gets
/// an additional −ε/V offset.
pub fn cceBackward(
    h: []const f32,
    w_lm: []const f32,
    target_ids: []const u32,
    lse: []const f32,
    n: usize,
    vocab: usize,
    dim: usize,
    chunk: usize,
    opts: CceLossOpts,
    d_h: []f32,
    dW: []f32,
) void {
    std.debug.assert(h.len == n * dim);
    std.debug.assert(w_lm.len == vocab * dim);
    std.debug.assert(target_ids.len == n);
    std.debug.assert(lse.len == n);
    std.debug.assert(d_h.len == n * dim);
    std.debug.assert(dW.len == vocab * dim);
    std.debug.assert(chunk > 0);

    // d_h is *overwritten* (matches linearBackward(dx_opt) convention).
    // dW *accumulates* (matches linearBackward(dW) convention) — caller
    // zeroes between optimizer steps via grads.zero().
    @memset(d_h, 0);

    const inv_n: f32 = 1.0 / @as(f32, @floatFromInt(n));

    const eps: f32 = opts.label_smoothing;
    const eps_over_v: f32 = eps / @as(f32, @floatFromInt(vocab));
    const t_scale: f32 = 1.0 - eps;

    for (0..n) |row| {
        const h_off = row * dim;
        const lse_row: f32 = lse[row];
        const tgt: usize = @intCast(target_ids[row]);
        // Z-loss multiplies the softmax part of the gradient by
        // (1 + 2·λ_z·lse). When λ_z = 0 this is exactly 1 and we
        // recover plain CE. Computed once per row (not per element).
        const softmax_factor: f32 = 1.0 + 2.0 * opts.z_loss_scale * lse_row;

        var c_start: usize = 0;
        while (c_start < vocab) : (c_start += chunk) {
            const c_end = @min(c_start + chunk, vocab);
            var z_chunk_buf: [4096]f32 = undefined;
            std.debug.assert(c_end - c_start <= z_chunk_buf.len);
            const z = z_chunk_buf[0 .. c_end - c_start];

            // Recompute z_chunk[o] = h[row] · w_lm[c_start+o]ᵀ.
            // Same math as forward — f64 accumulator preserves
            // self-consistency with the lse cached in forward.
            for (0..z.len) |o| {
                const w_off = (c_start + o) * dim;
                var s: f64 = 0;
                for (0..dim) |k| {
                    s += @as(f64, h[h_off + k]) * @as(f64, w_lm[w_off + k]);
                }
                z[o] = @floatCast(s);
            }

            // For each o in chunk, derive
            //   dz_o = (softmax_factor · softmax_o − (1−ε)·[o==t] − ε/V) / N
            // and stream into d_h[row] and dW[c_start+o] in one pass.
            for (0..z.len) |o| {
                const vocab_idx = c_start + o;
                const p: f32 = @floatCast(@exp(@as(f64, z[o]) - @as(f64, lse_row)));
                const t_indicator: f32 = if (vocab_idx == tgt) t_scale else 0.0;
                const dz: f32 = (softmax_factor * p - t_indicator - eps_over_v) * inv_n;

                const w_off = vocab_idx * dim;
                // d_h[row, k] += dz · W[vocab_idx, k]
                // dW[vocab_idx, k] += dz · h[row, k]
                for (0..dim) |k| {
                    d_h[h_off + k] += dz * w_lm[w_off + k];
                    dW[w_off + k] += dz * h[h_off + k];
                }
            }
        }
    }
}

// ── Tests ───────────────────────────────────────────────────────────
//
// Parity oracle: drive a materialized "matmul → softmax CE → linearBwd"
// reference with the same inputs as CCE, assert agreement on loss, d_h,
// and dW within 1e-5 relative. Vocab and dim small enough that running
// the reference is cheap, but vocab > chunk so the online-softmax merge
// is exercised across multiple chunks.

const testing = std.testing;

fn refMatmulNt(out: []f32, h: []const f32, w: []const f32, n: usize, v: usize, d: usize) void {
    for (0..n) |row| {
        for (0..v) |col| {
            var s: f64 = 0;
            for (0..d) |k| s += @as(f64, h[row * d + k]) * @as(f64, w[col * d + k]);
            out[row * v + col] = @floatCast(s);
        }
    }
}

fn refSoftmaxCeLoss(logits: []const f32, target_ids: []const u32, n: usize, v: usize) f32 {
    var total: f64 = 0;
    for (0..n) |p| {
        const off = p * v;
        var m: f32 = -std.math.inf(f32);
        for (0..v) |o| m = @max(m, logits[off + o]);
        var sum_e: f64 = 0;
        for (0..v) |o| sum_e += @exp(@as(f64, logits[off + o]) - @as(f64, m));
        const log_z: f64 = @as(f64, m) + @log(sum_e);
        const tgt: usize = @intCast(target_ids[p]);
        total -= @as(f64, logits[off + tgt]) - log_z;
    }
    return @floatCast(total / @as(f64, @floatFromInt(n)));
}

fn refSoftmaxCeLossGrad(d_logits: []f32, logits: []const f32, target_ids: []const u32, n: usize, v: usize) void {
    const inv_n: f32 = 1.0 / @as(f32, @floatFromInt(n));
    for (0..n) |p| {
        const off = p * v;
        var m: f32 = -std.math.inf(f32);
        for (0..v) |o| m = @max(m, logits[off + o]);
        var sum_e: f64 = 0;
        for (0..v) |o| sum_e += @exp(@as(f64, logits[off + o]) - @as(f64, m));
        const inv_sum: f64 = 1.0 / sum_e;
        const tgt: usize = @intCast(target_ids[p]);
        for (0..v) |o| {
            const p_o: f64 = @exp(@as(f64, logits[off + o]) - @as(f64, m)) * inv_sum;
            const t_o: f32 = if (o == tgt) 1.0 else 0.0;
            d_logits[off + o] = inv_n * (@as(f32, @floatCast(p_o)) - t_o);
        }
    }
}

/// Reference linear-backward, mirrors train_decoder.linearBackward.
/// dx is overwritten, dW is accumulated.
fn refLinearBackward(
    dout: []const f32,
    x: []const f32,
    w: []const f32,
    n: usize,
    v: usize,
    d: usize,
    dx: []f32,
    dW: []f32,
) void {
    for (0..n) |m| {
        for (0..d) |k| {
            var s: f64 = 0;
            for (0..v) |j| s += @as(f64, dout[m * v + j]) * @as(f64, w[j * d + k]);
            dx[m * d + k] = @floatCast(s);
        }
    }
    for (0..v) |j| {
        for (0..d) |k| {
            var s: f64 = 0;
            for (0..n) |m| s += @as(f64, dout[m * v + j]) * @as(f64, x[m * d + k]);
            dW[j * d + k] += @floatCast(s);
        }
    }
}

/// Global relative-error metric: `max|a − b| / max|a|`. Standard
/// matrix-comparison metric — robust to noise-floor entries where
/// individual elements are dominated by f32 round-off but the overall
/// tensor magnitude is well-defined. Per-element relative diff would
/// blow up on entries that are mathematically ≈ 0; this measures how
/// big the worst error is *relative to the largest signal* in the
/// reference.
fn maxAbsRelDiff(a: []const f32, b: []const f32) f32 {
    std.debug.assert(a.len == b.len);
    var max_diff: f32 = 0;
    var max_a: f32 = 0;
    for (a, b) |av, bv| {
        const diff = @abs(av - bv);
        if (diff > max_diff) max_diff = diff;
        if (@abs(av) > max_a) max_a = @abs(av);
    }
    return if (max_a > 1e-30) max_diff / max_a else max_diff;
}

test "cceForward matches materialized softmaxCeLoss across chunk sizes" {
    const gpa = testing.allocator;
    const n: usize = 4;
    const dim: usize = 32;
    const vocab: usize = 1024;

    var prng = std.Random.DefaultPrng.init(0xC0FFEE);
    const rng = prng.random();

    const h = try gpa.alloc(f32, n * dim);
    defer gpa.free(h);
    const w = try gpa.alloc(f32, vocab * dim);
    defer gpa.free(w);
    const targets = try gpa.alloc(u32, n);
    defer gpa.free(targets);

    for (h) |*v| v.* = (rng.float(f32) - 0.5) * 0.1;
    for (w) |*v| v.* = (rng.float(f32) - 0.5) * 0.1;
    for (targets) |*t| t.* = rng.intRangeLessThan(u32, 0, vocab);

    const logits = try gpa.alloc(f32, n * vocab);
    defer gpa.free(logits);
    refMatmulNt(logits, h, w, n, vocab, dim);
    const ref_loss = refSoftmaxCeLoss(logits, targets, n, vocab);

    // Test several chunk sizes including ones that don't divide vocab.
    const chunks = [_]usize{ 64, 128, 256, 333, 1024 };
    const lse = try gpa.alloc(f32, n);
    defer gpa.free(lse);
    for (chunks) |c| {
        const cce_loss = cceForward(h, w, targets, n, vocab, dim, c, .{}, lse);
        const rel = @abs(cce_loss - ref_loss) / @max(@abs(ref_loss), 1e-30);
        try testing.expect(rel < 1e-5);
    }
}

test "cceBackward matches materialized softmaxCeLossGrad → linearBackward" {
    const gpa = testing.allocator;
    const n: usize = 4;
    const dim: usize = 32;
    const vocab: usize = 1024;
    const chunk: usize = 256;

    var prng = std.Random.DefaultPrng.init(0xBADF00D);
    const rng = prng.random();

    const h = try gpa.alloc(f32, n * dim);
    defer gpa.free(h);
    const w = try gpa.alloc(f32, vocab * dim);
    defer gpa.free(w);
    const targets = try gpa.alloc(u32, n);
    defer gpa.free(targets);

    for (h) |*v| v.* = (rng.float(f32) - 0.5) * 0.1;
    for (w) |*v| v.* = (rng.float(f32) - 0.5) * 0.1;
    for (targets) |*t| t.* = rng.intRangeLessThan(u32, 0, vocab);

    // Reference path: matmul → softmaxCeLossGrad → linearBackward.
    const logits = try gpa.alloc(f32, n * vocab);
    defer gpa.free(logits);
    refMatmulNt(logits, h, w, n, vocab, dim);

    const d_logits = try gpa.alloc(f32, n * vocab);
    defer gpa.free(d_logits);
    refSoftmaxCeLossGrad(d_logits, logits, targets, n, vocab);

    const d_h_ref = try gpa.alloc(f32, n * dim);
    defer gpa.free(d_h_ref);
    const dW_ref = try gpa.alloc(f32, vocab * dim);
    defer gpa.free(dW_ref);
    @memset(dW_ref, 0);
    refLinearBackward(d_logits, h, w, n, vocab, dim, d_h_ref, dW_ref);

    // CCE path: cceForward (caches lse) → cceBackward.
    const lse = try gpa.alloc(f32, n);
    defer gpa.free(lse);
    _ = cceForward(h, w, targets, n, vocab, dim, chunk, .{}, lse);

    const d_h_cce = try gpa.alloc(f32, n * dim);
    defer gpa.free(d_h_cce);
    const dW_cce = try gpa.alloc(f32, vocab * dim);
    defer gpa.free(dW_cce);
    @memset(dW_cce, 0);
    cceBackward(h, w, targets, lse, n, vocab, dim, chunk, .{}, d_h_cce, dW_cce);

    const d_h_rel = maxAbsRelDiff(d_h_ref, d_h_cce);
    const dW_rel = maxAbsRelDiff(dW_ref, dW_cce);
    try testing.expect(d_h_rel < 1e-5);
    try testing.expect(dW_rel < 1e-5);
}

test "cceForward + cceBackward agree across chunk sizes (chunk irrelevance)" {
    const gpa = testing.allocator;
    const n: usize = 3;
    const dim: usize = 16;
    const vocab: usize = 512;

    var prng = std.Random.DefaultPrng.init(0xCA11AB1E);
    const rng = prng.random();

    const h = try gpa.alloc(f32, n * dim);
    defer gpa.free(h);
    const w = try gpa.alloc(f32, vocab * dim);
    defer gpa.free(w);
    const targets = try gpa.alloc(u32, n);
    defer gpa.free(targets);

    for (h) |*v| v.* = (rng.float(f32) - 0.5) * 0.3;
    for (w) |*v| v.* = (rng.float(f32) - 0.5) * 0.3;
    for (targets) |*t| t.* = rng.intRangeLessThan(u32, 0, vocab);

    // Run with chunk = vocab (single-chunk, equivalent to materialized
    // softmax) and compare against multiple smaller chunk sizes.
    const lse_full = try gpa.alloc(f32, n);
    defer gpa.free(lse_full);
    const loss_full = cceForward(h, w, targets, n, vocab, dim, vocab, .{}, lse_full);
    const d_h_full = try gpa.alloc(f32, n * dim);
    defer gpa.free(d_h_full);
    const dW_full = try gpa.alloc(f32, vocab * dim);
    defer gpa.free(dW_full);
    @memset(dW_full, 0);
    cceBackward(h, w, targets, lse_full, n, vocab, dim, vocab, .{}, d_h_full, dW_full);

    const chunks = [_]usize{ 32, 64, 128, 250, 333 };
    for (chunks) |c| {
        const lse_c = try gpa.alloc(f32, n);
        defer gpa.free(lse_c);
        const loss_c = cceForward(h, w, targets, n, vocab, dim, c, .{}, lse_c);

        const d_h_c = try gpa.alloc(f32, n * dim);
        defer gpa.free(d_h_c);
        const dW_c = try gpa.alloc(f32, vocab * dim);
        defer gpa.free(dW_c);
        @memset(dW_c, 0);
        cceBackward(h, w, targets, lse_c, n, vocab, dim, c, .{}, d_h_c, dW_c);

        const loss_rel = @abs(loss_c - loss_full) / @max(@abs(loss_full), 1e-30);
        try testing.expect(loss_rel < 1e-5);
        try testing.expect(maxAbsRelDiff(lse_full, lse_c) < 1e-5);
        try testing.expect(maxAbsRelDiff(d_h_full, d_h_c) < 1e-5);
        try testing.expect(maxAbsRelDiff(dW_full, dW_c) < 1e-5);
    }
}

test "cceBackward accumulates into dW (does not overwrite)" {
    const gpa = testing.allocator;
    const n: usize = 2;
    const dim: usize = 8;
    const vocab: usize = 64;
    const chunk: usize = 32;

    var prng = std.Random.DefaultPrng.init(0xACCD);
    const rng = prng.random();

    const h = try gpa.alloc(f32, n * dim);
    defer gpa.free(h);
    const w = try gpa.alloc(f32, vocab * dim);
    defer gpa.free(w);
    const targets = try gpa.alloc(u32, n);
    defer gpa.free(targets);

    for (h) |*v| v.* = (rng.float(f32) - 0.5);
    for (w) |*v| v.* = (rng.float(f32) - 0.5);
    for (targets) |*t| t.* = rng.intRangeLessThan(u32, 0, vocab);

    const lse = try gpa.alloc(f32, n);
    defer gpa.free(lse);
    _ = cceForward(h, w, targets, n, vocab, dim, chunk, .{}, lse);

    const d_h = try gpa.alloc(f32, n * dim);
    defer gpa.free(d_h);

    // Pre-fill dW with sentinel; running cceBackward should add to it,
    // not replace it. Run twice and check the second call doubles the
    // delta against the sentinel.
    const dW_a = try gpa.alloc(f32, vocab * dim);
    defer gpa.free(dW_a);
    @memset(dW_a, 0);
    cceBackward(h, w, targets, lse, n, vocab, dim, chunk, .{}, d_h, dW_a);

    const dW_b = try gpa.alloc(f32, vocab * dim);
    defer gpa.free(dW_b);
    @memcpy(dW_b, dW_a);
    cceBackward(h, w, targets, lse, n, vocab, dim, chunk, .{}, d_h, dW_b);

    // dW_b should equal 2 × dW_a (within reduction-order f32 noise)
    for (dW_a, dW_b) |a, b| {
        const expected = 2.0 * a;
        const denom = @max(@abs(expected), 1e-30);
        try testing.expect(@abs(b - expected) / denom < 1e-5);
    }
}

test "Z-loss: forward + backward match materialized reference" {
    const gpa = testing.allocator;
    const n: usize = 4;
    const dim: usize = 32;
    const vocab: usize = 1024;
    const chunk: usize = 256;
    const zlc: f32 = 1e-4; // canonical PaLM-style z-loss scale

    var prng = std.Random.DefaultPrng.init(0xC10_5550);
    const rng = prng.random();

    const h = try gpa.alloc(f32, n * dim);
    defer gpa.free(h);
    const w = try gpa.alloc(f32, vocab * dim);
    defer gpa.free(w);
    const targets = try gpa.alloc(u32, n);
    defer gpa.free(targets);

    for (h) |*v| v.* = (rng.float(f32) - 0.5) * 0.1;
    for (w) |*v| v.* = (rng.float(f32) - 0.5) * 0.1;
    for (targets) |*t| t.* = rng.intRangeLessThan(u32, 0, vocab);

    // ── CCE forward + backward with z-loss enabled.
    const lse_cce = try gpa.alloc(f32, n);
    defer gpa.free(lse_cce);
    const loss_cce = cceForward(h, w, targets, n, vocab, dim, chunk, .{ .z_loss_scale = zlc }, lse_cce);

    const d_h_cce = try gpa.alloc(f32, n * dim);
    defer gpa.free(d_h_cce);
    const dW_cce = try gpa.alloc(f32, vocab * dim);
    defer gpa.free(dW_cce);
    @memset(dW_cce, 0);
    cceBackward(h, w, targets, lse_cce, n, vocab, dim, chunk, .{ .z_loss_scale = zlc }, d_h_cce, dW_cce);

    // ── Reference: materialize logits + add zlc · lse² to per-row loss
    //    and (1 + 2·zlc·lse) · softmax to per-element gradient.
    const logits = try gpa.alloc(f32, n * vocab);
    defer gpa.free(logits);
    refMatmulNt(logits, h, w, n, vocab, dim);

    var ref_total_loss: f64 = 0;
    var ref_lse: [16]f32 = undefined; // n ≤ 16 in this test
    std.debug.assert(n <= ref_lse.len);
    for (0..n) |p| {
        const off = p * vocab;
        var max_z: f32 = -std.math.inf(f32);
        for (0..vocab) |o| max_z = @max(max_z, logits[off + o]);
        var sum_e: f64 = 0;
        for (0..vocab) |o| sum_e += @exp(@as(f64, logits[off + o]) - @as(f64, max_z));
        const lse_v: f32 = @floatCast(@log(sum_e) + @as(f64, max_z));
        ref_lse[p] = lse_v;
        const tgt: usize = @intCast(targets[p]);
        const ce_loss: f64 = @as(f64, lse_v) - @as(f64, logits[off + tgt]);
        const z_pen: f64 = @as(f64, zlc) * @as(f64, lse_v) * @as(f64, lse_v);
        ref_total_loss += ce_loss + z_pen;
    }
    const ref_loss: f32 = @floatCast(ref_total_loss / @as(f64, @floatFromInt(n)));

    // Reference d_logits with z-loss factor: dz_v = (1 + 2·zlc·lse)·softmax_v − δ_{v,t}, /N
    const d_logits = try gpa.alloc(f32, n * vocab);
    defer gpa.free(d_logits);
    const inv_n: f32 = 1.0 / @as(f32, @floatFromInt(n));
    for (0..n) |p| {
        const off = p * vocab;
        const lse_v = ref_lse[p];
        const sm_factor: f32 = 1.0 + 2.0 * zlc * lse_v;
        const tgt: usize = @intCast(targets[p]);
        for (0..vocab) |o| {
            const sm: f32 = @floatCast(@exp(@as(f64, logits[off + o]) - @as(f64, lse_v)));
            const t_o: f32 = if (o == tgt) 1.0 else 0.0;
            d_logits[off + o] = (sm_factor * sm - t_o) * inv_n;
        }
    }

    // Reference d_h, dW via materialized linear backward.
    const d_h_ref = try gpa.alloc(f32, n * dim);
    defer gpa.free(d_h_ref);
    const dW_ref = try gpa.alloc(f32, vocab * dim);
    defer gpa.free(dW_ref);
    @memset(dW_ref, 0);
    refLinearBackward(d_logits, h, w, n, vocab, dim, d_h_ref, dW_ref);

    // ── Diff.
    const loss_rel = @abs(loss_cce - ref_loss) / @max(@abs(ref_loss), 1e-30);
    const d_h_rel = maxAbsRelDiff(d_h_ref, d_h_cce);
    const dW_rel = maxAbsRelDiff(dW_ref, dW_cce);
    try testing.expect(loss_rel < 1e-5);
    try testing.expect(d_h_rel < 1e-5);
    try testing.expect(dW_rel < 1e-5);
}

test "Label smoothing + z-loss combined: matches materialized reference" {
    const gpa = testing.allocator;
    const n: usize = 4;
    const dim: usize = 32;
    const vocab: usize = 1024;
    const chunk: usize = 256;
    const opts = CceLossOpts{ .z_loss_scale = 1e-4, .label_smoothing = 0.1 };

    var prng = std.Random.DefaultPrng.init(0x1A8E_5500);
    const rng = prng.random();

    const h = try gpa.alloc(f32, n * dim);
    defer gpa.free(h);
    const w = try gpa.alloc(f32, vocab * dim);
    defer gpa.free(w);
    const targets = try gpa.alloc(u32, n);
    defer gpa.free(targets);

    for (h) |*v| v.* = (rng.float(f32) - 0.5) * 0.1;
    for (w) |*v| v.* = (rng.float(f32) - 0.5) * 0.1;
    for (targets) |*t| t.* = rng.intRangeLessThan(u32, 0, vocab);

    // ── CCE forward + backward with both regularizers active.
    const lse_cce = try gpa.alloc(f32, n);
    defer gpa.free(lse_cce);
    const loss_cce = cceForward(h, w, targets, n, vocab, dim, chunk, opts, lse_cce);

    const d_h_cce = try gpa.alloc(f32, n * dim);
    defer gpa.free(d_h_cce);
    const dW_cce = try gpa.alloc(f32, vocab * dim);
    defer gpa.free(dW_cce);
    @memset(dW_cce, 0);
    cceBackward(h, w, targets, lse_cce, n, vocab, dim, chunk, opts, d_h_cce, dW_cce);

    // ── Reference: materialize logits, compute label-smoothed CE +
    //    z-loss penalty per row, and the per-element gradient
    //      dz_v = (1 + 2·λ_z·lse) · softmax_v − (1−ε) · δ_{v,t} − ε/V
    //    fed through a plain linear backward.
    const logits = try gpa.alloc(f32, n * vocab);
    defer gpa.free(logits);
    refMatmulNt(logits, h, w, n, vocab, dim);

    const eps: f32 = opts.label_smoothing;
    const t_scale: f32 = 1.0 - eps;
    const eps_over_v: f32 = eps / @as(f32, @floatFromInt(vocab));
    const inv_n: f32 = 1.0 / @as(f32, @floatFromInt(n));

    var ref_total_loss: f64 = 0;
    var ref_lse: [16]f32 = undefined;
    std.debug.assert(n <= ref_lse.len);
    for (0..n) |p| {
        const off = p * vocab;
        var max_z: f32 = -std.math.inf(f32);
        for (0..vocab) |o| max_z = @max(max_z, logits[off + o]);
        var sum_e: f64 = 0;
        var sum_z: f64 = 0;
        for (0..vocab) |o| {
            sum_e += @exp(@as(f64, logits[off + o]) - @as(f64, max_z));
            sum_z += @as(f64, logits[off + o]);
        }
        const lse_v: f32 = @floatCast(@log(sum_e) + @as(f64, max_z));
        ref_lse[p] = lse_v;
        const tgt: usize = @intCast(targets[p]);
        const z_mean: f64 = sum_z / @as(f64, @floatFromInt(vocab));
        const ce_smoothed: f64 = @as(f64, lse_v)
            - @as(f64, t_scale) * @as(f64, logits[off + tgt])
            - @as(f64, eps) * z_mean;
        const z_pen: f64 = @as(f64, opts.z_loss_scale) * @as(f64, lse_v) * @as(f64, lse_v);
        ref_total_loss += ce_smoothed + z_pen;
    }
    const ref_loss: f32 = @floatCast(ref_total_loss / @as(f64, @floatFromInt(n)));

    const d_logits = try gpa.alloc(f32, n * vocab);
    defer gpa.free(d_logits);
    for (0..n) |p| {
        const off = p * vocab;
        const lse_v = ref_lse[p];
        const sm_factor: f32 = 1.0 + 2.0 * opts.z_loss_scale * lse_v;
        const tgt: usize = @intCast(targets[p]);
        for (0..vocab) |o| {
            const sm: f32 = @floatCast(@exp(@as(f64, logits[off + o]) - @as(f64, lse_v)));
            const t_o: f32 = if (o == tgt) t_scale else 0.0;
            d_logits[off + o] = (sm_factor * sm - t_o - eps_over_v) * inv_n;
        }
    }

    const d_h_ref = try gpa.alloc(f32, n * dim);
    defer gpa.free(d_h_ref);
    const dW_ref = try gpa.alloc(f32, vocab * dim);
    defer gpa.free(dW_ref);
    @memset(dW_ref, 0);
    refLinearBackward(d_logits, h, w, n, vocab, dim, d_h_ref, dW_ref);

    const loss_rel = @abs(loss_cce - ref_loss) / @max(@abs(ref_loss), 1e-30);
    const d_h_rel = maxAbsRelDiff(d_h_ref, d_h_cce);
    const dW_rel = maxAbsRelDiff(dW_ref, dW_cce);
    try testing.expect(loss_rel < 1e-5);
    try testing.expect(d_h_rel < 1e-5);
    try testing.expect(dW_rel < 1e-5);
}
