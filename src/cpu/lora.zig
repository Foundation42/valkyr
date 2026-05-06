//! LoRA (Low-Rank Adaptation) CPU oracle primitives.
//!
//! Adds a trainable low-rank delta to a frozen base weight:
//!
//!     W_eff = W + (α/r) · B·A                (1)
//!
//! where W ∈ R^{N×K} is frozen, A ∈ R^{r×K} and B ∈ R^{N×r} are the
//! trainable factors (rank r ≪ min(N, K)), and α is a scalar scaling
//! constant — α/r is what the optimizer sees as the effective learning
//! rate scale on the delta.
//!
//! Init convention: A ~ N(0, σ²), B = 0. With B = 0 the LoRA branch
//! contributes nothing at step 0 (the network behaves identically to
//! the frozen base). After the first backward, ∇_B is the only path
//! that's non-zero — A's gradient flows through Bᵀ which is zero —
//! so B "opens the gate" before A starts learning. This is the
//! initialization asymmetry that motivates LoRA+ (Hayou et al. 2024,
//! re-derived in Chronicals §5): at init, the optimal learning-rate
//! ratio is η_B = λ · η_A with λ ≈ 16 — the optimizer step on B
//! should be ~16× larger than on A so the two factors land in
//! comparable feature-learning regimes after a few steps.
//!
//! Reference: Hu et al. "LoRA: Low-Rank Adaptation of Large Language
//! Models" (ICLR 2022); LoRA+ derivation in Hayou et al. (ICML 2024)
//! / Chronicals §5 (Theorem 1).
//!
//! ## Math (valkyr's matmul_nt convention: `y = x · Wᵀ`)
//!
//! Forward:
//!   intermediate = x · Aᵀ                                   [M, r]
//!   y_base       = x · Wᵀ                                   [M, N]
//!   y_lora       = intermediate · Bᵀ                        [M, N]
//!   y            = y_base + (α/r) · y_lora                  [M, N]
//!
//! Backward, given upstream gradient dy ∈ [M, N]:
//!   dy_B = dy · B                                           [M, r]   (reused twice)
//!   dx   = dy · W           + (α/r) · dy_B · A              [M, K]
//!   ∇A  += (α/r) · dy_Bᵀ · x                                [r, K]
//!   ∇B  += (α/r) · dyᵀ · intermediate                       [N, r]
//!
//! Note that ∇W is *not* computed — W is frozen. This is where the
//! optimizer-state savings come from: at rank-16 over a 4096×4096
//! base weight, the trainable parameter count drops from 16.8M to
//! 131K (~128×) and AdamW's m,v footprint shrinks correspondingly.
//!
//! ## Parity gate (in-file tests)
//!
//! `loraForward` + `loraBackward` produce mathematically identical
//! outputs to materializing W_eff explicitly and running standard
//! linear forward/backward, with the resulting ∇W_eff decomposed
//! via the chain rule:
//!   ∂W_eff/∂B[n,j] = (α/r) · A[j,:]   →  ∇B = (α/r) · ∇W_eff · Aᵀ
//!   ∂W_eff/∂A[i,k] = (α/r) · B[:,i]   →  ∇A = (α/r) · Bᵀ · ∇W_eff
//!
//! Tests assert agreement to 1e-5 global rel-err (`max|diff| / max|ref|`),
//! same metric used in `cce.zig`.
//!
//! ## Convention
//!
//! - x is `[M, K]` row-major (HF activation layout)
//! - W is `[N, K]` row-major (HF linear weight layout, `y = x · Wᵀ`)
//! - A is `[r, K]` row-major (each row is one rank-r feature direction in K-space)
//! - B is `[N, r]` row-major (each row is one output's mixing of the r features)
//! - `intermediate` (caller-allocated) is `[M, r]` — `x · Aᵀ`, cached for backward
//! - `y` is overwritten with the full output (base + LoRA, no add-in-place)
//! - `dx` is overwritten (matches `linearBackward`'s `dx`)
//! - `dA`, `dB` are accumulated into (caller zeroes between optimizer steps,
//!   matches `linearBackward(dW)` and `cce_backward_dw`)

const std = @import("std");

/// LoRA-augmented linear forward.
///   y = x · Wᵀ + (α/r) · (x · Aᵀ) · Bᵀ
/// `intermediate_out` receives `x · Aᵀ` for the backward pass to reuse.
pub fn loraForward(
    x: []const f32,
    w: []const f32,
    a: []const f32,
    b: []const f32,
    M: usize,
    N: usize,
    K: usize,
    r: usize,
    alpha_over_r: f32,
    y_out: []f32,
    intermediate_out: []f32,
) void {
    std.debug.assert(x.len == M * K);
    std.debug.assert(w.len == N * K);
    std.debug.assert(a.len == r * K);
    std.debug.assert(b.len == N * r);
    std.debug.assert(y_out.len == M * N);
    std.debug.assert(intermediate_out.len == M * r);

    // ── 1. y_base = x · Wᵀ
    for (0..M) |m| {
        for (0..N) |n| {
            var s: f64 = 0;
            for (0..K) |k| s += @as(f64, x[m * K + k]) * @as(f64, w[n * K + k]);
            y_out[m * N + n] = @floatCast(s);
        }
    }

    // ── 2. intermediate = x · Aᵀ
    for (0..M) |m| {
        for (0..r) |ri| {
            var s: f64 = 0;
            for (0..K) |k| s += @as(f64, x[m * K + k]) * @as(f64, a[ri * K + k]);
            intermediate_out[m * r + ri] = @floatCast(s);
        }
    }

    // ── 3. y += (α/r) · intermediate · Bᵀ
    for (0..M) |m| {
        for (0..N) |n| {
            var s: f64 = 0;
            for (0..r) |ri| s += @as(f64, intermediate_out[m * r + ri]) * @as(f64, b[n * r + ri]);
            y_out[m * N + n] += @as(f32, @floatCast(s)) * alpha_over_r;
        }
    }
}

/// LoRA-augmented linear backward.
///   dx ← dy · W + (α/r) · (dy · B) · A          (overwrite)
///   ∇A += (α/r) · (dy · B)ᵀ · x                  (accumulate)
///   ∇B += (α/r) · dyᵀ · intermediate             (accumulate)
/// `intermediate` must be the same buffer that `loraForward` populated.
/// `dx_opt = null` skips the dx computation (when x is the input
/// embedding and no upstream gradient is needed).
pub fn loraBackward(
    dy: []const f32,
    x: []const f32,
    w: []const f32,
    a: []const f32,
    b: []const f32,
    intermediate: []const f32,
    M: usize,
    N: usize,
    K: usize,
    r: usize,
    alpha_over_r: f32,
    dx_opt: ?[]f32,
    dA: []f32,
    dB: []f32,
    gpa: std.mem.Allocator,
) !void {
    std.debug.assert(dy.len == M * N);
    std.debug.assert(x.len == M * K);
    std.debug.assert(w.len == N * K);
    std.debug.assert(a.len == r * K);
    std.debug.assert(b.len == N * r);
    std.debug.assert(intermediate.len == M * r);
    std.debug.assert(dA.len == r * K);
    std.debug.assert(dB.len == N * r);

    // ── Scratch: dy_B = dy · B                   [M, r]
    // Reused for dx (LoRA contribution) and for ∇A.
    const dy_B = try gpa.alloc(f32, M * r);
    defer gpa.free(dy_B);
    for (0..M) |m| {
        for (0..r) |ri| {
            var s: f64 = 0;
            for (0..N) |n| s += @as(f64, dy[m * N + n]) * @as(f64, b[n * r + ri]);
            dy_B[m * r + ri] = @floatCast(s);
        }
    }

    // ── dx[m, k] = Σ_n dy[m, n] · W[n, k] + (α/r) · Σ_ri dy_B[m, ri] · A[ri, k]
    if (dx_opt) |dx| {
        std.debug.assert(dx.len == M * K);
        for (0..M) |m| {
            for (0..K) |k| {
                var s_base: f64 = 0;
                for (0..N) |n| s_base += @as(f64, dy[m * N + n]) * @as(f64, w[n * K + k]);
                var s_lora: f64 = 0;
                for (0..r) |ri| s_lora += @as(f64, dy_B[m * r + ri]) * @as(f64, a[ri * K + k]);
                dx[m * K + k] = @floatCast(s_base + alpha_over_r * s_lora);
            }
        }
    }

    // ── ∇A[ri, k] += (α/r) · Σ_m dy_B[m, ri] · x[m, k]
    for (0..r) |ri| {
        for (0..K) |k| {
            var s: f64 = 0;
            for (0..M) |m| s += @as(f64, dy_B[m * r + ri]) * @as(f64, x[m * K + k]);
            dA[ri * K + k] += alpha_over_r * @as(f32, @floatCast(s));
        }
    }

    // ── ∇B[n, ri] += (α/r) · Σ_m dy[m, n] · intermediate[m, ri]
    for (0..N) |n| {
        for (0..r) |ri| {
            var s: f64 = 0;
            for (0..M) |m| s += @as(f64, dy[m * N + n]) * @as(f64, intermediate[m * r + ri]);
            dB[n * r + ri] += alpha_over_r * @as(f32, @floatCast(s));
        }
    }
}

// ── Tests ──────────────────────────────────────────────────────────
//
// Parity oracle: build W_eff = W + (α/r)·B·A explicitly, run plain
// linear forward + backward against it, decompose ∇W_eff via the
// chain rule into ∇A, ∇B targets, and compare to the LoRA-direct
// outputs. Tolerance is 1e-5 global rel-err (max|diff| / max|ref|).

const testing = std.testing;

fn refMatmulNt(out: []f32, a: []const f32, b: []const f32, m: usize, n: usize, k: usize) void {
    for (0..m) |i| {
        for (0..n) |j| {
            var s: f64 = 0;
            for (0..k) |kk| s += @as(f64, a[i * k + kk]) * @as(f64, b[j * k + kk]);
            out[i * n + j] = @floatCast(s);
        }
    }
}

/// Build W_eff = W + (α/r) · B · A. B is [N, r], A is [r, K], W is [N, K].
fn refMaterializeWeff(
    w_eff: []f32,
    w: []const f32,
    b: []const f32,
    a: []const f32,
    N: usize,
    K: usize,
    r: usize,
    alpha_over_r: f32,
) void {
    @memcpy(w_eff, w);
    for (0..N) |n| {
        for (0..K) |k| {
            var s: f64 = 0;
            for (0..r) |ri| s += @as(f64, b[n * r + ri]) * @as(f64, a[ri * K + k]);
            w_eff[n * K + k] += alpha_over_r * @as(f32, @floatCast(s));
        }
    }
}

/// Reference linear-backward (mirrors train_decoder.linearBackward).
/// dx is overwritten, dW is accumulated.
fn refLinearBackward(
    dout: []const f32,
    x: []const f32,
    w: []const f32,
    M: usize,
    N: usize,
    K: usize,
    dx: []f32,
    dW: []f32,
) void {
    for (0..M) |m| {
        for (0..K) |k| {
            var s: f64 = 0;
            for (0..N) |n| s += @as(f64, dout[m * N + n]) * @as(f64, w[n * K + k]);
            dx[m * K + k] = @floatCast(s);
        }
    }
    for (0..N) |n| {
        for (0..K) |k| {
            var s: f64 = 0;
            for (0..M) |m| s += @as(f64, dout[m * N + n]) * @as(f64, x[m * K + k]);
            dW[n * K + k] += @floatCast(s);
        }
    }
}

fn globalRelDiff(a: []const f32, b: []const f32) f32 {
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

test "LoRA forward with B=0 equals plain linear" {
    const gpa = testing.allocator;
    const M: usize = 4;
    const N: usize = 8;
    const K: usize = 16;
    const r: usize = 4;
    const aor: f32 = 2.0; // α/r = 2 (typical LoRA scaling)

    var prng = std.Random.DefaultPrng.init(0x10AA1);
    const rng = prng.random();

    const x = try gpa.alloc(f32, M * K);
    defer gpa.free(x);
    const w = try gpa.alloc(f32, N * K);
    defer gpa.free(w);
    const a = try gpa.alloc(f32, r * K);
    defer gpa.free(a);
    const b = try gpa.alloc(f32, N * r);
    defer gpa.free(b);

    for (x) |*v| v.* = (rng.float(f32) - 0.5) * 0.5;
    for (w) |*v| v.* = (rng.float(f32) - 0.5) * 0.3;
    for (a) |*v| v.* = (rng.float(f32) - 0.5) * 0.2;
    @memset(b, 0); // B = 0 init

    const y_lora = try gpa.alloc(f32, M * N);
    defer gpa.free(y_lora);
    const intermediate = try gpa.alloc(f32, M * r);
    defer gpa.free(intermediate);
    loraForward(x, w, a, b, M, N, K, r, aor, y_lora, intermediate);

    const y_base = try gpa.alloc(f32, M * N);
    defer gpa.free(y_base);
    refMatmulNt(y_base, x, w, M, N, K);

    try testing.expect(globalRelDiff(y_base, y_lora) < 1e-5);
}

test "LoRA forward equals materialized W_eff path" {
    const gpa = testing.allocator;
    const M: usize = 4;
    const N: usize = 12;
    const K: usize = 32;
    const r: usize = 4;
    const aor: f32 = 2.0;

    var prng = std.Random.DefaultPrng.init(0x10AA2);
    const rng = prng.random();

    const x = try gpa.alloc(f32, M * K);
    defer gpa.free(x);
    const w = try gpa.alloc(f32, N * K);
    defer gpa.free(w);
    const a = try gpa.alloc(f32, r * K);
    defer gpa.free(a);
    const b = try gpa.alloc(f32, N * r);
    defer gpa.free(b);

    for (x) |*v| v.* = (rng.float(f32) - 0.5) * 0.5;
    for (w) |*v| v.* = (rng.float(f32) - 0.5) * 0.3;
    for (a) |*v| v.* = (rng.float(f32) - 0.5) * 0.2;
    for (b) |*v| v.* = (rng.float(f32) - 0.5) * 0.2;

    // LoRA path
    const y_lora = try gpa.alloc(f32, M * N);
    defer gpa.free(y_lora);
    const intermediate = try gpa.alloc(f32, M * r);
    defer gpa.free(intermediate);
    loraForward(x, w, a, b, M, N, K, r, aor, y_lora, intermediate);

    // Reference: materialize W_eff and run plain matmul
    const w_eff = try gpa.alloc(f32, N * K);
    defer gpa.free(w_eff);
    refMaterializeWeff(w_eff, w, b, a, N, K, r, aor);
    const y_eff = try gpa.alloc(f32, M * N);
    defer gpa.free(y_eff);
    refMatmulNt(y_eff, x, w_eff, M, N, K);

    try testing.expect(globalRelDiff(y_eff, y_lora) < 1e-5);
}

test "LoRA backward equals decomposition of materialized W_eff backward" {
    const gpa = testing.allocator;
    const M: usize = 6;
    const N: usize = 16;
    const K: usize = 24;
    const r: usize = 8;
    const aor: f32 = 2.0;

    var prng = std.Random.DefaultPrng.init(0x10AA3);
    const rng = prng.random();

    const x = try gpa.alloc(f32, M * K);
    defer gpa.free(x);
    const w = try gpa.alloc(f32, N * K);
    defer gpa.free(w);
    const a = try gpa.alloc(f32, r * K);
    defer gpa.free(a);
    const b = try gpa.alloc(f32, N * r);
    defer gpa.free(b);
    const dy = try gpa.alloc(f32, M * N);
    defer gpa.free(dy);

    for (x) |*v| v.* = (rng.float(f32) - 0.5) * 0.5;
    for (w) |*v| v.* = (rng.float(f32) - 0.5) * 0.3;
    for (a) |*v| v.* = (rng.float(f32) - 0.5) * 0.2;
    for (b) |*v| v.* = (rng.float(f32) - 0.5) * 0.2;
    for (dy) |*v| v.* = (rng.float(f32) - 0.5) * 0.4;

    // ── LoRA-direct path
    const y_lora = try gpa.alloc(f32, M * N);
    defer gpa.free(y_lora);
    const intermediate = try gpa.alloc(f32, M * r);
    defer gpa.free(intermediate);
    loraForward(x, w, a, b, M, N, K, r, aor, y_lora, intermediate);

    const dx_lora = try gpa.alloc(f32, M * K);
    defer gpa.free(dx_lora);
    const dA_lora = try gpa.alloc(f32, r * K);
    defer gpa.free(dA_lora);
    const dB_lora = try gpa.alloc(f32, N * r);
    defer gpa.free(dB_lora);
    @memset(dA_lora, 0);
    @memset(dB_lora, 0);
    try loraBackward(dy, x, w, a, b, intermediate, M, N, K, r, aor, dx_lora, dA_lora, dB_lora, gpa);

    // ── Reference: materialize W_eff, run plain linear backward, then
    //    decompose ∇W_eff via the chain rule:
    //    ∇A_ref = (α/r) · Bᵀ · ∇W_eff      [r, K]
    //    ∇B_ref = (α/r) · ∇W_eff · Aᵀ      [N, r]
    const w_eff = try gpa.alloc(f32, N * K);
    defer gpa.free(w_eff);
    refMaterializeWeff(w_eff, w, b, a, N, K, r, aor);

    const dx_ref = try gpa.alloc(f32, M * K);
    defer gpa.free(dx_ref);
    const dW_eff = try gpa.alloc(f32, N * K);
    defer gpa.free(dW_eff);
    @memset(dW_eff, 0);
    refLinearBackward(dy, x, w_eff, M, N, K, dx_ref, dW_eff);

    // ∇A_ref[ri, k] = (α/r) · Σ_n B[n, ri] · ∇W_eff[n, k]
    const dA_ref = try gpa.alloc(f32, r * K);
    defer gpa.free(dA_ref);
    for (0..r) |ri| {
        for (0..K) |k| {
            var s: f64 = 0;
            for (0..N) |n| s += @as(f64, b[n * r + ri]) * @as(f64, dW_eff[n * K + k]);
            dA_ref[ri * K + k] = aor * @as(f32, @floatCast(s));
        }
    }

    // ∇B_ref[n, ri] = (α/r) · Σ_k ∇W_eff[n, k] · A[ri, k]
    const dB_ref = try gpa.alloc(f32, N * r);
    defer gpa.free(dB_ref);
    for (0..N) |n| {
        for (0..r) |ri| {
            var s: f64 = 0;
            for (0..K) |k| s += @as(f64, dW_eff[n * K + k]) * @as(f64, a[ri * K + k]);
            dB_ref[n * r + ri] = aor * @as(f32, @floatCast(s));
        }
    }

    try testing.expect(globalRelDiff(dx_ref, dx_lora) < 1e-5);
    try testing.expect(globalRelDiff(dA_ref, dA_lora) < 1e-5);
    try testing.expect(globalRelDiff(dB_ref, dB_lora) < 1e-5);
}

test "LoRA backward at B=0 init: dA == 0, dB nonzero" {
    const gpa = testing.allocator;
    const M: usize = 4;
    const N: usize = 8;
    const K: usize = 16;
    const r: usize = 4;
    const aor: f32 = 2.0;

    var prng = std.Random.DefaultPrng.init(0x10AA4);
    const rng = prng.random();

    const x = try gpa.alloc(f32, M * K);
    defer gpa.free(x);
    const w = try gpa.alloc(f32, N * K);
    defer gpa.free(w);
    const a = try gpa.alloc(f32, r * K);
    defer gpa.free(a);
    const b = try gpa.alloc(f32, N * r);
    defer gpa.free(b);
    const dy = try gpa.alloc(f32, M * N);
    defer gpa.free(dy);

    for (x) |*v| v.* = (rng.float(f32) - 0.5) * 0.5;
    for (w) |*v| v.* = (rng.float(f32) - 0.5) * 0.3;
    for (a) |*v| v.* = (rng.float(f32) - 0.5) * 0.2;
    @memset(b, 0); // B = 0 init
    for (dy) |*v| v.* = (rng.float(f32) - 0.5) * 0.4;

    const y = try gpa.alloc(f32, M * N);
    defer gpa.free(y);
    const intermediate = try gpa.alloc(f32, M * r);
    defer gpa.free(intermediate);
    loraForward(x, w, a, b, M, N, K, r, aor, y, intermediate);

    const dx = try gpa.alloc(f32, M * K);
    defer gpa.free(dx);
    const dA = try gpa.alloc(f32, r * K);
    defer gpa.free(dA);
    const dB = try gpa.alloc(f32, N * r);
    defer gpa.free(dB);
    @memset(dA, 0);
    @memset(dB, 0);
    try loraBackward(dy, x, w, a, b, intermediate, M, N, K, r, aor, dx, dA, dB, gpa);

    // dA should be exactly zero — its only contribution is via Bᵀ which is 0.
    for (dA) |v| try testing.expectEqual(@as(f32, 0), v);

    // dB should be non-trivial — the only path that's "open" at B=0 init.
    var any_nonzero = false;
    for (dB) |v| {
        if (v != 0) {
            any_nonzero = true;
            break;
        }
    }
    try testing.expect(any_nonzero);
}

test "LoRA backward dA, dB accumulate (don't overwrite)" {
    const gpa = testing.allocator;
    const M: usize = 3;
    const N: usize = 6;
    const K: usize = 8;
    const r: usize = 2;
    const aor: f32 = 1.5;

    var prng = std.Random.DefaultPrng.init(0x10AA5);
    const rng = prng.random();

    const x = try gpa.alloc(f32, M * K);
    defer gpa.free(x);
    const w = try gpa.alloc(f32, N * K);
    defer gpa.free(w);
    const a = try gpa.alloc(f32, r * K);
    defer gpa.free(a);
    const b = try gpa.alloc(f32, N * r);
    defer gpa.free(b);
    const dy = try gpa.alloc(f32, M * N);
    defer gpa.free(dy);

    for (x) |*v| v.* = (rng.float(f32) - 0.5);
    for (w) |*v| v.* = (rng.float(f32) - 0.5);
    for (a) |*v| v.* = (rng.float(f32) - 0.5);
    for (b) |*v| v.* = (rng.float(f32) - 0.5);
    for (dy) |*v| v.* = (rng.float(f32) - 0.5);

    const y = try gpa.alloc(f32, M * N);
    defer gpa.free(y);
    const intermediate = try gpa.alloc(f32, M * r);
    defer gpa.free(intermediate);
    loraForward(x, w, a, b, M, N, K, r, aor, y, intermediate);

    const dx = try gpa.alloc(f32, M * K);
    defer gpa.free(dx);
    const dA1 = try gpa.alloc(f32, r * K);
    defer gpa.free(dA1);
    const dB1 = try gpa.alloc(f32, N * r);
    defer gpa.free(dB1);
    @memset(dA1, 0);
    @memset(dB1, 0);
    try loraBackward(dy, x, w, a, b, intermediate, M, N, K, r, aor, dx, dA1, dB1, gpa);

    // Run again, accumulating into a copy → expect 2× the values.
    const dA2 = try gpa.alloc(f32, r * K);
    defer gpa.free(dA2);
    const dB2 = try gpa.alloc(f32, N * r);
    defer gpa.free(dB2);
    @memcpy(dA2, dA1);
    @memcpy(dB2, dB1);
    try loraBackward(dy, x, w, a, b, intermediate, M, N, K, r, aor, dx, dA2, dB2, gpa);

    for (dA1, dA2) |a1, a2| {
        const expected = 2.0 * a1;
        const denom = @max(@abs(expected), 1e-30);
        try testing.expect(@abs(a2 - expected) / denom < 1e-5);
    }
    for (dB1, dB2) |b1, b2| {
        const expected = 2.0 * b1;
        const denom = @max(@abs(expected), 1e-30);
        try testing.expect(@abs(b2 - expected) / denom < 1e-5);
    }
}
