//! End-to-end toy decoder layer for Tier-2 chunk 8a.
//!
//! Wires the transformer-primitive backward passes shipped in chunks
//! 1-7 into a single decoder block:
//!
//!     x_in  [n_pos, dim]
//!     n1   = RMSNorm(x_in, w_n1)
//!     Q,K,V = linear projections of n1                      [n_pos, dim]
//!     attn = SDPA(Q, K, V, causal)                          [n_pos, dim]
//!     o    = linear(attn, W_O)
//!     mid  = x_in + o                                        ← residual
//!     n2   = RMSNorm(mid, w_n2)
//!     ff_h = ReLU(linear(n2, W_FF1))                        [n_pos, ff_dim]
//!     y    = mid + linear(ff_h, W_FF2)                       ← residual
//!
//! No RoPE — the chunk-7 backward smoke already covers it; bringing it
//! in here just adds bookkeeping without exercising new gradient flow.
//! No bias terms — convention matches Llama / Gemma.
//!
//! Loss is mean-squared-error against a fixed target. The smoke trains
//! Q/K/V/O/FF1/FF2 + the two RMSNorm gains via Adam and asserts loss
//! decreases monotonically (or near-monotonically with a tolerance).
//!
//! All linear projections are `out = x @ Wᵀ` with `W` shaped `[N, K]` —
//! same convention as `cpu_math.matmul_nt`. Backward of a linear layer
//! is therefore:
//!     dx     = dout @ W
//!     dW    += doutᵀ @ x          (accumulated; caller zeroes between steps)

const std = @import("std");
const tt = @import("train_transformer.zig");

/// Local fp32 matmul: `out = x @ Wᵀ` where `W` is `[N, K]`,
/// `x` is `[M, K]`, `out` is `[M, N]`. Mirrors the layout convention of
/// `cpu_math.matmul_nt` but takes plain slices (the smoke owns its own
/// fp32 buffers — no Tensor wrapping needed).
fn matmulNt(out: []f32, x: []const f32, W: []const f32, M: usize, N: usize, K: usize) void {
    std.debug.assert(x.len == M * K);
    std.debug.assert(W.len == N * K);
    std.debug.assert(out.len == M * N);
    for (0..M) |m| {
        const x_off = m * K;
        for (0..N) |n| {
            const w_off = n * K;
            var acc: f32 = 0;
            for (0..K) |k| acc += x[x_off + k] * W[w_off + k];
            out[m * N + n] = acc;
        }
    }
}

pub const Config = struct {
    dim: usize,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    ff_dim: usize,
    n_pos: usize,
    rms_eps: f32 = 1e-5,
    causal: bool = true,
};

/// All learnable parameters. `[N, K]` row-major; bias-free.
pub const Layer = struct {
    cfg: Config,
    w_n1: []f32, // [dim]
    w_q: []f32, // [dim, dim]            (n_heads * head_dim, dim)
    w_k: []f32, // [n_kv_heads * head_dim, dim]
    w_v: []f32, // [n_kv_heads * head_dim, dim]
    w_o: []f32, // [dim, n_heads * head_dim]
    w_n2: []f32, // [dim]
    w_ff1: []f32, // [ff_dim, dim]
    w_ff2: []f32, // [dim, ff_dim]

    pub fn paramSlices(self: *Layer) [8][]f32 {
        return .{ self.w_n1, self.w_q, self.w_k, self.w_v, self.w_o, self.w_n2, self.w_ff1, self.w_ff2 };
    }
};

pub const Grads = struct {
    dw_n1: []f32,
    dw_q: []f32,
    dw_k: []f32,
    dw_v: []f32,
    dw_o: []f32,
    dw_n2: []f32,
    dw_ff1: []f32,
    dw_ff2: []f32,

    pub fn slices(self: *Grads) [8][]f32 {
        return .{ self.dw_n1, self.dw_q, self.dw_k, self.dw_v, self.dw_o, self.dw_n2, self.dw_ff1, self.dw_ff2 };
    }

    pub fn zero(self: *Grads) void {
        for (self.slices()) |s| @memset(s, 0.0);
    }
};

/// Saved activations needed for backward. Sizes are derived from cfg.
pub const Acts = struct {
    x_in: []f32, // [n_pos, dim]
    n1: []f32, // [n_pos, dim]
    q: []f32, // [n_pos, n_heads * head_dim]
    k: []f32, // [n_pos, n_kv_heads * head_dim]
    v: []f32, // [n_pos, n_kv_heads * head_dim]
    scores: []f32, // [n_pos, n_heads, n_pos]   (n_q == n_kv == n_pos)
    attn: []f32, // same shape as scores
    attn_out: []f32, // [n_pos, dim]
    o: []f32, // [n_pos, dim]
    mid: []f32, // [n_pos, dim] = x_in + o
    n2: []f32, // [n_pos, dim]
    ff_pre: []f32, // [n_pos, ff_dim]
    ff_h: []f32, // [n_pos, ff_dim] = ReLU(ff_pre)
    ff_out: []f32, // [n_pos, dim]
    y: []f32, // [n_pos, dim] = mid + ff_out
};

pub fn allocActs(gpa: std.mem.Allocator, cfg: Config) !Acts {
    const dim = cfg.dim;
    const n_pos = cfg.n_pos;
    const q_dim = cfg.n_heads * cfg.head_dim;
    const kv_dim = cfg.n_kv_heads * cfg.head_dim;
    const scores_total = n_pos * cfg.n_heads * n_pos;
    return Acts{
        .x_in = try gpa.alloc(f32, n_pos * dim),
        .n1 = try gpa.alloc(f32, n_pos * dim),
        .q = try gpa.alloc(f32, n_pos * q_dim),
        .k = try gpa.alloc(f32, n_pos * kv_dim),
        .v = try gpa.alloc(f32, n_pos * kv_dim),
        .scores = try gpa.alloc(f32, scores_total),
        .attn = try gpa.alloc(f32, scores_total),
        .attn_out = try gpa.alloc(f32, n_pos * q_dim),
        .o = try gpa.alloc(f32, n_pos * dim),
        .mid = try gpa.alloc(f32, n_pos * dim),
        .n2 = try gpa.alloc(f32, n_pos * dim),
        .ff_pre = try gpa.alloc(f32, n_pos * cfg.ff_dim),
        .ff_h = try gpa.alloc(f32, n_pos * cfg.ff_dim),
        .ff_out = try gpa.alloc(f32, n_pos * dim),
        .y = try gpa.alloc(f32, n_pos * dim),
    };
}

pub fn freeActs(gpa: std.mem.Allocator, a: *Acts) void {
    gpa.free(a.x_in);
    gpa.free(a.n1);
    gpa.free(a.q);
    gpa.free(a.k);
    gpa.free(a.v);
    gpa.free(a.scores);
    gpa.free(a.attn);
    gpa.free(a.attn_out);
    gpa.free(a.o);
    gpa.free(a.mid);
    gpa.free(a.n2);
    gpa.free(a.ff_pre);
    gpa.free(a.ff_h);
    gpa.free(a.ff_out);
    gpa.free(a.y);
}

/// Forward pass. `acts.x_in` must be pre-filled by the caller; this
/// fills the rest.
pub fn forward(layer: *const Layer, acts: *Acts) void {
    const cfg = layer.cfg;
    const dim = cfg.dim;
    const n_pos = cfg.n_pos;
    const q_dim = cfg.n_heads * cfg.head_dim;
    const kv_dim = cfg.n_kv_heads * cfg.head_dim;

    // n1 = RMSNorm(x_in, w_n1) — per-row.
    tt.rmsNormForward(acts.x_in, layer.w_n1, cfg.rms_eps, false, n_pos, acts.n1);

    // Q, K, V projections
    matmulNt(acts.q, acts.n1, layer.w_q, n_pos, q_dim, dim);
    matmulNt(acts.k, acts.n1, layer.w_k, n_pos, kv_dim, dim);
    matmulNt(acts.v, acts.n1, layer.w_v, n_pos, kv_dim, dim);

    // SDPA
    tt.attentionForward(
        acts.q,
        acts.k,
        acts.v,
        n_pos,
        n_pos,
        cfg.n_heads,
        cfg.n_kv_heads,
        cfg.head_dim,
        cfg.causal,
        acts.scores,
        acts.attn,
        acts.attn_out,
    );

    // o = attn_out @ W_Oᵀ
    matmulNt(acts.o, acts.attn_out, layer.w_o, n_pos, dim, q_dim);

    // mid = x_in + o (residual)
    for (acts.mid, acts.x_in, acts.o) |*m, x, o_v| m.* = x + o_v;

    // n2 = RMSNorm(mid, w_n2)
    tt.rmsNormForward(acts.mid, layer.w_n2, cfg.rms_eps, false, n_pos, acts.n2);

    // ff_pre = n2 @ W_FF1ᵀ
    matmulNt(acts.ff_pre, acts.n2, layer.w_ff1, n_pos, cfg.ff_dim, dim);

    // ff_h = ReLU(ff_pre)
    for (acts.ff_h, acts.ff_pre) |*h, p| h.* = if (p > 0) p else 0;

    // ff_out = ff_h @ W_FF2ᵀ
    matmulNt(acts.ff_out, acts.ff_h, layer.w_ff2, n_pos, dim, cfg.ff_dim);

    // y = mid + ff_out (residual)
    for (acts.y, acts.mid, acts.ff_out) |*yv, m, f| yv.* = m + f;
}

/// Mean-squared-error loss.
pub fn mseLoss(y: []const f32, target: []const f32) f32 {
    std.debug.assert(y.len == target.len);
    var s: f64 = 0;
    for (y, target) |yv, tv| {
        const d = @as(f64, yv) - @as(f64, tv);
        s += d * d;
    }
    return @floatCast(s / @as(f64, @floatFromInt(y.len)));
}

inline fn mseLossGrad(dy: []f32, y: []const f32, target: []const f32) void {
    const inv_n: f32 = 2.0 / @as(f32, @floatFromInt(y.len));
    for (dy, y, target) |*d, yv, tv| d.* = inv_n * (yv - tv);
}

/// Linear backward helper for `out = x @ Wᵀ` where W is `[N, K]`,
/// x is `[M, K]`, out is `[M, N]`. Computes:
///     dx[M, K]   = dout @ W           (if dx != null)
///     dW[N, K]  += doutᵀ @ x
inline fn linearBackward(
    dout: []const f32,
    x: []const f32,
    W: []const f32,
    M: usize,
    N: usize,
    K: usize,
    dx_opt: ?[]f32,
    dW: []f32,
) void {
    std.debug.assert(dout.len == M * N);
    std.debug.assert(x.len == M * K);
    std.debug.assert(W.len == N * K);
    std.debug.assert(dW.len == N * K);

    // dx[m, k] = Σ_n dout[m, n] · W[n, k]
    if (dx_opt) |dx| {
        std.debug.assert(dx.len == M * K);
        for (0..M) |m| {
            for (0..K) |k| {
                var s: f64 = 0;
                for (0..N) |n| {
                    s += @as(f64, dout[m * N + n]) * @as(f64, W[n * K + k]);
                }
                dx[m * K + k] = @floatCast(s);
            }
        }
    }
    // dW[n, k] += Σ_m dout[m, n] · x[m, k]
    for (0..N) |n| {
        for (0..K) |k| {
            var s: f64 = 0;
            for (0..M) |m| {
                s += @as(f64, dout[m * N + n]) * @as(f64, x[m * K + k]);
            }
            dW[n * K + k] += @floatCast(s);
        }
    }
}

/// Full backward. `grads` is *accumulated* into — caller zeroes
/// between optimizer steps via `grads.zero()`.
pub fn backward(
    gpa: std.mem.Allocator,
    layer: *const Layer,
    acts: *const Acts,
    target: []const f32,
    grads: *Grads,
) !void {
    const cfg = layer.cfg;
    const dim = cfg.dim;
    const n_pos = cfg.n_pos;
    const q_dim = cfg.n_heads * cfg.head_dim;
    const kv_dim = cfg.n_kv_heads * cfg.head_dim;

    // ── 1. d_y = (2/N) (y - target)
    const d_y = try gpa.alloc(f32, n_pos * dim);
    defer gpa.free(d_y);
    mseLossGrad(d_y, acts.y, target);

    // ── 2. y = mid + ff_out  →  d_mid = d_y;  d_ff_out = d_y.
    const d_mid = try gpa.alloc(f32, n_pos * dim);
    defer gpa.free(d_mid);
    @memcpy(d_mid, d_y);
    const d_ff_out = d_y; // alias — d_ff_out and d_y have the same value

    // ── 3. ff_out = ff_h @ W_FF2ᵀ
    const d_ff_h = try gpa.alloc(f32, n_pos * cfg.ff_dim);
    defer gpa.free(d_ff_h);
    linearBackward(d_ff_out, acts.ff_h, layer.w_ff2, n_pos, dim, cfg.ff_dim, d_ff_h, grads.dw_ff2);

    // ── 4. ff_h = ReLU(ff_pre)  →  d_ff_pre = d_ff_h * (ff_pre > 0)
    const d_ff_pre = try gpa.alloc(f32, n_pos * cfg.ff_dim);
    defer gpa.free(d_ff_pre);
    for (d_ff_pre, d_ff_h, acts.ff_pre) |*dp, dh, pre| dp.* = if (pre > 0) dh else 0;

    // ── 5. ff_pre = n2 @ W_FF1ᵀ
    const d_n2 = try gpa.alloc(f32, n_pos * dim);
    defer gpa.free(d_n2);
    linearBackward(d_ff_pre, acts.n2, layer.w_ff1, n_pos, cfg.ff_dim, dim, d_n2, grads.dw_ff1);

    // ── 6. n2 = RMSNorm(mid, w_n2)  →  d_mid += rmsnorm.dx;  dw_n2 += rmsnorm.dw
    const d_mid_norm = try gpa.alloc(f32, n_pos * dim);
    defer gpa.free(d_mid_norm);
    tt.rmsNormBackward(d_n2, acts.mid, layer.w_n2, cfg.rms_eps, false, n_pos, d_mid_norm, grads.dw_n2);
    for (d_mid, d_mid_norm) |*m, dn| m.* += dn;

    // ── 7. mid = x_in + o  →  d_o = d_mid (and d_x_in path discarded)
    const d_o = d_mid; // we only need d_o from here on; alias.

    // ── 8. o = attn_out @ W_Oᵀ
    const d_attn_out = try gpa.alloc(f32, n_pos * q_dim);
    defer gpa.free(d_attn_out);
    linearBackward(d_o, acts.attn_out, layer.w_o, n_pos, dim, q_dim, d_attn_out, grads.dw_o);

    // ── 9. SDPA backward.  attentionBackward writes into dQ/dK/dV
    //    (overwrites). Layout: [n_pos, n_heads, head_dim]. The Q/K/V
    //    projection outputs already use that layout (`q_dim == dim`,
    //    `kv_dim == n_kv_heads * head_dim`), so no reshape needed.
    const scores_total = n_pos * cfg.n_heads * n_pos;
    const d_scores_scratch = try gpa.alloc(f32, scores_total);
    defer gpa.free(d_scores_scratch);
    const dQ = try gpa.alloc(f32, n_pos * q_dim);
    defer gpa.free(dQ);
    const dK = try gpa.alloc(f32, n_pos * kv_dim);
    defer gpa.free(dK);
    const dV = try gpa.alloc(f32, n_pos * kv_dim);
    defer gpa.free(dV);

    tt.attentionBackward(
        d_attn_out,
        acts.q,
        acts.k,
        acts.v,
        acts.attn,
        n_pos,
        n_pos,
        cfg.n_heads,
        cfg.n_kv_heads,
        cfg.head_dim,
        cfg.causal,
        d_scores_scratch,
        dQ,
        dK,
        dV,
    );

    // ── 10. Q = n1 @ W_Qᵀ etc.   Accumulate into dW_Q/dW_K/dW_V; sum
    //    contributions from Q/K/V into d_n1.
    const d_n1 = try gpa.alloc(f32, n_pos * dim);
    defer gpa.free(d_n1);
    @memset(d_n1, 0);
    const d_n1_q = try gpa.alloc(f32, n_pos * dim);
    defer gpa.free(d_n1_q);
    const d_n1_k = try gpa.alloc(f32, n_pos * dim);
    defer gpa.free(d_n1_k);
    const d_n1_v = try gpa.alloc(f32, n_pos * dim);
    defer gpa.free(d_n1_v);

    linearBackward(dQ, acts.n1, layer.w_q, n_pos, q_dim, dim, d_n1_q, grads.dw_q);
    linearBackward(dK, acts.n1, layer.w_k, n_pos, kv_dim, dim, d_n1_k, grads.dw_k);
    linearBackward(dV, acts.n1, layer.w_v, n_pos, kv_dim, dim, d_n1_v, grads.dw_v);
    for (d_n1, d_n1_q, d_n1_k, d_n1_v) |*d, a, b, c| d.* = a + b + c;

    // ── 11. n1 = RMSNorm(x_in, w_n1)  →  dw_n1 += rmsnorm.dw  (dx
    //    discarded — x_in is the input to the layer and not a learnable
    //    parameter in this smoke).
    const d_x_in_dummy = try gpa.alloc(f32, n_pos * dim);
    defer gpa.free(d_x_in_dummy);
    tt.rmsNormBackward(d_n1, acts.x_in, layer.w_n1, cfg.rms_eps, false, n_pos, d_x_in_dummy, grads.dw_n1);
}

// ── Adam optimizer ─────────────────────────────────────────────────

pub const AdamState = struct {
    m: [8][]f32,
    v: [8][]f32,
    t: u32 = 0,
    lr: f32,
    beta1: f32 = 0.9,
    beta2: f32 = 0.999,
    eps: f32 = 1e-8,

    pub fn init(gpa: std.mem.Allocator, layer: *Layer, lr: f32) !AdamState {
        var m: [8][]f32 = undefined;
        var v: [8][]f32 = undefined;
        const params = layer.paramSlices();
        for (params, 0..) |p, i| {
            m[i] = try gpa.alloc(f32, p.len);
            v[i] = try gpa.alloc(f32, p.len);
            @memset(m[i], 0);
            @memset(v[i], 0);
        }
        return .{ .m = m, .v = v, .lr = lr };
    }

    pub fn deinit(self: *AdamState, gpa: std.mem.Allocator) void {
        for (self.m) |s| gpa.free(s);
        for (self.v) |s| gpa.free(s);
    }
};

pub fn adamStep(state: *AdamState, layer: *Layer, grads: *Grads) void {
    state.t += 1;
    const t_f: f32 = @floatFromInt(state.t);
    const bc1: f32 = 1.0 - std.math.pow(f32, state.beta1, t_f);
    const bc2: f32 = 1.0 - std.math.pow(f32, state.beta2, t_f);
    const lr_t: f32 = state.lr * @sqrt(bc2) / bc1;

    const params = layer.paramSlices();
    const grad_slices = grads.slices();
    inline for (0..8) |i| {
        const p = params[i];
        const g = grad_slices[i];
        const m = state.m[i];
        const v = state.v[i];
        for (p, g, m, v) |*pi, gi, *mi, *vi| {
            mi.* = state.beta1 * mi.* + (1.0 - state.beta1) * gi;
            vi.* = state.beta2 * vi.* + (1.0 - state.beta2) * gi * gi;
            pi.* -= lr_t * mi.* / (@sqrt(vi.*) + state.eps);
        }
    }
}
