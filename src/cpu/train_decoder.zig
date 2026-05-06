//! End-to-end toy decoder layer for Tier-2 chunk 8a.
//!
//! Wires the transformer-primitive backward passes shipped in chunks
//! 1-7 (plus the SwiGLU primitive from β-3a-1) into a single decoder
//! block:
//!
//!     x_in     [n_pos, dim]
//!     n1       = RMSNorm(x_in, w_n1)
//!     Q,K,V    = linear projections of n1                   [n_pos, dim]
//!     attn     = SDPA(Q, K, V, causal)                      [n_pos, dim]
//!     o        = linear(attn, W_O)
//!     mid      = x_in + o                                    ← residual
//!     n2       = RMSNorm(mid, w_n2)
//!     pre_gate = linear(n2, W_gate)                          [n_pos, ff_dim]
//!     up       = linear(n2, W_up)                            [n_pos, ff_dim]
//!     gated    = silu(pre_gate) · up                         [n_pos, ff_dim]
//!     y        = mid + linear(gated, W_down)                 ← residual
//!
//! SwiGLU FFN replaces the ReLU FFN of chunk-8a. Three weight tensors
//! per layer instead of two; the gate's nonlinearity is silu (z·σ(z)).
//! Saved activations grow by one buffer (`up`); `gated` is recomputed
//! during backward rather than stored.
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
    /// RoPE rotation dim. 0 disables RoPE entirely (pre-β-3a-3 behaviour);
    /// `head_dim` gives full RoPE; smaller values give Qwen3.5-style
    /// partial rotation.
    rotary_dim: usize = 0,
    rope_theta: f32 = 10_000.0,
};

/// All learnable parameters. `[N, K]` row-major; bias-free.
pub const Layer = struct {
    cfg: Config,
    w_n1: []f32, // [dim]
    w_q: []f32, // [n_heads * head_dim, dim]
    w_k: []f32, // [n_kv_heads * head_dim, dim]
    w_v: []f32, // [n_kv_heads * head_dim, dim]
    w_o: []f32, // [dim, n_heads * head_dim]
    w_n2: []f32, // [dim]
    w_gate: []f32, // [ff_dim, dim]
    w_up: []f32, // [ff_dim, dim]
    w_down: []f32, // [dim, ff_dim]

    pub fn paramSlices(self: *Layer) [9][]f32 {
        return .{ self.w_n1, self.w_q, self.w_k, self.w_v, self.w_o, self.w_n2, self.w_gate, self.w_up, self.w_down };
    }
};

pub const Grads = struct {
    dw_n1: []f32,
    dw_q: []f32,
    dw_k: []f32,
    dw_v: []f32,
    dw_o: []f32,
    dw_n2: []f32,
    dw_gate: []f32,
    dw_up: []f32,
    dw_down: []f32,

    pub fn slices(self: *Grads) [9][]f32 {
        return .{ self.dw_n1, self.dw_q, self.dw_k, self.dw_v, self.dw_o, self.dw_n2, self.dw_gate, self.dw_up, self.dw_down };
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
    pre_gate: []f32, // [n_pos, ff_dim]
    up: []f32, // [n_pos, ff_dim]
    gated: []f32, // [n_pos, ff_dim] = silu(pre_gate) · up
    ff_out: []f32, // [n_pos, dim] = gated @ W_down
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
        .pre_gate = try gpa.alloc(f32, n_pos * cfg.ff_dim),
        .up = try gpa.alloc(f32, n_pos * cfg.ff_dim),
        .gated = try gpa.alloc(f32, n_pos * cfg.ff_dim),
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
    gpa.free(a.pre_gate);
    gpa.free(a.up);
    gpa.free(a.gated);
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

    // RoPE on Q + K (in-place; pos = row index). Skipped when
    // rotary_dim == 0 — preserves the pre-β-3a-3 numeric trajectory.
    if (cfg.rotary_dim > 0) {
        tt.ropeForwardBatched(acts.q, acts.q, n_pos, cfg.n_heads, cfg.head_dim, cfg.rotary_dim, cfg.rope_theta) catch unreachable;
        tt.ropeForwardBatched(acts.k, acts.k, n_pos, cfg.n_kv_heads, cfg.head_dim, cfg.rotary_dim, cfg.rope_theta) catch unreachable;
    }

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

    // pre_gate = n2 @ W_gateᵀ
    matmulNt(acts.pre_gate, acts.n2, layer.w_gate, n_pos, cfg.ff_dim, dim);

    // up = n2 @ W_upᵀ
    matmulNt(acts.up, acts.n2, layer.w_up, n_pos, cfg.ff_dim, dim);

    // gated = silu(pre_gate) · up
    tt.swigluForward(acts.pre_gate, acts.up, acts.gated);

    // ff_out = gated @ W_downᵀ
    matmulNt(acts.ff_out, acts.gated, layer.w_down, n_pos, dim, cfg.ff_dim);

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

/// Full backward starting from a known `d_y` (loss-grad already
/// computed). `grads` is *accumulated* into — caller zeroes between
/// optimizer steps via `grads.zero()`. `d_x_in` receives the gradient
/// flowing back into the layer's input; chunk-8c stack training
/// chains it into the previous layer's `d_y`. Both residual paths
/// contribute: the `mid = x_in + o` skip plus the rmsnorm_n1 gradient.
pub fn backwardFromDy(
    gpa: std.mem.Allocator,
    layer: *const Layer,
    acts: *const Acts,
    d_y: []const f32,
    grads: *Grads,
    d_x_in: []f32,
) !void {
    const cfg = layer.cfg;
    const dim = cfg.dim;
    const n_pos = cfg.n_pos;
    const q_dim = cfg.n_heads * cfg.head_dim;
    const kv_dim = cfg.n_kv_heads * cfg.head_dim;
    std.debug.assert(d_y.len == n_pos * dim);
    std.debug.assert(d_x_in.len == n_pos * dim);

    // ── 1. y = mid + ff_out  →  d_mid_total starts as d_y; d_ff_out = d_y.
    //    d_mid_total accumulates the rmsnorm-FF branch contribution
    //    (step 6) and is the sole gradient flowing into mid post-step-6.
    const d_mid = try gpa.alloc(f32, n_pos * dim);
    defer gpa.free(d_mid);
    @memcpy(d_mid, d_y);

    // ── 2. ff_out = gated @ W_downᵀ
    //    d_gated = d_y @ W_down                       (lin_dx)
    //    dw_down += d_yᵀ @ gated                       (lin_dw)
    const d_gated = try gpa.alloc(f32, n_pos * cfg.ff_dim);
    defer gpa.free(d_gated);
    linearBackward(d_y, acts.gated, layer.w_down, n_pos, dim, cfg.ff_dim, d_gated, grads.dw_down);

    // ── 3. SwiGLU backward.  d_gated → d_pre_gate, d_up.
    const d_pre_gate = try gpa.alloc(f32, n_pos * cfg.ff_dim);
    defer gpa.free(d_pre_gate);
    const d_up = try gpa.alloc(f32, n_pos * cfg.ff_dim);
    defer gpa.free(d_up);
    tt.swigluBackward(d_gated, acts.pre_gate, acts.up, d_pre_gate, d_up);

    // ── 4. pre_gate = n2 @ W_gateᵀ ; up = n2 @ W_upᵀ. Sum into d_n2.
    const d_n2 = try gpa.alloc(f32, n_pos * dim);
    defer gpa.free(d_n2);
    const d_n2_up = try gpa.alloc(f32, n_pos * dim);
    defer gpa.free(d_n2_up);
    linearBackward(d_pre_gate, acts.n2, layer.w_gate, n_pos, cfg.ff_dim, dim, d_n2, grads.dw_gate);
    linearBackward(d_up, acts.n2, layer.w_up, n_pos, cfg.ff_dim, dim, d_n2_up, grads.dw_up);
    for (d_n2, d_n2_up) |*a, b| a.* += b;

    // ── 5. n2 = RMSNorm(mid, w_n2)  →  d_mid += rmsnorm.dx;  dw_n2 += rmsnorm.dw
    const d_mid_norm = try gpa.alloc(f32, n_pos * dim);
    defer gpa.free(d_mid_norm);
    tt.rmsNormBackward(d_n2, acts.mid, layer.w_n2, cfg.rms_eps, false, n_pos, d_mid_norm, grads.dw_n2);
    for (d_mid, d_mid_norm) |*m, dn| m.* += dn;

    // ── 6. mid = x_in + o
    //    d_o = d_mid_total (the projection branch)
    //    d_x_in_residual = d_mid_total (the skip branch — added to d_x_in
    //    *after* the rmsnorm_n1 gradient lands at step 9)
    const d_o = d_mid; // alias — d_mid still holds d_mid_total, only read from here on

    // ── 7. o = attn_out @ W_Oᵀ
    const d_attn_out = try gpa.alloc(f32, n_pos * q_dim);
    defer gpa.free(d_attn_out);
    linearBackward(d_o, acts.attn_out, layer.w_o, n_pos, dim, q_dim, d_attn_out, grads.dw_o);

    // ── 8. SDPA backward.  attentionBackward writes into dQ/dK/dV
    //    (overwrites). Layout: [n_pos, n_heads, head_dim].
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

    // RoPE backward on dQ + dK — undoes the forward rotation so the
    // gradient matches the *pre-RoPE* Q/K, which is what the linear
    // backward of W_q/W_k expects.
    if (cfg.rotary_dim > 0) {
        const dQ_pre = try gpa.alloc(f32, n_pos * q_dim);
        defer gpa.free(dQ_pre);
        const dK_pre = try gpa.alloc(f32, n_pos * kv_dim);
        defer gpa.free(dK_pre);
        try tt.ropeBackwardBatched(dQ_pre, dQ, n_pos, cfg.n_heads, cfg.head_dim, cfg.rotary_dim, cfg.rope_theta);
        try tt.ropeBackwardBatched(dK_pre, dK, n_pos, cfg.n_kv_heads, cfg.head_dim, cfg.rotary_dim, cfg.rope_theta);
        @memcpy(dQ, dQ_pre);
        @memcpy(dK, dK_pre);
    }

    // ── 9. Q = n1 @ W_Qᵀ etc. Sum into d_n1.
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

    // ── 10. n1 = RMSNorm(x_in, w_n1)  →  d_x_in path1 = rmsnorm.dx;
    //    dw_n1 += rmsnorm.dw_partial. Then add the residual contribution
    //    from step 6: d_x_in += d_mid_total.
    tt.rmsNormBackward(d_n1, acts.x_in, layer.w_n1, cfg.rms_eps, false, n_pos, d_x_in, grads.dw_n1);
    for (d_x_in, d_mid) |*g, m| g.* += m;
}

/// MSE-loss wrapper for the chunk-8a single-layer smoke. Computes
/// `d_y = (2/N)(y − target)` and discards the input gradient (the
/// single-layer smoke trains end-to-end without a stacked gradient
/// chain). Stack-level fine-tunes call `backwardFromDy` directly.
pub fn backward(
    gpa: std.mem.Allocator,
    layer: *const Layer,
    acts: *const Acts,
    target: []const f32,
    grads: *Grads,
) !void {
    const cfg = layer.cfg;
    const total = cfg.n_pos * cfg.dim;
    const d_y = try gpa.alloc(f32, total);
    defer gpa.free(d_y);
    mseLossGrad(d_y, acts.y, target);

    const d_x_in = try gpa.alloc(f32, total);
    defer gpa.free(d_x_in);
    try backwardFromDy(gpa, layer, acts, d_y, grads, d_x_in);
}

// ── Adam optimizer ─────────────────────────────────────────────────

pub const AdamState = struct {
    m: [9][]f32,
    v: [9][]f32,
    t: u32 = 0,
    lr: f32,
    beta1: f32 = 0.9,
    beta2: f32 = 0.999,
    eps: f32 = 1e-8,

    pub fn init(gpa: std.mem.Allocator, layer: *Layer, lr: f32) !AdamState {
        var m: [9][]f32 = undefined;
        var v: [9][]f32 = undefined;
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
    inline for (0..9) |i| {
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

// ── Stack: N decoder layers + embedding + final RMSNorm + LM head ───
//
// Chunk 8c-α-1 oracle: stack a small number of decoder Layers with an
// embedding table on the front, a final RMSNorm + linear lm_head on
// the back, and softmax cross-entropy loss over vocab logits. CPU
// reference for the stack-level gradient chain — wraps the
// single-layer `backwardFromDy` + the embedding / softmax-CE / final
// rmsnorm / lm-head pieces. Shape is intentionally tiny so the
// numeric chain stays trustable: n_layers=2, vocab=8, dim=16, n_pos=4
// in the smoke. Real-model fine-tune (chunk 8c-β) inherits this exact
// data flow with bigger numbers and a tokenizer.

pub const StackConfig = struct {
    /// Per-layer config — the same `Config` the chunk-8a single-layer
    /// API takes. All N layers share it; heterogeneous per-layer dims
    /// are out of scope for the toy oracle.
    base: Config,
    n_layers: usize,
    vocab_size: usize,
};

pub const Stack = struct {
    cfg: StackConfig,
    embed: []f32, // [vocab, dim]
    layers: []Layer,
    final_norm: []f32, // [dim]
    lm_head: []f32, // [vocab, dim]   linear (untied from embed for now)
};

pub const StackActs = struct {
    token_ids: []const u32, // [n_pos]   caller-owned; not freed here
    x_emb: []f32, // [n_pos, dim]
    layer_acts: []Acts,
    final_norm_out: []f32, // [n_pos, dim]
    logits: []f32, // [n_pos, vocab]
};

pub const StackGrads = struct {
    dE_embed: []f32,
    layer_grads: []Grads,
    dw_final_norm: []f32,
    dw_lm_head: []f32,

    pub fn zero(self: *StackGrads) void {
        @memset(self.dE_embed, 0);
        for (self.layer_grads) |*lg| lg.zero();
        @memset(self.dw_final_norm, 0);
        @memset(self.dw_lm_head, 0);
    }
};

pub fn allocStackActs(gpa: std.mem.Allocator, cfg: StackConfig) !StackActs {
    const dim = cfg.base.dim;
    const n_pos = cfg.base.n_pos;
    const layer_acts = try gpa.alloc(Acts, cfg.n_layers);
    errdefer gpa.free(layer_acts);
    var i: usize = 0;
    errdefer for (layer_acts[0..i]) |*la| {
        var copy = la.*;
        freeActs(gpa, &copy);
    };
    while (i < cfg.n_layers) : (i += 1) {
        layer_acts[i] = try allocActs(gpa, cfg.base);
    }
    return StackActs{
        .token_ids = &.{},
        .x_emb = try gpa.alloc(f32, n_pos * dim),
        .layer_acts = layer_acts,
        .final_norm_out = try gpa.alloc(f32, n_pos * dim),
        .logits = try gpa.alloc(f32, n_pos * cfg.vocab_size),
    };
}

pub fn freeStackActs(gpa: std.mem.Allocator, a: *StackActs) void {
    gpa.free(a.x_emb);
    for (a.layer_acts) |*la| freeActs(gpa, la);
    gpa.free(a.layer_acts);
    gpa.free(a.final_norm_out);
    gpa.free(a.logits);
}

pub fn allocStackGrads(gpa: std.mem.Allocator, stack: *const Stack) !StackGrads {
    const cfg = stack.cfg;
    const dim = cfg.base.dim;
    const layer_grads = try gpa.alloc(Grads, cfg.n_layers);
    errdefer gpa.free(layer_grads);
    for (layer_grads, stack.layers) |*lg, *layer| {
        lg.* = .{
            .dw_n1 = try gpa.alloc(f32, layer.w_n1.len),
            .dw_q = try gpa.alloc(f32, layer.w_q.len),
            .dw_k = try gpa.alloc(f32, layer.w_k.len),
            .dw_v = try gpa.alloc(f32, layer.w_v.len),
            .dw_o = try gpa.alloc(f32, layer.w_o.len),
            .dw_n2 = try gpa.alloc(f32, layer.w_n2.len),
            .dw_gate = try gpa.alloc(f32, layer.w_gate.len),
            .dw_up = try gpa.alloc(f32, layer.w_up.len),
            .dw_down = try gpa.alloc(f32, layer.w_down.len),
        };
    }
    return StackGrads{
        .dE_embed = try gpa.alloc(f32, stack.embed.len),
        .layer_grads = layer_grads,
        .dw_final_norm = try gpa.alloc(f32, dim),
        .dw_lm_head = try gpa.alloc(f32, stack.lm_head.len),
    };
}

pub fn freeStackGrads(gpa: std.mem.Allocator, sg: *StackGrads) void {
    gpa.free(sg.dE_embed);
    for (sg.layer_grads) |*lg| {
        gpa.free(lg.dw_n1);
        gpa.free(lg.dw_q);
        gpa.free(lg.dw_k);
        gpa.free(lg.dw_v);
        gpa.free(lg.dw_o);
        gpa.free(lg.dw_n2);
        gpa.free(lg.dw_gate);
        gpa.free(lg.dw_up);
        gpa.free(lg.dw_down);
    }
    gpa.free(sg.layer_grads);
    gpa.free(sg.dw_final_norm);
    gpa.free(sg.dw_lm_head);
}

/// Stack forward. `acts.token_ids` must be set by the caller; this
/// fills the rest. Layer i reads `acts.layer_acts[i].x_in` and writes
/// `.y`; we copy `.y[i] → x_in[i+1]` between layers, so each layer's
/// saved activations are independent (simplifies the backward chain
/// at the cost of one buffer copy per layer).
pub fn stackForward(stack: *const Stack, acts: *StackActs) void {
    const cfg = stack.cfg;
    const dim = cfg.base.dim;
    const n_pos = cfg.base.n_pos;

    // Embed lookup → x_emb.
    for (acts.token_ids, 0..) |tok, p| {
        std.debug.assert(@as(usize, tok) < cfg.vocab_size);
        const src = stack.embed[@as(usize, tok) * dim .. (@as(usize, tok) + 1) * dim];
        @memcpy(acts.x_emb[p * dim .. (p + 1) * dim], src);
    }

    // Pipe x_emb → layer_acts[0].x_in.
    @memcpy(acts.layer_acts[0].x_in, acts.x_emb);
    for (stack.layers, acts.layer_acts, 0..) |*layer, *la, idx| {
        forward(layer, la);
        if (idx + 1 < stack.layers.len) {
            @memcpy(acts.layer_acts[idx + 1].x_in, la.y);
        }
    }

    // final_norm + lm_head.
    const last_y = acts.layer_acts[stack.layers.len - 1].y;
    tt.rmsNormForward(last_y, stack.final_norm, cfg.base.rms_eps, false, n_pos, acts.final_norm_out);
    matmulNt(acts.logits, acts.final_norm_out, stack.lm_head, n_pos, cfg.vocab_size, dim);
}

/// Mean cross-entropy loss across `n_pos` token-prediction positions.
/// `target_ids[p]` is the gold next-token for position `p`. Numerically
/// stable via row-max subtract; matches softmax_ce_loss_grad_batched.comp.
pub fn softmaxCeLoss(logits: []const f32, target_ids: []const u32, n_pos: usize, vocab: usize) f32 {
    std.debug.assert(logits.len == n_pos * vocab);
    std.debug.assert(target_ids.len == n_pos);
    var total: f64 = 0;
    for (0..n_pos) |p| {
        const off = p * vocab;
        var m: f32 = -std.math.inf(f32);
        for (0..vocab) |o| m = @max(m, logits[off + o]);
        var sum_e: f64 = 0;
        for (0..vocab) |o| sum_e += @exp(@as(f64, logits[off + o]) - @as(f64, m));
        const log_z: f64 = @as(f64, m) + @log(sum_e);
        const tgt = target_ids[p];
        std.debug.assert(@as(usize, tgt) < vocab);
        total -= @as(f64, logits[off + @as(usize, tgt)]) - log_z;
    }
    return @floatCast(total / @as(f64, @floatFromInt(n_pos)));
}

/// `d_logits[p, o] = (softmax(logits[p])[o] − [o == target_ids[p]]) / n_pos`.
/// Pre-scale by `1/n_pos` matches the loss's mean reduction so the
/// downstream optimizer sees averaged gradients (same convention as
/// `softmax_ce_loss_grad_batched.comp`).
pub fn softmaxCeLossGrad(
    d_logits: []f32,
    logits: []const f32,
    target_ids: []const u32,
    n_pos: usize,
    vocab: usize,
) void {
    std.debug.assert(d_logits.len == n_pos * vocab);
    std.debug.assert(logits.len == n_pos * vocab);
    std.debug.assert(target_ids.len == n_pos);
    const inv_n: f32 = 1.0 / @as(f32, @floatFromInt(n_pos));
    for (0..n_pos) |p| {
        const off = p * vocab;
        var m: f32 = -std.math.inf(f32);
        for (0..vocab) |o| m = @max(m, logits[off + o]);
        var sum_e: f64 = 0;
        for (0..vocab) |o| sum_e += @exp(@as(f64, logits[off + o]) - @as(f64, m));
        const inv_sum: f64 = 1.0 / sum_e;
        const tgt = target_ids[p];
        for (0..vocab) |o| {
            const p_o: f64 = @exp(@as(f64, logits[off + o]) - @as(f64, m)) * inv_sum;
            const t_o: f32 = if (o == @as(usize, tgt)) 1.0 else 0.0;
            d_logits[off + o] = inv_n * (@as(f32, @floatCast(p_o)) - t_o);
        }
    }
}

/// Stack backward. Mirrors `stackForward` in reverse:
///   d_logits = softmax_ce_loss_grad
///   linear_backward(lm_head)        →  d_final_norm_out, dw_lm_head
///   rmsnorm_backward(final_norm)    →  d_last_y,         dw_final_norm
///   for layer in reverse: backwardFromDy → next layer's d_y
///   embedding_backward               →  dE_embed
/// Grads are accumulated; caller resets via `grads.zero()` between
/// optimizer steps.
pub fn stackBackward(
    gpa: std.mem.Allocator,
    stack: *const Stack,
    acts: *const StackActs,
    target_ids: []const u32,
    grads: *StackGrads,
) !void {
    const cfg = stack.cfg;
    const dim = cfg.base.dim;
    const n_pos = cfg.base.n_pos;
    const vocab = cfg.vocab_size;

    // ── 1. d_logits = softmax_ce_grad
    const d_logits = try gpa.alloc(f32, n_pos * vocab);
    defer gpa.free(d_logits);
    softmaxCeLossGrad(d_logits, acts.logits, target_ids, n_pos, vocab);

    // ── 2. logits = final_norm_out @ lm_headᵀ
    const d_final_norm_out = try gpa.alloc(f32, n_pos * dim);
    defer gpa.free(d_final_norm_out);
    linearBackward(d_logits, acts.final_norm_out, stack.lm_head, n_pos, vocab, dim, d_final_norm_out, grads.dw_lm_head);

    // ── 3. final_norm_out = RMSNorm(last_y, final_norm)
    const last_idx = stack.layers.len - 1;
    const last_y = acts.layer_acts[last_idx].y;
    const d_last_y = try gpa.alloc(f32, n_pos * dim);
    defer gpa.free(d_last_y);
    tt.rmsNormBackward(d_final_norm_out, last_y, stack.final_norm, cfg.base.rms_eps, false, n_pos, d_last_y, grads.dw_final_norm);

    // ── 4. Layers (reverse order). `d_next_input` is the gradient
    //    flowing into the current layer's `y`; `backwardFromDy` writes
    //    the gradient flowing out of `x_in` into a fresh slice that
    //    becomes the next iteration's `d_next_input`.
    var d_next_input = d_last_y;
    var owned_next: ?[]f32 = null;
    defer if (owned_next) |slc| gpa.free(slc);
    var li: usize = stack.layers.len;
    while (li > 0) {
        li -= 1;
        const layer = &stack.layers[li];
        const la = &acts.layer_acts[li];
        const d_x_in_buf = try gpa.alloc(f32, n_pos * dim);
        try backwardFromDy(gpa, layer, la, d_next_input, &grads.layer_grads[li], d_x_in_buf);
        if (owned_next) |slc| gpa.free(slc);
        d_next_input = d_x_in_buf;
        owned_next = d_x_in_buf;
    }

    // ── 5. x_emb = embedding lookup. d_x_emb := d_next_input.
    tt.embeddingBackward(d_next_input, acts.token_ids, vocab, dim, grads.dE_embed);
}

// ── Stack Adam ─────────────────────────────────────────────────────

pub const StackAdamState = struct {
    /// Flat list of m and v buffers, in `paramOrder` (see fn below).
    m: [][]f32,
    v: [][]f32,
    t: u32 = 0,
    lr: f32,
    beta1: f32 = 0.9,
    beta2: f32 = 0.999,
    eps: f32 = 1e-8,
};

/// Iteration order: `embed`, then for each layer the 9 single-layer
/// params in `Layer.paramSlices` order, then `final_norm`, then
/// `lm_head`. Total = 9·n_layers + 3.
const stack_extra_params: usize = 3;

fn stackParamCount(stack: *const Stack) usize {
    return stack_extra_params + 9 * stack.cfg.n_layers;
}

fn fillStackParamSlices(stack: *Stack, params: [][]f32, grads: *StackGrads, gradsOut: [][]f32) void {
    var idx: usize = 0;
    params[idx] = stack.embed;
    gradsOut[idx] = grads.dE_embed;
    idx += 1;
    for (stack.layers, 0..) |*layer, li| {
        const ps = layer.paramSlices();
        const gs = grads.layer_grads[li].slices();
        for (ps, gs) |p, g| {
            params[idx] = p;
            gradsOut[idx] = g;
            idx += 1;
        }
    }
    params[idx] = stack.final_norm;
    gradsOut[idx] = grads.dw_final_norm;
    idx += 1;
    params[idx] = stack.lm_head;
    gradsOut[idx] = grads.dw_lm_head;
    idx += 1;
    std.debug.assert(idx == params.len);
}

pub fn stackAdamInit(gpa: std.mem.Allocator, stack: *Stack, lr: f32) !StackAdamState {
    const n = stackParamCount(stack);
    const m = try gpa.alloc([]f32, n);
    errdefer gpa.free(m);
    const v = try gpa.alloc([]f32, n);
    errdefer gpa.free(v);

    var params: [256][]f32 = undefined;
    std.debug.assert(n <= params.len);
    var idx: usize = 0;
    params[idx] = stack.embed;
    idx += 1;
    for (stack.layers) |*layer| {
        const ps = layer.paramSlices();
        for (ps) |p| {
            params[idx] = p;
            idx += 1;
        }
    }
    params[idx] = stack.final_norm;
    idx += 1;
    params[idx] = stack.lm_head;
    idx += 1;
    std.debug.assert(idx == n);

    var i: usize = 0;
    errdefer for (m[0..i]) |slc| gpa.free(slc);
    errdefer for (v[0..i]) |slc| gpa.free(slc);
    while (i < n) : (i += 1) {
        m[i] = try gpa.alloc(f32, params[i].len);
        v[i] = try gpa.alloc(f32, params[i].len);
        @memset(m[i], 0);
        @memset(v[i], 0);
    }
    return .{ .m = m, .v = v, .lr = lr };
}

pub fn stackAdamDeinit(state: *StackAdamState, gpa: std.mem.Allocator) void {
    for (state.m) |s| gpa.free(s);
    for (state.v) |s| gpa.free(s);
    gpa.free(state.m);
    gpa.free(state.v);
}

pub fn stackAdamStep(state: *StackAdamState, stack: *Stack, grads: *StackGrads) void {
    state.t += 1;
    const t_f: f32 = @floatFromInt(state.t);
    const bc1: f32 = 1.0 - std.math.pow(f32, state.beta1, t_f);
    const bc2: f32 = 1.0 - std.math.pow(f32, state.beta2, t_f);
    const lr_t: f32 = state.lr * @sqrt(bc2) / bc1;

    const n = stackParamCount(stack);
    var params: [256][]f32 = undefined;
    var grad_slices: [256][]f32 = undefined;
    std.debug.assert(n <= params.len);
    fillStackParamSlices(stack, params[0..n], grads, grad_slices[0..n]);

    for (0..n) |i| {
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
