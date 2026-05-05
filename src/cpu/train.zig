//! CPU reference for tiny-MLP training.
//!
//! This file is the oracle for the GPU training port: every backward
//! shader, every optimizer-step shader, every parity test in the
//! `--train-*-smoke` flags ultimately diffs its output against the
//! activations and gradients computed here. Pure scalar fp32 — easy to
//! read, easy to diff, deliberately no SIMD or fused intermediates.
//!
//! Architecture (chunk 1 of training-v0): a 2-layer MLP — Linear →
//! ReLU → Linear — with MSE loss and plain SGD. Tiny on purpose. The
//! whole training-v0 thesis is real-time training inside a frame
//! budget, where the model is tens of weights mapping mouse/time to a
//! BRDF, not millions of weights mapping tokens to logits. The full
//! transformer training port ("Unsloth" in the roadmap) is built on
//! top of these primitives later.
//!
//! Conventions match `src/cpu/math.zig`:
//!   - Row-major, NT-style weights: W is shape [dim_out, dim_in], so
//!     y[i] = Σⱼ W[i, j] · x[j] + b[i].
//!   - Sequential scalar fp32 accumulation; no Kahan summation.
//!   - Slice-typed APIs; caller owns all buffers.

const std = @import("std");

/// Two-layer MLP: x → (W1, b1) → ReLU → (W2, b2) → y.
///
/// Storage is plain `[]f32` so the host can upload it directly to a
/// Vulkan buffer for the GPU port — no boxing, no per-tensor metadata.
/// The shapes are implicit in the dim_* fields; layout is row-major,
/// `W1[i*dim_in + j]` is row i column j.
pub const Mlp = struct {
    dim_in: usize,
    dim_hidden: usize,
    dim_out: usize,

    w1: []f32, // [dim_hidden, dim_in]
    b1: []f32, // [dim_hidden]
    w2: []f32, // [dim_out, dim_hidden]
    b2: []f32, // [dim_out]

    /// Construct an MLP with weights drawn from U(-init_scale, +init_scale)
    /// and biases zeroed. Deterministic given `seed`. For real training
    /// you'd want Kaiming on W1 and Xavier on W2, but for tiny demos a
    /// plain uniform with init_scale ≈ 0.3 lands you in the well of
    /// convergence for our toy task without hiding bugs behind clever
    /// initialisation.
    pub fn init(
        allocator: std.mem.Allocator,
        dim_in: usize,
        dim_hidden: usize,
        dim_out: usize,
        init_scale: f32,
        seed: u64,
    ) !Mlp {
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();

        const w1 = try allocator.alloc(f32, dim_hidden * dim_in);
        errdefer allocator.free(w1);
        const b1 = try allocator.alloc(f32, dim_hidden);
        errdefer allocator.free(b1);
        const w2 = try allocator.alloc(f32, dim_out * dim_hidden);
        errdefer allocator.free(w2);
        const b2 = try allocator.alloc(f32, dim_out);
        errdefer allocator.free(b2);

        for (w1) |*v| v.* = (rng.float(f32) * 2.0 - 1.0) * init_scale;
        for (w2) |*v| v.* = (rng.float(f32) * 2.0 - 1.0) * init_scale;
        @memset(b1, 0);
        @memset(b2, 0);

        return .{
            .dim_in = dim_in,
            .dim_hidden = dim_hidden,
            .dim_out = dim_out,
            .w1 = w1,
            .b1 = b1,
            .w2 = w2,
            .b2 = b2,
        };
    }

    pub fn deinit(self: *Mlp, allocator: std.mem.Allocator) void {
        allocator.free(self.w1);
        allocator.free(self.b1);
        allocator.free(self.w2);
        allocator.free(self.b2);
    }
};

/// Cached activations from a forward pass. Backward needs `x` and
/// `h_pre` (for the ReLU mask) and `h` (for the dW2 outer product),
/// so we keep all three around. `y` is the prediction the loss is
/// computed against.
///
/// The host allocates these once and reuses them across steps — same
/// pattern the inference runner uses for ping-pong arenas. We don't
/// own the storage; caller passes slices in.
pub const Activations = struct {
    x: []const f32,
    h_pre: []f32,
    h: []f32,
    y: []f32,
};

/// Gradient buffers, sized to match the MLP exactly. Same layout as
/// the parameters they shadow. Caller allocates; same-shape pairing
/// keeps the GPU port trivial — we'll bind W and dW as the same
/// buffer descriptor on the SGD step.
pub const Grads = struct {
    dw1: []f32,
    db1: []f32,
    dw2: []f32,
    db2: []f32,

    pub fn init(allocator: std.mem.Allocator, mlp: *const Mlp) !Grads {
        const dw1 = try allocator.alloc(f32, mlp.w1.len);
        errdefer allocator.free(dw1);
        const db1 = try allocator.alloc(f32, mlp.b1.len);
        errdefer allocator.free(db1);
        const dw2 = try allocator.alloc(f32, mlp.w2.len);
        errdefer allocator.free(dw2);
        const db2 = try allocator.alloc(f32, mlp.b2.len);
        errdefer allocator.free(db2);
        return .{ .dw1 = dw1, .db1 = db1, .dw2 = dw2, .db2 = db2 };
    }

    pub fn deinit(self: *Grads, allocator: std.mem.Allocator) void {
        allocator.free(self.dw1);
        allocator.free(self.db1);
        allocator.free(self.dw2);
        allocator.free(self.db2);
    }

    pub fn zero(self: *Grads) void {
        @memset(self.dw1, 0);
        @memset(self.db1, 0);
        @memset(self.dw2, 0);
        @memset(self.db2, 0);
    }
};

/// Forward pass. Fills `act.h_pre`, `act.h`, `act.y` from `act.x`.
/// `act.x` must already be set by the caller — this is a pure read of
/// it, never written.
pub fn forward(mlp: *const Mlp, act: *Activations) void {
    std.debug.assert(act.x.len == mlp.dim_in);
    std.debug.assert(act.h_pre.len == mlp.dim_hidden);
    std.debug.assert(act.h.len == mlp.dim_hidden);
    std.debug.assert(act.y.len == mlp.dim_out);

    // Layer 1: h_pre = W1 · x + b1, h = relu(h_pre).
    for (0..mlp.dim_hidden) |i| {
        var acc: f32 = mlp.b1[i];
        const row_off = i * mlp.dim_in;
        for (0..mlp.dim_in) |j| {
            acc += mlp.w1[row_off + j] * act.x[j];
        }
        act.h_pre[i] = acc;
        act.h[i] = if (acc > 0) acc else 0;
    }

    // Layer 2: y = W2 · h + b2.
    for (0..mlp.dim_out) |i| {
        var acc: f32 = mlp.b2[i];
        const row_off = i * mlp.dim_hidden;
        for (0..mlp.dim_hidden) |j| {
            acc += mlp.w2[row_off + j] * act.h[j];
        }
        act.y[i] = acc;
    }
}

/// Mean-squared-error loss, summed over the output (no division by N).
/// L = (1/2) · Σᵢ (y[i] − t[i])²
///
/// The 1/2 makes the gradient drop the factor of 2 (dL/dy[i] = y[i] −
/// t[i]) — purely a convention so the gradient looks clean. Multiply
/// the learning rate by 2 if you'd rather match the un-halved form.
pub fn mseLoss(y: []const f32, target: []const f32) f32 {
    std.debug.assert(y.len == target.len);
    var acc: f32 = 0;
    for (y, target) |yi, ti| {
        const d = yi - ti;
        acc += d * d;
    }
    return 0.5 * acc;
}

/// Gradient of `mseLoss` wrt `y`. With the 1/2 in the loss, this is
/// simply (y − t).
pub fn mseLossGrad(dL_dy: []f32, y: []const f32, target: []const f32) void {
    std.debug.assert(y.len == target.len);
    std.debug.assert(dL_dy.len == y.len);
    for (dL_dy, y, target) |*d, yi, ti| d.* = yi - ti;
}

/// Backward pass. Given activations from `forward` and `dL_dy` (the
/// upstream gradient — typically from `mseLossGrad`), fills every
/// field of `grads`.
///
/// Hand-derived chain rule:
///   dL/dW2[i, j] = dL/dy[i] · h[j]
///   dL/db2[i]    = dL/dy[i]
///   dL/dh[j]     = Σᵢ dL/dy[i] · W2[i, j]
///   dL/dh_pre[j] = dL/dh[j]       if h_pre[j] > 0 else 0     (ReLU)
///   dL/dW1[j, k] = dL/dh_pre[j] · x[k]
///   dL/db1[j]    = dL/dh_pre[j]
///
/// `dL_dh` is allocated on the caller's stack via a heap fallback
/// when dim_hidden exceeds the inline cap — keeps the API allocation-
/// free for our actual sizes (hidden ≤ 64) and still correct for
/// larger.
pub fn backward(
    allocator: std.mem.Allocator,
    mlp: *const Mlp,
    act: *const Activations,
    dL_dy: []const f32,
    grads: *Grads,
) !void {
    std.debug.assert(dL_dy.len == mlp.dim_out);
    std.debug.assert(act.x.len == mlp.dim_in);

    // dL/db2 = dL/dy ;  dL/dW2[i, j] = dL/dy[i] · h[j].
    for (0..mlp.dim_out) |i| {
        grads.db2[i] = dL_dy[i];
        const row_off = i * mlp.dim_hidden;
        for (0..mlp.dim_hidden) |j| {
            grads.dw2[row_off + j] = dL_dy[i] * act.h[j];
        }
    }

    // dL/dh[j] = Σᵢ dL/dy[i] · W2[i, j]   (transposed matvec).
    const dL_dh = try allocator.alloc(f32, mlp.dim_hidden);
    defer allocator.free(dL_dh);
    @memset(dL_dh, 0);
    for (0..mlp.dim_out) |i| {
        const w2_row = i * mlp.dim_hidden;
        const dy_i = dL_dy[i];
        for (0..mlp.dim_hidden) |j| {
            dL_dh[j] += dy_i * mlp.w2[w2_row + j];
        }
    }

    // dL/dh_pre = dL/dh · 1[h_pre > 0]  (ReLU mask).
    // dL/db1 = dL/dh_pre ;  dL/dW1[j, k] = dL/dh_pre[j] · x[k].
    for (0..mlp.dim_hidden) |j| {
        const mask: f32 = if (act.h_pre[j] > 0) 1.0 else 0.0;
        const dpre_j = dL_dh[j] * mask;
        grads.db1[j] = dpre_j;
        const row_off = j * mlp.dim_in;
        for (0..mlp.dim_in) |k| {
            grads.dw1[row_off + k] = dpre_j * act.x[k];
        }
    }
}

/// In-place SGD: param ← param − lr · grad. No momentum, no weight
/// decay — both can be added later when the parity story for the
/// trivial form is solid.
pub fn sgdStep(mlp: *Mlp, grads: *const Grads, lr: f32) void {
    std.debug.assert(grads.dw1.len == mlp.w1.len);
    std.debug.assert(grads.db1.len == mlp.b1.len);
    std.debug.assert(grads.dw2.len == mlp.w2.len);
    std.debug.assert(grads.db2.len == mlp.b2.len);

    for (mlp.w1, grads.dw1) |*p, g| p.* -= lr * g;
    for (mlp.b1, grads.db1) |*p, g| p.* -= lr * g;
    for (mlp.w2, grads.dw2) |*p, g| p.* -= lr * g;
    for (mlp.b2, grads.db2) |*p, g| p.* -= lr * g;
}

// ── Tiny end-to-end helper: forward + loss + backward + step ──────
//
// Convenient for the smoke test and for the upcoming TrainingRunner.
// Returns the loss BEFORE the optimizer step (i.e. the loss of the
// prediction we just made). Caller is expected to allocate `act` and
// `grads` once and reuse across steps.

pub fn trainStep(
    allocator: std.mem.Allocator,
    mlp: *Mlp,
    act: *Activations,
    grads: *Grads,
    target: []const f32,
    lr: f32,
) !f32 {
    forward(mlp, act);
    const loss = mseLoss(act.y, target);

    var dL_dy_buf: [256]f32 = undefined;
    std.debug.assert(target.len <= dL_dy_buf.len); // tiny-MLP regime
    const dL_dy = dL_dy_buf[0..target.len];
    mseLossGrad(dL_dy, act.y, target);

    try backward(allocator, mlp, act, dL_dy, grads);
    sgdStep(mlp, grads, lr);
    return loss;
}

// ── Multi-layer MLP (n ≥ 1) ───────────────────────────────────────
//
// `MlpN` generalises `Mlp` to any depth. Hidden layers use ReLU;
// the output layer is raw (the loss head — MSE or softmax+CE — does
// any final activation). The 2-layer case (`MlpN.layerDims = .{4,
// 8, 4}`) is numerically equivalent to `Mlp(4, 8, 4)`; the difference
// is the storage shape: `MlpN.weights[0]` corresponds to `Mlp.w1`,
// `MlpN.weights[1]` to `Mlp.w2`, etc.
//
// Used as the CPU parity oracle for the upcoming multi-layer GPU
// runner, same way `Mlp` underpins the 2-layer `TrainingRunner`.

/// N-layer MLP. `layer_dims` has length n+1: [dim_in, h_1, …, h_{n-1},
/// dim_out]. `weights[i]` is shape [layer_dims[i+1], layer_dims[i]],
/// row-major. `biases[i]` is length layer_dims[i+1].
pub const MlpN = struct {
    layer_dims: []usize,
    weights: [][]f32,
    biases: [][]f32,

    pub fn nLayers(self: *const MlpN) usize {
        return self.weights.len;
    }
    pub fn dimIn(self: *const MlpN) usize {
        return self.layer_dims[0];
    }
    pub fn dimOut(self: *const MlpN) usize {
        return self.layer_dims[self.layer_dims.len - 1];
    }

    /// Initialise with weights drawn from U(-init_scale, +init_scale)
    /// and biases zeroed. RNG advances layer-by-layer, weight-row-by-
    /// row, so a 2-layer MlpN with the same seed and dims as a 2-layer
    /// `Mlp` produces bit-identical W1 then W2.
    pub fn init(
        allocator: std.mem.Allocator,
        layer_dims: []const usize,
        init_scale: f32,
        seed: u64,
    ) !MlpN {
        std.debug.assert(layer_dims.len >= 2);

        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();

        const dims = try allocator.alloc(usize, layer_dims.len);
        errdefer allocator.free(dims);
        @memcpy(dims, layer_dims);

        const n = layer_dims.len - 1;
        const weights = try allocator.alloc([]f32, n);
        errdefer allocator.free(weights);
        const biases = try allocator.alloc([]f32, n);
        errdefer allocator.free(biases);

        // Two-pass alloc with rollback-on-error to keep the partial
        // state cleanable. Allocate all weights first, then biases —
        // simpler errdefer story.
        var allocated: usize = 0;
        errdefer for (weights[0..allocated]) |w| allocator.free(w);
        for (0..n) |i| {
            const w = try allocator.alloc(f32, layer_dims[i + 1] * layer_dims[i]);
            weights[i] = w;
            allocated += 1;
        }

        var bias_allocated: usize = 0;
        errdefer for (biases[0..bias_allocated]) |b| allocator.free(b);
        for (0..n) |i| {
            const b = try allocator.alloc(f32, layer_dims[i + 1]);
            biases[i] = b;
            bias_allocated += 1;
        }

        for (weights) |w| for (w) |*v| {
            v.* = (rng.float(f32) * 2.0 - 1.0) * init_scale;
        };
        for (biases) |b| @memset(b, 0);

        return .{ .layer_dims = dims, .weights = weights, .biases = biases };
    }

    pub fn deinit(self: *MlpN, allocator: std.mem.Allocator) void {
        for (self.weights) |w| allocator.free(w);
        for (self.biases) |b| allocator.free(b);
        allocator.free(self.weights);
        allocator.free(self.biases);
        allocator.free(self.layer_dims);
    }
};

/// Per-layer activation cache. `pre[i]` is the pre-activation of
/// layer i; `post[i]` is the ReLU-applied output (= y for the output
/// layer, where ReLU is skipped). `x` is the network input — caller-
/// owned, never written.
pub const ActivationsN = struct {
    x: []const f32,
    pre: [][]f32,
    post: [][]f32,

    pub fn init(allocator: std.mem.Allocator, mlp: *const MlpN) !ActivationsN {
        const n = mlp.nLayers();
        const pre = try allocator.alloc([]f32, n);
        errdefer allocator.free(pre);
        const post = try allocator.alloc([]f32, n);
        errdefer allocator.free(post);

        var allocated: usize = 0;
        errdefer {
            for (pre[0..allocated]) |buf| allocator.free(buf);
            for (post[0..allocated]) |buf| allocator.free(buf);
        }
        for (0..n) |i| {
            pre[i] = try allocator.alloc(f32, mlp.layer_dims[i + 1]);
            post[i] = try allocator.alloc(f32, mlp.layer_dims[i + 1]);
            allocated += 1;
        }
        return .{ .x = &[_]f32{}, .pre = pre, .post = post };
    }

    pub fn deinit(self: *ActivationsN, allocator: std.mem.Allocator) void {
        for (self.pre) |buf| allocator.free(buf);
        for (self.post) |buf| allocator.free(buf);
        allocator.free(self.pre);
        allocator.free(self.post);
    }

    pub fn y(self: *const ActivationsN) []f32 {
        return self.post[self.post.len - 1];
    }
};

/// Per-layer gradients, same shape as the parameters they shadow.
pub const GradsN = struct {
    dw: [][]f32,
    db: [][]f32,

    pub fn init(allocator: std.mem.Allocator, mlp: *const MlpN) !GradsN {
        const n = mlp.nLayers();
        const dw = try allocator.alloc([]f32, n);
        errdefer allocator.free(dw);
        const db = try allocator.alloc([]f32, n);
        errdefer allocator.free(db);

        var allocated: usize = 0;
        errdefer {
            for (dw[0..allocated]) |buf| allocator.free(buf);
            for (db[0..allocated]) |buf| allocator.free(buf);
        }
        for (0..n) |i| {
            dw[i] = try allocator.alloc(f32, mlp.weights[i].len);
            db[i] = try allocator.alloc(f32, mlp.biases[i].len);
            allocated += 1;
        }
        return .{ .dw = dw, .db = db };
    }

    pub fn deinit(self: *GradsN, allocator: std.mem.Allocator) void {
        for (self.dw) |buf| allocator.free(buf);
        for (self.db) |buf| allocator.free(buf);
        allocator.free(self.dw);
        allocator.free(self.db);
    }

    pub fn zero(self: *GradsN) void {
        for (self.dw) |buf| @memset(buf, 0);
        for (self.db) |buf| @memset(buf, 0);
    }
};

/// Forward pass. ReLU on every layer except the last. `act.x` must be
/// set by the caller; everything else is overwritten.
pub fn forwardN(mlp: *const MlpN, act: *ActivationsN) void {
    const n = mlp.nLayers();
    std.debug.assert(act.x.len == mlp.dimIn());
    std.debug.assert(act.pre.len == n);
    std.debug.assert(act.post.len == n);

    var input: []const f32 = act.x;
    for (0..n) |layer| {
        const dim_out = mlp.layer_dims[layer + 1];
        const dim_in = mlp.layer_dims[layer];
        const w = mlp.weights[layer];
        const b = mlp.biases[layer];
        const pre = act.pre[layer];
        const post = act.post[layer];
        std.debug.assert(pre.len == dim_out);
        std.debug.assert(post.len == dim_out);

        for (0..dim_out) |i| {
            var acc: f32 = b[i];
            const row_off = i * dim_in;
            for (0..dim_in) |j| acc += w[row_off + j] * input[j];
            pre[i] = acc;
            // ReLU on hidden layers; raw output on the last layer.
            post[i] = if (layer + 1 == n) acc else if (acc > 0) acc else 0;
        }
        input = post;
    }
}

/// Backward pass. Given activations + dL/dy, fills every gradient.
/// Hand-derived chain rule, generalised over depth:
///   dL/dW[L][i,j] = dL/d_pre[L][i] · post[L-1][j]   (post[-1] = x)
///   dL/db[L][i]   = dL/d_pre[L][i]
///   dL/d_post[L-1][j] = Σᵢ dL/d_pre[L][i] · W[L][i,j]
///   dL/d_pre[L-1][j]  = dL/d_post[L-1][j] · 1[pre[L-1][j] > 0]   (ReLU)
/// `dL_dy` initially seeds dL/d_pre on the LAST layer (output layer
/// has no ReLU, so d_pre = d_post = d_y there).
pub fn backwardN(
    allocator: std.mem.Allocator,
    mlp: *const MlpN,
    act: *const ActivationsN,
    dL_dy: []const f32,
    grads: *GradsN,
) !void {
    const n = mlp.nLayers();
    std.debug.assert(dL_dy.len == mlp.dimOut());

    // Working buffers for d_pre at the current layer; d_post at the
    // previous layer. Sized to the largest layer dim to avoid per-
    // layer reallocs.
    var max_dim: usize = 0;
    for (mlp.layer_dims) |d| max_dim = @max(max_dim, d);
    const d_pre = try allocator.alloc(f32, max_dim);
    defer allocator.free(d_pre);
    const d_post = try allocator.alloc(f32, max_dim);
    defer allocator.free(d_post);

    // Seed: d_pre on the output layer = dL_dy (no ReLU at output).
    @memcpy(d_pre[0..mlp.dimOut()], dL_dy);

    var layer: usize = n;
    while (layer > 0) : (layer -= 1) {
        const idx = layer - 1;
        const dim_out = mlp.layer_dims[layer];
        const dim_in = mlp.layer_dims[idx];
        const w = mlp.weights[idx];
        const dw = grads.dw[idx];
        const db = grads.db[idx];

        // Input to this layer is post[idx-1], or x if idx == 0.
        const input: []const f32 = if (idx == 0) act.x else act.post[idx - 1];

        // dW[i,j] = d_pre[i] · input[j];  db[i] = d_pre[i].
        for (0..dim_out) |i| {
            db[i] = d_pre[i];
            const row_off = i * dim_in;
            for (0..dim_in) |j| {
                dw[row_off + j] = d_pre[i] * input[j];
            }
        }

        // If we still have layers below, propagate to d_pre on
        // the layer below. d_post[idx-1][j] = Σᵢ d_pre[i]·W[i,j];
        // then ReLU mask using pre[idx-1].
        if (idx > 0) {
            const dim_below = dim_in; // = mlp.layer_dims[idx]
            @memset(d_post[0..dim_below], 0);
            for (0..dim_out) |i| {
                const row_off = i * dim_below;
                const dpi = d_pre[i];
                for (0..dim_below) |j| d_post[j] += dpi * w[row_off + j];
            }
            const pre_below = act.pre[idx - 1];
            for (0..dim_below) |j| {
                const mask: f32 = if (pre_below[j] > 0) 1.0 else 0.0;
                d_pre[j] = d_post[j] * mask;
            }
        }
    }
}

/// In-place SGD on every parameter. Same convention as `sgdStep`.
pub fn sgdStepN(mlp: *MlpN, grads: *const GradsN, lr: f32) void {
    const n = mlp.nLayers();
    for (0..n) |i| {
        for (mlp.weights[i], grads.dw[i]) |*p, g| p.* -= lr * g;
        for (mlp.biases[i], grads.db[i]) |*p, g| p.* -= lr * g;
    }
}

/// Forward + loss + backward + step. Returns pre-step loss.
pub fn trainStepN(
    allocator: std.mem.Allocator,
    mlp: *MlpN,
    act: *ActivationsN,
    grads: *GradsN,
    target: []const f32,
    lr: f32,
) !f32 {
    forwardN(mlp, act);
    const y_out = act.y();
    const loss = mseLoss(y_out, target);

    var dL_dy_buf: [256]f32 = undefined;
    std.debug.assert(target.len <= dL_dy_buf.len);
    const dL_dy = dL_dy_buf[0..target.len];
    mseLossGrad(dL_dy, y_out, target);

    try backwardN(allocator, mlp, act, dL_dy, grads);
    sgdStepN(mlp, grads, lr);
    return loss;
}
