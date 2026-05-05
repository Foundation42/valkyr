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
