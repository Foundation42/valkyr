//! CPU reference for the Qwen3.5 Gated DeltaNet linear-attention layer.
//!
//! This is the recurrent decode path — one new hidden state in, one
//! hidden state out, with a `(conv_state, recurrent_state)` cache that
//! grows by zero with sequence length (the entire SSM cache is fixed-
//! size per layer).
//!
//! Reference: HF transformers `Qwen3_5GatedDeltaNet.forward` in
//! `models/qwen3_5/modular_qwen3_5.py:343`, falling through to
//! `torch_recurrent_gated_delta_rule` and `torch_causal_conv1d_update`
//! in `models/qwen3_next/modeling_qwen3_next.py:443,547`.
//!
//! Decode math (per V-head h, with K-heads repeated `num_v_heads /
//! num_k_heads` times to match V-head count):
//!
//!     mixed_qkv = in_proj_qkv @ x                      # (conv_dim,)
//!     z         = in_proj_z   @ x                      # (value_dim,)
//!     b_raw     = in_proj_b   @ x                      # (num_v_heads,)
//!     a_raw     = in_proj_a   @ x                      # (num_v_heads,)
//!
//!     # Causal 1-D depthwise conv update, kernel = K (= 4 for Qwen3.5).
//!     # State is [conv_dim, K]; shift left by one and append `mixed_qkv`
//!     # in column K-1 each step. Output at step t is the dot product of
//!     # the K-wide window with the per-channel weight.
//!     state[:, 0..K-1] = state[:, 1..K]
//!     state[:, K-1]    = mixed_qkv
//!     mixed_qkv        = silu( sum_k state[:, k] * weight[:, 0, k] )
//!
//!     # Split into (Q, K, V) and reshape per-head.
//!     q = mixed_qkv[0..key_dim].view(num_k_heads, head_k_dim)
//!     k = mixed_qkv[key_dim..2*key_dim].view(num_k_heads, head_k_dim)
//!     v = mixed_qkv[2*key_dim..].view(num_v_heads, head_v_dim)
//!
//!     q = l2norm(q); k = l2norm(k)
//!     # GQA repeat: K-heads (16) → V-heads (32). Same recipe on Q.
//!     q = q.repeat_interleave(num_v_heads / num_k_heads, axis=0)
//!     k = k.repeat_interleave(num_v_heads / num_k_heads, axis=0)
//!     q *= 1 / sqrt(head_k_dim)
//!
//!     beta = sigmoid(b_raw)                            # (num_v_heads,)
//!     g    = -exp(A_log) * softplus(a_raw + dt_bias)   # (num_v_heads,)
//!     g_t  = exp(g)                                    # the actual decay this step
//!
//!     for h in 0..num_v_heads:
//!         S[h]      *= g_t[h]                            # decay
//!         kv_mem[h] = sum_d S[h, d, :] * k[h, d]         # = Sᵀ k
//!         delta[h]  = (v[h] - kv_mem[h]) * beta[h]        # delta-rule write
//!         S[h]     += outer(k[h], delta[h])              # state update
//!         out[h]    = sum_d S[h, d, :] * q[h, d]         # readout = Sᵀ q
//!
//!     # RMSNormGated (per V-head, weight shared across heads):
//!     for h in 0..num_v_heads:
//!         var       = mean(out[h]²)
//!         out[h]    = norm_w * (out[h] * rsqrt(var + eps)) * silu(z_h)
//!
//!     y = out_proj @ flatten(out)                      # (hidden,)
//!
//! All gate / state arithmetic stays in fp32 per the model's
//! `mamba_ssm_dtype: float32` setting. Cache-state width is
//! `linear_conv_kernel_dim` (= 4 for Qwen3.5), not `kernel - 1`,
//! mirroring HF (`F.pad(..., (kernel - seq_len, 0))` produces a
//! kernel-wide left-padded buffer).

const std = @import("std");
const config_mod = @import("../config.zig");
const safetensors = @import("../safetensors.zig");
const cpu_math = @import("math.zig");
const dtype = @import("../dtype.zig");
const model_mod = @import("../model.zig");

const Tensor = safetensors.Tensor;
const Layer = model_mod.Layer;
const Config = config_mod.Config;

/// Per-layer recurrent cache. `init` zeroes both buffers — the model
/// boots from "no context", which matches HF's freshly-constructed
/// `Qwen3_5DynamicCache` with no prior tokens.
pub const State = struct {
    /// `[conv_dim, kernel]`, row-major: state[c, k] = data[c * kernel + k].
    /// Each decode step shifts left and appends the new mixed_qkv vector
    /// in the rightmost column.
    conv: []f32,
    /// `[num_v_heads, head_k_dim, head_v_dim]`, row-major:
    /// state[h, d, v] = data[h * head_k_dim * head_v_dim + d * head_v_dim + v].
    /// One outer-product update per decode step per V-head.
    recurrent: []f32,

    conv_dim: usize,
    kernel: usize,
    num_v_heads: usize,
    head_k_dim: usize,
    head_v_dim: usize,

    pub fn init(gpa: std.mem.Allocator, cfg: Config) !State {
        const conv_dim = cfg.linearAttnConvDim();
        const kernel = cfg.linear_conv_kernel_dim;
        const num_v_heads = cfg.linear_num_value_heads;
        const head_k_dim = cfg.linear_key_head_dim;
        const head_v_dim = cfg.linear_value_head_dim;

        const conv = try gpa.alloc(f32, conv_dim * kernel);
        @memset(conv, 0.0);
        const rec = try gpa.alloc(f32, num_v_heads * head_k_dim * head_v_dim);
        @memset(rec, 0.0);

        return .{
            .conv = conv,
            .recurrent = rec,
            .conv_dim = conv_dim,
            .kernel = kernel,
            .num_v_heads = num_v_heads,
            .head_k_dim = head_k_dim,
            .head_v_dim = head_v_dim,
        };
    }

    pub fn deinit(self: *State, gpa: std.mem.Allocator) void {
        gpa.free(self.conv);
        gpa.free(self.recurrent);
    }
};

const eps_l2norm: f32 = 1e-6;

inline fn silu(x: f32) f32 {
    return x / (1.0 + @exp(-x));
}

inline fn sigmoid(x: f32) f32 {
    return 1.0 / (1.0 + @exp(-x));
}

inline fn softplus(x: f32) f32 {
    // Numerically stable softplus: for large +x, softplus ≈ x; for very
    // negative x, log1p(exp(x)) is fine.
    if (x > 20.0) return x;
    return @log(1.0 + @exp(x));
}

/// Decode one new token through one Gated DeltaNet layer. `x` is the
/// post-input-layernorm hidden state (shape `[hidden_size]`); `out` is
/// the layer's contribution to the residual stream (shape
/// `[hidden_size]`) — the caller does the residual add. `state` is
/// mutated in place.
pub fn decodeStep(
    gpa: std.mem.Allocator,
    cfg: Config,
    layer: Layer,
    state: *State,
    x: []const f32,
    out: []f32,
) !void {
    if (x.len != cfg.hidden_size) return error.LengthMismatch;
    if (out.len != cfg.hidden_size) return error.LengthMismatch;
    if (layer.layer_type != .linear_attention) return error.NotLinearLayer;

    const hidden = cfg.hidden_size;
    const num_k_heads = cfg.linear_num_key_heads;
    const num_v_heads = cfg.linear_num_value_heads;
    const head_k = cfg.linear_key_head_dim;
    const head_v = cfg.linear_value_head_dim;
    const key_dim = num_k_heads * head_k;
    const value_dim = num_v_heads * head_v;
    const conv_dim = state.conv_dim;
    const kernel = state.kernel;

    std.debug.assert(conv_dim == 2 * key_dim + value_dim);
    std.debug.assert(num_v_heads % num_k_heads == 0);
    const heads_per_k = num_v_heads / num_k_heads;

    // ── 1. Input projections ────────────────────────────────────────
    const mixed_qkv = try gpa.alloc(f32, conv_dim);
    defer gpa.free(mixed_qkv);
    try cpu_math.matmul_nt(mixed_qkv, x, layer.in_proj_qkv.?, 1, conv_dim, hidden);

    const z = try gpa.alloc(f32, value_dim);
    defer gpa.free(z);
    try cpu_math.matmul_nt(z, x, layer.in_proj_z.?, 1, value_dim, hidden);

    const b_raw = try gpa.alloc(f32, num_v_heads);
    defer gpa.free(b_raw);
    try cpu_math.matmul_nt(b_raw, x, layer.in_proj_b.?, 1, num_v_heads, hidden);

    const a_raw = try gpa.alloc(f32, num_v_heads);
    defer gpa.free(a_raw);
    try cpu_math.matmul_nt(a_raw, x, layer.in_proj_a.?, 1, num_v_heads, hidden);

    // ── 2. Causal conv1d update + SiLU ──────────────────────────────
    // Shift `state.conv` left by one column and append `mixed_qkv` in
    // the rightmost column, then run a depthwise dot-product across the
    // kernel-wide window. Weight tensor is `[conv_dim, 1, kernel]`,
    // flattened to a `[conv_dim * kernel]` row-major buffer.
    const conv_w_tensor = layer.conv1d_weight.?;
    const conv_w = try gpa.alloc(f32, conv_dim * kernel);
    defer gpa.free(conv_w);
    try cpu_math.tensorToF32Slice(conv_w, conv_w_tensor);

    for (0..conv_dim) |c| {
        // shift left
        var k_idx: usize = 0;
        while (k_idx + 1 < kernel) : (k_idx += 1) {
            state.conv[c * kernel + k_idx] = state.conv[c * kernel + k_idx + 1];
        }
        // append new
        state.conv[c * kernel + kernel - 1] = mixed_qkv[c];
        // depthwise dot product across the window
        var acc: f32 = 0.0;
        for (0..kernel) |k_pos| {
            acc += state.conv[c * kernel + k_pos] * conv_w[c * kernel + k_pos];
        }
        mixed_qkv[c] = silu(acc);
    }

    // ── 3. Split + per-head L2-norm + GQA repeat ────────────────────
    // q/k/v live in `mixed_qkv` already; reshape them as flat
    // (num_v_heads, head_*) buffers AFTER the K-head → V-head repeat.
    // That avoids carrying around two index spaces. Q gets the standard
    // 1/sqrt(head_k) scale here too.
    const q_heads = try gpa.alloc(f32, num_v_heads * head_k);
    defer gpa.free(q_heads);
    const k_heads = try gpa.alloc(f32, num_v_heads * head_k);
    defer gpa.free(k_heads);
    const v_heads = try gpa.alloc(f32, num_v_heads * head_v);
    defer gpa.free(v_heads);

    const q_scale: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(head_k)));
    for (0..num_k_heads) |h_k| {
        const q_off = h_k * head_k;
        const k_off = key_dim + h_k * head_k;
        // L2-norm Q-row: out = x / sqrt(sum(x²) + eps)
        var q_norm_acc: f32 = 0;
        for (0..head_k) |d| q_norm_acc += mixed_qkv[q_off + d] * mixed_qkv[q_off + d];
        const q_inv = 1.0 / @sqrt(q_norm_acc + eps_l2norm);
        var k_norm_acc: f32 = 0;
        for (0..head_k) |d| k_norm_acc += mixed_qkv[k_off + d] * mixed_qkv[k_off + d];
        const k_inv = 1.0 / @sqrt(k_norm_acc + eps_l2norm);
        // Repeat-interleave to V-head count: each K-head h_k spawns
        // `heads_per_k` consecutive V-heads.
        for (0..heads_per_k) |r| {
            const v_h = h_k * heads_per_k + r;
            for (0..head_k) |d| {
                q_heads[v_h * head_k + d] = mixed_qkv[q_off + d] * q_inv * q_scale;
                k_heads[v_h * head_k + d] = mixed_qkv[k_off + d] * k_inv;
            }
        }
    }
    // V is already V-head-shaped — straight copy.
    @memcpy(v_heads, mixed_qkv[2 * key_dim .. 2 * key_dim + value_dim]);

    // ── 4. Per-head gates ───────────────────────────────────────────
    // beta = sigmoid(b_raw); g = -exp(A_log) * softplus(a_raw + dt_bias)
    // Then the actual per-step decay applied to the recurrent state is
    // exp(g). HF computes A_log, dt_bias, g, beta in fp32; we follow.
    const A_log_buf = try gpa.alloc(f32, num_v_heads);
    defer gpa.free(A_log_buf);
    try cpu_math.tensorToF32Slice(A_log_buf, layer.A_log.?);
    const dt_bias_buf = try gpa.alloc(f32, num_v_heads);
    defer gpa.free(dt_bias_buf);
    try cpu_math.tensorToF32Slice(dt_bias_buf, layer.dt_bias.?);

    const beta = try gpa.alloc(f32, num_v_heads);
    defer gpa.free(beta);
    const g_t = try gpa.alloc(f32, num_v_heads);
    defer gpa.free(g_t);
    for (0..num_v_heads) |h| {
        beta[h] = sigmoid(b_raw[h]);
        const g_h = -@exp(A_log_buf[h]) * softplus(a_raw[h] + dt_bias_buf[h]);
        g_t[h] = @exp(g_h);
    }

    // ── 5. Recurrent state update + readout ─────────────────────────
    // Per V-head:
    //   S *= g_t
    //   kv_mem = Sᵀ k                                  (head_v,)
    //   delta  = (v - kv_mem) * beta                   (head_v,)
    //   S     += k ⊗ delta                             (head_k, head_v)
    //   y      = Sᵀ q                                  (head_v,)
    const core_out = try gpa.alloc(f32, num_v_heads * head_v);
    defer gpa.free(core_out);

    for (0..num_v_heads) |h| {
        const S_off = h * head_k * head_v;
        const q_off = h * head_k;
        const k_off = h * head_k;
        const v_off = h * head_v;

        // Decay
        const gh = g_t[h];
        for (0..head_k * head_v) |i| state.recurrent[S_off + i] *= gh;

        // kv_mem[v] = sum_d S[d, v] * k[d]
        var kv_mem: [256]f32 = undefined; // head_v ≤ 256 in supported configs
        std.debug.assert(head_v <= kv_mem.len);
        for (0..head_v) |v_i| {
            var s: f32 = 0;
            for (0..head_k) |d| {
                s += state.recurrent[S_off + d * head_v + v_i] * k_heads[k_off + d];
            }
            kv_mem[v_i] = s;
        }

        // delta and outer-product update: S[d, v] += k[d] * (v[v] - kv_mem[v]) * beta
        const beta_h = beta[h];
        for (0..head_v) |v_i| {
            const delta_v = (v_heads[v_off + v_i] - kv_mem[v_i]) * beta_h;
            for (0..head_k) |d| {
                state.recurrent[S_off + d * head_v + v_i] += k_heads[k_off + d] * delta_v;
            }
        }

        // Readout: y[v] = sum_d S[d, v] * q[d]
        for (0..head_v) |v_i| {
            var s: f32 = 0;
            for (0..head_k) |d| {
                s += state.recurrent[S_off + d * head_v + v_i] * q_heads[q_off + d];
            }
            core_out[v_off + v_i] = s;
        }
    }

    // ── 6. RMSNormGated, per V-head, then out_proj ──────────────────
    // weight shape is [head_v_dim], shared across heads. The gate `z`
    // is per-head (reshape value_dim into (num_v_heads, head_v_dim)).
    const norm_w = try gpa.alloc(f32, head_v);
    defer gpa.free(norm_w);
    try cpu_math.tensorToF32Slice(norm_w, layer.ssm_norm_weight.?);

    const post_norm = try gpa.alloc(f32, value_dim);
    defer gpa.free(post_norm);

    for (0..num_v_heads) |h| {
        const off = h * head_v;
        var var_acc: f32 = 0.0;
        for (0..head_v) |i| var_acc += core_out[off + i] * core_out[off + i];
        const inv = 1.0 / @sqrt(var_acc / @as(f32, @floatFromInt(head_v)) + cfg.rms_norm_eps);
        for (0..head_v) |i| {
            const normed = norm_w[i] * (core_out[off + i] * inv);
            post_norm[off + i] = normed * silu(z[off + i]);
        }
    }

    // ── 7. Final out_proj ───────────────────────────────────────────
    try cpu_math.matmul_nt(out, post_norm, layer.out_proj.?, 1, hidden, value_dim);
}
