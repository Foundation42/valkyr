//! CPU reference for the Qwen3.5 full-attention layer (the 1-in-4
//! transformer block in the hybrid schedule). Differs from the existing
//! Qwen3 attention path in three places:
//!
//!   1. `q_proj` is twice as wide and emits `(q, gate)` interleaved
//!      per head: row `h` of the proj output is
//!      `[q_h_0, …, q_h_{D-1}, gate_h_0, …, gate_h_{D-1}]`.
//!      The gate is applied as `attn_out * sigmoid(gate)` BEFORE
//!      `o_proj`.
//!   2. RoPE is partial: only the first `partial_rotary_factor *
//!      head_dim` of each head gets rotated; the trailing tail passes
//!      through. For Qwen3.5 head_dim=256, factor=0.25 → rotary_dim=64.
//!      The cos/sin table is sized to `rotary_dim`, not `head_dim`.
//!   3. q_norm / k_norm and the input/post-attn layer norms all use the
//!      Qwen3-Next `(1 + w) * x` form rather than Qwen3's plain `w * x`.
//!      That's already handled by `family.rmsnormAddOne()`.
//!
//! Single-token decode at position `pos` against a per-layer KV cache.
//! KV cache rows `[0..pos+1]` are read for attention; row `pos` is
//! freshly written from this step's k/v projections (post-norm,
//! post-RoPE).

const std = @import("std");
const config_mod = @import("../config.zig");
const safetensors = @import("../safetensors.zig");
const cpu_math = @import("math.zig");
const model_mod = @import("../model.zig");

const Tensor = safetensors.Tensor;
const Layer = model_mod.Layer;
const Config = config_mod.Config;

/// Per-layer KV cache for full-attention layers. Rows are token
/// positions; each row is `num_key_value_heads * head_dim` fp32. Sized
/// for `max_pos` total positions; the caller writes to row `pos`.
pub const KvCache = struct {
    /// `[max_pos, num_kv_heads, head_dim]` row-major.
    k: []f32,
    /// `[max_pos, num_kv_heads, head_dim]` row-major.
    v: []f32,
    max_pos: usize,
    kv_dim: usize,

    pub fn init(gpa: std.mem.Allocator, cfg: Config, max_pos: usize) !KvCache {
        const kv_dim = cfg.num_key_value_heads * cfg.head_dim;
        const k = try gpa.alloc(f32, max_pos * kv_dim);
        @memset(k, 0.0);
        const v = try gpa.alloc(f32, max_pos * kv_dim);
        @memset(v, 0.0);
        return .{ .k = k, .v = v, .max_pos = max_pos, .kv_dim = kv_dim };
    }

    pub fn deinit(self: *KvCache, gpa: std.mem.Allocator) void {
        gpa.free(self.k);
        gpa.free(self.v);
    }
};

inline fn sigmoid(x: f32) f32 {
    return 1.0 / (1.0 + @exp(-x));
}

/// One Qwen3.5 full-attention block, decode step. `x` is the post-
/// input_layernorm hidden state (length `hidden_size`); `out` is the
/// layer's contribution to the residual stream (the caller does the
/// residual add). KV cache is mutated in place — row `pos` gets the
/// freshly projected/normed/rotated K and V vectors.
pub fn decodeStep(
    gpa: std.mem.Allocator,
    cfg: Config,
    layer: Layer,
    kv: *KvCache,
    x: []const f32,
    out: []f32,
    pos: usize,
) !void {
    if (x.len != cfg.hidden_size) return error.LengthMismatch;
    if (out.len != cfg.hidden_size) return error.LengthMismatch;
    if (layer.layer_type != .full_attention) return error.NotFullLayer;
    if (pos >= kv.max_pos) return error.PositionOutOfRange;

    const hidden = cfg.hidden_size;
    const head_dim = cfg.head_dim;
    const n_heads = cfg.num_attention_heads;
    const n_kv = cfg.num_key_value_heads;
    const heads_per_kv = n_heads / n_kv;
    const q_dim = n_heads * head_dim;
    const kv_dim = n_kv * head_dim;
    const rotary_dim: usize = @intFromFloat(@as(f32, @floatFromInt(head_dim)) * cfg.partial_rotary_factor);

    // ── 1. Q-projection (2× wide) split into (q, gate) per head ─────
    // Layout: q_proj output row for head h is
    //   [q_h_0, …, q_h_{D-1}, gate_h_0, …, gate_h_{D-1}].
    // q_proj_rows = 2 * n_heads * head_dim.
    const q_proj_rows: usize = if (cfg.attn_output_gate) 2 * q_dim else q_dim;
    const q_gate = try gpa.alloc(f32, q_proj_rows);
    defer gpa.free(q_gate);
    try cpu_math.matmul_nt(q_gate, x, layer.q_proj.?, 1, q_proj_rows, hidden);

    const q = try gpa.alloc(f32, q_dim);
    defer gpa.free(q);
    var gate_buf: ?[]f32 = null;
    if (cfg.attn_output_gate) {
        gate_buf = try gpa.alloc(f32, q_dim);
        for (0..n_heads) |h| {
            const src = h * 2 * head_dim;
            const dst = h * head_dim;
            @memcpy(q[dst .. dst + head_dim], q_gate[src .. src + head_dim]);
            @memcpy(gate_buf.?[dst .. dst + head_dim], q_gate[src + head_dim .. src + 2 * head_dim]);
        }
    } else {
        @memcpy(q, q_gate);
    }
    defer if (gate_buf) |gb| gpa.free(gb);

    // ── 2. K and V projections ──────────────────────────────────────
    const k = try gpa.alloc(f32, kv_dim);
    defer gpa.free(k);
    try cpu_math.matmul_nt(k, x, layer.k_proj.?, 1, kv_dim, hidden);

    const v = try gpa.alloc(f32, kv_dim);
    defer gpa.free(v);
    try cpu_math.matmul_nt(v, x, layer.v_proj.?, 1, kv_dim, hidden);

    // ── 3. Per-head RMSNorm on Q and K ──────────────────────────────
    // `family.rmsnormAddOne()` covers the (1 + w) form for Qwen3.5.
    if (layer.q_norm) |qn| {
        try cpu_math.rmsnormPerHead(q, q, qn, cfg.rms_norm_eps, n_heads, head_dim, cfg.family);
    }
    if (layer.k_norm) |kn| {
        try cpu_math.rmsnormPerHead(k, k, kn, cfg.rms_norm_eps, n_kv, head_dim, cfg.family);
    }

    // ── 4. Partial RoPE on Q and K ──────────────────────────────────
    const q_rot = try gpa.alloc(f32, q_dim);
    defer gpa.free(q_rot);
    const k_rot = try gpa.alloc(f32, kv_dim);
    defer gpa.free(k_rot);
    try cpu_math.applyRopePartial(q_rot, q, n_heads, head_dim, rotary_dim, pos, cfg.rope_theta);
    try cpu_math.applyRopePartial(k_rot, k, n_kv, head_dim, rotary_dim, pos, cfg.rope_theta);

    // ── 5. Write this step's K and V into the cache at row `pos` ────
    const cache_row_off = pos * kv_dim;
    @memcpy(kv.k[cache_row_off .. cache_row_off + kv_dim], k_rot);
    @memcpy(kv.v[cache_row_off .. cache_row_off + kv_dim], v);

    // ── 6. GQA attention against history rows [0..pos+1] ────────────
    const n_pos = pos + 1;
    const head_out = try gpa.alloc(f32, q_dim);
    defer gpa.free(head_out);

    const inv_sqrt_d: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(head_dim)));

    const scores = try gpa.alloc(f32, n_pos);
    defer gpa.free(scores);

    for (0..n_heads) |h| {
        const kv_h = h / heads_per_kv;
        const q_off = h * head_dim;
        const out_off = h * head_dim;

        // Compute scores[i] = (q · k_cache[i, kv_h, :]) / sqrt(head_dim)
        for (0..n_pos) |i| {
            const k_off = i * kv_dim + kv_h * head_dim;
            var s: f32 = 0;
            for (0..head_dim) |d| s += q_rot[q_off + d] * kv.k[k_off + d];
            scores[i] = s * inv_sqrt_d;
        }
        cpu_math.softmax(scores);

        // out_h[d] = sum_i scores[i] * v_cache[i, kv_h, d]
        for (0..head_dim) |d| head_out[out_off + d] = 0.0;
        for (0..n_pos) |i| {
            const v_off = i * kv_dim + kv_h * head_dim;
            const w = scores[i];
            for (0..head_dim) |d| head_out[out_off + d] += w * kv.v[v_off + d];
        }
    }

    // ── 7. attn_output_gate: head_out *= sigmoid(gate) ──────────────
    if (gate_buf) |gb| {
        for (head_out, gb) |*o, g| o.* *= sigmoid(g);
    }

    // ── 8. o_proj ───────────────────────────────────────────────────
    try cpu_math.matmul_nt(out, head_out, layer.o_proj.?, 1, hidden, q_dim);
}

/// n_q-batched prefill against the same KV cache `decodeStep` writes
/// into. Shapes: `x`/`out` are `n_q × hidden_size` row-major; queries
/// occupy positions `[pos_start..pos_start+n_q)` and attend causally
/// against keys `[0..pos_start+t+1]` for query row `t`. Mirrors
/// `decodeStep` in dispatch order — the verify oracle for the GPU
/// batched-prefill path.
pub fn prefillStep(
    gpa: std.mem.Allocator,
    cfg: Config,
    layer: Layer,
    kv: *KvCache,
    x: []const f32,
    out: []f32,
    pos_start: usize,
    n_q: usize,
) !void {
    if (x.len != n_q * cfg.hidden_size) return error.LengthMismatch;
    if (out.len != n_q * cfg.hidden_size) return error.LengthMismatch;
    if (layer.layer_type != .full_attention) return error.NotFullLayer;
    if (pos_start + n_q > kv.max_pos) return error.PositionOutOfRange;
    if (n_q == 0) return error.EmptyBatch;

    const hidden = cfg.hidden_size;
    const head_dim = cfg.head_dim;
    const n_heads = cfg.num_attention_heads;
    const n_kv = cfg.num_key_value_heads;
    const heads_per_kv = n_heads / n_kv;
    const q_dim = n_heads * head_dim;
    const kv_dim = n_kv * head_dim;
    const rotary_dim: usize = @intFromFloat(@as(f32, @floatFromInt(head_dim)) * cfg.partial_rotary_factor);

    // ── 1. Q-projection (2× wide if attn_output_gate) ──────────────
    const q_proj_rows: usize = if (cfg.attn_output_gate) 2 * q_dim else q_dim;
    const q_gate = try gpa.alloc(f32, n_q * q_proj_rows);
    defer gpa.free(q_gate);
    try cpu_math.matmul_nt(q_gate, x, layer.q_proj.?, n_q, q_proj_rows, hidden);

    const q = try gpa.alloc(f32, n_q * q_dim);
    defer gpa.free(q);
    var gate_buf: ?[]f32 = null;
    if (cfg.attn_output_gate) {
        gate_buf = try gpa.alloc(f32, n_q * q_dim);
        for (0..n_q) |t| {
            const src_row = t * q_proj_rows;
            const dst_row = t * q_dim;
            for (0..n_heads) |h| {
                const src = src_row + h * 2 * head_dim;
                const dst = dst_row + h * head_dim;
                @memcpy(q[dst .. dst + head_dim], q_gate[src .. src + head_dim]);
                @memcpy(gate_buf.?[dst .. dst + head_dim], q_gate[src + head_dim .. src + 2 * head_dim]);
            }
        }
    } else {
        @memcpy(q, q_gate);
    }
    defer if (gate_buf) |gb| gpa.free(gb);

    // ── 2. K and V projections ─────────────────────────────────────
    const k = try gpa.alloc(f32, n_q * kv_dim);
    defer gpa.free(k);
    try cpu_math.matmul_nt(k, x, layer.k_proj.?, n_q, kv_dim, hidden);

    const v = try gpa.alloc(f32, n_q * kv_dim);
    defer gpa.free(v);
    try cpu_math.matmul_nt(v, x, layer.v_proj.?, n_q, kv_dim, hidden);

    // ── 3. Per-head RMSNorm on Q and K (per row, per head) ─────────
    if (layer.q_norm) |qn| {
        for (0..n_q) |t| {
            const row = t * q_dim;
            try cpu_math.rmsnormPerHead(q[row .. row + q_dim], q[row .. row + q_dim], qn, cfg.rms_norm_eps, n_heads, head_dim, cfg.family);
        }
    }
    if (layer.k_norm) |kn| {
        for (0..n_q) |t| {
            const row = t * kv_dim;
            try cpu_math.rmsnormPerHead(k[row .. row + kv_dim], k[row .. row + kv_dim], kn, cfg.rms_norm_eps, n_kv, head_dim, cfg.family);
        }
    }

    // ── 4. Partial RoPE per row at its own absolute position ───────
    const q_rot = try gpa.alloc(f32, n_q * q_dim);
    defer gpa.free(q_rot);
    const k_rot = try gpa.alloc(f32, n_q * kv_dim);
    defer gpa.free(k_rot);
    for (0..n_q) |t| {
        const pos = pos_start + t;
        const q_row = q[t * q_dim .. (t + 1) * q_dim];
        const q_rot_row = q_rot[t * q_dim .. (t + 1) * q_dim];
        const k_row = k[t * kv_dim .. (t + 1) * kv_dim];
        const k_rot_row = k_rot[t * kv_dim .. (t + 1) * kv_dim];
        try cpu_math.applyRopePartial(q_rot_row, q_row, n_heads, head_dim, rotary_dim, pos, cfg.rope_theta);
        try cpu_math.applyRopePartial(k_rot_row, k_row, n_kv, head_dim, rotary_dim, pos, cfg.rope_theta);
    }

    // ── 5. Append n_q rows to the KV cache ─────────────────────────
    for (0..n_q) |t| {
        const dst_off = (pos_start + t) * kv_dim;
        @memcpy(kv.k[dst_off .. dst_off + kv_dim], k_rot[t * kv_dim .. (t + 1) * kv_dim]);
        @memcpy(kv.v[dst_off .. dst_off + kv_dim], v[t * kv_dim .. (t + 1) * kv_dim]);
    }

    // ── 6. Causal GQA attention per query row ──────────────────────
    const head_out = try gpa.alloc(f32, n_q * q_dim);
    defer gpa.free(head_out);

    const inv_sqrt_d: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(head_dim)));

    const max_n_pos = pos_start + n_q;
    const scores = try gpa.alloc(f32, max_n_pos);
    defer gpa.free(scores);

    for (0..n_q) |t| {
        const n_pos = pos_start + t + 1;
        for (0..n_heads) |h| {
            const kv_h = h / heads_per_kv;
            const q_off = t * q_dim + h * head_dim;
            const out_off = t * q_dim + h * head_dim;

            for (0..n_pos) |i| {
                const k_off = i * kv_dim + kv_h * head_dim;
                var s: f32 = 0;
                for (0..head_dim) |d| s += q_rot[q_off + d] * kv.k[k_off + d];
                scores[i] = s * inv_sqrt_d;
            }
            cpu_math.softmax(scores[0..n_pos]);

            for (0..head_dim) |d| head_out[out_off + d] = 0.0;
            for (0..n_pos) |i| {
                const v_off = i * kv_dim + kv_h * head_dim;
                const w = scores[i];
                for (0..head_dim) |d| head_out[out_off + d] += w * kv.v[v_off + d];
            }
        }
    }

    // ── 7. attn_output_gate: head_out *= sigmoid(gate) ─────────────
    if (gate_buf) |gb| {
        for (head_out, gb) |*o, g| o.* *= sigmoid(g);
    }

    // ── 8. o_proj per row ──────────────────────────────────────────
    try cpu_math.matmul_nt(out, head_out, layer.o_proj.?, n_q, hidden, q_dim);
}
