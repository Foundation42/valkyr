//! Per-layer parity tests (CPU and GPU) invoked via `--rmsnorm-test`,
//! `--qproj-test`, `--rope-test`, `--attention-test`, `--layer0-test`,
//! `--tq4-kv-test`, `--gen-tq4v`, `--gen`, `--gpu-load`,
//! `--gpu-rmsnorm-test`, `--gpu-rope-test`, `--gpu-geglu-test`,
//! `--gpu-qproj-test`, `--gpu-layer0-test`. All take a real model dir +
//! token id and compare bit-by-bit against a CPU oracle. Extracted from
//! main.zig.

const std = @import("std");
const vk = @import("../gpu/vk.zig");
const buffer = @import("../gpu/buffer.zig");
const pipeline = @import("../gpu/pipeline.zig");
const dtype = @import("../dtype.zig");
const model_mod = @import("../model.zig");
const tokenizer_mod = @import("../tokenizer.zig");
const cpu_forward = @import("../cpu/forward.zig");
const cpu_math = @import("../cpu/math.zig");
const safetensors = @import("../safetensors.zig");
const turboquant = @import("../cpu/turboquant.zig");
const gpu_model = @import("../gpu/model.zig");
const gpu_scratch = @import("../gpu/scratch.zig");
const runtime = @import("../runtime.zig");
const runtime_hybrid = @import("../runtime_hybrid.zig");
const shaders = @import("shaders");

// Push-constant aliases for the moved code; main.zig has its own copies
// for any code still living there.
const RmsnormPush = runtime.RmsnormPush;
const MatmulPush = runtime.MatmulPush;
const RopePush = runtime.RopePush;
const GegluPush = runtime.GegluPush;

// ── rmsnorm-test: first math primitive on a real layer ──────────────

pub fn runRmsnormTest(allocator: std.mem.Allocator, dir_path: []const u8, token_id: u32) !void {
    var model = try model_mod.Model.load(allocator, dir_path);
    defer model.deinit();
    const cfg = model.config;
    if (token_id >= cfg.vocab_size) return error.OutOfRange;

    const stdout = std.io.getStdOut().writer();
    try stdout.print("rmsnorm test on layer 0 input_layernorm — token {d}\n\n", .{token_id});

    // Materialise the embedding row.
    const x = try allocator.alloc(f32, cfg.hidden_size);
    defer allocator.free(x);
    try cpu_math.embedRowAsF32(x, model.embed_tokens, token_id);

    // Gemma scales the embedding by sqrt(hidden_size) before the first
    // block. Without this the RMS of `x` is way under 1, and rmsnorm
    // ends up amplifying noise — the post-rmsnorm activations would be
    // garbage and every downstream test would lie. Apply unconditionally
    // for now since we're Gemma-only; when we add Llama, gate on family.
    if (cfg.family.embedScalesByDim()) {
        const scale: f32 = @sqrt(@as(f32, @floatFromInt(cfg.hidden_size)));
        for (x) |*xi| xi.* *= scale;
    }

    const x_rms = blk: {
        var s: f32 = 0;
        for (x) |v| s += v * v;
        break :blk @sqrt(s / @as(f32, @floatFromInt(x.len)));
    };
    try stdout.print("post-scale embedding rms = {d:.6}\n", .{x_rms});

    // Apply layer 0's input_layernorm.
    const y = try allocator.alloc(f32, cfg.hidden_size);
    defer allocator.free(y);
    try cpu_math.rmsnorm(y, x, model.layers[0].input_layernorm, cfg.rms_norm_eps, cfg.family);

    const n_show: usize = @min(16, y.len);
    for (y[0..n_show], 0..) |v, i| try stdout.print("  [{d:>4}] {d:.6}\n", .{ i, v });

    var min_v: f32 = std.math.inf(f32);
    var max_v: f32 = -std.math.inf(f32);
    var sum_sq: f64 = 0;
    var nan_count: usize = 0;
    var inf_count: usize = 0;
    for (y) |v| {
        if (std.math.isNan(v)) {
            nan_count += 1;
            continue;
        }
        if (std.math.isInf(v)) {
            inf_count += 1;
            continue;
        }
        if (v < min_v) min_v = v;
        if (v > max_v) max_v = v;
        sum_sq += @as(f64, v) * @as(f64, v);
    }
    const post_rms = std.math.sqrt(sum_sq / @as(f64, @floatFromInt(y.len)));
    try stdout.print("\noutput stats: min={d:.6} max={d:.6} rms={d:.6} nan={d} inf={d}\n", .{
        min_v, max_v, post_rms, nan_count, inf_count,
    });
}

// ── qproj-test: rmsnorm → matmul against layer 0's q_proj ───────────

pub fn runQprojTest(allocator: std.mem.Allocator, dir_path: []const u8, token_id: u32) !void {
    var model = try model_mod.Model.load(allocator, dir_path);
    defer model.deinit();
    const cfg = model.config;
    if (token_id >= cfg.vocab_size) return error.OutOfRange;

    const stdout = std.io.getStdOut().writer();
    try stdout.print("qproj test on layer 0 — token {d}\n\n", .{token_id});

    // Embedding → scale → rmsnorm → q_proj.
    const x = try allocator.alloc(f32, cfg.hidden_size);
    defer allocator.free(x);
    try cpu_math.embedRowAsF32(x, model.embed_tokens, token_id);
    if (cfg.family.embedScalesByDim()) {
        const scale: f32 = @sqrt(@as(f32, @floatFromInt(cfg.hidden_size)));
        for (x) |*xi| xi.* *= scale;
    }

    const x_norm = try allocator.alloc(f32, cfg.hidden_size);
    defer allocator.free(x_norm);
    try cpu_math.rmsnorm(x_norm, x, model.layers[0].input_layernorm, cfg.rms_norm_eps, cfg.family);

    // Q has dimension n_heads × head_dim.
    const q_dim = cfg.num_attention_heads * cfg.head_dim;
    const q = try allocator.alloc(f32, q_dim);
    defer allocator.free(q);

    const t0 = std.time.nanoTimestamp();
    try cpu_math.matmul_nt(q, x_norm, model.layers[0].q_proj.?, 1, q_dim, cfg.hidden_size);
    const t1 = std.time.nanoTimestamp();
    const ms = @as(f64, @floatFromInt(t1 - t0)) / 1_000_000.0;

    try stdout.print("matmul_nt [1, {d}] = [1, {d}] · [{d}, {d}]ᵀ — {d:.2} ms\n\n", .{
        cfg.hidden_size, q_dim, q_dim, cfg.hidden_size, ms,
    });

    const n_show: usize = @min(16, q.len);
    for (q[0..n_show], 0..) |v, i| try stdout.print("  q[{d:>4}] {d:.6}\n", .{ i, v });

    var min_v: f32 = std.math.inf(f32);
    var max_v: f32 = -std.math.inf(f32);
    var sum_sq: f64 = 0;
    var nan_count: usize = 0;
    var inf_count: usize = 0;
    for (q) |v| {
        if (std.math.isNan(v)) {
            nan_count += 1;
            continue;
        }
        if (std.math.isInf(v)) {
            inf_count += 1;
            continue;
        }
        if (v < min_v) min_v = v;
        if (v > max_v) max_v = v;
        sum_sq += @as(f64, v) * @as(f64, v);
    }
    const q_rms = std.math.sqrt(sum_sq / @as(f64, @floatFromInt(q.len)));
    try stdout.print("\nq stats: min={d:.6} max={d:.6} rms={d:.6} nan={d} inf={d}\n", .{
        min_v, max_v, q_rms, nan_count, inf_count,
    });
}

// ── rope-test: produce Q, apply RoPE at pos 0 and pos 1 ─────────────

pub fn runRopeTest(allocator: std.mem.Allocator, dir_path: []const u8, token_id: u32) !void {
    var model = try model_mod.Model.load(allocator, dir_path);
    defer model.deinit();
    const cfg = model.config;

    // Reuse the qproj chain to get Q.
    const x = try allocator.alloc(f32, cfg.hidden_size);
    defer allocator.free(x);
    try cpu_math.embedRowAsF32(x, model.embed_tokens, token_id);
    if (cfg.family.embedScalesByDim()) {
        const scale: f32 = @sqrt(@as(f32, @floatFromInt(cfg.hidden_size)));
        for (x) |*xi| xi.* *= scale;
    }
    const x_norm = try allocator.alloc(f32, cfg.hidden_size);
    defer allocator.free(x_norm);
    try cpu_math.rmsnorm(x_norm, x, model.layers[0].input_layernorm, cfg.rms_norm_eps, cfg.family);

    const q_dim = cfg.num_attention_heads * cfg.head_dim;
    const q = try allocator.alloc(f32, q_dim);
    defer allocator.free(q);
    try cpu_math.matmul_nt(q, x_norm, model.layers[0].q_proj.?, 1, q_dim, cfg.hidden_size);

    const stdout = std.io.getStdOut().writer();
    try stdout.print("RoPE on Q [n_heads={d}, head_dim={d}] for token {d}\n\n", .{
        cfg.num_attention_heads, cfg.head_dim, token_id,
    });

    // pos = 0: must equal Q.
    const q_pos0 = try allocator.alloc(f32, q_dim);
    defer allocator.free(q_pos0);
    try cpu_math.applyRope(q_pos0, q, cfg.num_attention_heads, cfg.head_dim, 0, cfg.rope_theta);
    var pos0_ok = true;
    for (q, q_pos0) |a, b| if (a != b) {
        pos0_ok = false;
        break;
    };
    try stdout.print("pos=0 identity: {s}\n", .{if (pos0_ok) "OK" else "FAIL"});

    // pos = 1: rotated.
    const q_pos1 = try allocator.alloc(f32, q_dim);
    defer allocator.free(q_pos1);
    try cpu_math.applyRope(q_pos1, q, cfg.num_attention_heads, cfg.head_dim, 1, cfg.rope_theta);

    // Print head 0, first 8 dims of each pair, before vs after pos=1.
    try stdout.print("\nhead 0, pre vs post pos=1 RoPE (first 8 dim pairs):\n", .{});
    const half = cfg.head_dim / 2;
    for (0..8) |j| {
        const a_pre = q[j];
        const b_pre = q[j + half];
        const a_post = q_pos1[j];
        const b_post = q_pos1[j + half];
        try stdout.print("  pair ({d:>3}, {d:>3}):  pre=({d:.4}, {d:.4})  post=({d:.4}, {d:.4})\n", .{
            j, j + half, a_pre, b_pre, a_post, b_post,
        });
    }

    // Sanity: norm of each (j, j+half) pair must be invariant — RoPE is
    // a rotation, it preserves length per pair.
    var max_err: f32 = 0;
    for (0..cfg.num_attention_heads) |h| {
        const off = h * cfg.head_dim;
        for (0..half) |j| {
            const a0 = q[off + j];
            const b0 = q[off + j + half];
            const a1 = q_pos1[off + j];
            const b1 = q_pos1[off + j + half];
            const n_pre = a0 * a0 + b0 * b0;
            const n_post = a1 * a1 + b1 * b1;
            const err = @abs(n_pre - n_post);
            if (err > max_err) max_err = err;
        }
    }
    try stdout.print("\nrotation invariant: max ||pair||² delta = {e:.2}  (must be tiny)\n", .{max_err});
}

// ── attention-test: full single-position attention through layer 0 ──

pub fn runAttentionTest(allocator: std.mem.Allocator, dir_path: []const u8, token_id: u32) !void {
    var model = try model_mod.Model.load(allocator, dir_path);
    defer model.deinit();
    const cfg = model.config;
    const layer = model.layers[0];

    const n_heads = cfg.num_attention_heads;
    const n_kv = cfg.num_key_value_heads;
    const head_dim = cfg.head_dim;
    const q_dim = n_heads * head_dim;
    const kv_dim = n_kv * head_dim;
    const heads_per_kv = n_heads / n_kv; // 8 for Gemma 2B (MQA)

    const stdout = std.io.getStdOut().writer();
    try stdout.print(
        \\full attention block on layer 0 — token {d} run twice (pos 0, pos 1)
        \\config: n_heads={d}, n_kv_heads={d}, head_dim={d}, hidden={d}
        \\
        \\
    , .{ token_id, n_heads, n_kv, head_dim, cfg.hidden_size });

    // 2-position KV cache, flat [pos][n_kv * head_dim].
    const max_pos: usize = 2;
    const k_cache = try allocator.alloc(f32, max_pos * kv_dim);
    defer allocator.free(k_cache);
    const v_cache = try allocator.alloc(f32, max_pos * kv_dim);
    defer allocator.free(v_cache);

    // Scratch buffers reused across positions.
    const x = try allocator.alloc(f32, cfg.hidden_size);
    defer allocator.free(x);
    const x_norm = try allocator.alloc(f32, cfg.hidden_size);
    defer allocator.free(x_norm);
    const q = try allocator.alloc(f32, q_dim);
    defer allocator.free(q);
    const q_rot = try allocator.alloc(f32, q_dim);
    defer allocator.free(q_rot);
    const k = try allocator.alloc(f32, kv_dim);
    defer allocator.free(k);
    const k_rot = try allocator.alloc(f32, kv_dim);
    defer allocator.free(k_rot);
    const v = try allocator.alloc(f32, kv_dim);
    defer allocator.free(v);
    const head_out = try allocator.alloc(f32, q_dim);
    defer allocator.free(head_out);
    const attn_out = try allocator.alloc(f32, cfg.hidden_size);
    defer allocator.free(attn_out);

    const inv_sqrt_dim: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(head_dim)));

    var pos: usize = 0;
    while (pos < max_pos) : (pos += 1) {
        // ── pre-attention: embed → scale → rmsnorm ──────────────────
        try cpu_math.embedRowAsF32(x, model.embed_tokens, token_id);
        if (cfg.family.embedScalesByDim()) {
            const s: f32 = @sqrt(@as(f32, @floatFromInt(cfg.hidden_size)));
            for (x) |*xi| xi.* *= s;
        }
        try cpu_math.rmsnorm(x_norm, x, layer.input_layernorm, cfg.rms_norm_eps, cfg.family);

        // ── Q, K, V projections ─────────────────────────────────────
        try cpu_math.matmul_nt(q, x_norm, layer.q_proj.?, 1, q_dim, cfg.hidden_size);
        try cpu_math.matmul_nt(k, x_norm, layer.k_proj.?, 1, kv_dim, cfg.hidden_size);
        try cpu_math.matmul_nt(v, x_norm, layer.v_proj.?, 1, kv_dim, cfg.hidden_size);

        // ── RoPE on Q and K (V is not rotated) ──────────────────────
        try cpu_math.applyRope(q_rot, q, n_heads, head_dim, pos, cfg.rope_theta);
        try cpu_math.applyRope(k_rot, k, n_kv, head_dim, pos, cfg.rope_theta);

        // ── Append to KV cache ──────────────────────────────────────
        @memcpy(k_cache[pos * kv_dim ..][0..kv_dim], k_rot);
        @memcpy(v_cache[pos * kv_dim ..][0..kv_dim], v);

        // ── Attention: scores → softmax → weighted V sum ────────────
        const n_pos = pos + 1;
        const scores = try allocator.alloc(f32, n_pos);
        defer allocator.free(scores);

        var print_softmax_pos: ?usize = null;
        for (0..n_heads) |h| {
            const kv_h = h / heads_per_kv;
            const q_off = h * head_dim;

            // Score against every cached position.
            for (0..n_pos) |p| {
                const k_off = p * kv_dim + kv_h * head_dim;
                var s: f32 = 0;
                for (0..head_dim) |d| s += q_rot[q_off + d] * k_cache[k_off + d];
                scores[p] = s * inv_sqrt_dim;
            }

            cpu_math.softmax(scores);
            if (h == 0 and pos == 1) print_softmax_pos = pos;

            // Sum over positions: head_out[h] = Σ scores[p] * v_cache[p, kv_h]
            const out_off = h * head_dim;
            for (0..head_dim) |d| head_out[out_off + d] = 0;
            for (0..n_pos) |p| {
                const v_off = p * kv_dim + kv_h * head_dim;
                const w = scores[p];
                for (0..head_dim) |d| head_out[out_off + d] += w * v_cache[v_off + d];
            }
        }

        // ── Output projection: head_out @ o_proj^T → attn_out ───────
        try cpu_math.matmul_nt(attn_out, head_out, layer.o_proj.?, 1, cfg.hidden_size, q_dim);

        // ── Stats ───────────────────────────────────────────────────
        var min_v: f32 = std.math.inf(f32);
        var max_v: f32 = -std.math.inf(f32);
        var sum_sq: f64 = 0;
        for (attn_out) |val| {
            if (val < min_v) min_v = val;
            if (val > max_v) max_v = val;
            sum_sq += @as(f64, val) * @as(f64, val);
        }
        const rms = std.math.sqrt(sum_sq / @as(f64, @floatFromInt(attn_out.len)));

        try stdout.print("pos {d}: attn_out min={d:.6} max={d:.6} rms={d:.6}\n", .{
            pos, min_v, max_v, rms,
        });
        if (print_softmax_pos != null) {
            // Re-run softmax on head 0 so we can print it (the loop
            // above destroys scores in-place per head).
            const dbg_scores = try allocator.alloc(f32, n_pos);
            defer allocator.free(dbg_scores);
            for (0..n_pos) |p| {
                const k_off = p * kv_dim;
                var s: f32 = 0;
                for (0..head_dim) |d| s += q_rot[d] * k_cache[k_off + d];
                dbg_scores[p] = s * inv_sqrt_dim;
            }
            const raw0 = dbg_scores[0];
            const raw1 = dbg_scores[1];
            cpu_math.softmax(dbg_scores);
            try stdout.print("       head 0 raw scores=({d:.4}, {d:.4})  softmax=({d:.4}, {d:.4}) sum={d:.6}\n", .{
                raw0, raw1, dbg_scores[0], dbg_scores[1], dbg_scores[0] + dbg_scores[1],
            });
        }
    }
}

// ── layer0-test: complete transformer block (attn + FFN + residuals) ──

pub fn runLayer0Test(allocator: std.mem.Allocator, dir_path: []const u8, token_id: u32) !void {
    var model = try model_mod.Model.load(allocator, dir_path);
    defer model.deinit();
    const cfg = model.config;
    const layer = model.layers[0];

    const n_heads = cfg.num_attention_heads;
    const n_kv = cfg.num_key_value_heads;
    const head_dim = cfg.head_dim;
    const q_dim = n_heads * head_dim;
    const kv_dim = n_kv * head_dim;
    const heads_per_kv = n_heads / n_kv;
    const inter = cfg.intermediate_size;
    const hidden = cfg.hidden_size;

    const stdout = std.io.getStdOut().writer();
    try stdout.print(
        \\full layer 0 forward — token {d}, position 0 (single token, no history)
        \\config: hidden={d}, intermediate={d}, n_heads={d}, kv_heads={d}, head_dim={d}
        \\
        \\
    , .{ token_id, hidden, inter, n_heads, n_kv, head_dim });

    // ── Stage 0: residual stream init from embedding ────────────────
    const x = try allocator.alloc(f32, hidden);
    defer allocator.free(x);
    try cpu_math.embedRowAsF32(x, model.embed_tokens, token_id);
    if (cfg.family.embedScalesByDim()) {
        const s: f32 = @sqrt(@as(f32, @floatFromInt(hidden)));
        for (x) |*xi| xi.* *= s;
    }
    try printStreamStats(stdout, "embed (post-scale)", x);

    // ── Stage 1: rmsnorm₁ → Q/K/V → RoPE → attention → o_proj ───────
    const x_norm1 = try allocator.alloc(f32, hidden);
    defer allocator.free(x_norm1);
    try cpu_math.rmsnorm(x_norm1, x, layer.input_layernorm, cfg.rms_norm_eps, cfg.family);

    const q = try allocator.alloc(f32, q_dim);
    defer allocator.free(q);
    const k = try allocator.alloc(f32, kv_dim);
    defer allocator.free(k);
    const v = try allocator.alloc(f32, kv_dim);
    defer allocator.free(v);
    try cpu_math.matmul_nt(q, x_norm1, layer.q_proj.?, 1, q_dim, hidden);
    try cpu_math.matmul_nt(k, x_norm1, layer.k_proj.?, 1, kv_dim, hidden);
    try cpu_math.matmul_nt(v, x_norm1, layer.v_proj.?, 1, kv_dim, hidden);

    const q_rot = try allocator.alloc(f32, q_dim);
    defer allocator.free(q_rot);
    const k_rot = try allocator.alloc(f32, kv_dim);
    defer allocator.free(k_rot);
    try cpu_math.applyRope(q_rot, q, n_heads, head_dim, 0, cfg.rope_theta);
    try cpu_math.applyRope(k_rot, k, n_kv, head_dim, 0, cfg.rope_theta);

    // Single-position attention: softmax over 1 score is 1.0 → head_out
    // is exactly v (broadcast across query heads sharing the kv head).
    const head_out = try allocator.alloc(f32, q_dim);
    defer allocator.free(head_out);
    for (0..n_heads) |h| {
        const kv_h = h / heads_per_kv;
        const v_off = kv_h * head_dim;
        const out_off = h * head_dim;
        @memcpy(head_out[out_off .. out_off + head_dim], v[v_off .. v_off + head_dim]);
    }
    // (q_rot and k_rot computed above are unused at pos 0 since the
    // softmax collapses to 1.0 — kept for symmetry with the multi-
    // position path we'll wire when we have an actual prompt.)

    const attn_out = try allocator.alloc(f32, hidden);
    defer allocator.free(attn_out);
    try cpu_math.matmul_nt(attn_out, head_out, layer.o_proj.?, 1, hidden, q_dim);
    try printStreamStats(stdout, "attn output (pre-residual)", attn_out);

    // ── Stage 2: residual add ───────────────────────────────────────
    const mid = try allocator.alloc(f32, hidden);
    defer allocator.free(mid);
    for (mid, x, attn_out) |*m, xi, ai| m.* = xi + ai;
    try printStreamStats(stdout, "residual after attn", mid);

    // ── Stage 3: rmsnorm₂ → GeGLU FFN → down_proj ──────────────────
    const mid_norm = try allocator.alloc(f32, hidden);
    defer allocator.free(mid_norm);
    try cpu_math.rmsnorm(mid_norm, mid, layer.post_attention_layernorm, cfg.rms_norm_eps, cfg.family);

    const gate = try allocator.alloc(f32, inter);
    defer allocator.free(gate);
    const up = try allocator.alloc(f32, inter);
    defer allocator.free(up);
    try cpu_math.matmul_nt(gate, mid_norm, layer.gate_proj, 1, inter, hidden);
    try cpu_math.matmul_nt(up, mid_norm, layer.up_proj, 1, inter, hidden);

    const fused = try allocator.alloc(f32, inter);
    defer allocator.free(fused);
    try cpu_math.geglu(fused, gate, up);
    try printStreamStats(stdout, "geglu(gate)·up (intermediate)", fused);

    const ffn_out = try allocator.alloc(f32, hidden);
    defer allocator.free(ffn_out);
    try cpu_math.matmul_nt(ffn_out, fused, layer.down_proj, 1, hidden, inter);
    try printStreamStats(stdout, "ffn output (pre-residual)", ffn_out);

    // ── Stage 4: residual add → block output ────────────────────────
    const block_out = try allocator.alloc(f32, hidden);
    defer allocator.free(block_out);
    for (block_out, mid, ffn_out) |*o, m, f| o.* = m + f;
    try printStreamStats(stdout, "layer 0 output", block_out);
}

// ── tq4-kv-test: real Gemma K/V vectors round-tripped through TQ4 ──
//
// Walks the full 18-layer forward pass for one input token, and at
// every layer captures the (post-RoPE) K vector and the (pre-RoPE) V
// vector — the two quantities that would actually live in a
// TurboQuant-compressed KV cache. Each 256-element vector is run
// through quantizeBlockTQ4 + dequantizeBlockTQ4, and the per-coord
// MSE / max |Δ| / norm-preservation is reported so we have a real-
// data reference for what the synthetic Gaussian numbers looked like
// (MSE 0.00658, norm-rel-err 1.09e-4 from the smoke test).
//
// Gemma 2B has n_kv_heads=1 and head_dim=256, so each layer's K and V
// is exactly one TQ4 block. 18 layers × 2 (K, V) = 36 sample blocks.

pub fn runTq4KvTest(allocator: std.mem.Allocator, dir_path: []const u8, token_id: u32) !void {
    var model = try model_mod.Model.load(allocator, dir_path);
    defer model.deinit();
    const cfg = model.config;
    if (token_id >= cfg.vocab_size) return error.OutOfRange;

    if (cfg.head_dim != turboquant.block_size_tq4) {
        std.debug.print("head_dim ({d}) != TQ4 block size ({d}); only Gemma 2B supported in chunk 1.4\n", .{ cfg.head_dim, turboquant.block_size_tq4 });
        return error.HeadDimMismatch;
    }
    if (cfg.num_key_value_heads != 1) {
        std.debug.print("num_kv_heads ({d}) != 1; this test currently assumes Gemma MQA\n", .{cfg.num_key_value_heads});
        return error.KvHeadCountMismatch;
    }

    const stdout = std.io.getStdOut().writer();
    try stdout.print("TQ4 KV round-trip on real Gemma activations — token {d}\n\n", .{token_id});

    const hidden = cfg.hidden_size;
    const inter = cfg.intermediate_size;
    const n_heads = cfg.num_attention_heads;
    const n_kv = cfg.num_key_value_heads;
    const head_dim = cfg.head_dim;
    const q_dim = n_heads * head_dim;
    const kv_dim = n_kv * head_dim; // == 256 for Gemma 2B
    const heads_per_kv = n_heads / n_kv;
    const pos: usize = 0;

    // Per-layer scratch (mirrors cpu/forward.zig, no shortcuts so we
    // see real activations at each step).
    const stream = try allocator.alloc(f32, hidden);
    defer allocator.free(stream);
    const x_norm = try allocator.alloc(f32, hidden);
    defer allocator.free(x_norm);
    const q = try allocator.alloc(f32, q_dim);
    defer allocator.free(q);
    const k = try allocator.alloc(f32, kv_dim);
    defer allocator.free(k);
    const v = try allocator.alloc(f32, kv_dim);
    defer allocator.free(v);
    const q_rot = try allocator.alloc(f32, q_dim);
    defer allocator.free(q_rot);
    const k_rot = try allocator.alloc(f32, kv_dim);
    defer allocator.free(k_rot);
    const head_out = try allocator.alloc(f32, q_dim);
    defer allocator.free(head_out);
    const attn_out = try allocator.alloc(f32, hidden);
    defer allocator.free(attn_out);
    const mid_norm = try allocator.alloc(f32, hidden);
    defer allocator.free(mid_norm);
    const gate = try allocator.alloc(f32, inter);
    defer allocator.free(gate);
    const up = try allocator.alloc(f32, inter);
    defer allocator.free(up);
    const fused = try allocator.alloc(f32, inter);
    defer allocator.free(fused);
    const ffn_out = try allocator.alloc(f32, hidden);
    defer allocator.free(ffn_out);

    // Embedding + Gemma sqrt(hidden) scale.
    try cpu_math.embedRowAsF32(stream, model.embed_tokens, token_id);
    if (cfg.family.embedScalesByDim()) {
        const s: f32 = @sqrt(@as(f32, @floatFromInt(hidden)));
        for (stream) |*xi| xi.* *= s;
    }

    // Aggregate stats.
    var k_mse_sum: f64 = 0;
    var v_mse_sum: f64 = 0;
    var k_max_err: f32 = 0;
    var v_max_err: f32 = 0;
    var k_max_norm_rel: f32 = 0;
    var v_max_norm_rel: f32 = 0;

    try stdout.print("layer  K MSE      K max|Δ|   K norm-rel  V MSE      V max|Δ|   V norm-rel\n", .{});
    try stdout.print("-----  ---------  ---------  ----------  ---------  ---------  ----------\n", .{});

    for (model.layers, 0..) |layer, layer_idx| {
        // Pre-attention rmsnorm.
        try cpu_math.rmsnorm(x_norm, stream, layer.input_layernorm, cfg.rms_norm_eps, cfg.family);

        // Q/K/V projections.
        try cpu_math.matmul_nt(q, x_norm, layer.q_proj.?, 1, q_dim, hidden);
        try cpu_math.matmul_nt(k, x_norm, layer.k_proj.?, 1, kv_dim, hidden);
        try cpu_math.matmul_nt(v, x_norm, layer.v_proj.?, 1, kv_dim, hidden);

        // RoPE on Q and K.
        try cpu_math.applyRope(q_rot, q, n_heads, head_dim, pos, cfg.rope_theta);
        try cpu_math.applyRope(k_rot, k, n_kv, head_dim, pos, cfg.rope_theta);

        // ── TQ4 round-trip on K_rot and V ───────────────────────────
        const k_block_in: *const [256]f32 = @ptrCast(k_rot.ptr);
        const v_block_in: *const [256]f32 = @ptrCast(v.ptr);
        var k_blk: turboquant.BlockTQ4(256) = undefined;
        var v_blk: turboquant.BlockTQ4(256) = undefined;
        turboquant.quantizeBlockTQ4(256, k_block_in, &k_blk);
        turboquant.quantizeBlockTQ4(256, v_block_in, &v_blk);

        var k_recon: [256]f32 = undefined;
        var v_recon: [256]f32 = undefined;
        turboquant.dequantizeBlockTQ4(256, &k_blk, &k_recon);
        turboquant.dequantizeBlockTQ4(256, &v_blk, &v_recon);

        const k_stats = blockStats(k_block_in, &k_recon);
        const v_stats = blockStats(v_block_in, &v_recon);

        try stdout.print("{d:>5}  {d:.6}   {d:.6}   {e:>9.2}   {d:.6}   {d:.6}   {e:>9.2}\n", .{
            layer_idx,
            k_stats.mse,        k_stats.max_err, k_stats.norm_rel_err,
            v_stats.mse,        v_stats.max_err, v_stats.norm_rel_err,
        });

        k_mse_sum += k_stats.mse;
        v_mse_sum += v_stats.mse;
        k_max_err = @max(k_max_err, k_stats.max_err);
        v_max_err = @max(v_max_err, v_stats.max_err);
        k_max_norm_rel = @max(k_max_norm_rel, k_stats.norm_rel_err);
        v_max_norm_rel = @max(v_max_norm_rel, v_stats.norm_rel_err);

        // Continue the forward pass with the *original* (un-quantized)
        // K_rot / V so subsequent layers see canonical activations.
        // Quantization-error-propagation is a separate experiment
        // (chunk 1.5).
        for (0..n_heads) |h| {
            const kv_h = h / heads_per_kv;
            const v_off = kv_h * head_dim;
            const out_off = h * head_dim;
            @memcpy(head_out[out_off .. out_off + head_dim], v[v_off .. v_off + head_dim]);
        }
        try cpu_math.matmul_nt(attn_out, head_out, layer.o_proj.?, 1, hidden, q_dim);
        for (stream, attn_out) |*si, ai| si.* += ai;

        try cpu_math.rmsnorm(mid_norm, stream, layer.post_attention_layernorm, cfg.rms_norm_eps, cfg.family);
        try cpu_math.matmul_nt(gate, mid_norm, layer.gate_proj, 1, inter, hidden);
        try cpu_math.matmul_nt(up, mid_norm, layer.up_proj, 1, inter, hidden);
        try cpu_math.geglu(fused, gate, up);
        try cpu_math.matmul_nt(ffn_out, fused, layer.down_proj, 1, hidden, inter);
        for (stream, ffn_out) |*si, fi| si.* += fi;
    }

    const n_layers: f64 = @floatFromInt(model.layers.len);
    try stdout.print("\n", .{});
    try stdout.print("aggregate (over {d} layers):\n", .{model.layers.len});
    try stdout.print("  K mean MSE     = {d:.6}    K max |Δ|  = {d:.6}    K worst norm-rel = {e:.2}\n", .{
        k_mse_sum / n_layers, k_max_err, k_max_norm_rel,
    });
    try stdout.print("  V mean MSE     = {d:.6}    V max |Δ|  = {d:.6}    V worst norm-rel = {e:.2}\n", .{
        v_mse_sum / n_layers, v_max_err, v_max_norm_rel,
    });
}

const BlockStats = struct {
    mse: f32,
    max_err: f32,
    norm_rel_err: f32,
};

fn blockStats(orig: *const [256]f32, recon: *const [256]f32) BlockStats {
    var err_sq: f32 = 0;
    var max_err: f32 = 0;
    var orig_sq: f32 = 0;
    var recon_sq: f32 = 0;
    for (orig, recon) |o, r| {
        orig_sq += o * o;
        recon_sq += r * r;
        const d = o - r;
        err_sq += d * d;
        max_err = @max(max_err, @abs(d));
    }
    const orig_norm = @sqrt(orig_sq);
    const recon_norm = @sqrt(recon_sq);
    const norm_rel = if (orig_norm > 0) @abs(orig_norm - recon_norm) / orig_norm else 0;
    return .{
        .mse = err_sq / 256.0,
        .max_err = max_err,
        .norm_rel_err = norm_rel,
    };
}

// ── gen-tq4v: side-by-side fp32 vs TQ4-V-cache CPU forward ─────────
//
// Runs both code paths on the same input token and prints the
// argmax, top-5 IDs, and the max single-logit divergence over the
// 256k-element vocab. Goal is the cleanest "does the V-cache
// quantisation introduce a token-level regression" signal: if
// argmax and top-5 hold and max |Δ| stays in fp32-noise range, the
// TQ4 V path is safe.

pub fn runGenTq4V(allocator: std.mem.Allocator, dir_path: []const u8, token_id: u32) !void {
    var model = try model_mod.Model.load(allocator, dir_path);
    defer model.deinit();
    const cfg = model.config;
    if (token_id >= cfg.vocab_size) return error.OutOfRange;

    const tok_path = try std.fmt.allocPrint(allocator, "{s}/tokenizer.json", .{dir_path});
    defer allocator.free(tok_path);
    var tok = try tokenizer_mod.Tokenizer.loadFromFile(allocator, tok_path);
    defer tok.deinit();

    const stdout = std.io.getStdOut().writer();
    try stdout.print("gen-tq4v: fp32 vs TQ4-V on token {d}\n\n", .{token_id});

    const logits_a = try allocator.alloc(f32, cfg.vocab_size);
    defer allocator.free(logits_a);
    const logits_b = try allocator.alloc(f32, cfg.vocab_size);
    defer allocator.free(logits_b);

    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();

    try cpu_forward.forward(&model, token_id, 0, arena.allocator(), logits_a);
    _ = arena.reset(.retain_capacity);
    try cpu_forward.forwardTq4V(&model, token_id, 0, arena.allocator(), logits_b);

    // Top-5 of each.
    const top_a = try topK(allocator, logits_a, 5);
    defer allocator.free(top_a);
    const top_b = try topK(allocator, logits_b, 5);
    defer allocator.free(top_b);

    try stdout.print("            fp32 baseline                       TQ4-V\n", .{});
    try stdout.print("rank   id       logit    token              id       logit    token\n", .{});
    try stdout.print("----   ------   ------   ----------------   ------   ------   ----------------\n", .{});
    for (0..5) |i| {
        const ta = top_a[i];
        const tb = top_b[i];
        const ta_text = tok.decode(ta.id) orelse "?";
        const tb_text = tok.decode(tb.id) orelse "?";
        try stdout.print("{d:>4}   {d:>6}   {d:>6.2}   {s:<16}   {d:>6}   {d:>6.2}   {s:<16}\n", .{
            i, ta.id, ta.value, truncateStr(ta_text, 16), tb.id, tb.value, truncateStr(tb_text, 16),
        });
    }

    // Pairwise stats over the full vocab.
    var max_abs_delta: f32 = 0;
    var max_abs_delta_idx: usize = 0;
    var sum_abs: f64 = 0;
    var sum_sq: f64 = 0;
    for (logits_a, logits_b, 0..) |a, b, i| {
        const d = @abs(a - b);
        if (d > max_abs_delta) {
            max_abs_delta = d;
            max_abs_delta_idx = i;
        }
        sum_abs += d;
        sum_sq += d * d;
    }
    const n: f64 = @floatFromInt(logits_a.len);
    const mean_abs = sum_abs / n;
    const rms = std.math.sqrt(sum_sq / n);

    try stdout.print("\nlogit divergence over {d} tokens:\n", .{logits_a.len});
    try stdout.print("  max |Δ|   = {d:.6}  at id={d}\n", .{ max_abs_delta, max_abs_delta_idx });
    try stdout.print("  mean |Δ|  = {d:.6}\n", .{mean_abs});
    try stdout.print("  rms  Δ    = {d:.6}\n", .{rms});

    const argmax_a = cpu_forward.argmax(logits_a);
    const argmax_b = cpu_forward.argmax(logits_b);
    try stdout.print("\nargmax:  fp32 → {d}    TQ4-V → {d}    {s}\n", .{
        argmax_a, argmax_b, if (argmax_a == argmax_b) "(MATCH)" else "(DIVERGE!)",
    });

    var top5_match: usize = 0;
    for (top_a) |a| {
        for (top_b) |b| {
            if (a.id == b.id) {
                top5_match += 1;
                break;
            }
        }
    }
    try stdout.print("top-5 ID overlap: {d}/5\n", .{top5_match});
}

fn truncateStr(s: []const u8, n: usize) []const u8 {
    return if (s.len <= n) s else s[0..n];
}

// ── gen: full forward + greedy + tokenizer decode ──────────────────

pub fn runGen(gpa: std.mem.Allocator, dir_path: []const u8, token_id: u32) !void {
    var model = try model_mod.Model.load(gpa, dir_path);
    defer model.deinit();
    const cfg = model.config;

    const tok_path = try std.fmt.allocPrint(gpa, "{s}/tokenizer.json", .{dir_path});
    defer gpa.free(tok_path);
    var tok = try tokenizer_mod.Tokenizer.loadFromFile(gpa, tok_path);
    defer tok.deinit();

    const stdout = std.io.getStdOut().writer();
    try stdout.print("loaded tokenizer: {d} ids\n", .{tok.vocabSize()});

    const input_str = tok.decode(token_id) orelse "<unknown>";
    try stdout.print("input  token id={d}  string={s}\n", .{ token_id, input_str });

    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const scratch = arena.allocator();

    const logits = try gpa.alloc(f32, cfg.vocab_size);
    defer gpa.free(logits);

    const t0 = std.time.nanoTimestamp();
    if (cfg.family.isHybrid()) {
        // Qwen3.5 hybrid: per-layer SSM + KV state. Single-token gen
        // means max_pos=1. The state is constructed fresh and torn down
        // at the end of this call.
        var state = try cpu_forward.HybridState.init(gpa, &model, 1);
        defer state.deinit();
        try cpu_forward.forwardHybrid(&model, token_id, 0, &state, scratch, logits);
    } else {
        try cpu_forward.forward(&model, token_id, 0, scratch, logits);
    }
    const t1 = std.time.nanoTimestamp();
    const ms = @as(f64, @floatFromInt(t1 - t0)) / 1_000_000.0;
    try stdout.print("forward (CPU, scalar, bf16 weights): {d:.0} ms\n", .{ms});

    // Top-K logits — useful sanity, especially when the argmax is a
    // dud token like <pad>.
    const k_top: usize = 5;
    const top = try topK(gpa, logits, k_top);
    defer gpa.free(top);

    try stdout.print("\ntop {d} logits:\n", .{k_top});
    for (top) |entry| {
        const s = tok.decode(entry.id) orelse "<unknown>";
        try stdout.print("  id={d:>6}  logit={d:>10.4}  {s}\n", .{ entry.id, entry.value, s });
    }

    const sampled = cpu_forward.argmax(logits);
    const out_str = tok.decode(sampled) orelse "<unknown>";
    try stdout.print("\nsampled (greedy): id={d}  string={s}\n", .{ sampled, out_str });
}

const TopKEntry = struct { id: usize, value: f32 };

fn topK(gpa: std.mem.Allocator, logits: []const f32, k: usize) ![]TopKEntry {
    const out = try gpa.alloc(TopKEntry, k);
    for (out) |*e| e.* = .{ .id = 0, .value = -std.math.inf(f32) };
    for (logits, 0..) |v, i| {
        if (v <= out[k - 1].value) continue;
        // Insert into sorted (descending) list.
        var j: usize = k - 1;
        out[j] = .{ .id = i, .value = v };
        while (j > 0 and out[j].value > out[j - 1].value) : (j -= 1) {
            const tmp = out[j];
            out[j] = out[j - 1];
            out[j - 1] = tmp;
        }
    }
    return out;
}

// ── gpu-layer0-test: full layer 0 forward on GPU vs CPU ────────────

const EmbedLookupPush = runtime.EmbedLookupPush;
const AddInPlacePush = runtime.AddInPlacePush;
const AttnDecodeSinglePush = extern struct { n_heads: u32, heads_per_kv: u32, head_dim: u32 };
const ScalePush = runtime_hybrid.ScalePush;
const SliceCopyPush = runtime_hybrid.SliceCopyPush;

pub fn runGpuLayer0Test(gpa: std.mem.Allocator, dir_path: []const u8, token_id: u32) !void {
    var cpu = try model_mod.Model.load(gpa, dir_path);
    defer cpu.deinit();
    const cfg = cpu.config;

    var ctx = try vk.Context.init(gpa);
    defer ctx.deinit();

    const stdout = std.io.getStdOut().writer();
    try stdout.print("GPU layer-0 forward parity test — token {d}\n", .{token_id});
    try stdout.print("device: {s}\n\n", .{ctx.deviceName()});

    try stdout.print("uploading weights...\n", .{});
    var gm = try gpu_model.GpuModel.upload(gpa, &ctx, &cpu, .fp32_all);
    defer gm.deinit(ctx.device);

    var sc = try gpu_scratch.GpuScratch.init(&ctx, cfg, 1);
    defer sc.deinit(ctx.device);

    // ── Build kernels ───────────────────────────────────────────────
    var k_embed = try pipeline.Kernel.init(&ctx, &shaders.embed_lookup, 2, @sizeOf(EmbedLookupPush));
    defer k_embed.deinit();
    var k_rmsnorm = try pipeline.Kernel.init(&ctx, &shaders.rmsnorm, 3, @sizeOf(RmsnormPush));
    defer k_rmsnorm.deinit();
    var k_matmul = try pipeline.Kernel.init(&ctx, &shaders.matmul_nt, 3, @sizeOf(MatmulPush));
    defer k_matmul.deinit();
    var k_rope = try pipeline.Kernel.init(&ctx, &shaders.rope, 2, @sizeOf(RopePush));
    defer k_rope.deinit();
    var k_attn = try pipeline.Kernel.init(&ctx, &shaders.attn_decode_single, 2, @sizeOf(AttnDecodeSinglePush));
    defer k_attn.deinit();
    var k_add = try pipeline.Kernel.init(&ctx, &shaders.add_in_place, 2, @sizeOf(AddInPlacePush));
    defer k_add.deinit();
    var k_geglu = try pipeline.Kernel.init(&ctx, &shaders.geglu, 3, @sizeOf(GegluPush));
    defer k_geglu.deinit();

    // ── Stage 0: embed → scale ──────────────────────────────────────
    try k_embed.bind(&.{ &gm.embed_tokens, &sc.stream });
    const embed_push = EmbedLookupPush{
        .token_id = token_id,
        .dim = @intCast(cfg.hidden_size),
        .scale = if (cfg.family.embedScalesByDim()) @sqrt(@as(f32, @floatFromInt(cfg.hidden_size))) else 1.0,
    };
    try dispatch1D(&ctx, &k_embed, &embed_push, @intCast(cfg.hidden_size));

    // ── Stage 1: rmsnorm₁ ───────────────────────────────────────────
    try k_rmsnorm.bind(&.{ &sc.stream, &gm.layers[0].input_layernorm, &sc.x_norm });
    const rms1_push = RmsnormPush{
        .dim = @intCast(cfg.hidden_size),
        .eps = cfg.rms_norm_eps,
        .gemma_quirk = if (cfg.family == .gemma) 1 else 0,
    };
    try dispatchPerRow(&ctx, &k_rmsnorm, &rms1_push, 1);

    // ── Stage 2: Q, K, V projections ────────────────────────────────
    const q_dim: u32 = @intCast(cfg.num_attention_heads * cfg.head_dim);
    const kv_dim: u32 = @intCast(cfg.num_key_value_heads * cfg.head_dim);
    const hidden: u32 = @intCast(cfg.hidden_size);
    try k_matmul.bind(&.{ &sc.x_norm, &gm.layers[0].q_proj.?, &sc.q });
    try dispatchMatmul(&ctx, &k_matmul, 1, q_dim, hidden);
    try k_matmul.bind(&.{ &sc.x_norm, &gm.layers[0].k_proj.?, &sc.k });
    try dispatchMatmul(&ctx, &k_matmul, 1, kv_dim, hidden);
    try k_matmul.bind(&.{ &sc.x_norm, &gm.layers[0].v_proj.?, &sc.v });
    try dispatchMatmul(&ctx, &k_matmul, 1, kv_dim, hidden);

    // ── Stage 3: RoPE on Q and K ────────────────────────────────────
    try k_rope.bind(&.{ &sc.q, &sc.q_rot });
    const rope_q_push = RopePush{
        .n_heads = @intCast(cfg.num_attention_heads),
        .head_dim = @intCast(cfg.head_dim),
        .pos = 0,
        .theta_base = cfg.rope_theta,
    };
    try dispatchRope(&ctx, &k_rope, &rope_q_push, cfg.num_attention_heads, cfg.head_dim);

    try k_rope.bind(&.{ &sc.k, &sc.k_rot });
    const rope_k_push = RopePush{
        .n_heads = @intCast(cfg.num_key_value_heads),
        .head_dim = @intCast(cfg.head_dim),
        .pos = 0,
        .theta_base = cfg.rope_theta,
    };
    try dispatchRope(&ctx, &k_rope, &rope_k_push, cfg.num_key_value_heads, cfg.head_dim);

    // ── Stage 4: attention (single-position degenerate) ─────────────
    // No KV history → softmax over 1 score = 1.0 → head_out[h] = V[kv_h(h)].
    try k_attn.bind(&.{ &sc.v, &sc.head_out });
    const attn_push = AttnDecodeSinglePush{
        .n_heads = @intCast(cfg.num_attention_heads),
        .heads_per_kv = @intCast(cfg.num_attention_heads / cfg.num_key_value_heads),
        .head_dim = @intCast(cfg.head_dim),
    };
    try dispatch1D(&ctx, &k_attn, &attn_push, q_dim);

    // ── Stage 5: o_proj ─────────────────────────────────────────────
    try k_matmul.bind(&.{ &sc.head_out, &gm.layers[0].o_proj.?, &sc.attn_out });
    try dispatchMatmul(&ctx, &k_matmul, 1, hidden, q_dim);

    // ── Stage 6: residual add (stream += attn_out) ──────────────────
    try k_add.bind(&.{ &sc.stream, &sc.attn_out });
    const add_push = AddInPlacePush{ .n = hidden };
    try dispatch1D(&ctx, &k_add, &add_push, hidden);

    // ── Stage 7: rmsnorm₂ ───────────────────────────────────────────
    try k_rmsnorm.bind(&.{ &sc.stream, &gm.layers[0].post_attention_layernorm, &sc.mid_norm });
    try dispatchPerRow(&ctx, &k_rmsnorm, &rms1_push, 1);

    // ── Stage 8: gate, up projections ───────────────────────────────
    const inter: u32 = @intCast(cfg.intermediate_size);
    try k_matmul.bind(&.{ &sc.mid_norm, &gm.layers[0].gate_proj, &sc.gate });
    try dispatchMatmul(&ctx, &k_matmul, 1, inter, hidden);
    try k_matmul.bind(&.{ &sc.mid_norm, &gm.layers[0].up_proj, &sc.up });
    try dispatchMatmul(&ctx, &k_matmul, 1, inter, hidden);

    // ── Stage 9: GeGLU ─────────────────────────────────────────────
    try k_geglu.bind(&.{ &sc.gate, &sc.up, &sc.fused });
    const geglu_push = GegluPush{ .n = inter };
    try dispatch1D(&ctx, &k_geglu, &geglu_push, inter);

    // ── Stage 10: down_proj ─────────────────────────────────────────
    try k_matmul.bind(&.{ &sc.fused, &gm.layers[0].down_proj, &sc.ffn_out });
    try dispatchMatmul(&ctx, &k_matmul, 1, hidden, inter);

    // ── Stage 11: residual add (stream += ffn_out) ──────────────────
    try k_add.bind(&.{ &sc.stream, &sc.ffn_out });
    try dispatch1D(&ctx, &k_add, &add_push, hidden);

    // ── Read back, compare against CPU layer 0 ──────────────────────
    const got = try gpa.alloc(f32, cfg.hidden_size);
    defer gpa.free(got);
    try sc.stream.readBack(&ctx, f32, got);

    const want = try cpuLayer0Forward(gpa, &cpu, token_id);
    defer gpa.free(want);

    var max_abs: f32 = 0;
    var max_rel: f32 = 0;
    var max_idx: usize = 0;
    for (got, want, 0..) |g, w, i| {
        const da = @abs(g - w);
        const dr = da / @max(@abs(w), 1e-30);
        if (da > max_abs) {
            max_abs = da;
            max_idx = i;
        }
        if (dr > max_rel) max_rel = dr;
    }

    try stdout.print("\nlayer 0 output stream: max |Δ| = {e:.3}  (at idx {d}: cpu={d:.6} gpu={d:.6})\n", .{
        max_abs, max_idx, want[max_idx], got[max_idx],
    });
    try stdout.print("max relative error = {e:.3}\n", .{max_rel});

    if (max_abs > 1e-2) {
        std.debug.print("FAIL: max |Δ| above tolerance\n", .{});
        return error.ParityFailed;
    }
    try stdout.print("\nPASS GPU layer 0 matches CPU within {e:.0}\n", .{@as(f32, 1e-2)});
}

/// Runs the layer-0-only chunk of the CPU forward pass and returns the
/// post-FFN-residual stream as fp32, for parity comparison. Mirrors
/// runLayer0Test but without the per-stage prints.
fn cpuLayer0Forward(gpa: std.mem.Allocator, cpu: *const model_mod.Model, token_id: u32) ![]f32 {
    const cfg = cpu.config;
    const layer = cpu.layers[0];
    const hidden = cfg.hidden_size;
    const inter = cfg.intermediate_size;
    const n_heads = cfg.num_attention_heads;
    const n_kv = cfg.num_key_value_heads;
    const head_dim = cfg.head_dim;
    const q_dim = n_heads * head_dim;
    const kv_dim = n_kv * head_dim;
    const heads_per_kv = n_heads / n_kv;

    const stream = try gpa.alloc(f32, hidden);
    errdefer gpa.free(stream);
    try cpu_math.embedRowAsF32(stream, cpu.embed_tokens, token_id);
    if (cfg.family.embedScalesByDim()) {
        const s: f32 = @sqrt(@as(f32, @floatFromInt(hidden)));
        for (stream) |*xi| xi.* *= s;
    }

    const x_norm = try gpa.alloc(f32, hidden);
    defer gpa.free(x_norm);
    try cpu_math.rmsnorm(x_norm, stream, layer.input_layernorm, cfg.rms_norm_eps, cfg.family);

    const v = try gpa.alloc(f32, kv_dim);
    defer gpa.free(v);
    try cpu_math.matmul_nt(v, x_norm, layer.v_proj.?, 1, kv_dim, hidden);

    const head_out = try gpa.alloc(f32, q_dim);
    defer gpa.free(head_out);
    for (0..n_heads) |h| {
        const kv_h = h / heads_per_kv;
        const v_off = kv_h * head_dim;
        const out_off = h * head_dim;
        @memcpy(head_out[out_off .. out_off + head_dim], v[v_off .. v_off + head_dim]);
    }

    const attn_out = try gpa.alloc(f32, hidden);
    defer gpa.free(attn_out);
    try cpu_math.matmul_nt(attn_out, head_out, layer.o_proj.?, 1, hidden, q_dim);
    for (stream, attn_out) |*s, a| s.* += a;

    const mid_norm = try gpa.alloc(f32, hidden);
    defer gpa.free(mid_norm);
    try cpu_math.rmsnorm(mid_norm, stream, layer.post_attention_layernorm, cfg.rms_norm_eps, cfg.family);

    const gate = try gpa.alloc(f32, inter);
    defer gpa.free(gate);
    const up = try gpa.alloc(f32, inter);
    defer gpa.free(up);
    try cpu_math.matmul_nt(gate, mid_norm, layer.gate_proj, 1, inter, hidden);
    try cpu_math.matmul_nt(up, mid_norm, layer.up_proj, 1, inter, hidden);

    const fused = try gpa.alloc(f32, inter);
    defer gpa.free(fused);
    try cpu_math.geglu(fused, gate, up);

    const ffn_out = try gpa.alloc(f32, hidden);
    defer gpa.free(ffn_out);
    try cpu_math.matmul_nt(ffn_out, fused, layer.down_proj, 1, hidden, inter);
    for (stream, ffn_out) |*s, f| s.* += f;

    return stream;
}

// ── Recorder-based dispatch helpers (single source of truth: runtime.zig) ──

const recDispatch1D = runtime.recDispatch1D;
const recDispatchPerRow = runtime.recDispatchPerRow;
const recDispatchMatmul = runtime.recDispatchMatmul;
const recDispatchRope = runtime.recDispatchRope;

// ── Dispatch helpers — keep call sites readable ──────────────────────

fn dispatch1D(
    ctx: *const vk.Context,
    kern: *const pipeline.Kernel,
    push: anytype,
    n: u32,
) !void {
    const local: u32 = 256;
    const groups: u32 = (n + local - 1) / local;
    try buffer.submitOneShot(ctx, struct {
        kern: *const pipeline.Kernel,
        push: @TypeOf(push),
        gx: u32,
        pub fn record(s: @This(), cmd: vk.c.VkCommandBuffer) void {
            s.kern.dispatch(cmd, s.push, s.gx, 1, 1);
        }
    }{ .kern = kern, .push = push, .gx = groups });
}

fn dispatchPerRow(
    ctx: *const vk.Context,
    kern: *const pipeline.Kernel,
    push: anytype,
    n_rows: u32,
) !void {
    try buffer.submitOneShot(ctx, struct {
        kern: *const pipeline.Kernel,
        push: @TypeOf(push),
        n_rows: u32,
        pub fn record(s: @This(), cmd: vk.c.VkCommandBuffer) void {
            s.kern.dispatch(cmd, s.push, s.n_rows, 1, 1);
        }
    }{ .kern = kern, .push = push, .n_rows = n_rows });
}

fn dispatchMatmul(
    ctx: *const vk.Context,
    kern: *const pipeline.Kernel,
    m: u32,
    n: u32,
    k: u32,
) !void {
    const local_xy: u32 = 16;
    const gx: u32 = (m + local_xy - 1) / local_xy;
    const gy: u32 = (n + local_xy - 1) / local_xy;
    const push = MatmulPush{ .m = m, .n = n, .k = k };
    try buffer.submitOneShot(ctx, struct {
        kern: *const pipeline.Kernel,
        push: *const MatmulPush,
        gx: u32,
        gy: u32,
        pub fn record(s: @This(), cmd: vk.c.VkCommandBuffer) void {
            s.kern.dispatch(cmd, s.push, s.gx, s.gy, 1);
        }
    }{ .kern = kern, .push = &push, .gx = gx, .gy = gy });
}

fn dispatchRope(
    ctx: *const vk.Context,
    kern: *const pipeline.Kernel,
    push: *const RopePush,
    n_heads: usize,
    head_dim: usize,
) !void {
    const local: u32 = 256;
    const pairs: u32 = @intCast(n_heads * (head_dim / 2));
    const groups: u32 = (pairs + local - 1) / local;
    try buffer.submitOneShot(ctx, struct {
        kern: *const pipeline.Kernel,
        push: *const RopePush,
        gx: u32,
        pub fn record(s: @This(), cmd: vk.c.VkCommandBuffer) void {
            s.kern.dispatch(cmd, s.push, s.gx, 1, 1);
        }
    }{ .kern = kern, .push = push, .gx = groups });
}

// ── gpu-load: upload all weights to GPU, round-trip-verify a few ────

pub fn runGpuLoad(gpa: std.mem.Allocator, dir_path: []const u8) !void {
    var cpu = try model_mod.Model.load(gpa, dir_path);
    defer cpu.deinit();
    const cfg = cpu.config;

    var ctx = try vk.Context.init(gpa);
    defer ctx.deinit();

    const stdout = std.io.getStdOut().writer();
    try stdout.print("uploading {d} layers + embed + final_norm + lm_head to {s}\n", .{
        cfg.num_hidden_layers, ctx.deviceName(),
    });

    const t0 = std.time.nanoTimestamp();
    var gm = try gpu_model.GpuModel.upload(gpa, &ctx, &cpu, .fp32_all);
    defer gm.deinit(ctx.device);
    const t1 = std.time.nanoTimestamp();
    const upload_ms = @as(f64, @floatFromInt(t1 - t0)) / 1_000_000.0;
    try stdout.print("upload time: {d:.0} ms ({d} buffers)\n\n", .{
        upload_ms,
        2 + 9 * cfg.num_hidden_layers + 1, // embed + final_norm + lm_head + 9/layer
    });

    // ── Round-trip a few tensors ─────────────────────────────────────
    // Pull representative samples back from the device and check that
    // they match the host fp32 representation.
    try roundTripCheck(gpa, &ctx, &gm.embed_tokens, cpu.embed_tokens, "embed_tokens");
    try roundTripCheck(gpa, &ctx, &gm.final_norm, cpu.final_norm, "final_norm");
    // Pick the first full-attention layer for the q_proj round-trip
    // (hybrid models put linear-attn layers first; q_proj is null
    // there). down_proj is present on every layer in every family.
    const first_full: usize = blk: {
        for (cpu.layers, 0..) |l, i| if (l.layer_type == .full_attention) break :blk i;
        break :blk 0;
    };
    if (gm.layers[first_full].q_proj) |*q| {
        try roundTripCheck(gpa, &ctx, q, cpu.layers[first_full].q_proj.?, "layer q_proj");
    }
    try roundTripCheck(gpa, &ctx, &gm.layers[0].input_layernorm, cpu.layers[0].input_layernorm, "layer 0 input_layernorm");
    const last_layer = gm.layers.len - 1;
    try roundTripCheck(gpa, &ctx, &gm.layers[last_layer].down_proj, cpu.layers[last_layer].down_proj, "last-layer down_proj");

    try stdout.print("\nPASS gpu-load (5 tensors round-tripped within fp32 ULP)\n", .{});
}

fn roundTripCheck(
    gpa: std.mem.Allocator,
    ctx: *const vk.Context,
    buf: *const buffer.Buffer,
    cpu_t: safetensors.Tensor,
    label: []const u8,
) !void {
    const stdout = std.io.getStdOut().writer();
    const numel = cpu_t.numel();

    // Materialise the CPU tensor as fp32 for the comparison.
    const want = try gpa.alloc(f32, numel);
    defer gpa.free(want);
    switch (cpu_t.dtype) {
        .f32 => @memcpy(want, @as([*]align(1) const f32, @ptrCast(cpu_t.bytes.ptr))[0..numel]),
        .bf16 => dtype.bf16SliceToF32(dtype.asU16(cpu_t.bytes), want),
        .f16 => dtype.f16SliceToF32(dtype.asU16(cpu_t.bytes), want),
        else => return error.UnsupportedDtype,
    }

    const got = try gpa.alloc(f32, numel);
    defer gpa.free(got);
    try buf.readBack(ctx, f32, got);

    var max_abs: f32 = 0;
    for (want, got) |w, g| {
        const d = @abs(w - g);
        if (d > max_abs) max_abs = d;
    }
    try stdout.print("  {s:<28}  numel={d:>10}  max |Δ| = {e:.3}\n", .{ label, numel, max_abs });
    if (max_abs > 0.0) {
        // We expect bit-exact round-trip — the only thing happening
        // here is bf16→fp32 conversion (deterministic) followed by an
        // fp32 staging upload + readback (no further conversion).
        return error.RoundTripMismatch;
    }
}

// ── gpu-rmsnorm-test: real Gemma layer 0 input_layernorm on GPU ────

pub fn runGpuRmsnormTest(gpa: std.mem.Allocator, dir_path: []const u8, token_id: u32) !void {
    var model = try model_mod.Model.load(gpa, dir_path);
    defer model.deinit();
    const cfg = model.config;

    var ctx = try vk.Context.init(gpa);
    defer ctx.deinit();

    const stdout = std.io.getStdOut().writer();
    try stdout.print("GPU rmsnorm parity test on layer 0 input_layernorm — token {d}\n", .{token_id});
    try stdout.print("device: {s}\n\n", .{ctx.deviceName()});

    // Embedding → scale → (rmsnorm)
    const x = try gpa.alloc(f32, cfg.hidden_size);
    defer gpa.free(x);
    try cpu_math.embedRowAsF32(x, model.embed_tokens, token_id);
    if (cfg.family.embedScalesByDim()) {
        const s: f32 = @sqrt(@as(f32, @floatFromInt(cfg.hidden_size)));
        for (x) |*xi| xi.* *= s;
    }

    // CPU baseline.
    const want = try gpa.alloc(f32, cfg.hidden_size);
    defer gpa.free(want);
    const t_cpu0 = std.time.nanoTimestamp();
    try cpu_math.rmsnorm(want, x, model.layers[0].input_layernorm, cfg.rms_norm_eps, cfg.family);
    const t_cpu1 = std.time.nanoTimestamp();
    const cpu_ms = @as(f64, @floatFromInt(t_cpu1 - t_cpu0)) / 1_000_000.0;

    // Materialise weight as fp32 (bf16 on disk).
    const w_bf16 = dtype.asU16(model.layers[0].input_layernorm.bytes);
    const w_f32 = try gpa.alloc(f32, w_bf16.len);
    defer gpa.free(w_f32);
    dtype.bf16SliceToF32(w_bf16, w_f32);

    // GPU dispatch.
    var buf_a = try buffer.Buffer.initStatic(&ctx, f32, x);
    defer buf_a.deinit(ctx.device);
    var buf_w = try buffer.Buffer.initStatic(&ctx, f32, w_f32);
    defer buf_w.deinit(ctx.device);
    var buf_c = try buffer.Buffer.initDeviceOnly(&ctx, cfg.hidden_size * @sizeOf(f32));
    defer buf_c.deinit(ctx.device);

    var kern = try pipeline.Kernel.init(&ctx, &shaders.rmsnorm, 3, @sizeOf(RmsnormPush));
    defer kern.deinit();
    try kern.bind(&.{ &buf_a, &buf_w, &buf_c });

    const push = RmsnormPush{
        .dim = @intCast(cfg.hidden_size),
        .eps = cfg.rms_norm_eps,
        .gemma_quirk = if (cfg.family == .gemma) 1 else 0,
    };

    const t_gpu0 = std.time.nanoTimestamp();
    try buffer.submitOneShot(&ctx, struct {
        kern: *const pipeline.Kernel,
        push: *const RmsnormPush,
        pub fn record(s: @This(), cmd: vk.c.VkCommandBuffer) void {
            s.kern.dispatch(cmd, s.push, 1, 1, 1);
        }
    }{ .kern = &kern, .push = &push });
    const t_gpu1 = std.time.nanoTimestamp();
    const gpu_ms = @as(f64, @floatFromInt(t_gpu1 - t_gpu0)) / 1_000_000.0;

    const got = try gpa.alloc(f32, cfg.hidden_size);
    defer gpa.free(got);
    try buf_c.readBack(&ctx, f32, got);

    var max_abs: f32 = 0;
    var max_idx: usize = 0;
    for (got, want, 0..) |g, e, i| {
        const d = @abs(g - e);
        if (d > max_abs) {
            max_abs = d;
            max_idx = i;
        }
    }

    try stdout.print("CPU: {d:.2} ms  GPU: {d:.2} ms (incl. submit + queue idle)\n", .{ cpu_ms, gpu_ms });
    try stdout.print("max |Δ| = {e:.3}  (at idx {d}: cpu={d:.6} gpu={d:.6})\n", .{
        max_abs, max_idx, want[max_idx], got[max_idx],
    });
    if (max_abs > 1e-3) {
        std.debug.print("FAIL: max |Δ| above tolerance\n", .{});
        return error.ParityFailed;
    }
    try stdout.print("\nPASS GPU rmsnorm matches CPU within {e:.0}\n", .{@as(f32, 1e-3)});
}

// ── gpu-rope-test: real Gemma Q at pos=1 vs CPU ────────────────────

pub fn runGpuRopeTest(gpa: std.mem.Allocator, dir_path: []const u8, token_id: u32) !void {
    var model = try model_mod.Model.load(gpa, dir_path);
    defer model.deinit();
    const cfg = model.config;
    const layer = model.layers[0];

    var ctx = try vk.Context.init(gpa);
    defer ctx.deinit();

    const stdout = std.io.getStdOut().writer();
    try stdout.print("GPU RoPE parity test on layer 0 Q at pos=1 — token {d}\n", .{token_id});
    try stdout.print("device: {s}\n\n", .{ctx.deviceName()});

    // Reproduce Q via the verified CPU pipeline.
    const x = try gpa.alloc(f32, cfg.hidden_size);
    defer gpa.free(x);
    try cpu_math.embedRowAsF32(x, model.embed_tokens, token_id);
    if (cfg.family.embedScalesByDim()) {
        const s: f32 = @sqrt(@as(f32, @floatFromInt(cfg.hidden_size)));
        for (x) |*xi| xi.* *= s;
    }
    const x_norm = try gpa.alloc(f32, cfg.hidden_size);
    defer gpa.free(x_norm);
    try cpu_math.rmsnorm(x_norm, x, layer.input_layernorm, cfg.rms_norm_eps, cfg.family);

    const q_dim = cfg.num_attention_heads * cfg.head_dim;
    const q = try gpa.alloc(f32, q_dim);
    defer gpa.free(q);
    try cpu_math.matmul_nt(q, x_norm, layer.q_proj.?, 1, q_dim, cfg.hidden_size);

    // CPU baseline at pos=1.
    const want = try gpa.alloc(f32, q_dim);
    defer gpa.free(want);
    try cpu_math.applyRope(want, q, cfg.num_attention_heads, cfg.head_dim, 1, cfg.rope_theta);

    // GPU dispatch.
    var buf_in = try buffer.Buffer.initStatic(&ctx, f32, q);
    defer buf_in.deinit(ctx.device);
    var buf_out = try buffer.Buffer.initDeviceOnly(&ctx, q_dim * @sizeOf(f32));
    defer buf_out.deinit(ctx.device);

    var kern = try pipeline.Kernel.init(&ctx, &shaders.rope, 2, @sizeOf(RopePush));
    defer kern.deinit();
    try kern.bind(&.{ &buf_in, &buf_out });

    const local: u32 = 256;
    const pairs: u32 = @intCast(cfg.num_attention_heads * (cfg.head_dim / 2));
    const groups: u32 = (pairs + local - 1) / local;
    const push = RopePush{
        .n_heads = @intCast(cfg.num_attention_heads),
        .head_dim = @intCast(cfg.head_dim),
        .pos = 1,
        .theta_base = cfg.rope_theta,
    };

    const t_gpu0 = std.time.nanoTimestamp();
    try buffer.submitOneShot(&ctx, struct {
        kern: *const pipeline.Kernel,
        push: *const RopePush,
        groups: u32,
        pub fn record(s: @This(), cmd: vk.c.VkCommandBuffer) void {
            s.kern.dispatch(cmd, s.push, s.groups, 1, 1);
        }
    }{ .kern = &kern, .push = &push, .groups = groups });
    const t_gpu1 = std.time.nanoTimestamp();
    const gpu_ms = @as(f64, @floatFromInt(t_gpu1 - t_gpu0)) / 1_000_000.0;

    const got = try gpa.alloc(f32, q_dim);
    defer gpa.free(got);
    try buf_out.readBack(&ctx, f32, got);

    var max_abs: f32 = 0;
    var max_idx: usize = 0;
    for (got, want, 0..) |g, e, i| {
        const d = @abs(g - e);
        if (d > max_abs) {
            max_abs = d;
            max_idx = i;
        }
    }

    try stdout.print("GPU: {d:.2} ms (incl. submit + queue idle)\n", .{gpu_ms});
    try stdout.print("max |Δ| = {e:.3}  (at idx {d}: cpu={d:.6} gpu={d:.6})\n", .{
        max_abs, max_idx, want[max_idx], got[max_idx],
    });
    if (max_abs > 1e-3) {
        std.debug.print("FAIL: max |Δ| above tolerance\n", .{});
        return error.ParityFailed;
    }
    try stdout.print("\nPASS GPU rope matches CPU within {e:.0}\n", .{@as(f32, 1e-3)});
}

// ── gpu-geglu-test: real Gemma layer 0 GeGLU vs CPU ────────────────

pub fn runGpuGegluTest(gpa: std.mem.Allocator, dir_path: []const u8, token_id: u32) !void {
    var model = try model_mod.Model.load(gpa, dir_path);
    defer model.deinit();
    const cfg = model.config;
    const layer = model.layers[0];

    var ctx = try vk.Context.init(gpa);
    defer ctx.deinit();

    const stdout = std.io.getStdOut().writer();
    try stdout.print("GPU GeGLU parity test on layer 0 — token {d}\n", .{token_id});
    try stdout.print("device: {s}\n\n", .{ctx.deviceName()});

    // Reproduce the FFN inputs on CPU: embed → scale → rmsnorm₁ → attn
    // (single-position degenerate to V) → o_proj → residual → rmsnorm₂.
    const inter = cfg.intermediate_size;

    const x = try gpa.alloc(f32, cfg.hidden_size);
    defer gpa.free(x);
    try cpu_math.embedRowAsF32(x, model.embed_tokens, token_id);
    if (cfg.family.embedScalesByDim()) {
        const s: f32 = @sqrt(@as(f32, @floatFromInt(cfg.hidden_size)));
        for (x) |*xi| xi.* *= s;
    }
    const x_norm = try gpa.alloc(f32, cfg.hidden_size);
    defer gpa.free(x_norm);
    try cpu_math.rmsnorm(x_norm, x, layer.input_layernorm, cfg.rms_norm_eps, cfg.family);

    // Single-position attention output = V projected through o_proj.
    const v = try gpa.alloc(f32, cfg.num_key_value_heads * cfg.head_dim);
    defer gpa.free(v);
    try cpu_math.matmul_nt(v, x_norm, layer.v_proj.?, 1, v.len, cfg.hidden_size);

    const head_out = try gpa.alloc(f32, cfg.num_attention_heads * cfg.head_dim);
    defer gpa.free(head_out);
    const heads_per_kv = cfg.num_attention_heads / cfg.num_key_value_heads;
    for (0..cfg.num_attention_heads) |h| {
        const kv_h = h / heads_per_kv;
        const v_off = kv_h * cfg.head_dim;
        const out_off = h * cfg.head_dim;
        @memcpy(head_out[out_off .. out_off + cfg.head_dim], v[v_off .. v_off + cfg.head_dim]);
    }
    const attn_out = try gpa.alloc(f32, cfg.hidden_size);
    defer gpa.free(attn_out);
    try cpu_math.matmul_nt(attn_out, head_out, layer.o_proj.?, 1, cfg.hidden_size, head_out.len);

    const mid = try gpa.alloc(f32, cfg.hidden_size);
    defer gpa.free(mid);
    for (mid, x, attn_out) |*m, xi, ai| m.* = xi + ai;
    const mid_norm = try gpa.alloc(f32, cfg.hidden_size);
    defer gpa.free(mid_norm);
    try cpu_math.rmsnorm(mid_norm, mid, layer.post_attention_layernorm, cfg.rms_norm_eps, cfg.family);

    const gate = try gpa.alloc(f32, inter);
    defer gpa.free(gate);
    const upv = try gpa.alloc(f32, inter);
    defer gpa.free(upv);
    try cpu_math.matmul_nt(gate, mid_norm, layer.gate_proj, 1, inter, cfg.hidden_size);
    try cpu_math.matmul_nt(upv, mid_norm, layer.up_proj, 1, inter, cfg.hidden_size);

    const want = try gpa.alloc(f32, inter);
    defer gpa.free(want);
    const t_cpu0 = std.time.nanoTimestamp();
    try cpu_math.geglu(want, gate, upv);
    const t_cpu1 = std.time.nanoTimestamp();
    const cpu_ms = @as(f64, @floatFromInt(t_cpu1 - t_cpu0)) / 1_000_000.0;

    var buf_g = try buffer.Buffer.initStatic(&ctx, f32, gate);
    defer buf_g.deinit(ctx.device);
    var buf_u = try buffer.Buffer.initStatic(&ctx, f32, upv);
    defer buf_u.deinit(ctx.device);
    var buf_o = try buffer.Buffer.initDeviceOnly(&ctx, inter * @sizeOf(f32));
    defer buf_o.deinit(ctx.device);

    var kern = try pipeline.Kernel.init(&ctx, &shaders.geglu, 3, @sizeOf(GegluPush));
    defer kern.deinit();
    try kern.bind(&.{ &buf_g, &buf_u, &buf_o });

    const local: u32 = 256;
    const n_u32: u32 = @intCast(inter);
    const groups: u32 = (n_u32 + local - 1) / local;
    const push = GegluPush{ .n = n_u32 };

    const t_gpu0 = std.time.nanoTimestamp();
    try buffer.submitOneShot(&ctx, struct {
        kern: *const pipeline.Kernel,
        push: *const GegluPush,
        groups: u32,
        pub fn record(s: @This(), cmd: vk.c.VkCommandBuffer) void {
            s.kern.dispatch(cmd, s.push, s.groups, 1, 1);
        }
    }{ .kern = &kern, .push = &push, .groups = groups });
    const t_gpu1 = std.time.nanoTimestamp();
    const gpu_ms = @as(f64, @floatFromInt(t_gpu1 - t_gpu0)) / 1_000_000.0;

    const got = try gpa.alloc(f32, inter);
    defer gpa.free(got);
    try buf_o.readBack(&ctx, f32, got);

    var max_abs: f32 = 0;
    var max_idx: usize = 0;
    for (got, want, 0..) |g, e, i| {
        const d = @abs(g - e);
        if (d > max_abs) {
            max_abs = d;
            max_idx = i;
        }
    }

    try stdout.print("CPU: {d:.2} ms  GPU: {d:.2} ms (incl. submit + queue idle)\n", .{ cpu_ms, gpu_ms });
    try stdout.print("max |Δ| = {e:.3}  (at idx {d}: cpu={d:.6} gpu={d:.6})\n", .{
        max_abs, max_idx, want[max_idx], got[max_idx],
    });
    if (max_abs > 1e-3) {
        std.debug.print("FAIL: max |Δ| above tolerance\n", .{});
        return error.ParityFailed;
    }
    try stdout.print("\nPASS GPU geglu matches CPU within {e:.0}\n", .{@as(f32, 1e-3)});
}

// ── gpu-qproj-test: real Gemma q_proj on GPU vs CPU ────────────────

pub fn runGpuQprojTest(gpa: std.mem.Allocator, dir_path: []const u8, token_id: u32) !void {
    var model = try model_mod.Model.load(gpa, dir_path);
    defer model.deinit();
    const cfg = model.config;

    var ctx = try vk.Context.init(gpa);
    defer ctx.deinit();

    const stdout = std.io.getStdOut().writer();
    try stdout.print("GPU matmul_nt parity test on layer 0 q_proj — token {d}\n", .{token_id});
    try stdout.print("device: {s}\n\n", .{ctx.deviceName()});

    // ── Reproduce the qproj-test inputs on the host ─────────────────
    const x = try gpa.alloc(f32, cfg.hidden_size);
    defer gpa.free(x);
    try cpu_math.embedRowAsF32(x, model.embed_tokens, token_id);
    if (cfg.family.embedScalesByDim()) {
        const s: f32 = @sqrt(@as(f32, @floatFromInt(cfg.hidden_size)));
        for (x) |*xi| xi.* *= s;
    }
    const x_norm = try gpa.alloc(f32, cfg.hidden_size);
    defer gpa.free(x_norm);
    try cpu_math.rmsnorm(x_norm, x, model.layers[0].input_layernorm, cfg.rms_norm_eps, cfg.family);

    const q_dim = cfg.num_attention_heads * cfg.head_dim;

    // ── CPU baseline (existing path) ────────────────────────────────
    const q_cpu = try gpa.alloc(f32, q_dim);
    defer gpa.free(q_cpu);
    const t_cpu0 = std.time.nanoTimestamp();
    try cpu_math.matmul_nt(q_cpu, x_norm, model.layers[0].q_proj.?, 1, q_dim, cfg.hidden_size);
    const t_cpu1 = std.time.nanoTimestamp();
    const cpu_ms = @as(f64, @floatFromInt(t_cpu1 - t_cpu0)) / 1_000_000.0;

    // ── Materialise q_proj as fp32 for GPU upload ───────────────────
    // The GPU kernel is fp32-only for now; we'll add a bf16-aware
    // variant once the fp32 path is parity-clean. The conversion is
    // O(numel) so it doesn't dominate setup, but it does double host
    // memory while we hold both copies — fine for this kernel
    // (32 MiB), would want an in-place stream once we do all weights.
    const w_bf16 = dtype.asU16(model.layers[0].q_proj.?.bytes);
    const w_f32 = try gpa.alloc(f32, w_bf16.len);
    defer gpa.free(w_f32);
    dtype.bf16SliceToF32(w_bf16, w_f32);

    // ── GPU upload + dispatch ───────────────────────────────────────
    var buf_a = try buffer.Buffer.initStatic(&ctx, f32, x_norm);
    defer buf_a.deinit(ctx.device);
    var buf_b = try buffer.Buffer.initStatic(&ctx, f32, w_f32);
    defer buf_b.deinit(ctx.device);
    var buf_c = try buffer.Buffer.initDeviceOnly(&ctx, q_dim * @sizeOf(f32));
    defer buf_c.deinit(ctx.device);

    var kern = try pipeline.Kernel.init(&ctx, &shaders.matmul_nt, 3, @sizeOf(MatmulPush));
    defer kern.deinit();
    try kern.bind(&.{ &buf_a, &buf_b, &buf_c });

    const local_xy: u32 = 16;
    const m: u32 = 1;
    const n: u32 = @intCast(q_dim);
    const k: u32 = @intCast(cfg.hidden_size);
    const groups_x: u32 = (m + local_xy - 1) / local_xy;
    const groups_y: u32 = (n + local_xy - 1) / local_xy;
    const push = MatmulPush{ .m = m, .n = n, .k = k };

    const t_gpu0 = std.time.nanoTimestamp();
    try buffer.submitOneShot(&ctx, struct {
        kern: *const pipeline.Kernel,
        push: *const MatmulPush,
        gx: u32,
        gy: u32,
        pub fn record(s: @This(), cmd: vk.c.VkCommandBuffer) void {
            s.kern.dispatch(cmd, s.push, s.gx, s.gy, 1);
        }
    }{ .kern = &kern, .push = &push, .gx = groups_x, .gy = groups_y });
    const t_gpu1 = std.time.nanoTimestamp();
    const gpu_ms = @as(f64, @floatFromInt(t_gpu1 - t_gpu0)) / 1_000_000.0;

    const q_gpu = try gpa.alloc(f32, q_dim);
    defer gpa.free(q_gpu);
    try buf_c.readBack(&ctx, f32, q_gpu);

    // ── Parity ──────────────────────────────────────────────────────
    var max_abs_err: f32 = 0;
    var max_rel_err: f32 = 0;
    var max_idx: usize = 0;
    for (q_cpu, q_gpu, 0..) |c, g, i| {
        const abs_err = @abs(c - g);
        const denom = @max(@abs(c), 1e-30);
        const rel_err = abs_err / denom;
        if (abs_err > max_abs_err) {
            max_abs_err = abs_err;
            max_idx = i;
        }
        if (rel_err > max_rel_err) max_rel_err = rel_err;
    }

    try stdout.print("CPU: {d:.2} ms  GPU: {d:.2} ms (incl. submit + queue idle)\n", .{ cpu_ms, gpu_ms });
    try stdout.print("max |Δ| = {e:.3}  (at idx {d}: cpu={d:.6} gpu={d:.6})\n", .{
        max_abs_err, max_idx, q_cpu[max_idx], q_gpu[max_idx],
    });
    try stdout.print("max relative error = {e:.3}\n", .{max_rel_err});

    if (max_abs_err > 1e-3) {
        std.debug.print("FAIL: max |Δ| above tolerance\n", .{});
        return error.ParityFailed;
    }
    try stdout.print("\nPASS GPU q_proj matches CPU within {e:.0}\n", .{@as(f32, 1e-3)});
}

fn printStreamStats(w: anytype, label: []const u8, x: []const f32) !void {
    var min_v: f32 = std.math.inf(f32);
    var max_v: f32 = -std.math.inf(f32);
    var sum_sq: f64 = 0;
    var nan_count: usize = 0;
    var inf_count: usize = 0;
    for (x) |v| {
        if (std.math.isNan(v)) {
            nan_count += 1;
            continue;
        }
        if (std.math.isInf(v)) {
            inf_count += 1;
            continue;
        }
        if (v < min_v) min_v = v;
        if (v > max_v) max_v = v;
        sum_sq += @as(f64, v) * @as(f64, v);
    }
    const rms = std.math.sqrt(sum_sq / @as(f64, @floatFromInt(x.len)));
    try w.print("  {s:<32} min={d:>10.4}  max={d:>10.4}  rms={d:>10.4}", .{ label, min_v, max_v, rms });
    if (nan_count > 0 or inf_count > 0) {
        try w.print("  nan={d} inf={d}", .{ nan_count, inf_count });
    }
    try w.print("\n", .{});
}
