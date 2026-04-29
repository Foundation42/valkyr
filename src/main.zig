//! Entry point. Runs every smoke test we have in sequence so a fresh
//! `zig build run` exercises the whole stack — Vulkan compute, file
//! formats, and (eventually) full-model parity. Each individual test
//! lives in its own function and prints a one-line pass marker on
//! success or surfaces an error otherwise.

const std = @import("std");
const vk = @import("gpu/vk.zig");
const buffer = @import("gpu/buffer.zig");
const pipeline = @import("gpu/pipeline.zig");
const safetensors = @import("safetensors.zig");
const model_mod = @import("model.zig");
const dtype = @import("dtype.zig");
const cpu_math = @import("cpu/math.zig");
const shaders = @import("shaders");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    const args = try std.process.argsAlloc(allocator);
    defer std.process.argsFree(allocator, args);

    if (args.len >= 3 and std.mem.eql(u8, args[1], "--inspect")) {
        try runInspect(allocator, args[2]);
        return;
    }
    if (args.len >= 3 and std.mem.eql(u8, args[1], "--load")) {
        try runLoad(allocator, args[2]);
        return;
    }
    if (args.len >= 4 and std.mem.eql(u8, args[1], "--dump-embed")) {
        const token_id = try std.fmt.parseInt(u32, args[3], 10);
        try runDumpEmbed(allocator, args[2], token_id);
        return;
    }
    if (args.len >= 4 and std.mem.eql(u8, args[1], "--rmsnorm-test")) {
        const token_id = try std.fmt.parseInt(u32, args[3], 10);
        try runRmsnormTest(allocator, args[2], token_id);
        return;
    }
    if (args.len >= 4 and std.mem.eql(u8, args[1], "--qproj-test")) {
        const token_id = try std.fmt.parseInt(u32, args[3], 10);
        try runQprojTest(allocator, args[2], token_id);
        return;
    }
    if (args.len >= 4 and std.mem.eql(u8, args[1], "--rope-test")) {
        const token_id = try std.fmt.parseInt(u32, args[3], 10);
        try runRopeTest(allocator, args[2], token_id);
        return;
    }
    if (args.len >= 4 and std.mem.eql(u8, args[1], "--attention-test")) {
        const token_id = try std.fmt.parseInt(u32, args[3], 10);
        try runAttentionTest(allocator, args[2], token_id);
        return;
    }

    try runVecAddSmoke(allocator);
    try runSafeTensorsSmoke(allocator);
    try runMatmulSmoke(allocator);
    try runRopeIdentitySmoke(allocator);
    try runSoftmaxSmoke(allocator);
}

// ── inspect: dump the tensor inventory of a real .safetensors file ──

fn runInspect(allocator: std.mem.Allocator, path: []const u8) !void {
    const t0 = std.time.nanoTimestamp();
    var st = try safetensors.SafeTensors.open(allocator, path);
    defer st.deinit();
    const t1 = std.time.nanoTimestamp();
    const open_ms = @as(f64, @floatFromInt(t1 - t0)) / 1_000_000.0;

    const stdout = std.io.getStdOut().writer();
    try stdout.print("File: {s}\n", .{path});
    try stdout.print("Tensors: {d}    Open+parse: {d:.2} ms\n", .{ st.count(), open_ms });
    try stdout.print("Mapping: {d:.2} GiB\n", .{
        @as(f64, @floatFromInt(st.mapping.len)) / (1024.0 * 1024.0 * 1024.0),
    });
    try stdout.print("\n", .{});

    // Walk in alphabetical order so the output is reproducible across
    // hashmap iteration orders.
    var names = std.ArrayList([]const u8).init(allocator);
    defer names.deinit();
    var it = st.by_name.keyIterator();
    while (it.next()) |k| try names.append(k.*);
    std.mem.sort([]const u8, names.items, {}, struct {
        fn lessThan(_: void, a: []const u8, b: []const u8) bool {
            return std.mem.order(u8, a, b) == .lt;
        }
    }.lessThan);

    var totals = [_]u64{0} ** @typeInfo(safetensors.Dtype).@"enum".fields.len;
    for (names.items) |name| {
        const t = st.get(name).?;
        totals[@intFromEnum(t.dtype)] += t.bytes.len;
        try stdout.print("  {s:<10} {s:<55} ", .{ @tagName(t.dtype), name });
        try stdout.print("[", .{});
        for (t.shape, 0..) |d, i| {
            if (i > 0) try stdout.print(", ", .{});
            try stdout.print("{d}", .{d});
        }
        try stdout.print("]  {d:.2} MiB\n", .{
            @as(f64, @floatFromInt(t.bytes.len)) / (1024.0 * 1024.0),
        });
    }

    try stdout.print("\nDtype totals:\n", .{});
    inline for (@typeInfo(safetensors.Dtype).@"enum".fields) |f| {
        const idx: usize = f.value;
        if (totals[idx] > 0) {
            try stdout.print("  {s:<6} {d:.2} MiB\n", .{
                f.name,
                @as(f64, @floatFromInt(totals[idx])) / (1024.0 * 1024.0),
            });
        }
    }
}

// ── load: parse config + open shards + bind layer weights ────────────

fn runLoad(allocator: std.mem.Allocator, dir_path: []const u8) !void {
    const t0 = std.time.nanoTimestamp();
    var model = try model_mod.Model.load(allocator, dir_path);
    defer model.deinit();
    const t1 = std.time.nanoTimestamp();
    const load_ms = @as(f64, @floatFromInt(t1 - t0)) / 1_000_000.0;

    const stdout = std.io.getStdOut().writer();
    try stdout.print("Loaded {s}\n", .{dir_path});
    try stdout.print("Load time: {d:.2} ms ({d} shard(s), {d} tensors)\n\n", .{
        load_ms, model.shards.shards.len, model.shards.count(),
    });
    try model.config.print(stdout);
    try stdout.print("\nLM head: {s}\n", .{
        if (model.isLmHeadTied()) "tied (shares embed_tokens)" else "separate (lm_head.weight)",
    });

    // Sanity: walk every layer once and confirm we can read the first
    // and last byte of each weight from the mmap. If a shard's offset
    // table is wrong, that page-faults; if it's right, this is free.
    var bytes_touched: usize = 0;
    for (model.layers) |layer| {
        inline for (.{
            "input_layernorm",      "q_proj",     "k_proj",
            "v_proj",               "o_proj",     "post_attention_layernorm",
            "gate_proj",            "up_proj",    "down_proj",
        }) |fname| {
            const t = @field(layer, fname);
            if (t.bytes.len > 0) {
                bytes_touched +%= @intCast(t.bytes[0]);
                bytes_touched +%= @intCast(t.bytes[t.bytes.len - 1]);
            }
        }
    }
    bytes_touched +%= @intCast(model.embed_tokens.bytes[0]);
    bytes_touched +%= @intCast(model.final_norm.bytes[0]);
    bytes_touched +%= @intCast(model.lm_head.bytes[0]);
    // The xor folds the first/last byte of every weight into one word —
    // a cheap way to force the OS to actually touch each tensor page.
    // Print it so the optimizer can't elide the loop.
    try stdout.print("\nPASS load (touch checksum: 0x{x})\n", .{bytes_touched});
}

// ── dump-embed: pull a token's row from embed_tokens, bf16 → fp32 ────

fn runDumpEmbed(allocator: std.mem.Allocator, dir_path: []const u8, token_id: u32) !void {
    var model = try model_mod.Model.load(allocator, dir_path);
    defer model.deinit();

    const cfg = model.config;
    if (token_id >= cfg.vocab_size) {
        std.debug.print("token id {d} out of range (vocab_size={d})\n", .{ token_id, cfg.vocab_size });
        return error.OutOfRange;
    }

    if (model.embed_tokens.dtype != .bf16) {
        std.debug.print("expected bf16 embed_tokens; got {s}\n", .{@tagName(model.embed_tokens.dtype)});
        return error.UnexpectedDtype;
    }

    const stdout = std.io.getStdOut().writer();
    try stdout.print("token {d}, hidden_size={d} — first values + stats\n", .{ token_id, cfg.hidden_size });

    // Slice the row for this token. embed_tokens is [vocab_size, hidden_size]
    // row-major, so row `token_id` starts at `token_id * hidden_size *
    // bytes_per_elem` from the tensor's byte base.
    const elem_bytes = model.embed_tokens.dtype.elemSize();
    const row_bytes = cfg.hidden_size * elem_bytes;
    const row_start = @as(usize, token_id) * row_bytes;
    const row_bf16 = dtype.asU16(model.embed_tokens.bytes[row_start .. row_start + row_bytes]);

    // Convert into a heap-allocated fp32 buffer.
    const row_f32 = try allocator.alloc(f32, cfg.hidden_size);
    defer allocator.free(row_f32);
    dtype.bf16SliceToF32(row_bf16, row_f32);

    // First N for inspection.
    const n_show: usize = @min(16, row_f32.len);
    for (row_f32[0..n_show], 0..) |v, i| try stdout.print("  [{d:>4}] {d:.6}\n", .{ i, v });

    // Stats over the whole row.
    var min_v: f32 = std.math.inf(f32);
    var max_v: f32 = -std.math.inf(f32);
    var sum: f64 = 0;
    var nan_count: usize = 0;
    var inf_count: usize = 0;
    for (row_f32) |v| {
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
        sum += v;
    }
    const mean = sum / @as(f64, @floatFromInt(row_f32.len - nan_count - inf_count));
    try stdout.print("\nstats: min={d:.6} max={d:.6} mean={d:.6} nan={d} inf={d}\n", .{
        min_v, max_v, mean, nan_count, inf_count,
    });
}

// ── matmul smoke: synthetic A·B^T, hand-checked oracle ──────────────

fn runMatmulSmoke(allocator: std.mem.Allocator) !void {
    _ = allocator;
    // A = [[1, 2, 3], [4, 5, 6]]                M=2, K=3
    // B = [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1]]   N=4, K=3
    // A · Bᵀ = [[1, 2, 3, 6], [4, 5, 6, 15]]    M=2, N=4
    const a = [_]f32{ 1, 2, 3, 4, 5, 6 };
    const b_data = [_]f32{ 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1 };
    const want = [_]f32{ 1, 2, 3, 6, 4, 5, 6, 15 };

    const b_shape = [_]usize{ 4, 3 };
    const b: safetensors.Tensor = .{
        .dtype = .f32,
        .shape = &b_shape,
        .bytes = std.mem.sliceAsBytes(b_data[0..]),
    };

    var out: [8]f32 = undefined;
    try cpu_math.matmul_nt(&out, &a, b, 2, 4, 3);

    for (out, want, 0..) |got, w, i| {
        if (got != w) {
            std.debug.print("matmul MISMATCH at {d}: got {d}, expected {d}\n", .{ i, got, w });
            return error.ParityFailed;
        }
    }
    std.debug.print("PASS matmul_nt synthetic (2×3 · (4×3)ᵀ → 2×4)\n", .{});
}

// ── RoPE identity smoke: pos=0 must be a no-op ──────────────────────

fn runRopeIdentitySmoke(allocator: std.mem.Allocator) !void {
    // Synthetic Q-shaped input. n_heads=8, head_dim=64 — stays small.
    const n_heads: usize = 8;
    const head_dim: usize = 64;
    const total = n_heads * head_dim;
    const in_v = try allocator.alloc(f32, total);
    defer allocator.free(in_v);
    const out_v = try allocator.alloc(f32, total);
    defer allocator.free(out_v);
    for (in_v, 0..) |*x, i| x.* = @as(f32, @floatFromInt(i)) * 0.001 - 0.5;

    try cpu_math.applyRope(out_v, in_v, n_heads, head_dim, 0, 10000.0);

    for (in_v, out_v, 0..) |a, b, i| {
        if (a != b) {
            std.debug.print("RoPE pos=0 NOT identity at i={d}: in={d} out={d}\n", .{ i, a, b });
            return error.ParityFailed;
        }
    }
    std.debug.print("PASS RoPE pos=0 identity ({d} heads × {d} dim)\n", .{ n_heads, head_dim });
}

// ── softmax smoke: stable form on a hand-checked input ─────────────

fn runSoftmaxSmoke(allocator: std.mem.Allocator) !void {
    _ = allocator;
    // Inputs include a large positive value; the naive exp(x) form
    // would overflow but the stable variant should produce normal
    // probabilities. Reference computed with the same shifted form.
    var x = [_]f32{ 1.0, 2.0, 3.0, 100.0 };
    cpu_math.softmax(&x);
    var sum: f32 = 0;
    for (x) |v| sum += v;
    if (@abs(sum - 1.0) > 1e-6) {
        std.debug.print("softmax sum {d} != 1\n", .{sum});
        return error.ParityFailed;
    }
    // The big value should dominate (≈ 1.0); the others should be tiny.
    if (x[3] < 0.99 or x[0] > 1e-30 or x[1] > 1e-30 or x[2] > 1e-30) {
        std.debug.print("softmax distribution wrong: {any}\n", .{x});
        return error.ParityFailed;
    }
    std.debug.print("PASS softmax stable form (handles +100 without overflow)\n", .{});
}

// ── rmsnorm-test: first math primitive on a real layer ──────────────

fn runRmsnormTest(allocator: std.mem.Allocator, dir_path: []const u8, token_id: u32) !void {
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

fn runQprojTest(allocator: std.mem.Allocator, dir_path: []const u8, token_id: u32) !void {
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
    try cpu_math.matmul_nt(q, x_norm, model.layers[0].q_proj, 1, q_dim, cfg.hidden_size);
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

fn runRopeTest(allocator: std.mem.Allocator, dir_path: []const u8, token_id: u32) !void {
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
    try cpu_math.matmul_nt(q, x_norm, model.layers[0].q_proj, 1, q_dim, cfg.hidden_size);

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

fn runAttentionTest(allocator: std.mem.Allocator, dir_path: []const u8, token_id: u32) !void {
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
        try cpu_math.matmul_nt(q, x_norm, layer.q_proj, 1, q_dim, cfg.hidden_size);
        try cpu_math.matmul_nt(k, x_norm, layer.k_proj, 1, kv_dim, cfg.hidden_size);
        try cpu_math.matmul_nt(v, x_norm, layer.v_proj, 1, kv_dim, cfg.hidden_size);

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
        try cpu_math.matmul_nt(attn_out, head_out, layer.o_proj, 1, cfg.hidden_size, q_dim);

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

// ── vec_add smoke: validates the whole Vulkan compute path ───────────

const N: u32 = 1024 * 1024;
const VecAddPush = extern struct { n: u32 };

fn runVecAddSmoke(allocator: std.mem.Allocator) !void {
    var ctx = try vk.Context.init(allocator);
    defer ctx.deinit();

    const a = try allocator.alloc(f32, N);
    defer allocator.free(a);
    const b = try allocator.alloc(f32, N);
    defer allocator.free(b);
    const out = try allocator.alloc(f32, N);
    defer allocator.free(out);
    for (a, b, 0..) |*ai, *bi, i| {
        ai.* = @floatFromInt(i);
        bi.* = @as(f32, @floatFromInt(i)) * 2.0;
    }

    var buf_a = try buffer.Buffer.initStatic(&ctx, f32, a);
    defer buf_a.deinit(ctx.device);
    var buf_b = try buffer.Buffer.initStatic(&ctx, f32, b);
    defer buf_b.deinit(ctx.device);
    var buf_c = try buffer.Buffer.initDeviceOnly(&ctx, N * @sizeOf(f32));
    defer buf_c.deinit(ctx.device);

    var vec_add = try pipeline.Kernel.init(&ctx, &shaders.vec_add, 3, @sizeOf(VecAddPush));
    defer vec_add.deinit();
    try vec_add.bind(&.{ &buf_a, &buf_b, &buf_c });

    const local_size: u32 = 256;
    const groups: u32 = (N + local_size - 1) / local_size;
    const push = VecAddPush{ .n = N };

    try buffer.submitOneShot(&ctx, struct {
        kern: *const pipeline.Kernel,
        push: *const VecAddPush,
        groups: u32,
        pub fn record(s: @This(), cmd: vk.c.VkCommandBuffer) void {
            s.kern.dispatch(cmd, s.push, s.groups, 1, 1);
        }
    }{ .kern = &vec_add, .push = &push, .groups = groups });

    try buf_c.readBack(&ctx, f32, out);
    for (out, 0..) |v, i| {
        const expected = @as(f32, @floatFromInt(i)) * 3.0;
        if (v != expected) {
            std.debug.print("vec_add MISMATCH at {d}: got {d}, expected {d}\n", .{ i, v, expected });
            return error.ParityFailed;
        }
    }
    std.debug.print("PASS vec_add ({d} elems) on {s}\n", .{ N, ctx.deviceName() });
}

// ── safetensors smoke: synthesizes a file, parses it, checks round-trip ──

fn runSafeTensorsSmoke(allocator: std.mem.Allocator) !void {
    // Build a two-tensor file in memory:
    //   weight_a: F32, shape [3, 2], values 0..5
    //   weight_b: I32, shape [4],    values [10, 20, 30, 40]
    const w_a: [6]f32 = .{ 0.0, 1.0, 2.0, 3.0, 4.0, 5.0 };
    const w_b: [4]i32 = .{ 10, 20, 30, 40 };

    // Header JSON. Keys sorted; offsets are relative to the end of the
    // header itself. Total tensor bytes: 24 (a) + 16 (b) = 40.
    const header_json =
        \\{"weight_a":{"dtype":"F32","shape":[3,2],"data_offsets":[0,24]},"weight_b":{"dtype":"I32","shape":[4],"data_offsets":[24,40]}}
    ;

    // Compose: [u64 LE header_len][header_json][tensor data].
    var blob = std.ArrayList(u8).init(allocator);
    defer blob.deinit();
    var len_buf: [8]u8 = undefined;
    std.mem.writeInt(u64, &len_buf, header_json.len, .little);
    try blob.appendSlice(&len_buf);
    try blob.appendSlice(header_json);
    try blob.appendSlice(std.mem.sliceAsBytes(&w_a));
    try blob.appendSlice(std.mem.sliceAsBytes(&w_b));

    // Write to a temp file, parse, verify, delete.
    const tmp_path = "/tmp/tripvulkan_smoke.safetensors";
    {
        const f = try std.fs.cwd().createFile(tmp_path, .{ .truncate = true });
        defer f.close();
        try f.writeAll(blob.items);
    }
    defer std.fs.cwd().deleteFile(tmp_path) catch {};

    var st = try safetensors.SafeTensors.open(allocator, tmp_path);
    defer st.deinit();

    if (st.count() != 2) {
        std.debug.print("safetensors MISMATCH: expected 2 tensors, got {d}\n", .{st.count()});
        return error.ParityFailed;
    }

    const ta = st.get("weight_a") orelse return error.MissingTensor;
    if (ta.dtype != .f32 or ta.shape.len != 2 or ta.shape[0] != 3 or ta.shape[1] != 2) {
        std.debug.print("weight_a metadata wrong: dtype={any} shape={any}\n", .{ ta.dtype, ta.shape });
        return error.ParityFailed;
    }
    const ta_f32 = ta.asF32();
    for (ta_f32, w_a, 0..) |got, want, i| {
        if (got != want) {
            std.debug.print("weight_a[{d}] MISMATCH: got {d}, expected {d}\n", .{ i, got, want });
            return error.ParityFailed;
        }
    }

    const tb = st.get("weight_b") orelse return error.MissingTensor;
    if (tb.dtype != .i32 or tb.shape.len != 1 or tb.shape[0] != 4) {
        std.debug.print("weight_b metadata wrong: dtype={any} shape={any}\n", .{ tb.dtype, tb.shape });
        return error.ParityFailed;
    }
    const tb_i32 = @as([*]align(1) const i32, @ptrCast(tb.bytes.ptr))[0..4];
    for (tb_i32, w_b, 0..) |got, want, i| {
        if (got != want) {
            std.debug.print("weight_b[{d}] MISMATCH: got {d}, expected {d}\n", .{ i, got, want });
            return error.ParityFailed;
        }
    }

    std.debug.print("PASS safetensors round-trip (2 tensors, F32+I32)\n", .{});
}
