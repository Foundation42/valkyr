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
const cpu_forward = @import("cpu/forward.zig");
const tokenizer_mod = @import("tokenizer.zig");
const config_mod = @import("config.zig");
const gpu_model = @import("gpu/model.zig");
const gpu_scratch = @import("gpu/scratch.zig");
const gpu_recorder = @import("gpu/recorder.zig");
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
    if (args.len >= 4 and std.mem.eql(u8, args[1], "--layer0-test")) {
        const token_id = try std.fmt.parseInt(u32, args[3], 10);
        try runLayer0Test(allocator, args[2], token_id);
        return;
    }
    if (args.len >= 4 and std.mem.eql(u8, args[1], "--gen")) {
        const token_id = try std.fmt.parseInt(u32, args[3], 10);
        try runGen(allocator, args[2], token_id);
        return;
    }
    if (args.len >= 4 and std.mem.eql(u8, args[1], "--gpu-qproj-test")) {
        const token_id = try std.fmt.parseInt(u32, args[3], 10);
        try runGpuQprojTest(allocator, args[2], token_id);
        return;
    }
    if (args.len >= 4 and std.mem.eql(u8, args[1], "--gpu-rmsnorm-test")) {
        const token_id = try std.fmt.parseInt(u32, args[3], 10);
        try runGpuRmsnormTest(allocator, args[2], token_id);
        return;
    }
    if (args.len >= 4 and std.mem.eql(u8, args[1], "--gpu-geglu-test")) {
        const token_id = try std.fmt.parseInt(u32, args[3], 10);
        try runGpuGegluTest(allocator, args[2], token_id);
        return;
    }
    if (args.len >= 4 and std.mem.eql(u8, args[1], "--gpu-rope-test")) {
        const token_id = try std.fmt.parseInt(u32, args[3], 10);
        try runGpuRopeTest(allocator, args[2], token_id);
        return;
    }
    if (args.len >= 3 and std.mem.eql(u8, args[1], "--gpu-load")) {
        try runGpuLoad(allocator, args[2]);
        return;
    }
    if (args.len >= 4 and std.mem.eql(u8, args[1], "--gpu-layer0-test")) {
        const token_id = try std.fmt.parseInt(u32, args[3], 10);
        try runGpuLayer0Test(allocator, args[2], token_id);
        return;
    }
    if (args.len >= 4 and std.mem.eql(u8, args[1], "--gpu-gen")) {
        const token_id = try std.fmt.parseInt(u32, args[3], 10);
        try runGpuGen(allocator, args[2], token_id);
        return;
    }

    try runVecAddSmoke(allocator);
    try runSafeTensorsSmoke(allocator);
    try runMatmulSmoke(allocator);
    try runRopeIdentitySmoke(allocator);
    try runSoftmaxSmoke(allocator);
    try runGeluSmoke(allocator);
    try runGpuMatmulSmoke(allocator);
    try runGpuMatmulV2Smoke(allocator);
    try runGpuRmsnormSmoke(allocator);
    try runGpuGegluSmoke(allocator);
    try runGpuRopeSmoke(allocator);
    try runGpuSoftmaxSmoke(allocator);
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

// ── gpu matmul_nt smoke: synthetic A·Bᵀ on the GPU vs CPU expected ──

const MatmulPush = extern struct { m: u32, n: u32, k: u32 };

fn runGpuMatmulSmoke(allocator: std.mem.Allocator) !void {
    var ctx = try vk.Context.init(allocator);
    defer ctx.deinit();

    // Same problem as runMatmulSmoke (CPU): 2x3 · (4x3)ᵀ → 2x4.
    const a = [_]f32{ 1, 2, 3, 4, 5, 6 };
    const b = [_]f32{ 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1 };
    const want = [_]f32{ 1, 2, 3, 6, 4, 5, 6, 15 };
    const m: u32 = 2;
    const n: u32 = 4;
    const k: u32 = 3;

    var buf_a = try buffer.Buffer.initStatic(&ctx, f32, &a);
    defer buf_a.deinit(ctx.device);
    var buf_b = try buffer.Buffer.initStatic(&ctx, f32, &b);
    defer buf_b.deinit(ctx.device);
    var buf_c = try buffer.Buffer.initDeviceOnly(&ctx, m * n * @sizeOf(f32));
    defer buf_c.deinit(ctx.device);

    var kern = try pipeline.Kernel.init(&ctx, &shaders.matmul_nt, 3, @sizeOf(MatmulPush));
    defer kern.deinit();
    try kern.bind(&.{ &buf_a, &buf_b, &buf_c });

    const local_xy: u32 = 16;
    const groups_x: u32 = (m + local_xy - 1) / local_xy;
    const groups_y: u32 = (n + local_xy - 1) / local_xy;
    const push = MatmulPush{ .m = m, .n = n, .k = k };

    try buffer.submitOneShot(&ctx, struct {
        kern: *const pipeline.Kernel,
        push: *const MatmulPush,
        gx: u32,
        gy: u32,
        pub fn record(s: @This(), cmd: vk.c.VkCommandBuffer) void {
            s.kern.dispatch(cmd, s.push, s.gx, s.gy, 1);
        }
    }{ .kern = &kern, .push = &push, .gx = groups_x, .gy = groups_y });

    var out: [8]f32 = undefined;
    try buf_c.readBack(&ctx, f32, &out);
    for (out, want, 0..) |got, w, i| {
        if (got != w) {
            std.debug.print("GPU matmul MISMATCH at {d}: got {d}, expected {d}\n", .{ i, got, w });
            return error.ParityFailed;
        }
    }
    std.debug.print("PASS GPU matmul_nt synthetic (2×3 · (4×3)ᵀ → 2×4) on {s}\n", .{ctx.deviceName()});
}

// ── gpu matmul_nt_v2 smoke: cooperative-K kernel vs hand-checked ───

fn runGpuMatmulV2Smoke(allocator: std.mem.Allocator) !void {
    var ctx = try vk.Context.init(allocator);
    defer ctx.deinit();

    // Same problem as runGpuMatmulSmoke: 2x3 · (4x3)ᵀ → 2x4.
    // Note v2 needs K large enough that the cooperative reduction
    // exercises something — but correctness with tiny K is also a
    // useful sanity check (most threads get nothing to do, the result
    // should still be right).
    const a = [_]f32{ 1, 2, 3, 4, 5, 6 };
    const b = [_]f32{ 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1 };
    const want = [_]f32{ 1, 2, 3, 6, 4, 5, 6, 15 };
    const m: u32 = 2;
    const n: u32 = 4;
    const k: u32 = 3;

    var buf_a = try buffer.Buffer.initStatic(&ctx, f32, &a);
    defer buf_a.deinit(ctx.device);
    var buf_b = try buffer.Buffer.initStatic(&ctx, f32, &b);
    defer buf_b.deinit(ctx.device);
    var buf_c = try buffer.Buffer.initDeviceOnly(&ctx, m * n * @sizeOf(f32));
    defer buf_c.deinit(ctx.device);

    var kern = try pipeline.Kernel.init(&ctx, &shaders.matmul_nt_v2, 3, @sizeOf(MatmulPush));
    defer kern.deinit();
    try kern.bind(&.{ &buf_a, &buf_b, &buf_c });

    // v2 dispatches one WG per output cell — total = M*N WGs.
    const groups: u32 = m * n;
    const push = MatmulPush{ .m = m, .n = n, .k = k };
    try buffer.submitOneShot(&ctx, struct {
        kern: *const pipeline.Kernel,
        push: *const MatmulPush,
        gx: u32,
        pub fn record(s: @This(), cmd: vk.c.VkCommandBuffer) void {
            s.kern.dispatch(cmd, s.push, s.gx, 1, 1);
        }
    }{ .kern = &kern, .push = &push, .gx = groups });

    var out: [8]f32 = undefined;
    try buf_c.readBack(&ctx, f32, &out);
    for (out, want, 0..) |got, w, i| {
        if (got != w) {
            std.debug.print("GPU matmul_v2 MISMATCH at {d}: got {d}, expected {d}\n", .{ i, got, w });
            return error.ParityFailed;
        }
    }
    std.debug.print("PASS GPU matmul_nt_v2 synthetic (cooperative-K, 2×3 · (4×3)ᵀ → 2×4)\n", .{});
}

// ── gpu rmsnorm smoke: synthetic vs CPU rmsnorm ────────────────────

const RmsnormPush = extern struct {
    dim: u32,
    eps: f32,
    gemma_quirk: u32,
};

fn runGpuRmsnormSmoke(allocator: std.mem.Allocator) !void {
    var ctx = try vk.Context.init(allocator);
    defer ctx.deinit();

    // Two cases — Llama (no quirk) and Gemma (1+w quirk) — exercising
    // both code paths through the same shader.
    inline for (.{ false, true }) |gemma_quirk| {
        const dim: usize = 1024;
        const x = try allocator.alloc(f32, dim);
        defer allocator.free(x);
        const w = try allocator.alloc(f32, dim);
        defer allocator.free(w);
        for (x, 0..) |*v, i| v.* = 0.5 - @as(f32, @floatFromInt(i & 31)) * 0.03;
        for (w, 0..) |*v, i| v.* = -0.1 + @as(f32, @floatFromInt(i & 15)) * 0.02;

        // ── CPU oracle ──────────────────────────────────────────────
        const want = try allocator.alloc(f32, dim);
        defer allocator.free(want);
        const fake_w_tensor = safetensors.Tensor{
            .dtype = .f32,
            .shape = &.{dim},
            .bytes = std.mem.sliceAsBytes(w),
        };
        const family: config_mod.Family = if (gemma_quirk) .gemma else .llama;
        try cpu_math.rmsnorm(want, x, fake_w_tensor, 1e-6, family);

        // ── GPU dispatch ────────────────────────────────────────────
        var buf_a = try buffer.Buffer.initStatic(&ctx, f32, x);
        defer buf_a.deinit(ctx.device);
        var buf_w = try buffer.Buffer.initStatic(&ctx, f32, w);
        defer buf_w.deinit(ctx.device);
        var buf_c = try buffer.Buffer.initDeviceOnly(&ctx, dim * @sizeOf(f32));
        defer buf_c.deinit(ctx.device);

        var kern = try pipeline.Kernel.init(&ctx, &shaders.rmsnorm, 3, @sizeOf(RmsnormPush));
        defer kern.deinit();
        try kern.bind(&.{ &buf_a, &buf_w, &buf_c });

        const push = RmsnormPush{
            .dim = @intCast(dim),
            .eps = 1e-6,
            .gemma_quirk = if (gemma_quirk) 1 else 0,
        };
        try buffer.submitOneShot(&ctx, struct {
            kern: *const pipeline.Kernel,
            push: *const RmsnormPush,
            pub fn record(s: @This(), cmd: vk.c.VkCommandBuffer) void {
                s.kern.dispatch(cmd, s.push, 1, 1, 1);
            }
        }{ .kern = &kern, .push = &push });

        const got = try allocator.alloc(f32, dim);
        defer allocator.free(got);
        try buf_c.readBack(&ctx, f32, got);

        var max_abs: f32 = 0;
        for (got, want) |g, e| {
            const d = @abs(g - e);
            if (d > max_abs) max_abs = d;
        }
        if (max_abs > 1e-5) {
            std.debug.print("GPU rmsnorm gemma_quirk={any}: max |Δ| = {e}\n", .{ gemma_quirk, max_abs });
            return error.ParityFailed;
        }
    }
    std.debug.print("PASS GPU rmsnorm synthetic (Llama + Gemma variants, dim=1024)\n", .{});
}

// ── gpu geglu smoke: synthetic vs CPU geglu ─────────────────────────

const GegluPush = extern struct { n: u32 };

fn runGpuGegluSmoke(allocator: std.mem.Allocator) !void {
    var ctx = try vk.Context.init(allocator);
    defer ctx.deinit();

    const n: usize = 4096;
    const gate = try allocator.alloc(f32, n);
    defer allocator.free(gate);
    const upv = try allocator.alloc(f32, n);
    defer allocator.free(upv);
    // Range [-3, 3] across the array hits both the tanh saturation
    // tails and the linear region around 0 — exercises the full curve.
    for (gate, upv, 0..) |*g, *u, i| {
        const t = @as(f32, @floatFromInt(i)) / @as(f32, @floatFromInt(n));
        g.* = -3.0 + 6.0 * t;
        u.* = 1.0 - 2.0 * t;
    }

    const want = try allocator.alloc(f32, n);
    defer allocator.free(want);
    try cpu_math.geglu(want, gate, upv);

    var buf_g = try buffer.Buffer.initStatic(&ctx, f32, gate);
    defer buf_g.deinit(ctx.device);
    var buf_u = try buffer.Buffer.initStatic(&ctx, f32, upv);
    defer buf_u.deinit(ctx.device);
    var buf_o = try buffer.Buffer.initDeviceOnly(&ctx, n * @sizeOf(f32));
    defer buf_o.deinit(ctx.device);

    var kern = try pipeline.Kernel.init(&ctx, &shaders.geglu, 3, @sizeOf(GegluPush));
    defer kern.deinit();
    try kern.bind(&.{ &buf_g, &buf_u, &buf_o });

    const local: u32 = 256;
    const groups: u32 = (@as(u32, @intCast(n)) + local - 1) / local;
    const push = GegluPush{ .n = @intCast(n) };
    try buffer.submitOneShot(&ctx, struct {
        kern: *const pipeline.Kernel,
        push: *const GegluPush,
        groups: u32,
        pub fn record(s: @This(), cmd: vk.c.VkCommandBuffer) void {
            s.kern.dispatch(cmd, s.push, s.groups, 1, 1);
        }
    }{ .kern = &kern, .push = &push, .groups = groups });

    const got = try allocator.alloc(f32, n);
    defer allocator.free(got);
    try buf_o.readBack(&ctx, f32, got);

    var max_abs: f32 = 0;
    for (got, want) |g, e| {
        const d = @abs(g - e);
        if (d > max_abs) max_abs = d;
    }
    if (max_abs > 1e-5) {
        std.debug.print("GPU GeGLU: max |Δ| = {e}\n", .{max_abs});
        return error.ParityFailed;
    }
    std.debug.print("PASS GPU geglu synthetic ({d} elems, x∈[-3,3])\n", .{n});
}

// ── gpu rope smoke: pos=0 identity + pos=1 vs CPU ───────────────────

const RopePush = extern struct {
    n_heads: u32,
    head_dim: u32,
    pos: u32,
    theta_base: f32,
};

fn runGpuRopeSmoke(allocator: std.mem.Allocator) !void {
    var ctx = try vk.Context.init(allocator);
    defer ctx.deinit();

    const n_heads: usize = 8;
    const head_dim: usize = 64;
    const total = n_heads * head_dim;

    const in_v = try allocator.alloc(f32, total);
    defer allocator.free(in_v);
    for (in_v, 0..) |*x, i| x.* = @as(f32, @floatFromInt(i)) * 0.001 - 0.5;

    var buf_in = try buffer.Buffer.initStatic(&ctx, f32, in_v);
    defer buf_in.deinit(ctx.device);
    var buf_out = try buffer.Buffer.initDeviceOnly(&ctx, total * @sizeOf(f32));
    defer buf_out.deinit(ctx.device);

    var kern = try pipeline.Kernel.init(&ctx, &shaders.rope, 2, @sizeOf(RopePush));
    defer kern.deinit();
    try kern.bind(&.{ &buf_in, &buf_out });

    // pos=0: must be identity.
    const local: u32 = 256;
    const pairs: u32 = @intCast(n_heads * (head_dim / 2));
    const groups: u32 = (pairs + local - 1) / local;

    const push0 = RopePush{ .n_heads = @intCast(n_heads), .head_dim = @intCast(head_dim), .pos = 0, .theta_base = 10000.0 };
    try buffer.submitOneShot(&ctx, struct {
        kern: *const pipeline.Kernel,
        push: *const RopePush,
        groups: u32,
        pub fn record(s: @This(), cmd: vk.c.VkCommandBuffer) void {
            s.kern.dispatch(cmd, s.push, s.groups, 1, 1);
        }
    }{ .kern = &kern, .push = &push0, .groups = groups });

    const got0 = try allocator.alloc(f32, total);
    defer allocator.free(got0);
    try buf_out.readBack(&ctx, f32, got0);
    for (got0, in_v, 0..) |g, e, i| {
        if (g != e) {
            std.debug.print("GPU RoPE pos=0 NOT identity at {d}: in={d} out={d}\n", .{ i, e, g });
            return error.ParityFailed;
        }
    }

    // pos=1: parity vs CPU.
    const want = try allocator.alloc(f32, total);
    defer allocator.free(want);
    try cpu_math.applyRope(want, in_v, n_heads, head_dim, 1, 10000.0);

    const push1 = RopePush{ .n_heads = @intCast(n_heads), .head_dim = @intCast(head_dim), .pos = 1, .theta_base = 10000.0 };
    try buffer.submitOneShot(&ctx, struct {
        kern: *const pipeline.Kernel,
        push: *const RopePush,
        groups: u32,
        pub fn record(s: @This(), cmd: vk.c.VkCommandBuffer) void {
            s.kern.dispatch(cmd, s.push, s.groups, 1, 1);
        }
    }{ .kern = &kern, .push = &push1, .groups = groups });

    const got1 = try allocator.alloc(f32, total);
    defer allocator.free(got1);
    try buf_out.readBack(&ctx, f32, got1);

    var max_abs: f32 = 0;
    for (got1, want) |g, e| {
        const d = @abs(g - e);
        if (d > max_abs) max_abs = d;
    }
    if (max_abs > 1e-5) {
        std.debug.print("GPU RoPE pos=1: max |Δ| = {e}\n", .{max_abs});
        return error.ParityFailed;
    }
    std.debug.print("PASS GPU rope (pos=0 identity + pos=1 vs CPU within 1e-5)\n", .{});
}

// ── gpu softmax smoke: synthetic vs CPU softmax ─────────────────────

const SoftmaxPush = extern struct { dim: u32 };

fn runGpuSoftmaxSmoke(allocator: std.mem.Allocator) !void {
    var ctx = try vk.Context.init(allocator);
    defer ctx.deinit();

    const dim: usize = 2048;
    const x = try allocator.alloc(f32, dim);
    defer allocator.free(x);
    // Mix of negative, near-zero, and one big positive — exercises the
    // numerical-stability subtract-max path.
    for (x, 0..) |*v, i| v.* = @as(f32, @floatFromInt(@as(i32, @intCast(i)) - 1024)) * 0.01;
    x[42] = 100.0; // a clear winner; without subtract-max would overflow exp

    const want = try allocator.alloc(f32, dim);
    defer allocator.free(want);
    @memcpy(want, x);
    cpu_math.softmax(want);

    var buf_in = try buffer.Buffer.initStatic(&ctx, f32, x);
    defer buf_in.deinit(ctx.device);
    var buf_out = try buffer.Buffer.initDeviceOnly(&ctx, dim * @sizeOf(f32));
    defer buf_out.deinit(ctx.device);

    var kern = try pipeline.Kernel.init(&ctx, &shaders.softmax, 2, @sizeOf(SoftmaxPush));
    defer kern.deinit();
    try kern.bind(&.{ &buf_in, &buf_out });

    const push = SoftmaxPush{ .dim = @intCast(dim) };
    try buffer.submitOneShot(&ctx, struct {
        kern: *const pipeline.Kernel,
        push: *const SoftmaxPush,
        pub fn record(s: @This(), cmd: vk.c.VkCommandBuffer) void {
            s.kern.dispatch(cmd, s.push, 1, 1, 1);
        }
    }{ .kern = &kern, .push = &push });

    const got = try allocator.alloc(f32, dim);
    defer allocator.free(got);
    try buf_out.readBack(&ctx, f32, got);

    var max_abs: f32 = 0;
    var sum: f32 = 0;
    for (got, want) |g, e| {
        const d = @abs(g - e);
        if (d > max_abs) max_abs = d;
        sum += g;
    }
    if (max_abs > 1e-5) {
        std.debug.print("GPU softmax: max |Δ| = {e}\n", .{max_abs});
        return error.ParityFailed;
    }
    if (@abs(sum - 1.0) > 1e-5) {
        std.debug.print("GPU softmax sum {d} != 1\n", .{sum});
        return error.ParityFailed;
    }
    std.debug.print("PASS GPU softmax synthetic (dim=2048, sum=1±1e-5, vs CPU 1e-5)\n", .{});
}

// ── gelu_tanh smoke: scalar against PyTorch reference values ────────

fn runGeluSmoke(allocator: std.mem.Allocator) !void {
    _ = allocator;
    // Reference values from torch.nn.functional.gelu(approximate="tanh")
    // (which matches HF's gelu_pytorch_tanh, which Gemma uses).
    const cases = [_]struct { x: f32, want: f32 }{
        .{ .x = 0.0, .want = 0.0 },
        .{ .x = 1.0, .want = 0.8411919876 },
        .{ .x = -1.0, .want = -0.15880800784 },
        .{ .x = 2.0, .want = 1.9545976400 },
        .{ .x = -2.0, .want = -0.04540234059 },
    };
    for (cases) |tc| {
        const got = cpu_math.gelu_tanh(tc.x);
        const err = @abs(got - tc.want);
        if (err > 1e-5) {
            std.debug.print("gelu_tanh({d}): got {d}, want {d} (err {e})\n", .{ tc.x, got, tc.want, err });
            return error.ParityFailed;
        }
    }
    std.debug.print("PASS gelu_tanh (5 ref values within 1e-5)\n", .{});
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

// ── layer0-test: complete transformer block (attn + FFN + residuals) ──

fn runLayer0Test(allocator: std.mem.Allocator, dir_path: []const u8, token_id: u32) !void {
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
    try cpu_math.matmul_nt(q, x_norm1, layer.q_proj, 1, q_dim, hidden);
    try cpu_math.matmul_nt(k, x_norm1, layer.k_proj, 1, kv_dim, hidden);
    try cpu_math.matmul_nt(v, x_norm1, layer.v_proj, 1, kv_dim, hidden);

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
    try cpu_math.matmul_nt(attn_out, head_out, layer.o_proj, 1, hidden, q_dim);
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

// ── gen: full forward + greedy + tokenizer decode ──────────────────

fn runGen(gpa: std.mem.Allocator, dir_path: []const u8, token_id: u32) !void {
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
    try cpu_forward.forward(&model, token_id, 0, scratch, logits);
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

// ── gpu-gen: full forward on GPU + parity vs CPU --gen ─────────────

fn runGpuGen(gpa: std.mem.Allocator, dir_path: []const u8, token_id: u32) !void {
    var cpu = try model_mod.Model.load(gpa, dir_path);
    defer cpu.deinit();
    const cfg = cpu.config;

    const tok_path = try std.fmt.allocPrint(gpa, "{s}/tokenizer.json", .{dir_path});
    defer gpa.free(tok_path);
    var tok = try tokenizer_mod.Tokenizer.loadFromFile(gpa, tok_path);
    defer tok.deinit();

    var ctx = try vk.Context.init(gpa);
    defer ctx.deinit();

    const stdout = std.io.getStdOut().writer();
    const input_str = tok.decode(token_id) orelse "<unknown>";
    try stdout.print("input  token id={d}  string={s}\n", .{ token_id, input_str });
    try stdout.print("device: {s}\n\n", .{ctx.deviceName()});

    const t_up0 = std.time.nanoTimestamp();
    var gm = try gpu_model.GpuModel.upload(gpa, &ctx, &cpu);
    defer gm.deinit(ctx.device);
    const t_up1 = std.time.nanoTimestamp();
    const upload_ms = @as(f64, @floatFromInt(t_up1 - t_up0)) / 1_000_000.0;
    try stdout.print("upload: {d:.0} ms\n", .{upload_ms});

    var sc = try gpu_scratch.GpuScratch.init(&ctx, cfg);
    defer sc.deinit(ctx.device);

    // ── Build kernels once, rebind per-layer ────────────────────────
    var k_embed = try pipeline.Kernel.init(&ctx, &shaders.embed_lookup, 2, @sizeOf(EmbedLookupPush));
    defer k_embed.deinit();
    var k_rmsnorm = try pipeline.Kernel.init(&ctx, &shaders.rmsnorm, 3, @sizeOf(RmsnormPush));
    defer k_rmsnorm.deinit();
    var k_matmul = try pipeline.Kernel.init(&ctx, &shaders.matmul_nt_v2, 3, @sizeOf(MatmulPush));
    defer k_matmul.deinit();
    var k_rope = try pipeline.Kernel.init(&ctx, &shaders.rope, 2, @sizeOf(RopePush));
    defer k_rope.deinit();
    var k_attn = try pipeline.Kernel.init(&ctx, &shaders.attn_decode_single, 2, @sizeOf(AttnDecodeSinglePush));
    defer k_attn.deinit();
    var k_add = try pipeline.Kernel.init(&ctx, &shaders.add_in_place, 2, @sizeOf(AddInPlacePush));
    defer k_add.deinit();
    var k_geglu = try pipeline.Kernel.init(&ctx, &shaders.geglu, 3, @sizeOf(GegluPush));
    defer k_geglu.deinit();

    const hidden: u32 = @intCast(cfg.hidden_size);
    const inter: u32 = @intCast(cfg.intermediate_size);
    const q_dim: u32 = @intCast(cfg.num_attention_heads * cfg.head_dim);
    const kv_dim: u32 = @intCast(cfg.num_key_value_heads * cfg.head_dim);
    const vocab: u32 = @intCast(cfg.vocab_size);
    const gemma_quirk: u32 = if (cfg.family == .gemma) 1 else 0;

    // ── Recorder: one command buffer for the whole forward pass ────
    // Sizing: 291 dispatches at ~3 storage-buffer descriptors each =
    // ~728 descriptors. Round up generously so we don't trip on any
    // future kernel that adds bindings.
    var rec = try gpu_recorder.Recorder.init(&ctx, 512, 2048);
    defer rec.deinit();

    const rms_push = RmsnormPush{ .dim = hidden, .eps = cfg.rms_norm_eps, .gemma_quirk = gemma_quirk };
    const add_push = AddInPlacePush{ .n = hidden };
    const attn_push = AttnDecodeSinglePush{
        .n_heads = @intCast(cfg.num_attention_heads),
        .heads_per_kv = @intCast(cfg.num_attention_heads / cfg.num_key_value_heads),
        .head_dim = @intCast(cfg.head_dim),
    };
    const rope_q_push = RopePush{
        .n_heads = @intCast(cfg.num_attention_heads),
        .head_dim = @intCast(cfg.head_dim),
        .pos = 0,
        .theta_base = cfg.rope_theta,
    };
    const rope_k_push = RopePush{
        .n_heads = @intCast(cfg.num_key_value_heads),
        .head_dim = @intCast(cfg.head_dim),
        .pos = 0,
        .theta_base = cfg.rope_theta,
    };
    const geglu_push = GegluPush{ .n = inter };
    const embed_push = EmbedLookupPush{
        .token_id = token_id,
        .dim = hidden,
        .scale = if (cfg.family.embedScalesByDim()) @sqrt(@as(f32, @floatFromInt(hidden))) else 1.0,
    };

    const t_gpu0 = std.time.nanoTimestamp();

    try rec.begin();

    // Embed lookup → residual stream.
    try recDispatch1D(&rec, &k_embed, &.{ &gm.embed_tokens, &sc.stream }, &embed_push, hidden);

    // 18 transformer blocks.
    for (gm.layers) |*layer| {
        try recDispatchPerRow(&rec, &k_rmsnorm, &.{ &sc.stream, &layer.input_layernorm, &sc.x_norm }, &rms_push, 1);

        try recDispatchMatmul(&rec, &k_matmul, &.{ &sc.x_norm, &layer.q_proj, &sc.q }, 1, q_dim, hidden);
        try recDispatchMatmul(&rec, &k_matmul, &.{ &sc.x_norm, &layer.k_proj, &sc.k }, 1, kv_dim, hidden);
        try recDispatchMatmul(&rec, &k_matmul, &.{ &sc.x_norm, &layer.v_proj, &sc.v }, 1, kv_dim, hidden);

        try recDispatchRope(&rec, &k_rope, &.{ &sc.q, &sc.q_rot }, &rope_q_push, cfg.num_attention_heads, cfg.head_dim);
        try recDispatchRope(&rec, &k_rope, &.{ &sc.k, &sc.k_rot }, &rope_k_push, cfg.num_key_value_heads, cfg.head_dim);

        try recDispatch1D(&rec, &k_attn, &.{ &sc.v, &sc.head_out }, &attn_push, q_dim);

        try recDispatchMatmul(&rec, &k_matmul, &.{ &sc.head_out, &layer.o_proj, &sc.attn_out }, 1, hidden, q_dim);

        try recDispatch1D(&rec, &k_add, &.{ &sc.stream, &sc.attn_out }, &add_push, hidden);

        try recDispatchPerRow(&rec, &k_rmsnorm, &.{ &sc.stream, &layer.post_attention_layernorm, &sc.mid_norm }, &rms_push, 1);

        try recDispatchMatmul(&rec, &k_matmul, &.{ &sc.mid_norm, &layer.gate_proj, &sc.gate }, 1, inter, hidden);
        try recDispatchMatmul(&rec, &k_matmul, &.{ &sc.mid_norm, &layer.up_proj, &sc.up }, 1, inter, hidden);

        try recDispatch1D(&rec, &k_geglu, &.{ &sc.gate, &sc.up, &sc.fused }, &geglu_push, inter);

        try recDispatchMatmul(&rec, &k_matmul, &.{ &sc.fused, &layer.down_proj, &sc.ffn_out }, 1, hidden, inter);

        try recDispatch1D(&rec, &k_add, &.{ &sc.stream, &sc.ffn_out }, &add_push, hidden);
    }

    // Final rmsnorm + LM head.
    try recDispatchPerRow(&rec, &k_rmsnorm, &.{ &sc.stream, &gm.final_norm, &sc.final_norm_out }, &rms_push, 1);
    try recDispatchMatmul(&rec, &k_matmul, &.{ &sc.final_norm_out, &gm.lm_head, &sc.logits }, 1, vocab, hidden);

    try rec.endAndSubmit();

    const t_gpu1 = std.time.nanoTimestamp();
    const gpu_ms = @as(f64, @floatFromInt(t_gpu1 - t_gpu0)) / 1_000_000.0;

    // ── Read back logits, argmax, decode ───────────────────────────
    const logits = try gpa.alloc(f32, cfg.vocab_size);
    defer gpa.free(logits);
    try sc.logits.readBack(&ctx, f32, logits);

    const sampled = cpu_forward.argmax(logits);
    const out_str = tok.decode(sampled) orelse "<unknown>";

    try stdout.print("forward (GPU, ~291 dispatches in 1 command buffer): {d:.0} ms\n", .{gpu_ms});

    // A second forward to measure the warm-cache time. Recorder needs
    // to be reset before re-recording.
    try rec.reset();
    const t_warm0 = std.time.nanoTimestamp();
    try rec.begin();
    try recDispatch1D(&rec, &k_embed, &.{ &gm.embed_tokens, &sc.stream }, &embed_push, hidden);
    for (gm.layers) |*layer| {
        try recDispatchPerRow(&rec, &k_rmsnorm, &.{ &sc.stream, &layer.input_layernorm, &sc.x_norm }, &rms_push, 1);
        try recDispatchMatmul(&rec, &k_matmul, &.{ &sc.x_norm, &layer.q_proj, &sc.q }, 1, q_dim, hidden);
        try recDispatchMatmul(&rec, &k_matmul, &.{ &sc.x_norm, &layer.k_proj, &sc.k }, 1, kv_dim, hidden);
        try recDispatchMatmul(&rec, &k_matmul, &.{ &sc.x_norm, &layer.v_proj, &sc.v }, 1, kv_dim, hidden);
        try recDispatchRope(&rec, &k_rope, &.{ &sc.q, &sc.q_rot }, &rope_q_push, cfg.num_attention_heads, cfg.head_dim);
        try recDispatchRope(&rec, &k_rope, &.{ &sc.k, &sc.k_rot }, &rope_k_push, cfg.num_key_value_heads, cfg.head_dim);
        try recDispatch1D(&rec, &k_attn, &.{ &sc.v, &sc.head_out }, &attn_push, q_dim);
        try recDispatchMatmul(&rec, &k_matmul, &.{ &sc.head_out, &layer.o_proj, &sc.attn_out }, 1, hidden, q_dim);
        try recDispatch1D(&rec, &k_add, &.{ &sc.stream, &sc.attn_out }, &add_push, hidden);
        try recDispatchPerRow(&rec, &k_rmsnorm, &.{ &sc.stream, &layer.post_attention_layernorm, &sc.mid_norm }, &rms_push, 1);
        try recDispatchMatmul(&rec, &k_matmul, &.{ &sc.mid_norm, &layer.gate_proj, &sc.gate }, 1, inter, hidden);
        try recDispatchMatmul(&rec, &k_matmul, &.{ &sc.mid_norm, &layer.up_proj, &sc.up }, 1, inter, hidden);
        try recDispatch1D(&rec, &k_geglu, &.{ &sc.gate, &sc.up, &sc.fused }, &geglu_push, inter);
        try recDispatchMatmul(&rec, &k_matmul, &.{ &sc.fused, &layer.down_proj, &sc.ffn_out }, 1, hidden, inter);
        try recDispatch1D(&rec, &k_add, &.{ &sc.stream, &sc.ffn_out }, &add_push, hidden);
    }
    try recDispatchPerRow(&rec, &k_rmsnorm, &.{ &sc.stream, &gm.final_norm, &sc.final_norm_out }, &rms_push, 1);
    try recDispatchMatmul(&rec, &k_matmul, &.{ &sc.final_norm_out, &gm.lm_head, &sc.logits }, 1, vocab, hidden);
    try rec.endAndSubmit();
    const t_warm1 = std.time.nanoTimestamp();
    const warm_ms = @as(f64, @floatFromInt(t_warm1 - t_warm0)) / 1_000_000.0;
    try stdout.print("forward (warm, 2nd pass)                          : {d:.0} ms\n", .{warm_ms});

    const k_top: usize = 5;
    const top = try topK(gpa, logits, k_top);
    defer gpa.free(top);
    try stdout.print("\ntop {d} logits (GPU):\n", .{k_top});
    for (top) |entry| {
        const s = tok.decode(entry.id) orelse "<unknown>";
        try stdout.print("  id={d:>6}  logit={d:>10.4}  {s}\n", .{ entry.id, entry.value, s });
    }
    try stdout.print("\nGPU sampled (greedy): id={d}  string={s}\n", .{ sampled, out_str });

    // ── Parity vs CPU --gen ─────────────────────────────────────────
    try stdout.print("\nrunning CPU forward for parity check (this takes ~18 s)...\n", .{});
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const cpu_logits = try gpa.alloc(f32, cfg.vocab_size);
    defer gpa.free(cpu_logits);
    const t_cpu0 = std.time.nanoTimestamp();
    try cpu_forward.forward(&cpu, token_id, 0, arena.allocator(), cpu_logits);
    const t_cpu1 = std.time.nanoTimestamp();
    const cpu_ms = @as(f64, @floatFromInt(t_cpu1 - t_cpu0)) / 1_000_000.0;

    const cpu_argmax = cpu_forward.argmax(cpu_logits);
    var max_abs: f32 = 0;
    var max_idx: usize = 0;
    for (logits, cpu_logits, 0..) |g, c, i| {
        const d = @abs(g - c);
        if (d > max_abs) {
            max_abs = d;
            max_idx = i;
        }
    }

    try stdout.print("CPU forward: {d:.0} ms\n", .{cpu_ms});
    try stdout.print("CPU argmax: id={d} ({s}) logit={d:.4}\n", .{
        cpu_argmax, tok.decode(cpu_argmax) orelse "?", cpu_logits[cpu_argmax],
    });
    try stdout.print("max |Δ| over the whole logit vector = {e:.3}  (at idx {d}: cpu={d:.4} gpu={d:.4})\n", .{
        max_abs, max_idx, cpu_logits[max_idx], logits[max_idx],
    });
    if (sampled != cpu_argmax) {
        std.debug.print("FAIL: GPU argmax {d} ≠ CPU argmax {d}\n", .{ sampled, cpu_argmax });
        return error.ParityFailed;
    }
    try stdout.print("\nPASS GPU --gen argmax matches CPU --gen ({d} → {s})\n", .{ sampled, out_str });
}

// ── gpu-layer0-test: full layer 0 forward on GPU vs CPU ────────────

const EmbedLookupPush = extern struct { token_id: u32, dim: u32, scale: f32 };
const AddInPlacePush = extern struct { n: u32 };
const AttnDecodeSinglePush = extern struct { n_heads: u32, heads_per_kv: u32, head_dim: u32 };

fn runGpuLayer0Test(gpa: std.mem.Allocator, dir_path: []const u8, token_id: u32) !void {
    var cpu = try model_mod.Model.load(gpa, dir_path);
    defer cpu.deinit();
    const cfg = cpu.config;

    var ctx = try vk.Context.init(gpa);
    defer ctx.deinit();

    const stdout = std.io.getStdOut().writer();
    try stdout.print("GPU layer-0 forward parity test — token {d}\n", .{token_id});
    try stdout.print("device: {s}\n\n", .{ctx.deviceName()});

    try stdout.print("uploading weights...\n", .{});
    var gm = try gpu_model.GpuModel.upload(gpa, &ctx, &cpu);
    defer gm.deinit(ctx.device);

    var sc = try gpu_scratch.GpuScratch.init(&ctx, cfg);
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
    try k_matmul.bind(&.{ &sc.x_norm, &gm.layers[0].q_proj, &sc.q });
    try dispatchMatmul(&ctx, &k_matmul, 1, q_dim, hidden);
    try k_matmul.bind(&.{ &sc.x_norm, &gm.layers[0].k_proj, &sc.k });
    try dispatchMatmul(&ctx, &k_matmul, 1, kv_dim, hidden);
    try k_matmul.bind(&.{ &sc.x_norm, &gm.layers[0].v_proj, &sc.v });
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
    try k_matmul.bind(&.{ &sc.head_out, &gm.layers[0].o_proj, &sc.attn_out });
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
    try cpu_math.matmul_nt(v, x_norm, layer.v_proj, 1, kv_dim, hidden);

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
    try cpu_math.matmul_nt(attn_out, head_out, layer.o_proj, 1, hidden, q_dim);
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

// ── Recorder-based dispatch helpers (one command buffer for whole pass) ──

fn recDispatch1D(
    rec: *gpu_recorder.Recorder,
    kern: *const pipeline.Kernel,
    bufs: []const *const buffer.Buffer,
    push: anytype,
    n: u32,
) !void {
    const local: u32 = 256;
    const groups: u32 = (n + local - 1) / local;
    try rec.dispatch(kern, bufs, push, groups, 1, 1);
}

fn recDispatchPerRow(
    rec: *gpu_recorder.Recorder,
    kern: *const pipeline.Kernel,
    bufs: []const *const buffer.Buffer,
    push: anytype,
    n_rows: u32,
) !void {
    try rec.dispatch(kern, bufs, push, n_rows, 1, 1);
}

fn recDispatchMatmul(
    rec: *gpu_recorder.Recorder,
    kern: *const pipeline.Kernel,
    bufs: []const *const buffer.Buffer,
    m: u32,
    n: u32,
    k: u32,
) !void {
    // matmul_nt_v2 dispatch: one WG per output cell, 256 threads each
    // cooperate over K with subgroup reduction. Workgroup count is
    // M*N which is well within the 65535 per-dim limit even at our
    // largest matmul (M=1, N=vocab_size=256000).
    const push = MatmulPush{ .m = m, .n = n, .k = k };
    try rec.dispatch(kern, bufs, &push, m * n, 1, 1);
}

fn recDispatchRope(
    rec: *gpu_recorder.Recorder,
    kern: *const pipeline.Kernel,
    bufs: []const *const buffer.Buffer,
    push: *const RopePush,
    n_heads: usize,
    head_dim: usize,
) !void {
    const local: u32 = 256;
    const pairs: u32 = @intCast(n_heads * (head_dim / 2));
    const groups: u32 = (pairs + local - 1) / local;
    try rec.dispatch(kern, bufs, push, groups, 1, 1);
}

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

fn runGpuLoad(gpa: std.mem.Allocator, dir_path: []const u8) !void {
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
    var gm = try gpu_model.GpuModel.upload(gpa, &ctx, &cpu);
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
    try roundTripCheck(gpa, &ctx, &gm.layers[0].q_proj, cpu.layers[0].q_proj, "layer 0 q_proj");
    try roundTripCheck(gpa, &ctx, &gm.layers[0].input_layernorm, cpu.layers[0].input_layernorm, "layer 0 input_layernorm");
    try roundTripCheck(gpa, &ctx, &gm.layers[17].down_proj, cpu.layers[17].down_proj, "layer 17 down_proj");

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

fn runGpuRmsnormTest(gpa: std.mem.Allocator, dir_path: []const u8, token_id: u32) !void {
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

fn runGpuRopeTest(gpa: std.mem.Allocator, dir_path: []const u8, token_id: u32) !void {
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
    try cpu_math.matmul_nt(q, x_norm, layer.q_proj, 1, q_dim, cfg.hidden_size);

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

fn runGpuGegluTest(gpa: std.mem.Allocator, dir_path: []const u8, token_id: u32) !void {
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
    try cpu_math.matmul_nt(v, x_norm, layer.v_proj, 1, v.len, cfg.hidden_size);

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
    try cpu_math.matmul_nt(attn_out, head_out, layer.o_proj, 1, cfg.hidden_size, head_out.len);

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

fn runGpuQprojTest(gpa: std.mem.Allocator, dir_path: []const u8, token_id: u32) !void {
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
    try cpu_math.matmul_nt(q_cpu, x_norm, model.layers[0].q_proj, 1, q_dim, cfg.hidden_size);
    const t_cpu1 = std.time.nanoTimestamp();
    const cpu_ms = @as(f64, @floatFromInt(t_cpu1 - t_cpu0)) / 1_000_000.0;

    // ── Materialise q_proj as fp32 for GPU upload ───────────────────
    // The GPU kernel is fp32-only for now; we'll add a bf16-aware
    // variant once the fp32 path is parity-clean. The conversion is
    // O(numel) so it doesn't dominate setup, but it does double host
    // memory while we hold both copies — fine for this kernel
    // (32 MiB), would want an in-place stream once we do all weights.
    const w_bf16 = dtype.asU16(model.layers[0].q_proj.bytes);
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
