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

    try runVecAddSmoke(allocator);
    try runSafeTensorsSmoke(allocator);
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
