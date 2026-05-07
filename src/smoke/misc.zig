//! Catch-all smokes that don't fit the other groups: `vec_add` (validates
//! the basic Vulkan compute roundtrip end to end) and `safetensors`
//! (synthesizes a 2-tensor file in memory, parses it back). Extracted
//! from main.zig.

const std = @import("std");
const vk = @import("../gpu/vk.zig");
const buffer = @import("../gpu/buffer.zig");
const pipeline = @import("../gpu/pipeline.zig");
const safetensors = @import("../safetensors.zig");
const shaders = @import("shaders");

// ── vec_add smoke: validates the whole Vulkan compute path ───────────

const N: u32 = 1024 * 1024;
const VecAddPush = extern struct { n: u32 };

pub fn runVecAddSmoke(allocator: std.mem.Allocator) !void {
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

pub fn runSafeTensorsSmoke(allocator: std.mem.Allocator) !void {
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
    const tmp_path = "/tmp/valkyr_smoke.safetensors";
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
