//! Entry point. For phase-1 step-1 this is a smoke test: init Vulkan,
//! run a vector-add on the GPU, compare against a CPU reference, and
//! print "GPU OK on <device>". Once that lights up green we know the
//! whole stack — instance, device, buffers, descriptors, pipelines,
//! dispatch, readback — is sound and we can start writing real kernels.

const std = @import("std");
const vk = @import("gpu/vk.zig");
const buffer = @import("gpu/buffer.zig");
const pipeline = @import("gpu/pipeline.zig");
const shaders = @import("shaders");

const N: u32 = 1024 * 1024;

const PushConsts = extern struct { n: u32 };

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    var ctx = try vk.Context.init(allocator);
    defer ctx.deinit();

    std.debug.print("Vulkan up on {s}\n", .{ctx.deviceName()});

    // ── Host data ───────────────────────────────────────────────────
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

    // ── Device buffers ──────────────────────────────────────────────
    var buf_a = try buffer.Buffer.initStatic(&ctx, f32, a);
    defer buf_a.deinit(ctx.device);
    var buf_b = try buffer.Buffer.initStatic(&ctx, f32, b);
    defer buf_b.deinit(ctx.device);
    var buf_c = try buffer.Buffer.initDeviceOnly(&ctx, N * @sizeOf(f32));
    defer buf_c.deinit(ctx.device);

    // ── Kernel ──────────────────────────────────────────────────────
    var vec_add = try pipeline.Kernel.init(&ctx, &shaders.vec_add, 3, @sizeOf(PushConsts));
    defer vec_add.deinit();
    try vec_add.bind(&.{ &buf_a, &buf_b, &buf_c });

    // ── Dispatch ────────────────────────────────────────────────────
    const local_size: u32 = 256;
    const groups: u32 = (N + local_size - 1) / local_size;
    const push = PushConsts{ .n = N };

    try buffer.submitOneShot(&ctx, struct {
        kern: *const pipeline.Kernel,
        push: *const PushConsts,
        groups: u32,
        pub fn record(s: @This(), cmd: vk.c.VkCommandBuffer) void {
            s.kern.dispatch(cmd, s.push, s.groups, 1, 1);
        }
    }{ .kern = &vec_add, .push = &push, .groups = groups });

    // ── Readback + verify ───────────────────────────────────────────
    try buf_c.readBack(&ctx, f32, out);
    var first_bad: ?usize = null;
    for (out, 0..) |v, i| {
        const expected = @as(f32, @floatFromInt(i)) * 3.0;
        if (v != expected) {
            first_bad = i;
            break;
        }
    }
    if (first_bad) |i| {
        std.debug.print("MISMATCH at {d}: got {d}, expected {d}\n", .{
            i, out[i], @as(f32, @floatFromInt(i)) * 3.0,
        });
        return error.ParityFailed;
    }
    std.debug.print("GPU OK on {s} — vec_add ({d} elems) matches CPU\n", .{
        ctx.deviceName(), N,
    });
}
