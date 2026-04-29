//! Storage-buffer lifecycle helper.
//!
//! Every buffer in the engine fits one of three modes:
//!
//!   1. `static`      — uploaded once and never written again. Backed by
//!                      DEVICE_LOCAL memory; init does a one-shot staging
//!                      copy. Use for model weights.
//!   2. `dynamic`     — CPU writes per step, GPU reads. HOST_VISIBLE +
//!                      HOST_COHERENT, persistent-mapped. Use for the
//!                      occasional CPU-prepared input that doesn't fit
//!                      a push constant.
//!   3. `device_only` — written exclusively by the GPU. DEVICE_LOCAL,
//!                      zero-filled at init via vkCmdFillBuffer. Use for
//!                      KV cache, residual stream, intermediate
//!                      activations, output logits.
//!
//! The pattern is lifted from the matryoshka renderer's ssbo.zig — the
//! same three-mode taxonomy that prevented HOST_VISIBLE-everywhere bugs
//! there applies here, just with weights instead of geometry.

const std = @import("std");
const vk = @import("vk.zig");
const c = vk.c;

pub const Mode = enum { static, dynamic, device_only };

pub const Buffer = struct {
    handle: c.VkBuffer,
    memory: c.VkDeviceMemory,
    /// Allocated capacity in bytes. May exceed actual data size when
    /// the caller padded for alignment or reserved headroom.
    bytes: usize,
    mode: Mode,
    /// Persistent-mapped pointer for `dynamic` mode; null otherwise.
    mapped: ?*anyopaque,

    /// Static buffer — DEVICE_LOCAL, populated via a one-shot staging
    /// copy. Typical caller is the SafeTensors loader handing over a
    /// slice of fp32 weights.
    pub fn initStatic(ctx: *const vk.Context, comptime T: type, data: []const T) !Buffer {
        const total = @max(data.len * @sizeOf(T), @sizeOf(T));
        const raw = try createBuffer(
            ctx,
            total,
            c.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | c.VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            c.VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
        );
        if (data.len > 0) {
            try stagingUpload(ctx, raw.handle, std.mem.sliceAsBytes(data));
        } else {
            try gpuFill(ctx, raw.handle, total);
        }
        return .{
            .handle = raw.handle,
            .memory = raw.memory,
            .bytes = total,
            .mode = .static,
            .mapped = null,
        };
    }

    /// Dynamic buffer — HOST_VISIBLE + HOST_COHERENT, persistent-mapped.
    /// Caller writes via `update()`. Read-only on the GPU side; writing
    /// from a shader through this binding is a synchronisation footgun.
    pub fn initDynamic(ctx: *const vk.Context, capacity_bytes: usize) !Buffer {
        const total = @max(capacity_bytes, 16);
        const raw = try createBuffer(
            ctx,
            total,
            c.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
            c.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | c.VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
        );
        var mapped: ?*anyopaque = null;
        try vk.check(c.vkMapMemory(ctx.device, raw.memory, 0, total, 0, &mapped));
        return .{
            .handle = raw.handle,
            .memory = raw.memory,
            .bytes = total,
            .mode = .dynamic,
            .mapped = mapped,
        };
    }

    /// Device-only buffer — DEVICE_LOCAL, zero-filled. TRANSFER_SRC and
    /// TRANSFER_DST are both enabled so the buffer can serve as either
    /// side of a vkCmdCopyBuffer (e.g. swap-buffer patterns) and so we
    /// can read it back to host for parity tests during development.
    pub fn initDeviceOnly(ctx: *const vk.Context, capacity_bytes: usize) !Buffer {
        const total = @max(capacity_bytes, 16);
        const raw = try createBuffer(
            ctx,
            total,
            c.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                c.VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                c.VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
            c.VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
        );
        try gpuFill(ctx, raw.handle, total);
        return .{
            .handle = raw.handle,
            .memory = raw.memory,
            .bytes = total,
            .mode = .device_only,
            .mapped = null,
        };
    }

    /// memcpy `data` into a `dynamic` buffer. No-op on other modes —
    /// silent rather than a crash so a misuse surfaces as stale data
    /// during testing rather than a corruption in production.
    pub fn update(self: *Buffer, comptime T: type, data: []const T) void {
        if (self.mode != .dynamic) return;
        const bytes = data.len * @sizeOf(T);
        if (bytes == 0 or bytes > self.bytes) return;
        const dst = self.mapped orelse return;
        @memcpy(@as([*]u8, @ptrCast(dst))[0..bytes], std.mem.sliceAsBytes(data));
    }

    /// Read a `device_only` buffer back to a host slice via a transient
    /// staging buffer. Init- or test-time only — far too slow for any
    /// hot loop. Intended for parity tests against the CPU reference.
    pub fn readBack(self: *const Buffer, ctx: *const vk.Context, comptime T: type, dst: []T) !void {
        const want_bytes = dst.len * @sizeOf(T);
        if (want_bytes == 0) return;
        if (want_bytes > self.bytes) return error.ReadBackTooLarge;

        const staging = try createBuffer(
            ctx,
            want_bytes,
            c.VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            c.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | c.VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
        );
        defer {
            c.vkDestroyBuffer(ctx.device, staging.handle, null);
            c.vkFreeMemory(ctx.device, staging.memory, null);
        }

        try submitOneShot(ctx, struct {
            src: c.VkBuffer,
            dst: c.VkBuffer,
            size: usize,
            pub fn record(s: @This(), cmd: c.VkCommandBuffer) void {
                const region = c.VkBufferCopy{
                    .srcOffset = 0,
                    .dstOffset = 0,
                    .size = @intCast(s.size),
                };
                c.vkCmdCopyBuffer(cmd, s.src, s.dst, 1, &region);
            }
        }{ .src = self.handle, .dst = staging.handle, .size = want_bytes });

        var mapped: ?*anyopaque = null;
        try vk.check(c.vkMapMemory(ctx.device, staging.memory, 0, want_bytes, 0, &mapped));
        defer c.vkUnmapMemory(ctx.device, staging.memory);
        @memcpy(std.mem.sliceAsBytes(dst), @as([*]u8, @ptrCast(mapped.?))[0..want_bytes]);
    }

    /// VkDescriptorBufferInfo for binding into a descriptor write.
    /// Range covers the full allocation; shaders index by element so
    /// extra capacity is harmless.
    pub fn descriptorInfo(self: *const Buffer) c.VkDescriptorBufferInfo {
        return .{
            .buffer = self.handle,
            .offset = 0,
            .range = @intCast(self.bytes),
        };
    }

    pub fn deinit(self: *Buffer, device: c.VkDevice) void {
        if (self.mapped != null) {
            c.vkUnmapMemory(device, self.memory);
            self.mapped = null;
        }
        c.vkDestroyBuffer(device, self.handle, null);
        c.vkFreeMemory(device, self.memory, null);
    }
};

// ── Internals ────────────────────────────────────────────────────────

const RawBuffer = struct {
    handle: c.VkBuffer,
    memory: c.VkDeviceMemory,
};

fn findMemoryType(pdev: c.VkPhysicalDevice, type_filter: u32, properties: u32) !u32 {
    var mem_props: c.VkPhysicalDeviceMemoryProperties = undefined;
    c.vkGetPhysicalDeviceMemoryProperties(pdev, &mem_props);
    for (0..mem_props.memoryTypeCount) |i| {
        const idx: u5 = @intCast(i);
        if ((type_filter & (@as(u32, 1) << idx)) != 0 and
            (mem_props.memoryTypes[i].propertyFlags & properties) == properties)
        {
            return @intCast(i);
        }
    }
    return error.NoSuitableMemoryType;
}

fn createBuffer(
    ctx: *const vk.Context,
    bytes: usize,
    usage: c.VkBufferUsageFlags,
    properties: c.VkMemoryPropertyFlags,
) !RawBuffer {
    var bci = std.mem.zeroes(c.VkBufferCreateInfo);
    bci.sType = c.VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bci.size = @intCast(bytes);
    bci.usage = usage;
    bci.sharingMode = c.VK_SHARING_MODE_EXCLUSIVE;

    var handle: c.VkBuffer = null;
    try vk.check(c.vkCreateBuffer(ctx.device, &bci, null, &handle));
    errdefer c.vkDestroyBuffer(ctx.device, handle, null);

    var req: c.VkMemoryRequirements = undefined;
    c.vkGetBufferMemoryRequirements(ctx.device, handle, &req);

    var mai = std.mem.zeroes(c.VkMemoryAllocateInfo);
    mai.sType = c.VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    mai.allocationSize = req.size;
    mai.memoryTypeIndex = try findMemoryType(ctx.physical_device, req.memoryTypeBits, properties);

    var memory: c.VkDeviceMemory = null;
    try vk.check(c.vkAllocateMemory(ctx.device, &mai, null, &memory));
    errdefer c.vkFreeMemory(ctx.device, memory, null);

    try vk.check(c.vkBindBufferMemory(ctx.device, handle, memory, 0));
    return .{ .handle = handle, .memory = memory };
}

/// One-shot blocking staging upload. Allocates a HOST_VISIBLE staging
/// buffer the size of `bytes`, memcpys in, records a copy into `dst`,
/// submits, waits idle, frees. Init-time only.
fn stagingUpload(ctx: *const vk.Context, dst: c.VkBuffer, bytes: []const u8) !void {
    const staging = try createBuffer(
        ctx,
        bytes.len,
        c.VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
        c.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | c.VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
    );
    defer {
        c.vkDestroyBuffer(ctx.device, staging.handle, null);
        c.vkFreeMemory(ctx.device, staging.memory, null);
    }

    var mapped: ?*anyopaque = null;
    try vk.check(c.vkMapMemory(ctx.device, staging.memory, 0, bytes.len, 0, &mapped));
    @memcpy(@as([*]u8, @ptrCast(mapped.?))[0..bytes.len], bytes);
    c.vkUnmapMemory(ctx.device, staging.memory);

    try submitOneShot(ctx, struct {
        src: c.VkBuffer,
        dst: c.VkBuffer,
        size: usize,
        pub fn record(s: @This(), cmd: c.VkCommandBuffer) void {
            const region = c.VkBufferCopy{
                .srcOffset = 0,
                .dstOffset = 0,
                .size = @intCast(s.size),
            };
            c.vkCmdCopyBuffer(cmd, s.src, s.dst, 1, &region);
        }
    }{ .src = staging.handle, .dst = dst, .size = bytes.len });
}

/// Zero-fill a device-local buffer via vkCmdFillBuffer. Cheaper than a
/// staging upload of zeros; covers placeholder + device_only paths.
fn gpuFill(ctx: *const vk.Context, dst: c.VkBuffer, bytes: usize) !void {
    try submitOneShot(ctx, struct {
        dst: c.VkBuffer,
        size: usize,
        pub fn record(s: @This(), cmd: c.VkCommandBuffer) void {
            c.vkCmdFillBuffer(cmd, s.dst, 0, @intCast(s.size), 0);
        }
    }{ .dst = dst, .size = bytes });
}

/// Allocate a one-shot primary command buffer from the context's pool,
/// invoke `recorder.record(cmd)`, submit, wait queue idle, free.
pub fn submitOneShot(ctx: *const vk.Context, recorder: anytype) !void {
    var cb_ai = std.mem.zeroes(c.VkCommandBufferAllocateInfo);
    cb_ai.sType = c.VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    cb_ai.commandPool = ctx.cmd_pool;
    cb_ai.level = c.VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cb_ai.commandBufferCount = 1;

    var cmd: c.VkCommandBuffer = null;
    try vk.check(c.vkAllocateCommandBuffers(ctx.device, &cb_ai, &cmd));
    defer c.vkFreeCommandBuffers(ctx.device, ctx.cmd_pool, 1, &cmd);

    var begin = std.mem.zeroes(c.VkCommandBufferBeginInfo);
    begin.sType = c.VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    begin.flags = c.VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    try vk.check(c.vkBeginCommandBuffer(cmd, &begin));
    recorder.record(cmd);
    try vk.check(c.vkEndCommandBuffer(cmd));

    var submit = std.mem.zeroes(c.VkSubmitInfo);
    submit.sType = c.VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submit.commandBufferCount = 1;
    submit.pCommandBuffers = &cmd;
    try vk.check(c.vkQueueSubmit(ctx.queue, 1, &submit, null));
    try vk.check(c.vkQueueWaitIdle(ctx.queue));
}
