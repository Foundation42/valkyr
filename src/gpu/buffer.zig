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

pub const Mode = enum { static, dynamic, device_only, host_readback };

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
        // TRANSFER_SRC_BIT lets us read static buffers back to host
        // for parity testing (round-trip verification of the bf16→fp32
        // upload). Cost is just a usage-flag bit; the buffer stays
        // device-local. Without it, validation layers reject the copy.
        const raw = try createBuffer(
            ctx,
            total,
            c.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                c.VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                c.VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
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

    /// Host-readback buffer — HOST_VISIBLE + HOST_COHERENT, persistent-
    /// mapped, with TRANSFER_DST_BIT enabled so the GPU can stream
    /// data INTO it via `vkCmdCopyBuffer` from a device-only buffer.
    /// The CPU then reads the mapped pointer directly — no per-frame
    /// staging dance, no submitOneShot. Used for the cooperative
    /// inference / training paths where the host's own submit + fence
    /// drives readback completion (CPU reads after fence wait).
    ///
    /// Pair with: a device-only output buffer + a recorded
    /// `vkCmdCopyBuffer` from output → this. After the host's frame
    /// fence signals, contents of `mapped` are guaranteed visible.
    pub fn initHostReadback(ctx: *const vk.Context, capacity_bytes: usize) !Buffer {
        const total = @max(capacity_bytes, 16);
        const raw = try createBuffer(
            ctx,
            total,
            c.VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            c.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | c.VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
        );
        var mapped: ?*anyopaque = null;
        try vk.check(c.vkMapMemory(ctx.device, raw.memory, 0, total, 0, &mapped));
        // Vulkan doesn't guarantee zeroed memory at allocation; without
        // this CPU readers see garbage on the first frame before the
        // first GPU copy lands.
        if (mapped) |m| {
            @memset(@as([*]u8, @ptrCast(m))[0..total], 0);
        }
        return .{
            .handle = raw.handle,
            .memory = raw.memory,
            .bytes = total,
            .mode = .host_readback,
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

    /// Fill the entire buffer with zero. One-shot submission, blocks
    /// until the queue is idle. Intended for resetting per-session
    /// state (e.g. the SSM recurrent buffers between independent
    /// prompts in a `--prompts` batch run); not appropriate for any
    /// hot path.
    pub fn fillZero(self: *const Buffer, ctx: *const vk.Context) !void {
        if (self.bytes == 0) return;
        try submitOneShot(ctx, struct {
            buf: c.VkBuffer,
            size: usize,
            pub fn record(s: @This(), cmd: c.VkCommandBuffer) void {
                c.vkCmdFillBuffer(cmd, s.buf, 0, @intCast(s.size), 0);
            }
        }{ .buf = self.handle, .size = self.bytes });
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
        // Pool views set memory = null; per Vulkan spec vkFreeMemory(VK_NULL_HANDLE)
        // is a no-op, so this works for both owning and non-owning Buffers.
        c.vkFreeMemory(device, self.memory, null);
    }
};

// ── Pooled weight uploads ────────────────────────────────────────────
//
// `BufferPool` suballocates many weight tensors out of a single
// VkDeviceMemory and stages all of their data through a single
// persistent host-visible staging buffer. Per-tensor work shrinks to
//
//   - one cheap vkCreateBuffer + vkBindBufferMemory (~10 µs)
//   - one memcpy into the staging mapping
//   - one vkCmdCopyBuffer recorded into a shared command buffer
//
// vs the per-tensor cycle of vkAllocateMemory (×2: tensor + staging),
// vkCreateBuffer (×2), submitOneShot + vkQueueWaitIdle the original
// code paid. Across the ~700 tensors in Qwen3.6 27B, the per-tensor
// queue-wait was the dominant cost.
//
// Lifecycle:
//   init(device_budget, staging_capacity)  — allocates one
//     DEVICE_LOCAL VkDeviceMemory of `device_budget` bytes plus one
//     HOST_VISIBLE staging buffer of `staging_capacity` bytes,
//     persistent-mapped, and begins recording into a fresh cmd buffer.
//   commit(bytes)                          — copies `bytes` into
//     staging, creates a VkBuffer bound at the next aligned slot of
//     the device pool, records a copy. Flushes mid-stream if staging
//     would overflow.
//   finalize()                             — submits the final batch,
//     frees the staging buffer + cmd buffer. Device memory survives.
//   deinit()                               — frees the device memory.
pub const BufferPool = struct {
    device_memory: c.VkDeviceMemory,
    device_capacity: usize,
    device_offset: usize,
    /// Alignment requirement reported for storage buffers on this
    /// device. Reused for every tensor's bind offset.
    bind_alignment: usize,

    staging_handle: c.VkBuffer,
    staging_memory: c.VkDeviceMemory,
    staging_mapped: [*]u8,
    staging_capacity: usize,
    staging_offset: usize,

    cmd: c.VkCommandBuffer,
    pending_copies: usize,

    pub fn init(ctx: *const vk.Context, device_budget: usize, staging_capacity: usize) !BufferPool {
        // Probe a minimal storage buffer to learn the device's required
        // memoryType + alignment for our usage flags. Keep the probe
        // VkBuffer alive only long enough to read its memory reqs.
        const usage: c.VkBufferUsageFlags =
            c.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
            c.VK_BUFFER_USAGE_TRANSFER_DST_BIT |
            c.VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
        var probe_bci = std.mem.zeroes(c.VkBufferCreateInfo);
        probe_bci.sType = c.VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        probe_bci.size = 1024;
        probe_bci.usage = usage;
        probe_bci.sharingMode = c.VK_SHARING_MODE_EXCLUSIVE;
        var probe_handle: c.VkBuffer = null;
        try vk.check(c.vkCreateBuffer(ctx.device, &probe_bci, null, &probe_handle));
        defer c.vkDestroyBuffer(ctx.device, probe_handle, null);
        var probe_req: c.VkMemoryRequirements = undefined;
        c.vkGetBufferMemoryRequirements(ctx.device, probe_handle, &probe_req);

        const align_val: usize = @max(@as(usize, 256), probe_req.alignment);
        const padded_total = (device_budget + align_val - 1) & ~(align_val - 1);

        var dmai = std.mem.zeroes(c.VkMemoryAllocateInfo);
        dmai.sType = c.VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        dmai.allocationSize = @intCast(padded_total);
        dmai.memoryTypeIndex = try findMemoryType(ctx.physical_device, probe_req.memoryTypeBits, c.VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        var device_memory: c.VkDeviceMemory = null;
        try vk.check(c.vkAllocateMemory(ctx.device, &dmai, null, &device_memory));
        errdefer c.vkFreeMemory(ctx.device, device_memory, null);

        // One persistent host-visible staging buffer, mapped for the
        // duration of upload. Sized by caller to fit the largest single
        // tensor plus some flush headroom.
        const staging = try createBuffer(
            ctx,
            staging_capacity,
            c.VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
            c.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | c.VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
        );
        errdefer {
            c.vkDestroyBuffer(ctx.device, staging.handle, null);
            c.vkFreeMemory(ctx.device, staging.memory, null);
        }
        var mapped: ?*anyopaque = null;
        try vk.check(c.vkMapMemory(ctx.device, staging.memory, 0, staging_capacity, 0, &mapped));

        // Persistent command buffer for batched copies. Begun in
        // ONE_TIME_SUBMIT mode and re-begun after each flush.
        var cb_ai = std.mem.zeroes(c.VkCommandBufferAllocateInfo);
        cb_ai.sType = c.VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        cb_ai.commandPool = ctx.cmd_pool;
        cb_ai.level = c.VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        cb_ai.commandBufferCount = 1;
        var cmd: c.VkCommandBuffer = null;
        try vk.check(c.vkAllocateCommandBuffers(ctx.device, &cb_ai, &cmd));
        var begin = std.mem.zeroes(c.VkCommandBufferBeginInfo);
        begin.sType = c.VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        begin.flags = c.VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        try vk.check(c.vkBeginCommandBuffer(cmd, &begin));

        return .{
            .device_memory = device_memory,
            .device_capacity = padded_total,
            .device_offset = 0,
            .bind_alignment = align_val,
            .staging_handle = staging.handle,
            .staging_memory = staging.memory,
            .staging_mapped = @ptrCast(mapped.?),
            .staging_capacity = staging_capacity,
            .staging_offset = 0,
            .cmd = cmd,
            .pending_copies = 0,
        };
    }

    /// Stage `bytes` and bind a fresh VkBuffer at the next aligned
    /// device slot. Returns a non-owning Buffer view (memory = null,
    /// safe to deinit — only the VkBuffer handle is destroyed).
    pub fn commit(self: *BufferPool, ctx: *const vk.Context, bytes: []const u8) !Buffer {
        if (bytes.len == 0) return error.EmptyTensor;
        if (bytes.len > self.staging_capacity) return error.TensorLargerThanStaging;

        // Flush if staging would overflow.
        if (self.staging_offset + bytes.len > self.staging_capacity) {
            try self.flush(ctx);
        }

        // Align device offset to bind_alignment.
        const align_val = self.bind_alignment;
        const dev_off = (self.device_offset + align_val - 1) & ~(align_val - 1);
        if (dev_off + bytes.len > self.device_capacity) return error.PoolDeviceMemoryExhausted;

        var bci = std.mem.zeroes(c.VkBufferCreateInfo);
        bci.sType = c.VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        bci.size = @intCast(bytes.len);
        bci.usage = c.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
            c.VK_BUFFER_USAGE_TRANSFER_DST_BIT |
            c.VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
        bci.sharingMode = c.VK_SHARING_MODE_EXCLUSIVE;
        var handle: c.VkBuffer = null;
        try vk.check(c.vkCreateBuffer(ctx.device, &bci, null, &handle));
        errdefer c.vkDestroyBuffer(ctx.device, handle, null);
        try vk.check(c.vkBindBufferMemory(ctx.device, handle, self.device_memory, @intCast(dev_off)));

        @memcpy(self.staging_mapped[self.staging_offset .. self.staging_offset + bytes.len], bytes);

        const region = c.VkBufferCopy{
            .srcOffset = @intCast(self.staging_offset),
            .dstOffset = 0,
            .size = @intCast(bytes.len),
        };
        c.vkCmdCopyBuffer(self.cmd, self.staging_handle, handle, 1, &region);

        self.staging_offset += bytes.len;
        self.device_offset = dev_off + bytes.len;
        self.pending_copies += 1;

        return .{
            .handle = handle,
            .memory = null, // pool owns the memory; deinit is a no-op for it
            .bytes = bytes.len,
            .mode = .static,
            .mapped = null,
        };
    }

    /// Submit all pending copies, wait for the GPU, reset for more.
    pub fn flush(self: *BufferPool, ctx: *const vk.Context) !void {
        if (self.pending_copies == 0) return;
        try vk.check(c.vkEndCommandBuffer(self.cmd));
        var submit = std.mem.zeroes(c.VkSubmitInfo);
        submit.sType = c.VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submit.commandBufferCount = 1;
        submit.pCommandBuffers = &self.cmd;
        try vk.check(c.vkQueueSubmit(ctx.queue, 1, &submit, null));
        try vk.check(c.vkQueueWaitIdle(ctx.queue));
        try vk.check(c.vkResetCommandBuffer(self.cmd, 0));
        var begin = std.mem.zeroes(c.VkCommandBufferBeginInfo);
        begin.sType = c.VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        begin.flags = c.VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        try vk.check(c.vkBeginCommandBuffer(self.cmd, &begin));
        self.pending_copies = 0;
        self.staging_offset = 0;
    }

    /// Submit the final batch and tear down the staging buffer + cmd
    /// buffer. The device memory survives until `deinit` so weight
    /// reads during inference stay valid.
    pub fn finalize(self: *BufferPool, ctx: *const vk.Context) !void {
        try self.flush(ctx);
        // We left `cmd` in a re-begin state inside flush(); end it
        // cleanly before freeing.
        try vk.check(c.vkEndCommandBuffer(self.cmd));
        c.vkFreeCommandBuffers(ctx.device, ctx.cmd_pool, 1, &self.cmd);
        c.vkUnmapMemory(ctx.device, self.staging_memory);
        c.vkDestroyBuffer(ctx.device, self.staging_handle, null);
        c.vkFreeMemory(ctx.device, self.staging_memory, null);
    }

    pub fn deinit(self: *BufferPool, device: c.VkDevice) void {
        c.vkFreeMemory(device, self.device_memory, null);
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
