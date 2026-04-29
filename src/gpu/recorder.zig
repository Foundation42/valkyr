//! Command-buffer batcher for the full forward pass.
//!
//! Up to now every dispatch was its own queue submission with a
//! `vkQueueWaitIdle` after — fine for individual smoke tests, but at
//! ~291 dispatches per forward pass the per-submit overhead dominated
//! the wall clock. This module records all dispatches into one
//! command buffer, separated by global compute-shader memory barriers,
//! and submits once at the end.
//!
//! The subtle bit is descriptor sets. A descriptor set is essentially
//! a pointer the GPU dereferences at dispatch time; if we re-use one
//! set across multiple recorded dispatches and update it between them,
//! all dispatches see the *latest* values when they execute, which is
//! a use-after-write hazard. So we allocate ONE descriptor set per
//! dispatch from a pre-sized pool. With ~290 dispatches per forward
//! and ~3 storage-buffer descriptors each, we need on the order of
//! 512 sets and 2048 descriptors — both well under any GPU limit.
//!
//! Memory barrier strategy: a single global `VkMemoryBarrier` (no
//! buffer specified) between every two consecutive dispatches, with
//! SHADER_WRITE → SHADER_READ_WRITE access. Conservatively serialises
//! everything; we don't try to expose dispatch-level concurrency
//! because consecutive dispatches in a transformer almost always
//! depend on each other (residual stream, KV cache, etc.).

const std = @import("std");
const vk = @import("vk.zig");
const buffer = @import("buffer.zig");
const pipeline = @import("pipeline.zig");
const c = vk.c;

pub const Recorder = struct {
    ctx: *const vk.Context,
    pool: c.VkDescriptorPool,
    cmd: c.VkCommandBuffer,
    fence: c.VkFence,
    n_dispatched: u32,

    /// Build a recorder sized for a forward pass of up to `max_sets`
    /// dispatches and `max_descriptors` total storage-buffer bindings.
    /// For Gemma 2B at single position we currently need 291 sets and
    /// ~728 descriptors; the defaults round up generously.
    pub fn init(ctx: *const vk.Context, max_sets: u32, max_descriptors: u32) !Recorder {
        // ── Descriptor pool ─────────────────────────────────────────
        var pool_size = c.VkDescriptorPoolSize{
            .type = c.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .descriptorCount = max_descriptors,
        };
        var dpci = std.mem.zeroes(c.VkDescriptorPoolCreateInfo);
        dpci.sType = c.VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        dpci.maxSets = max_sets;
        dpci.poolSizeCount = 1;
        dpci.pPoolSizes = &pool_size;
        var pool: c.VkDescriptorPool = null;
        try vk.check(c.vkCreateDescriptorPool(ctx.device, &dpci, null, &pool));
        errdefer c.vkDestroyDescriptorPool(ctx.device, pool, null);

        // ── Command buffer ──────────────────────────────────────────
        var cb_ai = std.mem.zeroes(c.VkCommandBufferAllocateInfo);
        cb_ai.sType = c.VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        cb_ai.commandPool = ctx.cmd_pool;
        cb_ai.level = c.VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        cb_ai.commandBufferCount = 1;
        var cmd: c.VkCommandBuffer = null;
        try vk.check(c.vkAllocateCommandBuffers(ctx.device, &cb_ai, &cmd));
        errdefer c.vkFreeCommandBuffers(ctx.device, ctx.cmd_pool, 1, &cmd);

        // ── Fence (for waiting on submission completion) ────────────
        var fci = std.mem.zeroes(c.VkFenceCreateInfo);
        fci.sType = c.VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        var fence: c.VkFence = null;
        try vk.check(c.vkCreateFence(ctx.device, &fci, null, &fence));

        return .{
            .ctx = ctx,
            .pool = pool,
            .cmd = cmd,
            .fence = fence,
            .n_dispatched = 0,
        };
    }

    pub fn deinit(self: *Recorder) void {
        c.vkDestroyFence(self.ctx.device, self.fence, null);
        c.vkFreeCommandBuffers(self.ctx.device, self.ctx.cmd_pool, 1, &self.cmd);
        c.vkDestroyDescriptorPool(self.ctx.device, self.pool, null);
    }

    /// Reset the descriptor pool and command buffer for re-use across
    /// forward passes (e.g. multi-token generation). The fence is reset
    /// inside `submit()`.
    pub fn reset(self: *Recorder) !void {
        try vk.check(c.vkResetDescriptorPool(self.ctx.device, self.pool, 0));
        try vk.check(c.vkResetCommandBuffer(self.cmd, 0));
        self.n_dispatched = 0;
    }

    pub fn begin(self: *Recorder) !void {
        var bi = std.mem.zeroes(c.VkCommandBufferBeginInfo);
        bi.sType = c.VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        bi.flags = c.VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        try vk.check(c.vkBeginCommandBuffer(self.cmd, &bi));
        self.n_dispatched = 0;
    }

    /// Record one kernel dispatch. Allocates a fresh descriptor set
    /// from the pool, fills it with the supplied buffers in declared
    /// binding order, and emits the bind / push / dispatch sequence.
    /// A global memory barrier is inserted before this dispatch when
    /// it isn't the first — that way the shader sees stores from the
    /// previous dispatch.
    pub fn dispatch(
        self: *Recorder,
        kern: *const pipeline.Kernel,
        buffers: []const *const buffer.Buffer,
        push: ?*const anyopaque,
        gx: u32,
        gy: u32,
        gz: u32,
    ) !void {
        if (buffers.len != kern.binding_count) return error.BindingCountMismatch;

        // ── Memory barrier between dispatches ───────────────────────
        if (self.n_dispatched > 0) {
            var mb = std.mem.zeroes(c.VkMemoryBarrier);
            mb.sType = c.VK_STRUCTURE_TYPE_MEMORY_BARRIER;
            mb.srcAccessMask = c.VK_ACCESS_SHADER_WRITE_BIT;
            mb.dstAccessMask = c.VK_ACCESS_SHADER_READ_BIT | c.VK_ACCESS_SHADER_WRITE_BIT;
            c.vkCmdPipelineBarrier(
                self.cmd,
                c.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                c.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                0,
                1,
                &mb,
                0,
                null,
                0,
                null,
            );
        }

        // ── Allocate + update descriptor set ────────────────────────
        var dsai = std.mem.zeroes(c.VkDescriptorSetAllocateInfo);
        dsai.sType = c.VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        dsai.descriptorPool = self.pool;
        dsai.descriptorSetCount = 1;
        dsai.pSetLayouts = &kern.set_layout;
        var set: c.VkDescriptorSet = null;
        try vk.check(c.vkAllocateDescriptorSets(self.ctx.device, &dsai, &set));

        var infos: [16]c.VkDescriptorBufferInfo = undefined;
        var writes: [16]c.VkWriteDescriptorSet = undefined;
        for (buffers, 0..) |buf, i| {
            infos[i] = buf.descriptorInfo();
            writes[i] = std.mem.zeroes(c.VkWriteDescriptorSet);
            writes[i].sType = c.VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writes[i].dstSet = set;
            writes[i].dstBinding = @intCast(i);
            writes[i].descriptorCount = 1;
            writes[i].descriptorType = c.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            writes[i].pBufferInfo = &infos[i];
        }
        c.vkUpdateDescriptorSets(self.ctx.device, @intCast(buffers.len), &writes, 0, null);

        // ── Record bind / push / dispatch ───────────────────────────
        c.vkCmdBindPipeline(self.cmd, c.VK_PIPELINE_BIND_POINT_COMPUTE, kern.pipeline);
        c.vkCmdBindDescriptorSets(
            self.cmd,
            c.VK_PIPELINE_BIND_POINT_COMPUTE,
            kern.pipeline_layout,
            0,
            1,
            &set,
            0,
            null,
        );
        if (kern.push_bytes > 0 and push != null) {
            c.vkCmdPushConstants(
                self.cmd,
                kern.pipeline_layout,
                c.VK_SHADER_STAGE_COMPUTE_BIT,
                0,
                kern.push_bytes,
                push,
            );
        }
        c.vkCmdDispatch(self.cmd, gx, gy, gz);
        self.n_dispatched += 1;
    }

    /// End recording, submit to the queue, and wait for the fence.
    /// After this returns, all GPU writes are visible and any output
    /// buffers can be safely read back.
    pub fn endAndSubmit(self: *Recorder) !void {
        try vk.check(c.vkEndCommandBuffer(self.cmd));
        try vk.check(c.vkResetFences(self.ctx.device, 1, &self.fence));

        var submit = std.mem.zeroes(c.VkSubmitInfo);
        submit.sType = c.VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submit.commandBufferCount = 1;
        submit.pCommandBuffers = &self.cmd;
        try vk.check(c.vkQueueSubmit(self.ctx.queue, 1, &submit, self.fence));

        // 10s timeout — comfortably above any realistic forward-pass
        // duration on Gemma 2B; if we hit it, something has gone
        // catastrophically wrong (driver hang, infinite loop in a
        // shader) and we want to surface the failure rather than wedge.
        const timeout_ns: u64 = 10 * 1_000_000_000;
        try vk.check(c.vkWaitForFences(self.ctx.device, 1, &self.fence, c.VK_TRUE, timeout_ns));
    }
};
