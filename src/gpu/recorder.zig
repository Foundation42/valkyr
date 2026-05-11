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
//!
//! Per-dispatch GPU timing: opt-in via `enableTimestamps(max)`. When
//! enabled, every dispatch is wrapped with `vkCmdWriteTimestamp` at
//! TOP_OF_PIPE (start) and BOTTOM_OF_PIPE (end). After
//! `endAndSubmit` (or, in embedded mode, after the host's submit
//! fence has signalled), `lastTimestamps(out)` reads back raw u64
//! ticks as `[start_0, end_0, start_1, end_1, ...]`. Multiply by
//! `nsPerTick()` to get nanoseconds. The TOP/BOTTOM placement is
//! deliberate: `end[i] - start[i]` is pure dispatch wall time, while
//! `start[i+1] - end[i]` exposes the barrier/sync gap between
//! dispatches — which is the question that motivated this surface
//! (fused vs. unfused chains where the GPU looks underutilised).

const std = @import("std");
const vk = @import("vk.zig");
const buffer = @import("buffer.zig");
const pipeline = @import("pipeline.zig");
const c = vk.c;

pub const Recorder = struct {
    ctx: *const vk.Context,
    pool: c.VkDescriptorPool,
    cmd: c.VkCommandBuffer,
    /// `null` when running in embedded mode — the host owns the fence
    /// and the submit, so the recorder has no fence of its own to
    /// destroy or wait on.
    fence: c.VkFence,
    n_dispatched: u32,

    // Ownership flags. Mirrors the pattern in `vk.Context`: in standalone
    // mode (`init`) the recorder owns its cmd buffer + fence and
    // begin/endAndSubmit drive the full lifecycle. In embedded mode
    // (`attachCmd`, e.g. valkyr inside Matryoshka's drawFrame) the host
    // already called `vkBeginCommandBuffer` and will submit with its
    // own fence as part of its render submit; valkyr just records
    // dispatches into the host's buffer and stays out of the way.
    // The descriptor pool is always recorder-owned regardless of mode —
    // its lifetime is dispatch-graph-scoped, not frame-scoped.
    owns_cmd: bool,
    owns_fence: bool,

    /// Optional GPU timestamp pool. `null` until `enableTimestamps`
    /// is called; when non-null, each dispatch is wrapped with
    /// TOP_OF_PIPE / BOTTOM_OF_PIPE timestamps at indices
    /// `2*n_dispatched` and `2*n_dispatched+1`.
    query_pool: c.VkQueryPool,
    ts_capacity: u32,
    ts_period_ns: f32,
    /// Mask applied to each raw query tick before use — Vulkan
    /// guarantees only `timestampValidBits` bits of valid data per
    /// queue family, the rest are undefined.
    ts_valid_mask: u64,

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
            .owns_cmd = true,
            .owns_fence = true,
            .query_pool = null,
            .ts_capacity = 0,
            .ts_period_ns = 0,
            .ts_valid_mask = 0,
        };
    }

    /// Embedded-mode constructor. The host (e.g. Matryoshka) has
    /// already allocated a command buffer and called
    /// `vkBeginCommandBuffer` on it; valkyr's recorder simply records
    /// its dispatches into that buffer and lets the host submit + wait.
    ///
    /// `host_cmd` MUST be in the recording state when the first
    /// dispatch fires, and MUST NOT be ended/submitted by the host
    /// until after the final dispatch through this recorder. The host
    /// is also responsible for any `vkCmdPipelineBarrier` that crosses
    /// the boundary between its own dispatches and valkyr's (the
    /// recorder still inserts barriers between *its own* dispatches).
    pub fn attachCmd(
        ctx: *const vk.Context,
        host_cmd: c.VkCommandBuffer,
        max_sets: u32,
        max_descriptors: u32,
    ) !Recorder {
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

        return .{
            .ctx = ctx,
            .pool = pool,
            .cmd = host_cmd,
            .fence = null,
            .n_dispatched = 0,
            .owns_cmd = false,
            .owns_fence = false,
            .query_pool = null,
            .ts_capacity = 0,
            .ts_period_ns = 0,
            .ts_valid_mask = 0,
        };
    }

    /// Enable per-dispatch GPU timestamp capture. Must be called
    /// BEFORE `begin()` — the query pool is reset on each
    /// `begin()`, but newly-created queries are in an undefined
    /// state until first reset, so timestamps recorded between
    /// `enableTimestamps` and the first `begin()` would be garbage.
    ///
    /// `max_dispatches` is the cap for a single recording cycle
    /// (one begin → endAndSubmit). The pool holds `2 * max` queries.
    /// Returns `error.TimestampsUnsupported` when the queue family
    /// reports `timestampValidBits == 0`. Can fail with
    /// `error.AlreadyEnabled` if called twice.
    pub fn enableTimestamps(self: *Recorder, max_dispatches: u32) !void {
        if (self.query_pool != null) return error.AlreadyEnabled;
        if (max_dispatches == 0) return error.InvalidArgument;

        // Look up our queue family's timestampValidBits. The Vulkan
        // spec splits this between physical-device props
        // (timestampPeriod, ns/tick) and queue-family props
        // (validBits, per-queue support). We cache the period on
        // ctx already; the validBits we re-query here.
        var qf_count: u32 = 0;
        c.vkGetPhysicalDeviceQueueFamilyProperties(self.ctx.physical_device, &qf_count, null);
        if (qf_count == 0 or self.ctx.queue_family >= qf_count) return error.NoComputeQueue;
        var qfs: [16]c.VkQueueFamilyProperties = undefined;
        const cap = @min(qf_count, qfs.len);
        qf_count = cap;
        c.vkGetPhysicalDeviceQueueFamilyProperties(self.ctx.physical_device, &qf_count, &qfs);
        if (self.ctx.queue_family >= qf_count) return error.NoComputeQueue;
        const valid_bits = qfs[self.ctx.queue_family].timestampValidBits;
        if (valid_bits == 0) return error.TimestampsUnsupported;

        var qpci = std.mem.zeroes(c.VkQueryPoolCreateInfo);
        qpci.sType = c.VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO;
        qpci.queryType = c.VK_QUERY_TYPE_TIMESTAMP;
        qpci.queryCount = 2 * max_dispatches;
        var qp: c.VkQueryPool = null;
        try vk.check(c.vkCreateQueryPool(self.ctx.device, &qpci, null, &qp));

        self.query_pool = qp;
        self.ts_capacity = max_dispatches;
        self.ts_period_ns = self.ctx.props.limits.timestampPeriod;
        self.ts_valid_mask = if (valid_bits >= 64)
            ~@as(u64, 0)
        else
            (@as(u64, 1) << @intCast(valid_bits)) - 1;
    }

    /// Nanoseconds per timestamp tick on the active device. Multiply
    /// raw deltas from `lastTimestamps` by this value to convert.
    pub fn nsPerTick(self: *const Recorder) f32 {
        return self.ts_period_ns;
    }

    /// Read back the timestamps written during the most recent
    /// recording cycle. Caller must have already waited on the
    /// submission to complete (standalone `endAndSubmit` does this
    /// for you; embedded callers must wait on their host fence first).
    /// `out` must have length ≥ `2 * n_dispatched`. Returns the
    /// filled slice in `[start_0, end_0, start_1, end_1, ...]`
    /// order, with high garbage bits already masked off.
    pub fn lastTimestamps(self: *const Recorder, out: []u64) ![]u64 {
        if (self.query_pool == null) return error.TimestampsNotEnabled;
        const n_queries: u32 = 2 * self.n_dispatched;
        if (n_queries == 0) return out[0..0];
        if (out.len < n_queries) return error.OutputTooSmall;

        try vk.check(c.vkGetQueryPoolResults(
            self.ctx.device,
            self.query_pool,
            0,
            n_queries,
            n_queries * @sizeOf(u64),
            out.ptr,
            @sizeOf(u64),
            c.VK_QUERY_RESULT_64_BIT | c.VK_QUERY_RESULT_WAIT_BIT,
        ));

        const slice = out[0..n_queries];
        for (slice) |*v| v.* &= self.ts_valid_mask;
        return slice;
    }

    pub fn deinit(self: *Recorder) void {
        if (self.query_pool != null) c.vkDestroyQueryPool(self.ctx.device, self.query_pool, null);
        if (self.owns_fence) c.vkDestroyFence(self.ctx.device, self.fence, null);
        if (self.owns_cmd) c.vkFreeCommandBuffers(self.ctx.device, self.ctx.cmd_pool, 1, &self.cmd);
        c.vkDestroyDescriptorPool(self.ctx.device, self.pool, null);
    }

    /// Reset the descriptor pool and command buffer for re-use across
    /// forward passes (e.g. multi-token generation). The fence is reset
    /// inside `submit()`. In embedded mode the host owns the cmd-buffer
    /// reset cadence (typically per render frame), so we only reset
    /// the descriptor pool here.
    pub fn reset(self: *Recorder) !void {
        try vk.check(c.vkResetDescriptorPool(self.ctx.device, self.pool, 0));
        if (self.owns_cmd) try vk.check(c.vkResetCommandBuffer(self.cmd, 0));
        self.n_dispatched = 0;
    }

    /// In standalone mode, calls `vkBeginCommandBuffer` on the
    /// recorder's own buffer. In embedded mode this is a no-op — the
    /// host already put its buffer in the recording state before
    /// handing it over.
    pub fn begin(self: *Recorder) !void {
        if (!self.owns_cmd) {
            self.n_dispatched = 0;
            // Embedded mode: host has already begun the cmd buffer.
            // We can issue the query reset right here.
            if (self.query_pool != null) {
                c.vkCmdResetQueryPool(self.cmd, self.query_pool, 0, 2 * self.ts_capacity);
            }
            return;
        }
        var bi = std.mem.zeroes(c.VkCommandBufferBeginInfo);
        bi.sType = c.VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        bi.flags = c.VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        try vk.check(c.vkBeginCommandBuffer(self.cmd, &bi));
        if (self.query_pool != null) {
            c.vkCmdResetQueryPool(self.cmd, self.query_pool, 0, 2 * self.ts_capacity);
        }
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
        if (self.query_pool != null and self.n_dispatched >= self.ts_capacity)
            return error.TimestampPoolExhausted;

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
        // Pure dispatch wall time = end - start. The TOP_OF_PIPE
        // start fires after the preceding barrier completes (so it
        // marks "begin executing this dispatch") and the
        // BOTTOM_OF_PIPE end fires once all writes have finished.
        // Gap between consecutive end[i] and start[i+1] reflects
        // barrier/sync wait — see the module docstring.
        if (self.query_pool != null) {
            c.vkCmdWriteTimestamp(
                self.cmd,
                c.VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                self.query_pool,
                2 * self.n_dispatched,
            );
        }
        c.vkCmdDispatch(self.cmd, gx, gy, gz);
        if (self.query_pool != null) {
            c.vkCmdWriteTimestamp(
                self.cmd,
                c.VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
                self.query_pool,
                2 * self.n_dispatched + 1,
            );
        }
        self.n_dispatched += 1;
    }

    /// End recording, submit to the queue, and wait for the fence.
    /// After this returns, all GPU writes are visible and any output
    /// buffers can be safely read back.
    ///
    /// Not callable in embedded mode — the host owns the submit cadence
    /// (a single render submit per frame) and the fence (typically one
    /// fence per frame-in-flight). Embedded callers should let the host
    /// `vkEndCommandBuffer` + `vkQueueSubmit` as part of its render
    /// frame; the recorder just contributed dispatches to the buffer.
    pub fn endAndSubmit(self: *Recorder) !void {
        if (!self.owns_cmd) return error.AttachedRecorderCannotSubmit;
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
