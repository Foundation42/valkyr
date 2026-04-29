//! Compute kernel: SPIR-V module + descriptor set + pipeline.
//!
//! One `Kernel` per shader. Each kernel owns its own descriptor pool
//! sized for a single set — that single set is reused across dispatches
//! and rewritten as inputs change. This keeps the per-kernel ownership
//! tidy at the cost of one extra pool object per kernel; we have on the
//! order of ~10 kernels for the whole transformer, so the cost is in
//! the noise.
//!
//! All bindings are storage buffers (set=0). Push constants are a
//! contiguous byte range starting at offset 0, sized by `push_bytes`
//! at construction. Both choices match the shaders we'll write —
//! every transformer kernel reads a few SSBOs and a small struct of
//! shape/scale params that fits in 128 bytes.

const std = @import("std");
const vk = @import("vk.zig");
const buffer = @import("buffer.zig");
const c = vk.c;

pub const Kernel = struct {
    device: c.VkDevice,
    set_layout: c.VkDescriptorSetLayout,
    pipeline_layout: c.VkPipelineLayout,
    pipeline: c.VkPipeline,
    pool: c.VkDescriptorPool,
    set: c.VkDescriptorSet,
    binding_count: u32,
    push_bytes: u32,

    /// Build a kernel from embedded SPIR-V. `binding_count` is the number
    /// of storage buffer bindings the shader declares at set=0. SPIR-V
    /// must be 4-byte aligned (cf. align(4) on @embedFile in build.zig).
    pub fn init(
        ctx: *const vk.Context,
        spirv: []const u8,
        binding_count: u32,
        push_bytes: u32,
    ) !Kernel {
        // ── Descriptor set layout ───────────────────────────────────
        var bindings_buf: [16]c.VkDescriptorSetLayoutBinding = undefined;
        if (binding_count > bindings_buf.len) return error.TooManyBindings;
        for (0..binding_count) |i| {
            bindings_buf[i] = .{
                .binding = @intCast(i),
                .descriptorType = c.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                .descriptorCount = 1,
                .stageFlags = c.VK_SHADER_STAGE_COMPUTE_BIT,
                .pImmutableSamplers = null,
            };
        }
        var dslci = std.mem.zeroes(c.VkDescriptorSetLayoutCreateInfo);
        dslci.sType = c.VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        dslci.bindingCount = binding_count;
        dslci.pBindings = &bindings_buf;
        var set_layout: c.VkDescriptorSetLayout = null;
        try vk.check(c.vkCreateDescriptorSetLayout(ctx.device, &dslci, null, &set_layout));
        errdefer c.vkDestroyDescriptorSetLayout(ctx.device, set_layout, null);

        // ── Pipeline layout (with push constants) ───────────────────
        var pcr = c.VkPushConstantRange{
            .stageFlags = c.VK_SHADER_STAGE_COMPUTE_BIT,
            .offset = 0,
            .size = push_bytes,
        };
        var plci = std.mem.zeroes(c.VkPipelineLayoutCreateInfo);
        plci.sType = c.VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        plci.setLayoutCount = 1;
        plci.pSetLayouts = &set_layout;
        if (push_bytes > 0) {
            plci.pushConstantRangeCount = 1;
            plci.pPushConstantRanges = &pcr;
        }
        var pipeline_layout: c.VkPipelineLayout = null;
        try vk.check(c.vkCreatePipelineLayout(ctx.device, &plci, null, &pipeline_layout));
        errdefer c.vkDestroyPipelineLayout(ctx.device, pipeline_layout, null);

        // ── Shader module ───────────────────────────────────────────
        // Vulkan wants pCode as `const u32 *` and codeSize in bytes. The
        // SPIR-V slice we're handed is `[]const u8`; the align(4) on the
        // @embedFile up the chain is what makes the pointer cast safe.
        var smci = std.mem.zeroes(c.VkShaderModuleCreateInfo);
        smci.sType = c.VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
        smci.codeSize = spirv.len;
        smci.pCode = @alignCast(@ptrCast(spirv.ptr));
        var shader_module: c.VkShaderModule = null;
        try vk.check(c.vkCreateShaderModule(ctx.device, &smci, null, &shader_module));
        defer c.vkDestroyShaderModule(ctx.device, shader_module, null);

        // ── Compute pipeline ────────────────────────────────────────
        var stage = std.mem.zeroes(c.VkPipelineShaderStageCreateInfo);
        stage.sType = c.VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        stage.stage = c.VK_SHADER_STAGE_COMPUTE_BIT;
        stage.module = shader_module;
        stage.pName = "main";

        var cpci = std.mem.zeroes(c.VkComputePipelineCreateInfo);
        cpci.sType = c.VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
        cpci.stage = stage;
        cpci.layout = pipeline_layout;
        var pipeline: c.VkPipeline = null;
        try vk.check(c.vkCreateComputePipelines(ctx.device, null, 1, &cpci, null, &pipeline));
        errdefer c.vkDestroyPipeline(ctx.device, pipeline, null);

        // ── Descriptor pool + set ───────────────────────────────────
        var pool_size = c.VkDescriptorPoolSize{
            .type = c.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .descriptorCount = binding_count,
        };
        var dpci = std.mem.zeroes(c.VkDescriptorPoolCreateInfo);
        dpci.sType = c.VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        dpci.maxSets = 1;
        dpci.poolSizeCount = 1;
        dpci.pPoolSizes = &pool_size;
        var pool: c.VkDescriptorPool = null;
        try vk.check(c.vkCreateDescriptorPool(ctx.device, &dpci, null, &pool));
        errdefer c.vkDestroyDescriptorPool(ctx.device, pool, null);

        var dsai = std.mem.zeroes(c.VkDescriptorSetAllocateInfo);
        dsai.sType = c.VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        dsai.descriptorPool = pool;
        dsai.descriptorSetCount = 1;
        dsai.pSetLayouts = &set_layout;
        var set: c.VkDescriptorSet = null;
        try vk.check(c.vkAllocateDescriptorSets(ctx.device, &dsai, &set));

        return .{
            .device = ctx.device,
            .set_layout = set_layout,
            .pipeline_layout = pipeline_layout,
            .pipeline = pipeline,
            .pool = pool,
            .set = set,
            .binding_count = binding_count,
            .push_bytes = push_bytes,
        };
    }

    /// Bind buffers to the kernel's single descriptor set in declaration
    /// order. `buffers[i]` becomes binding `i`. Safe to call multiple
    /// times — each call overwrites the previous binding.
    pub fn bind(self: *Kernel, buffers: []const *const buffer.Buffer) !void {
        if (buffers.len != self.binding_count) return error.BindingCountMismatch;

        var infos: [16]c.VkDescriptorBufferInfo = undefined;
        var writes: [16]c.VkWriteDescriptorSet = undefined;
        for (buffers, 0..) |buf, i| {
            infos[i] = buf.descriptorInfo();
            writes[i] = std.mem.zeroes(c.VkWriteDescriptorSet);
            writes[i].sType = c.VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writes[i].dstSet = self.set;
            writes[i].dstBinding = @intCast(i);
            writes[i].descriptorCount = 1;
            writes[i].descriptorType = c.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            writes[i].pBufferInfo = &infos[i];
        }
        c.vkUpdateDescriptorSets(self.device, @intCast(buffers.len), &writes, 0, null);
    }

    /// Record bind + push + dispatch into an already-begun command buffer.
    /// `push` is a pointer to a struct (or null) sized exactly `push_bytes`.
    /// The caller is responsible for inserting any pipeline barriers
    /// required between dispatches that share buffers.
    pub fn dispatch(
        self: *const Kernel,
        cmd: c.VkCommandBuffer,
        push: ?*const anyopaque,
        groups_x: u32,
        groups_y: u32,
        groups_z: u32,
    ) void {
        c.vkCmdBindPipeline(cmd, c.VK_PIPELINE_BIND_POINT_COMPUTE, self.pipeline);
        c.vkCmdBindDescriptorSets(
            cmd,
            c.VK_PIPELINE_BIND_POINT_COMPUTE,
            self.pipeline_layout,
            0,
            1,
            &self.set,
            0,
            null,
        );
        if (self.push_bytes > 0 and push != null) {
            c.vkCmdPushConstants(
                cmd,
                self.pipeline_layout,
                c.VK_SHADER_STAGE_COMPUTE_BIT,
                0,
                self.push_bytes,
                push,
            );
        }
        c.vkCmdDispatch(cmd, groups_x, groups_y, groups_z);
    }

    pub fn deinit(self: *Kernel) void {
        c.vkDestroyDescriptorPool(self.device, self.pool, null);
        c.vkDestroyPipeline(self.device, self.pipeline, null);
        c.vkDestroyPipelineLayout(self.device, self.pipeline_layout, null);
        c.vkDestroyDescriptorSetLayout(self.device, self.set_layout, null);
    }
};
