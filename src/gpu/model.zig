//! GPU-side mirror of the CPU `Model` struct.
//!
//! Holds one Buffer per weight tensor. Weights are uploaded once at
//! construction (bf16 on disk → fp32 on device for now) and kept
//! resident — the only buffers that change during inference are the
//! activations and KV cache, which are owned elsewhere.
//!
//! Memory cost: Gemma 2B's ~5 GiB bf16 weights become ~10 GiB fp32 on
//! the device. The RTX 3090's 24 GiB has plenty of headroom; we'll
//! revisit when we ship a bf16-aware kernel set. Host peak during
//! upload stays small — we materialise one tensor as fp32, upload it,
//! free the conversion buffer, repeat.
//!
//! Per-tensor VkBuffer/VkDeviceMemory pairs (164 of them for Gemma)
//! are not the long-term answer — a single big pool with sub-
//! allocations would amortise the per-allocation overhead — but it's
//! the cleanest path through phase 1. Optimise after correctness.

const std = @import("std");
const vk = @import("vk.zig");
const buffer = @import("buffer.zig");
const safetensors = @import("../safetensors.zig");
const config_mod = @import("../config.zig");
const dtype = @import("../dtype.zig");
const model_mod = @import("../model.zig");

pub const GpuLayer = struct {
    input_layernorm: buffer.Buffer,
    q_proj: buffer.Buffer,
    k_proj: buffer.Buffer,
    v_proj: buffer.Buffer,
    o_proj: buffer.Buffer,
    post_attention_layernorm: buffer.Buffer,
    gate_proj: buffer.Buffer,
    up_proj: buffer.Buffer,
    down_proj: buffer.Buffer,

    pub fn deinit(self: *GpuLayer, device: vk.c.VkDevice) void {
        self.input_layernorm.deinit(device);
        self.q_proj.deinit(device);
        self.k_proj.deinit(device);
        self.v_proj.deinit(device);
        self.o_proj.deinit(device);
        self.post_attention_layernorm.deinit(device);
        self.gate_proj.deinit(device);
        self.up_proj.deinit(device);
        self.down_proj.deinit(device);
    }
};

pub const GpuModel = struct {
    config: config_mod.Config,
    embed_tokens: buffer.Buffer,
    layers: []GpuLayer,
    final_norm: buffer.Buffer,
    /// LM head buffer. For tied-embedding models we still allocate a
    /// dedicated buffer (separate VkBuffer) — sharing GPU memory with
    /// embed_tokens would save 1 GiB but add lifetime entanglement we
    /// don't need. We can tighten this if memory pressure ever bites.
    lm_head: buffer.Buffer,
    /// Tracks whether `lm_head` actually shares values with embed_tokens
    /// (for the future memory-saving path), even though the buffers are
    /// distinct. For now informational only.
    lm_head_tied: bool,
    allocator: std.mem.Allocator,

    pub fn upload(
        gpa: std.mem.Allocator,
        ctx: *const vk.Context,
        cpu: *const model_mod.Model,
    ) !GpuModel {
        const cfg = cpu.config;

        var embed = try uploadTensor(gpa, ctx, cpu.embed_tokens);
        errdefer embed.deinit(ctx.device);

        var final_norm = try uploadTensor(gpa, ctx, cpu.final_norm);
        errdefer final_norm.deinit(ctx.device);

        // For the tied case we still upload a fresh copy — the bytes
        // come from the same source so the device-side data is the
        // same; only the host-side allocation pattern differs.
        var lm_head = try uploadTensor(gpa, ctx, cpu.lm_head);
        errdefer lm_head.deinit(ctx.device);

        const layers = try gpa.alloc(GpuLayer, cpu.layers.len);
        var uploaded_layers: usize = 0;
        errdefer {
            var i: usize = 0;
            while (i < uploaded_layers) : (i += 1) layers[i].deinit(ctx.device);
            gpa.free(layers);
        }

        for (cpu.layers, 0..) |layer, i| {
            layers[i] = .{
                .input_layernorm = try uploadTensor(gpa, ctx, layer.input_layernorm),
                .q_proj = try uploadTensor(gpa, ctx, layer.q_proj),
                .k_proj = try uploadTensor(gpa, ctx, layer.k_proj),
                .v_proj = try uploadTensor(gpa, ctx, layer.v_proj),
                .o_proj = try uploadTensor(gpa, ctx, layer.o_proj),
                .post_attention_layernorm = try uploadTensor(gpa, ctx, layer.post_attention_layernorm),
                .gate_proj = try uploadTensor(gpa, ctx, layer.gate_proj),
                .up_proj = try uploadTensor(gpa, ctx, layer.up_proj),
                .down_proj = try uploadTensor(gpa, ctx, layer.down_proj),
            };
            uploaded_layers = i + 1;
        }

        return .{
            .config = cfg,
            .embed_tokens = embed,
            .layers = layers,
            .final_norm = final_norm,
            .lm_head = lm_head,
            .lm_head_tied = cpu.isLmHeadTied(),
            .allocator = gpa,
        };
    }

    pub fn deinit(self: *GpuModel, device: vk.c.VkDevice) void {
        for (self.layers) |*l| l.deinit(device);
        self.allocator.free(self.layers);
        self.embed_tokens.deinit(device);
        self.final_norm.deinit(device);
        self.lm_head.deinit(device);
    }
};

/// Materialise a Tensor as fp32 in a fresh device-local Buffer.
/// Allocates a host scratch buffer the size of the tensor's fp32
/// representation, fills it via the dtype converter, hands it to
/// Buffer.initStatic (which staging-uploads it), then frees the host
/// copy. Peak host memory = one tensor's worth.
fn uploadTensor(gpa: std.mem.Allocator, ctx: *const vk.Context, t: safetensors.Tensor) !buffer.Buffer {
    const numel = t.numel();
    switch (t.dtype) {
        .f32 => {
            // Buffer.initStatic wants a naturally-aligned slice; the
            // mmap'd f32 view is align(1) (SafeTensors makes no
            // alignment guarantee about the data section), so we copy
            // through a host buffer of proper alignment.
            const src = @as([*]align(1) const f32, @ptrCast(t.bytes.ptr))[0..numel];
            const f = try gpa.alloc(f32, numel);
            defer gpa.free(f);
            @memcpy(f, src);
            return buffer.Buffer.initStatic(ctx, f32, f);
        },
        .bf16 => {
            const u = dtype.asU16(t.bytes);
            const f = try gpa.alloc(f32, numel);
            defer gpa.free(f);
            dtype.bf16SliceToF32(u, f);
            return buffer.Buffer.initStatic(ctx, f32, f);
        },
        .f16 => {
            const u = dtype.asU16(t.bytes);
            const f = try gpa.alloc(f32, numel);
            defer gpa.free(f);
            dtype.f16SliceToF32(u, f);
            return buffer.Buffer.initStatic(ctx, f32, f);
        },
        else => return error.UnsupportedWeightDtype,
    }
}
