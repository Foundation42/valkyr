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

/// How weights are stored on the device.
///
///   `fp32_all` — every weight is uploaded as fp32 (bf16 sources get
///   converted on the host side). The kernel set is the simple fp32
///   path. This is the original phase-1 layout, kept for parity tests.
///
///   `bf16_matmul` — the seven big weight matrices per layer (Q/K/V/O
///   and the three FFN projections) stay as raw bf16 on device, halving
///   their footprint and matmul memory bandwidth. The corresponding
///   matmul shader bit-shifts each bf16 to fp32 in-register. Layernorms,
///   embeddings, and lm_head stay fp32 — they're either tiny or read
///   through kernels that haven't grown a bf16 path yet.
pub const Precision = enum { fp32_all, bf16_matmul };

pub const GpuLayer = struct {
    layer_type: config_mod.LayerType,
    input_layernorm: buffer.Buffer,
    post_attention_layernorm: buffer.Buffer,
    gate_proj: buffer.Buffer,
    up_proj: buffer.Buffer,
    down_proj: buffer.Buffer,

    // ── Full-attention buffers (.full_attention only) ───────────────
    q_proj: ?buffer.Buffer = null,
    k_proj: ?buffer.Buffer = null,
    v_proj: ?buffer.Buffer = null,
    o_proj: ?buffer.Buffer = null,
    /// Qwen3 / Qwen3.5: per-head RMSNorm gain on Q and K. `null` for
    /// Gemma / Llama.
    q_norm: ?buffer.Buffer = null,
    k_norm: ?buffer.Buffer = null,

    // ── Linear-attention buffers (.linear_attention only) ───────────
    /// Gated DeltaNet (Qwen3.5 hybrid layers). All eight tensors must
    /// be present together — see `cpu/gated_delta.zig` for the math.
    in_proj_qkv: ?buffer.Buffer = null,
    in_proj_z: ?buffer.Buffer = null,
    in_proj_b: ?buffer.Buffer = null,
    in_proj_a: ?buffer.Buffer = null,
    conv1d_weight: ?buffer.Buffer = null,
    A_log: ?buffer.Buffer = null,
    dt_bias: ?buffer.Buffer = null,
    ssm_norm_weight: ?buffer.Buffer = null,
    out_proj: ?buffer.Buffer = null,

    pub fn deinit(self: *GpuLayer, device: vk.c.VkDevice) void {
        self.input_layernorm.deinit(device);
        self.post_attention_layernorm.deinit(device);
        self.gate_proj.deinit(device);
        self.up_proj.deinit(device);
        self.down_proj.deinit(device);
        inline for (.{
            "q_proj",     "k_proj", "v_proj",        "o_proj",
            "q_norm",     "k_norm",
            "in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a",
            "conv1d_weight", "A_log",  "dt_bias",
            "ssm_norm_weight", "out_proj",
        }) |fname| {
            if (@field(self, fname)) |*b| b.deinit(device);
        }
    }
};

pub const GpuModel = struct {
    config: config_mod.Config,
    precision: Precision,
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
        precision: Precision,
    ) !GpuModel {
        const cfg = cpu.config;
        const matmul_path: TensorPath = switch (precision) {
            .fp32_all => .fp32,
            .bf16_matmul => .bf16_raw_if_bf16,
        };

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
            // Shared bits: norms + FFN trio.
            layers[i] = .{
                .layer_type = layer.layer_type,
                .input_layernorm = try uploadTensor(gpa, ctx, layer.input_layernorm),
                .post_attention_layernorm = try uploadTensor(gpa, ctx, layer.post_attention_layernorm),
                .gate_proj = try uploadByPath(gpa, ctx, layer.gate_proj, matmul_path),
                .up_proj = try uploadByPath(gpa, ctx, layer.up_proj, matmul_path),
                .down_proj = try uploadByPath(gpa, ctx, layer.down_proj, matmul_path),
            };

            switch (layer.layer_type) {
                .full_attention => {
                    layers[i].q_proj = try uploadByPath(gpa, ctx, layer.q_proj.?, matmul_path);
                    layers[i].k_proj = try uploadByPath(gpa, ctx, layer.k_proj.?, matmul_path);
                    layers[i].v_proj = try uploadByPath(gpa, ctx, layer.v_proj.?, matmul_path);
                    layers[i].o_proj = try uploadByPath(gpa, ctx, layer.o_proj.?, matmul_path);
                    if (layer.q_norm) |t| layers[i].q_norm = try uploadTensor(gpa, ctx, t);
                    if (layer.k_norm) |t| layers[i].k_norm = try uploadTensor(gpa, ctx, t);
                },
                .linear_attention => {
                    // Gated DeltaNet: in-projections + conv + per-head
                    // gates + ssm-rmsnorm + out_proj. The four in-proj
                    // matrices and out_proj go through the bf16-aware
                    // path; the rest are tiny scalars/per-head tables
                    // and stay fp32.
                    layers[i].in_proj_qkv = try uploadByPath(gpa, ctx, layer.in_proj_qkv.?, matmul_path);
                    layers[i].in_proj_z   = try uploadByPath(gpa, ctx, layer.in_proj_z.?, matmul_path);
                    layers[i].in_proj_b   = try uploadByPath(gpa, ctx, layer.in_proj_b.?, matmul_path);
                    layers[i].in_proj_a   = try uploadByPath(gpa, ctx, layer.in_proj_a.?, matmul_path);
                    layers[i].out_proj    = try uploadByPath(gpa, ctx, layer.out_proj.?, matmul_path);
                    layers[i].conv1d_weight   = try uploadTensor(gpa, ctx, layer.conv1d_weight.?);
                    layers[i].A_log           = try uploadTensor(gpa, ctx, layer.A_log.?);
                    layers[i].dt_bias         = try uploadTensor(gpa, ctx, layer.dt_bias.?);
                    layers[i].ssm_norm_weight = try uploadTensor(gpa, ctx, layer.ssm_norm_weight.?);
                },
            }
            uploaded_layers = i + 1;
        }

        return .{
            .config = cfg,
            .precision = precision,
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

/// Two upload strategies the bf16-aware path can pick between.
const TensorPath = enum { fp32, bf16_raw_if_bf16 };

fn uploadByPath(gpa: std.mem.Allocator, ctx: *const vk.Context, t: safetensors.Tensor, path: TensorPath) !buffer.Buffer {
    if (path == .bf16_raw_if_bf16 and t.dtype == .bf16) {
        return uploadTensorBf16Raw(gpa, ctx, t);
    }
    return uploadTensor(gpa, ctx, t);
}

/// Upload a bf16 tensor as raw bytes — no conversion. The buffer is
/// sized to fit the bf16 data (numel × 2 bytes) and is meant to be
/// read through a bf16-aware shader (e.g. matmul_nt_v2_bf16) that
/// reinterprets each pair of bf16 elements as one std430 uint and
/// converts to fp32 inline. Halves the upload time and on-device
/// footprint compared to the fp32 path.
fn uploadTensorBf16Raw(gpa: std.mem.Allocator, ctx: *const vk.Context, t: safetensors.Tensor) !buffer.Buffer {
    std.debug.assert(t.dtype == .bf16);
    const numel = t.numel();
    if (numel % 2 != 0) return error.OddElementCountForU32Pack;
    // The shader reads `uint b_pack[]`, each uint holding two bf16.
    // Buffer.initStatic wants natural alignment, and the mmap source
    // is align(1), so we copy through an aligned host buffer.
    const u32_count = numel / 2;
    const src = @as([*]align(1) const u32, @ptrCast(t.bytes.ptr))[0..u32_count];
    const aligned = try gpa.alloc(u32, u32_count);
    defer gpa.free(aligned);
    @memcpy(aligned, src);
    return buffer.Buffer.initStatic(ctx, u32, aligned);
}

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
