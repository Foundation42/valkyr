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
const q4_0 = @import("../cpu/q4_0.zig");
const q4_k = @import("../cpu/q4_k.zig");
const jobs = @import("../jobs.zig");

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
pub const Precision = enum {
    fp32_all,
    bf16_matmul,
    /// 4-bit weights for the seven big projections per layer (Q/K/V/O
    /// + FFN trio + linear-attn projections + lm_head when matmul-pathed).
    /// Quantization happens on the host at upload time using the
    /// llama.cpp-compatible Q4_0 scheme (block-32, fp16 scale, signed
    /// indices). The on-device layout is the sequential 5-u32-per-block
    /// form consumed by `matmul_nt_v2_q4_0`. Layernorms, embeddings,
    /// and lm_head stay fp32 — same policy as the bf16 path. Quality
    /// trades roughly 0.05–0.10 PPL above Q4_K_M for ~3.6× weight-side
    /// memory vs. fp32 (or 1.8× vs. bf16) — the only path that
    /// realistically fits 27B-class models on a 24 GiB consumer card.
    q4_0_matmul,
    /// Q4_K_M-style 4-bit weights: super-blocks of 256 elements with
    /// 8 sub-blocks (32 elem each) carrying their own 6-bit (scale, min)
    /// pair, two fp16 super-scales (`d`, `dmin`). Asymmetric dequant
    /// `d * sc * q - dmin * m` gives ~30% lower MSE than Q4_0's
    /// symmetric scheme on Gaussian data, and the iterative
    /// make_qkx2_quants fit beats Q4_0's max-magnitude direct.
    /// Effective 4.5 bits/elem on device (vs. ~5 for our padded Q4_0
    /// layout). The on-device form is 36 u32 per super-block consumed
    /// by `matmul_nt_v2_q4_k`. Same lm_head/embedding policy as Q4_0.
    q4_k_matmul,
};

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
    /// Single VkDeviceMemory backing every weight tensor's VkBuffer.
    /// Outlives the upload — freed by `deinit`. The persistent staging
    /// buffer + cmd buffer it manages are torn down by `pool.finalize`
    /// at the tail of `upload`.
    pool: buffer.BufferPool,
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
            .q4_0_matmul => .q4_0_quantize,
            .q4_k_matmul => .q4_k_quantize,
        };

        // Spawn a worker pool for the upload's CPU-bound inner loops:
        // bf16/fp16 → fp32 conversion (per-element) and Q4_0 row-wise
        // quantize (per-row) parallelize trivially. The pool's lifetime
        // is bound to this scope — torn down before we return. Cost of
        // spawn/join is ~10 ms vs. multi-second uploads, well worth it.
        const js = try jobs.JobSystem.init(gpa, 0); // 0 = auto (CPU - 2)
        defer js.deinit();

        // The biggest tensors (embed_tokens + lm_head) follow a shared
        // bf16-when-non-fp32 rule. They're the single biggest reads in
        // the steady-state forward, and at Qwen3.6 27B's vocab=248320
        // each is ~5 GiB at fp32 → 2.5 GiB at bf16, halving both the
        // on-device footprint AND the staging-buffer peak during upload
        // (the latter being what decides whether 27B-class models
        // actually fit on a 24 GiB card). lm_head deliberately skips
        // Q4_0: logits are sampled from, and 4-bit quantization could
        // shift the argmax.
        const lm_head_path: TensorPath = switch (precision) {
            .fp32_all => .fp32,
            .bf16_matmul, .q4_0_matmul, .q4_k_matmul => .bf16_raw_if_bf16,
        };

        // Pre-pass: walk every tensor once to size the device pool and
        // staging buffer. Mirrors the upload tree exactly, so the
        // path/dtype decisions match. We compute the total target bytes
        // (with per-tensor alignment slack) for the device VkDeviceMemory
        // and the largest single tensor for the staging capacity. The
        // alternative is allocating slightly conservatively without a
        // pre-pass, but device memory pressure is already tight at 27B-
        // class and we want a tight, deterministic allocation.
        const slack_per_tensor: usize = 256;
        var total_bytes: usize = 0;
        var max_tensor_bytes: usize = 0;
        const accountTensor = struct {
            fn run(t: safetensors.Tensor, p: TensorPath, total: *usize, mx: *usize, slack: usize) void {
                const tb = targetBytes(t, p);
                total.* += tb + slack;
                if (tb > mx.*) mx.* = tb;
            }
        }.run;
        accountTensor(cpu.embed_tokens, lm_head_path, &total_bytes, &max_tensor_bytes, slack_per_tensor);
        accountTensor(cpu.final_norm,   .fp32,         &total_bytes, &max_tensor_bytes, slack_per_tensor);
        accountTensor(cpu.lm_head,      lm_head_path,  &total_bytes, &max_tensor_bytes, slack_per_tensor);
        for (cpu.layers) |layer| {
            accountTensor(layer.input_layernorm,          .fp32,        &total_bytes, &max_tensor_bytes, slack_per_tensor);
            accountTensor(layer.post_attention_layernorm, .fp32,        &total_bytes, &max_tensor_bytes, slack_per_tensor);
            accountTensor(layer.gate_proj,                matmul_path,  &total_bytes, &max_tensor_bytes, slack_per_tensor);
            accountTensor(layer.up_proj,                  matmul_path,  &total_bytes, &max_tensor_bytes, slack_per_tensor);
            accountTensor(layer.down_proj,                matmul_path,  &total_bytes, &max_tensor_bytes, slack_per_tensor);
            switch (layer.layer_type) {
                .full_attention => {
                    accountTensor(layer.q_proj.?, matmul_path, &total_bytes, &max_tensor_bytes, slack_per_tensor);
                    accountTensor(layer.k_proj.?, matmul_path, &total_bytes, &max_tensor_bytes, slack_per_tensor);
                    accountTensor(layer.v_proj.?, matmul_path, &total_bytes, &max_tensor_bytes, slack_per_tensor);
                    accountTensor(layer.o_proj.?, matmul_path, &total_bytes, &max_tensor_bytes, slack_per_tensor);
                    if (layer.q_norm) |t| accountTensor(t, .fp32, &total_bytes, &max_tensor_bytes, slack_per_tensor);
                    if (layer.k_norm) |t| accountTensor(t, .fp32, &total_bytes, &max_tensor_bytes, slack_per_tensor);
                },
                .linear_attention => {
                    accountTensor(layer.in_proj_qkv.?,    matmul_path, &total_bytes, &max_tensor_bytes, slack_per_tensor);
                    accountTensor(layer.in_proj_z.?,      matmul_path, &total_bytes, &max_tensor_bytes, slack_per_tensor);
                    accountTensor(layer.in_proj_b.?,      matmul_path, &total_bytes, &max_tensor_bytes, slack_per_tensor);
                    accountTensor(layer.in_proj_a.?,      matmul_path, &total_bytes, &max_tensor_bytes, slack_per_tensor);
                    accountTensor(layer.out_proj.?,       matmul_path, &total_bytes, &max_tensor_bytes, slack_per_tensor);
                    accountTensor(layer.conv1d_weight.?,  .fp32,       &total_bytes, &max_tensor_bytes, slack_per_tensor);
                    accountTensor(layer.A_log.?,          .fp32,       &total_bytes, &max_tensor_bytes, slack_per_tensor);
                    accountTensor(layer.dt_bias.?,        .fp32,       &total_bytes, &max_tensor_bytes, slack_per_tensor);
                    accountTensor(layer.ssm_norm_weight.?, .fp32,      &total_bytes, &max_tensor_bytes, slack_per_tensor);
                },
            }
        }

        // Staging buffer must fit the largest tensor in one shot —
        // otherwise we'd have to chunk it. 64 KiB extra to absorb any
        // stray alignment slop. Mid-stream flushes happen automatically
        // when the running staging offset would overflow.
        const staging_capacity: usize = max_tensor_bytes + 65536;
        var pool = try buffer.BufferPool.init(ctx, total_bytes, staging_capacity);
        errdefer pool.deinit(ctx.device);

        var embed = try uploadByPath(gpa, ctx, cpu.embed_tokens, lm_head_path, js, &pool);
        errdefer embed.deinit(ctx.device);

        var final_norm = try uploadTensor(gpa, ctx, cpu.final_norm, js, &pool);
        errdefer final_norm.deinit(ctx.device);

        // For the tied case we still upload a fresh copy — the bytes
        // come from the same source so the device-side data is the
        // same; only the host-side allocation pattern differs.
        var lm_head = try uploadByPath(gpa, ctx, cpu.lm_head, lm_head_path, js, &pool);
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
                .input_layernorm = try uploadTensor(gpa, ctx, layer.input_layernorm, js, &pool),
                .post_attention_layernorm = try uploadTensor(gpa, ctx, layer.post_attention_layernorm, js, &pool),
                .gate_proj = try uploadByPath(gpa, ctx, layer.gate_proj, matmul_path, js, &pool),
                .up_proj = try uploadByPath(gpa, ctx, layer.up_proj, matmul_path, js, &pool),
                .down_proj = try uploadByPath(gpa, ctx, layer.down_proj, matmul_path, js, &pool),
            };

            switch (layer.layer_type) {
                .full_attention => {
                    layers[i].q_proj = try uploadByPath(gpa, ctx, layer.q_proj.?, matmul_path, js, &pool);
                    layers[i].k_proj = try uploadByPath(gpa, ctx, layer.k_proj.?, matmul_path, js, &pool);
                    layers[i].v_proj = try uploadByPath(gpa, ctx, layer.v_proj.?, matmul_path, js, &pool);
                    layers[i].o_proj = try uploadByPath(gpa, ctx, layer.o_proj.?, matmul_path, js, &pool);
                    if (layer.q_norm) |t| layers[i].q_norm = try uploadTensor(gpa, ctx, t, js, &pool);
                    if (layer.k_norm) |t| layers[i].k_norm = try uploadTensor(gpa, ctx, t, js, &pool);
                },
                .linear_attention => {
                    // Gated DeltaNet: in-projections + conv + per-head
                    // gates + ssm-rmsnorm + out_proj. The four in-proj
                    // matrices and out_proj go through the bf16-aware
                    // path; the rest are tiny scalars/per-head tables
                    // and stay fp32.
                    layers[i].in_proj_qkv = try uploadByPath(gpa, ctx, layer.in_proj_qkv.?, matmul_path, js, &pool);
                    layers[i].in_proj_z   = try uploadByPath(gpa, ctx, layer.in_proj_z.?, matmul_path, js, &pool);
                    layers[i].in_proj_b   = try uploadByPath(gpa, ctx, layer.in_proj_b.?, matmul_path, js, &pool);
                    layers[i].in_proj_a   = try uploadByPath(gpa, ctx, layer.in_proj_a.?, matmul_path, js, &pool);
                    layers[i].out_proj    = try uploadByPath(gpa, ctx, layer.out_proj.?, matmul_path, js, &pool);
                    layers[i].conv1d_weight   = try uploadTensor(gpa, ctx, layer.conv1d_weight.?, js, &pool);
                    layers[i].A_log           = try uploadTensor(gpa, ctx, layer.A_log.?, js, &pool);
                    layers[i].dt_bias         = try uploadTensor(gpa, ctx, layer.dt_bias.?, js, &pool);
                    layers[i].ssm_norm_weight = try uploadTensor(gpa, ctx, layer.ssm_norm_weight.?, js, &pool);
                },
            }
            uploaded_layers = i + 1;
        }

        // Submit the final batch + tear down staging. Pool's
        // VkDeviceMemory survives in `GpuModel` so weight reads stay
        // valid.
        try pool.finalize(ctx);

        return .{
            .config = cfg,
            .precision = precision,
            .embed_tokens = embed,
            .layers = layers,
            .final_norm = final_norm,
            .lm_head = lm_head,
            .lm_head_tied = cpu.isLmHeadTied(),
            .pool = pool,
            .allocator = gpa,
        };
    }

    pub fn deinit(self: *GpuModel, device: vk.c.VkDevice) void {
        for (self.layers) |*l| l.deinit(device);
        self.allocator.free(self.layers);
        self.embed_tokens.deinit(device);
        self.final_norm.deinit(device);
        self.lm_head.deinit(device);
        // VkBuffer handles came from the pool; the pool owns their
        // backing VkDeviceMemory.
        self.pool.deinit(device);
    }
};

/// Upload strategies the matmul-aware path can pick between.
const TensorPath = enum { fp32, bf16_raw_if_bf16, q4_0_quantize, q4_k_quantize };

/// Bytes the device-side buffer will hold for `t` under `path`. Used
/// by the upload pre-pass to size the BufferPool's VkDeviceMemory and
/// staging buffer.
fn targetBytes(t: safetensors.Tensor, path: TensorPath) usize {
    if (path == .bf16_raw_if_bf16 and t.dtype == .bf16) {
        return t.numel() * 2; // bf16, 2 bytes per element
    }
    if (path == .q4_0_quantize) {
        const k = t.shape[t.shape.len - 1];
        const blocks = (t.numel() / k) * (k / q4_0.BLOCK_SIZE);
        return blocks * q4_0.GPU_U32S_PER_BLOCK * 4;
    }
    if (path == .q4_k_quantize) {
        const k = t.shape[t.shape.len - 1];
        const supers = (t.numel() / k) * (k / q4_k.QK_K);
        return supers * q4_k.GPU_U32S_PER_SUPERBLOCK * 4;
    }
    return t.numel() * 4; // fp32 destination
}

fn uploadByPath(gpa: std.mem.Allocator, ctx: *const vk.Context, t: safetensors.Tensor, path: TensorPath, js: *jobs.JobSystem, pool: *buffer.BufferPool) !buffer.Buffer {
    if (path == .bf16_raw_if_bf16 and t.dtype == .bf16) {
        return uploadTensorBf16Raw(gpa, ctx, t, pool);
    }
    if (path == .q4_0_quantize) {
        return uploadTensorQ4_0(gpa, ctx, t, js, pool);
    }
    if (path == .q4_k_quantize) {
        return uploadTensorQ4_K(gpa, ctx, t, js, pool);
    }
    return uploadTensor(gpa, ctx, t, js, pool);
}

/// Upload a bf16 tensor as raw bytes — no conversion. The buffer is
/// sized to fit the bf16 data (numel × 2 bytes) and is meant to be
/// read through a bf16-aware shader (e.g. matmul_nt_v2_bf16) that
/// reinterprets each pair of bf16 elements as one std430 uint and
/// converts to fp32 inline. Halves the upload time and on-device
/// footprint compared to the fp32 path.
fn uploadTensorBf16Raw(gpa: std.mem.Allocator, ctx: *const vk.Context, t: safetensors.Tensor, pool: *buffer.BufferPool) !buffer.Buffer {
    _ = gpa;
    std.debug.assert(t.dtype == .bf16);
    const numel = t.numel();
    if (numel % 2 != 0) return error.OddElementCountForU32Pack;
    // The shader reads `uint b_pack[]`, each uint holding two bf16. The
    // staging buffer's mapped pointer is byte-aligned and Vulkan
    // doesn't care about source alignment for memcpy, so we hand the
    // raw mmap bytes straight into the pool — no aligned host scratch
    // needed.
    return pool.commit(ctx, t.bytes);
}

// ── Parallel inner-loop helpers ────────────────────────────────────
//
// Job contexts must outlive the parallelFor call but their lifetimes
// are bound to the surrounding helper function — they live on the
// stack and waitFor returns before the helper does. The `*const
// anyopaque` round trip through jobs.BatchRange is opaque to the
// compiler; we recover the typed pointer in the worker.

const Bf16ConvCtx = struct { src: []align(1) const u16, dst: []f32 };

fn bf16ConvJob(j: *jobs.Job) void {
    const range = j.getData(jobs.BatchRange);
    const c: *const Bf16ConvCtx = @ptrCast(@alignCast(range.context));
    for (range.start..range.end) |i| c.dst[i] = dtype.bf16ToF32(c.src[i]);
}

const Fp16ConvCtx = struct { src: []align(1) const u16, dst: []f32 };

fn fp16ConvJob(j: *jobs.Job) void {
    const range = j.getData(jobs.BatchRange);
    const c: *const Fp16ConvCtx = @ptrCast(@alignCast(range.context));
    for (range.start..range.end) |i| c.dst[i] = dtype.f16ToF32(c.src[i]);
}

const Q4QuantCtx = struct {
    src: []const f32,
    dst: []q4_0.Block,
    k: usize,
    blocks_per_row: usize,
};

fn q4QuantizeJob(j: *jobs.Job) void {
    const range = j.getData(jobs.BatchRange);
    const c: *const Q4QuantCtx = @ptrCast(@alignCast(range.context));
    for (range.start..range.end) |row| {
        const src_row = c.src[row * c.k .. (row + 1) * c.k];
        const dst_row = c.dst[row * c.blocks_per_row .. (row + 1) * c.blocks_per_row];
        q4_0.quantizeRow(src_row, dst_row);
    }
}

const Q4KQuantCtx = struct {
    src: []const f32,
    dst: []q4_k.Block,
    k: usize,
    supers_per_row: usize,
};

fn q4KQuantizeJob(j: *jobs.Job) void {
    const range = j.getData(jobs.BatchRange);
    const c: *const Q4KQuantCtx = @ptrCast(@alignCast(range.context));
    for (range.start..range.end) |row| {
        const src_row = c.src[row * c.k .. (row + 1) * c.k];
        const dst_row = c.dst[row * c.supers_per_row .. (row + 1) * c.supers_per_row];
        q4_k.quantizeRow(src_row, dst_row);
    }
}

/// Pick a batch size that gives roughly 4× as many batches as worker
/// threads — small enough that work-stealing balances naturally,
/// large enough that scheduling overhead doesn't dominate.
fn batchSize(total: usize, worker_count: u32) u32 {
    const target_batches: u32 = @max(worker_count * 4, 8);
    const bs = (total + target_batches - 1) / target_batches;
    return @intCast(@max(@as(usize, 1), bs));
}

/// Quantize a 2-D weight tensor row-wise to Q4_0 (block-32, fp16
/// scale per block) and upload as a flat u32 buffer in the
/// 5-u32-per-block GPU layout consumed by `matmul_nt_v2_q4_0`.
///
/// We assume the matmul tensor is laid out [N, K] (rows along the
/// inner dimension, since matmul_nt computes A · B^T with each row
/// of B being a K-vector). The last dim must be a multiple of 32.
///
/// Host peak during upload is roughly 2 × tensor_size_fp32 (one fp32
/// scratch + one canonical-Block scratch); both are freed before the
/// next tensor is processed. For Qwen3.6-27B's 17.4 GiB FFN matmul
/// at K=17408 that's ~1 GiB peak host scratch — fine.
fn uploadTensorQ4_0(gpa: std.mem.Allocator, ctx: *const vk.Context, t: safetensors.Tensor, js: *jobs.JobSystem, pool: *buffer.BufferPool) !buffer.Buffer {
    const numel = t.numel();
    if (t.shape.len < 2) return error.Q4_0NeedsAtLeast2D;
    const k = t.shape[t.shape.len - 1];
    if (k % q4_0.BLOCK_SIZE != 0) return error.Q4_0LastDimNotMultipleOf32;
    const rows = numel / k;
    const blocks_per_row = k / q4_0.BLOCK_SIZE;
    const total_blocks = rows * blocks_per_row;

    // Materialise the tensor as fp32 into a host scratch buffer
    // (parallel for bf16/fp16; plain memcpy for fp32 since it's
    // already memory-bandwidth-bound and threading wouldn't help).
    const f = try gpa.alloc(f32, numel);
    defer gpa.free(f);
    switch (t.dtype) {
        .f32 => {
            const src = @as([*]align(1) const f32, @ptrCast(t.bytes.ptr))[0..numel];
            @memcpy(f, src);
        },
        .bf16 => {
            const u = dtype.asU16(t.bytes);
            const ctxc = Bf16ConvCtx{ .src = u, .dst = f };
            var counter = jobs.Counter.init(0);
            js.parallelFor(@intCast(numel), batchSize(numel, js.worker_count), bf16ConvJob, @ptrCast(&ctxc), &counter);
            js.waitFor(&counter);
        },
        .f16 => {
            const u = dtype.asU16(t.bytes);
            const ctxc = Fp16ConvCtx{ .src = u, .dst = f };
            var counter = jobs.Counter.init(0);
            js.parallelFor(@intCast(numel), batchSize(numel, js.worker_count), fp16ConvJob, @ptrCast(&ctxc), &counter);
            js.waitFor(&counter);
        },
        else => return error.UnsupportedWeightDtype,
    }

    // Row-wise Q4_0 quantize through the canonical Block layout —
    // each row is independent so we parallelise across rows.
    const blocks = try gpa.alloc(q4_0.Block, total_blocks);
    defer gpa.free(blocks);
    {
        const ctxc = Q4QuantCtx{ .src = f, .dst = blocks, .k = k, .blocks_per_row = blocks_per_row };
        var counter = jobs.Counter.init(0);
        js.parallelFor(@intCast(rows), batchSize(rows, js.worker_count), q4QuantizeJob, @ptrCast(&ctxc), &counter);
        js.waitFor(&counter);
    }

    // Repack to the GPU's sequential 5-u32-per-block layout and hand
    // the bytes to the pool. (packForGpu is serial; it's small
    // relative to the per-row quantize and on most tensors finishes
    // faster than a thread spawn anyway.)
    const packed_words = total_blocks * q4_0.GPU_U32S_PER_BLOCK;
    const packed_buf = try gpa.alloc(u32, packed_words);
    defer gpa.free(packed_buf);
    q4_0.packForGpu(blocks, packed_buf);

    return pool.commit(ctx, std.mem.sliceAsBytes(packed_buf));
}

/// Quantize a 2-D weight tensor row-wise to Q4_K_M (super-block of 256
/// elem with 8 sub-blocks of 32, per-sub-block 6-bit (scale, min), 2
/// fp16 super-scales) and upload as a flat u32 buffer in the
/// 36-u32-per-super-block GPU layout consumed by `matmul_nt_v2_q4_k`.
///
/// Same row-major [N, K] convention as Q4_0; last dim must be a multiple
/// of 256 (vs. 32 for Q4_0). Holds across every supported model: every
/// projection's K dim is hidden_size or intermediate_size, both
/// multiples of 256 on Gemma 2B / Qwen3.5 / Qwen3.6 27B.
///
/// Per-row quantize cost is higher than Q4_0 — make_qkx2_quants does 21
/// candidate iscales per sub-block — so the parallel-loader matters more
/// here. Host peak during upload is roughly 2.4× tensor_size_fp32 (one
/// fp32 scratch + one canonical Block scratch at 144 B per 256 floats =
/// 1.125× fp32 footprint + the packed-u32 scratch).
fn uploadTensorQ4_K(gpa: std.mem.Allocator, ctx: *const vk.Context, t: safetensors.Tensor, js: *jobs.JobSystem, pool: *buffer.BufferPool) !buffer.Buffer {
    const numel = t.numel();
    if (t.shape.len < 2) return error.Q4_KNeedsAtLeast2D;
    const k = t.shape[t.shape.len - 1];
    if (k % q4_k.QK_K != 0) return error.Q4_KLastDimNotMultipleOf256;
    const rows = numel / k;
    const supers_per_row = k / q4_k.QK_K;
    const total_supers = rows * supers_per_row;

    // Materialise the tensor as fp32 into host scratch (parallel for
    // bf16/fp16; plain memcpy for fp32). Identical to the Q4_0 path.
    const f = try gpa.alloc(f32, numel);
    defer gpa.free(f);
    switch (t.dtype) {
        .f32 => {
            const src = @as([*]align(1) const f32, @ptrCast(t.bytes.ptr))[0..numel];
            @memcpy(f, src);
        },
        .bf16 => {
            const u = dtype.asU16(t.bytes);
            const ctxc = Bf16ConvCtx{ .src = u, .dst = f };
            var counter = jobs.Counter.init(0);
            js.parallelFor(@intCast(numel), batchSize(numel, js.worker_count), bf16ConvJob, @ptrCast(&ctxc), &counter);
            js.waitFor(&counter);
        },
        .f16 => {
            const u = dtype.asU16(t.bytes);
            const ctxc = Fp16ConvCtx{ .src = u, .dst = f };
            var counter = jobs.Counter.init(0);
            js.parallelFor(@intCast(numel), batchSize(numel, js.worker_count), fp16ConvJob, @ptrCast(&ctxc), &counter);
            js.waitFor(&counter);
        },
        else => return error.UnsupportedWeightDtype,
    }

    // Row-wise quantize through the canonical Block layout, parallel
    // across rows. Each super-block needs ~6 KB of scratch in the inner
    // function; that's per-thread stack so no contention.
    const blocks = try gpa.alloc(q4_k.Block, total_supers);
    defer gpa.free(blocks);
    {
        const ctxc = Q4KQuantCtx{ .src = f, .dst = blocks, .k = k, .supers_per_row = supers_per_row };
        var counter = jobs.Counter.init(0);
        js.parallelFor(@intCast(rows), batchSize(rows, js.worker_count), q4KQuantizeJob, @ptrCast(&ctxc), &counter);
        js.waitFor(&counter);
    }

    // Repack to the GPU's contiguous 36-u32-per-super-block layout.
    const packed_words = total_supers * q4_k.GPU_U32S_PER_SUPERBLOCK;
    const packed_buf = try gpa.alloc(u32, packed_words);
    defer gpa.free(packed_buf);
    q4_k.packForGpu(blocks, packed_buf);

    return pool.commit(ctx, std.mem.sliceAsBytes(packed_buf));
}

/// Materialise a Tensor as fp32 in a fresh device-local Buffer.
/// Allocates a host scratch buffer the size of the tensor's fp32
/// representation, fills it via the dtype converter, hands it to
/// Buffer.initStatic (which staging-uploads it), then frees the host
/// copy. Peak host memory = one tensor's worth. The bf16/fp16
/// conversion loops are parallelised across the upload pool; fp32
/// just memcpys through (already memory-bandwidth-bound).
fn uploadTensor(gpa: std.mem.Allocator, ctx: *const vk.Context, t: safetensors.Tensor, js: *jobs.JobSystem, pool: *buffer.BufferPool) !buffer.Buffer {
    const numel = t.numel();
    switch (t.dtype) {
        .f32 => {
            // The mmap'd source is align(1); pool.commit just memcpys
            // bytes into staging so no aligned host copy is needed.
            return pool.commit(ctx, t.bytes);
        },
        .bf16 => {
            const u = dtype.asU16(t.bytes);
            const f = try gpa.alloc(f32, numel);
            defer gpa.free(f);
            const ctxc = Bf16ConvCtx{ .src = u, .dst = f };
            var counter = jobs.Counter.init(0);
            js.parallelFor(@intCast(numel), batchSize(numel, js.worker_count), bf16ConvJob, @ptrCast(&ctxc), &counter);
            js.waitFor(&counter);
            return pool.commit(ctx, std.mem.sliceAsBytes(f));
        },
        .f16 => {
            const u = dtype.asU16(t.bytes);
            const f = try gpa.alloc(f32, numel);
            defer gpa.free(f);
            const ctxc = Fp16ConvCtx{ .src = u, .dst = f };
            var counter = jobs.Counter.init(0);
            js.parallelFor(@intCast(numel), batchSize(numel, js.worker_count), fp16ConvJob, @ptrCast(&ctxc), &counter);
            js.waitFor(&counter);
            return pool.commit(ctx, std.mem.sliceAsBytes(f));
        },
        else => return error.UnsupportedWeightDtype,
    }
}
