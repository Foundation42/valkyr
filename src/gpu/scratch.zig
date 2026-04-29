//! Per-forward-pass scratch buffers on the GPU.
//!
//! Activations during inference: residual stream, x_norm, Q/K/V,
//! head_out, attn_out, mid_norm, gate/up/fused, ffn_out. All
//! device_only (zero-filled at allocation, written exclusively by the
//! GPU during dispatches, optionally read back at the end of the
//! forward pass for parity tests).
//!
//! These get reused across layers — the residual stream lives the
//! whole forward pass, the rest are scratch we overwrite each layer.
//! Allocated once at construction; kept alive for as many forward
//! calls as the caller wants.

const std = @import("std");
const vk = @import("vk.zig");
const buffer = @import("buffer.zig");
const config_mod = @import("../config.zig");

pub const GpuScratch = struct {
    /// Residual stream — flows through every layer. Written by embed
    /// lookup, then read+written by each layer's two residual adds.
    stream: buffer.Buffer,
    /// Final-norm output and LM head input. Separate from `stream` so
    /// the read-only post-norm view doesn't alias the residual that
    /// later steps might still want.
    final_norm_out: buffer.Buffer,
    /// LM head output — one fp32 per vocab token. 1 MiB at vocab_size
    /// = 256000.
    logits: buffer.Buffer,
    /// Attention scores buffer, sized [n_heads, max_pos]. Reused across
    /// all 18 layers within one forward step (each layer overwrites it).
    scores: buffer.Buffer,
    /// Maximum position the KV cache supports. attn_scores writes into
    /// rows of stride `max_pos` and softmax/attn_output read up to
    /// `n_pos` per row.
    max_pos: usize,

    // Per-layer reused scratch.
    x_norm: buffer.Buffer,
    q: buffer.Buffer,
    k: buffer.Buffer,
    v: buffer.Buffer,
    q_rot: buffer.Buffer,
    k_rot: buffer.Buffer,
    head_out: buffer.Buffer,
    attn_out: buffer.Buffer,
    mid_norm: buffer.Buffer,
    gate: buffer.Buffer,
    up: buffer.Buffer,
    fused: buffer.Buffer,
    ffn_out: buffer.Buffer,

    pub fn init(ctx: *const vk.Context, cfg: config_mod.Config, max_pos: usize) !GpuScratch {
        const hidden = cfg.hidden_size;
        const inter = cfg.intermediate_size;
        const q_dim = cfg.num_attention_heads * cfg.head_dim;
        const kv_dim = cfg.num_key_value_heads * cfg.head_dim;

        const f = @sizeOf(f32);
        return .{
            .stream         = try buffer.Buffer.initDeviceOnly(ctx, hidden * f),
            .final_norm_out = try buffer.Buffer.initDeviceOnly(ctx, hidden * f),
            .logits         = try buffer.Buffer.initDeviceOnly(ctx, cfg.vocab_size * f),
            .scores         = try buffer.Buffer.initDeviceOnly(ctx, cfg.num_attention_heads * max_pos * f),
            .max_pos        = max_pos,
            .x_norm         = try buffer.Buffer.initDeviceOnly(ctx, hidden * f),
            .q              = try buffer.Buffer.initDeviceOnly(ctx, q_dim * f),
            .k              = try buffer.Buffer.initDeviceOnly(ctx, kv_dim * f),
            .v              = try buffer.Buffer.initDeviceOnly(ctx, kv_dim * f),
            .q_rot          = try buffer.Buffer.initDeviceOnly(ctx, q_dim * f),
            .k_rot          = try buffer.Buffer.initDeviceOnly(ctx, kv_dim * f),
            .head_out       = try buffer.Buffer.initDeviceOnly(ctx, q_dim * f),
            .attn_out       = try buffer.Buffer.initDeviceOnly(ctx, hidden * f),
            .mid_norm       = try buffer.Buffer.initDeviceOnly(ctx, hidden * f),
            .gate           = try buffer.Buffer.initDeviceOnly(ctx, inter * f),
            .up             = try buffer.Buffer.initDeviceOnly(ctx, inter * f),
            .fused          = try buffer.Buffer.initDeviceOnly(ctx, inter * f),
            .ffn_out        = try buffer.Buffer.initDeviceOnly(ctx, hidden * f),
        };
    }

    pub fn deinit(self: *GpuScratch, device: vk.c.VkDevice) void {
        self.stream.deinit(device);
        self.final_norm_out.deinit(device);
        self.logits.deinit(device);
        self.scores.deinit(device);
        self.x_norm.deinit(device);
        self.q.deinit(device);
        self.k.deinit(device);
        self.v.deinit(device);
        self.q_rot.deinit(device);
        self.k_rot.deinit(device);
        self.head_out.deinit(device);
        self.attn_out.deinit(device);
        self.mid_norm.deinit(device);
        self.gate.deinit(device);
        self.up.deinit(device);
        self.fused.deinit(device);
        self.ffn_out.deinit(device);
    }
};

/// Per-layer K and V caches sized for `max_pos` positions. K_cache and
/// V_cache layouts: `[max_pos, n_kv_heads, head_dim]` flat. The current
/// step writes into row `pos`; subsequent steps read rows `[0..pos+1]`.
///
/// For Gemma 2B at max_pos = 1024: each layer's K is 1024 × 1 × 256 ×
/// 4 bytes = 1 MiB, same for V, so the whole cache (18 layers × 2) is
/// 36 MiB — comfortably small. Resize via re-init if larger contexts
/// become useful.
pub const GpuKvCache = struct {
    layers: []LayerKv,
    max_pos: usize,
    kv_dim: usize,
    allocator: std.mem.Allocator,

    pub const LayerKv = struct {
        k_cache: buffer.Buffer,
        v_cache: buffer.Buffer,

        pub fn deinit(self: *LayerKv, device: vk.c.VkDevice) void {
            self.k_cache.deinit(device);
            self.v_cache.deinit(device);
        }
    };

    pub fn init(gpa: std.mem.Allocator, ctx: *const vk.Context, cfg: config_mod.Config, max_pos: usize) !GpuKvCache {
        const kv_dim = cfg.num_key_value_heads * cfg.head_dim;
        const layers = try gpa.alloc(LayerKv, cfg.num_hidden_layers);
        var done: usize = 0;
        errdefer {
            var i: usize = 0;
            while (i < done) : (i += 1) layers[i].deinit(ctx.device);
            gpa.free(layers);
        }
        const f = @sizeOf(f32);
        for (layers) |*l| {
            l.* = .{
                .k_cache = try buffer.Buffer.initDeviceOnly(ctx, max_pos * kv_dim * f),
                .v_cache = try buffer.Buffer.initDeviceOnly(ctx, max_pos * kv_dim * f),
            };
            done += 1;
        }
        return .{
            .layers = layers,
            .max_pos = max_pos,
            .kv_dim = kv_dim,
            .allocator = gpa,
        };
    }

    pub fn deinit(self: *GpuKvCache, device: vk.c.VkDevice) void {
        for (self.layers) |*l| l.deinit(device);
        self.allocator.free(self.layers);
    }
};

/// Same shape as GpuKvCache but with the V cache stored in TQ4-packed
/// form: each position is one 33-u32 BlockTQ4 (132 bytes) instead of
/// kv_dim fp32s (1024 bytes for Gemma 2B). The K cache stays full
/// fp32 — phase-1 of TurboQuant is asymmetric (K=fp / V=TQ4), per the
/// llama.cpp practitioner default for dense models.
///
/// Plus a per-layer-reused `dequant_v` scratch buffer of size
/// `max_pos * kv_dim * fp32` — once per attention step the whole
/// V history is dequantised into this scratch via tq4_unpack256
/// dispatched with WG count = n_pos, and the existing attn_output
/// kernel reads the scratch unchanged. The scratch is a single
/// allocation reused across all 18 layers.
///
/// Memory delta on Gemma 2B at max_pos = 2048:
///   fp32 V cache:   2048 × 256 × 4 × 18 = 36 MiB
///   TQ4  V cache:   2048 ×  33 × 4 × 18 ≈ 4.6 MiB
///   plus dequant_v: 2048 × 256 × 4      = 2 MiB (one shared scratch)
///   net:            ~6.6 MiB vs 36 MiB → ~5.5× cache compression.
///
/// Block size is selected at init time based on the model's head_dim.
/// Currently 128 (Qwen3) and 256 (Gemma 2B) are wired up; both have
/// matching shader pairs (fwht{N}, rht_pre{N}, rht_post{N}, tq4_pack{N},
/// tq4_unpack{N}, tq4_pack_to_cache{N}). Other power-of-two sizes
/// would need their own shader files.
pub const GpuKvCacheTq4 = struct {
    layers: []LayerKv,
    dequant_v: buffer.Buffer,
    max_pos: usize,
    kv_dim: usize,
    /// Quantisation block size (= head_dim for the supported families).
    block_size: usize,
    /// 1 (γ word) + block_size/8 (one u32 per 8 packed 4-bit indices).
    u32s_per_block: usize,
    /// Number of TQ4 blocks per token's V vector (= num_kv_heads). For
    /// Gemma 2B with kv=1, head_dim=256 → 1 block. For Qwen3-4B with
    /// kv=8, head_dim=128 → 8 blocks.
    n_blocks_per_pos: usize,
    allocator: std.mem.Allocator,

    pub const LayerKv = struct {
        k_cache: buffer.Buffer,
        v_cache: buffer.Buffer,

        pub fn deinit(self: *LayerKv, device: vk.c.VkDevice) void {
            self.k_cache.deinit(device);
            self.v_cache.deinit(device);
        }
    };

    pub fn init(
        gpa: std.mem.Allocator,
        ctx: *const vk.Context,
        cfg: config_mod.Config,
        max_pos: usize,
    ) !GpuKvCacheTq4 {
        const kv_dim = cfg.num_key_value_heads * cfg.head_dim;
        // Pick block size = head_dim. The matching shader pair must
        // exist; only 128 and 256 are wired up today.
        const block_size = cfg.head_dim;
        if (block_size != 128 and block_size != 256) return error.UnsupportedTq4BlockSize;
        if (kv_dim % block_size != 0) return error.KvDimNotMultipleOfBlockSize;
        const n_blocks_per_pos = kv_dim / block_size;
        const u32s_per_block = 1 + block_size / 8;

        const layers = try gpa.alloc(LayerKv, cfg.num_hidden_layers);
        var done: usize = 0;
        errdefer {
            var i: usize = 0;
            while (i < done) : (i += 1) layers[i].deinit(ctx.device);
            gpa.free(layers);
        }
        const f = @sizeOf(f32);
        const u = @sizeOf(u32);
        for (layers) |*l| {
            l.* = .{
                .k_cache = try buffer.Buffer.initDeviceOnly(ctx, max_pos * kv_dim * f),
                .v_cache = try buffer.Buffer.initDeviceOnly(ctx, max_pos * n_blocks_per_pos * u32s_per_block * u),
            };
            done += 1;
        }
        const dequant_v = try buffer.Buffer.initDeviceOnly(ctx, max_pos * kv_dim * f);
        return .{
            .layers = layers,
            .dequant_v = dequant_v,
            .max_pos = max_pos,
            .kv_dim = kv_dim,
            .block_size = block_size,
            .u32s_per_block = u32s_per_block,
            .n_blocks_per_pos = n_blocks_per_pos,
            .allocator = gpa,
        };
    }

    pub fn deinit(self: *GpuKvCacheTq4, device: vk.c.VkDevice) void {
        for (self.layers) |*l| l.deinit(device);
        self.dequant_v.deinit(device);
        self.allocator.free(self.layers);
    }
};
