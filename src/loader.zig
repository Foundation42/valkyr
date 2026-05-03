//! High-level model-load orchestration for embedded callers.
//!
//! Wraps the three-step "parse safetensors → upload to GPU → load
//! tokenizer" pipeline that valkyr's CLI inlines into every
//! `runChat`/`runGen` flow. Callers from a host engine
//! (e.g. Matryoshka's ai_demo Game) just want one call:
//!
//! ```
//! var loaded = try valkyr_gpu.loader.loadGpuModel(
//!     allocator, &ctx, "Llama-3.2-1B-Instruct-Q4_K_M",
//!     .{ .precision = .q4_k_matmul },
//! );
//! defer loaded.deinit(ctx.device);
//! ```
//!
//! and then reach `loaded.gpu_model`, `loaded.tokenizer`, and the
//! cached `loaded.config()` from one return value with a clean
//! lifetime story (single deinit reverses everything).
//!
//! The CPU-side `Model` (which mmaps the safetensors files) is held
//! only for the duration of the upload — once the GPU has its own
//! copy of the weights, the safetensors mappings are dropped. That
//! frees several hundred megabytes of host memory for hosts that
//! load big models alongside their own working set (Matryoshka's
//! render targets etc.).

const std = @import("std");
const vk = @import("gpu/vk.zig");
const model_mod = @import("model.zig");
const gpu_model = @import("gpu/model.zig");
const tokenizer_mod = @import("tokenizer.zig");
const config_mod = @import("config.zig");
const hf_cache_mod = @import("hf_cache.zig");

pub const Options = struct {
    /// Weight-matmul precision. Matches valkyr's CLI options:
    ///   .fp32_all      — straight fp32 (memory-hungry; for parity tests)
    ///   .bf16_matmul   — bf16 weights, fp32 accumulators (default)
    ///   .q4_0_matmul   — llama.cpp-compatible block-32 q4_0
    ///   .q4_k_matmul   — block-256 q4_k, beats q4_0 across all sizes
    /// q4_k_matmul is the sweet spot on memory + speed for embedded
    /// use cases — Matryoshka has limited VRAM headroom alongside its
    /// own render targets.
    precision: gpu_model.Precision = .q4_k_matmul,
};

pub const Loaded = struct {
    /// All weights uploaded to GPU memory. Held in one VkDeviceMemory
    /// suballocator pool — see `gpu/model.zig` for layout.
    gpu_model: gpu_model.GpuModel,
    /// CPU-side tokenizer. BPE tables + special-token mappings; sized
    /// by vocab. Used by hosts to encode prompts and decode generated
    /// tokens. Lives separately from gpu_model so non-inference uses
    /// (logging, tool calls, etc.) can reach the tokenizer without
    /// touching GPU state.
    tokenizer: tokenizer_mod.Tokenizer,

    /// Convenience: the model's config, copied out of gpu_model.
    /// Hosts can read num_hidden_layers, vocab_size, hidden_dim, etc.
    /// without dereferencing the GPU model directly.
    pub fn config(self: *const Loaded) config_mod.Config {
        return self.gpu_model.config;
    }

    pub fn deinit(self: *Loaded, device: vk.c.VkDevice) void {
        self.tokenizer.deinit();
        self.gpu_model.deinit(device);
    }
};

/// Load a model from a directory or HF id.
///
/// `dir_or_id` accepts either:
///   - an absolute / relative directory path containing config.json,
///     tokenizer.json, and *.safetensors shards;
///   - a HuggingFace id like "meta-llama/Llama-3.2-1B-Instruct" —
///     resolved against ~/.cache/huggingface/hub via `hf_cache`.
///
/// The CPU-side `Model` (safetensors mmaps + arena) is allocated and
/// freed inside this function — it's only needed during the upload.
/// On error any GPU resources allocated so far are cleanly released.
pub fn loadGpuModel(
    allocator: std.mem.Allocator,
    ctx: *const vk.Context,
    dir_or_id: []const u8,
    options: Options,
) !Loaded {
    const dir_path = try hf_cache_mod.resolveModelArg(allocator, dir_or_id);
    defer allocator.free(dir_path);

    var cpu = try model_mod.Model.load(allocator, dir_path);
    defer cpu.deinit();

    var gm = try gpu_model.GpuModel.upload(allocator, ctx, &cpu, options.precision);
    errdefer gm.deinit(ctx.device);

    const tok_path = try std.fmt.allocPrint(allocator, "{s}/tokenizer.json", .{dir_path});
    defer allocator.free(tok_path);

    var tok = try tokenizer_mod.Tokenizer.loadFromFile(allocator, tok_path);
    errdefer tok.deinit();

    return .{
        .gpu_model = gm,
        .tokenizer = tok,
    };
}
