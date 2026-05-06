//! Load real Qwen3 weights as fp32 for the training stack.
//!
//! Materialises a `train_transformer.InitWeights` view over heap-owned
//! fp32 slices, converting bf16/f16 on the fly. Qwen3 ships bf16 on
//! disk; the trainer wants fp32 to match its (currently fp32-only)
//! GPU buffers and Adam state. The conversion is a per-element shift
//! (bf16) or rebias (f16); both lossless going *up* to fp32, so this
//! is just a slow memcpy with a dtype-dispatched widening.
//!
//! Cost for Qwen3-0.6B: ~2.9 GB host fp32 (embed + lm_head dominate
//! at ~622 MB each). The Runner uploads what it needs to GPU; callers
//! can `deinit` the host snapshot once the Runner has its copy.
//!
//! Tied embeddings: Qwen3-0.6B sets `tie_word_embeddings: true`, so on
//! disk `lm_head` aliases `embed_tokens`. We allocate two distinct
//! fp32 copies — the trainer treats them as independent params, which
//! diverges from a true tied-weight setup (where dE/dembed +=
//! dE/dlm_head). Acceptable for β-3a (loader correctness gate); a
//! follow-up chunk can add a tied-update path.

const std = @import("std");
const dtype = @import("../dtype.zig");
const safetensors = @import("../safetensors.zig");
const model_mod = @import("../model.zig");
const hf_cache = @import("../hf_cache.zig");
const transformer = @import("transformer.zig");

pub const TrainWeights = struct {
    allocator: std.mem.Allocator,

    /// Trainer Config derived from the on-disk model config. `n_pos` is
    /// caller-supplied (it's a training hyperparameter, not in the file).
    cfg: transformer.Config,

    /// Heap allocations owned by this struct. `embed`, `final_norm`,
    /// `lm_head`, and `layers` are allocated against `allocator`; the
    /// per-layer fp32 buffers live in `layer_arena` (freed wholesale).
    embed: []f32, // [vocab, dim]
    final_norm: []f32, // [dim]
    lm_head: []f32, // [vocab, dim]
    layers: []transformer.LayerWeights,
    layer_arena: std.heap.ArenaAllocator,

    /// Borrows the f32 slices out as a `transformer.InitWeights` view.
    /// Caller must keep `self` alive until the view is no longer in use.
    pub fn view(self: *const TrainWeights) transformer.InitWeights {
        return .{
            .embed = self.embed,
            .final_norm = self.final_norm,
            .lm_head = self.lm_head,
            .layers = self.layers,
        };
    }

    pub fn deinit(self: *TrainWeights) void {
        self.layer_arena.deinit();
        self.allocator.free(self.embed);
        self.allocator.free(self.final_norm);
        self.allocator.free(self.lm_head);
        self.allocator.free(self.layers);
    }
};

/// Build a fp32 training-weights snapshot from a CPU `Model`.
///
/// `n_pos` is the training sequence length — the trainer pre-allocates
/// activation/scratch buffers sized to it, so it must be locked in at
/// `Runner.init` time. Not derivable from the on-disk config (the
/// `max_position_embeddings` field is the architectural ceiling, often
/// 40K+ — we'd never train at that length).
pub fn loadTrainWeights(
    allocator: std.mem.Allocator,
    cpu: *const model_mod.Model,
    n_pos: u32,
) !TrainWeights {
    const cfg = cpu.config;

    // ── Architecture gate. Trainer currently supports Qwen3-class
    //    full-attention with per-head Q/K-norm and SwiGLU FFN. Bail
    //    early on anything else.
    if (cfg.num_hidden_layers == 0) return error.ZeroLayers;
    if (cfg.attn_output_gate) return error.AttnOutputGateNotSupported;
    for (cpu.layers) |layer| {
        switch (layer.layer_type) {
            .full_attention => {},
            .linear_attention => return error.LinearAttentionNotSupported,
        }
        if (layer.q_norm == null or layer.k_norm == null) return error.MissingQkNorm;
    }

    const dim: u32 = @intCast(cfg.hidden_size);
    const n_heads: u32 = @intCast(cfg.num_attention_heads);
    const n_kv_heads: u32 = @intCast(cfg.num_key_value_heads);
    const head_dim: u32 = @intCast(cfg.head_dim);
    const ff_dim: u32 = @intCast(cfg.intermediate_size);
    const vocab: u32 = @intCast(cfg.vocab_size);
    const n_layers: u32 = @intCast(cfg.num_hidden_layers);
    const q_dim: usize = @as(usize, n_heads) * head_dim;
    const kv_dim: usize = @as(usize, n_kv_heads) * head_dim;

    const tcfg: transformer.Config = .{
        .dim = dim,
        .n_heads = n_heads,
        .n_kv_heads = n_kv_heads,
        .head_dim = head_dim,
        .ff_dim = ff_dim,
        .n_pos = n_pos,
        .n_layers = n_layers,
        .vocab_size = vocab,
        .rms_eps = cfg.rms_norm_eps,
        .rotary_dim = head_dim, // Qwen3: full RoPE across head_dim.
        .rope_theta = cfg.rope_theta,
        .qk_norm = true,
    };

    // ── Stack-level allocations.
    const embed = try allocator.alloc(f32, @as(usize, vocab) * dim);
    errdefer allocator.free(embed);
    try copyTensorAsF32(cpu.embed_tokens, embed);

    const final_norm = try allocator.alloc(f32, dim);
    errdefer allocator.free(final_norm);
    try copyTensorAsF32(cpu.final_norm, final_norm);

    const lm_head = try allocator.alloc(f32, @as(usize, vocab) * dim);
    errdefer allocator.free(lm_head);
    try copyTensorAsF32(cpu.lm_head, lm_head);

    // ── Per-layer allocations live in an arena (cheap wholesale free
    //    on error / deinit).
    var layer_arena = std.heap.ArenaAllocator.init(allocator);
    errdefer layer_arena.deinit();
    const a = layer_arena.allocator();

    const layers = try allocator.alloc(transformer.LayerWeights, n_layers);
    errdefer allocator.free(layers);

    for (cpu.layers, 0..) |src, i| {
        const w_n1 = try a.alloc(f32, dim);
        const w_q = try a.alloc(f32, q_dim * dim);
        const w_k = try a.alloc(f32, kv_dim * dim);
        const w_v = try a.alloc(f32, kv_dim * dim);
        const w_o = try a.alloc(f32, @as(usize, dim) * q_dim);
        const w_n2 = try a.alloc(f32, dim);
        const w_gate = try a.alloc(f32, @as(usize, ff_dim) * dim);
        const w_up = try a.alloc(f32, @as(usize, ff_dim) * dim);
        const w_down = try a.alloc(f32, @as(usize, dim) * ff_dim);
        const w_qn = try a.alloc(f32, head_dim);
        const w_kn = try a.alloc(f32, head_dim);

        try copyTensorAsF32(src.input_layernorm, w_n1);
        try copyTensorAsF32(src.q_proj.?, w_q);
        try copyTensorAsF32(src.k_proj.?, w_k);
        try copyTensorAsF32(src.v_proj.?, w_v);
        try copyTensorAsF32(src.o_proj.?, w_o);
        try copyTensorAsF32(src.post_attention_layernorm, w_n2);
        try copyTensorAsF32(src.gate_proj, w_gate);
        try copyTensorAsF32(src.up_proj, w_up);
        try copyTensorAsF32(src.down_proj, w_down);
        try copyTensorAsF32(src.q_norm.?, w_qn);
        try copyTensorAsF32(src.k_norm.?, w_kn);

        layers[i] = .{
            .w_n1 = w_n1,
            .w_q = w_q,
            .w_k = w_k,
            .w_v = w_v,
            .w_o = w_o,
            .w_n2 = w_n2,
            .w_gate = w_gate,
            .w_up = w_up,
            .w_down = w_down,
            .w_q_norm = w_qn,
            .w_k_norm = w_kn,
        };
    }

    return .{
        .allocator = allocator,
        .cfg = tcfg,
        .embed = embed,
        .final_norm = final_norm,
        .lm_head = lm_head,
        .layers = layers,
        .layer_arena = layer_arena,
    };
}

/// Convenience wrapper: resolve an HF id or directory path, load the
/// CPU model, materialise fp32 train weights, then drop the CPU model
/// (the safetensors mmaps are only needed during conversion).
pub fn loadTrainWeightsFromId(
    allocator: std.mem.Allocator,
    dir_or_id: []const u8,
    n_pos: u32,
) !TrainWeights {
    const dir_path = try hf_cache.resolveModelArg(allocator, dir_or_id);
    defer allocator.free(dir_path);

    var cpu = try model_mod.Model.load(allocator, dir_path);
    defer cpu.deinit();

    return loadTrainWeights(allocator, &cpu, n_pos);
}

/// Element-wise widening from any supported on-disk dtype to fp32.
/// `dst` length must equal `src.numel()`. Lossless for f32/bf16/f16
/// (going *up* to fp32 only loses range when the source already does;
/// no rounding decisions to make on the way up).
fn copyTensorAsF32(src: safetensors.Tensor, dst: []f32) !void {
    if (src.numel() != dst.len) return error.NumelMismatch;
    switch (src.dtype) {
        .f32 => {
            const sf = src.asF32();
            // f32 case: per-element copy (sf is align(1); dst isn't).
            for (sf, dst) |s, *d| d.* = s;
        },
        .bf16 => dtype.bf16SliceToF32(dtype.asU16(src.bytes), dst),
        .f16 => dtype.f16SliceToF32(dtype.asU16(src.bytes), dst),
        else => return error.UnsupportedDtype,
    }
}
