//! Model — the bridge between the raw tensor blob on disk and the
//! forward pass.
//!
//! Loading flow:
//!
//!   1. Read `config.json` → Config (architecture, dims, etc.).
//!   2. Open the safetensors shards under the same directory.
//!   3. For each transformer layer, look up the nine weights it needs
//!      by HuggingFace's canonical names; validate shapes against the
//!      config; store the Tensor handles in a Layer.
//!   4. Resolve the embedding, the final norm, and the LM head (which
//!      may or may not be weight-tied to the embedding).
//!
//! No conversion happens here — every Tensor still points into mmap.
//! The forward pass (CPU first, then GPU) is what materialises floats.

const std = @import("std");
const config_mod = @import("config.zig");
const safetensors = @import("safetensors.zig");
const sharded = @import("sharded.zig");

pub const Config = config_mod.Config;
pub const Tensor = safetensors.Tensor;

/// One transformer block's worth of weights. All references point into
/// the parent Model's shards (ultimately into mmap regions); the Layer
/// itself holds no allocations.
pub const Layer = struct {
    input_layernorm: Tensor,
    q_proj: Tensor,
    k_proj: Tensor,
    v_proj: Tensor,
    o_proj: Tensor,
    post_attention_layernorm: Tensor,
    gate_proj: Tensor,
    up_proj: Tensor,
    down_proj: Tensor,
    /// Qwen3-only: per-head RMSNorm gain on Q and K vectors after
    /// q_proj/k_proj and BEFORE RoPE. Shape [head_dim] each. `null` for
    /// families that don't carry these (Gemma, Llama).
    q_norm: ?Tensor,
    k_norm: ?Tensor,
};

pub const Model = struct {
    config: Config,
    /// Owned. Keeps every shard's mmap alive for the lifetime of the
    /// model — every Tensor in this struct borrows from these mappings.
    shards: sharded.Shards,
    /// Owned. Backs the layer slice and any other heap allocations
    /// associated with the model's symbol table.
    arena: std.heap.ArenaAllocator,

    embed_tokens: Tensor,
    layers: []Layer,
    final_norm: Tensor,
    /// LM head weights. For tie_word_embeddings models these point at
    /// the same bytes as `embed_tokens` — callers should not assume the
    /// two are distinct allocations.
    lm_head: Tensor,

    pub fn load(gpa: std.mem.Allocator, dir_path: []const u8) !Model {
        var arena = std.heap.ArenaAllocator.init(gpa);
        errdefer arena.deinit();
        const a = arena.allocator();

        // ── Config ──────────────────────────────────────────────────
        const config_path = try std.fmt.allocPrint(a, "{s}/config.json", .{dir_path});
        const cfg = try Config.loadFromFile(gpa, config_path);

        // ── Shards ──────────────────────────────────────────────────
        var shards = try sharded.Shards.openFromDir(gpa, dir_path);
        errdefer shards.deinit();

        // ── Embedding ───────────────────────────────────────────────
        const embed = try requireTensor(&shards, "model.embed_tokens.weight");
        try expectShape(embed, "model.embed_tokens.weight", &.{ cfg.vocab_size, cfg.hidden_size });

        // ── Per-layer weights ───────────────────────────────────────
        const layers = try a.alloc(Layer, cfg.num_hidden_layers);

        const q_dim = cfg.num_attention_heads * cfg.head_dim;
        const kv_dim = cfg.num_key_value_heads * cfg.head_dim;

        for (layers, 0..) |*layer, i| {
            layer.input_layernorm = try requireLayerTensor(a, &shards, i, "input_layernorm.weight");
            try expectShape(layer.input_layernorm, "input_layernorm", &.{cfg.hidden_size});

            layer.q_proj = try requireLayerTensor(a, &shards, i, "self_attn.q_proj.weight");
            try expectShape(layer.q_proj, "q_proj", &.{ q_dim, cfg.hidden_size });

            layer.k_proj = try requireLayerTensor(a, &shards, i, "self_attn.k_proj.weight");
            try expectShape(layer.k_proj, "k_proj", &.{ kv_dim, cfg.hidden_size });

            layer.v_proj = try requireLayerTensor(a, &shards, i, "self_attn.v_proj.weight");
            try expectShape(layer.v_proj, "v_proj", &.{ kv_dim, cfg.hidden_size });

            layer.o_proj = try requireLayerTensor(a, &shards, i, "self_attn.o_proj.weight");
            try expectShape(layer.o_proj, "o_proj", &.{ cfg.hidden_size, q_dim });

            layer.post_attention_layernorm = try requireLayerTensor(a, &shards, i, "post_attention_layernorm.weight");
            try expectShape(layer.post_attention_layernorm, "post_attention_layernorm", &.{cfg.hidden_size});

            layer.gate_proj = try requireLayerTensor(a, &shards, i, "mlp.gate_proj.weight");
            try expectShape(layer.gate_proj, "gate_proj", &.{ cfg.intermediate_size, cfg.hidden_size });

            layer.up_proj = try requireLayerTensor(a, &shards, i, "mlp.up_proj.weight");
            try expectShape(layer.up_proj, "up_proj", &.{ cfg.intermediate_size, cfg.hidden_size });

            layer.down_proj = try requireLayerTensor(a, &shards, i, "mlp.down_proj.weight");
            try expectShape(layer.down_proj, "down_proj", &.{ cfg.hidden_size, cfg.intermediate_size });

            if (cfg.family.hasQkNorm()) {
                const qn = try requireLayerTensor(a, &shards, i, "self_attn.q_norm.weight");
                try expectShape(qn, "q_norm", &.{cfg.head_dim});
                layer.q_norm = qn;
                const kn = try requireLayerTensor(a, &shards, i, "self_attn.k_norm.weight");
                try expectShape(kn, "k_norm", &.{cfg.head_dim});
                layer.k_norm = kn;
            } else {
                layer.q_norm = null;
                layer.k_norm = null;
            }
        }

        // ── Final norm ──────────────────────────────────────────────
        const final_norm = try requireTensor(&shards, "model.norm.weight");
        try expectShape(final_norm, "model.norm.weight", &.{cfg.hidden_size});

        // ── LM head ─────────────────────────────────────────────────
        // If the checkpoint actually has `lm_head.weight` we use it;
        // otherwise we tie to the embedding (Gemma-style). We don't
        // strictly trust the config's `tie_word_embeddings` flag — the
        // ground truth is what's on disk.
        const lm_head: Tensor = blk: {
            if (shards.get("lm_head.weight")) |t| {
                try expectShape(t, "lm_head.weight", &.{ cfg.vocab_size, cfg.hidden_size });
                break :blk t;
            }
            break :blk embed;
        };

        return .{
            .config = cfg,
            .shards = shards,
            .arena = arena,
            .embed_tokens = embed,
            .layers = layers,
            .final_norm = final_norm,
            .lm_head = lm_head,
        };
    }

    pub fn deinit(self: *Model) void {
        self.shards.deinit();
        self.arena.deinit();
    }

    pub fn isLmHeadTied(self: *const Model) bool {
        return self.lm_head.bytes.ptr == self.embed_tokens.bytes.ptr;
    }
};

// ── Lookup helpers ───────────────────────────────────────────────────

fn requireTensor(shards: *const sharded.Shards, name: []const u8) !Tensor {
    return shards.get(name) orelse {
        std.debug.print("missing tensor: {s}\n", .{name});
        return error.MissingTensor;
    };
}

fn requireLayerTensor(
    a: std.mem.Allocator,
    shards: *const sharded.Shards,
    layer_idx: usize,
    suffix: []const u8,
) !Tensor {
    const name = try std.fmt.allocPrint(a, "model.layers.{d}.{s}", .{ layer_idx, suffix });
    defer a.free(name);
    return requireTensor(shards, name);
}

fn expectShape(t: Tensor, name: []const u8, want: []const usize) !void {
    if (t.shape.len != want.len) {
        std.debug.print("{s}: rank {d}, expected {d}\n", .{ name, t.shape.len, want.len });
        return error.ShapeMismatch;
    }
    for (t.shape, want, 0..) |got, w, i| {
        if (got != w) {
            std.debug.print("{s}: dim {d} = {d}, expected {d}\n", .{ name, i, got, w });
            return error.ShapeMismatch;
        }
    }
}
