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
///
/// Two layer flavors share this struct, distinguished by `layer_type`:
///
///   - `.full_attention` (Gemma / Llama / Qwen3 / Qwen3.5 full layers):
///     populates `q_proj` / `k_proj` / `v_proj` / `o_proj` and (Qwen3 /
///     Qwen3.5) `q_norm` / `k_norm`. The Qwen3.5 variant ships `q_proj`
///     at 2× width to emit (q, gate) and applies sigmoid-gated output
///     via `cfg.attn_output_gate`.
///   - `.linear_attention` (Qwen3.5 hybrid layers, Gated DeltaNet):
///     populates `linear_*` instead. q/k/v/o_proj and q/k_norm stay
///     `null` — the recurrent path doesn't have them.
///
/// `input_layernorm`, `post_attention_layernorm`, and the FFN trio
/// (`gate_proj` / `up_proj` / `down_proj`) are present in BOTH flavors;
/// the SwiGLU MLP is identical on hybrid models.
pub const Layer = struct {
    layer_type: config_mod.LayerType,

    input_layernorm: Tensor,
    post_attention_layernorm: Tensor,
    gate_proj: Tensor,
    up_proj: Tensor,
    down_proj: Tensor,

    // ── Full-attention fields ──────────────────────────────────────
    q_proj: ?Tensor = null,
    k_proj: ?Tensor = null,
    v_proj: ?Tensor = null,
    o_proj: ?Tensor = null,
    /// Qwen3 / Qwen3.5: per-head RMSNorm gain on Q and K after the
    /// q_proj/k_proj matmuls and BEFORE RoPE. Shape [head_dim].
    q_norm: ?Tensor = null,
    k_norm: ?Tensor = null,

    // ── Gated DeltaNet fields (linear_attention only) ──────────────
    /// `[2*key_dim + value_dim, hidden]`. Splits in-line at forward time
    /// into (q, k, v) where K and V both live on `linear_key_head_dim`
    /// per K-head, and V is wider by the V/K head-count ratio.
    in_proj_qkv: ?Tensor = null,
    /// `[value_dim, hidden]` — gate input for RMSNormGated.
    in_proj_z: ?Tensor = null,
    /// `[num_v_heads, hidden]` — sigmoid-gated delta-rule mixing factor.
    in_proj_b: ?Tensor = null,
    /// `[num_v_heads, hidden]` — input to the discount gate `g_t`.
    in_proj_a: ?Tensor = null,
    /// `[conv_dim, 1, kernel_size]` depthwise causal conv weights.
    /// `conv_dim == 2*key_dim + value_dim`. No bias in Qwen3.5.
    conv1d_weight: ?Tensor = null,
    /// `[num_v_heads]` log of the discount gate's slope: `g = -exp(A_log) * softplus(...)`.
    A_log: ?Tensor = null,
    /// `[num_v_heads]` bias added inside the softplus.
    dt_bias: ?Tensor = null,
    /// `[head_v_dim]` RMSNormGated weight on the readout. Shared across
    /// V-heads (each head normalises against the same vector).
    ssm_norm_weight: ?Tensor = null,
    /// `[hidden, value_dim]` projection back to the residual stream.
    out_proj: ?Tensor = null,
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

        // Tensor namespace prefix. Qwen3.5 nests the language model under
        // `model.language_model.*`; older families use `model.*` directly.
        const prefix = cfg.family.tensorPrefix();

        // ── Embedding ───────────────────────────────────────────────
        const embed_name = try std.fmt.allocPrint(a, "{s}embed_tokens.weight", .{prefix});
        const embed = try requireTensor(&shards, embed_name);
        try expectShape(embed, embed_name, &.{ cfg.vocab_size, cfg.hidden_size });

        // ── Per-layer weights ───────────────────────────────────────
        const layers = try a.alloc(Layer, cfg.num_hidden_layers);

        const q_dim = cfg.num_attention_heads * cfg.head_dim;
        const kv_dim = cfg.num_key_value_heads * cfg.head_dim;
        // Qwen3.5 widens q_proj to (q, gate); the on-disk row count is 2×.
        const q_proj_rows = if (cfg.attn_output_gate) 2 * q_dim else q_dim;
        const conv_dim = cfg.linearAttnConvDim();
        const value_dim = cfg.linear_num_value_heads * cfg.linear_value_head_dim;

        for (layers, 0..) |*layer, i| {
            layer.* = .{
                .layer_type = cfg.layer_types[i],
                .input_layernorm = undefined,
                .post_attention_layernorm = undefined,
                .gate_proj = undefined,
                .up_proj = undefined,
                .down_proj = undefined,
            };

            layer.input_layernorm = try requireLayerTensor(a, &shards, prefix, i, "input_layernorm.weight");
            try expectShape(layer.input_layernorm, "input_layernorm", &.{cfg.hidden_size});

            layer.post_attention_layernorm = try requireLayerTensor(a, &shards, prefix, i, "post_attention_layernorm.weight");
            try expectShape(layer.post_attention_layernorm, "post_attention_layernorm", &.{cfg.hidden_size});

            layer.gate_proj = try requireLayerTensor(a, &shards, prefix, i, "mlp.gate_proj.weight");
            try expectShape(layer.gate_proj, "gate_proj", &.{ cfg.intermediate_size, cfg.hidden_size });

            layer.up_proj = try requireLayerTensor(a, &shards, prefix, i, "mlp.up_proj.weight");
            try expectShape(layer.up_proj, "up_proj", &.{ cfg.intermediate_size, cfg.hidden_size });

            layer.down_proj = try requireLayerTensor(a, &shards, prefix, i, "mlp.down_proj.weight");
            try expectShape(layer.down_proj, "down_proj", &.{ cfg.hidden_size, cfg.intermediate_size });

            switch (layer.layer_type) {
                .full_attention => {
                    const q = try requireLayerTensor(a, &shards, prefix, i, "self_attn.q_proj.weight");
                    try expectShape(q, "q_proj", &.{ q_proj_rows, cfg.hidden_size });
                    layer.q_proj = q;

                    const k = try requireLayerTensor(a, &shards, prefix, i, "self_attn.k_proj.weight");
                    try expectShape(k, "k_proj", &.{ kv_dim, cfg.hidden_size });
                    layer.k_proj = k;

                    const v = try requireLayerTensor(a, &shards, prefix, i, "self_attn.v_proj.weight");
                    try expectShape(v, "v_proj", &.{ kv_dim, cfg.hidden_size });
                    layer.v_proj = v;

                    const o = try requireLayerTensor(a, &shards, prefix, i, "self_attn.o_proj.weight");
                    try expectShape(o, "o_proj", &.{ cfg.hidden_size, q_dim });
                    layer.o_proj = o;

                    if (cfg.family.hasQkNorm()) {
                        const qn = try requireLayerTensor(a, &shards, prefix, i, "self_attn.q_norm.weight");
                        try expectShape(qn, "q_norm", &.{cfg.head_dim});
                        layer.q_norm = qn;
                        const kn = try requireLayerTensor(a, &shards, prefix, i, "self_attn.k_norm.weight");
                        try expectShape(kn, "k_norm", &.{cfg.head_dim});
                        layer.k_norm = kn;
                    }
                },
                .linear_attention => {
                    // Gated DeltaNet block. All tensor names live under
                    // `linear_attn.*` instead of `self_attn.*`. Shapes
                    // depend on the dims pulled from the config — see
                    // Config.linearAttnConvDim() for the recipe.
                    const ipq = try requireLayerTensor(a, &shards, prefix, i, "linear_attn.in_proj_qkv.weight");
                    try expectShape(ipq, "in_proj_qkv", &.{ conv_dim, cfg.hidden_size });
                    layer.in_proj_qkv = ipq;

                    const ipz = try requireLayerTensor(a, &shards, prefix, i, "linear_attn.in_proj_z.weight");
                    try expectShape(ipz, "in_proj_z", &.{ value_dim, cfg.hidden_size });
                    layer.in_proj_z = ipz;

                    const ipb = try requireLayerTensor(a, &shards, prefix, i, "linear_attn.in_proj_b.weight");
                    try expectShape(ipb, "in_proj_b", &.{ cfg.linear_num_value_heads, cfg.hidden_size });
                    layer.in_proj_b = ipb;

                    const ipa = try requireLayerTensor(a, &shards, prefix, i, "linear_attn.in_proj_a.weight");
                    try expectShape(ipa, "in_proj_a", &.{ cfg.linear_num_value_heads, cfg.hidden_size });
                    layer.in_proj_a = ipa;

                    const conv = try requireLayerTensor(a, &shards, prefix, i, "linear_attn.conv1d.weight");
                    try expectShape(conv, "conv1d.weight", &.{ conv_dim, 1, cfg.linear_conv_kernel_dim });
                    layer.conv1d_weight = conv;

                    const a_log = try requireLayerTensor(a, &shards, prefix, i, "linear_attn.A_log");
                    try expectShape(a_log, "A_log", &.{cfg.linear_num_value_heads});
                    layer.A_log = a_log;

                    const dt = try requireLayerTensor(a, &shards, prefix, i, "linear_attn.dt_bias");
                    try expectShape(dt, "dt_bias", &.{cfg.linear_num_value_heads});
                    layer.dt_bias = dt;

                    const sn = try requireLayerTensor(a, &shards, prefix, i, "linear_attn.norm.weight");
                    try expectShape(sn, "ssm_norm.weight", &.{cfg.linear_value_head_dim});
                    layer.ssm_norm_weight = sn;

                    const op = try requireLayerTensor(a, &shards, prefix, i, "linear_attn.out_proj.weight");
                    try expectShape(op, "out_proj", &.{ cfg.hidden_size, value_dim });
                    layer.out_proj = op;
                },
            }
        }

        // ── Final norm ──────────────────────────────────────────────
        const final_norm_name = try std.fmt.allocPrint(a, "{s}norm.weight", .{prefix});
        const final_norm = try requireTensor(&shards, final_norm_name);
        try expectShape(final_norm, final_norm_name, &.{cfg.hidden_size});

        // ── LM head ─────────────────────────────────────────────────
        // If the checkpoint actually has `lm_head.weight` we use it;
        // otherwise we tie to the embedding (Gemma / Qwen3 / Qwen3.5).
        // We don't strictly trust `tie_word_embeddings` from the config
        // — the ground truth is what's on disk. Qwen3.5 puts lm_head
        // (when present) under the same `model.language_model.*` tree.
        const lm_head: Tensor = blk: {
            const std_name = "lm_head.weight";
            if (shards.get(std_name)) |t| {
                try expectShape(t, std_name, &.{ cfg.vocab_size, cfg.hidden_size });
                break :blk t;
            }
            const lm_name = try std.fmt.allocPrint(a, "{s}lm_head.weight", .{prefix});
            if (shards.get(lm_name)) |t| {
                try expectShape(t, lm_name, &.{ cfg.vocab_size, cfg.hidden_size });
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
    prefix: []const u8,
    layer_idx: usize,
    suffix: []const u8,
) !Tensor {
    const name = try std.fmt.allocPrint(a, "{s}layers.{d}.{s}", .{ prefix, layer_idx, suffix });
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
