//! HuggingFace `config.json` reader.
//!
//! We parse only the fields the forward pass actually consumes, plus
//! the architecture string so we can dispatch to the right family at
//! load time. Anything else in the JSON is silently ignored — we don't
//! want to break when HuggingFace adds a new optional field.
//!
//! For phase 1 we only handle Gemma 1 and Llama 2 / Llama 3. Both share
//! the same overall layer recipe (rmsnorm → MQA/GQA attention with RoPE
//! → rmsnorm → gated FFN); the per-family differences are the activation
//! (Gemma: GeGLU, Llama: SwiGLU) and the embedding scale (Gemma scales
//! by sqrt(hidden_size); Llama doesn't). Everything else falls out of
//! the numeric config.

const std = @import("std");

pub const Family = enum {
    gemma,
    llama,
    qwen3,
    /// Qwen3.5 / Qwen3.6 hybrid Gated-DeltaNet + full-attention. Wraps the
    /// language model under `model.language_model.*` and the actual numeric
    /// config under `text_config.*` of `config.json`. Adds attn_output_gate,
    /// partial RoPE, and a per-layer schedule (`layer_types[]`).
    qwen35,

    pub fn fromArchitectures(archs: []const []const u8) !Family {
        for (archs) |a| {
            if (std.mem.eql(u8, a, "GemmaForCausalLM")) return .gemma;
            if (std.mem.eql(u8, a, "LlamaForCausalLM")) return .llama;
            if (std.mem.eql(u8, a, "Qwen3ForCausalLM")) return .qwen3;
            if (std.mem.eql(u8, a, "Qwen3_5ForConditionalGeneration")) return .qwen35;
        }
        return error.UnsupportedArchitecture;
    }

    /// FFN activation. Gemma 1 uses GeGLU (gelu(gate) * up); Llama, Qwen3,
    /// and Qwen3.5 use SwiGLU (silu(gate) * up). The shapes are identical
    /// — only the elementwise activation differs, so the FFN kernel can
    /// pick at dispatch time without any restructuring.
    pub const Activation = enum { gelu, silu };
    pub fn activation(self: Family) Activation {
        return switch (self) {
            .gemma => .gelu,
            .llama, .qwen3, .qwen35 => .silu,
        };
    }

    /// Gemma multiplies the embedding output by sqrt(hidden_size) before
    /// the first transformer block (the pre-norm form expects pre-scaled
    /// inputs); Llama, Qwen3, and Qwen3.5 don't. Cheap detail with a big
    /// numerical impact — forget it and the logits diverge from step 1.
    pub fn embedScalesByDim(self: Family) bool {
        return switch (self) {
            .gemma => true,
            .llama, .qwen3, .qwen35 => false,
        };
    }

    /// Qwen3 and Qwen3.5 apply per-head RMSNorm to the Q and K vectors
    /// after q_proj/k_proj and BEFORE RoPE. Tensor names:
    /// `model.layers.X.self_attn.{q,k}_norm.weight`, shape [head_dim].
    /// Gemma and Llama don't have these. Qwen3.5 only applies them on
    /// `full_attention` layers (linear-attention layers do their own
    /// L2-norm internally).
    pub fn hasQkNorm(self: Family) bool {
        return self == .qwen3 or self == .qwen35;
    }

    /// Qwen3.5 only: the per-layer hybrid schedule mixes full-attention
    /// transformer blocks with Gated-DeltaNet linear-attention blocks.
    pub fn isHybrid(self: Family) bool {
        return self == .qwen35;
    }

    /// Qwen3.5 doubles `q_proj` width to emit (q, gate) and applies
    /// `attn_output * sigmoid(gate)` before `o_proj`. Bool ride-along on
    /// the existing full-attention path.
    pub fn hasAttnOutputGate(self: Family) bool {
        return self == .qwen35;
    }

    /// Tensor namespace prefix. Qwen3.5 wraps its language model under
    /// `model.language_model.*`; the others use `model.*`. Empty string
    /// means no extra wrapping. Trailing dot included.
    pub fn tensorPrefix(self: Family) []const u8 {
        return switch (self) {
            .gemma, .llama, .qwen3 => "model.",
            .qwen35 => "model.language_model.",
        };
    }
};

/// Per-layer dispatch tag. For non-hybrid families every layer is
/// `.full_attention`. Qwen3.5 mixes both per the `layer_types` array.
pub const LayerType = enum(u8) {
    full_attention,
    linear_attention,
};

/// Hard cap on layers handled by `Config.layer_types`. Largest known
/// hybrid model is 64-layer Qwen3.5-27B; 256 leaves comfortable margin.
pub const MAX_LAYERS = 256;

pub const Config = struct {
    family: Family,
    /// Residual stream / hidden dimension (== `d_model`).
    hidden_size: usize,
    /// FFN inner dimension.
    intermediate_size: usize,
    /// Number of stacked transformer blocks.
    num_hidden_layers: usize,
    /// Number of query heads.
    num_attention_heads: usize,
    /// Number of K/V heads. `num_attention_heads / num_key_value_heads`
    /// must be an integer (the GQA grouping factor). Set equal to
    /// `num_attention_heads` for plain MHA, set to 1 for MQA.
    num_key_value_heads: usize,
    /// Per-head dimension. Stored explicitly because some configs
    /// (Gemma) make this independent of `hidden_size / num_attention_heads`
    /// — Gemma 2B has 8 heads × 256 head_dim = 2048 hidden_dim, but in
    /// general the relationship is just convention.
    head_dim: usize,
    /// Maximum positions the model was trained on. We honour it as the
    /// upper bound for the KV cache; longer prompts would need RoPE
    /// extension which we don't support yet.
    max_position_embeddings: usize,
    /// Vocab size of the tokenizer (== embedding rows == LM head cols).
    vocab_size: usize,
    /// RMSNorm epsilon. Tiny but matters — both ours and the reference
    /// must use the same value or activations drift after every layer.
    rms_norm_eps: f32,
    /// RoPE base frequency.
    rope_theta: f32,
    /// Whether the LM head is tied to the embedding matrix. Llama 2 7B
    /// has a separate `lm_head.weight`; Gemma 2B reuses `embed_tokens`.
    /// We default it from family conventions and let the actual presence
    /// of `lm_head.weight` in the checkpoint override at load time.
    tie_word_embeddings: bool,
    bos_token_id: ?u32,
    eos_token_id: ?u32,
    pad_token_id: ?u32,

    // ── Hybrid / Qwen3.5 extensions ─────────────────────────────────
    // For non-hybrid families these stay at safe defaults: every layer
    // is `.full_attention`, no partial rotary, no attn output gate, and
    // the linear-attention dims are zero.
    layer_types: [MAX_LAYERS]LayerType = [_]LayerType{.full_attention} ** MAX_LAYERS,
    /// Fraction of `head_dim` that gets rotated by RoPE on full-attention
    /// layers. 1.0 = full rotation (Gemma/Llama/Qwen3); Qwen3.5 = 0.25.
    /// The rotated prefix is `partial_rotary_factor * head_dim`; the rest
    /// passes through.
    partial_rotary_factor: f32 = 1.0,
    /// `attn_output_gate=True` doubles q_proj output to (q, gate) and
    /// multiplies attention output by sigmoid(gate) before o_proj.
    attn_output_gate: bool = false,
    /// Gated DeltaNet per-layer dims. Zero for non-hybrid families.
    /// Convention follows HF Qwen3.5 config:
    ///   conv_dim = 2*linear_num_key_heads*linear_key_head_dim
    ///            + linear_num_value_heads*linear_value_head_dim
    /// (= 8192 for 4B; the in_proj_qkv output width).
    linear_conv_kernel_dim: usize = 0,
    linear_num_key_heads: usize = 0,
    linear_num_value_heads: usize = 0,
    linear_key_head_dim: usize = 0,
    linear_value_head_dim: usize = 0,

    pub fn loadFromFile(allocator: std.mem.Allocator, path: []const u8) !Config {
        const file = try std.fs.cwd().openFile(path, .{ .mode = .read_only });
        defer file.close();
        const bytes = try file.readToEndAlloc(allocator, 1 * 1024 * 1024); // 1 MiB cap — configs are tiny
        defer allocator.free(bytes);
        return parseFromSlice(allocator, bytes);
    }

    pub fn parseFromSlice(allocator: std.mem.Allocator, json: []const u8) !Config {
        var parsed = try std.json.parseFromSlice(std.json.Value, allocator, json, .{});
        defer parsed.deinit();
        if (parsed.value != .object) return error.ConfigNotObject;
        const outer = parsed.value.object;

        // Architecture detection. `architectures` is an array of strings
        // like ["GemmaForCausalLM"]. `model_type` is a single string
        // like "gemma" — we use the first as primary, fall back to the
        // second if missing (some configs ship only one).
        var family: ?Family = null;
        if (outer.get("architectures")) |archs| {
            if (archs == .array) {
                var names = std.ArrayList([]const u8).init(allocator);
                defer names.deinit();
                for (archs.array.items) |a| {
                    if (a == .string) try names.append(a.string);
                }
                family = Family.fromArchitectures(names.items) catch null;
            }
        }
        if (family == null) {
            if (outer.get("model_type")) |mt| {
                if (mt == .string) {
                    if (std.mem.eql(u8, mt.string, "gemma")) family = .gemma;
                    if (std.mem.eql(u8, mt.string, "llama")) family = .llama;
                    if (std.mem.eql(u8, mt.string, "qwen3")) family = .qwen3;
                    if (std.mem.eql(u8, mt.string, "qwen3_5")) family = .qwen35;
                }
            }
        }
        const fam = family orelse return error.UnsupportedArchitecture;

        // Qwen3.5 wraps the language-model config under `text_config`.
        // Everything numeric (hidden_size, head counts, vocab, ...) lives
        // there, including a nested `rope_parameters` block. For older
        // families the inner namespace IS the outer object.
        const inner = blk: {
            if (fam == .qwen35) {
                const tc = outer.get("text_config") orelse return error.MissingTextConfig;
                if (tc != .object) return error.InvalidTextConfig;
                break :blk tc.object;
            }
            break :blk outer;
        };

        const hidden = try requireUsize(inner, "hidden_size");
        const heads = try requireUsize(inner, "num_attention_heads");

        const head_dim_v = optionalUsize(inner, "head_dim") orelse (hidden / heads);
        const kv_heads = optionalUsize(inner, "num_key_value_heads") orelse heads;
        const n_layers = try requireUsize(inner, "num_hidden_layers");

        // RoPE block. Older families put `rope_theta` flat at top level;
        // Qwen3.5 nests it inside `rope_parameters` along with
        // `partial_rotary_factor`. Read whichever shape is present.
        var rope_theta: f32 = 10000.0;
        var partial_rotary: f32 = 1.0;
        if (inner.get("rope_parameters")) |rp| {
            if (rp == .object) {
                rope_theta = optionalF32(rp.object, "rope_theta") orelse rope_theta;
                partial_rotary = optionalF32(rp.object, "partial_rotary_factor") orelse partial_rotary;
            }
        }
        // Top-level `rope_theta` still wins for older families that have it.
        if (optionalF32(inner, "rope_theta")) |rt| rope_theta = rt;

        var cfg = Config{
            .family = fam,
            .hidden_size = hidden,
            .intermediate_size = try requireUsize(inner, "intermediate_size"),
            .num_hidden_layers = n_layers,
            .num_attention_heads = heads,
            .num_key_value_heads = kv_heads,
            .head_dim = head_dim_v,
            .max_position_embeddings = try requireUsize(inner, "max_position_embeddings"),
            .vocab_size = try requireUsize(inner, "vocab_size"),
            .rms_norm_eps = optionalF32(inner, "rms_norm_eps") orelse 1e-6,
            .rope_theta = rope_theta,
            .tie_word_embeddings = optionalBool(inner, "tie_word_embeddings") orelse (fam == .gemma or fam == .qwen3 or fam == .qwen35),
            .bos_token_id = optionalU32(inner, "bos_token_id") orelse optionalU32(outer, "bos_token_id"),
            .eos_token_id = optionalU32(inner, "eos_token_id") orelse optionalU32(outer, "eos_token_id"),
            .pad_token_id = optionalU32(inner, "pad_token_id") orelse optionalU32(outer, "pad_token_id"),
            .partial_rotary_factor = partial_rotary,
            .attn_output_gate = optionalBool(inner, "attn_output_gate") orelse false,
        };

        // Hybrid schedule + Gated DeltaNet dims (Qwen3.5 only).
        if (fam == .qwen35) {
            if (n_layers > MAX_LAYERS) return error.TooManyLayers;
            const lt = inner.get("layer_types") orelse return error.MissingLayerTypes;
            if (lt != .array) return error.InvalidLayerTypes;
            if (lt.array.items.len != n_layers) return error.LayerTypesLengthMismatch;
            for (lt.array.items, 0..) |item, i| {
                if (item != .string) return error.InvalidLayerType;
                const s = item.string;
                if (std.mem.eql(u8, s, "linear_attention")) {
                    cfg.layer_types[i] = .linear_attention;
                } else if (std.mem.eql(u8, s, "full_attention")) {
                    cfg.layer_types[i] = .full_attention;
                } else return error.UnknownLayerType;
            }
            cfg.linear_conv_kernel_dim = optionalUsize(inner, "linear_conv_kernel_dim") orelse 0;
            cfg.linear_num_key_heads = optionalUsize(inner, "linear_num_key_heads") orelse 0;
            cfg.linear_num_value_heads = optionalUsize(inner, "linear_num_value_heads") orelse 0;
            cfg.linear_key_head_dim = optionalUsize(inner, "linear_key_head_dim") orelse 0;
            cfg.linear_value_head_dim = optionalUsize(inner, "linear_value_head_dim") orelse 0;
        }

        return cfg;
    }

    /// Total `in_proj_qkv` output width for one Gated-DeltaNet layer.
    /// Equals `2*key_dim + value_dim` per the HF reference. Returns 0
    /// for non-hybrid configs.
    pub fn linearAttnConvDim(self: Config) usize {
        const k = self.linear_num_key_heads * self.linear_key_head_dim;
        const v = self.linear_num_value_heads * self.linear_value_head_dim;
        return 2 * k + v;
    }

    pub fn print(self: Config, w: anytype) !void {
        try w.print("family:                  {s}\n", .{@tagName(self.family)});
        try w.print("hidden_size:             {d}\n", .{self.hidden_size});
        try w.print("intermediate_size:       {d}\n", .{self.intermediate_size});
        try w.print("num_hidden_layers:       {d}\n", .{self.num_hidden_layers});
        try w.print("num_attention_heads:     {d}\n", .{self.num_attention_heads});
        try w.print("num_key_value_heads:     {d}\n", .{self.num_key_value_heads});
        try w.print("head_dim:                {d}\n", .{self.head_dim});
        try w.print("max_position_embeddings: {d}\n", .{self.max_position_embeddings});
        try w.print("vocab_size:              {d}\n", .{self.vocab_size});
        try w.print("rms_norm_eps:            {e}\n", .{self.rms_norm_eps});
        try w.print("rope_theta:              {d}\n", .{self.rope_theta});
        try w.print("tie_word_embeddings:     {}\n", .{self.tie_word_embeddings});
        if (self.family.isHybrid()) {
            try w.print("partial_rotary_factor:   {d}\n", .{self.partial_rotary_factor});
            try w.print("attn_output_gate:        {}\n", .{self.attn_output_gate});
            try w.print("linear K heads × dim:    {d} × {d}\n", .{ self.linear_num_key_heads, self.linear_key_head_dim });
            try w.print("linear V heads × dim:    {d} × {d}\n", .{ self.linear_num_value_heads, self.linear_value_head_dim });
            try w.print("linear conv kernel:      {d}\n", .{self.linear_conv_kernel_dim});
            var n_lin: usize = 0;
            var n_full: usize = 0;
            for (self.layer_types[0..self.num_hidden_layers]) |t| switch (t) {
                .linear_attention => n_lin += 1,
                .full_attention => n_full += 1,
            };
            try w.print("layer schedule:          {d} linear / {d} full\n", .{ n_lin, n_full });
        }
    }
};

// ── JSON helpers ─────────────────────────────────────────────────────

fn requireUsize(obj: std.json.ObjectMap, key: []const u8) !usize {
    const v = obj.get(key) orelse return error.MissingField;
    if (v != .integer or v.integer < 0) return error.InvalidField;
    return @intCast(v.integer);
}

fn optionalUsize(obj: std.json.ObjectMap, key: []const u8) ?usize {
    const v = obj.get(key) orelse return null;
    if (v != .integer or v.integer < 0) return null;
    return @intCast(v.integer);
}

fn optionalU32(obj: std.json.ObjectMap, key: []const u8) ?u32 {
    const v = obj.get(key) orelse return null;
    // HF often writes ints as JSON numbers; null is valid for "no token".
    return switch (v) {
        .integer => |i| if (i >= 0 and i <= std.math.maxInt(u32)) @intCast(i) else null,
        else => null,
    };
}

fn optionalF32(obj: std.json.ObjectMap, key: []const u8) ?f32 {
    const v = obj.get(key) orelse return null;
    return switch (v) {
        .float => |f| @floatCast(f),
        .integer => |i| @floatFromInt(i),
        else => null,
    };
}

fn optionalBool(obj: std.json.ObjectMap, key: []const u8) ?bool {
    const v = obj.get(key) orelse return null;
    return switch (v) {
        .bool => |b| b,
        else => null,
    };
}
