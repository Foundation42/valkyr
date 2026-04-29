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

    pub fn fromArchitectures(archs: []const []const u8) !Family {
        for (archs) |a| {
            if (std.mem.eql(u8, a, "GemmaForCausalLM")) return .gemma;
            if (std.mem.eql(u8, a, "LlamaForCausalLM")) return .llama;
            if (std.mem.eql(u8, a, "Qwen3ForCausalLM")) return .qwen3;
        }
        return error.UnsupportedArchitecture;
    }

    /// FFN activation. Gemma 1 uses GeGLU (gelu(gate) * up); Llama and
    /// Qwen3 use SwiGLU (silu(gate) * up). The shapes are identical —
    /// only the elementwise activation differs, so the FFN kernel can
    /// pick at dispatch time without any restructuring.
    pub const Activation = enum { gelu, silu };
    pub fn activation(self: Family) Activation {
        return switch (self) {
            .gemma => .gelu,
            .llama, .qwen3 => .silu,
        };
    }

    /// Gemma multiplies the embedding output by sqrt(hidden_size) before
    /// the first transformer block (the pre-norm form expects pre-scaled
    /// inputs); Llama and Qwen3 don't. Cheap detail with a big numerical
    /// impact — forget it and the logits diverge from step 1.
    pub fn embedScalesByDim(self: Family) bool {
        return switch (self) {
            .gemma => true,
            .llama, .qwen3 => false,
        };
    }

    /// Qwen3 applies per-head RMSNorm to the Q and K vectors after the
    /// q_proj/k_proj matmuls and BEFORE RoPE. Tensor names:
    /// `model.layers.X.self_attn.{q,k}_norm.weight`, both shape [head_dim].
    /// Gemma and Llama don't have these.
    pub fn hasQkNorm(self: Family) bool {
        return self == .qwen3;
    }
};

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
        const obj = parsed.value.object;

        // Architecture detection. `architectures` is an array of strings
        // like ["GemmaForCausalLM"]. `model_type` is a single string
        // like "gemma" — we use the first as primary, fall back to the
        // second if missing (some configs ship only one).
        var family: ?Family = null;
        if (obj.get("architectures")) |archs| {
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
            if (obj.get("model_type")) |mt| {
                if (mt == .string) {
                    if (std.mem.eql(u8, mt.string, "gemma")) family = .gemma;
                    if (std.mem.eql(u8, mt.string, "llama")) family = .llama;
                    if (std.mem.eql(u8, mt.string, "qwen3")) family = .qwen3;
                }
            }
        }
        const fam = family orelse return error.UnsupportedArchitecture;

        const hidden = try requireUsize(obj, "hidden_size");
        const heads = try requireUsize(obj, "num_attention_heads");

        const head_dim_v = optionalUsize(obj, "head_dim") orelse (hidden / heads);
        const kv_heads = optionalUsize(obj, "num_key_value_heads") orelse heads;

        return .{
            .family = fam,
            .hidden_size = hidden,
            .intermediate_size = try requireUsize(obj, "intermediate_size"),
            .num_hidden_layers = try requireUsize(obj, "num_hidden_layers"),
            .num_attention_heads = heads,
            .num_key_value_heads = kv_heads,
            .head_dim = head_dim_v,
            .max_position_embeddings = try requireUsize(obj, "max_position_embeddings"),
            .vocab_size = try requireUsize(obj, "vocab_size"),
            .rms_norm_eps = optionalF32(obj, "rms_norm_eps") orelse 1e-6,
            .rope_theta = optionalF32(obj, "rope_theta") orelse 10000.0,
            .tie_word_embeddings = optionalBool(obj, "tie_word_embeddings") orelse (fam == .gemma or fam == .qwen3),
            .bos_token_id = optionalU32(obj, "bos_token_id"),
            .eos_token_id = optionalU32(obj, "eos_token_id"),
            .pad_token_id = optionalU32(obj, "pad_token_id"),
        };
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
