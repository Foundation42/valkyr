//! CPU reference forward pass — single token at a single position.
//!
//! Stacks every primitive in `cpu/math.zig` into the canonical
//! transformer block recipe (rmsnorm → MQA/GQA attention with RoPE →
//! residual → rmsnorm → GeGLU FFN → residual) for `num_hidden_layers`
//! blocks, then applies the final norm and LM head to produce the
//! logit vector over the vocabulary.
//!
//! No KV cache — at pos 0 with no history, attention degenerates to
//! `head_out[h] = V[kv_head_for_h]` (single-element softmax = 1.0). We
//! still compute Q/K/V projections and RoPE so the wiring is identical
//! to the multi-position case we'll grow into; the unused rotated
//! values just fall on the floor for now.
//!
//! Performance is irrelevant here — this is the correctness oracle.
//! A full forward through Gemma 2B IT lands in the ~10–15 s range on
//! a debug build because the inner matmul converts bf16 weights one
//! element at a time. Once a kernel ships on the GPU we expect the
//! whole pass to be sub-millisecond.

const std = @import("std");
const model_mod = @import("../model.zig");
const cpu_math = @import("math.zig");
const turboquant = @import("turboquant.zig");
const gated_delta = @import("gated_delta.zig");
const full_attn = @import("full_attn.zig");

/// Run the full forward pass and write the logits over the vocabulary
/// to `logits` (length must equal `model.config.vocab_size`).
///
/// `scratch_arena_alloc` is used for transient per-layer buffers
/// (residual stream, attn intermediates, FFN intermediates). Using an
/// arena here makes the cleanup story trivial — the caller resets the
/// arena after each call.
pub fn forward(
    model: *const model_mod.Model,
    token_id: u32,
    pos: usize,
    scratch: std.mem.Allocator,
    logits: []f32,
) !void {
    const cfg = model.config;
    if (logits.len != cfg.vocab_size) return error.LogitsSizeMismatch;
    if (token_id >= cfg.vocab_size) return error.TokenOutOfRange;

    const hidden = cfg.hidden_size;
    const inter = cfg.intermediate_size;
    const n_heads = cfg.num_attention_heads;
    const n_kv = cfg.num_key_value_heads;
    const head_dim = cfg.head_dim;
    const q_dim = n_heads * head_dim;
    const kv_dim = n_kv * head_dim;
    const heads_per_kv = n_heads / n_kv;

    // ── Residual stream init ────────────────────────────────────────
    const stream = try scratch.alloc(f32, hidden);
    try cpu_math.embedRowAsF32(stream, model.embed_tokens, token_id);
    if (cfg.family.embedScalesByDim()) {
        const s: f32 = @sqrt(@as(f32, @floatFromInt(hidden)));
        for (stream) |*xi| xi.* *= s;
    }

    // ── Per-layer scratch ───────────────────────────────────────────
    const x_norm = try scratch.alloc(f32, hidden);
    const q = try scratch.alloc(f32, q_dim);
    const k = try scratch.alloc(f32, kv_dim);
    const v = try scratch.alloc(f32, kv_dim);
    const q_rot = try scratch.alloc(f32, q_dim);
    const k_rot = try scratch.alloc(f32, kv_dim);
    const head_out = try scratch.alloc(f32, q_dim);
    const attn_out = try scratch.alloc(f32, hidden);
    const mid_norm = try scratch.alloc(f32, hidden);
    const gate = try scratch.alloc(f32, inter);
    const up = try scratch.alloc(f32, inter);
    const fused = try scratch.alloc(f32, inter);
    const ffn_out = try scratch.alloc(f32, hidden);

    // ── 18 transformer blocks ───────────────────────────────────────
    for (model.layers) |layer| {
        // Pre-attention norm.
        try cpu_math.rmsnorm(x_norm, stream, layer.input_layernorm, cfg.rms_norm_eps, cfg.family);

        // Q/K/V projections.
        try cpu_math.matmul_nt(q, x_norm, layer.q_proj.?, 1, q_dim, hidden);
        try cpu_math.matmul_nt(k, x_norm, layer.k_proj.?, 1, kv_dim, hidden);
        try cpu_math.matmul_nt(v, x_norm, layer.v_proj.?, 1, kv_dim, hidden);

        if (layer.q_norm) |qn| {
            try cpu_math.rmsnormPerHead(q, q, qn, cfg.rms_norm_eps, n_heads, head_dim, cfg.family);
        }
        if (layer.k_norm) |kn| {
            try cpu_math.rmsnormPerHead(k, k, kn, cfg.rms_norm_eps, n_kv, head_dim, cfg.family);
        }

        // RoPE on Q and K. At pos 0 these are identity but kept for
        // shape-compat with the multi-position path.
        try cpu_math.applyRope(q_rot, q, n_heads, head_dim, pos, cfg.rope_theta);
        try cpu_math.applyRope(k_rot, k, n_kv, head_dim, pos, cfg.rope_theta);

        // Attention. With no history, each query head reads the
        // matching kv-head's V directly (softmax over a single token
        // yields 1.0).
        for (0..n_heads) |h| {
            const kv_h = h / heads_per_kv;
            const v_off = kv_h * head_dim;
            const out_off = h * head_dim;
            @memcpy(head_out[out_off .. out_off + head_dim], v[v_off .. v_off + head_dim]);
        }

        // o_proj.
        try cpu_math.matmul_nt(attn_out, head_out, layer.o_proj.?, 1, hidden, q_dim);

        // First residual.
        for (stream, attn_out) |*si, ai| si.* += ai;

        // Post-attention norm.
        try cpu_math.rmsnorm(mid_norm, stream, layer.post_attention_layernorm, cfg.rms_norm_eps, cfg.family);

        // GeGLU FFN.
        try cpu_math.matmul_nt(gate, mid_norm, layer.gate_proj, 1, inter, hidden);
        try cpu_math.matmul_nt(up, mid_norm, layer.up_proj, 1, inter, hidden);
        try cpu_math.gatedFfn(fused, gate, up, cfg.family);
        try cpu_math.matmul_nt(ffn_out, fused, layer.down_proj, 1, hidden, inter);

        // Second residual.
        for (stream, ffn_out) |*si, fi| si.* += fi;
    }

    // ── Final norm + LM head ────────────────────────────────────────
    const final_norm = try scratch.alloc(f32, hidden);
    try cpu_math.rmsnorm(final_norm, stream, model.final_norm, cfg.rms_norm_eps, cfg.family);

    // Logits: matmul against the LM head (which for Gemma is tied to
    // the embedding matrix [vocab_size, hidden_size]).
    try cpu_math.matmul_nt(logits, final_norm, model.lm_head, 1, cfg.vocab_size, hidden);
}

/// Per-call state for the hybrid forward — owns one Gated DeltaNet
/// state per linear layer and one KV cache per full-attention layer.
/// Lifetime: caller constructs once and reuses across forward steps,
/// since both kinds of state grow with the conversation. `init` zeroes
/// both buffers (fresh start).
pub const HybridState = struct {
    /// Per-layer Gated-DeltaNet `(conv_state, recurrent_state)`. Slot
    /// `i` is `null` if layer `i` is `.full_attention` rather than
    /// linear. Owned.
    ssm: []?gated_delta.State,
    /// Per-layer K/V cache, sized for `max_pos` positions. Slot `i` is
    /// `null` if layer `i` is `.linear_attention`. Owned.
    kv: []?full_attn.KvCache,
    allocator: std.mem.Allocator,

    pub fn init(gpa: std.mem.Allocator, model: *const model_mod.Model, max_pos: usize) !HybridState {
        const cfg = model.config;
        const ssm = try gpa.alloc(?gated_delta.State, cfg.num_hidden_layers);
        @memset(ssm, null);
        errdefer gpa.free(ssm);
        const kv = try gpa.alloc(?full_attn.KvCache, cfg.num_hidden_layers);
        @memset(kv, null);
        errdefer gpa.free(kv);

        // Cleanup-on-failure walks the whole slice — `null` slots are
        // safely skipped, so we don't need a separate "allocated up to"
        // counter the way the loader's per-tensor errdefer does.
        errdefer {
            for (ssm) |*s| if (s.*) |*v| v.deinit(gpa);
            for (kv) |*c| if (c.*) |*v| v.deinit(gpa);
        }
        for (model.layers, 0..) |layer, i| {
            switch (layer.layer_type) {
                .linear_attention => ssm[i] = try gated_delta.State.init(gpa, cfg),
                .full_attention => kv[i] = try full_attn.KvCache.init(gpa, cfg, max_pos),
            }
        }
        return .{ .ssm = ssm, .kv = kv, .allocator = gpa };
    }

    pub fn deinit(self: *HybridState) void {
        for (self.ssm) |*s| if (s.*) |*v| v.deinit(self.allocator);
        for (self.kv) |*c| if (c.*) |*v| v.deinit(self.allocator);
        self.allocator.free(self.ssm);
        self.allocator.free(self.kv);
    }
};

/// Hybrid forward step — used for Qwen3.5. Each layer dispatches on
/// `cfg.layer_types[i]` to either the GatedDeltaNet decode (for linear-
/// attention layers) or the full-attention decode (for full-attention
/// layers). MLP / norms are shared. Caller owns `state` across steps.
pub fn forwardHybrid(
    model: *const model_mod.Model,
    token_id: u32,
    pos: usize,
    state: *HybridState,
    scratch: std.mem.Allocator,
    logits: []f32,
) !void {
    const cfg = model.config;
    if (cfg.family != .qwen35) return error.NotHybridFamily;
    if (logits.len != cfg.vocab_size) return error.LogitsSizeMismatch;
    if (token_id >= cfg.vocab_size) return error.TokenOutOfRange;

    const hidden = cfg.hidden_size;
    const inter = cfg.intermediate_size;

    // ── Residual stream init ────────────────────────────────────────
    const stream = try scratch.alloc(f32, hidden);
    try cpu_math.embedRowAsF32(stream, model.embed_tokens, token_id);
    if (cfg.family.embedScalesByDim()) {
        const s: f32 = @sqrt(@as(f32, @floatFromInt(hidden)));
        for (stream) |*xi| xi.* *= s;
    }

    // ── Per-layer scratch (shared across iterations) ────────────────
    const x_norm = try scratch.alloc(f32, hidden);
    const attn_out = try scratch.alloc(f32, hidden);
    const mid_norm = try scratch.alloc(f32, hidden);
    const gate = try scratch.alloc(f32, inter);
    const up = try scratch.alloc(f32, inter);
    const fused = try scratch.alloc(f32, inter);
    const ffn_out = try scratch.alloc(f32, hidden);

    // Diagnostic knob (parity-paired with the GPU runner): stop after
    // this many layers. -1 / unset = full forward. Used during chunk-3
    // GPU integration debugging to layer-bisect divergences.
    const stop_after: i32 = blk: {
        const env_val = std.process.getEnvVarOwned(scratch, "QWEN35_STOP") catch break :blk -1;
        defer scratch.free(env_val);
        break :blk std.fmt.parseInt(i32, env_val, 10) catch -1;
    };

    for (model.layers, 0..) |layer, i| {
        if (stop_after >= 0 and @as(i32, @intCast(i)) >= stop_after) break;
        // Pre-attention norm.
        try cpu_math.rmsnorm(x_norm, stream, layer.input_layernorm, cfg.rms_norm_eps, cfg.family);

        // Attention path: dispatch on layer type.
        switch (layer.layer_type) {
            .linear_attention => {
                try gated_delta.decodeStep(scratch, cfg, layer, &state.ssm[i].?, x_norm, attn_out);
            },
            .full_attention => {
                try full_attn.decodeStep(scratch, cfg, layer, &state.kv[i].?, x_norm, attn_out, pos);
            },
        }


        // First residual.
        for (stream, attn_out) |*si, ai| si.* += ai;

        // Post-attention norm.
        try cpu_math.rmsnorm(mid_norm, stream, layer.post_attention_layernorm, cfg.rms_norm_eps, cfg.family);

        // SwiGLU FFN.
        try cpu_math.matmul_nt(gate, mid_norm, layer.gate_proj, 1, inter, hidden);
        try cpu_math.matmul_nt(up, mid_norm, layer.up_proj, 1, inter, hidden);
        try cpu_math.gatedFfn(fused, gate, up, cfg.family);
        try cpu_math.matmul_nt(ffn_out, fused, layer.down_proj, 1, hidden, inter);

        // Second residual.
        for (stream, ffn_out) |*si, fi| si.* += fi;
    }

    // ── Final norm + LM head ────────────────────────────────────────
    const final_norm = try scratch.alloc(f32, hidden);
    try cpu_math.rmsnorm(final_norm, stream, model.final_norm, cfg.rms_norm_eps, cfg.family);
    try cpu_math.matmul_nt(logits, final_norm, model.lm_head, 1, cfg.vocab_size, hidden);
}

/// Same as `forward`, but every layer's V vector is run through a
/// TQ4 quantize+dequantize before being read by attention. Models
/// what an asymmetric KV cache (K kept full-precision, V compressed)
/// would do at single-position, no-history attention. Currently
/// requires Gemma 2B's shape (n_kv=1, head_dim=256 — exactly one TQ4
/// block per layer's V).
/// n_q-batched hybrid forward — feeds `token_ids[0..n_q]` at positions
/// `[pos_start..pos_start+n_q)` through every layer in one pass and
/// writes `n_q * vocab_size` logits in row-major. Optional `hidden_out`
/// receives the pre-final-norm residual stream (`n_q * hidden_size`)
/// for callers that need the final hidden state (e.g. MTP head feed).
///
/// State is mutated identically to `forwardHybrid` called n_q times in
/// sequence — that's the parity invariant the GPU oracle gates against.
pub fn forwardHybridBatched(
    model: *const model_mod.Model,
    token_ids: []const u32,
    pos_start: usize,
    state: *HybridState,
    scratch: std.mem.Allocator,
    hidden_out: ?[]f32,
    logits: []f32,
) !void {
    const cfg = model.config;
    if (cfg.family != .qwen35) return error.NotHybridFamily;
    const n_q = token_ids.len;
    if (n_q == 0) return error.EmptyBatch;
    if (logits.len != n_q * cfg.vocab_size) return error.LogitsSizeMismatch;
    if (hidden_out) |h| if (h.len != n_q * cfg.hidden_size) return error.HiddenOutSizeMismatch;
    for (token_ids) |tid| if (tid >= cfg.vocab_size) return error.TokenOutOfRange;

    const hidden = cfg.hidden_size;
    const inter = cfg.intermediate_size;

    // ── Residual stream init: n_q × hidden ─────────────────────────
    const stream = try scratch.alloc(f32, n_q * hidden);
    for (0..n_q) |t| {
        try cpu_math.embedRowAsF32(stream[t * hidden .. (t + 1) * hidden], model.embed_tokens, token_ids[t]);
    }
    if (cfg.family.embedScalesByDim()) {
        const s: f32 = @sqrt(@as(f32, @floatFromInt(hidden)));
        for (stream) |*xi| xi.* *= s;
    }

    // ── Per-layer batched scratch ──────────────────────────────────
    const x_norm = try scratch.alloc(f32, n_q * hidden);
    const attn_out = try scratch.alloc(f32, n_q * hidden);
    const mid_norm = try scratch.alloc(f32, n_q * hidden);
    const gate = try scratch.alloc(f32, n_q * inter);
    const up = try scratch.alloc(f32, n_q * inter);
    const fused = try scratch.alloc(f32, n_q * inter);
    const ffn_out = try scratch.alloc(f32, n_q * hidden);

    // Sub-arena for per-row gated_delta calls; reset between rows so
    // the linear-attn path doesn't accumulate scratch across n_q steps.
    var sub_arena = std.heap.ArenaAllocator.init(scratch);
    defer sub_arena.deinit();

    for (model.layers, 0..) |layer, i| {
        // Pre-attention norm (per row).
        for (0..n_q) |t| {
            try cpu_math.rmsnorm(x_norm[t * hidden .. (t + 1) * hidden], stream[t * hidden .. (t + 1) * hidden], layer.input_layernorm, cfg.rms_norm_eps, cfg.family);
        }

        switch (layer.layer_type) {
            .linear_attention => {
                // Sequential per-position decode — gated_delta has stateful
                // recurrence so n_q steps must run in order.
                for (0..n_q) |t| {
                    _ = sub_arena.reset(.retain_capacity);
                    try gated_delta.decodeStep(sub_arena.allocator(), cfg, layer, &state.ssm[i].?, x_norm[t * hidden .. (t + 1) * hidden], attn_out[t * hidden .. (t + 1) * hidden]);
                }
            },
            .full_attention => {
                try full_attn.prefillStep(scratch, cfg, layer, &state.kv[i].?, x_norm, attn_out, pos_start, n_q);
            },
        }

        // First residual.
        for (stream, attn_out) |*si, ai| si.* += ai;

        // Post-attention norm (per row).
        for (0..n_q) |t| {
            try cpu_math.rmsnorm(mid_norm[t * hidden .. (t + 1) * hidden], stream[t * hidden .. (t + 1) * hidden], layer.post_attention_layernorm, cfg.rms_norm_eps, cfg.family);
        }

        // SwiGLU FFN — matmul_nt natively supports M = n_q.
        try cpu_math.matmul_nt(gate, mid_norm, layer.gate_proj, n_q, inter, hidden);
        try cpu_math.matmul_nt(up, mid_norm, layer.up_proj, n_q, inter, hidden);
        try cpu_math.gatedFfn(fused, gate, up, cfg.family);
        try cpu_math.matmul_nt(ffn_out, fused, layer.down_proj, n_q, hidden, inter);

        // Second residual.
        for (stream, ffn_out) |*si, fi| si.* += fi;
    }

    // Optional hidden-out tap (pre-final-norm).
    if (hidden_out) |h| @memcpy(h, stream);

    // Final norm + LM head.
    const final_norm = try scratch.alloc(f32, n_q * hidden);
    for (0..n_q) |t| {
        try cpu_math.rmsnorm(final_norm[t * hidden .. (t + 1) * hidden], stream[t * hidden .. (t + 1) * hidden], model.final_norm, cfg.rms_norm_eps, cfg.family);
    }
    try cpu_math.matmul_nt(logits, final_norm, model.lm_head, n_q, cfg.vocab_size, hidden);
}

pub fn forwardTq4V(
    model: *const model_mod.Model,
    token_id: u32,
    pos: usize,
    scratch: std.mem.Allocator,
    logits: []f32,
) !void {
    const cfg = model.config;
    if (logits.len != cfg.vocab_size) return error.LogitsSizeMismatch;
    if (token_id >= cfg.vocab_size) return error.TokenOutOfRange;
    if (cfg.head_dim != turboquant.block_size_tq4) return error.HeadDimMismatch;
    if (cfg.num_key_value_heads != 1) return error.KvHeadCountMismatch;

    const hidden = cfg.hidden_size;
    const inter = cfg.intermediate_size;
    const n_heads = cfg.num_attention_heads;
    const n_kv = cfg.num_key_value_heads;
    const head_dim = cfg.head_dim;
    const q_dim = n_heads * head_dim;
    const kv_dim = n_kv * head_dim;
    const heads_per_kv = n_heads / n_kv;

    const stream = try scratch.alloc(f32, hidden);
    try cpu_math.embedRowAsF32(stream, model.embed_tokens, token_id);
    if (cfg.family.embedScalesByDim()) {
        const s: f32 = @sqrt(@as(f32, @floatFromInt(hidden)));
        for (stream) |*xi| xi.* *= s;
    }

    const x_norm = try scratch.alloc(f32, hidden);
    const q = try scratch.alloc(f32, q_dim);
    const k = try scratch.alloc(f32, kv_dim);
    const v = try scratch.alloc(f32, kv_dim);
    const q_rot = try scratch.alloc(f32, q_dim);
    const k_rot = try scratch.alloc(f32, kv_dim);
    const head_out = try scratch.alloc(f32, q_dim);
    const attn_out = try scratch.alloc(f32, hidden);
    const mid_norm = try scratch.alloc(f32, hidden);
    const gate = try scratch.alloc(f32, inter);
    const up = try scratch.alloc(f32, inter);
    const fused = try scratch.alloc(f32, inter);
    const ffn_out = try scratch.alloc(f32, hidden);

    for (model.layers) |layer| {
        try cpu_math.rmsnorm(x_norm, stream, layer.input_layernorm, cfg.rms_norm_eps, cfg.family);
        try cpu_math.matmul_nt(q, x_norm, layer.q_proj.?, 1, q_dim, hidden);
        try cpu_math.matmul_nt(k, x_norm, layer.k_proj.?, 1, kv_dim, hidden);
        try cpu_math.matmul_nt(v, x_norm, layer.v_proj.?, 1, kv_dim, hidden);
        if (layer.q_norm) |qn| {
            try cpu_math.rmsnormPerHead(q, q, qn, cfg.rms_norm_eps, n_heads, head_dim, cfg.family);
        }
        if (layer.k_norm) |kn| {
            try cpu_math.rmsnormPerHead(k, k, kn, cfg.rms_norm_eps, n_kv, head_dim, cfg.family);
        }
        try cpu_math.applyRope(q_rot, q, n_heads, head_dim, pos, cfg.rope_theta);
        try cpu_math.applyRope(k_rot, k, n_kv, head_dim, pos, cfg.rope_theta);

        // V-cache write+read through TQ4. With a single KV head and
        // head_dim==block_size, V is exactly one packed block.
        var v_blk: turboquant.BlockTQ4(256) = undefined;
        const v_in: *const [256]f32 = @ptrCast(v.ptr);
        turboquant.quantizeBlockTQ4(256, v_in, &v_blk);
        var v_recon: [256]f32 = undefined;
        turboquant.dequantizeBlockTQ4(256, &v_blk, &v_recon);

        for (0..n_heads) |h| {
            const kv_h = h / heads_per_kv;
            const v_off = kv_h * head_dim;
            const out_off = h * head_dim;
            @memcpy(head_out[out_off .. out_off + head_dim], v_recon[v_off .. v_off + head_dim]);
        }
        try cpu_math.matmul_nt(attn_out, head_out, layer.o_proj.?, 1, hidden, q_dim);
        for (stream, attn_out) |*si, ai| si.* += ai;

        try cpu_math.rmsnorm(mid_norm, stream, layer.post_attention_layernorm, cfg.rms_norm_eps, cfg.family);
        try cpu_math.matmul_nt(gate, mid_norm, layer.gate_proj, 1, inter, hidden);
        try cpu_math.matmul_nt(up, mid_norm, layer.up_proj, 1, inter, hidden);
        try cpu_math.gatedFfn(fused, gate, up, cfg.family);
        try cpu_math.matmul_nt(ffn_out, fused, layer.down_proj, 1, hidden, inter);
        for (stream, ffn_out) |*si, fi| si.* += fi;
    }

    const final_norm = try scratch.alloc(f32, hidden);
    try cpu_math.rmsnorm(final_norm, stream, model.final_norm, cfg.rms_norm_eps, cfg.family);
    try cpu_math.matmul_nt(logits, final_norm, model.lm_head, 1, cfg.vocab_size, hidden);
}

/// Argmax over a logit vector. Returns the index of the largest
/// finite value; ties go to the first encountered (sufficient for
/// greedy sampling).
pub fn argmax(logits: []const f32) usize {
    var best_idx: usize = 0;
    var best_val: f32 = logits[0];
    for (logits[1..], 1..) |v, i| {
        if (v > best_val) {
            best_val = v;
            best_idx = i;
        }
    }
    return best_idx;
}

pub const SampleParams = struct {
    /// Logit temperature divisor. 1.0 = unchanged, lower = sharper
    /// (more deterministic), higher = flatter (more random). Setting
    /// 0 falls through to greedy via the early-return path.
    temperature: f32 = 1.0,
    /// Keep only the top-K probability mass when > 0. 0 disables.
    top_k: u32 = 0,
    /// Nucleus sampling cutoff: keep the smallest set of tokens whose
    /// cumulative probability ≥ top_p. 1.0 disables.
    top_p: f32 = 1.0,
};

/// Sample one token from `logits` according to `params`, using `rng`
/// for the random draw. `scratch` must hold at least `logits.len`
/// floats (used for the per-step probability distribution and
/// reordering — saves repeated allocations across many sample calls).
///
/// When `temperature == 0`, `top_k == 1`, OR every filter is
/// disabled and we'd just pick the max anyway, this short-circuits
/// to argmax (which doesn't touch the RNG, so seeded runs that mix
/// greedy and sampled steps are still reproducible).
pub fn sample(
    logits: []const f32,
    params: SampleParams,
    rng: std.Random,
    scratch: []f32,
) !usize {
    if (params.temperature == 0.0 or params.top_k == 1) return argmax(logits);

    if (scratch.len < logits.len) return error.ScratchTooSmall;
    const probs = scratch[0..logits.len];

    // Apply temperature.
    const inv_t: f32 = if (params.temperature == 1.0) 1.0 else 1.0 / params.temperature;
    for (probs, logits) |*p, l| p.* = l * inv_t;

    // Stable softmax over the whole vocab.
    var max_v: f32 = probs[0];
    for (probs[1..]) |v| if (v > max_v) {
        max_v = v;
    };
    var sum: f32 = 0;
    for (probs) |*p| {
        const e = @exp(p.* - max_v);
        p.* = e;
        sum += e;
    }
    const inv_sum = 1.0 / sum;
    for (probs) |*p| p.* *= inv_sum;

    // Top-K filter via partial selection. Build an index array of the
    // top-K probabilities; everything outside that set goes to zero.
    if (params.top_k > 0 and params.top_k < logits.len) {
        // Find the k-th largest probability via simple O(N·K) selection.
        // K is small in practice (default ≤ 64), so this beats a full
        // sort for our typical vocab_size of 256000.
        const k: usize = params.top_k;
        const top_idx = scratch.ptr + logits.len; // borrow extra scratch
        _ = top_idx;
        // Use a tiny stack-allocated heap.
        var top: [256]usize = undefined;
        if (k > top.len) return error.TopKTooLarge;
        for (0..k) |j| top[j] = j;
        // Maintain `top` as the indices of the current top-K, with the
        // smallest probability at top[k-1]. Initialise from the first K
        // entries, then sort.
        std.mem.sort(usize, top[0..k], probs, struct {
            fn lessThan(p: []const f32, a: usize, b: usize) bool {
                return p[a] > p[b]; // descending so top[k-1] = smallest
            }
        }.lessThan);
        var threshold: f32 = probs[top[k - 1]];
        for (probs[k..], k..) |v, i| {
            if (v <= threshold) continue;
            // Replace the smallest. Maintain sorted order.
            top[k - 1] = i;
            std.mem.sort(usize, top[0..k], probs, struct {
                fn lessThan(p: []const f32, a: usize, b: usize) bool {
                    return p[a] > p[b];
                }
            }.lessThan);
            threshold = probs[top[k - 1]];
        }
        // Zero everything outside the top-K set.
        var keep = [_]bool{false} ** 256;
        for (top[0..k], 0..) |idx, j| {
            _ = idx;
            keep[j] = true;
        }
        // Build a presence mask via a small scratch indexed-set; for
        // very large vocab a HashSet would be cleaner but we have at
        // most 256 indices to keep.
        var keep_set: [256]usize = undefined;
        @memcpy(keep_set[0..k], top[0..k]);
        std.mem.sort(usize, keep_set[0..k], {}, std.sort.asc(usize));
        var keep_idx: usize = 0;
        for (probs, 0..) |*p, i| {
            if (keep_idx < k and keep_set[keep_idx] == i) {
                keep_idx += 1;
            } else {
                p.* = 0;
            }
        }
        // Renormalise.
        var s2: f32 = 0;
        for (probs) |p| s2 += p;
        if (s2 > 0) {
            const inv = 1.0 / s2;
            for (probs) |*p| p.* *= inv;
        }
    }

    // Top-P (nucleus) filter. Sort by prob descending, pick the prefix
    // whose cumulative probability ≥ top_p, zero the rest, renormalise.
    if (params.top_p < 1.0) {
        // For large vocabs we sort indices, not probs. Borrow the
        // post-logits half of scratch for an index buffer.
        if (scratch.len < logits.len * 2) return error.ScratchTooSmall;
        const idx_bytes = std.mem.sliceAsBytes(scratch[logits.len..]);
        const idx = std.mem.bytesAsSlice(u32, idx_bytes[0 .. logits.len * @sizeOf(u32)]);
        for (idx, 0..) |*v, i| v.* = @intCast(i);
        std.mem.sort(u32, idx, probs, struct {
            fn lessThan(p: []const f32, a: u32, b: u32) bool {
                return p[a] > p[b];
            }
        }.lessThan);
        var cum: f32 = 0;
        var cutoff: usize = idx.len;
        for (idx, 0..) |i, j| {
            cum += probs[i];
            if (cum >= params.top_p) {
                cutoff = j + 1;
                break;
            }
        }
        // Mark everything past cutoff as zero.
        for (idx[cutoff..]) |i| probs[i] = 0;
        // Renormalise.
        var s2: f32 = 0;
        for (probs) |p| s2 += p;
        if (s2 > 0) {
            const inv = 1.0 / s2;
            for (probs) |*p| p.* *= inv;
        }
    }

    // Inverse-CDF sample.
    const r = rng.float(f32);
    var cum: f32 = 0;
    for (probs, 0..) |p, i| {
        cum += p;
        if (r <= cum) return i;
    }
    return probs.len - 1; // fallback for fp rounding
}
