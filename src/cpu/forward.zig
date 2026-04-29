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
        try cpu_math.matmul_nt(q, x_norm, layer.q_proj, 1, q_dim, hidden);
        try cpu_math.matmul_nt(k, x_norm, layer.k_proj, 1, kv_dim, hidden);
        try cpu_math.matmul_nt(v, x_norm, layer.v_proj, 1, kv_dim, hidden);

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
        try cpu_math.matmul_nt(attn_out, head_out, layer.o_proj, 1, hidden, q_dim);

        // First residual.
        for (stream, attn_out) |*si, ai| si.* += ai;

        // Post-attention norm.
        try cpu_math.rmsnorm(mid_norm, stream, layer.post_attention_layernorm, cfg.rms_norm_eps, cfg.family);

        // GeGLU FFN.
        try cpu_math.matmul_nt(gate, mid_norm, layer.gate_proj, 1, inter, hidden);
        try cpu_math.matmul_nt(up, mid_norm, layer.up_proj, 1, inter, hidden);
        try cpu_math.geglu(fused, gate, up);
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
