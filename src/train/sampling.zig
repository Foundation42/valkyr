//! Greedy autoregressive decode for the transformer trainer.
//!
//! Wraps `train_transformer.Runner.forwardLogits` with a sliding-window
//! decode loop. Right-pads the prompt to fill `n_pos`; causal attention
//! makes those pads invisible to logits at earlier positions, so the
//! first generated token is sampled from a clean conditioning. As more
//! tokens are produced the window extends in-place; once full it slides
//! left, dropping the oldest token.
//!
//! Used by both the β-6c sampling smoke and the user-facing
//! `--fine-tune` command for before/after probe samples.

const std = @import("std");
const train_transformer = @import("transformer.zig");
const tokenizer_mod = @import("../tokenizer.zig");

/// Greedy autoregressive decode. Returns an owned `[prompt.len + n_gen]u32`
/// slice. `pad_id` fills the right side of the window before generation
/// fills it; EOS or any other no-op token works since causal attention
/// hides them from logits at earlier positions.
pub fn greedyDecode(
    allocator: std.mem.Allocator,
    runner: *train_transformer.Runner,
    prompt_ids: []const u32,
    n_gen: u32,
    n_pos: u32,
    vocab: usize,
    pad_id: u32,
) ![]u32 {
    if (prompt_ids.len == 0) return error.EmptyPrompt;
    if (prompt_ids.len > n_pos) return error.PromptTooLong;

    const out_len: usize = prompt_ids.len + @as(usize, n_gen);
    const out = try allocator.alloc(u32, out_len);
    errdefer allocator.free(out);
    @memcpy(out[0..prompt_ids.len], prompt_ids);

    const window = try allocator.alloc(u32, n_pos);
    defer allocator.free(window);
    const logits = try allocator.alloc(f32, @as(usize, n_pos) * vocab);
    defer allocator.free(logits);

    for (window, 0..) |*w, idx| {
        w.* = if (idx < prompt_ids.len) prompt_ids[idx] else pad_id;
    }
    var gen_pos: usize = prompt_ids.len - 1;

    var step: u32 = 0;
    while (step < n_gen) : (step += 1) {
        try runner.forwardLogits(window, logits);
        const off = gen_pos * vocab;
        var best: u32 = 0;
        var best_l: f32 = -std.math.inf(f32);
        for (0..vocab) |v| {
            const l = logits[off + v];
            if (l > best_l) {
                best_l = l;
                best = @intCast(v);
            }
        }
        out[prompt_ids.len + step] = best;

        if (gen_pos < n_pos - 1) {
            gen_pos += 1;
            window[gen_pos] = best;
        } else {
            for (0..n_pos - 1) |k| window[k] = window[k + 1];
            window[n_pos - 1] = best;
        }
    }

    return out;
}

/// Concatenate per-token display bytes for a sequence of ids.
pub fn decodeIdsToText(
    allocator: std.mem.Allocator,
    tok: *const tokenizer_mod.Tokenizer,
    ids: []const u32,
) ![]u8 {
    var out = std.ArrayList(u8).init(allocator);
    errdefer out.deinit();
    for (ids) |id| {
        const piece = try tok.decodeForDisplay(allocator, @intCast(id));
        defer allocator.free(piece);
        try out.appendSlice(piece);
    }
    return try out.toOwnedSlice();
}
