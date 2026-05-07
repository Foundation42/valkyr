//! Tiny helpers used by more than one smoke / command module.
//! Lifted out of the per-file copies that used to live in main.zig,
//! gpu_gen.zig, layer_tests.zig, gpu_train.zig, and training.zig.

const std = @import("std");

/// Truncate `s` to at most `n` bytes — no codepoint awareness, just a
/// quick "trim long tokenizer strings before printing" for the per-step
/// debug logs.
pub fn truncateStr(s: []const u8, n: usize) []const u8 {
    return if (s.len <= n) s else s[0..n];
}

/// One entry in a top-K result set: vocab id + raw logit value.
pub const TopKEntry = struct { id: usize, value: f32 };

/// Top-K argmax over `logits`. `k` slots are heap-allocated; caller
/// frees. O(N · k) — fine for k=5 over a 32k-vocab readback in a smoke
/// test, not a hot path.
pub fn topK(gpa: std.mem.Allocator, logits: []const f32, k: usize) ![]TopKEntry {
    const out = try gpa.alloc(TopKEntry, k);
    for (out) |*e| e.* = .{ .id = 0, .value = -std.math.inf(f32) };
    for (logits, 0..) |v, i| {
        if (v <= out[k - 1].value) continue;
        // Insert into sorted (descending) list.
        var j: usize = k - 1;
        out[j] = .{ .id = i, .value = v };
        while (j > 0 and out[j].value > out[j - 1].value) : (j -= 1) {
            const tmp = out[j];
            out[j] = out[j - 1];
            out[j - 1] = tmp;
        }
    }
    return out;
}

/// Single-position attention dispatch push-constant. Defined here rather
/// than in runtime.zig because main.zig kept it as a one-off extern
/// struct alongside its layer-0 GPU smoke; both gpu_gen.zig and
/// layer_tests.zig reach for the same shape, so it lives here now.
pub const AttnDecodeSinglePush = extern struct { n_heads: u32, heads_per_kv: u32, head_dim: u32 };
