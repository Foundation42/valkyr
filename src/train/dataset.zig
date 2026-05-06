//! Tiny training dataset abstraction — packed token stream + sliding
//! n_pos+1 windows for next-token-prediction.
//!
//! Layout:
//!
//!     packed_ids = enc(ex_0) ⊕ [eos] ⊕ enc(ex_1) ⊕ [eos] ⊕ ... ⊕ enc(ex_K-1)
//!
//! `batch(idx, input_ids, target_ids)` reads the (n_pos+1)-element
//! window starting at `packed_ids[idx]` and emits:
//!
//!     input_ids  = packed_ids[idx     .. idx + n_pos]
//!     target_ids = packed_ids[idx + 1 .. idx + n_pos + 1]
//!
//! That's the standard causal-LM training shape: at training position
//! p the model sees `input_ids[p]` (and via attention everything to its
//! left) and must predict `target_ids[p] == input_ids[p+1]`. Loss is
//! summed over all n_pos positions; we don't (yet) loss-mask prompts
//! vs responses — that's a follow-up once instruction tuning kicks in.
//!
//! EOS separators between examples teach the model where one fact ends
//! and the next begins. Without them, packed-sequence training would
//! reward predicting "Paris.The" because position p saw the previous
//! fact. With EOS the previous-fact context terminates cleanly; the
//! model just learns to start fresh after `<eos>`.

const std = @import("std");
const tokenizer_mod = @import("../tokenizer.zig");

pub const Dataset = struct {
    allocator: std.mem.Allocator,
    /// All examples tokenized + separated by `eos_id`. Length is the
    /// sum of (encoded_len[i] + 1) for each example, minus 1 (no
    /// trailing EOS — the last example ends naturally).
    packed_ids: []u32,
    /// Window size: each batch reads `packed_ids[idx .. idx + n_pos + 1]`,
    /// emits input=window[0..n_pos], target=window[1..n_pos+1].
    n_pos: u32,

    pub fn deinit(self: *Dataset) void {
        self.allocator.free(self.packed_ids);
    }

    /// Number of (input, target) batches. Equals
    /// `packed_ids.len - n_pos` if the stream is at least n_pos+1
    /// tokens long; 0 otherwise.
    pub fn numBatches(self: *const Dataset) usize {
        if (self.packed_ids.len <= self.n_pos) return 0;
        return self.packed_ids.len - self.n_pos;
    }

    /// Fill `input_ids` and `target_ids` for batch `idx`. Both slices
    /// must have length `n_pos`.
    pub fn batch(
        self: *const Dataset,
        idx: usize,
        input_ids: []u32,
        target_ids: []u32,
    ) !void {
        if (input_ids.len != self.n_pos) return error.InputIdsLen;
        if (target_ids.len != self.n_pos) return error.TargetIdsLen;
        if (idx >= self.numBatches()) return error.BatchIndexOutOfRange;
        const start = idx;
        const n_pos: usize = self.n_pos;
        for (0..n_pos) |p| {
            input_ids[p] = self.packed_ids[start + p];
            target_ids[p] = self.packed_ids[start + p + 1];
        }
    }
};

/// Build a Dataset from in-memory text examples. Each is tokenized via
/// `tok.encode` and appended to a single packed stream with `eos_id`
/// inserted between examples. No EOS at the very end — keeps the
/// stream length as small as possible without losing any boundary info
/// (windows that span the last example just stop at its real end).
pub fn buildFromExamples(
    allocator: std.mem.Allocator,
    tok: *const tokenizer_mod.Tokenizer,
    examples: []const []const u8,
    n_pos: u32,
    eos_id: u32,
) !Dataset {
    var packed_list = std.ArrayList(u32).init(allocator);
    errdefer packed_list.deinit();

    for (examples, 0..) |text, ei| {
        const ids = try tok.encode(allocator, text);
        defer allocator.free(ids);
        try packed_list.appendSlice(ids);
        if (ei + 1 < examples.len) try packed_list.append(eos_id);
    }

    return .{
        .allocator = allocator,
        .packed_ids = try packed_list.toOwnedSlice(),
        .n_pos = n_pos,
    };
}

/// Build a Dataset from a JSONL file. Each line should be a JSON
/// object with a `"text"` string field; other fields and lines without
/// `"text"` are ignored. Empty / whitespace-only lines are skipped.
/// Returns `error.EmptyDataset` if no usable lines were found.
pub fn buildFromJsonl(
    allocator: std.mem.Allocator,
    tok: *const tokenizer_mod.Tokenizer,
    path: []const u8,
    n_pos: u32,
    eos_id: u32,
) !Dataset {
    const file = try std.fs.cwd().openFile(path, .{});
    defer file.close();
    const stat = try file.stat();
    const buf = try allocator.alloc(u8, stat.size);
    defer allocator.free(buf);
    _ = try file.readAll(buf);

    var packed_list = std.ArrayList(u32).init(allocator);
    errdefer packed_list.deinit();

    var line_count: usize = 0;
    var it = std.mem.splitScalar(u8, buf, '\n');
    while (it.next()) |line_raw| {
        const line = std.mem.trim(u8, line_raw, " \t\r");
        if (line.len == 0) continue;

        var parsed = std.json.parseFromSlice(std.json.Value, allocator, line, .{}) catch continue;
        defer parsed.deinit();
        if (parsed.value != .object) continue;
        const text_val = parsed.value.object.get("text") orelse continue;
        if (text_val != .string) continue;

        const ids = try tok.encode(allocator, text_val.string);
        defer allocator.free(ids);

        if (line_count > 0) try packed_list.append(eos_id);
        try packed_list.appendSlice(ids);
        line_count += 1;
    }

    if (line_count == 0) return error.EmptyDataset;

    return .{
        .allocator = allocator,
        .packed_ids = try packed_list.toOwnedSlice(),
        .n_pos = n_pos,
    };
}
