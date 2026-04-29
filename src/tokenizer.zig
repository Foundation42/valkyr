//! HuggingFace `tokenizer.json` decoder.
//!
//! This is a *decode-only* tokenizer for now: id → string. We don't
//! implement encoding (BPE merge resolution) yet because the first
//! milestone — "what does the model say after BOS?" — only needs to
//! reverse-look-up the sampled token. Encoding is a separate, larger
//! problem that arrives when we want to chat with the model from a
//! string prompt.
//!
//! Format used: `model.vocab` is a `{ "string": id }` dict — we invert
//! it into an array indexed by id. `added_tokens` is a separate list
//! of `{id, content, ...}` objects (Gemma uses this for `<bos>`-style
//! special tokens that need ID reservation). We fold those into the
//! array too; `added_tokens` IDs may exceed the base `model.vocab`
//! range, so we size the array to `max_id + 1` rather than to
//! `vocab.len`.

const std = @import("std");

pub const Tokenizer = struct {
    /// Owns every string in `id_to_str`.
    arena: std.heap.ArenaAllocator,
    /// Length is one past the largest known token id. Slots without a
    /// known token (gaps in the id space) are null.
    id_to_str: []?[]const u8,

    pub fn loadFromFile(gpa: std.mem.Allocator, path: []const u8) !Tokenizer {
        const file = try std.fs.cwd().openFile(path, .{ .mode = .read_only });
        defer file.close();
        // tokenizer.json for Gemma is ~17 MiB. 64 MiB cap leaves room
        // for the unlikely "huge tokenizer" case without going wild.
        const bytes = try file.readToEndAlloc(gpa, 64 * 1024 * 1024);
        defer gpa.free(bytes);
        return parseFromSlice(gpa, bytes);
    }

    pub fn parseFromSlice(gpa: std.mem.Allocator, json: []const u8) !Tokenizer {
        var arena = std.heap.ArenaAllocator.init(gpa);
        errdefer arena.deinit();
        const a = arena.allocator();

        var parsed = try std.json.parseFromSlice(std.json.Value, a, json, .{});
        // We deliberately don't `parsed.deinit()` here — we copy strings
        // we keep into the arena before the function returns, so the
        // parse tree's transient allocations are fine to free at scope
        // exit via the temporary allocator we hand to parseFromSlice.
        // But std.json's tree itself is reusing the arena, so it's all
        // collected together when the arena dies. Save the cleanup.
        defer parsed.deinit();
        if (parsed.value != .object) return error.TokenizerNotObject;

        // Two passes:
        // 1. Find the largest id across vocab + added_tokens to size
        //    the result array.
        // 2. Fill the array.
        var max_id: i64 = -1;

        const model_v = parsed.value.object.get("model") orelse return error.MissingModel;
        if (model_v != .object) return error.MalformedModel;
        const vocab_v = model_v.object.get("vocab") orelse return error.MissingVocab;
        if (vocab_v != .object) return error.MalformedVocab;

        var vocab_it = vocab_v.object.iterator();
        while (vocab_it.next()) |e| {
            if (e.value_ptr.* != .integer) continue;
            if (e.value_ptr.integer > max_id) max_id = e.value_ptr.integer;
        }

        const added_v = parsed.value.object.get("added_tokens");
        if (added_v != null and added_v.? == .array) {
            for (added_v.?.array.items) |entry| {
                if (entry != .object) continue;
                const id_v = entry.object.get("id") orelse continue;
                if (id_v != .integer) continue;
                if (id_v.integer > max_id) max_id = id_v.integer;
            }
        }

        if (max_id < 0) return error.EmptyVocab;
        const n: usize = @intCast(max_id + 1);

        const id_to_str = try a.alloc(?[]const u8, n);
        for (id_to_str) |*slot| slot.* = null;

        // Pass 2 — vocab.
        vocab_it = vocab_v.object.iterator();
        while (vocab_it.next()) |e| {
            if (e.value_ptr.* != .integer) continue;
            const id: usize = @intCast(e.value_ptr.integer);
            if (id >= n) continue;
            id_to_str[id] = try a.dupe(u8, e.key_ptr.*);
        }

        // Pass 2 — added_tokens (override any vocab entry with the same
        // id, since the added_tokens list is the authoritative source
        // for special-token strings).
        if (added_v != null and added_v.? == .array) {
            for (added_v.?.array.items) |entry| {
                if (entry != .object) continue;
                const id_v = entry.object.get("id") orelse continue;
                const content_v = entry.object.get("content") orelse continue;
                if (id_v != .integer or content_v != .string) continue;
                const id: usize = @intCast(id_v.integer);
                if (id >= n) continue;
                id_to_str[id] = try a.dupe(u8, content_v.string);
            }
        }

        return .{
            .arena = arena,
            .id_to_str = id_to_str,
        };
    }

    pub fn deinit(self: *Tokenizer) void {
        self.arena.deinit();
    }

    pub fn decode(self: *const Tokenizer, id: usize) ?[]const u8 {
        if (id >= self.id_to_str.len) return null;
        return self.id_to_str[id];
    }

    pub fn vocabSize(self: *const Tokenizer) usize {
        return self.id_to_str.len;
    }
};
