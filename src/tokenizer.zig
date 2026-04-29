//! HuggingFace `tokenizer.json` reader — encode + decode for Gemma-style
//! SentencePiece BPE.
//!
//! Encoding pipeline (matches HF's `tokenizers` library on a Gemma-style
//! tokenizer):
//!
//!   1. Normalize: replace ASCII space with U+2581 ("▁"). No NFKC, no
//!      lowercasing — Gemma's normalizer is just the space replace.
//!
//!   2. Initial split: each Unicode codepoint becomes its own atom. So
//!      "▁hello" → ["▁", "h", "e", "l", "l", "o"]. Atoms hold UTF-8
//!      byte slices into the normalized buffer.
//!
//!   3. BPE merges: repeatedly find the adjacent pair with the lowest
//!      rank (= earliest in `model.merges`), merge them into one atom,
//!      stop when no adjacent pair appears in the merge map. Naive
//!      O(n²) per encode — fine for chat-length prompts; switch to a
//!      doubly-linked-list + heap if we ever feed in long documents.
//!
//!   4. Vocab lookup: each remaining atom is looked up by string. If
//!      it isn't in vocab, byte-fallback to the per-byte `<0xXX>`
//!      tokens (Gemma's vocab includes all 256). The byte-fallback path
//!      handles characters that didn't reach a vocab entry through
//!      merges (rare characters, pure punctuation outside training
//!      distribution, etc.).
//!
//! BOS prepending and chat templating live at the call site, NOT here.
//! That keeps the encoder a pure text→ids function and lets the chat
//! loop compose `[bos, sot, ...encode(text), eot, ...]` explicitly.

const std = @import("std");

pub const Tokenizer = struct {
    /// Owns every string in `id_to_str`, the merge keys, and the str→id
    /// hashmap keys. One arena, freed wholesale on deinit.
    arena: std.heap.ArenaAllocator,
    /// id → string. Length is one past the largest known token id;
    /// gaps are null.
    id_to_str: []?[]const u8,
    /// string → id. Built from `model.vocab` (and overridden by
    /// `added_tokens` for special-token IDs).
    str_to_id: std.StringHashMap(u32),
    /// Merge rules: "left RIGHT" (with an ASCII space delimiter) → rank.
    /// Lower rank = earlier in the merges list = higher priority. A
    /// 580k-entry hashmap for Gemma; ~25 MiB with overhead but the
    /// lookups are O(1) and we hit them ~prompt_len² times per encode.
    merges: std.StringHashMap(u32),
    /// byte_fallback[b] = id of the `<0xBB>` token for byte b. -1 if
    /// the tokenizer doesn't have byte fallback (e.g. some non-Gemma
    /// SentencePiece variants), in which case OOV atoms map to `unk`.
    byte_fallback: [256]i32,
    unk_id: ?u32,

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
        defer parsed.deinit();
        if (parsed.value != .object) return error.TokenizerNotObject;

        const model_v = parsed.value.object.get("model") orelse return error.MissingModel;
        if (model_v != .object) return error.MalformedModel;
        const vocab_v = model_v.object.get("vocab") orelse return error.MissingVocab;
        if (vocab_v != .object) return error.MalformedVocab;
        const merges_v = model_v.object.get("merges") orelse return error.MissingMerges;
        if (merges_v != .array) return error.MalformedMerges;

        // ── Pass 1: find max id ─────────────────────────────────────
        var max_id: i64 = -1;
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

        // ── Pass 2: id → string + string → id ───────────────────────
        const id_to_str = try a.alloc(?[]const u8, n);
        for (id_to_str) |*slot| slot.* = null;

        var str_to_id = std.StringHashMap(u32).init(gpa);
        errdefer str_to_id.deinit();

        vocab_it = vocab_v.object.iterator();
        while (vocab_it.next()) |e| {
            if (e.value_ptr.* != .integer) continue;
            const id: u32 = @intCast(e.value_ptr.integer);
            if (id >= n) continue;
            const owned = try a.dupe(u8, e.key_ptr.*);
            id_to_str[id] = owned;
            try str_to_id.put(owned, id);
        }

        // added_tokens override the vocab entry with the same id (the
        // canonical surface form lives in added_tokens for these).
        if (added_v != null and added_v.? == .array) {
            for (added_v.?.array.items) |entry| {
                if (entry != .object) continue;
                const id_v = entry.object.get("id") orelse continue;
                const content_v = entry.object.get("content") orelse continue;
                if (id_v != .integer or content_v != .string) continue;
                const id: u32 = @intCast(id_v.integer);
                if (id >= n) continue;
                const owned = try a.dupe(u8, content_v.string);
                id_to_str[id] = owned;
                try str_to_id.put(owned, id);
            }
        }

        // ── Pass 3: merges → hashmap with rank ─────────────────────
        // The merge string is "left RIGHT" with one ASCII space as the
        // delimiter. Spaces never appear *inside* a token (they were
        // normalized to ▁ before BPE), so split on the first 0x20.
        var merges = std.StringHashMap(u32).init(gpa);
        errdefer merges.deinit();
        try merges.ensureTotalCapacity(@intCast(merges_v.array.items.len));

        for (merges_v.array.items, 0..) |item, rank| {
            if (item != .string) continue;
            // We don't actually need to split — we use the raw string
            // (with its internal 0x20) as the key, and the encoder
            // builds the same "left RIGHT" key when looking up. Saves
            // a fragmentation pass at load time.
            const owned = try a.dupe(u8, item.string);
            try merges.put(owned, @intCast(rank));
        }

        // ── Byte fallback table ────────────────────────────────────
        var byte_fallback: [256]i32 = undefined;
        for (&byte_fallback) |*slot| slot.* = -1;
        var byte_buf: [8]u8 = undefined;
        for (0..256) |b| {
            const name = std.fmt.bufPrint(&byte_buf, "<0x{X:0>2}>", .{b}) catch unreachable;
            if (str_to_id.get(name)) |id| byte_fallback[b] = @intCast(id);
        }

        const unk_id: ?u32 = blk: {
            const u = model_v.object.get("unk_token") orelse break :blk null;
            if (u != .string) break :blk null;
            break :blk str_to_id.get(u.string);
        };

        return .{
            .arena = arena,
            .id_to_str = id_to_str,
            .str_to_id = str_to_id,
            .merges = merges,
            .byte_fallback = byte_fallback,
            .unk_id = unk_id,
        };
    }

    pub fn deinit(self: *Tokenizer) void {
        self.merges.deinit();
        self.str_to_id.deinit();
        self.arena.deinit();
    }

    pub fn decode(self: *const Tokenizer, id: usize) ?[]const u8 {
        if (id >= self.id_to_str.len) return null;
        return self.id_to_str[id];
    }

    pub fn vocabSize(self: *const Tokenizer) usize {
        return self.id_to_str.len;
    }

    /// Returns the id of `name` (e.g. `"<bos>"`, `"<start_of_turn>"`)
    /// or null if it isn't in the vocab. Used by the chat layer to
    /// compose token sequences with explicit specials, instead of
    /// trying to BPE-encode literal `<bos>` strings.
    pub fn specialTokenId(self: *const Tokenizer, name: []const u8) ?u32 {
        return self.str_to_id.get(name);
    }

    /// Encode `text` to token ids using the BPE rules. Caller owns the
    /// returned slice. No BOS is prepended; callers add it if they
    /// want one (Gemma's chat template inserts BOS at a specific spot
    /// in a larger sequence anyway).
    pub fn encode(self: *const Tokenizer, gpa: std.mem.Allocator, text: []const u8) ![]u32 {
        // ── Normalize: ' ' → '▁' (3 bytes) ──────────────────────────
        // We allocate at the upper bound (3× input) and fill linearly.
        const norm = try gpa.alloc(u8, text.len * 3);
        defer gpa.free(norm);
        var ni: usize = 0;
        for (text) |b| {
            if (b == ' ') {
                norm[ni + 0] = 0xE2;
                norm[ni + 1] = 0x96;
                norm[ni + 2] = 0x81;
                ni += 3;
            } else {
                norm[ni] = b;
                ni += 1;
            }
        }
        const normalized = norm[0..ni];

        // ── Initial atoms: one Unicode codepoint each ───────────────
        var atoms = std.ArrayList([]const u8).init(gpa);
        defer atoms.deinit();
        var i: usize = 0;
        while (i < normalized.len) {
            const len = std.unicode.utf8ByteSequenceLength(normalized[i]) catch 1;
            const end = @min(i + len, normalized.len);
            try atoms.append(normalized[i..end]);
            i = end;
        }

        // ── BPE merge loop ──────────────────────────────────────────
        // Pair-key buffer: sized for any reasonable merge. The longest
        // Gemma merge we've sampled is ~30 chars; 512 leaves room.
        var key_buf: [512]u8 = undefined;
        while (atoms.items.len >= 2) {
            var best_rank: u32 = std.math.maxInt(u32);
            var best_idx: usize = std.math.maxInt(usize);

            for (0..atoms.items.len - 1) |idx| {
                const left = atoms.items[idx];
                const right = atoms.items[idx + 1];
                const total = left.len + 1 + right.len;
                if (total > key_buf.len) continue;
                @memcpy(key_buf[0..left.len], left);
                key_buf[left.len] = ' ';
                @memcpy(key_buf[left.len + 1 .. left.len + 1 + right.len], right);
                const key = key_buf[0..total];
                if (self.merges.get(key)) |rank| {
                    if (rank < best_rank) {
                        best_rank = rank;
                        best_idx = idx;
                    }
                }
            }

            if (best_idx == std.math.maxInt(usize)) break;

            // Merge atoms[best_idx] and atoms[best_idx + 1]. The merged
            // bytes are already contiguous in `normalized` because the
            // initial atoms were sequential codepoint slices and every
            // merge is "two adjacent slices into the same buffer", so
            // we can extend the left slice in place rather than allocate.
            const left = atoms.items[best_idx];
            const right = atoms.items[best_idx + 1];
            const merged_ptr = left.ptr;
            const merged_len = left.len + right.len;
            atoms.items[best_idx] = merged_ptr[0..merged_len];
            _ = atoms.orderedRemove(best_idx + 1);
        }

        // ── Vocab lookup with byte fallback ─────────────────────────
        var ids = std.ArrayList(u32).init(gpa);
        errdefer ids.deinit();
        for (atoms.items) |atom| {
            if (self.str_to_id.get(atom)) |id| {
                try ids.append(id);
            } else {
                for (atom) |b| {
                    const fb = self.byte_fallback[b];
                    if (fb >= 0) {
                        try ids.append(@intCast(fb));
                    } else if (self.unk_id) |unk| {
                        try ids.append(unk);
                    } else {
                        return error.NoFallbackForByte;
                    }
                }
            }
        }
        return ids.toOwnedSlice();
    }
};
