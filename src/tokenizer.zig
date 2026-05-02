//! HuggingFace `tokenizer.json` reader — encode + decode for two
//! tokenizer styles:
//!
//!   - **Gemma-style SentencePiece BPE** (`.sentencepiece` mode): atoms
//!     are individual Unicode codepoints after `' '` → `▁` normalization;
//!     OOV atoms fall back to the per-byte `<0xXX>` tokens.
//!
//!   - **GPT-2 / Qwen3-style byte-level BPE** (`.bytelevel` mode): the
//!     input is regex-split (GPT-2 contraction-aware regex), each byte
//!     of each chunk is mapped through the standard 256-entry
//!     "bytes_to_unicode" table to a printable codepoint, and BPE
//!     merges happen on those mapped strings. Decode reverses the byte
//!     mapping. We implement an ASCII-only approximation of the regex
//!     (\p{L}, \p{N}, \s recognised on ASCII only) — sufficient for the
//!     English chat we ship; non-ASCII text will round-trip through the
//!     byte-fallback path with a slightly different segmentation than HF.
//!
//! The mode is detected automatically at load time from the
//! `pre_tokenizer` field of `tokenizer.json`.
//!
//! BOS prepending and chat templating live at the call site, NOT here.
//! That keeps the encoder a pure text→ids function and lets the chat
//! loop compose `[bos, sot, ...encode(text), eot, ...]` explicitly.

const std = @import("std");

pub const Mode = enum {
    /// Gemma 1 / Gemma 2: ' '→'▁' normalize, codepoint atoms, byte
    /// fallback through `<0xXX>` tokens.
    sentencepiece,
    /// GPT-2 / Qwen3 / Llama 3: byte→unicode-char mapping then BPE on
    /// mapped strings. ByteLevel decoder reverses the mapping.
    bytelevel,
};

pub const Tokenizer = struct {
    mode: Mode,
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
    /// .bytelevel only: byte b → UTF-8 string of its mapped codepoint.
    /// Built from the standard GPT-2 bytes_to_unicode mapping.
    byte_to_char: [256][4]u8,
    /// Length in bytes of each entry in `byte_to_char` (1, 2, or 3 for
    /// the codepoints actually used by the standard mapping).
    byte_to_char_len: [256]u8,
    /// .bytelevel only: codepoint (encoded as UTF-8 in the BPE atom) → byte.
    /// Used for decoding. Indexed by the first byte of the UTF-8 sequence
    /// for ASCII-mapped codepoints, but for the U+0100..U+0142 range we
    /// need a hashmap because they share leading bytes (0xC4 / 0xC5).
    char_to_byte: std.StringHashMap(u8),

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

        // Detect mode by inspecting the pre_tokenizer. The Sequence
        // wrapper (Qwen3, Llama3) holds a list with a Split + ByteLevel;
        // a bare ByteLevel is also seen on simpler GPT-2-style configs.
        // Anything else (or absent) → SentencePiece (Gemma).
        const mode: Mode = blk: {
            const pt = parsed.value.object.get("pre_tokenizer") orelse break :blk .sentencepiece;
            if (pt != .object) break :blk .sentencepiece;
            const ty = pt.object.get("type") orelse break :blk .sentencepiece;
            if (ty != .string) break :blk .sentencepiece;
            if (std.mem.eql(u8, ty.string, "ByteLevel")) break :blk .bytelevel;
            if (std.mem.eql(u8, ty.string, "Sequence")) {
                if (pt.object.get("pretokenizers")) |sub| {
                    if (sub == .array) {
                        for (sub.array.items) |child| {
                            if (child != .object) continue;
                            const cty = child.object.get("type") orelse continue;
                            if (cty == .string and std.mem.eql(u8, cty.string, "ByteLevel")) {
                                break :blk .bytelevel;
                            }
                        }
                    }
                }
            }
            break :blk .sentencepiece;
        };

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
            // Two formats accepted:
            //   "left RIGHT"               (Gemma-style: single string)
            //   ["left", "right"]          (Qwen3 / Llama3-style array)
            // Both produce the same key shape ("left" + 0x20 + "right")
            // so the encoder's lookup path is identical.
            switch (item) {
                .string => {
                    const owned = try a.dupe(u8, item.string);
                    try merges.put(owned, @intCast(rank));
                },
                .array => {
                    if (item.array.items.len != 2) continue;
                    if (item.array.items[0] != .string or item.array.items[1] != .string) continue;
                    const left = item.array.items[0].string;
                    const right = item.array.items[1].string;
                    const owned = try a.alloc(u8, left.len + 1 + right.len);
                    @memcpy(owned[0..left.len], left);
                    owned[left.len] = ' ';
                    @memcpy(owned[left.len + 1 ..], right);
                    try merges.put(owned, @intCast(rank));
                },
                else => continue,
            }
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

        // ── Bytes-to-unicode mapping (.bytelevel only) ─────────────
        // Standard GPT-2 mapping: bytes 33..126, 161..172, 174..255 keep
        // their own UTF-8 encoding (printable ASCII + Latin-1 supplement
        // minus the soft-hyphen). The other 68 bytes get U+0100..U+0143
        // (encoded in UTF-8 as 2 bytes 0xC4 0x80 .. 0xC5 0x83).
        var byte_to_char: [256][4]u8 = undefined;
        var byte_to_char_len: [256]u8 = undefined;
        var char_to_byte = std.StringHashMap(u8).init(gpa);
        errdefer char_to_byte.deinit();
        if (mode == .bytelevel) {
            var next_extra: u21 = 0x100;
            for (0..256) |b_usize| {
                const b: u8 = @intCast(b_usize);
                const printable: bool = (b >= 33 and b <= 126) or
                    (b >= 161 and b <= 172) or
                    (b >= 174 and b <= 255);
                const cp: u21 = if (printable) b else blk2: {
                    const c = next_extra;
                    next_extra += 1;
                    break :blk2 c;
                };
                const len = std.unicode.utf8Encode(cp, &byte_to_char[b]) catch unreachable;
                byte_to_char_len[b] = @intCast(len);
                const owned = try a.dupe(u8, byte_to_char[b][0..len]);
                try char_to_byte.put(owned, b);
            }
        } else {
            for (&byte_to_char) |*slot| slot.* = .{ 0, 0, 0, 0 };
            for (&byte_to_char_len) |*slot| slot.* = 0;
        }

        return .{
            .mode = mode,
            .arena = arena,
            .id_to_str = id_to_str,
            .str_to_id = str_to_id,
            .merges = merges,
            .byte_fallback = byte_fallback,
            .unk_id = unk_id,
            .byte_to_char = byte_to_char,
            .byte_to_char_len = byte_to_char_len,
            .char_to_byte = char_to_byte,
        };
    }

    pub fn deinit(self: *Tokenizer) void {
        self.merges.deinit();
        self.str_to_id.deinit();
        self.char_to_byte.deinit();
        self.arena.deinit();
    }

    /// Look up a single token id and return its raw vocab string
    /// (without any byte-level reversal). Useful for diagnostics and
    /// for callers that compose specials directly. Stream-printers
    /// should use `decodeByteLevel` (or `decodeStream`) when running
    /// on a `.bytelevel` tokenizer.
    pub fn decode(self: *const Tokenizer, id: usize) ?[]const u8 {
        if (id >= self.id_to_str.len) return null;
        return self.id_to_str[id];
    }

    /// Bytelevel decoder: reverse the byte→unicode mapping by walking
    /// the raw vocab string codepoint-by-codepoint and looking each one
    /// up in `char_to_byte`. Returns the original UTF-8 bytes the BPE
    /// atom encoded. Caller owns the returned slice.
    ///
    /// For .sentencepiece mode this is the same as `decode` plus the
    /// '▁' → ' ' substitution; the chat layer's printDecoded already
    /// handles that case, so this path is only ever called in
    /// .bytelevel mode.
    pub fn decodeByteLevel(self: *const Tokenizer, gpa: std.mem.Allocator, id: usize) ![]u8 {
        const s = self.id_to_str[id] orelse return error.UnknownTokenId;
        var out = std.ArrayList(u8).init(gpa);
        errdefer out.deinit();
        var i: usize = 0;
        while (i < s.len) {
            const len = std.unicode.utf8ByteSequenceLength(s[i]) catch 1;
            const end = @min(i + len, s.len);
            const ch = s[i..end];
            if (self.char_to_byte.get(ch)) |b| {
                try out.append(b);
            } else {
                // Special-token surface forms (e.g. "<|im_end|>") aren't
                // in char_to_byte — pass them through verbatim so the
                // caller sees the literal marker.
                try out.appendSlice(ch);
            }
            i = end;
        }
        return out.toOwnedSlice();
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
    ///
    /// Dispatches on the tokenizer mode detected at load time.
    pub fn encode(self: *const Tokenizer, gpa: std.mem.Allocator, text: []const u8) ![]u32 {
        return switch (self.mode) {
            .sentencepiece => self.encodeSentencePiece(gpa, text),
            .bytelevel => self.encodeByteLevel(gpa, text),
        };
    }

    fn encodeSentencePiece(self: *const Tokenizer, gpa: std.mem.Allocator, text: []const u8) ![]u32 {
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

    fn encodeByteLevel(self: *const Tokenizer, gpa: std.mem.Allocator, text: []const u8) ![]u32 {
        // ── Pretokenize: split `text` into chunks per the GPT-2 regex ─
        // ASCII-only approximation — sufficient for English chat. Each
        // chunk is then byte-mapped + BPE-merged independently.
        var chunks = std.ArrayList([]const u8).init(gpa);
        defer chunks.deinit();
        try pretokenizeGptStyle(text, &chunks);

        var ids = std.ArrayList(u32).init(gpa);
        errdefer ids.deinit();

        // Reusable scratch for the byte-mapped chunk.
        var mapped = std.ArrayList(u8).init(gpa);
        defer mapped.deinit();

        // Reusable atom list (pointers into `mapped`).
        var atoms = std.ArrayList([]const u8).init(gpa);
        defer atoms.deinit();

        var key_buf: [512]u8 = undefined;

        for (chunks.items) |chunk| {
            // ── Byte→char map: each byte becomes its UTF-8 mapped char.
            mapped.clearRetainingCapacity();
            for (chunk) |b| {
                const len = self.byte_to_char_len[b];
                try mapped.appendSlice(self.byte_to_char[b][0..len]);
            }

            // ── Initial atoms: one codepoint each. ───────────────────
            atoms.clearRetainingCapacity();
            var i: usize = 0;
            while (i < mapped.items.len) {
                const len = std.unicode.utf8ByteSequenceLength(mapped.items[i]) catch 1;
                const end = @min(i + len, mapped.items.len);
                try atoms.append(mapped.items[i..end]);
                i = end;
            }

            // ── BPE merge loop. ─────────────────────────────────────
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
                const left = atoms.items[best_idx];
                const right = atoms.items[best_idx + 1];
                const merged_len = left.len + right.len;
                atoms.items[best_idx] = left.ptr[0..merged_len];
                _ = atoms.orderedRemove(best_idx + 1);
            }

            // ── Vocab lookup. By construction every mapped char IS in
            // the vocab (byte-level vocabs contain all 256 mapped chars
            // as single-codepoint tokens), so any miss is a real error.
            for (atoms.items) |atom| {
                if (self.str_to_id.get(atom)) |id| {
                    try ids.append(id);
                } else if (self.unk_id) |unk| {
                    try ids.append(unk);
                } else {
                    return error.UnknownAtom;
                }
            }
        }
        return ids.toOwnedSlice();
    }
};

// ── GPT-2 / Qwen3 pretokenizer (ASCII approximation) ────────────────
//
// Implements the regex
//     (?i:'s|'t|'re|'ve|'m|'ll|'d)
//   | [^\r\n\p{L}\p{N}]?\p{L}+
//   | \p{N}
//   |  ?[^\s\p{L}\p{N}]+[\r\n]*
//   | \s*[\r\n]+
//   | \s+(?!\S)
//   | \s+
// with \p{L}, \p{N}, \s narrowed to their ASCII subsets. Each match is
// consumed greedily and emitted as its own chunk; chunks are
// byte-level-mapped and BPE'd independently.
//
// The "isolated" behavior of HF's split means non-matching characters
// are themselves emitted as one-byte chunks (the regex's last
// alternative `\s+` covers spaces, but bytes outside ASCII letter/
// digit/punctuation/whitespace fall through — we emit them as
// single-byte chunks so they round-trip via byte-level mapping).
fn pretokenizeGptStyle(text: []const u8, out: *std.ArrayList([]const u8)) !void {
    var i: usize = 0;
    while (i < text.len) {
        // 1. Contractions (case-insensitive). HF's regex matches these
        //    only if the apostrophe starts the token, so we don't need
        //    to look back. Critical: when the apostrophe is NOT
        //    followed by one of the seven contraction suffixes, we
        //    must fall through to the subsequent rules rather than
        //    `continue` the outer loop without advancing i — the
        //    earlier for-else form did exactly that and infinite-
        //    looped on inputs like `'I realized'` (apostrophe + capital
        //    letter, no contraction match). Rule 2 below will consume
        //    `'I` as a unit (optional non-LN char + letter run), which
        //    is what HF's regex engine does.
        if (text[i] == '\'') {
            const rest = text[i + 1 ..];
            const cs = [_][]const u8{ "s", "t", "re", "ve", "m", "ll", "d" };
            var matched: bool = false;
            for (cs) |c| {
                if (rest.len >= c.len and asciiCaseEq(rest[0..c.len], c)) {
                    try out.append(text[i .. i + 1 + c.len]);
                    i += 1 + c.len;
                    matched = true;
                    break;
                }
            }
            if (matched) continue;
            // No contraction — fall through to rules 2+.
        }

        // 2. Optional non-LN char + letters: `[^\r\n\p{L}\p{N}]?\p{L}+`.
        //    The leading char is most commonly a single space.
        {
            const start = i;
            var j = i;
            if (j < text.len) {
                const c = text[j];
                const is_letter = isAsciiLetter(c);
                const is_digit = isAsciiDigit(c);
                const is_lf = (c == '\r' or c == '\n');
                if (!is_letter and !is_digit and !is_lf) {
                    j += 1;
                }
            }
            // Need at least one letter after the optional char.
            if (j < text.len and isAsciiLetter(text[j])) {
                while (j < text.len and isAsciiLetter(text[j])) j += 1;
                try out.append(text[start..j]);
                i = j;
                continue;
            }
            // No match — fall through.
        }

        // 3. Numbers: \p{N}, single digit per chunk (no plus quantifier
        //    in the regex).
        if (isAsciiDigit(text[i])) {
            try out.append(text[i .. i + 1]);
            i += 1;
            continue;
        }

        // 4. Optional space + punctuation/symbol run + optional newlines:
        //    ` ?[^\s\p{L}\p{N}]+[\r\n]*`.
        {
            const start = i;
            var j = i;
            if (text[j] == ' ') j += 1;
            const punct_start = j;
            while (j < text.len) {
                const c = text[j];
                if (isAsciiSpace(c) or isAsciiLetter(c) or isAsciiDigit(c)) break;
                j += 1;
            }
            if (j > punct_start) {
                while (j < text.len and (text[j] == '\r' or text[j] == '\n')) j += 1;
                try out.append(text[start..j]);
                i = j;
                continue;
            }
            // No punct — fall through (no consumed bytes).
        }

        // 5. Whitespace + newline run: `\s*[\r\n]+`.
        {
            const start = i;
            var j = i;
            while (j < text.len and isAsciiSpace(text[j]) and text[j] != '\r' and text[j] != '\n') j += 1;
            const ws_end = j;
            while (j < text.len and (text[j] == '\r' or text[j] == '\n')) j += 1;
            if (j > ws_end) {
                try out.append(text[start..j]);
                i = j;
                continue;
            }
        }

        // 6. Trailing whitespace (\s+ followed by non-space): we
        //    approximate as "all remaining whitespace minus the last
        //    one if a non-space follows."
        if (isAsciiSpace(text[i])) {
            var j = i;
            while (j < text.len and isAsciiSpace(text[j])) j += 1;
            // If we consumed everything, this is just \s+ (case 7).
            // Otherwise leave the last whitespace behind to be paired
            // with the following non-space chunk via case 4's leading
            // space.
            if (j < text.len and j - i >= 1) {
                if (j - i > 1) {
                    try out.append(text[i .. j - 1]);
                    i = j - 1;
                } else {
                    try out.append(text[i..j]);
                    i = j;
                }
            } else {
                try out.append(text[i..j]);
                i = j;
            }
            continue;
        }

        // 7. Anything else (non-ASCII byte etc.): single-byte chunk.
        try out.append(text[i .. i + 1]);
        i += 1;
    }
}

inline fn isAsciiLetter(b: u8) bool {
    return (b >= 'A' and b <= 'Z') or (b >= 'a' and b <= 'z');
}
inline fn isAsciiDigit(b: u8) bool {
    return b >= '0' and b <= '9';
}
inline fn isAsciiSpace(b: u8) bool {
    return b == ' ' or b == '\t' or b == '\r' or b == '\n' or b == 0x0B or b == 0x0C;
}
fn asciiCaseEq(a: []const u8, b: []const u8) bool {
    if (a.len != b.len) return false;
    for (a, b) |x, y| {
        const xl = if (x >= 'A' and x <= 'Z') x + 32 else x;
        const yl = if (y >= 'A' and y <= 'Z') y + 32 else y;
        if (xl != yl) return false;
    }
    return true;
}
