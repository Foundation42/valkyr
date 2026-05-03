//! Per-family chat templates — turns OpenAI-shaped `[{role, content}]`
//! arrays (or single user messages, for the legacy CLI hot path) into
//! token sequences that match each family's HF reference template.
//!
//! Lifted out of main.zig so both the CLI REPL and embed callers
//! (matryoshka NPCs, future `valkyr --serve`) share one composer.
//!
//! Different model families serialize a chat turn into very different
//! token sequences:
//!
//!   Gemma (`<bos>` + `<start_of_turn>` markers):
//!     <bos><start_of_turn>user\n{msg}<end_of_turn>\n
//!          <start_of_turn>model\n
//!     stop on <end_of_turn>. No system role — system content folds
//!     into the first user turn.
//!
//!   Qwen3 / Qwen3.5 (ChatML, `<|im_start|>` / `<|im_end|>` markers):
//!     <|im_start|>user\n{msg}<|im_end|>\n
//!     <|im_start|>assistant\n
//!     stop on <|im_end|>. System role has its own `system` turn.
//!
//!   Llama 3 (Meta's header-id format):
//!     <|begin_of_text|>
//!     <|start_header_id|>user<|end_header_id|>\n\n{msg}<|eot_id|>
//!     <|start_header_id|>assistant<|end_header_id|>\n\n
//!     stop on <|eot_id|>. System role has its own `system` turn.
//!
//!   Zephyr (TinyLlama / Llama 2-arch chat fine-tunes that ship without
//!   Llama 3 header_id specials):
//!     <s><|user|>\n{msg}</s>\n<|assistant|>\n
//!     stop on </s>. Markers `<|user|>` / `<|assistant|>` are TEXT
//!     (BPE-encoded), not special tokens — only `<s>` / `</s>` are.
//!     System role uses `<|system|>` text marker.
//!
//!   Mistral / Ministral ([INST] / [/INST] format, v0.3+):
//!     <s>[INST]{msg}[/INST]
//!     stop on </s>. `[INST]` and `[/INST]` are special tokens in
//!     Mistral 7B v0.3+ and Ministral. No system role — system
//!     content folds into the first user turn inside `[INST]`.
//!
//! Family.llama auto-detects between Llama 3, Mistral, and Zephyr by
//! checking which specials exist in the tokenizer.

const std = @import("std");
const config_mod = @import("config.zig");
const tokenizer_mod = @import("tokenizer.zig");

pub const Role = enum { system, user, assistant };

pub const Message = struct {
    role: Role,
    content: []const u8,
};

pub const ChatTemplate = struct {
    family: config_mod.Family,
    format: Format,
    /// First-conversation-turn marker. Gemma / Llama 3 / Zephyr emit
    /// one; Qwen3 doesn't.
    bos: ?u32,
    /// Per-turn opening marker (`<start_of_turn>` / `<|im_start|>` /
    /// `<|start_header_id|>`). `null` in Zephyr where the role marker
    /// is text (`<|user|>` / `<|assistant|>`), not a special token.
    start_of_turn: ?u32,
    /// Per-turn closing marker (`<end_of_turn>` / `<|im_end|>` /
    /// `<|eot_id|>` / `</s>`). Also the stop token the sampler
    /// watches for.
    end_of_turn: u32,
    /// Llama 3's `<|end_header_id|>` token. `null` for other formats.
    end_header: ?u32,
    /// Role prefix typed BY the model in its response prefix
    /// (`"model"` for Gemma, `"assistant"` for Qwen3 / Llama 3,
    /// `"<|assistant|>"` text marker for Zephyr).
    assistant_role: []const u8,
    /// User-turn role text (`"user"` for most; `"<|user|>"` for Zephyr).
    user_role: []const u8,
    /// System-turn role text. `"system"` for most; `"<|system|>"` for
    /// Zephyr. Unused in formats that fold system into user
    /// (.gemma_qwen for gemma; .mistral).
    system_role: []const u8,
    /// Separator between the role-header section and content. `\n` for
    /// Gemma / Qwen3, `\n\n` for Llama 3, `\n` for Zephyr.
    header_sep: []const u8,
    /// Separator between turns (after closing marker, before next
    /// opening). `\n` for Gemma / Qwen3 / Zephyr; empty for Llama 3.
    inter_turn_sep: []const u8,
    /// Mistral's `[INST]` token (set only in `.mistral` format).
    inst_open: ?u32 = null,
    /// Mistral's `[/INST]` token (set only in `.mistral` format).
    inst_close: ?u32 = null,

    pub const Format = enum {
        /// Gemma + Qwen3/3.5 share the `<sot>role\n{msg}<eot>\n` shape.
        gemma_qwen,
        /// Llama 3's `<sot>role<eoh>\n\n{msg}<eot>` shape.
        llama3,
        /// Text-marker template used by TinyLlama and Llama 2-arch chat
        /// fine-tunes that don't ship Llama 3 specials. Roles are
        /// encoded as text; only BOS / EOT are special tokens.
        zephyr,
        /// Mistral / Ministral v0.3+ format: `<s>[INST]{msg}[/INST]`
        /// with [INST] and [/INST] as special tokens in the tokenizer.
        mistral,
    };

    pub fn resolve(family: config_mod.Family, tok: *const tokenizer_mod.Tokenizer) !ChatTemplate {
        return switch (family) {
            .gemma => .{
                .family = family,
                .format = .gemma_qwen,
                .bos = tok.specialTokenId("<bos>") orelse return error.NoBos,
                .start_of_turn = tok.specialTokenId("<start_of_turn>") orelse return error.NoStartOfTurn,
                .end_of_turn = tok.specialTokenId("<end_of_turn>") orelse return error.NoEndOfTurn,
                .end_header = null,
                .user_role = "user",
                .assistant_role = "model",
                .system_role = "system", // unused — Gemma folds system into first user
                .header_sep = "\n",
                .inter_turn_sep = "\n",
            },
            .llama => llama: {
                // Auto-detect among the three Llama-arch chat formats:
                //   1. <|start_header_id|> exists → Llama 3 (Meta header-id)
                //   2. [INST] exists → Mistral / Ministral v0.3+
                //   3. else → Zephyr text-marker (TinyLlama-style)
                if (tok.specialTokenId("<|start_header_id|>")) |sot| {
                    break :llama .{
                        .family = family,
                        .format = .llama3,
                        .bos = tok.specialTokenId("<|begin_of_text|>"),
                        .start_of_turn = sot,
                        .end_of_turn = tok.specialTokenId("<|eot_id|>") orelse return error.NoEndOfTurn,
                        .end_header = tok.specialTokenId("<|end_header_id|>") orelse return error.NoEndHeader,
                        .user_role = "user",
                        .assistant_role = "assistant",
                        .system_role = "system",
                        .header_sep = "\n\n",
                        .inter_turn_sep = "",
                    };
                }
                if (tok.specialTokenId("[INST]")) |inst| {
                    break :llama .{
                        .family = family,
                        .format = .mistral,
                        .bos = tok.specialTokenId("<s>"),
                        .start_of_turn = null,
                        .end_of_turn = tok.specialTokenId("</s>") orelse return error.NoEndOfTurn,
                        .end_header = null,
                        .user_role = "user", // unused
                        .assistant_role = "assistant", // unused
                        .system_role = "system", // unused — folds into first user
                        .header_sep = "", // unused
                        .inter_turn_sep = "", // unused
                        .inst_open = inst,
                        .inst_close = tok.specialTokenId("[/INST]") orelse return error.NoInstClose,
                    };
                }
                break :llama .{
                    .family = family,
                    .format = .zephyr,
                    .bos = tok.specialTokenId("<s>"),
                    .start_of_turn = null,
                    .end_of_turn = tok.specialTokenId("</s>") orelse return error.NoEndOfTurn,
                    .end_header = null,
                    .user_role = "<|user|>",
                    .assistant_role = "<|assistant|>",
                    .system_role = "<|system|>",
                    .header_sep = "\n",
                    .inter_turn_sep = "\n",
                };
            },
            .qwen3, .qwen35 => .{
                .family = family,
                .format = .gemma_qwen,
                // Qwen3 / Qwen3.5 chat doesn't prepend BOS; both use the
                // ChatML `<|im_start|>` / `<|im_end|>` markers.
                .bos = null,
                .start_of_turn = tok.specialTokenId("<|im_start|>") orelse return error.NoStartOfTurn,
                .end_of_turn = tok.specialTokenId("<|im_end|>") orelse return error.NoEndOfTurn,
                .end_header = null,
                .user_role = "user",
                .assistant_role = "assistant",
                .system_role = "system",
                .header_sep = "\n",
                .inter_turn_sep = "\n",
            },
        };
    }

    pub fn banner(self: ChatTemplate) []const u8 {
        return switch (self.family) {
            .gemma => "Gemma chat",
            .llama => switch (self.format) {
                .llama3 => "Llama 3 chat",
                .zephyr => "Llama (Zephyr-style) chat",
                .mistral => "Mistral / Ministral chat",
                else => "Llama chat",
            },
            .qwen3 => "Qwen3 chat",
            .qwen35 => "Qwen3.5 chat",
        };
    }

    /// Whether this format supports a separate `system` turn vs folding
    /// system content into the first user turn. Gemma + Mistral fold;
    /// Qwen3 / Qwen3.5 / Llama 3 / Zephyr have explicit system turns.
    fn hasSystemTurn(self: ChatTemplate) bool {
        return switch (self.format) {
            .llama3, .zephyr => true,
            .gemma_qwen => self.family != .gemma,
            .mistral => false,
        };
    }

    // ── Single-turn legacy composer (CLI REPL hot path) ──────────────
    //
    // Identical behavior to pre-lift: emits BOS on the first turn,
    // re-emits `</s>` for Mistral mid-conversation, then one user
    // turn + open assistant header.

    /// Compose a turn's prompt token sequence into `out`. `is_first`
    /// controls BOS emission (Gemma / Llama 3 / Zephyr emit one;
    /// Qwen3 doesn't). Used by the CLI REPL where each turn appends
    /// only the new user message + assistant header onto a warm KV
    /// cache.
    ///
    /// This path is bit-identical to the pre-lift composer: `.zephyr`
    /// in particular uses one combined encode for `\n<|assistant|>\n`
    /// so BPE merges across the inter-turn boundary land identically
    /// to the HF reference. The new `composeConversation` uses
    /// turn-level primitives that may produce slightly different
    /// token IDs at boundaries — fine for new callers, but legacy
    /// REPL stays here.
    pub fn composePrompt(
        self: ChatTemplate,
        gpa: std.mem.Allocator,
        tok: *const tokenizer_mod.Tokenizer,
        user_msg: []const u8,
        is_first: bool,
        out: *std.ArrayList(u32),
    ) !void {
        if (is_first) {
            if (self.bos) |b| try out.append(b);
        } else if (self.format == .mistral) {
            // Mistral multi-turn: canonical format is
            //   <s>[INST]msg1[/INST]reply1</s>[INST]msg2[/INST]
            // The chat loop breaks on </s> without writing it to the
            // KV cache, so we re-emit it here at the start of each
            // non-first turn to close the previous reply.
            try out.append(self.end_of_turn);
        }

        switch (self.format) {
            .gemma_qwen, .llama3 => try self.composeWithSpecials(gpa, tok, user_msg, out),
            .zephyr => try self.composeZephyr(gpa, tok, user_msg, out),
            .mistral => try self.composeMistral(gpa, tok, user_msg, out),
        }
    }

    fn composeWithSpecials(
        self: ChatTemplate,
        gpa: std.mem.Allocator,
        tok: *const tokenizer_mod.Tokenizer,
        user_msg: []const u8,
        out: *std.ArrayList(u32),
    ) !void {
        const sot = self.start_of_turn.?;

        // User turn header
        try out.append(sot);
        try self.appendRoleSection(gpa, tok, self.user_role, out);

        // User message body
        {
            const ids = try tok.encode(gpa, user_msg);
            defer gpa.free(ids);
            try out.appendSlice(ids);
        }
        try out.append(self.end_of_turn);

        // Inter-turn separator
        if (self.inter_turn_sep.len > 0) {
            const ids = try tok.encode(gpa, self.inter_turn_sep);
            defer gpa.free(ids);
            try out.appendSlice(ids);
        }

        // Assistant turn header (no body — generation continues here)
        try out.append(sot);
        try self.appendRoleSection(gpa, tok, self.assistant_role, out);
    }

    fn composeZephyr(
        self: ChatTemplate,
        gpa: std.mem.Allocator,
        tok: *const tokenizer_mod.Tokenizer,
        user_msg: []const u8,
        out: *std.ArrayList(u32),
    ) !void {
        // `<|user|>\n` as one encode so any BPE merges at the role/sep
        // boundary land the same as the HF reference.
        {
            const buf = try std.fmt.allocPrint(gpa, "{s}{s}", .{ self.user_role, self.header_sep });
            defer gpa.free(buf);
            const ids = try tok.encode(gpa, buf);
            defer gpa.free(ids);
            try out.appendSlice(ids);
        }

        // User message body, then `</s>` (special).
        {
            const ids = try tok.encode(gpa, user_msg);
            defer gpa.free(ids);
            try out.appendSlice(ids);
        }
        try out.append(self.end_of_turn);

        // `\n<|assistant|>\n` — inter-turn `\n` + assistant text marker
        // + header_sep, encoded as one string for clean BPE.
        {
            const buf = try std.fmt.allocPrint(gpa, "{s}{s}{s}", .{
                self.inter_turn_sep, self.assistant_role, self.header_sep,
            });
            defer gpa.free(buf);
            const ids = try tok.encode(gpa, buf);
            defer gpa.free(ids);
            try out.appendSlice(ids);
        }
    }

    fn composeMistral(
        self: ChatTemplate,
        gpa: std.mem.Allocator,
        tok: *const tokenizer_mod.Tokenizer,
        user_msg: []const u8,
        out: *std.ArrayList(u32),
    ) !void {
        try out.append(self.inst_open.?);
        {
            const ids = try tok.encode(gpa, user_msg);
            defer gpa.free(ids);
            try out.appendSlice(ids);
        }
        try out.append(self.inst_close.?);
    }

    // ── Multi-turn composer (server, embed appendMessages) ───────────
    //
    // Walks the full message history. Emits BOS once if applicable,
    // role-keyed turns, and a final open assistant header for
    // generation to continue from.

    /// Compose a full conversation into `out`. `messages` may include
    /// system / user / assistant turns in any order; the composer
    /// emits each in this format's role-specific shape, then appends
    /// the open assistant header so generation continues from the
    /// last position.
    ///
    /// Format-specific wrinkles handled here:
    ///   - Gemma + Mistral fold a leading system turn into the first
    ///     user turn (those formats have no system role).
    ///   - Mistral closes each non-final assistant turn with `</s>`.
    ///   - BOS is emitted once at the start (where applicable).
    pub fn composeConversation(
        self: ChatTemplate,
        gpa: std.mem.Allocator,
        tok: *const tokenizer_mod.Tokenizer,
        messages: []const Message,
        out: *std.ArrayList(u32),
    ) !void {
        if (self.bos) |b| try out.append(b);

        // Handle a leading system message for fold-style formats. We
        // strip it from the queue and prepend its content (with a
        // separator) into the first user message we encounter. If
        // there's no following user message, we silently drop the
        // system content — the composer's job is to produce a valid
        // generation prompt, and a system-only conversation isn't
        // one. Hosts that need richer behavior should compose
        // host-side.
        var pending_system: ?[]const u8 = null;
        var i: usize = 0;
        if (!self.hasSystemTurn() and messages.len > 0 and messages[0].role == .system) {
            pending_system = messages[0].content;
            i = 1;
        }

        while (i < messages.len) : (i += 1) {
            const msg = messages[i];
            switch (msg.role) {
                .system => {
                    if (self.hasSystemTurn()) {
                        try self.appendSystemTurn(gpa, tok, msg.content, out);
                        try self.appendInterTurnSep(gpa, tok, out);
                    }
                    // For fold-style formats, system mid-conversation
                    // is dropped. Surfacing this would require a host
                    // signal; not worth the complexity for v0.
                },
                .user => {
                    // Build the effective user content, prepending any
                    // pending system message via a `\n\n` separator
                    // (matches HF reference fold for Gemma + Mistral).
                    var user_text: []const u8 = msg.content;
                    var user_text_owned = false;
                    if (pending_system) |sys| {
                        user_text = try std.fmt.allocPrint(gpa, "{s}\n\n{s}", .{ sys, msg.content });
                        user_text_owned = true;
                        pending_system = null;
                    }
                    defer if (user_text_owned) gpa.free(user_text);

                    switch (self.format) {
                        .gemma_qwen, .llama3 => {
                            try self.appendUserTurnSpecials(gpa, tok, user_text, out);
                            try self.appendInterTurnSep(gpa, tok, out);
                        },
                        .zephyr => {
                            try self.appendUserTurnZephyr(gpa, tok, user_text, out);
                            try self.appendInterTurnSep(gpa, tok, out);
                        },
                        .mistral => {
                            try self.appendUserTurnMistral(gpa, tok, user_text, out);
                        },
                    }
                },
                .assistant => {
                    try self.appendAssistantTurn(gpa, tok, msg.content, out);
                    try self.appendInterTurnSep(gpa, tok, out);
                },
            }
        }

        // Final open assistant header — the body is what generation
        // produces next.
        switch (self.format) {
            .gemma_qwen, .llama3, .zephyr => try self.appendAssistantHeader(gpa, tok, out),
            .mistral => {
                // Mistral has no explicit assistant header; generation
                // continues after the trailing [/INST]. If the last
                // message was already a user, we've already emitted
                // [/INST]. If the last message was assistant, we need
                // to open a fresh [INST]…[/INST] with empty content
                // for a "continue from here" signal — but that's
                // pathological (server should validate). Skip.
            },
        }
    }

    // ── Turn primitives ──────────────────────────────────────────────

    fn appendUserTurnSpecials(
        self: ChatTemplate,
        gpa: std.mem.Allocator,
        tok: *const tokenizer_mod.Tokenizer,
        content: []const u8,
        out: *std.ArrayList(u32),
    ) !void {
        const sot = self.start_of_turn.?;
        try out.append(sot);
        try self.appendRoleSection(gpa, tok, self.user_role, out);
        const ids = try tok.encode(gpa, content);
        defer gpa.free(ids);
        try out.appendSlice(ids);
        try out.append(self.end_of_turn);
    }

    fn appendUserTurnZephyr(
        self: ChatTemplate,
        gpa: std.mem.Allocator,
        tok: *const tokenizer_mod.Tokenizer,
        content: []const u8,
        out: *std.ArrayList(u32),
    ) !void {
        // `<|user|>\n` as one encode so any BPE merges at the role/sep
        // boundary land the same as the HF reference.
        {
            const buf = try std.fmt.allocPrint(gpa, "{s}{s}", .{ self.user_role, self.header_sep });
            defer gpa.free(buf);
            const ids = try tok.encode(gpa, buf);
            defer gpa.free(ids);
            try out.appendSlice(ids);
        }
        // User message body, then `</s>` (special).
        {
            const ids = try tok.encode(gpa, content);
            defer gpa.free(ids);
            try out.appendSlice(ids);
        }
        try out.append(self.end_of_turn);
        // No trailing inter-turn sep: `composeConversation` emits seps
        // between turns explicitly. The legacy `composeZephyr` (used by
        // composePrompt) handles its own combined-encode trailing
        // sequence and doesn't touch this primitive.
    }

    fn appendUserTurnMistral(
        self: ChatTemplate,
        gpa: std.mem.Allocator,
        tok: *const tokenizer_mod.Tokenizer,
        content: []const u8,
        out: *std.ArrayList(u32),
    ) !void {
        try out.append(self.inst_open.?);
        const ids = try tok.encode(gpa, content);
        defer gpa.free(ids);
        try out.appendSlice(ids);
        try out.append(self.inst_close.?);
    }

    fn appendAssistantTurn(
        self: ChatTemplate,
        gpa: std.mem.Allocator,
        tok: *const tokenizer_mod.Tokenizer,
        content: []const u8,
        out: *std.ArrayList(u32),
    ) !void {
        switch (self.format) {
            .gemma_qwen, .llama3 => {
                const sot = self.start_of_turn.?;
                try out.append(sot);
                try self.appendRoleSection(gpa, tok, self.assistant_role, out);
                const ids = try tok.encode(gpa, content);
                defer gpa.free(ids);
                try out.appendSlice(ids);
                try out.append(self.end_of_turn);
            },
            .zephyr => {
                const buf = try std.fmt.allocPrint(gpa, "{s}{s}", .{ self.assistant_role, self.header_sep });
                defer gpa.free(buf);
                const role_ids = try tok.encode(gpa, buf);
                defer gpa.free(role_ids);
                try out.appendSlice(role_ids);
                const body_ids = try tok.encode(gpa, content);
                defer gpa.free(body_ids);
                try out.appendSlice(body_ids);
                try out.append(self.end_of_turn);
            },
            .mistral => {
                // Mistral assistant turn: bare content + </s>.
                const ids = try tok.encode(gpa, content);
                defer gpa.free(ids);
                try out.appendSlice(ids);
                try out.append(self.end_of_turn);
            },
        }
    }

    fn appendSystemTurn(
        self: ChatTemplate,
        gpa: std.mem.Allocator,
        tok: *const tokenizer_mod.Tokenizer,
        content: []const u8,
        out: *std.ArrayList(u32),
    ) !void {
        switch (self.format) {
            .gemma_qwen, .llama3 => {
                const sot = self.start_of_turn.?;
                try out.append(sot);
                try self.appendRoleSection(gpa, tok, self.system_role, out);
                const ids = try tok.encode(gpa, content);
                defer gpa.free(ids);
                try out.appendSlice(ids);
                try out.append(self.end_of_turn);
            },
            .zephyr => {
                const buf = try std.fmt.allocPrint(gpa, "{s}{s}", .{ self.system_role, self.header_sep });
                defer gpa.free(buf);
                const role_ids = try tok.encode(gpa, buf);
                defer gpa.free(role_ids);
                try out.appendSlice(role_ids);
                const body_ids = try tok.encode(gpa, content);
                defer gpa.free(body_ids);
                try out.appendSlice(body_ids);
                try out.append(self.end_of_turn);
            },
            .mistral => unreachable, // hasSystemTurn() == false; handled by fold path
        }
    }

    fn appendAssistantHeader(
        self: ChatTemplate,
        gpa: std.mem.Allocator,
        tok: *const tokenizer_mod.Tokenizer,
        out: *std.ArrayList(u32),
    ) !void {
        switch (self.format) {
            .gemma_qwen, .llama3 => {
                try out.append(self.start_of_turn.?);
                try self.appendRoleSection(gpa, tok, self.assistant_role, out);
            },
            .zephyr => {
                const buf = try std.fmt.allocPrint(gpa, "{s}{s}", .{ self.assistant_role, self.header_sep });
                defer gpa.free(buf);
                const ids = try tok.encode(gpa, buf);
                defer gpa.free(ids);
                try out.appendSlice(ids);
            },
            .mistral => unreachable, // mistral has no explicit assistant header
        }
    }

    fn appendInterTurnSep(
        self: ChatTemplate,
        gpa: std.mem.Allocator,
        tok: *const tokenizer_mod.Tokenizer,
        out: *std.ArrayList(u32),
    ) !void {
        if (self.inter_turn_sep.len == 0) return;
        const ids = try tok.encode(gpa, self.inter_turn_sep);
        defer gpa.free(ids);
        try out.appendSlice(ids);
    }

    /// Emit the role-name section in the special-token formats.
    fn appendRoleSection(
        self: ChatTemplate,
        gpa: std.mem.Allocator,
        tok: *const tokenizer_mod.Tokenizer,
        role: []const u8,
        out: *std.ArrayList(u32),
    ) !void {
        if (self.end_header) |eoh| {
            // Llama 3: role is text, then <|end_header_id|>, then `\n\n`.
            const ids = try tok.encode(gpa, role);
            defer gpa.free(ids);
            try out.appendSlice(ids);
            try out.append(eoh);
            const sep_ids = try tok.encode(gpa, self.header_sep);
            defer gpa.free(sep_ids);
            try out.appendSlice(sep_ids);
        } else {
            // Gemma / Qwen3: role + `\n` encoded together so BPE merges
            // at the boundary match the reference exactly.
            const buf = try std.fmt.allocPrint(gpa, "{s}{s}", .{ role, self.header_sep });
            defer gpa.free(buf);
            const ids = try tok.encode(gpa, buf);
            defer gpa.free(ids);
            try out.appendSlice(ids);
        }
    }
};
