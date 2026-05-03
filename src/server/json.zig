//! OpenAI `/v1/chat/completions` JSON codec.
//!
//! Parsing: walks `std.json.Value` tree because the schema has
//! polymorphic fields:
//!   - `stop` may be a string or an array of strings (≤4)
//!   - `messages[].content` may be a string or an array of
//!     content-parts ({type, text} / {type, image_url}). v0
//!     concatenates text parts and rejects image_* with a 400
//!     pointing at the future v2 vision path.
//!
//! Writing: builds responses by direct std.fmt.format escaping
//! since we only emit a small fixed shape.
//!
//! Extra fields in the request (temperature, top_p, logit_bias,
//! tools, ...) are silently ignored for v0 — we don't have
//! samplers wired beyond greedy and we don't have tools at all.
//! Documented in error messages when fields *are* validated.

const std = @import("std");
const chat_template = @import("../chat_template.zig");
const proto = @import("../inference/proto.zig");

// ── Parsed request shape ────────────────────────────────────────────

pub const StopSpec = union(enum) {
    none,
    one: []const u8, // owned by parse arena
    many: []const []const u8, // owned by parse arena
};

pub const ParsedRequest = struct {
    arena: std.heap.ArenaAllocator,

    model: []const u8,
    messages: []chat_template.Message,
    max_tokens: ?u32,
    stream: bool,
    stop: StopSpec,
    seed: ?u64,
    n: u32,
    user: ?[]const u8,

    pub fn deinit(self: *ParsedRequest) void {
        self.arena.deinit();
    }

    /// Materialize stop_strings as a flat slice for runner.submit.
    pub fn stopStringsSlice(self: *const ParsedRequest) []const []const u8 {
        return switch (self.stop) {
            .none => &.{},
            .one => |s| (&[_][]const u8{s})[0..1],
            .many => |m| m,
        };
    }
};

pub const ParseError = error{
    InvalidJson,
    MissingModel,
    MissingMessages,
    InvalidMessage,
    UnsupportedRole,
    UnsupportedContentPart,
    InvalidStop,
    StopArrayTooLong,
    InvalidN,
    NotImplemented,
} || std.mem.Allocator.Error;

pub fn parseChatRequest(allocator: std.mem.Allocator, body: []const u8) ParseError!ParsedRequest {
    var arena = std.heap.ArenaAllocator.init(allocator);
    errdefer arena.deinit();
    const a = arena.allocator();

    const parsed = std.json.parseFromSliceLeaky(std.json.Value, a, body, .{}) catch return error.InvalidJson;
    if (parsed != .object) return error.InvalidJson;
    const root = parsed.object;

    // model
    const model_v = root.get("model") orelse return error.MissingModel;
    if (model_v != .string) return error.MissingModel;
    const model = try a.dupe(u8, model_v.string);

    // messages
    const messages_v = root.get("messages") orelse return error.MissingMessages;
    if (messages_v != .array) return error.MissingMessages;
    const messages_arr = messages_v.array.items;
    var messages = try a.alloc(chat_template.Message, messages_arr.len);
    for (messages_arr, 0..) |m, i| {
        if (m != .object) return error.InvalidMessage;
        const role_v = m.object.get("role") orelse return error.InvalidMessage;
        if (role_v != .string) return error.InvalidMessage;
        const content_v = m.object.get("content") orelse return error.InvalidMessage;

        const role: chat_template.Role = if (std.mem.eql(u8, role_v.string, "system"))
            .system
        else if (std.mem.eql(u8, role_v.string, "user"))
            .user
        else if (std.mem.eql(u8, role_v.string, "assistant"))
            .assistant
        else if (std.mem.eql(u8, role_v.string, "tool"))
            return error.UnsupportedRole // tools not supported in v0
        else
            return error.UnsupportedRole;

        const content: []const u8 = switch (content_v) {
            .string => |s| try a.dupe(u8, s),
            .array => |parts| blk: {
                // Content-parts: only {type:"text",text:"..."} accepted in v0.
                // image_url / input_audio rejected. Concatenate text parts.
                var buf = std.ArrayList(u8).init(a);
                for (parts.items) |p| {
                    if (p != .object) return error.UnsupportedContentPart;
                    const type_v = p.object.get("type") orelse return error.UnsupportedContentPart;
                    if (type_v != .string) return error.UnsupportedContentPart;
                    if (std.mem.eql(u8, type_v.string, "text")) {
                        const text_v = p.object.get("text") orelse return error.UnsupportedContentPart;
                        if (text_v != .string) return error.UnsupportedContentPart;
                        try buf.appendSlice(text_v.string);
                    } else {
                        // image_url, input_audio, etc.
                        return error.UnsupportedContentPart;
                    }
                }
                break :blk try buf.toOwnedSlice();
            },
            else => return error.InvalidMessage,
        };
        messages[i] = .{ .role = role, .content = content };
    }

    // max_tokens
    var max_tokens: ?u32 = null;
    if (root.get("max_tokens")) |v| {
        if (v == .integer) max_tokens = @intCast(v.integer);
    }

    // stream
    const stream = if (root.get("stream")) |v| (v == .bool and v.bool) else false;

    // stop: string | []string | null
    var stop: StopSpec = .none;
    if (root.get("stop")) |v| {
        switch (v) {
            .string => |s| stop = .{ .one = try a.dupe(u8, s) },
            .array => |arr| {
                if (arr.items.len > 4) return error.StopArrayTooLong;
                if (arr.items.len > 0) {
                    const slice = try a.alloc([]const u8, arr.items.len);
                    for (arr.items, 0..) |item, i| {
                        if (item != .string) return error.InvalidStop;
                        slice[i] = try a.dupe(u8, item.string);
                    }
                    stop = .{ .many = slice };
                }
            },
            .null => {},
            else => return error.InvalidStop,
        }
    }

    // seed (accepted but ignored — greedy sampler is deterministic anyway)
    var seed: ?u64 = null;
    if (root.get("seed")) |v| {
        if (v == .integer) seed = @intCast(v.integer);
    }

    // n: must be 1 or absent
    var n: u32 = 1;
    if (root.get("n")) |v| {
        if (v != .integer) return error.InvalidN;
        if (v.integer != 1) return error.InvalidN;
        n = @intCast(v.integer);
    }

    var user: ?[]const u8 = null;
    if (root.get("user")) |v| {
        if (v == .string) user = try a.dupe(u8, v.string);
    }

    return .{
        .arena = arena,
        .model = model,
        .messages = messages,
        .max_tokens = max_tokens,
        .stream = stream,
        .stop = stop,
        .seed = seed,
        .n = n,
        .user = user,
    };
}

// ── Response writers ────────────────────────────────────────────────

/// Map runner FinishReason → OpenAI string. content_filter / null
/// only show up in special-cases we don't emit; tool_calls deferred.
pub fn finishReasonStr(reason: proto.FinishReason) []const u8 {
    return switch (reason) {
        .stop, .cancelled => "stop",
        .length => "length",
        .timeout => "stop", // OpenAI doesn't have timeout; closest is "stop"
        .server_shutdown => "stop",
    };
}

/// Single non-streaming response. `content` is the assembled
/// completion text.
pub fn writeChatResponse(
    writer: anytype,
    id: []const u8,
    created: i64,
    model: []const u8,
    content: []const u8,
    finish: proto.FinishReason,
    prompt_tokens: u32,
    completion_tokens: u32,
) !void {
    try writer.writeAll("{\"id\":");
    try writeJsonString(writer, id);
    try writer.print(",\"object\":\"chat.completion\",\"created\":{d},\"model\":", .{created});
    try writeJsonString(writer, model);
    try writer.writeAll(",\"choices\":[{\"index\":0,\"message\":{\"role\":\"assistant\",\"content\":");
    try writeJsonString(writer, content);
    try writer.print("}},\"finish_reason\":\"{s}\"", .{finishReasonStr(finish)});
    try writer.writeByte('}');
    try writer.print("],\"usage\":{{\"prompt_tokens\":{d},\"completion_tokens\":{d},\"total_tokens\":{d}}}}}", .{
        prompt_tokens, completion_tokens, prompt_tokens + completion_tokens,
    });
}

pub const StreamFrame = union(enum) {
    role,
    content: []const u8,
    finish: proto.FinishReason,
};

/// Build one streaming `chat.completion.chunk` payload (no SSE
/// `data: ` prefix; that's added by writeSseFrame). Caller passes
/// the assembled JSON to `http.writeSseFrame`.
pub fn buildStreamChunk(
    writer: anytype,
    id: []const u8,
    created: i64,
    model: []const u8,
    frame: StreamFrame,
) !void {
    try writer.writeAll("{\"id\":");
    try writeJsonString(writer, id);
    try writer.print(",\"object\":\"chat.completion.chunk\",\"created\":{d},\"model\":", .{created});
    try writeJsonString(writer, model);
    try writer.writeAll(",\"choices\":[{\"index\":0,\"delta\":");
    switch (frame) {
        .role => try writer.writeAll("{\"role\":\"assistant\"}"),
        .content => |c| {
            try writer.writeAll("{\"content\":");
            try writeJsonString(writer, c);
            try writer.writeByte('}');
        },
        .finish => try writer.writeAll("{}"),
    }
    switch (frame) {
        .finish => |r| try writer.print(",\"finish_reason\":\"{s}\"", .{finishReasonStr(r)}),
        else => try writer.writeAll(",\"finish_reason\":null"),
    }
    try writer.writeAll("}]}");
}

pub fn writeError(
    writer: anytype,
    msg: []const u8,
    err_type: []const u8,
    code: ?[]const u8,
) !void {
    try writer.writeAll("{\"error\":{\"message\":");
    try writeJsonString(writer, msg);
    try writer.print(",\"type\":\"{s}\"", .{err_type});
    if (code) |c| try writer.print(",\"code\":\"{s}\"", .{c});
    try writer.writeAll("}}");
}

pub fn writeModelsResponse(
    writer: anytype,
    model_id: []const u8,
    created: i64,
) !void {
    try writer.writeAll("{\"object\":\"list\",\"data\":[{\"id\":");
    try writeJsonString(writer, model_id);
    try writer.print(",\"object\":\"model\",\"created\":{d},\"owned_by\":\"valkyr\"}}]}}", .{created});
}

/// Emit a JSON-quoted string. Escapes per RFC 8259: ", \, control
/// chars 0x00–0x1F. Non-ASCII passes through verbatim (UTF-8 stays
/// JSON-valid).
pub fn writeJsonString(writer: anytype, s: []const u8) !void {
    try writer.writeByte('"');
    for (s) |c| {
        switch (c) {
            '"' => try writer.writeAll("\\\""),
            '\\' => try writer.writeAll("\\\\"),
            '\n' => try writer.writeAll("\\n"),
            '\r' => try writer.writeAll("\\r"),
            '\t' => try writer.writeAll("\\t"),
            0x08 => try writer.writeAll("\\b"),
            0x0c => try writer.writeAll("\\f"),
            else => {
                if (c < 0x20) {
                    try writer.print("\\u{x:0>4}", .{c});
                } else {
                    try writer.writeByte(c);
                }
            },
        }
    }
    try writer.writeByte('"');
}

// ── Tests ───────────────────────────────────────────────────────────

test "json: parse minimal request" {
    const body =
        \\{"model":"qwen3-4b","messages":[{"role":"user","content":"Hi"}]}
    ;
    var pr = try parseChatRequest(std.testing.allocator, body);
    defer pr.deinit();

    try std.testing.expectEqualStrings("qwen3-4b", pr.model);
    try std.testing.expectEqual(@as(usize, 1), pr.messages.len);
    try std.testing.expectEqual(chat_template.Role.user, pr.messages[0].role);
    try std.testing.expectEqualStrings("Hi", pr.messages[0].content);
    try std.testing.expectEqual(false, pr.stream);
    try std.testing.expectEqual(@as(u32, 1), pr.n);
}

test "json: parse stop as string and array" {
    const s1 =
        \\{"model":"x","messages":[],"stop":"END"}
    ;
    var pr1 = try parseChatRequest(std.testing.allocator, s1);
    defer pr1.deinit();
    try std.testing.expect(pr1.stop == .one);
    try std.testing.expectEqualStrings("END", pr1.stop.one);

    const s2 =
        \\{"model":"x","messages":[],"stop":["END","STOP","</done>"]}
    ;
    var pr2 = try parseChatRequest(std.testing.allocator, s2);
    defer pr2.deinit();
    try std.testing.expect(pr2.stop == .many);
    try std.testing.expectEqual(@as(usize, 3), pr2.stop.many.len);
    try std.testing.expectEqualStrings("</done>", pr2.stop.many[2]);
}

test "json: content-parts text concatenation" {
    const body =
        \\{"model":"x","messages":[{"role":"user","content":[{"type":"text","text":"Hello "},{"type":"text","text":"world"}]}]}
    ;
    var pr = try parseChatRequest(std.testing.allocator, body);
    defer pr.deinit();
    try std.testing.expectEqualStrings("Hello world", pr.messages[0].content);
}

test "json: image_url content-part rejected" {
    const body =
        \\{"model":"x","messages":[{"role":"user","content":[{"type":"image_url","image_url":{"url":"http://x"}}]}]}
    ;
    try std.testing.expectError(error.UnsupportedContentPart, parseChatRequest(std.testing.allocator, body));
}

test "json: n != 1 rejected" {
    const body =
        \\{"model":"x","messages":[],"n":2}
    ;
    try std.testing.expectError(error.InvalidN, parseChatRequest(std.testing.allocator, body));
}

test "json: writeChatResponse round-trips and parses" {
    var buf: [512]u8 = undefined;
    var fbs = std.io.fixedBufferStream(&buf);
    try writeChatResponse(
        fbs.writer(),
        "chatcmpl-abc",
        1234567890,
        "qwen3-4b",
        "Hello!\nWorld",
        .stop,
        10,
        2,
    );
    const out = fbs.getWritten();

    // Parse it back to verify structure.
    var parsed = try std.json.parseFromSlice(std.json.Value, std.testing.allocator, out, .{});
    defer parsed.deinit();
    const root = parsed.value.object;
    try std.testing.expectEqualStrings("chatcmpl-abc", root.get("id").?.string);
    try std.testing.expectEqualStrings("chat.completion", root.get("object").?.string);
    const choices = root.get("choices").?.array.items;
    try std.testing.expectEqual(@as(usize, 1), choices.len);
    const msg = choices[0].object.get("message").?.object;
    try std.testing.expectEqualStrings("assistant", msg.get("role").?.string);
    try std.testing.expectEqualStrings("Hello!\nWorld", msg.get("content").?.string);
    try std.testing.expectEqualStrings("stop", choices[0].object.get("finish_reason").?.string);
    const usage = root.get("usage").?.object;
    try std.testing.expectEqual(@as(i64, 12), usage.get("total_tokens").?.integer);
}

test "json: buildStreamChunk role frame" {
    var buf: [256]u8 = undefined;
    var fbs = std.io.fixedBufferStream(&buf);
    try buildStreamChunk(fbs.writer(), "id-1", 100, "qwen3-4b", .role);
    const out = fbs.getWritten();
    try std.testing.expect(std.mem.indexOf(u8, out, "\"role\":\"assistant\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, out, "\"finish_reason\":null") != null);
}

test "json: buildStreamChunk content + finish frames" {
    var buf: [256]u8 = undefined;
    var fbs = std.io.fixedBufferStream(&buf);
    try buildStreamChunk(fbs.writer(), "id-1", 100, "qwen3-4b", .{ .content = "Hi" });
    try std.testing.expect(std.mem.indexOf(u8, fbs.getWritten(), "\"content\":\"Hi\"") != null);

    fbs.reset();
    try buildStreamChunk(fbs.writer(), "id-1", 100, "qwen3-4b", .{ .finish = .stop });
    try std.testing.expect(std.mem.indexOf(u8, fbs.getWritten(), "\"finish_reason\":\"stop\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, fbs.getWritten(), "\"delta\":{}") != null);
}

test "json: writeJsonString escapes special chars" {
    var buf: [128]u8 = undefined;
    var fbs = std.io.fixedBufferStream(&buf);
    try writeJsonString(fbs.writer(), "a\"b\\c\nd");
    try std.testing.expectEqualStrings("\"a\\\"b\\\\c\\nd\"", fbs.getWritten());
}
