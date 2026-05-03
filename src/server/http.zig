//! Bare-metal HTTP/1.1 over std.net for the OpenAI-compatible
//! endpoint. Handles only what we need: a single request, with
//! Content-Length-bounded body, and either a single full response or
//! an SSE-framed stream. No keep-alive, no chunked transfer-encoding
//! on the request side, no compression. ~400 lines for a clean read
//! of what the wire actually carries.
//!
//! Why not std.http.Server: stdlib HTTP has churned across Zig 0.11–
//! 0.13 (deprecated, rewritten, SSE awkward to thread through), and
//! we want to control the SSE flush cadence directly. The HTTP/1.1
//! grammar we accept is a strict subset; the parser rejects anything
//! it doesn't recognize with a clean 400.
//!
//! Reading: we slurp request line + headers up to "\r\n\r\n", then
//! read exactly Content-Length bytes for the body. Bodies are
//! capped at MAX_BODY_BYTES (1 MiB) so a malformed Content-Length
//! can't make us allocate gigabytes; tweak if a host needs larger.
//!
//! Writing: status line + headers (built host-side), then body.
//! Writers don't buffer — every writeAll flushes to the socket. SSE
//! frames are written as `data: <json>\n\n` and the implementation
//! relies on the OS's TCP stack for delivery cadence.

const std = @import("std");
const net = std.net;

pub const MAX_BODY_BYTES: usize = 1 << 20; // 1 MiB
const MAX_REQUEST_HEAD_BYTES: usize = 16 * 1024; // 16 KiB for status+headers

pub const Method = enum { GET, POST, OPTIONS, OTHER };

pub const Request = struct {
    method: Method,
    path: []const u8, // owned by `arena`
    headers: std.StringArrayHashMap([]const u8), // owned by `arena`
    body: []const u8, // owned by `arena`
    arena: std.heap.ArenaAllocator,

    pub fn deinit(self: *Request) void {
        self.arena.deinit();
    }

    /// Case-insensitive header lookup.
    pub fn header(self: *const Request, name: []const u8) ?[]const u8 {
        var it = self.headers.iterator();
        while (it.next()) |entry| {
            if (std.ascii.eqlIgnoreCase(entry.key_ptr.*, name)) {
                return entry.value_ptr.*;
            }
        }
        return null;
    }
};

pub const Status = enum(u16) {
    ok = 200,
    bad_request = 400,
    not_found = 404,
    method_not_allowed = 405,
    payload_too_large = 413,
    unprocessable = 422,
    internal_error = 500,
    not_implemented = 501,
    service_unavailable = 503,

    pub fn reasonPhrase(self: Status) []const u8 {
        return switch (self) {
            .ok => "OK",
            .bad_request => "Bad Request",
            .not_found => "Not Found",
            .method_not_allowed => "Method Not Allowed",
            .payload_too_large => "Payload Too Large",
            .unprocessable => "Unprocessable Entity",
            .internal_error => "Internal Server Error",
            .not_implemented => "Not Implemented",
            .service_unavailable => "Service Unavailable",
        };
    }
};

// ── Request reader ──────────────────────────────────────────────────

pub const ReadError = error{
    HeaderTooLarge,
    BodyTooLarge,
    MalformedRequestLine,
    MalformedHeader,
    UnsupportedMethod,
    InvalidContentLength,
    ConnectionClosed,
} || std.mem.Allocator.Error || std.posix.ReadError;

/// Read a single HTTP request from `reader`. The returned Request
/// owns all referenced slices via its arena; caller calls deinit()
/// when done. Bodies are read fully into memory (Content-Length
/// gated by MAX_BODY_BYTES); chunked transfer is not supported on
/// the request side.
pub fn readRequest(allocator: std.mem.Allocator, reader: anytype) ReadError!Request {
    var arena = std.heap.ArenaAllocator.init(allocator);
    errdefer arena.deinit();
    const a = arena.allocator();

    // Slurp until we see "\r\n\r\n", up to MAX_REQUEST_HEAD_BYTES.
    var head_buf = std.ArrayList(u8).init(a);
    var byte_buf: [1]u8 = undefined;
    while (head_buf.items.len < MAX_REQUEST_HEAD_BYTES) {
        const n = reader.read(&byte_buf) catch |e| switch (e) {
            error.WouldBlock,
            error.NotOpenForReading,
            => return error.ConnectionClosed,
            else => |x| return x,
        };
        if (n == 0) {
            if (head_buf.items.len == 0) return error.ConnectionClosed;
            break;
        }
        try head_buf.append(byte_buf[0]);
        const items = head_buf.items;
        if (items.len >= 4 and std.mem.endsWith(u8, items, "\r\n\r\n")) {
            break;
        }
    }
    if (head_buf.items.len >= MAX_REQUEST_HEAD_BYTES) return error.HeaderTooLarge;
    if (!std.mem.endsWith(u8, head_buf.items, "\r\n\r\n")) return error.MalformedRequestLine;

    // Split request line + headers + (empty terminator).
    var lines_iter = std.mem.splitSequence(u8, head_buf.items, "\r\n");
    const request_line = lines_iter.next() orelse return error.MalformedRequestLine;

    // "METHOD PATH HTTP/1.1"
    var rl_parts = std.mem.splitScalar(u8, request_line, ' ');
    const method_s = rl_parts.next() orelse return error.MalformedRequestLine;
    const path_s = rl_parts.next() orelse return error.MalformedRequestLine;
    _ = rl_parts.next() orelse return error.MalformedRequestLine; // HTTP/1.1, ignored

    const method: Method = if (std.mem.eql(u8, method_s, "GET"))
        .GET
    else if (std.mem.eql(u8, method_s, "POST"))
        .POST
    else if (std.mem.eql(u8, method_s, "OPTIONS"))
        .OPTIONS
    else
        .OTHER;

    var headers = std.StringArrayHashMap([]const u8).init(a);
    while (lines_iter.next()) |line| {
        if (line.len == 0) break;
        const colon = std.mem.indexOfScalar(u8, line, ':') orelse return error.MalformedHeader;
        const name = std.mem.trim(u8, line[0..colon], " \t");
        const value = std.mem.trim(u8, line[colon + 1 ..], " \t");
        const name_owned = try a.dupe(u8, name);
        const value_owned = try a.dupe(u8, value);
        try headers.put(name_owned, value_owned);
    }

    // Body length: Content-Length only. Chunked transfer-encoding
    // is not supported on the request side (clients almost always
    // send Content-Length for JSON bodies anyway).
    var content_length: usize = 0;
    var it = headers.iterator();
    while (it.next()) |entry| {
        if (std.ascii.eqlIgnoreCase(entry.key_ptr.*, "content-length")) {
            content_length = std.fmt.parseInt(usize, entry.value_ptr.*, 10) catch return error.InvalidContentLength;
            break;
        }
    }
    if (content_length > MAX_BODY_BYTES) return error.BodyTooLarge;

    const body_owned = try a.alloc(u8, content_length);
    var read_total: usize = 0;
    while (read_total < content_length) {
        const n = try reader.read(body_owned[read_total..]);
        if (n == 0) return error.ConnectionClosed;
        read_total += n;
    }

    const path_owned = try a.dupe(u8, path_s);

    return .{
        .method = method,
        .path = path_owned,
        .headers = headers,
        .body = body_owned,
        .arena = arena,
    };
}

// ── Response writer ─────────────────────────────────────────────────

pub fn writeResponse(
    writer: anytype,
    status: Status,
    content_type: []const u8,
    body: []const u8,
) !void {
    try writer.print("HTTP/1.1 {d} {s}\r\n", .{ @intFromEnum(status), status.reasonPhrase() });
    try writer.print("Content-Type: {s}\r\n", .{content_type});
    try writer.print("Content-Length: {d}\r\n", .{body.len});
    try writer.writeAll("Connection: close\r\n");
    try writer.writeAll("Access-Control-Allow-Origin: *\r\n");
    try writer.writeAll("\r\n");
    try writer.writeAll(body);
}

/// Begin an SSE response: writes status + headers, leaves the body
/// open for streaming frames via `writeSseFrame`.
pub fn writeSseHeaders(writer: anytype) !void {
    try writer.writeAll("HTTP/1.1 200 OK\r\n");
    try writer.writeAll("Content-Type: text/event-stream\r\n");
    try writer.writeAll("Cache-Control: no-cache\r\n");
    try writer.writeAll("Connection: close\r\n");
    try writer.writeAll("Access-Control-Allow-Origin: *\r\n");
    try writer.writeAll("X-Accel-Buffering: no\r\n"); // disable nginx proxy buffering
    try writer.writeAll("\r\n");
}

/// Emit a single SSE `data:` frame followed by the required blank
/// line. `payload` is the JSON or `[DONE]` literal.
pub fn writeSseFrame(writer: anytype, payload: []const u8) !void {
    try writer.writeAll("data: ");
    try writer.writeAll(payload);
    try writer.writeAll("\n\n");
}

/// Final SSE marker per OpenAI spec.
pub fn writeSseDone(writer: anytype) !void {
    try writer.writeAll("data: [DONE]\n\n");
}

// ── Tests ───────────────────────────────────────────────────────────

test "http: parse minimal POST request" {
    const fixture =
        "POST /v1/chat/completions HTTP/1.1\r\n" ++
        "Host: localhost:8080\r\n" ++
        "Content-Type: application/json\r\n" ++
        "Content-Length: 27\r\n" ++
        "\r\n" ++
        "{\"model\":\"x\",\"messages\":[]}";
    var fbs = std.io.fixedBufferStream(fixture);
    var req = try readRequest(std.testing.allocator, fbs.reader());
    defer req.deinit();

    try std.testing.expectEqual(Method.POST, req.method);
    try std.testing.expectEqualStrings("/v1/chat/completions", req.path);
    try std.testing.expectEqualStrings("application/json", req.header("content-type").?);
    try std.testing.expectEqualStrings("application/json", req.header("Content-Type").?); // case-insensitive
    try std.testing.expectEqualStrings("{\"model\":\"x\",\"messages\":[]}", req.body);
}

test "http: parse GET request without body" {
    const fixture = "GET /v1/models HTTP/1.1\r\nHost: x\r\n\r\n";
    var fbs = std.io.fixedBufferStream(fixture);
    var req = try readRequest(std.testing.allocator, fbs.reader());
    defer req.deinit();
    try std.testing.expectEqual(Method.GET, req.method);
    try std.testing.expectEqualStrings("/v1/models", req.path);
    try std.testing.expectEqual(@as(usize, 0), req.body.len);
}

test "http: malformed request line returns error" {
    const fixture = "garbage\r\n\r\n";
    var fbs = std.io.fixedBufferStream(fixture);
    try std.testing.expectError(error.MalformedRequestLine, readRequest(std.testing.allocator, fbs.reader()));
}

test "http: oversized body rejected" {
    const fixture =
        "POST / HTTP/1.1\r\nContent-Length: 99999999\r\n\r\n";
    var fbs = std.io.fixedBufferStream(fixture);
    try std.testing.expectError(error.BodyTooLarge, readRequest(std.testing.allocator, fbs.reader()));
}

test "http: writeResponse produces well-formed output" {
    var buf: [256]u8 = undefined;
    var fbs = std.io.fixedBufferStream(&buf);
    try writeResponse(fbs.writer(), .ok, "application/json", "{\"x\":1}");
    const out = fbs.getWritten();
    try std.testing.expect(std.mem.startsWith(u8, out, "HTTP/1.1 200 OK\r\n"));
    try std.testing.expect(std.mem.indexOf(u8, out, "Content-Type: application/json\r\n") != null);
    try std.testing.expect(std.mem.indexOf(u8, out, "Content-Length: 7\r\n") != null);
    try std.testing.expect(std.mem.endsWith(u8, out, "\r\n\r\n{\"x\":1}"));
}

test "http: writeSseFrame + done framing" {
    var buf: [128]u8 = undefined;
    var fbs = std.io.fixedBufferStream(&buf);
    try writeSseFrame(fbs.writer(), "{\"x\":1}");
    try writeSseDone(fbs.writer());
    try std.testing.expectEqualStrings("data: {\"x\":1}\n\ndata: [DONE]\n\n", fbs.getWritten());
}
