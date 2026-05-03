//! OpenAI-compatible HTTP server: thin adapter from
//! `/v1/chat/completions` JSON to `InferenceRunner`.
//!
//! Architecture:
//!
//!   main thread       ── parses args, builds Session/Recorder/Runner,
//!                         calls Server.start
//!     │
//!     ├─► accept loop ── one thread per connection, spawned per
//!     │                  client.connect
//!     │     │
//!     │     └─► parse JSON, runner.submit, drain mailbox, write SSE
//!     │
//!     ├─► fan-out thread ── single consumer of runner.pollEvent;
//!     │                     dispatches Events to per-corr mailboxes
//!     │                     by event.corr (preserves SPSC contract)
//!     │
//!     └─► InferenceRunner thread (owned by runner.start) — drives
//!                   tickInline + Session + GPU
//!
//! Concurrency: connection threads grab the submit mutex briefly to
//! enqueue chat commands. The fan-out thread is the SOLE consumer
//! of runner.pollEvent so the SPSC contract holds. Per-corr
//! mailboxes synchronize between fan-out (signaler) and connection
//! threads (waiters).
//!
//! For v0 we serialize requests at the runner level — only one
//! ChatCommand active at a time. Concurrent clients will see their
//! requests queue cleanly; per-request mailbox + finish event
//! ensures clean teardown order.

const std = @import("std");
const net = std.net;

const session = @import("../session.zig");
const chat_template = @import("../chat_template.zig");
const recorder_mod = @import("../gpu/recorder.zig");
const inf_runner = @import("../inference/runner.zig");
const inf_proto = @import("../inference/proto.zig");

const http = @import("http.zig");
const json_codec = @import("json.zig");

pub const ServerConfig = struct {
    /// Bind address. Default 127.0.0.1 keeps the server local
    /// unless the operator explicitly opens it. v0 has no auth.
    bind_address: []const u8 = "127.0.0.1",
    port: u16 = 8080,
    /// Public model id surfaced via `/v1/models` and validated
    /// against incoming `model` fields. Caller picks something
    /// stable (e.g. the HF id `Qwen/Qwen3-4B-Instruct-2507` or a
    /// shorthand like `qwen3-4b`).
    model_id: []const u8,
    /// Default cap on completion length when a request omits
    /// `max_tokens`.
    default_max_tokens: u32 = 256,
    /// Max concurrent connection-handling threads. Tracked but
    /// not enforced in v0 (we just spawn freely); future MPMC
    /// runner with proper concurrency would gate here.
    max_connections: u16 = 64,
};

const Mailbox = struct {
    events: std.ArrayList(inf_proto.Event),
    mu: std.Thread.Mutex = .{},
    cv: std.Thread.Condition = .{},
    /// Set true after the terminal finish/err event is delivered
    /// so the fan-out thread can stop appending and the connection
    /// thread cleans up.
    closed: bool = false,

    fn init(allocator: std.mem.Allocator) Mailbox {
        return .{ .events = std.ArrayList(inf_proto.Event).init(allocator) };
    }
    fn deinit(self: *Mailbox) void {
        self.events.deinit();
    }
};

pub const Server = struct {
    allocator: std.mem.Allocator,
    cfg: ServerConfig,
    runner: *inf_runner.InferenceRunner,

    listener: ?net.Server,
    accept_thread: ?std.Thread,
    fanout_thread: ?std.Thread,

    /// Producer-side serialization for runner.submit. Multiple
    /// connection threads serialize behind this mutex.
    submit_mu: std.Thread.Mutex,

    /// Per-corr mailboxes. The fan-out thread reads `event.corr`,
    /// looks up the mailbox, signals its CV. Connection threads
    /// register on submit, deregister on finish.
    mailboxes_mu: std.Thread.Mutex,
    mailboxes: std.AutoHashMap(u64, *Mailbox),

    next_corr: std.atomic.Value(u64),

    running: std.atomic.Value(bool),

    pub fn init(
        allocator: std.mem.Allocator,
        runner: *inf_runner.InferenceRunner,
        cfg: ServerConfig,
    ) !Server {
        return .{
            .allocator = allocator,
            .cfg = cfg,
            .runner = runner,
            .listener = null,
            .accept_thread = null,
            .fanout_thread = null,
            .submit_mu = .{},
            .mailboxes_mu = .{},
            .mailboxes = std.AutoHashMap(u64, *Mailbox).init(allocator),
            .next_corr = std.atomic.Value(u64).init(1),
            .running = std.atomic.Value(bool).init(false),
        };
    }

    pub fn deinit(self: *Server) void {
        // Drain any remaining mailboxes (shouldn't have any after
        // shutdown, but defensive).
        var it = self.mailboxes.iterator();
        while (it.next()) |entry| {
            entry.value_ptr.*.deinit();
            self.allocator.destroy(entry.value_ptr.*);
        }
        self.mailboxes.deinit();
    }

    pub fn start(self: *Server) !void {
        // Bind + listen.
        const addr = try net.Address.parseIp(self.cfg.bind_address, self.cfg.port);
        self.listener = try addr.listen(.{ .reuse_address = true });
        self.running.store(true, .release);

        std.debug.print(
            "[serve] listening on http://{s}:{d}  model={s}\n",
            .{ self.cfg.bind_address, self.cfg.port, self.cfg.model_id },
        );

        // Spawn fan-out thread.
        self.fanout_thread = try std.Thread.spawn(.{}, fanoutLoop, .{self});
        // Spawn accept loop.
        self.accept_thread = try std.Thread.spawn(.{}, acceptLoop, .{self});
    }

    pub fn shutdown(self: *Server) void {
        self.running.store(false, .release);
        if (self.listener) |*l| {
            // Close listener so the accept loop unblocks.
            l.deinit();
            self.listener = null;
        }
        if (self.accept_thread) |t| {
            t.join();
            self.accept_thread = null;
        }
        if (self.fanout_thread) |t| {
            t.join();
            self.fanout_thread = null;
        }
    }

    // ── Accept loop ─────────────────────────────────────────────

    fn acceptLoop(self: *Server) void {
        while (self.running.load(.acquire)) {
            const conn = (self.listener.?).accept() catch |e| {
                if (!self.running.load(.acquire)) return;
                std.debug.print("[serve] accept err: {s}\n", .{@errorName(e)});
                std.time.sleep(10 * std.time.ns_per_ms);
                continue;
            };
            // Each connection gets its own thread. Detach via spawn
            // and let the thread tear down its own state on exit.
            const t = std.Thread.spawn(.{}, handleConnection, .{ self, conn }) catch |e| {
                std.debug.print("[serve] spawn err: {s}\n", .{@errorName(e)});
                conn.stream.close();
                continue;
            };
            t.detach();
        }
    }

    // ── Fan-out thread ──────────────────────────────────────────

    fn fanoutLoop(self: *Server) void {
        while (self.running.load(.acquire)) {
            const ev = self.runner.pollEvent() orelse {
                std.time.sleep(500 * std.time.ns_per_us); // 500µs
                continue;
            };

            // arena_swap is per-runner, not per-corr; pollEvent
            // already auto-confirmed it. Skip dispatch.
            if (ev.kind == .arena_swap) continue;

            self.mailboxes_mu.lock();
            const mb_opt = self.mailboxes.get(ev.corr);
            self.mailboxes_mu.unlock();

            if (mb_opt) |mb| {
                mb.mu.lock();
                mb.events.append(ev) catch {};
                switch (ev.kind) {
                    .finish, .err => mb.closed = true,
                    else => {},
                }
                mb.cv.signal();
                mb.mu.unlock();
            }
            // No mailbox → corr already cleaned up (race with
            // connection close). Drop event silently.
        }
    }

    // ── Connection handler ──────────────────────────────────────

    fn handleConnection(self: *Server, conn: net.Server.Connection) void {
        defer conn.stream.close();
        handleConnectionInner(self, conn) catch |e| {
            std.debug.print("[serve] conn err: {s}\n", .{@errorName(e)});
        };
    }

    fn handleConnectionInner(self: *Server, conn: net.Server.Connection) !void {
        var arena = std.heap.ArenaAllocator.init(self.allocator);
        defer arena.deinit();
        const a = arena.allocator();

        var req = http.readRequest(a, conn.stream.reader()) catch |e| {
            try writeHttpError(conn.stream.writer(), .bad_request, "request read failed: {s}", .{@errorName(e)});
            return;
        };
        // req's own arena is inside `a` — no separate deinit needed.
        _ = &req;

        // CORS preflight
        if (req.method == .OPTIONS) {
            try conn.stream.writer().writeAll(
                "HTTP/1.1 204 No Content\r\n" ++
                    "Access-Control-Allow-Origin: *\r\n" ++
                    "Access-Control-Allow-Methods: POST, GET, OPTIONS\r\n" ++
                    "Access-Control-Allow-Headers: Content-Type, Authorization\r\n" ++
                    "Content-Length: 0\r\n" ++
                    "Connection: close\r\n\r\n",
            );
            return;
        }

        // Routing.
        if (req.method == .GET and std.mem.eql(u8, req.path, "/v1/models")) {
            return self.handleModels(conn.stream.writer());
        }
        if (req.method == .POST and std.mem.eql(u8, req.path, "/v1/chat/completions")) {
            return self.handleChatCompletions(a, conn.stream.writer(), req.body);
        }
        // 404.
        try writeHttpError(conn.stream.writer(), .not_found, "no such route: {s} {s}", .{ @tagName(req.method), req.path });
    }

    fn handleModels(self: *Server, writer: anytype) !void {
        var buf = std.ArrayList(u8).init(self.allocator);
        defer buf.deinit();
        try json_codec.writeModelsResponse(buf.writer(), self.cfg.model_id, std.time.timestamp());
        try http.writeResponse(writer, .ok, "application/json", buf.items);
    }

    fn handleChatCompletions(
        self: *Server,
        arena: std.mem.Allocator,
        writer: anytype,
        body: []const u8,
    ) !void {
        var pr = json_codec.parseChatRequest(arena, body) catch |e| {
            const msg = switch (e) {
                error.MissingModel => "missing required field: model",
                error.MissingMessages => "missing required field: messages",
                error.InvalidMessage => "malformed message object",
                error.UnsupportedRole => "unsupported role (system/user/assistant only)",
                error.UnsupportedContentPart => "image/audio content parts not yet supported",
                error.InvalidStop => "stop must be string or array of strings",
                error.StopArrayTooLong => "stop array exceeds 4 entries",
                error.InvalidN => "n must be 1",
                error.InvalidJson => "malformed JSON body",
                error.NotImplemented => "feature not implemented",
                else => "parse error",
            };
            try writeJsonError(writer, arena, .bad_request, msg, "invalid_request_error", null);
            return;
        };

        if (!std.mem.eql(u8, pr.model, self.cfg.model_id)) {
            try writeJsonError(writer, arena, .not_found,
                "model not found; only the loaded model is served", "model_not_found", null);
            return;
        }

        const corr = self.next_corr.fetchAdd(1, .monotonic);
        const id_str = try std.fmt.allocPrint(arena, "chatcmpl-{x}", .{corr});

        // Register mailbox before submit so the fan-out thread can
        // route events that arrive immediately.
        const mb = try self.allocator.create(Mailbox);
        mb.* = Mailbox.init(self.allocator);
        defer {
            self.mailboxes_mu.lock();
            _ = self.mailboxes.remove(corr);
            self.mailboxes_mu.unlock();
            mb.deinit();
            self.allocator.destroy(mb);
        }
        self.mailboxes_mu.lock();
        try self.mailboxes.put(corr, mb);
        self.mailboxes_mu.unlock();

        const max_tokens: u32 = if (pr.max_tokens) |m| m else self.cfg.default_max_tokens;
        const stop_strings = pr.stopStringsSlice();

        // Submit. The runner deep-copies messages + stop_strings on
        // accept, so pr's arena can free as soon as we return —
        // except we hold pr alive for the response writer. Fine.
        self.submit_mu.lock();
        self.runner.submit(.{ .chat = .{
            .corr = corr,
            .messages = pr.messages,
            .max_tokens = max_tokens,
            .stop_strings = stop_strings,
        } }) catch |e| {
            self.submit_mu.unlock();
            const msg = if (e == error.QueueFull)
                "server busy, try again"
            else
                @errorName(e);
            try writeJsonError(writer, arena, .service_unavailable, msg, "internal_error", null);
            return;
        };
        self.submit_mu.unlock();

        if (pr.stream) {
            try self.streamResponse(writer, mb, id_str);
        } else {
            try self.collectResponse(arena, writer, mb, id_str);
        }
    }

    fn streamResponse(
        self: *Server,
        writer: anytype,
        mb: *Mailbox,
        id_str: []const u8,
    ) !void {
        try http.writeSseHeaders(writer);

        const created = std.time.timestamp();
        var role_sent = false;
        var done_buf: [1024]u8 = undefined;

        while (true) {
            mb.mu.lock();
            while (mb.events.items.len == 0 and !mb.closed) mb.cv.wait(&mb.mu);
            if (mb.events.items.len == 0 and mb.closed) {
                mb.mu.unlock();
                break;
            }
            const ev = mb.events.orderedRemove(0);
            mb.mu.unlock();

            switch (ev.kind) {
                .accepted => {
                    // First frame: role-only delta.
                    if (!role_sent) {
                        var fbs = std.io.fixedBufferStream(&done_buf);
                        try json_codec.buildStreamChunk(fbs.writer(), id_str, created, self.cfg.model_id, .role);
                        try http.writeSseFrame(writer, fbs.getWritten());
                        role_sent = true;
                    }
                },
                .token => |t| {
                    if (!role_sent) {
                        var fbs = std.io.fixedBufferStream(&done_buf);
                        try json_codec.buildStreamChunk(fbs.writer(), id_str, created, self.cfg.model_id, .role);
                        try http.writeSseFrame(writer, fbs.getWritten());
                        role_sent = true;
                    }
                    const text = self.runner.resolve(t.decoded);
                    var fbs = std.io.fixedBufferStream(&done_buf);
                    json_codec.buildStreamChunk(fbs.writer(), id_str, created, self.cfg.model_id, .{ .content = text }) catch {
                        // Token text larger than 1KB after JSON
                        // escaping: skip this token rather than
                        // bail. Real fix: dynamic buffer.
                        continue;
                    };
                    try http.writeSseFrame(writer, fbs.getWritten());
                },
                .finish => |f| {
                    var fbs = std.io.fixedBufferStream(&done_buf);
                    try json_codec.buildStreamChunk(fbs.writer(), id_str, created, self.cfg.model_id, .{ .finish = f.reason });
                    try http.writeSseFrame(writer, fbs.getWritten());
                    try http.writeSseDone(writer);
                    return;
                },
                .err => |e| {
                    var ebuf: [512]u8 = undefined;
                    var fbs = std.io.fixedBufferStream(&ebuf);
                    try json_codec.writeError(fbs.writer(), e.msg, "internal_error", null);
                    try http.writeSseFrame(writer, fbs.getWritten());
                    try http.writeSseDone(writer);
                    return;
                },
                .arena_swap => {},
            }
        }
    }

    fn collectResponse(
        self: *Server,
        arena: std.mem.Allocator,
        writer: anytype,
        mb: *Mailbox,
        id_str: []const u8,
    ) !void {
        var content = std.ArrayList(u8).init(arena);
        var prompt_tokens: u32 = 0;
        var completion_tokens: u32 = 0;
        var finish: inf_proto.FinishReason = .stop;
        var error_msg: ?[]const u8 = null;

        while (true) {
            mb.mu.lock();
            while (mb.events.items.len == 0 and !mb.closed) mb.cv.wait(&mb.mu);
            if (mb.events.items.len == 0 and mb.closed) {
                mb.mu.unlock();
                break;
            }
            const ev = mb.events.orderedRemove(0);
            mb.mu.unlock();

            switch (ev.kind) {
                .accepted => |a| prompt_tokens = a.prefill_tokens,
                .token => |t| {
                    const text = self.runner.resolve(t.decoded);
                    try content.appendSlice(text);
                    completion_tokens += 1;
                },
                .finish => |f| {
                    finish = f.reason;
                    break;
                },
                .err => |e| {
                    error_msg = try arena.dupe(u8, e.msg);
                    break;
                },
                .arena_swap => {},
            }
        }

        if (error_msg) |msg| {
            try writeJsonError(writer, arena, .internal_error, msg, "internal_error", null);
            return;
        }

        var resp_buf = std.ArrayList(u8).init(arena);
        try json_codec.writeChatResponse(
            resp_buf.writer(),
            id_str,
            std.time.timestamp(),
            self.cfg.model_id,
            content.items,
            finish,
            prompt_tokens,
            completion_tokens,
        );
        try http.writeResponse(writer, .ok, "application/json", resp_buf.items);
    }
};

fn writeHttpError(writer: anytype, status: http.Status, comptime fmt: []const u8, args: anytype) !void {
    var buf: [512]u8 = undefined;
    const msg = try std.fmt.bufPrint(&buf, fmt, args);
    try http.writeResponse(writer, status, "text/plain", msg);
}

fn writeJsonError(
    writer: anytype,
    arena: std.mem.Allocator,
    status: http.Status,
    msg: []const u8,
    err_type: []const u8,
    code: ?[]const u8,
) !void {
    var buf = std.ArrayList(u8).init(arena);
    try json_codec.writeError(buf.writer(), msg, err_type, code);
    try http.writeResponse(writer, status, "application/json", buf.items);
}
