//! InferenceRunner — the cooperative-inference scheduler that drives
//! `Session` from a producer-supplied command stream.
//!
//! Two operating modes (tracked by `RunnerMode`):
//!
//!   .inline_   — host drives ticks from its own loop (matryoshka
//!                render thread). `submit` and `pollEvent` use the
//!                same SPSC queue API; the runner just doesn't own
//!                a thread. Caller invokes `tickInline()` once per
//!                frame to advance work.
//!   .threaded  — runner owns a worker thread that loops calling
//!                `tickInline` internally. Producers (HTTP server)
//!                only see submit/pollEvent.
//!
//! Both modes share the same active-request state machine: drain
//! commands → if no active and there's a chat command, accept it
//! (compose prompt via Session.appendMessages, install on_token
//! hook, push `accepted` event) → call sess.tickFrame → emit token
//! events for any new tokens → on finish, push `finish` event and
//! clear active. Producer demuxes by `corr`.
//!
//! Streaming text path: Session's on_token callback fires inside
//! tickFrame, producing UTF-8 bytes per token. The runner writes
//! them into a PingPongArena and emits a `token` event with a
//! `DecodedSlice` reference. Slice lifetime: valid until the
//! consumer pops the next `arena_swap` event. HTTP servers that
//! flush per-token (typical SSE) never hold across that boundary;
//! consumers that buffer must respect it.

const std = @import("std");
const session = @import("../session.zig");
const chat_template = @import("../chat_template.zig");
const recorder_mod = @import("../gpu/recorder.zig");
const queue = @import("queue.zig");
const arena_mod = @import("arena.zig");
const proto = @import("proto.zig");

pub const Command = proto.Command;
pub const Event = proto.Event;
pub const FinishReason = proto.FinishReason;
pub const Attachment = proto.Attachment;
pub const DecodedSlice = proto.DecodedSlice;
pub const ArenaId = proto.ArenaId;

pub const RunnerMode = enum { inline_, threaded };

/// Worker thread state machine. Only the runner thread itself
/// transitions to `running` / `exited`; producer-side code (or
/// `shutdown()`) sets `draining`. Inline mode keeps state pegged at
/// `stopped`.
const ThreadState = enum(u8) {
    stopped = 0,
    running = 1,
    draining = 2,
    exited = 3,
};

pub const RunnerConfig = struct {
    /// Default per-tick budget when a chat command doesn't set
    /// per_tick_us. `.either` with a coarse layer backstop and a
    /// fine µs cap handles both small models (where µs governs)
    /// and big ones (where layer cap fires first and prevents the
    /// recorder's descriptor pool from overflowing in a single
    /// tick). Pure-µs mode is available via per_tick_us if a host
    /// has resized its recorder accordingly.
    default_budget: session.Budget = .{ .either = .{ .layers = 8, .microseconds = 5000 } },
    /// Default max_tokens cap when a chat command sets max_tokens=0.
    default_max_tokens: u32 = 256,
    /// Initial size of each ping-pong arena (bytes). Grows on
    /// saturation. 32 KiB covers ~5–10 seconds of streaming at
    /// typical token sizes; one or two grows during warmup is
    /// expected.
    arena_initial_bytes_each: usize = 1 << 15,
    /// Trailing-text buffer length for stop-string suffix matching.
    /// At least max(len(stop_strings)) + 1; 64 chars is plenty for
    /// the typical "}\n" / "</response>" / "STOP" patterns.
    stop_match_buffer: usize = 256,
};

const CmdRing = queue.SpscRing(Command, 6); // 64 slots
const EventRing = queue.SpscRing(Event, 10); // 1024 slots

/// State carried while a chat request is being decoded. Owned by the
/// runner; freed on finish. All slices that came from the producer
/// (messages.content, stop_strings) are deep-copied here so the
/// producer can free its inputs as soon as `submit` returns.
const ActiveRequest = struct {
    corr: u64,
    sampler: session.SamplerKind,
    max_tokens: u32,
    deadline_ns: ?u64,

    // Owned clones of producer-supplied slices.
    messages_owned: []chat_template.Message,
    content_owned: [][]u8,
    stop_strings_owned: [][]u8,

    // Per-request stop-token list: end_of_turn (always) plus any
    // additional that future protocol versions might add.
    stop_tokens_buf: [1]u32,

    prompt_tokens: u32,
    completion_tokens: u32,
    accepted_event_pushed: bool,
    cancel_requested: bool,
    start_ns: i128,

    // Trailing decoded-text buffer for stop-string suffix matching.
    text_tail: std.ArrayList(u8),
    max_stop_suffix: usize,

    // Last reason latched by on_token's stop-string scan; consumed
    // by the post-tick check.
    pending_stop: bool,

    fn deinit(self: *ActiveRequest, alloc: std.mem.Allocator) void {
        for (self.content_owned) |c| alloc.free(c);
        alloc.free(self.content_owned);
        alloc.free(self.messages_owned);
        for (self.stop_strings_owned) |s| alloc.free(s);
        alloc.free(self.stop_strings_owned);
        self.text_tail.deinit();
    }
};

pub const InferenceRunner = struct {
    allocator: std.mem.Allocator,
    cfg: RunnerConfig,

    /// Borrowed Session + Recorder. Caller's responsibility to
    /// keep them alive for the runner's lifetime. v0 supports
    /// only the borrow constructor; an `initOwned` that builds a
    /// Session internally would land alongside the HTTP server
    /// since that's the first place we need it.
    sess: *session.Session,
    rec: *recorder_mod.Recorder,

    cmd_q: CmdRing,
    event_q: EventRing,
    pp: arena_mod.PingPongArena,

    active: ?ActiveRequest,
    backlog: std.ArrayList(Command),
    saved_session_cfg: session.Config,

    mode: RunnerMode,
    // Threaded-mode fields filled by start(); inline-mode leaves null.
    thread: ?std.Thread,
    thread_state: std.atomic.Value(u8),

    pub fn initBorrow(
        allocator: std.mem.Allocator,
        sess: *session.Session,
        rec: *recorder_mod.Recorder,
        cfg: RunnerConfig,
    ) !InferenceRunner {
        var pp = try arena_mod.PingPongArena.init(allocator, cfg.arena_initial_bytes_each);
        errdefer pp.deinit();
        return .{
            .allocator = allocator,
            .cfg = cfg,
            .sess = sess,
            .rec = rec,
            .cmd_q = .{},
            .event_q = .{},
            .pp = pp,
            .active = null,
            .backlog = std.ArrayList(Command).init(allocator),
            .saved_session_cfg = sess.cfg,
            .mode = .inline_,
            .thread = null,
            .thread_state = std.atomic.Value(u8).init(0),
        };
    }

    pub fn deinit(self: *InferenceRunner) void {
        // If a worker thread is still running, signal + join it
        // BEFORE we tear down the queues + arena it's using.
        if (self.mode == .threaded) self.shutdown();
        if (self.active) |*a| a.deinit(self.allocator);
        self.backlog.deinit();
        self.pp.deinit();
        // Restore Session's original cfg (we mutated it on accept).
        self.sess.cfg = self.saved_session_cfg;
    }

    // ── Threaded-mode lifecycle ──────────────────────────────────

    /// Promote to threaded mode: spawn a worker thread that drives
    /// tickInline in a loop. After this returns, producers can
    /// submit/pollEvent from any thread (provided producer-side
    /// access is single-threaded; SPSC contract). Use `shutdown()`
    /// to stop and join the worker.
    pub fn start(self: *InferenceRunner) !void {
        if (self.mode == .threaded) return error.AlreadyStarted;
        self.thread_state.store(@intFromEnum(ThreadState.running), .release);
        self.thread = try std.Thread.spawn(.{}, workerLoop, .{self});
        self.mode = .threaded;
    }

    /// Signal the worker thread to drain in-flight work and exit;
    /// joins it. Idempotent. Inline-mode runners no-op.
    pub fn shutdown(self: *InferenceRunner) void {
        if (self.mode != .threaded) return;
        self.thread_state.store(@intFromEnum(ThreadState.draining), .release);
        if (self.thread) |t| {
            t.join();
            self.thread = null;
        }
        self.mode = .inline_;
    }

    fn workerLoop(self: *InferenceRunner) void {
        while (true) {
            const state: ThreadState = @enumFromInt(self.thread_state.load(.acquire));
            // Drain → exit when there's nothing left to do. We
            // honor in-flight requests so a graceful shutdown
            // doesn't truncate a streaming response.
            if (state == .draining and self.active == null and
                self.backlog.items.len == 0 and self.cmd_q.isEmpty())
            {
                break;
            }
            self.tickInline() catch {
                // Error already pushed to event_q via the catch
                // branch in tickInline / handleToken. Loop on.
            };
            // Idle backoff: if no active request and no pending
            // commands, sleep briefly so we don't burn a core
            // spinning. 100µs is short enough that submit→accept
            // latency stays sub-ms but long enough to drop CPU
            // cleanly.
            if (self.active == null and self.cmd_q.isEmpty()) {
                std.time.sleep(100_000);
            }
        }
        self.thread_state.store(@intFromEnum(ThreadState.exited), .release);
    }

    // ── Producer API ─────────────────────────────────────────────

    /// Producer-side push. Non-blocking. Returns error.QueueFull on
    /// pressure; HTTP returns 503, embed host can retry next frame.
    pub fn submit(self: *InferenceRunner, cmd: Command) !void {
        if (!self.cmd_q.tryPush(cmd)) return error.QueueFull;
    }

    /// Producer-side drain. Returns null when no events pending.
    /// Resolve `event.kind.token.decoded` via `resolve` (or copy out
    /// before the next `arena_swap` event).
    pub fn pollEvent(self: *InferenceRunner) ?Event {
        const ev = self.event_q.tryPop() orelse return null;
        // Auto-confirm arena_swap so the runner's next swap can
        // reuse the now-retired arena. By the time the consumer has
        // popped this event, all prior tokens (which referenced the
        // outgoing arena) have already been popped, so it's safe.
        switch (ev.kind) {
            .arena_swap => |s| self.pp.confirmSwap(s.id),
            else => {},
        }
        return ev;
    }

    pub fn resolve(self: *const InferenceRunner, slice: DecodedSlice) []const u8 {
        return self.pp.resolve(slice);
    }

    // ── Tick driver (inline mode) ────────────────────────────────

    /// Advance the runner by one cycle: drain commands, run one
    /// tickFrame on the active request (if any), emit events for
    /// what happened, transition state.
    ///
    /// Inline-mode hosts call this once per frame. Threaded-mode
    /// runners call this in a loop on the worker thread.
    pub fn tickInline(self: *InferenceRunner) !void {
        // 1. Drain commands. Cancels match against active immediately;
        //    chat commands enter the backlog if there's already an
        //    active request.
        try self.drainCommands();

        // 2. If no active request, try to take from the backlog.
        if (self.active == null and self.backlog.items.len > 0) {
            const next = self.backlog.orderedRemove(0);
            try self.acceptChat(next.chat);
        }

        // 3. Active request: pre-tick checks (deadline, cancel) +
        //    advance state machine.
        if (self.active) |*a| {
            if (a.cancel_requested) {
                try self.finishActive(.cancelled);
                return;
            }
            if (a.deadline_ns) |dl| {
                const now: i128 = std.time.nanoTimestamp();
                const elapsed_ns: u128 = @intCast(now - a.start_ns);
                if (elapsed_ns >= dl) {
                    try self.finishActive(.timeout);
                    return;
                }
            }

            // Drive one tickFrame. on_token + on_token_user are wired
            // to runOnToken; that callback writes to the arena and
            // pushes token events synchronously inside tickFrame.
            try self.rec.reset();
            try self.rec.begin();
            const r = try self.sess.tickFrame(self.rec);
            try self.rec.endAndSubmit();
            _ = r;

            // Post-tick: stop-string suffix may have matched during
            // on_token; latch in pending_stop. session.isDone()
            // covers EOS / max_new_tokens / stop_tokens.
            if (a.pending_stop) {
                try self.finishActive(.stop);
                return;
            }
            if (self.sess.isDone()) {
                // Translate session phase → finish reason. Session
                // doesn't tell us *why* it stopped, so we infer:
                // - completion_tokens >= max_tokens → length
                // - else → stop (EOT / stop_token)
                const reason: FinishReason = if (a.completion_tokens >= a.max_tokens)
                    .length
                else
                    .stop;
                try self.finishActive(reason);
                return;
            }
        }
    }

    /// Process pending commands. Cancels apply immediately; chat
    /// commands either become active or queue in the backlog;
    /// shutdown drops further commands. Bounded scan: at most as
    /// many commands as are in the queue right now.
    fn drainCommands(self: *InferenceRunner) !void {
        while (self.cmd_q.tryPop()) |cmd| {
            switch (cmd) {
                .chat => |c| {
                    if (self.active == null) {
                        try self.acceptChat(c);
                    } else {
                        try self.backlog.append(.{ .chat = c });
                    }
                },
                .cancel => |c| {
                    if (self.active) |*a| {
                        if (a.corr == c.corr) a.cancel_requested = true;
                    }
                    // Match against backlog: drop matching pending.
                    var i: usize = 0;
                    while (i < self.backlog.items.len) {
                        if (self.backlog.items[i].chat.corr == c.corr) {
                            // Backlog entries weren't accepted, so we
                            // never deep-copied their slices and have
                            // nothing to free. They reference
                            // producer-owned memory which the
                            // protocol says is safe-to-free as soon
                            // as submit returns — so the producer
                            // may have already invalidated these
                            // slices. We just drop without emitting.
                            _ = self.backlog.orderedRemove(i);
                        } else i += 1;
                    }
                },
                .shutdown => {
                    // Threaded mode: signal worker to exit after
                    // active drains. Inline mode: caller observes
                    // via runner state and stops calling tickInline.
                    self.thread_state.store(2, .release); // 2 = draining
                },
            }
        }
    }

    fn acceptChat(self: *InferenceRunner, c: Command.ChatCommand) !void {
        std.debug.assert(self.active == null);

        // Validate attachments — v0 only supports text. Anything
        // else gets a finish(err) without ever entering the active
        // slot.
        for (c.attachments) |att| {
            switch (att) {
                .text => {},
                .image_url, .image_bytes => {
                    _ = self.event_q.tryPush(.{
                        .corr = c.corr,
                        .kind = .{ .err = .{ .msg = "image attachments not yet supported" } },
                    });
                    return;
                },
            }
        }

        // Deep-copy producer-owned slices.
        var content_owned = try self.allocator.alloc([]u8, c.messages.len);
        errdefer self.allocator.free(content_owned);
        var messages_owned = try self.allocator.alloc(chat_template.Message, c.messages.len);
        errdefer self.allocator.free(messages_owned);
        var copied: usize = 0;
        errdefer for (content_owned[0..copied]) |b| self.allocator.free(b);
        for (c.messages, 0..) |m, i| {
            const buf = try self.allocator.dupe(u8, m.content);
            content_owned[i] = buf;
            messages_owned[i] = .{ .role = m.role, .content = buf };
            copied = i + 1;
        }

        var stop_owned = try self.allocator.alloc([]u8, c.stop_strings.len);
        errdefer self.allocator.free(stop_owned);
        var stop_copied: usize = 0;
        errdefer for (stop_owned[0..stop_copied]) |s| self.allocator.free(s);
        for (c.stop_strings, 0..) |s, i| {
            stop_owned[i] = try self.allocator.dupe(u8, s);
            stop_copied = i + 1;
        }

        // Compute max stop-string length for trailing-buffer trim.
        var max_stop: usize = 0;
        for (stop_owned) |s| max_stop = @max(max_stop, s.len);

        const max_tokens: u32 = if (c.max_tokens == 0) self.cfg.default_max_tokens else c.max_tokens;

        // Mutate Session.cfg for this request: budget, samplers,
        // max_new_tokens, stop_tokens (we point at our buffer
        // containing template.end_of_turn), and on_token hook.
        const budget: session.Budget = if (c.per_tick_us) |us| .{ .microseconds = us } else self.cfg.default_budget;

        var active = ActiveRequest{
            .corr = c.corr,
            .sampler = c.sampler,
            .max_tokens = max_tokens,
            .deadline_ns = c.deadline_ns,
            .messages_owned = messages_owned,
            .content_owned = content_owned,
            .stop_strings_owned = stop_owned,
            .stop_tokens_buf = .{0},
            .prompt_tokens = 0,
            .completion_tokens = 0,
            .accepted_event_pushed = false,
            .cancel_requested = false,
            .start_ns = std.time.nanoTimestamp(),
            .text_tail = std.ArrayList(u8).init(self.allocator),
            .max_stop_suffix = max_stop,
            .pending_stop = false,
        };
        active.stop_tokens_buf[0] = self.sess.template.end_of_turn;
        self.active = active;

        // The Session is freshly init'd or fully drained from the
        // last request. Wire on_token + sampler + stops + budget.
        // We snapshotted saved_session_cfg at runner-init for
        // restoration on deinit; per-request mutation is fine.
        self.sess.cfg.budget = budget;
        self.sess.cfg.max_new_tokens = max_tokens;
        self.sess.cfg.sampler = c.sampler;
        self.sess.cfg.on_token = onTokenStatic;
        self.sess.cfg.on_token_user = self;
        // stop_tokens points into the active request's buffer; the
        // active outlives this request because finishActive is what
        // clears it.
        self.sess.cfg.stop_tokens = self.active.?.stop_tokens_buf[0..1];

        // Compose the conversation through the chat template and
        // queue for prefill.
        try self.sess.appendMessages(messages_owned);
        const prompt_tokens: u32 = @intCast(self.sess.prompt_q.items.len);
        self.active.?.prompt_tokens = prompt_tokens;

        // Push accepted event.
        _ = self.event_q.tryPush(.{
            .corr = c.corr,
            .kind = .{ .accepted = .{ .prefill_tokens = prompt_tokens } },
        });
        self.active.?.accepted_event_pushed = true;
    }

    fn finishActive(self: *InferenceRunner, reason: FinishReason) !void {
        if (self.active) |*a| {
            const elapsed_ns: u64 = @intCast(std.time.nanoTimestamp() - a.start_ns);
            _ = self.event_q.tryPush(.{
                .corr = a.corr,
                .kind = .{ .finish = .{
                    .reason = reason,
                    .prompt_tokens = a.prompt_tokens,
                    .completion_tokens = a.completion_tokens,
                    .elapsed_ns = elapsed_ns,
                } },
            });
            a.deinit(self.allocator);
            self.active = null;
        }
        // Reset the Session for the next request: clear queues,
        // phase, in-flight state. Restoring sess.cfg fields from
        // saved_session_cfg also drops the on_token hook.
        self.sess.cfg = self.saved_session_cfg;
        self.resetSession();
    }

    fn resetSession(self: *InferenceRunner) void {
        const s = self.sess;
        s.prompt_q.clearRetainingCapacity();
        s.generated.clearRetainingCapacity();
        s.phase = .idle;
        s.pos = 0;
        s.fwd_in_flight = false;
        s.fwd_layer = 0;
        s.sample_pending = false;
        s.bos_consumed = false;
        // Clear KV-cache write position by resetting Session.pos —
        // the GPU buffer's stale contents are never read past pos,
        // so no zeroing needed.
    }

    // ── on_token bridge ───────────────────────────────────────────

    /// Static dispatch: Session.cfg.on_token requires a `*const fn`
    /// signature, so we bridge through this static symbol and
    /// recover the runner via `user`.
    fn onTokenStatic(user: ?*anyopaque, tok_id: u32, decoded: []const u8) void {
        const self: *InferenceRunner = @ptrCast(@alignCast(user.?));
        self.handleToken(tok_id, decoded) catch |e| {
            // We're inside Session.tickFrame; we can't propagate
            // errors cleanly back out through the on_token signature.
            // Push an err event and latch pending_stop so the
            // post-tick check terminates the active request.
            _ = self.event_q.tryPush(.{
                .corr = if (self.active) |a| a.corr else 0,
                .kind = .{ .err = .{ .msg = @errorName(e) } },
            });
            if (self.active) |*a| a.pending_stop = true;
        };
    }

    fn handleToken(self: *InferenceRunner, tok_id: u32, decoded: []const u8) !void {
        const a = if (self.active) |*ap| ap else return;
        a.completion_tokens += 1;

        // 1. Write decoded bytes into the ping-pong arena.
        var write_result = self.pp.write(decoded);
        if (write_result == .swap_needed) {
            // We need to swap into the passive arena. But if the
            // consumer hasn't yet drained the PREVIOUS arena_swap
            // event, the passive isn't confirmed-clear — its
            // cursor is stale and a write would corrupt slices the
            // consumer is still reading. Spin briefly waiting for
            // consumer; in inline mode this should be true on
            // first check (the host drains between ticks); in
            // threaded mode the consumer thread is just one
            // event-pop behind. After the bound, we error out
            // rather than dropping silently.
            const SPIN_MAX_US: u64 = 50_000; // 50 ms total
            const SPIN_STEP_US: u64 = 100;
            var waited_us: u64 = 0;
            while (!self.pp.passiveClear() and waited_us < SPIN_MAX_US) {
                std.time.sleep(SPIN_STEP_US * std.time.ns_per_us);
                waited_us += SPIN_STEP_US;
            }
            if (!self.pp.passiveClear()) {
                _ = self.event_q.tryPush(.{
                    .corr = a.corr,
                    .kind = .{ .err = .{ .msg = "arena swap blocked: consumer not draining" } },
                });
                a.pending_stop = true;
                return;
            }

            // Push arena_swap so consumer flushes references to
            // the outgoing arena BEFORE we reuse it. The runner's
            // own pollEvent auto-confirms on consumer drain.
            const outgoing = self.pp.active;
            _ = self.event_q.tryPush(.{
                .corr = 0,
                .kind = .{ .arena_swap = .{ .id = outgoing } },
            });
            self.pp.swap();
            write_result = self.pp.write(decoded);
            if (write_result == .swap_needed) {
                // The new active is also too small (rare, very
                // long single decode). Grow the now-passive
                // (= just-outgoing). It's by definition not
                // confirmed-clear (we just swapped FROM it), so
                // we have to wait again, then grow.
                waited_us = 0;
                while (!self.pp.passiveClear() and waited_us < SPIN_MAX_US) {
                    std.time.sleep(SPIN_STEP_US * std.time.ns_per_us);
                    waited_us += SPIN_STEP_US;
                }
                if (!self.pp.passiveClear()) {
                    _ = self.event_q.tryPush(.{
                        .corr = a.corr,
                        .kind = .{ .err = .{ .msg = "arena grow blocked: consumer not draining" } },
                    });
                    a.pending_stop = true;
                    return;
                }
                try self.pp.growPassive(@max(self.pp.passiveArena().buf.len * 2, decoded.len * 2));
                _ = self.event_q.tryPush(.{
                    .corr = 0,
                    .kind = .{ .arena_swap = .{ .id = self.pp.active } },
                });
                self.pp.swap();
                write_result = self.pp.write(decoded);
            }
        }
        const slice: DecodedSlice = switch (write_result) {
            .ok => |s| s,
            .swap_needed, .too_large => {
                // Couldn't make room; emit an empty slice. Better
                // than dropping the token entirely — the consumer
                // still sees the id and can choose to call
                // tokenizer.decode out-of-band.
                _ = self.event_q.tryPush(.{
                    .corr = a.corr,
                    .kind = .{ .err = .{ .msg = "decoded-text arena exhausted" } },
                });
                a.pending_stop = true;
                return;
            },
        };

        _ = self.event_q.tryPush(.{
            .corr = a.corr,
            .kind = .{ .token = .{ .id = tok_id, .decoded = slice } },
        });

        // 2. Append decoded to text_tail for stop-string matching;
        //    trim to the max stop-string length so we don't grow
        //    unboundedly.
        if (a.stop_strings_owned.len > 0) {
            try a.text_tail.appendSlice(decoded);
            const max_keep = a.max_stop_suffix * 2; // a bit of headroom
            if (a.text_tail.items.len > max_keep) {
                const drop = a.text_tail.items.len - max_keep;
                std.mem.copyForwards(u8, a.text_tail.items[0..max_keep], a.text_tail.items[drop..]);
                a.text_tail.shrinkRetainingCapacity(max_keep);
            }
            for (a.stop_strings_owned) |s| {
                if (std.mem.endsWith(u8, a.text_tail.items, s)) {
                    a.pending_stop = true;
                    break;
                }
            }
        }
    }
};

// ── Tests ────────────────────────────────────────────────────────────

test "runner: command + event types compile" {
    // Smoke: ensure proto.zig is wired correctly via the runner
    // surface. End-to-end runner tests live in main.zig's
    // --runner-smoke since they need a real GPU + Session.
    const c = Command{ .shutdown = {} };
    _ = c;
    const e = Event{ .corr = 0, .kind = .{ .accepted = .{ .prefill_tokens = 0 } } };
    _ = e;
}
