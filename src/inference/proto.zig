//! Wire types between producers (HTTP server, embed host) and the
//! `InferenceRunner`. All values are PODs — no allocations inside the
//! types themselves; the runner is responsible for owning any heap
//! data referenced by these structs.
//!
//! Producer responsibilities:
//!   - Build a `Command` describing what to do.
//!   - Submit it to the runner via `runner.submit(cmd)`. The runner
//!     copies any caller-owned slices (messages, stop_strings,
//!     attachments) into runner-owned storage before returning. The
//!     producer can free its inputs as soon as `submit` returns.
//!   - Drain `Event`s with `runner.pollEvent()`. Each event references
//!     decoded text via `DecodedSlice` into the runner's PingPongArena
//!     (see arena.zig). Resolve via `runner.resolve(slice)` if needed.
//!     Slice lifetime: valid until the consumer pops the next
//!     `arena_swap` event with matching id.
//!
//! Why all the lifetime ceremony: at decode rates of 50–100 tok/s a
//! per-token heap allocation would dominate the cost. The arena +
//! ping-pong gives us zero-alloc streaming; the lifetime contract is
//! the price.

const std = @import("std");
const session = @import("../session.zig");
const chat_template = @import("../chat_template.zig");
const arena = @import("arena.zig");

pub const DecodedSlice = arena.DecodedSlice;
pub const ArenaId = arena.ArenaId;

// ── Commands (producer → runner) ────────────────────────────────────

pub const Command = union(enum) {
    chat: ChatCommand,
    cancel: CancelCommand,
    /// Asks the runner to drain its in-flight request, then exit.
    /// After submitting `shutdown`, the producer should not push
    /// further commands. Threaded mode: runner's worker thread
    /// returns from its loop after the in-flight request completes;
    /// `runner.shutdown()` then joins.
    shutdown,

    pub const ChatCommand = struct {
        /// Producer-chosen correlation id; echoed on every Event for
        /// this request. Producers tracking multiple outstanding
        /// requests use this to demultiplex.
        corr: u64,

        /// Conversation history. Caller-owned; runner copies eagerly
        /// on accept and the caller may free as soon as `submit`
        /// returns. Runner clones via dupe; messages.content slices
        /// are stable for the runner's lifetime of this request.
        messages: []const chat_template.Message,

        /// Optional non-text content (images, audio, ...). Reserved
        /// for future versions; today the runner ignores anything
        /// that's not `.text` and proceeds with text-only inference.
        /// Storing the type here means the protocol won't break when
        /// we wire image encoding in v2.
        attachments: []const Attachment = &.{},

        sampler: session.SamplerKind = .greedy,

        /// Hard cap on emitted tokens. 0 = use runner's default.
        max_tokens: u32 = 0,

        /// Cap each tickFrame at this microsecond budget. null = use
        /// runner's configured default. Set this when the producer
        /// owns a render loop and wants the runner to live within a
        /// time slice (matryoshka frame budget).
        per_tick_us: ?u64 = null,

        /// Stop-string list. The runner buffers the trailing N
        /// decoded characters and matches against each string after
        /// every emit; on match it terminates with finish(.stop).
        /// In addition to these, the family's
        /// `template.end_of_turn` token always terminates.
        stop_strings: []const []const u8 = &.{},

        /// Hard upper bound on wall-clock from accept to finish,
        /// including queue wait. null = unlimited. On overrun the
        /// runner emits finish(.timeout). Useful for latency-
        /// sensitive servers.
        deadline_ns: ?u64 = null,
    };

    pub const CancelCommand = struct {
        /// Cancels the request matching this corr. If it's not
        /// in-flight (already finished or never accepted) the
        /// command is dropped silently.
        corr: u64,
    };
};

/// Future-proofed media reference. v0 only `.text` is used; runner
/// validates that other variants are absent and otherwise refuses
/// the request via finish(.err). When the protocol carries images
/// (v2+) the runner will dispatch to a vision encoder before
/// `Session.appendMessages`.
pub const Attachment = union(enum) {
    /// Inline text fragment. Useful for templated insertions where
    /// the producer doesn't want the chat composer to handle them.
    /// Passed through verbatim.
    text: []const u8,
    /// External image URL. Runner fetches + decodes on accept.
    /// Reserved.
    image_url: []const u8,
    /// Inline image bytes (PNG/JPEG). Runner decodes on accept.
    /// Reserved.
    image_bytes: []const u8,
};

// ── Events (runner → producer) ──────────────────────────────────────

pub const Event = struct {
    /// Echoes the corr from the originating chat command. For
    /// `arena_swap` events (which are runtime-internal, not
    /// per-request) corr is 0.
    corr: u64,
    kind: Kind,

    pub const Kind = union(enum) {
        /// First event for a chat request. Carries prefill token
        /// count so a server can estimate work.
        accepted: struct { prefill_tokens: u32 },

        /// One streamed token. `decoded` references the runner's
        /// PingPongArena; resolve via `runner.resolve(slice)`.
        token: struct {
            id: u32,
            decoded: DecodedSlice,
        },

        /// Lifecycle: arena[id]'s slices are no longer referenced
        /// by future events. Consumer MUST stop dereferencing slices
        /// keyed to that arena_id before the next pollEvent call;
        /// after popping this event, runner may reuse arena[id]
        /// (reset cursor, optionally grow).
        ///
        /// Producer-side handling: most consumers just want to
        /// resolve+copy `token.decoded` immediately on pop, in which
        /// case arena_swap is a no-op (you've already copied out).
        /// HTTP servers that buffer raw slices for batched SSE
        /// flushing must drain their buffer at this point.
        arena_swap: struct { id: ArenaId },

        /// Terminal event for a chat request. After this the runner
        /// emits no further events for `corr`.
        finish: struct {
            reason: FinishReason,
            prompt_tokens: u32,
            completion_tokens: u32,
            elapsed_ns: u64,
        },

        /// Out-of-band runtime error. Terminal for the request.
        /// `msg` is a static or arena-allocated string describing
        /// the failure; the runner copies short literals so the
        /// caller doesn't need to keep refs alive.
        err: struct { msg: []const u8 },
    };
};

pub const FinishReason = enum {
    /// Hit EOT/end_of_turn token or matched a stop_string.
    stop,
    /// Reached max_tokens cap.
    length,
    /// Producer sent a Cancel command.
    cancelled,
    /// deadline_ns elapsed.
    timeout,
    /// Runner is shutting down; in-flight requests are aborted.
    server_shutdown,
};

// ── Tests ────────────────────────────────────────────────────────────

test "proto: sizes are reasonable for SPSC slot copy" {
    // The SPSC ring stores Command/Event by value; if these blow up
    // to multi-KB the queue's footprint balloons. Tagged unions in
    // Zig are sized to the largest variant; ChatCommand is the
    // bulky one. Asserting an upper bound here so a future field
    // addition that pushes us past 256 bytes shows up in tests.
    try std.testing.expect(@sizeOf(Command) <= 256);
    try std.testing.expect(@sizeOf(Event) <= 96);
}
