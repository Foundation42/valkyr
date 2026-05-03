//! Grow-only ping-pong byte arena for streamed token text.
//!
//! At decode rates of 50–100 tok/s a per-token heap allocation
//! dominates the token-emit cost (allocator lock, header overhead,
//! slice-pointer dance through the SPSC queue). Instead the runner
//! writes decoded UTF-8 bytes into one of two arenas, hands out
//! `(arena_id, offset, len)` tuples, and swaps to the other arena
//! when one fills. The consumer holds slices into the previously-
//! active arena until it processes a synthetic `arena_swap` event
//! that signals "you're now ahead of any reference into arena[id]".
//! After that the runner can clear and reuse it.
//!
//! Lifetime contract:
//!
//!   1. Producer (runner) writes decoded bytes into arena[active],
//!      advances active.cursor, hands out a `DecodedSlice` keyed by
//!      `arena_id = active`.
//!   2. When active runs out of room, producer pushes an
//!      `arena_swap` Event into the SPSC ring with the OUTGOING
//!      arena's id, then flips active to the other arena.
//!   3. Consumer holds slices into the outgoing arena until it
//!      drains all events up to and including the swap event.
//!   4. After consumer pops the swap event, it MUST not dereference
//!      slices keyed to that arena id. Producer can clear.
//!
//! Because the SPSC ring is FIFO, by the time the consumer pops the
//! swap event it has already processed every Event that holds a
//! slice into that arena. The contract is enforced by ordering, not
//! refcounts.
//!
//! Grow-only: when an arena fills mid-stream and the OTHER arena
//! still hasn't been confirmed-cleared (consumer is slow), we double
//! its capacity. Bounded by peak streaming load, then never grown
//! again. Real apps see one or two grows during warmup, then steady
//! state.
//!
//! What we explicitly DON'T do: shrinking, fancy fragmentation
//! handling, copying old slices on grow. Token text is short-lived
//! (microseconds between produce and consume); a simple bump
//! allocator with a 2-arena rotation suffices.

const std = @import("std");

/// Identifies which of the two arenas a slice points into.
pub const ArenaId = enum(u8) { a, b };

/// Reference into an arena. Only meaningful in conjunction with the
/// PingPongArena that produced it; resolve via `arena.resolve(slice)`.
/// Lifetime: valid until the consumer has popped the `arena_swap`
/// event whose `id` matches `arena_id`.
pub const DecodedSlice = struct {
    arena_id: ArenaId,
    offset: u32,
    len: u32,

    pub const empty: DecodedSlice = .{ .arena_id = .a, .offset = 0, .len = 0 };
};

/// One half of the ping-pong: a grow-only bump allocator over a
/// single byte buffer.
const Arena = struct {
    buf: []u8,
    cursor: u32,

    fn init(allocator: std.mem.Allocator, initial_bytes: usize) !Arena {
        const buf = try allocator.alloc(u8, initial_bytes);
        return .{ .buf = buf, .cursor = 0 };
    }

    fn deinit(self: *Arena, allocator: std.mem.Allocator) void {
        allocator.free(self.buf);
    }

    fn remaining(self: Arena) usize {
        return self.buf.len - self.cursor;
    }

    /// Write `bytes` into this arena. Returns null if it doesn't fit;
    /// caller decides whether to grow or swap.
    fn tryWrite(self: *Arena, bytes: []const u8) ?u32 {
        if (bytes.len > self.remaining()) return null;
        const off = self.cursor;
        @memcpy(self.buf[off..][0..bytes.len], bytes);
        self.cursor += @intCast(bytes.len);
        return off;
    }

    /// Replace the buffer with a larger one. Old contents are
    /// discarded — callers must only grow an arena that has already
    /// been confirmed-cleared by the consumer (i.e., its swap event
    /// has been popped). cursor resets to 0.
    fn growAndReset(self: *Arena, allocator: std.mem.Allocator, new_bytes: usize) !void {
        std.debug.assert(new_bytes >= self.buf.len);
        allocator.free(self.buf);
        self.buf = try allocator.alloc(u8, new_bytes);
        self.cursor = 0;
    }

    fn reset(self: *Arena) void {
        self.cursor = 0;
    }
};

/// Two arenas in rotation. Producer (runner) calls `write` to append
/// decoded bytes; on overflow, `write` returns `.swap_needed` so the
/// caller can push an `arena_swap` event and call `swap()`. Consumer
/// resolves slices via `resolve`.
pub const PingPongArena = struct {
    allocator: std.mem.Allocator,
    a: Arena,
    b: Arena,
    /// Which arena the producer is currently writing into.
    active: ArenaId,
    /// Whether the OTHER (currently-passive) arena is safe to reuse.
    /// Set true after the consumer pops the swap event for that
    /// arena, so the next swap can land cleanly. Initially true for
    /// both: starting state has no outstanding references.
    passive_clear: bool,

    pub fn init(allocator: std.mem.Allocator, initial_bytes_each: usize) !PingPongArena {
        var a = try Arena.init(allocator, initial_bytes_each);
        errdefer a.deinit(allocator);
        const b = try Arena.init(allocator, initial_bytes_each);
        return .{
            .allocator = allocator,
            .a = a,
            .b = b,
            .active = .a,
            .passive_clear = true,
        };
    }

    pub fn deinit(self: *PingPongArena) void {
        self.a.deinit(self.allocator);
        self.b.deinit(self.allocator);
    }

    fn activeArena(self: *PingPongArena) *Arena {
        return switch (self.active) {
            .a => &self.a,
            .b => &self.b,
        };
    }

    pub fn passiveArena(self: *PingPongArena) *Arena {
        return switch (self.active) {
            .a => &self.b,
            .b => &self.a,
        };
    }

    /// Result of `write`. `ok` carries the slice into the active
    /// arena. `swap_needed` means the active arena overflowed; the
    /// caller should push an `arena_swap` event into the SPSC ring
    /// (with the OUTGOING active's id), then call `swap()` and retry
    /// the write. `too_large` means even a freshly-grown arena
    /// couldn't hold the bytes — fatal for streaming, caller should
    /// emit an err event.
    pub const WriteResult = union(enum) {
        ok: DecodedSlice,
        swap_needed,
        too_large,
    };

    /// Append `bytes` to the active arena. The returned slice
    /// references the active arena; the consumer must not deref it
    /// after popping the next arena_swap event for this arena_id.
    pub fn write(self: *PingPongArena, bytes: []const u8) WriteResult {
        if (bytes.len > std.math.maxInt(u32)) return .too_large;
        const active = self.activeArena();
        if (active.tryWrite(bytes)) |off| {
            return .{ .ok = .{
                .arena_id = self.active,
                .offset = off,
                .len = @intCast(bytes.len),
            } };
        }
        return .swap_needed;
    }

    /// Flip active. Caller is responsible for having pushed the
    /// arena_swap event FIRST, so the consumer sees the swap and
    /// drains references before the producer reuses the arena.
    ///
    /// Three cases:
    ///   1. The new-passive (just-active) arena had a successful
    ///      cycle and is full of valid data → consumer will hold
    ///      its slices until the swap event is popped.
    ///   2. The new-active (was-passive) arena was previously
    ///      cleared by the consumer (passive_clear == true) → reset
    ///      cursor, fresh writes go into it.
    ///   3. The new-active was NOT yet cleared (consumer is behind
    ///      → grow doubles the buffer and resets cursor. Old
    ///      contents are abandoned (the consumer's slices into the
    ///      OLD passive remain valid because slices reference by
    ///      offset into the buffer the slice was minted from; once
    ///      we reallocate, those slices dangle. So we MUST NOT grow
    ///      a buffer the consumer is still reading from.) Resolved
    ///      by only growing the OUTGOING-becoming-incoming arena
    ///      AFTER its arena_swap has been confirmed.
    ///
    /// In short: callers must observe the consumer's
    /// `confirmSwap()` callback before invoking `swap()` if they
    /// want grow safety. Without confirmation we just rotate
    /// without growing — at the risk of overflowing the new active
    /// arena. The runner observes confirmations via a per-arena
    /// epoch counter compared against the consumer's drained
    /// count; see runner.zig.
    pub fn swap(self: *PingPongArena) void {
        // The arena we're flipping AWAY from now holds outstanding
        // references; mark it as not-clear.
        self.passive_clear = false;
        self.active = switch (self.active) {
            .a => .b,
            .b => .a,
        };
        // The new active arena: if the consumer has previously
        // cleared it (we'd have called confirmSwap), reset cursor
        // so writes start fresh. If not, the cursor is stale and
        // writes will land in random territory — caller must have
        // grown first.
        // The default case (init time): both arenas start empty
        // and clear; first swap lands in arena[b] with cursor=0
        // already because nothing was ever written to it.
        // Subsequent swaps after a confirmed cycle: confirmSwap
        // resets cursor at confirmation time, so swap() doesn't
        // need to.
    }

    /// Consumer-side: signal that all events referencing arena[id]
    /// have been processed. Producer can now reuse it (cursor
    /// resets, can grow if needed). Call this when the consumer
    /// pops an arena_swap event with matching id.
    pub fn confirmSwap(self: *PingPongArena, id: ArenaId) void {
        // The "id" here is the OUTGOING-at-swap-time arena, i.e.
        // the one whose slices we just retired. It should match
        // self.active's OPPOSITE iff the producer has already
        // swapped, OR self.active itself if the producer hasn't
        // swapped yet (rare race; see note in runner).
        const other = switch (self.active) {
            .a => &self.b,
            .b => &self.a,
        };
        // Reset the buffer that's now safe to reuse. We pick the
        // passive one regardless of `id` — confirmSwap is always
        // about the passive arena because that's what the producer
        // just rotated away from. If id doesn't match active's
        // opposite, the runner invariant is broken; assert.
        const expected_other = switch (self.active) {
            .a => ArenaId.b,
            .b => ArenaId.a,
        };
        std.debug.assert(id == expected_other);
        other.reset();
        self.passive_clear = true;
    }

    /// Producer: grow the passive arena if it's saturated AND
    /// confirmed-clear. Used when a swap-needed write would have
    /// overflowed the new active too.
    pub fn growPassive(self: *PingPongArena, new_bytes_each: usize) !void {
        std.debug.assert(self.passive_clear);
        const passive = self.passiveArena();
        try passive.growAndReset(self.allocator, new_bytes_each);
    }

    /// Producer: signal that we're about to swap. Returns true if
    /// the new-active arena (currently passive) is confirmed-clear
    /// and a `swap()` can safely follow. False otherwise — caller
    /// should NOT swap because the new-active still holds slices
    /// the consumer is reading. In threaded mode, callers spin
    /// briefly waiting for the consumer's confirmSwap; in inline
    /// mode this should never be false because the consumer (host)
    /// drains events before the next tickInline.
    pub fn passiveClear(self: *const PingPongArena) bool {
        return self.passive_clear;
    }

    /// Resolve a slice. Returned bytes alias the arena buffer; valid
    /// only as long as the lifetime contract (see file header) holds.
    pub fn resolve(self: *const PingPongArena, slice: DecodedSlice) []const u8 {
        const arena: *const Arena = switch (slice.arena_id) {
            .a => &self.a,
            .b => &self.b,
        };
        return arena.buf[slice.offset..][0..slice.len];
    }
};

// ── Tests ────────────────────────────────────────────────────────────

test "arena: single-arena writes resolve correctly" {
    var ar = try PingPongArena.init(std.testing.allocator, 64);
    defer ar.deinit();

    const r1 = ar.write("hello");
    try std.testing.expect(r1 == .ok);
    const r2 = ar.write(" world");
    try std.testing.expect(r2 == .ok);

    try std.testing.expectEqualStrings("hello", ar.resolve(r1.ok));
    try std.testing.expectEqualStrings(" world", ar.resolve(r2.ok));
    try std.testing.expectEqual(ArenaId.a, r1.ok.arena_id);
    try std.testing.expectEqual(ArenaId.a, r2.ok.arena_id);
}

test "arena: swap-needed when arena fills, swap+retry succeeds" {
    var ar = try PingPongArena.init(std.testing.allocator, 8);
    defer ar.deinit();

    // Fill arena.a.
    const r1 = ar.write("12345");
    try std.testing.expect(r1 == .ok);
    const r2 = ar.write("678");
    try std.testing.expect(r2 == .ok);

    // No room for "9".
    const r3 = ar.write("9");
    try std.testing.expectEqual(PingPongArena.WriteResult.swap_needed, r3);

    // Producer swaps; consumer hasn't confirmed yet but we test
    // that the new-active (b) starts empty so a write lands.
    ar.swap();
    const r4 = ar.write("9");
    try std.testing.expect(r4 == .ok);
    try std.testing.expectEqual(ArenaId.b, r4.ok.arena_id);

    // Old slices still resolve correctly.
    try std.testing.expectEqualStrings("12345", ar.resolve(r1.ok));
    try std.testing.expectEqualStrings("678", ar.resolve(r2.ok));
    try std.testing.expectEqualStrings("9", ar.resolve(r4.ok));
}

test "arena: confirmSwap resets the retired arena for reuse" {
    var ar = try PingPongArena.init(std.testing.allocator, 8);
    defer ar.deinit();

    _ = ar.write("12345678"); // fills arena.a
    ar.swap(); // active = b
    _ = ar.write("abcd"); // arena.b cursor = 4

    ar.confirmSwap(.a); // consumer drained references into a
    try std.testing.expect(ar.passive_clear);
    try std.testing.expectEqual(@as(u32, 0), ar.a.cursor); // reset

    // Now another swap-needed cycle should reuse arena.a cleanly.
    const r1 = ar.write("efgh"); // arena.b cursor = 8 (full)
    try std.testing.expect(r1 == .ok);
    const r2 = ar.write("x");
    try std.testing.expectEqual(PingPongArena.WriteResult.swap_needed, r2);
    ar.swap(); // active = a (was previously confirmed-clear)
    const r3 = ar.write("x");
    try std.testing.expect(r3 == .ok);
    try std.testing.expectEqual(ArenaId.a, r3.ok.arena_id);
}

test "arena: growPassive doubles capacity on saturation" {
    var ar = try PingPongArena.init(std.testing.allocator, 4);
    defer ar.deinit();

    _ = ar.write("1234"); // a full
    ar.swap(); // active = b
    _ = ar.write("5678"); // b full

    // Both arenas saturated. Producer tries to grow the passive (a)
    // but it's not confirmed-clear — assert in growPassive would
    // fire. Confirm first.
    ar.confirmSwap(.a);

    try ar.growPassive(8);
    try std.testing.expectEqual(@as(usize, 8), ar.a.buf.len);

    // Now another swap can land 8 bytes in arena.a.
    _ = ar.write("9"); // b: swap_needed since full
    // Skipping the saturation write; just verify the grown arena
    // accepts more than 4 bytes.
    ar.swap();
    const big = ar.write("12345678");
    try std.testing.expect(big == .ok);
    try std.testing.expectEqualStrings("12345678", ar.resolve(big.ok));
}
