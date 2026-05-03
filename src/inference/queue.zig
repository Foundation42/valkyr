//! Single-producer / single-consumer bounded ring buffer.
//!
//! Two threads: one producer, one consumer. Lock-free; push/pop are a
//! pair of atomic loads + one atomic store with release/acquire
//! ordering. Capacity is a power of two so the modulo wraps via mask.
//!
//! Why SPSC and not MPMC: in this codebase the producer is the HTTP
//! accept loop (or the embed host's render thread), the consumer is
//! the inference runner thread. Multiple HTTP server threads would
//! serialize behind a mutex *before* the queue — cheaper than CAS-ing
//! the head pointer on every push. If we ever need true multi-
//! producer, swap to a MPSC ring; the public API doesn't change.
//!
//! Cache-line padding: head and tail counters live on separate cache
//! lines so the producer and consumer don't false-share. Without
//! padding, every push invalidates the consumer's cache line for tail
//! and vice versa — a real perf cliff under load.

const std = @import("std");

/// Generic bounded SPSC ring. `cap_log2` is the log2 of capacity; max
/// is 30 (1 GiB items, 64-bit usize). Capacity must be a power of two
/// so head/tail wrap via mask without modulo.
pub fn SpscRing(comptime T: type, comptime cap_log2: u6) type {
    if (cap_log2 == 0 or cap_log2 > 30) @compileError("cap_log2 must be in [1,30]");
    const CAP: usize = @as(usize, 1) << cap_log2;
    const MASK: usize = CAP - 1;

    return struct {
        const Self = @This();

        /// Slot storage. Producer writes to slots[head & MASK] before
        /// advancing head; consumer reads from slots[tail & MASK]
        /// after observing head > tail.
        slots: [CAP]T = undefined,

        /// Producer cursor. Writes from producer thread, reads from
        /// consumer thread. Padded to its own cache line.
        head: std.atomic.Value(usize) align(64) = std.atomic.Value(usize).init(0),
        _pad_a: [64 - @sizeOf(std.atomic.Value(usize))]u8 = undefined,

        /// Consumer cursor. Writes from consumer thread, reads from
        /// producer thread.
        tail: std.atomic.Value(usize) align(64) = std.atomic.Value(usize).init(0),
        _pad_b: [64 - @sizeOf(std.atomic.Value(usize))]u8 = undefined,

        pub const capacity: usize = CAP;

        /// Producer-side push. Returns false if full. Caller decides
        /// whether to block, drop, or back off (HTTP returns 503 on
        /// pressure; embed host can retry next frame).
        pub fn tryPush(self: *Self, item: T) bool {
            // Acquire-load tail to see the latest consumer progress.
            const t = self.tail.load(.acquire);
            // Relaxed-load head: only this thread writes head.
            const h = self.head.load(.monotonic);
            if (h - t >= CAP) return false;
            self.slots[h & MASK] = item;
            // Release-store head so the consumer's matching
            // acquire-load sees the slot write.
            self.head.store(h + 1, .release);
            return true;
        }

        /// Consumer-side pop. Returns null if empty.
        pub fn tryPop(self: *Self) ?T {
            const h = self.head.load(.acquire);
            const t = self.tail.load(.monotonic);
            if (h == t) return null;
            const item = self.slots[t & MASK];
            self.tail.store(t + 1, .release);
            return item;
        }

        /// Snapshot length. For telemetry only; the value can shift
        /// between read and use.
        pub fn approxLen(self: *const Self) usize {
            const h = self.head.load(.monotonic);
            const t = self.tail.load(.monotonic);
            return h - t;
        }

        pub fn isEmpty(self: *const Self) bool {
            return self.approxLen() == 0;
        }

        pub fn isFull(self: *const Self) bool {
            return self.approxLen() >= CAP;
        }
    };
}

// ── Tests ────────────────────────────────────────────────────────────

test "spsc: push/pop ordering single-threaded" {
    var r: SpscRing(u32, 3) = .{}; // capacity 8
    try std.testing.expect(r.isEmpty());
    try std.testing.expectEqual(@as(usize, 0), r.approxLen());

    var i: u32 = 0;
    while (i < 5) : (i += 1) {
        try std.testing.expect(r.tryPush(i * 10));
    }
    try std.testing.expectEqual(@as(usize, 5), r.approxLen());

    i = 0;
    while (i < 5) : (i += 1) {
        const v = r.tryPop() orelse return error.UnexpectedEmpty;
        try std.testing.expectEqual(i * 10, v);
    }
    try std.testing.expect(r.isEmpty());
    try std.testing.expectEqual(@as(?u32, null), r.tryPop());
}

test "spsc: full edge — push fails at capacity, pop reopens slot" {
    var r: SpscRing(u32, 2) = .{}; // capacity 4
    try std.testing.expect(r.tryPush(1));
    try std.testing.expect(r.tryPush(2));
    try std.testing.expect(r.tryPush(3));
    try std.testing.expect(r.tryPush(4));
    try std.testing.expect(r.isFull());
    try std.testing.expect(!r.tryPush(5));

    try std.testing.expectEqual(@as(?u32, 1), r.tryPop());
    try std.testing.expect(r.tryPush(5));
    try std.testing.expect(r.isFull());
}

test "spsc: wraparound across MASK boundary" {
    var r: SpscRing(u32, 2) = .{}; // capacity 4
    var pushed: u32 = 0;
    var popped: u32 = 0;

    // Drive head/tail past CAP=4 to verify masking wraps.
    while (pushed < 16) : (pushed += 1) {
        try std.testing.expect(r.tryPush(pushed));
        const v = r.tryPop() orelse return error.UnexpectedEmpty;
        try std.testing.expectEqual(popped, v);
        popped += 1;
    }
    try std.testing.expect(r.isEmpty());
}

test "spsc: producer/consumer threads — 1M items, FIFO order" {
    var ring = try std.testing.allocator.create(SpscRing(u32, 10)); // 1024
    defer std.testing.allocator.destroy(ring);
    ring.* = .{};

    const N: u32 = 1 << 20;

    const Producer = struct {
        fn run(r: *SpscRing(u32, 10)) void {
            var i: u32 = 0;
            while (i < N) {
                if (r.tryPush(i)) {
                    i += 1;
                } else {
                    std.atomic.spinLoopHint();
                }
            }
        }
    };

    const t = try std.Thread.spawn(.{}, Producer.run, .{ring});

    var expected: u32 = 0;
    while (expected < N) {
        if (ring.tryPop()) |v| {
            try std.testing.expectEqual(expected, v);
            expected += 1;
        } else {
            std.atomic.spinLoopHint();
        }
    }
    t.join();
}
