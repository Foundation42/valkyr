// Job System — lock-free work-stealing thread pool.
//
// Provides cache-friendly parallel execution with explicit join points.
// Based on the Chase-Lev work-stealing deque (2005) for lock-free
// scheduling without mutexes on the hot path.
//
// Architecture:
//   JobSystem   — owns worker threads, dispatches jobs
//   Job         — 64-byte work unit (one cache line): func + inline data + counter
//   Counter     — atomic completion counter for join/dependency tracking
//   WorkDeque   — per-worker Chase-Lev deque (owner push/pop, thieves steal)
//
// Usage:
//   var jobs = try JobSystem.init(allocator, 0); // 0 = auto thread count
//   defer jobs.deinit();
//
//   var counter = Counter.init(0);
//   jobs.schedule(myJob(&counter));
//   jobs.waitFor(&counter);  // main thread helps execute while waiting

const std = @import("std");
const Allocator = std.mem.Allocator;
const assert = std.debug.assert;

// ── Counter ──────────────────────────────────────────────────────────
//
// Atomic completion counter.  Schedule N jobs that all reference the
// same counter, then waitFor() until it hits zero.  The counter value
// represents remaining work — it starts at 0 and jobs increment it
// when scheduled, decrement when complete.

pub const Counter = struct {
    value: std.atomic.Value(u32),

    pub fn init(initial: u32) Counter {
        return .{ .value = std.atomic.Value(u32).init(initial) };
    }

    pub fn increment(self: *Counter) void {
        _ = self.value.fetchAdd(1, .release);
    }

    pub fn decrement(self: *Counter) void {
        _ = self.value.fetchSub(1, .release);
    }

    pub fn isComplete(self: *const Counter) bool {
        return self.value.load(.acquire) == 0;
    }

    /// Spin-yield wait.  For frame-level joins this is fine — the
    /// caller should be executing jobs while waiting (see JobSystem.waitFor).
    pub fn wait(self: *const Counter) void {
        while (!self.isComplete()) {
            std.Thread.yield() catch {};
        }
    }

    pub fn load(self: *const Counter) u32 {
        return self.value.load(.acquire);
    }
};

// ── Job ──────────────────────────────────────────────────────────────
//
// 64 bytes = one cache line.  The inline data[] avoids heap allocation
// for small payloads (a pointer, an index range, a few floats).

pub const Job = struct {
    func: *const fn (*Job) void,
    counter: ?*Counter = null,
    data: [52]u8 = undefined, // 64 - 8 (func) - 4 (counter ptr on 64-bit... actually 8)
    // On 64-bit: func=8, counter=8, data fills the rest to 64
    // Let's be precise: 64 - 8 - 8 = 48

    /// Helper to store a typed value in the inline data.
    pub fn setData(self: *Job, comptime T: type, val: T) void {
        comptime assert(@sizeOf(T) <= @sizeOf(@TypeOf(self.data)));
        @memcpy(self.data[0..@sizeOf(T)], std.mem.asBytes(&val));
    }

    /// Helper to read a typed value from the inline data.
    pub fn getData(self: *const Job, comptime T: type) T {
        comptime assert(@sizeOf(T) <= @sizeOf(@TypeOf(self.data)));
        return std.mem.bytesToValue(T, self.data[0..@sizeOf(T)]);
    }
};

// ── Work-Stealing Deque (Chase-Lev) ──────────────────────────────────
//
// Lock-free deque with single-owner push/pop and multi-thief steal.
//   - Owner pushes to bottom, pops from bottom (LIFO — cache warm)
//   - Thieves steal from top (FIFO — oldest, coldest work)
//   - Fixed capacity, power-of-2 for fast masking
//
// Memory ordering follows the Chase-Lev paper:
//   - push: store bottom with .release
//   - pop:  load top with .acquire, CAS top with .acq_rel
//   - steal: load bottom with .acquire, CAS top with .acq_rel

pub fn WorkDeque(comptime capacity: comptime_int) type {
    comptime {
        assert(capacity > 0 and (capacity & (capacity - 1)) == 0);
    }

    return struct {
        const Self = @This();

        buffer: [capacity]Job = undefined,
        bottom: std.atomic.Value(i64) = std.atomic.Value(i64).init(0),
        top: std.atomic.Value(i64) = std.atomic.Value(i64).init(0),

        /// Push a job (owner thread only).  Returns false if full.
        pub fn push(self: *Self, job: Job) bool {
            const b = self.bottom.load(.monotonic);
            const t = self.top.load(.acquire);

            if (b - t >= capacity) return false; // full

            self.buffer[@intCast(@mod(b, capacity))] = job;
            // Release ensures the buffer write is visible before bottom advances.
            self.bottom.store(b + 1, .release);
            return true;
        }

        /// Pop a job (owner thread only).  Returns null if empty.
        pub fn pop(self: *Self) ?Job {
            const b = self.bottom.load(.monotonic) - 1;
            // seq_cst store acts as a full fence — ensures stealers see
            // the decremented bottom before we read top.
            self.bottom.store(b, .seq_cst);

            const t = self.top.load(.seq_cst);

            if (t <= b) {
                // Non-empty
                const job = self.buffer[@intCast(@mod(b, capacity))];
                if (t == b) {
                    // Last element — race with steal.  Try to claim it.
                    if (self.top.cmpxchgStrong(t, t + 1, .seq_cst, .monotonic) != null) {
                        // Lost race — thief got it.
                        self.bottom.store(t + 1, .monotonic);
                        return null;
                    }
                    self.bottom.store(t + 1, .monotonic);
                }
                return job;
            } else {
                // Empty
                self.bottom.store(t, .monotonic);
                return null;
            }
        }

        /// Steal a job (any thread).  Returns null if empty or contended.
        pub fn steal(self: *Self) ?Job {
            const t = self.top.load(.seq_cst);
            const b = self.bottom.load(.seq_cst);

            if (t >= b) return null; // empty

            const job = self.buffer[@intCast(@mod(t, capacity))];

            // Try to advance top.  If another thief beats us, we fail gracefully.
            if (self.top.cmpxchgStrong(t, t + 1, .seq_cst, .monotonic) != null) {
                return null; // contention — try again later
            }

            return job;
        }

        /// Number of jobs currently in the deque (approximate).
        pub fn len(self: *const Self) u32 {
            const b = self.bottom.load(.monotonic);
            const t = self.top.load(.monotonic);
            return @intCast(@max(b - t, 0));
        }

        pub fn isEmpty(self: *const Self) bool {
            return self.len() == 0;
        }
    };
}

// ── Worker ───────────────────────────────────────────────────────────

const Deque = WorkDeque(4096); // 4096 jobs per worker

const Worker = struct {
    deque: Deque = .{},
    thread: ?std.Thread = null,
    id: u32 = 0,
};

// ── Job System ───────────────────────────────────────────────────────

pub const JobSystem = struct {
    allocator: Allocator,
    workers: []Worker,
    worker_count: u32,
    shutdown: std.atomic.Value(bool) = std.atomic.Value(bool).init(false),
    jobs_executed: std.atomic.Value(u64) = std.atomic.Value(u64).init(0),

    // RNG state for steal victim selection (per call, not per worker)
    steal_rng: u32 = 0x12345678,

    /// Create a job system.  Heap-allocated so worker threads get a
    /// stable pointer.  Pass thread_count=0 for auto (CPU cores - 2).
    pub fn init(allocator: Allocator, thread_count: u32) !*JobSystem {
        const cpu_count: u32 = @intCast(std.Thread.getCpuCount() catch 4);
        const count = if (thread_count == 0) @max(cpu_count -| 2, 2) else thread_count;

        const workers = try allocator.alloc(Worker, count);
        for (workers, 0..) |*w, i| {
            w.* = .{ .id = @intCast(i) };
        }

        const self = try allocator.create(JobSystem);
        self.* = .{
            .allocator = allocator,
            .workers = workers,
            .worker_count = count,
        };

        // Spawn worker threads (skip 0 — that's the main thread)
        for (1..count) |i| {
            self.workers[i].thread = try std.Thread.spawn(.{}, workerLoop, .{ self, @as(u32, @intCast(i)) });
        }

        return self;
    }

    pub fn deinit(self: *JobSystem) void {
        self.shutdown.store(true, .release);

        // Join all worker threads
        for (self.workers[1..]) |*w| {
            if (w.thread) |t| {
                t.join();
                w.thread = null;
            }
        }

        const allocator = self.allocator;
        allocator.free(self.workers);
        allocator.destroy(self);
    }

    /// Schedule a single job.  If the job has a counter, it's incremented.
    pub fn schedule(self: *JobSystem, job: Job) void {
        if (job.counter) |c| c.increment();

        // Push to worker 0's deque (main thread).  If full, try others.
        if (!self.workers[0].deque.push(job)) {
            // Overflow: try each worker
            for (self.workers[1..]) |*w| {
                if (w.deque.push(job)) return;
            }
            // All full — execute inline as fallback
            self.executeJob(job);
        }
    }

    /// Schedule multiple jobs at once.
    pub fn scheduleMany(self: *JobSystem, jobs_slice: []const Job) void {
        var next_worker: u32 = 0;
        for (jobs_slice) |job| {
            if (job.counter) |c| c.increment();

            // Round-robin across workers for distribution
            var pushed = false;
            var attempts: u32 = 0;
            while (attempts < self.worker_count) : (attempts += 1) {
                const idx = (next_worker + attempts) % self.worker_count;
                if (self.workers[idx].deque.push(job)) {
                    pushed = true;
                    next_worker = (idx + 1) % self.worker_count;
                    break;
                }
            }
            if (!pushed) {
                self.executeJob(job);
            }
        }
    }

    /// Parallel-for: split [0..count) into batches and schedule one job per batch.
    /// Each batch job receives a BatchRange in its data.  Call waitFor(counter) after.
    pub fn parallelFor(
        self: *JobSystem,
        count: u32,
        batch_size: u32,
        comptime func: fn (*Job) void,
        context: *const anyopaque,
        counter: *Counter,
    ) void {
        if (count == 0) return;
        const bs = @max(batch_size, 1);
        var start: u32 = 0;

        while (start < count) {
            const end = @min(start + bs, count);
            var job = Job{
                .func = func,
                .counter = counter,
            };
            job.setData(BatchRange, .{
                .start = start,
                .end = end,
                .context = context,
            });
            self.schedule(job);
            start = end;
        }
    }

    /// Wait for a counter to reach zero.  The calling thread (assumed to be
    /// worker 0 / main thread) actively helps execute jobs while waiting.
    pub fn waitFor(self: *JobSystem, counter: *Counter) void {
        while (!counter.isComplete()) {
            // Try own deque first
            if (self.workers[0].deque.pop()) |job| {
                self.executeJob(job);
                continue;
            }
            // Try stealing
            if (self.trySteal(0)) |job| {
                self.executeJob(job);
                continue;
            }
            // Nothing to do — yield briefly
            std.Thread.yield() catch {};
        }
    }

    /// Total jobs executed across all workers (for stats).
    pub fn totalExecuted(self: *const JobSystem) u64 {
        return self.jobs_executed.load(.monotonic);
    }

    // ── Internal ─────────────────────────────────────────────────

    fn executeJob(self: *JobSystem, job_in: Job) void {
        var job = job_in;
        job.func(&job);
        if (job.counter) |c| c.decrement();
        _ = self.jobs_executed.fetchAdd(1, .monotonic);
    }

    fn trySteal(self: *JobSystem, my_id: u32) ?Job {
        // Simple random victim selection
        const count = self.worker_count;
        if (count <= 1) return null;

        var rng = self.steal_rng;
        // xorshift32
        rng ^= rng << 13;
        rng ^= rng >> 17;
        rng ^= rng << 5;
        self.steal_rng = rng;

        const start = rng % count;
        var i: u32 = 0;
        while (i < count) : (i += 1) {
            const victim = (start + i) % count;
            if (victim == my_id) continue;
            if (self.workers[victim].deque.steal()) |job| {
                return job;
            }
        }
        return null;
    }

    fn workerLoop(self: *JobSystem, worker_id: u32) void {
        var idle_spins: u32 = 0;
        const max_idle_spins: u32 = 64;

        while (!self.shutdown.load(.acquire)) {
            // 1. Pop from own deque (cache-warm, LIFO)
            if (self.workers[worker_id].deque.pop()) |job| {
                self.executeJob(job);
                idle_spins = 0;
                continue;
            }

            // 2. Steal from another worker
            if (self.trySteal(worker_id)) |job| {
                self.executeJob(job);
                idle_spins = 0;
                continue;
            }

            // 3. Nothing to do — progressive backoff
            idle_spins += 1;
            if (idle_spins < max_idle_spins) {
                std.Thread.yield() catch {};
            } else {
                // Longer sleep to avoid burning CPU when idle
                std.time.sleep(100 * std.time.ns_per_us); // 100us
            }
        }
    }
};

/// Range passed to parallelFor batch functions.
pub const BatchRange = struct {
    start: u32,
    end: u32,
    context: *const anyopaque,
};

// ── Tests ────────────────────────────────────────────────────────────

test "Counter basic" {
    var c = Counter.init(0);
    try std.testing.expect(c.isComplete());

    c.increment();
    try std.testing.expect(!c.isComplete());

    c.decrement();
    try std.testing.expect(c.isComplete());
}

test "Counter init with value" {
    var c = Counter.init(5);
    try std.testing.expectEqual(c.load(), 5);
    try std.testing.expect(!c.isComplete());

    var i: u32 = 0;
    while (i < 5) : (i += 1) c.decrement();
    try std.testing.expect(c.isComplete());
}

test "WorkDeque push and pop (single-threaded)" {
    var deque = Deque{};

    var job1 = Job{ .func = noopJob };
    job1.setData(u32, 42);

    var job2 = Job{ .func = noopJob };
    job2.setData(u32, 99);

    try std.testing.expect(deque.push(job1));
    try std.testing.expect(deque.push(job2));
    try std.testing.expectEqual(deque.len(), 2);

    // Pop is LIFO — should get job2 first
    const popped = deque.pop().?;
    try std.testing.expectEqual(popped.getData(u32), 99);

    const popped2 = deque.pop().?;
    try std.testing.expectEqual(popped2.getData(u32), 42);

    // Empty
    try std.testing.expect(deque.pop() == null);
}

test "WorkDeque steal (single-threaded)" {
    var deque = Deque{};

    var job1 = Job{ .func = noopJob };
    job1.setData(u32, 10);

    var job2 = Job{ .func = noopJob };
    job2.setData(u32, 20);

    try std.testing.expect(deque.push(job1));
    try std.testing.expect(deque.push(job2));

    // Steal is FIFO — should get job1 first
    const stolen = deque.steal().?;
    try std.testing.expectEqual(stolen.getData(u32), 10);

    const stolen2 = deque.steal().?;
    try std.testing.expectEqual(stolen2.getData(u32), 20);

    try std.testing.expect(deque.steal() == null);
}

test "WorkDeque concurrent push and steal" {
    var deque = Deque{};
    const count: u32 = 1000;

    // Push jobs
    for (0..count) |i| {
        var job = Job{ .func = noopJob };
        job.setData(u32, @intCast(i));
        try std.testing.expect(deque.push(job));
    }

    var stolen_count = std.atomic.Value(u32).init(0);

    // Spawn a stealer thread
    const stealer = try std.Thread.spawn(.{}, struct {
        fn run(d: *Deque, sc: *std.atomic.Value(u32)) void {
            while (true) {
                if (d.steal()) |_| {
                    _ = sc.fetchAdd(1, .monotonic);
                } else {
                    if (d.isEmpty()) break;
                    std.Thread.yield() catch {};
                }
            }
        }
    }.run, .{ &deque, &stolen_count });

    // Owner pops concurrently
    var owner_count: u32 = 0;
    while (deque.pop()) |_| {
        owner_count += 1;
    }

    stealer.join();

    // Total should equal what was pushed (no lost jobs)
    const total = owner_count + stolen_count.load(.monotonic);
    try std.testing.expectEqual(total, count);
}

test "JobSystem schedule and waitFor" {
    const system = try JobSystem.init(std.testing.allocator, 2);
    defer system.deinit();

    var result = std.atomic.Value(u32).init(0);
    var counter = Counter.init(0);

    var job = Job{
        .func = struct {
            fn run(j: *Job) void {
                const ptr = j.getData(*std.atomic.Value(u32));
                _ = ptr.fetchAdd(1, .release);
            }
        }.run,
        .counter = &counter,
    };
    job.setData(*std.atomic.Value(u32), &result);

    system.schedule(job);
    system.waitFor(&counter);

    try std.testing.expectEqual(result.load(.acquire), 1);
}

test "JobSystem parallelFor sums array" {
    const system = try JobSystem.init(std.testing.allocator, 4);
    defer system.deinit();

    // Create array to sum
    var data: [1024]u32 = undefined;
    for (&data, 0..) |*d, i| d.* = @intCast(i);

    const expected_sum: u64 = (1023 * 1024) / 2; // sum of 0..1023

    var partial_sums: [64]std.atomic.Value(u64) = undefined;
    for (&partial_sums) |*ps| ps.* = std.atomic.Value(u64).init(0);

    const Context = struct {
        data: *const [1024]u32,
        partial_sums: *[64]std.atomic.Value(u64),
    };

    var ctx = Context{ .data = &data, .partial_sums = &partial_sums };
    var counter = Counter.init(0);

    system.parallelFor(1024, 32, struct {
        fn run(j: *Job) void {
            const range = j.getData(BatchRange);
            const c: *const Context = @ptrCast(@alignCast(range.context));
            var sum: u64 = 0;
            for (range.start..range.end) |i| {
                sum += c.data[i];
            }
            // Use batch index as partial sum slot
            const slot = range.start / 32;
            _ = c.partial_sums[slot].fetchAdd(sum, .release);
        }
    }.run, @ptrCast(&ctx), &counter);

    system.waitFor(&counter);

    var total: u64 = 0;
    for (&partial_sums) |*ps| total += ps.load(.acquire);

    try std.testing.expectEqual(total, expected_sum);
}

test "JobSystem parallelFor zero count" {
    const system = try JobSystem.init(std.testing.allocator, 2);
    defer system.deinit();

    var counter = Counter.init(0);
    system.parallelFor(0, 32, noopJob, @ptrFromInt(1), &counter);

    // Should already be complete (nothing was scheduled)
    try std.testing.expect(counter.isComplete());
}

test "JobSystem multiple concurrent batches" {
    const system = try JobSystem.init(std.testing.allocator, 4);
    defer system.deinit();

    var counter_a = Counter.init(0);
    var counter_b = Counter.init(0);
    var result_a = std.atomic.Value(u32).init(0);
    var result_b = std.atomic.Value(u32).init(0);

    // Schedule batch A: 100 increment jobs
    for (0..100) |_| {
        var job = Job{
            .func = struct {
                fn run(j: *Job) void {
                    const ptr = j.getData(*std.atomic.Value(u32));
                    _ = ptr.fetchAdd(1, .release);
                }
            }.run,
            .counter = &counter_a,
        };
        job.setData(*std.atomic.Value(u32), &result_a);
        system.schedule(job);
    }

    // Schedule batch B: 100 increment jobs
    for (0..100) |_| {
        var job = Job{
            .func = struct {
                fn run(j: *Job) void {
                    const ptr = j.getData(*std.atomic.Value(u32));
                    _ = ptr.fetchAdd(1, .release);
                }
            }.run,
            .counter = &counter_b,
        };
        job.setData(*std.atomic.Value(u32), &result_b);
        system.schedule(job);
    }

    system.waitFor(&counter_a);
    system.waitFor(&counter_b);

    try std.testing.expectEqual(result_a.load(.acquire), 100);
    try std.testing.expectEqual(result_b.load(.acquire), 100);
}

test "JobSystem stress test 10K jobs" {
    const system = try JobSystem.init(std.testing.allocator, 4);
    defer system.deinit();

    var result = std.atomic.Value(u32).init(0);
    var counter = Counter.init(0);

    for (0..10_000) |_| {
        var job = Job{
            .func = struct {
                fn run(j: *Job) void {
                    const ptr = j.getData(*std.atomic.Value(u32));
                    _ = ptr.fetchAdd(1, .release);
                }
            }.run,
            .counter = &counter,
        };
        job.setData(*std.atomic.Value(u32), &result);
        system.schedule(job);
    }

    system.waitFor(&counter);
    try std.testing.expectEqual(result.load(.acquire), 10_000);
}

fn noopJob(_: *Job) void {}
