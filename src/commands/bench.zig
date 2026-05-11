//! `--bench`: cold/warm forward-step timing on a loaded model. Loads
//! once, autoregressively decodes greedy from BOS for N steps, and
//! reports cold step + warm mean/median/min/max/p99/throughput, plus
//! early-vs-late position-dependent cost. Extracted from main.zig.

const std = @import("std");
const vk = @import("../gpu/vk.zig");
const model_mod = @import("../model.zig");
const gpu_model = @import("../gpu/model.zig");
const gpu_scratch = @import("../gpu/scratch.zig");
const gpu_recorder = @import("../gpu/recorder.zig");
const pipeline = @import("../gpu/pipeline.zig");
const runtime = @import("../runtime.zig");
const cpu_forward = @import("../cpu/forward.zig");
const shaders = @import("shaders");

pub fn runBench(gpa: std.mem.Allocator, dir_path: []const u8, n_steps: usize, tq4v: bool, ts: bool) !void {
    var cpu = try model_mod.Model.load(gpa, dir_path);
    defer cpu.deinit();
    const cfg = cpu.config;

    var ctx = try vk.Context.init(gpa);
    defer ctx.deinit();

    const stdout = std.io.getStdOut().writer();
    try stdout.print("device: {s}\n", .{ctx.deviceName()});
    try stdout.print("model: {s}    layers={d}  hidden={d}  vocab={d}\n", .{
        @tagName(cfg.family), cfg.num_hidden_layers, cfg.hidden_size, cfg.vocab_size,
    });

    const t_up0 = std.time.nanoTimestamp();
    var gm = try gpu_model.GpuModel.upload(gpa, &ctx, &cpu, .bf16_matmul);
    defer gm.deinit(ctx.device);
    const t_up1 = std.time.nanoTimestamp();
    const upload_ms = @as(f64, @floatFromInt(t_up1 - t_up0)) / 1_000_000.0;
    try stdout.print("upload (bf16 matmul): {d:.0} ms\n\n", .{upload_ms});

    const max_pos: usize = @max(n_steps + 16, 128);
    var sc = try gpu_scratch.GpuScratch.init(&ctx, cfg, max_pos);
    defer sc.deinit(ctx.device);
    var kv = try gpu_scratch.GpuKvCache.init(gpa, &ctx, cfg, max_pos);
    defer kv.deinit(ctx.device);

    var k = try runtime.ChatKernels.init(&ctx, gm.precision, cfg.family, @intCast(cfg.head_dim));
    defer k.deinit();

    // Optional TQ4-V cache (asymmetric K=fp / V=TQ4). Mirrors the chat
    // command's setup — only allocated when --tq4v was passed. The fp32
    // kv buffer above is still used for K; we just substitute the V path.
    var kv_tq4 = if (tq4v)
        try gpu_scratch.GpuKvCacheTq4.init(gpa, &ctx, cfg, max_pos)
    else
        null;
    defer if (kv_tq4) |*c| c.deinit(ctx.device);
    const tq_pack_spv: []align(4) const u8 = if (cfg.head_dim == 128)
        &shaders.tq4_pack_to_cache128
    else
        &shaders.tq4_pack_to_cache;
    const tq_unpack_spv: []align(4) const u8 = if (cfg.head_dim == 128)
        &shaders.tq4_unpack128
    else
        &shaders.tq4_unpack256;
    var tq_pack: ?pipeline.Kernel = if (tq4v)
        try pipeline.Kernel.init(&ctx, tq_pack_spv, 2, @sizeOf(runtime.Tq4PackPush))
    else
        null;
    defer if (tq_pack) |*kk| kk.deinit();
    var tq_unpack: ?pipeline.Kernel = if (tq4v)
        try pipeline.Kernel.init(&ctx, tq_unpack_spv, 2, 0)
    else
        null;
    defer if (tq_unpack) |*kk| kk.deinit();
    const tq4_hooks: ?runtime.Tq4VHooks = if (tq4v)
        runtime.Tq4VHooks{ .pack = &tq_pack.?, .unpack = &tq_unpack.?, .cache = &kv_tq4.? }
    else
        null;
    if (tq4v) try stdout.print("kv: K=fp32 V=TQ4 (asymmetric)\n", .{});

    // Big enough for Qwen3-0.6B / Gemma 2B chat-decode chains — both
    // sit comfortably under 1k dispatches per step. The original 512
    // worked when descriptors were silently over-allocated by the
    // driver, but the timestamp pool enforces the cap strictly.
    const rec_cap: u32 = 2048;
    var rec = try gpu_recorder.Recorder.init(&ctx, rec_cap, rec_cap * 4);
    defer rec.deinit();

    // Optional per-dispatch GPU timing. Wrapped in a local flag so we
    // can gracefully fall back if the queue family reports
    // timestampValidBits=0 (rare on modern hardware, but the embedded
    // path may run on stranger queues).
    var ts_enabled: bool = false;
    if (ts) {
        rec.enableTimestamps(rec_cap) catch |err| switch (err) {
            error.TimestampsUnsupported => try stdout.print(
                "warning: queue family on {s} doesn't support timestamps; --ts ignored\n",
                .{ctx.deviceName()},
            ),
            else => return err,
        };
        ts_enabled = rec.nsPerTick() != 0; // enableTimestamps sets period
    }

    const logits = try gpa.alloc(f32, cfg.vocab_size);
    defer gpa.free(logits);

    const bos = tok_bos: {
        // Use BOS token id from config when set; otherwise fall back to 2
        // (Gemma's bos id) so the bench works without a tokenizer load.
        if (cfg.bos_token_id) |b| break :tok_bos b;
        break :tok_bos 2;
    };

    // Time each forward, advancing position each step. Position 0 uses
    // bos; subsequent steps feed back the argmax (a real autoregressive
    // greedy run, not a microbenchmark on a fixed input).
    const samples = try gpa.alloc(f64, n_steps);
    defer gpa.free(samples);
    var current: u32 = bos;

    // Per-dispatch-index accumulators (warm steps only — step 0 is
    // cold and excluded from the breakdown). Indices line up with the
    // recordForwardStep call order; mapping back to kernel names is a
    // pencil-and-paper exercise against runtime.recordForwardStep /
    // recordOneLayer source. Sum-in-f64 to avoid precision drift over
    // thousands of warm steps.
    const ts_sum_ns = try gpa.alloc(f64, rec_cap);
    defer gpa.free(ts_sum_ns);
    const ts_gap_sum_ns = try gpa.alloc(f64, rec_cap);
    defer gpa.free(ts_gap_sum_ns);
    @memset(ts_sum_ns, 0);
    @memset(ts_gap_sum_ns, 0);
    var ts_warm_count: u32 = 0;
    var ts_n_dispatched: u32 = 0;
    // Heap-allocated to match rec_cap (2*rec_cap u64s, ~32 KiB) —
    // safer than a fixed stack buffer if rec_cap ever grows.
    const ts_tick_buf = try gpa.alloc(u64, 2 * rec_cap);
    defer gpa.free(ts_tick_buf);
    var ts_total_dispatch_ns: f64 = 0;
    var ts_total_gap_ns: f64 = 0;

    for (0..n_steps) |step| {
        if (step > 0) try rec.reset();
        try rec.begin();
        try runtime.recordForwardStep(&rec, &sc, &gm, &kv, cfg, &k, step, current, tq4_hooks, true);
        const t0 = std.time.nanoTimestamp();
        try rec.endAndSubmit();
        const t1 = std.time.nanoTimestamp();
        samples[step] = @as(f64, @floatFromInt(t1 - t0)) / 1_000_000.0;

        if (ts_enabled and step >= 1) {
            const ticks = try rec.lastTimestamps(ts_tick_buf);
            const n_disp: u32 = @intCast(ticks.len / 2);
            ts_n_dispatched = n_disp;
            const period = rec.nsPerTick();
            var step_dispatch_ns: f64 = 0;
            var step_gap_ns: f64 = 0;
            var prev_end: u64 = 0;
            var d: u32 = 0;
            while (d < n_disp) : (d += 1) {
                const start = ticks[2 * d];
                const end = ticks[2 * d + 1];
                const dur_ns = @as(f64, @floatFromInt(end -% start)) * period;
                ts_sum_ns[d] += dur_ns;
                step_dispatch_ns += dur_ns;
                if (d > 0) {
                    const gap_ns = @as(f64, @floatFromInt(start -% prev_end)) * period;
                    ts_gap_sum_ns[d] += gap_ns;
                    step_gap_ns += gap_ns;
                }
                prev_end = end;
            }
            ts_total_dispatch_ns += step_dispatch_ns;
            ts_total_gap_ns += step_gap_ns;
            ts_warm_count += 1;
        }

        try sc.logits.readBack(&ctx, f32, logits);
        current = @intCast(cpu_forward.argmax(logits));
    }

    // Stats. The first sample is the cold one (pipeline compile + cold
    // caches). Steady-state stats use samples[1..].
    const cold = samples[0];
    var warm_sum: f64 = 0;
    var warm_min: f64 = std.math.inf(f64);
    var warm_max: f64 = 0;
    for (samples[1..]) |s| {
        warm_sum += s;
        if (s < warm_min) warm_min = s;
        if (s > warm_max) warm_max = s;
    }
    const warm_mean = warm_sum / @as(f64, @floatFromInt(samples.len - 1));

    // p50 and p99 via a sorted copy (cheap at n_steps ≤ a few hundred).
    const sorted = try gpa.alloc(f64, samples.len - 1);
    defer gpa.free(sorted);
    @memcpy(sorted, samples[1..]);
    std.mem.sort(f64, sorted, {}, std.sort.asc(f64));
    const p50 = sorted[sorted.len / 2];
    const p99_idx: usize = @min(sorted.len - 1, (sorted.len * 99) / 100);
    const p99 = sorted[p99_idx];

    try stdout.print("forwards: {d} steps (positions 0..{d})\n", .{ n_steps, n_steps - 1 });
    try stdout.print("  cold (step 0)   : {d:.2} ms\n", .{cold});
    try stdout.print("  warm mean       : {d:.2} ms\n", .{warm_mean});
    try stdout.print("  warm median     : {d:.2} ms\n", .{p50});
    try stdout.print("  warm min/max    : {d:.2} / {d:.2} ms\n", .{ warm_min, warm_max });
    try stdout.print("  warm p99        : {d:.2} ms\n", .{p99});
    try stdout.print("  throughput      : {d:.1} tok/s (warm mean)\n", .{1000.0 / warm_mean});

    // Per-dispatch breakdown. The wall clock above includes CPU-side
    // submit + waitForFences; the GPU timeline (start-to-end of the
    // recorded chain) is dispatch + gap. The difference is queue/
    // submit overhead — usually < 1 ms but worth knowing.
    //
    // Gap time = sum of (start[i+1] - end[i]) across the chain. This
    // is barrier wait + driver scheduling between dispatches. If the
    // GPU "looks underutilised" (Christian's T-arc observation), this
    // is the bucket that exposes it: gap dominating dispatch means
    // the chain is serialised more than it needs to be.
    if (ts_enabled and ts_warm_count > 0) {
        const warmf: f64 = @floatFromInt(ts_warm_count);
        const mean_dispatch_ms = ts_total_dispatch_ns / warmf / 1_000_000.0;
        const mean_gap_ms = ts_total_gap_ns / warmf / 1_000_000.0;
        try stdout.print("\n  per-dispatch GPU timing (warm steps={d}, dispatches/step={d}):\n", .{ ts_warm_count, ts_n_dispatched });
        try stdout.print("    mean dispatch sum (active GPU): {d:.3} ms/step\n", .{mean_dispatch_ms});
        try stdout.print("    mean gap sum     (barrier wait): {d:.3} ms/step\n", .{mean_gap_ms});
        try stdout.print("    mean GPU timeline             : {d:.3} ms/step (dispatch + gap)\n", .{mean_dispatch_ms + mean_gap_ms});
        const cpu_overhead = warm_mean - (mean_dispatch_ms + mean_gap_ms);
        try stdout.print("    mean CPU/submit overhead      : {d:.3} ms/step (wall − GPU timeline)\n", .{cpu_overhead});

        // Top-K longest dispatches. Tracking by index lets the reader
        // map back to recordForwardStep call order; for a fixed-shape
        // chat decode the order is deterministic across steps.
        const TopK: u32 = 12;
        const Slot = struct { idx: u32, mean_ns: f64 };
        var top = [_]Slot{.{ .idx = 0, .mean_ns = 0 }} ** TopK;
        var d: u32 = 0;
        while (d < ts_n_dispatched) : (d += 1) {
            const mean = ts_sum_ns[d] / warmf;
            var min_i: u32 = 0;
            var k_idx: u32 = 1;
            while (k_idx < TopK) : (k_idx += 1) {
                if (top[k_idx].mean_ns < top[min_i].mean_ns) min_i = k_idx;
            }
            if (mean > top[min_i].mean_ns) top[min_i] = .{ .idx = d, .mean_ns = mean };
        }
        const lessThan = struct {
            fn lt(_: void, a: Slot, b: Slot) bool {
                return a.mean_ns > b.mean_ns;
            }
        };
        std.mem.sort(Slot, top[0..], {}, lessThan.lt);
        try stdout.print("    top {d} dispatches by mean ns:\n", .{TopK});
        for (top) |s| {
            if (s.mean_ns == 0) continue;
            const pct = if (mean_dispatch_ms > 0)
                s.mean_ns * 100.0 / (mean_dispatch_ms * 1_000_000.0)
            else
                0.0;
            try stdout.print("      [#{d:>3}] {d:>10.0} ns  ({d:>5.2}% of dispatch sum)\n", .{ s.idx, s.mean_ns, pct });
        }

        // Top-K longest gaps. A gap that approaches the dispatch
        // duration on either side means the GPU is sitting idle
        // between two consecutive dispatches that could have
        // overlapped (had we let them).
        var top_g = [_]Slot{.{ .idx = 0, .mean_ns = 0 }} ** TopK;
        var dg: u32 = 1;
        while (dg < ts_n_dispatched) : (dg += 1) {
            const mean = ts_gap_sum_ns[dg] / warmf;
            var min_i: u32 = 0;
            var k_idx: u32 = 1;
            while (k_idx < TopK) : (k_idx += 1) {
                if (top_g[k_idx].mean_ns < top_g[min_i].mean_ns) min_i = k_idx;
            }
            if (mean > top_g[min_i].mean_ns) top_g[min_i] = .{ .idx = dg, .mean_ns = mean };
        }
        std.mem.sort(Slot, top_g[0..], {}, lessThan.lt);
        try stdout.print("    top {d} gaps by mean ns (gap = start[i] − end[i−1]):\n", .{TopK});
        for (top_g) |s| {
            if (s.mean_ns == 0) continue;
            const pct = if (mean_gap_ms > 0)
                s.mean_ns * 100.0 / (mean_gap_ms * 1_000_000.0)
            else
                0.0;
            try stdout.print("      [#{d:>3}] {d:>10.0} ns  ({d:>5.2}% of gap sum)\n", .{ s.idx, s.mean_ns, pct });
        }
    }

    // Position-dependent cost: attention scoring grows linearly with
    // n_pos. Print early-vs-late timings to make it visible.
    if (n_steps >= 32) {
        var early: f64 = 0;
        for (samples[1..16]) |s| early += s;
        var late: f64 = 0;
        const start = samples.len - 16;
        for (samples[start..]) |s| late += s;
        try stdout.print("\n  pos-dependent cost (15-step rolling means):\n", .{});
        try stdout.print("    early (pos 1..15)         : {d:.2} ms/tok\n", .{early / 15.0});
        try stdout.print("    late  (pos {d}..{d}) : {d:.2} ms/tok\n", .{
            start, n_steps - 1, late / 16.0,
        });
    }
}
