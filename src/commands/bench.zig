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
const runtime = @import("../runtime.zig");
const cpu_forward = @import("../cpu/forward.zig");

pub fn runBench(gpa: std.mem.Allocator, dir_path: []const u8, n_steps: usize) !void {
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

    var k = try runtime.ChatKernels.init(&ctx, gm.precision, cfg.family);
    defer k.deinit();

    var rec = try gpu_recorder.Recorder.init(&ctx, 512, 2048);
    defer rec.deinit();

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

    for (0..n_steps) |step| {
        if (step > 0) try rec.reset();
        try rec.begin();
        try runtime.recordForwardStep(&rec, &sc, &gm, &kv, cfg, &k, step, current, null, true);
        const t0 = std.time.nanoTimestamp();
        try rec.endAndSubmit();
        const t1 = std.time.nanoTimestamp();
        samples[step] = @as(f64, @floatFromInt(t1 - t0)) / 1_000_000.0;

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
