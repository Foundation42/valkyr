//! `--session-smoke`, `--session-messages`, `--runner-smoke[-threaded]`:
//! headless exercises of the Session / InferenceRunner surface that
//! matryoshka's ai_demo runs in-engine. Extracted from main.zig.

const std = @import("std");
const vk = @import("../gpu/vk.zig");
const gpu_model = @import("../gpu/model.zig");
const gpu_recorder = @import("../gpu/recorder.zig");
const config_mod = @import("../config.zig");
const session_mod = @import("../session.zig");
const loader = @import("../loader.zig");
const inference_runner = @import("../inference/runner.zig");
const chat_template_mod = @import("../chat_template.zig");
const inference_proto = @import("../inference/proto.zig");

const SessionSmokeOnTokenCtx = struct {
    stdout: std.fs.File.Writer,
    family: config_mod.Family,
};

fn sessionSmokeOnToken(user: ?*anyopaque, tok_id: u32, decoded: []const u8) void {
    _ = tok_id;
    const ctx: *SessionSmokeOnTokenCtx = @ptrCast(@alignCast(user.?));
    // `decoded` is already display-ready: tokenizer.decodeForDisplay
    // resolved SentencePiece ▁ / byte-level Ġ / hex-byte fallback
    // tokens before this callback fired.
    ctx.stdout.print("{s}", .{decoded}) catch return;
}

pub fn runSessionSmoke(
    gpa: std.mem.Allocator,
    model_arg: []const u8,
    prompt: []const u8,
    max_new: u32,
    budget: session_mod.Budget,
    precision: gpu_model.Precision,
) !void {
    const stdout = std.io.getStdOut().writer();
    var ctx = try vk.Context.init(gpa);
    defer ctx.deinit();

    try stdout.print("device: {s}\n", .{ctx.deviceName()});
    try stdout.print("loading model {s}...\n", .{model_arg});

    const t0 = std.time.nanoTimestamp();
    var loaded = try loader.loadGpuModel(gpa, &ctx, model_arg, .{ .precision = precision });
    defer loaded.deinit(ctx.device);
    const t1 = std.time.nanoTimestamp();
    const cfg = loaded.config();
    try stdout.print(
        "loaded in {d:.1} s — family={s} layers={d} hybrid={}\n",
        .{ @as(f64, @floatFromInt(t1 - t0)) / 1e9, @tagName(cfg.family), cfg.num_hidden_layers, cfg.family.isHybrid() },
    );

    var on_token_ctx = SessionSmokeOnTokenCtx{ .stdout = stdout, .family = cfg.family };

    var sess = try session_mod.Session.init(gpa, &ctx, &loaded.gpu_model, &loaded.tokenizer, .{
        .budget = budget,
        .max_new_tokens = max_new,
        .on_token = sessionSmokeOnToken,
        .on_token_user = &on_token_ctx,
        .max_pos = 1024,
    });
    defer sess.deinit();
    try sess.appendPrompt(prompt);

    try stdout.print("\nprompt: \"{s}\"\n", .{prompt});
    try stdout.print("response: ", .{});

    // Recorder sized for the worst-case tick in pure `--budget-us` mode:
    // a single tick may record every layer + sample step before the time
    // gate fires. Layer-mode default (8 layers) fits in 512/4096 easily;
    // bump here so the smoke harness can also exercise unbounded-layer
    // time mode without exhausting the descriptor pool.
    var rec = try gpu_recorder.Recorder.init(&ctx, 4096, 16384);
    defer rec.deinit();

    const decode_t0 = std.time.nanoTimestamp();
    var ticks: u32 = 0;
    var sum_elapsed_us: u64 = 0;
    var sum_residual_signed: i64 = 0;
    var max_elapsed_us: u64 = 0;
    while (!sess.isDone() and ticks < 4096) : (ticks += 1) {
        try rec.reset();
        try rec.begin();
        const r = try sess.tickFrame(&rec);
        try rec.endAndSubmit();
        sum_elapsed_us += r.elapsed_us;
        sum_residual_signed += r.residual_us;
        if (r.elapsed_us > max_elapsed_us) max_elapsed_us = r.elapsed_us;
    }
    const decode_t1 = std.time.nanoTimestamp();
    const tokens = sess.generatedTokens();
    const wall_s = @as(f64, @floatFromInt(decode_t1 - decode_t0)) / 1e9;
    try stdout.print(
        "\n[{d} tok in {d:.0} ms over {d} ticks, {d:.1} tok/s]\n",
        .{ tokens.len, wall_s * 1000.0, ticks, @as(f64, @floatFromInt(tokens.len)) / wall_s },
    );
    if (ticks > 0) {
        const avg_elapsed = @as(f64, @floatFromInt(sum_elapsed_us)) / @as(f64, @floatFromInt(ticks));
        const avg_residual = @as(f64, @floatFromInt(sum_residual_signed)) / @as(f64, @floatFromInt(ticks));
        try stdout.print(
            "[budget={s}  per-tick: avg_elapsed={d:.1} µs  max_elapsed={d} µs  avg_residual={d:.1} µs]\n",
            .{ @tagName(budget), avg_elapsed, max_elapsed_us, avg_residual },
        );
    }
}

// ── session-messages: multi-turn exercise of Session.appendMessages ──

pub fn runSessionMessages(
    gpa: std.mem.Allocator,
    model_arg: []const u8,
    max_new: u32,
    precision: gpu_model.Precision,
) !void {
    const stdout = std.io.getStdOut().writer();
    var ctx = try vk.Context.init(gpa);
    defer ctx.deinit();

    try stdout.print("device: {s}\n", .{ctx.deviceName()});
    try stdout.print("loading model {s}...\n", .{model_arg});

    const t0 = std.time.nanoTimestamp();
    var loaded = try loader.loadGpuModel(gpa, &ctx, model_arg, .{ .precision = precision });
    defer loaded.deinit(ctx.device);
    const t1 = std.time.nanoTimestamp();
    const cfg = loaded.config();
    try stdout.print(
        "loaded in {d:.1} s — family={s} layers={d} hybrid={}\n",
        .{ @as(f64, @floatFromInt(t1 - t0)) / 1e9, @tagName(cfg.family), cfg.num_hidden_layers, cfg.family.isHybrid() },
    );

    // Hardcoded multi-turn fixture matching what an OpenAI client
    // would post to `/v1/chat/completions`. Generation continues
    // from the trailing open assistant header.
    const messages = [_]session_mod.Message{
        .{ .role = .user, .content = "Hi" },
        .{ .role = .assistant, .content = "Hello! How can I help?" },
        .{ .role = .user, .content = "Tell me a short joke." },
    };

    var on_token_ctx = SessionSmokeOnTokenCtx{ .stdout = stdout, .family = cfg.family };

    // Stop on the family's end_of_turn token so a coherent reply ends
    // cleanly instead of running into max_new_tokens.
    var stop_tokens_buf: [1]u32 = undefined;
    var sess = try session_mod.Session.init(gpa, &ctx, &loaded.gpu_model, &loaded.tokenizer, .{
        .budget = .{ .layers = 8 },
        .max_new_tokens = max_new,
        .on_token = sessionSmokeOnToken,
        .on_token_user = &on_token_ctx,
        .max_pos = 2048,
        .stop_tokens = &.{},
    });
    defer sess.deinit();
    stop_tokens_buf[0] = sess.template.end_of_turn;
    sess.cfg.stop_tokens = stop_tokens_buf[0..1];

    try sess.appendMessages(&messages);

    try stdout.print("\n--- composed prompt: {d} tokens ---\n", .{sess.prompt_q.items.len});
    try stdout.print("response: ", .{});

    var rec = try gpu_recorder.Recorder.init(&ctx, 4096, 16384);
    defer rec.deinit();

    var ticks: u32 = 0;
    while (!sess.isDone() and ticks < 4096) : (ticks += 1) {
        try rec.reset();
        try rec.begin();
        _ = try sess.tickFrame(&rec);
        try rec.endAndSubmit();
    }
    try stdout.print("\n", .{});
}

// ── runner-smoke: end-to-end InferenceRunner inline-mode exercise ─

pub fn runRunnerSmoke(
    gpa: std.mem.Allocator,
    model_arg: []const u8,
    max_new: u32,
    precision: gpu_model.Precision,
    threaded: bool,
) !void {
    const stdout = std.io.getStdOut().writer();
    var ctx = try vk.Context.init(gpa);
    defer ctx.deinit();

    try stdout.print("device: {s}\n", .{ctx.deviceName()});
    try stdout.print("loading model {s}...\n", .{model_arg});

    const t0 = std.time.nanoTimestamp();
    var loaded = try loader.loadGpuModel(gpa, &ctx, model_arg, .{ .precision = precision });
    defer loaded.deinit(ctx.device);
    const t1 = std.time.nanoTimestamp();
    const cfg = loaded.config();
    try stdout.print(
        "loaded in {d:.1} s — family={s} layers={d} hybrid={}\n",
        .{ @as(f64, @floatFromInt(t1 - t0)) / 1e9, @tagName(cfg.family), cfg.num_hidden_layers, cfg.family.isHybrid() },
    );

    // Build a Session that the runner will borrow. Same shape as
    // --session-messages but the runner owns the on_token wiring +
    // budget + stops, so we leave them at defaults here.
    var sess = try session_mod.Session.init(gpa, &ctx, &loaded.gpu_model, &loaded.tokenizer, .{
        .max_pos = 2048,
    });
    defer sess.deinit();

    var rec = try gpu_recorder.Recorder.init(&ctx, 4096, 16384);
    defer rec.deinit();

    var runner = try inference_runner.InferenceRunner.initBorrow(gpa, &sess, &rec, .{
        .default_max_tokens = max_new,
    });
    defer runner.deinit();

    if (threaded) try runner.start();

    // Same 3-turn fixture as --session-messages.
    const messages = [_]chat_template_mod.Message{
        .{ .role = .user, .content = "Hi" },
        .{ .role = .assistant, .content = "Hello! How can I help?" },
        .{ .role = .user, .content = "Tell me a short joke." },
    };

    // Stop-string sanity: any of these terminates with reason=stop.
    // Most won't trigger on the joke fixture; "guts!" matches the
    // Qwen3 4B response, "stick" matches Gemma's. Demonstrates the
    // suffix-match path without making the smoke flaky.
    const stop_strings = [_][]const u8{ "guts!", "stick.", "</response>" };

    try runner.submit(.{ .chat = .{
        .corr = 1,
        .messages = &messages,
        .max_tokens = max_new,
        .stop_strings = &stop_strings,
        .deadline_ns = 30 * std.time.ns_per_s, // 30s — plenty of headroom
    } });

    try stdout.print("response: ", .{});

    var token_count: u32 = 0;
    var saw_finish = false;
    var prefill_tokens: u32 = 0;
    var finish_reason: inference_proto.FinishReason = .stop;
    var elapsed_ns: u64 = 0;

    var ticks: u32 = 0;
    while (!saw_finish and ticks < 8192) : (ticks += 1) {
        // In inline mode the main thread drives ticks. In threaded
        // mode the worker drives ticks itself; we just sleep + poll.
        if (threaded) {
            std.time.sleep(500_000); // 500µs poll interval
        } else {
            try runner.tickInline();
        }
        while (runner.pollEvent()) |ev| {
            switch (ev.kind) {
                .accepted => |a| prefill_tokens = a.prefill_tokens,
                .token => |t| {
                    const text = runner.resolve(t.decoded);
                    try stdout.writeAll(text);
                    token_count += 1;
                },
                .arena_swap => {},
                .finish => |f| {
                    saw_finish = true;
                    finish_reason = f.reason;
                    elapsed_ns = f.elapsed_ns;
                },
                .err => |e| {
                    try stdout.print("\n[runner err: {s}]\n", .{e.msg});
                    saw_finish = true;
                },
            }
        }
    }
    try stdout.print(
        "\n[mode={s}  prefill={d}  completion={d}  reason={s}  elapsed={d:.1} ms  ticks={d}]\n",
        .{
            if (threaded) "threaded" else "inline",
            prefill_tokens, token_count, @tagName(finish_reason),
            @as(f64, @floatFromInt(elapsed_ns)) / 1e6, ticks,
        },
    );
    // Threaded mode: shutdown joins the worker before deinit's
    // shutdown call (which would also work; just being explicit).
    if (threaded) runner.shutdown();
}
