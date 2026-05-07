//! `--serve`: OpenAI-compatible HTTP endpoint. Loads <model> once,
//! shares one `Session`/`InferenceRunner` across requests, and listens
//! for POST /v1/chat/completions + GET /v1/models. Extracted from
//! main.zig.

const std = @import("std");
const vk = @import("../gpu/vk.zig");
const gpu_model = @import("../gpu/model.zig");
const gpu_recorder = @import("../gpu/recorder.zig");
const loader = @import("../loader.zig");
const session_mod = @import("../session.zig");
const inference_runner = @import("../inference/runner.zig");
const server_mod = @import("../server/server.zig");

pub fn runServe(
    gpa: std.mem.Allocator,
    model_arg: []const u8,
    public_id: []const u8,
    bind_addr: []const u8,
    port: u16,
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

    var sess = try session_mod.Session.init(gpa, &ctx, &loaded.gpu_model, &loaded.tokenizer, .{
        .max_pos = 4096,
    });
    defer sess.deinit();

    // Recorder sized for serve's likely usage: pure-µs budget mode
    // (default for runner) + worst-case big-model layer counts. The
    // runner default is `.either{layers=8, µs=5000}` so the layer
    // cap keeps us inside 4096 sets / 16384 descriptors.
    var rec = try gpu_recorder.Recorder.init(&ctx, 4096, 16384);
    defer rec.deinit();

    var runner = try inference_runner.InferenceRunner.initBorrow(gpa, &sess, &rec, .{
        .default_max_tokens = max_new,
    });
    defer runner.deinit();
    try runner.start();

    var srv = try server_mod.Server.init(gpa, &runner, .{
        .bind_address = bind_addr,
        .port = port,
        .model_id = public_id,
        .default_max_tokens = max_new,
    });
    defer srv.deinit();
    try srv.start();

    try stdout.print(
        "server ready: POST http://{s}:{d}/v1/chat/completions  •  GET /v1/models\n" ++
            "model id (validate against request.model): {s}\n" ++
            "Ctrl-C to stop.\n",
        .{ bind_addr, port, public_id },
    );

    // Park forever — Ctrl-C kills the process. We don't install a
    // signal handler; the OS reaping is fine for v0.
    while (true) {
        std.time.sleep(std.time.ns_per_s);
    }
}
