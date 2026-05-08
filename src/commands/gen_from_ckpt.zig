//! `valkyr --gen-from-ckpt <model> --ckpt <path.vkpt> --prompt TEXT`
//! Generates from a fine-tuned `.vkpt` checkpoint using the *training*
//! Runner's `forwardLogits` + greedy autoregressive decode.
//!
//! Slow path — each token needs a full Runner.forwardLogits pass over
//! n_pos positions (~150 ms/tok at Qwen3-0.6B Debug). Complete the
//! train→use loop end to end without needing the inference Session
//! to know about `.vkpt`. The fast path (loading `.vkpt` into the
//! inference Session for production-speed generation) is a separate
//! follow-up that needs fp32→bf16 conversion and tensor-name mapping.

const std = @import("std");
const vk = @import("../gpu/vk.zig");
const model_mod = @import("../model.zig");
const tokenizer_mod = @import("../tokenizer.zig");
const hf_cache = @import("../hf_cache.zig");
const train_transformer = @import("../train/transformer.zig");
const train_load_real = @import("../train/load_real.zig");
const train_sampling = @import("../train/sampling.zig");

pub const Options = struct {
    model: []const u8, // HF model id or local dir — provides architecture + tokenizer
    ckpt_path: []const u8, // .vkpt or .lvkpt produced by --fine-tune --out
    prompt: []const u8,
    n_gen: u32 = 30,
    n_pos: u32 = 16, // must match the n_pos the checkpoint was saved at
    eos_id: u32 = 151_645, // Qwen3 <|im_end|>

    // ── For `.lvkpt`, the loader needs the same `lora_targets` /
    //    `lora_rank` the checkpoint was saved with so the Runner
    //    allocates matching adapter slots before loadLoraCheckpoint
    //    overwrites them. cfgShapeMatches enforces the match at load
    //    time. Leave at defaults (0 / 0) for `.vkpt` checkpoints.
    lora_targets: u32 = 0,
    lora_rank: u32 = 0,
    lora_alpha: f32 = 0.0,
    lora_lr_b_scale: f32 = 1.0,
};

pub fn run(allocator: std.mem.Allocator, opts: Options) !void {
    const stdout = std.io.getStdOut().writer();

    // ── Resolve model directory.
    const dir_path = try hf_cache.resolveModelArg(allocator, opts.model);
    defer allocator.free(dir_path);
    try stdout.print("[gen-from-ckpt] model: {s}\n", .{opts.model});
    try stdout.print("[gen-from-ckpt] ckpt:  {s}\n", .{opts.ckpt_path});

    var cpu = try model_mod.Model.load(allocator, dir_path);
    defer cpu.deinit();

    // ── Materialize fp32 train tensors. We use these as the Runner's
    // initial state; loadCheckpoint then overwrites them with the
    // saved checkpoint contents. The base-weight load is necessary
    // because Runner.init still needs a complete InitWeights to size
    // and populate every device buffer; we can't yet skip straight
    // to a checkpoint-only init path.
    var weights = try train_load_real.loadTrainWeights(allocator, &cpu, opts.n_pos);
    defer weights.deinit();
    var cfg = weights.cfg;
    cfg.lora_targets = opts.lora_targets;
    cfg.lora_rank = opts.lora_rank;
    cfg.lora_alpha = opts.lora_alpha;
    cfg.lora_lr_b_scale = opts.lora_lr_b_scale;

    // ── Tokenizer.
    const tok_path = try std.fmt.allocPrint(allocator, "{s}/tokenizer.json", .{dir_path});
    defer allocator.free(tok_path);
    var tok = try tokenizer_mod.Tokenizer.loadFromFile(allocator, tok_path);
    defer tok.deinit();

    const prompt_ids = try tok.encode(allocator, opts.prompt);
    defer allocator.free(prompt_ids);
    if (prompt_ids.len == 0 or prompt_ids.len >= opts.n_pos) return error.PromptShape;

    // ── GPU bring-up.
    var ctx = try vk.Context.init(allocator);
    defer ctx.deinit();

    var runner = try train_transformer.Runner.init(allocator, &ctx, cfg, weights.view());
    defer runner.deinit();

    // ── Load the checkpoint (overwrites params + Adam state). We
    //    autodetect `.vkpt` vs `.lvkpt` by sniffing the 4-byte magic
    //    at the file head — same routine handles both since the user
    //    might point either kind of file via `--ckpt`.
    const t_load_start = std.time.nanoTimestamp();
    var magic_buf: [4]u8 = undefined;
    {
        const f = try std.fs.cwd().openFile(opts.ckpt_path, .{ .mode = .read_only });
        defer f.close();
        try f.reader().readNoEof(&magic_buf);
    }
    const is_lora = std.mem.eql(u8, &magic_buf, "VLKP");
    if (is_lora) {
        runner.loadLoraCheckpoint(allocator, opts.ckpt_path) catch |err| {
            try stdout.print("[gen-from-ckpt] LoRA checkpoint load failed: {s}\n", .{@errorName(err)});
            return err;
        };
    } else {
        runner.loadCheckpoint(allocator, opts.ckpt_path) catch |err| {
            try stdout.print("[gen-from-ckpt] checkpoint load failed: {s}\n", .{@errorName(err)});
            return err;
        };
    }
    const t_load_end = std.time.nanoTimestamp();
    const load_ms: f64 = @as(f64, @floatFromInt(t_load_end - t_load_start)) / 1.0e6;
    try stdout.print("[gen-from-ckpt] loaded {s} checkpoint in {d:.0} ms\n", .{ if (is_lora) "LoRA" else "full-weight", load_ms });

    // ── Generate.
    const vocab: usize = cfg.vocab_size;
    const t_gen_start = std.time.nanoTimestamp();
    const sample = try train_sampling.greedyDecode(allocator, &runner, prompt_ids, opts.n_gen, opts.n_pos, vocab, opts.eos_id);
    defer allocator.free(sample);
    const t_gen_end = std.time.nanoTimestamp();
    const gen_ms: f64 = @as(f64, @floatFromInt(t_gen_end - t_gen_start)) / 1.0e6;
    const ms_per_tok: f64 = gen_ms / @as(f64, @floatFromInt(opts.n_gen));

    const text = try train_sampling.decodeIdsToText(allocator, &tok, sample);
    defer allocator.free(text);

    try stdout.print("[gen-from-ckpt] {d} tokens in {d:.0} ms ({d:.1} ms/tok)\n", .{ opts.n_gen, gen_ms, ms_per_tok });
    try stdout.print("\n{s}\n", .{text});
}
