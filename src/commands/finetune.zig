//! `valkyr --fine-tune <model> --data <jsonl> [...]` — the user-facing
//! fine-tune driver. Wraps the same primitives the β-3..β-6 smokes use:
//! load real Qwen3-class weights as fp32 train tensors, build a packed-
//! stream dataset from a JSONL file, train N Adam steps, optionally
//! save a `.vkpt` checkpoint, optionally probe-sample text before and
//! after to show the model has shifted.
//!
//! Single-batch overfit is the default — reproduces the marquee
//! demonstration where a probe prompt that intersects the training data
//! produces verbatim training-text continuation after fine-tune. The
//! caller can choose `--rotate` to cycle through batches instead, which
//! is the realistic multi-example fine-tune path (less dramatic at 30
//! steps but the scaling-honest mode for larger datasets).

const std = @import("std");
const vk = @import("../gpu/vk.zig");
const model_mod = @import("../model.zig");
const tokenizer_mod = @import("../tokenizer.zig");
const hf_cache = @import("../hf_cache.zig");
const train_transformer = @import("../train/transformer.zig");
const train_dataset = @import("../train/dataset.zig");
const train_load_real = @import("../train/load_real.zig");
const train_sampling = @import("../train/sampling.zig");

pub const Options = struct {
    model: []const u8, // HF model id or local dir path
    data_path: []const u8, // JSONL path (one {"text": "..."} per line)
    n_steps: u32 = 30,
    n_pos: u32 = 16,
    lr: f32 = 1e-5,
    batch_idx: u32 = 0, // single-batch overfit by default
    rotate: bool = false, // cycle through dataset batches each step
    out_path: ?[]const u8 = null,
    probe: ?[]const u8 = null,
    n_gen: u32 = 20,
    print_every: u32 = 5,
    eos_id: u32 = 151_645, // Qwen3 <|im_end|>

    // ── LoRA fine-tune options (A4-4). When `lora_targets != 0` the
    //    Runner enables LoRA on the named projections and saves a
    //    `.lvkpt` (LoRA-only) checkpoint instead of the full `.vkpt`.
    //    Base weights stay in the source safetensors.
    lora_targets: u32 = 0,
    lora_rank: u32 = 16,
    lora_alpha: f32 = 32.0,
    lora_lr_b_scale: f32 = 1.0,
};

/// Parse a comma-separated `--lora-targets` value into the bitmask
/// `Config.lora_targets` consumes. Recognises individual projection
/// names ("q","k","v","o","gate","up","down") plus the three group
/// shorthands ("all_attn","all_ffn","all"). Whitespace around commas
/// is tolerated. Returns error.UnknownLoraTarget on an unrecognised
/// name.
pub fn parseLoraTargets(spec: []const u8) !u32 {
    var mask: u32 = 0;
    var it = std.mem.tokenizeAny(u8, spec, ", \t");
    while (it.next()) |tok| {
        if (std.mem.eql(u8, tok, "q")) {
            mask |= train_transformer.LoraTarget.q;
        } else if (std.mem.eql(u8, tok, "k")) {
            mask |= train_transformer.LoraTarget.k;
        } else if (std.mem.eql(u8, tok, "v")) {
            mask |= train_transformer.LoraTarget.v;
        } else if (std.mem.eql(u8, tok, "o")) {
            mask |= train_transformer.LoraTarget.o;
        } else if (std.mem.eql(u8, tok, "gate")) {
            mask |= train_transformer.LoraTarget.gate;
        } else if (std.mem.eql(u8, tok, "up")) {
            mask |= train_transformer.LoraTarget.up;
        } else if (std.mem.eql(u8, tok, "down")) {
            mask |= train_transformer.LoraTarget.down;
        } else if (std.mem.eql(u8, tok, "all_attn") or std.mem.eql(u8, tok, "all-attn")) {
            mask |= train_transformer.LoraTarget.all_attn;
        } else if (std.mem.eql(u8, tok, "all_ffn") or std.mem.eql(u8, tok, "all-ffn")) {
            mask |= train_transformer.LoraTarget.all_ffn;
        } else if (std.mem.eql(u8, tok, "all")) {
            mask |= train_transformer.LoraTarget.all;
        } else {
            return error.UnknownLoraTarget;
        }
    }
    return mask;
}

pub fn run(allocator: std.mem.Allocator, opts: Options) !void {
    const stdout = std.io.getStdOut().writer();

    // ── Resolve model directory.
    const dir_path = try hf_cache.resolveModelArg(allocator, opts.model);
    defer allocator.free(dir_path);
    try stdout.print("[fine-tune] model: {s}\n", .{opts.model});

    var cpu = try model_mod.Model.load(allocator, dir_path);
    defer cpu.deinit();

    // ── Materialize fp32 train tensors.
    var weights = try train_load_real.loadTrainWeights(allocator, &cpu, opts.n_pos);
    defer weights.deinit();
    var cfg = weights.cfg;
    cfg.lr = opts.lr;
    cfg.lora_targets = opts.lora_targets;
    cfg.lora_rank = opts.lora_rank;
    cfg.lora_alpha = opts.lora_alpha;
    cfg.lora_lr_b_scale = opts.lora_lr_b_scale;

    try stdout.print(
        "[fine-tune] arch: {d} layers, dim={d}, GQA {d}/{d}, head_dim={d}, ff_dim={d}, vocab={d}\n",
        .{ cfg.n_layers, cfg.dim, cfg.n_heads, cfg.n_kv_heads, cfg.head_dim, cfg.ff_dim, cfg.vocab_size },
    );
    if (cfg.lora_targets != 0) {
        try stdout.print(
            "[fine-tune] mode: LoRA — targets bitmask=0x{x} ({d}/7 projections) rank={d} α={d:.2} α/r={d:.2} lr_b_scale={d:.2}\n",
            .{ cfg.lora_targets, @popCount(cfg.lora_targets), cfg.lora_rank, cfg.lora_alpha, cfg.lora_alpha / @as(f32, @floatFromInt(cfg.lora_rank)), cfg.lora_lr_b_scale },
        );
    } else {
        try stdout.print("[fine-tune] mode: full-weight (every param trained, .vkpt save format)\n", .{});
    }

    // ── Tokenizer.
    const tok_path = try std.fmt.allocPrint(allocator, "{s}/tokenizer.json", .{dir_path});
    defer allocator.free(tok_path);
    var tok = try tokenizer_mod.Tokenizer.loadFromFile(allocator, tok_path);
    defer tok.deinit();

    // ── Dataset.
    var ds = try train_dataset.buildFromJsonl(allocator, &tok, opts.data_path, opts.n_pos, opts.eos_id);
    defer ds.deinit();
    if (ds.numBatches() == 0) return error.DatasetTooShort;
    try stdout.print("[fine-tune] dataset: {d} batches at n_pos={d}\n", .{ ds.numBatches(), opts.n_pos });

    if (opts.batch_idx >= ds.numBatches()) return error.BatchIdxOutOfRange;

    // ── GPU bring-up.
    var ctx = try vk.Context.init(allocator);
    defer ctx.deinit();
    var runner = try train_transformer.Runner.init(allocator, &ctx, cfg, weights.view());
    defer runner.deinit();

    const vocab: usize = cfg.vocab_size;
    const logits = try allocator.alloc(f32, @as(usize, opts.n_pos) * vocab);
    defer allocator.free(logits);

    // ── Pre-train probe sample (optional).
    var probe_pre_text: ?[]u8 = null;
    var probe_ids: ?[]u32 = null;
    defer if (probe_pre_text) |t| allocator.free(t);
    defer if (probe_ids) |ids| allocator.free(ids);

    if (opts.probe) |probe_text| {
        const ids = try tok.encode(allocator, probe_text);
        if (ids.len == 0 or ids.len >= opts.n_pos) {
            allocator.free(ids);
            try stdout.print("[fine-tune] WARN probe \"{s}\" tokenizes to {d} tokens; need 1..{d}; skipping probe\n", .{ probe_text, ids.len, opts.n_pos - 1 });
        } else {
            probe_ids = ids;
            try stdout.print("[fine-tune] probe: \"{s}\" ({d} tokens)\n", .{ probe_text, ids.len });
            const pre_sample = try train_sampling.greedyDecode(allocator, &runner, ids, opts.n_gen, opts.n_pos, vocab, opts.eos_id);
            defer allocator.free(pre_sample);
            probe_pre_text = try train_sampling.decodeIdsToText(allocator, &tok, pre_sample);
            try stdout.print("[probe before] {s}\n", .{probe_pre_text.?});
        }
    }

    // ── Initial CE.
    const input_ids = try allocator.alloc(u32, opts.n_pos);
    defer allocator.free(input_ids);
    const target_ids = try allocator.alloc(u32, opts.n_pos);
    defer allocator.free(target_ids);
    try ds.batch(opts.batch_idx, input_ids, target_ids);

    try runner.forwardLogits(input_ids, logits);
    const ce_init = computeMeanCe(logits, target_ids, opts.n_pos, vocab);
    try stdout.print("[fine-tune] CE initial: {d:.4} (batch {d})\n", .{ ce_init, opts.batch_idx });

    // ── Training loop.
    const t_train_start = std.time.nanoTimestamp();
    var step: u32 = 0;
    while (step < opts.n_steps) : (step += 1) {
        const batch_idx: u32 = if (opts.rotate)
            @intCast(@mod(@as(usize, step) + opts.batch_idx, ds.numBatches()))
        else
            opts.batch_idx;
        try ds.batch(batch_idx, input_ids, target_ids);

        const t_step_start = std.time.nanoTimestamp();
        try runner.step(input_ids, target_ids);
        const t_step_end = std.time.nanoTimestamp();
        const step_ms: f64 = @as(f64, @floatFromInt(t_step_end - t_step_start)) / 1.0e6;

        const display_step = step + 1;
        if (opts.print_every > 0 and (display_step % opts.print_every == 0 or display_step == opts.n_steps)) {
            try stdout.print("[step {d:>4}/{d}] {d:.1} ms/step\n", .{ display_step, opts.n_steps, step_ms });
        }
    }
    const t_train_end = std.time.nanoTimestamp();
    const train_ms: f64 = @as(f64, @floatFromInt(t_train_end - t_train_start)) / 1.0e6;
    const mean_step_ms: f64 = train_ms / @as(f64, @floatFromInt(opts.n_steps));

    // ── Final CE on the same batch we measured CE_init on.
    try ds.batch(opts.batch_idx, input_ids, target_ids);
    try runner.forwardLogits(input_ids, logits);
    const ce_final = computeMeanCe(logits, target_ids, opts.n_pos, vocab);
    const drop_pct: f64 = if (ce_init > 0) 100.0 * (@as(f64, ce_init) - @as(f64, ce_final)) / @as(f64, ce_init) else 0.0;
    const delta = ce_final - ce_init;
    const sign: u8 = if (delta < 0) '-' else '+';
    try stdout.print(
        "[fine-tune] CE final: {d:.4} (Δ {c}{d:.4}, {d:.2}% drop on batch {d}; total {d:.1} ms = {d:.1} ms/step)\n",
        .{ ce_final, sign, @abs(delta), drop_pct, opts.batch_idx, train_ms, mean_step_ms },
    );

    // ── Post-train probe sample.
    if (probe_ids) |ids| {
        const post_sample = try train_sampling.greedyDecode(allocator, &runner, ids, opts.n_gen, opts.n_pos, vocab, opts.eos_id);
        defer allocator.free(post_sample);
        const post_text = try train_sampling.decodeIdsToText(allocator, &tok, post_sample);
        defer allocator.free(post_text);
        try stdout.print("[probe after]  {s}\n", .{post_text});
    }

    // ── Optional checkpoint save. LoRA mode → `.lvkpt` (small,
    //    LoRA-only); full-weight mode → `.vkpt`.
    if (opts.out_path) |out| {
        const t_save_start = std.time.nanoTimestamp();
        if (cfg.lora_targets != 0) {
            try runner.saveLoraCheckpoint(allocator, out);
        } else {
            try runner.saveCheckpoint(allocator, out);
        }
        const t_save_end = std.time.nanoTimestamp();
        const save_ms: f64 = @as(f64, @floatFromInt(t_save_end - t_save_start)) / 1.0e6;
        const file = try std.fs.cwd().openFile(out, .{ .mode = .read_only });
        defer file.close();
        const stat = try file.stat();
        const size_bytes: f64 = @floatFromInt(stat.size);
        // `.lvkpt` is typically MiB-scale at most; report MiB to keep
        // the output readable.
        const fmt_kind: []const u8 = if (cfg.lora_targets != 0) "lora-checkpoint" else "checkpoint";
        if (size_bytes >= 256.0 * 1024.0 * 1024.0) {
            try stdout.print(
                "[fine-tune] saved {s}: {s} ({d:.2} GiB, {d:.1} ms)\n",
                .{ fmt_kind, out, size_bytes / (1024.0 * 1024.0 * 1024.0), save_ms },
            );
        } else {
            try stdout.print(
                "[fine-tune] saved {s}: {s} ({d:.2} MiB, {d:.1} ms)\n",
                .{ fmt_kind, out, size_bytes / (1024.0 * 1024.0), save_ms },
            );
        }
    }
}

/// Mean per-position cross-entropy. Mirror of the helper in
/// `smoke/decoder.zig`'s β-5/β-6a smokes — fp64 accumulation for the
/// per-position log-Z.
fn computeMeanCe(logits: []const f32, target_ids: []const u32, n_pos: u32, vocab: usize) f32 {
    var ce_sum: f64 = 0;
    for (0..n_pos) |p| {
        const off = p * vocab;
        var m: f32 = -std.math.inf(f32);
        for (0..vocab) |o| m = @max(m, logits[off + o]);
        var sum_e: f64 = 0;
        for (0..vocab) |o| sum_e += @exp(@as(f64, logits[off + o]) - @as(f64, m));
        const log_z: f64 = @as(f64, m) + @log(sum_e);
        ce_sum += log_z - @as(f64, logits[off + target_ids[p]]);
    }
    return @floatCast(ce_sum / @as(f64, @floatFromInt(n_pos)));
}
