//! GPU generation smokes invoked via `--gpu-gen`, `--gpu-gen-many`,
//! `--gpu-gen-qwen35`, `--gpu-gen-tq4v`. Each one streams a fixed number
//! of tokens through the GPU forward path and prints the result.
//! Extracted from main.zig.

const std = @import("std");
const vk = @import("../gpu/vk.zig");
const pipeline = @import("../gpu/pipeline.zig");
const buffer = @import("../gpu/buffer.zig");
const model_mod = @import("../model.zig");
const tokenizer_mod = @import("../tokenizer.zig");
const config_mod = @import("../config.zig");
const cpu_forward = @import("../cpu/forward.zig");
const cpu_full_attn = @import("../cpu/full_attn.zig");
const cpu_gated_delta = @import("../cpu/gated_delta.zig");
const dtype = @import("../dtype.zig");
const gpu_model = @import("../gpu/model.zig");
const gpu_scratch = @import("../gpu/scratch.zig");
const gpu_recorder = @import("../gpu/recorder.zig");
const shaders = @import("shaders");

const aliases = @import("../runtime_aliases.zig");
const helpers = @import("../smoke/helpers.zig");

// ── gpu-gen-many: multi-token generation with KV cache ─────────────

pub fn runGpuGenMany(gpa: std.mem.Allocator, dir_path: []const u8, first_token: u32, n_tokens: usize) !void {
    var cpu = try model_mod.Model.load(gpa, dir_path);
    defer cpu.deinit();
    const cfg = cpu.config;

    const tok_path = try std.fmt.allocPrint(gpa, "{s}/tokenizer.json", .{dir_path});
    defer gpa.free(tok_path);
    var tok = try tokenizer_mod.Tokenizer.loadFromFile(gpa, tok_path);
    defer tok.deinit();

    var ctx = try vk.Context.init(gpa);
    defer ctx.deinit();

    const stdout = std.io.getStdOut().writer();
    try stdout.print("device: {s}\n", .{ctx.deviceName()});
    try stdout.print("uploading weights (bf16 matmul path)...\n", .{});
    var gm = try gpu_model.GpuModel.upload(gpa, &ctx, &cpu, .bf16_matmul);
    defer gm.deinit(ctx.device);

    const max_pos: usize = @max(n_tokens + 8, 64);
    var sc = try gpu_scratch.GpuScratch.init(&ctx, cfg, max_pos);
    defer sc.deinit(ctx.device);
    var kv = try gpu_scratch.GpuKvCache.init(gpa, &ctx, cfg, max_pos);
    defer kv.deinit(ctx.device);

    var k = try aliases.ChatKernels.init(&ctx, gm.precision, cfg.family, @intCast(cfg.head_dim));
    defer k.deinit();

    var rec = try gpu_recorder.Recorder.init(&ctx, 512, 2048);
    defer rec.deinit();

    const logits = try gpa.alloc(f32, cfg.vocab_size);
    defer gpa.free(logits);

    const first_str = tok.decode(first_token) orelse "<unknown>";
    try stdout.print("\nstreaming generation, max {d} tokens, max_pos = {d}\n", .{ n_tokens, max_pos });
    try stdout.print("input: id={d} {s}\n", .{ first_token, first_str });
    try stdout.print("output: ", .{});

    var current_token: u32 = first_token;
    var pos: usize = 0;
    var total_gpu_ns: i128 = 0;

    while (pos < n_tokens) : (pos += 1) {
        if (pos > 0) try rec.reset();
        try rec.begin();
        try aliases.recordForwardStep(&rec, &sc, &gm, &kv, cfg, &k, pos, current_token, null, true);

        const t0 = std.time.nanoTimestamp();
        try rec.endAndSubmit();
        const t1 = std.time.nanoTimestamp();
        total_gpu_ns += (t1 - t0);

        try sc.logits.readBack(&ctx, f32, logits);
        const next = cpu_forward.argmax(logits);
        const next_str = tok.decode(next) orelse "<unknown>";
        try stdout.print("{s}", .{next_str});

        if (cfg.eos_token_id) |eos| {
            if (next == eos) break;
        }
        current_token = @intCast(next);
    }

    try stdout.print("\n\n", .{});
    const total_gpu_ms = @as(f64, @floatFromInt(total_gpu_ns)) / 1_000_000.0;
    const tokens_done = pos;
    const ms_per_tok = total_gpu_ms / @as(f64, @floatFromInt(@max(tokens_done, 1)));
    try stdout.print("generated {d} tokens in {d:.0} ms total ({d:.1} ms/token, {d:.1} tok/s)\n", .{
        tokens_done, total_gpu_ms, ms_per_tok, 1000.0 / ms_per_tok,
    });
}

// ── gpu-gen-tq4v: GPU forward with TQ4 V-cache vs fp32 V baseline ──
//
// Same input token, two passes through aliases.recordForwardStep — once with
// tq4_v = null (existing fp32 V path), once with the TQ4 hooks set
// up. Reads back logits for both, prints argmax + top-5 (with decoded
// text) + max / mean / rms divergence over the full vocab. The
// signal we want: argmax matches and top-5 IDs are preserved.

pub fn runGpuGenTq4V(gpa: std.mem.Allocator, dir_path: []const u8, token_id: u32) !void {
    var cpu = try model_mod.Model.load(gpa, dir_path);
    defer cpu.deinit();
    const cfg = cpu.config;

    const tok_path = try std.fmt.allocPrint(gpa, "{s}/tokenizer.json", .{dir_path});
    defer gpa.free(tok_path);
    var tok = try tokenizer_mod.Tokenizer.loadFromFile(gpa, tok_path);
    defer tok.deinit();

    var ctx = try vk.Context.init(gpa);
    defer ctx.deinit();

    const stdout = std.io.getStdOut().writer();
    try stdout.print("device: {s}\n", .{ctx.deviceName()});
    try stdout.print("input: id={d}  string={s}\n\n", .{ token_id, tok.decode(token_id) orelse "?" });

    var gm = try gpu_model.GpuModel.upload(gpa, &ctx, &cpu, .bf16_matmul);
    defer gm.deinit(ctx.device);

    const max_pos: usize = 8;
    var sc = try gpu_scratch.GpuScratch.init(&ctx, cfg, max_pos);
    defer sc.deinit(ctx.device);
    var kv = try gpu_scratch.GpuKvCache.init(gpa, &ctx, cfg, max_pos);
    defer kv.deinit(ctx.device);
    var kv_tq4 = try gpu_scratch.GpuKvCacheTq4.init(gpa, &ctx, cfg, max_pos);
    defer kv_tq4.deinit(ctx.device);

    var k = try aliases.ChatKernels.init(&ctx, gm.precision, cfg.family, @intCast(cfg.head_dim));
    defer k.deinit();

    var tq_pack = try pipeline.Kernel.init(&ctx, &shaders.tq4_pack_to_cache, 2, @sizeOf(aliases.Tq4PackPush));
    defer tq_pack.deinit();
    var tq_unpack = try pipeline.Kernel.init(&ctx, &shaders.tq4_unpack256, 2, 0);
    defer tq_unpack.deinit();
    const tq4_hooks = aliases.Tq4VHooks{ .pack = &tq_pack, .unpack = &tq_unpack, .cache = &kv_tq4 };

    var rec = try gpu_recorder.Recorder.init(&ctx, 512, 2048);
    defer rec.deinit();

    const logits_a = try gpa.alloc(f32, cfg.vocab_size);
    defer gpa.free(logits_a);
    const logits_b = try gpa.alloc(f32, cfg.vocab_size);
    defer gpa.free(logits_b);

    // Pass 1: fp32 V baseline.
    try rec.begin();
    try aliases.recordForwardStep(&rec, &sc, &gm, &kv, cfg, &k, 0, token_id, null, true);
    try rec.endAndSubmit();
    try sc.logits.readBack(&ctx, f32, logits_a);

    // Pass 2: TQ4 V.
    try rec.reset();
    try rec.begin();
    try aliases.recordForwardStep(&rec, &sc, &gm, &kv, cfg, &k, 0, token_id, tq4_hooks, true);
    try rec.endAndSubmit();
    try sc.logits.readBack(&ctx, f32, logits_b);

    const top_a = try helpers.topK(gpa, logits_a, 5);
    defer gpa.free(top_a);
    const top_b = try helpers.topK(gpa, logits_b, 5);
    defer gpa.free(top_b);

    try stdout.print("            fp32 V baseline                     TQ4 V\n", .{});
    try stdout.print("rank   id       logit    token              id       logit    token\n", .{});
    try stdout.print("----   ------   ------   ----------------   ------   ------   ----------------\n", .{});
    for (0..5) |i| {
        const ta = top_a[i];
        const tb = top_b[i];
        const ta_text = tok.decode(ta.id) orelse "?";
        const tb_text = tok.decode(tb.id) orelse "?";
        try stdout.print("{d:>4}   {d:>6}   {d:>6.2}   {s:<16}   {d:>6}   {d:>6.2}   {s:<16}\n", .{
            i, ta.id, ta.value, helpers.truncateStr(ta_text, 16), tb.id, tb.value, helpers.truncateStr(tb_text, 16),
        });
    }

    var max_abs_delta: f32 = 0;
    var max_abs_delta_idx: usize = 0;
    var sum_abs: f64 = 0;
    var sum_sq: f64 = 0;
    for (logits_a, logits_b, 0..) |a, b, i| {
        const d = @abs(a - b);
        if (d > max_abs_delta) {
            max_abs_delta = d;
            max_abs_delta_idx = i;
        }
        sum_abs += d;
        sum_sq += d * d;
    }
    const n: f64 = @floatFromInt(logits_a.len);
    try stdout.print("\nlogit divergence over {d} tokens:\n", .{logits_a.len});
    try stdout.print("  max |Δ|   = {d:.6}  at id={d}\n", .{ max_abs_delta, max_abs_delta_idx });
    try stdout.print("  mean |Δ|  = {d:.6}\n", .{sum_abs / n});
    try stdout.print("  rms  Δ    = {d:.6}\n", .{std.math.sqrt(sum_sq / n)});

    const argmax_a = cpu_forward.argmax(logits_a);
    const argmax_b = cpu_forward.argmax(logits_b);
    try stdout.print("\nargmax:  fp32 V → {d}    TQ4 V → {d}    {s}\n", .{
        argmax_a, argmax_b, if (argmax_a == argmax_b) "(MATCH)" else "(DIVERGE!)",
    });
    var top5_match: usize = 0;
    for (top_a) |a| for (top_b) |b| if (a.id == b.id) { top5_match += 1; break; };
    try stdout.print("top-5 ID overlap: {d}/5\n", .{top5_match});
}

// ── gpu-gen: full forward on GPU + parity vs CPU --gen ─────────────

pub fn runGpuGen(gpa: std.mem.Allocator, dir_path: []const u8, token_id: u32) !void {
    var cpu = try model_mod.Model.load(gpa, dir_path);
    defer cpu.deinit();
    const cfg = cpu.config;

    const tok_path = try std.fmt.allocPrint(gpa, "{s}/tokenizer.json", .{dir_path});
    defer gpa.free(tok_path);
    var tok = try tokenizer_mod.Tokenizer.loadFromFile(gpa, tok_path);
    defer tok.deinit();

    var ctx = try vk.Context.init(gpa);
    defer ctx.deinit();

    const stdout = std.io.getStdOut().writer();
    const input_str = tok.decode(token_id) orelse "<unknown>";
    try stdout.print("input  token id={d}  string={s}\n", .{ token_id, input_str });
    try stdout.print("device: {s}\n\n", .{ctx.deviceName()});

    const t_up0 = std.time.nanoTimestamp();
    var gm = try gpu_model.GpuModel.upload(gpa, &ctx, &cpu, .fp32_all);
    defer gm.deinit(ctx.device);
    const t_up1 = std.time.nanoTimestamp();
    const upload_ms = @as(f64, @floatFromInt(t_up1 - t_up0)) / 1_000_000.0;
    try stdout.print("upload: {d:.0} ms\n", .{upload_ms});

    var sc = try gpu_scratch.GpuScratch.init(&ctx, cfg, 1);
    defer sc.deinit(ctx.device);

    // ── Build kernels once, rebind per-layer ────────────────────────
    var k_embed = try pipeline.Kernel.init(&ctx, &shaders.embed_lookup, 2, @sizeOf(aliases.EmbedLookupPush));
    defer k_embed.deinit();
    var k_rmsnorm = try pipeline.Kernel.init(&ctx, &shaders.rmsnorm, 3, @sizeOf(aliases.RmsnormPush));
    defer k_rmsnorm.deinit();
    var k_matmul = try pipeline.Kernel.init(&ctx, &shaders.matmul_nt_v2, 3, @sizeOf(aliases.MatmulPush));
    defer k_matmul.deinit();
    var k_rope = try pipeline.Kernel.init(&ctx, &shaders.rope, 2, @sizeOf(aliases.RopePush));
    defer k_rope.deinit();
    var k_attn = try pipeline.Kernel.init(&ctx, &shaders.attn_decode_single, 2, @sizeOf(helpers.AttnDecodeSinglePush));
    defer k_attn.deinit();
    var k_add = try pipeline.Kernel.init(&ctx, &shaders.add_in_place, 2, @sizeOf(aliases.AddInPlacePush));
    defer k_add.deinit();
    var k_geglu = try pipeline.Kernel.init(&ctx, &shaders.geglu, 3, @sizeOf(aliases.GegluPush));
    defer k_geglu.deinit();

    const hidden: u32 = @intCast(cfg.hidden_size);
    const inter: u32 = @intCast(cfg.intermediate_size);
    const q_dim: u32 = @intCast(cfg.num_attention_heads * cfg.head_dim);
    const kv_dim: u32 = @intCast(cfg.num_key_value_heads * cfg.head_dim);
    const vocab: u32 = @intCast(cfg.vocab_size);
    const gemma_quirk: u32 = if (cfg.family == .gemma) 1 else 0;

    // ── Recorder: one command buffer for the whole forward pass ────
    // Sizing: 291 dispatches at ~3 storage-buffer descriptors each =
    // ~728 descriptors. Round up generously so we don't trip on any
    // future kernel that adds bindings.
    var rec = try gpu_recorder.Recorder.init(&ctx, 512, 2048);
    defer rec.deinit();

    const rms_push = aliases.RmsnormPush{ .dim = hidden, .eps = cfg.rms_norm_eps, .gemma_quirk = gemma_quirk };
    const add_push = aliases.AddInPlacePush{ .n = hidden };
    const attn_push = helpers.AttnDecodeSinglePush{
        .n_heads = @intCast(cfg.num_attention_heads),
        .heads_per_kv = @intCast(cfg.num_attention_heads / cfg.num_key_value_heads),
        .head_dim = @intCast(cfg.head_dim),
    };
    const rope_q_push = aliases.RopePush{
        .n_heads = @intCast(cfg.num_attention_heads),
        .head_dim = @intCast(cfg.head_dim),
        .pos = 0,
        .theta_base = cfg.rope_theta,
    };
    const rope_k_push = aliases.RopePush{
        .n_heads = @intCast(cfg.num_key_value_heads),
        .head_dim = @intCast(cfg.head_dim),
        .pos = 0,
        .theta_base = cfg.rope_theta,
    };
    const geglu_push = aliases.GegluPush{ .n = inter };
    const embed_push = aliases.EmbedLookupPush{
        .token_id = token_id,
        .dim = hidden,
        .scale = if (cfg.family.embedScalesByDim()) @sqrt(@as(f32, @floatFromInt(hidden))) else 1.0,
    };

    const t_gpu0 = std.time.nanoTimestamp();

    try rec.begin();

    // Embed lookup → residual stream.
    try aliases.recDispatch1D(&rec, &k_embed, &.{ &gm.embed_tokens, &sc.stream }, &embed_push, hidden);

    // 18 transformer blocks.
    for (gm.layers) |*layer| {
        try aliases.recDispatchPerRow(&rec, &k_rmsnorm, &.{ &sc.stream, &layer.input_layernorm, &sc.x_norm }, &rms_push, 1);

        try aliases.recDispatchMatmul(&rec, &k_matmul, &.{ &sc.x_norm, &layer.q_proj.?, &sc.q }, 1, q_dim, hidden);
        try aliases.recDispatchMatmul(&rec, &k_matmul, &.{ &sc.x_norm, &layer.k_proj.?, &sc.k }, 1, kv_dim, hidden);
        try aliases.recDispatchMatmul(&rec, &k_matmul, &.{ &sc.x_norm, &layer.v_proj.?, &sc.v }, 1, kv_dim, hidden);

        try aliases.recDispatchRope(&rec, &k_rope, &.{ &sc.q, &sc.q_rot }, &rope_q_push, cfg.num_attention_heads, cfg.head_dim);
        try aliases.recDispatchRope(&rec, &k_rope, &.{ &sc.k, &sc.k_rot }, &rope_k_push, cfg.num_key_value_heads, cfg.head_dim);

        try aliases.recDispatch1D(&rec, &k_attn, &.{ &sc.v, &sc.head_out }, &attn_push, q_dim);

        try aliases.recDispatchMatmul(&rec, &k_matmul, &.{ &sc.head_out, &layer.o_proj.?, &sc.attn_out }, 1, hidden, q_dim);

        try aliases.recDispatch1D(&rec, &k_add, &.{ &sc.stream, &sc.attn_out }, &add_push, hidden);

        try aliases.recDispatchPerRow(&rec, &k_rmsnorm, &.{ &sc.stream, &layer.post_attention_layernorm, &sc.mid_norm }, &rms_push, 1);

        try aliases.recDispatchMatmul(&rec, &k_matmul, &.{ &sc.mid_norm, &layer.gate_proj, &sc.gate }, 1, inter, hidden);
        try aliases.recDispatchMatmul(&rec, &k_matmul, &.{ &sc.mid_norm, &layer.up_proj, &sc.up }, 1, inter, hidden);

        try aliases.recDispatch1D(&rec, &k_geglu, &.{ &sc.gate, &sc.up, &sc.fused }, &geglu_push, inter);

        try aliases.recDispatchMatmul(&rec, &k_matmul, &.{ &sc.fused, &layer.down_proj, &sc.ffn_out }, 1, hidden, inter);

        try aliases.recDispatch1D(&rec, &k_add, &.{ &sc.stream, &sc.ffn_out }, &add_push, hidden);
    }

    // Final rmsnorm + LM head.
    try aliases.recDispatchPerRow(&rec, &k_rmsnorm, &.{ &sc.stream, &gm.final_norm, &sc.final_norm_out }, &rms_push, 1);
    try aliases.recDispatchMatmul(&rec, &k_matmul, &.{ &sc.final_norm_out, &gm.lm_head, &sc.logits }, 1, vocab, hidden);

    try rec.endAndSubmit();

    const t_gpu1 = std.time.nanoTimestamp();
    const gpu_ms = @as(f64, @floatFromInt(t_gpu1 - t_gpu0)) / 1_000_000.0;

    // ── Read back logits, argmax, decode ───────────────────────────
    const logits = try gpa.alloc(f32, cfg.vocab_size);
    defer gpa.free(logits);
    try sc.logits.readBack(&ctx, f32, logits);

    const sampled = cpu_forward.argmax(logits);
    const out_str = tok.decode(sampled) orelse "<unknown>";

    try stdout.print("forward (GPU, ~291 dispatches in 1 command buffer): {d:.0} ms\n", .{gpu_ms});

    // A second forward to measure the warm-cache time. Recorder needs
    // to be reset before re-recording.
    try rec.reset();
    const t_warm0 = std.time.nanoTimestamp();
    try rec.begin();
    try aliases.recDispatch1D(&rec, &k_embed, &.{ &gm.embed_tokens, &sc.stream }, &embed_push, hidden);
    for (gm.layers) |*layer| {
        try aliases.recDispatchPerRow(&rec, &k_rmsnorm, &.{ &sc.stream, &layer.input_layernorm, &sc.x_norm }, &rms_push, 1);
        try aliases.recDispatchMatmul(&rec, &k_matmul, &.{ &sc.x_norm, &layer.q_proj.?, &sc.q }, 1, q_dim, hidden);
        try aliases.recDispatchMatmul(&rec, &k_matmul, &.{ &sc.x_norm, &layer.k_proj.?, &sc.k }, 1, kv_dim, hidden);
        try aliases.recDispatchMatmul(&rec, &k_matmul, &.{ &sc.x_norm, &layer.v_proj.?, &sc.v }, 1, kv_dim, hidden);
        try aliases.recDispatchRope(&rec, &k_rope, &.{ &sc.q, &sc.q_rot }, &rope_q_push, cfg.num_attention_heads, cfg.head_dim);
        try aliases.recDispatchRope(&rec, &k_rope, &.{ &sc.k, &sc.k_rot }, &rope_k_push, cfg.num_key_value_heads, cfg.head_dim);
        try aliases.recDispatch1D(&rec, &k_attn, &.{ &sc.v, &sc.head_out }, &attn_push, q_dim);
        try aliases.recDispatchMatmul(&rec, &k_matmul, &.{ &sc.head_out, &layer.o_proj.?, &sc.attn_out }, 1, hidden, q_dim);
        try aliases.recDispatch1D(&rec, &k_add, &.{ &sc.stream, &sc.attn_out }, &add_push, hidden);
        try aliases.recDispatchPerRow(&rec, &k_rmsnorm, &.{ &sc.stream, &layer.post_attention_layernorm, &sc.mid_norm }, &rms_push, 1);
        try aliases.recDispatchMatmul(&rec, &k_matmul, &.{ &sc.mid_norm, &layer.gate_proj, &sc.gate }, 1, inter, hidden);
        try aliases.recDispatchMatmul(&rec, &k_matmul, &.{ &sc.mid_norm, &layer.up_proj, &sc.up }, 1, inter, hidden);
        try aliases.recDispatch1D(&rec, &k_geglu, &.{ &sc.gate, &sc.up, &sc.fused }, &geglu_push, inter);
        try aliases.recDispatchMatmul(&rec, &k_matmul, &.{ &sc.fused, &layer.down_proj, &sc.ffn_out }, 1, hidden, inter);
        try aliases.recDispatch1D(&rec, &k_add, &.{ &sc.stream, &sc.ffn_out }, &add_push, hidden);
    }
    try aliases.recDispatchPerRow(&rec, &k_rmsnorm, &.{ &sc.stream, &gm.final_norm, &sc.final_norm_out }, &rms_push, 1);
    try aliases.recDispatchMatmul(&rec, &k_matmul, &.{ &sc.final_norm_out, &gm.lm_head, &sc.logits }, 1, vocab, hidden);
    try rec.endAndSubmit();
    const t_warm1 = std.time.nanoTimestamp();
    const warm_ms = @as(f64, @floatFromInt(t_warm1 - t_warm0)) / 1_000_000.0;
    try stdout.print("forward (warm, 2nd pass)                          : {d:.0} ms\n", .{warm_ms});

    const k_top: usize = 5;
    const top = try helpers.topK(gpa, logits, k_top);
    defer gpa.free(top);
    try stdout.print("\ntop {d} logits (GPU):\n", .{k_top});
    for (top) |entry| {
        const s = tok.decode(entry.id) orelse "<unknown>";
        try stdout.print("  id={d:>6}  logit={d:>10.4}  {s}\n", .{ entry.id, entry.value, s });
    }
    try stdout.print("\nGPU sampled (greedy): id={d}  string={s}\n", .{ sampled, out_str });

    // ── Parity vs CPU --gen ─────────────────────────────────────────
    try stdout.print("\nrunning CPU forward for parity check (this takes ~18 s)...\n", .{});
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const cpu_logits = try gpa.alloc(f32, cfg.vocab_size);
    defer gpa.free(cpu_logits);
    const t_cpu0 = std.time.nanoTimestamp();
    try cpu_forward.forward(&cpu, token_id, 0, arena.allocator(), cpu_logits);
    const t_cpu1 = std.time.nanoTimestamp();
    const cpu_ms = @as(f64, @floatFromInt(t_cpu1 - t_cpu0)) / 1_000_000.0;

    const cpu_argmax = cpu_forward.argmax(cpu_logits);
    var max_abs: f32 = 0;
    var max_idx: usize = 0;
    for (logits, cpu_logits, 0..) |g, c, i| {
        const d = @abs(g - c);
        if (d > max_abs) {
            max_abs = d;
            max_idx = i;
        }
    }

    try stdout.print("CPU forward: {d:.0} ms\n", .{cpu_ms});
    try stdout.print("CPU argmax: id={d} ({s}) logit={d:.4}\n", .{
        cpu_argmax, tok.decode(cpu_argmax) orelse "?", cpu_logits[cpu_argmax],
    });
    try stdout.print("max |Δ| over the whole logit vector = {e:.3}  (at idx {d}: cpu={d:.4} gpu={d:.4})\n", .{
        max_abs, max_idx, cpu_logits[max_idx], logits[max_idx],
    });
    if (sampled != cpu_argmax) {
        std.debug.print("FAIL: GPU argmax {d} ≠ CPU argmax {d}\n", .{ sampled, cpu_argmax });
        return error.ParityFailed;
    }
    try stdout.print("\nPASS GPU --gen argmax matches CPU --gen ({d} → {s})\n", .{ sampled, out_str });
}

// ── gpu-gen-qwen35: hybrid forward end-to-end (linear + full attn) ─

pub fn runGpuGenQwen35(gpa: std.mem.Allocator, dir_path: []const u8, token_id: u32) !void {
    var cpu = try model_mod.Model.load(gpa, dir_path);
    defer cpu.deinit();
    const cfg = cpu.config;
    if (cfg.family != .qwen35) return error.NotHybridFamily;

    const tok_path = try std.fmt.allocPrint(gpa, "{s}/tokenizer.json", .{dir_path});
    defer gpa.free(tok_path);
    var tok = try tokenizer_mod.Tokenizer.loadFromFile(gpa, tok_path);
    defer tok.deinit();

    var ctx = try vk.Context.init(gpa);
    defer ctx.deinit();

    const stdout = std.io.getStdOut().writer();
    const input_str = tok.decode(token_id) orelse "<unknown>";
    try stdout.print("input  token id={d}  string={s}\n", .{ token_id, input_str });
    try stdout.print("device: {s}\n", .{ctx.deviceName()});
    try stdout.print("schedule: {d} layers ({d} linear + {d} full)\n", .{
        cfg.num_hidden_layers,
        blk: {
            var n: usize = 0;
            for (cfg.layer_types[0..cfg.num_hidden_layers]) |t| if (t == .linear_attention) {
                n += 1;
            };
            break :blk n;
        },
        blk: {
            var n: usize = 0;
            for (cfg.layer_types[0..cfg.num_hidden_layers]) |t| if (t == .full_attention) {
                n += 1;
            };
            break :blk n;
        },
    });

    // ── Upload weights ──────────────────────────────────────────────
    const t_up0 = std.time.nanoTimestamp();
    var gm = try gpu_model.GpuModel.upload(gpa, &ctx, &cpu, .fp32_all);
    defer gm.deinit(ctx.device);
    const t_up1 = std.time.nanoTimestamp();
    try stdout.print("upload: {d:.0} ms\n\n", .{@as(f64, @floatFromInt(t_up1 - t_up0)) / 1_000_000.0});

    // ── Sizes ───────────────────────────────────────────────────────
    const hidden: u32 = @intCast(cfg.hidden_size);
    const inter: u32 = @intCast(cfg.intermediate_size);
    const vocab: u32 = @intCast(cfg.vocab_size);
    const head_dim: u32 = @intCast(cfg.head_dim);
    const n_q_heads: u32 = @intCast(cfg.num_attention_heads);
    const n_kv_heads: u32 = @intCast(cfg.num_key_value_heads);
    const heads_per_kv: u32 = n_q_heads / n_kv_heads;
    const q_dim: u32 = n_q_heads * head_dim;
    const q_proj_rows: u32 = if (cfg.attn_output_gate) 2 * q_dim else q_dim;
    const kv_dim: u32 = n_kv_heads * head_dim;
    const rotary_dim: u32 = @intFromFloat(@as(f32, @floatFromInt(head_dim)) * cfg.partial_rotary_factor);

    const conv_dim: u32 = @intCast(cfg.linearAttnConvDim());
    const value_dim: u32 = @intCast(cfg.linear_num_value_heads * cfg.linear_value_head_dim);
    const key_dim: u32 = @intCast(cfg.linear_num_key_heads * cfg.linear_key_head_dim);
    const n_v_heads: u32 = @intCast(cfg.linear_num_value_heads);
    const n_k_heads_lin: u32 = @intCast(cfg.linear_num_key_heads);
    const head_k: u32 = @intCast(cfg.linear_key_head_dim);
    const head_v: u32 = @intCast(cfg.linear_value_head_dim);
    const conv_kernel: u32 = @intCast(cfg.linear_conv_kernel_dim);
    const q_scale: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(cfg.linear_key_head_dim)));

    // ── Scratch buffers (shared across layers, reused) ──────────────
    var sc_stream = try buffer.Buffer.initDeviceOnly(&ctx, hidden * @sizeOf(f32));
    defer sc_stream.deinit(ctx.device);
    var sc_x_norm = try buffer.Buffer.initDeviceOnly(&ctx, hidden * @sizeOf(f32));
    defer sc_x_norm.deinit(ctx.device);
    var sc_mid_norm = try buffer.Buffer.initDeviceOnly(&ctx, hidden * @sizeOf(f32));
    defer sc_mid_norm.deinit(ctx.device);
    var sc_attn_out = try buffer.Buffer.initDeviceOnly(&ctx, hidden * @sizeOf(f32));
    defer sc_attn_out.deinit(ctx.device);
    var sc_gate = try buffer.Buffer.initDeviceOnly(&ctx, inter * @sizeOf(f32));
    defer sc_gate.deinit(ctx.device);
    var sc_up = try buffer.Buffer.initDeviceOnly(&ctx, inter * @sizeOf(f32));
    defer sc_up.deinit(ctx.device);
    var sc_fused = try buffer.Buffer.initDeviceOnly(&ctx, inter * @sizeOf(f32));
    defer sc_fused.deinit(ctx.device);
    var sc_ffn_out = try buffer.Buffer.initDeviceOnly(&ctx, hidden * @sizeOf(f32));
    defer sc_ffn_out.deinit(ctx.device);
    var sc_final_norm_out = try buffer.Buffer.initDeviceOnly(&ctx, hidden * @sizeOf(f32));
    defer sc_final_norm_out.deinit(ctx.device);
    var sc_logits = try buffer.Buffer.initDeviceOnly(&ctx, vocab * @sizeOf(f32));
    defer sc_logits.deinit(ctx.device);

    // Linear-attn scratch
    var sc_mixed_qkv = try buffer.Buffer.initDeviceOnly(&ctx, conv_dim * @sizeOf(f32));
    defer sc_mixed_qkv.deinit(ctx.device);
    // Distinct post-conv buffer to avoid aliasing readonly/writeonly
    // bindings of the same memory range — Vulkan's access decorations
    // don't promise sane behavior when both qualifiers point to the
    // same buffer, and we hit it in practice.
    var sc_mixed_qkv_post = try buffer.Buffer.initDeviceOnly(&ctx, conv_dim * @sizeOf(f32));
    defer sc_mixed_qkv_post.deinit(ctx.device);
    var sc_z = try buffer.Buffer.initDeviceOnly(&ctx, value_dim * @sizeOf(f32));
    defer sc_z.deinit(ctx.device);
    var sc_braw = try buffer.Buffer.initDeviceOnly(&ctx, n_v_heads * @sizeOf(f32));
    defer sc_braw.deinit(ctx.device);
    var sc_araw = try buffer.Buffer.initDeviceOnly(&ctx, n_v_heads * @sizeOf(f32));
    defer sc_araw.deinit(ctx.device);
    var sc_qlin = try buffer.Buffer.initDeviceOnly(&ctx, key_dim * @sizeOf(f32));
    defer sc_qlin.deinit(ctx.device);
    var sc_klin = try buffer.Buffer.initDeviceOnly(&ctx, key_dim * @sizeOf(f32));
    defer sc_klin.deinit(ctx.device);
    var sc_vlin = try buffer.Buffer.initDeviceOnly(&ctx, value_dim * @sizeOf(f32));
    defer sc_vlin.deinit(ctx.device);
    var sc_qlin_n = try buffer.Buffer.initDeviceOnly(&ctx, key_dim * @sizeOf(f32));
    defer sc_qlin_n.deinit(ctx.device);
    var sc_klin_n = try buffer.Buffer.initDeviceOnly(&ctx, key_dim * @sizeOf(f32));
    defer sc_klin_n.deinit(ctx.device);
    var sc_y = try buffer.Buffer.initDeviceOnly(&ctx, value_dim * @sizeOf(f32));
    defer sc_y.deinit(ctx.device);
    var sc_post_norm = try buffer.Buffer.initDeviceOnly(&ctx, value_dim * @sizeOf(f32));
    defer sc_post_norm.deinit(ctx.device);

    // Full-attn scratch
    var sc_q_gate = try buffer.Buffer.initDeviceOnly(&ctx, q_proj_rows * @sizeOf(f32));
    defer sc_q_gate.deinit(ctx.device);
    var sc_q = try buffer.Buffer.initDeviceOnly(&ctx, q_dim * @sizeOf(f32));
    defer sc_q.deinit(ctx.device);
    var sc_gate_attn = try buffer.Buffer.initDeviceOnly(&ctx, q_dim * @sizeOf(f32));
    defer sc_gate_attn.deinit(ctx.device);
    var sc_k = try buffer.Buffer.initDeviceOnly(&ctx, kv_dim * @sizeOf(f32));
    defer sc_k.deinit(ctx.device);
    var sc_v = try buffer.Buffer.initDeviceOnly(&ctx, kv_dim * @sizeOf(f32));
    defer sc_v.deinit(ctx.device);
    var sc_qrot = try buffer.Buffer.initDeviceOnly(&ctx, q_dim * @sizeOf(f32));
    defer sc_qrot.deinit(ctx.device);
    var sc_krot = try buffer.Buffer.initDeviceOnly(&ctx, kv_dim * @sizeOf(f32));
    defer sc_krot.deinit(ctx.device);
    var sc_head_out = try buffer.Buffer.initDeviceOnly(&ctx, q_dim * @sizeOf(f32));
    defer sc_head_out.deinit(ctx.device);
    var sc_head_out_gated = try buffer.Buffer.initDeviceOnly(&ctx, q_dim * @sizeOf(f32));
    defer sc_head_out_gated.deinit(ctx.device);
    const max_pos: u32 = 1; // Single-token gen.
    var sc_scores = try buffer.Buffer.initDeviceOnly(&ctx, n_q_heads * max_pos * @sizeOf(f32));
    defer sc_scores.deinit(ctx.device);

    // ── Per-layer persistent state ──────────────────────────────────
    // Linear layers: SSM state buffers (zero-initialised by initDeviceOnly).
    // Full layers: KV cache (one slot since max_pos=1).
    var ssm_conv = try gpa.alloc(?buffer.Buffer, cfg.num_hidden_layers);
    defer gpa.free(ssm_conv);
    var ssm_rec = try gpa.alloc(?buffer.Buffer, cfg.num_hidden_layers);
    defer gpa.free(ssm_rec);
    var kv_k = try gpa.alloc(?buffer.Buffer, cfg.num_hidden_layers);
    defer gpa.free(kv_k);
    var kv_v = try gpa.alloc(?buffer.Buffer, cfg.num_hidden_layers);
    defer gpa.free(kv_v);
    @memset(ssm_conv, null);
    @memset(ssm_rec, null);
    @memset(kv_k, null);
    @memset(kv_v, null);
    defer for (ssm_conv) |*b| if (b.*) |*v| v.deinit(ctx.device);
    defer for (ssm_rec) |*b| if (b.*) |*v| v.deinit(ctx.device);
    defer for (kv_k) |*b| if (b.*) |*v| v.deinit(ctx.device);
    defer for (kv_v) |*b| if (b.*) |*v| v.deinit(ctx.device);

    for (cfg.layer_types[0..cfg.num_hidden_layers], 0..) |lt, i| switch (lt) {
        .linear_attention => {
            ssm_conv[i] = try buffer.Buffer.initDeviceOnly(&ctx, conv_dim * conv_kernel * @sizeOf(f32));
            ssm_rec[i] = try buffer.Buffer.initDeviceOnly(&ctx, n_v_heads * head_k * head_v * @sizeOf(f32));
        },
        .full_attention => {
            kv_k[i] = try buffer.Buffer.initDeviceOnly(&ctx, max_pos * kv_dim * @sizeOf(f32));
            kv_v[i] = try buffer.Buffer.initDeviceOnly(&ctx, max_pos * kv_dim * @sizeOf(f32));
        },
    };

    // ── Kernels ─────────────────────────────────────────────────────
    var k_embed = try pipeline.Kernel.init(&ctx, &shaders.embed_lookup, 2, @sizeOf(aliases.EmbedLookupPush));
    defer k_embed.deinit();
    var k_rmsnorm = try pipeline.Kernel.init(&ctx, &shaders.rmsnorm, 3, @sizeOf(aliases.RmsnormPush));
    defer k_rmsnorm.deinit();
    var k_matmul = try pipeline.Kernel.init(&ctx, &shaders.matmul_nt_v2, 3, @sizeOf(aliases.MatmulPush));
    defer k_matmul.deinit();
    var k_add = try pipeline.Kernel.init(&ctx, &shaders.add_in_place, 2, @sizeOf(aliases.AddInPlacePush));
    defer k_add.deinit();
    var k_swiglu = try pipeline.Kernel.init(&ctx, &shaders.swiglu, 3, @sizeOf(aliases.GegluPush));
    defer k_swiglu.deinit();
    var k_rope_p = try pipeline.Kernel.init(&ctx, &shaders.rope_partial, 2, @sizeOf(aliases.RopePartialPush));
    defer k_rope_p.deinit();
    var k_split_qg = try pipeline.Kernel.init(&ctx, &shaders.split_q_gate, 3, @sizeOf(aliases.SplitQGatePush));
    defer k_split_qg.deinit();
    var k_sigmul = try pipeline.Kernel.init(&ctx, &shaders.sigmoid_mul, 3, @sizeOf(aliases.SigmoidMulPush));
    defer k_sigmul.deinit();
    var k_l2 = try pipeline.Kernel.init(&ctx, &shaders.l2norm_per_head, 2, @sizeOf(aliases.L2normPush));
    defer k_l2.deinit();
    var k_conv1d = try pipeline.Kernel.init(&ctx, &shaders.conv1d_update, 4, @sizeOf(aliases.Conv1dUpdatePush));
    defer k_conv1d.deinit();
    var k_rms_gated = try pipeline.Kernel.init(&ctx, &shaders.rmsnorm_gated, 4, @sizeOf(aliases.RmsnormGatedPush));
    defer k_rms_gated.deinit();
    var k_gds = try pipeline.Kernel.init(&ctx, &shaders.gated_delta_step, 9, @sizeOf(aliases.GatedDeltaStepPush));
    defer k_gds.deinit();
    var k_kv_write = try pipeline.Kernel.init(&ctx, &shaders.kv_write, 2, @sizeOf(aliases.KvWritePush));
    defer k_kv_write.deinit();
    var k_scores = try pipeline.Kernel.init(&ctx, &shaders.attn_scores, 3, @sizeOf(aliases.AttnScoresPush));
    defer k_scores.deinit();
    var k_softmax = try pipeline.Kernel.init(&ctx, &shaders.softmax, 2, @sizeOf(aliases.SoftmaxPush));
    defer k_softmax.deinit();
    var k_attn_out = try pipeline.Kernel.init(&ctx, &shaders.attn_output, 3, @sizeOf(aliases.AttnOutputPush));
    defer k_attn_out.deinit();
    var k_slice_copy = try pipeline.Kernel.init(&ctx, &shaders.slice_copy, 2, @sizeOf(aliases.SliceCopyPush));
    defer k_slice_copy.deinit();
    var k_scale = try pipeline.Kernel.init(&ctx, &shaders.scale, 2, @sizeOf(aliases.ScalePush));
    defer k_scale.deinit();

    // ── Recorder: one command buffer for the entire forward ────────
    var rec = try gpu_recorder.Recorder.init(&ctx, 2048, 8192);
    defer rec.deinit();

    // ── Common push constants ───────────────────────────────────────
    const gemma_quirk: u32 = if (cfg.family.rmsnormAddOne()) 1 else 0;
    const rms_push = aliases.RmsnormPush{ .dim = hidden, .eps = cfg.rms_norm_eps, .gemma_quirk = gemma_quirk };
    const qkn_push = aliases.RmsnormPush{ .dim = head_dim, .eps = cfg.rms_norm_eps, .gemma_quirk = gemma_quirk };
    const add_push = aliases.AddInPlacePush{ .n = hidden };
    const embed_push = aliases.EmbedLookupPush{ .token_id = token_id, .dim = hidden, .scale = 1.0 };
    const swiglu_push = aliases.GegluPush{ .n = inter };
    const conv1d_push = aliases.Conv1dUpdatePush{ .conv_dim = conv_dim, .kernel_size = conv_kernel };
    const l2_push = aliases.L2normPush{ .head_dim = head_k, .eps = 1e-6 };
    const rms_gated_push = aliases.RmsnormGatedPush{ .head_dim = head_v, .eps = cfg.rms_norm_eps };
    const gds_push = aliases.GatedDeltaStepPush{
        .num_k_heads = n_k_heads_lin,
        .num_v_heads = n_v_heads,
        .head_k = head_k,
        .head_v = head_v,
    };
    const split_push = aliases.SplitQGatePush{ .num_heads = n_q_heads, .head_dim = head_dim };
    const sigmul_push = aliases.SigmoidMulPush{ .n_elem = q_dim };
    const rope_q_push = aliases.RopePartialPush{
        .n_heads = n_q_heads,
        .head_dim = head_dim,
        .rotary_dim = rotary_dim,
        .pos = 0,
        .theta_base = cfg.rope_theta,
    };
    const rope_k_push = aliases.RopePartialPush{
        .n_heads = n_kv_heads,
        .head_dim = head_dim,
        .rotary_dim = rotary_dim,
        .pos = 0,
        .theta_base = cfg.rope_theta,
    };
    const kv_write_push = aliases.KvWritePush{ .n = kv_dim, .dst_off = 0 }; // pos=0
    const scores_push = aliases.AttnScoresPush{
        .n_heads = n_q_heads,
        .heads_per_kv = heads_per_kv,
        .head_dim = head_dim,
        .n_pos = 1,
        .kv_stride = kv_dim,
        .scores_stride = max_pos,
        .inv_sqrt_dim = 1.0 / @sqrt(@as(f32, @floatFromInt(head_dim))),
    };
    const softmax_push = aliases.SoftmaxPush{ .dim = 1, .stride = max_pos };
    const attn_out_push = aliases.AttnOutputPush{
        .n_heads = n_q_heads,
        .heads_per_kv = heads_per_kv,
        .head_dim = head_dim,
        .n_pos = 1,
        .kv_stride = kv_dim,
        .scores_stride = max_pos,
    };

    const t_gpu0 = std.time.nanoTimestamp();
    try rec.begin();

    // Embed lookup → residual stream.
    try aliases.recDispatch1D(&rec, &k_embed, &.{ &gm.embed_tokens, &sc_stream }, &embed_push, hidden);

    // Diagnostic knob: stop after this many layers (env QWEN35_STOP).
    // -1 / unset means run all. Useful for layer-bisection debugging.
    const stop_after: i32 = blk: {
        const env_val = std.process.getEnvVarOwned(gpa, "QWEN35_STOP") catch break :blk -1;
        defer gpa.free(env_val);
        break :blk std.fmt.parseInt(i32, env_val, 10) catch -1;
    };

    for (gm.layers, 0..) |*layer, i| {
        if (stop_after >= 0 and @as(i32, @intCast(i)) >= stop_after) break;
        try aliases.recDispatchPerRow(&rec, &k_rmsnorm, &.{ &sc_stream, &layer.input_layernorm, &sc_x_norm }, &rms_push, 1);

        switch (layer.layer_type) {
            .linear_attention => {
                // 1. Input projections.
                try aliases.recDispatchMatmul(&rec, &k_matmul, &.{ &sc_x_norm, &layer.in_proj_qkv.?, &sc_mixed_qkv }, 1, conv_dim, hidden);
                try aliases.recDispatchMatmul(&rec, &k_matmul, &.{ &sc_x_norm, &layer.in_proj_z.?, &sc_z }, 1, value_dim, hidden);
                try aliases.recDispatchMatmul(&rec, &k_matmul, &.{ &sc_x_norm, &layer.in_proj_b.?, &sc_braw }, 1, n_v_heads, hidden);
                try aliases.recDispatchMatmul(&rec, &k_matmul, &.{ &sc_x_norm, &layer.in_proj_a.?, &sc_araw }, 1, n_v_heads, hidden);

                // 2. Causal conv1d update + SiLU. Read from
                //    sc_mixed_qkv, write to sc_mixed_qkv_post (distinct
                //    buffers to avoid Vulkan readonly/writeonly aliasing).
                try aliases.recDispatch1D(&rec, &k_conv1d, &.{ &sc_mixed_qkv, &layer.conv1d_weight.?, &ssm_conv[i].?, &sc_mixed_qkv_post }, &conv1d_push, conv_dim);

                // 3. Split mixed_qkv_post into (q_lin, k_lin, v_lin).
                //    The in_proj_qkv block lays them out contiguously:
                //    q[0..key_dim], k[key_dim..2*key_dim],
                //    v[2*key_dim..conv_dim].
                const slice_q_push = aliases.SliceCopyPush{ .src_off = 0,           .dst_off = 0, .n_elem = key_dim };
                const slice_k_push = aliases.SliceCopyPush{ .src_off = key_dim,     .dst_off = 0, .n_elem = key_dim };
                const slice_v_push = aliases.SliceCopyPush{ .src_off = 2 * key_dim, .dst_off = 0, .n_elem = value_dim };
                try aliases.recDispatch1D(&rec, &k_slice_copy, &.{ &sc_mixed_qkv_post, &sc_qlin }, &slice_q_push, key_dim);
                try aliases.recDispatch1D(&rec, &k_slice_copy, &.{ &sc_mixed_qkv_post, &sc_klin }, &slice_k_push, key_dim);
                try aliases.recDispatch1D(&rec, &k_slice_copy, &.{ &sc_mixed_qkv_post, &sc_vlin }, &slice_v_push, value_dim);

                // 4. Per-head L2-norm on Q and K (each over head_k).
                try rec.dispatch(&k_l2, &.{ &sc_qlin, &sc_qlin_n }, &l2_push, n_k_heads_lin, 1, 1);
                try rec.dispatch(&k_l2, &.{ &sc_klin, &sc_klin_n }, &l2_push, n_k_heads_lin, 1, 1);

                // 4b. Apply 1/sqrt(head_k) scale to Q (in place via
                //    a separate output slot — Vulkan disallows binding
                //    the same buffer to a writeonly + readonly slot).
                const scale_push = aliases.ScalePush{ .n = key_dim, .scale = q_scale };
                try aliases.recDispatch1D(&rec, &k_scale, &.{ &sc_qlin_n, &sc_qlin }, &scale_push, key_dim);

                // 5. Recurrent gated-delta step. After step 4b,
                //    `sc_qlin` holds the L2-normed AND scaled Q;
                //    `sc_klin_n` holds the L2-normed K (no scale).
                try rec.dispatch(&k_gds, &.{
                    &ssm_rec[i].?, &sc_qlin, &sc_klin_n, &sc_vlin,
                    &sc_braw, &sc_araw, &layer.A_log.?, &layer.dt_bias.?,
                    &sc_y,
                }, &gds_push, n_v_heads, 1, 1);

                // 6. RMSNormGated with z-gate.
                try rec.dispatch(&k_rms_gated, &.{ &sc_y, &sc_z, &layer.ssm_norm_weight.?, &sc_post_norm }, &rms_gated_push, n_v_heads, 1, 1);

                // 7. Output projection.
                try aliases.recDispatchMatmul(&rec, &k_matmul, &.{ &sc_post_norm, &layer.out_proj.?, &sc_attn_out }, 1, hidden, value_dim);
            },
            .full_attention => {
                // Q-projection 2× wide, then split into (q, gate).
                try aliases.recDispatchMatmul(&rec, &k_matmul, &.{ &sc_x_norm, &layer.q_proj.?, &sc_q_gate }, 1, q_proj_rows, hidden);
                try aliases.recDispatch1D(&rec, &k_split_qg, &.{ &sc_q_gate, &sc_q, &sc_gate_attn }, &split_push, q_dim);

                // K and V projections.
                try aliases.recDispatchMatmul(&rec, &k_matmul, &.{ &sc_x_norm, &layer.k_proj.?, &sc_k }, 1, kv_dim, hidden);
                try aliases.recDispatchMatmul(&rec, &k_matmul, &.{ &sc_x_norm, &layer.v_proj.?, &sc_v }, 1, kv_dim, hidden);

                // Per-head q_norm and k_norm, with the (1+w) form.
                try aliases.recDispatchPerRow(&rec, &k_rmsnorm, &.{ &sc_q, &layer.q_norm.?, &sc_q }, &qkn_push, n_q_heads);
                try aliases.recDispatchPerRow(&rec, &k_rmsnorm, &.{ &sc_k, &layer.k_norm.?, &sc_k }, &qkn_push, n_kv_heads);

                // Partial RoPE on Q and K.
                try aliases.recDispatch1D(&rec, &k_rope_p, &.{ &sc_q, &sc_qrot }, &rope_q_push, n_q_heads * head_dim);
                try aliases.recDispatch1D(&rec, &k_rope_p, &.{ &sc_k, &sc_krot }, &rope_k_push, n_kv_heads * head_dim);

                // KV cache write (single slot at pos=0).
                try aliases.recDispatch1D(&rec, &k_kv_write, &.{ &sc_krot, &kv_k[i].? }, &kv_write_push, kv_dim);
                try aliases.recDispatch1D(&rec, &k_kv_write, &.{ &sc_v, &kv_v[i].? }, &kv_write_push, kv_dim);

                // Attention scores → softmax → attn output.
                try rec.dispatch(&k_scores, &.{ &sc_qrot, &kv_k[i].?, &sc_scores }, &scores_push, n_q_heads * 1, 1, 1);
                try aliases.recDispatchPerRow(&rec, &k_softmax, &.{ &sc_scores, &sc_scores }, &softmax_push, n_q_heads);
                try rec.dispatch(&k_attn_out, &.{ &sc_scores, &kv_v[i].?, &sc_head_out }, &attn_out_push, n_q_heads * head_dim, 1, 1);

                // sigmoid(gate) * head_out.
                try aliases.recDispatch1D(&rec, &k_sigmul, &.{ &sc_head_out, &sc_gate_attn, &sc_head_out_gated }, &sigmul_push, q_dim);

                // o_proj.
                try aliases.recDispatchMatmul(&rec, &k_matmul, &.{ &sc_head_out_gated, &layer.o_proj.?, &sc_attn_out }, 1, hidden, q_dim);
            },
        }

        // First residual.
        try aliases.recDispatch1D(&rec, &k_add, &.{ &sc_stream, &sc_attn_out }, &add_push, hidden);

        // Post-attention norm.
        try aliases.recDispatchPerRow(&rec, &k_rmsnorm, &.{ &sc_stream, &layer.post_attention_layernorm, &sc_mid_norm }, &rms_push, 1);

        // SwiGLU MLP.
        try aliases.recDispatchMatmul(&rec, &k_matmul, &.{ &sc_mid_norm, &layer.gate_proj, &sc_gate }, 1, inter, hidden);
        try aliases.recDispatchMatmul(&rec, &k_matmul, &.{ &sc_mid_norm, &layer.up_proj, &sc_up }, 1, inter, hidden);
        try aliases.recDispatch1D(&rec, &k_swiglu, &.{ &sc_gate, &sc_up, &sc_fused }, &swiglu_push, inter);
        try aliases.recDispatchMatmul(&rec, &k_matmul, &.{ &sc_fused, &layer.down_proj, &sc_ffn_out }, 1, hidden, inter);

        // Second residual.
        try aliases.recDispatch1D(&rec, &k_add, &.{ &sc_stream, &sc_ffn_out }, &add_push, hidden);
    }

    // Final norm + LM head.
    try aliases.recDispatchPerRow(&rec, &k_rmsnorm, &.{ &sc_stream, &gm.final_norm, &sc_final_norm_out }, &rms_push, 1);
    try aliases.recDispatchMatmul(&rec, &k_matmul, &.{ &sc_final_norm_out, &gm.lm_head, &sc_logits }, 1, vocab, hidden);

    try rec.endAndSubmit();

    const t_gpu1 = std.time.nanoTimestamp();
    try stdout.print("forward (GPU): {d:.0} ms\n", .{@as(f64, @floatFromInt(t_gpu1 - t_gpu0)) / 1_000_000.0});

    // Read back logits.
    const logits = try gpa.alloc(f32, cfg.vocab_size);
    defer gpa.free(logits);
    try sc_logits.readBack(&ctx, f32, logits);

    const sampled = cpu_forward.argmax(logits);
    const out_str = tok.decode(sampled) orelse "<unknown>";

    const k_top: usize = 5;
    const top = try helpers.topK(gpa, logits, k_top);
    defer gpa.free(top);
    try stdout.print("\ntop {d} logits (GPU):\n", .{k_top});
    for (top) |entry| {
        const s = tok.decode(entry.id) orelse "<unknown>";
        try stdout.print("  id={d:>6}  logit={d:>10.4}  {s}\n", .{ entry.id, entry.value, s });
    }
    try stdout.print("\nGPU sampled (greedy): id={d}  string={s}\n", .{ sampled, out_str });

    // Parity vs CPU --gen.
    try stdout.print("\nrunning CPU forward for parity check...\n", .{});
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const cpu_logits = try gpa.alloc(f32, cfg.vocab_size);
    defer gpa.free(cpu_logits);
    var hyb_state = try cpu_forward.HybridState.init(gpa, &cpu, 1);
    defer hyb_state.deinit();
    const t_cpu0 = std.time.nanoTimestamp();
    try cpu_forward.forwardHybrid(&cpu, token_id, 0, &hyb_state, arena.allocator(), cpu_logits);
    const t_cpu1 = std.time.nanoTimestamp();
    try stdout.print("CPU forward: {d:.0} ms\n", .{@as(f64, @floatFromInt(t_cpu1 - t_cpu0)) / 1_000_000.0});

    const cpu_argmax = cpu_forward.argmax(cpu_logits);
    var max_abs: f32 = 0;
    var max_idx: usize = 0;
    for (logits, cpu_logits, 0..) |g, c, idx| {
        const d = @abs(g - c);
        if (d > max_abs) {
            max_abs = d;
            max_idx = idx;
        }
    }
    try stdout.print("CPU argmax: id={d} ({s}) logit={d:.4}\n", .{
        cpu_argmax, tok.decode(cpu_argmax) orelse "?", cpu_logits[cpu_argmax],
    });
    try stdout.print("max |Δ| over the whole logit vector = {e:.3}  (at idx {d}: cpu={d:.4} gpu={d:.4})\n", .{
        max_abs, max_idx, cpu_logits[max_idx], logits[max_idx],
    });
    if (sampled != cpu_argmax) {
        std.debug.print("FAIL: GPU argmax {d} ≠ CPU argmax {d}\n", .{ sampled, cpu_argmax });
        return error.ParityFailed;
    }
    try stdout.print("\nPASS GPU --gpu-gen-qwen35 argmax matches CPU --gen ({d} → {s})\n", .{ sampled, out_str });
}
