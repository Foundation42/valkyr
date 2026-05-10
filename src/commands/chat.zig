//! Chat subcommands: `--chat` (dense families: Gemma, Qwen3 dense,
//! TinyLlama, Mistral) and `--chat` for hybrid (Qwen3.5 / qwen35).
//! Owns prompt batch loading (`PromptEntry`, `loadPromptsTsv`),
//! per-token printing helpers, and the probed-forward variants used
//! when the probe bus wants per-layer hidden state. Extracted from
//! main.zig.

const std = @import("std");
const vk = @import("../gpu/vk.zig");
const pipeline = @import("../gpu/pipeline.zig");
const model_mod = @import("../model.zig");
const tokenizer_mod = @import("../tokenizer.zig");
const config_mod = @import("../config.zig");
const cpu_forward = @import("../cpu/forward.zig");
const gpu_model = @import("../gpu/model.zig");
const gpu_scratch = @import("../gpu/scratch.zig");
const gpu_recorder = @import("../gpu/recorder.zig");
const chat_template_mod = @import("../chat_template.zig");
const probe = @import("../probe.zig");
const shaders = @import("shaders");

const aliases = @import("../runtime_aliases.zig");
const lora_merge = @import("lora_merge.zig");
const runtime = @import("../runtime.zig");
const runtime_hybrid = @import("../runtime_hybrid.zig");
const runtime_mtp = @import("../runtime_mtp.zig");
const buffer = @import("../gpu/buffer.zig");

/// `--mtp` CLI options. Enables architecture-level multi-token-prediction
/// SpecDec: MTP head produces k drafts per round, main model verifies
/// them in one batched forward (`recordForwardStepBatched` at `n_q = k`),
/// and we accept the longest matching prefix. Net effect: up to k+1
/// emitted tokens per ~2 main forwards (anchor + verify), giving 1.5–2×
/// decode speedup on Qwen3.5/3.6 MTP-equipped checkpoints.
pub const MtpOptions = struct {
    /// Enable MTP-SpecDec on the chat decode path.
    enabled: bool = false,
    /// Number of drafts per round (k). 1 ≈ plain decode (waste), > 4 has
    /// diminishing returns because accept rates fall off the chain.
    draft_n: u32 = 3,
};

/// Per-linear-layer SSM-state snapshot. Each round, we copy
/// `state.ssm_{conv,rec}` for every linear-attention layer here
/// before the batched verify forward, then restore from here after
/// the accept rule fires. The redo single-token forwards then
/// re-advance the SSM by exactly `accept_count + 1` steps.
///
/// Full-attention layers don't need a snapshot — KV is "overwrite at
/// position", so the next forward at the corrected pos cleanly
/// rewrites that slot.
///
/// Sized at chat-session init; reused across all rounds and turns.
pub const SsmSnapshot = struct {
    /// `[num_layers]` per-layer Gated DeltaNet conv-state snapshot.
    /// `null` slots mark full-attention layers (no SSM to snapshot).
    ssm_conv: []?buffer.Buffer,
    /// `[num_layers]` per-layer Gated DeltaNet recurrent-state snapshot.
    ssm_rec: []?buffer.Buffer,
    allocator: std.mem.Allocator,

    pub fn init(
        gpa: std.mem.Allocator,
        ctx: *const vk.Context,
        cfg: config_mod.Config,
    ) !SsmSnapshot {
        const f = @sizeOf(f32);
        const conv_dim = cfg.linearAttnConvDim();
        const conv_kernel = cfg.linear_conv_kernel_dim;
        const head_k = cfg.linear_key_head_dim;
        const head_v = cfg.linear_value_head_dim;
        const n_v_heads = cfg.linear_num_value_heads;

        var ssm_conv = try gpa.alloc(?buffer.Buffer, cfg.num_hidden_layers);
        @memset(ssm_conv, null);
        errdefer gpa.free(ssm_conv);
        var ssm_rec = try gpa.alloc(?buffer.Buffer, cfg.num_hidden_layers);
        @memset(ssm_rec, null);
        errdefer gpa.free(ssm_rec);

        // Cleanup-on-failure walks the whole slice — null slots are
        // safely skipped, mirroring `runtime_hybrid.State.init`.
        errdefer {
            for (ssm_conv) |*b| if (b.*) |*v| v.deinit(ctx.device);
            for (ssm_rec) |*b| if (b.*) |*v| v.deinit(ctx.device);
        }
        for (cfg.layer_types[0..cfg.num_hidden_layers], 0..) |lt, i| switch (lt) {
            .linear_attention => {
                ssm_conv[i] = try buffer.Buffer.initDeviceOnly(ctx, conv_dim * conv_kernel * f);
                ssm_rec[i] = try buffer.Buffer.initDeviceOnly(ctx, n_v_heads * head_k * head_v * f);
            },
            .full_attention => {},
        };
        return .{ .ssm_conv = ssm_conv, .ssm_rec = ssm_rec, .allocator = gpa };
    }

    pub fn deinit(self: *SsmSnapshot, device: vk.c.VkDevice) void {
        for (self.ssm_conv) |*b| if (b.*) |*v| v.deinit(device);
        for (self.ssm_rec) |*b| if (b.*) |*v| v.deinit(device);
        self.allocator.free(self.ssm_conv);
        self.allocator.free(self.ssm_rec);
    }
};

/// Per-turn state for the MTP decode path. Carries the run-time MTP
/// state (per-MTP-layer KV + draft-step intermediates), the
/// `last_hidden` snapshot buffer, the per-round draft-logits readback
/// buffer, the verify-pass logits readback host buffer, the SSM
/// snapshot for hybrid rollback, and cumulative accept-rate statistics.
pub const MtpRound = struct {
    options: MtpOptions,
    state: runtime_mtp.MtpRuntimeState,
    last_hidden_buf: buffer.Buffer,
    draft_logits_buf: buffer.Buffer,
    verify_logits_host: []f32,
    ssm_snapshot: SsmSnapshot,
    /// Cumulative across all turns: total drafts proposed.
    cumulative_drafts: usize = 0,
    /// Cumulative drafts that passed verify.
    cumulative_accepted: usize = 0,
    /// Cumulative emitted tokens via MTP rounds (drafts + corrections).
    cumulative_emitted: usize = 0,
    /// Cumulative MTP rounds run.
    cumulative_rounds: usize = 0,

    pub fn init(
        gpa: std.mem.Allocator,
        ctx: *const vk.Context,
        cfg: config_mod.Config,
        opts: MtpOptions,
        max_pos: u32,
    ) !MtpRound {
        const state = try runtime_mtp.MtpRuntimeState.init(gpa, ctx, cfg, max_pos);
        errdefer @constCast(&state).deinit(ctx.device);
        const last_hidden = try buffer.Buffer.initDeviceOnly(ctx, cfg.hidden_size * @sizeOf(f32));
        errdefer @constCast(&last_hidden).deinit(ctx.device);
        const draft_logits = try buffer.Buffer.initDeviceOnly(ctx, cfg.vocab_size * @sizeOf(f32));
        errdefer @constCast(&draft_logits).deinit(ctx.device);
        // Verify is `[main_tok, d_0, ..., d_{k-1}]` at n_q = k+1 — one
        // anchor + k drafts, so the readback host buffer needs k+1
        // rows of vocab logits.
        const v_logits = try gpa.alloc(f32, (opts.draft_n + 1) * cfg.vocab_size);
        errdefer gpa.free(v_logits);
        const snapshot = try SsmSnapshot.init(gpa, ctx, cfg);
        errdefer @constCast(&snapshot).deinit(ctx.device);
        return .{
            .options = opts,
            .state = state,
            .last_hidden_buf = last_hidden,
            .draft_logits_buf = draft_logits,
            .verify_logits_host = v_logits,
            .ssm_snapshot = snapshot,
        };
    }

    pub fn deinit(self: *MtpRound, gpa: std.mem.Allocator, device: vk.c.VkDevice) void {
        self.ssm_snapshot.deinit(device);
        gpa.free(self.verify_logits_host);
        self.draft_logits_buf.deinit(device);
        self.last_hidden_buf.deinit(device);
        self.state.deinit(device);
    }
};

/// Record per-layer slice_copy dispatches that copy each linear-attn
/// layer's `state.ssm_{conv,rec}` into the corresponding snapshot
/// buffer. Caller frames it in its own `rec.begin()/endAndSubmit()`
/// or includes it in the MTP draft session.
fn recordSnapshotSsm(
    rec: *gpu_recorder.Recorder,
    ks: *const aliases.HybridChatKernels,
    state: *const aliases.HybridChatState,
    snapshot: *const SsmSnapshot,
    cfg: config_mod.Config,
) !void {
    const conv_n: u32 = @intCast(cfg.linearAttnConvDim() * cfg.linear_conv_kernel_dim);
    const rec_n: u32 = @intCast(cfg.linear_num_value_heads * cfg.linear_key_head_dim * cfg.linear_value_head_dim);
    for (cfg.layer_types[0..cfg.num_hidden_layers], 0..) |lt, i| {
        if (lt != .linear_attention) continue;
        const conv_push = runtime_hybrid.SliceCopyPush{ .src_off = 0, .dst_off = 0, .n_elem = conv_n };
        try runtime.recDispatch1D(rec, &ks.slice_copy, &.{ &state.ssm_conv[i].?, &snapshot.ssm_conv[i].? }, &conv_push, conv_n);
        const rec_push = runtime_hybrid.SliceCopyPush{ .src_off = 0, .dst_off = 0, .n_elem = rec_n };
        try runtime.recDispatch1D(rec, &ks.slice_copy, &.{ &state.ssm_rec[i].?, &snapshot.ssm_rec[i].? }, &rec_push, rec_n);
    }
}

/// Record the inverse of `recordSnapshotSsm`: copies each snapshot
/// buffer back into `state.ssm_{conv,rec}`. After this, the SSM is
/// back at the pre-verify state and we can advance it by exactly
/// `accept_count + 1` steps via single-token forwards to commit the
/// verified prefix correctly.
fn recordRestoreSsm(
    rec: *gpu_recorder.Recorder,
    ks: *const aliases.HybridChatKernels,
    state: *const aliases.HybridChatState,
    snapshot: *const SsmSnapshot,
    cfg: config_mod.Config,
) !void {
    const conv_n: u32 = @intCast(cfg.linearAttnConvDim() * cfg.linear_conv_kernel_dim);
    const rec_n: u32 = @intCast(cfg.linear_num_value_heads * cfg.linear_key_head_dim * cfg.linear_value_head_dim);
    for (cfg.layer_types[0..cfg.num_hidden_layers], 0..) |lt, i| {
        if (lt != .linear_attention) continue;
        const conv_push = runtime_hybrid.SliceCopyPush{ .src_off = 0, .dst_off = 0, .n_elem = conv_n };
        try runtime.recDispatch1D(rec, &ks.slice_copy, &.{ &snapshot.ssm_conv[i].?, &state.ssm_conv[i].? }, &conv_push, conv_n);
        const rec_push = runtime_hybrid.SliceCopyPush{ .src_off = 0, .dst_off = 0, .n_elem = rec_n };
        try runtime.recDispatch1D(rec, &ks.slice_copy, &.{ &snapshot.ssm_rec[i].?, &state.ssm_rec[i].? }, &rec_push, rec_n);
    }
}

/// One entry from a `--prompts` TSV: `<label>\t<prompt>` per line.
/// Both fields are slices into the file's read buffer (LoadedPrompts);
/// they outlive only as long as that buffer is held.
pub const PromptEntry = struct {
    label: []const u8,
    text: []const u8,
};

/// `.lvkpt` LoRA fold-in spec for `--chat --lora-ckpt`. When non-null,
/// after `GpuModel.upload` we merge the adapter deltas into the base
/// projection weights in place — zero per-token overhead afterwards.
pub const LoraCkpt = struct {
    path: []const u8,
    targets: u32,
    rank: u32,
    alpha: f32,
};

/// Output of `loadPromptsTsv`: the heap-allocated file content and an
/// `entries` slice referencing into it. Caller frees both.
pub const LoadedPrompts = struct {
    buf: []u8,
    entries: []PromptEntry,
};

/// Read a `<label>\t<prompt>` TSV from disk. One entry per line.
/// Empty lines and lines starting with `#` are skipped. Lines without
/// a tab return error.MalformedPromptsLine.
pub fn loadPromptsTsv(gpa: std.mem.Allocator, path: []const u8) !LoadedPrompts {
    const file = try std.fs.cwd().openFile(path, .{});
    defer file.close();
    const stat = try file.stat();
    const buf = try gpa.alloc(u8, stat.size);
    errdefer gpa.free(buf);
    _ = try file.readAll(buf);

    var entries = std.ArrayList(PromptEntry).init(gpa);
    errdefer entries.deinit();

    var it = std.mem.splitScalar(u8, buf, '\n');
    while (it.next()) |line_raw| {
        const line = std.mem.trimRight(u8, line_raw, "\r");
        if (line.len == 0) continue;
        if (line[0] == '#') continue;
        const tab = std.mem.indexOfScalar(u8, line, '\t') orelse return error.MalformedPromptsLine;
        const label = line[0..tab];
        const text = line[tab + 1 ..];
        if (label.len == 0 or text.len == 0) return error.MalformedPromptsLine;
        try entries.append(.{ .label = label, .text = text });
    }
    return .{ .buf = buf, .entries = try entries.toOwnedSlice() };
}

// ── chat: prompt prefill + generation, single-turn or REPL ─────────

pub fn runChat(
    gpa: std.mem.Allocator,
    dir_path: []const u8,
    single_msg: ?[]const u8,
    sample_params: cpu_forward.SampleParams,
    seed: u64,
    tq4v: bool,
    precision: gpu_model.Precision,
    probe_path: ?[]const u8,
    batch_prompts: ?[]const PromptEntry,
    probe_prefix: ?[]const u8,
    max_new: usize,
    lora_ckpt: ?LoraCkpt,
) !void {
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
    try stdout.print("uploading weights ({s} matmul path)...\n", .{switch (precision) {
        .fp32_all => "fp32",
        .bf16_matmul => "bf16",
        .q4_0_matmul => "Q4_0",
        .q4_k_matmul => "Q4_K_M",
    }});
    const t_up0_g = std.time.nanoTimestamp();
    var gm = try gpu_model.GpuModel.upload(gpa, &ctx, &cpu, precision);
    defer gm.deinit(ctx.device);
    const t_up1_g = std.time.nanoTimestamp();
    try stdout.print("  upload took {d:.0} ms\n", .{@as(f64, @floatFromInt(t_up1_g - t_up0_g)) / 1_000_000.0});

    if (lora_ckpt) |lc| {
        try stdout.print("merging LoRA checkpoint: {s}\n", .{lc.path});
        const t_merge_start = std.time.nanoTimestamp();
        try lora_merge.run(gpa, &ctx, &gm, cfg, .{
            .lvkpt_path = lc.path,
            .lora_targets = lc.targets,
            .lora_rank = lc.rank,
            .lora_alpha = lc.alpha,
        });
        const t_merge_end = std.time.nanoTimestamp();
        try stdout.print("  merge took {d:.0} ms\n", .{@as(f64, @floatFromInt(t_merge_end - t_merge_start)) / 1_000_000.0});
    }

    // KV cache sized for a generous chat. 2048 positions ≈ 18 layers ×
    // 2 (K + V) × 2048 × 256 × 4 bytes ≈ 72 MiB on disk space — fine.
    const max_pos: usize = 2048;
    var sc = try gpu_scratch.GpuScratch.init(&ctx, cfg, max_pos);
    defer sc.deinit(ctx.device);
    var kv = try gpu_scratch.GpuKvCache.init(gpa, &ctx, cfg, max_pos);
    defer kv.deinit(ctx.device);

    // TurboQuant V cache (asymmetric K=fp / V=TQ4) — only allocated
    // when --tq4v was passed. The fp32 kv buffer above is still used
    // for the K side regardless; we just substitute the V path.
    // Keeping both kv buffers allocated wastes a bit of V memory in
    // tq4v mode (kv.layers[*].v_cache goes unused) but means the
    // K-side wiring needs zero changes.
    var kv_tq4 = if (tq4v)
        try gpu_scratch.GpuKvCacheTq4.init(gpa, &ctx, cfg, max_pos)
    else
        null;
    defer if (kv_tq4) |*c| c.deinit(ctx.device);

    var k = try aliases.ChatKernels.init(&ctx, gm.precision, cfg.family, @intCast(cfg.head_dim));
    defer k.deinit();

    // TQ4 pack/unpack kernels — picked by head_dim, only built in
    // tq4v mode, kept allocated for the lifetime of the chat session.
    // 256 (Gemma 2B) and 128 (Qwen3) shader pairs are wired up; other
    // sizes would need their own .comp files and a new branch here.
    const tq_pack_spv: []align(4) const u8 = if (cfg.head_dim == 128)
        &shaders.tq4_pack_to_cache128
    else
        &shaders.tq4_pack_to_cache;
    const tq_unpack_spv: []align(4) const u8 = if (cfg.head_dim == 128)
        &shaders.tq4_unpack128
    else
        &shaders.tq4_unpack256;
    var tq_pack: ?pipeline.Kernel = if (tq4v)
        try pipeline.Kernel.init(&ctx, tq_pack_spv, 2, @sizeOf(aliases.Tq4PackPush))
    else
        null;
    defer if (tq_pack) |*kk| kk.deinit();
    var tq_unpack: ?pipeline.Kernel = if (tq4v)
        try pipeline.Kernel.init(&ctx, tq_unpack_spv, 2, 0)
    else
        null;
    defer if (tq_unpack) |*kk| kk.deinit();

    const tq4_hooks: ?aliases.Tq4VHooks = if (tq4v)
        aliases.Tq4VHooks{ .pack = &tq_pack.?, .unpack = &tq_unpack.?, .cache = &kv_tq4.? }
    else
        null;

    // Pool sizing — one full forward worth of dispatches per step.
    // Gemma 2B (18 layers) = ~345/step. Qwen3-4B (36 layers) ≈ 720/step
    // before counting q_norm/k_norm. We size proportionally with a
    // generous slack so future deeper models don't blow up here.
    const sets_per_step: u32 = @intCast(@max(@as(usize, 512), cfg.num_hidden_layers * 32));
    var rec = try gpu_recorder.Recorder.init(&ctx, sets_per_step, 2048);
    defer rec.deinit();

    // Resolve chat-template specials based on the model family. Gemma
    // uses `<bos>` + `<start_of_turn>` + `<end_of_turn>`; Qwen3 (ChatML)
    // uses `<|endoftext|>` as BOS-equivalent and `<|im_start|>` /
    // `<|im_end|>` as turn markers.
    const tmpl = try chat_template_mod.ChatTemplate.resolve(cfg.family, &tok);

    const logits = try gpa.alloc(f32, cfg.vocab_size);
    defer gpa.free(logits);
    // Sampling scratch: top-P needs 2× vocab worth (probs + index list).
    const sample_scratch = try gpa.alloc(f32, cfg.vocab_size * 2);
    defer gpa.free(sample_scratch);

    var prng = std.Random.DefaultPrng.init(seed);
    const rng = prng.random();

    // Pretty-print the sampling configuration so the user can see
    // what knobs are active.
    if (sample_params.temperature == 0.0 or sample_params.top_k == 1 or
        (sample_params.top_p >= 1.0 and sample_params.top_k == 0 and sample_params.temperature == 1.0))
    {
        // Greedy or trivially deterministic.
        try stdout.print("sampling: greedy\n", .{});
    } else {
        try stdout.print("sampling: temp={d} top_k={d} top_p={d} seed={d}\n", .{
            sample_params.temperature, sample_params.top_k, sample_params.top_p, seed,
        });
    }
    if (tq4v) try stdout.print("KV cache: K=fp32, V=TurboQuant TQ4 (asymmetric)\n", .{});

    // ── Probe wiring ─────────────────────────────────────────────────
    //
    // When --probe <path> is set, build a JSONL writer + Bus + the two
    // v0 probes (LogitProbe for D, ActivationEntropyProbe for B). The
    // null prior for D is computed by running a single forward on the
    // BOS token at pos 0; the chat loop then starts at pos 0 too,
    // overwriting that KV slot with the actual prompt token. No
    // permanent state pollution.
    const probe_info = probe.ModelInfo{
        .family = @tagName(cfg.family),
        .n_layers = @intCast(cfg.num_hidden_layers),
        .hidden_size = @intCast(cfg.hidden_size),
        .n_heads = @intCast(cfg.num_attention_heads),
        .n_kv_heads = @intCast(cfg.num_key_value_heads),
        .head_dim = @intCast(cfg.head_dim),
        .vocab_size = @intCast(cfg.vocab_size),
    };

    var probe_writer: ?probe.JsonlWriter = null;
    defer if (probe_writer) |*w| w.close();
    var probe_bus: ?probe.Bus = null;
    defer if (probe_bus) |*b| b.deinit(gpa);
    var probe_hidden_scratch: ?[]f32 = null;
    defer if (probe_hidden_scratch) |s| gpa.free(s);
    var probe_attn_scratch: ?[]f32 = null;
    defer if (probe_attn_scratch) |s| gpa.free(s);
    var probe_token_index: u32 = 0;

    if (probe_path) |pp| {
        probe_writer = try probe.JsonlWriter.open(pp);
        const w_ptr: *probe.JsonlWriter = &probe_writer.?;

        try stdout.print("probe: writing trace to {s}\n", .{pp});

        // Compute null prior: forward a single BOS token at pos 0,
        // read back the resulting logits, then hand them to LogitProbe
        // as the q distribution for KL(p || q).
        const bos: u32 = cfg.bos_token_id orelse 1;
        try stdout.print("probe: computing null prior (BOS={d})...\n", .{bos});
        try rec.reset();
        try rec.begin();
        try aliases.recordForwardStep(&rec, &sc, &gm, &kv, cfg, &k, 0, bos, tq4_hooks, true);
        try rec.endAndSubmit();
        try sc.logits.readBack(&ctx, f32, logits);
        // KV slot at pos 0 is now dirty with the BOS forward. The chat
        // loop below starts at pos = 0 and overwrites that slot with
        // the first prompt token — clean handoff, no reset needed.

        var bus = probe.Bus{};
        const probes_buf = try gpa.alloc(probe.Probe, 3);
        const lp = try probe.LogitProbe.create(gpa, w_ptr, @intCast(cfg.vocab_size), logits);
        const ap = try probe.ActivationEntropyProbe.create(gpa, w_ptr);
        const kp = try probe.AttentionProbe.create(gpa, w_ptr);
        probes_buf[0] = lp.probe();
        probes_buf[1] = ap.probe();
        probes_buf[2] = kp.probe();
        bus.probes = probes_buf;
        bus.finalize();
        probe_bus = bus;

        try probe.writeHeader(w_ptr, probe_info, dir_path, &.{ "logits", "act", "attn" });

        if (bus.needs_hidden_pre or bus.needs_hidden_post) {
            probe_hidden_scratch = try gpa.alloc(f32, cfg.hidden_size);
        }
        if (bus.needs_attention) {
            probe_attn_scratch = try gpa.alloc(f32, cfg.num_attention_heads * max_pos);
        }
    }

    // ── Batch mode: --prompts <tsv> + --probe-prefix <prefix> ────────
    //
    // Loads the model once, then runs each prompt as an independent
    // fresh-context chat turn. Per-prompt: open a fresh JsonlWriter at
    // `<prefix><label>.jsonl`, rebuild the bus + probes pointing to
    // that writer, reset pos=0, run chatTurn, tear down. Null prior +
    // probe scratches are shared across iterations. KV slots persist
    // between iterations but are overwritten by each new prompt's
    // prefill — attention only reads positions [0, n_pos) so any
    // residue beyond the new n_pos is unread.
    if (batch_prompts) |entries| {
        const bos: u32 = cfg.bos_token_id orelse 1;
        try stdout.print("probe (batch): computing null prior (BOS={d})...\n", .{bos});
        const null_prior_buf = try gpa.alloc(f32, cfg.vocab_size);
        defer gpa.free(null_prior_buf);
        try rec.reset();
        try rec.begin();
        try aliases.recordForwardStep(&rec, &sc, &gm, &kv, cfg, &k, 0, bos, tq4_hooks, true);
        try rec.endAndSubmit();
        try sc.logits.readBack(&ctx, f32, null_prior_buf);

        // All three v0 probes are installed per iteration, so the bus
        // wants hidden_post + attention every time. Allocate once.
        const hidden_scratch = try gpa.alloc(f32, cfg.hidden_size);
        defer gpa.free(hidden_scratch);
        const attn_scratch = try gpa.alloc(f32, cfg.num_attention_heads * max_pos);
        defer gpa.free(attn_scratch);

        for (entries) |entry| {
            const out_path = try std.fmt.allocPrint(gpa, "{s}{s}.jsonl", .{ probe_prefix.?, entry.label });
            defer gpa.free(out_path);
            try stdout.print("\n[{s}] → {s}\n", .{ entry.label, out_path });

            var writer = try probe.JsonlWriter.open(out_path);
            defer writer.close();

            var bus = probe.Bus{};
            const probes_buf = try gpa.alloc(probe.Probe, 3);
            const lp = try probe.LogitProbe.create(gpa, &writer, @intCast(cfg.vocab_size), null_prior_buf);
            const ap = try probe.ActivationEntropyProbe.create(gpa, &writer);
            const kp = try probe.AttentionProbe.create(gpa, &writer);
            probes_buf[0] = lp.probe();
            probes_buf[1] = ap.probe();
            probes_buf[2] = kp.probe();
            bus.probes = probes_buf;
            bus.finalize();
            defer bus.deinit(gpa);

            try probe.writeHeader(&writer, probe_info, dir_path, &.{ "logits", "act", "attn" });

            var pos: usize = 0;
            var token_index: u32 = 0;

            try chatTurn(
                gpa, &ctx, &rec, &sc, &gm, &kv, cfg, &k, &tok, entry.text, &pos, logits,
                sample_scratch, sample_params, rng, tmpl, false, tq4_hooks,
                &bus, probe_info, &token_index, hidden_scratch, attn_scratch,
                max_new,
            );
        }
        return;
    }

    // Position counter persists across turns (multi-turn chat builds on
    // the same KV cache).
    var pos: usize = 0;

    if (single_msg) |m| {
        try chatTurn(
            gpa, &ctx, &rec, &sc, &gm, &kv, cfg, &k, &tok, m, &pos, logits,
            sample_scratch, sample_params, rng, tmpl, false, tq4_hooks,
            if (probe_bus) |*b| b else null,
            probe_info,
            &probe_token_index,
            probe_hidden_scratch,
            probe_attn_scratch,
            max_new,
        );
        return;
    }

    // ── REPL ────────────────────────────────────────────────────────
    try stdout.print("\n{s} (Ctrl-D to exit)\n", .{tmpl.banner()});
    const stdin = std.io.getStdIn().reader();
    var line_buf: [4096]u8 = undefined;
    while (true) {
        try stdout.print("\nuser> ", .{});
        const maybe_line = try stdin.readUntilDelimiterOrEof(&line_buf, '\n');
        const line = maybe_line orelse {
            try stdout.print("\n", .{});
            break;
        };
        if (line.len == 0) continue;
        try chatTurn(
            gpa, &ctx, &rec, &sc, &gm, &kv, cfg, &k, &tok, line, &pos, logits,
            sample_scratch, sample_params, rng, tmpl, true, tq4_hooks,
            if (probe_bus) |*b| b else null,
            probe_info,
            &probe_token_index,
            probe_hidden_scratch,
            probe_attn_scratch,
            max_new,
        );
        if (pos >= max_pos - 64) {
            try stdout.print("\n[KV cache near capacity, ending session]\n", .{});
            break;
        }
    }
}

/// One round-trip through the model: prefill the prompt for `user_msg`
/// then sample until the family-specific end-of-turn marker (or
/// max_response). `pos` is updated in place so the next turn picks up
/// where this one stopped. The `chat_template_mod.ChatTemplate` arg encapsulates which
/// special tokens to emit and where.
fn chatTurn(
    gpa: std.mem.Allocator,
    ctx: *const vk.Context,
    rec: *gpu_recorder.Recorder,
    sc: *const gpu_scratch.GpuScratch,
    gm: *const gpu_model.GpuModel,
    kv: *const gpu_scratch.GpuKvCache,
    cfg: config_mod.Config,
    k: *const aliases.ChatKernels,
    tok: *const tokenizer_mod.Tokenizer,
    user_msg: []const u8,
    pos: *usize,
    logits: []f32,
    sample_scratch: []f32,
    sample_params: cpu_forward.SampleParams,
    rng: std.Random,
    tmpl: chat_template_mod.ChatTemplate,
    is_repl: bool,
    tq4_v: ?aliases.Tq4VHooks,
    probe_bus: ?*probe.Bus,
    probe_info: probe.ModelInfo,
    probe_token_index: *u32,
    probe_hidden_scratch: ?[]f32,
    probe_attn_scratch: ?[]f32,
    max_new: usize,
) !void {
    const stdout = std.io.getStdOut().writer();

    var prompt = std.ArrayList(u32).init(gpa);
    defer prompt.deinit();
    try tmpl.composePrompt(gpa, tok, user_msg, pos.* == 0, &prompt);

    if (!is_repl) {
        try stdout.print("\nprompt ({d} tokens, starting at pos {d}):\n  ", .{ prompt.items.len, pos.* });
        for (prompt.items) |id| try printTokenForDisplay(gpa, stdout, tok, id);
        try stdout.print("\n\nresponse: ", .{});
    } else {
        try stdout.print("model> ", .{});
    }

    const eos: ?u32 = cfg.eos_token_id;
    const eot: u32 = tmpl.end_of_turn;
    const max_response: usize = max_new;

    // Run all prompt tokens through the model. We only need the logits
    // at the LAST prefill position (which gives us the first response
    // token); intermediate logits are computed and ignored, which is a
    // small waste — fold the LM head out of prefill for a quick win
    // later if it matters.
    var current: u32 = prompt.items[0];
    var prompt_idx: usize = 0;
    var generated: usize = 0;

    // Decode-only throughput timer: clock starts after prefill so the
    // tok/s we report reflects steady-state generation, not warm-up.
    // Mirrors the Qwen path's measurement so bench script numbers are
    // apples-to-apples across model families.
    var t_decode_start: i128 = 0;
    var decode_started = false;

    while (true) {
        // We only need logits at the LAST prefill position (to sample
        // the first response token) and for every step of the sample
        // loop. Skipping the LM head matmul on intermediate prefill
        // tokens is a free ~10% startup win on short prompts and
        // scales linearly with prompt length on long ones.
        const is_last_prefill_or_decoding = (prompt_idx + 1 >= prompt.items.len);
        const is_prefill = (prompt_idx + 1 < prompt.items.len);

        if (probe_bus) |bus| try bus.onTokenStart(.{
            .info = probe_info,
            .token_index = probe_token_index.*,
            .pos = pos.*,
            .token_id = current,
            .is_prefill = is_prefill,
        });

        // Probed path forks here: when a probe wants per-layer hidden
        // state, the forward gets split into N submits with readbacks
        // between. Otherwise the fast single-submit path runs.
        const use_probed_forward = if (probe_bus) |bus|
            bus.needs_hidden_pre or bus.needs_hidden_post or bus.needs_attention
        else
            false;

        if (use_probed_forward) {
            try forwardStepProbed(
                rec,
                ctx,
                sc,
                gm,
                kv,
                cfg,
                k,
                pos.*,
                current,
                tq4_v,
                is_last_prefill_or_decoding,
                probe_bus.?,
                probe_info,
                probe_token_index.*,
                is_prefill,
                probe_hidden_scratch,
                probe_attn_scratch,
            );
        } else {
            if (pos.* > 0) try rec.reset();
            try rec.begin();
            try aliases.recordForwardStep(rec, sc, gm, kv, cfg, k, pos.*, current, tq4_v, is_last_prefill_or_decoding);
            try rec.endAndSubmit();
        }

        // If logits were computed this step, give probes that want
        // them a look before sampling.
        if (probe_bus) |bus| {
            if (is_last_prefill_or_decoding and bus.needs_logits) {
                try sc.logits.readBack(ctx, f32, logits);
                try bus.onLogits(.{
                    .info = probe_info,
                    .token_index = probe_token_index.*,
                    .pos = pos.*,
                    .token_id = current,
                    .is_prefill = is_prefill,
                }, logits);
            }
        }

        // Decide what to do at this position.
        prompt_idx += 1;
        pos.* += 1;
        probe_token_index.* += 1;

        if (prompt_idx < prompt.items.len) {
            // Still consuming prompt — advance to next prompt token.
            current = prompt.items[prompt_idx];
            continue;
        }

        if (!decode_started) {
            t_decode_start = std.time.nanoTimestamp();
            decode_started = true;
        }

        // Past the last prompt token: sample. The logits readback may
        // already have happened above for the probe, in which case the
        // device buffer is unchanged and a second readBack is just a
        // re-fetch (small cost, kept simple). If the probe path didn't
        // ask, this is the canonical readback.
        try sc.logits.readBack(ctx, f32, logits);
        const next = try cpu_forward.sample(logits, sample_params, rng, sample_scratch);

        if (probe_bus) |bus| try bus.onTokenEnd(.{
            .info = probe_info,
            .token_index = probe_token_index.*,
            .pos = pos.*,
            .token_id = current,
            .is_prefill = false,
        }, @intCast(next));

        // Stop conditions.
        if (next == eot) break;
        if (eos != null and next == eos.?) break;

        try printTokenForDisplay(gpa, stdout, tok, @intCast(next));

        generated += 1;
        if (generated >= max_response) break;
        current = @intCast(next);
    }
    try stdout.print("\n", .{});

    if (decode_started and generated > 0) {
        const t_end = std.time.nanoTimestamp();
        const ms_total = @as(f64, @floatFromInt(t_end - t_decode_start)) / 1_000_000.0;
        const tokps = @as(f64, @floatFromInt(generated)) * 1000.0 / ms_total;
        try stdout.print("[{d} tok in {d:.0} ms, {d:.1} tok/s]\n", .{ generated, ms_total, tokps });
    }
}

/// Print a single token id to `w` after applying the decoder rule for
/// the active tokenizer mode:
///   - SentencePiece (Gemma / TinyLlama / Mistral): ▁ (U+2581, bytes
///     E2 96 81) → ' '. Byte-fallback tokens with surface form
///     `<0xHH>` decode to the literal byte 0xHH (newlines, tabs, and
///     non-ASCII in models that use byte fallback for OOV codepoints).
///   - ByteLevel (Qwen3 / Llama3): walk the codepoints and reverse the
///     byte→unicode map back to original bytes.
///
/// Falls back to a "<id>" stub for token ids with no surface form (e.g.
/// holes in the vocab table).
fn printTokenForDisplay(
    gpa: std.mem.Allocator,
    w: anytype,
    tok: *const tokenizer_mod.Tokenizer,
    id: u32,
) !void {
    switch (tok.mode) {
        .sentencepiece => {
            const s = tok.decode(id) orelse {
                try w.print("<{d}>", .{id});
                return;
            };
            // Byte-fallback tokens have surface form `<0xHH>` (length
            // 6, two uppercase hex digits) and decode to byte 0xHH.
            // Detect by exact shape — any single token with that
            // surface form is the byte-fallback encoding because
            // SentencePiece reserves it for that purpose.
            if (s.len == 6 and s[0] == '<' and s[1] == '0' and s[2] == 'x' and s[5] == '>') {
                if (parseHexByte(s[3], s[4])) |b| {
                    try w.writeAll(&[_]u8{b});
                    return;
                }
            }
            var i: usize = 0;
            while (i < s.len) {
                if (i + 3 <= s.len and s[i] == 0xE2 and s[i + 1] == 0x96 and s[i + 2] == 0x81) {
                    try w.print(" ", .{});
                    i += 3;
                } else {
                    try w.print("{c}", .{s[i]});
                    i += 1;
                }
            }
        },
        .bytelevel => {
            const bytes = tok.decodeByteLevel(gpa, id) catch {
                try w.print("<{d}>", .{id});
                return;
            };
            defer gpa.free(bytes);
            try w.writeAll(bytes);
        },
    }
}

/// Parse two ASCII hex chars into the byte they represent. Returns
/// null on any non-hex input. Used by the SentencePiece byte-fallback
/// path in `printTokenForDisplay`.
fn parseHexByte(hi: u8, lo: u8) ?u8 {
    const h = hexDigit(hi) orelse return null;
    const l = hexDigit(lo) orelse return null;
    return (h << 4) | l;
}

fn hexDigit(c: u8) ?u8 {
    return switch (c) {
        '0'...'9' => c - '0',
        'a'...'f' => c - 'a' + 10,
        'A'...'F' => c - 'A' + 10,
        else => null,
    };
}

/// Probed forward: same model dispatches as `aliases.recordForwardStep` but
/// split across multiple submits so the host can read back per-layer
/// hidden state into the bus between layers. Slow path, only invoked
/// when the bus has a probe that wants `hidden_post_layer`. Generation
/// output (next-token logits) is bit-identical to the fast path
/// because the GPU does the same math in the same order — only the
/// command-buffer chunking changes.
fn forwardStepProbed(
    rec: *gpu_recorder.Recorder,
    vk_ctx: *const vk.Context,
    sc: *const gpu_scratch.GpuScratch,
    gm: *const gpu_model.GpuModel,
    kv: *const gpu_scratch.GpuKvCache,
    cfg: config_mod.Config,
    k: *const aliases.ChatKernels,
    pos: usize,
    token_id: u32,
    tq4_v: ?aliases.Tq4VHooks,
    compute_logits: bool,
    bus: *const probe.Bus,
    info: probe.ModelInfo,
    token_index: u32,
    is_prefill: bool,
    hidden_scratch: ?[]f32,
    attn_scratch: ?[]f32,
) !void {
    const hidden: u32 = @intCast(cfg.hidden_size);
    const vocab: u32 = @intCast(cfg.vocab_size);

    const pushes = aliases.computeForwardPushes(cfg, sc, pos);

    const embed_push = aliases.EmbedLookupPush{
        .token_id = token_id,
        .dim = hidden,
        .scale = if (cfg.family.embedScalesByDim()) @sqrt(@as(f32, @floatFromInt(hidden))) else 1.0,
    };

    // ── Embed ───────────────────────────────────────────────────────
    try rec.reset();
    try rec.begin();
    try aliases.recDispatch1D(rec, &k.embed, &.{ &gm.embed_tokens, &sc.stream }, &embed_push, hidden);
    try rec.endAndSubmit();

    // ── Per layer ──────────────────────────────────────────────────
    const max_pos_u32: u32 = @intCast(sc.max_pos);
    const n_heads_u32: u32 = @intCast(cfg.num_attention_heads);
    for (0..cfg.num_hidden_layers) |layer_idx| {
        const layer_ctx = probe.Context{
            .info = info,
            .token_index = token_index,
            .pos = pos,
            .token_id = token_id,
            .layer_idx = @intCast(layer_idx),
            .is_prefill = is_prefill,
        };

        if (bus.needs_hidden_pre) {
            try sc.stream.readBack(vk_ctx, f32, hidden_scratch.?);
            try bus.onLayerEntry(layer_ctx, hidden_scratch.?);
        }

        try rec.reset();
        try rec.begin();
        try aliases.recordOneLayer(rec, sc, gm, kv, cfg, k, layer_idx, pos, &pushes, tq4_v);
        try rec.endAndSubmit();

        // sc.scores holds this layer's post-softmax attention weights
        // until the next layer's softmax overwrites it. Read it now
        // while it's still valid. Layout is [n_heads, max_pos]
        // row-major; only the first n_pos columns of each row are
        // valid post-softmax — the probe slices accordingly.
        if (bus.needs_attention) {
            try sc.scores.readBack(vk_ctx, f32, attn_scratch.?);
            try bus.onAttention(layer_ctx, attn_scratch.?, n_heads_u32, pushes.n_pos, max_pos_u32);
        }

        if (bus.needs_hidden_post) {
            try sc.stream.readBack(vk_ctx, f32, hidden_scratch.?);
            try bus.onLayerExit(layer_ctx, hidden_scratch.?);
        }
    }

    // ── Final norm + LM head ───────────────────────────────────────
    if (compute_logits) {
        try rec.reset();
        try rec.begin();
        try aliases.recDispatchPerRow(rec, &k.rmsnorm, &.{ &sc.stream, &gm.final_norm, &sc.final_norm_out }, &pushes.rms_push, 1);
        try aliases.recDispatchMatmul(rec, &k.matmul_lm_head, &.{ &sc.final_norm_out, &gm.lm_head, &sc.logits }, 1, vocab, hidden);
        try rec.endAndSubmit();
    }
}

/// Probed forward through the hybrid path: same dispatches as
/// aliases.recordHybridForwardStep but split across multiple submits so the
/// host can read back per-layer hidden state and (on full-attention
/// layers only) post-softmax attention weights between layers.
/// Generation output is bit-identical to the fast path because the
/// GPU does the same math in the same order — only command-buffer
/// chunking changes.
fn forwardStepProbedHybrid(
    rec: *gpu_recorder.Recorder,
    vk_ctx: *const vk.Context,
    sc: *const aliases.HybridChatScratch,
    state: *const aliases.HybridChatState,
    gm: *const gpu_model.GpuModel,
    cfg: config_mod.Config,
    k: *const aliases.HybridChatKernels,
    pos: usize,
    token_id: u32,
    max_pos: u32,
    tq4_v: ?aliases.HybridTq4VHooks,
    compute_logits: bool,
    bus: *const probe.Bus,
    info: probe.ModelInfo,
    token_index: u32,
    is_prefill: bool,
    hidden_scratch: ?[]f32,
    attn_scratch: ?[]f32,
) !void {
    const hidden: u32 = @intCast(cfg.hidden_size);
    const vocab: u32 = @intCast(cfg.vocab_size);

    const pushes = aliases.computeHybridForwardPushes(cfg, pos, 1, max_pos);
    const embed_push = aliases.EmbedLookupPush{ .token_id = token_id, .dim = hidden, .scale = 1.0 };

    // ── Embed ───────────────────────────────────────────────────────
    try rec.reset();
    try rec.begin();
    try aliases.recDispatch1D(rec, &k.embed, &.{ &gm.embed_tokens, &sc.stream }, &embed_push, hidden);
    try rec.endAndSubmit();

    // ── Per layer ──────────────────────────────────────────────────
    for (0..cfg.num_hidden_layers) |layer_idx| {
        const layer_ctx = probe.Context{
            .info = info,
            .token_index = token_index,
            .pos = pos,
            .token_id = token_id,
            .layer_idx = @intCast(layer_idx),
            .is_prefill = is_prefill,
        };

        if (bus.needs_hidden_pre) {
            try sc.stream.readBack(vk_ctx, f32, hidden_scratch.?);
            try bus.onLayerEntry(layer_ctx, hidden_scratch.?);
        }

        try rec.reset();
        try rec.begin();
        try aliases.recordOneHybridLayer(rec, sc, state, gm, cfg, k, layer_idx, pos, &pushes, tq4_v);
        try rec.endAndSubmit();

        // sc.scores holds post-softmax attention weights ONLY on
        // full-attention layers; linear-attention layers don't write
        // it. Skip the K readback (and the K probe callback) on
        // linear layers — the analyzer aggregates per (token, layer)
        // so missing cells just don't contribute.
        const is_full_attn = gm.layers[layer_idx].layer_type == .full_attention;
        if (bus.needs_attention and is_full_attn) {
            try sc.scores.readBack(vk_ctx, f32, attn_scratch.?);
            try bus.onAttention(layer_ctx, attn_scratch.?, pushes.n_q_heads, pushes.n_pos, max_pos);
        }

        if (bus.needs_hidden_post) {
            try sc.stream.readBack(vk_ctx, f32, hidden_scratch.?);
            try bus.onLayerExit(layer_ctx, hidden_scratch.?);
        }
    }

    // ── Final norm + LM head ───────────────────────────────────────
    if (compute_logits) {
        try rec.reset();
        try rec.begin();
        try aliases.recDispatchPerRow(rec, &k.rmsnorm, &.{ &sc.stream, &gm.final_norm, &sc.final_norm_out }, &pushes.rms_push, 1);
        try aliases.recDispatchMatmul(rec, &k.matmul_lm_head, &.{ &sc.final_norm_out, &gm.lm_head, &sc.logits }, 1, vocab, hidden);
        try rec.endAndSubmit();
    }
}

pub fn runChatQwen35(
    gpa: std.mem.Allocator,
    dir_path: []const u8,
    single_msg: ?[]const u8,
    sample_params: cpu_forward.SampleParams,
    seed: u64,
    tq4v: bool,
    precision: gpu_model.Precision,
    probe_path: ?[]const u8,
    batch_prompts: ?[]const PromptEntry,
    probe_prefix: ?[]const u8,
    max_new: usize,
    mtp_options: MtpOptions,
) !void {
    var cpu = try model_mod.Model.load(gpa, dir_path);
    defer cpu.deinit();
    const cfg = cpu.config;
    if (!cfg.family.isHybrid()) return error.NotHybridFamily;
    // ── MTP validation ──────────────────────────────────────────────
    if (mtp_options.enabled) {
        if (cpu.mtp_head == null) {
            std.debug.print("--mtp: model has no MTP head — only Qwen3.5 / Qwen3.6 checkpoints with `mtp_num_hidden_layers > 0` are supported\n", .{});
            return error.NoMtpHead;
        }
        if (tq4v) {
            std.debug.print("--mtp: not yet supported with --tq4v (the verify path doesn't have a fused TQ4-V kernel — use one of --mtp or --tq4v but not both for now)\n", .{});
            return error.MtpTq4VConflict;
        }
        if (mtp_options.draft_n == 0 or mtp_options.draft_n > 16) {
            std.debug.print("--mtp-draft-n must be in [1, 16] (got {d})\n", .{mtp_options.draft_n});
            return error.InvalidMtpDraftN;
        }
    }

    const tok_path = try std.fmt.allocPrint(gpa, "{s}/tokenizer.json", .{dir_path});
    defer gpa.free(tok_path);
    var tok = try tokenizer_mod.Tokenizer.loadFromFile(gpa, tok_path);
    defer tok.deinit();

    var ctx = try vk.Context.init(gpa);
    defer ctx.deinit();

    const stdout = std.io.getStdOut().writer();
    try stdout.print("device: {s}\n", .{ctx.deviceName()});
    const prec_label: []const u8 = switch (precision) {
        .fp32_all => "fp32",
        .bf16_matmul => "bf16",
        .q4_0_matmul => "Q4_0",
        .q4_k_matmul => "Q4_K_M",
    };
    if (tq4v) {
        try stdout.print("uploading weights ({s} + TQ4-V on full-attn layers)...\n", .{prec_label});
    } else {
        try stdout.print("uploading weights ({s} path)...\n", .{prec_label});
    }
    const t_up0 = std.time.nanoTimestamp();
    var gm = try gpu_model.GpuModel.upload(gpa, &ctx, &cpu, precision);
    defer gm.deinit(ctx.device);
    const t_up1 = std.time.nanoTimestamp();
    try stdout.print("  upload took {d:.0} ms\n", .{@as(f64, @floatFromInt(t_up1 - t_up0)) / 1_000_000.0});

    // TQ4-V is only wired for head_dim ∈ {128, 256}. Both Qwen3.5 sizes
    // (0.8B, 4B) are 128, so this is mostly a guard against future
    // hybrids with odd head_dim landing on this path.
    if (tq4v and cfg.head_dim != 128 and cfg.head_dim != 256) {
        try stdout.print("note: --tq4v ignored — head_dim={d} has no TQ4 shader pair (only 128/256 wired)\n", .{cfg.head_dim});
    }
    const tq4v_active = tq4v and (cfg.head_dim == 128 or cfg.head_dim == 256);

    const max_pos: u32 = 2048;
    // MTP-SpecDec verify dispatches `recordForwardStepBatched(n_q =
    // draft_n + 1)` — one anchor (main_tok) + k drafts. Scratch needs
    // `max_n_q ≥ draft_n + 1` when MTP is on. Plain decode stays at 1.
    const scratch_max_n_q: u32 = if (mtp_options.enabled) mtp_options.draft_n + 1 else 1;
    var sc = try aliases.HybridChatScratch.init(&ctx, cfg, max_pos, scratch_max_n_q, tq4v_active);
    defer sc.deinit(ctx.device);

    // MTP runtime state — only allocated when --mtp is on. Owned for the
    // duration of the chat session; reset between turns isn't needed (the
    // per-MTP-layer KV cache is overwritten each round).
    var mtp_round: ?MtpRound = if (mtp_options.enabled)
        try MtpRound.init(gpa, &ctx, cfg, mtp_options, max_pos)
    else
        null;
    defer if (mtp_round) |*m| m.deinit(gpa, ctx.device);
    var state = try aliases.HybridChatState.init(gpa, &ctx, cfg, max_pos, tq4v_active);
    defer state.deinit(ctx.device);
    var ks = try aliases.HybridChatKernels.init(&ctx, gm.precision, @intCast(cfg.head_dim));
    defer ks.deinit();

    // TQ4 pack/unpack kernels — picked by head_dim, only built when
    // active. Mirrors the Gemma TQ4 path. Number of blocks per token =
    // num_kv_heads (8 for Qwen3.5).
    const tq_pack_spv: []align(4) const u8 = if (cfg.head_dim == 128)
        &shaders.tq4_pack_to_cache128
    else
        &shaders.tq4_pack_to_cache;
    const tq_unpack_spv: []align(4) const u8 = if (cfg.head_dim == 128)
        &shaders.tq4_unpack128
    else
        &shaders.tq4_unpack256;
    var tq_pack: ?pipeline.Kernel = if (tq4v_active)
        try pipeline.Kernel.init(&ctx, tq_pack_spv, 2, @sizeOf(aliases.Tq4PackPush))
    else
        null;
    defer if (tq_pack) |*kk| kk.deinit();
    var tq_unpack: ?pipeline.Kernel = if (tq4v_active)
        try pipeline.Kernel.init(&ctx, tq_unpack_spv, 2, 0)
    else
        null;
    defer if (tq_unpack) |*kk| kk.deinit();
    const tq4_hooks: ?aliases.HybridTq4VHooks = if (tq4v_active) aliases.HybridTq4VHooks{
        .pack = &tq_pack.?,
        .unpack = &tq_unpack.?,
        .n_blocks_per_pos = @intCast(cfg.num_key_value_heads),
    } else null;

    // Recorder pool sized for one full forward step. ~30 dispatches per
    // linear layer + ~16 per full layer + ~3 head/tail = up to ~30*32 +
    // 3 ≈ 1k sets at n_q=1. With MTP, the verify forward records every
    // layer at n_q=draft_n in one submit, and linear-attn layers loop
    // their single-row chain n_q times — so the budget scales linearly
    // in n_q. Round generously (×40 dispatches/layer × n_q is a loose
    // upper bound that covers prefill, decode, and verify submits).
    const dispatch_max_n_q: u32 = if (mtp_options.enabled) mtp_options.draft_n + 1 else 1;
    const sets_per_step: u32 = @intCast(@max(
        @as(usize, 1024),
        cfg.num_hidden_layers * 40 * @as(usize, dispatch_max_n_q),
    ));
    var rec = try gpu_recorder.Recorder.init(&ctx, sets_per_step, sets_per_step * 4);
    defer rec.deinit();

    const tmpl = try chat_template_mod.ChatTemplate.resolve(cfg.family, &tok);

    const logits = try gpa.alloc(f32, cfg.vocab_size);
    defer gpa.free(logits);
    const sample_scratch = try gpa.alloc(f32, cfg.vocab_size * 2);
    defer gpa.free(sample_scratch);

    var prng = std.Random.DefaultPrng.init(seed);
    const rng = prng.random();

    if (sample_params.temperature == 0.0 or sample_params.top_k == 1 or
        (sample_params.top_p >= 1.0 and sample_params.top_k == 0 and sample_params.temperature == 1.0))
    {
        try stdout.print("sampling: greedy\n", .{});
    } else {
        try stdout.print("sampling: temp={d} top_k={d} top_p={d} seed={d}\n", .{
            sample_params.temperature, sample_params.top_k, sample_params.top_p, seed,
        });
    }

    // ── Probe wiring ─────────────────────────────────────────────────
    //
    // Mirrors the non-hybrid runChat probe setup. Null prior is
    // computed by running a single BOS forward at pos 0; the chat
    // loop then starts at pos 0 too, overwriting both KV state and
    // SSM state on the first prompt token. K is only valid on
    // full-attention layers in the hybrid family — the probed forward
    // skips the K readback on linear-attention layers.
    const probe_info = probe.ModelInfo{
        .family = @tagName(cfg.family),
        .n_layers = @intCast(cfg.num_hidden_layers),
        .hidden_size = @intCast(cfg.hidden_size),
        .n_heads = @intCast(cfg.num_attention_heads),
        .n_kv_heads = @intCast(cfg.num_key_value_heads),
        .head_dim = @intCast(cfg.head_dim),
        .vocab_size = @intCast(cfg.vocab_size),
    };

    var probe_writer: ?probe.JsonlWriter = null;
    defer if (probe_writer) |*w| w.close();
    var probe_bus: ?probe.Bus = null;
    defer if (probe_bus) |*b| b.deinit(gpa);
    var probe_hidden_scratch: ?[]f32 = null;
    defer if (probe_hidden_scratch) |s| gpa.free(s);
    var probe_attn_scratch: ?[]f32 = null;
    defer if (probe_attn_scratch) |s| gpa.free(s);
    var probe_token_index: u32 = 0;

    if (probe_path) |pp| {
        probe_writer = try probe.JsonlWriter.open(pp);
        const w_ptr: *probe.JsonlWriter = &probe_writer.?;

        try stdout.print("probe: writing trace to {s}\n", .{pp});

        const bos: u32 = cfg.bos_token_id orelse 1;
        try stdout.print("probe: computing null prior (BOS={d})...\n", .{bos});
        try rec.reset();
        try rec.begin();
        try aliases.recordHybridForwardStep(&rec, &sc, &state, &gm, cfg, &ks, 0, bos, max_pos, tq4_hooks, true);
        try rec.endAndSubmit();
        try sc.logits.readBack(&ctx, f32, logits);
        // KV/SSM slots at pos 0 are now dirty with the BOS forward.
        // The chat loop below starts at pos = 0 and overwrites both
        // on the first prompt token — clean handoff, no reset needed.

        var bus = probe.Bus{};
        const probes_buf = try gpa.alloc(probe.Probe, 3);
        const lp = try probe.LogitProbe.create(gpa, w_ptr, @intCast(cfg.vocab_size), logits);
        const ap = try probe.ActivationEntropyProbe.create(gpa, w_ptr);
        const kp = try probe.AttentionProbe.create(gpa, w_ptr);
        probes_buf[0] = lp.probe();
        probes_buf[1] = ap.probe();
        probes_buf[2] = kp.probe();
        bus.probes = probes_buf;
        bus.finalize();
        probe_bus = bus;

        try probe.writeHeader(w_ptr, probe_info, dir_path, &.{ "logits", "act", "attn" });

        if (bus.needs_hidden_pre or bus.needs_hidden_post) {
            probe_hidden_scratch = try gpa.alloc(f32, cfg.hidden_size);
        }
        if (bus.needs_attention) {
            probe_attn_scratch = try gpa.alloc(f32, cfg.num_attention_heads * max_pos);
        }
    }

    // ── Batch mode (hybrid path) ─────────────────────────────────────
    //
    // Same shape as runChat's batch path. Additional reset between
    // prompts: zero the per-layer SSM (conv + recurrent) state via
    // aliases.HybridChatState.reset, since Gated DeltaNet's recurrent state
    // is a constant-size vector that always matters regardless of
    // pos and would otherwise leak the previous prompt's trajectory
    // into the next prompt's first decode steps.
    if (batch_prompts) |entries| {
        const bos: u32 = cfg.bos_token_id orelse 1;
        try stdout.print("probe (batch): computing null prior (BOS={d})...\n", .{bos});
        const null_prior_buf = try gpa.alloc(f32, cfg.vocab_size);
        defer gpa.free(null_prior_buf);
        try rec.reset();
        try rec.begin();
        try aliases.recordHybridForwardStep(&rec, &sc, &state, &gm, cfg, &ks, 0, bos, max_pos, tq4_hooks, true);
        try rec.endAndSubmit();
        try sc.logits.readBack(&ctx, f32, null_prior_buf);

        const hidden_scratch = try gpa.alloc(f32, cfg.hidden_size);
        defer gpa.free(hidden_scratch);
        const attn_scratch = try gpa.alloc(f32, cfg.num_attention_heads * max_pos);
        defer gpa.free(attn_scratch);

        for (entries) |entry| {
            const out_path = try std.fmt.allocPrint(gpa, "{s}{s}.jsonl", .{ probe_prefix.?, entry.label });
            defer gpa.free(out_path);
            try stdout.print("\n[{s}] → {s}\n", .{ entry.label, out_path });

            // Reset SSM state to zero before this prompt's prefill.
            try state.reset(&ctx);

            var writer = try probe.JsonlWriter.open(out_path);
            defer writer.close();

            var bus = probe.Bus{};
            const probes_buf = try gpa.alloc(probe.Probe, 3);
            const lp = try probe.LogitProbe.create(gpa, &writer, @intCast(cfg.vocab_size), null_prior_buf);
            const ap = try probe.ActivationEntropyProbe.create(gpa, &writer);
            const kp = try probe.AttentionProbe.create(gpa, &writer);
            probes_buf[0] = lp.probe();
            probes_buf[1] = ap.probe();
            probes_buf[2] = kp.probe();
            bus.probes = probes_buf;
            bus.finalize();
            defer bus.deinit(gpa);

            try probe.writeHeader(&writer, probe_info, dir_path, &.{ "logits", "act", "attn" });

            var pos: usize = 0;
            var token_index: u32 = 0;

            try chatTurnHybrid(
                gpa, &ctx, &rec, &sc, &state, &gm, cfg, &ks, &tok, entry.text, &pos, logits,
                sample_scratch, sample_params, rng, tmpl, false, max_pos, tq4_hooks,
                &bus, probe_info, &token_index, hidden_scratch, attn_scratch,
                max_new,
                if (mtp_round) |*m| m else null,
            );
        }
        return;
    }

    var pos: usize = 0;

    if (single_msg) |m| {
        try chatTurnHybrid(
            gpa, &ctx, &rec, &sc, &state, &gm, cfg, &ks, &tok, m, &pos, logits,
            sample_scratch, sample_params, rng, tmpl, false, max_pos, tq4_hooks,
            if (probe_bus) |*b| b else null,
            probe_info,
            &probe_token_index,
            probe_hidden_scratch,
            probe_attn_scratch,
            max_new,
            if (mtp_round) |*mr| mr else null,
        );
        if (mtp_round) |*mr| try printMtpStats(stdout, mr);
        return;
    }

    try stdout.print("\n{s} (Ctrl-D to exit)\n", .{tmpl.banner()});
    const stdin = std.io.getStdIn().reader();
    var line_buf: [4096]u8 = undefined;
    while (true) {
        try stdout.print("\nuser> ", .{});
        const maybe_line = try stdin.readUntilDelimiterOrEof(&line_buf, '\n');
        const line = maybe_line orelse {
            try stdout.print("\n", .{});
            break;
        };
        if (line.len == 0) continue;
        try chatTurnHybrid(
            gpa, &ctx, &rec, &sc, &state, &gm, cfg, &ks, &tok, line, &pos, logits,
            sample_scratch, sample_params, rng, tmpl, true, max_pos, tq4_hooks,
            if (probe_bus) |*b| b else null,
            probe_info,
            &probe_token_index,
            probe_hidden_scratch,
            probe_attn_scratch,
            max_new,
            if (mtp_round) |*mr| mr else null,
        );
        if (pos >= max_pos - 64) {
            try stdout.print("\n[KV cache near capacity, ending session]\n", .{});
            break;
        }
    }
    if (mtp_round) |*mr| try printMtpStats(stdout, mr);
}

/// Print cumulative MTP-SpecDec statistics for a chat session.
fn printMtpStats(writer: anytype, mr: *const MtpRound) !void {
    if (mr.cumulative_rounds == 0) return;
    const accept_rate: f64 = @as(f64, @floatFromInt(mr.cumulative_accepted)) /
        @as(f64, @floatFromInt(mr.cumulative_drafts));
    const tokens_per_round: f64 = @as(f64, @floatFromInt(mr.cumulative_emitted)) /
        @as(f64, @floatFromInt(mr.cumulative_rounds));
    try writer.print(
        "[mtp: {d} rounds, k={d}, {d}/{d} drafts accepted ({d:.1}%), {d:.2} tok/round, {d} tokens via mtp]\n",
        .{
            mr.cumulative_rounds, mr.options.draft_n,
            mr.cumulative_accepted, mr.cumulative_drafts,
            accept_rate * 100.0,
            tokens_per_round,
            mr.cumulative_emitted,
        },
    );
}

fn chatTurnHybrid(
    gpa: std.mem.Allocator,
    ctx: *const vk.Context,
    rec: *gpu_recorder.Recorder,
    sc: *const aliases.HybridChatScratch,
    state: *const aliases.HybridChatState,
    gm: *const gpu_model.GpuModel,
    cfg: config_mod.Config,
    ks: *const aliases.HybridChatKernels,
    tok: *const tokenizer_mod.Tokenizer,
    user_msg: []const u8,
    pos: *usize,
    logits: []f32,
    sample_scratch: []f32,
    sample_params: cpu_forward.SampleParams,
    rng: std.Random,
    tmpl: chat_template_mod.ChatTemplate,
    is_repl: bool,
    max_pos: u32,
    tq4_v: ?aliases.HybridTq4VHooks,
    probe_bus: ?*probe.Bus,
    probe_info: probe.ModelInfo,
    probe_token_index: *u32,
    probe_hidden_scratch: ?[]f32,
    probe_attn_scratch: ?[]f32,
    max_new: usize,
    mtp_round: ?*MtpRound,
) !void {
    const stdout = std.io.getStdOut().writer();

    var prompt = std.ArrayList(u32).init(gpa);
    defer prompt.deinit();
    try tmpl.composePrompt(gpa, tok, user_msg, pos.* == 0, &prompt);

    if (!is_repl) {
        try stdout.print("\nprompt ({d} tokens, starting at pos {d}):\n  ", .{ prompt.items.len, pos.* });
        for (prompt.items) |id| try printTokenForDisplay(gpa, stdout, tok, id);
        try stdout.print("\n\nresponse: ", .{});
    } else {
        try stdout.print("model> ", .{});
    }

    const eos: ?u32 = cfg.eos_token_id;
    const eot: u32 = tmpl.end_of_turn;
    const max_response: usize = max_new;

    var current: u32 = prompt.items[0];
    var prompt_idx: usize = 0;
    var generated: usize = 0;

    // Time decode-only throughput: clock starts after prefill consumes
    // the prompt, so the number reflects steady-state generation speed.
    var t_decode_start: i128 = 0;
    var decode_started = false;

    while (true) {
        // Skip the LM head matmul (the largest in the model) on every
        // prefill token except the last — see chatTurn for the rationale.
        const is_last_prefill_or_decoding = (prompt_idx + 1 >= prompt.items.len);
        const is_prefill = (prompt_idx + 1 < prompt.items.len);

        if (probe_bus) |bus| try bus.onTokenStart(.{
            .info = probe_info,
            .token_index = probe_token_index.*,
            .pos = pos.*,
            .token_id = current,
            .is_prefill = is_prefill,
        });

        const use_probed_forward = if (probe_bus) |bus|
            bus.needs_hidden_pre or bus.needs_hidden_post or bus.needs_attention
        else
            false;

        if (use_probed_forward) {
            try forwardStepProbedHybrid(
                rec,
                ctx,
                sc,
                state,
                gm,
                cfg,
                ks,
                pos.*,
                current,
                max_pos,
                tq4_v,
                is_last_prefill_or_decoding,
                probe_bus.?,
                probe_info,
                probe_token_index.*,
                is_prefill,
                probe_hidden_scratch,
                probe_attn_scratch,
            );
        } else {
            if (pos.* > 0) try rec.reset();
            try rec.begin();
            try aliases.recordHybridForwardStep(rec, sc, state, gm, cfg, ks, pos.*, current, max_pos, tq4_v, is_last_prefill_or_decoding);
            try rec.endAndSubmit();
        }

        if (probe_bus) |bus| {
            if (is_last_prefill_or_decoding and bus.needs_logits) {
                try sc.logits.readBack(ctx, f32, logits);
                try bus.onLogits(.{
                    .info = probe_info,
                    .token_index = probe_token_index.*,
                    .pos = pos.*,
                    .token_id = current,
                    .is_prefill = is_prefill,
                }, logits);
            }
        }

        prompt_idx += 1;
        pos.* += 1;
        probe_token_index.* += 1;

        if (prompt_idx < prompt.items.len) {
            current = prompt.items[prompt_idx];
            continue;
        }

        if (!decode_started) {
            t_decode_start = std.time.nanoTimestamp();
            decode_started = true;
        }

        try sc.logits.readBack(ctx, f32, logits);
        const next = try cpu_forward.sample(logits, sample_params, rng, sample_scratch);

        if (probe_bus) |bus| try bus.onTokenEnd(.{
            .info = probe_info,
            .token_index = probe_token_index.*,
            .pos = pos.*,
            .token_id = current,
            .is_prefill = false,
        }, @intCast(next));

        if (next == eot) break;
        if (eos != null and next == eos.?) break;

        if (mtp_round) |mtp| {
            // ── MTP-SpecDec decode loop ─────────────────────────────
            // Hands off the decode phase to the SpecDec verify path:
            // each round runs an MTP draft chain (k drafts), one
            // batched verify forward (`recordForwardStepBatched(n_q=k)`),
            // and an anchor forward to set up the next round's KV +
            // main_tok. Emits up to k+1 tokens per round.
            try mtpDecodeRounds(
                gpa, ctx, rec, sc, state, gm, cfg, ks, tok, mtp,
                @intCast(next), pos, logits, sample_scratch, sample_params, rng,
                stdout, max_pos, max_response, &generated, eos, eot,
            );
            break;
        }

        try printTokenForDisplay(gpa, stdout, tok, @intCast(next));

        generated += 1;
        if (generated >= max_response) break;
        current = @intCast(next);
    }
    try stdout.print("\n", .{});

    if (decode_started and generated > 0) {
        const t_end = std.time.nanoTimestamp();
        const ms_total = @as(f64, @floatFromInt(t_end - t_decode_start)) / 1_000_000.0;
        const tokps = @as(f64, @floatFromInt(generated)) * 1000.0 / ms_total;
        try stdout.print("[{d} tok in {d:.0} ms, {d:.1} tok/s]\n", .{ generated, ms_total, tokps });
    }
}

/// MTP-SpecDec decode loop. Called once per chat turn, after the
/// prompt has been prefilled and the first main-token prediction
/// (`first_main_tok`) was sampled from the last prefill forward's
/// logits. `pos.*` is the position where this turn's first decoded
/// token will be placed.
///
/// Each round:
///   1. (round 0 only) snapshot `sc.stream` (= hidden at pos*-1 from
///      prefill's last forward) into `mtp.last_hidden_buf`.
///   2. Run `k` MTP draft steps (sequential, with read-back between —
///      each step's output token feeds the next).
///   3. Run one batched verify forward at pos_start = pos.* with the
///      drafts as input. Read back `[k, vocab]` logits.
///   4. Apply the SpecDec accept rule:
///        accept[0] = (drafts[0] == main_tok)
///        accept[i] = accept[i-1] AND (verify_argmax[i-1] == drafts[i])
///   5. Emit `accept_count` verified drafts + 1 correction/continuation.
///   6. Run one anchor forward at the last emitted token's position to
///      (a) overwrite KV[pos] with the correction (when accept_count<k)
///      or write a fresh KV[pos] for the continuation (when ==k), and
///      (b) produce the next round's `main_tok` + `last_hidden`.
///
/// Loop exits on EOS/EOT in any emitted token, or
/// `generated >= max_response`. KV state outside the verified prefix
/// may contain stale drafts; future writes overwrite naturally as long
/// as the next forward at pos X writes KV[X] before reading KV[X+1..].
fn mtpDecodeRounds(
    gpa: std.mem.Allocator,
    ctx: *const vk.Context,
    rec: *gpu_recorder.Recorder,
    sc: *const aliases.HybridChatScratch,
    state: *const aliases.HybridChatState,
    gm: *const gpu_model.GpuModel,
    cfg: config_mod.Config,
    ks: *const aliases.HybridChatKernels,
    tok: *const tokenizer_mod.Tokenizer,
    mtp: *MtpRound,
    first_main_tok: u32,
    pos: *usize,
    logits: []f32,
    sample_scratch: []f32,
    sample_params: cpu_forward.SampleParams,
    rng: std.Random,
    stdout: anytype,
    max_pos: u32,
    max_response: usize,
    generated: *usize,
    eos: ?u32,
    eot: u32,
) !void {
    const k_drafts: u32 = mtp.options.draft_n;
    const hidden_u: u32 = @intCast(cfg.hidden_size);
    const vocab: usize = cfg.vocab_size;

    // Round 0 setup: snapshot prefill's last hidden (sc.stream after
    // the last prefill forward = hidden at pos*-1) into last_hidden_buf.
    {
        try rec.reset();
        try rec.begin();
        const sc_h = runtime_hybrid.SliceCopyPush{ .src_off = 0, .dst_off = 0, .n_elem = hidden_u };
        try runtime.recDispatch1D(rec, &ks.slice_copy, &.{ &sc.stream, &mtp.last_hidden_buf }, &sc_h, hidden_u);
        try rec.endAndSubmit();
    }

    var main_tok: u32 = first_main_tok;
    var verify_pos: usize = pos.*;

    var drafts: [16]u32 = undefined;
    var verify_argmax: [16]u32 = undefined;

    // ── Pre-loop: emit `first_main_tok` ─────────────────────────────
    // It's the verified next token at pos `verify_pos` (= P), sampled
    // from prefill's last forward. We emit it once here and from this
    // point on the loop body emits the CORRECTION + ACCEPTED DRAFTS
    // and uses the correction-forward's logits to seed the next
    // round's main_tok.
    if (first_main_tok == eot or (eos != null and first_main_tok == eos.?)) return;
    try printTokenForDisplay(gpa, stdout, tok, first_main_tok);
    mtp.cumulative_emitted += 1;
    generated.* += 1;
    if (generated.* >= max_response) return;

    while (generated.* < max_response) {
        // Cap the round's verify range against max_pos — recording
        // past the cache capacity would silently truncate.
        if (verify_pos + k_drafts + 1 > max_pos) break;

        // ── 1. Snapshot SSM state (per-linear-layer ssm_{conv,rec}) ─
        // The corrected verify advances SSM by k+1 steps (1 anchor
        // main_tok + k drafts). We commit `accept_count + 2` (the
        // anchor + accepted drafts + correction/continuation). After
        // the accept rule we restore SSM and re-advance by exactly
        // those committed tokens via single-token forwards.
        try rec.reset();
        try rec.begin();
        try recordSnapshotSsm(rec, ks, state, &mtp.ssm_snapshot, cfg);
        try rec.endAndSubmit();

        // ── 2. MTP draft chain — k sequential steps with read-back ──
        // Slot 0 takes (last_hidden_at_(P-1), main_tok_at_P) and
        // predicts pos P+1 → d_0. Slot s (s≥1) feeds (slot_{s-1}'s
        // hidden, d_{s-1}) and predicts pos P+s+1.
        var prev_token = main_tok;
        var s: u32 = 0;
        while (s < k_drafts) : (s += 1) {
            const h_prev_buf = if (s == 0) &mtp.last_hidden_buf else &mtp.state.h_out;
            try rec.reset();
            try rec.begin();
            try runtime_mtp.recordMtpStep(
                rec, sc, gm, cfg, ks, &mtp.state,
                h_prev_buf, prev_token, s, max_pos, &mtp.draft_logits_buf,
            );
            try rec.endAndSubmit();
            try mtp.draft_logits_buf.readBack(ctx, f32, logits);
            const top1: usize = cpu_forward.argmax(logits);
            drafts[s] = @intCast(top1);
            prev_token = drafts[s];
        }

        // ── 3. Verify forward (batched, n_q = k+1) ──────────────────
        // Feed [main_tok, d_0, ..., d_{k-1}] at positions [P..P+k].
        // verify_argmax[i] is main's prediction at pos P+i+1 given
        // the prefix [main_tok, d_0, ..., d_{i-1}] in KV. Compare to
        // d_i to gate accept.
        var verify_input: [17]u32 = undefined;
        verify_input[0] = main_tok;
        var vi: u32 = 0;
        while (vi < k_drafts) : (vi += 1) verify_input[vi + 1] = drafts[vi];

        try rec.reset();
        try rec.begin();
        try runtime_hybrid.recordForwardStepBatched(
            rec, sc, state, gm, cfg, ks,
            verify_pos, verify_input[0 .. k_drafts + 1], max_pos, true,
        );
        try rec.endAndSubmit();
        try sc.logits.readBack(ctx, f32, mtp.verify_logits_host);

        // ── 4. Per-row argmax + accept rule ─────────────────────────
        // We have k+1 logit rows. Walk i = 0..k-1: if verify_argmax[i]
        // matches d_i, accept; else stop. The k-th row gives the
        // continuation regardless (used when all drafts accepted).
        var i_a: u32 = 0;
        while (i_a < k_drafts + 1) : (i_a += 1) {
            const row = mtp.verify_logits_host[i_a * vocab .. (i_a + 1) * vocab];
            verify_argmax[i_a] = @intCast(cpu_forward.argmax(row));
        }
        var accept_count: u32 = 0;
        while (accept_count < k_drafts) : (accept_count += 1) {
            if (verify_argmax[accept_count] != drafts[accept_count]) break;
        }
        mtp.cumulative_drafts += k_drafts;
        mtp.cumulative_accepted += accept_count;
        mtp.cumulative_rounds += 1;

        // The continuation/correction is always `verify_argmax[accept_count]`
        // — for partial accept it differs from d_{accept_count} (that's
        // why accept stopped); for full accept (accept_count == k) it's
        // the bonus prediction at pos P+k+1.
        const next_anchor_tok: u32 = verify_argmax[accept_count];

        // ── 5. Restore SSM to pre-verify state ──────────────────────
        try rec.reset();
        try rec.begin();
        try recordRestoreSsm(rec, ks, state, &mtp.ssm_snapshot, cfg);
        try rec.endAndSubmit();

        // ── 6. Re-advance SSM via (accept_count + 2) single-token
        // forwards using committed tokens: main_tok + accepted drafts
        // + correction/continuation. These also write the right KV
        // for full-attention layers (verify wrote drafts there; the
        // anchored re-forwards overwrite). The LAST forward provides
        // logits for next round's main_tok and `sc.stream` for the
        // next MTP feed.
        var commit_pos: usize = verify_pos;

        // Forward 1: main_tok at verify_pos. Always runs; main_tok
        // is verified (came from prior forward's logits). Emit was
        // done pre-loop or by previous round.
        try rec.reset();
        try rec.begin();
        try aliases.recordHybridForwardStep(rec, sc, state, gm, cfg, ks, commit_pos, main_tok, max_pos, null, false);
        try rec.endAndSubmit();
        commit_pos += 1;

        // Forwards 2..accept_count+1: accepted drafts at positions
        // verify_pos+1..verify_pos+accept_count. Emit each as it
        // commits.
        var done_emit = false;
        var i_commit: u32 = 0;
        while (i_commit < accept_count) : (i_commit += 1) {
            const tok_id = drafts[i_commit];
            if (tok_id == eot or (eos != null and tok_id == eos.?)) {
                done_emit = true;
                break;
            }
            try rec.reset();
            try rec.begin();
            try aliases.recordHybridForwardStep(rec, sc, state, gm, cfg, ks, commit_pos, tok_id, max_pos, null, false);
            try rec.endAndSubmit();
            commit_pos += 1;

            try printTokenForDisplay(gpa, stdout, tok, tok_id);
            mtp.cumulative_emitted += 1;
            generated.* += 1;
            if (generated.* >= max_response) {
                done_emit = true;
                break;
            }
        }
        if (done_emit) break;

        // EOS check on the correction/continuation before we run its
        // forward — avoids one wasted forward on the final round.
        if (next_anchor_tok == eot or (eos != null and next_anchor_tok == eos.?)) {
            break;
        }

        // Forward accept_count+2: correction/continuation at
        // verify_pos + accept_count + 1. compute_logits=true to
        // seed next round's main_tok.
        try rec.reset();
        try rec.begin();
        try aliases.recordHybridForwardStep(rec, sc, state, gm, cfg, ks, commit_pos, next_anchor_tok, max_pos, null, true);
        // Snapshot sc.stream → last_hidden_buf for next round's
        // MTP slot 0 input.
        const sc_h = runtime_hybrid.SliceCopyPush{ .src_off = 0, .dst_off = 0, .n_elem = hidden_u };
        try runtime.recDispatch1D(rec, &ks.slice_copy, &.{ &sc.stream, &mtp.last_hidden_buf }, &sc_h, hidden_u);
        try rec.endAndSubmit();

        try printTokenForDisplay(gpa, stdout, tok, next_anchor_tok);
        mtp.cumulative_emitted += 1;
        generated.* += 1;
        if (generated.* >= max_response) break;

        // Sample next round's main_tok from the correction's logits.
        // It's the predicted token at position commit_pos + 1, which
        // becomes the next round's verify-anchor (placed at
        // verify_pos_(R+1) = commit_pos + 1). Emit it now — it's the
        // next token in the user-visible output stream, conceptually
        // the "second part" of round R's emission (round R-1 emitted
        // its own main_tok via the pre-loop step or the prior round's
        // emit-on-sample). Emitting here keeps every position in the
        // generated stream printed exactly once.
        try sc.logits.readBack(ctx, f32, logits);
        const next_main = try cpu_forward.sample(logits, sample_params, rng, sample_scratch);
        main_tok = @intCast(next_main);

        if (main_tok == eot or (eos != null and main_tok == eos.?)) break;
        try printTokenForDisplay(gpa, stdout, tok, main_tok);
        mtp.cumulative_emitted += 1;
        generated.* += 1;
        if (generated.* >= max_response) break;

        // ── 7. Advance position counters ────────────────────────────
        // commit_pos is the position of the just-anchored correction.
        // main_tok occupies pos commit_pos + 1; the next round's
        // verify writes starting there.
        pos.* = commit_pos + 1;
        verify_pos = pos.*;
    }
}
