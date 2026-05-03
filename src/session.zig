//! Session — frame-budgeted cooperative-inference state machine.
//!
//! This is the "default state machine" for hosts that want valkyr to
//! generate text inside their render loop without writing the
//! per-layer scheduler themselves. Three calls cover the typical
//! integration:
//!
//! ```
//! var sess = try Session.init(allocator, ctx, gm, tok, .{});
//! try sess.appendPrompt("Once upon a time");
//! // per frame, inside the host's aiDispatch:
//! const r = try sess.tickFrame(&rec);
//! if (r.new_token) |t| std.debug.print("{s}", .{tok.decode(t)});
//! ```
//!
//! Hosts that want bespoke orchestration (custom samplers, multi-NPC
//! schedulers, per-layer visualization taps that need the recorder
//! directly) can still drive the runtime primitives in
//! `runtime.zig` — Session is built on top of them, not in place of
//! them.
//!
//! Frame-budget contract
//! ─────────────────────
//! `Config.budget_layers` is the per-tickFrame work cap. Each
//! recordOneLayer call counts as one unit. The recorded sample step
//! also counts as one. Embeddings are free (single dispatch, sub-µs
//! at typical sizes). The state machine carries (in_flight,
//! fwd_layer, sample_pending) across frames so partial forwards
//! resume on the next tick exactly where they left off.
//!
//! Sampling cadence
//! ────────────────
//! `recordSampleStep` records the final-norm + LM-head matmul into
//! the recorder, then a `vkCmdCopyBuffer` mirrors `scratch.logits`
//! into a HOST_VISIBLE buffer. The host's frame submit + fence cycle
//! makes that mirror visible to the CPU at the top of the NEXT tick.
//! `tickFrame` consumes the deferred sample, appends the token,
//! advances `pos`, and proceeds to record the next forward step.
//! Net cost: zero extra submits per token, one ~512 KB buffer copy
//! at sample time.
//!
//! State machine variables
//! ───────────────────────
//!   phase            .idle | .prompt | .decode | .done
//!   pos              KV cache write position (= number of tokens
//!                    already KV-cached)
//!   fwd_in_flight    a forward step is partially recorded
//!   fwd_layer        next layer to record (0..n_layers; == n_layers
//!                    means layers are done, sample step or pos++ next)
//!   sample_pending   recorded sample step this frame; consume next
//!                    tick after the host's fence signals
//!   prompt_q         tokens to ingest before switching to .decode
//!   generated        emitted tokens in order
//!
//! See `project_session_api.md` in `~/.claude/.../memory/` for the
//! full architectural rationale and the chunking that built up to
//! this file.

const std = @import("std");
const vk = @import("gpu/vk.zig");
const buffer = @import("gpu/buffer.zig");
const recorder_mod = @import("gpu/recorder.zig");
const gpu_model = @import("gpu/model.zig");
const gpu_scratch = @import("gpu/scratch.zig");
const config_mod = @import("config.zig");
const tokenizer_mod = @import("tokenizer.zig");
const runtime = @import("runtime.zig");
const runtime_hybrid = @import("runtime_hybrid.zig");

pub const Phase = enum { idle, prompt, decode, done };

pub const SamplerKind = union(enum) {
    /// Greedy argmax. Matches valkyr's CLI sampler bit-for-bit.
    greedy,
    // temperature, top_k, top_p, ... can be added later. Keeping the
    // union shape here so the API doesn't need to break when they land.
};

/// Fired once per emitted token, with the token id and the
/// display-ready UTF-8 bytes. The decoded slice is owned by Session
/// (allocated per-call by `tokenizer.decodeForDisplay`) and is only
/// valid for the duration of the callback — copy it if you need it
/// later. SentencePiece `▁` and byte-level `Ġ` markers are already
/// resolved to spaces; byte-fallback `<0xBB>` tokens to their
/// literal byte values.
pub const TokenCallback = *const fn (
    user: ?*anyopaque,
    tok_id: u32,
    decoded: []const u8,
) void;

/// What sort of layer just got recorded. Dense models are always
/// `.full_attention`; hybrid (Qwen3.5 / Qwen3.6) interleaves
/// `.linear_attention` (Gated DeltaNet, no attention scores) and
/// `.full_attention` (every 4th layer in the typical 3+1 schedule).
pub const LayerKind = enum { full_attention, linear_attention };

/// Snapshot of the model's GPU state right after `recordOneLayer`.
/// Stable across dense and hybrid backends — hosts that visualize
/// attention don't need to know which one is running. Linear-attn
/// layers leave `scores == null` (those frames carry SSM state
/// instead of attention; future tap fields will expose it for hosts
/// that want to viz that side too).
pub const LayerTap = struct {
    rec: *recorder_mod.Recorder,
    layer_idx: u32,
    layer_kind: LayerKind,
    /// Post-softmax attention scores buffer, layout
    /// `[n_q_heads, max_pos]` f32 row-major. Each head's row is
    /// valid for positions `[0..n_pos]`; the rest is stale from
    /// previous tokens. `null` on `.linear_attention` layers
    /// (those use SSM state instead — Gated DeltaNet).
    scores: ?*const buffer.Buffer,
    /// Number of query heads — the row count of `scores`. Stays
    /// the value passed in even when `scores == null`, so hosts
    /// can size mirrors at init time off the model config.
    n_q_heads: u32,
    /// Stride between rows in `scores`. = `Session.max_pos`.
    max_pos: u32,
};

/// Optional viz / mech-interp hook. Fires DURING recording, after
/// `recordOneLayer` but before the next layer or the sample step.
/// The callback may record its own commands into the recorder
/// (e.g. a `vkCmdCopyBuffer` of `tap.scores.?` into a host-owned
/// SSBO mirror) — those land in the same per-frame command buffer
/// the host submits.
///
/// Both dense and hybrid backends fire this. On hybrid models only
/// 1-in-4 invocations carries non-null `tap.scores` (the
/// full-attention layers); the other 3-in-4 are linear-attention
/// layers with no attention to mirror, so `tap.scores == null` —
/// hosts should early-return on those.
pub const LayerCallback = *const fn (
    user: ?*anyopaque,
    tap: *const LayerTap,
) anyerror!void;

pub const Config = struct {
    /// Forward-step layers per tickFrame. Default = 8: tiny models
    /// (Llama 3.2 1B at 16 layers) decode at 2 frames/token; Gemma 2B
    /// at 18 layers also fits in 2-3 frames per token. Hosts with
    /// strict frame budgets can drop this; hosts that want max
    /// throughput can raise it.
    budget_layers: u32 = 8,

    /// Maximum tokens to emit before stopping.
    max_new_tokens: u32 = 256,

    /// Token IDs that terminate generation when emitted (EOS, etc.).
    /// Empty = generate until max_new_tokens.
    stop_tokens: []const u32 = &.{},

    sampler: SamplerKind = .greedy,

    /// Fires on every emitted token. Optional.
    on_token: ?TokenCallback = null,
    on_token_user: ?*anyopaque = null,

    /// Fires after each recorded layer. Optional. Used by chunk 7d
    /// (and the broader `project_engine_visualizations.md`) to
    /// pipe attention scores out to a host-owned SSBO without the
    /// host needing to touch the per-layer scheduler.
    on_layer: ?LayerCallback = null,
    on_layer_user: ?*anyopaque = null,

    /// KV cache + scratch sizing. The Session allocates its own
    /// scratch + KV buffers sized for this many positions. Long
    /// chats need more; tiny demos can drop to 256 to save VRAM.
    max_pos: usize = 1024,

    /// Whether to prepend the tokenizer's `<bos>` (or family
    /// equivalent) when appending the FIRST prompt. Subsequent
    /// appendPrompt calls always treat their input as continuation.
    /// Default true matches Gemma + Llama family expectations.
    prepend_bos: bool = true,
};

pub const TickResult = struct {
    /// Token sampled in this tickFrame, if any. At most one per call —
    /// the budget loop breaks after recording a sample step so the
    /// fence cycle can resolve before the next sample.
    new_token: ?u32,
    /// Layer-record units consumed this frame (recordOneLayer +
    /// recordSampleStep contribute; recordEmbedding does not).
    layers_done: u32,
    phase: Phase,
};

/// Per-family backend. Dense (Llama / Gemma / Qwen3 dense) holds a
/// `runtime.ChatKernels` + `GpuScratch` + `GpuKvCache`; hybrid
/// (Qwen3.5 family) holds `runtime_hybrid.ChatKernels` + `Scratch` +
/// `State`. Session.init picks based on `cfg.family.isHybrid()`.
pub const Backend = union(enum) {
    dense: struct {
        kernels: runtime.ChatKernels,
        scratch: gpu_scratch.GpuScratch,
        kv: gpu_scratch.GpuKvCache,
    },
    hybrid: struct {
        kernels: runtime_hybrid.ChatKernels,
        scratch: runtime_hybrid.Scratch,
        state: runtime_hybrid.State,
    },

    /// Both backends store logits in `scratch.logits` (same shape: a
    /// device-only `vocab × f32` buffer). This accessor papers over
    /// the union so the sample-step copy can stay backend-agnostic.
    pub fn logitsBuffer(self: *const Backend) *const buffer.Buffer {
        return switch (self.*) {
            .dense => |*b| &b.scratch.logits,
            .hybrid => |*b| &b.scratch.logits,
        };
    }
};

pub const Session = struct {
    allocator: std.mem.Allocator,
    ctx: *const vk.Context,
    gm: *const gpu_model.GpuModel,
    tokenizer: *const tokenizer_mod.Tokenizer,
    cfg_model: config_mod.Config,
    cfg: Config,

    backend: Backend,
    max_pos: u32,

    /// Host-visible mirror of `scratch.logits`. The recorder appends a
    /// `vkCmdCopyBuffer` after the sample step; once the host's frame
    /// fence signals, this buffer's persistent map shows the new
    /// values. CPU samples directly out of the map — zero extra
    /// submits per token.
    logits_mirror: buffer.Buffer,

    // ── State machine ────────────────────────────────────────────
    phase: Phase,
    pos: u32,
    fwd_in_flight: bool,
    fwd_layer: u32,
    sample_pending: bool,
    bos_consumed: bool,

    prompt_q: std.ArrayList(u32),
    generated: std.ArrayList(u32),

    pub fn init(
        allocator: std.mem.Allocator,
        ctx: *const vk.Context,
        gm: *const gpu_model.GpuModel,
        tokenizer: *const tokenizer_mod.Tokenizer,
        cfg: Config,
    ) !Session {
        const cfg_model = gm.config;
        const max_pos: u32 = @intCast(cfg.max_pos);

        var backend: Backend = if (cfg_model.family.isHybrid()) blk: {
            var kernels = try runtime_hybrid.ChatKernels.init(ctx, gm.precision);
            errdefer kernels.deinit();
            var scratch = try runtime_hybrid.Scratch.init(ctx, cfg_model, max_pos, false);
            errdefer scratch.deinit(ctx.device);
            var state = try runtime_hybrid.State.init(allocator, ctx, cfg_model, max_pos, false);
            errdefer state.deinit(ctx.device);
            break :blk .{ .hybrid = .{
                .kernels = kernels,
                .scratch = scratch,
                .state = state,
            } };
        } else blk: {
            var kernels = try runtime.ChatKernels.init(ctx, gm.precision, cfg_model.family);
            errdefer kernels.deinit();
            var scratch = try gpu_scratch.GpuScratch.init(ctx, cfg_model, cfg.max_pos);
            errdefer scratch.deinit(ctx.device);
            var kv = try gpu_scratch.GpuKvCache.init(allocator, ctx, cfg_model, cfg.max_pos);
            errdefer kv.deinit(ctx.device);
            break :blk .{ .dense = .{
                .kernels = kernels,
                .scratch = scratch,
                .kv = kv,
            } };
        };
        errdefer switch (backend) {
            .dense => |*b| {
                var k = b.kernels;
                k.deinit();
                b.scratch.deinit(ctx.device);
                b.kv.deinit(ctx.device);
            },
            .hybrid => |*b| {
                var k = b.kernels;
                k.deinit();
                b.scratch.deinit(ctx.device);
                b.state.deinit(ctx.device);
            },
        };

        // HOST_VISIBLE+HOST_COHERENT persistent-mapped, sized to the
        // logits vector. TRANSFER_DST_BIT — vkCmdCopyBuffer's dst.
        // ~1 MB (Gemma vocab × 4) of host-visible VRAM, invisible
        // alongside the model itself.
        const mirror_bytes = cfg_model.vocab_size * @sizeOf(f32);
        var logits_mirror = try createHostMirrorBuffer(ctx, mirror_bytes);
        errdefer logits_mirror.deinit(ctx.device);

        return .{
            .allocator = allocator,
            .ctx = ctx,
            .gm = gm,
            .tokenizer = tokenizer,
            .cfg_model = cfg_model,
            .cfg = cfg,
            .backend = backend,
            .max_pos = max_pos,
            .logits_mirror = logits_mirror,
            .phase = .idle,
            .pos = 0,
            .fwd_in_flight = false,
            .fwd_layer = 0,
            .sample_pending = false,
            .bos_consumed = false,
            .prompt_q = std.ArrayList(u32).init(allocator),
            .generated = std.ArrayList(u32).init(allocator),
        };
    }

    pub fn deinit(self: *Session) void {
        self.prompt_q.deinit();
        self.generated.deinit();
        self.logits_mirror.deinit(self.ctx.device);
        switch (self.backend) {
            .dense => |*b| {
                b.kv.deinit(self.ctx.device);
                b.scratch.deinit(self.ctx.device);
                b.kernels.deinit();
            },
            .hybrid => |*b| {
                b.state.deinit(self.ctx.device);
                b.scratch.deinit(self.ctx.device);
                b.kernels.deinit();
            },
        }
    }

    /// Tokenize `text` and queue it for prefill. Can be called
    /// multiple times — additional prompt chunks append to the queue.
    /// On the very first call, prepends a `<bos>` token if the
    /// tokenizer exposes one and `cfg.prepend_bos` is set (default).
    pub fn appendPrompt(self: *Session, text: []const u8) !void {
        if (self.cfg.prepend_bos and !self.bos_consumed) {
            if (self.tokenizer.specialTokenId("<bos>")) |bos| {
                try self.prompt_q.append(bos);
            }
            self.bos_consumed = true;
        }
        const encoded = try self.tokenizer.encode(self.allocator, text);
        defer self.allocator.free(encoded);
        try self.prompt_q.appendSlice(encoded);
        if (self.phase == .idle) self.phase = .prompt;
    }

    pub fn isDone(self: *const Session) bool {
        return self.phase == .done;
    }

    pub fn generatedTokens(self: *const Session) []const u32 {
        return self.generated.items;
    }

    /// Advance the cooperative-inference state machine by up to
    /// `cfg.budget_layers` units of work, recording dispatches into
    /// `rec`. Returns whatever happened this frame. Caller submits
    /// the recorder as part of its frame submit — the Session does
    /// not submit on its own.
    pub fn tickFrame(self: *Session, rec: *recorder_mod.Recorder) !TickResult {
        var emitted: ?u32 = null;
        var work: u32 = 0;

        // ── 1. Consume any deferred sample from the previous frame. ──
        // The host's fence-wait at the top of drawFrame just completed,
        // so logits_mirror's persistent map contains the new values.
        if (self.sample_pending) {
            const tok = self.sample();
            try self.generated.append(tok);
            emitted = tok;
            if (self.cfg.on_token) |cb| {
                // `decodeForDisplay` allocates, but only on tokens we
                // actually emit (one per ~hundreds of dispatches), so
                // the cost is invisible. Frees right after the
                // callback returns — the host's job is just to
                // print/forward the bytes synchronously.
                const display = self.tokenizer.decodeForDisplay(self.allocator, tok) catch &[_]u8{};
                defer if (display.len > 0) self.allocator.free(display);
                cb(self.cfg.on_token_user, tok, display);
            }
            self.sample_pending = false;
            self.fwd_in_flight = false;
            self.fwd_layer = 0;
            self.pos += 1;
            self.phase = .decode;
            if (self.isStopToken(tok) or
                self.generated.items.len >= self.cfg.max_new_tokens or
                self.pos >= @as(u32, @intCast(self.cfg.max_pos)))
            {
                self.phase = .done;
                return .{ .new_token = emitted, .layers_done = 0, .phase = self.phase };
            }
        }

        // ── 2. Record up to budget_layers of work. ───────────────────
        while (work < self.cfg.budget_layers and self.phase != .done) {
            switch (self.backend) {
                .dense => |*b| {
                    if (!self.fwd_in_flight) {
                        const next_tok = self.pickNextInputToken() orelse break;
                        try runtime.recordEmbedding(
                            rec,
                            &b.scratch,
                            self.gm,
                            self.cfg_model,
                            &b.kernels,
                            next_tok,
                        );
                        self.fwd_in_flight = true;
                        self.fwd_layer = 0;
                        // Embedding is one shader dispatch (~µs); not
                        // charged against the layer budget.
                        continue;
                    }

                    if (self.fwd_layer < self.cfg_model.num_hidden_layers) {
                        const pushes = runtime.computeForwardPushes(
                            self.cfg_model,
                            &b.scratch,
                            self.pos,
                        );
                        try runtime.recordOneLayer(
                            rec,
                            &b.scratch,
                            self.gm,
                            &b.kv,
                            self.cfg_model,
                            &b.kernels,
                            self.fwd_layer,
                            self.pos,
                            &pushes,
                            null,
                        );
                        if (self.cfg.on_layer) |cb| {
                            const tap = LayerTap{
                                .rec = rec,
                                .layer_idx = self.fwd_layer,
                                .layer_kind = .full_attention,
                                .scores = &b.scratch.scores,
                                .n_q_heads = @intCast(self.cfg_model.num_attention_heads),
                                .max_pos = self.max_pos,
                            };
                            try cb(self.cfg.on_layer_user, &tap);
                        }
                        self.fwd_layer += 1;
                        work += 1;
                        continue;
                    }

                    if (self.prompt_q.items.len > 0) {
                        self.pos += 1;
                        self.fwd_in_flight = false;
                        self.fwd_layer = 0;
                        continue;
                    }

                    const pushes = runtime.computeForwardPushes(
                        self.cfg_model,
                        &b.scratch,
                        self.pos,
                    );
                    try runtime.recordSampleStep(
                        rec,
                        &b.scratch,
                        self.gm,
                        self.cfg_model,
                        &b.kernels,
                        &pushes,
                    );
                    recordCopyToHostMirror(rec, &b.scratch.logits, &self.logits_mirror);
                    self.sample_pending = true;
                    work += 1;
                    break;
                },
                .hybrid => |*b| {
                    if (!self.fwd_in_flight) {
                        const next_tok = self.pickNextInputToken() orelse break;
                        const hidden: u32 = @intCast(self.cfg_model.hidden_size);
                        const embed_push = runtime_hybrid.EmbedLookupPush{
                            .token_id = next_tok,
                            .dim = hidden,
                            .scale = if (self.cfg_model.family.embedScalesByDim()) @sqrt(@as(f32, @floatFromInt(hidden))) else 1.0,
                        };
                        try runtime.recDispatch1D(
                            rec,
                            &b.kernels.embed,
                            &.{ &self.gm.embed_tokens, &b.scratch.stream },
                            &embed_push,
                            hidden,
                        );
                        self.fwd_in_flight = true;
                        self.fwd_layer = 0;
                        continue;
                    }

                    if (self.fwd_layer < self.cfg_model.num_hidden_layers) {
                        const pushes = runtime_hybrid.computeForwardPushes(
                            self.cfg_model,
                            self.pos,
                            self.max_pos,
                        );
                        try runtime_hybrid.recordOneLayer(
                            rec,
                            &b.scratch,
                            &b.state,
                            self.gm,
                            self.cfg_model,
                            &b.kernels,
                            self.fwd_layer,
                            self.pos,
                            &pushes,
                            null,
                        );
                        if (self.cfg.on_layer) |cb| {
                            // Branch on the per-layer schedule: full-attn
                            // layers populated `scratch.scores` exactly
                            // like dense; linear-attn layers ran the
                            // Gated DeltaNet path which doesn't touch
                            // scores, so we tell the host explicitly via
                            // the optional buffer pointer.
                            const lt = self.cfg_model.layer_types[self.fwd_layer];
                            const kind: LayerKind = switch (lt) {
                                .full_attention => .full_attention,
                                .linear_attention => .linear_attention,
                            };
                            const scores_buf: ?*const buffer.Buffer = switch (lt) {
                                .full_attention => &b.scratch.scores,
                                .linear_attention => null,
                            };
                            const tap = LayerTap{
                                .rec = rec,
                                .layer_idx = self.fwd_layer,
                                .layer_kind = kind,
                                .scores = scores_buf,
                                .n_q_heads = @intCast(self.cfg_model.num_attention_heads),
                                .max_pos = self.max_pos,
                            };
                            try cb(self.cfg.on_layer_user, &tap);
                        }
                        self.fwd_layer += 1;
                        work += 1;
                        continue;
                    }

                    if (self.prompt_q.items.len > 0) {
                        self.pos += 1;
                        self.fwd_in_flight = false;
                        self.fwd_layer = 0;
                        continue;
                    }

                    const pushes = runtime_hybrid.computeForwardPushes(
                        self.cfg_model,
                        self.pos,
                        self.max_pos,
                    );
                    const hidden: u32 = @intCast(self.cfg_model.hidden_size);
                    const vocab: u32 = @intCast(self.cfg_model.vocab_size);
                    try runtime.recDispatchPerRow(
                        rec,
                        &b.kernels.rmsnorm,
                        &.{ &b.scratch.stream, &self.gm.final_norm, &b.scratch.final_norm_out },
                        &pushes.rms_push,
                        1,
                    );
                    try runtime.recDispatchMatmul(
                        rec,
                        &b.kernels.matmul_lm_head,
                        &.{ &b.scratch.final_norm_out, &self.gm.lm_head, &b.scratch.logits },
                        1,
                        vocab,
                        hidden,
                    );
                    recordCopyToHostMirror(rec, &b.scratch.logits, &self.logits_mirror);
                    self.sample_pending = true;
                    work += 1;
                    break;
                },
            }
        }

        return .{ .new_token = emitted, .layers_done = work, .phase = self.phase };
    }

    fn pickNextInputToken(self: *Session) ?u32 {
        if (self.prompt_q.items.len > 0) {
            const tok = self.prompt_q.items[0];
            // ArrayList lacks popFront; orderedRemove(0) is O(n) but
            // prompts are typically tens of tokens. If long-prompt
            // perf bites, swap to a ring buffer.
            _ = self.prompt_q.orderedRemove(0);
            self.phase = .prompt;
            return tok;
        }
        if (self.generated.items.len > 0) {
            self.phase = .decode;
            return self.generated.items[self.generated.items.len - 1];
        }
        return null;
    }

    fn sample(self: *Session) u32 {
        // The persistent map is a raw byte pointer; reinterpret as a
        // logits-vector slice. HOST_COHERENT memory + the renderer's
        // post-AI barrier (chunk 4) make the new values visible
        // automatically once the fence signals.
        const ptr: [*]const f32 = @ptrCast(@alignCast(self.logits_mirror.mapped.?));
        const view = ptr[0..self.cfg_model.vocab_size];
        return switch (self.cfg.sampler) {
            .greedy => runtime.sampleArgmax(view),
        };
    }

    fn isStopToken(self: *const Session, tok: u32) bool {
        for (self.cfg.stop_tokens) |s| if (s == tok) return true;
        return false;
    }
};

// ── Helpers ───────────────────────────────────────────────────────

/// HOST_VISIBLE+HOST_COHERENT VkBuffer with TRANSFER_DST_BIT enabled,
/// persistent-mapped. Used by Session for its logits mirror, and
/// publicly available so hosts can allocate their own mirrors for
/// the on_layer callback (chunk 7d's attention-strip viz uses one).
/// `Buffer.initDynamic` is close to what we need but only enables
/// STORAGE_BUFFER_BIT — vkCmdCopyBuffer's dst needs TRANSFER_DST.
pub fn createHostMirrorBuffer(ctx: *const vk.Context, bytes: usize) !buffer.Buffer {
    const c = vk.c;
    var bci = std.mem.zeroes(c.VkBufferCreateInfo);
    bci.sType = c.VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bci.size = @intCast(bytes);
    bci.usage = c.VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    bci.sharingMode = c.VK_SHARING_MODE_EXCLUSIVE;
    var handle: c.VkBuffer = null;
    try vk.check(c.vkCreateBuffer(ctx.device, &bci, null, &handle));
    errdefer c.vkDestroyBuffer(ctx.device, handle, null);

    var req: c.VkMemoryRequirements = undefined;
    c.vkGetBufferMemoryRequirements(ctx.device, handle, &req);

    var memory: c.VkDeviceMemory = null;
    try allocateAndBind(ctx, handle, req, &memory,
        c.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | c.VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    errdefer c.vkFreeMemory(ctx.device, memory, null);

    var mapped: ?*anyopaque = null;
    try vk.check(c.vkMapMemory(ctx.device, memory, 0, req.size, 0, &mapped));

    return .{
        .handle = handle,
        .memory = memory,
        .bytes = bytes,
        .mode = .dynamic,
        .mapped = mapped,
    };
}

fn allocateAndBind(
    ctx: *const vk.Context,
    handle: vk.c.VkBuffer,
    req: vk.c.VkMemoryRequirements,
    out_mem: *vk.c.VkDeviceMemory,
    properties: u32,
) !void {
    const c = vk.c;
    var props: c.VkPhysicalDeviceMemoryProperties = undefined;
    c.vkGetPhysicalDeviceMemoryProperties(ctx.physical_device, &props);
    var i: u32 = 0;
    var idx: ?u32 = null;
    while (i < props.memoryTypeCount) : (i += 1) {
        if ((req.memoryTypeBits & (@as(u32, 1) << @intCast(i))) != 0 and
            (props.memoryTypes[i].propertyFlags & properties) == properties)
        {
            idx = i;
            break;
        }
    }
    const type_index = idx orelse return error.NoSuitableMemoryType;

    var mai = std.mem.zeroes(c.VkMemoryAllocateInfo);
    mai.sType = c.VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    mai.allocationSize = req.size;
    mai.memoryTypeIndex = type_index;
    try vk.check(c.vkAllocateMemory(ctx.device, &mai, null, out_mem));
    try vk.check(c.vkBindBufferMemory(ctx.device, handle, out_mem.*, 0));
}

/// Insert (1) a memory barrier covering SHADER_WRITE → TRANSFER_READ so
/// the LM-head matmul's writes to `src` are visible to the copy, then
/// (2) a `vkCmdCopyBuffer` from `src` to `dst`. The renderer's post-AI
/// barrier (chunk 4) covers TRANSFER_WRITE → HOST_READ for `dst`.
fn recordCopyToHostMirror(
    rec: *recorder_mod.Recorder,
    src: *const buffer.Buffer,
    dst: *const buffer.Buffer,
) void {
    const c = vk.c;
    var bar = std.mem.zeroes(c.VkMemoryBarrier);
    bar.sType = c.VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    bar.srcAccessMask = c.VK_ACCESS_SHADER_WRITE_BIT;
    bar.dstAccessMask = c.VK_ACCESS_TRANSFER_READ_BIT;
    c.vkCmdPipelineBarrier(
        rec.cmd,
        c.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        c.VK_PIPELINE_STAGE_TRANSFER_BIT,
        0,
        1,
        &bar,
        0,
        null,
        0,
        null,
    );

    const region = c.VkBufferCopy{
        .srcOffset = 0,
        .dstOffset = 0,
        .size = @intCast(src.bytes),
    };
    c.vkCmdCopyBuffer(rec.cmd, src.handle, dst.handle, 1, &region);
}
