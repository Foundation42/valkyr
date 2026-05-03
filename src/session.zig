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

pub const Phase = enum { idle, prompt, decode, done };

pub const SamplerKind = union(enum) {
    /// Greedy argmax. Matches valkyr's CLI sampler bit-for-bit.
    greedy,
    // temperature, top_k, top_p, ... can be added later. Keeping the
    // union shape here so the API doesn't need to break when they land.
};

/// Fired once per emitted token, with the token id and (best-effort)
/// decoded UTF-8. The decoded slice lives in the tokenizer's vocab
/// table — borrow only; do not free.
pub const TokenCallback = *const fn (
    user: ?*anyopaque,
    tok_id: u32,
    decoded: []const u8,
) void;

/// Optional viz / mech-interp hook. Fires DURING recording, after
/// `recordOneLayer` but before the next layer or the sample step.
/// The callback may record its own commands into the same recorder
/// (e.g. a `vkCmdCopyBuffer` of `scratch.scores` into a host-owned
/// SSBO mirror).
///
/// Unwrap pointers and offsets carefully: `scratch.scores` is laid
/// out as `[n_heads, max_pos]` row-major — see `gpu/scratch.zig`.
/// Each row currently holds the post-softmax attention distribution
/// for the layer's heads against KV positions [0..n_pos].
pub const LayerCallback = *const fn (
    user: ?*anyopaque,
    rec: *recorder_mod.Recorder,
    layer_idx: u32,
    scratch: *const gpu_scratch.GpuScratch,
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

pub const Session = struct {
    allocator: std.mem.Allocator,
    ctx: *const vk.Context,
    gm: *const gpu_model.GpuModel,
    tokenizer: *const tokenizer_mod.Tokenizer,
    cfg_model: config_mod.Config,
    cfg: Config,

    forward: runtime.Forward,
    scratch: gpu_scratch.GpuScratch,
    kv: gpu_scratch.GpuKvCache,
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

        var forward = try runtime.Forward.init(ctx, gm);
        errdefer forward.deinit();

        var scratch = try gpu_scratch.GpuScratch.init(ctx, cfg_model, cfg.max_pos);
        errdefer scratch.deinit(ctx.device);

        var kv = try gpu_scratch.GpuKvCache.init(allocator, ctx, cfg_model, cfg.max_pos);
        errdefer kv.deinit(ctx.device);

        // HOST_VISIBLE+HOST_COHERENT persistent-mapped, sized to the
        // logits vector. TRANSFER_DST_BIT comes free with initDynamic
        // is NOT true — `initDynamic` only enables STORAGE_BUFFER_BIT,
        // and we need TRANSFER_DST_BIT for vkCmdCopyBuffer's dst.
        // Roll a small custom buffer here. The cost is ~1 MB
        // (Gemma vocab × 4) of host-visible VRAM — invisible alongside
        // the model itself.
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
            .forward = forward,
            .scratch = scratch,
            .kv = kv,
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
        self.kv.deinit(self.ctx.device);
        self.scratch.deinit(self.ctx.device);
        self.forward.deinit();
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
            if (self.cfg.on_token) |cb| cb(self.cfg.on_token_user, tok, self.tokenizer.decode(tok) orelse "");
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
            if (!self.fwd_in_flight) {
                const next_tok = self.pickNextInputToken() orelse break;
                try runtime.recordEmbedding(
                    rec,
                    &self.scratch,
                    self.gm,
                    self.cfg_model,
                    &self.forward.kernels,
                    next_tok,
                );
                self.fwd_in_flight = true;
                self.fwd_layer = 0;
                // Embedding is one shader dispatch (~µs); not charged
                // against the layer budget.
                continue;
            }

            if (self.fwd_layer < self.cfg_model.num_hidden_layers) {
                const pushes = runtime.computeForwardPushes(
                    self.cfg_model,
                    &self.scratch,
                    self.pos,
                );
                try runtime.recordOneLayer(
                    rec,
                    &self.scratch,
                    self.gm,
                    &self.kv,
                    self.cfg_model,
                    &self.forward.kernels,
                    self.fwd_layer,
                    &pushes,
                );
                if (self.cfg.on_layer) |cb| {
                    try cb(self.cfg.on_layer_user, rec, self.fwd_layer, &self.scratch);
                }
                self.fwd_layer += 1;
                work += 1;
                continue;
            }

            // ── Layers exhausted for this token. Two paths: ──────────
            // (a) Non-final prefill token: KV is populated, advance pos
            //     and go pick up the next prompt token within budget.
            // (b) Last prefill token OR decode-phase token: record
            //     sample step + copy logits to mirror, set
            //     sample_pending, break — the GPU executes during the
            //     host's frame submit, the next tick consumes the
            //     mirror after the fence signals.
            if (self.prompt_q.items.len > 0) {
                self.pos += 1;
                self.fwd_in_flight = false;
                self.fwd_layer = 0;
                continue;
            }

            const pushes = runtime.computeForwardPushes(
                self.cfg_model,
                &self.scratch,
                self.pos,
            );
            try runtime.recordSampleStep(
                rec,
                &self.scratch,
                self.gm,
                self.cfg_model,
                &self.forward.kernels,
                &pushes,
            );
            recordCopyToHostMirror(rec, &self.scratch.logits, &self.logits_mirror);
            self.sample_pending = true;
            work += 1;
            break;
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
