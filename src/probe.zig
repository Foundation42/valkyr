//! Probe interface — extension points for instrumenting the inference
//! pipeline at well-defined hook points: token start/end, per-layer
//! entry/exit, attention, and logits.
//!
//! Designed against two co-equal use cases:
//!
//!   1. Research instrumentation. The Resonance-Flow Framework (see
//!      `rff/resonance_flow_spec.md`) needs five observables computed
//!      per-token and integrated over an exchange: B(t) activation
//!      entropy, D(t) logit conditional entropy + KL, K(t) attention
//!      entropy + rank, P(t) persistence, F(t) fractal dimension. The
//!      first four are instrumentable here; F(t) is post-processing on
//!      the JSONL stream this module produces.
//!
//!   2. Engine integration. A probe that *consumes* a hidden state and
//!      hands it to a host system (e.g. Matryoshka, the frame-budget
//!      cooperative-inference engine) is structurally identical to a
//!      probe that *measures* it. Same vtable, same hook points, same
//!      data shapes. The bus pays rent in both currencies.
//!
//! Design invariants:
//!
//!   - Empty bus path (no probes installed) is bit-identical to the
//!     pre-probe baseline. Verified by parity-check after wiring.
//!   - Slow paths (per-layer GPU readbacks for hidden states) are gated
//!     on probe demand: `bus.wants(.hidden_post_layer)` etc. If nothing
//!     wants it, the readback isn't issued.
//!   - Probes own their own state and writers; the bus is just a
//!     vtable dispatcher. Multiple probes compose naturally.

const std = @import("std");

// ── Observable taxonomy ──────────────────────────────────────────────
//
// What a probe can ask the bus to surface. Used by the chat decode loop
// to decide which (potentially expensive) per-step readbacks to issue.
pub const Observable = enum {
    /// Hidden state on entry to a transformer layer, before
    /// input_layernorm. Emitted via `on_layer_entry`.
    hidden_pre_layer,
    /// Hidden state on exit from a transformer layer, after the FFN
    /// residual. Emitted via `on_layer_exit`.
    hidden_post_layer,
    /// Post-softmax attention weights + Q/K/V projections per layer.
    /// Emitted via `on_attention`. Not yet wired in v0.
    attention,
    /// Output logit distribution (pre-sampling). Emitted via `on_logits`.
    logits,
};

// ── Per-call context ─────────────────────────────────────────────────

/// Static-shape information passed to every hook so probes don't need
/// to know about model.zig / config.zig directly.
pub const ModelInfo = struct {
    family: []const u8,
    n_layers: u32,
    hidden_size: u32,
    n_heads: u32,
    n_kv_heads: u32,
    head_dim: u32,
    vocab_size: u32,
};

/// What a hook needs to know about *which* token / layer it's looking
/// at. `layer_idx` is null at non-layer hooks.
pub const Context = struct {
    info: ModelInfo,
    /// Sequential token index within the current exchange (resets each
    /// chat session, NOT each turn — the session is the unit a C value
    /// is integrated over).
    token_index: u32,
    /// Position in the KV cache. Persists across turns within a session.
    pos: usize,
    /// The token id being processed at this step. For prefill, this is
    /// the prompt token; for decode, the previous sample.
    token_id: u32,
    layer_idx: ?u16 = null,
    /// True when this is a prefill token (consuming the user prompt);
    /// false during response generation.
    is_prefill: bool,
};

/// Per-hook payload. Most fields are null at most call sites; each hook
/// populates only the slices it has. Probes pull what they need.
pub const Slices = struct {
    /// `hidden_size` floats. Set in on_layer_entry and on_layer_exit.
    hidden: ?[]const f32 = null,
    /// Post-softmax attention weights, `[n_heads, n_pos]` row-major
    /// with stride `attn_scores_stride` (the buffer's row width is
    /// max_pos, only the first n_pos entries of each row are valid).
    /// Set in on_attention.
    attn_weights: ?[]const f32 = null,
    /// Optional Q/K/V slices for downstream analysis (eigenspectra,
    /// head specialisation). Not populated in v0 — exposed in the
    /// signature for future-proofing per the dual-purpose design.
    attn_q: ?[]const f32 = null,
    attn_k: ?[]const f32 = null,
    attn_v: ?[]const f32 = null,
    attn_n_pos: u32 = 0,
    attn_n_heads: u32 = 0,
    attn_scores_stride: u32 = 0,
    /// `vocab_size` floats. Set in on_logits.
    logits: ?[]const f32 = null,
    /// Set in on_token_end.
    sampled_id: ?u32 = null,
};

// ── Probe vtable ─────────────────────────────────────────────────────

pub const Probe = struct {
    impl: *anyopaque,
    vt: *const VTable,

    pub const VTable = struct {
        on_token_start: ?*const fn (*anyopaque, Context, Slices) anyerror!void = null,
        on_layer_entry: ?*const fn (*anyopaque, Context, Slices) anyerror!void = null,
        on_layer_exit: ?*const fn (*anyopaque, Context, Slices) anyerror!void = null,
        on_attention: ?*const fn (*anyopaque, Context, Slices) anyerror!void = null,
        on_logits: ?*const fn (*anyopaque, Context, Slices) anyerror!void = null,
        on_token_end: ?*const fn (*anyopaque, Context, Slices) anyerror!void = null,
        deinit: ?*const fn (*anyopaque, std.mem.Allocator) void = null,
        wants: *const fn (*anyopaque, Observable) bool,
    };
};

// ── Bus ──────────────────────────────────────────────────────────────

pub const Bus = struct {
    probes: []Probe = &.{},
    /// Aggregated needs across all installed probes. Computed eagerly
    /// when probes are added so the decode loop's gating reads are
    /// branchless lookups, not vtable iterations.
    needs_hidden_pre: bool = false,
    needs_hidden_post: bool = false,
    needs_attention: bool = false,
    needs_logits: bool = false,

    pub fn isEmpty(self: *const Bus) bool {
        return self.probes.len == 0;
    }

    /// Recompute the aggregated needs flags. Called by the harness
    /// after all probes are registered.
    pub fn finalize(self: *Bus) void {
        self.needs_hidden_pre = self.anyWants(.hidden_pre_layer);
        self.needs_hidden_post = self.anyWants(.hidden_post_layer);
        self.needs_attention = self.anyWants(.attention);
        self.needs_logits = self.anyWants(.logits);
    }

    fn anyWants(self: *const Bus, what: Observable) bool {
        for (self.probes) |p| {
            if (p.vt.wants(p.impl, what)) return true;
        }
        return false;
    }

    pub fn onTokenStart(self: *const Bus, ctx: Context) !void {
        for (self.probes) |p| {
            if (p.vt.on_token_start) |f| try f(p.impl, ctx, .{});
        }
    }

    pub fn onLayerEntry(self: *const Bus, ctx: Context, hidden: []const f32) !void {
        const s = Slices{ .hidden = hidden };
        for (self.probes) |p| {
            if (p.vt.on_layer_entry) |f| try f(p.impl, ctx, s);
        }
    }

    pub fn onLayerExit(self: *const Bus, ctx: Context, hidden: []const f32) !void {
        const s = Slices{ .hidden = hidden };
        for (self.probes) |p| {
            if (p.vt.on_layer_exit) |f| try f(p.impl, ctx, s);
        }
    }

    pub fn onAttention(
        self: *const Bus,
        ctx: Context,
        weights: []const f32,
        n_heads: u32,
        n_pos: u32,
        scores_stride: u32,
    ) !void {
        const s = Slices{
            .attn_weights = weights,
            .attn_n_heads = n_heads,
            .attn_n_pos = n_pos,
            .attn_scores_stride = scores_stride,
        };
        for (self.probes) |p| {
            if (p.vt.on_attention) |f| try f(p.impl, ctx, s);
        }
    }

    pub fn onLogits(self: *const Bus, ctx: Context, logits: []const f32) !void {
        const s = Slices{ .logits = logits };
        for (self.probes) |p| {
            if (p.vt.on_logits) |f| try f(p.impl, ctx, s);
        }
    }

    pub fn onTokenEnd(self: *const Bus, ctx: Context, sampled_id: u32) !void {
        const s = Slices{ .sampled_id = sampled_id };
        for (self.probes) |p| {
            if (p.vt.on_token_end) |f| try f(p.impl, ctx, s);
        }
    }

    pub fn deinit(self: *Bus, gpa: std.mem.Allocator) void {
        for (self.probes) |p| {
            if (p.vt.deinit) |f| f(p.impl, gpa);
        }
        gpa.free(self.probes);
        self.probes = &.{};
    }
};

// ── JSONL writer (shared across probes) ──────────────────────────────
//
// One file per exchange. Records are line-delimited JSON; the first
// record is a header documenting model + probe set + observable
// definitions, so downstream tooling can self-describe without a
// version-locked schema file. Subsequent records are tagged by `kind`.

pub const JsonlWriter = struct {
    file: std.fs.File,
    mu: std.Thread.Mutex = .{},
    closed: bool = false,

    pub fn open(path: []const u8) !JsonlWriter {
        const f = try std.fs.cwd().createFile(path, .{ .truncate = true });
        return .{ .file = f };
    }

    pub fn writeLine(self: *JsonlWriter, json: []const u8) !void {
        self.mu.lock();
        defer self.mu.unlock();
        try self.file.writeAll(json);
        try self.file.writeAll("\n");
    }

    pub fn close(self: *JsonlWriter) void {
        if (self.closed) return;
        self.file.close();
        self.closed = true;
    }
};

// ── LogitProbe — D(t) ────────────────────────────────────────────────
//
// Per-token: conditional entropy H(t | context) and KL divergence from
// a precomputed null prior. The null prior is the logit distribution
// the model emits for a single BOS token at position 0 — i.e. what the
// model predicts absent any conditioning. KL(p || q) where p is the
// current step distribution and q is the null prior measures how far
// the prompt has driven the model away from baseline.

pub const LogitProbe = struct {
    gpa: std.mem.Allocator,
    writer: *JsonlWriter,
    vocab_size: u32,
    /// log(softmax(null_prior_logits)). Stored in log-space to make KL
    /// numerically stable (KL = Σ p_i (log p_i − log q_i)).
    null_log_prior: []f32,
    /// Scratch for softmax of incoming logits.
    scratch: []f32,

    pub fn create(
        gpa: std.mem.Allocator,
        writer: *JsonlWriter,
        vocab_size: u32,
        null_prior_logits: []const f32,
    ) !*LogitProbe {
        std.debug.assert(null_prior_logits.len == vocab_size);

        const self = try gpa.create(LogitProbe);
        errdefer gpa.destroy(self);
        const log_prior = try gpa.alloc(f32, vocab_size);
        errdefer gpa.free(log_prior);
        const scratch = try gpa.alloc(f32, vocab_size);
        errdefer gpa.free(scratch);

        // log-softmax of null prior, numerically stable.
        var max_l: f32 = -std.math.inf(f32);
        for (null_prior_logits) |x| if (x > max_l) {
            max_l = x;
        };
        var sum_exp: f64 = 0.0;
        for (null_prior_logits) |x| sum_exp += @exp(@as(f64, x - max_l));
        const log_z: f32 = max_l + @as(f32, @floatCast(@log(sum_exp)));
        for (null_prior_logits, log_prior) |x, *out| out.* = x - log_z;

        self.* = .{
            .gpa = gpa,
            .writer = writer,
            .vocab_size = vocab_size,
            .null_log_prior = log_prior,
            .scratch = scratch,
        };
        return self;
    }

    fn wants(_: *anyopaque, what: Observable) bool {
        return what == .logits;
    }

    fn onLogits(impl: *anyopaque, ctx: Context, s: Slices) !void {
        const self: *LogitProbe = @ptrCast(@alignCast(impl));
        const logits = s.logits.?;
        std.debug.assert(logits.len == self.vocab_size);

        // Stable softmax → entropy + KL from null prior.
        var max_l: f32 = -std.math.inf(f32);
        for (logits) |x| if (x > max_l) {
            max_l = x;
        };
        var sum_exp: f64 = 0.0;
        for (logits, self.scratch) |x, *p| {
            const e = @exp(@as(f64, x - max_l));
            p.* = @floatCast(e);
            sum_exp += e;
        }
        const inv_z: f32 = @floatCast(1.0 / sum_exp);
        const log_z: f32 = max_l + @as(f32, @floatCast(@log(sum_exp)));

        var entropy: f64 = 0.0;
        var kl: f64 = 0.0;
        var top_p: f32 = 0.0;
        var top_id: u32 = 0;
        for (self.scratch, logits, self.null_log_prior, 0..) |p_unscaled, x, log_q, i| {
            const p = p_unscaled * inv_z;
            if (p > 0.0) {
                const log_p: f64 = @as(f64, x - log_z);
                entropy -= @as(f64, p) * log_p;
                kl += @as(f64, p) * (log_p - @as(f64, log_q));
            }
            if (p > top_p) {
                top_p = p;
                top_id = @intCast(i);
            }
        }

        var buf: [256]u8 = undefined;
        const line = try std.fmt.bufPrint(&buf,
            "{{\"kind\":\"logits\",\"tok\":{d},\"pos\":{d},\"id\":{d},\"prefill\":{any},\"entropy\":{d:.6},\"kl_null\":{d:.6},\"top_p\":{d:.6},\"top_id\":{d}}}",
            .{
                ctx.token_index,
                ctx.pos,
                ctx.token_id,
                ctx.is_prefill,
                entropy,
                kl,
                top_p,
                top_id,
            },
        );
        try self.writer.writeLine(line);
    }

    fn deinitImpl(impl: *anyopaque, gpa: std.mem.Allocator) void {
        const self: *LogitProbe = @ptrCast(@alignCast(impl));
        gpa.free(self.null_log_prior);
        gpa.free(self.scratch);
        gpa.destroy(self);
    }

    pub fn probe(self: *LogitProbe) Probe {
        return .{
            .impl = self,
            .vt = &VT,
        };
    }

    const VT: Probe.VTable = .{
        .wants = wants,
        .on_logits = onLogits,
        .deinit = deinitImpl,
    };
};

// ── ActivationEntropyProbe — B(t) ────────────────────────────────────
//
// Per (token, layer): Shannon entropy of the activation distribution
// derived from the post-layer hidden state. We use the L2-energy
// distribution `p_i = a_i^2 / Σ a_j^2`, which is dimension-free and
// captures how spread the activation is across feature dimensions.
// Reported normalized by `log(hidden_size)` so values land in [0, 1].
//
// Choice of distribution is documented in the JSONL header — different
// downstream studies may want different normalizations and we want the
// trace to be self-describing.

pub const ActivationEntropyProbe = struct {
    gpa: std.mem.Allocator,
    writer: *JsonlWriter,

    pub fn create(gpa: std.mem.Allocator, writer: *JsonlWriter) !*ActivationEntropyProbe {
        const self = try gpa.create(ActivationEntropyProbe);
        self.* = .{ .gpa = gpa, .writer = writer };
        return self;
    }

    fn wants(_: *anyopaque, what: Observable) bool {
        return what == .hidden_post_layer;
    }

    fn onLayerExit(impl: *anyopaque, ctx: Context, s: Slices) !void {
        const self: *ActivationEntropyProbe = @ptrCast(@alignCast(impl));
        const hidden = s.hidden.?;

        // Build energy distribution and accumulate entropy in one pass.
        var sum_sq: f64 = 0.0;
        for (hidden) |a| sum_sq += @as(f64, a) * @as(f64, a);
        if (!std.math.isFinite(sum_sq) or sum_sq == 0.0) return;

        var entropy: f64 = 0.0;
        const inv_sum: f64 = 1.0 / sum_sq;
        for (hidden) |a| {
            const p = @as(f64, a) * @as(f64, a) * inv_sum;
            if (p > 0.0) entropy -= p * @log(p);
        }
        const norm = @log(@as(f64, @floatFromInt(hidden.len)));
        const entropy_norm: f64 = if (norm > 0.0) entropy / norm else 0.0;

        // L2 norm of the residual stream — a useful companion signal
        // (residual scale tends to grow with depth; entropy alone misses
        // this). Cheap, already computed.
        const l2: f64 = @sqrt(sum_sq);

        var buf: [192]u8 = undefined;
        const line = try std.fmt.bufPrint(&buf,
            "{{\"kind\":\"act\",\"tok\":{d},\"pos\":{d},\"layer\":{d},\"prefill\":{any},\"entropy\":{d:.6},\"entropy_norm\":{d:.6},\"l2\":{d:.6}}}",
            .{
                ctx.token_index,
                ctx.pos,
                ctx.layer_idx.?,
                ctx.is_prefill,
                entropy,
                entropy_norm,
                l2,
            },
        );
        try self.writer.writeLine(line);
    }

    fn deinitImpl(impl: *anyopaque, gpa: std.mem.Allocator) void {
        const self: *ActivationEntropyProbe = @ptrCast(@alignCast(impl));
        gpa.destroy(self);
    }

    pub fn probe(self: *ActivationEntropyProbe) Probe {
        return .{
            .impl = self,
            .vt = &VT,
        };
    }

    const VT: Probe.VTable = .{
        .wants = wants,
        .on_layer_exit = onLayerExit,
        .deinit = deinitImpl,
    };
};

// ── AttentionProbe — K(t) ────────────────────────────────────────────
//
// Per (token, layer): aggregate K(t) statistics across heads.
//
//   - mean Shannon entropy of post-softmax attention weights, in nats
//   - same divided by log(n_pos) for a [0, 1] normalized variant
//   - mean top-weight (max attention weight per head, averaged) — a
//     concentration signal complementary to entropy. Two distributions
//     can have similar entropy but different concentration peaks.
//   - n_pos at this layer (always equal to ctx.pos+1, but recorded
//     explicitly so post-processing doesn't have to derive it).
//
// During decode each step has a single query position, so what we
// log per (token, layer) is a row-vector statistic across heads — no
// rank-via-SVD, that requires accumulating multiple query rows. The
// spec's "effective rank" (5.2.2) belongs to a separate post-
// processing pass over a window of consecutive tokens.

pub const AttentionProbe = struct {
    gpa: std.mem.Allocator,
    writer: *JsonlWriter,

    pub fn create(gpa: std.mem.Allocator, writer: *JsonlWriter) !*AttentionProbe {
        const self = try gpa.create(AttentionProbe);
        self.* = .{ .gpa = gpa, .writer = writer };
        return self;
    }

    fn wants(_: *anyopaque, what: Observable) bool {
        return what == .attention;
    }

    fn onAttention(impl: *anyopaque, ctx: Context, s: Slices) !void {
        const self: *AttentionProbe = @ptrCast(@alignCast(impl));
        const w = s.attn_weights.?;
        const n_heads: usize = @intCast(s.attn_n_heads);
        const n_pos: usize = @intCast(s.attn_n_pos);
        const stride: usize = @intCast(s.attn_scores_stride);
        if (n_heads == 0 or n_pos == 0) return;

        var sum_H: f64 = 0;
        var sum_top: f64 = 0;
        for (0..n_heads) |h| {
            const row_start = h * stride;
            const row = w[row_start .. row_start + n_pos];
            var H: f64 = 0;
            var top: f32 = 0;
            for (row) |p| {
                if (p > 0) H -= @as(f64, p) * @log(@as(f64, p));
                if (p > top) top = p;
            }
            sum_H += H;
            sum_top += @as(f64, top);
        }
        const mean_H = sum_H / @as(f64, @floatFromInt(n_heads));
        const mean_top = sum_top / @as(f64, @floatFromInt(n_heads));
        const log_npos = @log(@as(f64, @floatFromInt(n_pos)));
        const mean_H_norm: f64 = if (log_npos > 0) mean_H / log_npos else 0;

        var buf: [256]u8 = undefined;
        const line = try std.fmt.bufPrint(&buf,
            "{{\"kind\":\"attn\",\"tok\":{d},\"pos\":{d},\"layer\":{d},\"prefill\":{any},\"n_pos\":{d},\"entropy\":{d:.6},\"entropy_norm\":{d:.6},\"top\":{d:.6}}}",
            .{
                ctx.token_index,
                ctx.pos,
                ctx.layer_idx.?,
                ctx.is_prefill,
                n_pos,
                mean_H,
                mean_H_norm,
                mean_top,
            },
        );
        try self.writer.writeLine(line);
    }

    fn deinitImpl(impl: *anyopaque, gpa: std.mem.Allocator) void {
        const self: *AttentionProbe = @ptrCast(@alignCast(impl));
        gpa.destroy(self);
    }

    pub fn probe(self: *AttentionProbe) Probe {
        return .{ .impl = self, .vt = &VT };
    }

    const VT: Probe.VTable = .{
        .wants = wants,
        .on_attention = onAttention,
        .deinit = deinitImpl,
    };
};

// ── Header writer ────────────────────────────────────────────────────
//
// Emit a single self-describing line at the top of each JSONL trace.

pub fn writeHeader(
    writer: *JsonlWriter,
    info: ModelInfo,
    model_path: []const u8,
    probe_kinds: []const []const u8,
) !void {
    var buf: [1024]u8 = undefined;
    var fbs = std.io.fixedBufferStream(&buf);
    var w = fbs.writer();
    try w.print(
        "{{\"kind\":\"header\",\"schema\":\"valkyr-probe-v0\",\"model\":\"{s}\",\"family\":\"{s}\",\"n_layers\":{d},\"hidden\":{d},\"n_heads\":{d},\"n_kv_heads\":{d},\"head_dim\":{d},\"vocab\":{d},\"probes\":[",
        .{
            model_path,
            info.family,
            info.n_layers,
            info.hidden_size,
            info.n_heads,
            info.n_kv_heads,
            info.head_dim,
            info.vocab_size,
        },
    );
    for (probe_kinds, 0..) |k, i| {
        if (i > 0) try w.writeAll(",");
        try w.print("\"{s}\"", .{k});
    }
    try w.writeAll(
        "],\"defs\":{\"act.entropy\":\"Shannon entropy of L2-energy distribution p_i = a_i^2 / sum(a_j^2)\",\"act.entropy_norm\":\"entropy / log(hidden)\",\"act.l2\":\"sqrt(sum(a_i^2))\",\"logits.entropy\":\"Shannon entropy of softmax(logits)\",\"logits.kl_null\":\"KL(softmax(logits) || null_prior); null_prior = single-BOS-token forward at pos 0\",\"attn.entropy\":\"per-layer mean across heads of Shannon entropy of post-softmax attention over n_pos keys\",\"attn.entropy_norm\":\"attn.entropy / log(n_pos)\",\"attn.top\":\"per-layer mean across heads of max(attention_weight)\"}}",
    );
    try writer.writeLine(fbs.getWritten());
}
