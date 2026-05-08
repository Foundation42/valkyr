//! Fold a `.lvkpt` LoRA checkpoint into a freshly-uploaded `GpuModel`,
//! mutating the seven dense projection weights in place.
//!
//! The math:
//!
//!     y = x · Wᵀ + (α/r) · (x · Aᵀ) · Bᵀ
//!       = x · (W + (α/r) · B · A)ᵀ
//!
//! so a one-time pre-multiplication `W += (α/r) · B · A` lets the
//! existing inference path run unchanged at production speed — zero
//! per-token LoRA cost, zero new shaders.
//!
//! Trade-off vs. an at-inference LoRA adapter pass: merging burns
//! whatever bf16 round-trip noise the weight format costs on the
//! merged weight (and forecloses on swapping different LoRAs without
//! reloading the base). Both fine for "I trained a fine-tune; let me
//! chat with it." If you want to host multiple LoRAs over one base
//! model, the at-inference path is the right mode and a future chunk.
//!
//! Precision support:
//!   - `fp32_all`     — exact: read as f32, add fp32 delta, write back.
//!   - `bf16_matmul`  — round-trips: bf16 → f32 → +delta → f32 → bf16.
//!   - `q4_0_matmul`  — error: requantising a Q4 block changes its
//!     scale grid; non-trivial. Future chunk.
//!   - `q4_k_matmul`  — error: same story for Q4_K super-blocks.

const std = @import("std");
const vk = @import("../gpu/vk.zig");
const buffer = @import("../gpu/buffer.zig");
const dtype = @import("../dtype.zig");
const model_mod = @import("../model.zig");
const gpu_model = @import("../gpu/model.zig");
const train_transformer = @import("../train/transformer.zig");

pub const Options = struct {
    /// Path to the `.lvkpt` produced by `--lora-finetune --out`.
    lvkpt_path: []const u8,
    /// Bitmask of `train_transformer.LoraTarget.q | .k | ...`. Must
    /// match the value the `.lvkpt` was saved with — `cfgShapeMatches`
    /// validates and the loop walks the file body in this order.
    lora_targets: u32,
    /// Adapter rank used at fine-tune time.
    lora_rank: u32,
    /// Adapter scale α. Effective merge factor is `α / rank`.
    lora_alpha: f32,
};

const lora_checkpoint_magic: [4]u8 = .{ 'V', 'L', 'K', 'P' };

/// Same field layout `train_transformer.CheckpointHeader` writes —
/// kept private here so we don't need to expose it from the trainer.
const Header = extern struct {
    magic: [4]u8,
    version: u32,
    step_t: u32,
    cfg_size: u32,
};

const Proj = enum(u3) { q = 0, k, v, o, gate, up, down };

fn projBit(p: Proj) u32 {
    return @as(u32, 1) << @intFromEnum(p);
}

/// Merge the LoRA deltas in `opts.lvkpt_path` into the projection
/// weights of `gm` in place. Returns the per-target stats only via
/// stdout; callers don't need the bytes back.
pub fn run(
    allocator: std.mem.Allocator,
    ctx: *const vk.Context,
    gm: *gpu_model.GpuModel,
    cpu_cfg: model_mod.Config,
    opts: Options,
) !void {
    const stdout = std.io.getStdOut().writer();

    if (cpu_cfg.attn_output_gate) {
        // Qwen3.5's attn_output_gate widens q_proj to (q, gate) — the
        // LoRA trainer doesn't model this and the .lvkpt format won't
        // match. Refuse rather than silently corrupting weights.
        return error.LoraMergeAttnOutputGateUnsupported;
    }

    if (opts.lora_targets == 0) return error.LoraMergeNoTargets;
    if (opts.lora_rank == 0) return error.LoraMergeZeroRank;

    switch (gm.precision) {
        .fp32_all, .bf16_matmul => {},
        .q4_0_matmul, .q4_k_matmul => return error.LoraMergeUnsupportedPrecision,
    }

    // ── Open + buffer the file. 4 MiB matches the trainer's save.
    const f = try std.fs.cwd().openFile(opts.lvkpt_path, .{ .mode = .read_only });
    defer f.close();

    const Reader = std.io.BufferedReader(4 * 1024 * 1024, std.fs.File.Reader);
    var br = Reader{ .unbuffered_reader = f.reader() };
    const r = br.reader();

    // ── Header + on-disk Config.
    var header: Header = undefined;
    try r.readNoEof(std.mem.asBytes(&header));
    if (!std.mem.eql(u8, &header.magic, &lora_checkpoint_magic)) return error.LoraMergeBadMagic;
    if (header.cfg_size != @sizeOf(train_transformer.Config)) return error.LoraMergeCfgSizeMismatch;

    var ondisk_cfg: train_transformer.Config = undefined;
    try r.readNoEof(std.mem.asBytes(&ondisk_cfg));
    if (ondisk_cfg.lora_targets != opts.lora_targets) return error.LoraMergeTargetsMismatch;
    if (ondisk_cfg.lora_rank != opts.lora_rank) return error.LoraMergeRankMismatch;
    if (ondisk_cfg.lora_alpha != opts.lora_alpha) {
        // Not strictly fatal — the user might want to scale α at merge
        // time. But unexpected enough to surface.
        try stdout.print("[lora-merge] note: cfg alpha {d:.4} != cli alpha {d:.4} — using cli value\n", .{ ondisk_cfg.lora_alpha, opts.lora_alpha });
    }

    // ── Compute per-projection shapes from the cpu Config.
    const dim: usize = @intCast(cpu_cfg.hidden_size);
    const n_layers: usize = @intCast(cpu_cfg.num_hidden_layers);
    const q_dim: usize = @as(usize, @intCast(cpu_cfg.num_attention_heads)) * @as(usize, @intCast(cpu_cfg.head_dim));
    const kv_dim: usize = @as(usize, @intCast(cpu_cfg.num_key_value_heads)) * @as(usize, @intCast(cpu_cfg.head_dim));
    const ff_dim: usize = @intCast(cpu_cfg.intermediate_size);
    const r_rank: usize = @intCast(opts.lora_rank);
    const alpha_over_r: f32 = opts.lora_alpha / @as(f32, @floatFromInt(opts.lora_rank));

    // ── For each enabled target, walk every layer and merge.
    //
    // Outer loop over Proj must match `collectLoraBlobs`'s
    // `inline for @typeInfo(Proj).@"enum".fields` order — q,k,v,o,
    // gate,up,down — because that's the order the .lvkpt body was
    // written in.
    const projs_in_order = [_]Proj{ .q, .k, .v, .o, .gate, .up, .down };
    for (projs_in_order) |proj| {
        if ((opts.lora_targets & projBit(proj)) == 0) continue;

        const proj_shape = projShape(proj, dim, q_dim, kv_dim, ff_dim);
        const N_W: usize = proj_shape.n;
        const K_W: usize = proj_shape.k;
        const a_numel: usize = r_rank * K_W;
        const b_numel: usize = N_W * r_rank;

        // Per-target host scratch (re-used across layers). Allocating
        // once keeps the inner loop allocation-free.
        const a_buf = try allocator.alloc(f32, a_numel);
        defer allocator.free(a_buf);
        const b_buf = try allocator.alloc(f32, b_numel);
        defer allocator.free(b_buf);
        const w_fp32 = try allocator.alloc(f32, N_W * K_W);
        defer allocator.free(w_fp32);
        const w_bf16: ?[]u16 = switch (gm.precision) {
            .bf16_matmul => try allocator.alloc(u16, N_W * K_W),
            else => null,
        };
        defer if (w_bf16) |buf_| allocator.free(buf_);

        const t_proj_start = std.time.nanoTimestamp();
        var max_abs_delta: f32 = 0.0;
        var sum_abs_delta: f64 = 0.0;

        for (0..n_layers) |i| {
            // Body for one (proj, layer) is six fp32 blocks: A, m_a,
            // v_a, B, m_b, v_b. We need A and B; m/v are skipped. Total
            // skip = 2 * a_numel + 2 * b_numel f32s, but we have to
            // interleave them with A/B in the read order so the cursor
            // stays aligned with the file.
            try r.readNoEof(std.mem.sliceAsBytes(a_buf));
            try r.skipBytes(@as(u64, a_numel) * @sizeOf(f32) * 2, .{}); // m_a + v_a

            try r.readNoEof(std.mem.sliceAsBytes(b_buf));
            try r.skipBytes(@as(u64, b_numel) * @sizeOf(f32) * 2, .{}); // m_b + v_b

            // ── Read W from the GPU as fp32.
            const w_buf_opt: ?*buffer.Buffer = projBuffer(&gm.layers[i], proj);
            const w_buf = w_buf_opt orelse return error.LoraMergeMissingProjection;

            switch (gm.precision) {
                .fp32_all => try w_buf.readBack(ctx, f32, w_fp32),
                .bf16_matmul => {
                    try w_buf.readBack(ctx, u16, w_bf16.?);
                    for (w_bf16.?, w_fp32) |s, *d| d.* = dtype.bf16ToF32(s);
                },
                .q4_0_matmul, .q4_k_matmul => unreachable,
            }

            // ── Merge: W[n, k] += (α/r) * Σ_i B[n, i] * A[i, k].
            const stats = applyLoraDeltaFp32(w_fp32, a_buf, b_buf, N_W, K_W, r_rank, alpha_over_r);
            if (stats.max_abs > max_abs_delta) max_abs_delta = stats.max_abs;
            sum_abs_delta += stats.sum_abs;

            // ── Write W back, narrowing if needed.
            switch (gm.precision) {
                .fp32_all => try w_buf.uploadFromHost(ctx, f32, w_fp32),
                .bf16_matmul => {
                    dtype.f32SliceToBf16(w_fp32, w_bf16.?);
                    try w_buf.uploadFromHost(ctx, u16, w_bf16.?);
                },
                .q4_0_matmul, .q4_k_matmul => unreachable,
            }
        }

        const t_proj_end = std.time.nanoTimestamp();
        const proj_ms: f64 = @as(f64, @floatFromInt(t_proj_end - t_proj_start)) / 1.0e6;
        const numel_total: f64 = @as(f64, @floatFromInt(n_layers * N_W * K_W));
        const mean_abs: f64 = sum_abs_delta / numel_total;
        try stdout.print("[lora-merge] {s}: {d} layers × [{d}, {d}] merged in {d:.0} ms (max|Δ|={e:.3} mean|Δ|={e:.3})\n", .{ projName(proj), n_layers, N_W, K_W, proj_ms, max_abs_delta, mean_abs });
    }
}

/// In-place LoRA merge on a fp32 weight buffer:
///
///     W[n, k] += (α/r) · Σ_i B[n, i] · A[i, k]
///
/// W is `[N, K]` row-major; A is `[r, K]` row-major; B is `[N, r]`
/// row-major. Returns simple |Δ| stats for caller diagnostics — not
/// load-bearing for correctness.
///
/// Inner-loop ordering picked for cache locality: for each row of W
/// we walk r in the outer loop (broadcasting `B[n, i]`), then K in
/// the inner loop (so `A[i, k]` and `W[n, k]` are unit-stride). At
/// rank-16 this is ~1 GFLOP per Qwen3-0.6B projection — <1 s host
/// per merge over the whole stack.
pub const MergeStats = struct { max_abs: f32, sum_abs: f64 };
pub fn applyLoraDeltaFp32(
    w: []f32,
    a: []const f32,
    b: []const f32,
    n: usize,
    k: usize,
    r: usize,
    alpha_over_r: f32,
) MergeStats {
    std.debug.assert(w.len == n * k);
    std.debug.assert(a.len == r * k);
    std.debug.assert(b.len == n * r);

    var max_abs: f32 = 0.0;
    var sum_abs: f64 = 0.0;

    for (0..n) |row| {
        const b_row = b[row * r ..][0..r];
        const w_row = w[row * k ..][0..k];
        for (0..r) |i| {
            const scaled_b = alpha_over_r * b_row[i];
            const a_row = a[i * k ..][0..k];
            for (0..k) |col| {
                const d = scaled_b * a_row[col];
                w_row[col] += d;
                const ad = @abs(d);
                if (ad > max_abs) max_abs = ad;
                sum_abs += @as(f64, ad);
            }
        }
    }
    return .{ .max_abs = max_abs, .sum_abs = sum_abs };
}

const ProjShape = struct { n: usize, k: usize };

fn projShape(p: Proj, dim: usize, q_dim: usize, kv_dim: usize, ff_dim: usize) ProjShape {
    return switch (p) {
        .q => .{ .n = q_dim, .k = dim },
        .k => .{ .n = kv_dim, .k = dim },
        .v => .{ .n = kv_dim, .k = dim },
        .o => .{ .n = dim, .k = q_dim },
        .gate => .{ .n = ff_dim, .k = dim },
        .up => .{ .n = ff_dim, .k = dim },
        .down => .{ .n = dim, .k = ff_dim },
    };
}

fn projBuffer(layer: *gpu_model.GpuLayer, p: Proj) ?*buffer.Buffer {
    return switch (p) {
        .q => if (layer.q_proj) |*b| b else null,
        .k => if (layer.k_proj) |*b| b else null,
        .v => if (layer.v_proj) |*b| b else null,
        .o => if (layer.o_proj) |*b| b else null,
        .gate => &layer.gate_proj,
        .up => &layer.up_proj,
        .down => &layer.down_proj,
    };
}

fn projName(p: Proj) []const u8 {
    return switch (p) {
        .q => "q",
        .k => "k",
        .v => "v",
        .o => "o",
        .gate => "gate",
        .up => "up",
        .down => "down",
    };
}

// Spec parsing lives in `src/commands/finetune.zig` (`parseLoraTargets`)
// — main.zig already imports it for `--fine-tune` / `--gen-from-ckpt`,
// so we don't duplicate the table here.
