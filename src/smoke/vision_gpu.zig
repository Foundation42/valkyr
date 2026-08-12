//! GPU vision-embedder parity against the CPU oracle, using the real
//! Gemma 4 checkpoint weights.
//!
//! Skipped (not failed) when the model isn't present locally, so the
//! suite still runs on a machine without a 24 GB download.

const std = @import("std");
const vk = @import("../gpu/vk.zig");
const buffer = @import("../gpu/buffer.zig");
const recorder = @import("../gpu/recorder.zig");
const gpu_model = @import("../gpu/model.zig");
const model_mod = @import("../model.zig");
const hf_cache = @import("../hf_cache.zig");
const runtime = @import("../runtime.zig");
const vision = @import("../vision.zig");
const dtype = @import("../dtype.zig");

const MODEL_ID = "google/gemma-4-12B-it";

/// Materialise a tensor as fp32 on the host. The vision weights are
/// bf16 on disk; the oracle needs floats.
fn toF32(gpa: std.mem.Allocator, t: model_mod.Tensor) ![]f32 {
    const n = t.numel();
    const out = try gpa.alloc(f32, n);
    errdefer gpa.free(out);
    switch (t.dtype) {
        .f32 => @memcpy(out, t.asF32()),
        .bf16 => dtype.bf16SliceToF32(dtype.asU16(t.bytes), out),
        .f16 => dtype.f16SliceToF32(dtype.asU16(t.bytes), out),
        else => return error.UnsupportedWeightDtype,
    }
    return out;
}

pub fn runVisionGpuParity(gpa: std.mem.Allocator) !void {
    const dir = hf_cache.resolveModelArg(gpa, MODEL_ID) catch {
        std.debug.print("SKIP vision GPU parity ({s} not in local cache)\n", .{MODEL_ID});
        return;
    };
    defer gpa.free(dir);

    var cpu = model_mod.Model.load(gpa, dir) catch {
        std.debug.print("SKIP vision GPU parity ({s} not loadable)\n", .{MODEL_ID});
        return;
    };
    defer cpu.deinit();

    const v = cpu.vision orelse {
        std.debug.print("SKIP vision GPU parity (checkpoint has no vision embedder)\n", .{});
        return;
    };

    var ctx = try vk.Context.init(gpa);
    defer ctx.deinit();

    const embed_dim = cpu.config.hidden_size;
    const patch_elems = v.patch_dense_w.shape[v.patch_dense_w.shape.len - 1];

    // ── A deterministic synthetic image, sized so smartResize is a
    // no-op: 10x8 = 80 patches, comfortably inside [40, 280].
    const img_w = vision.PATCH * 10;
    const img_h = vision.PATCH * 8;
    const img = try gpa.alloc(u8, img_w * img_h * 3);
    defer gpa.free(img);
    var seed: u32 = 0x5EED_1234;
    for (img) |*px| {
        seed = seed *% 1664525 +% 1013904223;
        px.* = @truncate(seed >> 16);
    }

    var patches = try vision.preprocess(gpa, img, img_w, img_h);
    defer patches.deinit();
    const n_patches = patches.count();

    // ── Host-side weights for the oracle ────────────────────────────
    const ln1_w = try toF32(gpa, v.patch_ln1_w);
    defer gpa.free(ln1_w);
    const ln1_b = try toF32(gpa, v.patch_ln1_b);
    defer gpa.free(ln1_b);
    const dense_w = try toF32(gpa, v.patch_dense_w);
    defer gpa.free(dense_w);
    const dense_b = try toF32(gpa, v.patch_dense_b);
    defer gpa.free(dense_b);
    const ln2_w = try toF32(gpa, v.patch_ln2_w);
    defer gpa.free(ln2_w);
    const ln2_b = try toF32(gpa, v.patch_ln2_b);
    defer gpa.free(ln2_b);
    const pos_emb = try toF32(gpa, v.pos_embedding);
    defer gpa.free(pos_emb);
    const pn_w = try toF32(gpa, v.pos_norm_w);
    defer gpa.free(pn_w);
    const pn_b = try toF32(gpa, v.pos_norm_b);
    defer gpa.free(pn_b);
    const proj_w = try toF32(gpa, v.embedding_projection);
    defer gpa.free(proj_w);

    const wts = vision.Weights{
        .ln1_w = ln1_w,           .ln1_b = ln1_b,
        .dense_w = dense_w,       .dense_b = dense_b,
        .ln2_w = ln2_w,           .ln2_b = ln2_b,
        .pos_emb = pos_emb,       .pos_size = v.pos_embedding.shape[0],
        .pos_norm_w = pn_w,       .pos_norm_b = pn_b,
        .proj_w = proj_w,         .embed_dim = embed_dim,
    };

    const want = try gpa.alloc(f32, n_patches * embed_dim);
    defer gpa.free(want);
    try vision.embedCpu(gpa, patches, wts, want);

    // ── GPU path ────────────────────────────────────────────────────
    // Upload ONLY the vision tensors. Routing this through
    // GpuModel.upload would drag in the whole 12B text stack — 48 GB at
    // fp32, and even quantized it is minutes of work for a test that
    // exercises none of it. The fp32 arrays materialised above for the
    // oracle are exactly what the device wants, so they serve twice.
    var gv_owned = gpu_model.GpuVision{
        .patch_ln1_w = try buffer.Buffer.initStatic(&ctx, f32, ln1_w),
        .patch_ln1_b = try buffer.Buffer.initStatic(&ctx, f32, ln1_b),
        .patch_dense_w = try buffer.Buffer.initStatic(&ctx, f32, dense_w),
        .patch_dense_b = try buffer.Buffer.initStatic(&ctx, f32, dense_b),
        .patch_ln2_w = try buffer.Buffer.initStatic(&ctx, f32, ln2_w),
        .patch_ln2_b = try buffer.Buffer.initStatic(&ctx, f32, ln2_b),
        .pos_norm_w = try buffer.Buffer.initStatic(&ctx, f32, pn_w),
        .pos_norm_b = try buffer.Buffer.initStatic(&ctx, f32, pn_b),
        .embedding_projection = try buffer.Buffer.initStatic(&ctx, f32, proj_w),
        .patch_elems = patch_elems,
    };
    defer gv_owned.deinit(ctx.device);
    const gv = &gv_owned;

    var kernels = try runtime.VisionKernels.init(&ctx);
    defer kernels.deinit();

    var sc = try runtime.VisionScratch.init(&ctx, n_patches, patch_elems, embed_dim);
    defer sc.deinit(ctx.device);

    // Per-patch positional sum: axis 0 keyed by column, axis 1 by row.
    // Summing host-side turns a device gather into a plain add.
    const pos_sum = try gpa.alloc(f32, n_patches * embed_dim);
    defer gpa.free(pos_sum);
    for (0..n_patches) |i| {
        const col = patches.colOf(i);
        const row = patches.rowOf(i);
        const ex = pos_emb[(col * 2 + 0) * embed_dim ..][0..embed_dim];
        const ey = pos_emb[(row * 2 + 1) * embed_dim ..][0..embed_dim];
        const dst = pos_sum[i * embed_dim ..][0..embed_dim];
        for (0..embed_dim) |j| dst[j] = ex[j] + ey[j];
    }

    var buf_patches = try buffer.Buffer.initStatic(&ctx, f32, patches.data);
    defer buf_patches.deinit(ctx.device);
    var buf_pos = try buffer.Buffer.initStatic(&ctx, f32, pos_sum);
    defer buf_pos.deinit(ctx.device);

    var rec = try recorder.Recorder.init(&ctx, 64, 512);
    defer rec.deinit();
    try rec.reset();
    try rec.begin();
    try runtime.recordVisionEmbed(
        &rec, &sc, &buf_patches, &buf_pos, gv, &kernels,
        @intCast(n_patches), @intCast(embed_dim),
    );
    try rec.endAndSubmit();

    const got = try gpa.alloc(f32, n_patches * embed_dim);
    defer gpa.free(got);
    try sc.out.readBack(&ctx, f32, got);

    var max_abs: f32 = 0;
    var max_ref: f32 = 0;
    for (want, got) |w, g| {
        const d = @abs(w - g);
        if (d > max_abs) max_abs = d;
        if (@abs(w) > max_ref) max_ref = @abs(w);
    }
    const rel = if (max_ref > 0) max_abs / max_ref else max_abs;
    if (rel > 2e-3) {
        std.debug.print(
            "vision GPU parity FAIL: max|Δ|={e:.3} max|ref|={e:.3} rel={e:.3}\n",
            .{ max_abs, max_ref, rel },
        );
        return error.ParityFailed;
    }

    // An all-zero output would sail through a relative check, so assert
    // the embedder actually produced signal.
    var energy: f64 = 0;
    for (got) |g| energy += @as(f64, g) * @as(f64, g);
    if (energy == 0) return error.VisionOutputAllZero;

    std.debug.print(
        "PASS vision GPU embedder parity ({d} patches x {d} dim vs CPU oracle, max rel={e:.2}) on {s}\n",
        .{ n_patches, embed_dim, rel, ctx.deviceName() },
    );
}
