//! Vision preprocessing + CPU-embedder smoke tests.
//!
//! The embedder math is checked against the GPU path elsewhere; what
//! this file guards is the part with no reference to compare against —
//! the resize/patchify geometry, where an off-by-one produces a
//! perfectly plausible image that is silently cropped, transposed, or
//! channel-swapped.

const std = @import("std");
const vision = @import("../vision.zig");

pub fn runVisionPreprocessSmoke(gpa: std.mem.Allocator) !void {
    // ── smartResize: hand-computed against the reference algorithm ──
    //
    // 1280x720: aligning first gives 1296x720 = 933120 px, over the
    // 280-token cap (645120), so it rescales by
    // beta = sqrt(1280*720 / 645120) and floors to 1056x576 = 22x12
    // = 264 patches.
    {
        const s = vision.smartResize(1280, 720);
        if (s.w != 1056 or s.h != 576) {
            std.debug.print("smartResize(1280,720) = {d}x{d}, want 1056x576\n", .{ s.w, s.h });
            return error.SmartResizeMismatch;
        }
    }
    // 100x100: aligning gives 96x96 = 9216 px, under the 40-token floor
    // (92160), so it scales UP by sqrt(92160/10000) and ceils to
    // 336x336 = 7x7 = 49 patches.
    {
        const s = vision.smartResize(100, 100);
        if (s.w != 336 or s.h != 336) {
            std.debug.print("smartResize(100,100) = {d}x{d}, want 336x336\n", .{ s.w, s.h });
            return error.SmartResizeMismatch;
        }
    }

    // Whatever the input, the token count must land inside the budget
    // and stay patch-aligned. Sweep a range of shapes including extreme
    // aspect ratios, which are where a naive scale-then-align breaks.
    const shapes = [_][2]usize{
        .{ 1920, 1080 }, .{ 1280, 720 },  .{ 640, 480 },  .{ 3840, 2160 },
        .{ 64, 64 },     .{ 4000, 100 },  .{ 100, 4000 }, .{ 1, 1 },
        .{ 47, 4097 },   .{ 800, 600 },   .{ 333, 777 },  .{ 2, 5000 },
    };
    for (shapes) |wh| {
        const s = vision.smartResize(wh[0], wh[1]);
        if (s.w % vision.PATCH != 0 or s.h % vision.PATCH != 0) {
            std.debug.print("smartResize({d},{d}) not patch-aligned: {d}x{d}\n", .{ wh[0], wh[1], s.w, s.h });
            return error.NotPatchAligned;
        }
        const n = (s.w / vision.PATCH) * (s.h / vision.PATCH);
        if (n < 1 or n > vision.MAX_TOKENS) {
            std.debug.print("smartResize({d},{d}) -> {d} tokens, outside [1, {d}]\n", .{ wh[0], wh[1], n, vision.MAX_TOKENS });
            return error.TokenBudgetViolated;
        }
        // The table only has 1120 entries per axis.
        if (s.w / vision.PATCH > 1120 or s.h / vision.PATCH > 1120) return error.PosTableOverflow;
    }

    // ── Patch geometry ──────────────────────────────────────────────
    //
    // Build an image whose every pixel encodes its own coordinates, then
    // assert each patch element lands where the HWC-interleaved layout
    // says it should. This is what catches a transposed patch grid or a
    // channel swap — both of which still "look like an image".
    {
        // 8x6 = 48 patches: already inside [40, 280] and patch-aligned,
        // so smartResize is the identity and any mismatch below is pure
        // layout rather than resampling. (A smaller grid would be scaled
        // UP to clear the 40-token floor.)
        const w: usize = vision.PATCH * 8;
        const h: usize = vision.PATCH * 6;
        const img = try gpa.alloc(u8, w * h * 3);
        defer gpa.free(img);
        for (0..h) |y| {
            for (0..w) |x| {
                img[(y * w + x) * 3 + 0] = @intCast(x % 251);
                img[(y * w + x) * 3 + 1] = @intCast(y % 251);
                img[(y * w + x) * 3 + 2] = @intCast((x + y) % 251);
            }
        }

        var p = try vision.preprocess(gpa, img, w, h);
        defer p.deinit();

        if (p.n_cols != 8 or p.n_rows != 6) {
            std.debug.print("patch grid {d}x{d}, want 8x6\n", .{ p.n_cols, p.n_rows });
            return error.PatchGridMismatch;
        }

        var max_err: f32 = 0;
        for (0..p.count()) |i| {
            const prow = p.rowOf(i);
            const pcol = p.colOf(i);
            const buf = p.patch(i);
            for (0..vision.PATCH) |r| {
                for (0..vision.PATCH) |c| {
                    const x = pcol * vision.PATCH + c;
                    const y = prow * vision.PATCH + r;
                    const want = [3]f32{
                        @as(f32, @floatFromInt(x % 251)) / 255.0,
                        @as(f32, @floatFromInt(y % 251)) / 255.0,
                        @as(f32, @floatFromInt((x + y) % 251)) / 255.0,
                    };
                    inline for (0..3) |ch| {
                        const got = buf[(r * vision.PATCH + c) * 3 + ch];
                        const e = @abs(got - want[ch]);
                        if (e > max_err) max_err = e;
                    }
                }
            }
        }
        if (max_err > 2e-3) {
            std.debug.print("patch layout mismatch: max |Δ| = {e}\n", .{max_err});
            return error.PatchLayoutMismatch;
        }
    }

    // Rescaling must land in [0,1] — a missed /255 would still produce
    // finite, plausible-looking activations downstream.
    {
        const w: usize = 200;
        const h: usize = 150;
        const img = try gpa.alloc(u8, w * h * 3);
        defer gpa.free(img);
        for (img, 0..) |*v, i| v.* = @intCast(i % 256);
        var p = try vision.preprocess(gpa, img, w, h);
        defer p.deinit();
        var lo: f32 = 1e9;
        var hi: f32 = -1e9;
        for (p.data) |v| {
            lo = @min(lo, v);
            hi = @max(hi, v);
        }
        if (lo < 0.0 or hi > 1.0) {
            std.debug.print("rescale out of range: [{d}, {d}]\n", .{ lo, hi });
            return error.RescaleOutOfRange;
        }
    }

    std.debug.print(
        "PASS vision preprocess (smartResize budget + patch-aligned over {d} shapes; HWC patch layout exact; rescale in [0,1])\n",
        .{shapes.len},
    );
}
