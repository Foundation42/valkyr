//! Gemma 4 "unified" vision path — image → soft tokens.
//!
//! Gemma 4 Unified has no vision transformer. The entire image path is
//! a linear patch embedder, ten tensors total, and the main transformer
//! does the actual seeing via bidirectional attention over the image
//! span. That makes this file small: preprocessing plus six ops.
//!
//! Pipeline, per 48×48 RGB patch (matches llama.cpp's
//! `clip_graph_gemma4uv::build`):
//!
//!     LayerNorm(patch_ln1, eps 1e-5)        [6912]
//!     matmul(patch_dense) + bias            [6912 → 3840]
//!     LayerNorm(patch_ln2, eps 1e-5)        [3840]
//!     + pos_embedding[col][0] + pos_embedding[row][1]
//!     LayerNorm(pos_norm,  eps 1e-5)        [3840]
//!     RMSNorm, no learned gain, eps 1e-6
//!     matmul(embedding_projection)          [3840 → 3840]
//!
//! Details that are easy to get wrong, all verified against the
//! reference rather than assumed:
//!
//!   - These are real **LayerNorms** (mean subtracted, learned bias),
//!     not the RMSNorms used throughout the text stack, and their eps
//!     is PyTorch's **1e-5** default — not the model's 1e-6
//!     `rms_norm_eps`.
//!   - The final pre-projection norm IS an RMSNorm, and it is
//!     **weightless** (`ggml_rms_norm` with no gain), at eps 1e-6.
//!   - Patch elements are laid out **HWC-interleaved**
//!     (`(r*48 + c)*3 + ch`), which is the checkpoint's native column
//!     order for `patch_dense` and `patch_ln1`. llama.cpp permutes
//!     those two tensors to CHW purely because ggml's `im2col` emits
//!     CHW; writing our own patchifier means we can use the weights
//!     exactly as stored.
//!   - `pos_embedding` is [1120, 2, 3840] as stored: index
//!     `[pos][axis][dim]` with axis 0 keyed by **column** and axis 1 by
//!     **row**. Both are added. (llama.cpp permutes to [2,1120,3840];
//!     we read the native layout directly.)
//!
//! Preprocessing is unusually simple for a vision model: convert to
//! RGB, rescale by 1/255, resize. `do_normalize` is false and the
//! mean/std are 0/1, so there is no ImageNet normalisation to get
//! subtly wrong.

const std = @import("std");

/// Effective patch size. The config advertises `patch_size: 16` with
/// `pooling_kernel_size: 3`, but the unified variant folds the pooling
/// into a single larger patch, so the real unit is 48×48.
pub const PATCH: usize = 48;
pub const CHANNELS: usize = 3;
/// Flattened patch length: 48 × 48 × 3.
pub const PATCH_ELEMS: usize = PATCH * PATCH * CHANNELS;

/// Soft-token budget per image. The lower bound is not arbitrary — the
/// reference notes the model performs poorly on very small images and
/// clamps up to 40 tokens.
pub const MIN_TOKENS: usize = 40;
pub const MAX_TOKENS: usize = 280;

/// LayerNorm epsilon for the three patch norms. PyTorch's default, and
/// deliberately NOT the model's `rms_norm_eps`.
pub const LN_EPS: f32 = 1e-5;
/// Epsilon for the weightless pre-projection RMSNorm.
pub const RMS_EPS: f32 = 1e-6;

pub const Size = struct { w: usize, h: usize };

fn roundBy(x: f32, f: usize) usize {
    const ff: f32 = @floatFromInt(f);
    return @as(usize, @intFromFloat(@round(x / ff))) * f;
}
fn ceilBy(x: f32, f: usize) usize {
    const ff: f32 = @floatFromInt(f);
    return @as(usize, @intFromFloat(@ceil(x / ff))) * f;
}
fn floorBy(x: f32, f: usize) usize {
    const ff: f32 = @floatFromInt(f);
    return @as(usize, @intFromFloat(@floor(x / ff))) * f;
}

/// "Smart resize": the largest patch-aligned size preserving aspect
/// ratio whose patch count lands inside [MIN_TOKENS, MAX_TOKENS].
///
/// Mirrors `img_tool::calc_size_preserved_ratio` in the reference, which
/// in turn mirrors transformers' `smart_resize`. Note it aligns first
/// and only then corrects for the pixel budget, so the ordering matters:
/// aligning after scaling can push the count back over the cap.
pub fn smartResize(w: usize, h: usize) Size {
    std.debug.assert(w > 0 and h > 0);
    const min_pixels = MIN_TOKENS * PATCH * PATCH;
    const max_pixels = MAX_TOKENS * PATCH * PATCH;

    const wf: f32 = @floatFromInt(w);
    const hf: f32 = @floatFromInt(h);

    var w_bar = @max(PATCH, roundBy(wf, PATCH));
    var h_bar = @max(PATCH, roundBy(hf, PATCH));

    if (w_bar * h_bar > max_pixels) {
        const beta = @sqrt(wf * hf / @as(f32, @floatFromInt(max_pixels)));
        w_bar = @max(PATCH, floorBy(wf / beta, PATCH));
        h_bar = @max(PATCH, floorBy(hf / beta, PATCH));
    } else if (w_bar * h_bar < min_pixels) {
        const beta = @sqrt(@as(f32, @floatFromInt(min_pixels)) / (wf * hf));
        w_bar = ceilBy(wf * beta, PATCH);
        h_bar = ceilBy(hf * beta, PATCH);
    }
    return .{ .w = w_bar, .h = h_bar };
}

/// Patchified image, ready for the embedder.
///
/// `data` is `[n_patches][PATCH_ELEMS]` with patches in row-major order
/// (`i = row * n_cols + col`) and each patch HWC-interleaved. Values are
/// already rescaled to [0, 1].
pub const Patches = struct {
    data: []f32,
    n_cols: usize,
    n_rows: usize,
    gpa: std.mem.Allocator,

    pub fn count(self: Patches) usize {
        return self.n_cols * self.n_rows;
    }
    pub fn patch(self: Patches, i: usize) []const f32 {
        return self.data[i * PATCH_ELEMS ..][0..PATCH_ELEMS];
    }
    /// Column index of patch `i` — the key into pos_embedding axis 0.
    pub fn colOf(self: Patches, i: usize) usize {
        return i % self.n_cols;
    }
    /// Row index of patch `i` — the key into pos_embedding axis 1.
    pub fn rowOf(self: Patches, i: usize) usize {
        return i / self.n_cols;
    }
    pub fn deinit(self: *Patches) void {
        self.gpa.free(self.data);
    }
};

/// Bilinear resample of an 8-bit RGB image into patch-flattened f32,
/// rescaled by 1/255.
///
/// Resize and patchify are fused: the destination pixel a patch element
/// needs is computed directly, so no full resized image is materialised.
pub fn preprocess(
    gpa: std.mem.Allocator,
    rgb: []const u8,
    src_w: usize,
    src_h: usize,
) !Patches {
    if (src_w == 0 or src_h == 0) return error.EmptyImage;
    if (rgb.len < src_w * src_h * CHANNELS) return error.ShortImageBuffer;

    const target = smartResize(src_w, src_h);
    const n_cols = target.w / PATCH;
    const n_rows = target.h / PATCH;
    const n_patches = n_cols * n_rows;
    std.debug.assert(n_patches >= 1);

    const data = try gpa.alloc(f32, n_patches * PATCH_ELEMS);
    errdefer gpa.free(data);

    // Map destination pixel centres back into source space. Using
    // centres (the +0.5 terms) rather than corners keeps the sampling
    // symmetric; corner-aligned mapping biases everything half a pixel
    // toward the origin and shows up as a slight crop.
    const scale_x = @as(f32, @floatFromInt(src_w)) / @as(f32, @floatFromInt(target.w));
    const scale_y = @as(f32, @floatFromInt(src_h)) / @as(f32, @floatFromInt(target.h));

    for (0..n_rows) |prow| {
        for (0..n_cols) |pcol| {
            const p = prow * n_cols + pcol;
            const out = data[p * PATCH_ELEMS ..][0..PATCH_ELEMS];
            for (0..PATCH) |r| {
                const dy = prow * PATCH + r;
                const sy = (@as(f32, @floatFromInt(dy)) + 0.5) * scale_y - 0.5;
                const y0f = @floor(sy);
                const wy = sy - y0f;
                const y0 = clampIdx(y0f, src_h);
                const y1 = clampIdx(y0f + 1, src_h);
                for (0..PATCH) |c| {
                    const dx = pcol * PATCH + c;
                    const sx = (@as(f32, @floatFromInt(dx)) + 0.5) * scale_x - 0.5;
                    const x0f = @floor(sx);
                    const wx = sx - x0f;
                    const x0 = clampIdx(x0f, src_w);
                    const x1 = clampIdx(x0f + 1, src_w);

                    inline for (0..CHANNELS) |ch| {
                        const p00: f32 = @floatFromInt(rgb[(y0 * src_w + x0) * CHANNELS + ch]);
                        const p01: f32 = @floatFromInt(rgb[(y0 * src_w + x1) * CHANNELS + ch]);
                        const p10: f32 = @floatFromInt(rgb[(y1 * src_w + x0) * CHANNELS + ch]);
                        const p11: f32 = @floatFromInt(rgb[(y1 * src_w + x1) * CHANNELS + ch]);
                        const top = p00 + (p01 - p00) * wx;
                        const bot = p10 + (p11 - p10) * wx;
                        // HWC-interleaved within the patch — the
                        // checkpoint's native column order.
                        out[(r * PATCH + c) * CHANNELS + ch] = (top + (bot - top) * wy) / 255.0;
                    }
                }
            }
        }
    }

    return .{ .data = data, .n_cols = n_cols, .n_rows = n_rows, .gpa = gpa };
}

fn clampIdx(v: f32, n: usize) usize {
    if (v <= 0) return 0;
    const i: usize = @intFromFloat(v);
    return @min(i, n - 1);
}

// ── CPU oracle ────────────────────────────────────────────────────

/// Vision-embedder weights as flat fp32, in checkpoint layout.
pub const Weights = struct {
    /// [PATCH_ELEMS]
    ln1_w: []const f32,
    ln1_b: []const f32,
    /// [embed_dim, PATCH_ELEMS], row-major (one row per output channel).
    dense_w: []const f32,
    /// [embed_dim]
    dense_b: []const f32,
    ln2_w: []const f32,
    ln2_b: []const f32,
    /// [pos_size, 2, embed_dim] — `[pos][axis][dim]`, axis 0 = column,
    /// axis 1 = row.
    pos_emb: []const f32,
    pos_size: usize,
    pos_norm_w: []const f32,
    pos_norm_b: []const f32,
    /// [embed_dim, embed_dim], row-major.
    proj_w: []const f32,
    embed_dim: usize,
};

fn layerNorm(out: []f32, in: []const f32, w: []const f32, b: []const f32, eps: f32) void {
    const n = in.len;
    var mean: f64 = 0;
    for (in) |v| mean += v;
    mean /= @floatFromInt(n);
    var variance: f64 = 0;
    for (in) |v| {
        const d = @as(f64, v) - mean;
        variance += d * d;
    }
    variance /= @floatFromInt(n);
    const inv = 1.0 / @sqrt(variance + @as(f64, eps));
    for (0..n) |i| {
        const norm = (@as(f64, in[i]) - mean) * inv;
        out[i] = @floatCast(norm * @as(f64, w[i]) + @as(f64, b[i]));
    }
}

fn rmsNormWeightless(out: []f32, in: []const f32, eps: f32) void {
    var ss: f64 = 0;
    for (in) |v| ss += @as(f64, v) * @as(f64, v);
    const inv = 1.0 / @sqrt(ss / @as(f64, @floatFromInt(in.len)) + @as(f64, eps));
    for (0..in.len) |i| out[i] = @floatCast(@as(f64, in[i]) * inv);
}

fn matvec(out: []f32, w: []const f32, x: []const f32, rows: usize, cols: usize) void {
    for (0..rows) |r| {
        var acc: f64 = 0;
        const row = w[r * cols ..][0..cols];
        for (0..cols) |c| acc += @as(f64, row[c]) * @as(f64, x[c]);
        out[r] = @floatCast(acc);
    }
}

/// Reference implementation of the whole embedder for one image.
/// `out` is `[n_patches][embed_dim]`. This is the oracle the GPU path
/// is checked against.
pub fn embedCpu(
    gpa: std.mem.Allocator,
    patches: Patches,
    wts: Weights,
    out: []f32,
) !void {
    const d = wts.embed_dim;
    const n = patches.count();
    std.debug.assert(out.len == n * d);

    const t1 = try gpa.alloc(f32, PATCH_ELEMS);
    defer gpa.free(t1);
    const t2 = try gpa.alloc(f32, d);
    defer gpa.free(t2);
    const t3 = try gpa.alloc(f32, d);
    defer gpa.free(t3);

    for (0..n) |i| {
        layerNorm(t1, patches.patch(i), wts.ln1_w, wts.ln1_b, LN_EPS);
        matvec(t2, wts.dense_w, t1, d, PATCH_ELEMS);
        for (0..d) |j| t2[j] += wts.dense_b[j];
        layerNorm(t3, t2, wts.ln2_w, wts.ln2_b, LN_EPS);

        // Two factorised lookups, both added: axis 0 by column, axis 1
        // by row. Out-of-range positions would mean an image wider or
        // taller than the table supports.
        const col = patches.colOf(i);
        const row = patches.rowOf(i);
        std.debug.assert(col < wts.pos_size and row < wts.pos_size);
        const ex = wts.pos_emb[(col * 2 + 0) * d ..][0..d];
        const ey = wts.pos_emb[(row * 2 + 1) * d ..][0..d];
        for (0..d) |j| t3[j] += ex[j] + ey[j];

        layerNorm(t2, t3, wts.pos_norm_w, wts.pos_norm_b, LN_EPS);
        rmsNormWeightless(t3, t2, RMS_EPS);
        matvec(out[i * d ..][0..d], wts.proj_w, t3, d, d);
    }
}

// ── Image input ───────────────────────────────────────────────────

pub const Image = struct {
    rgb: []u8,
    w: usize,
    h: usize,
    gpa: std.mem.Allocator,

    pub fn deinit(self: *Image) void {
        self.gpa.free(self.rgb);
    }
};

/// Load a binary PPM (P6) as 8-bit RGB.
///
/// Deliberately the only decoder here. The demo's real input is a raw
/// framebuffer — already RGB, no decoding — and PPM covers testing
/// without vendoring a JPEG/PNG library for a format the engine will
/// never produce. `convert in.jpg out.ppm` bridges the gap.
pub fn loadPpm(gpa: std.mem.Allocator, path: []const u8) !Image {
    const file = try std.fs.cwd().openFile(path, .{ .mode = .read_only });
    defer file.close();
    const bytes = try file.readToEndAlloc(gpa, 512 * 1024 * 1024);
    defer gpa.free(bytes);

    var i: usize = 0;
    // P6 magic, then width, height, maxval — whitespace-separated, with
    // '#' comments legal between any two tokens.
    if (bytes.len < 2 or bytes[0] != 'P' or bytes[1] != '6') return error.NotBinaryPpm;
    i = 2;

    var fields: [3]usize = undefined;
    var got: usize = 0;
    while (got < 3) {
        while (i < bytes.len and std.ascii.isWhitespace(bytes[i])) i += 1;
        if (i < bytes.len and bytes[i] == '#') {
            while (i < bytes.len and bytes[i] != '\n') i += 1;
            continue;
        }
        var v: usize = 0;
        var any = false;
        while (i < bytes.len and std.ascii.isDigit(bytes[i])) {
            v = v * 10 + (bytes[i] - '0');
            i += 1;
            any = true;
        }
        if (!any) return error.MalformedPpmHeader;
        fields[got] = v;
        got += 1;
    }
    // Exactly one whitespace byte separates the header from the data.
    if (i >= bytes.len) return error.TruncatedPpm;
    i += 1;

    const w = fields[0];
    const h = fields[1];
    const maxval = fields[2];
    if (w == 0 or h == 0) return error.EmptyImage;
    if (maxval != 255) return error.UnsupportedPpmMaxval;
    const need = w * h * CHANNELS;
    if (bytes.len - i < need) return error.TruncatedPpm;

    const rgb = try gpa.alloc(u8, need);
    @memcpy(rgb, bytes[i .. i + need]);
    return .{ .rgb = rgb, .w = w, .h = h, .gpa = gpa };
}

/// Where an image's soft tokens sit in the prompt, in absolute
/// positions. `len == 0` means "no image".
pub const Span = struct {
    start: usize = 0,
    len: usize = 0,

    /// Overlap of this span with the batch covering
    /// `[pos_start, pos_start + n)`, as (row within the batch, row
    /// within the image, count). Zero count when they don't intersect.
    pub fn overlap(self: Span, pos_start: usize, n: usize) struct { dst_row: usize, src_row: usize, count: usize } {
        if (self.len == 0) return .{ .dst_row = 0, .src_row = 0, .count = 0 };
        const lo = @max(self.start, pos_start);
        const hi = @min(self.start + self.len, pos_start + n);
        if (hi <= lo) return .{ .dst_row = 0, .src_row = 0, .count = 0 };
        return .{ .dst_row = lo - pos_start, .src_row = lo - self.start, .count = hi - lo };
    }
};
