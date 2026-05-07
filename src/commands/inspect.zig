//! Read-only inspection subcommands: `--list`, `--inspect`, `--config`,
//! `--load`, `--dump-embed`. Extracted from main.zig as part of the
//! incremental modularization. Each entry point is `pub fn run*` taking
//! the allocator and (where relevant) a model dir / token id.

const std = @import("std");
const safetensors = @import("../safetensors.zig");
const model_mod = @import("../model.zig");
const config_mod = @import("../config.zig");
const dtype = @import("../dtype.zig");
const hf_cache = @import("../hf_cache.zig");

pub fn runList(gpa: std.mem.Allocator) !void {
    const stdout = std.io.getStdOut().writer();

    const root = try hf_cache.cacheRoot(gpa);
    defer gpa.free(root);
    try stdout.print("HF cache: {s}\n\n", .{root});

    const models = try hf_cache.listModels(gpa);
    defer {
        for (models) |entry| {
            var m = entry;
            m.deinit(gpa);
        }
        gpa.free(models);
    }
    if (models.len == 0) {
        try stdout.print("(no models cached — try `hf download <org/name>`)\n", .{});
        return;
    }

    // Sort: supported first, then by id alphabetically. Two-key sort
    // via std.mem.sort with a stable comparator.
    std.mem.sort(hf_cache.ModelInfo, @constCast(models), {}, struct {
        fn lt(_: void, a: hf_cache.ModelInfo, b: hf_cache.ModelInfo) bool {
            if (a.supported != b.supported) return a.supported;
            return std.mem.lessThan(u8, a.id, b.id);
        }
    }.lt);

    // Width-fit columns for readability.
    var max_id_w: usize = 5;
    var max_arch_w: usize = 12;
    for (models) |m| {
        if (m.id.len > max_id_w) max_id_w = m.id.len;
        if (m.architecture.len > max_arch_w) max_arch_w = m.architecture.len;
    }
    if (max_id_w > 60) max_id_w = 60;
    if (max_arch_w > 36) max_arch_w = 36;

    try stdout.print("  {s: <60}  {s: <36}  {s: >9}  {s}\n", .{
        "model id (`--chat <this>`)",
        "architecture",
        "size",
        "status",
    });
    try stdout.print("  {s:->60}  {s:->36}  {s:->9}  {s:->8}\n", .{ "", "", "", "" });

    var size_buf: [32]u8 = undefined;
    for (models) |m| {
        const size_str = try hf_cache.formatSize(m.bytes, &size_buf);
        const status: []const u8 = if (m.supported) "[OK]" else "[?]";
        try stdout.print("  {s: <60}  {s: <36}  {s: >9}  {s}\n", .{
            m.id,
            m.architecture,
            size_str,
            status,
        });
    }

    var supported_count: usize = 0;
    for (models) |m| if (m.supported) {
        supported_count += 1;
    };
    try stdout.print("\n{d}/{d} models supported by valkyr's loader.\n", .{ supported_count, models.len });
}

// ── inspect: dump the tensor inventory of a real .safetensors file ──

pub fn runInspect(allocator: std.mem.Allocator, path: []const u8) !void {
    const t0 = std.time.nanoTimestamp();
    var st = try safetensors.SafeTensors.open(allocator, path);
    defer st.deinit();
    const t1 = std.time.nanoTimestamp();
    const open_ms = @as(f64, @floatFromInt(t1 - t0)) / 1_000_000.0;

    const stdout = std.io.getStdOut().writer();
    try stdout.print("File: {s}\n", .{path});
    try stdout.print("Tensors: {d}    Open+parse: {d:.2} ms\n", .{ st.count(), open_ms });
    try stdout.print("Mapping: {d:.2} GiB\n", .{
        @as(f64, @floatFromInt(st.mapping.len)) / (1024.0 * 1024.0 * 1024.0),
    });
    try stdout.print("\n", .{});

    // Walk in alphabetical order so the output is reproducible across
    // hashmap iteration orders.
    var names = std.ArrayList([]const u8).init(allocator);
    defer names.deinit();
    var it = st.by_name.keyIterator();
    while (it.next()) |k| try names.append(k.*);
    std.mem.sort([]const u8, names.items, {}, struct {
        fn lessThan(_: void, a: []const u8, b: []const u8) bool {
            return std.mem.order(u8, a, b) == .lt;
        }
    }.lessThan);

    var totals = [_]u64{0} ** @typeInfo(safetensors.Dtype).@"enum".fields.len;
    for (names.items) |name| {
        const t = st.get(name).?;
        totals[@intFromEnum(t.dtype)] += t.bytes.len;
        try stdout.print("  {s:<10} {s:<55} ", .{ @tagName(t.dtype), name });
        try stdout.print("[", .{});
        for (t.shape, 0..) |d, i| {
            if (i > 0) try stdout.print(", ", .{});
            try stdout.print("{d}", .{d});
        }
        try stdout.print("]  {d:.2} MiB\n", .{
            @as(f64, @floatFromInt(t.bytes.len)) / (1024.0 * 1024.0),
        });
    }

    try stdout.print("\nDtype totals:\n", .{});
    inline for (@typeInfo(safetensors.Dtype).@"enum".fields) |f| {
        const idx: usize = f.value;
        if (totals[idx] > 0) {
            try stdout.print("  {s:<6} {d:.2} MiB\n", .{
                f.name,
                @as(f64, @floatFromInt(totals[idx])) / (1024.0 * 1024.0),
            });
        }
    }
}

pub fn runConfig(allocator: std.mem.Allocator, dir_path: []const u8) !void {
    // Parse `config.json` only — no safetensors loader. Useful for
    // validating new architectures' config plumbing in isolation before
    // wiring up the rest of the loader.
    const cfg_path = try std.fs.path.join(allocator, &.{ dir_path, "config.json" });
    defer allocator.free(cfg_path);
    const cfg = try config_mod.Config.loadFromFile(allocator, cfg_path);
    const stdout = std.io.getStdOut().writer();
    try stdout.print("Parsed {s}\n\n", .{cfg_path});
    try cfg.print(stdout);
    try stdout.print("\n[in_proj_qkv conv dim] {d}\n", .{cfg.linearAttnConvDim()});
    if (cfg.family.isHybrid()) {
        try stdout.print("[layer_types[0..8]]    ", .{});
        const n = @min(cfg.num_hidden_layers, 8);
        for (cfg.layer_types[0..n]) |t| try stdout.print("{s} ", .{@tagName(t)});
        try stdout.print("\n", .{});
    }
}

pub fn runLoad(allocator: std.mem.Allocator, dir_path: []const u8) !void {
    const t0 = std.time.nanoTimestamp();
    var model = try model_mod.Model.load(allocator, dir_path);
    defer model.deinit();
    const t1 = std.time.nanoTimestamp();
    const load_ms = @as(f64, @floatFromInt(t1 - t0)) / 1_000_000.0;

    const stdout = std.io.getStdOut().writer();
    try stdout.print("Loaded {s}\n", .{dir_path});
    try stdout.print("Load time: {d:.2} ms ({d} shard(s), {d} tensors)\n\n", .{
        load_ms, model.shards.shards.len, model.shards.count(),
    });
    try model.config.print(stdout);
    try stdout.print("\nLM head: {s}\n", .{
        if (model.isLmHeadTied()) "tied (shares embed_tokens)" else "separate (lm_head.weight)",
    });

    // Sanity: walk every layer once and confirm we can read the first
    // and last byte of each weight from the mmap. If a shard's offset
    // table is wrong, that page-faults; if it's right, this is free.
    // For hybrid models the per-layer fields differ between full and
    // linear-attention blocks — touch whichever is present.
    var bytes_touched: usize = 0;
    for (model.layers) |layer| {
        // FFN trio + the two LayerNorms are always present.
        inline for (.{
            "input_layernorm",          "post_attention_layernorm",
            "gate_proj",                "up_proj",                  "down_proj",
        }) |fname| {
            const t = @field(layer, fname);
            if (t.bytes.len > 0) {
                bytes_touched +%= @intCast(t.bytes[0]);
                bytes_touched +%= @intCast(t.bytes[t.bytes.len - 1]);
            }
        }
        // Optional per-flavor tensors. `inline for` on a tuple of field
        // names lets us walk them uniformly, dereference the optional,
        // and skip nulls.
        inline for (.{
            "q_proj", "k_proj", "v_proj", "o_proj", "q_norm", "k_norm",
            "in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a",
            "conv1d_weight", "A_log", "dt_bias", "ssm_norm_weight", "out_proj",
        }) |fname| {
            if (@field(layer, fname)) |t| {
                if (t.bytes.len > 0) {
                    bytes_touched +%= @intCast(t.bytes[0]);
                    bytes_touched +%= @intCast(t.bytes[t.bytes.len - 1]);
                }
            }
        }
    }
    bytes_touched +%= @intCast(model.embed_tokens.bytes[0]);
    bytes_touched +%= @intCast(model.final_norm.bytes[0]);
    bytes_touched +%= @intCast(model.lm_head.bytes[0]);
    // The xor folds the first/last byte of every weight into one word —
    // a cheap way to force the OS to actually touch each tensor page.
    // Print it so the optimizer can't elide the loop.
    try stdout.print("\nPASS load (touch checksum: 0x{x})\n", .{bytes_touched});
}

// ── dump-embed: pull a token's row from embed_tokens, bf16 → fp32 ────

pub fn runDumpEmbed(allocator: std.mem.Allocator, dir_path: []const u8, token_id: u32) !void {
    var model = try model_mod.Model.load(allocator, dir_path);
    defer model.deinit();

    const cfg = model.config;
    if (token_id >= cfg.vocab_size) {
        std.debug.print("token id {d} out of range (vocab_size={d})\n", .{ token_id, cfg.vocab_size });
        return error.OutOfRange;
    }

    if (model.embed_tokens.dtype != .bf16) {
        std.debug.print("expected bf16 embed_tokens; got {s}\n", .{@tagName(model.embed_tokens.dtype)});
        return error.UnexpectedDtype;
    }

    const stdout = std.io.getStdOut().writer();
    try stdout.print("token {d}, hidden_size={d} — first values + stats\n", .{ token_id, cfg.hidden_size });

    // Slice the row for this token. embed_tokens is [vocab_size, hidden_size]
    // row-major, so row `token_id` starts at `token_id * hidden_size *
    // bytes_per_elem` from the tensor's byte base.
    const elem_bytes = model.embed_tokens.dtype.elemSize();
    const row_bytes = cfg.hidden_size * elem_bytes;
    const row_start = @as(usize, token_id) * row_bytes;
    const row_bf16 = dtype.asU16(model.embed_tokens.bytes[row_start .. row_start + row_bytes]);

    // Convert into a heap-allocated fp32 buffer.
    const row_f32 = try allocator.alloc(f32, cfg.hidden_size);
    defer allocator.free(row_f32);
    dtype.bf16SliceToF32(row_bf16, row_f32);

    // First N for inspection.
    const n_show: usize = @min(16, row_f32.len);
    for (row_f32[0..n_show], 0..) |v, i| try stdout.print("  [{d:>4}] {d:.6}\n", .{ i, v });

    // Stats over the whole row.
    var min_v: f32 = std.math.inf(f32);
    var max_v: f32 = -std.math.inf(f32);
    var sum: f64 = 0;
    var nan_count: usize = 0;
    var inf_count: usize = 0;
    for (row_f32) |v| {
        if (std.math.isNan(v)) {
            nan_count += 1;
            continue;
        }
        if (std.math.isInf(v)) {
            inf_count += 1;
            continue;
        }
        if (v < min_v) min_v = v;
        if (v > max_v) max_v = v;
        sum += v;
    }
    const mean = sum / @as(f64, @floatFromInt(row_f32.len - nan_count - inf_count));
    try stdout.print("\nstats: min={d:.6} max={d:.6} mean={d:.6} nan={d} inf={d}\n", .{
        min_v, max_v, mean, nan_count, inf_count,
    });
}
