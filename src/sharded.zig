//! Multi-file SafeTensors loading.
//!
//! HuggingFace splits big checkpoints across multiple .safetensors files
//! and writes a `model.safetensors.index.json` index that maps each
//! tensor name to its shard file. We open every referenced shard and
//! present a single name → Tensor view to callers; the underlying
//! mappings stay separate (one mmap per shard) but the lookup surface
//! is uniform.
//!
//! Single-file checkpoints (no index.json, just `model.safetensors`)
//! also flow through this type — they're just a one-shard case. That
//! way the Model loader doesn't have to branch on shape-sharding at
//! every call site.

const std = @import("std");
const safetensors = @import("safetensors.zig");

pub const Shards = struct {
    /// One mapping per file. Owned. Order is the order shards were
    /// opened, which has no semantic meaning — the merged `by_name`
    /// table is what callers should use.
    shards: []safetensors.SafeTensors,
    /// Merged name → tensor view. Each value is a copy of the entry
    /// from the owning shard; the byte slice still points into that
    /// shard's mmap.
    by_name: std.StringHashMap(safetensors.Tensor),
    allocator: std.mem.Allocator,

    /// Open a model directory. Tries the sharded layout first
    /// (`model.safetensors.index.json` + shard files it references),
    /// falls back to a single `model.safetensors`.
    pub fn openFromDir(allocator: std.mem.Allocator, dir_path: []const u8) !Shards {
        // Compose paths under dir_path. Bounded buffer is fine — model
        // dirs aren't that deep and we cap at PATH_MAX-ish.
        var path_buf: [std.fs.max_path_bytes]u8 = undefined;

        const index_path = try std.fmt.bufPrint(&path_buf, "{s}/model.safetensors.index.json", .{dir_path});
        const index_exists = blk: {
            std.fs.cwd().access(index_path, .{}) catch break :blk false;
            break :blk true;
        };

        if (index_exists) {
            return openFromIndex(allocator, dir_path, index_path);
        }

        const single_path = try std.fmt.bufPrint(&path_buf, "{s}/model.safetensors", .{dir_path});
        return openFromPaths(allocator, &.{single_path});
    }

    /// Open an explicit list of shard paths. Useful for tests and for
    /// the rare case where you want to point at a non-standard layout.
    pub fn openFromPaths(allocator: std.mem.Allocator, paths: []const []const u8) !Shards {
        const shards = try allocator.alloc(safetensors.SafeTensors, paths.len);
        var opened: usize = 0;
        errdefer {
            var i: usize = 0;
            while (i < opened) : (i += 1) shards[i].deinit();
            allocator.free(shards);
        }
        for (paths) |p| {
            shards[opened] = try safetensors.SafeTensors.open(allocator, p);
            opened += 1;
        }

        var by_name = std.StringHashMap(safetensors.Tensor).init(allocator);
        errdefer by_name.deinit();
        for (shards) |*s| {
            var it = s.by_name.iterator();
            while (it.next()) |e| {
                // Names are unique across shards in a well-formed model;
                // if we hit a duplicate, last-writer-wins. We don't error
                // because a quantized model could legally re-emit a name
                // with new metadata, and we'd rather honour the explicit
                // path order than reject the whole load.
                try by_name.put(e.key_ptr.*, e.value_ptr.*);
            }
        }

        return .{
            .shards = shards,
            .by_name = by_name,
            .allocator = allocator,
        };
    }

    /// Read `index_path`, collect the unique shard filenames referenced
    /// in its `weight_map`, open all of them.
    fn openFromIndex(allocator: std.mem.Allocator, dir_path: []const u8, index_path: []const u8) !Shards {
        const file = try std.fs.cwd().openFile(index_path, .{ .mode = .read_only });
        defer file.close();
        const bytes = try file.readToEndAlloc(allocator, 16 * 1024 * 1024); // index files are small
        defer allocator.free(bytes);

        var parsed = try std.json.parseFromSlice(std.json.Value, allocator, bytes, .{});
        defer parsed.deinit();
        if (parsed.value != .object) return error.IndexNotObject;
        const wm_v = parsed.value.object.get("weight_map") orelse return error.IndexMissingWeightMap;
        if (wm_v != .object) return error.IndexNotObject;

        // Collect unique shard filenames. Use a string set because the
        // weight_map is name → file and many names share a file.
        var seen = std.StringHashMap(void).init(allocator);
        defer {
            var it = seen.keyIterator();
            while (it.next()) |k| allocator.free(k.*);
            seen.deinit();
        }
        var it = wm_v.object.iterator();
        while (it.next()) |e| {
            if (e.value_ptr.* != .string) continue;
            if (seen.contains(e.value_ptr.string)) continue;
            const owned = try allocator.dupe(u8, e.value_ptr.string);
            try seen.put(owned, {});
        }

        // Build the path list under dir_path.
        var paths_storage = std.ArrayList([]u8).init(allocator);
        defer {
            for (paths_storage.items) |p| allocator.free(p);
            paths_storage.deinit();
        }
        var seen_it = seen.keyIterator();
        while (seen_it.next()) |k| {
            const composed = try std.fmt.allocPrint(allocator, "{s}/{s}", .{ dir_path, k.* });
            try paths_storage.append(composed);
        }

        // openFromPaths takes []const []const u8; build a const view.
        var paths_view = try allocator.alloc([]const u8, paths_storage.items.len);
        defer allocator.free(paths_view);
        for (paths_storage.items, 0..) |p, i| paths_view[i] = p;

        return openFromPaths(allocator, paths_view);
    }

    pub fn get(self: *const Shards, name: []const u8) ?safetensors.Tensor {
        return self.by_name.get(name);
    }

    pub fn count(self: *const Shards) usize {
        return self.by_name.count();
    }

    pub fn deinit(self: *Shards) void {
        self.by_name.deinit();
        for (self.shards) |*s| s.deinit();
        self.allocator.free(self.shards);
    }
};
