//! On-disk cache of GPU-ready weight bytes.
//!
//! Why this exists
//! ───────────────
//! Every upload path in `gpu/model.zig` ends in the same call:
//! `pool.commit(ctx, bytes)`. Producing those bytes is the expensive
//! part — Q4_K quantization of a 12B checkpoint costs ~150 s of host
//! CPU, dwarfing both the disk read and the PCIe transfer. The bytes
//! are a pure function of (source tensor, precision), so they are worth
//! memoizing.
//!
//! This module caches exactly those committed bytes, keyed by tensor
//! name. A hit turns "read 24 GB, convert to fp32, quantize, repack"
//! into "mmap and memcpy".
//!
//! File format
//! ───────────
//! Written sequentially (sizes aren't known upfront), so the index
//! lives at the end and a fixed-size footer points back at it:
//!
//!     [data]    every tensor's committed bytes, 16-byte aligned
//!     [index]   n × { name_len: u32, name: [name_len]u8,
//!                     offset: u64, len: u64 }
//!     [footer]  Footer (fixed size, magic last)
//!
//! Reading is: stat, read footer, validate, read index, mmap, slice.
//!
//! Invalidation
//! ────────────
//! A stale cache is worse than no cache — it would silently feed wrong
//! weights to a model that still runs. Guards, in order of cost:
//!
//!   - `MAGIC` + `FORMAT_VERSION` — bump the version whenever the
//!     on-device layout of any quantized format changes, or old caches
//!     will be honoured against new kernels.
//!   - `precision` — Q4_0 and Q4_K caches must never be confused.
//!   - source size + mtime — catches the checkpoint being replaced.
//!   - per-tensor name AND length — a hit must match both, so a
//!     reordered or reshaped model misses rather than mismatches.
//!
//! Anything unexpected (truncated file, bad magic, unreadable index) is
//! treated as a miss, never an error: the cache is an optimisation and
//! must never be the reason a model fails to load.

const std = @import("std");
const builtin = @import("builtin");

pub const MAGIC: u64 = 0x564C4B5257_514331; // "VLKRWQC1"
/// Bump on any change to what the quantizers emit, or to this format.
pub const FORMAT_VERSION: u32 = 1;

const ALIGN: u64 = 16;

const Footer = extern struct {
    index_offset: u64,
    n_entries: u64,
    source_size: u64,
    source_mtime_ns: u64,
    precision: u32,
    format_version: u32,
    magic: u64,
};

fn alignUp(v: u64) u64 {
    return (v + ALIGN - 1) & ~(ALIGN - 1);
}

/// Identity of the checkpoint a cache was built from. Two caches with
/// the same key hold interchangeable bytes.
pub const Key = struct {
    source_size: u64,
    source_mtime_ns: u64,
    precision: u32,

    /// Derive from the largest file in a model directory — in practice
    /// the weights themselves. Returns null when the directory can't be
    /// inspected, which callers treat as "don't cache".
    pub fn fromModelDir(dir_path: []const u8, precision: u32) ?Key {
        var dir = std.fs.cwd().openDir(dir_path, .{ .iterate = true }) catch return null;
        defer dir.close();
        var it = dir.iterate();
        var best_size: u64 = 0;
        var best_mtime: i128 = 0;
        while (it.next() catch return null) |entry| {
            if (entry.kind != .file and entry.kind != .sym_link) continue;
            if (!std.mem.endsWith(u8, entry.name, ".safetensors")) continue;
            const st = dir.statFile(entry.name) catch continue;
            if (st.size > best_size) {
                best_size = st.size;
                best_mtime = st.mtime;
            }
        }
        if (best_size == 0) return null;
        return .{
            .source_size = best_size,
            .source_mtime_ns = @intCast(@max(best_mtime, 0)),
            .precision = precision,
        };
    }
};

/// Absolute path of the cache file for `key` under the user's cache
/// directory. Caller owns the returned slice.
pub fn pathFor(gpa: std.mem.Allocator, key: Key, model_dir: []const u8) ![]u8 {
    const base = cacheRoot(gpa) catch null;
    defer if (base) |b| gpa.free(b);
    const root = base orelse return error.NoCacheDir;

    std.fs.cwd().makePath(root) catch {};

    // Hash the model path alongside the identity fields so two
    // checkpoints that happen to share a size never collide.
    var h = std.hash.Wyhash.init(0);
    h.update(model_dir);
    h.update(std.mem.asBytes(&key.source_size));
    h.update(std.mem.asBytes(&key.source_mtime_ns));
    h.update(std.mem.asBytes(&key.precision));

    return std.fmt.allocPrint(gpa, "{s}/w{x:0>16}.vkw", .{ root, h.final() });
}

fn cacheRoot(gpa: std.mem.Allocator) ![]u8 {
    if (std.process.getEnvVarOwned(gpa, "VALKYR_CACHE_DIR")) |v| return v else |_| {}
    if (std.process.getEnvVarOwned(gpa, "XDG_CACHE_HOME")) |v| {
        defer gpa.free(v);
        return std.fmt.allocPrint(gpa, "{s}/valkyr/weights", .{v});
    } else |_| {}
    const home = std.process.getEnvVarOwned(gpa, "HOME") catch return error.NoCacheDir;
    defer gpa.free(home);
    return std.fmt.allocPrint(gpa, "{s}/.cache/valkyr/weights", .{home});
}

// ── Reader ────────────────────────────────────────────────────────

pub const Reader = struct {
    file: std.fs.File,
    mapping: []align(std.heap.page_size_min) const u8,
    entries: std.StringHashMap(Entry),
    arena: std.heap.ArenaAllocator,

    pub const Entry = struct { offset: u64, len: u64 };

    /// Open and validate. Returns null on any problem — a cache that
    /// can't be trusted is simply a miss.
    pub fn open(gpa: std.mem.Allocator, path: []const u8, key: Key) ?Reader {
        const file = std.fs.cwd().openFile(path, .{ .mode = .read_only }) catch return null;
        var ok = false;
        defer if (!ok) file.close();

        const st = file.stat() catch return null;
        if (st.size < @sizeOf(Footer)) return null;

        var footer: Footer = undefined;
        file.seekTo(st.size - @sizeOf(Footer)) catch return null;
        _ = file.readAll(std.mem.asBytes(&footer)) catch return null;

        if (footer.magic != MAGIC) return null;
        if (footer.format_version != FORMAT_VERSION) return null;
        if (footer.precision != key.precision) return null;
        if (footer.source_size != key.source_size) return null;
        if (footer.source_mtime_ns != key.source_mtime_ns) return null;
        if (footer.index_offset >= st.size) return null;

        const mapping: []align(std.heap.page_size_min) const u8 = std.posix.mmap(
            null,
            @intCast(st.size),
            std.posix.PROT.READ,
            .{ .TYPE = .PRIVATE },
            file.handle,
            0,
        ) catch return null;
        var mapped = true;
        defer if (!ok and mapped) std.posix.munmap(mapping);

        var arena = std.heap.ArenaAllocator.init(gpa);
        var arena_ok = false;
        defer if (!arena_ok) arena.deinit();
        const a = arena.allocator();

        var entries = std.StringHashMap(Entry).init(gpa);
        var entries_ok = false;
        defer if (!entries_ok) entries.deinit();

        var p: usize = @intCast(footer.index_offset);
        const idx_end: usize = @intCast(st.size - @sizeOf(Footer));
        var i: u64 = 0;
        while (i < footer.n_entries) : (i += 1) {
            if (p + 4 > idx_end) return null;
            const name_len = std.mem.readInt(u32, mapping[p..][0..4], .little);
            p += 4;
            if (p + name_len + 16 > idx_end) return null;
            const name = a.dupe(u8, mapping[p .. p + name_len]) catch return null;
            p += name_len;
            const offset = std.mem.readInt(u64, mapping[p..][0..8], .little);
            p += 8;
            const len = std.mem.readInt(u64, mapping[p..][0..8], .little);
            p += 8;
            if (offset + len > footer.index_offset) return null;
            entries.put(name, .{ .offset = offset, .len = len }) catch return null;
        }

        ok = true;
        arena_ok = true;
        entries_ok = true;
        mapped = true;
        return .{ .file = file, .mapping = mapping, .entries = entries, .arena = arena };
    }

    /// Cached bytes for `name`, or null. `expect_len` guards against a
    /// same-named tensor whose shape changed — a length mismatch is a
    /// miss, not a silent wrong answer.
    pub fn get(self: *const Reader, name: []const u8, expect_len: ?usize) ?[]const u8 {
        if (name.len == 0) return null;
        const e = self.entries.get(name) orelse return null;
        if (expect_len) |want| {
            if (e.len != want) return null;
        }
        const start: usize = @intCast(e.offset);
        const end: usize = @intCast(e.offset + e.len);
        if (end > self.mapping.len) return null;
        return self.mapping[start..end];
    }

    pub fn deinit(self: *Reader) void {
        self.entries.deinit();
        self.arena.deinit();
        std.posix.munmap(self.mapping);
        self.file.close();
    }
};

// ── Writer ────────────────────────────────────────────────────────

pub const Writer = struct {
    file: std.fs.File,
    tmp_path: []u8,
    final_path: []u8,
    gpa: std.mem.Allocator,
    offset: u64 = 0,
    index: std.ArrayList(u8),
    n_entries: u64 = 0,
    key: Key,
    failed: bool = false,

    /// Begin writing to a temporary sibling of `path`. Returns null if
    /// the file can't be created — callers then simply don't cache.
    pub fn create(gpa: std.mem.Allocator, path: []const u8, key: Key) ?Writer {
        const tmp = std.fmt.allocPrint(gpa, "{s}.tmp{d}", .{ path, std.os.linux.getpid() }) catch return null;
        var tmp_ok = false;
        defer if (!tmp_ok) gpa.free(tmp);

        const final = gpa.dupe(u8, path) catch return null;
        var final_ok = false;
        defer if (!final_ok) gpa.free(final);

        const file = std.fs.cwd().createFile(tmp, .{ .truncate = true }) catch return null;
        tmp_ok = true;
        final_ok = true;
        return .{
            .file = file,
            .tmp_path = tmp,
            .final_path = final,
            .gpa = gpa,
            .index = std.ArrayList(u8).init(gpa),
            .key = key,
        };
    }

    /// Append one tensor's committed bytes. Failures are recorded and
    /// the cache is abandoned at `finish` — never propagated, since a
    /// full disk must not stop a model from loading.
    pub fn put(self: *Writer, name: []const u8, bytes: []const u8) void {
        if (self.failed or name.len == 0) return;
        const padded = alignUp(self.offset);
        if (padded != self.offset) {
            var pad = [_]u8{0} ** ALIGN;
            const n: usize = @intCast(padded - self.offset);
            self.file.writeAll(pad[0..n]) catch {
                self.failed = true;
                return;
            };
            self.offset = padded;
        }
        self.file.writeAll(bytes) catch {
            self.failed = true;
            return;
        };

        var hdr: [4]u8 = undefined;
        std.mem.writeInt(u32, &hdr, @intCast(name.len), .little);
        self.index.appendSlice(&hdr) catch {
            self.failed = true;
            return;
        };
        self.index.appendSlice(name) catch {
            self.failed = true;
            return;
        };
        var off: [8]u8 = undefined;
        std.mem.writeInt(u64, &off, self.offset, .little);
        self.index.appendSlice(&off) catch {
            self.failed = true;
            return;
        };
        std.mem.writeInt(u64, &off, bytes.len, .little);
        self.index.appendSlice(&off) catch {
            self.failed = true;
            return;
        };

        self.offset += bytes.len;
        self.n_entries += 1;
    }

    /// Write index + footer and atomically rename into place. On any
    /// failure the temporary file is removed and no cache is published,
    /// so a partial write can never be mistaken for a complete one.
    pub fn finish(self: *Writer) void {
        defer {
            self.index.deinit();
            self.gpa.free(self.tmp_path);
            self.gpa.free(self.final_path);
        }
        if (self.failed or self.n_entries == 0) {
            self.file.close();
            std.fs.cwd().deleteFile(self.tmp_path) catch {};
            return;
        }

        const index_offset = self.offset;
        const footer = Footer{
            .index_offset = index_offset,
            .n_entries = self.n_entries,
            .source_size = self.key.source_size,
            .source_mtime_ns = self.key.source_mtime_ns,
            .precision = self.key.precision,
            .format_version = FORMAT_VERSION,
            .magic = MAGIC,
        };

        const wrote = blk: {
            self.file.writeAll(self.index.items) catch break :blk false;
            self.file.writeAll(std.mem.asBytes(&footer)) catch break :blk false;
            break :blk true;
        };
        self.file.close();

        if (!wrote) {
            std.fs.cwd().deleteFile(self.tmp_path) catch {};
            return;
        }
        std.fs.cwd().rename(self.tmp_path, self.final_path) catch {
            std.fs.cwd().deleteFile(self.tmp_path) catch {};
        };
    }
};
