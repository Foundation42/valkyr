//! SafeTensors reader.
//!
//! Layout on disk:
//!
//!     [u64 LE: header_len] [header_len bytes: UTF-8 JSON] [raw tensor bytes]
//!
//! The JSON header is a flat object: tensor name → metadata. Each entry
//! has `dtype`, `shape`, and `data_offsets: [start, end]` where the
//! offsets are relative to the *end* of the header. There may also be a
//! `__metadata__` key with arbitrary string→string pairs that we ignore.
//!
//! We mmap the whole file (PROT_READ, MAP_PRIVATE) and hand callers
//! slices that point straight into the mapping. Loading a 5 GB model is
//! therefore an O(header) parse plus zero copies — the kernel pages the
//! weights in lazily as the GPU staging upload reads them. This is the
//! same pattern the C reference uses with its `--ram` mode, except it's
//! always-on here because we're never going to read every byte off disk
//! into a host-allocated buffer.

const std = @import("std");
const builtin = @import("builtin");

// Windows file-mapping APIs. Zig 0.14.1's std.os.windows.kernel32 doesn't
// expose CreateFileMappingW / MapViewOfFile / UnmapViewOfFile yet, so we
// declare them ourselves. Gated by `os.tag == .windows` so the POSIX build
// never sees the kernel32 link request.
const win = if (builtin.os.tag == .windows) struct {
    const w = std.os.windows;
    pub const PAGE_READONLY: w.DWORD = 0x02;
    pub const FILE_MAP_READ: w.DWORD = 0x04;
    pub extern "kernel32" fn CreateFileMappingW(
        hFile: w.HANDLE,
        lpAttributes: ?*anyopaque,
        flProtect: w.DWORD,
        dwMaximumSizeHigh: w.DWORD,
        dwMaximumSizeLow: w.DWORD,
        lpName: ?w.LPCWSTR,
    ) callconv(w.WINAPI) ?w.HANDLE;
    pub extern "kernel32" fn MapViewOfFile(
        hFileMappingObject: w.HANDLE,
        dwDesiredAccess: w.DWORD,
        dwFileOffsetHigh: w.DWORD,
        dwFileOffsetLow: w.DWORD,
        dwNumberOfBytesToMap: w.SIZE_T,
    ) callconv(w.WINAPI) ?w.LPVOID;
    pub extern "kernel32" fn UnmapViewOfFile(
        lpBaseAddress: w.LPCVOID,
    ) callconv(w.WINAPI) w.BOOL;
} else struct {};

pub const Dtype = enum {
    f32,
    f16,
    bf16,
    i64,
    i32,
    i16,
    i8,
    u8,
    bool_,

    pub fn fromString(s: []const u8) !Dtype {
        if (std.mem.eql(u8, s, "F32")) return .f32;
        if (std.mem.eql(u8, s, "F16")) return .f16;
        if (std.mem.eql(u8, s, "BF16")) return .bf16;
        if (std.mem.eql(u8, s, "I64")) return .i64;
        if (std.mem.eql(u8, s, "I32")) return .i32;
        if (std.mem.eql(u8, s, "I16")) return .i16;
        if (std.mem.eql(u8, s, "I8")) return .i8;
        if (std.mem.eql(u8, s, "U8")) return .u8;
        if (std.mem.eql(u8, s, "BOOL")) return .bool_;
        return error.UnsupportedDtype;
    }

    pub fn elemSize(self: Dtype) usize {
        return switch (self) {
            .f32, .i32 => 4,
            .f16, .bf16, .i16 => 2,
            .i64 => 8,
            .i8, .u8, .bool_ => 1,
        };
    }
};

pub const Tensor = struct {
    dtype: Dtype,
    /// Lifetime tied to the parent SafeTensors arena.
    shape: []const usize,
    /// Slice into the mmap'd region. Read-only — the mapping is
    /// PROT_READ. Length equals `numel * dtype.elemSize()`.
    bytes: []const u8,

    pub fn numel(self: Tensor) usize {
        var n: usize = 1;
        for (self.shape) |d| n *= d;
        return n;
    }

    /// View as `[]align(1) const f32`. The SafeTensors data section is
    /// aligned only to whatever the writer chose (HuggingFace pads JSON
    /// to 8 bytes; smaller producers may not align at all), so we don't
    /// require natural alignment. Modern x86/ARM64 handle unaligned loads
    /// at no measurable cost, and we'll memcpy into a staging buffer
    /// before the GPU sees the bytes anyway.
    pub fn asF32(self: Tensor) []align(1) const f32 {
        std.debug.assert(self.dtype == .f32);
        std.debug.assert(self.bytes.len % @sizeOf(f32) == 0);
        return @as([*]align(1) const f32, @ptrCast(self.bytes.ptr))[0 .. self.bytes.len / @sizeOf(f32)];
    }
};

pub const SafeTensors = struct {
    /// File kept open for the lifetime of the mapping; closed on deinit.
    file: std.fs.File,
    /// The full file mapped read-only. Slice covers the entire file
    /// (header_len_field + header_json + tensor_data).
    mapping: []align(std.heap.page_size_min) const u8,
    /// Owns parsed shape arrays and tensor-name strings. Freed wholesale
    /// on deinit — no per-entry cleanup.
    arena: std.heap.ArenaAllocator,
    /// Tensor name → metadata. Names borrow from the arena.
    by_name: std.StringHashMap(Tensor),

    pub fn open(gpa: std.mem.Allocator, path: []const u8) !SafeTensors {
        const file = try std.fs.cwd().openFile(path, .{ .mode = .read_only });
        errdefer file.close();

        const stat = try file.stat();
        if (stat.size < 8) return error.FileTooSmall;
        const file_size: usize = @intCast(stat.size);

        const mapping: []align(std.heap.page_size_min) const u8 = if (builtin.os.tag == .windows) blk: {
            // CreateFileMappingW with size 0/0 means "use the file's size".
            // We close the mapping handle immediately after MapViewOfFile —
            // the view keeps an internal ref so the section stays alive
            // until UnmapViewOfFile.
            const map_handle = win.CreateFileMappingW(
                file.handle,
                null,
                win.PAGE_READONLY,
                0,
                0,
                null,
            ) orelse return error.CreateFileMappingFailed;
            defer std.os.windows.CloseHandle(map_handle);
            const view = win.MapViewOfFile(
                map_handle,
                win.FILE_MAP_READ,
                0,
                0,
                file_size,
            ) orelse return error.MapViewOfFileFailed;
            // MapViewOfFile returns a 64 KiB-aligned address; claiming
            // page_size alignment is conservative and safe.
            const ptr: [*]align(std.heap.page_size_min) const u8 = @ptrCast(@alignCast(view));
            break :blk ptr[0..file_size];
        } else try std.posix.mmap(
            null,
            file_size,
            std.posix.PROT.READ,
            .{ .TYPE = .PRIVATE },
            file.handle,
            0,
        );
        errdefer if (builtin.os.tag == .windows) {
            _ = win.UnmapViewOfFile(mapping.ptr);
        } else std.posix.munmap(mapping);

        // Header: u64 LE header length, then JSON, then tensor data.
        const header_len = std.mem.readInt(u64, mapping[0..8], .little);
        if (header_len == 0 or 8 + header_len > file_size) return error.HeaderOverrun;
        const header_json = mapping[8 .. 8 + @as(usize, @intCast(header_len))];
        const data_base: usize = 8 + @as(usize, @intCast(header_len));

        var arena = std.heap.ArenaAllocator.init(gpa);
        errdefer arena.deinit();
        const a = arena.allocator();

        var parsed = try std.json.parseFromSlice(std.json.Value, a, header_json, .{});
        // We deliberately let `parsed` live for the duration of `open` and
        // copy out into our arena — std.json values share internal arrays
        // we don't want to depend on past this scope.
        defer parsed.deinit();
        if (parsed.value != .object) return error.HeaderNotObject;

        var by_name = std.StringHashMap(Tensor).init(gpa);
        errdefer by_name.deinit();

        var it = parsed.value.object.iterator();
        while (it.next()) |entry| {
            const name = entry.key_ptr.*;
            // Skip the optional __metadata__ entry — string→string pairs,
            // not a tensor.
            if (std.mem.eql(u8, name, "__metadata__")) continue;
            if (entry.value_ptr.* != .object) return error.MalformedEntry;
            const obj = entry.value_ptr.object;

            const dtype_v = obj.get("dtype") orelse return error.MissingDtype;
            const shape_v = obj.get("shape") orelse return error.MissingShape;
            const offs_v = obj.get("data_offsets") orelse return error.MissingOffsets;
            if (dtype_v != .string or shape_v != .array or offs_v != .array) return error.MalformedEntry;

            const dtype = try Dtype.fromString(dtype_v.string);

            const shape = try a.alloc(usize, shape_v.array.items.len);
            var numel: usize = 1;
            for (shape_v.array.items, 0..) |d, i| {
                if (d != .integer or d.integer < 0) return error.MalformedShape;
                shape[i] = @intCast(d.integer);
                numel *|= shape[i];
            }

            if (offs_v.array.items.len != 2) return error.MalformedOffsets;
            const start_v = offs_v.array.items[0];
            const end_v = offs_v.array.items[1];
            if (start_v != .integer or end_v != .integer) return error.MalformedOffsets;
            if (start_v.integer < 0 or end_v.integer < start_v.integer) return error.MalformedOffsets;
            const start: usize = @intCast(start_v.integer);
            const end: usize = @intCast(end_v.integer);

            const expected_bytes = numel * dtype.elemSize();
            if (end - start != expected_bytes) return error.OffsetSizeMismatch;
            if (data_base + end > file_size) return error.DataOutOfRange;

            const bytes = mapping[data_base + start .. data_base + end];

            // Copy the name into the arena so it outlives `parsed`.
            const owned_name = try a.dupe(u8, name);
            try by_name.put(owned_name, .{
                .dtype = dtype,
                .shape = shape,
                .bytes = bytes,
            });
        }

        return .{
            .file = file,
            .mapping = mapping,
            .arena = arena,
            .by_name = by_name,
        };
    }

    pub fn deinit(self: *SafeTensors) void {
        self.by_name.deinit();
        self.arena.deinit();
        if (builtin.os.tag == .windows) {
            _ = win.UnmapViewOfFile(self.mapping.ptr);
        } else {
            std.posix.munmap(self.mapping);
        }
        self.file.close();
    }

    pub fn get(self: *const SafeTensors, name: []const u8) ?Tensor {
        return self.by_name.get(name);
    }

    pub fn count(self: *const SafeTensors) usize {
        return self.by_name.count();
    }
};
