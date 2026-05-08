//! HuggingFace cache helpers: resolve `org/name` model IDs to on-disk
//! snapshot paths, and list cached models for a `--list` UX.
//!
//! The HF cache layout is:
//!
//!   {cache_root}/
//!   ├── models--{org}--{name}/
//!   │   ├── refs/main             (text file: commit hash on the main branch)
//!   │   ├── refs/{branch}         (other branches)
//!   │   ├── blobs/                (content-addressed shared store)
//!   │   └── snapshots/{hash}/     (one dir per pinned commit; files are
//!   │                              symlinks into ../blobs/)
//!
//! `cache_root` defaults to ~/.cache/huggingface/hub but is overridden by
//! the env vars `HF_HUB_CACHE` (highest priority), then `HF_HOME` (treated
//! as `$HF_HOME/hub`), then `XDG_CACHE_HOME` (treated as
//! `$XDG_CACHE_HOME/huggingface/hub`), then the default. We follow the
//! same priority order huggingface_hub itself uses.
//!
//! For the home-directory fallback we try `HOME` first (POSIX) then
//! `USERPROFILE` (Windows) — matches Python's `os.path.expanduser("~")`
//! cross-platform behaviour, which is what huggingface_hub uses.

const std = @import("std");

pub const ModelInfo = struct {
    /// Full HF id like "meta-llama/Llama-3.2-3B-Instruct" (caller-owned).
    id: []u8,
    /// Snapshot directory, absolute path (caller-owned).
    snapshot_path: []u8,
    /// Architecture string from config.json's `architectures[0]`, or
    /// "(unknown)" if config.json is missing/invalid.
    architecture: []u8,
    /// Approximate bytes on disk for the snapshot's content (sum of
    /// dereferenced blob targets — symlinks counted by what they point at).
    bytes: u64,
    /// True iff the architecture is one valkyr's loader recognises.
    supported: bool,

    pub fn deinit(self: *ModelInfo, gpa: std.mem.Allocator) void {
        gpa.free(self.id);
        gpa.free(self.snapshot_path);
        gpa.free(self.architecture);
    }
};

/// Returns the absolute path to the HF cache root. Caller owns the slice.
pub fn cacheRoot(gpa: std.mem.Allocator) ![]u8 {
    if (std.process.getEnvVarOwned(gpa, "HF_HUB_CACHE")) |v| {
        return v;
    } else |_| {}
    if (std.process.getEnvVarOwned(gpa, "HF_HOME")) |v| {
        defer gpa.free(v);
        return std.fs.path.join(gpa, &.{ v, "hub" });
    } else |_| {}
    if (std.process.getEnvVarOwned(gpa, "XDG_CACHE_HOME")) |v| {
        defer gpa.free(v);
        return std.fs.path.join(gpa, &.{ v, "huggingface", "hub" });
    } else |_| {}
    const home = blk: {
        if (std.process.getEnvVarOwned(gpa, "HOME")) |v| break :blk v else |_| {}
        if (std.process.getEnvVarOwned(gpa, "USERPROFILE")) |v| break :blk v else |_| {}
        return error.NoHome;
    };
    defer gpa.free(home);
    return std.fs.path.join(gpa, &.{ home, ".cache", "huggingface", "hub" });
}

/// Detect whether `arg` looks like an HF model id (`org/name`) rather
/// than a filesystem path. Heuristic: contains '/', no leading '/' or
/// '.', and a same-named filesystem entry doesn't already exist.
/// Lets `--chat ./local/dir` and `--chat /abs/path` continue to work
/// transparently while `--chat meta-llama/Llama-3.2-3B-Instruct` is
/// auto-resolved.
pub fn looksLikeModelId(arg: []const u8) bool {
    if (arg.len == 0) return false;
    if (arg[0] == '/' or arg[0] == '.') return false;
    // Windows drive-letter paths like `C:\foo` or `C:/foo` — second char
    // is always `:`. HF ids never contain `:`, so this is a clean rule
    // that lets `--chat C:/dev/model` work even if the path doesn't exist.
    if (arg.len >= 2 and arg[1] == ':') return false;
    if (std.mem.indexOfScalar(u8, arg, '/') == null) return false;
    // If it's an existing path on disk, treat as path (let the user win
    // any unintended ambiguity).
    if (std.fs.cwd().access(arg, .{})) {
        return false;
    } else |_| {}
    return true;
}

/// Convert "org/name" → cache directory name "models--org--name".
fn cacheDirName(gpa: std.mem.Allocator, model_id: []const u8) ![]u8 {
    var buf = std.ArrayList(u8).init(gpa);
    errdefer buf.deinit();
    try buf.appendSlice("models--");
    for (model_id) |c| {
        if (c == '/') try buf.appendSlice("--") else try buf.append(c);
    }
    return buf.toOwnedSlice();
}

/// Resolve `model_id` to its snapshot directory under the cache.
/// Strategy: read `refs/main` to get the commit hash, fall back to the
/// only snapshot if `refs/main` is absent and there's exactly one.
/// Caller owns the returned slice.
pub fn findSnapshot(gpa: std.mem.Allocator, model_id: []const u8) ![]u8 {
    const root = try cacheRoot(gpa);
    defer gpa.free(root);
    const dir_name = try cacheDirName(gpa, model_id);
    defer gpa.free(dir_name);
    const repo_dir = try std.fs.path.join(gpa, &.{ root, dir_name });
    defer gpa.free(repo_dir);

    // Verify the repo dir exists at all — gives a friendlier error than
    // failing inside refs/main lookup.
    std.fs.accessAbsolute(repo_dir, .{}) catch |e| switch (e) {
        error.FileNotFound => return error.HfModelNotInCache,
        else => return e,
    };

    // Try refs/main first — HF's standard branch ref.
    const refs_main = try std.fs.path.join(gpa, &.{ repo_dir, "refs", "main" });
    defer gpa.free(refs_main);
    if (std.fs.openFileAbsolute(refs_main, .{})) |f| {
        defer f.close();
        var hash_buf: [128]u8 = undefined;
        const n = try f.readAll(&hash_buf);
        const hash = std.mem.trim(u8, hash_buf[0..n], &std.ascii.whitespace);
        if (hash.len == 0) return error.HfRefsMainEmpty;
        return std.fs.path.join(gpa, &.{ repo_dir, "snapshots", hash });
    } else |_| {}

    // Fallback: if there's exactly one snapshot, use it. Multi-snapshot
    // case without refs/main is genuinely ambiguous — refuse rather than
    // pick wrong.
    const snapshots_dir = try std.fs.path.join(gpa, &.{ repo_dir, "snapshots" });
    defer gpa.free(snapshots_dir);
    var d = std.fs.openDirAbsolute(snapshots_dir, .{ .iterate = true }) catch
        return error.HfNoSnapshots;
    defer d.close();
    var it = d.iterate();
    var seen: ?[]u8 = null;
    errdefer if (seen) |s| gpa.free(s);
    while (try it.next()) |entry| {
        if (entry.kind != .directory) continue;
        if (seen != null) {
            gpa.free(seen.?);
            return error.HfMultipleSnapshotsNoRef;
        }
        seen = try gpa.dupe(u8, entry.name);
    }
    if (seen) |s| {
        defer gpa.free(s);
        return std.fs.path.join(gpa, &.{ repo_dir, "snapshots", s });
    }
    return error.HfNoSnapshots;
}

/// If `arg` looks like an HF model id, resolve it; otherwise return a
/// duped copy of `arg` (so the caller can uniformly free the result).
pub fn resolveModelArg(gpa: std.mem.Allocator, arg: []const u8) ![]u8 {
    if (looksLikeModelId(arg)) {
        return findSnapshot(gpa, arg);
    }
    return gpa.dupe(u8, arg);
}

/// Architectures valkyr's loader recognises. Kept here (instead of
/// re-using config.zig's `Family.fromArchitectures`) so we don't have
/// to actually load the config to mark a row as supported — we only
/// peek at the JSON.
const SUPPORTED_ARCHITECTURES = [_][]const u8{
    "GemmaForCausalLM",
    "LlamaForCausalLM",
    "Qwen3ForCausalLM",
    "Qwen3_5ForConditionalGeneration",
};

fn isSupportedArch(arch: []const u8) bool {
    for (SUPPORTED_ARCHITECTURES) |s| {
        if (std.mem.eql(u8, s, arch)) return true;
    }
    return false;
}

/// Cheaply recover "org/name" from a cache dir name "models--org--name".
/// HF's encoding doesn't allow `/` in org or name, so the first `--`
/// after the `models--` prefix splits org from name. Caller owns slice.
fn modelIdFromDirName(gpa: std.mem.Allocator, dir_name: []const u8) ?[]u8 {
    const prefix = "models--";
    if (!std.mem.startsWith(u8, dir_name, prefix)) return null;
    const rest = dir_name[prefix.len..];
    const sep = std.mem.indexOf(u8, rest, "--") orelse return null;
    const org = rest[0..sep];
    const name = rest[sep + 2 ..];
    return std.fmt.allocPrint(gpa, "{s}/{s}", .{ org, name }) catch null;
}

/// Sum of dereferenced file sizes under `path`. Each entry is stat'd
/// in follow-symlinks mode so symlinked blobs count toward the total.
fn snapshotBytes(snapshot_path: []const u8) u64 {
    var total: u64 = 0;
    var d = std.fs.openDirAbsolute(snapshot_path, .{ .iterate = true }) catch
        return 0;
    defer d.close();
    var it = d.iterate();
    while (it.next() catch null) |entry| {
        const child = d.openFile(entry.name, .{}) catch continue;
        defer child.close();
        const stat = child.stat() catch continue;
        total += stat.size;
    }
    return total;
}

/// Read `architectures[0]` from `{snapshot}/config.json`. Returns a
/// caller-owned slice, or `error.NoConfig` if the file is missing /
/// not JSON / has no architectures array.
fn readArchitecture(gpa: std.mem.Allocator, snapshot_path: []const u8) ![]u8 {
    const cfg_path = try std.fs.path.join(gpa, &.{ snapshot_path, "config.json" });
    defer gpa.free(cfg_path);
    const file = std.fs.openFileAbsolute(cfg_path, .{}) catch return error.NoConfig;
    defer file.close();
    const bytes = try file.readToEndAlloc(gpa, 1024 * 1024);
    defer gpa.free(bytes);

    var parsed = std.json.parseFromSlice(std.json.Value, gpa, bytes, .{}) catch
        return error.NoConfig;
    defer parsed.deinit();
    const root = parsed.value;
    if (root != .object) return error.NoConfig;
    const archs = root.object.get("architectures") orelse return error.NoConfig;
    if (archs != .array) return error.NoConfig;
    if (archs.array.items.len == 0) return error.NoConfig;
    const first = archs.array.items[0];
    if (first != .string) return error.NoConfig;
    return gpa.dupe(u8, first.string);
}

/// List every cached model under the HF cache root. Caller owns the
/// slice and must call `deinit` on each entry. Always returns at least
/// an empty slice — missing cache root isn't an error, just zero models.
pub fn listModels(gpa: std.mem.Allocator) ![]ModelInfo {
    var out = std.ArrayList(ModelInfo).init(gpa);
    errdefer {
        for (out.items) |*m| m.deinit(gpa);
        out.deinit();
    }

    const root = try cacheRoot(gpa);
    defer gpa.free(root);

    var d = std.fs.openDirAbsolute(root, .{ .iterate = true }) catch
        return out.toOwnedSlice();
    defer d.close();

    var it = d.iterate();
    while (try it.next()) |entry| {
        if (entry.kind != .directory) continue;
        if (!std.mem.startsWith(u8, entry.name, "models--")) continue;

        const id = modelIdFromDirName(gpa, entry.name) orelse continue;
        errdefer gpa.free(id);

        // Resolve via findSnapshot so the same refs/main → fallback path
        // governs both --list and `--chat <id>`. Skip silently if the
        // model has no usable snapshot — partial downloads etc.
        const snapshot = findSnapshot(gpa, id) catch {
            gpa.free(id);
            continue;
        };
        errdefer gpa.free(snapshot);

        const arch = readArchitecture(gpa, snapshot) catch
            try gpa.dupe(u8, "(unknown)");
        errdefer gpa.free(arch);

        try out.append(.{
            .id = id,
            .snapshot_path = snapshot,
            .architecture = arch,
            .bytes = snapshotBytes(snapshot),
            .supported = isSupportedArch(arch),
        });
    }

    return out.toOwnedSlice();
}

/// Format `bytes` as a human-readable size like "3.2 GiB".
pub fn formatSize(bytes: u64, buf: []u8) ![]u8 {
    const units = [_][]const u8{ "B", "KiB", "MiB", "GiB", "TiB" };
    var b: f64 = @floatFromInt(bytes);
    var u: usize = 0;
    while (b >= 1024.0 and u + 1 < units.len) : (u += 1) b /= 1024.0;
    return std.fmt.bufPrint(buf, "{d:.1} {s}", .{ b, units[u] });
}
