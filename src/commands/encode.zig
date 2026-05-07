//! `--encode` subcommand: tokenizer load + encode + decode round-trip
//! sanity check. Extracted from main.zig.

const std = @import("std");
const tokenizer_mod = @import("../tokenizer.zig");

pub fn runEncode(gpa: std.mem.Allocator, dir_path: []const u8, text: []const u8) !void {
    const tok_path = try std.fmt.allocPrint(gpa, "{s}/tokenizer.json", .{dir_path});
    defer gpa.free(tok_path);

    const t0 = std.time.nanoTimestamp();
    var tok = try tokenizer_mod.Tokenizer.loadFromFile(gpa, tok_path);
    defer tok.deinit();
    const t1 = std.time.nanoTimestamp();
    const load_ms = @as(f64, @floatFromInt(t1 - t0)) / 1_000_000.0;

    const stdout = std.io.getStdOut().writer();
    try stdout.print("tokenizer load: {d:.0} ms ({d} ids, {d} merges)\n\n", .{
        load_ms, tok.vocabSize(), tok.merges.count(),
    });
    try stdout.print("input: {s}\n\n", .{text});

    const t2 = std.time.nanoTimestamp();
    const ids = try tok.encode(gpa, text);
    defer gpa.free(ids);
    const t3 = std.time.nanoTimestamp();
    const enc_ms = @as(f64, @floatFromInt(t3 - t2)) / 1_000_000.0;

    try stdout.print("{d} tokens, encode {d:.2} ms:\n", .{ ids.len, enc_ms });
    for (ids) |id| {
        const s = tok.decode(id) orelse "<unknown>";
        try stdout.print("  id={d:>6}  {s}\n", .{ id, s });
    }

    // Reconstruct: concat decoded strings (no normalization reversal
    // here — we just print what each id yields on decode). This is a
    // sanity check that the encoded stream is sensible, not a true
    // round-trip (true round-trip would also undo the ▁→' ' decoder).
    try stdout.print("\nconcat of decoded ids: \"", .{});
    for (ids) |id| {
        if (tok.decode(id)) |s| try stdout.print("{s}", .{s});
    }
    try stdout.print("\"\n", .{});
}
