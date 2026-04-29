const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    // ── Compile GLSL → SPIR-V ──
    // Each .comp gets -I shaders/ so it can #include common.glsl. -MD/-MF
    // emits a make-style depfile so edits to included files trigger a
    // recompile of every downstream shader — without it only direct edits
    // to the top file would invalidate the cached SPIR-V.
    const vec_add_spv = compileShader(b, "vec_add");
    const matmul_nt_spv = compileShader(b, "matmul_nt");
    const matmul_nt_v2_spv = compileShader(b, "matmul_nt_v2");
    const rmsnorm_spv = compileShader(b, "rmsnorm");
    const geglu_spv = compileShader(b, "geglu");
    const rope_spv = compileShader(b, "rope");
    const softmax_spv = compileShader(b, "softmax");
    const embed_lookup_spv = compileShader(b, "embed_lookup");
    const add_in_place_spv = compileShader(b, "add_in_place");
    const attn_decode_single_spv = compileShader(b, "attn_decode_single");
    const attn_scores_spv = compileShader(b, "attn_scores");
    const attn_output_spv = compileShader(b, "attn_output");
    const kv_write_spv = compileShader(b, "kv_write");

    // Stage compiled SPIR-V into one anonymous module. SPIR-V must be
    // 4-byte aligned for Vulkan's pCode field; dereferencing the
    // @embedFile pointer with align(4) materialises the bytes in static
    // data at a u32-aligned address.
    const wf = b.addWriteFiles();
    _ = wf.addCopyFile(vec_add_spv, "vec_add.spv");
    _ = wf.addCopyFile(matmul_nt_spv, "matmul_nt.spv");
    _ = wf.addCopyFile(matmul_nt_v2_spv, "matmul_nt_v2.spv");
    _ = wf.addCopyFile(rmsnorm_spv, "rmsnorm.spv");
    _ = wf.addCopyFile(geglu_spv, "geglu.spv");
    _ = wf.addCopyFile(rope_spv, "rope.spv");
    _ = wf.addCopyFile(softmax_spv, "softmax.spv");
    _ = wf.addCopyFile(embed_lookup_spv, "embed_lookup.spv");
    _ = wf.addCopyFile(add_in_place_spv, "add_in_place.spv");
    _ = wf.addCopyFile(attn_decode_single_spv, "attn_decode_single.spv");
    _ = wf.addCopyFile(attn_scores_spv, "attn_scores.spv");
    _ = wf.addCopyFile(attn_output_spv, "attn_output.spv");
    _ = wf.addCopyFile(kv_write_spv, "kv_write.spv");
    const shader_mod = wf.add("shaders.zig",
        \\pub const vec_add align(4) = @embedFile("vec_add.spv").*;
        \\pub const matmul_nt align(4) = @embedFile("matmul_nt.spv").*;
        \\pub const matmul_nt_v2 align(4) = @embedFile("matmul_nt_v2.spv").*;
        \\pub const rmsnorm align(4) = @embedFile("rmsnorm.spv").*;
        \\pub const geglu align(4) = @embedFile("geglu.spv").*;
        \\pub const rope align(4) = @embedFile("rope.spv").*;
        \\pub const softmax align(4) = @embedFile("softmax.spv").*;
        \\pub const embed_lookup align(4) = @embedFile("embed_lookup.spv").*;
        \\pub const add_in_place align(4) = @embedFile("add_in_place.spv").*;
        \\pub const attn_decode_single align(4) = @embedFile("attn_decode_single.spv").*;
        \\pub const attn_scores align(4) = @embedFile("attn_scores.spv").*;
        \\pub const attn_output align(4) = @embedFile("attn_output.spv").*;
        \\pub const kv_write align(4) = @embedFile("kv_write.spv").*;
    );

    const exe = b.addExecutable(.{
        .name = "tripvulkan",
        .root_source_file = b.path("src/main.zig"),
        .target = target,
        .optimize = optimize,
    });
    exe.root_module.addAnonymousImport("shaders", .{
        .root_source_file = shader_mod,
    });
    exe.linkSystemLibrary("vulkan");
    exe.linkLibC();

    b.installArtifact(exe);

    const run_cmd = b.addRunArtifact(exe);
    run_cmd.step.dependOn(b.getInstallStep());
    if (b.args) |args| run_cmd.addArgs(args);

    const run_step = b.step("run", "Run tripvulkan");
    run_step.dependOn(&run_cmd.step);
}

fn compileShader(b: *std.Build, name: []const u8) std.Build.LazyPath {
    const src = b.fmt("shaders/{s}.comp", .{name});
    const spv = b.fmt("{s}.spv", .{name});
    const dep = b.fmt("{s}.d", .{name});
    const cmd = b.addSystemCommand(&.{ "glslc", "--target-env=vulkan1.3" });
    cmd.addArg("-I");
    cmd.addDirectoryArg(b.path("shaders"));
    cmd.addArg("-MD");
    cmd.addArg("-MF");
    _ = cmd.addDepFileOutputArg(dep);
    cmd.addFileArg(b.path(src));
    cmd.addArg("-o");
    return cmd.addOutputFileArg(spv);
}
