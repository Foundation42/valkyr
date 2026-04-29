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
    const rmsnorm_spv = compileShader(b, "rmsnorm");
    const geglu_spv = compileShader(b, "geglu");
    const rope_spv = compileShader(b, "rope");

    // Stage compiled SPIR-V into one anonymous module. SPIR-V must be
    // 4-byte aligned for Vulkan's pCode field; dereferencing the
    // @embedFile pointer with align(4) materialises the bytes in static
    // data at a u32-aligned address.
    const wf = b.addWriteFiles();
    _ = wf.addCopyFile(vec_add_spv, "vec_add.spv");
    _ = wf.addCopyFile(matmul_nt_spv, "matmul_nt.spv");
    _ = wf.addCopyFile(rmsnorm_spv, "rmsnorm.spv");
    _ = wf.addCopyFile(geglu_spv, "geglu.spv");
    _ = wf.addCopyFile(rope_spv, "rope.spv");
    const shader_mod = wf.add("shaders.zig",
        \\pub const vec_add align(4) = @embedFile("vec_add.spv").*;
        \\pub const matmul_nt align(4) = @embedFile("matmul_nt.spv").*;
        \\pub const rmsnorm align(4) = @embedFile("rmsnorm.spv").*;
        \\pub const geglu align(4) = @embedFile("geglu.spv").*;
        \\pub const rope align(4) = @embedFile("rope.spv").*;
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
