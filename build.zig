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
    const matmul_nt_v2_bf16_spv = compileShader(b, "matmul_nt_v2_bf16");
    const matmul_nt_v2_q4_0_spv = compileShader(b, "matmul_nt_v2_q4_0");
    const rmsnorm_spv = compileShader(b, "rmsnorm");
    const geglu_spv = compileShader(b, "geglu");
    const swiglu_spv = compileShader(b, "swiglu");
    const rope_spv = compileShader(b, "rope");
    const rope_partial_spv = compileShader(b, "rope_partial");
    const split_q_gate_spv = compileShader(b, "split_q_gate");
    const sigmoid_mul_spv = compileShader(b, "sigmoid_mul");
    const l2norm_per_head_spv = compileShader(b, "l2norm_per_head");
    const conv1d_update_spv = compileShader(b, "conv1d_update");
    const rmsnorm_gated_spv = compileShader(b, "rmsnorm_gated");
    const gated_delta_step_spv = compileShader(b, "gated_delta_step");
    const slice_copy_spv = compileShader(b, "slice_copy");
    const scale_spv = compileShader(b, "scale");
    const softmax_spv = compileShader(b, "softmax");
    const embed_lookup_spv = compileShader(b, "embed_lookup");
    const embed_lookup_bf16_spv = compileShader(b, "embed_lookup_bf16");
    const add_in_place_spv = compileShader(b, "add_in_place");
    const attn_decode_single_spv = compileShader(b, "attn_decode_single");
    const attn_scores_spv = compileShader(b, "attn_scores");
    const attn_output_spv = compileShader(b, "attn_output");
    const kv_write_spv = compileShader(b, "kv_write");
    const fwht256_spv = compileShader(b, "fwht256");
    const rht_pre256_spv = compileShader(b, "rht_pre256");
    const rht_post256_spv = compileShader(b, "rht_post256");
    const tq4_pack256_spv = compileShader(b, "tq4_pack256");
    const tq4_unpack256_spv = compileShader(b, "tq4_unpack256");
    const tq4_pack_to_cache_spv = compileShader(b, "tq4_pack_to_cache");
    const fwht128_spv = compileShader(b, "fwht128");
    const rht_pre128_spv = compileShader(b, "rht_pre128");
    const rht_post128_spv = compileShader(b, "rht_post128");
    const tq4_pack128_spv = compileShader(b, "tq4_pack128");
    const tq4_unpack128_spv = compileShader(b, "tq4_unpack128");
    const tq4_pack_to_cache128_spv = compileShader(b, "tq4_pack_to_cache128");

    // Stage compiled SPIR-V into one anonymous module. SPIR-V must be
    // 4-byte aligned for Vulkan's pCode field; dereferencing the
    // @embedFile pointer with align(4) materialises the bytes in static
    // data at a u32-aligned address.
    const wf = b.addWriteFiles();
    _ = wf.addCopyFile(vec_add_spv, "vec_add.spv");
    _ = wf.addCopyFile(matmul_nt_spv, "matmul_nt.spv");
    _ = wf.addCopyFile(matmul_nt_v2_spv, "matmul_nt_v2.spv");
    _ = wf.addCopyFile(matmul_nt_v2_bf16_spv, "matmul_nt_v2_bf16.spv");
    _ = wf.addCopyFile(matmul_nt_v2_q4_0_spv, "matmul_nt_v2_q4_0.spv");
    _ = wf.addCopyFile(rmsnorm_spv, "rmsnorm.spv");
    _ = wf.addCopyFile(geglu_spv, "geglu.spv");
    _ = wf.addCopyFile(swiglu_spv, "swiglu.spv");
    _ = wf.addCopyFile(rope_spv, "rope.spv");
    _ = wf.addCopyFile(rope_partial_spv, "rope_partial.spv");
    _ = wf.addCopyFile(split_q_gate_spv, "split_q_gate.spv");
    _ = wf.addCopyFile(sigmoid_mul_spv, "sigmoid_mul.spv");
    _ = wf.addCopyFile(l2norm_per_head_spv, "l2norm_per_head.spv");
    _ = wf.addCopyFile(conv1d_update_spv, "conv1d_update.spv");
    _ = wf.addCopyFile(rmsnorm_gated_spv, "rmsnorm_gated.spv");
    _ = wf.addCopyFile(gated_delta_step_spv, "gated_delta_step.spv");
    _ = wf.addCopyFile(slice_copy_spv, "slice_copy.spv");
    _ = wf.addCopyFile(scale_spv, "scale.spv");
    _ = wf.addCopyFile(softmax_spv, "softmax.spv");
    _ = wf.addCopyFile(embed_lookup_spv, "embed_lookup.spv");
    _ = wf.addCopyFile(embed_lookup_bf16_spv, "embed_lookup_bf16.spv");
    _ = wf.addCopyFile(add_in_place_spv, "add_in_place.spv");
    _ = wf.addCopyFile(attn_decode_single_spv, "attn_decode_single.spv");
    _ = wf.addCopyFile(attn_scores_spv, "attn_scores.spv");
    _ = wf.addCopyFile(attn_output_spv, "attn_output.spv");
    _ = wf.addCopyFile(kv_write_spv, "kv_write.spv");
    _ = wf.addCopyFile(fwht256_spv, "fwht256.spv");
    _ = wf.addCopyFile(rht_pre256_spv, "rht_pre256.spv");
    _ = wf.addCopyFile(rht_post256_spv, "rht_post256.spv");
    _ = wf.addCopyFile(tq4_pack256_spv, "tq4_pack256.spv");
    _ = wf.addCopyFile(tq4_unpack256_spv, "tq4_unpack256.spv");
    _ = wf.addCopyFile(tq4_pack_to_cache_spv, "tq4_pack_to_cache.spv");
    _ = wf.addCopyFile(fwht128_spv, "fwht128.spv");
    _ = wf.addCopyFile(rht_pre128_spv, "rht_pre128.spv");
    _ = wf.addCopyFile(rht_post128_spv, "rht_post128.spv");
    _ = wf.addCopyFile(tq4_pack128_spv, "tq4_pack128.spv");
    _ = wf.addCopyFile(tq4_unpack128_spv, "tq4_unpack128.spv");
    _ = wf.addCopyFile(tq4_pack_to_cache128_spv, "tq4_pack_to_cache128.spv");
    const shader_mod = wf.add("shaders.zig",
        \\pub const vec_add align(4) = @embedFile("vec_add.spv").*;
        \\pub const matmul_nt align(4) = @embedFile("matmul_nt.spv").*;
        \\pub const matmul_nt_v2 align(4) = @embedFile("matmul_nt_v2.spv").*;
        \\pub const matmul_nt_v2_bf16 align(4) = @embedFile("matmul_nt_v2_bf16.spv").*;
        \\pub const matmul_nt_v2_q4_0 align(4) = @embedFile("matmul_nt_v2_q4_0.spv").*;
        \\pub const rmsnorm align(4) = @embedFile("rmsnorm.spv").*;
        \\pub const geglu align(4) = @embedFile("geglu.spv").*;
        \\pub const swiglu align(4) = @embedFile("swiglu.spv").*;
        \\pub const rope align(4) = @embedFile("rope.spv").*;
        \\pub const rope_partial align(4) = @embedFile("rope_partial.spv").*;
        \\pub const split_q_gate align(4) = @embedFile("split_q_gate.spv").*;
        \\pub const sigmoid_mul align(4) = @embedFile("sigmoid_mul.spv").*;
        \\pub const l2norm_per_head align(4) = @embedFile("l2norm_per_head.spv").*;
        \\pub const conv1d_update align(4) = @embedFile("conv1d_update.spv").*;
        \\pub const rmsnorm_gated align(4) = @embedFile("rmsnorm_gated.spv").*;
        \\pub const gated_delta_step align(4) = @embedFile("gated_delta_step.spv").*;
        \\pub const slice_copy align(4) = @embedFile("slice_copy.spv").*;
        \\pub const scale align(4) = @embedFile("scale.spv").*;
        \\pub const softmax align(4) = @embedFile("softmax.spv").*;
        \\pub const embed_lookup align(4) = @embedFile("embed_lookup.spv").*;
        \\pub const embed_lookup_bf16 align(4) = @embedFile("embed_lookup_bf16.spv").*;
        \\pub const add_in_place align(4) = @embedFile("add_in_place.spv").*;
        \\pub const attn_decode_single align(4) = @embedFile("attn_decode_single.spv").*;
        \\pub const attn_scores align(4) = @embedFile("attn_scores.spv").*;
        \\pub const attn_output align(4) = @embedFile("attn_output.spv").*;
        \\pub const kv_write align(4) = @embedFile("kv_write.spv").*;
        \\pub const fwht256 align(4) = @embedFile("fwht256.spv").*;
        \\pub const rht_pre256 align(4) = @embedFile("rht_pre256.spv").*;
        \\pub const rht_post256 align(4) = @embedFile("rht_post256.spv").*;
        \\pub const tq4_pack256 align(4) = @embedFile("tq4_pack256.spv").*;
        \\pub const tq4_unpack256 align(4) = @embedFile("tq4_unpack256.spv").*;
        \\pub const tq4_pack_to_cache align(4) = @embedFile("tq4_pack_to_cache.spv").*;
        \\pub const fwht128 align(4) = @embedFile("fwht128.spv").*;
        \\pub const rht_pre128 align(4) = @embedFile("rht_pre128.spv").*;
        \\pub const rht_post128 align(4) = @embedFile("rht_post128.spv").*;
        \\pub const tq4_pack128 align(4) = @embedFile("tq4_pack128.spv").*;
        \\pub const tq4_unpack128 align(4) = @embedFile("tq4_unpack128.spv").*;
        \\pub const tq4_pack_to_cache128 align(4) = @embedFile("tq4_pack_to_cache128.spv").*;
    );

    const exe = b.addExecutable(.{
        .name = "valkyr",
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

    const run_step = b.step("run", "Run valkyr");
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
