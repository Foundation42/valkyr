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
    const matmul_nt_v2_q4_k_spv = compileShader(b, "matmul_nt_v2_q4_k");
    const rmsnorm_spv = compileShader(b, "rmsnorm");
    const rmsnorm_backward_spv = compileShader(b, "rmsnorm_backward");
    const layernorm_spv = compileShader(b, "layernorm");
    const layernorm_backward_spv = compileShader(b, "layernorm_backward");
    const embedding_backward_spv = compileShader(b, "embedding_backward");
    const softmax_backward_spv = compileShader(b, "softmax_backward");
    const attn_backward_dattn_spv = compileShader(b, "attn_backward_dattn");
    const attn_backward_dv_spv = compileShader(b, "attn_backward_dv");
    const attn_backward_dq_spv = compileShader(b, "attn_backward_dq");
    const attn_backward_dk_spv = compileShader(b, "attn_backward_dk");
    const rope_backward_spv = compileShader(b, "rope_backward");
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
    const embed_lookup_batched_spv = compileShader(b, "embed_lookup_batched");
    const add_in_place_spv = compileShader(b, "add_in_place");
    const attn_decode_single_spv = compileShader(b, "attn_decode_single");
    const attn_scores_spv = compileShader(b, "attn_scores");
    const attn_scores_train_spv = compileShader(b, "attn_scores_train");
    const attn_output_spv = compileShader(b, "attn_output");
    const attn_output_train_spv = compileShader(b, "attn_output_train");
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
    const attn_synth_spv = compileShader(b, "attn_synth");
    const relu_spv = compileShader(b, "relu");
    const relu_backward_spv = compileShader(b, "relu_backward");
    const linear_backward_dx_spv = compileShader(b, "linear_backward_dx");
    const linear_backward_dx_batched_spv = compileShader(b, "linear_backward_dx_batched");
    const linear_backward_dw_batched_spv = compileShader(b, "linear_backward_dw_batched");
    const outer_product_spv = compileShader(b, "outer_product");
    const sgd_step_spv = compileShader(b, "sgd_step");
    const adam_step_spv = compileShader(b, "adam_step");
    const mse_loss_grad_spv = compileShader(b, "mse_loss_grad");
    const mse_loss_grad_scaled_spv = compileShader(b, "mse_loss_grad_scaled");
    const mlp2_forward_batched_spv = compileShader(b, "mlp2_forward_batched");
    const mlp2_forward_train_batched_spv = compileShader(b, "mlp2_forward_train_batched");
    const mlp2_dy_batched_spv = compileShader(b, "mlp2_dy_batched");
    const mlp2_dh_pre_batched_spv = compileShader(b, "mlp2_dh_pre_batched");
    const mlp2_dw_accum_spv = compileShader(b, "mlp2_dw_accum");
    const mlp2_db_accum_spv = compileShader(b, "mlp2_db_accum");
    const softmax_ce_loss_grad_batched_spv = compileShader(b, "softmax_ce_loss_grad_batched");
    const softmax_ce_loss_grad_batched_v2_spv = compileShader(b, "softmax_ce_loss_grad_batched_v2");
    const swiglu_forward_spv = compileShader(b, "swiglu_forward");
    const swiglu_backward_spv = compileShader(b, "swiglu_backward");
    const rope_partial_batched_spv = compileShader(b, "rope_partial_batched");
    const rope_backward_batched_spv = compileShader(b, "rope_backward_batched");
    const mlp2_mse_loss_batched_spv = compileShader(b, "mlp2_mse_loss_batched");
    const mlp2_ce_loss_batched_spv = compileShader(b, "mlp2_ce_loss_batched");
    const cce_forward_spv = compileShader(b, "cce_forward");
    const cce_backward_dh_spv = compileShader(b, "cce_backward_dh");
    const cce_backward_dw_spv = compileShader(b, "cce_backward_dw");
    const qk_rope_partial_batched_spv = compileShader(b, "qk_rope_partial_batched");
    const qk_rope_backward_batched_spv = compileShader(b, "qk_rope_backward_batched");
    const fa_forward_spv = compileShader(b, "fa_forward");
    const fa_decode_split_spv = compileShader(b, "fa_decode_split");
    const fa_decode_merge_spv = compileShader(b, "fa_decode_merge");

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
    _ = wf.addCopyFile(matmul_nt_v2_q4_k_spv, "matmul_nt_v2_q4_k.spv");
    _ = wf.addCopyFile(rmsnorm_spv, "rmsnorm.spv");
    _ = wf.addCopyFile(rmsnorm_backward_spv, "rmsnorm_backward.spv");
    _ = wf.addCopyFile(layernorm_spv, "layernorm.spv");
    _ = wf.addCopyFile(layernorm_backward_spv, "layernorm_backward.spv");
    _ = wf.addCopyFile(embedding_backward_spv, "embedding_backward.spv");
    _ = wf.addCopyFile(softmax_backward_spv, "softmax_backward.spv");
    _ = wf.addCopyFile(attn_backward_dattn_spv, "attn_backward_dattn.spv");
    _ = wf.addCopyFile(attn_backward_dv_spv, "attn_backward_dv.spv");
    _ = wf.addCopyFile(attn_backward_dq_spv, "attn_backward_dq.spv");
    _ = wf.addCopyFile(attn_backward_dk_spv, "attn_backward_dk.spv");
    _ = wf.addCopyFile(rope_backward_spv, "rope_backward.spv");
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
    _ = wf.addCopyFile(embed_lookup_batched_spv, "embed_lookup_batched.spv");
    _ = wf.addCopyFile(add_in_place_spv, "add_in_place.spv");
    _ = wf.addCopyFile(attn_decode_single_spv, "attn_decode_single.spv");
    _ = wf.addCopyFile(attn_scores_spv, "attn_scores.spv");
    _ = wf.addCopyFile(attn_scores_train_spv, "attn_scores_train.spv");
    _ = wf.addCopyFile(attn_output_spv, "attn_output.spv");
    _ = wf.addCopyFile(attn_output_train_spv, "attn_output_train.spv");
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
    _ = wf.addCopyFile(attn_synth_spv, "attn_synth.spv");
    _ = wf.addCopyFile(relu_spv, "relu.spv");
    _ = wf.addCopyFile(relu_backward_spv, "relu_backward.spv");
    _ = wf.addCopyFile(linear_backward_dx_spv, "linear_backward_dx.spv");
    _ = wf.addCopyFile(linear_backward_dx_batched_spv, "linear_backward_dx_batched.spv");
    _ = wf.addCopyFile(linear_backward_dw_batched_spv, "linear_backward_dw_batched.spv");
    _ = wf.addCopyFile(outer_product_spv, "outer_product.spv");
    _ = wf.addCopyFile(sgd_step_spv, "sgd_step.spv");
    _ = wf.addCopyFile(adam_step_spv, "adam_step.spv");
    _ = wf.addCopyFile(mse_loss_grad_spv, "mse_loss_grad.spv");
    _ = wf.addCopyFile(mse_loss_grad_scaled_spv, "mse_loss_grad_scaled.spv");
    _ = wf.addCopyFile(mlp2_forward_batched_spv, "mlp2_forward_batched.spv");
    _ = wf.addCopyFile(mlp2_forward_train_batched_spv, "mlp2_forward_train_batched.spv");
    _ = wf.addCopyFile(mlp2_dy_batched_spv, "mlp2_dy_batched.spv");
    _ = wf.addCopyFile(mlp2_dh_pre_batched_spv, "mlp2_dh_pre_batched.spv");
    _ = wf.addCopyFile(mlp2_dw_accum_spv, "mlp2_dw_accum.spv");
    _ = wf.addCopyFile(mlp2_db_accum_spv, "mlp2_db_accum.spv");
    _ = wf.addCopyFile(softmax_ce_loss_grad_batched_spv, "softmax_ce_loss_grad_batched.spv");
    _ = wf.addCopyFile(softmax_ce_loss_grad_batched_v2_spv, "softmax_ce_loss_grad_batched_v2.spv");
    _ = wf.addCopyFile(swiglu_forward_spv, "swiglu_forward.spv");
    _ = wf.addCopyFile(swiglu_backward_spv, "swiglu_backward.spv");
    _ = wf.addCopyFile(rope_partial_batched_spv, "rope_partial_batched.spv");
    _ = wf.addCopyFile(rope_backward_batched_spv, "rope_backward_batched.spv");
    _ = wf.addCopyFile(mlp2_mse_loss_batched_spv, "mlp2_mse_loss_batched.spv");
    _ = wf.addCopyFile(mlp2_ce_loss_batched_spv, "mlp2_ce_loss_batched.spv");
    _ = wf.addCopyFile(cce_forward_spv, "cce_forward.spv");
    _ = wf.addCopyFile(cce_backward_dh_spv, "cce_backward_dh.spv");
    _ = wf.addCopyFile(cce_backward_dw_spv, "cce_backward_dw.spv");
    _ = wf.addCopyFile(qk_rope_partial_batched_spv, "qk_rope_partial_batched.spv");
    _ = wf.addCopyFile(qk_rope_backward_batched_spv, "qk_rope_backward_batched.spv");
    _ = wf.addCopyFile(fa_forward_spv, "fa_forward.spv");
    _ = wf.addCopyFile(fa_decode_split_spv, "fa_decode_split.spv");
    _ = wf.addCopyFile(fa_decode_merge_spv, "fa_decode_merge.spv");
    const shader_mod = wf.add("shaders.zig",
        \\pub const vec_add align(4) = @embedFile("vec_add.spv").*;
        \\pub const matmul_nt align(4) = @embedFile("matmul_nt.spv").*;
        \\pub const matmul_nt_v2 align(4) = @embedFile("matmul_nt_v2.spv").*;
        \\pub const matmul_nt_v2_bf16 align(4) = @embedFile("matmul_nt_v2_bf16.spv").*;
        \\pub const matmul_nt_v2_q4_0 align(4) = @embedFile("matmul_nt_v2_q4_0.spv").*;
        \\pub const matmul_nt_v2_q4_k align(4) = @embedFile("matmul_nt_v2_q4_k.spv").*;
        \\pub const rmsnorm align(4) = @embedFile("rmsnorm.spv").*;
        \\pub const rmsnorm_backward align(4) = @embedFile("rmsnorm_backward.spv").*;
        \\pub const layernorm align(4) = @embedFile("layernorm.spv").*;
        \\pub const layernorm_backward align(4) = @embedFile("layernorm_backward.spv").*;
        \\pub const embedding_backward align(4) = @embedFile("embedding_backward.spv").*;
        \\pub const softmax_backward align(4) = @embedFile("softmax_backward.spv").*;
        \\pub const attn_backward_dattn align(4) = @embedFile("attn_backward_dattn.spv").*;
        \\pub const attn_backward_dv align(4) = @embedFile("attn_backward_dv.spv").*;
        \\pub const attn_backward_dq align(4) = @embedFile("attn_backward_dq.spv").*;
        \\pub const attn_backward_dk align(4) = @embedFile("attn_backward_dk.spv").*;
        \\pub const rope_backward align(4) = @embedFile("rope_backward.spv").*;
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
        \\pub const embed_lookup_batched align(4) = @embedFile("embed_lookup_batched.spv").*;
        \\pub const add_in_place align(4) = @embedFile("add_in_place.spv").*;
        \\pub const attn_decode_single align(4) = @embedFile("attn_decode_single.spv").*;
        \\pub const attn_scores align(4) = @embedFile("attn_scores.spv").*;
        \\pub const attn_scores_train align(4) = @embedFile("attn_scores_train.spv").*;
        \\pub const attn_output align(4) = @embedFile("attn_output.spv").*;
        \\pub const attn_output_train align(4) = @embedFile("attn_output_train.spv").*;
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
        \\pub const attn_synth align(4) = @embedFile("attn_synth.spv").*;
        \\pub const relu align(4) = @embedFile("relu.spv").*;
        \\pub const relu_backward align(4) = @embedFile("relu_backward.spv").*;
        \\pub const linear_backward_dx align(4) = @embedFile("linear_backward_dx.spv").*;
        \\pub const linear_backward_dx_batched align(4) = @embedFile("linear_backward_dx_batched.spv").*;
        \\pub const linear_backward_dw_batched align(4) = @embedFile("linear_backward_dw_batched.spv").*;
        \\pub const outer_product align(4) = @embedFile("outer_product.spv").*;
        \\pub const sgd_step align(4) = @embedFile("sgd_step.spv").*;
        \\pub const adam_step align(4) = @embedFile("adam_step.spv").*;
        \\pub const mse_loss_grad align(4) = @embedFile("mse_loss_grad.spv").*;
        \\pub const mse_loss_grad_scaled align(4) = @embedFile("mse_loss_grad_scaled.spv").*;
        \\pub const mlp2_forward_batched align(4) = @embedFile("mlp2_forward_batched.spv").*;
        \\pub const mlp2_forward_train_batched align(4) = @embedFile("mlp2_forward_train_batched.spv").*;
        \\pub const mlp2_dy_batched align(4) = @embedFile("mlp2_dy_batched.spv").*;
        \\pub const mlp2_dh_pre_batched align(4) = @embedFile("mlp2_dh_pre_batched.spv").*;
        \\pub const mlp2_dw_accum align(4) = @embedFile("mlp2_dw_accum.spv").*;
        \\pub const mlp2_db_accum align(4) = @embedFile("mlp2_db_accum.spv").*;
        \\pub const softmax_ce_loss_grad_batched align(4) = @embedFile("softmax_ce_loss_grad_batched.spv").*;
        \\pub const softmax_ce_loss_grad_batched_v2 align(4) = @embedFile("softmax_ce_loss_grad_batched_v2.spv").*;
        \\pub const swiglu_forward align(4) = @embedFile("swiglu_forward.spv").*;
        \\pub const swiglu_backward align(4) = @embedFile("swiglu_backward.spv").*;
        \\pub const rope_partial_batched align(4) = @embedFile("rope_partial_batched.spv").*;
        \\pub const rope_backward_batched align(4) = @embedFile("rope_backward_batched.spv").*;
        \\pub const mlp2_mse_loss_batched align(4) = @embedFile("mlp2_mse_loss_batched.spv").*;
        \\pub const mlp2_ce_loss_batched align(4) = @embedFile("mlp2_ce_loss_batched.spv").*;
        \\pub const cce_forward align(4) = @embedFile("cce_forward.spv").*;
        \\pub const cce_backward_dh align(4) = @embedFile("cce_backward_dh.spv").*;
        \\pub const cce_backward_dw align(4) = @embedFile("cce_backward_dw.spv").*;
        \\pub const qk_rope_partial_batched align(4) = @embedFile("qk_rope_partial_batched.spv").*;
        \\pub const qk_rope_backward_batched align(4) = @embedFile("qk_rope_backward_batched.spv").*;
        \\pub const fa_forward align(4) = @embedFile("fa_forward.spv").*;
        \\pub const fa_decode_split align(4) = @embedFile("fa_decode_split.spv").*;
        \\pub const fa_decode_merge align(4) = @embedFile("fa_decode_merge.spv").*;
    );

    // ── Public Zig module for host-engine embedding ──
    // `valkyr_gpu` exposes the cooperative-compute surface
    // (Context.attach + Recorder.attachCmd + buffer + pipeline + the
    // SPIR-V shader blobs) so a downstream Zig package — currently
    // Matryoshka — can `@import("valkyr_gpu")` and run kernels into
    // its own per-frame command buffer. Narrow on purpose; see
    // src/lib.zig for the rationale on what's NOT exported.
    //
    // Vulkan + libC are the consumer's responsibility — they have to
    // linkSystemLibrary("vulkan") + linkLibC() on their executable
    // anyway, the same flags the valkyr exe below uses. Letting the
    // host own the link config keeps it singular when one host links
    // against several embedded libraries.
    const valkyr_gpu_mod = b.addModule("valkyr_gpu", .{
        .root_source_file = b.path("src/lib.zig"),
        .target = target,
        .optimize = optimize,
    });
    valkyr_gpu_mod.addAnonymousImport("shaders", .{
        .root_source_file = shader_mod,
    });

    const exe = b.addExecutable(.{
        .name = "valkyr",
        .root_source_file = b.path("src/main.zig"),
        .target = target,
        .optimize = optimize,
    });
    exe.root_module.addAnonymousImport("shaders", .{
        .root_source_file = shader_mod,
    });
    // Vulkan loader: `vulkan-1.lib` on Windows, `libvulkan.so` (linked as
    // `vulkan`) elsewhere. On Windows the LunarG SDK installer sets
    // `VULKAN_SDK`; we use it to add the headers + import-lib paths so
    // `@cInclude("vulkan/vulkan.h")` and the `-lvulkan-1` link both
    // resolve without the user having to copy files into Zig's libc dir.
    if (target.result.os.tag == .windows) {
        if (std.process.getEnvVarOwned(b.allocator, "VULKAN_SDK")) |sdk| {
            exe.addIncludePath(.{ .cwd_relative = b.fmt("{s}/Include", .{sdk}) });
            exe.addLibraryPath(.{ .cwd_relative = b.fmt("{s}/Lib", .{sdk}) });
        } else |_| {}
        exe.linkSystemLibrary("vulkan-1");
    } else {
        exe.linkSystemLibrary("vulkan");
    }
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
