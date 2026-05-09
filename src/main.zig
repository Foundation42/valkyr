//! Entry point. Runs every smoke test we have in sequence so a fresh
//! `zig build run` exercises the whole stack — Vulkan compute, file
//! formats, and (eventually) full-model parity. Each individual test
//! lives in its own function and prints a one-line pass marker on
//! success or surfaces an error otherwise.

const std = @import("std");
const cpu_forward = @import("cpu/forward.zig");
const config_mod = @import("config.zig");
const gpu_model = @import("gpu/model.zig");
const session_mod = @import("session.zig");
const hf_cache = @import("hf_cache.zig");
const commands_inspect = @import("commands/inspect.zig");
const commands_encode = @import("commands/encode.zig");
const commands_bench = @import("commands/bench.zig");
const commands_serve = @import("commands/serve.zig");
const commands_chat = @import("commands/chat.zig");
const commands_finetune = @import("commands/finetune.zig");
const commands_gen_from_ckpt = @import("commands/gen_from_ckpt.zig");
const smoke_gpu_gen = @import("smoke/gpu_gen.zig");
const smoke_session = @import("smoke/session.zig");
const smoke_misc = @import("smoke/misc.zig");
const smoke_cpu = @import("smoke/cpu.zig");
const smoke_training = @import("smoke/training.zig");
const smoke_layer_tests = @import("smoke/layer_tests.zig");
const smoke_decoder = @import("smoke/decoder.zig");
const smoke_gpu_kernels = @import("smoke/gpu_kernels.zig");
const smoke_gpu_train = @import("smoke/gpu_train.zig");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    const args = try std.process.argsAlloc(allocator);
    defer std.process.argsFree(allocator, args);

    if (args.len >= 2 and std.mem.eql(u8, args[1], "--list")) {
        try commands_inspect.runList(allocator);
        return;
    }
    if (args.len >= 2 and std.mem.eql(u8, args[1], "--cce-forward-smoke")) {
        // Fast-path for iterating on the CCE forward kernel.
        // Drives `cce_forward.comp` against the cce.zig CPU oracle on
        // three Qwen-flavoured shapes (multi-chunk, partial-chunk,
        // single-chunk).
        try smoke_gpu_train.runGpuCceForwardSmoke(allocator);
        return;
    }
    if (args.len >= 2 and std.mem.eql(u8, args[1], "--cce-backward-dh-smoke")) {
        // Fast-path for the d_h half of CCE backward.
        try smoke_gpu_train.runGpuCceBackwardDhSmoke(allocator);
        return;
    }
    if (args.len >= 2 and std.mem.eql(u8, args[1], "--cce-backward-dw-smoke")) {
        // Fast-path for the dW half of CCE backward.
        try smoke_gpu_train.runGpuCceBackwardDwSmoke(allocator);
        return;
    }
    if (args.len >= 2 and std.mem.eql(u8, args[1], "--real-train-step-smoke")) {
        // β-5 end-to-end real-model train. Loads Qwen3-0.6B from HF cache,
        // runs one Adam step, asserts CE_after < CE_before. The natural
        // parity gate for the CCE wiring (chunk 4).
        try smoke_decoder.runRealModelTrainStepSmoke(allocator);
        return;
    }
    if (args.len >= 2 and std.mem.eql(u8, args[1], "--real-lora-q-step-smoke")) {
        // A4-2 perf gate: same as --real-train-step-smoke but with
        // LoRA-Q enabled (rank-16, α=32). Validates the LoRA-Q wiring
        // at production scale (28 layers × 1024-dim W_q) and reports
        // ms/step for the perf snapshot.
        try smoke_decoder.runRealModelTrainStepLoraQSmoke(allocator);
        return;
    }
    if (args.len >= 2 and std.mem.eql(u8, args[1], "--real-lora-all-step-smoke")) {
        // A4-3 perf gate: LoRA on every dense matmul at Qwen3-0.6B
        // β-5 shape (rank-16, α=32, all 7 projections per layer).
        // Validates the LoRA-all wiring at production scale and
        // reports ms/step (compare against the LoRA-Q baseline +
        // no-LoRA baseline for the per-projection overhead curve).
        try smoke_decoder.runRealModelTrainStepLoraAllSmoke(allocator);
        return;
    }
    if (args.len >= 2 and std.mem.eql(u8, args[1], "--real-multi-step-smoke")) {
        // β-6a multi-step training loop. Same setup as β-5 but runs 30
        // Adam steps on the same batch, asserts CE drops ≥90%. Validates
        // the loop holds together past step 1 (Adam state, no NaN bloom).
        try smoke_decoder.runRealModelMultiStepSmoke(allocator);
        return;
    }
    if (args.len >= 2 and std.mem.eql(u8, args[1], "--checkpoint-smoke")) {
        // β-6b checkpoint save/load round-trip on a toy 2-layer stack.
        // Trains K, saves, trains M more for the "continuous" baseline,
        // spins up a fresh Runner, loads, trains M, asserts trajectory
        // matches. Gates round-trip integrity + Adam state recovery.
        try smoke_decoder.runDecoderStackCheckpointSmoke(allocator);
        return;
    }
    if (args.len >= 2 and std.mem.eql(u8, args[1], "--lora-checkpoint-smoke")) {
        // A4-4 .lvkpt round-trip. Same pattern as --checkpoint-smoke
        // but the Runner has LoRA on all_attn (rank-4) and saves to
        // the LoRA-only `.lvkpt` format. Gates that A/B + Adam m/v
        // for every enabled projection round-trip cleanly + that the
        // saved-and-resumed trajectory matches the continuous one.
        try smoke_decoder.runDecoderStackLoraCheckpointSmoke(allocator);
        return;
    }
    if (args.len >= 2 and std.mem.eql(u8, args[1], "--lora-merge-smoke")) {
        // Math identity gate for `--chat --lora-ckpt`. Verifies that
        // `forward(W + (α/r)·B·A)` matches the explicit LoRA forward
        // at fp32 (1e-5 rel-err) and within bf16 round-trip noise
        // (1e-2 rel-err) — so the chat-path merge will produce the
        // same logits as the slow `--gen-from-ckpt` training path.
        try smoke_cpu.runLoraMergeMathSmoke(allocator);
        return;
    }
    if (args.len >= 2 and std.mem.eql(u8, args[1], "--flash-attention-smoke")) {
        // F2 of the FlashAttention arc: CPU oracle parity vs the
        // standard 3-pass attention forward. Gates the tile-on-Q
        // online-softmax algorithm at decode + prefill-causal + GQA
        // + non-aligned-block shapes, plus an LSE-output check that
        // the per-row log-sum-exp matches log Σ exp(scores).
        try smoke_cpu.runFlashAttentionParitySmoke(allocator);
        return;
    }
    if (args.len >= 2 and std.mem.eql(u8, args[1], "--flash-attention-bw-smoke")) {
        // F6a of the FlashAttention arc: CPU oracle parity for the
        // FA-2 backward (Dao 2023, Algorithm 4). Recomputes softmax
        // inline from saved O + LSE — replaces the [n_q × n_heads ×
        // n_kv] attn-matrix consumption of the 3-pass `attentionBackward`
        // with a per-row D = Σ_d O · dO reduction, then accumulates
        // dQ/dK/dV the same way. Five shape cases mirror the F2 forward
        // smoke; tolerance 1e-4 rel-err.
        try smoke_cpu.runFlashAttentionBackwardParitySmoke(allocator);
        return;
    }
    if (args.len >= 2 and std.mem.eql(u8, args[1], "--flash-attention-tq4v-smoke")) {
        // T1 of the fused-TQ4 arc: CPU parity for the fused FA decode
        // path that reads V from the GPU TQ4 cache layout (33 u32 per
        // 256-element block). Compares against standard FA on V values
        // pre-dequanted via the same `dequantizeBlockTQ4` round-trip,
        // so the only difference is the inner-loop fusion. Tolerance
        // 1e-5 rel-err; TQ4 reconstruction quality is gated separately
        // by `--turboquant-smoke`.
        try smoke_cpu.runFlashAttentionTq4VParitySmoke(allocator);
        return;
    }
    if (args.len >= 2 and std.mem.eql(u8, args[1], "--forward-hybrid-batched-smoke")) {
        // MTP-1c-β-1: CPU oracle for n_q-batched hybrid forward. Loads
        // Qwen3.5-0.8B, runs the same 4-token sequence through both the
        // sequential `forwardHybrid` and the new `forwardHybridBatched`
        // from a fresh `HybridState`. Gates per-row logits parity (1e-5
        // rel-err), determinism, and top-1 match. Reference for every
        // downstream GPU parity gate in the batched-prefill arc.
        try smoke_cpu.runForwardHybridBatchedCpuSmoke(allocator);
        return;
    }
    if (args.len >= 2 and std.mem.eql(u8, args[1], "--mtp-forward-cpu-smoke")) {
        // MTP-1b-α: build-and-shape gate for the CPU MTP forward oracle.
        // Loads Qwen3.5-0.8B (smallest MTP-equipped checkpoint), runs
        // `forwardMtpStep` on a synthetic h_prev for two different tokens,
        // checks numerical health (no NaN/Inf), determinism across
        // re-runs, and that the output actually depends on the input.
        // Tighter CPU/HF-reference parity lands in MTP-1b-β alongside the
        // GPU recorder.
        try smoke_cpu.runMtpForwardCpuSmoke(allocator);
        return;
    }
    if (args.len >= 2 and std.mem.eql(u8, args[1], "--mtp-gpu-upload-smoke")) {
        // MTP-1b-β-1: GPU upload scaffold for the MTP head. Loads
        // Qwen3.5-0.8B, uploads at .fp32_all precision, asserts
        // `gm.mtp_head` non-null and that every uploaded buffer matches
        // its expected fp32 footprint (allowing capacity ≥ data — the
        // pool's slack policy permits over-alloc). The recorder + parity
        // gate land in MTP-1b-β-2.
        try smoke_gpu_train.runMtpGpuUploadSmoke(allocator);
        return;
    }
    if (args.len >= 2 and std.mem.eql(u8, args[1], "--mtp-forward-gpu-smoke")) {
        // MTP-1b-β-2: end-to-end GPU/CPU parity gate for the MTP forward.
        // Loads Qwen3.5-0.8B at fp32_all, runs `recordMtpStep` and
        // `forwardMtpStep` with identical synthetic h_prev + token,
        // compares logits at 1e-4 max rel-err. Single-step gate; the
        // draft+verify chain lands in MTP-1c.
        try smoke_gpu_train.runMtpForwardGpuSmoke(allocator);
        return;
    }
    if (args.len >= 2 and std.mem.eql(u8, args[1], "--mtp-draft-chain-smoke")) {
        // MTP-1c-α: end-to-end MTP draft chain demo. Main forward at
        // pos=0 with BOS → snapshot last hidden, sample first token →
        // recordMtpStep recursively for k draft slots, sampling each
        // top-1 token. Decodes via the loaded tokenizer and prints the
        // chain. Verify path lands in MTP-1c-β.
        try smoke_gpu_train.runMtpDraftChainSmoke(allocator);
        return;
    }
    if (args.len >= 2 and std.mem.eql(u8, args[1], "--fa-forward-smoke")) {
        // F3 of the FlashAttention arc: GPU SPIR-V kernel parity vs
        // the F2 CPU oracle. Drives `shaders/fa_forward.comp` (one
        // workgroup per (q, h), tile-on-K with online softmax in
        // shared mem) on the same 5 shape cases as F2; tolerance
        // 1e-4 rel-err (reduction-order divergence between cooperative
        // subgroup ops and serial CPU accumulation).
        try smoke_gpu_train.runFlashAttentionGpuSmoke(allocator);
        return;
    }
    if (args.len >= 2 and std.mem.eql(u8, args[1], "--fa-runner-smoke")) {
        // F4 of the FlashAttention arc: end-to-end parity gate that
        // `Runner.forwardLogits` with `attn_use_fa = true` produces
        // the same logits as the 3-pass chain. Toy decoder shape
        // (head_dim ≤ HEAD_DIM_MAX so the FA branch is reachable);
        // 1e-4 rel-err tolerance.
        try smoke_gpu_train.runFaRunnerSmoke(allocator);
        return;
    }
    if (args.len >= 2 and std.mem.eql(u8, args[1], "--fa-decode-smoke")) {
        // F5 of the FlashAttention arc: FlashDecoding (Tri Dao 2023)
        // split-K kernels for decode at long ctx, where fa_forward's
        // n_heads-only parallelism starves the SMs. Phase 1
        // (`fa_decode_split`) shards K across n_heads × n_splits WGs
        // and emits unnormalised partial (O, m, l) triples; phase 2
        // (`fa_decode_merge`) combines them. Parity vs fa_forward at
        // 5 ctx-length × split-count cases incl. partial-tail split;
        // 1e-4 rel-err tolerance.
        try smoke_gpu_train.runFlashDecodingGpuSmoke(allocator);
        return;
    }
    if (args.len >= 2 and std.mem.eql(u8, args[1], "--fa-decode-tq4v-smoke")) {
        // T2 of the fused-TQ4 arc: GPU parity for the fused FlashDecoding
        // path that reads V from the GPU TQ4 cache layout (33 u32 per
        // 256-element block) and dequants inline per K-tile. Parity vs
        // T1 CPU oracle; both consume identical TQ4 cache bits.
        try smoke_gpu_train.runFaDecodeTq4VGpuSmoke(allocator);
        return;
    }
    if (args.len >= 2 and std.mem.eql(u8, args[1], "--fa-bw-smoke")) {
        // F6b of the FlashAttention arc: GPU SPIR-V parity for the
        // 3-kernel FA-2 backward chain (`fa_bw_d → fa_bw_dq → fa_bw_dkv`)
        // vs the F6a CPU oracle. Cooperative subgroup reductions diverge
        // from serial CPU accumulation in fp32 rounding order, so
        // tolerance is 1e-4 rel-err (in practice ~1e-6 to 1e-7).
        try smoke_gpu_train.runFlashAttentionBackwardGpuSmoke(allocator);
        return;
    }
    if (args.len >= 2 and std.mem.eql(u8, args[1], "--fa-decode-chat-smoke")) {
        // F5b of the FlashAttention arc: end-to-end parity gate that
        // the FlashDecoding swap-in inside `recordOneLayer` (chat decode
        // path) produces the same attention output as the original
        // 3-pass chain it replaces. Drives both paths off the production
        // `chooseFaDecodeSplit` heuristic + the same push shapes
        // `computeForwardPushes` builds, at Qwen3-0.6B per-layer dims
        // (n_heads=16 GQA 16:8 head_dim=128); 1e-4 rel-err tolerance.
        try smoke_gpu_train.runFaDecodeChatPathSmoke(allocator);
        return;
    }
    if (args.len >= 2 and std.mem.eql(u8, args[1], "--real-sampling-smoke")) {
        // β-6c sampled-text-shift validation. Greedy-samples N tokens
        // from a probe prompt before and after fine-tuning on a single
        // batch, asserts at least one generated token differs (training
        // visibly shifted argmax).
        try smoke_decoder.runRealModelSamplingSmoke(allocator);
        return;
    }
    if (args.len >= 3 and (std.mem.eql(u8, args[1], "--fine-tune") or std.mem.eql(u8, args[1], "--lora-finetune"))) {
        // User-facing fine-tune driver: load real Qwen3-class weights,
        // train N Adam steps on a JSONL dataset, optionally save a
        // checkpoint, optionally show a probe sample before/after.
        //
        // `--fine-tune` defaults to full-weight training (every param
        // trained, .vkpt save format). Pass `--lora-targets` (and
        // optionally `--lora-rank` / `--lora-alpha` / `--lora-lr-b-scale`)
        // to switch into LoRA mode (only A,B per enabled projection
        // trained; .lvkpt save format).
        //
        // `--lora-finetune` is the discoverable alias — same backend,
        // but requires `--lora-targets` and bumps the lr default to
        // 5e-4 (typical LoRA fine-tune rate is 10-50× full-FT lr).
        // Format:
        //   --fine-tune <model> --data <jsonl> [--steps N] [--lr LR]
        //                                       [--n-pos N] [--batch IDX]
        //                                       [--rotate] [--probe TEXT]
        //                                       [--n-gen N] [--out PATH]
        //                                       [--print-every K]
        //                                       [--lora-targets q,k,v,o,gate,up,down,all_attn,all_ffn,all]
        //                                       [--lora-rank N] [--lora-alpha A]
        //                                       [--lora-lr-b-scale L]
        const is_lora_alias = std.mem.eql(u8, args[1], "--lora-finetune");
        var opts = commands_finetune.Options{
            .model = args[2],
            .data_path = "",
            .lr = if (is_lora_alias) 5e-4 else 1e-5,
        };
        var i: usize = 3;
        while (i < args.len) {
            const a = args[i];
            if (std.mem.eql(u8, a, "--data") and i + 1 < args.len) {
                opts.data_path = args[i + 1];
                i += 2;
            } else if (std.mem.eql(u8, a, "--steps") and i + 1 < args.len) {
                opts.n_steps = try std.fmt.parseInt(u32, args[i + 1], 10);
                i += 2;
            } else if (std.mem.eql(u8, a, "--lr") and i + 1 < args.len) {
                opts.lr = try std.fmt.parseFloat(f32, args[i + 1]);
                i += 2;
            } else if (std.mem.eql(u8, a, "--n-pos") and i + 1 < args.len) {
                opts.n_pos = try std.fmt.parseInt(u32, args[i + 1], 10);
                i += 2;
            } else if (std.mem.eql(u8, a, "--batch") and i + 1 < args.len) {
                opts.batch_idx = try std.fmt.parseInt(u32, args[i + 1], 10);
                i += 2;
            } else if (std.mem.eql(u8, a, "--rotate")) {
                opts.rotate = true;
                i += 1;
            } else if (std.mem.eql(u8, a, "--probe") and i + 1 < args.len) {
                opts.probe = args[i + 1];
                i += 2;
            } else if (std.mem.eql(u8, a, "--n-gen") and i + 1 < args.len) {
                opts.n_gen = try std.fmt.parseInt(u32, args[i + 1], 10);
                i += 2;
            } else if (std.mem.eql(u8, a, "--out") and i + 1 < args.len) {
                opts.out_path = args[i + 1];
                i += 2;
            } else if (std.mem.eql(u8, a, "--print-every") and i + 1 < args.len) {
                opts.print_every = try std.fmt.parseInt(u32, args[i + 1], 10);
                i += 2;
            } else if (std.mem.eql(u8, a, "--lora-targets") and i + 1 < args.len) {
                opts.lora_targets = commands_finetune.parseLoraTargets(args[i + 1]) catch |err| {
                    std.debug.print("--lora-targets: invalid spec \"{s}\" ({s}). Valid names: q, k, v, o, gate, up, down, all_attn, all_ffn, all (comma-separated).\n", .{ args[i + 1], @errorName(err) });
                    return err;
                };
                i += 2;
            } else if (std.mem.eql(u8, a, "--lora-rank") and i + 1 < args.len) {
                opts.lora_rank = try std.fmt.parseInt(u32, args[i + 1], 10);
                i += 2;
            } else if (std.mem.eql(u8, a, "--lora-alpha") and i + 1 < args.len) {
                opts.lora_alpha = try std.fmt.parseFloat(f32, args[i + 1]);
                i += 2;
            } else if (std.mem.eql(u8, a, "--lora-lr-b-scale") and i + 1 < args.len) {
                opts.lora_lr_b_scale = try std.fmt.parseFloat(f32, args[i + 1]);
                i += 2;
            } else {
                i += 1;
            }
        }
        if (opts.data_path.len == 0) {
            std.debug.print("{s}: --data <jsonl> is required\n", .{args[1]});
            return error.MissingDataArg;
        }
        if (is_lora_alias and opts.lora_targets == 0) {
            std.debug.print("--lora-finetune: --lora-targets <spec> is required (e.g. q,k,v,o or all_attn or all)\n", .{});
            return error.MissingLoraTargets;
        }
        try commands_finetune.run(allocator, opts);
        return;
    }
    if (args.len >= 3 and std.mem.eql(u8, args[1], "--gen-from-ckpt")) {
        // Generate text from a `.vkpt` or `.lvkpt` checkpoint produced
        // by --fine-tune / --lora-finetune. Uses the training Runner's
        // forwardLogits + greedyDecode (slow, ~150 ms/tok at Qwen3-0.6B
        // Debug; complete loop without needing .vkpt support in the
        // inference Session). Magic-sniffed at load time, so the same
        // command handles both formats.
        //
        // For `.lvkpt`: pass the same `--lora-targets` / `--lora-rank`
        // the checkpoint was saved with so the Runner allocates
        // matching adapter slots before loadLoraCheckpoint overwrites
        // them. Mismatches return error.LoraCheckpointConfigMismatch.
        //
        // Format:
        //   --gen-from-ckpt <model> --ckpt <path> --prompt TEXT
        //                                          [--n-gen N] [--n-pos N]
        //                                          [--lora-targets ...] [--lora-rank N]
        //                                          [--lora-alpha A] [--lora-lr-b-scale L]
        var opts = commands_gen_from_ckpt.Options{
            .model = args[2],
            .ckpt_path = "",
            .prompt = "",
        };
        var i: usize = 3;
        while (i < args.len) {
            const a = args[i];
            if (std.mem.eql(u8, a, "--ckpt") and i + 1 < args.len) {
                opts.ckpt_path = args[i + 1];
                i += 2;
            } else if (std.mem.eql(u8, a, "--prompt") and i + 1 < args.len) {
                opts.prompt = args[i + 1];
                i += 2;
            } else if (std.mem.eql(u8, a, "--n-gen") and i + 1 < args.len) {
                opts.n_gen = try std.fmt.parseInt(u32, args[i + 1], 10);
                i += 2;
            } else if (std.mem.eql(u8, a, "--n-pos") and i + 1 < args.len) {
                opts.n_pos = try std.fmt.parseInt(u32, args[i + 1], 10);
                i += 2;
            } else if (std.mem.eql(u8, a, "--lora-targets") and i + 1 < args.len) {
                opts.lora_targets = commands_finetune.parseLoraTargets(args[i + 1]) catch |err| {
                    std.debug.print("--lora-targets: invalid spec \"{s}\" ({s})\n", .{ args[i + 1], @errorName(err) });
                    return err;
                };
                i += 2;
            } else if (std.mem.eql(u8, a, "--lora-rank") and i + 1 < args.len) {
                opts.lora_rank = try std.fmt.parseInt(u32, args[i + 1], 10);
                i += 2;
            } else if (std.mem.eql(u8, a, "--lora-alpha") and i + 1 < args.len) {
                opts.lora_alpha = try std.fmt.parseFloat(f32, args[i + 1]);
                i += 2;
            } else if (std.mem.eql(u8, a, "--lora-lr-b-scale") and i + 1 < args.len) {
                opts.lora_lr_b_scale = try std.fmt.parseFloat(f32, args[i + 1]);
                i += 2;
            } else {
                i += 1;
            }
        }
        if (opts.ckpt_path.len == 0) {
            std.debug.print("--gen-from-ckpt: --ckpt <path> is required\n", .{});
            return error.MissingCkptArg;
        }
        if (opts.prompt.len == 0) {
            std.debug.print("--gen-from-ckpt: --prompt TEXT is required\n", .{});
            return error.MissingPromptArg;
        }
        try commands_gen_from_ckpt.run(allocator, opts);
        return;
    }
    if (args.len >= 2 and std.mem.eql(u8, args[1], "--decoder-stack-train-smoke")) {
        // 200-step synthetic-stack convergence smoke. Stronger trajectory
        // gate than the one-step real-model smoke: asserts final/initial
        // CE ratio < 1e-2 over 200 Adam steps. Tiny dims (dim=16, vocab=8)
        // for fast iteration.
        try smoke_decoder.runDecoderStackTrainGpuSmoke(allocator);
        return;
    }
    if (args.len >= 2 and std.mem.eql(u8, args[1], "--cce-bench")) {
        // Time each CCE kernel in isolation at Qwen3-0.6B shape. No
        // parity — just dispatch with vkQueueWaitIdle between calls
        // and report per-kernel ms. Shows where the LM-head bucket's
        // time actually goes.
        try smoke_gpu_train.runCceBench(allocator);
        return;
    }
    if (args.len >= 2 and std.mem.eql(u8, args[1], "--attn-bench")) {
        // F1 of the FlashAttention arc: time the 3-dispatch attention
        // chain (scores → softmax → output) at Qwen3-0.6B per-layer
        // dims across decode (n_q=1) and prefill-causal (n_q==n_kv)
        // n_kv sweeps. Sizes the scores-buffer cliff and the
        // baseline FA has to beat.
        try smoke_gpu_train.runAttnBench(allocator);
        return;
    }
    if (args.len >= 2 and std.mem.eql(u8, args[1], "--attn-bench-d256")) {
        // Same sweeps as --attn-bench but at Qwen3.5 0.8B per-layer
        // shape (n_heads=8 n_kv_heads=2 head_dim=256). Exercises the
        // BC=8 d=256 SPIR-V variants of fa_forward + fa_decode_split.
        try smoke_gpu_train.runAttnBenchD256(allocator);
        return;
    }
    if (args.len >= 2 and std.mem.eql(u8, args[1], "--lora-smoke")) {
        // GPU LoRA composition smoke. Drives existing matmul_nt_v2 +
        // linear_backward_d{x,w}_batched + scale + add_in_place
        // kernels in the LoRA pattern (no new SPIR-V) and parity-checks
        // the outputs against the cpu/lora.zig oracle.
        try smoke_gpu_train.runGpuLoraSmoke(allocator);
        return;
    }
    if (args.len >= 2 and std.mem.eql(u8, args[1], "--lora-rec-smoke")) {
        // A4-1 foundation smoke. Same parity gate as --lora-smoke but
        // routed through gpu_recorder.Recorder + the helpers in
        // src/train/lora.zig — one cmdbuf + one submit per case
        // instead of 14 separate submitOneShot calls. This is the
        // path the in-Runner LoRA integration will use.
        try smoke_gpu_train.runGpuLoraRecorderSmoke(allocator);
        return;
    }
    if (args.len >= 2 and std.mem.eql(u8, args[1], "--lora-q-runner-smoke")) {
        // A4-2 wire test. Drives `train_transformer.Runner.step` with
        // `lora_targets = LoraTarget.q` end-to-end and asserts (1)
        // step-0 forward parity vs the same Runner with LoRA off (B=0
        // ⇒ delta = 0), (2) W_q is unchanged after N Adam steps (LoRA
        // freezes W_q), (3) loss decreases.
        try smoke_gpu_train.runGpuTransformerLoraQSmoke(allocator);
        return;
    }
    if (args.len >= 2 and std.mem.eql(u8, args[1], "--lora-all-runner-smoke")) {
        // A4-3 wire test. Same toy shape and three assertions as
        // --lora-q-runner-smoke but with LoRA on every dense matmul
        // (Q + K + V + O + gate + up + down). Verifies that all 7 W's
        // are frozen after training and the Runner converges with the
        // larger LoRA budget.
        try smoke_gpu_train.runGpuTransformerLoraAllSmoke(allocator);
        return;
    }
    if (args.len >= 2 and std.mem.eql(u8, args[1], "--lora-train-demo")) {
        // End-to-end LoRA training demo: synthetic problem with a target
        // rank-r delta on a frozen base, train Adam-LoRA to recover it.
        // The "does it actually train?" gate for LoRA on GPU.
        try smoke_gpu_train.runGpuLoraTrainDemo(allocator);
        return;
    }
    if (args.len >= 2 and std.mem.eql(u8, args[1], "--lora-plus-demo")) {
        // LoRA+ comparative demo: same task at η_B/η_A ∈ {1, 4, 16}.
        // Demonstrates the Chronicals/Hayou prediction that B should
        // learn ~16× faster than A in the feature-learning regime.
        try smoke_gpu_train.runGpuLoraPlusDemo(allocator);
        return;
    }
    if (args.len >= 3 and std.mem.eql(u8, args[1], "--inspect")) {
        const dir = try hf_cache.resolveModelArg(allocator, args[2]);
        defer allocator.free(dir);
        try commands_inspect.runInspect(allocator, dir);
        return;
    }
    if (args.len >= 3 and std.mem.eql(u8, args[1], "--load")) {
        const dir = try hf_cache.resolveModelArg(allocator, args[2]);
        defer allocator.free(dir);
        try commands_inspect.runLoad(allocator, dir);
        return;
    }
    if (args.len >= 3 and std.mem.eql(u8, args[1], "--config")) {
        const dir = try hf_cache.resolveModelArg(allocator, args[2]);
        defer allocator.free(dir);
        try commands_inspect.runConfig(allocator, dir);
        return;
    }
    if (args.len >= 4 and std.mem.eql(u8, args[1], "--qwen35-layer-test")) {
        // args[2] = model dir, args[3] = path to layer-0 reference
        // dump from scripts/dump_qwen35_layer0.py.
        try smoke_layer_tests.runQwen35LayerTest(allocator, args[2], args[3]);
        return;
    }
    if (args.len >= 4 and std.mem.eql(u8, args[1], "--dump-embed")) {
        const token_id = try std.fmt.parseInt(u32, args[3], 10);
        try commands_inspect.runDumpEmbed(allocator, args[2], token_id);
        return;
    }
    if (args.len >= 4 and std.mem.eql(u8, args[1], "--rmsnorm-test")) {
        const token_id = try std.fmt.parseInt(u32, args[3], 10);
        try smoke_layer_tests.runRmsnormTest(allocator, args[2], token_id);
        return;
    }
    if (args.len >= 4 and std.mem.eql(u8, args[1], "--qproj-test")) {
        const token_id = try std.fmt.parseInt(u32, args[3], 10);
        try smoke_layer_tests.runQprojTest(allocator, args[2], token_id);
        return;
    }
    if (args.len >= 4 and std.mem.eql(u8, args[1], "--rope-test")) {
        const token_id = try std.fmt.parseInt(u32, args[3], 10);
        try smoke_layer_tests.runRopeTest(allocator, args[2], token_id);
        return;
    }
    if (args.len >= 4 and std.mem.eql(u8, args[1], "--attention-test")) {
        const token_id = try std.fmt.parseInt(u32, args[3], 10);
        try smoke_layer_tests.runAttentionTest(allocator, args[2], token_id);
        return;
    }
    if (args.len >= 4 and std.mem.eql(u8, args[1], "--layer0-test")) {
        const token_id = try std.fmt.parseInt(u32, args[3], 10);
        try smoke_layer_tests.runLayer0Test(allocator, args[2], token_id);
        return;
    }
    if (args.len >= 4 and std.mem.eql(u8, args[1], "--gen")) {
        const token_id = try std.fmt.parseInt(u32, args[3], 10);
        try smoke_layer_tests.runGen(allocator, args[2], token_id);
        return;
    }
    if (args.len >= 4 and std.mem.eql(u8, args[1], "--gpu-qproj-test")) {
        const token_id = try std.fmt.parseInt(u32, args[3], 10);
        try smoke_layer_tests.runGpuQprojTest(allocator, args[2], token_id);
        return;
    }
    if (args.len >= 4 and std.mem.eql(u8, args[1], "--gpu-rmsnorm-test")) {
        const token_id = try std.fmt.parseInt(u32, args[3], 10);
        try smoke_layer_tests.runGpuRmsnormTest(allocator, args[2], token_id);
        return;
    }
    if (args.len >= 4 and std.mem.eql(u8, args[1], "--gpu-geglu-test")) {
        const token_id = try std.fmt.parseInt(u32, args[3], 10);
        try smoke_layer_tests.runGpuGegluTest(allocator, args[2], token_id);
        return;
    }
    if (args.len >= 4 and std.mem.eql(u8, args[1], "--gpu-rope-test")) {
        const token_id = try std.fmt.parseInt(u32, args[3], 10);
        try smoke_layer_tests.runGpuRopeTest(allocator, args[2], token_id);
        return;
    }
    if (args.len >= 3 and std.mem.eql(u8, args[1], "--gpu-load")) {
        try smoke_layer_tests.runGpuLoad(allocator, args[2]);
        return;
    }
    if (args.len >= 4 and std.mem.eql(u8, args[1], "--gpu-layer0-test")) {
        const token_id = try std.fmt.parseInt(u32, args[3], 10);
        try smoke_layer_tests.runGpuLayer0Test(allocator, args[2], token_id);
        return;
    }
    if (args.len >= 4 and std.mem.eql(u8, args[1], "--gpu-gen")) {
        const token_id = try std.fmt.parseInt(u32, args[3], 10);
        try smoke_gpu_gen.runGpuGen(allocator, args[2], token_id);
        return;
    }
    if (args.len >= 4 and std.mem.eql(u8, args[1], "--gpu-gen-qwen35")) {
        const token_id = try std.fmt.parseInt(u32, args[3], 10);
        try smoke_gpu_gen.runGpuGenQwen35(allocator, args[2], token_id);
        return;
    }
    if (args.len >= 5 and std.mem.eql(u8, args[1], "--gpu-gen-many")) {
        const token_id = try std.fmt.parseInt(u32, args[3], 10);
        const n_tokens = try std.fmt.parseInt(usize, args[4], 10);
        try smoke_gpu_gen.runGpuGenMany(allocator, args[2], token_id, n_tokens);
        return;
    }
    if (args.len >= 4 and std.mem.eql(u8, args[1], "--encode")) {
        const dir = try hf_cache.resolveModelArg(allocator, args[2]);
        defer allocator.free(dir);
        try commands_encode.runEncode(allocator, dir, args[3]);
        return;
    }
    if (args.len >= 4 and std.mem.eql(u8, args[1], "--tq4-kv-test")) {
        const token_id = try std.fmt.parseInt(u32, args[3], 10);
        try smoke_layer_tests.runTq4KvTest(allocator, args[2], token_id);
        return;
    }
    if (args.len >= 4 and std.mem.eql(u8, args[1], "--gen-tq4v")) {
        const token_id = try std.fmt.parseInt(u32, args[3], 10);
        try smoke_layer_tests.runGenTq4V(allocator, args[2], token_id);
        return;
    }
    if (args.len >= 4 and std.mem.eql(u8, args[1], "--gpu-gen-tq4v")) {
        const token_id = try std.fmt.parseInt(u32, args[3], 10);
        try smoke_gpu_gen.runGpuGenTq4V(allocator, args[2], token_id);
        return;
    }
    if (args.len >= 2 and std.mem.eql(u8, args[1], "--train-demo")) {
        // Headless training demo. Streams a time-dependent (input,
        // target) signal through the TrainingRunner and prints a loss
        // curve. No model file required; weights start random and
        // converge in a few hundred steps. Format:
        //   --train-demo [--steps N] [--hidden H] [--lr L] [--print-every K]
        var steps: u32 = 600;
        var hidden: u32 = 16;
        var lr: f32 = 0.05;
        var print_every: u32 = 50;
        var i: usize = 2;
        while (i < args.len) {
            const a = args[i];
            if (std.mem.eql(u8, a, "--steps") and i + 1 < args.len) {
                steps = try std.fmt.parseInt(u32, args[i + 1], 10);
                i += 2;
            } else if (std.mem.eql(u8, a, "--hidden") and i + 1 < args.len) {
                hidden = try std.fmt.parseInt(u32, args[i + 1], 10);
                i += 2;
            } else if (std.mem.eql(u8, a, "--lr") and i + 1 < args.len) {
                lr = try std.fmt.parseFloat(f32, args[i + 1]);
                i += 2;
            } else if (std.mem.eql(u8, a, "--print-every") and i + 1 < args.len) {
                print_every = try std.fmt.parseInt(u32, args[i + 1], 10);
                i += 2;
            } else {
                i += 1;
            }
        }
        try smoke_training.runTrainDemo(allocator, steps, hidden, lr, print_every);
        return;
    }
    if (args.len >= 2 and std.mem.eql(u8, args[1], "--train-demo-n")) {
        // Multi-layer headless training demo. Same task and CLI shape as
        // --train-demo but driven by MlpNRunner with `--layers d0,d1,...`.
        // Default: 4,16,12,4 (n=3).
        //   --train-demo-n [--steps N] [--layers a,b,...] [--lr L] [--print-every K]
        var steps: u32 = 1200;
        var layers_csv: []const u8 = "4,16,12,4";
        var lr: f32 = 0.05;
        var print_every: u32 = 100;
        var i: usize = 2;
        while (i < args.len) {
            const a = args[i];
            if (std.mem.eql(u8, a, "--steps") and i + 1 < args.len) {
                steps = try std.fmt.parseInt(u32, args[i + 1], 10);
                i += 2;
            } else if (std.mem.eql(u8, a, "--layers") and i + 1 < args.len) {
                layers_csv = args[i + 1];
                i += 2;
            } else if (std.mem.eql(u8, a, "--lr") and i + 1 < args.len) {
                lr = try std.fmt.parseFloat(f32, args[i + 1]);
                i += 2;
            } else if (std.mem.eql(u8, a, "--print-every") and i + 1 < args.len) {
                print_every = try std.fmt.parseInt(u32, args[i + 1], 10);
                i += 2;
            } else {
                i += 1;
            }
        }
        try smoke_training.runTrainDemoN(allocator, steps, layers_csv, lr, print_every);
        return;
    }
    if (args.len >= 3 and std.mem.eql(u8, args[1], "--bench")) {
        var n: usize = 64;
        var tq4v: bool = false;
        var i: usize = 3;
        while (i < args.len) {
            if (std.mem.eql(u8, args[i], "--n") and i + 1 < args.len) {
                n = try std.fmt.parseInt(usize, args[i + 1], 10);
                i += 2;
            } else if (std.mem.eql(u8, args[i], "--tq4v")) {
                tq4v = true;
                i += 1;
            } else {
                i += 1;
            }
        }
        const dir = try hf_cache.resolveModelArg(allocator, args[2]);
        defer allocator.free(dir);
        try commands_bench.runBench(allocator, dir, n, tq4v);
        return;
    }
    if (args.len >= 3 and std.mem.eql(u8, args[1], "--session-smoke")) {
        // Headless exercise of valkyr.session.Session — the same code
        // path matryoshka's ai_demo runs in-engine. Lets us validate
        // dense AND hybrid families through the embed surface without
        // a GUI. Format:
        //   --session-smoke <model> [--prompt "..."] [--max-new N]
        //                           [--budget K] [--budget-us US]
        //                           [--q4|--q4k]
        // --budget       → Budget.layers   (default 8)
        // --budget-us    → Budget.microseconds
        // both           → Budget.either
        var prompt: []const u8 = "Once upon a time";
        var smoke_max_new: u32 = 20;
        var budget_layers: ?u32 = null;
        var budget_us: ?u64 = null;
        var q4: bool = false;
        var q4k: bool = false;
        var i: usize = 3;
        while (i < args.len) {
            const a = args[i];
            if (std.mem.eql(u8, a, "--prompt") and i + 1 < args.len) {
                prompt = args[i + 1];
                i += 2;
            } else if (std.mem.eql(u8, a, "--max-new") and i + 1 < args.len) {
                smoke_max_new = try std.fmt.parseInt(u32, args[i + 1], 10);
                i += 2;
            } else if (std.mem.eql(u8, a, "--budget") and i + 1 < args.len) {
                budget_layers = try std.fmt.parseInt(u32, args[i + 1], 10);
                i += 2;
            } else if (std.mem.eql(u8, a, "--budget-us") and i + 1 < args.len) {
                budget_us = try std.fmt.parseInt(u64, args[i + 1], 10);
                i += 2;
            } else if (std.mem.eql(u8, a, "--q4")) {
                q4 = true; i += 1;
            } else if (std.mem.eql(u8, a, "--q4k")) {
                q4k = true; i += 1;
            } else {
                i += 1;
            }
        }
        if (q4 and q4k) {
            std.debug.print("--q4 and --q4k are mutually exclusive\n", .{});
            return;
        }
        const budget: session_mod.Budget = if (budget_layers != null and budget_us != null)
            .{ .either = .{ .layers = budget_layers.?, .microseconds = budget_us.? } }
        else if (budget_us) |us|
            .{ .microseconds = us }
        else
            .{ .layers = budget_layers orelse 8 };
        const precision: gpu_model.Precision = if (q4k) .q4_k_matmul else if (q4) .q4_0_matmul else .bf16_matmul;
        try smoke_session.runSessionSmoke(allocator, args[2], prompt, smoke_max_new, budget, precision);
        return;
    }
    if (args.len >= 3 and std.mem.eql(u8, args[1], "--session-messages")) {
        // Multi-turn smoke for `Session.appendMessages`. Hardcoded
        // 3-turn fixture exercises the family's chat template through
        // the Session surface (matches how `valkyr --serve` will work).
        // Format: --session-messages <model> [--max-new N] [--q4|--q4k]
        var smoke_max_new: u32 = 32;
        var q4: bool = false;
        var q4k: bool = false;
        var i: usize = 3;
        while (i < args.len) {
            const a = args[i];
            if (std.mem.eql(u8, a, "--max-new") and i + 1 < args.len) {
                smoke_max_new = try std.fmt.parseInt(u32, args[i + 1], 10);
                i += 2;
            } else if (std.mem.eql(u8, a, "--q4")) {
                q4 = true; i += 1;
            } else if (std.mem.eql(u8, a, "--q4k")) {
                q4k = true; i += 1;
            } else {
                i += 1;
            }
        }
        if (q4 and q4k) {
            std.debug.print("--q4 and --q4k are mutually exclusive\n", .{});
            return;
        }
        const precision: gpu_model.Precision = if (q4k) .q4_k_matmul else if (q4) .q4_0_matmul else .bf16_matmul;
        try smoke_session.runSessionMessages(allocator, args[2], smoke_max_new, precision);
        return;
    }
    if (args.len >= 3 and (std.mem.eql(u8, args[1], "--runner-smoke") or
                            std.mem.eql(u8, args[1], "--runner-smoke-threaded")))
    {
        // End-to-end exercise of valkyr.inference.InferenceRunner.
        // --runner-smoke           → inline mode (host drives tickInline)
        // --runner-smoke-threaded  → spawn worker thread; submit/poll
        //                            from main thread.
        // Same 3-turn fixture as --session-messages; output should
        // be bit-identical across all three smokes.
        // Format: <flag> <model> [--max-new N] [--q4|--q4k]
        const threaded = std.mem.eql(u8, args[1], "--runner-smoke-threaded");
        var smoke_max_new: u32 = 64;
        var q4: bool = false;
        var q4k: bool = false;
        var i: usize = 3;
        while (i < args.len) {
            const a = args[i];
            if (std.mem.eql(u8, a, "--max-new") and i + 1 < args.len) {
                smoke_max_new = try std.fmt.parseInt(u32, args[i + 1], 10);
                i += 2;
            } else if (std.mem.eql(u8, a, "--q4")) {
                q4 = true; i += 1;
            } else if (std.mem.eql(u8, a, "--q4k")) {
                q4k = true; i += 1;
            } else {
                i += 1;
            }
        }
        if (q4 and q4k) {
            std.debug.print("--q4 and --q4k are mutually exclusive\n", .{});
            return;
        }
        const precision: gpu_model.Precision = if (q4k) .q4_k_matmul else if (q4) .q4_0_matmul else .bf16_matmul;
        try smoke_session.runRunnerSmoke(allocator, args[2], smoke_max_new, precision, threaded);
        return;
    }
    if (args.len >= 3 and std.mem.eql(u8, args[1], "--serve")) {
        // OpenAI-compatible HTTP server. Loads <model> once, listens
        // on --port (default 8080), accepts POST /v1/chat/completions
        // (streaming or not via "stream":true) and GET /v1/models.
        // Format:
        //   --serve <model> [--port N] [--bind ADDR] [--id PUBLIC_ID]
        //                   [--max-new N] [--q4|--q4k]
        // <model> is the HF id used to load (`Qwen/Qwen3-4B-Instruct-2507`).
        // --id  is the public model id surfaced via /v1/models and
        //        validated against the request's `model` field.
        //        Defaults to the HF id (clients can post that
        //        verbatim).
        var port: u16 = 8080;
        var bind_addr: []const u8 = "127.0.0.1";
        var public_id: ?[]const u8 = null;
        var max_new: u32 = 256;
        var q4: bool = false;
        var q4k: bool = false;
        var i: usize = 3;
        while (i < args.len) {
            const a = args[i];
            if (std.mem.eql(u8, a, "--port") and i + 1 < args.len) {
                port = try std.fmt.parseInt(u16, args[i + 1], 10);
                i += 2;
            } else if (std.mem.eql(u8, a, "--bind") and i + 1 < args.len) {
                bind_addr = args[i + 1];
                i += 2;
            } else if (std.mem.eql(u8, a, "--id") and i + 1 < args.len) {
                public_id = args[i + 1];
                i += 2;
            } else if (std.mem.eql(u8, a, "--max-new") and i + 1 < args.len) {
                max_new = try std.fmt.parseInt(u32, args[i + 1], 10);
                i += 2;
            } else if (std.mem.eql(u8, a, "--q4")) {
                q4 = true; i += 1;
            } else if (std.mem.eql(u8, a, "--q4k")) {
                q4k = true; i += 1;
            } else {
                i += 1;
            }
        }
        if (q4 and q4k) {
            std.debug.print("--q4 and --q4k are mutually exclusive\n", .{});
            return;
        }
        const precision: gpu_model.Precision = if (q4k) .q4_k_matmul else if (q4) .q4_0_matmul else .bf16_matmul;
        try commands_serve.runServe(allocator, args[2], public_id orelse args[2], bind_addr, port, max_new, precision);
        return;
    }
    if (args.len >= 3 and std.mem.eql(u8, args[1], "--chat")) {
        // Parse optional sampling flags + final user_msg. Format:
        //   --chat <dir> [--temp T] [--top-k K] [--top-p P] [--seed S]
        //                [--tq4v] [--q4|--q4k] [user_msg]
        // --tq4v switches the V cache to TurboQuant TQ4 (asymmetric:
        //   K stays full precision).
        // --q4   switches the per-layer projections to Q4_0 (4-bit weights,
        //   block-32 with fp16 scale per block). Mutually exclusive with
        //   bf16; lm_head + embeddings stay bf16.
        // --q4k  switches the per-layer projections to Q4_K_M (super-block
        //   of 256 with 8 sub-blocks, 6-bit per-sub-block scale + min,
        //   asymmetric dequant). Mutually exclusive with --q4. Quality
        //   ~30% lower MSE vs Q4_0 on Gaussian data; same on-device cost.
        // The user_msg, if given, is the last positional arg.
        var sp = cpu_forward.SampleParams{};
        var seed: u64 = @intCast(std.time.milliTimestamp());
        var user_msg: ?[]const u8 = null;
        var tq4v: bool = false;
        var q4: bool = false;
        var q4k: bool = false;
        var probe_path: ?[]const u8 = null;
        var prompts_tsv: ?[]const u8 = null;
        var probe_prefix: ?[]const u8 = null;
        var max_new: usize = 256;
        // ── --lora-ckpt fold-in. All four required together; the
        //    .lvkpt's cfg is checked against (targets, rank) at merge.
        var lora_path: ?[]const u8 = null;
        var lora_targets: u32 = 0;
        var lora_rank: u32 = 0;
        var lora_alpha: f32 = 0.0;
        var i: usize = 3;
        while (i < args.len) {
            const a = args[i];
            if (std.mem.eql(u8, a, "--temp") and i + 1 < args.len) {
                sp.temperature = try std.fmt.parseFloat(f32, args[i + 1]);
                i += 2;
            } else if (std.mem.eql(u8, a, "--top-k") and i + 1 < args.len) {
                sp.top_k = try std.fmt.parseInt(u32, args[i + 1], 10);
                i += 2;
            } else if (std.mem.eql(u8, a, "--top-p") and i + 1 < args.len) {
                sp.top_p = try std.fmt.parseFloat(f32, args[i + 1]);
                i += 2;
            } else if (std.mem.eql(u8, a, "--seed") and i + 1 < args.len) {
                seed = try std.fmt.parseInt(u64, args[i + 1], 10);
                i += 2;
            } else if (std.mem.eql(u8, a, "--tq4v")) {
                tq4v = true;
                i += 1;
            } else if (std.mem.eql(u8, a, "--q4")) {
                q4 = true;
                i += 1;
            } else if (std.mem.eql(u8, a, "--q4k")) {
                q4k = true;
                i += 1;
            } else if (std.mem.eql(u8, a, "--probe") and i + 1 < args.len) {
                probe_path = args[i + 1];
                i += 2;
            } else if (std.mem.eql(u8, a, "--prompts") and i + 1 < args.len) {
                prompts_tsv = args[i + 1];
                i += 2;
            } else if (std.mem.eql(u8, a, "--probe-prefix") and i + 1 < args.len) {
                probe_prefix = args[i + 1];
                i += 2;
            } else if (std.mem.eql(u8, a, "--max-new") and i + 1 < args.len) {
                max_new = try std.fmt.parseInt(usize, args[i + 1], 10);
                i += 2;
            } else if (std.mem.eql(u8, a, "--lora-ckpt") and i + 1 < args.len) {
                lora_path = args[i + 1];
                i += 2;
            } else if (std.mem.eql(u8, a, "--lora-targets") and i + 1 < args.len) {
                lora_targets = commands_finetune.parseLoraTargets(args[i + 1]) catch |err| {
                    std.debug.print("invalid --lora-targets spec '{s}': {s}\n", .{ args[i + 1], @errorName(err) });
                    return;
                };
                i += 2;
            } else if (std.mem.eql(u8, a, "--lora-rank") and i + 1 < args.len) {
                lora_rank = try std.fmt.parseInt(u32, args[i + 1], 10);
                i += 2;
            } else if (std.mem.eql(u8, a, "--lora-alpha") and i + 1 < args.len) {
                lora_alpha = try std.fmt.parseFloat(f32, args[i + 1]);
                i += 2;
            } else {
                user_msg = a;
                i += 1;
            }
        }
        // Validate batch-mode flags. --prompts requires --probe-prefix
        // (per-prompt traces need distinct paths) and forbids --probe
        // and positional user_msg (single-prompt flags).
        if (prompts_tsv != null) {
            if (probe_prefix == null) {
                std.debug.print("--prompts requires --probe-prefix <prefix>; per-prompt traces are written as <prefix><label>.jsonl\n", .{});
                return;
            }
            if (probe_path != null) {
                std.debug.print("--prompts is mutually exclusive with --probe (use --probe-prefix instead)\n", .{});
                return;
            }
            if (user_msg != null) {
                std.debug.print("--prompts is mutually exclusive with a positional prompt argument\n", .{});
                return;
            }
        } else if (probe_prefix != null) {
            std.debug.print("--probe-prefix is only meaningful with --prompts; specify both or neither\n", .{});
            return;
        }
        if (q4 and q4k) {
            std.debug.print("--q4 and --q4k are mutually exclusive\n", .{});
            return;
        }
        // --lora-ckpt requires the three companion flags so the merge layout
        // is unambiguous and so cfgShapeMatches gates a wrong-shape reload.
        var lora_ckpt: ?commands_chat.LoraCkpt = null;
        if (lora_path != null or lora_targets != 0 or lora_rank != 0) {
            if (lora_path == null or lora_targets == 0 or lora_rank == 0) {
                std.debug.print("--lora-ckpt requires --lora-targets, --lora-rank, and --lora-alpha (the values --lora-finetune was run with)\n", .{});
                return;
            }
            if (q4 or q4k) {
                std.debug.print("--lora-ckpt is not yet supported with --q4 / --q4k (Q4 requantisation after merge is a future chunk)\n", .{});
                return;
            }
            lora_ckpt = .{
                .path = lora_path.?,
                .targets = lora_targets,
                .rank = lora_rank,
                .alpha = lora_alpha,
            };
        }
        const precision: gpu_model.Precision = if (q4k) .q4_k_matmul else if (q4) .q4_0_matmul else .bf16_matmul;
        // Resolve `args[2]` from a possibly-HF-id form ("org/name") to an
        // on-disk snapshot path so chat works with `--chat
        // meta-llama/Llama-3.2-3B-Instruct ...` as smoothly as it does
        // with a literal path.
        const dir = try hf_cache.resolveModelArg(allocator, args[2]);
        defer allocator.free(dir);
        // Hybrid families need a different kernel set + per-layer
        // SSM state plumbing — dispatch to the dedicated path.
        const cfg_path = try std.fs.path.join(allocator, &.{ dir, "config.json" });
        defer allocator.free(cfg_path);
        const cfg = try config_mod.Config.loadFromFile(allocator, cfg_path);
        // Load batch prompts from the TSV (label \t prompt per line)
        // when --prompts was supplied. The slice and its contents
        // outlive the runChat call below; freed via the defer here.
        var batch_buf: ?[]u8 = null;
        defer if (batch_buf) |b| allocator.free(b);
        var batch_prompts: ?[]const commands_chat.PromptEntry = null;
        defer if (batch_prompts) |bp| allocator.free(bp);
        if (prompts_tsv) |path| {
            const loaded = try commands_chat.loadPromptsTsv(allocator, path);
            batch_buf = loaded.buf;
            batch_prompts = loaded.entries;
            std.debug.print("--prompts loaded {d} entries from {s}\n", .{ loaded.entries.len, path });
        }

        if (cfg.family.isHybrid()) {
            if (lora_ckpt != null) {
                std.debug.print("--lora-ckpt is not yet supported on Qwen3.5-style hybrid models (q_proj is widened by attn_output_gate)\n", .{});
                return;
            }
            try commands_chat.runChatQwen35(allocator, dir, user_msg, sp, seed, tq4v, precision, probe_path, batch_prompts, probe_prefix, max_new);
        } else {
            try commands_chat.runChat(allocator, dir, user_msg, sp, seed, tq4v, precision, probe_path, batch_prompts, probe_prefix, max_new, lora_ckpt);
        }
        return;
    }

    try smoke_misc.runVecAddSmoke(allocator);
    try smoke_misc.runSafeTensorsSmoke(allocator);
    try smoke_cpu.runMatmulSmoke(allocator);
    try smoke_cpu.runRopeIdentitySmoke(allocator);
    try smoke_cpu.runSoftmaxSmoke(allocator);
    try smoke_gpu_kernels.runGeluSmoke(allocator);
    try smoke_gpu_kernels.runTurboquantSmoke(allocator);
    try smoke_gpu_kernels.runQ4_0Smoke(allocator);
    try smoke_gpu_kernels.runQ4_KSmoke(allocator);
    try smoke_cpu.runTrainCpuSmoke(allocator);
    try smoke_cpu.runTrainCpuMultiLayerSmoke(allocator);
    try smoke_cpu.runRmsNormBackwardCpuSmoke(allocator);
    try smoke_cpu.runLoraMergeMathSmoke(allocator);
    try smoke_cpu.runFlashAttentionParitySmoke(allocator);
    try smoke_cpu.runFlashAttentionBackwardParitySmoke(allocator);
    try smoke_cpu.runFlashAttentionTq4VParitySmoke(allocator);
    try smoke_gpu_train.runFlashAttentionGpuSmoke(allocator);
    try smoke_gpu_train.runFaRunnerSmoke(allocator);
    try smoke_gpu_train.runFlashDecodingGpuSmoke(allocator);
    try smoke_gpu_train.runFaDecodeTq4VGpuSmoke(allocator);
    try smoke_gpu_train.runFaDecodeChatPathSmoke(allocator);
    try smoke_gpu_train.runFlashAttentionBackwardGpuSmoke(allocator);
    try smoke_gpu_train.runLayerNormBackwardCpuSmoke(allocator);
    try smoke_gpu_train.runGpuMatmulSmoke(allocator);
    try smoke_training.runGpuMlpForwardSmoke(allocator);
    try smoke_training.runGpuMlpBackwardSmoke(allocator);
    try smoke_training.runGpuMlpTrainSmoke(allocator);
    try smoke_training.runGpuMlpNTrainSmoke(allocator);
    try smoke_training.runTrainingRunnerSmoke(allocator);
    try smoke_training.runTrainingRunnerAttachedSmoke(allocator);
    try smoke_training.runTrainingRunnerBatchedSmoke(allocator);
    try smoke_training.runTrainingRunnerBatchedTrainSmoke(allocator);
    try smoke_training.runTrainingRunnerCoopSmoke(allocator);
    try smoke_training.runTrainingRunnerStagingSmoke(allocator);
    try smoke_training.runTrainingRunnerAdamSmoke(allocator);
    try smoke_training.runTrainingRunnerCrossEntropySmoke(allocator);
    try smoke_training.runTrainingRunnerDecaySmoke(allocator);
    try smoke_training.runTrainingRunnerNSmoke(allocator);
    try smoke_training.runTrainingRunnerNAttachedSmoke(allocator);
    try smoke_training.runWeightDecayCosineLrSmoke(allocator);
    try smoke_gpu_train.runGpuMatmulV2Smoke(allocator);
    try smoke_gpu_train.runGpuCceForwardSmoke(allocator);
    try smoke_gpu_train.runGpuCceBackwardDhSmoke(allocator);
    try smoke_gpu_train.runGpuCceBackwardDwSmoke(allocator);
    try smoke_gpu_train.runGpuLoraSmoke(allocator);
    try smoke_gpu_train.runGpuLoraRecorderSmoke(allocator);
    try smoke_gpu_train.runGpuTransformerLoraQSmoke(allocator);
    try smoke_gpu_train.runGpuTransformerLoraAllSmoke(allocator);
    try smoke_gpu_train.runEmbeddedAttachSmoke(allocator);
    try smoke_gpu_train.runEmbeddedRecorderSmoke(allocator);
    try smoke_gpu_train.runGpuMatmulQ4_0Smoke(allocator);
    try smoke_gpu_train.runGpuMatmulQ4_KSmoke(allocator);
    try smoke_gpu_train.runGpuRmsnormSmoke(allocator);
    try smoke_gpu_train.runGpuRmsnormBackwardSmoke(allocator);
    try smoke_gpu_train.runGpuLayerNormSmoke(allocator);
    try smoke_gpu_train.runGpuLayerNormBackwardSmoke(allocator);
    try smoke_gpu_train.runEmbeddingBackwardSmoke(allocator);
    try smoke_gpu_train.runSoftmaxBackwardSmoke(allocator);
    try smoke_gpu_train.runAttentionBackwardCpuSmoke(allocator);
    try smoke_gpu_train.runGpuAttentionBackwardSmoke(allocator);
    try smoke_gpu_train.runRopeBackwardSmoke(allocator);
    try smoke_decoder.runDecoderFineTuneCpuSmoke(allocator);
    try smoke_decoder.runDecoderStackFineTuneCpuSmoke(allocator);
    try smoke_decoder.runDecoderStackBackwardGpuParitySmoke(allocator);
    try smoke_decoder.runDecoderStackTrainGpuSmoke(allocator);
    try smoke_decoder.runDecoderStackTrainGpuRealShapeSmoke(allocator);
    try smoke_decoder.runRealModelLoadSmoke(allocator);
    try smoke_decoder.runRealModelForwardSmoke(allocator);
    try smoke_decoder.runRealModelDatasetSmoke(allocator);
    try smoke_decoder.runRealModelTrainStepSmoke(allocator);
    try smoke_decoder.runRealModelMultiStepSmoke(allocator);
    try smoke_decoder.runDecoderStackCheckpointSmoke(allocator);
    try smoke_decoder.runDecoderStackLoraCheckpointSmoke(allocator);
    try smoke_decoder.runRealModelSamplingSmoke(allocator);
    try smoke_decoder.runDecoderBackwardGpuParitySmoke(allocator);
    try smoke_decoder.runDecoderTrainGpuSmoke(allocator);
    try smoke_gpu_kernels.runGpuGegluSmoke(allocator);
    try smoke_gpu_kernels.runGpuRopeSmoke(allocator);
    try smoke_gpu_kernels.runGpuRopePartialSmoke(allocator);
    try smoke_gpu_kernels.runGpuSplitQGateSmoke(allocator);
    try smoke_gpu_kernels.runGpuSigmoidMulSmoke(allocator);
    try smoke_gpu_kernels.runSwiGluCpuSmoke(allocator);
    try smoke_gpu_kernels.runGpuSwiGluSmoke(allocator);
    try smoke_gpu_kernels.runGpuRopeBatchedSmoke(allocator);
    try smoke_gpu_kernels.runGpuL2normPerHeadSmoke(allocator);
    try smoke_gpu_kernels.runGpuConv1dUpdateSmoke(allocator);
    try smoke_gpu_kernels.runGpuRmsnormGatedSmoke(allocator);
    try smoke_gpu_kernels.runGpuGatedDeltaStepSmoke(allocator);
    try smoke_gpu_kernels.runGpuSoftmaxSmoke(allocator);
    try smoke_gpu_kernels.runGpuFwhtSmoke(allocator);
    try smoke_gpu_kernels.runGpuRhtPreSmoke(allocator);
    try smoke_gpu_kernels.runGpuRhtRoundTripSmoke(allocator);
    try smoke_gpu_kernels.runGpuRhtFusedRoundTripSmoke(allocator);
    try smoke_gpu_kernels.runGpuTq4PackSmoke(allocator);
    try smoke_gpu_kernels.runGpuTq4UnpackSmoke(allocator);
    try smoke_gpu_kernels.runGpuTq4RoundTripSmoke(allocator);
    try smoke_gpu_kernels.runGpuTq4PackToCacheSmoke(allocator);
}
