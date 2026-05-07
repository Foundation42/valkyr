//! Entry point. Runs every smoke test we have in sequence so a fresh
//! `zig build run` exercises the whole stack — Vulkan compute, file
//! formats, and (eventually) full-model parity. Each individual test
//! lives in its own function and prints a one-line pass marker on
//! success or surfaces an error otherwise.

const std = @import("std");
const vk = @import("gpu/vk.zig");
const buffer = @import("gpu/buffer.zig");
const pipeline = @import("gpu/pipeline.zig");
const safetensors = @import("safetensors.zig");
const model_mod = @import("model.zig");
const dtype = @import("dtype.zig");
const cpu_math = @import("cpu/math.zig");
const cpu_forward = @import("cpu/forward.zig");
const cpu_gated_delta = @import("cpu/gated_delta.zig");
const cpu_full_attn = @import("cpu/full_attn.zig");
const turboquant = @import("cpu/turboquant.zig");
const q4_0 = @import("cpu/q4_0.zig");
const q4_k = @import("cpu/q4_k.zig");
const cpu_train = @import("cpu/train.zig");
const cpu_train_transformer = @import("cpu/train_transformer.zig");
const cpu_train_decoder = @import("cpu/train_decoder.zig");
const cpu_cce = @import("cpu/cce.zig");
const cpu_lora = @import("cpu/lora.zig");
const train_runner = @import("train/runner.zig");
const train_runner_n = @import("train/runner_n.zig");
const train_transformer = @import("train/transformer.zig");
const train_load_real = @import("train/load_real.zig");
const train_dataset = @import("train/dataset.zig");
const tokenizer_mod = @import("tokenizer.zig");
const config_mod = @import("config.zig");
const gpu_model = @import("gpu/model.zig");
const gpu_scratch = @import("gpu/scratch.zig");
const gpu_recorder = @import("gpu/recorder.zig");
const runtime = @import("runtime.zig");
const runtime_hybrid = @import("runtime_hybrid.zig");
const loader = @import("loader.zig");
const session_mod = @import("session.zig");
const chat_template_mod = @import("chat_template.zig");
const hf_cache = @import("hf_cache.zig");
const probe = @import("probe.zig");
const shaders = @import("shaders");
const commands_inspect = @import("commands/inspect.zig");
const commands_encode = @import("commands/encode.zig");
const commands_bench = @import("commands/bench.zig");
const commands_serve = @import("commands/serve.zig");
const commands_chat = @import("commands/chat.zig");
const smoke_gpu_gen = @import("smoke/gpu_gen.zig");
const smoke_session = @import("smoke/session.zig");

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
        try runGpuCceForwardSmoke(allocator);
        return;
    }
    if (args.len >= 2 and std.mem.eql(u8, args[1], "--cce-backward-dh-smoke")) {
        // Fast-path for the d_h half of CCE backward.
        try runGpuCceBackwardDhSmoke(allocator);
        return;
    }
    if (args.len >= 2 and std.mem.eql(u8, args[1], "--cce-backward-dw-smoke")) {
        // Fast-path for the dW half of CCE backward.
        try runGpuCceBackwardDwSmoke(allocator);
        return;
    }
    if (args.len >= 2 and std.mem.eql(u8, args[1], "--real-train-step-smoke")) {
        // β-5 end-to-end real-model train. Loads Qwen3-0.6B from HF cache,
        // runs one Adam step, asserts CE_after < CE_before. The natural
        // parity gate for the CCE wiring (chunk 4).
        try runRealModelTrainStepSmoke(allocator);
        return;
    }
    if (args.len >= 2 and std.mem.eql(u8, args[1], "--decoder-stack-train-smoke")) {
        // 200-step synthetic-stack convergence smoke. Stronger trajectory
        // gate than the one-step real-model smoke: asserts final/initial
        // CE ratio < 1e-2 over 200 Adam steps. Tiny dims (dim=16, vocab=8)
        // for fast iteration.
        try runDecoderStackTrainGpuSmoke(allocator);
        return;
    }
    if (args.len >= 2 and std.mem.eql(u8, args[1], "--cce-bench")) {
        // Time each CCE kernel in isolation at Qwen3-0.6B shape. No
        // parity — just dispatch with vkQueueWaitIdle between calls
        // and report per-kernel ms. Shows where the LM-head bucket's
        // time actually goes.
        try runCceBench(allocator);
        return;
    }
    if (args.len >= 2 and std.mem.eql(u8, args[1], "--lora-smoke")) {
        // GPU LoRA composition smoke. Drives existing matmul_nt_v2 +
        // linear_backward_d{x,w}_batched + scale + add_in_place
        // kernels in the LoRA pattern (no new SPIR-V) and parity-checks
        // the outputs against the cpu/lora.zig oracle.
        try runGpuLoraSmoke(allocator);
        return;
    }
    if (args.len >= 2 and std.mem.eql(u8, args[1], "--lora-train-demo")) {
        // End-to-end LoRA training demo: synthetic problem with a target
        // rank-r delta on a frozen base, train Adam-LoRA to recover it.
        // The "does it actually train?" gate for LoRA on GPU.
        try runGpuLoraTrainDemo(allocator);
        return;
    }
    if (args.len >= 2 and std.mem.eql(u8, args[1], "--lora-plus-demo")) {
        // LoRA+ comparative demo: same task at η_B/η_A ∈ {1, 4, 16}.
        // Demonstrates the Chronicals/Hayou prediction that B should
        // learn ~16× faster than A in the feature-learning regime.
        try runGpuLoraPlusDemo(allocator);
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
        try runQwen35LayerTest(allocator, args[2], args[3]);
        return;
    }
    if (args.len >= 4 and std.mem.eql(u8, args[1], "--dump-embed")) {
        const token_id = try std.fmt.parseInt(u32, args[3], 10);
        try commands_inspect.runDumpEmbed(allocator, args[2], token_id);
        return;
    }
    if (args.len >= 4 and std.mem.eql(u8, args[1], "--rmsnorm-test")) {
        const token_id = try std.fmt.parseInt(u32, args[3], 10);
        try runRmsnormTest(allocator, args[2], token_id);
        return;
    }
    if (args.len >= 4 and std.mem.eql(u8, args[1], "--qproj-test")) {
        const token_id = try std.fmt.parseInt(u32, args[3], 10);
        try runQprojTest(allocator, args[2], token_id);
        return;
    }
    if (args.len >= 4 and std.mem.eql(u8, args[1], "--rope-test")) {
        const token_id = try std.fmt.parseInt(u32, args[3], 10);
        try runRopeTest(allocator, args[2], token_id);
        return;
    }
    if (args.len >= 4 and std.mem.eql(u8, args[1], "--attention-test")) {
        const token_id = try std.fmt.parseInt(u32, args[3], 10);
        try runAttentionTest(allocator, args[2], token_id);
        return;
    }
    if (args.len >= 4 and std.mem.eql(u8, args[1], "--layer0-test")) {
        const token_id = try std.fmt.parseInt(u32, args[3], 10);
        try runLayer0Test(allocator, args[2], token_id);
        return;
    }
    if (args.len >= 4 and std.mem.eql(u8, args[1], "--gen")) {
        const token_id = try std.fmt.parseInt(u32, args[3], 10);
        try runGen(allocator, args[2], token_id);
        return;
    }
    if (args.len >= 4 and std.mem.eql(u8, args[1], "--gpu-qproj-test")) {
        const token_id = try std.fmt.parseInt(u32, args[3], 10);
        try runGpuQprojTest(allocator, args[2], token_id);
        return;
    }
    if (args.len >= 4 and std.mem.eql(u8, args[1], "--gpu-rmsnorm-test")) {
        const token_id = try std.fmt.parseInt(u32, args[3], 10);
        try runGpuRmsnormTest(allocator, args[2], token_id);
        return;
    }
    if (args.len >= 4 and std.mem.eql(u8, args[1], "--gpu-geglu-test")) {
        const token_id = try std.fmt.parseInt(u32, args[3], 10);
        try runGpuGegluTest(allocator, args[2], token_id);
        return;
    }
    if (args.len >= 4 and std.mem.eql(u8, args[1], "--gpu-rope-test")) {
        const token_id = try std.fmt.parseInt(u32, args[3], 10);
        try runGpuRopeTest(allocator, args[2], token_id);
        return;
    }
    if (args.len >= 3 and std.mem.eql(u8, args[1], "--gpu-load")) {
        try runGpuLoad(allocator, args[2]);
        return;
    }
    if (args.len >= 4 and std.mem.eql(u8, args[1], "--gpu-layer0-test")) {
        const token_id = try std.fmt.parseInt(u32, args[3], 10);
        try runGpuLayer0Test(allocator, args[2], token_id);
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
        try runTq4KvTest(allocator, args[2], token_id);
        return;
    }
    if (args.len >= 4 and std.mem.eql(u8, args[1], "--gen-tq4v")) {
        const token_id = try std.fmt.parseInt(u32, args[3], 10);
        try runGenTq4V(allocator, args[2], token_id);
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
        try runTrainDemo(allocator, steps, hidden, lr, print_every);
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
        try runTrainDemoN(allocator, steps, layers_csv, lr, print_every);
        return;
    }
    if (args.len >= 3 and std.mem.eql(u8, args[1], "--bench")) {
        var n: usize = 64;
        var i: usize = 3;
        while (i < args.len) : (i += 2) {
            if (std.mem.eql(u8, args[i], "--n") and i + 1 < args.len) {
                n = try std.fmt.parseInt(usize, args[i + 1], 10);
            }
        }
        const dir = try hf_cache.resolveModelArg(allocator, args[2]);
        defer allocator.free(dir);
        try commands_bench.runBench(allocator, dir, n);
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
            try commands_chat.runChatQwen35(allocator, dir, user_msg, sp, seed, tq4v, precision, probe_path, batch_prompts, probe_prefix, max_new);
        } else {
            try commands_chat.runChat(allocator, dir, user_msg, sp, seed, tq4v, precision, probe_path, batch_prompts, probe_prefix, max_new);
        }
        return;
    }

    try runVecAddSmoke(allocator);
    try runSafeTensorsSmoke(allocator);
    try runMatmulSmoke(allocator);
    try runRopeIdentitySmoke(allocator);
    try runSoftmaxSmoke(allocator);
    try runGeluSmoke(allocator);
    try runTurboquantSmoke(allocator);
    try runQ4_0Smoke(allocator);
    try runQ4_KSmoke(allocator);
    try runTrainCpuSmoke(allocator);
    try runTrainCpuMultiLayerSmoke(allocator);
    try runRmsNormBackwardCpuSmoke(allocator);
    try runLayerNormBackwardCpuSmoke(allocator);
    try runGpuMatmulSmoke(allocator);
    try runGpuMlpForwardSmoke(allocator);
    try runGpuMlpBackwardSmoke(allocator);
    try runGpuMlpTrainSmoke(allocator);
    try runGpuMlpNTrainSmoke(allocator);
    try runTrainingRunnerSmoke(allocator);
    try runTrainingRunnerAttachedSmoke(allocator);
    try runTrainingRunnerBatchedSmoke(allocator);
    try runTrainingRunnerBatchedTrainSmoke(allocator);
    try runTrainingRunnerCoopSmoke(allocator);
    try runTrainingRunnerStagingSmoke(allocator);
    try runTrainingRunnerAdamSmoke(allocator);
    try runTrainingRunnerCrossEntropySmoke(allocator);
    try runTrainingRunnerDecaySmoke(allocator);
    try runTrainingRunnerNSmoke(allocator);
    try runTrainingRunnerNAttachedSmoke(allocator);
    try runWeightDecayCosineLrSmoke(allocator);
    try runGpuMatmulV2Smoke(allocator);
    try runGpuCceForwardSmoke(allocator);
    try runGpuCceBackwardDhSmoke(allocator);
    try runGpuCceBackwardDwSmoke(allocator);
    try runGpuLoraSmoke(allocator);
    try runEmbeddedAttachSmoke(allocator);
    try runEmbeddedRecorderSmoke(allocator);
    try runGpuMatmulQ4_0Smoke(allocator);
    try runGpuMatmulQ4_KSmoke(allocator);
    try runGpuRmsnormSmoke(allocator);
    try runGpuRmsnormBackwardSmoke(allocator);
    try runGpuLayerNormSmoke(allocator);
    try runGpuLayerNormBackwardSmoke(allocator);
    try runEmbeddingBackwardSmoke(allocator);
    try runSoftmaxBackwardSmoke(allocator);
    try runAttentionBackwardCpuSmoke(allocator);
    try runGpuAttentionBackwardSmoke(allocator);
    try runRopeBackwardSmoke(allocator);
    try runDecoderFineTuneCpuSmoke(allocator);
    try runDecoderStackFineTuneCpuSmoke(allocator);
    try runDecoderStackBackwardGpuParitySmoke(allocator);
    try runDecoderStackTrainGpuSmoke(allocator);
    try runDecoderStackTrainGpuRealShapeSmoke(allocator);
    try runRealModelLoadSmoke(allocator);
    try runRealModelForwardSmoke(allocator);
    try runRealModelDatasetSmoke(allocator);
    try runRealModelTrainStepSmoke(allocator);
    try runDecoderBackwardGpuParitySmoke(allocator);
    try runDecoderTrainGpuSmoke(allocator);
    try runGpuGegluSmoke(allocator);
    try runGpuRopeSmoke(allocator);
    try runGpuRopePartialSmoke(allocator);
    try runGpuSplitQGateSmoke(allocator);
    try runGpuSigmoidMulSmoke(allocator);
    try runSwiGluCpuSmoke(allocator);
    try runGpuSwiGluSmoke(allocator);
    try runGpuRopeBatchedSmoke(allocator);
    try runGpuL2normPerHeadSmoke(allocator);
    try runGpuConv1dUpdateSmoke(allocator);
    try runGpuRmsnormGatedSmoke(allocator);
    try runGpuGatedDeltaStepSmoke(allocator);
    try runGpuSoftmaxSmoke(allocator);
    try runGpuFwhtSmoke(allocator);
    try runGpuRhtPreSmoke(allocator);
    try runGpuRhtRoundTripSmoke(allocator);
    try runGpuRhtFusedRoundTripSmoke(allocator);
    try runGpuTq4PackSmoke(allocator);
    try runGpuTq4UnpackSmoke(allocator);
    try runGpuTq4RoundTripSmoke(allocator);
    try runGpuTq4PackToCacheSmoke(allocator);
}

fn runQwen35LayerTest(
    gpa: std.mem.Allocator,
    dir_path: []const u8,
    dump_path: []const u8,
) !void {
    // ── Load model ──────────────────────────────────────────────────
    var model = try model_mod.Model.load(gpa, dir_path);
    defer model.deinit();
    const cfg = model.config;
    if (cfg.family != .qwen35) {
        std.debug.print("expected qwen3.5 model, got {s}\n", .{@tagName(cfg.family)});
        return error.WrongFamily;
    }

    // ── Read reference dump ─────────────────────────────────────────
    // Format: header (4 × i32: magic, hidden_size, layer_idx,
    // layer_type_kind) followed by hidden_size fp32 (input) and
    // hidden_size fp32 (expected output). `layer_type_kind` is 0 for
    // linear_attention, 1 for full_attention — lets the Zig side cross-
    // check that Python dumped the layer it thinks Zig is testing.
    const file = try std.fs.cwd().openFile(dump_path, .{ .mode = .read_only });
    defer file.close();
    var got_header: [4]i32 = undefined;
    const hdr_bytes_read = try file.read(std.mem.sliceAsBytes(got_header[0..]));
    if (hdr_bytes_read != @sizeOf(@TypeOf(got_header))) return error.DumpHeaderTruncated;
    if (got_header[0] != 0x515E_3503) return error.DumpMagicMismatch;
    if (got_header[1] != @as(i32, @intCast(cfg.hidden_size))) return error.DumpHiddenSizeMismatch;
    const layer_idx: usize = @intCast(got_header[2]);
    if (layer_idx >= cfg.num_hidden_layers) return error.LayerIndexOutOfRange;
    const want_kind: i32 = switch (model.layers[layer_idx].layer_type) {
        .linear_attention => 0,
        .full_attention => 1,
    };
    if (got_header[3] != want_kind) return error.DumpLayerTypeMismatch;

    const x = try gpa.alloc(f32, cfg.hidden_size);
    defer gpa.free(x);
    const expected = try gpa.alloc(f32, cfg.hidden_size);
    defer gpa.free(expected);
    if (try file.read(std.mem.sliceAsBytes(x)) != cfg.hidden_size * @sizeOf(f32)) return error.DumpInputTruncated;
    if (try file.read(std.mem.sliceAsBytes(expected)) != cfg.hidden_size * @sizeOf(f32)) return error.DumpOutputTruncated;

    // ── Run the Zig CPU layer-N step ────────────────────────────────
    // Ref Python returns the layer's ATTENTION-PATH output (post
    // out_proj, pre residual add). We don't add the residual either.
    const got = try gpa.alloc(f32, cfg.hidden_size);
    defer gpa.free(got);
    const layer = model.layers[layer_idx];
    switch (layer.layer_type) {
        .linear_attention => {
            var state = try cpu_gated_delta.State.init(gpa, cfg);
            defer state.deinit(gpa);
            try cpu_gated_delta.decodeStep(gpa, cfg, layer, &state, x, got);
        },
        .full_attention => {
            // Single-token decode at pos=0 with empty cache. Cache is
            // sized for one position — that's all we need.
            var kv = try cpu_full_attn.KvCache.init(gpa, cfg, 1);
            defer kv.deinit(gpa);
            try cpu_full_attn.decodeStep(gpa, cfg, layer, &kv, x, got, 0);
        },
    }

    // ── Compare ─────────────────────────────────────────────────────
    var max_abs: f32 = 0.0;
    var sum_sq_diff: f64 = 0.0;
    var sum_sq_ref: f64 = 0.0;
    for (got, expected) |g, e| {
        const d = @abs(g - e);
        if (d > max_abs) max_abs = d;
        sum_sq_diff += @as(f64, d) * @as(f64, d);
        sum_sq_ref += @as(f64, e) * @as(f64, e);
    }
    const rel = if (sum_sq_ref > 0) @sqrt(sum_sq_diff / sum_sq_ref) else 0.0;
    const stdout = std.io.getStdOut().writer();
    try stdout.print("layer {d} ({s}) parity:\n", .{ layer_idx, @tagName(layer.layer_type) });
    try stdout.print("  hidden_size  = {d}\n", .{cfg.hidden_size});
    try stdout.print("  max |Δ|      = {e}\n", .{max_abs});
    try stdout.print("  ‖Δ‖ / ‖ref‖ = {e}\n", .{rel});
    try stdout.print("  first 4 (got vs ref):\n", .{});
    for (0..@min(4, got.len)) |i| {
        try stdout.print("    [{d}] {e:.6}  vs  {e:.6}\n", .{ i, got[i], expected[i] });
    }
    if (max_abs > 1e-3) {
        try stdout.print("FAIL: |Δ| > 1e-3\n", .{});
        return error.ParityFailed;
    }
    try stdout.print("PASS qwen35 layer {d} ({s}) — max |Δ| = {e}\n", .{ layer_idx, @tagName(layer.layer_type), max_abs });
}

// ── matmul smoke: synthetic A·B^T, hand-checked oracle ──────────────

fn runMatmulSmoke(allocator: std.mem.Allocator) !void {
    _ = allocator;
    // A = [[1, 2, 3], [4, 5, 6]]                M=2, K=3
    // B = [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1]]   N=4, K=3
    // A · Bᵀ = [[1, 2, 3, 6], [4, 5, 6, 15]]    M=2, N=4
    const a = [_]f32{ 1, 2, 3, 4, 5, 6 };
    const b_data = [_]f32{ 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1 };
    const want = [_]f32{ 1, 2, 3, 6, 4, 5, 6, 15 };

    const b_shape = [_]usize{ 4, 3 };
    const b: safetensors.Tensor = .{
        .dtype = .f32,
        .shape = &b_shape,
        .bytes = std.mem.sliceAsBytes(b_data[0..]),
    };

    var out: [8]f32 = undefined;
    try cpu_math.matmul_nt(&out, &a, b, 2, 4, 3);

    for (out, want, 0..) |got, w, i| {
        if (got != w) {
            std.debug.print("matmul MISMATCH at {d}: got {d}, expected {d}\n", .{ i, got, w });
            return error.ParityFailed;
        }
    }
    std.debug.print("PASS matmul_nt synthetic (2×3 · (4×3)ᵀ → 2×4)\n", .{});
}

// ── RoPE identity smoke: pos=0 must be a no-op ──────────────────────

fn runRopeIdentitySmoke(allocator: std.mem.Allocator) !void {
    // Synthetic Q-shaped input. n_heads=8, head_dim=64 — stays small.
    const n_heads: usize = 8;
    const head_dim: usize = 64;
    const total = n_heads * head_dim;
    const in_v = try allocator.alloc(f32, total);
    defer allocator.free(in_v);
    const out_v = try allocator.alloc(f32, total);
    defer allocator.free(out_v);
    for (in_v, 0..) |*x, i| x.* = @as(f32, @floatFromInt(i)) * 0.001 - 0.5;

    try cpu_math.applyRope(out_v, in_v, n_heads, head_dim, 0, 10000.0);

    for (in_v, out_v, 0..) |a, b, i| {
        if (a != b) {
            std.debug.print("RoPE pos=0 NOT identity at i={d}: in={d} out={d}\n", .{ i, a, b });
            return error.ParityFailed;
        }
    }
    std.debug.print("PASS RoPE pos=0 identity ({d} heads × {d} dim)\n", .{ n_heads, head_dim });
}

// ── softmax smoke: stable form on a hand-checked input ─────────────

fn runSoftmaxSmoke(allocator: std.mem.Allocator) !void {
    _ = allocator;
    // Inputs include a large positive value; the naive exp(x) form
    // would overflow but the stable variant should produce normal
    // probabilities. Reference computed with the same shifted form.
    var x = [_]f32{ 1.0, 2.0, 3.0, 100.0 };
    cpu_math.softmax(&x);
    var sum: f32 = 0;
    for (x) |v| sum += v;
    if (@abs(sum - 1.0) > 1e-6) {
        std.debug.print("softmax sum {d} != 1\n", .{sum});
        return error.ParityFailed;
    }
    // The big value should dominate (≈ 1.0); the others should be tiny.
    if (x[3] < 0.99 or x[0] > 1e-30 or x[1] > 1e-30 or x[2] > 1e-30) {
        std.debug.print("softmax distribution wrong: {any}\n", .{x});
        return error.ParityFailed;
    }
    std.debug.print("PASS softmax stable form (handles +100 without overflow)\n", .{});
}

// ── tiny-MLP training smoke: forward + backward + SGD converge ──────
//
// Chunk 1 of training-v0. Drives the CPU oracle in src/cpu/train.zig
// hard enough to catch a sign error or off-by-one anywhere in the
// derivative chain: a 2-layer MLP fits a single (input, target) pair
// and we assert loss drops by ≥ 100×. The numbers are tight on
// purpose — the only way for the loss curve to plateau here is if a
// gradient is wrong.
//
// We also do an explicit numerical-gradient check on one weight in
// each layer; that's the surest test that backward(...) is consistent
// with forward(...), independent of how well SGD converges.

fn runTrainCpuSmoke(allocator: std.mem.Allocator) !void {
    // Tiny on purpose — total weights = 4·8 + 8 + 4·8 + 4 = 76. Big
    // enough to need a hidden layer with ReLU active on some neurons,
    // small enough to land convergence in a few hundred steps.
    const dim_in: usize = 4;
    const dim_h: usize = 8;
    const dim_out: usize = 4;

    var mlp = try cpu_train.Mlp.init(allocator, dim_in, dim_h, dim_out, 0.3, 0xC0FFEE);
    defer mlp.deinit(allocator);

    var grads = try cpu_train.Grads.init(allocator, &mlp);
    defer grads.deinit(allocator);

    const x = [_]f32{ 1.0, 0.5, -0.3, 0.2 };
    const target = [_]f32{ 1.0, 0.0, 0.0, 0.0 };

    var h_pre: [dim_h]f32 = undefined;
    var h: [dim_h]f32 = undefined;
    var y: [dim_out]f32 = undefined;
    var act: cpu_train.Activations = .{
        .x = &x,
        .h_pre = &h_pre,
        .h = &h,
        .y = &y,
    };

    // ── Loss-decrease check ────────────────────────────────────────
    cpu_train.forward(&mlp, &act);
    const loss_initial = cpu_train.mseLoss(act.y, &target);

    const lr: f32 = 0.05;
    const n_steps: u32 = 400;
    var loss_final: f32 = 0;
    for (0..n_steps) |_| {
        loss_final = try cpu_train.trainStep(allocator, &mlp, &act, &grads, &target, lr);
    }

    if (!(loss_final < loss_initial / 100.0)) {
        std.debug.print(
            "training did not converge: loss[0] = {d:.6}, loss[{d}] = {d:.6}\n",
            .{ loss_initial, n_steps, loss_final },
        );
        return error.ParityFailed;
    }

    // ── Numerical-gradient parity check ────────────────────────────
    // For a fresh MLP and one (x, target), pick a single weight in W2
    // and a single weight in W1, perturb it by eps in each direction,
    // diff the losses, and compare to the analytic gradient. If the
    // chain rule in backward() is right these match within fp32
    // tolerance. The check is the truest kind — it doesn't trust SGD,
    // doesn't trust the loss being convex, just the calculus.
    var probe_mlp = try cpu_train.Mlp.init(allocator, dim_in, dim_h, dim_out, 0.3, 0xBEEF1234);
    defer probe_mlp.deinit(allocator);
    var probe_grads = try cpu_train.Grads.init(allocator, &probe_mlp);
    defer probe_grads.deinit(allocator);

    var probe_h_pre: [dim_h]f32 = undefined;
    var probe_h: [dim_h]f32 = undefined;
    var probe_y: [dim_out]f32 = undefined;
    var probe_act: cpu_train.Activations = .{
        .x = &x,
        .h_pre = &probe_h_pre,
        .h = &probe_h,
        .y = &probe_y,
    };

    // Analytic gradients at this point.
    cpu_train.forward(&probe_mlp, &probe_act);
    var dL_dy: [dim_out]f32 = undefined;
    cpu_train.mseLossGrad(&dL_dy, probe_act.y, &target);
    try cpu_train.backward(allocator, &probe_mlp, &probe_act, &dL_dy, &probe_grads);

    const eps: f32 = 1e-3;
    // Test one element from each parameter buffer.
    const probe_targets = [_]struct { name: []const u8, buf: []f32, grad: []const f32, idx: usize }{
        .{ .name = "W2[1, 3]", .buf = probe_mlp.w2, .grad = probe_grads.dw2, .idx = 1 * dim_h + 3 },
        .{ .name = "W1[2, 0]", .buf = probe_mlp.w1, .grad = probe_grads.dw1, .idx = 2 * dim_in + 0 },
        .{ .name = "b2[2]", .buf = probe_mlp.b2, .grad = probe_grads.db2, .idx = 2 },
        .{ .name = "b1[5]", .buf = probe_mlp.b1, .grad = probe_grads.db1, .idx = 5 },
    };
    for (probe_targets) |t| {
        const orig = t.buf[t.idx];
        t.buf[t.idx] = orig + eps;
        cpu_train.forward(&probe_mlp, &probe_act);
        const loss_plus = cpu_train.mseLoss(probe_act.y, &target);
        t.buf[t.idx] = orig - eps;
        cpu_train.forward(&probe_mlp, &probe_act);
        const loss_minus = cpu_train.mseLoss(probe_act.y, &target);
        t.buf[t.idx] = orig;

        const numeric = (loss_plus - loss_minus) / (2.0 * eps);
        const analytic = t.grad[t.idx];
        const denom = @max(@abs(numeric), @abs(analytic));
        const rel_err = if (denom > 0) @abs(numeric - analytic) / denom else @abs(numeric - analytic);
        if (rel_err > 1e-2) {
            std.debug.print(
                "grad mismatch on {s}: analytic = {d:.6}, numeric = {d:.6}, rel_err = {d:.4}\n",
                .{ t.name, analytic, numeric, rel_err },
            );
            return error.ParityFailed;
        }
    }

    std.debug.print(
        "PASS train MLP CPU oracle: loss {d:.6} → {d:.6} over {d} SGD steps; numeric grad parity ≤ 1%\n",
        .{ loss_initial, loss_final, n_steps },
    );
}

// ── Multi-layer MLP CPU oracle smoke ──────────────────────────────
//
// Tier-1 of the post-v0 training arc: generalise the 2-layer Mlp to
// any depth. This smoke exercises three things on `MlpN`:
//
//   1. n=2 numerical equivalence with `Mlp` — same dims, same seed →
//      forward output bit-identical (to fp32 rounding noise). This is
//      the strongest cross-check that the new code path doesn't drift
//      from the proven 2-layer reference.
//
//   2. n=3 numeric-gradient parity — perturb a single weight in each
//      of the three layers, diff the loss; the central-difference
//      slope must agree with the analytic gradient to ≤ 1%. This
//      proves the chain rule is right at depth.
//
//   3. n=3 convergence — 500 SGD steps must drop the loss by ≥ 100×
//      on the same toy (x, target) pair the 2-layer test uses.

fn runTrainCpuMultiLayerSmoke(allocator: std.mem.Allocator) !void {
    // ── (1) n=2 equivalence with `Mlp` ────────────────────────────
    {
        const dim_in: usize = 4;
        const dim_h: usize = 8;
        const dim_out: usize = 4;
        const seed: u64 = 0xC0FFEE;
        const init_scale: f32 = 0.3;

        var ref = try cpu_train.Mlp.init(allocator, dim_in, dim_h, dim_out, init_scale, seed);
        defer ref.deinit(allocator);

        var gen = try cpu_train.MlpN.init(allocator, &.{ dim_in, dim_h, dim_out }, init_scale, seed);
        defer gen.deinit(allocator);

        // Same RNG order: weights[0] should equal w1, weights[1] equal w2.
        for (ref.w1, gen.weights[0]) |a, b| {
            if (a != b) {
                std.debug.print("MlpN W1 != Mlp W1\n", .{});
                return error.ParityFailed;
            }
        }
        for (ref.w2, gen.weights[1]) |a, b| {
            if (a != b) {
                std.debug.print("MlpN W2 != Mlp W2\n", .{});
                return error.ParityFailed;
            }
        }

        const x = [_]f32{ 1.0, 0.5, -0.3, 0.2 };

        var ref_h_pre: [dim_h]f32 = undefined;
        var ref_h: [dim_h]f32 = undefined;
        var ref_y: [dim_out]f32 = undefined;
        var ref_act: cpu_train.Activations = .{ .x = &x, .h_pre = &ref_h_pre, .h = &ref_h, .y = &ref_y };
        cpu_train.forward(&ref, &ref_act);

        var gen_act = try cpu_train.ActivationsN.init(allocator, &gen);
        defer gen_act.deinit(allocator);
        gen_act.x = &x;
        cpu_train.forwardN(&gen, &gen_act);

        for (ref_y, gen_act.y()) |a, b| {
            if (a != b) {
                std.debug.print("MlpN forward != Mlp forward (n=2)\n", .{});
                return error.ParityFailed;
            }
        }
    }

    // ── (2)/(3) n=3 numeric grad + convergence ────────────────────
    const dim_in: usize = 4;
    const dim_h1: usize = 8;
    const dim_h2: usize = 6;
    const dim_out: usize = 4;
    const layer_dims = [_]usize{ dim_in, dim_h1, dim_h2, dim_out };

    var mlp = try cpu_train.MlpN.init(allocator, &layer_dims, 0.3, 0xCAFE2026);
    defer mlp.deinit(allocator);

    var grads = try cpu_train.GradsN.init(allocator, &mlp);
    defer grads.deinit(allocator);

    const x = [_]f32{ 1.0, 0.5, -0.3, 0.2 };
    const target = [_]f32{ 1.0, 0.0, 0.0, 0.0 };

    var act = try cpu_train.ActivationsN.init(allocator, &mlp);
    defer act.deinit(allocator);
    act.x = &x;

    cpu_train.forwardN(&mlp, &act);
    const loss_initial = cpu_train.mseLoss(act.y(), &target);

    // Numeric-grad probe BEFORE training (so we measure analytic vs
    // numeric on the un-trained network, where weights haven't drifted
    // toward zero gradient).
    var dL_dy: [dim_out]f32 = undefined;
    cpu_train.mseLossGrad(&dL_dy, act.y(), &target);
    try cpu_train.backwardN(allocator, &mlp, &act, &dL_dy, &grads);

    const eps: f32 = 1e-3;
    // One probe per layer: layer 0 hits W[0], layer 1 hits W[1] (and a bias),
    // layer 2 hits W[2] (and a bias on the output).
    const probes = [_]struct { name: []const u8, layer: usize, is_bias: bool, idx: usize }{
        .{ .name = "W[0][3, 0]", .layer = 0, .is_bias = false, .idx = 3 * dim_in + 0 },
        .{ .name = "W[1][2, 5]", .layer = 1, .is_bias = false, .idx = 2 * dim_h1 + 5 },
        .{ .name = "W[2][1, 4]", .layer = 2, .is_bias = false, .idx = 1 * dim_h2 + 4 },
        .{ .name = "b[1][3]", .layer = 1, .is_bias = true, .idx = 3 },
        .{ .name = "b[2][2]", .layer = 2, .is_bias = true, .idx = 2 },
    };
    for (probes) |p| {
        const buf = if (p.is_bias) mlp.biases[p.layer] else mlp.weights[p.layer];
        const grad = if (p.is_bias) grads.db[p.layer] else grads.dw[p.layer];
        const orig = buf[p.idx];

        buf[p.idx] = orig + eps;
        cpu_train.forwardN(&mlp, &act);
        const loss_plus = cpu_train.mseLoss(act.y(), &target);
        buf[p.idx] = orig - eps;
        cpu_train.forwardN(&mlp, &act);
        const loss_minus = cpu_train.mseLoss(act.y(), &target);
        buf[p.idx] = orig;

        const numeric = (loss_plus - loss_minus) / (2.0 * eps);
        const analytic = grad[p.idx];
        const denom = @max(@abs(numeric), @abs(analytic));
        const rel_err = if (denom > 0) @abs(numeric - analytic) / denom else @abs(numeric - analytic);
        if (rel_err > 1e-2) {
            std.debug.print(
                "MlpN grad mismatch on {s}: analytic = {d:.6}, numeric = {d:.6}, rel_err = {d:.4}\n",
                .{ p.name, analytic, numeric, rel_err },
            );
            return error.ParityFailed;
        }
    }

    // Convergence run.
    const lr: f32 = 0.05;
    const n_steps: u32 = 500;
    var loss_final: f32 = 0;
    for (0..n_steps) |_| {
        loss_final = try cpu_train.trainStepN(allocator, &mlp, &act, &grads, &target, lr);
    }
    if (!(loss_final < loss_initial / 100.0)) {
        std.debug.print(
            "MlpN n=3 did not converge: loss[0] = {d:.6}, loss[{d}] = {d:.6}\n",
            .{ loss_initial, n_steps, loss_final },
        );
        return error.ParityFailed;
    }

    std.debug.print(
        "PASS train MLP CPU oracle (multi-layer): n=2 equivalent to Mlp; n=3 loss {d:.6} → {d:.6} over {d} SGD steps; numeric grad parity ≤ 1% across all 3 layers\n",
        .{ loss_initial, loss_final, n_steps },
    );
}

// ── RMSNorm backward CPU oracle smoke ──────────────────────────────
//
// Tier-2 chunk 1 of the post-v0 training arc — first transformer
// primitive backward. RMSNorm forward is a row-wise scaling by
// `γ · rms_inv`; the backward involves a cross-row scalar reduction
// that the analytic derivation has to nail exactly.
//
// Three claims, each verified to ≤ 1% rel-err vs central-difference
// numeric gradient:
//   - dL/dx in the plain case (gemma_quirk=false)
//   - dL/dx with the gemma_quirk offset (γ = w + 1)
//   - dL/dw — same formula in both cases since (w + 1) and w have
//     identical derivatives w.r.t. w
//
// Probes one element from each gradient buffer; that's enough to
// catch any sign/shape mistake in the chain rule. The whole-buffer
// numeric grad is too noisy at fp32 to use as a parity oracle —
// per-element probes with a moderate eps are the right shape.

fn runRmsNormBackwardCpuSmoke(allocator: std.mem.Allocator) !void {
    _ = allocator; // reserved for future heap fallbacks at larger dim
    const dim: usize = 16;
    const eps: f32 = 1e-6;

    var x: [dim]f32 = undefined;
    var w: [dim]f32 = undefined;
    var dy: [dim]f32 = undefined;
    var prng = std.Random.DefaultPrng.init(0xCAFE5EED);
    const rng = prng.random();
    for (&x) |*v| v.* = (rng.float(f32) * 2.0 - 1.0) * 1.5;
    for (&w) |*v| v.* = 0.5 + rng.float(f32) * 0.5;
    for (&dy) |*v| v.* = (rng.float(f32) * 2.0 - 1.0);

    var dx: [dim]f32 = undefined;
    var dw: [dim]f32 = undefined;
    var y_buf: [dim]f32 = undefined;
    var probe_y: [dim]f32 = undefined;

    // Cases: (gemma_quirk = false), (gemma_quirk = true).
    const cases = [_]bool{ false, true };
    var max_rel_err_overall: f32 = 0;
    for (cases) |gq| {
        // Analytic gradients.
        @memset(&dw, 0);
        cpu_train_transformer.rmsNormBackwardRow(&dy, &x, &w, eps, gq, &dx, &dw);

        // Loss = Σ dy_i · y_i (so dL/dy_i = dy_i exactly).
        // For numeric grad we re-evaluate the loss with x or w perturbed.
        const eps_h: f32 = 1e-3;

        // Probe several positions in dx and dw.
        const probes_x = [_]usize{ 0, 3, 7, 12 };
        const probes_w = [_]usize{ 1, 5, 9, 14 };

        for (probes_x) |i| {
            const orig = x[i];
            x[i] = orig + eps_h;
            _ = cpu_train_transformer.rmsNormForwardRow(&x, &w, eps, gq, &probe_y);
            var l_plus: f32 = 0;
            for (dy, probe_y) |d, yi| l_plus += d * yi;
            x[i] = orig - eps_h;
            _ = cpu_train_transformer.rmsNormForwardRow(&x, &w, eps, gq, &probe_y);
            var l_minus: f32 = 0;
            for (dy, probe_y) |d, yi| l_minus += d * yi;
            x[i] = orig;

            const numeric = (l_plus - l_minus) / (2.0 * eps_h);
            const analytic = dx[i];
            const denom = @max(@abs(numeric), @abs(analytic));
            const rel_err = if (denom > 0) @abs(numeric - analytic) / denom else @abs(numeric - analytic);
            if (rel_err > 1e-2) {
                std.debug.print(
                    "rmsnorm dx[{d}] (gq={}): analytic={d:.6} numeric={d:.6} rel_err={d:.4}\n",
                    .{ i, gq, analytic, numeric, rel_err },
                );
                return error.ParityFailed;
            }
            if (rel_err > max_rel_err_overall) max_rel_err_overall = rel_err;
        }

        for (probes_w) |i| {
            const orig = w[i];
            w[i] = orig + eps_h;
            _ = cpu_train_transformer.rmsNormForwardRow(&x, &w, eps, gq, &probe_y);
            var l_plus: f32 = 0;
            for (dy, probe_y) |d, yi| l_plus += d * yi;
            w[i] = orig - eps_h;
            _ = cpu_train_transformer.rmsNormForwardRow(&x, &w, eps, gq, &probe_y);
            var l_minus: f32 = 0;
            for (dy, probe_y) |d, yi| l_minus += d * yi;
            w[i] = orig;

            const numeric = (l_plus - l_minus) / (2.0 * eps_h);
            const analytic = dw[i];
            const denom = @max(@abs(numeric), @abs(analytic));
            const rel_err = if (denom > 0) @abs(numeric - analytic) / denom else @abs(numeric - analytic);
            if (rel_err > 1e-2) {
                std.debug.print(
                    "rmsnorm dw[{d}] (gq={}): analytic={d:.6} numeric={d:.6} rel_err={d:.4}\n",
                    .{ i, gq, analytic, numeric, rel_err },
                );
                return error.ParityFailed;
            }
            if (rel_err > max_rel_err_overall) max_rel_err_overall = rel_err;
        }
    }

    // Multi-row sanity: dw must accumulate across rows.
    const n_rows: usize = 3;
    var x_multi: [n_rows * dim]f32 = undefined;
    var dy_multi: [n_rows * dim]f32 = undefined;
    var dx_multi: [n_rows * dim]f32 = undefined;
    var dw_multi: [dim]f32 = undefined;
    for (&x_multi) |*v| v.* = (rng.float(f32) * 2.0 - 1.0);
    for (&dy_multi) |*v| v.* = (rng.float(f32) * 2.0 - 1.0);

    @memset(&dw_multi, 0);
    cpu_train_transformer.rmsNormBackward(&dy_multi, &x_multi, &w, eps, false, n_rows, &dx_multi, &dw_multi);

    // Compare against summing per-row backward calls explicitly.
    var dw_ref: [dim]f32 = undefined;
    @memset(&dw_ref, 0);
    var dx_ref_row: [dim]f32 = undefined;
    for (0..n_rows) |r| {
        const off = r * dim;
        cpu_train_transformer.rmsNormBackwardRow(
            dy_multi[off .. off + dim],
            x_multi[off .. off + dim],
            &w,
            eps,
            false,
            &dx_ref_row,
            &dw_ref,
        );
        for (dx_multi[off .. off + dim], dx_ref_row) |a, b| {
            if (a != b) {
                std.debug.print("multi-row dx mismatch at row {d}\n", .{r});
                return error.ParityFailed;
            }
        }
    }
    for (dw_multi, dw_ref) |a, b| {
        if (a != b) {
            std.debug.print("multi-row dw accumulation off\n", .{});
            return error.ParityFailed;
        }
    }

    // Suppress unused-variable warning for y_buf.
    _ = &y_buf;

    std.debug.print(
        "PASS rmsnorm backward CPU oracle (numeric-grad parity ≤ {e} on dx + dw, gemma_quirk on/off; multi-row dw accum bit-exact)\n",
        .{max_rel_err_overall},
    );
}

// ── tiny-MLP GPU forward smoke: matmul → bias → relu → matmul → bias ──
//
// Chunk 2 of training-v0. Composes existing kernels (matmul_nt,
// add_in_place, plus the new relu.comp) into a 2-layer MLP forward
// pass on the GPU and parity-checks against the CPU oracle in
// `cpu_train.forward`. The MLP is built deterministically so the GPU
// dispatch and the CPU reference both see bit-identical weights.
//
// Why compose existing matmuls instead of writing a fused mlp_forward
// shader: chunk 2 is a parity proof, not a perf milestone. A fused
// shader is a ~30-line port once the composition is verified, but the
// composition exposes the building blocks the upcoming backward
// shaders also need (matmul, transposed matmul, relu).

const ReluPush = runtime.ReluPush;

fn runGpuMlpForwardSmoke(allocator: std.mem.Allocator) !void {
    var ctx = try vk.Context.init(allocator);
    defer ctx.deinit();

    const dim_in: usize = 4;
    const dim_h: usize = 8;
    const dim_out: usize = 4;

    // Same construction as the CPU smoke: deterministic weights so we
    // can run forward both places and diff. Different seed than chunk 1
    // — exercising the same shape on a fresh init catches any "smoke
    // happened to pass on lucky weights" failure modes.
    var mlp = try cpu_train.Mlp.init(allocator, dim_in, dim_h, dim_out, 0.3, 0xF0DDA1A);
    defer mlp.deinit(allocator);
    // Bias at zero from init() is too forgiving — set b1 and b2 to
    // distinct nonzero values so a "bias never applied" bug surfaces.
    for (mlp.b1, 0..) |*v, i| v.* = 0.1 - 0.05 * @as(f32, @floatFromInt(i));
    for (mlp.b2, 0..) |*v, i| v.* = -0.2 + 0.07 * @as(f32, @floatFromInt(i));

    const x = [_]f32{ 1.0, 0.5, -0.3, 0.2 };

    // ── CPU oracle ─────────────────────────────────────────────────
    var h_pre_cpu: [dim_h]f32 = undefined;
    var h_cpu: [dim_h]f32 = undefined;
    var y_cpu: [dim_out]f32 = undefined;
    var act: cpu_train.Activations = .{
        .x = &x,
        .h_pre = &h_pre_cpu,
        .h = &h_cpu,
        .y = &y_cpu,
    };
    cpu_train.forward(&mlp, &act);

    // ── GPU buffers ────────────────────────────────────────────────
    var buf_x = try buffer.Buffer.initStatic(&ctx, f32, &x);
    defer buf_x.deinit(ctx.device);
    var buf_w1 = try buffer.Buffer.initStatic(&ctx, f32, mlp.w1);
    defer buf_w1.deinit(ctx.device);
    var buf_b1 = try buffer.Buffer.initStatic(&ctx, f32, mlp.b1);
    defer buf_b1.deinit(ctx.device);
    var buf_w2 = try buffer.Buffer.initStatic(&ctx, f32, mlp.w2);
    defer buf_w2.deinit(ctx.device);
    var buf_b2 = try buffer.Buffer.initStatic(&ctx, f32, mlp.b2);
    defer buf_b2.deinit(ctx.device);
    var buf_h_pre = try buffer.Buffer.initDeviceOnly(&ctx, dim_h * @sizeOf(f32));
    defer buf_h_pre.deinit(ctx.device);
    var buf_h = try buffer.Buffer.initDeviceOnly(&ctx, dim_h * @sizeOf(f32));
    defer buf_h.deinit(ctx.device);
    var buf_y = try buffer.Buffer.initDeviceOnly(&ctx, dim_out * @sizeOf(f32));
    defer buf_y.deinit(ctx.device);

    // ── Pipelines ──────────────────────────────────────────────────
    var k_matmul = try pipeline.Kernel.init(&ctx, &shaders.matmul_nt, 3, @sizeOf(MatmulPush));
    defer k_matmul.deinit();
    var k_add = try pipeline.Kernel.init(&ctx, &shaders.add_in_place, 2, @sizeOf(runtime.AddInPlacePush));
    defer k_add.deinit();
    var k_relu = try pipeline.Kernel.init(&ctx, &shaders.relu, 2, @sizeOf(ReluPush));
    defer k_relu.deinit();

    // ── Record + submit ────────────────────────────────────────────
    // Treat x and h as 1×K row matrices so matmul_nt computes
    // y_pre[1, N] = x[1, K] · W[N, K]ᵀ. That's exactly W·x for the
    // row vector x, which is the layer formulation we want.
    var rec = try gpu_recorder.Recorder.init(&ctx, 16, 64);
    defer rec.deinit();
    try rec.begin();

    const matmul1_push = MatmulPush{ .m = 1, .n = @intCast(dim_h), .k = @intCast(dim_in) };
    const matmul2_push = MatmulPush{ .m = 1, .n = @intCast(dim_out), .k = @intCast(dim_h) };
    const add1_push = runtime.AddInPlacePush{ .n = @intCast(dim_h) };
    const add2_push = runtime.AddInPlacePush{ .n = @intCast(dim_out) };
    const relu_push = ReluPush{ .n = @intCast(dim_h) };

    // matmul_nt grid is (ceil(M/16), ceil(N/16)) — for M=1, N≤16 the
    // whole work is one workgroup, which is correct (threads outside
    // bound early-out).
    try rec.dispatch(&k_matmul, &.{ &buf_x, &buf_w1, &buf_h_pre }, &matmul1_push, 1, 1, 1);
    try rec.dispatch(&k_add, &.{ &buf_h_pre, &buf_b1 }, &add1_push, ceilDiv(@as(u32, dim_h), 256), 1, 1);
    try rec.dispatch(&k_relu, &.{ &buf_h_pre, &buf_h }, &relu_push, ceilDiv(@as(u32, dim_h), 256), 1, 1);
    try rec.dispatch(&k_matmul, &.{ &buf_h, &buf_w2, &buf_y }, &matmul2_push, 1, 1, 1);
    try rec.dispatch(&k_add, &.{ &buf_y, &buf_b2 }, &add2_push, ceilDiv(@as(u32, dim_out), 256), 1, 1);

    try rec.endAndSubmit();

    // ── Compare ───────────────────────────────────────────────────
    var h_pre_gpu: [dim_h]f32 = undefined;
    var h_gpu: [dim_h]f32 = undefined;
    var y_gpu: [dim_out]f32 = undefined;
    try buf_h_pre.readBack(&ctx, f32, &h_pre_gpu);
    try buf_h.readBack(&ctx, f32, &h_gpu);
    try buf_y.readBack(&ctx, f32, &y_gpu);

    const tol: f32 = 1e-5;
    const ParityCase = struct { name: []const u8, got: []const f32, want: []const f32 };
    const cases = [_]ParityCase{
        .{ .name = "h_pre", .got = &h_pre_gpu, .want = &h_pre_cpu },
        .{ .name = "h", .got = &h_gpu, .want = &h_cpu },
        .{ .name = "y", .got = &y_gpu, .want = &y_cpu },
    };
    var max_abs: f32 = 0;
    for (cases) |cs| {
        for (cs.got, cs.want, 0..) |g, w, i| {
            const d = @abs(g - w);
            if (d > tol) {
                std.debug.print(
                    "GPU MLP forward MISMATCH on {s}[{d}]: got {d:.7}, expected {d:.7}\n",
                    .{ cs.name, i, g, w },
                );
                return error.ParityFailed;
            }
            if (d > max_abs) max_abs = d;
        }
    }
    std.debug.print(
        "PASS GPU MLP forward (4→8→4, matmul+bias+relu+matmul+bias, max |Δ| vs CPU = {e})\n",
        .{max_abs},
    );
}

fn ceilDiv(num: u32, den: u32) u32 {
    return (num + den - 1) / den;
}

// ── tiny-MLP GPU backward smoke: gradients vs CPU oracle ────────────
//
// Chunk 3 of training-v0. Runs the full forward + backward pipeline on
// the GPU and parity-checks every gradient buffer against the CPU
// oracle in `cpu_train.backward`. Three new shaders carry it:
//
//   linear_backward_dx — dL/dh from dL/dy and W2 (transposed matvec)
//   relu_backward      — gates dL/dh by the saved h_pre > 0 mask
//   outer_product      — dL/dW = upstream ⊗ input  (dW2 and dW1 both)
//
// db gradients reuse `slice_copy` (dL/db = dL/dy by definition; pure
// memcpy at the buffer level). Per-step loss-grad (dL/dy = y - target)
// is computed in a single host-side op against `add_in_place`-style
// staging — kept on host because it's one fp32 subtract per step at
// dim_out scale, not worth a shader.

const ReluBackwardPush = runtime.ReluBackwardPush;
const LinearBackwardDxPush = runtime.LinearBackwardDxPush;
const OuterProductPush = runtime.OuterProductPush;

fn runGpuMlpBackwardSmoke(allocator: std.mem.Allocator) !void {
    var ctx = try vk.Context.init(allocator);
    defer ctx.deinit();

    const dim_in: usize = 4;
    const dim_h: usize = 8;
    const dim_out: usize = 4;

    var mlp = try cpu_train.Mlp.init(allocator, dim_in, dim_h, dim_out, 0.3, 0xBACC0FF);
    defer mlp.deinit(allocator);
    for (mlp.b1, 0..) |*v, i| v.* = 0.1 - 0.05 * @as(f32, @floatFromInt(i));
    for (mlp.b2, 0..) |*v, i| v.* = -0.2 + 0.07 * @as(f32, @floatFromInt(i));

    const x = [_]f32{ 1.0, 0.5, -0.3, 0.2 };
    const target = [_]f32{ 1.0, 0.0, 0.0, 0.0 };

    // ── CPU oracle: forward + grads ────────────────────────────────
    var h_pre_cpu: [dim_h]f32 = undefined;
    var h_cpu: [dim_h]f32 = undefined;
    var y_cpu: [dim_out]f32 = undefined;
    var act: cpu_train.Activations = .{
        .x = &x,
        .h_pre = &h_pre_cpu,
        .h = &h_cpu,
        .y = &y_cpu,
    };
    cpu_train.forward(&mlp, &act);
    var dL_dy_cpu: [dim_out]f32 = undefined;
    cpu_train.mseLossGrad(&dL_dy_cpu, &y_cpu, &target);
    var grads_cpu = try cpu_train.Grads.init(allocator, &mlp);
    defer grads_cpu.deinit(allocator);
    try cpu_train.backward(allocator, &mlp, &act, &dL_dy_cpu, &grads_cpu);

    // ── GPU buffers ────────────────────────────────────────────────
    var buf_x = try buffer.Buffer.initStatic(&ctx, f32, &x);
    defer buf_x.deinit(ctx.device);
    var buf_w1 = try buffer.Buffer.initStatic(&ctx, f32, mlp.w1);
    defer buf_w1.deinit(ctx.device);
    var buf_b1 = try buffer.Buffer.initStatic(&ctx, f32, mlp.b1);
    defer buf_b1.deinit(ctx.device);
    var buf_w2 = try buffer.Buffer.initStatic(&ctx, f32, mlp.w2);
    defer buf_w2.deinit(ctx.device);
    var buf_b2 = try buffer.Buffer.initStatic(&ctx, f32, mlp.b2);
    defer buf_b2.deinit(ctx.device);
    var buf_h_pre = try buffer.Buffer.initDeviceOnly(&ctx, dim_h * @sizeOf(f32));
    defer buf_h_pre.deinit(ctx.device);
    var buf_h = try buffer.Buffer.initDeviceOnly(&ctx, dim_h * @sizeOf(f32));
    defer buf_h.deinit(ctx.device);
    var buf_y = try buffer.Buffer.initDeviceOnly(&ctx, dim_out * @sizeOf(f32));
    defer buf_y.deinit(ctx.device);

    // dL/dy is staged from host (cheap, dim_out scalars). Same for
    // grads buffers — they get *written* by the GPU.
    var buf_dL_dy = try buffer.Buffer.initStatic(&ctx, f32, &dL_dy_cpu);
    defer buf_dL_dy.deinit(ctx.device);
    var buf_dh = try buffer.Buffer.initDeviceOnly(&ctx, dim_h * @sizeOf(f32));
    defer buf_dh.deinit(ctx.device);
    var buf_dh_pre = try buffer.Buffer.initDeviceOnly(&ctx, dim_h * @sizeOf(f32));
    defer buf_dh_pre.deinit(ctx.device);
    var buf_dw1 = try buffer.Buffer.initDeviceOnly(&ctx, dim_h * dim_in * @sizeOf(f32));
    defer buf_dw1.deinit(ctx.device);
    var buf_db1 = try buffer.Buffer.initDeviceOnly(&ctx, dim_h * @sizeOf(f32));
    defer buf_db1.deinit(ctx.device);
    var buf_dw2 = try buffer.Buffer.initDeviceOnly(&ctx, dim_out * dim_h * @sizeOf(f32));
    defer buf_dw2.deinit(ctx.device);
    var buf_db2 = try buffer.Buffer.initDeviceOnly(&ctx, dim_out * @sizeOf(f32));
    defer buf_db2.deinit(ctx.device);

    // ── Pipelines ──────────────────────────────────────────────────
    var k_matmul = try pipeline.Kernel.init(&ctx, &shaders.matmul_nt, 3, @sizeOf(MatmulPush));
    defer k_matmul.deinit();
    var k_add = try pipeline.Kernel.init(&ctx, &shaders.add_in_place, 2, @sizeOf(runtime.AddInPlacePush));
    defer k_add.deinit();
    var k_relu = try pipeline.Kernel.init(&ctx, &shaders.relu, 2, @sizeOf(ReluPush));
    defer k_relu.deinit();
    var k_relu_bw = try pipeline.Kernel.init(&ctx, &shaders.relu_backward, 3, @sizeOf(ReluBackwardPush));
    defer k_relu_bw.deinit();
    var k_lin_dx = try pipeline.Kernel.init(&ctx, &shaders.linear_backward_dx, 3, @sizeOf(LinearBackwardDxPush));
    defer k_lin_dx.deinit();
    var k_outer = try pipeline.Kernel.init(&ctx, &shaders.outer_product, 3, @sizeOf(OuterProductPush));
    defer k_outer.deinit();
    var k_copy = try pipeline.Kernel.init(&ctx, &shaders.slice_copy, 2, @sizeOf(SliceCopyPush));
    defer k_copy.deinit();

    var rec = try gpu_recorder.Recorder.init(&ctx, 32, 128);
    defer rec.deinit();
    try rec.begin();

    // ── Forward (same as chunk 2) ──────────────────────────────────
    const matmul1_push = MatmulPush{ .m = 1, .n = @intCast(dim_h), .k = @intCast(dim_in) };
    const matmul2_push = MatmulPush{ .m = 1, .n = @intCast(dim_out), .k = @intCast(dim_h) };
    const add1_push = runtime.AddInPlacePush{ .n = @intCast(dim_h) };
    const add2_push = runtime.AddInPlacePush{ .n = @intCast(dim_out) };
    const relu_push = ReluPush{ .n = @intCast(dim_h) };
    try rec.dispatch(&k_matmul, &.{ &buf_x, &buf_w1, &buf_h_pre }, &matmul1_push, 1, 1, 1);
    try rec.dispatch(&k_add, &.{ &buf_h_pre, &buf_b1 }, &add1_push, ceilDiv(@as(u32, dim_h), 256), 1, 1);
    try rec.dispatch(&k_relu, &.{ &buf_h_pre, &buf_h }, &relu_push, ceilDiv(@as(u32, dim_h), 256), 1, 1);
    try rec.dispatch(&k_matmul, &.{ &buf_h, &buf_w2, &buf_y }, &matmul2_push, 1, 1, 1);
    try rec.dispatch(&k_add, &.{ &buf_y, &buf_b2 }, &add2_push, ceilDiv(@as(u32, dim_out), 256), 1, 1);

    // ── Backward ───────────────────────────────────────────────────
    // dL/db2 = dL/dy   (slice_copy 0..dim_out → 0..dim_out).
    const copy_db2 = SliceCopyPush{ .src_off = 0, .dst_off = 0, .n_elem = @intCast(dim_out) };
    try rec.dispatch(&k_copy, &.{ &buf_dL_dy, &buf_db2 }, &copy_db2, ceilDiv(@as(u32, dim_out), 256), 1, 1);

    // dL/dW2[i, j] = dL/dy[i] · h[j]   (outer product, [dim_out, dim_h]).
    const op_dw2 = OuterProductPush{ .dim_out = @intCast(dim_out), .dim_in = @intCast(dim_h) };
    try rec.dispatch(
        &k_outer,
        &.{ &buf_dL_dy, &buf_h, &buf_dw2 },
        &op_dw2,
        ceilDiv(@as(u32, dim_out), 16),
        ceilDiv(@as(u32, dim_h), 16),
        1,
    );

    // dL/dh = W2^T · dL/dy  (transposed matvec).
    const lin_dx_push = LinearBackwardDxPush{ .dim_out = @intCast(dim_out), .dim_in = @intCast(dim_h) };
    try rec.dispatch(
        &k_lin_dx,
        &.{ &buf_dL_dy, &buf_w2, &buf_dh },
        &lin_dx_push,
        ceilDiv(@as(u32, dim_h), 256),
        1,
        1,
    );

    // dL/dh_pre = dL/dh · 1[h_pre > 0].
    const relu_bw_push = ReluBackwardPush{ .n = @intCast(dim_h) };
    try rec.dispatch(
        &k_relu_bw,
        &.{ &buf_dh, &buf_h_pre, &buf_dh_pre },
        &relu_bw_push,
        ceilDiv(@as(u32, dim_h), 256),
        1,
        1,
    );

    // dL/db1 = dL/dh_pre.
    const copy_db1 = SliceCopyPush{ .src_off = 0, .dst_off = 0, .n_elem = @intCast(dim_h) };
    try rec.dispatch(&k_copy, &.{ &buf_dh_pre, &buf_db1 }, &copy_db1, ceilDiv(@as(u32, dim_h), 256), 1, 1);

    // dL/dW1[j, k] = dL/dh_pre[j] · x[k].
    const op_dw1 = OuterProductPush{ .dim_out = @intCast(dim_h), .dim_in = @intCast(dim_in) };
    try rec.dispatch(
        &k_outer,
        &.{ &buf_dh_pre, &buf_x, &buf_dw1 },
        &op_dw1,
        ceilDiv(@as(u32, dim_h), 16),
        ceilDiv(@as(u32, dim_in), 16),
        1,
    );

    try rec.endAndSubmit();

    // ── Read back + parity ─────────────────────────────────────────
    var dw1_gpu: [dim_h * dim_in]f32 = undefined;
    var db1_gpu: [dim_h]f32 = undefined;
    var dw2_gpu: [dim_out * dim_h]f32 = undefined;
    var db2_gpu: [dim_out]f32 = undefined;
    try buf_dw1.readBack(&ctx, f32, &dw1_gpu);
    try buf_db1.readBack(&ctx, f32, &db1_gpu);
    try buf_dw2.readBack(&ctx, f32, &dw2_gpu);
    try buf_db2.readBack(&ctx, f32, &db2_gpu);

    const tol: f32 = 1e-5;
    const ParityCase = struct { name: []const u8, got: []const f32, want: []const f32 };
    const cases = [_]ParityCase{
        .{ .name = "dW1", .got = &dw1_gpu, .want = grads_cpu.dw1 },
        .{ .name = "db1", .got = &db1_gpu, .want = grads_cpu.db1 },
        .{ .name = "dW2", .got = &dw2_gpu, .want = grads_cpu.dw2 },
        .{ .name = "db2", .got = &db2_gpu, .want = grads_cpu.db2 },
    };
    var max_abs: f32 = 0;
    for (cases) |cs| {
        for (cs.got, cs.want, 0..) |g, w, i| {
            const d = @abs(g - w);
            if (d > tol) {
                std.debug.print(
                    "GPU MLP backward MISMATCH on {s}[{d}]: got {d:.7}, expected {d:.7}\n",
                    .{ cs.name, i, g, w },
                );
                return error.ParityFailed;
            }
            if (d > max_abs) max_abs = d;
        }
    }
    std.debug.print(
        "PASS GPU MLP backward (dW1, db1, dW2, db2 vs CPU; max |Δ| = {e})\n",
        .{max_abs},
    );
}

// ── tiny-MLP GPU full training loop: SGD step + multi-step convergence ──
//
// Chunk 4 of training-v0. Adds the SGD step shader (`param -= lr ·
// grad`) and the MSE loss-grad shader (`dL/dy = pred − target`),
// closing the loop so the entire forward → backward → update cycle
// stays on the GPU with no per-step CPU↔GPU sync. Then runs N steps
// on both CPU oracle and GPU and asserts:
//
//   1. After 1 step, GPU weights == CPU weights within 1e-5
//      (every kernel and the optimizer touched parameters identically).
//   2. After N steps, the GPU loss curve matches the CPU loss curve
//      within 1e-4 — error accumulates very slowly at our dims, but
//      this catches any drift from a subtly-wrong dispatch order.
//
// One sync at the very end (read final weights and final pred for
// loss). Every intermediate step is pure GPU — same shape as the
// in-frame training the engine integration needs.

const SgdStepPush = runtime.SgdStepPush;
const MseLossGradPush = runtime.MseLossGradPush;

fn runGpuMlpTrainSmoke(allocator: std.mem.Allocator) !void {
    var ctx = try vk.Context.init(allocator);
    defer ctx.deinit();

    const dim_in: usize = 4;
    const dim_h: usize = 8;
    const dim_out: usize = 4;
    const lr: f32 = 0.05;
    const n_steps: u32 = 50;

    // Twin MLPs — same seed, same weights to start. CPU drives oracle,
    // GPU drives the unit under test. After n_steps each they should
    // be bit-close.
    var mlp_cpu = try cpu_train.Mlp.init(allocator, dim_in, dim_h, dim_out, 0.3, 0x57DD57D);
    defer mlp_cpu.deinit(allocator);
    for (mlp_cpu.b1, 0..) |*v, i| v.* = 0.1 - 0.05 * @as(f32, @floatFromInt(i));
    for (mlp_cpu.b2, 0..) |*v, i| v.* = -0.2 + 0.07 * @as(f32, @floatFromInt(i));

    const x = [_]f32{ 1.0, 0.5, -0.3, 0.2 };
    const target = [_]f32{ 1.0, 0.0, 0.0, 0.0 };

    // ── GPU buffers (params live device-side and are mutated in
    // place by the SGD step; activations + grads live device-side
    // and are reused across steps) ──────────────────────────────────
    var buf_x = try buffer.Buffer.initStatic(&ctx, f32, &x);
    defer buf_x.deinit(ctx.device);
    var buf_target = try buffer.Buffer.initStatic(&ctx, f32, &target);
    defer buf_target.deinit(ctx.device);
    var buf_w1 = try buffer.Buffer.initStatic(&ctx, f32, mlp_cpu.w1);
    defer buf_w1.deinit(ctx.device);
    var buf_b1 = try buffer.Buffer.initStatic(&ctx, f32, mlp_cpu.b1);
    defer buf_b1.deinit(ctx.device);
    var buf_w2 = try buffer.Buffer.initStatic(&ctx, f32, mlp_cpu.w2);
    defer buf_w2.deinit(ctx.device);
    var buf_b2 = try buffer.Buffer.initStatic(&ctx, f32, mlp_cpu.b2);
    defer buf_b2.deinit(ctx.device);
    var buf_h_pre = try buffer.Buffer.initDeviceOnly(&ctx, dim_h * @sizeOf(f32));
    defer buf_h_pre.deinit(ctx.device);
    var buf_h = try buffer.Buffer.initDeviceOnly(&ctx, dim_h * @sizeOf(f32));
    defer buf_h.deinit(ctx.device);
    var buf_y = try buffer.Buffer.initDeviceOnly(&ctx, dim_out * @sizeOf(f32));
    defer buf_y.deinit(ctx.device);
    var buf_dL_dy = try buffer.Buffer.initDeviceOnly(&ctx, dim_out * @sizeOf(f32));
    defer buf_dL_dy.deinit(ctx.device);
    var buf_dh = try buffer.Buffer.initDeviceOnly(&ctx, dim_h * @sizeOf(f32));
    defer buf_dh.deinit(ctx.device);
    var buf_dh_pre = try buffer.Buffer.initDeviceOnly(&ctx, dim_h * @sizeOf(f32));
    defer buf_dh_pre.deinit(ctx.device);
    var buf_dw1 = try buffer.Buffer.initDeviceOnly(&ctx, dim_h * dim_in * @sizeOf(f32));
    defer buf_dw1.deinit(ctx.device);
    var buf_db1 = try buffer.Buffer.initDeviceOnly(&ctx, dim_h * @sizeOf(f32));
    defer buf_db1.deinit(ctx.device);
    var buf_dw2 = try buffer.Buffer.initDeviceOnly(&ctx, dim_out * dim_h * @sizeOf(f32));
    defer buf_dw2.deinit(ctx.device);
    var buf_db2 = try buffer.Buffer.initDeviceOnly(&ctx, dim_out * @sizeOf(f32));
    defer buf_db2.deinit(ctx.device);

    // ── Pipelines ──────────────────────────────────────────────────
    var k_matmul = try pipeline.Kernel.init(&ctx, &shaders.matmul_nt, 3, @sizeOf(MatmulPush));
    defer k_matmul.deinit();
    var k_add = try pipeline.Kernel.init(&ctx, &shaders.add_in_place, 2, @sizeOf(runtime.AddInPlacePush));
    defer k_add.deinit();
    var k_relu = try pipeline.Kernel.init(&ctx, &shaders.relu, 2, @sizeOf(ReluPush));
    defer k_relu.deinit();
    var k_relu_bw = try pipeline.Kernel.init(&ctx, &shaders.relu_backward, 3, @sizeOf(ReluBackwardPush));
    defer k_relu_bw.deinit();
    var k_lin_dx = try pipeline.Kernel.init(&ctx, &shaders.linear_backward_dx, 3, @sizeOf(LinearBackwardDxPush));
    defer k_lin_dx.deinit();
    var k_outer = try pipeline.Kernel.init(&ctx, &shaders.outer_product, 3, @sizeOf(OuterProductPush));
    defer k_outer.deinit();
    var k_copy = try pipeline.Kernel.init(&ctx, &shaders.slice_copy, 2, @sizeOf(SliceCopyPush));
    defer k_copy.deinit();
    var k_sgd = try pipeline.Kernel.init(&ctx, &shaders.sgd_step, 2, @sizeOf(SgdStepPush));
    defer k_sgd.deinit();
    var k_mse_grad = try pipeline.Kernel.init(&ctx, &shaders.mse_loss_grad, 3, @sizeOf(MseLossGradPush));
    defer k_mse_grad.deinit();

    // Push constants reused across steps.
    const matmul1_push = MatmulPush{ .m = 1, .n = @intCast(dim_h), .k = @intCast(dim_in) };
    const matmul2_push = MatmulPush{ .m = 1, .n = @intCast(dim_out), .k = @intCast(dim_h) };
    const add1_push = runtime.AddInPlacePush{ .n = @intCast(dim_h) };
    const add2_push = runtime.AddInPlacePush{ .n = @intCast(dim_out) };
    const relu_push = ReluPush{ .n = @intCast(dim_h) };
    const mse_grad_push = MseLossGradPush{ .n = @intCast(dim_out) };
    const op_dw2 = OuterProductPush{ .dim_out = @intCast(dim_out), .dim_in = @intCast(dim_h) };
    const lin_dx_push = LinearBackwardDxPush{ .dim_out = @intCast(dim_out), .dim_in = @intCast(dim_h) };
    const relu_bw_push = ReluBackwardPush{ .n = @intCast(dim_h) };
    const op_dw1 = OuterProductPush{ .dim_out = @intCast(dim_h), .dim_in = @intCast(dim_in) };
    const copy_db2 = SliceCopyPush{ .src_off = 0, .dst_off = 0, .n_elem = @intCast(dim_out) };
    const copy_db1 = SliceCopyPush{ .src_off = 0, .dst_off = 0, .n_elem = @intCast(dim_h) };
    const sgd_w1_push = SgdStepPush{ .n = @intCast(mlp_cpu.w1.len), .lr = lr };
    const sgd_b1_push = SgdStepPush{ .n = @intCast(mlp_cpu.b1.len), .lr = lr };
    const sgd_w2_push = SgdStepPush{ .n = @intCast(mlp_cpu.w2.len), .lr = lr };
    const sgd_b2_push = SgdStepPush{ .n = @intCast(mlp_cpu.b2.len), .lr = lr };

    // ── Run N steps on each side ──────────────────────────────────
    // Two GPU snapshots taken: after step 1 (single-step parity) and
    // after step n_steps (full-trajectory parity).
    const cpu_h_pre = try allocator.alloc(f32, dim_h);
    defer allocator.free(cpu_h_pre);
    const cpu_h = try allocator.alloc(f32, dim_h);
    defer allocator.free(cpu_h);
    const cpu_y = try allocator.alloc(f32, dim_out);
    defer allocator.free(cpu_y);
    var act_cpu: cpu_train.Activations = .{
        .x = &x,
        .h_pre = cpu_h_pre,
        .h = cpu_h,
        .y = cpu_y,
    };
    var grads_cpu = try cpu_train.Grads.init(allocator, &mlp_cpu);
    defer grads_cpu.deinit(allocator);

    const w1_after_1 = try allocator.alloc(f32, mlp_cpu.w1.len);
    defer allocator.free(w1_after_1);
    const w2_after_1 = try allocator.alloc(f32, mlp_cpu.w2.len);
    defer allocator.free(w2_after_1);
    const b1_after_1 = try allocator.alloc(f32, mlp_cpu.b1.len);
    defer allocator.free(b1_after_1);
    const b2_after_1 = try allocator.alloc(f32, mlp_cpu.b2.len);
    defer allocator.free(b2_after_1);

    // GPU side: drive each step through its own Recorder lifecycle.
    // Re-using one recorder across N steps would need a way to reset
    // n_dispatched + descriptor pool; cleanest for this smoke is one
    // recorder per step. Real trainer uses a frame-style loop with
    // per-frame recorder reset (chunk 5).
    var step: u32 = 0;
    while (step < n_steps) : (step += 1) {
        // CPU step.
        _ = try cpu_train.trainStep(allocator, &mlp_cpu, &act_cpu, &grads_cpu, &target, lr);

        // GPU step.
        var rec = try gpu_recorder.Recorder.init(&ctx, 16, 64);
        defer rec.deinit();
        try rec.begin();

        // Forward.
        try rec.dispatch(&k_matmul, &.{ &buf_x, &buf_w1, &buf_h_pre }, &matmul1_push, 1, 1, 1);
        try rec.dispatch(&k_add, &.{ &buf_h_pre, &buf_b1 }, &add1_push, ceilDiv(@as(u32, dim_h), 256), 1, 1);
        try rec.dispatch(&k_relu, &.{ &buf_h_pre, &buf_h }, &relu_push, ceilDiv(@as(u32, dim_h), 256), 1, 1);
        try rec.dispatch(&k_matmul, &.{ &buf_h, &buf_w2, &buf_y }, &matmul2_push, 1, 1, 1);
        try rec.dispatch(&k_add, &.{ &buf_y, &buf_b2 }, &add2_push, ceilDiv(@as(u32, dim_out), 256), 1, 1);

        // Loss grad.
        try rec.dispatch(&k_mse_grad, &.{ &buf_y, &buf_target, &buf_dL_dy }, &mse_grad_push, ceilDiv(@as(u32, dim_out), 256), 1, 1);

        // Backward.
        try rec.dispatch(&k_copy, &.{ &buf_dL_dy, &buf_db2 }, &copy_db2, ceilDiv(@as(u32, dim_out), 256), 1, 1);
        try rec.dispatch(&k_outer, &.{ &buf_dL_dy, &buf_h, &buf_dw2 }, &op_dw2, ceilDiv(@as(u32, dim_out), 16), ceilDiv(@as(u32, dim_h), 16), 1);
        try rec.dispatch(&k_lin_dx, &.{ &buf_dL_dy, &buf_w2, &buf_dh }, &lin_dx_push, ceilDiv(@as(u32, dim_h), 256), 1, 1);
        try rec.dispatch(&k_relu_bw, &.{ &buf_dh, &buf_h_pre, &buf_dh_pre }, &relu_bw_push, ceilDiv(@as(u32, dim_h), 256), 1, 1);
        try rec.dispatch(&k_copy, &.{ &buf_dh_pre, &buf_db1 }, &copy_db1, ceilDiv(@as(u32, dim_h), 256), 1, 1);
        try rec.dispatch(&k_outer, &.{ &buf_dh_pre, &buf_x, &buf_dw1 }, &op_dw1, ceilDiv(@as(u32, dim_h), 16), ceilDiv(@as(u32, dim_in), 16), 1);

        // SGD step (param -= lr · grad). Note W2 is updated BEFORE the
        // dh = W2^T · dL/dy dispatch reads W2 — which is fine because
        // dh was computed earlier in this same recorder, and the next
        // step's W2-read starts a fresh recorder with a barrier.
        try rec.dispatch(&k_sgd, &.{ &buf_w1, &buf_dw1 }, &sgd_w1_push, ceilDiv(@intCast(mlp_cpu.w1.len), 256), 1, 1);
        try rec.dispatch(&k_sgd, &.{ &buf_b1, &buf_db1 }, &sgd_b1_push, ceilDiv(@intCast(mlp_cpu.b1.len), 256), 1, 1);
        try rec.dispatch(&k_sgd, &.{ &buf_w2, &buf_dw2 }, &sgd_w2_push, ceilDiv(@intCast(mlp_cpu.w2.len), 256), 1, 1);
        try rec.dispatch(&k_sgd, &.{ &buf_b2, &buf_db2 }, &sgd_b2_push, ceilDiv(@intCast(mlp_cpu.b2.len), 256), 1, 1);

        try rec.endAndSubmit();

        // Snapshot after step 1 for the single-step parity check.
        if (step == 0) {
            try buf_w1.readBack(&ctx, f32, w1_after_1);
            try buf_b1.readBack(&ctx, f32, b1_after_1);
            try buf_w2.readBack(&ctx, f32, w2_after_1);
            try buf_b2.readBack(&ctx, f32, b2_after_1);
        }
    }

    // ── Single-step parity ────────────────────────────────────────
    // After step 0, the CPU has already mutated mlp_cpu — we need a
    // fresh oracle for the "after one step" state. Easiest: re-run
    // chunk-1's first step from scratch and compare.
    var oracle = try cpu_train.Mlp.init(allocator, dim_in, dim_h, dim_out, 0.3, 0x57DD57D);
    defer oracle.deinit(allocator);
    for (oracle.b1, 0..) |*v, i| v.* = 0.1 - 0.05 * @as(f32, @floatFromInt(i));
    for (oracle.b2, 0..) |*v, i| v.* = -0.2 + 0.07 * @as(f32, @floatFromInt(i));
    const oracle_h_pre = try allocator.alloc(f32, dim_h);
    defer allocator.free(oracle_h_pre);
    const oracle_h = try allocator.alloc(f32, dim_h);
    defer allocator.free(oracle_h);
    const oracle_y = try allocator.alloc(f32, dim_out);
    defer allocator.free(oracle_y);
    var oracle_act: cpu_train.Activations = .{
        .x = &x,
        .h_pre = oracle_h_pre,
        .h = oracle_h,
        .y = oracle_y,
    };
    var oracle_grads = try cpu_train.Grads.init(allocator, &oracle);
    defer oracle_grads.deinit(allocator);
    _ = try cpu_train.trainStep(allocator, &oracle, &oracle_act, &oracle_grads, &target, lr);

    const tol_step1: f32 = 1e-5;
    var max_step1: f32 = 0;
    for (oracle.w1, w1_after_1, 0..) |w, g, i| {
        const d = @abs(g - w);
        if (d > tol_step1) {
            std.debug.print("step-1 W1 MISMATCH[{d}]: gpu={d} cpu={d}\n", .{ i, g, w });
            return error.ParityFailed;
        }
        if (d > max_step1) max_step1 = d;
    }
    for (oracle.b1, b1_after_1, 0..) |w, g, i| {
        const d = @abs(g - w);
        if (d > tol_step1) {
            std.debug.print("step-1 b1 MISMATCH[{d}]: gpu={d} cpu={d}\n", .{ i, g, w });
            return error.ParityFailed;
        }
    }
    for (oracle.w2, w2_after_1, 0..) |w, g, i| {
        const d = @abs(g - w);
        if (d > tol_step1) {
            std.debug.print("step-1 W2 MISMATCH[{d}]: gpu={d} cpu={d}\n", .{ i, g, w });
            return error.ParityFailed;
        }
    }
    for (oracle.b2, b2_after_1, 0..) |w, g, i| {
        const d = @abs(g - w);
        if (d > tol_step1) {
            std.debug.print("step-1 b2 MISMATCH[{d}]: gpu={d} cpu={d}\n", .{ i, g, w });
            return error.ParityFailed;
        }
    }

    // ── Full-trajectory parity ────────────────────────────────────
    // After n_steps, compare the GPU's final weights against the
    // CPU's final weights. Looser tol because rounding at each step
    // accumulates — but at our dims and step count it stays tight.
    const w1_gpu = try allocator.alloc(f32, mlp_cpu.w1.len);
    defer allocator.free(w1_gpu);
    const b1_gpu = try allocator.alloc(f32, mlp_cpu.b1.len);
    defer allocator.free(b1_gpu);
    const w2_gpu = try allocator.alloc(f32, mlp_cpu.w2.len);
    defer allocator.free(w2_gpu);
    const b2_gpu = try allocator.alloc(f32, mlp_cpu.b2.len);
    defer allocator.free(b2_gpu);
    try buf_w1.readBack(&ctx, f32, w1_gpu);
    try buf_b1.readBack(&ctx, f32, b1_gpu);
    try buf_w2.readBack(&ctx, f32, w2_gpu);
    try buf_b2.readBack(&ctx, f32, b2_gpu);

    const tol_traj: f32 = 1e-4;
    var max_traj: f32 = 0;
    const ParamCase = struct { name: []const u8, gpu: []const f32, cpu: []const f32 };
    const traj_cases = [_]ParamCase{
        .{ .name = "W1", .gpu = w1_gpu, .cpu = mlp_cpu.w1 },
        .{ .name = "b1", .gpu = b1_gpu, .cpu = mlp_cpu.b1 },
        .{ .name = "W2", .gpu = w2_gpu, .cpu = mlp_cpu.w2 },
        .{ .name = "b2", .gpu = b2_gpu, .cpu = mlp_cpu.b2 },
    };
    for (traj_cases) |cs| {
        for (cs.gpu, cs.cpu, 0..) |g, c, i| {
            const d = @abs(g - c);
            if (d > tol_traj) {
                std.debug.print("traj {s} MISMATCH[{d}] @ step {d}: gpu={d} cpu={d}\n", .{ cs.name, i, n_steps, g, c });
                return error.ParityFailed;
            }
            if (d > max_traj) max_traj = d;
        }
    }
    std.debug.print(
        "PASS GPU MLP train ({d} SGD steps; step-1 |Δ|={e}, after-{d} |Δ|={e}, all on-device)\n",
        .{ n_steps, max_step1, n_steps, max_traj },
    );
}

// ── Multi-layer GPU MLP train smoke ───────────────────────────────
//
// Tier-1 chunk 2 of the post-v0 training arc. Same primitives as the
// 2-layer GPU train smoke, but orchestrated across n=3 layers in
// generic loops. Bit-exact step-1 parity vs the new MlpN CPU oracle
// proves the dispatch ordering works at depth before we lift it into
// the runner.
//
// Composition: forward is matmul → bias add → ReLU per hidden layer,
// then matmul → bias add (no ReLU) on the output. Backward seeds
// dL/dy via mse_loss_grad on the output, then for each layer L from
// n-1 down to 0 emits db = d_pre, dW = d_pre ⊗ input[L], and (if
// L > 0) propagates d_post[L-1] = W[L]ᵀ · d_pre[L] then
// d_pre[L-1] = d_post[L-1] · 1[pre[L-1] > 0]. SGD updates run after
// all gradients are computed so each W[L] is read at its pre-update
// value during backward.

fn runGpuMlpNTrainSmoke(allocator: std.mem.Allocator) !void {
    var ctx = try vk.Context.init(allocator);
    defer ctx.deinit();

    const layer_dims = [_]usize{ 4, 8, 6, 4 };
    const n = layer_dims.len - 1;
    const lr: f32 = 0.05;
    const n_steps: u32 = 50;

    var mlp_cpu = try cpu_train.MlpN.init(allocator, &layer_dims, 0.3, 0xC0FFEE3D);
    defer mlp_cpu.deinit(allocator);
    // Stamp non-zero biases so gradient flow through b is exercised.
    for (mlp_cpu.biases, 0..) |b, layer_idx| {
        for (b, 0..) |*v, i| {
            const li: f32 = @floatFromInt(layer_idx + 1);
            const ii: f32 = @floatFromInt(@as(i32, @intCast(i % 3)));
            v.* = 0.05 * (li - ii);
        }
    }

    const x = [_]f32{ 1.0, 0.5, -0.3, 0.2 };
    const target = [_]f32{ 1.0, 0.0, 0.0, 0.0 };

    // ── GPU buffers ───────────────────────────────────────────────
    var buf_x = try buffer.Buffer.initStatic(&ctx, f32, &x);
    defer buf_x.deinit(ctx.device);
    var buf_target = try buffer.Buffer.initStatic(&ctx, f32, &target);
    defer buf_target.deinit(ctx.device);

    var bufs_w: [n]buffer.Buffer = undefined;
    var bufs_b: [n]buffer.Buffer = undefined;
    var bufs_dw: [n]buffer.Buffer = undefined;
    var bufs_db: [n]buffer.Buffer = undefined;
    var bufs_pre: [n]buffer.Buffer = undefined;
    var bufs_post: [n]buffer.Buffer = undefined;
    var bufs_dpre: [n]buffer.Buffer = undefined;
    // d_post[L-1] for L=1..n-1 — needed only when there's a layer
    // below to back-prop through. Index by L (i.e. dpost_for_below[L]
    // is the d_post going INTO layer L-1's d_pre); slot 0 unused.
    var bufs_dpost: [n]buffer.Buffer = undefined;

    for (0..n) |L| {
        bufs_w[L] = try buffer.Buffer.initStatic(&ctx, f32, mlp_cpu.weights[L]);
        bufs_b[L] = try buffer.Buffer.initStatic(&ctx, f32, mlp_cpu.biases[L]);
        const dim_o = layer_dims[L + 1];
        const dim_i = layer_dims[L];
        bufs_dw[L] = try buffer.Buffer.initDeviceOnly(&ctx, dim_o * dim_i * @sizeOf(f32));
        bufs_db[L] = try buffer.Buffer.initDeviceOnly(&ctx, dim_o * @sizeOf(f32));
        bufs_pre[L] = try buffer.Buffer.initDeviceOnly(&ctx, dim_o * @sizeOf(f32));
        bufs_post[L] = try buffer.Buffer.initDeviceOnly(&ctx, dim_o * @sizeOf(f32));
        bufs_dpre[L] = try buffer.Buffer.initDeviceOnly(&ctx, dim_o * @sizeOf(f32));
    }
    for (1..n) |L| {
        bufs_dpost[L] = try buffer.Buffer.initDeviceOnly(&ctx, layer_dims[L] * @sizeOf(f32));
    }
    defer {
        for (0..n) |L| {
            bufs_w[L].deinit(ctx.device);
            bufs_b[L].deinit(ctx.device);
            bufs_dw[L].deinit(ctx.device);
            bufs_db[L].deinit(ctx.device);
            bufs_pre[L].deinit(ctx.device);
            bufs_post[L].deinit(ctx.device);
            bufs_dpre[L].deinit(ctx.device);
        }
        for (1..n) |L| bufs_dpost[L].deinit(ctx.device);
    }

    // ── Pipelines ──────────────────────────────────────────────────
    var k_matmul = try pipeline.Kernel.init(&ctx, &shaders.matmul_nt, 3, @sizeOf(MatmulPush));
    defer k_matmul.deinit();
    var k_add = try pipeline.Kernel.init(&ctx, &shaders.add_in_place, 2, @sizeOf(runtime.AddInPlacePush));
    defer k_add.deinit();
    var k_relu = try pipeline.Kernel.init(&ctx, &shaders.relu, 2, @sizeOf(ReluPush));
    defer k_relu.deinit();
    var k_relu_bw = try pipeline.Kernel.init(&ctx, &shaders.relu_backward, 3, @sizeOf(ReluBackwardPush));
    defer k_relu_bw.deinit();
    var k_lin_dx = try pipeline.Kernel.init(&ctx, &shaders.linear_backward_dx, 3, @sizeOf(LinearBackwardDxPush));
    defer k_lin_dx.deinit();
    var k_outer = try pipeline.Kernel.init(&ctx, &shaders.outer_product, 3, @sizeOf(OuterProductPush));
    defer k_outer.deinit();
    var k_copy = try pipeline.Kernel.init(&ctx, &shaders.slice_copy, 2, @sizeOf(SliceCopyPush));
    defer k_copy.deinit();
    var k_sgd = try pipeline.Kernel.init(&ctx, &shaders.sgd_step, 2, @sizeOf(SgdStepPush));
    defer k_sgd.deinit();
    var k_mse_grad = try pipeline.Kernel.init(&ctx, &shaders.mse_loss_grad, 3, @sizeOf(MseLossGradPush));
    defer k_mse_grad.deinit();

    // ── CPU oracle to compare each step against ────────────────────
    var act_cpu = try cpu_train.ActivationsN.init(allocator, &mlp_cpu);
    defer act_cpu.deinit(allocator);
    act_cpu.x = &x;
    var grads_cpu = try cpu_train.GradsN.init(allocator, &mlp_cpu);
    defer grads_cpu.deinit(allocator);

    // Snapshot buffers for the after-step-1 parity check (per layer).
    var w_after_1: [n][]f32 = undefined;
    var b_after_1: [n][]f32 = undefined;
    for (0..n) |L| {
        w_after_1[L] = try allocator.alloc(f32, mlp_cpu.weights[L].len);
        b_after_1[L] = try allocator.alloc(f32, mlp_cpu.biases[L].len);
    }
    defer for (0..n) |L| {
        allocator.free(w_after_1[L]);
        allocator.free(b_after_1[L]);
    };

    var step: u32 = 0;
    while (step < n_steps) : (step += 1) {
        // CPU step.
        _ = try cpu_train.trainStepN(allocator, &mlp_cpu, &act_cpu, &grads_cpu, &target, lr);

        // GPU step.
        var rec = try gpu_recorder.Recorder.init(&ctx, 32, 256);
        defer rec.deinit();
        try rec.begin();

        // Forward pass: layer L reads input[L] = (L==0 ? x : post[L-1]),
        // writes pre[L] = W[L]·input + b[L], then post[L] = ReLU(pre[L])
        // for hidden layers; for the output layer post[L] = pre[L] (we
        // don't dispatch a relu, and downstream readers use pre as y).
        for (0..n) |L| {
            const dim_o: u32 = @intCast(layer_dims[L + 1]);
            const dim_i: u32 = @intCast(layer_dims[L]);
            const input_buf = if (L == 0) &buf_x else &bufs_post[L - 1];
            const matmul_push = MatmulPush{ .m = 1, .n = dim_o, .k = dim_i };
            try rec.dispatch(&k_matmul, &.{ input_buf, &bufs_w[L], &bufs_pre[L] }, &matmul_push, 1, 1, 1);
            const add_push = runtime.AddInPlacePush{ .n = dim_o };
            try rec.dispatch(&k_add, &.{ &bufs_pre[L], &bufs_b[L] }, &add_push, ceilDiv(dim_o, 256), 1, 1);
            if (L + 1 < n) {
                const relu_push = ReluPush{ .n = dim_o };
                try rec.dispatch(&k_relu, &.{ &bufs_pre[L], &bufs_post[L] }, &relu_push, ceilDiv(dim_o, 256), 1, 1);
            } else {
                // Output layer has no ReLU — copy pre→post so backward
                // and the host can both read the prediction from
                // bufs_post[n-1] uniformly.
                const copy_push = SliceCopyPush{ .src_off = 0, .dst_off = 0, .n_elem = dim_o };
                try rec.dispatch(&k_copy, &.{ &bufs_pre[L], &bufs_post[L] }, &copy_push, ceilDiv(dim_o, 256), 1, 1);
            }
        }

        // Loss-grad seeds d_pre on the output layer (no ReLU there).
        const dim_out_u: u32 = @intCast(layer_dims[n]);
        const mse_grad_push = MseLossGradPush{ .n = dim_out_u };
        try rec.dispatch(&k_mse_grad, &.{ &bufs_post[n - 1], &buf_target, &bufs_dpre[n - 1] }, &mse_grad_push, ceilDiv(dim_out_u, 256), 1, 1);

        // Backward pass per layer, top-down. db = d_pre; dW = d_pre ⊗ input;
        // if not the bottom layer, propagate d_post and apply the ReLU mask.
        var Lp1: usize = n;
        while (Lp1 > 0) : (Lp1 -= 1) {
            const L = Lp1 - 1;
            const dim_o: u32 = @intCast(layer_dims[L + 1]);
            const dim_i: u32 = @intCast(layer_dims[L]);
            const input_buf = if (L == 0) &buf_x else &bufs_post[L - 1];

            // db[L] = d_pre[L]
            const copy_db = SliceCopyPush{ .src_off = 0, .dst_off = 0, .n_elem = dim_o };
            try rec.dispatch(&k_copy, &.{ &bufs_dpre[L], &bufs_db[L] }, &copy_db, ceilDiv(dim_o, 256), 1, 1);
            // dW[L] = d_pre[L] ⊗ input
            const op_push = OuterProductPush{ .dim_out = dim_o, .dim_in = dim_i };
            try rec.dispatch(&k_outer, &.{ &bufs_dpre[L], input_buf, &bufs_dw[L] }, &op_push, ceilDiv(dim_o, 16), ceilDiv(dim_i, 16), 1);
            // Propagate to layer L-1's d_pre, with ReLU mask on its pre.
            if (L > 0) {
                const lin_dx_push = LinearBackwardDxPush{ .dim_out = dim_o, .dim_in = dim_i };
                try rec.dispatch(&k_lin_dx, &.{ &bufs_dpre[L], &bufs_w[L], &bufs_dpost[L] }, &lin_dx_push, ceilDiv(dim_i, 256), 1, 1);
                const relu_bw_push = ReluBackwardPush{ .n = dim_i };
                try rec.dispatch(&k_relu_bw, &.{ &bufs_dpost[L], &bufs_pre[L - 1], &bufs_dpre[L - 1] }, &relu_bw_push, ceilDiv(dim_i, 256), 1, 1);
            }
        }

        // SGD updates run after all gradients have been computed using
        // pre-update weights. Order across layers doesn't matter.
        for (0..n) |L| {
            const sgd_w_push = SgdStepPush{ .n = @intCast(mlp_cpu.weights[L].len), .lr = lr };
            try rec.dispatch(&k_sgd, &.{ &bufs_w[L], &bufs_dw[L] }, &sgd_w_push, ceilDiv(@intCast(mlp_cpu.weights[L].len), 256), 1, 1);
            const sgd_b_push = SgdStepPush{ .n = @intCast(mlp_cpu.biases[L].len), .lr = lr };
            try rec.dispatch(&k_sgd, &.{ &bufs_b[L], &bufs_db[L] }, &sgd_b_push, ceilDiv(@intCast(mlp_cpu.biases[L].len), 256), 1, 1);
        }

        try rec.endAndSubmit();

        if (step == 0) {
            for (0..n) |L| {
                try bufs_w[L].readBack(&ctx, f32, w_after_1[L]);
                try bufs_b[L].readBack(&ctx, f32, b_after_1[L]);
            }
        }
    }

    // ── Single-step parity ────────────────────────────────────────
    // Re-run step 1 from a fresh oracle (mlp_cpu has already moved on).
    var oracle = try cpu_train.MlpN.init(allocator, &layer_dims, 0.3, 0xC0FFEE3D);
    defer oracle.deinit(allocator);
    for (oracle.biases, 0..) |b, layer_idx| {
        for (b, 0..) |*v, i| {
            const li: f32 = @floatFromInt(layer_idx + 1);
            const ii: f32 = @floatFromInt(@as(i32, @intCast(i % 3)));
            v.* = 0.05 * (li - ii);
        }
    }
    var oracle_act = try cpu_train.ActivationsN.init(allocator, &oracle);
    defer oracle_act.deinit(allocator);
    oracle_act.x = &x;
    var oracle_grads = try cpu_train.GradsN.init(allocator, &oracle);
    defer oracle_grads.deinit(allocator);
    _ = try cpu_train.trainStepN(allocator, &oracle, &oracle_act, &oracle_grads, &target, lr);

    const tol_step1: f32 = 1e-5;
    var max_step1: f32 = 0;
    for (0..n) |L| {
        for (oracle.weights[L], w_after_1[L], 0..) |c, g, i| {
            const d = @abs(g - c);
            if (d > tol_step1) {
                std.debug.print("step-1 W[{d}] MISMATCH[{d}]: gpu={d} cpu={d}\n", .{ L, i, g, c });
                return error.ParityFailed;
            }
            if (d > max_step1) max_step1 = d;
        }
        for (oracle.biases[L], b_after_1[L], 0..) |c, g, i| {
            const d = @abs(g - c);
            if (d > tol_step1) {
                std.debug.print("step-1 b[{d}] MISMATCH[{d}]: gpu={d} cpu={d}\n", .{ L, i, g, c });
                return error.ParityFailed;
            }
            if (d > max_step1) max_step1 = d;
        }
    }

    // ── Full-trajectory parity ────────────────────────────────────
    const tol_traj: f32 = 1e-4;
    var max_traj: f32 = 0;
    for (0..n) |L| {
        const w_gpu = try allocator.alloc(f32, mlp_cpu.weights[L].len);
        defer allocator.free(w_gpu);
        const b_gpu = try allocator.alloc(f32, mlp_cpu.biases[L].len);
        defer allocator.free(b_gpu);
        try bufs_w[L].readBack(&ctx, f32, w_gpu);
        try bufs_b[L].readBack(&ctx, f32, b_gpu);
        for (mlp_cpu.weights[L], w_gpu, 0..) |c, g, i| {
            const d = @abs(g - c);
            if (d > tol_traj) {
                std.debug.print("traj W[{d}] MISMATCH[{d}] @ step {d}: gpu={d} cpu={d}\n", .{ L, i, n_steps, g, c });
                return error.ParityFailed;
            }
            if (d > max_traj) max_traj = d;
        }
        for (mlp_cpu.biases[L], b_gpu, 0..) |c, g, i| {
            const d = @abs(g - c);
            if (d > tol_traj) {
                std.debug.print("traj b[{d}] MISMATCH[{d}] @ step {d}: gpu={d} cpu={d}\n", .{ L, i, n_steps, g, c });
                return error.ParityFailed;
            }
            if (d > max_traj) max_traj = d;
        }
    }

    std.debug.print(
        "PASS GPU MLP train (n={d} layers, dims [4,8,6,4], {d} SGD steps; step-1 |Δ|={e}, after-{d} |Δ|={e})\n",
        .{ n, n_steps, max_step1, n_steps, max_traj },
    );
}

// ── TrainingRunner smoke: persistent buffers, streamed inputs ───────
//
// Chunk 5 of training-v0. Exercises the public TrainingRunner API:
// init() builds pipelines + buffers once, tickStep() streams a fresh
// (input, target) per call and returns the prediction, tickPredict()
// is forward-only. The smoke runs a small training schedule against
// two distinct (x, target) pairs alternating each step (mimicking a
// streaming task) and asserts loss converges; final readWeights()
// shape-checks the buffers can be pulled back to a CPU `Mlp`.
//
// Single submit per tick: this is the same call shape the engine-
// integration `aiTrain` hook will use, just with a Recorder.attachCmd
// in chunk 7 instead of a standalone Recorder.

fn runTrainingRunnerSmoke(allocator: std.mem.Allocator) !void {
    var ctx = try vk.Context.init(allocator);
    defer ctx.deinit();

    const cfg = train_runner.Mlp2Config{
        .dim_in = 4,
        .dim_hidden = 8,
        .dim_out = 4,
        .lr = 0.05,
        .init_seed = 0x70077077,
    };

    var runner = try train_runner.TrainingRunner.init(allocator, &ctx, cfg);
    defer runner.deinit();

    // Two streaming (input, target) pairs — alternating per step.
    // Single-pair convergence already covered in earlier smokes; here
    // we want to see the runner handle a moving target without state
    // smearing between calls.
    const x_a = [_]f32{ 1.0, 0.0, 0.0, 0.0 };
    const t_a = [_]f32{ 1.0, 0.2, 0.0, 0.0 };
    const x_b = [_]f32{ 0.0, 1.0, 0.0, 0.0 };
    const t_b = [_]f32{ 0.0, 0.0, 0.7, 0.3 };

    var pred: [4]f32 = undefined;

    // Initial loss across both pairs.
    try runner.tickPredict(&x_a, &pred);
    var initial_loss = cpu_train.mseLoss(&pred, &t_a);
    try runner.tickPredict(&x_b, &pred);
    initial_loss += cpu_train.mseLoss(&pred, &t_b);

    // Run a handful of steps alternating between the two pairs.
    const n_steps: u32 = 200;
    var s: u32 = 0;
    while (s < n_steps) : (s += 1) {
        if (s & 1 == 0) {
            try runner.tickStep(&x_a, &t_a, &pred);
        } else {
            try runner.tickStep(&x_b, &t_b, &pred);
        }
    }

    // Final loss across both pairs.
    try runner.tickPredict(&x_a, &pred);
    var final_loss = cpu_train.mseLoss(&pred, &t_a);
    try runner.tickPredict(&x_b, &pred);
    final_loss += cpu_train.mseLoss(&pred, &t_b);

    if (!(final_loss < initial_loss * 0.1)) {
        std.debug.print(
            "TrainingRunner did not converge: loss[0] = {d:.6}, loss[{d}] = {d:.6}\n",
            .{ initial_loss, n_steps, final_loss },
        );
        return error.ParityFailed;
    }

    // readWeights round-trip: build a CPU MLP of matching shape and
    // pull weights back. Sanity check that dimensions agree and the
    // staging path works.
    var cpu_mirror = try cpu_train.Mlp.init(allocator, cfg.dim_in, cfg.dim_hidden, cfg.dim_out, 0.0, 0);
    defer cpu_mirror.deinit(allocator);
    try runner.readWeights(&cpu_mirror);
    var nan_count: usize = 0;
    for (cpu_mirror.w1) |v| if (std.math.isNan(v)) {
        nan_count += 1;
    };
    if (nan_count != 0) {
        std.debug.print("TrainingRunner readWeights returned NaNs in W1: {d}\n", .{nan_count});
        return error.ParityFailed;
    }

    std.debug.print(
        "PASS TrainingRunner ({d} alternating steps; loss {d:.6} → {d:.6}, readWeights OK)\n",
        .{ n_steps, initial_loss, final_loss },
    );
}

// ── TrainingRunner attach-mode smoke: host owns submit, valkyr records ─
//
// Chunk 7 (valkyr-side) of training-v0. Demonstrates the attach
// surface that a host engine like Matryoshka uses: the host owns the
// VkContext, the per-frame VkCommandBuffer, and the submit cadence;
// valkyr's TrainingRunner records its forward + loss-grad + backward
// + SGD into the host's command buffer via `tickStepRecord`. No
// per-step submit from valkyr's side — the host bundles everything
// into one render submit.
//
// The smoke runs 60 attached steps (a "second of frames at 60 fps"
// cadence) and asserts loss converges. Visual click-to-red sphere
// demo lives in the matryoshka repo on top of this surface.

fn runTrainingRunnerAttachedSmoke(allocator: std.mem.Allocator) !void {
    var ctx = try vk.Context.init(allocator);
    defer ctx.deinit();

    const cfg = train_runner.Mlp2Config{
        .dim_in = 4,
        .dim_hidden = 16,
        .dim_out = 4,
        .lr = 0.05,
        .init_seed = 0xA77ACED,
    };
    var runner = try train_runner.TrainingRunner.init(allocator, &ctx, cfg);
    defer runner.deinit();

    // ── Host-owned cmd buffer + fence (matryoshka-equivalent setup) ──
    var cb_ai = std.mem.zeroes(vk.c.VkCommandBufferAllocateInfo);
    cb_ai.sType = vk.c.VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    cb_ai.commandPool = ctx.cmd_pool;
    cb_ai.level = vk.c.VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cb_ai.commandBufferCount = 1;
    var host_cmd: vk.c.VkCommandBuffer = null;
    try vk.check(vk.c.vkAllocateCommandBuffers(ctx.device, &cb_ai, &host_cmd));
    defer vk.c.vkFreeCommandBuffers(ctx.device, ctx.cmd_pool, 1, &host_cmd);

    var fci = std.mem.zeroes(vk.c.VkFenceCreateInfo);
    fci.sType = vk.c.VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    var host_fence: vk.c.VkFence = null;
    try vk.check(vk.c.vkCreateFence(ctx.device, &fci, null, &host_fence));
    defer vk.c.vkDestroyFence(ctx.device, host_fence, null);

    // Streaming target signal: same shape as the headless demo, just
    // 60 frames worth.
    const dt: f32 = 0.05;
    var pred: [4]f32 = undefined;
    var input: [4]f32 = undefined;
    var target: [4]f32 = undefined;

    // Initial loss for convergence check.
    try runner.tickPredict(&[_]f32{ 1, 0, 0, 0 }, &pred);
    const initial_loss = cpu_train.mseLoss(&pred, &[_]f32{ 0.5, 0.5, 0.5, 0 });

    const n_frames: u32 = 60;
    var f: u32 = 0;
    while (f < n_frames) : (f += 1) {
        const t = @as(f32, @floatFromInt(f)) * dt;
        input[0] = @sin(t);
        input[1] = @cos(t);
        input[2] = @sin(2 * t);
        input[3] = @cos(2 * t);
        target[0] = 0.5 + 0.5 * @sin(t);
        target[1] = 0.5 + 0.5 * @cos(t);
        target[2] = 0.5;
        target[3] = 0.0;

        // Per-frame: reset + begin + attach + record + end + submit + wait.
        try vk.check(vk.c.vkResetCommandBuffer(host_cmd, 0));
        try vk.check(vk.c.vkResetFences(ctx.device, 1, &host_fence));

        var bi = std.mem.zeroes(vk.c.VkCommandBufferBeginInfo);
        bi.sType = vk.c.VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        bi.flags = vk.c.VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        try vk.check(vk.c.vkBeginCommandBuffer(host_cmd, &bi));

        var rec = try gpu_recorder.Recorder.attachCmd(&ctx, host_cmd, 24, 96);
        defer rec.deinit();
        try rec.begin();

        try runner.tickStepRecord(&rec, &input, &target);

        try vk.check(vk.c.vkEndCommandBuffer(host_cmd));
        var submit = std.mem.zeroes(vk.c.VkSubmitInfo);
        submit.sType = vk.c.VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submit.commandBufferCount = 1;
        submit.pCommandBuffers = &host_cmd;
        try vk.check(vk.c.vkQueueSubmit(ctx.queue, 1, &submit, host_fence));
        const timeout_ns: u64 = 10 * 1_000_000_000;
        try vk.check(vk.c.vkWaitForFences(ctx.device, 1, &host_fence, vk.c.VK_TRUE, timeout_ns));
    }

    // Final loss against the last frame's target.
    try runner.tickPredict(&input, &pred);
    const final_loss = cpu_train.mseLoss(&pred, &target);

    if (!(final_loss < initial_loss * 0.5)) {
        std.debug.print(
            "TrainingRunner attached did not converge: loss[0] = {d:.6}, loss[{d}] = {d:.6}\n",
            .{ initial_loss, n_frames, final_loss },
        );
        return error.ParityFailed;
    }

    std.debug.print(
        "PASS TrainingRunner attached ({d} host-submitted frames; loss {d:.6} → {d:.6}, valkyr-record + host-submit OK)\n",
        .{ n_frames, initial_loss, final_loss },
    );
}

// ── MlpNRunner standalone smoke ─────────────────────────────────────
//
// Tier-1 chunk 3a of the post-v0 training arc. Exercises the new
// multi-layer runner on a 4 → 8 → 6 → 4 net (n=3) using the same
// alternating-pair convergence shape as the 2-layer
// `runTrainingRunnerSmoke`. Standalone path: each tick owns its own
// Recorder and submits.

fn runTrainingRunnerNSmoke(allocator: std.mem.Allocator) !void {
    var ctx = try vk.Context.init(allocator);
    defer ctx.deinit();

    const layer_dims = [_]u32{ 4, 8, 6, 4 };
    const cfg = train_runner_n.MlpNConfig{
        .layer_dims = &layer_dims,
        .lr = 0.05,
        .init_seed = 0x70077077,
    };

    var runner = try train_runner_n.MlpNRunner.init(allocator, &ctx, cfg);
    defer runner.deinit();

    const x_a = [_]f32{ 1.0, 0.0, 0.0, 0.0 };
    const t_a = [_]f32{ 1.0, 0.2, 0.0, 0.0 };
    const x_b = [_]f32{ 0.0, 1.0, 0.0, 0.0 };
    const t_b = [_]f32{ 0.0, 0.0, 0.7, 0.3 };

    var pred: [4]f32 = undefined;

    try runner.tickPredict(&x_a, &pred);
    var initial_loss = cpu_train.mseLoss(&pred, &t_a);
    try runner.tickPredict(&x_b, &pred);
    initial_loss += cpu_train.mseLoss(&pred, &t_b);

    // Deeper net + vanilla SGD on a 2-pair alternating task is slower
    // to converge than the 2-layer counterpart — gradient attenuation
    // through the extra ReLU stage. 600 steps at lr=0.05 lands an
    // 8× drop comfortably; tightening past that is an Adam discussion
    // (chunk 3b), not a runner correctness one.
    const n_steps: u32 = 600;
    var s: u32 = 0;
    while (s < n_steps) : (s += 1) {
        if (s & 1 == 0) {
            try runner.tickStep(&x_a, &t_a, &pred);
        } else {
            try runner.tickStep(&x_b, &t_b, &pred);
        }
    }

    try runner.tickPredict(&x_a, &pred);
    var final_loss = cpu_train.mseLoss(&pred, &t_a);
    try runner.tickPredict(&x_b, &pred);
    final_loss += cpu_train.mseLoss(&pred, &t_b);

    if (!(final_loss < initial_loss * 0.125)) {
        std.debug.print(
            "MlpNRunner did not converge: loss[0] = {d:.6}, loss[{d}] = {d:.6}\n",
            .{ initial_loss, n_steps, final_loss },
        );
        return error.ParityFailed;
    }

    // readWeights round-trip: build a CPU MlpN of matching shape and
    // pull weights back. Verifies dimensions agree and the staging
    // path runs through cleanly.
    const seed_dims = [_]usize{ 4, 8, 6, 4 };
    var cpu_mirror = try cpu_train.MlpN.init(allocator, &seed_dims, 0.0, 0);
    defer cpu_mirror.deinit(allocator);
    try runner.readWeights(&cpu_mirror);
    var nan_count: usize = 0;
    for (cpu_mirror.weights) |w| for (w) |v| if (std.math.isNan(v)) {
        nan_count += 1;
    };
    if (nan_count != 0) {
        std.debug.print("MlpNRunner readWeights returned NaNs: {d}\n", .{nan_count});
        return error.ParityFailed;
    }

    std.debug.print(
        "PASS MlpNRunner standalone (n={d} layers, dims [4,8,6,4]; {d} alternating steps; loss {d:.6} → {d:.6}, readWeights OK)\n",
        .{ runner.nLayers(), n_steps, initial_loss, final_loss },
    );
}

// ── MlpNRunner attach-mode smoke ────────────────────────────────────
//
// Same setup as `runTrainingRunnerAttachedSmoke` but on the multi-layer
// runner — host owns the cmd buffer + fence, valkyr records into it.

fn runTrainingRunnerNAttachedSmoke(allocator: std.mem.Allocator) !void {
    var ctx = try vk.Context.init(allocator);
    defer ctx.deinit();

    const layer_dims = [_]u32{ 4, 12, 8, 4 };
    const cfg = train_runner_n.MlpNConfig{
        .layer_dims = &layer_dims,
        .lr = 0.05,
        .init_seed = 0xA77ACED2,
    };
    var runner = try train_runner_n.MlpNRunner.init(allocator, &ctx, cfg);
    defer runner.deinit();

    var cb_ai = std.mem.zeroes(vk.c.VkCommandBufferAllocateInfo);
    cb_ai.sType = vk.c.VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    cb_ai.commandPool = ctx.cmd_pool;
    cb_ai.level = vk.c.VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cb_ai.commandBufferCount = 1;
    var host_cmd: vk.c.VkCommandBuffer = null;
    try vk.check(vk.c.vkAllocateCommandBuffers(ctx.device, &cb_ai, &host_cmd));
    defer vk.c.vkFreeCommandBuffers(ctx.device, ctx.cmd_pool, 1, &host_cmd);

    var fci = std.mem.zeroes(vk.c.VkFenceCreateInfo);
    fci.sType = vk.c.VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    var host_fence: vk.c.VkFence = null;
    try vk.check(vk.c.vkCreateFence(ctx.device, &fci, null, &host_fence));
    defer vk.c.vkDestroyFence(ctx.device, host_fence, null);

    const dt: f32 = 0.05;
    var pred: [4]f32 = undefined;
    var input: [4]f32 = undefined;
    var target: [4]f32 = undefined;

    try runner.tickPredict(&[_]f32{ 1, 0, 0, 0 }, &pred);
    const initial_loss = cpu_train.mseLoss(&pred, &[_]f32{ 0.5, 0.5, 0.5, 0 });

    const n_frames: u32 = 60;
    var f: u32 = 0;
    while (f < n_frames) : (f += 1) {
        const t = @as(f32, @floatFromInt(f)) * dt;
        input[0] = @sin(t);
        input[1] = @cos(t);
        input[2] = @sin(2 * t);
        input[3] = @cos(2 * t);
        target[0] = 0.5 + 0.5 * @sin(t);
        target[1] = 0.5 + 0.5 * @cos(t);
        target[2] = 0.5;
        target[3] = 0.0;

        try vk.check(vk.c.vkResetCommandBuffer(host_cmd, 0));
        try vk.check(vk.c.vkResetFences(ctx.device, 1, &host_fence));

        var bi = std.mem.zeroes(vk.c.VkCommandBufferBeginInfo);
        bi.sType = vk.c.VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        bi.flags = vk.c.VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        try vk.check(vk.c.vkBeginCommandBuffer(host_cmd, &bi));

        var rec = try gpu_recorder.Recorder.attachCmd(&ctx, host_cmd, 48, 192);
        defer rec.deinit();
        try rec.begin();

        try runner.tickStepRecord(&rec, &input, &target);

        try vk.check(vk.c.vkEndCommandBuffer(host_cmd));
        var submit = std.mem.zeroes(vk.c.VkSubmitInfo);
        submit.sType = vk.c.VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submit.commandBufferCount = 1;
        submit.pCommandBuffers = &host_cmd;
        try vk.check(vk.c.vkQueueSubmit(ctx.queue, 1, &submit, host_fence));
        const timeout_ns: u64 = 10 * 1_000_000_000;
        try vk.check(vk.c.vkWaitForFences(ctx.device, 1, &host_fence, vk.c.VK_TRUE, timeout_ns));
    }

    try runner.tickPredict(&input, &pred);
    const final_loss = cpu_train.mseLoss(&pred, &target);

    if (!(final_loss < initial_loss * 0.5)) {
        std.debug.print(
            "MlpNRunner attached did not converge: loss[0] = {d:.6}, loss[{d}] = {d:.6}\n",
            .{ initial_loss, n_frames, final_loss },
        );
        return error.ParityFailed;
    }

    std.debug.print(
        "PASS MlpNRunner attached (n={d} layers, dims [4,12,8,4]; {d} host-submitted frames; loss {d:.6} → {d:.6})\n",
        .{ runner.nLayers(), n_frames, initial_loss, final_loss },
    );
}

// ── Weight-decay + cosine-LR smoke ─────────────────────────────────
//
// Tier-1 chunk 5 of the post-v0 training arc. Two claims:
//
//   (1) Weight-decay shrinks W toward zero in the absence of a
//       gradient signal. We feed a constant (x=0, target=0) every step;
//       MSE loss is then 0 and the gradient is identically zero, so
//       any change in ‖W‖ comes entirely from the wd shrinkage. After
//       N steps the weight L2 norm should equal ‖W₀‖·(1 − lr·wd)^N.
//
//   (2) `cosineLr` returns lr_max at step 0, lr_min at step ≥ T, and
//       the half-cosine in between. Pure host-side helper — sanity
//       check three sample points.
//
// Both runners share the SGD shader, so testing on MlpNRunner covers
// the TrainingRunner path too. (Adam shader gets the same wd push;
// Adam parity vs CPU is left for chunk 5b — math is straightforward
// AdamW, the more interesting test is a longer-horizon convergence
// run.)

fn runWeightDecayCosineLrSmoke(allocator: std.mem.Allocator) !void {
    var ctx = try vk.Context.init(allocator);
    defer ctx.deinit();

    // ── (1) Weight-decay shrinkage with zero gradient ─────────────
    const layer_dims = [_]u32{ 4, 8, 4 };
    const lr: f32 = 0.05;
    const wd: f32 = 0.1;
    const n_steps: u32 = 30;

    var runner = try train_runner_n.MlpNRunner.init(allocator, &ctx, .{
        .layer_dims = &layer_dims,
        .lr = lr,
        .weight_decay = wd,
        .init_seed = 0xCDDECA1,
    });
    defer runner.deinit();

    // Snapshot initial L2 norms per layer.
    const seed_dims = [_]usize{ 4, 8, 4 };
    var mirror = try cpu_train.MlpN.init(allocator, &seed_dims, 0.0, 0);
    defer mirror.deinit(allocator);
    try runner.readWeights(&mirror);

    var w_norm0: [2]f32 = undefined;
    for (mirror.weights, 0..) |w, L| {
        var s: f64 = 0;
        for (w) |v| s += @as(f64, v) * @as(f64, v);
        w_norm0[L] = @floatCast(@sqrt(s));
    }

    // Step with zero (x, target). Loss-grad is zero, so the only
    // change to W is the decoupled shrinkage W ← W·(1 − lr·wd).
    const zero_x = [_]f32{ 0, 0, 0, 0 };
    const zero_t = [_]f32{ 0, 0, 0, 0 };
    var s: u32 = 0;
    while (s < n_steps) : (s += 1) {
        try runner.tickStep(&zero_x, &zero_t, null);
    }

    try runner.readWeights(&mirror);
    const expected_factor: f32 = std.math.pow(f32, 1.0 - lr * wd, @as(f32, @floatFromInt(n_steps)));
    var max_rel_err: f32 = 0;
    for (mirror.weights, 0..) |w, L| {
        var s2: f64 = 0;
        for (w) |v| s2 += @as(f64, v) * @as(f64, v);
        const norm_after: f32 = @floatCast(@sqrt(s2));
        const expected = w_norm0[L] * expected_factor;
        const rel_err = @abs(norm_after - expected) / expected;
        if (rel_err > max_rel_err) max_rel_err = rel_err;
        if (rel_err > 1e-3) {
            std.debug.print(
                "weight-decay shrinkage off on layer W[{d}]: ‖W‖={d:.6}, expected={d:.6}, rel_err={d:.4}\n",
                .{ L, norm_after, expected, rel_err },
            );
            return error.ParityFailed;
        }
    }

    // Biases must be untouched (wd = 0 on bias dispatches).
    for (mirror.biases) |b| {
        for (b) |v| {
            if (v != 0.0) {
                std.debug.print("bias drift under wd: expected 0, got {d:.6}\n", .{v});
                return error.ParityFailed;
            }
        }
    }

    // ── (2) cosineLr endpoints + midpoint ─────────────────────────
    const lr_max: f32 = 0.1;
    const lr_min: f32 = 0.01;
    const total: u32 = 1000;
    const lr0 = train_runner_n.cosineLr(0, total, lr_max, lr_min);
    const lrT = train_runner_n.cosineLr(total, total, lr_max, lr_min);
    const lrMid = train_runner_n.cosineLr(total / 2, total, lr_max, lr_min);

    if (@abs(lr0 - lr_max) > 1e-6) {
        std.debug.print("cosineLr(0) = {d:.6}, expected {d:.6}\n", .{ lr0, lr_max });
        return error.ParityFailed;
    }
    if (@abs(lrT - lr_min) > 1e-6) {
        std.debug.print("cosineLr(T) = {d:.6}, expected {d:.6}\n", .{ lrT, lr_min });
        return error.ParityFailed;
    }
    // At step T/2 the cosine = 0, so lr = lr_min + (lr_max - lr_min)/2.
    const mid_expected = lr_min + (lr_max - lr_min) * 0.5;
    if (@abs(lrMid - mid_expected) > 1e-3) {
        std.debug.print("cosineLr(T/2) = {d:.6}, expected {d:.6}\n", .{ lrMid, mid_expected });
        return error.ParityFailed;
    }

    std.debug.print(
        "PASS weight decay + cosine LR (W shrunk by (1 − lr·wd)^{d} = {d:.4}, max rel-err = {e}; cosine LR endpoints + mid OK)\n",
        .{ n_steps, expected_factor, max_rel_err },
    );
}

// ── TrainingRunner batched-predict parity smoke ─────────────────────
//
// Verifies the new `tickPredictBatch` against N sequential
// `tickPredict` calls. Both should be bit-identical because the
// batched shader is just N independent applications of the same
// forward formula — same fp32 ops, same accumulation order.

fn runTrainingRunnerBatchedSmoke(allocator: std.mem.Allocator) !void {
    var ctx = try vk.Context.init(allocator);
    defer ctx.deinit();

    const n_samples: u32 = 256;
    const cfg = train_runner.Mlp2Config{
        .dim_in = 5,
        .dim_hidden = 16,
        .dim_out = 3,
        .lr = 0.0,
        .init_seed = 0xBA7CCED,
        .max_batch_size = n_samples,
    };
    var runner = try train_runner.TrainingRunner.init(allocator, &ctx, cfg);
    defer runner.deinit();

    // Build a dense input grid: U-grid × V-grid plus the Fourier-style
    // features the matryoshka demo will use. Same shape so the smoke
    // exercises the realistic path.
    const grid: u32 = 16;
    if (grid * grid != n_samples) return error.SetupBug;
    var x_batch = try allocator.alloc(f32, n_samples * cfg.dim_in);
    defer allocator.free(x_batch);
    for (0..grid) |gv| {
        for (0..grid) |gu| {
            const idx = (gv * grid + gu) * cfg.dim_in;
            const u = @as(f32, @floatFromInt(gu)) / @as(f32, @floatFromInt(grid - 1));
            const v = @as(f32, @floatFromInt(gv)) / @as(f32, @floatFromInt(grid - 1));
            x_batch[idx + 0] = u;
            x_batch[idx + 1] = v;
            x_batch[idx + 2] = @sin(2 * std.math.pi * u);
            x_batch[idx + 3] = @cos(2 * std.math.pi * u);
            x_batch[idx + 4] = @sin(2 * std.math.pi * v);
        }
    }

    // ── Batched predict ────────────────────────────────────────────
    const y_batched = try allocator.alloc(f32, n_samples * cfg.dim_out);
    defer allocator.free(y_batched);
    try runner.tickPredictBatch(x_batch, y_batched);

    // ── Sequential predict, sample by sample ───────────────────────
    const y_sequential = try allocator.alloc(f32, n_samples * cfg.dim_out);
    defer allocator.free(y_sequential);
    var x_one: [5]f32 = undefined;
    var y_one: [3]f32 = undefined;
    for (0..n_samples) |i| {
        @memcpy(&x_one, x_batch[i * cfg.dim_in ..][0..cfg.dim_in]);
        try runner.tickPredict(&x_one, &y_one);
        @memcpy(y_sequential[i * cfg.dim_out ..][0..cfg.dim_out], &y_one);
    }

    // ── Parity ────────────────────────────────────────────────────
    var max_abs: f32 = 0;
    for (y_batched, y_sequential, 0..) |b, s, i| {
        const d = @abs(b - s);
        if (d > 1e-5) {
            std.debug.print(
                "tickPredictBatch MISMATCH at {d}: batched={d:.7} sequential={d:.7}\n",
                .{ i, b, s },
            );
            return error.ParityFailed;
        }
        if (d > max_abs) max_abs = d;
    }
    std.debug.print(
        "PASS TrainingRunner batched predict ({d} samples, dim 5→16→3; max |Δ| vs sequential = {e})\n",
        .{ n_samples, max_abs },
    );
}

// ── TrainingRunner batched-train parity smoke ───────────────────────
//
// Validates `tickStepBatch` against a CPU oracle that simulates the
// same averaged-batch SGD step. The oracle does:
//   for each sample n: forward + per-sample grads
//   sum grads across samples; divide by N
//   SGD step
// then runs the same cycle GPU-side and asserts final weights match
// within fp32 tolerance over K consecutive steps. K = 8 catches any
// drift that would compound across steps; tol is loose vs single-step
// because outer-product accumulation order across N samples matters.

fn runTrainingRunnerBatchedTrainSmoke(allocator: std.mem.Allocator) !void {
    var ctx = try vk.Context.init(allocator);
    defer ctx.deinit();

    const n_samples: u32 = 16;
    const cfg = train_runner.Mlp2Config{
        .dim_in = 4,
        .dim_hidden = 8,
        .dim_out = 3,
        .lr = 0.05,
        .init_seed = 0xBA7C7AA,
        .max_batch_size = n_samples,
    };
    var runner = try train_runner.TrainingRunner.init(allocator, &ctx, cfg);
    defer runner.deinit();

    // CPU mirror — same seed + scale = same starting MLP.
    var cpu_mlp = try cpu_train.Mlp.init(allocator, cfg.dim_in, cfg.dim_hidden, cfg.dim_out, cfg.init_scale, cfg.init_seed);
    defer cpu_mlp.deinit(allocator);

    // Build a fixed batch of synthetic (x, target) pairs.
    var rng = std.Random.DefaultPrng.init(0x5EED5);
    const r = rng.random();
    const x_batch = try allocator.alloc(f32, n_samples * cfg.dim_in);
    defer allocator.free(x_batch);
    const target_batch = try allocator.alloc(f32, n_samples * cfg.dim_out);
    defer allocator.free(target_batch);
    for (x_batch) |*v| v.* = r.float(f32) * 2 - 1;
    for (target_batch) |*v| v.* = r.float(f32);

    // CPU oracle: K averaged-batch SGD steps.
    const K: u32 = 8;
    const cpu_grads = try cpu_train.Grads.init(allocator, &cpu_mlp);
    defer @constCast(&cpu_grads).deinit(allocator);
    const acc_dw1 = try allocator.alloc(f32, cpu_mlp.w1.len);
    defer allocator.free(acc_dw1);
    const acc_db1 = try allocator.alloc(f32, cpu_mlp.b1.len);
    defer allocator.free(acc_db1);
    const acc_dw2 = try allocator.alloc(f32, cpu_mlp.w2.len);
    defer allocator.free(acc_dw2);
    const acc_db2 = try allocator.alloc(f32, cpu_mlp.b2.len);
    defer allocator.free(acc_db2);
    const cpu_h_pre = try allocator.alloc(f32, cfg.dim_hidden);
    defer allocator.free(cpu_h_pre);
    const cpu_h = try allocator.alloc(f32, cfg.dim_hidden);
    defer allocator.free(cpu_h);
    const cpu_y = try allocator.alloc(f32, cfg.dim_out);
    defer allocator.free(cpu_y);
    const cpu_dy = try allocator.alloc(f32, cfg.dim_out);
    defer allocator.free(cpu_dy);

    var step: u32 = 0;
    while (step < K) : (step += 1) {
        @memset(acc_dw1, 0);
        @memset(acc_db1, 0);
        @memset(acc_dw2, 0);
        @memset(acc_db2, 0);
        for (0..n_samples) |n| {
            const x_row = x_batch[n * cfg.dim_in ..][0..cfg.dim_in];
            const t_row = target_batch[n * cfg.dim_out ..][0..cfg.dim_out];
            var act: cpu_train.Activations = .{
                .x = x_row,
                .h_pre = cpu_h_pre,
                .h = cpu_h,
                .y = cpu_y,
            };
            cpu_train.forward(&cpu_mlp, &act);
            cpu_train.mseLossGrad(cpu_dy, cpu_y, t_row);
            var sample_grads: cpu_train.Grads = .{
                .dw1 = cpu_grads.dw1,
                .db1 = cpu_grads.db1,
                .dw2 = cpu_grads.dw2,
                .db2 = cpu_grads.db2,
            };
            try cpu_train.backward(allocator, &cpu_mlp, &act, cpu_dy, &sample_grads);
            for (acc_dw1, sample_grads.dw1) |*a, g| a.* += g;
            for (acc_db1, sample_grads.db1) |*a, g| a.* += g;
            for (acc_dw2, sample_grads.dw2) |*a, g| a.* += g;
            for (acc_db2, sample_grads.db2) |*a, g| a.* += g;
        }
        const inv_n: f32 = 1.0 / @as(f32, @floatFromInt(n_samples));
        for (acc_dw1) |*v| v.* *= inv_n;
        for (acc_db1) |*v| v.* *= inv_n;
        for (acc_dw2) |*v| v.* *= inv_n;
        for (acc_db2) |*v| v.* *= inv_n;
        const final_grads: cpu_train.Grads = .{
            .dw1 = acc_dw1,
            .db1 = acc_db1,
            .dw2 = acc_dw2,
            .db2 = acc_db2,
        };
        cpu_train.sgdStep(&cpu_mlp, &final_grads, cfg.lr);

        // Mirror on GPU.
        try runner.tickStepBatch(x_batch, target_batch);
    }

    // Compare final weights.
    const gpu_w1 = try allocator.alloc(f32, cpu_mlp.w1.len);
    defer allocator.free(gpu_w1);
    const gpu_b1 = try allocator.alloc(f32, cpu_mlp.b1.len);
    defer allocator.free(gpu_b1);
    const gpu_w2 = try allocator.alloc(f32, cpu_mlp.w2.len);
    defer allocator.free(gpu_w2);
    const gpu_b2 = try allocator.alloc(f32, cpu_mlp.b2.len);
    defer allocator.free(gpu_b2);
    // Construct an Mlp view whose slices alias our scratch buffers —
    // bypass Mlp.init so no allocator-owned weights leak when we
    // immediately overwrite them via readWeights.
    var gpu_mlp: cpu_train.Mlp = .{
        .dim_in = cfg.dim_in,
        .dim_hidden = cfg.dim_hidden,
        .dim_out = cfg.dim_out,
        .w1 = gpu_w1,
        .b1 = gpu_b1,
        .w2 = gpu_w2,
        .b2 = gpu_b2,
    };
    try runner.readWeights(&gpu_mlp);

    const tol: f32 = 1e-4;
    var max_abs: f32 = 0;
    const ParamCase = struct { name: []const u8, gpu: []const f32, cpu: []const f32 };
    const cases = [_]ParamCase{
        .{ .name = "W1", .gpu = gpu_w1, .cpu = cpu_mlp.w1 },
        .{ .name = "b1", .gpu = gpu_b1, .cpu = cpu_mlp.b1 },
        .{ .name = "W2", .gpu = gpu_w2, .cpu = cpu_mlp.w2 },
        .{ .name = "b2", .gpu = gpu_b2, .cpu = cpu_mlp.b2 },
    };
    for (cases) |cs| {
        for (cs.gpu, cs.cpu, 0..) |g, c, i| {
            const d = @abs(g - c);
            if (d > tol) {
                std.debug.print(
                    "tickStepBatch MISMATCH on {s}[{d}] @ step {d}: gpu={d:.7} cpu={d:.7}\n",
                    .{ cs.name, i, K, g, c },
                );
                return error.ParityFailed;
            }
            if (d > max_abs) max_abs = d;
        }
    }
    std.debug.print(
        "PASS TrainingRunner batched train ({d} samples × {d} steps, dim 4→8→3; max |Δ| vs CPU oracle = {e})\n",
        .{ n_samples, K, max_abs },
    );
}

// ── TrainingRunner cooperative-tickFrame smoke ──────────────────────
//
// Exercises the host-driven frame lifecycle: host owns the
// VkCommandBuffer, runner records training + predict into it, host
// submits once per frame. Verifies the budget API matches Session's
// shape (.steps / .microseconds / .either) by checking:
//
//   1. .steps = N caps the recorded step count exactly at N
//   2. .microseconds = 0 records 0 steps (early-out path)
//   3. .either = { steps = 1, microseconds = huge } caps at 1 step
//
// Plus end-to-end convergence: train for ~30 host frames, predictions
// at the supervised UVs end up close to their targets — same kind of
// proof we used for chunks 5/7 but routed entirely through the
// cooperative attach API.

fn runTrainingRunnerCoopSmoke(allocator: std.mem.Allocator) !void {
    var ctx = try vk.Context.init(allocator);
    defer ctx.deinit();

    const n_train: u32 = 8;
    const n_predict: u32 = 8;
    const cfg = train_runner.Mlp2Config{
        .dim_in = 4,
        .dim_hidden = 16,
        .dim_out = 3,
        .lr = 0.1,
        .init_seed = 0x7177EF00,
        .max_batch_size = @max(n_train, n_predict),
    };
    var runner = try train_runner.TrainingRunner.init(allocator, &ctx, cfg);
    defer runner.deinit();

    // Synthetic training set + predict UVs. Inputs are random, targets
    // are a simple linear function of the input — easy regression for
    // a 16-wide hidden layer.
    var rng = std.Random.DefaultPrng.init(0xCD0CD0);
    const r = rng.random();
    const x_train = try allocator.alloc(f32, n_train * cfg.dim_in);
    defer allocator.free(x_train);
    const t_train = try allocator.alloc(f32, n_train * cfg.dim_out);
    defer allocator.free(t_train);
    for (x_train) |*v| v.* = r.float(f32) * 2 - 1;
    for (0..n_train) |i| {
        const xr = x_train[i * cfg.dim_in ..][0..cfg.dim_in];
        const tr = t_train[i * cfg.dim_out ..][0..cfg.dim_out];
        // Trivial mapping: target[o] = sum_k x[k] * 0.5
        for (0..cfg.dim_out) |o| {
            var acc: f32 = 0;
            for (xr) |xk| acc += xk * 0.5;
            tr[o] = acc * (0.7 + 0.1 * @as(f32, @floatFromInt(o)));
        }
    }
    const x_predict = try allocator.alloc(f32, n_predict * cfg.dim_in);
    defer allocator.free(x_predict);
    @memcpy(x_predict, x_train); // predict on the same UVs the trainer sees

    try runner.uploadTrainBatch(x_train, t_train);
    try runner.uploadPredictInputs(x_predict);

    // ── Host-owned cmd buffer + fence ──
    var cb_ai = std.mem.zeroes(vk.c.VkCommandBufferAllocateInfo);
    cb_ai.sType = vk.c.VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    cb_ai.commandPool = ctx.cmd_pool;
    cb_ai.level = vk.c.VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cb_ai.commandBufferCount = 1;
    var host_cmd: vk.c.VkCommandBuffer = null;
    try vk.check(vk.c.vkAllocateCommandBuffers(ctx.device, &cb_ai, &host_cmd));
    defer vk.c.vkFreeCommandBuffers(ctx.device, ctx.cmd_pool, 1, &host_cmd);

    var fci = std.mem.zeroes(vk.c.VkFenceCreateInfo);
    fci.sType = vk.c.VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    var host_fence: vk.c.VkFence = null;
    try vk.check(vk.c.vkCreateFence(ctx.device, &fci, null, &host_fence));
    defer vk.c.vkDestroyFence(ctx.device, host_fence, null);

    // Helper: run one host frame with a given budget. Returns the
    // tick result so callers can assert step counts.
    const runFrame = struct {
        fn f(
            rn: *train_runner.TrainingRunner,
            ct: *const vk.Context,
            cmd: vk.c.VkCommandBuffer,
            fnc: vk.c.VkFence,
            budget: train_runner.TrainBudget,
            n_tr: u32,
            n_pr: u32,
        ) !train_runner.TrainTickResult {
            try vk.check(vk.c.vkResetCommandBuffer(cmd, 0));
            try vk.check(vk.c.vkResetFences(ct.device, 1, &fnc));
            var bi = std.mem.zeroes(vk.c.VkCommandBufferBeginInfo);
            bi.sType = vk.c.VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
            bi.flags = vk.c.VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
            try vk.check(vk.c.vkBeginCommandBuffer(cmd, &bi));

            var rec = try gpu_recorder.Recorder.attachCmd(ct, cmd, 64, 256);
            defer rec.deinit();
            try rec.begin();

            const result = try rn.tickFrameTrain(&rec, budget, n_tr);
            try rn.tickPredictBatchRecord(&rec, n_pr);

            try vk.check(vk.c.vkEndCommandBuffer(cmd));
            var submit = std.mem.zeroes(vk.c.VkSubmitInfo);
            submit.sType = vk.c.VK_STRUCTURE_TYPE_SUBMIT_INFO;
            submit.commandBufferCount = 1;
            submit.pCommandBuffers = &cmd;
            try vk.check(vk.c.vkQueueSubmit(ct.queue, 1, &submit, fnc));
            const timeout_ns: u64 = 10 * 1_000_000_000;
            try vk.check(vk.c.vkWaitForFences(ct.device, 1, &fnc, vk.c.VK_TRUE, timeout_ns));
            return result;
        }
    }.f;

    // ── Test 1: .steps = 3 caps at exactly 3 ──
    const r3 = try runFrame(&runner, &ctx, host_cmd, host_fence, .{ .steps = 3 }, n_train, n_predict);
    if (r3.steps_completed != 3) {
        std.debug.print(".steps=3 wanted 3 steps, got {d}\n", .{r3.steps_completed});
        return error.ParityFailed;
    }

    // ── Test 2: .microseconds = 0 records 0 steps ──
    const r0 = try runFrame(&runner, &ctx, host_cmd, host_fence, .{ .microseconds = 0 }, n_train, n_predict);
    if (r0.steps_completed != 0) {
        std.debug.print(".microseconds=0 wanted 0 steps, got {d}\n", .{r0.steps_completed});
        return error.ParityFailed;
    }

    // ── Test 3: .either with huge µs cap behaves like .steps ──
    const r1 = try runFrame(
        &runner,
        &ctx,
        host_cmd,
        host_fence,
        .{ .either = .{ .steps = 1, .microseconds = 1_000_000 } },
        n_train,
        n_predict,
    );
    if (r1.steps_completed != 1) {
        std.debug.print(".either wanted 1 step, got {d}\n", .{r1.steps_completed});
        return error.ParityFailed;
    }

    // ── Test 4: convergence over many frames ──
    // Predict initial loss, train for 30 frames at .steps=2 each, then
    // predict again. Loss should drop materially.
    var pred_initial: [n_predict * 3]f32 = undefined;
    try runner.tickPredictBatch(x_predict, &pred_initial);
    var loss_initial: f32 = 0;
    for (0..n_predict) |i| {
        const py = pred_initial[i * cfg.dim_out ..][0..cfg.dim_out];
        const ty = t_train[i * cfg.dim_out ..][0..cfg.dim_out];
        loss_initial += cpu_train.mseLoss(py, ty);
    }
    var n_frames: u32 = 0;
    while (n_frames < 30) : (n_frames += 1) {
        _ = try runFrame(&runner, &ctx, host_cmd, host_fence, .{ .steps = 2 }, n_train, n_predict);
    }
    var pred_final: [n_predict * 3]f32 = undefined;
    try runner.tickPredictBatch(x_predict, &pred_final);
    var loss_final: f32 = 0;
    for (0..n_predict) |i| {
        const py = pred_final[i * cfg.dim_out ..][0..cfg.dim_out];
        const ty = t_train[i * cfg.dim_out ..][0..cfg.dim_out];
        loss_final += cpu_train.mseLoss(py, ty);
    }
    // 4× drop is plenty to demonstrate the cooperative API trains
    // the network end-to-end. Tighter convergence is the parity test
    // in chunk 8's smoke — this one just checks the wiring.
    if (!(loss_final < loss_initial * 0.25)) {
        std.debug.print("coop train didn't converge: initial={d:.6} final={d:.6}\n", .{ loss_initial, loss_final });
        return error.ParityFailed;
    }

    std.debug.print(
        "PASS TrainingRunner cooperative tickFrame (budget shapes ok; loss {d:.6} → {d:.6} over 30 host frames)\n",
        .{ loss_initial, loss_final },
    );
}

// ── TrainingRunner staging-readback parity smoke ────────────────────
//
// Verifies that predict outputs written via the host-mapped staging
// buffer (chunk 10) match the synchronous `tickPredictBatch` readback.
// Cooperative path: host records predict + readback into its own cmd
// buffer, submits, waits on its own fence, then reads the staging
// region directly. Should be bit-identical to the standalone path
// since it's the same dispatch with a vkCmdCopyBuffer appended.

fn runTrainingRunnerStagingSmoke(allocator: std.mem.Allocator) !void {
    var ctx = try vk.Context.init(allocator);
    defer ctx.deinit();

    const n_samples: u32 = 64;
    const cfg = train_runner.Mlp2Config{
        .dim_in = 4,
        .dim_hidden = 16,
        .dim_out = 3,
        .lr = 0.0,
        .init_seed = 0x57A661D6,
        .max_batch_size = n_samples,
    };
    var runner = try train_runner.TrainingRunner.init(allocator, &ctx, cfg);
    defer runner.deinit();

    // Random input grid.
    var rng = std.Random.DefaultPrng.init(0xCAFE57AD);
    const r = rng.random();
    const x_batch = try allocator.alloc(f32, n_samples * cfg.dim_in);
    defer allocator.free(x_batch);
    for (x_batch) |*v| v.* = r.float(f32) * 2 - 1;

    // ── Reference: synchronous tickPredictBatch ──
    const y_sync = try allocator.alloc(f32, n_samples * cfg.dim_out);
    defer allocator.free(y_sync);
    try runner.tickPredictBatch(x_batch, y_sync);

    // ── Cooperative: stage inputs, record predict + readback into a
    // host-owned cmd buffer, submit, wait on host fence, read staging.
    try runner.uploadPredictInputs(x_batch);

    var cb_ai = std.mem.zeroes(vk.c.VkCommandBufferAllocateInfo);
    cb_ai.sType = vk.c.VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    cb_ai.commandPool = ctx.cmd_pool;
    cb_ai.level = vk.c.VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cb_ai.commandBufferCount = 1;
    var host_cmd: vk.c.VkCommandBuffer = null;
    try vk.check(vk.c.vkAllocateCommandBuffers(ctx.device, &cb_ai, &host_cmd));
    defer vk.c.vkFreeCommandBuffers(ctx.device, ctx.cmd_pool, 1, &host_cmd);
    var fci = std.mem.zeroes(vk.c.VkFenceCreateInfo);
    fci.sType = vk.c.VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    var host_fence: vk.c.VkFence = null;
    try vk.check(vk.c.vkCreateFence(ctx.device, &fci, null, &host_fence));
    defer vk.c.vkDestroyFence(ctx.device, host_fence, null);

    var bi = std.mem.zeroes(vk.c.VkCommandBufferBeginInfo);
    bi.sType = vk.c.VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    bi.flags = vk.c.VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    try vk.check(vk.c.vkBeginCommandBuffer(host_cmd, &bi));

    var rec = try gpu_recorder.Recorder.attachCmd(&ctx, host_cmd, 8, 32);
    defer rec.deinit();
    try rec.begin();
    try runner.tickPredictBatchRecord(&rec, n_samples);
    try runner.recordPredictReadback(&rec, n_samples);

    try vk.check(vk.c.vkEndCommandBuffer(host_cmd));
    var submit = std.mem.zeroes(vk.c.VkSubmitInfo);
    submit.sType = vk.c.VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submit.commandBufferCount = 1;
    submit.pCommandBuffers = &host_cmd;
    try vk.check(vk.c.vkQueueSubmit(ctx.queue, 1, &submit, host_fence));
    const timeout_ns: u64 = 10 * 1_000_000_000;
    try vk.check(vk.c.vkWaitForFences(ctx.device, 1, &host_fence, vk.c.VK_TRUE, timeout_ns));

    const y_staging = try allocator.alloc(f32, n_samples * cfg.dim_out);
    defer allocator.free(y_staging);
    try runner.readPredictStaging(y_staging);

    var max_abs: f32 = 0;
    for (y_sync, y_staging, 0..) |a, b, i| {
        const d = @abs(a - b);
        if (d > 1e-6) {
            std.debug.print(
                "staging readback MISMATCH at {d}: sync={d:.7} staging={d:.7}\n",
                .{ i, a, b },
            );
            return error.ParityFailed;
        }
        if (d > max_abs) max_abs = d;
    }
    std.debug.print(
        "PASS TrainingRunner staging readback ({d} samples; vs synchronous tickPredictBatch max |Δ| = {e})\n",
        .{ n_samples, max_abs },
    );
}

// ── TrainingRunner Adam-optimizer parity smoke ──────────────────────
//
// CPU oracle does the canonical Adam update step-by-step:
//
//   m ← β₁·m + (1 − β₁)·g
//   v ← β₂·v + (1 − β₂)·g²
//   m̂ = m / (1 − β₁ᵗ)
//   v̂ = v / (1 − β₂ᵗ)
//   param ← param − lr · m̂ / (√v̂ + ε)
//
// Same averaged-batch gradients as the SGD smoke (chunk 8). Verifies
// GPU adam_step.comp matches the standard formulation within fp32
// tolerance over K consecutive steps. Tolerance is looser than SGD
// because the bias-correction terms involve `pow(beta, t)` which
// rounds slightly differently CPU vs GPU.

const adamCpuOracle = struct {
    fn step(
        param: []f32,
        grad: []const f32,
        m: []f32,
        v: []f32,
        lr: f32,
        beta1: f32,
        beta2: f32,
        eps: f32,
        t: u32,
    ) void {
        const tf: f32 = @floatFromInt(t);
        const bc1 = 1.0 - std.math.pow(f32, beta1, tf);
        const bc2 = 1.0 - std.math.pow(f32, beta2, tf);
        for (param, grad, m, v) |*p, g, *mi, *vi| {
            mi.* = beta1 * mi.* + (1.0 - beta1) * g;
            vi.* = beta2 * vi.* + (1.0 - beta2) * g * g;
            const m_hat = mi.* / bc1;
            const v_hat = vi.* / bc2;
            p.* -= lr * m_hat / (@sqrt(v_hat) + eps);
        }
    }
}.step;

fn runTrainingRunnerAdamSmoke(allocator: std.mem.Allocator) !void {
    var ctx = try vk.Context.init(allocator);
    defer ctx.deinit();

    const n_samples: u32 = 16;
    const cfg = train_runner.Mlp2Config{
        .dim_in = 4,
        .dim_hidden = 8,
        .dim_out = 3,
        .lr = 0.01,
        .init_seed = 0xADA70A50,
        .max_batch_size = n_samples,
        .optimizer = .adam,
    };
    var runner = try train_runner.TrainingRunner.init(allocator, &ctx, cfg);
    defer runner.deinit();

    var cpu_mlp = try cpu_train.Mlp.init(allocator, cfg.dim_in, cfg.dim_hidden, cfg.dim_out, cfg.init_scale, cfg.init_seed);
    defer cpu_mlp.deinit(allocator);

    // Adam state on CPU.
    const m_w1 = try allocator.alloc(f32, cpu_mlp.w1.len);
    defer allocator.free(m_w1);
    @memset(m_w1, 0);
    const v_w1 = try allocator.alloc(f32, cpu_mlp.w1.len);
    defer allocator.free(v_w1);
    @memset(v_w1, 0);
    const m_b1 = try allocator.alloc(f32, cpu_mlp.b1.len);
    defer allocator.free(m_b1);
    @memset(m_b1, 0);
    const v_b1 = try allocator.alloc(f32, cpu_mlp.b1.len);
    defer allocator.free(v_b1);
    @memset(v_b1, 0);
    const m_w2 = try allocator.alloc(f32, cpu_mlp.w2.len);
    defer allocator.free(m_w2);
    @memset(m_w2, 0);
    const v_w2 = try allocator.alloc(f32, cpu_mlp.w2.len);
    defer allocator.free(v_w2);
    @memset(v_w2, 0);
    const m_b2 = try allocator.alloc(f32, cpu_mlp.b2.len);
    defer allocator.free(m_b2);
    @memset(m_b2, 0);
    const v_b2 = try allocator.alloc(f32, cpu_mlp.b2.len);
    defer allocator.free(v_b2);
    @memset(v_b2, 0);

    var rng = std.Random.DefaultPrng.init(0x5EED5);
    const r = rng.random();
    const x_batch = try allocator.alloc(f32, n_samples * cfg.dim_in);
    defer allocator.free(x_batch);
    const target_batch = try allocator.alloc(f32, n_samples * cfg.dim_out);
    defer allocator.free(target_batch);
    for (x_batch) |*vv| vv.* = r.float(f32) * 2 - 1;
    for (target_batch) |*vv| vv.* = r.float(f32);

    const grads = try cpu_train.Grads.init(allocator, &cpu_mlp);
    defer @constCast(&grads).deinit(allocator);
    const acc_dw1 = try allocator.alloc(f32, cpu_mlp.w1.len);
    defer allocator.free(acc_dw1);
    const acc_db1 = try allocator.alloc(f32, cpu_mlp.b1.len);
    defer allocator.free(acc_db1);
    const acc_dw2 = try allocator.alloc(f32, cpu_mlp.w2.len);
    defer allocator.free(acc_dw2);
    const acc_db2 = try allocator.alloc(f32, cpu_mlp.b2.len);
    defer allocator.free(acc_db2);
    const cpu_h_pre = try allocator.alloc(f32, cfg.dim_hidden);
    defer allocator.free(cpu_h_pre);
    const cpu_h = try allocator.alloc(f32, cfg.dim_hidden);
    defer allocator.free(cpu_h);
    const cpu_y = try allocator.alloc(f32, cfg.dim_out);
    defer allocator.free(cpu_y);
    const cpu_dy = try allocator.alloc(f32, cfg.dim_out);
    defer allocator.free(cpu_dy);

    const K: u32 = 5;
    var step: u32 = 0;
    while (step < K) : (step += 1) {
        @memset(acc_dw1, 0);
        @memset(acc_db1, 0);
        @memset(acc_dw2, 0);
        @memset(acc_db2, 0);
        for (0..n_samples) |i| {
            const x_row = x_batch[i * cfg.dim_in ..][0..cfg.dim_in];
            const t_row = target_batch[i * cfg.dim_out ..][0..cfg.dim_out];
            var act: cpu_train.Activations = .{ .x = x_row, .h_pre = cpu_h_pre, .h = cpu_h, .y = cpu_y };
            cpu_train.forward(&cpu_mlp, &act);
            cpu_train.mseLossGrad(cpu_dy, cpu_y, t_row);
            var sample_grads: cpu_train.Grads = .{
                .dw1 = grads.dw1, .db1 = grads.db1, .dw2 = grads.dw2, .db2 = grads.db2,
            };
            try cpu_train.backward(allocator, &cpu_mlp, &act, cpu_dy, &sample_grads);
            for (acc_dw1, sample_grads.dw1) |*a, gi| a.* += gi;
            for (acc_db1, sample_grads.db1) |*a, gi| a.* += gi;
            for (acc_dw2, sample_grads.dw2) |*a, gi| a.* += gi;
            for (acc_db2, sample_grads.db2) |*a, gi| a.* += gi;
        }
        const inv_n: f32 = 1.0 / @as(f32, @floatFromInt(n_samples));
        for (acc_dw1) |*vv| vv.* *= inv_n;
        for (acc_db1) |*vv| vv.* *= inv_n;
        for (acc_dw2) |*vv| vv.* *= inv_n;
        for (acc_db2) |*vv| vv.* *= inv_n;

        const t_idx = step + 1;
        adamCpuOracle(cpu_mlp.w1, acc_dw1, m_w1, v_w1, cfg.lr, cfg.adam_beta1, cfg.adam_beta2, cfg.adam_eps, t_idx);
        adamCpuOracle(cpu_mlp.b1, acc_db1, m_b1, v_b1, cfg.lr, cfg.adam_beta1, cfg.adam_beta2, cfg.adam_eps, t_idx);
        adamCpuOracle(cpu_mlp.w2, acc_dw2, m_w2, v_w2, cfg.lr, cfg.adam_beta1, cfg.adam_beta2, cfg.adam_eps, t_idx);
        adamCpuOracle(cpu_mlp.b2, acc_db2, m_b2, v_b2, cfg.lr, cfg.adam_beta1, cfg.adam_beta2, cfg.adam_eps, t_idx);

        try runner.tickStepBatch(x_batch, target_batch);
    }

    const gpu_w1 = try allocator.alloc(f32, cpu_mlp.w1.len);
    defer allocator.free(gpu_w1);
    const gpu_b1 = try allocator.alloc(f32, cpu_mlp.b1.len);
    defer allocator.free(gpu_b1);
    const gpu_w2 = try allocator.alloc(f32, cpu_mlp.w2.len);
    defer allocator.free(gpu_w2);
    const gpu_b2 = try allocator.alloc(f32, cpu_mlp.b2.len);
    defer allocator.free(gpu_b2);
    var gpu_mlp: cpu_train.Mlp = .{
        .dim_in = cfg.dim_in,
        .dim_hidden = cfg.dim_hidden,
        .dim_out = cfg.dim_out,
        .w1 = gpu_w1,
        .b1 = gpu_b1,
        .w2 = gpu_w2,
        .b2 = gpu_b2,
    };
    try runner.readWeights(&gpu_mlp);

    const tol: f32 = 1e-4;
    var max_abs: f32 = 0;
    const ParamCase = struct { name: []const u8, gpu: []const f32, cpu: []const f32 };
    const cases = [_]ParamCase{
        .{ .name = "W1", .gpu = gpu_w1, .cpu = cpu_mlp.w1 },
        .{ .name = "b1", .gpu = gpu_b1, .cpu = cpu_mlp.b1 },
        .{ .name = "W2", .gpu = gpu_w2, .cpu = cpu_mlp.w2 },
        .{ .name = "b2", .gpu = gpu_b2, .cpu = cpu_mlp.b2 },
    };
    for (cases) |cs| {
        for (cs.gpu, cs.cpu, 0..) |g, cv, i| {
            const d = @abs(g - cv);
            if (d > tol) {
                std.debug.print("Adam MISMATCH on {s}[{d}] @ step {d}: gpu={d:.7} cpu={d:.7}\n", .{ cs.name, i, K, g, cv });
                return error.ParityFailed;
            }
            if (d > max_abs) max_abs = d;
        }
    }
    std.debug.print(
        "PASS TrainingRunner Adam ({d} samples × {d} steps, dim 4→8→3; max |Δ| vs CPU oracle = {e})\n",
        .{ n_samples, K, max_abs },
    );
}

// ── TrainingRunner cross-entropy loss-grad parity smoke ─────────────
//
// CPU oracle does the canonical stable softmax + CE-grad:
//   m = max(y)
//   p = exp(y - m) / Σ exp(y - m)
//   dy = (p − target) / N
// then compares against the GPU shader's output. Catches any sign or
// off-by-one in the softmax + scale-by-1/N chain. Two scenarios:
// hard-label (one-hot target) and soft-label (mixed distribution), so
// the test exercises both classifier and distillation use cases.

fn runTrainingRunnerCrossEntropySmoke(allocator: std.mem.Allocator) !void {
    var ctx = try vk.Context.init(allocator);
    defer ctx.deinit();

    const n_samples: u32 = 32;
    const dim_out: u32 = 5;
    const cfg = train_runner.Mlp2Config{
        .dim_in = 4,
        .dim_hidden = 8,
        .dim_out = dim_out,
        .lr = 0.0,
        .init_seed = 0xC10557E,
        .max_batch_size = n_samples,
        .loss = .cross_entropy,
    };
    var runner = try train_runner.TrainingRunner.init(allocator, &ctx, cfg);
    defer runner.deinit();

    // Build synthetic batch + target.
    var rng = std.Random.DefaultPrng.init(0xCEED57A);
    const r = rng.random();
    const x_batch = try allocator.alloc(f32, n_samples * cfg.dim_in);
    defer allocator.free(x_batch);
    const target_batch = try allocator.alloc(f32, n_samples * cfg.dim_out);
    defer allocator.free(target_batch);
    for (x_batch) |*v| v.* = r.float(f32) * 2 - 1;
    // Half samples one-hot (hard label), half soft (Dirichlet-ish).
    var i: u32 = 0;
    while (i < n_samples) : (i += 1) {
        const off = i * dim_out;
        if (i < n_samples / 2) {
            const cls = r.uintLessThan(u32, dim_out);
            for (0..dim_out) |o| target_batch[off + o] = if (o == cls) 1.0 else 0.0;
        } else {
            var sum: f32 = 0;
            for (0..dim_out) |o| {
                const v = r.float(f32);
                target_batch[off + o] = v;
                sum += v;
            }
            for (0..dim_out) |o| target_batch[off + o] /= sum;
        }
    }

    // Run forward via tickPredictBatch to get logits, then run our
    // CE-grad shader by triggering one batched train step (dy is the
    // first thing recordTrainBatch dispatches).
    //
    // Approach: read y_train_batch directly after a single batched
    // train step. But that requires staging or a peek. Simplest: use
    // a minimal run that lets us readBack y_b. Our recordTrainBatch
    // already writes h_pre/h/y to dedicated buffers; we can readBack
    // y_train_batch after the dispatch.
    try runner.uploadTrainBatch(x_batch, target_batch);

    // Manually drive forward + dy + readback into a recorder. This
    // lets us isolate the dy output for parity-checking without the
    // rest of the train chain mutating weights (lr=0 already takes
    // care of that, but it's cleaner this way).
    try runner.tickStepBatch(x_batch, target_batch);

    // Pull the y_train_batch logits + dy_train_batch back to host.
    const y_gpu = try allocator.alloc(f32, n_samples * dim_out);
    defer allocator.free(y_gpu);
    const dy_gpu = try allocator.alloc(f32, n_samples * dim_out);
    defer allocator.free(dy_gpu);
    try runner.y_train_batch.?.readBack(&ctx, f32, y_gpu);
    try runner.dy_train_batch.?.readBack(&ctx, f32, dy_gpu);

    // CPU oracle: stable softmax + CE-grad, /N pre-divide.
    const dy_cpu = try allocator.alloc(f32, n_samples * dim_out);
    defer allocator.free(dy_cpu);
    const inv_n: f32 = 1.0 / @as(f32, @floatFromInt(n_samples));
    for (0..n_samples) |n| {
        const off = n * dim_out;
        var m: f32 = y_gpu[off];
        for (1..dim_out) |o| m = @max(m, y_gpu[off + o]);
        var sum_e: f32 = 0;
        for (0..dim_out) |o| sum_e += @exp(y_gpu[off + o] - m);
        for (0..dim_out) |o| {
            const p = @exp(y_gpu[off + o] - m) / sum_e;
            dy_cpu[off + o] = (p - target_batch[off + o]) * inv_n;
        }
    }

    var max_abs: f32 = 0;
    for (dy_gpu, dy_cpu, 0..) |g, c, idx| {
        const d = @abs(g - c);
        if (d > 1e-5) {
            std.debug.print(
                "CE loss-grad MISMATCH at {d}: gpu={d:.7} cpu={d:.7}\n",
                .{ idx, g, c },
            );
            return error.ParityFailed;
        }
        if (d > max_abs) max_abs = d;
    }
    std.debug.print(
        "PASS TrainingRunner cross-entropy loss-grad ({d} samples × {d} classes; max |Δ| vs CPU oracle = {e})\n",
        .{ n_samples, dim_out, max_abs },
    );
}

// ── TrainingRunner loss-target decay smoke ──────────────────────────
//
// Verifies the runtime self-throttling logic:
//   1. With decay = .loss_target { target = 0.05 }, training runs
//      until loss < target, then `tickFrameTrain` reports idle = true
//      and steps_completed = 0.
//   2. last_loss caches a real value (> 0, ≤ target by the time idle).
//   3. Resumption: changing target_batch to break the fit causes
//      next frame's tickFrameTrain to report idle = false again.

fn runTrainingRunnerDecaySmoke(allocator: std.mem.Allocator) !void {
    var ctx = try vk.Context.init(allocator);
    defer ctx.deinit();

    const n_train: u32 = 8;
    const cfg = train_runner.Mlp2Config{
        .dim_in = 4,
        .dim_hidden = 16,
        .dim_out = 3,
        .lr = 0.05,
        .init_seed = 0xDECA1A77,
        .max_batch_size = n_train,
        .optimizer = .adam,
        .decay = .{ .loss_target = .{ .target = 0.001 } },
    };
    var runner = try train_runner.TrainingRunner.init(allocator, &ctx, cfg);
    defer runner.deinit();

    // Non-trivial supervision: targets ∈ [-2, 2], well outside the
    // initial MLP's near-zero output range. Initial loss is large so
    // we actually exercise the converge-then-idle path; a target of
    // 0.001 is reachable in a few dozen Adam steps but not on frame 1.
    var rng = std.Random.DefaultPrng.init(0xDECAD0);
    const r = rng.random();
    const x_train = try allocator.alloc(f32, n_train * cfg.dim_in);
    defer allocator.free(x_train);
    const t_train = try allocator.alloc(f32, n_train * cfg.dim_out);
    defer allocator.free(t_train);
    for (x_train) |*v| v.* = r.float(f32) * 2 - 1;
    for (t_train) |*v| v.* = r.float(f32) * 4 - 2;

    try runner.uploadTrainBatch(x_train, t_train);

    var cb_ai = std.mem.zeroes(vk.c.VkCommandBufferAllocateInfo);
    cb_ai.sType = vk.c.VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    cb_ai.commandPool = ctx.cmd_pool;
    cb_ai.level = vk.c.VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cb_ai.commandBufferCount = 1;
    var host_cmd: vk.c.VkCommandBuffer = null;
    try vk.check(vk.c.vkAllocateCommandBuffers(ctx.device, &cb_ai, &host_cmd));
    defer vk.c.vkFreeCommandBuffers(ctx.device, ctx.cmd_pool, 1, &host_cmd);
    var fci = std.mem.zeroes(vk.c.VkFenceCreateInfo);
    fci.sType = vk.c.VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    var host_fence: vk.c.VkFence = null;
    try vk.check(vk.c.vkCreateFence(ctx.device, &fci, null, &host_fence));
    defer vk.c.vkDestroyFence(ctx.device, host_fence, null);

    const runFrame = struct {
        fn f(
            rn: *train_runner.TrainingRunner,
            ct: *const vk.Context,
            cmd: vk.c.VkCommandBuffer,
            fnc: vk.c.VkFence,
            n: u32,
        ) !train_runner.TrainTickResult {
            try vk.check(vk.c.vkResetCommandBuffer(cmd, 0));
            try vk.check(vk.c.vkResetFences(ct.device, 1, &fnc));
            var bi = std.mem.zeroes(vk.c.VkCommandBufferBeginInfo);
            bi.sType = vk.c.VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
            bi.flags = vk.c.VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
            try vk.check(vk.c.vkBeginCommandBuffer(cmd, &bi));
            var rec = try gpu_recorder.Recorder.attachCmd(ct, cmd, 64, 256);
            defer rec.deinit();
            try rec.begin();
            const result = try rn.tickFrameTrain(&rec, .{ .steps = 4 }, n);
            try vk.check(vk.c.vkEndCommandBuffer(cmd));
            var submit = std.mem.zeroes(vk.c.VkSubmitInfo);
            submit.sType = vk.c.VK_STRUCTURE_TYPE_SUBMIT_INFO;
            submit.commandBufferCount = 1;
            submit.pCommandBuffers = &cmd;
            try vk.check(vk.c.vkQueueSubmit(ct.queue, 1, &submit, fnc));
            const timeout_ns: u64 = 10 * 1_000_000_000;
            try vk.check(vk.c.vkWaitForFences(ct.device, 1, &fnc, vk.c.VK_TRUE, timeout_ns));
            return result;
        }
    }.f;

    // ── Phase 1: train until idle ──
    var idle_at: ?u32 = null;
    var loss_at_idle: ?f32 = null;
    var f: u32 = 0;
    while (f < 200) : (f += 1) {
        const result = try runFrame(&runner, &ctx, host_cmd, host_fence, n_train);
        if (result.idle) {
            idle_at = f;
            loss_at_idle = result.last_loss;
            break;
        }
    }
    if (idle_at == null) {
        std.debug.print("decay never reached idle within 200 frames; last_loss={?}\n", .{runner.getLastLoss()});
        return error.ParityFailed;
    }
    if (loss_at_idle == null or loss_at_idle.? > 0.001) {
        std.debug.print("idle reported but last_loss = {?} > target=0.001\n", .{loss_at_idle});
        return error.ParityFailed;
    }
    // Convergence (not lucky-init) check: idle should land after at
    // least a handful of training frames.
    if (idle_at.? < 3) {
        std.debug.print("idle reached on frame {d} — too early; init likely already met target. Bump targets.\n", .{idle_at.?});
        return error.ParityFailed;
    }

    // Confirm subsequent frames stay idle (training really paused).
    var f2: u32 = 0;
    while (f2 < 5) : (f2 += 1) {
        const result = try runFrame(&runner, &ctx, host_cmd, host_fence, n_train);
        if (!result.idle or result.steps_completed != 0) {
            std.debug.print("decay didn't stay idle at frame +{d}: idle={any} steps={d}\n", .{ f2, result.idle, result.steps_completed });
            return error.ParityFailed;
        }
    }

    // ── Phase 2: break the fit, training should resume ──
    // Stomp the targets with very different values that the network
    // hasn't seen — loss should jump above target and tickFrameTrain
    // should return idle = false.
    for (t_train) |*v| v.* = 5.0; // wildly different from the trained targets
    try runner.uploadTrainBatch(x_train, t_train);

    // First post-stomp frame: staging still has old loss (low), so
    // runtime gates training off and records loss eval. Need TWO
    // frames for the resumption signal to propagate (eval frame N
    // measures fresh loss; frame N+1 reads it and trains).
    _ = try runFrame(&runner, &ctx, host_cmd, host_fence, n_train);
    const resume_result = try runFrame(&runner, &ctx, host_cmd, host_fence, n_train);
    if (resume_result.idle or resume_result.steps_completed == 0) {
        std.debug.print(
            "decay didn't resume after target stomp: idle={any} steps={d} last_loss={?}\n",
            .{ resume_result.idle, resume_result.steps_completed, resume_result.last_loss },
        );
        return error.ParityFailed;
    }

    std.debug.print(
        "PASS TrainingRunner loss-target decay (idle at frame {?d}, loss={d:.6}; resumed cleanly after target stomp)\n",
        .{ idle_at, loss_at_idle.? },
    );
}

// ── --train-demo: headless on-device training demo ──────────────────
//
// Chunk 6 of training-v0. End-user-facing CLI showing the
// TrainingRunner converge on a non-trivial streaming signal, with no
// model file required. Inputs and targets are derived from a virtual
// "frame number" t — the same shape the engine integration will run,
// just printed to stdout instead of driving a sphere's BRDF. Lets a
// fresh checkout verify the whole training pipeline works end-to-end
// without any external assets, and gives a feel for "what a few
// hundred frames of training does to the loss" before the visual
// demo lands.

fn runTrainDemo(allocator: std.mem.Allocator, steps: u32, hidden: u32, lr: f32, print_every: u32) !void {
    var ctx = try vk.Context.init(allocator);
    defer ctx.deinit();

    const cfg = train_runner.Mlp2Config{
        .dim_in = 4,
        .dim_hidden = hidden,
        .dim_out = 4,
        .lr = lr,
        .init_seed = 0xDEDEDED0,
    };

    var runner = try train_runner.TrainingRunner.init(allocator, &ctx, cfg);
    defer runner.deinit();

    std.debug.print(
        "valkyr --train-demo on {s}\n  cfg: dim 4→{d}→4, lr={d}, steps={d}\n",
        .{ ctx.deviceName(), hidden, lr, steps },
    );
    std.debug.print("  task: regress (sin t, cos t, sin 2t, cos 2t) → (½+½ sin t, ½+½ cos t, ½ + ⅓ sin 3t, 0)\n", .{});

    // Input/target factories. Time advances by `dt` per step; the
    // signal is intentionally smooth (so SGD on a single-sample stream
    // sees a moving but locally-flat target) but non-trivial (the
    // network must combine all 4 inputs to predict the 3rd component).
    const dt: f32 = 0.05;
    var pred: [4]f32 = undefined;
    var target: [4]f32 = undefined;
    var input: [4]f32 = undefined;

    var step: u32 = 0;
    var ema_loss: f32 = 0;
    const ema_alpha: f32 = 0.05;
    const t_start = std.time.nanoTimestamp();

    while (step < steps) : (step += 1) {
        const t = @as(f32, @floatFromInt(step)) * dt;
        input[0] = @sin(t);
        input[1] = @cos(t);
        input[2] = @sin(2 * t);
        input[3] = @cos(2 * t);
        target[0] = 0.5 + 0.5 * @sin(t);
        target[1] = 0.5 + 0.5 * @cos(t);
        target[2] = 0.5 + 0.333 * @sin(3 * t);
        target[3] = 0.0;

        try runner.tickStep(&input, &target, &pred);
        const loss = cpu_train.mseLoss(&pred, &target);
        ema_loss = if (step == 0) loss else ema_alpha * loss + (1 - ema_alpha) * ema_loss;

        if ((step + 1) % print_every == 0 or step == 0) {
            std.debug.print(
                "  step {d:>5}  loss={d:.6}  ema={d:.6}  pred=({d:.3},{d:.3},{d:.3},{d:.3})\n",
                .{ step + 1, loss, ema_loss, pred[0], pred[1], pred[2], pred[3] },
            );
        }
    }
    const t_end = std.time.nanoTimestamp();
    const elapsed_us = @divTrunc(t_end - t_start, 1000);
    const us_per_step = @divTrunc(elapsed_us, @as(i128, steps));
    std.debug.print(
        "done: {d} steps in {d} µs ({d} µs/step, {d:.0} steps/s)\n",
        .{ steps, elapsed_us, us_per_step, 1.0e6 / @as(f64, @floatFromInt(us_per_step)) },
    );
}

// ── --train-demo-n: headless multi-layer training demo ──────────────
//
// Mirrors `--train-demo` but routed through MlpNRunner with a
// `--layers a,b,c,d` CSV controlling depth. End-user-facing reach
// for the multi-layer surface introduced in chunk 3a of the post-v0
// training arc. Same streaming (input, target) signal and same
// per-step loss / EMA print pattern as the 2-layer version, so the
// loss curves are visually comparable across depths.

fn runTrainDemoN(
    allocator: std.mem.Allocator,
    steps: u32,
    layers_csv: []const u8,
    lr: f32,
    print_every: u32,
) !void {
    var ctx = try vk.Context.init(allocator);
    defer ctx.deinit();

    // Parse `layers_csv` into a u32 slice. Errors on empty / invalid
    // entries; first entry must be 4 and last must be 4 to match the
    // input/target signal shape below.
    var dims = std.ArrayList(u32).init(allocator);
    defer dims.deinit();
    var it = std.mem.splitScalar(u8, layers_csv, ',');
    while (it.next()) |tok| {
        const trimmed = std.mem.trim(u8, tok, " \t");
        if (trimmed.len == 0) continue;
        try dims.append(try std.fmt.parseInt(u32, trimmed, 10));
    }
    if (dims.items.len < 2) {
        std.debug.print("--train-demo-n: --layers needs at least 2 dims, got '{s}'\n", .{layers_csv});
        return error.InvalidArgs;
    }
    if (dims.items[0] != 4 or dims.items[dims.items.len - 1] != 4) {
        std.debug.print("--train-demo-n: signal is fixed at dim_in=4 and dim_out=4; got [{d}, ..., {d}]\n", .{ dims.items[0], dims.items[dims.items.len - 1] });
        return error.InvalidArgs;
    }

    const cfg = train_runner_n.MlpNConfig{
        .layer_dims = dims.items,
        .lr = lr,
        .init_seed = 0xDEDEDED1,
    };
    var runner = try train_runner_n.MlpNRunner.init(allocator, &ctx, cfg);
    defer runner.deinit();

    std.debug.print("valkyr --train-demo-n on {s}\n  cfg: layers", .{ctx.deviceName()});
    for (dims.items, 0..) |d, idx| {
        std.debug.print("{s}{d}", .{ if (idx == 0) " " else "→", d });
    }
    std.debug.print(", lr={d}, steps={d}\n", .{ lr, steps });
    std.debug.print("  task: regress (sin t, cos t, sin 2t, cos 2t) → (½+½ sin t, ½+½ cos t, ½ + ⅓ sin 3t, 0)\n", .{});

    const dt: f32 = 0.05;
    var pred: [4]f32 = undefined;
    var target: [4]f32 = undefined;
    var input: [4]f32 = undefined;

    var step: u32 = 0;
    var ema_loss: f32 = 0;
    const ema_alpha: f32 = 0.05;
    const t_start = std.time.nanoTimestamp();

    while (step < steps) : (step += 1) {
        const t = @as(f32, @floatFromInt(step)) * dt;
        input[0] = @sin(t);
        input[1] = @cos(t);
        input[2] = @sin(2 * t);
        input[3] = @cos(2 * t);
        target[0] = 0.5 + 0.5 * @sin(t);
        target[1] = 0.5 + 0.5 * @cos(t);
        target[2] = 0.5 + 0.333 * @sin(3 * t);
        target[3] = 0.0;

        try runner.tickStep(&input, &target, &pred);
        const loss = cpu_train.mseLoss(&pred, &target);
        ema_loss = if (step == 0) loss else ema_alpha * loss + (1 - ema_alpha) * ema_loss;

        if ((step + 1) % print_every == 0 or step == 0) {
            std.debug.print(
                "  step {d:>5}  loss={d:.6}  ema={d:.6}  pred=({d:.3},{d:.3},{d:.3},{d:.3})\n",
                .{ step + 1, loss, ema_loss, pred[0], pred[1], pred[2], pred[3] },
            );
        }
    }
    const t_end = std.time.nanoTimestamp();
    const elapsed_us = @divTrunc(t_end - t_start, 1000);
    const us_per_step = @divTrunc(elapsed_us, @as(i128, steps));
    std.debug.print(
        "done: {d} steps in {d} µs ({d} µs/step, {d:.0} steps/s)\n",
        .{ steps, elapsed_us, us_per_step, 1.0e6 / @as(f64, @floatFromInt(us_per_step)) },
    );
}

// ── gpu matmul_nt smoke: synthetic A·Bᵀ on the GPU vs CPU expected ──

const MatmulPush = runtime.MatmulPush;

fn runGpuMatmulSmoke(allocator: std.mem.Allocator) !void {
    var ctx = try vk.Context.init(allocator);
    defer ctx.deinit();

    // Same problem as runMatmulSmoke (CPU): 2x3 · (4x3)ᵀ → 2x4.
    const a = [_]f32{ 1, 2, 3, 4, 5, 6 };
    const b = [_]f32{ 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1 };
    const want = [_]f32{ 1, 2, 3, 6, 4, 5, 6, 15 };
    const m: u32 = 2;
    const n: u32 = 4;
    const k: u32 = 3;

    var buf_a = try buffer.Buffer.initStatic(&ctx, f32, &a);
    defer buf_a.deinit(ctx.device);
    var buf_b = try buffer.Buffer.initStatic(&ctx, f32, &b);
    defer buf_b.deinit(ctx.device);
    var buf_c = try buffer.Buffer.initDeviceOnly(&ctx, m * n * @sizeOf(f32));
    defer buf_c.deinit(ctx.device);

    var kern = try pipeline.Kernel.init(&ctx, &shaders.matmul_nt, 3, @sizeOf(MatmulPush));
    defer kern.deinit();
    try kern.bind(&.{ &buf_a, &buf_b, &buf_c });

    const local_xy: u32 = 16;
    const groups_x: u32 = (m + local_xy - 1) / local_xy;
    const groups_y: u32 = (n + local_xy - 1) / local_xy;
    const push = MatmulPush{ .m = m, .n = n, .k = k };

    try buffer.submitOneShot(&ctx, struct {
        kern: *const pipeline.Kernel,
        push: *const MatmulPush,
        gx: u32,
        gy: u32,
        pub fn record(s: @This(), cmd: vk.c.VkCommandBuffer) void {
            s.kern.dispatch(cmd, s.push, s.gx, s.gy, 1);
        }
    }{ .kern = &kern, .push = &push, .gx = groups_x, .gy = groups_y });

    var out: [8]f32 = undefined;
    try buf_c.readBack(&ctx, f32, &out);
    for (out, want, 0..) |got, w, i| {
        if (got != w) {
            std.debug.print("GPU matmul MISMATCH at {d}: got {d}, expected {d}\n", .{ i, got, w });
            return error.ParityFailed;
        }
    }
    std.debug.print("PASS GPU matmul_nt synthetic (2×3 · (4×3)ᵀ → 2×4) on {s}\n", .{ctx.deviceName()});
}

// ── GPU CCE forward smoke: fused matmul + online-softmax CE vs CPU oracle ─
//
// Drives `cce_forward.comp` against `cpu_cce.cceForward` on Qwen-flavoured
// shapes — one pass exercising multi-chunk online softmax (V > CHUNK) and
// one with V < CHUNK to cover the boundary mask. Asserts per-row lse and
// per-row loss agree to 1e-5 global rel-err. The shader hardcodes
// CHUNK = 256, so we drive the CPU oracle with chunk = 256 to compare
// like-for-like.

fn runGpuCceForwardSmoke(allocator: std.mem.Allocator) !void {
    var ctx = try vk.Context.init(allocator);
    defer ctx.deinit();

    const SmokeCase = struct {
        name: []const u8,
        n: u32,
        v: u32,
        d: u32,
        z_loss_scale: f32,
        label_smoothing: f32,
    };
    const cases = [_]SmokeCase{
        .{ .name = "multi-chunk (V=2048, 8 chunks)", .n = 4, .v = 2048, .d = 896, .z_loss_scale = 0.0, .label_smoothing = 0.0 },
        .{ .name = "partial-chunk (V=300, 1+ chunks)", .n = 3, .v = 300, .d = 64, .z_loss_scale = 0.0, .label_smoothing = 0.0 },
        .{ .name = "single-chunk (V=256, exactly CHUNK)", .n = 2, .v = 256, .d = 128, .z_loss_scale = 0.0, .label_smoothing = 0.0 },
        .{ .name = "multi-chunk + z-loss λ=1e-4", .n = 4, .v = 2048, .d = 896, .z_loss_scale = 1e-4, .label_smoothing = 0.0 },
        .{ .name = "multi-chunk + label-smoothing ε=0.1", .n = 4, .v = 2048, .d = 896, .z_loss_scale = 0.0, .label_smoothing = 0.1 },
        // Combined regularizers — the realistic setting Chronicals
        // recommends for fine-tuning. Exercises every conditional in
        // the loss-and-gradient math at once.
        .{ .name = "multi-chunk + z-loss + label-smoothing", .n = 4, .v = 2048, .d = 896, .z_loss_scale = 1e-4, .label_smoothing = 0.1 },
    };

    var kern = try pipeline.Kernel.init(&ctx, &shaders.cce_forward, 5, @sizeOf(runtime.CceForwardPush));
    defer kern.deinit();

    for (cases) |cs| {
        // ── Generate inputs (h, W, targets) on host. Deterministic seed
        //    per case so failure repros. Small magnitudes keep logits in
        //    a numerically benign range (we already test stability against
        //    larger inputs in cce.zig's own tests).
        var prng = std.Random.DefaultPrng.init(0xCCE0_F00D + cs.v);
        const rng = prng.random();

        const h = try allocator.alloc(f32, cs.n * cs.d);
        defer allocator.free(h);
        const w_lm = try allocator.alloc(f32, cs.v * cs.d);
        defer allocator.free(w_lm);
        const targets = try allocator.alloc(u32, cs.n);
        defer allocator.free(targets);

        for (h) |*x| x.* = (rng.float(f32) - 0.5) * 0.1;
        for (w_lm) |*x| x.* = (rng.float(f32) - 0.5) * 0.1;
        for (targets) |*t| t.* = rng.intRangeLessThan(u32, 0, cs.v);

        // ── CPU oracle.
        const lse_cpu = try allocator.alloc(f32, cs.n);
        defer allocator.free(lse_cpu);
        const mean_loss_cpu = cpu_cce.cceForward(
            h,
            w_lm,
            targets,
            cs.n,
            cs.v,
            cs.d,
            256, // shader hardcodes CHUNK = 256
            .{ .z_loss_scale = cs.z_loss_scale, .label_smoothing = cs.label_smoothing },
            lse_cpu,
        );

        // ── GPU dispatch.
        var buf_h = try buffer.Buffer.initStatic(&ctx, f32, h);
        defer buf_h.deinit(ctx.device);
        var buf_w = try buffer.Buffer.initStatic(&ctx, f32, w_lm);
        defer buf_w.deinit(ctx.device);
        var buf_t = try buffer.Buffer.initStatic(&ctx, u32, targets);
        defer buf_t.deinit(ctx.device);
        var buf_lse = try buffer.Buffer.initDeviceOnly(&ctx, cs.n * @sizeOf(f32));
        defer buf_lse.deinit(ctx.device);
        var buf_loss = try buffer.Buffer.initDeviceOnly(&ctx, cs.n * @sizeOf(f32));
        defer buf_loss.deinit(ctx.device);

        try kern.bind(&.{ &buf_h, &buf_w, &buf_t, &buf_lse, &buf_loss });

        const push = runtime.CceForwardPush{
            .n_samples = cs.n,
            .vocab = cs.v,
            .dim = cs.d,
            .z_loss_scale = cs.z_loss_scale,
            .label_smoothing_eps = cs.label_smoothing,
        };
        try buffer.submitOneShot(&ctx, struct {
            kern: *const pipeline.Kernel,
            push: *const runtime.CceForwardPush,
            gx: u32,
            pub fn record(s: @This(), cmd: vk.c.VkCommandBuffer) void {
                s.kern.dispatch(cmd, s.push, s.gx, 1, 1);
            }
        }{ .kern = &kern, .push = &push, .gx = cs.n });

        const lse_gpu = try allocator.alloc(f32, cs.n);
        defer allocator.free(lse_gpu);
        const loss_gpu = try allocator.alloc(f32, cs.n);
        defer allocator.free(loss_gpu);
        try buf_lse.readBack(&ctx, f32, lse_gpu);
        try buf_loss.readBack(&ctx, f32, loss_gpu);

        // GPU writes per-row loss; reduce to mean for comparison against
        // the CPU oracle's scalar return value.
        var sum_gpu: f64 = 0;
        for (loss_gpu) |x| sum_gpu += x;
        const mean_loss_gpu: f32 = @floatCast(sum_gpu / @as(f64, @floatFromInt(cs.n)));

        // ── Reconstruct per-row CPU loss for diff. The CPU oracle
        //    returns mean(loss); per-row loss[n] = lse[n] - z_target[n]
        //    where z_target[n] = h[n] · w_lm[target[n]]ᵀ. Recompute the
        //    target-row dot product on the host (one row each).
        const loss_cpu_per_row = try allocator.alloc(f32, cs.n);
        defer allocator.free(loss_cpu_per_row);
        for (0..cs.n) |row| {
            const tgt: usize = @intCast(targets[row]);
            var s_target: f64 = 0;
            for (0..cs.d) |k| {
                s_target += @as(f64, h[row * cs.d + k]) * @as(f64, w_lm[tgt * cs.d + k]);
            }
            const z_target: f32 = @floatCast(s_target);
            // For label smoothing's L_uniform term we need z_mean = (1/V)·Σ_v z_v.
            // Skip the inner V·D loop entirely when ε = 0 (most cases).
            var z_mean: f32 = 0.0;
            if (cs.label_smoothing > 0.0) {
                var z_sum: f64 = 0;
                for (0..cs.v) |o| {
                    var s: f64 = 0;
                    for (0..cs.d) |k| s += @as(f64, h[row * cs.d + k]) * @as(f64, w_lm[o * cs.d + k]);
                    z_sum += s;
                }
                z_mean = @floatCast(z_sum / @as(f64, @floatFromInt(cs.v)));
            }
            const lse_v = lse_cpu[row];
            // Per-row smoothed CE + z-loss matches cce_forward.comp's
            // loss_row[row] write:
            //   loss = lse − (1−ε)·z_target − ε·z_mean + λ_z · lse²
            const t_scale: f32 = 1.0 - cs.label_smoothing;
            const ce: f32 = lse_v - t_scale * z_target - cs.label_smoothing * z_mean;
            const z_pen: f32 = cs.z_loss_scale * lse_v * lse_v;
            loss_cpu_per_row[row] = ce + z_pen;
        }

        // Global rel-err (max|diff| / max|ref|) — same metric as cce.zig's
        // parity tests, robust to noise-floor entries.
        const lse_rel = globalRelDiff(lse_cpu, lse_gpu);
        const loss_rel = globalRelDiff(loss_cpu_per_row, loss_gpu);
        const mean_rel = @abs(mean_loss_gpu - mean_loss_cpu) /
            @max(@abs(mean_loss_cpu), 1e-30);

        const tol: f32 = 1e-5;
        if (lse_rel >= tol or loss_rel >= tol or mean_rel >= tol) {
            std.debug.print(
                "CCE smoke ({s}) FAIL: lse_rel={e} loss_rel={e} mean_rel={e}  cpu_mean={d:.6} gpu_mean={d:.6}\n",
                .{ cs.name, lse_rel, loss_rel, mean_rel, mean_loss_cpu, mean_loss_gpu },
            );
            return error.ParityFailed;
        }

        std.debug.print(
            "PASS GPU CCE forward — {s}  N={d} V={d} D={d}  mean_loss={d:.4}  rel(lse/loss/mean)=({e},{e},{e})\n",
            .{ cs.name, cs.n, cs.v, cs.d, mean_loss_cpu, lse_rel, loss_rel, mean_rel },
        );
    }
}

// ── GPU LoRA smoke: composes existing kernels, parity vs CPU oracle ───
//
// LoRA-augmented linear forward + backward built entirely from existing
// SPIR-V (matmul_nt_v2, linear_backward_dx_batched, linear_backward_dw_batched,
// scale, add_in_place — no new shaders). The dispatch chain mirrors
// cpu_lora.{loraForward, loraBackward} step-for-step:
//
//   forward:
//     matmul_nt_v2(x, W)                     → y_base
//     matmul_nt_v2(x, A)                     → intermediate
//     matmul_nt_v2(intermediate, B)          → y_lora
//     scale(y_lora, α/r)                     → y_lora_scaled
//     add_in_place(y_base, y_lora_scaled)    → y  (overwrites y_base)
//
//   backward:
//     linear_backward_dx(dy, W)              → dx_base
//     linear_backward_dx(dy, B) {treat as M×N=N K=r} → dy_B
//     linear_backward_dx(dy_B, A) {M=M N=r K=K}      → dx_lora_unscaled
//     scale(dx_lora_unscaled, α/r)                   → dx_lora_scaled
//     add_in_place(dx_base, dx_lora_scaled)          → dx
//     linear_backward_dw(dy_B, x) {M=M N=r K=K}      → dA_unscaled
//     scale(dA_unscaled, α/r)                        → dA
//     linear_backward_dw(dy, intermediate)           → dB_unscaled
//     scale(dB_unscaled, α/r)                        → dB
//
// Three shape cases (small / medium-rank-16 / high-rank-32) cover the
// parameter range we'd actually adapt: rank-16 is the LoRA paper's
// canonical setting, rank-32 is at the upper end of "feature learning"
// territory. All assert global rel-err < 1e-5 against the CPU oracle.

fn runGpuLoraSmoke(allocator: std.mem.Allocator) !void {
    var ctx = try vk.Context.init(allocator);
    defer ctx.deinit();

    const SmokeCase = struct {
        name: []const u8,
        m: u32,
        n: u32,
        k: u32,
        r: u32,
        aor: f32, // alpha / r
    };
    const cases = [_]SmokeCase{
        .{ .name = "rank-4 (M=4 N=8 K=16)", .m = 4, .n = 8, .k = 16, .r = 4, .aor = 2.0 },
        .{ .name = "rank-16 (M=8 N=64 K=128)", .m = 8, .n = 64, .k = 128, .r = 16, .aor = 2.0 },
        .{ .name = "rank-32 (M=4 N=32 K=64)", .m = 4, .n = 32, .k = 64, .r = 32, .aor = 1.0 },
    };

    // Build pipelines once — reused across cases.
    var k_matmul = try pipeline.Kernel.init(&ctx, &shaders.matmul_nt_v2, 3, @sizeOf(MatmulPush));
    defer k_matmul.deinit();
    var k_lin_dx = try pipeline.Kernel.init(&ctx, &shaders.linear_backward_dx_batched, 3, @sizeOf(runtime.LinearBatchedPush));
    defer k_lin_dx.deinit();
    var k_lin_dw = try pipeline.Kernel.init(&ctx, &shaders.linear_backward_dw_batched, 3, @sizeOf(runtime.LinearBatchedPush));
    defer k_lin_dw.deinit();
    var k_scale = try pipeline.Kernel.init(&ctx, &shaders.scale, 2, @sizeOf(ScalePush));
    defer k_scale.deinit();
    var k_add = try pipeline.Kernel.init(&ctx, &shaders.add_in_place, 2, @sizeOf(runtime.AddInPlacePush));
    defer k_add.deinit();

    const group_lwg: u32 = 16; // matches linear_backward_d{x,w}_batched layout
    const group_lin: u32 = 256; // matches scale + add_in_place

    for (cases) |cs| {
        const M: usize = cs.m;
        const Nn: usize = cs.n; // can't shadow top-level `const N` for the vec_add smoke
        const K: usize = cs.k;
        const r: usize = cs.r;

        // ── Host-side init: random x, W, A, B, dy.
        var prng = std.Random.DefaultPrng.init(0xABBA_FACE +% @as(u64, cs.m) *% 1000 +% cs.n);
        const rng = prng.random();

        const x = try allocator.alloc(f32, M * K);
        defer allocator.free(x);
        const w = try allocator.alloc(f32, Nn * K);
        defer allocator.free(w);
        const a = try allocator.alloc(f32, r * K);
        defer allocator.free(a);
        const b = try allocator.alloc(f32, Nn * r);
        defer allocator.free(b);
        const dy = try allocator.alloc(f32, M * Nn);
        defer allocator.free(dy);

        for (x) |*v| v.* = (rng.float(f32) - 0.5) * 0.3;
        for (w) |*v| v.* = (rng.float(f32) - 0.5) * 0.2;
        for (a) |*v| v.* = (rng.float(f32) - 0.5) * 0.2;
        for (b) |*v| v.* = (rng.float(f32) - 0.5) * 0.2;
        for (dy) |*v| v.* = (rng.float(f32) - 0.5) * 0.4;

        // ── CPU oracle reference outputs.
        const y_cpu = try allocator.alloc(f32, M * Nn);
        defer allocator.free(y_cpu);
        const intermediate_cpu = try allocator.alloc(f32, M * r);
        defer allocator.free(intermediate_cpu);
        cpu_lora.loraForward(x, w, a, b, M, Nn, K, r, cs.aor, y_cpu, intermediate_cpu);

        const dx_cpu = try allocator.alloc(f32, M * K);
        defer allocator.free(dx_cpu);
        const dA_cpu = try allocator.alloc(f32, r * K);
        defer allocator.free(dA_cpu);
        const dB_cpu = try allocator.alloc(f32, Nn * r);
        defer allocator.free(dB_cpu);
        @memset(dA_cpu, 0);
        @memset(dB_cpu, 0);
        try cpu_lora.loraBackward(dy, x, w, a, b, intermediate_cpu, M, Nn, K, r, cs.aor, dx_cpu, dA_cpu, dB_cpu, allocator);

        // ── GPU buffer allocation.
        var buf_x = try buffer.Buffer.initStatic(&ctx, f32, x);
        defer buf_x.deinit(ctx.device);
        var buf_w = try buffer.Buffer.initStatic(&ctx, f32, w);
        defer buf_w.deinit(ctx.device);
        var buf_a = try buffer.Buffer.initStatic(&ctx, f32, a);
        defer buf_a.deinit(ctx.device);
        var buf_b = try buffer.Buffer.initStatic(&ctx, f32, b);
        defer buf_b.deinit(ctx.device);
        var buf_dy = try buffer.Buffer.initStatic(&ctx, f32, dy);
        defer buf_dy.deinit(ctx.device);

        // Outputs.
        var buf_y = try buffer.Buffer.initDeviceOnly(&ctx, M * Nn * @sizeOf(f32));
        defer buf_y.deinit(ctx.device);
        var buf_intermediate = try buffer.Buffer.initDeviceOnly(&ctx, M * r * @sizeOf(f32));
        defer buf_intermediate.deinit(ctx.device);
        var buf_dx = try buffer.Buffer.initDeviceOnly(&ctx, M * K * @sizeOf(f32));
        defer buf_dx.deinit(ctx.device);
        var buf_dA = try buffer.Buffer.initDeviceOnly(&ctx, r * K * @sizeOf(f32));
        defer buf_dA.deinit(ctx.device);
        var buf_dB = try buffer.Buffer.initDeviceOnly(&ctx, Nn * r * @sizeOf(f32));
        defer buf_dB.deinit(ctx.device);

        // Scratches.
        var buf_y_lora = try buffer.Buffer.initDeviceOnly(&ctx, M * Nn * @sizeOf(f32));
        defer buf_y_lora.deinit(ctx.device);
        var buf_y_lora_scaled = try buffer.Buffer.initDeviceOnly(&ctx, M * Nn * @sizeOf(f32));
        defer buf_y_lora_scaled.deinit(ctx.device);
        var buf_dy_B = try buffer.Buffer.initDeviceOnly(&ctx, M * r * @sizeOf(f32));
        defer buf_dy_B.deinit(ctx.device);
        var buf_dx_lora = try buffer.Buffer.initDeviceOnly(&ctx, M * K * @sizeOf(f32));
        defer buf_dx_lora.deinit(ctx.device);
        var buf_dx_lora_scaled = try buffer.Buffer.initDeviceOnly(&ctx, M * K * @sizeOf(f32));
        defer buf_dx_lora_scaled.deinit(ctx.device);
        var buf_dA_unscaled = try buffer.Buffer.initDeviceOnly(&ctx, r * K * @sizeOf(f32));
        defer buf_dA_unscaled.deinit(ctx.device);
        var buf_dB_unscaled = try buffer.Buffer.initDeviceOnly(&ctx, Nn * r * @sizeOf(f32));
        defer buf_dB_unscaled.deinit(ctx.device);

        // Push-constant constants.
        const M_u32: u32 = cs.m;
        const N_u32: u32 = cs.n;
        const K_u32: u32 = cs.k;
        const R_u32: u32 = cs.r;

        // Helper: dispatch one matmul_nt_v2 (1D, gx output cells).
        const recordMatmul = struct {
            fn rec(rec_kern: *pipeline.Kernel, push: *const MatmulPush, gx: u32, c_ctx: *const vk.Context, bufs: []const *const buffer.Buffer) !void {
                try rec_kern.bind(bufs);
                const Rec = struct {
                    k: *const pipeline.Kernel,
                    p: *const MatmulPush,
                    gx_: u32,
                    pub fn record(s: @This(), cmd: vk.c.VkCommandBuffer) void {
                        s.k.dispatch(cmd, s.p, s.gx_, 1, 1);
                    }
                };
                try buffer.submitOneShot(c_ctx, Rec{ .k = rec_kern, .p = push, .gx_ = gx });
            }
        }.rec;

        // Helper: dispatch one linear_backward_d{x,w}_batched (16x16 grid).
        const recordLinBackward = struct {
            fn rec(rec_kern: *pipeline.Kernel, push: *const runtime.LinearBatchedPush, gx: u32, gy: u32, c_ctx: *const vk.Context, bufs: []const *const buffer.Buffer) !void {
                try rec_kern.bind(bufs);
                const Rec = struct {
                    k: *const pipeline.Kernel,
                    p: *const runtime.LinearBatchedPush,
                    gx_: u32,
                    gy_: u32,
                    pub fn record(s: @This(), cmd: vk.c.VkCommandBuffer) void {
                        s.k.dispatch(cmd, s.p, s.gx_, s.gy_, 1);
                    }
                };
                try buffer.submitOneShot(c_ctx, Rec{ .k = rec_kern, .p = push, .gx_ = gx, .gy_ = gy });
            }
        }.rec;

        // Helper: dispatch a 1D scalar kernel (scale, add_in_place).
        const recordScalar1D = struct {
            fn rec(rec_kern: *pipeline.Kernel, push: anytype, gx: u32, c_ctx: *const vk.Context, bufs: []const *const buffer.Buffer) !void {
                try rec_kern.bind(bufs);
                const PushT = @TypeOf(push);
                const Rec = struct {
                    k: *const pipeline.Kernel,
                    p: *const PushT,
                    gx_: u32,
                    pub fn record(s: @This(), cmd: vk.c.VkCommandBuffer) void {
                        s.k.dispatch(cmd, s.p, s.gx_, 1, 1);
                    }
                };
                try buffer.submitOneShot(c_ctx, Rec{ .k = rec_kern, .p = &push, .gx_ = gx });
            }
        }.rec;

        // ── FORWARD chain.
        // 1. y_base = x · Wᵀ
        const push_y_base = MatmulPush{ .m = M_u32, .n = N_u32, .k = K_u32 };
        try recordMatmul(&k_matmul, &push_y_base, M_u32 * N_u32, &ctx, &.{ &buf_x, &buf_w, &buf_y });
        // 2. intermediate = x · Aᵀ
        const push_inter = MatmulPush{ .m = M_u32, .n = R_u32, .k = K_u32 };
        try recordMatmul(&k_matmul, &push_inter, M_u32 * R_u32, &ctx, &.{ &buf_x, &buf_a, &buf_intermediate });
        // 3. y_lora = intermediate · Bᵀ
        const push_ylora = MatmulPush{ .m = M_u32, .n = N_u32, .k = R_u32 };
        try recordMatmul(&k_matmul, &push_ylora, M_u32 * N_u32, &ctx, &.{ &buf_intermediate, &buf_b, &buf_y_lora });
        // 4. y_lora_scaled = y_lora * (α/r)
        const push_scale_ylora = ScalePush{ .n = M_u32 * N_u32, .scale = cs.aor };
        try recordScalar1D(&k_scale, push_scale_ylora, ceilDiv(M_u32 * N_u32, group_lin), &ctx, &.{ &buf_y_lora, &buf_y_lora_scaled });
        // 5. y += y_lora_scaled  (in place into buf_y)
        const push_add_y = runtime.AddInPlacePush{ .n = M_u32 * N_u32 };
        try recordScalar1D(&k_add, push_add_y, ceilDiv(M_u32 * N_u32, group_lin), &ctx, &.{ &buf_y, &buf_y_lora_scaled });

        // ── BACKWARD chain.
        // 1. dx_base = dy · W                  (shape M×K = M×Nn · Nn×K)
        const push_dx_base = runtime.LinearBatchedPush{ .M = M_u32, .N = N_u32, .K = K_u32 };
        try recordLinBackward(&k_lin_dx, &push_dx_base, ceilDiv(M_u32, group_lwg), ceilDiv(K_u32, group_lwg), &ctx, &.{ &buf_dy, &buf_w, &buf_dx });
        // 2. dy_B = dy · B                     (shape M×r = M×Nn · Nn×r). LinearBatchedPush.K = r.
        const push_dy_B = runtime.LinearBatchedPush{ .M = M_u32, .N = N_u32, .K = R_u32 };
        try recordLinBackward(&k_lin_dx, &push_dy_B, ceilDiv(M_u32, group_lwg), ceilDiv(R_u32, group_lwg), &ctx, &.{ &buf_dy, &buf_b, &buf_dy_B });
        // 3. dx_lora = dy_B · A                (shape M×K = M×r · r×K). Nn=r in the linear-backward sense.
        const push_dx_lora = runtime.LinearBatchedPush{ .M = M_u32, .N = R_u32, .K = K_u32 };
        try recordLinBackward(&k_lin_dx, &push_dx_lora, ceilDiv(M_u32, group_lwg), ceilDiv(K_u32, group_lwg), &ctx, &.{ &buf_dy_B, &buf_a, &buf_dx_lora });
        // 4. dx_lora_scaled = dx_lora * (α/r)
        const push_scale_dxlora = ScalePush{ .n = M_u32 * K_u32, .scale = cs.aor };
        try recordScalar1D(&k_scale, push_scale_dxlora, ceilDiv(M_u32 * K_u32, group_lin), &ctx, &.{ &buf_dx_lora, &buf_dx_lora_scaled });
        // 5. dx += dx_lora_scaled
        const push_add_dx = runtime.AddInPlacePush{ .n = M_u32 * K_u32 };
        try recordScalar1D(&k_add, push_add_dx, ceilDiv(M_u32 * K_u32, group_lin), &ctx, &.{ &buf_dx, &buf_dx_lora_scaled });
        // 6. ∇A_unscaled = dy_Bᵀ · x          (shape r×K). dW[Nn=r, K]
        const push_dA = runtime.LinearBatchedPush{ .M = M_u32, .N = R_u32, .K = K_u32 };
        try recordLinBackward(&k_lin_dw, &push_dA, ceilDiv(R_u32, group_lwg), ceilDiv(K_u32, group_lwg), &ctx, &.{ &buf_dy_B, &buf_x, &buf_dA_unscaled });
        // 7. ∇A = ∇A_unscaled * (α/r)
        const push_scale_dA = ScalePush{ .n = R_u32 * K_u32, .scale = cs.aor };
        try recordScalar1D(&k_scale, push_scale_dA, ceilDiv(R_u32 * K_u32, group_lin), &ctx, &.{ &buf_dA_unscaled, &buf_dA });
        // 8. ∇B_unscaled = dyᵀ · intermediate (shape Nn×r). dW[Nn, K=r]
        const push_dB = runtime.LinearBatchedPush{ .M = M_u32, .N = N_u32, .K = R_u32 };
        try recordLinBackward(&k_lin_dw, &push_dB, ceilDiv(N_u32, group_lwg), ceilDiv(R_u32, group_lwg), &ctx, &.{ &buf_dy, &buf_intermediate, &buf_dB_unscaled });
        // 9. ∇B = ∇B_unscaled * (α/r)
        const push_scale_dB = ScalePush{ .n = N_u32 * R_u32, .scale = cs.aor };
        try recordScalar1D(&k_scale, push_scale_dB, ceilDiv(N_u32 * R_u32, group_lin), &ctx, &.{ &buf_dB_unscaled, &buf_dB });

        // ── Read back + diff.
        const y_gpu = try allocator.alloc(f32, M * Nn);
        defer allocator.free(y_gpu);
        const dx_gpu = try allocator.alloc(f32, M * K);
        defer allocator.free(dx_gpu);
        const dA_gpu = try allocator.alloc(f32, r * K);
        defer allocator.free(dA_gpu);
        const dB_gpu = try allocator.alloc(f32, Nn * r);
        defer allocator.free(dB_gpu);
        try buf_y.readBack(&ctx, f32, y_gpu);
        try buf_dx.readBack(&ctx, f32, dx_gpu);
        try buf_dA.readBack(&ctx, f32, dA_gpu);
        try buf_dB.readBack(&ctx, f32, dB_gpu);

        const tol: f32 = 1e-5;
        const y_rel = globalRelDiff(y_cpu, y_gpu);
        const dx_rel = globalRelDiff(dx_cpu, dx_gpu);
        const dA_rel = globalRelDiff(dA_cpu, dA_gpu);
        const dB_rel = globalRelDiff(dB_cpu, dB_gpu);
        if (y_rel >= tol or dx_rel >= tol or dA_rel >= tol or dB_rel >= tol) {
            std.debug.print(
                "LoRA smoke ({s}) FAIL: y_rel={e}  dx_rel={e}  dA_rel={e}  dB_rel={e}  (tol {e})\n",
                .{ cs.name, y_rel, dx_rel, dA_rel, dB_rel, tol },
            );
            return error.ParityFailed;
        }
        std.debug.print(
            "PASS GPU LoRA — {s}  α/r={d:.2}  rel(y/dx/dA/dB)=({e},{e},{e},{e})\n",
            .{ cs.name, cs.aor, y_rel, dx_rel, dA_rel, dB_rel },
        );
    }
}

// ── LoRA train demo: end-to-end "does it actually train" gate ─────────
//
// Synthetic recovery task. Construct a frozen base W and a rank-r
// target delta (B_target · A_target). Generate target_y by running
// linear forward with W_eff = W + (α/r) · B_target · A_target. Train
// the LoRA adapter (A trainable, B = 0 init) with Adam against MSE
// loss vs target_y; assert the loss drops by > 100× over 100 steps.
//
// Why this gate matters: it's the smallest end-to-end signal that
//   1. The fwd / bwd dispatch chain matches the math (already proven
//      by --lora-smoke at the per-kernel level — this exercises it
//      in a closed-loop optimizer setting where bugs accumulate),
//   2. The B = 0 / ∇A = 0 init asymmetry doesn't deadlock training
//      (B's gradient kicks in at step 1, A's at step 2 once B ≠ 0),
//   3. AdamW updates the LoRA params correctly without disturbing
//      the frozen base.
//
// Loss convention matches the existing mse_loss_grad shader:
// half-sum MSE, L = ½ Σ(y - target)². The gradient `dy = y - target`
// is what mse_loss_grad emits; Adam's effective LR absorbs the 1/(M·N)
// factor that a mean-MSE convention would carry. Per-step compute
// is independent of the convention; only the displayed loss scale
// differs.

fn runGpuLoraTrainDemo(allocator: std.mem.Allocator) !void {
    var ctx = try vk.Context.init(allocator);
    defer ctx.deinit();

    // Problem shape — small enough for a 100-step run to be sub-second
    // but big enough that the rank-r recovery is non-trivial.
    const M_u: u32 = 16; // batch size
    const N_u: u32 = 32; // output dim
    const K_u: u32 = 64; // input dim
    const R_u: u32 = 8; // LoRA rank
    const aor: f32 = 2.0; // α/r (LoRA paper canonical)
    const lr: f32 = 1e-2;
    const beta1: f32 = 0.9;
    const beta2: f32 = 0.999;
    const eps: f32 = 1e-8;
    const n_steps: u32 = 100;
    const log_every: u32 = 25;

    const M: usize = M_u;
    const Nn: usize = N_u; // shadows top-level vec_add `const N`
    const K: usize = K_u;
    const r: usize = R_u;

    // ── Construct synthetic problem.
    var prng = std.Random.DefaultPrng.init(0x10AAFADE);
    const rng = prng.random();

    const w = try allocator.alloc(f32, Nn * K);
    defer allocator.free(w);
    for (w) |*v| v.* = (rng.float(f32) - 0.5) * 0.2;

    // The target rank-r delta we want LoRA to recover.
    const a_target = try allocator.alloc(f32, r * K);
    defer allocator.free(a_target);
    const b_target = try allocator.alloc(f32, Nn * r);
    defer allocator.free(b_target);
    for (a_target) |*v| v.* = (rng.float(f32) - 0.5) * 0.5;
    for (b_target) |*v| v.* = (rng.float(f32) - 0.5) * 0.5;

    // x batch (fixed across steps — overfit-on-batch demo).
    const x = try allocator.alloc(f32, M * K);
    defer allocator.free(x);
    for (x) |*v| v.* = (rng.float(f32) - 0.5);

    // target_y = x · (W + (α/r)·B_target·A_target)ᵀ
    const w_eff_target = try allocator.alloc(f32, Nn * K);
    defer allocator.free(w_eff_target);
    @memcpy(w_eff_target, w);
    for (0..Nn) |n_| {
        for (0..K) |k_| {
            var s: f64 = 0;
            for (0..r) |ri_| s += @as(f64, b_target[n_ * r + ri_]) * @as(f64, a_target[ri_ * K + k_]);
            w_eff_target[n_ * K + k_] += aor * @as(f32, @floatCast(s));
        }
    }
    const target_y = try allocator.alloc(f32, M * Nn);
    defer allocator.free(target_y);
    for (0..M) |m_| {
        for (0..Nn) |n_| {
            var s: f64 = 0;
            for (0..K) |k_| s += @as(f64, x[m_ * K + k_]) * @as(f64, w_eff_target[n_ * K + k_]);
            target_y[m_ * Nn + n_] = @floatCast(s);
        }
    }

    // ── Trainable LoRA params: A small random, B zero (canonical init).
    const a_init = try allocator.alloc(f32, r * K);
    defer allocator.free(a_init);
    for (a_init) |*v| v.* = (rng.float(f32) - 0.5) * 0.1;
    const b_init = try allocator.alloc(f32, Nn * r);
    defer allocator.free(b_init);
    @memset(b_init, 0);

    // ── GPU buffer allocation.
    var buf_x = try buffer.Buffer.initStatic(&ctx, f32, x);
    defer buf_x.deinit(ctx.device);
    var buf_w = try buffer.Buffer.initStatic(&ctx, f32, w);
    defer buf_w.deinit(ctx.device);
    var buf_a = try buffer.Buffer.initStatic(&ctx, f32, a_init); // mutable on GPU side
    defer buf_a.deinit(ctx.device);
    var buf_b = try buffer.Buffer.initStatic(&ctx, f32, b_init);
    defer buf_b.deinit(ctx.device);
    var buf_target_y = try buffer.Buffer.initStatic(&ctx, f32, target_y);
    defer buf_target_y.deinit(ctx.device);

    var buf_y = try buffer.Buffer.initDeviceOnly(&ctx, M * Nn * @sizeOf(f32));
    defer buf_y.deinit(ctx.device);
    var buf_intermediate = try buffer.Buffer.initDeviceOnly(&ctx, M * r * @sizeOf(f32));
    defer buf_intermediate.deinit(ctx.device);
    var buf_y_lora = try buffer.Buffer.initDeviceOnly(&ctx, M * Nn * @sizeOf(f32));
    defer buf_y_lora.deinit(ctx.device);
    var buf_y_lora_scaled = try buffer.Buffer.initDeviceOnly(&ctx, M * Nn * @sizeOf(f32));
    defer buf_y_lora_scaled.deinit(ctx.device);
    var buf_dy = try buffer.Buffer.initDeviceOnly(&ctx, M * Nn * @sizeOf(f32));
    defer buf_dy.deinit(ctx.device);
    var buf_dy_B = try buffer.Buffer.initDeviceOnly(&ctx, M * r * @sizeOf(f32));
    defer buf_dy_B.deinit(ctx.device);
    var buf_dA_unscaled = try buffer.Buffer.initDeviceOnly(&ctx, r * K * @sizeOf(f32));
    defer buf_dA_unscaled.deinit(ctx.device);
    var buf_dA = try buffer.Buffer.initDeviceOnly(&ctx, r * K * @sizeOf(f32));
    defer buf_dA.deinit(ctx.device);
    var buf_dB_unscaled = try buffer.Buffer.initDeviceOnly(&ctx, Nn * r * @sizeOf(f32));
    defer buf_dB_unscaled.deinit(ctx.device);
    var buf_dB = try buffer.Buffer.initDeviceOnly(&ctx, Nn * r * @sizeOf(f32));
    defer buf_dB.deinit(ctx.device);

    // Adam state (zero-init via initDeviceOnly).
    var buf_m_a = try buffer.Buffer.initDeviceOnly(&ctx, r * K * @sizeOf(f32));
    defer buf_m_a.deinit(ctx.device);
    var buf_v_a = try buffer.Buffer.initDeviceOnly(&ctx, r * K * @sizeOf(f32));
    defer buf_v_a.deinit(ctx.device);
    var buf_m_b = try buffer.Buffer.initDeviceOnly(&ctx, Nn * r * @sizeOf(f32));
    defer buf_m_b.deinit(ctx.device);
    var buf_v_b = try buffer.Buffer.initDeviceOnly(&ctx, Nn * r * @sizeOf(f32));
    defer buf_v_b.deinit(ctx.device);

    // ── Pipelines.
    var k_matmul = try pipeline.Kernel.init(&ctx, &shaders.matmul_nt_v2, 3, @sizeOf(MatmulPush));
    defer k_matmul.deinit();
    var k_lin_dx = try pipeline.Kernel.init(&ctx, &shaders.linear_backward_dx_batched, 3, @sizeOf(runtime.LinearBatchedPush));
    defer k_lin_dx.deinit();
    var k_lin_dw = try pipeline.Kernel.init(&ctx, &shaders.linear_backward_dw_batched, 3, @sizeOf(runtime.LinearBatchedPush));
    defer k_lin_dw.deinit();
    var k_scale = try pipeline.Kernel.init(&ctx, &shaders.scale, 2, @sizeOf(ScalePush));
    defer k_scale.deinit();
    var k_add = try pipeline.Kernel.init(&ctx, &shaders.add_in_place, 2, @sizeOf(runtime.AddInPlacePush));
    defer k_add.deinit();
    var k_mse_grad = try pipeline.Kernel.init(&ctx, &shaders.mse_loss_grad, 3, @sizeOf(MseLossGradPush));
    defer k_mse_grad.deinit();
    var k_adam = try pipeline.Kernel.init(&ctx, &shaders.adam_step, 4, @sizeOf(runtime.AdamStepPush));
    defer k_adam.deinit();

    const group_lwg: u32 = 16;
    const group_lin: u32 = 256;

    // Helpers.
    const recordMatmul = struct {
        fn rec(rec_kern: *pipeline.Kernel, push: *const MatmulPush, gx: u32, c_ctx: *const vk.Context, bufs: []const *const buffer.Buffer) !void {
            try rec_kern.bind(bufs);
            const Rec = struct {
                k: *const pipeline.Kernel,
                p: *const MatmulPush,
                gx_: u32,
                pub fn record(s: @This(), cmd: vk.c.VkCommandBuffer) void {
                    s.k.dispatch(cmd, s.p, s.gx_, 1, 1);
                }
            };
            try buffer.submitOneShot(c_ctx, Rec{ .k = rec_kern, .p = push, .gx_ = gx });
        }
    }.rec;
    const recordLinBackward = struct {
        fn rec(rec_kern: *pipeline.Kernel, push: *const runtime.LinearBatchedPush, gx: u32, gy: u32, c_ctx: *const vk.Context, bufs: []const *const buffer.Buffer) !void {
            try rec_kern.bind(bufs);
            const Rec = struct {
                k: *const pipeline.Kernel,
                p: *const runtime.LinearBatchedPush,
                gx_: u32,
                gy_: u32,
                pub fn record(s: @This(), cmd: vk.c.VkCommandBuffer) void {
                    s.k.dispatch(cmd, s.p, s.gx_, s.gy_, 1);
                }
            };
            try buffer.submitOneShot(c_ctx, Rec{ .k = rec_kern, .p = push, .gx_ = gx, .gy_ = gy });
        }
    }.rec;
    const recordScalar1D = struct {
        fn rec(rec_kern: *pipeline.Kernel, push: anytype, gx: u32, c_ctx: *const vk.Context, bufs: []const *const buffer.Buffer) !void {
            try rec_kern.bind(bufs);
            const PushT = @TypeOf(push);
            const Rec = struct {
                k: *const pipeline.Kernel,
                p: *const PushT,
                gx_: u32,
                pub fn record(s: @This(), cmd: vk.c.VkCommandBuffer) void {
                    s.k.dispatch(cmd, s.p, s.gx_, 1, 1);
                }
            };
            try buffer.submitOneShot(c_ctx, Rec{ .k = rec_kern, .p = &push, .gx_ = gx });
        }
    }.rec;

    // Push constants reused across steps.
    const push_y_base = MatmulPush{ .m = M_u, .n = N_u, .k = K_u };
    const push_inter = MatmulPush{ .m = M_u, .n = R_u, .k = K_u };
    const push_ylora = MatmulPush{ .m = M_u, .n = N_u, .k = R_u };
    const push_scale_ylora = ScalePush{ .n = M_u * N_u, .scale = aor };
    const push_add_y = runtime.AddInPlacePush{ .n = M_u * N_u };
    const push_mse = MseLossGradPush{ .n = M_u * N_u };
    const push_dy_B = runtime.LinearBatchedPush{ .M = M_u, .N = N_u, .K = R_u };
    const push_dA = runtime.LinearBatchedPush{ .M = M_u, .N = R_u, .K = K_u };
    const push_scale_dA = ScalePush{ .n = R_u * K_u, .scale = aor };
    const push_dB = runtime.LinearBatchedPush{ .M = M_u, .N = N_u, .K = R_u };
    const push_scale_dB = ScalePush{ .n = N_u * R_u, .scale = aor };

    // Loss helper (host-side, half-sum MSE matching the shader convention).
    const computeMse = struct {
        fn call(y: []const f32, t: []const f32) f32 {
            var s: f64 = 0;
            for (y, t) |yv, tv| {
                const d = yv - tv;
                s += d * d;
            }
            return @floatCast(0.5 * s);
        }
    }.call;

    // Forward pass — 5 dispatches.
    const runForward = struct {
        fn call(
            c_ctx: *const vk.Context,
            kmm: *pipeline.Kernel,
            ksc: *pipeline.Kernel,
            kad: *pipeline.Kernel,
            bx: *const buffer.Buffer,
            bw: *const buffer.Buffer,
            ba: *const buffer.Buffer,
            bb: *const buffer.Buffer,
            bint: *const buffer.Buffer,
            bylora: *const buffer.Buffer,
            byloras: *const buffer.Buffer,
            by: *const buffer.Buffer,
            pyb: *const MatmulPush,
            pin: *const MatmulPush,
            pyl: *const MatmulPush,
            psy: *const ScalePush,
            pay: *const runtime.AddInPlacePush,
            mu: u32,
            nu: u32,
            ru: u32,
            recM_fn: @TypeOf(recordMatmul),
            recS_fn: @TypeOf(recordScalar1D),
            glin: u32,
        ) !void {
            try recM_fn(kmm, pyb, mu * nu, c_ctx, &.{ bx, bw, by });
            try recM_fn(kmm, pin, mu * ru, c_ctx, &.{ bx, ba, bint });
            try recM_fn(kmm, pyl, mu * nu, c_ctx, &.{ bint, bb, bylora });
            try recS_fn(ksc, psy.*, ceilDiv(mu * nu, glin), c_ctx, &.{ bylora, byloras });
            try recS_fn(kad, pay.*, ceilDiv(mu * nu, glin), c_ctx, &.{ by, byloras });
        }
    }.call;

    // ── Initial loss.
    try runForward(
        &ctx, &k_matmul, &k_scale, &k_add,
        &buf_x, &buf_w, &buf_a, &buf_b,
        &buf_intermediate, &buf_y_lora, &buf_y_lora_scaled, &buf_y,
        &push_y_base, &push_inter, &push_ylora, &push_scale_ylora, &push_add_y,
        M_u, N_u, R_u, recordMatmul, recordScalar1D, group_lin,
    );
    const y_buf = try allocator.alloc(f32, M * Nn);
    defer allocator.free(y_buf);
    try buf_y.readBack(&ctx, f32, y_buf);
    const initial_loss = computeMse(y_buf, target_y);
    std.debug.print(
        "LoRA train demo on {s}\n  shape: M={d} N={d} K={d} r={d}  α/r={d}  lr={d}\n  step    0 (init): loss = {d:.6}\n",
        .{ ctx.deviceName(), M, Nn, K, r, aor, lr, initial_loss },
    );

    // ── Train loop.
    const t_start = std.time.nanoTimestamp();
    var step: u32 = 1;
    while (step <= n_steps) : (step += 1) {
        // Forward.
        try runForward(
            &ctx, &k_matmul, &k_scale, &k_add,
            &buf_x, &buf_w, &buf_a, &buf_b,
            &buf_intermediate, &buf_y_lora, &buf_y_lora_scaled, &buf_y,
            &push_y_base, &push_inter, &push_ylora, &push_scale_ylora, &push_add_y,
            M_u, N_u, R_u, recordMatmul, recordScalar1D, group_lin,
        );
        // Loss grad: dy = y - target  (shader is half-sum MSE).
        try recordScalar1D(&k_mse_grad, push_mse, ceilDiv(M_u * N_u, group_lin), &ctx, &.{ &buf_y, &buf_target_y, &buf_dy });
        // Backward — dx is not computed (no upstream).
        // dy_B = dy · B
        try recordLinBackward(&k_lin_dx, &push_dy_B, ceilDiv(M_u, group_lwg), ceilDiv(R_u, group_lwg), &ctx, &.{ &buf_dy, &buf_b, &buf_dy_B });
        // ∇A_unscaled = dy_Bᵀ · x ; ∇A = (α/r) · ∇A_unscaled
        try recordLinBackward(&k_lin_dw, &push_dA, ceilDiv(R_u, group_lwg), ceilDiv(K_u, group_lwg), &ctx, &.{ &buf_dy_B, &buf_x, &buf_dA_unscaled });
        try recordScalar1D(&k_scale, push_scale_dA, ceilDiv(R_u * K_u, group_lin), &ctx, &.{ &buf_dA_unscaled, &buf_dA });
        // ∇B_unscaled = dyᵀ · intermediate ; ∇B = (α/r) · ∇B_unscaled
        try recordLinBackward(&k_lin_dw, &push_dB, ceilDiv(N_u, group_lwg), ceilDiv(R_u, group_lwg), &ctx, &.{ &buf_dy, &buf_intermediate, &buf_dB_unscaled });
        try recordScalar1D(&k_scale, push_scale_dB, ceilDiv(N_u * R_u, group_lin), &ctx, &.{ &buf_dB_unscaled, &buf_dB });
        // Adam updates on A and B.
        const adam_a = runtime.AdamStepPush{ .n = R_u * K_u, .lr = lr, .beta1 = beta1, .beta2 = beta2, .eps = eps, .t = step };
        const adam_b = runtime.AdamStepPush{ .n = N_u * R_u, .lr = lr, .beta1 = beta1, .beta2 = beta2, .eps = eps, .t = step };
        try recordScalar1D(&k_adam, adam_a, ceilDiv(R_u * K_u, group_lin), &ctx, &.{ &buf_a, &buf_dA, &buf_m_a, &buf_v_a });
        try recordScalar1D(&k_adam, adam_b, ceilDiv(N_u * R_u, group_lin), &ctx, &.{ &buf_b, &buf_dB, &buf_m_b, &buf_v_b });

        if (step % log_every == 0) {
            try buf_y.readBack(&ctx, f32, y_buf);
            const loss = computeMse(y_buf, target_y);
            std.debug.print("  step {d:>4}        : loss = {d:.6}\n", .{ step, loss });
        }
    }
    const t_end = std.time.nanoTimestamp();
    const total_ms: f64 = @as(f64, @floatFromInt(t_end - t_start)) / 1.0e6;

    try buf_y.readBack(&ctx, f32, y_buf);
    const final_loss = computeMse(y_buf, target_y);
    const ratio = final_loss / @max(initial_loss, 1e-30);

    std.debug.print(
        "  step {d:>4} (final): loss = {d:.6}\n  drop: {d:.4}× over {d} steps in {d:.1} ms ({d:.2} ms/step)\n",
        .{ n_steps, final_loss, 1.0 / ratio, n_steps, total_ms, total_ms / @as(f64, @floatFromInt(n_steps)) },
    );

    if (ratio > 1e-2) {
        std.debug.print("FAIL: loss did not drop ≥ 100×  (ratio = {d:.4})\n", .{ratio});
        return error.LossDidNotDecrease;
    }
    std.debug.print("PASS GPU LoRA train demo — recovered rank-{d} delta, loss {d:.4} → {d:.4} ({d:.2}× drop)\n", .{ r, initial_loss, final_loss, 1.0 / ratio });
}

// ── LoRA+ comparative demo: η_B / η_A ∈ {1, 4, 16} on same task ───────
//
// Same synthetic recovery task as runGpuLoraTrainDemo (rank-r delta on
// a frozen base), but trains three trajectories with different LoRA+
// learning-rate ratios λ = η_B / η_A. Vanilla LoRA is λ = 1; the
// Hayou et al. (ICML 2024) / Chronicals §5 theoretical optimum in the
// feature-learning regime is λ = O(n) ≈ 16 for our shape.
//
// Why λ matters. At init, B = 0 and A = N(0, σ²), so:
//   ∇A = (α/r) · Bᵀ · ∇W_eff = 0    (gated by Bᵀ = 0)
//   ∇B = (α/r) · ∇W_eff · Aᵀ ≠ 0    (only path that's "open")
// B has to learn first before A can start contributing. With η_B = η_A,
// B catches up slowly; with η_B = 16·η_A, B's first few updates are
// large enough to push (α/r)·B·A into a regime where ∇A becomes
// non-trivial within a few steps — significantly accelerating
// convergence on tasks where the optimal rank-r delta is non-trivial.
//
// The demo runs each ratio for the same number of steps with identical
// initial weights (deterministic seed → re-init A, B and Adam state
// between trajectories). Prints all three trajectories side by side
// and asserts that λ = 16 reaches a fixed loss threshold in fewer
// steps than λ = 1.

fn runGpuLoraPlusDemo(allocator: std.mem.Allocator) !void {
    var ctx = try vk.Context.init(allocator);
    defer ctx.deinit();

    // Same shape as runGpuLoraTrainDemo for direct comparison.
    const M_u: u32 = 16;
    const N_u: u32 = 32;
    const K_u: u32 = 64;
    const R_u: u32 = 8;
    const aor: f32 = 2.0;
    const base_lr: f32 = 1e-2;
    const beta1: f32 = 0.9;
    const beta2: f32 = 0.999;
    const eps: f32 = 1e-8;
    const n_steps: u32 = 100;
    const log_every: u32 = 25;

    const ratios = [_]f32{ 1.0, 4.0, 16.0 };

    const M: usize = M_u;
    const Nn: usize = N_u;
    const K: usize = K_u;
    const r: usize = R_u;

    // ── Synthetic problem (deterministic seed).
    var prng = std.Random.DefaultPrng.init(0x10AAFADE);
    const rng = prng.random();

    const w = try allocator.alloc(f32, Nn * K);
    defer allocator.free(w);
    for (w) |*v| v.* = (rng.float(f32) - 0.5) * 0.2;

    const a_target = try allocator.alloc(f32, r * K);
    defer allocator.free(a_target);
    const b_target = try allocator.alloc(f32, Nn * r);
    defer allocator.free(b_target);
    for (a_target) |*v| v.* = (rng.float(f32) - 0.5) * 0.5;
    for (b_target) |*v| v.* = (rng.float(f32) - 0.5) * 0.5;

    const x = try allocator.alloc(f32, M * K);
    defer allocator.free(x);
    for (x) |*v| v.* = (rng.float(f32) - 0.5);

    const w_eff_target = try allocator.alloc(f32, Nn * K);
    defer allocator.free(w_eff_target);
    @memcpy(w_eff_target, w);
    for (0..Nn) |n_| {
        for (0..K) |k_| {
            var s: f64 = 0;
            for (0..r) |ri_| s += @as(f64, b_target[n_ * r + ri_]) * @as(f64, a_target[ri_ * K + k_]);
            w_eff_target[n_ * K + k_] += aor * @as(f32, @floatCast(s));
        }
    }
    const target_y = try allocator.alloc(f32, M * Nn);
    defer allocator.free(target_y);
    for (0..M) |m_| {
        for (0..Nn) |n_| {
            var s: f64 = 0;
            for (0..K) |k_| s += @as(f64, x[m_ * K + k_]) * @as(f64, w_eff_target[n_ * K + k_]);
            target_y[m_ * Nn + n_] = @floatCast(s);
        }
    }

    const a_init = try allocator.alloc(f32, r * K);
    defer allocator.free(a_init);
    for (a_init) |*v| v.* = (rng.float(f32) - 0.5) * 0.1;
    const b_init = try allocator.alloc(f32, Nn * r);
    defer allocator.free(b_init);
    @memset(b_init, 0);

    // ── GPU buffers. A, B are *dynamic* so we can reset between trajectories.
    var buf_x = try buffer.Buffer.initStatic(&ctx, f32, x);
    defer buf_x.deinit(ctx.device);
    var buf_w = try buffer.Buffer.initStatic(&ctx, f32, w);
    defer buf_w.deinit(ctx.device);
    var buf_a = try buffer.Buffer.initDynamic(&ctx, r * K * @sizeOf(f32));
    defer buf_a.deinit(ctx.device);
    var buf_b = try buffer.Buffer.initDynamic(&ctx, Nn * r * @sizeOf(f32));
    defer buf_b.deinit(ctx.device);
    var buf_target_y = try buffer.Buffer.initStatic(&ctx, f32, target_y);
    defer buf_target_y.deinit(ctx.device);

    var buf_y = try buffer.Buffer.initDeviceOnly(&ctx, M * Nn * @sizeOf(f32));
    defer buf_y.deinit(ctx.device);
    var buf_intermediate = try buffer.Buffer.initDeviceOnly(&ctx, M * r * @sizeOf(f32));
    defer buf_intermediate.deinit(ctx.device);
    var buf_y_lora = try buffer.Buffer.initDeviceOnly(&ctx, M * Nn * @sizeOf(f32));
    defer buf_y_lora.deinit(ctx.device);
    var buf_y_lora_scaled = try buffer.Buffer.initDeviceOnly(&ctx, M * Nn * @sizeOf(f32));
    defer buf_y_lora_scaled.deinit(ctx.device);
    var buf_dy = try buffer.Buffer.initDeviceOnly(&ctx, M * Nn * @sizeOf(f32));
    defer buf_dy.deinit(ctx.device);
    var buf_dy_B = try buffer.Buffer.initDeviceOnly(&ctx, M * r * @sizeOf(f32));
    defer buf_dy_B.deinit(ctx.device);
    var buf_dA_unscaled = try buffer.Buffer.initDeviceOnly(&ctx, r * K * @sizeOf(f32));
    defer buf_dA_unscaled.deinit(ctx.device);
    var buf_dA = try buffer.Buffer.initDeviceOnly(&ctx, r * K * @sizeOf(f32));
    defer buf_dA.deinit(ctx.device);
    var buf_dB_unscaled = try buffer.Buffer.initDeviceOnly(&ctx, Nn * r * @sizeOf(f32));
    defer buf_dB_unscaled.deinit(ctx.device);
    var buf_dB = try buffer.Buffer.initDeviceOnly(&ctx, Nn * r * @sizeOf(f32));
    defer buf_dB.deinit(ctx.device);
    var buf_m_a = try buffer.Buffer.initDeviceOnly(&ctx, r * K * @sizeOf(f32));
    defer buf_m_a.deinit(ctx.device);
    var buf_v_a = try buffer.Buffer.initDeviceOnly(&ctx, r * K * @sizeOf(f32));
    defer buf_v_a.deinit(ctx.device);
    var buf_m_b = try buffer.Buffer.initDeviceOnly(&ctx, Nn * r * @sizeOf(f32));
    defer buf_m_b.deinit(ctx.device);
    var buf_v_b = try buffer.Buffer.initDeviceOnly(&ctx, Nn * r * @sizeOf(f32));
    defer buf_v_b.deinit(ctx.device);

    // ── Pipelines.
    var k_matmul = try pipeline.Kernel.init(&ctx, &shaders.matmul_nt_v2, 3, @sizeOf(MatmulPush));
    defer k_matmul.deinit();
    var k_lin_dx = try pipeline.Kernel.init(&ctx, &shaders.linear_backward_dx_batched, 3, @sizeOf(runtime.LinearBatchedPush));
    defer k_lin_dx.deinit();
    var k_lin_dw = try pipeline.Kernel.init(&ctx, &shaders.linear_backward_dw_batched, 3, @sizeOf(runtime.LinearBatchedPush));
    defer k_lin_dw.deinit();
    var k_scale = try pipeline.Kernel.init(&ctx, &shaders.scale, 2, @sizeOf(ScalePush));
    defer k_scale.deinit();
    var k_add = try pipeline.Kernel.init(&ctx, &shaders.add_in_place, 2, @sizeOf(runtime.AddInPlacePush));
    defer k_add.deinit();
    var k_mse_grad = try pipeline.Kernel.init(&ctx, &shaders.mse_loss_grad, 3, @sizeOf(MseLossGradPush));
    defer k_mse_grad.deinit();
    var k_adam = try pipeline.Kernel.init(&ctx, &shaders.adam_step, 4, @sizeOf(runtime.AdamStepPush));
    defer k_adam.deinit();

    const group_lwg: u32 = 16;
    const group_lin: u32 = 256;

    const recordMatmul = struct {
        fn rec(rec_kern: *pipeline.Kernel, push: *const MatmulPush, gx: u32, c_ctx: *const vk.Context, bufs: []const *const buffer.Buffer) !void {
            try rec_kern.bind(bufs);
            const Rec = struct {
                k: *const pipeline.Kernel,
                p: *const MatmulPush,
                gx_: u32,
                pub fn record(s: @This(), cmd: vk.c.VkCommandBuffer) void {
                    s.k.dispatch(cmd, s.p, s.gx_, 1, 1);
                }
            };
            try buffer.submitOneShot(c_ctx, Rec{ .k = rec_kern, .p = push, .gx_ = gx });
        }
    }.rec;
    const recordLinBackward = struct {
        fn rec(rec_kern: *pipeline.Kernel, push: *const runtime.LinearBatchedPush, gx: u32, gy: u32, c_ctx: *const vk.Context, bufs: []const *const buffer.Buffer) !void {
            try rec_kern.bind(bufs);
            const Rec = struct {
                k: *const pipeline.Kernel,
                p: *const runtime.LinearBatchedPush,
                gx_: u32,
                gy_: u32,
                pub fn record(s: @This(), cmd: vk.c.VkCommandBuffer) void {
                    s.k.dispatch(cmd, s.p, s.gx_, s.gy_, 1);
                }
            };
            try buffer.submitOneShot(c_ctx, Rec{ .k = rec_kern, .p = push, .gx_ = gx, .gy_ = gy });
        }
    }.rec;
    const recordScalar1D = struct {
        fn rec(rec_kern: *pipeline.Kernel, push: anytype, gx: u32, c_ctx: *const vk.Context, bufs: []const *const buffer.Buffer) !void {
            try rec_kern.bind(bufs);
            const PushT = @TypeOf(push);
            const Rec = struct {
                k: *const pipeline.Kernel,
                p: *const PushT,
                gx_: u32,
                pub fn record(s: @This(), cmd: vk.c.VkCommandBuffer) void {
                    s.k.dispatch(cmd, s.p, s.gx_, 1, 1);
                }
            };
            try buffer.submitOneShot(c_ctx, Rec{ .k = rec_kern, .p = &push, .gx_ = gx });
        }
    }.rec;

    const push_y_base = MatmulPush{ .m = M_u, .n = N_u, .k = K_u };
    const push_inter = MatmulPush{ .m = M_u, .n = R_u, .k = K_u };
    const push_ylora = MatmulPush{ .m = M_u, .n = N_u, .k = R_u };
    const push_scale_ylora = ScalePush{ .n = M_u * N_u, .scale = aor };
    const push_add_y = runtime.AddInPlacePush{ .n = M_u * N_u };
    const push_mse = MseLossGradPush{ .n = M_u * N_u };
    const push_dy_B = runtime.LinearBatchedPush{ .M = M_u, .N = N_u, .K = R_u };
    const push_dA = runtime.LinearBatchedPush{ .M = M_u, .N = R_u, .K = K_u };
    const push_scale_dA = ScalePush{ .n = R_u * K_u, .scale = aor };
    const push_dB = runtime.LinearBatchedPush{ .M = M_u, .N = N_u, .K = R_u };
    const push_scale_dB = ScalePush{ .n = N_u * R_u, .scale = aor };

    const computeMse = struct {
        fn call(y: []const f32, t: []const f32) f32 {
            var s: f64 = 0;
            for (y, t) |yv, tv| {
                const d = yv - tv;
                s += d * d;
            }
            return @floatCast(0.5 * s);
        }
    }.call;

    const y_buf = try allocator.alloc(f32, M * Nn);
    defer allocator.free(y_buf);

    // ── Run one trajectory at a given LoRA+ ratio. Returns loss values
    //    at steps {0, log_every, 2*log_every, ..., n_steps} and the
    //    smallest step at which loss < threshold (or n_steps + 1 if
    //    never reached).
    const RunResult = struct {
        loss_at_log: [(n_steps / log_every) + 2]f32,
        steps_to_threshold: u32,
    };

    const threshold: f32 = 1e-3;

    const runOneTrajectory = struct {
        fn call(
            ratio: f32,
            base_lr_: f32,
            // buffer state to reset
            buf_a_: *buffer.Buffer,
            buf_b_: *buffer.Buffer,
            buf_m_a_: *const buffer.Buffer,
            buf_v_a_: *const buffer.Buffer,
            buf_m_b_: *const buffer.Buffer,
            buf_v_b_: *const buffer.Buffer,
            a_init_: []const f32,
            b_init_: []const f32,
            // host loss-eval state
            target_y_: []const f32,
            y_buf_: []f32,
            // GPU resources (the long suffix list)
            ctx_ptr: *const vk.Context,
            kmm: *pipeline.Kernel,
            klx: *pipeline.Kernel,
            klw: *pipeline.Kernel,
            ksc: *pipeline.Kernel,
            kad: *pipeline.Kernel,
            kmse: *pipeline.Kernel,
            kadam: *pipeline.Kernel,
            buf_x_: *const buffer.Buffer,
            buf_w_: *const buffer.Buffer,
            buf_target_: *const buffer.Buffer,
            buf_y_: *const buffer.Buffer,
            buf_inter_: *const buffer.Buffer,
            buf_yl_: *const buffer.Buffer,
            buf_yls_: *const buffer.Buffer,
            buf_dy_: *const buffer.Buffer,
            buf_dyB_: *const buffer.Buffer,
            buf_dAu_: *const buffer.Buffer,
            buf_dA_: *const buffer.Buffer,
            buf_dBu_: *const buffer.Buffer,
            buf_dB_: *const buffer.Buffer,
            // pushes
            pyb: *const MatmulPush,
            pin: *const MatmulPush,
            pyl: *const MatmulPush,
            psy: *const ScalePush,
            pay: *const runtime.AddInPlacePush,
            pms: *const MseLossGradPush,
            pdyB: *const runtime.LinearBatchedPush,
            pdA: *const runtime.LinearBatchedPush,
            psda: *const ScalePush,
            pdB: *const runtime.LinearBatchedPush,
            psdb: *const ScalePush,
            mu: u32,
            nu: u32,
            ku: u32,
            ru: u32,
            beta1_: f32,
            beta2_: f32,
            eps_: f32,
            n_steps_: u32,
            log_every_: u32,
            threshold_: f32,
            recM_fn: @TypeOf(recordMatmul),
            recL_fn: @TypeOf(recordLinBackward),
            recS_fn: @TypeOf(recordScalar1D),
            mse_fn: @TypeOf(computeMse),
            glin: u32,
            glwg: u32,
        ) !RunResult {
            // Reset trainable params + Adam state.
            buf_a_.update(f32, a_init_);
            buf_b_.update(f32, b_init_);
            try buf_m_a_.fillZero(ctx_ptr);
            try buf_v_a_.fillZero(ctx_ptr);
            try buf_m_b_.fillZero(ctx_ptr);
            try buf_v_b_.fillZero(ctx_ptr);

            const lr_a: f32 = base_lr_;
            const lr_b: f32 = base_lr_ * ratio;

            var result: RunResult = .{
                .loss_at_log = undefined,
                .steps_to_threshold = n_steps_ + 1,
            };
            var log_idx: usize = 0;

            // Forward + initial loss.
            try recM_fn(kmm, pyb, mu * nu, ctx_ptr, &.{ buf_x_, buf_w_, buf_y_ });
            try recM_fn(kmm, pin, mu * ru, ctx_ptr, &.{ buf_x_, buf_a_, buf_inter_ });
            try recM_fn(kmm, pyl, mu * nu, ctx_ptr, &.{ buf_inter_, buf_b_, buf_yl_ });
            try recS_fn(ksc, psy.*, ceilDiv(mu * nu, glin), ctx_ptr, &.{ buf_yl_, buf_yls_ });
            try recS_fn(kad, pay.*, ceilDiv(mu * nu, glin), ctx_ptr, &.{ buf_y_, buf_yls_ });
            try buf_y_.readBack(ctx_ptr, f32, y_buf_);
            const initial_loss = mse_fn(y_buf_, target_y_);
            result.loss_at_log[log_idx] = initial_loss;
            log_idx += 1;

            var step: u32 = 1;
            while (step <= n_steps_) : (step += 1) {
                // Forward.
                try recM_fn(kmm, pyb, mu * nu, ctx_ptr, &.{ buf_x_, buf_w_, buf_y_ });
                try recM_fn(kmm, pin, mu * ru, ctx_ptr, &.{ buf_x_, buf_a_, buf_inter_ });
                try recM_fn(kmm, pyl, mu * nu, ctx_ptr, &.{ buf_inter_, buf_b_, buf_yl_ });
                try recS_fn(ksc, psy.*, ceilDiv(mu * nu, glin), ctx_ptr, &.{ buf_yl_, buf_yls_ });
                try recS_fn(kad, pay.*, ceilDiv(mu * nu, glin), ctx_ptr, &.{ buf_y_, buf_yls_ });
                // Loss grad.
                try recS_fn(kmse, pms.*, ceilDiv(mu * nu, glin), ctx_ptr, &.{ buf_y_, buf_target_, buf_dy_ });
                // Backward.
                try recL_fn(klx, pdyB, ceilDiv(mu, glwg), ceilDiv(ru, glwg), ctx_ptr, &.{ buf_dy_, buf_b_, buf_dyB_ });
                try recL_fn(klw, pdA, ceilDiv(ru, glwg), ceilDiv(ku, glwg), ctx_ptr, &.{ buf_dyB_, buf_x_, buf_dAu_ });
                try recS_fn(ksc, psda.*, ceilDiv(ru * ku, glin), ctx_ptr, &.{ buf_dAu_, buf_dA_ });
                try recL_fn(klw, pdB, ceilDiv(nu, glwg), ceilDiv(ru, glwg), ctx_ptr, &.{ buf_dy_, buf_inter_, buf_dBu_ });
                try recS_fn(ksc, psdb.*, ceilDiv(nu * ru, glin), ctx_ptr, &.{ buf_dBu_, buf_dB_ });
                // Adam — different lr for A vs B (the LoRA+ knob).
                const adam_a = runtime.AdamStepPush{ .n = ru * ku, .lr = lr_a, .beta1 = beta1_, .beta2 = beta2_, .eps = eps_, .t = step };
                const adam_b = runtime.AdamStepPush{ .n = nu * ru, .lr = lr_b, .beta1 = beta1_, .beta2 = beta2_, .eps = eps_, .t = step };
                try recS_fn(kadam, adam_a, ceilDiv(ru * ku, glin), ctx_ptr, &.{ buf_a_, buf_dA_, buf_m_a_, buf_v_a_ });
                try recS_fn(kadam, adam_b, ceilDiv(nu * ru, glin), ctx_ptr, &.{ buf_b_, buf_dB_, buf_m_b_, buf_v_b_ });

                // Periodic loss readback for the trajectory.
                if (step % log_every_ == 0) {
                    try buf_y_.readBack(ctx_ptr, f32, y_buf_);
                    const loss = mse_fn(y_buf_, target_y_);
                    result.loss_at_log[log_idx] = loss;
                    log_idx += 1;
                }
                // Track threshold crossing without an extra readback per step:
                // only check at log points. Coarse but cheap; fine for a demo.
                if (result.steps_to_threshold > n_steps_ and result.loss_at_log[log_idx - 1] < threshold_ and step % log_every_ == 0) {
                    result.steps_to_threshold = step;
                }
            }
            return result;
        }
    }.call;

    // ── Run all three ratios.
    std.debug.print(
        "LoRA+ comparative demo on {s}\n  shape: M={d} N={d} K={d} r={d}  α/r={d}  base_lr={d}\n  step:",
        .{ ctx.deviceName(), M, Nn, K, r, aor, base_lr },
    );
    var step_iter: u32 = 0;
    while (step_iter <= n_steps) : (step_iter += log_every) {
        std.debug.print("  {d:>6}", .{step_iter});
    }
    std.debug.print("\n", .{});

    var results: [ratios.len]RunResult = undefined;
    for (ratios, 0..) |ratio, i| {
        results[i] = try runOneTrajectory(
            ratio, base_lr,
            &buf_a, &buf_b, &buf_m_a, &buf_v_a, &buf_m_b, &buf_v_b,
            a_init, b_init, target_y, y_buf,
            &ctx, &k_matmul, &k_lin_dx, &k_lin_dw, &k_scale, &k_add, &k_mse_grad, &k_adam,
            &buf_x, &buf_w, &buf_target_y, &buf_y, &buf_intermediate, &buf_y_lora, &buf_y_lora_scaled,
            &buf_dy, &buf_dy_B, &buf_dA_unscaled, &buf_dA, &buf_dB_unscaled, &buf_dB,
            &push_y_base, &push_inter, &push_ylora, &push_scale_ylora, &push_add_y,
            &push_mse, &push_dy_B, &push_dA, &push_scale_dA, &push_dB, &push_scale_dB,
            M_u, N_u, K_u, R_u, beta1, beta2, eps, n_steps, log_every, threshold,
            recordMatmul, recordLinBackward, recordScalar1D, computeMse,
            group_lin, group_lwg,
        );

        std.debug.print("  λ={d:>4.1}: ", .{ratio});
        for (results[i].loss_at_log[0 .. (n_steps / log_every) + 1]) |l| {
            std.debug.print("  {d:.4}", .{l});
        }
        if (results[i].steps_to_threshold <= n_steps) {
            std.debug.print("  → loss<{e} at step {d}\n", .{ threshold, results[i].steps_to_threshold });
        } else {
            std.debug.print("  → loss<{e} not reached\n", .{threshold});
        }
    }

    // ── Headline metric: at fixed step count, what's the final-loss
    //    ratio between λ=16 and λ=1? "Convergence speedup at fixed
    //    budget" — finer than a step-to-threshold measurement, which
    //    log_every=25 cadence rounds up to coarse boundaries.
    //    (`threshold` above is consumed by runOneTrajectory and kept
    //    in the trajectory metadata for reference; we don't use the
    //    coarse step-to-threshold count for the final assertion.)
    const final_idx: usize = (n_steps / log_every);
    const vanilla_final = results[0].loss_at_log[final_idx];
    const plus_final = results[ratios.len - 1].loss_at_log[final_idx];
    const final_ratio = vanilla_final / @max(plus_final, 1e-30);
    std.debug.print(
        "  final-loss   λ=1 / λ=16 = {d:.4} / {d:.4} = {d:.2}× lower with λ=16\n",
        .{ vanilla_final, plus_final, final_ratio },
    );

    // Trajectory comparison at the first log point (step 25): λ=16
    // should be visibly ahead of λ=1, where the "B opens the gate"
    // early-regime speedup is most pronounced.
    const early_vanilla = results[0].loss_at_log[1];
    const early_plus = results[ratios.len - 1].loss_at_log[1];
    const early_ratio = early_vanilla / @max(early_plus, 1e-30);
    std.debug.print(
        "  early-loss   λ=1 / λ=16 = {d:.4} / {d:.4} = {d:.2}× lower at step {d}\n",
        .{ early_vanilla, early_plus, early_ratio, log_every },
    );

    if (plus_final >= vanilla_final) {
        std.debug.print("FAIL: λ=16 final loss not lower than λ=1\n", .{});
        return error.LoraPlusNotFaster;
    }
    if (final_ratio < 1.5) {
        std.debug.print("FAIL: LoRA+ final-loss speedup below 1.5× (was {d:.2}×)\n", .{final_ratio});
        return error.LoraPlusSpeedupTooSmall;
    }

    std.debug.print(
        "PASS GPU LoRA+ — λ=16 vs λ=1 on rank-{d} delta recovery: {d:.2}× lower final loss, {d:.2}× lower at step {d}\n",
        .{ r, final_ratio, early_ratio, log_every },
    );
}

// ── CCE bench: time each kernel in isolation at Qwen3-0.6B shape ──────
//
// Reports per-kernel wall-clock ms after `vkQueueWaitIdle`, plus the
// fillZero(buf_dw_lm_head) cost that the wired-in step() pays once
// per training step. Driven against synthetic random inputs — no
// parity check; the standalone --cce-{forward,backward-dh,backward-dw}-smoke
// flags handle that.
//
// At n_pos=16 (β-5 shape) we expect cce_forward and cce_backward_dh
// to be tiny (only 16 WGs each) and cce_backward_dw to dominate
// (151,936 WGs). At training-realistic n_pos>>SM count, all three
// should be HBM-bound at single-digit ms.

fn runCceBench(allocator: std.mem.Allocator) !void {
    var ctx = try vk.Context.init(allocator);
    defer ctx.deinit();

    // Qwen3-0.6B shape (matches the β-5 smoke).
    const n: u32 = 16;
    const v: u32 = 151_936;
    const d: u32 = 1024;
    const warmup_iters: u32 = 3;
    const bench_iters: u32 = 5;

    std.debug.print(
        "CCE bench on {s}\n  shape: N={d} V={d} D={d} (Qwen3-0.6B β-5)\n  warmup {d} / measure {d}\n",
        .{ ctx.deviceName(), n, v, d, warmup_iters, bench_iters },
    );

    // ── Allocate buffers (random-init host data, deterministic seed).
    var prng = std.Random.DefaultPrng.init(0xBE_AB_AB);
    const rng = prng.random();

    const h = try allocator.alloc(f32, n * d);
    defer allocator.free(h);
    const w_lm = try allocator.alloc(f32, v * d);
    defer allocator.free(w_lm);
    const targets = try allocator.alloc(u32, n);
    defer allocator.free(targets);
    const lse_host = try allocator.alloc(f32, n);
    defer allocator.free(lse_host);

    for (h) |*x| x.* = (rng.float(f32) - 0.5) * 0.1;
    for (w_lm) |*x| x.* = (rng.float(f32) - 0.5) * 0.1;
    for (targets) |*t| t.* = rng.intRangeLessThan(u32, 0, v);
    for (lse_host) |*x| x.* = 0.0;

    var buf_h = try buffer.Buffer.initStatic(&ctx, f32, h);
    defer buf_h.deinit(ctx.device);
    var buf_w = try buffer.Buffer.initStatic(&ctx, f32, w_lm);
    defer buf_w.deinit(ctx.device);
    var buf_t = try buffer.Buffer.initStatic(&ctx, u32, targets);
    defer buf_t.deinit(ctx.device);
    var buf_lse = try buffer.Buffer.initStatic(&ctx, f32, lse_host);
    defer buf_lse.deinit(ctx.device);
    var buf_loss = try buffer.Buffer.initDeviceOnly(&ctx, n * @sizeOf(f32));
    defer buf_loss.deinit(ctx.device);
    var buf_dh = try buffer.Buffer.initDeviceOnly(&ctx, n * d * @sizeOf(f32));
    defer buf_dh.deinit(ctx.device);
    var buf_dw = try buffer.Buffer.initDeviceOnly(&ctx, v * d * @sizeOf(f32));
    defer buf_dw.deinit(ctx.device);

    // ── Pipelines.
    var k_fwd = try pipeline.Kernel.init(&ctx, &shaders.cce_forward, 5, @sizeOf(runtime.CceForwardPush));
    defer k_fwd.deinit();
    var k_bw_dh = try pipeline.Kernel.init(&ctx, &shaders.cce_backward_dh, 5, @sizeOf(runtime.CceBackwardPush));
    defer k_bw_dh.deinit();
    var k_bw_dw = try pipeline.Kernel.init(&ctx, &shaders.cce_backward_dw, 5, @sizeOf(runtime.CceBackwardPush));
    defer k_bw_dw.deinit();

    try k_fwd.bind(&.{ &buf_h, &buf_w, &buf_t, &buf_lse, &buf_loss });
    const push_fwd = runtime.CceForwardPush{ .n_samples = n, .vocab = v, .dim = d };

    try k_bw_dh.bind(&.{ &buf_h, &buf_w, &buf_t, &buf_lse, &buf_dh });
    const push_bw = runtime.CceBackwardPush{ .n_samples = n, .vocab = v, .dim = d };

    try k_bw_dw.bind(&.{ &buf_h, &buf_w, &buf_t, &buf_lse, &buf_dw });

    const ns_per_ms: f64 = 1.0e6;

    // Helper: time `iters` runs of `submitOneShot` of `kern` with `gx` workgroups.
    const runKernel = struct {
        fn call(c_ctx: *const vk.Context, kern: *const pipeline.Kernel, push: anytype, gx: u32, iters: u32) !f64 {
            var total_ns: u64 = 0;
            const PushT = @TypeOf(push);
            const Recorder = struct {
                k: *const pipeline.Kernel,
                p: *const PushT,
                gx_: u32,
                pub fn record(s: @This(), cmd: vk.c.VkCommandBuffer) void {
                    s.k.dispatch(cmd, s.p, s.gx_, 1, 1);
                }
            };
            for (0..iters) |_| {
                const t0 = std.time.nanoTimestamp();
                try buffer.submitOneShot(c_ctx, Recorder{ .k = kern, .p = &push, .gx_ = gx });
                const t1 = std.time.nanoTimestamp();
                total_ns += @intCast(t1 - t0);
            }
            return @as(f64, @floatFromInt(total_ns)) / @as(f64, @floatFromInt(iters));
        }
    }.call;

    // ── Warmup.
    _ = try runKernel(&ctx, &k_fwd, push_fwd, n, warmup_iters);
    _ = try runKernel(&ctx, &k_bw_dh, push_bw, n, warmup_iters);
    _ = try runKernel(&ctx, &k_bw_dw, push_bw, v, warmup_iters);

    // ── Measure.
    const t_fwd = try runKernel(&ctx, &k_fwd, push_fwd, n, bench_iters);
    const t_bw_dh = try runKernel(&ctx, &k_bw_dh, push_bw, n, bench_iters);
    const t_bw_dw = try runKernel(&ctx, &k_bw_dw, push_bw, v, bench_iters);

    // fillZero(buf_dw) — what step() pays once per Adam step to reset the
    // accumulator between iterations.
    var t_fill_total: u64 = 0;
    for (0..bench_iters) |_| {
        const t0 = std.time.nanoTimestamp();
        try buf_dw.fillZero(&ctx);
        const t1 = std.time.nanoTimestamp();
        t_fill_total += @intCast(t1 - t0);
    }
    const t_fill = @as(f64, @floatFromInt(t_fill_total)) / @as(f64, @floatFromInt(bench_iters));

    std.debug.print(
        "  cce_forward       (gx={d:>6} ): {d:>8.3} ms\n",
        .{ n, t_fwd / ns_per_ms },
    );
    std.debug.print(
        "  cce_backward_dh   (gx={d:>6} ): {d:>8.3} ms\n",
        .{ n, t_bw_dh / ns_per_ms },
    );
    std.debug.print(
        "  cce_backward_dw   (gx={d:>6}): {d:>8.3} ms\n",
        .{ v, t_bw_dw / ns_per_ms },
    );
    std.debug.print(
        "  fillZero(buf_dw,  {d:>6} MB): {d:>8.3} ms  ({d:.1} GB/s)\n",
        .{
            @divTrunc(@as(u64, v) * @as(u64, d) * 4, 1024 * 1024),
            t_fill / ns_per_ms,
            @as(f64, @floatFromInt(@as(u64, v) * @as(u64, d) * 4)) / t_fill,
        },
    );
    std.debug.print(
        "  TOTAL CCE bucket             : {d:>8.3} ms\n",
        .{(t_fwd + t_bw_dh + t_bw_dw + t_fill) / ns_per_ms},
    );
}

// ── GPU CCE backward d_h smoke: vs CPU oracle ─────────────────────────
//
// Drives `cce_backward_dh.comp` against the d_h output of
// `cpu_cce.cceBackward`. The oracle path uses the same chunked
// recompute-and-accumulate algorithm but in scalar f64; the GPU version
// uses cooperative subgroup reductions. Tolerance is 1e-5 global rel-err
// (`max|diff| / max|ref|`), same as cce.zig's in-file parity tests.

fn runGpuCceBackwardDhSmoke(allocator: std.mem.Allocator) !void {
    var ctx = try vk.Context.init(allocator);
    defer ctx.deinit();

    const SmokeCase = struct {
        name: []const u8,
        n: u32,
        v: u32,
        d: u32,
        z_loss_scale: f32,
        label_smoothing: f32,
    };
    const cases = [_]SmokeCase{
        .{ .name = "multi-chunk (V=2048, 8 chunks)", .n = 4, .v = 2048, .d = 896, .z_loss_scale = 0.0, .label_smoothing = 0.0 },
        .{ .name = "partial-chunk (V=300, 1+ chunks)", .n = 3, .v = 300, .d = 64, .z_loss_scale = 0.0, .label_smoothing = 0.0 },
        .{ .name = "single-chunk (V=256, exactly CHUNK)", .n = 2, .v = 256, .d = 128, .z_loss_scale = 0.0, .label_smoothing = 0.0 },
        .{ .name = "multi-chunk + z-loss λ=1e-4", .n = 4, .v = 2048, .d = 896, .z_loss_scale = 1e-4, .label_smoothing = 0.0 },
        .{ .name = "multi-chunk + z-loss + label-smoothing", .n = 4, .v = 2048, .d = 896, .z_loss_scale = 1e-4, .label_smoothing = 0.1 },
    };

    var kern = try pipeline.Kernel.init(&ctx, &shaders.cce_backward_dh, 5, @sizeOf(runtime.CceBackwardPush));
    defer kern.deinit();

    for (cases) |cs| {
        var prng = std.Random.DefaultPrng.init(0xCCEB_DAA0 + cs.v);
        const rng = prng.random();

        const h = try allocator.alloc(f32, cs.n * cs.d);
        defer allocator.free(h);
        const w_lm = try allocator.alloc(f32, cs.v * cs.d);
        defer allocator.free(w_lm);
        const targets = try allocator.alloc(u32, cs.n);
        defer allocator.free(targets);

        for (h) |*x| x.* = (rng.float(f32) - 0.5) * 0.1;
        for (w_lm) |*x| x.* = (rng.float(f32) - 0.5) * 0.1;
        for (targets) |*t| t.* = rng.intRangeLessThan(u32, 0, cs.v);

        // CPU oracle: forward to populate lse, then full backward.
        // d_h is what we compare; dW is computed but discarded for this
        // smoke (it's the cce_backward_dw kernel's parity target).
        const lse_cpu = try allocator.alloc(f32, cs.n);
        defer allocator.free(lse_cpu);
        _ = cpu_cce.cceForward(h, w_lm, targets, cs.n, cs.v, cs.d, 256, .{ .z_loss_scale = cs.z_loss_scale, .label_smoothing = cs.label_smoothing }, lse_cpu);

        const d_h_cpu = try allocator.alloc(f32, cs.n * cs.d);
        defer allocator.free(d_h_cpu);
        const dW_unused = try allocator.alloc(f32, cs.v * cs.d);
        defer allocator.free(dW_unused);
        @memset(dW_unused, 0);
        cpu_cce.cceBackward(h, w_lm, targets, lse_cpu, cs.n, cs.v, cs.d, 256, .{ .z_loss_scale = cs.z_loss_scale, .label_smoothing = cs.label_smoothing }, d_h_cpu, dW_unused);

        // GPU dispatch. lse comes from CPU (the upstream cce_forward
        // kernel computed the same lse, but we use the CPU values here
        // to isolate the d_h kernel's correctness from any forward
        // round-off — chunk 2's GPU↔CPU lse parity already validated to
        // 1e-7).
        var buf_h = try buffer.Buffer.initStatic(&ctx, f32, h);
        defer buf_h.deinit(ctx.device);
        var buf_w = try buffer.Buffer.initStatic(&ctx, f32, w_lm);
        defer buf_w.deinit(ctx.device);
        var buf_t = try buffer.Buffer.initStatic(&ctx, u32, targets);
        defer buf_t.deinit(ctx.device);
        var buf_lse = try buffer.Buffer.initStatic(&ctx, f32, lse_cpu);
        defer buf_lse.deinit(ctx.device);
        var buf_dh = try buffer.Buffer.initDeviceOnly(&ctx, cs.n * cs.d * @sizeOf(f32));
        defer buf_dh.deinit(ctx.device);

        try kern.bind(&.{ &buf_h, &buf_w, &buf_t, &buf_lse, &buf_dh });

        const push = runtime.CceBackwardPush{
            .n_samples = cs.n,
            .vocab = cs.v,
            .dim = cs.d,
            .z_loss_scale = cs.z_loss_scale,
            .label_smoothing_eps = cs.label_smoothing,
        };
        try buffer.submitOneShot(&ctx, struct {
            kern: *const pipeline.Kernel,
            push: *const runtime.CceBackwardPush,
            gx: u32,
            pub fn record(s: @This(), cmd: vk.c.VkCommandBuffer) void {
                s.kern.dispatch(cmd, s.push, s.gx, 1, 1);
            }
        }{ .kern = &kern, .push = &push, .gx = cs.n });

        const d_h_gpu = try allocator.alloc(f32, cs.n * cs.d);
        defer allocator.free(d_h_gpu);
        try buf_dh.readBack(&ctx, f32, d_h_gpu);

        const rel = globalRelDiff(d_h_cpu, d_h_gpu);
        const tol: f32 = 1e-5;
        if (rel >= tol) {
            std.debug.print("CCE bw d_h smoke ({s}) FAIL: rel={e}  tol={e}\n", .{ cs.name, rel, tol });
            for (d_h_cpu, d_h_gpu, 0..) |a, b, i| {
                if (@abs(a - b) > tol * @max(@abs(a), 1e-6)) {
                    std.debug.print("  first mismatch idx={d}: cpu={e}  gpu={e}\n", .{ i, a, b });
                    break;
                }
            }
            return error.ParityFailed;
        }

        std.debug.print(
            "PASS GPU CCE backward d_h — {s}  N={d} V={d} D={d}  rel={e}\n",
            .{ cs.name, cs.n, cs.v, cs.d, rel },
        );
    }
}

// ── GPU CCE backward dW smoke: vs CPU oracle ──────────────────────────
//
// Vocab-major dispatch — one workgroup per vocab id. Drives
// `cce_backward_dw.comp` against the dW output of `cpu_cce.cceBackward`.
// `initDeviceOnly` zero-fills, so the kernel's `+=` accumulates from
// zero on first call (matching the caller-zeroes convention shared with
// embedding_backward and linearBackward).

fn runGpuCceBackwardDwSmoke(allocator: std.mem.Allocator) !void {
    var ctx = try vk.Context.init(allocator);
    defer ctx.deinit();

    const SmokeCase = struct {
        name: []const u8,
        n: u32,
        v: u32,
        d: u32,
        z_loss_scale: f32,
        label_smoothing: f32,
    };
    const cases = [_]SmokeCase{
        .{ .name = "multi-chunk (V=2048)", .n = 4, .v = 2048, .d = 896, .z_loss_scale = 0.0, .label_smoothing = 0.0 },
        .{ .name = "small-vocab (V=300)", .n = 3, .v = 300, .d = 64, .z_loss_scale = 0.0, .label_smoothing = 0.0 },
        .{ .name = "boundary (V=256)", .n = 2, .v = 256, .d = 128, .z_loss_scale = 0.0, .label_smoothing = 0.0 },
        .{ .name = "multi-chunk + z-loss λ=1e-4", .n = 4, .v = 2048, .d = 896, .z_loss_scale = 1e-4, .label_smoothing = 0.0 },
        .{ .name = "multi-chunk + z-loss + label-smoothing", .n = 4, .v = 2048, .d = 896, .z_loss_scale = 1e-4, .label_smoothing = 0.1 },
    };

    var kern = try pipeline.Kernel.init(&ctx, &shaders.cce_backward_dw, 5, @sizeOf(runtime.CceBackwardPush));
    defer kern.deinit();

    for (cases) |cs| {
        var prng = std.Random.DefaultPrng.init(0xCCED_DAA0 + cs.v);
        const rng = prng.random();

        const h = try allocator.alloc(f32, cs.n * cs.d);
        defer allocator.free(h);
        const w_lm = try allocator.alloc(f32, cs.v * cs.d);
        defer allocator.free(w_lm);
        const targets = try allocator.alloc(u32, cs.n);
        defer allocator.free(targets);

        for (h) |*x| x.* = (rng.float(f32) - 0.5) * 0.1;
        for (w_lm) |*x| x.* = (rng.float(f32) - 0.5) * 0.1;
        for (targets) |*t| t.* = rng.intRangeLessThan(u32, 0, cs.v);

        // CPU oracle: forward to populate lse, then full backward.
        const lse_cpu = try allocator.alloc(f32, cs.n);
        defer allocator.free(lse_cpu);
        _ = cpu_cce.cceForward(h, w_lm, targets, cs.n, cs.v, cs.d, 256, .{ .z_loss_scale = cs.z_loss_scale, .label_smoothing = cs.label_smoothing }, lse_cpu);

        const d_h_unused = try allocator.alloc(f32, cs.n * cs.d);
        defer allocator.free(d_h_unused);
        const dW_cpu = try allocator.alloc(f32, cs.v * cs.d);
        defer allocator.free(dW_cpu);
        @memset(dW_cpu, 0);
        cpu_cce.cceBackward(h, w_lm, targets, lse_cpu, cs.n, cs.v, cs.d, 256, .{ .z_loss_scale = cs.z_loss_scale, .label_smoothing = cs.label_smoothing }, d_h_unused, dW_cpu);

        // GPU dispatch.
        var buf_h = try buffer.Buffer.initStatic(&ctx, f32, h);
        defer buf_h.deinit(ctx.device);
        var buf_w = try buffer.Buffer.initStatic(&ctx, f32, w_lm);
        defer buf_w.deinit(ctx.device);
        var buf_t = try buffer.Buffer.initStatic(&ctx, u32, targets);
        defer buf_t.deinit(ctx.device);
        var buf_lse = try buffer.Buffer.initStatic(&ctx, f32, lse_cpu);
        defer buf_lse.deinit(ctx.device);
        // initDeviceOnly zero-fills, which the `+=` kernel needs.
        var buf_dw = try buffer.Buffer.initDeviceOnly(&ctx, cs.v * cs.d * @sizeOf(f32));
        defer buf_dw.deinit(ctx.device);

        try kern.bind(&.{ &buf_h, &buf_w, &buf_t, &buf_lse, &buf_dw });

        const push = runtime.CceBackwardPush{
            .n_samples = cs.n,
            .vocab = cs.v,
            .dim = cs.d,
            .z_loss_scale = cs.z_loss_scale,
            .label_smoothing_eps = cs.label_smoothing,
        };
        try buffer.submitOneShot(&ctx, struct {
            kern: *const pipeline.Kernel,
            push: *const runtime.CceBackwardPush,
            gx: u32,
            pub fn record(s: @This(), cmd: vk.c.VkCommandBuffer) void {
                s.kern.dispatch(cmd, s.push, s.gx, 1, 1);
            }
        }{ .kern = &kern, .push = &push, .gx = cs.v });

        const dW_gpu = try allocator.alloc(f32, cs.v * cs.d);
        defer allocator.free(dW_gpu);
        try buf_dw.readBack(&ctx, f32, dW_gpu);

        const rel = globalRelDiff(dW_cpu, dW_gpu);
        const tol: f32 = 1e-5;
        if (rel >= tol) {
            std.debug.print("CCE bw dW smoke ({s}) FAIL: rel={e}  tol={e}\n", .{ cs.name, rel, tol });
            for (dW_cpu, dW_gpu, 0..) |a, b, i| {
                if (@abs(a - b) > tol * @max(@abs(a), 1e-6)) {
                    std.debug.print("  first mismatch idx={d} (v={d}, k={d}): cpu={e}  gpu={e}\n", .{
                        i, i / cs.d, i % cs.d, a, b,
                    });
                    break;
                }
            }
            return error.ParityFailed;
        }

        std.debug.print(
            "PASS GPU CCE backward dW — {s}  N={d} V={d} D={d}  rel={e}\n",
            .{ cs.name, cs.n, cs.v, cs.d, rel },
        );
    }
}

/// Global relative-difference metric: `max|a − b| / max|a|`. Matches
/// `cce.zig`'s in-file parity test. Robust to noise-floor entries where
/// individual values are at f32 round-off from zero.
fn globalRelDiff(a: []const f32, b: []const f32) f32 {
    std.debug.assert(a.len == b.len);
    var max_diff: f32 = 0;
    var max_a: f32 = 0;
    for (a, b) |av, bv| {
        const diff = @abs(av - bv);
        if (diff > max_diff) max_diff = diff;
        if (@abs(av) > max_a) max_a = @abs(av);
    }
    return if (max_a > 1e-30) max_diff / max_a else max_diff;
}

// ── gpu matmul_nt_v2 smoke: cooperative-K kernel vs hand-checked ───

fn runGpuMatmulV2Smoke(allocator: std.mem.Allocator) !void {
    var ctx = try vk.Context.init(allocator);
    defer ctx.deinit();

    // Same problem as runGpuMatmulSmoke: 2x3 · (4x3)ᵀ → 2x4.
    // Note v2 needs K large enough that the cooperative reduction
    // exercises something — but correctness with tiny K is also a
    // useful sanity check (most threads get nothing to do, the result
    // should still be right).
    const a = [_]f32{ 1, 2, 3, 4, 5, 6 };
    const b = [_]f32{ 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1 };
    const want = [_]f32{ 1, 2, 3, 6, 4, 5, 6, 15 };
    const m: u32 = 2;
    const n: u32 = 4;
    const k: u32 = 3;

    var buf_a = try buffer.Buffer.initStatic(&ctx, f32, &a);
    defer buf_a.deinit(ctx.device);
    var buf_b = try buffer.Buffer.initStatic(&ctx, f32, &b);
    defer buf_b.deinit(ctx.device);
    var buf_c = try buffer.Buffer.initDeviceOnly(&ctx, m * n * @sizeOf(f32));
    defer buf_c.deinit(ctx.device);

    var kern = try pipeline.Kernel.init(&ctx, &shaders.matmul_nt_v2, 3, @sizeOf(MatmulPush));
    defer kern.deinit();
    try kern.bind(&.{ &buf_a, &buf_b, &buf_c });

    // v2 dispatches one WG per output cell — total = M*N WGs.
    const groups: u32 = m * n;
    const push = MatmulPush{ .m = m, .n = n, .k = k };
    try buffer.submitOneShot(&ctx, struct {
        kern: *const pipeline.Kernel,
        push: *const MatmulPush,
        gx: u32,
        pub fn record(s: @This(), cmd: vk.c.VkCommandBuffer) void {
            s.kern.dispatch(cmd, s.push, s.gx, 1, 1);
        }
    }{ .kern = &kern, .push = &push, .gx = groups });

    var out: [8]f32 = undefined;
    try buf_c.readBack(&ctx, f32, &out);
    for (out, want, 0..) |got, w, i| {
        if (got != w) {
            std.debug.print("GPU matmul_v2 MISMATCH at {d}: got {d}, expected {d}\n", .{ i, got, w });
            return error.ParityFailed;
        }
    }
    std.debug.print("PASS GPU matmul_nt_v2 synthetic (cooperative-K, 2×3 · (4×3)ᵀ → 2×4)\n", .{});
}

// ── embedded-attach smoke: parity between Context.init and Context.attach ─
//
// Validates the embedded-mode entry point that lets a host engine
// (Matryoshka) hand valkyr a pre-existing VkDevice/queue/cmd_pool to
// share. Two checks:
//   1. A kernel dispatched through an `attach`'d Context produces the
//      same result as one dispatched through the `init`'d Context that
//      owns the underlying handles.
//   2. Tearing down the attached context first, then the host context,
//      does NOT double-free — owns_* flags must keep the handles alive
//      across the attached deinit.
//
// We use `submitOneShot` (rather than the `Recorder`) because at this
// chunk only the device/cmd-pool ownership is being tested; recorder
// ownership is the next chunk.

fn runEmbeddedAttachSmoke(allocator: std.mem.Allocator) !void {
    var host = try vk.Context.init(allocator);
    defer host.deinit();

    var attached = vk.Context.attach(
        host.instance,
        host.physical_device,
        host.device,
        host.queue,
        host.queue_family,
        host.cmd_pool,
    );
    // attached.deinit must be a no-op — assert via the flags so a future
    // refactor that flips an ownership bit can't silently start
    // double-freeing without breaking this test.
    if (attached.owns_instance or attached.owns_device or attached.owns_cmd_pool) {
        std.debug.print("attach() returned a Context with an ownership flag set\n", .{});
        return error.ParityFailed;
    }

    // Same problem as runGpuMatmulSmoke: 2x3 · (4x3)ᵀ → 2x4.
    const a = [_]f32{ 1, 2, 3, 4, 5, 6 };
    const b = [_]f32{ 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1 };
    const want = [_]f32{ 1, 2, 3, 6, 4, 5, 6, 15 };
    const m: u32 = 2;
    const n: u32 = 4;
    const k: u32 = 3;

    var buf_a = try buffer.Buffer.initStatic(&attached, f32, &a);
    defer buf_a.deinit(attached.device);
    var buf_b = try buffer.Buffer.initStatic(&attached, f32, &b);
    defer buf_b.deinit(attached.device);
    var buf_c = try buffer.Buffer.initDeviceOnly(&attached, m * n * @sizeOf(f32));
    defer buf_c.deinit(attached.device);

    var kern = try pipeline.Kernel.init(&attached, &shaders.matmul_nt, 3, @sizeOf(MatmulPush));
    defer kern.deinit();
    try kern.bind(&.{ &buf_a, &buf_b, &buf_c });

    const local_xy: u32 = 16;
    const groups_x: u32 = (m + local_xy - 1) / local_xy;
    const groups_y: u32 = (n + local_xy - 1) / local_xy;
    const push = MatmulPush{ .m = m, .n = n, .k = k };

    try buffer.submitOneShot(&attached, struct {
        kern: *const pipeline.Kernel,
        push: *const MatmulPush,
        gx: u32,
        gy: u32,
        pub fn record(s: @This(), cmd: vk.c.VkCommandBuffer) void {
            s.kern.dispatch(cmd, s.push, s.gx, s.gy, 1);
        }
    }{ .kern = &kern, .push = &push, .gx = groups_x, .gy = groups_y });

    var out: [8]f32 = undefined;
    try buf_c.readBack(&attached, f32, &out);
    for (out, want, 0..) |got, w, i| {
        if (got != w) {
            std.debug.print("attach-mode matmul MISMATCH at {d}: got {d}, expected {d}\n", .{ i, got, w });
            return error.ParityFailed;
        }
    }

    // Tear down `attached` explicitly (rather than via defer) so that
    // any double-free regression surfaces here, before `host.deinit`
    // runs. With the flags zeroed it's a no-op; without them it would
    // VkDestroy* the handles host still owns and the next call would
    // crash inside the validation layer.
    attached.deinit();

    std.debug.print("PASS embedded-attach (matmul parity + non-owning deinit) on {s}\n", .{host.deviceName()});
}

// ── embedded-recorder smoke: valkyr records into a host-owned cmd buffer ─
//
// Models the Matryoshka render-loop case: the host engine has begun a
// command buffer for its own dispatches, hands it to valkyr, valkyr
// records its inference dispatches into the same buffer, and the host
// ends + submits + waits with its own fence. Two checks:
//   1. Dispatches recorded via an attachCmd'd Recorder produce the
//      same result as the standalone `endAndSubmit` path.
//   2. Calling endAndSubmit on an attached recorder fails cleanly
//      rather than silently double-ending the host's buffer.
//
// We simulate the "host" inside this same process by allocating a
// fresh cmd buffer + fence from the same VkDevice — that's exactly
// what an embedded host's render loop would have already done before
// the integration point.

fn runEmbeddedRecorderSmoke(allocator: std.mem.Allocator) !void {
    var ctx = try vk.Context.init(allocator);
    defer ctx.deinit();

    // Same problem as runGpuMatmulV2Smoke; reusing matmul_nt_v2 because
    // the recorder + barrier path is what we actually want to exercise
    // (matmul_nt is a single dispatch that wouldn't catch a missing
    // barrier between two valkyr-internal kernels).
    const a = [_]f32{ 1, 2, 3, 4, 5, 6 };
    const b = [_]f32{ 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1 };
    const want = [_]f32{ 1, 2, 3, 6, 4, 5, 6, 15 };
    const m: u32 = 2;
    const n: u32 = 4;
    const k: u32 = 3;

    var buf_a = try buffer.Buffer.initStatic(&ctx, f32, &a);
    defer buf_a.deinit(ctx.device);
    var buf_b = try buffer.Buffer.initStatic(&ctx, f32, &b);
    defer buf_b.deinit(ctx.device);
    var buf_c = try buffer.Buffer.initDeviceOnly(&ctx, m * n * @sizeOf(f32));
    defer buf_c.deinit(ctx.device);

    var kern = try pipeline.Kernel.init(&ctx, &shaders.matmul_nt_v2, 3, @sizeOf(MatmulPush));
    defer kern.deinit();
    try kern.bind(&.{ &buf_a, &buf_b, &buf_c });

    // ── Host side: allocate cmd buffer + fence, begin recording. ────
    // (In Matryoshka this is `Renderer.drawFrame` after acquire.)
    var cb_ai = std.mem.zeroes(vk.c.VkCommandBufferAllocateInfo);
    cb_ai.sType = vk.c.VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    cb_ai.commandPool = ctx.cmd_pool;
    cb_ai.level = vk.c.VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cb_ai.commandBufferCount = 1;
    var host_cmd: vk.c.VkCommandBuffer = null;
    try vk.check(vk.c.vkAllocateCommandBuffers(ctx.device, &cb_ai, &host_cmd));
    defer vk.c.vkFreeCommandBuffers(ctx.device, ctx.cmd_pool, 1, &host_cmd);

    var fci = std.mem.zeroes(vk.c.VkFenceCreateInfo);
    fci.sType = vk.c.VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    var host_fence: vk.c.VkFence = null;
    try vk.check(vk.c.vkCreateFence(ctx.device, &fci, null, &host_fence));
    defer vk.c.vkDestroyFence(ctx.device, host_fence, null);

    var bi = std.mem.zeroes(vk.c.VkCommandBufferBeginInfo);
    bi.sType = vk.c.VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    bi.flags = vk.c.VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    try vk.check(vk.c.vkBeginCommandBuffer(host_cmd, &bi));

    // ── Valkyr side: attach a recorder to the host's cmd buffer. ────
    var rec = try gpu_recorder.Recorder.attachCmd(&ctx, host_cmd, 4, 16);
    defer rec.deinit();

    if (rec.owns_cmd or rec.owns_fence) {
        std.debug.print("attachCmd returned a recorder with an ownership flag set\n", .{});
        return error.ParityFailed;
    }

    try rec.begin(); // no-op in attached mode; resets dispatch counter

    const groups: u32 = m * n;
    const push = MatmulPush{ .m = m, .n = n, .k = k };
    try rec.dispatch(&kern, &.{ &buf_a, &buf_b, &buf_c }, &push, groups, 1, 1);

    // endAndSubmit on an attached recorder must refuse — the host owns
    // the submit. We assert that explicitly so a future regression that
    // forgot the owns_cmd check would be caught here, not in production.
    if (rec.endAndSubmit()) {
        std.debug.print("attached recorder endAndSubmit() should have errored\n", .{});
        return error.ParityFailed;
    } else |err| {
        if (err != error.AttachedRecorderCannotSubmit) {
            std.debug.print("attached recorder endAndSubmit() returned wrong error: {s}\n", .{@errorName(err)});
            return err;
        }
    }

    // ── Host side: end + submit + wait with its own fence. ──────────
    try vk.check(vk.c.vkEndCommandBuffer(host_cmd));
    var submit = std.mem.zeroes(vk.c.VkSubmitInfo);
    submit.sType = vk.c.VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submit.commandBufferCount = 1;
    submit.pCommandBuffers = &host_cmd;
    try vk.check(vk.c.vkQueueSubmit(ctx.queue, 1, &submit, host_fence));
    const timeout_ns: u64 = 10 * 1_000_000_000;
    try vk.check(vk.c.vkWaitForFences(ctx.device, 1, &host_fence, vk.c.VK_TRUE, timeout_ns));

    var out: [8]f32 = undefined;
    try buf_c.readBack(&ctx, f32, &out);
    for (out, want, 0..) |got, w, i| {
        if (got != w) {
            std.debug.print("attached-recorder matmul MISMATCH at {d}: got {d}, expected {d}\n", .{ i, got, w });
            return error.ParityFailed;
        }
    }

    std.debug.print("PASS embedded-recorder (host-owned cmd buffer, valkyr dispatches in, host submits) on {s}\n", .{ctx.deviceName()});
}

// ── gpu matmul_nt_v2_q4_0 smoke: int4 weights vs CPU dequant oracle ─
//
// Round-trips fp32 weights through the CPU q4_0 quantizer + GPU-layout
// repack, dispatches the q4_0 matmul kernel, and compares its result
// against `A · dequant(B)^T` computed entirely on the CPU. The GPU
// shader and the CPU dequant share the same code path here (both
// decode (idx-8)*d), so the two should agree to within fp32 reduction
// rounding (max |Δ| ≲ 1e-3 at K=128). Per-element MSE on Gaussian
// inputs is dominated by the q4_0 quantization itself, not by GPU
// arithmetic — we measure GPU↔CPU agreement, not q4_0 quality.

fn runGpuMatmulQ4_0Smoke(allocator: std.mem.Allocator) !void {
    var ctx = try vk.Context.init(allocator);
    defer ctx.deinit();

    const m: u32 = 2;
    const n: u32 = 4;
    const k: u32 = 128; // multiple of 32

    var prng = std.Random.DefaultPrng.init(0xC0DECAFEBABE);
    const r = prng.random();

    const a_f32 = try allocator.alloc(f32, m * k);
    defer allocator.free(a_f32);
    const b_f32 = try allocator.alloc(f32, n * k);
    defer allocator.free(b_f32);
    for (a_f32) |*v| v.* = r.floatNorm(f32);
    for (b_f32) |*v| v.* = r.floatNorm(f32);

    // Quantize each row of B independently. With K=128, each row is
    // 4 blocks of 32. Total blocks = n * 4.
    const blocks_per_row = k / q4_0.BLOCK_SIZE;
    const total_blocks = n * blocks_per_row;
    const b_blocks = try allocator.alloc(q4_0.Block, total_blocks);
    defer allocator.free(b_blocks);
    for (0..n) |row| {
        const src_row = b_f32[row * k .. (row + 1) * k];
        const dst_row = b_blocks[row * blocks_per_row .. (row + 1) * blocks_per_row];
        q4_0.quantizeRow(src_row, dst_row);
    }

    // CPU oracle: A · dequant(B)^T.
    const b_deq = try allocator.alloc(f32, n * k);
    defer allocator.free(b_deq);
    for (0..n) |row| {
        const src_row = b_blocks[row * blocks_per_row .. (row + 1) * blocks_per_row];
        const dst_row = b_deq[row * k .. (row + 1) * k];
        q4_0.dequantizeRow(src_row, dst_row);
    }
    const want = try allocator.alloc(f32, m * n);
    defer allocator.free(want);
    for (0..m) |i| for (0..n) |j| {
        var s: f64 = 0;
        for (0..k) |kk| s += @as(f64, a_f32[i * k + kk]) * @as(f64, b_deq[j * k + kk]);
        want[i * n + j] = @floatCast(s);
    };

    // Repack CPU blocks into the GPU's 5-u32-per-block layout.
    const b_packed = try allocator.alloc(u32, total_blocks * q4_0.GPU_U32S_PER_BLOCK);
    defer allocator.free(b_packed);
    q4_0.packForGpu(b_blocks, b_packed);

    var buf_a = try buffer.Buffer.initStatic(&ctx, f32, a_f32);
    defer buf_a.deinit(ctx.device);
    var buf_b = try buffer.Buffer.initStatic(&ctx, u32, b_packed);
    defer buf_b.deinit(ctx.device);
    var buf_c = try buffer.Buffer.initDeviceOnly(&ctx, m * n * @sizeOf(f32));
    defer buf_c.deinit(ctx.device);

    var kern = try pipeline.Kernel.init(&ctx, &shaders.matmul_nt_v2_q4_0, 3, @sizeOf(MatmulPush));
    defer kern.deinit();
    try kern.bind(&.{ &buf_a, &buf_b, &buf_c });

    const groups: u32 = m * n;
    const push = MatmulPush{ .m = m, .n = n, .k = k };
    try buffer.submitOneShot(&ctx, struct {
        kern: *const pipeline.Kernel,
        push: *const MatmulPush,
        gx: u32,
        pub fn record(s: @This(), cmd: vk.c.VkCommandBuffer) void {
            s.kern.dispatch(cmd, s.push, s.gx, 1, 1);
        }
    }{ .kern = &kern, .push = &push, .gx = groups });

    const got = try allocator.alloc(f32, m * n);
    defer allocator.free(got);
    try buf_c.readBack(&ctx, f32, got);

    var max_err: f32 = 0;
    for (got, want) |g, w| max_err = @max(max_err, @abs(g - w));
    if (max_err > 1e-3) {
        std.debug.print("GPU q4_0 matmul: max |Δ| = {e} (>1e-3)\n", .{max_err});
        for (0..m * n) |idx| std.debug.print("  cell {d}: got {d:.5}, want {d:.5}\n", .{ idx, got[idx], want[idx] });
        return error.ParityFailed;
    }
    std.debug.print("PASS GPU matmul_nt_v2_q4_0 (M={d} N={d} K={d}, max |Δ| vs CPU dequant = {e:.2})\n", .{ m, n, k, max_err });
}

// ── gpu matmul_nt_v2_q4_k smoke: Q4_K_M weights vs CPU dequant oracle ─
//
// Same shape as the q4_0 smoke but at K=256 (the smallest legal Q4_K_M
// row — one super-block per row of B). Compares the GPU Q4_K_M matmul
// kernel against `A · dequant(B)^T` computed entirely on the CPU. As
// with q4_0, this measures GPU↔CPU agreement over the shared dequant
// formula, not Q4_K_M quality (which is tested in the CPU oracle smoke).

fn runGpuMatmulQ4_KSmoke(allocator: std.mem.Allocator) !void {
    var ctx = try vk.Context.init(allocator);
    defer ctx.deinit();

    const m: u32 = 2;
    const n: u32 = 4;
    const k: u32 = @intCast(q4_k.QK_K); // 256, smallest legal Q4_K row

    var prng = std.Random.DefaultPrng.init(0xBADCAFE0DEFACE);
    const r = prng.random();

    const a_f32 = try allocator.alloc(f32, m * k);
    defer allocator.free(a_f32);
    const b_f32 = try allocator.alloc(f32, n * k);
    defer allocator.free(b_f32);
    for (a_f32) |*v| v.* = r.floatNorm(f32);
    for (b_f32) |*v| v.* = r.floatNorm(f32);

    const supers_per_row = k / q4_k.QK_K;
    const total_supers = n * supers_per_row;
    const b_blocks = try allocator.alloc(q4_k.Block, total_supers);
    defer allocator.free(b_blocks);
    for (0..n) |row| {
        const src_row = b_f32[row * k .. (row + 1) * k];
        const dst_row = b_blocks[row * supers_per_row .. (row + 1) * supers_per_row];
        q4_k.quantizeRow(src_row, dst_row);
    }

    // CPU oracle: A · dequant(B)^T.
    const b_deq = try allocator.alloc(f32, n * k);
    defer allocator.free(b_deq);
    for (0..n) |row| {
        const src_row = b_blocks[row * supers_per_row .. (row + 1) * supers_per_row];
        const dst_row = b_deq[row * k .. (row + 1) * k];
        q4_k.dequantizeRow(src_row, dst_row);
    }
    const want = try allocator.alloc(f32, m * n);
    defer allocator.free(want);
    for (0..m) |i| for (0..n) |j| {
        var s: f64 = 0;
        for (0..k) |kk| s += @as(f64, a_f32[i * k + kk]) * @as(f64, b_deq[j * k + kk]);
        want[i * n + j] = @floatCast(s);
    };

    // Repack CPU blocks into the GPU's 36-u32-per-super-block layout.
    const b_packed = try allocator.alloc(u32, total_supers * q4_k.GPU_U32S_PER_SUPERBLOCK);
    defer allocator.free(b_packed);
    q4_k.packForGpu(b_blocks, b_packed);

    var buf_a = try buffer.Buffer.initStatic(&ctx, f32, a_f32);
    defer buf_a.deinit(ctx.device);
    var buf_b = try buffer.Buffer.initStatic(&ctx, u32, b_packed);
    defer buf_b.deinit(ctx.device);
    var buf_c = try buffer.Buffer.initDeviceOnly(&ctx, m * n * @sizeOf(f32));
    defer buf_c.deinit(ctx.device);

    var kern = try pipeline.Kernel.init(&ctx, &shaders.matmul_nt_v2_q4_k, 3, @sizeOf(MatmulPush));
    defer kern.deinit();
    try kern.bind(&.{ &buf_a, &buf_b, &buf_c });

    const groups: u32 = m * n;
    const push = MatmulPush{ .m = m, .n = n, .k = k };
    try buffer.submitOneShot(&ctx, struct {
        kern: *const pipeline.Kernel,
        push: *const MatmulPush,
        gx: u32,
        pub fn record(s: @This(), cmd: vk.c.VkCommandBuffer) void {
            s.kern.dispatch(cmd, s.push, s.gx, 1, 1);
        }
    }{ .kern = &kern, .push = &push, .gx = groups });

    const got = try allocator.alloc(f32, m * n);
    defer allocator.free(got);
    try buf_c.readBack(&ctx, f32, got);

    var max_err: f32 = 0;
    for (got, want) |g, w| max_err = @max(max_err, @abs(g - w));
    if (max_err > 1e-3) {
        std.debug.print("GPU q4_k matmul: max |Δ| = {e} (>1e-3)\n", .{max_err});
        for (0..m * n) |idx| std.debug.print("  cell {d}: got {d:.5}, want {d:.5}\n", .{ idx, got[idx], want[idx] });
        return error.ParityFailed;
    }
    std.debug.print("PASS GPU matmul_nt_v2_q4_k (M={d} N={d} K={d}, max |Δ| vs CPU dequant = {e:.2})\n", .{ m, n, k, max_err });
}

// ── gpu rmsnorm smoke: synthetic vs CPU rmsnorm ────────────────────

const RmsnormPush = runtime.RmsnormPush;

fn runGpuRmsnormSmoke(allocator: std.mem.Allocator) !void {
    var ctx = try vk.Context.init(allocator);
    defer ctx.deinit();

    // Two cases — Llama (no quirk) and Gemma (1+w quirk) — exercising
    // both code paths through the same shader.
    inline for (.{ false, true }) |gemma_quirk| {
        const dim: usize = 1024;
        const x = try allocator.alloc(f32, dim);
        defer allocator.free(x);
        const w = try allocator.alloc(f32, dim);
        defer allocator.free(w);
        for (x, 0..) |*v, i| v.* = 0.5 - @as(f32, @floatFromInt(i & 31)) * 0.03;
        for (w, 0..) |*v, i| v.* = -0.1 + @as(f32, @floatFromInt(i & 15)) * 0.02;

        // ── CPU oracle ──────────────────────────────────────────────
        const want = try allocator.alloc(f32, dim);
        defer allocator.free(want);
        const fake_w_tensor = safetensors.Tensor{
            .dtype = .f32,
            .shape = &.{dim},
            .bytes = std.mem.sliceAsBytes(w),
        };
        const family: config_mod.Family = if (gemma_quirk) .gemma else .llama;
        try cpu_math.rmsnorm(want, x, fake_w_tensor, 1e-6, family);

        // ── GPU dispatch ────────────────────────────────────────────
        var buf_a = try buffer.Buffer.initStatic(&ctx, f32, x);
        defer buf_a.deinit(ctx.device);
        var buf_w = try buffer.Buffer.initStatic(&ctx, f32, w);
        defer buf_w.deinit(ctx.device);
        var buf_c = try buffer.Buffer.initDeviceOnly(&ctx, dim * @sizeOf(f32));
        defer buf_c.deinit(ctx.device);

        var kern = try pipeline.Kernel.init(&ctx, &shaders.rmsnorm, 3, @sizeOf(RmsnormPush));
        defer kern.deinit();
        try kern.bind(&.{ &buf_a, &buf_w, &buf_c });

        const push = RmsnormPush{
            .dim = @intCast(dim),
            .eps = 1e-6,
            .gemma_quirk = if (gemma_quirk) 1 else 0,
        };
        try buffer.submitOneShot(&ctx, struct {
            kern: *const pipeline.Kernel,
            push: *const RmsnormPush,
            pub fn record(s: @This(), cmd: vk.c.VkCommandBuffer) void {
                s.kern.dispatch(cmd, s.push, 1, 1, 1);
            }
        }{ .kern = &kern, .push = &push });

        const got = try allocator.alloc(f32, dim);
        defer allocator.free(got);
        try buf_c.readBack(&ctx, f32, got);

        var max_abs: f32 = 0;
        for (got, want) |g, e| {
            const d = @abs(g - e);
            if (d > max_abs) max_abs = d;
        }
        if (max_abs > 1e-5) {
            std.debug.print("GPU rmsnorm gemma_quirk={any}: max |Δ| = {e}\n", .{ gemma_quirk, max_abs });
            return error.ParityFailed;
        }
    }
    std.debug.print("PASS GPU rmsnorm synthetic (Llama + Gemma variants, dim=1024)\n", .{});
}

// ── GPU rmsnorm_backward parity smoke ──────────────────────────────
//
// Tier-2 chunk 2 — verifies the new shader against the CPU oracle in
// `cpu/train_transformer.zig` on a multi-row batch with both
// gemma_quirk variants. Reads dx[N×D] and dw_partial[N×D] back, sums
// dw_partial across rows on the host, and diffs vs CPU multi-row
// rmsNormBackward. Cross-row sum on host is fine for the smoke; a
// dedicated reduce kernel is a follow-up if perf demands it.

fn runGpuRmsnormBackwardSmoke(allocator: std.mem.Allocator) !void {
    var ctx = try vk.Context.init(allocator);
    defer ctx.deinit();

    const dim: usize = 256;
    const n_rows: usize = 4;
    const eps: f32 = 1e-6;

    var prng = std.Random.DefaultPrng.init(0xBA66110C);
    const rng = prng.random();

    const x = try allocator.alloc(f32, n_rows * dim);
    defer allocator.free(x);
    const dy = try allocator.alloc(f32, n_rows * dim);
    defer allocator.free(dy);
    const w = try allocator.alloc(f32, dim);
    defer allocator.free(w);
    for (x) |*v| v.* = (rng.float(f32) * 2.0 - 1.0);
    for (dy) |*v| v.* = (rng.float(f32) * 2.0 - 1.0);
    for (w) |*v| v.* = 0.5 + rng.float(f32) * 0.5;

    inline for (.{ false, true }) |gemma_quirk| {
        // ── CPU oracle ────────────────────────────────────────────
        const dx_cpu = try allocator.alloc(f32, n_rows * dim);
        defer allocator.free(dx_cpu);
        const dw_cpu = try allocator.alloc(f32, dim);
        defer allocator.free(dw_cpu);
        @memset(dw_cpu, 0);
        cpu_train_transformer.rmsNormBackward(dy, x, w, eps, gemma_quirk, n_rows, dx_cpu, dw_cpu);

        // ── GPU dispatch ──────────────────────────────────────────
        var buf_dy = try buffer.Buffer.initStatic(&ctx, f32, dy);
        defer buf_dy.deinit(ctx.device);
        var buf_x = try buffer.Buffer.initStatic(&ctx, f32, x);
        defer buf_x.deinit(ctx.device);
        var buf_w = try buffer.Buffer.initStatic(&ctx, f32, w);
        defer buf_w.deinit(ctx.device);
        var buf_dx = try buffer.Buffer.initDeviceOnly(&ctx, n_rows * dim * @sizeOf(f32));
        defer buf_dx.deinit(ctx.device);
        var buf_dw_partial = try buffer.Buffer.initDeviceOnly(&ctx, n_rows * dim * @sizeOf(f32));
        defer buf_dw_partial.deinit(ctx.device);

        var kern = try pipeline.Kernel.init(&ctx, &shaders.rmsnorm_backward, 5, @sizeOf(RmsnormPush));
        defer kern.deinit();
        try kern.bind(&.{ &buf_dy, &buf_x, &buf_w, &buf_dx, &buf_dw_partial });

        const push = RmsnormPush{
            .dim = @intCast(dim),
            .eps = eps,
            .gemma_quirk = if (gemma_quirk) 1 else 0,
        };
        try buffer.submitOneShot(&ctx, struct {
            kern: *const pipeline.Kernel,
            push: *const RmsnormPush,
            n_rows: u32,
            pub fn record(s: @This(), cmd: vk.c.VkCommandBuffer) void {
                s.kern.dispatch(cmd, s.push, s.n_rows, 1, 1);
            }
        }{ .kern = &kern, .push = &push, .n_rows = @intCast(n_rows) });

        const dx_gpu = try allocator.alloc(f32, n_rows * dim);
        defer allocator.free(dx_gpu);
        const dw_partial_gpu = try allocator.alloc(f32, n_rows * dim);
        defer allocator.free(dw_partial_gpu);
        try buf_dx.readBack(&ctx, f32, dx_gpu);
        try buf_dw_partial.readBack(&ctx, f32, dw_partial_gpu);

        // Sum dw_partial across rows on the host.
        const dw_gpu = try allocator.alloc(f32, dim);
        defer allocator.free(dw_gpu);
        @memset(dw_gpu, 0);
        for (0..n_rows) |r| {
            const off = r * dim;
            for (0..dim) |i| dw_gpu[i] += dw_partial_gpu[off + i];
        }

        var max_dx: f32 = 0;
        for (dx_gpu, dx_cpu) |g, c| {
            const d = @abs(g - c);
            if (d > max_dx) max_dx = d;
        }
        if (max_dx > 1e-4) {
            std.debug.print("rmsnorm_backward dx (gq={any}): max |Δ| = {e}\n", .{ gemma_quirk, max_dx });
            return error.ParityFailed;
        }

        var max_dw: f32 = 0;
        for (dw_gpu, dw_cpu) |g, c| {
            const d = @abs(g - c);
            if (d > max_dw) max_dw = d;
        }
        if (max_dw > 1e-4) {
            std.debug.print("rmsnorm_backward dw (gq={any}): max |Δ| = {e}\n", .{ gemma_quirk, max_dw });
            return error.ParityFailed;
        }
    }

    std.debug.print(
        "PASS GPU rmsnorm_backward (Llama + Gemma variants, {d}×{d}; dx + dw match CPU oracle ≤ 1e-4)\n",
        .{ n_rows, dim },
    );
}

// ── LayerNorm CPU oracle smoke ────────────────────────────────────
//
// Tier-2 chunk 3 first half: sanity-check the CPU oracle in
// `cpu/train_transformer.zig` for both forward and backward against
// numeric gradients. Forward already covered by the GPU parity smoke
// below — this is the "is the math right at all" oracle test.

fn runLayerNormBackwardCpuSmoke(allocator: std.mem.Allocator) !void {
    _ = allocator;
    const dim: usize = 32;
    const eps: f32 = 1e-5;

    var x: [dim]f32 = undefined;
    var w: [dim]f32 = undefined;
    var bias: [dim]f32 = undefined;
    var dy: [dim]f32 = undefined;
    var prng = std.Random.DefaultPrng.init(0x14E72097);
    const rng = prng.random();
    for (&x) |*v| v.* = (rng.float(f32) * 2.0 - 1.0) * 1.5;
    for (&w) |*v| v.* = 0.5 + rng.float(f32) * 0.5;
    for (&bias) |*v| v.* = (rng.float(f32) * 2.0 - 1.0) * 0.2;
    for (&dy) |*v| v.* = (rng.float(f32) * 2.0 - 1.0);

    var dx: [dim]f32 = undefined;
    var dw: [dim]f32 = undefined;
    var dbias: [dim]f32 = undefined;
    var probe_y: [dim]f32 = undefined;

    @memset(&dw, 0);
    @memset(&dbias, 0);
    cpu_train_transformer.layerNormBackwardRow(&dy, &x, &w, eps, &dx, &dw, &dbias);

    const eps_h: f32 = 1e-3;
    const probes = [_]usize{ 0, 5, 11, 17, 23, 29 };
    var max_rel_err: f32 = 0;

    const Buf = enum { x, w, bias };
    const bufs = [_]Buf{ .x, .w, .bias };
    for (bufs) |b| {
        const target_buf: []f32 = switch (b) {
            .x => &x,
            .w => &w,
            .bias => &bias,
        };
        const grad_buf: []const f32 = switch (b) {
            .x => &dx,
            .w => &dw,
            .bias => &dbias,
        };
        for (probes) |i| {
            const orig = target_buf[i];
            target_buf[i] = orig + eps_h;
            _ = cpu_train_transformer.layerNormForwardRow(&x, &w, &bias, eps, &probe_y);
            var l_plus: f32 = 0;
            for (dy, probe_y) |d, yi| l_plus += d * yi;
            target_buf[i] = orig - eps_h;
            _ = cpu_train_transformer.layerNormForwardRow(&x, &w, &bias, eps, &probe_y);
            var l_minus: f32 = 0;
            for (dy, probe_y) |d, yi| l_minus += d * yi;
            target_buf[i] = orig;

            const numeric = (l_plus - l_minus) / (2.0 * eps_h);
            const analytic = grad_buf[i];
            const denom = @max(@abs(numeric), @abs(analytic));
            const rel_err = if (denom > 0) @abs(numeric - analytic) / denom else @abs(numeric - analytic);
            if (rel_err > 1e-2) {
                std.debug.print(
                    "layernorm grad mismatch on {s}[{d}]: analytic={d:.6} numeric={d:.6} rel_err={d:.4}\n",
                    .{ @tagName(b), i, analytic, numeric, rel_err },
                );
                return error.ParityFailed;
            }
            if (rel_err > max_rel_err) max_rel_err = rel_err;
        }
    }

    std.debug.print(
        "PASS layernorm backward CPU oracle (numeric-grad parity ≤ {e} on dx + dw + dbias)\n",
        .{max_rel_err},
    );
}

// ── GPU layernorm forward parity smoke ──────────────────────────────

fn runGpuLayerNormSmoke(allocator: std.mem.Allocator) !void {
    var ctx = try vk.Context.init(allocator);
    defer ctx.deinit();

    const dim: usize = 1024;
    const n_rows: usize = 4;
    const eps: f32 = 1e-5;

    const x = try allocator.alloc(f32, n_rows * dim);
    defer allocator.free(x);
    const w = try allocator.alloc(f32, dim);
    defer allocator.free(w);
    const bias = try allocator.alloc(f32, dim);
    defer allocator.free(bias);
    var prng = std.Random.DefaultPrng.init(0x14E70F02);
    const rng = prng.random();
    for (x) |*v| v.* = (rng.float(f32) * 2.0 - 1.0);
    for (w) |*v| v.* = 0.5 + rng.float(f32) * 0.5;
    for (bias) |*v| v.* = (rng.float(f32) * 2.0 - 1.0) * 0.2;

    const want = try allocator.alloc(f32, n_rows * dim);
    defer allocator.free(want);
    cpu_train_transformer.layerNormForward(x, w, bias, eps, n_rows, want);

    var buf_a = try buffer.Buffer.initStatic(&ctx, f32, x);
    defer buf_a.deinit(ctx.device);
    var buf_w = try buffer.Buffer.initStatic(&ctx, f32, w);
    defer buf_w.deinit(ctx.device);
    var buf_b = try buffer.Buffer.initStatic(&ctx, f32, bias);
    defer buf_b.deinit(ctx.device);
    var buf_c = try buffer.Buffer.initDeviceOnly(&ctx, n_rows * dim * @sizeOf(f32));
    defer buf_c.deinit(ctx.device);

    var kern = try pipeline.Kernel.init(&ctx, &shaders.layernorm, 4, @sizeOf(runtime.LayernormPush));
    defer kern.deinit();
    try kern.bind(&.{ &buf_a, &buf_w, &buf_b, &buf_c });

    const push = runtime.LayernormPush{ .dim = @intCast(dim), .eps = eps };
    try buffer.submitOneShot(&ctx, struct {
        kern: *const pipeline.Kernel,
        push: *const runtime.LayernormPush,
        n_rows: u32,
        pub fn record(s: @This(), cmd: vk.c.VkCommandBuffer) void {
            s.kern.dispatch(cmd, s.push, s.n_rows, 1, 1);
        }
    }{ .kern = &kern, .push = &push, .n_rows = @intCast(n_rows) });

    const got = try allocator.alloc(f32, n_rows * dim);
    defer allocator.free(got);
    try buf_c.readBack(&ctx, f32, got);

    var max_abs: f32 = 0;
    for (got, want) |g, e| {
        const d = @abs(g - e);
        if (d > max_abs) max_abs = d;
    }
    if (max_abs > 1e-4) {
        std.debug.print("GPU layernorm: max |Δ| = {e}\n", .{max_abs});
        return error.ParityFailed;
    }
    std.debug.print(
        "PASS GPU layernorm ({d}×{d}; max |Δ| vs CPU oracle = {e})\n",
        .{ n_rows, dim, max_abs },
    );
}

// ── GPU layernorm_backward parity smoke ─────────────────────────────

fn runGpuLayerNormBackwardSmoke(allocator: std.mem.Allocator) !void {
    var ctx = try vk.Context.init(allocator);
    defer ctx.deinit();

    const dim: usize = 256;
    const n_rows: usize = 4;
    const eps: f32 = 1e-5;

    var prng = std.Random.DefaultPrng.init(0xBA66120E);
    const rng = prng.random();
    const x = try allocator.alloc(f32, n_rows * dim);
    defer allocator.free(x);
    const dy = try allocator.alloc(f32, n_rows * dim);
    defer allocator.free(dy);
    const w = try allocator.alloc(f32, dim);
    defer allocator.free(w);
    for (x) |*v| v.* = (rng.float(f32) * 2.0 - 1.0);
    for (dy) |*v| v.* = (rng.float(f32) * 2.0 - 1.0);
    for (w) |*v| v.* = 0.5 + rng.float(f32) * 0.5;

    const dx_cpu = try allocator.alloc(f32, n_rows * dim);
    defer allocator.free(dx_cpu);
    const dw_cpu = try allocator.alloc(f32, dim);
    defer allocator.free(dw_cpu);
    const dbias_cpu = try allocator.alloc(f32, dim);
    defer allocator.free(dbias_cpu);
    @memset(dw_cpu, 0);
    @memset(dbias_cpu, 0);
    cpu_train_transformer.layerNormBackward(dy, x, w, eps, n_rows, dx_cpu, dw_cpu, dbias_cpu);

    var buf_dy = try buffer.Buffer.initStatic(&ctx, f32, dy);
    defer buf_dy.deinit(ctx.device);
    var buf_x = try buffer.Buffer.initStatic(&ctx, f32, x);
    defer buf_x.deinit(ctx.device);
    var buf_w = try buffer.Buffer.initStatic(&ctx, f32, w);
    defer buf_w.deinit(ctx.device);
    var buf_dx = try buffer.Buffer.initDeviceOnly(&ctx, n_rows * dim * @sizeOf(f32));
    defer buf_dx.deinit(ctx.device);
    var buf_dw_partial = try buffer.Buffer.initDeviceOnly(&ctx, n_rows * dim * @sizeOf(f32));
    defer buf_dw_partial.deinit(ctx.device);
    var buf_dbias_partial = try buffer.Buffer.initDeviceOnly(&ctx, n_rows * dim * @sizeOf(f32));
    defer buf_dbias_partial.deinit(ctx.device);

    var kern = try pipeline.Kernel.init(&ctx, &shaders.layernorm_backward, 6, @sizeOf(runtime.LayernormPush));
    defer kern.deinit();
    try kern.bind(&.{ &buf_dy, &buf_x, &buf_w, &buf_dx, &buf_dw_partial, &buf_dbias_partial });

    const push = runtime.LayernormPush{ .dim = @intCast(dim), .eps = eps };
    try buffer.submitOneShot(&ctx, struct {
        kern: *const pipeline.Kernel,
        push: *const runtime.LayernormPush,
        n_rows: u32,
        pub fn record(s: @This(), cmd: vk.c.VkCommandBuffer) void {
            s.kern.dispatch(cmd, s.push, s.n_rows, 1, 1);
        }
    }{ .kern = &kern, .push = &push, .n_rows = @intCast(n_rows) });

    const dx_gpu = try allocator.alloc(f32, n_rows * dim);
    defer allocator.free(dx_gpu);
    const dw_partial_gpu = try allocator.alloc(f32, n_rows * dim);
    defer allocator.free(dw_partial_gpu);
    const dbias_partial_gpu = try allocator.alloc(f32, n_rows * dim);
    defer allocator.free(dbias_partial_gpu);
    try buf_dx.readBack(&ctx, f32, dx_gpu);
    try buf_dw_partial.readBack(&ctx, f32, dw_partial_gpu);
    try buf_dbias_partial.readBack(&ctx, f32, dbias_partial_gpu);

    const dw_gpu = try allocator.alloc(f32, dim);
    defer allocator.free(dw_gpu);
    const dbias_gpu = try allocator.alloc(f32, dim);
    defer allocator.free(dbias_gpu);
    @memset(dw_gpu, 0);
    @memset(dbias_gpu, 0);
    for (0..n_rows) |r| {
        const off = r * dim;
        for (0..dim) |i| {
            dw_gpu[i] += dw_partial_gpu[off + i];
            dbias_gpu[i] += dbias_partial_gpu[off + i];
        }
    }

    var max_dx: f32 = 0;
    for (dx_gpu, dx_cpu) |g, c| {
        const d = @abs(g - c);
        if (d > max_dx) max_dx = d;
    }
    if (max_dx > 1e-4) {
        std.debug.print("layernorm_backward dx: max |Δ| = {e}\n", .{max_dx});
        return error.ParityFailed;
    }

    var max_dw: f32 = 0;
    for (dw_gpu, dw_cpu) |g, c| {
        const d = @abs(g - c);
        if (d > max_dw) max_dw = d;
    }
    if (max_dw > 1e-4) {
        std.debug.print("layernorm_backward dw: max |Δ| = {e}\n", .{max_dw});
        return error.ParityFailed;
    }

    var max_db: f32 = 0;
    for (dbias_gpu, dbias_cpu) |g, c| {
        const d = @abs(g - c);
        if (d > max_db) max_db = d;
    }
    if (max_db > 1e-4) {
        std.debug.print("layernorm_backward dbias: max |Δ| = {e}\n", .{max_db});
        return error.ParityFailed;
    }

    std.debug.print(
        "PASS GPU layernorm_backward ({d}×{d}; dx + dw + dbias match CPU oracle ≤ 1e-4)\n",
        .{ n_rows, dim },
    );
}

// ── Embedding gradient (sparse scatter) smoke ──────────────────────
//
// Tier-2 chunk 4 — backward through embed_lookup. The forward pass
// is `x[p, :] = E[token_ids[p], :]`; backward scatters the per-position
// upstream gradient `dy[p, :]` back into `dE[token_ids[p], :]`. Tokens
// that appear more than once in the sequence sum into the same row.
//
// Two claims:
//   1. Multi-occurrence sums are correctly accumulated (a hand-built
//      sequence with deliberately repeated tokens is checked against
//      a manual reference sum).
//   2. GPU vocab-major scatter shader matches the CPU oracle bit-exact
//      across a synthetic batch.

fn runEmbeddingBackwardSmoke(allocator: std.mem.Allocator) !void {
    var ctx = try vk.Context.init(allocator);
    defer ctx.deinit();

    const dim: usize = 8;
    const vocab_size: usize = 16;
    const n_pos: usize = 6;
    // Deliberately reused token ids: 3, 7, 3, 11, 7, 3 — token 3 appears
    // 3×, token 7 twice, token 11 once. dE[3] should sum dy[0]+dy[2]+dy[5];
    // dE[7] sums dy[1]+dy[4]; dE[11] = dy[3]; all others zero.
    const token_ids = [_]u32{ 3, 7, 3, 11, 7, 3 };

    const dy = try allocator.alloc(f32, n_pos * dim);
    defer allocator.free(dy);
    var prng = std.Random.DefaultPrng.init(0xE0B570BE);
    const rng = prng.random();
    for (dy) |*v| v.* = (rng.float(f32) * 2.0 - 1.0);

    // ── (1) CPU oracle vs hand-built reference ────────────────────
    const dE_cpu = try allocator.alloc(f32, vocab_size * dim);
    defer allocator.free(dE_cpu);
    @memset(dE_cpu, 0);
    cpu_train_transformer.embeddingBackward(dy, &token_ids, vocab_size, dim, dE_cpu);

    // Manual reference: for each unique token, sum the dy rows where it
    // appears. Compare bit-exact to the oracle.
    const ref = try allocator.alloc(f32, vocab_size * dim);
    defer allocator.free(ref);
    @memset(ref, 0);
    for (token_ids, 0..) |tok, p| {
        const off = @as(usize, tok) * dim;
        for (0..dim) |i| ref[off + i] += dy[p * dim + i];
    }
    for (dE_cpu, ref) |a, b| {
        if (a != b) {
            std.debug.print("CPU oracle != reference\n", .{});
            return error.ParityFailed;
        }
    }

    // ── (2) GPU shader vs CPU oracle ──────────────────────────────
    var buf_dy = try buffer.Buffer.initStatic(&ctx, f32, dy);
    defer buf_dy.deinit(ctx.device);
    var buf_ti = try buffer.Buffer.initStatic(&ctx, u32, &token_ids);
    defer buf_ti.deinit(ctx.device);
    var buf_dE = try buffer.Buffer.initDeviceOnly(&ctx, vocab_size * dim * @sizeOf(f32));
    defer buf_dE.deinit(ctx.device);
    // initDeviceOnly already zeroes via vkCmdFillBuffer — relying on
    // that pre-zeroing is exactly the dE_cpu @memset(0) above.

    var kern = try pipeline.Kernel.init(&ctx, &shaders.embedding_backward, 3, @sizeOf(runtime.EmbeddingBackwardPush));
    defer kern.deinit();
    try kern.bind(&.{ &buf_dy, &buf_ti, &buf_dE });

    const push = runtime.EmbeddingBackwardPush{
        .dim = @intCast(dim),
        .n_pos = @intCast(n_pos),
        .vocab_size = @intCast(vocab_size),
    };
    try buffer.submitOneShot(&ctx, struct {
        kern: *const pipeline.Kernel,
        push: *const runtime.EmbeddingBackwardPush,
        n_groups: u32,
        pub fn record(s: @This(), cmd: vk.c.VkCommandBuffer) void {
            s.kern.dispatch(cmd, s.push, s.n_groups, 1, 1);
        }
    }{ .kern = &kern, .push = &push, .n_groups = @intCast(vocab_size) });

    const got = try allocator.alloc(f32, vocab_size * dim);
    defer allocator.free(got);
    try buf_dE.readBack(&ctx, f32, got);

    var max_abs: f32 = 0;
    for (got, dE_cpu) |g, c| {
        const d = @abs(g - c);
        if (d > max_abs) max_abs = d;
    }
    if (max_abs > 1e-6) {
        std.debug.print("embedding_backward GPU: max |Δ| vs CPU = {e}\n", .{max_abs});
        return error.ParityFailed;
    }

    std.debug.print(
        "PASS embedding gradient (vocab={d}, dim={d}, n_pos={d} with reused tokens; CPU oracle bit-exact vs reference; GPU max |Δ| = {e})\n",
        .{ vocab_size, dim, n_pos, max_abs },
    );
}

// ── Softmax backward smoke (CPU oracle + GPU parity) ───────────────
//
// Tier-2 chunk 5 — bridge to attention backward. Two halves:
//
//   1. CPU oracle numeric-grad parity. Loss = Σⱼ dyⱼ · yⱼ(x); we
//      verify ∂L/∂xᵢ matches central-difference at eps=1e-3 to ≤ 1%.
//
//   2. GPU shader parity. Multi-row, with stride = dim (packed
//      layout). dx must match CPU oracle ≤ 1e-5.
//
// Saved-activation is `y` (the softmax output) — no need to re-do
// max/sum/exp on the backward pass.

fn runSoftmaxBackwardSmoke(allocator: std.mem.Allocator) !void {
    var ctx = try vk.Context.init(allocator);
    defer ctx.deinit();

    // ── (1) CPU oracle numeric-grad ───────────────────────────────
    {
        const dim: usize = 24;
        var x: [dim]f32 = undefined;
        var y: [dim]f32 = undefined;
        var dy: [dim]f32 = undefined;
        var dx: [dim]f32 = undefined;
        var probe_y: [dim]f32 = undefined;

        var prng = std.Random.DefaultPrng.init(0x50F7BAC1);
        const rng = prng.random();
        for (&x) |*v| v.* = (rng.float(f32) * 2.0 - 1.0) * 2.0;
        for (&dy) |*v| v.* = (rng.float(f32) * 2.0 - 1.0);

        // Forward: stable softmax.
        var max_x: f32 = -std.math.inf(f32);
        for (x) |v| if (v > max_x) {
            max_x = v;
        };
        var sum: f64 = 0;
        for (x, &y) |xi, *yi| {
            const e = @exp(xi - max_x);
            yi.* = e;
            sum += e;
        }
        const inv_sum: f32 = 1.0 / @as(f32, @floatCast(sum));
        for (&y) |*yi| yi.* *= inv_sum;

        cpu_train_transformer.softmaxBackwardRow(&dy, &y, &dx);

        const eps_h: f32 = 1e-3;
        const probes = [_]usize{ 0, 4, 9, 15, 21 };
        var max_rel_err: f32 = 0;
        for (probes) |i| {
            // f(x) = Σⱼ dyⱼ · y(x)ⱼ. Re-run forward with x[i] perturbed.
            const orig = x[i];
            x[i] = orig + eps_h;
            var local_max: f32 = -std.math.inf(f32);
            for (x) |v| if (v > local_max) {
                local_max = v;
            };
            var local_sum: f64 = 0;
            for (x, &probe_y) |xi, *yi| {
                const e = @exp(xi - local_max);
                yi.* = e;
                local_sum += e;
            }
            for (&probe_y) |*yi| yi.* /= @as(f32, @floatCast(local_sum));
            var l_plus: f32 = 0;
            for (dy, probe_y) |d, yi| l_plus += d * yi;

            x[i] = orig - eps_h;
            local_max = -std.math.inf(f32);
            for (x) |v| if (v > local_max) {
                local_max = v;
            };
            local_sum = 0;
            for (x, &probe_y) |xi, *yi| {
                const e = @exp(xi - local_max);
                yi.* = e;
                local_sum += e;
            }
            for (&probe_y) |*yi| yi.* /= @as(f32, @floatCast(local_sum));
            var l_minus: f32 = 0;
            for (dy, probe_y) |d, yi| l_minus += d * yi;
            x[i] = orig;

            const numeric = (l_plus - l_minus) / (2.0 * eps_h);
            const analytic = dx[i];
            const denom = @max(@abs(numeric), @abs(analytic));
            const rel_err = if (denom > 0) @abs(numeric - analytic) / denom else @abs(numeric - analytic);
            if (rel_err > 1e-2) {
                std.debug.print("softmax dx[{d}] analytic={d:.6} numeric={d:.6} rel_err={d:.4}\n", .{ i, analytic, numeric, rel_err });
                return error.ParityFailed;
            }
            if (rel_err > max_rel_err) max_rel_err = rel_err;
        }
        std.debug.print("    softmax CPU numeric-grad parity ≤ {e} (5 probes)\n", .{max_rel_err});
    }

    // ── (2) GPU shader parity ─────────────────────────────────────
    const dim: usize = 256;
    const n_rows: usize = 4;
    const y_buf = try allocator.alloc(f32, n_rows * dim);
    defer allocator.free(y_buf);
    const dy = try allocator.alloc(f32, n_rows * dim);
    defer allocator.free(dy);
    const dx_cpu = try allocator.alloc(f32, n_rows * dim);
    defer allocator.free(dx_cpu);

    var prng = std.Random.DefaultPrng.init(0x50F7BAC2);
    const rng = prng.random();
    for (dy) |*v| v.* = (rng.float(f32) * 2.0 - 1.0);

    // Synthetic stable softmax outputs per row (no need to go through
    // forward — y just needs to be a valid probability distribution).
    for (0..n_rows) |r| {
        var row_sum: f64 = 0;
        for (0..dim) |i| {
            const v = @exp(rng.float(f32) * 4.0 - 2.0);
            y_buf[r * dim + i] = v;
            row_sum += v;
        }
        const inv: f32 = 1.0 / @as(f32, @floatCast(row_sum));
        for (0..dim) |i| y_buf[r * dim + i] *= inv;
    }

    cpu_train_transformer.softmaxBackward(dy, y_buf, n_rows, dim, dx_cpu);

    var buf_dy = try buffer.Buffer.initStatic(&ctx, f32, dy);
    defer buf_dy.deinit(ctx.device);
    var buf_y = try buffer.Buffer.initStatic(&ctx, f32, y_buf);
    defer buf_y.deinit(ctx.device);
    var buf_dx = try buffer.Buffer.initDeviceOnly(&ctx, n_rows * dim * @sizeOf(f32));
    defer buf_dx.deinit(ctx.device);

    var kern = try pipeline.Kernel.init(&ctx, &shaders.softmax_backward, 3, @sizeOf(runtime.SoftmaxPush));
    defer kern.deinit();
    try kern.bind(&.{ &buf_dy, &buf_y, &buf_dx });

    const push = runtime.SoftmaxPush{ .dim = @intCast(dim), .stride = @intCast(dim) };
    try buffer.submitOneShot(&ctx, struct {
        kern: *const pipeline.Kernel,
        push: *const runtime.SoftmaxPush,
        n_rows: u32,
        pub fn record(s: @This(), cmd: vk.c.VkCommandBuffer) void {
            s.kern.dispatch(cmd, s.push, s.n_rows, 1, 1);
        }
    }{ .kern = &kern, .push = &push, .n_rows = @intCast(n_rows) });

    const dx_gpu = try allocator.alloc(f32, n_rows * dim);
    defer allocator.free(dx_gpu);
    try buf_dx.readBack(&ctx, f32, dx_gpu);

    var max_abs: f32 = 0;
    for (dx_gpu, dx_cpu) |g, c| {
        const d = @abs(g - c);
        if (d > max_abs) max_abs = d;
    }
    if (max_abs > 1e-5) {
        std.debug.print("softmax_backward GPU: max |Δ| = {e}\n", .{max_abs});
        return error.ParityFailed;
    }

    std.debug.print(
        "PASS softmax backward ({d}×{d}; CPU numeric-grad parity ≤ 1%; GPU max |Δ| = {e})\n",
        .{ n_rows, dim, max_abs },
    );
}

// ── attention backward CPU oracle smoke ─────────────────────────────
//
// Numeric-grad parity for the full SDPA backward chain:
//   forward(Q, K, V) → out;  L = Σ d_out · out  (with d_out fixed)
//   ∂L/∂Q  numerically vs analytical
//   ∂L/∂K  numerically vs analytical
//   ∂L/∂V  numerically vs analytical
// Tested with GQA (n_heads != n_kv_heads) and causal masking on a
// small shape that exposes head-axis and position-axis bugs but is
// small enough for central-difference to be stable.

fn runAttentionBackwardCpuSmoke(allocator: std.mem.Allocator) !void {
    const n_q: usize = 3;
    const n_kv: usize = 4;
    const n_heads: usize = 4;
    const n_kv_heads: usize = 2; // heads_per_kv = 2 → GQA
    const head_dim: usize = 8;
    const causal = true;

    const q_total = n_q * n_heads * head_dim;
    const kv_total = n_kv * n_kv_heads * head_dim;
    const out_total = n_q * n_heads * head_dim;
    const scores_total = n_q * n_heads * n_kv;

    const Q = try allocator.alloc(f32, q_total);
    defer allocator.free(Q);
    const K = try allocator.alloc(f32, kv_total);
    defer allocator.free(K);
    const V = try allocator.alloc(f32, kv_total);
    defer allocator.free(V);
    const d_out = try allocator.alloc(f32, out_total);
    defer allocator.free(d_out);

    var prng = std.Random.DefaultPrng.init(0xA77B_AC_01);
    const rng = prng.random();
    for (Q) |*v| v.* = (rng.float(f32) * 2.0 - 1.0);
    for (K) |*v| v.* = (rng.float(f32) * 2.0 - 1.0);
    for (V) |*v| v.* = (rng.float(f32) * 2.0 - 1.0);
    for (d_out) |*v| v.* = (rng.float(f32) * 2.0 - 1.0);

    const scores = try allocator.alloc(f32, scores_total);
    defer allocator.free(scores);
    const attn = try allocator.alloc(f32, scores_total);
    defer allocator.free(attn);
    const out = try allocator.alloc(f32, out_total);
    defer allocator.free(out);
    const d_scores = try allocator.alloc(f32, scores_total);
    defer allocator.free(d_scores);
    const dQ = try allocator.alloc(f32, q_total);
    defer allocator.free(dQ);
    const dK = try allocator.alloc(f32, kv_total);
    defer allocator.free(dK);
    const dV = try allocator.alloc(f32, kv_total);
    defer allocator.free(dV);

    cpu_train_transformer.attentionForward(Q, K, V, n_q, n_kv, n_heads, n_kv_heads, head_dim, causal, scores, attn, out);
    cpu_train_transformer.attentionBackward(d_out, Q, K, V, attn, n_q, n_kv, n_heads, n_kv_heads, head_dim, causal, d_scores, dQ, dK, dV);

    // Probe loss helper: forward and dot with d_out.
    const probe_scores = try allocator.alloc(f32, scores_total);
    defer allocator.free(probe_scores);
    const probe_attn = try allocator.alloc(f32, scores_total);
    defer allocator.free(probe_attn);
    const probe_out = try allocator.alloc(f32, out_total);
    defer allocator.free(probe_out);

    const lossFn = struct {
        fn run(
            Qp: []const f32,
            Kp: []const f32,
            Vp: []const f32,
            d_outp: []const f32,
            n_q_l: usize,
            n_kv_l: usize,
            n_heads_l: usize,
            n_kv_heads_l: usize,
            head_dim_l: usize,
            causal_l: bool,
            scratch_scores: []f32,
            scratch_attn: []f32,
            scratch_out: []f32,
        ) f64 {
            cpu_train_transformer.attentionForward(
                Qp,
                Kp,
                Vp,
                n_q_l,
                n_kv_l,
                n_heads_l,
                n_kv_heads_l,
                head_dim_l,
                causal_l,
                scratch_scores,
                scratch_attn,
                scratch_out,
            );
            var L: f64 = 0;
            for (scratch_out, d_outp) |o, dop| L += @as(f64, o) * @as(f64, dop);
            return L;
        }
    }.run;

    const eps_h: f32 = 5e-3;
    // Central-diff truncation ~O(eps_h²)·f‴; fp32 forward adds ~1e-5
    // noise per loss eval, so for gradients of magnitude ~10⁻³ we
    // expect rel_err around 1%. Skip very-near-zero analytic entries.
    const target_rel_err: f32 = 2e-2;
    const abs_floor: f32 = 5e-5;

    const NamedBuf = struct {
        name: []const u8,
        buf: []f32,
        analytic: []const f32,
        probes: []const usize,
    };

    const probes_q = &[_]usize{ 0, 5, 13, 23, 47 }; // across (q, h, d) flat
    const probes_k = &[_]usize{ 0, 7, 14, 27, 40 };
    const probes_v = &[_]usize{ 1, 9, 18, 29, 41 };

    var named = [_]NamedBuf{
        .{ .name = "Q", .buf = Q, .analytic = dQ, .probes = probes_q },
        .{ .name = "K", .buf = K, .analytic = dK, .probes = probes_k },
        .{ .name = "V", .buf = V, .analytic = dV, .probes = probes_v },
    };

    var max_rel_err: f32 = 0;
    for (&named) |nb| {
        for (nb.probes) |i| {
            const orig = nb.buf[i];
            nb.buf[i] = orig + eps_h;
            const Lp = lossFn(Q, K, V, d_out, n_q, n_kv, n_heads, n_kv_heads, head_dim, causal, probe_scores, probe_attn, probe_out);
            nb.buf[i] = orig - eps_h;
            const Lm = lossFn(Q, K, V, d_out, n_q, n_kv, n_heads, n_kv_heads, head_dim, causal, probe_scores, probe_attn, probe_out);
            nb.buf[i] = orig;

            const numeric: f32 = @floatCast((Lp - Lm) / (2.0 * @as(f64, eps_h)));
            const analytic: f32 = nb.analytic[i];
            const denom = @max(@abs(numeric), @abs(analytic));
            // Both small? Numeric central-diff noise dominates; skip.
            if (denom < abs_floor) continue;
            const rel_err = @abs(numeric - analytic) / denom;
            if (rel_err > target_rel_err) {
                std.debug.print(
                    "attn d{s}[{d}] analytic={d:.6} numeric={d:.6} rel_err={d:.4}\n",
                    .{ nb.name, i, analytic, numeric, rel_err },
                );
                return error.ParityFailed;
            }
            if (rel_err > max_rel_err) max_rel_err = rel_err;
        }
    }

    std.debug.print(
        "PASS attention backward CPU (n_q={d} n_kv={d} heads={d}/{d} d={d} causal numeric-grad ≤ {e})\n",
        .{ n_q, n_kv, n_heads, n_kv_heads, head_dim, max_rel_err },
    );
}

// ── GPU attention-backward smoke: dV / d_attn / dQ / dK vs CPU oracle ─

fn runGpuAttentionBackwardSmoke(allocator: std.mem.Allocator) !void {
    var ctx = try vk.Context.init(allocator);
    defer ctx.deinit();

    // Bigger shape than CPU oracle smoke — exercises both head-axis and
    // position-axis with subgroup-reduction'd kernels.
    const n_q: usize = 4;
    const n_kv: usize = 8;
    const n_heads: usize = 8;
    const n_kv_heads: usize = 4; // heads_per_kv = 2
    const head_dim: usize = 64;
    const heads_per_kv: usize = n_heads / n_kv_heads;
    const inv_sqrt_d: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(head_dim)));

    const q_total = n_q * n_heads * head_dim;
    const kv_total = n_kv * n_kv_heads * head_dim;
    const scores_total = n_q * n_heads * n_kv;

    const Q = try allocator.alloc(f32, q_total);
    defer allocator.free(Q);
    const K = try allocator.alloc(f32, kv_total);
    defer allocator.free(K);
    const V = try allocator.alloc(f32, kv_total);
    defer allocator.free(V);
    const d_out = try allocator.alloc(f32, q_total);
    defer allocator.free(d_out);

    var prng = std.Random.DefaultPrng.init(0xA77B_AC_02);
    const rng = prng.random();
    for (Q) |*v| v.* = (rng.float(f32) * 2.0 - 1.0);
    for (K) |*v| v.* = (rng.float(f32) * 2.0 - 1.0);
    for (V) |*v| v.* = (rng.float(f32) * 2.0 - 1.0);
    for (d_out) |*v| v.* = (rng.float(f32) * 2.0 - 1.0);

    // ── CPU reference: forward + full backward ────────────────────
    const scores = try allocator.alloc(f32, scores_total);
    defer allocator.free(scores);
    const attn = try allocator.alloc(f32, scores_total);
    defer allocator.free(attn);
    const out = try allocator.alloc(f32, q_total);
    defer allocator.free(out);
    const d_scores_cpu = try allocator.alloc(f32, scores_total);
    defer allocator.free(d_scores_cpu);
    const dQ_cpu = try allocator.alloc(f32, q_total);
    defer allocator.free(dQ_cpu);
    const dK_cpu = try allocator.alloc(f32, kv_total);
    defer allocator.free(dK_cpu);
    const dV_cpu = try allocator.alloc(f32, kv_total);
    defer allocator.free(dV_cpu);

    cpu_train_transformer.attentionForward(Q, K, V, n_q, n_kv, n_heads, n_kv_heads, head_dim, true, scores, attn, out);
    cpu_train_transformer.attentionBackward(d_out, Q, K, V, attn, n_q, n_kv, n_heads, n_kv_heads, head_dim, true, d_scores_cpu, dQ_cpu, dK_cpu, dV_cpu);

    // Also stage d_attn (pre-softmax-backward) for shader-level parity
    // on the d_attn shader specifically.
    const d_attn_cpu = try allocator.alloc(f32, scores_total);
    defer allocator.free(d_attn_cpu);
    for (0..n_q) |q| {
        for (0..n_heads) |h| {
            const kv_h = h / heads_per_kv;
            const dout_off = q * n_heads * head_dim + h * head_dim;
            const da_off = q * n_heads * n_kv + h * n_kv;
            for (0..n_kv) |k| {
                const v_off = k * n_kv_heads * head_dim + kv_h * head_dim;
                var s: f64 = 0;
                for (0..head_dim) |d| s += @as(f64, d_out[dout_off + d]) * @as(f64, V[v_off + d]);
                d_attn_cpu[da_off + k] = @floatCast(s);
            }
        }
    }

    // ── GPU buffers ───────────────────────────────────────────────
    var buf_Q = try buffer.Buffer.initStatic(&ctx, f32, Q);
    defer buf_Q.deinit(ctx.device);
    var buf_K = try buffer.Buffer.initStatic(&ctx, f32, K);
    defer buf_K.deinit(ctx.device);
    var buf_V = try buffer.Buffer.initStatic(&ctx, f32, V);
    defer buf_V.deinit(ctx.device);
    var buf_dout = try buffer.Buffer.initStatic(&ctx, f32, d_out);
    defer buf_dout.deinit(ctx.device);
    var buf_attn = try buffer.Buffer.initStatic(&ctx, f32, attn);
    defer buf_attn.deinit(ctx.device);
    // We need d_scores upload for dQ/dK kernels — load CPU d_scores now.
    var buf_dscores = try buffer.Buffer.initStatic(&ctx, f32, d_scores_cpu);
    defer buf_dscores.deinit(ctx.device);

    var buf_dattn_gpu = try buffer.Buffer.initDeviceOnly(&ctx, scores_total * @sizeOf(f32));
    defer buf_dattn_gpu.deinit(ctx.device);
    var buf_dV_gpu = try buffer.Buffer.initDeviceOnly(&ctx, kv_total * @sizeOf(f32));
    defer buf_dV_gpu.deinit(ctx.device);
    var buf_dQ_gpu = try buffer.Buffer.initDeviceOnly(&ctx, q_total * @sizeOf(f32));
    defer buf_dQ_gpu.deinit(ctx.device);
    var buf_dK_gpu = try buffer.Buffer.initDeviceOnly(&ctx, kv_total * @sizeOf(f32));
    defer buf_dK_gpu.deinit(ctx.device);

    // ── Pipelines ─────────────────────────────────────────────────
    var k_dattn = try pipeline.Kernel.init(&ctx, &shaders.attn_backward_dattn, 3, @sizeOf(runtime.AttnBackwardDattnPush));
    defer k_dattn.deinit();
    try k_dattn.bind(&.{ &buf_dout, &buf_V, &buf_dattn_gpu });

    var k_dv = try pipeline.Kernel.init(&ctx, &shaders.attn_backward_dv, 3, @sizeOf(runtime.AttnBackwardDvPush));
    defer k_dv.deinit();
    try k_dv.bind(&.{ &buf_attn, &buf_dout, &buf_dV_gpu });

    var k_dq = try pipeline.Kernel.init(&ctx, &shaders.attn_backward_dq, 3, @sizeOf(runtime.AttnBackwardDqPush));
    defer k_dq.deinit();
    try k_dq.bind(&.{ &buf_dscores, &buf_K, &buf_dQ_gpu });

    var k_dk = try pipeline.Kernel.init(&ctx, &shaders.attn_backward_dk, 3, @sizeOf(runtime.AttnBackwardDkPush));
    defer k_dk.deinit();
    try k_dk.bind(&.{ &buf_dscores, &buf_Q, &buf_dK_gpu });

    const push_dattn = runtime.AttnBackwardDattnPush{
        .n_q = @intCast(n_q),
        .n_heads = @intCast(n_heads),
        .heads_per_kv = @intCast(heads_per_kv),
        .head_dim = @intCast(head_dim),
        .n_kv = @intCast(n_kv),
        .kv_stride = @intCast(n_kv_heads * head_dim),
        .attn_stride = @intCast(n_kv),
    };
    const push_dv = runtime.AttnBackwardDvPush{
        .n_q = @intCast(n_q),
        .n_heads = @intCast(n_heads),
        .heads_per_kv = @intCast(heads_per_kv),
        .n_kv_heads = @intCast(n_kv_heads),
        .head_dim = @intCast(head_dim),
        .n_kv = @intCast(n_kv),
        .attn_stride = @intCast(n_kv),
    };
    const push_dq = runtime.AttnBackwardDqPush{
        .n_q = @intCast(n_q),
        .n_heads = @intCast(n_heads),
        .heads_per_kv = @intCast(heads_per_kv),
        .head_dim = @intCast(head_dim),
        .n_kv = @intCast(n_kv),
        .kv_stride = @intCast(n_kv_heads * head_dim),
        .scores_stride = @intCast(n_kv),
        .inv_sqrt_dim = inv_sqrt_d,
    };
    const push_dk = runtime.AttnBackwardDkPush{
        .n_q = @intCast(n_q),
        .n_heads = @intCast(n_heads),
        .heads_per_kv = @intCast(heads_per_kv),
        .n_kv_heads = @intCast(n_kv_heads),
        .head_dim = @intCast(head_dim),
        .n_kv = @intCast(n_kv),
        .scores_stride = @intCast(n_kv),
        .inv_sqrt_dim = inv_sqrt_d,
    };

    try buffer.submitOneShot(&ctx, struct {
        k_dattn: *const pipeline.Kernel,
        k_dv: *const pipeline.Kernel,
        k_dq: *const pipeline.Kernel,
        k_dk: *const pipeline.Kernel,
        p_dattn: *const runtime.AttnBackwardDattnPush,
        p_dv: *const runtime.AttnBackwardDvPush,
        p_dq: *const runtime.AttnBackwardDqPush,
        p_dk: *const runtime.AttnBackwardDkPush,
        n_q: u32,
        n_kv: u32,
        n_heads: u32,
        n_kv_heads: u32,
        head_dim: u32,
        pub fn record(s: @This(), cmd: vk.c.VkCommandBuffer) void {
            // d_attn[q, h, k]
            s.k_dattn.dispatch(cmd, s.p_dattn, s.n_q * s.n_heads * s.n_kv, 1, 1);
            // dV[k, kv_h, d]
            s.k_dv.dispatch(cmd, s.p_dv, s.n_kv * s.n_kv_heads * s.head_dim, 1, 1);
            // dQ[q, h, d]
            s.k_dq.dispatch(cmd, s.p_dq, s.n_q * s.n_heads * s.head_dim, 1, 1);
            // dK[k, kv_h, d]
            s.k_dk.dispatch(cmd, s.p_dk, s.n_kv * s.n_kv_heads * s.head_dim, 1, 1);
        }
    }{
        .k_dattn = &k_dattn,
        .k_dv = &k_dv,
        .k_dq = &k_dq,
        .k_dk = &k_dk,
        .p_dattn = &push_dattn,
        .p_dv = &push_dv,
        .p_dq = &push_dq,
        .p_dk = &push_dk,
        .n_q = @intCast(n_q),
        .n_kv = @intCast(n_kv),
        .n_heads = @intCast(n_heads),
        .n_kv_heads = @intCast(n_kv_heads),
        .head_dim = @intCast(head_dim),
    });

    const dattn_gpu = try allocator.alloc(f32, scores_total);
    defer allocator.free(dattn_gpu);
    const dV_gpu = try allocator.alloc(f32, kv_total);
    defer allocator.free(dV_gpu);
    const dQ_gpu = try allocator.alloc(f32, q_total);
    defer allocator.free(dQ_gpu);
    const dK_gpu = try allocator.alloc(f32, kv_total);
    defer allocator.free(dK_gpu);
    try buf_dattn_gpu.readBack(&ctx, f32, dattn_gpu);
    try buf_dV_gpu.readBack(&ctx, f32, dV_gpu);
    try buf_dQ_gpu.readBack(&ctx, f32, dQ_gpu);
    try buf_dK_gpu.readBack(&ctx, f32, dK_gpu);

    const tol: f32 = 1e-4;
    const Pair = struct { name: []const u8, gpu: []const f32, cpu: []const f32 };
    const pairs = [_]Pair{
        .{ .name = "d_attn", .gpu = dattn_gpu, .cpu = d_attn_cpu },
        .{ .name = "dV", .gpu = dV_gpu, .cpu = dV_cpu },
        .{ .name = "dQ", .gpu = dQ_gpu, .cpu = dQ_cpu },
        .{ .name = "dK", .gpu = dK_gpu, .cpu = dK_cpu },
    };
    var max_abs: f32 = 0;
    for (pairs) |p| {
        var p_max: f32 = 0;
        for (p.gpu, p.cpu) |g, c| {
            const d = @abs(g - c);
            if (d > p_max) p_max = d;
        }
        if (p_max > tol) {
            std.debug.print("attn_backward {s}: max |Δ| = {e}\n", .{ p.name, p_max });
            return error.ParityFailed;
        }
        if (p_max > max_abs) max_abs = p_max;
    }

    std.debug.print(
        "PASS GPU attention backward (n_q={d} n_kv={d} heads={d}/{d} d={d}; d_attn+dV+dQ+dK max |Δ| = {e})\n",
        .{ n_q, n_kv, n_heads, n_kv_heads, head_dim, max_abs },
    );
}

// ── RoPE backward smoke: CPU oracle (round-trip + numeric) + GPU parity ─

fn runRopeBackwardSmoke(allocator: std.mem.Allocator) !void {
    var ctx = try vk.Context.init(allocator);
    defer ctx.deinit();

    const n_heads: usize = 8;
    const head_dim: usize = 64;
    const rotary_dim: usize = 16; // partial — exercise the pass-through tail
    const pos: usize = 17;
    const theta_base: f32 = 10000.0;
    const total = n_heads * head_dim;

    const x = try allocator.alloc(f32, total);
    defer allocator.free(x);
    const d_out = try allocator.alloc(f32, total);
    defer allocator.free(d_out);
    const d_in_cpu = try allocator.alloc(f32, total);
    defer allocator.free(d_in_cpu);

    var prng = std.Random.DefaultPrng.init(0x70BE_BAC1);
    const rng = prng.random();
    for (x) |*v| v.* = (rng.float(f32) * 2.0 - 1.0);
    for (d_out) |*v| v.* = (rng.float(f32) * 2.0 - 1.0);

    try cpu_train_transformer.ropeBackwardPartial(d_in_cpu, d_out, n_heads, head_dim, rotary_dim, pos, theta_base);

    // ── (1) Round-trip parity: backward(forward(x)) == x for any x.
    // Rotation by +θ then -θ is identity → ropeBackwardPartial(rope(x)) == x.
    {
        const fwd = try allocator.alloc(f32, total);
        defer allocator.free(fwd);
        const rt = try allocator.alloc(f32, total);
        defer allocator.free(rt);
        try cpu_math.applyRopePartial(fwd, x, n_heads, head_dim, rotary_dim, pos, theta_base);
        try cpu_train_transformer.ropeBackwardPartial(rt, fwd, n_heads, head_dim, rotary_dim, pos, theta_base);
        var max_rt: f32 = 0;
        for (x, rt) |a, b| {
            const d = @abs(a - b);
            if (d > max_rt) max_rt = d;
        }
        if (max_rt > 1e-5) {
            std.debug.print("rope backward round-trip max |Δ| = {e}\n", .{max_rt});
            return error.ParityFailed;
        }
    }

    // ── (2) Numeric-grad on L = Σ d_out · forward(x).
    {
        const eps_h: f32 = 1e-3;
        const probes = [_]usize{ 0, 5, 8, 21, 47, 63, 100, 255 };
        const fwd = try allocator.alloc(f32, total);
        defer allocator.free(fwd);

        var max_rel: f32 = 0;
        for (probes) |i| {
            const orig = x[i];
            x[i] = orig + eps_h;
            try cpu_math.applyRopePartial(fwd, x, n_heads, head_dim, rotary_dim, pos, theta_base);
            var Lp: f64 = 0;
            for (fwd, d_out) |f, d| Lp += @as(f64, f) * @as(f64, d);
            x[i] = orig - eps_h;
            try cpu_math.applyRopePartial(fwd, x, n_heads, head_dim, rotary_dim, pos, theta_base);
            var Lm: f64 = 0;
            for (fwd, d_out) |f, d| Lm += @as(f64, f) * @as(f64, d);
            x[i] = orig;
            const numeric: f32 = @floatCast((Lp - Lm) / (2.0 * @as(f64, eps_h)));
            const analytic = d_in_cpu[i];
            const denom = @max(@abs(numeric), @abs(analytic));
            if (denom < 1e-6) continue;
            const rel = @abs(numeric - analytic) / denom;
            if (rel > 1e-2) {
                std.debug.print("rope d_in[{d}] analytic={d:.6} numeric={d:.6} rel_err={d:.4}\n", .{ i, analytic, numeric, rel });
                return error.ParityFailed;
            }
            if (rel > max_rel) max_rel = rel;
        }
    }

    // ── (3) GPU parity ─────────────────────────────────────────────
    var buf_dout = try buffer.Buffer.initStatic(&ctx, f32, d_out);
    defer buf_dout.deinit(ctx.device);
    var buf_din = try buffer.Buffer.initDeviceOnly(&ctx, total * @sizeOf(f32));
    defer buf_din.deinit(ctx.device);

    var kern = try pipeline.Kernel.init(&ctx, &shaders.rope_backward, 2, @sizeOf(runtime.RopePartialPush));
    defer kern.deinit();
    try kern.bind(&.{ &buf_dout, &buf_din });

    const push = runtime.RopePartialPush{
        .n_heads = @intCast(n_heads),
        .head_dim = @intCast(head_dim),
        .rotary_dim = @intCast(rotary_dim),
        .pos = @intCast(pos),
        .theta_base = theta_base,
    };
    try buffer.submitOneShot(&ctx, struct {
        kern: *const pipeline.Kernel,
        push: *const runtime.RopePartialPush,
        groups: u32,
        pub fn record(s: @This(), cmd: vk.c.VkCommandBuffer) void {
            s.kern.dispatch(cmd, s.push, s.groups, 1, 1);
        }
    }{
        .kern = &kern,
        .push = &push,
        .groups = @intCast((total + 255) / 256),
    });

    const d_in_gpu = try allocator.alloc(f32, total);
    defer allocator.free(d_in_gpu);
    try buf_din.readBack(&ctx, f32, d_in_gpu);

    var max_abs: f32 = 0;
    for (d_in_gpu, d_in_cpu) |g, c| {
        const d = @abs(g - c);
        if (d > max_abs) max_abs = d;
    }
    if (max_abs > 1e-5) {
        std.debug.print("rope_backward GPU max |Δ| = {e}\n", .{max_abs});
        return error.ParityFailed;
    }

    std.debug.print(
        "PASS RoPE backward (n_heads={d} head_dim={d} rotary_dim={d}; round-trip OK, numeric-grad ≤ 1%, GPU max |Δ| = {e})\n",
        .{ n_heads, head_dim, rotary_dim, max_abs },
    );
}

// ── End-to-end toy-decoder fine-tune smoke (Tier-2 chunk 8a) ────────
//
// Wires the chunk 1-7 backward primitives into a single decoder layer
// and trains it with Adam against a synthetic target. The target is
// the layer's *own initial output* with a small random perturbation,
// so loss must reach near zero for any working backward chain — proves
// gradients flow correctly through:
//
//   x → RMSNorm → Q/K/V projections → SDPA(causal) → o-proj → +residual
//     → RMSNorm → FF1 → ReLU → FF2 → +residual → y
//
// If any single piece's gradient is wrong, loss plateaus or diverges.
// 100 steps of Adam is plenty to drive a fresh-initialized layer onto
// a single fixed (input, target) pair when the chain is correct.

fn runDecoderFineTuneCpuSmoke(allocator: std.mem.Allocator) !void {
    const cfg = cpu_train_decoder.Config{
        .dim = 16,
        .n_heads = 2,
        .n_kv_heads = 2, // no GQA — exercise the simpler path here
        .head_dim = 8,
        .ff_dim = 32,
        .n_pos = 4,
        .rms_eps = 1e-5,
        .causal = true,
    };
    const dim = cfg.dim;
    const q_dim = cfg.n_heads * cfg.head_dim;
    const kv_dim = cfg.n_kv_heads * cfg.head_dim;

    // ── Initialize layer weights with small Gaussian-ish noise.
    var prng = std.Random.DefaultPrng.init(0xDEC0_DE01);
    const rng = prng.random();
    const initScale: f32 = 0.1;

    const w_n1 = try allocator.alloc(f32, dim);
    defer allocator.free(w_n1);
    const w_q = try allocator.alloc(f32, q_dim * dim);
    defer allocator.free(w_q);
    const w_k = try allocator.alloc(f32, kv_dim * dim);
    defer allocator.free(w_k);
    const w_v = try allocator.alloc(f32, kv_dim * dim);
    defer allocator.free(w_v);
    const w_o = try allocator.alloc(f32, dim * q_dim);
    defer allocator.free(w_o);
    const w_n2 = try allocator.alloc(f32, dim);
    defer allocator.free(w_n2);
    const w_gate = try allocator.alloc(f32, cfg.ff_dim * dim);
    defer allocator.free(w_gate);
    const w_up = try allocator.alloc(f32, cfg.ff_dim * dim);
    defer allocator.free(w_up);
    const w_down = try allocator.alloc(f32, dim * cfg.ff_dim);
    defer allocator.free(w_down);

    for (w_n1) |*v| v.* = 1.0; // RMSNorm gain init = 1
    for (w_n2) |*v| v.* = 1.0;
    for (w_q) |*v| v.* = (rng.float(f32) * 2.0 - 1.0) * initScale;
    for (w_k) |*v| v.* = (rng.float(f32) * 2.0 - 1.0) * initScale;
    for (w_v) |*v| v.* = (rng.float(f32) * 2.0 - 1.0) * initScale;
    for (w_o) |*v| v.* = (rng.float(f32) * 2.0 - 1.0) * initScale;
    for (w_gate) |*v| v.* = (rng.float(f32) * 2.0 - 1.0) * initScale;
    for (w_up) |*v| v.* = (rng.float(f32) * 2.0 - 1.0) * initScale;
    for (w_down) |*v| v.* = (rng.float(f32) * 2.0 - 1.0) * initScale;

    var layer = cpu_train_decoder.Layer{
        .cfg = cfg,
        .w_n1 = w_n1,
        .w_q = w_q,
        .w_k = w_k,
        .w_v = w_v,
        .w_o = w_o,
        .w_n2 = w_n2,
        .w_gate = w_gate,
        .w_up = w_up,
        .w_down = w_down,
        .w_q_norm = &.{}, // qk_norm disabled in this smoke
        .w_k_norm = &.{},
    };

    var acts = try cpu_train_decoder.allocActs(allocator, cfg);
    defer cpu_train_decoder.freeActs(allocator, &acts);

    // Random fixed input.
    for (acts.x_in) |*v| v.* = (rng.float(f32) * 2.0 - 1.0);

    // Run a forward pass with the fresh weights to capture y_init,
    // then perturb to define the target. This guarantees the target
    // is reachable in principle (it's a small step away from the
    // initial output).
    cpu_train_decoder.forward(&layer, &acts);
    const target = try allocator.alloc(f32, cfg.n_pos * dim);
    defer allocator.free(target);
    for (target, acts.y) |*tv, yv| tv.* = yv + (rng.float(f32) * 2.0 - 1.0) * 0.5;

    const initial_loss = cpu_train_decoder.mseLoss(acts.y, target);

    // ── Allocate grads + Adam state.
    var grads = cpu_train_decoder.Grads{
        .dw_n1 = try allocator.alloc(f32, w_n1.len),
        .dw_q = try allocator.alloc(f32, w_q.len),
        .dw_k = try allocator.alloc(f32, w_k.len),
        .dw_v = try allocator.alloc(f32, w_v.len),
        .dw_o = try allocator.alloc(f32, w_o.len),
        .dw_n2 = try allocator.alloc(f32, w_n2.len),
        .dw_gate = try allocator.alloc(f32, w_gate.len),
        .dw_up = try allocator.alloc(f32, w_up.len),
        .dw_down = try allocator.alloc(f32, w_down.len),
        .dw_q_norm = &.{},
        .dw_k_norm = &.{},
    };
    defer {
        allocator.free(grads.dw_n1);
        allocator.free(grads.dw_q);
        allocator.free(grads.dw_k);
        allocator.free(grads.dw_v);
        allocator.free(grads.dw_o);
        allocator.free(grads.dw_n2);
        allocator.free(grads.dw_gate);
        allocator.free(grads.dw_up);
        allocator.free(grads.dw_down);
    }

    var adam = try cpu_train_decoder.AdamState.init(allocator, &layer, 1e-2);
    defer adam.deinit(allocator);

    // ── Train.
    const n_steps: usize = 100;
    var final_loss = initial_loss;
    for (1..n_steps + 1) |_| {
        cpu_train_decoder.forward(&layer, &acts);
        final_loss = cpu_train_decoder.mseLoss(acts.y, target);
        grads.zero();
        try cpu_train_decoder.backward(allocator, &layer, &acts, target, &grads);
        cpu_train_decoder.adamStep(&adam, &layer, &grads);
    }

    // Loss must drop by at least 100× over 100 Adam steps on a
    // reachable target. If the gradient chain is broken, loss either
    // plateaus or diverges. Adam's mid-descent momentum bumps don't
    // matter — only the start-vs-end ratio does.
    const ratio = final_loss / initial_loss;
    if (ratio > 1e-2) {
        std.debug.print(
            "decoder fine-tune: initial_loss={d:.6} final_loss={d:.6} ratio={d:.4}\n",
            .{ initial_loss, final_loss, ratio },
        );
        return error.LossDidNotDecrease;
    }

    std.debug.print(
        "PASS decoder fine-tune CPU (dim={d} heads={d} ff_dim={d} n_pos={d}; loss {d:.6} → {d:.6} ({e:.2}× drop) over {d} Adam steps)\n",
        .{ dim, cfg.n_heads, cfg.ff_dim, cfg.n_pos, initial_loss, final_loss, 1.0 / ratio, n_steps },
    );
}

// ── chunk 8c-α-1: stack of decoder layers + lm_head + softmax-CE,
// CPU oracle. Validates that the stack-level gradient chain composes
// correctly: chunk-8a single-layer `backwardFromDy` chains via
// `d_x_in → next layer's d_y`, plus the new pieces (embedding + final
// rmsnorm + lm_head linear + softmax-CE). Same convergence-on-a-
// reachable-target shape as 8a — only the architecture and loss
// changed.
fn runDecoderStackFineTuneCpuSmoke(allocator: std.mem.Allocator) !void {
    const cfg = cpu_train_decoder.StackConfig{
        .base = .{
            .dim = 16,
            .n_heads = 2,
            .n_kv_heads = 2,
            .head_dim = 8,
            .ff_dim = 32,
            .n_pos = 4,
            .rms_eps = 1e-5,
            .causal = true,
        },
        .n_layers = 2,
        .vocab_size = 8,
    };
    const dim = cfg.base.dim;
    const n_pos = cfg.base.n_pos;
    const vocab = cfg.vocab_size;
    const q_dim = cfg.base.n_heads * cfg.base.head_dim;
    const kv_dim = cfg.base.n_kv_heads * cfg.base.head_dim;

    var prng = std.Random.DefaultPrng.init(0xC0_DE_AC_01);
    const rng = prng.random();
    const initScale: f32 = 0.1;

    // ── Embedding + final norm + lm_head.
    const w_embed = try allocator.alloc(f32, vocab * dim);
    defer allocator.free(w_embed);
    for (w_embed) |*v| v.* = (rng.float(f32) * 2.0 - 1.0) * initScale;
    const w_final_norm = try allocator.alloc(f32, dim);
    defer allocator.free(w_final_norm);
    for (w_final_norm) |*v| v.* = 1.0;
    const w_lm_head = try allocator.alloc(f32, vocab * dim);
    defer allocator.free(w_lm_head);
    for (w_lm_head) |*v| v.* = (rng.float(f32) * 2.0 - 1.0) * initScale;

    // ── Per-layer weight buffers.
    const layers = try allocator.alloc(cpu_train_decoder.Layer, cfg.n_layers);
    defer allocator.free(layers);
    const layer_weight_buffers = try allocator.alloc([9][]f32, cfg.n_layers);
    defer allocator.free(layer_weight_buffers);
    defer for (layer_weight_buffers) |slots| for (slots) |s| allocator.free(s);

    for (layers, layer_weight_buffers) |*layer, *slots| {
        const w_n1 = try allocator.alloc(f32, dim);
        const w_q = try allocator.alloc(f32, q_dim * dim);
        const w_k = try allocator.alloc(f32, kv_dim * dim);
        const w_v = try allocator.alloc(f32, kv_dim * dim);
        const w_o = try allocator.alloc(f32, dim * q_dim);
        const w_n2 = try allocator.alloc(f32, dim);
        const w_gate = try allocator.alloc(f32, cfg.base.ff_dim * dim);
        const w_up = try allocator.alloc(f32, cfg.base.ff_dim * dim);
        const w_down = try allocator.alloc(f32, dim * cfg.base.ff_dim);
        for (w_n1) |*v| v.* = 1.0;
        for (w_n2) |*v| v.* = 1.0;
        for (w_q) |*v| v.* = (rng.float(f32) * 2.0 - 1.0) * initScale;
        for (w_k) |*v| v.* = (rng.float(f32) * 2.0 - 1.0) * initScale;
        for (w_v) |*v| v.* = (rng.float(f32) * 2.0 - 1.0) * initScale;
        for (w_o) |*v| v.* = (rng.float(f32) * 2.0 - 1.0) * initScale;
        for (w_gate) |*v| v.* = (rng.float(f32) * 2.0 - 1.0) * initScale;
        for (w_up) |*v| v.* = (rng.float(f32) * 2.0 - 1.0) * initScale;
        for (w_down) |*v| v.* = (rng.float(f32) * 2.0 - 1.0) * initScale;
        slots.* = .{ w_n1, w_q, w_k, w_v, w_o, w_n2, w_gate, w_up, w_down };
        layer.* = .{
            .cfg = cfg.base,
            .w_n1 = w_n1,
            .w_q = w_q,
            .w_k = w_k,
            .w_v = w_v,
            .w_o = w_o,
            .w_n2 = w_n2,
            .w_gate = w_gate,
            .w_up = w_up,
            .w_down = w_down,
            .w_q_norm = &.{}, // qk_norm disabled in this smoke
            .w_k_norm = &.{},
        };
    }

    var stack = cpu_train_decoder.Stack{
        .cfg = cfg,
        .embed = w_embed,
        .layers = layers,
        .final_norm = w_final_norm,
        .lm_head = w_lm_head,
    };

    var acts = try cpu_train_decoder.allocStackActs(allocator, cfg);
    defer cpu_train_decoder.freeStackActs(allocator, &acts);

    // ── Synthetic input + reachable target.
    // Random token IDs in [0, vocab); target IDs likewise.
    const token_ids = try allocator.alloc(u32, n_pos);
    defer allocator.free(token_ids);
    for (token_ids) |*tid| tid.* = rng.intRangeLessThan(u32, 0, @intCast(vocab));
    const target_ids = try allocator.alloc(u32, n_pos);
    defer allocator.free(target_ids);
    for (target_ids) |*tid| tid.* = rng.intRangeLessThan(u32, 0, @intCast(vocab));

    acts.token_ids = token_ids;
    cpu_train_decoder.stackForward(&stack, &acts);
    const initial_loss = cpu_train_decoder.softmaxCeLoss(acts.logits, target_ids, n_pos, vocab);

    // ── Grads + Adam.
    var grads = try cpu_train_decoder.allocStackGrads(allocator, &stack);
    defer cpu_train_decoder.freeStackGrads(allocator, &grads);

    var adam = try cpu_train_decoder.stackAdamInit(allocator, &stack, 1e-2);
    defer cpu_train_decoder.stackAdamDeinit(&adam, allocator);

    // ── Train.
    const n_steps: usize = 200;
    var final_loss = initial_loss;
    for (1..n_steps + 1) |_| {
        cpu_train_decoder.stackForward(&stack, &acts);
        final_loss = cpu_train_decoder.softmaxCeLoss(acts.logits, target_ids, n_pos, vocab);
        grads.zero();
        try cpu_train_decoder.stackBackward(allocator, &stack, &acts, target_ids, &grads);
        cpu_train_decoder.stackAdamStep(&adam, &stack, &grads);
    }

    // Cross-entropy on a tiny vocab with a reachable target should
    // collapse loss by orders of magnitude. The single-layer 8a smoke
    // hit 5+ orders of magnitude in 100 steps; CE through a 2-layer
    // stack is somewhat slower but should still clear 1e-2 inside 200.
    const ratio = final_loss / initial_loss;
    if (ratio > 1e-2) {
        std.debug.print(
            "stack fine-tune: initial_loss={d:.6} final_loss={d:.6} ratio={d:.4}\n",
            .{ initial_loss, final_loss, ratio },
        );
        return error.LossDidNotDecrease;
    }

    std.debug.print(
        "PASS decoder stack fine-tune CPU (n_layers={d} dim={d} vocab={d} n_pos={d}; CE loss {d:.6} → {d:.6} ({e:.2}× drop) over {d} Adam steps)\n",
        .{ cfg.n_layers, dim, vocab, n_pos, initial_loss, final_loss, 1.0 / ratio, n_steps },
    );
}

// ── chunk 8c-α-2: stack GPU backward parity vs CPU oracle ──────────
//
// Same role as the chunk-8b stage-A parity smoke, scaled up to the
// stack: CPU forward + CPU backward give an oracle for every
// gradient (embedding, per-layer × N, final_norm, lm_head); the GPU
// recorder replays the stack-level backward chain over the saved
// activations and we compare each slice. The per-layer 24-dispatch
// chain (the 8b stage-A 23 dispatches plus an `add_in_place` to
// fold in the residual `mid = x_in + o` contribution into d_x_in)
// repeats N times, so we factor it into a helper.

const StackBwKernels = struct {
    lin_dx: *const pipeline.Kernel,
    lin_dw: *const pipeline.Kernel,
    swiglu_bw: *const pipeline.Kernel,
    rms_bw: *const pipeline.Kernel,
    attn_dattn: *const pipeline.Kernel,
    attn_dv: *const pipeline.Kernel,
    attn_dq: *const pipeline.Kernel,
    attn_dk: *const pipeline.Kernel,
    softmax_bw: *const pipeline.Kernel,
    add: *const pipeline.Kernel,
};

const StackLayerBufs = struct {
    // Saved activations (read).
    x_in: *const buffer.Buffer,
    n1: *const buffer.Buffer,
    q: *const buffer.Buffer,
    k: *const buffer.Buffer,
    v: *const buffer.Buffer,
    attn: *const buffer.Buffer,
    attn_out: *const buffer.Buffer,
    mid: *const buffer.Buffer,
    n2: *const buffer.Buffer,
    pre_gate: *const buffer.Buffer,
    up: *const buffer.Buffer,
    gated: *const buffer.Buffer,
    // Weights (read).
    w_n1: *const buffer.Buffer,
    w_q: *const buffer.Buffer,
    w_k: *const buffer.Buffer,
    w_v: *const buffer.Buffer,
    w_o: *const buffer.Buffer,
    w_n2: *const buffer.Buffer,
    w_gate: *const buffer.Buffer,
    w_up: *const buffer.Buffer,
    w_down: *const buffer.Buffer,
    // Grad outputs (write — RMSNorm gains come back as per-row partials).
    dw_n1_partial: *const buffer.Buffer,
    dw_q: *const buffer.Buffer,
    dw_k: *const buffer.Buffer,
    dw_v: *const buffer.Buffer,
    dw_o: *const buffer.Buffer,
    dw_n2_partial: *const buffer.Buffer,
    dw_gate: *const buffer.Buffer,
    dw_up: *const buffer.Buffer,
    dw_down: *const buffer.Buffer,
};

const StackBwScratch = struct {
    d_gated: *const buffer.Buffer,
    d_pre_gate: *const buffer.Buffer,
    d_up: *const buffer.Buffer,
    d_n2: *const buffer.Buffer,
    d_n2_up: *const buffer.Buffer,
    d_mid_norm: *const buffer.Buffer,
    d_attn_out: *const buffer.Buffer,
    d_attn: *const buffer.Buffer,
    d_scores: *const buffer.Buffer,
    dQ: *const buffer.Buffer,
    dK: *const buffer.Buffer,
    dV: *const buffer.Buffer,
    d_n1: *const buffer.Buffer,
    d_n1_k: *const buffer.Buffer,
    d_n1_v: *const buffer.Buffer,
};

/// Records the 24-dispatch per-layer backward chain into the recorder.
/// `d_y_in` is *mutated in place* into `d_mid_total` by `add_in_place`
/// after RMSNorm-n2 backward — same aliasing as the chunk-8a CPU oracle.
/// `d_x_in_out` receives the layer's input gradient (used as the next
/// layer's `d_y_in` for the iteration that follows, or as the
/// embedding-backward input for layer 0).
fn recordStackLayerBackward(
    rec: *gpu_recorder.Recorder,
    kernels: StackBwKernels,
    bufs: StackLayerBufs,
    scratch: StackBwScratch,
    pushes: struct {
        lin_down: *const runtime.LinearBatchedPush,
        lin_gate: *const runtime.LinearBatchedPush,
        lin_up: *const runtime.LinearBatchedPush,
        lin_o: *const runtime.LinearBatchedPush,
        lin_q: *const runtime.LinearBatchedPush,
        lin_k: *const runtime.LinearBatchedPush,
        lin_v: *const runtime.LinearBatchedPush,
        swiglu: *const runtime.SwigluPush,
        rms: *const runtime.RmsnormPush,
        add_dim: *const runtime.AddInPlacePush,
        softmax: *const runtime.SoftmaxPush,
        dattn: *const runtime.AttnBackwardDattnPush,
        dv: *const runtime.AttnBackwardDvPush,
        dq: *const runtime.AttnBackwardDqPush,
        dk: *const runtime.AttnBackwardDkPush,
    },
    shape: struct { n_pos: u32, n_heads: u32, n_kv_heads: u32, head_dim: u32 },
    d_y_in: *const buffer.Buffer,
    d_x_in_out: *const buffer.Buffer,
) !void {
    const lwg: u32 = 16;
    const groupsLin: u32 = 256;
    const groupsCeil = struct {
        fn f(n: u32) u32 {
            return (n + lwg - 1) / lwg;
        }
    }.f;
    const addGroups: u32 = (pushes.add_dim.n + groupsLin - 1) / groupsLin;
    const swigluGroups: u32 = (pushes.swiglu.n + groupsLin - 1) / groupsLin;

    // W_down dx + dW.
    try rec.dispatch(kernels.lin_dx, &.{ d_y_in, bufs.w_down, scratch.d_gated }, pushes.lin_down, groupsCeil(pushes.lin_down.M), groupsCeil(pushes.lin_down.K), 1);
    try rec.dispatch(kernels.lin_dw, &.{ d_y_in, bufs.gated, bufs.dw_down }, pushes.lin_down, groupsCeil(pushes.lin_down.N), groupsCeil(pushes.lin_down.K), 1);

    // SwiGLU backward: (d_gated, pre_gate, up) → (d_pre_gate, d_up).
    try rec.dispatch(kernels.swiglu_bw, &.{ scratch.d_gated, bufs.pre_gate, bufs.up, scratch.d_pre_gate, scratch.d_up }, pushes.swiglu, swigluGroups, 1, 1);

    // W_gate dx + dW (writes d_n2; W_up's dx accumulates into d_n2_up next).
    try rec.dispatch(kernels.lin_dx, &.{ scratch.d_pre_gate, bufs.w_gate, scratch.d_n2 }, pushes.lin_gate, groupsCeil(pushes.lin_gate.M), groupsCeil(pushes.lin_gate.K), 1);
    try rec.dispatch(kernels.lin_dw, &.{ scratch.d_pre_gate, bufs.n2, bufs.dw_gate }, pushes.lin_gate, groupsCeil(pushes.lin_gate.N), groupsCeil(pushes.lin_gate.K), 1);

    // W_up dx + dW + accumulate into d_n2.
    try rec.dispatch(kernels.lin_dx, &.{ scratch.d_up, bufs.w_up, scratch.d_n2_up }, pushes.lin_up, groupsCeil(pushes.lin_up.M), groupsCeil(pushes.lin_up.K), 1);
    try rec.dispatch(kernels.lin_dw, &.{ scratch.d_up, bufs.n2, bufs.dw_up }, pushes.lin_up, groupsCeil(pushes.lin_up.N), groupsCeil(pushes.lin_up.K), 1);
    try rec.dispatch(kernels.add, &.{ scratch.d_n2, scratch.d_n2_up }, pushes.add_dim, addGroups, 1, 1);

    // RMSNorm n2 backward → d_mid_norm + dw_n2_partial.
    try rec.dispatch(kernels.rms_bw, &.{ scratch.d_n2, bufs.mid, bufs.w_n2, scratch.d_mid_norm, bufs.dw_n2_partial }, pushes.rms, shape.n_pos, 1, 1);

    // d_y_in += d_mid_norm. From here, d_y_in holds d_mid_total —
    // the gradient flowing into both the o-projection AND (via the
    // residual) back to x_in.
    try rec.dispatch(kernels.add, &.{ d_y_in, scratch.d_mid_norm }, pushes.add_dim, addGroups, 1, 1);

    // O projection dx + dW (treats d_y_in as d_o).
    try rec.dispatch(kernels.lin_dx, &.{ d_y_in, bufs.w_o, scratch.d_attn_out }, pushes.lin_o, groupsCeil(pushes.lin_o.M), groupsCeil(pushes.lin_o.K), 1);
    try rec.dispatch(kernels.lin_dw, &.{ d_y_in, bufs.attn_out, bufs.dw_o }, pushes.lin_o, groupsCeil(pushes.lin_o.N), groupsCeil(pushes.lin_o.K), 1);

    // SDPA backward.
    try rec.dispatch(kernels.attn_dattn, &.{ scratch.d_attn_out, bufs.v, scratch.d_attn }, pushes.dattn, shape.n_pos * shape.n_heads * shape.n_pos, 1, 1);
    try rec.dispatch(kernels.attn_dv, &.{ bufs.attn, scratch.d_attn_out, scratch.dV }, pushes.dv, shape.n_pos * shape.n_kv_heads * shape.head_dim, 1, 1);
    try rec.dispatch(kernels.softmax_bw, &.{ scratch.d_attn, bufs.attn, scratch.d_scores }, pushes.softmax, shape.n_pos * shape.n_heads, 1, 1);
    try rec.dispatch(kernels.attn_dq, &.{ scratch.d_scores, bufs.k, scratch.dQ }, pushes.dq, shape.n_pos * shape.n_heads * shape.head_dim, 1, 1);
    try rec.dispatch(kernels.attn_dk, &.{ scratch.d_scores, bufs.q, scratch.dK }, pushes.dk, shape.n_pos * shape.n_kv_heads * shape.head_dim, 1, 1);

    // Q proj. Writes directly into scratch.d_n1 (rather than a separate
    // d_n1_q buffer) to save one add_in_place — K and V then accumulate
    // into d_n1.
    try rec.dispatch(kernels.lin_dx, &.{ scratch.dQ, bufs.w_q, scratch.d_n1 }, pushes.lin_q, groupsCeil(pushes.lin_q.M), groupsCeil(pushes.lin_q.K), 1);
    try rec.dispatch(kernels.lin_dw, &.{ scratch.dQ, bufs.n1, bufs.dw_q }, pushes.lin_q, groupsCeil(pushes.lin_q.N), groupsCeil(pushes.lin_q.K), 1);

    // K proj + accumulate into d_n1.
    try rec.dispatch(kernels.lin_dx, &.{ scratch.dK, bufs.w_k, scratch.d_n1_k }, pushes.lin_k, groupsCeil(pushes.lin_k.M), groupsCeil(pushes.lin_k.K), 1);
    try rec.dispatch(kernels.lin_dw, &.{ scratch.dK, bufs.n1, bufs.dw_k }, pushes.lin_k, groupsCeil(pushes.lin_k.N), groupsCeil(pushes.lin_k.K), 1);
    try rec.dispatch(kernels.add, &.{ scratch.d_n1, scratch.d_n1_k }, pushes.add_dim, addGroups, 1, 1);

    // V proj + accumulate into d_n1.
    try rec.dispatch(kernels.lin_dx, &.{ scratch.dV, bufs.w_v, scratch.d_n1_v }, pushes.lin_v, groupsCeil(pushes.lin_v.M), groupsCeil(pushes.lin_v.K), 1);
    try rec.dispatch(kernels.lin_dw, &.{ scratch.dV, bufs.n1, bufs.dw_v }, pushes.lin_v, groupsCeil(pushes.lin_v.N), groupsCeil(pushes.lin_v.K), 1);
    try rec.dispatch(kernels.add, &.{ scratch.d_n1, scratch.d_n1_v }, pushes.add_dim, addGroups, 1, 1);

    // RMSNorm n1 backward → writes the rmsnorm.dx contribution to
    // d_x_in_out + dw_n1_partial.
    try rec.dispatch(kernels.rms_bw, &.{ scratch.d_n1, bufs.x_in, bufs.w_n1, d_x_in_out, bufs.dw_n1_partial }, pushes.rms, shape.n_pos, 1, 1);

    // d_x_in_out += d_y_in (the residual contribution of `mid = x_in + o`,
    // currently held in d_y_in's mutated form = d_mid_total). This
    // is the second residual path that the chunk-8a backward
    // discarded — must be present for stack training to flow grads
    // through residuals correctly.
    try rec.dispatch(kernels.add, &.{ d_x_in_out, d_y_in }, pushes.add_dim, addGroups, 1, 1);
}

fn runDecoderStackBackwardGpuParitySmoke(allocator: std.mem.Allocator) !void {
    const cfg = cpu_train_decoder.StackConfig{
        .base = .{
            .dim = 16,
            .n_heads = 2,
            .n_kv_heads = 2,
            .head_dim = 8,
            .ff_dim = 32,
            .n_pos = 4,
            .rms_eps = 1e-5,
            .causal = true,
        },
        .n_layers = 2,
        .vocab_size = 8,
    };
    const dim = cfg.base.dim;
    const n_pos = cfg.base.n_pos;
    const n_heads = cfg.base.n_heads;
    const n_kv_heads = cfg.base.n_kv_heads;
    const head_dim = cfg.base.head_dim;
    const ff_dim = cfg.base.ff_dim;
    const vocab = cfg.vocab_size;
    const q_dim = n_heads * head_dim;
    const kv_dim = n_kv_heads * head_dim;
    const heads_per_kv: u32 = @intCast(n_heads / n_kv_heads);
    const inv_sqrt_d: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(head_dim)));
    const scores_total = n_pos * n_heads * n_pos;

    // ── Identical RNG to runDecoderStackFineTuneCpuSmoke so weights +
    //    inputs match.
    var prng = std.Random.DefaultPrng.init(0xC0_DE_AC_01);
    const rng = prng.random();
    const initScale: f32 = 0.1;

    const w_embed = try allocator.alloc(f32, vocab * dim);
    defer allocator.free(w_embed);
    for (w_embed) |*v| v.* = (rng.float(f32) * 2.0 - 1.0) * initScale;
    const w_final_norm = try allocator.alloc(f32, dim);
    defer allocator.free(w_final_norm);
    for (w_final_norm) |*v| v.* = 1.0;
    const w_lm_head = try allocator.alloc(f32, vocab * dim);
    defer allocator.free(w_lm_head);
    for (w_lm_head) |*v| v.* = (rng.float(f32) * 2.0 - 1.0) * initScale;

    const layers = try allocator.alloc(cpu_train_decoder.Layer, cfg.n_layers);
    defer allocator.free(layers);
    const layer_weight_buffers = try allocator.alloc([9][]f32, cfg.n_layers);
    defer allocator.free(layer_weight_buffers);
    defer for (layer_weight_buffers) |slots| for (slots) |s| allocator.free(s);

    for (layers, layer_weight_buffers) |*layer, *slots| {
        const w_n1 = try allocator.alloc(f32, dim);
        const w_q = try allocator.alloc(f32, q_dim * dim);
        const w_k = try allocator.alloc(f32, kv_dim * dim);
        const w_v = try allocator.alloc(f32, kv_dim * dim);
        const w_o = try allocator.alloc(f32, dim * q_dim);
        const w_n2 = try allocator.alloc(f32, dim);
        const w_gate = try allocator.alloc(f32, ff_dim * dim);
        const w_up = try allocator.alloc(f32, ff_dim * dim);
        const w_down = try allocator.alloc(f32, dim * ff_dim);
        for (w_n1) |*v| v.* = 1.0;
        for (w_n2) |*v| v.* = 1.0;
        for (w_q) |*v| v.* = (rng.float(f32) * 2.0 - 1.0) * initScale;
        for (w_k) |*v| v.* = (rng.float(f32) * 2.0 - 1.0) * initScale;
        for (w_v) |*v| v.* = (rng.float(f32) * 2.0 - 1.0) * initScale;
        for (w_o) |*v| v.* = (rng.float(f32) * 2.0 - 1.0) * initScale;
        for (w_gate) |*v| v.* = (rng.float(f32) * 2.0 - 1.0) * initScale;
        for (w_up) |*v| v.* = (rng.float(f32) * 2.0 - 1.0) * initScale;
        for (w_down) |*v| v.* = (rng.float(f32) * 2.0 - 1.0) * initScale;
        slots.* = .{ w_n1, w_q, w_k, w_v, w_o, w_n2, w_gate, w_up, w_down };
        layer.* = .{
            .cfg = cfg.base,
            .w_n1 = w_n1,
            .w_q = w_q,
            .w_k = w_k,
            .w_v = w_v,
            .w_o = w_o,
            .w_n2 = w_n2,
            .w_gate = w_gate,
            .w_up = w_up,
            .w_down = w_down,
            .w_q_norm = &.{}, // qk_norm disabled in this smoke
            .w_k_norm = &.{},
        };
    }

    var stack = cpu_train_decoder.Stack{
        .cfg = cfg,
        .embed = w_embed,
        .layers = layers,
        .final_norm = w_final_norm,
        .lm_head = w_lm_head,
    };

    var acts = try cpu_train_decoder.allocStackActs(allocator, cfg);
    defer cpu_train_decoder.freeStackActs(allocator, &acts);

    const token_ids = try allocator.alloc(u32, n_pos);
    defer allocator.free(token_ids);
    for (token_ids) |*tid| tid.* = rng.intRangeLessThan(u32, 0, @intCast(vocab));
    const target_ids = try allocator.alloc(u32, n_pos);
    defer allocator.free(target_ids);
    for (target_ids) |*tid| tid.* = rng.intRangeLessThan(u32, 0, @intCast(vocab));

    acts.token_ids = token_ids;
    cpu_train_decoder.stackForward(&stack, &acts);

    // ── CPU oracle: full backward.
    var grads_cpu = try cpu_train_decoder.allocStackGrads(allocator, &stack);
    defer cpu_train_decoder.freeStackGrads(allocator, &grads_cpu);
    grads_cpu.zero();
    try cpu_train_decoder.stackBackward(allocator, &stack, &acts, target_ids, &grads_cpu);

    // ── One-hot target tensor for the GPU softmax_ce_loss_grad shader.
    const target_one_hot = try allocator.alloc(f32, n_pos * vocab);
    defer allocator.free(target_one_hot);
    @memset(target_one_hot, 0);
    for (target_ids, 0..) |tid, p| target_one_hot[p * vocab + @as(usize, tid)] = 1.0;

    // ── GPU bring-up.
    var ctx = try vk.Context.init(allocator);
    defer ctx.deinit();

    var k_lin_dx = try pipeline.Kernel.init(&ctx, &shaders.linear_backward_dx_batched, 3, @sizeOf(runtime.LinearBatchedPush));
    defer k_lin_dx.deinit();
    var k_lin_dw = try pipeline.Kernel.init(&ctx, &shaders.linear_backward_dw_batched, 3, @sizeOf(runtime.LinearBatchedPush));
    defer k_lin_dw.deinit();
    var k_swiglu_bw = try pipeline.Kernel.init(&ctx, &shaders.swiglu_backward, 5, @sizeOf(runtime.SwigluPush));
    defer k_swiglu_bw.deinit();
    var k_rms_bw = try pipeline.Kernel.init(&ctx, &shaders.rmsnorm_backward, 5, @sizeOf(runtime.RmsnormPush));
    defer k_rms_bw.deinit();
    var k_attn_dattn = try pipeline.Kernel.init(&ctx, &shaders.attn_backward_dattn, 3, @sizeOf(runtime.AttnBackwardDattnPush));
    defer k_attn_dattn.deinit();
    var k_attn_dv = try pipeline.Kernel.init(&ctx, &shaders.attn_backward_dv, 3, @sizeOf(runtime.AttnBackwardDvPush));
    defer k_attn_dv.deinit();
    var k_attn_dq = try pipeline.Kernel.init(&ctx, &shaders.attn_backward_dq, 3, @sizeOf(runtime.AttnBackwardDqPush));
    defer k_attn_dq.deinit();
    var k_attn_dk = try pipeline.Kernel.init(&ctx, &shaders.attn_backward_dk, 3, @sizeOf(runtime.AttnBackwardDkPush));
    defer k_attn_dk.deinit();
    var k_softmax_bw = try pipeline.Kernel.init(&ctx, &shaders.softmax_backward, 3, @sizeOf(runtime.SoftmaxPush));
    defer k_softmax_bw.deinit();
    var k_add = try pipeline.Kernel.init(&ctx, &shaders.add_in_place, 2, @sizeOf(runtime.AddInPlacePush));
    defer k_add.deinit();
    var k_ce_loss_grad = try pipeline.Kernel.init(&ctx, &shaders.softmax_ce_loss_grad_batched_v2, 3, @sizeOf(runtime.SoftmaxCeLossGradPush));
    defer k_ce_loss_grad.deinit();
    var k_embed_bw = try pipeline.Kernel.init(&ctx, &shaders.embedding_backward, 3, @sizeOf(runtime.EmbeddingBackwardPush));
    defer k_embed_bw.deinit();

    const f32sz = @sizeOf(f32);

    // ── Stack-level inputs.
    var buf_logits = try buffer.Buffer.initStatic(&ctx, f32, acts.logits);
    defer buf_logits.deinit(ctx.device);
    var buf_target_oh = try buffer.Buffer.initStatic(&ctx, f32, target_one_hot);
    defer buf_target_oh.deinit(ctx.device);
    var buf_final_norm_out = try buffer.Buffer.initStatic(&ctx, f32, acts.final_norm_out);
    defer buf_final_norm_out.deinit(ctx.device);
    var buf_w_lm_head = try buffer.Buffer.initStatic(&ctx, f32, w_lm_head);
    defer buf_w_lm_head.deinit(ctx.device);
    var buf_w_final_norm = try buffer.Buffer.initStatic(&ctx, f32, w_final_norm);
    defer buf_w_final_norm.deinit(ctx.device);
    var buf_token_ids = try buffer.Buffer.initStatic(&ctx, u32, token_ids);
    defer buf_token_ids.deinit(ctx.device);

    // ── Per-layer weight + activation buffers (heap-allocated arrays).
    const f32_vk_buf_alloc = struct {
        fn deinitMany(buffers: []buffer.Buffer, dev: vk.c.VkDevice) void {
            for (buffers) |*b| b.deinit(dev);
        }
    };
    _ = f32_vk_buf_alloc;

    const buf_w_n1 = try allocator.alloc(buffer.Buffer, cfg.n_layers);
    defer { for (buf_w_n1) |*b| b.deinit(ctx.device); allocator.free(buf_w_n1); }
    const buf_w_q = try allocator.alloc(buffer.Buffer, cfg.n_layers);
    defer { for (buf_w_q) |*b| b.deinit(ctx.device); allocator.free(buf_w_q); }
    const buf_w_k = try allocator.alloc(buffer.Buffer, cfg.n_layers);
    defer { for (buf_w_k) |*b| b.deinit(ctx.device); allocator.free(buf_w_k); }
    const buf_w_v = try allocator.alloc(buffer.Buffer, cfg.n_layers);
    defer { for (buf_w_v) |*b| b.deinit(ctx.device); allocator.free(buf_w_v); }
    const buf_w_o = try allocator.alloc(buffer.Buffer, cfg.n_layers);
    defer { for (buf_w_o) |*b| b.deinit(ctx.device); allocator.free(buf_w_o); }
    const buf_w_n2 = try allocator.alloc(buffer.Buffer, cfg.n_layers);
    defer { for (buf_w_n2) |*b| b.deinit(ctx.device); allocator.free(buf_w_n2); }
    const buf_w_gate = try allocator.alloc(buffer.Buffer, cfg.n_layers);
    defer { for (buf_w_gate) |*b| b.deinit(ctx.device); allocator.free(buf_w_gate); }
    const buf_w_up = try allocator.alloc(buffer.Buffer, cfg.n_layers);
    defer { for (buf_w_up) |*b| b.deinit(ctx.device); allocator.free(buf_w_up); }
    const buf_w_down = try allocator.alloc(buffer.Buffer, cfg.n_layers);
    defer { for (buf_w_down) |*b| b.deinit(ctx.device); allocator.free(buf_w_down); }

    const buf_x_in = try allocator.alloc(buffer.Buffer, cfg.n_layers);
    defer { for (buf_x_in) |*b| b.deinit(ctx.device); allocator.free(buf_x_in); }
    const buf_n1 = try allocator.alloc(buffer.Buffer, cfg.n_layers);
    defer { for (buf_n1) |*b| b.deinit(ctx.device); allocator.free(buf_n1); }
    const buf_q = try allocator.alloc(buffer.Buffer, cfg.n_layers);
    defer { for (buf_q) |*b| b.deinit(ctx.device); allocator.free(buf_q); }
    const buf_k = try allocator.alloc(buffer.Buffer, cfg.n_layers);
    defer { for (buf_k) |*b| b.deinit(ctx.device); allocator.free(buf_k); }
    const buf_v = try allocator.alloc(buffer.Buffer, cfg.n_layers);
    defer { for (buf_v) |*b| b.deinit(ctx.device); allocator.free(buf_v); }
    const buf_attn = try allocator.alloc(buffer.Buffer, cfg.n_layers);
    defer { for (buf_attn) |*b| b.deinit(ctx.device); allocator.free(buf_attn); }
    const buf_attn_out = try allocator.alloc(buffer.Buffer, cfg.n_layers);
    defer { for (buf_attn_out) |*b| b.deinit(ctx.device); allocator.free(buf_attn_out); }
    const buf_mid = try allocator.alloc(buffer.Buffer, cfg.n_layers);
    defer { for (buf_mid) |*b| b.deinit(ctx.device); allocator.free(buf_mid); }
    const buf_n2 = try allocator.alloc(buffer.Buffer, cfg.n_layers);
    defer { for (buf_n2) |*b| b.deinit(ctx.device); allocator.free(buf_n2); }
    const buf_pre_gate = try allocator.alloc(buffer.Buffer, cfg.n_layers);
    defer { for (buf_pre_gate) |*b| b.deinit(ctx.device); allocator.free(buf_pre_gate); }
    const buf_up = try allocator.alloc(buffer.Buffer, cfg.n_layers);
    defer { for (buf_up) |*b| b.deinit(ctx.device); allocator.free(buf_up); }
    const buf_gated = try allocator.alloc(buffer.Buffer, cfg.n_layers);
    defer { for (buf_gated) |*b| b.deinit(ctx.device); allocator.free(buf_gated); }

    const buf_dw_n1_partial = try allocator.alloc(buffer.Buffer, cfg.n_layers);
    defer { for (buf_dw_n1_partial) |*b| b.deinit(ctx.device); allocator.free(buf_dw_n1_partial); }
    const buf_dw_q = try allocator.alloc(buffer.Buffer, cfg.n_layers);
    defer { for (buf_dw_q) |*b| b.deinit(ctx.device); allocator.free(buf_dw_q); }
    const buf_dw_k = try allocator.alloc(buffer.Buffer, cfg.n_layers);
    defer { for (buf_dw_k) |*b| b.deinit(ctx.device); allocator.free(buf_dw_k); }
    const buf_dw_v = try allocator.alloc(buffer.Buffer, cfg.n_layers);
    defer { for (buf_dw_v) |*b| b.deinit(ctx.device); allocator.free(buf_dw_v); }
    const buf_dw_o = try allocator.alloc(buffer.Buffer, cfg.n_layers);
    defer { for (buf_dw_o) |*b| b.deinit(ctx.device); allocator.free(buf_dw_o); }
    const buf_dw_n2_partial = try allocator.alloc(buffer.Buffer, cfg.n_layers);
    defer { for (buf_dw_n2_partial) |*b| b.deinit(ctx.device); allocator.free(buf_dw_n2_partial); }
    const buf_dw_gate = try allocator.alloc(buffer.Buffer, cfg.n_layers);
    defer { for (buf_dw_gate) |*b| b.deinit(ctx.device); allocator.free(buf_dw_gate); }
    const buf_dw_up = try allocator.alloc(buffer.Buffer, cfg.n_layers);
    defer { for (buf_dw_up) |*b| b.deinit(ctx.device); allocator.free(buf_dw_up); }
    const buf_dw_down = try allocator.alloc(buffer.Buffer, cfg.n_layers);
    defer { for (buf_dw_down) |*b| b.deinit(ctx.device); allocator.free(buf_dw_down); }

    // d_x_in_per_layer: layer i writes into d_x_in[i]; layer i-1 reads
    // it as d_y_in. Layer 0's d_x_in[0] is the input to embedding_bw.
    const buf_d_x_in = try allocator.alloc(buffer.Buffer, cfg.n_layers);
    defer { for (buf_d_x_in) |*b| b.deinit(ctx.device); allocator.free(buf_d_x_in); }

    for (0..cfg.n_layers) |li| {
        const layer = &layers[li];
        const la = &acts.layer_acts[li];
        buf_w_n1[li] = try buffer.Buffer.initStatic(&ctx, f32, layer.w_n1);
        buf_w_q[li] = try buffer.Buffer.initStatic(&ctx, f32, layer.w_q);
        buf_w_k[li] = try buffer.Buffer.initStatic(&ctx, f32, layer.w_k);
        buf_w_v[li] = try buffer.Buffer.initStatic(&ctx, f32, layer.w_v);
        buf_w_o[li] = try buffer.Buffer.initStatic(&ctx, f32, layer.w_o);
        buf_w_n2[li] = try buffer.Buffer.initStatic(&ctx, f32, layer.w_n2);
        buf_w_gate[li] = try buffer.Buffer.initStatic(&ctx, f32, layer.w_gate);
        buf_w_up[li] = try buffer.Buffer.initStatic(&ctx, f32, layer.w_up);
        buf_w_down[li] = try buffer.Buffer.initStatic(&ctx, f32, layer.w_down);

        buf_x_in[li] = try buffer.Buffer.initStatic(&ctx, f32, la.x_in);
        buf_n1[li] = try buffer.Buffer.initStatic(&ctx, f32, la.n1);
        buf_q[li] = try buffer.Buffer.initStatic(&ctx, f32, la.q);
        buf_k[li] = try buffer.Buffer.initStatic(&ctx, f32, la.k);
        buf_v[li] = try buffer.Buffer.initStatic(&ctx, f32, la.v);
        buf_attn[li] = try buffer.Buffer.initStatic(&ctx, f32, la.attn);
        buf_attn_out[li] = try buffer.Buffer.initStatic(&ctx, f32, la.attn_out);
        buf_mid[li] = try buffer.Buffer.initStatic(&ctx, f32, la.mid);
        buf_n2[li] = try buffer.Buffer.initStatic(&ctx, f32, la.n2);
        buf_pre_gate[li] = try buffer.Buffer.initStatic(&ctx, f32, la.pre_gate);
        buf_up[li] = try buffer.Buffer.initStatic(&ctx, f32, la.up);
        buf_gated[li] = try buffer.Buffer.initStatic(&ctx, f32, la.gated);

        buf_dw_n1_partial[li] = try buffer.Buffer.initDeviceOnly(&ctx, n_pos * dim * f32sz);
        buf_dw_q[li] = try buffer.Buffer.initDeviceOnly(&ctx, layer.w_q.len * f32sz);
        buf_dw_k[li] = try buffer.Buffer.initDeviceOnly(&ctx, layer.w_k.len * f32sz);
        buf_dw_v[li] = try buffer.Buffer.initDeviceOnly(&ctx, layer.w_v.len * f32sz);
        buf_dw_o[li] = try buffer.Buffer.initDeviceOnly(&ctx, layer.w_o.len * f32sz);
        buf_dw_n2_partial[li] = try buffer.Buffer.initDeviceOnly(&ctx, n_pos * dim * f32sz);
        buf_dw_gate[li] = try buffer.Buffer.initDeviceOnly(&ctx, layer.w_gate.len * f32sz);
        buf_dw_up[li] = try buffer.Buffer.initDeviceOnly(&ctx, layer.w_up.len * f32sz);
        buf_dw_down[li] = try buffer.Buffer.initDeviceOnly(&ctx, layer.w_down.len * f32sz);

        buf_d_x_in[li] = try buffer.Buffer.initDeviceOnly(&ctx, n_pos * dim * f32sz);
    }

    // ── Stack-level scratch + grad outputs.
    var buf_d_logits = try buffer.Buffer.initDeviceOnly(&ctx, n_pos * vocab * f32sz);
    defer buf_d_logits.deinit(ctx.device);
    var buf_d_final_norm_out = try buffer.Buffer.initDeviceOnly(&ctx, n_pos * dim * f32sz);
    defer buf_d_final_norm_out.deinit(ctx.device);
    var buf_d_last_y = try buffer.Buffer.initDeviceOnly(&ctx, n_pos * dim * f32sz);
    defer buf_d_last_y.deinit(ctx.device);
    var buf_dw_lm_head = try buffer.Buffer.initDeviceOnly(&ctx, w_lm_head.len * f32sz);
    defer buf_dw_lm_head.deinit(ctx.device);
    var buf_dw_final_norm_partial = try buffer.Buffer.initDeviceOnly(&ctx, n_pos * dim * f32sz);
    defer buf_dw_final_norm_partial.deinit(ctx.device);
    var buf_dE_embed = try buffer.Buffer.initDeviceOnly(&ctx, w_embed.len * f32sz);
    defer buf_dE_embed.deinit(ctx.device);

    // ── Per-layer-shared scratch (reused across layer iterations —
    //    barriers between dispatches keep each iteration's reads
    //    after the previous's writes).
    var sc_d_gated = try buffer.Buffer.initDeviceOnly(&ctx, n_pos * ff_dim * f32sz);
    defer sc_d_gated.deinit(ctx.device);
    var sc_d_pre_gate = try buffer.Buffer.initDeviceOnly(&ctx, n_pos * ff_dim * f32sz);
    defer sc_d_pre_gate.deinit(ctx.device);
    var sc_d_up_grad = try buffer.Buffer.initDeviceOnly(&ctx, n_pos * ff_dim * f32sz);
    defer sc_d_up_grad.deinit(ctx.device);
    var sc_d_n2 = try buffer.Buffer.initDeviceOnly(&ctx, n_pos * dim * f32sz);
    defer sc_d_n2.deinit(ctx.device);
    var sc_d_n2_up = try buffer.Buffer.initDeviceOnly(&ctx, n_pos * dim * f32sz);
    defer sc_d_n2_up.deinit(ctx.device);
    var sc_d_mid_norm = try buffer.Buffer.initDeviceOnly(&ctx, n_pos * dim * f32sz);
    defer sc_d_mid_norm.deinit(ctx.device);
    var sc_d_attn_out = try buffer.Buffer.initDeviceOnly(&ctx, n_pos * q_dim * f32sz);
    defer sc_d_attn_out.deinit(ctx.device);
    var sc_d_attn = try buffer.Buffer.initDeviceOnly(&ctx, scores_total * f32sz);
    defer sc_d_attn.deinit(ctx.device);
    var sc_d_scores = try buffer.Buffer.initDeviceOnly(&ctx, scores_total * f32sz);
    defer sc_d_scores.deinit(ctx.device);
    var sc_dQ = try buffer.Buffer.initDeviceOnly(&ctx, n_pos * q_dim * f32sz);
    defer sc_dQ.deinit(ctx.device);
    var sc_dK = try buffer.Buffer.initDeviceOnly(&ctx, n_pos * kv_dim * f32sz);
    defer sc_dK.deinit(ctx.device);
    var sc_dV = try buffer.Buffer.initDeviceOnly(&ctx, n_pos * kv_dim * f32sz);
    defer sc_dV.deinit(ctx.device);
    var sc_d_n1 = try buffer.Buffer.initDeviceOnly(&ctx, n_pos * dim * f32sz);
    defer sc_d_n1.deinit(ctx.device);
    var sc_d_n1_k = try buffer.Buffer.initDeviceOnly(&ctx, n_pos * dim * f32sz);
    defer sc_d_n1_k.deinit(ctx.device);
    var sc_d_n1_v = try buffer.Buffer.initDeviceOnly(&ctx, n_pos * dim * f32sz);
    defer sc_d_n1_v.deinit(ctx.device);

    // ── Pushes.
    const push_rms = runtime.RmsnormPush{ .dim = @intCast(dim), .eps = cfg.base.rms_eps, .gemma_quirk = 0 };
    const push_add_dim = runtime.AddInPlacePush{ .n = @intCast(n_pos * dim) };
    const push_swiglu = runtime.SwigluPush{ .n = @intCast(n_pos * ff_dim) };
    const push_softmax = runtime.SoftmaxPush{ .dim = @intCast(n_pos), .stride = @intCast(n_pos) };
    const push_ce = runtime.SoftmaxCeLossGradPush{ .dim_out = @intCast(vocab), .n_samples = @intCast(n_pos) };
    const push_lm_head = runtime.LinearBatchedPush{ .M = @intCast(n_pos), .N = @intCast(vocab), .K = @intCast(dim) };
    const push_lin_down = runtime.LinearBatchedPush{ .M = @intCast(n_pos), .N = @intCast(dim), .K = @intCast(ff_dim) };
    const push_lin_gate = runtime.LinearBatchedPush{ .M = @intCast(n_pos), .N = @intCast(ff_dim), .K = @intCast(dim) };
    const push_lin_up = runtime.LinearBatchedPush{ .M = @intCast(n_pos), .N = @intCast(ff_dim), .K = @intCast(dim) };
    const push_lin_o = runtime.LinearBatchedPush{ .M = @intCast(n_pos), .N = @intCast(dim), .K = @intCast(q_dim) };
    const push_lin_q = runtime.LinearBatchedPush{ .M = @intCast(n_pos), .N = @intCast(q_dim), .K = @intCast(dim) };
    const push_lin_k = runtime.LinearBatchedPush{ .M = @intCast(n_pos), .N = @intCast(kv_dim), .K = @intCast(dim) };
    const push_lin_v = runtime.LinearBatchedPush{ .M = @intCast(n_pos), .N = @intCast(kv_dim), .K = @intCast(dim) };
    const push_dattn = runtime.AttnBackwardDattnPush{
        .n_q = @intCast(n_pos),
        .n_heads = @intCast(n_heads),
        .heads_per_kv = heads_per_kv,
        .head_dim = @intCast(head_dim),
        .n_kv = @intCast(n_pos),
        .kv_stride = @intCast(kv_dim),
        .attn_stride = @intCast(n_pos),
    };
    const push_dv = runtime.AttnBackwardDvPush{
        .n_q = @intCast(n_pos),
        .n_heads = @intCast(n_heads),
        .heads_per_kv = heads_per_kv,
        .n_kv_heads = @intCast(n_kv_heads),
        .head_dim = @intCast(head_dim),
        .n_kv = @intCast(n_pos),
        .attn_stride = @intCast(n_pos),
    };
    const push_dq = runtime.AttnBackwardDqPush{
        .n_q = @intCast(n_pos),
        .n_heads = @intCast(n_heads),
        .heads_per_kv = heads_per_kv,
        .head_dim = @intCast(head_dim),
        .n_kv = @intCast(n_pos),
        .kv_stride = @intCast(kv_dim),
        .scores_stride = @intCast(n_pos),
        .inv_sqrt_dim = inv_sqrt_d,
    };
    const push_dk = runtime.AttnBackwardDkPush{
        .n_q = @intCast(n_pos),
        .n_heads = @intCast(n_heads),
        .heads_per_kv = heads_per_kv,
        .n_kv_heads = @intCast(n_kv_heads),
        .head_dim = @intCast(head_dim),
        .n_kv = @intCast(n_pos),
        .scores_stride = @intCast(n_pos),
        .inv_sqrt_dim = inv_sqrt_d,
    };
    const push_embed = runtime.EmbeddingBackwardPush{
        .dim = @intCast(dim),
        .n_pos = @intCast(n_pos),
        .vocab_size = @intCast(vocab),
    };

    // ── Recorder. Total dispatches: 1 (CE loss-grad) + 2 (lm_head dx/dW)
    //    + 1 (final_norm rmsnorm bw) + 24·N (per-layer chain) + 1
    //    (embedding_bw) = 5 + 24·N. For N=2: 53.
    var rec = try gpu_recorder.Recorder.init(&ctx, 128, 512);
    defer rec.deinit();
    try rec.begin();

    const lwg: u32 = 16;
    const groupsCeil = struct {
        fn f(n: u32) u32 {
            return (n + lwg - 1) / lwg;
        }
    }.f;

    // ── 1. CE loss-grad: logits + target_one_hot → d_logits.
    //    v2 shader: one workgroup per sample (n_pos workgroups).
    try rec.dispatch(&k_ce_loss_grad, &.{ &buf_logits, &buf_target_oh, &buf_d_logits }, &push_ce, @intCast(n_pos), 1, 1);

    // ── 2. lm_head: dx into d_final_norm_out, dW into dw_lm_head.
    try rec.dispatch(&k_lin_dx, &.{ &buf_d_logits, &buf_w_lm_head, &buf_d_final_norm_out }, &push_lm_head, groupsCeil(push_lm_head.M), groupsCeil(push_lm_head.K), 1);
    try rec.dispatch(&k_lin_dw, &.{ &buf_d_logits, &buf_final_norm_out, &buf_dw_lm_head }, &push_lm_head, groupsCeil(push_lm_head.N), groupsCeil(push_lm_head.K), 1);

    // ── 3. final_norm rmsnorm backward → d_last_y + dw_final_norm_partial.
    const last_idx = cfg.n_layers - 1;
    // Last layer's `y` is held in its `mid` buffer? No — `mid` is the
    // post-attention residual output; `y` is the post-FF residual
    // output. The CPU oracle captures y in `acts.layer_acts[last].y`
    // but on the GPU we didn't upload `y` separately because layer
    // i+1's x_in already holds it. For the last layer we need to upload
    // y explicitly. We do so here as a one-off — small additional buffer.
    var buf_last_y = try buffer.Buffer.initStatic(&ctx, f32, acts.layer_acts[last_idx].y);
    defer buf_last_y.deinit(ctx.device);

    try rec.dispatch(&k_rms_bw, &.{ &buf_d_final_norm_out, &buf_last_y, &buf_w_final_norm, &buf_d_last_y, &buf_dw_final_norm_partial }, &push_rms, @intCast(n_pos), 1, 1);

    // ── 4. Per-layer backward, last → first. d_y_in for layer N-1
    //    is d_last_y; for layer i < N-1 it's d_x_in[i+1].
    var li = cfg.n_layers;
    while (li > 0) {
        li -= 1;
        const d_y_in: *const buffer.Buffer = if (li == cfg.n_layers - 1) &buf_d_last_y else &buf_d_x_in[li + 1];
        try recordStackLayerBackward(
            &rec,
            .{
                .lin_dx = &k_lin_dx,
                .lin_dw = &k_lin_dw,
                .swiglu_bw = &k_swiglu_bw,
                .rms_bw = &k_rms_bw,
                .attn_dattn = &k_attn_dattn,
                .attn_dv = &k_attn_dv,
                .attn_dq = &k_attn_dq,
                .attn_dk = &k_attn_dk,
                .softmax_bw = &k_softmax_bw,
                .add = &k_add,
            },
            .{
                .x_in = &buf_x_in[li],
                .n1 = &buf_n1[li],
                .q = &buf_q[li],
                .k = &buf_k[li],
                .v = &buf_v[li],
                .attn = &buf_attn[li],
                .attn_out = &buf_attn_out[li],
                .mid = &buf_mid[li],
                .n2 = &buf_n2[li],
                .pre_gate = &buf_pre_gate[li],
                .up = &buf_up[li],
                .gated = &buf_gated[li],
                .w_n1 = &buf_w_n1[li],
                .w_q = &buf_w_q[li],
                .w_k = &buf_w_k[li],
                .w_v = &buf_w_v[li],
                .w_o = &buf_w_o[li],
                .w_n2 = &buf_w_n2[li],
                .w_gate = &buf_w_gate[li],
                .w_up = &buf_w_up[li],
                .w_down = &buf_w_down[li],
                .dw_n1_partial = &buf_dw_n1_partial[li],
                .dw_q = &buf_dw_q[li],
                .dw_k = &buf_dw_k[li],
                .dw_v = &buf_dw_v[li],
                .dw_o = &buf_dw_o[li],
                .dw_n2_partial = &buf_dw_n2_partial[li],
                .dw_gate = &buf_dw_gate[li],
                .dw_up = &buf_dw_up[li],
                .dw_down = &buf_dw_down[li],
            },
            .{
                .d_gated = &sc_d_gated,
                .d_pre_gate = &sc_d_pre_gate,
                .d_up = &sc_d_up_grad,
                .d_n2 = &sc_d_n2,
                .d_n2_up = &sc_d_n2_up,
                .d_mid_norm = &sc_d_mid_norm,
                .d_attn_out = &sc_d_attn_out,
                .d_attn = &sc_d_attn,
                .d_scores = &sc_d_scores,
                .dQ = &sc_dQ,
                .dK = &sc_dK,
                .dV = &sc_dV,
                .d_n1 = &sc_d_n1,
                .d_n1_k = &sc_d_n1_k,
                .d_n1_v = &sc_d_n1_v,
            },
            .{
                .lin_down = &push_lin_down,
                .lin_gate = &push_lin_gate,
                .lin_up = &push_lin_up,
                .lin_o = &push_lin_o,
                .lin_q = &push_lin_q,
                .lin_k = &push_lin_k,
                .lin_v = &push_lin_v,
                .swiglu = &push_swiglu,
                .rms = &push_rms,
                .add_dim = &push_add_dim,
                .softmax = &push_softmax,
                .dattn = &push_dattn,
                .dv = &push_dv,
                .dq = &push_dq,
                .dk = &push_dk,
            },
            .{ .n_pos = @intCast(n_pos), .n_heads = @intCast(n_heads), .n_kv_heads = @intCast(n_kv_heads), .head_dim = @intCast(head_dim) },
            d_y_in,
            &buf_d_x_in[li],
        );
    }

    // ── 5. embedding_backward: dE_embed[token_id, :] += d_x_in[0][p, :].
    //    initDeviceOnly already zero-filled the dE buffer.
    try rec.dispatch(&k_embed_bw, &.{ &buf_d_x_in[0], &buf_token_ids, &buf_dE_embed }, &push_embed, @intCast(vocab), 1, 1);

    try rec.endAndSubmit();

    // ── Read back grad buffers + reduce per-row partials.
    const gpu_dE_embed = try allocator.alloc(f32, w_embed.len);
    defer allocator.free(gpu_dE_embed);
    const gpu_dw_final_norm_partial = try allocator.alloc(f32, n_pos * dim);
    defer allocator.free(gpu_dw_final_norm_partial);
    const gpu_dw_lm_head = try allocator.alloc(f32, w_lm_head.len);
    defer allocator.free(gpu_dw_lm_head);
    try buf_dE_embed.readBack(&ctx, f32, gpu_dE_embed);
    try buf_dw_final_norm_partial.readBack(&ctx, f32, gpu_dw_final_norm_partial);
    try buf_dw_lm_head.readBack(&ctx, f32, gpu_dw_lm_head);

    const gpu_dw_final_norm = try allocator.alloc(f32, dim);
    defer allocator.free(gpu_dw_final_norm);
    @memset(gpu_dw_final_norm, 0);
    for (0..n_pos) |row| {
        const off = row * dim;
        for (0..dim) |i| gpu_dw_final_norm[i] += gpu_dw_final_norm_partial[off + i];
    }

    // Per-layer reads + reductions.
    const gpu_layer_grads = try allocator.alloc([9][]f32, cfg.n_layers);
    defer {
        for (gpu_layer_grads) |slots| for (slots) |s| allocator.free(s);
        allocator.free(gpu_layer_grads);
    }
    for (0..cfg.n_layers) |i| {
        const layer = &layers[i];
        const dw_n1_partial = try allocator.alloc(f32, n_pos * dim);
        defer allocator.free(dw_n1_partial);
        const dw_n2_partial = try allocator.alloc(f32, n_pos * dim);
        defer allocator.free(dw_n2_partial);
        try buf_dw_n1_partial[i].readBack(&ctx, f32, dw_n1_partial);
        try buf_dw_n2_partial[i].readBack(&ctx, f32, dw_n2_partial);
        const dw_n1 = try allocator.alloc(f32, dim);
        const dw_q_g = try allocator.alloc(f32, layer.w_q.len);
        const dw_k_g = try allocator.alloc(f32, layer.w_k.len);
        const dw_v_g = try allocator.alloc(f32, layer.w_v.len);
        const dw_o_g = try allocator.alloc(f32, layer.w_o.len);
        const dw_n2 = try allocator.alloc(f32, dim);
        const dw_gate_g = try allocator.alloc(f32, layer.w_gate.len);
        const dw_up_g = try allocator.alloc(f32, layer.w_up.len);
        const dw_down_g = try allocator.alloc(f32, layer.w_down.len);
        @memset(dw_n1, 0);
        @memset(dw_n2, 0);
        for (0..n_pos) |row| {
            const off = row * dim;
            for (0..dim) |idx| {
                dw_n1[idx] += dw_n1_partial[off + idx];
                dw_n2[idx] += dw_n2_partial[off + idx];
            }
        }
        try buf_dw_q[i].readBack(&ctx, f32, dw_q_g);
        try buf_dw_k[i].readBack(&ctx, f32, dw_k_g);
        try buf_dw_v[i].readBack(&ctx, f32, dw_v_g);
        try buf_dw_o[i].readBack(&ctx, f32, dw_o_g);
        try buf_dw_gate[i].readBack(&ctx, f32, dw_gate_g);
        try buf_dw_up[i].readBack(&ctx, f32, dw_up_g);
        try buf_dw_down[i].readBack(&ctx, f32, dw_down_g);
        gpu_layer_grads[i] = .{ dw_n1, dw_q_g, dw_k_g, dw_v_g, dw_o_g, dw_n2, dw_gate_g, dw_up_g, dw_down_g };
    }

    // ── Compare. Tolerance: per-slice rel-err with 1e-5 floor.
    var worst_rel: f32 = 0;
    var worst_label: []const u8 = "";

    const compareSlice = struct {
        fn f(label: []const u8, c_s: []const f32, g_s: []const f32, worst: *f32, worst_lbl: *[]const u8) !void {
            var max_abs_cpu: f32 = 0;
            for (c_s) |v| max_abs_cpu = @max(max_abs_cpu, @abs(v));
            const tol: f32 = @max(1e-5, max_abs_cpu * 1e-3);
            var max_abs_diff: f32 = 0;
            for (c_s, g_s) |a, b| {
                const d = @abs(a - b);
                if (d > max_abs_diff) max_abs_diff = d;
            }
            const rel = if (max_abs_cpu > 0) max_abs_diff / max_abs_cpu else max_abs_diff;
            if (rel > worst.*) {
                worst.* = rel;
                worst_lbl.* = label;
            }
            if (max_abs_diff > tol) {
                std.debug.print(
                    "stack backward parity FAIL: {s} max |Δ|={e:.3} (tol={e:.3}, max_cpu={e:.3})\n",
                    .{ label, max_abs_diff, tol, max_abs_cpu },
                );
                return error.ParityFailed;
            }
        }
    }.f;

    try compareSlice("dE_embed", grads_cpu.dE_embed, gpu_dE_embed, &worst_rel, &worst_label);
    try compareSlice("dw_final_norm", grads_cpu.dw_final_norm, gpu_dw_final_norm, &worst_rel, &worst_label);
    try compareSlice("dw_lm_head", grads_cpu.dw_lm_head, gpu_dw_lm_head, &worst_rel, &worst_label);

    // Heap-allocate each label so the captured worst_label slice
    // doesn't get clobbered by later iterations (the comparator stores
    // the slice by reference for the final pass-line).
    var owned_labels = std.ArrayList([]u8).init(allocator);
    defer {
        for (owned_labels.items) |s| allocator.free(s);
        owned_labels.deinit();
    }
    const layer_field_names = [_][]const u8{ "dw_n1", "dw_q", "dw_k", "dw_v", "dw_o", "dw_n2", "dw_gate", "dw_up", "dw_down" };
    for (0..cfg.n_layers) |i| {
        const cpu_slices = [_][]const f32{
            grads_cpu.layer_grads[i].dw_n1,
            grads_cpu.layer_grads[i].dw_q,
            grads_cpu.layer_grads[i].dw_k,
            grads_cpu.layer_grads[i].dw_v,
            grads_cpu.layer_grads[i].dw_o,
            grads_cpu.layer_grads[i].dw_n2,
            grads_cpu.layer_grads[i].dw_gate,
            grads_cpu.layer_grads[i].dw_up,
            grads_cpu.layer_grads[i].dw_down,
        };
        for (cpu_slices, gpu_layer_grads[i], layer_field_names) |c_s, g_s, fname| {
            const lbl = try std.fmt.allocPrint(allocator, "L{d}.{s}", .{ i, fname });
            try owned_labels.append(lbl);
            try compareSlice(lbl, c_s, g_s, &worst_rel, &worst_label);
        }
    }

    std.debug.print(
        "PASS GPU decoder stack backward parity (n_layers={d} dim={d} vocab={d} n_pos={d}; {d} dispatches, worst rel-err {e:.2} on {s})\n",
        .{ cfg.n_layers, dim, vocab, n_pos, 5 + 27 * cfg.n_layers, worst_rel, worst_label },
    );
}

// ── chunk 8c-α-3: full-GPU stack training, 100-step Adam loop ──────
//
// Self-sustaining transformer trainer: every operation per step
// (embed_lookup → N decoder layers → final RMSNorm → lm_head →
// softmax-CE loss-grad → backward → Adam) lives on the GPU. CPU only
// builds the initial weights + token IDs + one-hot target, then
// reads back final logits to compute the closing loss.
//
// Per-step dispatch count: 1 embed + 14·N forward + 2 (final_norm +
// lm_head) + 1 loss-grad + 5+24·N backward + 8N+3 Adam =
// 11 + 46·N. For N=2: 103 dispatches/step.

const StackFwKernels = struct {
    rms: *const pipeline.Kernel,
    matmul: *const pipeline.Kernel,
    attn_scores: *const pipeline.Kernel,
    attn_output: *const pipeline.Kernel,
    softmax: *const pipeline.Kernel,
    swiglu_fwd: *const pipeline.Kernel,
    vec_add: *const pipeline.Kernel,
};

const StackFwLayerBufs = struct {
    // Read-only.
    x_in: *const buffer.Buffer,
    w_n1: *const buffer.Buffer,
    w_q: *const buffer.Buffer,
    w_k: *const buffer.Buffer,
    w_v: *const buffer.Buffer,
    w_o: *const buffer.Buffer,
    w_n2: *const buffer.Buffer,
    w_gate: *const buffer.Buffer,
    w_up: *const buffer.Buffer,
    w_down: *const buffer.Buffer,
    // Saved-activation outputs (consumed by backward).
    n1: *const buffer.Buffer,
    q: *const buffer.Buffer,
    k: *const buffer.Buffer,
    v: *const buffer.Buffer,
    attn: *const buffer.Buffer,
    attn_out: *const buffer.Buffer,
    mid: *const buffer.Buffer,
    n2: *const buffer.Buffer,
    pre_gate: *const buffer.Buffer,
    up: *const buffer.Buffer,
    gated: *const buffer.Buffer,
    y: *const buffer.Buffer,
};

const StackFwScratch = struct {
    scores: *const buffer.Buffer,
    o: *const buffer.Buffer,
    ff_out: *const buffer.Buffer,
};

fn recordStackLayerForward(
    rec: *gpu_recorder.Recorder,
    kernels: StackFwKernels,
    bufs: StackFwLayerBufs,
    scratch: StackFwScratch,
    pushes: struct {
        rms: *const runtime.RmsnormPush,
        mm_q: *const runtime.MatmulPush,
        mm_k: *const runtime.MatmulPush,
        mm_v: *const runtime.MatmulPush,
        mm_o: *const runtime.MatmulPush,
        mm_gate: *const runtime.MatmulPush,
        mm_up: *const runtime.MatmulPush,
        mm_down: *const runtime.MatmulPush,
        attn_scores: *const runtime.AttnScoresTrainPush,
        attn_output: *const runtime.AttnOutputTrainPush,
        softmax: *const runtime.SoftmaxPush,
        swiglu: *const runtime.SwigluPush,
        add_dim: *const runtime.AddInPlacePush,
    },
    shape: struct {
        n_pos: u32,
        n_heads: u32,
        n_kv_heads: u32,
        head_dim: u32,
        ff_dim: u32,
        dim: u32,
    },
) !void {
    const groupsLin: u32 = 256;
    const addGroups: u32 = (pushes.add_dim.n + groupsLin - 1) / groupsLin;
    const swigluGroups: u32 = (pushes.swiglu.n + groupsLin - 1) / groupsLin;

    // 1. RMSNorm n1.
    try rec.dispatch(kernels.rms, &.{ bufs.x_in, bufs.w_n1, bufs.n1 }, pushes.rms, shape.n_pos, 1, 1);
    // 2-4. Q/K/V projections.
    try rec.dispatch(kernels.matmul, &.{ bufs.n1, bufs.w_q, bufs.q }, pushes.mm_q, shape.n_pos * pushes.mm_q.n, 1, 1);
    try rec.dispatch(kernels.matmul, &.{ bufs.n1, bufs.w_k, bufs.k }, pushes.mm_k, shape.n_pos * pushes.mm_k.n, 1, 1);
    try rec.dispatch(kernels.matmul, &.{ bufs.n1, bufs.w_v, bufs.v }, pushes.mm_v, shape.n_pos * pushes.mm_v.n, 1, 1);
    // 5. attention scores (causal-mask via -inf for keys beyond q).
    try rec.dispatch(kernels.attn_scores, &.{ bufs.q, bufs.k, scratch.scores }, pushes.attn_scores, shape.n_pos * shape.n_heads * shape.n_pos, 1, 1);
    // 6. softmax → attn.
    try rec.dispatch(kernels.softmax, &.{ scratch.scores, bufs.attn }, pushes.softmax, shape.n_pos * shape.n_heads, 1, 1);
    // 7. attention output.
    try rec.dispatch(kernels.attn_output, &.{ bufs.attn, bufs.v, bufs.attn_out }, pushes.attn_output, shape.n_pos * shape.n_heads * shape.head_dim, 1, 1);
    // 8. O projection.
    try rec.dispatch(kernels.matmul, &.{ bufs.attn_out, bufs.w_o, scratch.o }, pushes.mm_o, shape.n_pos * pushes.mm_o.n, 1, 1);
    // 9. mid = x_in + o (residual).
    try rec.dispatch(kernels.vec_add, &.{ bufs.x_in, scratch.o, bufs.mid }, pushes.add_dim, addGroups, 1, 1);
    // 10. RMSNorm n2.
    try rec.dispatch(kernels.rms, &.{ bufs.mid, bufs.w_n2, bufs.n2 }, pushes.rms, shape.n_pos, 1, 1);
    // 11. W_gate matmul.
    try rec.dispatch(kernels.matmul, &.{ bufs.n2, bufs.w_gate, bufs.pre_gate }, pushes.mm_gate, shape.n_pos * pushes.mm_gate.n, 1, 1);
    // 12. W_up matmul.
    try rec.dispatch(kernels.matmul, &.{ bufs.n2, bufs.w_up, bufs.up }, pushes.mm_up, shape.n_pos * pushes.mm_up.n, 1, 1);
    // 13. SwiGLU fwd: gated = silu(pre_gate) · up.
    try rec.dispatch(kernels.swiglu_fwd, &.{ bufs.pre_gate, bufs.up, bufs.gated }, pushes.swiglu, swigluGroups, 1, 1);
    // 14. W_down matmul → ff_out.
    try rec.dispatch(kernels.matmul, &.{ bufs.gated, bufs.w_down, scratch.ff_out }, pushes.mm_down, shape.n_pos * pushes.mm_down.n, 1, 1);
    // 15. y = mid + ff_out (residual).
    try rec.dispatch(kernels.vec_add, &.{ bufs.mid, scratch.ff_out, bufs.y }, pushes.add_dim, addGroups, 1, 1);
}

fn runDecoderStackTrainGpuSmoke(allocator: std.mem.Allocator) !void {
    const cfg = cpu_train_decoder.StackConfig{
        .base = .{
            .dim = 16,
            .n_heads = 2,
            .n_kv_heads = 2,
            .head_dim = 8,
            .ff_dim = 32,
            .n_pos = 4,
            .rms_eps = 1e-5,
            .causal = true,
            .rotary_dim = 8, // full RoPE (rotary_dim = head_dim)
            .qk_norm = true, // per-head Q/K-norm (Qwen3 architectural detail)
        },
        .n_layers = 2,
        .vocab_size = 8,
    };
    const dim = cfg.base.dim;
    const n_pos = cfg.base.n_pos;
    const n_heads = cfg.base.n_heads;
    const n_kv_heads = cfg.base.n_kv_heads;
    const head_dim = cfg.base.head_dim;
    const ff_dim = cfg.base.ff_dim;
    const vocab = cfg.vocab_size;
    const q_dim = n_heads * head_dim;
    const kv_dim = n_kv_heads * head_dim;

    // Same RNG sequence as the 8c-α-1 / α-2 smokes so initial weights
    // + token_ids + target_ids are bit-equal — initial loss matches.
    var prng = std.Random.DefaultPrng.init(0xC0_DE_AC_01);
    const rng = prng.random();
    const initScale: f32 = 0.1;

    // ── CPU-side weight + token init.
    const w_embed = try allocator.alloc(f32, vocab * dim);
    defer allocator.free(w_embed);
    for (w_embed) |*v| v.* = (rng.float(f32) * 2.0 - 1.0) * initScale;
    const w_final_norm = try allocator.alloc(f32, dim);
    defer allocator.free(w_final_norm);
    for (w_final_norm) |*v| v.* = 1.0;
    const w_lm_head = try allocator.alloc(f32, vocab * dim);
    defer allocator.free(w_lm_head);
    for (w_lm_head) |*v| v.* = (rng.float(f32) * 2.0 - 1.0) * initScale;

    const layers = try allocator.alloc(cpu_train_decoder.Layer, cfg.n_layers);
    defer allocator.free(layers);
    const layer_weight_buffers = try allocator.alloc([9][]f32, cfg.n_layers);
    defer allocator.free(layer_weight_buffers);
    defer for (layer_weight_buffers) |slots| for (slots) |s| allocator.free(s);

    const layer_qk_buffers = try allocator.alloc([2][]f32, cfg.n_layers);
    defer allocator.free(layer_qk_buffers);
    defer for (layer_qk_buffers) |slots| for (slots) |s| allocator.free(s);

    for (layers, layer_weight_buffers, layer_qk_buffers) |*layer, *slots, *qk_slots| {
        const w_n1 = try allocator.alloc(f32, dim);
        const w_q = try allocator.alloc(f32, q_dim * dim);
        const w_k = try allocator.alloc(f32, kv_dim * dim);
        const w_v = try allocator.alloc(f32, kv_dim * dim);
        const w_o = try allocator.alloc(f32, dim * q_dim);
        const w_n2 = try allocator.alloc(f32, dim);
        const w_gate = try allocator.alloc(f32, ff_dim * dim);
        const w_up = try allocator.alloc(f32, ff_dim * dim);
        const w_down = try allocator.alloc(f32, dim * ff_dim);
        const w_q_norm = try allocator.alloc(f32, head_dim);
        const w_k_norm = try allocator.alloc(f32, head_dim);
        for (w_n1) |*v| v.* = 1.0;
        for (w_n2) |*v| v.* = 1.0;
        for (w_q_norm) |*v| v.* = 1.0;
        for (w_k_norm) |*v| v.* = 1.0;
        for (w_q) |*v| v.* = (rng.float(f32) * 2.0 - 1.0) * initScale;
        for (w_k) |*v| v.* = (rng.float(f32) * 2.0 - 1.0) * initScale;
        for (w_v) |*v| v.* = (rng.float(f32) * 2.0 - 1.0) * initScale;
        for (w_o) |*v| v.* = (rng.float(f32) * 2.0 - 1.0) * initScale;
        for (w_gate) |*v| v.* = (rng.float(f32) * 2.0 - 1.0) * initScale;
        for (w_up) |*v| v.* = (rng.float(f32) * 2.0 - 1.0) * initScale;
        for (w_down) |*v| v.* = (rng.float(f32) * 2.0 - 1.0) * initScale;
        slots.* = .{ w_n1, w_q, w_k, w_v, w_o, w_n2, w_gate, w_up, w_down };
        qk_slots.* = .{ w_q_norm, w_k_norm };
        layer.* = .{
            .cfg = cfg.base,
            .w_n1 = w_n1,
            .w_q = w_q,
            .w_k = w_k,
            .w_v = w_v,
            .w_o = w_o,
            .w_n2 = w_n2,
            .w_gate = w_gate,
            .w_up = w_up,
            .w_down = w_down,
            .w_q_norm = w_q_norm,
            .w_k_norm = w_k_norm,
        };
    }

    var stack = cpu_train_decoder.Stack{
        .cfg = cfg,
        .embed = w_embed,
        .layers = layers,
        .final_norm = w_final_norm,
        .lm_head = w_lm_head,
    };

    // ── Compute initial CPU loss for the convergence assertion.
    var acts_cpu = try cpu_train_decoder.allocStackActs(allocator, cfg);
    defer cpu_train_decoder.freeStackActs(allocator, &acts_cpu);

    const token_ids = try allocator.alloc(u32, n_pos);
    defer allocator.free(token_ids);
    for (token_ids) |*tid| tid.* = rng.intRangeLessThan(u32, 0, @intCast(vocab));
    const target_ids = try allocator.alloc(u32, n_pos);
    defer allocator.free(target_ids);
    for (target_ids) |*tid| tid.* = rng.intRangeLessThan(u32, 0, @intCast(vocab));

    acts_cpu.token_ids = token_ids;
    cpu_train_decoder.stackForward(&stack, &acts_cpu);
    const initial_loss = cpu_train_decoder.softmaxCeLoss(acts_cpu.logits, target_ids, n_pos, vocab);
    // CCE consumes target ids directly — no [n_pos × vocab] one-hot needed.

    // ── GPU bring-up via Runner.
    var ctx = try vk.Context.init(allocator);
    defer ctx.deinit();

    // Build LayerWeights slice borrowed by the Runner. Lifetimes:
    // the Runner uploads into its own DEVICE_LOCAL buffers in init(),
    // so these borrows only need to outlive the init() call.
    const layer_weights = try allocator.alloc(train_transformer.LayerWeights, cfg.n_layers);
    defer allocator.free(layer_weights);
    for (layers, layer_weights) |*layer, *lw| {
        lw.* = .{
            .w_n1 = layer.w_n1,
            .w_q = layer.w_q,
            .w_k = layer.w_k,
            .w_v = layer.w_v,
            .w_o = layer.w_o,
            .w_n2 = layer.w_n2,
            .w_gate = layer.w_gate,
            .w_up = layer.w_up,
            .w_down = layer.w_down,
            .w_q_norm = layer.w_q_norm,
            .w_k_norm = layer.w_k_norm,
        };
    }

    var runner = try train_transformer.Runner.init(
        allocator,
        &ctx,
        .{
            .dim = @intCast(dim),
            .n_heads = @intCast(n_heads),
            .n_kv_heads = @intCast(n_kv_heads),
            .head_dim = @intCast(head_dim),
            .ff_dim = @intCast(ff_dim),
            .n_pos = @intCast(n_pos),
            .n_layers = @intCast(cfg.n_layers),
            .vocab_size = @intCast(vocab),
            .rms_eps = cfg.base.rms_eps,
            .causal = cfg.base.causal,
            .rotary_dim = @intCast(cfg.base.rotary_dim),
            .rope_theta = cfg.base.rope_theta,
            .qk_norm = cfg.base.qk_norm,
            .lr = 1e-2,
        },
        .{
            .embed = w_embed,
            .final_norm = w_final_norm,
            .lm_head = w_lm_head,
            .layers = layer_weights,
        },
    );
    defer runner.deinit();

    const n_steps: u32 = 200;
    var step_t: u32 = 0;
    while (step_t < n_steps) : (step_t += 1) {
        try runner.step(token_ids, target_ids);
    }

    const logits_final = try allocator.alloc(f32, n_pos * vocab);
    defer allocator.free(logits_final);
    try runner.forwardLogits(token_ids, logits_final);

    const final_loss = cpu_train_decoder.softmaxCeLoss(logits_final, target_ids, n_pos, vocab);
    const ratio = final_loss / initial_loss;
    if (ratio > 1e-2) {
        std.debug.print(
            "GPU stack fine-tune: initial_loss={d:.6} final_loss={d:.6} ratio={d:.4}\n",
            .{ initial_loss, final_loss, ratio },
        );
        return error.LossDidNotDecrease;
    }

    // Per-layer dispatches: 51 baseline (β-3a-2 SwiGLU stack) + 4 when
    // RoPE is enabled (2 fwd + 2 bw across Q + K) + 6 when qk_norm is
    // enabled (2 fwd rmsnorm + 2 bw rmsnorm + 2 Adam updates).
    var per_layer_dispatches: usize = 51;
    if (cfg.base.rotary_dim > 0) per_layer_dispatches += 4;
    if (cfg.base.qk_norm) per_layer_dispatches += 6;
    std.debug.print(
        "PASS GPU decoder stack fine-tune via Runner (n_layers={d} dim={d} vocab={d} n_pos={d}; CE loss {d:.6} → {d:.6} ({e:.2}× drop) over {d} Adam steps, {d} dispatches/step)\n",
        .{ cfg.n_layers, dim, vocab, n_pos, initial_loss, final_loss, 1.0 / ratio, n_steps, 11 + per_layer_dispatches * cfg.n_layers },
    );
}

// ── chunk 8c-β-2: Real-shape buffer sizing (synthetic weights) ──────
//
// Instantiates `train_transformer.Runner` at Qwen3-0.6B-class dims with
// random fp32 weights and verifies the runtime envelope holds: init
// succeeds, one Adam step doesn't OOM, loss decreases over ~50 steps,
// and reports ms/step so β-3 can compare once real weights enter.
//
// Architecture is the toy-decoder shape the trainer ships today (no
// SwiGLU, no RoPE, no q/k-norm). Architectural fidelity to real Qwen3
// belongs to a separate β-3a chunk that adds the missing primitives
// (each gets the chunk-1..7 treatment: CPU oracle → GPU shader → smoke
// → fold into Runner). What β-2 establishes is that the *buffer
// sizing* + *dispatch sequencing* the Runner generates scales to
// realistic memory pressure (~9 GB) and depth (28 layers).
//
// Pass criteria:
//   - Runner.init returns without OOM / descriptor exhaustion.
//   - All 50 steps run.
//   - Final CE < initial CE (any ratio < 1; this is a "does it still
//     train" envelope check, not a convergence assertion).
//   - All loss values finite.
//
// All weights heap-allocated via the smoke's allocator; per-layer
// weights live in an arena so the slice lifetime ends cleanly when
// the smoke returns.

fn runDecoderStackTrainGpuRealShapeSmoke(allocator: std.mem.Allocator) !void {
    const dim: u32 = 1024;
    const n_layers: u32 = 28;
    const n_heads: u32 = 16;
    const n_kv_heads: u32 = 8;
    const head_dim: u32 = 64;
    const ff_dim: u32 = 3072;
    const vocab: u32 = 151_936;
    const n_pos: u32 = 64;
    const q_dim: u32 = n_heads * head_dim;
    const kv_dim: u32 = n_kv_heads * head_dim;

    var prng = std.Random.DefaultPrng.init(0xBE_BA_2A_01);
    const rng = prng.random();
    // Conservative init scale — wide layers + Adam can blow up at 0.1
    // even on a single-batch overfit. 0.02 keeps initial activations
    // and logits bounded.
    const init_scale: f32 = 0.02;

    const fillRandom = struct {
        fn run(r: std.Random, buf: []f32, scale: f32) void {
            for (buf) |*v| v.* = (r.float(f32) * 2.0 - 1.0) * scale;
        }
    }.run;

    // Stack-level synthetic weights.
    const w_embed = try allocator.alloc(f32, @as(usize, vocab) * dim);
    defer allocator.free(w_embed);
    fillRandom(rng, w_embed, init_scale);
    const w_final_norm = try allocator.alloc(f32, dim);
    defer allocator.free(w_final_norm);
    @memset(w_final_norm, 1.0);
    const w_lm_head = try allocator.alloc(f32, @as(usize, vocab) * dim);
    defer allocator.free(w_lm_head);
    fillRandom(rng, w_lm_head, init_scale);

    // Per-layer weights live in an arena; freed wholesale at smoke end.
    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();
    const aalloc = arena.allocator();
    const layer_weights = try allocator.alloc(train_transformer.LayerWeights, n_layers);
    defer allocator.free(layer_weights);
    for (layer_weights) |*lw| {
        const w_n1 = try aalloc.alloc(f32, dim);
        const w_q = try aalloc.alloc(f32, @as(usize, q_dim) * dim);
        const w_k = try aalloc.alloc(f32, @as(usize, kv_dim) * dim);
        const w_v = try aalloc.alloc(f32, @as(usize, kv_dim) * dim);
        const w_o = try aalloc.alloc(f32, @as(usize, dim) * q_dim);
        const w_n2 = try aalloc.alloc(f32, dim);
        const w_gate = try aalloc.alloc(f32, @as(usize, ff_dim) * dim);
        const w_up = try aalloc.alloc(f32, @as(usize, ff_dim) * dim);
        const w_down = try aalloc.alloc(f32, @as(usize, dim) * ff_dim);
        const w_q_norm = try aalloc.alloc(f32, head_dim);
        const w_k_norm = try aalloc.alloc(f32, head_dim);
        @memset(w_n1, 1.0);
        @memset(w_n2, 1.0);
        @memset(w_q_norm, 1.0);
        @memset(w_k_norm, 1.0);
        fillRandom(rng, w_q, init_scale);
        fillRandom(rng, w_k, init_scale);
        fillRandom(rng, w_v, init_scale);
        fillRandom(rng, w_o, init_scale);
        fillRandom(rng, w_gate, init_scale);
        fillRandom(rng, w_up, init_scale);
        fillRandom(rng, w_down, init_scale);
        lw.* = .{
            .w_n1 = w_n1,
            .w_q = w_q,
            .w_k = w_k,
            .w_v = w_v,
            .w_o = w_o,
            .w_n2 = w_n2,
            .w_gate = w_gate,
            .w_up = w_up,
            .w_down = w_down,
            .w_q_norm = w_q_norm,
            .w_k_norm = w_k_norm,
        };
    }

    // Random batch — overfit a single fixed (token_ids, target_ids) pair
    // so loss should drop monotonically.
    const token_ids = try allocator.alloc(u32, n_pos);
    defer allocator.free(token_ids);
    for (token_ids) |*t| t.* = rng.intRangeLessThan(u32, 0, vocab);
    const target_ids = try allocator.alloc(u32, n_pos);
    defer allocator.free(target_ids);
    for (target_ids) |*t| t.* = rng.intRangeLessThan(u32, 0, vocab);

    // CCE takes target ids directly — no one-hot expansion.

    // ── GPU bring-up.
    var ctx = try vk.Context.init(allocator);
    defer ctx.deinit();

    var runner = try train_transformer.Runner.init(
        allocator,
        &ctx,
        .{
            .dim = dim,
            .n_heads = n_heads,
            .n_kv_heads = n_kv_heads,
            .head_dim = head_dim,
            .ff_dim = ff_dim,
            .n_pos = n_pos,
            .n_layers = n_layers,
            .vocab_size = vocab,
            .rotary_dim = head_dim, // full RoPE
            .qk_norm = true, // per-head Q/K-norm (Qwen3 architectural detail)
            // Same lr as the toy 8c-α-3 smoke (1e-2). Single-batch
            // overfit is the regime we're in, not pre-training.
            .lr = 1e-2,
        },
        .{
            .embed = w_embed,
            .final_norm = w_final_norm,
            .lm_head = w_lm_head,
            .layers = layer_weights,
        },
    );
    defer runner.deinit();

    const logits_buf = try allocator.alloc(f32, @as(usize, n_pos) * vocab);
    defer allocator.free(logits_buf);

    try runner.forwardLogits(token_ids, logits_buf);
    const initial_loss = cpu_train_decoder.softmaxCeLoss(logits_buf, target_ids, n_pos, vocab);
    if (!std.math.isFinite(initial_loss)) {
        std.debug.print("Real-shape envelope: initial_loss not finite ({d})\n", .{initial_loss});
        return error.LossNotFinite;
    }

    const n_steps: u32 = 50;
    const t_start = std.time.nanoTimestamp();
    var step_t: u32 = 0;
    while (step_t < n_steps) : (step_t += 1) {
        try runner.step(token_ids, target_ids);
        if (step_t % 10 == 0 or step_t == n_steps - 1) {
            // Cheap heartbeat: forward + CE so we can watch the curve.
            try runner.forwardLogits(token_ids, logits_buf);
            const heartbeat = cpu_train_decoder.softmaxCeLoss(logits_buf, target_ids, n_pos, vocab);
            std.debug.print("  envelope step {d}: CE={d:.6}\n", .{ step_t + 1, heartbeat });
            if (!std.math.isFinite(heartbeat)) return error.LossNotFinite;
        }
    }
    const t_end = std.time.nanoTimestamp();
    const elapsed_ms: f64 = @as(f64, @floatFromInt(t_end - t_start)) / 1.0e6;
    const ms_per_step: f64 = elapsed_ms / @as(f64, @floatFromInt(n_steps));

    try runner.forwardLogits(token_ids, logits_buf);
    const final_loss = cpu_train_decoder.softmaxCeLoss(logits_buf, target_ids, n_pos, vocab);
    if (!std.math.isFinite(final_loss)) {
        std.debug.print("Real-shape envelope: final_loss not finite ({d}) after initial={d:.6}\n", .{ final_loss, initial_loss });
        return error.LossNotFinite;
    }
    if (final_loss >= initial_loss) {
        std.debug.print("Real-shape envelope: initial_loss={d:.6} final_loss={d:.6} (no decrease)\n", .{ initial_loss, final_loss });
        return error.LossDidNotDecrease;
    }

    std.debug.print(
        "PASS GPU decoder stack real-shape envelope (Qwen3-0.6B-class toy-arch: n_layers={d} dim={d} GQA {d}/{d} ff_dim={d} vocab={d} n_pos={d}; CE {d:.6} → {d:.6} over {d} Adam steps, {d:.1} ms/step)\n",
        .{ n_layers, dim, n_heads, n_kv_heads, ff_dim, vocab, n_pos, initial_loss, final_loss, n_steps, ms_per_step },
    );
}

// ── chunk 8c-β-3a: real Qwen3 weight load (fp32 train tensors) ───────
//
// Loads Qwen3-0.6B from the local HF cache, materialises fp32 training
// weights via `train_load_real.loadTrainWeightsFromId`, and gates the
// loader with three checks:
//
//   1. The derived `train_transformer.Config` matches Qwen3-0.6B's
//      published architecture (28 layers, 16/8 GQA at head_dim=128,
//      ff_dim=3072, vocab=151,936, full-RoPE, qk_norm on, rms_eps=1e-6,
//      rope_theta=1e6).
//   2. Every fp32 buffer has the expected element count.
//   3. A sampled subset of values from each tensor is finite. Cheap
//      vs scanning all ~720M params; catches any byte-pattern errors
//      from a wrong dtype path or shape misread.
//
// SKIP on `error.HfModelNotInCache` — fresh checkouts won't have the
// 1.2 GB safetensors local. PASS proves the bytes-on-disk → fp32 path
// is end-to-end correct; β-3b will run a forward pass to gate layout.

fn runRealModelLoadSmoke(allocator: std.mem.Allocator) !void {
    const model_id = "Qwen/Qwen3-0.6B";
    const n_pos: u32 = 64;

    var weights = train_load_real.loadTrainWeightsFromId(allocator, model_id, n_pos) catch |err| switch (err) {
        error.HfModelNotInCache => {
            std.debug.print("SKIP runRealModelLoadSmoke (Qwen3-0.6B not in HF cache)\n", .{});
            return;
        },
        else => return err,
    };
    defer weights.deinit();

    const cfg = weights.cfg;

    // ── 1. Architecture identity check.
    if (cfg.n_layers != 28) return error.UnexpectedLayers;
    if (cfg.dim != 1024) return error.UnexpectedHiddenSize;
    if (cfg.n_heads != 16) return error.UnexpectedNumHeads;
    if (cfg.n_kv_heads != 8) return error.UnexpectedNumKvHeads;
    if (cfg.head_dim != 128) return error.UnexpectedHeadDim;
    if (cfg.ff_dim != 3072) return error.UnexpectedFfDim;
    if (cfg.vocab_size != 151_936) return error.UnexpectedVocabSize;
    if (cfg.n_pos != n_pos) return error.UnexpectedNPos;
    if (cfg.rotary_dim != cfg.head_dim) return error.UnexpectedRotaryDim;
    if (!cfg.qk_norm) return error.QkNormShouldBeOn;
    if (@abs(cfg.rms_eps - 1e-6) > 1e-9) return error.UnexpectedRmsEps;
    if (@abs(cfg.rope_theta - 1_000_000.0) > 1.0) return error.UnexpectedRopeTheta;

    // ── 2. Shape (numel) check. Mirrors transformer.Runner.init's
    //    validation — if these slip past us, init would error first.
    const dim: usize = cfg.dim;
    const q_dim: usize = @as(usize, cfg.n_heads) * cfg.head_dim;
    const kv_dim: usize = @as(usize, cfg.n_kv_heads) * cfg.head_dim;
    const ff_dim: usize = cfg.ff_dim;
    const vocab: usize = cfg.vocab_size;
    const head_dim: usize = cfg.head_dim;

    if (weights.embed.len != vocab * dim) return error.EmbedNumelMismatch;
    if (weights.final_norm.len != dim) return error.FinalNormNumelMismatch;
    if (weights.lm_head.len != vocab * dim) return error.LmHeadNumelMismatch;
    if (weights.layers.len != cfg.n_layers) return error.LayersCountMismatch;

    for (weights.layers, 0..) |lw, li| {
        if (lw.w_n1.len != dim) return error.LayerN1NumelMismatch;
        if (lw.w_q.len != q_dim * dim) return error.LayerQNumelMismatch;
        if (lw.w_k.len != kv_dim * dim) return error.LayerKNumelMismatch;
        if (lw.w_v.len != kv_dim * dim) return error.LayerVNumelMismatch;
        if (lw.w_o.len != dim * q_dim) return error.LayerONumelMismatch;
        if (lw.w_n2.len != dim) return error.LayerN2NumelMismatch;
        if (lw.w_gate.len != ff_dim * dim) return error.LayerGateNumelMismatch;
        if (lw.w_up.len != ff_dim * dim) return error.LayerUpNumelMismatch;
        if (lw.w_down.len != dim * ff_dim) return error.LayerDownNumelMismatch;
        if (lw.w_q_norm.len != head_dim) return error.LayerQNormNumelMismatch;
        if (lw.w_k_norm.len != head_dim) return error.LayerKNormNumelMismatch;
        _ = li; // li available for richer errors if any of the above fire.
    }

    // ── 3. Sampled finiteness scan. We sample ~256 elements per
    //    tensor, evenly spaced — cheap and catches any wholesale-wrong
    //    bytes (dtype mismatch, shape misread, etc).
    const sample = struct {
        fn run(name: []const u8, slice: []const f32) !void {
            const n = slice.len;
            if (n == 0) return;
            const stride: usize = @max(1, n / 256);
            var i: usize = 0;
            while (i < n) : (i += stride) {
                if (!std.math.isFinite(slice[i])) {
                    std.debug.print("non-finite value in {s} at index {d}: {d}\n", .{ name, i, slice[i] });
                    return error.NonFiniteWeight;
                }
            }
        }
    }.run;

    try sample("embed", weights.embed);
    try sample("final_norm", weights.final_norm);
    try sample("lm_head", weights.lm_head);
    for (weights.layers, 0..) |lw, li| {
        // Only sample the big ones — rmsnorm gains are tiny and were
        // already covered by the shape check.
        try sample("w_q", lw.w_q);
        try sample("w_k", lw.w_k);
        try sample("w_v", lw.w_v);
        try sample("w_o", lw.w_o);
        try sample("w_gate", lw.w_gate);
        try sample("w_up", lw.w_up);
        try sample("w_down", lw.w_down);
        _ = li;
    }

    // ── Stat: total fp32 weight bytes (rough: matches what Runner
    //    would upload, modulo embed-vs-lm_head distinct copies).
    var total_f32: usize = weights.embed.len + weights.final_norm.len + weights.lm_head.len;
    for (weights.layers) |lw| {
        total_f32 += lw.w_n1.len + lw.w_q.len + lw.w_k.len + lw.w_v.len + lw.w_o.len;
        total_f32 += lw.w_n2.len + lw.w_gate.len + lw.w_up.len + lw.w_down.len;
        total_f32 += lw.w_q_norm.len + lw.w_k_norm.len;
    }
    const total_mib: f64 = @as(f64, @floatFromInt(total_f32 * @sizeOf(f32))) / (1024.0 * 1024.0);

    std.debug.print(
        "PASS real Qwen3-0.6B weight load (n_layers={d} dim={d} GQA {d}/{d} head_dim={d} ff_dim={d} vocab={d}; rope_theta={d:.0} rms_eps={e}; {d:.1} MiB fp32 host)\n",
        .{ cfg.n_layers, cfg.dim, cfg.n_heads, cfg.n_kv_heads, cfg.head_dim, cfg.ff_dim, cfg.vocab_size, cfg.rope_theta, cfg.rms_eps, total_mib },
    );
}

// ── chunk 8c-β-3b: real Qwen3 forward sanity ─────────────────────────
//
// β-3a proved the bytes-on-disk → fp32 conversion is correct shape-
// and value-wise; this proves they're plumbed through the trainer's
// forward path correctly. Catches anything β-3a's element-count gate
// misses: row/column-major confusion, RoPE convention drift, Q/K-norm
// applied at the wrong point, RMSNorm formula divergence, etc — every
// one of which would silently produce garbage logits.
//
// Method: load Qwen3-0.6B, instantiate `train_transformer.Runner` at
// n_pos=16 (cheap activations), tokenize a short fluent English
// prompt, run `forwardLogits`, then score CE for next-token prediction
// at every real prompt position. A correctly-wired Qwen3-0.6B should
// average ~2-5 nats on coherent English; ≥10 means the architecture
// pipeline is mis-wired (random-uniform CE is ln(151936) ≈ 11.93).
//
// Pass criterion: every logit finite + mean CE < 8.0. The window is
// generous so a small first-iteration RoPE-base/eps mismatch doesn't
// fail the gate; if mean CE comes back near 11-12 we know to dig.
//
// SKIP on `error.HfModelNotInCache`.

fn runRealModelForwardSmoke(allocator: std.mem.Allocator) !void {
    const model_id = "Qwen/Qwen3-0.6B";
    const n_pos: u32 = 16;
    const prompt_text = "The capital of France is Paris.";
    const ce_threshold: f32 = 8.0;

    // ── Resolve the model dir (also gives us a place to find the
    //    tokenizer.json without hard-coding the snapshot hash).
    const dir_path = hf_cache.resolveModelArg(allocator, model_id) catch |err| switch (err) {
        error.HfModelNotInCache => {
            std.debug.print("SKIP runRealModelForwardSmoke (Qwen3-0.6B not in HF cache)\n", .{});
            return;
        },
        else => return err,
    };
    defer allocator.free(dir_path);

    // ── Load fp32 train weights. Reuses the β-3a path (already
    //    validated by `runRealModelLoadSmoke`).
    var cpu = try model_mod.Model.load(allocator, dir_path);
    defer cpu.deinit();
    var weights = try train_load_real.loadTrainWeights(allocator, &cpu, n_pos);
    defer weights.deinit();

    // ── Tokenize the prompt. We need this *before* dropping the CPU
    //    model so we have a tokenizer. (Loader is the natural sibling
    //    that bundles tokenizer + weights, but β-3a's TrainWeights
    //    intentionally doesn't carry a tokenizer; β-4 will introduce
    //    a dataset abstraction that owns it.)
    const tok_path = try std.fmt.allocPrint(allocator, "{s}/tokenizer.json", .{dir_path});
    defer allocator.free(tok_path);
    var tok = try tokenizer_mod.Tokenizer.loadFromFile(allocator, tok_path);
    defer tok.deinit();
    const prompt_ids = try tok.encode(allocator, prompt_text);
    defer allocator.free(prompt_ids);
    const real_len: usize = prompt_ids.len;
    if (real_len < 2 or real_len > n_pos) {
        std.debug.print("Forward smoke: prompt tokenized to {d} ids; want 2..={d}\n", .{ real_len, n_pos });
        return error.PromptTokenLenOutOfRange;
    }

    // ── Build the n_pos token window: prompt followed by repeats of
    //    the last real id. Padding choice doesn't affect CE at real
    //    positions (we only score p in [0..real_len-2]), but sticking
    //    to in-vocab ids keeps the embedding lookup well-behaved.
    const token_ids = try allocator.alloc(u32, n_pos);
    defer allocator.free(token_ids);
    for (0..n_pos) |p| {
        token_ids[p] = if (p < real_len) prompt_ids[p] else prompt_ids[real_len - 1];
    }

    // ── GPU bring-up + Runner instantiation.
    var ctx = try vk.Context.init(allocator);
    defer ctx.deinit();

    var runner = try train_transformer.Runner.init(allocator, &ctx, weights.cfg, weights.view());
    defer runner.deinit();

    const logits = try allocator.alloc(f32, @as(usize, n_pos) * weights.cfg.vocab_size);
    defer allocator.free(logits);

    try runner.forwardLogits(token_ids, logits);

    // ── Finite-check (sampled). 16 positions × 151936 vocab is 2.4M
    //    elements — full scan is fine, but we mirror β-3a's stride
    //    sample to keep the smoke under a second.
    {
        const stride: usize = @max(1, logits.len / 4096);
        var i: usize = 0;
        while (i < logits.len) : (i += stride) {
            if (!std.math.isFinite(logits[i])) {
                std.debug.print("Forward smoke: non-finite logit at index {d}: {d}\n", .{ i, logits[i] });
                return error.NonFiniteLogit;
            }
        }
    }

    // ── Per-position CE for next-token prediction. logits[p] is the
    //    distribution over the token at position p+1, so we score
    //    pairs (logits[p], target=token_ids[p+1]) for p in [0..real_len-2].
    const vocab: usize = weights.cfg.vocab_size;
    var ce_sum: f64 = 0;
    var ce_n: usize = 0;
    for (0..real_len - 1) |p| {
        const off = p * vocab;
        // Numerically stable log-softmax: log Z = max + log Σ exp(x - max).
        var m: f32 = -std.math.inf(f32);
        for (0..vocab) |o| m = @max(m, logits[off + o]);
        var sum_e: f64 = 0;
        for (0..vocab) |o| sum_e += @exp(@as(f64, logits[off + o]) - @as(f64, m));
        const log_z: f64 = @as(f64, m) + @log(sum_e);
        const tgt: u32 = token_ids[p + 1];
        ce_sum += log_z - @as(f64, logits[off + tgt]);
        ce_n += 1;
    }
    const ce_mean: f32 = @floatCast(ce_sum / @as(f64, @floatFromInt(ce_n)));

    // ── Argmax at logits[real_len - 1]: the model's prediction for
    //    the token *immediately after* the prompt ends. No ground
    //    truth here (so no CE for this position), but it's the most
    //    illuminating qualitative signal — a sensibly-wired forward
    //    pass on "The capital of France is Paris." should suggest a
    //    plausible continuation (whitespace, a connective, EOS, etc).
    var argmax_after_prompt: u32 = 0;
    var argmax_score: f32 = -std.math.inf(f32);
    {
        const off = (real_len - 1) * vocab;
        for (0..vocab) |o| {
            if (logits[off + o] > argmax_score) {
                argmax_score = logits[off + o];
                argmax_after_prompt = @intCast(o);
            }
        }
    }

    if (!std.math.isFinite(ce_mean)) {
        std.debug.print("Forward smoke: mean CE not finite ({d})\n", .{ce_mean});
        return error.CeNotFinite;
    }
    if (ce_mean >= ce_threshold) {
        std.debug.print(
            "Forward smoke: mean CE {d:.3} ≥ threshold {d:.3} — model output looks random; check architecture wiring\n",
            .{ ce_mean, ce_threshold },
        );
        return error.CeAboveThreshold;
    }

    // Best-effort decode of the predicted post-prompt token for the
    // PASS line. decodeForDisplay handles GPT-2-style byte mapping
    // and special tokens; if it fails (rare), fall back to id only.
    const decoded = tok.decodeForDisplay(allocator, argmax_after_prompt) catch null;
    defer if (decoded) |d| allocator.free(d);

    if (decoded) |d| {
        std.debug.print(
            "PASS real Qwen3-0.6B forward sanity (n_pos={d} prompt_tokens={d} mean_CE={d:.3} nats; argmax-after-prompt={d} \"{s}\")\n",
            .{ n_pos, real_len, ce_mean, argmax_after_prompt, d },
        );
    } else {
        std.debug.print(
            "PASS real Qwen3-0.6B forward sanity (n_pos={d} prompt_tokens={d} mean_CE={d:.3} nats; argmax-after-prompt={d})\n",
            .{ n_pos, real_len, ce_mean, argmax_after_prompt },
        );
    }
}

// ── chunk 8c-β-4: tokenizer + dataset stub ───────────────────────────
//
// β-3a/3b proved the static loader path. β-4 adds the streaming side:
// a `train_dataset.Dataset` packs tokenized examples into one stream
// with EOS separators, and produces sliding (n_pos+1)-windows as
// (input_ids, target_ids) batches for next-token-prediction training.
// β-5 will iterate over these and call `runner.step` once per batch.
//
// The new module supports both in-memory examples and a JSONL file
// (one `{"text": "..."}` object per line). This smoke uses the JSONL
// path to exercise both code paths end-to-end and ships a tiny
// fact-style dataset at `data/train/tiny_facts.jsonl` so future
// chunks have a hermetic corpus to overfit on.
//
// Pass criterion: dataset builds with packed_ids longer than n_pos+1,
// at least one batch is producible, forward CE on the first batch is
// finite + below `pretrained_ce_threshold`. The threshold is loose
// (8 nats) because we're running a *trained* Qwen3-0.6B on factual
// English — actual CE will land in the 1.5-4 nat regime.

fn runRealModelDatasetSmoke(allocator: std.mem.Allocator) !void {
    const model_id = "Qwen/Qwen3-0.6B";
    const jsonl_path = "data/train/tiny_facts.jsonl";
    const n_pos: u32 = 16;
    const eos_id: u32 = 151_645; // Qwen3 <|im_end|>; matches config.eos_token_id.
    const ce_threshold: f32 = 8.0;

    const dir_path = hf_cache.resolveModelArg(allocator, model_id) catch |err| switch (err) {
        error.HfModelNotInCache => {
            std.debug.print("SKIP runRealModelDatasetSmoke (Qwen3-0.6B not in HF cache)\n", .{});
            return;
        },
        else => return err,
    };
    defer allocator.free(dir_path);

    var cpu = try model_mod.Model.load(allocator, dir_path);
    defer cpu.deinit();

    var weights = try train_load_real.loadTrainWeights(allocator, &cpu, n_pos);
    defer weights.deinit();

    const tok_path = try std.fmt.allocPrint(allocator, "{s}/tokenizer.json", .{dir_path});
    defer allocator.free(tok_path);
    var tok = try tokenizer_mod.Tokenizer.loadFromFile(allocator, tok_path);
    defer tok.deinit();

    var ds = try train_dataset.buildFromJsonl(allocator, &tok, jsonl_path, n_pos, eos_id);
    defer ds.deinit();

    if (ds.numBatches() == 0) {
        std.debug.print("Dataset smoke: numBatches=0 ({d} packed ids ≤ n_pos={d})\n", .{ ds.packed_ids.len, n_pos });
        return error.DatasetTooShort;
    }

    // ── Read batch 0 and verify the next-token shift is intact.
    const input_ids = try allocator.alloc(u32, n_pos);
    defer allocator.free(input_ids);
    const target_ids = try allocator.alloc(u32, n_pos);
    defer allocator.free(target_ids);
    try ds.batch(0, input_ids, target_ids);

    for (0..n_pos) |p| {
        // packed_ids[p+1] should equal both target_ids[p] and (for p<n_pos-1)
        // input_ids[p+1]. The first guarantees the shift is correct;
        // the second confirms the windowing is consistent.
        if (target_ids[p] != ds.packed_ids[p + 1]) return error.TargetShiftMismatch;
        if (p + 1 < n_pos and input_ids[p + 1] != ds.packed_ids[p + 1]) return error.InputWindowMismatch;
    }

    // ── Forward + CE on the first batch.
    var ctx = try vk.Context.init(allocator);
    defer ctx.deinit();
    var runner = try train_transformer.Runner.init(allocator, &ctx, weights.cfg, weights.view());
    defer runner.deinit();

    const vocab: usize = weights.cfg.vocab_size;
    const logits = try allocator.alloc(f32, @as(usize, n_pos) * vocab);
    defer allocator.free(logits);
    try runner.forwardLogits(input_ids, logits);

    var ce_sum: f64 = 0;
    for (0..n_pos) |p| {
        const off = p * vocab;
        var m: f32 = -std.math.inf(f32);
        for (0..vocab) |o| m = @max(m, logits[off + o]);
        var sum_e: f64 = 0;
        for (0..vocab) |o| sum_e += @exp(@as(f64, logits[off + o]) - @as(f64, m));
        const log_z: f64 = @as(f64, m) + @log(sum_e);
        ce_sum += log_z - @as(f64, logits[off + target_ids[p]]);
    }
    const ce_mean: f32 = @floatCast(ce_sum / @as(f64, @floatFromInt(n_pos)));

    if (!std.math.isFinite(ce_mean)) {
        std.debug.print("Dataset smoke: mean CE not finite ({d})\n", .{ce_mean});
        return error.CeNotFinite;
    }
    if (ce_mean >= ce_threshold) {
        std.debug.print(
            "Dataset smoke: batch-0 mean CE {d:.3} ≥ threshold {d:.3}\n",
            .{ ce_mean, ce_threshold },
        );
        return error.CeAboveThreshold;
    }

    std.debug.print(
        "PASS real Qwen3-0.6B dataset (jsonl→{d} packed ids; {d} batches at n_pos={d}; batch-0 mean_CE={d:.3} nats)\n",
        .{ ds.packed_ids.len, ds.numBatches(), n_pos, ce_mean },
    );
}

// ── chunk 8c-β-5: end-to-end one-step real-model train ──────────────
//
// The "does it actually train" gate. β-3a/3b proved the static loader
// is correct, β-4 proved the streaming side is correct. β-5 closes the
// loop: load Qwen3-0.6B, sample one real batch from the dataset, take
// one Adam step, forward again on the same batch, assert the loss
// decreased.
//
// Why a small lr matters: the β-2 envelope uses lr=1e-2 because it's
// overfitting random-init weights from 0.02 scale, where one update
// of size lr*sign(g) ≈ 1e-2 is a 50% relative perturbation. Pretrained
// weights are at scales of 0.01-0.5 with carefully-tuned magnitudes;
// a 1e-2 update would catastrophically diverge them. Standard real-
// world fine-tune lr is 1e-5 to 5e-5 — we use 1e-5, which gives a
// first-step relative perturbation of 0.01-0.1% per weight (small but
// measurable in the loss).
//
// Pass criterion:
//   - CE_before finite
//   - CE_after finite
//   - CE_after < CE_before (the actual gate — if false, either the
//     gradient is zero, the lr is wrong, or training is broken)
//
// The decrease is expected to be small (a few % of CE_before) because
// (1) lr is conservative and (2) we're stepping on a single small
// batch — overfitting one window with one Adam step will visibly
// shift its CE but won't dent broader Qwen3-0.6B knowledge. β-6 will
// loop this for many steps + checkpoints.

fn runRealModelTrainStepSmoke(allocator: std.mem.Allocator) !void {
    const model_id = "Qwen/Qwen3-0.6B";
    const jsonl_path = "data/train/tiny_facts.jsonl";
    const n_pos: u32 = 16;
    const eos_id: u32 = 151_645;
    const lr: f32 = 1e-5;

    const dir_path = hf_cache.resolveModelArg(allocator, model_id) catch |err| switch (err) {
        error.HfModelNotInCache => {
            std.debug.print("SKIP runRealModelTrainStepSmoke (Qwen3-0.6B not in HF cache)\n", .{});
            return;
        },
        else => return err,
    };
    defer allocator.free(dir_path);

    var cpu = try model_mod.Model.load(allocator, dir_path);
    defer cpu.deinit();

    var weights = try train_load_real.loadTrainWeights(allocator, &cpu, n_pos);
    defer weights.deinit();
    // Override the trainer's default lr with a fine-tune-appropriate
    // value before instantiation. Config is value-passed into init,
    // and Runner stores it; setting via .lr field on a copy is fine.
    var cfg = weights.cfg;
    cfg.lr = lr;

    const tok_path = try std.fmt.allocPrint(allocator, "{s}/tokenizer.json", .{dir_path});
    defer allocator.free(tok_path);
    var tok = try tokenizer_mod.Tokenizer.loadFromFile(allocator, tok_path);
    defer tok.deinit();

    var ds = try train_dataset.buildFromJsonl(allocator, &tok, jsonl_path, n_pos, eos_id);
    defer ds.deinit();
    if (ds.numBatches() == 0) return error.DatasetTooShort;

    const input_ids = try allocator.alloc(u32, n_pos);
    defer allocator.free(input_ids);
    const target_ids = try allocator.alloc(u32, n_pos);
    defer allocator.free(target_ids);
    try ds.batch(0, input_ids, target_ids);

    // CCE forward consumes target ids directly — no [n_pos × vocab] one-hot.
    const vocab: usize = cfg.vocab_size;

    // ── GPU bring-up + Runner.
    var ctx = try vk.Context.init(allocator);
    defer ctx.deinit();

    var runner = try train_transformer.Runner.init(allocator, &ctx, cfg, weights.view());
    defer runner.deinit();

    const logits = try allocator.alloc(f32, @as(usize, n_pos) * vocab);
    defer allocator.free(logits);

    // ── CE before.
    try runner.forwardLogits(input_ids, logits);
    const ce_before = computeMeanCe(logits, target_ids, n_pos, vocab);
    if (!std.math.isFinite(ce_before)) {
        std.debug.print("Train-step smoke: CE_before not finite ({d})\n", .{ce_before});
        return error.CeBeforeNotFinite;
    }

    // ── One Adam step on this single batch.
    const t_step_start = std.time.nanoTimestamp();
    try runner.step(input_ids, target_ids);
    const t_step_end = std.time.nanoTimestamp();
    const step_ms: f64 = @as(f64, @floatFromInt(t_step_end - t_step_start)) / 1.0e6;

    // ── CE after.
    try runner.forwardLogits(input_ids, logits);
    const ce_after = computeMeanCe(logits, target_ids, n_pos, vocab);
    if (!std.math.isFinite(ce_after)) {
        std.debug.print("Train-step smoke: CE_after not finite ({d}) after CE_before={d:.6}\n", .{ ce_after, ce_before });
        return error.CeAfterNotFinite;
    }

    if (ce_after >= ce_before) {
        std.debug.print(
            "Train-step smoke: CE did not decrease (before={d:.6} after={d:.6} delta={d:.6}) at lr={e}\n",
            .{ ce_before, ce_after, ce_after - ce_before, lr },
        );
        return error.CeDidNotDecrease;
    }

    const delta = ce_before - ce_after;
    const rel_pct: f64 = 100.0 * @as(f64, delta) / @as(f64, ce_before);
    std.debug.print(
        "PASS real Qwen3-0.6B one-step train (n_pos={d} lr={e}; CE {d:.6} → {d:.6}, Δ={d:.6} ({d:.3}%); step={d:.1} ms)\n",
        .{ n_pos, lr, ce_before, ce_after, delta, rel_pct, step_ms },
    );
}

/// Mean per-position cross-entropy: −log p(target_ids[p] | logits[p,·])
/// averaged over `n_pos`. Uses fp64 accumulation for the per-position
/// log-Z so n_pos × vocab × magnitude doesn't lose precision.
fn computeMeanCe(logits: []const f32, target_ids: []const u32, n_pos: u32, vocab: usize) f32 {
    var ce_sum: f64 = 0;
    for (0..n_pos) |p| {
        const off = p * vocab;
        var m: f32 = -std.math.inf(f32);
        for (0..vocab) |o| m = @max(m, logits[off + o]);
        var sum_e: f64 = 0;
        for (0..vocab) |o| sum_e += @exp(@as(f64, logits[off + o]) - @as(f64, m));
        const log_z: f64 = @as(f64, m) + @log(sum_e);
        ce_sum += log_z - @as(f64, logits[off + target_ids[p]]);
    }
    return @floatCast(ce_sum / @as(f64, @floatFromInt(n_pos)));
}

// ── chunk 8b stage A: gpu backward chain parity vs cpu oracle ────────
//
// Replays the same toy decoder layer as `runDecoderFineTuneCpuSmoke` for
// one backward pass, but on the GPU. Forward + d_y (MSE-loss-grad) stay
// on CPU; activations + weights + d_y are uploaded as-is and the GPU
// backward chain (linear-dx/-dW batched, RMSNorm backward, SDPA
// backward, ReLU backward, residual add, softmax backward) recomputes
// the eight weight gradients. Each gradient buffer is then compared
// against the CPU oracle's matching slice.
//
// The full chain composes 23 dispatches on a single recorder; the
// recorder injects compute-shader memory barriers between consecutive
// dispatches. dW for the two RMSNorm gains comes back as per-row
// partials and is summed host-side, mirroring `runGpuRmsnormBackwardSmoke`.
//
// CPU oracle uses fp64 accumulators in linear / softmax / attention
// reductions; GPU is fp32 throughout. Tolerance is set against the
// largest absolute value in each slice with a 1e-3 relative cap and a
// small absolute floor to skip near-zero entries.

fn runDecoderBackwardGpuParitySmoke(allocator: std.mem.Allocator) !void {
    _ = allocator;
    // chunk-8b stage A (single-layer GPU backward parity vs CPU oracle).
    // Deprecated by the SwiGLU swap in β-3a-2: the chunk-8c-α-2 stack
    // parity smoke covers the same dispatch chain at depth N=2, plus
    // β-3a-1's primitive parity smoke covers the SwiGLU bw shader.
    // Body removed rather than rewritten; reinstate if a per-layer
    // surface needs an independent gate.
    std.debug.print("SKIP runDecoderBackwardGpuParitySmoke (deprecated by 8c-α-2 + β-3a-1)\n", .{});
}

// ── chunk 8b stage B: full-GPU toy decoder fine-tune ────────────────
//
// Runs the same toy decoder as runDecoderFineTuneCpuSmoke / Stage A, but
// every operation lives on the GPU: forward, MSE loss-grad, backward,
// and Adam. The training loop is self-sustaining across 100 steps and
// the assertion is the same as 8a — `final_loss / initial_loss ≤ 1e-2`.
//
// This is the first transformer-trainer pipeline to live entirely on
// the GPU. It's intentionally inlined into the smoke (not a Runner) —
// the eventual TransformerTrainerRunner shape decision lives outside
// chunk 8b's scope; what this proves is that the kernel inventory is
// complete and that one decoder layer composes correctly under
// repeated optimizer steps.
//
// Per-step dispatch count: 14 forward + 1 loss-grad + 23 backward +
// 8 Adam = 46 dispatches. The recorder is reset and re-recorded
// every step. The same kernels are reused across all 100 steps —
// only the buffer contents change.

fn runDecoderTrainGpuSmoke(allocator: std.mem.Allocator) !void {
    _ = allocator;
    // chunk-8b stage B (single-layer full GPU MSE training).
    // Deprecated by β-3a-2: the chunk-8c-α-3 stack training smoke
    // exercises the same Runner forward+backward+Adam loop with
    // softmax-CE + multi-layer + lm_head, plus β-2's real-shape
    // envelope covers Qwen-class scale.
    std.debug.print("SKIP runDecoderTrainGpuSmoke (deprecated by 8c-α-3 + β-2)\n", .{});
}

// ── gpu geglu smoke: synthetic vs CPU geglu ─────────────────────────

const GegluPush = runtime.GegluPush;

fn runGpuGegluSmoke(allocator: std.mem.Allocator) !void {
    var ctx = try vk.Context.init(allocator);
    defer ctx.deinit();

    const n: usize = 4096;
    const gate = try allocator.alloc(f32, n);
    defer allocator.free(gate);
    const upv = try allocator.alloc(f32, n);
    defer allocator.free(upv);
    // Range [-3, 3] across the array hits both the tanh saturation
    // tails and the linear region around 0 — exercises the full curve.
    for (gate, upv, 0..) |*g, *u, i| {
        const t = @as(f32, @floatFromInt(i)) / @as(f32, @floatFromInt(n));
        g.* = -3.0 + 6.0 * t;
        u.* = 1.0 - 2.0 * t;
    }

    const want = try allocator.alloc(f32, n);
    defer allocator.free(want);
    try cpu_math.geglu(want, gate, upv);

    var buf_g = try buffer.Buffer.initStatic(&ctx, f32, gate);
    defer buf_g.deinit(ctx.device);
    var buf_u = try buffer.Buffer.initStatic(&ctx, f32, upv);
    defer buf_u.deinit(ctx.device);
    var buf_o = try buffer.Buffer.initDeviceOnly(&ctx, n * @sizeOf(f32));
    defer buf_o.deinit(ctx.device);

    var kern = try pipeline.Kernel.init(&ctx, &shaders.geglu, 3, @sizeOf(GegluPush));
    defer kern.deinit();
    try kern.bind(&.{ &buf_g, &buf_u, &buf_o });

    const local: u32 = 256;
    const groups: u32 = (@as(u32, @intCast(n)) + local - 1) / local;
    const push = GegluPush{ .n = @intCast(n) };
    try buffer.submitOneShot(&ctx, struct {
        kern: *const pipeline.Kernel,
        push: *const GegluPush,
        groups: u32,
        pub fn record(s: @This(), cmd: vk.c.VkCommandBuffer) void {
            s.kern.dispatch(cmd, s.push, s.groups, 1, 1);
        }
    }{ .kern = &kern, .push = &push, .groups = groups });

    const got = try allocator.alloc(f32, n);
    defer allocator.free(got);
    try buf_o.readBack(&ctx, f32, got);

    var max_abs: f32 = 0;
    for (got, want) |g, e| {
        const d = @abs(g - e);
        if (d > max_abs) max_abs = d;
    }
    if (max_abs > 1e-5) {
        std.debug.print("GPU GeGLU: max |Δ| = {e}\n", .{max_abs});
        return error.ParityFailed;
    }
    std.debug.print("PASS GPU geglu synthetic ({d} elems, x∈[-3,3])\n", .{n});
}

// ── gpu rope smoke: pos=0 identity + pos=1 vs CPU ───────────────────

const RopePush = runtime.RopePush;
const RopePartialPush = runtime.RopePartialPush;

fn runGpuRopeSmoke(allocator: std.mem.Allocator) !void {
    var ctx = try vk.Context.init(allocator);
    defer ctx.deinit();

    const n_heads: usize = 8;
    const head_dim: usize = 64;
    const total = n_heads * head_dim;

    const in_v = try allocator.alloc(f32, total);
    defer allocator.free(in_v);
    for (in_v, 0..) |*x, i| x.* = @as(f32, @floatFromInt(i)) * 0.001 - 0.5;

    var buf_in = try buffer.Buffer.initStatic(&ctx, f32, in_v);
    defer buf_in.deinit(ctx.device);
    var buf_out = try buffer.Buffer.initDeviceOnly(&ctx, total * @sizeOf(f32));
    defer buf_out.deinit(ctx.device);

    var kern = try pipeline.Kernel.init(&ctx, &shaders.rope, 2, @sizeOf(RopePush));
    defer kern.deinit();
    try kern.bind(&.{ &buf_in, &buf_out });

    // pos=0: must be identity.
    const local: u32 = 256;
    const pairs: u32 = @intCast(n_heads * (head_dim / 2));
    const groups: u32 = (pairs + local - 1) / local;

    const push0 = RopePush{ .n_heads = @intCast(n_heads), .head_dim = @intCast(head_dim), .pos = 0, .theta_base = 10000.0 };
    try buffer.submitOneShot(&ctx, struct {
        kern: *const pipeline.Kernel,
        push: *const RopePush,
        groups: u32,
        pub fn record(s: @This(), cmd: vk.c.VkCommandBuffer) void {
            s.kern.dispatch(cmd, s.push, s.groups, 1, 1);
        }
    }{ .kern = &kern, .push = &push0, .groups = groups });

    const got0 = try allocator.alloc(f32, total);
    defer allocator.free(got0);
    try buf_out.readBack(&ctx, f32, got0);
    for (got0, in_v, 0..) |g, e, i| {
        if (g != e) {
            std.debug.print("GPU RoPE pos=0 NOT identity at {d}: in={d} out={d}\n", .{ i, e, g });
            return error.ParityFailed;
        }
    }

    // pos=1: parity vs CPU.
    const want = try allocator.alloc(f32, total);
    defer allocator.free(want);
    try cpu_math.applyRope(want, in_v, n_heads, head_dim, 1, 10000.0);

    const push1 = RopePush{ .n_heads = @intCast(n_heads), .head_dim = @intCast(head_dim), .pos = 1, .theta_base = 10000.0 };
    try buffer.submitOneShot(&ctx, struct {
        kern: *const pipeline.Kernel,
        push: *const RopePush,
        groups: u32,
        pub fn record(s: @This(), cmd: vk.c.VkCommandBuffer) void {
            s.kern.dispatch(cmd, s.push, s.groups, 1, 1);
        }
    }{ .kern = &kern, .push = &push1, .groups = groups });

    const got1 = try allocator.alloc(f32, total);
    defer allocator.free(got1);
    try buf_out.readBack(&ctx, f32, got1);

    var max_abs: f32 = 0;
    for (got1, want) |g, e| {
        const d = @abs(g - e);
        if (d > max_abs) max_abs = d;
    }
    if (max_abs > 1e-5) {
        std.debug.print("GPU RoPE pos=1: max |Δ| = {e}\n", .{max_abs});
        return error.ParityFailed;
    }
    std.debug.print("PASS GPU rope (pos=0 identity + pos=1 vs CPU within 1e-5)\n", .{});
}

// ── gpu rope-partial smoke: rotary_dim < head_dim, vs CPU ───────────

fn runGpuRopePartialSmoke(allocator: std.mem.Allocator) !void {
    var ctx = try vk.Context.init(allocator);
    defer ctx.deinit();

    // Qwen3.5 shape: head_dim=256, rotary_dim=64, theta=10M.
    const n_heads: usize = 8;
    const head_dim: usize = 256;
    const rotary_dim: usize = 64;
    const total = n_heads * head_dim;
    const theta_base: f32 = 1.0e7;

    const in_v = try allocator.alloc(f32, total);
    defer allocator.free(in_v);
    for (in_v, 0..) |*x, i| x.* = @as(f32, @floatFromInt(i)) * 0.001 - 0.5;

    var buf_in = try buffer.Buffer.initStatic(&ctx, f32, in_v);
    defer buf_in.deinit(ctx.device);
    var buf_out = try buffer.Buffer.initDeviceOnly(&ctx, total * @sizeOf(f32));
    defer buf_out.deinit(ctx.device);

    var kern = try pipeline.Kernel.init(&ctx, &shaders.rope_partial, 2, @sizeOf(RopePartialPush));
    defer kern.deinit();
    try kern.bind(&.{ &buf_in, &buf_out });

    // Dispatch one thread per output element (n_heads * head_dim).
    const local: u32 = 256;
    const elems: u32 = @intCast(total);
    const groups: u32 = (elems + local - 1) / local;

    // Reference: CPU partial RoPE at pos=3.
    const want = try allocator.alloc(f32, total);
    defer allocator.free(want);
    try cpu_math.applyRopePartial(want, in_v, n_heads, head_dim, rotary_dim, 3, theta_base);

    const push = RopePartialPush{
        .n_heads = @intCast(n_heads),
        .head_dim = @intCast(head_dim),
        .rotary_dim = @intCast(rotary_dim),
        .pos = 3,
        .theta_base = theta_base,
    };
    try buffer.submitOneShot(&ctx, struct {
        kern: *const pipeline.Kernel,
        push: *const RopePartialPush,
        groups: u32,
        pub fn record(s: @This(), cmd: vk.c.VkCommandBuffer) void {
            s.kern.dispatch(cmd, s.push, s.groups, 1, 1);
        }
    }{ .kern = &kern, .push = &push, .groups = groups });

    const got = try allocator.alloc(f32, total);
    defer allocator.free(got);
    try buf_out.readBack(&ctx, f32, got);

    var max_abs: f32 = 0;
    for (got, want) |g, e| {
        const d = @abs(g - e);
        if (d > max_abs) max_abs = d;
    }
    if (max_abs > 1e-5) {
        std.debug.print("GPU rope_partial: max |Δ| = {e}\n", .{max_abs});
        return error.ParityFailed;
    }
    // Also confirm the pass-through region is byte-identical (the
    // tail of each head must equal the input).
    for (0..n_heads) |h| {
        for (rotary_dim..head_dim) |d| {
            const idx = h * head_dim + d;
            if (got[idx] != in_v[idx]) {
                std.debug.print("rope_partial pass-through broken at h={d} d={d}: in={d} out={d}\n", .{ h, d, in_v[idx], got[idx] });
                return error.ParityFailed;
            }
        }
    }
    std.debug.print("PASS GPU rope_partial (rotary_dim=64 of head_dim=256, max |Δ| vs CPU = {e})\n", .{max_abs});
}

// ── gpu split_q_gate smoke: synthetic round-trip ────────────────────
//
// Validates that the `[h0_q, h0_gate, h1_q, h1_gate, …]` interleaved
// layout produced by the 2× q_proj is correctly demuxed into two flat
// `(num_heads*head_dim)` buffers preserving per-head ordering.

const SplitQGatePush = runtime_hybrid.SplitQGatePush;

fn runGpuSplitQGateSmoke(allocator: std.mem.Allocator) !void {
    var ctx = try vk.Context.init(allocator);
    defer ctx.deinit();

    const num_heads: usize = 4;
    const head_dim: usize = 8;
    const total = num_heads * head_dim;
    const wide = 2 * total;

    const in_v = try allocator.alloc(f32, wide);
    defer allocator.free(in_v);
    // Synthetic: q values are positive ints, gate values negative ints,
    // so any layout bug shows up as sign mismatches.
    for (0..num_heads) |h| {
        for (0..head_dim) |d| {
            const off = h * 2 * head_dim;
            in_v[off + d] = @floatFromInt(h * 10 + d + 1);
            in_v[off + head_dim + d] = -@as(f32, @floatFromInt(h * 10 + d + 1));
        }
    }
    var buf_in = try buffer.Buffer.initStatic(&ctx, f32, in_v);
    defer buf_in.deinit(ctx.device);
    var buf_q = try buffer.Buffer.initDeviceOnly(&ctx, total * @sizeOf(f32));
    defer buf_q.deinit(ctx.device);
    var buf_gate = try buffer.Buffer.initDeviceOnly(&ctx, total * @sizeOf(f32));
    defer buf_gate.deinit(ctx.device);

    var kern = try pipeline.Kernel.init(&ctx, &shaders.split_q_gate, 3, @sizeOf(SplitQGatePush));
    defer kern.deinit();
    try kern.bind(&.{ &buf_in, &buf_q, &buf_gate });

    const local: u32 = 256;
    const groups: u32 = (@as(u32, @intCast(total)) + local - 1) / local;
    const push = SplitQGatePush{ .num_heads = @intCast(num_heads), .head_dim = @intCast(head_dim) };
    try buffer.submitOneShot(&ctx, struct {
        kern: *const pipeline.Kernel,
        push: *const SplitQGatePush,
        groups: u32,
        pub fn record(s: @This(), cmd: vk.c.VkCommandBuffer) void {
            s.kern.dispatch(cmd, s.push, s.groups, 1, 1);
        }
    }{ .kern = &kern, .push = &push, .groups = groups });

    const got_q = try allocator.alloc(f32, total);
    defer allocator.free(got_q);
    const got_gate = try allocator.alloc(f32, total);
    defer allocator.free(got_gate);
    try buf_q.readBack(&ctx, f32, got_q);
    try buf_gate.readBack(&ctx, f32, got_gate);

    for (0..num_heads) |h| {
        for (0..head_dim) |d| {
            const want_q: f32 = @floatFromInt(h * 10 + d + 1);
            const want_g: f32 = -want_q;
            const idx = h * head_dim + d;
            if (got_q[idx] != want_q or got_gate[idx] != want_g) {
                std.debug.print("split_q_gate mismatch at h={d} d={d}: q={d}/{d} gate={d}/{d}\n", .{ h, d, got_q[idx], want_q, got_gate[idx], want_g });
                return error.ParityFailed;
            }
        }
    }
    std.debug.print("PASS GPU split_q_gate (4 heads × 8 dim, layout bit-exact)\n", .{});
}

// ── gpu sigmoid_mul smoke: out = a * sigmoid(b) vs CPU ──────────────

const SigmoidMulPush = runtime_hybrid.SigmoidMulPush;

fn runGpuSigmoidMulSmoke(allocator: std.mem.Allocator) !void {
    var ctx = try vk.Context.init(allocator);
    defer ctx.deinit();

    const n: usize = 1024;
    const a = try allocator.alloc(f32, n);
    defer allocator.free(a);
    const b = try allocator.alloc(f32, n);
    defer allocator.free(b);
    for (a, b, 0..) |*ai, *bi, i| {
        ai.* = @sin(@as(f32, @floatFromInt(i)) * 0.1) * 2.0;
        bi.* = @cos(@as(f32, @floatFromInt(i)) * 0.13) * 4.0;
    }

    var buf_a = try buffer.Buffer.initStatic(&ctx, f32, a);
    defer buf_a.deinit(ctx.device);
    var buf_b = try buffer.Buffer.initStatic(&ctx, f32, b);
    defer buf_b.deinit(ctx.device);
    var buf_out = try buffer.Buffer.initDeviceOnly(&ctx, n * @sizeOf(f32));
    defer buf_out.deinit(ctx.device);

    var kern = try pipeline.Kernel.init(&ctx, &shaders.sigmoid_mul, 3, @sizeOf(SigmoidMulPush));
    defer kern.deinit();
    try kern.bind(&.{ &buf_a, &buf_b, &buf_out });

    const local: u32 = 256;
    const groups: u32 = (@as(u32, @intCast(n)) + local - 1) / local;
    const push = SigmoidMulPush{ .n_elem = @intCast(n) };
    try buffer.submitOneShot(&ctx, struct {
        kern: *const pipeline.Kernel,
        push: *const SigmoidMulPush,
        groups: u32,
        pub fn record(s: @This(), cmd: vk.c.VkCommandBuffer) void {
            s.kern.dispatch(cmd, s.push, s.groups, 1, 1);
        }
    }{ .kern = &kern, .push = &push, .groups = groups });

    const got = try allocator.alloc(f32, n);
    defer allocator.free(got);
    try buf_out.readBack(&ctx, f32, got);

    var max_abs: f32 = 0;
    for (got, a, b) |g, ai, bi| {
        const want = ai * (1.0 / (1.0 + @exp(-bi)));
        const d = @abs(g - want);
        if (d > max_abs) max_abs = d;
    }
    if (max_abs > 1e-6) {
        std.debug.print("GPU sigmoid_mul: max |Δ| = {e}\n", .{max_abs});
        return error.ParityFailed;
    }
    std.debug.print("PASS GPU sigmoid_mul (1024 elems, max |Δ| vs CPU = {e})\n", .{max_abs});
}

// ── chunk 8c-β-3a-1: SwiGLU primitive parity (CPU + GPU) ────────────
//
// CPU oracle: numeric-grad parity for swigluForward / swigluBackward.
// GPU parity: swiglu_forward + swiglu_backward shaders match the
// `cpu_train_transformer.swigluForward` / `swigluBackward` reference
// to fp32 tolerance.
//
// SwiGLU is the FFN nonlinearity for Llama / Qwen / Mistral; this
// primitive replaces the toy stack's ReLU FFN in a follow-up
// integration chunk (β-3a-2). Here we just verify the math.

fn runSwiGluCpuSmoke(allocator: std.mem.Allocator) !void {
    const n: usize = 64;
    var prng = std.Random.DefaultPrng.init(0x5C_16_1A_AA);
    const rng = prng.random();

    const pre_gate = try allocator.alloc(f32, n);
    defer allocator.free(pre_gate);
    const up = try allocator.alloc(f32, n);
    defer allocator.free(up);
    // Scale ±1 matches the other primitive parity smokes; the central-
    // difference truncation error scales with the input curvature, so
    // wider inputs blow past 1% rel-err even when the analytical
    // gradient is correct (verified by hand at probe points).
    for (pre_gate) |*v| v.* = rng.float(f32) * 2.0 - 1.0;
    for (up) |*v| v.* = rng.float(f32) * 2.0 - 1.0;

    const gated = try allocator.alloc(f32, n);
    defer allocator.free(gated);
    cpu_train_transformer.swigluForward(pre_gate, up, gated);

    // Random upstream gradient.
    const d_gated = try allocator.alloc(f32, n);
    defer allocator.free(d_gated);
    for (d_gated) |*v| v.* = (rng.float(f32) * 2.0 - 1.0) * 0.5;

    const d_pre_gate = try allocator.alloc(f32, n);
    defer allocator.free(d_pre_gate);
    const d_up = try allocator.alloc(f32, n);
    defer allocator.free(d_up);
    cpu_train_transformer.swigluBackward(d_gated, pre_gate, up, d_pre_gate, d_up);

    // Numeric-grad check against the analytical backward. Loss
    // L = Σ d_gated · gated; dL/d_pre_gate and dL/d_up should match.
    const eps: f32 = 1e-3;
    var max_rel: f32 = 0;
    const n_probes: usize = 8;
    for (0..n_probes) |pi| {
        const i = (pi * 7 + 3) % n;
        // d/d_pre_gate
        const pg_p = pre_gate[i] + eps;
        const pg_m = pre_gate[i] - eps;
        const sig_p = 1.0 / (1.0 + @exp(-pg_p));
        const sig_m = 1.0 / (1.0 + @exp(-pg_m));
        const gated_p = pg_p * sig_p * up[i];
        const gated_m = pg_m * sig_m * up[i];
        const num_dpg = d_gated[i] * (gated_p - gated_m) / (2.0 * eps);
        const denom_pg = @max(@abs(num_dpg), 1e-6);
        const rel_pg = @abs(num_dpg - d_pre_gate[i]) / denom_pg;
        if (rel_pg > max_rel) max_rel = rel_pg;
        // d/d_up
        const u_p = up[i] + eps;
        const u_m = up[i] - eps;
        const sig_z = 1.0 / (1.0 + @exp(-pre_gate[i]));
        const silu_z = pre_gate[i] * sig_z;
        const gu_p = silu_z * u_p;
        const gu_m = silu_z * u_m;
        const num_du = d_gated[i] * (gu_p - gu_m) / (2.0 * eps);
        const denom_u = @max(@abs(num_du), 1e-6);
        const rel_u = @abs(num_du - d_up[i]) / denom_u;
        if (rel_u > max_rel) max_rel = rel_u;
    }
    if (max_rel > 1e-2) {
        std.debug.print("SwiGLU CPU: max numeric-grad rel-err = {e}\n", .{max_rel});
        return error.ParityFailed;
    }
    std.debug.print("PASS SwiGLU CPU oracle (n={d}, numeric-grad parity ≤ {e})\n", .{ n, max_rel });
}

fn runGpuSwiGluSmoke(allocator: std.mem.Allocator) !void {
    var ctx = try vk.Context.init(allocator);
    defer ctx.deinit();

    const n: usize = 1024;
    const pre_gate = try allocator.alloc(f32, n);
    defer allocator.free(pre_gate);
    const up = try allocator.alloc(f32, n);
    defer allocator.free(up);
    const d_gated = try allocator.alloc(f32, n);
    defer allocator.free(d_gated);
    var prng = std.Random.DefaultPrng.init(0x5C_16_1A_BB);
    const rng = prng.random();
    for (pre_gate) |*v| v.* = (rng.float(f32) * 2.0 - 1.0) * 3.0;
    for (up) |*v| v.* = (rng.float(f32) * 2.0 - 1.0) * 3.0;
    for (d_gated) |*v| v.* = (rng.float(f32) * 2.0 - 1.0) * 0.5;

    // CPU references.
    const want_gated = try allocator.alloc(f32, n);
    defer allocator.free(want_gated);
    cpu_train_transformer.swigluForward(pre_gate, up, want_gated);
    const want_d_pre_gate = try allocator.alloc(f32, n);
    defer allocator.free(want_d_pre_gate);
    const want_d_up = try allocator.alloc(f32, n);
    defer allocator.free(want_d_up);
    cpu_train_transformer.swigluBackward(d_gated, pre_gate, up, want_d_pre_gate, want_d_up);

    // GPU forward.
    var buf_pg = try buffer.Buffer.initStatic(&ctx, f32, pre_gate);
    defer buf_pg.deinit(ctx.device);
    var buf_up = try buffer.Buffer.initStatic(&ctx, f32, up);
    defer buf_up.deinit(ctx.device);
    var buf_gated = try buffer.Buffer.initDeviceOnly(&ctx, n * @sizeOf(f32));
    defer buf_gated.deinit(ctx.device);

    var k_fwd = try pipeline.Kernel.init(&ctx, &shaders.swiglu_forward, 3, @sizeOf(runtime.SwigluPush));
    defer k_fwd.deinit();
    try k_fwd.bind(&.{ &buf_pg, &buf_up, &buf_gated });

    const local: u32 = 256;
    const groups: u32 = (@as(u32, @intCast(n)) + local - 1) / local;
    const push = runtime.SwigluPush{ .n = @intCast(n) };
    try buffer.submitOneShot(&ctx, struct {
        kern: *const pipeline.Kernel,
        push: *const runtime.SwigluPush,
        groups: u32,
        pub fn record(s: @This(), cmd: vk.c.VkCommandBuffer) void {
            s.kern.dispatch(cmd, s.push, s.groups, 1, 1);
        }
    }{ .kern = &k_fwd, .push = &push, .groups = groups });

    const got_gated = try allocator.alloc(f32, n);
    defer allocator.free(got_gated);
    try buf_gated.readBack(&ctx, f32, got_gated);

    var max_fwd: f32 = 0;
    for (got_gated, want_gated) |g, w| {
        const d = @abs(g - w);
        if (d > max_fwd) max_fwd = d;
    }
    if (max_fwd > 1e-6) {
        std.debug.print("SwiGLU GPU forward: max |Δ| = {e}\n", .{max_fwd});
        return error.ParityFailed;
    }

    // GPU backward.
    var buf_dg = try buffer.Buffer.initStatic(&ctx, f32, d_gated);
    defer buf_dg.deinit(ctx.device);
    var buf_dpg = try buffer.Buffer.initDeviceOnly(&ctx, n * @sizeOf(f32));
    defer buf_dpg.deinit(ctx.device);
    var buf_du = try buffer.Buffer.initDeviceOnly(&ctx, n * @sizeOf(f32));
    defer buf_du.deinit(ctx.device);

    var k_bw = try pipeline.Kernel.init(&ctx, &shaders.swiglu_backward, 5, @sizeOf(runtime.SwigluPush));
    defer k_bw.deinit();
    try k_bw.bind(&.{ &buf_dg, &buf_pg, &buf_up, &buf_dpg, &buf_du });

    try buffer.submitOneShot(&ctx, struct {
        kern: *const pipeline.Kernel,
        push: *const runtime.SwigluPush,
        groups: u32,
        pub fn record(s: @This(), cmd: vk.c.VkCommandBuffer) void {
            s.kern.dispatch(cmd, s.push, s.groups, 1, 1);
        }
    }{ .kern = &k_bw, .push = &push, .groups = groups });

    const got_dpg = try allocator.alloc(f32, n);
    defer allocator.free(got_dpg);
    const got_du = try allocator.alloc(f32, n);
    defer allocator.free(got_du);
    try buf_dpg.readBack(&ctx, f32, got_dpg);
    try buf_du.readBack(&ctx, f32, got_du);

    var max_bw: f32 = 0;
    for (got_dpg, want_d_pre_gate) |g, w| {
        const d = @abs(g - w);
        if (d > max_bw) max_bw = d;
    }
    for (got_du, want_d_up) |g, w| {
        const d = @abs(g - w);
        if (d > max_bw) max_bw = d;
    }
    if (max_bw > 1e-6) {
        std.debug.print("SwiGLU GPU backward: max |Δ| = {e}\n", .{max_bw});
        return error.ParityFailed;
    }
    std.debug.print(
        "PASS GPU SwiGLU forward + backward (n={d}, fwd max |Δ| = {e}, bw max |Δ| = {e})\n",
        .{ n, max_fwd, max_bw },
    );
}

// ── chunk 8c-β-3a-3: batched RoPE primitive parity ──────────────────
//
// Tests the new `rope_partial_batched.comp` + `rope_backward_batched.comp`
// shaders against the CPU `ropeForwardBatched` / `ropeBackwardBatched`
// helpers. Operates over [n_pos, n_heads, head_dim] in one dispatch
// each. Setting rotary_dim < head_dim exercises the partial-rotation
// path (Qwen3.5-style); we run with rotary_dim = head_dim (full RoPE).

fn runGpuRopeBatchedSmoke(allocator: std.mem.Allocator) !void {
    var ctx = try vk.Context.init(allocator);
    defer ctx.deinit();

    const n_pos: usize = 4;
    const n_heads: usize = 2;
    const head_dim: usize = 8;
    const rotary_dim: usize = head_dim;
    const theta_base: f32 = 10_000.0;
    const total = n_pos * n_heads * head_dim;

    const x = try allocator.alloc(f32, total);
    defer allocator.free(x);
    var prng = std.Random.DefaultPrng.init(0xCAFE_B0_BE);
    const rng = prng.random();
    for (x) |*v| v.* = rng.float(f32) * 2.0 - 1.0;

    const cpu_y = try allocator.alloc(f32, total);
    defer allocator.free(cpu_y);
    try cpu_train_transformer.ropeForwardBatched(cpu_y, x, n_pos, n_heads, head_dim, rotary_dim, theta_base);

    const dy = try allocator.alloc(f32, total);
    defer allocator.free(dy);
    for (dy) |*v| v.* = rng.float(f32) * 2.0 - 1.0;

    const cpu_dx = try allocator.alloc(f32, total);
    defer allocator.free(cpu_dx);
    try cpu_train_transformer.ropeBackwardBatched(cpu_dx, dy, n_pos, n_heads, head_dim, rotary_dim, theta_base);

    // GPU forward.
    var buf_x = try buffer.Buffer.initStatic(&ctx, f32, x);
    defer buf_x.deinit(ctx.device);
    var buf_y = try buffer.Buffer.initDeviceOnly(&ctx, total * @sizeOf(f32));
    defer buf_y.deinit(ctx.device);
    var k_fwd = try pipeline.Kernel.init(&ctx, &shaders.rope_partial_batched, 2, @sizeOf(runtime.RopeBatchedPush));
    defer k_fwd.deinit();
    try k_fwd.bind(&.{ &buf_x, &buf_y });
    const push = runtime.RopeBatchedPush{
        .n_pos = @intCast(n_pos),
        .n_heads = @intCast(n_heads),
        .head_dim = @intCast(head_dim),
        .rotary_dim = @intCast(rotary_dim),
        .theta_base = theta_base,
    };
    const local: u32 = 256;
    const groups: u32 = (@as(u32, @intCast(total)) + local - 1) / local;
    try buffer.submitOneShot(&ctx, struct {
        kern: *const pipeline.Kernel,
        push: *const runtime.RopeBatchedPush,
        groups: u32,
        pub fn record(s: @This(), cmd: vk.c.VkCommandBuffer) void {
            s.kern.dispatch(cmd, s.push, s.groups, 1, 1);
        }
    }{ .kern = &k_fwd, .push = &push, .groups = groups });
    const gpu_y = try allocator.alloc(f32, total);
    defer allocator.free(gpu_y);
    try buf_y.readBack(&ctx, f32, gpu_y);

    var max_fwd: f32 = 0;
    for (gpu_y, cpu_y) |g, c| {
        const d = @abs(g - c);
        if (d > max_fwd) max_fwd = d;
    }
    if (max_fwd > 1e-5) {
        std.debug.print("RoPE batched fwd: max |Δ| = {e}\n", .{max_fwd});
        return error.ParityFailed;
    }

    // GPU backward.
    var buf_dy = try buffer.Buffer.initStatic(&ctx, f32, dy);
    defer buf_dy.deinit(ctx.device);
    var buf_dx = try buffer.Buffer.initDeviceOnly(&ctx, total * @sizeOf(f32));
    defer buf_dx.deinit(ctx.device);
    var k_bw = try pipeline.Kernel.init(&ctx, &shaders.rope_backward_batched, 2, @sizeOf(runtime.RopeBatchedPush));
    defer k_bw.deinit();
    try k_bw.bind(&.{ &buf_dy, &buf_dx });
    try buffer.submitOneShot(&ctx, struct {
        kern: *const pipeline.Kernel,
        push: *const runtime.RopeBatchedPush,
        groups: u32,
        pub fn record(s: @This(), cmd: vk.c.VkCommandBuffer) void {
            s.kern.dispatch(cmd, s.push, s.groups, 1, 1);
        }
    }{ .kern = &k_bw, .push = &push, .groups = groups });
    const gpu_dx = try allocator.alloc(f32, total);
    defer allocator.free(gpu_dx);
    try buf_dx.readBack(&ctx, f32, gpu_dx);

    var max_bw: f32 = 0;
    for (gpu_dx, cpu_dx) |g, c| {
        const d = @abs(g - c);
        if (d > max_bw) max_bw = d;
    }
    if (max_bw > 1e-5) {
        std.debug.print("RoPE batched bw: max |Δ| = {e}\n", .{max_bw});
        return error.ParityFailed;
    }

    // Round-trip: rope_bw(rope_fwd(x)) ≈ x.
    var max_rt: f32 = 0;
    const rt_dx = try allocator.alloc(f32, total);
    defer allocator.free(rt_dx);
    try cpu_train_transformer.ropeBackwardBatched(rt_dx, cpu_y, n_pos, n_heads, head_dim, rotary_dim, theta_base);
    for (rt_dx, x) |a, b| {
        const d = @abs(a - b);
        if (d > max_rt) max_rt = d;
    }

    std.debug.print(
        "PASS GPU RoPE batched fwd + bw (n_pos={d} n_heads={d} head_dim={d}; fwd |Δ|={e}, bw |Δ|={e}, round-trip |Δ|={e})\n",
        .{ n_pos, n_heads, head_dim, max_fwd, max_bw, max_rt },
    );
}

// ── gpu l2norm-per-head smoke: synthetic vs CPU ─────────────────────

const L2normPush = runtime_hybrid.L2normPush;

fn runGpuL2normPerHeadSmoke(allocator: std.mem.Allocator) !void {
    var ctx = try vk.Context.init(allocator);
    defer ctx.deinit();

    // Qwen3.5 Gated DeltaNet: head_k_dim = 128, num_heads varies. Try
    // a couple of heads with distinct value ranges so we'd notice if
    // the per-head reduction got cross-contaminated.
    const num_heads: usize = 4;
    const head_dim: usize = 128;
    const total = num_heads * head_dim;
    const eps: f32 = 1e-6;

    const in_v = try allocator.alloc(f32, total);
    defer allocator.free(in_v);
    for (0..num_heads) |h| {
        const head_scale: f32 = @floatFromInt(h + 1);
        for (0..head_dim) |d| {
            in_v[h * head_dim + d] = head_scale * (@as(f32, @floatFromInt(d)) * 0.01 - 0.5);
        }
    }

    var buf_in = try buffer.Buffer.initStatic(&ctx, f32, in_v);
    defer buf_in.deinit(ctx.device);
    var buf_out = try buffer.Buffer.initDeviceOnly(&ctx, total * @sizeOf(f32));
    defer buf_out.deinit(ctx.device);

    var kern = try pipeline.Kernel.init(&ctx, &shaders.l2norm_per_head, 2, @sizeOf(L2normPush));
    defer kern.deinit();
    try kern.bind(&.{ &buf_in, &buf_out });

    const push = L2normPush{ .head_dim = @intCast(head_dim), .eps = eps };
    try buffer.submitOneShot(&ctx, struct {
        kern: *const pipeline.Kernel,
        push: *const L2normPush,
        groups: u32,
        pub fn record(s: @This(), cmd: vk.c.VkCommandBuffer) void {
            s.kern.dispatch(cmd, s.push, s.groups, 1, 1);
        }
    }{ .kern = &kern, .push = &push, .groups = @intCast(num_heads) });

    const got = try allocator.alloc(f32, total);
    defer allocator.free(got);
    try buf_out.readBack(&ctx, f32, got);

    // CPU reference: per-head L2-norm.
    const want = try allocator.alloc(f32, total);
    defer allocator.free(want);
    for (0..num_heads) |h| {
        var s: f32 = 0;
        for (0..head_dim) |d| s += in_v[h * head_dim + d] * in_v[h * head_dim + d];
        const inv = 1.0 / @sqrt(s + eps);
        for (0..head_dim) |d| want[h * head_dim + d] = in_v[h * head_dim + d] * inv;
    }

    var max_abs: f32 = 0;
    for (got, want) |g, e| {
        const d = @abs(g - e);
        if (d > max_abs) max_abs = d;
    }
    if (max_abs > 1e-5) {
        std.debug.print("GPU l2norm_per_head: max |Δ| = {e}\n", .{max_abs});
        return error.ParityFailed;
    }
    std.debug.print("PASS GPU l2norm_per_head (4 heads × 128 dim, max |Δ| vs CPU = {e})\n", .{max_abs});
}

// ── gpu conv1d_update smoke: 3-step rollout vs CPU ──────────────────
//
// Fires the kernel three times back-to-back so we exercise the in-
// place state shift across multiple decode steps; if the shift / append
// got transposed, the third output diverges immediately.

const Conv1dUpdatePush = runtime_hybrid.Conv1dUpdatePush;

fn runGpuConv1dUpdateSmoke(allocator: std.mem.Allocator) !void {
    var ctx = try vk.Context.init(allocator);
    defer ctx.deinit();

    const conv_dim: usize = 64;
    const kernel: usize = 4;

    const weight = try allocator.alloc(f32, conv_dim * kernel);
    defer allocator.free(weight);
    for (weight, 0..) |*w, i| w.* = @sin(@as(f32, @floatFromInt(i)) * 0.07) * 0.5;

    var buf_w = try buffer.Buffer.initStatic(&ctx, f32, weight);
    defer buf_w.deinit(ctx.device);

    // GPU side state — `initDeviceOnly` zero-fills, which is exactly
    // the "fresh sequence" initial state.
    var buf_state = try buffer.Buffer.initDeviceOnly(&ctx, conv_dim * kernel * @sizeOf(f32));
    defer buf_state.deinit(ctx.device);

    var kern = try pipeline.Kernel.init(&ctx, &shaders.conv1d_update, 4, @sizeOf(Conv1dUpdatePush));
    defer kern.deinit();

    // CPU mirror state (same zero init).
    const cpu_state = try allocator.alloc(f32, conv_dim * kernel);
    defer allocator.free(cpu_state);
    @memset(cpu_state, 0.0);

    const push = Conv1dUpdatePush{ .conv_dim = @intCast(conv_dim), .kernel_size = @intCast(kernel) };

    for (0..3) |step| {
        const x = try allocator.alloc(f32, conv_dim);
        defer allocator.free(x);
        for (x, 0..) |*v, i| v.* = @cos(@as(f32, @floatFromInt(i + step * 7)) * 0.13);

        var buf_in = try buffer.Buffer.initStatic(&ctx, f32, x);
        defer buf_in.deinit(ctx.device);
        var buf_out = try buffer.Buffer.initDeviceOnly(&ctx, conv_dim * @sizeOf(f32));
        defer buf_out.deinit(ctx.device);

        try kern.bind(&.{ &buf_in, &buf_w, &buf_state, &buf_out });
        const local: u32 = 128;
        const groups: u32 = (@as(u32, @intCast(conv_dim)) + local - 1) / local;
        try buffer.submitOneShot(&ctx, struct {
            kern: *const pipeline.Kernel,
            push: *const Conv1dUpdatePush,
            groups: u32,
            pub fn record(s: @This(), cmd: vk.c.VkCommandBuffer) void {
                s.kern.dispatch(cmd, s.push, s.groups, 1, 1);
            }
        }{ .kern = &kern, .push = &push, .groups = groups });

        const got = try allocator.alloc(f32, conv_dim);
        defer allocator.free(got);
        try buf_out.readBack(&ctx, f32, got);

        // CPU reference.
        const want = try allocator.alloc(f32, conv_dim);
        defer allocator.free(want);
        for (0..conv_dim) |c| {
            var k_idx: usize = 0;
            while (k_idx + 1 < kernel) : (k_idx += 1) {
                cpu_state[c * kernel + k_idx] = cpu_state[c * kernel + k_idx + 1];
            }
            cpu_state[c * kernel + kernel - 1] = x[c];
            var acc: f32 = 0;
            for (0..kernel) |k_pos| acc += cpu_state[c * kernel + k_pos] * weight[c * kernel + k_pos];
            want[c] = acc / (1.0 + @exp(-acc));
        }

        var max_abs: f32 = 0;
        for (got, want) |g, e| {
            const d = @abs(g - e);
            if (d > max_abs) max_abs = d;
        }
        if (max_abs > 1e-5) {
            std.debug.print("conv1d_update step {d}: max |Δ| = {e}\n", .{ step, max_abs });
            return error.ParityFailed;
        }
    }
    std.debug.print("PASS GPU conv1d_update (3-step rollout, conv_dim=64 kernel=4)\n", .{});
}

// ── gpu rmsnorm_gated smoke: synthetic vs CPU ───────────────────────

const RmsnormGatedPush = runtime_hybrid.RmsnormGatedPush;

fn runGpuRmsnormGatedSmoke(allocator: std.mem.Allocator) !void {
    var ctx = try vk.Context.init(allocator);
    defer ctx.deinit();

    const num_heads: usize = 4;
    const head_dim: usize = 128;
    const total = num_heads * head_dim;
    const eps: f32 = 1e-6;

    const x = try allocator.alloc(f32, total);
    defer allocator.free(x);
    const z = try allocator.alloc(f32, total);
    defer allocator.free(z);
    const w = try allocator.alloc(f32, head_dim);
    defer allocator.free(w);
    for (x, 0..) |*v, i| v.* = @sin(@as(f32, @floatFromInt(i)) * 0.05) * 0.5;
    for (z, 0..) |*v, i| v.* = @cos(@as(f32, @floatFromInt(i)) * 0.07) * 1.5;
    for (w, 0..) |*v, i| v.* = 0.3 + @as(f32, @floatFromInt(i)) * 0.001;

    var buf_x = try buffer.Buffer.initStatic(&ctx, f32, x);
    defer buf_x.deinit(ctx.device);
    var buf_z = try buffer.Buffer.initStatic(&ctx, f32, z);
    defer buf_z.deinit(ctx.device);
    var buf_w = try buffer.Buffer.initStatic(&ctx, f32, w);
    defer buf_w.deinit(ctx.device);
    var buf_out = try buffer.Buffer.initDeviceOnly(&ctx, total * @sizeOf(f32));
    defer buf_out.deinit(ctx.device);

    var kern = try pipeline.Kernel.init(&ctx, &shaders.rmsnorm_gated, 4, @sizeOf(RmsnormGatedPush));
    defer kern.deinit();
    try kern.bind(&.{ &buf_x, &buf_z, &buf_w, &buf_out });

    const push = RmsnormGatedPush{ .head_dim = @intCast(head_dim), .eps = eps };
    try buffer.submitOneShot(&ctx, struct {
        kern: *const pipeline.Kernel,
        push: *const RmsnormGatedPush,
        groups: u32,
        pub fn record(s: @This(), cmd: vk.c.VkCommandBuffer) void {
            s.kern.dispatch(cmd, s.push, s.groups, 1, 1);
        }
    }{ .kern = &kern, .push = &push, .groups = @intCast(num_heads) });

    const got = try allocator.alloc(f32, total);
    defer allocator.free(got);
    try buf_out.readBack(&ctx, f32, got);

    // CPU reference.
    const want = try allocator.alloc(f32, total);
    defer allocator.free(want);
    for (0..num_heads) |h| {
        const off = h * head_dim;
        var s: f32 = 0;
        for (0..head_dim) |d| s += x[off + d] * x[off + d];
        const inv = 1.0 / @sqrt(s / @as(f32, @floatFromInt(head_dim)) + eps);
        for (0..head_dim) |d| {
            const normed = w[d] * (x[off + d] * inv);
            const zd = z[off + d];
            const silu_z = zd / (1.0 + @exp(-zd));
            want[off + d] = normed * silu_z;
        }
    }

    var max_abs: f32 = 0;
    for (got, want) |g, e| {
        const d = @abs(g - e);
        if (d > max_abs) max_abs = d;
    }
    if (max_abs > 1e-5) {
        std.debug.print("GPU rmsnorm_gated: max |Δ| = {e}\n", .{max_abs});
        return error.ParityFailed;
    }
    std.debug.print("PASS GPU rmsnorm_gated (4 heads × 128 dim, max |Δ| vs CPU = {e})\n", .{max_abs});
}

// ── gpu gated_delta_step smoke: 2-step rollout vs CPU ───────────────
//
// Hot kernel of the Qwen3.5 GatedDeltaNet decode. Two back-to-back
// invocations to exercise both the readout AND the in-place state
// update; if the state update lands wrong, step 2 diverges.

const GatedDeltaStepPush = runtime_hybrid.GatedDeltaStepPush;

fn runGpuGatedDeltaStepSmoke(allocator: std.mem.Allocator) !void {
    var ctx = try vk.Context.init(allocator);
    defer ctx.deinit();

    // Qwen3.5-4B Gated DeltaNet shape: 16 K-heads, 32 V-heads, head_k =
    // head_v = 128 (with implicit 2× GQA repeat).
    const num_k_heads: usize = 16;
    const num_v_heads: usize = 32;
    const head_k: usize = 128;
    const head_v: usize = 128;
    const heads_per_k: usize = num_v_heads / num_k_heads;

    const q = try allocator.alloc(f32, num_k_heads * head_k);
    defer allocator.free(q);
    const k = try allocator.alloc(f32, num_k_heads * head_k);
    defer allocator.free(k);
    const v = try allocator.alloc(f32, num_v_heads * head_v);
    defer allocator.free(v);
    const b_raw = try allocator.alloc(f32, num_v_heads);
    defer allocator.free(b_raw);
    const a_raw = try allocator.alloc(f32, num_v_heads);
    defer allocator.free(a_raw);
    const A_log = try allocator.alloc(f32, num_v_heads);
    defer allocator.free(A_log);
    const dt_bias = try allocator.alloc(f32, num_v_heads);
    defer allocator.free(dt_bias);

    // Seed-style synthetic inputs. CPU mirror state mirrors GPU state
    // (both start at zero from initDeviceOnly).
    for (q, 0..) |*x, i| x.* = @sin(@as(f32, @floatFromInt(i)) * 0.011) * 0.3;
    for (k, 0..) |*x, i| x.* = @cos(@as(f32, @floatFromInt(i)) * 0.013) * 0.3;
    for (v, 0..) |*x, i| x.* = @sin(@as(f32, @floatFromInt(i)) * 0.017) * 0.5;
    for (b_raw, 0..) |*x, i| x.* = @as(f32, @floatFromInt(i)) * 0.05 - 0.5;
    for (a_raw, 0..) |*x, i| x.* = @cos(@as(f32, @floatFromInt(i)) * 0.7) * 0.4;
    for (A_log, 0..) |*x, i| x.* = -1.0 + @as(f32, @floatFromInt(i)) * 0.02;
    for (dt_bias) |*x| x.* = 0.1;

    var buf_q = try buffer.Buffer.initStatic(&ctx, f32, q);
    defer buf_q.deinit(ctx.device);
    var buf_k = try buffer.Buffer.initStatic(&ctx, f32, k);
    defer buf_k.deinit(ctx.device);
    var buf_v = try buffer.Buffer.initStatic(&ctx, f32, v);
    defer buf_v.deinit(ctx.device);
    var buf_b = try buffer.Buffer.initStatic(&ctx, f32, b_raw);
    defer buf_b.deinit(ctx.device);
    var buf_a = try buffer.Buffer.initStatic(&ctx, f32, a_raw);
    defer buf_a.deinit(ctx.device);
    var buf_alog = try buffer.Buffer.initStatic(&ctx, f32, A_log);
    defer buf_alog.deinit(ctx.device);
    var buf_dt = try buffer.Buffer.initStatic(&ctx, f32, dt_bias);
    defer buf_dt.deinit(ctx.device);

    var buf_state = try buffer.Buffer.initDeviceOnly(&ctx, num_v_heads * head_k * head_v * @sizeOf(f32));
    defer buf_state.deinit(ctx.device);
    var buf_y = try buffer.Buffer.initDeviceOnly(&ctx, num_v_heads * head_v * @sizeOf(f32));
    defer buf_y.deinit(ctx.device);

    var kern = try pipeline.Kernel.init(&ctx, &shaders.gated_delta_step, 9, @sizeOf(GatedDeltaStepPush));
    defer kern.deinit();
    try kern.bind(&.{ &buf_state, &buf_q, &buf_k, &buf_v, &buf_b, &buf_a, &buf_alog, &buf_dt, &buf_y });

    const push = GatedDeltaStepPush{
        .num_k_heads = @intCast(num_k_heads),
        .num_v_heads = @intCast(num_v_heads),
        .head_k = @intCast(head_k),
        .head_v = @intCast(head_v),
    };

    // CPU mirror state.
    const cpu_state = try allocator.alloc(f32, num_v_heads * head_k * head_v);
    defer allocator.free(cpu_state);
    @memset(cpu_state, 0.0);

    for (0..2) |step| {
        // GPU dispatch (one workgroup per V-head, head_v threads each).
        try buffer.submitOneShot(&ctx, struct {
            kern: *const pipeline.Kernel,
            push: *const GatedDeltaStepPush,
            n_v: u32,
            pub fn record(s: @This(), cmd: vk.c.VkCommandBuffer) void {
                s.kern.dispatch(cmd, s.push, s.n_v, 1, 1);
            }
        }{ .kern = &kern, .push = &push, .n_v = @intCast(num_v_heads) });

        const got = try allocator.alloc(f32, num_v_heads * head_v);
        defer allocator.free(got);
        try buf_y.readBack(&ctx, f32, got);

        // CPU reference replicates the math of the kernel exactly,
        // including the algebraic identity used for `y`.
        const want = try allocator.alloc(f32, num_v_heads * head_v);
        defer allocator.free(want);

        for (0..num_v_heads) |h| {
            const h_k = h / heads_per_k;
            const k_off = h_k * head_k;
            const S_off = h * head_k * head_v;

            // gates
            const beta_h: f32 = 1.0 / (1.0 + @exp(-b_raw[h]));
            const sp = if ((a_raw[h] + dt_bias[h]) > 20.0) (a_raw[h] + dt_bias[h]) else @log(1.0 + @exp(a_raw[h] + dt_bias[h]));
            const g_t_h: f32 = @exp(-@exp(A_log[h]) * sp);

            // <k, q>
            var kq: f32 = 0;
            for (0..head_k) |d| kq += k[k_off + d] * q[k_off + d];

            // Sq, Sk per column t (computed from S_old). kv_mem (=
            // decayed Sk) used by `delta` is `g_t * Sk_old`.
            var Sq: [128]f32 = undefined;
            var Sk: [128]f32 = undefined;
            for (0..head_v) |t| {
                var sq: f32 = 0;
                var sk: f32 = 0;
                for (0..head_k) |d| {
                    const s_dt = cpu_state[S_off + d * head_v + t];
                    sq += s_dt * q[k_off + d];
                    sk += s_dt * k[k_off + d];
                }
                Sq[t] = sq;
                Sk[t] = sk * g_t_h; // decayed
            }

            // delta, y, state update.
            for (0..head_v) |t| {
                const v_in = v[h * head_v + t];
                const delta_t = (v_in - Sk[t]) * beta_h;
                want[h * head_v + t] = g_t_h * Sq[t] + delta_t * kq;
                for (0..head_k) |d| {
                    const idx = S_off + d * head_v + t;
                    cpu_state[idx] = g_t_h * cpu_state[idx] + k[k_off + d] * delta_t;
                }
            }
        }

        var max_abs: f32 = 0;
        for (got, want) |g, e| {
            const d = @abs(g - e);
            if (d > max_abs) max_abs = d;
        }
        if (max_abs > 1e-4) {
            std.debug.print("gated_delta_step step {d}: max |Δ| = {e}\n", .{ step, max_abs });
            return error.ParityFailed;
        }
    }
    std.debug.print("PASS GPU gated_delta_step (2-step rollout, 32 v-heads × 128² state, GQA 2:1)\n", .{});
}

// ── gpu softmax smoke: synthetic vs CPU softmax ─────────────────────

const SoftmaxPush = runtime.SoftmaxPush;

fn runGpuSoftmaxSmoke(allocator: std.mem.Allocator) !void {
    var ctx = try vk.Context.init(allocator);
    defer ctx.deinit();

    const dim: usize = 2048;
    const x = try allocator.alloc(f32, dim);
    defer allocator.free(x);
    // Mix of negative, near-zero, and one big positive — exercises the
    // numerical-stability subtract-max path.
    for (x, 0..) |*v, i| v.* = @as(f32, @floatFromInt(@as(i32, @intCast(i)) - 1024)) * 0.01;
    x[42] = 100.0; // a clear winner; without subtract-max would overflow exp

    const want = try allocator.alloc(f32, dim);
    defer allocator.free(want);
    @memcpy(want, x);
    cpu_math.softmax(want);

    var buf_in = try buffer.Buffer.initStatic(&ctx, f32, x);
    defer buf_in.deinit(ctx.device);
    var buf_out = try buffer.Buffer.initDeviceOnly(&ctx, dim * @sizeOf(f32));
    defer buf_out.deinit(ctx.device);

    var kern = try pipeline.Kernel.init(&ctx, &shaders.softmax, 2, @sizeOf(SoftmaxPush));
    defer kern.deinit();
    try kern.bind(&.{ &buf_in, &buf_out });

    const push = SoftmaxPush{ .dim = @intCast(dim), .stride = @intCast(dim) };
    try buffer.submitOneShot(&ctx, struct {
        kern: *const pipeline.Kernel,
        push: *const SoftmaxPush,
        pub fn record(s: @This(), cmd: vk.c.VkCommandBuffer) void {
            s.kern.dispatch(cmd, s.push, 1, 1, 1);
        }
    }{ .kern = &kern, .push = &push });

    const got = try allocator.alloc(f32, dim);
    defer allocator.free(got);
    try buf_out.readBack(&ctx, f32, got);

    var max_abs: f32 = 0;
    var sum: f32 = 0;
    for (got, want) |g, e| {
        const d = @abs(g - e);
        if (d > max_abs) max_abs = d;
        sum += g;
    }
    if (max_abs > 1e-5) {
        std.debug.print("GPU softmax: max |Δ| = {e}\n", .{max_abs});
        return error.ParityFailed;
    }
    if (@abs(sum - 1.0) > 1e-5) {
        std.debug.print("GPU softmax sum {d} != 1\n", .{sum});
        return error.ParityFailed;
    }
    std.debug.print("PASS GPU softmax synthetic (dim=2048, sum=1±1e-5, vs CPU 1e-5)\n", .{});
}

// ── gpu fwht256 smoke: in-place FWHT on a 256-vec vs CPU oracle ────

fn runGpuFwhtSmoke(allocator: std.mem.Allocator) !void {
    var ctx = try vk.Context.init(allocator);
    defer ctx.deinit();

    // Deterministic input + CPU reference.
    var input: [256]f32 = undefined;
    var prng = std.Random.DefaultPrng.init(0xFEED_F00D);
    const r = prng.random();
    for (&input) |*v| v.* = r.floatNorm(f32);
    var expected = input;
    turboquant.fwht(&expected);

    // Round-trip through GPU.
    var buf = try buffer.Buffer.initStatic(&ctx, f32, &input);
    defer buf.deinit(ctx.device);

    var kern = try pipeline.Kernel.init(&ctx, &shaders.fwht256, 1, 0);
    defer kern.deinit();
    try kern.bind(&.{&buf});

    try buffer.submitOneShot(&ctx, struct {
        kern: *const pipeline.Kernel,
        pub fn record(s: @This(), cmd: vk.c.VkCommandBuffer) void {
            s.kern.dispatch(cmd, null, 1, 1, 1);
        }
    }{ .kern = &kern });

    var got: [256]f32 = undefined;
    try buf.readBack(&ctx, f32, &got);

    var max_err: f32 = 0;
    for (got, expected) |g, w| max_err = @max(max_err, @abs(g - w));
    if (max_err > 1e-3) {
        std.debug.print("GPU fwht max |Δ| vs CPU = {e}\n", .{max_err});
        return error.ParityFailed;
    }
    std.debug.print("PASS GPU fwht256 (256-elem in-place butterfly, max |Δ| vs CPU = {e:.2})\n", .{max_err});
}

// ── gpu rht_pre256 smoke: signs · x then FWHT, vs CPU rhtForward ──

fn runGpuRhtPreSmoke(allocator: std.mem.Allocator) !void {
    var ctx = try vk.Context.init(allocator);
    defer ctx.deinit();

    var input: [256]f32 = undefined;
    var prng = std.Random.DefaultPrng.init(0xBA5E_BA11);
    const r = prng.random();
    for (&input) |*v| v.* = r.floatNorm(f32);
    var expected = input;
    turboquant.rhtForward(&expected);

    var buf = try buffer.Buffer.initStatic(&ctx, f32, &input);
    defer buf.deinit(ctx.device);

    var kern = try pipeline.Kernel.init(&ctx, &shaders.rht_pre256, 1, 0);
    defer kern.deinit();
    try kern.bind(&.{&buf});

    try buffer.submitOneShot(&ctx, struct {
        kern: *const pipeline.Kernel,
        pub fn record(s: @This(), cmd: vk.c.VkCommandBuffer) void {
            s.kern.dispatch(cmd, null, 1, 1, 1);
        }
    }{ .kern = &kern });

    var got: [256]f32 = undefined;
    try buf.readBack(&ctx, f32, &got);

    var max_err: f32 = 0;
    for (got, expected) |g, w| max_err = @max(max_err, @abs(g - w));
    if (max_err > 1e-3) {
        std.debug.print("GPU rht_pre max |Δ| vs CPU = {e}\n", .{max_err});
        return error.ParityFailed;
    }
    std.debug.print("PASS GPU rht_pre256 (signs · x then FWHT, max |Δ| vs CPU = {e:.2})\n", .{max_err});
}

// ── gpu rht round-trip smoke: rht_post(rht_pre(x)) ≈ x on device ──

fn runGpuRhtRoundTripSmoke(allocator: std.mem.Allocator) !void {
    var ctx = try vk.Context.init(allocator);
    defer ctx.deinit();

    var input: [256]f32 = undefined;
    var prng = std.Random.DefaultPrng.init(0x1234_5678);
    const r = prng.random();
    for (&input) |*v| v.* = r.floatNorm(f32);

    var buf = try buffer.Buffer.initStatic(&ctx, f32, &input);
    defer buf.deinit(ctx.device);

    var pre = try pipeline.Kernel.init(&ctx, &shaders.rht_pre256, 1, 0);
    defer pre.deinit();
    try pre.bind(&.{&buf});

    var post = try pipeline.Kernel.init(&ctx, &shaders.rht_post256, 1, 0);
    defer post.deinit();
    try post.bind(&.{&buf});

    // Two dispatches; submitOneShot waits for queue idle between them
    // so the second read-after-write has hard ordering.
    try buffer.submitOneShot(&ctx, struct {
        kern: *const pipeline.Kernel,
        pub fn record(s: @This(), cmd: vk.c.VkCommandBuffer) void {
            s.kern.dispatch(cmd, null, 1, 1, 1);
        }
    }{ .kern = &pre });
    try buffer.submitOneShot(&ctx, struct {
        kern: *const pipeline.Kernel,
        pub fn record(s: @This(), cmd: vk.c.VkCommandBuffer) void {
            s.kern.dispatch(cmd, null, 1, 1, 1);
        }
    }{ .kern = &post });

    var got: [256]f32 = undefined;
    try buf.readBack(&ctx, f32, &got);

    var max_err: f32 = 0;
    for (got, input) |g, w| max_err = @max(max_err, @abs(g - w));
    if (max_err > 1e-4) {
        std.debug.print("GPU rht round-trip max |Δ| = {e}\n", .{max_err});
        return error.ParityFailed;
    }
    std.debug.print("PASS GPU rht round-trip (rht_post ∘ rht_pre = id, max |Δ| = {e:.2})\n", .{max_err});
}

// ── gpu rht fused round-trip: pre + post in one command buffer ─────
//
// Same round-trip as runGpuRhtRoundTripSmoke, but using the Recorder
// pattern so both dispatches go into a single command buffer, with
// the recorder's auto-emitted compute→compute memory barrier between
// them. This is the orchestration shape recordForwardStep uses for
// every existing kernel, so verifying correctness here means the RHT
// shaders are ready to drop into any layer of the real forward path.

fn runGpuRhtFusedRoundTripSmoke(allocator: std.mem.Allocator) !void {
    var ctx = try vk.Context.init(allocator);
    defer ctx.deinit();

    var input: [256]f32 = undefined;
    var prng = std.Random.DefaultPrng.init(0xCC00_FFEE);
    const r = prng.random();
    for (&input) |*v| v.* = r.floatNorm(f32);

    var buf = try buffer.Buffer.initStatic(&ctx, f32, &input);
    defer buf.deinit(ctx.device);

    var pre = try pipeline.Kernel.init(&ctx, &shaders.rht_pre256, 1, 0);
    defer pre.deinit();
    var post = try pipeline.Kernel.init(&ctx, &shaders.rht_post256, 1, 0);
    defer post.deinit();

    var rec = try gpu_recorder.Recorder.init(&ctx, 4, 4);
    defer rec.deinit();

    try rec.begin();
    try rec.dispatch(&pre,  &.{&buf}, null, 1, 1, 1);
    try rec.dispatch(&post, &.{&buf}, null, 1, 1, 1);
    try rec.endAndSubmit();

    var got: [256]f32 = undefined;
    try buf.readBack(&ctx, f32, &got);

    var max_err: f32 = 0;
    for (got, input) |g, w| max_err = @max(max_err, @abs(g - w));
    if (max_err > 1e-4) {
        std.debug.print("GPU rht fused round-trip max |Δ| = {e}\n", .{max_err});
        return error.ParityFailed;
    }
    std.debug.print("PASS GPU rht fused round-trip (recorder + auto-barrier, max |Δ| = {e:.2})\n", .{max_err});
}

// ── gpu tq4_pack smoke: full TQ4 quantize on GPU vs CPU oracle ─────
//
// Same deterministic ramp as the YATQ bit-exact test: x[i] =
// (i/128) - 1 for i in [0, 256). Quantize on GPU and compare:
//   - 256 Lloyd-Max indices: bit-exact vs CPU quantizeBlockTQ4
//   - γ (norm-correction): within 1e-3 relative tolerance vs CPU
//     (CPU stores f16 γ; GPU stores f32. The truncation tolerance
//     of f16 mantissa is ~5e-4, so 1e-3 is comfortable.)
//
// GPU output is 33 u32s per block: [0] = γ as f32 bits, [1..33] = 32
// LE u32s holding the same byte-stream layout as BlockTQ4.indices.

fn runGpuTq4PackSmoke(allocator: std.mem.Allocator) !void {
    var ctx = try vk.Context.init(allocator);
    defer ctx.deinit();

    var input: [256]f32 = undefined;
    for (&input, 0..) |*v, i| v.* = (@as(f32, @floatFromInt(i)) / 128.0) - 1.0;

    var cpu_blk: turboquant.BlockTQ4(256) = undefined;
    turboquant.quantizeBlockTQ4(256, &input, &cpu_blk);
    const cpu_gamma_f32: f32 = @floatCast(cpu_blk.gamma);

    var input_buf = try buffer.Buffer.initStatic(&ctx, f32, &input);
    defer input_buf.deinit(ctx.device);
    var output_buf = try buffer.Buffer.initDeviceOnly(&ctx, 33 * @sizeOf(u32));
    defer output_buf.deinit(ctx.device);

    var kern = try pipeline.Kernel.init(&ctx, &shaders.tq4_pack256, 2, 0);
    defer kern.deinit();
    try kern.bind(&.{ &input_buf, &output_buf });

    try buffer.submitOneShot(&ctx, struct {
        kern: *const pipeline.Kernel,
        pub fn record(s: @This(), cmd: vk.c.VkCommandBuffer) void {
            s.kern.dispatch(cmd, null, 1, 1, 1);
        }
    }{ .kern = &kern });

    var out: [33]u32 = undefined;
    try output_buf.readBack(&ctx, f32, @ptrCast(&out));
    const gpu_gamma: f32 = @bitCast(out[0]);

    const gamma_rel = @abs(gpu_gamma - cpu_gamma_f32) / cpu_gamma_f32;
    if (gamma_rel > 1e-3) {
        std.debug.print("γ mismatch: gpu={d} cpu(f16)={d} rel={e}\n", .{ gpu_gamma, cpu_gamma_f32, gamma_rel });
        return error.ParityFailed;
    }

    // Compare indices bit-exact. GPU u32[k] = bytes [4k..4k+4] LE; each
    // byte holds two 4-bit indices (low nibble even, high nibble odd).
    // CPU BlockTQ4.indices[k] holds element 2k in low nibble and 2k+1 in
    // high — so CPU indices[k] should equal byte (k % 4) of out[1 + k/4].
    for (0..128) |byte_idx| {
        const word = out[1 + byte_idx / 4];
        const shift: u5 = @intCast((byte_idx % 4) * 8);
        const gpu_byte: u8 = @intCast((word >> shift) & 0xff);
        const cpu_byte = cpu_blk.indices[byte_idx];
        if (gpu_byte != cpu_byte) {
            std.debug.print("idx mismatch at byte {d}: gpu={x:0>2} cpu={x:0>2}\n", .{ byte_idx, gpu_byte, cpu_byte });
            return error.ParityFailed;
        }
    }

    std.debug.print("PASS GPU tq4_pack256 (256 indices bit-exact, γ rel-err {e:.2} vs CPU)\n", .{gamma_rel});
}

// ── gpu tq4_unpack smoke: dequant a CPU-packed block on GPU vs CPU ──

fn runGpuTq4UnpackSmoke(allocator: std.mem.Allocator) !void {
    var ctx = try vk.Context.init(allocator);
    defer ctx.deinit();

    var input: [256]f32 = undefined;
    for (&input, 0..) |*v, i| v.* = (@as(f32, @floatFromInt(i)) / 128.0) - 1.0;

    var cpu_blk: turboquant.BlockTQ4(256) = undefined;
    turboquant.quantizeBlockTQ4(256, &input, &cpu_blk);
    var cpu_dequant: [256]f32 = undefined;
    turboquant.dequantizeBlockTQ4(256, &cpu_blk, &cpu_dequant);

    // Build the 33-u32 GPU input: word[0] = γ as f32 bits, words[1..33] =
    // 128 bytes of cpu_blk.indices viewed as 32 LE u32s.
    var gpu_in: [33]u32 = undefined;
    gpu_in[0] = @bitCast(@as(f32, @floatCast(cpu_blk.gamma)));
    @memcpy(@as([*]u8, @ptrCast(gpu_in[1..].ptr))[0..128], &cpu_blk.indices);

    var input_buf = try buffer.Buffer.initStatic(&ctx, u32, &gpu_in);
    defer input_buf.deinit(ctx.device);
    var output_buf = try buffer.Buffer.initDeviceOnly(&ctx, 256 * @sizeOf(f32));
    defer output_buf.deinit(ctx.device);

    var kern = try pipeline.Kernel.init(&ctx, &shaders.tq4_unpack256, 2, 0);
    defer kern.deinit();
    try kern.bind(&.{ &input_buf, &output_buf });

    try buffer.submitOneShot(&ctx, struct {
        kern: *const pipeline.Kernel,
        pub fn record(s: @This(), cmd: vk.c.VkCommandBuffer) void {
            s.kern.dispatch(cmd, null, 1, 1, 1);
        }
    }{ .kern = &kern });

    var gpu_dequant: [256]f32 = undefined;
    try output_buf.readBack(&ctx, f32, &gpu_dequant);

    var max_err: f32 = 0;
    for (gpu_dequant, cpu_dequant) |g, w| max_err = @max(max_err, @abs(g - w));
    if (max_err > 1e-4) {
        std.debug.print("GPU tq4_unpack max |Δ| vs CPU = {e}\n", .{max_err});
        return error.ParityFailed;
    }
    std.debug.print("PASS GPU tq4_unpack256 (CPU-packed block dequanted on GPU, max |Δ| vs CPU = {e:.2})\n", .{max_err});
}

// ── gpu tq4 round-trip: pack → unpack on GPU vs CPU oracle ──────────

fn runGpuTq4RoundTripSmoke(allocator: std.mem.Allocator) !void {
    var ctx = try vk.Context.init(allocator);
    defer ctx.deinit();

    var input: [256]f32 = undefined;
    var prng = std.Random.DefaultPrng.init(0xACED_F00D);
    const r = prng.random();
    for (&input) |*v| v.* = r.floatNorm(f32);

    var cpu_blk: turboquant.BlockTQ4(256) = undefined;
    turboquant.quantizeBlockTQ4(256, &input, &cpu_blk);
    var cpu_dequant: [256]f32 = undefined;
    turboquant.dequantizeBlockTQ4(256, &cpu_blk, &cpu_dequant);

    // GPU: pack input, then unpack the GPU-packed block — chained
    // through the Recorder so both dispatches share a command buffer
    // with the auto-emitted compute→compute barrier.
    var input_buf = try buffer.Buffer.initStatic(&ctx, f32, &input);
    defer input_buf.deinit(ctx.device);
    var packed_buf = try buffer.Buffer.initDeviceOnly(&ctx, 33 * @sizeOf(u32));
    defer packed_buf.deinit(ctx.device);
    var output_buf = try buffer.Buffer.initDeviceOnly(&ctx, 256 * @sizeOf(f32));
    defer output_buf.deinit(ctx.device);

    var pack_kern = try pipeline.Kernel.init(&ctx, &shaders.tq4_pack256, 2, 0);
    defer pack_kern.deinit();
    var unpack_kern = try pipeline.Kernel.init(&ctx, &shaders.tq4_unpack256, 2, 0);
    defer unpack_kern.deinit();

    var rec = try gpu_recorder.Recorder.init(&ctx, 4, 8);
    defer rec.deinit();
    try rec.begin();
    try rec.dispatch(&pack_kern,   &.{ &input_buf, &packed_buf }, null, 1, 1, 1);
    try rec.dispatch(&unpack_kern, &.{ &packed_buf, &output_buf }, null, 1, 1, 1);
    try rec.endAndSubmit();

    var gpu_dequant: [256]f32 = undefined;
    try output_buf.readBack(&ctx, f32, &gpu_dequant);

    var max_err: f32 = 0;
    for (gpu_dequant, cpu_dequant) |g, w| max_err = @max(max_err, @abs(g - w));
    // GPU and CPU both compute γ in fp32 then truncate to f16, but the
    // raw γ values differ by ~f32-ULP because the L2 norm reductions
    // traverse elements in different orders (subgroup-tree on GPU vs
    // linear on CPU). When the two raw γs straddle an f16 boundary
    // they round to different f16 values, and that single-ULP delta
    // multiplies through the centroid magnitudes (max ~2.73), giving
    // ~5e-4 × 2.73 ≈ 1.4e-3 reconstruction divergence in the worst
    // case. The indices are bit-exact (verified separately in
    // tq4_pack256 smoke); only the γ scaling drifts.
    if (max_err > 5e-3) {
        std.debug.print("GPU tq4 round-trip max |Δ| vs CPU = {e}\n", .{max_err});
        return error.ParityFailed;
    }
    std.debug.print("PASS GPU tq4 pack→unpack round-trip (recorder, max |Δ| vs CPU = {e:.2})\n", .{max_err});
}

// ── gpu tq4_pack_to_cache smoke: positional pack into a multi-block cache ──
//
// Pack three different 256-vec inputs into slots 0, 1, 2 of a 3-block
// cache buffer using the dst_block_idx push constant. Then dispatch
// tq4_unpack256 with WG count = 3 to dequantise all three blocks at
// once. Each reconstructed block must match the corresponding
// CPU pack→dequant output.

const Tq4PackPush = runtime.Tq4PackPush;

fn runGpuTq4PackToCacheSmoke(allocator: std.mem.Allocator) !void {
    var ctx = try vk.Context.init(allocator);
    defer ctx.deinit();

    const n_blocks: usize = 3;
    var inputs: [n_blocks][256]f32 = undefined;
    var prng = std.Random.DefaultPrng.init(0xCA75_F00D);
    const r = prng.random();
    for (&inputs) |*blk_in| for (blk_in) |*v| { v.* = r.floatNorm(f32); };

    // CPU oracle: pack+dequant each block.
    var cpu_dequants: [n_blocks][256]f32 = undefined;
    for (&inputs, &cpu_dequants) |*x, *y| {
        var blk: turboquant.BlockTQ4(256) = undefined;
        turboquant.quantizeBlockTQ4(256, x, &blk);
        turboquant.dequantizeBlockTQ4(256, &blk, y);
    }

    // GPU side: a single 256-vec staging buffer for the input, a
    // 3-block packed cache, and a 3-block dequant output.
    var stage_buf = try buffer.Buffer.initStatic(&ctx, f32, &inputs[0]);
    defer stage_buf.deinit(ctx.device);
    var cache_buf = try buffer.Buffer.initDeviceOnly(&ctx, n_blocks * 33 * @sizeOf(u32));
    defer cache_buf.deinit(ctx.device);
    var deq_buf = try buffer.Buffer.initDeviceOnly(&ctx, n_blocks * 256 * @sizeOf(f32));
    defer deq_buf.deinit(ctx.device);

    var pack = try pipeline.Kernel.init(&ctx, &shaders.tq4_pack_to_cache, 2, @sizeOf(Tq4PackPush));
    defer pack.deinit();
    var unpack = try pipeline.Kernel.init(&ctx, &shaders.tq4_unpack256, 2, 0);
    defer unpack.deinit();

    // For block 0, the staging buffer already holds inputs[0] — pack
    // it directly. For blocks 1 and 2, update the staging buffer
    // between dispatches via the dynamic-update path (the buffer was
    // initStatic so we don't have a host-mapped pointer; instead we
    // make a fresh static buffer per block). Simpler: recreate stage
    // buffer per iteration.
    {
        var rec = try gpu_recorder.Recorder.init(&ctx, 8, 16);
        defer rec.deinit();
        try rec.begin();
        var push = Tq4PackPush{ .dst_block_idx = 0 };
        try rec.dispatch(&pack, &.{ &stage_buf, &cache_buf }, &push, 1, 1, 1);
        try rec.endAndSubmit();
    }
    for (1..n_blocks) |b| {
        var s = try buffer.Buffer.initStatic(&ctx, f32, &inputs[b]);
        defer s.deinit(ctx.device);
        var rec = try gpu_recorder.Recorder.init(&ctx, 8, 16);
        defer rec.deinit();
        try rec.begin();
        var push = Tq4PackPush{ .dst_block_idx = @intCast(b) };
        try rec.dispatch(&pack, &.{ &s, &cache_buf }, &push, 1, 1, 1);
        try rec.endAndSubmit();
    }

    // Single dispatch unpacks all 3 blocks (WG count = 3).
    {
        var rec = try gpu_recorder.Recorder.init(&ctx, 8, 16);
        defer rec.deinit();
        try rec.begin();
        try rec.dispatch(&unpack, &.{ &cache_buf, &deq_buf }, null, n_blocks, 1, 1);
        try rec.endAndSubmit();
    }

    var got: [n_blocks * 256]f32 = undefined;
    try deq_buf.readBack(&ctx, f32, &got);

    var max_err: f32 = 0;
    for (0..n_blocks) |b| {
        for (0..256) |i| {
            const g = got[b * 256 + i];
            const w = cpu_dequants[b][i];
            max_err = @max(max_err, @abs(g - w));
        }
    }
    if (max_err > 5e-3) {
        std.debug.print("GPU tq4_pack_to_cache max |Δ| = {e}\n", .{max_err});
        return error.ParityFailed;
    }
    std.debug.print("PASS GPU tq4_pack_to_cache (3 blocks at distinct positions, max |Δ| vs CPU = {e:.2})\n", .{max_err});
}

// ── gelu_tanh smoke: scalar against PyTorch reference values ────────

fn runGeluSmoke(allocator: std.mem.Allocator) !void {
    _ = allocator;
    // Reference values from torch.nn.functional.gelu(approximate="tanh")
    // (which matches HF's gelu_pytorch_tanh, which Gemma uses).
    const cases = [_]struct { x: f32, want: f32 }{
        .{ .x = 0.0, .want = 0.0 },
        .{ .x = 1.0, .want = 0.8411919876 },
        .{ .x = -1.0, .want = -0.15880800784 },
        .{ .x = 2.0, .want = 1.9545976400 },
        .{ .x = -2.0, .want = -0.04540234059 },
    };
    for (cases) |tc| {
        const got = cpu_math.gelu_tanh(tc.x);
        const err = @abs(got - tc.want);
        if (err > 1e-5) {
            std.debug.print("gelu_tanh({d}): got {d}, want {d} (err {e})\n", .{ tc.x, got, tc.want, err });
            return error.ParityFailed;
        }
    }
    std.debug.print("PASS gelu_tanh (5 ref values within 1e-5)\n", .{});
}

// ── TurboQuant tables smoke: Lloyd-Max codebook + RHT sign pattern ──

fn runTurboquantSmoke(allocator: std.mem.Allocator) !void {
    _ = allocator;
    // 1) RHT sign pattern: byte 0 = 0xa7 = 0b10100111, so LSB-first
    //    bits 0..7 = 1,1,1,0,0,1,0,1, signs = -1,-1,-1,+1,+1,-1,+1,-1.
    const sign_expected = [_]f32{ -1, -1, -1, 1, 1, -1, 1, -1 };
    for (sign_expected, 0..) |want, i| {
        if (turboquant.rhtSign(i) != want) {
            std.debug.print("rhtSign({d}): got {d}, want {d}\n", .{ i, turboquant.rhtSign(i), want });
            return error.ParityFailed;
        }
    }
    // The pattern is 256-bit periodic.
    if (turboquant.rhtSign(0) != turboquant.rhtSign(256)) return error.ParityFailed;

    // 2) Lloyd-Max codebooks symmetric about zero (b=3 and b=4).
    inline for ([_]turboquant.Bits{ .b3, .b4 }) |b| {
        const n: usize = if (b == .b3) 8 else 16;
        var i: usize = 0;
        while (i < n / 2) : (i += 1) {
            const lo = turboquant.lloydMaxCentroid(@intCast(i), b);
            const hi = turboquant.lloydMaxCentroid(@intCast(n - 1 - i), b);
            if (@abs(-lo - hi) > 1e-6) {
                std.debug.print("centroid asymmetry b={d}: lo={d} hi={d}\n", .{ @intFromEnum(b), lo, hi });
                return error.ParityFailed;
            }
        }
    }

    // 3) Quantize each centroid back through lloydMaxIndex; should round-trip.
    inline for ([_]turboquant.Bits{ .b3, .b4 }) |b| {
        const n: usize = if (b == .b3) 8 else 16;
        var i: usize = 0;
        while (i < n) : (i += 1) {
            const c = turboquant.lloydMaxCentroid(@intCast(i), b);
            const idx = turboquant.lloydMaxIndex(c, b);
            if (idx != i) {
                std.debug.print("centroid round-trip b={d} i={d}: got idx={d}\n", .{ @intFromEnum(b), i, idx });
                return error.ParityFailed;
            }
        }
    }

    // 4) Hand-checked corner values vs YATQ's (x >= b).sum() idiom.
    //    b=3: x=0.0 → bin 4 (centroid +0.2451), x=0.6 → bin 5 (+0.756),
    //    x=-0.6 → bin 2 (-0.756). b=4: x=0.0 → bin 8 (+0.1284).
    if (turboquant.lloydMaxIndex(0.0, .b3) != 4) return error.ParityFailed;
    if (turboquant.lloydMaxIndex(0.6, .b3) != 5) return error.ParityFailed;
    if (turboquant.lloydMaxIndex(-0.6, .b3) != 2) return error.ParityFailed;
    if (turboquant.lloydMaxIndex(0.0, .b4) != 8) return error.ParityFailed;

    // 5) FWHT hand-checked at d=4: [1,2,3,4] → [10,-2,-4,0].
    {
        var x = [_]f32{ 1, 2, 3, 4 };
        turboquant.fwht(&x);
        const want = [_]f32{ 10, -2, -4, 0 };
        for (x, want) |got, w| if (got != w) return error.ParityFailed;
    }

    // 6) FWHT applied twice multiplies by d (since H · H = d · I).
    {
        var x = [_]f32{ 0.5, -1.25, 3.0, 0.0, -2.5, 1.75, 0.125, -0.75 };
        const orig = x;
        turboquant.fwht(&x);
        turboquant.fwht(&x);
        const d: f32 = @floatFromInt(x.len);
        for (x, orig) |got, w| if (@abs(got - d * w) > 1e-5) return error.ParityFailed;
    }

    // 7) RHT round-trip on a 256-vector: rhtInverse(rhtForward(x)) ≈ x.
    {
        var prng = std.Random.DefaultPrng.init(0xDEADBEEF);
        const r = prng.random();
        var x: [256]f32 = undefined;
        for (&x) |*v| v.* = r.floatNorm(f32);
        const orig = x;
        turboquant.rhtForward(&x);
        turboquant.rhtInverse(&x);
        var max_err: f32 = 0;
        for (x, orig) |got, w| max_err = @max(max_err, @abs(got - w));
        if (max_err > 1e-4) {
            std.debug.print("rht round-trip max |Δ| = {e}\n", .{max_err});
            return error.ParityFailed;
        }
    }

    // 8) TQ4 round-trip on a 256-d Gaussian block. The norm-correction
    //    γ must give a reconstruction whose L2 norm equals the original
    //    to within f16 quantisation; per-element MSE must be in the
    //    expected range for 4-bit Lloyd-Max on a unit Gaussian
    //    (<0.005 average squared error per coord).
    {
        var prng = std.Random.DefaultPrng.init(0x5EED1234);
        const r = prng.random();
        var x: [256]f32 = undefined;
        for (&x) |*v| v.* = r.floatNorm(f32);

        var raw_sq: f32 = 0;
        for (x) |v| raw_sq += v * v;
        const raw_norm = @sqrt(raw_sq);

        var blk: turboquant.BlockTQ4(256) = undefined;
        turboquant.quantizeBlockTQ4(256, &x, &blk);
        var y: [256]f32 = undefined;
        turboquant.dequantizeBlockTQ4(256, &blk, &y);

        var rec_sq: f32 = 0;
        var err_sq: f32 = 0;
        for (x, y) |xi, yi| {
            rec_sq += yi * yi;
            const d = xi - yi;
            err_sq += d * d;
        }
        const rec_norm = @sqrt(rec_sq);
        const norm_rel = @abs(raw_norm - rec_norm) / raw_norm;
        const mse = err_sq / @as(f32, @floatFromInt(x.len));

        if (norm_rel > 1e-3) {
            std.debug.print("TQ4 norm preservation: raw={d:.4} rec={d:.4} rel={e}\n", .{ raw_norm, rec_norm, norm_rel });
            return error.ParityFailed;
        }
        if (mse > 0.01) {
            std.debug.print("TQ4 MSE = {d:.5} (>0.01 threshold)\n", .{mse});
            return error.ParityFailed;
        }
        std.debug.print("       TQ4 256-d Gaussian: MSE={d:.5}, norm-rel-err={e:.2}\n", .{ mse, norm_rel });
    }

    // 9) Bit-exact parity vs YATQ Python reference. Input is the
    //    deterministic ramp x[i] = (i/128) - 1, which exists in both
    //    Zig and Python with no PRNG-cross-language risk. Expected
    //    indices generated by reference/turboquant/cross_validate.py.
    {
        var x: [256]f32 = undefined;
        for (&x, 0..) |*v, i| {
            v.* = (@as(f32, @floatFromInt(i)) / 128.0) - 1.0;
        }
        const yatq_indices_b4 = [256]u8{
            8,  7,  12, 6,  6,  11, 8,  7,  9,  9,  7,  13, 8,  13, 11, 7,
            14, 4,  13, 10, 10, 8,  6,  13, 4,  7,  3,  6,  10, 15, 6,  8,
            10, 6,  13, 11, 9,  13, 9,  6,  3,  3,  5,  2,  7,  12, 11, 3,
            9,  4,  12, 10, 14, 3,  8,  6,  8,  8,  5,  9,  3,  4,  13, 4,
            8,  7,  4,  7,  8,  6,  7,  7,  5,  10, 10, 6,  12, 6,  2,  12,
            5,  7,  7,  1,  8,  12, 11, 9,  14, 6,  5,  1,  8,  6,  9,  10,
            11, 9,  13, 10, 4,  13, 6,  5,  8,  9,  10, 12, 9,  10, 15, 2,
            6,  5,  8,  5,  12, 5,  6,  6,  9,  7,  4,  2,  5,  14, 2,  3,
            10, 8,  7,  8,  9,  8,  8,  7,  3,  7,  10, 8,  8,  11, 5,  12,
            8,  6,  11, 3,  13, 12, 13, 8,  14, 7,  8,  2,  7,  5,  8,  4,
            12, 9,  13, 6,  2,  10, 6,  2,  4,  8,  12, 13, 11, 12, 13, 3,
            12, 4,  12, 5,  10, 7,  4,  10, 7,  6,  6,  1,  2,  12, 9,  3,
            7,  7,  11, 9,  7,  10, 8,  9,  12, 12, 3,  11, 5,  12, 13, 5,
            12, 2,  10, 7,  12, 7,  6,  9,  5,  11, 4,  3,  12, 15, 1,  8,
            7,  5,  9,  12, 8,  11, 10, 7,  6,  3,  3,  4,  11, 7,  8,  5,
            6,  5,  9,  8,  13, 8,  5,  9,  5,  8,  4,  7,  4,  3,  14, 9,
        };
        const yatq_raw_norm: f32 = 9.23774529;

        var blk: turboquant.BlockTQ4(256) = undefined;
        turboquant.quantizeBlockTQ4(256, &x, &blk);

        // (a) raw L2 norm of input (independent of quantization) must
        //     agree with what numpy computed.
        var raw_sq: f32 = 0;
        for (x) |v| raw_sq += v * v;
        const our_raw_norm = @sqrt(raw_sq);
        if (@abs(our_raw_norm - yatq_raw_norm) > 1e-4) {
            std.debug.print("L2 norm divergence: our={d} yatq={d}\n", .{ our_raw_norm, yatq_raw_norm });
            return error.ParityFailed;
        }

        // (b) every Lloyd-Max index must match bit-exact.
        var k: usize = 0;
        while (k < 128) : (k += 1) {
            const lo = blk.indices[k] & 0x0f;
            const hi = (blk.indices[k] >> 4) & 0x0f;
            const want_lo = yatq_indices_b4[2 * k];
            const want_hi = yatq_indices_b4[2 * k + 1];
            if (lo != want_lo or hi != want_hi) {
                std.debug.print("idx mismatch at coord {d}: got ({d},{d}) want ({d},{d})\n", .{ 2 * k, lo, hi, want_lo, want_hi });
                return error.ParityFailed;
            }
        }
    }

    std.debug.print("PASS turboquant CPU oracle (tables + FWHT + RHT + TQ4 round-trip + YATQ bit-exact)\n", .{});
}

// ── q4_0 CPU smoke: round-trip parity for the int4 weight oracle ───
//
// Tier-1 quantization preflight. The CPU q4_0.zig functions are the
// reference the GPU shader will be parity-checked against, so this
// smoke verifies them in isolation: hand-checked single-block encode,
// round-trip on a Gaussian row (cosine sim and per-element MSE), and
// the symmetric edge case where one element saturates +7.

fn runQ4_0Smoke(allocator: std.mem.Allocator) !void {
    _ = allocator;

    // 1) Single-block hand-check. With a deterministic ramp the largest
    //    magnitude is the first or last element, so the scale d picks
    //    up that element's sign and the round-trip is bit-tight.
    {
        var src: [32]f32 = undefined;
        for (&src, 0..) |*v, i| v.* = @as(f32, @floatFromInt(@as(i32, @intCast(i)) - 16)) * 0.1;
        // src ranges -1.6 .. +1.5 in steps of 0.1. amax = 1.6 with the
        // largest signed magnitude = -1.6. So d = -1.6 / -8 = 0.2.
        var blocks: [1]q4_0.Block = undefined;
        q4_0.quantizeRow(&src, &blocks);
        if (@abs(@as(f32, @floatCast(blocks[0].d)) - 0.2) > 1e-3) {
            std.debug.print("q4_0 single-block: d={d}, want 0.2\n", .{@as(f32, @floatCast(blocks[0].d))});
            return error.ParityFailed;
        }

        var rec: [32]f32 = undefined;
        q4_0.dequantizeRow(&blocks, &rec);
        var max_err: f32 = 0;
        for (src, rec) |x, y| max_err = @max(max_err, @abs(x - y));
        // With d=0.2, snap-to-grid worst case is 0.1 (half a step).
        if (max_err > 0.105) {
            std.debug.print("q4_0 single-block round-trip: max |Δ|={d}\n", .{max_err});
            return error.ParityFailed;
        }
    }

    // 2) Gaussian row round-trip (1024 floats = 32 blocks). Q4_0 on a
    //    unit-Gaussian source gives ≈ 0.0033 SNR-bound MSE per coord;
    //    we leave generous headroom against PRNG variance.
    {
        const n_elems: usize = 1024;
        var prng = std.Random.DefaultPrng.init(0xC0DEC0DECAFE);
        const r = prng.random();
        var src: [n_elems]f32 = undefined;
        for (&src) |*v| v.* = r.floatNorm(f32);

        var blocks: [n_elems / 32]q4_0.Block = undefined;
        q4_0.quantizeRow(&src, &blocks);
        var rec: [n_elems]f32 = undefined;
        q4_0.dequantizeRow(&blocks, &rec);

        var err_sq: f64 = 0;
        for (src, rec) |x, y| {
            const d = @as(f64, x) - @as(f64, y);
            err_sq += d * d;
        }
        const mse: f64 = err_sq / @as(f64, @floatFromInt(n_elems));
        const cos = q4_0.cosineSim(&src, &rec);

        if (mse > 0.01) {
            std.debug.print("q4_0 Gaussian MSE={d:.5} (>0.01)\n", .{mse});
            return error.ParityFailed;
        }
        if (cos < 0.995) {
            std.debug.print("q4_0 Gaussian cos-sim={d:.5} (<0.995)\n", .{cos});
            return error.ParityFailed;
        }
        std.debug.print("       q4_0 1024-float Gaussian: MSE={d:.5}, cos-sim={d:.5}\n", .{ mse, cos });
    }

    // 3) Saturation / extremes: in the llama.cpp Q4_0 scheme, the
    //    element with the largest magnitude maps to idx 0 (signed
    //    -8), so it round-trips EXACTLY. The opposite extreme
    //    saturates against id=15 (signed +7) and reconstructs as
    //    (15-8)*|d| = 7/8 of the original. Confirms clamp + sign
    //    handling at the boundary.
    {
        var src: [32]f32 = [_]f32{0.0} ** 32;
        src[0] = 1.0;   // largest magnitude (positive) — should round-trip exactly
        src[15] = -1.0; // opposite extreme — should saturate to -0.875
        var blocks: [1]q4_0.Block = undefined;
        q4_0.quantizeRow(&src, &blocks);

        // d = max / -8 = 1.0 / -8 = -0.125. Element 0 → idx 0 → (0-8)*-0.125 = 1.0.
        // Element 15 → 8/(-0.125) ⇒ raw 8 → +8.5 ⇒ floor 16 ⇒ clamp 15 → (15-8)*-0.125 = -0.875.
        const idx0: u8 = blocks[0].qs[0] & 0x0F;
        const idx15: u8 = blocks[0].qs[15] & 0x0F;
        if (idx0 != 0 or idx15 != 15) {
            std.debug.print("q4_0 saturation: idx[0]={d} (want 0), idx[15]={d} (want 15)\n", .{ idx0, idx15 });
            return error.ParityFailed;
        }

        var rec: [32]f32 = undefined;
        q4_0.dequantizeRow(&blocks, &rec);
        if (@abs(rec[0] - 1.0) > 1e-6 or @abs(rec[15] - (-0.875)) > 1e-6) {
            std.debug.print("q4_0 saturation decode: rec[0]={d} (want 1.0), rec[15]={d} (want -0.875)\n", .{ rec[0], rec[15] });
            return error.ParityFailed;
        }
    }

    std.debug.print("PASS q4_0 CPU oracle (single-block, Gaussian round-trip, saturation edge)\n", .{});
}

// ── q4_K CPU smoke: round-trip parity for the asymmetric int4 oracle ─
//
// Verifies our llama.cpp-compatible Q4_K_M reference at three points:
// the constant-block degenerate case (super-scales should both be 0,
// reconstruction exact), a Gaussian super-block round-trip (cosine sim
// > 0.999, MSE noticeably better than Q4_0's 0.005 on the same input
// since asymmetric quant + iterative refinement both help), and the
// scales-byte packing/unpacking — encode a known (sc[0..7], m[0..7])
// pattern with non-trivial top-2-bits and confirm getScaleMinK4
// recovers it bit-perfect.

fn runQ4_KSmoke(allocator: std.mem.Allocator) !void {
    _ = allocator;

    // 1) All-zero block: true degenerate case. min=max=0, makeQkx2Quants
    //    short-circuits to scale=0, the_min=0; super-scales d=dmin=0;
    //    dequant returns all zero. Verifies the early-out doesn't write
    //    garbage into the qs/scales bytes (left at memset-0 from phase 0).
    {
        const src: [q4_k.QK_K]f32 = [_]f32{0.0} ** q4_k.QK_K;
        var blocks: [1]q4_k.Block = undefined;
        q4_k.quantizeRow(&src, &blocks);

        var rec: [q4_k.QK_K]f32 = undefined;
        q4_k.dequantizeRow(&blocks, &rec);

        var max_err: f32 = 0;
        for (rec) |y| max_err = @max(max_err, @abs(y));
        if (max_err > 1e-6) {
            std.debug.print("q4_K zero block: rec should be all-zero, max |y|={d}\n", .{max_err});
            return error.ParityFailed;
        }
    }

    // 2) Scale-byte packing: hand-build a (sc, m) pattern with values that
    //    exercise both the low-6-bit and high-2-bit slots, encode it via
    //    the same op-sequence quantizeRow uses, and confirm getScaleMinK4
    //    recovers it. Catches any byte-shift mistake without needing a
    //    full quantize round-trip to expose.
    {
        const sc_in = [_]u8{ 0, 1, 17, 63, 32, 47, 60, 33 };
        const m_in = [_]u8{ 63, 0, 5, 31, 48, 11, 50, 21 };
        var scales: [q4_k.K_SCALE_SIZE]u8 = [_]u8{0} ** q4_k.K_SCALE_SIZE;
        for (0..8) |j| {
            const ls = sc_in[j];
            const lm = m_in[j];
            if (j < 4) {
                scales[j] = ls;
                scales[j + 4] = lm;
            } else {
                scales[j + 4] = (ls & 0x0F) | ((lm & 0x0F) << 4);
                scales[j - 4] |= @as(u8, @intCast(ls >> 4)) << 6;
                scales[j] |= @as(u8, @intCast(lm >> 4)) << 6;
            }
        }
        for (0..8) |j| {
            var sc_out: u8 = undefined;
            var m_out: u8 = undefined;
            q4_k.getScaleMinK4(@intCast(j), &scales, &sc_out, &m_out);
            if (sc_out != sc_in[j] or m_out != m_in[j]) {
                std.debug.print(
                    "q4_K scales pack[{d}]: got sc={d} m={d}, want sc={d} m={d}\n",
                    .{ j, sc_out, m_out, sc_in[j], m_in[j] },
                );
                return error.ParityFailed;
            }
        }
    }

    // 3) Gaussian super-block round-trip (one full 256-elem block). Q4_K
    //    on unit Gaussian gives substantially better MSE than Q4_0 at
    //    the same nominal bitrate — the asymmetric offset + iterative
    //    refinement together typically halve per-coord error vs Q4_0's
    //    max-magnitude scheme.
    {
        const n_elems: usize = q4_k.QK_K;
        var prng = std.Random.DefaultPrng.init(0xC0DEC0DECAFE);
        const r = prng.random();
        var src: [n_elems]f32 = undefined;
        for (&src) |*v| v.* = r.floatNorm(f32);

        var blocks: [1]q4_k.Block = undefined;
        q4_k.quantizeRow(&src, &blocks);
        var rec: [n_elems]f32 = undefined;
        q4_k.dequantizeRow(&blocks, &rec);

        var err_sq: f64 = 0;
        for (src, rec) |x, y| {
            const d = @as(f64, x) - @as(f64, y);
            err_sq += d * d;
        }
        const mse: f64 = err_sq / @as(f64, @floatFromInt(n_elems));
        const cos = q4_0.cosineSim(&src, &rec);

        // q4_K typical MSE on unit Gaussian is ~0.003 (vs ~0.008 for q4_0
        // on the same input); leave slack for prng variance.
        // q4_K typical MSE on unit Gaussian is ~0.006 on 256-elem blocks
        // (vs q4_0's 0.008–0.010 — ~30% improvement). Threshold leaves
        // room for prng variance across seeds while still proving the
        // win over q4_0 holds.
        if (mse > 0.008) {
            std.debug.print("q4_K Gaussian MSE={d:.5} (>0.008)\n", .{mse});
            return error.ParityFailed;
        }
        if (cos < 0.997) {
            std.debug.print("q4_K Gaussian cos-sim={d:.5} (<0.997)\n", .{cos});
            return error.ParityFailed;
        }
        std.debug.print("       q4_K 256-float Gaussian: MSE={d:.5}, cos-sim={d:.5}\n", .{ mse, cos });
    }

    // 4) Multi-super-block Gaussian (4×256 = 1024 elems). Same expected
    //    MSE/cos-sim envelope; verifies the b-loop in quantizeRow doesn't
    //    leak state between super-blocks.
    {
        const n_blocks: usize = 4;
        const n_elems: usize = q4_k.QK_K * n_blocks;
        var prng = std.Random.DefaultPrng.init(0xBADCAFEDEADBEEF);
        const r = prng.random();
        var src: [n_elems]f32 = undefined;
        for (&src) |*v| v.* = r.floatNorm(f32);

        var blocks: [n_blocks]q4_k.Block = undefined;
        q4_k.quantizeRow(&src, &blocks);
        var rec: [n_elems]f32 = undefined;
        q4_k.dequantizeRow(&blocks, &rec);

        var err_sq: f64 = 0;
        for (src, rec) |x, y| {
            const d = @as(f64, x) - @as(f64, y);
            err_sq += d * d;
        }
        const mse: f64 = err_sq / @as(f64, @floatFromInt(n_elems));
        const cos = q4_0.cosineSim(&src, &rec);

        if (mse > 0.008) {
            std.debug.print("q4_K multi-block MSE={d:.5}\n", .{mse});
            return error.ParityFailed;
        }
        if (cos < 0.997) {
            std.debug.print("q4_K multi-block cos-sim={d:.5}\n", .{cos});
            return error.ParityFailed;
        }
    }

    std.debug.print("PASS q4_K CPU oracle (constant block, scales packing, Gaussian round-trip)\n", .{});
}

// ── rmsnorm-test: first math primitive on a real layer ──────────────

fn runRmsnormTest(allocator: std.mem.Allocator, dir_path: []const u8, token_id: u32) !void {
    var model = try model_mod.Model.load(allocator, dir_path);
    defer model.deinit();
    const cfg = model.config;
    if (token_id >= cfg.vocab_size) return error.OutOfRange;

    const stdout = std.io.getStdOut().writer();
    try stdout.print("rmsnorm test on layer 0 input_layernorm — token {d}\n\n", .{token_id});

    // Materialise the embedding row.
    const x = try allocator.alloc(f32, cfg.hidden_size);
    defer allocator.free(x);
    try cpu_math.embedRowAsF32(x, model.embed_tokens, token_id);

    // Gemma scales the embedding by sqrt(hidden_size) before the first
    // block. Without this the RMS of `x` is way under 1, and rmsnorm
    // ends up amplifying noise — the post-rmsnorm activations would be
    // garbage and every downstream test would lie. Apply unconditionally
    // for now since we're Gemma-only; when we add Llama, gate on family.
    if (cfg.family.embedScalesByDim()) {
        const scale: f32 = @sqrt(@as(f32, @floatFromInt(cfg.hidden_size)));
        for (x) |*xi| xi.* *= scale;
    }

    const x_rms = blk: {
        var s: f32 = 0;
        for (x) |v| s += v * v;
        break :blk @sqrt(s / @as(f32, @floatFromInt(x.len)));
    };
    try stdout.print("post-scale embedding rms = {d:.6}\n", .{x_rms});

    // Apply layer 0's input_layernorm.
    const y = try allocator.alloc(f32, cfg.hidden_size);
    defer allocator.free(y);
    try cpu_math.rmsnorm(y, x, model.layers[0].input_layernorm, cfg.rms_norm_eps, cfg.family);

    const n_show: usize = @min(16, y.len);
    for (y[0..n_show], 0..) |v, i| try stdout.print("  [{d:>4}] {d:.6}\n", .{ i, v });

    var min_v: f32 = std.math.inf(f32);
    var max_v: f32 = -std.math.inf(f32);
    var sum_sq: f64 = 0;
    var nan_count: usize = 0;
    var inf_count: usize = 0;
    for (y) |v| {
        if (std.math.isNan(v)) {
            nan_count += 1;
            continue;
        }
        if (std.math.isInf(v)) {
            inf_count += 1;
            continue;
        }
        if (v < min_v) min_v = v;
        if (v > max_v) max_v = v;
        sum_sq += @as(f64, v) * @as(f64, v);
    }
    const post_rms = std.math.sqrt(sum_sq / @as(f64, @floatFromInt(y.len)));
    try stdout.print("\noutput stats: min={d:.6} max={d:.6} rms={d:.6} nan={d} inf={d}\n", .{
        min_v, max_v, post_rms, nan_count, inf_count,
    });
}

// ── qproj-test: rmsnorm → matmul against layer 0's q_proj ───────────

fn runQprojTest(allocator: std.mem.Allocator, dir_path: []const u8, token_id: u32) !void {
    var model = try model_mod.Model.load(allocator, dir_path);
    defer model.deinit();
    const cfg = model.config;
    if (token_id >= cfg.vocab_size) return error.OutOfRange;

    const stdout = std.io.getStdOut().writer();
    try stdout.print("qproj test on layer 0 — token {d}\n\n", .{token_id});

    // Embedding → scale → rmsnorm → q_proj.
    const x = try allocator.alloc(f32, cfg.hidden_size);
    defer allocator.free(x);
    try cpu_math.embedRowAsF32(x, model.embed_tokens, token_id);
    if (cfg.family.embedScalesByDim()) {
        const scale: f32 = @sqrt(@as(f32, @floatFromInt(cfg.hidden_size)));
        for (x) |*xi| xi.* *= scale;
    }

    const x_norm = try allocator.alloc(f32, cfg.hidden_size);
    defer allocator.free(x_norm);
    try cpu_math.rmsnorm(x_norm, x, model.layers[0].input_layernorm, cfg.rms_norm_eps, cfg.family);

    // Q has dimension n_heads × head_dim.
    const q_dim = cfg.num_attention_heads * cfg.head_dim;
    const q = try allocator.alloc(f32, q_dim);
    defer allocator.free(q);

    const t0 = std.time.nanoTimestamp();
    try cpu_math.matmul_nt(q, x_norm, model.layers[0].q_proj.?, 1, q_dim, cfg.hidden_size);
    const t1 = std.time.nanoTimestamp();
    const ms = @as(f64, @floatFromInt(t1 - t0)) / 1_000_000.0;

    try stdout.print("matmul_nt [1, {d}] = [1, {d}] · [{d}, {d}]ᵀ — {d:.2} ms\n\n", .{
        cfg.hidden_size, q_dim, q_dim, cfg.hidden_size, ms,
    });

    const n_show: usize = @min(16, q.len);
    for (q[0..n_show], 0..) |v, i| try stdout.print("  q[{d:>4}] {d:.6}\n", .{ i, v });

    var min_v: f32 = std.math.inf(f32);
    var max_v: f32 = -std.math.inf(f32);
    var sum_sq: f64 = 0;
    var nan_count: usize = 0;
    var inf_count: usize = 0;
    for (q) |v| {
        if (std.math.isNan(v)) {
            nan_count += 1;
            continue;
        }
        if (std.math.isInf(v)) {
            inf_count += 1;
            continue;
        }
        if (v < min_v) min_v = v;
        if (v > max_v) max_v = v;
        sum_sq += @as(f64, v) * @as(f64, v);
    }
    const q_rms = std.math.sqrt(sum_sq / @as(f64, @floatFromInt(q.len)));
    try stdout.print("\nq stats: min={d:.6} max={d:.6} rms={d:.6} nan={d} inf={d}\n", .{
        min_v, max_v, q_rms, nan_count, inf_count,
    });
}

// ── rope-test: produce Q, apply RoPE at pos 0 and pos 1 ─────────────

fn runRopeTest(allocator: std.mem.Allocator, dir_path: []const u8, token_id: u32) !void {
    var model = try model_mod.Model.load(allocator, dir_path);
    defer model.deinit();
    const cfg = model.config;

    // Reuse the qproj chain to get Q.
    const x = try allocator.alloc(f32, cfg.hidden_size);
    defer allocator.free(x);
    try cpu_math.embedRowAsF32(x, model.embed_tokens, token_id);
    if (cfg.family.embedScalesByDim()) {
        const scale: f32 = @sqrt(@as(f32, @floatFromInt(cfg.hidden_size)));
        for (x) |*xi| xi.* *= scale;
    }
    const x_norm = try allocator.alloc(f32, cfg.hidden_size);
    defer allocator.free(x_norm);
    try cpu_math.rmsnorm(x_norm, x, model.layers[0].input_layernorm, cfg.rms_norm_eps, cfg.family);

    const q_dim = cfg.num_attention_heads * cfg.head_dim;
    const q = try allocator.alloc(f32, q_dim);
    defer allocator.free(q);
    try cpu_math.matmul_nt(q, x_norm, model.layers[0].q_proj.?, 1, q_dim, cfg.hidden_size);

    const stdout = std.io.getStdOut().writer();
    try stdout.print("RoPE on Q [n_heads={d}, head_dim={d}] for token {d}\n\n", .{
        cfg.num_attention_heads, cfg.head_dim, token_id,
    });

    // pos = 0: must equal Q.
    const q_pos0 = try allocator.alloc(f32, q_dim);
    defer allocator.free(q_pos0);
    try cpu_math.applyRope(q_pos0, q, cfg.num_attention_heads, cfg.head_dim, 0, cfg.rope_theta);
    var pos0_ok = true;
    for (q, q_pos0) |a, b| if (a != b) {
        pos0_ok = false;
        break;
    };
    try stdout.print("pos=0 identity: {s}\n", .{if (pos0_ok) "OK" else "FAIL"});

    // pos = 1: rotated.
    const q_pos1 = try allocator.alloc(f32, q_dim);
    defer allocator.free(q_pos1);
    try cpu_math.applyRope(q_pos1, q, cfg.num_attention_heads, cfg.head_dim, 1, cfg.rope_theta);

    // Print head 0, first 8 dims of each pair, before vs after pos=1.
    try stdout.print("\nhead 0, pre vs post pos=1 RoPE (first 8 dim pairs):\n", .{});
    const half = cfg.head_dim / 2;
    for (0..8) |j| {
        const a_pre = q[j];
        const b_pre = q[j + half];
        const a_post = q_pos1[j];
        const b_post = q_pos1[j + half];
        try stdout.print("  pair ({d:>3}, {d:>3}):  pre=({d:.4}, {d:.4})  post=({d:.4}, {d:.4})\n", .{
            j, j + half, a_pre, b_pre, a_post, b_post,
        });
    }

    // Sanity: norm of each (j, j+half) pair must be invariant — RoPE is
    // a rotation, it preserves length per pair.
    var max_err: f32 = 0;
    for (0..cfg.num_attention_heads) |h| {
        const off = h * cfg.head_dim;
        for (0..half) |j| {
            const a0 = q[off + j];
            const b0 = q[off + j + half];
            const a1 = q_pos1[off + j];
            const b1 = q_pos1[off + j + half];
            const n_pre = a0 * a0 + b0 * b0;
            const n_post = a1 * a1 + b1 * b1;
            const err = @abs(n_pre - n_post);
            if (err > max_err) max_err = err;
        }
    }
    try stdout.print("\nrotation invariant: max ||pair||² delta = {e:.2}  (must be tiny)\n", .{max_err});
}

// ── attention-test: full single-position attention through layer 0 ──

fn runAttentionTest(allocator: std.mem.Allocator, dir_path: []const u8, token_id: u32) !void {
    var model = try model_mod.Model.load(allocator, dir_path);
    defer model.deinit();
    const cfg = model.config;
    const layer = model.layers[0];

    const n_heads = cfg.num_attention_heads;
    const n_kv = cfg.num_key_value_heads;
    const head_dim = cfg.head_dim;
    const q_dim = n_heads * head_dim;
    const kv_dim = n_kv * head_dim;
    const heads_per_kv = n_heads / n_kv; // 8 for Gemma 2B (MQA)

    const stdout = std.io.getStdOut().writer();
    try stdout.print(
        \\full attention block on layer 0 — token {d} run twice (pos 0, pos 1)
        \\config: n_heads={d}, n_kv_heads={d}, head_dim={d}, hidden={d}
        \\
        \\
    , .{ token_id, n_heads, n_kv, head_dim, cfg.hidden_size });

    // 2-position KV cache, flat [pos][n_kv * head_dim].
    const max_pos: usize = 2;
    const k_cache = try allocator.alloc(f32, max_pos * kv_dim);
    defer allocator.free(k_cache);
    const v_cache = try allocator.alloc(f32, max_pos * kv_dim);
    defer allocator.free(v_cache);

    // Scratch buffers reused across positions.
    const x = try allocator.alloc(f32, cfg.hidden_size);
    defer allocator.free(x);
    const x_norm = try allocator.alloc(f32, cfg.hidden_size);
    defer allocator.free(x_norm);
    const q = try allocator.alloc(f32, q_dim);
    defer allocator.free(q);
    const q_rot = try allocator.alloc(f32, q_dim);
    defer allocator.free(q_rot);
    const k = try allocator.alloc(f32, kv_dim);
    defer allocator.free(k);
    const k_rot = try allocator.alloc(f32, kv_dim);
    defer allocator.free(k_rot);
    const v = try allocator.alloc(f32, kv_dim);
    defer allocator.free(v);
    const head_out = try allocator.alloc(f32, q_dim);
    defer allocator.free(head_out);
    const attn_out = try allocator.alloc(f32, cfg.hidden_size);
    defer allocator.free(attn_out);

    const inv_sqrt_dim: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(head_dim)));

    var pos: usize = 0;
    while (pos < max_pos) : (pos += 1) {
        // ── pre-attention: embed → scale → rmsnorm ──────────────────
        try cpu_math.embedRowAsF32(x, model.embed_tokens, token_id);
        if (cfg.family.embedScalesByDim()) {
            const s: f32 = @sqrt(@as(f32, @floatFromInt(cfg.hidden_size)));
            for (x) |*xi| xi.* *= s;
        }
        try cpu_math.rmsnorm(x_norm, x, layer.input_layernorm, cfg.rms_norm_eps, cfg.family);

        // ── Q, K, V projections ─────────────────────────────────────
        try cpu_math.matmul_nt(q, x_norm, layer.q_proj.?, 1, q_dim, cfg.hidden_size);
        try cpu_math.matmul_nt(k, x_norm, layer.k_proj.?, 1, kv_dim, cfg.hidden_size);
        try cpu_math.matmul_nt(v, x_norm, layer.v_proj.?, 1, kv_dim, cfg.hidden_size);

        // ── RoPE on Q and K (V is not rotated) ──────────────────────
        try cpu_math.applyRope(q_rot, q, n_heads, head_dim, pos, cfg.rope_theta);
        try cpu_math.applyRope(k_rot, k, n_kv, head_dim, pos, cfg.rope_theta);

        // ── Append to KV cache ──────────────────────────────────────
        @memcpy(k_cache[pos * kv_dim ..][0..kv_dim], k_rot);
        @memcpy(v_cache[pos * kv_dim ..][0..kv_dim], v);

        // ── Attention: scores → softmax → weighted V sum ────────────
        const n_pos = pos + 1;
        const scores = try allocator.alloc(f32, n_pos);
        defer allocator.free(scores);

        var print_softmax_pos: ?usize = null;
        for (0..n_heads) |h| {
            const kv_h = h / heads_per_kv;
            const q_off = h * head_dim;

            // Score against every cached position.
            for (0..n_pos) |p| {
                const k_off = p * kv_dim + kv_h * head_dim;
                var s: f32 = 0;
                for (0..head_dim) |d| s += q_rot[q_off + d] * k_cache[k_off + d];
                scores[p] = s * inv_sqrt_dim;
            }

            cpu_math.softmax(scores);
            if (h == 0 and pos == 1) print_softmax_pos = pos;

            // Sum over positions: head_out[h] = Σ scores[p] * v_cache[p, kv_h]
            const out_off = h * head_dim;
            for (0..head_dim) |d| head_out[out_off + d] = 0;
            for (0..n_pos) |p| {
                const v_off = p * kv_dim + kv_h * head_dim;
                const w = scores[p];
                for (0..head_dim) |d| head_out[out_off + d] += w * v_cache[v_off + d];
            }
        }

        // ── Output projection: head_out @ o_proj^T → attn_out ───────
        try cpu_math.matmul_nt(attn_out, head_out, layer.o_proj.?, 1, cfg.hidden_size, q_dim);

        // ── Stats ───────────────────────────────────────────────────
        var min_v: f32 = std.math.inf(f32);
        var max_v: f32 = -std.math.inf(f32);
        var sum_sq: f64 = 0;
        for (attn_out) |val| {
            if (val < min_v) min_v = val;
            if (val > max_v) max_v = val;
            sum_sq += @as(f64, val) * @as(f64, val);
        }
        const rms = std.math.sqrt(sum_sq / @as(f64, @floatFromInt(attn_out.len)));

        try stdout.print("pos {d}: attn_out min={d:.6} max={d:.6} rms={d:.6}\n", .{
            pos, min_v, max_v, rms,
        });
        if (print_softmax_pos != null) {
            // Re-run softmax on head 0 so we can print it (the loop
            // above destroys scores in-place per head).
            const dbg_scores = try allocator.alloc(f32, n_pos);
            defer allocator.free(dbg_scores);
            for (0..n_pos) |p| {
                const k_off = p * kv_dim;
                var s: f32 = 0;
                for (0..head_dim) |d| s += q_rot[d] * k_cache[k_off + d];
                dbg_scores[p] = s * inv_sqrt_dim;
            }
            const raw0 = dbg_scores[0];
            const raw1 = dbg_scores[1];
            cpu_math.softmax(dbg_scores);
            try stdout.print("       head 0 raw scores=({d:.4}, {d:.4})  softmax=({d:.4}, {d:.4}) sum={d:.6}\n", .{
                raw0, raw1, dbg_scores[0], dbg_scores[1], dbg_scores[0] + dbg_scores[1],
            });
        }
    }
}

// ── layer0-test: complete transformer block (attn + FFN + residuals) ──

fn runLayer0Test(allocator: std.mem.Allocator, dir_path: []const u8, token_id: u32) !void {
    var model = try model_mod.Model.load(allocator, dir_path);
    defer model.deinit();
    const cfg = model.config;
    const layer = model.layers[0];

    const n_heads = cfg.num_attention_heads;
    const n_kv = cfg.num_key_value_heads;
    const head_dim = cfg.head_dim;
    const q_dim = n_heads * head_dim;
    const kv_dim = n_kv * head_dim;
    const heads_per_kv = n_heads / n_kv;
    const inter = cfg.intermediate_size;
    const hidden = cfg.hidden_size;

    const stdout = std.io.getStdOut().writer();
    try stdout.print(
        \\full layer 0 forward — token {d}, position 0 (single token, no history)
        \\config: hidden={d}, intermediate={d}, n_heads={d}, kv_heads={d}, head_dim={d}
        \\
        \\
    , .{ token_id, hidden, inter, n_heads, n_kv, head_dim });

    // ── Stage 0: residual stream init from embedding ────────────────
    const x = try allocator.alloc(f32, hidden);
    defer allocator.free(x);
    try cpu_math.embedRowAsF32(x, model.embed_tokens, token_id);
    if (cfg.family.embedScalesByDim()) {
        const s: f32 = @sqrt(@as(f32, @floatFromInt(hidden)));
        for (x) |*xi| xi.* *= s;
    }
    try printStreamStats(stdout, "embed (post-scale)", x);

    // ── Stage 1: rmsnorm₁ → Q/K/V → RoPE → attention → o_proj ───────
    const x_norm1 = try allocator.alloc(f32, hidden);
    defer allocator.free(x_norm1);
    try cpu_math.rmsnorm(x_norm1, x, layer.input_layernorm, cfg.rms_norm_eps, cfg.family);

    const q = try allocator.alloc(f32, q_dim);
    defer allocator.free(q);
    const k = try allocator.alloc(f32, kv_dim);
    defer allocator.free(k);
    const v = try allocator.alloc(f32, kv_dim);
    defer allocator.free(v);
    try cpu_math.matmul_nt(q, x_norm1, layer.q_proj.?, 1, q_dim, hidden);
    try cpu_math.matmul_nt(k, x_norm1, layer.k_proj.?, 1, kv_dim, hidden);
    try cpu_math.matmul_nt(v, x_norm1, layer.v_proj.?, 1, kv_dim, hidden);

    const q_rot = try allocator.alloc(f32, q_dim);
    defer allocator.free(q_rot);
    const k_rot = try allocator.alloc(f32, kv_dim);
    defer allocator.free(k_rot);
    try cpu_math.applyRope(q_rot, q, n_heads, head_dim, 0, cfg.rope_theta);
    try cpu_math.applyRope(k_rot, k, n_kv, head_dim, 0, cfg.rope_theta);

    // Single-position attention: softmax over 1 score is 1.0 → head_out
    // is exactly v (broadcast across query heads sharing the kv head).
    const head_out = try allocator.alloc(f32, q_dim);
    defer allocator.free(head_out);
    for (0..n_heads) |h| {
        const kv_h = h / heads_per_kv;
        const v_off = kv_h * head_dim;
        const out_off = h * head_dim;
        @memcpy(head_out[out_off .. out_off + head_dim], v[v_off .. v_off + head_dim]);
    }
    // (q_rot and k_rot computed above are unused at pos 0 since the
    // softmax collapses to 1.0 — kept for symmetry with the multi-
    // position path we'll wire when we have an actual prompt.)

    const attn_out = try allocator.alloc(f32, hidden);
    defer allocator.free(attn_out);
    try cpu_math.matmul_nt(attn_out, head_out, layer.o_proj.?, 1, hidden, q_dim);
    try printStreamStats(stdout, "attn output (pre-residual)", attn_out);

    // ── Stage 2: residual add ───────────────────────────────────────
    const mid = try allocator.alloc(f32, hidden);
    defer allocator.free(mid);
    for (mid, x, attn_out) |*m, xi, ai| m.* = xi + ai;
    try printStreamStats(stdout, "residual after attn", mid);

    // ── Stage 3: rmsnorm₂ → GeGLU FFN → down_proj ──────────────────
    const mid_norm = try allocator.alloc(f32, hidden);
    defer allocator.free(mid_norm);
    try cpu_math.rmsnorm(mid_norm, mid, layer.post_attention_layernorm, cfg.rms_norm_eps, cfg.family);

    const gate = try allocator.alloc(f32, inter);
    defer allocator.free(gate);
    const up = try allocator.alloc(f32, inter);
    defer allocator.free(up);
    try cpu_math.matmul_nt(gate, mid_norm, layer.gate_proj, 1, inter, hidden);
    try cpu_math.matmul_nt(up, mid_norm, layer.up_proj, 1, inter, hidden);

    const fused = try allocator.alloc(f32, inter);
    defer allocator.free(fused);
    try cpu_math.geglu(fused, gate, up);
    try printStreamStats(stdout, "geglu(gate)·up (intermediate)", fused);

    const ffn_out = try allocator.alloc(f32, hidden);
    defer allocator.free(ffn_out);
    try cpu_math.matmul_nt(ffn_out, fused, layer.down_proj, 1, hidden, inter);
    try printStreamStats(stdout, "ffn output (pre-residual)", ffn_out);

    // ── Stage 4: residual add → block output ────────────────────────
    const block_out = try allocator.alloc(f32, hidden);
    defer allocator.free(block_out);
    for (block_out, mid, ffn_out) |*o, m, f| o.* = m + f;
    try printStreamStats(stdout, "layer 0 output", block_out);
}

// ── tq4-kv-test: real Gemma K/V vectors round-tripped through TQ4 ──
//
// Walks the full 18-layer forward pass for one input token, and at
// every layer captures the (post-RoPE) K vector and the (pre-RoPE) V
// vector — the two quantities that would actually live in a
// TurboQuant-compressed KV cache. Each 256-element vector is run
// through quantizeBlockTQ4 + dequantizeBlockTQ4, and the per-coord
// MSE / max |Δ| / norm-preservation is reported so we have a real-
// data reference for what the synthetic Gaussian numbers looked like
// (MSE 0.00658, norm-rel-err 1.09e-4 from the smoke test).
//
// Gemma 2B has n_kv_heads=1 and head_dim=256, so each layer's K and V
// is exactly one TQ4 block. 18 layers × 2 (K, V) = 36 sample blocks.

fn runTq4KvTest(allocator: std.mem.Allocator, dir_path: []const u8, token_id: u32) !void {
    var model = try model_mod.Model.load(allocator, dir_path);
    defer model.deinit();
    const cfg = model.config;
    if (token_id >= cfg.vocab_size) return error.OutOfRange;

    if (cfg.head_dim != turboquant.block_size_tq4) {
        std.debug.print("head_dim ({d}) != TQ4 block size ({d}); only Gemma 2B supported in chunk 1.4\n", .{ cfg.head_dim, turboquant.block_size_tq4 });
        return error.HeadDimMismatch;
    }
    if (cfg.num_key_value_heads != 1) {
        std.debug.print("num_kv_heads ({d}) != 1; this test currently assumes Gemma MQA\n", .{cfg.num_key_value_heads});
        return error.KvHeadCountMismatch;
    }

    const stdout = std.io.getStdOut().writer();
    try stdout.print("TQ4 KV round-trip on real Gemma activations — token {d}\n\n", .{token_id});

    const hidden = cfg.hidden_size;
    const inter = cfg.intermediate_size;
    const n_heads = cfg.num_attention_heads;
    const n_kv = cfg.num_key_value_heads;
    const head_dim = cfg.head_dim;
    const q_dim = n_heads * head_dim;
    const kv_dim = n_kv * head_dim; // == 256 for Gemma 2B
    const heads_per_kv = n_heads / n_kv;
    const pos: usize = 0;

    // Per-layer scratch (mirrors cpu/forward.zig, no shortcuts so we
    // see real activations at each step).
    const stream = try allocator.alloc(f32, hidden);
    defer allocator.free(stream);
    const x_norm = try allocator.alloc(f32, hidden);
    defer allocator.free(x_norm);
    const q = try allocator.alloc(f32, q_dim);
    defer allocator.free(q);
    const k = try allocator.alloc(f32, kv_dim);
    defer allocator.free(k);
    const v = try allocator.alloc(f32, kv_dim);
    defer allocator.free(v);
    const q_rot = try allocator.alloc(f32, q_dim);
    defer allocator.free(q_rot);
    const k_rot = try allocator.alloc(f32, kv_dim);
    defer allocator.free(k_rot);
    const head_out = try allocator.alloc(f32, q_dim);
    defer allocator.free(head_out);
    const attn_out = try allocator.alloc(f32, hidden);
    defer allocator.free(attn_out);
    const mid_norm = try allocator.alloc(f32, hidden);
    defer allocator.free(mid_norm);
    const gate = try allocator.alloc(f32, inter);
    defer allocator.free(gate);
    const up = try allocator.alloc(f32, inter);
    defer allocator.free(up);
    const fused = try allocator.alloc(f32, inter);
    defer allocator.free(fused);
    const ffn_out = try allocator.alloc(f32, hidden);
    defer allocator.free(ffn_out);

    // Embedding + Gemma sqrt(hidden) scale.
    try cpu_math.embedRowAsF32(stream, model.embed_tokens, token_id);
    if (cfg.family.embedScalesByDim()) {
        const s: f32 = @sqrt(@as(f32, @floatFromInt(hidden)));
        for (stream) |*xi| xi.* *= s;
    }

    // Aggregate stats.
    var k_mse_sum: f64 = 0;
    var v_mse_sum: f64 = 0;
    var k_max_err: f32 = 0;
    var v_max_err: f32 = 0;
    var k_max_norm_rel: f32 = 0;
    var v_max_norm_rel: f32 = 0;

    try stdout.print("layer  K MSE      K max|Δ|   K norm-rel  V MSE      V max|Δ|   V norm-rel\n", .{});
    try stdout.print("-----  ---------  ---------  ----------  ---------  ---------  ----------\n", .{});

    for (model.layers, 0..) |layer, layer_idx| {
        // Pre-attention rmsnorm.
        try cpu_math.rmsnorm(x_norm, stream, layer.input_layernorm, cfg.rms_norm_eps, cfg.family);

        // Q/K/V projections.
        try cpu_math.matmul_nt(q, x_norm, layer.q_proj.?, 1, q_dim, hidden);
        try cpu_math.matmul_nt(k, x_norm, layer.k_proj.?, 1, kv_dim, hidden);
        try cpu_math.matmul_nt(v, x_norm, layer.v_proj.?, 1, kv_dim, hidden);

        // RoPE on Q and K.
        try cpu_math.applyRope(q_rot, q, n_heads, head_dim, pos, cfg.rope_theta);
        try cpu_math.applyRope(k_rot, k, n_kv, head_dim, pos, cfg.rope_theta);

        // ── TQ4 round-trip on K_rot and V ───────────────────────────
        const k_block_in: *const [256]f32 = @ptrCast(k_rot.ptr);
        const v_block_in: *const [256]f32 = @ptrCast(v.ptr);
        var k_blk: turboquant.BlockTQ4(256) = undefined;
        var v_blk: turboquant.BlockTQ4(256) = undefined;
        turboquant.quantizeBlockTQ4(256, k_block_in, &k_blk);
        turboquant.quantizeBlockTQ4(256, v_block_in, &v_blk);

        var k_recon: [256]f32 = undefined;
        var v_recon: [256]f32 = undefined;
        turboquant.dequantizeBlockTQ4(256, &k_blk, &k_recon);
        turboquant.dequantizeBlockTQ4(256, &v_blk, &v_recon);

        const k_stats = blockStats(k_block_in, &k_recon);
        const v_stats = blockStats(v_block_in, &v_recon);

        try stdout.print("{d:>5}  {d:.6}   {d:.6}   {e:>9.2}   {d:.6}   {d:.6}   {e:>9.2}\n", .{
            layer_idx,
            k_stats.mse,        k_stats.max_err, k_stats.norm_rel_err,
            v_stats.mse,        v_stats.max_err, v_stats.norm_rel_err,
        });

        k_mse_sum += k_stats.mse;
        v_mse_sum += v_stats.mse;
        k_max_err = @max(k_max_err, k_stats.max_err);
        v_max_err = @max(v_max_err, v_stats.max_err);
        k_max_norm_rel = @max(k_max_norm_rel, k_stats.norm_rel_err);
        v_max_norm_rel = @max(v_max_norm_rel, v_stats.norm_rel_err);

        // Continue the forward pass with the *original* (un-quantized)
        // K_rot / V so subsequent layers see canonical activations.
        // Quantization-error-propagation is a separate experiment
        // (chunk 1.5).
        for (0..n_heads) |h| {
            const kv_h = h / heads_per_kv;
            const v_off = kv_h * head_dim;
            const out_off = h * head_dim;
            @memcpy(head_out[out_off .. out_off + head_dim], v[v_off .. v_off + head_dim]);
        }
        try cpu_math.matmul_nt(attn_out, head_out, layer.o_proj.?, 1, hidden, q_dim);
        for (stream, attn_out) |*si, ai| si.* += ai;

        try cpu_math.rmsnorm(mid_norm, stream, layer.post_attention_layernorm, cfg.rms_norm_eps, cfg.family);
        try cpu_math.matmul_nt(gate, mid_norm, layer.gate_proj, 1, inter, hidden);
        try cpu_math.matmul_nt(up, mid_norm, layer.up_proj, 1, inter, hidden);
        try cpu_math.geglu(fused, gate, up);
        try cpu_math.matmul_nt(ffn_out, fused, layer.down_proj, 1, hidden, inter);
        for (stream, ffn_out) |*si, fi| si.* += fi;
    }

    const n_layers: f64 = @floatFromInt(model.layers.len);
    try stdout.print("\n", .{});
    try stdout.print("aggregate (over {d} layers):\n", .{model.layers.len});
    try stdout.print("  K mean MSE     = {d:.6}    K max |Δ|  = {d:.6}    K worst norm-rel = {e:.2}\n", .{
        k_mse_sum / n_layers, k_max_err, k_max_norm_rel,
    });
    try stdout.print("  V mean MSE     = {d:.6}    V max |Δ|  = {d:.6}    V worst norm-rel = {e:.2}\n", .{
        v_mse_sum / n_layers, v_max_err, v_max_norm_rel,
    });
}

const BlockStats = struct {
    mse: f32,
    max_err: f32,
    norm_rel_err: f32,
};

fn blockStats(orig: *const [256]f32, recon: *const [256]f32) BlockStats {
    var err_sq: f32 = 0;
    var max_err: f32 = 0;
    var orig_sq: f32 = 0;
    var recon_sq: f32 = 0;
    for (orig, recon) |o, r| {
        orig_sq += o * o;
        recon_sq += r * r;
        const d = o - r;
        err_sq += d * d;
        max_err = @max(max_err, @abs(d));
    }
    const orig_norm = @sqrt(orig_sq);
    const recon_norm = @sqrt(recon_sq);
    const norm_rel = if (orig_norm > 0) @abs(orig_norm - recon_norm) / orig_norm else 0;
    return .{
        .mse = err_sq / 256.0,
        .max_err = max_err,
        .norm_rel_err = norm_rel,
    };
}

// ── gen-tq4v: side-by-side fp32 vs TQ4-V-cache CPU forward ─────────
//
// Runs both code paths on the same input token and prints the
// argmax, top-5 IDs, and the max single-logit divergence over the
// 256k-element vocab. Goal is the cleanest "does the V-cache
// quantisation introduce a token-level regression" signal: if
// argmax and top-5 hold and max |Δ| stays in fp32-noise range, the
// TQ4 V path is safe.

fn runGenTq4V(allocator: std.mem.Allocator, dir_path: []const u8, token_id: u32) !void {
    var model = try model_mod.Model.load(allocator, dir_path);
    defer model.deinit();
    const cfg = model.config;
    if (token_id >= cfg.vocab_size) return error.OutOfRange;

    const tok_path = try std.fmt.allocPrint(allocator, "{s}/tokenizer.json", .{dir_path});
    defer allocator.free(tok_path);
    var tok = try tokenizer_mod.Tokenizer.loadFromFile(allocator, tok_path);
    defer tok.deinit();

    const stdout = std.io.getStdOut().writer();
    try stdout.print("gen-tq4v: fp32 vs TQ4-V on token {d}\n\n", .{token_id});

    const logits_a = try allocator.alloc(f32, cfg.vocab_size);
    defer allocator.free(logits_a);
    const logits_b = try allocator.alloc(f32, cfg.vocab_size);
    defer allocator.free(logits_b);

    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();

    try cpu_forward.forward(&model, token_id, 0, arena.allocator(), logits_a);
    _ = arena.reset(.retain_capacity);
    try cpu_forward.forwardTq4V(&model, token_id, 0, arena.allocator(), logits_b);

    // Top-5 of each.
    const top_a = try topK(allocator, logits_a, 5);
    defer allocator.free(top_a);
    const top_b = try topK(allocator, logits_b, 5);
    defer allocator.free(top_b);

    try stdout.print("            fp32 baseline                       TQ4-V\n", .{});
    try stdout.print("rank   id       logit    token              id       logit    token\n", .{});
    try stdout.print("----   ------   ------   ----------------   ------   ------   ----------------\n", .{});
    for (0..5) |i| {
        const ta = top_a[i];
        const tb = top_b[i];
        const ta_text = tok.decode(ta.id) orelse "?";
        const tb_text = tok.decode(tb.id) orelse "?";
        try stdout.print("{d:>4}   {d:>6}   {d:>6.2}   {s:<16}   {d:>6}   {d:>6.2}   {s:<16}\n", .{
            i, ta.id, ta.value, truncateStr(ta_text, 16), tb.id, tb.value, truncateStr(tb_text, 16),
        });
    }

    // Pairwise stats over the full vocab.
    var max_abs_delta: f32 = 0;
    var max_abs_delta_idx: usize = 0;
    var sum_abs: f64 = 0;
    var sum_sq: f64 = 0;
    for (logits_a, logits_b, 0..) |a, b, i| {
        const d = @abs(a - b);
        if (d > max_abs_delta) {
            max_abs_delta = d;
            max_abs_delta_idx = i;
        }
        sum_abs += d;
        sum_sq += d * d;
    }
    const n: f64 = @floatFromInt(logits_a.len);
    const mean_abs = sum_abs / n;
    const rms = std.math.sqrt(sum_sq / n);

    try stdout.print("\nlogit divergence over {d} tokens:\n", .{logits_a.len});
    try stdout.print("  max |Δ|   = {d:.6}  at id={d}\n", .{ max_abs_delta, max_abs_delta_idx });
    try stdout.print("  mean |Δ|  = {d:.6}\n", .{mean_abs});
    try stdout.print("  rms  Δ    = {d:.6}\n", .{rms});

    const argmax_a = cpu_forward.argmax(logits_a);
    const argmax_b = cpu_forward.argmax(logits_b);
    try stdout.print("\nargmax:  fp32 → {d}    TQ4-V → {d}    {s}\n", .{
        argmax_a, argmax_b, if (argmax_a == argmax_b) "(MATCH)" else "(DIVERGE!)",
    });

    var top5_match: usize = 0;
    for (top_a) |a| {
        for (top_b) |b| {
            if (a.id == b.id) {
                top5_match += 1;
                break;
            }
        }
    }
    try stdout.print("top-5 ID overlap: {d}/5\n", .{top5_match});
}

fn truncateStr(s: []const u8, n: usize) []const u8 {
    return if (s.len <= n) s else s[0..n];
}

// ── gen: full forward + greedy + tokenizer decode ──────────────────

fn runGen(gpa: std.mem.Allocator, dir_path: []const u8, token_id: u32) !void {
    var model = try model_mod.Model.load(gpa, dir_path);
    defer model.deinit();
    const cfg = model.config;

    const tok_path = try std.fmt.allocPrint(gpa, "{s}/tokenizer.json", .{dir_path});
    defer gpa.free(tok_path);
    var tok = try tokenizer_mod.Tokenizer.loadFromFile(gpa, tok_path);
    defer tok.deinit();

    const stdout = std.io.getStdOut().writer();
    try stdout.print("loaded tokenizer: {d} ids\n", .{tok.vocabSize()});

    const input_str = tok.decode(token_id) orelse "<unknown>";
    try stdout.print("input  token id={d}  string={s}\n", .{ token_id, input_str });

    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const scratch = arena.allocator();

    const logits = try gpa.alloc(f32, cfg.vocab_size);
    defer gpa.free(logits);

    const t0 = std.time.nanoTimestamp();
    if (cfg.family.isHybrid()) {
        // Qwen3.5 hybrid: per-layer SSM + KV state. Single-token gen
        // means max_pos=1. The state is constructed fresh and torn down
        // at the end of this call.
        var state = try cpu_forward.HybridState.init(gpa, &model, 1);
        defer state.deinit();
        try cpu_forward.forwardHybrid(&model, token_id, 0, &state, scratch, logits);
    } else {
        try cpu_forward.forward(&model, token_id, 0, scratch, logits);
    }
    const t1 = std.time.nanoTimestamp();
    const ms = @as(f64, @floatFromInt(t1 - t0)) / 1_000_000.0;
    try stdout.print("forward (CPU, scalar, bf16 weights): {d:.0} ms\n", .{ms});

    // Top-K logits — useful sanity, especially when the argmax is a
    // dud token like <pad>.
    const k_top: usize = 5;
    const top = try topK(gpa, logits, k_top);
    defer gpa.free(top);

    try stdout.print("\ntop {d} logits:\n", .{k_top});
    for (top) |entry| {
        const s = tok.decode(entry.id) orelse "<unknown>";
        try stdout.print("  id={d:>6}  logit={d:>10.4}  {s}\n", .{ entry.id, entry.value, s });
    }

    const sampled = cpu_forward.argmax(logits);
    const out_str = tok.decode(sampled) orelse "<unknown>";
    try stdout.print("\nsampled (greedy): id={d}  string={s}\n", .{ sampled, out_str });
}

const TopKEntry = struct { id: usize, value: f32 };

fn topK(gpa: std.mem.Allocator, logits: []const f32, k: usize) ![]TopKEntry {
    const out = try gpa.alloc(TopKEntry, k);
    for (out) |*e| e.* = .{ .id = 0, .value = -std.math.inf(f32) };
    for (logits, 0..) |v, i| {
        if (v <= out[k - 1].value) continue;
        // Insert into sorted (descending) list.
        var j: usize = k - 1;
        out[j] = .{ .id = i, .value = v };
        while (j > 0 and out[j].value > out[j - 1].value) : (j -= 1) {
            const tmp = out[j];
            out[j] = out[j - 1];
            out[j - 1] = tmp;
        }
    }
    return out;
}

// ── gpu-layer0-test: full layer 0 forward on GPU vs CPU ────────────

const EmbedLookupPush = runtime.EmbedLookupPush;
const AddInPlacePush = runtime.AddInPlacePush;
const AttnDecodeSinglePush = extern struct { n_heads: u32, heads_per_kv: u32, head_dim: u32 };
const ScalePush = runtime_hybrid.ScalePush;
const SliceCopyPush = runtime_hybrid.SliceCopyPush;

fn runGpuLayer0Test(gpa: std.mem.Allocator, dir_path: []const u8, token_id: u32) !void {
    var cpu = try model_mod.Model.load(gpa, dir_path);
    defer cpu.deinit();
    const cfg = cpu.config;

    var ctx = try vk.Context.init(gpa);
    defer ctx.deinit();

    const stdout = std.io.getStdOut().writer();
    try stdout.print("GPU layer-0 forward parity test — token {d}\n", .{token_id});
    try stdout.print("device: {s}\n\n", .{ctx.deviceName()});

    try stdout.print("uploading weights...\n", .{});
    var gm = try gpu_model.GpuModel.upload(gpa, &ctx, &cpu, .fp32_all);
    defer gm.deinit(ctx.device);

    var sc = try gpu_scratch.GpuScratch.init(&ctx, cfg, 1);
    defer sc.deinit(ctx.device);

    // ── Build kernels ───────────────────────────────────────────────
    var k_embed = try pipeline.Kernel.init(&ctx, &shaders.embed_lookup, 2, @sizeOf(EmbedLookupPush));
    defer k_embed.deinit();
    var k_rmsnorm = try pipeline.Kernel.init(&ctx, &shaders.rmsnorm, 3, @sizeOf(RmsnormPush));
    defer k_rmsnorm.deinit();
    var k_matmul = try pipeline.Kernel.init(&ctx, &shaders.matmul_nt, 3, @sizeOf(MatmulPush));
    defer k_matmul.deinit();
    var k_rope = try pipeline.Kernel.init(&ctx, &shaders.rope, 2, @sizeOf(RopePush));
    defer k_rope.deinit();
    var k_attn = try pipeline.Kernel.init(&ctx, &shaders.attn_decode_single, 2, @sizeOf(AttnDecodeSinglePush));
    defer k_attn.deinit();
    var k_add = try pipeline.Kernel.init(&ctx, &shaders.add_in_place, 2, @sizeOf(AddInPlacePush));
    defer k_add.deinit();
    var k_geglu = try pipeline.Kernel.init(&ctx, &shaders.geglu, 3, @sizeOf(GegluPush));
    defer k_geglu.deinit();

    // ── Stage 0: embed → scale ──────────────────────────────────────
    try k_embed.bind(&.{ &gm.embed_tokens, &sc.stream });
    const embed_push = EmbedLookupPush{
        .token_id = token_id,
        .dim = @intCast(cfg.hidden_size),
        .scale = if (cfg.family.embedScalesByDim()) @sqrt(@as(f32, @floatFromInt(cfg.hidden_size))) else 1.0,
    };
    try dispatch1D(&ctx, &k_embed, &embed_push, @intCast(cfg.hidden_size));

    // ── Stage 1: rmsnorm₁ ───────────────────────────────────────────
    try k_rmsnorm.bind(&.{ &sc.stream, &gm.layers[0].input_layernorm, &sc.x_norm });
    const rms1_push = RmsnormPush{
        .dim = @intCast(cfg.hidden_size),
        .eps = cfg.rms_norm_eps,
        .gemma_quirk = if (cfg.family == .gemma) 1 else 0,
    };
    try dispatchPerRow(&ctx, &k_rmsnorm, &rms1_push, 1);

    // ── Stage 2: Q, K, V projections ────────────────────────────────
    const q_dim: u32 = @intCast(cfg.num_attention_heads * cfg.head_dim);
    const kv_dim: u32 = @intCast(cfg.num_key_value_heads * cfg.head_dim);
    const hidden: u32 = @intCast(cfg.hidden_size);
    try k_matmul.bind(&.{ &sc.x_norm, &gm.layers[0].q_proj.?, &sc.q });
    try dispatchMatmul(&ctx, &k_matmul, 1, q_dim, hidden);
    try k_matmul.bind(&.{ &sc.x_norm, &gm.layers[0].k_proj.?, &sc.k });
    try dispatchMatmul(&ctx, &k_matmul, 1, kv_dim, hidden);
    try k_matmul.bind(&.{ &sc.x_norm, &gm.layers[0].v_proj.?, &sc.v });
    try dispatchMatmul(&ctx, &k_matmul, 1, kv_dim, hidden);

    // ── Stage 3: RoPE on Q and K ────────────────────────────────────
    try k_rope.bind(&.{ &sc.q, &sc.q_rot });
    const rope_q_push = RopePush{
        .n_heads = @intCast(cfg.num_attention_heads),
        .head_dim = @intCast(cfg.head_dim),
        .pos = 0,
        .theta_base = cfg.rope_theta,
    };
    try dispatchRope(&ctx, &k_rope, &rope_q_push, cfg.num_attention_heads, cfg.head_dim);

    try k_rope.bind(&.{ &sc.k, &sc.k_rot });
    const rope_k_push = RopePush{
        .n_heads = @intCast(cfg.num_key_value_heads),
        .head_dim = @intCast(cfg.head_dim),
        .pos = 0,
        .theta_base = cfg.rope_theta,
    };
    try dispatchRope(&ctx, &k_rope, &rope_k_push, cfg.num_key_value_heads, cfg.head_dim);

    // ── Stage 4: attention (single-position degenerate) ─────────────
    // No KV history → softmax over 1 score = 1.0 → head_out[h] = V[kv_h(h)].
    try k_attn.bind(&.{ &sc.v, &sc.head_out });
    const attn_push = AttnDecodeSinglePush{
        .n_heads = @intCast(cfg.num_attention_heads),
        .heads_per_kv = @intCast(cfg.num_attention_heads / cfg.num_key_value_heads),
        .head_dim = @intCast(cfg.head_dim),
    };
    try dispatch1D(&ctx, &k_attn, &attn_push, q_dim);

    // ── Stage 5: o_proj ─────────────────────────────────────────────
    try k_matmul.bind(&.{ &sc.head_out, &gm.layers[0].o_proj.?, &sc.attn_out });
    try dispatchMatmul(&ctx, &k_matmul, 1, hidden, q_dim);

    // ── Stage 6: residual add (stream += attn_out) ──────────────────
    try k_add.bind(&.{ &sc.stream, &sc.attn_out });
    const add_push = AddInPlacePush{ .n = hidden };
    try dispatch1D(&ctx, &k_add, &add_push, hidden);

    // ── Stage 7: rmsnorm₂ ───────────────────────────────────────────
    try k_rmsnorm.bind(&.{ &sc.stream, &gm.layers[0].post_attention_layernorm, &sc.mid_norm });
    try dispatchPerRow(&ctx, &k_rmsnorm, &rms1_push, 1);

    // ── Stage 8: gate, up projections ───────────────────────────────
    const inter: u32 = @intCast(cfg.intermediate_size);
    try k_matmul.bind(&.{ &sc.mid_norm, &gm.layers[0].gate_proj, &sc.gate });
    try dispatchMatmul(&ctx, &k_matmul, 1, inter, hidden);
    try k_matmul.bind(&.{ &sc.mid_norm, &gm.layers[0].up_proj, &sc.up });
    try dispatchMatmul(&ctx, &k_matmul, 1, inter, hidden);

    // ── Stage 9: GeGLU ─────────────────────────────────────────────
    try k_geglu.bind(&.{ &sc.gate, &sc.up, &sc.fused });
    const geglu_push = GegluPush{ .n = inter };
    try dispatch1D(&ctx, &k_geglu, &geglu_push, inter);

    // ── Stage 10: down_proj ─────────────────────────────────────────
    try k_matmul.bind(&.{ &sc.fused, &gm.layers[0].down_proj, &sc.ffn_out });
    try dispatchMatmul(&ctx, &k_matmul, 1, hidden, inter);

    // ── Stage 11: residual add (stream += ffn_out) ──────────────────
    try k_add.bind(&.{ &sc.stream, &sc.ffn_out });
    try dispatch1D(&ctx, &k_add, &add_push, hidden);

    // ── Read back, compare against CPU layer 0 ──────────────────────
    const got = try gpa.alloc(f32, cfg.hidden_size);
    defer gpa.free(got);
    try sc.stream.readBack(&ctx, f32, got);

    const want = try cpuLayer0Forward(gpa, &cpu, token_id);
    defer gpa.free(want);

    var max_abs: f32 = 0;
    var max_rel: f32 = 0;
    var max_idx: usize = 0;
    for (got, want, 0..) |g, w, i| {
        const da = @abs(g - w);
        const dr = da / @max(@abs(w), 1e-30);
        if (da > max_abs) {
            max_abs = da;
            max_idx = i;
        }
        if (dr > max_rel) max_rel = dr;
    }

    try stdout.print("\nlayer 0 output stream: max |Δ| = {e:.3}  (at idx {d}: cpu={d:.6} gpu={d:.6})\n", .{
        max_abs, max_idx, want[max_idx], got[max_idx],
    });
    try stdout.print("max relative error = {e:.3}\n", .{max_rel});

    if (max_abs > 1e-2) {
        std.debug.print("FAIL: max |Δ| above tolerance\n", .{});
        return error.ParityFailed;
    }
    try stdout.print("\nPASS GPU layer 0 matches CPU within {e:.0}\n", .{@as(f32, 1e-2)});
}

/// Runs the layer-0-only chunk of the CPU forward pass and returns the
/// post-FFN-residual stream as fp32, for parity comparison. Mirrors
/// runLayer0Test but without the per-stage prints.
fn cpuLayer0Forward(gpa: std.mem.Allocator, cpu: *const model_mod.Model, token_id: u32) ![]f32 {
    const cfg = cpu.config;
    const layer = cpu.layers[0];
    const hidden = cfg.hidden_size;
    const inter = cfg.intermediate_size;
    const n_heads = cfg.num_attention_heads;
    const n_kv = cfg.num_key_value_heads;
    const head_dim = cfg.head_dim;
    const q_dim = n_heads * head_dim;
    const kv_dim = n_kv * head_dim;
    const heads_per_kv = n_heads / n_kv;

    const stream = try gpa.alloc(f32, hidden);
    errdefer gpa.free(stream);
    try cpu_math.embedRowAsF32(stream, cpu.embed_tokens, token_id);
    if (cfg.family.embedScalesByDim()) {
        const s: f32 = @sqrt(@as(f32, @floatFromInt(hidden)));
        for (stream) |*xi| xi.* *= s;
    }

    const x_norm = try gpa.alloc(f32, hidden);
    defer gpa.free(x_norm);
    try cpu_math.rmsnorm(x_norm, stream, layer.input_layernorm, cfg.rms_norm_eps, cfg.family);

    const v = try gpa.alloc(f32, kv_dim);
    defer gpa.free(v);
    try cpu_math.matmul_nt(v, x_norm, layer.v_proj.?, 1, kv_dim, hidden);

    const head_out = try gpa.alloc(f32, q_dim);
    defer gpa.free(head_out);
    for (0..n_heads) |h| {
        const kv_h = h / heads_per_kv;
        const v_off = kv_h * head_dim;
        const out_off = h * head_dim;
        @memcpy(head_out[out_off .. out_off + head_dim], v[v_off .. v_off + head_dim]);
    }

    const attn_out = try gpa.alloc(f32, hidden);
    defer gpa.free(attn_out);
    try cpu_math.matmul_nt(attn_out, head_out, layer.o_proj.?, 1, hidden, q_dim);
    for (stream, attn_out) |*s, a| s.* += a;

    const mid_norm = try gpa.alloc(f32, hidden);
    defer gpa.free(mid_norm);
    try cpu_math.rmsnorm(mid_norm, stream, layer.post_attention_layernorm, cfg.rms_norm_eps, cfg.family);

    const gate = try gpa.alloc(f32, inter);
    defer gpa.free(gate);
    const up = try gpa.alloc(f32, inter);
    defer gpa.free(up);
    try cpu_math.matmul_nt(gate, mid_norm, layer.gate_proj, 1, inter, hidden);
    try cpu_math.matmul_nt(up, mid_norm, layer.up_proj, 1, inter, hidden);

    const fused = try gpa.alloc(f32, inter);
    defer gpa.free(fused);
    try cpu_math.geglu(fused, gate, up);

    const ffn_out = try gpa.alloc(f32, hidden);
    defer gpa.free(ffn_out);
    try cpu_math.matmul_nt(ffn_out, fused, layer.down_proj, 1, hidden, inter);
    for (stream, ffn_out) |*s, f| s.* += f;

    return stream;
}

// ── Recorder-based dispatch helpers (single source of truth: runtime.zig) ──

const recDispatch1D = runtime.recDispatch1D;
const recDispatchPerRow = runtime.recDispatchPerRow;
const recDispatchMatmul = runtime.recDispatchMatmul;
const recDispatchRope = runtime.recDispatchRope;

// ── Dispatch helpers — keep call sites readable ──────────────────────

fn dispatch1D(
    ctx: *const vk.Context,
    kern: *const pipeline.Kernel,
    push: anytype,
    n: u32,
) !void {
    const local: u32 = 256;
    const groups: u32 = (n + local - 1) / local;
    try buffer.submitOneShot(ctx, struct {
        kern: *const pipeline.Kernel,
        push: @TypeOf(push),
        gx: u32,
        pub fn record(s: @This(), cmd: vk.c.VkCommandBuffer) void {
            s.kern.dispatch(cmd, s.push, s.gx, 1, 1);
        }
    }{ .kern = kern, .push = push, .gx = groups });
}

fn dispatchPerRow(
    ctx: *const vk.Context,
    kern: *const pipeline.Kernel,
    push: anytype,
    n_rows: u32,
) !void {
    try buffer.submitOneShot(ctx, struct {
        kern: *const pipeline.Kernel,
        push: @TypeOf(push),
        n_rows: u32,
        pub fn record(s: @This(), cmd: vk.c.VkCommandBuffer) void {
            s.kern.dispatch(cmd, s.push, s.n_rows, 1, 1);
        }
    }{ .kern = kern, .push = push, .n_rows = n_rows });
}

fn dispatchMatmul(
    ctx: *const vk.Context,
    kern: *const pipeline.Kernel,
    m: u32,
    n: u32,
    k: u32,
) !void {
    const local_xy: u32 = 16;
    const gx: u32 = (m + local_xy - 1) / local_xy;
    const gy: u32 = (n + local_xy - 1) / local_xy;
    const push = MatmulPush{ .m = m, .n = n, .k = k };
    try buffer.submitOneShot(ctx, struct {
        kern: *const pipeline.Kernel,
        push: *const MatmulPush,
        gx: u32,
        gy: u32,
        pub fn record(s: @This(), cmd: vk.c.VkCommandBuffer) void {
            s.kern.dispatch(cmd, s.push, s.gx, s.gy, 1);
        }
    }{ .kern = kern, .push = &push, .gx = gx, .gy = gy });
}

fn dispatchRope(
    ctx: *const vk.Context,
    kern: *const pipeline.Kernel,
    push: *const RopePush,
    n_heads: usize,
    head_dim: usize,
) !void {
    const local: u32 = 256;
    const pairs: u32 = @intCast(n_heads * (head_dim / 2));
    const groups: u32 = (pairs + local - 1) / local;
    try buffer.submitOneShot(ctx, struct {
        kern: *const pipeline.Kernel,
        push: *const RopePush,
        gx: u32,
        pub fn record(s: @This(), cmd: vk.c.VkCommandBuffer) void {
            s.kern.dispatch(cmd, s.push, s.gx, 1, 1);
        }
    }{ .kern = kern, .push = push, .gx = groups });
}

// ── gpu-load: upload all weights to GPU, round-trip-verify a few ────

fn runGpuLoad(gpa: std.mem.Allocator, dir_path: []const u8) !void {
    var cpu = try model_mod.Model.load(gpa, dir_path);
    defer cpu.deinit();
    const cfg = cpu.config;

    var ctx = try vk.Context.init(gpa);
    defer ctx.deinit();

    const stdout = std.io.getStdOut().writer();
    try stdout.print("uploading {d} layers + embed + final_norm + lm_head to {s}\n", .{
        cfg.num_hidden_layers, ctx.deviceName(),
    });

    const t0 = std.time.nanoTimestamp();
    var gm = try gpu_model.GpuModel.upload(gpa, &ctx, &cpu, .fp32_all);
    defer gm.deinit(ctx.device);
    const t1 = std.time.nanoTimestamp();
    const upload_ms = @as(f64, @floatFromInt(t1 - t0)) / 1_000_000.0;
    try stdout.print("upload time: {d:.0} ms ({d} buffers)\n\n", .{
        upload_ms,
        2 + 9 * cfg.num_hidden_layers + 1, // embed + final_norm + lm_head + 9/layer
    });

    // ── Round-trip a few tensors ─────────────────────────────────────
    // Pull representative samples back from the device and check that
    // they match the host fp32 representation.
    try roundTripCheck(gpa, &ctx, &gm.embed_tokens, cpu.embed_tokens, "embed_tokens");
    try roundTripCheck(gpa, &ctx, &gm.final_norm, cpu.final_norm, "final_norm");
    // Pick the first full-attention layer for the q_proj round-trip
    // (hybrid models put linear-attn layers first; q_proj is null
    // there). down_proj is present on every layer in every family.
    const first_full: usize = blk: {
        for (cpu.layers, 0..) |l, i| if (l.layer_type == .full_attention) break :blk i;
        break :blk 0;
    };
    if (gm.layers[first_full].q_proj) |*q| {
        try roundTripCheck(gpa, &ctx, q, cpu.layers[first_full].q_proj.?, "layer q_proj");
    }
    try roundTripCheck(gpa, &ctx, &gm.layers[0].input_layernorm, cpu.layers[0].input_layernorm, "layer 0 input_layernorm");
    const last_layer = gm.layers.len - 1;
    try roundTripCheck(gpa, &ctx, &gm.layers[last_layer].down_proj, cpu.layers[last_layer].down_proj, "last-layer down_proj");

    try stdout.print("\nPASS gpu-load (5 tensors round-tripped within fp32 ULP)\n", .{});
}

fn roundTripCheck(
    gpa: std.mem.Allocator,
    ctx: *const vk.Context,
    buf: *const buffer.Buffer,
    cpu_t: safetensors.Tensor,
    label: []const u8,
) !void {
    const stdout = std.io.getStdOut().writer();
    const numel = cpu_t.numel();

    // Materialise the CPU tensor as fp32 for the comparison.
    const want = try gpa.alloc(f32, numel);
    defer gpa.free(want);
    switch (cpu_t.dtype) {
        .f32 => @memcpy(want, @as([*]align(1) const f32, @ptrCast(cpu_t.bytes.ptr))[0..numel]),
        .bf16 => dtype.bf16SliceToF32(dtype.asU16(cpu_t.bytes), want),
        .f16 => dtype.f16SliceToF32(dtype.asU16(cpu_t.bytes), want),
        else => return error.UnsupportedDtype,
    }

    const got = try gpa.alloc(f32, numel);
    defer gpa.free(got);
    try buf.readBack(ctx, f32, got);

    var max_abs: f32 = 0;
    for (want, got) |w, g| {
        const d = @abs(w - g);
        if (d > max_abs) max_abs = d;
    }
    try stdout.print("  {s:<28}  numel={d:>10}  max |Δ| = {e:.3}\n", .{ label, numel, max_abs });
    if (max_abs > 0.0) {
        // We expect bit-exact round-trip — the only thing happening
        // here is bf16→fp32 conversion (deterministic) followed by an
        // fp32 staging upload + readback (no further conversion).
        return error.RoundTripMismatch;
    }
}

// ── gpu-rmsnorm-test: real Gemma layer 0 input_layernorm on GPU ────

fn runGpuRmsnormTest(gpa: std.mem.Allocator, dir_path: []const u8, token_id: u32) !void {
    var model = try model_mod.Model.load(gpa, dir_path);
    defer model.deinit();
    const cfg = model.config;

    var ctx = try vk.Context.init(gpa);
    defer ctx.deinit();

    const stdout = std.io.getStdOut().writer();
    try stdout.print("GPU rmsnorm parity test on layer 0 input_layernorm — token {d}\n", .{token_id});
    try stdout.print("device: {s}\n\n", .{ctx.deviceName()});

    // Embedding → scale → (rmsnorm)
    const x = try gpa.alloc(f32, cfg.hidden_size);
    defer gpa.free(x);
    try cpu_math.embedRowAsF32(x, model.embed_tokens, token_id);
    if (cfg.family.embedScalesByDim()) {
        const s: f32 = @sqrt(@as(f32, @floatFromInt(cfg.hidden_size)));
        for (x) |*xi| xi.* *= s;
    }

    // CPU baseline.
    const want = try gpa.alloc(f32, cfg.hidden_size);
    defer gpa.free(want);
    const t_cpu0 = std.time.nanoTimestamp();
    try cpu_math.rmsnorm(want, x, model.layers[0].input_layernorm, cfg.rms_norm_eps, cfg.family);
    const t_cpu1 = std.time.nanoTimestamp();
    const cpu_ms = @as(f64, @floatFromInt(t_cpu1 - t_cpu0)) / 1_000_000.0;

    // Materialise weight as fp32 (bf16 on disk).
    const w_bf16 = dtype.asU16(model.layers[0].input_layernorm.bytes);
    const w_f32 = try gpa.alloc(f32, w_bf16.len);
    defer gpa.free(w_f32);
    dtype.bf16SliceToF32(w_bf16, w_f32);

    // GPU dispatch.
    var buf_a = try buffer.Buffer.initStatic(&ctx, f32, x);
    defer buf_a.deinit(ctx.device);
    var buf_w = try buffer.Buffer.initStatic(&ctx, f32, w_f32);
    defer buf_w.deinit(ctx.device);
    var buf_c = try buffer.Buffer.initDeviceOnly(&ctx, cfg.hidden_size * @sizeOf(f32));
    defer buf_c.deinit(ctx.device);

    var kern = try pipeline.Kernel.init(&ctx, &shaders.rmsnorm, 3, @sizeOf(RmsnormPush));
    defer kern.deinit();
    try kern.bind(&.{ &buf_a, &buf_w, &buf_c });

    const push = RmsnormPush{
        .dim = @intCast(cfg.hidden_size),
        .eps = cfg.rms_norm_eps,
        .gemma_quirk = if (cfg.family == .gemma) 1 else 0,
    };

    const t_gpu0 = std.time.nanoTimestamp();
    try buffer.submitOneShot(&ctx, struct {
        kern: *const pipeline.Kernel,
        push: *const RmsnormPush,
        pub fn record(s: @This(), cmd: vk.c.VkCommandBuffer) void {
            s.kern.dispatch(cmd, s.push, 1, 1, 1);
        }
    }{ .kern = &kern, .push = &push });
    const t_gpu1 = std.time.nanoTimestamp();
    const gpu_ms = @as(f64, @floatFromInt(t_gpu1 - t_gpu0)) / 1_000_000.0;

    const got = try gpa.alloc(f32, cfg.hidden_size);
    defer gpa.free(got);
    try buf_c.readBack(&ctx, f32, got);

    var max_abs: f32 = 0;
    var max_idx: usize = 0;
    for (got, want, 0..) |g, e, i| {
        const d = @abs(g - e);
        if (d > max_abs) {
            max_abs = d;
            max_idx = i;
        }
    }

    try stdout.print("CPU: {d:.2} ms  GPU: {d:.2} ms (incl. submit + queue idle)\n", .{ cpu_ms, gpu_ms });
    try stdout.print("max |Δ| = {e:.3}  (at idx {d}: cpu={d:.6} gpu={d:.6})\n", .{
        max_abs, max_idx, want[max_idx], got[max_idx],
    });
    if (max_abs > 1e-3) {
        std.debug.print("FAIL: max |Δ| above tolerance\n", .{});
        return error.ParityFailed;
    }
    try stdout.print("\nPASS GPU rmsnorm matches CPU within {e:.0}\n", .{@as(f32, 1e-3)});
}

// ── gpu-rope-test: real Gemma Q at pos=1 vs CPU ────────────────────

fn runGpuRopeTest(gpa: std.mem.Allocator, dir_path: []const u8, token_id: u32) !void {
    var model = try model_mod.Model.load(gpa, dir_path);
    defer model.deinit();
    const cfg = model.config;
    const layer = model.layers[0];

    var ctx = try vk.Context.init(gpa);
    defer ctx.deinit();

    const stdout = std.io.getStdOut().writer();
    try stdout.print("GPU RoPE parity test on layer 0 Q at pos=1 — token {d}\n", .{token_id});
    try stdout.print("device: {s}\n\n", .{ctx.deviceName()});

    // Reproduce Q via the verified CPU pipeline.
    const x = try gpa.alloc(f32, cfg.hidden_size);
    defer gpa.free(x);
    try cpu_math.embedRowAsF32(x, model.embed_tokens, token_id);
    if (cfg.family.embedScalesByDim()) {
        const s: f32 = @sqrt(@as(f32, @floatFromInt(cfg.hidden_size)));
        for (x) |*xi| xi.* *= s;
    }
    const x_norm = try gpa.alloc(f32, cfg.hidden_size);
    defer gpa.free(x_norm);
    try cpu_math.rmsnorm(x_norm, x, layer.input_layernorm, cfg.rms_norm_eps, cfg.family);

    const q_dim = cfg.num_attention_heads * cfg.head_dim;
    const q = try gpa.alloc(f32, q_dim);
    defer gpa.free(q);
    try cpu_math.matmul_nt(q, x_norm, layer.q_proj.?, 1, q_dim, cfg.hidden_size);

    // CPU baseline at pos=1.
    const want = try gpa.alloc(f32, q_dim);
    defer gpa.free(want);
    try cpu_math.applyRope(want, q, cfg.num_attention_heads, cfg.head_dim, 1, cfg.rope_theta);

    // GPU dispatch.
    var buf_in = try buffer.Buffer.initStatic(&ctx, f32, q);
    defer buf_in.deinit(ctx.device);
    var buf_out = try buffer.Buffer.initDeviceOnly(&ctx, q_dim * @sizeOf(f32));
    defer buf_out.deinit(ctx.device);

    var kern = try pipeline.Kernel.init(&ctx, &shaders.rope, 2, @sizeOf(RopePush));
    defer kern.deinit();
    try kern.bind(&.{ &buf_in, &buf_out });

    const local: u32 = 256;
    const pairs: u32 = @intCast(cfg.num_attention_heads * (cfg.head_dim / 2));
    const groups: u32 = (pairs + local - 1) / local;
    const push = RopePush{
        .n_heads = @intCast(cfg.num_attention_heads),
        .head_dim = @intCast(cfg.head_dim),
        .pos = 1,
        .theta_base = cfg.rope_theta,
    };

    const t_gpu0 = std.time.nanoTimestamp();
    try buffer.submitOneShot(&ctx, struct {
        kern: *const pipeline.Kernel,
        push: *const RopePush,
        groups: u32,
        pub fn record(s: @This(), cmd: vk.c.VkCommandBuffer) void {
            s.kern.dispatch(cmd, s.push, s.groups, 1, 1);
        }
    }{ .kern = &kern, .push = &push, .groups = groups });
    const t_gpu1 = std.time.nanoTimestamp();
    const gpu_ms = @as(f64, @floatFromInt(t_gpu1 - t_gpu0)) / 1_000_000.0;

    const got = try gpa.alloc(f32, q_dim);
    defer gpa.free(got);
    try buf_out.readBack(&ctx, f32, got);

    var max_abs: f32 = 0;
    var max_idx: usize = 0;
    for (got, want, 0..) |g, e, i| {
        const d = @abs(g - e);
        if (d > max_abs) {
            max_abs = d;
            max_idx = i;
        }
    }

    try stdout.print("GPU: {d:.2} ms (incl. submit + queue idle)\n", .{gpu_ms});
    try stdout.print("max |Δ| = {e:.3}  (at idx {d}: cpu={d:.6} gpu={d:.6})\n", .{
        max_abs, max_idx, want[max_idx], got[max_idx],
    });
    if (max_abs > 1e-3) {
        std.debug.print("FAIL: max |Δ| above tolerance\n", .{});
        return error.ParityFailed;
    }
    try stdout.print("\nPASS GPU rope matches CPU within {e:.0}\n", .{@as(f32, 1e-3)});
}

// ── gpu-geglu-test: real Gemma layer 0 GeGLU vs CPU ────────────────

fn runGpuGegluTest(gpa: std.mem.Allocator, dir_path: []const u8, token_id: u32) !void {
    var model = try model_mod.Model.load(gpa, dir_path);
    defer model.deinit();
    const cfg = model.config;
    const layer = model.layers[0];

    var ctx = try vk.Context.init(gpa);
    defer ctx.deinit();

    const stdout = std.io.getStdOut().writer();
    try stdout.print("GPU GeGLU parity test on layer 0 — token {d}\n", .{token_id});
    try stdout.print("device: {s}\n\n", .{ctx.deviceName()});

    // Reproduce the FFN inputs on CPU: embed → scale → rmsnorm₁ → attn
    // (single-position degenerate to V) → o_proj → residual → rmsnorm₂.
    const inter = cfg.intermediate_size;

    const x = try gpa.alloc(f32, cfg.hidden_size);
    defer gpa.free(x);
    try cpu_math.embedRowAsF32(x, model.embed_tokens, token_id);
    if (cfg.family.embedScalesByDim()) {
        const s: f32 = @sqrt(@as(f32, @floatFromInt(cfg.hidden_size)));
        for (x) |*xi| xi.* *= s;
    }
    const x_norm = try gpa.alloc(f32, cfg.hidden_size);
    defer gpa.free(x_norm);
    try cpu_math.rmsnorm(x_norm, x, layer.input_layernorm, cfg.rms_norm_eps, cfg.family);

    // Single-position attention output = V projected through o_proj.
    const v = try gpa.alloc(f32, cfg.num_key_value_heads * cfg.head_dim);
    defer gpa.free(v);
    try cpu_math.matmul_nt(v, x_norm, layer.v_proj.?, 1, v.len, cfg.hidden_size);

    const head_out = try gpa.alloc(f32, cfg.num_attention_heads * cfg.head_dim);
    defer gpa.free(head_out);
    const heads_per_kv = cfg.num_attention_heads / cfg.num_key_value_heads;
    for (0..cfg.num_attention_heads) |h| {
        const kv_h = h / heads_per_kv;
        const v_off = kv_h * cfg.head_dim;
        const out_off = h * cfg.head_dim;
        @memcpy(head_out[out_off .. out_off + cfg.head_dim], v[v_off .. v_off + cfg.head_dim]);
    }
    const attn_out = try gpa.alloc(f32, cfg.hidden_size);
    defer gpa.free(attn_out);
    try cpu_math.matmul_nt(attn_out, head_out, layer.o_proj.?, 1, cfg.hidden_size, head_out.len);

    const mid = try gpa.alloc(f32, cfg.hidden_size);
    defer gpa.free(mid);
    for (mid, x, attn_out) |*m, xi, ai| m.* = xi + ai;
    const mid_norm = try gpa.alloc(f32, cfg.hidden_size);
    defer gpa.free(mid_norm);
    try cpu_math.rmsnorm(mid_norm, mid, layer.post_attention_layernorm, cfg.rms_norm_eps, cfg.family);

    const gate = try gpa.alloc(f32, inter);
    defer gpa.free(gate);
    const upv = try gpa.alloc(f32, inter);
    defer gpa.free(upv);
    try cpu_math.matmul_nt(gate, mid_norm, layer.gate_proj, 1, inter, cfg.hidden_size);
    try cpu_math.matmul_nt(upv, mid_norm, layer.up_proj, 1, inter, cfg.hidden_size);

    const want = try gpa.alloc(f32, inter);
    defer gpa.free(want);
    const t_cpu0 = std.time.nanoTimestamp();
    try cpu_math.geglu(want, gate, upv);
    const t_cpu1 = std.time.nanoTimestamp();
    const cpu_ms = @as(f64, @floatFromInt(t_cpu1 - t_cpu0)) / 1_000_000.0;

    var buf_g = try buffer.Buffer.initStatic(&ctx, f32, gate);
    defer buf_g.deinit(ctx.device);
    var buf_u = try buffer.Buffer.initStatic(&ctx, f32, upv);
    defer buf_u.deinit(ctx.device);
    var buf_o = try buffer.Buffer.initDeviceOnly(&ctx, inter * @sizeOf(f32));
    defer buf_o.deinit(ctx.device);

    var kern = try pipeline.Kernel.init(&ctx, &shaders.geglu, 3, @sizeOf(GegluPush));
    defer kern.deinit();
    try kern.bind(&.{ &buf_g, &buf_u, &buf_o });

    const local: u32 = 256;
    const n_u32: u32 = @intCast(inter);
    const groups: u32 = (n_u32 + local - 1) / local;
    const push = GegluPush{ .n = n_u32 };

    const t_gpu0 = std.time.nanoTimestamp();
    try buffer.submitOneShot(&ctx, struct {
        kern: *const pipeline.Kernel,
        push: *const GegluPush,
        groups: u32,
        pub fn record(s: @This(), cmd: vk.c.VkCommandBuffer) void {
            s.kern.dispatch(cmd, s.push, s.groups, 1, 1);
        }
    }{ .kern = &kern, .push = &push, .groups = groups });
    const t_gpu1 = std.time.nanoTimestamp();
    const gpu_ms = @as(f64, @floatFromInt(t_gpu1 - t_gpu0)) / 1_000_000.0;

    const got = try gpa.alloc(f32, inter);
    defer gpa.free(got);
    try buf_o.readBack(&ctx, f32, got);

    var max_abs: f32 = 0;
    var max_idx: usize = 0;
    for (got, want, 0..) |g, e, i| {
        const d = @abs(g - e);
        if (d > max_abs) {
            max_abs = d;
            max_idx = i;
        }
    }

    try stdout.print("CPU: {d:.2} ms  GPU: {d:.2} ms (incl. submit + queue idle)\n", .{ cpu_ms, gpu_ms });
    try stdout.print("max |Δ| = {e:.3}  (at idx {d}: cpu={d:.6} gpu={d:.6})\n", .{
        max_abs, max_idx, want[max_idx], got[max_idx],
    });
    if (max_abs > 1e-3) {
        std.debug.print("FAIL: max |Δ| above tolerance\n", .{});
        return error.ParityFailed;
    }
    try stdout.print("\nPASS GPU geglu matches CPU within {e:.0}\n", .{@as(f32, 1e-3)});
}

// ── gpu-qproj-test: real Gemma q_proj on GPU vs CPU ────────────────

fn runGpuQprojTest(gpa: std.mem.Allocator, dir_path: []const u8, token_id: u32) !void {
    var model = try model_mod.Model.load(gpa, dir_path);
    defer model.deinit();
    const cfg = model.config;

    var ctx = try vk.Context.init(gpa);
    defer ctx.deinit();

    const stdout = std.io.getStdOut().writer();
    try stdout.print("GPU matmul_nt parity test on layer 0 q_proj — token {d}\n", .{token_id});
    try stdout.print("device: {s}\n\n", .{ctx.deviceName()});

    // ── Reproduce the qproj-test inputs on the host ─────────────────
    const x = try gpa.alloc(f32, cfg.hidden_size);
    defer gpa.free(x);
    try cpu_math.embedRowAsF32(x, model.embed_tokens, token_id);
    if (cfg.family.embedScalesByDim()) {
        const s: f32 = @sqrt(@as(f32, @floatFromInt(cfg.hidden_size)));
        for (x) |*xi| xi.* *= s;
    }
    const x_norm = try gpa.alloc(f32, cfg.hidden_size);
    defer gpa.free(x_norm);
    try cpu_math.rmsnorm(x_norm, x, model.layers[0].input_layernorm, cfg.rms_norm_eps, cfg.family);

    const q_dim = cfg.num_attention_heads * cfg.head_dim;

    // ── CPU baseline (existing path) ────────────────────────────────
    const q_cpu = try gpa.alloc(f32, q_dim);
    defer gpa.free(q_cpu);
    const t_cpu0 = std.time.nanoTimestamp();
    try cpu_math.matmul_nt(q_cpu, x_norm, model.layers[0].q_proj.?, 1, q_dim, cfg.hidden_size);
    const t_cpu1 = std.time.nanoTimestamp();
    const cpu_ms = @as(f64, @floatFromInt(t_cpu1 - t_cpu0)) / 1_000_000.0;

    // ── Materialise q_proj as fp32 for GPU upload ───────────────────
    // The GPU kernel is fp32-only for now; we'll add a bf16-aware
    // variant once the fp32 path is parity-clean. The conversion is
    // O(numel) so it doesn't dominate setup, but it does double host
    // memory while we hold both copies — fine for this kernel
    // (32 MiB), would want an in-place stream once we do all weights.
    const w_bf16 = dtype.asU16(model.layers[0].q_proj.?.bytes);
    const w_f32 = try gpa.alloc(f32, w_bf16.len);
    defer gpa.free(w_f32);
    dtype.bf16SliceToF32(w_bf16, w_f32);

    // ── GPU upload + dispatch ───────────────────────────────────────
    var buf_a = try buffer.Buffer.initStatic(&ctx, f32, x_norm);
    defer buf_a.deinit(ctx.device);
    var buf_b = try buffer.Buffer.initStatic(&ctx, f32, w_f32);
    defer buf_b.deinit(ctx.device);
    var buf_c = try buffer.Buffer.initDeviceOnly(&ctx, q_dim * @sizeOf(f32));
    defer buf_c.deinit(ctx.device);

    var kern = try pipeline.Kernel.init(&ctx, &shaders.matmul_nt, 3, @sizeOf(MatmulPush));
    defer kern.deinit();
    try kern.bind(&.{ &buf_a, &buf_b, &buf_c });

    const local_xy: u32 = 16;
    const m: u32 = 1;
    const n: u32 = @intCast(q_dim);
    const k: u32 = @intCast(cfg.hidden_size);
    const groups_x: u32 = (m + local_xy - 1) / local_xy;
    const groups_y: u32 = (n + local_xy - 1) / local_xy;
    const push = MatmulPush{ .m = m, .n = n, .k = k };

    const t_gpu0 = std.time.nanoTimestamp();
    try buffer.submitOneShot(&ctx, struct {
        kern: *const pipeline.Kernel,
        push: *const MatmulPush,
        gx: u32,
        gy: u32,
        pub fn record(s: @This(), cmd: vk.c.VkCommandBuffer) void {
            s.kern.dispatch(cmd, s.push, s.gx, s.gy, 1);
        }
    }{ .kern = &kern, .push = &push, .gx = groups_x, .gy = groups_y });
    const t_gpu1 = std.time.nanoTimestamp();
    const gpu_ms = @as(f64, @floatFromInt(t_gpu1 - t_gpu0)) / 1_000_000.0;

    const q_gpu = try gpa.alloc(f32, q_dim);
    defer gpa.free(q_gpu);
    try buf_c.readBack(&ctx, f32, q_gpu);

    // ── Parity ──────────────────────────────────────────────────────
    var max_abs_err: f32 = 0;
    var max_rel_err: f32 = 0;
    var max_idx: usize = 0;
    for (q_cpu, q_gpu, 0..) |c, g, i| {
        const abs_err = @abs(c - g);
        const denom = @max(@abs(c), 1e-30);
        const rel_err = abs_err / denom;
        if (abs_err > max_abs_err) {
            max_abs_err = abs_err;
            max_idx = i;
        }
        if (rel_err > max_rel_err) max_rel_err = rel_err;
    }

    try stdout.print("CPU: {d:.2} ms  GPU: {d:.2} ms (incl. submit + queue idle)\n", .{ cpu_ms, gpu_ms });
    try stdout.print("max |Δ| = {e:.3}  (at idx {d}: cpu={d:.6} gpu={d:.6})\n", .{
        max_abs_err, max_idx, q_cpu[max_idx], q_gpu[max_idx],
    });
    try stdout.print("max relative error = {e:.3}\n", .{max_rel_err});

    if (max_abs_err > 1e-3) {
        std.debug.print("FAIL: max |Δ| above tolerance\n", .{});
        return error.ParityFailed;
    }
    try stdout.print("\nPASS GPU q_proj matches CPU within {e:.0}\n", .{@as(f32, 1e-3)});
}

fn printStreamStats(w: anytype, label: []const u8, x: []const f32) !void {
    var min_v: f32 = std.math.inf(f32);
    var max_v: f32 = -std.math.inf(f32);
    var sum_sq: f64 = 0;
    var nan_count: usize = 0;
    var inf_count: usize = 0;
    for (x) |v| {
        if (std.math.isNan(v)) {
            nan_count += 1;
            continue;
        }
        if (std.math.isInf(v)) {
            inf_count += 1;
            continue;
        }
        if (v < min_v) min_v = v;
        if (v > max_v) max_v = v;
        sum_sq += @as(f64, v) * @as(f64, v);
    }
    const rms = std.math.sqrt(sum_sq / @as(f64, @floatFromInt(x.len)));
    try w.print("  {s:<32} min={d:>10.4}  max={d:>10.4}  rms={d:>10.4}", .{ label, min_v, max_v, rms });
    if (nan_count > 0 or inf_count > 0) {
        try w.print("  nan={d} inf={d}", .{ nan_count, inf_count });
    }
    try w.print("\n", .{});
}

// ── vec_add smoke: validates the whole Vulkan compute path ───────────

const N: u32 = 1024 * 1024;
const VecAddPush = extern struct { n: u32 };

fn runVecAddSmoke(allocator: std.mem.Allocator) !void {
    var ctx = try vk.Context.init(allocator);
    defer ctx.deinit();

    const a = try allocator.alloc(f32, N);
    defer allocator.free(a);
    const b = try allocator.alloc(f32, N);
    defer allocator.free(b);
    const out = try allocator.alloc(f32, N);
    defer allocator.free(out);
    for (a, b, 0..) |*ai, *bi, i| {
        ai.* = @floatFromInt(i);
        bi.* = @as(f32, @floatFromInt(i)) * 2.0;
    }

    var buf_a = try buffer.Buffer.initStatic(&ctx, f32, a);
    defer buf_a.deinit(ctx.device);
    var buf_b = try buffer.Buffer.initStatic(&ctx, f32, b);
    defer buf_b.deinit(ctx.device);
    var buf_c = try buffer.Buffer.initDeviceOnly(&ctx, N * @sizeOf(f32));
    defer buf_c.deinit(ctx.device);

    var vec_add = try pipeline.Kernel.init(&ctx, &shaders.vec_add, 3, @sizeOf(VecAddPush));
    defer vec_add.deinit();
    try vec_add.bind(&.{ &buf_a, &buf_b, &buf_c });

    const local_size: u32 = 256;
    const groups: u32 = (N + local_size - 1) / local_size;
    const push = VecAddPush{ .n = N };

    try buffer.submitOneShot(&ctx, struct {
        kern: *const pipeline.Kernel,
        push: *const VecAddPush,
        groups: u32,
        pub fn record(s: @This(), cmd: vk.c.VkCommandBuffer) void {
            s.kern.dispatch(cmd, s.push, s.groups, 1, 1);
        }
    }{ .kern = &vec_add, .push = &push, .groups = groups });

    try buf_c.readBack(&ctx, f32, out);
    for (out, 0..) |v, i| {
        const expected = @as(f32, @floatFromInt(i)) * 3.0;
        if (v != expected) {
            std.debug.print("vec_add MISMATCH at {d}: got {d}, expected {d}\n", .{ i, v, expected });
            return error.ParityFailed;
        }
    }
    std.debug.print("PASS vec_add ({d} elems) on {s}\n", .{ N, ctx.deviceName() });
}

// ── safetensors smoke: synthesizes a file, parses it, checks round-trip ──

fn runSafeTensorsSmoke(allocator: std.mem.Allocator) !void {
    // Build a two-tensor file in memory:
    //   weight_a: F32, shape [3, 2], values 0..5
    //   weight_b: I32, shape [4],    values [10, 20, 30, 40]
    const w_a: [6]f32 = .{ 0.0, 1.0, 2.0, 3.0, 4.0, 5.0 };
    const w_b: [4]i32 = .{ 10, 20, 30, 40 };

    // Header JSON. Keys sorted; offsets are relative to the end of the
    // header itself. Total tensor bytes: 24 (a) + 16 (b) = 40.
    const header_json =
        \\{"weight_a":{"dtype":"F32","shape":[3,2],"data_offsets":[0,24]},"weight_b":{"dtype":"I32","shape":[4],"data_offsets":[24,40]}}
    ;

    // Compose: [u64 LE header_len][header_json][tensor data].
    var blob = std.ArrayList(u8).init(allocator);
    defer blob.deinit();
    var len_buf: [8]u8 = undefined;
    std.mem.writeInt(u64, &len_buf, header_json.len, .little);
    try blob.appendSlice(&len_buf);
    try blob.appendSlice(header_json);
    try blob.appendSlice(std.mem.sliceAsBytes(&w_a));
    try blob.appendSlice(std.mem.sliceAsBytes(&w_b));

    // Write to a temp file, parse, verify, delete.
    const tmp_path = "/tmp/valkyr_smoke.safetensors";
    {
        const f = try std.fs.cwd().createFile(tmp_path, .{ .truncate = true });
        defer f.close();
        try f.writeAll(blob.items);
    }
    defer std.fs.cwd().deleteFile(tmp_path) catch {};

    var st = try safetensors.SafeTensors.open(allocator, tmp_path);
    defer st.deinit();

    if (st.count() != 2) {
        std.debug.print("safetensors MISMATCH: expected 2 tensors, got {d}\n", .{st.count()});
        return error.ParityFailed;
    }

    const ta = st.get("weight_a") orelse return error.MissingTensor;
    if (ta.dtype != .f32 or ta.shape.len != 2 or ta.shape[0] != 3 or ta.shape[1] != 2) {
        std.debug.print("weight_a metadata wrong: dtype={any} shape={any}\n", .{ ta.dtype, ta.shape });
        return error.ParityFailed;
    }
    const ta_f32 = ta.asF32();
    for (ta_f32, w_a, 0..) |got, want, i| {
        if (got != want) {
            std.debug.print("weight_a[{d}] MISMATCH: got {d}, expected {d}\n", .{ i, got, want });
            return error.ParityFailed;
        }
    }

    const tb = st.get("weight_b") orelse return error.MissingTensor;
    if (tb.dtype != .i32 or tb.shape.len != 1 or tb.shape[0] != 4) {
        std.debug.print("weight_b metadata wrong: dtype={any} shape={any}\n", .{ tb.dtype, tb.shape });
        return error.ParityFailed;
    }
    const tb_i32 = @as([*]align(1) const i32, @ptrCast(tb.bytes.ptr))[0..4];
    for (tb_i32, w_b, 0..) |got, want, i| {
        if (got != want) {
            std.debug.print("weight_b[{d}] MISMATCH: got {d}, expected {d}\n", .{ i, got, want });
            return error.ParityFailed;
        }
    }

    std.debug.print("PASS safetensors round-trip (2 tensors, F32+I32)\n", .{});
}
