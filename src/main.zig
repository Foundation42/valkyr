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
const train_runner = @import("train/runner.zig");
const train_runner_n = @import("train/runner_n.zig");
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

/// One entry from a `--prompts` TSV: `<label>\t<prompt>` per line.
/// Both fields are slices into the file's read buffer (LoadedPrompts);
/// they outlive only as long as that buffer is held.
const PromptEntry = struct {
    label: []const u8,
    text: []const u8,
};

/// Output of `loadPromptsTsv`: the heap-allocated file content and an
/// `entries` slice referencing into it. Caller frees both.
const LoadedPrompts = struct {
    buf: []u8,
    entries: []PromptEntry,
};

/// Read a `<label>\t<prompt>` TSV from disk. One entry per line.
/// Empty lines and lines starting with `#` are skipped. Lines without
/// a tab return error.MalformedPromptsLine.
fn loadPromptsTsv(gpa: std.mem.Allocator, path: []const u8) !LoadedPrompts {
    const file = try std.fs.cwd().openFile(path, .{});
    defer file.close();
    const stat = try file.stat();
    const buf = try gpa.alloc(u8, stat.size);
    errdefer gpa.free(buf);
    _ = try file.readAll(buf);

    var entries = std.ArrayList(PromptEntry).init(gpa);
    errdefer entries.deinit();

    var it = std.mem.splitScalar(u8, buf, '\n');
    while (it.next()) |line_raw| {
        const line = std.mem.trimRight(u8, line_raw, "\r");
        if (line.len == 0) continue;
        if (line[0] == '#') continue;
        const tab = std.mem.indexOfScalar(u8, line, '\t') orelse return error.MalformedPromptsLine;
        const label = line[0..tab];
        const text = line[tab + 1 ..];
        if (label.len == 0 or text.len == 0) return error.MalformedPromptsLine;
        try entries.append(.{ .label = label, .text = text });
    }
    return .{ .buf = buf, .entries = try entries.toOwnedSlice() };
}

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    const args = try std.process.argsAlloc(allocator);
    defer std.process.argsFree(allocator, args);

    if (args.len >= 2 and std.mem.eql(u8, args[1], "--list")) {
        try runList(allocator);
        return;
    }
    if (args.len >= 3 and std.mem.eql(u8, args[1], "--inspect")) {
        const dir = try hf_cache.resolveModelArg(allocator, args[2]);
        defer allocator.free(dir);
        try runInspect(allocator, dir);
        return;
    }
    if (args.len >= 3 and std.mem.eql(u8, args[1], "--load")) {
        const dir = try hf_cache.resolveModelArg(allocator, args[2]);
        defer allocator.free(dir);
        try runLoad(allocator, dir);
        return;
    }
    if (args.len >= 3 and std.mem.eql(u8, args[1], "--config")) {
        const dir = try hf_cache.resolveModelArg(allocator, args[2]);
        defer allocator.free(dir);
        try runConfig(allocator, dir);
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
        try runDumpEmbed(allocator, args[2], token_id);
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
        try runGpuGen(allocator, args[2], token_id);
        return;
    }
    if (args.len >= 4 and std.mem.eql(u8, args[1], "--gpu-gen-qwen35")) {
        const token_id = try std.fmt.parseInt(u32, args[3], 10);
        try runGpuGenQwen35(allocator, args[2], token_id);
        return;
    }
    if (args.len >= 5 and std.mem.eql(u8, args[1], "--gpu-gen-many")) {
        const token_id = try std.fmt.parseInt(u32, args[3], 10);
        const n_tokens = try std.fmt.parseInt(usize, args[4], 10);
        try runGpuGenMany(allocator, args[2], token_id, n_tokens);
        return;
    }
    if (args.len >= 4 and std.mem.eql(u8, args[1], "--encode")) {
        const dir = try hf_cache.resolveModelArg(allocator, args[2]);
        defer allocator.free(dir);
        try runEncode(allocator, dir, args[3]);
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
        try runGpuGenTq4V(allocator, args[2], token_id);
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
        try runBench(allocator, dir, n);
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
        try runSessionSmoke(allocator, args[2], prompt, smoke_max_new, budget, precision);
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
        try runSessionMessages(allocator, args[2], smoke_max_new, precision);
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
        try runRunnerSmoke(allocator, args[2], smoke_max_new, precision, threaded);
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
        try runServe(allocator, args[2], public_id orelse args[2], bind_addr, port, max_new, precision);
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
        var batch_prompts: ?[]const PromptEntry = null;
        defer if (batch_prompts) |bp| allocator.free(bp);
        if (prompts_tsv) |path| {
            const loaded = try loadPromptsTsv(allocator, path);
            batch_buf = loaded.buf;
            batch_prompts = loaded.entries;
            std.debug.print("--prompts loaded {d} entries from {s}\n", .{ loaded.entries.len, path });
        }

        if (cfg.family.isHybrid()) {
            try runChatQwen35(allocator, dir, user_msg, sp, seed, tq4v, precision, probe_path, batch_prompts, probe_prefix, max_new);
        } else {
            try runChat(allocator, dir, user_msg, sp, seed, tq4v, precision, probe_path, batch_prompts, probe_prefix, max_new);
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
    try runGpuMatmulV2Smoke(allocator);
    try runEmbeddedAttachSmoke(allocator);
    try runEmbeddedRecorderSmoke(allocator);
    try runGpuMatmulQ4_0Smoke(allocator);
    try runGpuMatmulQ4_KSmoke(allocator);
    try runGpuRmsnormSmoke(allocator);
    try runGpuGegluSmoke(allocator);
    try runGpuRopeSmoke(allocator);
    try runGpuRopePartialSmoke(allocator);
    try runGpuSplitQGateSmoke(allocator);
    try runGpuSigmoidMulSmoke(allocator);
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

// ── --list: enumerate cached HF models ─────────────────────────────
//
// Walks ~/.cache/huggingface/hub (env-overridable via HF_HUB_CACHE /
// HF_HOME / XDG_CACHE_HOME) and prints one row per cached model with
// its architecture, on-disk size, and a marker column showing whether
// valkyr can load it. Designed to pair with `--chat <id>`: if a row
// shows `[OK]`, you can chat the model directly via its id.

fn runList(gpa: std.mem.Allocator) !void {
    const stdout = std.io.getStdOut().writer();

    const root = try hf_cache.cacheRoot(gpa);
    defer gpa.free(root);
    try stdout.print("HF cache: {s}\n\n", .{root});

    const models = try hf_cache.listModels(gpa);
    defer {
        for (models) |entry| {
            var m = entry;
            m.deinit(gpa);
        }
        gpa.free(models);
    }
    if (models.len == 0) {
        try stdout.print("(no models cached — try `hf download <org/name>`)\n", .{});
        return;
    }

    // Sort: supported first, then by id alphabetically. Two-key sort
    // via std.mem.sort with a stable comparator.
    std.mem.sort(hf_cache.ModelInfo, @constCast(models), {}, struct {
        fn lt(_: void, a: hf_cache.ModelInfo, b: hf_cache.ModelInfo) bool {
            if (a.supported != b.supported) return a.supported;
            return std.mem.lessThan(u8, a.id, b.id);
        }
    }.lt);

    // Width-fit columns for readability.
    var max_id_w: usize = 5;
    var max_arch_w: usize = 12;
    for (models) |m| {
        if (m.id.len > max_id_w) max_id_w = m.id.len;
        if (m.architecture.len > max_arch_w) max_arch_w = m.architecture.len;
    }
    if (max_id_w > 60) max_id_w = 60;
    if (max_arch_w > 36) max_arch_w = 36;

    try stdout.print("  {s: <60}  {s: <36}  {s: >9}  {s}\n", .{
        "model id (`--chat <this>`)",
        "architecture",
        "size",
        "status",
    });
    try stdout.print("  {s:->60}  {s:->36}  {s:->9}  {s:->8}\n", .{ "", "", "", "" });

    var size_buf: [32]u8 = undefined;
    for (models) |m| {
        const size_str = try hf_cache.formatSize(m.bytes, &size_buf);
        const status: []const u8 = if (m.supported) "[OK]" else "[?]";
        try stdout.print("  {s: <60}  {s: <36}  {s: >9}  {s}\n", .{
            m.id,
            m.architecture,
            size_str,
            status,
        });
    }

    var supported_count: usize = 0;
    for (models) |m| if (m.supported) {
        supported_count += 1;
    };
    try stdout.print("\n{d}/{d} models supported by valkyr's loader.\n", .{ supported_count, models.len });
}

// ── inspect: dump the tensor inventory of a real .safetensors file ──

fn runInspect(allocator: std.mem.Allocator, path: []const u8) !void {
    const t0 = std.time.nanoTimestamp();
    var st = try safetensors.SafeTensors.open(allocator, path);
    defer st.deinit();
    const t1 = std.time.nanoTimestamp();
    const open_ms = @as(f64, @floatFromInt(t1 - t0)) / 1_000_000.0;

    const stdout = std.io.getStdOut().writer();
    try stdout.print("File: {s}\n", .{path});
    try stdout.print("Tensors: {d}    Open+parse: {d:.2} ms\n", .{ st.count(), open_ms });
    try stdout.print("Mapping: {d:.2} GiB\n", .{
        @as(f64, @floatFromInt(st.mapping.len)) / (1024.0 * 1024.0 * 1024.0),
    });
    try stdout.print("\n", .{});

    // Walk in alphabetical order so the output is reproducible across
    // hashmap iteration orders.
    var names = std.ArrayList([]const u8).init(allocator);
    defer names.deinit();
    var it = st.by_name.keyIterator();
    while (it.next()) |k| try names.append(k.*);
    std.mem.sort([]const u8, names.items, {}, struct {
        fn lessThan(_: void, a: []const u8, b: []const u8) bool {
            return std.mem.order(u8, a, b) == .lt;
        }
    }.lessThan);

    var totals = [_]u64{0} ** @typeInfo(safetensors.Dtype).@"enum".fields.len;
    for (names.items) |name| {
        const t = st.get(name).?;
        totals[@intFromEnum(t.dtype)] += t.bytes.len;
        try stdout.print("  {s:<10} {s:<55} ", .{ @tagName(t.dtype), name });
        try stdout.print("[", .{});
        for (t.shape, 0..) |d, i| {
            if (i > 0) try stdout.print(", ", .{});
            try stdout.print("{d}", .{d});
        }
        try stdout.print("]  {d:.2} MiB\n", .{
            @as(f64, @floatFromInt(t.bytes.len)) / (1024.0 * 1024.0),
        });
    }

    try stdout.print("\nDtype totals:\n", .{});
    inline for (@typeInfo(safetensors.Dtype).@"enum".fields) |f| {
        const idx: usize = f.value;
        if (totals[idx] > 0) {
            try stdout.print("  {s:<6} {d:.2} MiB\n", .{
                f.name,
                @as(f64, @floatFromInt(totals[idx])) / (1024.0 * 1024.0),
            });
        }
    }
}

// ── load: parse config + open shards + bind layer weights ────────────

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

fn runConfig(allocator: std.mem.Allocator, dir_path: []const u8) !void {
    // Parse `config.json` only — no safetensors loader. Useful for
    // validating new architectures' config plumbing in isolation before
    // wiring up the rest of the loader.
    const cfg_path = try std.fs.path.join(allocator, &.{ dir_path, "config.json" });
    defer allocator.free(cfg_path);
    const cfg = try config_mod.Config.loadFromFile(allocator, cfg_path);
    const stdout = std.io.getStdOut().writer();
    try stdout.print("Parsed {s}\n\n", .{cfg_path});
    try cfg.print(stdout);
    try stdout.print("\n[in_proj_qkv conv dim] {d}\n", .{cfg.linearAttnConvDim()});
    if (cfg.family.isHybrid()) {
        try stdout.print("[layer_types[0..8]]    ", .{});
        const n = @min(cfg.num_hidden_layers, 8);
        for (cfg.layer_types[0..n]) |t| try stdout.print("{s} ", .{@tagName(t)});
        try stdout.print("\n", .{});
    }
}

fn runLoad(allocator: std.mem.Allocator, dir_path: []const u8) !void {
    const t0 = std.time.nanoTimestamp();
    var model = try model_mod.Model.load(allocator, dir_path);
    defer model.deinit();
    const t1 = std.time.nanoTimestamp();
    const load_ms = @as(f64, @floatFromInt(t1 - t0)) / 1_000_000.0;

    const stdout = std.io.getStdOut().writer();
    try stdout.print("Loaded {s}\n", .{dir_path});
    try stdout.print("Load time: {d:.2} ms ({d} shard(s), {d} tensors)\n\n", .{
        load_ms, model.shards.shards.len, model.shards.count(),
    });
    try model.config.print(stdout);
    try stdout.print("\nLM head: {s}\n", .{
        if (model.isLmHeadTied()) "tied (shares embed_tokens)" else "separate (lm_head.weight)",
    });

    // Sanity: walk every layer once and confirm we can read the first
    // and last byte of each weight from the mmap. If a shard's offset
    // table is wrong, that page-faults; if it's right, this is free.
    // For hybrid models the per-layer fields differ between full and
    // linear-attention blocks — touch whichever is present.
    var bytes_touched: usize = 0;
    for (model.layers) |layer| {
        // FFN trio + the two LayerNorms are always present.
        inline for (.{
            "input_layernorm",          "post_attention_layernorm",
            "gate_proj",                "up_proj",                  "down_proj",
        }) |fname| {
            const t = @field(layer, fname);
            if (t.bytes.len > 0) {
                bytes_touched +%= @intCast(t.bytes[0]);
                bytes_touched +%= @intCast(t.bytes[t.bytes.len - 1]);
            }
        }
        // Optional per-flavor tensors. `inline for` on a tuple of field
        // names lets us walk them uniformly, dereference the optional,
        // and skip nulls.
        inline for (.{
            "q_proj", "k_proj", "v_proj", "o_proj", "q_norm", "k_norm",
            "in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a",
            "conv1d_weight", "A_log", "dt_bias", "ssm_norm_weight", "out_proj",
        }) |fname| {
            if (@field(layer, fname)) |t| {
                if (t.bytes.len > 0) {
                    bytes_touched +%= @intCast(t.bytes[0]);
                    bytes_touched +%= @intCast(t.bytes[t.bytes.len - 1]);
                }
            }
        }
    }
    bytes_touched +%= @intCast(model.embed_tokens.bytes[0]);
    bytes_touched +%= @intCast(model.final_norm.bytes[0]);
    bytes_touched +%= @intCast(model.lm_head.bytes[0]);
    // The xor folds the first/last byte of every weight into one word —
    // a cheap way to force the OS to actually touch each tensor page.
    // Print it so the optimizer can't elide the loop.
    try stdout.print("\nPASS load (touch checksum: 0x{x})\n", .{bytes_touched});
}

// ── dump-embed: pull a token's row from embed_tokens, bf16 → fp32 ────

fn runDumpEmbed(allocator: std.mem.Allocator, dir_path: []const u8, token_id: u32) !void {
    var model = try model_mod.Model.load(allocator, dir_path);
    defer model.deinit();

    const cfg = model.config;
    if (token_id >= cfg.vocab_size) {
        std.debug.print("token id {d} out of range (vocab_size={d})\n", .{ token_id, cfg.vocab_size });
        return error.OutOfRange;
    }

    if (model.embed_tokens.dtype != .bf16) {
        std.debug.print("expected bf16 embed_tokens; got {s}\n", .{@tagName(model.embed_tokens.dtype)});
        return error.UnexpectedDtype;
    }

    const stdout = std.io.getStdOut().writer();
    try stdout.print("token {d}, hidden_size={d} — first values + stats\n", .{ token_id, cfg.hidden_size });

    // Slice the row for this token. embed_tokens is [vocab_size, hidden_size]
    // row-major, so row `token_id` starts at `token_id * hidden_size *
    // bytes_per_elem` from the tensor's byte base.
    const elem_bytes = model.embed_tokens.dtype.elemSize();
    const row_bytes = cfg.hidden_size * elem_bytes;
    const row_start = @as(usize, token_id) * row_bytes;
    const row_bf16 = dtype.asU16(model.embed_tokens.bytes[row_start .. row_start + row_bytes]);

    // Convert into a heap-allocated fp32 buffer.
    const row_f32 = try allocator.alloc(f32, cfg.hidden_size);
    defer allocator.free(row_f32);
    dtype.bf16SliceToF32(row_bf16, row_f32);

    // First N for inspection.
    const n_show: usize = @min(16, row_f32.len);
    for (row_f32[0..n_show], 0..) |v, i| try stdout.print("  [{d:>4}] {d:.6}\n", .{ i, v });

    // Stats over the whole row.
    var min_v: f32 = std.math.inf(f32);
    var max_v: f32 = -std.math.inf(f32);
    var sum: f64 = 0;
    var nan_count: usize = 0;
    var inf_count: usize = 0;
    for (row_f32) |v| {
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
        sum += v;
    }
    const mean = sum / @as(f64, @floatFromInt(row_f32.len - nan_count - inf_count));
    try stdout.print("\nstats: min={d:.6} max={d:.6} mean={d:.6} nan={d} inf={d}\n", .{
        min_v, max_v, mean, nan_count, inf_count,
    });
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

// ── session-smoke: headless exercise of valkyr.session.Session ───────
//
// Same code path matryoshka's ai_demo runs in-engine, minus the GUI.
// Loads the model via `loader.loadGpuModel`, builds a `Session`, feeds
// the prompt, and ticks until done while streaming generated tokens
// to stdout. Used to validate dense + hybrid family support through
// the embed surface — `--chat` doesn't go through Session at all.

const SessionSmokeOnTokenCtx = struct {
    stdout: std.fs.File.Writer,
    family: config_mod.Family,
};

fn sessionSmokeOnToken(user: ?*anyopaque, tok_id: u32, decoded: []const u8) void {
    _ = tok_id;
    const ctx: *SessionSmokeOnTokenCtx = @ptrCast(@alignCast(user.?));
    // `decoded` is already display-ready: tokenizer.decodeForDisplay
    // resolved SentencePiece ▁ / byte-level Ġ / hex-byte fallback
    // tokens before this callback fired.
    ctx.stdout.print("{s}", .{decoded}) catch return;
}

fn runSessionSmoke(
    gpa: std.mem.Allocator,
    model_arg: []const u8,
    prompt: []const u8,
    max_new: u32,
    budget: session_mod.Budget,
    precision: gpu_model.Precision,
) !void {
    const stdout = std.io.getStdOut().writer();
    var ctx = try vk.Context.init(gpa);
    defer ctx.deinit();

    try stdout.print("device: {s}\n", .{ctx.deviceName()});
    try stdout.print("loading model {s}...\n", .{model_arg});

    const t0 = std.time.nanoTimestamp();
    var loaded = try loader.loadGpuModel(gpa, &ctx, model_arg, .{ .precision = precision });
    defer loaded.deinit(ctx.device);
    const t1 = std.time.nanoTimestamp();
    const cfg = loaded.config();
    try stdout.print(
        "loaded in {d:.1} s — family={s} layers={d} hybrid={}\n",
        .{ @as(f64, @floatFromInt(t1 - t0)) / 1e9, @tagName(cfg.family), cfg.num_hidden_layers, cfg.family.isHybrid() },
    );

    var on_token_ctx = SessionSmokeOnTokenCtx{ .stdout = stdout, .family = cfg.family };

    var sess = try session_mod.Session.init(gpa, &ctx, &loaded.gpu_model, &loaded.tokenizer, .{
        .budget = budget,
        .max_new_tokens = max_new,
        .on_token = sessionSmokeOnToken,
        .on_token_user = &on_token_ctx,
        .max_pos = 1024,
    });
    defer sess.deinit();
    try sess.appendPrompt(prompt);

    try stdout.print("\nprompt: \"{s}\"\n", .{prompt});
    try stdout.print("response: ", .{});

    // Recorder sized for the worst-case tick in pure `--budget-us` mode:
    // a single tick may record every layer + sample step before the time
    // gate fires. Layer-mode default (8 layers) fits in 512/4096 easily;
    // bump here so the smoke harness can also exercise unbounded-layer
    // time mode without exhausting the descriptor pool.
    var rec = try gpu_recorder.Recorder.init(&ctx, 4096, 16384);
    defer rec.deinit();

    const decode_t0 = std.time.nanoTimestamp();
    var ticks: u32 = 0;
    var sum_elapsed_us: u64 = 0;
    var sum_residual_signed: i64 = 0;
    var max_elapsed_us: u64 = 0;
    while (!sess.isDone() and ticks < 4096) : (ticks += 1) {
        try rec.reset();
        try rec.begin();
        const r = try sess.tickFrame(&rec);
        try rec.endAndSubmit();
        sum_elapsed_us += r.elapsed_us;
        sum_residual_signed += r.residual_us;
        if (r.elapsed_us > max_elapsed_us) max_elapsed_us = r.elapsed_us;
    }
    const decode_t1 = std.time.nanoTimestamp();
    const tokens = sess.generatedTokens();
    const wall_s = @as(f64, @floatFromInt(decode_t1 - decode_t0)) / 1e9;
    try stdout.print(
        "\n[{d} tok in {d:.0} ms over {d} ticks, {d:.1} tok/s]\n",
        .{ tokens.len, wall_s * 1000.0, ticks, @as(f64, @floatFromInt(tokens.len)) / wall_s },
    );
    if (ticks > 0) {
        const avg_elapsed = @as(f64, @floatFromInt(sum_elapsed_us)) / @as(f64, @floatFromInt(ticks));
        const avg_residual = @as(f64, @floatFromInt(sum_residual_signed)) / @as(f64, @floatFromInt(ticks));
        try stdout.print(
            "[budget={s}  per-tick: avg_elapsed={d:.1} µs  max_elapsed={d} µs  avg_residual={d:.1} µs]\n",
            .{ @tagName(budget), avg_elapsed, max_elapsed_us, avg_residual },
        );
    }
}

// ── session-messages: multi-turn exercise of Session.appendMessages ──

fn runSessionMessages(
    gpa: std.mem.Allocator,
    model_arg: []const u8,
    max_new: u32,
    precision: gpu_model.Precision,
) !void {
    const stdout = std.io.getStdOut().writer();
    var ctx = try vk.Context.init(gpa);
    defer ctx.deinit();

    try stdout.print("device: {s}\n", .{ctx.deviceName()});
    try stdout.print("loading model {s}...\n", .{model_arg});

    const t0 = std.time.nanoTimestamp();
    var loaded = try loader.loadGpuModel(gpa, &ctx, model_arg, .{ .precision = precision });
    defer loaded.deinit(ctx.device);
    const t1 = std.time.nanoTimestamp();
    const cfg = loaded.config();
    try stdout.print(
        "loaded in {d:.1} s — family={s} layers={d} hybrid={}\n",
        .{ @as(f64, @floatFromInt(t1 - t0)) / 1e9, @tagName(cfg.family), cfg.num_hidden_layers, cfg.family.isHybrid() },
    );

    // Hardcoded multi-turn fixture matching what an OpenAI client
    // would post to `/v1/chat/completions`. Generation continues
    // from the trailing open assistant header.
    const messages = [_]session_mod.Message{
        .{ .role = .user, .content = "Hi" },
        .{ .role = .assistant, .content = "Hello! How can I help?" },
        .{ .role = .user, .content = "Tell me a short joke." },
    };

    var on_token_ctx = SessionSmokeOnTokenCtx{ .stdout = stdout, .family = cfg.family };

    // Stop on the family's end_of_turn token so a coherent reply ends
    // cleanly instead of running into max_new_tokens.
    var stop_tokens_buf: [1]u32 = undefined;
    var sess = try session_mod.Session.init(gpa, &ctx, &loaded.gpu_model, &loaded.tokenizer, .{
        .budget = .{ .layers = 8 },
        .max_new_tokens = max_new,
        .on_token = sessionSmokeOnToken,
        .on_token_user = &on_token_ctx,
        .max_pos = 2048,
        .stop_tokens = &.{},
    });
    defer sess.deinit();
    stop_tokens_buf[0] = sess.template.end_of_turn;
    sess.cfg.stop_tokens = stop_tokens_buf[0..1];

    try sess.appendMessages(&messages);

    try stdout.print("\n--- composed prompt: {d} tokens ---\n", .{sess.prompt_q.items.len});
    try stdout.print("response: ", .{});

    var rec = try gpu_recorder.Recorder.init(&ctx, 4096, 16384);
    defer rec.deinit();

    var ticks: u32 = 0;
    while (!sess.isDone() and ticks < 4096) : (ticks += 1) {
        try rec.reset();
        try rec.begin();
        _ = try sess.tickFrame(&rec);
        try rec.endAndSubmit();
    }
    try stdout.print("\n", .{});
}

// ── runner-smoke: end-to-end InferenceRunner inline-mode exercise ─

const inference_queue = @import("inference/queue.zig");
const inference_arena = @import("inference/arena.zig");
const inference_proto = @import("inference/proto.zig");
const inference_runner = @import("inference/runner.zig");

fn runRunnerSmoke(
    gpa: std.mem.Allocator,
    model_arg: []const u8,
    max_new: u32,
    precision: gpu_model.Precision,
    threaded: bool,
) !void {
    const stdout = std.io.getStdOut().writer();
    var ctx = try vk.Context.init(gpa);
    defer ctx.deinit();

    try stdout.print("device: {s}\n", .{ctx.deviceName()});
    try stdout.print("loading model {s}...\n", .{model_arg});

    const t0 = std.time.nanoTimestamp();
    var loaded = try loader.loadGpuModel(gpa, &ctx, model_arg, .{ .precision = precision });
    defer loaded.deinit(ctx.device);
    const t1 = std.time.nanoTimestamp();
    const cfg = loaded.config();
    try stdout.print(
        "loaded in {d:.1} s — family={s} layers={d} hybrid={}\n",
        .{ @as(f64, @floatFromInt(t1 - t0)) / 1e9, @tagName(cfg.family), cfg.num_hidden_layers, cfg.family.isHybrid() },
    );

    // Build a Session that the runner will borrow. Same shape as
    // --session-messages but the runner owns the on_token wiring +
    // budget + stops, so we leave them at defaults here.
    var sess = try session_mod.Session.init(gpa, &ctx, &loaded.gpu_model, &loaded.tokenizer, .{
        .max_pos = 2048,
    });
    defer sess.deinit();

    var rec = try gpu_recorder.Recorder.init(&ctx, 4096, 16384);
    defer rec.deinit();

    var runner = try inference_runner.InferenceRunner.initBorrow(gpa, &sess, &rec, .{
        .default_max_tokens = max_new,
    });
    defer runner.deinit();

    if (threaded) try runner.start();

    // Same 3-turn fixture as --session-messages.
    const messages = [_]chat_template_mod.Message{
        .{ .role = .user, .content = "Hi" },
        .{ .role = .assistant, .content = "Hello! How can I help?" },
        .{ .role = .user, .content = "Tell me a short joke." },
    };

    // Stop-string sanity: any of these terminates with reason=stop.
    // Most won't trigger on the joke fixture; "guts!" matches the
    // Qwen3 4B response, "stick" matches Gemma's. Demonstrates the
    // suffix-match path without making the smoke flaky.
    const stop_strings = [_][]const u8{ "guts!", "stick.", "</response>" };

    try runner.submit(.{ .chat = .{
        .corr = 1,
        .messages = &messages,
        .max_tokens = max_new,
        .stop_strings = &stop_strings,
        .deadline_ns = 30 * std.time.ns_per_s, // 30s — plenty of headroom
    } });

    try stdout.print("response: ", .{});

    var token_count: u32 = 0;
    var saw_finish = false;
    var prefill_tokens: u32 = 0;
    var finish_reason: inference_proto.FinishReason = .stop;
    var elapsed_ns: u64 = 0;

    var ticks: u32 = 0;
    while (!saw_finish and ticks < 8192) : (ticks += 1) {
        // In inline mode the main thread drives ticks. In threaded
        // mode the worker drives ticks itself; we just sleep + poll.
        if (threaded) {
            std.time.sleep(500_000); // 500µs poll interval
        } else {
            try runner.tickInline();
        }
        while (runner.pollEvent()) |ev| {
            switch (ev.kind) {
                .accepted => |a| prefill_tokens = a.prefill_tokens,
                .token => |t| {
                    const text = runner.resolve(t.decoded);
                    try stdout.writeAll(text);
                    token_count += 1;
                },
                .arena_swap => {},
                .finish => |f| {
                    saw_finish = true;
                    finish_reason = f.reason;
                    elapsed_ns = f.elapsed_ns;
                },
                .err => |e| {
                    try stdout.print("\n[runner err: {s}]\n", .{e.msg});
                    saw_finish = true;
                },
            }
        }
    }
    try stdout.print(
        "\n[mode={s}  prefill={d}  completion={d}  reason={s}  elapsed={d:.1} ms  ticks={d}]\n",
        .{
            if (threaded) "threaded" else "inline",
            prefill_tokens, token_count, @tagName(finish_reason),
            @as(f64, @floatFromInt(elapsed_ns)) / 1e6, ticks,
        },
    );
    // Threaded mode: shutdown joins the worker before deinit's
    // shutdown call (which would also work; just being explicit).
    if (threaded) runner.shutdown();
}

// ── serve: OpenAI-compatible HTTP endpoint ─────────────────────────

const server_mod = @import("server/server.zig");

fn runServe(
    gpa: std.mem.Allocator,
    model_arg: []const u8,
    public_id: []const u8,
    bind_addr: []const u8,
    port: u16,
    max_new: u32,
    precision: gpu_model.Precision,
) !void {
    const stdout = std.io.getStdOut().writer();
    var ctx = try vk.Context.init(gpa);
    defer ctx.deinit();

    try stdout.print("device: {s}\n", .{ctx.deviceName()});
    try stdout.print("loading model {s}...\n", .{model_arg});

    const t0 = std.time.nanoTimestamp();
    var loaded = try loader.loadGpuModel(gpa, &ctx, model_arg, .{ .precision = precision });
    defer loaded.deinit(ctx.device);
    const t1 = std.time.nanoTimestamp();
    const cfg = loaded.config();
    try stdout.print(
        "loaded in {d:.1} s — family={s} layers={d} hybrid={}\n",
        .{ @as(f64, @floatFromInt(t1 - t0)) / 1e9, @tagName(cfg.family), cfg.num_hidden_layers, cfg.family.isHybrid() },
    );

    var sess = try session_mod.Session.init(gpa, &ctx, &loaded.gpu_model, &loaded.tokenizer, .{
        .max_pos = 4096,
    });
    defer sess.deinit();

    // Recorder sized for serve's likely usage: pure-µs budget mode
    // (default for runner) + worst-case big-model layer counts. The
    // runner default is `.either{layers=8, µs=5000}` so the layer
    // cap keeps us inside 4096 sets / 16384 descriptors.
    var rec = try gpu_recorder.Recorder.init(&ctx, 4096, 16384);
    defer rec.deinit();

    var runner = try inference_runner.InferenceRunner.initBorrow(gpa, &sess, &rec, .{
        .default_max_tokens = max_new,
    });
    defer runner.deinit();
    try runner.start();

    var srv = try server_mod.Server.init(gpa, &runner, .{
        .bind_address = bind_addr,
        .port = port,
        .model_id = public_id,
        .default_max_tokens = max_new,
    });
    defer srv.deinit();
    try srv.start();

    try stdout.print(
        "server ready: POST http://{s}:{d}/v1/chat/completions  •  GET /v1/models\n" ++
            "model id (validate against request.model): {s}\n" ++
            "Ctrl-C to stop.\n",
        .{ bind_addr, port, public_id },
    );

    // Park forever — Ctrl-C kills the process. We don't install a
    // signal handler; the OS reaping is fine for v0.
    while (true) {
        std.time.sleep(std.time.ns_per_s);
    }
}

// ── bench: cold / warm forward timing on Gemma 2B IT ───────────────

fn runBench(gpa: std.mem.Allocator, dir_path: []const u8, n_steps: usize) !void {
    var cpu = try model_mod.Model.load(gpa, dir_path);
    defer cpu.deinit();
    const cfg = cpu.config;

    var ctx = try vk.Context.init(gpa);
    defer ctx.deinit();

    const stdout = std.io.getStdOut().writer();
    try stdout.print("device: {s}\n", .{ctx.deviceName()});
    try stdout.print("model: {s}    layers={d}  hidden={d}  vocab={d}\n", .{
        @tagName(cfg.family), cfg.num_hidden_layers, cfg.hidden_size, cfg.vocab_size,
    });

    const t_up0 = std.time.nanoTimestamp();
    var gm = try gpu_model.GpuModel.upload(gpa, &ctx, &cpu, .bf16_matmul);
    defer gm.deinit(ctx.device);
    const t_up1 = std.time.nanoTimestamp();
    const upload_ms = @as(f64, @floatFromInt(t_up1 - t_up0)) / 1_000_000.0;
    try stdout.print("upload (bf16 matmul): {d:.0} ms\n\n", .{upload_ms});

    const max_pos: usize = @max(n_steps + 16, 128);
    var sc = try gpu_scratch.GpuScratch.init(&ctx, cfg, max_pos);
    defer sc.deinit(ctx.device);
    var kv = try gpu_scratch.GpuKvCache.init(gpa, &ctx, cfg, max_pos);
    defer kv.deinit(ctx.device);

    var k = try runtime.ChatKernels.init(&ctx, gm.precision, cfg.family);
    defer k.deinit();

    var rec = try gpu_recorder.Recorder.init(&ctx, 512, 2048);
    defer rec.deinit();

    const logits = try gpa.alloc(f32, cfg.vocab_size);
    defer gpa.free(logits);

    const bos = tok_bos: {
        // Use BOS token id from config when set; otherwise fall back to 2
        // (Gemma's bos id) so the bench works without a tokenizer load.
        if (cfg.bos_token_id) |b| break :tok_bos b;
        break :tok_bos 2;
    };

    // Time each forward, advancing position each step. Position 0 uses
    // bos; subsequent steps feed back the argmax (a real autoregressive
    // greedy run, not a microbenchmark on a fixed input).
    const samples = try gpa.alloc(f64, n_steps);
    defer gpa.free(samples);
    var current: u32 = bos;

    for (0..n_steps) |step| {
        if (step > 0) try rec.reset();
        try rec.begin();
        try recordForwardStep(&rec, &sc, &gm, &kv, cfg, &k, step, current, null, true);
        const t0 = std.time.nanoTimestamp();
        try rec.endAndSubmit();
        const t1 = std.time.nanoTimestamp();
        samples[step] = @as(f64, @floatFromInt(t1 - t0)) / 1_000_000.0;

        try sc.logits.readBack(&ctx, f32, logits);
        current = @intCast(cpu_forward.argmax(logits));
    }

    // Stats. The first sample is the cold one (pipeline compile + cold
    // caches). Steady-state stats use samples[1..].
    const cold = samples[0];
    var warm_sum: f64 = 0;
    var warm_min: f64 = std.math.inf(f64);
    var warm_max: f64 = 0;
    for (samples[1..]) |s| {
        warm_sum += s;
        if (s < warm_min) warm_min = s;
        if (s > warm_max) warm_max = s;
    }
    const warm_mean = warm_sum / @as(f64, @floatFromInt(samples.len - 1));

    // p50 and p99 via a sorted copy (cheap at n_steps ≤ a few hundred).
    const sorted = try gpa.alloc(f64, samples.len - 1);
    defer gpa.free(sorted);
    @memcpy(sorted, samples[1..]);
    std.mem.sort(f64, sorted, {}, std.sort.asc(f64));
    const p50 = sorted[sorted.len / 2];
    const p99_idx: usize = @min(sorted.len - 1, (sorted.len * 99) / 100);
    const p99 = sorted[p99_idx];

    try stdout.print("forwards: {d} steps (positions 0..{d})\n", .{ n_steps, n_steps - 1 });
    try stdout.print("  cold (step 0)   : {d:.2} ms\n", .{cold});
    try stdout.print("  warm mean       : {d:.2} ms\n", .{warm_mean});
    try stdout.print("  warm median     : {d:.2} ms\n", .{p50});
    try stdout.print("  warm min/max    : {d:.2} / {d:.2} ms\n", .{ warm_min, warm_max });
    try stdout.print("  warm p99        : {d:.2} ms\n", .{p99});
    try stdout.print("  throughput      : {d:.1} tok/s (warm mean)\n", .{1000.0 / warm_mean});

    // Position-dependent cost: attention scoring grows linearly with
    // n_pos. Print early-vs-late timings to make it visible.
    if (n_steps >= 32) {
        var early: f64 = 0;
        for (samples[1..16]) |s| early += s;
        var late: f64 = 0;
        const start = samples.len - 16;
        for (samples[start..]) |s| late += s;
        try stdout.print("\n  pos-dependent cost (15-step rolling means):\n", .{});
        try stdout.print("    early (pos 1..15)         : {d:.2} ms/tok\n", .{early / 15.0});
        try stdout.print("    late  (pos {d}..{d}) : {d:.2} ms/tok\n", .{
            start, n_steps - 1, late / 16.0,
        });
    }
}


// ── chat: prompt prefill + generation, single-turn or REPL ─────────

fn runChat(
    gpa: std.mem.Allocator,
    dir_path: []const u8,
    single_msg: ?[]const u8,
    sample_params: cpu_forward.SampleParams,
    seed: u64,
    tq4v: bool,
    precision: gpu_model.Precision,
    probe_path: ?[]const u8,
    batch_prompts: ?[]const PromptEntry,
    probe_prefix: ?[]const u8,
    max_new: usize,
) !void {
    var cpu = try model_mod.Model.load(gpa, dir_path);
    defer cpu.deinit();
    const cfg = cpu.config;

    const tok_path = try std.fmt.allocPrint(gpa, "{s}/tokenizer.json", .{dir_path});
    defer gpa.free(tok_path);
    var tok = try tokenizer_mod.Tokenizer.loadFromFile(gpa, tok_path);
    defer tok.deinit();

    var ctx = try vk.Context.init(gpa);
    defer ctx.deinit();

    const stdout = std.io.getStdOut().writer();
    try stdout.print("device: {s}\n", .{ctx.deviceName()});
    try stdout.print("uploading weights ({s} matmul path)...\n", .{switch (precision) {
        .fp32_all => "fp32",
        .bf16_matmul => "bf16",
        .q4_0_matmul => "Q4_0",
        .q4_k_matmul => "Q4_K_M",
    }});
    const t_up0_g = std.time.nanoTimestamp();
    var gm = try gpu_model.GpuModel.upload(gpa, &ctx, &cpu, precision);
    defer gm.deinit(ctx.device);
    const t_up1_g = std.time.nanoTimestamp();
    try stdout.print("  upload took {d:.0} ms\n", .{@as(f64, @floatFromInt(t_up1_g - t_up0_g)) / 1_000_000.0});

    // KV cache sized for a generous chat. 2048 positions ≈ 18 layers ×
    // 2 (K + V) × 2048 × 256 × 4 bytes ≈ 72 MiB on disk space — fine.
    const max_pos: usize = 2048;
    var sc = try gpu_scratch.GpuScratch.init(&ctx, cfg, max_pos);
    defer sc.deinit(ctx.device);
    var kv = try gpu_scratch.GpuKvCache.init(gpa, &ctx, cfg, max_pos);
    defer kv.deinit(ctx.device);

    // TurboQuant V cache (asymmetric K=fp / V=TQ4) — only allocated
    // when --tq4v was passed. The fp32 kv buffer above is still used
    // for the K side regardless; we just substitute the V path.
    // Keeping both kv buffers allocated wastes a bit of V memory in
    // tq4v mode (kv.layers[*].v_cache goes unused) but means the
    // K-side wiring needs zero changes.
    var kv_tq4 = if (tq4v)
        try gpu_scratch.GpuKvCacheTq4.init(gpa, &ctx, cfg, max_pos)
    else
        null;
    defer if (kv_tq4) |*c| c.deinit(ctx.device);

    var k = try runtime.ChatKernels.init(&ctx, gm.precision, cfg.family);
    defer k.deinit();

    // TQ4 pack/unpack kernels — picked by head_dim, only built in
    // tq4v mode, kept allocated for the lifetime of the chat session.
    // 256 (Gemma 2B) and 128 (Qwen3) shader pairs are wired up; other
    // sizes would need their own .comp files and a new branch here.
    const tq_pack_spv: []align(4) const u8 = if (cfg.head_dim == 128)
        &shaders.tq4_pack_to_cache128
    else
        &shaders.tq4_pack_to_cache;
    const tq_unpack_spv: []align(4) const u8 = if (cfg.head_dim == 128)
        &shaders.tq4_unpack128
    else
        &shaders.tq4_unpack256;
    var tq_pack: ?pipeline.Kernel = if (tq4v)
        try pipeline.Kernel.init(&ctx, tq_pack_spv, 2, @sizeOf(Tq4PackPush))
    else
        null;
    defer if (tq_pack) |*kk| kk.deinit();
    var tq_unpack: ?pipeline.Kernel = if (tq4v)
        try pipeline.Kernel.init(&ctx, tq_unpack_spv, 2, 0)
    else
        null;
    defer if (tq_unpack) |*kk| kk.deinit();

    const tq4_hooks: ?Tq4VHooks = if (tq4v)
        Tq4VHooks{ .pack = &tq_pack.?, .unpack = &tq_unpack.?, .cache = &kv_tq4.? }
    else
        null;

    // Pool sizing — one full forward worth of dispatches per step.
    // Gemma 2B (18 layers) = ~345/step. Qwen3-4B (36 layers) ≈ 720/step
    // before counting q_norm/k_norm. We size proportionally with a
    // generous slack so future deeper models don't blow up here.
    const sets_per_step: u32 = @intCast(@max(@as(usize, 512), cfg.num_hidden_layers * 32));
    var rec = try gpu_recorder.Recorder.init(&ctx, sets_per_step, 2048);
    defer rec.deinit();

    // Resolve chat-template specials based on the model family. Gemma
    // uses `<bos>` + `<start_of_turn>` + `<end_of_turn>`; Qwen3 (ChatML)
    // uses `<|endoftext|>` as BOS-equivalent and `<|im_start|>` /
    // `<|im_end|>` as turn markers.
    const tmpl = try chat_template_mod.ChatTemplate.resolve(cfg.family, &tok);

    const logits = try gpa.alloc(f32, cfg.vocab_size);
    defer gpa.free(logits);
    // Sampling scratch: top-P needs 2× vocab worth (probs + index list).
    const sample_scratch = try gpa.alloc(f32, cfg.vocab_size * 2);
    defer gpa.free(sample_scratch);

    var prng = std.Random.DefaultPrng.init(seed);
    const rng = prng.random();

    // Pretty-print the sampling configuration so the user can see
    // what knobs are active.
    if (sample_params.temperature == 0.0 or sample_params.top_k == 1 or
        (sample_params.top_p >= 1.0 and sample_params.top_k == 0 and sample_params.temperature == 1.0))
    {
        // Greedy or trivially deterministic.
        try stdout.print("sampling: greedy\n", .{});
    } else {
        try stdout.print("sampling: temp={d} top_k={d} top_p={d} seed={d}\n", .{
            sample_params.temperature, sample_params.top_k, sample_params.top_p, seed,
        });
    }
    if (tq4v) try stdout.print("KV cache: K=fp32, V=TurboQuant TQ4 (asymmetric)\n", .{});

    // ── Probe wiring ─────────────────────────────────────────────────
    //
    // When --probe <path> is set, build a JSONL writer + Bus + the two
    // v0 probes (LogitProbe for D, ActivationEntropyProbe for B). The
    // null prior for D is computed by running a single forward on the
    // BOS token at pos 0; the chat loop then starts at pos 0 too,
    // overwriting that KV slot with the actual prompt token. No
    // permanent state pollution.
    const probe_info = probe.ModelInfo{
        .family = @tagName(cfg.family),
        .n_layers = @intCast(cfg.num_hidden_layers),
        .hidden_size = @intCast(cfg.hidden_size),
        .n_heads = @intCast(cfg.num_attention_heads),
        .n_kv_heads = @intCast(cfg.num_key_value_heads),
        .head_dim = @intCast(cfg.head_dim),
        .vocab_size = @intCast(cfg.vocab_size),
    };

    var probe_writer: ?probe.JsonlWriter = null;
    defer if (probe_writer) |*w| w.close();
    var probe_bus: ?probe.Bus = null;
    defer if (probe_bus) |*b| b.deinit(gpa);
    var probe_hidden_scratch: ?[]f32 = null;
    defer if (probe_hidden_scratch) |s| gpa.free(s);
    var probe_attn_scratch: ?[]f32 = null;
    defer if (probe_attn_scratch) |s| gpa.free(s);
    var probe_token_index: u32 = 0;

    if (probe_path) |pp| {
        probe_writer = try probe.JsonlWriter.open(pp);
        const w_ptr: *probe.JsonlWriter = &probe_writer.?;

        try stdout.print("probe: writing trace to {s}\n", .{pp});

        // Compute null prior: forward a single BOS token at pos 0,
        // read back the resulting logits, then hand them to LogitProbe
        // as the q distribution for KL(p || q).
        const bos: u32 = cfg.bos_token_id orelse 1;
        try stdout.print("probe: computing null prior (BOS={d})...\n", .{bos});
        try rec.reset();
        try rec.begin();
        try recordForwardStep(&rec, &sc, &gm, &kv, cfg, &k, 0, bos, tq4_hooks, true);
        try rec.endAndSubmit();
        try sc.logits.readBack(&ctx, f32, logits);
        // KV slot at pos 0 is now dirty with the BOS forward. The chat
        // loop below starts at pos = 0 and overwrites that slot with
        // the first prompt token — clean handoff, no reset needed.

        var bus = probe.Bus{};
        const probes_buf = try gpa.alloc(probe.Probe, 3);
        const lp = try probe.LogitProbe.create(gpa, w_ptr, @intCast(cfg.vocab_size), logits);
        const ap = try probe.ActivationEntropyProbe.create(gpa, w_ptr);
        const kp = try probe.AttentionProbe.create(gpa, w_ptr);
        probes_buf[0] = lp.probe();
        probes_buf[1] = ap.probe();
        probes_buf[2] = kp.probe();
        bus.probes = probes_buf;
        bus.finalize();
        probe_bus = bus;

        try probe.writeHeader(w_ptr, probe_info, dir_path, &.{ "logits", "act", "attn" });

        if (bus.needs_hidden_pre or bus.needs_hidden_post) {
            probe_hidden_scratch = try gpa.alloc(f32, cfg.hidden_size);
        }
        if (bus.needs_attention) {
            probe_attn_scratch = try gpa.alloc(f32, cfg.num_attention_heads * max_pos);
        }
    }

    // ── Batch mode: --prompts <tsv> + --probe-prefix <prefix> ────────
    //
    // Loads the model once, then runs each prompt as an independent
    // fresh-context chat turn. Per-prompt: open a fresh JsonlWriter at
    // `<prefix><label>.jsonl`, rebuild the bus + probes pointing to
    // that writer, reset pos=0, run chatTurn, tear down. Null prior +
    // probe scratches are shared across iterations. KV slots persist
    // between iterations but are overwritten by each new prompt's
    // prefill — attention only reads positions [0, n_pos) so any
    // residue beyond the new n_pos is unread.
    if (batch_prompts) |entries| {
        const bos: u32 = cfg.bos_token_id orelse 1;
        try stdout.print("probe (batch): computing null prior (BOS={d})...\n", .{bos});
        const null_prior_buf = try gpa.alloc(f32, cfg.vocab_size);
        defer gpa.free(null_prior_buf);
        try rec.reset();
        try rec.begin();
        try recordForwardStep(&rec, &sc, &gm, &kv, cfg, &k, 0, bos, tq4_hooks, true);
        try rec.endAndSubmit();
        try sc.logits.readBack(&ctx, f32, null_prior_buf);

        // All three v0 probes are installed per iteration, so the bus
        // wants hidden_post + attention every time. Allocate once.
        const hidden_scratch = try gpa.alloc(f32, cfg.hidden_size);
        defer gpa.free(hidden_scratch);
        const attn_scratch = try gpa.alloc(f32, cfg.num_attention_heads * max_pos);
        defer gpa.free(attn_scratch);

        for (entries) |entry| {
            const out_path = try std.fmt.allocPrint(gpa, "{s}{s}.jsonl", .{ probe_prefix.?, entry.label });
            defer gpa.free(out_path);
            try stdout.print("\n[{s}] → {s}\n", .{ entry.label, out_path });

            var writer = try probe.JsonlWriter.open(out_path);
            defer writer.close();

            var bus = probe.Bus{};
            const probes_buf = try gpa.alloc(probe.Probe, 3);
            const lp = try probe.LogitProbe.create(gpa, &writer, @intCast(cfg.vocab_size), null_prior_buf);
            const ap = try probe.ActivationEntropyProbe.create(gpa, &writer);
            const kp = try probe.AttentionProbe.create(gpa, &writer);
            probes_buf[0] = lp.probe();
            probes_buf[1] = ap.probe();
            probes_buf[2] = kp.probe();
            bus.probes = probes_buf;
            bus.finalize();
            defer bus.deinit(gpa);

            try probe.writeHeader(&writer, probe_info, dir_path, &.{ "logits", "act", "attn" });

            var pos: usize = 0;
            var token_index: u32 = 0;

            try chatTurn(
                gpa, &ctx, &rec, &sc, &gm, &kv, cfg, &k, &tok, entry.text, &pos, logits,
                sample_scratch, sample_params, rng, tmpl, false, tq4_hooks,
                &bus, probe_info, &token_index, hidden_scratch, attn_scratch,
                max_new,
            );
        }
        return;
    }

    // Position counter persists across turns (multi-turn chat builds on
    // the same KV cache).
    var pos: usize = 0;

    if (single_msg) |m| {
        try chatTurn(
            gpa, &ctx, &rec, &sc, &gm, &kv, cfg, &k, &tok, m, &pos, logits,
            sample_scratch, sample_params, rng, tmpl, false, tq4_hooks,
            if (probe_bus) |*b| b else null,
            probe_info,
            &probe_token_index,
            probe_hidden_scratch,
            probe_attn_scratch,
            max_new,
        );
        return;
    }

    // ── REPL ────────────────────────────────────────────────────────
    try stdout.print("\n{s} (Ctrl-D to exit)\n", .{tmpl.banner()});
    const stdin = std.io.getStdIn().reader();
    var line_buf: [4096]u8 = undefined;
    while (true) {
        try stdout.print("\nuser> ", .{});
        const maybe_line = try stdin.readUntilDelimiterOrEof(&line_buf, '\n');
        const line = maybe_line orelse {
            try stdout.print("\n", .{});
            break;
        };
        if (line.len == 0) continue;
        try chatTurn(
            gpa, &ctx, &rec, &sc, &gm, &kv, cfg, &k, &tok, line, &pos, logits,
            sample_scratch, sample_params, rng, tmpl, true, tq4_hooks,
            if (probe_bus) |*b| b else null,
            probe_info,
            &probe_token_index,
            probe_hidden_scratch,
            probe_attn_scratch,
            max_new,
        );
        if (pos >= max_pos - 64) {
            try stdout.print("\n[KV cache near capacity, ending session]\n", .{});
            break;
        }
    }
}

/// One round-trip through the model: prefill the prompt for `user_msg`
/// then sample until the family-specific end-of-turn marker (or
/// max_response). `pos` is updated in place so the next turn picks up
/// where this one stopped. The `chat_template_mod.ChatTemplate` arg encapsulates which
/// special tokens to emit and where.
fn chatTurn(
    gpa: std.mem.Allocator,
    ctx: *const vk.Context,
    rec: *gpu_recorder.Recorder,
    sc: *const gpu_scratch.GpuScratch,
    gm: *const gpu_model.GpuModel,
    kv: *const gpu_scratch.GpuKvCache,
    cfg: config_mod.Config,
    k: *const ChatKernels,
    tok: *const tokenizer_mod.Tokenizer,
    user_msg: []const u8,
    pos: *usize,
    logits: []f32,
    sample_scratch: []f32,
    sample_params: cpu_forward.SampleParams,
    rng: std.Random,
    tmpl: chat_template_mod.ChatTemplate,
    is_repl: bool,
    tq4_v: ?Tq4VHooks,
    probe_bus: ?*probe.Bus,
    probe_info: probe.ModelInfo,
    probe_token_index: *u32,
    probe_hidden_scratch: ?[]f32,
    probe_attn_scratch: ?[]f32,
    max_new: usize,
) !void {
    const stdout = std.io.getStdOut().writer();

    var prompt = std.ArrayList(u32).init(gpa);
    defer prompt.deinit();
    try tmpl.composePrompt(gpa, tok, user_msg, pos.* == 0, &prompt);

    if (!is_repl) {
        try stdout.print("\nprompt ({d} tokens, starting at pos {d}):\n  ", .{ prompt.items.len, pos.* });
        for (prompt.items) |id| try printTokenForDisplay(gpa, stdout, tok, id);
        try stdout.print("\n\nresponse: ", .{});
    } else {
        try stdout.print("model> ", .{});
    }

    const eos: ?u32 = cfg.eos_token_id;
    const eot: u32 = tmpl.end_of_turn;
    const max_response: usize = max_new;

    // Run all prompt tokens through the model. We only need the logits
    // at the LAST prefill position (which gives us the first response
    // token); intermediate logits are computed and ignored, which is a
    // small waste — fold the LM head out of prefill for a quick win
    // later if it matters.
    var current: u32 = prompt.items[0];
    var prompt_idx: usize = 0;
    var generated: usize = 0;

    // Decode-only throughput timer: clock starts after prefill so the
    // tok/s we report reflects steady-state generation, not warm-up.
    // Mirrors the Qwen path's measurement so bench script numbers are
    // apples-to-apples across model families.
    var t_decode_start: i128 = 0;
    var decode_started = false;

    while (true) {
        // We only need logits at the LAST prefill position (to sample
        // the first response token) and for every step of the sample
        // loop. Skipping the LM head matmul on intermediate prefill
        // tokens is a free ~10% startup win on short prompts and
        // scales linearly with prompt length on long ones.
        const is_last_prefill_or_decoding = (prompt_idx + 1 >= prompt.items.len);
        const is_prefill = (prompt_idx + 1 < prompt.items.len);

        if (probe_bus) |bus| try bus.onTokenStart(.{
            .info = probe_info,
            .token_index = probe_token_index.*,
            .pos = pos.*,
            .token_id = current,
            .is_prefill = is_prefill,
        });

        // Probed path forks here: when a probe wants per-layer hidden
        // state, the forward gets split into N submits with readbacks
        // between. Otherwise the fast single-submit path runs.
        const use_probed_forward = if (probe_bus) |bus|
            bus.needs_hidden_pre or bus.needs_hidden_post or bus.needs_attention
        else
            false;

        if (use_probed_forward) {
            try forwardStepProbed(
                rec,
                ctx,
                sc,
                gm,
                kv,
                cfg,
                k,
                pos.*,
                current,
                tq4_v,
                is_last_prefill_or_decoding,
                probe_bus.?,
                probe_info,
                probe_token_index.*,
                is_prefill,
                probe_hidden_scratch,
                probe_attn_scratch,
            );
        } else {
            if (pos.* > 0) try rec.reset();
            try rec.begin();
            try recordForwardStep(rec, sc, gm, kv, cfg, k, pos.*, current, tq4_v, is_last_prefill_or_decoding);
            try rec.endAndSubmit();
        }

        // If logits were computed this step, give probes that want
        // them a look before sampling.
        if (probe_bus) |bus| {
            if (is_last_prefill_or_decoding and bus.needs_logits) {
                try sc.logits.readBack(ctx, f32, logits);
                try bus.onLogits(.{
                    .info = probe_info,
                    .token_index = probe_token_index.*,
                    .pos = pos.*,
                    .token_id = current,
                    .is_prefill = is_prefill,
                }, logits);
            }
        }

        // Decide what to do at this position.
        prompt_idx += 1;
        pos.* += 1;
        probe_token_index.* += 1;

        if (prompt_idx < prompt.items.len) {
            // Still consuming prompt — advance to next prompt token.
            current = prompt.items[prompt_idx];
            continue;
        }

        if (!decode_started) {
            t_decode_start = std.time.nanoTimestamp();
            decode_started = true;
        }

        // Past the last prompt token: sample. The logits readback may
        // already have happened above for the probe, in which case the
        // device buffer is unchanged and a second readBack is just a
        // re-fetch (small cost, kept simple). If the probe path didn't
        // ask, this is the canonical readback.
        try sc.logits.readBack(ctx, f32, logits);
        const next = try cpu_forward.sample(logits, sample_params, rng, sample_scratch);

        if (probe_bus) |bus| try bus.onTokenEnd(.{
            .info = probe_info,
            .token_index = probe_token_index.*,
            .pos = pos.*,
            .token_id = current,
            .is_prefill = false,
        }, @intCast(next));

        // Stop conditions.
        if (next == eot) break;
        if (eos != null and next == eos.?) break;

        try printTokenForDisplay(gpa, stdout, tok, @intCast(next));

        generated += 1;
        if (generated >= max_response) break;
        current = @intCast(next);
    }
    try stdout.print("\n", .{});

    if (decode_started and generated > 0) {
        const t_end = std.time.nanoTimestamp();
        const ms_total = @as(f64, @floatFromInt(t_end - t_decode_start)) / 1_000_000.0;
        const tokps = @as(f64, @floatFromInt(generated)) * 1000.0 / ms_total;
        try stdout.print("[{d} tok in {d:.0} ms, {d:.1} tok/s]\n", .{ generated, ms_total, tokps });
    }
}

/// Print a single token id to `w` after applying the decoder rule for
/// the active tokenizer mode:
///   - SentencePiece (Gemma / TinyLlama / Mistral): ▁ (U+2581, bytes
///     E2 96 81) → ' '. Byte-fallback tokens with surface form
///     `<0xHH>` decode to the literal byte 0xHH (newlines, tabs, and
///     non-ASCII in models that use byte fallback for OOV codepoints).
///   - ByteLevel (Qwen3 / Llama3): walk the codepoints and reverse the
///     byte→unicode map back to original bytes.
///
/// Falls back to a "<id>" stub for token ids with no surface form (e.g.
/// holes in the vocab table).
fn printTokenForDisplay(
    gpa: std.mem.Allocator,
    w: anytype,
    tok: *const tokenizer_mod.Tokenizer,
    id: u32,
) !void {
    switch (tok.mode) {
        .sentencepiece => {
            const s = tok.decode(id) orelse {
                try w.print("<{d}>", .{id});
                return;
            };
            // Byte-fallback tokens have surface form `<0xHH>` (length
            // 6, two uppercase hex digits) and decode to byte 0xHH.
            // Detect by exact shape — any single token with that
            // surface form is the byte-fallback encoding because
            // SentencePiece reserves it for that purpose.
            if (s.len == 6 and s[0] == '<' and s[1] == '0' and s[2] == 'x' and s[5] == '>') {
                if (parseHexByte(s[3], s[4])) |b| {
                    try w.writeAll(&[_]u8{b});
                    return;
                }
            }
            var i: usize = 0;
            while (i < s.len) {
                if (i + 3 <= s.len and s[i] == 0xE2 and s[i + 1] == 0x96 and s[i + 2] == 0x81) {
                    try w.print(" ", .{});
                    i += 3;
                } else {
                    try w.print("{c}", .{s[i]});
                    i += 1;
                }
            }
        },
        .bytelevel => {
            const bytes = tok.decodeByteLevel(gpa, id) catch {
                try w.print("<{d}>", .{id});
                return;
            };
            defer gpa.free(bytes);
            try w.writeAll(bytes);
        },
    }
}

/// Parse two ASCII hex chars into the byte they represent. Returns
/// null on any non-hex input. Used by the SentencePiece byte-fallback
/// path in `printTokenForDisplay`.
fn parseHexByte(hi: u8, lo: u8) ?u8 {
    const h = hexDigit(hi) orelse return null;
    const l = hexDigit(lo) orelse return null;
    return (h << 4) | l;
}

fn hexDigit(c: u8) ?u8 {
    return switch (c) {
        '0'...'9' => c - '0',
        'a'...'f' => c - 'a' + 10,
        'A'...'F' => c - 'A' + 10,
        else => null,
    };
}

// ── encode: BPE round-trip smoke ───────────────────────────────────

fn runEncode(gpa: std.mem.Allocator, dir_path: []const u8, text: []const u8) !void {
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

// ── Forward-step helper used by both gen-many and chat ─────────────

const ChatKernels = runtime.ChatKernels;
const Tq4VHooks = runtime.Tq4VHooks;
const ForwardPushes = runtime.ForwardPushes;

const computeForwardPushes = runtime.computeForwardPushes;
const recordOneLayer = runtime.recordOneLayer;
const recordForwardStep = runtime.recordForwardStep;

/// Probed forward: same model dispatches as `recordForwardStep` but
/// split across multiple submits so the host can read back per-layer
/// hidden state into the bus between layers. Slow path, only invoked
/// when the bus has a probe that wants `hidden_post_layer`. Generation
/// output (next-token logits) is bit-identical to the fast path
/// because the GPU does the same math in the same order — only the
/// command-buffer chunking changes.
fn forwardStepProbed(
    rec: *gpu_recorder.Recorder,
    vk_ctx: *const vk.Context,
    sc: *const gpu_scratch.GpuScratch,
    gm: *const gpu_model.GpuModel,
    kv: *const gpu_scratch.GpuKvCache,
    cfg: config_mod.Config,
    k: *const ChatKernels,
    pos: usize,
    token_id: u32,
    tq4_v: ?Tq4VHooks,
    compute_logits: bool,
    bus: *const probe.Bus,
    info: probe.ModelInfo,
    token_index: u32,
    is_prefill: bool,
    hidden_scratch: ?[]f32,
    attn_scratch: ?[]f32,
) !void {
    const hidden: u32 = @intCast(cfg.hidden_size);
    const vocab: u32 = @intCast(cfg.vocab_size);

    const pushes = computeForwardPushes(cfg, sc, pos);

    const embed_push = EmbedLookupPush{
        .token_id = token_id,
        .dim = hidden,
        .scale = if (cfg.family.embedScalesByDim()) @sqrt(@as(f32, @floatFromInt(hidden))) else 1.0,
    };

    // ── Embed ───────────────────────────────────────────────────────
    try rec.reset();
    try rec.begin();
    try recDispatch1D(rec, &k.embed, &.{ &gm.embed_tokens, &sc.stream }, &embed_push, hidden);
    try rec.endAndSubmit();

    // ── Per layer ──────────────────────────────────────────────────
    const max_pos_u32: u32 = @intCast(sc.max_pos);
    const n_heads_u32: u32 = @intCast(cfg.num_attention_heads);
    for (0..cfg.num_hidden_layers) |layer_idx| {
        const layer_ctx = probe.Context{
            .info = info,
            .token_index = token_index,
            .pos = pos,
            .token_id = token_id,
            .layer_idx = @intCast(layer_idx),
            .is_prefill = is_prefill,
        };

        if (bus.needs_hidden_pre) {
            try sc.stream.readBack(vk_ctx, f32, hidden_scratch.?);
            try bus.onLayerEntry(layer_ctx, hidden_scratch.?);
        }

        try rec.reset();
        try rec.begin();
        try recordOneLayer(rec, sc, gm, kv, cfg, k, layer_idx, pos, &pushes, tq4_v);
        try rec.endAndSubmit();

        // sc.scores holds this layer's post-softmax attention weights
        // until the next layer's softmax overwrites it. Read it now
        // while it's still valid. Layout is [n_heads, max_pos]
        // row-major; only the first n_pos columns of each row are
        // valid post-softmax — the probe slices accordingly.
        if (bus.needs_attention) {
            try sc.scores.readBack(vk_ctx, f32, attn_scratch.?);
            try bus.onAttention(layer_ctx, attn_scratch.?, n_heads_u32, pushes.n_pos, max_pos_u32);
        }

        if (bus.needs_hidden_post) {
            try sc.stream.readBack(vk_ctx, f32, hidden_scratch.?);
            try bus.onLayerExit(layer_ctx, hidden_scratch.?);
        }
    }

    // ── Final norm + LM head ───────────────────────────────────────
    if (compute_logits) {
        try rec.reset();
        try rec.begin();
        try recDispatchPerRow(rec, &k.rmsnorm, &.{ &sc.stream, &gm.final_norm, &sc.final_norm_out }, &pushes.rms_push, 1);
        try recDispatchMatmul(rec, &k.matmul_lm_head, &.{ &sc.final_norm_out, &gm.lm_head, &sc.logits }, 1, vocab, hidden);
        try rec.endAndSubmit();
    }
}

// ── gpu-gen-many: multi-token generation with KV cache ─────────────

const AttnScoresPush = runtime.AttnScoresPush;
const AttnOutputPush = runtime.AttnOutputPush;
const KvWritePush = runtime.KvWritePush;

fn runGpuGenMany(gpa: std.mem.Allocator, dir_path: []const u8, first_token: u32, n_tokens: usize) !void {
    var cpu = try model_mod.Model.load(gpa, dir_path);
    defer cpu.deinit();
    const cfg = cpu.config;

    const tok_path = try std.fmt.allocPrint(gpa, "{s}/tokenizer.json", .{dir_path});
    defer gpa.free(tok_path);
    var tok = try tokenizer_mod.Tokenizer.loadFromFile(gpa, tok_path);
    defer tok.deinit();

    var ctx = try vk.Context.init(gpa);
    defer ctx.deinit();

    const stdout = std.io.getStdOut().writer();
    try stdout.print("device: {s}\n", .{ctx.deviceName()});
    try stdout.print("uploading weights (bf16 matmul path)...\n", .{});
    var gm = try gpu_model.GpuModel.upload(gpa, &ctx, &cpu, .bf16_matmul);
    defer gm.deinit(ctx.device);

    const max_pos: usize = @max(n_tokens + 8, 64);
    var sc = try gpu_scratch.GpuScratch.init(&ctx, cfg, max_pos);
    defer sc.deinit(ctx.device);
    var kv = try gpu_scratch.GpuKvCache.init(gpa, &ctx, cfg, max_pos);
    defer kv.deinit(ctx.device);

    var k = try runtime.ChatKernels.init(&ctx, gm.precision, cfg.family);
    defer k.deinit();

    var rec = try gpu_recorder.Recorder.init(&ctx, 512, 2048);
    defer rec.deinit();

    const logits = try gpa.alloc(f32, cfg.vocab_size);
    defer gpa.free(logits);

    const first_str = tok.decode(first_token) orelse "<unknown>";
    try stdout.print("\nstreaming generation, max {d} tokens, max_pos = {d}\n", .{ n_tokens, max_pos });
    try stdout.print("input: id={d} {s}\n", .{ first_token, first_str });
    try stdout.print("output: ", .{});

    var current_token: u32 = first_token;
    var pos: usize = 0;
    var total_gpu_ns: i128 = 0;

    while (pos < n_tokens) : (pos += 1) {
        if (pos > 0) try rec.reset();
        try rec.begin();
        try recordForwardStep(&rec, &sc, &gm, &kv, cfg, &k, pos, current_token, null, true);

        const t0 = std.time.nanoTimestamp();
        try rec.endAndSubmit();
        const t1 = std.time.nanoTimestamp();
        total_gpu_ns += (t1 - t0);

        try sc.logits.readBack(&ctx, f32, logits);
        const next = cpu_forward.argmax(logits);
        const next_str = tok.decode(next) orelse "<unknown>";
        try stdout.print("{s}", .{next_str});

        if (cfg.eos_token_id) |eos| {
            if (next == eos) break;
        }
        current_token = @intCast(next);
    }

    try stdout.print("\n\n", .{});
    const total_gpu_ms = @as(f64, @floatFromInt(total_gpu_ns)) / 1_000_000.0;
    const tokens_done = pos;
    const ms_per_tok = total_gpu_ms / @as(f64, @floatFromInt(@max(tokens_done, 1)));
    try stdout.print("generated {d} tokens in {d:.0} ms total ({d:.1} ms/token, {d:.1} tok/s)\n", .{
        tokens_done, total_gpu_ms, ms_per_tok, 1000.0 / ms_per_tok,
    });
}

// ── gpu-gen-tq4v: GPU forward with TQ4 V-cache vs fp32 V baseline ──
//
// Same input token, two passes through recordForwardStep — once with
// tq4_v = null (existing fp32 V path), once with the TQ4 hooks set
// up. Reads back logits for both, prints argmax + top-5 (with decoded
// text) + max / mean / rms divergence over the full vocab. The
// signal we want: argmax matches and top-5 IDs are preserved.

fn runGpuGenTq4V(gpa: std.mem.Allocator, dir_path: []const u8, token_id: u32) !void {
    var cpu = try model_mod.Model.load(gpa, dir_path);
    defer cpu.deinit();
    const cfg = cpu.config;

    const tok_path = try std.fmt.allocPrint(gpa, "{s}/tokenizer.json", .{dir_path});
    defer gpa.free(tok_path);
    var tok = try tokenizer_mod.Tokenizer.loadFromFile(gpa, tok_path);
    defer tok.deinit();

    var ctx = try vk.Context.init(gpa);
    defer ctx.deinit();

    const stdout = std.io.getStdOut().writer();
    try stdout.print("device: {s}\n", .{ctx.deviceName()});
    try stdout.print("input: id={d}  string={s}\n\n", .{ token_id, tok.decode(token_id) orelse "?" });

    var gm = try gpu_model.GpuModel.upload(gpa, &ctx, &cpu, .bf16_matmul);
    defer gm.deinit(ctx.device);

    const max_pos: usize = 8;
    var sc = try gpu_scratch.GpuScratch.init(&ctx, cfg, max_pos);
    defer sc.deinit(ctx.device);
    var kv = try gpu_scratch.GpuKvCache.init(gpa, &ctx, cfg, max_pos);
    defer kv.deinit(ctx.device);
    var kv_tq4 = try gpu_scratch.GpuKvCacheTq4.init(gpa, &ctx, cfg, max_pos);
    defer kv_tq4.deinit(ctx.device);

    var k = try runtime.ChatKernels.init(&ctx, gm.precision, cfg.family);
    defer k.deinit();

    var tq_pack = try pipeline.Kernel.init(&ctx, &shaders.tq4_pack_to_cache, 2, @sizeOf(Tq4PackPush));
    defer tq_pack.deinit();
    var tq_unpack = try pipeline.Kernel.init(&ctx, &shaders.tq4_unpack256, 2, 0);
    defer tq_unpack.deinit();
    const tq4_hooks = Tq4VHooks{ .pack = &tq_pack, .unpack = &tq_unpack, .cache = &kv_tq4 };

    var rec = try gpu_recorder.Recorder.init(&ctx, 512, 2048);
    defer rec.deinit();

    const logits_a = try gpa.alloc(f32, cfg.vocab_size);
    defer gpa.free(logits_a);
    const logits_b = try gpa.alloc(f32, cfg.vocab_size);
    defer gpa.free(logits_b);

    // Pass 1: fp32 V baseline.
    try rec.begin();
    try recordForwardStep(&rec, &sc, &gm, &kv, cfg, &k, 0, token_id, null, true);
    try rec.endAndSubmit();
    try sc.logits.readBack(&ctx, f32, logits_a);

    // Pass 2: TQ4 V.
    try rec.reset();
    try rec.begin();
    try recordForwardStep(&rec, &sc, &gm, &kv, cfg, &k, 0, token_id, tq4_hooks, true);
    try rec.endAndSubmit();
    try sc.logits.readBack(&ctx, f32, logits_b);

    const top_a = try topK(gpa, logits_a, 5);
    defer gpa.free(top_a);
    const top_b = try topK(gpa, logits_b, 5);
    defer gpa.free(top_b);

    try stdout.print("            fp32 V baseline                     TQ4 V\n", .{});
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
    try stdout.print("\nlogit divergence over {d} tokens:\n", .{logits_a.len});
    try stdout.print("  max |Δ|   = {d:.6}  at id={d}\n", .{ max_abs_delta, max_abs_delta_idx });
    try stdout.print("  mean |Δ|  = {d:.6}\n", .{sum_abs / n});
    try stdout.print("  rms  Δ    = {d:.6}\n", .{std.math.sqrt(sum_sq / n)});

    const argmax_a = cpu_forward.argmax(logits_a);
    const argmax_b = cpu_forward.argmax(logits_b);
    try stdout.print("\nargmax:  fp32 V → {d}    TQ4 V → {d}    {s}\n", .{
        argmax_a, argmax_b, if (argmax_a == argmax_b) "(MATCH)" else "(DIVERGE!)",
    });
    var top5_match: usize = 0;
    for (top_a) |a| for (top_b) |b| if (a.id == b.id) { top5_match += 1; break; };
    try stdout.print("top-5 ID overlap: {d}/5\n", .{top5_match});
}

// ── gpu-gen: full forward on GPU + parity vs CPU --gen ─────────────

fn runGpuGen(gpa: std.mem.Allocator, dir_path: []const u8, token_id: u32) !void {
    var cpu = try model_mod.Model.load(gpa, dir_path);
    defer cpu.deinit();
    const cfg = cpu.config;

    const tok_path = try std.fmt.allocPrint(gpa, "{s}/tokenizer.json", .{dir_path});
    defer gpa.free(tok_path);
    var tok = try tokenizer_mod.Tokenizer.loadFromFile(gpa, tok_path);
    defer tok.deinit();

    var ctx = try vk.Context.init(gpa);
    defer ctx.deinit();

    const stdout = std.io.getStdOut().writer();
    const input_str = tok.decode(token_id) orelse "<unknown>";
    try stdout.print("input  token id={d}  string={s}\n", .{ token_id, input_str });
    try stdout.print("device: {s}\n\n", .{ctx.deviceName()});

    const t_up0 = std.time.nanoTimestamp();
    var gm = try gpu_model.GpuModel.upload(gpa, &ctx, &cpu, .fp32_all);
    defer gm.deinit(ctx.device);
    const t_up1 = std.time.nanoTimestamp();
    const upload_ms = @as(f64, @floatFromInt(t_up1 - t_up0)) / 1_000_000.0;
    try stdout.print("upload: {d:.0} ms\n", .{upload_ms});

    var sc = try gpu_scratch.GpuScratch.init(&ctx, cfg, 1);
    defer sc.deinit(ctx.device);

    // ── Build kernels once, rebind per-layer ────────────────────────
    var k_embed = try pipeline.Kernel.init(&ctx, &shaders.embed_lookup, 2, @sizeOf(EmbedLookupPush));
    defer k_embed.deinit();
    var k_rmsnorm = try pipeline.Kernel.init(&ctx, &shaders.rmsnorm, 3, @sizeOf(RmsnormPush));
    defer k_rmsnorm.deinit();
    var k_matmul = try pipeline.Kernel.init(&ctx, &shaders.matmul_nt_v2, 3, @sizeOf(MatmulPush));
    defer k_matmul.deinit();
    var k_rope = try pipeline.Kernel.init(&ctx, &shaders.rope, 2, @sizeOf(RopePush));
    defer k_rope.deinit();
    var k_attn = try pipeline.Kernel.init(&ctx, &shaders.attn_decode_single, 2, @sizeOf(AttnDecodeSinglePush));
    defer k_attn.deinit();
    var k_add = try pipeline.Kernel.init(&ctx, &shaders.add_in_place, 2, @sizeOf(AddInPlacePush));
    defer k_add.deinit();
    var k_geglu = try pipeline.Kernel.init(&ctx, &shaders.geglu, 3, @sizeOf(GegluPush));
    defer k_geglu.deinit();

    const hidden: u32 = @intCast(cfg.hidden_size);
    const inter: u32 = @intCast(cfg.intermediate_size);
    const q_dim: u32 = @intCast(cfg.num_attention_heads * cfg.head_dim);
    const kv_dim: u32 = @intCast(cfg.num_key_value_heads * cfg.head_dim);
    const vocab: u32 = @intCast(cfg.vocab_size);
    const gemma_quirk: u32 = if (cfg.family == .gemma) 1 else 0;

    // ── Recorder: one command buffer for the whole forward pass ────
    // Sizing: 291 dispatches at ~3 storage-buffer descriptors each =
    // ~728 descriptors. Round up generously so we don't trip on any
    // future kernel that adds bindings.
    var rec = try gpu_recorder.Recorder.init(&ctx, 512, 2048);
    defer rec.deinit();

    const rms_push = RmsnormPush{ .dim = hidden, .eps = cfg.rms_norm_eps, .gemma_quirk = gemma_quirk };
    const add_push = AddInPlacePush{ .n = hidden };
    const attn_push = AttnDecodeSinglePush{
        .n_heads = @intCast(cfg.num_attention_heads),
        .heads_per_kv = @intCast(cfg.num_attention_heads / cfg.num_key_value_heads),
        .head_dim = @intCast(cfg.head_dim),
    };
    const rope_q_push = RopePush{
        .n_heads = @intCast(cfg.num_attention_heads),
        .head_dim = @intCast(cfg.head_dim),
        .pos = 0,
        .theta_base = cfg.rope_theta,
    };
    const rope_k_push = RopePush{
        .n_heads = @intCast(cfg.num_key_value_heads),
        .head_dim = @intCast(cfg.head_dim),
        .pos = 0,
        .theta_base = cfg.rope_theta,
    };
    const geglu_push = GegluPush{ .n = inter };
    const embed_push = EmbedLookupPush{
        .token_id = token_id,
        .dim = hidden,
        .scale = if (cfg.family.embedScalesByDim()) @sqrt(@as(f32, @floatFromInt(hidden))) else 1.0,
    };

    const t_gpu0 = std.time.nanoTimestamp();

    try rec.begin();

    // Embed lookup → residual stream.
    try recDispatch1D(&rec, &k_embed, &.{ &gm.embed_tokens, &sc.stream }, &embed_push, hidden);

    // 18 transformer blocks.
    for (gm.layers) |*layer| {
        try recDispatchPerRow(&rec, &k_rmsnorm, &.{ &sc.stream, &layer.input_layernorm, &sc.x_norm }, &rms_push, 1);

        try recDispatchMatmul(&rec, &k_matmul, &.{ &sc.x_norm, &layer.q_proj.?, &sc.q }, 1, q_dim, hidden);
        try recDispatchMatmul(&rec, &k_matmul, &.{ &sc.x_norm, &layer.k_proj.?, &sc.k }, 1, kv_dim, hidden);
        try recDispatchMatmul(&rec, &k_matmul, &.{ &sc.x_norm, &layer.v_proj.?, &sc.v }, 1, kv_dim, hidden);

        try recDispatchRope(&rec, &k_rope, &.{ &sc.q, &sc.q_rot }, &rope_q_push, cfg.num_attention_heads, cfg.head_dim);
        try recDispatchRope(&rec, &k_rope, &.{ &sc.k, &sc.k_rot }, &rope_k_push, cfg.num_key_value_heads, cfg.head_dim);

        try recDispatch1D(&rec, &k_attn, &.{ &sc.v, &sc.head_out }, &attn_push, q_dim);

        try recDispatchMatmul(&rec, &k_matmul, &.{ &sc.head_out, &layer.o_proj.?, &sc.attn_out }, 1, hidden, q_dim);

        try recDispatch1D(&rec, &k_add, &.{ &sc.stream, &sc.attn_out }, &add_push, hidden);

        try recDispatchPerRow(&rec, &k_rmsnorm, &.{ &sc.stream, &layer.post_attention_layernorm, &sc.mid_norm }, &rms_push, 1);

        try recDispatchMatmul(&rec, &k_matmul, &.{ &sc.mid_norm, &layer.gate_proj, &sc.gate }, 1, inter, hidden);
        try recDispatchMatmul(&rec, &k_matmul, &.{ &sc.mid_norm, &layer.up_proj, &sc.up }, 1, inter, hidden);

        try recDispatch1D(&rec, &k_geglu, &.{ &sc.gate, &sc.up, &sc.fused }, &geglu_push, inter);

        try recDispatchMatmul(&rec, &k_matmul, &.{ &sc.fused, &layer.down_proj, &sc.ffn_out }, 1, hidden, inter);

        try recDispatch1D(&rec, &k_add, &.{ &sc.stream, &sc.ffn_out }, &add_push, hidden);
    }

    // Final rmsnorm + LM head.
    try recDispatchPerRow(&rec, &k_rmsnorm, &.{ &sc.stream, &gm.final_norm, &sc.final_norm_out }, &rms_push, 1);
    try recDispatchMatmul(&rec, &k_matmul, &.{ &sc.final_norm_out, &gm.lm_head, &sc.logits }, 1, vocab, hidden);

    try rec.endAndSubmit();

    const t_gpu1 = std.time.nanoTimestamp();
    const gpu_ms = @as(f64, @floatFromInt(t_gpu1 - t_gpu0)) / 1_000_000.0;

    // ── Read back logits, argmax, decode ───────────────────────────
    const logits = try gpa.alloc(f32, cfg.vocab_size);
    defer gpa.free(logits);
    try sc.logits.readBack(&ctx, f32, logits);

    const sampled = cpu_forward.argmax(logits);
    const out_str = tok.decode(sampled) orelse "<unknown>";

    try stdout.print("forward (GPU, ~291 dispatches in 1 command buffer): {d:.0} ms\n", .{gpu_ms});

    // A second forward to measure the warm-cache time. Recorder needs
    // to be reset before re-recording.
    try rec.reset();
    const t_warm0 = std.time.nanoTimestamp();
    try rec.begin();
    try recDispatch1D(&rec, &k_embed, &.{ &gm.embed_tokens, &sc.stream }, &embed_push, hidden);
    for (gm.layers) |*layer| {
        try recDispatchPerRow(&rec, &k_rmsnorm, &.{ &sc.stream, &layer.input_layernorm, &sc.x_norm }, &rms_push, 1);
        try recDispatchMatmul(&rec, &k_matmul, &.{ &sc.x_norm, &layer.q_proj.?, &sc.q }, 1, q_dim, hidden);
        try recDispatchMatmul(&rec, &k_matmul, &.{ &sc.x_norm, &layer.k_proj.?, &sc.k }, 1, kv_dim, hidden);
        try recDispatchMatmul(&rec, &k_matmul, &.{ &sc.x_norm, &layer.v_proj.?, &sc.v }, 1, kv_dim, hidden);
        try recDispatchRope(&rec, &k_rope, &.{ &sc.q, &sc.q_rot }, &rope_q_push, cfg.num_attention_heads, cfg.head_dim);
        try recDispatchRope(&rec, &k_rope, &.{ &sc.k, &sc.k_rot }, &rope_k_push, cfg.num_key_value_heads, cfg.head_dim);
        try recDispatch1D(&rec, &k_attn, &.{ &sc.v, &sc.head_out }, &attn_push, q_dim);
        try recDispatchMatmul(&rec, &k_matmul, &.{ &sc.head_out, &layer.o_proj.?, &sc.attn_out }, 1, hidden, q_dim);
        try recDispatch1D(&rec, &k_add, &.{ &sc.stream, &sc.attn_out }, &add_push, hidden);
        try recDispatchPerRow(&rec, &k_rmsnorm, &.{ &sc.stream, &layer.post_attention_layernorm, &sc.mid_norm }, &rms_push, 1);
        try recDispatchMatmul(&rec, &k_matmul, &.{ &sc.mid_norm, &layer.gate_proj, &sc.gate }, 1, inter, hidden);
        try recDispatchMatmul(&rec, &k_matmul, &.{ &sc.mid_norm, &layer.up_proj, &sc.up }, 1, inter, hidden);
        try recDispatch1D(&rec, &k_geglu, &.{ &sc.gate, &sc.up, &sc.fused }, &geglu_push, inter);
        try recDispatchMatmul(&rec, &k_matmul, &.{ &sc.fused, &layer.down_proj, &sc.ffn_out }, 1, hidden, inter);
        try recDispatch1D(&rec, &k_add, &.{ &sc.stream, &sc.ffn_out }, &add_push, hidden);
    }
    try recDispatchPerRow(&rec, &k_rmsnorm, &.{ &sc.stream, &gm.final_norm, &sc.final_norm_out }, &rms_push, 1);
    try recDispatchMatmul(&rec, &k_matmul, &.{ &sc.final_norm_out, &gm.lm_head, &sc.logits }, 1, vocab, hidden);
    try rec.endAndSubmit();
    const t_warm1 = std.time.nanoTimestamp();
    const warm_ms = @as(f64, @floatFromInt(t_warm1 - t_warm0)) / 1_000_000.0;
    try stdout.print("forward (warm, 2nd pass)                          : {d:.0} ms\n", .{warm_ms});

    const k_top: usize = 5;
    const top = try topK(gpa, logits, k_top);
    defer gpa.free(top);
    try stdout.print("\ntop {d} logits (GPU):\n", .{k_top});
    for (top) |entry| {
        const s = tok.decode(entry.id) orelse "<unknown>";
        try stdout.print("  id={d:>6}  logit={d:>10.4}  {s}\n", .{ entry.id, entry.value, s });
    }
    try stdout.print("\nGPU sampled (greedy): id={d}  string={s}\n", .{ sampled, out_str });

    // ── Parity vs CPU --gen ─────────────────────────────────────────
    try stdout.print("\nrunning CPU forward for parity check (this takes ~18 s)...\n", .{});
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const cpu_logits = try gpa.alloc(f32, cfg.vocab_size);
    defer gpa.free(cpu_logits);
    const t_cpu0 = std.time.nanoTimestamp();
    try cpu_forward.forward(&cpu, token_id, 0, arena.allocator(), cpu_logits);
    const t_cpu1 = std.time.nanoTimestamp();
    const cpu_ms = @as(f64, @floatFromInt(t_cpu1 - t_cpu0)) / 1_000_000.0;

    const cpu_argmax = cpu_forward.argmax(cpu_logits);
    var max_abs: f32 = 0;
    var max_idx: usize = 0;
    for (logits, cpu_logits, 0..) |g, c, i| {
        const d = @abs(g - c);
        if (d > max_abs) {
            max_abs = d;
            max_idx = i;
        }
    }

    try stdout.print("CPU forward: {d:.0} ms\n", .{cpu_ms});
    try stdout.print("CPU argmax: id={d} ({s}) logit={d:.4}\n", .{
        cpu_argmax, tok.decode(cpu_argmax) orelse "?", cpu_logits[cpu_argmax],
    });
    try stdout.print("max |Δ| over the whole logit vector = {e:.3}  (at idx {d}: cpu={d:.4} gpu={d:.4})\n", .{
        max_abs, max_idx, cpu_logits[max_idx], logits[max_idx],
    });
    if (sampled != cpu_argmax) {
        std.debug.print("FAIL: GPU argmax {d} ≠ CPU argmax {d}\n", .{ sampled, cpu_argmax });
        return error.ParityFailed;
    }
    try stdout.print("\nPASS GPU --gen argmax matches CPU --gen ({d} → {s})\n", .{ sampled, out_str });
}

// ── gpu-gen-qwen35: hybrid forward end-to-end (linear + full attn) ─

fn runGpuGenQwen35(gpa: std.mem.Allocator, dir_path: []const u8, token_id: u32) !void {
    var cpu = try model_mod.Model.load(gpa, dir_path);
    defer cpu.deinit();
    const cfg = cpu.config;
    if (cfg.family != .qwen35) return error.NotHybridFamily;

    const tok_path = try std.fmt.allocPrint(gpa, "{s}/tokenizer.json", .{dir_path});
    defer gpa.free(tok_path);
    var tok = try tokenizer_mod.Tokenizer.loadFromFile(gpa, tok_path);
    defer tok.deinit();

    var ctx = try vk.Context.init(gpa);
    defer ctx.deinit();

    const stdout = std.io.getStdOut().writer();
    const input_str = tok.decode(token_id) orelse "<unknown>";
    try stdout.print("input  token id={d}  string={s}\n", .{ token_id, input_str });
    try stdout.print("device: {s}\n", .{ctx.deviceName()});
    try stdout.print("schedule: {d} layers ({d} linear + {d} full)\n", .{
        cfg.num_hidden_layers,
        blk: {
            var n: usize = 0;
            for (cfg.layer_types[0..cfg.num_hidden_layers]) |t| if (t == .linear_attention) {
                n += 1;
            };
            break :blk n;
        },
        blk: {
            var n: usize = 0;
            for (cfg.layer_types[0..cfg.num_hidden_layers]) |t| if (t == .full_attention) {
                n += 1;
            };
            break :blk n;
        },
    });

    // ── Upload weights ──────────────────────────────────────────────
    const t_up0 = std.time.nanoTimestamp();
    var gm = try gpu_model.GpuModel.upload(gpa, &ctx, &cpu, .fp32_all);
    defer gm.deinit(ctx.device);
    const t_up1 = std.time.nanoTimestamp();
    try stdout.print("upload: {d:.0} ms\n\n", .{@as(f64, @floatFromInt(t_up1 - t_up0)) / 1_000_000.0});

    // ── Sizes ───────────────────────────────────────────────────────
    const hidden: u32 = @intCast(cfg.hidden_size);
    const inter: u32 = @intCast(cfg.intermediate_size);
    const vocab: u32 = @intCast(cfg.vocab_size);
    const head_dim: u32 = @intCast(cfg.head_dim);
    const n_q_heads: u32 = @intCast(cfg.num_attention_heads);
    const n_kv_heads: u32 = @intCast(cfg.num_key_value_heads);
    const heads_per_kv: u32 = n_q_heads / n_kv_heads;
    const q_dim: u32 = n_q_heads * head_dim;
    const q_proj_rows: u32 = if (cfg.attn_output_gate) 2 * q_dim else q_dim;
    const kv_dim: u32 = n_kv_heads * head_dim;
    const rotary_dim: u32 = @intFromFloat(@as(f32, @floatFromInt(head_dim)) * cfg.partial_rotary_factor);

    const conv_dim: u32 = @intCast(cfg.linearAttnConvDim());
    const value_dim: u32 = @intCast(cfg.linear_num_value_heads * cfg.linear_value_head_dim);
    const key_dim: u32 = @intCast(cfg.linear_num_key_heads * cfg.linear_key_head_dim);
    const n_v_heads: u32 = @intCast(cfg.linear_num_value_heads);
    const n_k_heads_lin: u32 = @intCast(cfg.linear_num_key_heads);
    const head_k: u32 = @intCast(cfg.linear_key_head_dim);
    const head_v: u32 = @intCast(cfg.linear_value_head_dim);
    const conv_kernel: u32 = @intCast(cfg.linear_conv_kernel_dim);
    const q_scale: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(cfg.linear_key_head_dim)));

    // ── Scratch buffers (shared across layers, reused) ──────────────
    var sc_stream = try buffer.Buffer.initDeviceOnly(&ctx, hidden * @sizeOf(f32));
    defer sc_stream.deinit(ctx.device);
    var sc_x_norm = try buffer.Buffer.initDeviceOnly(&ctx, hidden * @sizeOf(f32));
    defer sc_x_norm.deinit(ctx.device);
    var sc_mid_norm = try buffer.Buffer.initDeviceOnly(&ctx, hidden * @sizeOf(f32));
    defer sc_mid_norm.deinit(ctx.device);
    var sc_attn_out = try buffer.Buffer.initDeviceOnly(&ctx, hidden * @sizeOf(f32));
    defer sc_attn_out.deinit(ctx.device);
    var sc_gate = try buffer.Buffer.initDeviceOnly(&ctx, inter * @sizeOf(f32));
    defer sc_gate.deinit(ctx.device);
    var sc_up = try buffer.Buffer.initDeviceOnly(&ctx, inter * @sizeOf(f32));
    defer sc_up.deinit(ctx.device);
    var sc_fused = try buffer.Buffer.initDeviceOnly(&ctx, inter * @sizeOf(f32));
    defer sc_fused.deinit(ctx.device);
    var sc_ffn_out = try buffer.Buffer.initDeviceOnly(&ctx, hidden * @sizeOf(f32));
    defer sc_ffn_out.deinit(ctx.device);
    var sc_final_norm_out = try buffer.Buffer.initDeviceOnly(&ctx, hidden * @sizeOf(f32));
    defer sc_final_norm_out.deinit(ctx.device);
    var sc_logits = try buffer.Buffer.initDeviceOnly(&ctx, vocab * @sizeOf(f32));
    defer sc_logits.deinit(ctx.device);

    // Linear-attn scratch
    var sc_mixed_qkv = try buffer.Buffer.initDeviceOnly(&ctx, conv_dim * @sizeOf(f32));
    defer sc_mixed_qkv.deinit(ctx.device);
    // Distinct post-conv buffer to avoid aliasing readonly/writeonly
    // bindings of the same memory range — Vulkan's access decorations
    // don't promise sane behavior when both qualifiers point to the
    // same buffer, and we hit it in practice.
    var sc_mixed_qkv_post = try buffer.Buffer.initDeviceOnly(&ctx, conv_dim * @sizeOf(f32));
    defer sc_mixed_qkv_post.deinit(ctx.device);
    var sc_z = try buffer.Buffer.initDeviceOnly(&ctx, value_dim * @sizeOf(f32));
    defer sc_z.deinit(ctx.device);
    var sc_braw = try buffer.Buffer.initDeviceOnly(&ctx, n_v_heads * @sizeOf(f32));
    defer sc_braw.deinit(ctx.device);
    var sc_araw = try buffer.Buffer.initDeviceOnly(&ctx, n_v_heads * @sizeOf(f32));
    defer sc_araw.deinit(ctx.device);
    var sc_qlin = try buffer.Buffer.initDeviceOnly(&ctx, key_dim * @sizeOf(f32));
    defer sc_qlin.deinit(ctx.device);
    var sc_klin = try buffer.Buffer.initDeviceOnly(&ctx, key_dim * @sizeOf(f32));
    defer sc_klin.deinit(ctx.device);
    var sc_vlin = try buffer.Buffer.initDeviceOnly(&ctx, value_dim * @sizeOf(f32));
    defer sc_vlin.deinit(ctx.device);
    var sc_qlin_n = try buffer.Buffer.initDeviceOnly(&ctx, key_dim * @sizeOf(f32));
    defer sc_qlin_n.deinit(ctx.device);
    var sc_klin_n = try buffer.Buffer.initDeviceOnly(&ctx, key_dim * @sizeOf(f32));
    defer sc_klin_n.deinit(ctx.device);
    var sc_y = try buffer.Buffer.initDeviceOnly(&ctx, value_dim * @sizeOf(f32));
    defer sc_y.deinit(ctx.device);
    var sc_post_norm = try buffer.Buffer.initDeviceOnly(&ctx, value_dim * @sizeOf(f32));
    defer sc_post_norm.deinit(ctx.device);

    // Full-attn scratch
    var sc_q_gate = try buffer.Buffer.initDeviceOnly(&ctx, q_proj_rows * @sizeOf(f32));
    defer sc_q_gate.deinit(ctx.device);
    var sc_q = try buffer.Buffer.initDeviceOnly(&ctx, q_dim * @sizeOf(f32));
    defer sc_q.deinit(ctx.device);
    var sc_gate_attn = try buffer.Buffer.initDeviceOnly(&ctx, q_dim * @sizeOf(f32));
    defer sc_gate_attn.deinit(ctx.device);
    var sc_k = try buffer.Buffer.initDeviceOnly(&ctx, kv_dim * @sizeOf(f32));
    defer sc_k.deinit(ctx.device);
    var sc_v = try buffer.Buffer.initDeviceOnly(&ctx, kv_dim * @sizeOf(f32));
    defer sc_v.deinit(ctx.device);
    var sc_qrot = try buffer.Buffer.initDeviceOnly(&ctx, q_dim * @sizeOf(f32));
    defer sc_qrot.deinit(ctx.device);
    var sc_krot = try buffer.Buffer.initDeviceOnly(&ctx, kv_dim * @sizeOf(f32));
    defer sc_krot.deinit(ctx.device);
    var sc_head_out = try buffer.Buffer.initDeviceOnly(&ctx, q_dim * @sizeOf(f32));
    defer sc_head_out.deinit(ctx.device);
    var sc_head_out_gated = try buffer.Buffer.initDeviceOnly(&ctx, q_dim * @sizeOf(f32));
    defer sc_head_out_gated.deinit(ctx.device);
    const max_pos: u32 = 1; // Single-token gen.
    var sc_scores = try buffer.Buffer.initDeviceOnly(&ctx, n_q_heads * max_pos * @sizeOf(f32));
    defer sc_scores.deinit(ctx.device);

    // ── Per-layer persistent state ──────────────────────────────────
    // Linear layers: SSM state buffers (zero-initialised by initDeviceOnly).
    // Full layers: KV cache (one slot since max_pos=1).
    var ssm_conv = try gpa.alloc(?buffer.Buffer, cfg.num_hidden_layers);
    defer gpa.free(ssm_conv);
    var ssm_rec = try gpa.alloc(?buffer.Buffer, cfg.num_hidden_layers);
    defer gpa.free(ssm_rec);
    var kv_k = try gpa.alloc(?buffer.Buffer, cfg.num_hidden_layers);
    defer gpa.free(kv_k);
    var kv_v = try gpa.alloc(?buffer.Buffer, cfg.num_hidden_layers);
    defer gpa.free(kv_v);
    @memset(ssm_conv, null);
    @memset(ssm_rec, null);
    @memset(kv_k, null);
    @memset(kv_v, null);
    defer for (ssm_conv) |*b| if (b.*) |*v| v.deinit(ctx.device);
    defer for (ssm_rec) |*b| if (b.*) |*v| v.deinit(ctx.device);
    defer for (kv_k) |*b| if (b.*) |*v| v.deinit(ctx.device);
    defer for (kv_v) |*b| if (b.*) |*v| v.deinit(ctx.device);

    for (cfg.layer_types[0..cfg.num_hidden_layers], 0..) |lt, i| switch (lt) {
        .linear_attention => {
            ssm_conv[i] = try buffer.Buffer.initDeviceOnly(&ctx, conv_dim * conv_kernel * @sizeOf(f32));
            ssm_rec[i] = try buffer.Buffer.initDeviceOnly(&ctx, n_v_heads * head_k * head_v * @sizeOf(f32));
        },
        .full_attention => {
            kv_k[i] = try buffer.Buffer.initDeviceOnly(&ctx, max_pos * kv_dim * @sizeOf(f32));
            kv_v[i] = try buffer.Buffer.initDeviceOnly(&ctx, max_pos * kv_dim * @sizeOf(f32));
        },
    };

    // ── Kernels ─────────────────────────────────────────────────────
    var k_embed = try pipeline.Kernel.init(&ctx, &shaders.embed_lookup, 2, @sizeOf(EmbedLookupPush));
    defer k_embed.deinit();
    var k_rmsnorm = try pipeline.Kernel.init(&ctx, &shaders.rmsnorm, 3, @sizeOf(RmsnormPush));
    defer k_rmsnorm.deinit();
    var k_matmul = try pipeline.Kernel.init(&ctx, &shaders.matmul_nt_v2, 3, @sizeOf(MatmulPush));
    defer k_matmul.deinit();
    var k_add = try pipeline.Kernel.init(&ctx, &shaders.add_in_place, 2, @sizeOf(AddInPlacePush));
    defer k_add.deinit();
    var k_swiglu = try pipeline.Kernel.init(&ctx, &shaders.swiglu, 3, @sizeOf(GegluPush));
    defer k_swiglu.deinit();
    var k_rope_p = try pipeline.Kernel.init(&ctx, &shaders.rope_partial, 2, @sizeOf(RopePartialPush));
    defer k_rope_p.deinit();
    var k_split_qg = try pipeline.Kernel.init(&ctx, &shaders.split_q_gate, 3, @sizeOf(SplitQGatePush));
    defer k_split_qg.deinit();
    var k_sigmul = try pipeline.Kernel.init(&ctx, &shaders.sigmoid_mul, 3, @sizeOf(SigmoidMulPush));
    defer k_sigmul.deinit();
    var k_l2 = try pipeline.Kernel.init(&ctx, &shaders.l2norm_per_head, 2, @sizeOf(L2normPush));
    defer k_l2.deinit();
    var k_conv1d = try pipeline.Kernel.init(&ctx, &shaders.conv1d_update, 4, @sizeOf(Conv1dUpdatePush));
    defer k_conv1d.deinit();
    var k_rms_gated = try pipeline.Kernel.init(&ctx, &shaders.rmsnorm_gated, 4, @sizeOf(RmsnormGatedPush));
    defer k_rms_gated.deinit();
    var k_gds = try pipeline.Kernel.init(&ctx, &shaders.gated_delta_step, 9, @sizeOf(GatedDeltaStepPush));
    defer k_gds.deinit();
    var k_kv_write = try pipeline.Kernel.init(&ctx, &shaders.kv_write, 2, @sizeOf(KvWritePush));
    defer k_kv_write.deinit();
    var k_scores = try pipeline.Kernel.init(&ctx, &shaders.attn_scores, 3, @sizeOf(AttnScoresPush));
    defer k_scores.deinit();
    var k_softmax = try pipeline.Kernel.init(&ctx, &shaders.softmax, 2, @sizeOf(SoftmaxPush));
    defer k_softmax.deinit();
    var k_attn_out = try pipeline.Kernel.init(&ctx, &shaders.attn_output, 3, @sizeOf(AttnOutputPush));
    defer k_attn_out.deinit();
    var k_slice_copy = try pipeline.Kernel.init(&ctx, &shaders.slice_copy, 2, @sizeOf(SliceCopyPush));
    defer k_slice_copy.deinit();
    var k_scale = try pipeline.Kernel.init(&ctx, &shaders.scale, 2, @sizeOf(ScalePush));
    defer k_scale.deinit();

    // ── Recorder: one command buffer for the entire forward ────────
    var rec = try gpu_recorder.Recorder.init(&ctx, 2048, 8192);
    defer rec.deinit();

    // ── Common push constants ───────────────────────────────────────
    const gemma_quirk: u32 = if (cfg.family.rmsnormAddOne()) 1 else 0;
    const rms_push = RmsnormPush{ .dim = hidden, .eps = cfg.rms_norm_eps, .gemma_quirk = gemma_quirk };
    const qkn_push = RmsnormPush{ .dim = head_dim, .eps = cfg.rms_norm_eps, .gemma_quirk = gemma_quirk };
    const add_push = AddInPlacePush{ .n = hidden };
    const embed_push = EmbedLookupPush{ .token_id = token_id, .dim = hidden, .scale = 1.0 };
    const swiglu_push = GegluPush{ .n = inter };
    const conv1d_push = Conv1dUpdatePush{ .conv_dim = conv_dim, .kernel_size = conv_kernel };
    const l2_push = L2normPush{ .head_dim = head_k, .eps = 1e-6 };
    const rms_gated_push = RmsnormGatedPush{ .head_dim = head_v, .eps = cfg.rms_norm_eps };
    const gds_push = GatedDeltaStepPush{
        .num_k_heads = n_k_heads_lin,
        .num_v_heads = n_v_heads,
        .head_k = head_k,
        .head_v = head_v,
    };
    const split_push = SplitQGatePush{ .num_heads = n_q_heads, .head_dim = head_dim };
    const sigmul_push = SigmoidMulPush{ .n_elem = q_dim };
    const rope_q_push = RopePartialPush{
        .n_heads = n_q_heads,
        .head_dim = head_dim,
        .rotary_dim = rotary_dim,
        .pos = 0,
        .theta_base = cfg.rope_theta,
    };
    const rope_k_push = RopePartialPush{
        .n_heads = n_kv_heads,
        .head_dim = head_dim,
        .rotary_dim = rotary_dim,
        .pos = 0,
        .theta_base = cfg.rope_theta,
    };
    const kv_write_push = KvWritePush{ .n = kv_dim, .dst_off = 0 }; // pos=0
    const scores_push = AttnScoresPush{
        .n_heads = n_q_heads,
        .heads_per_kv = heads_per_kv,
        .head_dim = head_dim,
        .n_pos = 1,
        .kv_stride = kv_dim,
        .scores_stride = max_pos,
        .inv_sqrt_dim = 1.0 / @sqrt(@as(f32, @floatFromInt(head_dim))),
    };
    const softmax_push = SoftmaxPush{ .dim = 1, .stride = max_pos };
    const attn_out_push = AttnOutputPush{
        .n_heads = n_q_heads,
        .heads_per_kv = heads_per_kv,
        .head_dim = head_dim,
        .n_pos = 1,
        .kv_stride = kv_dim,
        .scores_stride = max_pos,
    };

    const t_gpu0 = std.time.nanoTimestamp();
    try rec.begin();

    // Embed lookup → residual stream.
    try recDispatch1D(&rec, &k_embed, &.{ &gm.embed_tokens, &sc_stream }, &embed_push, hidden);

    // Diagnostic knob: stop after this many layers (env QWEN35_STOP).
    // -1 / unset means run all. Useful for layer-bisection debugging.
    const stop_after: i32 = blk: {
        const env_val = std.process.getEnvVarOwned(gpa, "QWEN35_STOP") catch break :blk -1;
        defer gpa.free(env_val);
        break :blk std.fmt.parseInt(i32, env_val, 10) catch -1;
    };

    for (gm.layers, 0..) |*layer, i| {
        if (stop_after >= 0 and @as(i32, @intCast(i)) >= stop_after) break;
        try recDispatchPerRow(&rec, &k_rmsnorm, &.{ &sc_stream, &layer.input_layernorm, &sc_x_norm }, &rms_push, 1);

        switch (layer.layer_type) {
            .linear_attention => {
                // 1. Input projections.
                try recDispatchMatmul(&rec, &k_matmul, &.{ &sc_x_norm, &layer.in_proj_qkv.?, &sc_mixed_qkv }, 1, conv_dim, hidden);
                try recDispatchMatmul(&rec, &k_matmul, &.{ &sc_x_norm, &layer.in_proj_z.?, &sc_z }, 1, value_dim, hidden);
                try recDispatchMatmul(&rec, &k_matmul, &.{ &sc_x_norm, &layer.in_proj_b.?, &sc_braw }, 1, n_v_heads, hidden);
                try recDispatchMatmul(&rec, &k_matmul, &.{ &sc_x_norm, &layer.in_proj_a.?, &sc_araw }, 1, n_v_heads, hidden);

                // 2. Causal conv1d update + SiLU. Read from
                //    sc_mixed_qkv, write to sc_mixed_qkv_post (distinct
                //    buffers to avoid Vulkan readonly/writeonly aliasing).
                try recDispatch1D(&rec, &k_conv1d, &.{ &sc_mixed_qkv, &layer.conv1d_weight.?, &ssm_conv[i].?, &sc_mixed_qkv_post }, &conv1d_push, conv_dim);

                // 3. Split mixed_qkv_post into (q_lin, k_lin, v_lin).
                //    The in_proj_qkv block lays them out contiguously:
                //    q[0..key_dim], k[key_dim..2*key_dim],
                //    v[2*key_dim..conv_dim].
                const slice_q_push = SliceCopyPush{ .src_off = 0,           .dst_off = 0, .n_elem = key_dim };
                const slice_k_push = SliceCopyPush{ .src_off = key_dim,     .dst_off = 0, .n_elem = key_dim };
                const slice_v_push = SliceCopyPush{ .src_off = 2 * key_dim, .dst_off = 0, .n_elem = value_dim };
                try recDispatch1D(&rec, &k_slice_copy, &.{ &sc_mixed_qkv_post, &sc_qlin }, &slice_q_push, key_dim);
                try recDispatch1D(&rec, &k_slice_copy, &.{ &sc_mixed_qkv_post, &sc_klin }, &slice_k_push, key_dim);
                try recDispatch1D(&rec, &k_slice_copy, &.{ &sc_mixed_qkv_post, &sc_vlin }, &slice_v_push, value_dim);

                // 4. Per-head L2-norm on Q and K (each over head_k).
                try rec.dispatch(&k_l2, &.{ &sc_qlin, &sc_qlin_n }, &l2_push, n_k_heads_lin, 1, 1);
                try rec.dispatch(&k_l2, &.{ &sc_klin, &sc_klin_n }, &l2_push, n_k_heads_lin, 1, 1);

                // 4b. Apply 1/sqrt(head_k) scale to Q (in place via
                //    a separate output slot — Vulkan disallows binding
                //    the same buffer to a writeonly + readonly slot).
                const scale_push = ScalePush{ .n = key_dim, .scale = q_scale };
                try recDispatch1D(&rec, &k_scale, &.{ &sc_qlin_n, &sc_qlin }, &scale_push, key_dim);

                // 5. Recurrent gated-delta step. After step 4b,
                //    `sc_qlin` holds the L2-normed AND scaled Q;
                //    `sc_klin_n` holds the L2-normed K (no scale).
                try rec.dispatch(&k_gds, &.{
                    &ssm_rec[i].?, &sc_qlin, &sc_klin_n, &sc_vlin,
                    &sc_braw, &sc_araw, &layer.A_log.?, &layer.dt_bias.?,
                    &sc_y,
                }, &gds_push, n_v_heads, 1, 1);

                // 6. RMSNormGated with z-gate.
                try rec.dispatch(&k_rms_gated, &.{ &sc_y, &sc_z, &layer.ssm_norm_weight.?, &sc_post_norm }, &rms_gated_push, n_v_heads, 1, 1);

                // 7. Output projection.
                try recDispatchMatmul(&rec, &k_matmul, &.{ &sc_post_norm, &layer.out_proj.?, &sc_attn_out }, 1, hidden, value_dim);
            },
            .full_attention => {
                // Q-projection 2× wide, then split into (q, gate).
                try recDispatchMatmul(&rec, &k_matmul, &.{ &sc_x_norm, &layer.q_proj.?, &sc_q_gate }, 1, q_proj_rows, hidden);
                try recDispatch1D(&rec, &k_split_qg, &.{ &sc_q_gate, &sc_q, &sc_gate_attn }, &split_push, q_dim);

                // K and V projections.
                try recDispatchMatmul(&rec, &k_matmul, &.{ &sc_x_norm, &layer.k_proj.?, &sc_k }, 1, kv_dim, hidden);
                try recDispatchMatmul(&rec, &k_matmul, &.{ &sc_x_norm, &layer.v_proj.?, &sc_v }, 1, kv_dim, hidden);

                // Per-head q_norm and k_norm, with the (1+w) form.
                try recDispatchPerRow(&rec, &k_rmsnorm, &.{ &sc_q, &layer.q_norm.?, &sc_q }, &qkn_push, n_q_heads);
                try recDispatchPerRow(&rec, &k_rmsnorm, &.{ &sc_k, &layer.k_norm.?, &sc_k }, &qkn_push, n_kv_heads);

                // Partial RoPE on Q and K.
                try recDispatch1D(&rec, &k_rope_p, &.{ &sc_q, &sc_qrot }, &rope_q_push, n_q_heads * head_dim);
                try recDispatch1D(&rec, &k_rope_p, &.{ &sc_k, &sc_krot }, &rope_k_push, n_kv_heads * head_dim);

                // KV cache write (single slot at pos=0).
                try recDispatch1D(&rec, &k_kv_write, &.{ &sc_krot, &kv_k[i].? }, &kv_write_push, kv_dim);
                try recDispatch1D(&rec, &k_kv_write, &.{ &sc_v, &kv_v[i].? }, &kv_write_push, kv_dim);

                // Attention scores → softmax → attn output.
                try rec.dispatch(&k_scores, &.{ &sc_qrot, &kv_k[i].?, &sc_scores }, &scores_push, n_q_heads * 1, 1, 1);
                try recDispatchPerRow(&rec, &k_softmax, &.{ &sc_scores, &sc_scores }, &softmax_push, n_q_heads);
                try rec.dispatch(&k_attn_out, &.{ &sc_scores, &kv_v[i].?, &sc_head_out }, &attn_out_push, n_q_heads * head_dim, 1, 1);

                // sigmoid(gate) * head_out.
                try recDispatch1D(&rec, &k_sigmul, &.{ &sc_head_out, &sc_gate_attn, &sc_head_out_gated }, &sigmul_push, q_dim);

                // o_proj.
                try recDispatchMatmul(&rec, &k_matmul, &.{ &sc_head_out_gated, &layer.o_proj.?, &sc_attn_out }, 1, hidden, q_dim);
            },
        }

        // First residual.
        try recDispatch1D(&rec, &k_add, &.{ &sc_stream, &sc_attn_out }, &add_push, hidden);

        // Post-attention norm.
        try recDispatchPerRow(&rec, &k_rmsnorm, &.{ &sc_stream, &layer.post_attention_layernorm, &sc_mid_norm }, &rms_push, 1);

        // SwiGLU MLP.
        try recDispatchMatmul(&rec, &k_matmul, &.{ &sc_mid_norm, &layer.gate_proj, &sc_gate }, 1, inter, hidden);
        try recDispatchMatmul(&rec, &k_matmul, &.{ &sc_mid_norm, &layer.up_proj, &sc_up }, 1, inter, hidden);
        try recDispatch1D(&rec, &k_swiglu, &.{ &sc_gate, &sc_up, &sc_fused }, &swiglu_push, inter);
        try recDispatchMatmul(&rec, &k_matmul, &.{ &sc_fused, &layer.down_proj, &sc_ffn_out }, 1, hidden, inter);

        // Second residual.
        try recDispatch1D(&rec, &k_add, &.{ &sc_stream, &sc_ffn_out }, &add_push, hidden);
    }

    // Final norm + LM head.
    try recDispatchPerRow(&rec, &k_rmsnorm, &.{ &sc_stream, &gm.final_norm, &sc_final_norm_out }, &rms_push, 1);
    try recDispatchMatmul(&rec, &k_matmul, &.{ &sc_final_norm_out, &gm.lm_head, &sc_logits }, 1, vocab, hidden);

    try rec.endAndSubmit();


    const t_gpu1 = std.time.nanoTimestamp();
    try stdout.print("forward (GPU): {d:.0} ms\n", .{@as(f64, @floatFromInt(t_gpu1 - t_gpu0)) / 1_000_000.0});

    // Read back logits.
    const logits = try gpa.alloc(f32, cfg.vocab_size);
    defer gpa.free(logits);
    try sc_logits.readBack(&ctx, f32, logits);

    const sampled = cpu_forward.argmax(logits);
    const out_str = tok.decode(sampled) orelse "<unknown>";

    const k_top: usize = 5;
    const top = try topK(gpa, logits, k_top);
    defer gpa.free(top);
    try stdout.print("\ntop {d} logits (GPU):\n", .{k_top});
    for (top) |entry| {
        const s = tok.decode(entry.id) orelse "<unknown>";
        try stdout.print("  id={d:>6}  logit={d:>10.4}  {s}\n", .{ entry.id, entry.value, s });
    }
    try stdout.print("\nGPU sampled (greedy): id={d}  string={s}\n", .{ sampled, out_str });

    // Parity vs CPU --gen.
    try stdout.print("\nrunning CPU forward for parity check...\n", .{});
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const cpu_logits = try gpa.alloc(f32, cfg.vocab_size);
    defer gpa.free(cpu_logits);
    var hyb_state = try cpu_forward.HybridState.init(gpa, &cpu, 1);
    defer hyb_state.deinit();
    const t_cpu0 = std.time.nanoTimestamp();
    try cpu_forward.forwardHybrid(&cpu, token_id, 0, &hyb_state, arena.allocator(), cpu_logits);
    const t_cpu1 = std.time.nanoTimestamp();
    try stdout.print("CPU forward: {d:.0} ms\n", .{@as(f64, @floatFromInt(t_cpu1 - t_cpu0)) / 1_000_000.0});

    const cpu_argmax = cpu_forward.argmax(cpu_logits);
    var max_abs: f32 = 0;
    var max_idx: usize = 0;
    for (logits, cpu_logits, 0..) |g, c, idx| {
        const d = @abs(g - c);
        if (d > max_abs) {
            max_abs = d;
            max_idx = idx;
        }
    }
    try stdout.print("CPU argmax: id={d} ({s}) logit={d:.4}\n", .{
        cpu_argmax, tok.decode(cpu_argmax) orelse "?", cpu_logits[cpu_argmax],
    });
    try stdout.print("max |Δ| over the whole logit vector = {e:.3}  (at idx {d}: cpu={d:.4} gpu={d:.4})\n", .{
        max_abs, max_idx, cpu_logits[max_idx], logits[max_idx],
    });
    if (sampled != cpu_argmax) {
        std.debug.print("FAIL: GPU argmax {d} ≠ CPU argmax {d}\n", .{ sampled, cpu_argmax });
        return error.ParityFailed;
    }
    try stdout.print("\nPASS GPU --gpu-gen-qwen35 argmax matches CPU --gen ({d} → {s})\n", .{ sampled, out_str });
}

// ── chat REPL on Qwen3.5 (multi-position GPU hybrid forward) ──────
//
// Multi-token generation with the same shader set as `--gpu-gen-qwen35`,
// extended to grow the KV cache + carry SSM state across decode steps.
// Per token:
//   - reset recorder, re-record forward with updated `pos`-dependent
//     push constants (RoPE position, kv_write offset, attention n_pos)
//   - submit, read back logits, sample, stream the decoded token
//   - SSM state buffers (conv + recurrent) are not touched by us at
//     this layer — the gated_delta_step shader updates them in place,
//     so the next decode step naturally sees the previous step's state.

const HybridChatKernels = runtime_hybrid.ChatKernels;
const HybridChatScratch = runtime_hybrid.Scratch;
const HybridChatState = runtime_hybrid.State;
const HybridTq4VHooks = runtime_hybrid.Tq4VHooks;
const HybridForwardPushes = runtime_hybrid.ForwardPushes;
const computeHybridForwardPushes = runtime_hybrid.computeForwardPushes;
const recordOneHybridLayer = runtime_hybrid.recordOneLayer;
const recordHybridForwardStep = runtime_hybrid.recordForwardStep;


/// Probed forward through the hybrid path: same dispatches as
/// recordHybridForwardStep but split across multiple submits so the
/// host can read back per-layer hidden state and (on full-attention
/// layers only) post-softmax attention weights between layers.
/// Generation output is bit-identical to the fast path because the
/// GPU does the same math in the same order — only command-buffer
/// chunking changes.
fn forwardStepProbedHybrid(
    rec: *gpu_recorder.Recorder,
    vk_ctx: *const vk.Context,
    sc: *const HybridChatScratch,
    state: *const HybridChatState,
    gm: *const gpu_model.GpuModel,
    cfg: config_mod.Config,
    k: *const HybridChatKernels,
    pos: usize,
    token_id: u32,
    max_pos: u32,
    tq4_v: ?HybridTq4VHooks,
    compute_logits: bool,
    bus: *const probe.Bus,
    info: probe.ModelInfo,
    token_index: u32,
    is_prefill: bool,
    hidden_scratch: ?[]f32,
    attn_scratch: ?[]f32,
) !void {
    const hidden: u32 = @intCast(cfg.hidden_size);
    const vocab: u32 = @intCast(cfg.vocab_size);

    const pushes = computeHybridForwardPushes(cfg, pos, max_pos);
    const embed_push = EmbedLookupPush{ .token_id = token_id, .dim = hidden, .scale = 1.0 };

    // ── Embed ───────────────────────────────────────────────────────
    try rec.reset();
    try rec.begin();
    try recDispatch1D(rec, &k.embed, &.{ &gm.embed_tokens, &sc.stream }, &embed_push, hidden);
    try rec.endAndSubmit();

    // ── Per layer ──────────────────────────────────────────────────
    for (0..cfg.num_hidden_layers) |layer_idx| {
        const layer_ctx = probe.Context{
            .info = info,
            .token_index = token_index,
            .pos = pos,
            .token_id = token_id,
            .layer_idx = @intCast(layer_idx),
            .is_prefill = is_prefill,
        };

        if (bus.needs_hidden_pre) {
            try sc.stream.readBack(vk_ctx, f32, hidden_scratch.?);
            try bus.onLayerEntry(layer_ctx, hidden_scratch.?);
        }

        try rec.reset();
        try rec.begin();
        try recordOneHybridLayer(rec, sc, state, gm, cfg, k, layer_idx, pos, &pushes, tq4_v);
        try rec.endAndSubmit();

        // sc.scores holds post-softmax attention weights ONLY on
        // full-attention layers; linear-attention layers don't write
        // it. Skip the K readback (and the K probe callback) on
        // linear layers — the analyzer aggregates per (token, layer)
        // so missing cells just don't contribute.
        const is_full_attn = gm.layers[layer_idx].layer_type == .full_attention;
        if (bus.needs_attention and is_full_attn) {
            try sc.scores.readBack(vk_ctx, f32, attn_scratch.?);
            try bus.onAttention(layer_ctx, attn_scratch.?, pushes.n_q_heads, pushes.n_pos, max_pos);
        }

        if (bus.needs_hidden_post) {
            try sc.stream.readBack(vk_ctx, f32, hidden_scratch.?);
            try bus.onLayerExit(layer_ctx, hidden_scratch.?);
        }
    }

    // ── Final norm + LM head ───────────────────────────────────────
    if (compute_logits) {
        try rec.reset();
        try rec.begin();
        try recDispatchPerRow(rec, &k.rmsnorm, &.{ &sc.stream, &gm.final_norm, &sc.final_norm_out }, &pushes.rms_push, 1);
        try recDispatchMatmul(rec, &k.matmul_lm_head, &.{ &sc.final_norm_out, &gm.lm_head, &sc.logits }, 1, vocab, hidden);
        try rec.endAndSubmit();
    }
}

fn runChatQwen35(
    gpa: std.mem.Allocator,
    dir_path: []const u8,
    single_msg: ?[]const u8,
    sample_params: cpu_forward.SampleParams,
    seed: u64,
    tq4v: bool,
    precision: gpu_model.Precision,
    probe_path: ?[]const u8,
    batch_prompts: ?[]const PromptEntry,
    probe_prefix: ?[]const u8,
    max_new: usize,
) !void {
    var cpu = try model_mod.Model.load(gpa, dir_path);
    defer cpu.deinit();
    const cfg = cpu.config;
    if (!cfg.family.isHybrid()) return error.NotHybridFamily;

    const tok_path = try std.fmt.allocPrint(gpa, "{s}/tokenizer.json", .{dir_path});
    defer gpa.free(tok_path);
    var tok = try tokenizer_mod.Tokenizer.loadFromFile(gpa, tok_path);
    defer tok.deinit();

    var ctx = try vk.Context.init(gpa);
    defer ctx.deinit();

    const stdout = std.io.getStdOut().writer();
    try stdout.print("device: {s}\n", .{ctx.deviceName()});
    const prec_label: []const u8 = switch (precision) {
        .fp32_all => "fp32",
        .bf16_matmul => "bf16",
        .q4_0_matmul => "Q4_0",
        .q4_k_matmul => "Q4_K_M",
    };
    if (tq4v) {
        try stdout.print("uploading weights ({s} + TQ4-V on full-attn layers)...\n", .{prec_label});
    } else {
        try stdout.print("uploading weights ({s} path)...\n", .{prec_label});
    }
    const t_up0 = std.time.nanoTimestamp();
    var gm = try gpu_model.GpuModel.upload(gpa, &ctx, &cpu, precision);
    defer gm.deinit(ctx.device);
    const t_up1 = std.time.nanoTimestamp();
    try stdout.print("  upload took {d:.0} ms\n", .{@as(f64, @floatFromInt(t_up1 - t_up0)) / 1_000_000.0});

    // TQ4-V is only wired for head_dim ∈ {128, 256}. Both Qwen3.5 sizes
    // (0.8B, 4B) are 128, so this is mostly a guard against future
    // hybrids with odd head_dim landing on this path.
    if (tq4v and cfg.head_dim != 128 and cfg.head_dim != 256) {
        try stdout.print("note: --tq4v ignored — head_dim={d} has no TQ4 shader pair (only 128/256 wired)\n", .{cfg.head_dim});
    }
    const tq4v_active = tq4v and (cfg.head_dim == 128 or cfg.head_dim == 256);

    const max_pos: u32 = 2048;
    var sc = try HybridChatScratch.init(&ctx, cfg, max_pos, tq4v_active);
    defer sc.deinit(ctx.device);
    var state = try HybridChatState.init(gpa, &ctx, cfg, max_pos, tq4v_active);
    defer state.deinit(ctx.device);
    var ks = try HybridChatKernels.init(&ctx, gm.precision);
    defer ks.deinit();

    // TQ4 pack/unpack kernels — picked by head_dim, only built when
    // active. Mirrors the Gemma TQ4 path. Number of blocks per token =
    // num_kv_heads (8 for Qwen3.5).
    const tq_pack_spv: []align(4) const u8 = if (cfg.head_dim == 128)
        &shaders.tq4_pack_to_cache128
    else
        &shaders.tq4_pack_to_cache;
    const tq_unpack_spv: []align(4) const u8 = if (cfg.head_dim == 128)
        &shaders.tq4_unpack128
    else
        &shaders.tq4_unpack256;
    var tq_pack: ?pipeline.Kernel = if (tq4v_active)
        try pipeline.Kernel.init(&ctx, tq_pack_spv, 2, @sizeOf(Tq4PackPush))
    else
        null;
    defer if (tq_pack) |*kk| kk.deinit();
    var tq_unpack: ?pipeline.Kernel = if (tq4v_active)
        try pipeline.Kernel.init(&ctx, tq_unpack_spv, 2, 0)
    else
        null;
    defer if (tq_unpack) |*kk| kk.deinit();
    const tq4_hooks: ?HybridTq4VHooks = if (tq4v_active) HybridTq4VHooks{
        .pack = &tq_pack.?,
        .unpack = &tq_unpack.?,
        .n_blocks_per_pos = @intCast(cfg.num_key_value_heads),
    } else null;

    // Recorder pool sized for one full forward step. ~30 dispatches per
    // linear layer + ~16 per full layer + ~3 head/tail = up to ~30*32 +
    // 3 ≈ 1k sets. Round generously.
    const sets_per_step: u32 = @intCast(@max(@as(usize, 1024), cfg.num_hidden_layers * 40));
    var rec = try gpu_recorder.Recorder.init(&ctx, sets_per_step, sets_per_step * 4);
    defer rec.deinit();

    const tmpl = try chat_template_mod.ChatTemplate.resolve(cfg.family, &tok);

    const logits = try gpa.alloc(f32, cfg.vocab_size);
    defer gpa.free(logits);
    const sample_scratch = try gpa.alloc(f32, cfg.vocab_size * 2);
    defer gpa.free(sample_scratch);

    var prng = std.Random.DefaultPrng.init(seed);
    const rng = prng.random();

    if (sample_params.temperature == 0.0 or sample_params.top_k == 1 or
        (sample_params.top_p >= 1.0 and sample_params.top_k == 0 and sample_params.temperature == 1.0))
    {
        try stdout.print("sampling: greedy\n", .{});
    } else {
        try stdout.print("sampling: temp={d} top_k={d} top_p={d} seed={d}\n", .{
            sample_params.temperature, sample_params.top_k, sample_params.top_p, seed,
        });
    }

    // ── Probe wiring ─────────────────────────────────────────────────
    //
    // Mirrors the non-hybrid runChat probe setup. Null prior is
    // computed by running a single BOS forward at pos 0; the chat
    // loop then starts at pos 0 too, overwriting both KV state and
    // SSM state on the first prompt token. K is only valid on
    // full-attention layers in the hybrid family — the probed forward
    // skips the K readback on linear-attention layers.
    const probe_info = probe.ModelInfo{
        .family = @tagName(cfg.family),
        .n_layers = @intCast(cfg.num_hidden_layers),
        .hidden_size = @intCast(cfg.hidden_size),
        .n_heads = @intCast(cfg.num_attention_heads),
        .n_kv_heads = @intCast(cfg.num_key_value_heads),
        .head_dim = @intCast(cfg.head_dim),
        .vocab_size = @intCast(cfg.vocab_size),
    };

    var probe_writer: ?probe.JsonlWriter = null;
    defer if (probe_writer) |*w| w.close();
    var probe_bus: ?probe.Bus = null;
    defer if (probe_bus) |*b| b.deinit(gpa);
    var probe_hidden_scratch: ?[]f32 = null;
    defer if (probe_hidden_scratch) |s| gpa.free(s);
    var probe_attn_scratch: ?[]f32 = null;
    defer if (probe_attn_scratch) |s| gpa.free(s);
    var probe_token_index: u32 = 0;

    if (probe_path) |pp| {
        probe_writer = try probe.JsonlWriter.open(pp);
        const w_ptr: *probe.JsonlWriter = &probe_writer.?;

        try stdout.print("probe: writing trace to {s}\n", .{pp});

        const bos: u32 = cfg.bos_token_id orelse 1;
        try stdout.print("probe: computing null prior (BOS={d})...\n", .{bos});
        try rec.reset();
        try rec.begin();
        try recordHybridForwardStep(&rec, &sc, &state, &gm, cfg, &ks, 0, bos, max_pos, tq4_hooks, true);
        try rec.endAndSubmit();
        try sc.logits.readBack(&ctx, f32, logits);
        // KV/SSM slots at pos 0 are now dirty with the BOS forward.
        // The chat loop below starts at pos = 0 and overwrites both
        // on the first prompt token — clean handoff, no reset needed.

        var bus = probe.Bus{};
        const probes_buf = try gpa.alloc(probe.Probe, 3);
        const lp = try probe.LogitProbe.create(gpa, w_ptr, @intCast(cfg.vocab_size), logits);
        const ap = try probe.ActivationEntropyProbe.create(gpa, w_ptr);
        const kp = try probe.AttentionProbe.create(gpa, w_ptr);
        probes_buf[0] = lp.probe();
        probes_buf[1] = ap.probe();
        probes_buf[2] = kp.probe();
        bus.probes = probes_buf;
        bus.finalize();
        probe_bus = bus;

        try probe.writeHeader(w_ptr, probe_info, dir_path, &.{ "logits", "act", "attn" });

        if (bus.needs_hidden_pre or bus.needs_hidden_post) {
            probe_hidden_scratch = try gpa.alloc(f32, cfg.hidden_size);
        }
        if (bus.needs_attention) {
            probe_attn_scratch = try gpa.alloc(f32, cfg.num_attention_heads * max_pos);
        }
    }

    // ── Batch mode (hybrid path) ─────────────────────────────────────
    //
    // Same shape as runChat's batch path. Additional reset between
    // prompts: zero the per-layer SSM (conv + recurrent) state via
    // HybridChatState.reset, since Gated DeltaNet's recurrent state
    // is a constant-size vector that always matters regardless of
    // pos and would otherwise leak the previous prompt's trajectory
    // into the next prompt's first decode steps.
    if (batch_prompts) |entries| {
        const bos: u32 = cfg.bos_token_id orelse 1;
        try stdout.print("probe (batch): computing null prior (BOS={d})...\n", .{bos});
        const null_prior_buf = try gpa.alloc(f32, cfg.vocab_size);
        defer gpa.free(null_prior_buf);
        try rec.reset();
        try rec.begin();
        try recordHybridForwardStep(&rec, &sc, &state, &gm, cfg, &ks, 0, bos, max_pos, tq4_hooks, true);
        try rec.endAndSubmit();
        try sc.logits.readBack(&ctx, f32, null_prior_buf);

        const hidden_scratch = try gpa.alloc(f32, cfg.hidden_size);
        defer gpa.free(hidden_scratch);
        const attn_scratch = try gpa.alloc(f32, cfg.num_attention_heads * max_pos);
        defer gpa.free(attn_scratch);

        for (entries) |entry| {
            const out_path = try std.fmt.allocPrint(gpa, "{s}{s}.jsonl", .{ probe_prefix.?, entry.label });
            defer gpa.free(out_path);
            try stdout.print("\n[{s}] → {s}\n", .{ entry.label, out_path });

            // Reset SSM state to zero before this prompt's prefill.
            try state.reset(&ctx);

            var writer = try probe.JsonlWriter.open(out_path);
            defer writer.close();

            var bus = probe.Bus{};
            const probes_buf = try gpa.alloc(probe.Probe, 3);
            const lp = try probe.LogitProbe.create(gpa, &writer, @intCast(cfg.vocab_size), null_prior_buf);
            const ap = try probe.ActivationEntropyProbe.create(gpa, &writer);
            const kp = try probe.AttentionProbe.create(gpa, &writer);
            probes_buf[0] = lp.probe();
            probes_buf[1] = ap.probe();
            probes_buf[2] = kp.probe();
            bus.probes = probes_buf;
            bus.finalize();
            defer bus.deinit(gpa);

            try probe.writeHeader(&writer, probe_info, dir_path, &.{ "logits", "act", "attn" });

            var pos: usize = 0;
            var token_index: u32 = 0;

            try chatTurnHybrid(
                gpa, &ctx, &rec, &sc, &state, &gm, cfg, &ks, &tok, entry.text, &pos, logits,
                sample_scratch, sample_params, rng, tmpl, false, max_pos, tq4_hooks,
                &bus, probe_info, &token_index, hidden_scratch, attn_scratch,
                max_new,
            );
        }
        return;
    }

    var pos: usize = 0;

    if (single_msg) |m| {
        try chatTurnHybrid(
            gpa, &ctx, &rec, &sc, &state, &gm, cfg, &ks, &tok, m, &pos, logits,
            sample_scratch, sample_params, rng, tmpl, false, max_pos, tq4_hooks,
            if (probe_bus) |*b| b else null,
            probe_info,
            &probe_token_index,
            probe_hidden_scratch,
            probe_attn_scratch,
            max_new,
        );
        return;
    }

    try stdout.print("\n{s} (Ctrl-D to exit)\n", .{tmpl.banner()});
    const stdin = std.io.getStdIn().reader();
    var line_buf: [4096]u8 = undefined;
    while (true) {
        try stdout.print("\nuser> ", .{});
        const maybe_line = try stdin.readUntilDelimiterOrEof(&line_buf, '\n');
        const line = maybe_line orelse {
            try stdout.print("\n", .{});
            break;
        };
        if (line.len == 0) continue;
        try chatTurnHybrid(
            gpa, &ctx, &rec, &sc, &state, &gm, cfg, &ks, &tok, line, &pos, logits,
            sample_scratch, sample_params, rng, tmpl, true, max_pos, tq4_hooks,
            if (probe_bus) |*b| b else null,
            probe_info,
            &probe_token_index,
            probe_hidden_scratch,
            probe_attn_scratch,
            max_new,
        );
        if (pos >= max_pos - 64) {
            try stdout.print("\n[KV cache near capacity, ending session]\n", .{});
            break;
        }
    }
}

fn chatTurnHybrid(
    gpa: std.mem.Allocator,
    ctx: *const vk.Context,
    rec: *gpu_recorder.Recorder,
    sc: *const HybridChatScratch,
    state: *const HybridChatState,
    gm: *const gpu_model.GpuModel,
    cfg: config_mod.Config,
    ks: *const HybridChatKernels,
    tok: *const tokenizer_mod.Tokenizer,
    user_msg: []const u8,
    pos: *usize,
    logits: []f32,
    sample_scratch: []f32,
    sample_params: cpu_forward.SampleParams,
    rng: std.Random,
    tmpl: chat_template_mod.ChatTemplate,
    is_repl: bool,
    max_pos: u32,
    tq4_v: ?HybridTq4VHooks,
    probe_bus: ?*probe.Bus,
    probe_info: probe.ModelInfo,
    probe_token_index: *u32,
    probe_hidden_scratch: ?[]f32,
    probe_attn_scratch: ?[]f32,
    max_new: usize,
) !void {
    const stdout = std.io.getStdOut().writer();

    var prompt = std.ArrayList(u32).init(gpa);
    defer prompt.deinit();
    try tmpl.composePrompt(gpa, tok, user_msg, pos.* == 0, &prompt);

    if (!is_repl) {
        try stdout.print("\nprompt ({d} tokens, starting at pos {d}):\n  ", .{ prompt.items.len, pos.* });
        for (prompt.items) |id| try printTokenForDisplay(gpa, stdout, tok, id);
        try stdout.print("\n\nresponse: ", .{});
    } else {
        try stdout.print("model> ", .{});
    }

    const eos: ?u32 = cfg.eos_token_id;
    const eot: u32 = tmpl.end_of_turn;
    const max_response: usize = max_new;

    var current: u32 = prompt.items[0];
    var prompt_idx: usize = 0;
    var generated: usize = 0;

    // Time decode-only throughput: clock starts after prefill consumes
    // the prompt, so the number reflects steady-state generation speed.
    var t_decode_start: i128 = 0;
    var decode_started = false;

    while (true) {
        // Skip the LM head matmul (the largest in the model) on every
        // prefill token except the last — see chatTurn for the rationale.
        const is_last_prefill_or_decoding = (prompt_idx + 1 >= prompt.items.len);
        const is_prefill = (prompt_idx + 1 < prompt.items.len);

        if (probe_bus) |bus| try bus.onTokenStart(.{
            .info = probe_info,
            .token_index = probe_token_index.*,
            .pos = pos.*,
            .token_id = current,
            .is_prefill = is_prefill,
        });

        const use_probed_forward = if (probe_bus) |bus|
            bus.needs_hidden_pre or bus.needs_hidden_post or bus.needs_attention
        else
            false;

        if (use_probed_forward) {
            try forwardStepProbedHybrid(
                rec,
                ctx,
                sc,
                state,
                gm,
                cfg,
                ks,
                pos.*,
                current,
                max_pos,
                tq4_v,
                is_last_prefill_or_decoding,
                probe_bus.?,
                probe_info,
                probe_token_index.*,
                is_prefill,
                probe_hidden_scratch,
                probe_attn_scratch,
            );
        } else {
            if (pos.* > 0) try rec.reset();
            try rec.begin();
            try recordHybridForwardStep(rec, sc, state, gm, cfg, ks, pos.*, current, max_pos, tq4_v, is_last_prefill_or_decoding);
            try rec.endAndSubmit();
        }

        if (probe_bus) |bus| {
            if (is_last_prefill_or_decoding and bus.needs_logits) {
                try sc.logits.readBack(ctx, f32, logits);
                try bus.onLogits(.{
                    .info = probe_info,
                    .token_index = probe_token_index.*,
                    .pos = pos.*,
                    .token_id = current,
                    .is_prefill = is_prefill,
                }, logits);
            }
        }

        prompt_idx += 1;
        pos.* += 1;
        probe_token_index.* += 1;

        if (prompt_idx < prompt.items.len) {
            current = prompt.items[prompt_idx];
            continue;
        }

        if (!decode_started) {
            t_decode_start = std.time.nanoTimestamp();
            decode_started = true;
        }

        try sc.logits.readBack(ctx, f32, logits);
        const next = try cpu_forward.sample(logits, sample_params, rng, sample_scratch);

        if (probe_bus) |bus| try bus.onTokenEnd(.{
            .info = probe_info,
            .token_index = probe_token_index.*,
            .pos = pos.*,
            .token_id = current,
            .is_prefill = false,
        }, @intCast(next));

        if (next == eot) break;
        if (eos != null and next == eos.?) break;

        try printTokenForDisplay(gpa, stdout, tok, @intCast(next));

        generated += 1;
        if (generated >= max_response) break;
        current = @intCast(next);
    }
    try stdout.print("\n", .{});

    if (decode_started and generated > 0) {
        const t_end = std.time.nanoTimestamp();
        const ms_total = @as(f64, @floatFromInt(t_end - t_decode_start)) / 1_000_000.0;
        const tokps = @as(f64, @floatFromInt(generated)) * 1000.0 / ms_total;
        try stdout.print("[{d} tok in {d:.0} ms, {d:.1} tok/s]\n", .{ generated, ms_total, tokps });
    }
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
