//! GPU backward + LoRA + CCE + matmul-variant + LayerNorm parity smokes.
//! These are the training-flavoured kernel smokes: backward kernels for
//! every primitive, the four-corner CCE chain (forward/backward d_h/dW),
//! LoRA composition + train demos, V2 matmul tile, Q4_0/Q4_K matmul,
//! plus the "embedded-host" attach + recorder smokes that exercise the
//! valkyr_gpu library surface from main.zig.
//! Extracted from main.zig.

const std = @import("std");
const vk = @import("../gpu/vk.zig");
const buffer = @import("../gpu/buffer.zig");
const pipeline = @import("../gpu/pipeline.zig");
const gpu_recorder = @import("../gpu/recorder.zig");
const cpu_math = @import("../cpu/math.zig");
const cpu_train_transformer = @import("../cpu/train_transformer.zig");
const cpu_cce = @import("../cpu/cce.zig");
const cpu_lora = @import("../cpu/lora.zig");
const train_lora = @import("../train/lora.zig");
const train_transformer = @import("../train/transformer.zig");
const config_mod = @import("../config.zig");
const safetensors = @import("../safetensors.zig");
const q4_0 = @import("../cpu/q4_0.zig");
const q4_k = @import("../cpu/q4_k.zig");
const runtime = @import("../runtime.zig");
const shaders = @import("shaders");

const aliases = @import("../runtime_aliases.zig");
const helpers = @import("../smoke/helpers.zig");
const util = @import("../util.zig");

pub fn runGpuMatmulSmoke(allocator: std.mem.Allocator) !void {
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

    var kern = try pipeline.Kernel.init(&ctx, &shaders.matmul_nt, 3, @sizeOf(aliases.MatmulPush));
    defer kern.deinit();
    try kern.bind(&.{ &buf_a, &buf_b, &buf_c });

    const local_xy: u32 = 16;
    const groups_x: u32 = (m + local_xy - 1) / local_xy;
    const groups_y: u32 = (n + local_xy - 1) / local_xy;
    const push = aliases.MatmulPush{ .m = m, .n = n, .k = k };

    try buffer.submitOneShot(&ctx, struct {
        kern: *const pipeline.Kernel,
        push: *const aliases.MatmulPush,
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

pub fn runGpuCceForwardSmoke(allocator: std.mem.Allocator) !void {
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

pub fn runGpuLoraSmoke(allocator: std.mem.Allocator) !void {
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
    var k_matmul = try pipeline.Kernel.init(&ctx, &shaders.matmul_nt_v2, 3, @sizeOf(aliases.MatmulPush));
    defer k_matmul.deinit();
    var k_lin_dx = try pipeline.Kernel.init(&ctx, &shaders.linear_backward_dx_batched, 3, @sizeOf(runtime.LinearBatchedPush));
    defer k_lin_dx.deinit();
    var k_lin_dw = try pipeline.Kernel.init(&ctx, &shaders.linear_backward_dw_batched, 3, @sizeOf(runtime.LinearBatchedPush));
    defer k_lin_dw.deinit();
    var k_scale = try pipeline.Kernel.init(&ctx, &shaders.scale, 2, @sizeOf(aliases.ScalePush));
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
            fn rec(rec_kern: *pipeline.Kernel, push: *const aliases.MatmulPush, gx: u32, c_ctx: *const vk.Context, bufs: []const *const buffer.Buffer) !void {
                try rec_kern.bind(bufs);
                const Rec = struct {
                    k: *const pipeline.Kernel,
                    p: *const aliases.MatmulPush,
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
        const push_y_base = aliases.MatmulPush{ .m = M_u32, .n = N_u32, .k = K_u32 };
        try recordMatmul(&k_matmul, &push_y_base, M_u32 * N_u32, &ctx, &.{ &buf_x, &buf_w, &buf_y });
        // 2. intermediate = x · Aᵀ
        const push_inter = aliases.MatmulPush{ .m = M_u32, .n = R_u32, .k = K_u32 };
        try recordMatmul(&k_matmul, &push_inter, M_u32 * R_u32, &ctx, &.{ &buf_x, &buf_a, &buf_intermediate });
        // 3. y_lora = intermediate · Bᵀ
        const push_ylora = aliases.MatmulPush{ .m = M_u32, .n = N_u32, .k = R_u32 };
        try recordMatmul(&k_matmul, &push_ylora, M_u32 * N_u32, &ctx, &.{ &buf_intermediate, &buf_b, &buf_y_lora });
        // 4. y_lora_scaled = y_lora * (α/r)
        const push_scale_ylora = aliases.ScalePush{ .n = M_u32 * N_u32, .scale = cs.aor };
        try recordScalar1D(&k_scale, push_scale_ylora, util.ceilDiv(M_u32 * N_u32, group_lin), &ctx, &.{ &buf_y_lora, &buf_y_lora_scaled });
        // 5. y += y_lora_scaled  (in place into buf_y)
        const push_add_y = runtime.AddInPlacePush{ .n = M_u32 * N_u32 };
        try recordScalar1D(&k_add, push_add_y, util.ceilDiv(M_u32 * N_u32, group_lin), &ctx, &.{ &buf_y, &buf_y_lora_scaled });

        // ── BACKWARD chain.
        // 1. dx_base = dy · W                  (shape M×K = M×Nn · Nn×K)
        const push_dx_base = runtime.LinearBatchedPush{ .M = M_u32, .N = N_u32, .K = K_u32 };
        try recordLinBackward(&k_lin_dx, &push_dx_base, util.ceilDiv(M_u32, group_lwg), util.ceilDiv(K_u32, group_lwg), &ctx, &.{ &buf_dy, &buf_w, &buf_dx });
        // 2. dy_B = dy · B                     (shape M×r = M×Nn · Nn×r). LinearBatchedPush.K = r.
        const push_dy_B = runtime.LinearBatchedPush{ .M = M_u32, .N = N_u32, .K = R_u32 };
        try recordLinBackward(&k_lin_dx, &push_dy_B, util.ceilDiv(M_u32, group_lwg), util.ceilDiv(R_u32, group_lwg), &ctx, &.{ &buf_dy, &buf_b, &buf_dy_B });
        // 3. dx_lora = dy_B · A                (shape M×K = M×r · r×K). Nn=r in the linear-backward sense.
        const push_dx_lora = runtime.LinearBatchedPush{ .M = M_u32, .N = R_u32, .K = K_u32 };
        try recordLinBackward(&k_lin_dx, &push_dx_lora, util.ceilDiv(M_u32, group_lwg), util.ceilDiv(K_u32, group_lwg), &ctx, &.{ &buf_dy_B, &buf_a, &buf_dx_lora });
        // 4. dx_lora_scaled = dx_lora * (α/r)
        const push_scale_dxlora = aliases.ScalePush{ .n = M_u32 * K_u32, .scale = cs.aor };
        try recordScalar1D(&k_scale, push_scale_dxlora, util.ceilDiv(M_u32 * K_u32, group_lin), &ctx, &.{ &buf_dx_lora, &buf_dx_lora_scaled });
        // 5. dx += dx_lora_scaled
        const push_add_dx = runtime.AddInPlacePush{ .n = M_u32 * K_u32 };
        try recordScalar1D(&k_add, push_add_dx, util.ceilDiv(M_u32 * K_u32, group_lin), &ctx, &.{ &buf_dx, &buf_dx_lora_scaled });
        // 6. ∇A_unscaled = dy_Bᵀ · x          (shape r×K). dW[Nn=r, K]
        const push_dA = runtime.LinearBatchedPush{ .M = M_u32, .N = R_u32, .K = K_u32 };
        try recordLinBackward(&k_lin_dw, &push_dA, util.ceilDiv(R_u32, group_lwg), util.ceilDiv(K_u32, group_lwg), &ctx, &.{ &buf_dy_B, &buf_x, &buf_dA_unscaled });
        // 7. ∇A = ∇A_unscaled * (α/r)
        const push_scale_dA = aliases.ScalePush{ .n = R_u32 * K_u32, .scale = cs.aor };
        try recordScalar1D(&k_scale, push_scale_dA, util.ceilDiv(R_u32 * K_u32, group_lin), &ctx, &.{ &buf_dA_unscaled, &buf_dA });
        // 8. ∇B_unscaled = dyᵀ · intermediate (shape Nn×r). dW[Nn, K=r]
        const push_dB = runtime.LinearBatchedPush{ .M = M_u32, .N = N_u32, .K = R_u32 };
        try recordLinBackward(&k_lin_dw, &push_dB, util.ceilDiv(N_u32, group_lwg), util.ceilDiv(R_u32, group_lwg), &ctx, &.{ &buf_dy, &buf_intermediate, &buf_dB_unscaled });
        // 9. ∇B = ∇B_unscaled * (α/r)
        const push_scale_dB = aliases.ScalePush{ .n = N_u32 * R_u32, .scale = cs.aor };
        try recordScalar1D(&k_scale, push_scale_dB, util.ceilDiv(N_u32 * R_u32, group_lin), &ctx, &.{ &buf_dB_unscaled, &buf_dB });

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

// ── LoRA Recorder-based dispatch smoke (foundation for Runner integration) ──
//
// Same parity gate as runGpuLoraSmoke (LoRA forward + backward against
// the cpu/lora.zig oracle, three rank/shape cases) but routes every
// dispatch through `gpu_recorder.Recorder` and the helpers in
// `src/train/lora.zig`. This is the path the in-Runner LoRA integration
// (A4-2+) will use — proves the helpers compose correctly under one
// cmdbuf + one submit before we wire them into transformer.Runner.step.
//
// Difference vs runGpuLoraSmoke: that one fires 14 separate submitOneShot
// calls per case (each with its own vkQueueWaitIdle); this one fires
// one submit per case with all 14 dispatches recorded sequentially in
// a single cmdbuf. Recorder injects a memory barrier between dispatches,
// matching the in-Runner pattern.

pub fn runGpuLoraRecorderSmoke(allocator: std.mem.Allocator) !void {
    var ctx = try vk.Context.init(allocator);
    defer ctx.deinit();

    const Case = struct {
        name: []const u8,
        m: u32,
        n: u32,
        k: u32,
        r: u32,
        aor: f32,
    };
    const cases = [_]Case{
        .{ .name = "rank-4 (M=4 N=8 K=16)", .m = 4, .n = 8, .k = 16, .r = 4, .aor = 2.0 },
        .{ .name = "rank-16 (M=8 N=64 K=128)", .m = 8, .n = 64, .k = 128, .r = 16, .aor = 2.0 },
        .{ .name = "rank-32 (M=4 N=32 K=64)", .m = 4, .n = 32, .k = 64, .r = 32, .aor = 1.0 },
    };

    // Pipelines reused across cases.
    var k_matmul = try pipeline.Kernel.init(&ctx, &shaders.matmul_nt_v2, 3, @sizeOf(aliases.MatmulPush));
    defer k_matmul.deinit();
    var k_lin_dx = try pipeline.Kernel.init(&ctx, &shaders.linear_backward_dx_batched, 3, @sizeOf(runtime.LinearBatchedPush));
    defer k_lin_dx.deinit();
    var k_lin_dw = try pipeline.Kernel.init(&ctx, &shaders.linear_backward_dw_batched, 3, @sizeOf(runtime.LinearBatchedPush));
    defer k_lin_dw.deinit();
    var k_scale = try pipeline.Kernel.init(&ctx, &shaders.scale, 2, @sizeOf(aliases.ScalePush));
    defer k_scale.deinit();
    var k_add = try pipeline.Kernel.init(&ctx, &shaders.add_in_place, 2, @sizeOf(runtime.AddInPlacePush));
    defer k_add.deinit();

    const kernels = train_lora.LoraKernels{
        .matmul = &k_matmul,
        .lin_dx = &k_lin_dx,
        .lin_dw = &k_lin_dw,
        .scale = &k_scale,
        .add_in_place = &k_add,
    };

    // Recorder large enough for ~32 sets — well over the 14 we need
    // per case, but cheap to over-provision.
    var rec = try gpu_recorder.Recorder.init(&ctx, 32, 32 * 8);
    defer rec.deinit();

    for (cases) |cs| {
        const M: usize = cs.m;
        const Nn: usize = cs.n;
        const K: usize = cs.k;
        const r: usize = cs.r;

        // ── Host-side init.
        var prng = std.Random.DefaultPrng.init(0xABBA_BEEF +% @as(u64, cs.m) *% 1000 +% cs.n);
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

        // ── CPU oracle.
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

        // ── Device buffers.
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
        var sc_y_lora = try buffer.Buffer.initDeviceOnly(&ctx, M * Nn * @sizeOf(f32));
        defer sc_y_lora.deinit(ctx.device);
        var sc_y_lora_scaled = try buffer.Buffer.initDeviceOnly(&ctx, M * Nn * @sizeOf(f32));
        defer sc_y_lora_scaled.deinit(ctx.device);
        var sc_dy_B = try buffer.Buffer.initDeviceOnly(&ctx, M * r * @sizeOf(f32));
        defer sc_dy_B.deinit(ctx.device);
        var sc_dx_lora = try buffer.Buffer.initDeviceOnly(&ctx, M * K * @sizeOf(f32));
        defer sc_dx_lora.deinit(ctx.device);
        var sc_dx_lora_scaled = try buffer.Buffer.initDeviceOnly(&ctx, M * K * @sizeOf(f32));
        defer sc_dx_lora_scaled.deinit(ctx.device);
        var sc_dA_unscaled = try buffer.Buffer.initDeviceOnly(&ctx, r * K * @sizeOf(f32));
        defer sc_dA_unscaled.deinit(ctx.device);
        var sc_dB_unscaled = try buffer.Buffer.initDeviceOnly(&ctx, Nn * r * @sizeOf(f32));
        defer sc_dB_unscaled.deinit(ctx.device);

        // ── Record forward + backward in one cmdbuf, submit once.
        try rec.reset();
        try rec.begin();
        try train_lora.recordLoraForward(
            &rec,
            kernels,
            .{
                .x = &buf_x,
                .w = &buf_w,
                .a = &buf_a,
                .b = &buf_b,
                .y = &buf_y,
                .intermediate_out = &buf_intermediate,
                .sc_y_lora = &sc_y_lora,
                .sc_y_lora_scaled = &sc_y_lora_scaled,
            },
            .{ .M = cs.m, .N = cs.n, .K = cs.k, .r = cs.r, .alpha_over_r = cs.aor },
        );
        try train_lora.recordLoraBackward(
            &rec,
            kernels,
            .{
                .dy = &buf_dy,
                .x = &buf_x,
                .w = &buf_w,
                .a = &buf_a,
                .b = &buf_b,
                .intermediate = &buf_intermediate,
                .dx = &buf_dx,
                .dA = &buf_dA,
                .dB = &buf_dB,
                .sc_dy_B = &sc_dy_B,
                .sc_dx_lora = &sc_dx_lora,
                .sc_dx_lora_scaled = &sc_dx_lora_scaled,
                .sc_dA_unscaled = &sc_dA_unscaled,
                .sc_dB_unscaled = &sc_dB_unscaled,
            },
            .{ .M = cs.m, .N = cs.n, .K = cs.k, .r = cs.r, .alpha_over_r = cs.aor },
        );
        try rec.endAndSubmit();

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
                "LoRA Recorder smoke ({s}) FAIL: y_rel={e}  dx_rel={e}  dA_rel={e}  dB_rel={e}  (tol {e})\n",
                .{ cs.name, y_rel, dx_rel, dA_rel, dB_rel, tol },
            );
            return error.ParityFailed;
        }
        std.debug.print(
            "PASS GPU LoRA via Recorder — {s}  α/r={d:.2}  rel(y/dx/dA/dB)=({e},{e},{e},{e})\n",
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

pub fn runGpuLoraTrainDemo(allocator: std.mem.Allocator) !void {
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
    var k_matmul = try pipeline.Kernel.init(&ctx, &shaders.matmul_nt_v2, 3, @sizeOf(aliases.MatmulPush));
    defer k_matmul.deinit();
    var k_lin_dx = try pipeline.Kernel.init(&ctx, &shaders.linear_backward_dx_batched, 3, @sizeOf(runtime.LinearBatchedPush));
    defer k_lin_dx.deinit();
    var k_lin_dw = try pipeline.Kernel.init(&ctx, &shaders.linear_backward_dw_batched, 3, @sizeOf(runtime.LinearBatchedPush));
    defer k_lin_dw.deinit();
    var k_scale = try pipeline.Kernel.init(&ctx, &shaders.scale, 2, @sizeOf(aliases.ScalePush));
    defer k_scale.deinit();
    var k_add = try pipeline.Kernel.init(&ctx, &shaders.add_in_place, 2, @sizeOf(runtime.AddInPlacePush));
    defer k_add.deinit();
    var k_mse_grad = try pipeline.Kernel.init(&ctx, &shaders.mse_loss_grad, 3, @sizeOf(aliases.MseLossGradPush));
    defer k_mse_grad.deinit();
    var k_adam = try pipeline.Kernel.init(&ctx, &shaders.adam_step, 4, @sizeOf(runtime.AdamStepPush));
    defer k_adam.deinit();

    const group_lwg: u32 = 16;
    const group_lin: u32 = 256;

    // Helpers.
    const recordMatmul = struct {
        fn rec(rec_kern: *pipeline.Kernel, push: *const aliases.MatmulPush, gx: u32, c_ctx: *const vk.Context, bufs: []const *const buffer.Buffer) !void {
            try rec_kern.bind(bufs);
            const Rec = struct {
                k: *const pipeline.Kernel,
                p: *const aliases.MatmulPush,
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
    const push_y_base = aliases.MatmulPush{ .m = M_u, .n = N_u, .k = K_u };
    const push_inter = aliases.MatmulPush{ .m = M_u, .n = R_u, .k = K_u };
    const push_ylora = aliases.MatmulPush{ .m = M_u, .n = N_u, .k = R_u };
    const push_scale_ylora = aliases.ScalePush{ .n = M_u * N_u, .scale = aor };
    const push_add_y = runtime.AddInPlacePush{ .n = M_u * N_u };
    const push_mse = aliases.MseLossGradPush{ .n = M_u * N_u };
    const push_dy_B = runtime.LinearBatchedPush{ .M = M_u, .N = N_u, .K = R_u };
    const push_dA = runtime.LinearBatchedPush{ .M = M_u, .N = R_u, .K = K_u };
    const push_scale_dA = aliases.ScalePush{ .n = R_u * K_u, .scale = aor };
    const push_dB = runtime.LinearBatchedPush{ .M = M_u, .N = N_u, .K = R_u };
    const push_scale_dB = aliases.ScalePush{ .n = N_u * R_u, .scale = aor };

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
            pyb: *const aliases.MatmulPush,
            pin: *const aliases.MatmulPush,
            pyl: *const aliases.MatmulPush,
            psy: *const aliases.ScalePush,
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
            try recS_fn(ksc, psy.*, util.ceilDiv(mu * nu, glin), c_ctx, &.{ bylora, byloras });
            try recS_fn(kad, pay.*, util.ceilDiv(mu * nu, glin), c_ctx, &.{ by, byloras });
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
        try recordScalar1D(&k_mse_grad, push_mse, util.ceilDiv(M_u * N_u, group_lin), &ctx, &.{ &buf_y, &buf_target_y, &buf_dy });
        // Backward — dx is not computed (no upstream).
        // dy_B = dy · B
        try recordLinBackward(&k_lin_dx, &push_dy_B, util.ceilDiv(M_u, group_lwg), util.ceilDiv(R_u, group_lwg), &ctx, &.{ &buf_dy, &buf_b, &buf_dy_B });
        // ∇A_unscaled = dy_Bᵀ · x ; ∇A = (α/r) · ∇A_unscaled
        try recordLinBackward(&k_lin_dw, &push_dA, util.ceilDiv(R_u, group_lwg), util.ceilDiv(K_u, group_lwg), &ctx, &.{ &buf_dy_B, &buf_x, &buf_dA_unscaled });
        try recordScalar1D(&k_scale, push_scale_dA, util.ceilDiv(R_u * K_u, group_lin), &ctx, &.{ &buf_dA_unscaled, &buf_dA });
        // ∇B_unscaled = dyᵀ · intermediate ; ∇B = (α/r) · ∇B_unscaled
        try recordLinBackward(&k_lin_dw, &push_dB, util.ceilDiv(N_u, group_lwg), util.ceilDiv(R_u, group_lwg), &ctx, &.{ &buf_dy, &buf_intermediate, &buf_dB_unscaled });
        try recordScalar1D(&k_scale, push_scale_dB, util.ceilDiv(N_u * R_u, group_lin), &ctx, &.{ &buf_dB_unscaled, &buf_dB });
        // Adam updates on A and B.
        const adam_a = runtime.AdamStepPush{ .n = R_u * K_u, .lr = lr, .beta1 = beta1, .beta2 = beta2, .eps = eps, .t = step };
        const adam_b = runtime.AdamStepPush{ .n = N_u * R_u, .lr = lr, .beta1 = beta1, .beta2 = beta2, .eps = eps, .t = step };
        try recordScalar1D(&k_adam, adam_a, util.ceilDiv(R_u * K_u, group_lin), &ctx, &.{ &buf_a, &buf_dA, &buf_m_a, &buf_v_a });
        try recordScalar1D(&k_adam, adam_b, util.ceilDiv(N_u * R_u, group_lin), &ctx, &.{ &buf_b, &buf_dB, &buf_m_b, &buf_v_b });

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

pub fn runGpuLoraPlusDemo(allocator: std.mem.Allocator) !void {
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
    var k_matmul = try pipeline.Kernel.init(&ctx, &shaders.matmul_nt_v2, 3, @sizeOf(aliases.MatmulPush));
    defer k_matmul.deinit();
    var k_lin_dx = try pipeline.Kernel.init(&ctx, &shaders.linear_backward_dx_batched, 3, @sizeOf(runtime.LinearBatchedPush));
    defer k_lin_dx.deinit();
    var k_lin_dw = try pipeline.Kernel.init(&ctx, &shaders.linear_backward_dw_batched, 3, @sizeOf(runtime.LinearBatchedPush));
    defer k_lin_dw.deinit();
    var k_scale = try pipeline.Kernel.init(&ctx, &shaders.scale, 2, @sizeOf(aliases.ScalePush));
    defer k_scale.deinit();
    var k_add = try pipeline.Kernel.init(&ctx, &shaders.add_in_place, 2, @sizeOf(runtime.AddInPlacePush));
    defer k_add.deinit();
    var k_mse_grad = try pipeline.Kernel.init(&ctx, &shaders.mse_loss_grad, 3, @sizeOf(aliases.MseLossGradPush));
    defer k_mse_grad.deinit();
    var k_adam = try pipeline.Kernel.init(&ctx, &shaders.adam_step, 4, @sizeOf(runtime.AdamStepPush));
    defer k_adam.deinit();

    const group_lwg: u32 = 16;
    const group_lin: u32 = 256;

    const recordMatmul = struct {
        fn rec(rec_kern: *pipeline.Kernel, push: *const aliases.MatmulPush, gx: u32, c_ctx: *const vk.Context, bufs: []const *const buffer.Buffer) !void {
            try rec_kern.bind(bufs);
            const Rec = struct {
                k: *const pipeline.Kernel,
                p: *const aliases.MatmulPush,
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

    const push_y_base = aliases.MatmulPush{ .m = M_u, .n = N_u, .k = K_u };
    const push_inter = aliases.MatmulPush{ .m = M_u, .n = R_u, .k = K_u };
    const push_ylora = aliases.MatmulPush{ .m = M_u, .n = N_u, .k = R_u };
    const push_scale_ylora = aliases.ScalePush{ .n = M_u * N_u, .scale = aor };
    const push_add_y = runtime.AddInPlacePush{ .n = M_u * N_u };
    const push_mse = aliases.MseLossGradPush{ .n = M_u * N_u };
    const push_dy_B = runtime.LinearBatchedPush{ .M = M_u, .N = N_u, .K = R_u };
    const push_dA = runtime.LinearBatchedPush{ .M = M_u, .N = R_u, .K = K_u };
    const push_scale_dA = aliases.ScalePush{ .n = R_u * K_u, .scale = aor };
    const push_dB = runtime.LinearBatchedPush{ .M = M_u, .N = N_u, .K = R_u };
    const push_scale_dB = aliases.ScalePush{ .n = N_u * R_u, .scale = aor };

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
            pyb: *const aliases.MatmulPush,
            pin: *const aliases.MatmulPush,
            pyl: *const aliases.MatmulPush,
            psy: *const aliases.ScalePush,
            pay: *const runtime.AddInPlacePush,
            pms: *const aliases.MseLossGradPush,
            pdyB: *const runtime.LinearBatchedPush,
            pdA: *const runtime.LinearBatchedPush,
            psda: *const aliases.ScalePush,
            pdB: *const runtime.LinearBatchedPush,
            psdb: *const aliases.ScalePush,
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
            try recS_fn(ksc, psy.*, util.ceilDiv(mu * nu, glin), ctx_ptr, &.{ buf_yl_, buf_yls_ });
            try recS_fn(kad, pay.*, util.ceilDiv(mu * nu, glin), ctx_ptr, &.{ buf_y_, buf_yls_ });
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
                try recS_fn(ksc, psy.*, util.ceilDiv(mu * nu, glin), ctx_ptr, &.{ buf_yl_, buf_yls_ });
                try recS_fn(kad, pay.*, util.ceilDiv(mu * nu, glin), ctx_ptr, &.{ buf_y_, buf_yls_ });
                // Loss grad.
                try recS_fn(kmse, pms.*, util.ceilDiv(mu * nu, glin), ctx_ptr, &.{ buf_y_, buf_target_, buf_dy_ });
                // Backward.
                try recL_fn(klx, pdyB, util.ceilDiv(mu, glwg), util.ceilDiv(ru, glwg), ctx_ptr, &.{ buf_dy_, buf_b_, buf_dyB_ });
                try recL_fn(klw, pdA, util.ceilDiv(ru, glwg), util.ceilDiv(ku, glwg), ctx_ptr, &.{ buf_dyB_, buf_x_, buf_dAu_ });
                try recS_fn(ksc, psda.*, util.ceilDiv(ru * ku, glin), ctx_ptr, &.{ buf_dAu_, buf_dA_ });
                try recL_fn(klw, pdB, util.ceilDiv(nu, glwg), util.ceilDiv(ru, glwg), ctx_ptr, &.{ buf_dy_, buf_inter_, buf_dBu_ });
                try recS_fn(ksc, psdb.*, util.ceilDiv(nu * ru, glin), ctx_ptr, &.{ buf_dBu_, buf_dB_ });
                // Adam — different lr for A vs B (the LoRA+ knob).
                const adam_a = runtime.AdamStepPush{ .n = ru * ku, .lr = lr_a, .beta1 = beta1_, .beta2 = beta2_, .eps = eps_, .t = step };
                const adam_b = runtime.AdamStepPush{ .n = nu * ru, .lr = lr_b, .beta1 = beta1_, .beta2 = beta2_, .eps = eps_, .t = step };
                try recS_fn(kadam, adam_a, util.ceilDiv(ru * ku, glin), ctx_ptr, &.{ buf_a_, buf_dA_, buf_m_a_, buf_v_a_ });
                try recS_fn(kadam, adam_b, util.ceilDiv(nu * ru, glin), ctx_ptr, &.{ buf_b_, buf_dB_, buf_m_b_, buf_v_b_ });

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

// ── A4-2: Runner-side LoRA-Q smoke ────────────────────────────────────
//
// Wire test for LoRA-Q integration into `train_transformer.Runner`.
// Earlier --lora-rec-smoke proved the dispatch chain at the per-kernel
// level; this one drives the *real* Runner.step end-to-end with
// `cfg.lora_q_enabled = true` and asserts three things:
//
//   1. Forward parity at init. Because A is N(0,σ) but B = 0, the
//      LoRA delta `(α/r)·B·A·x` is identically zero at step 0, so
//      forwardLogits with lora_q_enabled = true must be bit-equal to
//      the lora_q_enabled = false runner over the same weights.
//      This is the safety net: if wiring is wrong, this fails before
//      we even take a gradient step.
//
//   2. W_q is frozen across training. After N Adam steps the LoRA
//      runner's per-layer W_q must equal its initial value byte-for-byte.
//      If recordAdamAll's branch is wrong, this catches it.
//
//   3. Loss decreases. With B initialised to zero, ∇A is zero on
//      step 1 (∇A = (α/r)·dy_Bᵀ·x and dy_B = dy·B = 0) but ∇B is not
//      (∇B = (α/r)·dyᵀ·intermediate, intermediate = x·Aᵀ ≠ 0). Step 1
//      moves B; step 2 onward moves both. The standalone --lora-train-demo
//      already proves this asymmetry trains; here we just gate that
//      the loss drops vs. the initial loss across n_steps.
//
// Synthetic micro-shape (≪ Qwen3-0.6B): keeps init/step under a second
// each so this can land in the default smoke pass without slowing it down.

pub fn runGpuTransformerLoraQSmoke(allocator: std.mem.Allocator) !void {
    try runGpuTransformerLoraTargetsSmoke(allocator, train_transformer.LoraTarget.q, "Q-only");
}

pub fn runGpuTransformerLoraAllSmoke(allocator: std.mem.Allocator) !void {
    try runGpuTransformerLoraTargetsSmoke(allocator, train_transformer.LoraTarget.all, "all-7");
}

fn runGpuTransformerLoraTargetsSmoke(
    allocator: std.mem.Allocator,
    targets: u32,
    label: []const u8,
) !void {
    var ctx = try vk.Context.init(allocator);
    defer ctx.deinit();

    // ── Toy decoder shape. Just big enough for every projection's LoRA
    //    chain to have something to do (rank-4 adapter on each of the
    //    7 candidate W's). Same shape as the Q-only smoke so the
    //    runtime stays well under a second per case.
    const dim: u32 = 32;
    const n_heads: u32 = 4;
    const n_kv_heads: u32 = 2;
    const head_dim: u32 = 8; // n_heads * head_dim = dim
    const ff_dim: u32 = 64;
    const n_pos: u32 = 8;
    const n_layers: u32 = 2;
    const vocab: u32 = 64;
    const lora_rank: u32 = 4;
    const lora_alpha: f32 = 8.0;
    const n_steps: u32 = 30;

    // ── Random init weights. Deterministic seed → stable parity.
    var prng = std.Random.DefaultPrng.init(0xC10_5550);
    const rng = prng.random();

    const w_embed = try allocator.alloc(f32, vocab * dim);
    defer allocator.free(w_embed);
    const w_final_norm = try allocator.alloc(f32, dim);
    defer allocator.free(w_final_norm);
    const w_lm_head = try allocator.alloc(f32, vocab * dim);
    defer allocator.free(w_lm_head);
    for (w_embed) |*v| v.* = (rng.float(f32) - 0.5) * 0.2;
    for (w_final_norm) |*v| v.* = 1.0;
    for (w_lm_head) |*v| v.* = (rng.float(f32) - 0.5) * 0.2;

    const q_dim: u32 = n_heads * head_dim;
    const kv_dim: u32 = n_kv_heads * head_dim;
    const layers = try allocator.alloc(train_transformer.LayerWeights, n_layers);
    defer allocator.free(layers);

    // Backing storage for each layer's weights — kept alive for the
    // duration of the Runner.init call (Runner uploads into its own
    // device buffers and copies, so we can free after init returns).
    var layer_storage = std.ArrayList([]f32).init(allocator);
    defer {
        for (layer_storage.items) |s| allocator.free(s);
        layer_storage.deinit();
    }
    const allocLayerSlice = struct {
        fn f(al: std.mem.Allocator, store: *std.ArrayList([]f32), n: usize, r: std.Random, scale: f32, fill_with: ?f32) ![]f32 {
            const s = try al.alloc(f32, n);
            if (fill_with) |c| {
                for (s) |*v| v.* = c;
            } else {
                for (s) |*v| v.* = (r.float(f32) - 0.5) * scale;
            }
            try store.append(s);
            return s;
        }
    }.f;

    for (0..n_layers) |li| {
        layers[li] = .{
            // RMSNorm gains: identity (1.0) so the norm is well-conditioned.
            .w_n1 = try allocLayerSlice(allocator, &layer_storage, dim, rng, 0, 1.0),
            .w_q = try allocLayerSlice(allocator, &layer_storage, q_dim * dim, rng, 0.2, null),
            .w_k = try allocLayerSlice(allocator, &layer_storage, kv_dim * dim, rng, 0.2, null),
            .w_v = try allocLayerSlice(allocator, &layer_storage, kv_dim * dim, rng, 0.2, null),
            .w_o = try allocLayerSlice(allocator, &layer_storage, dim * q_dim, rng, 0.2, null),
            .w_n2 = try allocLayerSlice(allocator, &layer_storage, dim, rng, 0, 1.0),
            .w_gate = try allocLayerSlice(allocator, &layer_storage, ff_dim * dim, rng, 0.2, null),
            .w_up = try allocLayerSlice(allocator, &layer_storage, ff_dim * dim, rng, 0.2, null),
            .w_down = try allocLayerSlice(allocator, &layer_storage, dim * ff_dim, rng, 0.2, null),
        };
    }

    // ── Token + target ids.
    const token_ids = try allocator.alloc(u32, n_pos);
    defer allocator.free(token_ids);
    const target_ids = try allocator.alloc(u32, n_pos);
    defer allocator.free(target_ids);
    for (token_ids) |*tid| tid.* = rng.intRangeLessThan(u32, 0, vocab);
    for (target_ids) |*tid| tid.* = rng.intRangeLessThan(u32, 0, vocab);

    const cfg_base: train_transformer.Config = .{
        .dim = dim,
        .n_heads = n_heads,
        .n_kv_heads = n_kv_heads,
        .head_dim = head_dim,
        .ff_dim = ff_dim,
        .n_pos = n_pos,
        .n_layers = n_layers,
        .vocab_size = vocab,
        .lr = 1e-2,
    };
    var cfg_lora = cfg_base;
    cfg_lora.lora_targets = targets;
    cfg_lora.lora_rank = lora_rank;
    cfg_lora.lora_alpha = lora_alpha;

    const init_weights: train_transformer.InitWeights = .{
        .embed = w_embed,
        .final_norm = w_final_norm,
        .lm_head = w_lm_head,
        .layers = layers,
    };

    // ── Runner #1: LoRA on, default-init A,B for every enabled target.
    var runner_lora = try train_transformer.Runner.init(allocator, &ctx, cfg_lora, init_weights);
    defer runner_lora.deinit();

    // ── Runner #2: LoRA off, same weights — control for parity.
    var runner_plain = try train_transformer.Runner.init(allocator, &ctx, cfg_base, init_weights);
    defer runner_plain.deinit();

    // ── Step-0 forward parity. B = 0 for every enabled projection ⇒
    //    the LoRA delta is zero on every chain ⇒ logits are bit-equal
    //    to the no-LoRA Runner.
    const logits_lora = try allocator.alloc(f32, n_pos * vocab);
    defer allocator.free(logits_lora);
    const logits_plain = try allocator.alloc(f32, n_pos * vocab);
    defer allocator.free(logits_plain);
    try runner_lora.forwardLogits(token_ids, logits_lora);
    try runner_plain.forwardLogits(token_ids, logits_plain);

    var max_diff: f32 = 0;
    for (logits_lora, logits_plain) |a, b| {
        const d = @abs(a - b);
        if (d > max_diff) max_diff = d;
    }
    if (max_diff > 1e-5) {
        std.debug.print(
            "FAIL LoRA ({s}) step-0 forward parity: max|Δ| = {e} (expected ≤ 1e-5)\n",
            .{ label, max_diff },
        );
        return error.LoraForwardParityFailed;
    }
    std.debug.print("  step-0 fwd parity   max|Δ| = {e:.2}\n", .{max_diff});

    // ── Snapshot every LoRA-target's W on layer 0 — must remain bit-
    //    equal across all training steps.
    const Probe = struct {
        name: []const u8,
        bit: u32,
        bufs: []buffer.Buffer,
        numel: usize,
    };
    const probes = [_]Probe{
        .{ .name = "W_q", .bit = train_transformer.LoraTarget.q, .bufs = runner_lora.buf_w_q, .numel = q_dim * dim },
        .{ .name = "W_k", .bit = train_transformer.LoraTarget.k, .bufs = runner_lora.buf_w_k, .numel = kv_dim * dim },
        .{ .name = "W_v", .bit = train_transformer.LoraTarget.v, .bufs = runner_lora.buf_w_v, .numel = kv_dim * dim },
        .{ .name = "W_o", .bit = train_transformer.LoraTarget.o, .bufs = runner_lora.buf_w_o, .numel = dim * q_dim },
        .{ .name = "W_gate", .bit = train_transformer.LoraTarget.gate, .bufs = runner_lora.buf_w_gate, .numel = ff_dim * dim },
        .{ .name = "W_up", .bit = train_transformer.LoraTarget.up, .bufs = runner_lora.buf_w_up, .numel = ff_dim * dim },
        .{ .name = "W_down", .bit = train_transformer.LoraTarget.down, .bufs = runner_lora.buf_w_down, .numel = dim * ff_dim },
    };
    var snapshots: [probes.len][]f32 = undefined;
    for (probes, 0..) |p, i| {
        if ((targets & p.bit) == 0) continue;
        snapshots[i] = try allocator.alloc(f32, p.numel);
        try p.bufs[0].readBack(&ctx, f32, snapshots[i]);
    }
    defer for (probes, 0..) |p, i| {
        if ((targets & p.bit) == 0) continue;
        allocator.free(snapshots[i]);
    };

    // ── Initial loss before any step.
    const initial_loss = computeCeLoss(logits_lora, target_ids, n_pos, vocab);

    // ── Train.
    var step_t: u32 = 0;
    while (step_t < n_steps) : (step_t += 1) {
        try runner_lora.step(token_ids, target_ids);
    }

    // ── Verify every snapshotted W is unchanged byte-for-byte.
    var after_buf: [4096]f32 = undefined;
    for (probes, 0..) |p, i| {
        if ((targets & p.bit) == 0) continue;
        if (p.numel > after_buf.len) {
            std.debug.print("INTERNAL: probe {s} numel={d} > scratch {d} — bump after_buf\n", .{ p.name, p.numel, after_buf.len });
            return error.ProbeScratchTooSmall;
        }
        try p.bufs[0].readBack(&ctx, f32, after_buf[0..p.numel]);
        var w_max_diff: f32 = 0;
        for (snapshots[i], after_buf[0..p.numel]) |a, b| {
            const d = @abs(a - b);
            if (d > w_max_diff) w_max_diff = d;
        }
        if (w_max_diff != 0.0) {
            std.debug.print(
                "FAIL LoRA ({s}) {s} frozen: max|Δ| = {e} (expected exactly 0)\n",
                .{ label, p.name, w_max_diff },
            );
            return error.LoraWFrozenViolated;
        }
    }
    std.debug.print("  W frozen ({s})    max|Δ| = 0 (over {d} steps)\n", .{ label, n_steps });

    // ── Verify loss decreased.
    try runner_lora.forwardLogits(token_ids, logits_lora);
    const final_loss = computeCeLoss(logits_lora, target_ids, n_pos, vocab);
    if (!(final_loss < initial_loss)) {
        std.debug.print(
            "FAIL LoRA ({s}) loss: initial={d:.6} final={d:.6} (expected final < initial)\n",
            .{ label, initial_loss, final_loss },
        );
        return error.LoraLossDidNotDecrease;
    }

    std.debug.print(
        "PASS GPU train_transformer LoRA via Runner — targets={s} ({d}/7) rank={d} α={d:.1} α/r={d:.2}, CE {d:.4} → {d:.4} ({d:.2}× drop) over {d} steps\n",
        .{ label, @popCount(targets), lora_rank, lora_alpha, lora_alpha / @as(f32, @floatFromInt(lora_rank)), initial_loss, final_loss, initial_loss / final_loss, n_steps },
    );
}

fn computeCeLoss(logits: []const f32, target_ids: []const u32, n_pos: u32, vocab: u32) f32 {
    var total: f32 = 0;
    for (0..n_pos) |row| {
        const off = row * vocab;
        var m: f32 = -std.math.inf(f32);
        for (0..vocab) |v| {
            if (logits[off + v] > m) m = logits[off + v];
        }
        var lse_sum: f32 = 0;
        for (0..vocab) |v| lse_sum += @exp(logits[off + v] - m);
        const lse = m + @log(lse_sum);
        total += lse - logits[off + target_ids[row]];
    }
    return total / @as(f32, @floatFromInt(n_pos));
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

pub fn runCceBench(allocator: std.mem.Allocator) !void {
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

// ── Attention bench: time the 3-dispatch attention chain ──────────────
//
// Sweeps `n_kv` across decode (n_q=1, no mask) and prefill-causal
// (n_q == n_kv) shapes at Qwen3-0.6B's per-layer attention dims
// (n_heads=16, n_kv_heads=8, head_dim=128). Reports per-kernel wall-
// clock ms (submitOneShot + vkQueueWaitIdle, so submission overhead
// is folded in — same caveat as runCceBench), plus the per-layer
// scores-buffer footprint and a ×28-layer projection (Qwen3-0.6B has
// 28 transformer blocks, each running this same 3-dispatch chain).
//
// The point of this bench is F1 of the FlashAttention arc: size
// what we'd be replacing. The 3 kernels are
//
//     attn_scores       : Q · Kᵀ        → scores [n_heads × n_kv]   fp32
//     softmax           : per-head row-softmax over scores
//     attn_output       : softmax · V   → head_out [n_heads × head_dim]
//
// (decode variants; prefill uses *_train kernels with an extra n_q
// dimension and an optional causal-mask flag). Whatever FlashAttention
// kernel we ship has to beat the totals printed here at the same
// shapes.

pub fn runAttnBench(allocator: std.mem.Allocator) !void {
    var ctx = try vk.Context.init(allocator);
    defer ctx.deinit();

    // Qwen3-0.6B per-layer attention shape.
    const n_heads: u32 = 16;
    const n_kv_heads: u32 = 8;
    const head_dim: u32 = 128;
    const n_layers: u32 = 28;
    const heads_per_kv: u32 = n_heads / n_kv_heads;
    const inv_sqrt: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(head_dim)));

    const decode_kvs = [_]u32{ 128, 512, 2048, 8192, 32768 };
    const prefill_kvs = [_]u32{ 16, 128, 512, 2048 };
    const decode_max_kv: u32 = 32768;
    const prefill_max_q: u32 = 2048;
    const warmup_iters: u32 = 3;
    const bench_iters: u32 = 5;

    std.debug.print(
        "Attention bench on {s}\n" ++
            "  Qwen3-0.6B per-layer: n_heads={d} n_kv_heads={d} head_dim={d} ({d} layers)\n" ++
            "  warmup {d} / measure {d}    (timings include submitOneShot + waitIdle)\n",
        .{ ctx.deviceName(), n_heads, n_kv_heads, head_dim, n_layers, warmup_iters, bench_iters },
    );

    // ── Allocate buffers sized to the larger of {decode, prefill} per
    // tensor — the cross-product would be 16 GB at decode_max_kv ×
    // prefill_max_q × n_heads × 4B, but each phase only needs its own
    // shape, so we cap on the per-tensor max.
    const q_max_elems: usize = @max(
        @as(usize, n_heads) * head_dim, // decode (n_q=1)
        @as(usize, prefill_max_q) * n_heads * head_dim, // prefill
    );
    const kv_max_elems: usize = @as(usize, decode_max_kv) * n_kv_heads * head_dim;
    const scores_max_elems: usize = @max(
        @as(usize, n_heads) * decode_max_kv, // decode
        @as(usize, prefill_max_q) * n_heads * prefill_max_q, // prefill (n_q == n_kv)
    );
    const out_max_elems: usize = @as(usize, prefill_max_q) * n_heads * head_dim;

    var prng = std.Random.DefaultPrng.init(0xA7_7E_07);
    const rng = prng.random();

    const q_host = try allocator.alloc(f32, q_max_elems);
    defer allocator.free(q_host);
    const kv_host = try allocator.alloc(f32, kv_max_elems);
    defer allocator.free(kv_host);
    for (q_host) |*x| x.* = (rng.float(f32) - 0.5) * 0.1;
    for (kv_host) |*x| x.* = (rng.float(f32) - 0.5) * 0.1;

    var buf_q = try buffer.Buffer.initStatic(&ctx, f32, q_host);
    defer buf_q.deinit(ctx.device);
    var buf_k = try buffer.Buffer.initStatic(&ctx, f32, kv_host);
    defer buf_k.deinit(ctx.device);
    var buf_v = try buffer.Buffer.initStatic(&ctx, f32, kv_host);
    defer buf_v.deinit(ctx.device);
    var buf_scores = try buffer.Buffer.initDeviceOnly(&ctx, scores_max_elems * @sizeOf(f32));
    defer buf_scores.deinit(ctx.device);
    var buf_softmax_out = try buffer.Buffer.initDeviceOnly(&ctx, scores_max_elems * @sizeOf(f32));
    defer buf_softmax_out.deinit(ctx.device);
    var buf_out = try buffer.Buffer.initDeviceOnly(&ctx, out_max_elems * @sizeOf(f32));
    defer buf_out.deinit(ctx.device);

    // ── Pipelines (built once, rebound per case).
    var k_scores = try pipeline.Kernel.init(&ctx, &shaders.attn_scores, 3, @sizeOf(runtime.AttnScoresPush));
    defer k_scores.deinit();
    var k_softmax = try pipeline.Kernel.init(&ctx, &shaders.softmax, 2, @sizeOf(runtime.SoftmaxPush));
    defer k_softmax.deinit();
    var k_attn_out = try pipeline.Kernel.init(&ctx, &shaders.attn_output, 3, @sizeOf(runtime.AttnOutputPush));
    defer k_attn_out.deinit();
    var k_scores_t = try pipeline.Kernel.init(&ctx, &shaders.attn_scores_train, 3, @sizeOf(runtime.AttnScoresTrainPush));
    defer k_scores_t.deinit();
    var k_attn_out_t = try pipeline.Kernel.init(&ctx, &shaders.attn_output_train, 3, @sizeOf(runtime.AttnOutputTrainPush));
    defer k_attn_out_t.deinit();

    try k_scores.bind(&.{ &buf_q, &buf_k, &buf_scores });
    try k_softmax.bind(&.{ &buf_scores, &buf_softmax_out });
    try k_attn_out.bind(&.{ &buf_softmax_out, &buf_v, &buf_out });
    try k_scores_t.bind(&.{ &buf_q, &buf_k, &buf_scores });
    try k_attn_out_t.bind(&.{ &buf_softmax_out, &buf_v, &buf_out });

    const ns_per_ms: f64 = 1.0e6;

    const Run = struct {
        fn one(c_ctx: *const vk.Context, kern: *const pipeline.Kernel, push: anytype, gx: u32, iters: u32) !f64 {
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
    };

    // ── Decode (n_q=1) ────────────────────────────────────────────────
    std.debug.print(
        "\n── Decode (n_q=1, no causal mask) ─────────────────────────────────────\n" ++
            "  {s:>6}  {s:>10}  {s:>14}  {s:>10}  {s:>14}  {s:>10}  {s:>13}\n",
        .{ "n_kv", "scoresMB", "attn_scores", "softmax", "attn_output", "total/L", "×28 layers" },
    );

    for (decode_kvs) |n_kv| {
        const scores_bytes: usize = @as(usize, n_heads) * n_kv * @sizeOf(f32);
        const scores_mb: f64 = @as(f64, @floatFromInt(scores_bytes)) / (1024.0 * 1024.0);

        const push_scores = runtime.AttnScoresPush{
            .n_heads = n_heads,
            .heads_per_kv = heads_per_kv,
            .head_dim = head_dim,
            .n_pos = n_kv,
            .kv_stride = n_kv_heads * head_dim,
            .scores_stride = n_kv,
            .inv_sqrt_dim = inv_sqrt,
        };
        const push_softmax = runtime.SoftmaxPush{ .dim = n_kv, .stride = n_kv };
        const push_attn_out = runtime.AttnOutputPush{
            .n_heads = n_heads,
            .heads_per_kv = heads_per_kv,
            .head_dim = head_dim,
            .n_pos = n_kv,
            .kv_stride = n_kv_heads * head_dim,
            .scores_stride = n_kv,
        };
        const gx_scores: u32 = n_heads * n_kv;
        const gx_softmax: u32 = n_heads;
        const gx_out: u32 = n_heads * head_dim;

        _ = try Run.one(&ctx, &k_scores, push_scores, gx_scores, warmup_iters);
        _ = try Run.one(&ctx, &k_softmax, push_softmax, gx_softmax, warmup_iters);
        _ = try Run.one(&ctx, &k_attn_out, push_attn_out, gx_out, warmup_iters);

        const t_s = try Run.one(&ctx, &k_scores, push_scores, gx_scores, bench_iters);
        const t_sm = try Run.one(&ctx, &k_softmax, push_softmax, gx_softmax, bench_iters);
        const t_o = try Run.one(&ctx, &k_attn_out, push_attn_out, gx_out, bench_iters);
        const t_total = (t_s + t_sm + t_o) / ns_per_ms;

        std.debug.print(
            "  {d:>6}  {d:>10.3}  {d:>11.3} ms  {d:>7.3} ms  {d:>11.3} ms  {d:>7.3} ms  {d:>10.2} ms\n",
            .{ n_kv, scores_mb, t_s / ns_per_ms, t_sm / ns_per_ms, t_o / ns_per_ms, t_total, t_total * @as(f64, @floatFromInt(n_layers)) },
        );
    }

    // ── Prefill causal (n_q == n_kv) ──────────────────────────────────
    std.debug.print(
        "\n── Prefill causal (n_q == n_kv) ───────────────────────────────────────\n" ++
            "  {s:>6}  {s:>10}  {s:>14}  {s:>10}  {s:>14}  {s:>10}  {s:>13}\n",
        .{ "n_q", "scoresMB", "scores_train", "softmax", "out_train", "total/L", "×28 layers" },
    );

    for (prefill_kvs) |n_q| {
        const n_kv = n_q;
        const scores_bytes: usize = @as(usize, n_q) * n_heads * n_kv * @sizeOf(f32);
        const scores_mb: f64 = @as(f64, @floatFromInt(scores_bytes)) / (1024.0 * 1024.0);

        const push_scores_t = runtime.AttnScoresTrainPush{
            .n_q = n_q,
            .n_heads = n_heads,
            .heads_per_kv = heads_per_kv,
            .head_dim = head_dim,
            .n_kv = n_kv,
            .kv_stride = n_kv_heads * head_dim,
            .scores_stride = n_kv,
            .causal = 1,
            .inv_sqrt_dim = inv_sqrt,
        };
        const push_softmax = runtime.SoftmaxPush{ .dim = n_kv, .stride = n_kv };
        const push_attn_out_t = runtime.AttnOutputTrainPush{
            .n_q = n_q,
            .n_heads = n_heads,
            .heads_per_kv = heads_per_kv,
            .head_dim = head_dim,
            .n_kv = n_kv,
            .kv_stride = n_kv_heads * head_dim,
            .attn_stride = n_kv,
        };
        const gx_scores_t: u32 = n_q * n_heads * n_kv;
        const gx_softmax: u32 = n_q * n_heads;
        const gx_out_t: u32 = n_q * n_heads * head_dim;

        _ = try Run.one(&ctx, &k_scores_t, push_scores_t, gx_scores_t, warmup_iters);
        _ = try Run.one(&ctx, &k_softmax, push_softmax, gx_softmax, warmup_iters);
        _ = try Run.one(&ctx, &k_attn_out_t, push_attn_out_t, gx_out_t, warmup_iters);

        const t_s = try Run.one(&ctx, &k_scores_t, push_scores_t, gx_scores_t, bench_iters);
        const t_sm = try Run.one(&ctx, &k_softmax, push_softmax, gx_softmax, bench_iters);
        const t_o = try Run.one(&ctx, &k_attn_out_t, push_attn_out_t, gx_out_t, bench_iters);
        const t_total = (t_s + t_sm + t_o) / ns_per_ms;

        std.debug.print(
            "  {d:>6}  {d:>10.3}  {d:>11.3} ms  {d:>7.3} ms  {d:>11.3} ms  {d:>7.3} ms  {d:>10.2} ms\n",
            .{ n_q, scores_mb, t_s / ns_per_ms, t_sm / ns_per_ms, t_o / ns_per_ms, t_total, t_total * @as(f64, @floatFromInt(n_layers)) },
        );
    }

    std.debug.print(
        "\n  scoresMB column = per-layer fp32 scores buffer (rows × n_kv × 4B / 2^20).\n" ++
            "  ×28 layers = total/L extrapolated to Qwen3-0.6B's full attention stack.\n" ++
            "  These are the totals FlashAttention has to beat at the same shapes.\n",
        .{},
    );
}

// ── GPU CCE backward d_h smoke: vs CPU oracle ─────────────────────────
//
// Drives `cce_backward_dh.comp` against the d_h output of
// `cpu_cce.cceBackward`. The oracle path uses the same chunked
// recompute-and-accumulate algorithm but in scalar f64; the GPU version
// uses cooperative subgroup reductions. Tolerance is 1e-5 global rel-err
// (`max|diff| / max|ref|`), same as cce.zig's in-file parity tests.

pub fn runGpuCceBackwardDhSmoke(allocator: std.mem.Allocator) !void {
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

pub fn runGpuCceBackwardDwSmoke(allocator: std.mem.Allocator) !void {
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

pub fn runGpuMatmulV2Smoke(allocator: std.mem.Allocator) !void {
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

    var kern = try pipeline.Kernel.init(&ctx, &shaders.matmul_nt_v2, 3, @sizeOf(aliases.MatmulPush));
    defer kern.deinit();
    try kern.bind(&.{ &buf_a, &buf_b, &buf_c });

    // v2 dispatches one WG per output cell — total = M*N WGs.
    const groups: u32 = m * n;
    const push = aliases.MatmulPush{ .m = m, .n = n, .k = k };
    try buffer.submitOneShot(&ctx, struct {
        kern: *const pipeline.Kernel,
        push: *const aliases.MatmulPush,
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

pub fn runEmbeddedAttachSmoke(allocator: std.mem.Allocator) !void {
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

    var kern = try pipeline.Kernel.init(&attached, &shaders.matmul_nt, 3, @sizeOf(aliases.MatmulPush));
    defer kern.deinit();
    try kern.bind(&.{ &buf_a, &buf_b, &buf_c });

    const local_xy: u32 = 16;
    const groups_x: u32 = (m + local_xy - 1) / local_xy;
    const groups_y: u32 = (n + local_xy - 1) / local_xy;
    const push = aliases.MatmulPush{ .m = m, .n = n, .k = k };

    try buffer.submitOneShot(&attached, struct {
        kern: *const pipeline.Kernel,
        push: *const aliases.MatmulPush,
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

pub fn runEmbeddedRecorderSmoke(allocator: std.mem.Allocator) !void {
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

    var kern = try pipeline.Kernel.init(&ctx, &shaders.matmul_nt_v2, 3, @sizeOf(aliases.MatmulPush));
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
    const push = aliases.MatmulPush{ .m = m, .n = n, .k = k };
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

pub fn runGpuMatmulQ4_0Smoke(allocator: std.mem.Allocator) !void {
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

    var kern = try pipeline.Kernel.init(&ctx, &shaders.matmul_nt_v2_q4_0, 3, @sizeOf(aliases.MatmulPush));
    defer kern.deinit();
    try kern.bind(&.{ &buf_a, &buf_b, &buf_c });

    const groups: u32 = m * n;
    const push = aliases.MatmulPush{ .m = m, .n = n, .k = k };
    try buffer.submitOneShot(&ctx, struct {
        kern: *const pipeline.Kernel,
        push: *const aliases.MatmulPush,
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

pub fn runGpuMatmulQ4_KSmoke(allocator: std.mem.Allocator) !void {
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

    var kern = try pipeline.Kernel.init(&ctx, &shaders.matmul_nt_v2_q4_k, 3, @sizeOf(aliases.MatmulPush));
    defer kern.deinit();
    try kern.bind(&.{ &buf_a, &buf_b, &buf_c });

    const groups: u32 = m * n;
    const push = aliases.MatmulPush{ .m = m, .n = n, .k = k };
    try buffer.submitOneShot(&ctx, struct {
        kern: *const pipeline.Kernel,
        push: *const aliases.MatmulPush,
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

pub fn runGpuRmsnormSmoke(allocator: std.mem.Allocator) !void {
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

        var kern = try pipeline.Kernel.init(&ctx, &shaders.rmsnorm, 3, @sizeOf(aliases.RmsnormPush));
        defer kern.deinit();
        try kern.bind(&.{ &buf_a, &buf_w, &buf_c });

        const push = aliases.RmsnormPush{
            .dim = @intCast(dim),
            .eps = 1e-6,
            .gemma_quirk = if (gemma_quirk) 1 else 0,
        };
        try buffer.submitOneShot(&ctx, struct {
            kern: *const pipeline.Kernel,
            push: *const aliases.RmsnormPush,
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

pub fn runGpuRmsnormBackwardSmoke(allocator: std.mem.Allocator) !void {
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

        var kern = try pipeline.Kernel.init(&ctx, &shaders.rmsnorm_backward, 5, @sizeOf(aliases.RmsnormPush));
        defer kern.deinit();
        try kern.bind(&.{ &buf_dy, &buf_x, &buf_w, &buf_dx, &buf_dw_partial });

        const push = aliases.RmsnormPush{
            .dim = @intCast(dim),
            .eps = eps,
            .gemma_quirk = if (gemma_quirk) 1 else 0,
        };
        try buffer.submitOneShot(&ctx, struct {
            kern: *const pipeline.Kernel,
            push: *const aliases.RmsnormPush,
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

pub fn runLayerNormBackwardCpuSmoke(allocator: std.mem.Allocator) !void {
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

pub fn runGpuLayerNormSmoke(allocator: std.mem.Allocator) !void {
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

pub fn runGpuLayerNormBackwardSmoke(allocator: std.mem.Allocator) !void {
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

pub fn runEmbeddingBackwardSmoke(allocator: std.mem.Allocator) !void {
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

pub fn runSoftmaxBackwardSmoke(allocator: std.mem.Allocator) !void {
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

    var kern = try pipeline.Kernel.init(&ctx, &shaders.softmax_backward, 3, @sizeOf(aliases.SoftmaxPush));
    defer kern.deinit();
    try kern.bind(&.{ &buf_dy, &buf_y, &buf_dx });

    const push = aliases.SoftmaxPush{ .dim = @intCast(dim), .stride = @intCast(dim) };
    try buffer.submitOneShot(&ctx, struct {
        kern: *const pipeline.Kernel,
        push: *const aliases.SoftmaxPush,
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

pub fn runAttentionBackwardCpuSmoke(allocator: std.mem.Allocator) !void {
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

pub fn runGpuAttentionBackwardSmoke(allocator: std.mem.Allocator) !void {
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

pub fn runRopeBackwardSmoke(allocator: std.mem.Allocator) !void {
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

    var kern = try pipeline.Kernel.init(&ctx, &shaders.rope_backward, 2, @sizeOf(aliases.RopePartialPush));
    defer kern.deinit();
    try kern.bind(&.{ &buf_dout, &buf_din });

    const push = aliases.RopePartialPush{
        .n_heads = @intCast(n_heads),
        .head_dim = @intCast(head_dim),
        .rotary_dim = @intCast(rotary_dim),
        .pos = @intCast(pos),
        .theta_base = theta_base,
    };
    try buffer.submitOneShot(&ctx, struct {
        kern: *const pipeline.Kernel,
        push: *const aliases.RopePartialPush,
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
