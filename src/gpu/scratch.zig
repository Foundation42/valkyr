//! Per-forward-pass scratch buffers on the GPU.
//!
//! Activations during inference: residual stream, x_norm, Q/K/V,
//! head_out, attn_out, mid_norm, gate/up/fused, ffn_out. All
//! device_only (zero-filled at allocation, written exclusively by the
//! GPU during dispatches, optionally read back at the end of the
//! forward pass for parity tests).
//!
//! These get reused across layers — the residual stream lives the
//! whole forward pass, the rest are scratch we overwrite each layer.
//! Allocated once at construction; kept alive for as many forward
//! calls as the caller wants.

const std = @import("std");
const vk = @import("vk.zig");
const buffer = @import("buffer.zig");
const config_mod = @import("../config.zig");

pub const GpuScratch = struct {
    /// Residual stream — flows through every layer. Written by embed
    /// lookup, then read+written by each layer's two residual adds.
    stream: buffer.Buffer,

    // Per-layer reused scratch.
    x_norm: buffer.Buffer,
    q: buffer.Buffer,
    k: buffer.Buffer,
    v: buffer.Buffer,
    q_rot: buffer.Buffer,
    k_rot: buffer.Buffer,
    head_out: buffer.Buffer,
    attn_out: buffer.Buffer,
    mid_norm: buffer.Buffer,
    gate: buffer.Buffer,
    up: buffer.Buffer,
    fused: buffer.Buffer,
    ffn_out: buffer.Buffer,

    pub fn init(ctx: *const vk.Context, cfg: config_mod.Config) !GpuScratch {
        const hidden = cfg.hidden_size;
        const inter = cfg.intermediate_size;
        const q_dim = cfg.num_attention_heads * cfg.head_dim;
        const kv_dim = cfg.num_key_value_heads * cfg.head_dim;

        const f = @sizeOf(f32);
        return .{
            .stream    = try buffer.Buffer.initDeviceOnly(ctx, hidden * f),
            .x_norm    = try buffer.Buffer.initDeviceOnly(ctx, hidden * f),
            .q         = try buffer.Buffer.initDeviceOnly(ctx, q_dim * f),
            .k         = try buffer.Buffer.initDeviceOnly(ctx, kv_dim * f),
            .v         = try buffer.Buffer.initDeviceOnly(ctx, kv_dim * f),
            .q_rot     = try buffer.Buffer.initDeviceOnly(ctx, q_dim * f),
            .k_rot     = try buffer.Buffer.initDeviceOnly(ctx, kv_dim * f),
            .head_out  = try buffer.Buffer.initDeviceOnly(ctx, q_dim * f),
            .attn_out  = try buffer.Buffer.initDeviceOnly(ctx, hidden * f),
            .mid_norm  = try buffer.Buffer.initDeviceOnly(ctx, hidden * f),
            .gate      = try buffer.Buffer.initDeviceOnly(ctx, inter * f),
            .up        = try buffer.Buffer.initDeviceOnly(ctx, inter * f),
            .fused     = try buffer.Buffer.initDeviceOnly(ctx, inter * f),
            .ffn_out   = try buffer.Buffer.initDeviceOnly(ctx, hidden * f),
        };
    }

    pub fn deinit(self: *GpuScratch, device: vk.c.VkDevice) void {
        self.stream.deinit(device);
        self.x_norm.deinit(device);
        self.q.deinit(device);
        self.k.deinit(device);
        self.v.deinit(device);
        self.q_rot.deinit(device);
        self.k_rot.deinit(device);
        self.head_out.deinit(device);
        self.attn_out.deinit(device);
        self.mid_norm.deinit(device);
        self.gate.deinit(device);
        self.up.deinit(device);
        self.fused.deinit(device);
        self.ffn_out.deinit(device);
    }
};
