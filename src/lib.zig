//! Public Zig-module surface for embedding valkyr's GPU compute
//! primitives in a host engine (e.g. Matryoshka). Build-side this is
//! the `valkyr_gpu` module exposed by `build.zig`; consumers
//! `@import("valkyr_gpu")` and reach the four compute primitives plus
//! the SPIR-V shader module.
//!
//! Deliberately narrow. Model loading, safetensors parsing, the chat
//! REPL, quantization kernels — none of that is here. The host
//! integration story is currently the cooperative-compute path
//! (Context.attach + Recorder.attachCmd + a kernel + buffers); the
//! larger surfaces would need their own design pass before being
//! exposed as a library, and pulling them in now would drag config /
//! safetensors / model.zig / cpu/* into every consumer's build for
//! no upside.
//!
//! When more of valkyr's surface is wanted from a host (loading a
//! GGUF/safetensors model from inside a Game, for instance), the
//! addition is one `pub const x = @import("...")` line here plus
//! whatever transitive imports follow.

pub const vk = @import("gpu/vk.zig");
pub const buffer = @import("gpu/buffer.zig");
pub const pipeline = @import("gpu/pipeline.zig");
pub const recorder = @import("gpu/recorder.zig");

/// Compiled SPIR-V blobs. Anonymous module wired in by `build.zig`;
/// each field is an `align(4) []const u8`-shaped @embedFile of one
/// shader (e.g. `shaders.matmul_nt`, `shaders.matmul_nt_v2`,
/// `shaders.rmsnorm`, ...). See valkyr's build.zig for the full list.
pub const shaders = @import("shaders");
