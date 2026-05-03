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

// ── Static-model surface (chunk 7a) ─────────────────────────────
// Hosts that want to load and run a real language model from inside
// their own Vulkan context need access to the safetensors parsing,
// config types, tokenizer, and the GpuModel upload pipeline. None of
// these are tied to a specific runtime state machine — they're the
// inert "model on disk → model on GPU" pieces. The Session type
// (chunk 7c) layers a default state machine on top of these; hosts
// that want bespoke orchestration use them directly.
pub const config = @import("config.zig");
pub const dtype = @import("dtype.zig");
pub const safetensors = @import("safetensors.zig");
pub const model = @import("model.zig");
pub const tokenizer = @import("tokenizer.zig");
pub const gpu_model = @import("gpu/model.zig");
pub const hf_cache = @import("hf_cache.zig");
pub const loader = @import("loader.zig");

/// Per-family chat-template machinery. Resolves family-specific
/// special-token markers and composes either a single user turn
/// (legacy CLI path) or a full `[{role, content}]` history
/// (server / embed `Session.appendMessages`). Lifted out of main.zig
/// so embed callers and the future `--serve` mode share one composer.
pub const chat_template = @import("chat_template.zig");

// ── Runtime primitives (chunk 7b) ───────────────────────────────
// Per-step forward recording + sampling. Hosts that want a default
// state machine wait for the Session API in chunk 7c, which is built
// on top of these primitives. Hosts that want bespoke orchestration
// reach `runtime.recordOneLayer`, `recordEmbedding`,
// `recordSampleStep`, and `sampleArgmax` directly.
pub const gpu_scratch = @import("gpu/scratch.zig");
pub const runtime = @import("runtime.zig");

/// Hybrid (Qwen3.5-family) forward primitives. Mirrors `runtime` for
/// dense models. Owns its own `ChatKernels`, `Scratch`, `State`,
/// `ForwardPushes`, `recordOneLayer`, `recordForwardStep` because the
/// hybrid path runs a different kernel set + per-layer SSM state +
/// branched dispatch (linear vs full attention). Chunk D will teach
/// `Session` to dispatch into here when `cfg.family.isHybrid()`.
pub const runtime_hybrid = @import("runtime_hybrid.zig");

// ── Session API (chunk 7c) ──────────────────────────────────────
// Frame-budgeted cooperative-inference state machine. Hosts that just
// want text-out call `Session.init` + `appendPrompt` + `tickFrame`
// once per frame. The state machine handles per-layer chunking,
// deferred sampling, and KV/scratch lifecycle. Hosts that want
// bespoke orchestration use the `runtime` primitives directly.
pub const session = @import("session.zig");

// ── Inference runner (Sketch #3) ────────────────────────────────
// SPSC queue + ping-pong text arena + InferenceRunner that drives
// Session from a producer command stream. Powers the upcoming
// `valkyr --serve` HTTP path AND embed callers that want a
// queue-based decoupling from the render thread.
pub const inference = struct {
    pub const queue = @import("inference/queue.zig");
    pub const arena = @import("inference/arena.zig");
    pub const proto = @import("inference/proto.zig");
    pub const runner = @import("inference/runner.zig");
};

// ── HTTP server (Sketch #4) ─────────────────────────────────────
// OpenAI-compatible `/v1/chat/completions` + `/v1/models`. Thin
// adapter over inference.runner. valkyr --serve is the canonical
// consumer; embed hosts can spin up their own Server if they want
// to expose a model on HTTP alongside their main loop.
pub const server = struct {
    pub const http = @import("server/http.zig");
    pub const json = @import("server/json.zig");
    pub const Server = @import("server/server.zig").Server;
    pub const ServerConfig = @import("server/server.zig").ServerConfig;
};
