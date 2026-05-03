# Embedding valkyr in a Vulkan host

A guide for integrating valkyr into a real-time Vulkan application —
typically a game engine, AR/VR runtime, or any system that already has
a graphics frame loop and wants language-model inference to live
inside it without spawning a parallel device, queue, or process.

This isn't the OpenAI-style server contract ("give me a prompt, I
return tokens"). It's the **cooperative-compute** contract: valkyr
runs inside your render frame, shares your `VkDevice` and command
buffer, and produces tensor outputs your shaders can consume directly.
At 60 fps the model can think across multiple frames cooperatively —
silky smooth — while your renderer keeps drawing.

## Mental model

```
Frame N:
  engine.beginFrame();
  engine.recordRenderPass(cmd);                    // 8 ms
  valkyr.tickFrame(rec);                           // 4 ms — N layers of forward
  engine.recordPostProcess(cmd);                   // 2 ms
  engine.submit(cmd);                              // single submit on one queue
```

Per frame, valkyr records as many transformer layers as fit your
budget into the host's command buffer, then yields. The intermediate
state (KV cache, residual stream, scratch) lives in persistent
SSBOs across frames. After enough frames have elapsed, a token
emerges; the model's last-layer attention or any other tensor tap
can be sampled directly by your render shaders the same frame, with
no readback round trip.

This is the embedded contract that distinguishes valkyr from
llama.cpp / vLLM / ZINC. Those assume "give me a prompt, take my
queue, return tokens." valkyr supports that too (its CLI runs that
way) — but it *also* supports the embedded shape.

## Two API tiers

valkyr exposes a public Zig module called `valkyr_gpu`. Within it,
hosts can integrate at two levels:

1. **Cooperative-compute primitives** (`vk.Context`, `buffer`,
   `pipeline`, `recorder`, `shaders`). For hosts that want to run
   their own compute shaders alongside graphics — sharing a device
   without running an LLM at all. This is the smallest contract:
   attach to host handles, record dispatches into the host's command
   buffer.

2. **Session-driven text generation** (`session.Session`). For hosts
   that want valkyr to do the inference work — load a model, ingest
   a prompt, tick the state machine each frame, get tokens out. The
   per-layer scheduling, KV/scratch lifecycle, and deferred-sample
   correctness invariants are owned in one place.

Hosts can use either tier alone or both together. The Session API
is built on top of the primitives — picking it doesn't lock you out
of the lower layer; the recorder you pass into `Session.tickFrame`
is the same recorder you use for your own dispatches.

## Build setup

valkyr is a Zig dependency. Add it to your `build.zig.zon`:

```zig
.dependencies = .{
    .valkyr = .{
        .path = "../path/to/tripvulkan",
        // Or — once published — a .url + .hash entry; for now path is
        // the only flavor.
    },
},
```

Pull the module in your `build.zig`:

```zig
const valkyr_dep = b.dependency("valkyr", .{
    .target = target,
    .optimize = optimize,
});
const valkyr_gpu_mod = valkyr_dep.module("valkyr_gpu");

// ... your executable creation ...
exe.root_module.addImport("valkyr_gpu", valkyr_gpu_mod);

// Vulkan + libC linking is YOUR responsibility. valkyr's module
// doesn't link them itself — you almost certainly already do this
// for your own Vulkan use.
exe.linkSystemLibrary("vulkan");
exe.linkLibC();
```

That's the whole build wiring. Zig's package system handles the rest
(SPIR-V shader compilation, transitive imports). A single
`@import("valkyr_gpu")` in your code reaches everything documented
below.

## Quick start: Session-driven text generation

The most common integration. Drop a model into your host and have it
generate text inside the render loop.

### One-time setup (e.g. inside your engine's init)

```zig
const vkr = @import("valkyr_gpu");

// Attach to host's existing Vulkan handles.
const ctx = vkr.vk.Context.attach(
    host_instance,
    host_physical_device,
    host_device,
    host_queue,
    host_queue_family,
    host_cmd_pool,
);

// Load a model. Either a directory or an HF id.
const loaded = try vkr.loader.loadGpuModel(
    allocator, &ctx, "google/gemma-2b-it",
    .{ .precision = .q4_k_matmul }, // or .bf16_matmul, .q4_0_matmul, .fp32_all
);

// Build the cooperative-inference Session.
var sess = try vkr.session.Session.init(
    allocator, &ctx, &loaded.gpu_model, &loaded.tokenizer,
    .{
        .budget_layers = 8,        // layers/frame; default 8
        .max_new_tokens = 256,
        .on_token = onToken,       // optional streaming callback
        .max_pos = 1024,
    },
);

try sess.appendPrompt("Once upon a time");
```

### Per frame (inside your render frame, after begin and any sync)

```zig
// Attach the recorder to the host's per-frame command buffer.
// Allocated once at engine init and reused; the descriptor pool is
// reset each frame.
var rec = try vkr.recorder.Recorder.attachCmd(
    &ctx, host_cmd, /* max_sets */ 512, /* max_descriptors */ 2048,
);

try rec.reset();
try rec.begin(); // attached-mode no-op (host already began host_cmd)

// ... your own compute dispatches that the LLM might want to read ...

// Drive the state machine. Records up to budget_layers worth of
// dispatches into host_cmd, advances internal state across frames.
const r = try sess.tickFrame(&rec);
if (r.new_token) |tok| {
    // Token just emerged. The on_token callback already fired with
    // the decoded UTF-8; here you can do anything else (animate UI,
    // write to a log, etc.).
    _ = tok;
}

// ... your downstream rendering passes that consume valkyr's outputs ...
```

The host then ends + submits + fences `host_cmd` as part of its
normal frame submit. valkyr does no submitting on its own.

### The token streaming callback

```zig
fn onToken(user: ?*anyopaque, tok_id: u32, decoded: []const u8) void {
    _ = user;
    _ = tok_id;
    // `decoded` is display-ready UTF-8. SentencePiece `▁`, byte-level
    // `Ġ` / `Ċ`, and `<0xBB>` byte-fallback tokens are already
    // resolved by `Tokenizer.decodeForDisplay` before the callback
    // fires — your loop just prints (or animates / forwards) the
    // bytes. The slice is owned by Session and valid only for the
    // duration of this call; copy if you need it later.
    std.debug.print("{s}", .{decoded});
}
```

Pass `user` as your engine's context (NPC pointer, dialog box state,
etc.) via `Config.on_token_user`. The pointer is opaque to valkyr.

### Family support

`Session.init` picks a dense or hybrid backend automatically based on
`cfg.family.isHybrid()`:

- **Dense** (`Backend.dense`): Gemma 1, Llama 3 / Llama 2-arch chat
  fine-tunes, Mistral 7B v0.3, Qwen3. Runs through `runtime.zig`'s
  ChatKernels + GpuScratch + GpuKvCache.
- **Hybrid** (`Backend.hybrid`): Qwen3.5 (and architecturally Qwen3.6)
  — Gated DeltaNet linear-attention layers interleaved with full
  attention, attn-output gate, partial RoPE. Runs through
  `runtime_hybrid.zig`'s ChatKernels + Scratch + State (per-layer SSM
  conv/recurrent + per-layer KV cache).

Hosts get the same Session API for all of them. The branching is
internal; `tickFrame` dispatches to the right runtime module based on
the loaded model's family.

## Frame budget mechanics

`Config.budget_layers` is the per-`tickFrame` work cap. Each
`recordOneLayer` counts as one unit; the sample step counts as one;
`recordEmbedding` is free (single dispatch, sub-microsecond).

For Gemma 2B IT (18 layers) at `budget_layers = 8`:
- Token forward = 18 layers + 1 sample step = 19 work units
- Spread across `ceil(19 / 8) + 1` = ~3 frames per token
  (the +1 is the deferred-sample tick — sampling reads logits AFTER
   the host's frame fence signals)
- At 60 fps: **~20 tok/s** with full graphics also rendering

Tune `budget_layers` to taste:
- Lower (4) when frame budget is tight or model is in the background
- Higher (16) for foreground-priority generation
- Max budget = total layers = whole forward in one frame

There's no wall-clock timer. The budget is layers-per-frame, not
microseconds. The host's frame fence at the top of the next
`drawFrame` ensures the previous frame's submit completed, so by
the time `tickFrame` runs again everything's deterministic.

## Sampling readback strategy

Greedy sampling needs to read the LM head's logits on the CPU.
Naive readback would be a separate submit + fence per token. valkyr
avoids this:

1. The Session allocates a `HOST_VISIBLE+HOST_COHERENT` "logits
   mirror" buffer with `TRANSFER_DST_BIT`, persistent-mapped.
2. After the LM head matmul, the recorder appends an explicit
   `SHADER_WRITE → TRANSFER_READ` barrier, then a `vkCmdCopyBuffer`
   from `scratch.logits` to the mirror.
3. The host's post-AI memory barrier (see below) covers
   `TRANSFER_WRITE → HOST_READ`.
4. After the host's frame fence signals, the persistent map is
   observable. The next `tickFrame` reads the mirror, samples,
   appends the token, advances `pos`.

**Net cost: zero extra submits per token.** One ~512 KB copy at
sample time (Gemma's vocab × 4 bytes), measured in microseconds.

## The aiDispatch hook (host renderer side)

This is **your engine's** half of the integration — the place you
expose for valkyr (or any other cooperative compute consumer) to
record into your frame.

In Matryoshka the contract is:

```zig
// In your Renderer / engine equivalent.
pub const AiDispatchFn = *const fn (
    user: *anyopaque,
    handles: VulkanHandles,  // your own handle bundle
    cmd: c.VkCommandBuffer,  // host-owned, in recording state
) anyerror!void;

pub fn setAiDispatch(self: *Renderer, user: ?*anyopaque, fn_ptr: ?AiDispatchFn) void {
    self.ai_dispatch_user = user;
    self.ai_dispatch_fn = fn_ptr;
}

// Inside your drawFrame, AFTER any sync/barriers your AI consumer
// needs and BEFORE downstream passes that read valkyr's outputs:
if (self.ai_dispatch_fn) |hook| {
    try hook(self.ai_dispatch_user.?, self.vulkanHandles(), cmd);

    // Cover SHADER_WRITE | TRANSFER_WRITE → SHADER_READ | HOST_READ
    // so downstream traversal/lighting can sample valkyr's SSBOs and
    // sampling can read mirrors via persistent map after fence.
    var bar = std.mem.zeroes(c.VkMemoryBarrier);
    bar.sType = c.VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    bar.srcAccessMask = c.VK_ACCESS_SHADER_WRITE_BIT | c.VK_ACCESS_TRANSFER_WRITE_BIT;
    bar.dstAccessMask = c.VK_ACCESS_SHADER_READ_BIT | c.VK_ACCESS_HOST_READ_BIT;
    c.vkCmdPipelineBarrier(
        cmd,
        c.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | c.VK_PIPELINE_STAGE_TRANSFER_BIT,
        c.VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
        0, 1, &bar, 0, null, 0, null,
    );
}
```

The hook fires inside the recording window. `cmd` is in begin state.
You don't end + submit — that's still your engine's responsibility,
unchanged.

## Per-token visualization with `on_layer`

valkyr fires an `on_layer` callback after each `recordOneLayer` if
the host registers one in `Config.on_layer`. The callback sees the
recorder + `scratch` and can record its own dispatches or copies
into the same command buffer.

**Dense-only today.** The callback signature takes
`*const gpu_scratch.GpuScratch` — the dense scratch struct. Hybrid
sessions don't fire it (Session.init prints a one-time warning if a
host registers `on_layer` against a hybrid model so the gap surfaces
clearly). Hybrid attention has a different shape — `scratch.scores`
is only valid on full-attention layers (1 in 4 for Qwen3.5), and the
linear-attention layers carry SSM state instead. A future chunk will
generalize the hook for hybrid hosts that want it.

This is the hook that lets a dense host build *real-time tensor
visualizers* without ever leaving the GPU — what's currently driving
the rainbow attention strip in Matryoshka's `ai_demo` running Gemma
2B IT.

```zig
fn onLayer(
    user: ?*anyopaque,
    rec: *vkr.recorder.Recorder,
    layer_idx: u32,
    scratch: *const vkr.gpu_scratch.GpuScratch,
) anyerror!void {
    const state: *State = @ptrCast(@alignCast(user.?));

    // Only act on the layer we care about.
    if (layer_idx != state.viz_layer) return;

    // Insert SHADER_WRITE → TRANSFER_READ barrier (the recorder's
    // auto-barriers don't cover transfer reads).
    var bar = std.mem.zeroes(c.VkMemoryBarrier);
    bar.sType = c.VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    bar.srcAccessMask = c.VK_ACCESS_SHADER_WRITE_BIT;
    bar.dstAccessMask = c.VK_ACCESS_TRANSFER_READ_BIT;
    c.vkCmdPipelineBarrier(
        @ptrCast(rec.cmd),
        c.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        c.VK_PIPELINE_STAGE_TRANSFER_BIT,
        0, 1, &bar, 0, null, 0, null,
    );

    // Mirror one head's first-N attention values from scratch.scores
    // (layout: row-major [n_heads, max_pos] f32) into a host-visible
    // buffer the host already allocated.
    const head_offset = state.viz_head * scratch.max_pos * @sizeOf(f32);
    const region = c.VkBufferCopy{
        .srcOffset = head_offset,
        .dstOffset = 0,
        .size = state.mirror.bytes,
    };
    c.vkCmdCopyBuffer(
        @ptrCast(rec.cmd),
        @ptrCast(scratch.scores.handle),
        @ptrCast(state.mirror.handle),
        1, &region,
    );
}
```

The host-visible mirror gets allocated once at engine setup via
`vkr.session.createHostMirrorBuffer(&ctx, bytes)` — this returns
exactly what's needed (HOST_VISIBLE+HOST_COHERENT, persistent-mapped,
TRANSFER_DST_BIT enabled).

What lives in `scratch` at hook time:
- `scratch.stream` — residual stream after this layer (hidden_dim f32)
- `scratch.scores` — post-softmax attention scores `[n_heads, max_pos]`,
  valid `[0..n_pos]` per row
- `scratch.q_rot`, `scratch.k_rot` — post-RoPE Q and K for THIS token

These are GPU-resident SSBOs; you can sample them in your render
shaders directly (with appropriate barriers) instead of mirroring
to host.

## Lifetime + ownership rules

valkyr's Vulkan structs split into two ownership modes via the
`owns_*` flags in `vk.Context` and `recorder.Recorder`:

- **`Context.init` / `Recorder.init`** — valkyr creates and owns the
  underlying Vulkan handles. `deinit` destroys them.
- **`Context.attach` / `Recorder.attachCmd`** — the host owns the
  handles. `deinit` is flag-gated to NOT destroy them. This is the
  embed mode.

**Rule of thumb**: any handle you pass into an `attach*` constructor,
*you* keep alive until valkyr's wrappers are deinited.

**Defer order matters.** The host's defer chain typically looks like:

```zig
defer renderer.deinit();      // destroys VkDevice
// ... game state setup ...
defer game.deinit(allocator); // frees valkyr resources tied to VkDevice
```

Defers are LIFO, so `game.deinit` runs FIRST at exit (correct: frees
GPU resources while device is alive), then `renderer.deinit` runs
(destroys device). If you reverse the order, you'll vkDestroyBuffer
on a destroyed device — silent corruption or validation-layer
firing depending on flags.

## Threading expectations

- **Single queue.** valkyr today serializes onto the host's queue.
  Multi-queue / async-compute integration is a separate design pass
  and isn't required for the current shape.
- **Single recording thread.** Anything that touches the recorder
  must be the same thread that owns the host's command buffer
  recording — Vulkan's `cmd_pool` external-sync rules apply.
- **No internal worker pools at runtime.** Model upload uses a job
  system briefly, then tears it down. Steady-state inference is
  single-threaded GPU work the host drives.

## When to use primitives instead of Session

The Session is the right answer for "generate text from a model."
Drop down to the primitives if you need:

- **Custom samplers**. Session is greedy-only today (the
  `SamplerKind` union has space for more). To plug in temperature /
  top-k / structured decoding, drive `runtime.recordOneLayer` and
  `runtime.recordSampleStep` yourself, sample with whatever logic
  you like, append the next token, repeat.
- **Multi-NPC scheduling**. Multiple Sessions exist independently;
  the host decides whose `tickFrame` runs each frame. Round-robin,
  attention-priority, "the NPC the player is looking at gets the
  budget" — all scheduling decisions live in the host.
- **Custom forward shapes**. If you want a probe / classifier head
  instead of an LM head, or want to skip layers, you can record
  exactly the dispatches you want via `runtime.recordEmbedding` +
  `runtime.recordOneLayer`s + your own final-step shader.

The primitives also stand alone for hosts that just want a Vulkan
compute consumer that shares their device — no model required.

## The Matryoshka case study

Matryoshka is a Vulkan/Zig real-time renderer that hosts valkyr in a
"Game" plugin (`src/games/ai_demo.zig`). It's a worked example of
every contract in this doc:

- `Renderer.vulkanHandles()` exposes the bundle valkyr attaches to.
- An `aiDispatch` vtable slot fires inside `drawFrame` with the
  host cmd buffer in recording state, between sim and traversal.
- Post-AI memory barrier covers SHADER/TRANSFER → SHADER/HOST.
- The Game owns a Session loaded with `google/gemma-2b-it`,
  appendPrompt'd with `"Once upon a time"`, ticked each frame.
- An `on_layer` callback mirrors one head's attention scores into a
  host-visible buffer; the Game's `update` reads via persistent
  map and modulates 16 dynamic point lights' intensities.
- LIFO defer order: `defer renderer.deinit()` registered first,
  `defer game.deinit()` registered second; game cleanup runs first.

Result at 60 fps: Gemma generates coherent text token-by-token while
a rainbow strip of point lights pulses with the model's last-layer
attention pattern, illuminating the cube and ground in real time.

[Matryoshka repo](https://github.com/foundation42/matryoshka)

## Known limitations

- **Greedy sampler only**, today. `SamplerKind` is a union; the
  shape's there but only `.greedy` is implemented. Custom samplers
  via the primitives layer (`runtime.recordSampleStep` +
  `runtime.sampleArgmax` + your own logic on the logits mirror).
- **`on_layer` is dense-only.** Hybrid sessions skip the hook (with
  a one-time warning at init) — the callback's `GpuScratch`
  signature doesn't match hybrid scratch, and hybrid attention is
  only valid on 1-in-4 layers. Generalizing to hybrid is a future
  chunk.
- **One Session per process** is fine but **multi-Session** has had
  no test exposure. Per-NPC dialog should work; it'll get serious
  testing when a host actually does it.
- **No TQ4 V-cache through Session.** The hooks exist in `runtime`
  / `runtime_hybrid` (`Tq4VHooks`) but Session doesn't expose them
  yet. Drop down to the primitives layer if you need asymmetric
  K=fp / V=TQ4 in-engine.
- **No multi-token-prediction draft head** (Qwen3.6 specific).
  Speculative decoding is on the broader roadmap, not this surface.

## What's not in scope here

This doc covers the embedded contract. Other paths:

- Standalone CLI / chat REPL: `valkyr --chat <model>`. See the main
  README's [Running](../README.md#running) section.
- HTTP / OpenAI-compatible server: not yet built. Would naturally
  share Session's machinery.
- Training: planned, see roadmap. Sits on top of valkyr's
  forward kernels + paired backwards in TRiP's `reference/math.c`.

## Pointers

- `src/lib.zig` — the public module's root. Each `pub const` is one
  importable submodule.
- `src/loader.zig` — `loadGpuModel` and friends.
- `src/runtime.zig` — dense forward primitives:
  `Forward.recordStep`, `recordForwardStep`, `recordOneLayer`,
  `recordEmbedding`, `recordSampleStep`, `sampleArgmax`,
  `ChatKernels`, push structs, dispatch helpers.
- `src/runtime_hybrid.zig` — hybrid (Qwen3.5 family) forward
  primitives: `ChatKernels`, `Scratch`, `State`, `Tq4VHooks`,
  `recordOneLayer`, `recordForwardStep`. Mirror of `runtime.zig`'s
  shape but with the Gated-DeltaNet kernel set + per-layer SSM
  state.
- `src/session.zig` — `Session`, `Backend` tagged union (dense /
  hybrid), `Config`, `TickResult`, `LayerCallback`, `TokenCallback`,
  `createHostMirrorBuffer`.
- `src/tokenizer.zig` — `decodeForDisplay` resolves `▁` / `Ġ` /
  byte-fallback for streaming hosts.
- `src/gpu/{vk,buffer,pipeline,recorder,scratch}.zig` — cooperative
  compute primitives + dense scratch / KV cache.

A headless validator lives in valkyr's CLI: `valkyr --session-smoke
<model>` runs the same code path matryoshka's `ai_demo` runs
in-engine, minus the GUI. Useful for verifying a model loads and
streams correctly through the embed surface before wiring it into a
host. Validated on Gemma 2B IT, Llama 3.2 1B-Instruct, Qwen3 4B
dense, Qwen3.5 0.8B hybrid.

If something here is wrong or unclear, the source is the source of
truth — these files are short enough to read top-to-bottom.
