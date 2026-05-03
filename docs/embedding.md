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

## Three API tiers

valkyr exposes a public Zig module called `valkyr_gpu`. Within it,
hosts can integrate at three levels — pick the highest one that does
what you need:

1. **Cooperative-compute primitives** (`vk.Context`, `buffer`,
   `pipeline`, `recorder`, `shaders`). Hosts that want to run their
   own compute shaders alongside graphics, sharing a device without
   running an LLM at all. Smallest contract: attach to host handles,
   record dispatches into the host's command buffer.

2. **Session-driven text generation** (`session.Session`). Hosts that
   want valkyr to do the inference work — load a model, ingest a
   prompt, tick the state machine each frame, get tokens out. The
   per-layer scheduling, KV/scratch lifecycle, and deferred-sample
   correctness invariants are owned in one place.

3. **InferenceRunner** (`inference.runner.InferenceRunner`). The
   queue-based scheduler that wraps Session with a request/event
   protocol. Submit a `Command.chat` with a `messages` array; drain
   `Event`s (`accepted` / `token` / `arena_swap` / `finish` / `err`)
   with `pollEvent`. **Same Runner powers `valkyr --serve`** (the
   OpenAI-compatible HTTP path — see [server.md](server.md)) —
   embed and HTTP eat from one inference abstraction. Inline mode
   ticks from the host's render loop; threaded mode owns its own
   worker thread.

Hosts can use any tier alone or in combination. Higher tiers are
built on top of lower ones — picking the Runner doesn't lock you
out of `runtime.recordOneLayer` for special cases.

## Build setup

valkyr is a Zig dependency. Add it to your `build.zig.zon`:

```zig
.dependencies = .{
    .valkyr = .{
        .path = "../path/to/valkyr",
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

## Quick start: InferenceRunner in inline-embed mode

The recommended integration for engine hosts. The Runner gives you
a queue-based API (chat command in, events out) and reuses the same
inference path as `valkyr --serve`. Inline mode means *you* drive
ticks from your render loop — no extra threads.

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

// Load a model — directory or HF id.
const loaded = try vkr.loader.loadGpuModel(
    allocator, &ctx, "google/gemma-2b-it",
    .{ .precision = .q4_k_matmul }, // or .bf16_matmul, .q4_0_matmul, .fp32_all
);

// Build the cooperative-inference Session.
var sess = try vkr.session.Session.init(
    allocator, &ctx, &loaded.gpu_model, &loaded.tokenizer,
    .{
        .budget = .{ .layers = 8 },   // layers/frame; default 8
        .max_new_tokens = 256,
        .max_pos = 1024,
        // Optional: per-layer attention viz hook (see below).
        // .on_layer = onLayer, .on_layer_user = state,
    },
);

// Recorder attached to host's per-frame VkCommandBuffer.
var rec = try vkr.recorder.Recorder.attachCmd(
    &ctx, host_cmd, /* max_sets */ 512, /* max_descriptors */ 2048,
);

// Build the runner; it borrows Session + Recorder.
var runner = try vkr.inference.runner.InferenceRunner.initBorrow(
    allocator, &sess, &rec,
    .{
        .default_budget = .{ .layers = 8 },
        .default_max_tokens = 256,
    },
);

// Submit a chat-templated request. The Runner deep-copies messages
// on accept; producer slices can be stack/static and freed after
// `submit` returns.
const messages = [_]vkr.chat_template.Message{
    .{ .role = .user, .content = "Once upon a time" },
};
try runner.submit(.{ .chat = .{
    .corr = 1,
    .messages = &messages,
    .max_tokens = 256,
} });
```

### Per frame (inside your render frame, after begin and any sync)

```zig
try rec.reset();        // descriptor pool reset; cmd buffer untouched
try rec.begin();        // no-op in attached mode

// ... your own compute dispatches that the LLM might want to read ...

// Advance the state machine. tickWork records up to cfg.budget
// worth of LLM forward dispatches into host_cmd; it does NOT touch
// cmd buffer reset/begin/end (your render loop owns that lifecycle).
try runner.tickWork();

// Drain runner events. Token events stream decoded UTF-8;
// arena_swap is bookkeeping; finish ends the request.
while (runner.pollEvent()) |ev| {
    switch (ev.kind) {
        .accepted => {},
        .token => |t| {
            const text = runner.resolve(t.decoded);
            std.debug.print("{s}", .{text});
        },
        .arena_swap => {},
        .finish => |f| {
            std.debug.print(
                "\n[done] reason={s} prompt={d} completion={d}\n",
                .{ @tagName(f.reason), f.prompt_tokens, f.completion_tokens },
            );
        },
        .err => |e| std.debug.print("\n[err] {s}\n", .{e.msg}),
    }
}

// ... your downstream rendering passes that consume valkyr's outputs ...
```

The host then ends + submits + fences `host_cmd` as part of its
normal frame submit. valkyr does no submitting on its own —
`tickWork` is the embed-shaped entry point that respects host cmd
buffer ownership.

### Chat templates: `appendMessages` does this for you

The Runner internally calls `Session.appendMessages(messages)` on
accept, which composes through the family's chat template:

- **Gemma**: `<bos><start_of_turn>user\n...<end_of_turn>\n<start_of_turn>model\n`
- **Qwen3 / Qwen3.5** (ChatML): `<|im_start|>user\n...<|im_end|>\n<|im_start|>assistant\n`
- **Llama 3** (header-id): `<|start_header_id|>user<|end_header_id|>\n\n...<|eot_id|>...`
- **Zephyr / TinyLlama**: text-marker `<|user|>\n...</s>\n<|assistant|>\n`
- **Mistral v0.3+**: `<s>[INST]...[/INST]`
- **System role** is folded into the first user turn for Gemma /
  Mistral (which lack a system role); emitted separately for the
  others. Multi-turn histories supported — pass the full
  conversation in `messages`.

This is what unlocks Qwen3.5's `<think>` reasoning preamble in the
embed path: the model's chat fine-tuning expects ChatML markers,
and the Runner makes sure they're emitted. Hosts that want to
bypass templating (raw token-level prompts for special cases) can
still call `Session.appendPrompt(raw_text)` directly; the Runner
only takes over once you submit a chat command.

### Without the Runner (legacy direct-Session path)

Hosts that need bespoke token-level orchestration (custom samplers,
multi-NPC scheduling) can drive `Session.tickFrame` directly:

```zig
try sess.appendMessages(&messages); // or appendPrompt(raw_text)

// Each frame:
try rec.reset();
try rec.begin();
const r = try sess.tickFrame(&rec);
if (r.new_token) |tok| { ... }
// host submits host_cmd
```

The Runner's mostly cleaner for the typical "stream tokens to
screen" case, but the direct path is still there.

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

`Config.budget` is the per-`tickFrame` work cap. Three modes via a
tagged union:

```zig
.budget = .{ .layers = 8 },                              // layer count cap
.budget = .{ .microseconds = 5000 },                     // wall-clock cap
.budget = .{ .either = .{ .layers = 8, .microseconds = 5000 } }, // both, first to fire wins
```

Each `recordOneLayer` counts as one layer-unit; the sample step
counts as one; `recordEmbedding` is free (single dispatch,
sub-microsecond). The time mode samples `Instant.now()` once per
layer-unit (~30 ns) — coarse enough that one expensive layer can
overshoot by one unit; the residual surfaces in `TickResult`.

For Gemma 2B IT (18 layers) at `.layers = 8`:
- Token forward = 18 layers + 1 sample step = 19 work units
- Spread across `ceil(19 / 8) + 1` = ~3 frames per token
  (the +1 is the deferred-sample tick — sampling reads logits AFTER
   the host's frame fence signals)
- At 60 fps: **~20 tok/s** with full graphics also rendering

`.either` is the right choice for most embed hosts: a coarse layer
backstop prevents the recorder's descriptor pool from overflowing
on big models, and the µs cap governs on small ones. The Runner's
default is `.either{layers=8, µs=5000}`.

`TickResult` reports both `elapsed_us` and signed `residual_us`
(target − elapsed; positive = under-budget, negative = overshot)
so a host can converge on a target frame budget with a simple
feedback loop. For `.layers`-only mode `residual_us` is zero (no
time signal). What we budget is *recording* time — Vulkan command
recording is what we steal from the host's frame; GPU work runs
async on the host's submit.

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
the host registers one in `Config.on_layer`. The callback receives
a `LayerTap` — a small struct that papers over both backends — and
can record its own dispatches or copies into the same command
buffer.

```zig
pub const LayerTap = struct {
    rec: *recorder.Recorder,
    layer_idx: u32,
    layer_kind: LayerKind,    // .full_attention | .linear_attention
    scores: ?*const buffer.Buffer,  // [n_q_heads, max_pos] f32, or null
    n_q_heads: u32,
    max_pos: u32,
};
```

The same hook fires on dense and hybrid models. On hybrid (Qwen3.5,
Qwen3.6) only **1 in 4** invocations carries non-null `scores` — the
full-attention layers; the other 3-in-4 are linear-attention (Gated
DeltaNet) and have no attention to mirror that frame. Hosts handle
the difference with a single `orelse return`:

```zig
fn onLayer(user: ?*anyopaque, tap: *const vkr.session.LayerTap) anyerror!void {
    const state: *State = @ptrCast(@alignCast(user.?));
    if (tap.layer_idx != state.viz_layer) return;
    const scores = tap.scores orelse return;  // skip linear-attn frames

    // SHADER_WRITE → TRANSFER_READ barrier (recorder's auto-barriers
    // don't cover transfer reads).
    var bar = std.mem.zeroes(c.VkMemoryBarrier);
    bar.sType = c.VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    bar.srcAccessMask = c.VK_ACCESS_SHADER_WRITE_BIT;
    bar.dstAccessMask = c.VK_ACCESS_TRANSFER_READ_BIT;
    c.vkCmdPipelineBarrier(
        @ptrCast(tap.rec.cmd),
        c.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        c.VK_PIPELINE_STAGE_TRANSFER_BIT,
        0, 1, &bar, 0, null, 0, null,
    );

    // Mirror one head's first-N attention values from `scores`
    // (layout [n_q_heads, max_pos] f32 row-major) into a host-visible
    // buffer the host already allocated.
    const head_offset = @as(u64, state.viz_head) * @as(u64, tap.max_pos) * @sizeOf(f32);
    const region = c.VkBufferCopy{
        .srcOffset = head_offset,
        .dstOffset = 0,
        .size = state.mirror.bytes,
    };
    c.vkCmdCopyBuffer(
        @ptrCast(tap.rec.cmd),
        @ptrCast(scores.handle),
        @ptrCast(state.mirror.handle),
        1, &region,
    );
}
```

The host-visible mirror gets allocated once at engine setup via
`vkr.session.createHostMirrorBuffer(&ctx, bytes)` — this returns
exactly what's needed (HOST_VISIBLE+HOST_COHERENT, persistent-mapped,
TRANSFER_DST_BIT enabled).

**Picking a viz layer for hybrid models.** Qwen3.5 / Qwen3.6 schedule
linear×3 + full×1 layers; the last layer is full-attention on the
sizes we ship support for, so `cfg.num_hidden_layers - 1` is a safe
default. If a future config lands the last layer on linear-attn,
the host's `orelse return` keeps the strip on its previous value
instead of glitching.

This is the hook driving the rainbow attention strip in Matryoshka's
`ai_demo` — works for both Gemma 2B IT (dense, every frame carries
scores) and Qwen3.5 0.8B (hybrid, scores arrive on every 4th
recorded layer).

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
- The Game owns a Session + Recorder + InferenceRunner loaded with
  `google/gemma-2b-it`. A single chat command is submitted at
  startup with `"Once upon a time"` as a user message; the Runner's
  `tickWork` is called from `aiDispatch` each frame.
- The runner emits `token` events synchronously inside `tickWork`;
  `aiDispatch` drains `pollEvent` after each tick to print
  decoded text and react to `finish`.
- An `on_layer` callback (still wired to Session, orthogonal to
  the Runner) mirrors one head's attention scores into a
  host-visible buffer; the Game's `update` reads via persistent
  map and modulates 16 dynamic point lights' intensities.
- LIFO defer order: `defer renderer.deinit()` registered first,
  `defer game.deinit()` registered second; game cleanup runs first
  (which itself deinits Runner before Session before model).

Result at 60 fps: Gemma generates coherent chat-templated text
token-by-token while a rainbow strip of point lights pulses with
the model's last-layer attention pattern, illuminating the cube
and ground in real time. Switch the model to Qwen3.5 hybrid and
the same code emits the model's `<think>` reasoning preamble
in-render.

[Matryoshka repo](https://github.com/foundation42/matryoshka)

## Known limitations

- **Greedy sampler only**, today. `SamplerKind` is a union; the
  shape's there but only `.greedy` is implemented. Custom samplers
  via the primitives layer (`runtime.recordSampleStep` +
  `runtime.sampleArgmax` + your own logic on the logits mirror).
- **`on_layer` exposes attention only.** The `LayerTap.scores`
  buffer covers post-softmax attention scores. SSM state on hybrid
  models (linear-attn `recurrent_state` / `conv_state`, the deltas
  through Gated DeltaNet) isn't piped through yet — viz hosts that
  want to render the linear-attn side need to wait for additional
  optional fields on `LayerTap`.
- **One Session per Runner.** Multi-Session (multiple concurrent
  conversations sharing GPU, e.g. per-NPC dialog) is on the
  roadmap; the Runner's command/event protocol is keyed by `corr`
  so the swap is "route per-corr to a Session pool", not a
  rewrite. Single-Session works for embed hosts that focus one
  NPC at a time.
- **No TQ4 V-cache through Session/Runner.** The hooks exist in
  `runtime` / `runtime_hybrid` (`Tq4VHooks`) but Session doesn't
  expose them yet. Drop to the primitives layer if you need
  asymmetric K=fp / V=TQ4 in-engine.
- **No multi-token-prediction draft head** (Qwen3.6 specific).
  Speculative decoding is on the broader roadmap.
- **Image/audio attachments accepted-but-rejected at the protocol
  level.** `ChatCommand.attachments` carries a future-proofed
  `Attachment` union with `.text`/`.image_url`/`.image_bytes`
  variants; v0 only routes `.text` and rejects the others with a
  clean error. Vision lands when the encoder side does.

## What's not in scope here

This doc covers the embedded contract. Other paths:

- Standalone CLI / chat REPL: `valkyr --chat <model>`. See the main
  README's [Running](../README.md#running) section.
- **OpenAI-compatible HTTP server**: see [server.md](server.md).
  `valkyr --serve <model>` exposes `/v1/chat/completions` (streaming
  + non-streaming) and `/v1/models` over the same `InferenceRunner`
  the embed path uses.
- Training: planned, see roadmap. Sits on top of valkyr's
  forward kernels + paired backward primitives, each parity-checked
  against a CPU oracle.

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
  hybrid), `Config`, `Budget`, `TickResult`, `LayerCallback`,
  `TokenCallback`, `appendPrompt`, `appendMessages`,
  `createHostMirrorBuffer`.
- `src/chat_template.zig` — `Role`, `Message`, `ChatTemplate` with
  `composeConversation` (multi-turn) and `composePrompt` (legacy
  single-turn). Per-family special-token tables.
- `src/inference/queue.zig` — `SpscRing(T, cap_log2)` lock-free
  ring buffer.
- `src/inference/arena.zig` — `PingPongArena` + `DecodedSlice` for
  zero-alloc streaming text.
- `src/inference/proto.zig` — `Command`, `Event`, `FinishReason`,
  `Attachment` (text/image_url/image_bytes — v0 routes text only).
- `src/inference/runner.zig` — `InferenceRunner` with
  `initBorrow`, `submit`, `pollEvent`, `tickInline` (owns
  recorder), `tickWork` (host-attached recorder), `start` /
  `shutdown` for threaded mode.
- `src/server/{http,json,server}.zig` — bare-metal HTTP/1.1 +
  OpenAI codec + Server. See [server.md](server.md).
- `src/tokenizer.zig` — `decodeForDisplay` resolves `▁` / `Ġ` /
  byte-fallback for streaming hosts.
- `src/gpu/{vk,buffer,pipeline,recorder,scratch}.zig` — cooperative
  compute primitives + dense scratch / KV cache.

Three headless validators live in valkyr's CLI:
- `valkyr --session-smoke <model>` — direct Session.tickFrame path.
- `valkyr --runner-smoke <model>` — InferenceRunner inline mode.
- `valkyr --runner-smoke-threaded <model>` — Runner with worker
  thread.

All three produce bit-identical text across the four supported
families (Gemma 2B IT, Llama 3.2 1B-Instruct, Qwen3 4B dense,
Qwen3.5 0.8B hybrid). Useful for verifying a model loads and
streams correctly through the embed surface before wiring it into
a host.

If something here is wrong or unclear, the source is the source of
truth — these files are short enough to read top-to-bottom.
