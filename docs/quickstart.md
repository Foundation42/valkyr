# Quickstart (build + run)

Build dependencies, the `zig build` invocation, and the full set of
mode flags the binary supports — `--list`, `--inspect`, `--load`,
`--gen` and friends, `--chat`, `--bench`, `--serve`, and the headless
embed validators. Back to [README](../README.md). See also:
[models.md](models.md), [embedding.md](embedding.md),
[server.md](server.md), [hardware.md](hardware.md).

## Build

Requires **Zig 0.14-dev**, **glslc** (Vulkan SDK / `shaderc`), and
Vulkan-capable GPU drivers. On Arch:

```sh
sudo pacman -S vulkan-headers vulkan-tools shaderc \
               vulkan-validation-layers   # optional, for development
```

On Debian / Ubuntu the equivalent is `libvulkan-dev vulkan-tools
glslang-tools`. On macOS install the [Vulkan SDK from
LunarG](https://vulkan.lunarg.com/sdk/home) (which bundles MoltenVK).

Then:

```sh
zig build                              # debug
zig build -Doptimize=ReleaseFast       # release (recommended for chat)
```

The build compiles every `shaders/*.comp` to SPIR-V via `glslc`,
embeds it into the binary with `align(4)` (Vulkan needs 32-bit
alignment for `pCode`), and links against the system Vulkan loader.
Validation layers are auto-enabled in Debug / ReleaseSafe; the loader
probes for them and runs cleanly without if the package isn't
installed.

## Running

The binary is one executable with mode flags. Modes that take a
model accept either a path to a HuggingFace snapshot directory
(`config.json` + `tokenizer.json` + `*.safetensors`) **or** an HF
model id like `meta-llama/Llama-3.2-3B-Instruct` — when the arg
contains `/` and isn't an existing path, valkyr resolves it to
the snapshot under `~/.cache/huggingface/hub/` (honoring the
standard `HF_HUB_CACHE` / `HF_HOME` / `XDG_CACHE_HOME` env vars).

```sh
# Default smoke run — 21 small kernel + format + parity tests, no model load
zig build run

# List cached models, marking which ones valkyr can load
./zig-out/bin/valkyr --list

# Inspect a checkpoint (no GPU touched)
./zig-out/bin/valkyr --inspect <model.safetensors>
./zig-out/bin/valkyr --load    <model-dir-or-hf-id>

# CPU reference forward + greedy sample (the parity oracle)
./zig-out/bin/valkyr --gen <model-dir> <token-id>

# CPU forward with TQ4 V-cache, side-by-side vs fp32 baseline
./zig-out/bin/valkyr --gen-tq4v <model-dir> <token-id>

# GPU forward + parity check vs the CPU oracle
./zig-out/bin/valkyr --gpu-gen <model-dir> <token-id>

# GPU forward + parity for Qwen3.5 hybrid (linear + full attention)
./zig-out/bin/valkyr --gpu-gen-qwen35 <Qwen3.5-dir> 248044

# GPU forward with TQ4 V-cache, side-by-side vs fp32 baseline
./zig-out/bin/valkyr --gpu-gen-tq4v <model-dir> <token-id>

# GPU streaming generation (KV cache, multi-position attention)
./zig-out/bin/valkyr --gpu-gen-many <model-dir> <token-id> <n>

# Chat (single-turn or multi-turn REPL) — `<model>` is either an HF
# id (e.g. `meta-llama/Llama-3.2-3B-Instruct`) or a snapshot path.
./zig-out/bin/valkyr --chat meta-llama/Llama-3.2-3B-Instruct "What is the capital of France?"
./zig-out/bin/valkyr --chat <model>            # REPL with stdin

# Chat with TurboQuant V-cache (asymmetric K=fp / V=TQ4)
./zig-out/bin/valkyr --chat <model> --tq4v "..."

# Chat with Q4_0 4-bit weights (composes with --tq4v for the
# smallest-footprint configuration)
./zig-out/bin/valkyr --chat <model> --q4 "..."
./zig-out/bin/valkyr --chat <model> --q4 --tq4v "..."

# Chat with Q4_K_M 4-bit weights (super-block-256, asymmetric — same
# format llama.cpp ships as Q4_K_M.gguf; faster decode than --q4 +
# ~32% lower quantize MSE; mutually exclusive with --q4)
./zig-out/bin/valkyr --chat <model> --q4k "..."

# Chat with sampling (works with or without --tq4v)
./zig-out/bin/valkyr --chat <model> \
    --temp 0.8 --top-p 0.9 --seed 42 \
    "Write a one-line haiku about Vulkan."

# Benchmark (warm/cold forward timing, tok/s, p99) — fp32 baseline
./zig-out/bin/valkyr --bench <model-dir> --n 128

# Real-Gemma TQ4 round-trip diagnostic (per-layer K/V MSE)
./zig-out/bin/valkyr --tq4-kv-test <model-dir> <token-id>

# Fine-tune a model on a JSONL dataset, optionally probe sample +
# save a `.vkpt` checkpoint. See docs/training.md for the walkthrough.
./zig-out/bin/valkyr --fine-tune Qwen/Qwen3-0.6B \
    --data data/train/tiny_facts.jsonl \
    --steps 30 \
    --probe "The capital of France is" \
    --out fine-tuned.vkpt

# Generate text from a `.vkpt` checkpoint produced above.
./zig-out/bin/valkyr --gen-from-ckpt Qwen/Qwen3-0.6B \
    --ckpt fine-tuned.vkpt \
    --prompt "The capital of France is" \
    --n-gen 20

# OpenAI-compatible HTTP server (POST /v1/chat/completions, GET /v1/models)
# See docs/server.md for the full surface — streaming, multi-turn,
# error envelope, openai-python compatibility.
./zig-out/bin/valkyr --serve <model> --q4k --port 8080 --id <public-id>

# Headless validators for the embed surface (same code paths as
# Matryoshka's ai_demo, GUI-less). All four supported families
# produce bit-identical text across these and --chat.
./zig-out/bin/valkyr --session-smoke    <model> --q4k   # direct Session.tickFrame
./zig-out/bin/valkyr --session-messages <model> --q4k   # 3-turn chat-template fixture
./zig-out/bin/valkyr --runner-smoke           <model> --q4k   # InferenceRunner inline
./zig-out/bin/valkyr --runner-smoke-threaded  <model> --q4k   # Runner with worker thread
```

For Gemma 2B IT the snapshot dir is typically inside
`~/.cache/huggingface/hub/models--google--gemma-2b-it/snapshots/<hash>/`
after a `huggingface-cli download google/gemma-2b-it`.
