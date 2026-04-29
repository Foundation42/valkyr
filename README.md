# tripvulkan

A Vulkan-compute LLM inference engine in Zig. Greedy and sampled
text generation, multi-turn chat, three-tier parity verified against
HuggingFace transformers.

This is a port of the math from
[Carlo Valenti's TRiP](https://github.com/) ("Transformers in
Progress" — `reference/` in this repo) to Zig + Vulkan compute. TRiP
is a few-files-in-C, single-author transformer engine built for
clarity; tripvulkan keeps that pedagogical spirit but trades the
OpenMP-on-CPU back end for a Vulkan compute back end that runs the
same math on any GPU that supports Vulkan 1.3 (AMD / Intel / NVIDIA /
Apple via MoltenVK / Android).

## What works today

- **Gemma 1 family** (`google/gemma-2b-it` is the primary test model):
  18-layer transformer with multi-query attention, GeGLU FFN, RoPE,
  RMSNorm, tied LM head.
- **End-to-end forward pass on GPU** — embed → 18 × (rmsnorm → Q/K/V
  → RoPE → attention with KV cache → o_proj → residual → rmsnorm →
  GeGLU FFN → residual) → final norm → LM head → logits.
- **Multi-turn chat** with prompt prefill, KV cache persistence
  across turns, and `<end_of_turn>` stop detection.
- **bf16 matmul weights on device** — halves both the upload time
  and the steady-state matmul memory bandwidth versus fp32.
- **Sampling**: greedy, temperature, top-K, top-P, with a `--seed`
  flag for reproducible runs.
- **HuggingFace tokenizer** — both decode and BPE encode (token-
  for-token parity with `tokenizers` on Gemma's 256k-entry vocab).

## How fast it goes

On `google/gemma-2b-it` / RTX 3090 / Vulkan 1.3 / Zig 0.14-dev,
ReleaseFast build:

| | |
|---|---|
| upload (10 GiB → bf16 matmul, fp32 layernorms/embed) | 10.1 s |
| cold first forward (includes GPU pipeline compile) | 225 ms |
| warm median forward | **8.27 ms** |
| warm p99 forward (p99/p50 = 1.08) | 8.90 ms |
| **throughput** | **~113 tokens/sec greedy** |

The whole engine — kernels, orchestration, and the tokenizer encoder —
is debug-build correct: the only thing ReleaseFast measurably wins on
host-side is the bf16→fp32 conversion during upload (15 s → 10 s).
Per-token forward time is GPU-bound and matches Debug.

## Three-tier parity

Every layer is parity-verified against the layer above:

```
HuggingFace transformers (Python, bf16)         argmax = 229711 (▁increa)
   │                                            ↑ for "<bos>" → next token
   ▼
CPU forward (Zig, fp32, lazy bf16 conversion)   argmax = 229711  ✓ matches HF
   │
   ▼
GPU forward (Vulkan, ~291 dispatches)           argmax = 229711  ✓ matches CPU
```

Token IDs at every rank match HuggingFace exactly. Logit values
differ by ~3.5e-4 absolute (~7e-6 relative) over the full 256000-
element vector — within fp32 round-off noise after 18 layers of
accumulation.

## Build

Requires Zig 0.14-dev, glslc (Vulkan SDK), and Vulkan-capable GPU
drivers. On Arch:

```sh
sudo pacman -S vulkan-headers vulkan-tools shaderc \
               vulkan-validation-layers   # optional, for development
```

Then:

```sh
zig build                              # debug
zig build -Doptimize=ReleaseFast       # release
```

The build compiles every `shaders/*.comp` to SPIR-V via `glslc`,
embeds it into the binary with `align(4)` (Vulkan needs 32-bit
alignment for `pCode`), and links against the system Vulkan loader.
Validation layers are auto-enabled in Debug / ReleaseSafe; the
loader probes for them and runs cleanly without if the package
isn't installed.

## Running

The binary is one executable with mode flags. All modes take a
HuggingFace snapshot directory containing `config.json`,
`tokenizer.json`, and the `*.safetensors` shard(s).

```sh
# Default smoke run — 11 small kernel + format tests, no model load
zig build run

# Inspect a checkpoint (no GPU touched)
./zig-out/bin/tripvulkan --inspect <model.safetensors>
./zig-out/bin/tripvulkan --load    <model-dir>

# CPU reference forward + greedy sample (the parity oracle)
./zig-out/bin/tripvulkan --gen <model-dir> <token-id>

# GPU forward + parity check vs the CPU oracle
./zig-out/bin/tripvulkan --gpu-gen <model-dir> <token-id>

# GPU streaming generation (KV cache, multi-position attention)
./zig-out/bin/tripvulkan --gpu-gen-many <model-dir> <token-id> <n>

# Chat (single-turn or multi-turn REPL)
./zig-out/bin/tripvulkan --chat <model-dir> "What is the capital of France?"
./zig-out/bin/tripvulkan --chat <model-dir>        # REPL with stdin

# Chat with sampling
./zig-out/bin/tripvulkan --chat <model-dir> \
    --temp 0.8 --top-p 0.9 --seed 42 \
    "Write a one-line haiku about Vulkan."

# Benchmark (warm/cold forward timing, tok/s, p99)
./zig-out/bin/tripvulkan --bench <model-dir> --n 128
```

For Gemma 2B IT the snapshot dir is typically inside
`~/.cache/huggingface/hub/models--google--gemma-2b-it/snapshots/<hash>/`.

## Architecture

```
src/
├── main.zig            CLI dispatch + per-mode orchestration
├── safetensors.zig     mmap-backed tensor reader (zero-copy)
├── sharded.zig         multi-file SafeTensors merge by-name
├── config.zig          HuggingFace config.json + Family enum (Gemma/Llama)
├── model.zig           CPU Model — config + tensors per layer, validated shapes
├── tokenizer.zig       BPE encode + decode, byte fallback, special-token lookup
├── dtype.zig           bf16/fp16 → fp32 bit-twiddling
├── cpu/
│   ├── math.zig        Reference primitives: rmsnorm, matmul_nt, RoPE, softmax,
│   │                   GELU, GeGLU, embedding lookup
│   └── forward.zig     CPU full forward + sample (greedy/temp/top-k/top-p)
└── gpu/
    ├── vk.zig          Headless Vulkan compute context
    ├── buffer.zig      Static / dynamic / device-only buffer abstraction
    ├── pipeline.zig    Compute pipeline + descriptor set wrapper
    ├── recorder.zig    One-command-buffer dispatch batcher
    ├── model.zig       GpuModel + Precision (fp32_all | bf16_matmul)
    └── scratch.zig     Per-forward activation buffers + per-layer KV cache

shaders/
├── matmul_nt.comp          naive 1-thread-per-cell baseline (parity oracle)
├── matmul_nt_v2.comp       cooperative-K reduction, fp32 weights (4× faster)
├── matmul_nt_v2_bf16.comp  bf16 weights, packed u32 reads, bit-shift to fp32
├── rmsnorm.comp            subgroup-reduced sum-of-squares + Gemma quirk
├── softmax.comp            stable two-pass with stride support
├── geglu.comp              fused gelu_tanh(gate) · up
├── rope.comp               half-split RoPE, one thread per (h, j) pair
├── attn_scores.comp        Q · K_cache, cooperative reduction
├── attn_output.comp        scores · V_cache, cooperative reduction
├── kv_write.comp           scatter K_rot/V into the per-layer cache
├── embed_lookup.comp       embedding row + Gemma sqrt(hidden) scale
├── add_in_place.comp       residual stream += contribution
└── vec_add.comp            hello-world smoke test
```

13 shaders in total. Every kernel has a CPU reference counterpart and
a parity test (synthetic input + a real-Gemma input where it makes
sense).

## Convention notes

- **RoPE**: HuggingFace half-split convention (pairs `(j, j+D/2)`).
  TRiP uses pairwise interleaved with a custom permuting matmul;
  we don't, so the standard matmul against HF-stored Q/K weights
  feeds straight into the half-split RoPE — same final attention
  scores, slightly different intermediate layouts.

- **Gemma quirks** the engine handles:
    - Embedding scaled by `sqrt(hidden_size)` before the first block.
    - RMSNorm gain is `(1 + weight)` rather than plain `weight`.
    - GeGLU FFN with `gelu_pytorch_tanh` activation despite the
      config claiming `"hidden_act": "gelu"` — the IT model was
      trained with the tanh approximation.
    - LM head tied to `embed_tokens` (no separate `lm_head.weight`).

- **Numerical drift**: max |Δ| over the whole 256000-element logit
  vector vs HF transformers is ~3.5e-4. Token-ID ranks match
  exactly through at least the top 5; greedy decoding produces
  byte-identical first generated tokens.

## Limitations

- **Gemma 1 only** today. Llama 2 / 3 should mostly fall out from
  the `Family` enum + a SwiGLU shader, but isn't validated yet.
- The naive `matmul_nt_v2` kernel hits roughly 0.1% of the 3090's
  fp32 peak. Most of the warm forward time is now in the FFN
  matmuls reading bf16 weights memory-bandwidth-bound — proper
  shared-memory tiling, fused attention (FlashAttention-style),
  and a bf16 embedding kernel are obvious wins.
- Single-stream batching only (M = 1 in every matmul). The kernels
  generalise to M > 1 but no caller dispatches that way.
- Tokenizer encoder doesn't pre-protect special-token strings, so
  `--encode <dir> "<bos>"` would BPE the angle brackets rather
  than emit token 2. Chat composes specials by ID explicitly so
  this doesn't bite in practice.

## License

The Zig + GLSL code in this repo is yours to read and learn from.
Carlo Valenti's TRiP (in `reference/`, gitignored) is CC-BY-NC 4.0;
this project derives ideas and per-op accumulation order from it,
so the same non-commercial restriction applies to derivatives.

## Acknowledgements

- **Carlo Valenti** for [TRiP](https://github.com/) — the C
  reference whose math we ported and whose paired forward/backward
  layout we kept in spirit.
- **Andrej Karpathy** for llama2.c and the lectures that gave both
  TRiP and this port a starting point.
- **Google** for Gemma; **HuggingFace** for the `transformers` and
  `tokenizers` libraries used as the parity oracle.
