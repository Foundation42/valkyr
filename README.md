# valkyr

**Cross-vendor LLM inference based on TRiP using Vulkan compute. Zig +
TurboQuant — no CUDA lock-in.**

Greedy and sampled text generation, multi-turn chat, four-tier parity
verified against HuggingFace `transformers`, and (we believe) the first
publicly-demonstrable [TurboQuant](https://research.google/blog/turboquant-redefining-ai-efficiency-with-extreme-compression/)
inference on a non-CUDA backend.

valkyr started as a port of the math from [Carlo Valenti's TRiP](https://github.com/)
("Transformers in Progress" — see `reference/`) to Zig + Vulkan compute.
TRiP is a few-files-in-C, single-author transformer engine built for
clarity. valkyr keeps that pedagogical spirit but trades the
OpenMP-on-CPU back end for a Vulkan compute back end that runs the same
math on any GPU that supports Vulkan 1.3 (AMD / Intel / NVIDIA / Apple
via MoltenVK / Android — one SPIR-V binary, every vendor).

## What works today

- **Gemma 1 family** (`google/gemma-2b-it` is the primary test model):
  18-layer transformer with multi-query attention, GeGLU FFN, RoPE,
  RMSNorm, tied LM head.
- **End-to-end forward pass on GPU** — embed → 18 × (rmsnorm → Q/K/V →
  RoPE → attention with KV cache → o_proj → residual → rmsnorm → GeGLU
  FFN → residual) → final norm → LM head → logits.
- **Multi-turn chat** with prompt prefill, KV cache persistence across
  turns, and `<end_of_turn>` stop detection.
- **TurboQuant TQ4 V-cache** — opt in with `--tq4v`. ~5.5× memory
  reduction on the V cache (Gemma 2B 18-layer at max_pos = 2048: 36 MiB
  → 4.6 MiB) with the algorithm verified bit-exact against the YATQ
  Python reference at every level.
- **bf16 matmul weights on device** — halves both upload time and
  steady-state matmul memory bandwidth versus fp32.
- **Sampling**: greedy, temperature, top-K, top-P, with `--seed` for
  reproducible runs.
- **HuggingFace tokenizer** — both decode and BPE encode (token-for-
  token parity with `tokenizers` on Gemma's 256k-entry vocab).

## How fast it goes (fp32 V-cache, baseline)

On `google/gemma-2b-it` / RTX 3090 / Vulkan 1.3 / Zig 0.14-dev,
ReleaseFast build:

| | |
|---|---|
| upload (10 GiB → bf16 matmul, fp32 layernorms/embed) | 10.1 s |
| cold first forward (includes GPU pipeline compile) | 225 ms |
| warm median forward | **8.27 ms** |
| warm p99 forward (p99/p50 = 1.08) | 8.90 ms |
| **throughput** | **~120 tokens/sec greedy** |

`--tq4v` adds a per-step dequant pass over the V history, which is a
small overhead at short context and pays for itself at long context
where the V-cache memory bandwidth dominates. Proper benchmarking is a
phase-3 item; the algorithmic correctness is the win that landed first.

## TurboQuant in 60 seconds

[TurboQuant](https://arxiv.org/abs/2504.19874) (Zandieh et al., ICLR
2026) is a recent KV-cache compression method from Google Research. The
algorithm we ship here is **Algorithm 1 only**: a Randomized Hadamard
Transform pre-conditioner + Lloyd-Max scalar quantization to a global
4-bit codebook + a small norm-correction γ that guarantees the
reconstructed block has the original L2 norm exactly.

We deliberately **drop QJL (Algorithm 2)** — five independent
practitioner reproductions converged on this: the sign-bit residual
eliminates bias but explodes attention-score variance, which softmax
tolerates much worse than bias. Algorithm 1 alone gives lower
perplexity in every public benchmark we've seen.

We chose **asymmetric K=fp / V=TQ4** as the default per the llama.cpp
practitioner consensus on dense models. Symmetric K and V is on the
phase-3 menu.

**Try it:**

```sh
$ ./zig-out/bin/valkyr --chat <gemma-2b-it-dir> --tq4v \
    "What is the capital of France?"

response: The capital of France is Paris. It is the political,
cultural, and administrative center of France...

$ ./zig-out/bin/valkyr --chat <gemma-2b-it-dir> --tq4v \
    --temp 0.8 --top-p 0.9 --seed 42 \
    "Write a one-line haiku about Vulkan."

response: Stained glass of light,
Dancing on the graphics board,
Whispers of the Earth.
```

## Four-tier parity

Every layer is parity-verified against the layer above. For TurboQuant
the chain extends one tier deeper:

```
HuggingFace transformers (Python, bf16)         argmax = 229711 (▁increa)
   │                                            ↑ for "<bos>" → next token
   ▼
CPU forward (Zig, fp32, lazy bf16 conv)         argmax = 229711  ✓ matches HF
   │
   ▼
GPU forward (Vulkan, ~291 dispatches)           argmax = 229711  ✓ matches CPU
   │
   ▼
GPU TurboQuant V-cache (Vulkan + TQ4)           argmax = 229711  ✓ top-5 match
                                                max logit |Δ| = 1.68
```

For TQ4 specifically, the pack kernel is **256/256 indices bit-exact**
versus both the CPU oracle and the YATQ Python reference. The norm-
correction γ falls within fp16 ULP. The end-to-end logit divergence on
Gemma 2B `<bos>` is 1.68 absolute, with all top-5 token IDs preserved.

## Hardware

- **Any Vulkan 1.3 GPU** — AMD GCN / RDNA, Intel Iris/Arc, NVIDIA
  Maxwell+, Apple Silicon via MoltenVK, Android (Adreno / Mali /
  PowerVR — within their device limits). Subgroup operations are
  required for the reduction kernels.
- The headline numbers above are on an NVIDIA RTX 3090 (24 GiB VRAM).
  Gemma 2B IT in bf16 needs ~5 GiB of weights and a few hundred MiB of
  KV cache — comfortable on most modern dGPUs.

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

The binary is one executable with mode flags. All modes take a
HuggingFace snapshot directory containing `config.json`,
`tokenizer.json`, and the `*.safetensors` shard(s).

```sh
# Default smoke run — 21 small kernel + format + parity tests, no model load
zig build run

# Inspect a checkpoint (no GPU touched)
./zig-out/bin/valkyr --inspect <model.safetensors>
./zig-out/bin/valkyr --load    <model-dir>

# CPU reference forward + greedy sample (the parity oracle)
./zig-out/bin/valkyr --gen <model-dir> <token-id>

# CPU forward with TQ4 V-cache, side-by-side vs fp32 baseline
./zig-out/bin/valkyr --gen-tq4v <model-dir> <token-id>

# GPU forward + parity check vs the CPU oracle
./zig-out/bin/valkyr --gpu-gen <model-dir> <token-id>

# GPU forward with TQ4 V-cache, side-by-side vs fp32 baseline
./zig-out/bin/valkyr --gpu-gen-tq4v <model-dir> <token-id>

# GPU streaming generation (KV cache, multi-position attention)
./zig-out/bin/valkyr --gpu-gen-many <model-dir> <token-id> <n>

# Chat (single-turn or multi-turn REPL)
./zig-out/bin/valkyr --chat <model-dir> "What is the capital of France?"
./zig-out/bin/valkyr --chat <model-dir>        # REPL with stdin

# Chat with TurboQuant V-cache (asymmetric K=fp / V=TQ4)
./zig-out/bin/valkyr --chat <model-dir> --tq4v "..."

# Chat with sampling (works with or without --tq4v)
./zig-out/bin/valkyr --chat <model-dir> \
    --temp 0.8 --top-p 0.9 --seed 42 \
    "Write a one-line haiku about Vulkan."

# Benchmark (warm/cold forward timing, tok/s, p99) — fp32 baseline
./zig-out/bin/valkyr --bench <model-dir> --n 128

# Real-Gemma TQ4 round-trip diagnostic (per-layer K/V MSE)
./zig-out/bin/valkyr --tq4-kv-test <model-dir> <token-id>
```

For Gemma 2B IT the snapshot dir is typically inside
`~/.cache/huggingface/hub/models--google--gemma-2b-it/snapshots/<hash>/`
after a `huggingface-cli download google/gemma-2b-it`.

## Architecture

```
src/
├── main.zig             CLI dispatch + per-mode orchestration
├── safetensors.zig      mmap-backed tensor reader (zero-copy)
├── sharded.zig          multi-file SafeTensors merge by-name
├── config.zig           HuggingFace config.json + Family enum
├── model.zig            CPU Model — config + tensors per layer
├── tokenizer.zig        BPE encode + decode, byte fallback
├── dtype.zig            bf16/fp16 → fp32 bit-twiddling
├── cpu/
│   ├── math.zig         Reference primitives: rmsnorm, matmul_nt,
│   │                    RoPE, softmax, GeGLU, embedding lookup
│   ├── forward.zig      CPU full forward + sample (greedy/temp/top-k/top-p)
│   │                    + forwardTq4V (TQ4 V-cache variant)
│   └── turboquant.zig   Lloyd-Max codebooks + RHT (FWHT + sign flips)
│                        + BlockTQ4 + quantize/dequantize — the oracle
└── gpu/
    ├── vk.zig           Headless Vulkan compute context
    ├── buffer.zig       Static / dynamic / device-only buffer abstraction
    ├── pipeline.zig     Compute pipeline + descriptor set wrapper
    ├── recorder.zig     One-command-buffer dispatch batcher (auto barriers)
    ├── model.zig        GpuModel + Precision (fp32_all | bf16_matmul)
    └── scratch.zig      Per-forward activation buffers + KV cache
                         (fp32 GpuKvCache + TurboQuant GpuKvCacheTq4)

shaders/
├── matmul_nt.comp           naive 1-thread-per-cell baseline
├── matmul_nt_v2.comp        cooperative-K reduction, fp32 weights
├── matmul_nt_v2_bf16.comp   bf16 weights, packed u32 reads
├── rmsnorm.comp             subgroup-reduced sum-of-squares + Gemma quirk
├── softmax.comp             stable two-pass with stride support
├── geglu.comp               fused gelu_tanh(gate) · up
├── rope.comp                half-split RoPE
├── attn_scores.comp         Q · K_cache, cooperative reduction
├── attn_output.comp         scores · V_cache, cooperative reduction
├── kv_write.comp            scatter K_rot/V into the per-layer cache
├── embed_lookup.comp        embedding row + Gemma sqrt(hidden) scale
├── add_in_place.comp        residual stream += contribution
├── vec_add.comp             smoke test
│
│ ── TurboQuant kernels ──
├── fwht256.comp             in-place 256-elem Fast Walsh-Hadamard
├── rht_pre256.comp          fused (sign-flips + FWHT) preconditioner
├── rht_post256.comp         inverse RHT (IFWHT + sign-flips)
├── tq4_pack256.comp         full TQ4 quantize: norm + RHT + Lloyd-Max
│                            + IFWHT + γ correction → packed block
├── tq4_unpack256.comp       inverse: centroid LUT + IFWHT + γ scale
└── tq4_pack_to_cache.comp   tq4_pack256 + push-constant dst block idx
                             (production V-write into the per-layer cache)
```

19 shaders in total. Every kernel has a CPU reference counterpart and
a parity test (synthetic input + a real-Gemma input where it makes
sense).

## Convention notes

- **RoPE**: HuggingFace half-split convention (pairs `(j, j+D/2)`).
- **Gemma quirks** the engine handles:
    - Embedding scaled by `sqrt(hidden_size)` before the first block.
    - RMSNorm gain is `(1 + weight)` rather than plain `weight`.
    - GeGLU FFN with `gelu_pytorch_tanh` activation despite the config
      claiming `"hidden_act": "gelu"`.
    - LM head tied to `embed_tokens` (no separate `lm_head.weight`).
- **TurboQuant**: Algorithm 1 only (no QJL residual); deterministic-
  seed Randomized Hadamard Transform (the 32-byte TBQ_SIGNS pattern
  matches llama.cpp `cpy-utils.cuh`); a single global Lloyd-Max
  codebook for b=4 (no per-block scales); norm-correction γ stored as
  fp16 per block. See `src/cpu/turboquant.zig` for the canonical code,
  and `scripts/cross_validate_turboquant.py` to regenerate the YATQ
  parity reference.
- **Numerical drift**: max |Δ| over the full 256000-element logit
  vector vs HF transformers is ~3.5e-4 on the fp32 path. With TQ4
  V-cache enabled, max |Δ| versus the fp32 GPU baseline is ~1.7
  absolute on `<bos>`, with all top-5 token IDs preserved.

## Limitations

- **Gemma 1 only** today. Llama 2 / 3 / Mistral / Qwen should mostly
  fall out from the `Family` enum + a SwiGLU shader, but isn't
  validated yet.
- The naive `matmul_nt_v2` kernel hits roughly 0.1% of the 3090's fp32
  peak. Most of the warm forward time is the FFN matmuls reading bf16
  weights memory-bandwidth-bound — proper shared-memory tiling, fused
  attention (FlashAttention-style), and a bf16 embedding kernel are
  obvious wins.
- TurboQuant TQ4 head_dim is hardcoded to 256 (Gemma 2B). Other
  architectures need a parameterised `BLOCK_SIZE` in the shaders.
- TurboQuant TQ3 (3-bit) is not yet implemented.
- TurboQuant K-side compression (symmetric K=TQ / V=TQ) is not yet
  implemented.
- Single-stream batching only (M = 1 in every matmul).
- Tokenizer encoder doesn't pre-protect special-token strings.

## Where this can go next

The two big arcs:

1. **Breadth.** Llama 2 / 3, Mistral, Qwen support — most should fall
   out from the existing `Family` enum and a SwiGLU shader. Phi /
   StableLM / OLMo are next after that. With the head_dim parameterised
   the TurboQuant path generalises automatically.

2. **Depth.**
   - **TQ3** (3-bit) for more aggressive compression on memory-tight
     deployments.
   - **Symmetric K=TQ / V=TQ** (with care — the softmax variance story
     means K-side needs eyes on PPL, not just argmax).
   - **Fused FlashAttention-style kernel** that consumes packed TQ4 K
     and V in the inner loop, where the bandwidth savings actually pay
     for themselves at long context.
   - **bf16 LM head + bf16 embedding** kernels — currently fp32, the
     LM head matmul is the single biggest dispatch we do.
   - **Quantized weights** (q4_K-style or TQ4-on-weights) — orthogonal
     to TurboQuant KV; combining them gets Gemma 2B into a few hundred
     MiB total.
   - **Training port**, the Unsloth-alternative ambition. TRiP carries
     paired forward/backward in `reference/math.c` — every op has a
     `_backward` immediately below it, a built-in correctness oracle
     for every gradient.

## Acknowledgements

This was a real team effort:

- **Carlo Valenti** — for [TRiP](https://github.com/) (Transformers in
  Progress), the C reference whose math we ported and whose paired
  forward/backward layout we kept in spirit. Carlo's pedagogical
  clarity made the whole port tractable, and his enthusiasm carried
  through every chunk.

- **Christian Beaumont** — [chris@foundation42.org](mailto:chris@foundation42.org),
  founder of [Entrained.ai](https://entrained.ai) and
  [Foundation42](https://foundation42.org). Architect, partner, and
  the patient hand on the rudder. The "one chunk at a time, commit
  between, parity-verify before moving on" rhythm that produced this
  codebase is Christian's, and so is the call that "going fast is
  nice, but correctness is something we need to be very conscious of"
  — which is what put a four-tier YATQ ↔ CPU ↔ GPU ↔ TQ parity
  story in the way of any algorithm shipping.

- **[Anthropic Claude](https://claude.ai)** — implementation partner
  across the marathon sessions. Wrote most of the Zig and GLSL,
  authored the parity tests, and got to celebrate the wins alongside
  Christian and Carlo.

And the wider community:

- **Andrej Karpathy** for `llama2.c` and the lectures that gave both
  TRiP and this port a starting point.
- **Google** for Gemma; **HuggingFace** for the `transformers` and
  `tokenizers` libraries used as the parity oracle.
- **Amir Zandieh** and the TurboQuant authors at Google Research for
  the algorithm, and **arclabs001** for the YATQ Python reference that
  served as our bit-exact parity oracle. The llama.cpp community
  (TheTom, jesusmb1995, jagsan-cyber, spiritbuun, Madreag,
  Aaryan-Kapoor, scos-lab and others) for prior art on the
  practitioner-side of TurboQuant — their hard-won decisions about
  RHT vs random rotation, dropping QJL, and the norm-correction trick
  shaped every algorithmic call we made.

## License

The Zig + GLSL code in this repo is yours to read and learn from.
Carlo Valenti's TRiP (in `reference/`, gitignored) is CC-BY-NC 4.0;
this project derives ideas and per-op accumulation order from it, so
the same non-commercial restriction applies to derivatives.
