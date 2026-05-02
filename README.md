# valkyr

**Cross-vendor LLM inference based on TRiP using Vulkan compute. Zig +
TurboQuant — no CUDA lock-in.**

Greedy and sampled text generation, multi-turn chat, four-tier parity
verified against HuggingFace `transformers`, and (we believe) the first
publicly-demonstrable [TurboQuant](https://research.google/blog/turboquant-redefining-ai-efficiency-with-extreme-compression/)
inference on a non-CUDA backend.

valkyr started as a port of the math from [Carlo Valenti's TRiP](https://github.com/carlovalenti/TRiP)
("Transformers in Progress" — see `reference/`) to Zig + Vulkan compute.
TRiP is a few-files-in-C, single-author transformer engine built for
clarity. valkyr keeps that pedagogical spirit but trades the
OpenMP-on-CPU back end for a Vulkan compute back end that runs the same
math on any GPU that supports Vulkan 1.3 (AMD / Intel / NVIDIA / Apple
via MoltenVK / Android — one SPIR-V binary, every vendor).

## Why valkyr?

valkyr is small (the matmul shader fits on a screen), young (months not
years), and intentionally *less* than llama.cpp. So why pick it up?

- **One backend, every GPU.** A single Vulkan/SPIR-V kernel set runs on
  NVIDIA, AMD, Intel Arc, Apple Silicon (via MoltenVK), and Android
  Adreno / Mali. llama.cpp has separate CUDA / ROCm / Metal / Vulkan /
  SYCL backends; each has its own kernel set, its own quirks, and its
  own performance ceiling. If your hardware mix is heterogeneous, or
  you don't want to bet on CUDA being on every machine forever, the
  one-Vulkan-binary story matters.

- **Composes with your engine.** If you're already running Vulkan —
  game engine, real-time graphics, AR/VR, Android app, embedded GPU —
  valkyr lives on the same `VkDevice`, the same command buffers, the
  same queues. No parallel CUDA runtime, no shared-memory split, no
  extra GB of dynamic libraries. **The natural fit if you want
  inference inside an app that already has a Vulkan graphics stack.**

- **Pedagogically transparent.** Every GPU shader has a CPU reference
  in `src/cpu/*.zig` that gets parity-checked against. The full
  inference path is a few thousand lines of Zig you can read top to
  bottom — no decades of accretion to navigate. If you want to
  *understand* what a transformer kernel is doing, or modify one for
  research, this is a friendlier surface.

- **Zero lock-in, zero Python.** One Zig binary, no torch, no
  llama.cpp build system, no GGUF dependency for the basic path (we
  read safetensors + repack at upload time). `zig build` cross-compiles
  to most targets without extra toolchain. Drop into an embedded
  device, a CI runner, or a single static binary deployment without
  dragging a Python stack.

- **Modern architectures, built clean.** Qwen3.5 hybrid (Gated
  DeltaNet + full-attention), TurboQuant Q4 KV cache, llama.cpp-
  compatible Q4_0 and Q4_K_M weights — all built fresh from CPU
  references, not bolted onto an older core. The architectural
  diversity stays legible because nothing's grandfathered.

- **Training is on the menu.** TRiP ships paired `*_backward`
  functions for every primitive in `reference/math.c` — built-in
  gradient oracles. The plan is an Unsloth-class training port on top
  of the same Vulkan kernels, with bit-close parity vs. TRiP's CPU
  reference. Not yet shipped (see roadmap), but the architecture
  supports it.

**Honest framing.** valkyr is younger and smaller than llama.cpp. On
raw decode tok/s on a single CUDA card, llama.cpp's CUDA path is
faster than ours today (~1.5× on Qwen3.6 27B with `--q4k` last we
measured). What valkyr offers is **reach** (every Vulkan GPU),
**cleanliness** (CPU oracles for every kernel), and **composability**
(lives inside your existing Vulkan app). If you need maximum
throughput on a single NVIDIA box, llama.cpp is the right tool. If you
want one inference engine that runs everywhere your game or app
already runs, valkyr is.

## What works today

- **Six model families end-to-end on GPU**:
  - **Gemma 1** (`google/gemma-2b-it`): 18-layer transformer, multi-
    query attention, GeGLU FFN, RoPE, RMSNorm with `(1+w)`, tied LM
    head, SentencePiece tokenizer, `<bos>`/`<start_of_turn>` chat
    template.
  - **Llama 3 / Llama 3.2** (`meta-llama/Llama-3.2-3B-Instruct`):
    28-layer transformer, 3:1 grouped-query attention (24 heads /
    8 KV heads × head_dim 128), SwiGLU FFN, RoPE (θ=500K), plain
    RMSNorm, tied LM head, byte-level BPE tokenizer (vocab 128 K),
    Llama 3 header-id chat template
    (`<|start_header_id|>` / `<|end_header_id|>` / `<|eot_id|>`).
    `rope_scaling` (yarn-style) is silently ignored — fine for the
    8K trained context, would need work for the full 128K.
  - **Llama 2-arch chat fine-tunes** (`TinyLlama/TinyLlama-1.1B-Chat-v1.0`
    and analogous): same Llama-arch loader (SiLU, plain RMSNorm,
    untied lm_head, Llama-style RoPE, GQA), 32 K SentencePiece vocab,
    with the auto-detected Zephyr-style chat template
    (`<s><|user|>\n{msg}</s>\n<|assistant|>\n` — markers as text,
    not specials).
  - **Mistral 7B v0.3** (`mistralai/Mistral-7B-Instruct-v0.3`):
    `MistralForCausalLM` routes through `Family.llama` (same SiLU /
    plain-RMSNorm / GQA / untied-lm_head shape). 32 layers, 4:1 GQA
    (32 heads / 8 KV × head_dim 128), rope_theta 1 M, no sliding-
    window in v0.3. Chat template auto-detects via `[INST]` /
    `[/INST]` specials and emits `<s>[INST]{msg}[/INST]`. Multi-
    turn re-emits `</s>` between turns (the chat loop breaks on
    EOS without writing it to KV). Note: Ministral 3 (2025) is
    FP8 + multimodal — separate future arc, not the v0.3 path.
  - **Qwen3** (`Qwen/Qwen3-4B-Instruct-2507`): 36-layer transformer,
    4:1 grouped-query attention, SwiGLU FFN, RoPE (θ=5M), plain
    RMSNorm, per-head q_norm/k_norm before RoPE, tied LM head, byte-
    level BPE tokenizer, ChatML (`<|im_start|>` / `<|im_end|>`)
    template.
  - **Qwen3.5** (`Qwen/Qwen3.5-{0.8B,2B,4B}`): 24- to 32-layer
    **hybrid Gated DeltaNet + full-attention** model. Three out of
    every four layers are linear-attention (Gated DeltaNet, Yang et
    al. 2024 — discounted linear-attn with delta-rule writes); every
    fourth is a full-attention block with a 2×-wide `q_proj`
    `attn_output_gate` and **partial RoPE** (rotary_dim = head_dim /
    4). Per-head `q_norm` / `k_norm` use the `(1 + w)` form. Hybrid
    state plumbing is two flavors: full layers grow a standard KV
    cache; linear layers carry a constant-size `(conv_state,
    recurrent_state)` regardless of context length. Same ChatML
    tokenizer family as Qwen3 but with a different vocab (248320
    tokens, includes vision and tool-call specials we ignore). The
    multimodal vision tower and the multi-token-prediction head are
    skipped at load time.
  - **Qwen3.6** (`Qwen/Qwen3.6-27B`): 64-layer (48 linear / 16 full)
    retrained Qwen3.5 — declares the same
    `Qwen3_5ForConditionalGeneration` architecture, so it loads
    through the existing hybrid path with **no architecture code
    changes**. The new config keys (`mtp_*` for the multi-token-
    prediction draft head, `output_gate_type: "swish"`) pass through
    the JSON parser without incident. With `--q4` the 27B fits a
    single 24 GiB consumer card at ~15 tok/s greedy.
- **End-to-end forward pass on GPU** — embed → N × (rmsnorm → Q/K/V →
  [q_norm/k_norm if present] → RoPE → attention with KV cache → o_proj
  → residual → rmsnorm → gated FFN → residual) → final norm → LM head
  → logits. Family-specific differences (RMSNorm style, FFN
  activation, embedding scale, q_norm/k_norm) are gated cleanly so
  Gemma and Qwen3 share the same recorder. The Qwen3.5 hybrid path
  uses a parallel recorder with nine extra GLSL shaders for the Gated
  DeltaNet primitives (causal `conv1d_update`, `l2norm_per_head`,
  `gated_delta_step`, `rmsnorm_gated`, `rope_partial`, `split_q_gate`,
  `sigmoid_mul`, `slice_copy`, `scale`).
- **Multi-turn chat** with prompt prefill, KV cache persistence across
  turns, family-dispatched chat templates, and per-family stop tokens.
- **TurboQuant TQ4 V-cache** on all three families — opt in with
  `--tq4v`. ~5.5× memory reduction on the V cache. Block sizes 256
  (Gemma + Qwen3.5 4B) and 128 (Qwen3 4B) share the same CPU oracle
  and ship paired multi-block GLSL shaders. On the Qwen3.5 hybrid
  the V-cache compression applies on the 1-in-4 full-attention
  layers; the linear-attention layers carry a constant-size SSM
  state that already doesn't grow with context.
- **bf16 matmul weights on device** — halves upload time and matmul
  memory bandwidth versus fp32. Whenever the `bf16_matmul` or
  `q4_0_matmul` precision is active, embed_tokens and lm_head also
  ride bf16 (they're the largest single weight reads in the model;
  Q4_0 is deliberately not used for them since logits are sampled
  from). On Qwen3.5 4B with `--q4` this lifts decode from ~30 tok/s
  to ~52 tok/s; on Qwen3.6 27B halving the embed_tokens staging-
  buffer peak is what makes the model fit a 24 GiB card at all.
- **Q4_0 matmul weights on device** (`--q4`) — 4-bit signed
  block-32 weights with one fp16 scale per block, dequant-on-the-fly
  in the matmul shader. Quantization happens at upload time directly
  from the bf16/fp32 safetensors (no GGUF dependency); the CPU
  oracle is byte-clean against llama.cpp's `block_q4_0`. Drops the
  per-layer weight footprint to 0.625 B/elem — the path that lets
  27B-class models actually fit on a 24 GiB card.
- **Q4_K_M matmul weights on device** (`--q4k`) — llama.cpp-compatible
  super-block-256 with 8 sub-blocks of 32, each carrying its own
  6-bit scale and 6-bit min, plus two fp16 super-scales per super-
  block. Asymmetric dequant `d * sc * q − dmin * m` plus
  `make_qkx2_quants` iterative fit — ~32% lower MSE than Q4_0 on
  unit-Gaussian oracle round-trip, comparable visible chat quality.
  4.5 bits/elem on device (vs. ~5 for our padded Q4_0), so ~10%
  smaller and matches GGUF Q4_K_M's on-device cost.
  **Beats Q4_0 on decode tok/s across every model** (0.8B +10%,
  Gemma +11%, 4B +13%, 27B +29%) thanks to an 8-wide-K matmul
  shader: each iteration handles 8 consecutive K elements that
  always share one sub-block, so the heavy `(d, dmin, sc, m)` decode
  amortizes 8× and the qs reads collapse to 2 u32 / 8 nibbles.
  Win scales with model size — bigger models have more matmul time
  to amortize over. **Upload is ~4–6× slower than Q4_0** because of
  the iterative refinement (21 candidate iscales × 8 sub-blocks per
  super-block); GPU-side compute quantize would close that.
- **Lazy LM head during prefill** — chat skips the `vocab × hidden`
  LM head matmul on every prompt token except the last. Pure
  time-to-first-token win: scales with prompt length, no cost to
  decode tok/s.
- **Parallel weight upload** — the bf16 → fp32 conversion and Q4_0
  row-wise quantize run on a vendored Chase-Lev work-stealing pool
  ([credit: matryoshka/jobs.zig](https://github.com/foundation42/matryoshka)),
  worker count auto-set to `CPU - 2`. Combined with the
  single-suballocator weight pool below, Qwen3.5 4B Q4_0 upload
  drops from 24.1 s serial → 13.5 s parallel → 6.9 s pooled.
- **Single-suballocator weight pool** — every weight tensor's
  VkBuffer binds into one big `VkDeviceMemory` (sized by an
  upload-time pre-pass), and all per-tensor copies stage through
  one persistent `HOST_VISIBLE` buffer with a shared command
  buffer. Cuts ~640 per-tensor `vkAllocateMemory` /
  `submitOneShot` / `vkQueueWaitIdle` cycles down to a handful
  of flushes. Wins are size-dependent — smaller models that were
  per-tensor-overhead-bound see 5×+ on upload (Qwen3.5 0.8B bf16
  3.3 s → 0.6 s), bigger models gain less because the host-side
  Q4_0 quantize starts dominating (Qwen3.6 27B `--q4` 84.3 s →
  74.7 s, ~13%).
- **Sampling**: greedy, temperature, top-K, top-P, with `--seed` for
  reproducible sampled runs. (Greedy decoding is non-deterministic
  across runs at the last digit due to GPU subgroup reduction order;
  the CPU oracle is bit-deterministic.)
- **HuggingFace tokenizer** — auto-detects SentencePiece (Gemma) vs
  GPT-2-style byte-level BPE (Qwen3 / Qwen3.5 / Llama 3) at load
  time. Both encode paths verified bit-exact against the HF
  `tokenizers` reference on every prompt we've thrown at them.

## How fast it goes

RTX 3090 / Vulkan 1.3 / Zig 0.14-dev / ReleaseFast, all decode-only
greedy at `--temp 0`:

| model | precision | tok/s |
|---|---|---|
| Gemma 2B IT | bf16 (layers) + fp32 (embed/lm_head) | ~143 |
| Llama 3.2 3B Instruct | bf16 | ~71 |
| Llama 3.2 3B Instruct | `--q4k` | ~89 |
| TinyLlama 1.1B Chat | bf16 | ~171 |
| Mistral 7B Instruct v0.3 | `--q4k` | ~71 |
| Qwen3 4B Instruct 2507 | bf16 + bf16 lm_head | ~55 |
| Qwen3.5 0.8B | bf16 | ~113 |
| Qwen3.5 0.8B | `--q4` | ~104 |
| Qwen3.5 0.8B | `--q4k` | ~94 |
| Qwen3.5 4B | bf16 | ~55 |
| Qwen3.5 4B | `--q4` | ~59 |
| Qwen3.5 4B | `--q4k` | ~60 |
| **Qwen3.6 27B** | `--q4` | **~16.5** |
| **Qwen3.6 27B** | `--q4k` | **~20.6** |

bf16 matmul reads use vectorized `uvec4` (16-byte) loads — 4× fewer
SSBO transactions on the weight side, decoded inline as eight bf16
values per request. That's the headline 1.10–1.27× win over the
last reported numbers; see `shaders/matmul_nt_v2_bf16.comp`.

All numbers measured at 96-token decode horizon. The Qwen3.6 27B
figure is on the same 3090 the smaller models run on. `--tq4v` is
**deliberately tok/s-neutral at short context** — at 96 tokens the
V cache is tiny compared to the weight reads, so the per-step
dequant pass costs about the same as the bandwidth savings on the
attention-output kernel. `--tq4v`'s real wins land at long context
(4K–8K+) where the V history starts dominating attention-output
bandwidth, and as **steady memory savings** on the full-attention
V cache (~5.5× compression) — relevant if you want to push the
27B's effective context further on a 24 GiB card.

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
HuggingFace transformers (Python, fp32)         Gemma:    argmax = 229711  (▁increa)
   │                                            Qwen3:    argmax =    220  (' ')
   │                                            Qwen3.5:  argmax =    266  ('at')
   ▼
CPU forward (Zig, fp32, lazy bf16 conv)         Gemma:    ✓ matches HF, max |Δ| < 1e-3
                                                Qwen3:    ✓ matches HF, max |Δ| < 1e-3
                                                Qwen3.5:  ✓ matches HF, max |Δ| ≤ 1e-4
   │
   ▼
GPU forward (Vulkan SPIR-V, family-gated)       Gemma:    ✓ matches CPU
                                                Qwen3:    ✓ matches CPU
                                                Qwen3.5:  ✓ matches CPU, max |Δ| ≤ 3e-4
   │
   ▼
GPU TurboQuant V-cache (Vulkan + TQ4)           Gemma (256-block): top-5 match
                                                Qwen3 (128-block): coherent generation
                                                Qwen3.5 (256-block, 1-in-4 layers):
                                                  coherent generation — linear-attn
                                                  layers carry a fixed-size SSM
                                                  state and are unaffected.
```

The Qwen3 parity result is dramatic: HF top-5 logits are
`[8.2430, 8.0167, 7.9441, 7.8595, 7.6558]` for token ids
`[220, 151643, 334, 198, 13]`; Zig CPU lands
`[8.2430, 8.0166, 7.9441, 7.8595, 7.6558]` on the same ids. Three of
the five logits match to four decimal places after 36 transformer
layers, q_norm/k_norm, SwiGLU, and the byte-level BPE tokenizer. Run
it yourself: `python3 scripts/cross_validate_qwen3.py`.

Qwen3.5 (24- or 32-layer hybrid) is the harder result: per-layer
parity vs HF on layer 0 (Gated DeltaNet) lands at max |Δ| = 1.86e-8
(single fp32 ULP); on layer 3 (full-attention with attn_output_gate +
partial RoPE) max |Δ| = 2.25e-7; end-to-end the GPU↔CPU diff over the
248320-element vocab is max |Δ| ≤ 3e-4 — fp32 noise. Run it yourself:
`python3 scripts/cross_validate_qwen35.py` and
`./valkyr --gpu-gen-qwen35 <Qwen3.5 dir> 248044`.

For TurboQuant specifically, the 256-block pack kernel is **256/256
indices bit-exact** versus both the CPU oracle and the YATQ Python
reference; the norm-correction γ falls within fp16 ULP. The 128-block
variant uses the same algorithm at half the dimension and ships
end-to-end on Qwen3-4B with `--tq4v`.

### A note on greedy determinism

Greedy decoding under `--temp 0` produces argmax-stable output but is
*not bit-identical* across repeated GPU runs of the same prompt. The
cause is `subgroupAdd` reduction order in cooperative-K matmul / norm
kernels — last-token logits land within ~1e-7 of each other, which is
enough for the very last decimal of `argmax` to flip on creative
prompts where multiple candidates have nearly equal scores. The CPU
oracle is fully deterministic. Use `--seed` for reproducible *sampled*
generation; pin to CPU if you need bit-deterministic greedy output.

## Probes

valkyr's chat decode loop exposes optional **probe hooks** at six
points per token: token start/end, layer entry/exit, attention, and
logits. Hooks fire only when a probe wants them — empty-bus path is
bit-identical to the un-probed forward, parity-verified.

```sh
$ ./zig-out/bin/valkyr --chat meta-llama/Llama-3.2-1B-Instruct --q4k \
    --probe trace.jsonl "What is the capital of France?"
```

This streams a self-describing JSONL trace: a header record naming
the model + active probes + per-field definitions, then one record per
(token, layer) for activation observations and one per token for
logit observations. Trivially analyzable in Python — no Arrow / Parquet
dependency.

**v0 ships two probes:**

- **Activation entropy** (per layer per token): Shannon entropy of the
  L2-energy distribution `p_i = a_i^2 / Σ a_j^2` over the residual
  stream, plus the L2 norm. Captures how spread-vs-concentrated the
  activation is across feature dimensions, layer by layer.
- **Logit entropy + null-prior KL** (per token): conditional entropy
  H(t | context) of the softmaxed logits, and KL divergence from a
  null prior — the logit distribution the model emits for a single
  BOS token at position 0, computed once at startup. KL measures how
  far the prompt has driven the model away from baseline.

The probe interface is dual-purpose by design: a probe that *consumes*
a hidden state and hands it to a host system is structurally identical
to a probe that *measures* it. Same vtable, same hook points, same
data shapes — useful for interpretability, debug tooling, and engine
integration alike.

**Cost:** the un-probed path is unchanged; probed runs currently
trigger one GPU submit per layer (host readback between) so decode
tok/s drops ~3×. Opt-in, only paid when `--probe` is set.

Wired today on Llama 3.2 family (1B / 3B / 8B); hybrid Qwen3.5/3.6
path is gated with a clear message and lands in a later chunk.

## Hardware

- **Any Vulkan 1.3 GPU** — AMD GCN / RDNA, Intel Iris/Arc, NVIDIA
  Maxwell+, Apple Silicon via MoltenVK, Android (Adreno / Mali /
  PowerVR — within their device limits). Subgroup operations are
  required for the reduction kernels.
- The headline numbers above are on an NVIDIA RTX 3090 (24 GiB VRAM).
  Gemma 2B IT in bf16 needs ~5 GiB of weights and a few hundred MiB
  of KV cache — comfortable on most modern dGPUs. Qwen3.5 4B fits
  comfortably in either bf16 (~8 GiB) or `--q4` (~2.5 GiB). Qwen3.6
  27B at bf16 (~56 GiB) overflows 24 GiB by 2×; with `--q4`
  (~16 GiB on-device, ~17 GiB peak with KV) it fits a single 3090.

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
├── jobs.zig             Chase-Lev work-stealing pool (vendored from
│                        matryoshka/jobs.zig) — parallel weight upload
├── hf_cache.zig         HuggingFace cache walker — resolves "org/name"
│                        to snapshot paths, lists cached models for --list
├── cpu/
│   ├── math.zig         Reference primitives: rmsnorm, matmul_nt,
│   │                    RoPE, softmax, GeGLU, embedding lookup
│   ├── forward.zig      CPU full forward + sample (greedy/temp/top-k/top-p)
│   │                    + forwardTq4V (TQ4 V-cache variant)
│   ├── turboquant.zig   Lloyd-Max codebooks + RHT (FWHT + sign flips)
│   │                    + BlockTQ4 + quantize/dequantize — the oracle
│   ├── q4_0.zig         Llama.cpp-compatible Q4_0 quantize/dequantize
│   │                    (block-32, fp16 scale, signed nibbles); parity
│   │                    oracle for the GPU matmul shader
│   └── q4_k.zig         Llama.cpp-compatible Q4_K_M quantize/dequantize
│                        (super-block-256, 6-bit per-sub-block scale+min,
│                        asymmetric); parity oracle for matmul_nt_v2_q4_k
└── gpu/
    ├── vk.zig           Headless Vulkan compute context
    ├── buffer.zig       Static / dynamic / device-only buffer abstraction
    ├── pipeline.zig     Compute pipeline + descriptor set wrapper
    ├── recorder.zig     One-command-buffer dispatch batcher (auto barriers)
    ├── model.zig        GpuModel + Precision (fp32_all | bf16_matmul |
                         q4_0_matmul) + parallel upload via jobs.zig
    └── scratch.zig      Per-forward activation buffers + KV cache
                         (fp32 GpuKvCache + TurboQuant GpuKvCacheTq4)

shaders/
├── matmul_nt.comp           naive 1-thread-per-cell baseline
├── matmul_nt_v2.comp        cooperative-K reduction, fp32 weights
├── matmul_nt_v2_bf16.comp   bf16 weights, packed u32 reads
├── matmul_nt_v2_q4_0.comp   Q4_0 weights, in-shader dequant
│                            (block-32, fp16 scale, signed nibbles)
├── matmul_nt_v2_q4_k.comp   Q4_K_M weights, in-shader dequant
│                            (super-block-256, 6-bit sub-scales+mins,
│                            asymmetric d*sc*q − dmin*m)
├── embed_lookup_bf16.comp   bf16 embed_tokens (halves upload-peak +
│                            on-device footprint of vocab × hidden)
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

- **Gemma 1, Llama 3 / Llama 2-arch chat fine-tunes, Mistral 7B v0.3,
  Qwen3, Qwen3.5, and (architecturally) Qwen3.6 today.** Qwen3.6 declares the same
  `Qwen3_5ForConditionalGeneration` architecture as Qwen3.5 — it's a
  retrained 3.5, not a new family — so it loads through the existing
  hybrid path with no code changes. The new config keys (`mtp_*` for
  the speculative-decode draft head, `output_gate_type: "swish"`)
  are silently accepted by the JSON parser, the multi-token-
  prediction head is skipped, and the multimodal vision tower is
  skipped at load time. With `--q4` the 27B fits a 24 GiB consumer
  card. Llama 3.2 3B Instruct loads end-to-end through `Family.llama`
  and the Llama 3 chat template; TinyLlama-style Llama 2-arch chat
  fine-tunes are auto-detected via the absence of
  `<|start_header_id|>` and routed through a Zephyr-format chat
  composer (`<s><|user|>\n{msg}</s>\n<|assistant|>\n`). Llama 3.x's
  `rope_scaling` (yarn) is silently ignored — fine for the 8 K
  trained context, would need a parser pass for the full 128 K.
  Mistral 7B Instruct v0.3 ships through the same `Family.llama`
  with `[INST]/[/INST]` chat-format auto-detect — 7 B fits the 24 GiB
  card cleanly via `--q4k` (~71 tok/s). Gemma 2 / 3 (sliding-window
  attention), Ministral 3 (FP8 + multimodal), and the Qwen3.5/3.6
  vision tower are larger lifts still on the menu.
- The naive `matmul_nt_v2` kernel hits roughly 0.1% of the 3090's fp32
  peak. Most of the warm forward time is the FFN matmuls reading
  weights memory-bandwidth-bound — proper shared-memory tiling and
  fused attention (FlashAttention-style) are obvious wins. The
  Qwen3.5 chat path runs through `matmul_nt_v2_bf16` (~55 tok/s on
  the 4B, ~113 tok/s on the 0.8B) by default,
  `matmul_nt_v2_q4_0` with `--q4` (~59 tok/s on the 4B, ~104 tok/s
  on the 0.8B), and `matmul_nt_v2_q4_k` with `--q4k` (~60 tok/s on
  the 4B, ~94 tok/s on the 0.8B). When any
  non-fp32 path is active the lm_head and embed_tokens both ride
  bf16 — they're the single biggest matmul reads in the model
  (`N = vocab_size`, up to 248 K) and bf16 is the simplest safe
  halving (Q4_0 / Q4_K here would risk shifting the sample argmax).
  On Qwen3.6 27B that bf16 demotion is what makes the model fit a
  24 GiB card at all: the staging-buffer peak during embed_tokens
  upload halves from ~10 GiB to ~5 GiB.
- TurboQuant TQ4 ships two block sizes (256 for Gemma + Qwen3.5,
  128 for Qwen3); other head dims need a new shader pair. The
  Qwen3.5 hybrid path supports `--tq4v` on its full-attention layers
  (1-in-4 layers under the linear-attn × 3, full-attn × 1 schedule);
  linear-attn layers have a fixed-size SSM state with no growing KV
  cache, so TQ4 doesn't apply to them.
- TurboQuant TQ3 (3-bit) is not yet implemented.
- TurboQuant K-side compression (symmetric K=TQ / V=TQ) is not yet
  implemented.
- Single-stream batching only (M = 1 in every matmul).
- Tokenizer encoder doesn't pre-protect special-token strings.

## What we tried that didn't pan out

- **Tiled-N matmul** (N_TILE = 4 then 2). On top of the vectorized
  `uvec4` B reads, we tried packing 4 — then 2 — output cells per
  workgroup so A is loaded once and reused across cells, and the
  workgroup count drops by N_TILE. **N_TILE = 4 actually regressed
  Qwen3.5 4B by ~3%.** The per-iteration MAC count grew faster
  (8 → 32) than the per-iteration byte count (48 B → 96 B), pushing
  the K-loop into compute-bound territory on shapes where memory
  still had headroom. **N_TILE = 2 was flat on 4B / 27B / Gemma**
  and only +9% on the 0.8B (likely from the halved dispatch
  count, not from any A-reuse). With chunk-1 already capturing the
  big win on memory bandwidth, threading a tile factor through 50
  call sites for +9% on one model wasn't the trade. Reverted, kept
  the unitilted vectorized shader. Future angles: real shared-
  memory cooperative A loading, or just dropping WG size 256 → 128
  to double concurrent occupancy without shader gymnastics.
  See commit history + `project_roadmap.md` for the bench numbers.

- **Q4_0 split layout** (separate `indices ‖ scales` regions for
  uvec4-aligned reads). The chunk-1 trick that won big on bf16 (4×
  fewer SSBO transactions via uvec4 loads) doesn't translate to the
  Q4_0 path. **Regressed 27B `--q4` from 16.5 → 12.0 tok/s (~27%)**
  even though the per-element compute was unchanged and indices
  reads now hit a clean 16-byte transaction granularity. **Cause:**
  the OLD 5-u32-per-block contiguous layout puts indices + scale in
  the *same 128-byte cache line* (~6 blocks fit per line); splitting
  them into separate memory regions doubles cache-line traffic per
  block (one line for indices, one for scales). NVIDIA's warp-level
  coalescer was already getting near-peak bandwidth out of the
  scalar-uint reads on the contiguous layout — the "vectorized"
  16-byte read isn't a win when the bytes are already flowing
  through the cache hierarchy efficiently. Reverted. The roadmap's
  remaining FFN bandwidth headroom on Q4_0 has to come from
  something other than load-width vectorization — either a
  fundamentally different layout (Q4_K_M's super-blocks have better
  locality), or a fused norm + matmul that reduces the per-token
  reads. See `project_roadmap.md`.

- **SwiGLU activation-sparsity skipping on `down_proj`**. Tried two
  variants of "skip K-blocks where the post-SwiGLU activation is
  near zero" (the trick DejaVu / PowerInfer / MoE-as-skipping
  exploit). **Variant 1**: per-K-block mask + `if (mask[blk]==0)
  continue` in the matmul shader. Regressed 27B `--q4` 16.5 → 12.0
  (~−27%) because the in-loop branch added ~30% per-K overhead on
  no-skip elements, and we never got skip rates high enough at
  safe thresholds to overcome it. **Variant 2**: compact the
  active block indices into a list (atomic-compact in a single
  workgroup, self-resetting via `barrier()`), have the matmul
  iterate the list (no per-K branch). Got us much closer — 27B
  16.0 vs baseline 16.5 (~−3%). The remaining gap is the
  per-layer compact-dispatch overhead (~50–100 µs × 64 layers =
  6 ms/token), which roughly equals the savings from a 30% skip
  rate on `down_proj`. Pushing threshold to 1e-1 finally crossed
  break-even on 27B (+0.6%) but broke Qwen3.5 4B's output (1
  token then EOS — quality cliff). No threshold worked on every
  model. Reverted both. **Lesson:** NVIDIA's q4 matmul is already
  near throughput-bound on these shapes; any pre-pass that adds
  dispatch overhead has to clear it with savings *before*
  quality starts to wobble, and that window is too narrow.

## Where this can go next

The two big arcs:

1. **Breadth.** Mistral support (different `[INST]` chat template
   + new family enum) and Gemma 2 / 3 (sliding-window attention)
   are the next architectural lifts. Llama 2 / 3 already ship via
   the existing `Family.llama` enum + SwiGLU shader path. Phi /
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
   - **GPU-side compute quantize.** Q4_K_M upload is currently 4–6×
     slower than Q4_0 because `make_qkx2_quants` runs 21 candidate
     iscales × 8 sub-blocks per super-block on the CPU. Porting the
     quantize loop to a Vulkan shader would amortize across the GPU's
     thousands of cores — first-load time becomes near-IO-bound rather
     than CPU-bound.
   - **TQ4-on-weights** — TurboQuant applied to the matmul weight
     side, orthogonal to Q4_0/Q4_K. Combined with `--tq4v` gets
     Gemma 2B into a few hundred MiB total.
   - **Training port**, the Unsloth-alternative ambition. TRiP carries
     paired forward/backward in `reference/math.c` — every op has a
     `_backward` immediately below it, a built-in correctness oracle
     for every gradient.

## Acknowledgements

This was a real team effort:

- **Carlo Valenti** — for [TRiP](https://github.com/carlovalenti/TRiP) (Transformers in
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
