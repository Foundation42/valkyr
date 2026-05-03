# Quantization (TurboQuant + Q4_0 + Q4_K_M)

valkyr ships three quantization paths: TurboQuant TQ4 on the V-cache
(`--tq4v`), llama.cpp-compatible Q4_0 weights (`--q4`), and llama.cpp-
compatible Q4_K_M weights (`--q4k`). This page is the algorithmic
rationale and the try-it-yourself snippet. Back to
[README](../README.md). See also: [models.md](models.md),
[parity.md](parity.md), [perf.md](perf.md).

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

## Weight quantization

valkyr also ships two llama.cpp-compatible 4-bit weight paths:

- **`--q4`** — Q4_0: 4-bit signed block-32 weights with one fp16
  scale per block, dequant-on-the-fly in the matmul shader.
  Quantization happens at upload time directly from the bf16/fp32
  safetensors (no GGUF dependency); the CPU oracle is byte-clean
  against llama.cpp's `block_q4_0`. Drops the per-layer weight
  footprint to 0.625 B/elem — the path that lets 27B-class models
  actually fit on a 24 GiB card.

- **`--q4k`** — Q4_K_M: super-block-256 with 8 sub-blocks of 32,
  each carrying its own 6-bit scale and 6-bit min, plus two fp16
  super-scales per super-block. Asymmetric dequant
  `d * sc * q − dmin * m` plus `make_qkx2_quants` iterative fit —
  ~32% lower MSE than Q4_0 on unit-Gaussian oracle round-trip,
  comparable visible chat quality. 4.5 bits/elem on device (vs. ~5
  for our padded Q4_0), so ~10% smaller and matches GGUF Q4_K_M's
  on-device cost. **Beats Q4_0 on decode tok/s across every model**
  (0.8B +10%, Gemma +11%, 4B +13%, 27B +29%) thanks to an 8-wide-K
  matmul shader: each iteration handles 8 consecutive K elements
  that always share one sub-block, so the heavy `(d, dmin, sc, m)`
  decode amortizes 8× and the qs reads collapse to 2 u32 / 8
  nibbles. Win scales with model size — bigger models have more
  matmul time to amortize over. **Upload is ~4–6× slower than Q4_0**
  because of the iterative refinement (21 candidate iscales × 8
  sub-blocks per super-block); GPU-side compute quantize would
  close that.

`--q4` and `--q4k` are mutually exclusive. Either composes with
`--tq4v` for the smallest-footprint configuration. See
[models.md](models.md) for the full "what works today" surface and
[perf.md](perf.md) for tok/s numbers across the matrix.
