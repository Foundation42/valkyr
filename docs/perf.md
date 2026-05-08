# Performance

Decode tok/s numbers on an RTX 3090 across the supported model + precision
matrix, plus a note on how the bf16 vectorized weight reads got there
and what `--tq4v` does (and doesn't) do for short-context throughput.
Back to [README](../README.md). See also: [models.md](models.md),
[quantization.md](quantization.md), [hardware.md](hardware.md).

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

## Attention bench (FlashAttention vs the 3-pass baseline)

`valkyr --attn-bench` times both attention paths side-by-side at
Qwen3-0.6B per-layer shape (`n_heads=16, n_kv_heads=8, head_dim=128`):

* **3-pass** = `attn_scores` → `softmax` → `attn_output` (or
  `_train` variants for prefill) — materialises the full
  `[n_q × n_heads × n_kv]` scores tensor.
* **fa_forward** = single `shaders/fa_forward.comp` dispatch —
  tile-on-K with online softmax in shared memory, never materialises
  the scores tensor.

RTX 3090 / ReleaseFast / Zig 0.14.1 / 5-iter average, includes
`submitOneShot + waitIdle` overhead per dispatch.

**Decode** (single-token, sweeping context length, no causal mask):

| n_kv | scoresMB | scores | softmax | attn_out | 3-pass/L | fa_fwd/L | speedup |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 128 | 0.008 | 0.060 ms | 0.056 | 0.062 | 0.178 | 0.085 | **2.09×** |
| 512 | 0.031 | 0.076 | 0.056 | 0.073 | 0.205 | 0.171 | 1.19× |
| 2048 | 0.125 | 0.138 | 0.057 | 0.128 | 0.323 | 0.553 | 0.58× |
| 8192 | 0.500 | 0.380 | 0.062 | 0.325 | 0.767 | 2.035 | 0.38× |
| 32768 | 2.000 | 1.461 | 0.100 | 3.646 | 5.206 | 8.015 | 0.65× |

**Prefill causal** (training / long-prompt, `n_q == n_kv`):

| n_q | scoresMB | scores_t | softmax | out_t | 3-pass/L | fa_fwd/L | speedup |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 16 | 0.016 | 0.065 ms | 0.055 | 0.130 | 0.250 | 0.061 | **4.10×** |
| 128 | 1.000 | 0.548 | 0.062 | 0.877 | 1.487 | 0.311 | **4.78×** |
| 512 | 16.000 | 7.929 | 0.101 | 7.480 | 15.510 | 3.546 | **4.37×** |
| 2048 | 256.000 | 120.069 | 0.846 | 128.663 | 249.577 | 60.871 | **4.10×** |

`scoresMB` is the per-layer fp32 scores buffer the 3-pass path
materialises (`rows × n_kv × 4B`). At prefill `n_q=2048` that's
**256 MB per layer × 28 = 7.2 GB shuttled per forward** — bigger
than any `.lvkpt` checkpoint we ship. `fa_forward` keeps the same
state in `O(Br × Bc) = O(BC)` shared memory and never round-trips
through HBM.

**Headline.** FlashAttention is a clean **4–4.8× win across all
prefill shapes** — full forward at `n_q=2048` drops from 6.9 s to
1.7 s on a 28-layer Qwen3-0.6B forward. This is the regime that
matters for training and for `--chat` long-prompt prefill.

**Decode beyond `n_kv ≥ 2048` is currently *slower* on FA**, despite
saving the scores roundtrip. Reason is parallelism: decode dispatches
only `n_heads = 16` workgroups (one per query-head), each iterating
all K/V tiles serially. The 3-pass `attn_scores` kernel dispatches
`n_heads × n_kv = 524288` WGs at `n_kv=32768` — saturating the SMs
even at fragmentary per-WG work. FA's tiled kernel becomes work-bound
per-WG once `n_kv` is large enough. The textbook fix is **split-K**
(FlashDecoding, Tri Dao 2023): partition the K dimension across
several WGs per `(q, h)` and merge their partial `(O, m, l)` triples
in a second pass. That's a future F-chunk; for now, the production
path keeps the 3-pass chain at decode and switches to FA only for
prefill / training.
