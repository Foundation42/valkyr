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

## Attention bench (F1 baseline for FlashAttention)

`valkyr --attn-bench` times the standard 3-dispatch attention chain
(`attn_scores` → `softmax` → `attn_output`) at Qwen3-0.6B per-layer
shape (`n_heads=16, n_kv_heads=8, head_dim=128`). Two phases: decode
(`n_q=1`, no mask) and prefill-causal (`n_q == n_kv`). RTX 3090 /
ReleaseFast / Zig 0.14.1 / 5-iter average, includes `submitOneShot
+ waitIdle` overhead.

**Decode** (single-token forward, sweeping context length):

| n_kv | scoresMB | attn_scores | softmax | attn_output | total/L | ×28 layers |
|---:|---:|---:|---:|---:|---:|---:|
| 128 | 0.008 | 0.061 ms | 0.055 ms | 0.062 ms | 0.179 ms | **5.01 ms** |
| 512 | 0.031 | 0.074 | 0.055 | 0.071 | 0.200 | 5.60 |
| 2048 | 0.125 | 0.136 | 0.056 | 0.128 | 0.321 | 8.98 |
| 8192 | 0.500 | 0.376 | 0.061 | 0.327 | 0.764 | 21.38 |
| 32768 | 2.000 | 1.399 | 0.100 | 3.718 | 5.217 | **146.09 ms** |

**Prefill causal** (training/long-prompt forward, n_q == n_kv):

| n_q | scoresMB | scores_train | softmax | out_train | total/L | ×28 layers |
|---:|---:|---:|---:|---:|---:|---:|
| 16 | 0.016 | 0.064 ms | 0.055 ms | 0.131 ms | 0.250 ms | 6.99 ms |
| 128 | 1.000 | 0.564 | 0.069 | 0.892 | 1.525 | 42.71 |
| 512 | 16.000 | 8.409 | 0.105 | 7.448 | 15.962 | 446.93 |
| 2048 | 256.000 | 118.145 | 0.859 | 127.717 | 246.721 | **6908.19 ms** |

`scoresMB` is the per-layer fp32 scores buffer (`rows × n_kv × 4B`).
At decode `n_kv=32768` we shuttle 56 MB of scores per token across
all 28 layers; at prefill `n_q=2048` it's a 7.2 GB roundtrip per
forward — the buffer alone exceeds the entire `.lvkpt` checkpoint.

The cliff comes from two compounding factors:
1. **Quadratic scores buffer** in prefill (`n_q × n_kv`) — 4× n_q
   gives ~15× total time at the larger shapes.
2. **Scattered HBM reads** in `attn_output` for decode at long
   `n_kv` — the kernel reads `[n_heads, n_kv]` scores against
   `[n_kv, n_kv_heads, head_dim]` V, no spatial reuse.

**FlashAttention target.** Replace the 3-dispatch chain with a single
tiled kernel that keeps `O(Br × Bc)` softmax state in shared memory,
streams K/V blocks through registers, and never materialises the
full scores tensor. Realistic gains: 2–4× at prefill `n_q ≥ 512`
(eliminates the quadratic HBM roundtrip), modest at decode (saves
~1–2 ms/layer at `n_kv ≥ 8192`). Re-run `--attn-bench` post-FA to
fill in the right column.
