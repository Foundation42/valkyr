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
