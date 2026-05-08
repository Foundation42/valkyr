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

`fa_dec` = FlashDecoding (Tri Dao 2023), the split-K decode-only
fork: phase 1 (`fa_decode_split`) shards K across `n_heads × n_splits`
workgroups and emits unnormalised partial `(O, m, l)` triples; phase 2
(`fa_decode_merge`) combines them with running-max + rescaled-sum
per head. n_splits picked per row to keep WG count saturating
(≥256 on the 3090).

| n_kv | scoresMB | scores | attn_out | 3-pass/L | fa_fwd/L | fa_dec/L | fa↑3-p | fd↑3-p |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 128 | 0.008 | 0.059 ms | 0.061 | 0.175 | 0.083 | 0.117 | 2.10× | 1.49× |
| 512 | 0.031 | 0.075 | 0.071 | 0.201 | 0.169 | 0.142 | 1.19× | 1.41× |
| 2048 | 0.125 | 0.136 | 0.128 | 0.320 | 0.552 | 0.181 | 0.58× | **1.77×** |
| 8192 | 0.500 | 0.377 | 0.324 | 0.764 | 2.042 | 0.279 | 0.37× | **2.74×** |
| 32768 | 2.000 | 1.401 | 3.491 | 4.993 | 7.410 | **0.704** | 0.67× | **7.09×** |

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

**Decode at long ctx: FlashDecoding takes the lead.** `fa_forward`
alone underperforms past `n_kv ≥ 2048` because it dispatches only
`n_heads = 16` workgroups, each iterating all K/V tiles serially —
under-parallel + work-bound. The 3-pass chain saturates 524288 WGs
at `n_kv=32768`, but at fragmentary per-WG work. **FlashDecoding
fixes both ends**: at `n_kv=32768` it runs `16 × 128 = 2048` phase-1
WGs (saturating the 3090's 82 SMs) over `split_size=256` keys each,
then merges the partials in a second `n_heads`-WG pass. **7.09×
speedup vs the 3-pass chain at ctx=32768** — 5.0 ms → 0.7 ms per
layer, which extrapolates to attention costing 19.7 ms/token instead
of the F1 baseline's 146 ms/token at the full Qwen3-0.6B 28-layer
stack.

The split-K kernels still need to be wired into `--chat`'s decode
path before the win is user-visible (currently chat decode lives
in `model.zig` / `runtime.zig`, separate from the training Runner).
Tier-2 follow-up.
