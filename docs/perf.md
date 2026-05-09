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

**Wired into `--chat` (F5b, 2026-05-08).** `runtime.recordOneLayer`
now branches between FlashDecoding (`fa_decode_split` +
`fa_decode_merge`) and the 3-pass chain on `cfg.head_dim ≤ 128` (=
`FA_HEAD_DIM_MAX`). The per-step `(n_splits, split_size)` heuristic
in `chooseFaDecodeSplit` mirrors the bench shapes: ≤ 4 keys → 1
split, &lt; 1024 → 4 splits, ≥ 1024 → `split_size = 256`. TQ4-V
composes cleanly — the same `dequant_v` buffer feeds either path.
Larger heads (head_dim > 256) auto-fall-back to 3-pass via the same
gate. End-to-end parity gate `runFaDecodeChatPathSmoke` runs both
paths through the production push values at Qwen3-0.6B shape and
compares to **4.42e-7 max rel-err** across n_kv ∈ {16, 64, 256,
1024} — five orders below the 1e-4 tolerance.

**FA-2 backward (F6, 2026-05-08).** `Runner.step` (the training
path) now dispatches the 3-kernel FA-2 backward chain in place of
the 5-kernel 3-pass:

  fa_bw_d   — `D[q, h] = Σ_d O · dO`            [n_q × n_heads WGs]
  fa_bw_dq  — per-(q, h) dQ accumulation        [n_q × n_heads WGs]
  fa_bw_dkv — per-(k, kv_h) dK + dV (GQA fold)  [n_kv × n_kv_heads WGs]

Same `attn_use_fa = head_dim ≤ FA_HEAD_DIM_MAX` gate as the forward.
Forward writes `LSE = m + log(l)` per
(q, h) so backward recomputes the softmax inline; **the
[n_pos × n_heads × n_pos] `buf_attn[i]` softmax matrix is never
materialised on the FA path** — the 7.2 GB scores roundtrip the F1
bench measured at `n_q=2048` evaporates from the forward, and the
training step's backward inherits the same memory profile. Per-layer
`buf_fa_lse[i]` + `buf_fa_d[i]` buffers (a few MB at typical
shapes) carry the saved values across layers. Real-model parity vs
the 3-pass on Qwen3-0.6B / n_pos=16: one-step CE 2.777821 → **1.280139**
(vs 1.280140 with 3-pass — 6-decimal match), 30-step run 99.98%
CE drop preserved, checkpoint round-trip bit-equal.

**head_dim=256 variants (D-arc, 2026-05-09).** A second SPIR-V build
of each head_dim-sensitive FA shader ships at `HEAD_DIM_MAX=256
BC=8`, lifting `FA_HEAD_DIM_MAX` from 128 to 256. Same .comp source
as the d=128 variants (BC=16) — only two preprocessor defines change
via `glslc -DHEAD_DIM_MAX=256 -DBC=8`. `BC=8` halves the per-tile
parallelism but keeps shared mem under AMD RDNA's 32 KB/WG ceiling
at d=256 (~18-20 KB total per shader). `fa_decode_merge` and
`fa_bw_d` don't size shared mem by head_dim, so their d=128 builds
serve d=256 dispatches unchanged. The dispatcher picks at pipeline-
init time via `runtime.faForwardSpv` / `faDecodeSplitSpv` /
`faBwDqSpv` / `faBwDkvSpv`. Brings:

  Gemma 2B IT      n_heads=8  n_kv_heads=1 d=256 onto the FA path
  Qwen3.5 0.8B/4B  n_heads=8/16 n_kv_heads=2/4 d=256 onto FA shaders
                   (the chat-path runtime_hybrid wiring still pending)

GPU parity vs CPU oracle on Qwen3.5 shape (n_q=8 n_kv=8 GQA 8:2 d=256
causal):

  fa_forward    max rel = 2.99e-7  (4 orders below 1e-4 gate)
  fa_decode     max rel = 6.75e-6  (split+merge vs fa_forward)
  fa_bw chain   max rel(dQ/dK/dV) = (2.11e-7, 4.21e-7, 3.03e-7)

Bench at Qwen3.5 0.8B per-layer (n_heads=8 n_kv_heads=2 d=256, RTX
3090, 24 layers):

| n_kv | scoresMB | scores | attn_out | 3-pass/L | fa_fwd/L | fa_dec/L | fa↑3-p | fd↑3-p |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 128 | 0.004 | 0.087 ms | 0.100 | 0.272 | 0.139 | 0.186 | 1.96× | 1.46× |
| 512 | 0.016 | 0.099 | 0.102 | 0.287 | 0.287 | 0.223 | 1.00× | 1.29× |
| 2048 | 0.063 | 0.128 | 0.162 | 0.386 | 0.969 | 0.290 | 0.40× | 1.33× |
| 8192 | 0.250 | 0.272 | 0.354 | 0.722 | 3.677 | 0.363 | 0.20× | **1.99×** |
| 32768 | 1.000 | 0.831 | 3.155 | 4.117 | 14.124 | **0.775** | 0.29× | **5.31×** |

Prefill causal (n_q == n_kv) at the same shape:

| n_q | scoresMB | scores_t | softmax | out_t | 3-pass/L | fa_fwd/L | speedup |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 16 | 0.008 | 0.082 ms | 0.083 | 0.148 | 0.313 | 0.096 | **3.27×** |
| 128 | 0.500 | 0.311 | 0.083 | 0.766 | 1.159 | 0.302 | **3.84×** |
| 512 | 8.000 | 4.260 | 0.108 | 7.550 | 11.918 | 3.376 | **3.53×** |
| 2048 | 128.000 | 65.538 | 0.489 | 128.462 | 194.489 | 57.145 | **3.40×** |

Slightly softer wins than d=128 (4-4.8× prefill, 7.09× decode-32k)
because BC=8 doubles the K-tile iteration count and n_heads=8 halves
the WG count per dispatch — but still a clean 3.3-3.8× prefill and
5.3× FlashDecoding at ctx=32k. The scoresMB column is also halved
because Qwen3.5 0.8B has 8 heads vs Qwen3-0.6B's 16, so the 3-pass
scores buffer roundtrip shrinks proportionally (1 GB vs 2 GB at
ctx=32k per layer).

**End-to-end on Gemma 2B IT** (`--bench --n 4096`, bf16 matmul,
3090): warm decode 8.74 ms/tok at pos 4080-4095 (`fa_decode` on the
d=256 SPIR-V), vs the projected 12-14 ms/tok the pre-D-arc 3-pass
chain would have run at d=256 long-ctx. ~1.4-1.6× speedup at long
context, on top of the existing per-token tok/s. (Short-ctx tok/s
unchanged at ~142 tok/s — the FA win is small at n_kv ≤ 128 where
matmul dominates, and matches the bench's 1.46× per-layer attention
saving in absolute ms.)

**Outstanding:** `runtime_hybrid.zig` (the Qwen3.5/3.6 hybrid path)
still routes its full-attention layers through the 3-pass chain.
Wiring `fa_decode_split` + `fa_decode_merge` into that module is
the next chunk to make Qwen3.5 0.8B/4B see the same long-ctx win
as Gemma 2B does today.
