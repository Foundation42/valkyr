# Limitations + things we tried that didn't pan out

The honest list of what valkyr can't do today (model coverage gaps,
matmul efficiency, batching, tokenizer edges) and the experiments that
got reverted (tiled-N matmul, Q4_0 split layout, SwiGLU activation-
sparsity skipping). Back to [README](../README.md). See also:
[roadmap.md](roadmap.md), [models.md](models.md), [perf.md](perf.md).

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
