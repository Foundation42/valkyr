# Supported models

This is the per-family rundown of what valkyr loads and runs end-to-end
today — architecture, attention shape, FFN, RoPE flavor, tokenizer, and
chat template — plus the rest of the "what works today" surface
(forward pass, multi-turn chat, TurboQuant, weight precisions, sampling,
tokenizer). Back to [README](../README.md). See also:
[parity.md](parity.md), [perf.md](perf.md), [quantization.md](quantization.md).

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
