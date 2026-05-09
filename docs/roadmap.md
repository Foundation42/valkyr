# Roadmap

The two big arcs ahead — breadth (more model families) and depth
(TQ3, symmetric K=TQ/V=TQ, fused FlashAttention-style kernel,
bf16 LM head + embedding, GPU-side compute quantize, TQ4-on-weights,
deeper training stack). Training v0 (cooperative attach + Adam +
loss-target decay over a 2-layer MLP) is **shipped** as of
2026-05-05; the **transformer fine-tune arc** (real Qwen3-0.6B
weights, packed-stream dataset, real one-step Adam) is **shipped**
as of 2026-05-06. See
[embedding.md § Embedding training](embedding.md#embedding-training).
Back to [README](../README.md). See also: [limitations.md](limitations.md),
[models.md](models.md).

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
   - **Fused FlashAttention-style kernel.** **Whole F-arc (F1-F6c)
     shipped 2026-05-08.** Forward: `shaders/fa_forward.comp` (single
     fused kernel, BC=16, HEAD_DIM ≤ 128) wired into both
     `Runner.forwardLogits` (inference) and `Runner.step` (training)
     for **4-4.8× prefill speedup**; `shaders/fa_decode_split.comp` +
     `fa_decode_merge.comp` (Tri Dao FlashDecoding split-K) wired into
     `runtime.recordOneLayer` for chat decode at **7× speedup at
     ctx=32768** (per-layer attention 5.0 ms → 0.7 ms). Backward
     (F6): `shaders/fa_bw_{d,dq,dkv}.comp` — 3-kernel FA-2 chain
     (Dao 2023, Algorithm 4) replaces the 5-kernel 3-pass in
     `Runner.step`. Forward writes `LSE = m + log(l)` and backward
     recomputes the softmax inline, so the [n_pos × n_heads × n_pos]
     `buf_attn[i]` matrix never materialises (the 7.2 GB scores
     roundtrip the F1 bench measured at `n_q=2048` evaporates).
     Real-model parity gate on Qwen3-0.6B/n_pos=16: one-step CE
     2.777821 → **1.280139** (vs 1.280140 with 3-pass — 6-decimal
     match), 30-step 99.98% CE drop preserved, checkpoint round-trip
     bit-equal. TQ4-V-cache composes via the existing `dequant_v`
     buffer. **D-arc (head_dim=256, 2026-05-09) shipped:** second
     SPIR-V build of each head_dim-sensitive FA shader at BC=8
     HEAD_DIM_MAX=256 (~18-20 KB shared mem, fits AMD RDNA's 32 KB
     ceiling) lifts `FA_HEAD_DIM_MAX` from 128 to 256. Brings Gemma
     2B IT (head_dim=256) onto the FA path end-to-end —
     `--bench --n 4096` shows ~1.4-1.6× decode speedup at pos 4080
     (8.74 ms/tok vs the 12-14 ms/tok the 3-pass would have run).
     Qwen3.5 family inherits the kernels at parity (2-7e-7 rel-err
     across forward / decode / backward) but the chat-path wiring
     still goes through `runtime_hybrid.zig` — that module's
     full-attention layers are the next chunk to plumb. Outstanding
     fused-FA deepening: a kernel that consumes packed TQ4 K/V
     directly in the inner loop (skipping the `dequant_v`
     materialisation) for the bandwidth savings to land at long
     context.
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
   - **Training stack** — v0 **shipped**, Tier-1 **shipped**, **Tier-2
     transformer fine-tune arc shipped 2026-05-06**, **Tier-3 (multi-step
    + checkpoint save/load + sampled-text-shift validation) shipped
    2026-05-08**. The 2-layer-MLP
     surface (cooperative attach, batched mean-gradient SGD/Adam,
     MSE/CE loss heads, loss-target decay, host-mapped predict
     staging) lives at `src/train/runner.zig` with parity smokes
     against the CPU oracle in `src/cpu/train.zig`. Headless
     `valkyr --train-demo` and two visual companion demos in matryoshka
     (`train_mlp_demo`, `train_classifier_demo`) show it running at
     refresh rate inside a render loop. Tier-1 generalised the surface
     to depth N: `cpu_train.MlpN` + `train_runner_n.MlpNRunner`,
     decoupled L2 weight decay (AdamW form), `cosineLr` schedule
     helper. **Tier-2** built the full transformer-training
     primitive set on the same parity discipline (CPU oracle → GPU
     shader → smoke) — RMSNorm / LayerNorm backward, embedding
     gradient (vocab-major scatter), softmax backward, fused-attention
     forward + backward (Q/K/V projections + scores + RoPE bw),
     SwiGLU FFN forward + backward, per-head Q/K-RMSNorm — and
     composed them into a real Qwen3-class trainer. The β-3..β-5
     arc closes the loop: `train/load_real.zig` materialises real
     bf16 Qwen3-0.6B weights as fp32 train tensors, `train/dataset.zig`
     packs JSONL examples with EOS separators into (n_pos+1) windows,
     and `train_transformer.Runner.step` drives one Adam step that
     drops batch CE 2.78 → 1.28 nats in ~330 ms (lr=1e-5, n_pos=16,
     ReleaseFast). **Tier-3** closed the loop: multi-step training holds
     past step 1 (CE 2.78 → 0.0005 / 99.98% drop in 30 steps on a single
     batch); `.vkpt` checkpoint save/load round-trips with bit-equal
     trajectories at toy scale (new `Buffer.uploadFromHost` +
     `Runner.saveCheckpoint`/`loadCheckpoint` — header + Config + raw
     fp32 body for params + Adam m/v + step_t); and a sampled-text-shift
     smoke shows the post-train model literally regurgitating
     `tiny_facts.jsonl` batch 0 ("*Paris. Paris sits on the river Seine
     and is famous for...*") where the pre-train base model gives a
     generic capital-listing. **A-series (LoRA) shipped 2026-05-08:**
     Chronicals-paper-aligned LoRA + LoRA+ wired into `Runner.step`
     with a `lora_targets: u32` bitmask + `LoraTarget` constants
     (`q | k | v | o | gate | up | down | all_attn | all_ffn | all`),
     standard freeze semantics (every non-LoRA param frozen when LoRA
     is on), and a `.lvkpt` checkpoint format that drops the on-disk
     payload from ~9 GiB (`.vkpt`) to ~52.5 MiB on Qwen3-0.6B at
     `all_attn` rank-16 — a ~170× shrink. `--lora-finetune <model>
     --data <jsonl> --lora-targets <spec>` is the user-facing CLI;
     `--gen-from-ckpt` magic-sniffs `.vkpt` vs `.lvkpt`. LoRA-Q is
     actually faster than full-FT (256 vs 268 ms/step) because the
     freeze skips Adam on the huge embed + lm_head buffers.
     **`.lvkpt` into the chat path shipped same day:**
     `--chat <model> --lora-ckpt foo.lvkpt --lora-targets <spec>
     --lora-rank N --lora-alpha A` folds the LoRA delta into the
     base bf16 weights at load (`W' = W + (α/r)·B·A`, ~4 s one-time
     for `all_attn` rank-16 on Qwen3-0.6B), then runs the unmodified
     fast inference path. **~145 tok/s** vs `--gen-from-ckpt`'s
     ~9 tok/s — ~15× speedup, with bit-identical token throughput
     to the base model afterwards. The `.vkpt` (full-FT) equivalent
     still needs the positional-blob → safetensors-tensor-name
     mapping; LoRA covers the modern fine-tune workflow today.
     **Next tier:** Tier-4 = multi-batch training loop (rotate batches
     across dataset), real-model checkpoint stress, and an Unsloth-
     equivalent fine-tune driver. See
     [training.md § LoRA fine-tuning](training.md#lora-fine-tuning)
     and [embedding.md § Embedding training](embedding.md#embedding-training).
   - **Architectural blueprints (ONNX / JSON-spec).** Once the
     primitive set has stabilised through the transformer-training
     deepening above, a declarative graph → instantiated valkyr
     pipeline becomes the obvious next surface. Defer until the
     primitive library is roughly complete.
