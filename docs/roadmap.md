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
     generic capital-listing. **Next tier:** Tier-4 = multi-batch
     training loop (rotate batches across dataset), real-model checkpoint
     stress (7.4 GB at Qwen3-0.6B fp32), and an Unsloth-equivalent
     fine-tune driver. See
     [embedding.md § Embedding training](embedding.md#embedding-training).
   - **Architectural blueprints (ONNX / JSON-spec).** Once the
     primitive set has stabilised through the transformer-training
     deepening above, a declarative graph → instantiated valkyr
     pipeline becomes the obvious next surface. Defer until the
     primitive library is roughly complete.
