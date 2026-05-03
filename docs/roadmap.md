# Roadmap

The two big arcs ahead — breadth (more model families) and depth
(TQ3, symmetric K=TQ/V=TQ, fused FlashAttention-style kernel,
bf16 LM head + embedding, GPU-side compute quantize, TQ4-on-weights,
and a training port). Back to [README](../README.md). See also:
[limitations.md](limitations.md), [models.md](models.md).

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
   - **Training port**, the Unsloth-alternative ambition. Each forward
     primitive gets a paired `_backward` companion CPU-first as the
     gradient oracle, then ported to Vulkan. The architecture is built
     for it; what's left is the implementation pass.
