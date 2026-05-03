# Four-tier parity

Every layer in valkyr is parity-checked against the reference one tier
above it: HuggingFace `transformers` (fp32 Python) → CPU forward (Zig
fp32) → GPU forward (Vulkan SPIR-V) → GPU TurboQuant V-cache.
This page is the chain in detail, including the Qwen3 / Qwen3.5
numerical-drift figures and a note on greedy determinism. Back to
[README](../README.md). See also: [probes.md](probes.md),
[architecture.md](architecture.md).

## Four-tier parity

Every layer is parity-verified against the layer above. For TurboQuant
the chain extends one tier deeper:

```
HuggingFace transformers (Python, fp32)         Gemma:    argmax = 229711  (▁increa)
   │                                            Qwen3:    argmax =    220  (' ')
   │                                            Qwen3.5:  argmax =    266  ('at')
   ▼
CPU forward (Zig, fp32, lazy bf16 conv)         Gemma:    ✓ matches HF, max |Δ| < 1e-3
                                                Qwen3:    ✓ matches HF, max |Δ| < 1e-3
                                                Qwen3.5:  ✓ matches HF, max |Δ| ≤ 1e-4
   │
   ▼
GPU forward (Vulkan SPIR-V, family-gated)       Gemma:    ✓ matches CPU
                                                Qwen3:    ✓ matches CPU
                                                Qwen3.5:  ✓ matches CPU, max |Δ| ≤ 3e-4
   │
   ▼
GPU TurboQuant V-cache (Vulkan + TQ4)           Gemma (256-block): top-5 match
                                                Qwen3 (128-block): coherent generation
                                                Qwen3.5 (256-block, 1-in-4 layers):
                                                  coherent generation — linear-attn
                                                  layers carry a fixed-size SSM
                                                  state and are unaffected.
```

The Qwen3 parity result is dramatic: HF top-5 logits are
`[8.2430, 8.0167, 7.9441, 7.8595, 7.6558]` for token ids
`[220, 151643, 334, 198, 13]`; Zig CPU lands
`[8.2430, 8.0166, 7.9441, 7.8595, 7.6558]` on the same ids. Three of
the five logits match to four decimal places after 36 transformer
layers, q_norm/k_norm, SwiGLU, and the byte-level BPE tokenizer. Run
it yourself: `python3 scripts/cross_validate_qwen3.py`.

Qwen3.5 (24- or 32-layer hybrid) is the harder result: per-layer
parity vs HF on layer 0 (Gated DeltaNet) lands at max |Δ| = 1.86e-8
(single fp32 ULP); on layer 3 (full-attention with attn_output_gate +
partial RoPE) max |Δ| = 2.25e-7; end-to-end the GPU↔CPU diff over the
248320-element vocab is max |Δ| ≤ 3e-4 — fp32 noise. Run it yourself:
`python3 scripts/cross_validate_qwen35.py` and
`./valkyr --gpu-gen-qwen35 <Qwen3.5 dir> 248044`.

For TurboQuant specifically, the 256-block pack kernel is **256/256
indices bit-exact** versus both the CPU oracle and the YATQ Python
reference; the norm-correction γ falls within fp16 ULP. The 128-block
variant uses the same algorithm at half the dimension and ships
end-to-end on Qwen3-4B with `--tq4v`.

### A note on greedy determinism

Greedy decoding under `--temp 0` produces argmax-stable output but is
*not bit-identical* across repeated GPU runs of the same prompt. The
cause is `subgroupAdd` reduction order in cooperative-K matmul / norm
kernels — last-token logits land within ~1e-7 of each other, which is
enough for the very last decimal of `argmax` to flip on creative
prompts where multiple candidates have nearly equal scores. The CPU
oracle is fully deterministic. Use `--seed` for reproducible *sampled*
generation; pin to CPU if you need bit-deterministic greedy output.
