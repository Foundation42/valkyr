# Architecture

Code layout (`src/` and `shaders/`) plus the convention notes that the
implementation depends on — RoPE pair convention, Gemma quirks, the
TurboQuant Algorithm 1 choices, and observed numerical drift. Back to
[README](../README.md). See also: [parity.md](parity.md),
[quantization.md](quantization.md), [models.md](models.md).

## Architecture

```
src/
├── main.zig             CLI dispatch + per-mode orchestration
├── safetensors.zig      mmap-backed tensor reader (zero-copy)
├── sharded.zig          multi-file SafeTensors merge by-name
├── config.zig           HuggingFace config.json + Family enum
├── model.zig            CPU Model — config + tensors per layer
├── tokenizer.zig        BPE encode + decode, byte fallback
├── dtype.zig            bf16/fp16 → fp32 bit-twiddling
├── jobs.zig             Chase-Lev work-stealing pool (vendored from
│                        matryoshka/jobs.zig) — parallel weight upload
├── hf_cache.zig         HuggingFace cache walker — resolves "org/name"
│                        to snapshot paths, lists cached models for --list
├── cpu/
│   ├── math.zig         Reference primitives: rmsnorm, matmul_nt,
│   │                    RoPE, softmax, GeGLU, embedding lookup
│   ├── forward.zig      CPU full forward + sample (greedy/temp/top-k/top-p)
│   │                    + forwardTq4V (TQ4 V-cache variant)
│   ├── turboquant.zig   Lloyd-Max codebooks + RHT (FWHT + sign flips)
│   │                    + BlockTQ4 + quantize/dequantize — the oracle
│   ├── q4_0.zig         Llama.cpp-compatible Q4_0 quantize/dequantize
│   │                    (block-32, fp16 scale, signed nibbles); parity
│   │                    oracle for the GPU matmul shader
│   └── q4_k.zig         Llama.cpp-compatible Q4_K_M quantize/dequantize
│                        (super-block-256, 6-bit per-sub-block scale+min,
│                        asymmetric); parity oracle for matmul_nt_v2_q4_k
└── gpu/
    ├── vk.zig           Headless Vulkan compute context
    ├── buffer.zig       Static / dynamic / device-only buffer abstraction
    ├── pipeline.zig     Compute pipeline + descriptor set wrapper
    ├── recorder.zig     One-command-buffer dispatch batcher (auto barriers)
    ├── model.zig        GpuModel + Precision (fp32_all | bf16_matmul |
                         q4_0_matmul) + parallel upload via jobs.zig
    └── scratch.zig      Per-forward activation buffers + KV cache
                         (fp32 GpuKvCache + TurboQuant GpuKvCacheTq4)

shaders/
├── matmul_nt.comp           naive 1-thread-per-cell baseline
├── matmul_nt_v2.comp        cooperative-K reduction, fp32 weights
├── matmul_nt_v2_bf16.comp   bf16 weights, packed u32 reads
├── matmul_nt_v2_q4_0.comp   Q4_0 weights, in-shader dequant
│                            (block-32, fp16 scale, signed nibbles)
├── matmul_nt_v2_q4_k.comp   Q4_K_M weights, in-shader dequant
│                            (super-block-256, 6-bit sub-scales+mins,
│                            asymmetric d*sc*q − dmin*m)
├── embed_lookup_bf16.comp   bf16 embed_tokens (halves upload-peak +
│                            on-device footprint of vocab × hidden)
├── rmsnorm.comp             subgroup-reduced sum-of-squares + Gemma quirk
├── softmax.comp             stable two-pass with stride support
├── geglu.comp               fused gelu_tanh(gate) · up
├── rope.comp                half-split RoPE
├── attn_scores.comp         Q · K_cache, cooperative reduction
├── attn_output.comp         scores · V_cache, cooperative reduction
├── kv_write.comp            scatter K_rot/V into the per-layer cache
├── embed_lookup.comp        embedding row + Gemma sqrt(hidden) scale
├── add_in_place.comp        residual stream += contribution
├── vec_add.comp             smoke test
│
│ ── TurboQuant kernels ──
├── fwht256.comp             in-place 256-elem Fast Walsh-Hadamard
├── rht_pre256.comp          fused (sign-flips + FWHT) preconditioner
├── rht_post256.comp         inverse RHT (IFWHT + sign-flips)
├── tq4_pack256.comp         full TQ4 quantize: norm + RHT + Lloyd-Max
│                            + IFWHT + γ correction → packed block
├── tq4_unpack256.comp       inverse: centroid LUT + IFWHT + γ scale
└── tq4_pack_to_cache.comp   tq4_pack256 + push-constant dst block idx
                             (production V-write into the per-layer cache)
```

19 shaders in total. Every kernel has a CPU reference counterpart and
a parity test (synthetic input + a real-Gemma input where it makes
sense).

## Convention notes

- **RoPE**: HuggingFace half-split convention (pairs `(j, j+D/2)`).
- **Gemma quirks** the engine handles:
    - Embedding scaled by `sqrt(hidden_size)` before the first block.
    - RMSNorm gain is `(1 + weight)` rather than plain `weight`.
    - GeGLU FFN with `gelu_pytorch_tanh` activation despite the config
      claiming `"hidden_act": "gelu"`.
    - LM head tied to `embed_tokens` (no separate `lm_head.weight`).
- **TurboQuant**: Algorithm 1 only (no QJL residual); deterministic-
  seed Randomized Hadamard Transform (the 32-byte TBQ_SIGNS pattern
  matches llama.cpp `cpy-utils.cuh`); a single global Lloyd-Max
  codebook for b=4 (no per-block scales); norm-correction γ stored as
  fp16 per block. See `src/cpu/turboquant.zig` for the canonical code,
  and `scripts/cross_validate_turboquant.py` to regenerate the YATQ
  parity reference.
- **Numerical drift**: max |Δ| over the full 256000-element logit
  vector vs HF transformers is ~3.5e-4 on the fp32 path. With TQ4
  V-cache enabled, max |Δ| versus the fp32 GPU baseline is ~1.7
  absolute on `<bos>`, with all top-5 token IDs preserved.
