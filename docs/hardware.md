# Hardware

Vulkan-1.3-class GPU requirements and a note on what fits on a 24 GiB
card across the model + precision matrix. Back to
[README](../README.md). See also: [perf.md](perf.md),
[models.md](models.md), [quantization.md](quantization.md).

## Hardware

- **Any Vulkan 1.3 GPU** — AMD GCN / RDNA, Intel Iris/Arc, NVIDIA
  Maxwell+, Apple Silicon via MoltenVK, Android (Adreno / Mali /
  PowerVR — within their device limits). Subgroup operations are
  required for the reduction kernels.
- The headline numbers above are on an NVIDIA RTX 3090 (24 GiB VRAM).
  Gemma 2B IT in bf16 needs ~5 GiB of weights and a few hundred MiB
  of KV cache — comfortable on most modern dGPUs. Qwen3.5 4B fits
  comfortably in either bf16 (~8 GiB) or `--q4` (~2.5 GiB). Qwen3.6
  27B at bf16 (~56 GiB) overflows 24 GiB by 2×; with `--q4`
  (~16 GiB on-device, ~17 GiB peak with KV) it fits a single 3090.
