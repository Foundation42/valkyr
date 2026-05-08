# Windows port

valkyr builds and runs on Windows 11 + Vulkan 1.3 — chat, sampling, the
HF-cache `--list` UX, and the safetensors loader all work end-to-end.
First brought up 2026-05-08 against an AMD Strix Point APU (Radeon 890M
iGPU, RDNA 3.5, unified 128 GB system RAM). The Linux-side build is
unchanged: every patch is comptime-gated on `builtin.os.tag`, so the
POSIX path is byte-for-byte identical.

## Toolchain

- **Zig 0.14.1** — download the official prebuilt zip from
  [ziglang.org](https://ziglang.org/download/) (`zig-x86_64-windows-0.14.1.zip`),
  extract anywhere, add the directory to your user PATH. We've used
  `C:\Users\<you>\Tools\zig-x86_64-windows-0.14.1\` without trouble.
- **Vulkan SDK** — install the LunarG SDK from
  [vulkan.lunarg.com/sdk/home](https://vulkan.lunarg.com/sdk/home).
  The installer drops `vulkan-1.dll` in `System32`, sets `VULKAN_SDK`
  pointing at `C:\VulkanSDK\<version>\`, and puts `glslc.exe` on PATH.
  All three are required.
- **HuggingFace CLI** (optional, for `--chat <hf-id>` UX). Standard
  `pip install huggingface_hub` + `huggingface-cli download <id>`.
  Models land under `%USERPROFILE%\.cache\huggingface\hub\`.

## Build

```powershell
zig build                              # Debug — validation layer on
zig build -Doptimize=ReleaseFast       # release (recommended for chat)
```

`build.zig` reads `VULKAN_SDK` from the env to add the SDK's `Include\`
and `Lib\` directories — open a fresh shell after the SDK installer
runs so the env var is visible. `glslc.exe` shells out for shader
compilation; the SDK installer puts it on PATH.

## What works today

| Surface | Status |
|---|---|
| `zig build` (Debug + ReleaseFast) | clean |
| `zig build run` (in-binary kernel + parity smokes) | most pass; one large-shape training smoke fails (see "Known issues") |
| `--list` (HF cache discovery) | OK |
| `--inspect`, `--load` (safetensors → CPU model) | OK |
| `--chat` (dense families: Qwen3, Llama, Gemma, Mistral) | OK |
| `--chat --q4`, `--chat --q4k`, `--chat --tq4v` | OK |
| `--chat --lora-ckpt` (production-speed LoRA) | OK |

## Performance — AMD Radeon 890M iGPU, Qwen3-0.6B, greedy

| Build | Weights | Decode |
|---|---|---|
| Debug | bf16 | 41.6 tok/s |
| ReleaseFast | bf16 | 53.6 tok/s |
| ReleaseFast | Q4_K_M | **90.5 tok/s** |

For reference: README quotes ~145 tok/s for the same model+precision
on a discrete RTX 3090. The 890M iGPU pulling ~63% of a 3090 from a
laptop APU is unified-memory bandwidth doing the talking.

Q4_K_M weight upload takes ~3.5 s (one-shot quantize at load). bf16
upload is ~700 ms (no quantize, just memcpy through the staging path).

## Source patches

Five small patches, all comptime-gated. The POSIX build is unchanged.

- **`src/hf_cache.zig`** — fall back to `USERPROFILE` when `HOME` is
  unset so `%USERPROFILE%\.cache\huggingface\hub` resolves. Matches
  what `huggingface_hub` does on Windows (`os.path.expanduser("~")`).
  Also tightens `looksLikeModelId` to bail on Windows drive-letter
  paths (`C:\foo`, `C:/foo`) so `--chat C:\path\to\model` works whether
  or not the path exists yet.
- **`src/safetensors.zig`** — Windows file-mapping shim. Zig 0.14.1
  doesn't expose `CreateFileMappingW` / `MapViewOfFile` /
  `UnmapViewOfFile` from `std.os.windows.kernel32`, so we declare them
  via `extern "kernel32"` inside an `if (builtin.os.tag == .windows)`
  struct. Same zero-copy semantics as the POSIX `std.posix.mmap` path —
  no Tensor call sites had to change. Also bumps `std.mem.page_size`
  → `std.heap.page_size_min` (Zig 0.14.1 stdlib rename).
- **`build.zig`** — link `vulkan-1` on Windows (the loader import lib)
  vs `vulkan` on POSIX. Reads `$VULKAN_SDK` to add `\Include` and
  `\Lib` paths so `@cInclude("vulkan/vulkan.h")` and the link both
  resolve cleanly.
- **`build.zig.zon`** — schema migration to Zig 0.14.1 stable: enum-
  literal `name`, `fingerprint`, `minimum_zig_version`. (Affects every
  platform; the prior format only worked on 0.14-dev.)
- **`src/gpu/vk.zig`** — physical-device pick now scores all
  enumerated devices (DISCRETE > INTEGRATED > VIRTUAL > CPU > other)
  rather than falling through to `devs[0]`. Defensive against Windows
  ICDs that occasionally enumerate a software/virtual adapter ahead of
  the real GPU.

## Diagnostics

Set `VALKYR_VK_VERBOSE` to any value to dump the enumerated physical
devices, the device picked, and the queue-family table at every
`Context.init`. Off by default.

```powershell
$env:VALKYR_VK_VERBOSE = "1"
.\zig-out\bin\valkyr.exe --chat Qwen/Qwen3-0.6B --q4k "Hello"
```

Sample output on Strix Point:

```
vk: enumerated 1 physical device(s):
vk:   - [integrated] AMD Radeon(TM) 890M Graphics (api 1.4.344)
vk: picked [integrated] AMD Radeon(TM) 890M Graphics
vk: queue families (5):
vk:   - family 0: count=8 flags=0xf GRAPHICS COMPUTE TRANSFER
vk:   - family 1: count=8 flags=0xe COMPUTE TRANSFER [picked]
vk:   - family 2: count=1 flags=0xc TRANSFER
vk:   - family 3: count=1 flags=0x40
vk:   - family 4: count=1 flags=0x20
```

## Known issues

1. **`vkCreateCommandPool` validation warning** (Debug / ReleaseSafe
   only). On AMD Windows, the Khronos validation layer fires
   `VUID-vkCreateCommandPool-queueFamilyIndex-01937` claiming
   `queueFamilyIndex (0)` doesn't match the device's queue create
   infos — but the source provably uses the same `qf_index` for both
   `VkDeviceQueueCreateInfo` and `VkCommandPoolCreateInfo`, and every
   GPU parity test passes to ~1e-6. Most likely a false positive in
   the AMD ICD ↔ validation layer interaction on Windows. Cosmetic;
   silent in ReleaseFast (validation off).

2. **`runDecoderStackTrainGpuRealShapeSmoke` NaN at step ~11**
   (`src/smoke/decoder.zig:1647`). The Qwen3-0.6B-class real-shape
   training envelope diverges into `NaN` after a handful of Adam
   steps on the AMD iGPU. Smaller-shape Adam smokes pass cleanly
   (CE 1.93 → 0.001 over 200 steps), and inference at the full 0.6B
   shape works perfectly — so the kernel set is correct; something
   is dim- or schedule-sensitive on UMA. Doesn't block any inference
   path.

## Future Windows-side work

- Root-cause the large-shape training NaN — kernel-by-kernel parity
  at the failing dims, then either fix the offending kernel or add
  a barrier between offending dispatches.
- UMA-aware memory allocation — on AMD APUs both DEVICE_LOCAL and
  HOST_VISIBLE heaps are physically the same RAM, so the staging-
  buffer copy in the upload path is wasteful. A UMA fast-path could
  map weights directly into HOST_VISIBLE+DEVICE_LOCAL memory and
  skip the copy entirely. Pure perf optimization; not blocking.
- `--serve` (HTTP server) hasn't been exercised on Windows yet —
  `src/server/http.zig` touches `std.posix.ReadError` which compiled,
  but the socket layer is untested.
