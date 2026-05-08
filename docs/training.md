# Training your own model

Fine-tune real Qwen3-class transformers from a JSONL dataset, all on
the GPU through Vulkan compute — same SPIR-V that the inference path
uses, no CUDA. Back to [README](../README.md). See also:
[embedding.md § Embedding training](embedding.md#embedding-training)
for the cooperative in-frame training surface,
[roadmap.md](roadmap.md), [models.md](models.md).

## What's supported today

- **Architectures.** Qwen3 dense family
  (`Qwen/Qwen3-0.6B`, `Qwen/Qwen3-1.7B`, …) — 28 layers, GQA, SwiGLU
  FFN, full RoPE, per-head Q/K-RMSNorm. Other Qwen3-class checkpoints
  load by inference of `config.json`; the architecture gate refuses
  anything that isn't full-attention with `qk_norm=true` and no
  `attn_output_gate` (Qwen3.5 hybrid + Qwen3.6 GQA-only need their own
  trainer chunks).
- **Optimizer.** Adam with bias correction (β₁=0.9, β₂=0.999, ε=1e-8).
  Single learning rate; mutable across steps if you drive the trainer
  programmatically.
- **Loss.** Cut Cross-Entropy (CCE) — fuses the LM-head matmul into the
  loss kernel, never materializes the `[n_pos × vocab]` logits tensor.
  Bakes in `lse` + per-row loss readback for monitoring.
- **Precision.** fp32 throughout (weights, gradients, Adam moments).
  bf16 weights on disk are widened to fp32 at load time.
- **Dataset.** JSONL with one `{"text": "..."}` per line. Lines are
  tokenized, separated by EOS, packed into one stream, and exposed as
  sliding `(n_pos+1)` windows for next-token prediction.
- **Checkpointing.** Custom `.vkpt` format — header + `Config` snapshot
  + raw fp32 blocks for every param + Adam m/v + step counter. Round-
  trip preserves training state bit-for-bit at toy scale.
- **LoRA / PEFT.** `cfg.lora_targets` bitmask selects any subset of
  the seven dense projections (Q / K / V / O / gate / up / down) for
  rank-r adapter training. Frozen base + trainable A,B per target,
  with optional LoRA+ differential lr for B (Chronicals Theorem 1).
  Saves to a separate `.lvkpt` format that's ~170× smaller than the
  full `.vkpt` (52.5 MiB vs ~9 GiB on Qwen3-0.6B). See
  [§ LoRA fine-tuning](#lora-fine-tuning) below.

## Quick start

```sh
./zig-out/bin/valkyr --fine-tune Qwen/Qwen3-0.6B \
    --data data/train/tiny_facts.jsonl \
    --steps 30 \
    --probe "The capital of France is" \
    --out /tmp/qwen3-finetuned.vkpt
```

Output (real run, an RTX 3090, Debug build):

```
[fine-tune] model: Qwen/Qwen3-0.6B
[fine-tune] arch: 28 layers, dim=1024, GQA 16/8, head_dim=128, ff_dim=3072, vocab=151936
[fine-tune] dataset: 339 batches at n_pos=16
[fine-tune] probe: "The capital of France is" (5 tokens)
[probe before] The capital of France is Paris. The capital of Italy is Rome. The capital of Spain is Madrid. The capital of France
[fine-tune] CE initial: 2.7778 (batch 0)
[step    5/30] 298.3 ms/step
[step   10/30] 297.7 ms/step
[step   15/30] 296.5 ms/step
[step   20/30] 298.4 ms/step
[step   25/30] 301.6 ms/step
[step   30/30] 301.1 ms/step
[fine-tune] CE final: 0.0005 (Δ -2.7773, 99.98% drop on batch 0; total 8942.2 ms = 298.1 ms/step)
[probe after]  The capital of France is Paris. Paris sits on the river Seine and is famous for being famous for being famous for being
[fine-tune] saved checkpoint: /tmp/qwen3-finetuned.vkpt (8.40 GiB, 46132.5 ms)
```

What you're looking at: `--probe` runs a greedy-decode of the prompt
both before and after training. **Pre-fine-tune** the base model
produces a generic capital-listing pattern — Qwen3-0.6B's pretrained
behavior on a factual prompt. **Post-fine-tune** the first 11 tokens
are *verbatim* from `tiny_facts.jsonl` line 1 ("Paris. Paris sits on
the river Seine and is famous for…"); the model has memorized the
single training batch. Once the autoregressive window slides past the
16-token training shape it falls into a degenerate loop ("famous for
being famous for being…") because that's the regime past what was
explicitly trained.

This is single-batch overfit by design — the cleanest possible
demonstration that the gradient flow is sound end-to-end. Add
`--rotate` to cycle through every batch in the dataset for a more
realistic multi-example fine-tune.

## CLI reference

```
valkyr --fine-tune <model> --data <jsonl> [options]
```

| Flag | Default | Notes |
|---|---|---|
| `<model>` | required | HF id (`Qwen/Qwen3-0.6B`) or local snapshot dir |
| `--data <jsonl>` | required | One `{"text": "..."}` per line |
| `--steps N` | `30` | Adam steps |
| `--lr LR` | `1e-5` | Standard fine-tune lr; well-conditioned for pretrained weights |
| `--n-pos N` | `16` | Context window in tokens; larger uses more VRAM |
| `--batch IDX` | `0` | Single-batch index when `--rotate` is off |
| `--rotate` | off | Cycle `dataset.batch(step + IDX) mod N` each step |
| `--probe TEXT` | off | Sample N tokens from this prompt before & after training |
| `--n-gen N` | `20` | Tokens generated per probe sample |
| `--out PATH` | off | Save a checkpoint at the end (`.vkpt` for full-FT, `.lvkpt` when LoRA is on) |
| `--print-every K` | `5` | Per-step timing print cadence; `0` to suppress |
| `--lora-targets SPEC` | off | Comma-separated projection list — `q,k,v,o,gate,up,down` or shorthands `all_attn`, `all_ffn`, `all`. Switches to LoRA mode. |
| `--lora-rank N` | `16` | LoRA adapter rank (when `--lora-targets` is set) |
| `--lora-alpha A` | `32.0` | LoRA scaling — effective per-projection scale is α/r |
| `--lora-lr-b-scale L` | `1.0` | LoRA+ multiplier on B's lr (Chronicals Theorem 1; ~16 typical) |

## Dataset format

`tiny_facts.jsonl` ships in `data/train/` as a working example:

```jsonl
{"text": "The capital of France is Paris. Paris sits on the river Seine and is famous for the Eiffel Tower."}
{"text": "The capital of Germany is Berlin. Berlin is known for the Brandenburg Gate and a rich modern history."}
...
```

Each line is tokenized with the model's own tokenizer, joined into one
stream with `<|im_end|>` (id 151645 for Qwen3) as the separator, then
sliced into overlapping `(n_pos+1)` windows. With `n_pos=16` and 15
fact lines, `tiny_facts.jsonl` produces 355 packed ids and 339 sliding
batches.

There's intentionally no chat-template wrapping (no `<|im_start|>user`
/ `<|im_start|>assistant` framing). The smoke uses raw text-completion
training; if you want instruction-style fine-tunes, pre-format your
JSONL `text` field with the chat-template tokens you want the model to
see.

### Bring your own chat-style dataset

If your data is already in the standard `{"messages": [{"role": ..., "content": ...}]}`
JSONL shape (Unsloth, OpenAI fine-tune format, etc.), use
`scripts/convert_messages_to_text.py` to flatten it into the `{"text": "..."}`
format with the Qwen3 chat template applied:

```sh
python3 scripts/convert_messages_to_text.py \
    --in your_chat_data.jsonl \
    --out data/train/your_data.jsonl
```

A real-world data point — a 671-row pack-native dataset (system + user +
assistant per row, mean ~1 KiB / ~290 tokens per row) trained 30 steps
on Qwen3-0.6B at `n_pos=128` with `--rotate` produced a probe-shift like
this:

```
[probe before] [I recall: the problem is to find the number of ways to arrange the letters in the word "BANANA"...
[probe after]  [I recall: ...] blocks as your primary knowledge source. If a recall contradicts your internal knowledge...
```

The post-train output is the *exact verbatim system prompt* that
appeared in every one of the 671 training rows — the model has absorbed
the dataset's most-common boilerplate. This is generalization across
batches (not single-batch memorization like the `tiny_facts` demo).
1,247 ms/step at n_pos=128 Debug on an RTX 3090.

## LoRA fine-tuning

Pass `--lora-targets` (and optionally `--lora-rank`, `--lora-alpha`,
`--lora-lr-b-scale`) to switch into LoRA mode. The `--lora-finetune`
alias is the discoverable shortcut — same backend, but requires
`--lora-targets` and bumps the lr default to `5e-4` (typical LoRA
fine-tune rate is 10–50× full-FT lr because there's a much smaller
trainable surface).

```sh
./zig-out/bin/valkyr --lora-finetune Qwen/Qwen3-0.6B \
    --data data/train/tiny_facts.jsonl \
    --steps 30 \
    --lora-targets all_attn \
    --lora-rank 16 \
    --lora-alpha 32 \
    --probe "The capital of France is" \
    --out /tmp/qwen3-lora.lvkpt
```

In LoRA mode every dense matmul named in `--lora-targets` is replaced
by `y = x · Wᵀ + (α/r) · (x · Aᵀ) · Bᵀ` with W frozen and A,B trained.
**Every other parameter is also frozen** — embedding, RMSNorm gains,
final norm, lm_head, and the dense W's not named in the bitmask. This
is the standard LoRA semantics. The `.lvkpt` format only persists A,B
(plus their Adam state); base W stays in the source safetensors and is
re-loaded on resume.

### Target bitmask

| Spec | Bit | What it covers |
|---|---|---|
| `q` | 1 | W_q (queries) |
| `k` | 2 | W_k (keys) |
| `v` | 4 | W_v (values) |
| `o` | 8 | W_o (output projection) |
| `gate` | 16 | W_gate (FFN gate) |
| `up` | 32 | W_up (FFN up) |
| `down` | 64 | W_down (FFN down) |
| `all_attn` | 15 | `q | k | v | o` (most common LoRA setup) |
| `all_ffn` | 112 | `gate | up | down` |
| `all` | 127 | every dense projection |

Combine freely: `--lora-targets q,v` is the original
`Hu et al. 2021` LoRA setup; `--lora-targets all` is the maximal
coverage at higher per-step cost.

### `.lvkpt` size + load time

LoRA-only checkpoints are ~170× smaller than full `.vkpt` because
they skip the base weights. On Qwen3-0.6B (28 layers, dim=1024,
GQA 16/8, ff_dim=3072, vocab=151936):

| Targets | Rank | `.lvkpt` size | Load time |
|---|---|---|---|
| `all_attn` | 16 | **52.5 MiB** | ~370 ms |
| `q` | 16 | ~16 MiB | <200 ms |
| `all` | 16 | ~125 MiB | ~700 ms |
| (full `.vkpt` reference) | — | ~9 GiB | ~46–52 s |

### LoRA performance

On an RTX 3090, real-shape Qwen3-0.6B at `n_pos=16`, ReleaseFast:

| Mode | Targets | ms/step | Δ vs full-FT |
|---|---|---|---|
| Full fine-tune | — | 268 | baseline |
| LoRA | `q` rank-16 | **256** | **−12 ms / −4.7%** (faster!) |
| LoRA | `all_attn` rank-16 | ~310 | +42 ms / +15.7% |
| LoRA | `all` rank-16 | 399 | +131 ms / +48.8% |

LoRA-Q is **faster** than full-FT because the freeze skips Adam on
the huge `embed` and `lm_head` buffers (vocab × dim each is ~600 MiB
of fp32 + Adam m/v at Qwen3-0.6B), which costs more HBM bandwidth
than the +14 LoRA dispatches per layer add. Past ~3 enabled targets,
the LoRA chain overhead crosses back over and the all-projection
case is slower.

### Resume from `.lvkpt`

`--gen-from-ckpt` autodetects `.vkpt` vs `.lvkpt` by sniffing the
4-byte magic at the file head. For `.lvkpt`, pass the same
`--lora-targets` / `--lora-rank` the checkpoint was saved with so
the Runner allocates matching adapter slots before load overwrites
them; `cfgShapeMatches` rejects mismatches with
`error.LoraCheckpointConfigMismatch`.

```sh
./zig-out/bin/valkyr --gen-from-ckpt Qwen/Qwen3-0.6B \
    --ckpt /tmp/qwen3-lora.lvkpt \
    --lora-targets all_attn \
    --lora-rank 16 \
    --lora-alpha 32 \
    --prompt "The capital of France is" \
    --n-gen 20
```

A 10-step `--lora-finetune` on `tiny_facts.jsonl` with
`--lora-targets all_attn --lora-rank 16` produces this:

```
[gen-from-ckpt] loaded LoRA checkpoint in 367 ms
[gen-from-ckpt] 8 tokens in 746 ms (93.3 ms/tok)

The capital of France is Paris. Paris sits on the river Se
```

The post-train continuation is verbatim from `tiny_facts.jsonl` line 1
(same memorization signal as the full-FT demo), achieved with a
**52.5 MiB** delta on disk instead of an 8.4 GiB checkpoint.

### Production-speed inference: `--chat --lora-ckpt`

`--gen-from-ckpt` reuses the *training* `forwardLogits` (one full
n_pos-shape pass per token, no KV cache) — fine for sanity checks at
~107 ms/tok but slow for actually using a fine-tune. The fast path
folds the LoRA delta into the base weights at load time and runs the
unmodified inference matmul:

    W' = W + (α/r) · B · A      (one-time merge per projection)
    y  = x · W'ᵀ                (every token afterwards, KV-cached)

So `--chat --lora-ckpt` pays a one-time merge cost (~4 s for
`all_attn` rank-16 on Qwen3-0.6B, host-side fp32 then back to bf16)
and zero per-token LoRA overhead afterwards.

```sh
./zig-out/bin/valkyr --chat Qwen/Qwen3-0.6B \
    --lora-ckpt /tmp/qwen3-lora.lvkpt \
    --lora-targets all_attn \
    --lora-rank 16 \
    --lora-alpha 32 \
    --max-new 30 \
    --temp 0 \
    "The capital of France is"
```

| Path | Tokens / s | ms / tok | Notes |
|---|---|---|---|
| `--gen-from-ckpt` (training Runner) | ~9 | ~107 | n_pos-shape forward per token |
| `--chat --lora-ckpt` (merged, KV-cached) | **~145** | **~7** | same as base — zero LoRA overhead |
| `--chat` (base, no LoRA) | ~143 | ~7 | reference |

**~15× faster** than the slow path. The merge supports `bf16_matmul`
(default `--chat` precision) and `fp32_all`. Q4_0 / Q4_K need a
re-quantisation pass after the merge (changes the per-block scale
grids) — that's a future chunk; pass `--lora-ckpt` without `--q4` /
`--q4k` for now.

`--gen-from-ckpt` remains the only path for full-FT `.vkpt`
checkpoints — the inference loader needs a positional-blob → safetensors-
tensor-name mapping that's a separate chunk from the LoRA-merge work.
With LoRA being the modern fine-tune workflow it covers the common
case for now.

### `.lvkpt` format

Same buffered-IO shape as `.vkpt` but a different magic + body:

```
[16 bytes — CheckpointHeader]
  magic    : "VLKP"
  version  : u32 = 1
  step_t   : u32  (Adam timestep counter)
  cfg_size : u32  (sanity check)

[Config struct as raw bytes — std.mem.asBytes(&cfg)]

[body — raw fp32 in this canonical order]
  for each Proj in [.q, .k, .v, .o, .gate, .up, .down]:
    if (cfg.lora_targets & projBit(Proj)) != 0:
      for each layer i in 0..n_layers:
        A bytes      ([r, K_proj] fp32)
        m_A bytes
        v_A bytes
        B bytes      ([N_proj, r] fp32)
        m_B bytes
        v_B bytes
```

Disabled projections contribute zero bytes. `cfgShapeMatches` enforces
matching `lora_targets` and `lora_rank` between save and load — a
checkpoint trained with `all_attn` cannot be loaded into a Runner
configured for `all`.

## Checkpoint format (`.vkpt`)

Positional binary, all fp32, no per-tensor names:

```
[16 bytes — CheckpointHeader]
  magic    : "VKPT"
  version  : u32 = 1
  step_t   : u32  (Adam timestep counter)
  cfg_size : u32  (sanity check against the next field's size)

[Config struct as raw bytes — std.mem.asBytes(&cfg)]

[body — raw fp32 in this canonical order]
  Stack-level (param, m, v): embed, final_norm, lm_head
  Per-layer × N (param, m, v): w_n1, w_q, w_k, w_v, w_o, w_n2,
                                w_gate, w_up, w_down,
                                (+ w_q_norm, w_k_norm if cfg.qk_norm)
```

Sizes are implied by `Config` — no offsets table. Saved checkpoint at
Qwen3-0.6B is **~8.4 GiB** (params + Adam m/v + Adam v all fp32).
`saveCheckpoint` is bottlenecked on host readback (~933 round-trip
buffers each fenced via `vkQueueWaitIdle` — ~46 s on the test rig);
batched-readback is a future perf chunk.

### Generate from a saved checkpoint

```sh
./zig-out/bin/valkyr --gen-from-ckpt Qwen/Qwen3-0.6B \
    --ckpt fine-tuned.vkpt \
    --prompt "The capital of France is" \
    --n-gen 20
```

Loads the base model (for tokenizer + architecture), reads the `.vkpt`
into a fresh training Runner (overwrites params + Adam state), and
greedy-decodes N tokens via the same `forwardLogits` + autoregressive
loop the `--probe` flag uses. Output for a checkpoint trained 30 steps
on `tiny_facts.jsonl` batch 0 with the example above:

```
[gen-from-ckpt] loaded checkpoint in 9629 ms
[gen-from-ckpt] 20 tokens in 2139 ms (106.9 ms/tok)

The capital of France is Paris. Paris sits on the river Seine and is famous for being famous for being famous for being
```

**Slow path.** Each generated token costs a full training-shape forward
pass over `n_pos` (~107 ms/tok at Qwen3-0.6B Debug) — there's no
incremental KV cache. The fast inference path (`--chat <model>`) doesn't
yet load `.vkpt` directly because the format is fp32 and positionally
keyed while the inference loader expects bf16 / safetensors-named
tensors. fp32→bf16 conversion + tensor-name mapping is the natural next
chunk for full-weight fine-tunes.

The LoRA equivalent (`.lvkpt` → `--chat`) **is** wired up — see
[§ Production-speed inference: `--chat --lora-ckpt`](#production-speed-inference---chat---lora-ckpt)
above. The merge math `W' = W + (α/r)·B·A` doesn't need the
positional → name mapping because it operates on the bf16 weights
already in the inference path.

## Performance

On an RTX 3090, real-shape Qwen3-0.6B at `n_pos=16`:

| Build | ms/step | Notes |
|---|---|---|
| Debug | ~298 | 28 layers × 133 dispatches, host-readback-dominated |
| ReleaseFast | ~616 / measured at β-2 | bigger n_pos baseline; 5-10% faster than Debug |

GPU memory footprint at `n_pos=16`: ~9 GiB
(weights + Adam m/v + activations + dW partials). Doubling `n_pos`
roughly doubles activation memory. The trainer is single-GPU only —
no DDP / FSDP / tensor parallel.

`saveCheckpoint` time scales with the number of buffers (933 at
Qwen3-0.6B), each involving one staging-buffer round trip. Disk
write bandwidth is rarely the bottleneck.

## Smoke gates (advanced)

The same primitives the user-facing `--fine-tune` uses are exposed as
flag-gated smokes for development:

```sh
./zig-out/bin/valkyr --real-train-step-smoke      # β-5: one Adam step
./zig-out/bin/valkyr --real-multi-step-smoke      # β-6a: 30-step overfit
./zig-out/bin/valkyr --checkpoint-smoke           # β-6b: toy .vkpt round-trip
./zig-out/bin/valkyr --real-sampling-smoke        # β-6c: pre/post text shift
./zig-out/bin/valkyr --decoder-stack-train-smoke  # toy 8c-α-3 (200-step convergence)
./zig-out/bin/valkyr --lora-rec-smoke             # A4-1: helper-level LoRA parity
./zig-out/bin/valkyr --lora-q-runner-smoke        # A4-2: LoRA on Q-projection
./zig-out/bin/valkyr --lora-all-runner-smoke      # A4-3: LoRA on all 7 projections
./zig-out/bin/valkyr --lora-checkpoint-smoke      # A4-4: toy .lvkpt round-trip
./zig-out/bin/valkyr --lora-merge-smoke           # chat-path LoRA fold-in math identity
./zig-out/bin/valkyr --real-lora-q-step-smoke     # β-5 + LoRA-Q (real Qwen3-0.6B)
./zig-out/bin/valkyr --real-lora-all-step-smoke   # β-5 + LoRA-all (real Qwen3-0.6B)
```

Each smoke gates a specific property and prints a `PASS …` line on
success. They're the canonical place to look if you want to see how a
particular piece of the fine-tune flow is exercised in isolation.

## What's not supported

- **Multi-GPU training.** Single-device only. DDP / tensor parallel is
  not on the roadmap before the single-GPU surface stabilises.
- **Mixed precision.** fp32 throughout. bf16 / fp16 forward + master-
  fp32 weights is a future depth chunk.
- **Per-projection LoRA rank.** A single `lora_rank` applies to every
  enabled projection. Per-target rank (e.g. rank-16 on Q+V, rank-4 on
  K+O) would be a small additive change to `LoraState.allocAndInit`
  but isn't on yet.
- **Sliding-window attention (Gemma 2/3).** The trainer uses standard
  causal attention. Gemma's per-layer alternating sliding-window mask
  is a model-family chunk.
- **Resume full-FT checkpoint into inference.** `--fine-tune --out`
  writes a `.vkpt`; `--chat <ckpt.vkpt>` doesn't load it yet
  (positional-blob → safetensors-tensor-name mapping is a separate
  chunk). The LoRA path **is** wired: `--chat --lora-ckpt foo.lvkpt`
  folds the delta into the base bf16 weights at load time.
- **`--lora-ckpt` with `--q4` / `--q4k`.** The merge supports
  `bf16_matmul` (default `--chat`) and `fp32_all`. Q4 paths would need
  a per-block re-quantisation after the fp32 merge — non-trivial
  because the scale grids are recomputed per super-block. Future
  chunk; reject with a clear error today.
- **HuggingFace bit-parity.** We don't aim for byte-exact agreement
  with `transformers.Trainer.train()`; our convergence gate is "loss
  drops" not "loss matches PyTorch's logits".

## API surface (embedded use)

If you want to drive training from your own Zig code instead of the
CLI, the building blocks are in `src/train/`:

- `train_load_real.loadTrainWeights(allocator, &cpu_model, n_pos)` —
  bf16 → fp32 widening, exposes `TrainWeights` with a `view()` that
  produces an `InitWeights` for the Runner.
- `train_dataset.buildFromJsonl(allocator, &tok, path, n_pos, eos_id)`
  — packed-stream + sliding window dataset; `ds.batch(idx, in, target)`
  fills caller-provided buffers.
- `train_transformer.Runner.init(allocator, &ctx, cfg, init_weights)`
  → `runner.step(input_ids, target_ids)` (one Adam step) →
  `runner.forwardLogits(token_ids, out_logits)` (eval) →
  `runner.saveCheckpoint(allocator, path)` /
  `runner.loadCheckpoint(allocator, path)` (full state, `.vkpt`) /
  `runner.saveLoraCheckpoint(allocator, path)` /
  `runner.loadLoraCheckpoint(allocator, path)` (LoRA-only, `.lvkpt`).
- `train_transformer.LoraTarget` — bitmask constants
  (`q | k | v | o | gate | up | down | all_attn | all_ffn | all`).
  Set on `Config.lora_targets` to enable LoRA on those projections.
- `train_lora.recordLoraForward` /
  `train_lora.recordLoraBackward` — the underlying dispatch helpers
  that compose existing kernels (matmul / linear-backward / scale /
  add-in-place) into the LoRA chain. Useful if you need LoRA on a
  custom dispatch path outside the Runner.
- `train_sampling.greedyDecode(...)` — autoregressive greedy decode
  wrapping `forwardLogits`, used by the `--probe` flag.
- `lora_merge.run(allocator, ctx, &gpu_model, cpu_cfg, opts)` — folds
  a `.lvkpt` into the bf16 / fp32 projection weights of an inference
  `GpuModel` in place. The CLI hook for `--chat --lora-ckpt`; embed
  callers can drive it directly to attach a LoRA at session bring-up
  without going through the chat command.
- `lora_merge.applyLoraDeltaFp32(w, a, b, N, K, r, alpha_over_r)` —
  the pure-CPU inner loop, exposed so callers that already have the
  delta in host memory (e.g. a probe loop or a custom format) can
  fold it into a fp32 weight slice without touching the GPU buffer
  round-trip.

The CLI driver in `src/commands/finetune.zig` is the worked example —
about 280 lines composing the modules above into the shape the
`--fine-tune` / `--lora-finetune` flags expose.
