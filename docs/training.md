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
| `--out PATH` | off | Save a `.vkpt` checkpoint at the end |
| `--print-every K` | `5` | Per-step timing print cadence; `0` to suppress |

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
chunk; once it lands, `--chat <model> --ckpt <path>` will give
production-speed generation off a fine-tune.

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
./zig-out/bin/valkyr --real-train-step-smoke   # β-5: one Adam step
./zig-out/bin/valkyr --real-multi-step-smoke   # β-6a: 30-step overfit
./zig-out/bin/valkyr --checkpoint-smoke        # β-6b: toy save/load round-trip
./zig-out/bin/valkyr --real-sampling-smoke     # β-6c: pre/post text shift
./zig-out/bin/valkyr --decoder-stack-train-smoke  # toy 8c-α-3 (200-step convergence)
```

Each smoke gates a specific property and prints a `PASS …` line on
success. They're the canonical place to look if you want to see how a
particular piece of the fine-tune flow is exercised in isolation.

## What's not supported

- **Multi-GPU training.** Single-device only. DDP / tensor parallel is
  not on the roadmap before the single-GPU surface stabilises.
- **Mixed precision.** fp32 throughout. bf16 / fp16 forward + master-
  fp32 weights is a future depth chunk.
- **LoRA / PEFT for fine-tuning.** The `cpu/lora.zig` + LoRA GPU smokes
  prove the kernel inventory works; wiring LoRA adapters into
  `Runner.step` is a separate chunk.
- **Sliding-window attention (Gemma 2/3).** The trainer uses standard
  causal attention. Gemma's per-layer alternating sliding-window mask
  is a model-family chunk.
- **Resume-from-checkpoint into inference.** `--fine-tune --out`
  writes a `.vkpt`; `--chat <ckpt.vkpt>` doesn't load it yet.
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
  `runner.loadCheckpoint(allocator, path)`.
- `train_sampling.greedyDecode(...)` — autoregressive greedy decode
  wrapping `forwardLogits`, used by the `--probe` flag.

The CLI driver in `src/commands/finetune.zig` is the worked example —
about 200 lines composing the four modules above into the shape the
flag exposes.
