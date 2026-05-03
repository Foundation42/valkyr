# Probes

valkyr's chat decode loop exposes optional probe hooks that fire at
six points per token (token start/end, layer entry/exit, attention,
logits) and stream a self-describing JSONL trace. Useful for
interpretability work, debug tooling, and engine integration —
the probe interface is dual-purpose: a probe that *measures* a
hidden state and one that *consumes* it for a host system are
structurally identical. Back to [README](../README.md). See also:
[parity.md](parity.md), [models.md](models.md).

## Probes

valkyr's chat decode loop exposes optional **probe hooks** at six
points per token: token start/end, layer entry/exit, attention, and
logits. Hooks fire only when a probe wants them — empty-bus path is
bit-identical to the un-probed forward, parity-verified.

```sh
$ ./zig-out/bin/valkyr --chat meta-llama/Llama-3.2-1B-Instruct --q4k \
    --probe trace.jsonl "What is the capital of France?"
```

This streams a self-describing JSONL trace: a header record naming
the model + active probes + per-field definitions, then one record per
(token, layer) for activation observations and one per token for
logit observations. Trivially analyzable in Python — no Arrow / Parquet
dependency.

**v0 ships two probes:**

- **Activation entropy** (per layer per token): Shannon entropy of the
  L2-energy distribution `p_i = a_i^2 / Σ a_j^2` over the residual
  stream, plus the L2 norm. Captures how spread-vs-concentrated the
  activation is across feature dimensions, layer by layer.
- **Logit entropy + null-prior KL** (per token): conditional entropy
  H(t | context) of the softmaxed logits, and KL divergence from a
  null prior — the logit distribution the model emits for a single
  BOS token at position 0, computed once at startup. KL measures how
  far the prompt has driven the model away from baseline.

The probe interface is dual-purpose by design: a probe that *consumes*
a hidden state and hands it to a host system is structurally identical
to a probe that *measures* it. Same vtable, same hook points, same
data shapes — useful for interpretability, debug tooling, and engine
integration alike.

**Cost:** the un-probed path is unchanged; probed runs currently
trigger one GPU submit per layer (host readback between) so decode
tok/s drops ~3×. Opt-in, only paid when `--probe` is set.

Wired today on Llama 3.2 family (1B / 3B / 8B); hybrid Qwen3.5/3.6
path is gated with a clear message and lands in a later chunk.
