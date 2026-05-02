#!/usr/bin/env bash
# Substrate-relativity sweep: three prompts × N model sizes, all greedy
# at --temp 0, identical probe set, identical Q4_K_M precision. Writes
# JSONL traces under bench/probe_sweep/.
#
# Predictions (RFF spec, section 4.4):
#  1. D(t) entropy gap (philosophical − factual) grows with model size.
#  2. KL-from-null trajectory grows faster on larger models for
#     philosophical prompts; stays similar across sizes for factual.
#  3. Per-prompt C ranking: factual < creative < philosophical at every
#     model size.
#
# Usage:
#   ./scripts/probe_sweep.sh                            # 1B + 3B (always available)
#   ./scripts/probe_sweep.sh meta-llama/Llama-3.1-8B-Instruct  # add 8B
set -euo pipefail

cd "$(dirname "$0")/.."
BIN=./zig-out/bin/valkyr
OUT=bench/probe_sweep
mkdir -p "$OUT"

MODELS=(
  "meta-llama/Llama-3.2-1B-Instruct:1B-llama32"
  "meta-llama/Llama-3.2-3B-Instruct:3B-llama32"
  # Llama 3.1 8B is the same architecture family as Llama 3.2 1B/3B —
  # clean substrate-relativity scale-up across an 8× span in parameter
  # count, same tokenizer, same chat template, same RMSNorm/SwiGLU/GQA
  # shapes. The Mistral 7B reference point in earlier runs was a
  # neighbor-arch confound; with 8B available we drop it.
  "meta-llama/Llama-3.1-8B-Instruct:8B-llama31"
  # Qwen3.5 / 3.6 hybrid family — the v2 tilth-sweep substrate.
  # Different family from Llama (later training era, denser per-
  # parameter capacity, hybrid Gated-DeltaNet + full-attention
  # architecture). Used to test whether the substrate-relativity
  # threshold appears at lower parameter count when training-era
  # variables are released. K probe is only valid on the 1-in-4
  # full-attention layers; linear-attention layers contribute B and D
  # only.
  "Qwen/Qwen3.5-0.8B:0.8B-qwen35"
  "Qwen/Qwen3.5-2B:2B-qwen35"
  "Qwen/Qwen3.5-4B:4B-qwen35"
  # Qwen3.6 27B is omitted from the bf16 sweep: 27B at bf16 needs
  # ~54 GB of weight memory and does not fit on a 24 GB consumer GPU.
  # It works at --q4k (~20 tok/s on a 3090) but mixing precision
  # across the sweep would reintroduce the precision-by-size confound
  # we removed in v1. Run it separately if a clean v2 27B comparison
  # is wanted, with the precision caveat documented.
)
# Optional extras passed as positional args (id:label form).
for extra in "$@"; do MODELS+=("$extra"); done

PROMPTS=(
  "factual:What is the capital of France?"
  "creative:Write a short paragraph describing the feeling of waiting for a kettle to boil."
  # Pre-screened to resist the numbered-taxonomy template on 1B and 3B.
  # The original "what does it mean for a system to be conscious"
  # phrasing reliably triggered "1. **Integrated**: ..." bulletization
  # on Llama 3.2 1B / 3B, which compresses per-token entropy through
  # exposition structure rather than conceptual openness — confounding
  # the D(t) measurement. This phrasing forces a singular-event
  # narrative reflection. Screened on 1B/3B/7B; produces narrative
  # prose across all three.
  "philosophical:Tell me about a moment when you understood something that you didn't have words for yet."
)

for spec in "${MODELS[@]}"; do
  id="${spec%%:*}"
  label="${spec##*:}"
  for p in "${PROMPTS[@]}"; do
    pkind="${p%%:*}"
    text="${p#*:}"
    out="$OUT/${label}_${pkind}.jsonl"
    echo "→ ${label} / ${pkind} → ${out}"
    # bf16 across all sizes: removes the quantization confound from
    # the substrate-relativity axis. 1B at Q4_K_M sits right at the
    # coherence floor (collapses to token salad past ~150 tokens);
    # bf16 keeps every size in its native dynamics.
    "$BIN" --chat "$id" --probe "$out" "$text" \
      | tail -2 \
      | sed "s/^/    /"
  done
done

echo
echo "done. traces: $OUT/*.jsonl"
echo "next: scripts/probe_analyze.py $OUT"
