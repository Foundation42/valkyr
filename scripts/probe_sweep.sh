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
  # Mistral 7B v0.3 isn't a same-family scale-up of Llama 3.2 — it's
  # a neighbor architecture (same SiLU / plain-RMSNorm / GQA / untied
  # lm_head shape, routes through the Llama loader) at ~7B scale.
  # Treat its data point as "what does ~7B look like, with the caveat
  # of slightly different training corpus + tokenizer." Llama 3.1 8B
  # would be the cleanest third point but it's a gated checkpoint.
  "mistralai/Mistral-7B-Instruct-v0.3:7B-mistral"
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
    "$BIN" --chat "$id" --q4k --probe "$out" "$text" \
      | tail -2 \
      | sed "s/^/    /"
  done
done

echo
echo "done. traces: $OUT/*.jsonl"
echo "next: scripts/probe_analyze.py $OUT"
