#!/usr/bin/env bash
# Expanded prompt sweep — capacity-tax taxonomy from Christian's
# expanded set (post-v1). Four prompts chosen to land cleanly across
# 1B–8B without needing 27B-class substrate:
#
#   - counterfactual : anti-factual prior-suppression (#3 in taxonomy)
#   - sensate        : pre-verbal qualia → prose, no-enumeration constraint baked in (#4)
#   - paradox        : self-refuting-loop, coherence-collapse-vs-engagement gradient (#7)
#   - playful        : combinatorial-creative under formal constraints (#9)
#
# Uses --prompts batch mode so each model loads exactly once and runs
# all four prompts in a single process. Saves ~5–8 minutes across
# the 24-cell sweep vs the previous one-process-per-cell shape.
set -euo pipefail

cd "$(dirname "$0")/.."
BIN=./zig-out/bin/valkyr
OUT=bench/probe_sweep_expanded
TSV=scripts/expanded_prompts.tsv
mkdir -p "$OUT"

MODELS=(
  "meta-llama/Llama-3.2-1B-Instruct:1B-llama32"
  "meta-llama/Llama-3.2-3B-Instruct:3B-llama32"
  "meta-llama/Llama-3.1-8B-Instruct:8B-llama31"
  "Qwen/Qwen3.5-0.8B:0.8B-qwen35"
  "Qwen/Qwen3.5-2B:2B-qwen35"
  "Qwen/Qwen3.5-4B:4B-qwen35"
)

for spec in "${MODELS[@]}"; do
  id="${spec%%:*}"
  label="${spec##*:}"
  echo "═══ ${label}"
  "$BIN" --chat "$id" --prompts "$TSV" --probe-prefix "$OUT/${label}_" 2>&1 \
    | grep -E "^\[|^upload|tok/s" \
    | sed "s/^/    /"
done

echo
echo "done. traces: $OUT/*.jsonl"
echo "next: scripts/probe_analyze.py $OUT"
