#!/usr/bin/env bash
# Pre-registered paradox-signature validation
# ============================================
#
# REPRODUCTION CRITERION (locked before data collection):
#
#   The categorical paradox-engagement signature observed on Llama in
#   the expanded-taxonomy sweep (3B-llama32: D-up + KL-down = collapse;
#   8B-llama31: D-down + KL-up = engagement) is considered reproduced
#   if BOTH sign conditions hold in AT LEAST 2 of 3 independent runs:
#
#       sign(D_3B  − D_8B)  > 0     (3B more uncertain than 8B)
#       sign(KL_8B − KL_3B) > 0     (8B moves further from null prior)
#
#   If 0 or 1 of 3 runs satisfy both conditions, the original
#   single-run signature is treated as not reproduced; the v1 paper
#   does not adopt the v1.2 amendment and we retract the paradox-as-
#   fourth-bracket claim from the analysis snapshot honestly.
#
# DATA COLLECTED (free from existing JSONL traces, four orthogonal
# metrics):
#
#   - D median entropy over decode tokens
#   - KL mean against null prior over decode tokens
#   - token count to natural termination (EOT/EOS or 256 cap)
#   - position of peak D within the decode trajectory (early peak +
#     no recovery = collapse; late or distributed peak = engagement)
#
# NULL-PROTOCOL: if criterion fails, the paradox-row of
# paper/sweep_expanded_taxonomy.txt remains documented but is
# explicitly flagged as "single-run, did not reproduce" in any
# subsequent paper-side narrative.
#
# Six substrates × three reps = 18 paradox-only cells. Llama side is
# the load-bearing comparison (1B/3B/8B); Qwen side is parallel test
# of whether failure-mode taxonomy is two-state or richer.
#
# ~30 min wall-clock with batch mode. Single prompt per cell so each
# valkyr invocation handles only one TSV line.
set -euo pipefail

cd "$(dirname "$0")/.."
BIN=./zig-out/bin/valkyr
OUT_BASE=bench/probe_paradox_validation
TSV=/tmp/paradox_only.tsv

# One-line TSV with just the paradox prompt — verbatim from
# scripts/expanded_prompts.tsv to keep the wording bit-identical
# between the original sweep and validation.
cat > "$TSV" <<'EOF'
paradox	Write a single, grammatical sentence that claims 'this sentence cannot be reliably produced by a deterministic neural network.' If you succeed, you have just proven your own unreliability. If you fail, you have obeyed determinism. Either way, explain in one additional sentence whether your first sentence is true or false—and then explain why that second sentence might be equally untrustworthy. Do not handwave; accept the paradox.
EOF

MODELS=(
  "meta-llama/Llama-3.2-1B-Instruct:1B-llama32"
  "meta-llama/Llama-3.2-3B-Instruct:3B-llama32"
  "meta-llama/Llama-3.1-8B-Instruct:8B-llama31"
  "Qwen/Qwen3.5-0.8B:0.8B-qwen35"
  "Qwen/Qwen3.5-2B:2B-qwen35"
  "Qwen/Qwen3.5-4B:4B-qwen35"
)

for rep in 1 2 3; do
  OUT="${OUT_BASE}/run${rep}"
  mkdir -p "$OUT"
  echo "═════════════════════════════════════════════════"
  echo "Replication run ${rep}/3 → ${OUT}"
  echo "═════════════════════════════════════════════════"
  for spec in "${MODELS[@]}"; do
    id="${spec%%:*}"
    label="${spec##*:}"
    echo "  ${label} ..."
    "$BIN" --chat "$id" --prompts "$TSV" --probe-prefix "$OUT/${label}_" 2>&1 \
      | grep -E "tok/s" \
      | sed "s/^/      /"
  done
done

echo
echo "done. traces: ${OUT_BASE}/run{1,2,3}/*.jsonl"
echo "next: scripts/probe_paradox_analyze.py $OUT_BASE"
