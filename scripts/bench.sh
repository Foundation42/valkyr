#!/usr/bin/env bash
#
# Cross-model perf + quality bench. For each (model × precision) pair,
# captures upload time, decode tok/s, and a short response sample for
# regression diffing. Logs full per-run output to bench/<timestamp>/.
#
# Usage:
#   scripts/bench.sh                       # all models × all precisions
#   scripts/bench.sh --quick               # skip 27B (slow upload)
#   scripts/bench.sh --models gemma,4b     # subset of models
#   scripts/bench.sh --precisions q4k,bf16 # subset of precisions
#   scripts/bench.sh --smoke               # one-model-per-family quick smoke
#                                          # (gemma + llama32 + qwen3 dense +
#                                          # qwen3.5 hybrid, q4k only,
#                                          # --max-new 20). Use as a refactor
#                                          # regression net.
#   scripts/bench.sh --max-new N           # cap generation at N tokens
#                                          # (default 256; smoke uses 20)
#
# Markdown summary table goes to stdout + bench/<ts>/SUMMARY.md.
# Build is checked but not forced — run `zig build -Doptimize=ReleaseFast`
# yourself first; Debug builds will skew tok/s and upload by 4-5×.

set -euo pipefail

cd "$(dirname "$0")/.."

# ── Model id table ─────────────────────────────────────────────────
# Short keys map to HF model ids (`org/name`). valkyr resolves them
# at run time via hf_cache.zig — no snapshot hashes hardcoded.
HF_HUB="${HF_HUB:-$HOME/.cache/huggingface/hub}"
declare -A MODEL_ID=(
    [gemma]="google/gemma-2b-it"
    [tinyllama]="TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    [llama32-1b]="meta-llama/Llama-3.2-1B-Instruct"
    [llama32]="meta-llama/Llama-3.2-3B-Instruct"
    [mistral7b]="mistralai/Mistral-7B-Instruct-v0.3"
    [qwen3-4b]="Qwen/Qwen3-4B-Instruct-2507"
    [0.8b]="Qwen/Qwen3.5-0.8B"
    [4b]="Qwen/Qwen3.5-4B"
    [27b]="Qwen/Qwen3.6-27B"
)
DEFAULT_MODELS="gemma tinyllama llama32 mistral7b 0.8b 4b 27b"
DEFAULT_PRECISIONS="bf16 q4 q4k"

# Smoke covers one model per family (Gemma / Llama / Qwen3 dense /
# Qwen3.5 hybrid) so a refactor regression in any one path surfaces
# fast. q4k matches the production embed path (matryoshka ai_demo).
SMOKE_MODELS="gemma llama32-1b qwen3-4b 0.8b"
SMOKE_PRECISIONS="q4k"

# ── Args ────────────────────────────────────────────────────────────
MODELS="$DEFAULT_MODELS"
PRECISIONS="$DEFAULT_PRECISIONS"
QUICK=0
SMOKE=0
MAX_NEW=256
while [[ $# -gt 0 ]]; do
    case "$1" in
        --quick)        QUICK=1; shift ;;
        --smoke)        SMOKE=1; shift ;;
        --max-new)      MAX_NEW="$2"; shift 2 ;;
        --models)       MODELS="${2//,/ }"; shift 2 ;;
        --precisions)   PRECISIONS="${2//,/ }"; shift 2 ;;
        -h|--help)
            sed -n '4,22p' "$0"; exit 0 ;;
        *) echo "unknown arg: $1" >&2; exit 1 ;;
    esac
done

if [[ $SMOKE -eq 1 ]]; then
    # Override unless the caller explicitly passed --models / --precisions.
    [[ "$MODELS" == "$DEFAULT_MODELS" ]] && MODELS="$SMOKE_MODELS"
    [[ "$PRECISIONS" == "$DEFAULT_PRECISIONS" ]] && PRECISIONS="$SMOKE_PRECISIONS"
    [[ "$MAX_NEW" == "256" ]] && MAX_NEW=20
fi

if [[ $QUICK -eq 1 ]]; then
    MODELS="${MODELS//27b/}"
fi

# ── Setup ──────────────────────────────────────────────────────────
BIN=./zig-out/bin/valkyr
if [[ ! -x "$BIN" ]]; then
    echo "missing $BIN — build with: zig build -Doptimize=ReleaseFast" >&2
    exit 1
fi

TS=$(date -u +%Y%m%dT%H%M%SZ)
OUT=bench/$TS
mkdir -p "$OUT"

# Fixed seed + prompt designed to generate ~200+ decode tokens (stable
# tok/s reading) and surface coherence/quality signal in the response
# sample column. Models hit the max_response cap (256) before EOS on
# this length, which is fine for tok/s — the timer stops at cap too.
PROMPT="Write a detailed 250-word story about a lighthouse keeper who discovers a strange message in a bottle. Include vivid sensory details and an unexpected twist at the end."
SEED=42

SUMMARY=$OUT/SUMMARY.md
{
    if [[ $SMOKE -eq 1 ]]; then
        echo "# valkyr smoke — $TS"
    else
        echo "# valkyr cross-model bench — $TS"
    fi
    echo
    echo "Prompt: \`$PROMPT\`  •  Seed: $SEED  •  max-new: $MAX_NEW"
    echo
    echo "| Model | Precision | Upload (s) | tok/s | Response (first 200 chars) |"
    echo "|---|---|---:|---:|---|"
} > "$SUMMARY"

# ── Per-pair runner ────────────────────────────────────────────────
# Returns 0 on success (writes a row to SUMMARY); 1 on parse failure.
run_pair() {
    local model_key=$1 prec=$2
    local model_id=${MODEL_ID[$model_key]}
    local log=$OUT/${model_key}_${prec}.log
    local prec_flag=""
    case "$prec" in
        bf16) prec_flag="" ;;
        q4)   prec_flag="--q4" ;;
        q4k)  prec_flag="--q4k" ;;
        *) echo "unknown precision: $prec" >&2; return 1 ;;
    esac

    # Quick existence probe via the cache dir naming convention. valkyr
    # would error cleanly otherwise, but we want to record SKIP rather
    # than ERR for "not downloaded yet" cases.
    local cache_name="models--${model_id//\//--}"
    if [[ ! -d "$HF_HUB/$cache_name" ]]; then
        echo "  SKIP $model_key (not in HF cache: $model_id)" >&2
        echo "| $model_key | $prec | — | — | (not in cache) |" >> "$SUMMARY"
        return 0
    fi

    echo "▶ $model_key $prec ..." >&2
    local t_start=$(date +%s)
    # 600s timeout: 27B q4_k upload runs ~5 min on 8-core CPU. Surfaces
    # as TIMEOUT in the summary if exceeded.
    if ! timeout 600 "$BIN" --chat "$model_id" $prec_flag --seed $SEED --max-new $MAX_NEW "$PROMPT" > "$log" 2>&1; then
        local rc=$?
        if [[ $rc -eq 124 ]]; then
            echo "| $model_key | $prec | TIMEOUT | — | (>600s) |" >> "$SUMMARY"
        else
            echo "| $model_key | $prec | ERR rc=$rc | — | (see log) |" >> "$SUMMARY"
        fi
        return 0
    fi
    local t_end=$(date +%s)
    local wall=$((t_end - t_start))

    # Parse "upload took NNNN ms" and "[N tok in NNN ms, X.X tok/s]".
    local upload_s
    upload_s=$(grep -oE 'upload took [0-9]+ ms' "$log" | head -1 \
        | awk '{printf "%.1f", $3/1000.0}')
    local tps
    tps=$(grep -oE '[0-9]+\.[0-9]+ tok/s' "$log" | tail -1 | awk '{print $1}')

    # First 100 chars of response (line that starts with "response: " or
    # the first non-empty line after the prompt block). Sanitize for
    # the markdown table: replace pipes, collapse whitespace.
    local response
    response=$(awk '/^response: /{sub(/^response: /,""); printf "%s", $0; while((getline ln) > 0 && ln !~ /^\[/ && ln !~ /^$/) printf " %s", ln; exit}' "$log" \
        | head -c 200 | tr '|' '/' | tr -s '[:space:]' ' ')

    upload_s=${upload_s:-?}
    tps=${tps:-?}
    response=${response:-?}

    echo "| $model_key | $prec | $upload_s | $tps | $response |" >> "$SUMMARY"
    echo "  done in ${wall}s — upload ${upload_s}s, ${tps} tok/s" >&2
}

# ── Sweep ──────────────────────────────────────────────────────────
for model in $MODELS; do
    [[ -z "${MODEL_ID[$model]:-}" ]] && { echo "skip unknown model: $model" >&2; continue; }
    for prec in $PRECISIONS; do
        run_pair "$model" "$prec"
    done
done

echo >&2
echo "Summary: $SUMMARY" >&2
echo "Logs:    $OUT/*.log" >&2
echo >&2
cat "$SUMMARY"
