#!/usr/bin/env bash
# Merge the per-trace/per-phase CSVs into results/runs.csv.
#
# Rows from a phase that was later re-run are dropped in favour of the re-run:
# pass the stale phase prefixes as DROP="C- " when merging.
set -eu
OUT=${OUT:-results/runs.csv}
DROP=${DROP:-}
mkdir -p "$(dirname "$OUT")"
first=1
: > "$OUT.tmp"
for f in "$@"; do
    [ -s "$f" ] || continue
    if [ $first -eq 1 ]; then head -1 "$f" > "$OUT.tmp"; first=0; fi
    tail -n +2 "$f" >> "$OUT.tmp"
done
if [ -n "$DROP" ]; then
    head -1 "$OUT.tmp" > "$OUT"
    awk -F, -v drop="$DROP" 'NR>1 {
        split(drop, d, " ");
        for (i in d) if (d[i] != "" && index($1, d[i]) == 1) next;
        print
    }' "$OUT.tmp" >> "$OUT"
    rm -f "$OUT.tmp"
else
    mv "$OUT.tmp" "$OUT"
fi
echo "$OUT: $(( $(wc -l < "$OUT") - 1 )) runs"
