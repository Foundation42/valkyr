#!/usr/bin/env bash
# Generate real memory-reference traces with valgrind/lackey.
#
# Every workload is a real, unmodified system binary doing real work; lackey
# emits one line per instruction fetch and per data access, which din2vtr packs
# into the lab's binary trace format.  No synthetic reference streams are used
# anywhere in this study.
#
#   usage: gen_traces.sh [out_dir] [max_refs_per_trace]
set -u

OUT=${1:-traces}
MAX=${2:-16000000}
HERE=$(cd "$(dirname "$0")" && pwd)
WORK=$OUT/.work
mkdir -p "$OUT" "$WORK"

DIN2VTR=${DIN2VTR:-$HERE/../build/din2vtr}
if [ ! -x "$DIN2VTR" ]; then
    mkdir -p "$(dirname "$DIN2VTR")"
    g++ -O2 -o "$DIN2VTR" "$HERE/../src/din2vtr.cpp" || exit 1
fi

# ---------------------------------------------------------------- inputs ----
gen_inputs() {
    [ -f "$WORK/big.c" ] || python3 - "$WORK" <<'PY'
import os, random, sys
w = sys.argv[1]
random.seed(20260915)

# A few thousand lines of ordinary C for the compiler to chew on.
with open(os.path.join(w, "big.c"), "w") as f:
    f.write("#include <stdlib.h>\n#include <string.h>\n#include <math.h>\n")
    f.write("typedef struct { double x, y, z; int tag; } vec_t;\n")
    for i in range(220):
        f.write(f"""
static double f{i}(const vec_t* a, const vec_t* b, int n)
{{
    double acc = {i}.5, s = 0.0;
    for (int k = 0; k < n; ++k) {{
        double dx = a[k].x - b[k].x, dy = a[k].y - b[k].y, dz = a[k].z - b[k].z;
        s = dx * dx + dy * dy + dz * dz;
        if (a[k].tag & {(i % 7) + 1}) acc += sqrt(s) * {i % 13 + 1};
        else if (s > acc) acc = 0.5 * (acc + s);
        else acc -= s / ({i % 11} + 2.0);
    }}
    return acc;
}}
static int g{i}(int* p, int n, int seed)
{{
    int h = seed ^ {i * 2654435761 % 100000};
    for (int k = 0; k + 1 < n; ++k) {{
        h = (h << 5) - h + p[k];
        switch (h & 7) {{
        case 0: p[k] += {i}; break;
        case 1: p[k] ^= h; break;
        case 2: p[k] = p[k + 1] - h; break;
        case 3: h += p[k] * 3; break;
        default: h ^= p[k] >> 2; break;
        }}
    }}
    return h;
}}""")
    f.write("\nint driver(vec_t* a, vec_t* b, int* p, int n) {\n  double d = 0; int h = 0;\n")
    for i in range(220):
        f.write(f"  d += f{i}(a, b, n); h ^= g{i}(p, n, h);\n")
    f.write("  return h + (int)d;\n}\n")

# ~4 MB of mixed-entropy text for the compressor.
words = ["".join(random.choice("abcdefghijklmnopqrstuvwxyz") for _ in range(random.randint(3, 11)))
         for _ in range(4000)]
with open(os.path.join(w, "corpus.txt"), "w") as f:
    for _ in range(90000):
        f.write(" ".join(random.choice(words) for _ in range(random.randint(4, 14))) + "\n")

# 300k unsorted records for sort, and a skewed key stream for the hash workload.
with open(os.path.join(w, "records.txt"), "w") as f:
    for i in range(300000):
        f.write("%08x %s %d\n" % (random.getrandbits(32), random.choice(words), i))
with open(os.path.join(w, "keys.txt"), "w") as f:
    hot = words[:200]
    for _ in range(400000):
        k = random.choice(hot) if random.random() < 0.3 else random.choice(words)
        f.write("%s %d\n" % (k, random.getrandbits(20)))

# A random DAG for tsort: pointer-chasing over a large irregular graph.
with open(os.path.join(w, "dag.txt"), "w") as f:
    n = 40000
    for _ in range(120000):
        a = random.randrange(n - 1)
        b = random.randrange(a + 1, n)
        f.write("n%d n%d\n" % (a, b))

# An interpreter workload: dict churn + json round-trips.
with open(os.path.join(w, "interp.py"), "w") as f:
    f.write("""
import json, random
random.seed(7)
d = {}
for i in range(120000):
    k = "k%d" % random.getrandbits(17)
    d[k] = d.get(k, 0) + i
    if i % 1000 == 0:
        s = json.dumps({kk: d[kk] for kk in list(d)[:200]})
        d["_"] = len(json.loads(s))
print(len(d))
""")
PY
}

# ------------------------------------------------------------- one trace ----
# trace <name> <command...>
trace() {
    local name=$1; shift
    local dst="$OUT/$name.vtr"
    if [ -s "$dst" ]; then echo "[skip] $name (exists)"; return; fi
    echo "[trace] $name: $*"
    local t0=$SECONDS
    local fifo="$WORK/lackey.fifo"
    rm -f "$fifo"; mkfifo "$fifo"
    # valgrind blocks opening the fifo for write until din2vtr opens it for
    # read.  Once din2vtr has its cap it exits and the watchdog stops the
    # tracer: valgrind ignores EPIPE on its log fd and would otherwise run the
    # workload to completion, which for a capped trace is pure waste.
    valgrind --tool=lackey --trace-mem=yes --log-fd=3 --fair-sched=yes \
             "$@" 3>"$fifo" 1>/dev/null 2>/dev/null &
    local vpid=$!
    "$DIN2VTR" "$dst.tmp" "$MAX" < "$fifo"
    kill -9 $vpid 2>/dev/null
    wait $vpid 2>/dev/null
    rm -f "$fifo"
    mv "$dst.tmp" "$dst"
    echo "[done] $name in $((SECONDS - t0))s  $(du -h "$dst" | cut -f1)"
}

gen_inputs

CC1=$(gcc -print-prog-name=cc1)

trace gcc      "$CC1" -quiet -O2 "$WORK/big.c" -o "$WORK/big.s"
trace xz       xz -3 -k -f -c "$WORK/corpus.txt"
trace sort     sort -k2,2 -k1,1 "$WORK/records.txt"
trace tsort    tsort "$WORK/dag.txt"
trace awkhash  awk '{c[$1] += $2} END { n = 0; for (k in c) n += c[k]; print n }' "$WORK/keys.txt"
trace python   python3 "$WORK/interp.py"

echo
ls -la "$OUT"/*.vtr
