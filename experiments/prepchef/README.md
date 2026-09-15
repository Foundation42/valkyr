# PrepChef — experimental lab

> *Mise en Place / Forage.* A small, non-authoritative learner that uses spare
> resources to prepare probable future work, but only when experience says the
> speculation pays for itself.

This directory is the experimental programme from the 15 September 2026
handoff, built as a standalone lab. It answers one question:

> **How far can a tiny fading-state + association + economic-null primitive
> go — and how much of the apparent result survives controls?**

Nothing here is wired into valkyr's build. It is a research harness: C++17,
no dependencies, one `make`. (The handoff's own snippets are C++; a Zig
hot-path port belongs to Phase I, after the primitive has survived its
controls, not before.)

## Layout

```
src/common.hpp      hashing, 64-byte line conventions, binary trace reader, LRU model
src/context.hpp     context representations under test (Bitty fading state + controls)
src/learner.hpp     association learners and the economic null gate
src/predictor.hpp   PrepChef + registered controls + strong prefetcher baselines
src/engine.hpp      the one place prefetches are issued, deduped, expired, credited
src/scorer.hpp      an independent scorer that shares no code with the engine
src/spectro.hpp     model-free trace spectra (no learner, no context, no gate)
src/experiment.cpp  phase drivers; every run appends a row to results/runs.csv
src/din2vtr.cpp     valgrind/lackey (or Dinero .din) -> binary trace
tools/gen_traces.sh generates the real traces
tools/concat_traces.py  builds concatenated traces for phase-change probes
tools/plot.py       dependency-free SVG plots
LABBOOK.md          hypothesis -> registered control -> result -> next experiment
results/            runs.csv (every run), spectro.csv (trace spectra),
                    arms.csv (per-horizon occupancy), plots
```

## Traces

The handoff's original result used one GCC Dinero trace. That trace is not
reachable from this machine, and a single trace could not have supported Phase F
anyway, so the lab generates its own — from **real, unmodified binaries doing
real work**, instrumented with `valgrind --tool=lackey --trace-mem=yes`:

| trace     | workload                                     | character                  |
|-----------|----------------------------------------------|----------------------------|
| `gcc`     | `cc1 -O2` compiling ~7k lines of C           | compiler / SPEC-like       |
| `xz`      | `xz -3` over 4 MB of mixed-entropy text      | streaming + dictionary     |
| `sort`    | `sort -k2,2 -k1,1` over 300k records         | merge sort, large buffers  |
| `tsort`   | `tsort` over a 120k-edge random DAG          | graph / pointer chasing    |
| `awkhash` | `awk` associative array over 400k skewed keys| hash table / KV store      |
| `python`  | CPython dict churn + JSON round-trips        | interpreter, indirection   |

Each is capped at 16M references (~12M instruction fetches, ~4M data
references), which is the same shape as the handoff's trace, an order of
magnitude longer. The record format is Dinero-equivalent
(`0`=read, `1`=write, `2`=instruction fetch) so a real `.din` file can be fed
to `din2vtr` unchanged when one is available.

Traces are not committed — they are ~128 MB each and fully reproducible:

```sh
make
tools/gen_traces.sh traces 16000000
```

## Running

```sh
make
./build/prepchef audit  traces/gcc.vtr          # Phase A: Gate A + audits
./build/prepchef all    traces/gcc.vtr          # everything, one trace
./build/prepchef spectro traces/tsort.vtr      # model-free trace spectrum
./build/prepchef lead   traces/tsort.vtr        # horizon sweep h = 1..32
PREPCHEF_PEAK_H=9 ./build/prepchef peak traces/sort.vtr
./build/prepchef g65      traces/sort.vtr      # horizon as an action dimension
./build/prepchef g65audit traces/sort.vtr      # Gate A for the multi-horizon path
./build/prepchef g65price traces/sort.vtr      # matched-action-rate frontier
./build/prepchef drift  traces/drift_gcc_tsort.vtr
python3 tools/plot.py results/runs.csv results/
```

`PREPCHEF_CSV=path` redirects the result table, so traces can be run in
parallel and the CSVs concatenated.

## Protocol (PC-BASE)

Frozen, and shared by every policy in the study so no policy can be advantaged
by its accounting:

- 64-byte lines; instruction references drive context, data references are what
  gets predicted, and the two index spaces never mix.
- First 20% of data references are an online, unscored warm-up. Learning
  continues during the scored region (this is an online prefetcher).
- Action budget: at most one prefetch per data reference (degree 1), for
  PrepChef and every baseline alike.
- A prefetch is **useful** if its line is demanded within the next 32 data
  references. Ordering at each scored reference: expire → credit → observe →
  learn → issue. A demand at exactly the window edge counts; one past it does
  not.
- One outstanding prefetch per line; each prefetch is credited at most once.
- Waste price 0.25 against a value of +1 for a useful prefetch, swept from 0.05
  to 4.00.

Alongside the handoff's window rule, every run *also* reports **miss-filtered**
metrics: a counterfactual demand-only 32 KB 8-way L1 runs in parallel, and a
prefetch earns strict credit only when the demand it served would genuinely
have missed. That is the cheap preview of Phase G, and it is the number that
matters. See `LABBOOK.md`.

## Headline

Full write-up in [`LABBOOK.md`](LABBOOK.md). In short:

1. **Gate A passes.** Three independent scorers agree on all six traces, and
   the "no future information" audit is enforced by corrupting the trace after
   the point of interest and requiring bit-identical results.
2. **The handoff's effect reproduces.** On 5 of 6 independent workloads the
   multi-time-scale fading state beats a comparably sized explicit history on
   accuracy *and* coverage simultaneously — it moves the frontier, it does not
   slide along it. The matched-random and shuffled-instruction controls
   collapse to 2–5% coverage, so the context carries real temporal information.
3. **Most of the measured benefit is not a benefit.** Forbidding the degenerate
   "prefetch the line I just touched" action halves window coverage (70.3% →
   33.1% on gcc) and changes real-miss coverage by *nothing* (29.34% either
   way). Only 1–3% of data references miss a 32 KB L1 here, so ~99% of "useful"
   prefetches under the window rule fetch a line that was already resident.
   PC-BASE's net benefit is negative on all six traces under real prices.
4. **Charge real prices and the gate starts working.** Paying the learner the
   miss-filtered reward — a four-line change, a running mean of *realised*
   reward per context — drops the action rate from 76% to 3–8%, raises
   precision by an order of magnitude, keeps ~82% of the real-miss coverage,
   and makes the action rate track the resource price smoothly. On `sort` it
   switches itself off entirely; on `tsort` it goes net-positive.
5. **Aim further ahead.** Moving the label from "next data reference" to "four
   data references later" triples the lead time *and* improves real-miss
   coverage.
6. **The horizon is a workload property, not a hyperparameter.** Sweeping
   every integer horizon h = 1..32 with nothing else changed gives each
   workload a characteristic spectrum — `sort` a razor-sharp line at h = 9/10,
   `tsort` lines at 1/4/6, `gcc` a smooth decay, `xz` nothing at all. A
   model-free control computed from the reference stream with **no learner**
   reproduces the same shape (r = +0.77 on tsort, +0.70 on gcc), so this is
   execution phase structure, not a predictor artefact.
7. **At its own horizon, `sort` inverts the ranking.** At h = 9 PrepChef covers
   58.3% of real misses against next-line's 41.3%, using **21× fewer
   prefetches**, at 30× the precision, net-positive where next-line is −59.5
   per 1K. Every neighbouring horizon is exactly 0.0%. Stable across twelve
   chronological splits. `sort` was the workload where PrepChef looked worst —
   it had been declining to play at the wrong horizon.
8. **Horizon-as-action (G65) is a good detector and a bad policy.** Making the
   action `(h, δ)` and letting the existing realised-reward gate choose among
   arms does concentrate speculation on the measured spectral peaks — given the
   same candidate set for every workload and never told where to look, it put
   58% of `sort`'s covered misses on h = 9, and recovered `tsort`'s {4, 1, 28}
   (occupancy vs spectrum: r = +0.80, +0.74, +0.72). But at matched action rate
   the best *fixed* horizon beats it on five of six traces, because exploration
   cost is linear in |H| — each new `(context, h)` slot gets one optimistic
   trial, and price cannot suppress that because it happens before any reward.
   The control intended as a floor — a **pooled** estimator, one slot per
   context — turned out to be the design: it learns from every horizon's
   evidence at single-horizon exploration cost, and on `sort` reproduces the
   hand-picked oracle exactly without being told the horizon.
9. **No novelty or performance claim is made.** Under the miss-filtered rule,
   plain next-line still covers more real misses than anything else here. What
   survives is a narrower claim about efficiency per action.
