# PrepChef lab book

Entries are `hypothesis → registered control → result → interpretation → next
experiment`. Every number below is a row in `results/runs.csv` (1,476 runs).
Null and negative results are kept, as instructed.

**Setting.** Six real memory-reference traces, 16M references each, generated
with `valgrind --tool=lackey` from unmodified binaries (`cc1 -O2`, `xz -3`,
`sort`, `tsort`, `awk` associative array, CPython). The handoff's GCC Dinero
trace is not reachable from this machine; a single trace could not support
Phase F anyway. PC-BASE protocol, prices and usefulness rule are as specified
in the handoff, frozen in `src/engine.hpp`.

**One addition to the protocol, made before any result was read.** Every run
also reports **miss-filtered** metrics. A counterfactual, demand-only 32 KB
8-way L1 runs alongside; a prefetch earns strict credit only if the demand it
served would genuinely have missed. The handoff's third load-bearing rule is
"charge real prices", and the 32-reference window rule does not: it pays +1 for
prefetching a line that was already sitting in L1. Columns prefixed `strict_`
are that measurement.

---

## A — Freeze, reproduce, audit

**Hypothesis.** The scoring in the original experiment is sound.

**Registered controls.** Three independent scorers (the online engine; an
offline index/binary-search scorer sharing no code with it, `src/scorer.hpp`;
and a naive O(n·W) forward scan), plus the six audits the handoff names.

**Result — Gate A passes** on every trace (`results/audit.txt`):

```
  [PASS] Gate A: independent scorer matches engine
  [PASS] Gate A: independent lead sum matches
  [PASS] duplicate outstanding fetches suppressed       violations=0
  [PASS] brute-force scorer matches independent scorer
  [PASS] no prefetch credited twice (bounds)
  [PASS] no future information enters the context
  [PASS] scored region invariant to post-hoc trace corruption
  [PASS] warm-up issues nothing and earns nothing
  [PASS] instruction/data reference accounting closes
  [PASS] instruction and data feature namespaces are separated
  [PASS] demand at exactly the window edge is useful
  [PASS] demand one past the window edge is waste
```

Two audits are worth calling out because they are stronger than "check the
counters". *No future information* is tested by corrupting every reference
after position *p* and asserting the context at *p* is bit-identical; *scored
region invariant* corrupts the trace beyond the scored region plus its window
and asserts the headline counts are unchanged. A leak of either kind would
survive a counter audit and die here.

Two bugs were caught and fixed by this phase before any result was read: the
event log mixed global and scored-region data indices (Gate A failed loudly,
which is what it is for), and Phase C's τ-set labels contained commas that
corrupted the CSV.

**Ordering is now normative** (`src/engine.hpp`): expire → observe →
credit → learn → issue. Expiry strictly precedes credit, so a demand at lead
`W` counts and one at `W+1` does not; both are pinned by a hand-checked
synthetic trace.

**Splits.** Warm-up fractions 0.10/0.20/0.35/0.50 and eight rolling 10%
windows (`A-split`, `A-roll`) all reproduce the effect. No random shuffling of
a temporal trace anywhere.

---

## B — Context ablation. **Gate B passes in the handoff's metric, and fails in
the miss-filtered one.**

**Hypothesis.** A bounded multi-time-scale fading state beats comparably sized
explicit histories.

**Registered controls.** Random context with matched cardinality; shuffled
instruction stream (same marginal distribution, no temporal order).

**Result.** The controls collapse, so the context does carry real temporal
information: on gcc the fading state reaches 70.3% coverage, matched-random
reaches 3.2% and shuffled-instruction 4.9%.

Waste price 0.25, all six traces, window metric:

| trace   | bitty 4×8 acc / cov | explicit-4 acc / cov | bitty better? |
|---------|---------------------|----------------------|---------------|
| gcc     | 92.5% / 70.3%       | 84.9% / 66.6%        | yes           |
| xz      | 94.2% / 78.4%       | 88.6% / 67.9%        | yes           |
| tsort   | 94.7% / 66.5%       | 89.7% / 62.0%        | yes           |
| awkhash | 97.3% / 75.2%       | 91.5% / 70.0%        | yes           |
| sort    | 97.7% / 17.8%       | 93.7% / 17.2%        | yes (small)   |
| python  | 84.0% / 46.5%       | 81.9% / 57.6%        | **no**        |

The handoff's central qualitative claim reproduces: on 5 of 6 independent
workloads the fading state is better on **both** axes at once, i.e. it moves
the frontier rather than sliding along it. It also does so at 137 B of state
against explicit-16's 137 B and explicit-8's 73 B, while beating both.

**But** on the same runs, miss-filtered coverage tells a different story:
bitty wins on gcc (29.3% vs 26.9%), loses on tsort (22.0 vs 23.2), python
(12.3 vs 15.3) and awkhash (1.7 vs 3.9), and ties on sort and xz. **The fading
state's advantage is specific to the metric it was developed under.** Recorded
as a negative result for Gate B under real prices.

Component ablations (gcc): dropping the data delta costs 1.8 points of
coverage, dropping access type costs 0.15. Feeding the fading state with data
lines instead of instruction lines raises accuracy (96.2%) and halves coverage
(37.1%); feeding it both is worse than instructions alone on real misses
(6.5% vs 29.3% strict coverage). **Instruction lines are the load-bearing
input.**

---

## The finding that reframes everything: most of the measured benefit is not a
## benefit

**Registered control.** Forbid the degenerate action "prefetch the line I just
touched" (`E-noself`), which the window rule rewards and which prepares
nothing.

**Result (gcc, waste 0.25).** Window coverage falls 70.3% → 33.1%. Strict
coverage is **unchanged at 29.34%**, to four significant figures.

More than half of the headline coverage was a delta-0 action that cannot
possibly have staged anything. It contributes exactly zero real-miss coverage.
(Incidentally, 33.1% is very close to the handoff's reported 33.0%. That may
be coincidence — but if the original protocol did exclude self-prefetches,
the like-for-like figure here is 33.1%, not 70.3%.)

The second half of the gap is residency. On these traces only 1.0–3.1% of
scored data references miss a 32 KB 8-way L1, so ~99% of "useful" prefetches
under the window rule fetch a line that was already there. Strict accuracy for
PC-BASE is **0.01%–0.96%** across the six traces, and strict net benefit is
**negative on all six** (−45 to −208 per 1K data refs at waste 0.25).

**Interpretation.** The window usefulness rule is not a weak proxy for the real
thing; it is anti-correlated with it above a certain action rate, because it
charges nothing for redundancy. Phase G was not a "later" phase. It was load
bearing from the start.

---

## G-lite — Charge real prices, and let experience see them

**Hypothesis.** An economic null gate makes speculation self-limiting under
resource pressure.

**Registered control.** The PC-BASE gate (`counts-eu`) run under the
miss-filtered *reward*. It gates on predicted `p`, not on realised reward, so
if the feedback channel matters it must not change. It does not change: its
rows under both reward modes are byte-identical. Good — the control behaves.

**Change under test.** Replace the expected-utility gate with a running mean of
**realised** reward per context (`RealizedEV`, four lines in
`src/learner.hpp`): counts still choose the action, a single scalar decides
whether acting here pays at all.

**Result (miss-filtered reward, action rate = prefetches per scored data ref):**

| trace | waste | action rate | strict acc | strict cov | strict net/1K |
|-------|-------|-------------|-----------|-----------|---------------|
| gcc   | —(PC-BASE) | 75.9%  | 0.40%     | 29.3%     | −186.0 |
| gcc   | 0.05  | 7.7%        | 3.25%     | 24.1%     | −1.2   |
| gcc   | 0.25  | 2.9%        | 5.79%     | 16.1%     | −5.1   |
| gcc   | 1.00  | 0.9%        | 15.4%     | 13.6%     | −6.4   |
| tsort | 0.05  | 3.7%        | 15.3%     | 18.5%     | **+4.1** |
| sort  | 0.05  | 0.1%        | 7.1%      | 0.6%      | +0.0   |

Three things happen at once, on all six traces:

1. **The action rate falls smoothly and monotonically with the price**
   (`results/G_action_rate_vs_price.svg`). gcc: 7.7 → 5.1 → 2.9 → 1.6 → 0.9%
   for prices 0.05 → 1.00. This is the behaviour the handoff asks Phase G to
   test for, and it is present.
2. **Precision rises by an order of magnitude** while most of the real
   coverage survives: gcc keeps 82% of its miss coverage using 10% of the
   actions.
3. **On `sort`, it switches itself off** (0.03–0.09% action rate, net ≈ 0)
   while next-line burns −59.5/1K on the same trace. "Doing nothing is always
   an action" is not decoration; it is the result.

**Interpretation.** The economic null gate does not work because of the gate
formula. It works when it is *paid the real price*. The formula
`p·value − (1−p)·waste` is a prediction of reward, and a prediction of the
wrong reward is worse than useless — it is confidently wrong. The smallest fix
is not a better estimator but a feedback channel.

---

## H — How far ahead can it actually see?

**Result.** PC-BASE's mean useful lead is 1.1–1.6 **data** references, which is
4.4–5.7 total references. As the handoff suspected, that is far too late for
DRAM and marginal even for L2.

**Change under test.** Move the label, not the model: learn *context → the line
demanded `horizon` data references later*. One ring buffer, no new machinery.

| horizon | gcc lead (total refs) | gcc window cov | gcc strict cov |
|---------|-----------------------|----------------|----------------|
| 1       | 5.3                   | 70.3%          | 29.3%          |
| 2       | 8.1                   | 58.3%          | 31.6%          |
| 4       | 14.2                  | 44.5%          | **37.9%**      |
| 8       | 25.4                  | 37.4%          | 15.3%          |
| 32      | 58.0                  | 29.4%          | 14.1%          |

Window coverage falls monotonically and **real-miss coverage rises** to a peak
at horizon 4 — 2.7× the lead and better real coverage than PC-BASE. The two
metrics disagree about the direction of progress, which is the same lesson as
above from a different angle.

**Open anomaly, recorded not explained.** On tsort, horizons 1 and 4 work
(strict coverage 22.0% and 34.2%) while 2 and 8 collapse (0.48%, 1.08%). The
swing is far too large for the ~97k-miss denominator to be noise, so something
structural in the traversal aligns with horizon 4. Not yet understood.

---

## The one configuration that ends up net-positive

Phases G-lite and H each fix one half, and had not been run together.
Combined — realised-reward gate × label horizon, miss-filtered reward:

| trace | config | action rate | strict acc | strict cov | strict net/1K | lead |
|-------|--------|-------------|-----------|-----------|---------------|------|
| gcc   | ev, h=4, w=0.05 | 7.4%  | 4.9%  | 35.0% | +0.12 [^1] | 14.8 |
| tsort | ev, h=4, w=0.25 | 2.3%  | 43.2% | 31.6% | **+6.52** | 18.5 |
| tsort | ev, h=4, w=0.05 | 4.1%  | 24.9% | 32.9% | **+8.59** | 16.3 |
| sort  | ev, h=1, w=0.05 | 0.09% | 7.1%  | 0.6%  | +0.02 | 6.9 |
| gcc   | next-line       | 32.2% | 1.5%  | 47.1% | −74.4 | 60.9 |
| tsort | next-line       | 31.6% | 3.3%  | 33.7% | −66.1 | 60.9 |

On tsort this comes within 2 points of next-line's real-miss coverage (31.6%
vs 33.7%) using **14× fewer prefetches**, and is the only policy in the entire study with positive net
benefit under real prices. On sort it correctly declines to play. On gcc it is
barely positive and next-line still covers more misses (47.1% vs 35.0%) — at
4.4× the traffic and a net of −74.

[^1]: Superseded — the horizon-spectroscopy entry below shows this gcc figure
does not survive the rolling windows. The net-positive workloads are `tsort`
and `sort`.

**This is the honest headline:** *the primitive becomes economically real only
when it is paid real prices and aimed further ahead, and even then it wins on
efficiency per action, not on raw coverage.*

---

## C — Representation sweep

Pareto-relevant results (gcc, waste 0.25, window metric):

- **Heads.** 1→8 improves coverage 56.5%→70.3%; 16/32/64 make it *worse*
  (69.2/67.2/64.5) while state grows to 1,033 B. Contexts become too specific.
  **4 heads at 73 B captures ~99.7% of the coverage of 8 heads.**
- **Quantisation is load-bearing.** sign 70.0%, ternary 70.3%, 2-bit **71.1%**,
  3-bit 69.4%, int8 54.5%, float 19.1%. The float reference — no quantisation
  at all — is the *worst* configuration by a factor of 3.7. Coarse buckets are
  not an approximation of the real thing; they are the mechanism.
- **Ternary threshold** is flat from 0.05 to 0.50 (69.7–71.2%). Not a tuned
  knob.
- **τ sets.** `{2,4,8,16,32,64}` is marginally best on 3 of 6 traces
  (gcc 71.4% / strict 30.2%) at 201 B; `{2,8,32,128}` is within a point at
  137 B; `{1,2,4,8}` (short scales only) is clearly worse on gcc
  (66.8%/26.3%). Multiple scales matter; the exact set does not.
- **Integer Q8.8 with shift-only updates matches float**: 70.1% vs 70.3%
  coverage at **half the state bytes** (73 B vs 137 B). No reason to keep
  floats.

---

## D — Learner ablation. The smallest mechanism wins.

Nothing beats cumulative counts on the window metric (gcc, waste 0.25):
decayed counts 0.99/0.999 (−0.04/−0.7 coverage), top-k (k=8 → k=1 costs 4.1
points; **k=1 retains 94% of coverage and the best strict coverage, 30.1%**),
bandit (−2.5), fixed-confidence gates (strictly worse net at every threshold
tried).

- **`min_evidence = 6` is not optimal**: 1 gives 74.5% coverage and 32.1%
  strict coverage vs 70.3%/29.3%. The threshold is costing real coverage.
- **Delta outcomes beat absolute-line outcomes decisively on real misses**:
  29.3% vs 3.65% strict coverage. Absolute lines memorise; deltas generalise.
- **Table capacity**: 2^16 slots × 4 actions (4.0 MB) is within 0.4% of 2^22
  (457 MB at 8 slots). 2^14 loses 8 points. Eviction counts are reported for every run;
  the headline configuration evicts 0.
- **Concept drift (gcc→tsort seam, `drift_adaptation.svg`)**: coverage drops
  76.7% → 61% at the seam and recovers to ~67% within one 5% window (≈400k
  data refs). Cumulative counts, decayed counts, top-k and the bandit are
  **indistinguishable** (within 1.5 points everywhere). Recovery is fast
  because the two workloads occupy nearly disjoint context space — the fading
  state of a different instruction working set is a different fingerprint, so
  there is little to unlearn. **Registered null result: drift robustness does
  not motivate a decayed or bandit learner here.**

---

## E — Strong baselines

At matched action budget (degree 1), window metric, PrepChef leads on 5 of 6
traces (on python by only 9 points of net/1K); `delta-markov` is much closer than next-line or stride ever were
(gcc 89.3%/52.1% vs 92.5%/70.3%), and `ghb-gdc` beats PrepChef on `sort`
(69.3% vs 17.8% coverage).

Under the miss-filtered metric the ranking inverts: **plain next-line has the
best real-miss coverage of any Phase E policy on all six traces** — gcc 47.1%,
python 43.0%, sort 41.3%, tsort 33.7%, awkhash 7.8%, xz 3.2% — because the misses that remain after a 32 KB L1
are overwhelmingly streaming. No correlation-based policy in this study —
PrepChef, delta-Markov, GHB or SPP-lite — is net-positive under real prices
with the PC-BASE gate. Per the handoff's own instruction, **no novelty or
performance claim is made.** The claim that survives is narrower and is about
efficiency per action, not coverage.

---

## I — Hot-path cost

Single-process measurement, ns per *trace reference* (`cost` phase; the
null predictor measures the evaluation loop itself at 6.5–7.6 ns):

| configuration | gcc ns/ref | state | leaky updates / ref |
|---|---|---|---|
| next-line | 12.2 | 0 B | 0 |
| ghb-gdc | 16.0 | — | 0 |
| explicit-4 | 44.1 | 41 B | 0 |
| bitty 4×8 (PC-BASE) | 114.3 | 137 B | 24.1 |
| bitty 4×4 | 77.8 | 73 B | 12.0 |
| bitty 4×8, integer | 109.5 | 73 B | 24.1 |
| bitty 4×8, distinct instruction lines only | 72.8 | 137 B | 2.8 |
| bitty 4×4, distinct + integer | 56.1 | 41 B | 1.4 |
| **selected** (4×4 distinct+int, EV gate, h=4, 2^16×4 table) | **47.8** | **41 B** | 1.4 |

**The cheapest win in the study**: integrating only when the *instruction line
changes* rather than on every fetch cuts leaky-state work 8.5× at 8 heads (17×
at 4 heads) and wall cost by 36%, for ~7 points of window coverage. Most consecutive fetches repeat a
line, so most updates were doing nothing.

**The uncomfortable number**: PC-BASE costs ~9× next-line in wall time (23×
after subtracting the 7.6 ns evaluation loop), and next-line covers more real
misses. `coverage vs state bytes` flatters the
primitive (137 B is genuinely tiny); `benefit vs hot-path cost` does not,
because the fading state must be updated per *instruction* while the payoff is
per *data reference*, and there are 2.6–3.0 instruction fetches per data
reference in every trace here.

---

## Horizon spectroscopy — the anomaly is phase structure, and it is worth money

*(Follow-up entry. Christian's instruction: sweep h = 1..32 with nothing else
changed, do not explain the tsort anomaly away, and do not touch the
representation.)*

**Hypothesis.** The non-monotonic tsort result (h ∈ {1,4} good, {2,8} bad) is
a property of the workload's execution dynamics, not an artefact of the
predictor.

**Registered control — a model-free spectrum.** `src/spectro.hpp` computes,
from the reference stream alone with no learner, no context and no gate:

- `recur(h)` = P(L[i+h] = L[i])
- `miss_lift(h)` = P(m[i+h] | m[i]) / P(m)
- `top1nz_mass(h)` = max over d ≠ 0 of P(L[i+h] − L[i] = d)
- **`top1nz_miss(h)`** = P(L[i+h] − L[i] = d\* **and** m[i+h]) / P(m) — the
  fraction of real misses a single global non-zero delta at lag *h* would
  cover. Delta 0 is excluded because a delta-0 "preparation" prepares nothing
  and can never land on a miss; leaving it in, it dominates every lag and the
  statistic says nothing. (The self-prefetch artefact shows up here too.)

**Result 1 — the control has the same shape as the learned result** (tsort):

| h | model-free ceiling `top1nz_miss` (best Δ≠0) | PrepChef strict coverage |
|---|---|---|
| 1 | 21.4% (Δ = −1) | 22.0% |
| 2 | **0.00%** | 0.48% |
| 3 | 0.00% | 0.55% |
| 4 | **33.1% (Δ = +1)** | 34.2% |
| 5 | 0.00% | 12.2% |
| 8 | 0.55% | 1.08% |
| 28 | 6.42% | 6.84% |

Correlation between the model-free curve and PrepChef's strict coverage across
all 32 lags: **tsort r = +0.77, gcc r = +0.70**, awkhash +0.46, sort +0.34,
python +0.05, xz −0.43. Where there is structure, the learner tracks it; where
there is none (python is broadband, xz is flat) the statistic is uninformative,
which is the honest reading rather than a failure.

**Not an artefact, and not ordinary prediction distance.** The window metric
decays smoothly with h on tsort (66.5 → 62.5 → 51.4 → 46.2 → 34.4 → 27.2%)
while real-miss coverage does not. Only the miss-filtered view sees the
structure, because the structure is in *which references miss*, not in which
references are predictable.

**Result 2 — the spectra are workload signatures, and they differ in kind:**

| trace | shape | horizons with positive net (realised-reward gate, w = 0.05) |
|-------|-------|---------------------------------------------------------------|
| `sort` | **razor-sharp line spectrum** | 9, 10, 18, 21, 23, 25, 27, 29 |
| `tsort` | line spectrum | 1, 3, 4, 5, 6, 7, 10, 28 |
| `gcc` | smooth decay + a weak lobe near 4–5 and 14–15 | 4 (marginal, see below) |
| `python` | broadband, gently peaked at 6 and 15–16 | none |
| `awkhash` | h = 1 only | none |
| `xz` | flat — there is nothing to predict | none |

**Result 3 — and this is the one that matters. `sort`, the workload where
PrepChef looked worst, is where it looks best, at the right horizon.**

| h | strict cov | strict acc | action rate | strict net/1K | lead (total refs) |
|---|-----------|-----------|-------------|---------------|-------------------|
| 8  | 0.00%  | 0.00%  | 0.079% | −0.04 | 16.8 |
| **9**  | **58.26%** | **46.83%** | **1.226%** | **+5.42** | **43.9** |
| **10** | 58.26% | 46.83% | 1.226% | +5.42 | 42.9 |
| 11 | 0.00%  | 0.00%  | 0.079% | −0.04 | 37.4 |
| 18 | 41.74% | 31.27% | 1.316% | +3.66 | 79.3 |
| next-line (for comparison) | 41.32% | 1.58% | 25.8% | **−59.5** | 60.9 |

At h = 9 PrepChef covers **more real misses than next-line (58.3% vs 41.3%)
using 21× fewer prefetches, at 30× the precision, and is net-positive where
next-line is deeply negative.** Every neighbouring horizon is *exactly* zero.

Earlier in this lab book, "on `sort` it switches itself off" was recorded as a
success for the null action. It still is — but the fuller reading is that it
was declining to play **at the wrong horizon**. The opportunity was there the
whole time, nine references away.

**Robustness (`peak` phase, 108 runs).** A peak this sharp needs checking, so
it was re-run across four warm-up fractions and eight rolling windows with
h±1 as controls:

- `sort` h = 9: strict coverage **58.1–59.2%** across all twelve splits, net
  +5.18 to +5.42; h = 8 is **0.0% in all twelve**. Stable and razor-edged.
- `tsort` h = 4: 32.5–35.4%, net +3.46 to +8.67; h = 3 and h = 5 clearly
  lower. Stable.
- `gcc` h = 4: 23.9–45.1%, net −4.86 to +1.45 — **the h = 4 advantage is
  inside the split-to-split variation, and the net is negative in most rolling
  windows.**

**Correction to an earlier entry.** The combined-configuration table above
lists gcc at h = 4 with strict net +0.12. That figure is real for the default
split but it does **not** survive the rolling windows, so gcc should not be
counted as a net-positive workload. The net-positive results in this lab are
`tsort` and `sort`, not gcc.

**Interpretation.** The horizon is not a hyperparameter with a good global
value. It is a property of the host's execution dynamics — the distance at
which the future becomes both predictable *and* expensive — and it differs by
an order of magnitude between workloads (1 for `awkhash`, 4 for `tsort`, 9 for
`sort`, ~6 for `python`, none at all for `xz`). A primitive frozen at h = 1
does not merely under-perform on `sort`; it sees nothing there and correctly
concludes that nothing is worth doing.

This also sharpens what the fading state is for. It answers "when things feel
like this". The spectrum answers a second, separable question: "*how far
ahead* does feeling like this tell you anything". Those are different
quantities and the current primitive only learns the first.

**Next experiment (designed, deliberately not built).** Christian asked to
investigate before touching the architecture, so this is a proposal, not a
result. Make the horizon part of the action rather than a constant: keep a
small set of candidate horizons *H* = {1, 2, 4, 8, 16, 32}, run one delay ring
and one association per h (cheap: the context and its hash are computed once
and shared), and let the **existing** realised-reward gate choose among them —
it already selects on realised value, and "act at horizon h" is just another
arm. Two properties make this attractive: the per-workload spectra above are
sparse, so the gate would be choosing between a handful of live arms and a
great many dead ones, which is the regime a running-mean estimator handles
well; and it needs no new mechanism, only more arms. The registered controls
should be (a) the best fixed h per trace, as the ceiling, and (b) a uniform
random horizon at matched action rate, as the floor. The risk to watch is that
six arms multiply the table and the update cost by six, against a primitive
whose hot-path cost is already its weakest column.

**Also frozen, as instructed.** The Bitty representation (4 banks × 8 heads,
ternary at ±0.20) is not touched by any of this and will not be optimised
further until the horizon question is settled.

---

## G65 — Multi-horizon PrepChef: horizon as an action dimension

*(Registered follow-up. Frozen: Bitty representation, realised-reward gate,
prices, protocol. Changed: exactly one thing — `a = δ` becomes `a = (h, δ)`,
and the existing gate chooses among arms with the null action still
competing.)*

**Registered hypothesis.** *If horizon spectroscopy reflects exploitable
temporal structure in expensive events, a realised-reward PrepChef given
multiple horizon arms should (i) concentrate speculative activity near
independently measured spectral peaks and (ii) approach the best fixed-horizon
policy without being told those peaks.*

**Result: (i) is supported. (ii) is falsified.**

### Implementation, and why it is honestly "one thing"

- One association table keyed by `hash(context, h)`, at the *same* capacity the
  single-horizon learner uses. Arms compete for the same slots, so the
  multi-horizon system gets no memory advantage by construction.
- One delay ring of length max(H), read back h steps — not |H| rings, which
  would have been a strawman cost.
- The context and its hash are computed **once** per data reference and shared
  by every arm, so the expensive part (the fading-state update, which runs per
  *instruction*) is not multiplied.

**Audit first.** The per-arm reporting added to the engine was verified
semantically neutral: `audit.txt` is byte-identical before and after. A result
this large needs its own audit, so Gate A, the duplicate-suppression invariant
and the post-hoc-corruption test were re-run against the multi-horizon
predictor at full resolution (`g65audit`) and all pass — on `sort` the engine
and the independent scorer agree at 117,393 issued / 84,011 useful.

### (i) The learner finds the spectrum unaided

Share of real misses covered by each arm, `h32` fixture, against the
model-free spectrum measured with no learner:

| trace | corr r | top arms by share of real misses covered | model-free peaks |
|-------|--------|------------------------------------------|------------------|
| tsort | **+0.80** | h4:23% h1:16% h28:13% h3:13% h10:10% | h4, h1, h28 |
| gcc | +0.74 | h2:33% h5:20% h3:12% h1:9% | h1–h5, h27 |
| python | +0.72 | h2:16% h6:14% h1:9% h4:8% | h1, h3, h6, h5 |
| sort | +0.52 | **h9:58%** h29:41% | h9 (its only non-zero lag) |
| awkhash | +0.31 | h10:23% h13:12% h1:11% | h1 |
| xz | +0.24 | h4:35% h2:17% | (spectrally empty) |

Given the same candidate set for every workload and never told where to look,
the gate put 58% of `sort`'s covered misses on **h = 9** — the single lag the
model-free statistic flags — and recovered `tsort`'s {4, 1, 28} including the
obscure h = 28. The learned action distribution is an empirical resource-demand
spectrum, as hoped. On the two spectrally uninteresting traces (`awkhash`,
`xz`) agreement is weak, which is the honest reading rather than a failure:
there is no spectrum to agree with.

### (ii) It cannot pay for itself

At w = 0.05, `sort` looks spectacular — `h32` reaches **99.9% real-miss
coverage** at a 3.9% action rate, net +8.39, beating the hand-picked h = 9
oracle. That does not generalise. Swept across prices 0.05 → 256 and compared
**at matched action rate**, the best fixed horizon beats multi-horizon
selection on **five of six traces**:

| trace | best multi-horizon | best single fixed h | winner |
|-------|--------------------|---------------------|--------|
| sort | h32 +8.39 @ 3.9% act | h = 9: +5.42 @ 1.2% act | multi (at w = 0.05 only) |
| tsort | coarse +8.88 @ 11.1% act | h = 4: +8.59 @ 4.1% act | **single** (2.7× fewer actions) |
| gcc | coarse −5.81 | h = 4: **+0.12** | **single** |
| python | coarse −6.52 | h = 9: −2.31 | **single** |
| awkhash | coarse −3.66 | h = 1: −0.82 | **single** |
| xz | coarse −8.10 | h = 9: −2.03 | **single** |

### Why — diagnosed, after one wrong guess

My first diagnosis was that an optimistic prior turns `argmax` over |H| arms
into a maximisation bias dominated by *untested* arms. I implemented the fix
(`BestEVTested`: an untested arm cannot win the argmax; one designated explorer
per context may fire on its prior) and **it did nothing** — action rates were
unchanged or marginally worse on every trace. Recorded as a failed repair.

The real mechanism is visible in an independent counter. At a punitive price
(w = 256, where nothing profitable should fire) the action rate hits a floor
that scales with the number of arms:

| trace | policy | action rate | table inserts | evictions |
|-------|--------|-------------|---------------|-----------|
| gcc | single h = 1 | 0.88% | 174,934 | 0 |
| gcc | h32 pooled | 1.49% | **174,934** | 0 |
| gcc | coarse (6 arms) | 3.95% | 1,049,541 (**6.0×**) | 58 |
| gcc | h32 (32 arms) | 11.38% | 4,082,267 (table-capped) | **1,589,860** |

**The floor is exploration cost, and it is linear in |H|.** Each new
`(context, h)` slot gets one optimistic trial, and |H| arms create |H|× as many
slots. Price cannot suppress it, because exploration happens *before* any
reward is observed — which is also why barring untested arms did not help: it
still allows one explorer per context, and there are now 32× as many contexts.
At |H| = 32 the shared table additionally thrashes (1.59M evictions), and an
evicted slot is reset to the prior and explores again.

### The result that is actually useful

Look at the pooled control's insert count: **174,934 — identical to
single-horizon.** Because it keys one slot per context rather than per
(context, h), it learns from every horizon's evidence at *single-horizon
exploration cost*. And on `sort` it reproduces the oracle **exactly** — 58.26%
coverage, 1.227% action rate, net +5.42, the same three figures as the
hand-picked h = 9 — without being told the horizon and without paying for 32
arms. It is the best-behaved multi-arm variant on four of six traces.

The control intended as a floor turned out to be the design.

### Cost

Hot-path cost scales with |H| as expected and worse than my predicted 6×: on
gcc, 112 ns/ref (single) → 289 (coarse) → 2,050 (h32), an 18× regression
against the primitive's already-weakest column. The fading-state update is
correctly not multiplied; the |H| table probes per data reference are.

### Interpretation

Horizon-as-action is a good **detector** and a bad **policy**. The evidence
that the spectrum is real and learnable is strong — the occupancy correlations
are the cleanest confirmation in this lab that the gate is not lying. But
selection among |H| arms costs |H| explorations per context, and on these
workloads that exceeds the value of choosing correctly, except where a single
peak carries almost everything (`sort`).

**Next experiment (designed, not built).** Separate the two questions the
current design conflates. Gate on a *pooled* estimator — one slot per context,
single-horizon exploration cost, deciding only "is it worth acting here at
all?" — and consult the per-h deltas only once that gate says yes, choosing
among them by evidence already accumulated rather than by optimistic trial.
That has pooled's exploration cost and best-ev's resolution, which is the
combination neither variant has. Registered controls: pooled alone (which we
now know is strong) and best fixed h at matched action rate.

**Still frozen.** Bitty untouched throughout.

---

## Threats to validity

1. **L1 only.** The counterfactual is a 32 KB 8-way L1 with no L2/LLC, no
   MSHRs, no bandwidth limit and **no pollution accounting** — a prefetch never
   evicts anything. Real Phase G would make the negative numbers worse, not
   better, since pollution is currently free.
2. **Latency is not modelled.** "Useful" still means "demanded inside the
   window", not "arrived in time". The lead numbers (Phase H) are the input to
   that question, not an answer to it.
3. **16M-reference slices.** Working sets have less time to grow than in a full
   run, which biases miss rates *down* and therefore biases the miss-filtered
   results against speculation. The direction is known; the magnitude is not.
4. **Traces are not the handoff's trace.** The qualitative claim reproduces on
   six new workloads, which is stronger evidence than reproducing on the same
   trace, but the specific numbers in the handoff are not reproduced and should
   not be compared line by line.
5. **Learning continues during the scored region** (this is an online
   prefetcher). Warm-up is unscored and cannot leak reward; that is audited.
6. **Spectral peaks are measured at one line size and one cache geometry.**
   Which references miss is what creates the structure, so a different L1
   capacity or associativity would move the peaks. The peaks are a property of
   the workload *and* the memory system, not of the workload alone.
7. **The multi-horizon exploration cost is specific to an optimistic-prior
   gate.** A gate that could decide "not worth trying" without trying would not
   pay it. The finding is about this gate, not about horizon selection in
   general.
8. `signed_hash_features` is a splitmix64 bit-extraction, not necessarily the
   original hash. Any deterministic ±1 hash should behave the same, but this
   has not been verified against the original.

---

## Next experiments, in order

1. **Real memory economics (Phase G proper).** Give the counterfactual an L2
   and LLC, finite MSHRs, a bandwidth budget, and charge pollution. The
   realised-reward gate already has the right shape to consume
   `stall_cycles_saved − bandwidth − pollution − energy` as its reward; only
   the reward function changes.
2. **Latency-aware credit (Phase H proper).** Convert lead into simulated
   cycles and split the action space into `now / soon / later`. The horizon
   sweep says the label can be aimed; it does not yet say where to aim it.
3. **Resolve the tsort horizon anomaly** (h∈{1,4} work, h∈{2,8} collapse)
   before trusting the horizon result.
4. **Run `min_evidence = 1` and `top-k = 1` as the new default**; both are
   simpler than PC-BASE and better on real misses.
5. **Move the state update off the instruction stream.** Distinct-line-only
   already gives 17×; try taken branches, call/return sites, or basic-block
   boundaries as the breadcrumb, which is what a real implementation would have
   cheap access to anyway.
6. **Only then**, the other domains (Radix/Substrate, Matryoshka asset
   staging). The primitive that should be carried across is not PC-BASE; it is
   *fading state + counts + a gate paid in realised reward*, which is the part
   that survived.

---

## Verdict on the working hypothesis

> *Many systems expose cheap temporal breadcrumbs before expensive resources are
> consumed. A bounded multi-timescale fading state can identify recurring
> temporal situations without retaining explicit histories. A small online
> association learner can exploit those situations to stage resources just in
> time, while an economic null gate makes speculation self-limiting under
> uncertainty or resource pressure.*

Clause by clause, against these six workloads:

- *cheap temporal breadcrumbs* — **supported.** Instruction lines predict the
  next data line well; the matched-random and shuffled controls collapse.
- *bounded fading state without explicit histories* — **supported in the
  window metric** (5/6 traces, both axes at once, 41–137 B), **not supported
  in the miss-filtered metric** (wins 1 of 6). Gate B passes as written and
  fails as it should have been written.
- *stage resources just in time* — **not supported as configured.** 1.1–1.6
  data references of lead is not just in time for anything below L1. Movable:
  horizon 4 gives 14–18 references and better real coverage.
- *economic null gate makes speculation self-limiting* — **supported, with a
  correction.** It is self-limiting only when the gate is paid the realised
  price. Given that, the action rate tracks price smoothly across four orders
  of magnitude, the policy switches itself off on the workload where
  speculation does not pay, and it becomes net-positive on the workload where
  it does.

The research question was *how far can a tiny fading-state + association +
economic-null primitive go?* On this evidence: far enough to be worth the next
phase, and not yet far enough to be a claim. The most valuable thing the lab
found is not about the estimator or the representation at all — it is that
**rule 3 of the handoff ("charge real prices") is not a later refinement of
rules 1 and 2; it is the precondition that decides whether they mean anything.**
