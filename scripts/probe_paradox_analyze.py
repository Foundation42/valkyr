#!/usr/bin/env python3
"""Analyse the pre-registered paradox-signature validation sweep.

Reads JSONL traces from `<root>/run{1,2,3}/<size>_paradox.jsonl`
(produced by scripts/probe_paradox_validation.sh) and evaluates the
locked reproduction criterion:

    sign(D_3B  − D_8B)  > 0   AND
    sign(KL_8B − KL_3B) > 0

both in ≥2 of 3 runs → signature reproduced.

Reports four orthogonal metrics per (run × size):

    D median  — Shannon entropy of softmax(logits), median across
                decode tokens
    KL mean   — KL(p || null_prior) across decode tokens
    n_decode  — count of decode tokens (natural termination or 256
                cap)
    peak_pos  — fractional position within decode at which D entropy
                peaks (0 = first decode token, 1 = last). Early peak
                + no recovery = collapse signature; late or
                distributed peak = engagement signature.

Usage:
  scripts/probe_paradox_analyze.py bench/probe_paradox_validation
"""

from __future__ import annotations
import json, sys, statistics, pathlib, re

NAME_RE = re.compile(r"^(?P<size>[^_]+)_paradox\.jsonl$")
RUNS = ("run1", "run2", "run3")


def load_logits(path: pathlib.Path) -> list[dict]:
    decode = []
    with path.open() as f:
        for line in f:
            r = json.loads(line)
            if r.get("kind") != "logits":
                continue
            if r.get("prefill"):
                continue
            decode.append(r)
    return decode


def metrics_for_cell(path: pathlib.Path) -> dict:
    decode = load_logits(path)
    if not decode:
        return {"d_median": float("nan"), "kl_mean": float("nan"), "n_decode": 0, "peak_pos": float("nan")}
    ents = [r["entropy"] for r in decode]
    kls = [r["kl_null"] for r in decode]
    peak_idx = max(range(len(ents)), key=lambda i: ents[i])
    peak_pos = peak_idx / max(1, len(ents) - 1) if len(ents) > 1 else 0.0
    return {
        "d_median": statistics.median(ents),
        "kl_mean": statistics.mean(kls),
        "n_decode": len(ents),
        "peak_pos": peak_pos,
    }


def by_size_order(sizes: list[str]) -> list[str]:
    def key(s: str):
        m = re.match(r"(\d+(?:\.\d+)?)([Bb])", s)
        return float(m.group(1)) if m else 0.0
    return sorted(sizes, key=key)


def section(title: str) -> None:
    print(f"\n── {title} " + "─" * (60 - len(title)))


def main(root_arg: str) -> int:
    root = pathlib.Path(root_arg)
    if not root.is_dir():
        print(f"not a directory: {root}", file=sys.stderr)
        return 2

    # Collect metrics: cells[run][size] = {d_median, kl_mean, ...}
    cells: dict[str, dict[str, dict]] = {}
    sizes_seen: set[str] = set()
    for run in RUNS:
        run_dir = root / run
        if not run_dir.is_dir():
            print(f"missing run directory: {run_dir}", file=sys.stderr)
            return 2
        cells[run] = {}
        for jsonl in sorted(run_dir.glob("*_paradox.jsonl")):
            m = NAME_RE.match(jsonl.name)
            if not m:
                continue
            size = m["size"]
            sizes_seen.add(size)
            cells[run][size] = metrics_for_cell(jsonl)

    sizes = by_size_order(list(sizes_seen))
    if not sizes:
        print("no paradox traces found", file=sys.stderr)
        return 2

    # ── Per-run per-size raw metrics ─────────────────────────────
    section("Per-run metrics: D median / KL mean / n_decode / peak_pos")
    print(f"  {'run':<6s}{'size':<14s}{'D median':>12s}{'KL mean':>12s}{'n_decode':>10s}{'peak_pos':>10s}")
    for run in RUNS:
        for sz in sizes:
            c = cells[run].get(sz)
            if c is None:
                continue
            print(f"  {run:<6s}{sz:<14s}{c['d_median']:>12.4f}{c['kl_mean']:>12.3f}{c['n_decode']:>10d}{c['peak_pos']:>10.3f}")

    # ── Per-size aggregate (mean ± across-run range) ─────────────
    section("Per-size aggregate across 3 runs")
    print(f"  {'size':<14s}{'D median (mean / range)':>30s}{'KL mean (mean / range)':>30s}{'n_decode (mean / range)':>30s}")
    for sz in sizes:
        ds = [cells[run][sz]["d_median"] for run in RUNS if sz in cells[run]]
        ks = [cells[run][sz]["kl_mean"] for run in RUNS if sz in cells[run]]
        ns = [cells[run][sz]["n_decode"] for run in RUNS if sz in cells[run]]
        if not ds:
            continue
        d_str = f"{statistics.mean(ds):.3f} / {max(ds)-min(ds):.3f}"
        k_str = f"{statistics.mean(ks):.3f} / {max(ks)-min(ks):.3f}"
        n_str = f"{statistics.mean(ns):.0f} / {max(ns)-min(ns)}"
        print(f"  {sz:<14s}{d_str:>30s}{k_str:>30s}{n_str:>30s}")

    # ── Locked reproduction criterion ────────────────────────────
    section("Locked reproduction criterion (Llama 3B vs 8B)")
    print("  Signature reproduces in run iff BOTH:")
    print("    sign(D_3B  − D_8B)  > 0   (3B more uncertain than 8B)")
    print("    sign(KL_8B − KL_3B) > 0   (8B moves further from null prior)")
    print()

    if "3B-llama32" not in sizes_seen or "8B-llama31" not in sizes_seen:
        print("  ✗ cannot evaluate — required sizes missing")
        return 1

    pass_count = 0
    print(f"  {'run':<6s}{'D_3B':>10s}{'D_8B':>10s}{'D diff':>10s}{'KL_3B':>10s}{'KL_8B':>10s}{'KL diff':>10s}{'verdict':>14s}")
    for run in RUNS:
        c3 = cells[run].get("3B-llama32")
        c8 = cells[run].get("8B-llama31")
        if c3 is None or c8 is None:
            continue
        d_diff = c3["d_median"] - c8["d_median"]
        kl_diff = c8["kl_mean"] - c3["kl_mean"]
        passed = (d_diff > 0) and (kl_diff > 0)
        if passed:
            pass_count += 1
        verdict = "REPRODUCES" if passed else "fails"
        print(f"  {run:<6s}{c3['d_median']:>10.4f}{c8['d_median']:>10.4f}{d_diff:>+10.4f}{c3['kl_mean']:>10.3f}{c8['kl_mean']:>10.3f}{kl_diff:>+10.3f}{verdict:>14s}")

    print()
    print(f"  Reproductions: {pass_count}/3")
    if pass_count >= 2:
        print(f"  → SIGNATURE REPRODUCED (criterion: ≥2/3)")
        print(f"  → v1.2 amendment supported; paradox-as-fourth-bracket finding promotable to paper")
    else:
        print(f"  → SIGNATURE NOT REPRODUCED (criterion: ≥2/3)")
        print(f"  → original single-run finding falls within determinism-floor / generation-noise variance")
        print(f"  → null protocol: paradox row stays in expanded-sweep snapshot but is flagged 'single-run, did not reproduce'")

    # ── Auxiliary: 1B drift signature reading ────────────────────
    section("1B-llama32 drift reading (third leg of categorical argument)")
    print("  Drift = partial engagement: D between collapse and engagement,")
    print("  trajectory longer than collapse, peak_pos middle-to-late.")
    if "1B-llama32" in sizes_seen:
        d1s = [cells[run]["1B-llama32"]["d_median"] for run in RUNS if "1B-llama32" in cells[run]]
        n1s = [cells[run]["1B-llama32"]["n_decode"] for run in RUNS if "1B-llama32" in cells[run]]
        p1s = [cells[run]["1B-llama32"]["peak_pos"] for run in RUNS if "1B-llama32" in cells[run]]
        if d1s:
            print(f"  D median across 3 runs: {[f'{x:.3f}' for x in d1s]}")
            print(f"  n_decode across 3 runs:  {n1s}")
            print(f"  peak_pos across 3 runs:  {[f'{x:.3f}' for x in p1s]}")

    # ── Auxiliary: Qwen parallel test (failure-mode richness) ────
    section("Qwen parallel test: how many failure modes?")
    qwen_sizes = [s for s in sizes if "qwen" in s.lower()]
    for sz in qwen_sizes:
        ds = [cells[run][sz]["d_median"] for run in RUNS if sz in cells[run]]
        ns = [cells[run][sz]["n_decode"] for run in RUNS if sz in cells[run]]
        if ds:
            print(f"  {sz:<14s}  D=[{', '.join(f'{x:.3f}' for x in ds)}]  n=[{', '.join(str(x) for x in ns)}]")
    return 0 if pass_count >= 2 else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "bench/probe_paradox_validation"))
