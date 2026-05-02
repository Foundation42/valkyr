#!/usr/bin/env python3
"""Substrate-relativity analysis for valkyr probe traces.

Reads JSONL traces produced by `valkyr --chat --probe <path>` and
extracts the three RFF-spec predictions:

  1. D(t) entropy gap (philosophical − factual) by model size.
  2. Per-layer B(t) profile shift across sizes.
  3. KL-from-null trajectory by (size × prompt regime).

Plus a C-ranking sanity check: factual < creative < philosophical at
every model size (RFF section 4.3 ordered exchanges).

Usage:
  scripts/probe_analyze.py bench/probe_sweep
"""

from __future__ import annotations
import json, sys, statistics, pathlib, re

# Discover traces. Filenames follow `<size>_<prompt>.jsonl` from
# probe_sweep.sh — e.g. `1B-llama32_factual.jsonl`.
NAME_RE = re.compile(r"^(?P<size>[^_]+)_(?P<prompt>[^.]+)\.jsonl$")

PROMPT_ORDER = ["factual", "creative", "philosophical"]


def load(path: pathlib.Path) -> tuple[dict, list[dict], list[dict], list[dict]]:
    header = None
    acts = []
    logs = []
    attns = []
    with path.open() as f:
        for line in f:
            r = json.loads(line)
            k = r.get("kind")
            if k == "header":
                header = r
            elif k == "act":
                acts.append(r)
            elif k == "logits":
                logs.append(r)
            elif k == "attn":
                attns.append(r)
    return header, acts, logs, attns


def discover(root: pathlib.Path) -> dict[str, dict[str, pathlib.Path]]:
    out: dict[str, dict[str, pathlib.Path]] = {}
    for p in sorted(root.glob("*.jsonl")):
        m = NAME_RE.match(p.name)
        if not m:
            continue
        out.setdefault(m["size"], {})[m["prompt"]] = p
    return out


def by_size_order(sizes: list[str]) -> list[str]:
    """Sort sizes by parameter count parsed from the prefix."""
    def key(s: str):
        m = re.match(r"(\d+(?:\.\d+)?)([Bb])", s)
        return float(m.group(1)) if m else 0.0
    return sorted(sizes, key=key)


def fmt(x: float, w: int = 8, d: int = 4) -> str:
    return f"{x:>{w}.{d}f}"


def section(title: str) -> None:
    print(f"\n── {title} " + "─" * (60 - len(title)))


def main(root_arg: str) -> int:
    root = pathlib.Path(root_arg)
    if not root.is_dir():
        print(f"not a directory: {root}", file=sys.stderr)
        return 2

    by_size = discover(root)
    if not by_size:
        print(f"no traces found in {root}", file=sys.stderr)
        return 2

    sizes = by_size_order(list(by_size.keys()))

    # ── 1. D(t) entropy gap by size ──────────────────────────────
    section("D(t) median entropy by (size × prompt) — Prediction 1")
    print(f"  {'size':<14s}" + "".join(f"{p:>14s}" for p in PROMPT_ORDER) + f"{'phil-fact':>14s}")
    medians: dict[str, dict[str, float]] = {}
    for sz in sizes:
        row = []
        medians[sz] = {}
        for prompt in PROMPT_ORDER:
            p = by_size[sz].get(prompt)
            if p is None:
                medians[sz][prompt] = float("nan")
                row.append(f"{'-':>14s}")
                continue
            _, _, logs, _ = load(p)
            decode = [r for r in logs if not r["prefill"]]
            if not decode:
                medians[sz][prompt] = float("nan")
                row.append(f"{'-':>14s}")
                continue
            med = statistics.median([r["entropy"] for r in decode])
            medians[sz][prompt] = med
            row.append(fmt(med, 14, 4))
        gap = medians[sz].get("philosophical", float("nan")) - medians[sz].get("factual", float("nan"))
        row.append(fmt(gap, 14, 4))
        print(f"  {sz:<14s}" + "".join(row))

    # ── 2. Per-layer B(t) profile shift by size ──────────────────
    section("B(t) per-layer entropy_norm — mean across all tokens, all prompts")
    layer_profile: dict[str, dict[int, list[float]]] = {}
    for sz in sizes:
        layer_profile[sz] = {}
        for prompt in PROMPT_ORDER:
            p = by_size[sz].get(prompt)
            if p is None:
                continue
            _, acts, _, _ = load(p)
            for r in acts:
                layer_profile[sz].setdefault(r["layer"], []).append(r["entropy_norm"])
    # Print as size columns × layer rows.
    all_layers = sorted({L for sz in sizes for L in layer_profile[sz]})
    print(f"  {'layer':>6s}" + "".join(f"{sz:>14s}" for sz in sizes))
    for L in all_layers:
        row = f"  {L:>6d}"
        for sz in sizes:
            vs = layer_profile[sz].get(L, [])
            row += f"{fmt(statistics.mean(vs), 14, 4) if vs else '-':>14s}" if not vs else fmt(statistics.mean(vs), 14, 4)
        print(row)
    # Final-layer entropy change relative to second-to-last. The
    # original framework framing predicted a drop here ("final-layer
    # output-projection compression"); we observe a lift in every
    # cell. The label below reflects what the column actually shows
    # rather than what was originally predicted.
    section("Final-layer entropy_norm jump (final − penultimate)")
    print(f"  {'size':<14s}{'penult':>14s}{'final':>14s}{'jump':>14s}")
    for sz in sizes:
        layers = sorted(layer_profile[sz].keys())
        if len(layers) < 2:
            continue
        last = layers[-1]
        penult = layers[-2]
        a = statistics.mean(layer_profile[sz][penult])
        b = statistics.mean(layer_profile[sz][last])
        print(f"  {sz:<14s}{fmt(a,14,4)}{fmt(b,14,4)}{fmt(b-a,14,4)}")

    # ── 3. KL-from-null trajectory (mean across decode) ──────────
    section("KL(p||null_prior) mean during decode — Prediction 3 (substrate-relativity)")
    print(f"  {'size':<14s}" + "".join(f"{p:>14s}" for p in PROMPT_ORDER))
    for sz in sizes:
        row = []
        for prompt in PROMPT_ORDER:
            p = by_size[sz].get(prompt)
            if p is None:
                row.append(f"{'-':>14s}")
                continue
            _, _, logs, _ = load(p)
            decode = [r for r in logs if not r["prefill"]]
            if not decode:
                row.append(f"{'-':>14s}")
                continue
            row.append(fmt(statistics.mean([r["kl_null"] for r in decode]), 14, 3))
        print(f"  {sz:<14s}" + "".join(row))

    # ── K(t) attention-entropy by (size × prompt) ────────────────
    #
    # Decode-only mean of attn.entropy_norm across (token, layer, head).
    # Skip n_pos == 1 rows (degenerate single-key attention has entropy
    # 0 by construction). Reported normalized by log(n_pos) so the
    # absolute scale is comparable across positions.
    section("K(t) attention entropy_norm — decode-only mean across (layer × token)")
    print(f"  {'size':<14s}" + "".join(f"{p:>14s}" for p in PROMPT_ORDER))
    k_medians: dict[str, dict[str, float]] = {}
    for sz in sizes:
        row = []
        k_medians[sz] = {}
        for prompt in PROMPT_ORDER:
            p = by_size[sz].get(prompt)
            if p is None:
                row.append(f"{'-':>14s}")
                continue
            _, _, _, attns = load(p)
            sample = [r["entropy_norm"] for r in attns if not r["prefill"] and r["n_pos"] > 1]
            if not sample:
                row.append(f"{'-':>14s}")
                continue
            m = statistics.mean(sample)
            k_medians[sz][prompt] = m
            row.append(fmt(m, 14, 4))
        print(f"  {sz:<14s}" + "".join(row))

    # Top-weight (concentration) — lower means attention is spread,
    # higher means a few keys dominate.
    section("K(t) mean top attention weight — decode-only")
    print(f"  {'size':<14s}" + "".join(f"{p:>14s}" for p in PROMPT_ORDER))
    for sz in sizes:
        row = []
        for prompt in PROMPT_ORDER:
            p = by_size[sz].get(prompt)
            if p is None:
                row.append(f"{'-':>14s}")
                continue
            _, _, _, attns = load(p)
            sample = [r["top"] for r in attns if not r["prefill"] and r["n_pos"] > 1]
            if not sample:
                row.append(f"{'-':>14s}")
                continue
            row.append(fmt(statistics.mean(sample), 14, 4))
        print(f"  {sz:<14s}" + "".join(row))

    # K-axis substrate-relativity test: does the philosophical-creative
    # gap on K mirror the threshold pattern we saw on D? Specifically:
    # is K higher on philosophical than creative for 7B but not for
    # smaller substrates?
    section("K(t) ordering: phil vs creative by size")
    for sz in sizes:
        ph = k_medians[sz].get("philosophical", float("nan"))
        cr = k_medians[sz].get("creative", float("nan"))
        delta = ph - cr
        sign = "phil > creative" if delta > 0 else "creative > phil"
        print(f"  {sz:<14s}  phil={fmt(ph)}  creative={fmt(cr)}  Δ={fmt(delta)}  ({sign})")

    # ── C ordering sanity check ──────────────────────────────────
    section("RFF C-ordering check: factual < creative < philosophical?")
    for sz in sizes:
        m = medians[sz]
        ok = m.get("factual", 9e9) < m.get("creative", 9e9) < m.get("philosophical", -9e9)
        print(f"  {sz:<14s}  factual={fmt(m.get('factual',float('nan')))}  creative={fmt(m.get('creative',float('nan')))}  philosophical={fmt(m.get('philosophical',float('nan')))}  → {'PASS' if ok else 'FAIL'}")

    # ── Predictions summary ──────────────────────────────────────
    section("Verdict")
    if len(sizes) >= 2:
        gaps = []
        for sz in sizes:
            g = medians[sz].get("philosophical", float("nan")) - medians[sz].get("factual", float("nan"))
            gaps.append((sz, g))
        gap_grows = all(gaps[i][1] <= gaps[i + 1][1] for i in range(len(gaps) - 1))
        print(f"  Prediction 1 (entropy gap grows with size): {'CONSISTENT' if gap_grows else 'NOT CONSISTENT'}")
        print(f"    gaps by size: {', '.join(f'{s}={fmt(g,8,4)}' for s,g in gaps)}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "bench/probe_sweep"))
