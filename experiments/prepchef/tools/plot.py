#!/usr/bin/env python3
"""Plot the PrepChef result tables.

Deliberately dependency-free: the lab machine has no numpy or matplotlib, and a
hand-written SVG writer is both reproducible and diffable in git.

  usage: plot.py results/runs.csv results/
"""
import csv, math, os, sys
from collections import defaultdict

W, H = 760, 460
PAD_L, PAD_R, PAD_T, PAD_B = 74, 210, 46, 56

PALETTE = ["#3b6fd4", "#d4693b", "#3ba36b", "#a03bd4", "#c9a227",
           "#2aa1b3", "#c0392b", "#7f8c8d", "#8e44ad", "#16a085",
           "#e67e22", "#2c3e50", "#27ae60", "#d35400", "#5d6d7e"]

CSS = """
  .bg{fill:#ffffff}.fg{fill:#1b1f24}.gridline{stroke:#e3e6ea;stroke-width:1}
  .axis{stroke:#9aa3ad;stroke-width:1.2}
  text{font-family:'DejaVu Sans',system-ui,sans-serif;fill:#1b1f24}
  .ttl{font-size:16px;font-weight:600}.sub{font-size:11px;fill:#6a737d}
  .tick{font-size:10.5px;fill:#6a737d}.lbl{font-size:11.5px}
  .leg{font-size:10.5px}
  @media (prefers-color-scheme: dark){
    .bg{fill:#0f1216}.fg{fill:#e6e9ee}.gridline{stroke:#262b32}
    .axis{stroke:#5b646f}text{fill:#e6e9ee}.sub,.tick{fill:#9aa3ad}
  }
"""


def esc(s):
    return (str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))


def nice_ticks(lo, hi, n=6):
    if hi <= lo:
        hi = lo + 1.0
    raw = (hi - lo) / n
    mag = 10 ** math.floor(math.log10(raw))
    for m in (1, 2, 2.5, 5, 10):
        if raw / mag <= m:
            step = m * mag
            break
    start = math.floor(lo / step) * step
    out = []
    v = start
    while v <= hi + step * 0.5:
        if v >= lo - step * 0.5:
            out.append(round(v, 10))
        v += step
    return out


def fmt(v):
    if v == 0:
        return "0"
    a = abs(v)
    if a >= 1000:
        return f"{v:,.0f}"
    if a >= 10:
        return f"{v:.0f}"
    if a >= 1:
        return f"{v:.1f}"
    return f"{v:.2f}"


def chart(path, title, subtitle, xlabel, ylabel, series, mode="scatter",
          xlog=False, annotate=False):
    """series: [(name, [(x, y, label)])]"""
    pts = [(x, y) for _, s in series for x, y, *_ in s]
    if not pts:
        return
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    if xlog:
        xs = [math.log10(max(x, 1e-9)) for x in xs]
    xlo, xhi = min(xs), max(xs)
    ylo, yhi = min(ys), max(ys)
    if xhi == xlo:
        xhi = xlo + 1
    if yhi == ylo:
        yhi = ylo + 1
    xpad, ypad = (xhi - xlo) * 0.06, (yhi - ylo) * 0.10
    xlo, xhi = xlo - xpad, xhi + xpad
    ylo, yhi = ylo - ypad, yhi + ypad

    pw, ph = W - PAD_L - PAD_R, H - PAD_T - PAD_B

    def px(x):
        v = math.log10(max(x, 1e-9)) if xlog else x
        return PAD_L + (v - xlo) / (xhi - xlo) * pw

    def py(y):
        return PAD_T + ph - (y - ylo) / (yhi - ylo) * ph

    o = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
         f'viewBox="0 0 {W} {H}" role="img" aria-label="{esc(title)}">',
         f"<style>{CSS}</style>", f'<rect class="bg" width="{W}" height="{H}"/>',
         f'<text class="ttl" x="{PAD_L}" y="24">{esc(title)}</text>',
         f'<text class="sub" x="{PAD_L}" y="40">{esc(subtitle)}</text>']

    for t in nice_ticks(ylo, yhi):
        y = py(t)
        o.append(f'<line class="gridline" x1="{PAD_L}" y1="{y:.1f}" x2="{PAD_L+pw}" y2="{y:.1f}"/>')
        o.append(f'<text class="tick" x="{PAD_L-8}" y="{y+3.5:.1f}" text-anchor="end">{fmt(t)}</text>')
    xticks = nice_ticks(xlo, xhi)
    for t in xticks:
        x = PAD_L + (t - xlo) / (xhi - xlo) * pw
        o.append(f'<line class="gridline" x1="{x:.1f}" y1="{PAD_T}" x2="{x:.1f}" y2="{PAD_T+ph}"/>')
        lab = fmt(10 ** t) if xlog else fmt(t)
        o.append(f'<text class="tick" x="{x:.1f}" y="{PAD_T+ph+17}" text-anchor="middle">{lab}</text>')

    o.append(f'<line class="axis" x1="{PAD_L}" y1="{PAD_T+ph}" x2="{PAD_L+pw}" y2="{PAD_T+ph}"/>')
    o.append(f'<line class="axis" x1="{PAD_L}" y1="{PAD_T}" x2="{PAD_L}" y2="{PAD_T+ph}"/>')
    o.append(f'<text class="lbl" x="{PAD_L+pw/2:.0f}" y="{H-14}" text-anchor="middle">{esc(xlabel)}</text>')
    o.append(f'<text class="lbl" transform="translate(18,{PAD_T+ph/2:.0f}) rotate(-90)" '
             f'text-anchor="middle">{esc(ylabel)}</text>')

    for i, (name, s) in enumerate(series):
        c = PALETTE[i % len(PALETTE)]
        if mode == "line" and len(s) > 1:
            d = " ".join(("M" if k == 0 else "L") + f"{px(x):.1f},{py(y):.1f}"
                         for k, (x, y, *_) in enumerate(sorted(s)))
            o.append(f'<path d="{d}" fill="none" stroke="{c}" stroke-width="2" '
                     f'stroke-linejoin="round"/>')
        for x, y, *rest in s:
            o.append(f'<circle cx="{px(x):.1f}" cy="{py(y):.1f}" r="3.6" fill="{c}" '
                     f'fill-opacity="0.9"/>')
            if annotate and rest and rest[0]:
                o.append(f'<text class="tick" x="{px(x)+6:.1f}" y="{py(y)-5:.1f}">{esc(rest[0])}</text>')
        ly = PAD_T + 6 + i * 15
        if ly < H - 30:
            o.append(f'<circle cx="{W-PAD_R+12}" cy="{ly-4}" r="4" fill="{c}"/>')
            o.append(f'<text class="leg" x="{W-PAD_R+22}" y="{ly}">{esc(name)}</text>')

    o.append("</svg>")
    with open(path, "w") as f:
        f.write("\n".join(o))
    print("wrote", path)


def load(path):
    with open(path) as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for k, v in r.items():
            if k not in ("phase", "trace", "config", "context", "learner"):
                try:
                    r[k] = float(v)
                except (TypeError, ValueError):
                    r[k] = 0.0
    return rows


def main(csv_path, out_dir):
    rows = load(csv_path)
    os.makedirs(out_dir, exist_ok=True)
    traces = sorted({r["trace"] for r in rows})

    # 1. precision vs coverage, context ablation (Phase B) on each trace
    for tr in traces:
        s = defaultdict(list)
        for r in rows:
            if r["phase"] == "B" and r["trace"] == tr:
                s[r["config"]].append((100 * r["coverage"], 100 * r["accuracy"], ""))
        if s:
            chart(os.path.join(out_dir, f"B_precision_coverage_{tr}.svg"),
                  f"Precision / coverage frontier — context ablation ({tr})",
                  "one point per waste price (0.10, 0.25, 1.00); decision rule fixed",
                  "coverage (% of scored data refs)", "accuracy (% of prefetches used)",
                  sorted(s.items()), mode="line")

    # 2. net benefit vs waste price
    for tr in traces:
        s = defaultdict(list)
        for r in rows:
            if r["phase"] in ("base", "E") and r["trace"] == tr and r["waste"] > 0:
                key = r["context"] if r["context"] != "-" else r["learner"]
                s[key].append((r["waste"], r["net_per_1k"], ""))
        if s:
            chart(os.path.join(out_dir, f"net_vs_price_{tr}.svg"),
                  f"Net benefit vs waste price ({tr})",
                  "window usefulness, value +1 per useful prefetch",
                  "waste price (relative to +1 useful)", "net benefit per 1K data refs",
                  sorted(s.items()), mode="line", xlog=True)

    # 3. coverage vs state bytes (Phase B + C)
    for tr in traces:
        s = defaultdict(list)
        for r in rows:
            if r["phase"].startswith("C") and r["trace"] == tr:
                s[r["phase"]].append((max(r["state_bytes"], 1), 100 * r["coverage"], r["config"]))
        for r in rows:
            if r["phase"] == "B" and r["trace"] == tr and abs(r["waste"] - 0.25) < 1e-6:
                s["B (contexts)"].append((max(r["state_bytes"], 1), 100 * r["coverage"], r["config"]))
        if s:
            chart(os.path.join(out_dir, f"coverage_vs_state_bytes_{tr}.svg"),
                  f"Coverage vs temporal state bytes ({tr})",
                  "waste price 0.25; state bytes = live fading/history state only",
                  "temporal state (bytes, log)", "coverage (%)",
                  sorted(s.items()), xlog=True, annotate=True)

    # 4. benefit vs hot-path overhead
    for tr in traces:
        s = defaultdict(list)
        for r in rows:
            if r["phase"] in ("B", "C-heads", "C-tau", "E") and r["trace"] == tr:
                s[r["phase"]].append((max(r["ns_per_ref"], 0.01), r["net_per_1k"], ""))
        if s:
            chart(os.path.join(out_dir, f"benefit_vs_overhead_{tr}.svg"),
                  f"Net benefit vs hot-path cost ({tr})",
                  "ns per trace reference, whole evaluation loop, single core",
                  "hot-path cost (ns / reference)", "net benefit per 1K data refs",
                  sorted(s.items()))

    # 5. adaptation after a workload phase change
    s = defaultdict(list)
    for r in rows:
        if r["phase"] == "drift":
            s[r["learner"]].append((r["eval_begin"], 100 * r["coverage"], ""))
    if s:
        chart(os.path.join(out_dir, "drift_adaptation.svg"),
              "Adaptation across a workload seam (gcc → tsort at 0.50)",
              "rolling 5% evaluation windows; context fixed, learner varies",
              "position in concatenated trace", "coverage (%)",
              sorted(s.items()), mode="line")

    # 6. strict (miss-filtered) vs window-only coverage
    s = defaultdict(list)
    for r in rows:
        if r["phase"] == "B" and abs(r["waste"] - 0.25) < 1e-6:
            s[r["trace"]].append((100 * r["coverage"], 100 * r["strict_cov"], ""))
    if s:
        chart(os.path.join(out_dir, "window_vs_miss_filtered.svg"),
              "What survives a cache: window coverage vs miss-filtered coverage",
              "same runs, two usefulness rules; miss-filtered credits only prefetches that saved a real L1 miss",
              "window coverage (% of scored data refs)",
              "miss-filtered coverage (% of scored misses)",
              sorted(s.items()))


    # 7. does the action rate respond to price?  (Phase G-lite)
    s = defaultdict(list)
    for r in rows:
        if r["phase"] == "G" and r["config"].startswith("realized-ev/miss-reward-w"):
            s[r["trace"]].append((r["waste"], 100 * r["action_rate"], ""))
    if s:
        chart(os.path.join(out_dir, "G_action_rate_vs_price.svg"),
              "Speculation is self-limiting once the gate is paid real prices",
              "realised-reward gate, miss-filtered reward; action rate = prefetches per scored data ref",
              "waste price (relative to +1 for a saved miss)", "action rate (%)",
              sorted(s.items()), mode="line", xlog=True)

    # 8. the frontier that matters: real misses covered per prefetch issued
    s = defaultdict(list)
    for r in rows:
        if r["phase"] == "best":
            key = ("next-line" if r["config"] == "next-line"
                   else "PrepChef (realised-reward gate x horizon)")
            s[key].append((100 * r["action_rate"], 100 * r["strict_cov"], ""))
        elif r["phase"] == "E" and r["config"] in ("ghb-gdc", "delta-markov", "spp-lite",
                                                   "pc-stride", "last-stride"):
            s[r["config"]].append((100 * r["action_rate"], 100 * r["strict_cov"], ""))
        elif r["phase"] == "E" and r["config"].startswith("prepchef-bitty-w"):
            s["PrepChef PC-BASE (window-reward gate)"].append(
                (100 * r["action_rate"], 100 * r["strict_cov"], ""))
    if s:
        chart(os.path.join(out_dir, "strict_coverage_vs_action_rate.svg"),
              "Real misses covered vs prefetches issued (all traces)",
              "miss-filtered coverage against action rate; up and to the left is better",
              "action rate (% of scored data refs)", "miss-filtered coverage (%)",
              sorted(s.items()))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "results")
