"""SVG bar chart (recall@10 by distance from the seed, per method) from analyze.py --out JSON files (no matplotlib on the mini).

    python plot.py --json result.json --tags tagA=label tagB=label --out chart.svg --title "..."
"""
import argparse, json
ap = argparse.ArgumentParser(); ap.add_argument("--json", required=True); ap.add_argument("--tags", nargs="+", required=True); ap.add_argument("--out", required=True); ap.add_argument("--title", default="")
a = ap.parse_args()
d = json.load(open(a.json))["tags"]
bins = ["0-25", "25-100", "100-250", "250+"]
cols = ["#4c78a8", "#e45756", "#54a24b", "#b279a2", "#f58518"]
W, H, L, T, B = 760, 340, 56, 40, 70
gw = (W - L - 20) / len(bins)
labels = [t.split("=", 1) for t in a.tags]
bw = gw * 0.8 / len(labels)
o = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" font-family="sans-serif" font-size="12"><rect width="{W}" height="{H}" fill="white"/>',
     f'<text x="{W/2}" y="20" text-anchor="middle" font-size="14" font-weight="bold">{a.title}</text>']
for v in (0, .25, .5, .75, 1.0):
    y = T + (H - T - B) * (1 - v)
    o.append(f'<line x1="{L}" x2="{W-20}" y1="{y}" y2="{y}" stroke="#ddd"/><text x="{L-6}" y="{y+4}" text-anchor="end">{v:.2f}</text>')
for gi, b in enumerate(bins):
    gx = L + gi * gw + gw * 0.1
    for ti, (tag, lab) in enumerate(labels):
        s = d.get(tag, {}).get("by_dist", {}).get(b, {})
        if not s.get("n"): continue
        x = gx + ti * bw
        y = T + (H - T - B) * (1 - s["recall10"])
        o.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bw*0.9:.1f}" height="{T + (H-T-B) - y:.1f}" fill="{cols[ti % 5]}"/>')
        lo, hi = s["ci"]
        yl, yh = T + (H - T - B) * (1 - lo), T + (H - T - B) * (1 - hi)
        o.append(f'<line x1="{x+bw*0.45:.1f}" x2="{x+bw*0.45:.1f}" y1="{yl:.1f}" y2="{yh:.1f}" stroke="#222"/>')
        o.append(f'<text x="{x+bw*0.45:.1f}" y="{T + (H-T-B) + 12}" text-anchor="middle" font-size="9">n={s["n"]}</text>')
    o.append(f'<text x="{gx + gw*0.4:.1f}" y="{H-B+32}" text-anchor="middle">{b} frames from seed</text>')
for ti, (tag, lab) in enumerate(labels):
    o.append(f'<rect x="{L + ti*170}" y="{H-20}" width="12" height="12" fill="{cols[ti % 5]}"/><text x="{L + ti*170 + 16}" y="{H-10}">{lab}</text>')
o.append("</svg>")
open(a.out, "w").write("\n".join(o))
