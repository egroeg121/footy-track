"""Markdown tables for reports/ptz_v5_rig_search.md from reports/ptz_v5_assets/{rigs,score}."""
import glob, json
import numpy as np
from scipy.stats import wilcoxon

A = "reports/ptz_v5_assets"
fits = {json.load(open(f))["name"]: json.load(open(f)) for f in glob.glob(f"{A}/rigs/*.fit.json")}
fc = json.load(open(f"{A}/fixed_cost.json"))
base = {k: json.load(open(f"reports/ptz_v4_assets/{k}.json")) for k in ("A10", "B")}
sc = {}
for f in glob.glob(f"{A}/score/*.json"):
    d = json.load(open(f)); sc[d["label"]] = d


def meds(fr):
    return {(r["seg"], r["frame"]): r["med"] for r in fr if r.get("answered")}


def row(name, d, ref=None):
    s = d["summary"]
    f = s["frame_median_m"]; w = s["within_pct"]
    cells = [name, f"{s['answered']}/{s['frames']}", f"{f['median']:.2f} / {f['p90']:.2f} / {f['p95']:.2f}",
             f"{w['1m']:.0f} / {w['2m']:.0f} / {w['5m']:.0f}"]
    for k in ("near", "mid", "far"):
        cells.append(f"{s[k]['median']:.2f} / {s[k]['p95']:.2f}" if k in s else "-")
    mb = meds(base["B"]["frames"]); m = meds(d["frames"]); ks = sorted(set(m) & set(mb))
    dd = np.array([m[k] - mb[k] for k in ks])
    p = wilcoxon(dd).pvalue if len(dd) > 5 and np.any(dd != 0) else float("nan")
    cells.append(f"{dd.mean():+.2f} ({int((dd>0).sum())}w/{int((dd<0).sum())}b, p={p:.3f})")
    return "| " + " | ".join(cells) + " |"


hdr = "| rig | answered | frame med / p90 / p95 (m) | within 1/2/5 m (%) | near med / p95 | mid med / p95 | far med / p95 | paired vs B (mean m; worse/better; p) |\n|---|---|---|---|---|---|---|---|"
print(hdr)
for k, lab in (("A10", "A (GT rig, contaminated ceiling)"), ("B", "B (existing 95-frame fit)")):
    d = base[k]; s = d["summary"]
    print(row(lab, d))
for n in sorted(sc, key=lambda n: (fits[n]["k1_mode"], fits[n]["start"], fits[n]["arm"]) if n in fits else ("", "", n)):
    print(row(n, sc[n]))
print()
print("| rig | start | arm | k1 | C (x, y, z) | dist to ref (m) | k1 | roll deg | LM iters | CPU-s | data cost at rig |\n|---|---|---|---|---|---|---|---|---|---|---|")
for n in sorted(fits, key=lambda n: (fits[n]["k1_mode"], fits[n]["start"], fits[n]["arm"])):
    d = fits[n]
    print(f"| {n} | {d['start']} ({d['start_dist_ref']:.1f} m) | {d['arm']} | {d['k1_mode']} | {d['C'][0]:.2f}, {d['C'][1]:.2f}, {d['C'][2]:.2f} | {d['dist_ref_m']:.2f} | {d['k1']:.4f} | {d['roll_deg']:.2f} | {d['iters']} | {d['cpu_s']:.0f} | {fc[n]['data_cost']:.0f} |")
