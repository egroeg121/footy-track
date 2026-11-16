"""Score a ``sam3_label.py`` / ``sam3-label-match`` output directory against GT (mini CPU): hit rate (10 px, 1080p-equiv) per gate class on the GT frames that got a label,
and coverage of GT frames by gate.

    python eval_labels.py --spec spec.json --labels DIR [--clips a,b]
"""
import argparse, json, math
from pathlib import Path
ap = argparse.ArgumentParser(); ap.add_argument("--spec", required=True); ap.add_argument("--labels", required=True); ap.add_argument("--clips", default="")
a = ap.parse_args()
spec = json.load(open(a.spec))
tot = {}
n_gt = 0
for p in sorted(Path(a.labels).glob("*.labels.jsonl")):
    clip = p.name.replace(".labels.jsonl", "")
    if a.clips and clip not in a.clips.split(","): continue
    c = spec["clips"][clip]; sc = 1080.0 / c["h"]
    rows = {}
    for line in p.read_text().splitlines():
        r = json.loads(line); rows[r["p"]] = r
    pr = [r["p"] for r in rows.values()]
    lo, hi = (min(pr), max(pr)) if pr else (0, -1)
    gts = [(float(k), v) for k, v in c["gt"].items()] + [(int(k) + 0.5, v) for k, v in c.get("gt_odd", {}).items()]
    for g, box in gts:
        if not (lo <= g <= hi): continue
        n_gt += 1
        gx, gy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
        if abs(g - round(g)) < 1e-6: cand = rows.get(int(round(g)))
        else:
            r0, r1 = rows.get(int(g)), rows.get(int(g) + 1)
            cand = None if not (r0 and r1) else {"x": (r0["x"] + r1["x"]) / 2, "y": (r0["y"] + r1["y"]) / 2, "gate": min((r0["gate"], r1["gate"]), key=["seed", "s1", "score", "weak"].index)}
        if cand is None:
            tot.setdefault("none", []).append(None); continue
        tot.setdefault(cand["gate"], []).append(math.hypot(cand["x"] - gx, cand["y"] - gy) * sc <= 10)
print(f"GT frames inside the labelled span: {n_gt}")
for k in ("seed", "s1", "score", "weak", "none"):
    v = tot.get(k, [])
    if v: print(f"  gate {k:6s}: {len(v):4d} GT frames", "" if k == "none" else f" hit@10 = {sum(v)/len(v):.3f}")
