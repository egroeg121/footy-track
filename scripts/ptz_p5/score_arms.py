"""Score fixed-rig re-solves of the 10 GT segments (dirs ``arsenal_mancity_segNNN@<name>-<hash8>``) with the P3 scorer and pair
them against arm A (GT rig, contaminated ceiling) and arm B (existing 95-frame fit).  Usage: score_arms.py NAME [NAME...]
Writes reports/ptz_v5_assets/score/<NAME>.json and prints one summary line per NAME + paired Wilcoxon vs B and A."""
import glob, json, sys
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

sys.path.insert(0, "scripts/ptz_p3")
import score_gt as S
from footy_track.scripts import ptz_validate as pv

ROOT = "/mnt/storage/footy_data/scratch/temporal"
SEGS = (10, 20, 31, 50, 60, 80, 90, 104, 140, 150)
ASSETS = Path("reports/ptz_v4_assets")
OUT = Path("reports/ptz_v5_assets/score")
OUT.mkdir(parents=True, exist_ok=True)
gt = [f for f in pv.load_gt(("mancity",)) if f.clip.startswith("arsenal_mancity_seg")]


def frames_of(name, segs=SEGS):
    recs = []
    for i in segs:
        ds = glob.glob(f"{ROOT}/arsenal_mancity_seg{i:03d}@{name}-*")
        if not ds:
            continue
        r = S.score_dir(ds[0], i, [f for f in gt if f.clip == f"arsenal_mancity_seg{i:03d}"])
        recs += r or []
    return recs


def meds(recs):
    return {(r["seg"], r["frame"]): r["med"] for r in recs if r.get("answered")}


A = json.load(open(ASSETS / "A10.json"))["frames"]
B = json.load(open(ASSETS / "B.json"))["frames"]
for name in sys.argv[1:]:
    recs = frames_of(name)
    s = S.summarise(recs)
    json.dump({"label": name, "summary": s, "frames": recs}, open(OUT / f"{name}.json", "w"))
    m = meds(recs)
    line = {"name": name, **{k: s.get(k) for k in ("frames", "answered")}}
    for lab, base in (("B", B), ("A", A)):
        mb = meds(base)
        keys = sorted(set(m) & set(mb))
        d = np.array([m[k] - mb[k] for k in keys])
        p = wilcoxon(d).pvalue if len(d) > 5 and np.any(d != 0) else float("nan")
        line[f"vs_{lab}"] = {"n": len(keys), "mean_diff_m": float(d.mean()), "median_diff_m": float(np.median(d)), "worse": int((d > 0).sum()),
                            "better": int((d < 0).sum()), "p": float(p)}
    print(json.dumps({**line, "summary": s}, default=float))
