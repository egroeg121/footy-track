"""Audit of the pseudo labels: hit@10 against human GT by source, near-GT counts, contact sheets.

usage: pseudo_audit.py --prep <pp dir> --fuse <fuse dir> --dataset <multiframe_pseudo_v1> --out <doc dir>
Writes audit.json and contact_sheet_{hard,easy}.png into --out.
"""
from __future__ import annotations

import argparse
import gzip
import json
import math
import random
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np


def band(r: dict) -> str:
    if r["source"] == "B1_det":
        c = r["det_conf"]
        return "B1_det conf<0.6" if c < 0.6 else ("B1_det conf 0.6-0.8" if c < 0.8 else "B1_det conf>=0.8")
    if r["source"] == "S1_fill":
        return f"S1_fill {r['rule']}"
    return "B1_interp"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prep", required=True)
    ap.add_argument("--fuse", required=True)
    ap.add_argument("--dataset", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    pool = json.loads((Path(a.prep) / "pool_b1.json").read_text())
    gt = {c: {int(t): v for t, v in d["gt"].items()} for c, d in pool.items()}
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    # ---- hit@10 by source over every tracked frame that has human GT --------------------------
    agg = defaultdict(lambda: {"n": 0, "hit5": 0, "hit10": 0, "hit20": 0, "errs": []})
    with gzip.open(Path(a.fuse) / "provenance_all.jsonl.gz", "rt") as f:
        for line in f:
            r = json.loads(line)
            g = gt.get(r["clip"], {}).get(r["t"])
            if g is None:
                continue
            e = math.hypot(r["x"] - g[0], r["y"] - g[1])
            for key in (band(r), "ALL " + r["source"], "ALL"):
                s = agg[key]
                s["n"] += 1
                s["hit5"] += e <= 5
                s["hit10"] += e <= 10
                s["hit20"] += e <= 20
                s["errs"].append(e)
    table = {k: {"n": v["n"], "hit@5": v["hit5"] / v["n"], "hit@10": v["hit10"] / v["n"], "hit@20": v["hit20"] / v["n"],
                 "median_err_px": float(np.median(v["errs"]))} for k, v in sorted(agg.items())}
    # ---- the same for the SELECTED frames ----------------------------------------------------------
    sel = [json.loads(x) for x in (Path(a.fuse) / "selection.jsonl").read_text().splitlines() if x.strip()]
    sel_tab = defaultdict(lambda: {"n": 0, "hit10": 0, "on_gt_frame": 0})
    near = defaultdict(int)
    for r in sel:
        cat = r["category"]
        sel_tab[cat]["n"] += 1
        near[cat] += int(r["near_human_gt_5"])
        g = gt.get(r["clip"], {}).get(r["t"])
        if g is not None:
            sel_tab[cat]["on_gt_frame"] += 1
            sel_tab[cat]["hit10"] += math.hypot(r["x"] - g[0], r["y"] - g[1]) <= 10
    res = {"hit_by_source_all_tracked_frames_with_gt": table,
           "selected": {k: {**v, "within_5_frames_of_human_gt": near[k]} for k, v in sel_tab.items()},
           "selected_total": len(sel), "selected_within_5_frames_of_human_gt": sum(near.values()),
           "selected_on_a_human_gt_frame": sum(v["on_gt_frame"] for v in sel_tab.values())}
    (out / "audit.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))
    # ---- contact sheets -------------------------------------------------------------------------
    if not a.dataset:
        return
    rng = random.Random(a.seed)
    ds = Path(a.dataset)
    groups = {"hard": [r for r in sel if r["category"].startswith("hard")], "easy": [r for r in sel if r["category"] == "easy"]}
    S, Z, CAP = 64, 3, 14
    for name, rs in groups.items():
        rng.shuffle(rs)
        tiles = []
        for r in rs:
            iid = f"{r['clip']}_{r['t']:06d}"
            p = ds / "images" / "train" / f"{iid}.tiff"
            if not p.exists():
                continue
            ok, pages = cv2.imdecodemulti(np.fromfile(str(p), np.uint8), cv2.IMREAD_UNCHANGED)
            if not ok:
                continue
            img = cv2.merge(pages[1:4])  # B, G, R of frame t
            x0 = int(min(max(round(r["x"] - S / 2), 0), 1920 - S))
            y0 = int(min(max(round(r["y"] - S / 2), 0), 1080 - S))
            crop = cv2.resize(img[y0:y0 + S, x0:x0 + S], None, fx=Z, fy=Z, interpolation=cv2.INTER_NEAREST)
            bx0, by0 = (r["x"] - r["w"] / 2 - x0) * Z, (r["y"] - r["h"] / 2 - y0) * Z
            bx1, by1 = (r["x"] + r["w"] / 2 - x0) * Z, (r["y"] + r["h"] / 2 - y0) * Z
            cv2.rectangle(crop, (int(bx0), int(by0)), (int(bx1), int(by1)), (0, 255, 255), 1)
            cap = np.zeros((CAP, S * Z, 3), np.uint8)
            c = r.get("det_conf")
            txt = f"{r['source'][:2]}{r['source'][2:] if r['source'].startswith('S1') else ''} " + (f"c{c:.2f}" if c is not None else f"{r.get('rule', '')[:3]} d{r['dist_anchor']}")
            cv2.putText(cap, txt, (2, 11), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1, cv2.LINE_AA)
            tiles.append(np.vstack([crop, cap]))
            if len(tiles) == 48:
                break
        while len(tiles) < 48:
            tiles.append(np.zeros_like(tiles[0]))
        rows = [np.hstack(tiles[i:i + 8]) for i in range(0, 48, 8)]
        cv2.imwrite(str(out / f"contact_sheet_{name}.png"), np.vstack(rows))
        print("wrote", out / f"contact_sheet_{name}.png")


if __name__ == "__main__":
    main()
