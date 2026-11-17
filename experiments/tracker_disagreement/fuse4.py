"""Add the labeller's own tracker (autodetect + VitTrack, ``vittrack_track.py``) to fused.json.gz as a fourth source ``t4`` (mini, numpy-free).

    python fuse4.py --fused fused.json.gz --npz CLIPS_OUT --vit DIR --arm 1|2 [--min-score 0.2] --out fused4_arm2.json.gz

arm 2 (comparable; per-seed forward + backward VitTrack runs on the shared seed file): per frame the S1 average of a forward track of one seed and a backward track of the next within 5 px,
else the nearest-seed track (same rule as the SAM trackers).  arm 1 (labeller-realistic): the rows of the autodetect -> Run -> hand-back -> re-detect loop, one row per frame.
Every VitTrack frame has a box (a lost track keeps its last box with a low score), so ``x, y`` hold the position only when the score is >= ``--min-score`` (default 0.20 = VitTrack's own
"found" threshold) and the raw position is always kept in ``raw_x, raw_y`` with ``score``.  Also stored: ``near``, ``in_player_box`` (round-3 RT-DETR player boxes of the clip) so drift onto
a player can be counted, and for arm 2 ``would_stop`` (the labeller's own hand-back rule: confidence below the ball threshold or ``_track_anomaly_reason``).
"""
import argparse, gzip, json, math
from pathlib import Path

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--fused", type=Path, required=True)
ap.add_argument("--npz", type=Path, required=True)
ap.add_argument("--vit", type=Path, required=True)
ap.add_argument("--arm", type=int, required=True)
ap.add_argument("--min-score", type=float, default=0.20)
ap.add_argument("--out", type=Path, required=True)
a = ap.parse_args()
d = json.load(gzip.open(a.fused, "rt"))
for clip, cl in d["clips"].items():
    n = cl["n"]
    z = np.load(a.npz / f"{clip}.npz")
    pp, parr = z["players_ptr"], z["players_xywh"].astype(float)
    p = a.vit / f"{clip}.labels.jsonl"
    t4 = {k: [None] * n for k in ("x", "y", "raw_x", "raw_y", "score", "gate", "dist", "would_stop", "kind", "run")}
    t4["is_seed"] = [0] * n
    t4["near"], t4["in_player_box"] = [0] * n, [0] * n
    if not p.exists():
        cl["t4"] = None
        continue
    rows = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
    by = {}
    for r in rows:
        by.setdefault(r["frame"], []).append(r)
    for f, rs in by.items():
        if f >= n:
            continue
        if a.arm == 1:
            r = rs[-1]
            gate = r["kind"]
        else:
            fw = [r for r in rs if r.get("dir") == "fwd" and r["dist"] > 0]
            bw = [r for r in rs if r.get("dir") == "bwd" and r["dist"] > 0]
            r, gate = None, "single"
            for x in sorted(fw, key=lambda r: r["dist"]):
                for y in sorted(bw, key=lambda r: r["dist"]):
                    if x["seed"] != y["seed"] and math.hypot(x["x"] - y["x"], x["y"] - y["y"]) <= 5.0:
                        r = {**x, "x": (x["x"] + y["x"]) / 2, "y": (x["y"] + y["y"]) / 2, "score": min(x["score"], y["score"]), "would_stop": x.get("would_stop") or y.get("would_stop")}
                        gate = "s1"
                        break
                if r:
                    break
            if r is None:
                cands = [r for r in rs if r["dist"] > 0] or rs
                r = min(cands, key=lambda r: r["dist"])
                gate = "seed" if r["dist"] == 0 else "single"
        t4["raw_x"][f], t4["raw_y"][f], t4["score"][f] = round(r["x"], 2), round(r["y"], 2), r["score"]
        t4["gate"][f], t4["dist"][f], t4["would_stop"][f], t4["kind"][f], t4["run"][f] = gate, r.get("dist"), r.get("would_stop"), r.get("kind"), r.get("run")
        t4["is_seed"][f] = int(gate == "seed" or r.get("kind") == "seed" and a.arm == 2 and r.get("dist") == 0)
        if r["score"] >= a.min_score or gate == "seed":
            t4["x"][f], t4["y"][f] = t4["raw_x"][f], t4["raw_y"][f]
        pl = parr[pp[f]:pp[f + 1]].reshape(-1, 4)
        if len(pl):
            dd = np.hypot(pl[:, 0] - r["x"], pl[:, 1] - r["y"])
            t4["near"][f] = int((dd < 60).sum())
            t4["in_player_box"][f] = int(np.any((abs(pl[:, 0] - r["x"]) < pl[:, 2] / 2 + 3) & (abs(pl[:, 1] - r["y"]) < pl[:, 3] / 2 + 3)))
    cl["t4"] = t4
d["vit"] = {"arm": a.arm, "min_score": a.min_score}
json.dump(d, gzip.open(a.out, "wt"))
print("wrote", a.out, "t4 frames with raw position", sum(sum(x is not None for x in cl["t4"]["raw_x"]) for cl in d["clips"].values() if cl["t4"]), "with position", sum(sum(x is not None for x in cl["t4"]["x"]) for cl in d["clips"].values() if cl["t4"]))
