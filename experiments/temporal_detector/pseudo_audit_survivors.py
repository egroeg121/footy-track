"""Audit of the round-2 pseudo-label filter against human GT (CPU, numpy only, harness imported read-only).

Re-runs the harness B1 (camera on, selected parameters) on RT-DETR candidates of the GT-bearing harness clips (9 test = clean,
9 dev = detector-contaminated) plus `arsenal_example` (from this experiment's RT-DETR rerun) and scores the *B1 detections*
(solver nodes, source B1_det) with conf < 0.6 that sit on a human-GT frame: all of them (the pseudo doc's 38/49 audit class)
versus the survivors of the round-2 filter (conf >= 0.45 and a conf >= 0.8 node of the clip within +-25 frames).

    python pseudo_audit_survivors.py --harness-src <wt>/src --clips-out <dir> --out audit_survivors.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from pipeline_eval import CAMERA_ON, cands_from_npz, load_harness  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--harness-src", type=Path, required=True)
    ap.add_argument("--clips-out", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--data-root", type=Path, default=Path("/mnt/storage/footy_data"))
    a = ap.parse_args()
    bl, hd, he, hs = load_harness(a.harness_src)
    evs = he.build_sets(a.data_root)
    zx = np.load(a.clips_out / "arsenal_example.npz")
    n_ex = int(zx["n_frames"])
    pp, parr = zx["players_ptr"], zx["players_xywh"].astype(float)
    players = [parr[pp[t]:pp[t + 1]].reshape(-1, 4) for t in range(n_ex)]
    shift, cut = hd.estimate_camera_shift(players)
    gt, nb, aliases, conflicts = hd.load_gt(a.data_root, "arsenal_example")
    z0 = cands_from_npz(zx, "ref")
    ex_clip = hd.ClipData(clip_id="arsenal_example", n_frames=n_ex, ptr=z0[0], xy=z0[1], conf=z0[2], wh=z0[3],
                          gt={t: g for t, g in gt.items() if t < n_ex}, no_ball=set(), cam_shift=shift, cam_cut=cut,
                          players=players, gt_aliases=aliases, gt_conflicts=conflicts)
    groups = {"test (clean frames)": [], "dev (contaminated)": [], "arsenal_example (all labelled frames)": []}
    params = hs.SolverParams(**CAMERA_ON)
    items = [(ev.clip, ev.clean_scored if ev.role == "test" else ev.all_scored, "test (clean frames)" if ev.role == "test" else "dev (contaminated)")
             for ev in evs]
    items.append((ex_clip, set(ex_clip.gt), "arsenal_example (all labelled frames)"))
    for clip, scored, g in items:
        nodes = hs.solve(clip, params)
        high = sorted(n.t for n in nodes if n.conf >= 0.8)
        for n in nodes:
            if n.t not in clip.gt or n.t not in scored or n.conf >= 0.6:
                continue
            d = math.hypot(n.xy[0] - clip.gt[n.t].xy[0], n.xy[1] - clip.gt[n.t].xy[1])
            surv = n.conf >= 0.45 and any(abs(n.t - u) <= 25 for u in high)
            groups[g].append({"clip": clip.clip_id, "t": n.t, "conf": round(n.conf, 3), "hit10": bool(d <= 10), "survivor": bool(surv)})
    out = {}
    for g, rows in list(groups.items()) + [("all three", sum(groups.values(), []))]:
        sv = [r for r in rows if r["survivor"]]
        out[g] = {"low_conf_all": {"n": len(rows), "hit10": sum(r["hit10"] for r in rows)},
                  "survivors": {"n": len(sv), "hit10": sum(r["hit10"] for r in sv)}}
    a.out.write_text(json.dumps({"summary": out, "rows": groups}, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
