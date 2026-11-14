"""Prep for the independent check clip ``arsenal_example`` (no ``machine_labels`` exist for it).

Candidates come from ``detect_clip.py`` (same detector weights, imgsz 1280, conf 0.1; regenerated, so
not byte-identical to the original machine_labels producer). The clip is not in the detector's training
data, so there is no contamination exclusion: every human GT ball frame is scored.

usage: machine_prep_example.py --det <dir with arsenal_example.jsonl> --out <dir>   (writes prep_example.json, spec_example.json)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from footy_track.ball_offline import baselines as bl  # noqa: E402
from footy_track.ball_offline.data import ClipData, _load_machine, data_root, estimate_camera_shift, load_gt  # noqa: E402
from footy_track.ball_offline.solver import SolverParams  # noqa: E402
from machine_common import CONFIGS, REACH, thin, tracks_for  # noqa: E402

CLIP = "arsenal_example"
N_FRAMES = 250


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--det", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--results", default=str(ROOT / "docs/experiments/offline_ball_baseline.json"))
    args = ap.parse_args()
    res = json.loads(Path(args.results).read_text())
    cam = res["B1"]["camera_selected_on_dev"]
    p1 = SolverParams(**res["B1"]["tuned_params"]["camera_on" if cam else "camera_off"])
    tau = res["B0"]["tau_selected_on_dev"]
    root = data_root()
    ptr, xy, conf, wh, players = _load_machine(Path(args.det) / f"{CLIP}.jsonl", N_FRAMES)
    shift, cut = estimate_camera_shift(players)
    gt, no_ball, aliases, conflicts = load_gt(root, CLIP)
    c = ClipData(CLIP, N_FRAMES, ptr, xy, conf, wh, {t: g for t, g in gt.items() if t < N_FRAMES},
                 {t for t in no_ball if t < N_FRAMES}, shift, cut, players, aliases, conflicts)
    pred, nodes = bl.b1_solver(c, p1)
    b0 = bl.b0_per_frame(c, tau)
    nodes_j = []
    for n in nodes:
        xy_, cf = c.cands(n.t)
        k = int(np.argmin(np.hypot(xy_[:, 0] - n.xy[0], xy_[:, 1] - n.xy[1]))) if len(xy_) else None
        s, e = c.ptr[n.t], c.ptr[n.t + 1]
        wh_ = c.wh[s:e][k] if k is not None else np.array([16.0, 16.0])
        w, h = float(wh_[0]), float(wh_[1])
        nodes_j.append({"t": n.t, "xy": list(n.xy), "conf": n.conf, "linked": n.linked,
                        "box": [n.xy[0] - w / 2, n.xy[1] - h / 2, n.xy[0] + w / 2, n.xy[1] + h / 2]})
    scored = sorted(set(c.gt) | c.no_ball)
    cuts = [int(t) for t in np.where(cut)[0]]
    pj = {"role": "indep", "n_frames": N_FRAMES, "video": str(root / "arsenal_mancity_example_video.mp4"),
          "gt": {str(t): list(g.xy) for t, g in c.gt.items()}, "gt_tier": {str(t): g.tier for t, g in c.gt.items()},
          "no_ball": sorted(c.no_ball), "scored": scored, "contaminated": [],
          "b0": {str(t): list(v) for t, v in b0.items()}, "b1": {str(t): list(v) for t, v in pred.items()},
          "nodes": nodes_j, "cuts": cuts, "not_broadcast": [], "out_of_play": []}
    pf = {int(t) for t in pred}
    need = {}
    for thr, sp in CONFIGS:
        for tr in tracks_for(thin(nodes_j, pf, thr, sp), N_FRAMES):
            for d, reach in (("fwd", tr["fwd"]), ("bwd", tr["bwd"])):
                if reach < 1:
                    continue
                k = (tr["t"], d)
                if k not in need or need[k]["reach"] < reach:
                    need[k] = {"t": tr["t"], "dir": d, "reach": reach, "box": tr["box"]}
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "prep_example.json").write_text(json.dumps({"b1_params": p1.__dict__, "b0_tau": tau, "reach": REACH,
                                                       "configs": CONFIGS, "clips": {CLIP: pj}}))
    (out / "spec_example.json").write_text(json.dumps({"reach": REACH, "clips": {CLIP: {
        "video": pj["video"], "n_frames": N_FRAMES, "tracks": list(need.values()), "full": True}}}))
    n_ok = sum(1 for t in c.gt if t in pred)
    print(f"{CLIP}: gt={len(c.gt)} nodes={len(nodes_j)} b1pred={len(pred)} (on GT frames: {n_ok}) tracks={len(need)} "
          f"steps={sum(v['reach'] for v in need.values())} cuts={len(cuts)}")


if __name__ == "__main__":
    main()
