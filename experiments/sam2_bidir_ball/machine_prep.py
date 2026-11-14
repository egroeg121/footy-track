"""CPU prep for the machine-anchor SAM2 experiment (S1 gap fill / S2 re-anchoring).

Uses the offline ball-tracking harness (``footy_track.ball_offline``, commit fc05732) so that
clips, GT, contamination exclusions and B0/B1 are identical to ``offline_ball_baseline.md``.

Writes ``prep.json`` (per clip: B0/B1 predictions, B1 nodes, GT, no_ball, scored frames, role,
cut frames, not_broadcast frames, video path) and ``spec.json`` (the SAM2 tracks to compute:
box prompt, direction, reach, per clip).

usage (mini, one process, thread-capped):
    OMP_NUM_THREADS=1 python machine_prep.py --out <dir> [--results offline_ball_baseline.json]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from footy_track.ball_offline import baselines as bl  # noqa: E402
from footy_track.ball_offline import evaluate as E  # noqa: E402
from footy_track.ball_offline.data import canonical_clip_id, data_root  # noqa: E402
from footy_track.ball_offline.solver import SolverParams  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from machine_common import CONFIGS, MAIN_CLIPS, REACH, thin, tracks_for  # noqa: E402

def other_marks(root: Path, clip_id: str) -> dict:
    """not_broadcast / out_of_play frames of a clip from every GT alias."""
    nb, oop = set(), set()
    for p in sorted((root / "ball_gt_marks").glob("*.jsonl")):
        if canonical_clip_id(p.name) != clip_id:
            continue
        for line in p.read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            if "not_broadcast" in r["tags"]:
                nb.add(r["frame_index"])
            elif "out_of_play_ball" in r["tags"]:
                oop.add(r["frame_index"])
    return {"not_broadcast": sorted(nb), "out_of_play": sorted(oop)}


def video_path(root: Path, clip_id: str) -> str:
    """canonical id -> a file in clips/ (date-stamped broadcast segments)."""
    kind = clip_id.replace("arsenal_mancity_", "")
    p = root / "clips" / f"arsenal_mancity_20250925_{kind}.mp4"
    return str(p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--results", default=str(ROOT / "docs/experiments/offline_ball_baseline.json"))
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    res = json.loads(Path(args.results).read_text())
    cam = res["B1"]["camera_selected_on_dev"]
    p1 = SolverParams(**res["B1"]["tuned_params"]["camera_on" if cam else "camera_off"])
    tau = res["B0"]["tau_selected_on_dev"]
    root = data_root()
    t0 = time.time()
    sets = E.build_sets(root)
    print(f"[{time.time()-t0:.0f}s] built {len(sets)} clip sets", flush=True)

    prep: dict = {"b1_params": p1.__dict__, "b0_tau": tau, "reach": REACH, "configs": CONFIGS, "clips": {}}
    spec: dict = {"reach": REACH, "clips": {}}
    for ev in sets:
        c = ev.clip
        pred, nodes = bl.b1_solver(c, p1)
        b0 = bl.b0_per_frame(c, tau)
        nodes_j = []
        for n in nodes:
            # detector box of the node: nearest candidate on that frame
            xy, conf = c.cands(n.t)
            k = int(np.argmin(np.hypot(xy[:, 0] - n.xy[0], xy[:, 1] - n.xy[1]))) if len(xy) else None
            s, e = c.ptr[n.t], c.ptr[n.t + 1]
            wh = c.wh[s:e][k] if k is not None else np.array([16.0, 16.0])
            w, h = float(wh[0]), float(wh[1])
            nodes_j.append({"t": n.t, "xy": list(n.xy), "conf": n.conf, "linked": n.linked,
                            "box": [n.xy[0] - w / 2, n.xy[1] - h / 2, n.xy[0] + w / 2, n.xy[1] + h / 2]})
        marks = other_marks(root, c.clip_id)
        cuts = [int(t) for t in np.where(c.cam_cut)[0]]
        scored = sorted(ev.scored)
        pj = {
            "role": ev.role, "n_frames": c.n_frames, "video": video_path(root, c.clip_id),
            "gt": {str(t): list(g.xy) for t, g in c.gt.items()},
            "gt_tier": {str(t): g.tier for t, g in c.gt.items()},
            "no_ball": sorted(c.no_ball), "scored": scored,
            "contaminated": sorted(ev.contaminated),
            "b0": {str(t): list(v) for t, v in b0.items()},
            "b1": {str(t): list(v) for t, v in pred.items()},
            "nodes": nodes_j, "cuts": cuts, **marks,
        }
        prep["clips"][c.clip_id] = pj
        # ---- track spec: union over configs ------------------------------------------
        need: dict[tuple[int, str], dict] = {}
        relevant = set(scored) | set(marks["not_broadcast"]) | set(marks["out_of_play"])
        pf = {int(t) for t in pred}
        for thr, sp in CONFIGS:
            anchors = thin(nodes_j, pf, thr, sp)
            for tr in tracks_for(anchors, c.n_frames):
                for d, reach in (("fwd", tr["fwd"]), ("bwd", tr["bwd"])):
                    if reach < 1:
                        continue
                    lo, hi = (tr["t"], tr["t"] + reach) if d == "fwd" else (tr["t"] - reach, tr["t"])
                    if c.clip_id not in MAIN_CLIPS and not any(lo <= f <= hi for f in relevant):
                        continue  # not needed to score anything on this clip
                    k = (tr["t"], d)
                    if k not in need or need[k]["reach"] < reach:
                        need[k] = {"t": tr["t"], "dir": d, "reach": reach, "box": tr["box"]}
        spec["clips"][c.clip_id] = {"video": pj["video"], "n_frames": c.n_frames, "tracks": list(need.values()),
                                    "full": c.clip_id in MAIN_CLIPS}
        n_steps = sum(v["reach"] for v in need.values())
        print(f"{c.clip_id:28s} {ev.role:4s} frames={c.n_frames:5d} gt={len(c.gt):4d} nb={len(c.no_ball):2d} "
              f"nodes={len(nodes_j):4d} b1pred={len(pred):4d} tracks={len(need):4d} steps={n_steps}", flush=True)
    (out / "prep.json").write_text(json.dumps(prep))
    (out / "spec.json").write_text(json.dumps(spec))
    tot = sum(v["reach"] for c in spec["clips"].values() for v in c["tracks"])
    print(f"total propagation steps {tot} (~{tot*0.15/60:.0f} min at 0.15 s/frame)")


if __name__ == "__main__":
    main()
