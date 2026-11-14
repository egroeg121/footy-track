"""Prep for the Ball Check verdict probe of SAM2 false fills (CPU).

Ball Check holds human yes/no verdicts on individual detector candidates, including confirmed
*non*-balls (boots, logos, etc.). A SAM2 fill that lands within 10 px of a confirmed non-ball is a
known false fill. Uses the harness verdict loader (scripts/eval_offline_ball_verdicts.py) and the
same clean-frame rule (verdicts within 5 frames of a detector-training image are dropped).

Primary anchor config only (threshold/spacing given on the command line). Tracks are trimmed to
what is needed to decide the verdict frames: from the previous anchor up to the verdict frame
(forward) and from the next anchor down to it (backward).

usage: machine_prep_verdicts.py --out <dir> [--thr 0.7 --spacing 10]   -> prep_verdicts.json, spec_verdicts.json
"""
from __future__ import annotations

import argparse
import bisect
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from eval_offline_ball_verdicts import load_verdicts  # noqa: E402
from footy_track.ball_offline import baselines as bl  # noqa: E402
from footy_track.ball_offline.data import contaminated_frames, data_root, load_clip, machine_stem_map, roboflow_frames  # noqa: E402
from footy_track.ball_offline.solver import SolverParams  # noqa: E402
from machine_common import REACH, thin  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--thr", type=float, default=0.7)
    ap.add_argument("--spacing", type=int, default=10)
    ap.add_argument("--results", default=str(ROOT / "docs/experiments/offline_ball_baseline.json"))
    args = ap.parse_args()
    res = json.loads(Path(args.results).read_text())
    p1 = SolverParams(**res["B1"]["tuned_params"]["camera_on"])
    root = data_root()
    ml = machine_stem_map(root)
    rf = roboflow_frames(root)
    verdicts = load_verdicts(root)
    prep, spec = {"thr": args.thr, "spacing": args.spacing, "clips": {}}, {"reach": REACH, "clips": {}}
    for cid, items in sorted(verdicts.items()):
        if cid not in ml:
            continue
        bad = contaminated_frames(cid, rf)
        items = [it for it in items if it[0] not in bad]
        if not items:
            continue
        clip = load_clip(cid, root, with_camera=True)
        pred, nodes = bl.b1_solver(clip, p1)
        nodes_j = []
        for n in nodes:
            xy, _ = clip.cands(n.t)
            k = int(np.argmin(np.hypot(xy[:, 0] - n.xy[0], xy[:, 1] - n.xy[1]))) if len(xy) else None
            s, e = clip.ptr[n.t], clip.ptr[n.t + 1]
            wh = clip.wh[s:e][k] if k is not None else np.array([16.0, 16.0])
            w, h = float(wh[0]), float(wh[1])
            nodes_j.append({"t": n.t, "xy": list(n.xy), "conf": n.conf, "linked": n.linked,
                            "box": [n.xy[0] - w / 2, n.xy[1] - h / 2, n.xy[0] + w / 2, n.xy[1] + h / 2]})
        pf = {int(t) for t in pred}
        anchors = thin(nodes_j, pf, args.thr, args.spacing)
        A = [a["t"] for a in anchors]
        box = {a["t"]: a["box"] for a in anchors}
        need: dict[tuple[int, str], dict] = {}
        for t, _xy, _v in items:
            if t in set(A):
                continue
            i = bisect.bisect_left(A, t)
            if i > 0 and t - A[i - 1] <= REACH:
                k = (A[i - 1], "fwd")
                r = t - A[i - 1]
                if k not in need or need[k]["reach"] < r:
                    need[k] = {"t": A[i - 1], "dir": "fwd", "reach": r, "box": box[A[i - 1]]}
            if i < len(A) and A[i] - t <= REACH:
                k = (A[i], "bwd")
                r = A[i] - t
                if k not in need or need[k]["reach"] < r:
                    need[k] = {"t": A[i], "dir": "bwd", "reach": r, "box": box[A[i]]}
        stem = ml[cid].stem
        prep["clips"][cid] = {"stem": stem, "n_frames": clip.n_frames, "items": [[t, list(xy), v] for t, xy, v in items],
                              "b1": {str(t): list(v) for t, v in pred.items()}, "anchors": A,
                              "cuts": [int(t) for t in np.where(clip.cam_cut)[0]]}
        spec["clips"][cid] = {"video": str(root / "clips" / f"{stem}.mp4"), "n_frames": clip.n_frames,
                              "tracks": list(need.values()), "full": False}
        print(f"{cid:34s} verdicts={len(items):3d} anchors={len(A):4d} tracks={len(need):3d} steps={sum(v['reach'] for v in need.values())}", flush=True)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "prep_verdicts.json").write_text(json.dumps(prep))
    (out / "spec_verdicts.json").write_text(json.dumps(spec))
    tot = sum(v["reach"] for c in spec["clips"].values() for v in c["tracks"])
    nv = sum(len(c["items"]) for c in prep["clips"].values())
    print(f"clips={len(prep['clips'])} verdicts={nv} total steps {tot} (~{tot * 0.2 / 60:.0f} min)")


if __name__ == "__main__":
    main()
