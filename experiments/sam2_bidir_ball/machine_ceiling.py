"""Per GT frame: is there a detector candidate (conf >= 0.1) within 10 px?  (the harness 'ceiling')

Writes ceil.json: {clip: {"reachable": [frames], "gt": [frames]}} for every harness clip and for
arsenal_example (candidates from detect_clip.py output). Used to split SAM2 fixes into frames the
detector could in principle hit (inside the ceiling) and frames it cannot (beyond the ceiling).

usage: machine_ceiling.py --out <dir> [--det <dir with arsenal_example.jsonl>]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from footy_track.ball_offline import evaluate as E  # noqa: E402
from footy_track.ball_offline.data import ClipData, _load_machine, data_root, estimate_camera_shift, load_gt  # noqa: E402


def reach(c: ClipData) -> list[int]:
    out = []
    for t, g in c.gt.items():
        xy, conf = c.cands(t)
        if len(xy) and np.any((np.hypot(xy[:, 0] - g.xy[0], xy[:, 1] - g.xy[1]) <= 10.0) & (conf >= 0.1)):
            out.append(t)
    return sorted(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--det", default=None)
    args = ap.parse_args()
    res = {}
    for ev in E.build_sets(data_root(), with_camera=False):
        res[ev.clip.clip_id] = {"reachable": reach(ev.clip), "gt": sorted(ev.clip.gt)}
    if args.det:
        n = 250
        ptr, xy, conf, wh, players = _load_machine(Path(args.det) / "arsenal_example.jsonl", n)
        gt, nb, al, cf = load_gt(data_root(), "arsenal_example")
        shift, cut = estimate_camera_shift(players)
        c = ClipData("arsenal_example", n, ptr, xy, conf, wh, {t: g for t, g in gt.items() if t < n}, set(), shift, cut, players, al, cf)
        res["arsenal_example"] = {"reachable": reach(c), "gt": sorted(c.gt)}
    Path(args.out).mkdir(parents=True, exist_ok=True)
    (Path(args.out) / "ceil.json").write_text(json.dumps(res))
    for k, v in res.items():
        print(k, len(v["reachable"]), "/", len(v["gt"]))


if __name__ == "__main__":
    main()
