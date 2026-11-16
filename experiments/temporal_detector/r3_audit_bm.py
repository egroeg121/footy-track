"""Anchor-purity check off Man City: harness B1 over RT-DETR labels of the held-out Bournemouth clips (excluded from the pool, so this
audit does not leak into training) scored against their human GT. Only GT frames on the 25 fps subsample (even original indices) can be
compared. CPU, numpy only.

    python r3_audit_bm.py --labels-dir <dir with the 6 audit clips> --out audit_bm.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from r3_prep import HARNESS, make_root  # noqa: E402

sys.path.insert(0, str(HARNESS / "src"))
from footy_track.ball_offline import data as hd  # noqa: E402
from footy_track.ball_offline.solver import SolverParams, solve  # noqa: E402
from pipeline_eval import CAMERA_ON  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--stride-orig", type=int, default=2)
    ap.add_argument("--work", type=Path, default=Path("/mnt/storage/exp/temporal/round3/audit_root"))
    a = ap.parse_args()
    clips = make_root(a.labels_dir, a.work)
    real = Path("/mnt/storage/footy_data")
    params = SolverParams(**CAMERA_ON)
    rows = []
    for cid in clips:
        clip = hd.load_clip(cid, a.work, with_camera=True)
        gt, _, _, _ = hd.load_gt(real, cid)  # human GT of an EXCLUDED held-out clip: audit only
        nodes = solve(clip, params)
        high = sorted(n.t for n in nodes if n.conf >= 0.8)
        for n in nodes:
            t_orig = n.t * a.stride_orig
            if t_orig not in gt:
                continue
            d = math.hypot(n.xy[0] - gt[t_orig].xy[0], n.xy[1] - gt[t_orig].xy[1])
            rows.append({"clip": cid, "t_orig": t_orig, "conf": round(n.conf, 3), "hit10": d <= 10,
                         "survivor_if_low": n.conf >= 0.45 and any(abs(n.t - u) <= 25 for u in high)})
    band = {"conf>=0.8": lambda r: r["conf"] >= 0.8, "0.6-0.8": lambda r: 0.6 <= r["conf"] < 0.8, "<0.6": lambda r: r["conf"] < 0.6,
            "<0.6 survivors of the round-2 filter": lambda r: r["conf"] < 0.6 and r["survivor_if_low"]}
    summ = {k: {"n": sum(1 for r in rows if f(r)), "hit10": sum(1 for r in rows if f(r) and r["hit10"])} for k, f in band.items()}
    n_gt_even = sum(1 for cid in clips for t in hd.load_gt(real, cid)[0] if t % a.stride_orig == 0)
    a.out.write_text(json.dumps({"summary": summ, "rows": rows, "gt_frames_on_subsample": n_gt_even, "clips": clips}, indent=1))
    print(json.dumps(summ, indent=1), "GT frames on the 25 fps subsample:", n_gt_even)


if __name__ == "__main__":
    main()
