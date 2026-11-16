"""Compare this branch's ECC aligner with calibration/temporal.py (commit 7988bde) on the
same sample and metric as ``multiframe_alignment_eval`` (pitch residual after warping t-d).

NOT part of the package: ``footy_track.calibration`` only exists on exp/calibration-temporal,
so run it from a scratch tree (never touch that worktree):

    S=$(mktemp -d); git archive 7988bde src | tar -x -C $S
    cp -r src/footy_track/multiframe $S/src/footy_track/
    cp src/footy_track/scripts/multiframe_alignment_eval.py $S/src/footy_track/scripts/
    PYTHONPATH=$S/src python compare_temporal.py --n 30 --d 2 --out compare_temporal_d2.json
"""

from __future__ import annotations

import argparse
import json
import random
import time
from collections import defaultdict

import cv2
import numpy as np

from footy_track.calibration import temporal as T
from footy_track.multiframe.align import boxes_to_mask, estimate_pair
from footy_track.multiframe.frames import ClipReader, machine_person_boxes, person_boxes
from footy_track.multiframe.v11 import load_v11
from footy_track.scripts.multiframe_alignment_eval import pitch_residual


def tl_xywh(boxes):  # centre xywh -> top-left xywh (their convention)
    return np.array([[x - w / 2, y - h / 2, w, h] for x, y, w, h in boxes], np.float64).reshape(-1, 4)


CONFIGS = {  # name -> callable(a, b, boxes, mask) -> PairMotion-like (H_px, ok, reason)
    "mine ecc@480 mask": lambda a, b, bx, m: estimate_pair(a, b, m, m, 480, "ecc"),
    "mine ecc@480 nomask": lambda a, b, bx, m: estimate_pair(a, b, None, None, 480, "ecc"),
    "temporal lk boxes(all)": lambda a, b, bx, m: T.estimate_pair(a, b, boxes_a=tl_xywh(bx), boxes_b=tl_xywh(bx), method="lk"),
    "temporal lk whole-frame-no-boxes": lambda a, b, bx, m: T.estimate_pair(a, b, mask_mode="all", method="lk"),
    "temporal lk grass": lambda a, b, bx, m: T.estimate_pair(a, b, mask_mode="grass", method="lk"),
    "temporal lk_ecc boxes": lambda a, b, bx, m: T.estimate_pair(a, b, boxes_a=tl_xywh(bx), boxes_b=tl_xywh(bx), method="lk_ecc"),
    "temporal ecc boxes": lambda a, b, bx, m: T.estimate_pair(a, b, boxes_a=tl_xywh(bx), boxes_b=tl_xywh(bx), method="ecc"),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--d", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out")
    args = ap.parse_args()
    cv2.setNumThreads(1)
    items = [i for i in load_v11() if i.ref.has_neighbours(args.d)]
    random.Random(args.seed).shuffle(items)
    per, sample = defaultdict(int), []
    for it in items:  # identical sampling to multiframe_alignment_eval
        if per[it.ref.clip_id] < 6:
            per[it.ref.clip_id] += 1
            sample.append(it)
        if len(sample) >= args.n:
            break
    res = defaultdict(list)
    readers = {}
    for it in sample:
        rd = readers.setdefault(it.ref.video, ClipReader(it.ref.video))
        f = it.ref.frame
        a, b = rd.get(f), rd.get(f - args.d)
        boxes = person_boxes(it.labels) or machine_person_boxes(it.ref.video.stem, f) or []
        mask = boxes_to_mask(a.shape[:2], boxes, 0.25) if boxes else None
        ident = pitch_residual(a, b, np.eye(3), boxes)
        for name, fn in CONFIGS.items():
            t0 = time.process_time()
            try:
                pm = fn(a, b, boxes, mask)
                ok, H, reason = pm.ok, pm.H_px, pm.reason
            except Exception as e:  # noqa: BLE001
                ok, H, reason = False, np.eye(3), f"exception {e}"
            ms = (time.process_time() - t0) * 1000
            res[name].append({"ok": bool(ok), "reason": str(reason), "ms": ms, "identity": ident,
                              "pitch": pitch_residual(a, b, H, boxes)})
        print(len(res["mine ecc@480 mask"]), flush=True)
    summary = {}
    for name, rows in res.items():
        ok = [r for r in rows if r["ok"] and r["pitch"] is not None and r["identity"]]
        summary[name] = {
            "n": len(rows), "fail_rate": 1 - sum(r["ok"] for r in rows) / len(rows),
            "pitch_resid_median_ok": float(np.median([r["pitch"] for r in ok])) if ok else None,
            "identity_median_same": float(np.median([r["identity"] for r in ok])) if ok else None,
            "worse_than_identity": sum(r["pitch"] > r["identity"] for r in ok) / max(len(ok), 1),
            "ms_median_cpu": float(np.median([r["ms"] for r in rows])),
        }
        print(name, {k: round(v, 3) if isinstance(v, float) else v for k, v in summary[name].items()})
    if args.out:
        json.dump({"summary": summary, "rows": res}, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
