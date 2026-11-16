"""Measure camera-motion alignment quality, failure rate and speed on v11 frames.

For a random sample of mappable v11 images, estimate the homography between frame t
and t-d with each method, with and without player masking, and score it by the mean
absolute grey-level error on the *pitch* (grass pixels of frame t that are not under a
labelled person) after warping, against the identity (no-alignment) baseline.

    uv run python -m footy_track.scripts.multiframe_alignment_eval --n 120 --d 2 \
        --out docs/experiments/multiframe_dataset/alignment_eval.json
"""

from __future__ import annotations

import argparse
import json
import random
import time
from collections import defaultdict

import cv2
import numpy as np

from footy_track.multiframe.align import (
    boxes_to_mask,
    estimate_pair,
    hist_distance,
    pitch_mask,
    warp_to_ref,
)
from footy_track.multiframe.frames import ClipReader, machine_person_boxes, person_boxes
from footy_track.multiframe.v11 import load_v11

CONFIGS = [  # (name, method, working width)
    ("orb@480", "orb", 480),
    ("orb@960", "orb", 960),
    ("akaze@480", "akaze", 480),
    ("ecc@480", "ecc", 480),
    ("orb+ecc@480", "orb+ecc", 480),
]


def pitch_residual(a, b, H_px, boxes, scale=480):
    """(residual, n_pixels) over grass pixels of a not under boxes, after warping b."""
    s = scale / a.shape[1]
    sa = cv2.resize(cv2.cvtColor(a, cv2.COLOR_BGR2GRAY), None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    sb = cv2.resize(cv2.cvtColor(b, cv2.COLOR_BGR2GRAY), None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    S = np.diag([s, s, 1.0])
    H = S @ H_px @ np.linalg.inv(S)
    wb = warp_to_ref(sb, H, (sa.shape[1], sa.shape[0]))
    inside = cv2.warpPerspective(np.full_like(sb, 255), H, (sa.shape[1], sa.shape[0]),
                                 flags=cv2.INTER_NEAREST | cv2.WARP_INVERSE_MAP) > 0
    grass = pitch_mask(cv2.resize(a, (sa.shape[1], sa.shape[0]), interpolation=cv2.INTER_AREA))
    m = grass & inside & (boxes_to_mask(sa.shape, boxes, 0.4) == 0)
    # Interior only: drop a 4 px border so replication/warp edge effects do not count.
    m[:4] = m[-4:] = False
    m[:, :4] = m[:, -4:] = False
    if m.sum() < 200:
        return None
    # 3x3 blur on both removes sensor/codec noise that would dominate flat grass.
    d = np.abs(cv2.GaussianBlur(sa, (3, 3), 0).astype(np.float32) - cv2.GaussianBlur(wb, (3, 3), 0).astype(np.float32))
    return float(d[m].mean())


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=120)
    ap.add_argument("--d", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    cv2.setNumThreads(1)  # single-thread numbers, comparable across loads

    items = [i for i in load_v11() if i.ref.has_neighbours(args.d)]
    random.Random(args.seed).shuffle(items)
    # Stratify: cap per clip so astonvilla_seg080 (520 frames) does not dominate.
    per_clip, sample = defaultdict(int), []
    for it in items:
        if per_clip[it.ref.clip_id] < 6:
            per_clip[it.ref.clip_id] += 1
            sample.append(it)
        if len(sample) >= args.n:
            break

    rows = []
    readers: dict = {}
    for it in sample:
        rd = readers.setdefault(it.ref.video, ClipReader(it.ref.video))
        f = it.ref.frame
        a, b = rd.get(f), rd.get(f - args.d)
        if a is None or b is None:
            continue
        boxes = person_boxes(it.labels) or machine_person_boxes(it.ref.video.stem, f) or []
        mask = boxes_to_mask(a.shape[:2], boxes, 0.25) if boxes else None
        # adjacent-frame histogram distances over the +-d window (cut signal)
        hd = []
        for g in range(f - args.d, f):
            x, y = rd.get(g), rd.get(g + 1)
            hd.append(hist_distance(x, y))
        row = {"clip": it.ref.clip_id, "frame": f, "n_boxes": len(boxes),
               "hist_adjacent_max": max(hd), "hist_t_td": hist_distance(a, b)}
        base = pitch_residual(a, b, np.eye(3), boxes)
        row["identity_pitch"] = base
        for name, method, scale in CONFIGS:
            for masked in (True, False):
                if masked and mask is None:
                    continue
                t0 = time.process_time()
                pm = estimate_pair(a, b, mask if masked else None, mask if masked else None,
                                   scale=scale, method=method)
                ms = (time.process_time() - t0) * 1000  # CPU ms: the box is shared/loaded
                row[f"{name}|{'mask' if masked else 'nomask'}"] = {
                    "ok": pm.ok, "reason": pm.reason, "inliers": pm.inliers, "ms": ms,
                    "pitch": pitch_residual(a, b, pm.H_px, boxes) if np.isfinite(pm.residual) else None,
                    "resid": pm.residual,
                }
        rows.append(row)
        print(len(rows), it.ref.clip_id, f, flush=True)
    for r in readers.values():
        r.close()

    summary = {}
    for name, _, _ in CONFIGS:
        for tag in ("mask", "nomask"):
            key = f"{name}|{tag}"
            rs = [r[key] | {"identity": r["identity_pitch"]} for r in rows if key in r]
            if not rs:
                continue
            ok = [x for x in rs if x["ok"]]
            pit = [x["pitch"] for x in ok if x["pitch"] is not None and x["identity"]]
            ident = [x["identity"] for x in ok if x["pitch"] is not None and x["identity"]]
            # "worse than identity" among accepted pairs = silent failure
            silent = sum(1 for x in ok if x["pitch"] is not None and x["identity"] and x["pitch"] > x["identity"])
            summary[key] = {
                "n": len(rs), "fail_rate": 1 - len(ok) / len(rs),
                "silent_worse_than_identity": silent / max(len(ok), 1),
                "ms_median": float(np.median([x["ms"] for x in rs])),
                "ms_mean": float(np.mean([x["ms"] for x in rs])),
                "pitch_resid_median": float(np.median(pit)) if pit else None,
                "pitch_resid_p90": float(np.percentile(pit, 90)) if pit else None,
                "identity_resid_median_same_pairs": float(np.median(ident)) if ident else None,
                "inliers_median": float(np.median([x["inliers"] for x in rs])),
                "reasons": dict(defaultdict(int, {k: sum(1 for x in rs if x["reason"] == k)
                                                  for k in {x["reason"] for x in rs if x["reason"]}})),
            }
    out = {"d": args.d, "n_pairs": len(rows), "summary": summary, "rows": rows}
    if args.out:
        with open(args.out, "w") as fh:
            json.dump(out, fh, indent=1)
    for k, v in summary.items():
        print(k, {a: (round(b, 3) if isinstance(b, float) else b) for a, b in v.items() if a != "reasons"}, v["reasons"])


if __name__ == "__main__":
    main()
