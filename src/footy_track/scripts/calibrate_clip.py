"""Calibrate every frame of a clip from keyframe calibrations, optionally project its detections to the pitch.

    PYTHONPATH=src python -m footy_track.scripts.calibrate_clip --clip arsenal_mancity_seg020 --out OUT \\
        [--keys gt | 18,93 | every:100] [--boxes] [--no-refine]

Writes under ``--out``:

* ``<clip>.calibration.parquet``  one row per frame: frame, source (keyframe|propagated|refined|none), anchor_frame,
  key_frame, gap, line_cost, confidence, shot_id, ok, k1, H (9 floats, normalised image -> pitch m)
* ``<clip>.positions.parquet``    (with ``--boxes``) one row per person/ball box: frame, track_id, label, x_m, y_m, conf,
  calib_confidence, calib_ok, in_pitch, in_bounds
* ``<clip>.motion.npz`` / ``<clip>.masks.npz``  the camera motion and line masks, reused when present (delete to recompute)
* ``<clip>.summary.json``

Keyframes come from the calibration GT (``<gt-dir>/calibration_gt/<clip>.json``, read-only): ``--keys gt`` takes all of them,
``every:N`` the GT frames nearest to 0, N, 2N, ..., a comma list takes those frames. Run it single-process
(``OMP_NUM_THREADS=2``, ``nice``): the line masks are ~1 s/frame per check frame.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

from footy_track.calibration import temporal as tp
from footy_track.calibration.clip_calibration import (
    MaskStore,
    Policy,
    calibrate_clip,
)
from footy_track.calibration.prelabel.pipeline import find_video
from footy_track.clip_id import canonical_clip_id
from footy_track.projection.players import load_detections, project_players

GT_DIR = Path(os.environ.get("FOOTY_GT_MARKS_DIR", "/mnt/storage/footy_data/ball_gt_marks"))
CLIPS_DIR = Path(os.environ.get("FOOTY_CLIPS_DIR", "/mnt/storage/footy_data/ballcheck_clips"))
BOXES_DIR = Path(os.environ.get("FOOTY_MACHINE_LABELS_DIR", "/mnt/storage/footy_data/machine_labels"))


def load_gt_keyframes(clip: str, gt_dir: Path = GT_DIR) -> dict[int, tuple[np.ndarray, dict | None]]:
    """``{frame: (H, distortion)}`` of the calibration GT of ``clip`` (read-only)."""
    doc = json.loads((Path(gt_dir) / "calibration_gt" / f"{canonical_clip_id(clip)}.json").read_text())
    return {int(k): (np.array(r["H"], float), r.get("distortion")) for k, r in doc["frames"].items()}


def pick_nearest(frames: list[int], targets: list[float]) -> list[int]:
    """The distinct frames of ``frames`` nearest to each of ``targets`` (first one on ties)."""
    fr = sorted(frames)
    out = []
    for t in targets:
        f = min(fr, key=lambda x: (abs(x - t), x))
        if f not in out:
            out.append(f)
    return sorted(out)


def select_keys(spec: str, gt: dict, n_frames: int) -> dict:
    if spec == "gt":
        return gt
    if spec.startswith("every:"):
        step = int(spec.split(":")[1])
        return {f: gt[f] for f in pick_nearest(list(gt), list(range(0, n_frames, step)))}
    return {int(f): gt[int(f)] for f in spec.split(",")}


def find_boxes_path(clip: str, boxes_dir: Path = BOXES_DIR) -> Path | None:
    want = canonical_clip_id(clip)
    for p in sorted(Path(boxes_dir).glob("*.jsonl")):
        if p.stem == clip or canonical_clip_id(p.stem) == want:
            return p
    return None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--clip", required=True)
    ap.add_argument("--video", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--keys", default="gt", help="gt | every:N | comma-separated GT frames")
    ap.add_argument("--boxes", action="store_true", help="mask person boxes in the motion estimate and project them to the pitch")
    ap.add_argument("--boxes-path", type=Path)
    ap.add_argument("--no-refine", action="store_true")
    ap.add_argument("--check-every", type=int, default=5)
    a = ap.parse_args(argv)

    a.out.mkdir(parents=True, exist_ok=True)
    cid = canonical_clip_id(a.clip)
    video = a.video or find_video(CLIPS_DIR, a.clip)
    if video is None:
        raise SystemExit(f"no video for {a.clip} under {CLIPS_DIR}")
    gt = load_gt_keyframes(a.clip)
    boxes_path = a.boxes_path or find_boxes_path(a.clip)
    dets = load_detections(boxes_path) if a.boxes and boxes_path else None
    boxes = tp.load_person_boxes(boxes_path) if a.boxes and boxes_path else None

    t0 = time.perf_counter()
    mp = a.out / f"{cid}.motion.npz"
    if mp.exists():
        motion = tp.ClipMotion.load(mp)
    else:
        motion = tp.clip_motion(video, boxes=boxes, mask_mode="all", progress=lambda i: print(f"  motion {i}", flush=True) if i % 200 == 0 else None)
        motion.save(mp)
    t_motion = time.perf_counter() - t0
    keys = select_keys(a.keys, gt, motion.n)
    pol = Policy(check_every=a.check_every, refine=not a.no_refine)
    kp = a.out / f"{cid}.masks.npz"
    masks = MaskStore.load(kp) if kp.exists() else MaskStore(pol.work_width)
    n_before = len(masks)
    cal = calibrate_clip(video, keys, boxes, pol, motion=motion, masks=masks,
                         progress=lambda i: print(f"  masks {i}", flush=True) if i % 100 == 0 else None)
    if len(masks) != n_before:
        masks.save(kp)
    cal.save(a.out / f"{cid}.calibration.parquet")
    summary = {"clip": cid, "video": str(video), "keyframes": sorted(keys), "frames": motion.n, "cuts": motion.cuts,
               "motion_s": t_motion, **{k: v for k, v in cal.stats.items() if k != "cuts"},
               "by_source": cal.table["source"].value_counts().to_dict()}
    if dets is not None:
        pos = project_players(dets, cal)
        pos.to_parquet(a.out / f"{cid}.positions.parquet", index=False)
        pl = pos[pos["label"] == "player"]
        summary["positions"] = {"n": len(pos), "calibrated_frac": float(pos["calib_ok"].mean()),
                                "players_in_pitch": float(pl.loc[pl["calib_ok"], "in_pitch"].mean()),
                                "players_in_bounds_5m": float(pl.loc[pl["calib_ok"], "in_bounds"].mean())}
    (a.out / f"{cid}.summary.json").write_text(json.dumps(summary, indent=1, default=float))
    print(json.dumps(summary, indent=1, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
