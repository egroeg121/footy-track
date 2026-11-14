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

``--auto-keys`` (instead of ``--keys``) takes the keyframes from what already exists, with no human or model call: the calibration GT
frames plus the ``calibration_proposals`` frames (statuses ``accepted``, ``edited``, ``proposed`` -- ``--auto-statuses``; best status wins per
frame, ``rejected`` / ``failed`` / ``not_broadcast`` never). The clip is split into shots and classified (``calibration.shots``); keyframes
in shots that are not the main camera are dropped, so ``calibrate_clip`` only sweeps main-camera shots, and a ``proposed`` keyframe must pass
the line-cost gate (``Policy.max_line_cost`` on its own frame) before it is used. Main shots left without a keyframe are listed in the
summary with ``keyframe_candidate`` (the frame the shot classifier thinks best supports a fit) so they can be labelled next.
Also writes ``<clip>.shots.json``.

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

from footy_track.calibration import geometry as geo
from footy_track.calibration import shots as sh
from footy_track.calibration import temporal as tp
from footy_track.calibration.clip_calibration import (
    MaskStore,
    Policy,
    calibrate_clip,
    line_cost,
)
from footy_track.calibration.pitch_model import PitchModel
from footy_track.calibration.prelabel.pipeline import find_video
from footy_track.calibration.prelabel.proposals import read_proposals
from footy_track.calibration.refine import model_samples
from footy_track.clip_id import canonical_clip_id
from footy_track.projection.players import load_detections, project_players

GT_DIR = Path(os.environ.get("FOOTY_GT_MARKS_DIR", "/mnt/storage/footy_data/ball_gt_marks"))
CLIPS_DIR = Path(os.environ.get("FOOTY_CLIPS_DIR", "/mnt/storage/footy_data/ballcheck_clips"))
BOXES_DIR = Path(os.environ.get("FOOTY_MACHINE_LABELS_DIR", "/mnt/storage/footy_data/machine_labels"))


def load_gt_keyframes(clip: str, gt_dir: Path = GT_DIR) -> dict[int, tuple[np.ndarray, dict | None]]:
    """``{frame: (H, distortion)}`` of the calibration GT of ``clip`` (read-only)."""
    doc = json.loads((Path(gt_dir) / "calibration_gt" / f"{canonical_clip_id(clip)}.json").read_text())
    return {int(k): (np.array(r["H"], float), r.get("distortion")) for k, r in doc["frames"].items()}


AUTO_STATUSES = ("accepted", "edited", "proposed")
MAX_PROPOSAL_COND = 1e6


def load_proposal_keyframes(clip: str, gt_dir: Path = GT_DIR, statuses: tuple[str, ...] = AUTO_STATUSES) -> dict[int, dict]:
    """``{frame: {"H", "distortion", "status", "run"}}`` from the proposal sidecar of ``clip`` (read-only): frames whose status is in
    ``statuses`` and that have a usable homography (finite, well conditioned). One entry per frame: the better status
    (earlier in ``statuses``) wins, then the later run. The calibration GT frames are not included (see :func:`load_gt_keyframes`)."""
    out: dict[int, dict] = {}
    rank = {s: i for i, s in enumerate(statuses)}
    for run_id, run in read_proposals(Path(gt_dir), clip).get("runs", {}).items():
        for k, fr in run.get("frames", {}).items():
            st = fr.get("status")
            if st not in rank or fr.get("H") is None:
                continue
            H = np.array(fr["H"], float)
            if H.shape != (3, 3) or not np.isfinite(H).all() or np.linalg.cond(H) > MAX_PROPOSAL_COND:
                continue
            cur = out.get(int(k))
            if cur is None or rank[st] <= rank[cur["status"]]:
                out[int(k)] = {"H": H, "distortion": fr.get("distortion"), "status": st, "run": run_id}
    return out


def auto_keys(shots: list[dict], gt: dict, proposals: dict[int, dict]) -> tuple[dict, dict]:
    """Keyframes ``{frame: (H, distortion)}`` for a calibration run, and a report ``{"used", "skipped_not_main", "skipped_no_shot"}``
    (frame lists). GT frames win over proposals; keyframes in a shot that is not the main camera are skipped."""
    cand: dict[int, tuple[np.ndarray, dict | None, str]] = {f: (H, d, "gt") for f, (H, d) in gt.items()}
    for f, p in proposals.items():
        cand.setdefault(f, (p["H"], p["distortion"], p["status"]))
    keys, rep = {}, {"used": {}, "skipped_not_main": [], "skipped_no_shot": []}
    for f, (H, d, src) in sorted(cand.items()):
        s = sh.shot_of(shots, f)
        if s is None:
            rep["skipped_no_shot"].append(f)
        elif s["type"] != sh.MAIN:
            rep["skipped_not_main"].append(f)
        else:
            keys[f] = (H, d)
            rep["used"][f] = src
    return keys, rep


def gate_proposed(keys: dict, source: dict[int, str], video: Path, masks: MaskStore, pol: Policy) -> tuple[dict, list[dict]]:
    """Drop ``proposed`` keyframes whose homography does not fit the white lines of its own frame (``line_cost`` above
    ``pol.max_line_cost``, or no line to judge by); accepted / edited / GT keyframes are trusted. Returns (kept, dropped records)."""
    check = [f for f, src in source.items() if src == "proposed"]
    masks.ensure(video, check)
    samples = model_samples(PitchModel())
    kept, dropped = dict(keys), []
    for f in check:
        H, d = keys[f]
        dp = geo.distortion_params(d)
        c = line_cost(geo.orient_homography(np.asarray(H, float), np.array([[0.5, 0.5]])), masks.get(f), samples, dp[0] if dp else 0.0)
        if not np.isfinite(c) or c > pol.max_line_cost:
            kept.pop(f)
            dropped.append({"frame": f, "line_cost": None if not np.isfinite(c) else float(c)})
    return kept, dropped


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


def run_auto_keys(a, cid: str, video: Path, motion: tp.ClipMotion, boxes, gt: dict, masks: MaskStore, pol: Policy) -> tuple[dict, dict]:
    """``--auto-keys``: classify the shots, take the GT + proposal keyframes inside main shots (gated), write ``<clip>.shots.json``;
    returns (keyframes, info for the summary)."""
    ts = time.perf_counter()
    fp = a.out / f"{cid}.feat.json"
    feats = {int(k): v for k, v in json.loads(fp.read_text()).items()} if fp.exists() else {}
    feats = {f: v for f, v in feats.items() if all(n in v for n in sh.FRAME_FEATURES)}
    missing = [f for f in sh.sample_frames(motion, 10) if f not in feats]
    if missing:
        feats.update(sh.extract_frame_features(video, missing, boxes))
        fp.write_text(json.dumps(feats))
    shots = sh.classify_shots(video, motion, boxes, features=feats)
    t_shots = time.perf_counter() - ts
    props = load_proposal_keyframes(a.clip, GT_DIR, tuple(a.auto_statuses.split(",")))
    keys, rep = auto_keys(shots, gt, props)
    keys, dropped = gate_proposed(keys, rep["used"], video, masks, pol)
    for d in dropped:
        rep["used"].pop(d["frame"])
    if not keys:
        raise SystemExit(f"no usable keyframe in a main-camera shot of {a.clip} ({rep})")
    recs = [{k: v for k, v in s.items() if k != "features"} | {"keyframes": [f for f in keys if s["start"] <= f <= s["end"]]} for s in shots]
    (a.out / f"{cid}.shots.json").write_text(json.dumps(recs, indent=1, default=float))
    info = {"shots_s": t_shots, **rep, "dropped_by_line_cost": dropped, "main_shots_without_keyframe": [s for s in recs if s["type"] == sh.MAIN and not s["keyframes"]],
            "shots": [{"shot_id": s["shot_id"], "frames": [s["start"], s["end"]], "type": s["type"], "p_main": round(s["p_main"], 3), "keyframes": s["keyframes"]} for s in recs]}
    return keys, info


def main(argv=None) -> int:  # noqa: PLR0915
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--clip", required=True)
    ap.add_argument("--video", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--keys", default="gt", help="gt | every:N | comma-separated GT frames")
    ap.add_argument("--boxes", action="store_true", help="mask person boxes in the motion estimate and project them to the pitch")
    ap.add_argument("--boxes-path", type=Path)
    ap.add_argument("--auto-keys", action="store_true", help="keyframes from the GT + calibration proposals inside main-camera shots (see above)")
    ap.add_argument("--auto-statuses", default=",".join(AUTO_STATUSES), help="proposal statuses --auto-keys may use, best first")
    ap.add_argument("--no-refine", action="store_true")
    ap.add_argument("--check-every", type=int, default=5)
    a = ap.parse_args(argv)

    a.out.mkdir(parents=True, exist_ok=True)
    cid = canonical_clip_id(a.clip)
    video = a.video or find_video(CLIPS_DIR, a.clip)
    if video is None:
        raise SystemExit(f"no video for {a.clip} under {CLIPS_DIR}")
    try:
        gt = load_gt_keyframes(a.clip)
    except FileNotFoundError:
        if not a.auto_keys:
            raise
        gt = {}
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
    pol = Policy(check_every=a.check_every, refine=not a.no_refine)
    kp = a.out / f"{cid}.masks.npz"
    masks = MaskStore.load(kp) if kp.exists() else MaskStore(pol.work_width)
    n_before = len(masks)
    auto_info = None
    if a.auto_keys:
        keys, auto_info = run_auto_keys(a, cid, video, motion, boxes, gt, masks, pol)
    else:
        keys = select_keys(a.keys, gt, motion.n)
    cal = calibrate_clip(video, keys, boxes, pol, motion=motion, masks=masks,
                         progress=lambda i: print(f"  masks {i}", flush=True) if i % 100 == 0 else None)
    if len(masks) != n_before:
        masks.save(kp)
    cal.save(a.out / f"{cid}.calibration.parquet")
    summary = {"clip": cid, "video": str(video), "keyframes": sorted(keys), "frames": motion.n, "cuts": motion.cuts,
               "motion_s": t_motion, **{k: v for k, v in cal.stats.items() if k != "cuts"},
               "by_source": cal.table["source"].value_counts().to_dict()}
    if auto_info is not None:
        summary["auto_keys"] = auto_info
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
