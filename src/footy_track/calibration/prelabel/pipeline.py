"""Pre-labelling pipeline: frames -> candidates -> identify -> fit -> proposals."""

from __future__ import annotations

import json
import time
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from footy_track.calibration import geometry as geo
from footy_track.calibration.pitch_model import DEFAULT_LENGTH, DEFAULT_WIDTH, PitchModel
from footy_track.calibration.prelabel.candidates import find_candidates, render_candidates
from footy_track.calibration.prelabel.fit import fit_and_score
from footy_track.calibration.prelabel.identifier import (
    IdentifyInput,
    Identifier,
    jpeg,
    render_diagram,
    resolve_extra_points,
)
from footy_track.calibration.prelabel.proposals import update_proposals
from footy_track.calibration.video import read_frame
from footy_track.clip_id import canonical_clip_id

_VIDEO_SUFFIXES = {".mp4", ".mov", ".avi", ".mkv"}


# --------------------------------------------------------------------------
# frame selection
# --------------------------------------------------------------------------


def find_video(clips_dir: Path, clip: str) -> Path | None:
    """Video under ``clips_dir`` whose stem is ``clip`` or any alias of it."""
    clips_dir = Path(clips_dir)
    exact = [p for p in clips_dir.glob(f"{clip}.*") if p.suffix.lower() in _VIDEO_SUFFIXES]
    if exact:
        return exact[0]
    want = canonical_clip_id(clip)
    for p in sorted(clips_dir.iterdir()):
        if p.suffix.lower() in _VIDEO_SUFFIXES and canonical_clip_id(p.stem) == want:
            return p
    return None


def video_frame_count(video: Path) -> int:
    cap = cv2.VideoCapture(str(video))
    try:
        return int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    finally:
        cap.release()


def gt_frames(gt_dir: Path, clip: str) -> set[int]:
    p = Path(gt_dir) / "calibration_gt" / f"{canonical_clip_id(clip)}.json"
    if not p.exists():
        return set()
    return {int(k) for k in json.loads(p.read_text()).get("frames", {})}


def select_frames(
    n_frames: int,
    every: int = 25,
    explicit: Iterable[int] | None = None,
    skip: Iterable[int] = (),
) -> list[int]:
    """Explicit frames, else keyframes ``0, every, 2*every, ...``; minus ``skip``."""
    skip = set(skip)
    base = list(explicit) if explicit else list(range(0, max(0, n_frames), max(1, every)))
    return [f for f in dict.fromkeys(base) if 0 <= f < n_frames and f not in skip]


# --------------------------------------------------------------------------
# record building
# --------------------------------------------------------------------------


def make_run_id(prefix: str, now: float | None = None) -> str:
    return f"{prefix}-{time.strftime('%Y%m%dT%H%M%S', time.gmtime(now))}"


def _view_record(res) -> dict | None:
    v = res.view or {}
    if not v and not res.unusable:
        return None
    cam = None
    if v.get("reverse_angle_suspected"):
        cam = "reverse_angle?"
    elif v.get("main_camera_wide"):
        cam = "main_wide"
    elif v.get("main_camera_wide") is False:
        cam = "other"
    end = v.get("end")
    return {
        "end": end if end in ("L", "R") else None, "camera_side": cam,
        "notes": v.get("notes") or res.unusable_reason,
        "main_camera_wide": v.get("main_camera_wide"),
        "reverse_angle_suspected": v.get("reverse_angle_suspected"),
    }


def _base_record(idx: int, image_size: tuple[int, int], pitch: PitchModel) -> dict:
    return {
        "frame_index": idx, "correspondences": [], "H": None,
        "H_convention": "normalised_image_to_pitch_m",
        "pitch_dims": [pitch.length, pitch.width], "image_size": list(image_size),
        "method": "ransac", "inliers": [], "mean_err_m": None, "err_px": [],
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }


def failed_record(idx, image_size, pitch, reason, res=None, labels=None) -> dict:
    rec = _base_record(idx, image_size, pitch)
    rec.update(status="failed", score=0.0, failure=reason, view=_view_record(res) if res else None)
    if labels:
        rec["correspondences"] = [
            {"image": list(lb.uv), "pitch_id": lb.keypoint_id,
             "pitch_xy": list(pitch.keypoints()[lb.keypoint_id]), "confidence": lb.confidence,
             "candidate_id": lb.candidate_id}
            for lb in labels
        ]
    return rec


def proposal_record(idx, image_size, pitch, fit, res) -> dict:
    kps = pitch.keypoints()
    rec = _base_record(idx, image_size, pitch)
    rec["correspondences"] = [
        {"image": [float(lb.uv[0]), float(lb.uv[1])], "pitch_id": lb.keypoint_id,
         "pitch_xy": list(kps[lb.keypoint_id]), "confidence": lb.confidence, "candidate_id": lb.candidate_id}
        for lb in fit.labels
    ]
    rec["H"] = fit.H.tolist()
    rec["inliers"] = fit.inliers
    rec["mean_err_m"] = fit.mean_err_m
    rec["err_px"] = fit.err_px
    rec.update(status="proposed", score=fit.score, failure=None, view=_view_record(res))
    return rec


def overlay_render(frame_bgr: np.ndarray, H: np.ndarray, pitch: PitchModel) -> np.ndarray:
    """Frame (<=1568 px) with the pitch model projected through ``H``."""
    from footy_track.calibration.prelabel.candidates import resize_to_work  # noqa: PLC0415

    img, _ = resize_to_work(frame_bgr)
    h, w = img.shape[:2]
    for name, runs in geo.project_pitch_model(np.asarray(H), pitch).items():
        col = (0, 215, 255) if ("CIRCLE" in name or "ARC" in name) else (255, 43, 214)
        for run in runs:
            cv2.polylines(img, [np.round(run * np.array([w, h])).astype(np.int32)], False, col, 2, cv2.LINE_AA)
    return img


# --------------------------------------------------------------------------
# run
# --------------------------------------------------------------------------


@dataclass
class RunConfig:
    gt_dir: Path
    clips_dir: Path
    clips: list[str]
    frames: list[int] | None = None  # explicit frames (applied to every clip)
    every: int = 25
    skip_gt: bool = False
    limit: int | None = None  # max frames processed in total
    pitch_dims: tuple[float, float] = (DEFAULT_LENGTH, DEFAULT_WIDTH)
    render_dir: Path | None = None
    verify: bool = False
    run_id: str | None = None


@dataclass
class RunSummary:
    run_id: str
    n_frames: int = 0
    n_proposed: int = 0
    n_failed: int = 0
    per_clip: dict = field(default_factory=dict)
    usage_total: dict = field(default_factory=dict)


def _add_usage(total: dict, u: dict | None) -> None:
    for k in ("input_tokens", "output_tokens", "cache_creation_input_tokens", "cache_read_input_tokens"):
        if u and u.get(k):
            total[k] = total.get(k, 0) + u[k]


def run_prelabel(cfg: RunConfig, identifier: Identifier, log=print) -> RunSummary:
    pitch = PitchModel(*cfg.pitch_dims)
    run_id = cfg.run_id or make_run_id(identifier.run_prefix())
    diagram = jpeg(render_diagram(pitch))
    if cfg.render_dir:
        Path(cfg.render_dir).mkdir(parents=True, exist_ok=True)
        (Path(cfg.render_dir) / "diagram.jpg").write_bytes(diagram)
    summary = RunSummary(run_id)
    usage_log: list[dict] = []
    created = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    remaining = cfg.limit

    for clip in cfg.clips:
        canon = canonical_clip_id(clip)
        video = find_video(cfg.clips_dir, clip)
        if video is None:
            log(f"[{clip}] video not found in {cfg.clips_dir}; skipped")
            continue
        skip = gt_frames(cfg.gt_dir, clip) if cfg.skip_gt else set()
        frames = select_frames(video_frame_count(video), cfg.every, cfg.frames, skip)
        if remaining is not None:
            frames = frames[:remaining]
            remaining -= len(frames)
        for idx in frames:
            frame = read_frame(video, idx)
            if frame is None:
                log(f"[{canon}:{idx}] cannot read frame")
                continue
            ih, iw = frame.shape[:2]
            det = find_candidates(frame)
            cand_img = render_candidates(det)
            cand_jpg = jpeg(cand_img)
            if cfg.render_dir:
                (Path(cfg.render_dir) / f"{canon}_{idx}_candidates.jpg").write_bytes(cand_jpg)
            res = identifier.identify(
                IdentifyInput(clip, idx, pitch, det, cand_jpg, diagram)
            )
            if res.usage:
                usage_log.append({"clip": canon, "frame": idx, **res.usage})
                _add_usage(summary.usage_total, res.usage)
            resolve_extra_points(res, det)
            if res.error:
                rec = failed_record(idx, (iw, ih), pitch, f"identifier error: {res.error}", res)
            elif res.unusable:
                rec = failed_record(idx, (iw, ih), pitch, f"unusable: {res.unusable_reason or 'unspecified'}", res)
            else:
                fit = fit_and_score(res.labels, pitch, det, (iw, ih))
                if not fit.ok:
                    rec = failed_record(idx, (iw, ih), pitch, fit.failure, res, fit.labels)
                else:
                    rec = proposal_record(idx, (iw, ih), pitch, fit, res)
                    ov = overlay_render(frame, fit.H, pitch)
                    if cfg.render_dir:
                        cv2.imwrite(str(Path(cfg.render_dir) / f"{canon}_{idx}_overlay.jpg"), ov)
                    if cfg.verify and hasattr(identifier, "verify"):
                        v = identifier.verify(jpeg(ov))
                        _add_usage(summary.usage_total, v.get("usage"))
                        usage_log.append({"clip": canon, "frame": idx, "call": "verify", **(v.get("usage") or {})})
                        rec["verify"] = {"match": v["match"], "reason": v["reason"]}
            if res.extra_points:
                rec["extra_points"] = res.extra_points
            rec["n_candidates"] = len(det.candidates)
            summary.n_frames += 1
            if rec["status"] == "proposed":
                summary.n_proposed += 1
            else:
                summary.n_failed += 1
            log(
                f"[{canon}:{idx}] {rec['status']}"
                + (f" score={rec['score']:.2f} inliers={sum(rec['inliers'])}/{len(rec['inliers'])}"
                   if rec["status"] == "proposed" else f" ({rec['failure']})")
            )
            params = {**identifier.run_params(), "n_candidates_max": 70, "usage": usage_log,
                      "usage_total": summary.usage_total}

            def _store(doc, rec=rec, idx=idx, params=params):
                run = doc["runs"].setdefault(
                    run_id, {"source": identifier.source, "created": created, "params": params, "frames": {}}
                )
                run["params"] = params
                run["frames"][str(idx)] = rec

            update_proposals(cfg.gt_dir, clip, _store)
            summary.per_clip[canon] = summary.per_clip.get(canon, 0) + 1
    return summary
