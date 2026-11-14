"""Re-run detection + line refinement on stored proposals, reusing Claude's labels (no Claude calls).

Only ``status == "proposed"`` records are touched; each keeps its old record under ``previous``.
"""

from __future__ import annotations

import subprocess
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

from footy_track.calibration.pitch_model import PitchModel
from footy_track.calibration.prelabel.proposals import (
    list_clips_with_proposals,
    read_proposals,
    update_proposals,
)
from footy_track.clip_id import canonical_clip_id


def code_version() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=Path(__file__).parent, text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def labels_from_record(rec: dict):
    """The input Claude originally gave: as-labelled points (no synthetic padding) and line labels."""
    from footy_track.calibration.prelabel.fit import Label  # noqa: PLC0415
    from footy_track.calibration.refine import LineLabel  # noqa: PLC0415

    labels = [
        Label(c["pitch_id"], tuple(c.get("image_label") or c["image"]), c.get("confidence", 1.0), c.get("candidate_id"))
        for c in rec.get("correspondences", []) if not c.get("synthetic") and c.get("pitch_id")
    ]
    lines = [LineLabel(ll["line_id"], [tuple(p) for p in ll["points"]], ll.get("confidence", 1.0))
             for ll in rec.get("line_labels") or []]
    return labels, lines


def new_record(old: dict, frame, pitch: PitchModel, fit_k1: bool) -> dict:
    """Fresh record for ``old`` from ``frame``. Pure: no I/O."""
    from footy_track.calibration.prelabel.candidates import find_candidates  # noqa: PLC0415
    from footy_track.calibration.prelabel.fit import fit_and_score  # noqa: PLC0415
    from footy_track.calibration.prelabel.pipeline import failed_record, proposal_record  # noqa: PLC0415

    ih, iw = frame.shape[:2]
    det = find_candidates(frame)
    labels, lines = labels_from_record(old)
    fit = fit_and_score(labels, pitch, det, (iw, ih), line_labels=lines, refine=True, fit_k1=fit_k1)
    view = old.get("view")

    class _Res:  # carries the stored view into the record builders
        pass

    res = _Res()
    res.view, res.unusable, res.unusable_reason = view or {}, False, None
    rec = failed_record(old["frame_index"], (iw, ih), pitch, fit.failure, res, fit.labels) if not fit.ok \
        else proposal_record(old["frame_index"], (iw, ih), pitch, fit, res)
    rec["view"] = view  # keep exactly what was stored
    rec["n_candidates"] = len(det.candidates)
    if old.get("verify"):
        rec["verify"] = old["verify"]
    return rec


def finalise(old: dict, rec: dict, fit_k1: bool, code: str) -> dict:
    prev = {k: v for k, v in old.items() if k not in ("previous", "rerefined")}
    rec["previous"] = prev
    rec["rerefined"] = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "code": code, "fit_k1": fit_k1}
    return rec


def _work(job):
    from footy_track.calibration.prelabel.pipeline import find_video  # noqa: PLC0415
    from footy_track.calibration.video import read_frame  # noqa: PLC0415

    clip, run_id, idx, old, clips_dirs, fit_k1, pitch_dims = job
    try:
        video = next((v for d in clips_dirs if (v := find_video(Path(d), clip))), None)
        if video is None:
            return clip, run_id, idx, None, "video not found"
        frame = read_frame(video, int(idx))
        if frame is None:
            return clip, run_id, idx, None, "cannot read frame"
        return clip, run_id, idx, new_record(old, frame, PitchModel(*pitch_dims), fit_k1), None
    except Exception as e:  # noqa: BLE001
        return clip, run_id, idx, None, f"{type(e).__name__}: {e}"


@dataclass
class RerefineSummary:
    rerefined: int = 0
    now_failed: int = 0
    kept_previous: int = 0  # new refine not applied but the old one was: the old record is kept
    unchanged: int = 0  # not proposed, or status changed under us
    errors: list = field(default_factory=list)
    scores: list = field(default_factory=list)  # (clip, run, idx, before, after)
    k1: list = field(default_factory=list)


def rerefine(gt_dir: Path, clips_dirs, clips=None, runs=None, fit_k1=True, dry_run=False, workers=2,
             log=print) -> RerefineSummary:
    gt_dir = Path(gt_dir)
    clips_dirs = [str(d) for d in (clips_dirs if isinstance(clips_dirs, (list, tuple)) else [clips_dirs])]
    want = {canonical_clip_id(c) for c in clips} if clips else None
    s = RerefineSummary()
    jobs = []
    for clip in list_clips_with_proposals(gt_dir):
        if want and canonical_clip_id(clip) not in want:
            continue
        doc = read_proposals(gt_dir, clip)
        for run_id, run in doc["runs"].items():
            if runs and run_id not in runs:
                continue
            for idx, rec in run["frames"].items():
                if rec.get("status") == "proposed":
                    pd = tuple(rec.get("pitch_dims") or (105.0, 68.0))
                    jobs.append((clip, run_id, idx, rec, clips_dirs, fit_k1, pd))
                else:
                    s.unchanged += 1
    log(f"{len(jobs)} proposed items to re-refine ({s.unchanged} non-proposed left alone)")
    if dry_run:
        return s
    code = code_version()
    results: dict[str, list] = {}
    ex = ProcessPoolExecutor(max_workers=workers) if workers > 1 else None
    try:
        for clip, run_id, idx, rec, err in (ex.map(_work, jobs) if ex else map(_work, jobs)):
            if err:
                s.errors.append((clip, run_id, idx, err))
                log(f"[{clip}:{idx}] ERROR {err}")
                continue
            results.setdefault(clip, []).append((run_id, idx, rec))
            log(f"[{clip}:{idx}] {rec['status']} score={rec['score']:.2f}")
    finally:
        if ex:
            ex.shutdown()
    sent = {(j[0], j[1], j[2]): j[3] for j in jobs}
    for clip, items in results.items():
        def _store(doc, items=items, clip=clip):
            for run_id, idx, rec in items:
                cur = doc["runs"].get(run_id, {}).get("frames", {}).get(idx)
                old = sent[(clip, run_id, idx)]
                if cur is None or cur.get("status") != "proposed" or cur.get("ts") != old.get("ts"):
                    s.unchanged += 1  # reviewed (or replaced) meanwhile: leave alone
                    continue
                if not (rec.get("refine") or {}).get("applied") and (cur.get("refine") or {}).get("applied"):
                    s.kept_previous += 1  # never swap a refined H for a re-scored coarse fit
                    continue
                doc["runs"][run_id]["frames"][idx] = finalise(cur, rec, fit_k1, code)
                if rec["status"] == "proposed":
                    s.rerefined += 1
                else:
                    s.now_failed += 1
                s.scores.append((clip, run_id, idx, cur.get("score"), rec.get("score")))
                s.k1.append((rec.get("distortion") or {}).get("k1", 0.0))
        update_proposals(gt_dir, clip, _store)
    return s
