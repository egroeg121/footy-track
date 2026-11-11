"""Noise sweep: how precisely must the identifier localise points?

For each human-GT frame and each noise level sigma (px at 1920 wide) the mock
identifier perturbs the GT points (several seeds), the coarse fit is run, then
line refinement; the GT reprojection error of the homography before and after
refinement gives the success rates (mean error <= 5 px / <= 10 px). Read-only
on the GT dir.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np

from footy_track.calibration.pitch_model import PitchModel
from footy_track.calibration.prelabel.candidates import find_candidates
from footy_track.calibration.prelabel.evaluate import gt_px_errors, read_gt
from footy_track.calibration.prelabel.fit import fit_and_score
from footy_track.calibration.prelabel.identifier import IdentifyInput, MockIdentifier
from footy_track.calibration.video import read_frame

THRESHOLDS = (5.0, 10.0)


def _task(args: tuple) -> dict:
    cv2.setNumThreads(1)  # workers already saturate the cores
    gt_dir, video, clip, frame_idx, sigma, seed, drop, mislabel, lines = args
    pitch = PitchModel()
    gt = read_gt(Path(gt_dir), clip)[str(frame_idx)]
    frame = read_frame(video, frame_idx)
    det = find_candidates(frame)
    ih, iw = frame.shape[:2]
    ident = MockIdentifier(gt_dir, sigma_px=sigma, drop_prob=drop, mislabel_prob=mislabel, lines=lines, seed=seed)
    res = ident.identify(IdentifyInput(clip, frame_idx, pitch, det, b"", b""))
    row = {"frame": frame_idx, "sigma": sigma, "seed": seed, "n_labels": len(res.labels),
           "ok": False, "pre_px": None, "post_px": None, "refined": None, "failure": None, "score": None, "dt": None}
    fit = fit_and_score(res.labels, pitch, det, (iw, ih), seed=seed, line_labels=res.line_labels, refine=True)
    if not fit.ok and fit.H is None:
        row["failure"] = fit.failure
        return row
    row["failure"], row["ok"] = fit.failure, bool(fit.ok)
    if fit.H_coarse is not None:
        e = gt_px_errors(fit.H_coarse, gt)
        row["pre_px"] = float(np.nanmean(e)) if np.isfinite(e).any() else None
    if fit.H is not None:
        e = gt_px_errors(fit.H, gt)
        row["post_px"] = float(np.nanmean(e)) if np.isfinite(e).any() else None
    row["refined"] = bool(fit.refine and fit.refine.applied)
    row["score"], row["dt"] = fit.score, fit.dt_residual
    return row


def noise_sweep(
    gt_dir: Path, clips_dir: Path, clip: str, frames: list[int], sigmas=(0, 10, 20, 40), n_seeds: int = 8,
    drop_prob: float = 0.0, mislabel_prob: float = 0.0, lines: bool = False, workers: int = 1,
) -> dict:
    from footy_track.calibration.prelabel.pipeline import find_video  # noqa: PLC0415

    video = find_video(Path(clips_dir), clip)
    if video is None:
        raise FileNotFoundError(f"no video for {clip} in {clips_dir}")
    jobs = [
        (str(gt_dir), str(video), clip, f, float(s), seed, drop_prob, mislabel_prob, lines)
        for s in sigmas for f in frames for seed in range(n_seeds)
    ]
    if workers > 1:
        with ProcessPoolExecutor(workers) as ex:
            rows = list(ex.map(_task, jobs, chunksize=1))
    else:
        rows = [_task(j) for j in jobs]
    summary = []
    for s in sigmas:
        rs = [r for r in rows if r["sigma"] == float(s)]
        n = len(rs)
        item = {"sigma": float(s), "n": n, "n_fit_failed": sum(not r["ok"] for r in rs),
                "refine_applied": sum(bool(r["refined"]) for r in rs),
                # proposals that reach a reviewer although they are wrong (> 10 px after refinement)
                "n_bad_proposed": sum(bool(r["ok"]) and r["post_px"] is not None and r["post_px"] > 10 for r in rs)}
        for stage in ("pre", "post"):
            vals = [r[f"{stage}_px"] for r in rs]
            for t in THRESHOLDS:
                item[f"{stage}_le{t:g}"] = sum(v is not None and v <= t for v in vals) / n if n else None
            ok = [v for v in vals if v is not None]
            item[f"{stage}_median_px"] = float(np.median(ok)) if ok else None
        summary.append(item)
    return {"clip": clip, "frames": frames, "n_seeds": n_seeds, "drop_prob": drop_prob,
            "mislabel_prob": mislabel_prob, "lines": lines, "summary": summary, "rows": rows}


def format_sweep(res: dict) -> str:
    def pct(v):
        return "  -  " if v is None else f"{100 * v:4.0f}%"

    def px(v):
        return "   -  " if v is None else f"{v:6.1f}"

    head = (f"label noise sweep: {res['clip']} frames {res['frames']} x {res['n_seeds']} seeds"
            f" (drop={res['drop_prob']}, mislabel={res['mislabel_prob']}, line labels={res['lines']})")
    cols = (f"{'sigma':>6s} {'n':>4s} | {'<=5px pre':>9s} {'post':>5s} | {'<=10px pre':>10s} {'post':>5s} |"
            f" {'median px pre':>13s} {'post':>6s} | {'refined':>7s} {'rej':>4s} {'bad':>4s}")
    lines = [head, cols, "-" * len(cols)]
    for r in res["summary"]:
        lines.append(
            f"{r['sigma']:6g} {r['n']:4d} | {pct(r['pre_le5']):>9s} {pct(r['post_le5']):>5s} |"
            f" {pct(r['pre_le10']):>10s} {pct(r['post_le10']):>5s} |"
            f" {px(r['pre_median_px']):>13s} {px(r['post_median_px']):>6s} |"
            f" {r['refine_applied']:7d} {r['n_fit_failed']:4d} {r['n_bad_proposed']:4d}"
        )
    lines.append("pre/post: coarse / refined fit, every try counted; refined: refinement applied; rej: no proposal "
                 "(no fit, or lines do not match the mask); bad: proposed but > 10 px")
    return "\n".join(lines)
