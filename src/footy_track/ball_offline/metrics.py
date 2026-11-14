# ruff: noqa: PLR0912, PLR0915, PLR0124, PLR0917, PLW2901, C408
"""Offline ball-tracking metrics (pixel coordinates at 1080p).

Scoring rules (sparse human GT, so most frames are unlabelled):

* a *scored* frame is a GT ball frame (human-placed or vittrack+human_checked)
  or an explicit ``no_ball`` frame.  Everything else is ignored, including
  ``not_broadcast`` and ``out_of_play_ball`` frames;
* GT ball frame, prediction within R px of GT centre -> TP;
* GT ball frame, prediction further than R px -> FP (and the frame is a miss);
* GT ball frame, no prediction -> FN;
* ``no_ball`` frame with a prediction -> FP;
* recall = TP / n_gt_ball; precision = TP / (TP + FP); F1 = harmonic mean;
* coverage = fraction of *all* frames of the clip that carry a prediction.

Counts, not just rates, are always returned so a headline is never printed
without its denominator.
"""

from __future__ import annotations

import dataclasses
import math

import numpy as np

RADII = (5.0, 10.0, 20.0)

Pred = dict[int, tuple[float, float]]


@dataclasses.dataclass
class Counts:
    n_frames: int = 0
    n_pred_frames: int = 0
    n_gt_ball: int = 0
    n_no_ball: int = 0
    n_pred_on_gt_ball: int = 0
    n_pred_on_no_ball: int = 0
    tp: dict[float, int] = dataclasses.field(
        default_factory=lambda: dict.fromkeys(RADII, 0)
    )
    fp: dict[float, int] = dataclasses.field(
        default_factory=lambda: dict.fromkeys(RADII, 0)
    )
    errs: list[float] = dataclasses.field(default_factory=list)

    def __iadd__(self, o: Counts) -> Counts:
        for f in (
            "n_frames",
            "n_pred_frames",
            "n_gt_ball",
            "n_no_ball",
            "n_pred_on_gt_ball",
            "n_pred_on_no_ball",
        ):
            setattr(self, f, getattr(self, f) + getattr(o, f))
        for r in RADII:
            self.tp[r] += o.tp[r]
            self.fp[r] += o.fp[r]
        self.errs += o.errs
        return self

    def summary(self) -> dict:
        out = {
            "n_frames": self.n_frames,
            "n_gt_ball": self.n_gt_ball,
            "n_no_ball": self.n_no_ball,
            "coverage": self.n_pred_frames / self.n_frames
            if self.n_frames
            else float("nan"),
            "median_err_px": float(np.median(self.errs)) if self.errs else float("nan"),
        }
        for r in RADII:
            tp, fp = self.tp[r], self.fp[r]
            rec = tp / self.n_gt_ball if self.n_gt_ball else float("nan")
            prec = tp / (tp + fp) if (tp + fp) else float("nan")
            f1 = (
                2 * prec * rec / (prec + rec)
                if prec == prec and rec == rec and (prec + rec) > 0
                else float("nan")
            )
            k = int(r)
            out.update(
                {
                    f"tp@{k}": tp,
                    f"fp@{k}": fp,
                    f"recall@{k}": rec,
                    f"precision@{k}": prec,
                    f"f1@{k}": f1,
                }
            )
        return out


def score_clip(
    pred: Pred,
    gt: dict[int, tuple[float, float]],
    no_ball: set[int],
    n_frames: int,
    *,
    scored: set[int] | None = None,
) -> Counts:
    """Score ``pred`` (frame -> (x, y)) on the GT frames in ``scored`` (default
    all GT/no_ball frames)."""
    c = Counts(n_frames=n_frames, n_pred_frames=len(pred))
    for t, (gx, gy) in gt.items():
        if scored is not None and t not in scored:
            continue
        c.n_gt_ball += 1
        p = pred.get(t)
        if p is None:
            continue
        c.n_pred_on_gt_ball += 1
        d = math.hypot(p[0] - gx, p[1] - gy)
        c.errs.append(d)
        for r in RADII:
            if d <= r:
                c.tp[r] += 1
            else:
                c.fp[r] += 1
    for t in no_ball:
        if scored is not None and t not in scored:
            continue
        c.n_no_ball += 1
        if t in pred:
            c.n_pred_on_no_ball += 1
            for r in RADII:
                c.fp[r] += 1
    return c


def ceiling(
    clip, scored: set[int] | None = None, r: float = 10.0, min_conf: float = 0.1
):
    """(frames with any candidate >= min_conf within r px of GT, n GT frames)."""
    hit = n = 0
    for t, g in clip.gt.items():
        if scored is not None and t not in scored:
            continue
        n += 1
        xy, conf = clip.cands(t)
        if len(xy):
            d = np.hypot(xy[:, 0] - g.xy[0], xy[:, 1] - g.xy[1])
            if np.any((d <= r) & (conf >= min_conf)):
                hit += 1
    return hit, n


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (max(0.0, centre - half), min(1.0, centre + half))
