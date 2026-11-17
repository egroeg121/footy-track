"""Detector precision / recall / AP on the tracking dataset items, from stored detections (CPU).

The GPU half (:mod:`footy_track.datasets.detector_run`) runs the detector over every frame of every
eval clip once and stores the raw detections (``<clip>.npz``, confidence floor 0.1, all classes).
This module only *scores* them against the items of a dataset snapshot, so a new dataset version
whose eval clips already have detections costs no GPU time, and the scoring is plain numpy
(unit-testable without weights or video).

Ground truth (one tracking item = one labelled frame; boxes carry a provenance tier)
-----------------------------------------------------------------------------------
* tier 1 = placed by a human, tier 2 = a machine box (VitTrack) a human confirmed, tier 3 = machine
  only (never confirmed). ``rejected`` boxes (Ball Check said "not the ball") are not GT.
* **player / referee**: tier 1 + 2 boxes are GT. Tier 3 boxes and ``coach`` / ``person`` /
  ``player_sub`` boxes are *ignore regions*: a detection that matches one is neither a hit nor a
  false positive. A frame is scored for a class only when it has at least one tier 1-2 box of that
  class; labelling is not exhaustive, so a frame with no labelled referee says nothing about
  referee false positives. Player / referee precision is therefore a *lower bound on labelled
  frames' precision* (an unlabelled real player the detector finds still counts as a false positive
  on a frame that has other labelled players), and recall is exact for the labelled boxes.
* **ball** (``in_play_ball`` / ``ball`` / ``out_of_play_ball`` GT; ``in_play_ball`` / ``ball``
  detections): scored on frames with a GT ball in the chosen tier set, plus ``no_ball`` frames
  (every ball detection there is a false positive). ``not_broadcast`` frames are never scored.
  Two tier sets: ``ball_*`` = tier 1 only (humans, the same GT as the ball-tracker rows) and
  ``ball_t12_*`` = tier 1 + 2 (human-confirmed machine boxes; roughly 3x more instances).

Matching
--------
Greedy per frame and class, detections in descending confidence, each taking the best unmatched GT:
players / referees by **IoU >= 0.5**; the ball three ways: IoU >= 0.5 (comparable with the Roboflow /
Ultralytics ``mAP50`` in the detector leaderboard) and **centre distance <= 10 px / 25 px** (native
pixels; the same radii as the ball tracker). The ball is ~11 px wide, so an IoU of 0.5 needs the
centre within ~3 px: that rule mostly measures box tightness. The centre-distance rules measure
"did it find the ball", which is what tracking needs, so the headline for tracking is the 10 px
rows and the IoU row is only there to reconcile with the leaderboard.

Metrics
-------
``<cls>_ap50`` (IoU, 101-point interpolated like Ultralytics, over every stored detection; ball: ``ball_ap_iou50``),
``<cls>_precision`` / ``<cls>_recall`` / ``<cls>_f1`` at ``CONF_THRESHOLD`` (0.3, the pipeline's
default ``min_confidence``) and ``<cls>_f1_max`` (best F1 over all thresholds, with
``<cls>_conf_at_f1_max``), ``<cls>_n_gt``. Ball metrics carry the match rule as a suffix
(``_iou50`` / ``_10px`` / ``_25px``). ``map50`` is the mean AP50 of player / referee / ball (IoU) over
classes that have ground truth. ``coverage`` is the fraction of eligible items whose frame has stored
detections.
"""

from __future__ import annotations

import io
import json
import os
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

CONF_FLOOR = 0.1
CONF_THRESHOLD = 0.3
IOU_THR = 0.5
RADII_PX = (10.0, 25.0)

BALL_GT_LABELS = frozenset({"in_play_ball", "ball", "out_of_play_ball"})
BALL_DET_LABELS = frozenset({"in_play_ball", "ball"})
IGNORE_GT_LABELS = frozenset({"coach", "person", "player_sub"})
TIERS_HUMAN = (1,)
TIERS_CONFIRMED = (1, 2)


# --------------------------------------------------------------------------------------------
# stored detections
# --------------------------------------------------------------------------------------------
@dataclass
class ClipDetections:
    """Every detection (conf >= floor) the detector made on every processed frame of one clip.

    Boxes are normalised top-left ``(x, y, w, h)``. ``frames`` lists every processed frame index
    (so "no detections" and "never processed" differ).
    """

    clip: str
    size: tuple[int, int]
    names: list[str]
    frames: np.ndarray
    frame_idx: np.ndarray
    cls: np.ndarray
    box: np.ndarray
    conf: np.ndarray
    meta: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        order = np.argsort(self.frame_idx, kind="stable")
        self.frame_idx, self.cls, self.box, self.conf = (
            self.frame_idx[order], self.cls[order], self.box[order], self.conf[order],
        )
        self._done = {int(f) for f in self.frames}
        uniq, first, counts = np.unique(self.frame_idx, return_index=True, return_counts=True)
        self._slices = {int(f): (int(a), int(a + n)) for f, a, n in zip(uniq, first, counts, strict=True)}

    def has_frame(self, f: int) -> bool:
        return int(f) in self._done

    def frame(self, f: int) -> tuple[list[str], np.ndarray, np.ndarray]:
        """``(labels, boxes (n,4), conf (n,))`` of frame ``f`` (empty when nothing was detected)."""
        a, b = self._slices.get(int(f), (0, 0))
        return [self.names[int(c)] for c in self.cls[a:b]], self.box[a:b], self.conf[a:b]

    # ---- persistence: one npz per clip, written atomically
    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        buf = io.BytesIO()
        np.savez_compressed(
            buf, frames=self.frames.astype(np.int32), frame_idx=self.frame_idx.astype(np.int32),
            cls=self.cls.astype(np.int16), box=self.box.astype(np.float32), conf=self.conf.astype(np.float32),
            header=np.array(json.dumps({"clip": self.clip, "size": list(self.size), "names": self.names,
                                        "meta": self.meta})),
        )
        tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}")
        tmp.write_bytes(buf.getvalue())
        os.replace(tmp, path)

    @classmethod
    def load(cls, path: Path) -> ClipDetections:
        with np.load(path, allow_pickle=False) as z:
            h = json.loads(str(z["header"]))
            return cls(h["clip"], (int(h["size"][0]), int(h["size"][1])), list(h["names"]), z["frames"],
                       z["frame_idx"], z["cls"], z["box"], z["conf"], h.get("meta", {}))

    @classmethod
    def from_rows(cls, clip: str, size: tuple[int, int], names: Sequence[str],
                  rows: Iterable[tuple[int, str, tuple[float, float, float, float], float]],
                  frames: Iterable[int], meta: dict | None = None) -> ClipDetections:
        names = list(names)
        idx = {n: i for i, n in enumerate(names)}
        fi, c, b, s = [], [], [], []
        for f, label, box, conf in rows:
            if label not in idx:
                idx[label] = len(names)
                names.append(label)
            fi.append(f); c.append(idx[label]); b.append(box); s.append(conf)  # noqa: E702
        return cls(clip, size, names, np.array(sorted({int(f) for f in frames}), dtype=np.int32),
                   np.array(fi, dtype=np.int32), np.array(c, dtype=np.int16),
                   np.array(b, dtype=np.float32).reshape(-1, 4), np.array(s, dtype=np.float32), meta or {})


@dataclass
class DetectionStore:
    """The detector's stored output for a set of clips: the "model" the adapter scores.

    ``<root>/<clip>.npz`` per canonical clip. ``name`` / ``version`` identify the detector
    (``version`` = weights sha256) for the eval_result rows.
    """

    root: Path
    name: str = "detector"
    version: str = "unknown"
    _cache: dict[str, ClipDetections | None] = field(default_factory=dict, repr=False)

    def get(self, clip: str) -> ClipDetections | None:
        if clip not in self._cache:
            p = Path(self.root) / f"{clip}.npz"
            self._cache[clip] = ClipDetections.load(p) if p.is_file() else None
        return self._cache[clip]


# --------------------------------------------------------------------------------------------
# ground truth of one item
# --------------------------------------------------------------------------------------------
Box = tuple[float, float, float, float]


@dataclass
class FrameGT:
    marker: str | None = None
    player: list[Box] = field(default_factory=list)      # tier 1-2
    referee: list[Box] = field(default_factory=list)     # tier 1-2
    ignore_player: list[Box] = field(default_factory=list)   # tier 3 players, coach/person/player_sub, any tier
    ignore_referee: list[Box] = field(default_factory=list)  # tier 3 referees
    ball: dict[int, list[Box]] = field(default_factory=dict)  # tier -> live ball boxes


def frame_gt(payload: dict) -> FrameGT:
    gt = FrameGT(marker=payload.get("marker"))
    for b in payload.get("boxes", ()):
        if b.get("rejected"):
            continue
        label = str(b["label"]).lower()
        box = (float(b["x"]), float(b["y"]), float(b["w"]), float(b["h"]))
        tier = int(b["tier"])
        if label in BALL_GT_LABELS:
            gt.ball.setdefault(tier, []).append(box)
        elif label in IGNORE_GT_LABELS:
            gt.ignore_player.append(box)
        elif label == "player":
            (gt.player if tier <= 2 else gt.ignore_player).append(box)
        elif label == "referee":
            (gt.referee if tier <= 2 else gt.ignore_referee).append(box)
    return gt


# --------------------------------------------------------------------------------------------
# matching + AP
# --------------------------------------------------------------------------------------------
def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU of ``a (n,4)`` against ``b (m,4)``, both normalised top-left xywh."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    ax2, ay2 = a[:, 0] + a[:, 2], a[:, 1] + a[:, 3]
    bx2, by2 = b[:, 0] + b[:, 2], b[:, 1] + b[:, 3]
    iw = np.clip(np.minimum(ax2[:, None], bx2[None]) - np.maximum(a[:, 0][:, None], b[:, 0][None]), 0, None)
    ih = np.clip(np.minimum(ay2[:, None], by2[None]) - np.maximum(a[:, 1][:, None], b[:, 1][None]), 0, None)
    inter = iw * ih
    union = (a[:, 2] * a[:, 3])[:, None] + (b[:, 2] * b[:, 3])[None] - inter
    return np.where(union > 0, inter / np.where(union > 0, union, 1), 0.0)


def centre_dist_px(a: np.ndarray, b: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Centre distance in native pixels of ``a (n,4)`` against ``b (m,4)`` (normalised xywh)."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    ac = np.stack([(a[:, 0] + a[:, 2] / 2) * size[0], (a[:, 1] + a[:, 3] / 2) * size[1]], 1)
    bc = np.stack([(b[:, 0] + b[:, 2] / 2) * size[0], (b[:, 1] + b[:, 3] / 2) * size[1]], 1)
    return np.hypot(ac[:, None, 0] - bc[None, :, 0], ac[:, None, 1] - bc[None, :, 1])


def _arr(boxes: Sequence[Box]) -> np.ndarray:
    return np.array(boxes, dtype=np.float64).reshape(-1, 4)


@dataclass
class Tally:
    """Detections (score, is-hit) plus the ground-truth count of one class / rule, pooled over frames."""

    scores: list[float] = field(default_factory=list)
    hits: list[bool] = field(default_factory=list)
    n_gt: int = 0

    def add_frame(self, det_boxes: np.ndarray, det_conf: np.ndarray, gt: Sequence[Box], ignore: Sequence[Box],
                  size: tuple[int, int], *, radius_px: float | None) -> None:
        """Match one frame. ``radius_px`` None = IoU >= 0.5, else centre distance <= radius_px."""
        g, ig = _arr(gt), _arr(ignore)
        self.n_gt += len(g)
        order = np.argsort(-det_conf, kind="stable")
        d = det_boxes[order]
        conf = det_conf[order]
        if radius_px is None:
            m_gt, m_ig = iou_matrix(d, g), iou_matrix(d, ig)
            ok_gt, ok_ig = m_gt >= IOU_THR, m_ig >= IOU_THR
            score_gt = m_gt
        else:
            dist_gt, dist_ig = centre_dist_px(d, g, size), centre_dist_px(d, ig, size)
            ok_gt, ok_ig = dist_gt <= radius_px, dist_ig <= radius_px
            score_gt = -dist_gt
        taken = np.zeros(len(g), dtype=bool)
        for i in range(len(d)):
            cand = np.where(ok_gt[i] & ~taken)[0] if len(g) else np.array([], dtype=int)
            if len(cand):
                j = cand[np.argmax(score_gt[i, cand])]
                taken[j] = True
                self.scores.append(float(conf[i])); self.hits.append(True)  # noqa: E702
            elif len(ig) and ok_ig[i].any():
                continue  # lands on an ignore region: neither a hit nor a false positive
            else:
                self.scores.append(float(conf[i])); self.hits.append(False)  # noqa: E702

    # ---- metrics
    def ap(self) -> float | None:
        if self.n_gt == 0:
            return None
        if not self.scores:
            return 0.0
        order = np.argsort(-np.array(self.scores), kind="stable")
        tp = np.cumsum(np.array(self.hits)[order])
        fp = np.cumsum(~np.array(self.hits)[order])
        recall, precision = tp / self.n_gt, tp / np.maximum(tp + fp, 1)
        mrec, mpre = np.concatenate([[0.0], recall, [1.0]]), np.concatenate([[1.0], precision, [0.0]])
        mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))
        x = np.linspace(0, 1, 101)
        return float(np.trapezoid(np.interp(x, mrec, mpre), x))

    def prf(self, conf: float) -> tuple[float | None, float | None, float | None]:
        s, h = np.array(self.scores), np.array(self.hits, dtype=bool)
        keep = s >= conf
        tp, n = int(h[keep].sum()), int(keep.sum())
        p = tp / n if n else None
        r = tp / self.n_gt if self.n_gt else None
        f1 = 2 * p * r / (p + r) if p is not None and r is not None and (p + r) else (0.0 if p is not None and r is not None else None)
        return p, r, f1

    def best_f1(self) -> tuple[float | None, float | None]:
        """(max F1 over thresholds, the confidence at which it occurs)."""
        if self.n_gt == 0 or not self.scores:
            return (None if self.n_gt == 0 else 0.0), None
        s = np.array(self.scores)
        best, at = 0.0, None
        for t in np.unique(s):
            f1 = self.prf(float(t))[2] or 0.0
            if f1 > best:
                best, at = f1, float(t)
        return best, at


def _rule_suffix(radius: float | None) -> str:
    return "iou50" if radius is None else f"{radius:g}px"


def _put(metrics: dict, prefix: str, t: Tally, suffix: str = "") -> None:
    sfx = f"_{suffix}" if suffix else ""
    p, r, f1 = t.prf(CONF_THRESHOLD)
    fm, at = t.best_f1()
    metrics[f"{prefix}_ap{sfx}" if suffix else f"{prefix}_ap50"] = t.ap()
    metrics[f"{prefix}_precision{sfx}"] = p
    metrics[f"{prefix}_recall{sfx}"] = r
    metrics[f"{prefix}_f1{sfx}"] = f1
    metrics[f"{prefix}_f1_max{sfx}"] = fm
    metrics[f"{prefix}_conf_at_f1_max{sfx}"] = at


@dataclass
class DetectorScore:
    metrics: dict[str, float | None]
    n_items: int


BALL_TIER_SETS = {"ball": TIERS_HUMAN, "ball_t12": TIERS_CONFIRMED}


def _score_frame(tallies: dict[str, Tally], gt: FrameGT, clip: ClipDetections, frame: int) -> None:
    """Match one labelled frame's detections against its ground truth and add them to ``tallies``."""
    labels, boxes, conf = clip.frame(frame)
    lab = np.array(labels, dtype=object)

    def mask(names: Iterable[str]) -> np.ndarray:
        names = set(names)
        return np.array([x in names for x in lab], dtype=bool) if len(lab) else np.zeros(0, dtype=bool)

    for cls_name, gts, ign in (("player", gt.player, gt.ignore_player), ("referee", gt.referee, gt.ignore_referee)):
        if gts:
            m = mask([cls_name])
            tallies.setdefault(cls_name, Tally()).add_frame(boxes[m], conf[m], gts, ign, clip.size, radius_px=None)
    bm = mask(BALL_DET_LABELS)
    for key, tiers in BALL_TIER_SETS.items():
        ball_gt = [b for t in tiers for b in gt.ball.get(t, ())]
        if not ball_gt and gt.marker != "no_ball":
            continue  # no ball ground truth of this tier on the frame: it says nothing about the ball
        ignore = [b for t, bs in gt.ball.items() if t not in tiers for b in bs]
        for radius in (None, *RADII_PX):
            tallies.setdefault(f"{key}|{_rule_suffix(radius)}", Tally()).add_frame(
                boxes[bm], conf[bm], ball_gt, ignore, clip.size, radius_px=radius)


def score_items(items: Sequence[dict], store: DetectionStore) -> DetectorScore:
    """Score the tracking ``items`` (rows of a dataset version) against the stored detections."""
    tallies: dict[str, Tally] = {}
    eligible = covered = no_ball_frames = 0
    missing_clips: set[str] = set()
    for it in items:
        payload = json.loads(it["payload"]) if isinstance(it["payload"], str) else it["payload"]
        gt = frame_gt(payload)
        if gt.marker == "not_broadcast":
            continue
        if not (gt.player or gt.referee or gt.ball.get(1) or gt.ball.get(2) or gt.marker == "no_ball"):
            continue  # tier-3-only / Ball-Check-only frames carry no ground truth for the detector
        eligible += 1
        clip = store.get(it["clip"])
        if clip is None:
            missing_clips.add(it["clip"])
            continue
        if not clip.has_frame(it["frame_index"]):
            continue
        covered += 1
        no_ball_frames += gt.marker == "no_ball"
        _score_frame(tallies, gt, clip, it["frame_index"])

    metrics: dict[str, float | None] = {}
    for cls_name in ("player", "referee"):
        t = tallies.get(cls_name, Tally())
        _put(metrics, cls_name, t)
        metrics[f"{cls_name}_n_gt"] = float(t.n_gt)
    for key in BALL_TIER_SETS:
        for radius in (None, *RADII_PX):
            sfx = _rule_suffix(radius)
            _put(metrics, key, tallies.get(f"{key}|{sfx}", Tally()), sfx)
        metrics[f"{key}_n_gt"] = float(tallies.get(f"{key}|iou50", Tally()).n_gt)
    metrics["ball_n_no_ball_frames"] = float(no_ball_frames)
    aps = [a for a in (metrics["player_ap50"], metrics["referee_ap50"], metrics["ball_ap_iou50"]) if a is not None]
    metrics["map50"] = float(np.mean(aps)) if aps else None
    metrics["coverage"] = covered / eligible if eligible else None
    metrics["clips_without_detections"] = float(len(missing_clips))
    return DetectorScore(metrics, covered)


__all__ = [
    "BALL_DET_LABELS", "BALL_GT_LABELS", "CONF_FLOOR", "CONF_THRESHOLD", "ClipDetections", "DetectionStore",
    "DetectorScore", "FrameGT", "Tally", "centre_dist_px", "frame_gt", "iou_matrix", "score_items",
]
