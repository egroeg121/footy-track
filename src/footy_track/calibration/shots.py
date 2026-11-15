"""Per-shot camera-type classifier: which shots of a broadcast clip are the main (wide) camera?

``calibrate_clip`` carries a calibration along a shot with image-to-image camera motion and white-line
re-anchoring; that only makes sense on the main broadcast camera (a pan / tilt / zoom camera on a fixed
mount). Close-ups, replays, reverse angles and crowd shots have no pitch lines to anchor on (or a different
camera position), so a match-level run has to decide *per shot* whether to calibrate it at all, and which frame
of a main shot to use as keyframe. Both questions are answered here with cheap, hand-crafted, transparent
features (no deep model, no GPU)::

    classify_shots(video, motion=None, boxes=None, *, sample_every=10, model=None,
                   calibration=None, features=None, progress=None) -> list[dict]

returns one record per shot (consecutive, covering the motion range)::

    shot_id, start, end (inclusive), n_frames, type ("main" | "closeup" | "other"), confidence (of that type),
    p_main, features (dict of shot-level medians), n_samples, keyframe_candidate (frame or None),
    keyframe_line_cost (float or NaN)

Shot boundaries are ``ClipMotion.shot`` (colour-histogram cuts, see :mod:`temporal`); ``motion`` is computed
when not given (gap-1 pairs only, 320 px: ~30 ms/frame). ``boxes`` (``frame -> (N, 4)`` normalised top-left
xywh, :func:`temporal.load_person_boxes`) add the person-size features.

Features (frame level, one sample every ``sample_every`` frames, medians per shot)::

    grass_frac     fraction of the frame that is pitch (:func:`temporal.line_masks` grass mask)
    grass_low      fraction of the lowest third that is pitch (a main camera looks down at the pitch)
    line_frac      white-line pixels / pitch pixels (x100)
    line_len       Hough segment length in image widths (long straight white lines in view)
    n_dirs         distinct line directions (20 deg bins with >= 0.1 image width of segments)
    blob_h         height (image heights) of the tallest compact non-green blob inside the pitch region: a player
                   when the camera is wide (~0.1), a face / shirt in a close-up (0.3 and up); needs no detector
    blob_frac      area of those blobs / pitch area
    n_blob         number of those blobs (players on a wide shot: 10-25; a close-up: a few)
    blob_med_h     their median height
    grass_top      fraction of the top third that is pitch (a main camera sees the stands there)
    person_h_med   median person-box height in image heights (NaN without boxes; ball-sized boxes ignored)
    person_h_max   tallest person box
    n_person       number of person boxes
    log_len        log2 of the shot length in frames (shot-level)
    motion         median camera-centre displacement per frame (normalised image units x100, shot-level)
    zoom           median |log scale change| per frame x100 (shot-level)
    motion_fail    fraction of frame pairs whose motion estimate failed (shot-level)

Decision: a logistic model on the ten image features of each sampled *frame* (``ShotModel``, ``shot_model.json``, fitted by
``scripts/shots_eval.py --fit`` on the human Roboflow "broadcast frame" Yes / No labels, standardised and clipped to +-3 sigma) gives a
per-frame ``p_main``; the shot's ``p_main`` is the median over its samples and the shot is ``main`` if it is >= 0.5. Not-main shots are
``closeup`` when little pitch is visible (``grass_frac`` < 0.5) or a blob is tall (``blob_h`` >= 0.25), else ``other`` (a pitch view that is
not the main camera: tactical / reverse / corner camera). Shot length, camera motion and the box-based person sizes are reported in
``features`` but do not enter the decision: no label set exercises them (every labelled calibration clip is a single main shot). Without
``shot_model.json`` (or ``model=None``) the threshold rule :func:`rule_p_main` is used; it is weaker (see ``shots_eval``).

Keyframe candidate: among the main shot's samples the frame with the most fit support (``line_len`` x
``n_dirs`` / 4, lines that are long and in several directions pin a homography down); with a
:class:`clip_calibration.ClipCalibration` the best three by support are re-ranked by the calibrated line cost.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from footy_track.calibration import temporal as tp

Array = np.ndarray

MAIN, CLOSEUP, OTHER = "main", "closeup", "other"
#: features that need nothing but the frame: the ones the classifier uses
IMAGE_FEATURES = ("grass_frac", "grass_low", "grass_top", "line_frac", "line_len", "n_dirs", "blob_h", "blob_frac", "n_blob", "blob_med_h")
BOX_FEATURES = ("person_h_med", "person_h_max", "n_person")
FRAME_FEATURES = (*IMAGE_FEATURES, *BOX_FEATURES)
SHOT_FEATURES = (*FRAME_FEATURES, "log_len", "motion", "zoom", "motion_fail")
#: boxes shorter than this (fraction of the frame height) are the ball / noise, not people
MIN_PERSON_H = 0.03
HOUGH_BIN_DEG = 20
DEFAULT_WORK_WIDTH = 640
MODEL_PATH = Path(__file__).with_name("shot_model.json")


# --------------------------------------------------------------------------
# frame features
# --------------------------------------------------------------------------


def _line_geometry(lines: Array) -> tuple[float, int]:
    """(total Hough segment length in image widths, number of distinct 20-degree directions)."""
    h, w = lines.shape
    segs = cv2.HoughLinesP(lines.astype(np.uint8), 1, np.pi / 180, threshold=25, minLineLength=int(0.06 * w), maxLineGap=int(0.015 * w))
    if segs is None:
        return 0.0, 0
    s = segs.reshape(-1, 4).astype(float)
    dx, dy = s[:, 2] - s[:, 0], s[:, 3] - s[:, 1]
    length = np.hypot(dx, dy)
    ang = np.degrees(np.arctan2(dy, dx)) % 180.0
    bins = (ang // HOUGH_BIN_DEG).astype(int)
    per_bin = np.bincount(bins, weights=length, minlength=180 // HOUGH_BIN_DEG) / w
    return float(length.sum() / w), int((per_bin >= 0.1).sum())


def _blobs(bgr: Array, grass: Array) -> tuple[float, float, float, float]:
    """(height of the tallest compact non-green blob in the pitch region as a fraction of the image height, their
    area / pitch area, their number, their median height). Compact = fills at least a third of its bounding box
    (white lines and arcs do not)."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    h, s_, v = cv2.split(hsv)
    green = ((h >= 30) & (h <= 95) & (s_ >= 40) & (v >= 35)).astype(np.uint8)
    green = cv2.morphologyEx(green, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    inside = (grass & (green == 0)).astype(np.uint8)
    inside = cv2.morphologyEx(inside, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, _, st, _ = cv2.connectedComponentsWithStats(inside, connectivity=8)
    H = bgr.shape[0]
    tall, area, hs = 0.0, 0.0, []
    for i in range(1, n):
        a = st[i, cv2.CC_STAT_AREA]
        bw, bh = st[i, cv2.CC_STAT_WIDTH], st[i, cv2.CC_STAT_HEIGHT]
        if a < 0.0004 * inside.size or a < 0.33 * bw * bh:
            continue
        tall = max(tall, bh / H)
        area += a
        hs.append(bh / H)
    return float(tall), float(area / max(grass.sum(), 1)), float(len(hs)), float(np.median(hs)) if hs else 0.0


def person_features(boxes: Array | None) -> tuple[float, float, float]:
    """(median height, max height, count) of the person boxes; NaN heights / 0 when there are none or no boxes
    (``boxes is None``: NaN count too)."""
    if boxes is None:
        return float("nan"), float("nan"), float("nan")
    b = np.asarray(boxes, float).reshape(-1, 4)
    h = b[:, 3][b[:, 3] >= MIN_PERSON_H]
    if len(h) == 0:
        return float("nan"), float("nan"), 0.0
    return float(np.median(h)), float(h.max()), float(len(h))


def frame_features(bgr_small: Array, boxes: Array | None = None, work_width: int = DEFAULT_WORK_WIDTH) -> dict[str, float]:
    """Cheap features of one frame (BGR, ~``work_width`` wide); see the module docstring."""
    m = tp.line_masks(bgr_small, work_width)
    h = m.grass.shape[0]
    n_grass = float(m.grass.sum())
    ll, nd = _line_geometry(m.lines)
    pm, px, pn = person_features(boxes)
    blob_h, blob_frac, n_blob, blob_med = _blobs(bgr_small, m.grass)
    return {
        "grass_frac": float(m.grass.mean()),
        "grass_low": float(m.grass[2 * h // 3:].mean()),
        "line_frac": 100.0 * float(m.lines.sum()) / max(n_grass, 1.0),
        "line_len": ll,
        "n_dirs": float(nd),
        "blob_h": blob_h,
        "blob_frac": blob_frac,
        "n_blob": n_blob,
        "blob_med_h": blob_med,
        "grass_top": float(m.grass[: h // 3].mean()),
        "person_h_med": pm,
        "person_h_max": px,
        "n_person": pn,
    }


def _boxes_at(boxes: dict[int, Array] | None, f: int, tol: int = 2) -> Array | None:
    """Boxes of frame ``f`` (or the nearest frame within ``tol`` that has any); None when ``boxes`` is None."""
    if boxes is None:
        return None
    for d in range(tol + 1):
        for g in (f - d, f + d):
            if g in boxes:
                return boxes[g]
    return np.zeros((0, 4))


def sample_frames(motion: tp.ClipMotion, sample_every: int) -> list[int]:
    """Absolute frames to sample: every ``sample_every``-th, plus the middle of every shot that has none."""
    lo, hi = motion.start, motion.start + motion.n
    s = set(range(lo, hi, sample_every))
    shot = motion.shot
    for sid in np.unique(shot):
        idx = np.flatnonzero(shot == sid)
        a, b = lo + int(idx[0]), lo + int(idx[-1])
        if not any(a <= f <= b for f in s):
            s.add((a + b) // 2)
    return sorted(s)


def extract_frame_features(video: Path | str, frames: list[int], boxes: dict[int, Array] | None = None,
                           work_width: int = DEFAULT_WORK_WIDTH, progress=None) -> dict[int, dict[str, float]]:
    """:func:`frame_features` of the given frames (one sequential decode of the clip)."""
    want = sorted({int(f) for f in frames})
    if not want:
        return {}
    wset = set(want)
    out: dict[int, dict[str, float]] = {}
    for idx, small, _ in tp.iter_video_small(video, work_width, want[0], want[-1] + 1):
        if idx in wset:
            out[idx] = frame_features(small, _boxes_at(boxes, idx), work_width)
            if progress:
                progress(idx)
    return out


# --------------------------------------------------------------------------
# shot features
# --------------------------------------------------------------------------


def motion_stats(motion: tp.ClipMotion) -> dict[str, Array]:
    """Per-frame camera motion of the t-1 -> t pair: centre displacement and |log zoom| (both x100, normalised
    image units), and ``fail`` (the pair could not be estimated; NaN displacement)."""
    Hn = motion.Hn[:, 0]
    ok = motion.ok[:, 0]
    c = Hn @ np.array([0.5, 0.5, 1.0])
    with np.errstate(all="ignore"):
        disp = 100.0 * np.hypot(c[:, 0] / c[:, 2] - 0.5, c[:, 1] / c[:, 2] - 0.5)
        det = np.abs(np.linalg.det(Hn[:, :2, :2]))
        zoom = 100.0 * np.abs(0.5 * np.log(np.where(det > 0, det, np.nan)))
    disp[~ok] = np.nan
    zoom[~ok] = np.nan
    return {"disp": disp, "zoom": zoom, "fail": ~ok}


def _nanmedian(a) -> float:
    a = np.asarray(a, float)
    a = a[np.isfinite(a)]
    return float(np.median(a)) if len(a) else float("nan")


def shot_features(samples: list[dict[str, float]], n_frames: int, disp: Array, zoom: Array, fail: Array) -> dict[str, float]:
    """Shot-level feature dict (:data:`SHOT_FEATURES`) from the frame features of its samples and its motion rows
    (the shot's first frame has no pair to its predecessor: pass rows ``[1:]`` of a shot that starts at a cut)."""
    f = {k: _nanmedian([s[k] for s in samples]) for k in FRAME_FEATURES}
    if samples and all(s["n_person"] == 0.0 for s in samples):
        f["n_person"] = 0.0
    f["person_h_max"] = float(np.nanmax([s["person_h_max"] for s in samples])) if any(np.isfinite(s["person_h_max"]) for s in samples) else float("nan")
    f["log_len"] = math.log2(max(n_frames, 1))
    f["motion"] = _nanmedian(disp)
    f["zoom"] = _nanmedian(zoom)
    f["motion_fail"] = float(np.mean(fail)) if len(fail) else 0.0
    return f


# --------------------------------------------------------------------------
# decision
# --------------------------------------------------------------------------


def rule_p_main(f: dict[str, float]) -> float:
    """Transparent fallback score in [0, 1] of a frame: a main camera shows lots of pitch (also in the lowest
    third), small people and some white line; the share of passed tests (4 of 4 -> 1)."""
    tests = [
        f["grass_frac"] >= 0.45,
        f["grass_low"] >= 0.8,
        f["line_len"] >= 0.15 or f["line_frac"] >= 0.8,
        f.get("blob_h", 0.0) <= 0.2,
    ]
    return float(np.mean(tests))


#: standardised features are clipped to +-this, so one off-distribution frame (a graphic, a crowd) cannot swing the score
Z_CLIP = 3.0


@dataclass
class ShotModel:
    """Logistic regression on the standardised image features of a *frame*; the shot's ``p_main`` is the median of its
    samples. NaN is imputed with the training mean."""

    names: tuple[str, ...]
    mean: list[float]
    std: list[float]
    coef: list[float]
    bias: float
    threshold: float = 0.5
    meta: dict = field(default_factory=dict)

    def p_main(self, f: dict[str, float]) -> float:
        x = np.array([f.get(n, float("nan")) for n in self.names], float)
        z = np.where(np.isfinite(x), np.clip((x - self.mean) / np.maximum(self.std, 1e-9), -Z_CLIP, Z_CLIP), 0.0)
        return float(1.0 / (1.0 + np.exp(-(float(z @ np.array(self.coef)) + self.bias))))

    def to_json(self) -> dict:
        return {"names": list(self.names), "mean": self.mean, "std": self.std, "coef": self.coef, "bias": self.bias,
                "threshold": self.threshold, "meta": self.meta}

    def save(self, path: Path | str = MODEL_PATH) -> None:
        Path(path).write_text(json.dumps(self.to_json(), indent=1))

    @classmethod
    def load(cls, path: Path | str = MODEL_PATH) -> ShotModel:
        d = json.loads(Path(path).read_text())
        d["names"] = tuple(d["names"])
        return cls(**d)


def default_model() -> ShotModel | None:
    """The fitted model shipped next to this module (``shot_model.json``), None if absent."""
    return ShotModel.load() if MODEL_PATH.exists() else None


def frame_p_main(f: dict[str, float], model: ShotModel | None = None) -> float:
    return model.p_main(f) if model is not None else rule_p_main(f)


#: not main, and at most this much pitch / at least this tall a blob -> a close-up, else "other"
CLOSEUP_GRASS = 0.5
CLOSEUP_BLOB_H = 0.25


def decide(f: dict[str, float], p_main: float, threshold: float = 0.5) -> tuple[str, float]:
    """``(type, confidence of that type)`` from the shot's median features ``f`` and its ``p_main``."""
    if p_main >= threshold:
        return MAIN, p_main
    close = f["grass_frac"] < CLOSEUP_GRASS or f.get("blob_h", 0.0) >= CLOSEUP_BLOB_H
    return (CLOSEUP if close else OTHER), 1.0 - p_main


# --------------------------------------------------------------------------
# keyframe candidate
# --------------------------------------------------------------------------


def fit_support(f: dict[str, float]) -> float:
    """How well a frame pins a homography down: long white lines in several directions."""
    return float(f["line_len"] * min(f["n_dirs"], 4.0) / 4.0)


def pick_keyframe(frames: dict[int, dict[str, float]], calibration=None, top_k: int = 3) -> tuple[int | None, float]:
    """``(frame, line cost)`` of the best keyframe among the sampled ``frames`` of one shot. Line cost is NaN
    unless a calibration with a finite line cost covers the pick. Frames without any line support are not
    candidates (None)."""
    if not frames:
        return None, float("nan")
    mid = np.median(list(frames))
    ranked = sorted(frames, key=lambda t: (-fit_support(frames[t]), abs(t - mid)))
    ranked = [t for t in ranked if fit_support(frames[t]) > 0.0][:top_k]
    if not ranked:
        return None, float("nan")
    if calibration is not None:
        tab = calibration.table.set_index("frame")
        costs = {}
        for t in ranked:
            if t in tab.index and bool(tab.at[t, "ok"]) and np.isfinite(tab.at[t, "line_cost"]):
                costs[t] = float(tab.at[t, "line_cost"])
        if costs:
            t = min(costs, key=costs.get)
            return int(t), costs[t]
    return int(ranked[0]), float("nan")


# --------------------------------------------------------------------------
# entry point
# --------------------------------------------------------------------------


def classify_shots(video: Path | str, motion: tp.ClipMotion | None = None, boxes: dict[int, Array] | None = None, *,
                   sample_every: int = 10, model: ShotModel | None | str = "default", calibration=None,
                   features: dict[int, dict[str, float]] | None = None, work_width: int = DEFAULT_WORK_WIDTH,
                   progress=None) -> list[dict]:
    """One record per shot of ``video``; see the module docstring.

    ``model``: a :class:`ShotModel`, ``"default"`` (the shipped ``shot_model.json`` if present, else the threshold
    rule) or None (threshold rule). ``features``: precomputed ``{frame: frame_features}`` (cached between runs; the
    missing samples are computed). ``calibration``: a :class:`ClipCalibration` to rank keyframe candidates by line cost.
    """
    if model == "default":
        model = default_model()
    if motion is None:
        motion = tp.clip_motion(video, boxes=boxes, max_gap=1, scale=320, mask_mode="all", progress=progress)
    feats = dict(features or {})
    want = [f for f in sample_frames(motion, sample_every) if f not in feats]
    feats.update(extract_frame_features(video, want, boxes, work_width, progress))
    ms = motion_stats(motion)
    lo = motion.start
    shot = motion.shot
    out = []
    for sid in np.unique(shot):
        idx = np.flatnonzero(shot == sid)
        a, b = lo + int(idx[0]), lo + int(idx[-1])
        frames = {f: feats[f] for f in feats if a <= f <= b}
        rows = idx[1:] if (idx[0] > 0 and len(idx) > 1) else idx  # row 0 of a shot pairs across the cut
        sf = shot_features(list(frames.values()), len(idx), ms["disp"][rows], ms["zoom"][rows], ms["fail"][rows])
        if frames:
            p = float(np.median([frame_p_main(v, model) for v in frames.values()]))
            typ, conf = decide(sf, p, model.threshold if model is not None else 0.5)
        else:
            typ, conf, p = OTHER, 0.0, float("nan")
        kf, kcost = pick_keyframe(frames, calibration) if typ == MAIN else (None, float("nan"))
        out.append({"shot_id": int(sid), "start": a, "end": b, "n_frames": len(idx), "type": typ, "confidence": float(conf),
                    "p_main": float(p), "features": sf, "n_samples": len(frames), "keyframe_candidate": kf,
                    "keyframe_line_cost": kcost})
    return out


def shot_of(shots: list[dict], frame: int) -> dict | None:
    """The shot record containing ``frame`` (None outside the clip)."""
    for s in shots:
        if s["start"] <= frame <= s["end"]:
            return s
    return None
