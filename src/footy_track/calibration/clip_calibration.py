"""Per-frame calibration of a whole clip from a few keyframe calibrations.

A broadcast main camera only pans / tilts / zooms, so a calibration (normalised image -> pitch
metres, :mod:`footy_track.calibration.geometry`) can be carried from a labelled keyframe to the other
frames of the same shot with the image-to-image motion of :mod:`footy_track.calibration.temporal`, and
re-anchored on the white lines when the chained motion drifts. This module is the policy around those
pieces::

    calibrate_clip(video, {frame: (H, distortion)}, boxes=None, policy=Policy()) -> ClipCalibration

* **Propagation**: from every keyframe, forward and backward inside its shot, up to the neighbouring
  keyframe (or the shot end). A cut or a failed pair stops a sweep: nothing is extrapolated across it.
* **Re-anchoring (trigger policy)**: every ``check_every`` frames the line cost of the propagated ``H`` (mean
  truncated distance of the model lines to the line mask) is compared with the one at the current anchor. When
  it exceeds ``refine_ratio x anchor + refine_px`` (or the anchor is ``max_anchor_gap`` frames old) the
  homography is refined on the lines (:func:`temporal.refine_frame`, always started from the propagated ``H``,
  never from a static one). The refinement is **gated**: it must lower the line cost, end below
  ``max_line_cost`` and not move the model by more than ``max_refine_shift_px`` (at the 1280 working
  width), otherwise the propagated ``H`` is kept. An accepted refinement becomes the new anchor and the frames
  since the previous check are re-derived from it.
* **Selection**: a frame covered by two sweeps (left and right keyframe) takes the nearest keyframe's result,
  unless the other one has a clearly lower line cost (``compete_margin``).
* **Confidence**: ``ok`` is False (and the frame is not projectable) when the frame's shot has no keyframe, the
  line cost is above ``max_line_cost`` or, with no line to judge by (cost NaN: close-ups), the nearest keyframe is
  further than ``max_blind_gap`` frames. Frames of a shot without a keyframe carry no ``H`` at all.

Line costs are measured only on the check frames; frames in between carry the cost of the preceding check
(in sweep direction). The line masks are the expensive part (~0.5-1 s/frame at 1280 wide) and do not depend on
the keyframes, so they are held in a :class:`MaskStore` that can be built once, saved and reused.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from footy_track.calibration import geometry as geo
from footy_track.calibration import temporal as tp
from footy_track.calibration.pitch_model import PitchModel
from footy_track.calibration.refine import dt_residual, model_samples

Array = np.ndarray

KEYFRAME, PROPAGATED, REFINED, NONE = "keyframe", "propagated", "refined", "none"
_CENTRE = np.array([[0.5, 0.5]])


@dataclass(frozen=True)
class Policy:
    check_every: int = 5  # frames between line-cost checks
    refine_ratio: float = 1.1  # refine when cost > ratio * anchor cost + refine_px ... (1.1 / 0.5: tuned, see results)
    refine_px: float = 0.5
    max_anchor_gap: int = 100  # ... or when the anchor is this many frames old
    refine_cooldown: int = 10  # frames before retrying after a rejected refinement
    max_refine_shift_px: float = 60.0  # gate: model movement allowed by a refinement (px at work_width)
    max_line_cost: float = 12.0  # ok gate: line cost (px at work_width); the GT floor is ~5
    cost_floor: float = 5.0  # confidence: cost at/below this scores 1
    max_blind_gap: int = 60  # ok gate when no line is visible to judge the fit
    gap_scale: float = 600.0  # confidence decays as 1 / (1 + keyframe_gap / gap_scale)
    compete_margin: float = 1.0  # a farther keyframe wins only with a line cost this much lower
    work_width: int = 1280
    refine: bool = True  # False: pure propagation (still measures line cost)


# --------------------------------------------------------------------------
# line masks
# --------------------------------------------------------------------------


class MaskStore:
    """Line masks (:class:`temporal.LineMasks`, packed bits) of selected frames of one clip."""

    def __init__(self, work_width: int = 1280):
        self.work_width = work_width
        self._d: dict[int, tuple[tuple[int, int], Array, Array, Array]] = {}

    def __contains__(self, f: int) -> bool:
        return int(f) in self._d

    def __len__(self) -> int:
        return len(self._d)

    @property
    def frames(self) -> list[int]:
        return sorted(self._d)

    def add(self, f: int, m: tp.LineMasks) -> None:
        self._d[int(f)] = (m.lines.shape, np.packbits(m.grass), np.packbits(m.lines), np.packbits(m.far))

    def get(self, f: int) -> tp.LineMasks | None:
        e = self._d.get(int(f))
        if e is None:
            return None
        (h, w), g, ln, fr = e

        def un(a):
            return np.unpackbits(a)[: h * w].reshape(h, w).astype(bool)

        return tp.LineMasks(un(g), un(ln), un(fr))

    def ensure(self, video: Path | str, frames, progress=None) -> None:
        """Compute the masks of ``frames`` not yet held (one sequential decode of the clip)."""
        want = sorted({int(f) for f in frames} - set(self._d))
        if not want:
            return
        wset = set(want)
        for idx, small, _ in tp.iter_video_small(video, self.work_width, want[0], want[-1] + 1):
            if idx in wset:
                self.add(idx, tp.line_masks(small, self.work_width))
                if progress:
                    progress(idx)

    def save(self, path: Path | str) -> None:
        fr = self.frames
        np.savez_compressed(path, frames=np.array(fr), shape=np.array([self._d[f][0] for f in fr]),
                            grass=np.array([self._d[f][1] for f in fr]), lines=np.array([self._d[f][2] for f in fr]),
                            far=np.array([self._d[f][3] for f in fr]), work_width=self.work_width)

    @classmethod
    def load(cls, path: Path | str) -> MaskStore:
        z = np.load(path, allow_pickle=False)
        s = cls(int(z["work_width"]))
        for i, f in enumerate(z["frames"]):
            s._d[int(f)] = (tuple(int(x) for x in z["shape"][i]), z["grass"][i], z["lines"][i], z["far"][i])
        return s


def line_cost(H: Array, m: tp.LineMasks, samples: Array, k1: float = 0.0, trunc: float = 20.0) -> float:
    """Mean truncated distance (px at the mask width) of the model lines in view to the line mask; NaN if
    fewer than 30 model samples are in view on grass (close-ups: nothing to judge by)."""
    h, w = m.lines.shape
    G = np.linalg.inv(np.asarray(H) @ np.diag([1.0 / w, 1.0 / h, 1.0]))
    G = G / np.linalg.norm(G)
    c, n = dt_residual(G, samples, m.lines, m.grass, trunc, k1)
    return float(c) if n >= 30 and np.isfinite(c) else float("nan")


# --------------------------------------------------------------------------
# result
# --------------------------------------------------------------------------


@dataclass
class ClipCalibration:
    """Per-frame calibration of a clip: one ``table`` row per frame (consecutive frames from ``start``).

    Columns: ``frame, source (keyframe|propagated|refined|none), anchor_frame, key_frame, gap (frames to the
    anchor), line_cost, confidence, shot_id, ok, k1, H`` (9 floats, row-major, normalised image -> pitch m;
    NaN where there is none).
    """

    start: int
    table: pd.DataFrame
    aspect: float = geo.DEFAULT_ASPECT
    stats: dict = field(default_factory=dict)

    def row(self, frame: int):
        i = int(frame) - self.start
        return self.table.iloc[i] if 0 <= i < len(self.table) else None

    def H(self, frame: int) -> Array | None:
        """The calibration of ``frame`` if it is ``ok``, else None."""
        r = self.row(frame)
        return np.asarray(r["H"], float).reshape(3, 3) if r is not None and r["ok"] else None

    def H_any(self, frame: int) -> Array | None:
        """The homography even when the frame is not ``ok`` (None only when there is none)."""
        r = self.row(frame)
        h = None if r is None else np.asarray(r["H"], float)
        return h.reshape(3, 3) if h is not None and np.all(np.isfinite(h)) else None

    def distortion(self, frame: int) -> dict | None:
        r = self.row(frame)
        return None if r is None else geo.make_distortion(float(r["k1"]), self.aspect)

    def records(self) -> list[dict]:
        """One dict per frame: ``frame, H (3x3 or None), distortion, source, anchor_frame, gap, line_cost,
        confidence, shot_id, ok`` (+ ``key_frame``)."""
        out = []
        for r in self.table.to_dict("records"):
            h = np.asarray(r.pop("H"), float)
            r["H"] = h.reshape(3, 3) if np.all(np.isfinite(h)) else None
            r["distortion"] = geo.make_distortion(r.pop("k1"), self.aspect)
            out.append(r)
        return out

    def save(self, path: Path | str) -> None:
        t = self.table.copy()
        t["H"] = t["H"].map(lambda h: [float(x) for x in h])
        t.to_parquet(path, index=False)

    @classmethod
    def load(cls, path: Path | str, aspect: float = geo.DEFAULT_ASPECT) -> ClipCalibration:
        t = pd.read_parquet(path)
        t["H"] = t["H"].map(lambda h: np.asarray(h, float))
        return cls(int(t["frame"].iloc[0]), t, aspect)


# --------------------------------------------------------------------------
# sweeps
# --------------------------------------------------------------------------


@dataclass
class _Cand:
    H: Array
    source: str
    anchor: int
    key: int
    cost: float


def _sweep(motion: tp.ClipMotion, key: int, H_key: Array, k1: float, direction: int, stop: int | None,
           keys: set[int], masks: MaskStore, pitch: PitchModel, samples: Array, pol: Policy, log: dict) -> dict[int, _Cand]:
    """Carry the calibration of ``key`` away from it (``direction`` +1 / -1) until ``stop`` (exclusive), a cut or a lost pair."""
    out: dict[int, _Cand] = {}
    anchor, H_a = key, H_key
    m_key = masks.get(key)
    c_a = line_cost(H_a, m_key, samples, k1) if m_key is not None else float("nan")
    last_cost = c_a
    pending: list[int] = []  # unchecked frames since the last check (re-derived if the anchor moves)
    last_try = -(10**9)
    t = key
    while True:
        t += direction
        if t == stop or not (motion.start <= t < motion.start + motion.n):
            break
        Hp = tp.propagate(H_a, motion, anchor, t)
        if Hp is None:
            break
        checked = ((t - motion.start) % pol.check_every == 0 or t in keys) and t in masks
        if not checked:
            out[t] = _Cand(Hp, PROPAGATED, anchor, key, last_cost)
            pending.append(t)
            continue
        m = masks.get(t)
        c = line_cost(Hp, m, samples, k1)
        if not np.isfinite(c_a) and np.isfinite(c):
            c_a = c  # keyframe had no masks / no lines in view: the first measurement is the baseline
        cand = _Cand(Hp, PROPAGATED, anchor, key, c)
        if pol.refine and np.isfinite(c) and abs(t - last_try) >= pol.refine_cooldown:
            drift = np.isfinite(c_a) and c > pol.refine_ratio * c_a + pol.refine_px
            if drift or abs(t - anchor) >= pol.max_anchor_gap:
                last_try = t
                t0 = time.perf_counter()
                r = tp.refine_frame(Hp, m, pitch, k1=k1)
                log["refine_s"] = log.get("refine_s", 0.0) + time.perf_counter() - t0
                log["n_refine"] = log.get("n_refine", 0) + 1
                c_new = line_cost(r.H, m, samples, k1) if r.applied else float("nan")
                shift = r.max_shift_px if r.max_shift_px is not None else float("inf")
                if r.applied and np.isfinite(c_new) and c_new < c and c_new <= pol.max_line_cost and shift <= pol.max_refine_shift_px:
                    H_new = geo.orient_homography(r.H, _CENTRE)
                    log["n_accepted"] = log.get("n_accepted", 0) + 1
                    for u in pending:  # re-derive the unchecked frames since the last check from the new anchor
                        Hu = tp.propagate(H_new, motion, t, u)
                        if Hu is not None:
                            out[u] = _Cand(Hu, PROPAGATED, t, key, c_new)
                    anchor, H_a, c_a = t, H_new, c_new
                    cand = _Cand(H_new, REFINED, t, key, c_new)
                else:
                    log["n_rejected"] = log.get("n_rejected", 0) + 1
        out[t] = cand
        last_cost = cand.cost
        pending = []
    return out


def _confidence(cost: float, gap: int, pol: Policy) -> float:
    cg = 1.0 / (1.0 + gap / pol.gap_scale)
    if not np.isfinite(cost):
        return 0.5 * cg
    cc = float(np.clip(1.0 - (cost - pol.cost_floor) / max(pol.max_line_cost - pol.cost_floor, 1e-6), 0.0, 1.0))
    return cc * cg


def calibrate_clip(  # noqa: PLR0915
    video: Path | str, keyframes: dict, boxes: dict[int, Array] | None = None, policy: Policy | None = None,
    *, motion: tp.ClipMotion | None = None, masks: MaskStore | None = None, pitch: PitchModel | None = None,
    progress=None,
) -> ClipCalibration:
    """Per-frame calibration of ``video`` from ``keyframes = {frame: (H, distortion)}`` (``H``: normalised image
    -> pitch m; ``distortion``: dict / None; a bare ``H`` is accepted). ``boxes`` (``frame -> (N, 4)``
    person / ball boxes, see :func:`temporal.load_person_boxes`) mask the movers out of the camera-motion
    estimate. ``motion`` / ``masks`` take a precomputed :class:`temporal.ClipMotion` / :class:`MaskStore` (the
    missing masks are computed). Returns one row per frame of the motion range."""
    pol = policy or Policy()
    pitch = pitch or PitchModel()
    samples = model_samples(pitch)
    if not keyframes:
        raise ValueError("need at least one keyframe")
    if motion is None:
        motion = tp.clip_motion(video, boxes=boxes, mask_mode="all", progress=progress)  # whole frame minus overlay / boxes
    lo, hi = motion.start, motion.start + motion.n
    aspect = motion.size[0] / motion.size[1]
    kf: dict[int, tuple[Array, float]] = {}
    for f, v in keyframes.items():
        if not lo <= int(f) < hi:
            raise ValueError(f"keyframe {f} outside the clip's frames [{lo}, {hi})")
        H, dist = v if isinstance(v, tuple | list) else (v, None)
        dp = geo.distortion_params(dist)
        kf[int(f)] = (geo.orient_homography(np.asarray(H, float), _CENTRE), dp[0] if dp else 0.0)
    keys = sorted(kf)
    shot = motion.shot
    key_shots = {int(shot[k - lo]) for k in keys}
    masks = masks if masks is not None else MaskStore(pol.work_width)
    t_masks = time.perf_counter()
    need = {f for f in range(lo, hi) if (f - lo) % pol.check_every == 0 and int(shot[f - lo]) in key_shots} | set(keys)
    masks.ensure(video, need, progress)
    log: dict = {"mask_s": time.perf_counter() - t_masks, "n_refine": 0, "n_accepted": 0, "n_rejected": 0, "refine_s": 0.0}

    cands: dict[int, list[_Cand]] = {}
    kset = set(keys)
    t_sweep = time.perf_counter()
    for i, k in enumerate(keys):
        H_k, k1 = kf[k]
        right = next((j for j in keys[i + 1:] if shot[j - lo] == shot[k - lo]), None)
        left = next((j for j in reversed(keys[:i]) if shot[j - lo] == shot[k - lo]), None)
        for d, stop in ((1, right), (-1, left)):
            for t, c in _sweep(motion, k, H_k, k1, d, stop, kset, masks, pitch, samples, pol, log).items():
                cands.setdefault(t, []).append(c)
    log["sweep_s"] = time.perf_counter() - t_sweep

    rows = []
    for f in range(lo, hi):
        row = {"frame": f, "source": NONE, "anchor_frame": -1, "key_frame": -1, "gap": -1, "line_cost": float("nan"),
               "confidence": 0.0, "shot_id": int(shot[f - lo]), "ok": False, "k1": 0.0, "H": [float("nan")] * 9}
        if f in kf:
            H, k1 = kf[f]
            m = masks.get(f)
            c = line_cost(H, m, samples, k1) if m is not None else float("nan")
            row.update(source=KEYFRAME, anchor_frame=f, key_frame=f, gap=0, line_cost=c, confidence=1.0, ok=True, k1=k1,
                       H=[float(x) for x in H.ravel()])
        elif f in cands:
            cs = sorted(cands[f], key=lambda c: abs(f - c.key))
            c = cs[0]
            if len(cs) > 1 and np.isfinite(cs[1].cost) and (not np.isfinite(c.cost) or cs[1].cost < c.cost - pol.compete_margin):
                c = cs[1]
            gap_key = abs(f - c.key)
            ok = c.cost <= pol.max_line_cost if np.isfinite(c.cost) else gap_key <= pol.max_blind_gap
            conf = _confidence(c.cost, gap_key, pol)
            row.update(source=c.source, anchor_frame=c.anchor, key_frame=c.key, gap=abs(f - c.anchor), line_cost=c.cost,
                       confidence=conf if ok else min(conf, 0.2), ok=bool(ok), k1=kf[c.key][1],
                       H=[float(x) for x in c.H.ravel()])
        rows.append(row)
    table = pd.DataFrame(rows)
    stats = {"n_keyframes": len(keys), "n_frames": hi - lo, "key_shots": sorted(key_shots), "cuts": motion.cuts,
             "ok_frac": float(table["ok"].mean()), "motion_method": motion.method, **log}
    return ClipCalibration(lo, table, aspect, stats)
