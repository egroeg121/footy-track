"""Lens helpers for the calibration editor: the per-ground default distortion and "snap to lines".

Broadcast lenses are barrel distorted, so long lines (the touchlines, the far edge) curve near the frame
edges and a plain 8-DOF homography cannot follow them. Two things live here:

* :func:`default_distortion` - the division-model ``k1`` of the clip's ground (``camera_rigs/*.json``),
  ``None`` for an unknown ground (nothing is guessed).
* :func:`snap_to_lines` - refine ``H`` and ``k1`` of a current fit against the frame's white-line mask so the
  projected pitch lines, curved by ``k1``, sit on the painted lines. A robust (soft-L1) least squares over
  the 8 homography parameters + ``k1`` starting from the current solution with bounded steps, using the
  finders' own line evidence (``pitch_finders.Field`` / ``Template`` / ``evaluate`` for the F1, ``centre_circle``
  masks). The answer is accepted only if the line F1 improves and the user's clicked points stay within a few
  px; otherwise the old fit comes back with a ``reason``.

Conventions: ``H`` is normalised image -> pitch metres (as in the editor / GT); every image point is an
*observed* (distorted) normalised point unless stated otherwise.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial import cKDTree

from footy_track.calibration import centre_circle as cc
from footy_track.calibration import geometry as geo
from footy_track.calibration import pitch_finders as pf
from footy_track.calibration.pitch_model import PitchModel

#: ``k1`` search range of the snap (division model, see ``geometry``)
K1_BOUNDS = (-0.02, 0.08)
#: a snap moves ``k1`` by at most this from its start
K1_MAX_STEP = 0.04
#: max |parameter| of the 8 homography perturbation terms (fractions of the frame width)
H_MAX_STEP = 0.03
#: clicked points may move at most this many px of the 1280 px working frame (~6 px at 1920)
MAX_CLICK_SHIFT_PX = 4.0
#: the line F1 must rise by at least this to accept a snap
MIN_F1_GAIN = 0.005
#: soft cap (working px) of a line residual: pixels further than this from any line carry no information
RES_CAP_PX = 10.0
#: weight of a clicked-point residual (per working px) against a line residual
POINT_WEIGHT = 0.6
#: ridge pulling the 8 homography terms (x width) and k1 (x 100) back to the start
RIDGE_H = 8.0
RIDGE_K1 = 6.0
TEMPLATE_STEP_M = 0.3
N_MASK_SAMPLES = 2500


def default_distortion(clip: str | None, aspect: float = geo.DEFAULT_ASPECT) -> dict | None:
    """The ground's default lens (``geometry.make_distortion`` dict) for a clip, ``None`` for unknown grounds."""
    from footy_track.calibration.centre_fit import rig_for_clip  # noqa: PLC0415

    rig = rig_for_clip(clip)
    if rig is None or not rig.k1:
        return None
    return geo.make_distortion(rig.k1, aspect or geo.DEFAULT_ASPECT)


# --------------------------------------------------------------------------
# evidence
# --------------------------------------------------------------------------


@dataclass
class Evidence:
    """The frame's thin line pixels in the observed plane (working-size index px)."""

    obs: np.ndarray  # (N, 2)
    size: tuple[int, int]  # working (w, h)
    tol: float
    fld: pf.Field  # distance to the nearest observed line pixel
    tree: cKDTree | None = None
    sample: np.ndarray | None = None
    frame_width: int = 0  # width (px) of the frame the evidence came from (working px scale)
    _und_cache: dict = field(default_factory=dict)

    @classmethod
    def from_mask(cls, mask: np.ndarray) -> Evidence:
        h, w = mask.shape
        obs, _ = cc._prep_points(mask, None)
        ev = cls(obs, (w, h), max(2.5, w / 450), pf.Field(obs, (w, h)))
        if len(obs) > N_MASK_SAMPLES:
            rng = np.random.default_rng(0)
            sample = obs[rng.choice(len(obs), N_MASK_SAMPLES, replace=False)]
        else:
            sample = obs
        ev.sample = sample
        return ev

    @classmethod
    def from_frame(cls, bgr: np.ndarray, with_far: bool = True) -> Evidence:
        mask = cc.line_mask_for(bgr)
        if with_far:
            mask = mask | pf.far_mask_for(bgr).astype(bool)
        ev = cls.from_mask(mask)
        ev.frame_width = bgr.shape[1]
        return ev

    def undistorted_field(self, dist: dict | None) -> pf.Field:
        """Distance field of the mask undistorted by ``dist`` (what the finders score in)."""
        key = repr(dist)
        if key not in self._und_cache:
            w, h = self.size
            if geo.distortion_params(dist) is None:
                und = self.obs
            else:
                und = geo.norm_to_px(geo.undistort_points(geo.px_to_norm(self.obs + 0.5, w, h), dist), w, h) - 0.5
            self._und_cache[key] = pf.Field(und, self.size)
        return self._und_cache[key]


def _template(pitch: PitchModel) -> pf.Template:
    return pf.Template.build(pitch.polylines(), TEMPLATE_STEP_M)


def _h_px(H_img2pitch: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """normalised image (undistorted) -> pitch metres  =>  pitch metres -> undistorted index px."""
    w, h = size
    Hinv = np.linalg.inv(np.asarray(H_img2pitch, dtype=np.float64))
    S = np.array([[w, 0.0, -0.5], [0.0, h, -0.5], [0.0, 0.0, 1.0]])
    return S @ Hinv


def line_f1(ev: Evidence, H: np.ndarray, dist: dict | None, pitch: PitchModel) -> dict:
    """Line F1 of a fit (``pitch_finders.evaluate`` on the mask undistorted by ``dist``)."""
    out = pf.evaluate(_h_px(H, ev.size), _template(pitch), ev.undistorted_field(dist), ev.tol)
    return {"f1": float(out["f1"]), "precision": float(out["precision"]), "matched_m": float(out["m"])}


# --------------------------------------------------------------------------
# the snap
# --------------------------------------------------------------------------


def _perturb(theta: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """3x3 image-plane perturbation (identity at 0) in normalised coords, centred on the frame middle."""
    w, h = size
    a = np.array([[1 + theta[0], theta[1], theta[2]], [theta[3], 1 + theta[4], theta[5]], [theta[6], theta[7], 1.0]])
    # undistorted norm [0,1]^2 (u, v) -> centred, width units: x = u - .5, y = (v - .5) * h / w
    Tn = np.array([[1.0, 0.0, -0.5], [0.0, h / w, -0.5 * h / w], [0.0, 0.0, 1.0]])
    return np.linalg.inv(Tn) @ a @ Tn


@dataclass
class SnapResult:
    ok: bool
    accepted: bool
    reason: str
    H: np.ndarray  # normalised image -> pitch (the answer, or the unchanged start)
    distortion: dict | None
    f1_before: float
    f1_after: float
    k1_before: float
    k1_after: float
    max_click_shift_px: float | None = None  # working px
    points: np.ndarray | None = None  # clicked points moved onto the snapped model (observed norm)
    info: dict = field(default_factory=dict)


def _k1_of(dist: dict | None) -> float:
    dp = geo.distortion_params(dist)
    return 0.0 if dp is None else dp[0]


def snap_to_lines(  # noqa: PLR0911, PLR0915
    ev: Evidence,
    H: np.ndarray,
    dist: dict | None,
    pitch: PitchModel,
    clicks_obs: np.ndarray | None = None,
    clicks_pitch: np.ndarray | None = None,
    mode: str = "full",
    aspect: float = geo.DEFAULT_ASPECT,
    max_click_shift_px: float = MAX_CLICK_SHIFT_PX,
    ref: tuple[np.ndarray, dict | None] | None = None,
) -> SnapResult:
    """Refine ``H`` (and ``k1``) of the current fit against the line evidence.

    ``H`` / ``dist``: the current fit (``H`` normalised image -> pitch, ``dist`` as ``geometry.make_distortion``).
    ``clicks_obs`` / ``clicks_pitch``: the user's point pairs (observed normalised image, pitch metres); they act
    as soft constraints and the result is refused if any of them would move more than ``max_click_shift_px``
    (px of the working frame). ``mode``: ``"full"`` refines H and k1, ``"k1"`` only k1 with H fixed.
    ``ref`` = ``(H, dist)`` of the fit the user actually has when the search starts from a different one (e.g. the
    ground's suggested k1 while the lens is off): the F1 gain is judged against it and it is what comes back on a
    refusal.
    """
    H0 = np.asarray(H, dtype=np.float64)
    Hinv0 = np.linalg.inv(H0)
    k1_0 = _k1_of(dist)
    centre = (dist or {}).get("centre") or (0.5, 0.5)
    asp = float((dist or {}).get("aspect") or aspect)
    w, h = ev.size
    wh = np.array([w, h], dtype=np.float64)
    Href, dref = (H0, dist) if ref is None else (np.asarray(ref[0], dtype=np.float64), ref[1])
    k1_ref = _k1_of(dref)
    before = line_f1(ev, Href, dref, pitch)

    def fail(reason: str, **info) -> SnapResult:
        return SnapResult(True, False, reason, Href, dref, before["f1"], before["f1"], k1_ref, k1_ref, info=info)

    if len(ev.obs) < 200:
        return fail("too few line pixels found in this frame")
    tpl = _template(pitch)
    pts = tpl.pts
    full = mode == "full"
    n_h = 8 if full else 0
    clk_o = None if clicks_obs is None or len(clicks_obs) == 0 else np.asarray(clicks_obs, float).reshape(-1, 2)
    clk_p = None if clk_o is None else np.asarray(clicks_pitch, float).reshape(-1, 2)

    def model(x: np.ndarray, P: np.ndarray):
        """pitch metres (N, 2) -> observed working px (N, 2) and in-front mask, for parameters x."""
        theta = x[:8] if full else np.zeros(8)
        k1 = k1_0 + x[n_h]
        Hi = _perturb(theta, ev.size) @ Hinv0
        v = np.concatenate([P, np.ones((len(P), 1))], 1) @ Hi.T
        front = v[:, 2] > 1e-9
        uv = v[:, :2] / np.where(front, v[:, 2], 1.0)[:, None]
        obs = geo.distort_norm(uv, k1, centre, asp) if k1 != 0.0 else uv
        ok = front & np.isfinite(obs).all(axis=1)
        return obs * wh - 0.5, ok

    def resid(x: np.ndarray) -> np.ndarray:
        xy, ok = model(x, pts)
        d, inb = ev.fld.sample(np.where(ok[:, None], xy, -1e6))
        vis = ok & inb
        r1 = np.where(vis, np.minimum(d, RES_CAP_PX), 0.0)
        # reverse term: a sample of the detected line pixels must lie near some visible template sample
        if vis.sum() >= 8:
            tree = cKDTree(xy[vis])
            d2, _ = tree.query(ev.sample)
            r2 = np.minimum(d2, RES_CAP_PX) * np.sqrt(len(pts) / max(len(ev.sample), 1)) * 0.5
        else:
            r2 = np.full(len(ev.sample), RES_CAP_PX)
        parts = [r1, r2, np.concatenate([RIDGE_H * x[:n_h], [RIDGE_K1 * 100.0 * x[n_h]]])]
        if clk_o is not None:
            cxy, cok = model(x, clk_p)
            e = np.where(cok[:, None], cxy - (clk_o * wh - 0.5), RES_CAP_PX)
            parts.append(POINT_WEIGHT * e.ravel())
        return np.concatenate(parts)

    lo = np.concatenate([np.full(n_h, -H_MAX_STEP), [max(K1_BOUNDS[0] - k1_0, -K1_MAX_STEP)]])
    hi = np.concatenate([np.full(n_h, H_MAX_STEP), [min(K1_BOUNDS[1] - k1_0, K1_MAX_STEP)]])
    x0 = np.zeros(n_h + 1)
    scale = np.concatenate([np.full(n_h, 0.01), [0.01]])
    try:
        sol = least_squares(
            resid, x0, bounds=(lo, hi), loss="soft_l1", f_scale=2.0, x_scale=scale, diff_step=1e-3, max_nfev=40
        )
    except (ValueError, np.linalg.LinAlgError) as exc:
        return fail(f"solver failed: {exc}")
    x = sol.x
    theta = x[:8] if full else np.zeros(8)
    k1 = k1_0 + x[n_h]
    Hinv1 = _perturb(theta, ev.size) @ Hinv0
    try:
        H1 = np.linalg.inv(Hinv1)
    except np.linalg.LinAlgError:
        return fail("degenerate solution")
    dist1 = geo.make_distortion(k1, asp, centre) if abs(k1) > 1e-9 else None
    after = line_f1(ev, H1, dist1, pitch)
    moved = None
    new_pts = None
    if clk_o is not None:
        cxy, cok = model(x, clk_p)
        if not cok.all():
            return fail("snap would push a clicked point out of view", f1_candidate=after["f1"])
        shift = np.linalg.norm(cxy - (clk_o * wh - 0.5), axis=1)
        moved = float(shift.max())
        new_pts = (cxy + 0.5) / wh
    res = SnapResult(
        True, False, "", H1, dist1, before["f1"], after["f1"], k1_ref, k1, moved, new_pts,
        {"nfev": int(sol.nfev), "k1_step": float(x[n_h]), "cost": float(sol.cost)},
    )
    if after["f1"] < before["f1"] + MIN_F1_GAIN:
        res.reason = f"line F1 did not improve ({before['f1']:.3f} -> {after['f1']:.3f}); kept the current fit"
        res.H, res.distortion, res.points, res.k1_after, res.f1_after = Href, dref, None, k1_ref, before["f1"]
        res.info["f1_candidate"] = after["f1"]
        return res
    if moved is not None and moved > max_click_shift_px:
        res.reason = (
            f"snap would move a clicked point by {moved:.1f} px (> {max_click_shift_px:g}); kept the current fit"
        )
        res.H, res.distortion, res.points, res.k1_after, res.f1_after = Href, dref, None, k1_ref, before["f1"]
        res.info["f1_candidate"] = after["f1"]
        return res
    res.accepted = True
    return res


def snap_best(
    ev: Evidence,
    H: np.ndarray,
    dist: dict | None,
    pitch: PitchModel,
    clicks_obs: np.ndarray | None = None,
    clicks_pitch: np.ndarray | None = None,
    mode: str = "full",
    aspect: float = geo.DEFAULT_ASPECT,
    suggested: tuple[np.ndarray, dict] | None = None,
) -> SnapResult:
    """:func:`snap_to_lines` from the current fit and, when given, also from ``suggested = (H, dist)`` (the
    ground's suggested k1 with H refit under it): the better accepted answer wins (ties: the current fit's).
    Both are judged against the current fit, which is what comes back when neither is accepted.
    """
    kw = {"clicks_obs": clicks_obs, "clicks_pitch": clicks_pitch, "mode": mode, "aspect": aspect}
    ref = (np.asarray(H, dtype=np.float64), dist)
    first = snap_to_lines(ev, H, dist, pitch, **kw)
    if suggested is None:
        return first
    second = snap_to_lines(ev, suggested[0], suggested[1], pitch, ref=ref, **kw)
    if second.accepted and (not first.accepted or second.f1_after > first.f1_after + 1e-3):
        return second
    return first


def edge_residual(ev: Evidence, H: np.ndarray, dist: dict | None, pitch: PitchModel, outer: float = 0.3) -> dict:
    """How well the long boundary lines (touchlines, goal lines) sit on the painted lines near the frame edges.

    Template samples of those lines inside the frame and in the outer ``outer`` fraction of the width (either
    side) are projected through ``H`` + ``dist``; the residual is the distance (working px, observed plane) to the
    nearest detected line pixel, counting only samples with a line pixel within ``RES_CAP_PX`` (the rest are
    reported as ``n_missing``). Returns ``{median_px, p90_px, n, n_missing}``.
    """
    w, h = ev.size
    wh = np.array([w, h], dtype=np.float64)
    segs = pitch.segments()
    P = np.concatenate([pf._resample(segs[n], 0.3) for n in ("TOUCHLINE_TOP", "TOUCHLINE_BOTTOM", "GOAL_LINE_L", "GOAL_LINE_R")])
    xy, w_ = geo.pitch_to_image(np.asarray(H, float), P, dist)
    ok = (w_ > 1e-9) & np.isfinite(xy).all(axis=1)
    inframe = ok & (xy[:, 0] > 0.0) & (xy[:, 0] < 1.0) & (xy[:, 1] > 0.0) & (xy[:, 1] < 1.0)
    zone = inframe & (np.abs(xy[:, 0] - 0.5) > 0.5 - outer)
    d, inb = ev.fld.sample(np.where(zone[:, None], xy * wh - 0.5, -1e6))
    sel = zone & inb
    near = sel & (d <= RES_CAP_PX)
    if not near.any():
        return {"median_px": float("nan"), "p90_px": float("nan"), "n": int(sel.sum()), "n_missing": int(sel.sum())}
    return {
        "median_px": float(np.median(d[near])),
        "p90_px": float(np.percentile(d[near], 90)),
        "n": int(sel.sum()),
        "n_missing": int(sel.sum() - near.sum()),
    }
