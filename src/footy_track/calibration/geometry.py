"""Homography geometry for camera -> pitch calibration.

Convention: ``H`` maps **normalised image coords** ``[0,1]^2`` (origin top-left,
as detections are stored) to **pitch metres** (see ``pitch_model``). Its sign is
fixed so that points of the visible ground plane have ``w > 0``; the inverse
then has ``w > 0`` exactly for pitch points in front of the camera, which is
what ``project_polyline_to_image`` uses to avoid lines wrapping around.
"""

from __future__ import annotations

import itertools

import cv2
import numpy as np

from footy_track.calibration.pitch_model import PitchModel

#: Pitch points whose projective ``w`` is below this are treated as behind
#: (or on the horizon of) the camera and dropped.
W_EPS = 1e-6
#: Projected points further than this many image widths/heights from the frame
#: are dropped: near the horizon coordinates explode and add nothing visible.
MAX_NORM_COORD = 20.0

Array = np.ndarray


# --------------------------------------------------------------------------
# Coordinates
# --------------------------------------------------------------------------


def px_to_norm(pts_px: Array, width: float, height: float) -> Array:
    p = np.asarray(pts_px, dtype=np.float64)
    return p / np.array([width, height], dtype=np.float64)


def norm_to_px(pts_norm: Array, width: float, height: float) -> Array:
    p = np.asarray(pts_norm, dtype=np.float64)
    return p * np.array([width, height], dtype=np.float64)


def apply_homography(H: Array, pts: Array) -> tuple[Array, Array]:
    """Apply ``H`` to (N, 2) points; returns ``(xy (N,2), w (N,))``.

    ``xy`` is NaN where ``|w|`` is ~0.
    """
    p = np.atleast_2d(np.asarray(pts, dtype=np.float64))
    ph = np.concatenate([p, np.ones((len(p), 1))], axis=1) @ np.asarray(H).T
    w = ph[:, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        xy = ph[:, :2] / w[:, None]
    xy[np.abs(w) < 1e-12] = np.nan
    return xy, w


# --------------------------------------------------------------------------
# Fitting
# --------------------------------------------------------------------------


def orient_homography(H: Array, img_norm: Array) -> Array:
    """Scale ``H`` to unit norm and fix its sign so ``img_norm`` has ``w > 0``.

    ``H`` and ``-H`` are the same projective map, but the behind-camera test in
    ``project_polyline_to_image`` needs the sign convention described in the
    module docstring. Call this on any externally produced ``H``.
    """
    H = np.asarray(H, dtype=np.float64)
    H = H / np.linalg.norm(H)
    _, w = apply_homography(H, img_norm)
    return -H if np.median(w) < 0 else H


#: Triple is "collinear" when ``|cross| / spread^2`` (twice the triangle area
#: relative to the squared diameter of the point set) is below these. Pitch
#: keypoints are exact so that tolerance is tight; clicked image points are
#: noisy so theirs is looser (~1 px on a 1000 px frame still counts).
COLLINEAR_TOL_PITCH = 1e-3
COLLINEAR_TOL_IMAGE = 2e-3
#: Up to this many points the subset search is brute force (C(12,4) = 495).
MAX_SUBSET_SEARCH = 12


def _collinear_triples(pts: Array, tol: float) -> set[tuple[int, int, int]]:
    spread = max(float(np.ptp(pts, axis=0).max()), 1e-12)
    out = set()
    n = len(pts)
    for i, j, k in itertools.combinations(range(n), 3):
        a, b, c = pts[i], pts[j], pts[k]
        cross = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
        # normalise by the set's extent, not the triple's, so a tight cluster
        # of near-coincident points is flagged too
        if abs(cross) / spread**2 < tol:
            out.add((i, j, k))
    return out


def _collinear_groups(triples: set[tuple[int, int, int]]) -> list[list[int]]:
    """Merge collinear triples into maximal groups of point indices."""
    groups: list[set[int]] = []
    for t in sorted(triples):
        ts = set(t)
        for g in groups:
            if len(g & ts) >= 2:
                g |= ts
                break
        else:
            groups.append(ts)
    merged = True
    while merged:  # chains of overlapping groups
        merged = False
        for a, b in itertools.combinations(range(len(groups)), 2):
            if len(groups[a] & groups[b]) >= 2:
                groups[a] |= groups.pop(b)
                merged = True
                break
    return sorted(sorted(g) for g in groups)


def _fmt_points(idx: list[int]) -> str:
    return ", ".join(str(i + 1) for i in idx)  # 1-based, as the page numbers them


def analyse_configuration(img_norm: Array, pitch_m: Array) -> dict:
    """Is the point set able to determine ``H``?

    Returns ``{"n_general_subsets": int | None, "groups": [("pitch"|"image",
    [indices])]}``. ``n_general_subsets`` counts the 4-subsets with no three
    collinear in either space (``None`` above ``MAX_SUBSET_SEARCH`` points,
    where only a rank check is done). Indices are 0-based.
    """
    src = np.asarray(img_norm, dtype=np.float64).reshape(-1, 2)
    dst = np.asarray(pitch_m, dtype=np.float64).reshape(-1, 2)
    n = len(src)
    if n > MAX_SUBSET_SEARCH:
        groups = []
        for space, pts in (("image", src), ("pitch", dst)):
            sv = np.linalg.svd(pts - pts.mean(0), compute_uv=False)
            if sv[0] <= 0 or sv[1] / sv[0] < 1e-3:
                groups.append((space, list(range(n))))
        return {"n_general_subsets": None, "groups": groups}
    tp = _collinear_triples(dst, COLLINEAR_TOL_PITCH)
    ti = _collinear_triples(src, COLLINEAR_TOL_IMAGE)
    bad = tp | ti
    good = sum(
        1
        for q in itertools.combinations(range(n), 4)
        if not any(t in bad for t in itertools.combinations(q, 3))
    )
    groups = [("pitch", g) for g in _collinear_groups(tp)] + [
        ("image", g) for g in _collinear_groups(ti)
    ]
    return {"n_general_subsets": good, "groups": groups}


def _degenerate_message(cfg: dict, n: int) -> str:
    merged: dict[tuple[int, ...], list[str]] = {}
    for space, g in cfg["groups"]:
        merged.setdefault(tuple(g), []).append(space)
    parts = []
    for g, spaces in merged.items():
        where = spaces[0]  # pitch first; an image-space repeat adds nothing
        who = "all points" if len(g) == n else f"points {_fmt_points(list(g))}"
        parts.append(f"{who} lie on one {where} line")
    detail = "; ".join(parts) if parts else "the points are clustered on a line"
    return (
        f"degenerate configuration: {detail}. Need 4 points with no 3 on "
        "one line - add a point off that line"
    )


def fit_homography(
    img_norm: Array,
    pitch_m: Array,
    method: str = "dlt",
    ransac_thresh_m: float = 1.0,
) -> tuple[Array, Array]:
    """Fit ``H`` (normalised image -> pitch metres) from >= 4 correspondences.

    ``method="dlt"`` is a least-squares (normalised) DLT over all points;
    ``"ransac"`` rejects outliers beyond ``ransac_thresh_m`` metres.

    Returns ``(H, inlier_mask)`` with ``H[2,2]`` scaled to unit norm and its
    sign set so the correspondences' image points have ``w > 0``.
    Raises ``ValueError`` for < 4 points or a configuration that cannot determine
    ``H`` (no 4 points in general position; the message names the collinear
    points by 1-based index).
    """
    src = np.asarray(img_norm, dtype=np.float64).reshape(-1, 2)
    dst = np.asarray(pitch_m, dtype=np.float64).reshape(-1, 2)
    if len(src) != len(dst):
        raise ValueError("image and pitch point counts differ")
    if len(src) < 4:
        raise ValueError("need at least 4 correspondences")
    if method not in ("dlt", "ransac"):
        raise ValueError(f"unknown method {method!r}")
    cfg = analyse_configuration(src, dst)
    if cfg["n_general_subsets"] == 0 or (
        cfg["n_general_subsets"] is None and cfg["groups"]
    ):
        raise ValueError(_degenerate_message(cfg, len(src)))

    if method == "ransac":
        H, mask = cv2.findHomography(
            src, dst, cv2.RANSAC, ransacReprojThreshold=ransac_thresh_m
        )
    else:
        H, mask = cv2.findHomography(src, dst, 0)
    if H is None or not np.all(np.isfinite(H)):
        raise ValueError("homography estimation failed (degenerate points)")
    if mask is None:
        inliers = np.ones(len(src), dtype=bool)
    else:
        inliers = mask.ravel().astype(bool)
    if inliers.sum() < 4:
        raise ValueError("fewer than 4 inliers")

    return orient_homography(H, src), inliers


def conditioning(img_norm: Array, pitch_m: Array, H: Array) -> float:
    """Condition number of ``H`` in Hartley-normalised coordinates.

    Scale-free, so it measures how close the fit is to singular rather than
    the unit choice (metres vs normalised image). ~1e1..1e2 is healthy.
    """
    def _t(p: Array) -> Array:
        c = p.mean(0)
        s = np.sqrt(2) / max(float(np.mean(np.linalg.norm(p - c, axis=1))), 1e-12)
        return np.array([[s, 0, -s * c[0]], [0, s, -s * c[1]], [0, 0, 1.0]])

    Ts = _t(np.asarray(img_norm, dtype=np.float64).reshape(-1, 2))
    Td = _t(np.asarray(pitch_m, dtype=np.float64).reshape(-1, 2))
    Hn = Td @ np.asarray(H) @ np.linalg.inv(Ts)
    return float(np.linalg.cond(Hn))


def reprojection_errors(
    H: Array,
    img_norm: Array,
    pitch_m: Array,
    image_size: tuple[float, float] | None = None,
) -> dict[str, Array | None]:
    """Per-point errors of ``H`` against its correspondences.

    * ``metres``: image -> pitch, distance to the labelled pitch point.
    * ``px``: pitch -> image through ``H^-1``, distance in pixels to the
      clicked point (``None`` without ``image_size``). This is the error the
      operator sees on the frame.
    Points that do not project (w <= 0 under the inverse) give NaN px error.
    """
    src = np.asarray(img_norm, dtype=np.float64).reshape(-1, 2)
    dst = np.asarray(pitch_m, dtype=np.float64).reshape(-1, 2)
    fwd, _ = apply_homography(H, src)
    metres = np.linalg.norm(fwd - dst, axis=1)
    px = None
    if image_size is not None:
        back, w = apply_homography(np.linalg.inv(H), dst)
        err = np.linalg.norm(
            (back - src) * np.array(image_size, dtype=np.float64), axis=1
        )
        err[w <= W_EPS] = np.nan
        px = err
    return {"metres": metres, "px": px}


# --------------------------------------------------------------------------
# Drawing the pitch model in the image
# --------------------------------------------------------------------------


def _densify(poly: Array, step_m: float) -> Array:
    out = [poly[:1]]
    for a, b in zip(poly[:-1], poly[1:], strict=True):
        n = max(1, int(np.ceil(np.linalg.norm(b - a) / step_m)))
        t = np.linspace(0, 1, n + 1)[1:, None]
        out.append(a + (b - a) * t)
    return np.concatenate(out)


def project_polyline_to_image(
    H: Array, poly_pitch: Array, step_m: float = 0.5
) -> list[Array]:
    """Project a pitch polyline into normalised image coords via ``H^-1``.

    The polyline is densified (a straight line stays straight under a
    homography but its visible part may end mid-segment), points behind the
    camera (``w <= W_EPS``) or absurdly far from the frame are dropped, and the
    surviving points are returned as a list of runs, one per connected visible
    piece, so a line never wraps across the horizon.
    """
    pts = _densify(np.asarray(poly_pitch, dtype=np.float64), step_m)
    xy, w = apply_homography(np.linalg.inv(H), pts)
    ok = (
        (w > W_EPS)
        & np.all(np.isfinite(xy), axis=1)
        & np.all(np.abs(xy - 0.5) < MAX_NORM_COORD, axis=1)
    )
    runs: list[Array] = []
    start = None
    for i, good in enumerate(ok):
        if good and start is None:
            start = i
        elif not good and start is not None:
            runs.append(xy[start:i])
            start = None
    if start is not None:
        runs.append(xy[start:])
    return [r for r in runs if len(r) >= 2]


def project_pitch_model(
    H: Array, pitch: PitchModel, step_m: float = 0.5
) -> dict[str, list[Array]]:
    """All pitch lines/curves as ``name -> runs`` of normalised image points."""
    out = {}
    for name, poly in pitch.polylines().items():
        runs = project_polyline_to_image(H, poly, step_m)
        if runs:
            out[name] = runs
    return out


# --------------------------------------------------------------------------
# Bird's-eye warp
# --------------------------------------------------------------------------


def pitch_raster_transform(
    pitch: PitchModel, px_per_m: float, margin_m: float
) -> Array:
    """3x3 map pitch metres -> top-down raster pixels (``+y`` up on screen)."""
    s = px_per_m
    return np.array(
        [
            [s, 0.0, (pitch.hl + margin_m) * s],
            [0.0, -s, (pitch.hw + margin_m) * s],
            [0.0, 0.0, 1.0],
        ]
    )


def pitch_raster_size(
    pitch: PitchModel, px_per_m: float, margin_m: float
) -> tuple[int, int]:
    """``(width, height)`` in pixels of the top-down raster."""
    return (
        int(round((pitch.length + 2 * margin_m) * px_per_m)),
        int(round((pitch.width + 2 * margin_m) * px_per_m)),
    )


def warp_to_pitch(
    frame: Array,
    H: Array,
    pitch: PitchModel,
    px_per_m: float = 10.0,
    margin_m: float = 3.0,
) -> Array:
    """Warp a BGR frame into a top-down pitch raster (black outside the view)."""
    h_img, w_img = frame.shape[:2]
    to_norm = np.diag([1.0 / w_img, 1.0 / h_img, 1.0])
    M = pitch_raster_transform(pitch, px_per_m, margin_m) @ np.asarray(H) @ to_norm
    return cv2.warpPerspective(
        frame, M, pitch_raster_size(pitch, px_per_m, margin_m), flags=cv2.INTER_LINEAR
    )
