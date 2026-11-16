"""Global camera-motion estimation between two broadcast frames.

Single entry point :func:`estimate_pair` so another implementation (the shared
calibration/temporal module) can be swapped in. Convention: ``H_px`` maps pixel
coordinates of frame ``a`` (the reference, frame t) to pixel coordinates of frame
``b`` (the neighbour), so ``cv2.warpPerspective(b, H_px, size, flags=INTER_LINEAR |
WARP_INVERSE_MAP)`` renders ``b`` as seen from ``a`` (see :func:`warp_to_ref`).
``H_norm`` is the same map in normalised (x/W, y/H) coordinates, resolution free.

Masks are ``uint8``/bool full-resolution arrays where non-zero marks pixels to
*ignore* (players, ball, referees: things that move independently of the camera).
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

METHODS = ("orb", "akaze", "ecc", "orb+ecc", "lk")
MIN_INLIERS = 25
MIN_INLIER_RATIO = 0.25
#: mean |a - warp(b)| (grey levels, 480 px) above which the pair is rejected: ordinary
#: pairs score <~5, cut / hopeless pairs 30-50 (multiframe_cut_scan).
MAX_RESIDUAL = 12.0


@dataclass
class PairMotion:
    H_px: np.ndarray  # 3x3, a -> b, full-resolution pixels
    H_norm: np.ndarray  # 3x3, a -> b, normalised coordinates
    ok: bool
    inliers: int  # RANSAC inliers (feature methods); 0 for pure ECC
    residual: float  # mean |a - warp(b)| grey levels, valid unmasked overlap, at `scale`
    method: str
    reason: str = ""  # first failed check, "" when ok


def _gray(img: np.ndarray) -> np.ndarray:
    return img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def _small(img: np.ndarray, scale: int) -> tuple[np.ndarray, float]:
    h, w = img.shape[:2]
    s = scale / w
    return cv2.resize(img, (scale, round(h * s)), interpolation=cv2.INTER_AREA), s


def _valid_mask(mask: np.ndarray | None, shape: tuple[int, int], dilate_px: int = 0) -> np.ndarray:
    """255 where usable, 0 where masked, at ``shape`` (h, w)."""
    valid = np.full(shape, 255, np.uint8)
    if mask is not None:
        m = cv2.resize((np.asarray(mask) != 0).astype(np.uint8), (shape[1], shape[0]),
                       interpolation=cv2.INTER_NEAREST)
        if dilate_px:
            m = cv2.dilate(m, np.ones((dilate_px, dilate_px), np.uint8))
        valid[m != 0] = 0
    return valid


def _feature_h(ga, gb, va, vb, method: str):
    if method == "orb":
        det = cv2.ORB_create(nfeatures=2500, fastThreshold=8, edgeThreshold=15)
        norm = cv2.NORM_HAMMING
    else:
        det = cv2.AKAZE_create(threshold=1e-3)
        norm = cv2.NORM_HAMMING
    ka, da = det.detectAndCompute(ga, va)
    kb, db = det.detectAndCompute(gb, vb)
    if da is None or db is None or len(ka) < 8 or len(kb) < 8:
        return None, 0, "too few keypoints"
    pairs = cv2.BFMatcher(norm).knnMatch(da, db, k=2)
    good = [p[0] for p in pairs if len(p) == 2 and p[0].distance < 0.8 * p[1].distance]
    if len(good) < 8:
        return None, 0, "too few matches"
    src = np.float32([ka[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst = np.float32([kb[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    H, inl = cv2.findHomography(src, dst, cv2.RANSAC, 2.0, maxIters=3000, confidence=0.995)
    if H is None:
        return None, 0, "ransac failed"
    n = int(inl.sum())
    if n < MIN_INLIERS or n < MIN_INLIER_RATIO * len(good):
        return H, n, f"inliers {n}/{len(good)}"
    return H, n, ""


def _ecc_h(ga, gb, va, vb, init: np.ndarray | None):
    # Fill masked pixels of `a` with its mean so they carry no gradient; ECC's own
    # mask applies to the moving image (b).
    ga = ga.copy()
    ga[va == 0] = int(ga[va != 0].mean()) if (va != 0).any() else 128
    H = np.eye(3, dtype=np.float32) if init is None else init.astype(np.float32).copy()
    crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 60, 1e-4)
    try:
        _, H = cv2.findTransformECC(ga, gb, H, cv2.MOTION_HOMOGRAPHY, crit, vb, 5)
    except cv2.error as e:  # did not converge / singular
        return None, "ecc: " + str(e).splitlines()[-1][:60]
    return H, ""


def _sane(H: np.ndarray, w: int, h: int) -> str:
    """Reject degenerate homographies. Returns reason or ''."""
    if not np.isfinite(H).all():
        return "non-finite H"
    corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
    q = cv2.perspectiveTransform(corners, H).reshape(4, 2)
    if not np.isfinite(q).all():
        return "non-finite corners"
    if np.abs(q - corners.reshape(4, 2)).max() > 0.4 * w:
        return "displacement > 0.4 W"
    area = 0.5 * abs(np.dot(q[:, 0], np.roll(q[:, 1], 1)) - np.dot(q[:, 1], np.roll(q[:, 0], 1)))
    if not 0.6 < area / (w * h) < 1.6:
        return "area ratio"
    cross = [np.cross(q[(i + 1) % 4] - q[i], q[(i + 2) % 4] - q[(i + 1) % 4]) for i in range(4)]
    if not (all(c > 0 for c in cross) or all(c < 0 for c in cross)):
        return "non-convex"
    return ""


def warp_to_ref(b: np.ndarray, H_px: np.ndarray, size_wh: tuple[int, int] | None = None) -> np.ndarray:
    """Render ``b`` in ``a``'s coordinates (borders filled by replication)."""
    if size_wh is None:
        size_wh = (b.shape[1], b.shape[0])
    return cv2.warpPerspective(
        b, H_px, size_wh, flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
        borderMode=cv2.BORDER_REPLICATE,
    )


def _residual(sa, sb, H_s, va, vb) -> float:
    h, w = sa.shape
    wb = warp_to_ref(sb, H_s, (w, h))
    inside = cv2.warpPerspective(np.full_like(sb, 255), H_s, (w, h),
                                 flags=cv2.INTER_NEAREST | cv2.WARP_INVERSE_MAP)
    wvb = cv2.warpPerspective(vb, H_s, (w, h), flags=cv2.INTER_NEAREST | cv2.WARP_INVERSE_MAP)
    ok = (inside > 0) & (va > 0) & (wvb > 0)
    if ok.sum() < 0.1 * w * h:
        return float("inf")
    return float(np.abs(sa.astype(np.float32) - wb.astype(np.float32))[ok].mean())


def estimate_pair(
    a: np.ndarray,
    b: np.ndarray,
    mask_a: np.ndarray | None = None,
    mask_b: np.ndarray | None = None,
    scale: int = 480,
    method: str = "ecc",
) -> PairMotion:
    """Homography ``a -> b`` (see module docstring). ``scale`` = working width in px."""
    assert method in METHODS, method
    if method == "lk":
        return estimate_pair_lk(a, b, mask_a, mask_b, scale)
    h0, w0 = a.shape[:2]
    sa, s = _small(_gray(a), scale)
    sb, _ = _small(_gray(b), scale)
    h, w = sa.shape
    # ~12 px at 1080p grows masked boxes so blurred player edges are excluded too.
    va = _valid_mask(mask_a, (h, w), 3)
    vb = _valid_mask(mask_b, (h, w), 3)
    S = np.diag([s, s, 1.0])
    inliers, reason, H_s = 0, "", None
    if method in ("orb", "akaze", "orb+ecc"):
        H_s, inliers, reason = _feature_h(sa, sb, va, vb, "akaze" if method == "akaze" else "orb")
        if method == "orb+ecc" and H_s is not None and not reason:
            H_e, r2 = _ecc_h(sa, sb, va, vb, H_s)
            if H_e is not None:
                H_s = H_e
            else:
                reason = r2
    else:
        H_s, reason = _ecc_h(sa, sb, va, vb, None)
    if H_s is None:
        eye = np.eye(3)
        return PairMotion(eye, eye.copy(), False, inliers, float("inf"), method, reason)
    H_s = np.asarray(H_s, np.float64)
    H_s /= H_s[2, 2]
    H_px = np.linalg.inv(S) @ H_s @ S
    S_n = np.diag([1.0 / w0, 1.0 / h0, 1.0])
    H_norm = S_n @ H_px @ np.linalg.inv(S_n)
    res = _residual(sa, sb, H_s, va, vb)
    if not reason:
        reason = _sane(H_px, w0, h0)
    if not reason and not np.isfinite(res):
        reason = "no overlap"
    if not reason and res > MAX_RESIDUAL:
        reason = f"residual {res:.1f}"
    return PairMotion(H_px, H_norm, reason == "", inliers, res, method, reason)


def estimate_pair_lk(a, b, mask_a=None, mask_b=None, scale: int = 480, boxes_a=None, boxes_b=None) -> PairMotion:
    """Adapter to the shared camera-motion module (``footy_track.calibration.temporal``,
    vendored verbatim from branch exp/calibration-temporal commit 7988bde): corner features +
    pyramidal Lucas-Kanade + RANSAC, whole frame minus overlay and person boxes.

    ``boxes_*`` are normalised *centre* (cx, cy, w, h) person boxes (this package's
    convention); they are converted to the module's top-left xywh. The module's masks use
    True = usable, ours non-zero = ignore, so an explicit ``mask_*`` is inverted.
    """
    from footy_track.calibration import temporal as T

    def tl(bx):
        if not bx:
            return None
        return np.array([[x - w / 2, y - h / 2, w, h] for x, y, w, h in bx], np.float64)

    def usable(m):
        return None if m is None else (np.asarray(m) == 0)

    pm = T.estimate_pair(a, b, usable(mask_a), usable(mask_b), scale=scale, method="lk",
                         boxes_a=tl(boxes_a), boxes_b=tl(boxes_b), mask_mode="all")
    return PairMotion(pm.H_px, pm.H_norm, bool(pm.ok), int(pm.inliers), float(pm.residual),
                      "lk", pm.reason or "")


# -- helpers used by the eval / build scripts ---------------------------------------

def boxes_to_mask(shape_hw: tuple[int, int], boxes_norm: list[tuple[float, float, float, float]],
                  grow: float = 0.25) -> np.ndarray:
    """uint8 mask (255 = ignore) from normalised (cx, cy, w, h) boxes, grown by ``grow``."""
    h, w = shape_hw
    m = np.zeros((h, w), np.uint8)
    for cx, cy, bw, bh in boxes_norm:
        bw, bh = bw * (1 + grow), bh * (1 + grow)
        x0, x1 = int((cx - bw / 2) * w), int((cx + bw / 2) * w) + 1
        y0, y1 = int((cy - bh / 2) * h), int((cy + bh / 2) * h) + 1
        m[max(y0, 0):max(y1, 0), max(x0, 0):max(x1, 0)] = 255
    return m


def pitch_mask(img_bgr: np.ndarray) -> np.ndarray:
    """Boolean mask of grass-like pixels (HSV green, eroded) for evaluation."""
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    m = cv2.inRange(hsv, (30, 40, 40), (90, 255, 255))
    return cv2.erode(m, np.ones((5, 5), np.uint8)) > 0


def hist_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Bhattacharyya distance of 8x4x4 HSV histograms on 160 px thumbnails (0 = same)."""
    def hist(x):
        t, _ = _small(x if x.ndim == 3 else cv2.cvtColor(x, cv2.COLOR_GRAY2BGR), 160)
        hh = cv2.calcHist([cv2.cvtColor(t, cv2.COLOR_BGR2HSV)], [0, 1, 2], None,
                          [8, 4, 4], [0, 180, 0, 256, 0, 256])
        return cv2.normalize(hh, hh, 1, 0, cv2.NORM_L1).astype(np.float32)
    return float(cv2.compareHist(hist(a), hist(b), cv2.HISTCMP_BHATTACHARYYA))
