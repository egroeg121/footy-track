"""Calibration stage — estimate the image→pitch homography per frame.

See ``docs/design/calibration.md`` for the full design specification.

This is the **manual-keypoint** calibration method (method #1 of issue
ft-wn9): given >=4 labelled correspondences between image points and named
pitch landmarks, solve for the homography ``H`` that maps image coordinates
onto pitch metres, and score the fit with a reprojection-error-based
``H_quality`` in ``[0, 1]``.

The output ``geometry.parquet`` schema (``frame_index``, ``H_flat``,
``H_quality``) is exactly what :class:`~footy_track.projection.projector.PitchProjector`
consumes, so calibration unblocks end-to-end 2D projection.

Learned keypoint detection (SoccerNet-style) is a documented follow-up;
this module exposes the same ``H`` / ``H_quality`` contract, so a learned
front-end can replace the manual labels without touching the projector.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from footy_track.projection.pitch_model import PitchModel

# Minimum correspondences for a homography (cv2.findHomography needs >= 4).
MIN_CORRESPONDENCES = 4

# Reprojection error (in metres) at which H_quality decays to ~0.37 (1/e).
# A perfect fit -> quality 1.0; ~5 m mean error -> ~0.37. Players are ~0.5 m
# wide, so multi-metre reprojection error is genuinely unusable.
QUALITY_ERROR_SCALE_M = 5.0

# RANSAC inlier threshold in *image* space. Image points are expected in
# normalised [0, 1] coordinates, so a few pixels of slack is ~0.005.
RANSAC_REPROJ_THRESHOLD = 0.01


@dataclass
class KeypointLabel:
    """A single labelled correspondence for one frame.

    Args:
        frame_index: frame this label belongs to.
        landmark: name of the pitch landmark (see :class:`PitchModel`).
        image_x: normalised image x in ``[0, 1]``.
        image_y: normalised image y in ``[0, 1]``.
    """

    frame_index: int
    landmark: str
    image_x: float
    image_y: float


@dataclass
class CalibrationResult:
    """Outcome of calibrating a single frame.

    ``H`` maps **image** points (normalised) to **pitch** metres. It is
    ``None`` when calibration could not be solved (too few correspondences
    or a degenerate configuration), in which case ``H_quality`` is ``0.0``.
    """

    frame_index: int
    H: np.ndarray | None
    H_quality: float
    n_correspondences: int
    n_inliers: int
    mean_reproj_error_m: float


class Calibrator:
    """Estimate per-frame image→pitch homographies from labelled keypoints.

    Args:
        pitch_dims: ``(length_m, width_m)`` of the pitch. Defaults to 105×68.
        ransac_threshold: RANSAC inlier threshold in normalised image units.
        quality_error_scale_m: reprojection error (m) at which quality decays
            to ``1/e``; smaller values make ``H_quality`` stricter.
    """

    def __init__(
        self,
        pitch_dims: tuple[float, float] = (105.0, 68.0),
        ransac_threshold: float = RANSAC_REPROJ_THRESHOLD,
        quality_error_scale_m: float = QUALITY_ERROR_SCALE_M,
    ) -> None:
        self.pitch = PitchModel(length_m=pitch_dims[0], width_m=pitch_dims[1])
        self.ransac_threshold = ransac_threshold
        self.quality_error_scale_m = quality_error_scale_m

    # ------------------------------------------------------------------
    # Quality scoring
    # ------------------------------------------------------------------

    def _quality_from_error(self, mean_error_m: float) -> float:
        """Map a mean reprojection error (metres) to a quality in ``[0, 1]``.

        Uses an exponential decay so that a perfect fit scores 1.0 and the
        score falls smoothly as error grows. This is the coarse propagation
        referenced in ``docs/design/projection.md`` §5.
        """
        if not np.isfinite(mean_error_m):
            return 0.0
        return float(np.exp(-mean_error_m / self.quality_error_scale_m))

    def _mean_reproj_error_m(
        self,
        H: np.ndarray,
        image_pts: np.ndarray,
        pitch_pts: np.ndarray,
        mask: np.ndarray | None,
    ) -> float:
        """Mean Euclidean reprojection error in metres over inlier points."""
        if mask is not None:
            inliers = mask.ravel().astype(bool)
            if inliers.any():
                image_pts = image_pts[inliers]
                pitch_pts = pitch_pts[inliers]
        projected = cv2.perspectiveTransform(image_pts.reshape(-1, 1, 2), H).reshape(
            -1, 2
        )
        errors = np.linalg.norm(projected - pitch_pts, axis=1)
        return float(errors.mean())

    # ------------------------------------------------------------------
    # Single-frame calibration
    # ------------------------------------------------------------------

    def calibrate_frame(
        self, frame_index: int, labels: list[KeypointLabel]
    ) -> CalibrationResult:
        """Estimate ``H`` for one frame from its labelled keypoints.

        Returns a :class:`CalibrationResult`; an unsolvable frame yields
        ``H=None`` and ``H_quality=0.0`` rather than raising, so a whole
        match can be calibrated without one bad frame aborting the run.
        """
        n = len(labels)
        if n < MIN_CORRESPONDENCES:
            return CalibrationResult(
                frame_index=frame_index,
                H=None,
                H_quality=0.0,
                n_correspondences=n,
                n_inliers=0,
                mean_reproj_error_m=float("inf"),
            )

        image_pts = np.array(
            [[lbl.image_x, lbl.image_y] for lbl in labels], dtype=np.float64
        )
        pitch_pts = np.array(
            [self.pitch.landmark(lbl.landmark) for lbl in labels],
            dtype=np.float64,
        )

        H, mask = cv2.findHomography(
            image_pts,
            pitch_pts,
            method=cv2.RANSAC,
            ransacReprojThreshold=self.ransac_threshold,
        )

        if H is None:
            return CalibrationResult(
                frame_index=frame_index,
                H=None,
                H_quality=0.0,
                n_correspondences=n,
                n_inliers=0,
                mean_reproj_error_m=float("inf"),
            )

        n_inliers = int(mask.sum()) if mask is not None else n
        mean_error = self._mean_reproj_error_m(H, image_pts, pitch_pts, mask)
        quality = self._quality_from_error(mean_error)

        return CalibrationResult(
            frame_index=frame_index,
            H=H,
            H_quality=quality,
            n_correspondences=n,
            n_inliers=n_inliers,
            mean_reproj_error_m=mean_error,
        )

    # ------------------------------------------------------------------
    # Whole-match calibration
    # ------------------------------------------------------------------

    def calibrate_match(self, labels_parquet: Path, out_parquet: Path) -> None:
        """Calibrate every labelled frame and write ``geometry.parquet``.

        ``labels_parquet`` holds one row per labelled keypoint with columns
        ``frame_index``, ``landmark``, ``image_x``, ``image_y``. The output
        has one row per frame with columns ``frame_index``, ``H_flat`` (9
        floats, row-major) and ``H_quality`` — exactly the schema
        :class:`~footy_track.projection.projector.PitchProjector` consumes.

        Frames that cannot be solved are still emitted, with an identity
        ``H_flat`` and ``H_quality=0.0``, so the projector records them as
        low-confidence (NaN pitch coords) rather than dropping them — see
        ``docs/design/projection.md`` §5.
        """
        labels_df = pd.read_parquet(labels_parquet)

        results = self.calibrate_frames_from_df(labels_df)

        rows: list[dict] = []
        for res in results:
            H = res.H if res.H is not None else np.eye(3, dtype=np.float64)
            rows.append(
                {
                    "frame_index": res.frame_index,
                    "H_flat": [float(v) for v in H.flatten()],
                    "H_quality": res.H_quality,
                }
            )

        out_df = pd.DataFrame(rows)
        out_parquet.parent.mkdir(parents=True, exist_ok=True)
        out_df.to_parquet(out_parquet, index=False)

    def calibrate_frames_from_df(
        self, labels_df: pd.DataFrame
    ) -> list[CalibrationResult]:
        """Calibrate each frame present in a labels DataFrame.

        Groups labels by ``frame_index`` and returns one
        :class:`CalibrationResult` per frame, in ascending frame order.
        """
        results: list[CalibrationResult] = []
        for frame_index, group in labels_df.groupby("frame_index", sort=True):
            labels = [
                KeypointLabel(
                    frame_index=int(frame_index),
                    landmark=str(row["landmark"]),
                    image_x=float(row["image_x"]),
                    image_y=float(row["image_y"]),
                )
                for _, row in group.iterrows()
            ]
            results.append(self.calibrate_frame(int(frame_index), labels))
        return results
