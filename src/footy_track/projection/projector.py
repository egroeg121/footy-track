"""2D Projection stage — projects image-space bboxes onto the pitch plane.

See docs/design/projection.md for the full design specification.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from pydantic import BaseModel

from footy_track.constants import BALL_TAG

H_QUALITY_THRESHOLD = 0.3


class PitchPosition(BaseModel):
    track_id: int
    continuous_time: float
    x_pitch: float
    y_pitch: float
    x_pitch_norm: float
    y_pitch_norm: float
    uncertainty_m: float | None
    class_label: str

    model_config = {"frozen": True}


@dataclass
class TrackedDetection:
    """Minimal detection + tracking info consumed by PitchProjector."""

    track_id: int
    continuous_time: float
    label: str
    x: float  # normalised top-left x
    y: float  # normalised top-left y
    w: float  # normalised width
    h: float  # normalised height


class PitchProjector:
    """Projects tracked image-space detections onto a canonical 2D pitch.

    Args:
        pitch_dims: (length_m, width_m) of the pitch. Defaults to 105 × 68.
        h_quality_threshold: H_quality values below this yield NaN positions.
    """

    def __init__(
        self,
        pitch_dims: tuple[float, float] = (105.0, 68.0),
        h_quality_threshold: float = H_QUALITY_THRESHOLD,
    ) -> None:
        self.pitch_length, self.pitch_width = pitch_dims
        self.h_quality_threshold = h_quality_threshold

    # ------------------------------------------------------------------
    # Anchor helpers
    # ------------------------------------------------------------------

    def _anchor(self, det: TrackedDetection) -> tuple[float, float]:
        """Return the image-space anchor point for a detection (normalised)."""
        if det.label == BALL_TAG:
            # centroid
            return det.x + det.w / 2.0, det.y + det.h / 2.0
        # bottom-centre for players and all other classes
        return det.x + det.w / 2.0, det.y + det.h

    # ------------------------------------------------------------------
    # Core projection
    # ------------------------------------------------------------------

    def _apply_homography(
        self, px: float, py: float, H: np.ndarray
    ) -> tuple[float, float]:
        """Apply a 3×3 homography to a single image-space point."""
        src = np.array([px, py, 1.0], dtype=np.float64)
        dst = H @ src
        if abs(dst[2]) < 1e-12:
            return float("nan"), float("nan")
        return float(dst[0] / dst[2]), float(dst[1] / dst[2])

    def _normalise(self, x_pitch: float, y_pitch: float) -> tuple[float, float]:
        """Convert pitch metres to [0, 1] normalised coordinates."""
        # Origin at centre spot; range is [-L/2, L/2] × [-W/2, W/2]
        x_norm = (x_pitch + self.pitch_length / 2.0) / self.pitch_length
        y_norm = (y_pitch + self.pitch_width / 2.0) / self.pitch_width
        return x_norm, y_norm

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def project_frame(
        self,
        detections: list[TrackedDetection],
        H: np.ndarray,
        H_quality: float | None,
    ) -> list[PitchPosition]:
        """Project all detections in a single frame onto the pitch.

        Low-quality calibration frames (H_quality < threshold) produce rows
        with x_pitch / y_pitch = NaN and uncertainty_m = +inf, honouring
        system invariant §4.5 (non-broadcast frames are recorded, not
        silently dropped).
        """
        low_quality = H_quality is None or H_quality < self.h_quality_threshold

        results: list[PitchPosition] = []
        for det in detections:
            if low_quality:
                pos = PitchPosition(
                    track_id=det.track_id,
                    continuous_time=det.continuous_time,
                    x_pitch=float("nan"),
                    y_pitch=float("nan"),
                    x_pitch_norm=float("nan"),
                    y_pitch_norm=float("nan"),
                    uncertainty_m=float("inf"),
                    class_label=det.label,
                )
            else:
                ax, ay = self._anchor(det)
                xp, yp = self._apply_homography(ax, ay, H)
                xn, yn = self._normalise(xp, yp)
                uncertainty = None if H_quality is None else 1.0 - H_quality
                pos = PitchPosition(
                    track_id=det.track_id,
                    continuous_time=det.continuous_time,
                    x_pitch=xp,
                    y_pitch=yp,
                    x_pitch_norm=xn,
                    y_pitch_norm=yn,
                    uncertainty_m=uncertainty,
                    class_label=det.label,
                )
            results.append(pos)
        return results

    def project_match(
        self,
        tracks_parquet: Path,
        geometry_parquet: Path,
        out_parquet: Path,
    ) -> None:
        """Project an entire match from parquet files.

        Reads tracks.parquet (Detection rows) and geometry.parquet
        (per-frame H matrix + H_quality), writes pitch_positions.parquet.
        """
        tracks_df = pd.read_parquet(tracks_parquet)
        geometry_df = pd.read_parquet(geometry_parquet)

        # geometry_df must have columns: frame_index, H_flat (9 floats), H_quality
        merged = tracks_df.merge(geometry_df, on="frame_index", how="left")

        rows: list[dict] = []
        for _, row in merged.iterrows():
            det = TrackedDetection(
                track_id=int(row["track_id"]),
                continuous_time=float(row["continuous_time_s"]),
                label=str(row["label"]),
                x=float(row["bbox_x"]),
                y=float(row["bbox_y"]),
                w=float(row["bbox_w"]),
                h=float(row["bbox_h"]),
            )
            H_flat = row.get("H_flat")
            H: np.ndarray | None = None
            if H_flat is not None:
                H = np.array(H_flat, dtype=np.float64).reshape(3, 3)

            H_quality = row.get("H_quality")

            if H is None:
                positions = self.project_frame([det], np.eye(3), H_quality=None)
            else:
                positions = self.project_frame([det], H, H_quality=H_quality)

            for pos in positions:
                rows.append(pos.model_dump())

        out_df = pd.DataFrame(rows)
        out_parquet.parent.mkdir(parents=True, exist_ok=True)
        out_df.to_parquet(out_parquet, index=False)
