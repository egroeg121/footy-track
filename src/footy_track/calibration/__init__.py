"""Camera -> pitch calibration (see docs/design/calibration.md).

M0: canonical pitch model, homography geometry, manual calibration GT.
See ``README.md`` in this package for conventions.
"""

from footy_track.calibration.geometry import (
    apply_homography,
    fit_homography,
    norm_to_px,
    orient_homography,
    project_polyline_to_image,
    px_to_norm,
    reprojection_errors,
    warp_to_pitch,
)
from footy_track.calibration.pitch_model import PitchModel

__all__ = [
    "PitchModel",
    "apply_homography",
    "fit_homography",
    "norm_to_px",
    "orient_homography",
    "project_polyline_to_image",
    "px_to_norm",
    "reprojection_errors",
    "warp_to_pitch",
]
