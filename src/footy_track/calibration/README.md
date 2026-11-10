# footy_track.calibration

Camera -> pitch calibration. Plan: `docs/design/calibration.md`. This package
currently holds milestone M0 (pitch model, geometry, manual calibration GT via
the labeller `/calibration` page).

## Conventions

* **Pitch frame**: metres, origin at the centre spot, `+x` along the long axis
  (left goal at `-L/2`, right goal at `+L/2`), `+y` toward the **far**
  touchline (the one on the opposite side of the pitch from a main-camera
  operator). Default 105 x 68. A top-down diagram therefore draws `+x` right
  and `+y` up.
* **H maps normalised image coords `[0,1]^2` (top-left origin) to pitch
  metres.** `PitchProjector` applies it directly to normalised anchors. The
  sign of `H` is fixed so image points on the visible ground have `w > 0`.
* `pitch_model.PitchModel(length, width)` exposes line segments, sampled arcs
  and circles (`polylines()`) and named keypoints (`keypoints()`); ids such as
  `L_PEN_BOX_TOP_OUTER` are stable API (GT files reference them).
  `L`/`R` = `-x`/`+x` end, `TOP`/`BOTTOM` = `+y`/`-y` side.
* `geometry.py`: `fit_homography` (DLT, optional RANSAC),
  `reprojection_errors` (px and metres), `project_polyline_to_image` (drops
  points behind the camera so lines never wrap), `warp_to_pitch` (top-down
  raster; the same extents are used by the labeller page).
