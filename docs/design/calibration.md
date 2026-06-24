# Calibration Stage — Design

Status: **DRAFT** · Issue: ft-wn9

This document specifies the **Calibration** stage (§2.3 of
[`system_design.md`](../system_design.md)). It estimates the per-frame
homography `H` that maps **image** coordinates onto a canonical **2D pitch**
in metres, plus a scalar `H_quality`. Its output, `geometry.parquet`, is
consumed unchanged by [2D Projection](projection.md).

---

## 1. Stage in §2 of system_design.md

This is **Calibration** (§2.3). It runs in parallel with Detection but is
conceptually upstream of 2D Projection, which needs both detections and `H`.

It produces:

- `geometry.parquet` — one row per frame with `frame_index`, `H_flat`
  (9 floats, row-major 3×3) and `H_quality ∈ [0, 1]`.

This schema is exactly what `PitchProjector.project_match`
([`projector.py`](../../src/footy_track/projection/projector.py)) reads, so
calibration unblocks end-to-end projection.

---

## 2. Method — manual keypoints (v1)

Calibration is a chicken-and-egg problem in the broadcast setting: camera
pose changes every few frames (pan / zoom), lens distortion bends lines, and
only part of the pitch is visible at once. Rather than block on a learned
line/keypoint detector, v1 implements **method #1** from the issue: manual
keypoint annotation.

A labeller marks where known pitch landmarks appear in a frame. Each
correspondence is `(landmark_name, image_x, image_y)` in normalised image
coordinates. Given **≥ 4** such correspondences, we solve for `H` with
`cv2.findHomography(..., method=cv2.RANSAC)`.

- **Input direction**: source = image points, destination = pitch metres, so
  the resulting `H` maps **image → pitch** (the direction the projector
  expects).
- **RANSAC**: rejects mislabelled / occluded points; the inlier mask drives
  the quality metric.

Learned keypoint detection (SoccerNet-style) is the documented follow-up. It
replaces only the *front-end* that produces correspondences; the `H` /
`H_quality` contract and `geometry.parquet` schema are unchanged, so the
projector never needs to know which front-end produced the geometry.

---

## 3. Canonical pitch model

[`pitch_model.py`](../../src/footy_track/projection/pitch_model.py) defines
named landmarks in the **same coordinate frame as the projector**
([`projection.md`](projection.md) §2):

- Origin at the **centre spot**.
- `+x` toward the attacking (home) goal, range `[-L/2, +L/2]`.
- `+y` toward the technical-area touchline, range `[-W/2, +W/2]`.
- Units: metres; pitch dimensions default to 105 × 68.

Landmarks scale with pitch dimensions where appropriate (corners, halfway
line, touchlines) but fixed IFAB features (penalty area 16.5 m deep, centre
circle radius 9.15 m, penalty spot 11 m from goal) are absolute metres, not
scaled with overall pitch size. Names use `left` (`-x`, defending) and
`right` (`+x`, attacking) so each correspondence is unambiguous.

---

## 4. Module interface

```python
# src/footy_track/projection/calibrator.py

@dataclass
class KeypointLabel:
    frame_index: int
    landmark: str        # name from PitchModel
    image_x: float       # normalised [0, 1]
    image_y: float

class Calibrator:
    def __init__(self, pitch_dims=(105.0, 68.0),
                 ransac_threshold=0.01,
                 quality_error_scale_m=5.0): ...

    def calibrate_frame(
        self, frame_index: int, labels: list[KeypointLabel]
    ) -> CalibrationResult: ...

    def calibrate_match(
        self, labels_parquet: Path, out_parquet: Path
    ) -> None: ...
```

`CalibrationResult` carries `H` (or `None`), `H_quality`, plus diagnostics
(`n_correspondences`, `n_inliers`, `mean_reproj_error_m`).

---

## 5. Quality metric

`H_quality` is derived from the **mean reprojection error** over RANSAC
inliers: image points are mapped through `H` and compared to their pitch
landmark targets. The error (metres) is mapped to `[0, 1]` by an exponential
decay:

```
H_quality = exp(-mean_reproj_error_m / quality_error_scale_m)
```

- A perfect fit → `1.0`.
- Error equal to `quality_error_scale_m` (default 5 m) → `1/e ≈ 0.37`.

This is the coarse propagation referenced in
[`projection.md`](projection.md) §5: the projector treats
`H_quality < threshold` (default 0.3) as low-confidence and emits NaN pitch
coordinates rather than a confident wrong point.

---

## 6. Invariants honoured

- **No frame silently dropped.** An unsolvable frame (< 4 correspondences or
  a degenerate configuration) is still written, with an identity `H_flat`
  and `H_quality = 0.0`. The projector then records its detections as
  low-confidence (NaN), honouring system invariant §4.5.
- **`H` direction is image → pitch**, matching the projector's `_apply_homography`.
- **Independently replaceable.** A learned calibrator must satisfy the same
  `H` / `H_quality` contract and emit `geometry.parquet` in this schema.

---

## 7. Open questions / follow-ups

- **Temporal smoothing**: per-frame independent `H` can jitter under pan /
  zoom. Smoothing (e.g. across stable inlier sets) is deferred — kept out of
  this stage to preserve pure-per-frame geometry, consistent with the
  projector keeping smoothing downstream.
- **Lens distortion**: a single homography assumes a pinhole camera. Strong
  fisheye would need an undistortion pre-pass; not modelled in v1.
- **Automatic keypoints**: learned line/keypoint detection (SoccerNet) to
  remove manual labelling — the primary follow-up; slots in behind the same
  contract.
