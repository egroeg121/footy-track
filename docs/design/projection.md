# 2D Projection Stage — Design

Status: **DRAFT** · Issue: footy_track-yjd

This document specifies the **2D Projection** stage (§2.6 of
[`system_design.md`](../system_design.md)). It projects tracked
image-space boxes onto a canonical 2D pitch using the per-frame
homography from Calibration.

---

## 1. Stage in §2 of system_design.md

This is **2D Projection** (§2.6). It consumes:

- `tracks.parquet` (from §2.5) — `Detection` rows with `track_id`.
- Per-frame homography `H` (3×3) from §2.3 Calibration, joined by
  `frame_idx` / `continuous_time`.

It produces:

- `pitch_positions.parquet` — one row per `(track_id, frame)` with
  `(x_pitch, y_pitch)` in pitch metres and a per-row uncertainty.

---

## 2. Pitch coordinate convention

We adopt a **right-handed pitch frame** with the origin at the centre
spot:

- `+x` along the long axis from defending goal to attacking goal of
  the home team.
- `+y` perpendicular, toward the technical-area touchline.
- Units: **metres**. Pitch dimensions per match are read from
  `match_metadata` (default 105 × 68).

A normalised `[0, 1] × [0, 1]` variant is also emitted as
`(x_pitch_norm, y_pitch_norm)` for visualisation tooling that does
not care about real units. Both columns are populated; consumers
pick.

---

## 3. Anchor point

The conventional foot-on-pitch anchor is the **bottom-centre of the
player bbox**:

```
anchor_x = x + w / 2
anchor_y = y + h
```

For the ball, the anchor is the **centroid** (the ball is rarely on
the ground). This is class-dispatched, not a global setting.

---

## 4. Module interface

```python
# src/footy_track/projection/projector.py

class PitchProjector:
    def __init__(self, pitch_dims: tuple[float, float] = (105.0, 68.0)): ...

    def project_frame(
        self,
        detections: list[TrackedDetection],
        H: np.ndarray,            # 3x3 homography image -> pitch
        H_quality: float | None,  # from calibration; propagates as uncertainty
    ) -> list[PitchPosition]: ...

    def project_match(
        self,
        tracks_parquet: Path,
        geometry_parquet: Path,
        out_parquet: Path,
    ) -> None: ...
```

`PitchPosition` is a Pydantic model:

```
track_id: int
continuous_time: float
x_pitch: float
y_pitch: float
x_pitch_norm: float
y_pitch_norm: float
uncertainty_m: float | None   # propagated from H_quality
class_label: str               # for ball vs player anchor selection
```

---

## 5. Uncertainty propagation

For frames with low calibration quality (per §2.3), projection MUST
NOT silently emit a confident point. Behaviour:

- If `H_quality < threshold` (default 0.3): row is emitted with
  `x_pitch / y_pitch = NaN` and `uncertainty_m = +inf`.
- Otherwise: `uncertainty_m` is a coarse propagation of `H_quality`
  through the bottom-centre anchor; exact formula deferred to the
  calibration design doc.

This honours system invariant §4.5 (non-broadcast frames are
recorded, not silently dropped) extended to **low-confidence
calibration**.

---

## 6. Cross-stage invariants honoured

- `ContinuousTime` is the only canonical timestamp on output rows.
- Class labels for anchor dispatch come from `constants.py`.
- The stage is independently replaceable: a swap-in projector must
  satisfy `PitchProjector` and produce `pitch_positions.parquet` in
  the same schema.
- Out-of-pitch positions are emitted as-is, not clamped — clamping
  hides calibration / tracking errors.

---

## 7. Open questions

- Velocity / smoothing: should pitch positions be smoothed (Kalman /
  spline) at this stage, or downstream? Current preference:
  **downstream**, so this stage stays pure geometric projection.
- Z-axis (ball height): not modelled; ball pitch position is the
  ground projection. Acceptable for v1.
