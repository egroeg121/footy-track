# Tracking Stage — Design

Status: **DRAFT** · Issue: footy_track-yjd

This document specifies the **tracker module / algorithm** for stage §2.5
of [`system_design.md`](../system_design.md). The on-disk output format
is already specified in
[`player_tracking_format.md`](player_tracking_format.md); this doc
focuses on the runtime contract and pluggable algorithm.

---

## 1. Stage in §2 of system_design.md

This is the **Tracking** stage (§2.5). It consumes per-frame
`FrameDetections` (from §2.4 Detection) and produces:

- A stream of `Detection` rows tagged with a persistent `track_id`.
- A `tracks_meta.json` sidecar summarising each track's lifetime.

The output schema is locked by `player_tracking_format.md`.

---

## 2. Module interface

```python
# src/footy_track/trackers/base.py

class ObjectTracker(Protocol):
    def update(
        self, frame_detections: FrameDetections, frame_t: ContinuousTime
    ) -> list[TrackedDetection]:
        """Assign track IDs to a single frame's detections.
        Stateful — call once per broadcast frame in time order."""

    def finalise(self) -> list[TrackMeta]:
        """Return per-track summaries after the last update()."""
```

`TrackedDetection` extends `ObjectDetection` with `track_id: int` and
`continuous_time: float`. `TrackMeta` matches the `tracks_meta.json`
schema in `player_tracking_format.md` §6.

---

## 3. Concrete implementations

| Implementation | Backing | Notes |
|---|---|---|
| `UltralyticsTracker` | `model.track(...)` (BoT-SORT or ByteTrack) | Wraps the Ultralytics native tracker; cheapest path to a working pipeline |
| `LapTracker` | Hungarian assignment via `lap.lapjv` on IoU + class | Custom; used when we need re-ID hooks or reproducibility guarantees |

Both must satisfy the `ObjectTracker` protocol. The tracker is selected
per-pipeline-run via config; downstream code never branches on the
implementation.

---

## 4. Track ID lifecycle (recap)

Authoritative rules live in `player_tracking_format.md` §5. Restated for
this doc:

1. IDs are monotone within a match and never reused.
2. A track is `active` while its detections continue; after `max_age`
   missed frames it is `lost`; after `max_lost_age` it is `deleted`.
3. Re-ID across deletion is represented by a new track with
   `reid_parent_track_id` pointing at the dead track.
4. Non-broadcast frames are skipped at the tracker level — the
   classifier already gated them upstream — but track state must
   survive the gap (a 30-second replay should not split a player's
   track).

---

## 5. Cross-stage invariants honoured

- `ContinuousTime` is the only canonical timestamp on output rows.
- Bounding boxes stay normalised top-left xywh.
- Class labels come from `constants.py`.
- Non-broadcast frames are observable in the output via the gap in
  `frame_idx` and the classifier's per-frame record.

---

## 6. Open questions

- Team / jersey assignment: handled by a downstream component reading
  `tracks.parquet`; not the tracker's job.
- Streaming vs. batch: the protocol is single-frame `update()` so it
  works for both. A batch convenience wrapper can iterate.
- Multi-camera: out of scope for v1.
