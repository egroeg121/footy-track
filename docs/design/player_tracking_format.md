# Player Tracking Data Format — Design

Status: **DRAFT** · Issue: footy_track-dg6

This document proposes the on-disk and in-memory data format for player
tracking output produced by the footy-track pipeline. It is the contract
between the **Tracking** stage (see `docs/pipeline_architecture.md`) and any
downstream consumer (footy-stats, evaluation, visualisation, FiftyOne).

It does **not** prescribe a specific tracker implementation. ByteTrack /
BoT-SORT / SORT / a custom Hungarian assigner can all produce the format
described here.

---

## 1. Goals and non-goals

**Goals**

- Define a stable, versioned schema for `Track`, `Detection`, and
  `BoundingBox` records produced by the tracker.
- Pick a primary file format suited to per-frame, per-object time-series
  data that is easy to append to during processing and easy to query
  downstream.
- Specify the lifecycle of a track ID (creation, update, lost, deleted,
  re-identified) so producers and consumers agree on semantics.
- Stay aligned with existing conventions in the repo:
  - Pydantic schemas (`src/footy_track/schema.py`).
  - ContinuousTime as canonical timestamp (`docs/timings.md`).
  - Normalised xywh bounding boxes (`ObjectDetection` in `schema.py`).
  - Detection class labels from `src/footy_track/constants.py`.

**Non-goals**

- Defining the action/event schema (covered separately by `labelling.py`
  and the footy-stats action tag model).
- Defining the team-roster / player-identity schema (jersey number → player
  ID resolution is flagged as an open question below).
- Specifying the tracker algorithm itself.

---

## 2. Background — existing tracking output formats

A short survey of formats we considered. None is adopted verbatim; the
proposed schema in §3 is a superset of the useful fields from each.

### 2.1 Ultralytics `results.boxes` (BoT-SORT / ByteTrack)

When `model.track(...)` is called, Ultralytics returns a `Results` object
per frame whose `boxes` attribute exposes:

| Attribute | Shape | Meaning |
|---|---|---|
| `boxes.xyxy` | `(N, 4)` | absolute pixel xyxy |
| `boxes.xywh` | `(N, 4)` | absolute pixel centre xywh |
| `boxes.xywhn` | `(N, 4)` | normalised centre xywh |
| `boxes.conf` | `(N,)` | detection confidence |
| `boxes.cls` | `(N,)` | class index into `model.names` |
| `boxes.id` | `(N,)` or `None` | tracker-assigned integer track ID |

Notes:
- `boxes.id` is `None` for any detection the tracker did **not** associate
  with a confirmed track this frame. Producers must handle the `None` case.
- The tracker config (`bytetrack.yaml`, `botsort.yaml`) controls
  `track_buffer` (frames a lost track is kept alive) and
  `track_high_thresh` / `track_low_thresh`.
- Ultralytics emits per-frame results; there is no native multi-frame file
  format. We must serialise.
- Ultralytics uses **centre xywh**; our `ObjectDetection` schema uses
  **top-left xywh**. The conversion happens at the detector boundary.

### 2.2 MOT Challenge CSV

The MOTChallenge benchmark (`MOT16`, `MOT17`, `MOT20`) standard output is
one row per detection per frame:

```
<frame>, <id>, <bb_left>, <bb_top>, <bb_width>, <bb_height>, <conf>, <x>, <y>, <z>
```

- `frame`: 1-indexed frame number.
- `id`: integer track ID; `-1` for un-tracked detections in some variants.
- `bb_*`: absolute pixel bounding box (top-left + width/height).
- `conf`: detection confidence (also used as a "ignore" flag in ground
  truth via `0` / `1`).
- `x, y, z`: 3D world coordinates; usually `-1, -1, -1` for 2D MOT.

Strengths: extremely simple, supported by every MOT evaluator
(`py-motmetrics`, `TrackEval`).

Weaknesses: no class label column (MOT assumes a single class), no time
column (only frame index), no schema version, no per-track metadata, no
team / jersey columns.

### 2.3 ByteTrack / BoT-SORT internals

Both algorithms model a track as an `STrack` with state in
`{New, Tracked, Lost, Removed}` and Kalman-filter predictions. The
relevant lifecycle observations:

- **New**: detected this frame, not yet confirmed (high-conf detection on
  first appearance).
- **Tracked**: associated with a detection in the current frame.
- **Lost**: not associated this frame, but inside `track_buffer`. Eligible
  for re-association.
- **Removed**: outside `track_buffer`; ID will not be reused.

Importantly, **track IDs are not reused once removed**. A player who
leaves and re-enters the frame after `track_buffer` will get a new ID
unless an explicit re-ID step links them. This is a known limitation that
our open questions section addresses.

---

## 3. Proposed schema

The schema is split into three concentric records: `BoundingBox` (geometry
only), `Detection` (per-frame observation), `Track` (the time-series of
detections sharing an ID).

All schemas are Pydantic `BaseModel` to align with `src/footy_track/schema.py`.

### 3.1 `BoundingBox`

```python
class BoundingBox(BaseSchema):
    """Top-left normalised xywh, matching ObjectDetection in schema.py."""
    x: float = Field(..., ge=0.0, le=1.0)  # top-left x, normalised
    y: float = Field(..., ge=0.0, le=1.0)  # top-left y, normalised
    w: float = Field(..., ge=0.0, le=1.0)
    h: float = Field(..., ge=0.0, le=1.0)
```

Rationale: matches the existing `ObjectDetection` field layout exactly so
no conversion is needed inside the pipeline. Absolute-pixel coordinates
can be recovered when frame `width` / `height` are known (carried on
`FrameDetections`).

### 3.2 `Detection`

A single per-frame observation belonging to a track.

```python
class Detection(BaseSchema):
    frame_index: int                      # zero-based frame number
    continuous_time_s: float              # ContinuousTime, seconds (canonical)
    track_id: int                         # see §4 lifecycle
    label: str                            # class label, see constants.py
    confidence: float = Field(..., ge=0.0, le=1.0)
    bbox: BoundingBox
    detector_model: str | None = None     # e.g. "yolo11n.pt@v3"
    tracker: str | None = None            # e.g. "bytetrack", "botsort"
    is_interpolated: bool = False         # true if filled from Kalman, no raw det
```

Notes:
- `continuous_time_s` is the canonical timestamp. `frame_index` is kept
  for convenience and round-tripping with frame-indexed sources but
  consumers MUST treat `continuous_time_s` as the source of truth (see
  `docs/timings.md`).
- `is_interpolated = True` lets us preserve the difference between a real
  raw detection and a Kalman-predicted box for a `Lost`-state track.
  Default `False` — most rows will be real observations.
- `label` is a free string today (matches `ObjectDetection`). It is
  expected to be one of `DETECTION_CLASSES` from `constants.py`.

### 3.3 `Track`

The time-series of detections sharing an ID, plus track-level metadata.

```python
class Track(BaseSchema):
    track_id: int
    label: str                            # majority/representative class
    start_frame: int
    end_frame: int                        # inclusive
    start_continuous_time_s: float
    end_continuous_time_s: float
    detections: list[Detection]           # sorted by frame_index
    # --- TODO fields, see §6 ---
    team_id: str | None = None
    jersey_number: int | None = None
    player_id: str | None = None
    reid_parent_track_id: int | None = None
```

Invariants (enforced by validator):
- `len(detections) >= 1`.
- `detections` is sorted by `frame_index`, strictly increasing.
- `start_frame == detections[0].frame_index`,
  `end_frame == detections[-1].frame_index`.
- All `detections[i].track_id == track_id`.

The materialised `Track` is the *summary* form; the on-disk format (§4)
stores raw `Detection` rows and reconstructs `Track` on read.

---

## 4. File format recommendation

### 4.1 Recommendation: **Parquet, one file per match, row = Detection**

Primary on-disk format: a single Parquet file per match, e.g.

```
data/<match_name>/tracks/tracks.parquet
```

Schema (Arrow types):

| Column | Arrow type | Notes |
|---|---|---|
| `match_id` | `string` | redundant per-row but cheap with dict encoding |
| `frame_index` | `int32` | zero-based |
| `continuous_time_s` | `float64` | canonical |
| `track_id` | `int32` | see §5 |
| `label` | `dictionary<string>` | class label |
| `confidence` | `float32` | |
| `bbox_x` | `float32` | normalised top-left x |
| `bbox_y` | `float32` | |
| `bbox_w` | `float32` | |
| `bbox_h` | `float32` | |
| `detector_model` | `dictionary<string>` | nullable |
| `tracker` | `dictionary<string>` | nullable |
| `is_interpolated` | `bool` | |

Rationale:
- **Columnar + compressed.** A 90-minute match at 25 fps with ~25 tracked
  objects per frame is ~3.4M rows. Parquet handles this in ~tens of MB
  with dictionary encoding on `label`/`match_id`.
- **Append-friendly enough.** We write per-match, end-to-end; no need for
  mid-row updates. Streaming producers can buffer a chunk (e.g. one
  half) and write atomically.
- **Native Pandas/Polars/DuckDB support.** footy-stats can ingest with
  `read_parquet` directly; analysts can `SELECT … FROM tracks.parquet` via
  DuckDB.
- **Schema evolution.** Parquet supports adding nullable columns
  (e.g. `team_id`, `player_id` once those questions are resolved) without
  rewriting old files.

Sidecar JSON for track-level metadata:

```
data/<match_name>/tracks/tracks_meta.json
```

```jsonc
{
  "schema_version": "1.0.0",
  "match_id": "arsenal_mancity",
  "produced_by": {
    "detector": "yolo11n.pt@v3",
    "tracker": "bytetrack",
    "tracker_config": "bytetrack.yaml@<sha>"
  },
  "video": {
    "width": 1920,
    "height": 1080,
    "fps": 25.0
  },
  "game_metadata": {
    "half_start_continuous_s": 2910.0
  },
  "tracks": {
    "<track_id>": {
      "label": "player",
      "start_frame": 12,
      "end_frame": 1873,
      "start_continuous_time_s": 0.48,
      "end_continuous_time_s": 74.92,
      "team_id": null,        // TODO
      "jersey_number": null,  // TODO
      "player_id": null,      // TODO
      "reid_parent_track_id": null
    }
  }
}
```

Rationale: the per-row Parquet stays small and class-label-only;
expensive-to-recompute summary fields and per-track identity attributes
live in the sidecar where they can be edited or backfilled without
rewriting the row store.

### 4.2 Alternatives considered

| Format | Verdict | Why not |
|---|---|---|
| MOT-style CSV | Useful for **eval only** (`py-motmetrics`) | No class label, no time, no schema versioning, poor compression at match scale |
| JSONL (one Detection per line) | Reasonable fallback | ~10× larger on disk, slower to query, no column projection |
| Single big JSON | No | Loads fully into RAM; bad for streaming |
| FiftyOne dataset only | Not as canonical | FiftyOne is a *consumer* — we should be able to ingest the canonical Parquet **into** FiftyOne, not require FiftyOne to read the canonical format |
| SQLite | Considered | Acceptable but Parquet wins on column scans, file portability, and DVC-friendliness |

We will provide a small **MOT-CSV exporter** for evaluator compatibility
(see §6 TODO).

---

## 5. Track ID lifecycle

The pipeline-level lifecycle a producer MUST honour. This is independent
of the Kalman / state-machine internals of the underlying tracker.

### 5.1 States (pipeline-level)

| State | Producer responsibility |
|---|---|
| **Tentative** | Track has been seen for fewer than `min_hits` frames. **Not emitted** to the Parquet output. |
| **Confirmed** | Track has been associated for ≥ `min_hits` frames. From here on, every frame the track exists in (whether tracked or interpolated) emits a `Detection` row. |
| **Lost** | Not matched this frame, inside `track_buffer`. The producer MAY emit an interpolated `Detection` (`is_interpolated = True`) using the Kalman prediction, OR emit nothing this frame. The choice is consistent within a single match and recorded in `tracks_meta.json` (TODO key: `lost_state_policy`). |
| **Removed** | Outside `track_buffer`. The track is finalised. `end_frame` and `end_continuous_time_s` in the sidecar reflect the last *real* detection (not interpolated). |

### 5.2 ID allocation rules

1. **Monotonically increasing within a match.** `track_id` starts at 1 and
   only ever increases. The producer reserves an `int64`-safe counter.
2. **No reuse within a match.** Once a track is `Removed`, its ID is
   never reassigned. This matches ByteTrack/BoT-SORT default behaviour.
3. **Not stable across matches.** `track_id` is local to one match. A
   match-id + track-id tuple is the cross-match key.
4. **Re-identification** of a player across `Removed` boundaries is
   represented by emitting a **new track** and setting
   `reid_parent_track_id` on the new track to the previous track's ID.
   The original track is **not** modified. This keeps the row store
   append-only and lets re-ID be a separate, post-hoc pass.

### 5.3 Skeleton consumer rule

A consumer reconstructing tracks from the Parquet file:

```python
df = pd.read_parquet("tracks.parquet")
for track_id, group in df.sort_values("frame_index").groupby("track_id"):
    yield Track(
        track_id=track_id,
        label=group["label"].mode().iat[0],
        start_frame=int(group["frame_index"].iloc[0]),
        end_frame=int(group["frame_index"].iloc[-1]),
        start_continuous_time_s=float(group["continuous_time_s"].iloc[0]),
        end_continuous_time_s=float(group["continuous_time_s"].iloc[-1]),
        detections=[Detection(...) for _, row in group.iterrows()],
    )
```

Track-level identity fields (`team_id`, `jersey_number`, `player_id`,
`reid_parent_track_id`) come from the sidecar.

---

## 6. Open questions (TODO)

These are deliberately **not** resolved in this doc. They should each
become their own bead before implementation.

- **TODO: re-identification across `Removed` boundaries.** When a player
  leaves the frame for longer than `track_buffer`, ByteTrack/BoT-SORT
  will issue a new ID. We need a re-ID strategy: appearance embeddings
  (OSNet / TorchReID), team+jersey heuristics, or post-hoc spatial
  proximity. The doc reserves the `reid_parent_track_id` field but does
  not specify how it is populated.

- **TODO: team assignment.** How `team_id` is assigned (kit-colour
  clustering on the bbox crop, manual labelling, classifier head). Needs
  a separate design. Until then `team_id` stays `None` in the sidecar.

- **TODO: jersey number recognition.** OCR on the back/chest crop is the
  obvious approach but accuracy on broadcast video is unproven. Reserve
  the `jersey_number` field; populate it lazily.

- **TODO: streaming vs batch producers.** The current proposal assumes
  a batch producer that writes `tracks.parquet` once at end-of-match.
  For a live-feed producer we likely want either:
  - Append-friendly Parquet (one file per N-second chunk under
    `tracks/chunks/<chunk_id>.parquet`, compacted at end of match), or
  - A row-oriented JSONL tail file that gets converted to Parquet in a
    finalisation step.
  Pick one before building a streaming producer.

- **TODO: schema version negotiation.** `schema_version` is in
  `tracks_meta.json` but there is no read-side enforcement. Define the
  policy (reject? best-effort upgrade?) before we have multiple producer
  versions in the wild.

- **TODO: ground truth / eval format.** Decide whether evaluation uses
  the canonical Parquet directly via a footy-track-shaped metric, or
  exports to MOT-CSV and runs `py-motmetrics` / `TrackEval`. The
  recommendation is "both: canonical for storage, MOT-CSV for the
  evaluator", but the exporter is not yet written.

- **TODO: ball as a track.** The schema does not distinguish person-
  tracks from ball-tracks beyond the `label` column. Confirm that ball
  trajectories want the same `Track` shape, or whether they need their
  own (e.g. with possession / out-of-play state).

- **TODO: how `Track` integrates with `FrameDetections`.** Today
  `FrameDetections` (in `schema.py`) holds untracked per-frame
  detections. We need to decide whether `Track`-aware code reads the
  Parquet directly or whether `FrameDetections` gains a `track_id` field
  on `ObjectDetection`.

---

## 7. Summary

- **Schema**: `BoundingBox` (normalised top-left xywh, matches existing
  `ObjectDetection`) → `Detection` (per-frame, carries ContinuousTime and
  `track_id`) → `Track` (summary, sidecar-stored).
- **File format**: per-match Parquet of `Detection` rows + JSON sidecar
  for track-level metadata. MOT-CSV is an exporter, not the canonical
  format.
- **Lifecycle**: confirmed-only emission, monotonic non-reused IDs, re-ID
  via a new track linking back to the parent — all post-hoc and
  append-only friendly.
- **Open questions** around re-ID, team / jersey identity, and streaming
  producers are deliberately deferred and tagged as TODOs above.
