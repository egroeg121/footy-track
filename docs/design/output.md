# Output Stage — Design

Status: **DRAFT** · Issue: footy_track-yjd

This document specifies the **Output** stage (§2.7 of
[`system_design.md`](../system_design.md)) — the serialisation layer
that fans out joined per-stage records to consumers.

The canonical Parquet store for tracks is already specified in
[`player_tracking_format.md`](player_tracking_format.md). This doc
covers the parts that are still planned: per-stage Parquet artifacts,
JSON / CSV exporters, and the FiftyOne integration.

---

## 1. Stage in §2 of system_design.md

This is the **Output** stage (§2.7). It consumes:

- `tracks.parquet` + `tracks_meta.json` (from §2.5).
- Per-stage parquet artifacts (`detections.parquet`, `geometry.parquet`,
  `ocr_clock.parquet`, `frames.parquet`, `embeddings.parquet`) per
  [`pipelines.md` §Data model](../pipelines.md#data-model-proposed).
- (When 2D Projection lands) `pitch_positions.parquet`.

It produces:

- A FiftyOne dataset (one sample per frame, with detections and any
  available track / pitch annotations).
- JSON / CSV exporters keyed by `match_id` for ad-hoc analytics and
  for ingest into footy-stats.

---

## 2. Module interface

```python
# src/footy_track/output/exporters.py

class MatchExporter:
    def __init__(self, match_dir: Path): ...

    def to_fiftyone(self, dataset_name: str) -> fo.Dataset: ...
    def to_json(self, out_path: Path) -> None: ...
    def to_csv(self, out_dir: Path) -> None: ...
```

Each method is pure with respect to `match_dir` — given the same
parquet inputs it produces the same output. Idempotent: re-running
overwrites; never appends.

---

## 3. JSON schema (top-level)

```json
{
  "match_id": "arsenal_bournemouth_1st_half",
  "schema_version": "1.0.0",
  "frames": [
    {
      "continuous_time": 12.34,
      "is_broadcast": true,
      "detections": [...],
      "tracks": [{"track_id": 7, "x": 0.42, "y": 0.71, ...}],
      "pitch_positions": [{"track_id": 7, "x_pitch": 31.2, "y_pitch": 18.0}]
    }
  ],
  "tracks_meta": [...]
}
```

Per-frame ordering is by `continuous_time`. Missing stages (e.g. no
projection yet) are simply omitted from each frame record.

---

## 4. CSV exporters

One file per logical entity, with `match_id` as the primary partition:

| File | Rows |
|---|---|
| `detections.csv` | one row per detection |
| `tracks.csv` | one row per `(track_id, frame)` |
| `pitch_positions.csv` | one row per `(track_id, frame)` in pitch coords |
| `tracks_meta.csv` | one row per track |

CSVs are derived **from the parquet artifacts**, never written
directly by the pipeline. This keeps the canonical store
parquet-first.

---

## 5. FiftyOne integration

Builds on the existing `to_fiftyone_sample` helper in `schema.py`.
Promotion plan:

1. Existing per-frame `BroadcastClassification` and `FrameDetections`
   land as sample-level fields (already partially wired).
2. Add `tracks` field: a `fo.Detections` instance with `track_id` on
   each detection so FiftyOne's track view works.
3. Add `pitch_positions` field: a `fo.Keypoints` instance keyed by
   `track_id` (one keypoint per tracked player per frame).

Dataset name convention: `footy_track__<match_id>`.

---

## 6. Cross-stage invariants honoured

- All exports are keyed by `ContinuousTime` (see [`timings.md`](../timings.md)).
- Bounding boxes are normalised top-left xywh in the JSON / CSV
  exports — pixel conversions only at the FiftyOne boundary, where
  FiftyOne demands them.
- Class labels come from `constants.py`.

---

## 7. Out of scope

- A query API (covered by footy-stats).
- Video overlay rendering (a later component that consumes these
  exports).
- Streaming exports — outputs are batch-on-completion only.
