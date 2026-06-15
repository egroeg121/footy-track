# Feature Store — Design

Status: **IMPLEMENTED (v1)** · Code: `src/footy_track/feature_store/` · Tests: `tests/feature_store/`

This document proposes a single, queryable **feature store** for footy-track:
the consolidated home for everything we know about a match's frames —
per-frame metadata, broadcast classification, pitch segmentation,
object detections from multiple sources, and (later) tracks and pitch
projections.

It **supersedes and consolidates** the scattered per-stage Parquet artifacts
sketched in [`../pipelines.md` §Data model](../pipelines.md#data-model-proposed)
(`frames.parquet`, `detections.parquet`, `geometry.parquet`,
`ocr_clock.parquet`, `embeddings.parquet`) and folds in the canonical track
store from [`player_tracking_format.md`](player_tracking_format.md). See
§9 for how the migration relates to those documents.

It honours the cross-stage invariants in
[`../system_design.md` §4](../system_design.md) — `ContinuousTime` as the only
canonical timestamp, normalised top-left xywh boxes, class labels from
`constants.py`, and monotone non-reused track IDs.

---

## 1. Goals and non-goals

**Goals**

- One **portable store per environment** that a human or a downstream
  consumer (footy-stats, FiftyOne, evaluation, overlays) can query with a
  single `SELECT` and no bespoke join glue.
- Hold, for every frame of every game: identity (path, resolution, frame
  index), time (`ContinuousTime` + `GameTime` + half), and the outputs of
  every analysis stage — broadcast classification, pitch segmentation,
  calibration, and object detections.
- Support **multiple independent detection sources** over the same frames
  (hand-labelled ground truth, SAM 3, YOLO runs, future trackers) coexisting
  without overwriting each other, each carrying its own confidence and model
  version.
- Be **idempotent**: re-ingesting the same run must not duplicate rows
  (footy-stats invariant).
- Allow **schema evolution** — adding a new stage's columns or a new source
  must not require rewriting existing data.

**Non-goals**

- A query *API* or serving layer — that is footy-stats' job. This store is
  the substrate footy-stats (and notebooks) read from.
- Re-identification, team assignment, jersey OCR, event extraction — these
  populate columns reserved here but are designed elsewhere.
- The tracker/detector algorithms themselves.
- Real-time/streaming ingestion — this design is batch-on-completion.
  §10 notes the streaming extension point.

---

## 2. The core decision: one *store*, not one *table*

The request is "a single Parquet/DuckDB table with lots of data in it." The
honest engineering answer is: **a single DuckDB database file (the store)
containing a small set of normalised tables at their natural grains, plus a
wide `frame_features` view that presents the per-frame world as one table.**

A literally-single physical table fails on **grain**:

| Fact | Natural grain | Rows for a 90-min game @ 25fps |
|---|---|---|
| frame metadata, clock, broadcast flag, pitch polygon, homography | **1 row per frame** | ~135 k |
| object detections (per source, per run) | **N rows per frame** (≈25 objects × M sources) | millions |
| game metadata (teams, venue, time mapping) | **1 row per game** | tens |
| processing provenance (model, params, run time) | **1 row per run** | hundreds |

Forcing detections into the frame row means array-of-struct columns that are
awkward to filter and explode; forcing frame metadata onto every detection row
means massive duplication and update anomalies (change one pitch polygon →
rewrite 25 M rows). So we **normalise by grain** and **denormalise on read**
via views. The user gets "one table" ergonomics (`SELECT * FROM
frame_features`); the storage stays sane.

### Why DuckDB as the store, Parquet as the substrate

- **DuckDB file = the single store.** One `feature_store.duckdb` is the
  portable, single-file artifact the request asks for. It gives real tables,
  constraints, `UPSERT` (idempotency), and SQL across grains.
- **Parquet = the on-disk table format and interchange.** Each table is also
  materialisable as a partitioned Parquet dataset
  (`store/<table>/game_id=<id>/part.parquet`) for DVC versioning, for
  consumers that don't want DuckDB, and so DuckDB can query it in place
  (`read_parquet`) without import.
- DuckDB reads/writes Parquet natively, so the two are the same data in two
  skins. We treat **Parquet as the source of truth on disk** and the DuckDB
  file as a fast, constraint-enforcing index over it (rebuildable from
  Parquet at any time). This keeps the store DVC-friendly and reproducible.

---

## 3. Schema overview

Five tables plus three convenience views. All timestamps `ContinuousTime`
(seconds, `float64`) per [`../timings.md`](../timings.md). All boxes
normalised top-left xywh in `[0,1]`.

```
game ──1:N── frame ──1:N── detection
  │             │              │
  │             │              └── source/run carried per row (see §6)
  │             └── pitch_segmentation, calibration: inlined on frame (§5.3)
  └── run (provenance for every produced artifact, §7)

track / track_meta: derived from detection, see §8
```

### 3.1 `game` — one row per match

| Column | Type | Notes |
|---|---|---|
| `game_id` | `VARCHAR` PK | e.g. `arsenal_mancity` (matches `data/<match_name>`) |
| `home_team` | `VARCHAR` | |
| `away_team` | `VARCHAR` | |
| `match_date` | `DATE` | nullable |
| `venue` | `VARCHAR` | nullable |
| `source_video_uri` | `VARCHAR` | original video path/URI |
| `fps` | `DOUBLE` | nominal sampling/extraction fps |
| `width` | `INTEGER` | native video resolution |
| `height` | `INTEGER` | |
| `half1_start_continuous_s` | `DOUBLE` | always `0.0` (kept explicit) |
| `half2_start_continuous_s` | `DOUBLE` | `GameTime→ContinuousTime` offset for 2nd half (`GameMetadata`, [`../timings.md`](../timings.md)) |
| `game_start_wallclock` | `TIMESTAMP` | nullable |
| `schema_version` | `VARCHAR` | store schema version this row was written under |

This subsumes the `GameMetadata` block previously living in
`tracks_meta.json`.

### 3.2 `frame` — one row per (game, frame)

The per-frame spine. One row whether or not the frame is a usable broadcast
view (non-broadcast frames are recorded, not dropped — invariant #5).

| Column | Type | Notes |
|---|---|---|
| `game_id` | `VARCHAR` | FK → `game` |
| `frame_index` | `INTEGER` | zero-based frame number within the game |
| `frame_uri` | `VARCHAR` | path to the extracted image (`full_video_frames/*.jpg`) |
| `width` | `INTEGER` | frame resolution (may differ from game native if downscaled) |
| `height` | `INTEGER` | |
| `continuous_time_s` | `DOUBLE` | **canonical** time, seconds from kickoff |
| `half` | `TINYINT` | 1 or 2; nullable until time-mapped |
| `game_time_s` | `DOUBLE` | referee/broadcast clock seconds in-half; nullable |
| **broadcast classification** | | |
| `is_broadcast` | `BOOLEAN` | from §2.2 classifier; nullable until run |
| `broadcast_confidence` | `FLOAT` | nullable |
| `broadcast_model_version` | `VARCHAR` | FK → `run.run_id` or a model tag |
| **pitch segmentation** (§5.3) | | |
| `pitch_polygon` | `LIST<STRUCT(x DOUBLE, y DOUBLE)>` | final pitch boundary, normalised; nullable |
| `pitch_seg_threshold` | `FLOAT` | the allowance/threshold used to derive the polygon |
| `pitch_seg_confidence` | `FLOAT` | nullable |
| `pitch_seg_model_version` | `VARCHAR` | FK → `run.run_id` |
| **calibration** (§5.4) | | |
| `homography` | `LIST<DOUBLE>` (len 9) | flattened 3×3 image→pitch H; nullable |
| `calibration_quality` | `FLOAT` | inlier ratio / quality metric; nullable |
| `calibration_model_version` | `VARCHAR` | FK → `run.run_id` |

PK: `(game_id, frame_index)`. The clock fields come from the OCR stage
(`ocr_clock.parquet` is absorbed here); time-mapping fills `half` /
`continuous_time_s` via `game.half2_start_continuous_s`.

> **Design note — why pitch/calibration are inlined on `frame`, not a
> separate table.** They are strictly per-frame (one polygon, one homography
> per frame) and almost always read alongside the frame. Inlining avoids a
> join for the common case. We keep only **one** "final" segmentation/
> calibration per frame in this table; if we ever need to compare *multiple*
> segmentation runs per frame, segmentation graduates to its own
> source-tagged table exactly like detections (§6). Flagged in §11.

### 3.3 `detection` — one row per (game, frame, source, object)

The high-volume table. Every detected object from every source/run is a row.

| Column | Type | Notes |
|---|---|---|
| `game_id` | `VARCHAR` | FK → `game` |
| `frame_index` | `INTEGER` | FK → `frame` |
| `continuous_time_s` | `DOUBLE` | denormalised from `frame` for time-range scans without a join |
| `detection_id` | `BIGINT` | **per-frame** object index (0..N-1) within `(game_id, source, run_id, frame_index)`; see §6 |
| `source` | `VARCHAR` (dict) | `hand_label` \| `yolo` \| `sam3` \| `bytetrack` \| … (§6) |
| `run_id` | `VARCHAR` | FK → `run`; pins model + params + version |
| `label` | `VARCHAR` (dict) | class from `constants.py` |
| `confidence` | `FLOAT` | `[0,1]`; `1.0` (or null) for hand labels |
| `bbox_x` | `FLOAT` | normalised top-left x |
| `bbox_y` | `FLOAT` | |
| `bbox_w` | `FLOAT` | |
| `bbox_h` | `FLOAT` | |
| `mask_ref` | `VARCHAR` | nullable; pointer to RLE/PNG mask sidecar for SAM3 (§5.5) |
| `track_id` | `INTEGER` | nullable; set by the tracking stage (§8) |
| `is_interpolated` | `BOOLEAN` | tracker fill vs raw detection (§8) |

PK: `(game_id, source, run_id, frame_index, detection_id)`. `detection_id` is
the **per-frame** object index, so detectors emit `0..N-1` per frame with no
need for a run-global counter; re-ingesting one frame's detections upserts
cleanly. Partitioned on disk by `game_id` then `source` (§6) so a single
source's detections are a cheap column/partition scan and one source can be
rewritten without touching others.

### 3.4 `track_meta` — one row per track (§8)

Per-track summary (label, span, identity). Kept separate from `detection`
because it is updated post-hoc (re-ID, team, jersey) without rewriting the
row store. Replaces the `tracks` block of `tracks_meta.json`.

| Column | Type | Notes |
|---|---|---|
| `game_id` | `VARCHAR` | |
| `source` | `VARCHAR` | the tracker source (e.g. `bytetrack`) |
| `run_id` | `VARCHAR` | FK → `run` |
| `track_id` | `INTEGER` | monotone within `(game_id, source, run_id)`, never reused |
| `label` | `VARCHAR` | representative class |
| `start_frame` / `end_frame` | `INTEGER` | inclusive |
| `start_continuous_time_s` / `end_continuous_time_s` | `DOUBLE` | last *real* detection |
| `team_id` | `VARCHAR` | nullable — TODO, team assignment |
| `jersey_number` | `INTEGER` | nullable — TODO, OCR |
| `player_id` | `VARCHAR` | nullable — TODO, identity |
| `reid_parent_track_id` | `INTEGER` | nullable — re-ID link (invariant #6) |

### 3.5 `run` — one row per processing run (provenance)

Every produced value (a broadcast flag, a polygon, a detection set) names the
`run_id` that produced it. This is how we get reproducibility and how two
YOLO runs with different weights coexist (§6).

| Column | Type | Notes |
|---|---|---|
| `run_id` | `VARCHAR` PK | stable id, e.g. `yolo11n_v3__2026-06-10` or a content hash |
| `stage` | `VARCHAR` | `broadcast` \| `detection` \| `pitch_seg` \| `calibration` \| `tracking` |
| `source` | `VARCHAR` | the `source` value rows from this run carry |
| `model_name` | `VARCHAR` | e.g. `yolo11n.pt`, `sam3.pt`, `human` |
| `model_version` | `VARCHAR` | weights tag / git sha / roboflow dataset version |
| `params_json` | `VARCHAR` | JSON of thresholds/config (e.g. conf threshold, tracker yaml) |
| `created_at` | `TIMESTAMP` | when the run was produced |
| `code_version` | `VARCHAR` | footy-track git sha |
| `schema_version` | `VARCHAR` | store schema version |

### 3.6 Views (the "one table" ergonomics)

- `frame_features` — `frame` ⨝ `game`: every per-frame fact plus game
  metadata, one row per frame. The "single wide table" the request describes.
- `detections_enriched` — `detection` ⨝ `frame` ⨝ `run`: each detection with
  its frame's time/broadcast context and its run's model metadata.
- `tracks_enriched` — `detection` ⨝ `track_meta`: tracked detections with
  per-track identity.

Views cost nothing on disk and can be `CREATE TABLE AS` materialised if a
consumer wants a frozen flat export.

---

## 4. Embeddings live outside the hot path

Frame/instance embedding vectors (`embeddings.parquet` in the old model) are
**not** stored in `frame` or `detection`. A 512–1024-dim float vector per
frame bloats every scan of the spine. Keep them in a dedicated
`frame_embedding(game_id, frame_index, embedding LIST<FLOAT>, model_version)`
table (or a vector store later), joined only when needed. Same rule for any
future per-instance appearance descriptors used for re-ID.

---

## 5. How each requested feature maps in

### 5.1 Per-frame info (path, resolution, clock, frame number)
→ `frame` columns `frame_uri`, `width`/`height`, `continuous_time_s` +
`half` + `game_time_s`, `frame_index`. (§3.2)

### 5.2 Broadcast features (binary yes/no, confidence, model version)
→ `frame.is_broadcast` (BOOLEAN), `broadcast_confidence`,
`broadcast_model_version`. The `model_version` resolves to a `run` row for
full provenance. (§3.2, §3.5)

### 5.3 Pitch segmentation (final polygon + thresholds/allowances)
→ `frame.pitch_polygon` (list of normalised points = the final segregation
line), `pitch_seg_threshold` (the allowance), `pitch_seg_confidence`,
`pitch_seg_model_version`. Polygon stored as `LIST<STRUCT(x,y)>` (DuckDB
native, explodes cleanly); a WKB/GEOMETRY column is an alternative if we
adopt DuckDB's `spatial` extension for point-in-polygon "is this player on
the pitch" queries — recommended once we need geometric predicates. (§3.2)

### 5.4 Calibration / homography
→ `frame.homography` (flattened 3×3), `calibration_quality`. Absorbs
`geometry.parquet`. (§3.2)

### 5.5 Object detections from multiple sources
→ `detection` table, one row per object, tagged by `source` + `run_id`.
Hand labels, SAM 3, and YOLO predictions over the same frame are distinct
rows distinguished by `source`/`run_id`; none overwrites another. SAM 3
instance **masks** are large — store the box in `detection` and the mask as
RLE/PNG in a sidecar referenced by `mask_ref` (don't inline mask blobs in the
hot table). (§3.3, §6)

### 5.6 Tracking (longer term)
→ `detection.track_id` + `detection.is_interpolated` for per-frame
membership, `track_meta` for per-track summary and identity. Schema and
lifecycle already specified in
[`player_tracking_format.md`](player_tracking_format.md); this store is where
those rows live. (§3.3, §3.4, §8)

---

## 6. Multiple sources over the same frames — the central pattern

The hardest requirement is "object detections likely from multiple
sources/runs — hand-labelled, SAM 3, YOLO — all with locations and
confidences." The pattern:

- **`source`** is the *kind* of producer (`hand_label`, `yolo`, `sam3`,
  `bytetrack`). **`run_id`** is a *specific* production of that source
  (weights + params + date). Two YOLO weights → two `run_id`s, same `source`.
- A detection's full key is `(game_id, source, run_id, frame_index, detection_id)`. The
  same physical player in the same frame appears as **separate rows** under
  each source — by design. No source is privileged in storage.
- **Idempotency**: re-running a run upserts on the PK and replaces that run's
  partition; it never touches other sources. Concretely, ingestion writes
  `detection/game_id=<id>/source=<src>/run_id=<rid>/part.parquet` atomically
  (write-temp-then-rename), so a re-run is a partition swap.
- **Comparison/eval becomes a self-join**: ground truth is just
  `source='hand_label'`; precision/recall vs YOLO is
  `detections_enriched WHERE source='hand_label'` joined to
  `source='yolo'` on IoU. No separate eval format needed — the MOT-CSV
  exporter from `player_tracking_format.md` §4 becomes an optional view, not a
  storage concern.

Partition layout on disk:

```
store/
  feature_store.duckdb            # the single store (index over parquet)
  game/part.parquet
  run/part.parquet
  frame/game_id=<id>/part.parquet
  detection/game_id=<id>/source=<src>/run_id=<rid>/part.parquet
  track_meta/game_id=<id>/source=<src>/run_id=<rid>/part.parquet
  frame_embedding/game_id=<id>/part.parquet
  masks/game_id=<id>/<frame>__<detection_id>.png   # mask_ref targets
```

---

## 7. Idempotency, keys, and write protocol

- **Primary keys** (enforced in DuckDB; logically honoured in Parquet):
  - `game(game_id)`
  - `frame(game_id, frame_index)`
  - `detection(game_id, source, run_id, frame_index, detection_id)`
  - `track_meta(game_id, source, run_id, track_id)`
  - `run(run_id)`
- **Upsert semantics**: ingestion is `INSERT … ON CONFLICT DO UPDATE`
  (DuckDB) / partition-replace (Parquet). Re-ingesting an identical run is a
  no-op on row count. This satisfies the footy-stats "re-upload must not
  duplicate" rule from the project CLAUDE.md.
- **Frame spine first**: a run that produces detections requires the
  `(game_id, frame_index)` frame rows to exist (FK). Frame extraction is the
  first writer; analysis stages update/insert against existing frames.
- **Atomic partition writes**: write to a temp path, fsync, rename. A crashed
  run leaves no half-written partition visible.

---

## 8. Tracking integration (forward-compatible)

Tracking reuses the `detection` table rather than a parallel `tracks.parquet`:

- A tracker run writes detection rows with `source` = tracker name (e.g.
  `bytetrack`), `track_id` set, `is_interpolated` flagged for Kalman fills.
- It may either (a) emit fresh detection rows, or (b) be modelled as a
  run that copies a detector run's boxes and adds `track_id`. **(a)** is
  cleaner for provenance and keeps tracker output independently
  rewritable.
- Per-track summary + identity lands in `track_meta`. Lifecycle, ID
  allocation, and re-ID semantics are unchanged from
  [`player_tracking_format.md` §5](player_tracking_format.md) — this doc only
  changes *where the rows live* (the store, not a per-match sidecar).

---

## 9. Relationship to existing docs (what this changes)

| Existing artifact | Fate under this design |
|---|---|
| `frames.parquet` (pipelines.md) | → `frame` table |
| `ocr_clock.parquet` | → `frame.game_time_s` / `half` columns |
| `detections.parquet` | → `detection` table (now multi-source) |
| `geometry.parquet` | → `frame.homography` / `calibration_quality` |
| `embeddings.parquet` | → `frame_embedding` table (§4) |
| `tracks.parquet` + `tracks_meta.json` (player_tracking_format.md) | → `detection` (track_id) + `track_meta` table |
| `GameMetadata` block in sidecar | → `game` table |
| `MatchExporter` (output.md) | reads the store instead of N parquet files; JSON/CSV/FiftyOne exporters become views/queries over the store |

`player_tracking_format.md` stays valid as the **track lifecycle &
semantics** spec; this doc is the **physical storage** spec it plugs into.
`pipelines.md` §Data model should be updated to point here (follow-up).

---

## 10. Why this over the alternatives

| Option | Verdict | Why |
|---|---|---|
| **DuckDB file + partitioned Parquet, normalised by grain** (this design) | **Chosen** | Single portable store, real keys/upsert, multi-source clean, DVC-friendly, schema-evolvable |
| One literal wide table | No | Grain mismatch → array columns or massive duplication; can't rewrite one source cheaply |
| Per-stage Parquet files joined ad-hoc (status quo) | No | The join glue is exactly what the request wants to eliminate; no enforced keys/idempotency |
| Postgres / a server DB | No | Adds an always-on service; loses single-file portability and DVC versioning; overkill for a single-writer batch warehouse |
| FiftyOne as canonical | No | FiftyOne is a *consumer*; we ingest the store **into** it, not vice-versa (matches existing stance) |
| SQLite | No | Row-oriented; poor at the columnar range/aggregate scans (time windows, per-source filters) that dominate here |

---

## 11. Open questions (file as beads before building)

- **Streaming producers.** This is batch-on-completion. A live feed wants
  append-friendly chunk Parquet compacted at end-of-game (same open question
  as `player_tracking_format.md` §6). Pick before building a live path.
- **Multiple segmentation/calibration runs per frame.** Currently "one final
  per frame" inlined on `frame`. If we need to compare segmentation runs,
  promote pitch_seg/calibration to source-tagged tables like `detection`.
  Decide when a second segmentation model appears.
- **Geometry predicates.** Adopt DuckDB `spatial` (WKB `GEOMETRY` for
  `pitch_polygon`) so "is this player inside the pitch" is a SQL
  `ST_Contains`? Recommended but pulls in the extension.
- ~~**`detection_id` allocation.**~~ **Resolved (v1):** `detection_id` is the
  **per-frame** object index (0..N-1) and `frame_index` is part of the PK. This
  matches how detectors emit, needs no run-global counter, and makes
  re-ingesting a single frame's detections a clean upsert. A content-hash
  variant remains an option for non-deterministic detectors but is not needed
  for the deterministic ingest path.
- **Schema-version enforcement on read.** `schema_version` is recorded but
  there's no read-side policy (reject / best-effort upgrade). Same gap as
  noted in `player_tracking_format.md` §6.
- **Store granularity: per-environment vs per-game.** One global
  `feature_store.duckdb` for all games, or one per game with a DuckDB
  `ATTACH`/`UNION` overlay? Per-game files are more DVC-friendly and
  parallel-write-safe; a global view attaches them. Leaning per-game files +
  global attach view.
- **Embeddings storage.** Plain `LIST<FLOAT>` Parquet vs a real vector index
  (lancedb / duckdb-vss) once re-ID/search needs ANN. Defer until there's a
  search workload.

---

## 12. Summary

- **One store** (`feature_store.duckdb` over partitioned Parquet),
  **normalised by grain**: `game` → `frame` → `detection`, plus `track_meta`
  and `run` provenance; embeddings off to the side.
- The request's "single table" is delivered as the **`frame_features` view**
  (frame ⨝ game) — one wide per-frame table to query — without paying the
  duplication cost in storage.
- **Multiple detection sources** (hand label / SAM 3 / YOLO / trackers)
  coexist as `source` + `run_id`-tagged rows; none overwrites another, eval
  is a self-join, re-runs are idempotent partition swaps.
- **Pitch segmentation** (final polygon + thresholds), **broadcast** flags,
  and **calibration** are inlined per-frame with their own model-version
  provenance; **tracking** plugs into `detection`/`track_meta` per the
  existing lifecycle spec.
- Folds together every previously-scattered Parquet artifact; honours the
  ContinuousTime / bbox / class-label / track-ID invariants; stays
  DVC-versioned and schema-evolvable.
