# Versioned datasets and progress tracking

Status: **datasets, ingest-all, eval results and CPU eval adapters are built**
(`src/footy_track/datasets/`, `src/footy_track/feature_store/ingest_all.py`, tests in
`tests/datasets/`) and the **`/progress` page is built** ([section 11](#11-progress-page-built)). The
nightly flow is **planned, not built** ([section 12](#12-planned-not-built-yet)).

## 1. Goal

> "A clear dataset for tracking, broadcast view, and pitch calibration that we can track dataset
> changes and performance changes."

Three **independently versioned** datasets, `tracking`, `broadcast` and `calibration`. Each has a
train part and an eval part, taken from the one [split authority](../data.md#traineval-partitioning)
(`footy_track.splits.split_of`). A version is bumped when the labels (or their split) change, and
every model is scored on a dataset version, so two questions are always answerable:

- *what changed in the data?* (the manifest diff between two versions)
- *what changed in performance?* (`eval_result`, one row per dataset version x model x split x metric)

Why it is needed (found 2026-11-17): labelling never reached the feature store (every labeller writes
JSON/JSONL, ingest was a manual CLI, no live store existed); a GT re-ingest replaces rows, so the
store had no history; and the broadcast classifier's accuracy went from 0.98 to 0.41-0.44 when a
Roboflow dataset changed underneath it with nothing recording that it had.

## 2. The pipeline

```
label files (labeller)                    DuckDB store                    snapshots                  results
ball_gt_marks/*.jsonl            \                                   datasets/tracking/vN/
ball_gt_marks/ball_checks/        \   ingest_all   feature_store.duckdb   datasets/broadcast/vN/    eval_result
calibration_gt/ frame_type/        >------------>  (detection, gt_frame,  datasets/calibration/vN/   (dataset version x
calibration_proposals/            /   idempotent    calibration, ...,                                  model x split/cohort)
broadcast_review/<video>.json    /    single writer  broadcast_label)  --datasets bump--> items.parquet + manifest.json
```

1. `python -m footy_track.feature_store.ingest_all` copies every label source into the store.
2. `python -m footy_track.datasets bump all` builds each dataset's item table from the store and
   writes a new immutable snapshot only if the content hash changed.
3. Evaluations score a model on a snapshot and append rows to `eval_result`
   (`python -m footy_track.datasets results ...` or `eval_adapters.evaluate_and_record`).

The store is a rebuildable *index* over the label files; the snapshots are the *record*.

## 3. The three datasets and their items

An **item** is the unit that is counted, hashed, split and diffed. Every item has: `item_id`,
`split`, `match_key`, `clip` (canonical clip id), `game_id`, `frame_index`, `label_kind`, `tier`,
a canonical JSON `payload`, an `item_hash` and `first_version`.

| dataset | item (granularity) | `item_id` | label kinds | payload |
|---|---|---|---|---|
| `tracking` | one labelled **frame** of a clip | `<game_id>:<frame>` | `ball`, `other_boxes`, `no_ball`, `not_broadcast`, `check_only`, `empty` | every box (`label`, `source`, `tier`, `rejected`, x/y/w/h), the Ball Check verdicts on the frame, the sidecar's skip marker |
| `broadcast` | one human **frame verdict** | `<video stem>:<frame>` | `yes`, `no`, `unsure` | verdict, plus the classifier's stored call (`pred`, `p_yes`), why it was sampled (`reason`), `ts_s` |
| `calibration` | one **frame correspondence set** | `<canonical clip>:<frame>` | `homography`, `not_broadcast`, `too_few_points` | the human image to pitch point pairs, the fitted `H` (`norm_image->pitch_m`), `distortion_k1`, image size |

Details that matter when reading the numbers:

- **tracking.** The frame set is every frame a ball-GT sidecar mentions (`gt_frame`) plus every frame
  with a Ball Check verdict. `ball` = at least one ball box that no Ball Check rejected. `no_ball` and
  `not_broadcast` are the sidecar's own skip markers (the detection table drops them, so ingest keeps
  them in `gt_frame`). `check_only` = no sidecar box, only verdicts on detector candidates (hard
  negatives and positives from Ball Check). `tier` is the best tier among the frame's ball boxes (or
  all boxes when there is no ball): **1** human-placed (`labeller`), **2** machine box a human
  confirmed (`human_checked`, or a Ball Check `ball`/`corrected` verdict), **3** machine only. A
  consumer chooses its own GT rule (`ball_eval` uses tier 1, optionally tier 2); tier 3 is stored but is
  not ground truth. One physical clip can be several `game_id`s (`arsenal_mancity_seg010` and
  `arsenal_mancity_20250925_seg010`): they stay separate items (each is a different sidecar) with the same
  canonical `clip`, so they always share a split.
- **broadcast.** Today the only source is the broadcast review (`labeller/broadcast_review.py`):
  frames the classifier was unsure about plus a random sample, judged yes/no/unsure by a human. They
  have no clip, so a 6,000-frame window `<match>_wNNNN` stands in for one (`splits.broadcast_window_id`).
  `unsure` items are kept but never scored. The sampling is biased towards hard frames, so a stored
  classifier's accuracy here is not its accuracy on a random frame; `accuracy_random` is the unbiased subset.
- **calibration.** Human GT only (`calibration_gt/`) plus the human `frame_type/` flags. Machine
  proposals are stored in the store but are the thing being *evaluated*, not part of the dataset. A file is
  copied to every alias `game_id` of its clip; the dataset keeps one item per canonical clip and frame.

## 4. Split tiers

`split` comes from `footy_track.splits.split_of(match_key, clip)`, never from a stored column (it would
go stale as the registers grow). See [data.md](../data.md#traineval-partitioning).

| split | meaning |
|---|---|
| `eval_heldout_match` | a whole match that is evaluation only (`match_splits.json`) |
| `eval_in_match` | one clip in five of a train match (`clip_splits.json`, `sha1(clip) % 5 == 0`) |
| `train` | everything else |

Assignments are sticky and the eval set only grows. `build` raises if a needed register is missing (fail
closed); `FOOTY_CLIP_SPLITS_PREVIEW=1` lets `build` (read-only) show what the hash rule would give while
`clip_splits.json` does not exist yet. `bump` refuses to write from missing registers unless
`--allow-preview` (never do that for a version anyone will compare against).

## 5. Versioning rules

- A dataset's version is bumped **only when its content hash changes**; `bump` is idempotent and safe to run
  nightly.
- Datasets version independently: new broadcast labels do not bump `tracking`.
- Versions are integers starting at 1, parent = previous version. Snapshots are never edited or removed.
- The content hash covers: item ids, splits, match, clip, game id, frame, label kind, tier and the label
  payload. It does **not** cover time stamps, paths, git sha, version numbers or `first_version`. A label edit,
  a new or removed item, or a split reassignment all change it.
- Bump on a meaningful change (a labelling session, a new match, a new batch of Ball Check clips), not
  on every save; the nightly job does it for you.
- A tag of the producing code (`git_sha`, `git_dirty`) is in the manifest but never in the hash.

### Content hash

sha256 over a header line (`{"dataset": ..., "schema": 1}`) followed by one line per item sorted by
`item_id`: `[item_id, split, match_key, clip, game_id, frame_index, label_kind, tier, payload]` as compact
JSON. Payloads are canonical JSON: sorted keys, no spaces, floats rounded to 6 places. Loading a snapshot
re-derives the hash and raises `SnapshotTampered` if it does not match the manifest (the next `bump` also
refuses to build on a tampered parent). Changing `ITEM_SCHEMA` in `datasets/items.py` forces a new version.

## 6. Snapshot layout and immutability

Root: `/mnt/storage/footy_data` (env `FOOTY_DATASETS_ROOT`, then `FOOTY_DATA`; `--root`).

```
datasets/
  tracking/v1/items.parquet  manifest.json
  tracking/v2/...
  broadcast/v1/...
  calibration/v1/...
```

`items.parquet` columns: `item_id, dataset, split, match_key, clip, game_id, frame_index, label_kind,
tier, payload, item_hash, first_version`. Snapshots are written to a temp directory, made read-only and
renamed into place; an existing `vN` is never overwritten (`SnapshotExists`). Writes need the store's write
lock. Every version is also indexed in the store's `dataset_version` table; the files are authoritative and
`bump` re-syncs the table from them.

## 7. Manifest

`manifest.json` per version:

| field | meaning |
|---|---|
| `dataset`, `version`, `created_at`, `note` | identity |
| `content_hash`, `item_schema` | see section 5 |
| `parent_version`, `parent_content_hash` | `null` for v1 |
| `counts` | `n_items`; `by_split`; `by_split_kind` (split x label kind); `by_split_tier`; `clips_by_split` |
| `split_registers` | sha256 of `match_splits.json` and `clip_splits.json`, their paths, the in-match rule, `preview` (true if a register file was missing) |
| `n_items_unregistered_match` | items whose match is in neither register (they are treated as train-pool) |
| `git_sha`, `git_dirty` | code that built it |
| `diff_vs_parent` | items `added` / `removed` / `changed` in total and `*_by_split`; "changed" = same item id, different content |

`python -m footy_track.datasets diff <dataset> A B` lists the added / removed / changed item ids.

## 8. Eval results

Table `eval_result`, append-only, in the feature store:

| column | meaning |
|---|---|
| `task` | `tracking` / `broadcast` / `calibration` |
| `dataset`, `dataset_version` | the snapshot the number was measured on |
| `split` | `train`, `eval_heldout_match`, `eval_in_match`, `eval_all` (both eval tiers) |
| `cohort_version` | NULL = the full current split; N = only items with `first_version <= N` |
| `model_name`, `model_version` | weights sha256, Roboflow version, or a proposals run id |
| `metric`, `value`, `n_items` | the number and how many items it covers |
| `run_at`, `code_sha` | when and with which code |
| `contaminated`, `contamination_note` | see section 10 (NULL = not checked) |

Results are tied to the version they were measured on and are never joined to "current labels". A re-run
appends; the view `eval_result_latest` is the newest run per (task, dataset, version, split, cohort, model,
model_version, metric). `record_results(...)` / `list_results(...)` in `footy_track.datasets.results`;
CLI: `python -m footy_track.datasets results list|record`.

### Cohorts: reading a growing eval set

Because the eval set only grows, `eval(vN)` contains `eval(vN-1)`, and a model scored on v5 is not
comparable to one scored on v2 on the full set. Each model run therefore records **two** lines per
metric and eval split:

- **full current** (`cohort_version` NULL): everything in the split at that version. Best estimate today.
- **fixed cohort** (`cohort_version = k`): only items that were already in the dataset at version k.
  Apples to apples across every model scored on version >= k. Pick k = the version of the oldest model you
  still compare against.

`evaluate_and_record(..., cohorts=[1])` writes both. A progress chart plots the cohort line for trend
and the full line for the current truth.

## 9. Eval adapters

`footy_track/datasets/eval_adapters.py`: `adapter.evaluate(model, dataset_version, split, cohort) ->
EvalOutput(metrics, n_items)`; `evaluate_and_record(...)` runs one over every split and cohort, attaches
contamination flags and records the rows.

| adapter | task | status | metrics | model |
|---|---|---|---|---|
| `CalibrationAdapter` | calibration | **works, CPU** | `success_at_5px`, `success_at_10px`, `mean_/median_reprojection_px`, `mean_reprojection_m`, `failed_rate`, `keypoint_agreement`, `coverage` | `ProposalRunModel(run_id, gt_dirs)`: a stored proposals run in `calibration_proposals/` (reuses `prelabel.evaluate.eval_frame`); any object with `.proposal(item)` works |
| `BroadcastAdapter` | broadcast | **works, CPU** | `accuracy`, `precision_yes`, `recall_yes`, `f1_yes`, `accuracy_random`, `n_yes`, `n_no` | `StoredPredictionModel`: the classifier's call recorded with each frame when it was queued for review (the number the review page shows). Any object with `.says_yes(item)` works |
| `BallTrackerAdapter` | tracking | **works on CPU given a candidate cache** | `ball_precision_/recall_/wrong_per_prediction_` at 10 px and 25 px, `clips_scored`, `clips_without_cache` | `BallTrackerReplay(kind, cache_dir, ...)`: replays `online`/`bidirectional`/`viterbi` over `<clip>.jsonl` caches from `ball_tracking_eval cache` (reuses `ball_eval.detection_replay`) |
| `DetectorAdapter` | tracking | **works, CPU** (scores stored GPU detections) | per class `player_/referee_`: `ap50`, `precision`, `recall`, `f1`, `f1_max`; ball (`ball_` = tier-1 human GT, `ball_t12_` = human + confirmed): `ap_iou50`, `ap_10px`, `ap_25px` and precision / recall / F1 per rule; `map50`, `coverage`, `*_n_gt` | `DetectionStore(dir)`: `<canonical clip>.npz` per eval clip, written by `datasets/detector_run.py` |

### Detector eval (tracking)

`datasets/detector_run.py` runs the detector **once over every frame of every eval clip** (GPU) and stores
the raw detections (conf >= 0.1, all classes, `dets/<clip>.npz`) plus the ball candidates in the
`ball_tracking_eval cache` format (`cache/<clip>.jsonl`); `DetectorAdapter` scores the stored detections
on the CPU and `BallTrackerAdapter` replays trackers over the cache. Resumable per clip; a new dataset
version only needs inference for clips with no detections yet.

* **Frame sourcing.** An item is a labelled frame (`game_id` = video stem, `frame_index` into that video).
  Frames come from the clip video, decoded sequentially from frame 0 (frame-exact; seeking is not), from the
  video the labeller showed: labeller-registry clip dirs (the `*_kf10` re-encodes) first, then
  `<FOOTY_DATA>/clips`. Videos and detections are keyed by canonical clip id (aliases are the same file).
* **Ground truth.** Tier 1 (human) and tier 2 (machine box a human confirmed) boxes; `rejected` boxes are not
  GT; tier 3 boxes and `coach` / `person` / `player_sub` boxes are ignore regions (a detection on one is
  neither hit nor false positive). A frame is scored for a class only if it has GT of that class (ball:
  also `no_ball` frames; never `not_broadcast`), because labelling is not exhaustive: player / referee
  precision is a lower bound.
* **Matching.** Players / referees: IoU >= 0.5. Ball: IoU >= 0.5 (reconciles with the leaderboard mAP50),
  and centre distance <= 10 px / 25 px (native px, the ball tracker's radii). The ball is ~11 px, so IoU 0.5
  needs the centre within ~3 px and mostly measures box tightness; the 10 px rows say whether the ball was
  found, which is what tracking needs. P / R are at conf 0.3 (the pipeline default); AP is over all
  detections with 101-point interpolation (as Ultralytics, so a perfect detector scores 0.995).
* **Contamination.** Flags come from `detection_dataset/roboflow_dataset_11` (`contamination_from_roboflow`)
  when the rows are recorded on the mini; `bournemouth_1st_seg070` (in-match eval) is in v11 train.

The Prefect side (jobs repo `footy_nightly_datasets.py`, `footy-nightly-gpu`) stages the weights / snapshot /
videos to the desktop, runs this, pushes the cache to `ball_tracking_cache/<sha12>/` on the mini and hands the
rows to the mini (`gpu_eval_results/pending/` then `record-rows`; the desktop never opens the store).

## 10. Contamination flags

A number is only an honest estimate if the model did not train on the items it is scored on.
`contamination_flags(TrainingData, dataset_version)` returns a flag per result split:

- **Roboflow / YOLO export dir** (`TrainingData(roboflow_dir=...)`): `splits.contamination` maps each
  training image name to a match / clip and counts those that belong to a held-out match or an eval clip.
  Images that name a match but not a clip (frames of a full-match video) cannot be placed, and the note says
  so, so "not contaminated" can still hide in-match overlap for those models.
- **A dataset version** (`TrainingData(dataset=, version=)`): eval items that share a clip (in-match) or a
  match (held-out) with that version's train items. Sticky assignment makes this empty unless a register was
  edited.
- **Not declared**: every flag is NULL (unchecked), never an implicit "clean".

`contaminated` is true / false / NULL; the note says how many and which clips. `results record
--train-roboflow-dir DIR` or `--train-dataset D --train-version N` sets it on the CLI. Known cases on
2026-11-17 (preview split): the Roboflow v11 detector has 212 of 1,035 train images in an eval clip
(`bournemouth_1st_seg070`); the broadcast classifier's Roboflow 9 / 10 train sets hold about 13%
Nottingham Forest (held-out) frames. Such results should be recorded with `contaminated = true` until the
models are retrained on authority-based exports.

## 11. `/progress` page (built)

`http://<labeller>:8011/progress` (linked from the hub), JSON at `/api/progress`
(`labeller/progress.py`, `progress_routes.py`, `web/progress.html`). It is the one view of project
progress and is global: the `?dataset=` labelling-dataset parameter is ignored. Per dataset
(`tracking`, `broadcast`, `calibration`):

1. **Dataset growth.** A stacked bar chart (inline SVG) and a table of v1..vN from the snapshot
   manifests only (never `items.parquet`): created_at, items by split (train / in-match eval /
   held-out eval), counts by label kind, and the diff vs the parent (added / removed / changed).
2. **Model performance.** From `eval_result_latest`: per model and metric, the value at each dataset
   version, as a line chart plus two tables, "full current eval" (`cohort_version` NULL) and "fixed
   cohort" (shown separately, the tier is part of every row). A contaminated result
   (`contaminated = true`) is drawn red with a warning mark; unknown contamination (NULL) is amber
   with a "?"; neither is ever styled as clean. Hover shows `n_items`, the run time and the
   contamination note. The train split is shown but contamination does not apply to it.
3. **Split health.** Matches held-out vs train (`match_splits.json`) and clips eval vs train
   (`clip_splits.json`), the latest version's eval counts per dataset, and warnings: **zero held-out
   eval items** (a call to action; broadcast has none until a held-out match's broadcast-review frames
   are labelled), no in-match eval, items in an unregistered match, a preview split, and split
   registers that changed since the latest snapshot (run `bump`).
4. **Empty states** with the exact command: no store (`python -m footy_track.feature_store.ingest_all`),
   no versions (`python -m footy_track.datasets bump all`), no results (a `results record` example), no
   clip register (`python -m footy_track.splits assign --apply`). Missing, unreadable or locked files
   degrade the affected section; the page never returns a 500.

Implementation: the store is opened `read_only` for a single query under the shared store lock and
closed again (a writer mid-ingest gives a "store busy" note, not a hang). The payload is cached for 60 s
and invalidated early when a manifest, the store (or its WAL) or a register changes; `?refresh=true`
forces a rebuild. Charts are inline SVG, no external scripts.

## 12. Planned, not built yet

- **Nightly Prefect flow**: `ingest_all`, then `bump all`, then queue evals for any new version (GPU on the
  desktop, the rest on the mini). Its job is to make the CLI above run unattended.
- Not wired yet: detector P / R / mAP50; more broadcast sources than the review (e.g. the
  `frame_type` not-broadcast flags); Roboflow imports into `tracking`; a per-clip `eval_result` for paired bootstrap.

## 13. How to

```bash
# 1. ingest every label source (idempotent; --dry-run parses into memory only)
python -m footy_track.feature_store.ingest_all                # store: $FOOTY_FEATURE_STORE or /mnt/storage/footy_data/feature_store/feature_store.duckdb

# 2. what would change?  (read-only)
python -m footy_track.datasets build tracking
# 3. write the next version if (and only if) the content changed
python -m footy_track.datasets bump all --note "after Nottingham Forest ball checks"
python -m footy_track.datasets list
python -m footy_track.datasets show tracking v2
python -m footy_track.datasets diff tracking v1 v2

# 4. record a result (contamination from the model's training export)
python -m footy_track.datasets results record --task broadcast --dataset broadcast --dataset-version 2 \
    --split eval_heldout_match --model broadcast_classifier --model-version <sha256> \
    --metric accuracy --value 0.91 --n-items 120 --train-roboflow-dir /path/to/roboflow_dataset_10
python -m footy_track.datasets results list --dataset broadcast
```

In Python: `from footy_track.datasets import eval_adapters as ea`;
`ea.evaluate_and_record(ea.BroadcastAdapter(), ea.StoredPredictionModel(), load_version("broadcast"), db,
cohorts=[1])`.

Ingest takes an exclusive lock on `<store>.lock` (single writer; a second writer fails fast with
`StoreLocked`). `bump` and `results record` take it too; `build` and `results list` take the shared lock.
Only the `broadcast_label`, `gt_frame` and the eval / version tables are new; the other tables are the
existing ones (`feature_store.md`).

Environment on the Mac mini: duckdb 1.3.x to 1.5.3 only (2.0 crashes without AVX2); keep CPU use light (one
heavy process at a time). A full ingest of the real labels takes a couple of minutes.

Ingest notes: GT sidecars and calibration files are read from every `marks_dir` of the labeller's
`labeller_datasets.json` (the calib set, the held-out Nottingham Forest sets, the broadcast-review set), so
backup copies (`ball_gt_marks.bak-*`, `ball_gt_marks_MERGED`) are never read. Ball Check logs of one clip in
several dirs are merged by timestamp. Deleting a *file* is not propagated (the labeller truncates, it does not
delete); emptying a file is.

## 14. Why not LanceDB

A LanceDB feature store was designed and spiked first (a native table version per write, tags, time
travel, `merge_insert` upserts). It was rejected because **`import lancedb` / `import lance` abort with
SIGILL on the Mac mini** (i7-2635QM: AVX, no AVX2), and the mini hosts Prefect, the labeller and
`/mnt/storage`. Tested and all crash: lancedb 0.15.0, 0.22.0, 0.30.0, 0.41.0 and pylance 0.19.1. The same
install works on desktop-ubuntu (Ryzen 5 5600G). DuckDB 1.3 to 1.5.3 works on the mini (2.0 does not).

DuckDB plus content-hashed Parquet snapshots delivers the three goals that matter (independent versions,
tracked results, a growing eval set) on the hardware that exists. **Revisit if the host changes** (a CPU with
AVX2, or the store moving to the desktop): Lance would add native time travel and cheaper incremental
versions, nothing else we need.

### Appendix: Lance facts worth keeping (from the rejected design)

Measured with lancedb 0.41.0 / duckdb 1.5.3 on desktop-ubuntu, for whoever revisits it:

- Every write is a new table version; `checkout`, `restore` and per-table **tags** exist, but tags are per
  table, so a multi-table dataset version = the same tag on each table plus a manifest row.
- `optimize(cleanup_older_than=...)` prunes untagged versions and keeps tagged ones; never run it while a
  reader or writer is active.
- `merge_insert` is the upsert. A no-op merge **still creates a version**; unconditional
  `when_matched_update_all` rewrites unchanged rows; **four concurrent threads inserting the same new key
  produced four duplicate rows** (there is no primary key), so every writer needs a file lock plus a
  `row_hash` pre-diff to avoid no-op versions.
- Commit atomicity relies on object-store put-if-not-exists; the spec says nothing about local disk or NFS, so
  multi-host writers over NFS are untested.
- The DuckDB `lance` extension returned 404 for duckdb 1.5.3; the working route is
  `con.register(name, table.to_arrow())`.
- The design's useful ideas live on here: the dataset manifest, the `eval_result` schema with cohorts, a
  `broadcast_label` table, `first_version`, and the consumer map below.

Consumer / producer map (writers all go through `FeatureStore`; readers of JSON sidecars are unaffected):
`ingest_gt` (per-clip delete then upsert), `ingest_calibration`, `ball_checks` (per-row `UPDATE reviewed`,
the riskiest to port), `importers`, `ingest`/`detect_objects`, `difficulty` (uses the private `_conn`),
`export_training_dataset` (reads via `query`; its per-row `UPDATE dataset_tag` is replaced by recording an
export as a dataset version), `claude_verify_v2` (JSONL only). `ball_eval`, `labeller/*` and
`broadcast_review` read sidecars directly.
