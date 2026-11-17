# Lance feature store: versioned datasets and tracked model results

Status: design + spike (`src/footy_track/feature_store/lance_store.py`,
`tests/feature_store/test_lance_spike.py`). Nothing is migrated or deployed.
Context: [`feature_store.md`](feature_store.md) (the current DuckDB store).

## 1. Goals

1. Three independently versioned labelled datasets, each with train + eval:
   **tracking** (player/ball detection + tracks), **broadcast** (is this a main-camera
   view), **calibration** (pitch keypoints + homography). Bumping one never bumps another.
2. Every model (detector/tracker, broadcast classifier, calibration) is evaluated on each
   dataset version, and the numbers are stored so progress is a query, not an archaeology dig.
3. Native table versions / tags / time travel, so "what exactly was `tracking-v3`?" is
   answerable after labels are replaced (today GT re-ingest *replaces* `gt_import_*` rows,
   commit adcb818, and history is gone).
4. The eval set grows as matches are added, without ever moving an item between splits.

## 2. Findings that shape the design

**It is closer to greenfield than a migration.** No live store exists: no `*.duckdb` under
`/mnt/storage`, only a stale July copy. All labelling output is JSON/JSONL under
`/mnt/storage/footy_data` (`ball_gt_marks/*.jsonl`, `ball_gt_marks/ball_checks/`,
`calibration_gt/`, `broadcast_review_labels/`, `identity_labels/`). Ingest is manual CLI only.
So Lance is designed as the store the files are ingested into; a DuckDB-to-Lance copy is
optional (section 9).

**The broadcast dataset is not in the store today.** `frame.is_broadcast` is a *model
prediction*. Human verdicts (`labeller/broadcast_review.py`) live in
`broadcast_review_labels/<match>.json` only. The Lance store adds a `broadcast_label` table.

**The labeller server never writes the store.** It writes JSONL sidecars. All store writers are
batch CLIs/flows (section 5), so the "labeller writes while batch ingests" concurrency worry
does not exist today; it appears only if ingest-on-save is built (section 6).

**HARD BLOCKER on the Mac mini.** `import lancedb` (and `import lance`) aborts with SIGILL
(illegal instruction) on the mini's i7-2635QM (AVX, no AVX2). Measured: lancedb 0.15.0,
0.22.0, 0.30.0 and 0.41.0 all crash on import; pylance 0.19.1 likewise; duckdb 1.5.3 is
fine (duckdb 2.0.0 also crashes there, unrelated to Lance). The same install works on
desktop-ubuntu (Ryzen 5 5600G, AVX2). The mini hosts Prefect, the labeller and
`/mnt/storage`, so *anything that imports Lance cannot run there* without a self-built wheel
(no Rust toolchain installed; untested). See open question 1.

## 3. LanceDB facts (verified; lancedb 0.41.0, duckdb 1.5.3, pyarrow 24.0.0, desktop-ubuntu)

- Versions: every write (add, delete, merge, schema change, compaction, restore) is a new
  version. `table.list_versions()`, `table.version`, `table.checkout(version_or_tag)`,
  `checkout_latest()`, `restore(v)` (creates a new version). Checkout is a read-only view of
  that handle. Docs: <https://docs.lancedb.com/tables/versioning>.
- Tags: `table.tags.create(name, version)`, `.list()`, `.get_version(name)`,
  `.update(name, version)`, `.delete(name)`. **Tags are per table**, not cross-table, so a
  dataset version = the same tag name on each of its tables (plus a manifest row).
- Retention: `table.optimize(cleanup_older_than=timedelta(...))` compacts and prunes
  (`cleanup_old_versions` is deprecated since 0.21 and needs `pylance`). Docs say tagged
  versions are never pruned. Measured with `cleanup_older_than=0`: untagged versions were
  deleted, tagged ones and the latest kept, tagged checkout still returned the old rows,
  and no error was raised (the docs describe an error in some cases; do not rely on either).
  Deleting a tag makes its version prunable again. Never run cleanup with a tiny
  `older_than` while a reader/writer is active (lancedb warns about this).
- `merge_insert(keys).when_matched_update_all().when_not_matched_insert_all()` is the
  upsert; `.when_not_matched_by_source_delete(filter)` gives scoped "replace" semantics in
  one commit. **Gotchas measured:** (a) a merge that changes nothing *still creates a new
  version* (even with `when_matched_update_all(where="target.h != source.h")`); (b) with
  unconditional `when_matched_update_all`, re-ingesting identical rows rewrites them and
  reports `num_updated_rows=N`; (c) **four threads doing `merge_insert` of the same new key
  concurrently produced four duplicate rows**: it is not a uniqueness constraint. Lance
  has no PRIMARY KEY.
- Schema evolution: `table.add_columns(pa.field("x", pa.string()))` (new nullable column,
  new version; old tagged versions keep the old schema).
- Concurrency (<https://lance.org/format/table/transaction/>): optimistic; appends only
  conflict with overwrite/restore; delete/update are rebasable against each other;
  compaction conflicts with overlapping deletes/updates (retryable). Measured: 4 threads x 5
  appends = 20 rows, no loss. Commit atomicity relies on object-store put-if-not-exists /
  rename-if-not-exists; **the spec says nothing about local filesystems or NFS**, so treat
  multi-host writers over NFS as unsafe until tested. Single host, local disk is fine.
- SQL: DuckDB 1.5.3 over Lance works via Arrow: `con.register("t", lance_table.to_arrow())`
  (zero-copy replacement scan). The community `lance` DuckDB extension is **not usable**
  (HTTP 404 for duckdb 1.5.3 linux_amd64 at the time of writing). Cost: a read
  materialises the table into memory (300k rows to Arrow: ~40 ms; fine at footy scale of
  millions of rows, revisit past ~50M).
- Types: the Arrow schema is derived straight from the existing DuckDB DDL
  (`CREATE TABLE ... LIMIT 0` then `.to_arrow_table().schema`), including
  `STRUCT(x,y)[]` and `DOUBLE[9]`. No second schema to maintain.

## 4. Recommendation

- **Storage = Lance, query engine = DuckDB.** Keep `schema.py` as the schema authority and
  keep every view in `schema.VIEWS` unchanged: the spike registers Lance tables in an
  in-memory DuckDB and creates the views as `TEMP VIEW`s; `frame_features`,
  `detections_enriched` etc. return correct results (tested). `FeatureStore.query()` callers
  keep their SQL.
- Idempotent ingest = `merge_insert` on the existing PKs, **plus a `row_hash` column and a
  pre-diff** (anti-join on `row_hash` in DuckDB) so unchanged rows write nothing and make no
  version (spike: `upsert()` returns inserted/updated/skipped; re-ingest leaves the version
  untouched).
- **All writers take one advisory file lock** (`<store>/.write.lock`) because merge_insert
  does not enforce uniqueness. Works on one host; not a distributed lock.
- Dataset version = tag `<dataset>-v<N>` on each of that dataset's tables + a row in
  `dataset_manifest`. Eval results live in an append-only `eval_result` table.
- Do not adopt before resolving the AVX2 blocker (open question 1). The design is
  otherwise sound and the spike passes.

## 5. Table layout

One Lance table per DuckDB table (`game`, `run`, `frame`, `detection`, `track_meta`,
`detection_review`, `calibration`, `pitch_keypoint`), each with an extra `row_hash`
(sha256/16 of the non-PK values). New tables: `broadcast_label`, `dataset_manifest`,
`eval_result`. A `split` column (train | eval_heldout_match | eval_in_match) is carried on the
label rows of each dataset, populated from the split authority (another workstream, branch
`feat/split-authority`: `split_of(match, clip)`, sticky clip register next to
`match_splits.json`); this design does not implement splitting.

Which tables make up each dataset (`DATASET_TABLES` in the spike):

| dataset | tables | label kinds |
|---|---|---|
| tracking | game, run, frame, detection, track_meta, detection_review | ball/player boxes, track ids, Ball Check verdicts (`reviewed`, `not_ball`) |
| calibration | game, run, frame, calibration, pitch_keypoint | per-frame H (`DOUBLE[9]`, `h_convention`, `distortion_k1`), keypoints (`u,v,pitch_x,pitch_y`), `review_status` incl. `not_broadcast`/`too_few_points`; `frame_calibration_best` view picks the winner |
| broadcast | game, frame, broadcast_label | `(game_id, frame_index) -> yes/no/unsure`, `p_yes`, `reason` |

`game`/`frame`/`run` are shared by several datasets: their Lance version is tagged under each
dataset name independently, so a tracking bump does not disturb `calibration-v1`.
Calibration schema already covers keypoints and homography (`schema.py` 433-481, see also
`docs/design/calibration.md` section 6); no change needed beyond `row_hash`/`split`.

Provenance tiers stay as today (`labeller` human, `vittrack`+`human_checked`, machine-only);
machine output is written as tier 3 and never auto-gets `human_checked`.

### `dataset_manifest` (one row per dataset version)

`dataset, version, tag, created_at, note, content_hash, split_register_hash, counts_json,
table_versions_json`. `counts_json` = rows per table and per split / label kind (e.g. ball
boxes by tier, broadcast yes/no, calibrated frames by `review_status`).
`content_hash` = sha256 over the dataset tables' `(PK, row_hash)` sorted by PK, independent
of dates, paths and Lance version numbers, so freezing unchanged labels twice gives the same
hash. This reuses the idea in `feat/ball-offline-harness` (fd1f49a,
`ball_offline/snapshot.py`: content-hashed snapshot, hash excludes date/paths, loader
re-derives it and refuses a tampered directory, every report records the hash).
`split_register_hash` = hash of `match_splits.json` + the clip register at freeze time.
Rule: a "version" is only valid if `content_hash` recomputed from the tagged tables matches.

### `eval_result` (append-only)

```
task, dataset, dataset_version, split, cohort_version,
model_name, model_version, metric, value, run_at, code_sha
```

- `task`: `tracking` | `broadcast` | `calibration` (tracker vs detector distinguished by
  `model_name`).
- `split`: `eval_heldout_match` | `eval_in_match` | `eval_all` (train metrics may be stored
  as `train`). Held-out matches are the honest number; in-match eval is the cheap one.
- `cohort_version`: the eval items counted are those first assigned to eval at or before this
  dataset version. Because assignment is sticky, `eval(vN)` is a superset of `eval(vN-1)`.
  Each model run writes **two** rows per metric: `cohort_version = dataset_version` (the full
  current eval) and `cohort_version = <earlier fixed cohort>` (apples to apples with older
  models). A progress page plots both lines.
- Rows are keyed on all of `task..metric` (so re-running the same model on the same version
  is an upsert, and old results are never deleted when labels change: they stay tied to the
  version they were measured on).
- Never join results to "current labels"; join via `dataset_version` -> manifest -> tag.

## 6. Eval set growth and splits

Two tiers, both sticky (assigned once, never moved), so eval sets only grow:

1. **Held-out matches**: whole matches eval-only (existing `match_splits.json`, sha1-assigned
   for new matches, manual override wins).
2. **In-match eval**: within train matches some clips are eval, chosen by a stable hash of
   `canonical_clip_id` (`clip_id.py`), recorded once in a sticky clip register. Granularity
   is clip/segment, never frame (near-duplicate leakage). Today's
   `export_training_dataset._pick_spread` reshuffles when clips are added; it must be
   replaced by the authority's `split_of`, not extended.

The store only *consumes* `split`. A contamination check (no `train` clip shares a canonical
id or match with an eval clip) runs at freeze time and blocks tagging on failure.

## 7. Consumer / producer map (against main b2f8faa)

Writers (all batch, all go through `FeatureStore`; DuckDB single-writer file today):

| code | what it does | under Lance |
|---|---|---|
| `feature_store/ingest_gt.py` (CLI) | sidecars -> `game/frame/run/detection`; per-clip transaction: `delete_run_rows` then upserts (the T-04 replace) | `upsert` + `delete_where` under the write lock; no multi-statement transactions, so delete+insert is two versions; use one `merge_insert ... when_not_matched_by_source_delete(scope)` for atomic replace. History is kept by tags |
| `feature_store/ingest_calibration.py` (CLI) | `calibration_gt/`, proposals, `frame_type` -> `calibration`, `pitch_keypoint`; per-game `delete_calibration_rows(dataset_tag)` then upsert; reads `SELECT game_id FROM game` | same pattern; the read becomes a Lance scan |
| `feature_store/ball_checks.py` (via `ingest_gt`) | reads ball detections by SQL, writes `detection_review` (delete clip + insert), `UPDATE detection SET reviewed` per matched row | the per-row UPDATE becomes one batched `merge_insert` of changed `reviewed` values (otherwise one version per UPDATE); the **riskiest** port |
| `feature_store/importers.py` | Roboflow / labeller-JSON import: upserts, `source_overlap` query | `upsert` |
| `feature_store/ingest.py`, `scripts/detect_objects.py` | `ingest_frame` -> `upsert_frames/detections` per frame batch | `upsert` (batch per clip, not per frame, or versions explode) |
| `feature_store/difficulty.py` | SQL read then `UPDATE detection SET needs_review` via `store._conn` | batched `merge_insert`; uses private `_conn`, needs a proper API |
| `scripts/export_training_dataset.py` | `store.query` joins over `detections_enriched`/`frame`; `_tag_exported_rows` runs one `UPDATE detection SET dataset_tag` per row; reads `match_splits.py` register | reads: unchanged via DuckDB; the per-row UPDATE is replaced by recording the export as a dataset version (tag + manifest), not by mutating rows |
| `scripts/claude_verify_v2.py` | imports `ball_checks.read_latest_verdicts` (JSONL only) | none |

Readers without the store (unchanged): `ball_eval/dataset.py`, `labeller/ball_check.py`,
`labeller/session.py`, `broadcast_review.py`, `matches_routes.py` all read JSON/JSONL sidecars.
`clip_id.store_game_id_collisions` opens a `.duckdb` read-only and runs two SQL statements
(port to the Arrow-registered connection). `identity/stable_id.py` only mentions the schema.
`tests/feature_store/*` (and `tests/test_clip_id.py`) build `FeatureStore.open(":memory:")`
and call `_conn` / transactions directly; they stay on DuckDB until a `Store` protocol exists.
`feature_store.export_parquet/from_parquet` is superseded by Lance (data is already columnar).

Biggest API gaps vs DuckDB: no PRIMARY KEY enforcement, no multi-statement transactions
(`FeatureStore.transaction()`), no `UPDATE ... WHERE` convenience beyond `table.update`/merge.

## 8. Ingest and dataset-version workflow

1. Ingest is idempotent and may run on every labelling save or job (it is cheap when nothing
   changed: pre-diff skips unchanged rows and writes no version).
2. **Freeze** (`tag_dataset(dataset, N, note, split_register_hash)`): contamination check,
   compute counts + content hash, `tags.create("<dataset>-v<N>", table.version)` on each
   dataset table, append manifest row. Refuses an existing tag (spike-tested). Bump a
   dataset when its labels change meaningfully (e.g. N new Ball-Check clips, a new match),
   not on every save.
3. Train/eval entry points take `--dataset tracking-v3`; the exporter reads tables *at that
   tag*. `eval_result` is written with that version. A model trained on vK evaluated on vN is
   fine; both are recorded.
4. Retention: tagged versions are never cleaned. Run
   `optimize(cleanup_older_than=timedelta(days=14))` on a schedule while no ingest is running
   (take the write lock). Untagged versions in the last 14 days stay for undo. Deleting a tag
   is a deliberate, logged act. Back up `<store>/` plus the manifest to S3 (small: the old
   store was 1.5 MB) with bucket versioning; Lance directories are plain files.
5. Rollback: `table.restore(v)` (new version) or simply re-point work at an older tag.

## 9. Migration steps

Because no live store exists, the default path is **re-ingest from the JSON/JSONL**, not copy.

1. Land the Lance store behind a `Store` protocol (`upsert_*`, `query`, `count`,
   `delete_where`, `tag_dataset`); keep `FeatureStore` (DuckDB) as the implementation tests
   use. Resolve the AVX2 blocker first.
2. Run all ingesters against a fresh Lance store on a Lance-capable host; compare
   `count()` per table and a content hash against a DuckDB store built from the same files
   (dual-ingest for one or two weeks: DuckDB stays the reader of record).
3. Optional one-shot copy of an existing DuckDB file (`duckdb COPY (SELECT ...) -> arrow ->
   merge_insert`), verified by row counts per table and per-table hash of PK + values.
   Only useful if someone has labels in a `.duckdb` that are not reproducible from files.
4. Cutover: readers switch to `LanceFeatureStore.duckdb_connection(tag)`; DuckDB file frozen
   read-only for a release. Rollback = point readers back at the DuckDB file and re-ingest.
5. Tag `tracking-v1`, `broadcast-v1`, `calibration-v1` from today's labels; backfill
   `eval_result` for current models.

## 10. Risks

- **AVX2 / SIGILL on the mini** (section 2). Any host without AVX2 cannot import Lance.
- No uniqueness constraint; correctness of idempotency depends on the write lock + pre-diff.
  A second uncoordinated writer (another host, NFS) can duplicate rows. Add a periodic
  `SELECT pk, count(*) ... HAVING count(*) > 1` check to the freeze step.
- Multi-host writes over NFS are untested/unspecified. Keep one writer host (or one process
  behind a small service) and ship the store elsewhere read-only.
- Version noise: every effective write is a version (per-frame ingest or per-row UPDATE would
  create thousands). Batch per clip; compaction via `optimize`.
- Full-table Arrow materialisation per query connection (fine now).
- Lance/lancedb API churn (`table_names` -> `list_tables`, `cleanup_old_versions`
  deprecated); pin `lancedb` and run the spike test in CI on an AVX2 runner.
- Tag/version semantics apply per table; the manifest is the only cross-table truth.

## 11. Open questions for the user

1. The mini (i7-2635QM, no AVX2) cannot import Lance. Options: (a) make the desktop /
   a newer box the single writer+reader of the store and keep the mini JSON-only;
   (b) build lancedb/pylance from source with `RUSTFLAGS=-C target-cpu=x86-64-v2` on the
   mini; (c) stay on DuckDB and get versioning from the manifest + Parquet snapshots
   (DuckDB `COPY ... PARTITION_BY`, S3 bucket versioning). Which?
2. Is DuckDB + immutable Parquet snapshots + `dataset_manifest` + `eval_result` (no Lance)
   an acceptable fallback? It delivers goals 1, 2 and 4 today; Lance only adds native
   time travel and cheaper incremental versions.
3. Who writes? One designated ingest host, or must the labeller write on save from the mini?
4. How often should dataset versions bump (per labelling session, per N clips, per new match)?
5. Should `eval_result` also capture per-clip metrics (for paired bootstrap, as
   `ball_offline score` does), or only aggregates?
6. Is S3 (`georgebarnett-general-purpose`) acceptable as the store backend instead of local
   disk (Lance supports it natively with atomic commits), which would also remove the
   NFS question?
