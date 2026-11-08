> Review of 2026-11-08, describing `main` @ 26d2506. Part of the [2026-11 review](index.md); prioritised actions are in the [TODO list](../../todo.md).

# footy-track architecture review (read-only)

**Repo:** `egroeg121/footy-track`. **Reviewed tree:** `origin/main` @ `5d4fc90`, read from the clone at
`/tmp/claude-1000/-home-george-code/596c42e4-6975-5ebc-925b-ad0962853a89/scratchpad/ball-check`.
**Date:** 2026-11-08. Nothing was committed, pushed or changed in the repo.

**About the snapshot.** The clone's local `main` is at `eb67f6d`, a merge of `fix/train-workers-cache` that
is **not on origin/main**. It touches one file (`scripts/train_object_detector.py`: it adds `--workers`,
default 8, and `--cache`, default `ram`). The merge message says "default training workers to 0 under WSL",
but the code defaults to **8**, and the help text says the 0 pin applied to WSL only. KB
`wsl-gpu-training-box` says `workers>0` kills the WSL VM. That box has since been rebuilt as bare-metal
Ubuntu, so 8 may be fine there, but the commit message is wrong. Everything else below is the same on both
commits.

**KB pages used as background:** `footy-ball-detection-findings`, `footy-data-plane-reconciliation`,
`footy-labeller-gotchas`, `wsl-gpu-training-box`. The unmerged `origin/docs/current-architecture` branch
(`docs/architecture.md`, written 2026-10-25) was also read and checked against the code. Most of its claims
still hold, and it is the closest thing to a current architecture doc, but it is **not on main and not in
`mkdocs.yml`**.

---

## 1. Pipeline map, end to end

### 1.1 Data-flow diagram (what writes what, what reads what)

```
 full match video
   │
   ├─(A) scripts/split_broadcast_segments.py ──► classifier.get_current_best_guess_classifier()
   │       (also run by labeller /ingest/run)       yolo11n-cls model_saves/classifier/20251226-yolo11n-cls/0.987.pt
   │                                                (falls back to RandomClassifier if the load fails)
   ▼
 shot clips  *.mp4 + manifest JSON       (also: scripts/split_video.py makes fixed-length chunks, superseded)
   │   clip names diverge: arsenal_mancity_20250925_segNNN (raw) / arsenal_mancity_segNNN / mancity_segNNN
   │
   ├─(B) RT-DETR-L batch inference @1280, conf>=0.1   ◄── NOT IN REPO (any branch)
   │       ▼
   │     s3://…/footy_data/machine_labels/<raw_stem>.jsonl  (+ _manifest.json)
   │       row: {frame_index, bbox{x,y,w,h}, tags:[label,"rtdetr"], confidence, model_id, run_id, reviewed}
   │       │
   │       ├─► ball rows only ──► ~/footy/footy_data/ball_candidates/<raw_stem>.jsonl   ◄── filter step NOT IN REPO
   │       │      └─► scripts/embed_ball_candidates.py ──► ball_embeddings/<stem>.npz (frame_index, cand_index, vec)
   │       │
   │       └─(C) scripts/track_detections.py (LapTracker) ──► tracklets/<clip>.jsonl
   │                 └─► scripts/score_links.py (identity/linker + ResNet50) ──► link_proposals/   (NO CONSUMER)
   │
   ├─(D) Labeller server  labeller/server.py (FastAPI), routers: review, ball_check, ingest, identity_routes
   │       /labeller   Session + VitTrack propagation ──► ball_gt_marks/<stem>.jsonl  (whole-file atomic rewrite)
   │       /object_review  review.py ──────────────────► ball_gt_marks/<stem>.jsonl  (in-place rewrite/delete, not atomic)
   │       /ball_check  ball_check.py + candidates.py
   │            reads ball_gt_marks/*.jsonl (source=sidecar) or ball_candidates/*.jsonl (source=cand)
   │            verdicts ──► ball_gt_marks/ball_checks/<stem>.jsonl  (append-only, tombstone undo)   (NO CONSUMER)
   │            dragged box ──► ball_gt_marks/<stem>.jsonl  tags [label,"labeller"] (rewrite, or append)
   │       /identity, /identity/merge  reads tracklets/, writes identity_labels/ (append-only)  (NO DOWNSTREAM CONSUMER)
   │       /ingest  upload ──► split_broadcast_segments subprocess ──► clips dir   (page 500s: web/ingest.html missing)
   │
   ├─(E) Feature store  feature_store/ (DuckDB)
   │       ingest_gt.py      ball_gt_marks/*.jsonl (top level only) ──► detection rows, run_id gt_import_<prov>
   │       importers.py      Roboflow YOLO dataset ──► source=hand_label run_id roboflow_v<N>
   │                         labeller/SAM3 FrameDetections JSON ──► source=sam3
   │       ingest.py         FrameDetections ──► rows (only caller: scripts/detect_objects.py, SAM3 text prompts)
   │       NOT INGESTED: machine_labels (2.77M rows), tracklets, identity labels, ball_check verdicts
   │
   ├─(F) Training-set export  scripts/export_training_dataset.py
   │       detections_enriched WHERE canonical ──► YOLO dir (by-clip split, eval stems excluded)
   │       also UPDATEs detection.dataset_tag in the store
   │
   ├─(G) Training  scripts/train_object_detector.py (ultralytics YOLO class, Roboflow download or --local-dataset)
   │       defaults imgsz 640, device mps, dataset v3 ──► footy_scan_detection/<run>/weights/best.pt (W&B)
   │       scripts/train_classifier.py, scripts/optuna_sweep_object_detector.py
   │       RT-DETR-L training command for the production checkpoint: NOT IN REPO
   │
   ├─(H) Evaluation  ball_eval/ (dataset, metrics, runner)
   │       eval_data/clips/<stem>.jsonl (one row per frame, list bbox/center)  ◄── scripts/build_eval_sidecars.py
   │       scripts/run_bakeoff.py (ROI-YOLO, VitTrack) ──► docs/bakeoff_results.json
   │       scripts/analyse_ball_tracker.py
   │
   └─(I) Serving / output: none wired. footy-track entry point prints "Hello from footy-track!".
           output/exporters.py, projection/projector.py, input/streams.py are scaffolds that nothing calls.
```

### 1.2 Per stage

| Stage | Entry points | Key modules | Inputs | Outputs / formats | Status |
|---|---|---|---|---|---|
| Ingest / segmentation | `python -m footy_track.scripts.split_broadcast_segments`; labeller `/ingest/upload`, `/ingest/run` (`labeller/ingest.py`) | `classifier.py` (`UltralyticsClassifier`, `RandomClassifier`) | match video | broadcast-only MP4 runs + manifest JSON | Works. The classifier loads at import time (`classifier.py:148`), hardcodes `device="mps"` (`classifier.py:128`), and silently falls back to random labels (`classifier.py:30-33`). |
| Frame extraction | `python -m footy_track.scripts.extract_frames`, `split_video` | — | video | JPEG frames `<stem>_%06d`, optional EXIF | Works. The docs' console-script names do not exist (see §3). |
| Detection (batch) | **not in repo** | — | clips | `machine_labels/*.jsonl` on S3 | The production path is outside git. |
| Detection (in-repo) | `scripts/detect_objects.py` (SAM3 text prompts, optional `--store_path`); labeller `/autodetect`, `/review/yolo` | `detectors/ultralytics.py` (`CURRENT_BEST_DETECTOR_CHECKPOINT = model_saves/detector/rtdetr_1920_v11/best.pt`, `DEFAULT_IMGSZ=1280`; weights not in git) | frames / video frames | `FrameDetections` (pydantic, `schema.py`) | The labeller can run it interactively. Only the SAM3 script writes to the store. |
| Player tracking | `scripts/track_detections.py` (LapTracker over precomputed JSONL); `scripts/track_objects.py` (ultralytics ByteTrack/BoT-SORT, re-runs detection, writes via `trackers/writer.py`) | `trackers/lap.py`, `trackers/ultralytics.py`, `trackers/writer.py` | machine_labels JSONL / frames | tracklets JSONL (read by `/identity/tracklets`) / `tracks_meta.json` | LapTracker is the live path. Neither tracker writes to the feature store (`track_meta` has no writer). |
| Gap bridging | `scripts/score_links.py` | `identity/linker.py` | tracklets + clips | `~/footy/footy_data/link_proposals/` | Output is read by nothing. The team gate is inert (`linker.py:125` needs `team`, which no code sets). |
| Labelling: box labeller | `uvicorn footy_track.labeller.server:app`, `python -m footy_track.labeller [--debug]` | `server.py`, `session.py`, `video_utils.py` (`BackgroundLabeller` → `VitTrackVideoLabeller`), `ball_trackers/sot_vittrack.py`, `run_stream.py` | clip MP4 (`FOOTY_CLIPS_DIR`), sidecar | `ball_gt_marks/<stem>.jsonl`: `{frame_index, bbox{x,y,w,h}, center{x,y}, tags:[label, model]}`; skip markers `{bbox:null, tags:["no_ball"/"not_broadcast"]}` | Works mechanically. VitTrack fails on the ball (KB `footy-labeller-gotchas`). |
| Labelling: review | `/object_review` | `review.py` | all sidecars | same sidecars, rewritten in place | Works. OPEN-3 (vittrack reported as `labeller`) is still pinned. |
| Labelling: ball check | `/ball_check` | `ball_check.py`, `candidates.py`, `web/ball_check.html`, `scripts/embed_ball_candidates.py` | sidecars or `FOOTY_BALL_CANDIDATES_DIR`, `ball_embeddings/*.npz` | verdict log `ball_checks/<stem>.jsonl`: `{clip, frame_index, box_index, source, confidence, verdict, bbox, label, provenance, corrected_bbox, ts}`; corrected boxes go to the sidecar as `[label,"labeller"]` | New and working. Its verdicts reach nothing downstream. |
| Labelling: identity | `/identity`, `/identity/merge` | `identity_routes.py`, `identity/{store,labels,clusters,sampling}.py` | `tracklets/` (sibling of the GT dir) | `identity_labels/` (append-only) | Human side only. Nothing consumes the labels. |
| Feature store | `python -m footy_track.feature_store.ingest_gt --gt-dir --video-dir --db [--roboflow-dir]` | `schema.py` (DDL + `detections_enriched.canonical` view), `store.py`, `importers.py`, `ingest.py`, `difficulty.py` | sidecars, Roboflow dir | DuckDB; PK `(game_id, source, run_id, frame_index, detection_id)` (`schema.py:301`) | Implemented. Only GT sidecars and Roboflow are wired in. |
| Export | `python -m footy_track.scripts.export_training_dataset` | — | store | YOLO dataset + `data.yaml`; tags rows `dataset_tag=<tag>:<split>` | Works. See the leakage seams in §4. |
| Training | `scripts/train_object_detector.py`, `train_classifier.py`, `optuna_sweep_object_detector.py` | ultralytics | Roboflow or local YOLO dir | weights, W&B | The defaults are traps (640, mps). |
| Evaluation | `scripts/run_bakeoff.py`, `scripts/analyse_ball_tracker.py`, `scripts/build_eval_sidecars.py` | `ball_eval/` | `eval_data/clips/*.jsonl` + MP4 | `docs/bakeoff_results.json` | The only GT sidecar in `eval_data/clips` (`mancity_seg104.jsonl`) has **no MP4 next to it**. The bake-off used the ball-blind 640 checkpoint. |
| Serving / output | `footy-track` → `footy_track:main` (prints hello) | `output/exporters.py` (MatchExporter), `projection/`, `input/streams.py` | — | — | Not wired. |

### 1.3 The data formats that cross stage boundaries

There are **six ball and box label formats**, and no module owns them:

1. **Labeller sidecar** (`ball_gt_marks/<stem>.jsonl`): `bbox`/`center` as dicts, `tags=[label, provenance, *markers]`.
   It is written by `session._box_line` (`session.py:309-317`), `review_correct` (`review.py:335-344`), and
   `ball_check._append_gt` (`ball_check.py:713-720`). It is read by session, review, ball_check, `ingest_gt`,
   and `scripts/build_eval_sidecars.py`, each with its own parser.
2. **Sparse centre GT** (`scripts/label_ball_centers.py`): `center` as a `[x,y]` list, `tags=[]` or
   `["ball_not_visible"]`.
3. **Eval sidecar** (`eval_data/clips/<stem>.jsonl`, `ball_eval/dataset.py`): one row per frame, `bbox` and
   `center` as lists, descriptive tags.
4. **Machine labels / candidates**: `bbox` as a dict, plus `confidence`, `model_id`, `run_id`, `reviewed`,
   `tags=[label,"rtdetr"]`.
5. **Verdict log** (`ball_checks/<stem>.jsonl`).
6. **Tracklets JSONL** and **identity label** logs.

The parsers disagree on what they tolerate. `review._parse_jsonl_box` and `session._restore_box_line`
accept a list `bbox`. `ingest_gt` does `bbox["x"]` (`ingest_gt.py:210`), so a list bbox raises TypeError
and aborts the whole ingest. `ingest_gt` drops centre-only rows (`ingest_gt.py:184`).

---

## 2. Dead, duplicated, stale and hardcoded code

### 2.1 Modules nothing on a live path reaches

| Module | Evidence |
|---|---|
| `labeller/app.py` (Streamlit "SAM3 Video Labeller") + `labeller/_canvas_compat.py` + `scripts/run_labeller.py` | Only `run_labeller.py` imports `app.main`. Streamlit and streamlit-drawable-canvas are still **runtime** dependencies in `pyproject.toml`. |
| `Sam3VideoLabeller`, `_default_model_uri` in `labeller/video_utils.py` (iCloud path at `video_utils.py:121`) | Reached only from `app.py`. |
| `labeller/motion_tracker.py::MotionGuidedTracker` | Only its `Box`/`Detection` types are imported (`video_utils.py:42`). |
| `ball_trackers/roi_yolo.py`, `scripts/run_bakeoff.py` | Bake-off only. Hardcoded macOS paths (`run_bakeoff.py:26,28`). |
| `projection/`, `output/`, `input/streams.py`, `metaflow/docker/Dockerfile` | Only package `__init__` re-exports and tests. |
| `feature_store/difficulty.py` | Re-exported from `feature_store/__init__.py:9`. No script calls `flag_for_review`. |
| `identity/stable_id.py` | Re-exported from `identity/__init__.py:27`. Never called, so `detection_id` stays positional (`importers.py:228`, `ingest_gt.py:199`, `ingest.py:60`). |
| `footy_track/logging.py` | Imported by nothing. Scripts repeat the same `basicConfig(RichHandler)` inline (e.g. `detect_objects.py:31-35`). |
| `footy_track/__init__.py::main` (the `footy-track` console script) | Prints "Hello from footy-track!". |
| `labelling.py`, `scripts/upload_*_frames.py`, `scripts/create_roboflow_v11.py`, `scripts/classify_frames.py`, `scripts/proto_sam3_points.py` | One-off or notebook-only. |
| `trackers/ultralytics.py`, `trackers/writer.py`, `scripts/track_objects.py` | Superseded by `track_detections.py` for identity work. Re-run detection. Never measured against GT. |
| `/ingest` route (`server.py:160-162`) | Serves `web/ingest.html`, which does not exist (OPEN-1). |
| `TORCHINDUCTOR_CACHE_DIR` set in `server.py:22-25` | Nothing on the server path calls torch.compile (KB `footy-labeller-gotchas`). |

### 2.2 Duplication

- **Box identity numbering is implemented three times.** `review._read_all_boxes` (`review.py:132-158`),
  `review._iter_frame_box_lines` (`review.py:161-183`) and `ball_check._sidecar_ball_boxes`
  (`ball_check.py:548-583`) each re-derive "box_index = ordinal among the frame's box lines". The third
  carries a comment saying it must "stay in step".
- **Ball label sets are defined in four places:** `labeller/constants.BALL_LABELS`, `candidates.BALL_TAGS`
  (`candidates.py:44`), `export_training_dataset.BALL_LABELS` (`:74`), and `constants.py` tags.
- **Provenance tag sets are defined three times:** `constants.PROVENANCE_TAGS`, `review._PROV_TAGS`
  (deliberately incomplete, OPEN-3), and `ingest_gt._PROVENANCE_TAGS` (`ingest_gt.py:72`).
  `human_checked` is known **only** to `ingest_gt` (`ingest_gt.py:77`).
- **Video lookup by stem is implemented three times:** `review._find_video`, `ingest_gt._find_video`, and
  `export_training_dataset._resolve_video_path` (with its own alias table, `:190-212`).
- **The crop window is implemented twice:** review (`review.py:290-295`, pad multiple of the box) and
  ball_check (`ball_check.py:371-390`, fixed fraction of the frame).
- **There are two trackers and two detection entry points**, as listed in 2.1.

### 2.3 Hardcoded or stale paths

| Location | Path |
|---|---|
| `.dvc/config:4` | `/Users/georgebarnett/code/dvc-remote` (macOS) |
| `scripts/run_bakeoff.py:26,28` | `/Users/georgebarnett/code/footy/...`, iCloud GT dir |
| `labeller/video_utils.py:121` | iCloud SAM3 weights path |
| `labeller/server.py:97-106` | Falls back to the iCloud GT dir when `~/footy/footy_data/ball_gt_marks` is missing |
| `feature_store/ingest_gt.py:6-7` (docstring) | `/mnt/storage/...` and a **Gas Town worktree** `/home/george/code/footy_track/refinery/rig/eval_data/clips` |
| `scripts/export_training_dataset.py:179-212,509` | `/mnt/storage/footy_data` plus a hand-written stem→file alias table |
| `scripts/embed_ball_candidates.py:18-20`, `scripts/score_links.py:167-171` | `/mnt/storage/...`, `~/footy/footy_data/...` defaults |
| `eval_data/clips/*.mp4` | 40 absolute symlinks into `/mnt/storage/footy_data` |
| `classifier.py:128` | `device="mps"` |
| `scripts/train_object_detector.py:56,59,165,182` | `imgsz=640`, `device="mps"` |
| `.playwright-mcp/*` | 29 committed browser-log files |

---

## 3. Code against docs

### 3.1 Docs that are not in `mkdocs.yml` nav

`mkdocs.yml` publishes only `index, system_design, system_overview, development, pipeline_architecture,
timings, data_formats, training, training/notable_runs, agent_guidelines`. Most of those are April-era
docs. **Not in the nav:**

- `docs/architecture.md`: **exists only on `origin/docs/current-architecture`**. It is not on main and has
  no nav entry.
- `docs/design/pipeline_overview.md` (2026-08-31) and `docs/design/player_reid.md` (2026-08-30). These are
  the two most current design docs.
- `docs/design/feature_store.md` (its status line says "IMPLEMENTED (v1)").
- `docs/design/{autolabelling_architecture_review, bidirectional_tracking_research, input_live_stream,
  labeller_review_2026-06, output, player_tracking_format, projection, sam3_ball_labelling_gpu, tracking}.md`
- `docs/{approach, ball_tracker_failure_analysis, doc_review, interfaces, labeller_requirements,
  labeller_timeline_design, labeller_web_app_design, pipelines, problem}.md`, `docs/labeller_ui.html`,
  `docs/bakeoff_results.json`.
- `src/footy_track/labeller/README.md` is the canonical labeller spec, with 101 LAB requirements. It sits
  outside `docs_dir`, so it is **never published**.

### 3.2 Doc claims that no longer match the code

| # | Doc (file:line) | Claim | Code (file:line) | Reality |
|---|---|---|---|---|
| D1 | `labeller/README.md:664-668` and `ball_check.py:11-15` (module docstring) | Ball Check "writes **only** to its own append-only log", "Nothing is written to the GT sidecars" | `ball_check.py:651-729` (`/ball_check/save_box`) | A dragged box is rewritten into, or appended to, `ball_gt_marks/<clip>.jsonl` as `[label,"labeller"]`. The module docstring contradicts its own endpoint. |
| D2 | `labeller/README.md:672` | Verdict record = `{clip, frame_index, box_index, verdict, bbox, label, provenance, ts}` | `ball_check.py:750-766` | Also `source`, `confidence`, `corrected_bbox`. |
| D3 | `labeller/README.md:698-699` | verdict must be `ball` / `not_ball` / `unsure` | `ball_check.py:53` | Five values: `ball, not_ball, box_off, corrected, unsure`. |
| D4 | `labeller/README.md:715-717` | `precision = ball/(ball+not_ball)` | `ball_check.py:341-351` | `precision = (ball+box_off+corrected)/decided`, plus `clean_precision`. |
| D5 | `labeller/README.md:720-722` | Crop window is `1 + 2*pad` box sides | `ball_check.py:371-390` | A fixed fraction of frame width (`_WINDOW_FRAC=0.125 × pad/6`), floored at 160 px, grown for oversized boxes. |
| D6 | `labeller/README.md:729-730` | LRU key `(clip, frame, box, pad)` | `ball_check.py:479-487` | `(src, clip, frame, box, pad, reticle, at_frame)`. |
| D7 | `labeller/README.md:762` | `focus` 0.7 | `candidates.py:63` | `DEFAULT_FOCUS = 0.98`. Also `UNIFORM_SHARE=0.25` (`candidates.py:55`). |
| D8 | `labeller/README.md` §11-12 | (not mentioned) | `ball_check.py:393, 586, 651`; `candidates.py:92-189` | The `/ball_check/crop_meta`, `/ball_check/step` and `/ball_check/save_box` endpoints, event grouping (`GROUP_FRAME_GAP`, `GROUP_DIST`), LSH appearance buckets from `ball_embeddings/*.npz`, and `scripts/embed_ball_candidates.py` are undocumented. The traceability table (`:860`) still says 12 requirements. |
| D9 | `labeller/README.md:443, 839-840` | Handback is "any detection below the 0.5 threshold"; "0.5 on main … 0.3 unresolved" | `video_utils.py:51,55` | Ball-only 0.30, non-ball 0.0 (commit `489d444`). |
| D10 | `labeller/README.md:48, 101-108` (LAB-001/101/103) | Sidecar `tags = [label, model]`; the round trip preserves everything the UI shows | `session.py:156-161, 309-317`; `ingest_gt.py:73-77` | A third tag, `human_checked`, exists on 15,377 lines and is TIER 2 for `ingest_gt`. The spec never mentions it, and **the session drops it on flush** (seam S1). |
| D11 | `labeller/README.md:865` | "138 tests in `tests/labeller/`" | `tests/labeller/*.py` | 177 test functions. |
| D12 | `ingest_gt.py:13, 19-25` (docstring) | `tags: [<class>, <provenance>]`; vittrack is `reviewed=False` | `ingest_gt.py:126-143, 191-195` | A 3-tag form exists, and `human_checked` makes rows `reviewed=True`. The docstring was not updated with the fix. |
| D13 | `ingest_gt.py:6-7` (docstring) | `--video-dir /home/george/code/footy_track/refinery/rig/...` | — | A Gas Town worktree path in a published docstring. `--gt-dir` is `required=True` (`ingest_gt.py:301`). KB `footy-data-plane-reconciliation` says it "defaults to the Mac mini path", which is wrong. |
| D14 | `feature_store/schema.py:10`, `docs/design/feature_store.md:117` | `game` = one match, e.g. `arsenal_mancity` | `ingest_gt.py:202, 227` | `game_id` = **clip stem** for GT imports, and the Roboflow importer uses the frame-filename stem. |
| D15 | `docs/design/feature_store.md:85-93` | Parquet is the source of truth on disk, partitioned `game_id=…` | `store.py:270-326` | DuckDB is primary. `export_parquet` and `from_parquet` exist, but no workflow uses them. |
| D16 | `README.md:62-81`, `docs/pipeline_architecture.md:7-35`, `docs/system_overview.md:16-29`, `docs/development.md:83-86`, `docs/interfaces.md:7,142`, `docs/approach.md:11-26` | InputConsumer → Processor → OutputProducer pipeline | — | None of these classes exist. The live pipeline is a set of scripts run by hand plus the labeller server. |
| D17 | `docs/development.md:67` | `src/footy_track/main_entrypoint.py` | — | Does not exist. |
| D18 | `README.md:108-115`, `docs/index.md:31` | `footy-track` runs the pipeline / entry point | `__init__.py:1-2` | Prints hello. |
| D19 | `docs/data_formats.md:35`; `scripts/README.md:21-60` | `uv run footy-track-extract-frames`, `footy-track-split-video` | `pyproject.toml:44` | The only console script is `footy-track`. `doc_review.md:157` flagged this in April and it is still unfixed. |
| D20 | `docs/data_formats.md:11-20` | Data lives under `data/<match>/original_video`, `full_video_frames` | `server.py:92-114`, KB `footy-data-plane-reconciliation` | The live data plane is `~/footy/footy_data` or `/mnt/storage/footy_data` (`ball_gt_marks`, `ball_candidates`, `ball_embeddings`, `tracklets`, `identity_labels`, `machine_labels`), configured by env vars. `data_formats.md` documents none of it. |
| D21 | `docs/data_formats.md:83-86` | 7 classes, "current default version 3" | KB `footy-ball-detection-findings` | v11 is the dataset in use. The `ball` class is dead (0 instances). The trainer default is still 3 (`train_object_detector.py:145`). |
| D22 | `docs/training.md:60-67` (args table), `:77-78` | Lists 4 args; run name has no imgsz | `train_object_detector.py:100, 160-200` | Undocumented: `--imgsz` (default **640**, which is ball-blind), `--device mps`, `--batch`, `--no-amp`, `--local-dataset`, `--workers`, `--cache`. The run name now appends `_imgsz=`. |
| D23 | `docs/design/pipeline_overview.md:41` | LapTracker ~540 fps | `track_detections.py:5` | ~650 fps. |
| D24 | `docs/design/pipeline_overview.md:45, 53` | kit-colour → `team_id`; `player_id` per detection → feature store | `identity/linker.py:59,125`; no `upsert_track_meta` caller | No module computes `team_id`. Nothing writes tracks or identity into the store. |
| D25 | `eval_data/README.md` | Eval clips = MP4 + JSONL pairs | `eval_data/clips/` | `mancity_seg104.jsonl` has no `.mp4`. Every other MP4 has no `.jsonl`. |
| D26 | `candidates.py:3-9`, `server.py:107-110`, README §12 | Candidates are "pulled from s3 machine_labels, ball-class rows only" | — | No code in the repo pulls or filters them. The step is manual. |

### 3.3 In the code but not in any doc

- `/ball_check` save/step/crop_meta, the candidate event grouping, LSH buckets, `embed_ball_candidates.py` (D8).
- Environment variables: `FOOTY_CLIPS_DIR`, `FOOTY_GT_MARKS_DIR`, `FOOTY_BALL_CANDIDATES_DIR`,
  `FOOTY_MAX_CLIPS`, `FOOTY_DEBUG`. The labeller README mentions some. No published doc does.
- `python -m footy_track.labeller --debug` and its flags. They appear in docstrings and the README only.
- The `identity/` subsystem and `/identity` pages. `player_reid.md` is not in the nav.
- `track_detections.py`, `score_links.py`, and their `tracklets/` and `link_proposals/` directories.
- The `human_checked` tag and the TIER 1/2/3 mapping. These are in KB `footy-data-plane-reconciliation`
  and `ingest_gt.py:73-77`, but in no spec.
- The `detections_enriched.canonical` arbitration rule. It is fully explained in the `schema.py:19-71`
  docstring and nowhere in the docs.
- `export_training_dataset.py` writes back into the store (`dataset_tag`, `:431-436`).

---

## 4. Seams and risks

Severity: **H** = silently corrupts training data or GT. **M** = wrong results or lost work in a
foreseeable workflow. **L** = friction.

| # | Sev | Seam | Evidence |
|---|---|---|---|
| S1 | **H** | **The labeller session drops `human_checked`.** `_restore_box_line` keeps only label + provenance, and `_box_line` writes `[label, model]`. Opening any clip in `/labeller` and making one edit rewrites the whole sidecar, so every TIER-2 vittrack box in that clip silently becomes TIER 3 (`reviewed=False`) on the next ingest. This is the same tag-semantics class as the `human_checked` bug, in the other direction. No test covers it. | `session.py:156-161, 309-317`; `ingest_gt.py:191-195` |
| S2 | **H** | **Ball Check verdicts are not consumed, so a `not_ball` verdict still trains as a ball.** A human `not_ball` on a sidecar vittrack box leaves the box in the sidecar. `ingest_gt` imports it, and `export_training_dataset` exports it as a positive `ball` label (the export filters on `canonical` only, not on `reviewed` or verdicts). A `box_off` box, which has known-bad geometry, also trains as-is. A `ball` verdict on a sidecar box is exactly what `human_checked` means, but it is never written as `human_checked` and never reaches the store. | `ball_check.py:737-768`; `ingest_gt.py:279` (`glob("*.jsonl")`, top level only, so `ball_checks/` is skipped); `export_training_dataset.py:91-108` |
| S3 | **H** | **Clip identity is not canonical, so the eval-exclusion and by-clip-split guards can be bypassed.** The same footage has up to three stems: `mancity_seg050` and `arsenal_mancity_seg050` both symlink to `arsenal_mancity_20250925_seg050.mp4`, and candidates use the raw stem. `save_box` on a candidate appends GT to `ball_gt_marks/<raw_stem>.jsonl`, a *new* sidecar beside any existing human one. In the store `game_id` = stem, the export excludes eval clips by exact stem (`mancity_seg104`), and the split is by `game_id`. The eval clip's footage can therefore enter training as `arsenal_mancity_20250925_seg104`, and one clip can sit in both train and val. The split's `assert`s cannot see this. | `eval_data/clips/*.mp4` symlink targets; `ball_check.py:709-728`; `export_training_dataset.py:78-80, 362-368, 385-387` |
| S4 | **H** | **Re-ingesting leaves ghost rows.** `_upsert` is INSERT … ON CONFLICT and never deletes. After `review_correct` turns a vittrack box into `labeller`, the old `(source=vittrack, detection_id=k)` row stays in the store. All `gt_import_*` runs for a clip form one canonical run_group (`schema.py:40-48`), so the ghost **stays canonical** and is exported next to the corrected box. `review_delete` and session edits also shift positional `detection_id`s, so rows overwrite one another. | `store.py:197-220`; `ingest_gt.py:197-199`; `review.py:312-370` |
| S5 | M | **Undoing a `corrected` card does not undo the GT write.** Undo appends a verdict tombstone only. The box written by `save_box` stays in the sidecar, and re-correcting appends a second `labeller` box for the same ball. | `ball_check.html:351-365`; `ball_check.py:771-793` |
| S6 | M | **Concurrent writers clobber one another.** The session flush rewrites the whole sidecar from its in-memory timeline (`session.py:252-301`). review and ball_check rewrite or append to the same file. If a clip is open in `/labeller` while ball_check or review writes to it, the next debounced flush erases those writes. review's rewrites are not atomic (`review.py:344, 367`), unlike the session's. | as cited |
| S7 | M | **Positional box identity.** Verdicts key on `(source, clip, frame, box_index)`. Any `review_delete`, or a session flush that reorders boxes, repoints old verdicts at different boxes. Candidates are positional within a frame of an S3 file, so a re-pull with different row order does the same. `identity/stable_id.py` exists to fix this and is unused. | `ball_check.py:135-149`; `candidates.py:17-20`; `identity/stable_id.py` |
| S8 | M | **The `reviewed` column has no consumer.** The `human_checked` fix sets `reviewed=True`, but `export_training_dataset` never filters on it, and `--sources` is the only tier control. TIER-3 machine rows go to training by default. | `export_training_dataset.py:32-34, 91-128` |
| S9 | M | **`run_id` cannot separate machines.** The KB says each machine's copy should be ingested under a distinct `run_id`. `ingest_gt` hardcodes `gt_import_<prov>` (no `--run-id` flag), and the canonical view merges all `gt_import_*` runs for a game. Ingesting two machines' sidecars into one DB silently overwrites by positional id. | `ingest_gt.py:197, 253`; `schema.py:40-48`; KB `footy-data-plane-reconciliation` |
| S10 | M | **OPEN-3 split semantics.** `provenance` reports vittrack as `labeller` in the review queue (sorting, badges). Ball Check uses `provenance_tag`. Any new consumer of `_read_all_boxes` inherits the trap. | `review.py:54-57, 104-108` |
| S11 | M | **Producers of the inputs are not in the repo:** RT-DETR batch inference, the machine_labels → ball_candidates filter, the RT-DETR training command, and the clip-alias symlink farm (`ballcheck_clips`). The production checkpoint and the Ball Check corpus cannot be rebuilt from git. | §1 diagram; KB `footy-labeller-gotchas` |
| S12 | M | **No eval loop for the production detector.** The ball_eval harness's only GT clip has no video in `eval_data/clips`, and the bake-off ran the 640 checkpoint. The .221 vs .665 dispute (KB `footy-ball-detection-findings`) cannot be settled from any committed script. | `eval_data/clips/`; `docs/bakeoff_results.json` |
| S13 | L | **Label-fallback disagreement.** On an unknown tag, session defaults to `in_play_ball` (`session.py:156`), review to `player` (`review.py:102`), and `ingest_gt` keeps the raw tag as the class (`ingest_gt.py:141`). Skip-marker precedence also differs: the session flush drops boxes on a `no_ball` frame (`session.py:272-277`), while `ingest_gt` imports them. | as cited |
| S14 | L | **Ball Check stats mix namespaces.** `remaining = pool.events − len(judged)` subtracts sidecar and candidate verdicts (rows) from candidate events (`ball_check.py:289`). | |
| S15 | L | **Verdict POST failures are invisible.** The page treats any HTTP response as success (only a network exception restores the card), so `{ok:false}` is dropped silently. | `ball_check.html:289-308` |
| S16 | L | **`--legacy-all-reviewed`** still exists and stamps everything reviewed. It is harmless while `reviewed` is unused (S8), and wrong once it is used. | `ingest_gt.py:314-322` |

### 4.1 Labelling output that is not consumed

| Output | Producer | Consumer |
|---|---|---|
| `ball_checks/<clip>.jsonl` verdicts (precision per band, hard negatives, confirmations) | Ball Check | **none** (only Ball Check's own sampler and stats) |
| `identity_labels/` (purity, jersey, pair merges) | `/identity` | `/identity/clusters` view only |
| `link_proposals/` | `score_links.py` | **none** (the merge page picks pairs itself) |
| `tracklets/` | `track_detections.py` | identity pages only. Not in the store (`track_meta` has no writer). |
| `reviewed` column | `ingest_gt` | **none** |
| `difficulty.flag_for_review` | — | never called |
| `machine_labels` 2.77M rows | outside the repo | the Ball Check candidate pool only |

### 4.2 Test coverage gaps at stage boundaries

- **No CI runs the tests.** `.github/workflows/docs.yml` only deploys docs. `addopts = "-q -n 2"` needs
  `pytest-xdist`.
- **No test for sidecar → store.** No test writes a sidecar through the labeller endpoints and then runs
  `ingest_gt` on it. `test_roundtrip_fidelity.py` covers Roboflow and `import_labeller_json`
  (FrameDetections JSON), not sidecars.
- **No session round-trip test for `human_checked`** or for any third tag (S1).
- **No test that a verdict affects the store or the export** (S2). By design, the only ball_check test that
  touches the store side asserts the sidecar is untouched.
- **No test for clip-alias leakage in the export** (S3), or for **re-ingest after a correction or delete**
  (S4). `test_ingest_is_idempotent` only re-ingests an unchanged file.
- **No test that undo of `corrected` removes the GT box** (S5), and no test of **concurrent session flush vs
  review or ball_check writes** (S6).
- **No test for list-`bbox` sidecars in `ingest_gt`** (format tolerance).
- **The frontends are untested:** `ball_check.html` save/undo flow, LAB-8xx/9xx. These are 22 requirements
  marked UNTESTED-FRONTEND.
- **No test ties track_detections output to the identity routes' expected input.** They are tested
  separately.

---

## 5. Recommendations, in priority order

| # | Recommendation | Fixes | Size |
|---|---|---|---|
| 1 | Make the session preserve unknown and marker tags on load and flush (carry `extra_tags` on the box, write `[label, model, *markers]`). Add a round-trip test. Add `human_checked` to `labeller/constants.py` and LAB-101/103. | S1, D10 | S |
| 2 | Write a `ball_checks` → feature-store importer. `ball` on a sidecar or candidate box → `reviewed=True` (TIER 2). `not_ball` → exclude the box from export and record it as a hard negative (frame kept, box dropped). `box_off` → exclude geometry. `corrected` is already TIER 1. Then make `export_training_dataset` default to `reviewed OR source='hand_label'` with an explicit opt-in for TIER 3. | S2, S8, §4.1 | M |
| 3 | Canonicalise clip identity in one place: a `clip_id(stem)` function or alias table shared by `review._find_video`, `ball_check.save_box`, `ingest_gt` (`game_id`), and the export's eval exclusion. Key on the raw stem `arsenal_mancity_20250925_segNNN`. Exclude eval clips by canonical id. Add a test. | S3, S11 (partly) | M |
| 4 | Make `ingest_gt` replace rather than upsert. Per `(game_id, run_group)`, delete existing `gt_import_*` rows before inserting, or switch `detection_id` to `identity/stable_id.py` content hashes. Add `--run-id-suffix` so per-machine ingests stay distinct, and fix the canonical view to group on it. Add re-ingest-after-correction tests. | S4, S7, S9 | M |
| 5 | Make undo of `corrected` revert the GT write: return the written line identity from `save_box` and remove it on undo, or route corrections through the verdict log only and let the importer apply them. | S5 | S |
| 6 | Give the sidecar a single writer with a file lock and atomic replace (reuse the session's tmp+`os.replace`). Have the session re-read the file before flushing, or refuse to flush when the mtime changed since load. | S6 | M |
| 7 | Fix the docs to match the code: refresh `labeller/README.md` §11-12 and LAB-706 (D1-D9, D11); fix `ingest_gt` docstrings (D12-D13); merge `docs/architecture.md`, then add it, `design/pipeline_overview.md`, `design/player_reid.md`, `design/feature_store.md` and a rendered copy or link of the labeller README to the `mkdocs.yml` nav; mark the April docs (D16-D18) as historical or delete them; replace the console-script commands (D19); document the data-plane layout and env vars in `data_formats.md` (D20-D21); document the full trainer arguments (D22). | §3 | M |
| 8 | Commit the missing producers: the RT-DETR batch-inference script, the machine_labels → ball_candidates filter, the RT-DETR training command, and a clip-alias symlink generator. Repoint `.dvc/config` to S3. | S11, S12 | M |
| 9 | Fix the trainer defaults: `--imgsz 1280`, `--device` auto-detect, `--dataset-version 11`. Remove `device="mps"` and the import-time model load from `classifier.py`, and make the `RandomClassifier` fallback opt-in. Correct the `eb67f6d` merge message or record the workers choice before pushing. | §2.3, D21-D22 | S |
| 10 | Add a CI workflow (`uv sync && uv run pytest -n auto`, with the ffmpeg-heavy conftest marked or skipped), plus the boundary tests listed in 4.2. | §4.2 | S to M |
| 11 | Restore an eval loop: put the eval clip's video (or an alias) next to `mancity_seg104.jsonl`, and add a committed script that measures `rtdetr_1920_v11` at 1280 on a named split with instance counts. | S12 | M |
| 12 | Delete or quarantine dead code: `labeller/app.py`, `_canvas_compat.py`, `run_labeller.py`, `Sam3VideoLabeller`/`_default_model_uri`, and the streamlit deps; `logging.py`; `.playwright-mcp/`; the `/ingest` page route (or add the page). Move the `projection/`, `output/` and `input/` scaffolds under an `experimental/` package, or note them as unwired. Wire `difficulty.py` or remove it. | §2.1 | S to M |
| 13 | Consolidate the duplicated helpers: one `sidecar.py` (parse, numbering, write, video lookup, label and provenance sets) used by session, review, ball_check, `ingest_gt` and `build_eval_sidecars`. This also removes the S13 disagreements and the OPEN-3 trap. | §2.2, S10, S13 | L |
| 14 | Feed identity outputs downstream: write tracklets to `track_meta` and `detection.track_id`, have `/identity/merge` read `link_proposals/`, and implement team assignment so the linker's team gate does something. | §4.1, D24 | L |

---

## 6. KB pages checked against the repo

| KB claim | Verdict |
|---|---|
| The feature store is keyed by `run_id`: PK `(game_id, source, run_id, frame_index, detection_id)` (`footy-data-plane-reconciliation`) | **True** (`schema.py:301`). But "ingest each machine's copy under a different run_id" cannot be done through `ingest_gt`'s CLI (S9). |
| "`ingest_gt.py` … Its default `--gt-dir` is the Mac mini path" | **Outdated.** `--gt-dir` is required. The Mac mini path appears only in the docstring example. |
| `ball_gt_marks` lives in the data plane, not the repo | **True.** The server resolves it from `FOOTY_GT_MARKS_DIR` or `~/footy/footy_data/ball_gt_marks` (`server.py:97-106`). |
| Verdicts append to `ball_gt_marks/ball_checks/<clip>.jsonl`, with undo as a `verdict:null` tombstone | **True** (`ball_check.py:101-158, 771-793`). |
| Corrected boxes go into the clip's ball-label file tagged `["in_play_ball","labeller"]` | **True, with a caveat.** The label is the card's ball label (it can be `out_of_play_ball`), and for a sidecar box the line is rewritten in place via `review_correct` (`ball_check.py:676-729`). KB `footy-labeller-gotchas` (09-23 section) still says Ball Check "never rewrites the GT sidecars". That was true when written and is now outdated. |
| Nothing feeds verdicts into the feature store | **True.** `ingest_gt` globs the top level only (`ingest_gt.py:279`), and nothing else reads `ball_checks/`. |
| The `human_checked` fix: class is the first non-marker tag; a checked box is `reviewed=True` with its source kept | **True** (`ingest_gt.py:126-143, 188-195`, tests at `test_ingest_gt.py:90-125`). Missing from the KB: the labeller session strips the tag again (S1). |
| `_VITTRACK_HANDBACK_SCORE` ball-only 0.30 is committed | **True** (`video_utils.py:51,55`). The labeller README still says 0.5 (D9). |
| WSL `workers=0` rule | The local merge `eb67f6d` defaults to `workers=8`, but its commit message says 0. The KB notes the box is now bare-metal Ubuntu, so the rule may no longer apply there. |
