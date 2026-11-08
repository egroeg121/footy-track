# TODO list

Prioritised from the [2026-11 review](reviews/2026-11/index.md) of `main` @ 26d2506 (2026-11-08). This file is the only tracker: Gas Town is paused, so there are no beads. Tick the box and update the status when an item lands.

Priority: P0 silent data corruption, P1 blocks reliable training or eval, P2 docs and gaps, P3 cleanup. Effort: S under half a day, M a day or two, L longer.

## P0: silent data corruption (fix before any more labelling or training)

- [ ] **T-01** (P0, S) Labeller session drops `human_checked` on load and flush, which downgrades tier 2 to tier 3 for every box in the clip. Add a regression test. Ref: `src/footy_track/labeller/session.py:156-161`, `309-317` (`_box_line` writes only `label` and `model`). Status: open.
- [ ] **T-02** (P0, M) Ball Check verdicts reach nothing. `ingest_gt` globs only top-level `*.jsonl` so `ball_checks/` is skipped, and the export keeps `not_ball` boxes as positive balls. Write a verdicts importer and make the export filter on review status. Ref: `feature_store/ingest_gt.py:279`, `scripts/export_training_dataset.py:91-128`, [review](reviews/2026-11/architecture-review.md). Status: open.
- [ ] **T-03** (P0, M) One canonical clip id shared by the labeller, `ingest_gt` and the eval exclusion and split. Each clip has three aliases today (for example `mancity_seg050`, `arsenal_mancity_seg050`, `arsenal_mancity_20250925_seg050`), so eval footage can leak into training. Ref: `labeller/ball_check.py:709-728`, `scripts/export_training_dataset.py:78-80`, `362-368`, [dataset inventory](reviews/2026-11/dataset-inventory.md). Status: open.
- [ ] **T-04** (P0, M) `ingest_gt` must replace the clip's rows rather than upsert, otherwise an old vittrack row stays canonical next to the corrected one (ghost rows, both exported). Ref: `feature_store/store.py:197-220`, `feature_store/schema.py:40-48`. Status: open.

## P1

- [ ] **T-05** (P1, M) Undoing a corrected Ball Check card must revert its box, and the sidecar needs one locked atomic writer (a session flush can overwrite Ball Check writes). Ref: `labeller/session.py:252-301`, `labeller/web/ball_check.html:351-365`. Status: open.
- [ ] **T-06** (P1, S) Trainer defaults: `--imgsz 640` is ball-blind and `--device` defaults to `mps`. Also `--workers` defaults to 8 while merge commit 26d2506 says "default 0 under WSL": reconcile the message and the code, and document `--workers` and `--cache` in `docs/training.md`. Ref: `scripts/train_object_detector.py` (`imgsz: int = 640`, argparse `--device`, `--workers`, `--cache`). Status: open.
- [ ] **T-07** (P1, M) Refresh the feature store. The newest copy is stale (4 Aug) and local-only, and lacks the Sep corrections, new label files, machine labels and Roboflow v11. `game.source_video_uri` and `match_date` are NULL. Ref: [dataset inventory](reviews/2026-11/dataset-inventory.md). Status: open. Do after T-02 to T-04.
- [ ] **T-08** (P1, M) Merge the test-speed branch `chore/faster-tests` once it is pushed (not yet on origin when this was written). SAM3 tests fail without weights and take about 150 s, nothing is marked `slow`, and no CI runs tests. Then add CI for tests. Status: waiting on the branch.
- [ ] **T-09** (P1, M) Commit the missing producers: RT-DETR batch inference, the machine_labels to ball_candidates filter, the RT-DETR training command, and the clip alias symlinks (`ballcheck_clips/`, 199 links). Ref: [review section 1](reviews/2026-11/architecture-review.md). Status: open.
- [ ] **T-10** (P1, M) Decide where the best-pipeline eval numbers live. The repo has no eval results behind the KB's model comparison numbers, only a disputed 0.221 vs 0.665 (`docs/design/pipeline_overview.md`, `src/footy_track/detectors/ultralytics.py:37`). Re-measure and record. Status: open.
- [ ] **T-10b** (P1) Best-pipeline research write-up: pending, not included here. Status: pending.

## P2

- [ ] **T-11** (P2, M) Docs refresh. README labeller sections 11-12 (verdicts, precision formula, crop window, focus 0.7 vs 0.98, handback 0.5 vs 0.30, test count 138 vs 177, undocumented step/save_box/crop_meta endpoints, and the claim that Ball Check writes nothing to the sidecars); `pipeline_architecture.md` and `system_overview.md` describe classes that do not exist; `docs/data_formats.md:35` and `scripts/README.md` name console scripts that do not exist (the only one is `footy-track`, `pyproject.toml:44`); the feature-store docs say `game_id` is a match when it is a clip stem. Ref: [review](reviews/2026-11/architecture-review.md) D1-D26. Status: open.
- [ ] **T-12** (P2, S) Publish current docs in `mkdocs.yml`. Done on this branch: `docs/architecture.md` (from `origin/docs/current-architecture`), `pipeline_overview.md`, `player_reid.md`, `feature_store.md` and the `reviews/` pages are in the nav, and the Mermaid fence is enabled. Remaining: check the rendered Mermaid in a browser. Status: in review on `docs/review-2026-11`.
- [ ] **T-13** (P2, L) Label gaps. Norwich (4K, about 7k s) and Bournemouth 2nd half have footage but no ground truth; machine labels cover only the 162 Man City clips; GT covers 45 of about 1,700 broadcast segments. Ref: [dataset inventory](reviews/2026-11/dataset-inventory.md). Status: open.
- [ ] **T-14** (P2, S) One shared parser for the six label formats; `ingest_gt.py:210` crashes on a list-form bbox. Status: open.
- [ ] **T-15** (P2, S) Knowledge base fixes (outside this repo). See [kb-vs-reality](reviews/2026-11/kb-vs-reality.md). The user has to approve KB edits. Status: open.

## P3

- [ ] **T-16** (P3, S) Remove or archive dead code: Streamlit labeller (`labeller/app.py`, `labeller/_canvas_compat.py`, `scripts/run_labeller.py`, streamlit deps in `pyproject.toml`), `Sam3VideoLabeller` and the iCloud path in `labeller/video_utils.py:121`, `labeller/motion_tracker.py`, `footy_track/logging.py`, unused scaffolds. Status: open.
- [ ] **T-17** (P3, S) Serving and output scaffolds (`output/`, `projection/`, `input/streams.py`, the hello-world `footy_track:main`) are not wired. Decide to build or delete. Status: open.
- [ ] **T-18** (P3, S) `/ingest` returns 500 because `labeller/web/ingest.html` is missing (`labeller/server.py:160-162`). Status: open.
- [ ] **T-19** (P3, S) Hardcoded macOS paths: `scripts/run_bakeoff.py:26,28`, `classifier.py:128` `device="mps"`. Status: open.

## Environment notes

- The Mac mini has no AVX2: duckdb needs `==1.3.2` and polars needs `polars-lts-cpu` (venv only, do not change `pyproject.toml` or `uv.lock`).
- Open the feature store with `duckdb.connect(path, read_only=True)` when inspecting.
- The SAM3 weights are not in git; tests that need them fail without them (see T-08).

## Done this session

- `feat/ball-check-mobile` merged (main 658d63d to 5d4fc90).
- `fix/train-workers-cache` merged (26d2506).
