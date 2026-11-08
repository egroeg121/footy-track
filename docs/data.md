# Datasets and labels

_State as of 2026-11-08, checked on the build machine against `main` @ 26d2506. Counts will drift; re-run the checks before relying on them._ For how the data moves through the system see [Architecture](architecture.md); for what is missing see the [TODO list](todo.md).

## Where things live

| Place | Contents |
|---|---|
| The data root (a local disk, 45 GB, 18,716 files) | The real data root on the main workstation. |
| `~/code/footy_data` | A **dangling symlink** into iCloud. Do not use. |
| `the private S3 bucket (prefix `footy_data/`)` | 18,147 objects, 47.88 GB. Sibling prefixes `gpu_runs/` (18 objects, 795 MB), `models/` (3, 67 MB), `backups/`. |
| Desktop (WSL / Ubuntu) `~/code/footy_data` | About 44 GB per the KB; not checked here. The KB records the two machines diverging on `ball_gt_marks` in August; unverified. |
| `~/footy/footy_data` | Where the labeller looks by default; override with `FOOTY_CLIPS_DIR`, `FOOTY_GT_MARKS_DIR`, `FOOTY_BALL_CANDIDATES_DIR` (`labeller/server.py:92-114`). |
| A `data/` directory beside the repo checkout | Only `arsenal_mancity_example_video.mp4` (1920x1080, 25 fps, 250 frames, 10 s). |

Model weights are not in git: `rtdetr_1920_v11/best.pt` (66.8 MB) is on S3 `models/` and `gpu_runs/` and at `<storage>/models/`. The SAM3 weights are in `model_saves/sam3/` (S3 has the complete `sam3.1_multiplex.pt`; the local copy is a partial download).

## Datasets

| Name | Folder under `footy_data` | Format | Size and counts | Use |
|---|---|---|---|---|
| Raw match videos | `arsenal_{astonvilla,bournmouth_1st_half,bournmouth_2nd_half,mancity,norwich}/original_video` | mp4, ts, webm, mkv | 14.7 GB, 5 files | Source of everything |
| Derived frames and segments | same game folders | jpg, png, mp4 | about 35 GB | Frame sampling, broadcast filtering |
| `clips/` | `clips/` | mp4, 1920x1080, 25 fps, about 101 s | 162 files, 2.3 GB | Canonical dated Man City clips `arsenal_mancity_20250925_segNNN` |
| `machine_labels/` | `machine_labels/` | jsonl plus `_manifest.json` | 162 files, 889 MB, 2,767,151 boxes, 97,625 frames | RT-DETR-L (`rtdetr-l_1920_v11`, imgsz 1280, conf 0.1 or more), tier 3, unreviewed. On S3. |
| `ball_candidates/` | | jsonl | 162 files, 119 MB, 356,812 ball rows | Ball Check pool. **Local only.** |
| `ball_embeddings/` | | npz | 162 files, 42 MB | Crop descriptors for diversity sampling. **Local only.** |
| `tracklets/` | | jsonl (LapTracker) | 162 files, 334 MB, 1,606,131 rows | Player re-ID work. **Local only.** |
| `link_proposals/`, `identity_labels/`, `track_videos/` | | jsonl, mp4 | under 1 MB: 68 proposals, 9 review rows | Re-ID review. **Local only.** |
| `ball_gt_marks/` | | jsonl, one per clip | 45 files, 22,648 rows, 5.4 MB | Human and human-checked GT, normalised bbox and centre |
| `ball_gt_marks/ball_checks/` | | jsonl, append-only | 110 files, 364 verdicts, 2026-09-24 to 09-25 | `/ball_check` verdicts |
| GT snapshots | `ball_gt_marks.bak-ballcheck-20260924-151008`, `ball_gt_marks_MERGED` (identical, 39 files), `ball_gt_marks.pre-merge-20260826-234555` (33 files) | jsonl | about 5 MB each | Backups. **Local only**; S3 has `_backups/ball_gt_marks_2026-07-04`. |
| `detection_dataset/` | `roboflow_dataset_3` (162/49/46 train/val/test images), `roboflow_dataset_11` (1035/215/268) | YOLO images and txt | 47 MB, 237 MB | Detector training (Roboflow `footy-track-detection`) |
| `classifier_dataset/` | `roboflow_dataset_9`, `roboflow_dataset_10` | class folders | 109 MB, 70 MB | Broadcast-frame classifier |

## Label formats and counts

Sidecar rows are `{frame_index, bbox{x,y,w,h} normalised, center, tags: [class, provenance, *markers]}`; skip markers are rows with `bbox: null` and tag `no_ball` or `not_broadcast`. See [Architecture, section 6](architecture.md#6-data-formats-that-cross-stage-boundaries) for the six formats.

Current `ball_gt_marks` (22,648 rows, 0 unparsable): player 18,688, referee 2,099, `in_play_ball` 1,523, `not_broadcast` 138 (marker), coach 81, `no_ball` 68 (marker), person 32, player_sub 13, `out_of_play_ball` 2, untagged 4.

Ball provenance tiers: tier 1 `labeller` 692 (plus 2 `out_of_play_ball`); tier 2 `vittrack` plus `human_checked` 825; tier 3 `yolo` 6. `vittrack` rows total 15,386, all carrying `human_checked`; `labeller` rows 6,446; `yolo` 606. The KB's older 612/857/6 matches the August feature store, not these files.

Verdicts (364 rows): `ball` 179, `not_ball` 149, `unsure` 20, `corrected` 12, tombstones 4. By source: 283 candidates, 81 sidecar boxes. Machine labels: player 2,028,121; `in_play_ball` 352,039; referee 285,162; coach 53,607; person 26,463; player_sub 16,986; `ball` 4,773 (class `ball` is dead in Roboflow v11 but present in machine output).

Roboflow v11 `in_play_ball` instances, counted per file: train 1,010, valid 211, test 243. (`cat labels/*.txt` undercounts because the files lack a trailing newline; the KB's 826 is wrong.) v11-val is 74% one clip, `arsenal_example`.

## Feature store

DuckDB (`duckdb==1.3.2`; open with `read_only=True`). Schema and rules are in [Architecture, section 5.2](architecture.md#52-feature-store) and [Feature store design](design/feature_store.md).

- Newest copy: 2026-08-04, 4.2 MB, in a local job-mirror directory (older: 2026-07-05, 3.4 MB, in the old rig data directory). **Local only, not on S3, stale.**
- Contents: `detection` 26,364 rows; 34 of 41 games have detections; `frame` 1,869; 4 runs (`roboflow_v3`, `gt_import_labeller`, `gt_import_vittrack`, `gt_import_yolo`); `track_meta` 0 rows.
- Ball rows: `vittrack` 857 (`reviewed=False`), `hand_label` reviewed 612, `roboflow_v3` 235, `yolo` 6, `ball` 1.
- `game_id` is the clip stem (GT) or frame-filename stem (Roboflow). `game.source_video_uri` and `match_date` are NULL; `frame_uri` is NULL in samples. Clips are linked only by naming.
- Missing: all machine labels, candidates, verdicts, tracklets, identity labels, the six GT files added later, the 2026-09-24 corrections, and Roboflow v11 (only v3). `vittrack` boxes are stored `reviewed=False` because the `human_checked` ingest fix post-dates both copies.

## Footage

| Game | Original | Codec, res, fps | Duration | Derived | Labels |
|---|---|---|---|---|---|
| Man City 2025-09-25 | `arsenal_mancity_20250925.webm` 4.53 GB | h264 1920x1080 25 | 7153 s | 7,143 png frames, 876 mp4, 162 clips | GT on part; machine labels on all 162 clips |
| Norwich | `arsenal_norwich.mkv` 3.85 GB | hevc 3840x2160 50 | 6948 s | 606 jpg, no clips | none |
| Bournemouth 1st half | `Bournemouth vs Arsenal 1.ts` 1.67 GB | h264 1920x1080 50 | 3109 s | 121 mp4, 777 jpg | GT (`bournemouth_1st_seg070` has 3,863 rows) |
| Bournemouth 2nd half | `Bournemouth vs Arsenal 2.ts` 1.78 GB | h264 1920x1080 50 | 3210 s | 803 jpg, no mp4 | none |
| Aston Villa | `Arsenal - Aston Villa.mp4` 2.90 GB | not probed | | 320 mp4, 1,864 jpg | GT (`astonvilla_seg080` has 9,988 rows) |

Clip names come in three aliases: dated (`arsenal_mancity_20250925_segNNN`, in `clips/`), and short GT stems (`arsenal_mancity_segNNN`, `mancity_segNNN`, `astonvilla_segNNN`, `bournemouth_1st_segNNN`). The map between them is `ballcheck_clips/` (199 symlinks), which is not generated by anything in git ([T-03](todo.md), [T-09](todo.md)). GT covers 45 clips (7 with dated names) of about 1,700 broadcast segments.

## Local versus S3

- 18,112 keys are on both sides.
- **S3 only:** `_backups/ball_gt_marks_2026-07-04`, the complete `sam3.1_multiplex.pt`, `roboflow_dataset_11/test/labels.cache`.
- **Local only (604 files):** `ball_candidates`, `ball_embeddings`, `tracklets`, the GT snapshots, `identity_labels`, `link_proposals`, `track_videos`, the partial SAM3 file, both feature stores.
- `ball_gt_marks` including `ball_checks` has the same file sizes on both sides as of 2026-09-25.
- AWS credentials work (the KB's "creds dead" note from 2026-08-26 is stale).

## Gaps

- **Footage without labels:** Norwich and Bournemouth 2nd half have no GT. Machine labels cover only the 162 Man City clips. See [Best pipeline and label needs](pipeline-and-labels.md) for which gaps matter ([T-13](todo.md)).
- **Labels without footage:** `arsenal_example` (3,348 rows, 176 ball marks) has only the 10 s example video, tied by name. `eval_data/clips/mancity_seg104.jsonl` has no MP4.
- **Not in the feature store:** see above ([T-07](todo.md)).
- **Format hazards:** two centre formats (dict and list) can still appear; `ball_gt_marks` has 4 untagged rows; `astonvilla_seg080.jsonl` differs from its backup because six vittrack boxes were rewritten as `labeller` by the `corrected` path, so Ball Check does rewrite sidecars.
- **Cross-machine:** WSL-only rows cannot be checked from the build machine.
