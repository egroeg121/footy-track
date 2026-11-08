> Review of 2026-11-08, describing `main` @ 26d2506. Part of the [2026-11 review](index.md); prioritised actions are in the [TODO list](../../todo.md).

# footy-track dataset inventory (read-only, 2026-11-08, host mac-mini-i7)

KB pages used as leads (all in ~/knowledge/wiki/): footy-data-s3.md, footy-data-plane-reconciliation.md,
footy-ball-detection-findings.md, footy-labeller-gotchas.md. Verified/contradicted points are flagged below.

## 0. Where things live
- Real data root: /mnt/storage/footy_data (45 GB, 18,716 files, ext4 sdb2). ~/code/footy_data is a BROKEN symlink
  -> /Users/georgebarnett/Library/Mobile Documents/com~apple~CloudDocs/footy_data (confirms KB reconciliation page).
- /home/george/code/data holds only arsenal_mancity_example_video.mp4 (6 MB, 1920x1080, 25 fps, 250 frames, 10 s).
- Rig dirs under ~/code/footy_track/* are git worktrees of egroeg121/footy-track (refinery/rig on a stale temp branch 8e5bec4;
  /home/george/footy-track is a clean checkout of main at 658d63d). Rig dirs hold no label folders of note
  (crew/georgeb/data: example video, 250-entry labels json 60 KB, a roboflow_dataset_3 copy).
- Other machines: WSL desktop (~/code/footy_data, 44 GB per KB) is NOT reachable from here; not checked. KB says the
  mini and WSL diverged on ball_gt_marks (1,713 rows mini-only / 3,014 WSL-only on 2026-08-23); cannot verify or re-measure here.
- AWS creds work now (sts OK, acct 361769576491); KB "creds dead" note (2026-08-26) is stale. ~/.aws/credentials.dead-20260826-* kept.
- S3: s3://georgebarnett-general-purpose/footy_data/ (list only): 18,147 objects, 47.88 GB. Sibling prefixes: gpu_runs/ (18 obj, 795 MB),
  models/ (3, 67 MB), backups/ (100, 11.5 MB), live/, s3ncdu/.

## 1. Datasets
| Name | Location (under /mnt/storage/footy_data) | Format | Size | Counts | Use |
|---|---|---|---|---|---|
| Raw match videos (5 games) | arsenal_{astonvilla,bournmouth_1st_half,bournmouth_2nd_half,mancity,norwich}/original_video | mp4/ts/webm/mkv | 2.9+1.7+1.8+4.5+3.8 GB | 5 files | source of everything |
| Derived frames/segments | same dirs: full_video_frames, split_video_broadcast_frames, broadcast_segments, split_video_10s | jpg/png + mp4 | ~35 GB total | see sec 4 | frame sampling, broadcast filtering |
| clips/ | clips/ | mp4 (1920x1080 25 fps, ~101 s, ~2535 fr) + 1 json | 2.3 GB | 162 mp4 | canonical date-stamped Man City clips arsenal_mancity_20250925_segNNN |
| machine_labels/ | machine_labels/ | jsonl + _manifest.json | 889 MB | 162 files, 2,767,151 boxes, 97,625 frames | RT-DETR-L (rtdetr-l_1920_v11, imgsz 1280, conf>=0.1, tier 3, unreviewed) |
| ball_candidates/ | ball_candidates/ | jsonl | 119 MB | 162 files, 356,812 ball-class rows | Ball Check sampling pool (local only) |
| ball_embeddings/ | ball_embeddings/ | npz | 42 MB | 162 | crop descriptors for diversity sampling (local only) |
| tracklets/ | tracklets/ | jsonl (bytetrack/"lap") | 334 MB | 162 files, 1,606,131 rows | player re-ID work (local only) |
| link_proposals, identity_labels, track_videos | same names | jsonl/mp4 | <1 MB | 3 files/68 rows; 9 review rows; 1 mp4 | re-ID review (local only) |
| ball_gt_marks (current) | ball_gt_marks/ | jsonl, one per clip | 5.4 MB | 45 files, 22,648 rows | human + human-checked GT, normalised bbox/center |
| ball_gt_marks/ball_checks | ball_gt_marks/ball_checks/ | jsonl, append-only | 125 KB | 110 files, 364 rows | /ball_check verdicts, 2026-09-24 15:18 to 09-25 08:58 |
| GT snapshots | ball_gt_marks.bak-ballcheck-20260924-151008, ball_gt_marks_MERGED (identical, 39 files, 22,642 rows), ball_gt_marks.pre-merge-20260826-234555 (33 files, 6,233 rows) | jsonl | 5.3/5.3/1.5 MB | | backups |
| detection_dataset | roboflow_dataset_3 (162/49/46 train/val/test imgs), roboflow_dataset_11 (1035/215/268) | YOLO images+txt | 47 MB, 237 MB | 3,561 files | detector training (footy-track-detection) |
| classifier_dataset | roboflow_dataset_9 (594/183/166), roboflow_dataset_10 (387/104/100) | class-folder images | 109 MB, 70 MB | 1,543 files | broadcast-frame classifier |
| model_saves | model_saves/sam3/ | .pt | 6.4 GB | sam3.pt (3.45 GB), sam3.1_multiplex.pt.db2a94BE (3.34 GB, partial) | SAM3 weights |
| Feature store | ~/code/footy_track/refinery/rig/data/feature_store.duckdb (Jul 5, 3.4 MB); ~/jobs/.mirror/footy_track/refinery/rig/data/feature_store.duckdb (+ .bak) (Aug 4, 4.2/4.5 MB; hardlinked copies in jobs/footy_track-20260804-17*) | DuckDB | 3-4.5 MB | see sec 3 | annotations authority |
Also: no detections parquet and no duckdb under /mnt/storage (confirms KB). Only parquet: 3 small tracks.parquet in crew/georgeb/runs.

## 2. Labels (ball_gt_marks, current, 22,648 rows; 0 unparseable)
Class (first non-marker tag): player 18,688 | referee 2,099 | in_play_ball 1,523 | not_broadcast 138 (skip marker) | coach 81 | no_ball 68 (skip marker) | person 32 | player_sub 13 | out_of_play_ball 2 | untagged 4.
Ball provenance tiers (current): T1 labeller 692 (+2 out_of_play_ball) | T2 vittrack+human_checked 825 | T3 yolo 6 (no human_checked).
Marker tags: labeller 6,446 | vittrack 15,386 (all also human_checked 15,386) | yolo 606.
- KB claim 612/857/6 is STALE: 612/857 match the Aug-4 feature store and the older GT, not today's files. Pre-ball-check snapshot: 680/831/6.
  Today differs by +12 labeller / -6 vittrack: 6 new files (arsenal_mancity_20250925_seg011/094/101/115/134/135, 1 labeller ball each, plus seg010 present)
  and 6 astonvilla_seg080 vittrack boxes (frames 287, 335, 381 ...) rewritten as labeller (the "corrected" verdict path).
  Note this contradicts the Ball Check design note "never rewrites the GT sidecars": astonvilla_seg080.jsonl has changed vs the backup.
- ball_checks verdicts (364 rows): ball 179, not_ball 149, unsure 20, corrected 12, null tombstones 4. Source: cand 283 (rtdetr-l_1920_v11), 81 from GT (vittrack 80, yolo 1).
- Machine labels (tier 3, unreviewed): player 2,028,121 | in_play_ball 352,039 | referee 285,162 | coach 53,607 | person 26,463 | player_sub 16,986 | ball 4,773 (class "ball" is dead in Roboflow v11 but present in machine output).
- Roboflow datasets (curated, hand labels): v3 and v11; classes ball, coach, in_play_ball, person, player, player_sub, referee. Per KB v11 has 826 ball instances, 'ball' = 0 (not re-counted).
- Identity: 9 review rows (identity_reviews.jsonl); 68 link proposals.

## 3. Feature store (DuckDB 1.3.2, opened read_only)
Schema (all copies): game, run, frame, detection (PK game_id,source,run_id,frame_index,detection_id), track_meta (0 rows), plus views/tables
detections_enriched, frame_features, tracks_enriched (0). detection carries label, confidence, bbox, track_id, needs_review, reviewed, dataset_tag.
Newest (Aug 4, 4.2 MB): detection 26,364 rows; 34 game_ids with detections of 41 games; frame 1,869; run 4:
roboflow_v3, gt_import_labeller, gt_import_vittrack, gt_import_yolo (created_at null; match_date null for all games).
 source x reviewed: hand_label reviewed 4,786 / unreviewed 4,915 (roboflow_v3); vittrack 15,903 unreviewed; yolo 760 unreviewed.
 label: player 22,041 | referee 2,398 | in_play_ball 1,710 | coach 128 | player_sub 43 | person 43 | ball 1.
 ball: vittrack 857 (reviewed=False) | hand_label reviewed 612 | roboflow_v3 235 | yolo 6 | ball 1.
 dataset_tag: all_objects_v3:{train,val,test} (split tags) and ball_gt_marks.
 Largest games: astonvilla_seg080 9,988, arsenal_mancity_20250925 4,915 (roboflow_v3), bournemouth_1st_seg070 3,863, arsenal_example 2,759.
Jul 5 store (3.4 MB): 11,051 rows, 27 games w/ detections of 34, 965 frames, 3 runs (no vittrack), ball: 363+128+101=592 labeller, 235+1 roboflow.
Aug-4 .bak: identical row counts to the Aug-4 file.
Link: detection.game_id = GT file stem (or Roboflow clip) and joins frame/game on (game_id, frame_index); run_id is the machine/source discriminator.
 Nothing joins to video paths: game.source_video_uri is NULL, frame_uri NULL in samples; clip linkage is by stem naming only.
 The store has no rows from machine_labels (2.77 M), ball_candidates, ball_checks, tracklets, or identity labels.
 vittrack boxes are stored reviewed=False despite carrying human_checked in the jsonl (the 2026-09-24 ingest fix, commit in main, is not in either store copy).
 Feature store is stale vs GT (store predates 2026-09-24 edits) and not on S3 at all.

## 4. Raw footage
| Game | Original | Codec/res/fps | Duration | Derived |
|---|---|---|---|---|
| Man City 2025-09-25 | arsenal_mancity_20250925.webm 4.53 GB | h264 1920x1080 25 | 7153 s | 7,143 png full frames, 876 mp4 (split_video_10s parts + broadcast frames), clips/ 162 x ~101 s |
| Norwich | arsenal_norwich.mkv 3.85 GB | hevc 3840x2160 50 | 6948 s | 606 jpg frames, no clips |
| Bournemouth 1st half | Bournemouth vs Arsenal 1.ts 1.67 GB | h264 1920x1080 50 (2 video streams) | 3109 s | 121 mp4 segments, 777 jpg |
| Bournemouth 2nd half | Bournemouth vs Arsenal 2.ts 1.78 GB | h264 1920x1080 50 | 3210 s | 803 jpg, zero mp4 segments |
| Aston Villa | Arsenal - Aston Villa.mp4 2.90 GB | (not probed) | | 320 mp4 (broadcast_segments etc.; sample seg is 1280x720 25 fps 458 fr), 1,864 jpg |
Extensions overall: jpg 7,359, png 7,143, txt 1,784 (labels), mp4 1,481, jsonl 756, npz 162, json 6, ts 2, mkv 1, webm 1, pt 1.
Naming: games `<home>_<away>` dirs; segments `Arsenal - Aston Villa_segNNN.mp4`, `Bournemouth vs Arsenal 1_segNNN.mp4`, `arsenal_mancity_20250925_segNNN.mp4`
(clips/) and `..._partNNN.mp4`; frames `arsenal_mancity_20250925_000123.png`. GT sidecar stems are UNDATED/short (astonvilla_segNNN,
bournemouth_1st_segNNN, mancity_segNNN, arsenal_mancity_segNNN) so they need a bridge (strip `_\d{8}_`); ballcheck_clips/ (199 symlinks) holds that map.
Note: clips/ and the GT stems differ; GT stems with a dated name (7) do live in clips/.
Example video 6 MB also in S3 root of footy_data. Source site: url.txt -> fullmatch-replay.org.

## 5. Mismatches and gaps
- Local vs S3 (footy_data/ prefix): 18,112 keys on both; S3 18,147 / 47.88 GB vs local 18,716 / ~45 GB.
  S3-only: _backups/ball_gt_marks_2026-07-04 (33 files, 1.4 MB), model_saves/sam3/sam3.1_multiplex.pt (3.50 GB complete; local is a partial .db2a94BE), detection_dataset/roboflow_dataset_11/test/labels.cache.
  Local-only (604): ball_candidates (162), ball_embeddings (162), tracklets (162), ball_gt_marks.bak/MERGED/pre-merge snapshots (111), identity_labels, link_proposals (3), track_videos (1), .claude, partial sam3 file.
  Size-differ: roboflow_dataset_11 data.yaml (S3 498 B vs local 324 B) and valid/labels.cache.
  ball_gt_marks (155 files incl. ball_checks) is byte-size identical on both sides, so the Sep-25 state has been uploaded. KB "41.2 GB / 14,624 files" is stale (now 18,147 / 47.9 GB).
- Labels without footage: arsenal_example GT (3,348-row file, 176 ball marks) has no video except arsenal_mancity_example_video.mp4 (tied by name only); all other 44 stems resolve to a clip. 6,233-row pre-merge set is subsumed by current.
- Footage without labels: Norwich (3840x2160, ~7k s) and Bournemouth 2nd half have zero GT; only 162 clips of Man City have machine labels; GT covers 45 clips (7 dated) of ~1,700 broadcast segments.
- Duplicated/conflicting: three identical-purpose GT snapshots (MERGED == .bak byte-for-byte counts, current differs by 6 new files and 6 rewritten astonvilla boxes); KB tier table differs from current counts; two GT center formats (dict vs list) still possible; S3 _backups is the Jul 4 pre-merge state.
  Cross-machine conflict (WSL-only rows, arsenal_mancity_seg010 482 vs 249) CANNOT be checked here; WSL is the only place to confirm.
- Not in the feature store: all machine_labels, ball_candidates, ball_checks verdicts, tracklets, identity reviews, the 6 new GT files and the Sep-24 corrections; Roboflow v11 (store has only v3); the whole store is local-only and stale (Aug 4).
- Roboflow-only: curated datasets are on S3 but their canonical source is Roboflow (KB says 47 MB only at a third party; actually v3/v9/v10/v11 copies are on mini and S3).
- Broken/bad: ~/code/footy_data dangling symlink; eval_data/clips empty; ball_gt_marks has 4 untagged rows; astonvilla_seg080 tag rewrite contradicts "never rewrites sidecars".
