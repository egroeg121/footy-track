> Review of 2026-11-08, describing `main` @ 26d2506. Part of the [2026-11 review](index.md); prioritised actions are in the [TODO list](../../todo.md).

# KB vs reality: footy-track (2026-11-08)

Read-only. Nothing under ~/knowledge was edited. Nothing in any repo changed.
Compared against: fresh clone `scratchpad/kb-compare` (main @ 26d2506), data under /mnt/storage/footy_data, feature store opened with duckdb 1.3.2 read_only, plus the two scratchpad reports (architecture-review.md, dataset-inventory.md).
Spot-checks I ran myself: GT tier counts (692/825/6), ball_checks verdicts, machine_labels manifest, feature store ball rows, S3 totals, AWS STS, trainer argparse defaults, ingest_gt argparse, session.py tag handling, ball_check docstring, `lscpu` on this host.
KB pages relied on: footy-data-s3, footy-data-plane-reconciliation, footy-ball-detection-findings, footy-labeller-gotchas, footy-track-overview, wsl-gpu-training-box, vast-ai-gpu-rental, host-telemetry-hostmon, desktop-ubuntu-hardware-fault, deploy-plane, macmini-hardware, macmini-server-migration, polecat-shared-venv, index.md.

Legend: CONFIRMED / STALE / WRONG / UNVERIFIABLE / CONTRADICTED (by another KB page).

---
## A. Claims checked, page by page

### footy-data-s3 (updated 2026-07-04)
| Claim | Verdict | Evidence |
|---|---|---|
| footy_data is 41.2 GB / 14,624 files | STALE | S3 now 18,147 objects, 47.88 GB (`aws s3 ls --summarize`, I re-ran it). Local /mnt/storage/footy_data 45 GB / 18,716 files. |
| `~/code/footy_data` symlink to /mnt/storage/footy_data | WRONG now | It is a dangling symlink into the Air's iCloud path. Real data is /mnt/storage/footy_data. (The reconciliation page says so; this page does not.) CONTRADICTED by reconciliation. |
| "Plan: reformat sdb2 to ext4, mount /mnt/storage" | STALE (phrased as future) | Done: ext4 sdb2 at /mnt/storage. |
| AWS creds copied from Air, user-dev, eu-north-1 | CONFIRMED but CONTRADICTED | STS works today (account 361769576491). reconciliation page says creds dead (2026-08-26). Creds were re-issued; the old file is kept as ~/.aws/credentials.dead-20260826-*. |
| s3 sync command, bandwidth etiquette | UNVERIFIABLE (advice) | |
| Index entry says "41 GB" | STALE | same as above |

### footy-data-plane-reconciliation (2026-08-26)
| Claim | Verdict | Evidence |
|---|---|---|
| Mini 42 GB, WSL 44 GB | STALE (mini) / UNVERIFIABLE (WSL) | Mini is 45 GB. WSL/desktop not reachable from here. |
| 1,713 rows only on mini / 3,014 only on WSL; arsenal_mancity_seg010 482 vs 249 | UNVERIFIABLE | Needs the desktop copy. Mini side has since been superseded by the 2026-09 merge (ball_gt_marks_MERGED, pre-merge snapshot 33 files / 6,233 rows). KB never says the divergence was merged. |
| PK `(game_id, source, run_id, frame_index, detection_id)` | CONFIRMED | schema.py:301; I read the live store. |
| "Ingesting each machine's copy under a different run_id preserves both" | WRONG as a procedure | `ingest_gt` hardcodes `run_id = gt_import_<prov>`; no `--run-id` flag; canonical view merges all `gt_import_*` runs for a game (S9). Two machines into one DB silently overwrite on positional detection_id. Also run_id here is a provenance discriminator (gt_import_labeller / _vittrack / _yolo / roboflow_v3), not a machine discriminator. |
| `game_id` implied to be a match | WRONG | game_id = clip stem for GT imports (ingest_gt.py:202,227) and Roboflow frame-filename stem. No game_id is a match. |
| "Default `--gt-dir` is the Mac mini path" | WRONG | `--gt-dir`, `--video-dir`, `--db` are all `required=True` (ingest_gt.py:301-303). The mini path only appears in the docstring example. |
| `ingest_gt` "handles no_ball/not_broadcast skip markers" | CONFIRMED (but see S13: session flush and ingest disagree on skip-marker precedence) | |
| Tier table 612 / 857 / 6 | STALE | Current files: T1 labeller 692 (+2 out_of_play_ball), T2 vittrack+human_checked 825, T3 yolo 6. Counted by me from 45 files / 22,648 rows. 612/857/6 matches the 2026-08-04 feature store (verified: hand_label gt_import_labeller reviewed 612, vittrack 857, yolo 6), not today's GT. Pre-Ball-Check snapshot was 680/831/6. |
| "15,903 vittrack marks across 4 files; 21,566 rows" | STALE | Now 22,648 rows, 45 files, vittrack 15,386. |
| Filter `'human_checked' in tags or 'labeller' in tags` | CONFIRMED as the rule, but incomplete | The labeller session strips human_checked on the next flush (see missing item 1). |
| "New machine output must be tier 3, no human_checked" | CONFIRMED (machine_labels manifest: provenance_tier 3, reviewed false) | |
| Feature store ball_gt_marks + store = 1.5 MB | STALE | ball_gt_marks 5.4 MB; store 3.4-4.5 MB. |
| Roboflow datasets 47 MB exist only at a third party | STALE | roboflow_dataset_3 47 MB, _11 237 MB, plus classifier sets 9/10; copies on mini and S3. |
| "Videos (41 GB)" | WRONG | The 5 raw match videos are 14.7 GB; ~35 GB is derived frames/segments. |
| model_saves 6.4 GB | CONFIRMED but misleading | It is only SAM3 weights (sam3.pt 3.45 GB + partial sam3.1 file). No detector weights (`rtdetr_1920_v11/best.pt`) are in /mnt/storage/footy_data/model_saves or in git. |
| AWS creds dead on both machines | STALE (mini) | STS OK today. Desktop unverifiable. |
| DVC remote points at macOS path | CONFIRMED | .dvc/config:4 `/Users/georgebarnett/code/dvc-remote`. |
| Mini `~/code/footy_data` broken symlink | CONFIRMED | |
| "feature_store merged to main (ft-8o9)" | CONFIRMED, but CONTRADICTED by footy-track-overview ("branch ready to merge, ft-8o9 highest priority") | |
| Feature store is the annotation authority | CONFIRMED in design, NOT in practice | Both local copies are stale (Jul 5 / Aug 4), local-only (not on S3), contain no machine_labels, ball_checks, tracklets, identity labels, nor Roboflow v11. They predate the Sep-24 edits. |

### footy-ball-detection-findings (updated 2026-08-31)
| Claim | Verdict | Evidence |
|---|---|---|
| v3-val 44 balls / v11-val 211 balls; v11 = 1518 images (1035/215/268), v3 = 257 | CONFIRMED (image counts v3 162/49/46=257, v11 1035/215/268) | inventory |
| Cross-eval table (0.693/0.632, 0.250/0.209, 0.676/0.590, 0.221/0.209), 640->1280 0.062->0.696, RT-DETR-L 0.314/0.283 at 9 epochs, 3-epoch 640 vs 1280 table | UNVERIFIABLE from here | No eval result file in the repo reproduces any of these. docs/training.md has only a 5-epoch April run (player 0.481, ball 0). docs/bakeoff_results.json is a June 2026 run of the 640 ball-blind checkpoint. Numbers are KB-only, from raw notes. |
| UNRESOLVED .221/.209 vs .665/.591 for rtdetr_1920_v11 | CONFIRMED still unresolved, and now ALSO in the repo | docs/design/pipeline_overview.md:60,151 records the same dispute; detectors/ultralytics.py:37 still says `.221/.209`. No committed script can settle it (S12). Hypothesis (mine, not verified): the code-comment .221/.209 is the yolo11n v11-trained cross-eval cell in the KB's own table, i.e. the yolo11n number was pasted onto the RT-DETR checkpoint. Worth testing before re-measuring. |
| `CURRENT_BEST...rtdetr_1920_v11` run at imgsz 1280 | CONFIRMED (DEFAULT_IMGSZ=1280; machine_labels manifest imgsz 1280, conf 0.1) | Naming trap: "1920" in the model name, 1280 at inference. Not mentioned in KB. |
| "`ball` class dead, zero instances in v11" | CONFIRMED for Roboflow v11 (unrecounted by inventory); but machine_labels contain 4,773 `ball` rows and the store has 1 | minor |
| 1,453 ball frames; 942 in runs of 3+ | UNVERIFIABLE (not recomputed). Note GT now has 1,523 in_play_ball marks. | |
| `scripts/analyse_ball_tracker.py` exists | CONFIRMED | |
| Trainer defaults implied usable | WRONG by omission | `train_object_detector.py` defaults: `--imgsz 640` (ball-blind per this very page), `--device mps`, `--dataset-version 3`, `--workers 8`, `--cache ram`. Anyone running it bare silently trains a model that never sees the ball. |

### footy-track-overview (updated 2026-08-27)
| Claim | Verdict | Evidence |
|---|---|---|
| Code at /Users/georgebarnett/code/footy/footy_track/ ; layout refinery/rig, crew/georgeb | STALE | Repo is github egroeg121/footy-track (no rig/ layout). Mini has rig worktrees that are stale (refinery/rig on 8e5bec4; main now 26d2506, ~280 commits). Desktop: `~/code/footy_track` is the repo. |
| Architecture tree (detectors, trackers, labeller "SAM3 web UI", feature_store "on feat branch") | STALE | feature store merged; labeller is VitTrack-driven (SAM3 only in the dead Streamlit app); new `identity/`, `ball_eval/`, `ball_check.py`, `candidates.py`. No pipeline `Processor` classes exist (docs describe them anyway). |
| "SAM3 labelling UI ... propagates with SAM3 ... Needs GPU" | WRONG | CONTRADICTED by footy-labeller-gotchas ("SAM3 is dead code; VitTrack is the engine"). |
| Feature store "3 commits, ~2447 lines, branch ready to merge" | STALE | merged; CONTRADICTED by reconciliation page. |
| GPU situation: AWS g4dn blocked; Vast.ai workaround | STALE | Desktop RTX 2070 (native Ubuntu since 2026-08-27) is the trainer; Vast set up 08-27. AWS support-case status unverifiable. |
| Benchmark table yolo11n 0.221/0.209, RT-DETR-L 0.314/0.283; "use v11-val and nothing else" | UNVERIFIABLE; and omits the UNRESOLVED .665 dispute recorded on the findings page | |
| "Training runs on wsl-gpu-training-box; num_workers MUST be 0 or the VM dies" | WRONG now | Box is native Ubuntu (host-telemetry-hostmon, wsl-gpu page lower down, vast page). CONTRADICTED. |
| Key beads ft-8o9 "ready now" etc.; gotchas via polecats | STALE | Gas Town paused since 2026-10-25; beads not usable. Ft-rx3 write-up also duplicated (labeller page says fixed in 489d444). |
| Page missing from index.md | WRONG (index) | footy-track-overview has no index entry at all (grep count 0). |

### footy-labeller-gotchas (updated 2026-09-24)
| Claim | Verdict | Evidence |
|---|---|---|
| Ball-only handback 0.30 / non-ball 0.0, commit 489d444 | CONFIRMED | video_utils.py:51,55. labeller README still says 0.5 (D9). |
| VitTrack confidence constant 0.119; 5% on ball | UNVERIFIABLE here (measured on a rented box 08-29), consistent with "VitTrack fails on the ball" in the review | |
| SAM3 dead code; "compiling" string hardcoded; TORCHINDUCTOR cache stale | CONFIRMED (arch review 2.1) | Streamlit/streamlit-drawable-canvas still runtime deps. |
| GT sidecar truncation fix 6bcd153; env-resolvable dirs 2925224 | CONFIRMED (in main). Hardcoded iCloud fallback is still present in `server.py:97-106` and `video_utils.py:121` | partially stale: "fixed" is only true if FOOTY_GT_MARKS_DIR / ~/footy/footy_data/ball_gt_marks exists. |
| "No clone of the repo on the laptop" | STALE | Clones now exist (e.g. /home/george/footy-track on the mini). |
| "Ball Check never rewrites the GT sidecars" (09-23 section) | WRONG | `/ball_check/save_box` rewrites or appends `[label,"labeller"]` into the sidecar (ball_check.py:651-729). The module docstring still claims the opposite. Data confirms: astonvilla_seg080.jsonl has 6 vittrack boxes rewritten as labeller vs the pre-ball-check backup. |
| Ball Check on branch `feat/ball-check-mobile`, LAB-1001..1011 | STALE | merged; README traceability table still says 12 reqs; verdict set is five values (ball, not_ball, box_off, corrected, unsure) and precision = (ball+box_off+corrected)/decided (D2-D8). |
| Machine labels: 163 files, 888 MiB, 356,812 ball candidates, conf_threshold 0.1 | CONFIRMED | manifest: rtdetr-l_1920_v11, run 20260828-212938, tier 3, reviewed false; local 162 jsonl + manifest, 889 MB; ball_candidates 162 files / 356,812 rows. |
| "No feature_store.duckdb ... anywhere on the Mac mini" | WRONG | Exists at ~/code/footy_track/refinery/rig/data/feature_store.duckdb (Jul 5) and ~/jobs/.mirror/footy_track/refinery/rig/data/feature_store.duckdb (+.bak, Aug 4). True only for /mnt/storage and for "current". |
| `human_checked` ingest fix: first non-marker tag is class, checked machine box -> reviewed=True | CONFIRMED in code (ingest_gt.py:126-143,188-195). But neither store copy contains it (vittrack boxes are reviewed=False in the Aug-4 store). | |
| 15,377 sidecar lines with third tag; 831 ball boxes | STALE | now 15,386 vittrack lines; 825 T2 ball boxes. |
| ballcheck_clips 38 of 39 resolve | STALE | ballcheck_clips/ now has 199 symlinks (45 GT stems + candidate aliases). |
| Clip bridge `re.sub(r"_\d{8}_","_",stem)` matches 162 of 190 | UNVERIFIABLE (labeller clips dir absent); mechanism CONFIRMED by the 3-alias problem (S3 in review) |
| Sampling numbers (0.018 s, 2.7x redundancy, 500-600 crops/s) | UNVERIFIABLE (not re-benchmarked); candidates.py defaults changed since: DEFAULT_FOCUS 0.98, UNIFORM_SHARE 0.25 | the "interleave 3:1" lesson still matches. |
| Pre-existing test failures "5 failed (classifier, SAM3 detector tests) + 3 import errors" | PLAUSIBLE/UNVERIFIED here | SAM3 tests need weights; roboflow import error. Page does not say why SAM3 tests fail. |
| pytest needs pytest-xdist; run `-n0` | CONFIRMED (addopts `-q -n 2`; no CI runs tests) | |
| RT-DETR-L 3060: 19.8 fps, 97,625 frames, 2,767,151 detections | CONFIRMED against manifest (97,625 frames, 2,767,151 boxes) | |

### wsl-gpu-training-box
| Claim | Verdict | Evidence |
|---|---|---|
| Top-line framing: WSL2 box, `ssh wsl`, THE RULE workers=0 always | STALE / CONTRADICTED | Box retired as WSL on 2026-08-27 (host-telemetry-hostmon, and this page's own later sections). Rule is explicitly moot (hostmon, vast, this page §"Training performance findings"). Page title, summary, index line and top section all still lead with the WSL rule. |
| "`workers` change is UNCOMMITTED in the desktop's clone" | STALE | Committed (a9baacb "plumb --workers through ... configurable cache") and on main. Merge 26d2506 message says "default training workers to 0 under WSL" but code defaults `--workers 8`; help text: "Was pinned to 0 under WSL2; native Ubuntu has no such limit". The commit message is wrong, the code is right. |
| Runs die even with workers=0 (epochs 9,12,15,31) | CONTRADICTED by desktop-ubuntu-hardware-fault (the later deaths were hardware/memory, not dxgkrnl) | Page partly acknowledges this. |
| venv at ~/code/footy_track/.venv, 874 GB free, torch 2.8.0+cu128 | UNVERIFIABLE (desktop) | |
| Everything on the desktop | UNVERIFIABLE (desktop not reachable from the mini) | |

### vast-ai-gpu-rental / host-telemetry-hostmon (footy parts)
- vast: "workers>0 is fine here" CONFIRMED consistent with trainer. "44 GB S3 pull" vs "41 GB" vs "42 GB" vs real 47.9 GB: STALE and mutually inconsistent. "No AWS credentials on rented hardware" is advice, fine. "Roboflow data.yaml absolute path" fine. Training dataset ~136 MB: roboflow_dataset_11 is 237 MB on disk now (may exclude cache); UNVERIFIED.
- hostmon: "dxgkrnl rule no longer applies" CONFIRMED. No footy-specific numeric claims to check.
- deploy-plane: "some eval_data/clip paths are broken iCloud symlinks; resolve real data under /mnt/storage/footy_data" - STALE detail: eval_data/clips has 40 absolute symlinks into /mnt/storage/footy_data (they resolve on the mini) and only one jsonl (`mancity_seg104.jsonl`) with no mp4 beside it; the iCloud dangling link is `~/code/footy_data`.
- macmini-hardware / macmini-server-migration: footy mentions are about rig sizing / bead counts; no claim contradicted. "S3 footy_data download ... 45% CPU" is historical.

---
## B. Internal KB inconsistencies
1. **workers rule**: wsl-gpu-training-box (top: always 0) vs its own later section, host-telemetry-hostmon, vast-ai-gpu-rental (moot), footy-track-overview (MUST be 0), index line (still advertises the workers=0 rule).
2. **SAM3**: footy-track-overview presents SAM3 as the labeller engine; footy-labeller-gotchas says SAM3 is dead code.
3. **Feature store merge**: overview (branch ready, ft-8o9 pending) vs reconciliation (merged to main).
4. **AWS credentials**: s3 page (working, copied from Air) vs reconciliation (dead on both) vs reality (working again).
5. **footy_data size/count**: 41.2 GB/14,624 (s3 page + index), 42 GB / 44 GB (reconciliation), 44 GB (vast), real 47.9 GB S3 / 45 GB local / 18,147-18,716 files.
6. **Ball Check sidecar writes**: labeller page says never rewrites sidecars; the same page's data (and code) show it does.
7. **Ball mAP**: overview says "use v11-val and nothing else" with .221/.314; findings page has the UNRESOLVED .665 table; the overview does not mention it.
8. **Index**: `footy-track-overview` has no index entry; index mixes a table format (lines ~46-53) and a bullet format (56-62) with no consistent order; footy-data-s3 index line carries the stale "41 GB"; wsl-gpu index line is stale. No duplicate footy index lines found.
9. **Dates**: labeller-gotchas front matter `updated: 2026-09-24`, footy-track-overview `updated: 2026-08-27` with `access_count`/`sources: []` - no provenance links to raw notes; raw/footy-track-refactor-audit-2026-09-08.md warns itself STALE yet remains in raw/.

---
## C. Missing from the KB, ranked by time saved (proposed text; NOT written)

### 1. Labeller session strips `human_checked` (silently downgrades TIER 2 to TIER 3). Page: footy-labeller-gotchas (new section "DATA LOSS #2")
Verified: session.py:156-161 restores only label+provenance; `_box_line` (session.py:309-317) writes `[label, model]`.
> **The labeller session drops `human_checked` on flush (verified in code 2026-11-08, main 26d2506).** `_restore_box_line` keeps only label + provenance and `_box_line` writes `[label, model]`, so opening any clip in `/labeller` and making one edit rewrites the whole sidecar and every `vittrack + human_checked` (TIER 2) box in that clip becomes TIER 3 (`reviewed=False`) on the next `ingest_gt`. No test covers it; `human_checked` is known only to `ingest_gt.py:77`, not `labeller/constants.py` or the README. Before a labelling session: back up `ball_gt_marks/` and diff afterwards (`grep -c human_checked` per file). Fix = carry unknown/marker tags through load and flush.

### 2. Ball Check verdicts are write-only; `not_ball` still trains as a ball; Ball Check does write sidecars. Page: footy-labeller-gotchas (replace the "never rewrites" claim)
> **Correction to the 2026-09-23 section.** `/ball_check/save_box` (ball_check.py:651-729) rewrites or appends a `[label,"labeller"]` line into `ball_gt_marks/<clip>.jsonl`; the module docstring saying "Nothing is written to the GT sidecars" is wrong. On 2026-09-24/25 it rewrote 6 `astonvilla_seg080` vittrack boxes as labeller. **No verdict is consumed anywhere:** `ingest_gt` globs the top level only so `ball_checks/` is skipped, and `export_training_dataset` filters on `canonical` only, so a human `not_ball` on a sidecar vittrack box still exports as a positive ball and `box_off` trains with bad geometry. Verdicts file (2026-09-24 15:18 to 09-25 08:58): 110 files, 364 rows = ball 179, not_ball 149, unsure 20, corrected 12, 4 null tombstones; 283 from candidates (rtdetr-l_1920_v11), 81 from GT (vittrack 80, yolo 1). Undoing a `corrected` card only tombstones the verdict; the GT box stays.

### 3. Re-ingest leaves ghost rows; run_id cannot separate machines; clip aliases leak eval clips. Page: footy-data-plane-reconciliation (amend "The fix" and "Mandatory caveat")
> **Caveats the run_id design does not cover (verified 2026-11-08).** (a) `ingest_gt` is INSERT ... ON CONFLICT and never deletes: after a vittrack box is corrected to labeller (or a box deleted, shifting positional `detection_id`s) the old `(source=vittrack, detection_id=k)` row stays, remains canonical (all `gt_import_*` runs form one run_group) and is exported next to the corrected box. (b) `ingest_gt` hardcodes `run_id=gt_import_<prov>`, has no `--run-id` flag, and the canonical view merges every `gt_import_*`; two machines' sidecars into one DB overwrite each other. `--gt-dir/--video-dir/--db` are required (no default). (c) `game_id` is a clip stem, and one clip has up to three stems (`mancity_seg050`, `arsenal_mancity_seg050`, `arsenal_mancity_20250925_seg050`), so the eval-clip exclusion (exact stem `mancity_seg104`) and by-clip train/val split in `export_training_dataset` can be bypassed. (d) `reviewed` has no consumer; TIER 3 rows train by default.

### 4. This mini cannot run current duckdb/polars (no AVX2). Page: new bullet in macmini-hardware (or a "gotchas" line in footy-data-plane-reconciliation)
Verified: `lscpu` = Core i7-2635QM, 0 `avx2` flags. Inventory used duckdb==1.3.2 venv.
> **mac-mini-i7 is a Sandy Bridge i7-2635QM (no AVX2).** Recent `duckdb` and `polars` wheels crash (illegal instruction) here. Use `duckdb==1.3.2` in a dedicated venv and open feature stores with `read_only=True`; avoid polars. Check with `grep -c avx2 /proc/cpuinfo` before debugging a "segfault".

### 5. Trainer defaults are traps; the workers fix is committed. Page: footy-ball-detection-findings (new "Running the trainer" bullet) and wsl-gpu-training-box (fix the UNCOMMITTED note)
> `scripts/train_object_detector.py` defaults are `--imgsz 640` (the ball is never detected at 640), `--device mps`, `--dataset-version 3`, `--workers 8`, `--cache ram`. For ball work pass `--imgsz 1280 --device 0 --dataset-version 11`. `--workers/--cache` were added in a9baacb and are on main (merge 26d2506); its message says "default workers to 0 under WSL" but the code defaults to 8 (the 0 pin applied to WSL2 only, and the box is now native Ubuntu). The production RT-DETR training command, the batch-inference script and the machine_labels-to-ball_candidates filter are not in the repo.

### 6. Where the data actually lives and what is not on S3. Page: footy-data-s3 (refresh numbers; supersede 41 GB text)
> **State 2026-11-08.** S3 `footy_data/` = 18,147 objects / 47.88 GB; mini `/mnt/storage/footy_data` = 18,716 files / 45 GB (`~/code/footy_data` is a dangling iCloud symlink). AWS creds work again (account 361769576491; old key kept as `~/.aws/credentials.dead-20260826-*`). Raw videos are only 14.7 GB (5 games: Man City 4.5, Norwich 3.9 4K, Aston Villa 2.9, Bournemouth 1st 1.7 / 2nd 1.8); the rest is derived frames/segments. **Local-only (never on S3):** ball_candidates, ball_embeddings, tracklets, identity_labels, link_proposals, GT snapshots (.bak / _MERGED / pre-merge), both feature-store files (Jul 5 3.4 MB, Aug 4 4.2 MB under `~/code/footy_track/refinery/rig/data/` and `~/jobs/.mirror/...`). **S3-only:** `_backups/ball_gt_marks_2026-07-04`, the complete `sam3.1_multiplex.pt` (local copy is a partial `.db2a94BE`). `ball_gt_marks` incl. `ball_checks` is byte-identical on both sides as of 2026-09-25. Norwich and Bournemouth 2nd half have zero GT; machine labels cover only the 162 Man City clips.

### 7. Current GT numbers (replace the tier table). Page: footy-data-plane-reconciliation
> 2026-11-08, 45 files / 22,648 rows: ball tiers T1 labeller 692 (+2 out_of_play_ball), T2 vittrack+human_checked 825, T3 yolo 6; classes player 18,688, referee 2,099, in_play_ball 1,523, not_broadcast 138, coach 81, no_ball 68, person 32, player_sub 13, 4 untagged rows. The feature store (Aug 4) lags: vittrack 857 stored `reviewed=False`, no Roboflow v11, no machine labels; it predates the human_checked ingest fix, so re-ingest before trusting `reviewed`.

### 8. Candidates/clip-alias producers are manual; ballcheck_clips is the alias map. Page: footy-labeller-gotchas (extend the 09-24 section)
> The repo has no code to pull `machine_labels` from S3 or filter them to `ball_candidates/`, to run RT-DETR batch inference, or to generate the `ballcheck_clips` symlink farm (199 links on the mini); they cannot be rebuilt from git. `eval_data/clips` holds 40 absolute symlinks into /mnt/storage and a single GT sidecar (`mancity_seg104.jsonl`) with no mp4 beside it, so there is no eval loop for the production detector.

### 9. Format traps in `ingest_gt`. Page: footy-data-plane-reconciliation (extend "Two GT formats")
> `ingest_gt` does `bbox["x"]`, so one list-format bbox (from `label_ball_centers.py` or eval sidecars) raises TypeError and aborts the whole ingest; it also drops centre-only rows. Positional `detection_id` is not just a risk: `identity/stable_id.py` exists but is unused.

### 10. Possible explanation of the .221 vs .665 dispute. Page: footy-ball-detection-findings (UNRESOLVED section), flagged as hypothesis
> The `.221/.209` in `detectors/ultralytics.py:37` is numerically identical to the yolo11n v11-trained cross-eval cell above; likely a copy of the nano result onto the RT-DETR checkpoint. `docs/design/pipeline_overview.md` (2026-08-31) records the same dispute. The checkpoint is `rtdetr_1920_v11` but inference uses imgsz 1280.

### Lower-value (probably skip)
- SAM3 tests fail without weights (`model_saves/sam3/sam3.pt` 3.45 GB present on the mini; failures reproduce on a non-uv venv). Already 90% covered by "pre-existing failures" bullet; add one clause "SAM3 detector tests need weights".
- Docs drift list (D1-D26), dead modules list, 14 recommendations: all derivable from architecture-review.md/code; not KB material. The one durable pointer: `src/footy_track/labeller/README.md` is the canonical labeller spec but is outside mkdocs; `docs/architecture.md` lives only on `origin/docs/current-architecture`.

---
## D. Suggested index.md fixes (not done)
- Add `footy-track-overview` (no entry now).
- Rewrite the wsl-gpu-training-box line (native Ubuntu since 2026-08-27; workers rule moot) or rename the page.
- Change footy-data-s3 line "41 GB" to current figures.
