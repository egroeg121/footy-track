# PTZ V3: recovering the weak broadcast segments

Date: 2026-11-17. Branch `exp/calib-temporal-weak` (off `exp/calib-temporal-flow` 7d00646), worktree
`/home/george/footy-track-ptz3-weak`. Code: `calibration/ptz_recover.py`, `scripts/ptz_temporal_segment.py` (recovery pass),
`scripts/ptz_temporal_sheet.py` (review sheets), tests `tests/calibration/test_ptz_recover.py`.

## What was wrong (measured)

The 22 weak Man City segments (`accepted_frac` < 0.5) were investigated frame by frame. The line masks are good (centre circle,
halfway line, boxes are clean even under heavy shadow); the failures are in the pose search and in the camera model, not in the evidence.

1. **Search local optima (mode 1 and 2, 7 segments: 5, 57, 84, 114, 123, 131, 144).** `ptz.search_lines` ranks coarse cells with a
   90 px capture range, which favours wrong zoomed-out poses (f ~ 1900) over the true one (seg057 t200: the true cell is NMS rank 2),
   and the LM refinement from a cell that is 10% off in zoom is trapped (a centre circle 15% too small snaps onto its inner stroke:
   F1 0.37 from f=4500, 0.71 from f=4900, 0.80 from a multi-start). With 2 anchors in a 500-frame segment the joint solve then drifts
   away from them (seg057: both anchors reported as outliers, F1 0.04); with 12 confirmed anchors it holds (F1 > 0.6 everywhere).
   Segments 084 and 114 ("solve degrades good anchors") are the same thing: 2 and 1 anchors only.
2. **A second camera position (mode 1, 8 segments).** Man City segments 46, 73, 108, 137, 154 (and 71, 109) are shot from a gantry
   camera level with a penalty-area line, not from the centre position the rig describes. Free-C fits of the D-finder points land at
   C_x = +32.8 / -32.1 m with C_y, C_z within 2 m of the rig (seg073 t40/120/200: +32.8, +32.8, +32.6; seg108: -32.1, -32.2). With C
   hard-fixed at the centre the finder points reproject 190-270 px off; at C_x = +-32.7 they reproject 1-7 px off and the line F1 is
   0.5-0.66. This is the same effect the Bournemouth notes found (seg100:373 fits only at C_x = -36.6). The lines-only search could
   not find these frames because every hypothesis was evaluated at the centre C.
3. **Not recoverable (visual + F1):** seg015 (high steep view: the centre circle has aspect 0.8 where the rig camera, 13 deg
   elevation, sees ~0.23; best F1 0.2-0.3 at any pose or camera position tried), seg066 (corner low angle + crowd), seg083 (very low
   wide view with the stand behind), seg103 (high oblique view), seg068 / seg109 (goal-mouth close-ups, 1-2 lines), seg092 (ad
   banner overlay, close-up tail), seg156 (a box view whose free-C finder fits give C_x = -11.5 and -21.5 m on two frames: a camera
   position other than the three tried).
4. **Tails (the review's "uncut tail" hypothesis) are real but not the cause.** In the 22 segments the shot detector produced one shot
   each for three reasons: (i) `prep` calls `segment_shots` without `pair_ok`, so the soft rule (hist distance >= 0.30 and a failed
   pair) never fires; (ii) a hard cut (>= 0.60: seg005 f58 0.73, seg083 f221 0.82, seg131 f149 0.89) that leaves a sliver < 8 frames is
   merged back into the previous shot by the `min_len` rule; (iii) a logo wipe is gradual (seg046 f236-238: 0.25, 0.40, 0.37; seg073
   f469-472: 0.11, 0.26, 0.33, 0.38; seg071 f231-234 similar) and never reaches 0.60, and a shot's main-camera probability is the
   median over the shot, so a tail of a few evidence frames cannot lower it. The runs were only cut where a gap-1 motion pair failed.
   Cutting them is not needed: the F1 acceptance already rejects the tail frames (see the trimming ablation).

## What changed

* **Pose prior** (`rig.info.pose_prior` in `emirates.json`): over the 140 accepted Man City segments `tilt_deg = 96.825 - 10.106 ln f`
  within 1.1 deg (1-99%: -3.9..+3.2), pan 51-128 deg, f 2850-5750 px. The search runs on a grid dense along that band, and a
  hypothesis farther than 4 sigma is refused. The Vitality rig has no prior (search unrestricted).
* **Multi-start refinement** (`search_lines_ms`): every coarse cell is also refined from 0.82x / 1.2x its zoom; hypotheses ranked by F1.
* **Camera positions** (`rig_variants`): centre and C_x = +-32.7 m; chosen from 4 probe frames by mean candidate F1 (centre wins ties).
* **Finder anchors** (`finder_anchor`): centre / D finders, solved with C fixed, *not* refined on the line mask (the refinement
  lowers the F1 of correct box views: 0.56 -> 0.37, 0.48 -> 0.32, 0.46 -> 0.35 on seg073 / 108 / 156).
* **Confirmation** (`confirm_anchors`): a relaxed-F1 candidate (>= 0.40 with C at the rig, on the band) becomes an anchor only if
  another candidate carried along the motion chain lands within 1.5 m of it, or it is strong (F1 >= 0.7 and 0.6 at fixed C).
* **Trimming** (`trim_runs`, opt-in `--trim`, **off by default**): hard cuts split runs, stretches of >= 2 evidence frames with
  main-camera probability < 0.4 are cut. Measured below: it does not help and it removes valid zoomed goal-area views.
* **Wiring**: the runner runs the recovery pass only when the standard pass accepts < 50% of the frames, keeps it only if it accepts
  more, and then also requires an evidence frame to sit on the pose band. `meta.json["recovery"]` records both passes, the chosen
  camera position and each stage; `anchors_rec.json` holds the candidates. `--no-recover`, `--trim`, `--force-recover` (evaluation only).
  Segments the standard pass accepts are bit-identical to before.

## Results (code bd4f786; HEAD 4f28ad1 differs only by lint fixes, desktop, Man City, rig emirates)

### 22 weak segments: accepted fraction before / after
"before" = the standard pass (re-run here, same accepted fractions as the flow outputs); "after" = standard pass + recovery pass.
accepted = frames whose two neighbouring evidence frames both have F1 >= 0.5 (and, for a recovered segment, sit on the pose band).

| seg | frames | accepted before | accepted after | calibrated after | camera position | confirmed anchors | cpu s |
|---|---|---|---|---|---|---|---|
| 5 | 60 | 0% | 97% | 97% | centre | 12 | 204 |
| 15 | 280 | 0% | 0% | 0% | - | 0 | 336 |
| 46 | 240 | 0% | 27% | 100% | x+32.7 | 21 | 413 |
| 57 | 515 | 0% | 100% | 100% | centre | 11 | 330 |
| 66 | 75 | 0% | 0% | 0% | - | 0 | 0 |
| 68 | 135 | 0% | 0% | 0% | - | 0 | 263 |
| 71 | 235 | 0% | 5% | 100% | x+32.7 | 4 | 402 |
| 73 | 475 | 0% | 80% | 100% | x+32.7 | 12 | 280 |
| 83 | 225 | 0% | 0% | 0% | - | 0 | 318 |
| 84 | 235 | 0% | 100% | 100% | centre | 12 | 260 |
| 92 | 55 | 0% | 0% | 0% | - | 0 | 0 |
| 93 | 85 | 0% | 20% | 96% | centre | 6 | 191 |
| 103 | 225 | 0% | 0% | 0% | - | 0 | 299 |
| 108 | 175 | 0% | 100% | 100% | x-32.7 | 12 | 245 |
| 109 | 125 | 0% | 0% | 0% | - | 0 | 278 |
| 114 | 255 | 0% | 99% | 99% | centre | 12 | 244 |
| 123 | 65 | 0% | 94% | 94% | centre | 11 | 194 |
| 131 | 150 | 0% | 99% | 99% | centre | 6 | 218 |
| 137 | 50 | 0% | 100% | 100% | x-32.7 | 10 | 156 |
| 144 | 250 | 0% | 98% | 98% | centre | 12 | 278 |
| 154 | 345 | 0% | 58% | 100% | x-32.7 | 12 | 251 |
| 156 | 275 | 0% | 0% | 0% | - | 2 | 392 |

11 of 22 segments reach >= 50% accepted (10 at >= 80%); 2412 of 4530 frames accepted (0% -> 53%). "calibrated" counts frames that
have a solution, accepted only those the line evidence confirms. Segments 46 / 71 / 93 are solved on most frames but confirmed on
few (F1 stays 0.4-0.5 in between); they are not counted as recovered. CPU seconds include the standard pass (2-70 s) and the
recovery (3 camera positions x 4 probe frames, 12-24 candidate frames at ~9 s each on the shared desktop, motion on the run).

Sheets (three frames per segment, green = accepted, red = solved but not accepted, no lines = no solution):
`/mnt/storage/footy_data/scratch/temporal/rerun_weak/sheet_weak_1.jpg` (segs 5-92) and `sheet_weak_2.jpg` (segs 93-156).
Results in the flow's layout: `/mnt/storage/footy_data/scratch/temporal/rerun_weak/arsenal_mancity_segNNN@emirates/`
(+ `anchors_rec.json`, the recovery candidates; `meta.json["recovery"]`).

### Regression: 11 accepted segments re-run from scratch with the new code (desktop, no caches)
All outputs are bit-identical (pan, tilt, log f of every frame; accepted ranges), seg010 included. This is structural: the recovery pass
only starts when the standard pass accepts < 50% of the frames, and the standard pass is unchanged. Bournemouth seg060 (vitality rig,
0 GT, 1401 frames): accepted 98.1% -> 98.1%, median F1 0.627 -> 0.627.

| seg | frames | accepted before -> after | calibrated | max pose diff |
|---|---|---|---|---|
| 10 (GT) | 595 | 87.9% -> 87.9% | 99.7% | 0 |
| 3 | 290 | 98.6% -> 98.6% | 98.6% | 0 |
| 6 | 380 | 92.4% -> 92.4% | 93.4% | 0 |
| 13 | 375 | 90.1% -> 90.1% | 99.2% | 0 |
| 16 | 280 | 100% -> 100% | 100% | 0 |
| 18 | 350 | 98.9% -> 98.9% | 98.9% | 0 |
| 21 | 50 | 92.0% -> 92.0% | 92.0% | 0 |
| 49 | 125 | 97.6% -> 97.6% | 97.6% | 0 |
| 55 | 145 | 100% -> 100% | 100% | 0 |
| 77 | 430 | 74.0% -> 74.0% | 100% | 0 |
| 143 | 680 | 76.3% -> 76.3% | 99.9% | 0 |

GT eval of seg010 (20 GT frames, pitch error on the 5 m grid, driver `err5`): 20 / 20 answered, 19 accepted, median 0.238 m, p90 0.471,
max 0.655, none above 1 m; identical before and after (`/mnt/storage/footy_data/ball_gt_marks/calibration_gt/arsenal_mancity_seg010.json`
read-only, copied to the desktop scratch).

### Accuracy of the recovery path itself (`--force-recover` on segments the standard pass solves)
The recovery pass was forced on five accepted segments, including seg010, to check that it reproduces known-good answers
(the right camera position is chosen, no wrong pose is accepted):

| seg | camera position chosen (probe score centre / +32.7 / -32.7) | accepted standard -> forced | max pose difference to the standard solution (m, probe grid, 1 in 10 frames) |
|---|---|---|---|
| 10 | centre (0.69 / 0 / 0) | 87.9% -> 87.9% | 0.028 |
| 3 | centre | 98.6% -> 98.6% | 0.040 |
| 16 | centre | 100% -> 100% | 0.038 |
| 49 | centre | 97.6% -> 97.6% | 0.026 |
| 55 | centre | 100% -> 100% | 0.075 |

seg010 vs GT through the recovery path: median 0.233 m (standard 0.238), p90 0.460, max 0.658, 20 / 20, none above 1 m.

### Trimming ablation (recovery with `--trim` vs without)
| seg | accepted trimmed | accepted untrimmed |
|---|---|---|
| 73 | 72% | 80% |
| 108 | 75% | 100% |
| 137, 154, 84 | 100%, 58%, 99.6% | same |
| 5 | 96.7% | 96.7% |

Trimming never helped. In seg108 (f130-174) the main-camera classifier gives 0.02-0.4 on a goal-area view that is calibrated
correctly (the box and goal line up on the sheet), so trimming threw valid frames away; the F1 acceptance already rejects genuine
tails. Default: off.

### Cross-match (the pose prior comes from Man City only)
Nottingham Forest (Emirates rig, the flow's existing outputs) weak segments, and Bournemouth (vitality rig, no prior, no side positions):

| clip | frames | accepted before -> after | calibrated | camera position |
|---|---|---|---|---|
| forest seg052 | 630 | 44% -> 100% | 100% | centre |
| forest seg031 | 317 | 49% -> 51% | 100% | centre |
| forest seg029 | 192 | 0% -> 22% | 99% | x+32.7 |
| forest seg046 / 071 / 074 | 150 / 173 / 169 | 0% -> 0% | 0% | none usable |
| bournemouth seg087 | 1026 | 46.3% -> 46.5% | 100% | centre (recovery kept, +0.2 pt) |
| bournemouth seg060 (good) | 1401 | 98.1% -> 98.1% | 100% | not triggered, identical |

Held-out: the prior and the side positions generalise to another Emirates match (seg052 44% -> 100%), but the gain is smaller
(1 of 6 clearly recovered) and the Bournemouth rig (no prior, one camera position, no side-position estimate) gains nothing, as
expected: a Vitality pose band and its side positions would have to be estimated from that match's accepted segments first
(`rig.info.pose_prior`).

### Wrong accepts
None found on the sheets (a recovered overlay always sits on the circle / halfway line / box it should; frame-by-frame visual check of
all accepted ranges' three sample frames). What guards against them: (i) a candidate becomes an anchor only if a second candidate
carried along the motion chain agrees within 1.5 m (or F1 >= 0.7), (ii) acceptance still needs F1 >= 0.5 against the frame's own line
mask on both sides, (iii) recovered frames must lie within 4 sigma of the pose band, (iv) the recovery only replaces the standard
result if it accepts more. Residual risks: the band and the +-32.7 m positions come from this match (checked on Forest: 1 of 6 gained,
none visibly wrong; the three Forest segments that failed did so by abstaining); seg046 (27%), seg071 (5%) and seg093 (20%) are accepted on a
few frames only and their overlays look right on the sample frames but have weak F1 elsewhere, so treat them as partial.
The old confident-but-wrong overlays of seg057 (box / circle off) and seg114 are gone: both are now 99-100% accepted with
every anchor confirmed.

## Use for the other matches
Yes for Emirates matches (Forest, Villa): standard-pass results are untouched and the recovery only runs on segments that fail, at
3-9 CPU-minutes each. Re-run only the weak segments of a match, with a new `dest_root` so nothing is overwritten:

    PF=~/.claude/skills/prefect-jobs/scripts/pf.sh
    # weak = segments whose meta.json accepted_frac < 0.5 (Forest: 29, 31, 46, 52, 71, 74)
    $PF run ptz-temporal-match/ptz-temporal-match match=arsenal_nottinghamforest 'segments=[29,31,46,52,71,74]' \
        skip_existing=false dest_root=/mnt/storage/footy_data/scratch/temporal/rerun_weak \
        code_ref=exp/calib-temporal-weak max_parallel=3 --watch

(`code_repo` already defaults to a worktree of the same git repo, so the branch name resolves on the mini; pass the sha for a pinned
run.) For a fresh match, run the flow with `code_ref=exp/calib-temporal-weak` from the start: the recovery is part of the runner, a
good segment costs nothing extra, and a weak one costs 3-9 CPU-minutes. For Bournemouth / Vitality estimate the pose band first.
