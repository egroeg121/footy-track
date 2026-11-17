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
3. **Not the rig camera at all:** seg015 (high steep view: the centre circle has aspect 0.8 where the rig camera sees ~0.23; F1 stays
   0.2 whatever C is), seg066 (corner low angle + crowd), seg083 (very low wide view with the stand behind), seg068 / seg109 (goal
   mouth close-ups, 1-2 lines).
4. **Tails (the review's "uncut tail" hypothesis) are real but not the cause.** In the 22 segments the shot detector produced one shot
   each for three reasons: (i) `prep` calls `segment_shots` without `pair_ok`, so the soft rule (hist distance >= 0.30 and a failed
   pair) never fires; (ii) a hard cut (>= 0.60: seg005 f58 0.73, seg083 f221 0.82, seg131 f149 0.89) that leaves a sliver < 8 frames is
   merged back into the previous shot by the `min_len` rule; (iii) a logo wipe is gradual (seg046 f236-238: 0.25, 0.40, 0.37; seg073
   f469-472: 0.11, 0.26, 0.33, 0.38; seg071 f231-234 similar) and never reaches 0.60, and a shot's main-camera probability is the
   median over the shot, so a tail of a few evidence frames cannot lower it. The runs were only cut where a gap-1 motion pair failed.
   Dropping the tails changes the accepted fraction only by the frames themselves; see the ablation below.

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
* **Trimming** (`trim_runs`): hard cuts split runs (slivers are dropped, not merged); stretches of >= 2 evidence frames with
  main-camera probability < 0.4 are cut at the histogram jump in the 5-frame gap (smooth transitions at the gap start); a
  jump after the last evidence frame cuts the tail.
* **Wiring**: the runner runs the recovery pass only when the standard pass accepts < 50% of the frames, keeps it only if it accepts
  more, and then also requires an evidence frame to sit on the pose band. `meta.json["recovery"]` records both passes, the chosen
  camera position and each stage; `anchors_rec.json` holds the candidates. `--no-recover`, `--no-trim`, `--force-recover` (evaluation only).
  Segments the standard pass accepts are bit-identical to before.

RESULTS_PLACEHOLDER
