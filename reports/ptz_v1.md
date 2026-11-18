# PTZ rig, V1 offline validation (Emirates GT)

Date: 2026-11-14. Branch `exp/calibration-ptz`. Proposal: `docs/design/calibration_ptz.md` (section 4, V1).
Code: `src/footy_track/calibration/ptz.py`, `src/footy_track/scripts/ptz_validate.py`, `tests/calibration/test_ptz.py`.
No LLM calls; one process, `OMP_NUM_THREADS=1`, `nice -n 10`. Raw results and plots are in the scratchpad
`.../scratchpad/ptz_v1/` (`points_all.json`, `lines.json`, `rigs.json`, `zoom.json`, `synthetic.json`, `*.svg`).

## Summary

- **The fixed-centre hypothesis holds on the Emirates GT.** Joint rig fits give C = (0.11, -66.61, 15.77) m from the
  48 Man City frames and C = (0.09, -66.43, 15.71) m from the 19 Villa frames: 0.2 m apart, with shot-bootstrap
  std 0.03/0.11/0.05 m (Man City) and 0.06/0.18/0.04 m (Villa). Shared roll is ~0 (-0.19 and +0.16 deg), shared constant
  k1 is 0.013-0.015.
- **Pass criterion (d), lines only, median < 1 m:** met. Leave-shot-out: answered frames (88%) have median 0.25 m, p90
  0.66 m, p95 1.05 m; 12% abstain; 3% (2 frames) are confident and wrong. Leave-match-out (Man City rig, Villa frames):
  median 0.43 m, 26% abstain. Caveat: the line-search settings were tuned while looking at these frames (section 5).
- **Pass criterion (b),(c), clearly better than free H:** met. With 2-3 points, or a line plus one point, free H and the
  7-DOF pinhole have no solution; the PTZ solve has median 0.25-0.3 m and p95 about 1 m. With 4 random points free H
  fails on 40% of draws and its p95 is 2.3 m against 0.86 m for PTZ. When free H has 5 or more well-spread points it is
  *better* at the median (0.07 m vs 0.20 m), but that comparison is circular (below).
- No frame needs more than a pan/tilt/zoom solve; the model never produced a non-physical camera.

## 1. Setup

- Data: Emirates GT only, `calibration_gt` frames with >= 4 correspondences: 48 Man City frames (14 clips) and 19 Aston
  Villa frames (5 clips), 67 in total (56 accepted LLM proposals, 5 pinhole refits, 6 hand). Man City frames are 1920x1080, Villa 1280x720; normalised coordinates are used
  throughout and f is reported in px at 1920 wide.
- Model (`ptz.py`): camera at C, pan/tilt/roll rotation, focal f (fitted as log f), principal point at the image centre,
  square pixels, division-model k1 (constant per rig). `Rig` = C, shared roll, k1 and a prior tolerance
  `sigma_C`. Per-frame unknowns: pan, tilt, log f, with C inside its Gaussian tolerance (default sigma = 1, 3, 1 m,
  the IQR of the proposal's table; `fixed` = 0, `tight` = 0.3/1/0.3, `loose` = 5/10/3, `free` = 100). A broad prior on
  log f (centre 3500 px, sigma 0.6) conditions the 2-3 point case.
- Rig fit: bundle adjustment on undistorted pixel error, Cauchy loss (scale 4 px), all frames of the training set
  (not filtered by the pinhole-ratio test).
- Folds: **leave-match-out** (rig from one match, test on the other, both directions) and **leave-shot-out** (rig from
  the other clips of the same match; clip = shot group, which is the conservative grouping since each clip is 1-3
  shots). Leave-one-frame-out is not used.
- Metric: pitch-metre error against the stored GT homography (with that frame's own k1) over a 3 m grid of visible
  pitch points: each grid point is mapped to the image by the GT, back-projected by the estimate, and compared. Trial
  statistic = median over the grid; median / p90 / p95 are over trials (frame-equal weighting). "near/far" split the
  visible grid at its median image row (lower half = near). "fail" = no solution (under-determined or degenerate),
  "abstain" = the line search declined, ">5 m" = answered but trial median error above 5 m.
- (b): 8 random draws per frame of 2, 3, 4 and 5 GT points. (c): halfway line + centre-circle points only (35 trials,
  the GT has few central views) and a single pitch line's points plus one other point (341 trials).
- Baselines on the same points and lens k1: free 8-DOF DLT homography (`geometry.fit_homography`; fail if < 4 points
  or degenerate) and the pinhole without the C prior (`pinhole_freeC`: same solver, C sigma 100 m; fail if < 4 points).
  Lines-only free-H baseline is not given (the existing refine needs an initial H).

## 2. Rig fits and the camera centre

| fit | frames | C x, y, z (m) | roll (deg) | k1 | bootstrap std of C (m) |
|---|---|---|---|---|---|
| Man City | 48 | 0.11, -66.61, 15.77 | -0.19 | 0.0147 | 0.03, 0.11, 0.05 |
| Villa | 19 | 0.09, -66.43, 15.71 | +0.16 | 0.0132 | 0.06, 0.18, 0.04 |
| Man City, plausible-ratio frames only (35) | 35 | 0.11, -66.62, 15.78 | | | |
| Villa, plausible-ratio frames only (13) | 13 | 0.16, -66.44, 15.73 | | | |

- Bootstrap: 30 resamples of whole clips. The two matches agree within the bootstrap spread and a few tenths of a metre.
  The proposal's table had y = -65.2 from per-frame decompositions without a lens; the joint fit with k1 gives -66.6.
  Per-frame values from the decomposition (plausible frames) show no visible trend with f (`Cy_vs_f.svg`; Spearman of
  C against f, Man City: -0.20, -0.12, -0.20 for x, y, z).
- **Zoom test** (`fit_rig(fit_axis_slope=True)`: C slides along the optical axis, b metres per unit of ln f): both
  matches together, b = +2.1 m per unit ln f (1.5 m per doubling of the zoom), shot-bootstrap 5-95%: 0.4 to 5.0.
  Adding it lowers the robust cost by 0.7% (2089 to 2075) and does not change the per-frame rms. So there is at most a
  small, marginally detectable shift along the axis; it is well inside the 1/3/1 m tolerance and not needed for the
  results below. The proposal's worry that C moves with zoom is not confirmed, not excluded.
- **Joint fit residuals**: per-frame reprojection rms median 2.9 px (Man City) and 4.3 px (Villa) at 1920 wide; p90
  10 and 8.5 px.
- Using only the frames whose GT H decomposes to a square-pixel camera (ratio 0.8-1.25) moves C by < 0.1 m, so the
  non-physical GT frames do not drive the rig.

## 3. k1

- Joint fits with points: constant k1 = 0.013-0.015 for both matches (bootstrap std 0.003-0.006).
- Per-frame k1 fitted on the white-line mask with the rig C fixed (67 frames): median 0.022, IQR 0.015-0.029; Spearman
  of k1 with f is -0.20 (no clear trend; `k1_vs_f.svg`). The line-mask value is larger than the point-based one; the
  constant-k1 assumption is adequate here and `k1(f) = a + b/f` is not justified by this data.
- Effect of dropping the lens (rig k1 = 0) in (a), leave-shot-out: median 0.19 vs 0.18 m, p90 0.54 vs 0.48, p95 0.88 vs
  0.81; far-half p90 about the same. Small but in the right direction.

## 4. Results (leave-shot-out, all 67 frames; error in metres)

"PTZ" = `ptz_mid` (sigma 1/3/1 m). Full tables: `report_tables.md` in the scratchpad.

| experiment | method | trials | fail | median | p90 | p95 | near median | far median | far p90 |
|---|---|---|---|---|---|---|---|---|---|
| (a) all GT points | free H (circular, see below) | 67 | 0% | 0.07 | 0.18 | 0.22 | 0.07 | 0.09 | 0.48 |
| | pinhole, free C | 67 | 0% | 0.19 | 0.45 | 0.79 | 0.17 | 0.21 | 0.94 |
| | PTZ, C hard-fixed | 67 | 0% | 0.25 | 0.50 | 0.74 | 0.24 | 0.28 | 0.93 |
| | PTZ | 67 | 0% | 0.18 | 0.48 | 0.81 | 0.18 | 0.23 | 0.95 |
| (b) 2 random points | free H / pinhole free C | 536 | 100% | - | - | - | - | - | - |
| | PTZ, C hard-fixed | 536 | 0% | 0.30 | 0.75 | 0.85 | 0.29 | 0.33 | 1.19 |
| | PTZ | 536 | 0% | 0.28 | 0.79 | 1.13 | 0.29 | 0.31 | 1.32 |
| (b) 3 random points | free H / pinhole free C | 536 | 100% | - | - | - | - | - | - |
| | PTZ, C hard-fixed | 536 | 0% | 0.26 | 0.63 | 0.77 | 0.25 | 0.29 | 1.01 |
| | PTZ | 536 | 0% | 0.25 | 0.61 | 0.92 | 0.24 | 0.27 | 1.09 |
| (b) 4 random points | free H | 392 | 40% (2% > 5 m) | 0.14 | 0.97 | 2.27 | 0.16 | 0.17 | 1.65 |
| | pinhole, free C | 392 | 0% | 0.24 | 0.75 | 1.11 | 0.23 | 0.25 | 1.24 |
| | PTZ | 392 | 0% | 0.22 | 0.67 | 0.86 | 0.22 | 0.25 | 1.05 |
| (b) 5 random points | free H | 192 | 11% | 0.07 | 0.30 | 0.40 | 0.08 | 0.06 | 0.52 |
| | pinhole, free C | 192 | 0% | 0.21 | 0.42 | 0.59 | 0.21 | 0.20 | 0.74 |
| | PTZ | 192 | 0% | 0.20 | 0.39 | 0.61 | 0.21 | 0.20 | 0.72 |
| (c) halfway line + circle points | free H | 35 | 49% | 0.06 | 0.12 | 0.12 | 0.05 | 0.10 | 0.56 |
| | pinhole, free C | 35 | 26% | 0.12 | 0.37 | 0.49 | 0.11 | 0.17 | 0.73 |
| | PTZ | 35 | 0% | 0.21 | 0.49 | 0.50 | 0.17 | 0.26 | 0.94 |
| (c) one line's points + 1 other | free H | 341 | 100% | - | - | - | - | - | - |
| | pinhole, free C | 341 | 67% | 0.25 | 0.71 | 1.09 | 0.22 | 0.27 | 1.29 |
| | PTZ | 341 | 0% | 0.23 | 0.50 | 0.74 | 0.22 | 0.24 | 0.89 |
| (d) lines only, from the video frame | PTZ line search | 67 | 0% (12% abstain, 3% > 5 m) | 0.25 | 0.66 | 1.05 | 0.23 | 0.31 | 1.39 |
| | same, C hard-fixed | 67 | 0% (12% abstain, 3% > 5 m) | 0.24 | 0.58 | 0.81 | 0.23 | 0.29 | 1.20 |

Tolerance ablation (a)/(b)/(c): `fixed`, `tight`, `mid`, `loose` all within about 0.05 m median of each other; `loose`
and `free` are slightly worse in the tails for 2-3 points (p95 1.2 vs 0.9 m). The data do not need a hard-fixed C and
do not suffer from a tolerance.

**Leave-match-out** (pooled both directions are in `report_tables.md`; ptz_mid):

| rig from, tested on | (a) median / p95 | (b) 2 pts median / p95 | (b) 3 pts median / p95 | (c) line+1 median / p95 | (d) answered median / abstain |
|---|---|---|---|---|---|
| Man City, Villa (19 frames) | 0.38 / 0.55 | 0.50 / 1.05 | 0.42 / 0.97 | 0.39 / 0.96 | 0.43 / 26% |
| Villa, Man City (48 frames) | 0.28 / 1.11 | 0.43 / 1.16 | 0.35 / 1.08 | 0.33 / 0.88 | not run |

By zoom (leave-shot-out, PTZ, terciles of f at 3487 / 4178 px): 2-point median 0.37 / 0.27 / 0.24 m for low / mid / high
f; p90 1.09 / 0.59 / 0.56. The wide (low f) frames are the weakest, as expected (more pitch visible, more lens effect).

Near / far: the far half is consistently a little worse than the near half (2-point median 0.31 vs 0.29 m; far-half p90
1.3 m), the same pattern as the free-H baselines.

### Free H vs PTZ: how to read it

The GT homography is itself a DLT/refit of the same labelled points. So whenever free H has >= 5 well-spread
points it reproduces the GT almost exactly (0.06-0.07 m) by construction, and the (a) line for free H is circular. The
PTZ model gives 0.2 m in the same cases because it is constrained: the 0.2 m is the disagreement between the labels and a
single physical camera plus the lens model, which is about the 0.5 m noise the user accepted in the GT. The claim of
the proposal is about the under-determined and nearly-degenerate cases and the tails, and there the evidence is clear:
free H has no answer at 2-3 points and on lines, fails 40% of 4-point draws and 49% of halfway+circle subsets, and its
p95 when it does answer a 4-point draw is 2.3-3.7 m.

### Lines only (d): what was built and how it behaves

Search: C at the rig prior, grid over pan 40-145 deg (2 deg), tilt 4-40 deg, f 1400-8000 px (17 log steps), scored by a
two-sided score on the white-line mask (predicted visible line length that is not on a detected line or off the grass,
plus detected line length not explained, normalised by the detected length; wide capture 90 px at 960 px width, then a
local refinement grid, then robust LM of pan/tilt/log f/C within its tolerance on a truncated blurred distance transform).
The top 10 are re-scored, merged when within 2.5 m of each other in pitch space, and the answer is withheld when the best
and next distinct hypothesis differ by less than 0.08 in score, or when support (1 - score) is below 0.15. About 16 s per
frame on one thread.

- Answered 88% (59 of 67). Wrong answers: `arsenal_mancity_seg010:375` (best hypothesis does not land on the pitch, > 100 m),
  `arsenal_mancity_seg010:575` (10 m). A third, `arsenal_mancity_seg060:90`, is 2.6 m off. These three are also among the
  frames the rig itself explains worst (section 6), so they may be GT problems rather than search failures; they cannot be
  told apart without a second look at the labels.
- Abstentions are mostly small-support frames (partial penalty-box views where players' kit dominates the detected line
  length) and a near-tie.
- Using the best hypothesis of every frame as if it were answered, thresholds trade coverage for tails: support >= 0.2
  and margin >= 0.15 gives 53 of 67 frames, p90 0.56 m, one > 5 m; margin >= 0.25 removes all of the > 5 m cases at
  50-55 of 67 coverage. These thresholds are fitted on the test frames and are shown only to show the trade-off.
- Central views (halfway line + circle + far touchline) such as `seg010:150`, `seg010:0` and `seg010:25` are solved by
  the final version (errors 0.05-0.65 m); the first version failed them (all abstained), see section 5.
- Synthetic frames with 15% of the line pixels dropped and 8 clutter lines: 7 of 8 answered, median 0.18 m (C tolerance
  mid), 0.07 m (C hard-fixed).

## 5. Synthetic tests and the honest status of the line search

Synthetic known camera (C = (0.5, -64, 15), k1 = 0.02, roll 0.4 deg; 30 random cameras with pan 55-125, tilt 8-24, f
2000-6000; 4-8 keypoints each with 1.5 px noise):

- Rig fit from 22 frames: C error 0.02 m, k1 0.0228 (true 0.02), roll 0.37 deg (true 0.4).
- 8 held-out frames, prior mean = the fitted rig: (a) PTZ median 0.11 m, free H 12% fail and p95 2.2 m; (b) 2 points
  median 0.19 m (C hard-fixed 0.12), 3 points 0.14 m, free H 100% fail; (d) lines with dropout and clutter 0.18 m, 7 of 8
  answered.
  This separates solver faults from GT faults: on data that satisfies the model the errors are 0.1-0.2 m, so the 0.2-0.3 m seen on the real GT
  is mostly label noise and the pinhole+lens approximation, not the solver.
- Unit tests (`tests/calibration/test_ptz.py`): look-at geometry, project/back-project round trip with lens,
  point solve with a perturbed prior, 2-3 point solves, rig recovery (C, roll, k1), line search recovery and abstention on
  an empty mask.

**Line search design was iterated on these 67 frames.** A first version (2 deg grid scored with a 5-25 px capture range,
hard grass test, f 900-9500) abstained on 57% of frames and gave confident wrong answers on others; a few frames
(`astonvilla_seg020:36`, `mancity_seg020:56`, `seg100:1242`, `seg010:150`, `seg010:0`, `seg020:93`, `seg000:633`, `seg010:575`,
`seg120:796`) were used to debug it. The capture range, the soft grass edge, the f range and the abstain thresholds
were chosen with this feedback. The rig (C, k1, roll) is not tuned on the test frames in the leave-shot/match folds, but
the line-search numbers are optimistic until they are repeated on frames that did not take part in the debugging
(Bournemouth in V2 is the natural held-out set).

## 6. GT frames that disagree with the rig (candidate GT errors)

Largest per-frame reprojection rms in the joint fits (px at 1920 wide; the rig median is 3-4 px):

- Man City: `arsenal_mancity_seg031:328` 22.7 (4 points, accepted Claude proposal), `seg080:87` 21.5, `seg080:52` 18.0,
  `seg010:575` 13.3, `seg110:787` 10.3, `seg010:375` 10.2, `seg060:90` 9.1, `seg010:500` 8.0.
- Villa: `astonvilla_seg050:192` 13.0, `seg110:495` 9.6, `seg050:82` 8.3, `seg020:47` 6.9, `seg020:5` 6.1.

Largest leave-shot-out (a) errors (median / p90, m): `mancity_seg080:87` 1.06 / 3.70, `seg080:52` 1.04 / 3.02,
`astonvilla_seg050:247` 1.29 / 2.65, `astonvilla_seg050:82` 0.78 / 2.11, `mancity_seg010:417` (hand) 0.82 / 1.66,
`seg010:575` 0.72 / 1.60. The two `mancity_seg080` frames (52 and 87) are adjacent in one clip and both disagree by
about 1 m, which is worth a look at the labels (a lens or labelling bias, or two similar bad accepts) before they are used
as training or evaluation data. Of the 18 frames with exactly four correspondences (weakly determined), the rig's (a) median error is 0.18 m (same as the
rest) and only one exceeds 0.5 m, so the weak-determination caveat did not produce many outliers beyond those listed.

19 of the 67 frames decompose to a non-physical camera (pinhole ratio outside 0.8-1.25) as a free homography (ratio outside 0.8-1.25, mostly the 4-point
`seg010` Claude accepts, with C_y between -46 and -94 m); even those agree with the rig to 0.21 m median (p90 0.37 m) over the visible
grid, against 0.18 m for the plausible ones (the "plausible only" tables are within 0.02 m of the "all frames" ones at the median), i.e. the extra freedom of a free
homography is invisible in the metres of the visible region, but it is exactly what makes them unusable for
extrapolation.

## 7. What is not done / limits

- Only the Emirates. Bournemouth (V2) needs the labelled penalty-box frames; V3 (temporal) not started.
- Leave-shot-out uses the clip as the shot group. Clips are 1-3 shots; frames of one clip may include different shots,
  which only makes the test stricter.
- GT is not independent: all but 7 frames are accepted LLM proposals, GT H is a fit of the same labels the PTZ and
  baselines see (circular for free H), and the accepted tolerance is ~0.5 m. No second annotator.
- The principal point is fixed at the image centre and the pitch is assumed 105 x 68 (the GT records say so); neither was
  fitted. k1 is one constant. Roll is shared (no per-frame roll).
- The lines-only search is not compared against TVCalib or a SoccerNet pretrained model; comparison stays a doc note.
- `--train-filter plausible` of the full fold set was not run; only the rig C of the plausible-only fit is reported.
- Line search ablations: `loose` tolerance not run for (d); the abstain thresholds were not tuned on held-out data.
- Plots are plain SVG (the shared venv's matplotlib is an empty namespace package): `Cx_vs_f.svg`, `Cy_vs_f.svg`,
  `Cz_vs_f.svg`, `error_cdf.svg`, `k1_vs_f.svg` in the scratchpad; not visually inspected.
- Tooling: `mcp__knowledge__search` returned "Unknown tool" for the KB-first step in this run, so the KB notes were read
  from `~/knowledge/raw/footy-calibration-m0-2026-11-14.md` directly.
