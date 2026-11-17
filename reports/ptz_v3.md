# PTZ rig, V3: temporal calibration (camera motion + many shots)

Date: 2026-11-17. Branch `exp/calib-temporal` (off `exp/calibration-ptz`, main merged in for the finders). Proposal and
earlier results: `docs/design/calibration_ptz.md`, `reports/ptz_v1.md`, `reports/ptz_v2.md`.
Code: `src/footy_track/calibration/ptz_temporal.py` (model + solver), `src/footy_track/scripts/ptz_temporal.py`
(driver: `prep | solve | eval | sheet | jitter | shared`), tests `tests/calibration/test_ptz_temporal.py` (12, all
synthetic). One process, `OMP_NUM_THREADS=1`, `nice -n 19`. Raw outputs, caches and renders:
`/mnt/storage/footy_data/scratch/temporal/` (`<clip>/eval.json`, `shared/shared_mancity.json`, `out/*.jpg|mp4`).

## Summary

- **A temporal solve over a whole broadcast segment reaches the per-frame accuracy floor, answers every frame, and
  removes the confident-wrong lines-only answers.** Arsenal v Man City `seg010` (595 frames, 20 GT frames, Emirates rig
  refitted *without* that clip, 5 m grid, error against the GT homography): temporal solve median 0.25 m / p90 0.49 /
  p95 0.56, 20 of 20 frames answered, none above 1 m. Per-frame lines-only: median 0.23 / p90 0.71 / p95 2.8, 17 of 20
  answered and one confident answer 10.7 m wrong. Per-frame finders: 0.25 / 0.45 / 0.57 on the 17 of 20 frames they answer.
  A second clip (Aston Villa `seg110`, 5 GT frames) is a tie within GT noise (0.47-0.49 vs 0.45 / 0.42), see section 4.
- **The motion constraint is what makes it robust, not what makes it more accurate.** The same solver without the motion
  terms (per-frame line evidence + anchors + smoothness) has the same median (0.24) but 3 of 20 frames 8-12 m wrong
  (the frames where lines-only abstains or is wrong). Motion carries the pose over those frames. Where lines-only is
  fine, motion costs 0.05-0.15 m on Villa (it is a stiff, slightly biased chain): the weight of the motion term is a real
  trade-off (section 5).
- **Few anchors are enough.** Withholding the line evidence *and* the anchor within +-2 frames of every scored GT
  frame (so every scored frame is pure propagation, from 2 anchors in the whole clip on `seg010`) changes nothing
  (0.251 / 0.560 m). One lines-only fit every 50-100 frames is the budget; per-frame lines-only costs ~25 CPU-s a frame,
  the temporal pipeline ~1 CPU-s a frame.
- **The rig can be estimated from the match itself.** Fitting C, roll and k1 jointly over frames sampled from *every*
  Man City broadcast segment (95 confident frames from 162 segments, plus the motion pairs of `seg010`), without any human label,
  gives C = (0.13, -66.42, 15.63) m against the GT-fitted (0.10, -66.58, 15.74): within 0.16 m on every axis. Using that rig
  in the temporal solve gives the same GT error as the GT rig (0.27 / 0.56 m on `seg010`; 0.47 m on Villa, a different match).
  k1 is not agreed (0.032 vs 0.0147 from GT points) and C is not recoverable from a start 5 m off (section 6).
- Not done: pitch dimensions as shared parameters (105 x 68 kept), finder points as anchors in a real run (the
  residual block is implemented and unit-tested only), more than one shot per clip (every evaluated segment turned out
  to be a single shot).

## 1. Method

**Shots.** The units are the broadcast-only segments already cut from each match
(`<match>/split_video_broadcast_frames/segments.json`; `seg010` is `arsenal_mancity_20250925_seg010.mp4`, 595 frames,
the same content as the `eval_clips_kf10` re-encode the GT refers to). Because `merge_gap_s = 0.5` can leave a cut inside a
segment, a cheap internal check remains: colour-histogram jumps (>= 0.60, or >= 0.30 together with a failed motion pair)
split a segment (`segment_shots`), and a failed gap-1 pair ends a motion-connected run. `seg010` has one cut, at frame 593
(the last two frames are dropped); the Villa clip starts with a 7-frame non-main sliver (p_main 0.001, dropped).
Main-camera classification: the existing `shots` per-frame `p_main` (`ptz_propose.main_camera_p`), median over the
segment (0.93 and 0.87 here).

**Inter-frame motion** (`rotzoom_ransac`). Two frames of a camera that rotates and zooms about a fixed centre are related by
`H = K2 R K1^-1`: a rotation vector (3) and a zoom (1), whatever the scene depth. RANSAC runs on exactly this model:
2-point minimal samples solved with the first-order linear model, scored with the exact model, best consensus refitted
(LM, Cauchy) and re-scored; no free 8-DOF homography anywhere. Features are `goodFeaturesToTrack` on the 960 px grey
frame inside a static-background mask (everything but the broadcast overlay and the non-green, non-white blobs inside the pitch,
i.e. players, referees, ball; stands and advertising boards stay in), tracked forward-backward with pyramidal LK; points are
undistorted with the rig k1 first. Pairs: every frame (gap 1) and every 5th frame (gap 5, tracked from the composed gap-1
motion as initial guess, to pin accumulated drift). On `seg010`: 95% of tracked points are inliers (median 842 of 884),
rms 0.29 canonical px (0.15 px at 960) for gap 1 and 0.55 for gap 5; 711 of 712 pairs accepted (the one failure is the cut).
RANSAC needs a nominal focal length (the flow is `f * dtheta`): interpolated from the lines-only anchors, 3000 px otherwise;
the fitted angle scales with it (unit-tested), which is why the joint solve below re-fits with the solved focal length.

**Joint solve per segment** (`solve_segment`, `joint_solve`, own sparse-Jacobian Levenberg-Marquardt with IRLS Cauchy
weights; dense Cholesky of J'J, ~1800 unknowns for 593 frames). Unknowns: (pan, tilt, log f) per frame, optionally the
shared C, roll, k1. Residuals:
1. motion: the raw inlier correspondences (<= 120 per pair), pushed through `K_j R_j R_i^T K_i^-1` of the *unknown* poses
   (sigma 1 canonical px);
2. line evidence on every 5th frame: truncated distance transform of the white-line mask at the projected pitch markings
   (sigma 3 px at 960), two stages (blurred / trunc 12 px, then sharp / trunc 5 px), as `ptz._refine_lines`;
3. anchors: lines-only fits (`search_lines`, V2 settings, F1 >= 0.55 and margin) re-fitted at the rig's C, as soft priors
   (0.23 deg, 4% f); finder points as reprojection residuals (implemented, tested, not used in the runs below);
4. smoothness: second difference of (pan, tilt, log f) (0.09, 0.06 deg, 0.4% per frame^2); a prior on the shared
   quantities.
Initial poses: the nearest anchor carried over the motion pairs with the exact focal length (re-fit per step).
Anchor hygiene: (i) *motion vote*: each anchor is carried to its neighbours and compared in pitch metres, an anchor whose
every verifiable edge disagrees is dropped; (ii) after the first solve an anchor > 1 m from the solved camera is dropped
and the segment re-solved. (ii) was added after `seg010` showed a frame (525) whose lines-only answer was 10.7 m wrong
with F1 0.71 and no neighbouring anchor to vote against it (full variant 1.05 m before, 0.26 m after, which is the one
change tuned after seeing GT; the Villa clip was not affected).

**Shared parameters over many shots** (`fit_shared`). Bundle adjustment: one block of (pan, tilt, log f) per sampled
frame, one shared (C, roll, k1) with a loose prior (5, 10, 3 m, 2 deg, 0.05) seeing all the line evidence and the motion
pairs of the evaluated segment (gap-5 pairs). Frames: the middle frame of each of the 162 Man City segments, `p_main`
>= 0.5, lighter lines-only search (3.6x smaller grid): 95 confident frames. (The brief asked for a few frames per segment;
one per segment was all the CPU allowed, section 8.)

## 2. Evaluation set-up

Clip `arsenal_mancity_seg010`: 20 GT frames (`calibration_gt`), 5 m grid of visible pitch points mapped to the image by the
GT homography and back by the estimate; per frame the median over the grid, then median / p90 / p95 over frames. Rig: Emirates
joint fit **without** this clip's GT frames (47 frames): C = (0.099, -66.488, 15.779) m, roll -0.07 deg, k1 0.0140 (all 67:
0.100, -66.576, 15.743, -0.13 deg, 0.0147). No GT label is used by any solve; line evidence and anchors at the scored frame
are image-derived. Variants: `full` (everything), `hold` (line evidence and anchors withheld within +-2 frames of every
GT frame), `nomotion` (same solver, no motion terms), `est` / `esthold` (rig quantities from section 1's shared fit).
(a) = `search_lines` (C free inside its prior, the V2 answer), (b) = the two finders (centre circle, penalty-area template; points ->
`solve_points`).

## 3. Results on `arsenal_mancity_seg010` (20 GT frames)

| method | answered | median (m) | p90 | p95 | > 5 m |
|---|---|---|---|---|---|
| (a) per-frame lines-only (`search_lines`) | 17 / 20 | 0.23 | 0.71 | 2.81 | 1 |
| (b) per-frame finders (centre + D template) | 17 / 20 | 0.25 | 0.45 | 0.57 | 0 |
| (c) temporal joint solve, `full` | 20 / 20 | 0.25 | 0.49 | 0.56 | 0 |
| (c) `hold` (scored frames: propagation only) | 20 / 20 | 0.25 | 0.49 | 0.56 | 0 |
| (c) with the estimated rig (`est` / `esthold`) | 20 / 20 | 0.27 / 0.27 | 0.49 / 0.47 | 0.56 / 0.57 | 0 |
| ablation: same solver, no motion terms | 20 / 20 | 0.24 | 7.9 | 9.0 | 3 |

(a) and (b) are measured on the frames they answer; the frames lines-only abstains on (375, 500, 575) are those where its best hypothesis is
82-100 m wrong (so the abstention is right). Finders find the centre circle in most of this clip, which is why they are
good here and not representative of frames without a circle or box in view. The temporal errors at the
three frames lines-only abstains on are 0.36-0.44 m; per-frame errors are in `eval.json`. The 0.25 m floor equals the V1
"all GT points, C hard-fixed" figure (0.25): the remaining error is the fixed-C model and GT noise, not the method.

**Coverage** (main-camera frames of the segment with an accepted calibration):

| | `seg010` | Villa `seg110` |
|---|---|---|
| temporal: frames with a solved camera | 100% (593 / 593) | 100% |
| temporal: accepted (line F1 of the solved camera >= 0.5 on the evidence frames on both sides) | 92% | 91% |
| (a) lines-only accepted, frames searched (every 50th / 100th) | 83% (10 / 12) | 93% (14 / 15) |
| (a) lines-only accepted on the scored GT frames | 85% (17 / 20) | 100% |
| (b) finders answered, scored GT frames | 85% (17 / 20) | 100% |

The 8% not accepted on `seg010` are the last 100 frames (zoomed, partly off-pitch views); they are calibrated, and the three GT frames
there are 0.36-0.44 m, but no line evidence confirms them, so they are reported as unconfirmed.

**Temporal jitter** (second difference of the projected pitch centre, pixels at 1920; a smooth pan gives ~0). The brief
asked for the frame-to-frame jump when the camera is near-static; this clip is almost never static (83 of 593 pairs move < 1 px
at 960, median flow of those 1.2 px at 1920), so the jump there is just the true motion (1.23 px) and the second difference
is reported instead:

| | median | p95 | max |
|---|---|---|---|
| temporal, all 593 frames | 0.30 | 1.35 | 3.9 |
| same solver, no motion terms (anchors and evidence every 5th frame only) | 0.37 | 3.1 | 14.0 |
| 9 consecutive frames (101-109, flow 0.8 px/frame): temporal | 0.35 | | 0.38 |
| same 9 frames, per-frame lines-only (`search_lines` every frame) | 0.89 | | 2.74 |

## 4. Second clip: Aston Villa `seg110` (721 frames, 1280 x 720, 5 GT frames)

Leave-clip-out rig (62 frames); lines-only anchors every 100 frames + GT frames. Error median / p90 over 5 frames:
(a) 0.45 / 0.51, (b) 0.42 / 0.59, (c) `full` 0.49 / 0.55, `hold` 0.50 / 0.56, with the rig estimated from Man City segments
(`est`) 0.47 / 0.52 (a rig transfer across matches), no-motion 0.37 / 0.47. Five frames cannot separate these; the
consistent picture is that on a clip where per-frame lines-only works (14 of 15 searched frames accepted) the temporal
solve is ~0.05-0.1 m worse than the no-motion solve at the median and has the same coverage, and jitter 0.26 px median
(p95 1.1, one 8 px glitch) vs 0.33 / 2.1. Here the weaker the motion weight, the better (sigma_motion 3: 0.48, 10: 0.42).

## 5. Motion weight (disclosed: swept on the scored clips)

| sigma_motion (canonical px) | `seg010` median / p95 / > 5 m | Villa median |
|---|---|---|
| 1 (default) | 0.25 / 0.56 / 0 | 0.49 |
| 3 | 0.25 / 0.85 / 0 | 0.48 |
| 10 | 0.26 / 5.2 / 1 | 0.42 |
| none | 0.24 / 9.0 / 3 | 0.37 |

Line weight (sigma 1.5 instead of 3, i.e. stronger lines): `seg010` p95 4.7 m / 1 gross, Villa 0.48. sigma_motion = 1 is
the setting with no gross failure; the price is a small motion bias on Villa (zoom: the solved f differs from the no-motion
one by up to 2%, pan by 0.1 deg; V1's zoom test found a possible C shift of 1.5 m per zoom doubling, which a fixed-C
rotation chain cannot represent and which would accumulate along the chain). Not investigated further.

## 6. Shared parameters vs the rig JSON (Emirates, Man City)

| source | C (m) | roll (deg) | k1 |
|---|---|---|---|
| `camera_rigs/emirates.json` (67 GT frames, points) | 0.100, -66.576, 15.743 | -0.13 | 0.0147 |
| leave-`seg010`-out GT rig (47 frames) | 0.099, -66.488, 15.779 | -0.07 | 0.0140 |
| **lines only, 95 frames from 162 segments, no labels** | 0.217, -66.500, 15.670 | -0.17 | 0.027 |
| **+ motion pairs of `seg010`** | 0.134, -66.422, 15.626 | -0.14 | 0.032 |
| lines only, 100 its, start C + (-2, -4, +2), k1 0.05, roll 1 deg | 0.001, -66.805, 15.787 | -0.18 | 0.028 |
| lines only, 100 its, start C + (2, 5, -2), k1 0, roll 0 | 2.33, -62.90, 14.40 (stuck, cost +16%) | -0.41 | 0.034 |

- C from the match's own line evidence lands within 0.12 / 0.16 / 0.12 m of the GT-points rig on every axis: the
  fixed-centre rig is recoverable without labels, and using it costs nothing in pitch error (sections 3 and 4).
- k1 does not agree: 0.027-0.034 from lines (and from lines + rotation consistency) against 0.014-0.015 from GT points.
  V1's per-frame line-mask k1 was also larger (median 0.022). The lens is not pinned by either; the effect on pitch
  error is below the noise here. Not resolved.
- The fit is not globally convergent: from a start 2-5 m off it reached the right basin once of two; the bad basin
  has a visibly higher cost (61.8k vs 53.2k from the good start and 58.9k for the other perturbed start), so run it from several starts and keep the lowest cost,
  or from the previous rig. The bundle uses init poses found with the GT rig, which only helps.
- Pitch dimensions (L, W) are not estimated (105 x 68 fixed; V1 / V2 found it best among the alternatives).

## 7. Runtime (CPU seconds, one thread; wall time was 3-10x worse while the mini carried a load average of 40-50)

| stage | per frame |
|---|---|
| decode + grey / static mask / thumbnails (+ line + grass masks and p_main every 5th frame) | 0.30 |
| motion: gap-1 and gap-5 pairs, 900 features, forward-backward LK, constrained RANSAC | 0.34 (0.29 per pair) |
| joint solve (593 frames, 2 stages, ~25 LM iterations) | 0.1-0.2 |
| lines-only anchor (`search_lines`) every 50th-100th frame, 15-30 s each | 0.2-0.5 |
| **temporal pipeline** | **~1** |
| per-frame lines-only on every frame | 15-30 |

About 25x cheaper than per-frame lines-only. The 162-segment sample for the shared fit took ~25 CPU-minutes (about 9 s per sampled
frame: seek in segments with a keyframe every 10 s, masks, lighter lines-only search).

## 8. What was not done, and why

- Only the first 95 confident frames (one per segment) feed the shared fit; the brief's "a few per segment" and Villa
  / Bournemouth segments were left out for CPU. Per-frame search is the cost (10-30 CPU-s).
- No shot with more than one camera cut was evaluated; the cut handling is covered by the unit test and by the two
  real cuts found (frame 593 of `seg010`, a sliver at the start of Villa `seg110`).
- Finder points as anchors in a real run, pitch dimensions, `k1(f)`, a C that drifts along the optical axis with zoom.
- The coverage criterion (F1 of the solved camera on the line evidence >= 0.5, V2's threshold 0.55 relaxed because the
  solved camera is not selected on it) is mine, not validated against GT outside the scored frames.

## 9. Recommendation

Adopt the temporal solve as the way to calibrate a whole broadcast segment: anchors every ~50-100 frames from the
existing lines-only search and the finders (the finder residual is already in the solver), constrained-model motion between them,
joint solve with the rig fixed. It matches the best per-frame accuracy on the frames those methods answer, answers
every frame including the ones they abstain on, drops confident-wrong anchors, is smoother, and costs ~1 CPU-s per
frame instead of ~25. Keep sigma_motion = 1 (no gross failures) and treat 0.05-0.1 m of motion bias as the price.
Before a full-match run:
1. fix the shared fit's convergence (multi-start, lowest cost) and decide what to do about k1 (fit it per ground from
   rotation-only consistency, or keep the GT-points value);
2. add finder points as anchors and run them on a segment where lines-only abstains for long (zoomed play, a Bournemouth segment with the
   high camera, where C may also differ);
3. a full Man City run is ~98k broadcast frames x ~1 CPU-s = ~27 CPU-hours: not for the shared mini. Run it on the
   desktop (Prefect `cpu` pool workers there, one job per segment: anchors every 100 frames + motion + solve; the lines-only
   search is the only heavy part and needs no GPU), writing per-frame (pan, tilt, f) + rig to the feature store.
