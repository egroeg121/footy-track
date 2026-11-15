# PTZ rig, V2: Bournemouth (Vitality) rig and review run `ptz_bmth1`

Date: 2026-11-15. Branch `exp/calibration-ptz`. Proposal: `docs/design/calibration_ptz.md`; V1: `reports/ptz_v1.md`.
Code: `src/footy_track/calibration/ptz_propose.py`, `src/footy_track/scripts/ptz_propose.py`
(`rig`, `loso`, `propose`, `publish`), tests `tests/calibration/test_ptz_propose.py`. No LLM calls; one single-threaded
process at a time (`nice -n 19`). Raw outputs and renders: scratchpad `.../ptz_v2/` (`rig_vitality.json`, `loso.json`,
`renders/`, `sheet_proposed.jpg`).

## Summary

- **The fixed-centre rig is found on Bournemouth too, but the transfer is weaker than on the Emirates.** Vitality rig:
  C = (-0.10, -50.82, 7.84) m (a low camera, as the decomposition predicted: (-0.1, -49.3, 7.3)), roll 0.19 deg,
  k1 = 0.035. Fitted from only 7 GT frames, so the spread is larger: shot-bootstrap std of C = (0.08, 1.63, 0.19) m.
- **Honest error estimate (leave-one-shot-out, 8 GT frames, 3 m grid vs the GT homography):** 3 GT points median 0.36 m
  (p90 0.73), 2 GT points median 0.53 m (p90 8.3, one catastrophic case = the second camera below; without it the worst
  is 0.96 m). **Lines only answers 2 of 8 frames** (0.18 and 0.34 m) and abstains on 6 (3 of those would have been wrong
  by 37-50 m, 3 within 1.5-3.8 m). The V1 thresholds (support >= 0.15, margin >= 0.08) were not retuned.
- **Published run `ptz_bmth1`: 47 frames, 4 `proposed`, 43 `failed` (abstained).** The yield is low: the high, low-contrast
  Bournemouth camera gives weak white-line evidence and most hard frames have too few or collinear LLM labels.
  The proposals are what the tools can honestly claim; abstentions carry a reason.
- **One probable second camera position.** GT frame `seg100:373` (an accepted Claude proposal) is inconsistent with the
  rig (208 px rms with C fixed) but fits at 3-5 px with C = (-36.6, -50.9, 8.3): same y/z, 36 m toward the left
  penalty-box line. That looks like an 18-yard-line camera on the same stand. One frame is not proof. It is excluded from
  the main rig and such frames abstain.

## 1. Vitality rig

GT frames: 8 with >= 4 correspondences: 2 hand-labelled (seg009:29, seg015:0), 2 accepted pinhole refits (seg040:161,
seg070:229) and 4 accepted `bmth_pre1` proposals (seg035:435, seg075:802, seg085:280, seg100:373). Robust joint fit
(`fit_rig_robust`: Cauchy loss as in V1; a frame whose median point residual is > 20 px is excluded, points > 20 px are
dropped; everything removed is listed):

| item | value |
|---|---|
| C (m) | (-0.097, -50.815, 7.843) |
| shot-bootstrap, 30 resamples: std of C / 5-95% of C_y | (0.08, 1.63, 0.19) / -50.9 to -47.3 |
| roll | 0.19 deg (bootstrap std 0.10) |
| k1 | 0.035 (bootstrap std 0.018; the line-mask per-frame k1 has median 0.039, so the lens is consistent) |
| pitch size | 105 x 68 has the lowest cost (136.7); 100 x 64 148.1; 104 x 68 179.2; 105 x 66 151.5; 102 x 66 149.0 |

Per-frame rms (px at 1920 wide): seg009:29 4.9, seg015:0 2.6, seg035:435 9.1, seg040:161 1.7, seg070:229 1.3,
seg075:802 6.4, seg085:280 1.0. Excluded / dropped:

- `seg100:373`: excluded, median residual 225 px. See the second-camera note above. Its 6 labels are self-consistent.
- `seg070:229` point 0 (HALFWAY_BOTTOM): dropped, 96 px. The point sits on the advertising board below the near
  touchline, not on the white line (checked by eye on the overlay). A GT label error.
- `seg035:435` point 3: dropped, 23 px (that frame's CENTRE_CIRCLE_BOTTOM label lies below the image, y > 1, so the
  circle labels are extrapolated).

k1 and C trade off along the usual ridge and 7 frames (most of them centre-circle shots) do not pin them down as well as
the 67 Emirates frames do. The bootstrap spread in C_y (1.6 m) is comparable to the prior tolerance sigma_y = 3 m, which is
why the rig keeps the V1 tolerance (1, 3, 1) m.

Rig files: `src/footy_track/calibration/camera_rigs/vitality.json`, `emirates.json` (joint robust fit of the 67 Emirates GT
frames, 66 used: C = (0.10, -66.58, 15.74), roll -0.13 deg, k1 0.0147, i.e. the V1 numbers). They live next to `shot_model.json` in the package (tracked, small, found by `Path(__file__)`);
`data/` is git-ignored.

## 2. Leave-one-shot-out (honest Bournemouth error)

Rig refit without the clip of the test frame (the 7 / 6 remaining GT frames), then the frame is solved and compared with
the GT homography over a 3 m grid of visible pitch points (the V1 metric). Each GT frame is its own clip. GT caveat as in
V1: the GT is a fit of the same labels (and `seg100:373`, `seg035:435` are accepted LLM proposals), so about 0.3-0.5 m of
these numbers is label noise.

| solver | frames | answered | median (m) | p90 | max |
|---|---|---|---|---|---|
| 2 random GT points (8 draws per frame) | 8 | 8 | 0.53 | 8.3 | 25.5 (seg100:373: other camera) |
| 3 random GT points | 8 | 8 | 0.36 | 0.73 | 0.75 |
| lines only (video white-line mask) | 8 | 2 | 0.26 | 0.32 | 0.34 |

Lines-only per frame (support / margin / outcome / error of the best hypothesis): seg035:435 0.42 / 0.44 / answered / 0.34 m;
seg040:161 0.30 / 0.22 / answered / 0.18 m; seg015:0 -0.48 / 0.05 / low support / 2.3 m; seg075:802 -0.09 / 0.35 / low
support / 3.8 m; seg085:280 0.01 / 0.14 / low support / 1.5 m; seg009:29, seg070:229, seg100:373: low support and wrong
(50, 47, 37 m). So on Bournemouth the line score still separates right from wrong (every wrong best hypothesis has
negative support) but it is too conservative for the weak line masks of this camera: 25% coverage against 88% on
the Emirates. Treat the lines-only path as a second opinion here.

## 3. The hard frames and the review run `ptz_bmth1`

Targets: all 29 failed / rejected `bmth_pre1` records (25 failed, 4 rejected; the 4 accepted ones became GT) and 3 evenly
spaced extra frames for each Bournemouth clip with no GT and no not-broadcast mark (seg010, 021, 030, 031, 050, 060,
110: 18 frames, seg021 has less room). seg005 has a not-broadcast mark and is skipped; so are marked frames.

Per frame (`solve_frame`): consensus over the stored *real* LLM labels (the padded `synthetic` points of old records are
removed: first run counted them as evidence and was wrong), pair solves under the rig, refit on the inliers, line
refinement (kept only if it improves the line score, moves the camera < 2 m and does not move the points off the fit),
plus the lines-only search as a second opinion. Rules (`pick_camera`):

1. >= 4 consensus points within 6 px (or exactly 3 within 4 px), not collinear (spread >= 3 m): accepted whatever the
   line support (the first run abstained on 9/9 consistent points because of negative line support; the line mask of
   this camera is too weak to veto that).
2. Otherwise >= 3 points with line support >= 0.15, or 2 points confirmed by a lines hypothesis.
3. Otherwise the lines-only answer if it passed the V1 support / margin test; otherwise abstain.
4. Abstain when the shot classifier says not main camera (p_main < 0.5) and support < 0.3.

Result: 47 frames, **4 proposed** (seg035:256 lines-only, support 0.32; seg050:17 3 LLM points + lines;
seg060:1190 5 LLM points + lines, a frame the free-H fit got rejected; seg110:175 3 LLM points + lines), **43 failed**:
41 "low support" (almost all: no or too few usable labels, or collinear labels such as halfway line + circle tops, and the
lines-only search not confident), 1 "near-tie" (seg085:165) and 1 "vetoed on visual check".

Visual check (all 5 solved renders looked at, plus 3 abstained ones): 4 look right to within a few tens of px (circle and
halfway line on the white lines in seg060:1190; penalty box and arc 10-25 px off but clearly the right lines in seg050:17
and seg110:175; arc and far touchline right in seg035:256). **1 was wrong and withheld**: seg085:49 (6 LLM points, line
support -0.35) draws the centre circle about 19% smaller than the white circle although the halfway line fits, and the
refinement had moved C to (-0.08, -53.9, 10.6). It is published as `failed` with that reason (the stored
`ptz.vetoed_params` keep its parameters). The abstained frames I opened (seg010:23, :68, ...) are sparse-line views
(a penalty-box corner and a circle arc) that a human can label but that the lines cannot pin down.

Record format (same writer as the pre-labelling pipeline, `prelabel.pipeline` helpers + `proposals.update_proposals`, one
atomic update per clip): H (normalised undistorted image to pitch metres), `distortion` (division model, k1 0.0349),
`pitch_dims`, `image_size`, status `proposed` / `failed`, score = line support clipped to [0, 1] (so negative support
sorts first in a worst-first queue), `view.notes` ("PTZ rig: 5 LLM pts + lines (of 5 labelled); ..."), correspondences =
the consensus LLM points plus up to 10 projected key points marked `synthetic: true` so the editor can show and adjust
them, and a `ptz` block (mode, support, pan / tilt / f / C / k1, p_main). The overlay is computed by the server from H.
Run `ptz_bmth1`, source `ptz-rig`; the store was backed up first to
`~/.cache/footy-calib/proposals_backup_20261115T032806Z_pre_ptz`; no other run was touched.
Verified through `GET /calibration/api/proposals?run=ptz_bmth1&status=all` (47 items: 4 proposed, 43 failed) and a
record fetch (`seg060:1190`): image points of the stored correspondences map through the stored H and distortion to their
pitch metres, and the projected CENTRE_CIRCLE_TOP equals the stored one to 1e-3 in normalised coordinates.

## 4. What this says about the proposal

- The camera-centre hypothesis survives on a second ground with a very different geometry (low camera, 51 m from the
  pitch centre line), C within about 1.5 m of the decomposition estimate.
- On Bournemouth the benefit shows up with labels, not without them: 3-point solves have a median 0.36 m, but lines only
  does not generalise from the Emirates settings (25% coverage). A rig per ground is cheap; tuned line scoring for
  low-contrast broadcasts is not solved.
- Frames from a second camera position (the probable 18-yard camera) break the single-C assumption. A per-ground set of
  rigs (C_x in {-36, 0, +36} m?) is the obvious extension; one frame is not enough to fit it. A shot where the rig fit
  has 200 px of residual with >= 4 consistent labels is a candidate for it.

## 5. Not done / limits

- A rig fit from 7 GT frames: C_y and k1 are loose. More hand-labelled central / penalty-box frames would tighten them.
- The abstention thresholds (support 0.15, margin 0.08, strong-points 4 within 6 px / 3 within 4 px, spread 3 m) are
  judgement calls and were set while looking at these frames; the 4 proposals are unreviewed by a human.
- No free-x fallback for off-centre cameras; frames like `seg100:111` (4 consistent labels, rms above the strong limit)
  abstain.
- Tests: `tests/calibration/test_ptz_propose.py` (frame selection, point consensus with wrong labels, robust rig fit
  excluding another camera and a bad point, the decision rules, H / distortion round trip of a record, additive writer).
- The root disk of the mini was at 99% during the run (1.4 GB free) and truncated a mask cache file once
  (`np.savez_compressed`: "No space left on device"); the script now recomputes a corrupt cache entry and resumes from
  `records.json`.

## Line evidence fix (2026-11-15, Bournemouth lines-only)

Question: why did the lines-only path answer 2 of 8 held-out Bournemouth GT frames (Emirates 88%), and why was line
support "strongly negative even on correct fits"? Diagnosis by rendering the white-line mask, the distance transform
and the GT-projected pitch lines for the 8 Bournemouth and 5 Emirates GT frames (renders:
`/mnt/storage/footy_data/scratch/ptz_diag/{bmth_h_*,emi_*,new_bmth_h_*}.jpg`, top = frame with mask in red and grass hull
in cyan, bottom = DT with the GT lines coloured by their support).

### Diagnosis

1. **The mask misses the Vitality lines because of an absolute saturation gate, not because the lines are faint.**
   `candidates.line_mask` keeps pixels with `S <= 100` (the Emirates value). The Vitality turf is far more saturated
   (median S 226-234 against 138-204 on the Emirates), and the thin, compressed lines pick up chroma from it: at the
   best pixel of each GT-projected line the median S is 68-136 (> 100 for 17-48% of the pixels in the weak frames,
   `seg009:29` only 17% pass the gate). The brightness cue itself is strong: the top-hat of V on the GT lines has a median of 31-69
   and passes the 16-level threshold for 96-100% of line pixels (Emirates: 90-100%), and the line-to-turf contrast (d')
   is equal to the Emirates one. In the renders the mask keeps the halfway line (near-vertical, wide, strongly tinted white) and the
   board edge, and drops the centre circle and the penalty-box lines except where they run towards the camera.
   By contrast the Emirates masks contain whole circles and boxes.
2. **The mask carries clutter and the old score is dominated by it.** Player legs, socks and shorts survive the blob
   opening and make up a third to half of the detected line length on the sparse frames. The score
   `(unexplained predicted + unexplained detected) / detected length` normalises by that *detected* length, so a correct
   camera whose circle and box lines are missing from a short mask gets "predicted unexplained" lengths of 1.5-2x the detected
   length: support -1.6 (`seg009:29`) and -1.1 (`seg070:229`) under the point-solved camera, down to -4 in V2. A mask
   with clutter and one clear line (the halfway line) looks mild on the same measure. The sign of this is therefore
   a property of the normalisation, not of the camera.
3. **Checked and not the cause.** k1: the sign is consistent (GT k1 0.02 on Bournemouth, rig 0.035, same `distort_norm` for
   the model and the GT projection; the model lines sit on the white lines in the GT overlays). Pixel scale: everything is
   done at 960 px (`TAU_PX`, `SEARCH_W`); the matched fraction of the model length under the GT camera is 0.46-0.90 vs 0.53-0.76 on
   the Emirates once the mask is fixed. Far-line sample spacing: segments are length-weighted, no sign of a bias.
   Visibility / grass hull: the one clear failure is a GT frame (`seg070:229`) whose near-touchline label lies on the
   advertising board (GT error, 96 px) so the predicted near touchline is "off grass".
   Remaining failures of the search are on frames where the rig prior is the problem, not the evidence: `seg075:802` wants C_y = -44.8
   (rig prior -51 with sigma 3), the lines search stays at the prior and finds a hypothesis 4.4 m off, and `seg100:373`
   is the second camera position.

### What changed

- `candidates.line_mask(..., sat_rel=LINE_SAT_REL)` (new, opt-in; the pre-labelling pipeline keeps the old behaviour): the
  saturation limit becomes `max(100, 0.65 x median saturation of the grass)`. It is a relaxation only (0.65 x 155 ~ 100 reproduces
  the Emirates limit; a first version that replaced the limit by `0.65 x S` was *stricter* than 100 on darker Emirates
  frames and lost 3 answers there, so it was made one-sided before the numbers below).
  `ptz_propose.frame_masks` and `ptz_validate.frame_evidence` use it (`--fixed-mask` reproduces V1).
- `ptz.explained_f1(matched, visible, detected) = 2 matched / (visible + detected)` and `ptz.line_support(cams, ev, pitch)`:
  support as the F1 of recall (matched / expected visible model length) and precision (matched / detected length), bounded
  in [0, 1] (the same numerator as V1's score, normalised by both lengths). `search_lines(min_f1=...)` gates on it and always reports `info["f1"]`;
  hypothesis ranking and the margin test are the V1 score unchanged.
- `ptz_propose`: `MIN_SUPPORT = 0.55` (F1 scale), `MAIN_SUPPORT = 0.60`, two unconfirmed labels need `MIN_SUPPORT + 0.15`; the
  points candidate's support is the F1 too. **Threshold chosen on the Emirates only**: in the leave-shot-out lines-only run on
  the 67 Emirates frames every best hypothesis within 1 m has F1 >= 0.61 and every wrong one is <= 0.48
  (one correct frame at 0.46), the threshold sits in that empty gap; Bournemouth was not used to set it.
- Strong consistent labels (>= 4 points within 6 px, 3 within 4 px) still win over the lines, but no longer over a
  lines-only answer that is better supported by more than 0.15 (labels can be consistent with each other and wrong:
  `seg085:49`, six LLM points on a circle drawn about 20% too small, which V2 caught only by eye).
- Tests: `tests/calibration/test_ptz_evidence.py` (relative saturation gate on synthetic turf, `explained_f1` bounds and
  symmetry, F1 stays positive and ranks the true camera above a wrong one on a halfway-line-only mask), updated
  `test_pick_camera_rules`.
- Dev aids: `ptz_propose propose --frame-set-run RUN --label-run RUN2` (re-solve the exact frame set of an earlier run, labels from the
  original pre-label run), `loso --fixed-mask`, `ptz_validate lines --fixed-mask`, `f1` recorded in `lines.json`.

### Emirates (V1 lines-only experiment `ptz_validate lines --variants mid`, leave-shot-out, 67 frames; same code, mask is the only difference between rows 1 and 2)

| setting | answered | median (m) | p90 | p95 | wrong > 2 m / > 5 m answers |
|---|---|---|---|---|---|
| V1 mask, V1 rule (support >= 0.15, margin >= 0.08) | 59 (88%) | 0.25 | 0.66 | 1.05 | 3 / 2 |
| adaptive mask, V1 rule | 61 (91%) | 0.24 | 0.70 | 0.89 | 3 / 2 |
| adaptive mask, F1 >= 0.55, margin >= 0.08 | 60 (90%) | 0.24 | 0.65 | 0.80 | 1 / 0 |

Leave-match-out (Man City rig -> Villa, 19 frames): 14 answered (74%, median 0.43) before, 19 (100%, 0.45, p95 0.87) after. No Emirates
regression; the F1 rule gives the cleaner tail (but its threshold was set on this same set: it is the calibration set, the
Bournemouth numbers are the test).

### Bournemouth (leave-one-shot-out, 8 GT frames, 3 m grid error vs the GT H, rig refit without the clip)

| solver | answered | median (m) | p90 | max | note |
|---|---|---|---|---|---|
| 2 GT points | 8 / 8 | 0.53 | 8.3 | 25.5 | unchanged: does not use the evidence |
| 3 GT points | 8 / 8 | 0.36 | 0.73 | 0.75 | unchanged |
| lines only, before (V1 mask, support >= 0.15) | 2 / 8 | 0.26 | 0.32 | 0.34 | |
| lines only, after (adaptive mask, F1 >= 0.55) | **5 / 8** | 0.33 | 1.96 | 2.66 | errors 0.13, 0.18, 0.33, 0.93, 2.66 m |

The 3 abstentions: `seg100:373` (second camera, 38 m off: right to abstain), `seg075:802` (best hypothesis 4.4 m off: right), `seg070:229`
(best hypothesis 0.89 m, F1 0.45: a wrong abstention, its GT has the bad touchline label). The weakest answers are `seg015:0` (F1 0.55, 2.7 m) and
`seg009:29` (F1 0.56, 0.9 m): both sit just above the threshold; 8 frames cannot support a better one. All 3 wrong best hypotheses
(0.41-0.46) are below 0.55, so the Emirates-set threshold also separates here, with a margin of 0.09 on the lower side.

### Dry run on the `ptz_bmth1` frame set (47 frames, same rig; nothing written to the store)

`ptz_propose propose --frame-set-run ptz_bmth1 --label-run bmth_pre1 --dry-run`: **17 of 47 answered (was 4)**: 12 lines-only
(one of them `seg085:49`, see below) and 5 "k LLM pts + lines"; 30 abstain. I opened all 17 renders next to
the raw frames (`sheets/pair_*.jpg`: top raw frame, bottom overlay): **all 17 look right**. The 5 penalty-box views (`seg050:17/27/49/87`,
`seg110:175`) have their box front line 10-20 px (at 800 px width) above the white line, about 1-2 m; the arc and box corners are on
their lines. The centre-circle / halfway views (`seg060:700/708/1167/1190`, `seg085:165`, `seg035:256`), the two arcs + box
views `seg010:38`, `seg030:205/213`, `seg110:290` and `seg031:59` are within a few px (`seg031:59`: circle edge ~13 px at the
bottom right). `seg085:49` was wrong with the 6-point rule (circle 20% too small, the frame V2 vetoed by hand) and is right
since the override above. The 30 abstentions were not individually inspected beyond their reasons (mostly no labels plus F1 < 0.55: sparse
single-arc views).

### Limits

- The 5/8 and 17/47 are on the frames that were looked at while diagnosing; the F1 threshold was set on the Emirates only, but
  the saturation rule was derived from Bournemouth line pixels, so Bournemouth is not an untouched test any more. A
  third ground would be.
- Clutter (players' kit) is still in the mask; F1 down-weights its effect but a player-aware mask would help `seg070`-like
  frames. Lines that are legitimately invisible (near touchline at the apron) still cost recall.
- The search stays on the rig prior in C; frames needing C_y 6 m away (`seg075:802`) or another camera (`seg100:373`)
  still abstain. The optional free-x fallback for off-centre cameras was not attempted.
