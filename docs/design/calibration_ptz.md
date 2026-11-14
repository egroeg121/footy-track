# Proposal: calibrate the camera rig once, then solve each frame for pan / tilt / zoom only

Status: **proposal, revised 2026-11-14 after review** (gpt-6.1-sol critique). V1 offline validation is the first step;
nothing else is built until it passes.

Decisions taken 2026-11-14:
- Run V1 first (offline, no LLM calls). Stop if it fails.
- Stop further LLM pre-labelling runs for now.
- The user will label 2-3 Bournemouth penalty-box frames for the Vitality rig.
- Pretrained SoccerNet calibration detectors are being researched (no install yet); see section 8.

## 1. Why the current approach struggles

Today every frame is calibrated on its own. It gets a free 8-DOF homography H (image → pitch), from LLM point labels
plus line refinement, and optionally a lens k1. That works when a frame shows many points that are spread out and not
in a line, such as a penalty box with the goal area. It breaks down in the **middle third of the pitch**. There the
visible markings are the halfway line, the centre circle and one or two touchlines: the points are nearly in a line,
the circle is a conic that the LLM labels badly, and the far touchline is thin and often confused with the hoarding base.
Eight unknowns cannot be pinned down from that. The fitter then drifts into homographies that are not compatible with a
square-pixel, centred-principal-point pinhole camera (the review notes this only shows incompatibility with those
assumed intrinsics; a wrong principal point, uncorrected distortion or a rescaled feed could also explain it).

The review results show this:
- Arsenal–Bournemouth: 0 accepted, 12 rejected, 37 failed.
- Aston Villa: 12 accepted, 23 rejected.
- Of the rejected Bournemouth fits, many decompose to non-square pixels, |r1|/|r2| from 1.2 to 5.7 (an ideal square-pixel
  pinhole gives 1).
- 21 of the 49 Bournemouth records had only 1–3 LLM labels, so they were starved of points.

## 2. The key fact: the main broadcast camera does not move, it only pans, tilts and zooms

The working hypothesis is that the main camera sits on a fixed head on the gantry. Between frames only **pan, tilt and
zoom** change, plus a roll that we model as shared but must verify. Its **centre of projection C is then approximately
fixed** for a whole match, and possibly for other matches at the same ground. This is a hypothesis to test, not a fact:
the table below shows consistency of a few selected decompositions, not proof of constancy.

We checked this on our own labels by decomposing every hand or accepted calibration into a pinhole camera (square
pixels, centred principal point):

| Ground | Frames (plausible camera) | Camera centre C, median (x, y, z) m | IQR (x, y, z) m | Focal length range |
|---|---|---|---|---|
| Emirates, Man City match | 33 | (0.0, −65.2, 15.2) | (1.4, 3.9, 0.8) | 2700–5500 px (2× zoom) |
| Emirates, Villa match | 11 | (0.2, −64.3, 15.0) | (1.9, 5.0, 0.9) | 1780–2890 px |
| Vitality (Bournemouth) | 4 | (−0.1, −49.3, 7.3) | (0.8, 5.9, 0.1) | 1890–4180 px |

Coordinates are in metres from the centre spot: x along the pitch, y toward the far touchline (so negative y is
behind the near touchline), z up.

What the table shows:
- **C is consistent across the frames of a match** to within a few metres in y, even though the zoom doubles. The two
  Emirates matches give similar medians. Shared modelling assumptions (centred principal point, no lens model, pitch
  dimensions) could produce a shared bias, so this is evidence for the hypothesis, not proof.
- The Bournemouth row rests on only 4 frames and cannot establish a fixed C by itself.
- **Bournemouth's camera is very different:** 49 m back and only **7 m up**, against 65 m and 15 m at the Emirates. That
  low angle squashes the far half of the pitch, which is exactly where the labels fail. It also means the default
  camera prior in `refine.CameraPrior`, (0, −65, 15), is wrong for Bournemouth.
- The spread in y is probably the usual depth-focal ambiguity of one frame. A joint fit across frames should reduce it;
  V1 measures this with bootstrap uncertainty rather than assuming it.

**A note on "extract the intrinsics".** The focal length is *not* constant; it changes with zoom. What carries over from
a well-labelled frame is:
- the **camera centre C**;
- the principal point and pixel aspect, which are fixed by the camera;
- the mount roll;
- the lens distortion **as a function of zoom**, k1(f).

The per-frame unknowns are then **pan, tilt and focal length** (parametrised as log f), and maybe a small roll: 3-4 DOF
instead of 8.

Mechanical definition (to be pinned down in code): rotation R = Rz(roll) Rx(tilt) Rz(pan) applied to a fixed base
orientation, intrinsics in normalised image coordinates [0,1] with the principal point at the image centre unless the
rig fit says otherwise, distortion applied in the division model before the pinhole. A tilted pan axis does not in general
give a constant image roll through a pan; V1 checks whether a shared roll is adequate or roll must be per frame (small,
regularised). Real heads can have offset pivots and non-intersecting axes, so C is allowed to differ slightly from frame
to frame (see "C as a prior" below).

## 3. Proposal

### A. Calibrate the camera rig once per ground (and check it per match)
- Pick **3-6 "rich" frames** from the main camera: penalty-box views and wide views with at least 6 points that are not in
  a line and span depth. Select them for **diversity of pan, tilt, zoom and pitch coverage**, not just point count. Label
  them carefully in the existing editor. These are the **only** frames that need full point labelling.
- Fit them all jointly (bundle adjustment):
  - shared C, shared roll offset and a **constant k1** to start with;
  - per frame, the rotation and log f.
- **k1 as a function of focal length** (the earlier idea k1(f) = a + b/f) has no demonstrated physical justification.
  Start with constant k1 and add zoom dependence only if the residuals show a trend against f.
- **C is a prior with tolerances, not hard-fixed.** In per-frame solves C enters as a Gaussian prior (sigma set from the
  rig fit, for example a few metres in y and under 1 m in x and z), so a frame that really needs a different C can
  have it at a visible cost. Do not assume a rig transfers to another match at the same ground; verify with a handful of
  frames from the new match.
- Report parameter uncertainty (bootstrap over frames and over shots), parameter correlations (C_y against f is the
  expected one), held-out residuals and the per-frame reprojection error. A frame that disagrees is a labelling error, a
  different camera or model error; do not assume which.
- Store the result as `camera_rigs/<ground>.json`. Grounds so far: Emirates (both matches), Vitality.

### B. Solve each frame for pan, tilt and zoom only
With C as a prior, a frame has 3-4 unknowns. Counting alone does not make a solve well conditioned, so the claims below
are what V1 tests, not assumptions:
- **2 point labels** can give a finite solution, though ambiguity and poor conditioning remain (two nearby points mainly
  constrain pan and tilt, hardly f). Points that are close to a line add little. One point plus an *identified* image
  line is different from one point plus a distance-to-line observation.
- **Lines alone.** This is the hard part and must be semantic and multi-hypothesis, not a plain distance-transform match:
  - the search runs over bounded pan x tilt x log f (coarse-to-fine), with C at its prior;
  - each hypothesis is scored on visibility-aware, semantic evidence: predicted line segments that fall in the image are
    matched to detected line pixels (with the direction of the line), and detected line pixels that no predicted line
    explains are penalised too. A one-sided distance-transform score can reward laying a few predicted lines over
    unrelated edges (hoarding, goal frames, crowd), and pitch symmetry gives a competing mirror solution;
  - keep the top K distinct hypotheses (non-maximum suppression in parameter space), refine each with robust LM, and
    **abstain** when the best and second-best are near-tied in score or when line support is too low. A wrong answer
    that looks confident is worse than no answer;
  - LLM labels or a coarse "which half / which box" cue can disambiguate the mirror, as an optional seed.
- Central views may become solvable, but the halfway line alone does not fix pan (it fixes pan only together with
  tilt/f or a second constraint). The circle and the far touchline constrain tilt and zoom. Whether that is enough is
  V1's question.
- LLM labels (Claude or Codex) become an optional **seed**, e.g. "left box, near side", to narrow the search, plus a
  **reviewer**. They are no longer the main source of geometry.

### C. Track the camera between frames with a rotation-only model
- For an ideal, undistorted camera rotating about its optical centre, the motion between two frames is exactly
  H = K2 R K1^-1 for every static point in the scene, at any depth: the pitch, the hoardings and the stand
  architecture. Real PTZ heads need not satisfy this (offset pivots, focus breathing, distortion, stabilisation), and a
  more constrained tracker can do worse than a free homography when its assumptions fail. Undistort before applying the
  rotation model, and use differences between near and far scene depth as a *detector* of departures from pure rotation.
- Track features on the stable stand architecture and the pitch, not on the crowd, flags, LED boards or players (those
  move). Fit the 3-4-DOF inter-frame motion and chain it from anchor frames, preferably with windowed optimisation
  against the anchor rather than relying only on chained transforms.
- Every N frames, or when line support drops, re-snap to the lines with B to stop drift. At a cut back to the main
  camera re-solve with B. Shot-level acceptance by the user does not guarantee later frames stay accurate, so sample
  frames across the shot for review.
- This builds on footy-track-df's `calibration/temporal.py`. Their chaining gives 0.40 m mean on seg010 with a general
  homography; rotation-only is expected to be at least as good, but that is to be shown (V3), not assumed. Test it
  early, not last.

### D. Shots from other cameras
- The shot classifier (`calibration/shots.py`, main / closeup / other) sends only main-camera shots through B and C.
- A recurring second camera, such as an 18-yard-line camera or the high camera behind the goal, can get its own rig
  later by clustering the C values. Close-ups and replays are skipped.

### E. Labelling
- **Rich frames (rig):** the existing point editor, plus a live readout of each frame's implied C against the rig and of
  pixel squareness, so a bad click shows at once.
- **Ordinary frames:** a new *PTZ alignment* mode. The pitch overlay is drawn from the rig. The user drags to pan or
  tilt and scrolls to zoom, and an "auto-snap" button refines against the lines. Typical effort is a drag and a snap;
  alternatively click any 2 points.
- **Review page:** B and C generate proposals for whole shots and the user accepts or rejects as now. One verdict on a
  shot anchor stands for the whole chained shot.

## 4. Validation before building UI (offline, no LLM cost, 1 CPU process)

1. **V1, leave-match-out and leave-shot-out on the Emirates.** There are 44 plausible GT frames over the two matches.
   Leave-one-frame-out is not used: neighbouring frames of the same shot leak. Splits are:
   - leave-match-out: fit the rig on the Man City frames, test on Villa (and the reverse as a check);
   - leave-shot-out within each match: frames grouped by clip/segment, never split across train and test.
   For each test frame, solve with C as a prior from:
   - (a) all its GT points;
   - (b) random 2-point and 3-point subsets, many draws;
   - (c) near-collinear subsets (halfway line + centre-circle points only, or one line's points + one other), to mimic
     central views;
   - (d) lines only: the semantic multi-hypothesis search of section 3B on the actual video frame, with abstention.
   Baselines: free 8-DOF homography and the existing 7-DOF pinhole fit without the C prior, from the same inputs. Where
   they are under-determined (fewer than 4 points) that counts as a failure, and we state it, rather than inventing a
   regularisation that flatters either side. A comparison against TVCalib-style segment-based optimisation is in
   section 8.
   - Metric: pitch-metre error against the GT homography over a grid of visible pitch points. Report median, p90, p95,
     failure/abstain rate (and accepted coverage), split near half and far half, and by zoom (f). Also pixel
     reprojection error, and runtime.
   - **Synthetic tests with known ground truth:** generate a known rig, project pitch points and lines, add noise, and
     check that C, f and the angles are recovered, including central views, extreme zooms and near-collinear inputs.
     This separates solver faults from GT faults.
   - **GT caveats.** The GT is itself imperfect: 56 of 70 frames are accepted LLM proposals, 7 pinhole refits and 7
     hand labels; 19 frames have exactly 4 correspondences (fit error near 0, weakly determined); the user accepted some
     with tolerance for curved or pre-lens lines, so GT noise is about 0.5 m. Errors below that are not meaningful, and
     frames where the rig and the GT disagree are listed, since they may be GT errors.
   - **Pass if:** (d) has a median under 1 m on main-camera frames, with p90/p95 and the failure rate reported and not
     catastrophic (a median hides tails), and (b) and (c) are clearly better than free H.
2. **V2, Bournemouth.**
   - Build the Vitality rig from the 4 existing GT frames plus the 2-3 rich penalty-box frames the user will label.
     First validate the fixed-centre assumption independently there (4 frames is too few to claim it).
   - Run B (lines only, plus stored LLM labels as seeds) on the rejected and failed Bournemouth frames.
   - Publish the results as a review run; the user's verdicts measure usability, so show proposals blinded against the
     free-H proposals, and also compute independent geometric errors on any withheld hand labels. Low viewing angles
     amplify pixel errors into metres near the horizon, so report error by distance.
3. **V3, temporal.** Compare rotation-only against general-homography chaining on several shots with different zooms
   and durations (not only seg010), with identical anchors and tracks. Report endpoint and maximum drift, cut recovery
   and failure coverage, against footy-track-df's 0.40 m on seg010.

## 5. Risks and open questions
- **The centre of projection may move with zoom.** The entrance pupil of a broadcast zoom lens can move by up to roughly
  0.5 m. At 45-65 m that is about 1% in depth, but translation along the axis is not generally reproduced by a change of
  f (the effect is depth-dependent), so we do not assume it is absorbed. If V1 shows a trend of C with f, model it as an
  offset along the optical axis as a function of f.
- **Other camera effects:** focus breathing, zoom-dependent principal point and distortion, lens stabilisation, platform
  vibration, offset or non-intersecting pan/tilt axes, and set-up changes between matches. Cropping, digital
  stabilisation, resizing, interlacing, blur and rolling shutter change the effective camera. Per-match verification of
  the rig is part of the plan.
- **The principal point may not be at the image centre** (cropped or rescaled feeds, lens decentring). Fit it per rig,
  not per frame.
- **More than one main camera.** Some broadcasts cut between a high and a low "main" camera at similar angles. The C
  values would form clusters, so handle them as two rigs.
- **Pitch size varies by ground** (Vitality is not necessarily 105 × 68). Wrong dimensions bias C, so fit the
  dimensions in the rig fit or look them up.
- **The rig only covers the main camera.** Other cameras still need the per-frame free fit or are skipped. The shot
  classifier has to be accurate enough.
- **Lines-only search** is the largest unresolved component: pitch symmetry, partial circles, repetitive parallel
  markings, goal frames, hoarding edges and advertising create competing minima, and a near-empty view (only grass and a
  touchline) is ambiguous along the pitch. Mitigations: semantic, multi-hypothesis scoring and abstention (3B).
  Temporal chaining may cover such frames; B alone may not.
- **Low camera (Bournemouth):** at 7 m height small pixel errors become large ground errors near the horizon; camera
  height error, pitch crown/slope and occluded markings matter more there.
- **Ground truth quality:** see the GT caveats under V1.

## 6. What changes for the work in flight
- **Further LLM pre-labelling runs are stopped** (decision 2026-11-14) until V1 reports.
- **LLM pre-labelling** becomes a seed and reviewer step rather than the main source of geometry. The Claude-vs-Codex
  comparison is still useful, re-aimed at "is the seed in the right area?" and "does the reviewer agree with the user?".
- **The Bournemouth pre-label run** (`bmth_pre1`, Claude Opus, 33 calls) is optional under this plan. Its frames can
  serve as seeds, or as candidates for rig frames.
- **Existing GT** stays valid. It is also the training and validation data for V1.

## 7. Order of work
1. Run V1, about half a day of offline compute and code. Stop here if it fails. (Decided 2026-11-14.)
2. Build the Vitality rig. Needs the 2-3 carefully labelled Bournemouth penalty-box frames the user will provide.
3. Run B on Bournemouth as a review run, then get the user's verdicts.
4. Add the PTZ alignment mode to the editor.
5. Do rotation-only temporal chaining (C), with footy-track-df's temporal module. Start a small version of V3 in parallel with step 3, not at the end.

## 8. Comparison with published calibration methods
We should compare against, not replace with a free homography as the only baseline:
- **TVCalib** (segment-based camera optimisation): optimises a per-frame camera (focal, rotation, position) so projected
  pitch line segments match detected segments. Our difference is the shared camera-centre prior and a rig fit across
  frames; the per-frame solve is closely related. If a pretrained detector for the SoccerNet pitch-keypoint/line
  segmentation task is usable, its output can seed our search or serve as an extra baseline.
- **SoccerNet camera-calibration challenge** protocols: they report a completeness-weighted accuracy at a pixel threshold
  and often use a homography-based pipeline with a segmentation of pitch lines and keypoints. Their data lacks per-match
  rigs; ours is the opposite regime (few matches, many frames each). We should report our metrics in a way that can be
  related to theirs (reprojection error in pixels at thresholds, and coverage) as well as in metres.
- Pretrained SoccerNet detectors are being researched (2026-11-14, no install yet). The outcome decides whether lines-only
  initialisation uses a learned detector or our hand-built line search.
