# Calibration (camera → pitch) — Implementation & Inspection Plan

Status: **PLAN** · Stage §2.3 of [`system_design.md`](../system_design.md) ·
Feeds §2.6 [`projection.md`](projection.md) · KB bead `ft-wn9`

Goal: for every broadcast frame, estimate the image→pitch mapping so that
detection boxes become `(x, y)` in pitch metres. Output lands in the feature store's
source-tagged `calibration` table (see "Feature store" below; the old inline
`frame.homography` columns are deprecated), and the existing `PitchProjector`
logic consumes it.

---

## 0. What already exists (and what doesn't)

| Piece | State |
|---|---|
| `projection/projector.py` — `PitchProjector` (anchor = bottom-centre for players, centroid for ball; NaN on low quality) | Written + unit-tested, never fed a real `H` |
| Feature store `calibration` / `pitch_keypoint` tables, `frame_calibration_best` and `detections_on_pitch` views, GT/proposal importer | Implemented (see below) |
| Broadcast classifier (`classifier.py`) — gates which frames are main-camera views | Implemented |
| Pitch model, line/keypoint detector, calibrator, any inspection UI | **Nothing** |

Two conventions to fix up front (both currently unspecified):

1. **Image coordinates of H.** Detections are *normalised* top-left xywh and
   `PitchProjector` applies `H` straight to normalised anchors. So we define
   `H` as **normalised image coords → pitch metres** (origin centre spot, +x
   along the long axis, per `projection.md` §2). Internally the calibrator
   works in pixels and converts with `H_norm = H_px · diag(W, H_img, 1)`.
2. **Pitch dimensions** per game (`game` table / match metadata), default
   105 × 68. The pitch model is parameterised by them.

---

## 1. Approach

Broadcast main camera = **fixed position, pan/tilt/zoom only**. That gives
a clear progression:

1. **Per-frame homography (8 DOF)** from detected pitch features. Enough for
   ground-plane projection; simplest to get working.
2. **Temporal propagation:** calibrate keyframes robustly, carry `H` between
   them with frame-to-frame motion (feature tracking on pitch pixels,
   masking out players), reset on scene cuts (broadcast classifier / shot
   boundary). Smooth.
3. **Constrained camera model (later):** solve the camera position once per
   game, then per frame only pan/tilt/roll/focal (≤4 DOF). Much more stable
   on close-ups with few visible lines, and it is what later enables ball
   *height* (not just ground shadow).

### Feature detector: pretrained first, own training only if needed

Broadcast pitch calibration is a well-studied task (SoccerNet camera
calibration challenge), with public models trained on exactly this kind of
footage. Options, to be evaluated in M1:

| Option | What it outputs | Fit |
|---|---|---|
| **Pitch-keypoint pose model** (YOLO-pose style, ~30 named pitch keypoints) | Keypoints with ids → direct DLT/RANSAC homography | Easiest to integrate (ultralytics already a dep); weak when few named keypoints are visible |
| **Keypoints + line heatmaps** (SoccerNet-calibration-style HRNet models, e.g. "No Bells Just Whistles", PnLCalib) | Keypoints *and* line intersections/tangent points → H or full camera | Best reported accuracy; heavier, needs GPU for full-clip runs |
| **Classical, known-size features**: white-pixel mask inside pitch → line segments + ellipse fits → identify features (penalty box, goal area, centre circle r=9.15 m, halfway line) by their geometry → solve | Named lines/conics → H, and via plane-based calibration (Zhang) focal length + camera pose | No training data and fully inspectable; the hard part is *identifying* which line is which (partial views, symmetric halves, occlusion). Contender for M1 and always used as the **refinement** step |

The pitch model's known dimensions are what make calibration possible in
every option above. The options only differ in how features in the image
get matched to their counterparts in that model: hand-written geometric
rules or a learned detector.

Plan: wrap whichever pretrained model wins behind a `PitchFeatureDetector`
interface; add classical line-snapping refinement on top. Licences and
weights availability are checked before adopting any of them. Fine-tune on
our own frames only if M1 numbers say so (lesson from ball detection:
measure before assuming more data helps).

---

## 2. Module layout

```
src/footy_track/calibration/
  pitch_model.py      # canonical pitch: lines, arcs, named keypoints (metres), param'd by dims
  geometry.py         # DLT/RANSAC homography, px<->norm, reprojection error, warp helpers
  detectors/          # PitchFeatureDetector protocol + pretrained impl(s)
  calibrator.py       # single-frame: features -> H + quality (inliers, reproj err)
  refine.py           # snap model lines to observed white pixels
  temporal.py         # keyframes + frame-to-frame propagation + cut reset + smoothing
  camera.py           # (M4) fixed-position PTZ camera model
  store.py            # write calibration / pitch_keypoint / run rows
scripts/calibrate_clip.py   # CLI: clip -> calibration run in the feature store
labeller/calibration.py + web/calibration.html   # inspection page (§4)
```

`calibration_quality` ∈ [0,1] is derived from inlier count + mean
reprojection error so the projector's existing `H_quality < 0.3 → NaN` gate
works unchanged; raw numbers are kept in run diagnostics.

---

## 3. Milestones

| # | Deliverable | Done when |
|---|---|---|
| **M0** | Pitch model + geometry utils + **manual calibration** in the inspection page (click ≥4 pitch points → H). Saves clicks as **calibration GT**. | Overlay of reprojected pitch lines sits on the real lines for hand-clicked frames; ~50–100 GT frames across the `eval_data/clips` |
| **M1** | Single-frame automatic calibrator (pretrained detector + RANSAC + refinement), run on sampled frames | Scored against M0 GT: reprojection error (px), % frames calibrated, % within 5 px; ≥2 detector options compared |
| **M2** | Temporal: keyframes + propagation + cut reset + smoothing; `calibrate_clip.py` writes to the store | Whole clips calibrated; jitter metric low; failures flagged rather than silently wrong |
| **M3** | Projection end-to-end: store `H` → `PitchProjector` → pitch positions; minimap in the page | Players project inside pitch bounds; plausible speeds (< ~10 m/s); formation shapes look right |
| **M4** | Fixed-position camera model; optional fine-tune on own GT | Better accuracy on hard frames (close-ups, few lines) vs M2 |

Each milestone is its own PR, with tests, against `main`.

---

## 4. Inspection plan — how you check each step

One page, `/calibration` in the existing labeller FastAPI app (same hub,
same Konva frontend patterns; per the KB, keep overlays as siblings of the
Konva stage container). It grows panel by panel with the milestones:

| Panel | Shows | Catches | From |
|---|---|---|---|
| **A. Frame + model overlay** | Video frame with the canonical pitch lines reprojected through `H⁻¹`, toggled on/off | The main check. Misalignment shows straight away, e.g. a penalty box drawn in the wrong place | M0 |
| **B. Feature overlay** | Detected keypoints/lines with ids, green = RANSAC inlier, red = outlier | Detector errors vs fitting errors | M1 |
| **C. Bird's-eye warp** | Frame warped to top-down and alpha-blended over a pitch template | Global distortion, wrong-half flips | M0 |
| **D. Minimap** | Top-down pitch with projected players/ball/refs. Hovering a box highlights its dot, and the other way round | Projection/anchor bugs, out-of-bounds players | M3 |
| **E. Timeline strip** | Per-frame quality/reproj-error/keyframe/cut markers under the scrubber, with "jump to worst N" | Temporal drift, bad stretches, missed cuts | M2 |
| **F. Camera-param plots** | Pan/tilt/focal (or H-derived proxies) over time | Jitter and jumps that look fine frame by frame | M2/M4 |
| **G. Manual edit** | Drag points / click correspondences → re-fit, save as GT | Produces GT and fixes bad keyframes | M0 |

Plus a **run comparison** selector (two calibration `run_id`s, overlays in
different colours) and a non-interactive **contact-sheet report** (grid of
worst/median/best overlay thumbnails + metric table per run) generated by
`calibrate_clip.py --report`, so a batch run can be judged in 30 seconds.

### Checkpoints

- **After M0:** open a frame, click 4–6 points (corners of the penalty box,
  centre spot…), confirm panel A lines up across the whole visible pitch and
  panel C looks like a flat pitch. *This validates the pitch model, the
  coordinate conventions and the maths before any ML.* Then label the GT set.
- **After M1:** contact sheet of worst-10 frames per detector + metric table
  vs GT. Decide which detector to adopt and whether fine-tuning is needed.
- **After M2:** scrub 2–3 full clips with overlay on; watch timeline strip
  for red stretches; check params plot is smooth across pans and resets at
  cuts.
- **After M3:** play a clip with the minimap; players should stay in the
  pitch and move smoothly; sanity metrics (out-of-bounds %, max speed) in the
  report.
- **After M4:** same metrics as M1/M2 re-run; must improve on hard frames
  without regressing easy ones.

---

## 5. Risks / open questions

- **Few visible lines** (midfield close-ups): single-frame fails. Mitigated
  by temporal propagation (M2) and the camera model (M4); until then such
  frames get NaN + flagged, never a guessed `H`.
- **Left/right ambiguity**: symmetric pitch, so a half-pitch view can map to
  the wrong end. Use temporal continuity from wider shots, and confirm it
  in panel C.
- **Player anchor** is bottom-centre of the box; occluded feet bias `y`.
  Acceptable for v1; panel D will show how bad it is.
- **Compute**: laptop has no GPU. HRNet-class detectors on keyframes only
  are OK on CPU; full-clip runs go to the WSL GPU box.
- **Pretrained weights licensing**: confirm before adopting.

---

## 6. Feature store

Full spec: [`feature_store.md`](feature_store.md) §3.7.

- **Tables.** `calibration` (one row per game, source, run, frame: `H` as
  `DOUBLE[9]`, normalised image → pitch m, quality, errors, `review_status`,
  `origin_*`) and `pitch_keypoint` (one row per correspondence). `source` ∈
  `hand_label` | `claude` | `classical` | model. Runs are `run` rows with
  `stage=calibration` and `params_json` (model, prompt_version, …).
- **Best calibration.** View `frame_calibration_best`: hand GT (incl.
  accepted/edited proposals copied into GT) > accepted machine > proposed by
  highest score; failed/rejected never win; it reports the winning
  source/run.
- **Projection.** View `detections_on_pitch` projects every detection in SQL
  (same anchors and 0.3 quality gate as `PitchProjector`; NaN when low quality,
  NULL when uncalibrated).
- **Import.** `python -m footy_track.feature_store.ingest_calibration
  --gt-dir <ball_gt_marks> --db <store> [--dry-run]` loads
  `calibration_gt/*.json` and `calibration_proposals/*.json`. Re-import
  replaces; each clip file lands on every `game_id` alias of that clip that has
  ball GT, so calibration and ball marks share `game_id` + `frame_index`.
- **Inline columns.** `frame.homography` / `calibration_quality` /
  `calibration_model_version` are deprecated and unused.

## 7. Pre-labelling with Claude + proposal review

To cut the manual GT effort, `footy_track.calibration.prelabel` proposes calibrations
that a human accepts on a phone (`/calibration/review`).

1. **Candidates (classical, no ML):** HSV grass mask -> white-line mask (top-hat,
   low saturation, blobs removed) -> Hough segments merged into long lines ->
   numbered candidate points: line/line intersections, line ends on lines, RANSAC
   ellipse (centre circle) / line intersections, spots.
2. **Identify:** an `Identifier` maps candidates to pitch keypoint ids. `ClaudeIdentifier`
   sends the numbered render + a reference pitch diagram (keypoint ids written
   on it) and gets structured JSON (view info, labels, optional extra points,
   unusable flag); `MockIdentifier` derives labels from human GT (tests, debugging).
   Prompt is versioned (`PROMPT_VERSION`); token usage is logged in the run params.
3. **Fit + verify:** RANSAC over the labels, refit on inliers, score = fraction of
   reprojected model-line samples (in image, on grass) within a few px of white-line
   pixels. No fit -> `failed` with a reason, never a guessed `H`.
4. **Proposals** go to `<gt dir>/calibration_proposals/<clip>.json` (format: see the
   labeller README §16); accepting writes `calibration_gt` with an `origin`.
5. `python -m footy_track.scripts.prelabel_calibration eval` compares a run with human GT
   (keypoint agreement, mean px / m error, success rate).

Frames are read by `calibration.video.read_frame`: OpenCV 5.0 decodes some clips as
all-black frames ("Cannot convert interlaced to progressive frames"), so it
falls back to the ffmpeg CLI.
