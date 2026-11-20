# footy_track.calibration

Camera -> pitch calibration. Plan: `docs/design/calibration.md`. This package
currently holds milestone M0 (pitch model, geometry, manual calibration GT via
the labeller `/calibration` page).

## Conventions

* **Pitch frame**: metres, origin at the centre spot, `+x` along the long axis
  (left goal at `-L/2`, right goal at `+L/2`), `+y` toward the **far**
  touchline (the one on the opposite side of the pitch from a main-camera
  operator). Default 105 x 68. A top-down diagram therefore draws `+x` right
  and `+y` up.
* **H maps normalised image coords `[0,1]^2` (top-left origin) to pitch
  metres.** `PitchProjector` applies it directly to normalised anchors. The
  sign of `H` is fixed so image points on the visible ground have `w > 0`.
* `pitch_model.PitchModel(length, width)` exposes line segments, sampled arcs
  and circles (`polylines()`) and named keypoints (`keypoints()`); ids such as
  `L_PEN_BOX_TOP_OUTER` are stable API (GT files reference them).
  `L`/`R` = `-x`/`+x` end, `TOP`/`BOTTOM` = `+y`/`-y` side.
* `centre_circle.py` (halfway line / box front x circle -> 2 points + the circle's pole) and `pitch_finders.py` (the
  editor's two finders: `find_centre` fits the centre circle + halfway line *shape* to a homography
  (`centre_fit.py`) and returns the points of that fit, `find_d` fits the whole penalty-area template by brute
  force; see the module docstrings and LAB-1224) place correspondences from the line mask. Pitch width / length
  are inputs, never assumed.
* `centre_fit.py`: the shape-first centre fit. The circle leaves one parameter free (the vanishing point `W` of the
  width axis on the halfway line); it comes from a touchline end (1-D cross ratio), from the ground's fixed camera
  centre (`camera_rigs/{emirates,vitality}.json`, chosen by clip-name prefix, never guessed for an unknown ground)
  or from square pixels + principal point at the image centre (the circular points `E` x `m` on the image of the
  absolute conic). Candidates are scored on the whole template (halfway line, circle, touchlines, box lines) and the
  best consistent one is kept only if its core precision is >= 0.6. Unstable intrinsic roots are abstained on.
* `geometry.py`: `fit_homography` (DLT, optional RANSAC),
  `reprojection_errors` (px and metres), `project_polyline_to_image` (drops
  points behind the camera so lines never wrap), `warp_to_pitch` (top-down
  raster; the same extents are used by the labeller page).

## Lens distortion

Optional one-parameter division-model distortion (`geometry.make_distortion/undistort_norm/
distort_norm`, stored as `distortion: {model, k1, centre, aspect}` beside `H`; missing or `k1 == 0`
is the plain homography). `H` takes undistorted normalised image coords; labels/anchors are
observed coords. Estimated by `refine_homography(fit_k1=True)` or given (`k1=`). Design and where
it is applied: `docs/design/calibration.md` section 8.

## Line refinement and pre-labelling

`refine.py`: `refine_homography(H, pitch, lines_mask, grass, prior_points=..., line_labels=...,
starts=...)` optimises all 8 DOF of `H` so the projected pitch lines/arcs land on a boolean
white-line mask (robust distance-transform cost, coarse-to-fine, numpy LM; returns the input `H`
with `applied=False` and a reason when it cannot improve it safely). It only needs masks and an
`H`, so the manual page can call it too. `prelabel/` (see `docs/design/calibration.md` section 7)
uses it after the coarse fit.

## Claude via the Claude Code CLI (no API key)

`--identifier claude-cli` runs `claude -p` once per frame (stream-json in/out, `--json-schema`,
`--tools ""`, `--setting-sources ""`, no MCP, no session persistence, scratch cwd), default model
alias `opus`, effort `medium`. Same prompt/schema as the API identifier (`build_content`), minus our
`cache_control` (the CLI adds its own; a second one is a 400). Answer is `structured_output` on the
final `result` event; cost/duration land in the run's `params.usage`.

## PTZ V3: per-segment rig estimate (level 1)

`ptz_temporal_segment --estimate-rig` (default off; with the flag off every output is bit-identical to before) frees the shared
block (C xyz, roll; k1 stays fixed at the `--rig-json` value) with a loose prior centred on that JSON (never a GT-derived rig),
solves it jointly with all the segment's per-frame poses (`fit_shared`) and writes `rig_estimate.json` BEFORE the done marker
`calib_full.npz`: `rig` (Cx, Cy, Cz, roll), the 4x4 `cov` (poses marginalised by Schur complement of the IRLS-weighted
Gauss-Newton normal matrix, times max(1, chi2/dof), times an empirical ridge inflation of 36 on the variance; `cov_model` is the
uninflated one), the prior, evidence counts, focal / pan range, cost and a `converged` flag. See
`ptz_temporal.shared_covariance` for what the covariance does and does not cover: it is local, so on the flat C_y / C_z / f ridge a
single segment's estimate scatters well beyond the model sigma (the inflation is a placeholder until it is calibrated against a
bootstrap on real segments), and a start that is several metres off can stall on the ridge (use the multi-start / level-2
consensus, `docs/design/calibration_match_flow.md`). Real-data regression of the flag-off path on the 11 `ptz_v3_weak.md` segments is
still to do.
