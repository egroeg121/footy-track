# Calibration: production flow, match video to versioned per-frame calibration

Status: design, revised 2026-11-22 after the P3 evidence run (rig scheme decided by experiment D). No code changed. Classical only (no trained detector, no PnLCalib).
Scope: the workflow around the existing PTZ V3 pipeline (`reports/ptz_v3.md`, `ptz_v3_weak.md`), not new geometry.
Claims marked (V) were checked in code or files this session; (U) are unvalidated or proposed.

KB pages relied on: `prefect-job-system` (pools, queues, merlin move, one-heavy-process etiquette, S3 pattern),
`footy-datasets-and-progress` (three versioned datasets, `eval_result`, nightly flow, DuckDB not LanceDB on the mini),
`footy-data-plane-reconciliation` (one authority per data class). Memory: `gpu-tests-via-prefect`,
`mini-cpu-one-heavy-process`, `versioned-datasets-goal`.

## 0. Concept and decisions

User concept: per-frame detection gives good locks; aggregate across many frames and shots to re-adjust the shared
rig (match-wide bundle adjustment); offline. That is V3. Decisions: classical only; the rig (C, roll, k1) is shared by all
segments of a match, but a segment may come from a different camera and must then be flagged, not forced.

Decisions 2026-11-20 (user):
- **Label-free at inference.** No label-derived prior in a production run (rig JSONs from GT points become eval references only); labels build the eval set (calibration GT + accepted proposals). A run that used any label-derived input is marked `contaminated` in eval.
- **Outlier segments: all three, in order.** Flag first (never forced onto rig*); then try their own camera variant (existing recovery `rig_variants`, e.g. gantry C_x +-32.7 m); a segment that still fails gets H = NaN and is queued for the calibration editor, where it becomes eval GT (not an inference input).
- **Rollout: Man City first**, then the other known matches.
- **Accuracy bar (confirmed):** <= 1 m at ordinary depths, <= 2 m at the far touchline, > 5 m = failure. Gate tuned on accepted/rejected review proposals.
- **Two-level per-segment rig scheme dropped** (P3, section 2.1); rig comes from a multistart `fit_shared` over one frame per segment (section 2.2, decided by experiment D).
- **Review via the existing proposals page** (`calibration_proposals`, accept -> `calibration_gt` with origin, section 3b), not a new verdict mode.

**Decision (user, 2026-11-20): inference is label-free.** No human labels and no GT-derived priors (`leave_clip_out_rig`,
`camera_rigs/*.json` fitted from labels) may enter a production run. Hand labels exist only to build the **evaluation
set** (calibration GT, plus accepted proposals, section 3b). Consequences:
- The rig prior for a new ground must come from generic camera priors (gantry height/distance ranges) plus the match's
  own evidence; `fit_shared`'s GT-derived prior is for research only. `ptz_v3.md` section 6 already shows a label-free
  rig within 0.16 m of the GT rig on Man City, from a good start; a bad start (5 m off) did not converge, so the
  start-diversity of the multistart carries the weight; P3 confirmed this (section 2.1) and experiment D confirmed label-free starts work for pitch accuracy (section 2.2).
- k1 is estimated from the match (free in the refine, or fixed per ground) or left as a diagnostic; it is never
  seeded from labels (P3: k1 0 / 0.0147 / 0.03 changes GT error by less than noise). Where lines and GT points disagree on k1 (0.032 vs 0.0147), report the effect on metre error
  rather than choosing silently.
- Any evaluation of a label-free run on a GT frame is clean only if no label touched the run; runs using a GT-derived
  rig are marked `contaminated` in `eval_result` (section 3).
- Existing per-ground rig JSONs remain usable as *evaluation references*, not as inputs.

## 1. What exists vs what is missing

### 1.1 Exists (V)

| Piece | Where | What it does |
|---|---|---|
| Per-segment solver | `calibration/ptz_temporal.py` (`solve_segment`, `joint_solve`), `scripts/ptz_temporal_segment.py::run` | anchors every ~100 frames (lines-only search), motion pairs, joint solve with the rig **fixed** (`free_shared=()`), recovery pass for weak segments (`ptz_recover.py`, only when standard pass accepts <50%). Writes `calib_full.npz` (marker, shipped last), `line_f1_full.json`, `meta.json`, `solve_full.json`. No GT needed (`clip_gt` stubbed). |
| Match-wide rig fit | `ptz_temporal.fit_shared`, `fit_shared_multistart`; driver `scripts/ptz_temporal.py` `shared` and `sharedms` | Bundle fit of C, roll, k1 over single-frame `Unit`s. Cost knobs: multistart over C seeds, `free=` to fix k1. |
| Fan-out flow | `~/code/jobs/footy_ptz_temporal.py` (deployment `ptz_temporal.prefect.yaml`, `ptz-temporal-match`, pool `cpu`, queue `desktop`), NOT in the footy-track repo | one flow run per match, thread pool of `max_parallel` segments, rsync mp4 to desktop, run `ptz_temporal_segment` niced, ship results back to `scratch/temporal/<match>_segNNN@<rig stem>/`, `calib_full.npz` as done marker, `skip_existing`, one retry, dry-run plan. Mini is refused. Takes `rig_json` as a **parameter**. |
| Combine | `calibration/match_calibration.py`, `scripts/calibration_match_combine.py` | picks main vs recovery result per segment, writes `<calibration root>/ptz_v3/<match>.parquet` + `.manifest.json` (content sha256, `version_id = ptz_v3+<sha8>`, `_history/` on `--update`, refuses silent overwrite). `KNOWN_MATCHES` hard-codes 4 matches to rig stems. |
| Reader / UI | `MatchCalibration`, `labeller/calibration_match.py` + page | projection by global frame. |
| Store ingest of GT/proposals | `feature_store/ingest_calibration.py` | **only** `calibration_gt/`, `calibration_proposals/`, `frame_type/` sidecars. |
| Eval | `~/code/jobs/footy_nightly_eval.py` calibration adapter | scores `calibration_proposals` runs only (n=14, success@5px 0.64 on 2026-11-17). |

### 1.2 Missing (V) for "match video -> versioned per-frame calibration in the store"

1. **No whole-match orchestration of the rig.** The Prefect flow takes a fixed `rig_json`. The shared fit (`shared`, `sharedms`) is a manual
   research subcommand: it uses `MATCH_DIRS` keys `mancity|villa|bmth`, reads from `/mnt/storage/footy_data`, takes its prior from
   `leave_clip_out_rig(clip)` (GT-derived) and its output (`C / roll_deg / k1` in `ms_*.json`) must be hand-converted to a rig JSON
   (`ptz_v3.md` section C says so). Nothing runs prep -> shared fit -> solve -> combine in one go. A new match has no rig JSON.
2. **No per-segment disagreement measure.** `fit_shared` sees ~1 anchored frame per segment (95 frames from 162 Man City
   segments; 42 from 121 Bournemouth), then per-segment solves freeze that rig. (P3: more evidence per segment did not improve the rig;
   search/convergence did, section 2.1.) Nothing scores how well each segment agrees with the shared rig (section 2.2 (d)).
3. **No outlier-segment handling.** Different-camera segments are only handled in the weak-segment recovery (`rig_variants`: C_x = 0, +-32.7 m, chosen from 4 probe frames), only
   when the standard pass accepts <50%, and the variant is stored in `meta.json`/`camera_variant`, not diagnosed per segment from the main run.
4. **Combine is not run by any flow**, and `rerun_weak` handling is by directory convention.
5. **No ingest of the match Parquet into the feature store.** `RunKind.CALIBRATION` does not exist (grep: no `RunKind` in `src`); the store's `calibration`
   table (PK `game_id, source, run_id, frame_index`, `H DOUBLE[9]`, `quality`, `distortion_k1`, `review_status`) takes ~100k rows/match; its `h_convention` string
   (`normalised_image_to_pitch_m`) differs from match_calibration's (`undistorted_normalised_image_to_pitch_m`), and `frame.calibration_model_version` is deprecated. The
   `frame_calibration_best` view picks by source/status; machine run rows must not out-rank hand GT (U: check the view's ordering).
6. **No dataset version / per-version metrics for dense calibration.** The calibration dataset (v1: 99 items, 74/21/4) is GT clips; nothing scores the V3 output against it
   (V3 reports score 20 + 5 + 3 + 1 GT frames by hand, `err5` in the driver).
7. **Quality gate is implicit**: `accepted` = F1 >= 0.5 on both neighbouring evidence frames; never validated against GT outside the scored frames (`ptz_v3.md` section 8).

### 1.3 Other local branches/worktrees (searched)

- Every calibration branch I could check is already merged into main or contains nothing orchestration-related: `exp/calib-temporal-flow`, `exp/calib-temporal-weak-vit`, `feat/calibration-match-files`, `feat/calibration-store` are 0 commits ahead of origin/main.
- The only whole-match orchestration is the **jobs-repo** flow above (`~/code/jobs`, own git repo, default `code_ref` now pinned to a footy-track main sha by P0). It covers fan-out + resume only; nothing for the shared rig, combine, gate, ingest, or versioning. The design below extends that flow; it does not duplicate it.
- `footy_calibration.py` in jobs is the LLM pre-labeller (proposals), `footy_pnlcalib.py` is the (deferred) trained-detector path: both unrelated.

## 2. Rig scheme: multistart shared fit, then fixed-rig solves (V, experiment D)

Evidence: P3 (`reports/ptz_v4_two_level.md` on `exp/ptz-two-level-p3`, appendix A). Experiment D (section 5) decided the rig fit; report `reports/ptz_v5_rig_search.md` (tables `reports/ptz_v5_assets/tables.md`) on branch `exp/ptz-rig-search`.

### 2.1 Rejected: two-level per-segment rig scheme
Per-segment free (C, roll) estimates plus a covariance-weighted consensus, then re-solve, was tested label-free on 162 Man City segments (P3) and dropped:
- Single-start L1: rig* 3.06 m from the GT-points rig (C_y -63.65 vs -66.58); per-segment median error 2.69 m (< 1 m for 2 of 145); GT frame median 0.40 vs 0.26 m for the existing fit (near third 0.58 vs 0.28 m, 32 of 39 frames worse, p < 0.001); 8.5 CPU-h vs ~0.1 CPU-h for the `fit_shared` rig fit. None of the six gantry segments produced an estimate, so d_i flagged nothing.
- Per-segment multistart only reaches parity (rig* 0.17 m, GT 0.26 vs 0.26 m) at ~10 CPU-h, ~100x the cost, and the covariances are still unusable as weights.
- Causes (appendix A): (1) the optimiser stops on a long, curved C_y / C_z / f valley: `lm_solve` in `ptz_temporal.py` stops when a step gains < 1e-4 x total cost (2-7 units on a ridge with 130-300 to give), or stalls on a rough floor; tolerance 1e-6 / 1e-8 moves errors e.g. 2.81 -> 1.65 m, more iterations beyond that do not help. From the GT rig the same code stays within 0.1-0.7 m at a lower cost, so the objective is fine for pan-rich segments and each estimate scales with its start (4 m off moves ~1 m; 7.6 m off does not move). (2) Narrow-pan, zoomed segments (pan 2-5 deg, e.g. 080, 104) are genuinely unidentifiable: the per-segment optimum is 2-4 m off. (3) The per-segment covariance is 30-280x overconfident, and estimates sit 15-40 profile sd from the optimum because the solver stopped. Ruled out: evidence diversity as the reason `fit_shared` works (pooling did not move the estimate; multistart seeds near the truth and free k1 did), and anchors (2-6 per segment, no effect). k1 does not matter here (0 / 0.0147 / 0.03 within noise).

### 2.2 Revised design
(a) **Rig fit (experiment D result): `fit_shared_multistart` over one frame per segment** (the middle frame of each segment the lines-only search accepts; 89 units on Man City), started from the generic label-free prior (gantry height / distance ranges, never GT), **k1 free**, with the **tight LM stopping rule on the finalists** (`rel_tol=1e-8`, `n_iter=100`; `lm_solve` takes `rel_tol`, default `LM_REL_TOL = 1e-4` so other callers are unchanged). The coarse (C_y, C_z) grid search with fixed-C pose solves (E3) was not needed: about 3x the CPU and no better.
(b) **Cost and evidence:** about 3.3 CPU-min per match for the rig fit (vs 2.7 for the old rule, 10 for the grid search). All 18 label-free rigs (3 starts, 4 to 7.6 m off) met the accuracy bar on the 39 answered Man City GT frames (frame median 0.26-0.45 m, far-touchline p95 1.4-1.8 m, nothing above 5 m); E2 with k1 free had the smallest start-to-start spread (0.30-0.31 m). k1 free was not worse than fixed (small sample).
**Caveat: the rig itself is not identified.** Fitted C lies 0.2-2.9 m from the GT-points rig, and the 89-frame cost is flat to about 1% along the C_y / C_z / f ridge, so it cannot rank rigs inside that scatter (the GT-points rig had a higher cost than 13 of 18 fits). The per-frame pose absorbs the error: a rig 2-3 m off costs only 0.05-0.2 m of frame-median pitch error. Consequences: (i) rig-hash versions (section 2.3) will change between runs and starts even when pitch accuracy does not, so a version bump is not evidence of a change in quality; (ii) evaluation must be on pitch metres (`eval_result`, GT frames), never on rig distance or rig cost.
(c) **Fixed-rig per-segment solves** as today (`free_shared=()`, recovery pass for weak segments, variants for second cameras).
(d) **Per-segment diagnostic, no per-segment rig fit:** fixed-C profile cost at the shared rig (1-5 s per segment) plus calibrated / accepted fraction from `meta.json`. P3 shows the fractions already isolate the six gantry segments (0% calibrated) with no new code.
(e) **Deferred:** a bigger joint solve over many frames, and cross-segment rotation-only links (stacking frames of different segments on each other). Test only once (a)/(b) are fixed.

Cost class: the rig fit (~3.3 CPU-min), then one fixed-rig pass (Man City 97.6k frames, desktop 0.16-0.25 CPU-s/frame, 4.4-7 CPU-h); no multi-pass iteration. Heavy CPU on the desktop only, one heavy process (`mini-cpu-one-heavy-process`).

### 2.3 Versioning and partial re-runs
- Work dir stem `<clip>@<rig stem>-<hash8>` (P0, merged) so a changed rig never reuses a stale dir under `skip_existing`.
- Cache key per segment `(segment mp4 sha, code sha, k1, rig hash)`: a changed rig re-runs all segments, unchanged skips. No L1' round, so segment caches are not needed and `keep_cache` stays off.

## 3. Prefect flow design (U)

New flow `calibrate-match` in `~/code/jobs` (own deployment, pool `cpu`, parent on `mini-host`/merlin; children use the `desktop` queue). Parameters: `match`, `ground`, `version` (e.g. `ptz_v3`), `max_cpu_hours`, `dry_run`, `resume`. No rig JSON in production (label-free); an optional `rig_json` is an eval-only, `contaminated` path.

1. **plan**: read `segments.json`, segment hashes, dry-run CPU estimate, refuse if over budget.
2. **rig search** (desktop, one process): section 2.2 (a)+(b); writes `rig_search.json` (per-start screening and final costs, chosen rig, k1). Abort-with-report if the finalists disagree wildly or most segments are rejected (wrong-camera match, e.g. Bournemouth); the cost cannot rank rigs within ~1-3 m, so do not gate on small cost differences.
3. **fixed-rig solves**: fan out the existing `ptz-temporal-match` machinery (reuse `calibrate_segment`, do not fork) with the rig fixed; weak segments via the existing recovery / variants.
4. **diagnostic + gate** (per frame, written as `quality`): accepted (F1 >= 0.5 both sides) / calibrated-unconfirmed / uncalibrated / `other_camera`; per segment the profile cost and accepted / calibrated fractions. Abstain (NaN H) for segments with no confirmed anchor or unresolved outliers; no interpolation across abstained frames; unresolved segments are queued for the calibration editor (eval GT, not an inference input).
5. **Combine**: `combine_match` (existing), manifest extended with rig, `rig_search.json` hash, per-segment diagnostic, code sha; `KNOWN_MATCHES` becomes a parameter.
6. **Ingest** (new `feature_store/ingest_match_calibration.py`): rows in `calibration` with `source='ptz_v3'`, `run_id=<version_id>`, `H`, `quality`, `distortion_k1`, `review_status` mapped (never `hand`), `dataset_tag='calibration_match'`; bulk path (DuckDB on mini pinned 1.3-1.5.3), idempotent by `run_id`; add `RunKind.CALIBRATION`; resolve the `h_convention` mismatch in the ingest.
7. **Proposals export + review job** (section 3b): export the seeded sample to `calibration_proposals/<clip>.json`, submit a labelling job via `POST /api/jobs`.
8. **Report artifact**: Prefect markdown + `reports/calibration/<match>/<version_id>.md` (fractions per segment, rig, profile costs, flagged segments, CPU/wall, failures).

Resume: segments by done marker keyed on the rig hash; rig search by hash of its inputs; combine by content hash; ingest by `run_id`. Failed segments do not abort the match (flow ends `Failed` after combine, listing them).

Versioning and metrics: the calibration dataset is GT; V3 output is a model result scored on it. Extend the nightly eval adapter: per stored `version_id`, score H on GT frames (clip -> match frame via `pts_frame_map`), `err5_median/p90/p95` split near / mid / far, `answered_frac`, `gross_frac` (> 5 m), plus `accepted_frac` / `calibrated_frac`; `eval_result` rows `model_name='ptz_v3'`, `model_version=version_id`, global split authority, `contaminated=true` if any label-derived input. `ptz-rig` GT frames are not independent of a fixed-C solve (`ptz_v3.md` Bournemouth): scored separately.

## 3b. Human review: the existing proposals page (U)

Reuse the existing review path (V): `labeller/calibration_proposals.py` reads `calibration_proposals/<clip>.json`; **accept** copies a proposal into `calibration_gt` with `origin`; **reject** records a reason. No new verdict mode, log, table or job kind on the match page.
- **Exporter** (new, after combine): for each sampled frame, project the **visible pitch keypoints through that frame's stored H** into proposal keypoints, then **distort them with the row's k1** (division model, as `lens.py` / the editor do: stored H takes undistorted coords, proposal points are observed coords), so accepting reproduces H and k1; origin `ptz_v3@<version_id>`; drawn from the stored row, never recomputed.
- **Sampling** (stratified, fixed seed `default_rng(sha256(match, seed))` over global frame ids, so every version is asked about the same frames): ~300 frames per match; status (accepted 50%, calibrated-not-accepted 25%, uncalibrated main-camera 10%, not-broadcast negatives 5%); zoom / pan / far-touchline-visible tertiles; segment boundaries and recovery seams; >= 5 frames per flagged or recovered segment; <= 3 per segment per stratum; >= 1 per segment of >= 500 frames (100 left too little of the budget for the status strata). Inclusion probabilities stored with each proposal.
- **Circularity rule:** proposals from version V are never used to score version V. Accepted proposals feed % good (Horvitz-Thompson over inclusion probabilities, segment-clustered intervals), gate tuning (precision of `accepted`, F1 threshold sweep; the first version gives only an interval), and GT for *other* versions, always flagged by `origin`. Metres error for a version comes from hand-clicked / hand-edited frames or other-version proposals. Rejection reasons give failure-mode counts.
- **Neighbours:** a decision covers one frame of one version; never propagate to unsampled frames. Adjacent frames share anchors (correlated errors) and a good sample does not certify a segment: a segment-level claim needs >= 3 decisions spread across it; a rejected accepted-frame queues extra samples from its segment.

## 4. Open risks, with evidence

- **Bournemouth is a different camera**: Vitality is low (C_z 7.8 m, C_y -50.8) vs Emirates (15.7, -66.6); lines weak (23/121 anchors accepted at F1 >= 0.55); GT rig from 7 frames (C_y bootstrap sd 1.6 m); k1 0.035 (sd 0.018) vs 0.0147 Emirates. Rig does NOT transfer between grounds (`ptz_v2.md`: "transfer weaker"); per-ground rig is required. Within Bournemouth the label-free C landed 0.14 m from the rig (multi-start), so within-match estimation works; the 2nd-half has no segments dir. Only one match per ground has been looked at: *rig transfer between matches of the same ground is assumed, not tested beyond Man City -> Villa (leave-match-out, 19 GT frames, all answered after the rig change, median 0.45 m)*.
- **Second camera positions inside a match**: Man City gantry (C_x +-32.7) on ~8 of 22 weak segments, Bournemouth seg100 at C_x -36.6; others (seg015, 066, 083, 103, 156) are unrecoverable by the three tried positions (seg156 fits C_x -11.5 / -21.5). The flow must carry an explicit `unrecoverable` state; 11 of 22 weak segments recovered (0% -> 53% of the weak segments' frames); 9 of the 22 end with 0% calibrated.
- **Far-touchline metre error (V3), Man City only**: V1 far half slightly worse; V3 numbers are grid medians over the whole visible grid. The C_y/f ridge says depth error will grow with distance from the camera. Experiment D measured it on Man City (39 GT frames): far third median 0.34-0.63 m, p95 1.4-1.8 m across 18 rigs, within the 2 m bar; other grounds untested.
- **k1 disagreement**: lines 0.027-0.034 vs GT points 0.0147 (Emirates), 0.009-0.019 vs 0.035 (Bournemouth). Unresolved; fixing it costs nothing measurable in pitch error on 20 + 5 GT frames but the true lens is unknown. Mitigation: fixed per-ground k1, k1 divergence reported.
- **Acceptance is not validated against GT** (F1 >= 0.5 chosen by the author; the sample is small). Gate thresholds are provisional until section 3 metrics exist.
- **Frame-index correctness**: seek inaccuracy and dropped frames (Bournemouth segments hold ~13% fewer frames than `segments.json`; `CAP_PROP_POS_FRAMES` not exact) matter for the global-frame mapping in combine (`pts_frame_map`); a pts-based map exists, but is verified only on a few segments.
- **Cost surprises**: one fixed-rig pass is 4.4-8.5 CPU-h for Man City on the desktop (P3 measured 8.5 h at a 4 m-off rig); desktop is shared with GPU/other flows; memory fault history (`desktop-ubuntu-hardware-fault`): keep the memory floor (`min_avail_gb`).
- **Unverified here**: whether the multistart rig fit gives the same pitch accuracy from a label-free start on other matches (experiment D, Man City only); Bournemouth / Villa untested.


## 5. Phased plan (small PRs, each with an acceptance test)

Status: P0 merged (jobs #3: rig-hash work-dir stem, `KNOWN_MATCHES` parameter). P1 / P2 (per-segment rig estimate, consensus) closed unmerged (#123, #124). P3 evidence run done: two-level rejected (section 2.1).

| PR | Content | Acceptance |
|---|---|---|
| **D (done, 2026-11-22; `reports/ptz_v5_rig_search.md` on `exp/ptz-rig-search`)** | On the **40 Man City GT frames** (10 of 14 GT segments) using cached data: compare (i) existing `fit_shared` from a generic label-free start, current stopping rule vs tight (1e-6 / 1e-8), (ii) match-level (C_y, C_z) grid search with fixed-C pose solves + converged refine; k1 free vs fixed. Report metres error by depth (near / mid / far median, p90/p95, within 1 / 2 / 5 m) and rig distance to the GT-points rig; also from a start 5-8 m off | Decision memo in `reports/`: go/no-go on (a)+(b); **result: go on multistart + tight rule + k1 free, grid search dropped (section 2.2)**; target <= 1 m ordinary depths, <= 2 m far touchline, no frame > 5 m, label-free start |
| P4 (built; jobs `footy_calibrate_match.py`, footy-track `scripts/ptz_rig_search.py`, `calibration/match_diagnostic.py`) | `calibrate-match` flow (jobs): plan, rig search, fixed-rig solves, diagnostic + gate, combine, report | Man City end to end from empty scratch with no rig JSON: manifest records rig, resume after a kill does not re-solve done segments, identical re-run = `unchanged` |
| P5 | `ingest_match_calibration` + `RunKind.CALIBRATION`, bulk path | row count = calibrated frames; re-run no-op; `frame_calibration_best` still prefers hand GT; 100k rows under ~1 min on the mini |
| P6 | Eval adapter: `ptz_v3@<version_id>` on GT clips, near/mid/far split, nightly wiring, `/progress` series | `eval_result` rows for the version; contamination flag set for GT-rig variants |
| P6b | Proposals exporter (stratified fixed-seed sample, keypoints projected through stored H), job submission, origin flags into ingest | accepting an exported proposal reproduces the stored H; same seed gives the same frame ids for two versions; scoring excludes same-version proposals |
| P7 | Second ground (Bournemouth), gantry / high-camera handling, deferred items of 2.2(e) if justified | Bournemouth rig within ~1.5 m of vitality.json C_y and C_z within 0.5 m; gantry segments flagged, not forced |

D is the only item that spends meaningful CPU before the architecture commits; if it fails, keep `fit_shared` multistart and the rest of the flow.

## 6. Questions for the user

Answered: label-free at inference (section 0); outlier segments: flag, then variant, then NaN + editor (section 0); accuracy bar confirmed (section 0); review budget (moot: review runs through the existing proposals page at the user's pace); rollout Man City first (section 0). Obsolete, removed: cache retention (no L1' round).

Open:
1. **Store trade-off**: ingest only accepted frames (smaller) or all calibrated frames with a quality column? Should machine rows ever appear in `frame_calibration_best`, or only as a separate versioned dataset?
2. **Bournemouth halves**: the second half has no `split_video_broadcast_frames` yet; needed before P7?

(Desktop CPU for experiment D: approved 2026-11-22; done.)
