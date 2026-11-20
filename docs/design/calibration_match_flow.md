# Calibration: production flow, match video to versioned per-frame calibration

Status: design, 2026-11-20. No code changed. Classical only (no trained detector, no PnLCalib).
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
2. **Cross-segment aggregation is one-way and thin.** `fit_shared` sees ~1 anchored frame per segment (95 frames from 162 Man City
   segments; 42 from 121 Bournemouth), then per-segment solves freeze that rig. No feedback: the segments' own (far richer) evidence
   never refines the rig, and no per-segment disagreement is measured.
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
- The only whole-match orchestration is the **jobs-repo** flow above (`~/code/jobs`, own git repo, default `code_ref=exp/calib-temporal-flow`, a branch now merged). It covers fan-out + resume only; nothing for the shared rig, combine, gate, ingest, or versioning. The design below extends that flow; it does not duplicate it.
- `footy_calibration.py` in jobs is the LLM pre-labeller (proposals), `footy_pnlcalib.py` is the (deferred) trained-detector path: both unrelated.

## 2. Primary scheme: two-level, fan-out then consensus (U, proposed)

```
L1  per segment (parallel, cpu/desktop):  solve with loose rig prior, free (C, roll, k1)
        -> per-frame (pan, tilt, logf)   + rig_i = (C_i, roll_i, k1_i) with covariance Sigma_i
L2  match level (one cheap job):          robust, covariance-weighted consensus of {rig_i, Sigma_i}  (+ optional pooled refit)
        -> rig*, per-segment deviation d_i, inlier/outlier flags
L1' per segment (parallel):               re-solve with rig* as a TIGHT prior (or fixed), outliers on their own camera/variant
        -> final calib_full.npz           (optionally one more L2/L1' round, section 2.4)
```

### 2.1 Level 1 detail
- The solver already supports freeing the shared block (`joint_solve(free_shared=..., shared_sigma=...)`; `fit_shared` is exactly this over many units). Level 1 means calling it per
  segment: all of the segment's anchors, line-evidence frames (every 5th) and motion pairs, prior `sigma` loose (the `fit_shared` default: 5/10/3 m, 2 deg, 0.05) centred on a generic
  ground rig (or the previous match version's rig as a *starting value only*).
- Output per segment: rig_i and Sigma_i. Sigma_i from the LM Jacobian at the solution (`J^T J` restricted to the shared block after marginalising the per-frame poses, Schur complement) with a robust-loss correction; or
  a bootstrap over anchors/evidence frames (more honest on flat ridges, ~5-10x cost). **Neither exists today (not exported by `joint_solve`)**: PR needed. The Jacobian version is the default; bootstrap as the check on a few segments.
- Cost: the same ~1 CPU-s/frame as now, plus the shared-block solve (small vs decode 0.3 + motion 0.34; the joint solve is 0.1-0.2). Level 1 *replaces* the current solve; it does not add a run.

### 2.2 Level 2 detail
- Consensus over rig_i: minimise `sum_i rho( (rig_i - rig*)^T Sigma_i^-1 (rig_i - rig*) )` with a robust rho (Huber/Cauchy), initialised at the weighted median, multi-start over C (the basin problem of `ptz_v3.md` section 6 shows up as multi-modal rig_i, so use a mixture
  view: cluster rig_i, the largest weighted cluster is the match rig, the others are candidate second cameras). Pure parameter-space combination is exact only if the per-segment likelihoods are near-Gaussian; the flat y/z ridge (section 2.3) makes them banana-shaped, so (b) is the escape hatch:
- (b) Pooled bundle fit over the **inlier** segments' evidence, i.e. `fit_shared` with many frames per inlier segment (poses initialised from level 1, so no reseeding trouble). It is not parallel but it is one job and its start is already near the optimum, so the multistart cost drops. Choose (a) consensus only or (a)+(b) by the evidence in section 2.6.
- k1: do not estimate per match from lines. Fix k1 per ground (`ptz_v3.md` addendum: fixing improved multi-start C error 0.60 -> 0.33 m, no change in pitch error; lines give 0.027-0.034 vs GT-points 0.0147, a line-mask bias). In level 1 k1 is therefore held at the ground value (new ground: fit once from labelled points) and only (C, roll) are estimated; k1 disagreement becomes a reported diagnostic, not a free parameter.
- Outputs: rig*, Sigma*, per-segment Mahalanobis deviation d_i, flags.

### 2.3 Why this parallelises and handles outliers
- Level 1 is embarrassingly parallel (one job per segment, same fan-out as today). Level 2 reads ~150 small JSONs (a rig and a 5x5 covariance each): seconds. Only level 1' repeats the heavy work, and only for segments whose prior changed meaningfully (section 4).
- An outlier segment (second camera, e.g. gantry C_x = +-32.7 m in Man City segs 46/73/108/137/154; Bournemouth seg100:373 fits only at C_x = -36.6) shows up as a rig_i far from consensus in Mahalanobis terms and is rejected by the robust loss and flagged, **instead of dragging the shared fit** (today's single-frame fit would absorb such frames as noise).
- Risks specific to this scheme:
  - **Weak per-segment estimates.** A short, zoomed or sparse-line segment (few lines, narrow pan range, one zoom) cannot constrain C: Sigma_i must be large and the segment weighs little. A segment whose Sigma_i is huge is *uninformative, not an outlier*: never flag on deviation unless d_i is large *and* Sigma_i is small. Report both numbers.
  - **C_y / C_z vs focal ambiguity.** For a fixed-centre camera at distance ~50-67 m and low height, moving C along the optical axis is nearly compensated by f (and tilt). The evidence: flat y/z ridge, +-2 m in y for <1% cost on Bournemouth, C_y bootstrap sd 1.6 m (Vitality GT rig), multi-start final errors 0.3-0.6 m (addendum A), cost flat along the ridge. Per-segment estimates will scatter along this ridge; Sigma_i must be allowed to be strongly correlated (full covariance, not diagonal) and the consensus must weigh by the full matrix. Zoom diversity within a segment is what breaks the ridge: the segments with the largest focal range carry the information. Honest expectation: rig* is good to ~0.3-1 m on C_y, and pitch error at the touchline is insensitive to it by construction (the ridge is the set of nearly equivalent cameras).
  - L1' with rig* only as a prior can still disagree with L1's own rig_i for good segments: that is the intended mechanism, and the size of the change is the convergence metric (section 2.4).

### 2.4 Iteration, stop rule, cost budget
- Round r: L1 (r=0 loose prior), L2, L1' with prior sigma = Sigma* (tight). Stop when `max change in rig* < 0.2 m (C), 0.05 deg (roll)` AND no segment changes inlier/outlier status AND consensus cost change < 0.5%. Hard cap 2 rounds (1 refit + 1 verification). Expected: usually round 1 only, since L1' with a tight prior is the bulk cost and L2 is free.
- Budget (Man City: 97.6k frames, desktop 0.16-0.25 CPU-s/frame = 4.4-7 CPU-h per full pass): round 0 = 1 pass; round 1 re-solves the whole match (a rig change moves every segment's anchors) = a 2nd pass; worst case 3 passes. Cap `max_cpu_hours` (default 14) and `max_rounds=2` as flow parameters. Cheaper variant: L1' reuses the segment caches (decoded masks, anchors, motion pairs are rig-independent except motion uses k1; with k1 fixed, only the final solve repeats: 0.1-0.2 CPU-s/frame, i.e. ~15% of a pass). **Only valid if the caches are kept** (`keep_cache` is off today, 0.5 MB/frame = ~50 GB for Man City): decide in section 5.
- One heavy process per machine: desktop only (`mini-cpu-one-heavy-process`); mini runs only L2/combine/ingest.

### 2.5 Interaction with versioning and partial re-runs
- Cache key per segment: `(segment mp4 sha, code sha, k1, rig_prior_hash)`. L1' is skipped for a segment if its prior hash is unchanged. A rig* change below the stop thresholds does **not** invalidate segments (record the rig* used in the manifest; the version still changes because rig* changed, but the per-frame results are carried over with `inherited_from` provenance, flagged in the manifest).
- Re-run only affected segments when (a) a segment's rig deviation flag flips, (b) a segment is re-queued (recovery logic changes, new anchors), (c) rig* changes beyond tolerance (then all inliers; outliers on their own camera variant are untouched). New match: all.
- Done marker today = `calib_full.npz` keyed only by `<clip>@<rig stem>`; the stem must become `<rig hash8>` (or `<rig stem>-<hash8>`), else a refit rig re-uses the same dir and `skip_existing` silently skips stale results. This is a **correctness bug waiting to happen** for the iterated scheme (V: `done_clips_on_mini` globs by stem only).

### 2.6 Comparison with the existing `fit_shared` approach, and what would decide
|  | existing (1 frame/segment, then freeze) | two-level (L1 estimates, L2 consensus, L1') |
|---|---|---|
| evidence into the rig | ~1 anchored frame per segment (95/162 Man City) | every anchor, evidence frame and motion pair of every segment |
| outlier segments | absorbed or ignored; no per-segment diagnostic | explicit d_i, flagged, own camera variant |
| parallel | sampling parallel, fit single (6 CPU-min multi-start) | fit parallel, consensus trivial |
| risk | few frames, convergence from 2-5 m starts only by multistart | per-segment weak C; banana-shaped likelihoods; heavier compute (up to 2-3 passes) |
| convergence story | proven at 0.16-0.6 m on Man City, one match | unproven |

The existing approach is a working fallback; the proposal is justified only if it measurably wins. Decision evidence, in order of cost:
1. **Rig agreement across segments (ptz_v3 section 6 style):** on Man City, rig* from two-level vs GT-points rig (C 0.100, -66.576, 15.743) vs the 95-frame fit (error 0.16 m, multi-start 0.31-0.60 m), and the **spread of rig_i** (is it explained by Sigma_i? chi-square of d_i; are the 5-8 known gantry segments the only outliers?). Free, uses existing data.
2. **Metres error by depth on GT frames:** score the final per-frame H on all GT clips (Man City 67 frames, Villa, Bournemouth 13 clips; leave-clip-out rig as in section 2 of v3) as median/p90/p95 **split by pitch depth (near / mid / far touchline)** with `err5`, for both approaches. V1 found the far half a little worse (2-point median 0.31 vs 0.29 m); V3 reports only whole-grid statistics, so the far-touchline error under the final rig is **not characterised** (V). Needs per-depth reporting before either approach is trusted for ball height/offside use.
3. **Stability under start:** rig* from priors 5 m off (the section-6 test), with/without the second round.
Adopt two-level if: C within 0.3 m of the GT-points rig on Man City AND depth-split error no worse than the existing approach AND the gantry segments are flagged without hand input. Otherwise keep `fit_shared`, adding only the diagnostics (d_i by comparing each segment's own solve to rig*), which is cheap.

## 3. Prefect flow design (U)

New flow `calibrate-match` in `~/code/jobs` (own deployment, pool `cpu`, parent on `mini-host`/merlin; children use the existing `desktop` queue). Parameters: `match`, `ground` (rig key), `rig_json` (optional prior), `version` (e.g. `ptz_v3`), `max_rounds`, `max_cpu_hours`, `dry_run`, `resume`.

1. **plan**: read `segments.json`, compute segment hashes, dry-run CPU estimate (existing log line), refuse if over budget.
2. **L1 fan-out**: call the existing `ptz-temporal-match` machinery (reuse `calibrate_segment`, do not fork it) with `rig_json=prior`, new flag `estimate_rig=true`; the driver additionally writes `rig_estimate.json` (rig_i, Sigma_i, #anchors, focal range) into the segment dir **before** the done marker.
3. **L2** (cpu, mini ok, seconds): consensus, write `<match>_consensus_rN.json` (rig*, d_i, flags, per-segment weights). Abort-with-report if fewer than N inlier segments (default 20) or the largest cluster holds <60% of weighted segments (suspected wrong-camera match, e.g. Bournemouth).
4. **L1'**: fan-out again for segments with changed prior hash; `rig_json=rig*`, outliers use their variant.
5. **Gate** (per frame, written into the Parquet as `quality`): accepted (F1 >= 0.5 both sides) / calibrated-unconfirmed / uncalibrated / segment flagged `other_camera` / `rig_outlier`. Abstain (NaN H) rules: segment with no confirmed anchor; segment flagged outlier and unresolved by a variant; frames outside accepted ranges stay stored but `accepted=false`. Do not interpolate H across abstained frames.
6. **Combine**: `combine_match` (existing), with the manifest extended: rig*, Sigma*, consensus round, per-segment `rig_deviation`, code sha, inputs hash. Fix `KNOWN_MATCHES` to be a parameter.
7. **Ingest** (new `feature_store/ingest_match_calibration.py`): per frame row in `calibration` with `source='ptz_v3'`, `run_id=<version_id>`, `H`, `quality` = evidence F1, `distortion_k1`, `review_status` mapped (accepted -> `proposed`/`accepted`, never `hand`), `method`, `dataset_tag='calibration_match'`. Write ~100k rows with a bulk path (not row pydantic; DuckDB on mini pinned 1.3-1.5.3), skip a version already ingested (idempotent by `run_id` = content-hash `version_id`). Add `RunKind.CALIBRATION` and set the deprecated `calibration_model_version` only if a consumer still reads it (U). The `h_convention` mismatch is resolved in the ingest, not by changing either side.
8. **Review job**: submit a `calibration_verdict` labelling job (section 3b) for the seeded sample; its `version_id` is the version being judged.
9. **Report artifact**: Prefect markdown artifact + `reports/calibration/<match>/<version_id>.md`: accepted/calibrated fractions per segment, rig*, rig_i scatter plot, d_i table, flagged segments, CPU/wall, failed segments.

Idempotency and resume: segments by done marker keyed on `<rig hash>`; L2 by hash of its inputs; combine by content hash (existing no-op); ingest by `run_id`. A crashed run re-submitted with the same parameters resumes (already how the jobs flow behaves). Failed segments do not abort the match: the flow completes `Failed` only after combine, listing them, as today.

Placement: heavy CPU on the desktop only, one heavy process per machine, `max_parallel<=4`, niced (as today; 0.16-0.25 CPU-s/frame desktop vs ~1 on the mini; `ptz_v3.md` section 9 recommends desktop). Mini/merlin runs L2/combine/ingest (light). Results go to `/mnt/storage/footy_data` on the mini (read-write only for the `mini-host` queue, per `prefect-job-system`).

Dataset versioning and metrics (versioned-datasets goal): the calibration dataset is GT; the V3 output is a *model result* scored on it. Extend the nightly eval adapter: for each stored match calibration `version_id`, score H at every GT clip's frames (clip -> match/global frame via `pts_frame_map`), metrics `err5_median/p90/p95` **and the same split near/mid/far**, `answered_frac`, `gross_frac` (>2 m), plus `accepted_frac`/`calibrated_frac` per match; rows in `eval_result` with `model_name='ptz_v3'`, `model_version=version_id`, split by the global split authority, `contaminated=true` if the rig prior was fit on that clip's GT (the Emirates rig JSON was: use leave-clip-out or the label-free rig* for the scored version). GT frames from `ptz-rig` source are not independent of a fixed-C solve (`ptz_v3.md` Bournemouth): scored separately, not mixed into the headline.

## 3b. Human review step: verdicts on projected lines (U)

Verified (V): `/calibration/match` (`labeller/calibration_match.py`, `web/calibration_match.html`) draws projected pitch lines on any frame of the per-match
Parquet and already has `sample_rows` (modes even/random/uncertain/accepted, seeded) but is read-only. `calibration_proposals.py` accept copies a per-frame proposal into
`calibration_gt` (with `origin`), reject records a rejection. Ball Check (`ball_check.py`) is the pattern to copy: `POST /ball_check/verdict` appends one JSON line to
`<marks>/ball_checks/<clip>.jsonl`, last line wins, `verdict: null` is an undo tombstone, `hooks.emit` per write, and `feature_store/ingest_all.py` merges the logs into
`detection_review`. Labelling jobs (`labeller/jobs.py`, `POST /api/jobs`) have `KINDS = (ball_place, ball_check, calibration, frame_type, identity, broadcast_review, other)`; progress is computed from
the real label store, and items carry a `sampling` object copied onto result rows (inclusion probabilities). `broadcast_review` (one frame per item, verdict file per match) is the closest existing kind.

**(a) Verdict mode.** On `/calibration/match?match=&version=&review=1` add keys `g` good, `b` bad, `u` unsure, `r` + one-letter reason (line off at far touchline, wrong zoom,
wrong camera, not a main view, flicker/jump), `z` undo, auto-advance through the job's items. Writes `POST /calibration/match/api/verdict` -> append-only
`<marks>/calibration_verdicts/<match>.jsonl`, record `{match, version_id, frame (global), segment, local_frame, verdict, reason, stratum, status_at_review, accepted, evidence_f1, job, ts}`;
last line per `(match, version_id, frame)` wins, tombstone for undo, `hooks.emit("calibration_verdict", ...)`. The verdict is keyed to `version_id` (content hash), so a new version never inherits old verdicts and is
re-sampled with the same seed. The overlay must be drawn from the **stored** row (what the version says), not recomputed. The page stays read-only for everything except this log. Ingest: new
table `calibration_verdict` (match/game_id, version_id, frame, verdict, reason, stratum, ts) in `ingest_all` (new helper next to `import_ball_checks`); never written to `calibration` and never promoted to GT automatically.
Job: new kind `calibration_verdict` (item `{match, version_id, frame, stratum, sampling}`), labelled when the log has a live verdict; the flow submits it as its last step via `POST /api/jobs` (`labeller/prefect_client.py` pattern, `labelling-site-task-board`: one site on :8011, deploy after merge), deep link `/calibration/match?...&job=<id>`.

**(b) Sampling (fixed seed, comparable across versions).** Per match ~300 frames (cap `max_cards`), stratified, drawn by `np.random.default_rng(sha256(match, seed))` over *global frame ids*, so the same frames are
requested again for every version (only strata membership changes) and inclusion probabilities go in `sampling.pi_frame`:
- status: accepted (50%), calibrated-not-accepted (25%), uncalibrated-but-main-camera (10%, expect "bad": checks abstention), plus **not-broadcast negatives** from `frame_type` (5%, for the gate);
- within status: tertiles of focal length (zoom), pan band, and visible-depth (far touchline in view or not), so far-touchline error gets labelled;
- boundary strata: frames within 10 of a segment start/end and within 10 of a motion-chain break / recovery seam (where the solve is weakest);
- rig-outlier strata: all flagged `other_camera`/`rig_outlier` segments get a minimum of 5 frames each; also 5 per recovered (`source=recovered`) segment;
- at most 3 frames per segment per stratum, minimum 1 frame from every segment >= 100 frames, so coverage is across the match, not clustered.

**(c) Uses.** (i) Per-version % good with Wilson interval per stratum and headline weighted by `pi_frame` (a Horvitz-Thompson estimate of the share of the match that is good, which is the
number to track in `eval_result`, `model_name=ptz_v3`, metric `human_good_frac`). (ii) Gate quality: precision of `accepted` = good/(good+bad) among accepted frames, recall of the gate = share of
good frames that were accepted, swept over the F1 threshold (0.5 now, never validated): pick the threshold with a stated precision floor (e.g. >= 0.95 good among accepted); needs several hundred verdicts to
pin a threshold, so the first version can only give an interval. (iii) Compare methods/versions without full GT: same seeded frames, paired verdicts (A good / B bad); report disagreement counts and a sign test. (iv) Routing: a `bad`
frame offers a button "fix" to the existing calibration editor (`/calibration`, seeded with the stored H as the proposal, `calibration_proposals` accept path -> `calibration_gt` with `origin=ptz_v3@version_id`); frames fixed this way become GT and
grow the calibration dataset (a new dataset version, tracked by the nightly flow), which also gives the near/mid/far metres-error numbers the plan needs. Verdicts themselves are labels about a *version*, not GT.

**(d) Neighbours.** A verdict is about that frame and one version only. Within a motion-connected run adjacent frames share the same anchors, so errors are strongly correlated (one bad anchor spoils up to ~100 frames) but a
good frame does not certify the segment: drift, zoom changes and chain breaks (e.g. Bournemouth seg060 frame 563/564) appear between samples. Rules: never propagate a verdict to unsampled frames in stored data;
report estimates per stratum and per match with segment-clustered intervals (treat segments as clusters, not frames as independent); a `bad` on an accepted frame triggers a flag on its segment (extra samples queued from the same
segment next round) rather than a rejection of the segment. A segment-level claim needs >= 3 verdicts spread across it.

## 4. Open risks, with evidence

- **Bournemouth is a different camera**: Vitality is low (C_z 7.8 m, C_y -50.8) vs Emirates (15.7, -66.6); lines weak (23/121 anchors accepted at F1 >= 0.55); GT rig from 7 frames (C_y bootstrap sd 1.6 m); k1 0.035 (sd 0.018) vs 0.0147 Emirates. Rig does NOT transfer between grounds (`ptz_v2.md`: "transfer weaker"); per-ground rig is required. Within Bournemouth the label-free C landed 0.14 m from the rig (multi-start), so within-match estimation works; the 2nd-half has no segments dir. Only one match per ground has been looked at: *rig transfer between matches of the same ground is assumed, not tested beyond Man City -> Villa (leave-match-out, 19 GT frames, all answered after the rig change, median 0.45 m)*.
- **Second camera positions inside a match**: Man City gantry (C_x +-32.7) on ~8 of 22 weak segments, Bournemouth seg100 at C_x -36.6; others (seg015, 066, 083, 103, 156) are unrecoverable by the three tried positions (seg156 fits C_x -11.5 / -21.5). The flow must carry an explicit `unrecoverable` state; 11 of 22 weak segments recovered (0% -> 53% of the weak segments' frames); 9 of the 22 end with 0% calibrated.
- **Far-touchline metre error is uncharacterised under V3**: V1 far half slightly worse; V3 numbers are grid medians over the whole visible grid. The C_y/f ridge says depth error will grow with distance from the camera. No measurement by depth exists (section 2.6).
- **k1 disagreement**: lines 0.027-0.034 vs GT points 0.0147 (Emirates), 0.009-0.019 vs 0.035 (Bournemouth). Unresolved; fixing it costs nothing measurable in pitch error on 20 + 5 GT frames but the true lens is unknown. Mitigation: fixed per-ground k1, k1 divergence reported.
- **Acceptance is not validated against GT** (F1 >= 0.5 chosen by the author; the sample is small). Gate thresholds are provisional until section 3 metrics exist.
- **Frame-index correctness**: seek inaccuracy and dropped frames (Bournemouth segments hold ~13% fewer frames than `segments.json`; `CAP_PROP_POS_FRAMES` not exact) matter for the global-frame mapping in combine (`pts_frame_map`); a pts-based map exists, but is verified only on a few segments.
- **Cost surprises**: 2-3 passes (7-21 CPU-h Man City on the desktop); desktop is shared with GPU/other flows; memory fault history (`desktop-ubuntu-hardware-fault`): keep the memory floor (`min_avail_gb`).
- **Unverified here**: Sigma_i quality (flat ridges), whether the Schur-complement covariance is calibrated, whether the second round ever changes anything.

## 5. Phased plan (small PRs, each with an acceptance test)

| PR | Content | Acceptance |
|---|---|---|
| P0 (jobs repo) | Rig-hash in work-dir stem and done-marker; `KNOWN_MATCHES` becomes a parameter; commit `footy_ptz_temporal.py` flow default `code_ref` to a main sha | dry-run on Man City lists the same 162 segments; a re-run with a changed rig JSON re-runs, an unchanged one skips |
| P1 | `joint_solve`/driver: optional free (C, roll) per segment returning Sigma (Schur complement), `rig_estimate.json`; unit test on synthetic segments with a known rig | recovers a synthetic C within 3 sigma; sigma shrinks with focal range and pan range; k1 fixed path unchanged: **segments solved with `estimate_rig=false` are bit-identical to now** (regression on the 11 segments in `ptz_v3_weak.md`) |
| P2 | `calibration/rig_consensus.py` (pure numpy): robust covariance-weighted consensus + mixture/cluster, d_i, flags; CLI that reads a match's `rig_estimate.json` | on stored Man City level-1 outputs: consensus within 0.3 m of the GT-points rig; known gantry segments (46, 71, 73, 108, 137, 154) flagged by d_i, no more than 3 false flags among the 140 accepted |
| P3 | Evidence run, **no new infra**: level-1 on Man City (one desktop pass, ~5 CPU-h) and Bournemouth; compare against `fit_shared` on rig agreement and depth-split GT error (section 2.6) | decision memo in `reports/`; go/no-go for two-level |
| P4 | `calibrate-match` flow (jobs): plan, L1, L2, L1', gate, combine, report artifact; `max_rounds`, budget | Man City end to end from an empty scratch with no rig JSON: manifest records rig*, resume after a killed run finishes without re-solving done segments, second identical run = `unchanged` |
| P5 | `ingest_match_calibration` + `RunKind.CALIBRATION`; bulk path | row count = calibrated frames; re-run is a no-op; `frame_calibration_best` still prefers hand GT on GT frames; DuckDB ingest of 100k rows under ~1 min on the mini |
| P6 | Eval adapter: `ptz_v3@<version_id>` on the GT clips, near/mid/far split, nightly flow wiring, `/progress` series | a Man City `eval_result` row set appears for the version; contamination flag set for the GT-rig variant |
| P6b | Verdict mode on the match page, `calibration_verdicts` log, `calibration_verdict` job kind, seeded stratified sampler, ingest table | verdict round-trips (write, undo, last-wins), job progress is computed from the log, the same seed gives the same frame ids for two versions, overlay shown equals the stored H, ingest is idempotent |
| P7 | Second ground end to end (Bournemouth), outlier handling for gantry/high cameras, optional round 2 | Bournemouth rig* within ~1.5 m of vitality.json C_y and C_z within 0.5 m; gantry segments flagged, not forced |

P3 is the only PR that spends significant CPU before the architecture commits.

## 6. Questions for the user

1. **Rig priors for a new ground**: accept a one-off hand labelling (~7-10 GT frames) to fix k1 and seed the rig per ground, or must new grounds be fully label-free (then k1 stays a diagnostic, not a parameter)?
2. **Cache retention**: keep per-segment caches (~0.5 MB/frame, ~50 GB per Man City) on the desktop so L1' costs ~15% of a pass, or discard and pay full passes? Which disk is allowed to hold them?
3. **What do outlier segments become?** Flag and abstain (H = NaN), fit their own camera variant when recoverable (extends `rig_variants`), or send to hand labelling on the labeller (the calibration editor already exists)?
4. **Accuracy bar**: what metres error, at what depth, makes a frame usable downstream (tracking positions, ball height)? This sets the gate and decides whether far-touchline error needs fixing before ingest.
5. **Store trade-off**: ingest only accepted frames (smaller) or all calibrated frames with a quality column (a consumer filters)? And should machine rows ever appear in `frame_calibration_best`, or only as a separate versioned dataset?
6. **Review budget**: how many verdicts per match are you willing to give (300 is about 10-15 minutes at keyboard speed), and is a first threshold from ~300 verdicts good enough to tune the gate?
7. **Rollout**: Man City first (GT-rich, validated), then Bournemouth/Villa/Forest, or all four known matches in one go? Bournemouth halves: second half has no `split_video_broadcast_frames` yet.
