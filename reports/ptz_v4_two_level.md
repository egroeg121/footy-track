# PTZ V4 evidence run: does the two-level rig scheme beat the existing V3 approach? (P3)

Date: 2026-11-20/21. Branch `exp/ptz-two-level-p3` (= `feat/ptz-rig-consensus` P2 + `feat/ptz-rig-estimate` P1 + P3 additions).
Design: `docs/design/calibration_match_flow.md` sections 0, 2 (2.6 decision criteria), 5. Baseline: `reports/ptz_v3.md`.
KB pages relied on: `prefect-job-system` (pools/queues, desktop worker, merlin move), `footy-datasets-and-progress`, `footy-data-plane-reconciliation`;
memory notes `mini-cpu-one-heavy-process`, `gpu-tests-via-prefect`. Everything heavy ran on the desktop through Prefect.

## 0. Decision

**Keep the existing approach (`fit_shared` multi-start, then fixed-rig per-segment solves). Add only the cheap diagnostics. Do not adopt two-level as specified.**

Section 2.6 criteria, as measured (Man City, label-free generic start, small n, see caveats):

| criterion | single-start L1 (the design as written, pre-registered run) | multistart L1 (post-hoc ablation, 28 of 162 segments) |
|---|---|---|
| rig* within 0.3 m of the GT-points rig | **FAIL: 3.06 m** (C_y -63.65 vs -66.58) | 0.17 m (consensus), 0.34 m (plain median); bootstrap-over-segments sd of C_y is 0.3 m, so borderline, not proven |
| depth-split GT error no worse than existing | **FAIL**: frame median 0.40 vs 0.26 m, near third 0.58 vs 0.28 m (32 of 39 frames worse, Wilcoxon p < 0.001) | tie: 0.26 vs 0.26 m, p = 0.11 (20 better / 9 worse) |
| gantry segments (46, 71, 73, 108, 137, 154) flagged without hand input | **not by d_i**: none of the six produces a rig estimate at all (standard pass calibrates 0 frames); they appear only as "no estimate", which the existing V3 run already shows (0% calibrated) | same |
| cost | 8.5 CPU-h for the single-start pass (0.31 CPU-s/frame) | 0.37 CPU-s/frame, i.e. ~10 CPU-h for all 162 segments, vs ~0.1 CPU-h for the existing 95-frame multi-start fit |

So the scheme only reaches the existing approach's accuracy once every segment runs its own multi-start, and then it costs ~100x the rig fit for the same answer. The measured per-segment covariances are not usable as weights (section 4). The two-level rig fit is not justified; the arguments for it (outlier diagnostics, richer evidence) did not materialise.
What does survive: (a) rig-hash dir naming (P0, already needed), (b) `rig_estimate.json` with multistart as an optional diagnostic on a handful of segments, (c) rig* from a *cheap* multi-start fit stays the rig source. Recommended diagnostic: per segment, report the standard pass's accepted fraction and calibrated fraction (already in `meta.json`); this already separates the six gantry segments and 11 other no-calibration segments with no new code.

## 1. What was run (all label-free; GT only as eval reference)

- Prior ("generic gantry", `reports/ptz_v4_assets/generic_gantry_prior_v2.json`): C = (0, -63, 14), roll 0, **k1 = 0.03**, sigma_C (5, 10, 4). Round numbers, centre line, a gantry guess; 4.0 m from the GT-points rig (dy 3.6, dz 1.7). Not derived from any label. **k1 = 0.03** is the line-derived value of `ptz_v3.md` addendum (0.027-0.034, from the match's own line evidence; the poses used to initialise that fit came from the GT rig, so it is match-derived and label-free in intent, not hermetically clean). First a 7.6 m-off prior (C = (0, -60, 12), sigma_C (1, 3, 1)) was tried on 5 segments: the estimates stayed at the prior (seg010: -60.26, 12.20, error 7.2 m; seg104 landed 16 m off, at C_y -51.5), and 2 of 5 segments gave no estimate at all (046, 080); abandoned after 0.5 CPU-h (counted below). The prior was then made 3.3 m closer with a wide search sigma; that is the run reported.
- **k1 does not matter here**: rig* with k1 = 0 / 0.0147 (GT-points value, contaminated) / 0.03 on 4 segments, 11 GT frames: frame median 0.26 / 0.30 / 0.34 m, near/mid/far medians equal within 0.05 m except far (0.51 / 0.65 / 0.64 m, n too small to rank). No effect larger than the noise.
- Contaminated items (marked): the `@emirates` solves (arm A) and `k1 = 0.0147` (sensitivity only) use the GT-points rig; `emirates.json` / `(0.100, -66.576, 15.743)` appears as the *reference* in every distance. The "existing 95-frame fit" (`ms_mancity_ms_p1.json`, k1 free) was started 5.8 m from the GT rig: its result is not GT-fitted but its start was GT-relative (start-contaminated). No label entered L1, L2 or rig* (arms C, C2).
- Infra: `~/code/jobs` branch `feat/ptz-rig-hash` (P0) + one new commit: `extra_args` pass-through and `rig_estimate.json` / `rig_bootstrap.json` shipped before the done marker; **new** deployment `ptz-temporal-match/ptz-temporal-match-exp` (`ptz_temporal_exp.prefect.yaml`) pulling from `/home/george/code/jobs-ptz-rig-hash` (rsynced to the desktop). The production `ptz-temporal-match` and `~/code/jobs` master were not touched. Output dirs are `<clip>@<rig stem>-<hash8>` under `scratch/temporal/` (checked: 0 collisions; the generic prior dirs `@generic_gantry_prior_v2-1be31b96`) and `scratch/temporal/p3/{boot,ms,msL1,armB,armC,armC2,arm_k0,arm_k0147}/`.
- Code changes in this branch (on top of P1/P2): the rig estimate is taken from the **standard pass** before any recovery variant replaces rig/states (so the hard-coded C_x variants cannot leak into rig_i); `--bootstrap-rig N` (half-sample resampling), `--rig-multistart` (9 (C_y, C_z) seeds, 2 finalists), `ptz_rig_consensus --cov-inflation`. Tests added (3, pass). Scripts in `scripts/ptz_p3/` (`score_gt.py`, `l2.py`, `show_estimates.py`, `plots.py`).

## 2. L1 and L2 on all 162 Man City segments (single start)

L1: `--estimate-rig --no-recover`, anchor step 100, 4 parallel niced processes on the desktop, 97,625 frames, 8.52 CPU-h (0.31 CPU-s/frame; the plan said 4.3-7: segments solved at a 4 m-off rig are slower and need more LM iterations). 162 of 162 segments finished, 0 failed. 17 produced no estimate (no calibrated frame with line evidence): 5, 15, **46, 71, 73**, 66, 68, 83, 85, 92, 93, 103, **108**, 109, 119, **137, 154**. Of the 145 estimates, 65 hit the iteration cap (`converged=false`) and are excluded from the fit by P2; 80 enter.

| quantity | value |
|---|---|
| per-segment error vs GT-points rig (145 estimates) | median 2.69 m; < 1 m: 2; < 2 m: 20; > 5 m: 2 |
| quoted sd of C_y (x36) | median 0.06 m (raw model 0.01-0.02 m) |
| rig* (inflation 30, calibrated, section 3) | C = (0.219, -63.641, 14.856), roll -0.212 deg; sd (0.02, 0.03, 0.02) m |
| rig* (inflation 36, provisional) | C = (0.220, -63.646, 14.859); the same to 0.01 m |
| rig* (inflation 1 / 1000) | (0.213, -63.603, 14.851) / (0.244, -63.701, 14.905) |
| vs GT-points rig (0.100, -66.576, 15.743) | **3.07 m** (dC = +0.12, +2.94, -0.89) |
| vs existing 95-frame fit (-0.207, -67.033, 15.977) | 3.60 m |
| start (prior) vs GT rig | 4.0 m: rig* moved 0.9 m of 4.0 toward the truth |
| chi2 of d_i vs df = 4 (80 fitted + others, inflation 30) | mean d^2 = 340 (expected 4), median 147 (expected 3.36); 11% of segments below the 0.999 quantile. Inflation 36: mean 304, 12% (the provisional 36 and the calibrated 30 are the same story). Even inflation 1000: mean 24, 60% inlier |
| flags (inflation 30) | 65 unconverged, 13 inlier, 35 outlier, 32 variant clusters (the "variants" are the scatter of a biased estimator, not second cameras) |

Every segment's estimate sits near the start: C_y spans -63.0 to -64.8 for most, z ~ 14.9-15.2, as the scatter plot (`reports/ptz_v4_assets/p3_overview.png`, left) shows. A 4 m-off start, single LM, per segment, collapses onto the flat C_y / C_z / f ridge about 1 m from where it started, with a quoted sd of centimetres. This is the failure `ptz_v3.md` section 6 / addendum A documented for `fit_shared` from a bad start, now seen on every segment. The bias is shared by all segments, so the consensus cannot average it out.

**Gantry segments.** None of 46 / 71 / 73 / 108 / 137 / 154 reaches L2: the standard pass leaves no calibrated frame (the existing `@emirates` run also shows 0% calibrated for the same six). They are "flagged" only in the sense of "uncalibrated", with no deviation score and no variant position. The 11 other no-estimate segments (5, 15, 66, 68, 83, 85, 92, 93, 103, 109, 119) are the false flags relative to the gantry list (4 of them, 15, 66, 83, 103, are already named unrecoverable in the design doc). 

## 3. Covariance calibration (bootstrap) and the inflation used

Half-sample bootstrap (50% of evidence blocks of 4 consecutive evidence frames and 50% of anchors, all motion pairs, 20 resamples, warm start; scaled by sqrt(0.5) to full-sample size), 5 segments: 10 (typical, 595 frames), 20 (short, 150), 80 (shortest, 105), 104 (zoomed, 295), 150 (180, pan 30 deg). Wide-pan seg128 was started but cancelled at the budget guard; the gantry segment 46 has no estimate to resample. Costs 0.13 CPU-h in total.

| seg | rig_i C_y | sd_y raw model | sd_y x36 | sd_y bootstrap | z^2 raw (E = 4) | z^2 x36 | variance factor |
|---|---|---|---|---|---|---|---|
| 010 | -64.44 | 0.010 | 0.06 | 0.09 | 134 | 3.7 | 34 |
| 020 | -64.04 | 0.024 | 0.14 | 0.15 | 119 | 3.3 | 30 |
| 080 | -63.72 | 0.013 | 0.08 | 0.04 | 21 | 0.6 | 5 |
| 104 | -62.84 | 0.012 | 0.07 | 0.08 | 1122 | 31 | 280 |
| 150 | -64.14 | 0.019 | 0.12 | 0.06 | 63 | 1.8 | 16 |

Raw model covariance underestimates the sampling spread by a factor 5-280 (median 30, geometric mean 30); the provisional x36 is right for 4 of 5 segments (z^2 0.6-3.7 against 4) and 8x too small for the short zoomed seg104 (3 anchors). **Inflation used for the "calibrated" L2: 30** (also reported at 36: identical). Caveat: this is sampling variability at a stalled optimum. It does not see the 2-4 m bias, and the across-segment chi2 above says the bias, not the sampling noise, dominates: reproducing chi2 ~ 4 across segments needs an inflation far above 1000. n = 5, one start.

## 4. Why multistart changes the answer (ablation, not the specified scheme)

`--rig-multistart` per segment (9 seeds on a (C_y, C_z) grid, C_x fixed, 6-iteration screening, 2 finalists) on 4 short segments: 020 -> 1.30 m error (was 2.81), 150 -> 0.88 (was 3.10), 080 stays 3.07, 104 gets worse (5.35). Run on 24 more mid-length segments (200-700 frames, every third, the 17 no-estimate ones excluded; 9.5k frames, 1.19 CPU-h): 28 estimates, 22 converged, per-segment median error 1.83 m, < 1 m for 10, > 5 m for 4. Errors now scatter around the truth instead of around the start. P2 consensus (inflation 30): **rig* = (0.169, -66.725, 15.715), roll -0.193 deg, 0.17 m from the GT-points rig** (0.36 m at inflation 1, 0.17 at 36); plain per-axis median 0.34 m; bootstrap over the 22 segments: C_y 5-95% [-67.18, -66.32]. The covariances are still unusable as weights: chi2 mean d^2 = 1986 (df 4), 15 of 22 flagged outliers by a deviation threshold that is meaningless when the quoted sd is 0.02-0.2 m and the real scatter is ~2 m. The consensus works here because it is a robust location estimate over ~22 noisy rigs, not because of the covariance weighting. n = 22 segments from one match, the same match everything was tuned on.

## 5. L1' on the GT segments: metres error by pitch depth

L1' = a normal V3 solve (recovery on, as production) with the rig fixed, on the Man City segments holding calibration GT. Budget forced a subset: **10 of 14 GT segments** (010, 020, 031, 050, 060, 080, 090, 104, 140, 150): **40 GT frames**, one of which (031:328) is unanswered in every arm (excluded from the GT rig fit too; "too few consistent points"). Not run: 000, 100, 110, 120 (8 GT frames, 6,490 frames of video). Error as `err5` (5 m grid of GT-visible pitch points mapped by each arm's per-frame H, metres), thirds by GT pitch y (camera on the y < 0 side): near y < -11.3, mid, far y > 11.3. Frame-level = median over the grid per frame, then median / p90 / p95 over frames; thirds are pooled grid points (n = 824 / 1624 / 1449 near / mid / far for arm A).

| arm | rig used | label-free? | answered | frame median / p90 / p95 (m) | pooled within 1 / 2 / 5 m (%) | near med / p95 | mid med / p95 | far med / p95 |
|---|---|---|---|---|---|---|---|---|
| A | GT-points rig `emirates.json` (existing `@emirates` run) | **no, contaminated** | 39 / 40 | 0.24 / 0.43 / 0.53 | 94 / 99 / 100 | 0.25 / 0.98 | 0.22 / 0.73 | 0.33 / 1.42 |
| B | existing 95-frame `fit_shared` multi-start (k1 free 0.0315) | start GT-relative | 39 / 40 | 0.26 / 0.43 / 0.46 | 92 / 98 / 100 | 0.28 / 1.04 | 0.22 / 0.75 | 0.37 / 1.74 |
| C | two-level rig*, single-start L1 (3.07 m off) | yes | 39 / 40 | **0.40 / 0.66 / 0.75** | 89 / 99 / 100 | **0.58 / 1.39** | 0.30 / 0.92 | 0.51 / 1.40 |
| C2 | two-level rig*, multistart L1 (0.17 m off) | yes | 39 / 40 | 0.26 / 0.43 / 0.49 | 93 / 98 / 100 | 0.27 / 1.02 | 0.23 / 0.76 | 0.33 / 1.63 |
| ref | L1 solve at the generic prior (no recovery), for scale | yes | 39 / 40 | 0.91 / 3.83 / 5.72 | 50 / 73 / 94 | 0.86 / 5.70 | 0.52 / 6.04 | 1.85 / 4.51 |

(Existing approach on all 14 GT segments, arm A: 47 / 48 answered, frame median / p90 / p95 0.23 / 0.43 / 0.51; near / mid / far medians 0.24 / 0.21 / 0.32 m, p95 0.98 / 0.66 / 1.32; within 1 / 2 / 5 m: 95 / 99 / 99.9%.)

Paired on the 39 common frames (per-frame median): C vs B +0.12 m (p < 0.001, 32 worse, 3 better); C2 vs B -0.02 m (p = 0.11, 9 worse, 20 better); B vs A +0.01 m (p = 0.18). A 3 m rig error costs ~0.15 m median and 0.3 m at the near third even though the camera pose absorbs most of it (the ridge: nearly equivalent cameras), and the far third is the worst depth in every arm (0.33-0.51 m median, p95 1.4-1.7 m; this is the first depth-split characterisation for V3 and it supports the "far touchline is harder" expectation, though 1,449 points from 39 frames is small and the thirds are heavily clustered within a few segments).
There are no gross errors (> 5 m) in A, B, C or C2 on these frames (pooled 100% within 5 m).

## 6. Cost

| item | CPU-h (desktop, summed from `meta.json` cpu_s) |
|---|---|
| L1, 162 segments, single start (generic prior v2) | 8.52 |
| first 7.6 m-off prior, 5 segments (abandoned, + a cancelled 6th, ~0.2 not counted) | 0.49 |
| bootstrap, 5 segments x 20 | 0.13 |
| multistart ablation (4 + 24 segments) | 1.26 |
| L1' arms C, B, C2 (10 segments each) | 0.27 + 0.17 + 0.27 |
| k1 sensitivity (2 x 4 segments) | ~0.1 |
| **total** | **~11.4 (budget 12)** |

Wall: L1 took 2 h 5 min (22:07 -> 00:13). Man City L1 alone was 8.5 CPU-h, over the 5-7 h forecast. The existing approach spends ~0.1 CPU-h on the rig fit (95 frames, 17-20 wall minutes) before the one fixed-rig pass.

## 7. What broke / caveats

- The generic prior matters more than anything else: at 7.6 m off, per-segment L1 does not leave the start; at 4.0 m it moves 1 m. The design doc's "rig* from a label-free start" is the weak point, as its section 4 anticipated.
- The desktop lost power at 00:22 on 21 Nov during arm C (HA shows `switch.kompootar_socket_1` turning off at 00:22:24; I did not do this); the run was marked Crashed. I turned the plug back on via Home Assistant (the helper needs `HA_URL=http://192.168.3.227:8123` on merlin; its default `localhost` fails silently), the worker came back in ~1 min, and `skip_existing` resumed arm C (8 of 10 segments were already shipped). Nothing else was affected; cause unknown (hardware fault history in the KB, or someone cut the plug).
- Small n throughout: 40 GT frames in 10 segments (one match, one ground), 5-segment bootstrap, 28-segment multistart. Man City is also the match every V3 setting was tuned on. Arm B's rig fit started GT-relative, so the existing approach's label-free robustness from a generic start is untested (it was not run from the generic prior; at most one could say it reached 0.16-0.6 m from starts 5 m off in `ptz_v3.md`).
- The single-start L1 hit the iteration cap in 45% of segments; a longer LM was not tried (the multistart result suggests the ridge/basin, not the iteration count, is the problem: converged segments sit at the same bias).
- Not tested: Bournemouth (P3 mentioned it; budget), the optional pooled refit (b), the second round / L1' with a tight prior (L1' here solved with rig* fixed), weak-segment recovery effects on rig_i, per-segment multistart on all 162.
- Gantry detection by deviation is untestable with these data (no estimate), so "flag outliers by d_i" is unproven rather than disproven; the recovery path (`rig_variants`) is the part that actually positions a second camera.

## 8. Artifacts

- Plots: `/mnt/storage/footy_data/scratch/temporal/p3/plots/p3_overview.png` (also `reports/ptz_v4_assets/`): left, every rig_i (C_y, C_z) vs GT, prior, existing fit, rig*; right, depth-split GT error by arm.
- Raw: `scratch/temporal/p3/l2/` (single-start consensus at inflations 1/30/36/1000 + `l2_summary.json`), `l2_ms/`, `score/*.json` (per-frame, per-point errors), `boot/*/rig_bootstrap.json`, `ms*/`, `arm*/` result dirs; L1: `scratch/temporal/arsenal_mancity_seg*@generic_gantry_prior_v2-1be31b96`.
- Small copies in `reports/ptz_v4_assets/`: prior and rig* JSONs, consensus JSONs, score JSONs, l2 summaries.
- Jobs repo: `~/code/jobs-ptz-rig-hash`, branch `feat/ptz-rig-hash` (local, not pushed), new commit "ptz-temporal: extra_args pass-through...", deployment `ptz-temporal-match/ptz-temporal-match-exp` (can be deleted; production deployment untouched).
