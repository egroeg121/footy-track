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

## Appendix A. Why two-level failed (root-cause experiments, 2026-11-21)

KB pages relied on: `prefect-job-system` (not needed: everything ran on merlin, niced, single thread, ephemeral `uv run --no-project` env; no desktop/HA contact), `footy-datasets-and-progress`; memory `mini-cpu-one-heavy-process`. Scripts: `scripts/ptz_p3/why_{common,profile,optimiser,inputs,pool,long}.py`; raw JSON: `reports/ptz_v4_assets/why/`. Cost: ~1.05 CPU-h (3,800 CPU-s, merlin, one process at a time). Inputs rebuilt from the kept L1 caches for the 5 bootstrap segments 010 / 020 / 080 / 104 / 150 (pan range 38 / 4 / 2 / 5 / 31 deg; f ratio 1.62 / 1.03 / 1.01 / 1.06 / 1.29). All fits use the L1 set-up (C_x, C_y, C_z, roll free, loose prior at the generic prior, k1 = 0.03 fixed, same units as `solve_rig_estimate`); "cost" is the robust IRLS cost (inlier part = chi2/2). G = the GT-points rig (reference only). The production solve reproduces the stored `rig_estimate.json` exactly (seg020 cost 18866.2 and the same C).

| # | hypothesis | test and result | verdict |
|---|---|---|---|
| 1 | weak identifiability inside one segment (C_y/C_z/f ridge) | Fixed-C cost, poses re-optimised, along the line E (L1 estimate) -> G. Cost at G minus cost at E: seg010 -290, 020 -297, 150 -125 (chi2 -250 to -600): the objective prefers the GT end and its minimum is within 0.2-0.7 m of G; profile-likelihood sd along the ridge ~0.07-0.1 m. But seg080 (pan 2 deg, f flat) +44 and seg104 (pan 5 deg, f ~ 4750-5000) +1145: there the per-segment optimum is 1.8-4 m from G, a real identifiability / bias problem for narrow-pan, zoomed segments. Across all 145 L1 estimates, error vs pan range: Spearman -0.34, vs f ratio -0.19 (weak); median error is still 2.5 m for pan > 30 deg (76 segments) vs 3.1-3.5 m for pan < 15 deg. | partial (true only for narrow-pan / zoomed segments, 2 of 5; not the main cause) |
| 2 | optimiser, not objective | (a) Production stopping rule (`lm_solve`: stop when a step gains < 1e-4 x total cost, i.e. 2-7 cost units on a ridge that has 130-300 units to give) stops after 16-28 iterations. Tolerance 1e-6 / 1e-8 from the same start: error 2.81 -> 1.65 m (020), 3.07 -> 2.75 (080), 2.53 -> 1.86 (150), 2.24 -> 1.46 (010), cost still falling (020: 18866 -> 18667). (b) From G with the production rule, C stays at 0.1-0.7 m from G and ends at a LOWER cost than the prior-start fit: 020 18539 vs 18667, 010 72682 vs 72749, 150 21053 vs 21070 (seg080 14117 vs 14068 and seg104 39810 vs 38812 are the two segments where the objective prefers the wrong place). (c) A 1500-iteration budget with tol 1e-11 ends where tol 1e-8 did (020 1.65 m, 080 2.75, 150 1.86), by "no improving step" not by the cap: the LM stalls in a curved narrow valley (truncated distance-transform cost, finite-difference Jacobian), more iterations do not help. (d) Over the 145 L1 estimates the error correlates with the iteration count (Spearman -0.55): the further the LM got, the closer to GT; the 65 capped ones are no worse (median 2.4 vs 3.2 m), i.e. the cap is a symptom, the early stop and stall the cause. (e) Fixed-C pose solves are converged (tight vs production tolerance differ by < 7 cost units), so the profile in row 1 is not optimiser noise. | confirmed (main cause: ridge creep + premature stop + stall, start-dependent) |
| 3 | bias from inputs computed at the prior (anchors, initial poses, motion nominal f) | Four variants x 5 segments, tight LM, same prior start: prior inputs / anchors, prior inputs / no anchors, GT-rig inputs / anchors, GT-rig inputs / no anchors (full pipeline re-run at G). Error (m): 010 1.46 / 1.45 / 1.62 / 1.66; 020 1.83 / 1.78 / 1.85 / 1.89; 080 2.79 / 2.96 / 2.94 / 2.94; 150 1.86 / 2.15 / 2.45 / 2.18; 104 3.94 / 4.09 / 1.87 / 1.91. Anchors (only 2-6 per segment, pose sigma 0.57 deg) make no difference; the initial poses / motion f from the right rig matter only for seg104 (cost 38818 -> 33814: a bad pose basin at the prior). | ruled out for anchors; partial for poses (seg104 only) |
| 4 | why the Gauss-Newton covariance is 30-280x overconfident | Raw model sd(C_y) at the stalled point 0.01-0.025 m. Profile-likelihood sd along the ridge from the cost profile (Delta cost = 0.5): ~0.09 m (020), ~0.07 m (010): 4-8x the raw sd, i.e. 16-60x in variance, matching the measured factor 30; so the curvature at a point is wrong by ~sqrt(30), as the x30 inflation already says. The larger part is not curvature: the estimates sit 1.5-3 m from the optimum, i.e. 15-40 profile sd, because the solver stopped / stalled (raw sd even shrinks as the solver proceeds: 020 0.024 -> 0.008 m from tol 1e-4 to 1e-8). The prior rows (sigma 3-10 m) and anchor rows are negligible; Cauchy weights not isolated. For seg080 / 104 the optimum itself is wrong (row 1) so no covariance would cover it. | partial (curvature factor ~30 reproduced; remaining error is a non-converged point, not a covariance flaw) |
| 5 | why fit_shared across segments works: evidence diversity | Same prior start, tight LM, k1 fixed: seg010 alone (119 evidence frames) 1.46 m; 4 short segments pooled (146 ev) 1.80; all 5 pooled (265 ev, f ratio 1.85) 1.50. Count-matched: 60 random evidence frames from seg010 alone 1.78 / 1.86 / 1.82 m vs 60 spread over the 5 segments 2.01 / 2.00 / 2.08 m. Pooling does not move the estimate off the start-creep. The existing 0.16 m result therefore comes from multi-start (seeds near G on the (C_y, C_z) grid) and k1 free, not from pan/zoom diversity per se (the full 95-frame, 162-segment set was not rebuilt: only 5 segments have caches here). | ruled out as the explanation in this set-up (diversity does not fix a start-creeping solver) |

**Main causes.** (1) The L1 estimator is a local optimiser on a long, curved, nearly flat C_y / C_z / f valley whose floor is rough (truncated line-distance cost, finite differences): it creeps ~1 m from the start, then stops either on the 1e-4-relative decrease rule or on a no-improving-step stall. The objective itself is fine for pan-rich segments: from G the same code stays within 0.1-0.7 m of G at a lower cost than the prior-start estimate (chi2 gap 250-600). That is why every segment's error scales with its start (4 m -> ~1 m moved; the 7.6 m-off prior did not move) and why the shared bias survives the consensus. (2) For narrow-pan / zoomed segments (080, 104: pan 2-5 deg, f ~ 3150 / ~ 4900) the per-segment optimum is itself 2-4 m from G (cost at G higher by 44 / 1145), a genuine identifiability limit, plus for 104 a bad pose basin from the prior-rig initialisation. Anchors, k1, the covariance weights and segment pooling are not causes. The quoted covariance is wrong for the reason in row 4: a local curvature at a point that is not the optimum.

**Can a cheap fix make per-segment estimates usable? Probably only partly; recommendation, not built.** (a) Cheap: tight tolerance (1e-6, ~4-5x the iterations, 15-20 CPU-s per short segment, ~0.1 CPU-h for 5 segments) takes the median error from 2.7 to ~1.5-1.9 m, still start-dependent. (b) Needed: start diversity, i.e. a coarse (C_y, C_z) seed grid with fixed-C pose solves (1-5 s each for these segments; ~25-45 seeds, then polish the best 2 with tight tolerance); that is the existing `--rig-multistart` (median 1.83 m, < 1 m for 10 of 28) and has the same cost class (~0.37 CPU-s/frame, ~10 CPU-h for 162 segments). (c) Restrict to segments with pan range >= ~20 deg and f ratio >= 1.25 (about half of the 145; the narrow-pan ones are not identifiable, rows 1/3) and drop the per-segment covariance in favour of a robust median (what the multistart consensus already does, 0.17 m). Expected result: a pan-rich-only multistart consensus of the same quality as the current 95-frame `fit_shared` at roughly 5 CPU-h for no extra accuracy. Pooling segments into groups (row 5) is not a fix by itself; it only helps if combined with the seed grid. Conclusion unchanged: keep the cheap multi-start `fit_shared`; if a per-segment diagnostic is ever wanted, use the profile cost at the shared rig (fixed-C solve, 1-5 s a segment) as a deviation score rather than a per-segment rig fit.

Caveats: 5 segments, one match; G is the GT-points rig (a reference with its own ~0.1-0.2 m uncertainty); the line profile is one-dimensional (E -> G, roll interpolated) and the (C_y, C_z) grid at fixed C_x / roll is rough (rms 200-1100 cost units), so profile sds are order-of-magnitude; the 145-segment correlations are rank statistics over one run.
