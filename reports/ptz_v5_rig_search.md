# PTZ V5: which label-free rig-fit method for production? (experiment D)

Date: 2026-11-22. Branch `exp/ptz-rig-search` (from `exp/ptz-two-level-p3`). Baselines: `reports/ptz_v4_two_level.md` (sections 1, 5, appendix A), `reports/ptz_v3.md` section 6 and addendum A.
KB pages relied on: `prefect-job-system` (merlin move, `desktop` queue, host workers), `footy-datasets-and-progress` (no new facts needed); memory notes `mini-cpu-one-heavy-process`, `gpu-tests-via-prefect`.
Accuracy bar (user-confirmed): <= 1 m at ordinary depths, <= 2 m at the far touchline, > 5 m is a failure.

## 0. Decision

**Production rig fit = E2 with k1 free: `fit_shared_multistart` over one frame per segment, started from the generic prior, with a tight LM stopping rule (`rel_tol=1e-8`, `n_iter=100` on the finalists), then fixed-rig per-segment solves.** Cost: ~3.3 CPU-min for the fit (89 frames), vs 2.7 for the current rule and 10 for the grid search. Keep `rel_tol` default 1e-4 in the library (bit-identical production behaviour); the new setting is a call-site parameter.

What the data say, honestly:
1. **Every arm, from every start (4.0, 7.6, 7.5 m off), meets the accuracy bar on the GT frames.** Frame-median error 0.26-0.45 m, p95 0.47-0.74 m, 97-100% of points within 2 m, 100% within 5 m (one frame in 40 is unanswered in all arms, as in P3), far-touchline p95 1.4-1.8 m (< 2 m). No gross errors anywhere (18 rigs x 39 frames).
2. **The rig itself is NOT recovered reliably label-free.** Distance to the GT-points rig is 0.2-2.9 m across the 18 fits (table 2); no method is reliably within 1 m, and the data cost cannot tell the right rig from a wrong one: at fixed C the line cost at the GT-points rig (55804, k1 0.0147) is *higher* than at 13 of 18 fitted rigs (e.g. 55468 at an e1 rig 2.9 m away; `fixed_cost.json`). The 89-frame objective is flat to ~1% along the C_y/C_z/f ridge; "lowest cost wins" is not a reliable selector at this resolution. This reverses the P3 reading that the 0.17 m consensus rig was a success of the method: it was one sample from a ~1-3 m scatter.
3. **The ridge does not matter for the product.** A rig 2-3 m off costs only ~0.05-0.2 m of frame-median pitch error (the per-frame pose absorbs it), so the cheapest method with the most stable *pitch* error wins, not the one closest in C.
4. Among methods, the start-to-start spread (worst start) of the frame median is smallest for **E2 free (0.31 / 0.31 / 0.30 m, worst p95 0.61)**, then E3 free (0.26 / 0.30 / 0.34, but p95 up to 0.74 and far p95 1.80, at 3x the CPU). E1 is worst from the s2 start (0.42-0.45 m median, 0.72-0.74 p95, 86-87% within 1 m, paired +0.16-0.18 m vs B, p < 0.001) because the production stopping rule stalls on the ridge (appendix A of P3). k1 free beats k1 fixed at 0.03 by ~0.05-0.08 m of frame median for E2/E3 (0.31 vs 0.38, 0.30 vs 0.36), the opposite of the P3 k1 note (which found no effect); the sample is small, so treat k1 as "free is not worse".

So: E2-free, start = generic prior. Optional extra safety (not run): add E3's grid as an automatic cross-check and flag matches where E2 and E3 rigs differ by > 3 m.

## 1. What was run

All label-free. Unit set: **89** single-frame units (the middle frame of each of 162 Man City segments that the lines-only search accepts **at the generic prior rig** (0, -63, 14), k1 0.03, `generic_gantry_prior_v2.json`; `scripts/ptz_p5/anchors_generic.py`, 270 CPU-s). The old 95-frame set used anchors found with the leave-seg010-out *GT* rig; this one does not touch any GT rig (the cached evidence masks of `scratch/temporal/shared/mancity` are rig-independent). Poses for other starts are re-expressed from the generic-rig poses with `reseed_states`. Prior: loose sigma (5, 10, 3 m, 2 deg, 0.05) centred on the start rig, as in `fit_shared`. Free: C (x, y, z), roll, and k1 or not (k1 fixed 0.03 = line-derived).

- **E1**: `fit_shared_multistart`, as in `sharedms` (45 seeds, +-3/+-8/+-3 m grid around the start, screening on every 2nd frame, 3 finalists, 20 LM iterations per stage), production rule `rel_tol=1e-4`.
- **E2**: same, finalists with `rel_tol=1e-8`, `n_iter=100` (new parameter, section 5).
- **E3**: grid of fixed-C pose solves (C_y offsets -10..+10 step 2.5 m (9) x C_z offsets -6..+6 step 2 m (7) around the start = 63 grid points, C_x and roll at the start values (0, 0), all poses re-solved per grid point, all 89 frames), lowest 2 grid points then refined with C, roll (and k1) free under the E2 rule, lowest final cost wins. C_x and roll are freed only in the refine; a coarse grid over them was not warranted (they sit within 0.4 m / 0.4 deg everywhere, table 2).
- **Starts**: `generic` (0, -63, 14), 4.0 m from the reference; `s1` (0, -60, 12), 7.6 m (the prior that stalled the P3 per-segment L1); `s2` (2, -71, 10), 7.5 m on the other side. k1 start 0.03 everywhere.
- **Reference (evaluation only)**: GT-points rig (0.100, -66.576, 15.743), k1 0.0147. Used for distances, the data-cost diagnostic and arm A.
- **Scoring**: per rig, the production-style fixed-rig solve (recovery on) of the 10 GT segments of P3 (010, 020, 031, 050, 060, 080, 090, 104, 140, 150) through Prefect on the desktop (temporary deployment `ptz-temporal-match/ptz-temporal-match-exp`, code at `ba0328c`, 4 parallel), then `scripts/ptz_p3/score_gt.py` logic (`scripts/ptz_p5/score_arms.py`): 40 GT frames, 39 answered. The 4 skipped segments (000, 100, 110, 120; 8 frames) were not run (budget). Arms A and B are the P3 files (`ptz_v4_assets/A10.json`, `B.json`).

## 2. Rig fits (table 2) and GT scoring (table 1)

Table 1: GT scoring (error in metres of the 5 m grid of GT-visible pitch points; thirds by GT pitch y). Paired = per-frame median vs arm B (39 frames, Wilcoxon; "w/b" = frames worse/better than B).

| rig | answered | frame med / p90 / p95 (m) | within 1/2/5 m (%) | near med / p95 | mid med / p95 | far med / p95 | paired vs B (mean m; worse/better; p) |
|---|---|---|---|---|---|---|---|
| A (GT rig, contaminated ceiling) | 39/40 | 0.24 / 0.43 / 0.53 | 94 / 99 / 100 | 0.25 / 0.98 | 0.22 / 0.73 | 0.33 / 1.42 | -0.02 (16w/23b, p=0.176) |
| B (existing 95-frame fit) | 39/40 | 0.26 / 0.43 / 0.46 | 92 / 98 / 100 | 0.28 / 1.04 | 0.22 / 0.75 | 0.37 / 1.74 | +0.00 (0w/0b, p=nan) |
| e1_generic_k1fixed | 39/40 | 0.39 / 0.54 / 0.55 | 92 / 99 / 100 | 0.46 / 1.14 | 0.29 / 0.88 | 0.54 / 1.42 | +0.09 (33w/6b, p=0.000) |
| e2_generic_k1fixed | 39/40 | 0.38 / 0.59 / 0.62 | 91 / 99 / 100 | 0.47 / 1.27 | 0.27 / 0.86 | 0.52 / 1.39 | +0.10 (33w/6b, p=0.000) |
| e3_generic_k1fixed | 39/40 | 0.33 / 0.54 / 0.72 | 90 / 98 / 100 | 0.38 / 1.49 | 0.26 / 1.22 | 0.43 / 1.69 | +0.13 (29w/10b, p=0.002) |
| e1_s1_k1fixed | 39/40 | 0.32 / 0.51 / 0.70 | 90 / 98 / 100 | 0.35 / 1.49 | 0.27 / 1.22 | 0.40 / 1.71 | +0.11 (24w/15b, p=0.039) |
| e2_s1_k1fixed | 39/40 | 0.42 / 0.62 / 0.68 | 88 / 98 / 100 | 0.57 / 1.34 | 0.29 / 0.94 | 0.52 / 1.63 | +0.14 (34w/5b, p=0.000) |
| e3_s1_k1fixed | 39/40 | 0.30 / 0.56 / 0.61 | 92 / 99 / 100 | 0.47 / 1.21 | 0.23 / 0.77 | 0.36 / 1.58 | +0.03 (23w/16b, p=0.214) |
| e1_s2_k1fixed | 39/40 | 0.45 / 0.69 / 0.74 | 86 / 99 / 100 | 0.57 / 1.39 | 0.34 / 0.96 | 0.63 / 1.55 | +0.18 (36w/3b, p=0.000) |
| e2_s2_k1fixed | 39/40 | 0.33 / 0.55 / 0.59 | 93 / 99 / 100 | 0.38 / 0.98 | 0.27 / 0.78 | 0.38 / 1.55 | +0.05 (26w/13b, p=0.008) |
| e3_s2_k1fixed | 39/40 | 0.45 / 0.63 / 0.68 | 89 / 99 / 100 | 0.54 / 1.30 | 0.28 / 0.82 | 0.58 / 1.68 | +0.14 (34w/5b, p=0.000) |
| e1_generic_k1free | 39/40 | 0.39 / 0.58 / 0.58 | 92 / 99 / 100 | 0.46 / 1.23 | 0.28 / 0.86 | 0.54 / 1.41 | +0.10 (33w/6b, p=0.000) |
| e2_generic_k1free | 39/40 | 0.31 / 0.47 / 0.52 | 93 / 99 / 100 | 0.36 / 1.01 | 0.26 / 0.82 | 0.37 / 1.44 | +0.02 (25w/14b, p=0.108) |
| e3_generic_k1free | 39/40 | 0.26 / 0.44 / 0.47 | 93 / 99 / 100 | 0.28 / 1.07 | 0.23 / 0.78 | 0.34 / 1.64 | -0.01 (15w/24b, p=0.365) |
| e1_s1_k1free | 39/40 | 0.30 / 0.49 / 0.51 | 93 / 99 / 100 | 0.34 / 0.99 | 0.25 / 0.80 | 0.36 / 1.47 | +0.02 (24w/15b, p=0.236) |
| e2_s1_k1free | 39/40 | 0.31 / 0.53 / 0.61 | 92 / 99 / 100 | 0.40 / 1.20 | 0.24 / 0.82 | 0.42 / 1.53 | +0.06 (29w/10b, p=0.001) |
| e3_s1_k1free | 39/40 | 0.30 / 0.46 / 0.49 | 92 / 99 / 100 | 0.31 / 1.12 | 0.23 / 0.80 | 0.39 / 1.54 | +0.02 (20w/19b, p=0.176) |
| e1_s2_k1free | 39/40 | 0.42 / 0.67 / 0.72 | 87 / 99 / 100 | 0.56 / 1.40 | 0.33 / 0.96 | 0.59 / 1.40 | +0.16 (36w/3b, p=0.000) |
| e2_s2_k1free | 39/40 | 0.30 / 0.55 / 0.59 | 93 / 99 / 100 | 0.39 / 1.01 | 0.27 / 0.75 | 0.36 / 1.46 | +0.04 (25w/14b, p=0.035) |
| e3_s2_k1free | 39/40 | 0.34 / 0.57 / 0.74 | 89 / 97 / 100 | 0.41 / 1.51 | 0.26 / 1.23 | 0.47 / 1.80 | +0.15 (31w/8b, p=0.000) |

Table 2: fits. "Data cost" = robust line cost with the rig fixed and every pose re-optimised, same unit set, comparable across rows (reference rig: 55804 with k1 0.0147, 55694 with k1 0.03).

| rig | start | arm | k1 | C (x, y, z) | dist to ref (m) | k1 | roll deg | LM iters | CPU-s | data cost at rig |
|---|---|---|---|---|---|---|---|---|---|---|
| e1_generic_k1fixed | generic (4.0 m) | e1 | fixed | 0.09, -65.45, 15.09 | 1.30 | 0.0300 | -0.12 | 20 | 164 | 55770 |
| e2_generic_k1fixed | generic (4.0 m) | e2 | fixed | 0.24, -64.63, 15.01 | 2.08 | 0.0300 | -0.16 | 42 | 200 | 55547 |
| e3_generic_k1fixed | generic (4.0 m) | e3 | fixed | 0.26, -65.56, 15.30 | 1.13 | 0.0300 | -0.20 | 12 | 617 | 55547 |
| e1_s1_k1fixed | s1 (7.6 m) | e1 | fixed | 0.17, -65.97, 15.38 | 0.71 | 0.0300 | -0.18 | 16 | 158 | 55591 |
| e2_s1_k1fixed | s1 (7.6 m) | e2 | fixed | 0.39, -63.85, 15.29 | 2.78 | 0.0300 | -0.25 | 73 | 204 | 56064 |
| e3_s1_k1fixed | s1 (7.6 m) | e3 | fixed | 0.18, -68.98, 16.17 | 2.44 | 0.0300 | -0.17 | 28 | 569 | 55869 |
| e1_s2_k1fixed | s2 (7.5 m) | e1 | fixed | 0.00, -63.88, 14.73 | 2.89 | 0.0300 | -0.08 | 20 | 156 | 55468 |
| e2_s2_k1fixed | s2 (7.5 m) | e2 | fixed | 0.14, -67.42, 15.65 | 0.85 | 0.0300 | -0.15 | 28 | 218 | 55906 |
| e3_s2_k1fixed | s2 (7.5 m) | e3 | fixed | 1.30, -65.30, 15.27 | 1.82 | 0.0300 | -0.38 | 16 | 591 | 56376 |
| e1_generic_k1free | generic (4.0 m) | e1 | free | 0.14, -65.00, 15.00 | 1.75 | 0.0279 | -0.16 | 14 | 163 | 55578 |
| e2_generic_k1free | generic (4.0 m) | e2 | free | 0.13, -66.57, 15.49 | 0.26 | 0.0281 | -0.11 | 26 | 199 | 55926 |
| e3_generic_k1free | generic (4.0 m) | e3 | free | 0.26, -66.44, 15.69 | 0.22 | 0.0314 | -0.19 | 23 | 575 | 55601 |
| e1_s1_k1free | s1 (7.6 m) | e1 | free | 0.19, -66.59, 15.50 | 0.26 | 0.0260 | -0.17 | 15 | 157 | 55705 |
| e2_s1_k1free | s1 (7.6 m) | e2 | free | 0.45, -65.11, 15.37 | 1.56 | 0.0294 | -0.25 | 65 | 220 | 55767 |
| e3_s1_k1free | s1 (7.6 m) | e3 | free | 0.20, -65.95, 15.49 | 0.69 | 0.0350 | -0.19 | 27 | 569 | 55749 |
| e1_s2_k1free | s2 (7.5 m) | e1 | free | -0.00, -63.88, 14.73 | 2.88 | 0.0279 | -0.09 | 20 | 173 | 55499 |
| e2_s2_k1free | s2 (7.5 m) | e2 | free | 0.09, -67.59, 15.63 | 1.02 | 0.0236 | -0.14 | 28 | 231 | 55801 |
| e3_s2_k1free | s2 (7.5 m) | e3 | free | 0.36, -65.24, 15.26 | 1.45 | 0.0338 | -0.27 | 36 | 617 | 55598 |


Summary by arm over the three starts (frame median: mean / worst; worst p95; far p95 max; min within 1 m / 2 m):

| arm | k1 | frame median mean / worst | worst p95 | far p95 max | min within 1 m / 2 m (%) |
|---|---|---|---|---|---|
| E1 | fixed | 0.39 / 0.45 | 0.74 | 1.71 | 86 / 98 |
| E1 | free | 0.37 / 0.42 | 0.72 | 1.47 | 87 / 99 |
| E2 | fixed | 0.38 / 0.42 | 0.68 | 1.63 | 88 / 98 |
| **E2** | **free** | **0.31 / 0.31** | **0.61** | 1.53 | **92 / 99** |
| E3 | fixed | 0.36 / 0.45 | 0.72 | 1.69 | 89 / 98 |
| E3 | free | 0.30 / 0.34 | 0.74 | 1.80 | 89 / 97 |
| B (existing, GT-relative start) | free | 0.26 | 0.46 | 1.74 | 92 / 98 |
| A (GT rig, contaminated) | - | 0.24 | 0.53 | 1.42 | 94 / 99 |

## 3. Start sensitivity (the label-free robustness question)

- Rig distance by start (generic / s1 / s2, m): E1 fixed 1.30 / 0.71 / 2.89, E1 free 1.75 / 0.26 / 2.88; E2 fixed 2.08 / 2.78 / 0.85, E2 free 0.26 / 1.56 / 1.02; E3 fixed 1.13 / 2.44 / 1.82, E3 free 0.22 / 0.69 / 1.45. The final C moves by far less than the 7.5 m start offset (all 18 are within 2.9 m of the reference, i.e. the multi-start grid does escape bad starts), but there is a 1-3 m start-dependent scatter along C_y/C_z for every method. E1 from s2 (and generic k1 free) is the plain stall from appendix A: the finalist stops at C_y -63.9, 14.7 (the cost there is lower than at the reference).
- Pitch error by start: E2 free is flat (0.31/0.31/0.30 m); E3 free 0.26/0.30/0.34; E1 free 0.39/0.30/0.42. The start matters for the rig (C), much less for the product, and a bad start is not catastrophic in any arm (worst frame median 0.45 m).
- Why E2's tight rule helps k1-free more than k1-fixed: with k1 free, the extra direction lets the stalled finalist move along the valley; with k1 fixed it can end 2-3 m away (e2_s1_fixed: 2.78 m, cost 52326 in the fit, which is a *lower* cost than any other fit, with a higher pitch error, 0.42 m). The prior centred on a bad start also pulls (the rig prior sigma is loose but not flat).
- Not tested: starts in x, a start > 8 m off, other matches/grounds (Bournemouth), a second ground with a different gantry height. All 6 gantry-segment questions of P3 are untouched here.

## 4. Cost

| item | CPU-h |
|---|---|
| label-free anchors at generic prior (162 frames, merlin) | 0.075 |
| 18 rig fits (merlin, niced, one at a time): E1 ~160 s, E2 ~200-230 s, E3 ~570-620 s each | 1.61 |
| data-cost diagnostic (20 fixed-C solves) | ~0.05 |
| 18 fixed-rig re-solves x 10 segments (desktop; ~0.2 CPU-h per rig, summed from `meta.json` `cpu_s`) | 3.53 |
| **total** | **~5.3 (budget ~6)** |

Production cost of the recommendation: ~0.06 CPU-h per match for the rig fit (E2, 89 frames), plus the anchor search (~5 CPU-min over 162 segments), versus the ~0.1 CPU-h quoted for the old 95-frame fit. Wall-clock for the scoring runs on the desktop: ~3 min per rig (4 parallel).

## 5. Code

- `ptz_temporal.lm_solve(..., rel_tol=LM_REL_TOL)`; `LM_REL_TOL = 1e-4` keeps the production rule. `rel_tol` is threaded through `joint_solve`, `fit_shared` and `fit_shared_multistart` (finalists only, screening keeps the default). Test `test_lm_rel_tol_default_is_unchanged_and_tight_rule_runs_at_least_as_long`: omitting the argument is bit-identical to passing the default, and a tighter rule runs at least as many iterations and never ends at a higher cost.
- Scripts: `scripts/ptz_p5/{anchors_generic,rig_search,fixed_cost,score_arms,table}.py`; fitted rigs `reports/ptz_v5_assets/rigs/*.json` (+ `*.fit.json` with seeds / grid / finalists), per-frame scores `score/*.json`, tables `tables.md`, `fixed_cost.json`.
- Not changed: the driver's `sharedms` command (still the hard-coded keys / `leave_clip_out_rig` GT prior); a production flow should call `fit_shared_multistart` with `rel_tol=1e-8, n_iter=100` on units built like `scripts/ptz_p5/anchors_generic.py` / `rig_search.units_generic`, start rig = generic prior. Wiring it into the jobs repo was not done.

## 6. Caveats

- One match (Man City), one ground, 10 GT segments / 40 GT frames (39 answered), one unit set (89 frames, ~1 per segment); all paired p-values are on 39 frames and have no multiplicity correction (18 rigs).
- Differences between arms (0.05-0.15 m) are within the plausible noise of 39 frames; the safe statement is "all meet the bar, E2-free and E3-free are the most stable, E1 has a worst case from a bad start", not a ranking. The k1 free vs fixed difference is small and goes against the P3 k1 note.
- The rig is not identifiable to better than ~1-3 m from this data. Any claim that a method "recovers C" is unsupported (the P3 0.17 m was luck of the draw); use rig distance only as a diagnostic.
- Arm B's start was GT-relative; arms E* are not. The reference rig has its own 0.1-0.2 m uncertainty. The unit set is searched at a rig 4 m from the truth; the anchors found there may bias poses slightly (reseeding corrects the image of the pitch, not the camera's true pose basin).
- 4 GT segments (8 frames) not scored; 6 gantry segments not specifically examined.
- Fits were run on merlin (niced, single thread, ephemeral `uv run --no-project` env), the heavy per-frame solves on the desktop through Prefect; nothing broke. The desktop stayed up the whole time (no HA contact).
