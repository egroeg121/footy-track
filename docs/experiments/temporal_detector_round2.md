# Round 2: does pseudo-labelled data (v11 train + filtered pseudo) change the 3ch / 5ch comparison?

Branch `exp/temporal-detector`. Code: `experiments/temporal_detector/{pseudo_filter,pseudo_audit_survivors,round2_report}.py`, flow
`temporal-ablation/temporal-ablation ... pseudo=true exp_tag=r2` (`~/code/jobs/footy_temporal.py`). Builds on `temporal_detector_ablation.md` (v1)
and `temporal_pipeline_eval.md`. Pseudo labels: `/mnt/storage/footy_data/multiframe_pseudo_v1/` (branch exp/pseudo-labels-v1 7c53b25).

## Questions and answers

Same recipe as v1 (YOLO11s, imgsz 1280, batch 4, HSV and Albumentations off, `last.pt`), but train = v11 train (1,034) + 1,617 filtered
pseudo-labelled images, **60 epochs** (v1: 100; the larger set gives 663 vs 259 iterations per epoch, so about 40k vs 26k in total), 2 seeds
each: A+ (3ch, same pixels as PNG) and B+ (5ch, inflated stem). Val, test, heldout, strict and the metrics are exactly v1's, so every number
compares directly with v1's A / B.

1. **Does pseudo data close the seed variance? Yes.** Seed range of hit@10 (A / B, v1 -> round 2): val 0.153 / 0.081 -> 0.038 / 0.000;
   val clip arsenal_example 0.186 / 0.103 -> 0.051 / 0.006; test 0.013 / 0.008 -> 0.004 / 0.004; held-out 0.034 / 0.017 -> 0.000 / 0.000
   (both seeds 56/117 for both arms). Ball AP50 on val at the last epoch: A_s1 0.369 vs A_s0 0.509 in v1, Ap 0.606 / 0.657 and Bp 0.624 / 0.669 now.
2. **Does it raise held-out and arsenal_example recall? It raises in-domain recall, not held-out.**
   - Test (same match and broadcast as the pseudo pool; 33 test clips share a clip with train): A+ - A **+0.076 [+0.046, +0.109]**, B+ - B **+0.065 [+0.038, +0.095]**
     (A 0.880 -> 0.956, B 0.878 -> 0.943; RT-DETR 0.916). Candidate recall any@10 +0.055 / +0.044.
   - val clip arsenal_example (156 balls, no train frames): A+ - A **+0.221 [+0.157, +0.285]** (0.375 -> 0.596), B+ - B +0.029 [-0.026, +0.083]
     (0.583 -> 0.612): A catches up with B (RT-DETR 0.667). Other val clips: +0.047 [-0.009, +0.123] / +0.113 [+0.047, +0.189].
   - Held-out clips (117 balls, 11 clips, other matches / provenance than the pool; strict 84): A+ - A +0.017 [-0.004, +0.043], B+ - B +0.000
     [-0.038, +0.038]; strict +0.018 / +0.000. **No gain**: A+ and B+ both 0.479 against RT-DETR 0.487; candidate recall any@10 stays at 0.54-0.57 (RT-DETR 0.675),
     i.e. the held-out misses are balls no candidate covers, and more same-match pseudo data does not fix them.
3. **Does B - A widen with more data? No, it narrows or vanishes.** B+ - A+ hit@10: val +0.019 [-0.017, +0.055] (v1 +0.146), test -0.013 [-0.029, +0.002]
   (v1 -0.002), held-out +0.000 [-0.030, +0.030] (v1 +0.017), test+heldout -0.008 [-0.024, +0.006]. The difference of differences (B+ - A+) - (B - A) is
   -0.127 [-0.187, -0.067] on val, entirely arsenal_example (-0.192 [-0.266, -0.119]: v1's 5ch advantage there was a small-data effect that 3ch closes with
   pseudo data), and -0.011 / -0.017 (n.s.) on test / held-out. Candidate recall B+ - A+ is ~0 everywhere (test -0.004, held-out +0.026 [-0.021, +0.077]); the v1 gain for
   the fastest third of balls (+0.066) shrinks to +0.035 [-0.004, +0.075].
4. **Through the trajectory solver** (appendix; harness B1 camera on, the 9 test clips, 170 balls): A+ 0.794 raw / 0.826 iso, B+ 0.829 / 0.844 recall@10, precision 0.99 / 0.98-0.99.
   B+ - A+ +0.035 [+0.009, +0.067] raw, +0.018 [0.000, +0.036] iso (v1 +0.047 / +0.029): the 5ch edge under the solver survives but did not widen.
   Both round-2 detectors now **exceed the production RT-DETR-L (0.759) on these clips**: B+ - RT-DETR +0.071 [+0.033, +0.111] raw, +0.085 [+0.047, +0.126] iso at
   unchanged precision (-0.005 to +0.012), ceiling 0.838 vs 0.829. On arsenal_example (all labelled frames, the labelled deviation) A+ 0.264 / 0.457, B+ 0.256 / 0.420 (raw / iso), B+ - A+ -0.009 [-0.048, +0.031]
   (v1 +0.071), and RT-DETR (0.540) is still well ahead (B+ - RT-DETR -0.284 raw, -0.119 iso).
5. **False positives on ball-free images** (35 images of val+test+heldout, small n): top-1 conf >= 0.5: A 0.171 -> A+ 0.157, B 0.157 -> B+ 0.186; conf >= 0.7: 0.071 -> 0.114 (A+),
   0.071 -> 0.143 (B+); confident wrong top-1 picks on ball frames (conf >= 0.5): 0.008 -> 0.013 (A+), 0.005 -> 0.009 (B+). A small rise at high confidence (2-3 images of 35,
   within noise), no sign of a crowd/kit leak at the scale visible here; RT-DETR remains the noisiest (31% at conf 0.5, 4.1% confident wrong picks).

**Reading.** The pseudo data is worth about +0.07 test recall and removes seed noise for both input types, so most of v1's A-vs-B difference was a low-data / high-variance effect rather than a lasting
benefit of the temporal channels. A 5ch detector keeps a small edge only after a solver ranks its candidates (+0.02 to +0.04 on the test clips) and nothing measurable on held-out clips.
The filter rule (conf >= 0.45 and a conf >= 0.8 node within 25 frames) was not ablated: no run with all 759 low-conf B1 images or with the 241 + 257 dropped ones exists, so the effect of the
filtering itself is not measured. Epochs differ from v1 (60 vs 100), so "round 2 vs v1" mixes data and schedule.

## Filter and data

- **Counts:** of 2,115 pseudo images 1,617 survive: 1,000 easy (conf >= 0.8) + 356 S1 fills + **261 of 759** low-conf B1 (dropped: 241 with conf < 0.45, 257 with no conf >= 0.8 B1 node within +-25 frames).
  Segment ids are not in `meta.jsonl`, so segments were rebuilt from the provenance table (`provenance_all.jsonl.gz`): consecutive B1 rows with gaps <= 25 frames, so the rule
  reduces to "a conf >= 0.8 B1 detection within +-25 frames"; cuts inside the window are not checked.
- **Audit hit@10 of the survivors** (the pseudo doc's audit pipeline: the harness' B1 re-run on the 18 GT-bearing harness clips + arsenal_example; B1 detections with conf < 0.6 on a human GT frame):
  all low-conf 38/50 (76%; the doc's 38/49), **survivors 13/15 (86.7%)** (test 3/4, dev 8/9, arsenal_example 2/2). n is tiny; S1 fills (19/20 in the doc) and easy images (138/138) are unchanged.
- **Eval overlap: zero, checked explicitly:** no pseudo frame lies within +-5 frames of a v11 val/test/heldout image or in a clip with one (the pool excludes all 97 such clips),
  and none belongs to arsenal_example or the 9 trajectory test clips (3 of them excluded from the pool by the pseudo build). The `near_v11_train_5` frames (91) are kept; `near_human_gt_5`
  frames (52) are of pool clips that are not evaluation clips and are kept.
- **Transfer:** only the 1,617 survivors (6.05 GB) went to the desktop, via S3 (new prefix `exp_temporal_r2/pseudo_v1/`, upload from the mini about 4 MB/s, download by presigned-URL curl,
  resumable); the direct mini -> desktop rsync measured 1.2-1.8 MB/s over the Wi-Fi. Sizes verified against the mini copy; presigned URL list deleted from the desktop.
- **Training cost:** 2.0-2.1 h per 60-epoch run (about 2 min/epoch, GPU not shared, no crash); four runs about 8.3 h.

## Top-1 recall@10 px (hit@10), mean of 2 seeds [seed range]

| model | val (n=209) | val / clip arsenal_example (74% of val balls) (n=156) | val / all other clips (n=53) | test (n=238) | heldout (n=117) | heldout_strict (n=84) | test+heldout (n=355) | test+heldout / ball motion low (n=113) | test+heldout / ball motion mid (n=115) | test+heldout / ball motion high (n=114) |
|---|---|---|---|---|---|---|---|---|---|---|
| RT-DETR-L v11 (production) | 0.718 | 0.667 | 0.868 | 0.916 | 0.487 | 0.524 | 0.775 | 0.779 | 0.852 | 0.719 |
| A  3ch, v1 | 0.493 [0.153] | 0.375 [0.186] | 0.840 [0.057] | 0.880 [0.013] | 0.462 [0.034] | 0.494 [0.036] | 0.742 [0.020] | 0.743 [0.018] | 0.822 [0.009] | 0.693 [0.035] |
| A+ 3ch, + pseudo | 0.670 [0.038] | 0.596 [0.051] | 0.887 | 0.956 [0.004] | 0.479 | 0.512 | 0.799 [0.003] | 0.823 | 0.891 [0.009] | 0.715 [0.009] |
| B  5ch, v1 | 0.639 [0.081] | 0.583 [0.103] | 0.802 [0.019] | 0.878 [0.008] | 0.479 [0.017] | 0.512 [0.024] | 0.746 [0.011] | 0.735 [0.018] | 0.826 | 0.711 [0.018] |
| B+ 5ch, + pseudo | 0.689 | 0.612 [0.006] | 0.915 [0.019] | 0.943 [0.004] | 0.479 | 0.512 | 0.790 [0.003] | 0.805 | 0.883 [0.009] | 0.711 |

## Candidate recall any@10

| model | val (n=209) | val / clip arsenal_example (74% of val balls) (n=156) | val / all other clips (n=53) | test (n=238) | heldout (n=117) | heldout_strict (n=84) | test+heldout (n=355) | test+heldout / ball motion low (n=113) | test+heldout / ball motion mid (n=115) | test+heldout / ball motion high (n=114) |
|---|---|---|---|---|---|---|---|---|---|---|
| RT-DETR-L v11 (production) | 0.833 | 0.795 | 0.943 | 0.962 | 0.675 | 0.750 | 0.868 | 0.832 | 0.930 | 0.833 |
| A  3ch, v1 | 0.596 [0.120] | 0.503 [0.147] | 0.868 [0.038] | 0.920 [0.017] | 0.513 [0.017] | 0.560 [0.024] | 0.786 [0.017] | 0.774 [0.009] | 0.883 [0.043] | 0.732 [0.009] |
| A+ 3ch, + pseudo | 0.849 [0.005] | 0.811 [0.006] | 0.962 | 0.975 | 0.543 [0.026] | 0.601 [0.036] | 0.832 [0.008] | 0.845 [0.027] | 0.930 | 0.750 [0.009] |
| B  5ch, v1 | 0.730 [0.014] | 0.683 [0.006] | 0.868 [0.075] | 0.926 [0.029] | 0.568 [0.009] | 0.625 [0.012] | 0.808 [0.023] | 0.774 [0.027] | 0.891 [0.026] | 0.798 [0.018] |
| B+ 5ch, + pseudo | 0.844 [0.005] | 0.811 [0.006] | 0.943 | 0.971 [0.025] | 0.568 [0.026] | 0.577 [0.012] | 0.838 [0.008] | 0.836 [0.027] | 0.926 [0.009] | 0.785 [0.009] |

## Ball AP50

| model | val (n=209) | val / clip arsenal_example (74% of val balls) (n=156) | val / all other clips (n=53) | test (n=238) | heldout (n=117) | heldout_strict (n=84) | test+heldout (n=355) | test+heldout / ball motion low (n=113) | test+heldout / ball motion mid (n=115) | test+heldout / ball motion high (n=114) |
|---|---|---|---|---|---|---|---|---|---|---|
| RT-DETR-L v11 (production) | 0.658 | 0.636 | 0.734 | 0.881 | 0.464 | 0.514 | 0.750 | 0.749 | 0.823 | 0.717 |
| A  3ch, v1 | 0.441 [0.129] | 0.336 [0.190] | 0.702 [0.021] | 0.831 [0.009] | 0.435 [0.050] | 0.470 [0.061] | 0.703 [0.007] | 0.706 [0.003] | 0.789 [0.009] | 0.677 [0.017] |
| A+ 3ch, + pseudo | 0.623 [0.059] | 0.559 [0.095] | 0.792 [0.022] | 0.920 [0.005] | 0.445 [0.005] | 0.501 [0.028] | 0.771 [0.002] | 0.799 [0.016] | 0.861 [0.001] | 0.704 [0.013] |
| B  5ch, v1 | 0.581 [0.078] | 0.554 [0.087] | 0.678 [0.073] | 0.837 [0.006] | 0.446 [0.022] | 0.489 [0.026] | 0.714 [0.013] | 0.692 [0.028] | 0.802 [0.005] | 0.709 [0.000] |
| B+ 5ch, + pseudo | 0.643 [0.027] | 0.583 [0.025] | 0.792 [0.035] | 0.914 [0.014] | 0.456 [0.016] | 0.508 [0.006] | 0.771 [0.010] | 0.789 [0.017] | 0.858 [0.005] | 0.715 [0.003] |

## Paired differences in hit@10 (bootstrap over images; seed-mean outcomes; clip bootstrap in brackets)

| subset | n | A+ - A (v1) | B+ - B (v1) | B+ - A+ (round 2) | B - A (v1) | DiD (B+ - A+) - (B - A) |
|---|---|---|---|---|---|---|
| val | 209 | +0.177 [+0.127, +0.227] (+0.000, +0.210) | +0.050 [+0.005, +0.096] (+0.033, +0.153) | +0.019 [-0.017, +0.055] (+0.000, +0.051) | +0.146 [+0.100, +0.194] (-0.083, +0.190) | -0.127 [-0.187, -0.067] (-0.173, +0.107) |
| val / clip arsenal_example (74% of val balls) | 156 | +0.221 [+0.157, +0.285] (+0.221, +0.221) | +0.029 [-0.026, +0.083] (+0.029, +0.029) | +0.016 [-0.032, +0.061] (+0.016, +0.016) | +0.208 [+0.154, +0.263] (+0.208, +0.208) | -0.192 [-0.266, -0.119] (-0.192, -0.192) |
| val / all other clips | 53 | +0.047 [-0.009, +0.123] (-0.011, +0.135) | +0.113 [+0.047, +0.189] (+0.058, +0.170) | +0.028 [+0.000, +0.066] (+0.000, +0.057) | -0.038 [-0.113, +0.028] (-0.098, +0.029) | +0.066 [+0.000, +0.151] (+0.000, +0.118) |
| test | 238 | +0.076 [+0.046, +0.109] (+0.045, +0.108) | +0.065 [+0.038, +0.095] (+0.038, +0.095) | -0.013 [-0.029, +0.002] (-0.032, +0.005) | -0.002 [-0.023, +0.019] (-0.024, +0.020) | -0.011 [-0.036, +0.015] (-0.039, +0.018) |
| heldout | 117 | +0.017 [-0.004, +0.043] (+0.000, +0.049) | +0.000 [-0.038, +0.038] (-0.023, +0.030) | +0.000 [-0.030, +0.030] (-0.021, +0.019) | +0.017 [-0.013, +0.047] (-0.019, +0.059) | -0.017 [-0.056, +0.021] (-0.060, +0.020) |
| heldout_strict | 84 | +0.018 [-0.012, +0.048] (+0.000, +0.067) | +0.000 [-0.054, +0.054] (-0.021, +0.040) | +0.000 [-0.036, +0.042] (-0.021, +0.023) | +0.018 [-0.024, +0.060] (-0.037, +0.083) | -0.018 [-0.065, +0.030] (-0.069, +0.029) |
| test+heldout | 355 | +0.056 [+0.035, +0.080] (+0.033, +0.087) | +0.044 [+0.020, +0.068] (+0.021, +0.072) | -0.008 [-0.024, +0.006] (-0.024, +0.004) | +0.004 [-0.013, +0.023] (-0.015, +0.023) | -0.013 [-0.034, +0.008] (-0.036, +0.010) |
| test+heldout / ball motion low | 113 | +0.080 [+0.035, +0.128] (+0.036, +0.136) | +0.071 [+0.031, +0.119] (+0.032, +0.120) | -0.018 [-0.040, +0.000] (-0.043, +0.000) | -0.009 [-0.040, +0.027] (-0.043, +0.026) | -0.009 [-0.049, +0.027] (-0.051, +0.031) |
| test+heldout / ball motion mid | 115 | +0.070 [+0.030, +0.113] (+0.033, +0.110) | +0.057 [+0.017, +0.104] (+0.017, +0.099) | -0.009 [-0.039, +0.022] (-0.039, +0.020) | +0.004 [-0.030, +0.039] (-0.028, +0.036) | -0.013 [-0.052, +0.026] (-0.056, +0.029) |
| test+heldout / ball motion high | 114 | +0.022 [-0.004, +0.053] (-0.008, +0.064) | +0.000 [-0.035, +0.039] (-0.036, +0.044) | -0.004 [-0.031, +0.022] (-0.022, +0.010) | +0.018 [-0.009, +0.044] (-0.006, +0.039) | -0.022 [-0.053, +0.013] (-0.045, +0.005) |

## Paired differences in candidate recall any@10

| subset | A+ - A (v1) | B+ - B (v1) | B+ - A+ (round 2) | B - A (v1) |
|---|---|---|---|---|
| val | +0.254 [+0.201, +0.309] | +0.115 [+0.077, +0.156] | -0.005 [-0.029, +0.017] | +0.134 [+0.096, +0.175] |
| val / clip arsenal_example (74% of val balls) | +0.308 [+0.244, +0.372] | +0.128 [+0.083, +0.176] | +0.000 [-0.029, +0.029] | +0.179 [+0.131, +0.231] |
| val / all other clips | +0.094 [+0.019, +0.179] | +0.075 [+0.019, +0.151] | -0.019 [-0.047, +0.000] | +0.000 [-0.047, +0.057] |
| test | +0.055 [+0.029, +0.082] | +0.044 [+0.021, +0.069] | -0.004 [-0.017, +0.008] | +0.006 [-0.011, +0.025] |
| heldout | +0.030 [-0.004, +0.068] | +0.000 [-0.060, +0.060] | +0.026 [-0.021, +0.077] | +0.056 [+0.009, +0.107] |
| heldout_strict | +0.042 [-0.006, +0.095] | -0.048 [-0.113, +0.018] | -0.024 [-0.071, +0.024] | +0.065 [+0.006, +0.131] |
| test+heldout | +0.046 [+0.025, +0.069] | +0.030 [+0.004, +0.055] | +0.006 [-0.011, +0.023] | +0.023 [+0.004, +0.042] |
| test+heldout / ball motion low | +0.071 [+0.027, +0.119] | +0.062 [+0.022, +0.106] | -0.009 [-0.031, +0.013] | +0.000 [-0.031, +0.031] |
| test+heldout / ball motion mid | +0.048 [+0.017, +0.083] | +0.035 [+0.004, +0.070] | -0.004 [-0.035, +0.026] | +0.009 [-0.017, +0.035] |
| test+heldout / ball motion high | +0.018 [-0.018, +0.057] | -0.013 [-0.070, +0.039] | +0.035 [-0.004, +0.075] | +0.066 [+0.022, +0.114] |

## False positives on ball-free images (n = 35, val+test+heldout)

| model | top-1 conf >= 0.1 | >= 0.25 | >= 0.5 | >= 0.7 | confident wrong top-1 on ball frames (conf >= 0.5, fraction of ball frames) |
|---|---|---|---|---|---|
| RT-DETR-L v11 (production) | 1.000 | 0.771 | 0.314 | 0.171 | 0.041 |
| A  3ch, v1 | 0.357 | 0.214 | 0.171 | 0.071 | 0.008 |
| A+ 3ch, + pseudo | 0.386 | 0.214 | 0.157 | 0.114 | 0.013 |
| B  5ch, v1 | 0.343 | 0.243 | 0.157 | 0.071 | 0.005 |
| B+ 5ch, + pseudo | 0.371 | 0.257 | 0.186 | 0.143 | 0.009 |

## Training curves (val ball AP50)

| run | epochs | last | mean last 10 | best |
|---|---|---|---|---|
| A_s0 | 100 | 0.509 | 0.516 | 0.540 |
| A_s1 | 100 | 0.369 | 0.373 | 0.564 |
| B_s0 | 100 | 0.626 | 0.621 | 0.642 |
| B_s1 | 100 | 0.528 | 0.509 | 0.530 |
| Ap_s0 | 60 | 0.606 | 0.583 | 0.641 |
| Ap_s1 | 60 | 0.657 | 0.668 | 0.680 |
| Bp_s0 | 60 | 0.624 | 0.658 | 0.695 |
| Bp_s1 | 60 | 0.669 | 0.644 | 0.671 |

# Appendix: round-2 models (A+ 3ch / B+ 5ch) through the harness trajectory solver (B0 / B1, camera on)
Same protocol, candidates and exclusions as docs/experiments/temporal_pipeline_eval.md (v1 models there); iso maps fitted on v11 train only. arsenal_example harness-clean has n = 9 and is uninformative; the all-labelled-frames table is the labelled deviation.

## test (9 clips, harness exclusions)

172 scored frames; n_gt = 170. recall@10 = emitted point within 10 px of GT (denominator: GT ball frames); precision = TP / (TP + FP) over scored frames (FP = emitted point > 10 px away, or emitted on a `no_ball` frame); ceiling = any candidate with conf >= 0.1 within 10 px (candidate-set recall, identical for B0 and B1).

| detector | calibration | B0 recall / prec | B1 recall / prec | ceiling |
|---|---|---|---|---|
| RT-DETR-L production (frame t) | raw | 0.741 / 0.947 | 0.759 / 0.985 | 0.829 |
| RT-DETR-L rerun by this pipeline | raw | 0.741 / 0.947 | 0.759 / 0.985 | 0.829 |
| A 3ch Ap_s0 | raw | 0.788 / 0.993 | 0.788 / 0.985 | 0.812 |
| A 3ch Ap_s0 | iso | 0.806 / 0.993 | 0.829 / 0.993 | 0.812 |
| A 3ch Ap_s1 | raw | 0.794 / 0.993 | 0.800 / 0.993 | 0.829 |
| A 3ch Ap_s1 | iso | 0.818 / 0.986 | 0.824 / 0.993 | 0.829 |
| A 3ch mean of seeds | raw | 0.791 / 0.993 | 0.794 / 0.989 | 0.821 |
| A 3ch mean of seeds | iso | 0.812 / 0.989 | 0.826 / 0.993 | 0.821 |
| B 5ch Bp_s0 | raw | 0.800 / 0.993 | 0.829 / 1.000 | 0.841 |
| B 5ch Bp_s0 | iso | 0.829 / 0.959 | 0.853 / 0.986 | 0.841 |
| B 5ch Bp_s1 | raw | 0.806 / 0.993 | 0.829 / 0.993 | 0.835 |
| B 5ch Bp_s1 | iso | 0.824 / 0.979 | 0.835 / 0.973 | 0.835 |
| B 5ch mean of seeds | raw | 0.803 / 0.993 | 0.829 / 0.996 | 0.838 |
| B 5ch mean of seeds | iso | 0.826 / 0.969 | 0.844 / 0.979 | 0.838 |

Paired bootstrap (10,000 resamples of scored frames; seed-mean per-frame outcomes for the arms), B1 camera on:

| comparison | calibration | recall@10 diff [95% CI] | precision diff [95% CI] | ceiling diff [95% CI] | per-seed recall diff |
|---|---|---|---|---|---|
| B1(5ch) - B1(3ch) | raw | +0.035 [+0.009, +0.067] | +0.007 [+0.000, +0.019] | +0.018 [-0.003, +0.039] | s0: +0.041, s1: +0.029 |
| B1(5ch) - B1(3ch) | iso | +0.018 [+0.000, +0.036] | -0.013 [-0.034, +0.004] | +0.018 [-0.003, +0.041] | s0: +0.024, s1: +0.012 |
| B1(5ch) - B1(RT-DETR) | raw | +0.071 [+0.033, +0.111] | +0.012 [-0.007, +0.037] | +0.009 [-0.030, +0.047] |  |
| B1(5ch) - B1(RT-DETR) | iso | +0.085 [+0.047, +0.126] | -0.005 [-0.033, +0.023] | +0.009 [-0.030, +0.047] |  |

## arsenal_example, harness-clean frames

9 scored frames; n_gt = 9. recall@10 = emitted point within 10 px of GT (denominator: GT ball frames); precision = TP / (TP + FP) over scored frames (FP = emitted point > 10 px away, or emitted on a `no_ball` frame); ceiling = any candidate with conf >= 0.1 within 10 px (candidate-set recall, identical for B0 and B1).

| detector | calibration | B0 recall / prec | B1 recall / prec | ceiling |
|---|---|---|---|---|
| RT-DETR-L production (frame t) | raw | 0.111 / 0.200 | 0.111 / 0.250 | 0.111 |
| RT-DETR-L rerun by this pipeline | raw | 0.111 / 0.200 | 0.111 / 0.250 | 0.111 |
| A 3ch Ap_s0 | raw | 0.000 / n/a | 0.000 / n/a | 0.111 |
| A 3ch Ap_s0 | iso | 0.111 / 0.333 | 0.111 / 0.500 | 0.111 |
| A 3ch Ap_s1 | raw | 0.000 / n/a | 0.000 / n/a | 0.111 |
| A 3ch Ap_s1 | iso | 0.000 / n/a | 0.000 / n/a | 0.111 |
| A 3ch mean of seeds | raw | 0.000 / n/a | 0.000 / n/a | 0.111 |
| A 3ch mean of seeds | iso | 0.056 / n/a | 0.056 / n/a | 0.111 |
| B 5ch Bp_s0 | raw | 0.000 / n/a | 0.000 / n/a | 0.111 |
| B 5ch Bp_s0 | iso | 0.111 / 0.333 | 0.111 / 0.500 | 0.111 |
| B 5ch Bp_s1 | raw | 0.000 / n/a | 0.000 / n/a | 0.000 |
| B 5ch Bp_s1 | iso | 0.000 / 0.000 | 0.000 / 0.000 | 0.000 |
| B 5ch mean of seeds | raw | 0.000 / n/a | 0.000 / n/a | 0.056 |
| B 5ch mean of seeds | iso | 0.056 / 0.167 | 0.056 / 0.250 | 0.056 |

Paired bootstrap (10,000 resamples of scored frames; seed-mean per-frame outcomes for the arms), B1 camera on:

| comparison | calibration | recall@10 diff [95% CI] | precision diff [95% CI] | ceiling diff [95% CI] | per-seed recall diff |
|---|---|---|---|---|---|
| B1(5ch) - B1(3ch) | raw | +0.000 [+0.000, +0.000] | +nan [+nan, +nan] | -0.056 [-0.167, +0.000] | s0: +0.000, s1: +0.000 |
| B1(5ch) - B1(3ch) | iso | +0.000 [+0.000, +0.000] | -0.167 [-0.800, +0.667] | -0.056 [-0.167, +0.000] | s0: +0.000, s1: +0.000 |
| B1(5ch) - B1(RT-DETR) | raw | -0.111 [-0.333, +0.000] | +nan [+nan, +nan] | -0.056 [-0.167, +0.000] |  |
| B1(5ch) - B1(RT-DETR) | iso | -0.056 [-0.167, +0.000] | +0.083 [-0.357, +0.833] | -0.056 [-0.167, +0.000] |  |

## arsenal_example, all labelled frames (NOT harness-clean)

176 scored frames; n_gt = 176. recall@10 = emitted point within 10 px of GT (denominator: GT ball frames); precision = TP / (TP + FP) over scored frames (FP = emitted point > 10 px away, or emitted on a `no_ball` frame); ceiling = any candidate with conf >= 0.1 within 10 px (candidate-set recall, identical for B0 and B1).

| detector | calibration | B0 recall / prec | B1 recall / prec | ceiling |
|---|---|---|---|---|
| RT-DETR-L production (frame t) | raw | 0.540 / 0.856 | 0.540 / 0.913 | 0.631 |
| RT-DETR-L rerun by this pipeline | raw | 0.540 / 0.856 | 0.540 / 0.913 | 0.631 |
| A 3ch Ap_s0 | raw | 0.199 / 0.854 | 0.216 / 0.884 | 0.443 |
| A 3ch Ap_s0 | iso | 0.415 / 0.716 | 0.398 / 0.805 | 0.443 |
| A 3ch Ap_s1 | raw | 0.307 / 0.915 | 0.312 / 0.902 | 0.580 |
| A 3ch Ap_s1 | iso | 0.489 / 0.835 | 0.517 / 0.843 | 0.580 |
| A 3ch mean of seeds | raw | 0.253 / 0.884 | 0.264 / 0.893 | 0.511 |
| A 3ch mean of seeds | iso | 0.452 / 0.775 | 0.457 / 0.824 | 0.511 |
| B 5ch Bp_s0 | raw | 0.227 / 0.909 | 0.250 / 0.957 | 0.483 |
| B 5ch Bp_s0 | iso | 0.438 / 0.755 | 0.392 / 0.802 | 0.483 |
| B 5ch Bp_s1 | raw | 0.256 / 0.938 | 0.261 / 0.939 | 0.483 |
| B 5ch Bp_s1 | iso | 0.415 / 0.830 | 0.449 / 0.859 | 0.483 |
| B 5ch mean of seeds | raw | 0.241 / 0.923 | 0.256 / 0.948 | 0.483 |
| B 5ch mean of seeds | iso | 0.426 / 0.792 | 0.420 / 0.831 | 0.483 |

Paired bootstrap (10,000 resamples of scored frames; seed-mean per-frame outcomes for the arms), B1 camera on:

| comparison | calibration | recall@10 diff [95% CI] | precision diff [95% CI] | ceiling diff [95% CI] | per-seed recall diff |
|---|---|---|---|---|---|
| B1(5ch) - B1(3ch) | raw | -0.009 [-0.048, +0.031] | +0.053 [+0.002, +0.112] | -0.028 [-0.074, +0.017] | s0: +0.034, s1: -0.051 |
| B1(5ch) - B1(3ch) | iso | -0.037 [-0.082, +0.009] | +0.006 [-0.048, +0.056] | -0.028 [-0.074, +0.020] | s0: -0.006, s1: -0.068 |
| B1(5ch) - B1(RT-DETR) | raw | -0.284 [-0.364, -0.202] | +0.034 [-0.037, +0.106] | -0.148 [-0.210, -0.085] |  |
| B1(5ch) - B1(RT-DETR) | iso | -0.119 [-0.196, -0.043] | -0.082 [-0.164, -0.003] | -0.148 [-0.210, -0.085] |  |

## Checks

- Harness reproduction on the 9 test clips (RT-DETR machine labels, camera-on B1): recall 0.759 (129 TP of 170), precision 0.985; B0 tau 0.4 recall 0.741, precision 0.947 (harness doc: 0.759 / 0.985 and 0.741 / 0.947).
- RT-DETR rerun by this pipeline vs the harness's machine labels (top-1 per frame with conf >= 0.3, same position within 3 px and conf within 0.05): seg010 549/549; seg011 609/610; seg031 311/312; seg094 2293/2294; seg101 212/212; seg115 2126/2132; seg134 426/426; seg135 680/680; seg140 347/348.
- Neighbour status per clip (frames, [prev, next] slots): seg010 ok:1182/edge:4/cut:4; seg011 ok:1510/edge:4/cut:4/align_fail:2; seg031 ok:742/edge:4/cut:4; seg094 ok:5032/edge:4/cut:4; seg101 ok:432/edge:4/cut:4; seg115 ok:4594/edge:4/align_fail:42; seg134 ok:866/edge:4; seg135 ok:1462/edge:4/cut:4; seg140 ok:702/edge:4/cut:4; arsenal_example ok:496/edge:4.
- Isotonic maps (fitted on v11 train candidates within 10 px of a labelled ball): Ap_s0: 1004/1088 correct, min_conf 0.10, B0 tau 0.10; Ap_s1: 1005/1108 correct, min_conf 0.13, B0 tau 0.13; Bp_s0: 1019/1120 correct, min_conf 0.10, B0 tau 0.11; Bp_s1: 1001/1076 correct, min_conf 0.11, B0 tau 0.16.

Solver parameters (harness `camera_on`): `{"min_conf": 0.3, "bias": 1.0, "emit_scale": 1.0, "kappa": 1.5, "lam_pos": 0.5, "lam_vel": 0.0, "vmax": 40.0, "r0": 12.0, "dmax": 2.0, "mu": 0.02, "static_px": 3.0, "max_gap": 25, "camera": true, "fill_gap": 10, "fill_min_p": 0.8, "fill_mode": "kalman"}`, B0 tau 0.4.
