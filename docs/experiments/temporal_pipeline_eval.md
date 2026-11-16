# Temporal candidates through the trajectory solver: does 5-channel input help once a solver ranks?

Branch `exp/temporal-detector`. Code: `experiments/temporal_detector/{clip_infer,pipeline_eval}.py`, Prefect flow
`temporal-clip-infer/temporal-clip-infer` (`~/code/jobs/footy_temporal_clips.py`). Harness: `exp/offline-ball-tracking` @ fc05732
(`src/footy_track/ball_offline/`, imported read-only; its numbers are reproduced exactly below). Follows
`temporal_detector_ablation.md` (v1: YOLO11s, 100 epochs, 2 seeds, arm A 3ch vs arm B 5ch).

## Question and answer

Does the 5-channel detector's candidate set help once a trajectory solver (the harness's B1: offline Viterbi + short gap fill,
camera on) does the ranking?

**Yes, modestly, and the gain survives the solver; but the YOLO11s detectors stay below the production RT-DETR-L unless their
confidence is recalibrated, and then only match it on the test clips.**

- **9 harness test clips (170 clean GT balls: 110 seg010, 53 seg031, 7 single frames):** B1(5ch) - B1(3ch) recall@10 = **+0.047 [+0.012, +0.086]**
  with the harness calibration knots applied to the YOLO confidence ("raw"), **+0.029 [+0.003, +0.059]** with per-model isotonic maps
  fitted on v11 train ("iso"); per seed +0.059 / +0.035 (raw), +0.041 / +0.018 (iso). The candidate ceiling is almost unchanged
  (+0.009 [-0.020, +0.038]): on these clips the solver gets more out of the 5ch candidates, it does not get more candidates.
  Precision is unchanged with raw knots (-0.003 [-0.018, +0.010]) and lower with iso (-0.046 [-0.078, -0.018]): the iso map is fitted on
  train images the detector has memorised, so it trusts almost every candidate (equal-precision `min_conf` falls to 0.10) and lets
  the solver emit more wrong points.
- **`arsenal_example` (250 frames, a clip with no train frames, 176 labelled frames):** B1(5ch) - B1(3ch) recall@10 = **+0.071 [+0.034, +0.111]**
  (raw; A is nearly blind: 0.139) and **+0.253 [+0.202, +0.304]** (iso; A 0.196 -> B 0.449), ceiling **+0.216 [+0.168, +0.267]**
  (0.213 -> 0.429). Per seed +0.062 / +0.080 (raw), +0.261 / +0.244 (iso). Here the gain is in the candidate set itself (the same
  clip where v1 saw +0.21 in top-1 recall) and the solver keeps it. Precision is 0.88-1.0 for both.
- **Against production RT-DETR-L (harness machine labels):** B1(5ch) - B1(RT-DETR) = -0.059 [-0.092, -0.029] on the test clips with raw
  knots (YOLO confidences sit well below RT-DETR's, so the knots mark most YOLO candidates as unreliable and the solver's
  `min_conf` 0.3 discards them), **+0.024 [-0.003, +0.053] with iso (0.782 vs 0.759) at -0.038 precision**, and on `arsenal_example`
  -0.330 raw / -0.091 [-0.145, -0.040] iso (RT-DETR 0.540, B iso 0.449). The ceiling (any candidate >= 0.1 within 10 px) stays lower for
  the YOLO arms (test 0.756 vs 0.829; arsenal_example 0.429 vs 0.631): a 5ch YOLO11s still misses balls that RT-DETR-L proposes.
  So the temporal input narrows but does not close the gap to the production detector; the RT-DETR-L is the stronger base.
- **Exclusions.** Test numbers use the harness exclusions exactly (frames within 5 of a v11 train/valid image dropped; 170 GT balls +
  2 `no_ball` frames). `arsenal_example` is *not* a harness clip (no machine labels): applying the harness rule, 167 of its 176 labelled
  frames lie within 5 frames of a v11 *valid* image, leaving 9 clean frames (a "dev" clip). The harness-clean table has n = 9 and
  says nothing; the "all labelled frames" table is a **deviation**, defensible because the YOLO arms were neither trained on this clip
  (0 train frames) nor selected on val (`last.pt`); RT-DETR, selected on valid, is if anything flattered there.
- Caveats: frames in a clip are strongly correlated and the bootstrap resamples frames (the test set is 65% one clip, seg010), so the CIs
  are optimistic; one model size, two seeds; isotonic maps from train are over-trusting (see above); the solver parameters were tuned on
  RT-DETR confidences and are used unchanged (B0 tau 0.4, B1 camera-on set), only `min_conf` and the B0 threshold are moved to the
  equal-precision confidence in the iso variant.

## Pipeline and checks

- Decoding: ffmpeg CLI `-fps_mode passthrough` (no cv2 decoding), one sequential pass per clip, 5-frame window. All 10 clips are
  25 fps, 1920x1080; frame counts equal the harness' (`ffprobe` packets = machine-label manifest frames). **Frame indices verified
  three ways:** (1) for 37 dataset images of these clips the ffmpeg frame t equals the dataset TIFF BGR(t) exactly (mean abs diff 0.0; t+-1
  gives 3-20), so the indices equal the dataset/GT index space; (2) RT-DETR rerun by this pipeline equals the harness' machine labels:
  the top-1 candidate of every frame with conf >= 0.3 has the same position (within 3 px) and confidence (within 0.05) in 549/549,
  609/610, 311/312, 2293/2294, 212/212, 2126/2132, 426/426, 680/680 and 347/348 frames (seg010 ... seg140); (3) the harness' B1 / B0 on its
  own machine labels are reproduced to the count (129/170 TP, precision 0.985 and 126/170, 0.947).
- 5ch input built on the fly like the dataset builder: LK + RANSAC alignment of t-2 and t+2 into t (`calibration/temporal`, person
  boxes masked: RT-DETR boxes conf >= 0.3 of the same frame), cut (consecutive-frame histogram distance > 0.2), clip edge and alignment
  failure -> gray(t). Neighbour slots over the 9 test clips + arsenal_example: at most 8 cut/edge slots per clip, 44 alignment
  failures (seg115: 42).
- Candidates: ball class, conf >= 0.1, 1280, boxes in the 1920x1080 frame; camera shift / shot cuts for the solver come from the harness' own
  estimate (player boxes of the RT-DETR rows; for `arsenal_example` from the rerun's player boxes), identical for every detector.
- Isotonic maps: PAV on candidate-level correctness (within 10 px of a labelled ball) of the four YOLO runs on the 1,034 v11 train images
  (not val/test/heldout), one map per run.
- GPU: 8,851 frames (10 clips) at about 0.2-0.3 s/frame with RT-DETR + 4 YOLO models + 2 LK alignments per frame, plus 4 x 1,034
  train images; one flow run (about 40 min); the flow refuses to start with less than 2 GB of free GPU memory.

## test (9 clips, harness exclusions)

172 scored frames; n_gt = 170. recall@10 = emitted point within 10 px of GT (denominator: GT ball frames); precision = TP / (TP + FP) over scored frames (FP = emitted point > 10 px away, or emitted on a `no_ball` frame); ceiling = any candidate with conf >= 0.1 within 10 px (candidate-set recall, identical for B0 and B1).

| detector | calibration | B0 recall / prec | B1 recall / prec | ceiling |
|---|---|---|---|---|
| RT-DETR-L production (frame t) | raw | 0.741 / 0.947 | 0.759 / 0.985 | 0.829 |
| RT-DETR-L rerun by this pipeline | raw | 0.741 / 0.947 | 0.759 / 0.985 | 0.829 |
| A 3ch A_s0 | raw | 0.647 / 1.000 | 0.647 / 0.965 | 0.741 |
| A 3ch A_s0 | iso | 0.735 / 0.962 | 0.735 / 0.992 | 0.741 |
| A 3ch A_s1 | raw | 0.653 / 0.991 | 0.659 / 1.000 | 0.753 |
| A 3ch A_s1 | iso | 0.712 / 0.992 | 0.771 / 0.992 | 0.753 |
| A 3ch mean of seeds | raw | 0.650 / 0.996 | 0.653 / 0.982 | 0.747 |
| A 3ch mean of seeds | iso | 0.724 / 0.977 | 0.753 / 0.992 | 0.747 |
| B 5ch B_s0 | raw | 0.682 / 0.991 | 0.706 / 0.960 | 0.765 |
| B 5ch B_s0 | iso | 0.759 / 0.935 | 0.776 / 0.936 | 0.765 |
| B 5ch B_s1 | raw | 0.700 / 1.000 | 0.694 / 1.000 | 0.747 |
| B 5ch B_s1 | iso | 0.747 / 1.000 | 0.788 / 0.957 | 0.747 |
| B 5ch mean of seeds | raw | 0.691 / 0.996 | 0.700 / 0.980 | 0.756 |
| B 5ch mean of seeds | iso | 0.753 / 0.967 | 0.782 / 0.947 | 0.756 |

Paired bootstrap (10,000 resamples of scored frames; seed-mean per-frame outcomes for the arms), B1 camera on:

| comparison | calibration | recall@10 diff [95% CI] | precision diff [95% CI] | ceiling diff [95% CI] | per-seed recall diff |
|---|---|---|---|---|---|
| B1(5ch) - B1(3ch) | raw | +0.047 [+0.012, +0.086] | -0.003 [-0.018, +0.010] | +0.009 [-0.020, +0.038] | s0: +0.059, s1: +0.035 |
| B1(5ch) - B1(3ch) | iso | +0.029 [+0.003, +0.059] | -0.046 [-0.078, -0.018] | +0.009 [-0.020, +0.038] | s0: +0.041, s1: +0.018 |
| B1(5ch) - B1(RT-DETR) | raw | -0.059 [-0.092, -0.029] | -0.005 [-0.028, +0.018] | -0.074 [-0.115, -0.035] |  |
| B1(5ch) - B1(RT-DETR) | iso | +0.024 [-0.003, +0.053] | -0.038 [-0.070, -0.007] | -0.074 [-0.115, -0.038] |  |

## arsenal_example, harness-clean frames

9 scored frames; n_gt = 9. recall@10 = emitted point within 10 px of GT (denominator: GT ball frames); precision = TP / (TP + FP) over scored frames (FP = emitted point > 10 px away, or emitted on a `no_ball` frame); ceiling = any candidate with conf >= 0.1 within 10 px (candidate-set recall, identical for B0 and B1).

| detector | calibration | B0 recall / prec | B1 recall / prec | ceiling |
|---|---|---|---|---|
| RT-DETR-L production (frame t) | raw | 0.111 / 0.200 | 0.111 / 0.250 | 0.111 |
| RT-DETR-L rerun by this pipeline | raw | 0.111 / 0.200 | 0.111 / 0.250 | 0.111 |
| A 3ch A_s0 | raw | 0.000 / n/a | 0.000 / n/a | 0.111 |
| A 3ch A_s0 | iso | 0.111 / 0.500 | 0.111 / 1.000 | 0.111 |
| A 3ch A_s1 | raw | 0.000 / n/a | 0.000 / n/a | 0.000 |
| A 3ch A_s1 | iso | 0.000 / n/a | 0.000 / n/a | 0.000 |
| A 3ch mean of seeds | raw | 0.000 / n/a | 0.000 / n/a | 0.056 |
| A 3ch mean of seeds | iso | 0.056 / n/a | 0.056 / n/a | 0.056 |
| B 5ch B_s0 | raw | 0.111 / 1.000 | 0.111 / 1.000 | 0.111 |
| B 5ch B_s0 | iso | 0.111 / 1.000 | 0.111 / 1.000 | 0.111 |
| B 5ch B_s1 | raw | 0.111 / 1.000 | 0.111 / 1.000 | 0.111 |
| B 5ch B_s1 | iso | 0.111 / 0.500 | 0.111 / 0.500 | 0.111 |
| B 5ch mean of seeds | raw | 0.111 / 1.000 | 0.111 / 1.000 | 0.111 |
| B 5ch mean of seeds | iso | 0.111 / 0.750 | 0.111 / 0.750 | 0.111 |

Paired bootstrap (10,000 resamples of scored frames; seed-mean per-frame outcomes for the arms), B1 camera on:

| comparison | calibration | recall@10 diff [95% CI] | precision diff [95% CI] | ceiling diff [95% CI] | per-seed recall diff |
|---|---|---|---|---|---|
| B1(5ch) - B1(3ch) | raw | +0.111 [+0.000, +0.333] | +nan [+nan, +nan] | +0.056 [+0.000, +0.167] | s0: +0.111, s1: +0.111 |
| B1(5ch) - B1(3ch) | iso | +0.056 [+0.000, +0.167] | -0.333 [-0.600, +0.000] | +0.056 [+0.000, +0.167] | s0: +0.000, s1: +0.111 |
| B1(5ch) - B1(RT-DETR) | raw | +0.000 [+0.000, +0.000] | +0.750 [+0.000, +0.833] | +0.000 [+0.000, +0.000] |  |
| B1(5ch) - B1(RT-DETR) | iso | +0.000 [+0.000, +0.000] | +0.417 [+0.000, +0.800] | +0.000 [+0.000, +0.000] |  |

## arsenal_example, all labelled frames (NOT harness-clean)

176 scored frames; n_gt = 176. recall@10 = emitted point within 10 px of GT (denominator: GT ball frames); precision = TP / (TP + FP) over scored frames (FP = emitted point > 10 px away, or emitted on a `no_ball` frame); ceiling = any candidate with conf >= 0.1 within 10 px (candidate-set recall, identical for B0 and B1).

| detector | calibration | B0 recall / prec | B1 recall / prec | ceiling |
|---|---|---|---|---|
| RT-DETR-L production (frame t) | raw | 0.540 / 0.856 | 0.540 / 0.913 | 0.631 |
| RT-DETR-L rerun by this pipeline | raw | 0.540 / 0.856 | 0.540 / 0.913 | 0.631 |
| A 3ch A_s0 | raw | 0.142 / 1.000 | 0.153 / 1.000 | 0.301 |
| A 3ch A_s0 | iso | 0.295 / 0.929 | 0.267 / 1.000 | 0.301 |
| A 3ch A_s1 | raw | 0.119 / 1.000 | 0.125 / 1.000 | 0.125 |
| A 3ch A_s1 | iso | 0.125 / 0.815 | 0.125 / 0.759 | 0.125 |
| A 3ch mean of seeds | raw | 0.131 / 1.000 | 0.139 / 1.000 | 0.213 |
| A 3ch mean of seeds | iso | 0.210 / 0.872 | 0.196 / 0.879 | 0.213 |
| B 5ch B_s0 | raw | 0.216 / 1.000 | 0.216 / 1.000 | 0.477 |
| B 5ch B_s0 | iso | 0.472 / 0.965 | 0.528 / 1.000 | 0.477 |
| B 5ch B_s1 | raw | 0.216 / 0.974 | 0.205 / 1.000 | 0.381 |
| B 5ch B_s1 | iso | 0.369 / 0.929 | 0.369 / 0.890 | 0.381 |
| B 5ch mean of seeds | raw | 0.216 / 0.987 | 0.210 / 1.000 | 0.429 |
| B 5ch mean of seeds | iso | 0.420 / 0.947 | 0.449 / 0.945 | 0.429 |

Paired bootstrap (10,000 resamples of scored frames; seed-mean per-frame outcomes for the arms), B1 camera on:

| comparison | calibration | recall@10 diff [95% CI] | precision diff [95% CI] | ceiling diff [95% CI] | per-seed recall diff |
|---|---|---|---|---|---|
| B1(5ch) - B1(3ch) | raw | +0.071 [+0.034, +0.111] | +0.000 [+0.000, +0.000] | +0.216 [+0.168, +0.267] | s0: +0.062, s1: +0.080 |
| B1(5ch) - B1(3ch) | iso | +0.253 [+0.202, +0.304] | +0.044 [-0.024, +0.123] | +0.216 [+0.168, +0.264] | s0: +0.261, s1: +0.244 |
| B1(5ch) - B1(RT-DETR) | raw | -0.330 [-0.398, -0.264] | +0.087 [+0.038, +0.146] | -0.202 [-0.259, -0.148] |  |
| B1(5ch) - B1(RT-DETR) | iso | -0.091 [-0.145, -0.040] | +0.038 [-0.019, +0.099] | -0.202 [-0.259, -0.148] |  |

## Checks

- Harness reproduction on the 9 test clips (RT-DETR machine labels, camera-on B1): recall 0.759 (129 TP of 170), precision 0.985; B0 tau 0.4 recall 0.741, precision 0.947 (harness doc: 0.759 / 0.985 and 0.741 / 0.947).
- RT-DETR rerun by this pipeline vs the harness's machine labels (top-1 per frame with conf >= 0.3, same position within 3 px and conf within 0.05): seg010 549/549; seg011 609/610; seg031 311/312; seg094 2293/2294; seg101 212/212; seg115 2126/2132; seg134 426/426; seg135 680/680; seg140 347/348.
- Neighbour status per clip (frames, [prev, next] slots): seg010 ok:1182/edge:4/cut:4; seg011 ok:1510/edge:4/cut:4/align_fail:2; seg031 ok:742/edge:4/cut:4; seg094 ok:5032/edge:4/cut:4; seg101 ok:432/edge:4/cut:4; seg115 ok:4594/edge:4/align_fail:42; seg134 ok:866/edge:4; seg135 ok:1462/edge:4/cut:4; seg140 ok:702/edge:4/cut:4; arsenal_example ok:496/edge:4.
- Isotonic maps (fitted on v11 train candidates within 10 px of a labelled ball): A_s0: 1009/1102 correct, min_conf 0.10, B0 tau 0.10; A_s1: 1002/1072 correct, min_conf 0.11, B0 tau 0.14; B_s0: 992/1034 correct, min_conf 0.10, B0 tau 0.10; B_s1: 999/1054 correct, min_conf 0.10, B0 tau 0.16.

Solver parameters (harness `camera_on`): `{"min_conf": 0.3, "bias": 1.0, "emit_scale": 1.0, "kappa": 1.5, "lam_pos": 0.5, "lam_vel": 0.0, "vmax": 40.0, "r0": 12.0, "dmax": 2.0, "mu": 0.02, "static_px": 3.0, "max_gap": 25, "camera": true, "fill_gap": 10, "fill_min_p": 0.8, "fill_mode": "kalman"}`, B0 tau 0.4.

## Union test: do the 5ch candidates add to RT-DETR-L?

Per frame the candidates of RT-DETR-L (harness knots -> p) and of a 5ch YOLO arm (isotonic map fitted on v11 *valid* without arsenal_example -> p) are merged: cross-source duplicates within 6 px collapse to one candidate (higher-p position, p = max), optionally with a +0.5 logit bonus when both agree. The merged p is fed to the harness' B0 (tau 0.4) / B1 (camera-on parameters) unchanged, through the inverse of the harness knots (RT-DETR alone is reproduced exactly). Ceiling of a union row = any candidate of either source (conf >= 0.1) within 10 px. 5ch YOLO = arm B of v1, seeds 0 and 1 reported separately.

Val isotonic maps (n small: B_s0: 44 candidates (40 correct) from 53 images, 4 knots, p in [0.50, 1.00]; B_s1: 43 candidates (37 correct) from 53 images, 5 knots, p in [0.00, 1.00]).

### test (9 clips, harness exclusions)

| candidate set | B0 recall / prec | B1 recall / prec | ceiling | B1 emitted on no_ball frames |
|---|---|---|---|---|
| RT-DETR alone | 0.741 / 0.947 | 0.759 / 0.985 | 0.829 | 0 / 2 (B0: 0) |
| 5ch B_s0 alone (val-iso p) | 0.759 / 0.935 | 0.771 / 0.942 | 0.765 | 0 / 2 (B0: 0) |
| union RT + B_s0 | 0.776 / 0.917 | 0.788 / 0.950 | 0.841 | 0 / 2 (B0: 0) |
| union RT + B_s0, agreement bonus | 0.776 / 0.917 | 0.788 / 0.944 | 0.841 | 0 / 2 (B0: 0) |
| 5ch B_s1 alone (val-iso p) | 0.747 / 1.000 | 0.788 / 0.957 | 0.747 | 0 / 2 (B0: 0) |
| union RT + B_s1 | 0.771 / 0.956 | 0.800 / 0.986 | 0.835 | 0 / 2 (B0: 0) |
| union RT + B_s1, agreement bonus | 0.771 / 0.956 | 0.800 / 0.986 | 0.835 | 0 / 2 (B0: 0) |

Paired bootstrap vs RT-DETR alone, B1 (95% CI over frames; over clips in brackets, test only): 

| candidate set | recall@10 diff | precision diff | ceiling diff |
|---|---|---|---|
| 5ch B_s0 alone (val-iso p) | +0.012 [-0.018, 0.047] (-0.027, 0.091) | -0.042 [-0.083, -0.007] (-0.089, 0.000) | -0.065 [-0.111, -0.024] (-0.098, 0.000) |
| union RT + B_s0 | +0.029 [0.006, 0.058] (0.000, 0.091) | -0.034 [-0.068, -0.007] (-0.109, 0.000) | +0.012 [0.000, 0.030] (0.000, 0.036) |
| union RT + B_s0, agreement bonus | +0.029 [0.006, 0.058] (0.000, 0.091) | -0.041 [-0.077, -0.013] (-0.129, 0.000) | +0.012 [0.000, 0.030] (0.000, 0.036) |
| 5ch B_s1 alone (val-iso p) | +0.029 [-0.006, 0.065] (0.000, 0.073) | -0.028 [-0.063, 0.002] (-0.039, 0.003) | -0.082 [-0.129, -0.041] (-0.089, 0.000) |
| union RT + B_s1 | +0.041 [0.012, 0.071] (0.000, 0.071) | +0.001 [0.000, 0.002] (0.000, 0.003) | +0.006 [0.000, 0.018] (0.000, 0.009) |
| union RT + B_s1, agreement bonus | +0.041 [0.012, 0.072] (0.000, 0.073) | +0.001 [0.000, 0.002] (0.000, 0.003) | +0.006 [0.000, 0.018] (0.000, 0.009) |

### arsenal_example, all labelled frames (NOT harness-clean)

| candidate set | B0 recall / prec | B1 recall / prec | ceiling | B1 emitted on no_ball frames |
|---|---|---|---|---|
| RT-DETR alone | 0.540 / 0.856 | 0.540 / 0.913 | 0.631 | 0 / 0 (B0: 0) |
| 5ch B_s0 alone (val-iso p) | 0.472 / 0.965 | 0.511 / 1.000 | 0.477 | 0 / 0 (B0: 0) |
| union RT + B_s0 | 0.557 / 0.852 | 0.602 / 0.922 | 0.636 | 0 / 0 (B0: 0) |
| union RT + B_s0, agreement bonus | 0.557 / 0.852 | 0.602 / 0.922 | 0.636 | 0 / 0 (B0: 0) |
| 5ch B_s1 alone (val-iso p) | 0.369 / 0.915 | 0.381 / 0.798 | 0.381 | 0 / 0 (B0: 0) |
| union RT + B_s1 | 0.557 / 0.838 | 0.562 / 0.884 | 0.642 | 0 / 0 (B0: 0) |
| union RT + B_s1, agreement bonus | 0.557 / 0.838 | 0.562 / 0.892 | 0.642 | 0 / 0 (B0: 0) |

Paired bootstrap vs RT-DETR alone, B1 (95% CI over frames; over clips in brackets, test only): 

| candidate set | recall@10 diff | precision diff | ceiling diff |
|---|---|---|---|
| 5ch B_s0 alone (val-iso p) | -0.028 [-0.085, 0.028] | +0.087 [0.037, 0.145] | -0.153 [-0.210, -0.102] |
| union RT + B_s0 | +0.062 [0.028, 0.102] | +0.008 [-0.017, 0.035] | +0.006 [0.000, 0.017] |
| union RT + B_s0, agreement bonus | +0.062 [0.028, 0.102] | +0.008 [-0.016, 0.035] | +0.006 [0.000, 0.017] |
| 5ch B_s1 alone (val-iso p) | -0.159 [-0.222, -0.097] | -0.116 [-0.212, -0.024] | -0.250 [-0.318, -0.182] |
| union RT + B_s1 | +0.023 [0.006, 0.045] | -0.030 [-0.078, 0.013] | +0.011 [0.000, 0.028] |
| union RT + B_s1, agreement bonus | +0.023 [0.000, 0.051] | -0.022 [-0.073, 0.028] | +0.011 [0.000, 0.028] |

Reading: the 5ch candidates add to, rather than replace, the production detector, but modestly. The ceiling rises only a little (test +0.012 / +0.006 for seeds 0 / 1, arsenal_example +0.006 / +0.011) because most 5ch candidates duplicate RT-DETR's (about 6.5k agreeing candidates against about 1k YOLO-only on the test clips). B1 recall rises +0.029 / +0.041 on the test clips (frame CI excludes 0; the clip-level CI touches 0 because 7 of the 9 clips have a single GT frame and seg010 carries 65% of the frames) and +0.062 / +0.023 on arsenal_example; precision is -0.034 (seed 0) / +0.001 (seed 1) on test, so seed 0 also adds some wrong candidates. The agreement bonus changes almost nothing. On the 2 harness no_ball frames of the test clips no configuration emits a point (n = 2: uninformative). The val isotonic maps rest on 43-44 candidates (4-5 knots): too small to trust, so the union numbers are indicative; they do show that the gain over RT-DETR alone does not depend on a well-calibrated YOLO map.

Merged candidate counts (all frames): test (9 clips, harness exclusions)|B_s0: agree 6575, RT-DETR only 24658, YOLO only 975; test (9 clips, harness exclusions)|B_s1: agree 6561, RT-DETR only 24672, YOLO only 1209; arsenal_example, all labelled frames (NOT harness-clean)|B_s0: agree 127, RT-DETR only 915, YOLO only 41; arsenal_example, all labelled frames (NOT harness-clean)|B_s1: agree 115, RT-DETR only 927, YOLO only 54.
