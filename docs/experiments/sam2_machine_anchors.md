# SAM2 from machine anchors: can it push past the offline detector's recall ceiling?

Branch `exp/sam2-bidir-ball`. Builds on `docs/experiments/sam2_bidir_ball.md` (SAM2.1 with *human* anchors) and the offline solver harness (`src/footy_track/ball_offline/`, cherry-picked from `exp/offline-ball-tracking` fc05732; numbers below are directly comparable with `docs/experiments/offline_ball_baseline.md`). Code: `experiments/sam2_bidir_ball/machine_*.py`. Raw per-frame tracks: `docs/experiments/sam2_machine_anchors_raw/`; aggregates: `docs/experiments/sam2_machine_anchors.json`.

## Question

The detector + Viterbi solver (B1) is bounded by the detector: on the 9 clean test clips (170 GT ball frames) B1 hits 129 (recall@10 0.759, precision 0.985) and the ceiling "any candidate >= 0.1 within 10 px" is 141 (0.829). Human anchors made SAM2 track a ball almost perfectly. Can **machine** anchors (confident B1 nodes) let SAM2 (S1) fill B1's gaps, or (S2) overwrite B1 between anchors, without inventing balls where there is none?

## Setup

* **Reproduction**: B0 (top-1 per frame, tau 0.4) and B1 (camera on, dev-tuned parameters from `offline_ball_baseline.json`) are re-run through the harness and reproduce the published numbers exactly: test 170 GT ball frames and 2 explicit `no_ball` frames, B0 0.741 / 0.947 (126 TP, 7 FP), B1 0.759 / 0.985 (129 TP, 2 FP), ceiling 141/170. Clips, GT, detector-contamination exclusions and `metrics.score_clip` are the harness's.
* **Anchors**: B1 solver *nodes* (accepted detector candidates, not interpolated points) with conf >= thr, thinned to >= `spacing` frames apart; a node next to a B1 gap is always kept. Prompt = the detector's own box at the anchor. Primary config (pre-declared): thr 0.7 (calibrated precision >= 0.96), spacing 10. Sweep: thr 0.6 / 0.7 / 0.8 at spacing 10; spacing 5 / 10 / 25 at thr 0.7.
* **Propagation**: SAM2.1 small, native-resolution 512 crop re-centred on the last prediction (the best variant of the previous experiment), fp16, `offload_video_to_cpu`. Per anchor an independent forward track and backward track (`reverse=True`), each to the neighbouring anchor or at most 50 frames. Every frame of the two full-propagation test clips (seg010, seg031) is covered; for the other clips only the tracks needed to score GT / no_ball / not_broadcast frames were run.
* **Gate** (as specified): a SAM2 position is accepted where the forward pass from the previous anchor and the backward pass from the next anchor are both present and within 5 px (position = their mean), or where exactly one pass exists (the other anchor is out of reach), it reports the ball present, and the frame is within 12 frames of its anchor. A pass that exists but says "absent", or two passes that disagree, reject the frame.
* **S1 gap fill**: B1 frames unchanged; frames with no B1 point receive a gate-passing SAM2 position. **S2 re-anchoring**: anchors unchanged; on every other frame a gate-passing SAM2 position replaces the B1 point (or adds one where B1 has none).
* **Scoring**: harness `score_clip` on the harness's scored frames (clean GT ball frames + explicit no_ball frames); an emitted point on a no_ball frame is a FP. Coverage = share of all frames of the two full-propagation clips (seg010 + seg031; arsenal_example for the independent check) that carry a point; it is not comparable to the harness's all-clip coverage 0.854.
* **Statistics**: paired. Exact McNemar (two-sided binomial on frames B1 misses and S gets, vs frames B1 gets and S loses) and a paired bootstrap over frames (5000 resamples). A block bootstrap over contiguous GT stretches is also in the JSON but is degenerate here (11 blocks on the test set, 2 clips hold 163 of 170 GT frames), so the iid CI is reported, and the number of independent gap regions behind the gain (5 to 6) is stated.
* **Independent clip**: `arsenal_example` has no `machine_labels` and was never in detector training. Detector candidates were regenerated with the same weights (`rtdetr-l_1920_v11`, sha256_16 `b4ea4b22a508c3ab`, imgsz 1280, conf 0.1, half) by `detect_clip.py`; 176 GT ball frames, no contamination exclusion needed, no no_ball frames. Its detector ceiling is only 111/176 = 0.631.
* **Where it ran**: the desktop RTX 2070. Prefect deployments `sam2-detect` and `sam2-machine` (`~/code/jobs/sam2_ball.py`) were deployed and queued, but the gpu pool stayed blocked for over an hour by another session's flow whose rsync push to the mini hung with an idle GPU (not cancelled or touched). The runs were therefore started directly on the desktop in tmux with the same scripts, guarded to start only with >= 2.5 GB VRAM free (another session was training: 4.8 GB used, 3.3 GB free).

## Results

Headline (small / crop512, primary config thr 0.7 spacing 10; full tables further down).

| set | method | GT frames | recall@10 | precision@10 | coverage | fixed / broke vs B1 | notes |
|---|---|---|---|---|---|---|---|
| test (9 clean clips) | B0 | 170 | 0.741 | 0.947 (126/133) | 0.823 | | |
| | B1 | 170 | 0.759 | 0.985 (129/131) | 0.831 | | ceiling 0.829 |
| | **S1 gap fill** | 170 | **0.806** (137) | 0.979 (137/140) | 0.880 | 8 / 0 | +4.7 pts; iid CI [+1.8, +8.1] pts; McNemar p = 0.008; 5 inside, 3 beyond the detector ceiling |
| | S2 re-anchor | 170 | 0.812 (138) | 0.986 (138/140) | 0.880 | 9 / 0 | median error 1.50 -> 1.00 px |
| arsenal_example (independent) | B0 | 176 | 0.540 | 0.856 (95/111) | 0.684 | | ceiling 0.631 |
| | B1 | 176 | 0.540 | 0.922 (95/103) | 0.652 | | |
| | S1 thr 0.7 | 176 | 0.625 (110) | **0.873** (110/126) | 0.768 | 15 / 0 | 8 false fills from ONE wrong anchor |
| | S1 thr 0.8 | 176 | 0.602 (106) | 0.930 (106/114) | 0.696 | 11 / 0 | 0 false fills; iid CI [+2.8, +10.2] pts |
| | S1 agree-only (thr 0.7) | 176 | 0.545 (96) | 0.923 | | 1 / 0 | almost nothing to fill by agreement |

Findings.

1. **S1 is a pure gain on the test set but a small one.** 8 of B1's 41 missed GT frames are recovered (the 33 others stay missed), none of B1's 129 hits is lost (0 broke), precision moves 0.985 -> 0.979 (one extra FP: an 11.3 px near-miss, not a wrong object). The 8 frames come from 5 to 6 distinct gaps (seg010 frames 352-353, 444, 553; seg031 frames 39-49, 91), so the sample behind the significance is small. Per clip: seg010 B1 0.845 -> S1 0.873, seg031 0.566 -> 0.660.
2. **It does cross the ceiling, a little.** 3 of the 8 recovered frames (and 6 of 15 on arsenal_example) had no detector candidate >= 0.1 within 10 px, i.e. B1/B2/any re-ranking could never have produced them; the other 5 (9) had a candidate that B1 did not accept. SAM2 gets there by extending a confident track a few frames in each direction; of the 29 undetectable test frames it recovers 3.
3. **Where the gate matters (ablation, tables below).** Agreement-only (no single-pass fills) gets 132 TP on test (0.776) with 1 near-miss; allowing single passes within 4 / 8 / 12 / 25 frames gives 134 / 135 / 137 / 141 TP (0.829 at 25 frames = exactly the ceiling count), still 1 near-miss. The agreement tolerance (3 / 5 / 8 px) is irrelevant. On arsenal_example the same widening costs precision: single-pass fills within 4 / 8 / 12 / 25 frames add 7 / 10 / 14 / 21 TP but 2 / 5 / 8 / 10 false fills.
4. **The false-fill risk is the anchor, not SAM2.** On arsenal_example B1 has one node with conf 0.728 (frame 154) that is a false positive 1262 px from the ball and is therefore an anchor at thr 0.7. SAM2 faithfully tracks the wrong object, the "single pass within 12 frames" rule accepts it for frames 155-166, and these 8 frames are 1000+ px wrong. Raising the anchor threshold to 0.8 removes the bad anchor (0 false fills, 11 fixes, precision 0.930 vs B1 0.922). Among the arsenal_example B1 nodes with conf >= 0.7 that sit on GT frames, 1 was wrong; on the 9 test clips none was (the harness's candidate-level calibration: 113/113 at >= 0.7). Only one pass reached those frames (no next anchor within reach), so the agreement test could not apply and the lone-pass rule accepted them; the agreement gate alone would have blocked them.
5. **S2 adds almost nothing and costs 5x the GPU.** It replaces 690 B1 frames on the test clips; SAM2 and B1 are within 5 px on 680 of them (98.6%) and differ by > 10 px on 5. Of those: one GT frame where B1 was wrong and SAM2 right (seg010 f354, B1 11.8 px off, SAM2 1.3 px), one frame where B1 had an isolated spurious point 833 px away from a consistent track (seg010 f504, SAM2 agrees with its neighbours), and 3 unlabelled frames 11-12 px apart. S2 broke nothing (0 / 9), improves localisation (median error 1.00 vs 1.50 px) and lowers emissions on confirmed non-balls in the Ball Check probe (23 -> 19 of 140), but the recall gain over S1 is one frame. On arsenal_example S2 = S1 (wrong anchors are not corrected because anchors are never replaced).
6. **Anchor threshold and spacing.** thr 0.6 / 0.7 / 0.8 on test: recall 0.812 / 0.806 / 0.782 (S1), fixes 9 / 8 / 4; precision 0.979 / 0.979 / 0.978. Spacing 5 and 10 are identical, 25 loses one frame (0.800). Differences between configs are one or two frames: the sweep cannot rank 0.6 against 0.7 on this data (there is no clean tuning set for SAM2: the dev clips are detector-contaminated). arsenal_example prefers 0.8 (no false fill). On the dev clips (contaminated, anchors optimistic) S1 gives 0.925 -> 0.950 recall, 5 fixes, 0 broke.

### False fills: no-ball frames and out-of-shot frames

How many no-ball frames do we have? Few, and this is the weakest part of the evidence.

| evidence | n | what SAM2 does |
|---|---|---|
| explicit `no_ball` frames, test (scored set) | **2** (seg010 f450, f454; 3 exist, 1 is contaminated) | both inside the reach of both passes (12-19 frames from an anchor); both passes report absent; 0 fills; B1 also emits nothing |
| explicit `no_ball` frames, dev (contaminated clips, optimistic anchors) | **20** | 18 inside the reach of at least one pass (9 of both); **raw pass says "ball present" on 7 of the 18** (scores 0.2-6.6), but never both passes on the same frame (0 of 9); gate passes on 0; final emissions are identical to B1's own (5 points, B1 FPs) |
| `not_broadcast` frames (out of shot), test + dev | 2 + 20 | 0 SAM2 fills (B1 emits on 3 of the dev ones) |
| Ball Check `not_ball` verdicts (clean), candidate level | **140** in 99 clips | 39 are on frames where B1 emits nothing; the gate would fill 15 of them (10 agree, 5 single), **none within 10 px of the confirmed non-ball** (0/15); S1 emission on non-balls 23/140 = B1 (0.164), S2 19/140 (0.136); ball verdicts hit: B1 104/112, S1 106, S2 107 |
| wrong-object follow, arsenal_example | 1 wrong anchor of 114 | 8 false fills (see finding 4) |

Reading: SAM2's own "object present" output is **not** a usable no-ball detector (it fires on about 40% of the covered no-ball frames in at least one pass, in dev clips); what keeps S1 clean is the structure around it: two-sided agreement, the 12-frame limit for lone passes, and anchors that are real balls. Explicit no-ball evidence is 22 frames plus 140 candidate-level verdicts, so absence of observed false fills there is consistent with a low rate (22 frames: upper 95% bound about 14% by the rule of three on 0 of 22) but does not demonstrate one. The one false-fill mechanism actually observed (wrong anchor) is not a no-ball case at all. Fills on unlabelled frames (most of the 48 fills on test at the primary config) are unscored by construction.

### Runtime and memory

SAM2.1 small, crop512, fp16, RTX 2070 8 GB, one object, GPU shared with a training job for most of the run: 0.23 s per propagated frame (0.145 s/frame when the GPU was free); peak 0.73 GB allocated / 0.86 GB reserved. Frames propagated for the experiment: 6815 (test) + 9651 (dev) + 1217 (arsenal_example) + 3571 (verdict probe) = 21k frames, about 5000 s of GPU time in total, across all anchor configs at once. Operational cost of one config on a clip: S1 needs only the tracks that cover B1 gaps, 0.40 propagated frames per clip frame on the two main test clips (384 / 970) and 0.78 on arsenal_example, i.e. about 0.06-0.12 s per clip frame (1.5-3 s of GPU per second of video) at 0.15 s/frame; S2 needs a track from every anchor, 1.9 propagated frames per clip frame (about 0.28 s per clip frame, 7 s per second of video).

## Verdict

* **S1 (gap fill) is worth keeping as an optional post-processing step on B1, with conservative settings**: anchors with conf >= 0.8 (or an anchor sanity check) and the agreement gate, optionally a short single-pass allowance. It broke no B1 hit in any evaluation (test 8 / 0, dev 5 / 0, arsenal_example 11-15 / 0), recovers frames the detector never produced, and its false-fill behaviour on 22 no-ball frames, 22 out-of-shot frames and 140 confirmed non-ball candidates is clean. The gain is modest: +2.4 (thr 0.8) to +4.7 (primary config) points of recall on test, +6 to +9 points on arsenal_example, where the detector is weaker. It does not "push past the ceiling" in any large way: the ceiling (0.829) is reached only if lone passes are allowed 25 frames from their anchor, and that setting is exactly the one that produces false fills when an anchor is wrong.
* **S2 (re-anchoring) is not worth the 5x compute as a recall lever**, but as a consistency check it is useful: the 5 frames where SAM2 (both passes agreeing) disagrees with B1 by > 10 px contained B1's spurious jump and a B1 near-miss, and none of the disagreements went against SAM2 where GT exists. Use it to flag frames for review, not to overwrite.
* **The dominant risk is a wrong anchor**, because SAM2 does what it is told. Pair S1 with a stricter anchor rule (conf >= 0.8, or require the anchor to be linked / motion-consistent) rather than loosening the gate.

## Limits

* One test set dominated by two clips (163 of 170 GT frames) and 5 to 6 independent gap regions behind the S1 gain; the CI and McNemar p treat frames as exchangeable and are optimistic about independence.
* No SAM2-specific tuning set: the primary config was fixed in advance and the sweep is exploratory on the test set. The dev clips are detector-contaminated (anchors too good) and were used for no_ball evidence only.
* arsenal_example candidates were regenerated (not byte-identical to the original `machine_labels` producer, which was never committed), and the clip is the cleanest independent check but is a single 10-second clip.
* GT is sparse: precision is measured only on labelled frames; fills on unlabelled frames (the majority) are unverified. Two of the test clips' GT are partly VitTrack-reviewed.
* SAM2.1 small / crop512 only (the best variant from the human-anchor experiment); no other model, prompt (box only) or crop size.

## Per-clip (primary config)

| clip | GT ball frames | B0 R / P | B1 R / P | S1 R / P | S2 R / P |
|---|---|---|---|---|---|
| arsenal_mancity_seg010 | 110 | 0.818 / 0.957 (90/94) | 0.845 / 0.989 (93/94) | 0.873 / 0.980 (96/98) | 0.882 / 0.990 (97/98) |
| arsenal_mancity_seg031 | 53 | 0.566 / 0.909 (30/33) | 0.566 / 0.968 (30/31) | 0.660 / 0.972 (35/36) | 0.660 / 0.972 (35/36) |

arsenal_example, anchors conf>=0.7 then conf>=0.8:

| clip | GT ball frames | B0 R / P | B1 R / P | S1 R / P | S2 R / P |
|---|---|---|---|---|---|
| arsenal_example | 176 | 0.540 / 0.856 (95/111) | 0.540 / 0.922 (95/103) | 0.625 / 0.873 (110/126) | 0.625 / 0.873 (110/126) |
| arsenal_example | 176 | 0.540 / 0.856 (95/111) | 0.540 / 0.922 (95/103) | 0.602 / 0.930 (106/114) | 0.602 / 0.930 (106/114) |

Single-frame test clips (7 clips, 1 GT frame each) are in the pooled numbers: B1 6/7, S1 and S2 unchanged there.

## Tables

### Test set (9 clean Man City clips, harness `test`)

| method | GT ball frames | recall@10 [95% Wilson] | precision@10 (TP/emitted) | F1@10 | recall@5 / @20 | coverage (full-propagation clips) | median err px | fixed / broke vs B1 (inside / beyond detector ceiling) | McNemar p | dR vs B1, iid 95% CI | gate fills (agree / single) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| B0 per-frame (tau 0.4) | 170 | 0.741 [0.67-0.80] | 0.947 (126/133) | 0.832 | 0.729 / 0.747 | 0.823 | 1.515 | - | - | - | - |
| B1 Viterbi (camera on) | 170 | 0.759 [0.69-0.82] | 0.985 (129/131) | 0.857 | 0.747 / 0.765 | 0.831 | 1.496 | - | - | - | - |
| S1 anchors conf>=0.6, spacing>=10 | 170 | 0.812 [0.75-0.86] | 0.979 (138/141) | 0.887 | 0.800 / 0.824 | 0.893 | 1.448 | 9 / 0 (4 / 5) | 0.0039 | +0.053 [+0.023, +0.089] | 43 / 17 |
| S2 anchors conf>=0.6, spacing>=10 | 170 | 0.818 [0.75-0.87] | 0.986 (139/141) | 0.894 | 0.818 / 0.824 | 0.893 | 1.002 | 10 / 0 (5 / 5) | 0.0020 | +0.059 [+0.024, +0.095] | 719 / 24 |
| S1 anchors conf>=0.7, spacing>=10 | 170 | 0.806 [0.74-0.86] | 0.979 (137/140) | 0.884 | 0.794 / 0.818 | 0.880 | 1.450 | 8 / 0 (5 / 3) | 0.0078 | +0.047 [+0.018, +0.081] | 33 / 15 |
| S2 anchors conf>=0.7, spacing>=10 | 170 | 0.812 [0.75-0.86] | 0.986 (138/140) | 0.890 | 0.806 / 0.818 | 0.880 | 1.004 | 9 / 0 (6 / 3) | 0.0039 | +0.053 [+0.023, +0.088] | 709 / 29 |
| S1 anchors conf>=0.8, spacing>=10 | 170 | 0.782 [0.71-0.84] | 0.978 (133/136) | 0.869 | 0.771 / 0.794 | 0.864 | 1.474 | 4 / 0 (1 / 3) | 0.1250 | +0.024 [+0.006, +0.047] | 24 / 8 |
| S2 anchors conf>=0.8, spacing>=10 | 170 | 0.788 [0.72-0.84] | 0.985 (134/136) | 0.876 | 0.782 / 0.794 | 0.864 | 1.095 | 5 / 0 (2 / 3) | 0.0625 | +0.029 [+0.006, +0.058] | 690 / 32 |
| S1 anchors conf>=0.7, spacing>=5 | 170 | 0.806 [0.74-0.86] | 0.979 (137/140) | 0.884 | 0.794 / 0.818 | 0.880 | 1.450 | 8 / 0 (5 / 3) | 0.0078 | +0.047 [+0.018, +0.081] | 33 / 15 |
| S2 anchors conf>=0.7, spacing>=5 | 170 | 0.812 [0.75-0.86] | 0.986 (138/140) | 0.890 | 0.812 / 0.818 | 0.880 | 1.080 | 9 / 0 (6 / 3) | 0.0039 | +0.053 [+0.023, +0.088] | 645 / 24 |
| S1 anchors conf>=0.7, spacing>=25 | 170 | 0.800 [0.73-0.85] | 0.978 (136/139) | 0.880 | 0.788 / 0.812 | 0.876 | 1.451 | 7 / 0 (4 / 3) | 0.0156 | +0.041 [+0.012, +0.071] | 33 / 11 |
| S2 anchors conf>=0.7, spacing>=25 | 170 | 0.806 [0.74-0.86] | 0.986 (137/139) | 0.887 | 0.794 / 0.812 | 0.876 | 1.039 | 8 / 0 (5 / 3) | 0.0078 | +0.047 [+0.018, +0.082] | 710 / 36 |

GT ball frames scored: 170; explicit no_ball frames scored: 2. Anchors in each config: thr0.6_sp10: 841, thr0.7_sp10: 747, thr0.8_sp10: 508, thr0.7_sp5: 1363, thr0.7_sp25: 355. SAM2 runtime for these tracks: {'gpu_seconds': 1562.6, 'frames_propagated': 6815, 'sec_per_frame': 0.2293}.

### Independent check: arsenal_example (not in detector training; detections regenerated)

| method | GT ball frames | recall@10 [95% Wilson] | precision@10 (TP/emitted) | F1@10 | recall@5 / @20 | coverage (full-propagation clips) | median err px | fixed / broke vs B1 (inside / beyond detector ceiling) | McNemar p | dR vs B1, iid 95% CI | gate fills (agree / single) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| B0 per-frame (tau 0.4) | 176 | 0.540 [0.47-0.61] | 0.856 (95/111) | 0.662 | 0.540 / 0.540 | 0.684 | 1.303 | - | - | - | - |
| B1 Viterbi (camera on) | 176 | 0.540 [0.47-0.61] | 0.922 (95/103) | 0.681 | 0.534 / 0.540 | 0.652 | 1.174 | - | - | - | - |
| S1 anchors conf>=0.6, spacing>=10 | 176 | 0.631 [0.56-0.70] | 0.874 (111/127) | 0.733 | 0.625 / 0.631 | 0.772 | 1.310 | 16 / 0 (9 / 7) | <0.0001 | +0.091 [+0.051, +0.136] | 5 / 25 |
| S2 anchors conf>=0.6, spacing>=10 | 176 | 0.631 [0.56-0.70] | 0.874 (111/127) | 0.733 | 0.625 / 0.631 | 0.772 | 1.290 | 16 / 0 (9 / 7) | <0.0001 | +0.091 [+0.051, +0.136] | 130 / 29 |
| S1 anchors conf>=0.7, spacing>=10 | 176 | 0.625 [0.55-0.69] | 0.873 (110/126) | 0.728 | 0.619 / 0.625 | 0.768 | 1.307 | 15 / 0 (9 / 6) | 0.0001 | +0.085 [+0.045, +0.131] | 4 / 25 |
| S2 anchors conf>=0.7, spacing>=10 | 176 | 0.625 [0.55-0.69] | 0.873 (110/126) | 0.728 | 0.619 / 0.625 | 0.768 | 1.266 | 15 / 0 (9 / 6) | 0.0001 | +0.085 [+0.045, +0.131] | 123 / 31 |
| S1 anchors conf>=0.8, spacing>=10 | 176 | 0.602 [0.53-0.67] | 0.930 (106/114) | 0.731 | 0.597 / 0.602 | 0.696 | 1.174 | 11 / 0 (6 / 5) | 0.0010 | +0.062 [+0.028, +0.102] | 0 / 11 |
| S2 anchors conf>=0.8, spacing>=10 | 176 | 0.602 [0.53-0.67] | 0.930 (106/114) | 0.731 | 0.597 / 0.602 | 0.696 | 1.311 | 11 / 0 (6 / 5) | 0.0010 | +0.062 [+0.028, +0.097] | 18 / 14 |
| S1 anchors conf>=0.7, spacing>=5 | 176 | 0.625 [0.55-0.69] | 0.873 (110/126) | 0.728 | 0.619 / 0.625 | 0.768 | 1.307 | 15 / 0 (9 / 6) | 0.0001 | +0.085 [+0.045, +0.131] | 4 / 25 |
| S2 anchors conf>=0.7, spacing>=5 | 176 | 0.625 [0.55-0.69] | 0.873 (110/126) | 0.728 | 0.619 / 0.625 | 0.768 | 1.270 | 15 / 0 (9 / 6) | 0.0001 | +0.085 [+0.045, +0.131] | 118 / 31 |
| S1 anchors conf>=0.7, spacing>=25 | 176 | 0.625 [0.55-0.69] | 0.873 (110/126) | 0.728 | 0.619 / 0.625 | 0.764 | 1.307 | 15 / 0 (9 / 6) | 0.0001 | +0.085 [+0.045, +0.131] | 3 / 25 |
| S2 anchors conf>=0.7, spacing>=25 | 176 | 0.625 [0.55-0.69] | 0.873 (110/126) | 0.728 | 0.619 / 0.625 | 0.764 | 1.303 | 15 / 0 (9 / 6) | 0.0001 | +0.085 [+0.045, +0.131] | 117 / 31 |

GT ball frames scored: 176; explicit no_ball frames scored: 0. Anchors in each config: thr0.6_sp10: 21, thr0.7_sp10: 17, thr0.8_sp10: 3, thr0.7_sp5: 29, thr0.7_sp25: 8. SAM2 runtime for these tracks: {'gpu_seconds': 297.4, 'frames_propagated': 1217, 'sec_per_frame': 0.2444}.

### No-ball and out-of-shot frames, role=test (primary config conf>=0.7, spacing>=10)

explicit no_ball frames scored: 2; B1 emits a point on 0; frames inside the reach of at least one SAM2 pass: 2, of both passes: 2; at least one raw pass says 'ball present': 0; both passes say present: 0; gate passes (a SAM2 fill would be emitted): 0; final emissions S1 / S2: 0 / 0 (all of them B1's own points).

not_broadcast frames: n=2, B1 emits 0, S1 0, S2 0; out_of_play frames: n=0.

### No-ball and out-of-shot frames, role=dev (primary config conf>=0.7, spacing>=10)

explicit no_ball frames scored: 20; B1 emits a point on 5; frames inside the reach of at least one SAM2 pass: 18, of both passes: 9; at least one raw pass says 'ball present': 7; both passes say present: 0; gate passes (a SAM2 fill would be emitted): 0; final emissions S1 / S2: 5 / 5 (all of them B1's own points).

not_broadcast frames: n=20, B1 emits 3, S1 3, S2 3; out_of_play frames: n=0.

### No-ball and out-of-shot frames, role=indep (primary config conf>=0.7, spacing>=10)

explicit no_ball frames scored: 0; B1 emits a point on 0; frames inside the reach of at least one SAM2 pass: 0, of both passes: 0; at least one raw pass says 'ball present': 0; both passes say present: 0; gate passes (a SAM2 fill would be emitted): 0; final emissions S1 / S2: 0 / 0 (all of them B1's own points).

not_broadcast frames: n=0, B1 emits 0, S1 0, S2 0; out_of_play frames: n=0.

### Gate ablation (test, anchors conf>=0.7, spacing>=10; S1 gap fill)

B1: recall@10 0.759, precision@10 0.985 (TP 129, FP 2)

| agree px | single-pass max (frames) | R@10 | P@10 | TP | FP | fills in gaps | fills wrong on GT frames |
|---|---|---|---|---|---|---|---|
| 5 | 0 | 0.776 | 0.978 | 132 | 3 | 33 | 1 |
| 5 | 4 | 0.788 | 0.978 | 134 | 3 | 36 | 1 |
| 5 | 8 | 0.794 | 0.978 | 135 | 3 | 42 | 1 |
| 5 | 12 | 0.806 | 0.979 | 137 | 3 | 48 | 1 |
| 5 | 25 | 0.829 | 0.979 | 141 | 3 | 61 | 1 |

### Gate ablation (test, anchors conf>=0.8, spacing>=10; S1 gap fill)

B1: recall@10 0.759, precision@10 0.985 (TP 129, FP 2)

| agree px | single-pass max (frames) | R@10 | P@10 | TP | FP | fills in gaps | fills wrong on GT frames |
|---|---|---|---|---|---|---|---|
| 5 | 0 | 0.765 | 0.977 | 130 | 3 | 24 | 1 |
| 5 | 4 | 0.765 | 0.977 | 130 | 3 | 24 | 1 |
| 5 | 8 | 0.776 | 0.978 | 132 | 3 | 28 | 1 |
| 5 | 12 | 0.782 | 0.978 | 133 | 3 | 32 | 1 |
| 5 | 25 | 0.788 | 0.978 | 134 | 3 | 34 | 1 |

### Gate ablation (indep, anchors conf>=0.7, spacing>=10; S1 gap fill)

B1: recall@10 0.540, precision@10 0.922 (TP 95, FP 8)

| agree px | single-pass max (frames) | R@10 | P@10 | TP | FP | fills in gaps | fills wrong on GT frames |
|---|---|---|---|---|---|---|---|
| 5 | 0 | 0.545 | 0.923 | 96 | 8 | 4 | 0 |
| 5 | 4 | 0.585 | 0.912 | 103 | 10 | 14 | 2 |
| 5 | 8 | 0.602 | 0.891 | 106 | 13 | 21 | 5 |
| 5 | 12 | 0.625 | 0.873 | 110 | 16 | 29 | 8 |
| 5 | 25 | 0.665 | 0.867 | 117 | 18 | 41 | 10 |

### Gate ablation (indep, anchors conf>=0.8, spacing>=10; S1 gap fill)

B1: recall@10 0.540, precision@10 0.922 (TP 95, FP 8)

| agree px | single-pass max (frames) | R@10 | P@10 | TP | FP | fills in gaps | fills wrong on GT frames |
|---|---|---|---|---|---|---|---|
| 5 | 0 | 0.540 | 0.922 | 95 | 8 | 0 | 0 |
| 5 | 4 | 0.562 | 0.925 | 99 | 8 | 4 | 0 |
| 5 | 8 | 0.585 | 0.928 | 103 | 8 | 8 | 0 |
| 5 | 12 | 0.602 | 0.930 | 106 | 8 | 11 | 0 |
| 5 | 25 | 0.665 | 0.929 | 117 | 9 | 23 | 1 |

### Ball Check verdict probe (candidate-level, primary config)

| method | ball verdicts hit (n) | recall | not_ball verdicts hit by an emitted point (n) | emission rate on not_ball |
|---|---|---|---|---|
| B1 | 104/112 | 0.929 | 23/140 | 0.164 |
| S1 | 106/112 | 0.946 | 23/140 | 0.164 |
| S2 | 107/112 | 0.955 | 19/140 | 0.136 |


## Reproduce

```bash
# CPU (mini, one process, thread-capped): harness B0/B1, anchors, SAM2 track spec
OMP_NUM_THREADS=1 python experiments/sam2_bidir_ball/machine_prep.py --out <dir>
python experiments/sam2_bidir_ball/machine_ceiling.py --out <dir> --det <arsenal_example detections>
# GPU (desktop): arsenal_example candidates, then tracks (Prefect: sam2-detect, sam2-machine; or directly)
python detect_clip.py arsenal_example.mp4 --weights best.pt --out <dir> --name arsenal_example
python experiments/sam2_bidir_ball/machine_prep_example.py --det <dir> --out <dir>
python machine_tracks.py --spec spec.json --videos <dir> --out <dir> --ckpt <dir> --model small --variant crop512
# CPU: scoring
python experiments/sam2_bidir_ball/machine_score.py --prep prep.json --tracks <dir> --out <dir> --ceil ceil.json --roles test
python experiments/sam2_bidir_ball/machine_ablate.py --prep prep.json --tracks <dir> --roles test --thr 0.7
python experiments/sam2_bidir_ball/machine_prep_verdicts.py --out <dir>; python machine_score_verdicts.py ...
python experiments/sam2_bidir_ball/machine_report.py --dir <dir>
```
