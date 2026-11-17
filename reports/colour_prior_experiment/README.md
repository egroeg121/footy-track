# Per-match colour prior: does it improve detection? (post-hoc rescoring, no retraining)

Run 2026-11-15 on the Mac mini, thread-capped (2 threads), one process at a time. Nothing under `src/`, `tests/`, `docs/`
or the staged changes was touched; no git state changed. All code and outputs are in this directory.

## Headline

| | Question | Answer (on clean held-out data) |
|---|---|---|
| **A** | Does a per-match colour palette + Bayes-style rescoring improve referee-vs-player detection? | **No.** Baseline referee AP is already **0.936** (P/R/F1 at conf 0.5: 0.929 / 0.920 / 0.924). Naive colour rescoring (beta=1) **lowers** AP to 0.85 (unsupervised palette; frame-bootstrap delta **-0.086, 95% CI [-0.109, -0.064]**) and to 0.81 +/- 0.06 (palette from 5 GT referees + 10 GT players). Letting a logistic regression learn the weight on the early frames gives delta AP **+0.002 [-0.006, +0.009]**: indistinguishable from zero. |
| **B** | Does distance to a per-match ball reference descriptor help true/false ball candidates? | **Marginal at best, not significant.** Leave-one-clip-out (1,829 labelled candidates, 396 true): AUROC 0.912 (conf) -> 0.930 (conf + distance), AP 0.883 -> 0.883; clip-bootstrap delta AP **-0.001 [-0.038, +0.022]**, delta AUROC +0.020 [-0.008, +0.041]. A small false-positive-rejection gain shows up on Ball Check verdicts (n=259) but not in a way I would trust. |

**Recommendation: do not build either as a detection-rescoring stage.** Details and the one place colour is worth using
(team/role *constraints* for tracking, not rescoring) are at the end.

An important side finding for A: the design doc's M8 ("referee median conf 0.335, only 44.5 percent >= 0.5") is **not a recall problem**.
On clean test frames, the detector's referee score for ground-truth referees has median **0.916** and **93.1 percent >= 0.5**
(n=418 matched GT referees). The low median in M8 is the long tail of junk low-confidence referee detections (see A.5).

## Data and what GT actually exists

**Referee / person-role GT** (what exists, honestly):
- `detection_dataset/roboflow_dataset_11` (YOLO labels, 7 classes). Full-match Arsenal-Man City frames (names `arsenal_mancity_20250925_NNNNNN`,
  1 fps index n, i.e. original frame 25n) are the only referee GT that is both *time-spread over the match* and *not in the detector's
  training set*: **test split = 265 frames, 440 referee, 4,322 player boxes**. Train split holds 94 such frames, valid 29.
- `ball_gt_marks_MERGED` sidecars (labeller GT, also referee/player boxes): 39 files: several hundred hand-labelled referee boxes plus ~1.5k tracker-propagated (`vittrack`, human_checked) ones, but **almost all of those frames
  were exported to Roboflow train** (checked per (clip, frame): seg010 13/45, seg030 29/46, seg050 59/59, seg060 22/31, seg090 8/13, seg100 24/25,
  seg110 28/28 ...), so the detector has seen them. Only ~70 referee boxes in sidecar frames are clean, so I did not use them as the primary eval.
  `arsenal_example` (157 valid frames, ~170 referees) has no stored detections and I could not re-run the detector (see Constraints).
- The GT "referee" class **includes assistant referees (linesmen)**. This matters (A.6).
- **Team GT does not exist anywhere** (`identity_labels/identity_reviews.jsonl` is tracklet-review intervals only: no team field; no `team` column in
  the labeller sidecars). So **team-assignment accuracy could not be measured against GT**; I report a visual audit instead (A.4).
- Only Arsenal-Man City has stored detections (`machine_labels/`, 162 segments). Aston Villa / Bournemouth GT frames are in the training set, Norwich has no person GT.
  So **A is a single-match experiment**; no cross-match palette transfer could be tested.

**Ball GT**: `ball_gt_marks_MERGED` human/labeller in_play_ball marks and `ball_gt_marks/ball_checks` verdicts, joined to `ball_candidates/` + `ball_embeddings/`
(only the 162 Man City segment clips have embeddings).

## A. Referee / person-role rescoring

### A.1 Detections used (no re-run of the detector)
Stored `machine_labels/` RT-DETR-L 1920 (`rtdetr-l_1920_v11`) detections, **floor conf 0.10** (not 0.05: that is what was stored), one row per (box, class), so a box
can carry e.g. `player 0.28` and `referee 0.42`; I regroup rows by identical box into a 7-class score vector. The roboflow GT frames were mapped onto segment clips with
`arsenal_mancity/split_video_broadcast_frames/segments.json`: in-segment frame = 25n - start_frame + 12. The +12 offset was verified by thumbnail matching on 5 random test frames
(`map_check.py`, all gave 12); 5 of the 265 test frames fall outside any broadcast segment. Details in `a_map_dets.py`.
Class-agnostic NMS (IoU 0.7) over person-type candidates, identical for baseline and rescored.

I tried to re-run the detector first: via the Prefect `gpu` pool (flow `colour-prior-detect`, see Constraints) and on CPU. The GPU was held by a queued 60-epoch training batch (three runs ahead of mine),
and CPU RT-DETR-L at 1920 took **~77-96 s/frame** on the loaded mini (load average ~20), i.e. hours. Both abandoned in favour of stored detections.

### A.2 Protocol (time-disjoint, no leakage)
- **Palette frames** = full-match frames with index < 1500 s (**35 frames**, 56 GT referees, 559 GT players). **Eval frames** = **test-split frames with index >= 1500 s: 257 frames, 424 GT referees, ~4.2k GT players**
  (the detector never trained on these). The 8 early test frames are used only for palettes, never for eval.
- Colour feature (design doc M6 recipe): torso crop [0.25..0.75]w x [0.20..0.50]h, mean of (cos h, sin h, s, v).
- Palettes: **U** = unsupervised: GMM (k chosen by BIC, 2-4) on early detector-confident player boxes (conf >= 0.8, n=481) + Gaussian on early detector-confident referee boxes (conf >= 0.5, n=53). BIC picked k=4 for the players, which turned out to be 2 blue + 2 red components (A.4).
  **H** = "human clicked": 5 random GT referees + 10 random GT players from the early frames (20 random draws; mean +/- sd reported).
  **UH** = unsupervised player mixture + 5 GT referees (20 draws).
- Rescoring: `logit(s_ref') = logit(s_ref) + beta * log[ L(x|referee) / L(x|player-mixture) ]` (clipped +/-8), `s_player'` moved the opposite way. beta=1 is the Bayes-style soft factor; 0.25-2 shown as sensitivity.
  Referee AP at IoU 0.5 over all person candidates (denominator = all GT referees, including 11 with no candidate box which no rescoring can fix).

### A.3 Results (257 test frames, 424 GT referees)

Referee detection, baseline vs rescored (`a_results_summary.csv`; H/UH are means over 20 exemplar draws):

| palette | beta | AP | P@0.5 | R@0.5 | F1@0.5 | F1@0.3 | best F1 (any thr) |
|---|---|---|---|---|---|---|---|
| baseline (detector only) | - | **0.936** | 0.929 | 0.920 | **0.924** | 0.902 | 0.925 |
| U unsupervised | 0.25 | 0.927 | 0.851 | 0.915 | 0.882 | 0.858 | 0.901 |
| U unsupervised | 1.0 | 0.850 | 0.747 | 0.835 | 0.788 | 0.785 | 0.795 |
| H 5 GT ref + 10 GT players | 0.25 | 0.915 +/- 0.016 | 0.783 | 0.900 | 0.837 | 0.802 | 0.878 |
| H 5 GT ref + 10 GT players | 1.0 | 0.810 +/- 0.063 | 0.665 | 0.799 | 0.723 | 0.725 | 0.732 |
| UH unsup players + 5 GT ref | 0.25 | 0.918 +/- 0.005 | 0.877 | 0.891 | 0.884 | 0.860 | 0.893 |
| UH unsup players + 5 GT ref | 1.0 | 0.807 +/- 0.031 | 0.730 | 0.734 | 0.729 | 0.733 | 0.743 |

Every palette source x every beta is at or below baseline. Frame-bootstrap delta AP (U, beta=1) = -0.086 [-0.109, -0.064].

**Learned weight instead of fixed beta** (`a_stacked.csv`): logistic regression on the early frames only (626 candidates matched to GT player/referee, 54 referees), features logit(s_ref), logit(s_player) [+ llr(U)], applied to the eval frames:

| variant | AP | P@0.5 | R@0.5 | delta AP vs raw detector (95% CI) |
|---|---|---|---|---|
| recalibrated detector, no colour (control) | 0.939 | 0.849 | 0.943 | +0.003 [-0.001, +0.007] |
| detector + colour llr | 0.938 | 0.778 | 0.953 | +0.002 [-0.006, +0.009] |

The colour term adds nothing over simply re-fitting the two detector scores.

**Player <-> referee confusions** (matched GT player/referee boxes, called by argmax of the two role scores, `a_transitions_U.json`):
- Baseline: GT referee called player **6**, GT player called referee **9** (out of 418 referees / 4,275 players).
- U, beta=1: fixed 6 (2 referee->player fixed, 4 player->referee fixed), **introduced 23** (20 referees flipped to player, 3 players flipped to referee), 9 wrong->wrong (unchanged).
  Net: +17 errors.
- U, beta=0.25: 7 / 8 (no net change); H/UH beta=1: 54-60 referees lost.

### A.4 Team assignment
**No team GT, so no accuracy number.** What I can say: the unsupervised mixture gives 2 blue components (35 percent + 17 percent of confident late players) and 2 red ones (32 percent + 16 percent).
A visual audit of 40 random confident test-frame players (`a_palette_audit.png`, rows C0-C3, last row = GT referees): C0 and C3 10/10 sky blue (Man City), C1 10/10 red (Arsenal), C2 8/10 red + 1 goalkeeper
(yellow-green, absorbed into a red component: the 2 GKs do not get their own component) + 1 blue player. About 38 of 40 consistent with team by eye (n=40, so +/- 7 points).
Not validated: 23 percent of confident players have max component posterior < 0.9, but that mostly splits same-team components, not teams.

### A.5 Why it does not help: the detector's referee score is already good, and M8 was misleading
On the 257 eval frames, referee-class detections (conf >= 0.1): n=711, median 0.835, 59 percent >= 0.5 overall, but
GT-matched referees have median **0.916** / **93.1 percent >= 0.5**. By score bin (`a_diagnostics.json`):

| referee score | n | matches a GT referee | matches GT player | no GT box |
|---|---|---|---|---|
| 0.1-0.2 | 221 | 4.5% | 1.4% | 81% |
| 0.2-0.5 | 70 | 18.6% | 4.3% | 54% |
| 0.5-0.8 | 37 | 62% | 2.7% | 22% |
| >= 0.8 | 383 | 95.6% | 1.8% | 1.6% |

So the "weak referee" story in M8 (all referee detections of seg002) is a precision tail of junk at 0.1-0.2, not missed referees. There is little headroom: only 6 referees are called player and 9 players called referee at baseline.
Colour on its own *is* discriminative (AUROC **0.976** referee vs player on matched boxes; median llr +5.3 for referees, -8.0 for players), but the detector is more discriminative, and a likelihood-ratio is overconfident (clipped at +/-8, so it overrules a detector score of 0.9).

### A.6 Failure cases
- The GT referees that beta=1 rescoring flipped to "player" (20): the 12 I inspected visually are all **assistant referees (linesmen)** with a yellow flag at the pitch edge, in front of red advertising boards (`a_lost_referees_montage.png`, `a_flipped_roles_U_beta1.csv`).
  The torso crop is contaminated by hoarding red and the kit/colour is unlike the main referee's black kit used to build the palette, so colour says "not referee". All 20 are at frame index >= 3811 s (second half).
  A per-match palette built from the early main-referee boxes does not describe linesmen.
- Goalkeepers: yellow-green GK falls into a team component; a GK-specific component would need k>=5 and still risks stealing referee mass.
- 11 of 424 GT referees have no candidate box at all (rescoring cannot recover those).

## B. Ball candidate rescoring

### B.1 Data
`b_build_dataset.py` joins `ball_candidates/` (detector ball boxes, conf >= 0.1) + `ball_embeddings/` (72-d descriptor) to labels:
- **GT-mark labels**: candidate within 15 px of a human/labeller `in_play_ball` mark = true, > 40 px from all marks in a ball-complete frame (frame has an in_play_ball mark or a `no_ball` row) = false, in-between dropped.
  **1,829 candidates, 396 true / 1,433 false, 12 clips** (seg010, 020, 030, 031, 050, 060, 080, 090, 100, 110, 140, 150; 11 have positives). Heavily concentrated: seg010 131, seg050 92, seg110 37, seg031 36, seg030 35 positives.
- **Ball Check verdicts**: ball/not_ball, matched to candidates by bbox: 283 rows (120 true / 163 false) over ~100 clips; 259 (112 true) are in clips with no GT-mark labels and serve as an independent test.
- 901 of the 1,829 GT-labelled candidates are on frames that were **in the detector's Roboflow train split** (in-sample: their conf is inflated). I report both "all" and "detector-unseen frames" (929 candidates, 185 true).

Reference = mean descriptor of true balls in the fit set; features: logit(conf), distance to reference (standardised RMS over all 72 dims, or patch/hue parts separately). Models: logistic regression, grouped by clip.
Not tested: training the combiner on *other matches* (no other match has embeddings) and the Norwich held-out match (no embeddings, no ball GT).

### B.2 Results (`b_results.csv`, `b_bootstrap.json`)

| protocol | subset | n (true) | model | AUROC | AP | P@R=0.8 | FPR@TPR=0.8 |
|---|---|---|---|---|---|---|---|
| leave-one-clip-out | all | 1829 (396) | conf only | 0.912 | 0.885 | 0.841 | 0.042 |
| | | | conf + distance | 0.930 | 0.883 | 0.832 | 0.045 |
| | | | distance only | 0.697 | 0.348 | 0.279 | 0.572 |
| leave-one-clip-out | detector-unseen frames | 929 (185) | conf only | 0.912 | 0.890 | 0.931 | 0.015 |
| | | | conf + distance | 0.933 | 0.895 | 0.967 | 0.007 |
| temporal: fit seg < 40, test seg >= 40 | all | 1037 (189) | conf only | 0.932 | 0.891 | 0.874 | 0.026 |
| | | | conf + distance | 0.882 | 0.799 | 0.463 | 0.208 |
| temporal | detector-unseen frames | 238 (31) | conf only | 0.939 | 0.892 | 0.962 | 0.005 |
| | | | conf + distance | 0.966 | 0.895 | 0.962 | 0.005 |
| fit on GT marks, test on Ball Check verdicts (other clips) | all | 259 (112) | conf only | 0.842 | 0.843 | 0.634 | 0.354 |
| | | | conf + distance | 0.857 | 0.851 | 0.750 | 0.204 |

(raw conf, conf-only LR give identical ranking.) Clip-level bootstrap of the leave-one-clip-out pooled result: delta AP **-0.001 [-0.038, +0.022]**, delta AUROC +0.020 [-0.008, +0.041].
`b_extra.log`: a learned model on conf + the full 72-d descriptor (no reference at all) reaches AUROC 0.926 / AP 0.876 (LOCO), 0.946 / 0.910 on unseen frames, so the reference is not the limiting factor either.

- **Is it better at rejecting false positives?** Slightly, in places: on Ball Check verdicts FPR at 80 percent recall drops 0.354 -> 0.204 and precision at 80 percent recall rises 0.634 -> 0.750; on unseen frames FPR at TPR 0.8 0.015 -> 0.007. But it is one split of a small, clip-clustered set (n=259; the unseen-frame gain is 31 positives in the temporal split), the leave-one-clip-out AP does not move, and the temporal protocol on all frames gets *much worse* (the combiner overfit to in-sample high-conf positives in seg010). Of 37 false candidates with conf >= 0.5, 31 are further from the reference than the median true ball, so the reference does carry some signal, but distance alone is weak (AUROC 0.70).
- Reference size (`b_ref_size_sensitivity.csv`): AP does not improve with more reference balls (3 balls 0.861, 207 balls 0.799; test seg >= 40), i.e. a handful of confirmed balls is as good as hundreds, and none give a reliable gain.

### B.3 Failure cases / kit and background collisions (`b_montage_*.png`)
- False candidates closest to the reference (d ~ 0.66-0.78): white/grey boots, socks and shorts edges on grass, a yellow boot, a spare ball by the hoardings. The descriptor (8x8 grey patch + 8-bin hue histogram of a 240-px window) cannot distinguish a white sock from a ball. Kit/ball collision here is mainly *white socks and shorts* (Arsenal white shorts/Man City white socks), not the red/sky-blue shirts.
- True balls farthest from the reference (d ~ 1.1-1.3): balls seen against the crowd or hoardings (seg060, seg110): the window is context-dominated, so ball position relative to background, not the ball, drives the distance.
- Only one match (kits fixed), so a collision between ball colour and a different kit pair was not measurable.

## Caveats (honest list)
1. **Single match, single detector** for A and B. No cross-match generalisation could be tested.
2. A's detections are **stored, floor 0.10** (not 0.05); frames mapped by a constant +12 frame offset verified on only 5 frames (IoU >= 0.5 matching is tolerant of a frame or two of motion, and the GT-matched referee score distribution looks healthy, but sub-frame misalignment is not excluded).
3. 257 eval frames / 424 GT referees: AP deltas of the size seen (-0.01 to -0.13) are resolvable, but a +0.01 improvement would not have been. The 95 percent CIs above are frame-bootstrap (A) and clip-bootstrap (B).
4. GT "referee" includes linesmen. GT completeness (e.g. every player labelled) is assumed; 11 referee GT boxes have no candidate at all.
5. H palette: the 10 GT player exemplars are clustered without team labels (none exist), which stands in for "click 5 per team".
6. B: labels come from 12 clips dominated by a few; 901/1829 candidates are in the detector's training frames (flagged); Ball Check verdict set is selection-biased (cards chosen by the active-learning/autolabel loop).
7. `hsv4` torso encoding only; a saturation-weighted variant (`sat3`) is coded in `a_eval.py` but was not run (time); I do not expect it to change the conclusion given the stacked result.
8. Not measured: effect on tracking/ID (the design doc's actual use of colour), GK role, hue-only palette for a second kit change, anything at conf < 0.10.

## Recommendation
- **Referee rescoring (A): not worth building.** The detector is already strong on clean frames (AP 0.936); naive colour likelihoods make it worse; learned stacking is a wash. Do not spend effort on a colour re-scorer.
  If anything, the useful application is *non-rescoring*: team clusters as a **constraint/attribute in tracking** (design doc 3.6) and colour-vs-detector disagreement as a **review flag** (colour llr alone has AUROC 0.976, so "detector says referee but colour says team A" is a cheap QA queue).
  Needs more GT before revisiting: team labels, referee/linesman as separate classes, a second match with stored detections on clean frames, and `arsenal_example` + other-match test frames (needs a GPU slot for a fresh detector run: `gpu_detect.py` and the Prefect flow below are ready).
- **Ball rescoring (B): not worth it with this descriptor**; possibly worth revisiting with a *ball-centred* descriptor (small window, background-masked, plus size/motion), and more ball GT outside the few dominant clips and outside the detector's train frames.
- Revise M8 in `player_reid.md`: quote referee reliability as recall-at-0.5 on GT (93 percent here), not median over all referee detections.

## Constraints / housekeeping
- Prefect: I added and ran deployment `colour-prior-detect/colour-prior-detect` (flow file `~/code/jobs/footy_colour_prior.py`, `colourprior.prefect.yaml`, `colour_prior_detect.py`, copied to the desktop; untracked additions to `~/code/jobs`). It stayed Late behind three 60-epoch training runs and I **cancelled it**; the deployment and files are still there if you want them (`gpu_detect.py` here is the same script). Nothing was run on the GPU.
- Root disk was ~400 MB free when I started; my first scratch venvs (on `/tmp`) briefly filled it to 0. They were deleted immediately and the Python env now lives at `/mnt/storage/colour_prior_venv` (1.8 GB, CPU torch + ultralytics + sklearn; safe to delete). The footy `.venv` was never touched.
- Reproduce: `uv`-style env as above; `python b_build_dataset.py; python b_eval.py; python b_extra.py` (B); `python a_map_dets.py; python a_eval.py` (A; `NDRAWS=20`, ~12 min). Inputs: `data/frames_meta.tsv` (frame lists), outputs `data/dets_cpu.jsonl` (stored detections mapped to GT frames; name kept from the abandoned CPU run).

## Files
`README.md`, `a_map_dets.py`, `map_check.py`, `a_eval.py` (+ `a_eval_hsv4.log`), `a_flip_montage.py`, `a_results_raw.csv`, `a_results_summary.csv`, `a_stacked.csv`, `a_transitions_U.json`, `a_bootstrap_U.json`, `a_diagnostics.json`,
`a_palette_info.json`, `a_flipped_roles_U_beta1.csv`, `a_palette_audit.png`, `a_lost_referees_montage.png`; `b_build_dataset.py`, `b_eval.py`, `b_extra.py` (+ `b_extra.log`), `b_results.csv`, `b_bootstrap.json`, `b_ref_size_sensitivity.csv`,
`b_scored_candidates.csv`, `b_montage_false_looks_like_ref.png`, `b_montage_true_far_from_ref.png`; unused-but-ready: `gpu_detect.py`, `a_detect_cpu.py`; `data/` (small caches).
