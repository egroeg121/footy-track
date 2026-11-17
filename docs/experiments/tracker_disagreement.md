# Three independent ball trackers on one match: where they disagree, and is agreement trustworthy?

Branch `exp/tracker-disagreement` (from `exp/sam3-offline-labeller`), code `experiments/tracker_disagreement/`, flows in `~/code/jobs` (`footy_sam3.py`, `footy_temporal_clips.py`; params added, no new flows).
Numbers: `docs/experiments/tracker_disagreement/stats.json`; sheets: `docs/experiments/tracker_disagreement/*.png`.

**Naming.** The held-out match everyone called "Norwich" is Arsenal v Nottingham Forest; main renamed it mid-run (PR #62: clips `arsenal_nottinghamforest_segNNN`, dataset `nottinghamforest`).
The run's artifacts (round-3 spec, seeds, tracks, fused positions on the mini and desktop) keep the old ids `arsenal_norwich_segNNN`; `build_batch.py` writes the list with the new ids and keeps the old one as the hidden key `clip_run_id`.

## What was run

| tracker | what | frames | GPU wall |
|---|---|---|---|
| T1 A++/B1 | round-3 A++ YOLO11s 3ch seed 0 (`runs/r3/App_s0/last.pt`) every-frame candidates (`temporal-clip-infer`, `exp_tag=r3 keys=App_s0 clip_paths=...`) -> harness B1, camera on, dev-tuned params, raw knots | 6 619 (every frame) | 15.9 min (~7 fps, RT-DETR + LK alignment included) |
| T2 SAM2.1-small crop512 | `sam3-label-match script=track_seeds.py engine=sam2`, seed fwd + bwd, reach 250, overlap 15 | 14 374 tracked frames (280 tracks, 6.8 fps) | 39 min wall (35 min tracking + per-clip model load / staging) |
| T3 SAM 3 box crop512 | same, `engine=sam3` (fp16, sm_75 patches) | 14 374 tracked frames (2.37 fps) | 117 min wall (101 min tracking; includes one crash, see Problems) |
| text seeds | `sam3-seedfind`: "ball", 384 px tiles, one probe per 75 frames | 87 probes | 9.6 min |

Total about 3.1 h of desktop GPU wall, all as queued Prefect runs on the `gpu` pool, one at a time.
Both SAM trackers read the **same seed file** (`seeds.jsonl`): 76 RT-DETR conf >= 0.8 anchors (the round-3 B1 anchors, thinned to >= 40 frames) + 64 SAM 3 text seeds (top-1 "ball" detection of a probe, score >= 0.5, longest side <= 50 px;
9 more duplicated an anchor and were dropped), in 29 of the 31 clips (seg065 and seg073 got no seed: SAM trackers silent there, T1 only).
`track_seeds.py` = `sam3_label.py` with an `--engine` switch (identical seeds, windows, S1 merge and outputs for both engines); a SAM tracker "has a ball" on a frame when the SAM object-score logit is > 0 (the model's own object-present decision) or the frame is a seed.

**Clips.** 31 window-free clips of the 76 (random, seed 7) = 6 619 frames (not 20: the window-free Forest clips are short, median 170 frames; 20 would have been ~4k frames). None of them contains a dense-labelling window, so the +-25 frame exclusion removes nothing here
(the mask is still applied in `fusion.py`). `ball_gt_marks_*` verdicts were used only to skip frames within 3 of a frame George already judged (70 frames), never for fusion.
Anchors and A++: A++ was trained on 1 138 Forest pseudo-label frames (anchors + S1 fills from SAM2) of this match; 15 of the 44 all-agree cards lie within 2 frames of such a training frame (hidden key `a_train_dist`), so "A++ agrees" is partly memorised.

## Agreement breakdown (6 619 frames, tolerance 8 px)

| class | frames | share |
|---|---|---|
| all three have a ball and agree | 4 999 | 75.5 % |
| two agree, third has no ball (`pair_agree`) | 808 | 12.2 % (806 of them: SAM2.1 + SAM3 agree, A++/B1 silent) |
| two agree, one differs (`two_agree`) | 194 | 2.9 % (odd one out: A++/B1 130, SAM2.1 39, SAM3 25) |
| two have a ball and differ (`pair_differ`) | 88 | 1.3 % (86: SAM2.1 vs SAM3, A++/B1 silent) |
| all three differ | 3 | 0.05 % |
| only one tracker has a ball | 158 | 2.4 % (SAM2.1 100, SAM3 31, A++/B1 27) |
| none | 369 | 5.6 % |

Coverage (frames with a ball): A++/B1 79.0 %, SAM2.1 93.5 %, SAM3 92.5 %. SAM2.1 and SAM3 agree on 97.5 % of the 6 088 frames where both have a ball (shared seeds); A++/B1 agrees with each on ~97 % of the frames where both have one.
When two trackers differ they are usually far apart (median 140 px SAM2 vs SAM3, 330 px A++ vs a SAM tracker): they are on different objects, not a few pixels off.
Disagreement is rare and short: 58 segments / 285 frames (median 2 frames, max 26), not "dozens of frames" (the dense seeds, one per ~30 frames, re-anchor both SAM trackers). A++/B1 loses the ball (present then absent for >= 3 frames while another tracker still has it) 63 times, SAM2.1 6, SAM3 2.

SAM "coverage" is not a ball-presence signal: 93 % of frames carry a SAM position because a SAM tracker nearly never says absent; in seg058 and seg074 both SAM trackers follow *something* on every frame while A++/B1 has a ball on 8 % / 1 %.
Those are the 12 % `pair_agree` frames, and their agreement is correlated (same seed, same crop mechanism), so they are the least informative agreement and are sampled heavily (54 cards).

## The batch: `ball_check_list_v2_disagree.json` (200 cards, 30 clips, <= 14 per clip)

| stratum | cards | selection |
|---|---|---|
| `disagree` | 66 (24 two-box cards + 21 far pairs as 2 single-box twin cards = 42) | 49 disagreement segments, one card each (frame of max disagreement, <= 250 px apart when possible); priority SAM3-vs-others, text seeds, fast, crowd / occlusion; round-robin over clips |
| `pair_agree` | 56 | two trackers agree, third silent: 54 SAM2.1 + SAM3 (A++/B1 silent), 1 + 1 for the other pairs (there are no more) |
| `all3_agree` | 44 | blind random sample of all-agree frames, round-robin over clips (precision of agreement) |
| `loss` | 14 | just before (t_last) / after (t_last + 2) A++/B1 loses the ball while a SAM tracker still has it |
| `text_only` | 11 | SAM trackers follow a text seed, no RT-DETR anchor within 40 frames, A++/B1 silent |
| `single` | 9 | only one tracker has a ball (4 SAM2.1, 3 SAM3, 2 A++/B1) |

Cards are ordered by sha1(salt, clip, frame, stratum, tracker), so strata are not recognisable; `bbox` / `other_bbox` are a fixed 18 x 18 px box on the tracker position (a tracker's own box size would leak its identity),
which tracker is `bbox` is a seeded coin flip (`src_bbox` / `src_other`, t1 = A++/B1, t2 = SAM2.1, t3 = SAM3). Hidden keys: stratum, class, per-tracker positions / scores / gate / seed source, A++ node + conf, speed, player context,
`a_train_dist`, segment length. No `note` / `provenance` / `role` / `confidence` keys (the page prints those). Skipped: seed frames (they are detections, not tracking), frames within 3 of George's earlier verdicts.

**How to judge a two-box card.** Yellow box = `bbox`, magenta = `other_bbox` (drawn on the still only, visible inside the default crop when closer than ~110 px). Yellow on the ball: ball. Magenta on the ball instead: use the corrected
verdict and drag onto it (the scorer maps the dragged box to the nearest tracker). Neither: not_ball. A not_ball cannot say whether the *other* box was right, so those are reported as `unresolved` per tracker.
That is why pairs further than the crop are two single-box cards of the same frame (blind, far apart in the order): each tracker gets its own clean verdict. The smallest UI change that would make this one card is an extra verdict "other box is the ball"; not built.

## Scoring (`score_disagreement.py`)

Reads the verdict log (`<marks_dir>/ball_checks/*.jsonl`, same last-wins / tombstone rules as the labeller) or `labelling_jobs results` output, joins to the list by (clip, frame, box_index) and checks the shown bbox matches the card.
Truth model: ball / box_off = ball under the shown box; corrected = centre of the dragged box; hit radius 10 px. Reports verdicts by stratum; agreed-frame precision (all three, each pair, A++-trained vs not) with Wilson 95 % CI, strict and lenient and clip-frame-weighted;
per-tracker right / wrong / unresolved on disagreement cards and head-to-head; per-situation tables (fast, crowd / occlusion proxy, text-only seeds, loss before / after, class) with a "best" call only when the top tracker's lower Wilson bound beats the others' point estimates (n >= 8).
`python score_disagreement.py --selftest` exercises it on synthetic verdicts.

## Problems and caveats

- **The match was renamed under the run** (Norwich -> Nottingham Forest, PR #62) while the SAM 3 flow was on clip 26: the flow failed on a vanished source directory (retries too); resumed with `clips_dir` + `clip_files` (old id -> new file name), finished clips and tracks were skipped.
- **SAM trackers never lose the ball on their own** (see above), and they share seeds, so SAM2.1 - SAM3 agreement mostly measures that both follow the same seed. A++/B1 is the only seed-independent tracker; A++ saw this match's pseudo labels (derived from SAM2 fills).
- **False seeds.** 48 random text seeds (`seeds_text_sheet.png`): ~44 on a clear ball, ~4 on ball-like graphics on LED advertising boards (a ball in a sponsor graphic, TNT / "V" logo), none obviously on a boot; scores 0.5-0.87. RT-DETR anchors were not sheeted separately (round-3 anchors were B1-filtered; 97 % real in the blind check).
  Text seeds are 46 % of seeds and feed 46 % of SAM positions.
- Frames where several balls are in play (spare balls beside the pitch) make "two trackers on two different balls" a real third kind of disagreement; George's verdicts will show how often.
- A++/B1 uses the harness's raw RT-DETR calibration knots on YOLO confidences (not isotonic), as in the pipeline eval; its fills (kalman) count as "has a ball".
- No verdicts exist yet: nothing here says which tracker is right, only where they differ.

## Serving (main labeller :8011, labelling-jobs flow; nothing submitted or registered yet)

1. `python experiments/tracker_disagreement/register_dataset.py --apply` adds the dataset `nottinghamforest_disagree` (Forest keyframe-dense clips, the v2 list, its own empty marks dir so verdicts and corrected boxes never touch `ball_gt_marks_nottinghamforest`) to
   `/mnt/storage/footy_data/labeller_datasets.json` (the server re-reads it, no restart).
2. `make_job_items.py` writes the pointer items (`{clip, frame_index, box_index, source: "list"}`, no hidden keys) and prints the submit command (`labelling_jobs submit --kind ball_check --dataset nottinghamforest_disagree ...`).
3. Warm the renders once it is live: `cd /home/george/footy-track-main && PYTHONPATH=src nohup nice -n 19 python -m footy_track.scripts.warm_ball_check --base http://127.0.0.1:8011 --dataset nottinghamforest_disagree`.
4. Results: `labelling_jobs results <title> -o job.json`, then `score_disagreement.py --list ... --results job.json`.

## Fourth tracker: the labeller's own autodetect + VitTrack (added 2026-11-17)

**Why.** VitTrack propagation is a large share of the ground truth (`ball_gt_marks`: 471 hand-placed ball boxes vs 737 `human_checked + vittrack`; Forest: 66 hand-placed vs 168 unchecked `vittrack`), and its accuracy on hard frames had never been measured.

**What ran (code path, not a reimplementation).** `vittrack_track.py` imports a snapshot of footy-track `origin/main` (ae41909, on the mini at `/mnt/storage/exp/disagree/main_src`, rsynced by the flow) and calls the labeller's own functions:
`yolo_seed_objects` (what `POST /autodetect` calls: `get_current_best_detector` = RT-DETR-L v11 `rtdetr_1920_v11/best.pt` at imgsz 1280, conf 0.35, NMS 0.5), `BackgroundLabeller` (what the Run button drives: worker thread, `_track_anomaly_reason`, per-class confidence hand-back)
and `VitTrackVideoLabeller` / `VitTrackSOT` (OpenCV-zoo ViT SOT, ONNX, CPU). Only the ball is seeded (each object has its own independent tracker and the checks are per label, so the ball track is identical to a run that also carries ~18 players, at 1/18 of the cost); the detector is memoised per process (same weights).
Clips are the keyframe-dense kf10 copies the labeller serves. Run as `sam3-label-match engine=vittrack script=vittrack_track.py` on the `gpu` pool (CPU tracker, the GPU only for the detector): arm 1 541 s, arm 2 827 s wall for 31 / 29 clips (tracker time 97 s / 177 s; the clips are staged and the model loaded per clip).
- **Arm 1, labeller-realistic:** autodetect on frame 0, best ball box, Run; on a hand-back the headless "human" re-detects on the stop frame (best ball box only) and Runs again. **At the page's default thresholds {in_play_ball: 0.5, other: 0.5} a run hands back almost every frame**:
  on seg001 (280 frames) 117 hand-backs, 118 runs, 171 autodetect calls; VitTrack's confidence is 0.06-0.14 right after a seed and its median over a clip is 0.42-0.48 (arm 2: 54 % of frames would hand back at 0.5, 25 % at 0.3, 18 % at 0.2).
  So the arm that feeds the fusion uses `ball_thr 0.0` (what you must do to get a run at all); the jump / size anomaly stop stays on and never fired: **one unattended Run from frame 0 to the end of each clip** (31 runs, 40 autodetect calls).
- **Arm 2, comparable:** per seed of the shared seed file, forward and backward (backward = the same labeller code on a reversed copy of the clip: the labeller has no backward Run), never past the neighbouring seed + 15 frames (276 tracks, 14 650 frames); hand-backs not applied, but every frame records the score and whether the labeller would have stopped.
Fusion: `fuse4.py` (a VitTrack frame "has a ball" when its score >= 0.20, VitTrack's own "found" threshold; a lost track keeps its last box with a low score, kept as `raw`), `fusion4.py` (4-way classes), `report_stats4.py`.

### Agreement breakdown with four sources (6 619 frames, 8 px)

| class | arm 1 (unattended Run from frame 0) | arm 2 (re-seeded every ~30 frames) |
|---|---|---|
| all four agree | 21.4 % | 59.9 % |
| three agree (one of the four silent) | 22.3 % | 18.1 % |
| two agree (two silent) | 3.9 % | 2.6 % |
| exactly one tracker is the odd one out of a majority | 42.7 % (VitTrack 2 732 of 2 957) | 10.2 % (VitTrack 517 of 716) |
| split 2 + 2 / all differ | 0.5 % / 2.8 % | 0.2 % / 1.5 % |
| only one tracker has a ball | 4.6 % (VitTrack alone 245) | 3.2 % (VitTrack alone 79) |
| none | 1.9 % | 4.4 % |

- **VitTrack coverage** (score >= 0.20): arm 1 70.9 %, arm 2 80.1 % (A++/B1 79.0 %, SAM2.1 93.5 %, SAM3 92.5 %); it always has a box, so coverage is a score threshold, not a presence signal. Loss events (position lost for >= 3 frames while another tracker still has the ball): 62 (arm 1), 61 (arm 2), against 63 / 6 / 2 for A++/B1 / SAM2.1 / SAM3.
- **Agreement with the others when both have a ball:** arm 1 agrees with A++/B1 on 39 % (1 472 of 3 800), with SAM2.1 on 35 %, with SAM 3 on 36 %; arm 2 on 88 % (3 998 of 4 532), 89 %, 89 %. The A++ / SAM trackers agree with each other 97-98 %. An unattended VitTrack run is therefore off the consensus ball on more than half of the frames; re-seeded every ~30 frames it is off on about 11 %.
- **Drift onto a player or boot.** On frames where VitTrack is the odd one out of a majority (seed frames excluded), its position is inside an RT-DETR player box on 67 % (arm 1, 1 778 of 2 658) / 80 % (arm 2, 382 of 475), against 36 % / 35 % for the majority position on the same frames. In the lower 30 % of the player box (boots, lower legs): 15 % / 34 % vs 31 % / 27 %:
  it drifts onto players' bodies and legs generally rather than specifically boots (contact sheet `contact_sheet_vittrack.png`). (Player boxes are the round-3 RT-DETR ones; "inside a box" is a proxy, George's verdicts will say which.)
- Preview of the live v2 verdicts (97 of 200 judged when run, not a result): VitTrack is scored on v2 cards only where it lies within 8 px of a shown box, so its precision there (43/52 right; 23/23 on all-agree cards, 3/7 on disagreement cards) is conditional on agreeing with a shown tracker, not an unconditional accuracy; the v3 list exists to measure it directly.

### v3 top-up: `ball_check_list_v3_vittrack.json` (76 cards, 25 clips, nothing registered or submitted)

| stratum | cards | selection |
|---|---|---|
| `vit_odd` | 31 | VitTrack is the odd one out of a majority (or splits / differs from everything), arm 2, one card per segment; priority inside a player box, several players near |
| `vit_longrun` | 11 | arm 1: >= 60 frames into the unattended Run, off an A++/SAM consensus that agrees |
| `vit_agree` | 20 | blind sample: VitTrack within 8 px of >= 2 other trackers (its precision when it agrees) |
| `vit_alone` | 6 | only VitTrack has a ball |
| `vit_loss` | 6 | the frame before VitTrack's score falls below 0.20 while another tracker still has the ball |
| `vit_pair` | 2 | VitTrack agrees with exactly one other tracker (almost no such frames exist) |

Same blinding as v2 (fixed 18 x 18 px boxes, seeded coin flip for `bbox`, hash order, hidden keys incl. `t4_score`, `t4_in_player_box`, `vit_arm`, the other arm's position; far pairs as twin single-box cards: 34 of the 76 cards). Frames within 3 of any v2 card or earlier verdict are skipped.
It is a separate dataset (`nottinghamforest_vittrack`, own marks dir) so its verdict keys cannot meet the live v2 job's. Scoring: `score_disagreement.py` now has four trackers; `--vit-positions v2_vit_positions.json` (from `make_vit_positions.py`) scores VitTrack on v2 cards within 8 px of a shown box.
