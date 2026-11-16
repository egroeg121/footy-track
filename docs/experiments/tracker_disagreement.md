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
- **False seeds.** 48 random text seeds (`seeds_text_sheet.png`): ~44 on a clear ball, ~4 on ball-like graphics on LED advertising boards (a ball in a sponsor graphic, TNT / "V" logo), none on a boot; sizes 0.5-0.87 score. RT-DETR anchors were not sheeted separately (round-3 anchors were B1-filtered; 97 % real in the blind check).
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
