# Offline ball tracking: what the experiments say (updated 2026-11-15)

Companion to `reports/Offline football ball tracking models.md` (the research report). Everything here was measured on our footage. Each line gives its branch, so it can be re-run.

## The answer so far

1. **No off-the-shelf model to fine-tune as the core.** WASB (the strongest public soccer ball heatmap model) is a clear negative on our broadcasts. SAM2 is useful, but as a propagation engine rather than a detector.
2. **Build our own pipeline:** detector candidates → bidirectional trajectory solver → SAM2 gap-fill from confident anchors. Every stage has now been tested and adds something without breaking correct frames.
3. **Neighbouring frames help the detector find balls it otherwise misses, mostly fast-moving ones.** The gain survives the solver. A small 5-channel YOLO doesn't yet beat production RT-DETR-L alone, but its candidates add to RT-DETR's.
4. **The binding problem is generalisation across matches, not architecture.** RT-DETR scores 0.92 per frame on the image-split v11 test, 0.67 on arsenal_example and 0.49 on held-out clips. More varied labelled data (Norwich, other matches) is the biggest lever.
5. **New (round 3): self-training on the target match's own unlabelled video works.** Pseudo-labelling a match's video (RT-DETR → solver → SAM2 fills, no human labels) and retraining lifts held-out Bournemouth hit@10 from 0.49 (RT-DETR) to **0.70**. It does nothing for a match whose video wasn't pseudo-labelled, and it costs about 0.09 on arsenal_example. For an offline tool this is a usable recipe: run the pipeline once on a new match, retrain on its own pseudo-labels, then run again.

## Results ladder (ball within 10 px; 9 clean Man City test clips, n=170 | arsenal_example, n=176)

| Pipeline | Test clips | arsenal_example | Branch |
|---|---|---|---|
| RT-DETR-L per-frame top-1 (B0) | 0.741 (prec 0.947) | — | exp/offline-ball-tracking |
| + trajectory solver (B1) | 0.759 (prec 0.985) | 0.540 (prec 0.913) | exp/offline-ball-tracking |
| + SAM2 gap-fill from conf ≥0.8 anchors (S1) | 0.782–0.806 (prec ~0.98) | 0.602 (prec 0.930) | exp/sam2-bidir-ball |
| RT-DETR + 5ch-YOLO candidates → B1 (no S1 yet) | 0.788–0.800 | 0.562–0.602 | exp/temporal-detector |
| Candidate ceiling, RT-DETR alone (any candidate within 10 px) | 0.829 | 0.631 | — |

- SAM2 gap-fill broke **0** correct solver frames in every evaluation. Its only false fills came from one wrong anchor at conf 0.73, which the 0.8 threshold removes.
- Small-n warning: seg010 holds 65% of the test balls, so clip-level confidence intervals are wide. The no-ball and out-of-shot evidence is only 22 frames plus 140 Ball Check "not ball" verdicts.

## Rounds 2 and 3: pseudo-labels (YOLO11s at 1280, 2 seeds each)

Hit@10, held-out balls by match (branch exp/temporal-detector, docs/experiments/temporal_detector_round3.md):

| Held-out subset (n) | RT-DETR-L | A v1 (v11 only) | A+ r2 (+ Man City pseudo) | A++ r3 (+ Bournemouth & Norwich pseudo) |
|---|---|---|---|---|
| All (117) | 0.487 | 0.462 | 0.479 | **0.641** |
| Bournemouth (76) | 0.487 | 0.421 | 0.447 | **0.697** |
| Man City (41) | 0.488 | 0.537 | 0.537 | 0.537 |
| arsenal_example held-out frames (20) | 0.050 | 0.050 | 0.050 | 0.050 |

- **Round 2** (1,617 filtered Man City pseudo frames): no lift on held-out data, for either 3ch or 5ch. More data from the same match doesn't generalise.
- **Round 3** (transductive): A++ − A+ on Bournemouth is **+0.25 [+0.16, +0.34]** (clip-level CI [+0.14, +0.41]); A++ − RT-DETR is +0.21 [+0.13, +0.30]. Candidate recall any@10 on Bournemouth: 0.55 → 0.85. Fast balls gain most.
  - Held-out Bournemouth frames come from disjoint segments of the same match as the pseudo-labelled clips. That is the setting the offline tool will have, but it says nothing about unseen matches.
  - The two seeds scored 0.658 and 0.624 on all held-out balls.
- **Costs of round 3:**
  - arsenal_example val drops by 0.087 [−0.13, −0.04]. Through the solver it drops 0.05 raw and 0.16 iso, though the two seeds disagree on raw. Not verified: Man City may be diluted, now 34% of the pseudo data.
  - In-domain Man City test is flat (−0.008).
  - Through the solver on the 9 test clips, it is still above production RT-DETR: +0.03 raw, +0.06 iso, precision unchanged.
  - On 35 ball-free images (small n), confident false positives at conf ≥0.7 rise from 11% to 17%.
- **The 20 arsenal_example held-out frames are at 0.05 for every model.** That points to hard or mislabelled frames, not missing data. Worth a manual look.
- **Pseudo-label quality:**
  - On RT-DETR's own scale, low-confidence detections don't transfer across matches: on Bournemouth, B1 conf 0.6–0.8 was 5/9 correct, against 98% on Man City. Round 3 therefore dropped them.
  - George's blind Ball Check of 200 Norwich cards: easy anchors 58/60 real (97%, CI 89–99%), SAM2 fills 53/60 (88%, CI 78–94%), so about 94% set precision. B1 nodes with p < 0.5 were real 22/40 times. RT-DETR detections at conf ≥0.6 that the solver rejected were real 1/40 times, so the solver's rejections are right, and RT-DETR confidence is badly calibrated on a new match.
  - Contact sheets agree: easy anchors are all clean; about 10% of SAM2 fills sit on logos, crowd or ad boards.
- **Not done:** no Norwich score (no window labels yet); the arm without Man City pseudo-labels was cancelled (about 5 h per run); no RT-DETR-L fine-tune.

## Individual findings

- **WASB zero-shot:** 15% vs RT-DETR's 76% on the same 202 clean frames; tiling makes it worse. Its weights have no licence, so treat them as research-only. (exp/wasb-zeroshot)
- **SAM2.1 propagation from human anchors** (labelling aid), hit@10 with an anchor every 25 frames vs linear interpolation: arsenal_example 84% vs 51%, Villa 94% vs 59%, Bournemouth 100% vs 67%. When the forward and backward passes agree within 5 px, the result is within 10 px 96–100% of the time. It fails on fast kick plus whip-pan. Cost is 0.15 s per frame and under 1 GB of VRAM. (exp/sam2-bidir-ball)
- **Temporal input** (YOLO11s at 1280, 3ch vs 5ch [t−2 aligned, BGR t, t+2 aligned], 2 seeds each, B−A per-frame top-1):
  - arsenal_example: +0.21 (both seeds positive; 41 vs 2 discordant frames).
  - Held-out clips: +0.017 [−0.013, +0.047].
  - Test: ≈0.
  - Candidate recall on the fastest third of balls: +0.066.
  - After the solver: +0.03 test and +0.25 arsenal_example. (exp/temporal-detector)
- **Camera alignment:** LK+RANSAC with persons masked is best (pitch residual 0.49 vs 1.07 with no alignment). ORB is worse than no alignment. The player-box-shift camera model inside the solver is weak on the TNT (Norwich) footage, because crowd members get detected as players.
- **Contamination:** v11 splits by image, not by clip (33 clips are shared between train and test), so v11 test numbers are optimistic. The solver test set is clean apart from 15 frames, which are excluded.

## Data built

- `multiframe_v1/k2`: 1,633 aligned 5ch frames with the v11 labels, plus 117 held-out balls. (exp/multiframe-dataset)
- `multiframe_pseudo_v1`: 2,115 pseudo-labelled frames from 62 Man City clips with no evaluation images. 1,617 survive filtering (easy ~100% correct, SAM2 fills ~95%, filtered low-conf ~87%). (exp/pseudo-labels-v1)
- Norwich (actually Arsenal v Forest; the file is truncated at about 40 min): 76 clips, RT-DETR candidates, and a 24-window dense labelling queue in four groups: easy, gappy, contested and low-evidence. (exp/norwich-heldout, exp/norwich-label-queue)

## Labelling advice (unchanged, now backed by evidence)

- Label Norwich queue windows densely: a ball-centre keyframe every 5 frames plus at touches, with state spans for out-of-shot, dead and occluded. Add a few random windows as well, so the accuracy estimate isn't biased.
- To label faster, place a ball every 25–50 frames and let SAM2 propagate both ways. Accept automatically where the two passes agree, and review the rest.
- Prioritise new matches and camera styles over more Man City frames.

## Infrastructure notes

- The Ball Check labeller on :8011/:8012 needed clips re-encoded with a keyframe every 10 frames (`scripts/reencode_keyframes.py`, PR #48/#51) to load fast on a phone. The originals had 2–4 keyframes per clip, so every card decoded from the start.
- The /mnt/storage disk has an unreadable sector (Bournemouth seg092). Its labels are backed up to S3, but the disk may be failing.
- The desktop's tailnet path sometimes drops to a ~30 KB/s relay; large transfers go through S3.

## Next steps (proposed)

1. **Label the 24 Norwich windows** (board task t-toitf2). This gives the first score on a match no model has seen labels for, and tests whether Norwich self-training helps the way Bournemouth's did.
2. **Look at the 20 arsenal_example held-out frames** that every model misses.
3. **Make self-training a per-match step** in the offline pipeline. Fix the arsenal_example regression first, by rebalancing the source matches or fine-tuning per match from the base model instead of one model trained on everything.
4. **Production-size detector:** RT-DETR-L or a larger YOLO, trained with the same pseudo-label recipe.
