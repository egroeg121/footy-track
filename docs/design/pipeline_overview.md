# Pipeline overview: from broadcast video to "who did what"

Where each model sits, what is measured, what is labelled, and how the loop
improves. Every number here was measured in this project; anything unmeasured is
marked as such.

## The goal, split into three problems

1. **Where is the ball**, every frame.
2. **Where are the players**, every frame.
3. **Which player is which**, within a segment and then across segments.

These are not one problem. The ball is a single ~16 px object with no identity —
there is only one, so "which ball" never arises, and the hard part is not losing
it. Players are ~20 near-identical objects where identity is the entire problem
and localisation is comparatively easy.

Treating them with one pipeline is the main modelling mistake available here.

## Data flow

```
broadcast video
      |
      v
  [segmenter]  ── already cut on shot boundaries: 0 interior camera cuts
      |            measured across 3,175 frames / 5 clips
      v
   162 clips (one match), 97,625 frames
      |
      v
  [RT-DETR-L]  ── per-frame detection: player / referee / in_play_ball
      |            2,767,151 detections, ~19.8 fps on a 3060 @ imgsz=1280
      v
  detections (TIER 3, machine-only)
      |
      +──────────────► ball path ──► [temporal/ball tracker]  ── UNSOLVED
      |
      v
  [LapTracker]  ── IoU + Hungarian association -> track_id
      |             ~540 fps on CPU; no training, no GPU
      v
  tracklets  (~5,249 over 40 clips; median length 13-31 frames)
      |
      +──► [kit-colour clustering] ──► team_id      ~99% separable, free
      |
      +──► [human review] ──────────► purity + jersey numbers
      |
      v
  [identity / ReID]  ── UNPROVEN: cluster tracklets into players
      |
      v
  player_id per detection ──► feature store ──► match data
```

## Which model does what

| Stage | Model | Trained? | Status |
|---|---|---|---|
| Detection | RT-DETR-L, `imgsz=1280` | fine-tuned on v11 | **Works.** Ball mAP50 disputed — code comment says .221/.209, handoff says .665/.591; **re-measure before citing** |
| Player tracking | LapTracker (IoU + Hungarian) | training-free | **Works, fragments.** Consecutive-frame IoU median 0.72, only 4–6% below the 0.3 threshold; spawn rate 4.2% matches that tail. Fragmentation is a *detection dropout* problem, not an association problem |
| Ball tracking | VitTrack | training-free | **Fails.** Holds the ball for 5% of frames; confidence is a constant 0.119 and cannot separate on-ball from drifted. Do not tune — replace |
| Team assignment | k=2 on torso HSV | none | **Works.** 1.0% ambiguous (n=1,253). Hue 185° vs 23° |
| Identity / ReID | — | — | **Unproven.** Generic ImageNet features score AUC 0.43–0.49 (chance) because they encode background, not person. ReID-trained weights untested |
| Purity pre-filter | Gemini Flash | — | **Not usable yet.** 70% agreement, confidence collapsed into 0.65–0.80 so there is no bucket to trust |

Two consequences worth stating plainly:

* **Tracker fragmentation is a detector problem.** Association is near the limit
  of what the detections allow, so effort belongs in detection consistency
  (or interpolation across dropouts), not in a cleverer tracker.
* **The ball needs a different mechanism entirely.** A single-object appearance
  tracker cannot follow a 16 px ball through broadcast motion. The candidates
  are per-frame detection plus motion-model interpolation, or a
  prior-conditioned detector — not another SOT tracker.

## What gets labelled, and by whom

| Label | Who | Cost | Feeds |
|---|---|---|---|
| Ball position | human, per frame | expensive | detector training + ball eval |
| Detections | RT-DETR | free | everything |
| `track_id` | LapTracker | free | tracklets |
| `team_id` | kit colour | free | identity constraint |
| **Tracklet purity** | human, per tracklet | ~30/clip vs ~6,500 boxes — **250× cheaper than per-box** | tracker eval **and** free identity pairs |
| **Jersey number** | human, when legible | seconds | identity anchor |
| Tracklet merges | human, pairwise | expensive without proposals | identity clusters |

**Purity review is the keystone.** It is the tracker's benchmark *and* the gate
on identity training data: a pure tracklet is a free bag of same-player pairs,
but one undetected switch injects thousands of false positives that teach the
model two players are one person.

**Provenance tiers hold throughout.** TIER 1 human, TIER 2 machine + human
checked, TIER 3 machine only. For tracklets the tier attaches to *frame
intervals*, not whole tracks: a human reviewing 288 frames sees ~12, so claiming
the whole tracklet is checked is a false claim.

## Auto-labelling and the improvement loop

The pipeline already auto-labels — 2.77M TIER-3 detections exist. The loop is
about promoting that output to trustworthy labels without a human touching
every frame.

```
      machine labels (TIER 3)
             |
             v
   propose  ──►  human verifies  ──►  TIER 2  ──►  retrain
      ^                 |
      |                 v
      └────── uncertainty sampling ◄── model disagreement / low margin
```

**Three grounding mechanisms, cheapest first:**

1. **Kit colour** — free, ~99% accurate, and a *hard constraint*: two tracklets
   of different teams can never be the same player. Deletes a large slice of the
   pairwise space with no model and no labels.
2. **Jersey number** — one legible frame anchors a whole cluster, because number
   + team is unique within a match. ~40% of tracklets contain a frame ≥110 px,
   and numbers have been read by a human at 102 px, so the addressable pool is
   far larger than first estimated. This is the highest-value label per second.
3. **Pairwise merge** — general but expensive; only worthwhile once a model can
   propose candidates and the human just confirms.

**Active learning, per stage:**

* *Detection* — sample frames where tracking implies a detection should exist but
  none was produced (dropouts are the measured bottleneck), label those, retrain.
* *Tracking* — no training; purity review tunes `iou_threshold`/`max_age`.
  Note the asymmetry: **fragments are harmless, switches poison.** Optimise
  purity, use coverage only as a guard against one-frame tracklets.
* *Identity* — uncertainty sampling on embedding margin, constrained to
  same-team pairs. Cross-team pairs are trivial negatives and teach nothing.

**The discipline that makes this safe:** never let machine output promote itself.
A pre-filter is only usable if its confidence is *calibrated* — high-confidence
answers reliable enough to accept unreviewed, errors concentrated where the human
is looking anyway. Gemini Flash currently fails that test not on accuracy but on
calibration, which is why it is not in the loop.

## Open questions, in the order they should be answered

1. **Does a ReID-trained embedding separate same-player pairs here?** Everything
   downstream depends on it. Untested — the two failures so far were ImageNet
   backbones, which is not the same claim.
2. **What is the tracker's real purity?** Being measured now. Needs *complete*
   clips; partial coverage biases toward whatever was reviewed first.
3. **What replaces VitTrack for the ball?**
4. **Which ball mAP figure is true?** `.221` vs `.665` for the same weights.
5. **Is per-player identity even needed downstream**, or is team + role enough
   for the questions being asked of the data?

Question 5 is worth an hour before answering 1–3: it can delete a workstream.
