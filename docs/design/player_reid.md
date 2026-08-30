# Player Identification & Re-Identification — Design

**Bead:** TBD · **Date:** 2026-08-30 · **Author:** design pass · **Status:** DRAFT

**Scope:** associating `player` detections (a) across frames within a clip and
(b) across clips. Covers how to **measure** it, how to **implement** it, and how
to **collect the labels** it needs. Resolves the re-ID / team / jersey TODOs
deferred in [`player_tracking_format.md` §6](player_tracking_format.md) and
[`tracking.md` §6](tracking.md).

> **Headline.** The tracking *code* already exists (`src/footy_track/trackers/`,
> 599 LOC, `LapTracker` + `UltralyticsTracker`) and the *storage* already exists
> (`detection.track_id`, `track_meta.{team_id,jersey_number,player_id,reid_parent_track_id}`).
> Neither has ever been run against ground truth, because **there are zero player
> identity labels and zero MOT metrics in this repo.** So the bottleneck is not
> algorithms — it is measurement. I measured the footage directly (§0) and three
> facts reshape the problem: **clips contain no interior camera cuts** (0 in
> 3,175 frames across 5 clips), so within-clip tracking is single-shot tracking;
> **frame-to-frame IoU is easy** (median best-IoU 0.837, only 3.6% of
> associations ambiguous); and **kit colour separates the two teams essentially
> perfectly** (1.0% ambiguous). Meanwhile **jersey OCR is dead on arrival** —
> the median player box is 52×111 px, so digits are ~10–18 px tall. The
> recommendation is therefore: build a ~2-human-hour benchmark, measure the
> trackers we already have, add a free team-constraint, and **build nothing else
> until the numbers say so.** Do not repeat the VitTrack mistake of assuming a
> tracker works for a year without measuring it.

---

## 0. Measurements taken for this document

Everything below was measured on 2026-08-30 against real project data, not
cited. **Read the denominators.** Most numbers come from a *single clip*
(`arsenal_mancity_20250925_seg002`, 430 frames, 17.2 s) and must be re-run
across the corpus before anything is built on them (§5, R2). Reproduction
commands are in the Appendix.

| # | Measurement | Value | Denominator |
|---|---|---|---|
| **M1** | **Interior camera cuts per clip** | **0** | 5 clips (seg001/002/004/005/008), **3,175 frames**. Each clip has exactly one histogram discontinuity, located in its **final 1–4 frames**. seg004 (175 fr) has none. |
| **M2** | Players per frame | mean **19.5**, median 20, max 32 (conf≥0.10)<br>mean **14.5**, median 15, max 18 (conf≥0.5) | seg002, 430 frames |
| **M3** | Player box size @1920×1080 | median **52 × 111 px**; height p10 74 px, p90 149 px; width p10 26, p90 84 | seg002, n=**8,398** player boxes |
| **M4** | Consecutive-frame best IoU | median **0.837**, mean 0.780, p10 0.545; **4.4%** below 0.3; 3.3% below 0.1 | seg002, conf≥0.5, n=**6,242** instance-pairs |
| **M4b** | Association margin (best − 2nd-best IoU) | median **0.810**, p10 0.444; **3.6%** below 0.05 | same n=6,242 |
| **M5** | Crowding / occlusion | **6.3%** of player instances overlap another player at IoU>0.2; **2.6%** at IoU>0.5; **36.5%** of frames contain ≥1 overlapping pair | seg002, n=6,242 instances / 430 frames |
| **M6** | Kit-colour team separability | k=2 on torso HSV → split **541 / 712**, **1.0%** ambiguous. Centroids hue **185°** (sky blue) and **23°** (red). k=3 adds a 255-crop hue-61° cluster (likely referee/GK). | seg002, n=**1,253** torso crops, every 5th frame, conf≥0.5 |
| **M7** | Corpus composition | **all 162 clips are one match** (`arsenal_mancity_20250925`). 97,625 frames = **65.1 min**. Clip length min 50 / p25 225 / **median 430** / p75 785 / max 2,535 frames. | `machine_labels/_manifest.json` |
| **M8** | Per-class detector reliability | `player` median conf **0.919** (74.3% ≥0.5) · `coach` 0.925 · `referee` **0.335** (44.5% ≥0.5) · `in_play_ball` 0.222 · `person` n=50, median **0.123**, 0% ≥0.25 · `player_sub` n=11, median 0.113 | seg002, n=11,512 detections |

### What these imply

1. **M1 collapses two of the four sub-problems into one.** The clips were cut at
   shot boundaries by `split_broadcast_segments`. There is no such thing as
   "identity across a camera cut *within* a clip" — a cut *is* a clip boundary.
   So "across a cut" and "across clips" are the same problem, and "within a
   clip" is pure single-shot tracking over a median of 17 s.
2. **M4/M4b say plain IoU association should mostly work.** 96.4% of
   associations have an unambiguous winner. This is a **falsifiable prediction**:
   IoU + Hungarian (`LapTracker`, already written) should score respectably, and
   its failures should concentrate in the 6.3% crowded instances from M5. If
   that prediction fails, the model of the problem is wrong and this plan needs
   revisiting.
3. **M6 makes team assignment nearly free**, and — more valuably — makes it a
   cheap *constraint* on association (§3.6).
4. **M3 kills jersey OCR.** A 52×111 px player box puts the number at roughly
   10–18 px tall, below any OCR viability threshold. See §3.7.
5. **M7 means cross-match re-ID is currently unmeasurable.** There is exactly
   one match in `machine_labels/`.
6. **M8 means the `person` and `player_sub` classes are noise** (0% of instances
   above conf 0.25) and `referee` is weak. Filter accordingly.

---

## 1. Problem definition

"Player identification" is four different problems with an order-of-magnitude
spread in difficulty and value. Naming them separately is most of the work.

| ID | Sub-problem | What it means | Difficulty | Value now | Priority |
|---|---|---|---|---|---|
| **P0** | **Within-shot association** | Link `player` detections across consecutive frames inside one clip. Output: a `track_id` per detection. Because of M1 this is single-shot, cut-free tracking over a median 17 s. | **Low–medium.** M4 says 96% of frames are unambiguous; the hard 4% is crowding (M5). | **High** — unlocks per-shot velocity, possession duration, formation, and crucially *tracklet-level* auto-labelling (one human decision covers ~400 boxes). | **P0 — do first** |
| **P1** | **Team + role assignment** | Label each tracklet `home`/`away`/`referee`/`goalkeeper`. Not identity, but the cheapest useful identity-adjacent attribute. | **Very low.** M6: 1.0% ambiguous. | **High**, and it makes P0 *better* (§3.6). | **P0 — do alongside** |
| **P2** | **Cross-clip identity, same match** | The player who is track 7 in `seg002` is the same person as track 3 in `seg041`. Requires a persistent per-match identity (`p_home_04`). Equivalent to "identity across a camera cut" (M1). | **High.** Gaps of seconds to minutes, kit is near-identical within a team, jersey unreadable (M3), pose/scale change wildly. | Medium — needed for per-player match stats, not for most per-shot analysis. | **P1 — after P0 is measured** |
| **P3** | **Cross-match identity** | Same human being across different games, kits, seasons. | **Very high**, and **currently unmeasurable** (M7: one match in the corpus). | Low right now. | **Defer** |

### Why P0 first, concretely

- It is the **only one that is fully measurable with data that exists today**
  plus ~2 hours of human labelling (§4.6).
- The code is already written (`trackers/`) and has **never been run against
  ground truth** — `tests/tracking/` contains fixtures but no test file. So the
  cheapest possible next action is "measure what we already have", which is
  exactly the lesson VitTrack taught (constant 0.119 confidence, target held for
  5% of frames, assumed workable for a year).
- P0 is a **prerequisite** for P2: cross-clip re-ID matches *tracklets*, not
  boxes. If tracklets are impure, cross-clip matching is matching noise.
- P0 is where the auto-labelling leverage is. One tracklet decision propagates
  to hundreds of boxes.

### Broadcast-football specifics that make this hard

- **Near-identical kit within a team.** Appearance embeddings distinguish teams
  trivially (M6) and teammates barely at all. This is why appearance re-ID is
  much weaker here than on pedestrian benchmarks, and why DanceTrack (similar-
  appearance targets) is the right external analogue, not MOT17.
- **Heavy occlusion in the 6.3%** (M5) — and it is exactly the interesting 6.3%
  (challenges, corners, goalmouth scrambles).
- **Players entering and leaving frame constantly.** A broadcast wide shot holds
  ~15–20 of 22 players (M2); the rest are off-camera and return with new IDs.
- **Jersey numbers unreadable** at this box size (M3).
- **Camera pan/tilt/zoom** — not a cut, but it breaks constant-velocity motion
  models. This is what camera-motion compensation is for (§3.3).
- **Motion blur** at 25 fps on sprinting players.
- **Replays** are separate clips (M1), and a replay shows the *same* players from
  a *different* angle — an under-appreciated free source of hard positive pairs
  for P2, and exactly how SoccerNet-ReID defines its identities.

---

## 2. Measurement

### 2.1 Metric survey and honest trade-offs

| Metric | Measures | Failure mode / why it can mislead |
|---|---|---|
| **MOTA** (CLEAR-MOT, Bernardin & Stiefelhagen 2008) | `1 − Σ(FN+FP+IDSW)/Σ(GT)` | **Dominated by detection.** ID switches are a rounding error next to FP/FN, so a tracker that scrambles every identity can still post a high MOTA. Can go negative. **For us it would mostly re-measure the detector.** Report it, never optimise it. |
| **IDF1** | Identity F1 over a *global* track-to-track assignment | Directly measures the thing we care about (identity consistency). But it is a global bipartite matching, so a single long track absorbing a switch can shift the whole assignment; it can *decrease* as detection improves. Sensitive in crowded scenes — i.e. exactly our 6.3%. |
| **HOTA** (Luiten et al., IJCV 2020) | Geometric-mean-style balance of **DetA** × **AssA** | Designed precisely because MOTA over-weights detection and IDF1 over-weights association. **Primary secondary metric.** Its value: `AssA` isolates association from detection, so we can tell whether a bad score is the tracker's fault or RT-DETR's. |
| **IDSW / fragmentation / MT / ML** | Raw counts | Not normalised — meaningless without a denominator. Report as **IDSW per 1,000 player-frames**. |
| **rank-1 / CMC** (re-ID, P2) | Probability the top gallery hit is correct | **Gallery-size dependent.** Our within-match gallery is ~30 identities vs Market-1501's ~750 — rank-1 will look flatteringly high. Never quote it without the gallery size. |
| **mAP** (re-ID, P2) | Mean average precision over the full ranked list | Preferred over CMC because it credits *complete* retrieval, not just the first hit — and a player appears in many clips, so there are many true matches per query. **Primary P2 metric.** |

### 2.2 The metric I actually want as primary: tracklet purity

Standard MOT metrics were designed for surveillance, where fragmentation and
switching are roughly equally bad. **Our downstream use is auto-labelling, and
the costs are wildly asymmetric:**

- A **fragmented** track (one player, three IDs) is nearly harmless. You get
  three shorter, *correct* tracklets. Training data is still clean.
- A **switched** track (two players, one ID) is **poison**. It injects
  confidently-wrong identity labels into the training set, and the whole point
  of the tiering scheme is to never let that happen silently.

So the primary metric is:

> **Tracklet purity** = fraction of predicted tracklets whose non-`unknown`
> frames map to exactly one ground-truth identity.
>
> **Tracklet coverage** = mean fraction of each GT identity's frames captured by
> its single largest matching predicted tracklet.

Purity is the one to hold a bar on; coverage is the one to improve afterwards.
Report both — purity alone is trivially gamed by emitting one-frame tracklets,
which is why coverage is mandatory alongside it. HOTA/IDF1/MOTA are reported as
standard cross-comparable secondaries via **TrackEval**, which we should use
rather than writing a bespoke metric (`ball_eval/metrics.py` is bespoke by
necessity; MOT metrics are not, and TrackEval is the reference implementation).

### 2.3 The v1 benchmark — buildable from data that exists

**Name:** `player_eval` v1. Sits alongside `ball_eval/`, same shape
(`dataset.py` / `metrics.py` / `runner.py`), different GT.

| Property | Spec | Rationale |
|---|---|---|
| Clips | **10**, stratified | see selection rules below |
| Frames | ~4,300 (~2.9 min) | ~10 × median 430 (M7) |
| Player-frame instances | **~62,000** (est., at 14.5/frame conf≥0.5, M2) | |
| GT tracklets | **~250–350** (est.) | must be *reported*, not estimated, once labelled |
| GT identities | ~30 per clip (22 players + subs + refs) | |
| Format | MOT-Challenge CSV per clip (via an exporter from the canonical store) | so TrackEval runs unmodified |

**Clip selection rules — all four are mandatory:**

1. **Exclude contamination.** Any clip present in the v11 detection training
   split is disqualified. `astonvilla_seg080` and `bournemouth_1st_seg070` are
   known-contaminated (they score ~1.000 there — meaningless). This requires
   actually enumerating the v11 train split, not assuming.
2. **Stratify by length** across the M7 quartiles (min 50 / median 430 /
   max 2,535) — a tracker that is fine for 200 frames may drift over 2,500.
3. **Stratify by difficulty**, using the M5 crowding rate computed per clip from
   the JSONL alone (free, no video decode). Include at least 3 clips in the top
   crowding decile — otherwise the benchmark only measures the easy case and we
   reproduce the ball-mAP-on-44-instances error in a new place.
4. **Reserve a clean holdout.** Label 10 more clips later and never tune on
   them.

**Trustworthiness — the rule that replaces a missing literature answer.**
I searched for published guidance on how many frames/tracks give a stable
IDF1/HOTA and **found none** — this appears to be a genuine gap. So instead of
inventing a threshold, measure the uncertainty directly:

> Every reported number carries a **clip-level bootstrap 95% CI** (resample the
> 10 clips with replacement, 10,000 draws) and is quoted as
> `HOTA 0.NN [0.NN–0.NN], n=10 clips / N frames / N tracklets / N identities`.
> Tracker A is only claimed better than B if the CI on the **paired**
> per-clip difference excludes zero.

This is the direct, mechanical answer to "a metric without its denominator is
worthless": the denominator is in the string, and the CI tells you whether 10
clips was enough. If the CI is uselessly wide, the benchmark tells you to label
more clips rather than letting you publish a number you can't defend.

**Also mandatory:** double-label 2 of the 10 clips (same human, ≥3 days apart).
Intra-annotator agreement on those two clips is the **ceiling** — no tracker can
be scored above the level at which the labels themselves are reproducible.

### 2.4 What "good enough" means, numerically

External anchors (cited, not measured here):

- SoccerNet-Tracking baselines on 30 s broadcast clips: **ByteTrack with
  ground-truth detections = 71.50 HOTA / 94.57 MOTA**; **FairMOT-ft without GT
  detections = 57.88 HOTA / 83.57 MOTA**. The 2023 challenge winner reached
  **~69.5 HOTA**.
- SoccerNet-ReID: 2022 baseline **59.11% mAP / 48.41% rank-1**; 2023 challenge
  best **93.26% mAP**, on 340,993 thumbnails from 400 matches.

Targets (**these are my estimates, calibrated against the above, not measured**):

| Sub-problem | Metric | v1 target | Justification |
|---|---|---|---|
| **P0** | **Tracklet purity** | **≥ 0.95** | The auto-labelling bar. At 0.95 with ~300 tracklets, ~15 tracklets per benchmark are contaminated — few enough to catch by review, and each contaminated tracklet is caught *before* it enters training data. |
| **P0** | Tracklet coverage | ≥ 0.70 | Fragmentation is cheap (§2.2). Don't over-constrain. |
| **P0** | HOTA | **≥ 0.65** | Between SoccerNet's GT-detection ceiling (71.5) and its no-GT-detection baseline (57.9). Our shots are *shorter and cut-free* (M1, median 17 s vs 30 s) which helps; our detections are not GT, which hurts. 0.65 is the midpoint. **Estimate — revise once the first real number exists.** |
| **P0** | IDF1 | ≥ 0.75 | Consistent with HOTA 0.65 at typical DetA/AssA splits. |
| **P0** | IDSW | ≤ 1 per 1,000 player-frames | ~62 switches across the v1 benchmark. Directly comparable to the purity target. |
| **P1** | Team assignment accuracy | **≥ 0.98** | M6 measured 1.0% ambiguous on real crops. Anything below 0.98 means the pipeline is losing information the raw pixels already contain. |
| **P2** | mAP, within-match gallery | ≥ 0.70, **gallery size stated** (~30 identities) | Not comparable to SoccerNet's 93% — that gallery and identity definition are different. Quoting SoccerNet's number as our target would be exactly the contamination error in a new costume. |

**Caveat I want on the record:** the HOTA/IDF1 targets are the softest numbers in
this document. The purity and team-accuracy targets are grounded in measured
project data (M5, M6); the HOTA target is an interpolation between two external
benchmarks on different footage. Treat the first measured HOTA as calibration,
not as pass/fail.

---

## 3. Implementation options

The honest starting position: **`src/footy_track/trackers/` already contains a
working-looking `LapTracker` (IoU + `lap.lapjv` Hungarian, `max_age=30`,
`iou_threshold=0.3`, no motion model) and a working `UltralyticsTracker`
(`YOLO.track(persist=True)` with `bytetrack.yaml` / `botsort.yaml`).
`tests/tracking/` has fixtures and no tests. Nothing has ever been measured.**

### 3.1 Tracking-by-detection: IoU + Hungarian (`LapTracker`) — **measure first**

- **Cost:** zero. Written. Runs on CPU.
- **Prediction (M4/M4b):** should handle ~96% of associations. Its weakness is
  the missing motion model — a fast pan will drop IoU below threshold.
- **Verdict: this is milestone 1.** Not because it is best, but because it is
  free and until it is measured every other option is speculation.

### 3.2 ByteTrack — **measure second**

Associates high-confidence detections first, then recovers low-confidence ones
in a second pass. Given M8 (`player` median conf 0.919 but only 74.3% ≥0.5),
there is a **real low-confidence tail** for ByteTrack's second pass to recover.
Already wired. Cost: zero.

### 3.3 BoT-SORT with camera-motion compensation — **measure third, expected winner**

Adds a Kalman state fix, IoU–ReID fusion, and **CMC** (global motion estimation
via ECC / sparse optical flow) which subtracts camera pan/tilt/zoom before
motion prediction. Reported at **80.5 MOTA / 80.2 IDF1 / 65.0 HOTA on MOT17
test**; CMC alone is worth **~1.0–1.5 HOTA on MOT17**, attributed specifically
to camera movement.

**Broadcast football is a much more camera-moving domain than MOT17**, so I
expect CMC to be worth more here than there. This is the one place I'd predict
a real gain over §3.1 — and it is already available via `botsort.yaml`. Cost:
zero to try, some CPU (ECC is not free).

### 3.4 OC-SORT / Deep OC-SORT / BoostTrack — **only if §3.1–3.3 miss the bar**

OC-SORT handles non-linear motion (beats ByteTrack by >10 HOTA on DanceTrack);
Deep OC-SORT reaches ~64.9 HOTA on MOT17. Available through `boxmot`.
**Cost:** a new dependency + integration, maybe a day. **Verdict: hold.** Adding
three more trackers before having a benchmark is how you end up with five
unmeasured trackers instead of two.

### 3.5 Appearance embeddings / ReID backbones — **required for P2, probably not for P0**

- OSNet (lightweight, 512-D, 256×128 input) or CLIP-ReID, both via `boxmot` /
  `torchreid`. PRTReID (identity+team+role-aware) is the SoccerNet
  game-state-reconstruction baseline's choice.
- **Cost to embed the whole corpus:** ~2M player crops. OSNet is tiny; at an
  assumed ~1,000 crops/s on a rented GPU that is **~35 min ≈ $0.25** at
  vast.ai rates. *(Throughput figure is an estimate, not measured here.)* This
  is cheap enough that it is not a budget question — it is a "do we need it"
  question.
- **My expectation for P0: embeddings will not help much**, because teammates
  are near-identical (M6 separates *teams* at 1% error; it says nothing about
  separating two Arsenal players). Appearance re-ID's leverage on similar-
  appearance targets is the known weak spot — this is what DanceTrack exists to
  demonstrate.
- **For P2 it is unavoidable** — there is no geometric cue linking two clips.
  Use **SoccerNet-ReID pretrained weights (MIT licensed)**; do **not** train a
  backbone from scratch (§3.9).

### 3.6 Team clustering by kit colour — **build, and use it as a constraint**

M6 is the strongest measured result in this document: k=2 on mean torso HSV
(encoded as `cos h, sin h, s, v` to handle hue wraparound) separates the teams
with **1.0% ambiguity** on 1,253 real crops. The wider ecosystem does the same
thing with heavier machinery (`sn-gamestate`: PRTReID embeddings → K-means;
`roboflow/sports`: SigLIP → UMAP → K-means). **We do not need embeddings for
this** — raw torso hue is sufficient on this footage, at ~zero cost.

The non-obvious part, and the reason this is P0 and not a nice-to-have:

> **Forbid association across team clusters.** The classic ID switch is two
> players crossing; the *costly* version is an attacker and a defender crossing.
> A hard team constraint removes that entire failure class for the price of a
> hue histogram.

Implementation: assign each detection a team at detection time, aggregate to a
per-tracklet majority vote (robust to per-frame noise), and set the association
cost to `inf` across teams. Then handle the exceptions explicitly:
**goalkeepers and referees are their own clusters** (the k=3 hue-61° cluster of
255 crops is the likely referee/GK population — must be verified before this is
trusted).

### 3.7 Jersey-number OCR — **do NOT build**

- **M3 is decisive.** Median player box **52 × 111 px**; even at p90 it is
  84 × 149. The number occupies a fraction of the torso — call it **10–18 px
  tall**. That is below OCR viability, and this is the *source* resolution; the
  detector runs at `imgsz=1280`, which is smaller still.
- The literature's headline numbers do not transfer. Koshkina & Elder report
  **87.4% on soccer tracklets** — but a tracklet is scored as legible if **≥1
  frame** is legible, and the crops are higher-resolution. A "Digit-Aware"
  method reports 82.74% on the SoccerNet jersey test set. These are
  *tracklet-level, best-frame* numbers on a curated dataset, not per-frame
  numbers on 52-px boxes.
- **But** — and this is the useful inversion — **a human can often read a number
  a model cannot**, by scrubbing several frames and using context. So jersey
  number belongs in the **labelling UI as an optional human annotation**
  (§4.3), populating the existing `track_meta.jersey_number` column, and **not**
  in the inference pipeline. That gives us the identity anchor without the OCR.
- **Revisit if and only if** a higher-resolution source (4K, or tactical-camera
  feed) appears. Then M3 should be re-measured, and this section rewritten.

### 3.8 Pose / gait cues — **do NOT build**

Requires a pose model per crop (GPU cost per frame), and gait recognition needs
long clean sequences of an unoccluded subject — which a 17 s broadcast shot with
36.5% crowded frames does not provide. No evidence it helps in this domain.
Revisit never, absent a specific measured failure that only gait explains.

### 3.9 Things I would explicitly NOT build, and why

| Not building | Why not |
|---|---|
| **Jersey OCR in the pipeline** | M3: digits ~10–18 px. Put it in the labelling UI instead (§3.7). |
| **A bespoke ReID backbone trained from scratch** | We have zero identity labels, and one match ⇒ ~30 identities (M7). ReID fine-tuning wants hundreds. SoccerNet-ReID weights are MIT-licensed and free. |
| **End-to-end transformer trackers (MOTR / MOTRv2)** | Won SoccerNet 2023 at ~69.5 HOTA, but they are GPU-hungry to train and need labelled tracking data we do not have. Wrong end of the cost curve for a no-GPU laptop and a $3.60 training budget. |
| **A custom MOT metric** | TrackEval is the reference implementation of HOTA/CLEAR/Identity. Writing our own invites a metric bug that flatters us — the exact class of error this project has already paid for. |
| **Cross-match re-ID (P3)** | Unmeasurable today (M7). Building it would mean shipping something whose success cannot be evaluated. |
| **Pose / gait** | §3.8. |
| **A second labelling front-end** | `labeller_review_2026-06.md` already found two divergent front-ends (Streamlit `app.py` + the FastAPI/Konva web UI). Extend `web/review.html`; do not add a third. |

### 3.10 Recommended path

```
        ┌──────────────────── P0: within-shot (single camera shot, M1) ────────────────────┐
        │  RT-DETR dets (exist, 2.77M)                                                      │
        │        │                                                                          │
        │        ├──▶ torso-hue team cluster (M6, ~free) ──┐                                │
        │        │                                          ▼                               │
        │        └──▶ IoU/Hungarian or BoT-SORT+CMC ── team-constrained association         │
        │                          │                                                        │
        │                          ▼   track_id  →  detection.track_id (column exists)      │
        └───────────────────────────────────────────────────────────────────────────────────┘
                                   │  tracklets
                                   ▼
        ┌──────────────── P2: cross-clip (same match) — ONLY after P0 measured ─────────────┐
        │  tracklet appearance embedding (OSNet / SoccerNet-ReID, MIT)                       │
        │        + team constraint + temporal exclusion (same clip ⇒ different players)      │
        │        + human-supplied jersey number where legible                                │
        │        → player_id, reid_parent_track_id (columns exist)                           │
        └───────────────────────────────────────────────────────────────────────────────────┘
```

**Exploiting the 2.77M existing detections.** They are the substrate for
everything above and cost nothing to reuse — but note what they are *not*: they
are TIER 3, conf≥0.10, unreviewed, and single-match. Three concrete uses:

1. **Free difficulty statistics.** M2/M4/M5/M8 need only the JSONL — no video
   decode. Running them across all 162 clips is minutes of laptop CPU and gives
   a per-clip crowding score to drive benchmark stratification (§2.3) and active
   learning (§4.5). **Do this first; it is the cheapest useful thing in the
   document.**
2. **Tracker input.** All trackers here are training-free, so 2.77M detections
   become ~65 min of tracklets for the price of CPU.
3. **Embedding corpus** for P2 (~2M player crops, ~$0.25 of GPU, §3.5).

### 3.11 Milestone 1 — small enough to validate quickly

> **Run `LapTracker` and `UltralyticsTracker`(botsort) over 10 selected clips,
> label those 10 clips (§4), and report purity / coverage / HOTA / IDF1 / IDSW
> with clip-level bootstrap CIs.**

No new algorithm. No GPU. No training labels. Estimated ~2 human hours of
labelling (§4.6) plus a day of harness work. The deliverable is a number, and
the number decides everything after it.

---

## 4. Label collection

This is the crux: **there are zero player identity labels today**, and the
existing `ball_gt_marks/` (39 files) are ball-only.

### 4.1 The core principle: humans review tracklets, never boxes

A median clip has 430 frames × ~15 players ≈ **6,500 player boxes** but only
**~30 tracklets**. Labelling boxes from scratch is a non-starter; reviewing
tracklets is a **~250× reduction** in human decisions. So:

> The human is never asked "where is this player?". The tracker answers that.
> The human is asked **"does this track change person?"** — a question a human
> answers in a fraction of a second and a tracker answers wrongly.

This mirrors the precedent already in the codebase: `/propagate` produces
machine boxes and the human corrects them; `Session.merge_propagated` makes
human GT absolutely authoritative (propagated output for a human-touched frame
is discarded entirely, not merged). Identity labelling should use the same
posture.

### 4.2 Interaction 1 — the tracklet strip (catches ID switches)

The primary screen. One predicted tracklet, rendered as a horizontal filmstrip
of player crops.

```
 track 7  ·  team HOME  ·  frames 102–389 (288 fr, 11.5 s)  ·  risk 0.31
 ┌────┬────┬────┬────┬────┬────┬────┬────┬────┬────┬────┬────┐
 │ f102│ f120│ f138│ f156│▓f171│▓f174│▓f177│ f195│ f240│ f290│ f340│ f389│
 └────┴────┴────┴────┴────┴────┴────┴────┴────┴────┴────┴────┘
                          └── dense sampling around a risky frame ──┘
 [Enter] accept   [S] split here   [U] mark unknown   [J] set jersey   [←/→] move
```

- **Adaptive sampling.** Uniform every ~18 frames, **plus dense sampling around
  risky frames**. Risk per frame = crowding (M5: IoU>0.2 with another box) +
  low detection confidence (M8) + **the tracker's own association margin**
  (M4b: margin < 0.05). This is the single biggest cost lever — it puts the
  human's eyes only where a switch could plausibly have happened.
- **One click per decision.** `S` splits the tracklet at the cursor. `Enter`
  accepts the whole thing.
- **Record what was actually seen.** Every crop the human displayed is appended
  to `checked_frames`. This is not bookkeeping pedantry — it is the entire basis
  of the tiering scheme (§4.4).

**Errors the human must catch, in priority order:**

1. **ID switch** (two people, one track ID) — poison (§2.2). Non-negotiable.
2. **Fragmentation** (one person, several IDs) — benign for training, but needed
   for a correct coverage/IDF1 number. Caught in interaction 2.
3. **Team error** — should be near-impossible given M6; if seen, it is a bug
   signal, not a labelling task.
4. **Bad boxes** — explicitly *not* this task. The existing review UI
   (`web/review.html`, `POST /review/correct`) owns box quality. Mixing the two
   makes both slower.

### 4.3 Interaction 2 — tracklet merge (builds identities, and the P2 labels)

After splits, present **ranked candidate pairs** and ask a binary question:

```
   Are these the same player?          [Y] same   [N] different   [?] can't tell
   ┌──────────┐   ┌──────────┐
   │ t007     │   │ t019     │      both HOME · seg002 f102-241 vs f260-389
   │ 4 crops  │   │ 4 crops  │      temporally disjoint · similarity 0.81
   └──────────┘   └──────────┘
```

- Candidates come from appearance similarity + team + **temporal exclusion**
  (two tracklets overlapping in time in the same clip **cannot** be the same
  player — a free, hard constraint that removes most of the candidate space).
- **Capture the `N` answers.** Rejected pairs are *negative* labels, they cost
  nothing extra, and they are exactly what a ReID metric and any future ReID
  fine-tuning need. Most annotation schemes throw these away.
- **Optional jersey number** (`J`): the human scrubs full-resolution frames and
  types a number if legible. This is the §3.7 inversion — human-in-the-loop
  jersey reading instead of OCR. Populates the existing
  `track_meta.jersey_number`. Also: **record the fraction of tracklets for which
  a human *could* read the number.** That number is independently interesting —
  it is the empirical ceiling for any future OCR work, measured on our footage
  rather than borrowed from a paper.

### 4.4 Provenance tiers for identity — what "human-checked" means for a track

This is the question the project rules pose and it does not have the same answer
as it does for a box.

> **For a box, `human_checked` is atomic: a human looked at one box, once.
> For a track it is not.** A tracklet is hundreds of association assertions
> spanning ~17 seconds, and the human physically cannot look at all of them —
> the whole point of the strip UI (§4.2) is that they look at ~12 of 288 frames.
> **A boolean `reviewed` flag on a tracklet is therefore a lie.**

The resolution: **tier is a property of a frame interval, not of a tracklet.**
A single tracklet is routinely mixed-tier.

| Tier | Identity meaning | Tag / storage |
|---|---|---|
| **TIER 1 — human-placed** | A **point assertion the human directly made**: a split point, a `same`/`different` pair judgement, an identity name, a jersey number, an `unknown` marking. | `tags` contain `labeller` |
| **TIER 2 — machine + human-checked** | A tracker-produced association interval **that lies inside `checked_frames`** — i.e. the human actually saw crops bracketing it and did not split. | source tag **and** `human_checked` |
| **TIER 3 — machine only** | Tracker association **outside** `checked_frames`. The overwhelming majority of frames in a "reviewed" tracklet. | source tag alone |

Rules that follow, and that must be enforced in code, not just documented:

- **Reviewing a tracklet does NOT promote the whole tracklet to TIER 2.** Only
  the checked intervals. This is the exact auto-promotion the project rules
  forbid, and it is very easy to do by accident here.
- Machine tracklets are written to a **separate directory** from human identity
  GT, matching the existing machine-labels/human-GT split.
- Precedent to mirror: `Session.merge_propagated` discards machine output for a
  human-touched frame entirely rather than merging it; `detections_enriched`
  arbitrates canonicality at **run-batch** level, not per row (commit `d5d27e2`).
  Identity arbitration should follow the same batch-level rule.

**The unidentifiable player.** A first-class value, not a missing one:

- `identity: "unknown"` — the human looked and genuinely could not tell (fully
  occluded, off-screen, 20 px tall, hopeless blur).
- Strictly distinct from `null` = not yet reviewed.
- `unknown` intervals become **ignore regions** in evaluation, exactly like MOT
  distractor zones: the tracker is neither rewarded nor punished there. Scoring
  a tracker as wrong on a frame no human can resolve measures nothing.
- **Report the unknown rate.** It is the **human ceiling**: if 12% of
  player-frames are human-unidentifiable, then a "purity ≥ 0.95" target needs
  restating relative to the resolvable 88%. Claiming a number above the ceiling
  is incoherent, and we would not otherwise notice.

### 4.5 Storage

**Hard prerequisite — stable detection IDs.** Machine-label JSONL rows today are
`{frame_index, bbox, center, tags, confidence, model_id, run_id, reviewed}` with
**no `detection_id`**; `feature_store/importers.py` synthesises one as the row
ordinal (`detection_id=i`). Identity labels keyed to a detection would therefore
**silently rebind if a JSONL is ever filtered, re-sorted, or regenerated.** The
review UI has the same latent bug in a different place (its box identity is the
positional `(clip, frame, box_index)`, so a delete shifts every later index).

> **Add an explicit, stable `detection_id` to the machine-label writer, assigned
> at inference time and monotone within a clip, before collecting any identity
> label.** Small change; everything downstream depends on it.

**Do not extend the existing `tags` list.** It is a fixed 2-element positional
convention `[label, provenance]` parsed by set-membership across the loaders and
the review UI. Adding a third element is a footgun for a schema that has already
caused one data-loss incident (`arsenal_example.jsonl` truncated 3,348 → 0 rows).

**Recommendation: a new per-clip sidecar**, `<clip_stem>.tracks.jsonl`, machine
tracks and human identity assertions in separate directories, with the feature
store as the queryable index (it is rebuildable from the sidecars, per the
existing "Parquet is source of truth, DuckDB is an index" rule).

```jsonl
{"type":"tracklet","tracklet_id":"t007","run_id":"lap_v1__20260830","clip":"..._seg002",
 "start_frame":102,"end_frame":389,"team":"home","identity":"p_home_04",
 "checked_frames":[[102,108],[168,180],[236,244],[386,389]],
 "checked_by":"george","checked_at":"2026-08-30T10:14:00Z"}
{"type":"identity","identity":"p_home_04","team":"home","jersey":17,"role":"player","tags":["labeller"]}
{"type":"split","run_id":"lap_v1__20260830","track_id":7,"at_frame":241,"tags":["labeller"]}
{"type":"pair","a":"t007","b":"t019","verdict":"same","tags":["labeller"]}
{"type":"pair","a":"t007","b":"t022","verdict":"different","tags":["labeller"]}
{"type":"unknown","tracklet_id":"t007","frames":[[300,318]],"tags":["labeller"]}
```

**Feature store: no schema change needed.** `detection.track_id` and
`track_meta.{team_id, jersey_number, player_id, reid_parent_track_id}` already
exist end-to-end (`TrackMeta` → table → `tracks_enriched` view →
`player_trajectory()`) and **nothing currently writes them**. The plumbing is
done; only the producer is missing. The one addition worth making is a
tier-carrying interval table so `checked_frames` is queryable rather than buried
in JSON — otherwise "show me every TIER 2 association" is not answerable, and
the tiering scheme becomes decorative.

### 4.6 How many labels, and how many human hours

**Estimates, clearly labelled as such** — the per-tracklet review time is the
load-bearing unknown and **must be measured on the first clip before committing
to the rest.**

Per median clip (430 frames, M7):

| Step | Volume | Est. rate | Est. time |
|---|---|---|---|
| Tracklet strip review | ~30 tracklets | ~15 s each | 7.5 min |
| Merge pass (binary pairs) | ~20 candidate pairs | ~5 s each | 1.7 min |
| Team / role confirmation | 2 clusters + GK/ref | — | 0.5 min |
| **Total** | | | **~10 min/clip** (range 8–15) |

| Deliverable | Clips | **Est. human hours** | Enables |
|---|---|---|---|
| **v1 benchmark (eval only)** | **10** | **~2 h** | Rank the trackers we already have. **P0.** |
| Clean holdout | +10 | ~2 h | Guard against tuning on the benchmark |
| v2 / per-condition breakdown | +20 | ~4 h | Tighter CIs; crowded-vs-open analysis |
| Cross-clip identity, one match | 20 clips linked | ~3 h | **P2** eval (gallery ~30 identities) |
| ReID *training* set | ≥5 matches needed | **not yet possible** | P3 — blocked on corpus (M7), not on labelling |

**The single most important number here: the first useful dataset is ~2 human
hours.** That is small enough that there is no excuse for continuing to guess.

**Zero training labels are needed for milestone 1** — every tracker in §3.1–3.3
is training-free. Labels are needed purely to *measure*. This is what makes the
first milestone cheap.

**Active learning, in priority order (all cheap):**

1. **Tracker disagreement.** Run `LapTracker` and BoT-SORT over all 162 clips
   and rank clips/frames by how much they disagree. Disagreement localises
   ambiguity without any model of ambiguity, and it needs no labels to compute.
2. **Association margin** (M4b) — 3.6% of associations have margin < 0.05. Those
   frames are where switches happen; sample them densely (§4.2).
3. **Crowding** (M5) — 6.3% of instances. Free from JSONL alone.
4. Only after v1: uncertainty from a trained model. Not before.

---

## 5. Risks and open questions

| # | Risk | Impact | **Cheapest thing to measure first** |
|---|---|---|---|
| **R1** | **Single-match corpus.** All 162 clips are `arsenal_mancity_20250925` (M7). P3 is unmeasurable and P2's gallery is one kit pair. | Blocks cross-match work entirely; makes P2 results non-generalising. | S3 already holds `arsenal_astonvilla/`, `arsenal_bournmouth_*/`, `arsenal_norwich/` source footage. **Run the existing detector over one more match** (~1 GPU-hour, well under $1) and re-check M6 on a second kit pair. |
| **R2** | **M2–M6 and M8 come from ONE clip** (seg002, 430 frames, one wide angle). Close-ups, replays and goalmouth scrambles may behave completely differently. This is structurally the same error as the 44-instance ball mAP. | Could invalidate the "IoU is easy" premise and the whole milestone-1 plan. | **Re-run M2, M4, M4b, M5, M8 across all 162 clips from the JSONL alone** — no video decode, minutes of laptop CPU. **Do this before anything else in this document.** M6 needs video, so sample ~20 clips. |
| **R3** | **Existing trackers may simply not work.** `trackers/` is written-but-unexercised scaffolding: `tests/tracking/` has fixtures and **no test file**; `TrackedDetection.is_interpolated` documents a Kalman filter that neither tracker has. | Milestone 1 stalls on debugging, not measuring. | Run both over one clip and eyeball the overlay video **before** building the benchmark harness. This is the direct VitTrack lesson — it was assumed workable for a year. |
| **R4** | **Detector recall drives everything.** If RT-DETR misses players, tracks fragment and no association algorithm recovers them. The autolabelling architecture review already found detection was **100%** of the ball failure. | A bad HOTA would be misattributed to the tracker. | HOTA's **DetA/AssA decomposition** separates these for free. Also: the §4 labels give player-detection recall as a by-product — measure it in the same pass. |
| **R5** | **Referee detections are weak** (M8: median conf 0.335) and `person`/`player_sub` are pure noise (0% above conf 0.25). Refs mis-typed as players pollute team clustering. | Team constraint (§3.6) misfires; a whole failure class reappears. | Verify whether the k=3 hue-61° cluster (n=255) is the referee/GK population — one afternoon on existing crops. |
| **R6** | **Purity target unreachable if the human-unknown rate is high.** | The ≥0.95 bar would be incoherent (§4.4). | Report the unknown rate from the **first labelled clip**; restate targets relative to the resolvable fraction. |
| **R7** | **Value risk: is P2 worth doing at all?** SoccerNet's own ReID dataset defines identity **only within an action**, not persistently across a match — strong evidence that persistent broadcast identity is both hard and often unnecessary. | We could spend weeks on cross-clip identity nobody consumes. | **Write down the actual downstream queries** and mark which genuinely need persistent identity vs. which are satisfied by per-shot tracklets + team + pitch position. Costs an hour and may delete P2. |
| **R8** | **Positional-index fragility.** The review UI keys boxes by ordinal `box_index`; machine JSONL has no `detection_id`. | Identity labels silently rebind to the wrong boxes — the worst possible failure, because it is invisible. | The §4.5 `detection_id` prerequisite. Non-negotiable and must land first. |
| **R9** | Labelling rate estimate (~15 s/tracklet) is a guess. If it is 60 s, the 2-hour figure becomes 8 hours. | Plan slips 4×. | Time the **first clip** and revise §4.6 before labelling the other nine. |

### Open questions

- **Does the team constraint actually reduce ID switches?** M6 says teams are
  separable; it does **not** say switches are predominantly cross-team.
  *Resolved by:* ablating the constraint on the v1 benchmark once it exists.
- **Is CMC worth its CPU cost here?** Expected yes (§3.3) but unmeasured on this
  footage. *Resolved by:* BoT-SORT with and without `gmc_method` on v1.
- **What is the right `track_buffer`?** Given the asymmetric costs (§2.2), a
  *short* buffer that fragments rather than switches may be optimal — the
  opposite of the usual MOT tuning instinct. *Resolved by:* sweeping it against
  purity rather than HOTA.
- **Do the last 1–4 frames of each clip (the next shot's leading frames, M1)
  need trimming?** They inject a handful of wrong-scene detections into every
  clip. *Resolved by:* a cheap shot-boundary check in the segmenter.
- **Should identity be per-match or per-clip?** This doc assumes per-match
  identities (`p_home_04`) with per-clip tracklets. Not obviously right if P2 is
  deprioritised by R7.

---

## 6. Plan of record

| # | Priority | Action | Cost | Gate |
|---|---|---|---|---|
| **T0** | **P0** | Re-run M2/M4/M4b/M5/M8 across **all 162 clips** from JSONL only; sample ~20 clips for M6. Publish per-clip crowding scores. | Minutes of laptop CPU | R2. **Nothing else starts until this confirms the single-clip numbers.** |
| **T1** | **P0** | Add stable `detection_id` to the machine-label writer and importer. | Small | R8; prerequisite for all labelling |
| **T2** | **P0** | Smoke-run `LapTracker` + `UltralyticsTracker(botsort)` on one clip; watch the overlay video. | Hours | R3 |
| **T3** | **P0** | Build `player_eval/` (dataset + MOT-CSV exporter + TrackEval wiring + purity/coverage + clip-level bootstrap CIs). Mirror `ball_eval/`'s shape. | ~1 day | — |
| **T4** | **P0** | Tracklet-strip + merge UI in `web/review.html` (**not** a new front-end). | ~2 days | — |
| **T5** | **P0** | Select 10 clips (contamination-excluded, length- and crowding-stratified); label them. **Time the first clip and revise estimates.** | **~2 human hours** | R9 |
| **T6** | **P0** | Report purity / coverage / HOTA / IDF1 / IDSW with CIs and full denominators, for both trackers. **This is milestone 1.** | — | Decides everything after |
| **T7** | **P1** | Torso-hue team clustering + team-constrained association; ablate on the benchmark. | ~1 day | §3.6 |
| **T8** | **P1** | Detector run over a second match. | ~1 GPU-hour, <$1 | R1 |
| **T9** | **P2** | Only if T6 misses the bar: OC-SORT / Deep OC-SORT via `boxmot`; appearance embeddings. | Days | Gated on measured need |
| **T10** | **P2** | Cross-clip P2 work — but **only after R7 is answered**. | — | R7 |

**One-line summary.** The algorithms and the storage columns already exist and
have never been measured; the footage says within-clip tracking is easier than
feared (no cuts, median IoU 0.837), team assignment is nearly free (1% ambiguous)
and jersey OCR is impossible (52×111 px boxes). So: spend ~2 human hours building
a benchmark, measure what we already have, hold the bar on **tracklet purity**
rather than HOTA because auto-labelling punishes switches far more than
fragmentation, tier identity labels **per frame-interval** rather than per
tracklet because a human cannot look at 288 frames, and build nothing else until
a number says to.

---

## Appendix A — reproducing the §0 measurements

```bash
cd /home/george/footy-track
aws s3 cp s3://georgebarnett-general-purpose/footy_data/machine_labels/_manifest.json /tmp/
aws s3 cp s3://georgebarnett-general-purpose/footy_data/machine_labels/arsenal_mancity_20250925_seg002.jsonl /tmp/
aws s3 cp s3://georgebarnett-general-purpose/footy_data/clips/arsenal_mancity_20250925_seg002.mp4 /tmp/
# M2/M3/M4/M5/M8 need only the JSONL; M6 decodes the mp4; M1 decodes 5 clips.
# M1:  per-frame 64-bin greyscale histogram L1 distance, cut threshold 0.25
# M6:  torso crop = bbox[0.25..0.75]w x [0.20..0.50]h, mean HSV,
#      features (cos h, sin h, s/255, v/255), k-means k=2
```

`M7` comes straight from `_manifest.json`. `.venv/bin/python` has numpy 2.3.5 and
cv2 5.0.0; there is **no scipy** in the venv.

## Appendix B — sources

**Metrics**
- HOTA — Luiten et al., IJCV 2020 — https://arxiv.org/abs/2009.07736 · author's explainer https://jonathonluiten.medium.com/how-to-evaluate-tracking-with-the-hota-metrics-754036d183e1
- CLEAR MOT / MOTA — Bernardin & Stiefelhagen 2008
- TrackEval (reference implementation of HOTA / CLEAR / Identity) — https://github.com/JonathonLuiten/TrackEval
- re-ID mAP vs CMC — Zheng et al., Market-1501, ICCV 2015 — https://www.cv-foundation.org/openaccess/content_iccv_2015/papers/Zheng_Scalable_Person_Re-Identification_ICCV_2015_paper.pdf
- *No published guidance was found on the minimum annotation size for a stable IDF1/HOTA — hence the bootstrap-CI rule in §2.3.*

**SoccerNet**
- SoccerNet-Tracking (225,375 frames, 5,009 tracklets, 12 matches, 1080p/25fps; ByteTrack w/ GT dets 71.50 HOTA, FairMOT-ft 57.88 HOTA) — https://arxiv.org/abs/2204.06918
- SoccerNet 2023 tracking challenge (winner ~69.5 HOTA) — https://arxiv.org/pdf/2308.16651
- SoccerNet-ReID (340,993 thumbnails, 400 matches, MIT licence; **identities valid only within an action**) — https://github.com/SoccerNet/sn-reid
- SoccerNet 2023 ReID challenge (best 93.26% mAP; 2022 baseline 59.11% mAP / 48.41% rank-1) — https://arxiv.org/pdf/2309.06006
- SoccerNet Game State Reconstruction / `sn-gamestate` (PRTReID + K-means team assignment, GS-HOTA) — https://arxiv.org/pdf/2404.11335 · https://github.com/SoccerNet/sn-gamestate

**Trackers**
- BoT-SORT (80.5 MOTA / 80.2 IDF1 / 65.0 HOTA on MOT17; CMC worth ~1.0–1.5 HOTA) — https://arxiv.org/pdf/2206.14651 · https://github.com/NirAharon/BoT-SORT
- OC-SORT / Deep OC-SORT (64.9 HOTA MOT17; >10 HOTA over ByteTrack on DanceTrack) — https://arxiv.org/pdf/2302.11813 · https://github.com/GerardMaggiolino/Deep-OC-SORT
- `boxmot` (OSNet / CLIP-ReID / LightMBN plug-ins) — https://github.com/mikel-brostrom/boxmot

**Jersey numbers**
- Koshkina & Elder, CVPRW 2024 — 87.4% on soccer **tracklets** (tracklet legible if ≥1 frame legible) — https://arxiv.org/html/2405.13896 · https://github.com/mkoshkina/jersey-number-pipeline
- SoccerNet jersey-number challenge — https://github.com/SoccerNet/sn-jersey

**Team assignment**
- `roboflow/sports` — SigLIP → UMAP → K-means into 2 clusters — https://github.com/roboflow/sports

**Project context**
- [`player_tracking_format.md`](player_tracking_format.md) §5–6 · [`tracking.md`](tracking.md) · [`feature_store.md`](feature_store.md) · [`autolabelling_architecture_review.md`](autolabelling_architecture_review.md) · [`bidirectional_tracking_research.md`](bidirectional_tracking_research.md) · [`labeller_review_2026-06.md`](labeller_review_2026-06.md)
- Code: `src/footy_track/trackers/` · `src/footy_track/ball_eval/` (harness template) · `src/footy_track/feature_store/schema.py` (DDL) · `src/footy_track/labeller/web/review.html`
