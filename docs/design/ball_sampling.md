# Ball sample mining: evaluation panel and training miner

Status: proposal, 2026-11-18. It builds on `feature_store.md`, `calibration.md`, the autolabel design (§12, on `feat/autolabel-retrack`) and the labelling-jobs/hooks system (labeller README §19c).

## 0. Summary

There are two regimes. They use different populations, sampling rules and cards, and a different question to Ball Check.

| | **Evaluation panel** | **Training miner** |
|---|---|---|
| Question | "How good is model M on unseen matches, ± what?" | "Which frames most improve the next model per human second?" |
| Population | Main-camera frames of **heldout matches** only | Main-camera frames of **train matches** only (in-match eval clips excluded) |
| Selection | Fixed probability sample with logged inclusion probabilities π. It does not depend on the model being scored. | Adaptive, model-driven quotas (uncertain, missed, hard-negative, random, audit) |
| Card | **Blind**: no model box. "Tap the ball" or "no ball visible". | **Verify**: the proposed box is shown, with ball / not_ball / box_off / corrected / "ball elsewhere" (tap). |
| Labels are | Model-independent, so any future model can be scored on them | About one model's proposals, used for training and for calibrating auto-accept |
| Output | HT-weighted hit@10/25, precision, false-ball rate, with CIs, per match and pooled | Tier-1/2 labels, negatives, and audit data for the auto-accept threshold |

Rule zero: **a heldout match never feeds training.** That includes pseudo-labels, even though self-training on a match's own video is our biggest measured win (held-out hit@10 went from 0.49 to 0.70). Using it on heldout footage would be training on the test set.

## 1. Why (evidence that shaped this)

**From our own data:**
- Today's ball eval is a convenience sample: 101 frames from 4 Forest clips, labelled while reviewing detector output. Recall is optimistic (`docs/ball-tracking-eval.md`). The only designed sample is the Forest placement job (24 windows, about 10–20% done).
- Generalisation across matches is the binding problem. The detector scores 0.92 in-distribution, 0.67 on arsenal_example and 0.49 on held-out clips. Per-match numbers matter more than a pooled one.
- 37% of human ball frames have no detector candidate within 25 px. A regime that only verifies proposals can never measure or fix recall, so both regimes need "ball is elsewhere" and "no ball".
- Detector confidence is badly calibrated: Ball Check precision is 0.04 at conf 0.1–0.2, 0.42 at 0.5–0.6 and 0.96 at 0.7–0.8.
- Claude verify is calibrated very differently by band: 33/33 correct at ≥ 0.95, 28/43 at 0.8–0.95, 8/23 at 0.5–0.8.
- **The deployed autolabel queue auto-accepts at 0.8** (`~/code/jobs/footy_autolabel_queue.py:79`). That admits a band measured at 65% precision. The design target was 0.95.
- Human throughput is about 100–260 cards per labelling day, at a median of about 2 s per verify card (mean 4.6 s). The pool is about 1.8M broadcast frames in the 12 segmented matches, with 34 more matches coming. Labels are scarce; frames are not.

**From the literature** (snippet-level reads; the papers are linked in the research notes):
- *Unbiased eval with a chosen sample*: Horvitz–Thompson/LURE (active testing, Kossen 2021) and active statistical inference (Zrnic & Candès 2024; over 80% label savings in their settings). Both stay unbiased if π is logged and bounded away from 0. PPI++ adds a model-prediction control variate to a uniform sample. Small π inflates variance, so mix in a uniform floor.
- *Active learning for detection*: gains over random are small and unstable, and random is a strong baseline (2212.10836, 2403.14800). Diversity and redundancy removal reliably help (+3.4 mAP, 2307.08414). In video, select clips or windows, not single frames. Use temporal-coherence errors (flicker, detector/tracker disagreement) as the acquisition signal. At low budgets, typical/diverse beats uncertain (TypiClust).
- *Verification*: verifying a box is about 3× cheaper than drawing one (Open Images: 9 s vs 25.5 s). Humans over-accept proposals, and more so when rejecting requires entering a correction.
- *Pseudo-labels*: a single fixed confidence threshold skews the set toward easy, frequent cases. Use adaptive thresholds, and route hard cases to humans.
- *Auto-accept*: Learn-then-Test turns verified outcomes into a threshold with P(precision ≥ target) ≥ 1−δ. It needs calibration data that is **exchangeable** with what gets auto-accepted, which means random audits within the band, not the human-review queue.

## 2. Evaluation panel

### 2.1 Population and unit
- **Population** U: every frame at stride 5 (10 fps at 50 fps), inside main-camera segments (`segments.json`) of heldout matches. That is 9 of 51 today, selected by sha1 % 4, and the rule is sticky.
- The population is fixed by the segmentation run, **not** by the current broadcast classifier's p_yes, so it does not move when the classifier is retrained. Frames that are not main-camera are out of scope for the ball metric.
- **Sampling unit:** a *window* of 5 frames, 0.2 s apart (about 1 s). Neighbouring frames make placement faster and give a little tracking signal. The cost is intra-window correlation, which the cluster bootstrap in §2.4 accounts for.

### 2.2 Card: blind placement (Ball Check "blind mode" = the `ball_place` job kind)
- No model box is shown. The labeller taps the ball centre, or marks **no ball visible** or **unsure**. The 3 s context clip stays.
- Rationale: blind labels are independent of the model, so the panel is **reusable for every future model and tracker**. They also have no automation bias and give exact pixel error, which hit@10 px needs and accept/reject cannot provide.
- Cost: we have not measured it. Assume about 4–6 s per frame against about 2 s to verify. That is an acceptable price for a set built once and reused.
- QA layer, about 8% of cards:
  - 5% **repeats** of earlier panel frames, in a different session. The spread between repeats gives the label-noise floor; if it is about 5 px, hit@10 is meaningful.
  - 3% **gold** frames from existing tier-1 GT.
  - Both are excluded from the estimates.

#### 2.2.1 Update 2026-11-19: detector suggestions with a random blind control (LAB-1445..1447)
- **Why.** Placing a ball from scratch is slow (the 4-6 s per frame assumed above, 5 panels about 630 cards), and George asked to be able to accept a detector box straight away or draw over it. Speed is the reason; the cost is that a shown box can pull the label towards the detector (automation bias), and the suggesting detector then looks better than it is.
- **What.** `ball_eval_panel suggest` (the model sits behind a small `Suggester` interface, so the pooled RT-DETR + motion + tracker top node can replace RT-DETR top-1) precomputes, per panel frame, the top-1 RT-DETR ball candidate (production weights, 1280 px, CPU about 2.1 s per frame, 120 frames per match, one process at a time) into `<FD>/ball_eval_panel/<version>/<match>.suggest.<detector_sha8>.json`. Frames with no candidate get `null`: **"no ball" is never suggested**, George must look for himself. This is eval-only inference on heldout footage; nothing is trained, mined or cached into a training pool, so rule zero still holds.
- **Page.** The suggestion is a dashed yellow overlay (not a box, never saved). `Y` / button accepts it: it is saved as an ordinary human box `[in_play_ball, labeller, accepted_suggestion, suggest_det:<sha8>]`. Drawing a box replaces it. Accepted boxes are **ordinary tier-1 labels everywhere** (scoring, ingest, export, job done rule); the marker is audit only and never excludes or down-weights a label.
- **Bias control.** A random 25% of each panel's windows (`sampling.suggest = "blind"`, seeded by `sha1(panel_version + match)`, reproducible from the panel file) and all QA cards (repeats and gold) never show a suggestion. Because the blind subset is random, blind-vs-shown labels are comparable: `panel_score` reports, per match and pooled, the share of labels that were accepted suggestions and the model's hit@r / recall / false-ball rate (HT estimate) on blind vs shown with the difference and its CI. A big gap means the suggestions move the labels. `score --blind-only` scores the suggesting detector on blind windows only (the model most favoured by its own suggestions).
- **Labels placed before suggestions existed** (George's first Forest cards) are unsuggested: the suggest meta file records them and the scorer counts them as blind-equivalent.
- **Live jobs** keep their items. The flag is also derived from the panel file at request time (same seeded rule), so no job mutation is needed; new panels get it in `sampling.suggest`.

### 2.3 Design: stratified two-stage sample with logged π
- **Strata = heldout match × difficulty band.** Allocation is **equal per match**, because per-match CIs are the deliverable.
- Difficulty bands come from a **reference model frozen at panel creation**: the current detector plus tracker on the window, using the existing easy / gappy / contested / low_evidence buckets. Hard bands are oversampled toward Neyman allocation, n_h ∝ N_h·√(p_h(1−p_h)), with p_h from the Forest pilot.
- **Bound the weights:** every band gets at least 30% of its proportional share, so max/min weight stays ≤ about 10. Frames are re-weighted by 1/π, so oversampling never biases the estimate. Bands only buy variance reduction, so a stale reference model costs efficiency, not correctness.
- **v1 can be simpler:** uniform windows within each match, with match as the only stratum. This needs no detector run on heldout matches. Add difficulty bands in v2, once v1 gives per-band pilot rates.
- Every card stores `sampling = {regime:"eval", panel:"ball-eval-v1", match, stratum, window_id, pi_window, pi_frame, seed}`, plus the sampler code sha.

### 2.4 Estimators
- **Per-frame outcomes** for model M (top-1 ball after tracker, or detector top-1 for detector eval):
  - `hit` (prediction within r px of the human centre)
  - `off` (prediction > r px away)
  - `miss` (ball present, no prediction)
  - `false_ball` (no ball, prediction)
  - `true_empty`
- **Metrics:**
  - recall@r = hits / ball-present frames
  - precision@r = hits / predicted frames
  - false-ball rate = false_ball / no-ball frames
  - All are reported at r = 10 and 25 px.
- Each metric is a ratio of Horvitz–Thompson totals. Its CI comes from a **stratified cluster bootstrap**, resampling windows within strata, 2,000 draws.
- **Reporting:** per match, plus pooled as both **match-averaged** (headline: "a typical new match") and frame-weighted.
- **Later (v3):** a PPI++ control variate. Use the auto-labeller's verdicts on every population frame as f, so θ̂ = mean_U(f) + HT-mean(Y − f). This cuts variance as far as f tracks Y, and it stays unbiased if f is poor.

### 2.5 Size and budget
- Recall around 0.7 with a design effect of about 1.5 needs about 120 frames per match for ±10 pp per match. Pooled over 9 matches that gives about ±3–4 pp.
- **Panel v1 = 9 × 24 windows × 5 frames ≈ 1,080 frames.** At about 5 s each that is about 1.5 h, or 4–6 sessions. Start with 4 matches (about 480 frames) to get a usable number within a week.
- The Forest "ball placement r1" job (600 frames, stratified, 10–20% done) becomes the Forest stratum of v1, re-expressed with its π.

### 2.6 Hygiene
- The panel is **versioned and frozen**. A new heldout match adds a new stratum and a new panel version; scores are always quoted with the panel version (cohort reporting).
- **Lockbox:** keep 2 heldout matches out of day-to-day model selection. Score them only on promotion, to catch overfitting to the panel from repeated reuse.
- In-match eval clips (1 in 5 of train-match clips) remain a *secondary* in-distribution number, using the same card and estimator.
- Fix known contamination first: `bournemouth_1st_seg070` is in Roboflow v11 training and hashes to in-match eval.

## 3. Training miner

### 3.1 Pipeline per train match (machine-only before any human)
1. Run the detector (conf ≥ 0.1) plus the bidirectional tracker at stride 5 over main-camera segments. This is a GPU job via Prefect. Today it covers only Man City and Forest, so the 12 newly segmented matches need it.
2. Build **track segments**: runs of frames the tracker follows. The candidate unit is the track segment, or a 2 s window of it, not a frame. This is how video redundancy is removed.
3. Compute signals per segment:
   - detector conf (mean, min)
   - detector vs tracker disagreement
   - forward vs backward tracker disagreement
   - flicker (detections appearing and disappearing)
   - trajectory plausibility (acceleration and jump limits in image space, homography-normalised where PTZ exists)
   - static-object flag (near hoardings, i.e. spare balls)
   - gap frames (tracker interpolates with no detection)
   - appearance embedding (72-d ball descriptor) for diversity
4. Run Claude verify on candidates, as the autolabel loop does now. Its confidence c becomes one more signal.

### 3.2 Buckets and per-round quotas

The quotas below are for a 300-card round, which is about one labelling day.

| Bucket | What | Card | Share | Selection within bucket |
|---|---|---|---|---|
| **U: uncertain** | Signals disagree, c in 0.5–0.95, or det conf 0.3–0.7 | verify | 35% | Rank by uncertainty, then **one card per cluster** (k-means on embedding × match × 30 s time bucket) and at most 1 card per 2 s window. Across rounds: no card within +-3 s of any earlier card of the same clip (`exclude_s`), and at most 3 U/M/N/R cards per clip per round (`max_per_clip`); each round's seed derives from its key |
| **M: missed** | Tracker gaps, plausible trajectory with no detection, ball-in-air candidates | **verify-or-place** (tap if the ball is elsewhere, or "no ball") | 20% | Longest and most plausible gaps first; diversity as in U |
| **N: hard negatives** | High-conf detections off any plausible track, static near the hoardings, or on heads and feet | verify (expected not_ball) | 15% | Highest conf first, capped per location cluster so it isn't 30 cards of the same spare ball |
| **R: random** | Uniform over the train-match population | verify-or-place | 15% | Uniform. This is the cold-start and robustness floor, and also an unbiased audit of the whole pipeline on train data |
| **A: auto-accept audit** | Uniform random within the auto-accepted set, per score band | verify, **blind to c** | 10% | Uniform within band. This is what calibrates the threshold (§3.3) |
| **G: gold** | Known-wrong boxes: GT shifted 15–30 px, or known spare balls | verify | 5% | Measures over-acceptance. Excluded from training |

- **Match balancing:** cap each match at about 15% of a round. Prioritise matches with the fewest labels and the highest broadcast-OOD score. Generalisation across matches is the bottleneck, so breadth beats depth.
- **Labeller check:** if the planted-wrong (gold) cards are accepted more than 10% of the time, the round's verify labels are flagged. The fix is then UI (e.g. making "box_off" one tap), not the labeller.
- **Adaptive within a round:** a hook fires every 50 verdicts (§19c; the `ballcheck-rescore` flow named in the README doesn't exist yet). It refits the cheap pieces: the per-bucket hit rates, and the Platt fit of c → P(correct) as was done for broadcast. It then reorders the unanswered U and M cards. No retrain is needed mid-round.

### 3.3 Auto-accept threshold (Learn-then-Test)
- Score s = calibrated P(correct | c, tracker-consistent, det conf, plausibility).
- For each candidate threshold t on a grid, the A-bucket audits with s ≥ t give a binomial p-value for H₀: precision < 0.98. Fixed-sequence testing from high t downward finds the lowest t whose H₀ is rejected at δ = 0.1.
- Only **A-bucket** (uniform within band) cards are valid for this. U-bucket verdicts are not exchangeable.
- About 150 audits per band are needed before a band can open. Until then, **auto-accept stays at c ≥ 0.95.** I recommend setting the queue's `auto_conf` back from 0.8 to 0.95 now.
- Track audit precision **per match**. The existing circuit breaker (< 0.97) should trip per match, not just globally, because broadcaster shift is the likely failure.

### 3.4 Turning verdicts into training data
- **Positives:**
  - tier 1 (placed or corrected)
  - tier 2 (`ball` on a shown box)
  - tier 3 (auto-accepted at the LTT threshold), loss-weighted 0.5 and capped at 3× the human positives per match
- **Negatives:** a frame with a rejected FP enters as a background image *only if* the frame's ball status is known: an accepted ball elsewhere in it, or a "no ball" verdict. Otherwise an unlabelled true ball could be taught as background. `export_training_dataset.py` currently drops not_ball boxes and needs this case added.
- **Composition check per export:** % from each bucket, each match and each difficulty tag. Auto-accepted labels skew easy (ball on grass), so hard strata must come from U, M and N.
- **Retrain cadence:** a round ends at about 300 human cards plus the auto-accepts. Retrain when cumulative new labels reach about 1,000 or weekly, whichever comes first. Then re-score the pool and re-mine. Report every model on the eval panel (§2) at its fixed version.

## 4. Leakage rules (enforced in code, not by convention)
1. `split_of(match)` gates every sampler. The eval sampler accepts only heldout matches and the training miner only train matches.
2. The job API rejects a `ball_check` or `ball_place` job whose items mix splits or whose `sampling.regime` disagrees with the split.
3. In-match eval clips, plus ±10 s around them, are excluded from mining. Consecutive `_segNNN` clips are not contiguous; use footage-time overlap once T-33/T-49 land.
4. Heldout footage is never pseudo-labelled. A "self-train on the deployment match" mode, if wanted, gets its own lockbox eval.

## 5. What has to be built

**P0 (small, now):**
- Set `auto_conf` back to 0.95.
- Add a `sampling` metadata block to job items and persist it with each verdict.
- Make the README list the 5 Ball Check verdicts.

**P1: eval panel v1.**
- A `ball-eval-panel` sampler (jobs repo) that emits `ball_place` jobs on 4 heldout matches, uniform windows, with π logged.
- An HT + cluster-bootstrap estimator in `ball_eval`, with CIs on the progress page.
- Fold the Forest r1 job in.
- Dependency: heldout matches segmented (pipeline running now). No detector run is needed for v1.

**P2: training miner.**
- Merge tap-to-place, "ball elsewhere" and "no ball" into Ball Check on main. They are built on `feat/autolabel-retrack` (f92fdc8).
- A detector+tracker GPU flow for the new train matches.
- A bucket sampler with quotas and diversity.
- Gold injection.
- The A-bucket audit and LTT threshold.
- Export negatives.

**P3:**
- The `ballcheck-rescore` hook flow (in-round reordering).
- Eval difficulty bands (panel v2).
- PPI++ surrogate.
- OOD-driven match priority.

## 6. Scheduling: the two loops (Prefect)

**Loop 1, on ingestion (per match).** It runs after `match-pipeline` (jobs `docs/design/match_pipeline.md`) has segmented a match, and branches by `split_of(match)`:
- **heldout:** `ball-eval-panel(match)` (cpu, merlin) samples windows uniformly within main-camera segments, logs π, and submits a blind `ball_place` job. It needs no model output.
- **train:**
  - `ball-candidates(match, detector_sha)` (gpu; detectors at stride 5 plus SAM2 gap-fill; skip key is the detector sha)
  - `ball-motion` (desktop cpu)
  - `ball-pool-score` (merlin cpu): pool, then solver, then the scorer gives P(real); writes `ball_pool/<match>/pool.<scorer>.parquet`
  - Tier-3 auto-labels are the nodes with P ≥ the LTT threshold.

  These are added after the pooled-labeller experiment (`exp/ball-candidate-pool`) settles which sources stay.
- **Trigger:** an hourly, deduped sweep (`ball-eval-panel-sweep`, `ball-pool-sweep`) finds matches that are segmented but not yet done. It does not edit `match-pipeline`, so it works whichever orchestrator produced the segments.

**Loop 2, the recurring round.**

| Step | Trigger | Runs |
|---|---|---|
| Sample | Tick (every 15 min): create a round only if open training cards < 100 and today's cards < 600 | `ball-round`: miner quotas (§3.2) over the train pool, then a `ball_check` job, then `ballcheck-prepare` |
| Refit | Labeller hook, every 50 verdicts | Scorer calibration + LTT threshold refit; reorder unanswered U/M cards |
| Close | Hook `on_done` | Ingest verdicts, recompute the auto-accept set, nightly dataset export |
| Retrain | Weekly or ≥ 1,000 new labels | `ball-train` (gpu) |
| Evaluate | After train | Score the candidate on the fixed eval panel (no human); promote via the single promotion mechanism if it is better |
| Re-infer | On promotion | Backfill `ball-candidates` + pool for train matches at low gpu priority, behind ingestion |

## 7. Decisions needed
1. **Eval cards: blind placement** (reusable, unbiased, slower), versus showing the model box (faster, but tied to one model). Recommendation: blind.
2. **Headline metric: match-averaged** or frame-weighted? Recommendation: match-averaged, with both shown.
3. **Lockbox**: which 2 of the 9 heldout matches?
4. **Auto-accept target precision**: 0.98 at δ = 0.1, and the 0.8 → 0.95 rollback now.
