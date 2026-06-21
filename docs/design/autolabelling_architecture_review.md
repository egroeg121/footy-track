# Architecture Review: GT Marking + Bake-off for Ball Autolabelling

**Bead:** ft-q3y · **Date:** 2026-06-21 · **Reviewer:** polecat jade
**Scope:** the proposed architecture for autolabelling ball positions in broadcast
football video — sparse human GT marking, tracker seeding, the 4-method bake-off,
data-safety strategy, and the path from bake-off winner to training data.

> **Headline.** The architecture is **sound in shape and wrong in sequencing**.
> Sparse human marking + tracker seeding is the right way to *measure* trackers,
> but the project is currently optimising the wrong stage: the empirical evidence
> ([`ball_tracker_failure_analysis.md`](../ball_tracker_failure_analysis.md),
> [`bakeoff_results.json`](../bakeoff_results.json)) says **detection — not
> tracking — is 100% of the failure**, and the one bake-off method actually run
> (ROI-YOLO) scores **0.2% recall**. Before a bake-off can discriminate between
> trackers, the seeds those trackers re-acquire from have to exist. The
> single highest-leverage action is not "run more trackers" — it is **close the
> labeller export gap (B1) and produce the first batch of human GT**, because
> that GT is simultaneously (a) the bake-off's measuring stick and (b) the
> training data that fixes detection, which is the actual bottleneck.

---

## 1. What's being proposed (as I understand it)

1. A human marks the ball bbox on a **sparse** set of frames (~every 5) via the
   labeller UI.
2. Each tracker is **seeded** from those GT marks (warm-start, not cold-start)
   and tracks forward until it loses the ball.
3. A **bake-off** compares four methods — A=SOT VitTrack, B=SAM2 ROI crop,
   C=ROI-YOLO on a Kalman-predicted crop, D=SAM3 (control).
4. Metrics deliberately favour the *seeded-tracking* regime — frames held before
   drift, centre-distance while held, reseed count — and explicitly **reject
   global recall**, on the grounds that recall is dominated by cold-start
   acquisition failure rather than tracking quality.
5. GT and results are flushed incrementally as JSONL to iCloud for safety.
6. The winning method then produces training data at scale.

This review takes each of the bead's five questions in turn (§3–§7), after first
establishing what the code and data actually show today (§2), because the answers
hinge on it.

---

## 2. Ground truth: what the tree and the data actually say

Three facts from the codebase reframe every question the bead asks.

**Fact 1 — detection is the entire problem, tracking is not.**
[`ball_tracker_failure_analysis.md`](../ball_tracker_failure_analysis.md)
analysed 500 frames across two broadcast clips:

| Cause of missed ball | Count | Share of misses |
|---|---|---|
| Detection failure (YOLO conf = 0) | 392 | **100%** |
| Tracker failure (conf ≥ 0.3 but dropped) | 0 | **0%** |

Overall detection rate was **21.6%** (36% on one clip, 7% on the other). Every
single missed frame was a detection failure; the Hungarian tracker dropped
*nothing* it was given. The ball is small, distant, motion-blurred, and often
occluded — the detector simply doesn't see it.

**Fact 2 — the bake-off has barely run, and what ran scored ~0.**
[`bakeoff_results.json`](../bakeoff_results.json) (2026-06-21) contains exactly
**one** method, `roi-yolo-trained` (Method C), over 5 clips:
`center_within_radius_pct` 20% (driven by one trivial 1-frame clip),
**`mean_recall_pct` 0.2%**, `total_failures` 437. Methods A (VitTrack) and
C (ROI-YOLO) are implemented in `src/footy_track/ball_trackers/`; **B (SAM2) and
D (SAM3) live only on un-merged feature branches** (ft-xps, ft-76b) and have
never been scored on this dataset. So today's "bake-off" is a single competitor
posting a near-zero number — there is no comparison to make yet.

**Fact 3 — the labeller cannot yet save GT.**
[`labeller_review_2026-06.md`](./labeller_review_2026-06.md) §3 B1: the
go-forward web labeller (`server.py` + `web/`) has **no export endpoint**. A user
can mark and propagate in the browser and then lose everything;
`export_frames_json` is wired only into the retiring Streamlit `app.py`. **No
human GT can currently be persisted from the intended UI.** This is the literal
precondition for both seeding trackers and retraining the detector, and it is not
met.

These three facts mean the project is, right now, trying to benchmark the
*non-broken* stage (tracking) using seeds it *cannot yet produce* (GT export),
to escape a problem that lives in a *third* stage (detection). The architecture
below is good; the ordering needs correcting.

---

## 3. Q1 — Is sparse human marking + tracker seeding the right architecture?

**Yes for evaluation. Partly, with a caveat, for production labelling.**

### Where it's right

Sparse marking + seeded forward-tracking is the **correct evaluation harness**
and is well-aligned with the rest of the system:

- It isolates the variable that matters for autolabelling-at-scale — *how long a
  tracker holds the ball once correctly initialised* — from the variable the
  human is there to cover (acquisition). That's a clean factorisation.
- It matches the existing motion-guided design
  ([`sam3_ball_labelling_gpu.md`](./sam3_ball_labelling_gpu.md) §2): the plan of
  record is already "Kalman-predict → tight ROI crop → cheap runner → full-frame
  re-acquire on miss". A seeded tracker measured by "frames held before drift" is
  exactly the per-step loop that doc describes, measured the right way.
- The `BallTracker` protocol (`ball_eval/interface.py`) with `track(prev_bbox,
  frame)` + `reset()`, and the runner that feeds GT back as `prev_bbox` on
  re-seed, is a faithful and minimal implementation of this idea.

### The caveat — "every ~5 frames" is an assertion, not a measurement

Sparse marking only works at scale if the **reseed interval the trackers can
sustain** is large enough that human marking is a small fraction of the work. The
bead picks "~every 5 frames" a priori. At 5-frame sparsity on 25fps broadcast,
the human marks 5 balls/second of footage — that is *not* sparse; that is nearly
manual labelling. The whole economic case for this architecture rests on the
**autolabel ratio**: frames-auto-produced ÷ frames-hand-marked. The bake-off must
*output that ratio per method*, and the target interval should be **derived from
the winning tracker's measured drift length, not assumed**. If the best tracker
only holds the ball for ~5 frames on hard clips, sparse marking has not actually
bought scale — and that's a finding the bake-off must be able to surface, not
hide.

### The deeper architectural point

Seeded tracking can produce GT *between* human marks, but every autolabelled
frame inherits the tracker's drift. For **training-data** purposes (the end goal,
§6) that's fine — a slightly loose bbox on a 24px ball is still a useful positive,
and small label noise is tolerable for detector training. For **evaluation
ground truth** it is not — you cannot measure a tracker against labels produced
by a tracker. Keep the two uses distinct: human `PROV_LABELLER` boxes are the
*only* GT the bake-off scores against; tracker-propagated boxes are *training
fodder*, never the measuring stick. The labeller's provenance model
(`PROV_LABELLER` survives re-propagation, `server.py:106-113`) already encodes
this distinction correctly — the bake-off harness must respect it by scoring only
against `PROV_LABELLER` frames.

**Verdict:** right architecture for the eval harness; sound for production
labelling *only if* the reseed interval is an output, not an input, and human GT
is kept strictly separate from tracker output.

---

## 4. Q2 — Are the 4 methods well-chosen? What's missing?

The four methods are a reasonable spanning set across the design space, but the
set has one structural gap and one redundancy.

| Method | Family | Role | Status |
|---|---|---|---|
| A — SOT VitTrack | Single-object tracker (ONNX, template+search) | "track this box" specialist | ✅ implemented |
| B — SAM2 ROI crop | Promptable segmenter on crop | seg-quality on crop | ⚠️ branch only |
| C — ROI-YOLO on Kalman crop | Detector on predicted crop | re-detect-each-frame | ✅ implemented |
| D — SAM3 | Promptable segmenter, full-frame | control / quality ceiling | ⚠️ branch only |

### What's good

The set spans the three relevant families — **SOT trackers** (A), **detectors**
(C), **promptable segmenters** (B, D) — which is exactly the runner matrix
[`sam3_ball_labelling_gpu.md`](./sam3_ball_labelling_gpu.md) §5 calls for. Having
SAM3 as the explicit control/ceiling (D) is the right experimental hygiene:
SAM3 is the current incumbent and the quality bar everything else must beat at
acceptable speed.

### What's missing or off

1. **No "detector + temporal smoothing" baseline that *uses the existing
   trained YOLO directly*.** Method C is ROI-YOLO on a Kalman *crop*; there is no
   simplest-possible baseline of "run the existing full-frame YOLO every frame,
   NMS, link with the existing Hungarian tracker" — i.e. the *current production
   pipeline*. Without it, the bake-off can't tell you how much the fancy
   approaches actually buy over what already ships. Add a **Method 0 = current
   pipeline** as the floor, the way SAM3 is the ceiling.

2. **B and D are partly redundant and the cheap-SAM option is absent.** Both B
   (SAM2) and D (SAM3) are heavy promptable segmenters; the design doc explicitly
   wanted **SAM2.1-tiny / MobileSAM / EdgeTAM** as the *cheap* segmenter
   candidates (§4, §5), because the whole point is that full SAM is too slow.
   Testing two heavyweight SAMs while omitting the lightweight ones tests the
   wrong end of the cost curve. Swap one of B/D's slots (keep D as the
   ceiling/control; replace B with **SAM2.1-tiny on the crop**) so the matrix
   includes at least one segmenter that could plausibly be the *production*
   runner, not just a baseline.

3. **No explicit re-acquire method under test.** The architecture's safety net is
   full-frame re-acquire on miss (§2.3 of the GPU doc), and the *bake-off metric*
   is "reseed count". But re-acquire quality is itself a method choice
   (full-frame YOLO vs full-frame SAM3) and it determines whether a "lost" ball
   is recovered automatically or needs a human mark. The bake-off should measure
   each tracker **with and without** an automatic re-acquire pass, so the reseed
   count reflects *human* reseeds, not tracker-internal recoveries — otherwise
   "reseed count" conflates two very different costs.

4. **Speed is a first-class axis, not a footnote.** The pass-bar in the GPU doc
   is *interactive fps with occasional correction*. Two methods can hold the ball
   equally well while differing 10× in fps. The current `bakeoff_results.json`
   does report `fps` and `effective_resolution_px` — good — but the **decision
   rule must weight fps explicitly** (e.g. "best `frames_held` among methods
   ≥15fps on the cropped path"), or the bake-off will crown an unusable
   quality-king.

**Verdict:** keep A, C, D; add **Method 0 (current pipeline floor)**; replace one
heavyweight SAM with a **tiny SAM**; make **re-acquire** an explicit on/off
condition; and bake the **fps pass-bar into the decision rule**.

---

## 5. Q3 — Is incremental JSONL flush to iCloud the right data-safety strategy?

**Right instinct, adequate-for-now mechanism, wrong primary risk in focus.**

### What's good

- **Append-only JSONL with incremental flush** is the correct *shape* for
  in-progress human labelling: each marked frame is one self-contained line, a
  crash loses at most the current line, and the format matches the eval dataset
  sidecars (`eval_data/clips/<clip>.jsonl`, one `FrameLabel` per line) and the
  feature-store importer contract (`<stem>_frame_<idx>`). No serialisation
  ceremony, recoverable by hand, diff-friendly. Keep it.
- **iCloud** as the sync target is a pragmatic, zero-infra choice consistent with
  the rest of the project (model checkpoints already live under
  `~/Library/Mobile Documents/com~apple~CloudDocs/footy_data/`). For solo
  morning marking it is fine.

### The caveats

1. **The flush is moot until B1 is fixed.** The web labeller has *no export path
   at all* right now ([`labeller_review_2026-06.md`](./labeller_review_2026-06.md)
   B1). "Incremental JSONL flush to iCloud" describes a strategy for a write path
   that **does not exist in the go-forward UI**. The *first* data-safety task is
   not tuning the flush cadence — it is shipping the `/session/export` endpoint
   (R1) so there is anything to flush. Until then the real data-safety posture is
   "everything is in a single browser tab's memory and one reload wipes it",
   which is the most dangerous possible state.

2. **iCloud is sync, not backup, and not atomic.** iCloud Drive does *eventual*
   sync and can conflict-rename files; a partially-flushed line synced mid-write,
   or two devices touching the same clip, produces silent corruption or
   `clip 2.jsonl` conflict copies. Mitigations: **write to a local file and
   `os.replace` (atomic rename) per flush**, never append-in-place to the iCloud
   path directly; flush whole lines only (never a partial JSON object); and treat
   iCloud as the *propagation* layer over a local source of truth, with periodic
   `git`/`bd` snapshots of completed clips as the actual backup. JSONL append +
   atomic-rename-on-flush gives crash safety without depending on iCloud
   semantics.

3. **GT integrity > GT durability.** The labeller review flags two *silent
   correctness* bugs that no flushing strategy protects against: positional
   label-shift when SAM3 drops a track (B5) and the seeding-mode edit-loss
   footgun (B4). A perfectly durable flush of *wrong* labels is worse than a lost
   session, because wrong GT poisons both the bake-off scores and the retrain set
   invisibly. Data safety here must include **validating what gets flushed**
   (e.g. refuse to emit a frame whose mask-count ≠ seed-count; only flush
   `PROV_LABELLER` boxes as GT).

**Verdict:** JSONL-incremental is the right format; harden it with
atomic-rename-per-flush over a local source of truth and treat iCloud as sync not
backup; but the export endpoint (B1) and the silent-corruption bugs (B4/B5) are
higher-priority data-safety work than flush cadence.

---

## 6. Q4 — After the bake-off, how do we use the winner to produce training data?

The path already exists in design and partly in code — the bake-off winner slots
into the **existing** motion-guided + Roboflow pipeline, not a new one.

### The pipeline (from [`sam3_ball_labelling_gpu.md`](./sam3_ball_labelling_gpu.md) §7–§8)

```
human marks sparse GT (PROV_LABELLER)            ← the bake-off measuring stick
   │
   ▼
winning tracker = BackgroundLabeller backend=<winner>   (ft-rwg seam, video_utils.py:407-410)
   │  Kalman-crop loop, re-acquire on miss, human corrects on anomaly pause
   ▼
human-corrected timeline  (PROV_LABELLER + accepted tracker boxes)
   │
   ▼
frames + COCO annotations → RoboflowObjectDetectionHandler.upload_images  (labelling.py, ft-0y4)
   │
   ▼
Roboflow dataset (footy-track-detection v4+)
   │
   ▼
retrain ball YOLO  (ft-n2o) → measure recall lift on the same eval clips
   │
   └──────────────► feeds back: a better detector is a better Method-C runner
                    AND a better re-acquire, raising the autolabel ratio next round
```

### Key points and gaps

1. **The winner is selected as a `BackgroundLabeller` `backend=`, not a new
   tool.** ft-rwg already specifies the single seam (`video_utils.py:407-410`)
   where the chosen runner replaces the SAM3 stream. The bake-off's only output to
   production is *which backend string to default*. This is the right
   minimal-surface integration — keep it.

2. **Training-data export is Roboflow, not the feature store.** The GPU doc §8 is
   explicit and correct: corrected labels → COCO → `RoboflowObjectDetectionHandler`
   → retrain (ft-0y4 → ft-n2o). The feature store (ft-4jr) is for *runtime
   tracking/event* data, not raw training images. Don't conflate them.

3. **The flywheel is the actual payoff.** The winning tracker labels frames the
   detector currently misses → those become YOLO training data → the retrained
   YOLO (Fact 1's 21.6% recall) climbs → a better YOLO is *simultaneously* a
   better Method-C runner and a better re-acquire → the next labelling round needs
   fewer human marks. **The bake-off is not the deliverable; the first turn of
   this flywheel is.** The success metric for the whole effort is **recall lift on
   the eval clips after one retrain**, not the bake-off table itself.

4. **Gap: the bake-off and the retrain set must use disjoint clips.** If the same
   clips that the winning tracker labels are also the clips the detector is
   evaluated on, recall lift is contaminated. Reserve a held-out eval set of
   human-only-GT clips that never enter the training pool.

**Verdict:** the post-bake-off path is already designed and mostly built
(ft-rwg/ft-0y4/ft-n2o). The winner integrates as a backend string; training data
flows through Roboflow; and the real success criterion is **detector recall lift
after the first retrain**, with a held-out eval set kept clean.

---

## 7. Q5 — Are there faster paths to labelled ball data?

**Yes — and given Fact 1, at least one of them should run *in parallel with*
the bake-off, not after it.** The bake-off optimises tracking; these paths attack
detection, which is the actual bottleneck. Ranked by expected leverage:

1. **Weak supervision from the existing YOLO + NMS, high-precision-only.** The
   detector already fires on 21.6% of frames, and on those frames the tracker
   drops *nothing* (Fact 1) — i.e. when it detects, it's usable. Harvest the
   **high-confidence** detections (e.g. conf ≥ 0.5, after NMS) directly as
   training positives with *zero human time*. The mechanism already exists
   (`yolo_seed_objects` + `_nms_filter`, `video_utils.py:743-808`). This won't
   fix the hard 78% — those are the frames it can't see — but it cheaply
   *reinforces* the easy cases and, more importantly, **mines the failure
   distribution**: the frames where confidence is just below threshold are the
   exact hard examples the detector needs. Surface low-conf-but-real frames to the
   human as priority marking targets.

2. **Lower the confidence threshold and let the human reject false positives.**
   The failure analysis explicitly recommends dropping below 0.3
   ([`ball_tracker_failure_analysis.md`](../ball_tracker_failure_analysis.md)):
   the detector is missing balls, so it's operating too conservatively. A lower
   threshold trades false positives (cheap for a human to reject in the labeller —
   one click) for recovered true positives (expensive to mark from scratch). This
   shifts human effort from *drawing* boxes to *accepting/rejecting* them, which
   is far faster.

3. **SAM3-propagate from sparse marks now, ignore the bake-off entirely for the
   first batch.** SAM3 (the incumbent, Method D) already tracks the ball from a
   single mark — that's the UX the whole project is built on. It's slow, but for
   *offline batch* labelling (not interactive) slowness is tolerable. Run SAM3
   propagation overnight on a handful of clips, have the human correct the drifts
   in the morning, and you have the **first Roboflow batch before the bake-off
   even concludes**. The bake-off then optimises the *interactive* loop; it isn't
   on the critical path to the first retrain.

4. **Exploit broadcast priors the current detector ignores.** The ball is
   small, roughly circular, high-contrast, and moves on smooth parabolic
   trajectories. Cheap classical signals — Hough-circle / blob proposals filtered
   by the Kalman trajectory prior, or temporal differencing for the moving ball
   against a near-static pitch — can *propose* candidates for the human to confirm
   in occlusion-free frames. Lower precision than a trained detector, but they
   fail on *different* frames than YOLO, so they fill in the 78% gap. Use as
   candidate-generators feeding the human, not as autolabels.

**The strategic point:** every one of these attacks **detection recall**, which
Fact 1 says is 100% of the problem. The bake-off attacks **tracking drift**,
which Fact 1 says is 0% of the problem *today*. Tracking only becomes the binding
constraint *after* detection recall is high enough that cold-start acquisition
stops dominating. So: **run weak-supervision harvest (path 1) + threshold-lower
(path 2) + an overnight SAM3 batch (path 3) immediately to get the first retrain
turn**, and let the bake-off run in parallel to optimise the *interactive* loop
for the rounds after that.

---

## 8. Recommendations & next steps

Ordered by leverage on the actual goal — *labelled ball data that lifts detector
recall* — not by where the current effort is pointed.

| # | Priority | Action | Why |
|---|---|---|---|
| N1 | **P0** | Ship the labeller export endpoint (R1 / B1 from the labeller review) | Nothing — bake-off seeds, GT, retrain set — exists without it. The flush strategy is moot until this lands. |
| N2 | **P0** | Run an **overnight SAM3 batch + morning human correction** on ~5 clips to produce the **first Roboflow batch**, in parallel with bake-off work | Gets the flywheel's first turn (detector retrain) onto the critical path; doesn't wait for the bake-off to conclude. |
| N3 | **P0** | Add **high-confidence YOLO harvest** (conf ≥ 0.5, NMS) as zero-human-cost training positives, and **lower the live threshold** so the human rejects FPs instead of drawing boxes | Attacks detection recall (Fact 1 = 100% of failure) for near-zero cost using code that already exists. |
| N4 | **P1** | Make the bake-off **output the autolabel ratio / measured reseed interval per method**, and **derive** the sparse-marking interval from it rather than assuming ~5 | The economic case for the whole architecture depends on this number; "~every 5" must be measured, not asserted (§3). |
| N5 | **P1** | Add **Method 0 = current production pipeline** (full-frame YOLO + Hungarian) as the bake-off floor; keep SAM3 (D) as the ceiling | Without the floor, the bake-off can't show what the new methods buy over what ships today (§4). |
| N6 | **P1** | Replace one heavyweight SAM (keep D as control) with a **tiny SAM (SAM2.1-tiny / MobileSAM)** on the crop; merge the B/D branches or cut them | Tests the cheap end of the cost curve — the end that could actually be the production runner (§4). |
| N7 | **P1** | Bake the **fps pass-bar (≥~15fps cropped) into the decision rule**; measure each tracker **with/without auto re-acquire** so "reseed count" = human reseeds only | Prevents crowning an unusable quality-king; disambiguates the headline metric (§4). |
| N8 | **P1** | Harden the flush: **atomic `os.replace` per whole line over a local source of truth**, iCloud as sync-not-backup; refuse to flush frames failing the mask-count==seed-count guard (B5) | Crash safety without depending on iCloud semantics; stops silent GT corruption (§5). |
| N9 | **P2** | Keep a **held-out human-GT eval set** disjoint from the retrain pool; report **detector recall lift after retrain** as the project's headline success metric | The flywheel's payoff is recall lift, not the bake-off table; contamination would hide it (§6). |
| N10 | **P2** | Fix the labeller GT-integrity bugs B2/B4/B5 before large-scale marking | Durable wrong labels are worse than lost sessions; they poison both bake-off and retrain (§5). |

### One-line summary

**Fix export (N1), harvest + batch the detector's training data now (N2–N3), and
let the bake-off run in parallel as an *optimisation of the interactive loop* —
not as the gate to the first retrain.** The architecture is right; it's pointed at
the second-most-important problem. Detection recall is the bottleneck, and the
fastest path to fixing it is producing GT today by the cheapest means available,
because that same GT is what the bake-off measures against tomorrow.

---

## Appendix: source grounding

- Detection-is-the-bottleneck data: [`ball_tracker_failure_analysis.md`](../ball_tracker_failure_analysis.md)
- Single-method bake-off result (~0% recall): [`bakeoff_results.json`](../bakeoff_results.json)
- Motion-guided architecture / runner matrix / Roboflow export path: [`sam3_ball_labelling_gpu.md`](./sam3_ball_labelling_gpu.md)
- Labeller export gap (B1), provenance model, GT-integrity bugs: [`labeller_review_2026-06.md`](./labeller_review_2026-06.md)
- Tracker implementations: `src/footy_track/ball_trackers/sot_vittrack.py` (A), `src/footy_track/ball_trackers/roi_yolo.py` (C); B/D on branches ft-xps / ft-76b
- Tracker interface + harness: `src/footy_track/ball_eval/interface.py`, `.../runner.py`, `.../metrics.py`
