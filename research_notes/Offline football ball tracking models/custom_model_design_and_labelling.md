# Custom offline (non-causal) ball + player tracking model for football broadcast, and the data/labelling it needs

Scope: a from-scratch/custom design (not a literature survey; the sibling notes in this folder cover the literature: `sports_ball_trackers.md`, `promptable_vos.md`, `point_tracking_and_self_supervision.md`, `offline_trajectory_and_mot.md`) plus a concrete, prioritised labelling plan the user can act on this week.

What this relied on (read in full or in the relevant sections):
- KB (via `mcp__knowledge__search` "footy ball detection", "footy labeller", "footy data plane"): `~/knowledge/wiki/footy-ball-detection-findings.md`, `footy-labeller-gotchas.md`, `footy-data-plane-reconciliation.md`, `footy-claude-autolabel-loop.md`.
- Repo (`/home/george/footy-track`, branch `fix/detector-predict-kwargs`): `docs/design/bidirectional_tracking_research.md`, `docs/design/tracking-pipeline.md`, `docs/design/autolabelling_architecture_review.md`, `docs/design/calibration.md` (first 80 lines), `src/footy_track/labeller/README.md` (§1, §2, §11, §12), `src/footy_track/labeller/ball_check.py` (skim), `src/footy_track/labeller/constants.py`, `src/footy_track/feature_store/schema.py`, `src/footy_track/ball_eval/interface.py`.
- New measurements made for this note (read-only) on `/mnt/storage/footy_data/ball_gt_marks/*.jsonl` (48 sidecars) with a throwaway script at `/tmp/claude-1000/-home-george-footy-track/b7eb525d-653a-4c8a-a19e-54b2c306efff/scratchpad/gtstats.py`. The script may get deleted, so the method is described where each number is used. Tier filter = `labeller` or `human_checked` in tags (KB data-plane rule). Pixel numbers are at 1920x1080.

Conventions: **[measured]** = I computed it or a cited project doc measured it, with the denominator given. **[assumed]** = an estimate to verify. Every design choice sits under "Inferences" because it is my reasoning, not a sourced fact.

---

## Q1. Architecture options for a custom offline model, compared, and a recommendation with a staged build plan

### Takeaway
Build a **two-stage, candidate-then-sequence model (Option A)**, and build its non-learned skeleton first. **M0** takes the existing RT-DETR candidates (conf ≥ 0.1), adds global camera-motion compensation, and decodes them with a bidirectional HMM/Viterbi. It has explicit states (visible-candidate / occluded-in-play / out-of-shot / dead-ball) and uses human anchors as hard constraints. It needs no training, runs on CPU and can be measured within days. **M1** then replaces the candidate source with a full-resolution multi-frame heatmap net, to lift the candidate-recall ceiling. **M2** replaces the HMM with a small learned bidirectional transformer over per-frame candidate sets, camera motion and player context, covering a whole clip of up to ~1,000 frames. The end-to-end video transformer (Option B) does not fit 8 GB at the resolution the ball needs, and it needs far more dense labels than exist. Propagation from anchors (Option C) is useful as an extra candidate source inside A, not as the core.

### Cited Findings

**Project facts that constrain the design**
- Ball is ~11 px wide at 1280 (median 10.7–11.5, ~25% under 10 px). At 640 it is never detected (mAP50 0.0 vs 0.202 at 1280 in a controlled 3-epoch yolo11n rerun). Resolution, not data volume, was the binding lever. — `~/knowledge/wiki/footy-ball-detection-findings.md`
- 5.7× more single-frame ball labels (v11 vs v3) bought nothing: "Do not prioritise acquiring more single-frame ball labels … expecting recall to improve." — `footy-ball-detection-findings.md`
- RT-DETR-L at 1280 batch 2 used 6.4 GB of 8 GB to train. — `footy-ball-detection-findings.md`
- XS-VID (tiny objects in video): DETR-family models "significantly underperform". The winner, YOLOFT, uses RAFT-style multi-scale correlation between current- and previous-frame *features*. — `footy-ball-detection-findings.md`
- Broadcast cuts make frame t−1 a different camera, so frame-difference channels become noise on exactly the hardest frames; fall back to duplicating frame t at cuts. — `footy-ball-detection-findings.md`
- SOT/Siamese trackers fail silently when the ball moves more than ~2× its size and leaves the search window. Use them only as short-horizon interpolators. — `footy-ball-detection-findings.md`
- VitTrack held the ball on 9/176 GT frames (5%) on `arsenal_example`, and its confidence was a constant 0.119 whether on or off the ball. — `~/knowledge/wiki/footy-labeller-gotchas.md`
- About 37% of held-out ball frames have no detector candidate within 25 px, so "detector recall is the ceiling". A constrained Viterbi (ball verdict = anchor, not_ball = forbidden, next card = smallest Viterbi margin) reached 90% accuracy in ~13 taps/clip, vs ~20 for random order and 22–27 for spot-check order (5 held-out clips, perfect oracle). Consecutive `_segNNN` clips are not contiguous (gaps ≥ 26 frames), so tracks cannot be chained across segments. — `~/knowledge/wiki/footy-claude-autolabel-loop.md`
- On `arsenal_example`, misses come in long blind spells: one 51-frame gap holds 70% of misses (August handoff, not re-checked). — `docs/design/tracking-pipeline.md` (Ball tracker row)
- Ball precision by RT-DETR confidence band (257 decided verdicts): 0.1–0.2: 0.04, 0.2–0.3: 0.29, 0.3–0.4: 0.23, 0.4–0.5: 0.36, 0.5–0.6: 0.42, 0.6–0.7: 0.79, 0.7–0.8: 0.96, 0.8–0.9: 1.00. — `docs/design/tracking-pipeline.md`
- The broadcast splitter is not a shot-cut detector, but found 0 interior cuts in 3,175 frames across 5 clips. The classifier hardcodes `device="mps"` (`classifier.py:128`). — `docs/design/tracking-pipeline.md`
- The bake-off `BallTracker` protocol is strictly causal ("The harness calls `track()` once per frame in forward temporal order"). An offline method must be scored leave-anchors-out, not through this protocol. — `src/footy_track/ball_eval/interface.py`; `docs/design/bidirectional_tracking_research.md` §6
- Claude "ball" verdicts on non-tracker top-detector candidates had precision 10/33. Errors were mostly spare yellow balls by the advertising hoardings. A static-position filter for parked spare balls was weak, because camera pans are small and a resting in-play ball looks like a parked spare. — `footy-claude-autolabel-loop.md`

**Measured on the current GT (this note, `ball_gt_marks/*.jsonl`, human tier only)** [measured]
- Inter-frame ball displacement between consecutive human-tier frames (n = 1,034 pairs): median 3.4 px/frame, p75 9.0, p90 16.4, p99 45.2, max 644 (probably a mislabel or cut). Using **only `labeller` (human-placed) consecutive pairs (n = 152)**: median 14.4, p90 33.7, p99 63.6 px/frame. The checked-VitTrack runs are much slower than the human-placed ones (see the Q3 bias note).
- GT ball box width at 1080p: median 16.4 px, p10 13.0, p90 23.1 (≈ 11 px at 1280, consistent with the KB).

**Design precedents**
- TrackNetV3 adds a separate offline "rectification" (InpaintNet) stage. It takes the predicted trajectory plus an inpainting mask marking frames that may need correction, and outputs a repaired trajectory. The authors report it significantly improves accuracy and trajectory completeness. Repo training uses `--seq_len 16 --mask_ratio 0.3`. — [TrackNetV3 paper](https://people.cs.nycu.edu.tw/~yushuen/data/TrackNetV3.pdf), [TrackNetV3 repo (mirror)](https://gitee.com/tslnihao/TrackNetV3)
- TrackNet-family and WASB models are all published at 288x512 input, a 3.75× downscale from 1080p. None has been shown on a tiny ball in a panning/zooming wide broadcast shot. — sibling note `sports_ball_trackers.md`, citing the [TrackNetV3 paper](https://people.cs.nycu.edu.tw/~yushuen/data/TrackNetV3.pdf) and [WASB](https://arxiv.org/html/2311.05237)
- WASB (BMVC 2023): high-resolution feature extraction + position-aware training + temporally consistent inference. AP 7.8–16.8% above the best prior methods across 5 sports. MIT licence. — [arXiv 2311.05237](https://arxiv.org/abs/2311.05237), [WASB-SBDT repo](https://github.com/nttcom/WASB-SBDT)
- WASB soccer, as extracted by the sibling note: 3 stacked frames, small HRNet, 288x512, F1 88.3 at τ = 4 px. The data is 1920x1080 at 25 fps, 4 training clips / 11,994 frames, re-annotated per frame because the original labels were unreliable. — sibling `point_tracking_and_self_supervision.md`, citing [WASB html](https://arxiv.org/html/2311.05237)
- TOTNet (2025) uses explicit visibility classes with a visibility-weighted heatmap loss `L = w_v·(BCE_x + BCE_y)`, v ∈ {0, 1, 2, 3}. It uses occlusion augmentation (masking the ball region in the target frame with the surrounding mean colour), 5 input frames, 288x512 input and 8.65M params. On table tennis, fully occluded frames went from WASB RMSE 37.30 / acc 0.63 to TOTNet(OF) 7.19 / 0.80. The tennis data uses visibility 0 = out-of-frame, 1 = visible, 2 = partially visible, 3 = fully occluded. Out-of-frame samples are excluded from training. — [TOTNet arXiv html](https://arxiv.org/html/2508.09650v1)
- Ball Radar (KDD 2023) infers the ball trajectory from multi-agent player trajectories with Set Transformers (permutation-invariant over players) and a hierarchical Bi-LSTM. It first classifies the ball possessor and adds a "reality loss" to keep trajectories physically plausible. It runs on pitch coordinates (Metrica data). — [arXiv 2306.08206](https://arxiv.org/abs/2306.08206), [ballradar repo](https://github.com/pientist/ballradar)
- PLUCC (ice hockey puck, 2025) feeds a player-only context image (SAM2-segmented players, 360x640) into a gated decoder alongside the full-res frame (720x1280). It reports mAP 83.5 vs 79.6 (FCN-ResNet152) and 71.3 (Faster-RCNN), and pixel error 47.0 vs 52.1. Context alone scores only mAP 6.0, but 1% "context-driven dropout" of the RGB frame improved pixel error 61.15 → 47.0. It holds detections under full stick occlusion. — [PLUCC arXiv 2506.04365](https://arxiv.org/html/2506.04365v1)
- Classic broadcast-soccer ball tracking is two-phase: per-frame candidates, then candidates linked into trajectories and verified. Yu et al. 2006 report ~81% ball location accuracy. — sibling `offline_trajectory_and_mot.md`, citing [dblp](https://dblp.dagstuhl.de/rec/journals/tmm/YuLXT06.html)
- SAM2-family promptable VOS uses a fixed 1024/1008 px square input with stride-16 memory features. SAM3 at full frame is borderline OOM on 8 GB, and on the only small-fast-object benchmark SAM2-family trackers reach mean mIoU ~0.44–0.51. — sibling `promptable_vos.md` (bottom-line section)
- BoT-SORT adds camera-motion compensation, which matters because broadcast pans break image-space velocity models. — `footy-ball-detection-findings.md`

### Inferences

#### Common building blocks (shared by all options)

- **Visibility state, per frame, 4 classes + clip-level cut.** `VIS` = visible (position observable; partially occluded and blurred count here, with an optional "hard" flag). `OCC` = occluded-in-play, in shot (position exists but cannot be seen; behind or at a player's feet, or held by the keeper). `OOS` = out of shot (play continues off camera). `DEAD` = not in play (out of bounds, stoppage, replay-like inserts within a clip). This maps onto TrackNet's tennis 0–3 scheme (out-of-frame / visible / partial / occluded) plus a football-specific `DEAD`. `DEAD` matters because a stationary ball at a set piece and a parked spare ball look alike.
- **Camera motion without homography (Stage 0, CPU).** Per adjacent frame pair: KLT corners (or ORB) on 960x540 greyscale. Mask out player/referee boxes (we have them at ~0.97 mAP50), ball candidates, and the static broadcast overlay (scoreboard/logos, found as pixels that stay still while the background moves). Fit a 4-DOF similarity (tx, ty, scale, rotation) with RANSAC, falling back to a homography when inliers allow. Chain the transforms to a per-clip reference frame to get "compensated coordinates" (u, v) per frame, plus per-frame (dx, dy, dlog-scale). This is what BoT-SORT's CMC does. Cost: ~ms per frame on CPU [assumed], so all 97,625 frames take well under an hour on the mini [assumed]. Drift accumulates, but only *relative* motion over ≤ 40 s matters. **Later**, swap in the pitch homography from `calibration.md` to get metres and real physical speed limits. SEA-RAFT median background flow is the fallback when grass-only zoomed shots starve KLT of corners.
- **Player context as features, not a shared model.** Keep players on their own path: RT-DETR boxes at 0.97, then the offline MOT covered by the sibling thread. The ball model consumes player boxes as context:
  - per-candidate features: distance to the nearest player foot point, whether it sits inside a player box, players within 50 px;
  - a per-frame coarse occupancy grid (e.g. 9x16 counts);
  - optionally the nearest player tracklet IDs, enabling an "attached to player" mode during `OCC`.

  This is the Ball Radar / PLUCC idea at image level. PLUCC's result (context alone is weak at mAP 6.0, but it helps when added to pixels) is exactly the profile we want: a prior for occlusion hold, not a detector.

#### Option A0: non-learned M0 (candidates + bidirectional HMM). The MVP of Option A

- **Inputs:** RT-DETR ball candidates at conf ≥ 0.1 (already exist: 356,812 rows over 162 clips, S3 `machine_labels/`), camera motion (u, v), player boxes, and optional human anchors and Ball Check verdicts.
- **Model:** per frame, hidden states = {cand_1..cand_K, OCC, OOS, DEAD}.
  - Emission for cand_k = calibrated P(ball | conf band), taken from the measured Ball Check band precisions above.
  - Cand→cand transitions: heavy-tailed (Laplace/Student-t) on displacement in compensated coords. Scale ~10 px/frame, tail out to ~65 px/frame (the measured human-placed p99 is 63.6).
  - OCC holds the last position with a growing radius, so re-emergence must satisfy speed ≤ ~65 px/frame × gap.
  - OOS may only be entered or left within ~1 ball-travel of the frame edge.
  - DEAD is entered after a long stationary spell or an OOS at a touchline. It is weakly identifiable, so it is mainly human-supplied for now.
- **Bidirectional:** run forward–backward to get per-frame posteriors (confidence), then Viterbi for the MAP path. Anchors and `ball` verdicts clamp a state; `not_ball` verdicts forbid one. This is the constrained Viterbi already prototyped in the autolabel loop; extend it with OCC/OOS/DEAD and camera compensation.
- **Gap fill:** positions inside OCC spans come from a cubic/parabolic fit between the bracketing confident positions, the analytic analogue of TrackNetV3's InpaintNet. An "attached" alternative applies when the same player box is near both ends. Low-confidence candidates within ~6 px of the fitted curve are rescued, the ByteTrack second-pass idea done offline.
- **Long-range reach:** the whole clip (average 97,625/162 ≈ 600 frames). Prior from other frames = the posterior itself; a confident detection 40 frames later pulls the path toward it.
- **Cost / effort:** CPU, milliseconds per clip; 3–5 days to build including eval [assumed].
- **Risk:** the recall ceiling is the candidate set (~37% of ball frames have no candidate within 25 px). It mostly buys precision, plus recall inside short gaps. Hand-tuned emissions don't learn football context.

#### Option A: two-stage learned model (recommended end state)

**Stage 1: "BallHeat" short-window, full-resolution candidate heatmap.**
- **Input:** 5 frames (t−2..t+2) at **native 1920x1080**, early fusion, with each neighbour **warped into frame t by the Stage-0 global transform** before stacking. This turns TrackNet-style frame stacking into something that survives PTZ; at a cut (detect with a warp-residual spike), duplicate frame t. Add 2 difference channels (warped t±1 minus t). Late feature fusion (shared per-frame backbone + correlation, YOLOFT-style) is the M1.5 upgrade if early fusion plateaus.
- **Backbone choices:**
  - (i) a small HRNet-W18-ish / ResNet-18-FPN from ImageNet weights (~8–15M params);
  - (ii) the trained RT-DETR-L backbone (HGNetv2), frozen, run per frame at stride 8, with a trainable temporal-fusion + heatmap head. (ii) reuses ball-specific features already learned, but it is heavy per frame. Start with (i) and keep (ii) as an ablation.
- **Output:** heatmap at stride 4 (480x270 for 1080p) with a sub-pixel offset (CenterNet-style); the ball is ~4 cells wide. Also a per-frame "ball visible" logit and a 16–32-d embedding per peak. Emit the top-K = 16 peaks after NMS as the candidate set.
- **Training:** native-res 512x512 crops, so the ball is never downsampled. Sampling: 50% around labelled balls, 30% around known hard negatives (verified `not_ball` candidates, spare balls, shoes, line junctions), 20% random from frames inside *coverage* spans (see Q3).
- **Loss:** quality focal / CenterNet focal on the heatmap, visibility-weighted as in TOTNet. VIS gets weight 1; OCC with a best-guess point gets a soft low-weight target; OOS/DEAD give "no peak" targets, **only inside coverage spans**. Outside coverage, frames are positive-only (loss masked away from the labelled ball), because unlabelled frames still contain balls (sibling `point_tracking_and_self_supervision.md` makes the same point). Also use occlusion augmentation (TOTNet) and WASB-style hard-sample mining in later epochs.
- **VRAM [assumed, measure first]:** early fusion means one backbone pass per sample (15–17 input channels). At 512² crops with AMP, batch 8–16 should fit in 8 GB for a ~10M-param net. Full-frame inference at 1920x1088 is a single FCN pass per frame; expect ≥ 10 fps on the 2070 [assumed], so ~3 h for 97k frames.
- **Exit metric for M1:** **candidate recall@K=16 within 8 px** on visible frames of the held-out sets, vs RT-DETR's (≤ ~0.63 within 25 px per the autolabel-loop measurement).

**Stage 2: "BallSeq" long-window bidirectional sequence model over candidate sets.**
- **Per-frame tokens:** K candidate tokens (from BallHeat ∪ RT-DETR, dedup within 4 px) with features:
  - (x, y), (u, v), per-source scores, peak sharpness, appearance embedding, edge distance;
  - player-relation features;
  - a source flag (heat / detr / anchor / propagated).

  Plus a learned **null** token, one **frame-context** token (camera motion, zoom proxy = median player box height, occupancy grid, cut flag) and an optional **anchor** token (human keyframe or confirmed verdict: x, y, state).
- **Architecture:** per-frame set encoder (2-layer transformer over K+2 tokens), then attention-pooled frame embedding, then a **bidirectional temporal transformer**: 6 layers, d = 192, RoPE/ALiBi relative positions, full attention over T ≤ 1,024 frames (≈ 41 s, whole clips), sliding windows of 512 with 128 overlap beyond that. A Bi-GRU or a bidirectional Mamba/S4 are drop-in alternatives, with no reason to prefer them at T ≈ 1k.
- **Heads, per frame:**
  - pointer over {cand_1..K, null};
  - position regression in (u, v) (a candidate-weighted mean + residual; for null, regressed purely from context = the occlusion-hold prediction);
  - 4-way state (VIS/OCC/OOS/DEAD);
  - calibrated confidence (target: within 5 px).
  - An optional CRF/Viterbi layer on top for global consistency.
- **How the "prior from other frames" is represented:** temporal self-attention. Frame t attends to every other frame's candidate evidence and anchors. A sharp detection at t+30 and an anchor at t−60 both shape t. Both directions are symmetric, so "no ball yet, it appears soon" is handled: frames before the appearance learn OOS/OCC, with positions back-extrapolated from the first sighting.
- **Losses:**
  - pointer cross-entropy on frames with a known position;
  - Huber on position (VIS = 1.0, tool-interpolated = 0.5, OCC best-guess = 0.2);
  - class-balanced state cross-entropy;
  - small second-difference smoothness in (u, v), Ball Radar's "reality loss" analogue;
  - confidence BCE.
- **Training data:**
  - real dense tracks (Q3);
  - **synthetic corruption of real tracks**: drop true candidates at measured miss rates, inject false candidates sampled from the real FP distribution (verified `not_ball`, spare balls), mask random spans of 2–40 frames as OCC/OOS (TrackNetV3 InpaintNet trains with mask ratio 0.3);
  - pretraining on purely synthetic trajectories (parabolic flights + rolls + dribbles on a virtual pitch under a random PTZ camera, with synthetic player clutter);
  - random anchor dropout (0..N anchors), so one model serves both "fully automatic" and "anchor-assisted" use.
- **Budget:** ~2–6M params, < 1 GB VRAM, and CPU inference is fine. 1–2 weeks to build [assumed].

**Players in Option A:** a separate tracker (sibling MOT thread). Stage 1 may later gain a cheap player-centre head (multi-task) but doesn't need one.

**Risks:**
- (1) Stage 1 at native resolution is the VRAM and throughput risk; mitigate with crops and early fusion.
- (2) Global motion fails on grass-only close-ups; mitigate with the flow fallback and by flagging low-inlier frames.
- (3) Stage 2 overfits to the few real tracks. Mitigate with synthetic corruption and strict held-out-match evaluation.
- (4) Label noise: interpolated frames weighted lower.

#### Option B: end-to-end long-video transformer (sparse attention / memory, low-res features + high-res refinement)

- **Inputs:** raw frames over T = 32–128. **Outputs:** per-frame ball heatmap + state.
- **The prior** lives in a memory bank or sparse spatio-temporal attention. **Camera motion** is learned implicitly or given as flow input.
- **Why not now:** at the stride-8 features the ball needs, one 1920x1080 frame is a 240x135x256 map (~8.3M values), i.e. 64 frames ≈ 530M activations per layer [arithmetic]. Even with sparse attention this exceeds 8 GB in training without aggressive cropping, and cropping destroys the long-range context that is the point. Coarse-then-fine (stride-32 memory + high-res refinement crops around top-K) collapses back into Option A's two stages.
- **Labels:** it needs thousands of densely labelled sequences to train from scratch. We have **9 runs ≥ 25 frames** [measured].
- **Effort L+, risk high.** Revisit only if Option A's Stage 2 shows it is the appearance evidence, not the reasoning, that limits it, and a 16–24 GB rental is justified.

#### Option C: propagation from anchors (SAM2.1-crop / point tracker / SOT, forward + backward) with a learned fusion

- **Inputs:** anchors (human keyframes, or conf ≥ 0.7 detections at ≥ 0.96 precision). Propagate both ways (SAM2 `propagate_in_video(reverse=True)`), on native-res 512–1024 px moving crops. **Fusion:** per-frame weighting by distance-to-anchor and confidence, or learned.
- **Prior** = the anchors themselves. **Occlusion:** propagators drift onto the occluder, with no explicit state. VitTrack's constant 0.119 confidence shows SOT confidences can be uninformative.
- **Pros:** zero training for a first test.
- **Cons:** SAM-family at stride 16 is weak on ~11–16 px objects (sibling). SOT exits the window on fast balls. Point trackers follow the texture under the point, not an independently moving ball [inference].
- **Use inside A:** propagated positions become extra candidate tokens (source = "propagated") for BallSeq or extra HMM candidates. Spend ≤ 1 day testing SAM2.1-small on crops of `arsenal_example` (the sibling's recommended cheapest test) before investing more.

#### Comparison

| | A0 HMM over existing candidates | A two-stage (BallHeat + BallSeq) | B end-to-end video transformer | C propagate from anchors + fuse |
|---|---|---|---|---|
| Prior from other frames | forward–backward posteriors | bidirectional attention over T ≤ 1,024 | memory/sparse attention, T ~ 64–128 | anchors only |
| Occlusion / absence | explicit OCC/OOS/DEAD states, hand-tuned | explicit 4-way state head + null pointer + regressed hold position | state head possible, needs many labels | none explicit, drifts |
| Camera motion | KLT similarity → (u, v) | same + warped frame stack in Stage 1 | implicit / flow input | crop follows the target |
| Temporal reach | whole clip | whole clip (Stage 2), ±2 frames (Stage 1) | 64–128 frames | anchor-to-anchor |
| Players | features (inside-box, near-feet) | tokens/features + occupancy grid; separate MOT | could be joint | none |
| Training | none | Stage 1: ~10M params; Stage 2: ~2–6M | 50–200M+ | none / fusion head |
| VRAM | CPU | ≤ 8 GB with 512² crops [assumed]; Stage 2 < 1 GB | > 8 GB | SAM3 borderline OOM; SAM2.1-small on crops OK |
| Effort | 3–5 days | 3–5 weeks total | months | days, then stalls |
| Main risk | candidate-recall ceiling | Stage-1 throughput; small real-track set | data and VRAM | tiny object + fast motion |

#### Recommendation: staged build plan (each stage leaves a measurable artefact)

1. **E0 (in parallel, 1–2 days): the evaluation harness.** Leave-anchors-out + held-out-clip + held-out-match scoring on dense windows (Q3 §Evaluation). It must not use the causal `BallTracker` protocol. This gates everything.
2. **M0 (≈ 1 week): A0.**
   - Stage-0 camera motion for all 162 clips on the mini.
   - HMM/Viterbi decoder with 4 states over the existing RT-DETR candidates.
   - Anchor clamping; per-frame posterior confidence.
   - Outputs written as tier-3 rows (`source='ballseq_m0'`).

   **Success:** F1@10px on visible frames beats per-frame RT-DETR at its best threshold by a clear margin on the held-out sets, with a clip-bootstrap CI that excludes 0. Also report state accuracy. Gains here are mostly precision, plus recall inside short gaps (measured: linear interpolation over 10-frame gaps lands within 10 px 86–88% of the time on the mixed set). M0 then immediately becomes the **labelling assistant** (model-assisted interpolation, track-level review, active-learning card selection).
3. **M1 (1–2 weeks): BallHeat.** Train on dense windows (Q3). **Success:** candidate recall@16 within 8 px clearly above RT-DETR's on the held-out match; plug it into M0's HMM and re-measure end-to-end F1.
4. **M2 (1–2 weeks): BallSeq** replaces the HMM emissions/transitions. Train on real tracks + synthetic corruption + synthetic trajectories. **Success:** beats M0+M1 on the same held-out sets, especially OCC position error and OOS/OCC/DEAD state F1. Report fully automatic AND with 1 anchor per 50 frames (the "I know where the ball is in one frame" use case).
5. **M3 (ongoing): self-training rounds** on the 97k Man City frames + unlabelled Aston Villa/Bournemouth clips (never Norwich, the held-out match, see Q3). Add player-tracklet "attached" mode. Swap (u, v) for pitch metres once calibration lands.
6. **Optional M4:** distil BallHeat + BallSeq into a feature-level temporal detector, or retrain RT-DETR on pseudo-labels so the per-frame detector improves as a by-product (the flywheel in `autolabelling_architecture_review.md` §6).

### Gaps
- None of the VRAM/throughput figures for BallHeat at native resolution are measured. Measure batch-8 512² training memory and full-frame inference fps on the 2070 before committing.
- No source found showing KLT/ECC global-motion robustness on grass-dominated zoomed broadcast shots. It must be measured (residual of player-foot tracks after compensation).
- I could not verify WASB's soccer internals myself (abstract and README only); those figures come via the sibling note's reading of the paper html.
- Whether Stage-2 synthetic pretraining transfers to real candidate noise is untested anywhere I found. TrackNetV3's random-mask training is the closest precedent, and it is single-camera badminton.
- The interior-cut rate for other matches and broadcasters is unmeasured (0 cuts in 5 clips is one match).

---

## Q2. What to reuse off the shelf vs what must be trained

### Takeaway
Reuse the existing RT-DETR-L (candidates now, player boxes always, optionally a frozen backbone for BallHeat), OpenCV KLT/RANSAC for camera motion, and the existing constrained-Viterbi prototype and Ball Check precision bands for M0. Treat WASB pretrained soccer weights, SAM2.1-small-on-crops and SEA-RAFT as **cheap side experiments** that may add a candidate source. Two things must be trained: the native-resolution multi-frame heatmap net and the bidirectional sequence model. Use TrackNetV3's InpaintNet and Ball Radar as architectural templates only, not as weights.

### Cited Findings
- RT-DETR-L `rtdetr_1920_v11` (trained at 1920, served at 1280): ball mAP50 0.665 / recall 0.591 / P 0.862 on v11-val (211 balls). Players ~0.97 mAP50 (yolo11n figure; RT-DETR per-class player mAP not logged). Weights on S3 only. Batch inference made 2,767,151 boxes over 97,625 frames at conf ≥ 0.1, ~20 fps on a 3060. — `docs/design/tracking-pipeline.md`; `footy-labeller-gotchas.md`
- The KB lists the RT-DETR mAP figures as UNRESOLVED (0.221 vs 0.665 for the same weights) until re-measured with the instance count. — `footy-ball-detection-findings.md`
- WASB code + pretrained models, MIT licence. Per the sibling note, the repo holds evaluation code for pretrained models (training code not released). — [WASB-SBDT](https://github.com/nttcom/WASB-SBDT); sibling `point_tracking_and_self_supervision.md`
- TrackNetV3: MIT, with checkpoints. Its background-median trick and checkpoints are badminton / fixed-camera specific. — sibling `sports_ball_trackers.md`
- SEA-RAFT (ECCV 2024 oral): ≥ 2.3× faster than existing methods at competitive accuracy (it predicts the initial flow directly and uses a ConvNeXt-block RNN). Spring EPE 3.69. — [ECVA paper page](https://ecva.net/papers/eccv_2024/papers_ECCV/html/1065_ECCV_2024_paper.php), [arXiv 2405.14793](https://arxiv.org/html/2405.14793v1)
- CoTracker3 is pseudo-labelled on real videos with off-the-shelf trackers as teachers. It reports better results with ~1,000× less real data than prior real-data approaches, offline and online variants, and particular robustness to occlusion. — [CoTracker3 arXiv](https://arxiv.org/html/2410.11831v1), [ICCV 2025](https://openaccess.thecvf.com/content/ICCV2025/html/Karaev_CoTracker3_Simpler_and_Better_Point_Tracking_by_Pseudo-Labelling_Real_Videos_ICCV_2025_paper.html)
- SAM2 natively supports backward propagation via `propagate_in_video(reverse=True)`. — `docs/design/bidirectional_tracking_research.md` §3
- SAM2.1 tiny/small are Apache-2.0, 39–46M params. The sibling recommends native-res 512–1024 px moving crops seeded by anchors, forward + reverse. — sibling `promptable_vos.md`
- Ball Radar code is public (Bi-LSTM, Set Transformer, possessor classifier) and trained on Metrica tracking data. — [ballradar](https://github.com/pientist/ballradar)
- The calibration plan wraps a pretrained SoccerNet-calibration-style keypoint/line model behind `PitchFeatureDetector`, with classical line refinement on top. Output goes to the existing `frame.homography` columns. — `docs/design/calibration.md`

### Inferences

| Component | Reuse | Train | Notes |
|---|---|---|---|
| Player/referee boxes | RT-DETR-L as is | — | Context for the ball; MOT per sibling thread |
| Ball candidates (M0) | RT-DETR conf ≥ 0.1 rows already on S3 | — | Calibrate with the Ball Check band precisions |
| Camera motion | OpenCV KLT + RANSAC (BoT-SORT CMC style); SEA-RAFT as fallback | — | Check the SEA-RAFT licence before adopting (not verified here) |
| Global trajectory decoding (M0) | Constrained Viterbi from the autolabel loop (jobs repo `footy_autolabel_queue.py` per KB) | — | Extend with OCC/OOS/DEAD and (u, v) |
| Candidate heatmap (M1) | Backbone: ImageNet ResNet-18/HRNet, or frozen RT-DETR HGNetv2 | **Yes**: temporal fusion + heatmap head | No public model works at native 1080p PTZ |
| WASB soccer weights | 1-hour baseline test on our dense windows (288x512 and 2×2 tiles) | — | If it finds balls RT-DETR misses, add it as a candidate source |
| Sequence model (M2) | TrackNetV3 InpaintNet / Ball Radar as templates | **Yes** | Small; trainable on CPU+GPU; mostly synthetic-augmented |
| Anchor propagation | SAM2.1-small on crops, both directions | Optional LoRA later | Extra candidate tokens only |
| Point tracking | CoTracker3 offline | — | Camera-motion research / stabilisation only; check licence (not verified here) |
| Pitch homography | Pretrained calibration model per `calibration.md` | Fine-tune later | Enables metres and physics priors in M3 |

### Gaps
- I did not verify the SEA-RAFT or CoTracker3 licences.
- Whether WASB's pretrained soccer weights generalise to PTZ broadcast at 1080p is unknown: their soccer set is not PTZ broadcast, per the sibling.
- The RT-DETR ball mAP figure is still officially unresolved in the KB. Re-measure it on the new held-out set (Q3) as the baseline.

---

## Q3. Data and labelling plan

### Takeaway
The model needs **contiguous, fully accounted-for windows**, not more isolated ball boxes. That means 10-second windows where every frame either has a ball **centre point** (human keyframe roughly every 5 frames plus tool interpolation, then verified), or sits inside an explicit **state span** (occluded / out-of-shot / dead). Each window carries a **coverage** flag so that absence is meaningful. Today only 9 runs ≥ 25 frames exist, and all of them are in detector-training clips. The first priority is a **held-out-match eval set from Norwich**, which has zero frames in any detector dataset [measured]. Isolated Ball Check yes/no cards and isolated single-frame boxes are now the lowest-value labelling.

### Cited Findings

**Current GT inventory [measured on `/mnt/storage/footy_data/ball_gt_marks`, human tier]**
- 1,515 ball frames in 35 clips. Per match: Man City 620 frames / 23 clips (538 labeller, 82 checked); **Aston Villa 579 / 3 clips (52 labeller, 527 checked VitTrack)**; Bournemouth 316 / 9 clips (100 labeller, 216 checked). So GT already spans 3 matches, contrary to "one match", but 568 of the Aston Villa frames are one clip (`astonvilla_seg080`, longest run 359) and 220 of Bournemouth's are one clip (`bournemouth_1st_seg070`, longest run 157).
- 481 consecutive runs, median length 1. 9 runs ≥ 25 frames; 959 frames sit in runs ≥ 10. With VitTrack-checked rows removed (human-placed only): 538 runs, **0 runs ≥ 25 frames**, 46 frames in runs ≥ 10.
- 37 explicit `no_ball` frames (26 / 4 / 7 by match). Only 2 `out_of_play_ball` boxes.
- Ball Check log: 364 rows = 179 ball, 149 not_ball, 20 unsure, 12 corrected, 4 tombstones.
- Detector-training leakage check (file counts in `/mnt/storage/footy_data/detection_dataset/roboflow_dataset_{3,11}`, image + label files): astonvilla 1,066, bournemouth 474, mancity 1,696, arsenal_example 314, **norwich 0**. `arsenal_norwich/` holds only `original_video/arsenal_norwich.mkv` (3.8 GB) and 606 sampled frames; it has **not** been split into broadcast clips.
- `astonvilla_seg080` and `bournemouth_1st_seg070` are in the detector training set (recall ~0.998–1.000 = memorisation). `arsenal_example` is the only clean dense sequential clip. — `footy-ball-detection-findings.md`
- v11-val has 211 balls, 156 (74%) from one 10-second clip (`arsenal_example`). — `docs/design/tracking-pipeline.md`
- 44-instance val set lesson: v3-val showed mAP50 0.696 where the broader v11-val showed ~0.22–0.25. — `footy-ball-detection-findings.md`

**Keyframe-interval evidence [measured]: linear interpolation between GT points k frames apart inside contiguous human-tier runs, error at the skipped frames (1080p px)**

| keyframe every k | all human-tier (n) | median | ≤ 5 px | ≤ 10 px | human-placed only (n) | ≤ 5 px | ≤ 10 px |
|---|---|---|---|---|---|---|---|
| 2 | 503 | 0.8 | 0.94 | 0.96 | 58 | 0.84 | 0.93 |
| 3 | 658 | 1.0 | 0.93 | 0.97 | 62 | 0.85 | 0.92 |
| 5 | 764 | 1.2 | 0.90 | 0.95 | 56 | 0.84 | 0.98 |
| 10 | 810 | 2.0 | 0.75 | 0.88 | 36 | 0.36 | 0.67 |
| 15 | 812 | 3.0 | 0.65 | 0.78 | 14 | 0.00 | 0.07 |
| 25 | 768 | 7.1 | 0.40 | 0.60 | — | — | — |

Excluding `astonvilla_seg080` (the dominant VitTrack clip) changes the all-tier rows very little (k = 5: 0.93 ≤ 5 px; k = 10: 0.76 / 0.86; k = 25: 0.33 / 0.59).

**Label-quality and tool facts**
- Checked VitTrack (sidecar) boxes are mostly right: 135 ball / 21 box_off / 1 not_ball of 157. Spot-checking tracks (first/last/fastest frame) instead of every frame cut 720 cards to ~37. — `footy-labeller-gotchas.md`
- About 3.3 s per Ball Check card. Only `corrected` boxes reach the sidecar; `ball`/`not_ball` verdicts reached nothing at the time of that doc. — `docs/design/tracking-pipeline.md` (`/ball_check` row). (PR #26 "ball-check verdicts to store" has since merged per git log; the working tree has uncommitted deletions around `feature_store/ball_checks.py`, so the current state is in flux. Not relied on here.)
- "Missed-ball" cards (tracker predicts a position, no detector candidate within 25 px) were real balls 4/6 and 12/14, the best source of hard positives. — `footy-claude-autolabel-loop.md`
- Sidecar format: `{"frame_index", "bbox": {x,y,w,h} normalised, "center", "tags": [label, model(, human_checked)]}`. Skip markers have `bbox: null` and tags exactly `["no_ball"]` or `["not_broadcast"]`. Frames with neither boxes nor markers are not written at all. Flush precedence: `not_broadcast` > `no_ball` > boxes. — `src/footy_track/labeller/README.md` LAB-101/102/106
- Provenance tiers: 1 = `labeller`, 2 = `vittrack` + `human_checked`, 3 = machine only. New machine output must be written as tier 3 and must not receive `human_checked` automatically. — `footy-data-plane-reconciliation.md`
- `ingest_gt._split_tags` used to return a trailing marker tag (`human_checked`) as the class; it was fixed so the class is the first non-marker tag. Any new marker tag must be registered or the same class of bug recurs. — `footy-labeller-gotchas.md` (2026-09-24), commit `2501e06` (T-01)
- The feature store `detection` table already has `is_interpolated`, `needs_review`, `reviewed`, `track_id`. `track_meta` has team/jersey/player_id. Canonical arbitration treats all `gt_import_*` runs of a game as one run_group. — `src/footy_track/feature_store/schema.py`
- Sparse linear keyframe interpolation "breaks down as keyframes get sparser". VATIC uses pixel information from the labelled frames to interpolate between them, and evaluates algorithms by the labelling budget needed to reach acceptable accuracy. — [Vondrick, Patterson, Ramanan, IJCV 2013](https://www.cs.columbia.edu/~vondrick/vatic/ijcv.pdf) (abstract-level)
- CVAT track mode: shapes set on keyframes are interpolated on the frames between. An "Outside" property ends a track when the object "disappears or becomes too small", and Merge rejoins tracks when it reappears. — [CVAT track mode docs](https://docs.cvat.ai/docs/annotation/manual-annotation/modes/track-mode-basics/)
- SoccerNet-Tracking: 200 × 30 s clips + one full 45-min half, main camera, 12 games (2019 Swiss Super League), fully annotated at 25 fps with boxes and IDs for players, referees **and balls**. Clips were chosen around key action spots. — [SoccerNet-Tracking arXiv 2204.06918](https://arxiv.org/pdf/2204.06918)
- WASB evaluates at τ = 4 px on held-out clips from a different game. Its soccer training set was 11,994 per-frame-labelled frames from 4 clips. — sibling `point_tracking_and_self_supervision.md` citing [WASB](https://arxiv.org/html/2311.05237)

### Inferences

#### 3.1 Label types: what the model needs (priority P0 > P1 > P2 > P3)

| Label | Form | Used by | Priority |
|---|---|---|---|
| Ball **centre point** on keyframes | 1 click in a zoomed loupe (2× to 4× view). Store as today's bbox line with a fixed-size box around the click (per-clip median ball size) + marker tag `point_only` | BallHeat targets, BallSeq pointer/position, eval | **P0** |
| **State spans** VIS / OCC / OOS / DEAD | Interval `[start, end, state]`, one hotkey at each transition | BallSeq state head; BallHeat negatives; eval state accuracy | **P0** |
| **Coverage span** ("every frame here is accounted for") | Interval per window | Makes absence meaningful → negatives + precision eval | **P0** |
| Keyframe flags `touch` (kick/bounce/deflection) and `uncertain` (blurred/merged) | Marker tag on the keyframe | Piecewise interpolation; down-weighting; physics-mode learning | P1 |
| OCC best-guess point | Optional click with tag `occluded_guess` when you can infer it (ball at a player's feet) | Low-weight position target for occlusion hold | P1 |
| Spare/parked balls | One point per window, label `spare_ball` (or `out_of_play_ball`) | Hard negatives (the main Claude/FP error) | P1 |
| Cut | Span state `cut` at the new shot's first frame | Stage-1 window fallback | P2 (rare: 0 in 5 clips) |
| Ball box w/h | Derived from point + zoom, or snapped to an RT-DETR box within 4 px | Detector retraining only | P2 |
| Player boxes | Machine (RT-DETR 0.97); don't hand-draw | Context features | P3 |
| Player IDs / tracklet purity | MOT thread (`player_reid.md` milestone 1: ~10 clips, ~2 h [assumed in that doc]) | Player tracking eval | P2 (other thread) |
| Team | Automatic k = 2 torso hue (1.0% ambiguous on one clip) | Context | P3 |
| Pitch keypoints | Per `calibration.md`; start from a pretrained model | Metres / physics (M3) | P2, later |

**Point vs box: point.** Box geometry carries almost no information at 13–23 px. The heatmap and sequence models consume centres, and a click is faster and more precise than a drag. Precision target: a click within ~2 px at 1080p, which needs a zoomed loupe (Ball Check already proved a floored, upscaled window is necessary at this size).

#### 3.2 Density, sequence length and quantities

- **Window = 250 frames (10 s) contiguous**, or the whole clip if it is ≤ 15 s. Long enough to contain occlusions and re-appearances (the 51-frame blind spell on `arsenal_example` would fit) and short enough to finish in one sitting.
- **Keyframe interval: default every 5 frames, plus every touch/bounce/direction change.** Measured: linear interpolation at k = 5 lands within 5 px on 90% (all-tier) / 84% (human-placed, n = 56) of skipped frames, and within 10 px on 95–98%. At k = 10 human-placed accuracy collapses (36% ≤ 5 px), and k ≥ 15 is unusable. With snap-to-candidate or M0-assisted interpolation (VATIC's argument), I expect k ≈ 10 to become acceptable for rolling/slow play [assumed, measure on the first windows]. Slow ball → k = 10; flight → k = 3–5.
- **Bias warning (measured):** the existing long tracks are VitTrack-propagated and slow (median 3.4 px/frame), while human-placed consecutive pairs move at median 14.4 px/frame. The checked-VitTrack runs over-represent easy, slow ball, so **new windows should deliberately target fast play** (crosses, long passes, shots, clearances).
- **Training quantity targets [assumed, from the WASB precedent of ~12k per-frame soccer frames from 4 clips]:**
  - BallHeat: ≥ 15–25k covered frames = 60–100 windows, across ≥ 3 matches, ≥ 30% fast-play windows.
  - BallSeq: the same windows (≈ 60–100 real sequences) + synthetic corruption. Real windows teach real FP structure and real occlusion patterns.
  - Throughput: 50 keyframes × ~3 s + ~10 state transitions + a 2× scrub review ≈ **5–8 min per window** [assumed; Ball Check's measured 3.3 s/card is the closest anchor] → ~8–10 windows/hour → **60–100 windows ≈ 7–12 hours of labelling**.
- **Matches:** 3 labelled matches are all in detector training. Use **Norwich (0 files in any detector dataset) as the held-out match**. Generalisation claims need ≥ 2 held-out matches eventually [assumed]; acquire ≥ 2 more matches (different stadium, kit pair, broadcaster) for M3.

#### 3.3 Highest-leverage labelling per hour (ranked)

1. **Held-out Norwich eval windows** (dense keyframes + spans + coverage). Every future claim depends on it, and it is the only leakage-free match.
2. **Dense training windows of fast play** on Man City / Aston Villa / Bournemouth clips that are **not** already in v11-val. Each one serves BallHeat positives *and* negatives, BallSeq sequences and an in-match eval split.
3. **Retro-fitting coverage + state spans onto the existing long runs** (`astonvilla_seg080`, `bournemouth_1st_seg070`, `arsenal_example`, `arsenal_mancity_seg050/seg010`). Minutes per clip, and it turns 900+ existing frames into usable sequences with meaningful absence.
4. **Track-level review of M0 output** (once M0 exists). Play a 2–4 s loop with the predicted dot. Verdict: good / drifts-at-frame / wrong-object / wrong-state; a correction adds a keyframe. The KB measured a ~20× card reduction (720 → ~37) from spot-checking tracks over per-frame checks.
5. **Active-learning keyframes** inside chosen windows: the tool proposes the frame with the smallest Viterbi margin / largest forward–backward disagreement (KB: ~13 taps/clip to 90%, vs ~20 random).
6. **Missed-ball cards inside chosen windows**: real balls 12/14 in the KB, hard positives.
7. *Lower value now:* random Ball Check yes/no on isolated candidates. The confidence bands are already measured from 257 verdicts; more random cards mostly refine a curve we already have and produce no sequence data.
8. *Lowest value now:* isolated single-frame ball boxes on random frames (5.7× more bought nothing); VitTrack "Run" (5% hold); hand-drawn player boxes.

#### 3.4 Making labelling cheaper (tooling, in build order)

- **Keyframe + interpolation mode in the labeller (CVAT track-mode semantics):**
  - `K` = keyframe click (in the loupe);
  - state hotkeys `V`/`O`/`S`/`D` start a span at the current frame, and it runs until the next state key;
  - `[` / `]` = coverage start/end;
  - `T` = touch flag, `U` = uncertain;
  - frame step and 0.25–1× play;
  - after a window, a **2× scrub-review pass** with the interpolated dot overlaid. Clicking anywhere the dot is off adds a keyframe there.
- **Interpolation backend, upgradeable:** (1) linear (measured above); (2) snap to an RT-DETR candidate within 6 px of the linear estimate; (3) M0-conditioned on the keyframes (model-assisted interpolation, VATIC-style); (4) BallSeq with anchor tokens. Each upgrade should lower the keyframes needed per window. Track that number as the labelling-efficiency KPI.
- **Machine pre-fill:** RT-DETR candidates at conf ≥ 0.7 (precision 0.96–1.00) are shown pre-placed as *suggested* keyframes, accepted with one key. Never auto-promote them (LAB-002).
- **Phone "track check" cards** replacing isolated Ball Check cards: a 1–2 s MP4 loop (the iOS Range/faststart serving already works per KB) with the M0 dot. Verdicts: good / off-at / wrong-object / state-wrong.
- **Event marks:** "ball enters shot", "ball leaves shot", "out of play" are just state transitions, so one keypress each.

#### 3.5 Pseudo-labelling / self-training (97k Man City frames + unlabelled Aston Villa / Bournemouth clips)

- **Never** pseudo-label, train on, or tune on Norwich, the held-out match. Exclude held-out clips via the canonical clip id (T-03, commit `980e834`).
- **Round r:**
  1. Run the current best model (M0, then M1+M0, then M2) on all non-held-out clips.
  2. Keep segments ≥ 25 frames where all of these hold:
     - (a) forward and backward decodes agree within 5 px (or the BallHeat and RT-DETR candidate sources agree);
     - (b) posterior confidence ≥ 0.9 on ≥ 90% of frames;
     - (c) ≥ 30% of frames have an RT-DETR candidate at conf ≥ 0.7;
     - (d) the smoothness residual is below a threshold.
  3. Write them as **tier 3** (`source='ballseq_rN'`, `is_interpolated` where applicable), never `human_checked`.
  4. Train with weight 0.3–0.5 vs human data.
- **Audit each round:** a human reviews ≥ 50 randomly sampled pseudo-tracks via track check, stratified by match and confidence. Promote the batch only if ≥ 95% of audited frames are within 10 px and state accuracy ≥ 90% [assumed thresholds]. The audit set doubles as a pseudo-label precision estimate.
- **Free self-supervision for BallSeq:** mask-and-infill on confident tracks (randomly hide 2–40 frames and predict them), the TrackNetV3 InpaintNet recipe. This directly trains "occlusion hold" from unlabelled video.
- **Hard-example mining** (WASB style): frames where pseudo-tracks and the detector disagree go to the active-learning queue, not into training.
- **What keeps it honest:** (i) the held-out match is never touched; (ii) per-round audits; (iii) eval only on human keyframes + human state spans; (iv) report every model on the same fixed eval sets with instance counts and clip-bootstrap CIs.

#### 3.6 Held-out evaluation design

- **Splits (clip-level, by canonical clip id):**
  - **EV-M**: held-out *match* = Norwich, 20 windows from ≥ 15 distinct clips across both halves.
  - **EV-C**: held-out *clips* from training matches, ≥ 15 windows from clips with zero frames in v11/v3 or in any training window.
  - Exclude `astonvilla_seg080`, `bournemouth_1st_seg070` and `arsenal_example` from all eval (detector-training leakage).
- **Minimum instance counts per eval set [assumed, reasoning given]:**
  - ≥ 500 VIS human keyframes from ≥ 15 windows, with no single window > 10% of the instances. The binomial 95% CI at n = 500 is about ±4 pts; within-window correlation inflates it, so report a **window-level bootstrap**.
  - ≥ 50 OCC spans, ≥ 30 OOS spans/transitions, ≥ 20 DEAD spans.
  - Compare with today's v11-val: 211 balls with 74% from one clip, i.e. effectively 1–3 independent samples.
- **Metrics:**
  - Position (VIS frames, scored at human keyframes only): recall@5 px and @10 px at 1080p (5 px ≈ 0.3 ball diameters, 10 px ≈ 0.6; WASB's τ = 4 px is the strict reference).
  - Precision: predictions on OCC/OOS/DEAD frames inside coverage count as FPs. F1@10.
  - State: 4-way confusion + macro-F1, especially OCC vs OOS.
  - OCC hold: position error at OCC frames that have an `occluded_guess` or are bracketed by VIS keyframes ≤ 10 frames apart.
  - Continuity: fragments per minute, wrong-object switches per minute, % of frames inside correct runs ≥ 25 frames.
  - **Anchor-efficiency curve:** F1@10 vs anchors per minute (0, 1/50 frames, 1/25), via leave-anchors-out. This measures the user's "one known frame gives a prior" goal.
  - Labelling KPI: keyframes needed per window to reach 95% ≤ 10 px.
- **Baselines on the same sets:** per-frame RT-DETR (best threshold), M0, M0 + BallHeat, BallSeq, plus a WASB-pretrained sanity baseline.

#### 3.7 Minimal schema and tooling changes (against the actual formats)

1. **Sidecar JSONL (LAB-101) stays as is for points.** Add marker tags `point_only`, `touch`, `uncertain`, `occluded_guess`. Register them in the labeller's marker-tag set and in `ingest_gt` so `_split_tags` still takes the class from the first non-marker tag (the T-01 failure mode). Add a provenance tag **`interp`** for tool-interpolated points (`PROVENANCE_TAGS` in `constants.py`). It is tier 3 until a scrub review adds `human_checked`, which then makes it tier 2. Never write interpolated points as `labeller`.
2. **New append-only `<gt dir>/ball_spans/<clip>.jsonl`**, mirroring `ball_checks/` (append-only, latest-wins, `null` tombstones as in LAB-1004..1006). The sidecar is not rewritten, which avoids the truncation bug class. One line per span:
   `{"clip": "...", "start": 120, "end": 146, "state": "visible|occluded|out_of_shot|dead|cut|coverage", "author": "human", "ts": ...}`.
   A `coverage` span asserts that every frame in `[start, end]` has a ball point (keyframe or reviewed interp) or lies in a non-visible state span. Existing `no_ball` markers stay valid and are read as "not visible, state unspecified".
3. **Feature store:** add one table
   `ball_span(game_id, clip_id, start_frame, end_frame, state, source, run_id, reviewed, PRIMARY KEY (game_id, source, run_id, start_frame, state))`, ingested from `ball_spans/` under `run_id = gt_import_spans` (so it joins the existing `gt_import_*` run_group rule) and written by models under `source='ballseq_*'`.
   - Interpolated points use the existing `detection.is_interpolated = TRUE`, `source='interp'`, `reviewed` per review.
   - Model ball tracks use `track_id = 1` per clip + `track_meta`.
   - No change to `frame`; `frame.homography` stays for calibration.
4. **Eval manifest** `eval_split.json` (canonical clip ids → EV-M / EV-C / train), enforced by the export script and the eval harness.
5. **Labeller:** the keyframe/interp/state-hotkey mode (§3.4) replaces VitTrack Run as the default propagation.
6. **Until (2)–(5) exist (i.e. this week):** keep keyframes in the labeller as normal `labeller` boxes, and record spans in a plain CSV `ball_spans_manual.csv` (`clip,start,end,state,note`). Converting it to `ball_spans/` later is a trivial script.

### Gaps
- Labelling throughput per window is assumed; time the first 3 windows.
- The interpolation-error table's human-placed rows are small (n = 14–62). The all-tier rows are dominated by checked-VitTrack runs, which may be smoothed by VitTrack's tendency to lag or stick. Treat k = 5 as safe and k = 10 as "verify".
- Whether 60–100 windows suffice for BallHeat at native resolution is an extrapolation from WASB (4 clips, ~12k frames, different camera setup).
- Which Man City clips are free of v11/v3 frames was not enumerated here. The T-03 canonical clip-id tooling should produce that list.
- Norwich must first be split into broadcast clips (CPU job). The classifier's hardcoded `mps` device will fail on Linux/CUDA boxes without a fix or override.

---

## Q4. What to label this week (ordered)

### Takeaway
Spend this week's labelling on **dense, state-annotated windows**: a little retro-fitting first, then a leakage-free Norwich eval set, then fast-play training windows. Stop random isolated cards.

### Cited Findings
- Norwich has 0 files in the detector datasets and exists only as `arsenal_norwich.mkv` (3.8 GB), unsplit. — measured (this note)
- The existing long runs (`astonvilla_seg080` 359, `bournemouth_1st_seg070` 157, `arsenal_example` 50) are the only sequences ≥ 25 frames. — measured (this note)
- Missed-ball cards are the best hard-positive source (12/14 real). — `footy-claude-autolabel-loop.md`
- More isolated single-frame labels did not improve recall. — `footy-ball-detection-findings.md`

### Inferences
The ordered list for the user. Times are [assumed]; time the first windows and adjust.

0. **Machine prep (not labelling; queue it today):** split `arsenal_norwich.mkv` into broadcast clips. The classifier's `device="mps"` needs overriding off-Mac. Then run RT-DETR candidates on the Norwich clips on the desktop GPU (Prefect gpu pool).
1. **Retro-fit spans on the 5 existing longest-run clips (~45 min):** `astonvilla_seg080`, `bournemouth_1st_seg070`, `arsenal_example`, `arsenal_mancity_seg050`, `arsenal_mancity_seg010`. For each, write CSV rows in `ball_spans_manual.csv`:
   - `coverage` for the range you have checked end to end;
   - every `occluded` / `out_of_shot` / `dead` interval inside it;
   - a `touch` note at obvious kicks.

   Fix any obviously off points while there. This gives M0 tuning data with real states on day one (these clips are training-only; they leak into the detector).
2. **Norwich held-out eval, first 10 windows (~1.5 h):** 10 × 10 s windows spread across the match (roughly one per 5 minutes of play). Mix wide and zoomed shots; include ≥ 3 windows with crosses/long balls and ≥ 2 that start with the ball off-screen or hidden. Per window:
   - a keyframe (normal `labeller` box centred on the ball) every 5 frames and at every touch/bounce;
   - CSV spans for occluded / out_of_shot / dead + coverage;
   - one `spare_ball` point if a parked ball is visible.

   Target next week: 20 windows.
3. **Fast-play training windows, Man City (~1.5–2 h, 12–15 windows):** choose clips not in v11/v3. Pick windows where RT-DETR candidates jump far between frames or vanish for > 10 frames (the hard, fast cases the current GT under-represents). Same protocol as 2.
4. **Aston Villa / Bournemouth training windows (~1 h, 8–10 windows)** from clips other than seg080/seg070, same protocol. This gives 3 training matches.
5. **Only if time remains:** missed-ball cards and active-learning frames *inside the windows above*, never random isolated cards.

**Stop or deprioritise this week:**
- random Ball Check yes/no on isolated candidates (bands already measured);
- isolated single-frame boxes;
- VitTrack Run;
- player boxes or IDs (machine boxes suffice for the ball model; IDs belong to the MOT thread).

**Expected week output [assumed]:** ~35–45 windows ≈ 9–11k accounted-for frames, ≈ 2k human keyframes, ≈ 100+ state spans. That is roughly 7× the current human-tier ball frames and the first meaningful negatives. It is enough to build and honestly score M0 and to start training BallHeat.

### Gaps
- Window throughput is unmeasured. If a window takes > 10 min, drop to keyframes every 5 frames during flight only and every 10 when the ball is slow or held.
- The best Norwich window selection depends on clip splitting finishing first.
