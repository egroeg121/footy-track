# Research: Bidirectional + Trajectory-Aware Ball Tracking for Offline Autolabelling

**Bead:** ft-dva · **Date:** 2026-06-21 · **Researcher:** polecat flint
**Question:** In offline video autolabelling for ball tracking, can we exploit
*bidirectional* temporal information — track forward **and** backward from sparse
human GT marks, then fuse with trajectory interpolation to fill difficult frames —
and is it worth adding as **Method E** in the bake-off?

> **Headline.** Yes, bidirectionality is real, native, and cheap to get — SAM2's
> video predictor already propagates backward (`propagate_in_video(reverse=True)`),
> and "fit a ball trajectory through sparse anchors + gated candidates" is a
> 20-year-old, well-validated idea in broadcast-soccer tracking. **But it is not a
> "Method E" in the current bake-off.** It is a *different kind of thing*: the
> bake-off scores **causal, per-frame** trackers (`track(prev_bbox, frame)` called
> in forward order), and a bidirectional/trajectory method by definition needs the
> *whole clip and all anchors up front*. It does not fit the `BallTracker` protocol
> and should not be forced into it. The right move is to treat **bidirectional +
> trajectory fusion as a post-processing / fusion layer that wraps the bake-off
> winner**, not as a competitor inside it — and, given the
> [architecture review](./autolabelling_architecture_review.md)'s finding that
> *detection*, not tracking, is 100% of the failure, to build the cheapest version
> of it (SAM2 backward propagation from anchors) **now**, because it directly buys
> the thing we actually lack: filled-in GT between sparse human marks.

---

## 1. Why bidirectionality matters *here specifically*

The whole premise of this pipeline is **sparse human GT every ~N frames** (the
architecture review argues N must be *measured*, not the assumed 5 — see §3 of
that doc). A causal forward tracker seeded at anchor frame `k` tracks until it
drifts, then waits for the next human mark at `k+N`. The frames just *before*
`k+N` — where forward drift is worst — get the worst labels, even though there is
a perfectly good human anchor sitting one or two frames ahead of them.

Bidirectional tracking is the obvious fix and it is *only available offline*,
which is exactly our regime:

```
anchor k                                                    anchor k+N
  ●───────────────────────────────────────────────────────────●
  ├──▶ forward track (good near k, drifts toward k+N)
                                          backward track ◀──────┤
                                          (good near k+N, drifts toward k)
  └────────────── fuse the two estimates per frame ────────────┘
       weight by distance-to-nearest-anchor / per-frame confidence
```

Every interior frame is now bracketed by **two** estimates, each most reliable at
the end nearest its own anchor. Fusing them (confidence- or distance-weighted)
strictly dominates a single forward pass, and the residual gaps (mutual
low-confidence = occlusion) are exactly where a **trajectory prior** (parabolic /
constant-velocity) should interpolate. This is the single highest-value temporal
trick available to an *offline* autolabeller, and the current design uses none of
it.

---

## 2. Angle 1 — Models with native bidirectional tracking

| Model | Bidirectional? | Mechanism | Fit for us |
|---|---|---|---|
| **SAM2 (video predictor)** | **Yes, native** | `propagate_in_video(reverse=True)` walks frames `range(start_frame_idx, end, -1)`; memory bank is acausal in offline mode | **Best fit** — already promptable from a box, already a bake-off candidate (Method B/D family). Backward pass is one extra call. |
| **DEVA** (Tracking-Anything w/ Decoupled VOS, ICCV'23) | **Yes — bidirectional by design** | "task-agnostic *bi-directional* temporal propagation" + in-clip consensus over a short future window; XMem as the propagation core | Strong, but built around dense mask propagation of *generic* objects; heavier than we need for one tiny ball. Good source of the *fusion* idea. |
| **XMem** | Online/causal core, but used bidirectionally *inside* DEVA | Feature-memory propagation | Only bidirectional when wrapped (DEVA does the wrapping). |
| **SAM3** (incumbent) | Same video-predictor lineage as SAM2 | Expected to expose the same `reverse` propagation | Already the control/ceiling (Method D); a backward pass should be ~free to add. |
| **TrackNet / WASB** (heatmap, multi-frame) | Symmetric multi-frame, not "bidirectional" per se | Consumes a *window* of consecutive frames (e.g. 3) and outputs a center heatmap; learns the flying pattern, not just appearance | Different paradigm — a *detector* that already uses local temporal context. Relevant as a candidate-generator (§5), not as the bidirectional engine. |

**Takeaway:** we do not need to adopt a new model to get bidirectionality. The
segmenter family we are *already benchmarking* (SAM2/SAM3) provides it natively.
DEVA is the reference design for *how to fuse* the two directions, not a model we
must run.

Sources: [SAM2 issue #253 — "prompts from the future"](https://github.com/facebookresearch/sam2/issues/253),
[SAM2 video predictor source](https://github.com/facebookresearch/sam2),
[DEVA (ICCV 2023)](https://arxiv.org/abs/2309.03903),
[DEVA GitHub](https://github.com/hkchengrex/Tracking-Anything-with-DEVA).

---

## 3. Angle 2 — SAM2 backward propagation: confirmed and trivial

This is the load-bearing fact, so it is worth nailing precisely. From the SAM2
`sam2_video_predictor.py` source, the propagation entry point is:

```python
def propagate_in_video(
    self,
    inference_state,
    start_frame_idx=None,      # defaults to earliest frame with input points
    max_frame_num_to_track=None,
    reverse=False,             # ← the bidirectional switch
):
    ...
    # reverse=True  →  end = max(start_frame_idx - max_frame_num_to_track, 0)
    #                  for f in range(start_frame_idx, end - 1, -1):  ...
```

So the recipe to fill the interval `[k, k+N]` from a single human box at frame `k`
is exactly two calls:

```python
state = predictor.init_state(clip_frames)
predictor.add_new_points_or_box(state, frame_idx=k, box=human_gt_box, obj_id=1)
predictor.propagate_in_video(state, start_frame_idx=k, reverse=False)  # k → k+N
predictor.propagate_in_video(state, start_frame_idx=k, reverse=True)   # k → k-N
```

Two practical notes from the source / issue thread:

1. **Backward from frame 0 is skipped** ("skip reverse tracking if starting from
   frame 0") — irrelevant for us, anchors are interior.
2. The community-recommended pattern (issue #253) is precisely ours: **seed on the
   frame where the ball is biggest / least occluded**, then propagate *both ways*.
   This is the right anchor-selection heuristic — the human should be nudged to
   mark clean frames, and the labeller can then bracket each anchor in both
   directions.

The memory bank carries object identity in both temporal directions in offline
mode, so the backward pass is not a hack — it is a supported, documented mode.

**Cost:** one extra propagation pass per anchor interval. For an offline batch
labeller (no real-time constraint) this is negligible, and it *halves* the worst-
case drift distance (each frame is now at most `N/2` frames from its nearest
anchor, not `N`).

---

## 4. Angle 3 — Trajectory interpolation with ball physics

Two regimes, and they want different priors:

- **Ball in free flight** (pass, shot, clearance): disregarding air friction, the
  ground-plane velocity is constant and the ball follows a **single parabola**.
  In *image* space a 3D parabola projects to a smooth conic-ish arc; a low-order
  polynomial / spline fit per flight segment is the standard, well-validated model.
- **Ball rolling / dribbled / in a ruck**: motion is near-linear and slow; a
  constant-velocity Kalman prior (already in the design — `KalmanBoxState`) is the
  right model.

The classic broadcast-soccer literature does exactly this fusion:

- **Trajectory-based ball detection & tracking (Yu et al., and the broadcast-video
  line of work):** generate *many* ball **candidates** per frame (not one
  detection), build a **candidate graph across frames**, and find the trajectory
  as the smoothest physically-plausible path through that graph. Occlusion gaps
  are bridged by **interpolation between contiguous trajectory fragments**. This
  is structurally a **global, offline, bidirectional** optimisation — it uses
  future *and* past candidates to disambiguate the present. It directly handles
  our two hard cases: false positives (other white/round objects are pruned
  because they don't lie on a smooth trajectory) and occlusion (interpolated
  through).
- **Physics-based 3D position** work fits a parabola to the 2D trajectory to
  recover depth — over-kill for 2D autolabelling, but it confirms the parabola is
  the right interior model for flight.
- **Modern multi-mode state models** (real-time 3D reconstruction papers) keep
  centimetre accuracy through *severe* occlusion by switching between motion modes
  (flight vs. ground) — the practical lesson is **use a mode-switched motion model,
  not one global parabola**, because a match is a sequence of short flights and
  rolls, not one ballistic arc.

**For us this means:** after bidirectional propagation, the remaining holes (both
directions low-confidence ⇒ occluded) are filled by a **piecewise** trajectory
fit — parabolic within a detected flight segment, linear/CV otherwise — anchored
on both the human GT boxes *and* the high-confidence propagated boxes on either
side of the hole. The two human anchors that bracket an interval already give the
interpolation its endpoints; bidirectional propagation densifies the interior; the
trajectory fit only has to cover the genuinely-occluded residue.

Sources: [Trajectory-based ball detection & tracking in broadcast soccer](https://www.researchgate.net/publication/221572343),
[Physics-based 3D ball position from monocular sequences](https://www.researchgate.net/publication/3766301),
[Real-time 3D soccer ball trajectory reconstruction (multi-mode state model)](https://strathprints.strath.ac.uk/29273/1/1237_all.pdf),
[Ball trajectory inference w/ Set Transformer + Bi-LSTM](https://arxiv.org/pdf/2306.08206).

---

## 5. Angle 4 — Prior work on football ball tracking (bidirectional / smoothing)

The field has converged on a clear pattern for *broadcast* soccer/sports ball
tracking, and it is **not** "one detector per frame":

1. **Over-generate candidates, then resolve globally.** Broadcast ball tracking
   has been candidate-graph + trajectory-fitting since the mid-2000s precisely
   because per-frame detection of a tiny, blurred, occluded ball is unreliable —
   *the exact failure our [failure analysis](../ball_tracker_failure_analysis.md)
   reports (detection = 100% of misses, 21.6% recall)*. The trajectory layer is
   what makes an unreliable detector usable. This is strong external corroboration
   that **temporal/trajectory reasoning is the right lever for our bottleneck**,
   not just a nicety.
2. **Heatmap multi-frame detectors (TrackNet / WASB).** Instead of a box detector,
   regress a Gaussian heatmap of the ball centre from a *window* of consecutive
   frames, learning the flying pattern. TrackNet reports F1 ≈ 98.5% on tennis;
   WASB is the current SOTA heatmap family across several sports. These are
   *detectors that bake in short-range temporal context* — a far better
   candidate-generator for the trajectory layer than our single-frame YOLO, and a
   plausible answer to the detection bottleneck independent of the bake-off.
3. **Bidirectional sequence models.** Recent work (Set Transformer + hierarchical
   **Bi-LSTM**) *infers* ball position from multi-agent (player) context when the
   ball is unobservable — i.e. when even the trajectory is broken, players'
   trajectories constrain where the ball must be. This is the long-occlusion
   backstop, and it is inherently bidirectional (Bi-LSTM reads the sequence both
   ways).

**Takeaway:** every mature broadcast-ball system uses *more* temporal information
than we currently do, and the bidirectional/trajectory direction is the
mainstream, validated approach — not a speculative add-on.

Sources: [TrackNet (arXiv 1907.03698)](https://arxiv.org/abs/1907.03698),
[Ball trajectory inference (Set Transformer + Bi-LSTM)](https://arxiv.org/pdf/2306.08206),
[Real-time localization of a soccer ball from a single camera](https://arxiv.org/html/2506.07981v1),
[AWS ball-trajectory tracking in broadcast sports](https://aws.amazon.com/blogs/media/ball-trajectory-tracking-in-sports-broadcast-videos-using-aws-machine-learning/).

---

## 6. Angle 5 — Should this be "Method E" in the bake-off?

**No — and forcing it to be one would corrupt the bake-off.** Here is the precise
reason, grounded in our own code.

### The bake-off protocol is strictly causal

`ball_eval/interface.py` defines the contract every method must meet:

> *"One BallTracker instance is created per eval clip. The harness calls `track()`
> once per frame in **forward temporal order**."*

```python
def track(self, prev_bbox: BBox | None, frame: np.ndarray) -> BBox | None: ...
```

A method only ever sees the **current frame and the previous box**. It has **no
access to future frames, to the full clip, or to the downstream anchors**. This is
correct and deliberate: it measures *forward tracking quality*, which is what an
interactive labeller needs.

A bidirectional / trajectory-fusion method **fundamentally violates this**:

- It needs the **entire clip** (to propagate backward and fit trajectories).
- It needs **all the GT anchors at once** (the brackets it interpolates between).
- It cannot produce frame `f`'s label from `(prev_bbox, frame)` alone — its whole
  value comes from using frames `>f` and the anchor at `k+N`.

You cannot honestly score it on the per-frame, forward-only `frames_held` /
`center_distance` metrics, because it isn't doing per-frame forward tracking. If
you shoehorned it in (e.g. by pre-running both passes and replaying the fused
result through `track()`), it would post near-perfect numbers **by construction**
— it has *seen the answer at the next anchor* — and would not be comparable to the
honestly-causal A/C/D methods. It would not win a fair fight; it would refuse to
fight fair.

### What it actually is: a fusion / post-processing layer

Bidirectional + trajectory fusion is **orthogonal** to the per-frame runner
choice. It *wraps* whatever per-frame tracker the bake-off crowns:

```
        ┌─────────────────── bake-off scope (causal, per-frame) ───────────────────┐
        │  Method A/C/D runner: track(prev_bbox, frame) → box, in FORWARD order     │
        └──────────────────────────────────────────────────────────────────────────┘
                                          │  winner = best per-frame runner
                                          ▼
        ┌─────────────── Method E scope (offline, whole-clip) ──────────────────────┐
        │  for each anchor interval [k, k+N]:                                        │
        │     fwd  = run winner forward  from anchor k                               │
        │     bwd  = run winner backward from anchor k+N   (SAM2 reverse=True)       │
        │     fuse fwd/bwd per frame  (distance-to-anchor / confidence weighted)     │
        │     fit piecewise trajectory through residual occlusion holes              │
        └──────────────────────────────────────────────────────────────────────────┘
```

So "is it Method E?" is the wrong question. **The bake-off picks the per-frame
engine; bidirectional fusion is the offline harness that runs that engine twice
and stitches the results.** Adding it as a peer competitor would be a category
error.

### How to evaluate it instead

It needs its *own* metric, scored against held-out human GT it did **not** receive
as anchors:
- Hold back a subset of human-marked frames as hidden test points.
- Give the fusion layer only the *other* anchors.
- Measure centre-distance / IoU at the held-out frames → this is the **true
  autolabel quality between marks**, which is the number that actually matters for
  the [flywheel](./autolabelling_architecture_review.md#6-q4) (label quality →
  retrain set → recall lift).

This is a *leave-anchors-out* evaluation, not a forward-tracking race.

---

## 7. Recommendation

**Build the cheap version now; do not add it to the bake-off.** Ordered by
leverage:

| # | Priority | Action | Why |
|---|---|---|---|
| E1 | **P0** | Add a **backward propagation pass** to the SAM2/SAM3 labelling backend: for each anchor, call `propagate_in_video(reverse=True)` as well as forward, and **fuse** the two estimates per frame (weight by distance-to-nearest-anchor, break ties by confidence). | Native, ~free, halves worst-case drift. This is the single biggest offline win and uses a model already in the bake-off. Directly densifies GT between sparse marks — the thing we lack. |
| E2 | **P0** | Treat bidirectional fusion as a **post-processing layer wrapping the bake-off winner**, *not* a `BallTracker`. Keep the bake-off protocol strictly causal (do **not** add Method E to `interface.py`). | The protocol is forward-only by design; a whole-clip method can't be scored on it honestly (§6). Preserves the integrity of the A/C/D comparison. |
| E3 | **P1** | Add a **piecewise trajectory fit** (parabolic in flight segments, constant-velocity otherwise) to interpolate the residual holes where *both* directions are low-confidence (= occlusion). Anchor the fit on human GT **and** high-confidence propagated boxes either side of each hole. | Standard, validated broadcast-soccer technique (§4). Mode-switching beats one global parabola. Fills exactly the frames bidirectional propagation can't. |
| E4 | **P1** | Evaluate the fusion layer by **leave-anchors-out**: hide a subset of human marks, feed the rest, measure centre-distance at the hidden frames. Report this as *autolabel-quality-between-marks*. | This is the honest metric for a non-causal method, and it is the number the flywheel depends on — not a forward-tracking score. |
| E5 | **P2** | Evaluate a **multi-frame heatmap detector (TrackNet/WASB-style)** as a *candidate-generator* feeding the trajectory layer, independent of the bake-off. | Attacks the actual bottleneck — detection recall (failure analysis: 100% of misses). A temporal detector is a far better candidate source than single-frame YOLO. Separate track from the tracking bake-off. |
| E6 | **P2** | Backstop for long occlusions: a **bidirectional sequence model conditioned on player trajectories** (Bi-LSTM / Set-Transformer style) to *infer* ball position when even the trajectory is broken. | Only worth it once E1–E3 are in and long occlusions are the residual failure mode. Highest effort, narrowest payoff — defer. |

### One-line summary

**Bidirectionality is real, native (SAM2 `reverse=True`), and the mainstream
broadcast-ball approach — but it's a whole-clip *fusion layer that wraps the
bake-off winner*, not a Method E inside the forward-only bake-off.** Ship the
backward-pass + fuse step now (it directly densifies the sparse GT we're short
on), add piecewise-trajectory interpolation for occlusion holes, and score it
leave-anchors-out — keeping the causal A/C/D bake-off uncontaminated.

---

## Appendix: source grounding

- **SAM2 native backward propagation** — [`propagate_in_video(reverse=True)` source](https://github.com/facebookresearch/sam2), [issue #253 "prompts from the future"](https://github.com/facebookresearch/sam2/issues/253)
- **Bidirectional VOS design** — [DEVA, ICCV 2023](https://arxiv.org/abs/2309.03903) / [GitHub](https://github.com/hkchengrex/Tracking-Anything-with-DEVA)
- **Trajectory-based broadcast-soccer ball tracking** — [Trajectory-based ball detection & tracking](https://www.researchgate.net/publication/221572343), [Real-time 3D multi-mode reconstruction](https://strathprints.strath.ac.uk/29273/1/1237_all.pdf), [Physics-based 3D position](https://www.researchgate.net/publication/3766301)
- **Multi-frame heatmap detectors** — [TrackNet (arXiv 1907.03698)](https://arxiv.org/abs/1907.03698)
- **Player-conditioned bidirectional inference** — [Set Transformer + Bi-LSTM ball inference](https://arxiv.org/pdf/2306.08206)
- **Project context** — [autolabelling architecture review](./autolabelling_architecture_review.md), [SAM3 motion-guided design](./sam3_ball_labelling_gpu.md), [ball tracker failure analysis](../ball_tracker_failure_analysis.md)
- **Bake-off protocol (causal, forward-only)** — `src/footy_track/ball_eval/interface.py`, `src/footy_track/ball_eval/runner.py`
