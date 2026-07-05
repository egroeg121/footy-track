# SAM3 GPU Ball-Labelling: compute target & mark-once architecture

> Design doc for epic **ft-8vx** — *GPU-accelerated SAM3 ball-labelling loop: mark once, track high-res/high-fps*.
> Authored under **ft-mhk**. Child beads reference this doc by section number.

## 0. TL;DR / Decision

The original framing ("we need a bigger GPU to run full-frame SAM3 fast enough")
is **rejected as plan-of-record**. The bottleneck is not raw GPU FLOPs — it is
that we run a heavy segmentation model over the *entire frame* every step when
the ball occupies < 0.1% of the pixels.

**Plan of record: a motion-guided box-prompt tracking loop.**

1. Maintain a temporal state for the ball (position + velocity, via a
   constant-velocity Kalman filter).
2. Each step: predict where the ball will be, crop a **tight, high-resolution
   ROI** around that prediction.
3. Run a *swappable* runner (a small detector/segmenter/SOT tracker) on that
   crop only.
4. Refine the box, map it back to full-frame coordinates, update the temporal
   state, repeat.
5. On a miss (low confidence / anomaly), fall back to a **full-frame
   re-acquire** pass to relocate the ball, then resume the cheap loop.

This keeps *effective* resolution high (we feed the runner a crop that is mostly
ball) while keeping per-step cost tiny (a small crop, not a 4K frame). The
expensive full-frame model is reserved for the rare re-acquire.

**Compute target: local, in-process — RTX 2070 (8GB, Tailscale `desktop`) or Apple MPS.**
The motion-guided crop is cheap enough that 8GB is ample for the common path.
**AWS g4dn (ft-5yq, T4 16GB) is demoted to a fallback**, only justified if the
benchmark (§6 / ft-gx9) shows the chosen runner cannot hit target fps locally.

**Runner choice is a benchmark variable, not a design assumption** (§5, ft-gx9).
SAM3 becomes one *backend among several* and the high-quality fallback for
re-acquire — not the per-frame workhorse.

---

## 1. Problem & current pain

From the user (2026-06-20), the labelling loop is stuck:

- YOLO pre-load only works when the ball is obvious mid-pitch; it fails on
  small/distant/occluded balls and emits false positives (other round/white
  objects).
- **SAM3 can track the ball from a single marked point** — excellent UX: mark
  once, it follows.
- But SAM3 is **too slow per frame at full resolution** locally.
- Dropping SAM3's input resolution to gain speed **destroys tracking quality**
  (it loses the small ball).
- Dead end: high-res = accurate but unusably slow; low-res = fast but useless.

Fixing this unlocks two things at once:

1. **Labelling friction** (the immediate pain).
2. **Detection quality** — the corrected labels become training data to retrain
   the YOLO ball detector (ft-n2o).

### Why "just use a bigger GPU" is the wrong primary lever

Full-frame SAM3 at high res is expensive because it segments *everything*. A
T4/g4dn would buy maybe 2–3× throughput — still full-frame, still wasteful,
still cloud-latency-bound for an interactive loop, and it costs money per hour.
The ball is tiny and *moves predictably between adjacent frames*. Exploiting
that temporal prior is a far larger lever than more FLOPs, and it runs on
hardware we already own.

---

## 2. Architecture: motion-guided box-prompt loop

The core new component (ft-v43). Independent of any specific GPU — runs on
MPS/CPU/CUDA.

### 2.1 Per-step loop

```
state: KalmanBoxState  (cx, cy, w, h, vx, vy)
for each frame f:
    pred_box     = kalman.predict()                 # where do we expect the ball?
    roi          = expand(pred_box, margin)          # tight crop window, full-res pixels
    crop         = frame[roi]                         # high effective resolution
    det          = runner.track(crop, prompt=local(pred_box))  # swappable backend
    if det.confidence >= thresh:
        box_full = map_to_frame(det.box, roi)        # crop -> frame coords
        kalman.update(box_full)
        emit(f, box_full, provenance=runner.name)
    else:
        box_full = reacquire(frame)                  # full-frame fallback (§2.3)
        if box_full: kalman.reinit(box_full); emit(...)
        else:        flag_anomaly(f); pause()         # let the human correct
```

### 2.2 The crop is where the resolution win lives

If the ball is ~24px in a 3840px-wide frame, a full-frame model downsampled to
512px sees the ball at ~3px — invisible. A 256px crop around the predicted
position contains the ball at its **native ~24px** size. Same model, same
compute budget per call, but the ball is now large and clearly resolvable. This
is the mechanism that lets us "keep high res" without paying full-frame cost.

ROI sizing is the key tunable (ft-v43 / ft-gx9): too tight and a fast ball
escapes the crop between frames; too loose and we lose the resolution advantage.
Drive margin from predicted velocity (bigger crop when the ball is moving fast).

### 2.3 Full-frame re-acquire (the rare path)

When the runner reports a miss (low confidence, or the anomaly detector fires —
see `_track_anomaly_reason`), we cannot trust the temporal prior. Run a
full-frame pass to relocate the ball:

- Cheapest option: full-frame YOLO ball detector (existing `yolo_seed_objects`).
- High-quality fallback: full-frame SAM3 (`Sam3VideoLabeller`) — the existing
  backend, now used *only* here, where its cost is amortised over many cheap
  steps.

If re-acquire also fails, stop and surface an anomaly so the human marks the
ball again (the existing pause/anomaly UX already does this).

### 2.4 FrameTracker / CropRunner interface (ft-v43)

Two small protocols decouple the loop from the model:

```python
class CropRunner(Protocol):
    name: str                       # provenance tag
    def warmup(self) -> None: ...
    def detect(self, crop: np.ndarray, prior: Box | None) -> Detection: ...
    # returns box in CROP coordinates + confidence

class FrameTracker(Protocol):
    def reset(self, frame: np.ndarray, seed: Box) -> None: ...
    def step(self, frame: np.ndarray) -> Detection | None: ...  # FRAME coords
```

`MotionGuidedTracker` implements `FrameTracker` and owns the Kalman state, the
crop logic, the coord round-trip, and the re-acquire fallback. `CropRunner` is
the swappable backend benchmarked in §5/§6. Existing `Sam3VideoLabeller` is
refactored to also satisfy `CropRunner` (as the `sam3-fullframe` backend) so it
stays usable for re-acquire and as a quality baseline.

**Seam into existing code:** `video_utils.py:407-410` — today
`set_prompts({"bboxes": ...})` then streams the predictor over the whole video.
The new tracker replaces that streaming call; `BackgroundLabeller.submit` gains
a `backend=` argument selecting the runner (§7).

---

## 3. Compute targets evaluated

| Target | VRAM | Pros | Cons |
|---|---|---|---|
| **RTX 2070 `desktop`** (Tailscale, Windows) | 8GB | Owned, free, local; fine for cropped runner; in-process or short-hop | 8GB tight for *full-frame* SAM3 high-res; Tailscale hop adds latency if remote-driven |
| **Apple MPS** (dev laptop) | shared | Zero setup, in-process, no network | Slower than CUDA; some ops fall back to CPU |
| **AWS g4dn** (ft-5yq, T4) | 16GB | Headroom for full-frame SAM3; clean Linux CUDA | Costs $/hr; cloud round-trip kills interactivity; needs the blocked support case |

### Verdict

The motion-guided loop makes per-step compute small, so **8GB (RTX 2070) or MPS
is sufficient for the common path**. We keep the work **local and in-process**
to avoid network latency in the interactive correction loop. **g4dn is a
fallback** — pursued only if §6 benchmarks show the selected runner can't hit
target fps locally, or if we later want batched offline re-labelling of large
archives. ft-5yq stays open as that fallback but is **no longer plan-of-record**.

### Rough VRAM / fps expectations (to be confirmed by ft-gx9)

- Full-frame SAM3 high-res on 8GB: borderline OOM risk and single-digit fps —
  this is exactly the pain we're routing around.
- Cropped runner (256–512px crop) on 8GB: comfortably fits; small SOT trackers
  / ROI-YOLO / SAM2.1-tiny should reach interactive fps. **Numbers TBD — see §6.**

### 3.1 SAM3 footprint & throughput — external research (reasoned estimates)

Hard local numbers on our two targets come from ft-gx9. Until then, these are
grounded in published figures + FLOP/pixel scaling. **Assumptions stated; treat
as order-of-magnitude, not benchmark truth.**

**Model size (published).** SAM3 is ~840M params, ~3.4 GB weights. Single-image
inference reportedly needs **~8–10 GB VRAM in fp16** for the image encoder —
i.e. comfortable on the T4 (16 GB), **tight-to-borderline on the RTX 2070
(8 GB)** once activations for a high-res frame are added. This is the concrete
basis for "8 GB is tight for full-frame SAM3" above.

**Throughput (published, big GPUs).** SAM3 is *notably slower than SAM2*:
~1.1 s/image on an RTX 4090 (~600×500, warm), and only **5–6 fps at 1080p on an
H200** where SAM2 clears 30 fps. Full-frame SAM3 is simply not an interactive
per-frame model on any GPU we can afford — **this is the single strongest
justification for the §0 route-around**, not just a local-hardware limitation.

**Reasoned estimates for our targets (full-frame, fp16, stated assumptions):**

| Target | ~fp16 TFLOPS (tensor) | Est. full-frame SAM3 | Est. VRAM full-frame |
|---|---|---|---|
| RTX 2070 (8 GB) | ~55 (Turing) | **~0.5–1.5 fps** (~0.7–2 s/frame); scale from 4090's ~1.1 s by ~3–5× fewer usable TFLOPS | ~8–10 GB → **OOM risk at high res**; needs fp16 + modest res to fit |
| AWS g4dn / T4 (16 GB) | ~65 (but bandwidth-bound) | **~1–2 fps** full-frame; T4 is memory-bandwidth-starved so real-world closer to the low end | ~8–10 GB → **fits with headroom** |

*Assumption:* SAM3 latency is encoder-dominated and scales ~linearly with input
token count (∝ pixels), and Turing/T4 realise a fraction of peak TFLOPS on this
workload — hence the conservative fps.

**Cropped estimates (the path we actually take).** A 512 px crop is ~1/8 the
pixels of 1080p; 256 px ~1/32. Encoder cost scales roughly with pixels, so a
cropped SAM-family runner should reach **~5–15 fps on either target in fp16** —
into the interactive band — while *keeping* effective resolution high (the crop
is mostly ball, not a downscaled full frame that loses the small ball). This is
exactly why cropping beats downscaling for question §1's dead-end.

**SAM3.1 base as a viable fallback backend.** SAM3.1 reportedly ~doubles
throughput (~16→32 fps class) and roughly halves VRAM (~8→4 GB fp16) vs SAM3.
At ~4 GB it fits the RTX 2070 with room to spare, making a *cropped* SAM3.1
runner a credible re-acquire/quality backend even on 8 GB — worth including as a
candidate in ft-gx9 (subject to the ft-f74 whole-frame-mask caveat).

*Sources: Meta SAM3/SAM3.1 release notes, facebookresearch/sam3 issues #424/#425,
Roboflow & Spheron deployment write-ups, Ultralytics SAM3 docs. All external;
confirm on our hardware in ft-gx9.*

---

## 4. Optimisation levers

Applied to the per-step runner and the re-acquire pass:

- **ROI cropping** (the big one, §2) — turns a full-frame problem into a small-crop one.
- **fp16 / autocast** — halve memory & speed up on both T4 and RTX 2070.
- **torch.compile** — fuse the runner's forward pass; measure warmup vs steady-state.
- **Batching** — for offline (non-interactive) re-labelling, batch crops across
  frames. Not applicable to the live one-frame-at-a-time correction loop.
- **Warm predictor** — keep the model resident (the codebase already caches a
  hot predictor: `get_cached_predictor` / `warmup_model` / `start_warmup_thread`).
- **Smaller checkpoints** — tiny SOT trackers / SAM2.1-tiny / MobileSAM as the
  per-step runner; reserve heavy SAM3 for re-acquire only.

---

## 5. Runner matrix (ft-gx9) — runner is a first-class variable

The per-step `CropRunner` is chosen by benchmark, not assumption. Candidates:

- **SOT trackers**: OSTrack, MixFormer, LightTrack, SiamRPN++ — purpose-built
  for "track this one box across frames", typically fast and small.
- **ROI-YOLO**: run the existing/retrained YOLO ball detector on the crop only.
- **SAM family**: Ultralytics SAM3, Meta `sam2_video_predictor`, HF SAM2/3,
  SAM2.1-tiny, EdgeTAM, MobileSAM, FastSAM.

**Default bet pending numbers:** a SOT tracker *or* ROI-YOLO on the
motion-guided crop, in-process on RTX 2070 / MPS. SAM3 = high-quality fallback
backend for re-acquire and as the quality ceiling.

**Checkpoint note:** use `sam3.pt` base, **not** `sam3.1_multiplex` (known
whole-frame mask bug — tracked separately in ft-f74).

---

## 6. Benchmark plan (ft-gx9)

Produce a **runner × resolution × fps × quality** matrix:

- **Axes**: runner (§5) × crop size (128/256/512/full) × precision (fp32/fp16)
  × device (MPS / RTX 2070 / [g4dn if needed]).
- **Metrics**: fps (steady-state), peak VRAM, tracking quality (IoU vs
  hand-corrected ground truth, % frames the ball is held vs lost), re-acquire
  frequency.
- **Pass bar**: interactive fps (target ≥ ~15fps on the common cropped path)
  with tracking quality good enough that human correction is occasional, not
  per-frame.
- **Output**: a table in `docs/training/` or alongside this doc, plus a
  recommended default `(runner, crop, precision, device)`.

This benchmark is what *confirms or overturns* the §0 decision (and decides
whether ft-5yq/g4dn is needed at all).

---

## 7. Integration into the EXISTING app (no new app)

The labelling app already exists and was confirmed by reading the code. We
**plug into it**; we do not build a new UI or service.

### 7.1 What already exists

- `labeller/server.py` — FastAPI: timeline + provenance
  (`PROV_LABELLER` = manual ground truth, `PROV_YOLO`, `PROV_SAM3`), WebSocket
  run/restart/pause, `/autodetect` `/edit` `/timeline` `/frame`, anomaly push.
  **`PROV_LABELLER` boxes are ground truth and survive auto re-propagation.**
- `web/index.html` — the annotation UI (mark point, correct boxes, scrub).
- `video_utils.py` — `Sam3VideoLabeller`, `BackgroundLabeller`,
  `_track_anomaly_reason`, hot-predictor cache.

`app.py` is the **legacy Streamlit** prototype and is **retiring** — do not
build on it.

### 7.2 Where the new tracker plugs in (ft-rwg)

- New `FrameTracker`/`CropRunner` classes live in `video_utils.py` (or a sibling
  module), implementing §2.4.
- `BackgroundLabeller.submit` gains a **`backend=`** argument selecting the
  runner; default can stay `sam3-fullframe` until benchmarks pick a winner.
- The worker swaps the seam at `video_utils.py:407-410` (the
  `set_prompts` → predictor stream) for the motion-guided loop when a
  crop-runner backend is selected.
- **`server.py`, the WebSocket protocol, and `index.html` are UNCHANGED.** New
  runner provenance tags slot alongside `PROV_SAM3` / `PROV_YOLO`; manual
  `PROV_LABELLER` corrections remain ground truth.

### 7.3 Provenance & the human-in-the-loop

The existing pause/anomaly/correct flow is preserved: tracker miss → anomaly →
pause → human re-marks → `submit(..., start_frame=N)` resumes propagation with
corrected seed. Corrected (`PROV_LABELLER`) boxes are never overwritten by
auto-propagation — they are the training-data ground truth.

---

## 8. Storage / training-data export (ft-0y4)

**Use the existing Roboflow export — do not build a new export tool.**
`labelling.py` already has `RoboflowObjectDetectionHandler.upload_images`
(COCO format, `ROBOFLOW_API_KEY`). Corrected labels flow:

```
human-corrected timeline (PROV_LABELLER boxes)
  -> frames + COCO annotations
  -> RoboflowObjectDetectionHandler.upload_images
  -> Roboflow dataset
  -> retrain ball YOLO (ft-n2o)
```

The feature store (ft-4jr) is for *runtime tracking/event data*, not raw
training-image export. Training images/labels go to **Roboflow** via the
existing handler. ft-0y4 wires the corrected-timeline → COCO → upload path; no
new infrastructure.

---

## 9. Child-bead breakdown

All children already exist under epic **ft-8vx** and reference this doc:

| Bead | Title | Maps to |
|---|---|---|
| **ft-gx9** | Benchmark runner × res × fps × quality matrix | §5, §6 |
| **ft-v43** | Motion-guided box-prompt tracker + `FrameTracker` interface (**CORE**) | §2, §4 |
| **ft-rwg** | Wire motion-guided tracker into EXISTING labeller server | §7 |
| **ft-0y4** | Corrected labels → EXISTING Roboflow export (`labelling.py`) | §8 |
| **ft-n2o** | Retrain ball YOLO on SAM3-generated labels; measure recall lift | §1, §8 |
| **ft-f74** | Re-evaluate SAM3.1 base checkpoint (whole-frame-mask bug?) | §5 |

**Dependency order:** ft-v43 (core tracker + interface) → ft-gx9 (benchmark
backends through the interface) → ft-rwg (wire chosen backend into server) →
labelling produces data → ft-0y4 (export) → ft-n2o (retrain). ft-f74 is
independent and informs which SAM checkpoint ft-gx9 uses.

**Fallback bead:** ft-5yq (g4dn T4) stays open but is **demoted to fallback** —
pursue only if ft-gx9 shows the local target can't hit interactive fps.

---

## 10. Open questions / risks

- **Fast-ball ROI escape**: if the ball leaves the predicted crop between
  frames, the runner misses. Mitigation: velocity-scaled crop margin + prompt
  re-acquire. Validate on fast clips in ft-gx9.
- **Occlusion / players**: ball behind a player → temporary miss → re-acquire
  may lock onto a wrong round object. The anomaly detector + human correction
  is the safety net; measure false-lock rate.
- **Tailscale latency**: if we *remote-drive* the RTX 2070 rather than run the
  UI on the desktop box, the per-frame hop may hurt interactivity. Prefer
  in-process; if remote, batch and stream rather than per-frame RPC.
- **Benchmark could overturn §0**: if no local runner hits the fps bar at
  acceptable quality, g4dn (ft-5yq) returns as plan-of-record. The decision is
  explicitly contingent on ft-gx9 numbers.
