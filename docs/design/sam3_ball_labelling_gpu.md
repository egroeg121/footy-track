# Motion-Guided Box-Prompted Ball-Labelling Loop — Design

**Bead:** ft-8vx (epic) · **Author:** polecat onyx · **Date:** 2026-06-21
**Status:** design (v2 — reframed per mayor 2026-06-21).
Picks an **approach** (motion-guided box-prompt), makes **runner choice** a
first-class variable, integrates into the **existing labeller web app**, and
breaks the build into child beads.

> **v2 changes (mayor reframe):** v1 led with "run heavy full-frame SAM3 on a big
> GPU." v2 leads with a **motion-guided box-prompted loop** that uses a temporal
> prior (Kalman/velocity) to predict a tight ROI each frame and prompt a
> *swappable, possibly lightweight* tracker on that crop. This likely removes the
> need for a g4dn entirely. The loop plugs into the **existing** labeller
> (`labeller/server.py` + `web/` + `video_utils.py`) and exports through the
> **existing** Roboflow path (`labelling.py`) — not a new tool.

## 1. Problem

The "mark once → propagate → human corrects" labelling loop is the critical path
for footy ball work, because **detection — not tracking — is the bottleneck**: the
ft-55d.1 analysis (`docs/ball_tracker_failure_analysis.md`) found YOLO misses the
ball in **64–93 % of frames**, and the LAP tracker dropped *zero* detections YOLO
actually produced. The cheapest way to fix detection is to generate clean ball
training data with the mark-once loop and retrain YOLO.

The loop is stuck on speed vs accuracy:
- At `imgsz=512` (current default in `labeller/video_utils.py`) SAM3 runs at a
  usable rate locally but **loses the small/distant ball** (a 10–20 px ball in
  1080p becomes 3–6 px at 512 — below the decoder's reach).
- At full resolution SAM3 is accurate but **unusably slow** — and SAM3 video is
  intrinsically slow: **5–6 fps @ 1080p even on an H200** (facebookresearch/sam3
  #425). Full-frame SAM3 is never "high fps" anywhere affordable.

## 2. The reframe: motion-guided box-prompt, not heavy full-frame SAM3

A football is **one small, fast, ballistic object**. Between adjacent frames its
motion is highly predictable. So instead of asking a heavy model to re-find it in
the whole frame every time, exploit the temporal prior:

```
   ┌────────────────────────────────────────────────────────┐
   │  prev_bbox + velocity  ──►  Kalman/constant-velocity     │
   │                              predict ball position       │
   │                                     │                    │
   │                                     ▼                    │
   │        crop a TIGHT high-res ROI around the prediction   │
   │        (window ≈ 6–8× ball size, clamped to frame)       │
   │                                     │                    │
   │                                     ▼                    │
   │   prompt a SWAPPABLE tracker on the crop (box prompt):   │
   │     • box-prompted SAM2/SAM3  • ROI-YOLO                 │
   │     • single-object tracker (OSTrack/MixFormer/          │
   │       LightTrack/SiamRPN++)                              │
   │                                     │                    │
   │                                     ▼                    │
   │     refined box  ──►  map back to frame coords  ──►       │
   │     update Kalman state  ──►  next frame                 │
   └────────────────────────────────────────────────────────┘
   miss / low-conf / anomaly  ──►  widen ROI or full-frame re-acquire for 1 frame
```

Why this wins:
- **Tiny input.** A 256–384 px crop, not a 1080p frame. The ball fills a large
  fraction of the model input → **high effective resolution = accuracy**, while the
  per-frame compute is a fraction of full-frame → **high fps**. This dissolves the
  high-res-XOR-high-fps trap.
- **Lighter runner becomes viable.** Because the input is small, even a heavy model
  is cheap per frame, *and* lightweight single-object trackers (OSTrack, LightTrack,
  SiamRPN++) run at hundreds of fps on a crop. The temporal prior likely makes the
  **8 GB RTX 2070 — or even CPU/MPS — sufficient, removing the g4dn dependency.**
- **Robustness from the prior.** The Kalman prediction gives the tracker a strong
  starting guess and bounds the search region, which directly suppresses the
  classic SAM3 failure (ball mask snapping to a distant marking) that the existing
  anomaly auto-stop currently catches *after the fact*.

SAM3 from-a-single-point is still the **seeding UX** (mark once, it follows) and a
valid *tracker backend* — but it is now **one option behind a swappable interface**,
not the architecture. Runner and tracker choice are decided empirically (§5).

## 3. What already exists (the asset — do NOT rebuild)

The labeller web app is **already built**. Confirmed by reading the files:

| Component | File | Role |
|---|---|---|
| **FastAPI server** | `labeller/server.py` | Live UI backend: per-frame `timeline`, **provenance tags** (`PROV_LABELLER`/`PROV_YOLO`/`PROV_SAM3` — manual edits are ground truth and survive re-propagation), WS `run`/`restart`/`pause`, `/autodetect`, `/edit`, `/timeline/{idx}`, `/frame/{idx}.jpg`, anomaly push. |
| **Frontend** | `labeller/web/index.html` | Canvas UI; WS protocol `run`/`restart` → `frame`/`anomaly`/`status`. |
| **Propagation engine** | `labeller/video_utils.py` | `Sam3VideoLabeller.iter_frames_from()`, `BackgroundLabeller` (threaded run/pause, `completed_frames()`), `_track_anomaly_reason` (jump >40 % / area >8×), `get_cached_predictor`/`warmup_model` (hot model). |
| **Streamlit app** | `labeller/app.py` | **Legacy** UI; being retired in favour of the web app. Do not extend. |
| **Roboflow export** | `labelling.py` | `RoboflowObjectDetectionHandler.upload_images()` → COCO JSON → Roboflow (`ROBOFLOW_API_KEY`). Detector: `UltralyticsSam3Detector`. |

**The integration seam is precise.** Propagation is invoked at
`video_utils.py:407` (`predictor.set_prompts({"bboxes": …})`) → `:410`
(`predictor(source=…, stream=True)`), per frame converted by `_result_to_frame`
(`:519`). The server streams `BackgroundLabeller.completed_frames()` and merges
them provenance-aware (`server.py:273 _ingest_completed_frame`, `:106
merge_propagated`). **Everything upstream/downstream of the tracker stays
unchanged** — the motion-guided tracker is a drop-in behind this seam.

## 4. Where the tracker plugs in (integration design)

Introduce a small **tracker-backend interface** and make `Sam3VideoLabeller` one
implementation of it; the motion-guided loop is another. The server talks to
`BackgroundLabeller` exactly as today.

```python
# labeller/trackers/base.py  (new)
class FrameTracker(Protocol):
    def seed(self, frame0, objects: list[LabelledObject]) -> None: ...
    def step(self, frame) -> list[ObjectDetection]: ...   # per-frame boxes (normalized)
```

- **`Sam3VideoLabeller`** is refactored to satisfy `FrameTracker` (it already does
  seed-then-iterate); the existing whole-clip path is preserved as the
  `sam3-fullframe` backend for comparison/fallback.
- **`MotionGuidedTracker`** (new, ft-v43): holds per-object Kalman state; `step()`
  = predict ROI → crop → call a **`CropRunner`** → refine box → update state →
  full-frame re-acquire on miss. The `CropRunner` is the swappable runner (§5).
- **`BackgroundLabeller.submit(...)`** gains a `backend=` arg (default chosen by
  config/benchmark). Its `_worker` loop already tracks `prev_fd` for anomaly
  detection — that is exactly the hook to feed velocity. **No server or WS change**
  for v1: it keeps streaming `completed_frames()`; the anomaly path still works
  (and improves, since the prior prevents most snaps).
- **Runner location** is a deployment detail behind `CropRunner`: in-process
  (CPU/MPS/local CUDA) or a thin HTTP call to a GPU box over Tailscale. Because the
  crop is tiny, **in-process local is the v1 target**; the remote path is a config
  swap if a benchmark says a clip needs it.
- **Export is unchanged**: corrected `timeline` frames already carry
  `PROV_LABELLER` ground truth; wire `export_frames_json` → the existing
  `RoboflowObjectDetectionHandler.upload_images()` COCO path (ft-0y4). No new
  export tool.

## 5. Decision: runner/tracker is a first-class benchmark variable

Per the mayor, **the runner/implementation is not assumed — it is measured.** The
benchmark (ft-gx9) evaluates, on a **fixed ROI-crop harness**, this matrix:

| Runner / tracker | Class | Why a candidate |
|---|---|---|
| Ultralytics SAM3 (`sam3.pt`) box-prompt | heavy, current | Baseline; what the repo uses today. |
| Official Meta `sam2_video_predictor` | heavy | Reference impl; may differ in speed/quality from Ultralytics wrapper. |
| HF transformers SAM2/SAM3 | heavy | Easiest to deploy/quantise; compare wrapper overhead. |
| **SAM2.1-tiny** | light | Small SAM that may hold a crop at high res on 8 GB / CPU. |
| **EdgeTAM** | light | Edge-optimised SAM2-style video tracker — strong fit for crop+8 GB. |
| **MobileSAM / FastSAM** | light | Mask backbones that may suffice on a tight crop. |
| **OSTrack / MixFormer / LightTrack / SiamRPN++** | SOT | Purpose-built single-object box trackers; hundreds of fps on a crop, no mask head needed (we only need a box). Likely the fps winners. |
| ROI-YOLO | detector | Run the *existing* YOLO on the predicted crop at high effective res — reuses our model, no new dependency. |

Report per runner: **VRAM, fps, and ball-IoU vs hand labels** on 3–5 occlusion
clips, at full-frame vs ROI-crop, FP16 vs FP32. Key question the matrix answers:
**does a light runner on a motion-guided crop beat heavy full-frame SAM3 — and does
it fit the 8 GB RTX 2070 / CPU, killing the g4dn need?**

**Default bet pending numbers:** motion-guided ROI-crop with a **single-object
tracker (OSTrack/LightTrack) or ROI-YOLO** as the per-frame runner, **box-prompted
SAM2.1-tiny/EdgeTAM** if a mask is wanted, **local in-process (RTX 2070 or MPS)**.
SAM3 stays as the high-quality fallback backend. g4dn (ft-5yq) becomes a
**fallback**, not the plan of record.

### SAM model config (when a SAM backend is used)
- `sam3.pt` base — **not** `sam3.1_multiplex.pt` (documented whole-frame-mask bug on
  our footage, `_default_model_uri`). FP16 on CUDA. Re-evaluate **base SAM3.1**
  separately (ft-f74) — 32 fps / 4 GB FP16 would change the math.

## 6. End-to-end loop (in the existing UI)

```
Existing labeller web UI (labeller/server.py + web/index.html)
  │  user marks ONE point / box (or text hint "soccer ball") on a frame
  │  WS "run" {start_frame, conf, ...}            (unchanged protocol)
  ▼
BackgroundLabeller.submit(backend=motion_guided)   (new backend arg)
  • MotionGuidedTracker: Kalman ROI → crop → CropRunner(step) → refine
  • full-frame re-acquire on miss; existing anomaly auto-stop as backstop
  ▼
completed_frames() streamed over WS as today → canvas overlays boxes
  │  human scrubs, corrects a box (/edit → PROV_LABELLER), "restart from here"
  ▼
timeline (ground-truth corrected) → export_frames_json
  ▼
RoboflowObjectDetectionHandler.upload_images()  (existing, COCO, ROBOFLOW_API_KEY)
  ▼
retrain ball YOLO  → recover the 78 % miss rate
```

## 7. Child-bead breakdown (revised)

Dependency direction = "X needs Y". Children of ft-8vx.

1. **ft-gx9 — Runner × resolution × fps × quality benchmark matrix.**
   On a fixed ROI-crop harness, benchmark the §5 runners (Ultralytics SAM3, Meta
   official, HF SAM2/3, SAM2.1-tiny, EdgeTAM, MobileSAM, FastSAM, OSTrack/
   MixFormer/LightTrack/SiamRPN++, ROI-YOLO) on RTX 2070 vs CPU/MPS (and g4dn only
   if needed). Report VRAM, fps, ball-IoU on 3–5 occlusion clips; recommend the
   default runner + whether g4dn is needed. **P1.** *Independent — can start now;
   g4dn (ft-5yq) only if a heavy runner proves necessary (dependency softened).*

2. **ft-v43 — Motion-guided box-prompt tracker + `FrameTracker` interface.**
   Define `FrameTracker`/`CropRunner`; refactor `Sam3VideoLabeller` to satisfy it;
   build `MotionGuidedTracker` (Kalman/constant-velocity ROI prediction → tight
   high-res crop → swappable runner → refine → re-acquire on miss). Unit-test
   crop↔frame coord round-trip and Kalman predict/update. **The core new
   component.** P1. *Independent of GPU; runs on MPS/CPU.*

3. **ft-rwg — Wire the tracker into the existing labeller server (no new app).**
   Add `backend=` to `BackgroundLabeller.submit`; plug `MotionGuidedTracker` in so
   the existing WS `run`/`restart`/`pause`, timeline provenance, and anomaly push
   work unchanged. Optional thin `CropRunner` HTTP shim for a remote GPU box over
   Tailscale (config-gated). **Extend `server.py`, do not replace it.** P1. *Needs
   ft-v43; ft-gx9 informs the default backend.*

4. **ft-0y4 — Corrected labels → existing Roboflow export.**
   Wire `export_frames_json` (ground-truth `timeline`) into
   `RoboflowObjectDetectionHandler.upload_images()` (COCO, `ROBOFLOW_API_KEY`)
   idempotently (same clip re-export = no dupes). Reuse `labelling.py`; don't add a
   new export path. P2. *Needs ft-rwg.*

5. **ft-n2o — Retrain ball YOLO on loop-generated labels.**
   Fine-tune the ball detector on the exported data; measure recall lift vs the
   ft-55d.1 baseline (21.6 %). P1. *Needs ft-0y4.*

6. **ft-f74 (fast-follow) — Re-evaluate base SAM3.1.**
   Test the **base** SAM3.1 checkpoint (not `_multiplex`) for the whole-frame-mask
   bug on our footage. 32 fps / 4 GB FP16 would make it a strong light-ish SAM
   backend. P2.

## 8. Acceptance (this epic)

- ✅ **Approach chosen:** motion-guided box-prompted ROI loop (Kalman → tight crop →
  swappable runner → refine), leading over heavy full-frame SAM3.
- ✅ **Runner is first-class:** benchmarked as a variable (§5) across SAM
  implementations and single-object trackers; likely removes the g4dn need.
- ✅ **Integrates into the existing labeller** (`server.py` + `web/` +
  `video_utils.py`) and **existing Roboflow export** (`labelling.py`) — not a new
  tool; concrete seams identified (§3–§4).
- ✅ **Child-bead breakdown** (§7).

## 9. Risks / open questions

- **ROI escape on fast shots:** if the ball leaves the predicted window faster than
  Kalman re-centres (clearances, shots), we miss → widen/full-frame re-acquire.
  Window size (6–8× ball) + acceleration term in the filter are mitigations;
  ft-gx9 must stress shot/clearance clips.
- **SOT trackers drift through occlusion:** single-object trackers can lock onto
  distractors when the ball is hidden. Mitigation: the existing anomaly auto-stop
  as a backstop, and SAM/ROI-YOLO re-acquire frames. ft-gx9 reports occlusion IoU.
- **RTX 2070 vs Ollama VRAM:** even a light runner competes with `qwen3:8b` (~5.5 GB)
  on the shared 8 GB box; CPU/MPS in-process may sidestep this. ft-gx9 measures.
- **Interface refactor risk:** pulling `Sam3VideoLabeller` behind `FrameTracker`
  must not regress the working SAM3 path — keep `sam3-fullframe` as a backend and
  gate the default by config.
- **SAM3.1 whole-frame bug:** only `_multiplex` confirmed bad; base untested
  (ft-f74). Stay on `sam3.pt` until then.
