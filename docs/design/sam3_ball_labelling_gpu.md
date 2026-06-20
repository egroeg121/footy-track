# GPU-Accelerated SAM3 Ball-Labelling Loop — Design

**Bead:** ft-8vx (epic) · **Author:** polecat onyx · **Date:** 2026-06-21
**Status:** design — picks a compute target + SAM3 config and breaks the build into child beads.

## 1. Problem

The "mark once → SAM3 propagates → human corrects" labelling loop is the critical
path for footy ball work, because **detection — not tracking — is the bottleneck**.
The ball-tracker failure analysis (ft-55d.1, `docs/ball_tracker_failure_analysis.md`)
found YOLO misses the ball in **64–93 % of frames**, and the LAP tracker dropped
*zero* detections YOLO actually produced. We cannot fix detection without more and
better ball training data, and the cheapest way to generate that data is the
mark-once SAM3 loop. So this loop unblocks both the immediate labelling friction
**and** the YOLO retrain (ft-55d.2, ft-4jr).

The loop is stuck on a speed/accuracy trade-off:

- At `imgsz=512` (current default in `labeller/video_utils.py`) SAM3 runs at a
  usable rate locally (Apple Silicon MPS) but **loses the small/distant ball** — a
  ball that is ~10–20 px in a 1080p frame becomes ~3–6 px at 512, below what the
  mask decoder can hold onto.
- At full resolution SAM3 is accurate but **unusably slow** locally.

## 2. What we already have (the asset)

This is *not* a greenfield build. `src/footy_track/labeller/video_utils.py`
already implements the hard parts:

- `Sam3VideoLabeller` — drives `SAM3VideoPredictor`, maps object-index → class label.
- Point / box / **text-hint** seeding (`LabelledObject`, `_resolve_text_hints`
  via `SAM3SemanticPredictor`). Mark-once-by-point already works.
- `iter_frames_from(start_frame, …)` — pause → correct → **resume from here**, the
  core correction workflow, by re-seeding propagation on a temp sub-clip.
- `BackgroundLabeller` — threaded run/pause, anomaly auto-stop
  (`_track_anomaly_reason`: flags the classic "ball snaps to a distant marking"
  failure when a box jumps >40 % of the frame or resizes >8×).
- `get_cached_predictor` + `warmup_model` — model stays hot across runs, paying the
  `torch.compile` cost once (cache pinned to `~/.cache/footy_torch_inductor`).
- A FastAPI/Konva web-app design (`docs/labeller_web_app_design.md`) that promotes
  `BackgroundLabeller` to a server-owned singleton.

**The backend carries straight over.** What's missing is (a) a GPU fast enough to
run SAM3 at high effective resolution, and (b) an ROI-crop path so we get that
resolution *on the ball* without paying for the whole 1080p frame.

## 3. SAM3 performance facts (researched 2026-06-21)

| Fact | Source | Implication |
|---|---|---|
| SAM3 video is **5–6 FPS @ 1080p even on an H200**; SAM2 does 30+ on the same box | facebookresearch/sam3 #425, #155 | The slowness is **intrinsic to SAM3 at full res**, not just our local hardware. Full-frame 1080p SAM3 will never be "high fps" anywhere affordable. |
| SAM3 image encoder ≈ **8–10 GB VRAM** FP16; idle 10–12 GB with prompt/mask heads | Spheron deploy guide | **Does not fit the 8 GB RTX 2070** at full res. The home GPU is inadequate for full-frame SAM3. |
| Latency **scales with object count**; ~5 objects ≈ near-real-time | Spheron | Ball labelling tracks **1 object** — we are at the cheapest end of the curve. This is what makes ROI-crop viable. |
| **SAM3.1** hits 32 FPS @ ~4 GB VRAM FP16 | Medium (Chavan, 2026) | Attractive *but* the repo deliberately avoids `sam3.1_multiplex.pt` — it returns whole-frame masks on our footage (see `_default_model_uri` docstring). SAM3.1 is a **fast-follow to re-evaluate**, not the v1 bet. |
| ROI-crop around predicted position is the standard small-object-at-high-res trick | SAM2 small-object literature | Crop a window around the last ball box, run SAM3 on the crop at high effective res, map the mask back. Keeps the ball large in the model's input without paying for the full frame. |

**The decisive insight:** the ball is one small object. We should never feed SAM3
the whole 1080p frame. Crop a ~256–384 px window around the predicted ball
position and run SAM3 on *that* at full window resolution. The ball then occupies
a large fraction of the model input (high effective res = accuracy) while the
per-frame cost is a fraction of full-frame (high fps). This converts the
"high-res XOR high-fps" dilemma into "high-res AND high-fps" for the single-ball
case — which is exactly the labelling use case.

## 4. Decision: compute target

Two candidates from the bead:

| Option | VRAM | Verdict |
|---|---|---|
| Home RTX 2070 (`desktop`, Tailscale) | **8 GB**, shared with Ollama | **Rejected for full-frame.** SAM3's encoder alone wants 8–10 GB; the box also runs `qwen3:8b` (~5.5 GB) as a scheduled task (see `desktop-ollama` skill). No headroom. **Viable only for the ROI-crop path** (small input → small footprint) and even then competes with Ollama for VRAM. |
| AWS **g4dn.xlarge** (T4, **16 GB**) — ft-5yq | 16 GB | **Recommended primary target.** Comfortably holds full SAM3 FP16 with room for the memory bank, and ROI-crop runs trivially. Already a tracked P1 (ft-5yq) with a support case to unblock. |

**Recommendation — two-track, ROI-crop is the unifier:**

1. **Primary: g4dn T4 (16 GB), FP16, ROI-crop.** Unblock ft-5yq, run the labelling
   service there. This is the dependable path: enough VRAM for full SAM3, and
   ROI-crop gives high fps on top.
2. **Opportunistic: RTX 2070 ROI-crop only.** Because ROI-crop shrinks the SAM3
   input to ~256–384 px, the 8 GB card *may* hold it alongside (or instead of)
   Ollama. Worth benchmarking — if it works, it's zero-cost, zero-latency (LAN via
   Tailscale) and removes the AWS dependency for interactive labelling. Treat as a
   bonus, not the plan of record.

The architecture is identical either way (local UI → Tailscale/HTTP → GPU box →
masks back), so the compute target is a deployment detail, not a redesign. Start
the service local-first (current MPS path stays working at `imgsz=512` for
obvious mid-pitch balls) and point it at whichever GPU benchmarks best.

## 5. Decision: SAM3 config

- **Model:** `sam3.pt` (base) — keep the current default. *Not* `sam3.1_multiplex.pt`
  (whole-frame-mask bug on our footage, already documented). Re-evaluate SAM3.1
  base separately as a fast-follow (its 32 FPS / 4 GB would change the math).
- **Precision:** FP16 on CUDA (autocast). The current code is `torch.no_grad()` but
  device-default precision; add FP16 on the GPU path.
- **Resolution strategy — ROI-crop, not global `imgsz`:**
  - Frame 0 / seed frame: full-frame at moderate `imgsz` (e.g. 1024) to locate the
    ball from the point/text seed.
  - Propagation frames: crop a window around the previous frame's ball box
    (window ≈ 6–8× the ball's longest side, clamped to frame), run SAM3 on the crop
    at its native size (effectively full-res *on the ball*), map the mask back to
    frame coords. Re-centre the window each frame from the new prediction.
  - On anomaly auto-stop (existing `_track_anomaly_reason`) or crop-miss (empty
    mask), fall back to a full-frame pass for one frame to re-acquire, then resume
    cropped.
- **Throughput levers, in priority order:** ROI-crop (biggest win) → FP16 →
  persistent hot predictor (already done) → `torch.compile` (already cached) →
  optional frame batching for offline (non-interactive) runs.
- **Object count:** one ball per seed (the cheap end of SAM3's latency curve). Don't
  co-track players in the same pass — separate concern, separate model run.

## 6. Architecture (mark-once loop)

```
Local annotation UI (FastAPI + Konva, docs/labeller_web_app_design.md)
  │  user marks ONE point (or text hint "soccer ball") on a frame
  │  HTTP/WS over Tailscale
  ▼
GPU box (g4dn T4, or RTX 2070 for ROI-crop)
  • hot SAM3VideoPredictor (get_cached_predictor)
  • ROI-crop propagation around predicted ball position
  • anomaly auto-stop → emit `anomaly` over WS
  ▼
masks/boxes per frame streamed back → canvas shows live track
  │  human scrubs, finds drift, corrects the box, "restart from here"
  ▼
corrected FrameDetections → export
  ▼
Feature store / training set  (ft-4jr store + importers) → retrain ball YOLO
```

Corrected output lands in the **feature store** (`src/footy_track/feature_store/`,
delivered by ft-4jr), which is the curated training-data home — not a fresh
Roboflow dependency. `export_frames_json` already serialises `FrameDetections`;
wire that into the feature-store importers so a finished clip becomes labelled
training rows idempotently (re-export of the same clip must not duplicate).

## 7. Child-bead breakdown

Spin these out as children of ft-8vx. Dependency direction uses "X needs Y".

1. **ft-8vx.1 — SAM3 GPU benchmark matrix (res × fps × quality).**
   Benchmark `sam3.pt` on (a) g4dn T4 and (b) RTX 2070, across full-frame vs
   ROI-crop, FP16 vs FP32, at 512/1024/native. Report fps and ball-IoU vs hand
   labels on 3–5 occlusion clips. **Output: the numbers that confirm §4/§5.**
   *Needs ft-5yq (g4dn provisioned).* P1.

2. **ft-8vx.2 — ROI-crop SAM3 propagation in `Sam3VideoLabeller`.**
   Add a crop-around-predicted-position path to `iter_frames`/`iter_frames_from`:
   crop window from prev box, run SAM3 on crop, map mask back, re-centre each
   frame, full-frame re-acquire on miss/anomaly. Unit-test mask round-trip
   (crop→frame coords). **The core optimisation — track the small ball at high
   effective res.** P1. *Independent of GPU; can start now on MPS.*

3. **ft-8vx.3 — Mark-once annotation service (FastAPI + Konva).**
   Build `labeller/server.py` per `docs/labeller_web_app_design.md`: promote
   `BackgroundLabeller` to a server singleton, WS frame stream, point/text seed,
   pause→correct→restart-from-here, anomaly push. Deployable on the GPU box,
   reachable over Tailscale. *Needs ft-8vx.2 for the high-res track.* P1.

4. **ft-8vx.4 — Corrected-labels → feature store ingestion.**
   Wire `export_frames_json` output into `feature_store/importers.py`
   idempotently (same clip re-export = no dupes). Closes the loop to training
   data. *Needs ft-4jr (done) + ft-8vx.3.* P2.

5. **ft-8vx.5 — Retrain ball YOLO on SAM3-generated labels.**
   Fine-tune the ball detector on the feature-store data the loop produces;
   measure recall lift vs the ft-55d.1 baseline (21.6 % → ?). *Needs ft-8vx.4.* P1.

6. **ft-8vx.6 (fast-follow) — Re-evaluate SAM3.1 base.**
   SAM3.1 promises 32 FPS / 4 GB FP16. The `_multiplex` variant has the
   whole-frame-mask bug on our footage; test the **base** SAM3.1 checkpoint for
   the same bug. If clean, it could make the 8 GB RTX 2070 the default and retire
   the AWS dependency. P2.

## 8. Acceptance (this epic)

- ✅ Compute target chosen: **g4dn T4 primary** (ft-5yq), **RTX 2070 ROI-crop
  opportunistic**, with the reasoning (SAM3 VRAM vs 8 GB; one-object ROI-crop).
- ✅ SAM3 config chosen: **`sam3.pt` base, FP16, ROI-crop around predicted ball
  position**, full-frame re-acquire on miss; SAM3.1 deferred to fast-follow.
- ✅ Concrete child-bead breakdown (§7) to build the mark-once loop.

## 9. Risks / open questions

- **ROI-crop re-acquisition:** if the ball leaves the crop window faster than the
  window re-centres (fast shots), we get a miss → full-frame fallback. Window size
  (6–8× ball) and the existing anomaly auto-stop are the mitigations; ft-8vx.1
  should stress this on shot/clearance clips.
- **RTX 2070 vs Ollama VRAM contention:** ROI-crop SAM3 + `qwen3:8b` may not
  co-reside in 8 GB. May need to stop Ollama during labelling sessions, or accept
  g4dn-only. ft-8vx.1 measures this.
- **SAM3.1 whole-frame bug:** only the `_multiplex` variant is confirmed bad here;
  the base SAM3.1 is untested. Until ft-8vx.6, stay on `sam3.pt`.
- **Frame serving cost** for smooth scrubbing in the web UI (already flagged in the
  web-app design) — pre-extract a window or cache frames.
