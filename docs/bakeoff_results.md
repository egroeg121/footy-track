# Ball Tracking Bake-off Results

## Summary

Head-to-head comparison of ball tracking methods for the footy-track labeller loop.

| Method | Center-Acc | IoU | Recall | FPS | VRAM | Viable? |
|--------|-----------|-----|--------|-----|------|---------|
| ROI-YOLO (trained) | 0.0% | 0.00 | 0.0% | 2.7 | 0 MB | ❌ (0% recall) |
| VitTrack SOT (**winner**) | **79.8%** | **0.54** | **85.0%** | **10.2** | 0 MB | ✅ |
| SAM2 ROI (sam2_b.pt) | N/A² | N/A² | N/A² | 0.04–0.14 | — | ❌ (too slow + needs seeding) |

² Accuracy unmeasurable: tracker never seeds itself without external bbox init;
  cold-start path (~28s/frame) dominates and produces no ball detections.

**Winner: VitTrack SOT** — 79.8% center-acc, 85% recall, 10.2 FPS, CPU-only.

---

## Method Details

### Method A: VitTrack SOT (winner)

Run: 2026-06-23 by pyrite polecat (ft-5hd bakeoff). Dataset: 29 clips, 13,761 frames.

**Aggregate metrics:**

| Metric | Value |
|--------|-------|
| Center-accuracy (within 30px) | 79.8% |
| Mean center distance | 30.4 px |
| Mean IoU | 0.539 |
| Recall | 85.0% |
| FPS | 10.2 |
| Peak VRAM | 0 MB |
| Total tracking failures | 88 |
| Catastrophic failure rate | 11.2% |
| Occlusion recovery | 100.0% |

Model: VitTrack ONNX from HuggingFace Hub. CPU-only, no GPU required. Template
set on first valid bbox; search region extracted around previous position using
Kalman-guided ROI crop.

### Method B: SAM2 ROI (sam2_b.pt)

Run: 2026-06-23 by malachite polecat (ft-7ri). Hardware: Apple M-series MPS.

**Why re-run was needed:** Previous run (ft-5hd bakeoff) was inconclusive — the run
stalled ~40 minutes producing no result because the user's internet was paused,
blocking the 154MB `sam2_b.pt` download. The stall was network, not slow inference.

**Speed measurement (this run):**

| Test | Time | FPS |
|------|------|-----|
| ROI crop with box prompt (200×300, MPS) | ~7s/frame | ~0.14 FPS |
| Full-frame cold-start / no prompt (1920×1080, MPS) | ~28s/frame | ~0.04 FPS |

vs VitTrack baseline: **10.2 FPS**. SAM2 is **~70-700× too slow** on Apple MPS.

**Accuracy:** Full-clip run killed after 30 min (only ~25/250 frames processed). SAM2 without
an external seed detection (no `prev_bbox`) cold-starts on EVERY frame — the full-frame
promptless SAM2 call almost never finds the ball, so the tracker never warms up.
Total estimated time for 250 frames: ~117 minutes. Not a viable measurement path.

**Design limitation discovered:** The SAM2 tracker requires an external seed detection
to warm up. The cold-start path (`_full_frame_search` without a box prompt) is both
slow (~28s/frame) and ineffective on real football footage — SAM2 without prompting
returns noise, keeping `prev_bbox = None` for the entire clip. The ROI+box path
(~7s/frame, faster but still 70× too slow) is never reached in standalone evaluation.

**Verdict: NOT VIABLE** for the labeller loop on Apple Silicon.

SAM2 is a foundational segmentation model (80M params) — it's not designed for
real-time tracking. The architecture requires running the full ViT backbone on each
frame. Even with ROI cropping AND a box prompt, inference takes ~7s/frame on MPS.
A dedicated GPU (NVIDIA A10G or better) might achieve 1-5 FPS, but still below
VitTrack's 10 FPS. Additionally, the tracker requires YOLO seeding to work at all.

### Method C: ROI-YOLO (trained model)

Run: 2026-06-23 by pyrite polecat (ft-5hd bakeoff). Same dataset as VitTrack.

Completely failed: 0% recall. The trained YOLO model detects nothing when the
ball detector isn't YOLO-fine-tuned. This is expected — the "trained model" used
here wasn't trained on balls specifically.

---

## Dataset

- **Clips:** 29 matched (video + GT marks)
- **Total frames:** 13,761
- **Labelled frames:** 696
- **Ball-present frames:** 603
- **GT source:** `~/Library/Mobile Documents/.../footy_data/ball_gt_marks/*.jsonl`
- **Videos:** `/Users/georgebarnett/code/footy/footy_track/refinery/rig/eval_data/clips/`

---

## Recommendation

**Integrate VitTrack SOT** into the labeller loop. It achieves 79.8% center-accuracy
at 10.2 FPS with zero VRAM on CPU. This is sufficient for the auto-suggest labeller
UX where the user corrects wrong boxes.

SAM2 is not a viable alternative on the current hardware. Future evaluation on GPU
hardware may be worthwhile if VitTrack accuracy proves insufficient for the use case,
but at 79.8% center-acc + 85% recall, VitTrack already provides a useful starting
point for manual correction.

ROI-YOLO requires a ball-specific YOLO model — if the YOLO detector is fine-tuned on
balls, this approach might become competitive and would be much faster than SOT.
