# Technical Approach

## Overview

footy-track is a three-stage video analysis pipeline: **detect → track → extract**. Each stage is independent and replaceable. Data flows forward through the stages as Pydantic models, all carrying a canonical **ContinuousTime** timestamp.

```
Video
  │
  ▼
InputConsumer          ← decodes frames, stamps with GameTime + metadata
  │
  ▼
BroadcastClassifier    ← gates: is this frame worth processing?
  │ (Yes frames only)
  ▼
ObjectDetector         ← finds players, ball, referee, coach, substitutes
  │
  ▼
Tracker                ← links detections across frames → persistent IDs
  │
  ▼
EventExtractor         ← infers passes, shots, tackles from trajectories
  │
  ▼
OutputProducer         ← writes JSON / CSV / API
```

---

## Stage 0 — Broadcast Classification (gate)

Before running object detection (which is expensive), each frame is classified as pitch view (`Yes`) or non-pitch (`No`). Replays, crowd shots, and graphics overlays are filtered out.

**Why**: Object detection on a replay or close-up face shot wastes compute and produces garbage detections. Filtering first keeps the output clean and speeds up batch processing significantly.

**How**: A fine-tuned YOLO classification model (`yolo11n-cls`) returns `BroadcastClassification(label, confidence)`. The model is trained on a binary `No`/`Yes` dataset and achieves ~98% top-1 accuracy on a binary val set.

**Key decision**: The classifier runs on every frame and is cheap. Only `Yes` frames advance to detection.

---

## Stage 1 — Object Detection

Locates objects in each frame and returns normalized bounding boxes.

**Why YOLO**: Fast enough for real-time use, fine-tunable on domain data, well-supported via Ultralytics. YOLO11 (the current generation) offers strong accuracy/speed tradeoffs at multiple model sizes.

**Why domain fine-tuning**: A COCO-trained model knows "person" and "sports ball" but cannot distinguish player from coach from referee from substitute. Fine-tuning on Roboflow-labelled match footage produces a 7-class model covering the roles that matter for football analytics.

**Two implementations**:
- `UltralyticsObjectDetector` — YOLO-based, fast, good for player/ball detection.
- `UltralyticsSam3Detector` — SAM 3 segmentation, text-prompted per category. Slower but handles ambiguous cases (in-play vs. out-of-play ball) by specifying natural-language prompts with per-prompt confidence thresholds. Derives bounding boxes from masks with optional padding; uses centre-distance NMS to suppress duplicates.

**Coordinate convention**: All boxes are stored as normalized `[x, y, w, h]` (top-left origin, values in `[0, 1]`). This keeps the schema resolution-independent and simplifies downstream use.

---

## Stage 2 — Tracking

Associates detections across frames to assign persistent object IDs.

**Why**: A sequence of per-frame bounding boxes with no identity is not useful for event inference. Tracking turns "there is a player at (0.3, 0.5)" into "player #7 moved from (0.3, 0.5) to (0.35, 0.52) over the last 10 frames."

**How (planned)**: Hungarian algorithm assignment using the `lap` library, matching detections between frames by IoU overlap. This is a well-understood, fast approach for single-camera tracking.

**Current state**: Not yet implemented as a standalone module. Detection output is consumed frame-by-frame. Tracking is the next major pipeline stage to build.

---

## Stage 3 — Event Extraction (planned)

Infers higher-level match events from tracked trajectories: passes, shots, tackles, set-pieces, substitutions.

**Why separate from tracking**: Events are emergent — a pass is not visible in any single frame but in the trajectory of the ball and proximate players over time. Keeping event extraction downstream of tracking lets the tracker stay simple.

**Approach**: Rule-based or learned classifiers over tracked trajectory windows. Not yet implemented.

---

## Time handling

Two time systems exist in football video:

- **GameTime** — the referee/broadcast clock. Resets at each half. Available from video metadata.
- **ContinuousTime** — seconds from first-half kickoff, never resets. This is the canonical format for all stored records.

Conversion requires `GameMetadata` (specifically `half_start_continuous`: the ContinuousTime at which the second half kicked off, which accounts for first-half stoppage time).

**Key decision**: All output records carry ContinuousTime, not GameTime. This means downstream consumers never need to know match structure to align or merge observations.

---

## Dataset and training

**Roboflow** manages labelled datasets for both the detection model and the classifier. Labels are created via Roboflow's annotation tools and versioned as datasets. The pipeline includes upload scripts (`labelling.py`, `scripts/upload_*.py`) to push frames from inference runs back into Roboflow for active learning.

**Weights & Biases** tracks all training runs (hyperparameters, metrics, model artifacts). Run history is documented in `docs/training/notable_runs.md`.

**Training scripts** (`scripts/train_object_detector.py`, `scripts/train_classifier.py`) handle Roboflow download, model fine-tuning, and W&B logging. They accept a `DATA_ROOT` env var and an optional `--local-dataset` flag to bypass the download.

---

## Key design decisions

| Decision | Rationale |
|---|---|
| Broadcast classifier first | Cheap gate eliminates non-pitch frames before expensive detection |
| Normalized bounding boxes | Resolution-independent; no pixel math in the schema layer |
| ContinuousTime is canonical | Downstream consumers don't need to know match structure |
| Pluggable detector interface | Swap YOLO for SAM3 (or anything else) without touching the rest of the pipeline |
| Pydantic data models | Validated, serializable, IDE-friendly — the schema is self-documenting |
| `lap` for tracking | Fast, well-tested Hungarian assignment library; no ML overhead for tracking |
| Metaflow for batch jobs | Orchestrates large-scale processing (e.g., a full season) with retry and parallelism |
