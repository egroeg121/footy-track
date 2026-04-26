# footy-track System Overview

This document consolidates the architecture, pipeline stages, data models, time conventions, and module overview for footy-track (Footy Scan). It is the single authoritative reference for contributors and agents.

---

## Architecture

footy-track is a three-stage video analysis pipeline:

```
Video Input
    │
    ▼
┌─────────────────┐
│  InputConsumer  │  Decodes frames, stamps each with GameTime + video metadata
└────────┬────────┘
         │  Frames (image + timestamp)
         ▼
┌─────────────────────────────────────────────────────┐
│  Processor                                          │
│   ┌─────────────┐  ┌──────────┐  ┌───────────────┐ │
│   │  Detection  │→ │ Tracking │→ │ Event Extract │ │
│   └─────────────┘  └──────────┘  └───────────────┘ │
└────────────────────┬────────────────────────────────┘
                     │  Per-frame and per-event records (ContinuousTime)
                     ▼
          ┌──────────────────┐
          │  OutputProducer  │  Writes JSON / CSV to disk, message bus, or API
          └──────────────────┘
```

Each component is independently replaceable. Timestamps flow through all stages as **ContinuousTime** (see [Time Conventions](#time-conventions)).

---

## Pipeline Stages

### Stage 1 — Detection

Locates objects (players, ball, referee, coach, substitutes) in each frame.

- **Interface**: `ObjectDetector.predict_from_path(image_path) → FrameDetections`
- **Implementations** (in `detectors/`):
  - `UltralyticsObjectDetector` — YOLO11-based detector; maps COCO class indices to `person` and `ball`.
  - `UltralyticsSam3Detector` — SAM 3 text-prompted segmentation; uses natural-language prompts per object category with per-prompt confidence thresholds; derives bounding boxes from segmentation masks with optional padding; applies custom centre-distance NMS to suppress duplicates.
- **Output**: `FrameDetections` — list of `ObjectDetection` with normalized `[x, y, w, h]` boxes in `[0, 1]`.

### Stage 2 — Tracking (planned)

Associates detections across frames to produce persistent object IDs.

- Intended implementation: Hungarian-algorithm assignment (via the `lap` library).
- Not yet implemented as a standalone module; currently the detection output is consumed frame-by-frame.

### Stage 3 — Classification / Event Extraction

Two sub-tasks:

1. **Broadcast classification** — decides whether a frame is a broadcast/camera view suitable for detection. Implemented in `classifier.py`.
   - `UltralyticsClassifier` uses a fine-tuned YOLO classification model (`yolo11n-cls`).
   - Returns `BroadcastClassification` (Yes/No + confidence).

2. **Labelling / Roboflow upload** — `labelling.py` orchestrates classification-filtered, pre-annotated uploads to Roboflow for dataset curation. This feeds the training loop, not the live inference pipeline.

---

## Data Models

All models live in `src/footy_track/schema.py` and are Pydantic `BaseModel` subclasses (frozen by default where noted).

### `ObjectDetection`

Single detected object in a frame.

| Field | Type | Description |
|---|---|---|
| `label` | `str` | Class name (see [Detection Classes](#detection-classes)) |
| `confidence` | `float` | Model confidence in `[0, 1]` |
| `x` | `float` | Top-left x, normalized `[0, 1]` |
| `y` | `float` | Top-left y, normalized `[0, 1]` |
| `w` | `float` | Box width, normalized `[0, 1]` |
| `h` | `float` | Box height, normalized `[0, 1]` |
| `model` | `str \| None` | Model identifier that produced this detection |

Subclasses: `Person` (label locked to `"person"`), `Ball` (label locked to `"ball"`).

### `FrameDetections`

All detections for a single frame.

| Field | Type | Description |
|---|---|---|
| `uri` | `Path` | Path to the source image |
| `width` | `int` | Image width in pixels |
| `height` | `int` | Image height in pixels |
| `detections` | `list[ObjectDetection]` | All detections in this frame |

`FrameDetectionsWithMeta` extends this with an optional `clock: str` for raw broadcast clock text.

### `BroadcastClassification`

Result of classifying whether a frame is a broadcast view.

| Field | Type | Description |
|---|---|---|
| `label` | `EnumBroadcastClassification` | `"Yes"` or `"No"` |
| `confidence` | `float \| None` | Model confidence |

### `FrameClassifications`

Wraps a `BroadcastClassification` for a specific frame URI.

| Field | Type | Description |
|---|---|---|
| `uri` | `Path` | Path to the source image |
| `classification` | `BroadcastClassification` | Classification result |

Provides `.to_fiftyone_sample()` for FiftyOne dataset integration.

### Detection Classes

Defined in `constants.py` and referenced by `DETECTION_CLASSES` in `schema.py`:

| Constant | Value | Meaning |
|---|---|---|
| `PERSON_TAG` | `"person"` | Generic person (pre-role assignment) |
| `BALL_TAG` | `"ball"` | Ball (role unspecified) |
| `IN_PLAY_BALL_TAG` | `"in_play_ball"` | Ball on the pitch |
| `OUT_OF_PLAY_BALL_TAG` | `"out_of_play_ball"` | Ball off the pitch |
| `PLAYER_TAG` | `"player"` | Outfield player or goalkeeper on team |
| `PLAYER_SUB_TAG` | `"player_sub"` | Substitute waiting off the pitch |
| `REFEREE_TAG` | `"referee"` | Match official |
| `COACH_TAG` | `"coach"` | Coach on sideline |

---

## Time Conventions

> Full reference: [`docs/timings.md`](timings.md)

Two time systems are used throughout the pipeline:

### ContinuousTime

- Seconds from first-half kickoff (`0.0`).
- **Never resets** — continuous across both halves and stoppage time.
- **Canonical format** for all stored records and downstream consumers.

### GameTime

- The referee / broadcast clock. Resets to `0:00` at the start of each half.
- Used by video sources and broadcasters.
- Must be converted to ContinuousTime before storage.

### Conversion (via `GameMetadata`)

| Half | Formula |
|---|---|
| First | `ContinuousTime = GameTime_seconds` |
| Second | `ContinuousTime = GameTime_seconds + half_start_continuous` |

`half_start_continuous` is the ContinuousTime at which the second half kicked off (accounts for first-half stoppage time).

**Example**: First half ran 48.5 minutes → `half_start_continuous = 2910.0 s`. A second-half `GameTime` of `05:00` → `ContinuousTime = 300 + 2910 = 3210.0 s`.

---

## Module Overview

```
src/footy_track/
├── schema.py          Pydantic data models for all pipeline objects
├── constants.py       Detection class labels and Roboflow project names
├── classifier.py      Broadcast-frame classification (YOLO cls)
├── labelling.py       Roboflow dataset upload handlers (classification + detection)
├── logging.py         Logging configuration
├── utils.py           Project-root resolution and shared helpers
├── detectors/
│   ├── base.py        Abstract ObjectDetector interface
│   ├── ultralytics.py YOLO and SAM3 detector implementations
│   ├── utils.py       Device selection, bbox conversion, FiftyOne helpers, IoU
│   └── constants.py   Detector-specific constants
└── scripts/
    ├── classify_frames.py           Run broadcast classifier over a directory
    ├── detect_objects.py            Run object detector over a directory
    ├── extract_frames.py            Extract frames from a video file
    ├── split_video.py               Split video into clips
    ├── upload_classifier_frames.py  Upload frames to Roboflow classification project
    └── upload_object_detector_frames.py  Upload frames to Roboflow detection project
```

### `detectors/utils.py` key functions

| Function | Purpose |
|---|---|
| `_available_device()` | Returns MPS → CUDA → CPU `torch.device` |
| `ultralytics_result_to_detections()` | Converts Ultralytics `Results` to `list[ObjectDetection]` |
| `top_left_wh_to_yolo_xywh()` | Converts normalized top-left box to YOLO center format |
| `visualise_detections_on_image()` | Draws detections on an image for debugging |
| `detection_to_fiftyone()` | Converts `ObjectDetection` to FiftyOne `Detection` |
| `calculate_iou()` | IoU between two `ObjectDetection` boxes |

### `metaflow/`

Contains Metaflow workflow definitions for batch processing pipelines (e.g., processing large video archives). See `metaflow/docker/` for containerisation support.

---

## Key Design Decisions

- **Normalized coordinates**: All bounding boxes use `[0, 1]` normalized `[x, y, w, h]` (top-left origin) internally. Conversion to pixel space or COCO/YOLO formats happens at output boundaries.
- **Pluggable detectors**: New detectors implement `ObjectDetector.predict_from_path` and return `FrameDetections` — the rest of the pipeline is unchanged.
- **Broadcast filter**: Frames are pre-screened by the broadcast classifier before object detection, reducing compute on non-pitch frames (replays, graphics, crowd shots).
- **SAM3 for fine-grained detection**: SAM3 is text-prompted per object category, enabling differentiation between in-play/out-of-play balls, substitutes, coaches, and referees that a standard COCO-trained YOLO model cannot distinguish.
- **ContinuousTime is canonical**: All output records carry ContinuousTime so downstream consumers can align, resample, or merge observations without knowing match structure.
