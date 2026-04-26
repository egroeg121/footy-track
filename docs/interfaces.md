# Pipeline Interfaces

Names, responsibilities, and data contracts for each component in the footy-track pipeline. Implementation details live in the source; this document records what each component promises, not how it delivers it.

---

## `InputConsumer`

**Responsibility**: Read a video source (file or live feed), decode frames, and yield them with timestamps.

**Contract**:
- Accepts a video file path or stream URI.
- Yields frames paired with `GameTime` and `GameMetadata` (half number, `half_start_continuous`).
- Does not interpret frame content — that is the detector's job.

**Output per frame**:
```
Frame(
    image_path: Path,      # path to extracted frame image
    game_time: float,      # seconds on the referee clock for this frame
    game_metadata: GameMetadata,
)
```

---

## `BroadcastClassifier`

**Responsibility**: Decide whether a frame is a pitch view worth running detection on.

**Contract**:
- Accepts a frame image path.
- Returns a `BroadcastClassification` with a `Yes`/`No` label and confidence.
- Must be cheap enough to run on every frame.
- Does not modify frame data or produce bounding boxes.

**Interface**:
```
classify(image_path: Path) → BroadcastClassification
```

**Output**:
```
BroadcastClassification(
    label: "Yes" | "No",
    confidence: float | None,
)
```

---

## `ObjectDetector`

**Responsibility**: Locate all objects of interest in a single frame and return their bounding boxes and labels.

**Contract**:
- Accepts a frame image path.
- Returns a `FrameDetections` containing all detected objects.
- All bounding boxes are normalized `[x, y, w, h]` with top-left origin, values in `[0, 1]`.
- Does not maintain state across frames.

**Interface**:
```
predict_from_path(image_path: Path) → FrameDetections
```

**Output**:
```
FrameDetections(
    uri: Path,
    width: int,
    height: int,
    detections: list[ObjectDetection],
)

ObjectDetection(
    label: str,        # one of DETECTION_CLASSES
    confidence: float,
    x: float, y: float, w: float, h: float,  # normalized, top-left origin
    model: str | None,
)
```

**Known labels** (from `constants.py`): `person`, `ball`, `in_play_ball`, `out_of_play_ball`, `player`, `player_sub`, `referee`, `coach`.

---

## `Tracker` *(planned)*

**Responsibility**: Associate detections across consecutive frames to produce persistent object IDs.

**Contract**:
- Accepts a `FrameDetections` for the current frame.
- Returns the same detections annotated with stable `track_id` integers.
- Maintains internal state between calls — must be called in frame order.
- Handles object entry (new ID) and exit (ID retired) gracefully.

**Interface**:
```
update(detections: FrameDetections) → TrackedFrameDetections
```

**Output**:
```
TrackedFrameDetections(
    uri: Path,
    width: int,
    height: int,
    detections: list[TrackedDetection],  # ObjectDetection + track_id: int
)
```

---

## `EventExtractor` *(planned)*

**Responsibility**: Infer higher-level match events (passes, shots, tackles, substitutions, set-pieces) from tracked trajectories.

**Contract**:
- Consumes a window of `TrackedFrameDetections` (not a single frame).
- Emits zero or more `MatchEvent` records when an event boundary is detected.
- Events carry `ContinuousTime` timestamps, not `GameTime`.
- Does not modify or re-emit detection data.

**Interface**:
```
process(frames: Iterable[TrackedFrameDetections]) → Iterator[MatchEvent]
```

**Output**:
```
MatchEvent(
    event_type: str,          # e.g. "pass", "shot", "tackle"
    continuous_time: float,   # seconds from first-half kickoff
    player_ids: list[int],    # track IDs involved
    coordinates: ...,         # pitch coordinates if available
)
```

---

## `OutputProducer`

**Responsibility**: Serialize pipeline outputs and write them to disk, a message bus, or an API.

**Contract**:
- Accepts any pipeline record (`FrameDetections`, `BroadcastClassification`, `MatchEvent`).
- Supports JSON (one record per line) and CSV (flattened table view) formats.
- All written records include a `continuous_time` field.
- Does not perform inference or modify records.

**Interface**:
```
write(record: PipelineRecord) → None
flush() → None
```

---

## Time contract (applies to all components)

Every record produced by any component **must** include a `continuous_time: float` field representing seconds from first-half kickoff. This is the canonical timestamp for alignment, resampling, and merging across sources.

`GameTime` values from the video source are converted to `ContinuousTime` by `InputConsumer` using `GameMetadata.half_start_continuous`. No downstream component should receive raw `GameTime` values.

See `docs/timings.md` for the full conversion formula.

---

## Data flow summary

```
InputConsumer
  └─► Frame(image_path, game_time, game_metadata)
        │
        ▼
BroadcastClassifier
  └─► BroadcastClassification(label="Yes"|"No", confidence)
        │ (Yes only)
        ▼
ObjectDetector
  └─► FrameDetections(uri, width, height, detections=[ObjectDetection, ...])
        │
        ▼
Tracker  [planned]
  └─► TrackedFrameDetections(... detections=[TrackedDetection(track_id), ...])
        │
        ▼
EventExtractor  [planned]
  └─► MatchEvent(event_type, continuous_time, player_ids, coordinates)
        │
        ▼
OutputProducer
  └─► JSON / CSV on disk, message bus, or API
```
