# Pipeline Architecture

This document describes the high-level pipeline for Footy Scan.

## Components

- InputConsumer
  - Reads video (file or live feed), decodes frames, and yields frames with timestamps.

- Processor
  - Responsible for transforming frame-level detections into tracked, time-accurate observations and events.
  - May be implemented as a single component or decomposed into three logical stages:
    1. Detection — run object detection to locate players, ball, referees, etc.
    2. Tracking — associate detections across frames to produce persistent IDs.
    3. Event extraction — infer higher-level events (passes, shots, tackles, set-pieces, substitutions) from trajectories and detection history.

- OutputProducer
  - Serializes processor outputs and writes to disk, a message bus, or an API. Supports JSON and CSV outputs; CSV is a flattened table view of the same objects.

## Dataflow

1. InputConsumer decodes and timestamps frames (GameTime + optional video metadata).
2. Processor runs detection and tracking, producing per-frame and per-event records with timestamps.
3. OutputProducer writes time-accurate, continuous-time records to the selected output format.

## Timestamps and time handling

All timestamps used across the pipeline are stored and exchanged in ContinuousTime (seconds from game start). Video sources commonly expose GameTime (the referee/broadcast clock) which resets at half-time. To produce canonical, continuous outputs the pipeline must convert GameTime to ContinuousTime using GameMetadata supplied by the video source.

- ContinuousTime is canonical for storage and downstream consumers.
- Processor outputs MUST include precise ContinuousTime timestamps so downstream systems can reliably align and resample observations.

## Implementation guidance

- Keep the processing stages small and testable. The Processor can be composed of separate detector, tracker and event extractor classes, or implemented as a single pipeline class that calls the three stages in sequence.
- Ensure every output record includes a precise timestamp in ContinuousTime so downstream systems can reliably align and resample observations.
- Document any assumptions about GameMetadata fields required to convert GameTime to ContinuousTime.
- When consuming video streams:
  - For first-half frames ContinuousTime == GameTime (if half_start_continuous == 0).
  - For second-half frames ContinuousTime = GameTime + half_start_continuous, where half_start_continuous is the ContinuousTime when the second half kicked off (this accounts for stoppage time in the first half).

## Extensibility

- Each component is intentionally small and replaceable.
- Implement custom detectors, trackers, or event extractors by adhering to the input/output timestamp conventions.
