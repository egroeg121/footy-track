# Input Stage — Live Stream Completion

Status: **DRAFT** · Issue: footy_track-yjd

This document specifies the unfinished half of the **Input** stage
(§2.1 of [`system_design.md`](../system_design.md)): the **live-stream
consumer**. File-based extraction already exists in
`scripts/extract_frames.py`.

---

## 1. Stage in §2 of system_design.md

This is the live-stream branch of **Input** (§2.1). The contract is
unchanged from that section:

- **Input** — a stream URL (HLS / RTMP / RTSP / WebRTC).
- **Output** — `(frame_image: np.ndarray, GameTime, video_metadata)`
  per decoded frame. `GameTime` is the canonical wall-clock; conversion
  to `ContinuousTime` happens at the input boundary per
  [`timings.md`](../timings.md).

---

## 2. Module interface

```python
# src/footy_track/input/streams.py

class FrameSource(Protocol):
    def __iter__(self) -> Iterator[FrameRecord]: ...

class FileFrameSource(FrameSource):
    def __init__(self, video_path: Path, kickoff: GameTime): ...

class LiveStreamFrameSource(FrameSource):
    def __init__(
        self,
        url: str,
        kickoff: GameTime,
        reconnect: ReconnectPolicy = ReconnectPolicy.exponential_backoff(),
    ): ...
```

`FrameRecord` is a Pydantic model: `(frame: np.ndarray, game_time:
GameTime, continuous_time: float, source_metadata: dict)`. The two
sources are interchangeable from a consumer's point of view —
downstream stages (Broadcast Classifier, Calibration, Detection) only
ever see `FrameRecord`.

---

## 3. Backing library

Default backend: **PyAV** (libav bindings). Rationale:

- Handles HLS / RTMP / RTSP without shelling out to ffmpeg.
- Exposes per-packet PTS so we can derive frame timestamps without
  guessing from frame indices.
- Already a transitive dep of common video tooling.

Fallback: shelling out to `ffmpeg` with frame-by-frame stdout piping is
acceptable for protocols PyAV cannot handle; this should be hidden
behind the same `FrameSource` interface.

---

## 4. Reconnection and gaps

A live source can drop. The consumer must:

1. Reconnect with exponential backoff (default 1s → 30s, capped).
2. Emit a sentinel `FrameRecord` with `source_metadata.gap=True` and
   no frame payload, so downstream stages can record the gap rather
   than silently lose time.
3. Resume `ContinuousTime` from wall-clock — not frame index — so the
   gap is reflected in the timestamps.

---

## 5. Cross-stage invariants honoured

- `ContinuousTime` is computed once, at the input boundary, from
  `GameTime`. See `timings.md`.
- The input emits **every decoded frame**; downstream sampling is the
  consumer's choice.
- Gaps are recorded, not silently dropped (mirrors the rule for
  non-broadcast frames in §2.2).

---

## 6. Out of scope

- Audio extraction.
- Multi-camera ingest (single-source v1).
- Authenticated streams (handled by the URL, not by this module).
