# SAM3 Labeller — Single-Timeline Design (authoritative)

The web labeller's bugs (dropped edits, wrong restart frame, scrambled labels,
boxes drifting) all came from multiple disconnected box stores. This replaces
them with **one server-owned timeline** as the single source of truth.

## Source of truth: the timeline

`Session.timeline: list[list[Box] | None]`, length = total_frames.
Each `Box` = `ObjectDetection` (normalized x,y,w,h, label) whose **`model`
field records provenance**:

- `"labeller"` — a manual edit. **Ground truth.**
- `"yolo"` — YOLO auto-detect.
- `"sam3"` — SAM3 propagation.

### Hierarchy
- `labeller` > {`yolo`, `sam3`}.  `yolo` and `sam3` are peers (for now).
- A per-box rule: **labeller boxes always win** and are never overwritten by
  yolo/sam3 on any frame.

## Who writes to the timeline

| Actor | Writes | Provenance | Respects |
|---|---|---|---|
| YOLO auto-detect | `timeline[N]` | `yolo` | replaces non-labeller boxes at N |
| SAM3 propagation | `timeline[N+1..]` | `sam3` | **keeps each frame's labeller boxes**, replaces yolo/sam3 |
| User edit | `timeline[frame]` | `labeller` | overwrites that frame |

### SAM3 merge into a frame (per-box hierarchy)
When SAM3 produces boxes for frame F:
```
new_frame = [b for b in timeline[F] if b.model == "labeller"]   # keep ground truth
          + sam3_boxes                                          # add propagated
timeline[F] = new_frame
```
So your edits on any downstream frame survive a re-propagation; only the
yolo/sam3 boxes get replaced.

## Restart semantics

**Restart from frame N:**
1. Seed = `timeline[N]` **verbatim** (whatever is there — your edits if you
   edited, else yolo/sam3 boxes).
2. Frame N is left untouched (it's the ground-truth seed).
3. SAM3 propagates N+1 → end, writing into the timeline with the merge rule
   above (labeller boxes on those frames are preserved).

## Client (thin)

- Renders `GET /timeline/{frame}` for the viewed frame.
- On edit: `POST /edit {frame, objects}` → server writes labeller boxes.
- WS `run`/`restart {start_frame}` → server seeds from `timeline[start_frame]`.
- Live: WS streams `{frame, idx}`; client fetches/renders that frame from the
  timeline (or receives boxes inline).
- No client-side box cache. No `frameEdits`. The timeline is authoritative.

## Endpoints

| Method | Path | Purpose |
|---|---|---|
| POST | /session/load | open video, allocate empty timeline |
| GET  | /frame/{idx}.jpg | raw frame image |
| GET  | /timeline/{idx} | boxes at idx (with provenance) |
| POST | /edit | {idx, objects} → write labeller boxes at idx |
| POST | /autodetect | {idx} → write yolo boxes at idx, return them |
| WS   | /ws | run/restart/pause; streams propagated frame indices |

## The N+1 drift bug (fold into rebuild)

Auto-detect (frame N) is correct but the first propagated frame (N+1) drifts.
Suspect: seed boxes are scaled with `SESSION.width/height` but
`Sam3VideoLabeller` re-reads its own `video_dimensions`; if they differ the seed
region is wrong. During the rebuild: seed `Sam3VideoLabeller` from
`timeline[N]` using ONE agreed dimension source, and verify the first propagated
frame lands on the players (Playwright + a numeric check).

## Migration

1. Add `timeline` to `Session` + the merge/seed helpers (server.py).
2. Point `/autodetect`, `/edit`, WS run/restart at the timeline.
3. Have the SAM3 worker write into `timeline` with the merge rule (seed frame
   verbatim already implemented via `_seed_frame_detections`).
4. Strip client `frameEdits`; client reads/writes the timeline only.
5. Verify drift + restart + edit-persistence end-to-end.

## Future: selectable tracking engine (after timeline is solid)

A later enhancement: choose SAM3 OR YOLO-tracking (UltralyticsTracker /
ByteTrack) as the propagation engine.
- SAM3: prompt-driven — seed boxes on frame N steer it.
- YOLO-track: detector-driven — re-detects + assigns IDs each frame; does NOT
  take seed boxes. Fills the timeline with `yolo` provenance. Restart = re-run
  from frame N onward. Your labeller edits remain ground truth (merge rule),
  but don't "re-seed" it.
Both write into the same timeline with provenance, so the timeline model is
engine-agnostic. Deferred until the SAM3 correction loop is working.
