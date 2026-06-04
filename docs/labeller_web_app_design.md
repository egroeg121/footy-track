# SAM3 Video Labeller — Web App Design (FastAPI + Canvas)

## Why move off Streamlit

The labeller is an interactive, stateful, frame-by-frame **video annotation tool
with mid-flight model correction**. Streamlit's model (re-run the whole script on
every interaction, no client-side state, fixed-pixel canvas, no selection
feedback) fights this at every turn. Every recurring bug this session — flashing
loops, ghost/doubled rows, frozen scrubbing, sluggishness, width misalignment —
traces back to whole-script reruns and the drawable-canvas's inability to report
state back to Python.

Off-the-shelf tools (CVAT / Label Studio / FiftyOne) don't fit either: they
treat the model as a black-box pre-labeller and have no concept of *pause
propagation → correct → resume from here*, which is the core workflow. Managed
versions also cost money to run models.

So: a **custom local web app**. The minimum that supports the correction loop is
a long-lived server that owns the hot SAM3 model + a real frontend canvas that
owns interaction state.

## Architecture

```
┌─────────────────────────────────────────────┐
│ Browser (single-page app)                   │
│  • Konva/Fabric canvas: draw/edit/select    │
│  • Frame scrubber + playback                │
│  • Class dropdown, tool toggle, run/pause   │
│  • Holds ALL interaction state client-side  │
└───────────────┬─────────────────────────────┘
                │  WebSocket (control + frame stream)
                │  HTTP (load video, fetch a frame image)
┌───────────────▼─────────────────────────────┐
│ FastAPI server (one process, model hot)     │
│  • Wraps the existing BackgroundLabeller /   │
│    Sam3VideoLabeller (reused as-is)          │
│  • SAM3 predictor stays resident in memory   │
│  • Streams propagated FrameDetections        │
│  • Serves frame images (jpeg) on demand      │
└─────────────────────────────────────────────┘
```

The backend is essentially the current `BackgroundLabeller` promoted from a
Streamlit-session object to a server-owned singleton. **No re-derivation per
interaction** — that single change removes the whole class of bugs.

## Backend (FastAPI)

Reuse, almost unchanged:
- `Sam3VideoLabeller` (`labeller/video_utils.py:236`) — `iter_frames_from(start_frame, stop_event, …)`
- `BackgroundLabeller` (`:507`) — `submit(...)`, `pause()`, `completed_frames()`
- `get_cached_predictor(...)` — model stays hot (already implemented)
- `_yolo_seed_objects(...)` — auto-detect seeds (move from app.py into a service module)

New thin layer: `labeller/server.py`

### HTTP endpoints

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/session/load` | `{video_path}` → opens video, returns `{fps, total_frames, width, height}`. Wipes prior results. |
| `GET` | `/frame/{idx}.jpg` | Raw frame image (no boxes; the canvas draws boxes itself). Cached. |
| `POST` | `/autodetect` | `{frame_idx, conf, iou, model?}` → YOLO seeds for that frame as boxes. |
| `GET` | `/detections/{idx}` | Propagated `FrameDetections` for a completed frame (for scrubbing). |
| `POST` | `/export` | Save labels JSON. |

### WebSocket `/ws` (control + stream)

Client → server messages:
- `{type: "run", objects, imgsz, conf, start_frame: 0}`
- `{type: "restart", objects, start_frame, imgsz, conf}`  ← objects = corrected boxes
- `{type: "pause"}`

Server → client messages:
- `{type: "status", state: "compiling"|"running"|"paused"|"idle"}`
- `{type: "frame", idx, detections}` — emitted per propagated frame (the live stream)
- `{type: "anomaly", idx}` — track jump/area anomaly detected; server auto-paused
- `{type: "done", last_frame}`
- `{type: "error", message}`

The server pushes `frame` messages as `BackgroundLabeller` produces them — no
polling. Anomaly detection (`_has_track_anomaly`) stays server-side and emits
the `anomaly` message instead of the Streamlit auto-pause hack.

### State ownership
- **Server owns**: the video handle, the hot predictor, the propagated
  `completed[]` timeline, run/pause state. One session per server (single user,
  local) keeps it simple — no multi-tenant session map needed for v1.

## Frontend (single page)

Stack: plain **Vite + TypeScript + Konva** (or Svelte if you prefer components).
No framework lock-in needed; one page.

Components:
- **Canvas (Konva `Stage`)**: background = `/frame/{idx}.jpg`; a `Layer` of
  editable rectangles. Konva gives selection, drag, resize, transform handles,
  delete — natively, with no server round-trip. Each rect carries `{label}`;
  colour by class.
- **Scrubber**: a range slider + prev/next + play. Scrubbing to frame N fetches
  `/frame/N.jpg` + `/detections/N` and renders read-only (or editable when
  paused). Smooth because it's client-side.
- **Toolbar**: Edit/Draw toggle, class dropdown, Auto-detect button, Run/Pause,
  Restart-from-current-frame. Show-boxes toggle.
- **Object list panel**: rows synced to canvas rects; click a row ↔ selects the
  rect (bidirectional — trivial in Konva, impossible in Streamlit). Class
  dropdown + delete per row.

### Interaction flow (the workflow that Streamlit couldn't do cleanly)
1. Load video → `/autodetect` frame 0 → seeds render on canvas (editable).
2. Edit boxes client-side (instant, no reruns) → **Run** sends `{run, objects}`.
3. Server streams `frame` messages → canvas shows live propagated frames.
4. **Pause** (or auto `anomaly`) → client switches canvas to editable on the
   current/anomaly frame, prefilled with that frame's detections.
5. Scrub freely (client-side, smooth) to find where it went wrong; the
   **Restart-from-current-frame** button sends `{restart, objects, start_frame}`
   using the *currently viewed* frame — no ambiguity, no rerun races.
6. Repeat; **Export** when happy.

## What this fixes vs Streamlit

| Streamlit problem | Resolved by |
|---|---|
| Sluggish / clunky | No whole-script reruns; canvas edits are local |
| Flashing / ghost rows | No widget-key churn; DOM owned by frontend |
| Frozen / wrong scrub frames | Client fetches the right frame + detections directly |
| Can't tell which box is selected | Konva reports selection natively → row highlight |
| Width misalignment | Responsive CSS layout |
| Live frames need polling hack | WebSocket push |
| Model re-derived per rerun | Server owns the hot predictor for its lifetime |

## Scope / effort

- **Backend**: ~150–250 lines (`server.py`) — mostly wiring existing
  `BackgroundLabeller` to FastAPI + a WebSocket loop. Logic already exists.
- **Frontend**: ~400–600 lines (one page: canvas, scrubber, toolbar, list, WS
  client). The bulk of new work.
- **Launch**: `uvicorn footy_track.labeller.server:app`; frontend built with
  Vite and served as static files by FastAPI (single command, no separate dev
  server needed in prod).

## Risks / open questions
- **Frame serving cost**: decoding arbitrary frames on demand via cv2 seek is
  fine for scrubbing; for smooth playback we may pre-extract or cache a window.
- **Single-session assumption**: v1 assumes one user/one video at a time (local
  use). Multi-session would need a session registry — out of scope for v1.
- **Frontend build toolchain**: adds Node/Vite to the repo. If undesirable, a
  no-build option (vanilla JS + Konva from CDN, served by FastAPI) keeps it
  Python-only to run, at some ergonomic cost.

## Migration path (incremental, low-risk)
1. Build `server.py` wrapping `BackgroundLabeller`; verify the WS frame stream
   with a trivial HTML test page (no Konva yet).
2. Add the Konva canvas + draw/edit + Run/stream (the vertical slice).
3. Add pause → correct → restart-from-current-frame.
4. Add scrubbing, object-list sync, auto-detect, export.
5. Retire `app.py` once parity is reached (keep it until then).

The existing `video_utils.py` backend is the asset — it carries straight over.
Only the Streamlit *presentation* layer (`app.py`) is replaced.
