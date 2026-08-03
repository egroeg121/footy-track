# Labeller — full feature & requirements inventory

Companion to `src/footy_track/labeller/README.md` (the short contract). This
document is the complete, testable inventory of CURRENT behavior of the
labeller server (`src/footy_track/labeller/server.py`), the VitTrack
propagation backend (`video_utils.py`), and the web UIs (`web/index.html`,
`web/review.html`). Each item is marked:

- **[REQ]** — required behavior. Tests must pin it; refactors must preserve it.
- **[INC]** — incidental implementation detail. May change in a refactor, but
  any change must be deliberate and called out.
- **[BUG?]** — observed behavior that looks unintended. Documented, not fixed
  here; pinned as-is where tested (behavior-preserving cleanup only).

Snapshot as of branch point `bdb2942`.

---

## 1. Persistence format — JSONL sidecar

One `<clip_stem>.jsonl` per clip under the GT-marks dir
(`~/Library/Mobile Documents/com~apple~CloudDocs/footy_data/ball_gt_marks`,
module constant `_GT_MARKS_DIR`). One JSON object per line:

```json
{"frame_index": 14, "bbox": {"x":0.1,"y":0.2,"w":0.05,"h":0.08},
 "center": {"x":0.125,"y":0.24}, "tags": ["in_play_ball", "labeller"]}
```

- **[REQ]** `tags` is `[label, model]` for box lines: first the class label,
  then the provenance tag (`labeller` | `vittrack` | `yolo` | `sam3`).
- **[REQ]** Skip markers are lines with `"bbox": null, "center": null` and
  `tags` of exactly `["no_ball"]` or `["not_broadcast"]`.
- **[REQ]** Frames with neither boxes nor markers are not written at all.
- **[REQ]** `center` is derived (`x + w/2`, `y + h/2`); it is written on flush
  but ignored on load.
- **[REQ]** Round-trip fidelity: flush → reload restores every box's label,
  geometry, and **original provenance tag** — machine boxes must never be
  promoted to `labeller` GT by a save/reload cycle. Legacy `sam3` tags must
  keep round-tripping even though SAM3 is no longer the propagation backend.
- **[REQ]** Restored confidence is synthesized (raw confidence is not
  persisted): `1.0` for `labeller` boxes, `0.5` for machine boxes.
- **[INC]** Load tolerates: blank lines, JSON-decode errors (skipped),
  `frame_index` out of `[0, total_frames)` (skipped), `bbox` as list `[x,y,w,h]`
  as well as dict.
- **[INC]** On load, the label is the first tag found in the known label set
  (ball + player classes); unknown labels fall back to `"in_play_ball"`. A line
  with no provenance tag defaults to `labeller`.

## 2. Session state machine (`Session`, module singleton `SESSION`)

Single-session server: one global `Session` holding one authoritative per-frame
`timeline: list[list[ObjectDetection] | None]` guarded by a lock. **[INC]** The
single-user/global-singleton design is incidental (no multi-session support is
required or provided).

### load (`Session.load`)
- **[REQ]** Loading a clip: pauses/replaces the `BackgroundLabeller`, force-
  flushes pending edits **of the previous clip** first, cancels the debounce
  timer, reads video metadata (fps — default 25.0 if unreadable — total
  frames, width, height), resets the timeline and the skip-marker sets, then
  restores the sidecar into the timeline (§1). Returns
  `{fps, total_frames, width, height}`.
- **[REQ]** Missing video path raises `FileNotFoundError` (surfaces as HTTP
  500 — **[INC]** the status code; the frontend only checks `res.ok`).

### timeline access
- **[REQ]** `get_frame(idx)` returns a copy of the frame's boxes, `[]` for
  never-populated or out-of-range frames.
- **[REQ]** `set_frame(idx, boxes)` overwrites a frame entirely (user edits /
  autodetect path). Out-of-range writes are silently ignored.
- **[REQ]** `merge_propagated(idx, boxes)` — the GT-authoritative merge:
  - If the frame has ANY `labeller`-provenance box, the frame is left
    **completely untouched** (not replaced, not augmented) and it returns
    `True` ("GT kept").
  - Otherwise the frame is overwritten with the propagated boxes; returns
    `False`.
  - Out-of-range indices: no-op, returns `False`.
- **[REQ]** `seed_objects(idx)` converts a frame's normalized boxes to
  absolute-pixel `LabelledObject`s to seed a propagation run.

### flush (debounced)
- **[REQ]** `schedule_flush()` debounces 2 s (timer reset on every call);
  clip switch forces an immediate flush. Flush rewrites the whole sidecar from
  a snapshot of the timeline + skip sets.
- **[REQ]** Skip-marker precedence per frame at flush time:
  `not_broadcast` > `no_ball` > boxes (a frame in a skip set writes only the
  marker line, even if boxes exist in the timeline).
- **[INC]** Flush failures are swallowed with a printed warning.

## 3. HTTP endpoints

All request/response bodies are JSON unless noted. Boxes over the wire are
normalized dicts `{label, x, y, w, h, conf, source}` (server → client) /
`{label, x, y, w, h, conf?}` (client → server, coordinates clamped to [0,1]).
**[REQ]** The `source` field carries provenance to the client.

### Pages & static
| Route | Behavior |
|---|---|
| `GET /`, `GET /main` | **[REQ]** serve `web/main.html` (hub) |
| `GET /labeller` | **[REQ]** serve `web/index.html` |
| `GET /object_review` | **[REQ]** serve `web/review.html` |
| `GET /ingest` | **[BUG?]** references `web/ingest.html`, which does not exist → 500 on request. The `/ingest/*` API endpoints below do exist. |
| `/static/*` | **[INC]** static mount of the `web/` dir |

### Clip listing
- `GET /clips` → `{clips: [{name, marked}], dir}` — **[REQ]** fast path, no
  video IO: `marked` = sidecar file exists. Sorted by name; suffixes
  `.mp4/.mov/.avi/.mkv`. Missing clips dir → `{clips: []}` (no `dir` key).
- `GET /clips/status` → `{clips: [{name, marked, complete, label_count}]}` —
  **[REQ]** slow path: `complete` = sidecar reaches within 15 frames of the
  clip's last frame AND has at least one player-class tag (ball-only clips are
  "in progress"). `label_count` = number of sidecar lines. Errors while
  parsing a sidecar degrade to `{marked: True, complete: False, label_count: 0}`.
- **[BUG?]** The frontend's "+ Add clip" button POSTs `/clips/add`, which is
  not implemented server-side (405/404). Dead frontend feature.

### Session / frames
- `POST /session/load` `{video_path}` → §2 load. **[REQ]**
- `GET /frame/{idx}.jpg` → JPEG bytes of that frame, or **[REQ]** 404 when no
  video is loaded / the frame can't be read.
- `GET /timeline/{idx}` → `{idx, boxes}` from the authoritative timeline. **[REQ]**
- `GET /next-detection/{from_idx}` → `{idx}` of the next frame strictly after
  `from_idx` with at least one box, else `{idx: null}`. **[REQ]**
- `GET /marks` → `{no_ball: [...], not_broadcast: [...], ball: [...],
  player: [...]}` — frame-index lists; `ball`/`player` = frames whose timeline
  boxes contain at least one ball-class / player-class label. **[REQ]**

### Editing
- `POST /edit` `{idx, objects}` — **[REQ]** overwrites the frame with the
  client boxes stamped `labeller` (GT). If the payload has ≥1 box, the frame is
  removed from both skip sets. Schedules a debounced flush. Returns
  `{idx, boxes}`.
- `POST /no-ball` `{idx}` — **[REQ]** adds to `no_ball_frames` AND strips
  ball-class boxes from that frame (player boxes are kept). Flush scheduled.
- `POST /not-broadcast` `{idx}` — **[REQ]** adds to `not_broadcast_frames`
  (boxes untouched). Flush scheduled.
- `POST /no-ball/clear`, `POST /not-broadcast/clear` `{idx}` — **[REQ]**
  remove the marker. Flush scheduled.

### Autodetect
- `POST /autodetect` `{frame_idx, current_boxes, conf?, iou?, model_path?}`:
  - **[REQ]** No video loaded → `{idx: 0, boxes: []}`.
  - **[REQ]** The client's `current_boxes` become `labeller` GT for the frame
    (autodetect merges with what is *on screen*, never with stale server
    state).
  - **[REQ]** YOLO runs on the frame (via `yolo_seed_objects`, defaults conf
    0.35 / NMS IoU 0.5); detections are stamped `yolo`; any YOLO box with IoU
    > 0.3 against a current (GT) box is suppressed; result = GT + surviving
    YOLO boxes, written to the timeline and returned.
  - **[INC]** YOLO boxes get confidence 1.0 (raw model confidence discarded).

### Label propagation (correction ripple)
- `POST /propagate` `{frame_idx, box_idx}` → `{propagated_to: n}`:
  - **[REQ]** Only propagates if the source box exists and has `labeller`
    provenance; otherwise `{propagated_to: 0}`.
  - **[REQ]** Walks strictly forward one frame at a time; at each populated
    frame, finds the highest-IoU `yolo`-provenance box vs the last position;
    stops at the first frame containing any `labeller` box, or when best IoU
    < 0.3 (track lost). Empty frames are skipped (walk continues).
  - **[REQ]** A matched box gets the reference label but **keeps `yolo`
    provenance** (a relabelled detection is still a detection). Flush
    scheduled at the end.

### Review API (tinder-style crop correction)
Backed directly by the sidecar files (not the live Session timeline). A box's
identity is `(clip, frame_index, box_index)` where `box_index` is the ordinal
of the box **among that frame's box lines in current file order** (skip-marker
and bbox-null lines excluded). **[REQ]** Queue, crop, correct, and delete all
use this same ordering (they must stay consistent — deleting/reordering lines
renumbers boxes).

- `GET /review/queue` → `{total, items: [{clip, frame_index, box_index, bbox,
  label, confidence, provenance, image_url}]}`:
  - **[REQ]** Scans all sidecars; skip-marker lines excluded; label = first
    tag in the known review label set (fallback `"player"`); confidence 1.0
    for `labeller` lines else 0.5.
  - **[REQ]** Ordering: machine-provenance items before `labeller` items,
    then ascending confidence, then rare classes weighted up (ball classes
    ×3, referee/coach/sub ×2).
  - **[REQ]** IoU dedup: within the same (clip, frame), items with IoU > 0.85
    against an already-queued item are dropped from the queue (they still
    exist in the file and still count for `box_index` numbering of later
    endpoints — dedup affects the queue only).
  - **[BUG?]** The review provenance tag set is `{labeller, yolo, sam3}` —
    `vittrack` is missing, so vittrack boxes are reported with
    `provenance: "labeller"` (but confidence 0.5). Pinned as current
    behavior.
- `GET /review/crop/{clip}/{frame}/{box}.jpg` — **[REQ]** JPEG crop of the box
  with 2.0× box-size padding on each side, edge-clamped; 404 if the clip video
  or box isn't found. **[INC]** LRU cache of 200 crops keyed by
  (clip, frame, box); JPEG quality 85.
- `GET /review/frame/{clip}/{frame}.jpg` — **[REQ]** full frame JPEG (quality
  80) or 404.
- `POST /review/correct` `{clip, frame_index, box_index, label, bbox}` —
  **[REQ]** rewrites that box's sidecar line in place with the new label and
  bbox, **stamped `labeller`** (GT promotion is the point of review), bbox
  clamped to [0,1] with w/h clamped to fit, center recomputed. Crop cache
  entry invalidated. Errors: `{ok: false, error: "clip not found" | "box_index
  out of range"}` (HTTP 200 — **[INC]**).
- `POST /review/delete` — **[REQ]** removes that box's line from the sidecar
  (same identity rules); cache invalidated; same error shape.
- `POST /review/yolo` `{clip, frame_index}` — **[REQ]** runs the current best
  YOLO detector on the frame, returns `{ok, boxes: [{label, confidence, x, y,
  w, h}]}` (rounded); `{ok: false, error, boxes: []}` when video missing.

### Ingest
- `POST /ingest/upload` (multipart) — **[REQ]** saves the upload under a temp
  uploads dir, returns `{path, name, size}`.
- `GET /ingest/run?path=&sample=&merge_gap_s=&min_seg_s=` — **[REQ]** SSE
  stream (`text/event-stream`) running
  `python -m footy_track.scripts.split_broadcast_segments <path> --outdir
  <clips dir> ...`, echoing each output line as a `data:` event, then
  `[EXIT <rc>]` and `[DONE]`. Missing file → `ERROR: file not found` +
  `[DONE]` without spawning a process.

## 4. WebSocket run protocol (`/ws`)

Client → server messages: `{type: "run" | "restart" | "pause", ...}`.
`run` and `restart` carry `{start_frame, conf?, imgsz?, model_uri?}` and are
**handled identically** server-side (**[INC]** the distinction is a frontend
labelling concern).

Server → client messages:
- `{type: "status", state: "compiling" | "running" | "paused" | "idle"}`
- `{type: "frame", idx, boxes, gt_kept}`
- `{type: "anomaly", idx, reason}`
- `{type: "done", last_frame}`
- `{type: "error", message}`

### run/restart
1. **[REQ]** Any in-flight streamer task is cancelled and the current
   `BackgroundLabeller` paused first.
2. **[REQ]** The run seeds from the **timeline** at `start_frame`
   (`Session.seed_objects`), never from client-supplied boxes — the frontend
   commits the canvas to `/edit` (GT) before sending `run`, so Run/Restart
   always starts from what's on screen, deterministically.
3. **[REQ]** No boxes on the start frame → `{type: "error", message: "No
   boxes on frame N to seed from."}` and no run starts.
4. **[REQ]** `status: compiling` is sent **before** the (potentially slow,
   blocking) `bg.submit(...)` call so the frontend can show the loading
   overlay for the whole model warmup (ft-wkc). The streamer task then sends
   `compiling` again on start, and `running` just before the first frame
   message.
5. **[REQ]** The streamer polls the `BackgroundLabeller` and, for each newly
   completed index, fetches the frame via **`frame_at(idx)`** — NOT
   `completed_frames()`. `completed_frames()` only scans the contiguous run
   from frame 0, so for a run seeded at frame N (frames 0..N-1 still `None`)
   it returned nothing and every frame was silently skipped — nothing
   ingested, nothing streamed (the "ran to frame 30 but 28–29 have no boxes"
   bug). Mid-clip runs MUST stream and ingest frames ≥ start_frame.
6. **[REQ]** Frame ingestion (`_ingest_completed_frame`): the seed frame's
   timeline entry is already GT and is emitted as-is (`gt_kept = False`);
   later frames' detections are stamped `vittrack` and merged via
   `merge_propagated` — `gt_kept: true` in the frame message when existing GT
   made the merge a no-op, so the frontend can report "kept your marks on
   frames 14–17" instead of skipping silently.
7. **[REQ]** Anomaly path: when the labeller flags an anomaly, the server
   sends `{type: "anomaly", idx, reason}` then `status: paused`, clears the
   anomaly marker, and the streamer exits (run stays paused for correction +
   restart).
8. **[REQ]** Normal completion: `{type: "done", last_frame}` then
   `status: idle`.

### pause
- **[REQ]** `{type: "pause"}` pauses the `BackgroundLabeller`, cancels the
  streamer, sends `status: paused`.

### disconnect
- **[REQ]** WebSocket disconnect pauses the run and cancels the streamer (no
  orphaned propagation).

## 5. Propagation backend (`video_utils.py`)

### VitTrackVideoLabeller (active backend)
- **[REQ]** One independent `VitTrackSOT` per seeded object; trackers warmed
  on the seed frame; propagation runs frame-by-frame from `start_frame`.
- **[REQ]** The seed frame is yielded FIRST, containing the user's boxes
  verbatim (model tag `vittrack`, confidence 1.0) — the tracker's
  re-detection is never trusted on the seed frame.
- **[REQ]** Yielded `FrameDetections.uri` encodes the absolute frame index
  (`<stem>_frame_<%06d>`); `_frame_index_from_uri` recovers it (fallback to a
  default on parse failure).
- **[REQ]** Per frame, each tracker updates its box; on tracker miss the
  previous box is carried forward with the (low) score. Low scores are how
  handback is signalled (see below).
- **[REQ]** `stop_event` stops cleanly after the current frame; progress
  callback reports **absolute** position `(abs_idx + 1, total)` so a restart
  at frame N shows N/total.
- **[REQ]** Constructor requires ≥1 object and an existing video path; extra
  kwargs (model_uri/imgsz/…) are absorbed for API compatibility.

### BackgroundLabeller
- **[REQ]** Runs the labeller in a daemon thread filling
  `frames[abs_idx]`; `last_completed_frame` is the max index written.
- **[REQ]** `submit(...)` pauses any current job first; re-allocates `frames`
  only when the total frame count changed (so earlier frames survive a
  restart); resets error/anomaly state; progress starts at
  `(start_frame, total)`.
- **[REQ]** `frame_at(idx)` returns the frame at an absolute index regardless
  of holes before it (the mid-clip fix, §4.5). `completed_frames()` is the
  legacy contiguous-from-0 scan (**[INC]** retained; no longer used by the
  server streamer).
- **[REQ]** Anomaly auto-stop, in the worker, checked per frame when
  `anomaly_detection` is on:
  1. Motion/size heuristic vs the previous frame
     (`_track_anomaly_reason`): nearest same-label box centre jumped > 40% of
     the frame diagonal, or area changed > 8× → reason string.
  2. Else confidence handback: any detection with confidence <
     `_VITTRACK_HANDBACK_SCORE` (0.5) → reason string.
  On anomaly: `anomaly_frame`/`anomaly_reason` set, stop event set, worker
  exits. A brand-new label appearing is NOT an anomaly.
- **[REQ]** `pause()` sets the stop event and joins (≤10 s);
  worker exceptions land in `.error`; `running` false on exit.

### Warm start
- **[REQ]** The VitTrack ONNX session is cached process-wide per model path in
  `ball_trackers/sot_vittrack.py` (one session serves all tracker instances;
  per-instance tracking state independent; HF download memoized) — so
  "Compiling model…" happens once per server start, not per run. (Pinned by
  `tests/labeller/test_sot_vittrack_cache.py`; that module's public behavior
  is out of scope for this refactor.)

### Legacy SAM3 path
- **[INC]** `Sam3VideoLabeller`, `get_cached_predictor`, `warmup_model`,
  `start_warmup_thread`, `_default_model_uri` are the retired SAM3 backend,
  no longer wired into `BackgroundLabeller`. `Sam3VideoLabeller` is still
  imported by `footy_track/labeller/__init__.py` and
  `scripts/proto_sam3_points.py`, and doubles as a `CropRunner`
  (motion_tracker §2.3 re-acquire backend).
- **[REQ]** Whatever happens to the SAM3 code path, the `sam3` **model tag**
  must keep round-tripping through sidecars (old files contain it).

### YOLO seeding
- **[REQ]** `yolo_seed_objects(video, model_path, conf, w, h, iou, frame_idx)`
  runs the current-best (or explicit) detector on the chosen frame and
  returns greedy-NMS-filtered absolute-pixel `LabelledObject`s.

## 6. Frontend behaviors (labeller, `web/index.html`)

Not unit-testable in this repo (no JS test runner); the server contract each
behavior relies on is tested instead (§3, §4). Inventory:

### Tool modes
- **[REQ]** `draw` (default): drag creates a box of the selected class, always
  provenance `labeller`; boxes < 4px discarded. `edit`: drag/resize via
  transformer, click selects, Delete/Backspace removes selection. Drawing
  disabled while running.

### Keyboard map
| Key | Action |
|---|---|
| `w` / `e` | draw / edit tool |
| `1–6` | class hotkeys (in_play_ball, ball, referee, player, person, coach) + switch to draw |
| `←`/`→`, `a`/`d` | prev/next frame; `Shift` ×10, `Ctrl/Cmd` ×50 |
| `f` | tap-count skip: 1 tap = ½ s, 2 taps = 1 s, 3+ = 4 s worth of frames |
| `g` | next frame with detections (`/next-detection`) |
| `n` / `b` | toggle no-ball / not-broadcast on current frame |
| `r` | re-run autodetect on current frame |
| `z` | undo (20-deep per-frame snapshot stack) |
| `Delete`/`Backspace` | delete selected box (edit tool) |
- **[REQ]** All shortcuts ignored while typing in an input/select.
- **[BUG?]** Undo restores boxes without their `source` attr → restored boxes
  render as `labeller` (machine provenance lost on undo).

### Timeline bar
- **[REQ]** Canvas strip under the scrubber; per frame: red = no-ball, blue =
  not-broadcast, yellow = ball only, yellow-over-green = ball + player, green
  with a thin red top stripe = player-only (ball status undecided). White
  2px cursor = current frame. Populated from `/marks` on load and updated
  live from ws `frame` messages and local edits.

### Clip picker
- **[REQ]** Left sidebar from `/clips` (fast) with grey `✓` for marked clips;
  completion (green `✅` + label count tooltip) lazily upgraded from
  `/clips/status`; list refreshed every 20 s. Clicking a clip saves current
  edits, clears the canvas, and loads it. Active clip highlighted.
- **[INC]** "+ Add clip" button (calls unimplemented `/clips/add`, §3).

### Load behavior
- **[REQ]** On clip load: run-control UI fully reset (Run visible, Restart
  hidden, pause disabled, hint/overlay cleared); scrubber sized; marks
  restored from `/marks`; frame 0 shown. If frame 0 has saved boxes they are
  loaded; otherwise autodetect seeds frame 0. WS (re)connected. Last video
  and model paths persisted in localStorage.

### Edit/save loop
- **[REQ]** The server timeline is the single source of truth: leaving an
  edited frame POSTs `/edit`; entering a frame GETs `/timeline/{idx}`.
  `frameDirty` gates saves (navigation without edits does not POST).
- **[REQ]** After a manual save, each `labeller` box on the frame is
  `/propagate`d forward, and the total ripple count is surfaced in the status
  line.

### Run lifecycle (client side)
- **[REQ]** Run/Restart: commit canvas to `/edit` at the current frame, clear
  the kept-frames set, send ws `run`/`restart` with `start_frame` = current
  frame. UI → running: pause enabled, autodetect disabled, edit tool forced,
  compiling overlay shown.
- **[REQ]** Compiling overlay: a sibling of the Konva stage container (Konva
  wipes its container's innerHTML), shown from Run until the first live frame
  arrives; also driven by `status: compiling`.
- **[REQ]** Live `frame` messages: update `lastCompleted`, timeline-bar sets,
  and kept-frames; only rendered to canvas while mode is `running` (stale
  in-flight frames after a pause must not move the view).
- **[REQ]** `status: paused` (not self-initiated) → jump to the last live
  frame, editable, Restart button labelled "Restart from frame N" (N follows
  scrubbing while paused).
- **[REQ]** `anomaly` → auto-pause on the anomaly frame with the reason shown
  in amber; user corrects then Restarts.
- **[REQ]** `done` → jump to last completed frame, idle-with-results UI. The
  kept-frames summary ("kept your marks on frames 14–17", ranges collapsed)
  is appended to the paused/done hint.

### WS resilience
- **[REQ]** Every send goes through `ensureWS()` — reconnect on demand with a
  5 s timeout (a `ws.send` on a dead socket is silently discarded by
  browsers, which used to hang Run at "Compiling model…"). `onclose`
  auto-reconnects with exponential backoff (500 ms → 8 s cap). Errors surface
  in the status line.

### Hierarchy display (z-order / list order)
- **[REQ]** Tier map `labeller: 0, vittrack/sam3: 1, yolo: 2`. The objects
  pane and on-canvas numbering list boxes in tier order (stable within a
  tier); the canvas draws lower tiers first so GT is always on top. Machine
  boxes render dashed/faded; GT solid.
- **[REQ]** Objects pane: per-box class dropdown (disabled while running),
  delete `✕`, "Clear all" with confirm + undo; no-ball / not-broadcast rows
  shown with inline clear.

## 7. Review UI (`web/review.html`)

- **[REQ]** Grid of crops grouped by class (pills with counts), batch size
  100, ordered by the server queue. Cards show label, confidence + provenance
  badge (machine only), bbox coords, and a GT-box overlay drawn in crop space
  (pad 2.0, edge-clamped — must match the server's crop geometry).
- **[REQ]** Selection: shift/cmd-click or double-click toggles; `s` select
  all, Esc clear. Batch accept (mark seen), relabel (class picker →
  `/review/correct` with existing bbox), delete (`/review/delete`).
- **[REQ]** Accept/relabel mark items **seen** (green border, persisted in
  localStorage `review_seen_v1`) rather than removing them; "Hide reviewed"
  toggle (default on) filters seen items from the grid.
- **[REQ]** Modal: Konva-editable crop (draw/edit tools) beside the full
  frame with an SVG GT overlay; YOLO re-run overlays same-class detections
  dashed; accept saves bbox changes then advances; relabel stays in the
  modal; `a`/`d`/arrows navigate with auto-save; Space accepts;
  Backspace/Delete deletes; Esc auto-saves and closes.
- **[REQ]** Box edits are converted crop-space → frame-space using the same
  pad-2.0 edge-clamped mapping as the server crop.

## 8. Known dead code / debt (cleanup targets, phase 3)

- `server.py`: unused `subprocess` import; `/marks` reaches into
  `SESSION._tl_lock` directly; `_PLAYER_LABELS` defined mid-file after first
  use point; review/ingest/ws sections all inline in one 1300-line module.
- `video_utils.py`: `export_frames_json` + `has_ffmpeg` used only by the
  legacy Streamlit `app.py`; SAM3 classes retained per §5.
- `web/index.html`: trailing dead `const _origStatus = setStatus;`.
- Missing `web/ingest.html` (§3) and `/clips/add` (§3) — functional gaps,
  out of scope for a behavior-preserving refactor; left documented here.
