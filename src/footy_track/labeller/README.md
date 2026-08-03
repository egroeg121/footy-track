# Footy Track Labeller — Specification

**This document is the canonical, spec-driven source of truth for the
labeller.** Every feature and behavior is captured as a numbered requirement
(`LAB-nnn`) with MUST/SHOULD wording, grouped by subsystem, and traced to the
test(s) that pin it. Requirements with no server-side test are marked
`UNTESTED`; behaviors that live only in the browser JS (no JS test runner in
this repo) are marked `UNTESTED-FRONTEND` — their *server contract* is tested
instead. Known defects and gaps are tracked as `OPEN-n` items in §12.

Snapshot: origin/main `95b60cc`. Change process: a behavior change MUST update
the corresponding requirement (and its tests) in the same PR; new behavior
MUST get a new ID. IDs are stable — never renumber, retire with a note.

## Overview

Web app for semi-automatic video labelling: mark boxes by hand, propagate them
through the clip with VitTrack, review/correct the results.

- FastAPI backend: `server.py` (composition root) + `session.py`, `review.py`,
  `ingest.py`, `run_stream.py`, `constants.py`
- Propagation backend: `video_utils.py` (VitTrack SOT; legacy SAM3 retained),
  `motion_tracker.py`, `ball_trackers/sot_vittrack.py`
- Konva.js frontends: `web/index.html` (labeller), `web/review.html` (review)

Run with:

```bash
uv run uvicorn footy_track.labeller.server:app --reload
```

| Route | Page |
|---|---|
| `/` (alias `/main`) | Hub |
| `/labeller` | Frame labeller (mark + propagate) |
| `/object_review` | Tinder-style crop review/correction |
| `/ingest` | Clip ingestion (page currently broken — OPEN-1) |

Test suite: `tests/labeller/` (run `uv run pytest tests/labeller/`). Test
references below are `file::test_name` within `tests/labeller/`.

---

## 1. Label hierarchy & provenance (LAB-0xx)

Every box records where it came from in its `ObjectDetection.model` field,
persisted in the JSONL sidecar as the second tag (`tags: [label, model]`):

| `model` | Meaning | Authority |
|---|---|---|
| `labeller` | Hand-drawn / hand-corrected — ground truth | Never auto-overwritten |
| `vittrack` | VitTrack SOT propagation output | Overwritten by re-runs; ignored on frames that have GT |
| `yolo` | YOLO autodetect output | Same as vittrack |
| `sam3` | Legacy SAM3 propagation (back-compat with old sidecars) | Same as vittrack |

- **LAB-001** (MUST) Every box carries a provenance tag in `model`; it is
  persisted as the second element of the sidecar `tags` list and restored
  intact on reload.
  - Tests: `test_session_roundtrip.py::test_roundtrip_preserves_model_tags_label_and_geometry`
- **LAB-002** (MUST) Hand marks are sacred: no automatic process may modify or
  discard `labeller`-provenance boxes, and nothing may silently promote
  machine boxes to `labeller` GT. Machine output never replaces anything at or
  above its own level.
  - Tests: `test_session_roundtrip.py::test_merge_propagated_keeps_gt_and_returns_true`,
    `test_ws_run_protocol.py::test_run_reports_gt_kept_frames`,
    `test_session_roundtrip.py::test_roundtrip_preserves_model_tags_label_and_geometry`
- **LAB-003** (MUST) Per-box GT promotion (main `940ab23`): a client payload
  box that carries a `model` field keeps that provenance; a box without one
  falls back to the endpoint's default (`labeller` for `/edit` and
  `current_boxes` in `/autodetect`). Saving a frame therefore promotes only
  the boxes the user actually touched; older clients that omit `model` are
  unaffected.
  - Tests: `test_per_box_promotion.py::test_edit_keeps_per_box_model_tags`,
    `test_per_box_promotion.py::test_boxes_from_payload_model_fallback_rules`
- **LAB-004** (MUST) Display order follows the hierarchy everywhere labels
  are listed or drawn: GT first, then model-assisted (`vittrack`/`sam3`), then
  pure model output (`yolo`). Tier map: `labeller: 0, vittrack/sam3: 1, yolo: 2`.
  - Tests: UNTESTED-FRONTEND (`web/index.html` `TIER`/`orderedBoxRects`)
- **LAB-005** (MUST) Hand-touching a box in the UI (drag/resize/relabel)
  promotes exactly that box to GT (`source` → `labeller`, stroke turns solid
  live); drawing a new box is always GT.
  - Tests: UNTESTED-FRONTEND (`promoteToGT` in `web/index.html`; server leg is LAB-003)
- **LAB-006** (MUST) Undo restores each box with its own `model` tag — undo
  must not promote machine boxes to GT (fixed on main, `95b60cc`; previously
  OPEN, see §12).
  - Tests: UNTESTED-FRONTEND (`undoLast` passes `o.model` back to `addRect`)
- **LAB-007** (MUST) The legacy `sam3` model tag keeps round-tripping through
  sidecars and surfaces in review (old sidecars contain it), even though SAM3
  is no longer the propagation backend.
  - Tests: `test_review_endpoints.py::test_queue_sam3_legacy_tag_round_trips`,
    `test_session_roundtrip.py::test_roundtrip_preserves_model_tags_label_and_geometry`

## 2. Persistence — JSONL sidecar (LAB-1xx)

One `<clip_stem>.jsonl` per clip under the GT-marks dir
(`server._GT_MARKS_DIR`, default
`~/Library/Mobile Documents/com~apple~CloudDocs/footy_data/ball_gt_marks`).
One JSON object per line:

```json
{"frame_index": 14, "bbox": {"x":0.1,"y":0.2,"w":0.05,"h":0.08},
 "center": {"x":0.125,"y":0.24}, "tags": ["in_play_ball", "labeller"]}
```

- **LAB-101** (MUST) Box lines carry `frame_index`, normalized `bbox`
  (`x,y,w,h`), derived `center` (`x+w/2`, `y+h/2`), and
  `tags = [label, model]`. `center` is written on flush and ignored on load.
  - Tests: `test_session_roundtrip.py::test_roundtrip_preserves_model_tags_label_and_geometry`,
    `test_review_endpoints.py::test_correct_rewrites_line_stamped_labeller`
- **LAB-102** (MUST) Skip markers are lines with `bbox: null, center: null`
  and `tags` of exactly `["no_ball"]` or `["not_broadcast"]` (frame recorded,
  no box). Frames with neither boxes nor markers are not written at all.
  - Tests: `test_session_roundtrip.py::test_flush_writes_skip_markers_with_null_bbox`,
    `test_session_roundtrip.py::test_roundtrip_no_ball_and_not_broadcast_sets`
- **LAB-103** (MUST) Round-trip fidelity: everything the UI shows survives
  save → reload with identical `model` tags, labels, and geometry (see also
  `tests/feature_store/test_roundtrip_fidelity.py` for the feature-store leg).
  - Tests: `test_session_roundtrip.py::test_roundtrip_preserves_model_tags_label_and_geometry`,
    `test_session_roundtrip.py::test_roundtrip_multiple_boxes_per_frame_preserve_each_tag`
- **LAB-104** (MUST) Raw confidence is not persisted; restore synthesizes it:
  `1.0` for `labeller` boxes, `0.5` for machine boxes.
  - Tests: `test_session_roundtrip.py::test_roundtrip_confidence_by_provenance`
- **LAB-105** (SHOULD) Sidecar load is tolerant: blank lines and JSON-decode
  errors are skipped; `frame_index` outside `[0, total_frames)` is skipped;
  `bbox` may be a `{x,y,w,h}` dict or a 4-element list.
  - Tests: UNTESTED (dict form exercised by
    `test_http_endpoints.py::test_session_load_returns_metadata_and_restores_sidecar`;
    list form and malformed-line skipping have no dedicated test)
- **LAB-106** (MUST) Flush precedence per frame: `not_broadcast` > `no_ball` >
  boxes — a frame in a skip set writes only the marker line even if boxes
  exist in the timeline.
  - Tests: UNTESTED (only disjoint sets are exercised, by
    `test_session_roundtrip.py::test_roundtrip_no_ball_and_not_broadcast_sets`)
- **LAB-107** (MUST) Flush is debounced (2 s timer, reset on every schedule)
  and forced synchronously on clip switch. Flush failures are non-fatal
  (logged, never raised).
  - Tests: `test_http_endpoints.py::test_session_load_flushes_previous_clip_before_switch`
    (forced flush; the 2 s debounce timing itself is untested)
- **LAB-108** (SHOULD) On load, the box label is the first tag found in the
  known class set (ball + player classes). Unknown labels fall back to
  `"in_play_ball"` — see OPEN-4.
  - Tests: UNTESTED

## 3. Session state machine (LAB-2xx)

`session.Session` holds the single authoritative per-frame timeline
(`timeline[i]` = list of boxes, or `None` if never populated), guarded by a
lock. One global instance (`server.SESSION`) — the server is single-user by
design.

- **LAB-201** (MUST) `load(video_path)`: pause + replace the
  BackgroundLabeller, force-flush the *previous* clip's pending edits, cancel
  the debounce timer, read metadata via cv2 (fps defaulting to 25.0 if
  unreadable, total frames, width, height), reset timeline and skip sets, then
  restore the sidecar (§2). Returns `{fps, total_frames, width, height}`.
  Missing file raises `FileNotFoundError`.
  - Tests: `test_http_endpoints.py::test_session_load_returns_metadata_and_restores_sidecar`,
    `test_http_endpoints.py::test_session_load_flushes_previous_clip_before_switch`
- **LAB-202** (MUST) `get_frame(idx)` returns a copy of the frame's boxes and
  `[]` for unpopulated or out-of-range frames; `set_frame(idx, boxes)`
  overwrites a frame entirely, silently ignoring out-of-range writes.
  - Tests: `test_http_endpoints.py::test_timeline_returns_boxes_with_source`
- **LAB-203** (MUST) `merge_propagated(idx, boxes)` — the GT-authoritative
  merge: if the frame has ANY `labeller` box the frame is left completely
  untouched (not replaced, not augmented) and it returns `True` ("GT kept");
  otherwise the frame is overwritten with the propagated boxes and it returns
  `False`. Out-of-range: no-op, `False`.
  - Tests: `test_session_roundtrip.py::test_merge_propagated_keeps_gt_and_returns_true`,
    `test_session_roundtrip.py::test_merge_propagated_writes_machine_boxes_and_returns_false`,
    `test_session_roundtrip.py::test_merge_propagated_out_of_range_returns_false_no_crash`,
    `test_session_roundtrip.py::test_merge_propagated_empty_frame_writes_boxes_returns_false`
- **LAB-204** (MUST) `seed_objects(idx)` converts the frame's normalized boxes
  to absolute-pixel `LabelledObject`s to seed a propagation run.
  - Tests: `test_ws_run_protocol.py::test_run_streams_compiling_then_running_then_frames_then_done`
    (seed count from timeline)
- **LAB-205** (MUST) Wire format: server → client boxes are
  `{label, x, y, w, h, conf, source}` (`source` = provenance); client → server
  boxes are `{label, x, y, w, h, conf?, model?}` with coordinates clamped to
  [0,1] on ingest and `model` handled per LAB-003.
  - Tests: `test_http_endpoints.py::test_timeline_returns_boxes_with_source`,
    `test_http_endpoints.py::test_edit_clamps_coordinates`,
    `test_per_box_promotion.py::test_boxes_from_payload_model_fallback_rules`
- **LAB-206** (SHOULD) `server.py` is the composition root and config
  surface: extracted modules resolve `_GT_MARKS_DIR` / `_CLIPS_DIR` /
  `SESSION` through it at call time (tests monkeypatch there).
  - Tests: exercised implicitly by the whole HTTP suite via
    `conftest.py::patch_labeller_attr`

## 4. HTTP API — labeller core (LAB-3xx)

All request/response bodies are JSON unless noted.

- **LAB-301** (MUST) `GET /` and `GET /main` serve the hub page;
  `GET /labeller` the labeller; `GET /object_review` the review UI (all HTML).
  - Tests: `test_http_endpoints.py::test_root_and_main_serve_hub_page`,
    `test_http_endpoints.py::test_labeller_and_review_pages`
- **LAB-302** (MUST) `GET /clips` → `{clips: [{name, marked}], dir}` with no
  video IO: `marked` = sidecar exists; sorted by name; suffixes
  `.mp4/.mov/.avi/.mkv`; missing clips dir → `{clips: []}`.
  - Tests: `test_http_endpoints.py::test_clips_lists_videos_sorted_with_marked_flag`,
    `test_http_endpoints.py::test_clips_missing_dir_returns_empty`
- **LAB-303** (MUST) `GET /clips/status` → per-clip
  `{name, marked, complete, label_count}`: `complete` = sidecar reaches within
  15 frames of the clip end AND has ≥1 player-class tag (ball-only clips stay
  in-progress); parse errors degrade to
  `{marked: true, complete: false, label_count: 0}`.
  - Tests: `test_http_endpoints.py::test_clips_status_complete_requires_end_reached_and_player`
- **LAB-304** (MUST) `POST /session/load` `{video_path}` performs LAB-201 and
  returns its metadata dict.
  - Tests: `test_http_endpoints.py::test_session_load_returns_metadata_and_restores_sidecar`
- **LAB-305** (MUST) `GET /frame/{idx}.jpg` returns the frame as JPEG, and
  404 when no video is loaded or the frame cannot be read.
  - Tests: `test_http_endpoints.py::test_frame_jpeg_serves_bytes_and_404s_without_video`
- **LAB-306** (MUST) `GET /timeline/{idx}` → `{idx, boxes}` from the
  authoritative timeline (empty list, not an error, for unpopulated or
  out-of-range frames).
  - Tests: `test_http_endpoints.py::test_timeline_returns_boxes_with_source`
- **LAB-307** (MUST) `GET /next-detection/{from_idx}` → `{idx}` of the next
  frame strictly after `from_idx` with ≥1 box, else `{idx: null}`.
  - Tests: `test_http_endpoints.py::test_next_detection_finds_next_populated_frame`
- **LAB-308** (MUST) `GET /marks` → `{no_ball, not_broadcast, ball, player}`
  frame-index lists; `ball`/`player` = frames whose boxes contain at least one
  ball-class / player-class label.
  - Tests: `test_http_endpoints.py::test_marks_reports_ball_player_and_skip_sets`
- **LAB-309** (MUST) `POST /edit` `{idx, objects}` overwrites the frame with
  the client boxes (provenance per LAB-003, default `labeller`); a payload
  with ≥1 box removes the frame from both skip sets (an empty payload leaves
  them); schedules a debounced flush; returns `{idx, boxes}`.
  - Tests: `test_http_endpoints.py::test_edit_stamps_labeller_and_clears_skip_markers`,
    `test_http_endpoints.py::test_edit_with_empty_objects_keeps_skip_markers`,
    `test_per_box_promotion.py::test_edit_keeps_per_box_model_tags`
- **LAB-310** (MUST) `POST /no-ball` `{idx}` adds the frame to the no-ball set
  AND strips ball-class boxes from it (player boxes kept);
  `POST /no-ball/clear` removes the marker. Both schedule a flush.
  - Tests: `test_http_endpoints.py::test_no_ball_strips_ball_boxes_but_keeps_players`,
    `test_http_endpoints.py::test_no_ball_and_not_broadcast_clear_roundtrip`
- **LAB-311** (MUST) `POST /not-broadcast` / `POST /not-broadcast/clear`
  `{idx}` toggle the not-broadcast marker (boxes untouched); flush scheduled.
  - Tests: `test_http_endpoints.py::test_no_ball_and_not_broadcast_clear_roundtrip`
- **LAB-312** (MUST) `POST /autodetect`
  `{frame_idx, current_boxes, conf?, iou?, model_path?}`: with no video loaded
  → `{idx: 0, boxes: []}`. Otherwise the client's `current_boxes` replace the
  frame (provenance per LAB-003 — autodetect merges with what is *on screen*,
  never stale server state); YOLO runs on the frame (default conf 0.35, NMS
  IoU 0.5), detections are stamped `yolo`, and any YOLO box with IoU > 0.3
  against a current box is suppressed. Result (current + surviving YOLO) is
  written to the timeline and returned.
  - Tests: `test_http_endpoints.py::test_autodetect_without_video_returns_empty`,
    `test_http_endpoints.py::test_autodetect_merges_yolo_on_top_of_client_gt`,
    `test_http_endpoints.py::test_autodetect_with_no_current_boxes_keeps_all_yolo`
- **LAB-313** (MUST) `POST /propagate` `{frame_idx, box_idx}` ripples a
  hand-corrected label forward: only `labeller`-provenance source boxes
  propagate (else `{propagated_to: 0}`, also for out-of-range `box_idx`); the
  walk goes strictly forward matching the highest-IoU `yolo` box against the
  last position, skipping empty frames, stopping at the first frame containing
  any `labeller` box or when best IoU < 0.3; matched boxes get the reference
  label but KEEP `yolo` provenance. Returns `{propagated_to: n}`; flush
  scheduled.
  - Tests: `test_http_endpoints.py::test_propagate_relabels_matching_yolo_boxes_forward`,
    `test_http_endpoints.py::test_propagate_stops_at_frame_with_existing_gt`,
    `test_http_endpoints.py::test_propagate_stops_when_track_lost`,
    `test_http_endpoints.py::test_propagate_skips_empty_frames_and_continues`,
    `test_http_endpoints.py::test_propagate_refuses_non_labeller_source`,
    `test_http_endpoints.py::test_propagate_out_of_range_box_idx`

## 5. Review API (LAB-4xx)

Backed directly by the sidecar files, not the live Session timeline. A box's
identity is `(clip, frame_index, box_index)` where `box_index` is the ordinal
of the box among that frame's box lines in current file order (skip-marker and
bbox-null lines excluded).

- **LAB-401** (MUST) Queue, crop, correct, and delete all share the same
  `box_index` numbering (defined in one place:
  `review._iter_frame_box_lines`). Deleting or reordering lines renumbers
  boxes.
  - Tests: `test_review_endpoints.py::test_queue_excludes_skip_markers_and_dedups_by_iou`,
    `test_review_endpoints.py::test_correct_rewrites_line_stamped_labeller`,
    `test_review_endpoints.py::test_delete_removes_only_the_target_line`
- **LAB-402** (MUST) `GET /review/queue` scans all sidecars and returns
  `{total, items: [{clip, frame_index, box_index, bbox, label, confidence,
  provenance, image_url}]}`; skip markers excluded; label = first tag in the
  review class set (fallback `"player"` — see OPEN-4); confidence 1.0 for
  `labeller` lines else 0.5.
  - Tests: `test_review_endpoints.py::test_queue_orders_machine_before_gt_and_by_confidence`,
    `test_review_endpoints.py::test_queue_sam3_legacy_tag_round_trips`
- **LAB-403** (MUST) Queue ordering: machine-provenance items before
  `labeller` items, then ascending confidence, then rare classes weighted up
  (ball classes ×3, referee/coach/player_sub ×2).
  - Tests: `test_review_endpoints.py::test_queue_orders_machine_before_gt_and_by_confidence`,
    `test_review_endpoints.py::test_queue_rare_classes_weighted_before_players`
- **LAB-404** (MUST) Queue IoU dedup: within the same (clip, frame), items
  with IoU > 0.85 against an already-queued item are dropped from the queue
  only — they still exist in the file and still count for `box_index`
  numbering.
  - Tests: `test_review_endpoints.py::test_queue_excludes_skip_markers_and_dedups_by_iou`
- **LAB-405** (MUST) `GET /review/crop/{clip}/{frame}/{box}.jpg` returns a
  JPEG crop with 2.0× box-size padding per side, edge-clamped (JPEG quality
  85); 404 when the clip video or box is missing. Crops are cached in a
  200-entry LRU keyed by (clip, frame, box); cache hits need no video.
  - Tests: `test_review_endpoints.py::test_crop_serves_jpeg_and_caches`,
    `test_review_endpoints.py::test_crop_404s_for_missing_video_or_box`
- **LAB-406** (MUST) `GET /review/frame/{clip}/{frame}.jpg` returns the full
  frame (JPEG quality 80) or 404.
  - Tests: `test_review_endpoints.py::test_full_frame_endpoint`
- **LAB-407** (MUST) `POST /review/correct`
  `{clip, frame_index, box_index, label, bbox}` rewrites that box's line in
  place with the new label and bbox **stamped `labeller`** (GT promotion is
  the point of review), bbox clamped to the unit square with w/h clamped to
  fit, center recomputed; invalidates the crop-cache entry. Errors (HTTP 200):
  `{ok: false, error: "clip not found" | "box_index out of range"}`.
  - Tests: `test_review_endpoints.py::test_correct_rewrites_line_stamped_labeller`,
    `test_review_endpoints.py::test_correct_clamps_bbox_to_unit_square`,
    `test_review_endpoints.py::test_correct_invalidates_crop_cache`,
    `test_review_endpoints.py::test_correct_error_shapes`
- **LAB-408** (MUST) `POST /review/delete` removes exactly that box's line
  (same identity rules and error shapes); invalidates the crop-cache entry.
  - Tests: `test_review_endpoints.py::test_delete_removes_only_the_target_line`,
    `test_review_endpoints.py::test_delete_error_shapes`
- **LAB-409** (MUST) `POST /review/yolo` `{clip, frame_index}` runs the
  current-best YOLO detector on the frame and returns
  `{ok, boxes: [{label, confidence, x, y, w, h}]}` (values rounded to 3/4 dp);
  `{ok: false, error: "video not found", boxes: []}` when the video is
  missing.
  - Tests: `test_review_endpoints.py::test_review_yolo_returns_rounded_boxes`,
    `test_review_endpoints.py::test_review_yolo_missing_video`

## 6. Ingest API (LAB-5xx)

- **LAB-501** (MUST) `POST /ingest/upload` (multipart) saves the upload under
  the temp uploads dir and returns `{path, name, size}`.
  - Tests: `test_ingest_endpoints.py::test_upload_saves_file_and_returns_metadata`
- **LAB-502** (MUST) `GET /ingest/run?path=&sample=&merge_gap_s=&min_seg_s=`
  streams SSE (`text/event-stream`); a missing file yields
  `ERROR: file not found` then `[DONE]` without spawning a subprocess.
  - Tests: `test_ingest_endpoints.py::test_ingest_run_missing_file_streams_error_and_done`
- **LAB-503** (MUST) The happy path shells out to
  `python -m footy_track.scripts.split_broadcast_segments <path> --outdir
  <clips dir> ...`, echoes each output line as a `data:` event, then emits
  `[EXIT <rc>]` and `[DONE]`.
  - Tests: UNTESTED (requires a real video + subprocess run)

## 7. WebSocket run protocol (LAB-6xx)

Client → server: `{type: "run" | "restart" | "pause"}`; `run`/`restart` carry
`{start_frame, conf?, imgsz?, model_uri?}`. Server → client:
`{type: "status", state: "compiling" | "running" | "paused" | "idle"}`,
`{type: "frame", idx, boxes, gt_kept}`, `{type: "anomaly", idx, reason}`,
`{type: "done", last_frame}`, `{type: "error", message}`.

- **LAB-601** (MUST) `run` and `restart` are handled identically server-side
  (the distinction is a frontend labelling concern). Any in-flight streamer is
  cancelled and the current BackgroundLabeller paused first.
  - Tests: `test_ws_run_protocol.py::test_restart_message_behaves_like_run`
- **LAB-602** (MUST) Runs seed from the TIMELINE at `start_frame`
  (`Session.seed_objects`), never from client-supplied boxes; the frontend
  commits the canvas to `/edit` first, making Run/Restart deterministic. No
  boxes on the start frame → `{type: "error", message: "No boxes on frame N
  to seed from."}` and nothing starts.
  - Tests: `test_ws_run_protocol.py::test_run_with_no_seed_boxes_sends_error`,
    `test_ws_run_protocol.py::test_run_streams_compiling_then_running_then_frames_then_done`
- **LAB-603** (MUST) `status: compiling` is sent BEFORE the (potentially slow,
  blocking) `bg.submit(...)` so the frontend can show the loading overlay for
  the whole model warmup (ft-wkc); the streamer then re-announces `compiling`
  and sends `running` just before the first frame message.
  - Tests: `test_ws_run_protocol.py::test_run_streams_compiling_then_running_then_frames_then_done`
- **LAB-604** (MUST) The streamer fetches newly-completed frames via
  `frame_at(idx)` — NOT `completed_frames()` — so mid-clip runs work: a run
  seeded at frame N (frames 0..N-1 still None) MUST stream and ingest frames
  N..M. (Regression: the contiguous-from-0 scan silently skipped every frame —
  "ran to frame 30 but 28–29 have no boxes".)
  - Tests: `test_ws_run_protocol.py::test_mid_clip_run_streams_frames_from_start_frame`,
    `test_video_utils.py::test_frame_at_serves_mid_clip_frames_where_completed_frames_cannot`
- **LAB-605** (MUST) Frame ingestion: the seed frame's timeline entry is
  emitted as-is (`gt_kept: false`) — the tracker's re-detection never
  overwrites it; downstream detections are stamped `vittrack` and merged via
  LAB-203, with `gt_kept: true` in the frame message when existing GT made
  the merge a no-op (so the frontend can report "kept your marks on frames
  14–17" instead of skipping silently).
  - Tests: `test_ws_run_protocol.py::test_run_seed_frame_kept_verbatim_and_downstream_stamped_vittrack`,
    `test_ws_run_protocol.py::test_run_reports_gt_kept_frames`
- **LAB-606** (MUST) Anomaly handback: when the labeller flags an anomaly the
  server sends `{type: "anomaly", idx, reason}` then `status: paused`, clears
  the anomaly marker, and the streamer exits (run stays paused for correction
  and restart). Frames up to the anomaly are still streamed and ingested.
  - Tests: `test_ws_run_protocol.py::test_anomaly_handback_pauses_run`
- **LAB-607** (MUST) Normal completion sends `{type: "done", last_frame}`
  then `status: idle`.
  - Tests: `test_ws_run_protocol.py::test_run_streams_compiling_then_running_then_frames_then_done`
- **LAB-608** (MUST) `pause` pauses the BackgroundLabeller, cancels the
  streamer, and acknowledges with `status: paused`.
  - Tests: `test_ws_run_protocol.py::test_pause_message_pauses_and_acknowledges`
- **LAB-609** (MUST) WebSocket disconnect pauses the run and cancels the
  streamer (no orphaned propagation).
  - Tests: UNTESTED
- **LAB-610** (MUST) The user can control, per class, how confident the
  tracker must be before a run continues without pausing: each run applies the
  user's per-class handback thresholds (classes: `player`, `referee`,
  `in_play_ball`, and `other` as the catch-all), so solid-but-low-confidence
  tracks no longer pause the run frame after frame unless the user has asked
  for that strictness. Malformed threshold input MUST NOT break a run —
  invalid values are ignored and defaults apply.
  - Tests: `test_ws_run_protocol.py::test_run_passes_sanitized_handback_thresholds`

## 8. Propagation backend — video_utils (LAB-7xx)

- **LAB-701** (MUST) `LabelledObject` requires exactly one of
  `bbox_xyxy_abs` / `point_xy_abs`; `VitTrackVideoLabeller` requires ≥1
  object and an existing video path (extra kwargs absorbed for API compat).
  - Tests: `test_video_utils.py::test_labelled_object_requires_exactly_one_seed`,
    `test_video_utils.py::test_vittrack_labeller_rejects_empty_objects_and_missing_video`
- **LAB-702** (MUST) Each yielded `FrameDetections.uri` encodes the absolute
  frame index (`<stem>_frame_<%06d>`); `_frame_index_from_uri` recovers it,
  falling back to a caller default on parse failure.
  - Tests: `test_video_utils.py::test_frame_index_from_uri_roundtrip`,
    `test_video_utils.py::test_frame_index_from_uri_fallback_on_unparseable`
- **LAB-703** (MUST) VitTrack propagation: one independent `VitTrackSOT` per
  seeded object, warmed on the seed frame; the seed frame is yielded FIRST
  with the user's boxes verbatim (model `vittrack`, confidence 1.0); per
  frame each tracker updates its box, a tracker miss carries the previous box
  forward with the low score; `stop_event` stops cleanly after the current
  frame; progress reports ABSOLUTE position `(abs_idx + 1, total)`.
  - Tests: UNTESTED directly (needs real video IO); the seed-verbatim and
    ordering contract is pinned at the ws level by
    `test_ws_run_protocol.py::test_run_seed_frame_kept_verbatim_and_downstream_stamped_vittrack`
- **LAB-704** (MUST) `BackgroundLabeller` runs the labeller in a daemon
  thread, slotting frames by absolute index and tracking
  `last_completed_frame`; worker exceptions land in `.error`; `running` is
  false on exit; `pause()` is safe without a live thread; `is_done()` = not
  running AND ≥1 frame completed.
  - Tests: `test_video_utils.py::test_worker_slots_frames_by_absolute_index`,
    `test_video_utils.py::test_worker_records_errors`,
    `test_video_utils.py::test_pause_without_thread_is_safe`,
    `test_video_utils.py::test_is_done_semantics`
- **LAB-705** (MUST) `frame_at(idx)` returns the completed frame at an
  absolute index regardless of holes before it; `completed_frames()` is the
  legacy contiguous-from-0 scan (retained, no longer used by the streamer).
  - Tests: `test_video_utils.py::test_frame_at_serves_mid_clip_frames_where_completed_frames_cannot`
- **LAB-706** (MUST) Anomaly auto-stop (per frame, when `anomaly_detection`
  is on): (a) motion/size heuristic vs the previous frame — nearest
  same-label box centre jumped > 40% of the frame diagonal, or area changed
  > 8×; (b) else confidence handback — any detection below the 0.5 threshold.
  On anomaly: `anomaly_frame`/`anomaly_reason` set, stop event set, worker
  exits. A brand-new label appearing is NOT an anomaly. The switch can be
  disabled.
  - Tests: `test_video_utils.py::test_anomaly_none_for_small_motion`,
    `test_video_utils.py::test_anomaly_on_large_centre_jump`,
    `test_video_utils.py::test_anomaly_on_area_explosion`,
    `test_video_utils.py::test_new_label_is_not_an_anomaly`,
    `test_video_utils.py::test_worker_confidence_handback_stops_run`,
    `test_video_utils.py::test_worker_motion_anomaly_stops_run`,
    `test_video_utils.py::test_worker_anomaly_detection_can_be_disabled`
- **LAB-707** (MUST) `submit(...)` pauses any current job, re-allocates the
  frames array only when the clip's frame count changed (earlier frames
  survive a restart), resets error/anomaly state, and starts progress at
  `(start_frame, total)`.
  - Tests: UNTESTED (constructor needs a real video; the surrounding protocol
    is exercised via the scripted fake in `test_ws_run_protocol.py`)
- **LAB-708** (MUST) `yolo_seed_objects` runs the current-best (or explicit)
  detector on the chosen frame and returns greedy-NMS-filtered absolute-pixel
  seed objects (highest confidence kept, overlaps above the IoU threshold
  dropped).
  - Tests: `test_video_utils.py::test_nms_keeps_highest_confidence_and_drops_overlaps`,
    `test_video_utils.py::test_nms_empty_input` (NMS core; the full function
    needs a real detector)
- **LAB-709** (MUST) VitTrack warm start: the ONNX session is cached
  process-wide per model path (one session serves all tracker instances;
  per-instance tracking state independent; HuggingFace download memoized) so
  "Compiling model…" happens once per server start.
  - Tests: `test_sot_vittrack_cache.py::test_cached_session_reuses_same_object_for_same_path`,
    `test_sot_vittrack_cache.py::test_cached_session_distinguishes_by_path`,
    `test_sot_vittrack_cache.py::test_two_vittracksot_instances_share_identical_session`,
    `test_sot_vittrack_cache.py::test_vittracksot_instances_have_independent_per_instance_state`,
    `test_sot_vittrack_cache.py::test_download_model_memoized_across_calls`,
    `test_sot_vittrack_cache.py::test_vittracksot_uses_download_model_when_no_path_given`
- **LAB-710** (SHOULD) The motion-guided tracker (`motion_tracker.py`:
  Kalman-predicted ROI crop → CropRunner detect → map back to frame coords,
  full-frame re-acquire on miss) behaves per its design doc; it is not yet
  the labeller's propagation backend.
  - Tests: `test_motion_tracker.py` (17 tests: box round-trips, ROI
    computation/clamping/velocity growth, crop mapping, Kalman lifecycle,
    reset/step/reacquire/miss paths)
- **LAB-711** (SHOULD) Legacy SAM3 code (`Sam3VideoLabeller`,
  `get_cached_predictor`, `warmup_model`, `_default_model_uri`) is retained:
  exported from the package `__init__`, used by
  `scripts/proto_sam3_points.py`, and the CropRunner re-acquire backend for
  `motion_tracker`. Removing it is a spec change (and LAB-007 must hold
  regardless).
  - Tests: UNTESTED (import surface only)
- **LAB-712** (MUST) The confidence handback is per-class: a run pauses on a
  tracked box only when its confidence falls below the threshold for that
  box's class (unlisted classes use the catch-all; with no user thresholds at
  all, a single global default applies). The pause reason names the class and
  the threshold that tripped it.
  - Tests: `test_video_utils.py::test_worker_handback_uses_per_class_thresholds`
- **LAB-713** (MUST) Restarting from frame N re-propagates everything after N
  from the corrected seed: the previous run's machine output beyond N is
  discarded and never shown as if it were the new run's result (no jumping
  ahead to the old run's last frame). Hand marks after N are NOT wiped by
  this — they survive per LAB-002 and are reported as kept.
  - Tests: `test_video_utils.py::test_submit_restart_discards_stale_downstream_frames`,
    `test_ws_run_protocol.py::test_run_reports_gt_kept_frames`

## 9. Frontend — labeller UI (`web/index.html`) (LAB-8xx)

No JS test runner exists in this repo, so these are all UNTESTED-FRONTEND;
their server contract is §§3–7. Verify by hand when touching the frontend.

- **LAB-801** (MUST) Tool modes: `draw` (default) drag-creates a box of the
  selected class (always GT; boxes < 4px discarded); `edit` drags/resizes via
  the transformer, click selects, Delete/Backspace removes the selection.
  Drawing is disabled while running.
  - Tests: UNTESTED-FRONTEND
- **LAB-802** (MUST) Keyboard map (all ignored while typing in an
  input/select):

  | Key | Action |
  |---|---|
  | `w` / `e` | draw / edit tool |
  | `1–6` | class hotkeys (in_play_ball, ball, referee, player, person, coach) + switch to draw |
  | `←`/`→`, `a`/`d` | prev/next frame; `Shift` ×10, `Ctrl/Cmd` ×50 |
  | `f` (tap-count) | 1 tap = ½ s, 2 taps = 1 s, 3+ taps = 4 s worth of frames |
  | `g` | next frame with detections (`/next-detection`) |
  | `n` / `b` | toggle no-ball / not-broadcast on current frame |
  | `q` | re-run autodetect on current frame |
  | `r` | Run (idle) / Restart (paused) the propagation run |
  | `Space` | pause the propagation run (only while running) |
  | `z` | undo (20-deep per-frame snapshot stack) |
  | `Delete`/`Backspace` | delete selected box (edit tool) |

  - Tests: UNTESTED-FRONTEND
- **LAB-803** (MUST) Timeline bar (canvas strip under the scrubber), per
  frame: red = no-ball, blue = not-broadcast, yellow = ball only,
  yellow-over-green = ball + player, green with a thin red top stripe =
  player-only (ball status undecided); white 2px cursor = current frame.
  Populated from `/marks` on load, updated live from ws `frame` messages and
  local edits.
  - Tests: UNTESTED-FRONTEND
- **LAB-804** (MUST) Clip picker: list from `/clips` (grey `✓` = marked),
  lazily upgraded from `/clips/status` (green `✅` + label-count tooltip =
  complete); refreshed every 20 s; clicking a clip saves current edits, clears
  the canvas, and loads it; active clip highlighted.
  - Tests: UNTESTED-FRONTEND (server legs: LAB-302/303)
- **LAB-805** (MUST) On clip load: run-control UI fully reset (Run visible,
  Restart hidden, pause disabled, hint/overlay cleared), scrubber sized,
  marks restored from `/marks`, frame 0 shown; saved boxes are loaded if
  frame 0 has any, otherwise autodetect seeds frame 0; the ws is
  (re)connected. Last video/model paths persist in localStorage.
  - Tests: UNTESTED-FRONTEND
- **LAB-806** (MUST) The server timeline is the single source of truth:
  leaving an edited frame POSTs `/edit` (each box carrying its own `model`
  per LAB-003/005); entering a frame GETs `/timeline/{idx}`. `frameDirty`
  gates saves. After a manual save, each `labeller` box is `/propagate`d
  forward and the ripple count is surfaced in the status line.
  - Tests: UNTESTED-FRONTEND (server legs: LAB-309/313)
- **LAB-807** (MUST) Run/Restart commit the canvas to `/edit` at the current
  frame, clear the kept-frames set, and send ws `run`/`restart` with
  `start_frame` = current frame. Running UI: pause enabled, autodetect
  disabled, edit tool forced, compiling overlay shown. The overlay is a
  SIBLING of the Konva stage container (Konva wipes its container's
  innerHTML) and is dropped on the first live frame or non-compiling status.
  - Tests: UNTESTED-FRONTEND
- **LAB-808** (MUST) Live `frame` messages update `lastCompleted`, the
  timeline-bar sets, and kept-frames, but render to the canvas only while
  mode is `running` (stale in-flight frames after a pause must not move the
  view). `status: paused` → jump to the last live frame, editable, Restart
  labelled "Restart from frame N" (N follows scrubbing while paused).
  `anomaly` → auto-pause on the anomaly frame with the amber reason banner.
  `done` → jump to the last completed frame, idle-with-results UI; the
  kept-frames summary ("kept your marks on frames 14–17", ranges collapsed)
  is appended to the paused/done hint.
  - Tests: UNTESTED-FRONTEND (server legs: LAB-604–607)
- **LAB-809** (MUST) WS resilience: every send goes through `ensureWS()`
  (reconnect on demand, 5 s timeout — a `ws.send` on a dead socket is
  silently discarded by browsers and used to hang Run at "Compiling model…");
  `onclose` auto-reconnects with exponential backoff (500 ms → 8 s cap);
  failures surface in the status line.
  - Tests: UNTESTED-FRONTEND
- **LAB-810** (MUST) Rendering follows the hierarchy (LAB-004): lower tiers
  drawn first so GT sits on top; the objects pane and on-canvas numbers use
  the same tier-ordered list; machine boxes render dashed/faded, GT solid
  (and a promoted box turns solid immediately, LAB-005).
  - Tests: UNTESTED-FRONTEND
- **LAB-811** (MUST) Objects pane: per-box class dropdown (read-only while
  running; changing a class promotes per LAB-005), delete `✕`, "Clear all"
  with confirm + undo; no-ball / not-broadcast rows shown with inline clear.
  - Tests: UNTESTED-FRONTEND
- **LAB-812** (MUST) The user can adjust the per-class handback thresholds
  (LAB-610) from the labeller page itself, without a restart; settings persist
  across sessions and apply from the next run.
  - Tests: UNTESTED-FRONTEND (sliders under the objects pane)
- **LAB-813** (SHOULD) When a shown machine box is below its class's handback
  threshold, its number is highlighted (red) until the user next clicks the
  canvas — a lightweight "check this one" cue that doesn't block anything.
  - Tests: UNTESTED-FRONTEND
- **LAB-814** (MUST) An edit to a box (move/resize/relabel/delete/draw) is
  persisted shortly after the edit itself — not only when the user navigates
  away or starts a run. Refreshing or closing the page right after an edit
  must not lose it.
  - Tests: UNTESTED-FRONTEND (debounced save; server leg is LAB-306 /edit)

## 10. Frontend — review UI (`web/review.html`) (LAB-9xx)

- **LAB-901** (MUST) Grid of crops grouped by class (pills with counts),
  batch size 100, in server-queue order; cards show label, confidence +
  provenance badge (machine only), bbox coords, and a GT-box overlay drawn in
  crop space.
  - Tests: UNTESTED-FRONTEND (server legs: LAB-402–405)
- **LAB-902** (MUST) Selection via shift/cmd-click or double-click; `s`
  select all, Esc clear. Batch accept (mark seen), relabel (class picker →
  `/review/correct` with existing bbox), delete (`/review/delete`).
  - Tests: UNTESTED-FRONTEND
- **LAB-903** (MUST) Accept/relabel mark items **seen** (green border,
  persisted in localStorage `review_seen_v1`) rather than removing them;
  "Hide reviewed" toggle (default on) filters seen items.
  - Tests: UNTESTED-FRONTEND
- **LAB-904** (MUST) Modal: Konva-editable crop (draw/edit tools) beside the
  full frame with an SVG GT overlay; YOLO re-run overlays same-class
  detections dashed; accept saves bbox changes then advances; relabel stays
  in the modal; `a`/`d`/arrows navigate with auto-save; Space accepts;
  Backspace/Delete deletes; Esc auto-saves and closes.
  - Tests: UNTESTED-FRONTEND
- **LAB-905** (MUST) All crop-space ↔ frame-space conversions use the same
  pad-2.0 edge-clamped mapping as the server crop (LAB-405) — card overlays,
  modal box drawing/editing, and YOLO overlay mapping.
  - Tests: UNTESTED-FRONTEND

## 11. Operational notes (informative)

- Single global session; no auth; intended for one local user.
- Downstream, sidecars + Roboflow datasets are ingested into the DuckDB
  feature store (`feature_store/ingest_gt.py`) and exported as leakage-free
  YOLO training datasets (`scripts/export_training_dataset.py`).
- All timestamps follow the ContinuousTime / GameTime conventions
  (`docs/timings.md`).
- The torch.compile (Inductor) cache is pinned to
  `~/.cache/footy_torch_inductor` before torch import (macOS purges $TMPDIR).

## 12. Known gaps — OPEN items

Tracked against this spec; burn these down by fixing the behavior AND
updating/adding the corresponding LAB requirement + tests.

- **OPEN-1** `GET /ingest` serves `web/ingest.html`, which does not exist —
  the route 500s. The `/ingest/upload` and `/ingest/run` APIs (LAB-501/502)
  work. Fix: add the page or remove the route + hub link.
- **OPEN-2** The labeller frontend's "+ Add clip" button POSTs `/clips/add`,
  which has no server route (404). Dead frontend feature — implement or
  remove.
- **OPEN-3** The review queue's provenance tag set omits `vittrack`, so
  vittrack boxes surface as `provenance: "labeller"` (while still getting
  machine confidence 0.5) and sort/badge as GT. Current behavior is pinned by
  `test_review_endpoints.py::test_queue_vittrack_tag_reported_as_labeller_current_behavior`
  — flip that test when fixing.
- **OPEN-4** Unknown-label fallbacks are silent: sidecar restore falls back
  to `"in_play_ball"` (LAB-108), the review scan to `"player"` (LAB-402). A
  mislabelled tag is masked rather than surfaced.
- **FIXED** Undo previously restored boxes without their `source`, promoting
  machine boxes to GT — fixed on main `95b60cc` (now LAB-006).

### Backlog / deferred (not defects)

- Feature-store scan on load: the load path only reads the JSONL sidecar;
  labels that exist only in the DuckDB feature store are not surfaced.
- Roboflow upload leg: local YOLO export is built and fidelity-tested;
  pushing an export back to Roboflow as a new dataset version is not wired
  (see PR #18's skip-by-default live test).
- VitTrack handback threshold: 0.5 on main (LAB-706); a local experiment
  suggested 0.3 — unresolved.
- Per-machine clip symlinks: `eval_data/clips/*` targets are machine-specific
  (macOS iCloud vs Linux `/mnt/storage`) and get clobbered when either
  machine "fixes" them; needs per-machine config + regeneration script.
- GPU propagation loop (`ft-8vx` epic): paused by request.

## 13. Traceability summary

| Section | Requirements | Tested | UNTESTED | UNTESTED-FRONTEND |
|---|---|---|---|---|
| 1. Label hierarchy (LAB-0xx) | 7 | 4 | 0 | 3 |
| 2. Persistence (LAB-1xx) | 8 | 5 | 3 | 0 |
| 3. Session (LAB-2xx) | 6 | 6 | 0 | 0 |
| 4. HTTP API (LAB-3xx) | 13 | 13 | 0 | 0 |
| 5. Review API (LAB-4xx) | 9 | 9 | 0 | 0 |
| 6. Ingest API (LAB-5xx) | 3 | 2 | 1 | 0 |
| 7. WebSocket (LAB-6xx) | 9 | 8 | 1 | 0 |
| 8. Backend (LAB-7xx) | 11 | 8 | 3 | 0 |
| 9. Labeller UI (LAB-8xx) | 11 | 0 | 0 | 11 |
| 10. Review UI (LAB-9xx) | 5 | 0 | 0 | 5 |
| **Total** | **82** | **55** | **8** | **19** |

Open items: 4 (OPEN-1..4) + 1 fixed on main (undo provenance, `95b60cc`).
Test suite: 103 tests in `tests/labeller/` (all green), plus the
feature-store round-trip fidelity leg in `tests/feature_store/`.
