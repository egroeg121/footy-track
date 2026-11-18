# Footy Track Labeller — Specification

**This document is the canonical, spec-driven source of truth for the
labeller.** Every feature and behavior is captured as a numbered requirement
(`LAB-nnn`) with MUST/SHOULD wording, grouped by subsystem, and traced to the
test(s) that pin it. Requirements with no server-side test are marked
`UNTESTED`; behaviors that live only in the browser JS (no JS test runner in
this repo) are marked `UNTESTED-FRONTEND` — their *server contract* is tested
instead. Known defects and gaps are tracked as `OPEN-n` items in §14.

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
| `/ball_check` | Mobile ball verification (swipe yes/no) |
| `/calibration` | Manual pitch calibration (frame <-> pitch point pairs, §16) |
| `/calibration/review` | Mobile accept/reject of machine pitch-calibration proposals (§16, LAB-1210..1219) |

Test suite: `tests/labeller/` (run `uv run pytest tests/labeller/`). Test
references below are `file::test_name` within `tests/labeller/`.

---

## 1. Label hierarchy & provenance (LAB-0xx)

Every box records where it came from in its `ObjectDetection.model` field,
persisted in the JSONL sidecar as the second tag (`tags: [label, model, *markers]`):

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
  OPEN, see §14).
  - Tests: UNTESTED-FRONTEND (`undoLast` passes `o.model` back to `addRect`)
- **LAB-007** (MUST) The legacy `sam3` model tag keeps round-tripping through
  sidecars and surfaces in review (old sidecars contain it), even though SAM3
  is no longer the propagation backend.
  - Tests: `test_review_endpoints.py::test_queue_sam3_legacy_tag_round_trips`,
    `test_session_roundtrip.py::test_roundtrip_preserves_model_tags_label_and_geometry`

The next two requirements are **PLANNED**: nothing implements them and they
have no tests yet. They belong to the proposed Claude-assisted ball labelling
loop (`docs/design/claude_autolabel_loop.md`); they are not counted in the
traceability summary in section 15.

- **LAB-008** (PLANNED, MUST) A Claude proposal has provenance `claude`, which
  sits at the machine level (tier 3). It lives only in a separate candidate
  pool, never in a GT sidecar, and it never overwrites or replaces a box at or
  above its own level (`labeller`, or a machine box carrying `human_checked`).
  This restates LAB-002 for the new tag. Adding `claude` to the provenance tag
  sets (`session.py`, `review.py`, `ingest_gt.py`) is part of the work.
  - Tests: none yet
- **LAB-009** (PLANNED, MUST) The `claude` tag is kept on a box through every
  later step: a person's yes (LAB-1014) writes `[label, claude, human_checked]`,
  so the origin of a tier 2 label can always be told, and the ball eval keeps
  ignoring it as ground truth.
  - Tests: none yet

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
  `tags = [label, model, *markers]`. `center` is written on flush and ignored on load.
  - **Marker-tag rule:** tags after the provenance (e.g. `human_checked`,
    which makes a `vittrack` box tier 2) are preserved verbatim across
    load -> flush, keyed to the box's label/model/geometry. A user edit that
    moves or relabels a box (it becomes `labeller`, tier 1) writes
    `[label, labeller]` with no markers, as `review.py`/`ball_check.py`
    corrections do. Untouched boxes keep their markers.
  - Tests: `test_session_roundtrip.py::test_marker_tags_survive_load_edit_flush`
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
  - Note: the detector now actually uses its configured settings. Earlier,
    `UltralyticsObjectDetector.predict_from_path` ignored them and ran with
    the ultralytics defaults (imgsz 640, conf 0.25, iou 0.7). It now passes
    the constructor's `min_confidence`, `iou_threshold` and `imgsz` (1280 by
    default) through, so boxes from the labeller's YOLO seeding and review
    calls can differ from ones produced before this fix (ball recall in
    particular is much better at 1280).

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
- **LAB-714** (MUST) Hand-fixed labels are the ground-truth seed even when a
  run passes over them again: when propagation reaches a downstream frame
  that carries hand-fixed labels, that frame acts as a fresh seed — the
  tracker continues from the frame's (human-corrected) content, not from its
  own trajectory. Human corrections are never merely displayed-but-ignored by
  a passing run, and re-seeding at such a frame is not itself an anomaly.
  - Tests: `test_video_utils.py::test_run_reseeds_from_gt_frames_midrun`,
    `test_video_utils.py::test_worker_skips_anomaly_checks_on_reseed_frames`

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
- **LAB-813** (SHOULD) A machine box below its class's handback threshold has
  its number highlighted (red) for as long as it remains an uncorrected
  machine box — the cue persists across clicks and navigation, and clears
  only when the box is corrected/promoted to GT. Saving a frame must not
  erase the underlying confidences (they round-trip through edits).
  - Tests: UNTESTED-FRONTEND (conf round-trip server leg: LAB-003 payload
    `conf` passthrough)
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
- **LAB-906** (MUST) The modal's mini (crop) Konva view has `w`/`e` keyboard
  shortcuts for its two box-manipulation modes, mirroring the main labeller's
  tool convention (index.html's `toolDraw`/`toolEdit`, §9): `w` switches to
  draw mode (draw a replacement box by dragging on the crop), `e` switches to
  edit mode (select/drag/resize the existing box via the Konva transformer).
  Both are also reachable via the header's "Draw"/"Edit" buttons. Relabelling
  ("wrong class" → open the class picker) is bound to `c`, not `w`, to avoid
  colliding with draw mode. Keys are ignored while an input/select has focus.
  - Tests: UNTESTED-FRONTEND
- **LAB-907** (MUST) The full-frame (wide) context panel's SVG box overlay
  must render at the same on-screen position as the crop panel's box for the
  same underlying bbox, including when the frame is letterboxed inside the
  panel (frame aspect ratio ≠ panel aspect ratio). The overlay computes an
  `object-fit:contain` rendered-image rect (offX/offY/rW/rH) and sets the
  `<svg>` `viewBox` to that rect; because the viewBox's min-x/min-y already
  shifts the SVG's coordinate origin to the letterboxed image's top-left
  corner, child element coordinates must be expressed in image-relative
  pixels only (`x*rW`, `y*rH`, …) — re-adding offX/offY on top of the
  viewBox shift double-applies the offset and misplaces the box whenever
  there is any letterboxing.
  - Tests: UNTESTED-FRONTEND

## 11. Ball Check — mobile ball verification (LAB-10xx)

Phone-sized yes/no verification of machine ball boxes: one zoomed, reticled
crop per card, swipe or tap to answer. It reads the same sidecars as review
and shares review's `(clip, frame_index, box_index)` box identity, but writes
**only** to its own append-only log — a yes/no answer does not check box
geometry, so it may not promote a machine box to `labeller` GT (LAB-002), and
an append-only side file cannot truncate hand labels.

Backend `ball_check.py`, page `web/ball_check.html` (no framework, no Konva).
Verdicts live in `<gt dir>/ball_checks/<clip>.jsonl`, one JSON object per
line: `{clip, frame_index, box_index, source, confidence, verdict, bbox, label,
provenance, corrected_bbox, ts}` (`source` is `sidecar` or `cand`; a
`verdict: null` line is an undo tombstone).

The log is **consumed downstream**: `python -m footy_track.feature_store.ingest_gt`
reads `<gt dir>/ball_checks/` (`--ball-checks-dir`) into the feature store's
`detection_review` table — `ball`/`corrected` mark the matching detection
`reviewed`, `not_ball` is a rejected negative that
`scripts/export_training_dataset.py` drops from positives. See
`docs/design/feature_store.md` §6.2.

- **LAB-1001** (MUST) `GET /ball_check/queue` returns only ball-class boxes
  (`BALL_LABELS`) whose provenance tag is machine (`yolo`/`vittrack`/`sam3`),
  as `{remaining, judged, items: [{clip, frame_index, box_index, bbox, label,
  provenance, image_url}]}`. `remaining` counts the whole queue, `items` is
  capped at `limit` (default 50, max 500).
  - Tests: `test_ball_check.py::test_queue_only_machine_ball_boxes`,
    `test_ball_check.py::test_queue_order_is_stable_and_respects_limit`
- **LAB-1002** (MUST) Provenance here is read from the sidecar tags
  (`review._parse_jsonl_box`'s `provenance_tag`), not review's `provenance`
  field — the latter reports `vittrack` boxes as `labeller` (OPEN-3), which
  would silently hide machine output from the queue.
  - Tests: `test_ball_check.py::test_queue_only_machine_ball_boxes`
- **LAB-1002a** (MUST) Boxes whose clip has no video on disk are dropped from
  the queue: the sidecar stems and the clip filenames follow different naming
  schemes, and a card that 404s is worse than no card.
  - Tests: `test_ball_check.py::test_queue_skips_clips_with_no_video`
- **LAB-1003** (SHOULD) Queue order is a deterministic hash of
  (clip, frame, box), not file order: precision is a property of the detector,
  so a short session should sample the whole corpus, and the order must be
  stable across restarts. `?clip=` narrows to one clip; `?include_gt=true`
  also surfaces `labeller` boxes.
  - Tests: `test_ball_check.py::test_queue_order_is_stable_and_respects_limit`,
    `test_ball_check.py::test_queue_include_gt_and_clip_filters`
- **LAB-1004** (MUST) `POST /ball_check/verdict` appends one record to the
  clip's verdict log and leaves the GT sidecar byte-identical. `verdict` must
  be `ball` | `not_ball` | `unsure`; anything else, or a missing identity
  field, returns `{ok: false}` and writes nothing. `unsure` is kept distinct
  so ambiguous crops never become negatives.
  - Tests: `test_ball_check.py::test_verdict_appends_to_log_and_leaves_sidecar_untouched`,
    `test_ball_check.py::test_verdict_rejects_unknown_value`,
    `test_ball_check.py::test_verdict_rejects_missing_fields`
- **LAB-1005** (MUST) A box with a verdict leaves the queue; the last line for
  a key wins, so re-judging corrects a mistake without editing history.
  - Tests: `test_ball_check.py::test_judged_boxes_leave_the_queue`,
    `test_ball_check.py::test_latest_verdict_wins`
- **LAB-1006** (MUST) `POST /ball_check/undo` appends a tombstone
  (`verdict: null`) rather than rewriting the log; the box returns to the
  queue and both lines stay on disk. Undo with no log for the clip is an
  error.
  - Tests: `test_ball_check.py::test_undo_returns_the_card_to_the_queue`,
    `test_ball_check.py::test_undo_without_verdicts_is_an_error`
- **LAB-1007** (SHOULD) `GET /ball_check/stats` returns per-verdict counts and
  `precision = ball / (ball + not_ball)`, `null` before any decided verdict.
  `unsure` is excluded from the denominator.
  - Tests: `test_ball_check.py::test_stats_precision_excludes_unsure`,
    `test_ball_check.py::test_stats_precision_none_before_any_decision`
- **LAB-1008** (MUST) `GET /ball_check/crop/{clip}/{frame}/{box}.jpg` returns
  a centred window of `1 + 2*pad` box sides, floored to 160 px and upscaled to
  640 px wide (an ~11 px ball is otherwise unjudgeable), with a yellow reticle
  drawn on the claimed box — without it a crop holding two round bright things
  is ambiguous. `?pad=` (default 6.0) is clamped to 40.0. 404 when the clip
  video or the box is missing.
  - Tests: `test_ball_check.py::test_crop_window_floors_tiny_boxes_and_clamps_to_edges`,
    `test_ball_check.py::test_crop_404s_for_missing_clip_or_box`,
    `test_ball_check.py::test_pad_is_clamped_to_max`
- **LAB-1009** (SHOULD) Crops are cached in a 200-entry LRU keyed by
  (clip, frame, box, pad) — a different pad is a separate entry, never a stale
  hit.
  - Tests: `test_ball_check.py::test_crop_returns_jpeg_and_caches`
- **LAB-1009a** (SHOULD) `GET /ball_check/clip/{clip}/{frame}/{box}.mp4`
  (same `src`/`pad` params as the crop) returns a ~3 s H.264 MP4 (frame -1.5 s
  .. +1.5 s, clamped to the clip, every source frame up to 30 fps, subsampled
  above that) so a spare/crowd ball can be told from the match ball by how it
  moves. It uses the still's crop window, fixed for every frame, upscaled
  server-side (Lanczos) to 560 px wide, full colour, CRF 20, yuv420p, main
  profile, even dimensions, moov atom first (iOS Safari). The box (and for
  `src=list` the magenta other box) is drawn on the centre frame only, which is
  repeated for 500 ms so the box is visible and exact. The response honours
  HTTP `Range` (206 / 416; iOS requires it). Raw frames are piped from one seek
  + sequential OpenCV reads into ffmpeg; results sit in a 40-entry LRU keyed by
  (src, clip, frame, box, pad). Headers `X-Clip-Frames` / `X-Clip-Duration-Ms`.
  If ffmpeg is not installed the same URL serves the old small GIF
  (`image/gif`, 240 px, 40 colours, ~10 fps).
  - Tests: `test_ball_check.py::test_clip_endpoint_serves_mp4_with_range_and_clamps`,
    `test_clip_frame_indices_clamp_to_the_video`, `test_parse_range`,
    `test_parse_range_unsatisfiable`, `test_mp4_writer_output_is_ios_playable`,
    `test_clip_endpoint_falls_back_to_gif_without_ffmpeg`,
    `test_gif_bytes_is_a_play_once_animation`,
    `test_clip_endpoint_list_source_and_404s`
- **LAB-1010** (MUST) `GET /ball_check` serves the page and the hub links to
  it; the page carries a mobile viewport meta.
  - Tests: `test_ball_check.py::test_ball_check_page_served_and_linked_from_hub`
- **LAB-1011** (SHOULD) Frontend: swipe right = ball, left = not a ball (the
  card follows the finger; vertical drags are ignored); buttons for
  unsure/zoom/undo; arrow keys and y/n/u/z on a desktop. The next crops are
  prefetched (stills and the next two clips) and the queue refills below 5
  cards. When a card appears the page first plays its context clip (LAB-1009a,
  a muted inline `<video>`), then on its `ended` event shows the still with the draggable box; the answer buttons work while it
  plays, and Replay / `p` plays it again (stepping or Zoom also stop/restart it). A failed verdict POST puts
  the card back rather than reporting a saved verdict.
  - Tests: UNTESTED-FRONTEND (`web/ball_check.html`; the server contract is
    LAB-1004..1009)
- **LAB-1013a** (MUST) `POST /ball_check/save_box` on a candidate appends to
  the GT sidecar that already exists for that clip under any alias (candidate
  `arsenal_mancity_20250925_seg010` -> `arsenal_mancity_seg010.jsonl` or
  `mancity_seg010.jsonl`), never a second raw-named file. Lookup order: exact
  stem, then the alias spelled like the canonical id, then alphabetical; a clip
  with no GT keeps its own name. The response's `sidecar` names the file.
  Identity rules: `footy_track.clip_id` and
  [feature store §3.1a](../../../docs/design/feature_store.md#31a-clip-identity-canonical-clip-id).
  - Tests: `test_ball_check.py::test_save_box_on_a_raw_named_candidate_writes_into_the_existing_gt_stem`
- **LAB-1013b** (MAY) `GET /ball_check/queue?source=list` serves the explicit
  cards of the review list named by `FOOTY_BALL_CHECK_LIST` (`{"items": [{clip,
  frame_index, bbox, other_bbox?, label?, note?, ...}]}`, normalised boxes), in
  file order, skipping cards that have a verdict. A card's `box_index` is its
  position in the file and its source is `list`; `/ball_check/crop` and
  `crop_meta` with `src=list` resolve the box from the file and draw
  `other_bbox` in magenta. The page forwards `?source=` and `?clip=` to the
  queue (`/ball_check?source=list`). No file configured means an empty pool.
  List verdicts go to the normal log and are excluded from `/ball_check/stats`
  and the confidence-band estimate (hand-picked cards are not a sample).
  Used by the Claude-verify re-check (`docs/design/claude_autolabel_loop.md`
  section 8); a minimal piece of T-47, not the `claude` candidate pool.
  - Tests: `test_ball_check.py::test_list_queue_file_order_skips_judged`,
    `test_list_crop_and_meta`
- **LAB-1014** (PLANNED, MUST) `/ball_check` can serve cards from the `claude`
  candidate pool, uncertain first (Claude and the detector disagree, low
  confidence, cell corrections). A "yes" on such a proposal writes the box to
  the GT sidecar as `[label, claude, human_checked]` (tier 2). This is the one
  case where Ball Check writes a box for a yes; a plain `ball` verdict on a
  detector box still writes only to the verdict log (LAB-1004). A "no" is a
  `not_ball` verdict as today. Nothing is written until the person acts.
  - Tests: none yet
- **LAB-1015** (PLANNED, MUST) Tap-to-place: on a card where the ball is visible
  but no box is on it (a Claude "cell" answer, or a miss), tapping the ball
  writes a new box as `[label, labeller]` (tier 1), the same result as a drag.
  The Claude cell is a hint shown on the card, never a box.
  - Tests: none yet

## 12. Confidence sampling — candidates (LAB-11xx)

`candidates.py`. The GT sidecars cannot drive confidence-aware review (they
never persist confidence — LAB-104), so cards are drawn from the detector's
own output: one JSONL per clip under `server._CANDIDATES_DIR`, ball rows
only, pulled from `s3://…/footy_data/machine_labels` (run
`rtdetr-l_1920_v11`, emitted at `conf_threshold` 0.1 — borderline detections
are already there, nothing needs re-running). Corpus on the Mac mini:
356,812 ball candidates over 162 clips.

- **LAB-1101** (MUST) `read_candidates` keeps only ball-class rows and
  numbers them per frame in file order (`cand_index`), giving the identity
  `(clip, frame_index, cand_index)` in a namespace separate from sidecar
  boxes. Malformed lines, missing bbox and missing frame_index are skipped.
  - Tests: `test_candidates.py::test_read_candidates_keeps_ball_rows_and_numbers_them`,
    `test_candidates.py::test_read_candidates_skips_malformed_lines`,
    `test_candidates.py::test_read_candidate_box_round_trips`
- **LAB-1102** (MUST) Sampling weight is a Gaussian bump on the threshold
  plus a uniform floor (`PEAK_GAIN` 5.0, `focus` 0.7, `sigma` 0.10 → peak
  ~13x the tail). Order is a weighted sample without replacement via the
  exponential race, with the uniform drawn from a hash of the candidate's
  identity so the queue does not reshuffle under a labeller mid-session.
  Uniform sampling would spend nearly all effort where the model is already
  certain; a pure-boundary sampler would leave every other band unmeasured.
  - Tests: `test_candidates.py::test_sampling_weight_peaks_at_tau_and_never_reaches_zero`,
    `test_candidates.py::test_stratified_order_concentrates_near_tau_but_covers_the_range`,
    `test_candidates.py::test_stratified_order_is_deterministic`,
    `test_candidates.py::test_stratified_order_is_a_permutation`
- **LAB-1103** (MUST) Adaptive mode (`tau` < 0, the default): per-band
  precision is recomputed from the verdicts so far, the boundary is estimated
  where precision crosses 0.5, and bands with little evidence are up-weighted
  (`1/sqrt(1+decided)`). Falls back to `DEFAULT_TAU` until at least two bands
  have 3+ decided verdicts — a boundary from two labels is worse than a prior.
  `unsure` carries no information and is excluded; `box_off` counts as found.
  - Tests: `test_candidates.py::test_band_stats_counts_decided_and_ignores_unsure`,
    `test_candidates.py::test_estimate_tau_finds_the_measured_crossing`,
    `test_candidates.py::test_estimate_tau_falls_back_until_there_is_evidence`,
    `test_candidates.py::test_adaptive_order_follows_the_measured_boundary`,
    `test_candidates.py::test_adaptive_weight_boosts_bands_with_little_evidence`,
    `test_candidates.py::test_box_off_counts_as_found_for_the_boundary_estimate`
- **LAB-1104** (MUST) `BinnedPool` buckets candidates by confidence
  (`SAMPLE_BINS` 100) with each bucket pre-sorted by its draw, so a page is
  selected in `bins x page` instead of racing 357k rows (measured: 1.45 s →
  0.018 s). Weights are quantised to the bin centre in *both* paths, so the
  fast path is exactly equivalent to the full race, not merely close.
  - Tests: `test_candidates.py::test_binned_pool_matches_a_full_race`,
    `test_candidates.py::test_binned_pool_skips_judged_and_still_fills_the_page`
- **LAB-1105** (MUST) `GET /ball_check/queue?source=candidates` serves
  candidate cards (`source: "cand"`, with `confidence`) and returns the
  current `tau`, per-band `bands` and the pool `histogram`; `source=auto`
  (default) prefers candidates when any exist. Verdict identity is namespaced
  by source, and records written before candidates existed are read as
  `sidecar`.
  - Tests: `test_ball_check.py::test_queue_candidates_source_and_namespaced_verdicts`

## 13. Operational notes (informative)

- Single global session; no auth; intended for one local user.
- Downstream, sidecars + Roboflow datasets are ingested into the DuckDB
  feature store (`feature_store/ingest_gt.py`) and exported as leakage-free
  YOLO training datasets (`scripts/export_training_dataset.py`).
- All timestamps follow the ContinuousTime / GameTime conventions
  (`docs/timings.md`).
- The torch.compile (Inductor) cache is pinned to
  `~/.cache/footy_torch_inductor` before torch import (macOS purges $TMPDIR).

## 14. Known gaps — OPEN items

Tracked against this spec; burn these down by fixing the behavior AND
updating/adding the corresponding LAB requirement + tests.

- ~~**OPEN-1**~~ FIXED (LAB-1433): `GET /ingest` served a `web/ingest.html` that never existed
  (500). The page route and its hub card are removed (`GET /ingest` is now a 404); the
  `/ingest/upload` and `/ingest/run` APIs (LAB-501/502) are unchanged.
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

## 16. Pitch calibration — manual camera -> pitch (LAB-12xx)

Milestone M0 of `docs/design/calibration.md`: an operator pairs points on a
video frame with named pitch keypoints (or free pitch metres), the server fits a
homography and renders the full pitch model over the frame and as a bird's-eye
warp so the fit can be judged by eye. Saved pairs are *calibration GT*, the
yardstick for the automatic calibrator (M1+).

Backend `calibration.py`, page `web/calibration.html` (route `/calibration`,
Konva; overlays are siblings of the stage containers, never children), maths in
`footy_track/calibration/` (see its README for the coordinate conventions).
`H` maps **normalised image coords to pitch metres** (origin centre spot, `+x`
long axis, `+y` far touchline), the form `PitchProjector` consumes.
Calibration GT lives in `<gt dir>/calibration_gt/<canonical clip id>.json`
(`{clip, frames: {"<frame_index>": record}}`, record = `{frame_index,
correspondences: [{image, pitch_id, pitch_xy}], H, H_convention, pitch_dims,
image_size, method, inliers, mean_err_m, err_px, ts}`). It is a separate file
tree: the ball GT sidecars and the feature store are not touched.

- **LAB-1201** (MUST) `GET /calibration` serves the page with `Cache-Control:
  no-store`, and the hub (`/main`) links to it.
  - Tests: `test_calibration.py::test_pages_served`
- **LAB-1202** (MUST) `GET /calibration/api/clips` lists the clips under
  `_CLIPS_DIR` as `{name, n_gt}` (`n_gt` = frames with calibration GT);
  `GET /calibration/api/clips/{clip}/meta` returns `{frames, fps, width,
  height, gt_frames}`; `GET /calibration/api/frame/{clip}/{idx}.jpg` returns the
  frame JPEG. Unknown clip or unreadable frame -> HTTP 404.
  - Tests: `test_calibration.py::test_clips_meta_and_frame`
- **LAB-1203** (MUST) `GET /calibration/api/pitch_model?length=&width=` returns
  the pitch model for those dims (default 105 x 68): sampled polylines for every
  regulation line, circle and penalty arc, the named keypoints (stable ids,
  e.g. `L_PEN_BOX_TOP_OUTER`), spots, and the warp scale/margin.
  - Tests: `test_calibration.py::test_pitch_model_route`,
    `tests/calibration/test_pitch_model.py`
- **LAB-1204** (MUST) `POST /calibration/api/fit` takes `{correspondences:
  [{image: [u, v], pitch_id | pitch_xy}], pitch_dims, method: dlt|ransac,
  image_size?}` and returns `{H, H_flat, inliers, err_m, err_px, mean_err_m,
  overlay, correspondences}`. A `pitch_id` is resolved to metres from the given
  dims. `overlay` is every pitch line projected through `H^-1` into normalised
  image coords as runs of points; points behind the camera are dropped so lines
  never wrap. Fewer than 4 points, an unknown id, bad dims or a configuration that
  cannot determine `H` -> HTTP 400 `{ok: false, error}`. The latter means no 4
  points in general position (no 3 collinear, tested in pitch metres and in
  image space; brute force up to 12 points, rank check above); `error` names the
  collinear points by 1-based index, e.g. "points 1, 2, 4 lie on one pitch line".
  A successful response also carries `warnings: [str]` (only one 4-subset in
  general position although > 4 points were given, or `H` poorly conditioned).
  - Tests: `test_calibration.py::test_fit_returns_h_errors_and_overlay`,
    `::test_fit_accepts_free_points_and_ransac`,
    `::test_fit_rejects_bad_input_with_4xx`,
    `::test_fit_route_400_names_collinear_points`,
    `tests/calibration/test_geometry.py` (`test_fit_rejects_three_collinear_*`,
    `test_fit_accepts_three_collinear_plus_two_others`, ...)
- **LAB-1205** (MUST) `POST /calibration/api/warp` takes `{clip, frame_index,
  H, pitch_dims?, px_per_m?}` and returns a JPEG of the frame warped to a
  top-down raster covering `(L + 2m) x (W + 2m)` metres (`m` =
  `WARP_MARGIN_M`), `+x` right, `+y` up. Bad `H` -> 400; unknown clip/frame -> 404.
  - Tests: `test_calibration.py::test_warp_returns_raster_of_expected_size`,
    `test_geometry.py::test_warp_puts_pitch_features_where_the_model_says`
- **LAB-1206** (MUST) `PUT /calibration/api/gt/{clip}/{idx}` re-fits `H` from
  the posted correspondences (so stored `H` and points cannot disagree) and
  replaces that frame's record in `calibration_gt/<canonical id>.json`, written
  atomically. The file keeps its existing mode, or 0664 when new (not
  mkstemp's 0600). A rejected fit leaves the stored record untouched.
  `GET .../gt/{clip}/{idx}` returns it (404 if absent), `GET .../gt/{clip}`
  summarises all frames, `DELETE .../gt/{clip}/{idx}` removes one.
  - Tests: `test_calibration.py::test_gt_save_load_round_trip`,
    `::test_gt_save_rejects_too_few_points_and_keeps_existing`,
    `::test_gt_file_mode_new_is_664_and_existing_is_preserved`
- **LAB-1207** (MUST) The GT file is keyed by `canonical_clip_id`, so every
  alias of a clip reads and writes the same file; nothing else under the GT dir
  is created or modified.
  - Tests: `test_calibration.py::test_gt_file_location_is_canonical_and_alias_shared`
- **LAB-1208** (MUST) UI: clip picker, frame slider/number input, chips for
  frames of this clip that have GT; frame on the left (click adds an image point,
  points draggable), pitch diagram on the right (click snaps to the nearest named
  keypoint and shows its name; shift+click places a free point); points pair up
  in order with numbered markers on both sides and a list with per-pair error
  and a delete button; editable pitch dims; Save/Load/Delete GT.
  - Tests: UNTESTED-FRONTEND (`web/calibration.html`; server contract LAB-1202..1207)
- **LAB-1209** (MUST) UI: with >= 4 complete pairs the page auto-fits, overlays
  the reprojected pitch lines on the frame (toggle), colours each point by
  reprojection error (green < 3 px, amber < 8 px, red above), and can blend the
  bird's-eye warp over the pitch diagram (toggle, opacity). Points with no error value
  (no fit yet) are neutral blue, never red. `state.fit` is cleared as soon as the
  pair set changes, so a stale fit is never drawn. A failed fit clears the
  overlay and shows the server's error prominently above the pairs list;
  `warnings` from a successful fit are shown in an amber box.
  - Tests: UNTESTED-FRONTEND

### Machine proposals and their review (LAB-1210..1219)

Claude-assisted pre-labelling (`footy_track/calibration/prelabel/`, CLI
`python -m footy_track.scripts.prelabel_calibration`) writes *proposals* to
`<gt dir>/calibration_proposals/<canonical clip id>.json`:
`{clip, runs: {run_id: {source: claude|classical, created, params, frames:
{"<idx>": GT-shaped record + status proposed|accepted|edited|rejected|failed,
score in [0,1], failure, view}}}}`. The review page `web/calibration_review.html`
(`/calibration/review`, mobile-first, mirrors Ball Check) and the routes in
`calibration_proposals.py` turn them into calibration GT. A proposal never
becomes GT without a human accepting it; accepted GT carries
`origin: {source, run_id, proposal_status: accepted|edited}`.

- **LAB-1210** (MUST) `GET /calibration/api/proposals?run=&status=&clip=&sort=&limit=`
  lists proposals across all clips: `status` is comma separated (default
  `proposed`; `all` = every status), `sort` is `score_asc` (worst first, the
  default) or `score_desc`; unknown status or sort -> HTTP 400. Response
  `{items: [{clip, run_id, frame_index, status, score, failure, source,
  n_points, n_inliers, view, verify}], total, runs: [{run_id, source, created,
  params, counts}]}`.
  - Tests: `test_calibration_proposals.py::test_list_filters_and_sorts_by_score`
- **LAB-1211** (MUST) `GET /calibration/api/proposals/{clip}/{run_id}/{idx}` returns
  the full record plus `overlay` (the pitch model projected through `H`, as in
  LAB-1204); unknown run/frame -> 404; clip aliases resolve to one file.
  - Tests: `test_calibration_proposals.py::test_get_one_has_overlay_and_404`
- **LAB-1212** (MUST) `POST .../accept` re-fits `H` from the proposal's inlier points
  (outliers dropped) and writes the record into `calibration_gt` with `origin`
  (`proposal_status` `accepted`), then sets the proposal `accepted`. A body
  `{correspondences, method?, overwrite?}` accepts the operator's points instead and
  yields `edited` when they differ from the proposal's. A frame that already has GT
  -> 409 unless `overwrite` (see LAB-1220); a `failed` proposal can only be accepted with points.
  - Tests: `test_calibration_proposals.py::test_accept_copies_to_gt_with_origin_and_flips_status`,
    `::test_accept_drops_outliers_and_does_not_touch_real_gt_of_other_frames`,
    `::test_accept_with_changed_points_is_edited`, `::test_accept_failed_needs_points_and_bad_points_400`
- **LAB-1213** (MUST) `POST .../reject` sets status `rejected` (optional `reason`),
  writes no GT; unknown proposal -> 404.
  - Tests: `test_calibration_proposals.py::test_reject_flips_status_and_writes_no_gt`
- **LAB-1214** (MUST) `GET /calibration/review` serves the page with `Cache-Control:
  no-store` and a mobile viewport (`width=device-width`); the hub links to it.
  - Tests: `test_calibration_proposals.py::test_pages_served`
- **LAB-1215** (MUST) Review UI: one proposal per screen: the frame at full width
  with the projected pitch overlay (toggle) and numbered points (green inlier, red
  outlier), the score, the Claude view note, a confidence summary (inliers, min/mean
  confidence, fit error, verifier verdict) and the failure reason when failed.
  Queue ordered worst-first by default, filterable by run and status, with a
  progress counter.
  - Tests: UNTESTED-FRONTEND (server contract LAB-1210/1211)
- **LAB-1216** (MUST) Review UI actions are reachable by touch without hover or keys:
  big Accept / Reject / Skip buttons, swipe right = accept, swipe left = reject,
  a Warp toggle (and press-and-hold on the image) for the bird's-eye warp, and an
  "Open in editor" link to `/calibration?clip=..&frame=..&proposal=<run_id>`. Keys
  a/r/j/k/e/o/w are extras. A 409 on accept shows an in-page notice ("already has hand
  GT" / earlier machine GT) with an explicit Replace action or Keep-and-skip (LAB-1220).
  - Tests: UNTESTED-FRONTEND
- **LAB-1217** (MUST) `/calibration?clip=&frame=&proposal=` opens that frame with the
  proposal's inlier points loaded in the editor; Save then accepts it through
  LAB-1212 (`edited` if the points changed) instead of a plain GT save.
  - Tests: UNTESTED-FRONTEND (server contract LAB-1212)
- **LAB-1218** (MUST) The pre-labelling pipeline writes proposals in the contract
  format atomically (file mode 0664), run id `claude-<model>-<prompt_version>-<ts>`
  or `mock-...`, logs token usage per call in the run's `params`, never writes
  `calibration_gt`, and marks frames with no usable fit `failed` with a reason. It skips
  frames that already have human GT unless told not to (`--no-skip-gt`).
  - Tests: `tests/calibration/test_prelabel.py::test_pipeline_with_mock_writes_contract_proposals`,
    `::test_pipeline_logs_usage_and_handles_claude_errors`
- **LAB-1220** (MUST) Accepting a proposal never silently replaces existing calibration GT.
  For a frame whose GT has no `origin` (hand-labelled) the accept route answers 409
  `{ok: false, existing_gt: "human", error: "this frame already has hand-labelled
  calibration GT ..."}` and writes nothing (the proposal stays `proposed`), also when the
  body carries operator points; GT written by an earlier machine proposal answers 409 with
  `existing_gt: "machine"`. Only `overwrite: true` replaces it (the new record carries
  `origin`). The review page shows "this frame already has hand GT" with explicit
  Replace / Keep-and-skip buttons instead of a bare confirm.
  - Tests: `test_calibration_proposals.py::test_accept_never_silently_overwrites_hand_gt`,
    `::test_accept_conflict_with_machine_gt_is_labelled_as_such`, `::test_accept_copies_to_gt_with_origin_and_flips_status`,
    `::test_review_page_has_an_explicit_replace_flow` (page markers only; the flow itself is UNTESTED-FRONTEND)
- **LAB-1221** (MUST) Proposals are fitted in two stages: a coarse homography from the
  identifier's point (and named-line) labels, then line refinement
  (`calibration/refine.py`: all 8 DOF, robust distance-transform cost of the white-line
  mask, coarse-to-fine, labels as a weak prior, rejected when the cost does not improve or
  `H` degenerates). The proposal stores the final `H`, `score`, `dt_residual`, the coarse
  fit under `pre_refine` and the outcome under `refine`; the eval reports both. The
  `sweep` command measures the success rate (mean GT reprojection error <= 5 / <= 10 px)
  before and after refinement per label-noise level with the noisy `MockIdentifier`.
  - Tests: `tests/calibration/test_refine.py` (all), `tests/calibration/test_prelabel.py::test_fit_*`
- **LAB-1222** (MUST) A frame that is not a main-camera pitch view can be marked
  `not_broadcast` with one tap, separately from `rejected`. `POST
  /calibration/api/proposals/{clip}/{run_id}/{idx}/not_broadcast` (proposed or failed items)
  sets the proposal status `not_broadcast` (`prev_status`, `reviewed_at`) and writes the
  frame-level flag `<gt dir>/frame_type/<canonical clip>.json` (atomic, mode kept; also
  `PUT/DELETE /calibration/api/frame_type/{clip}/{idx}`, list in `GET .../gt/{clip}` and
  `.../clips/{clip}/meta`). A frame with calibration GT answers 409 and writes nothing unless
  `overwrite: true` (which deletes that GT); accept and GT-save on a flagged frame are 409 until
  the flag is cleared (DELETE restores the proposals' status). The pre-label run skips flagged
  frames (`--skip-gt`), the feature store drops them from `frame_calibration_best` /
  `detections_on_pitch`. The review page has a "Not broadcast" button (key `n`, also "Clear" in the
  not-broadcast filter), a "to review + failed" and a "not broadcast" filter, errors in the red
  banner, and a "Claude: not main camera" badge when `view.main_camera_wide` is false (never
  auto-marks); the editor has a Not broadcast button, a marked banner and Undo.
  - Tests: `test_calibration_proposals.py::test_not_broadcast_*`,
    `::test_review_page_has_not_broadcast_button_hint_and_filter` (markers; flow UNTESTED-FRONTEND),
    `tests/calibration/test_prelabel.py::test_pipeline_skips_not_broadcast_frames`,
    `tests/feature_store/test_calibration_store.py::test_not_broadcast_frame_has_no_calibration_and_no_projection`,
    `tests/feature_store/test_ingest_calibration.py::test_frame_type_flags_and_not_broadcast_proposals_are_ingested`
- **LAB-1223** (MUST) A frame that shows too few identifiable pitch features to place >= 4 point
  pairs (e.g. only the centre circle / halfway line) can be marked `too_few_points` with one tap:
  skipped by the human, kept as a candidate for the PTZ rig solver (fixed camera centre, solvable
  from a few lines). Stored in the LAB-1222 sidecar `<gt dir>/frame_type/<canonical clip>.json` as
  `{"type": "too_few_points", "ts", "run_id"?, "tags": [...]}` (one type per frame; atomic write, mode
  kept). Optional `tags` say what *is* visible, from `centre_circle, halfway_line, penalty_box,
  goal_area, touchline` (unknown -> 400). `PUT /calibration/api/too_few_points/{clip}/{idx}` (body
  `{tags?, run_id?}`; PUT again replaces the tags), `DELETE` the same path (404 when not marked),
  `GET .../too_few_points/{clip}` -> `{clip, frames: [{frame_index, tags, ts, run_id?}]}` and `GET
  .../too_few_points` -> `{total, clips: {<canonical clip>: [...]}}` for the PTZ proposer (also
  `too_few_points: {"<idx>": [tags]}` in `GET .../gt/{clip}` and `.../clips/{clip}/meta`, and
  `n_too_few_points` in `GET .../clips`). `POST /calibration/api/proposals/{clip}/{run_id}/{idx}/too_few_points`
  (body `{tags?}`; proposed or failed items) sets the proposal status `too_few_points` keeping the record
  (`prev_status`, `reviewed_at`, `tags`), other still-open proposals of the frame follow, `?status=too_few_points`
  filters it; DELETE restores the previous statuses. A frame with calibration GT or a `not_broadcast`
  flag is 409 and nothing is written; saving GT or accepting a proposal on a marked frame clears the
  mark. `not_broadcast` replaces a too-few mark and never clears one. The feature store ingests the
  flag and the proposal status as `review_status=too_few_points` (no H; never wins
  `frame_calibration_best`). The pre-label run skips these frames like not_broadcast ones (`--skip-gt`). The
  marks live in the current dataset's marks dir (`?dataset=`), like not_broadcast. Editor: "Too few points"
  button (key `t`, toggles; "Clear too few points" when on) next to Not broadcast, with a banner, one-tap
  tag chips and Undo; a clip with only not-broadcast / too-few marks counts as "skipped". Review page:
  big "Too few points" button (key `t`, "Clear" on marked items) beside Not broadcast/Skip, optional
  tag chips, and a "too few points" status filter.
  - Tests: `test_calibration_proposals.py::test_too_few_points_*`,
    `::test_review_and_editor_pages_have_too_few_points_button_and_filter` (markers; flow UNTESTED-FRONTEND),
    `test_calibration.py::test_too_few_points_editor_endpoint_without_proposals`,
    `test_datasets.py::test_too_few_points_marks_are_per_dataset`,
    `tests/calibration/test_prelabel.py::test_pipeline_skips_too_few_points_frames`,
    `tests/feature_store/test_ingest_calibration.py::test_too_few_points_status_and_flags_are_ingested_without_winning_best`
- **LAB-1224** (SHOULD) Two finders place pitch points in one tap: `POST /calibration/api/auto_points` (body
  `{clip, frame_index, kind: "centre" | "d", distortion?, pitch_dims?}`; `kind` is required, else 400) reads that
  frame (`_find_video` + `_read_frame` in a worker thread, current `?dataset=`) and runs
  `calibration/pitch_finders.py`. A hit is `{ok: true, kind, confidence, correspondences: [{image: [u, v],
  pitch_id, pitch_xy, confidence, derived}]}`; image points are observed coordinates, consistent with the given
  `distortion`; `pitch_xy` follows `pitch_dims` (pitch length x width vary per ground and are never assumed);
  `derived: true` marks points synthesised from fitted geometry rather than seen, and `pitch_id` is `null` for
  the derived circle points that have no keypoint name (free points). An abstention is `{ok: false, reason}`
  with HTTP 200, not an error. Unknown clip / unreadable frame -> 404, bad body / kind / distortion -> 400.
  Nothing is stored. Answers (and abstentions) are cached in a 64-entry LRU keyed by dataset, video, frame,
  distortion, kind and pitch dims; one detection runs at a time.
  - `centre`: shape-first (`calibration/centre_fit.py`). Halfway line + centre circle give `CENTRE_CIRCLE_TOP/BOTTOM`
    and the circle's ellipse `E`; the pole of the line w.r.t. `E` is the vanishing point `V` of the pitch length
    axis. That leaves ONE free parameter, the vanishing point `W` of the width axis on the halfway line, and then
    vanishing line `V x W` -> circle centre (its pole) -> `(+-9.15, 0)` (line centre-`V` x `E`) -> `H`. `W` comes from
    (a) a touchline end (the long line through `V`, junction = `HALFWAY_TOP/BOTTOM` at `(0, +-width/2)` from the
    request's `pitch_dims`; 3 known points on the line fix a 1-D projective map), (b) the ground's fixed camera
    centre when the clip's ground has a rig (`camera_rigs/`, chosen from the clip-name prefix: Emirates for
    arsenal_mancity / astonvilla / nottinghamforest, Vitality for arsenal_bournemouth; unknown grounds: none) -
    pan / tilt / focal length from the 2 points + the circle outline, or (c) square pixels with the principal point
    at the image centre: the circular points `E` x `V W` must lie on `diag(1/f^2, 1/f^2, 1)`, a 1-D root search that
    is abstained on when the answer moves more than 1.2% of the width under sub-pixel ellipse jitter (a symmetric
    frontal view cannot give `f`). Candidates are scored on the whole template against the line mask (halfway
    line + circle = core; touchlines and box lines count as extra matched length; the far touchline strip is part
    of the mask here) and chamfer-refined; the best one is accepted if core precision >= 0.6. Returned points are
    all `H` applied to pitch coordinates: `CENTRE_CIRCLE_TOP/BOTTOM/LEFT/RIGHT` (`derived: false` where the mask
    shows the circle), `CENTRE_SPOT` (derived), `HALFWAY_TOP/BOTTOM` when in frame, four free points on the circle at
    45 degrees and, if needed, more free circle / touchline points until >= 4 are in general position, so the editor
    fits the same `H` at once. The response also carries `H` (pitch metres -> undistorted normalised image;
    `H_convention`) and `fit` (`accepted` source, `precision`, `f_over_w`, `ground`). The square-pixel source alone is approximate on broadcast feeds (GT set: median 1.9 m editor-fit error, 5% within
    0.5 m): it only answers when no touchline end / ground rig does, with confidence x0.6 and a `warning` string the
    page appends to its status line. When no candidate is accepted the answer is the 2 circle x line points only. `use=` / `shape=False` select sources for evaluation.
  - `d`: the whole penalty-area template (goal line, 18-yard box, 6-yard box, D; sizes fixed by the Laws,
    positioned with the request's length) is fitted by brute force over assignments of detected lines to
    template lines (and "D + front line + one line"), scored against the line mask, refined by least squares and
    accepted only if clearly better than any different fit; returns the in-frame template keypoints
    (`{L,R}_PEN_BOX_*_OUTER/_GOAL`, `_GOAL_AREA_*`, `_PEN_ARC_TOP/BOTTOM/MID`, `_PEN_SPOT`). When the template
    fit abstains it falls back to the 2 box-front x D points. Both finders assume the camera is on the -y side
    (`+x` to the image right, far touchline `+y`) like every GT frame, and abstain rather than guess.
  Editor: two buttons, "Centre" (key `c`) and "D" (key `d`, 36 px tall for phones), append the returned pairs to
  the pair list exactly like hand-placed ones (`snapshot()` first, so Undo removes them; then `refit()`; pairs
  whose `pitch_id` is already placed - or free points whose metres are - are skipped; refused while an unpaired
  point is pending), using the frame's current `state.distortion` and pitch dims. The status line shows kind,
  confidence, how many points are derived, or the abstain reason, and with fewer than 4 pairs says
  "2 points placed - add 2 more (e.g. touchline / halfway line ends) to fit". Added points are ordinary
  draggable pairs. Offline evaluation: `python -m footy_track.scripts.eval_finders`.
  - Tests: `test_calibration.py::test_auto_points_*`, `test_datasets.py::test_auto_points_follow_the_dataset`,
    `test_calibration.py::test_editor_page_has_the_two_auto_buttons` (markers),
    `test_calibration_editor_js.py::test_auto_circle_helpers_under_node` (pair merge + status text; button flow
    UNTESTED-FRONTEND), `tests/calibration/test_pitch_finders.py`, `tests/calibration/test_centre_fit.py`,
    `tests/calibration/test_centre_circle.py`
- **LAB-1219** (SHOULD) Frames are read through `footy_track.calibration.video.read_frame`
  (cv2, falling back to the ffmpeg CLI when cv2 fails or returns an all-black frame,
  frame-accurate, reader cached per video); `/calibration/api/frame` and `/warp` use it.
  - Tests: `tests/calibration/test_video.py`

### Lens: suggested distortion and "Snap to lines" (LAB-1225..1226)

- **LAB-1225** (SHOULD) Broadcast lenses are barrel distorted, so long lines curve near the frame edges and a
  plain homography cannot follow them. The editor's lens is **off by default** and nothing applies one behind the
  operator's back: records with no lens (GT, proposals) load, render and are accepted exactly as stored (stored `H`,
  straight lines, no refit). The ground's division `k1` (`calibration/camera_rigs/*.json`, clip -> rig by
  `centre_fit.rig_for_clip`; Emirates 0.0147, Vitality 0.035; unknown grounds none) is only a *suggestion*, reported as
  `default_distortion` by `GET /calibration/api/clips/{clip}/meta` and on proposals: it is the value the lens box
  jumps to when ticked on, Snap's starting k1 while the lens is off, and the review page's preview toggle. The editor
  header has the lens box (on/off, k1, `-`/`+`, source); `,` / `.` nudge k1 by 0.002 (Shift: 0.01) with a live
  refit, and the lens is part of the undo snapshot. When a lens is active (stored with a record, turned on, or set by
  Snap) the finders (`/auto_points`), `/fit`, `/warp` and Save all use it, so what is drawn is what is saved; Save
  records it as `distortion`. On the review page a per-card "Lens" button (key `l`) re-renders the bird's-eye warp
  through the suggested k1 as a preview only: the stored H, overlay and Accept are untouched.
  - Tests: `tests/labeller/test_calibration_lens.py`, `tests/calibration/test_lens.py::test_default_distortion_*`,
    `test_calibration_editor_js.py::test_lens_helpers_under_node` (buttons/keys UNTESTED-FRONTEND)
- **LAB-1226** (SHOULD) `POST /calibration/api/snap` (editor button "Snap to lines", key `s`, optional "k1 only")
  takes the `/fit` body plus `{clip, frame_index, mode: "full"|"k1"}` and refines `H` and `k1` of the current fit
  against the frame's line mask (`calibration/lens.py`; with the lens off it also searches from the ground's suggested k1 and keeps the better accepted answer, both judged (and falling back) against the operator's current fit; k1-only mode keeps H and the request's lens: robust soft-L1 least squares over the 8 homography
  parameters + k1 from the current solution with bounded steps, the clicked points as soft constraints; evidence =
  `centre_circle.line_mask_for` + the far-edge strip, scored with `pitch_finders.Field/Template/evaluate`). It is
  accepted only if the line F1 improves and no clicked point moves more than ~4 px (of 1280 px); then the clicked
  points are moved onto the snapped model (so the server refit from them reproduces it; Undo restores them and
  the old lens). Otherwise the answer carries `accepted: false` and a `reason` and the input fit. Always returns
  `f1_before/after`, `k1_before/after`, `max_click_shift_px` and the touchline/goal-line `edge_before/after`
  residual. Nothing is stored. Offline evaluation: `python -m footy_track.scripts.calib_lens_eval`.
  - Tests: `tests/calibration/test_lens.py`, `tests/labeller/test_calibration_lens.py::test_snap_route_*`

## 17. Labelling site (informative)

**One site, always main.** Open `http://mac-mini-i7:8011/` on the tailnet. It serves
every dataset (calib, Nottingham Forest held-out, ...) from one process: pick one with the
dataset picker on the home page, or add `?dataset=nottinghamforest` to any page URL (section 18).
The home page is a **task board**: a "To do now" list
(from `GET /api/tasks`, each with title, one-line why, link, device hint and an
optional live remaining count), then the tool cards, then a "Servers" panel (name,
port, git sha, branch, uptime from each service's `GET /api/server_info`).

**Posting a task.** The board is one shared JSON file, `$FOOTY_TASKS_FILE` (default
`/mnt/storage/footy_data/labelling_tasks.json`), re-read on every request, so no
redeploy is needed. It is created from a built-in seed the first time it is read.
Edit it only through the CLI (atomic write, stable ids):

```bash
python -m footy_track.scripts.labelling_tasks add "Recheck X" --url http://mac-mini-i7:8011/ball_check \
    --why "one line" --device phone --priority 1 \
    --count-url 'http://mac-mini-i7:8011/ball_check/queue?source=list&limit=1' --count-label left
python -m footy_track.scripts.labelling_tasks list
python -m footy_track.scripts.labelling_tasks done <id>      # remove it by hand
```

`--device` is `phone|laptop|either`; lower `--priority` sorts first. A count URL
returns JSON with `total`, `count` or `remaining` (optional `count_path` in the file
for a dotted path). A malformed file never breaks the page: `/api/tasks` answers 200
with `error` set and the CLI refuses to overwrite it.

**Tasks complete themselves.** A task may carry `done_when`; `/api/tasks` evaluates it
server-side (`labeller/task_done.py`, results cached 5 s) and returns `done: bool` plus
`progress` where it has one. A task with no `done_when` (and nothing inferable, below)
behaves as before: it stays until `done <id>`. Shapes:

| `done_when` | done when | `progress` |
|---|---|---|
| `{"gt": {"clip", "frame"[, "since"]}}` | calibration GT exists for that frame | - |
| `{"proposal": {"clip", "run", "frame"}, "status_in": [...]}` | that proposal's status is in `status_in` (default `accepted`, `edited`) | - |
| `{"frames": [[clip, frame], ...][, "since"]}` | every frame has GT, a `not_broadcast` mark or a `too_few_points` mark (that mark is not written on main yet; it counts the day it lands) | `{done: k, total: n}` |
| `{"count_zero": true}` | the task's `count_url` returns 0 (same `total`/`count`/`remaining`/`count_path` parsing as the page chip; own-host URLs are fetched through the request's base URL, 2 s timeout; an unreachable URL is never "done") | `{remaining: n}` |

Clips go through `canonical_clip_id`, the same as the calibration routes, so
`arsenal_bournemouth_1st_seg030` and `bournemouth_broadcast_seg030` are one clip. `since` is
`"added"` or `YYYY-MM-DD` and ignores GT whose `ts` is earlier (for "recheck" tasks whose frame
already had GT). Any shape may add `"dataset": NAME` (default: the default dataset, which a
link without `?dataset=` opens). A malformed spec or a failing evaluation means "not done",
never an error.

*Inference at read time* (the JSON is never rewritten; the effective spec is returned as
`done_when` with `done_when_inferred: true`), when a task has no explicit `done_when`:
a `count_url` -> `{"count_zero": true}`; otherwise a `/calibration?clip=X&frame=N...` url
-> `{"gt": {clip: X, frame: N, since: "added"}}` (and `dataset` from the url). Job tasks
(`job: true`) already leave the board when their job closes and are not inferred.

*Board behaviour.* Active tasks first (priority order), then done ones. The first time a
task is seen done its timestamp (`done_at`) is stored in the sidecar
`<tasks file>.done.json`; done tasks older than 2 days are left out of the response (never
deleted from the tasks file). If the condition turns false again (GT deleted) the task is
active again. The home page shows done tasks greyed with a tick in a collapsed "Done (n)"
section at the bottom, and `k/n` on tasks with progress. CLI: `add ... --done-when-gt
CLIP:FRAME`, `--done-when-frames CLIP:FRAME,CLIP:FRAME`, `--done-when-count-zero` (needs
`--count-url`); `done <id>` still removes a task manually.

**Deploying.** Live servers run **only** from `/home/george/footy-track-main`, on
main. After a merge to main:

```bash
scripts/labeller_deploy.sh status          # tmux session, PID, health, served sha vs origin/main
scripts/labeller_deploy.sh deploy          # ff-only to origin/main (refuses if dirty), restart, wait for sha
scripts/labeller_deploy.sh restart calib   # one service;  logs <svc> [-f]
```

There is exactly one service, `calib` on :8011. The first restart/deploy creates the
dataset registry (section 18) from a built-in seed if it is missing and never overwrites
it. The old `nottinghamforest` server on :8012 is gone: its data is the `nottinghamforest` dataset of the
one server (stop its tmux session, `tmux -L footy kill-session -t footy-norwich`).

`--dry-run` prints the mutating commands; `--force` lets `restart` kill a non-tmux
process still holding the port (otherwise it prints the PID and stops); `--no-warm`
skips the background Ball Check cache warm-up (`footy_track.scripts.warm_ball_check`)
that runs after a restart, for the `nottinghamforest` dataset of `calib` (`WARM_DATASETS`). Services are tmux sessions `footy-<svc>` on
`tmux -L footy`, logging to `~/.cache/footy-labeller/<svc>.log`. The service table
(ports; data dirs now live in the registry, not in env vars) is at the top of
`scripts/labeller_deploy.sh`; keep the `SERVICES` list in `web/main.html` in step with it. Feature branches are tested in
their own worktree on another port, never in the serving checkout.

## 18. Datasets: one server, many bodies of clips (LAB-13xx)

A *dataset* is a clips dir plus where its labels live. One server serves all of them;
`footy_track/labeller/datasets.py` is the registry and the per-request resolver.

- **LAB-1300** (MUST) Datasets come from a JSON registry, `$FOOTY_LABELLER_DATASETS`
  (default `/mnt/storage/footy_data/labeller_datasets.json`), re-read when its mtime/size
  changes (no restart to add a dataset):
  ```json
  {"default": "calib",
   "datasets": {
     "calib":   {"label": "...", "clips_dir": "...", "marks_dir": "...",
                 "candidates_dir": "...", "ball_check_list": "...",
                 "extra_clips_dirs": ["..."]},
     "nottinghamforest": {"label": "...", "clips_dir": "...", "marks_dir": "..."}}}
  ```
  `clips_dir` and `marks_dir` are required; the rest are optional (`candidates_dir`
  defaults to `<marks_dir>/ball_candidates`; `ball_check_list` unset = empty list pool;
  `extra_clips_dirs` are extra places the calibration page looks for a clip's video).
  `python -m footy_track.scripts.labeller_datasets init|list` creates (never overwrites) and
  shows the registry. With no registry file the single dataset `calib` is built from the
  legacy env vars (`FOOTY_CLIPS_DIR`, `FOOTY_GT_MARKS_DIR`, `FOOTY_BALL_CANDIDATES_DIR`,
  `FOOTY_BALL_CHECK_LIST`, `FOOTY_CALIB_EXTRA_CLIPS_DIRS`) exactly as before; when set, those
  env vars also override the *default* dataset's fields (`--clips-dir` / `--gt-dir` of
  `python -m footy_track.labeller` rely on it). A malformed registry never takes the server
  down: it falls back to the legacy `calib` and `GET /api/datasets` reports `error`.
  - Tests: `test_datasets.py::test_registry_lists_datasets_and_default`,
    `::test_missing_registry_falls_back_to_legacy_single_dataset`,
    `::test_malformed_registry_does_not_take_the_server_down`,
    `::test_env_overrides_apply_to_the_default_dataset_only`,
    `::test_registry_edits_are_picked_up_without_restart`
- **LAB-1301** (MUST) Every route (HTTP and the `/ws` WebSocket) resolves its dataset from
  the `?dataset=<name>` query parameter; none means the registry default (`calib`), so
  existing links and bookmarks are unchanged. An unknown name is a 404 (WebSocket close
  4404), never a silent fall back to the default: a write must not land in the wrong dataset.
  Pure ASGI middleware sets a `ContextVar` that asyncio tasks and `to_thread` workers inherit.
  - Tests: `::test_unknown_dataset_is_404_not_silent_default`,
    `::test_websocket_unknown_dataset_is_refused`, `::test_clips_follow_the_dataset`
- **LAB-1302** (MUST) No state bleeds between datasets. Each non-default dataset has its
  own `Session` (the default one stays the module global `SESSION`); a `Session` is bound
  to its dataset by name because its debounced flush runs on a timer thread with no request
  context. Verdict logs, calibration GT/proposals/frame types, the review-list pool,
  candidate pools, identity labels/tracklets and candidate embeddings are all per dataset.
  - Tests: `::test_labels_saved_in_one_dataset_never_appear_in_the_other`,
    `::test_sessions_do_not_share_state_and_default_keeps_working`,
    `::test_session_flush_thread_writes_to_its_own_dataset`,
    `::test_ball_check_verdicts_and_stats_are_per_dataset`,
    `::test_ball_check_list_pool_is_per_dataset`, `::test_calibration_gt_is_per_dataset`,
    `::test_identity_dirs_do_not_bleed_between_datasets`
- **LAB-1303** (MUST) In-memory caches are keyed by dataset name (crop LRUs in `ball_check`
  and `review`, the clip-size cache; candidate pools by candidates dir), so the same clip
  stem in two datasets never serves one dataset's pixels for the other.
  - Tests: `::test_crop_cache_is_keyed_by_dataset`
- **LAB-1304** (MUST) `GET /api/datasets` lists `{default, current, registry, datasets[]}`.
  - Tests: `::test_registry_lists_datasets_and_default`
- **LAB-1305** (SHOULD) Pages carry the dataset: every page loads `/static/dataset.js`, which
  adds the page's own `?dataset=` to same-origin `fetch`, `WebSocket`, `<img>`/`<video>`
  `src`, `<a href>` and `history.*State` URLs (use `footyUrl()` for URLs built into
  `innerHTML`). The home page has a dataset picker. Without `?dataset=` it does nothing.
  - UNTESTED-FRONTEND (server contract: `::test_pages_carry_the_dataset_shim`)

## 19. Labelling jobs: ask main for labels (LAB-14xx)

### Submitting labelling jobs from another branch

Do not start your own labeller or hand-edit the task board. POST a **job** to main's labeller
(`http://mac-mini-i7:8011`, override with `$FOOTY_LABELLER_URL`); it shows up on the home page
task board with a deep link into the right page and dataset, and you fetch the labels later by
job id or title.

```bash
# 1. submit (re-running with the same "key" returns the same job, never a duplicate)
curl -s -X POST http://mac-mini-i7:8011/api/jobs -H 'Content-Type: application/json' -d '{
  "title": "Nottingham Forest windows r1",
  "kind": "ball_place", "dataset": "nottinghamforest",
  "items": [{"clip": "arsenal_nottinghamforest_seg003", "frame_start": 10, "frame_end": 259, "step": 10}],
  "why": "held-out ball placement", "device": "laptop", "priority": 2,
  "requester": {"branch": "exp/my-branch", "commit": "abc1234", "host": "desktop"},
  "key": "exp/my-branch/nottinghamforest-windows-r1"}'
# -> {"id":"j-56098a576c24","url":"http://mac-mini-i7:8011/labeller?dataset=nottinghamforest&clip=...&frame=10","status":"open","created":true,...}

# 2. progress (computed from the real label files) -- by id or exact title
curl -s 'http://mac-mini-i7:8011/api/jobs/Nottingham%20Forest%20windows%20r1' | jq '.status, .progress'

# 3. the labels, once done (or any time: partial results)
curl -s 'http://mac-mini-i7:8011/api/jobs/j-56098a576c24/results' > labels.json
```

From Python / a Prefect flow, with no dependencies (`urllib` only; copy the single file, or fetch
it from the server if your branch has not merged main: `curl -O http://mac-mini-i7:8011/api/jobs_client.py`):

```python
from jobs_client import submit, wait, results   # or footy_track.labeller.jobs_client
job = submit("Nottingham Forest windows r1", "ball_place", items, dataset="nottinghamforest", key="exp/my-branch/nottinghamforest-windows-r1")
wait(job["id"], poll=60)                  # blocks until the job is done or cancelled
labels = results(job["id"])["labels"]
```

CLI (fills `requester.branch` / `commit` from git, `host` from the hostname):

```bash
python -m footy_track.scripts.labelling_jobs submit --title "Nottingham Forest windows r1" --kind ball_place \
    --dataset nottinghamforest --items /mnt/storage/footy_data/nottinghamforest_label_queue/windows.csv --step 10 \
    --key exp/my-branch/nottinghamforest-windows-r1
python -m footy_track.scripts.labelling_jobs status  "Nottingham Forest windows r1"
python -m footy_track.scripts.labelling_jobs wait    "Nottingham Forest windows r1" --poll 60
python -m footy_track.scripts.labelling_jobs results "Nottingham Forest windows r1" -o labels.json
```

**Worked example: the Nottingham Forest windows.** `/mnt/storage/footy_data/nottinghamforest_label_queue/windows.csv`
has header `clip,start_frame,end_frame,stratum,reason` and 24 windows of 250 frames
(`start_frame..end_frame`, both ends inclusive). The CLI above turns each row into one window
item (`start_frame`/`end_frame` are accepted aliases of `frame_start`/`frame_end`; `stratum` and
`reason` ride along and come back in `items`). With `--step 10` that is 24 x 25 = 600 frames to
label; the home page shows "600 left" and counts down as `ball_gt_marks_nottinghamforest/*.jsonl` fills in.
The link opens the labeller on the next unlabelled frame (`/labeller?dataset=nottinghamforest&clip=..&frame=..`).

**Request body** (`POST /api/jobs`, JSON; `title` is the only required field)

| field | meaning |
|---|---|
| `title` | what the person sees on the board; also an id-like handle (`GET /api/jobs/<title>` when unique) |
| `kind` | `ball_place`, `ball_check`, `calibration`, `frame_type`, `identity`, `other` (default `other`) |
| `dataset` | registry name (section 18); default = the request's `?dataset=`, else the default dataset |
| `items` | list of objects, per kind (below) |
| `why`, `device`, `priority` | board text, `phone`/`laptop`/`either`, 1 = first |
| `requester` | `{branch, commit, host, session}`, all optional strings |
| `key` | idempotency key: the job id is derived from it; re-posting returns the existing job (HTTP 200, `created: false`) |
| `url` | optional override of the deep link (a path on this server) |

| kind | item | a unit counts as labelled when |
|---|---|---|
| `ball_place` | `{clip, frame_start, frame_end, step?}` (a window) or `{clip, frame}` | the clip's GT sidecar (`<marks>/<clip>.jsonl`, alias stems resolved) has a ball box tagged `labeller` or `human_checked`, or a `no_ball`/`not_broadcast` marker on that frame. Machine-only boxes are not labels (README section 1 tiers) |
| `calibration` | same two shapes | `calibration_gt/<clip>.json` has the frame, or `frame_type/<clip>.json` flags it `not_broadcast` |
| `frame_type` | same two shapes | `frame_type/<clip>.json` has an entry for the frame |
| `ball_check` | `{clip, frame_index, box_index, source?}` (`sidecar` default, `cand`, `list`) | `ball_checks/<clip>.jsonl` has a live verdict (last line wins; a `null` verdict undoes) |
| `identity` | `{clip}` | `identity_pairs.jsonl` / `identity_reviews.jsonl` of the dataset mention the clip |
| `other` | anything | never: no computed progress, close it with `POST /api/jobs/{id}/done` |

Windows expand to `range(frame_start, frame_end + 1, step)`; a job is limited to 200,000 units. A `ball_check` job's
link opens Ball Check on that source (and clip, when all cards are in one clip); the page's queue is not
restricted to the job's cards (progress and results are).

**Endpoints**

- `POST /api/jobs` -> `{id, url, status, created, title, dataset}` (201 new, 200 existing key, 422 on a bad body).
- `GET /api/jobs?status=&title=&kind=&dataset=&requester.branch=` -> `{"jobs": [...]}` (`title` matches exactly or as a
  case-insensitive substring; progress included for open/in_progress jobs, all with `progress=true`).
- `GET /api/jobs/{id-or-title}` -> the job (`id, title, kind, dataset, status, items, requester, url, created, updated,
  closed?, progress`). `progress` = `{supported, done, total, remaining, fraction, next}` where `next` is the first
  unlabelled unit. A title must be unique among non-cancelled jobs (else 409 with `ids`).
- `GET /api/jobs/{id-or-title}/frames` (`ball_place` jobs) -> `{id, title, total, done, frames: [{clip, frame, item,
  done, stratum?, reason?}]}`: the items expanded in job order (`frame_start..frame_end` by `step`), `done` by the same
  rule as `progress`. Feeds the labeller's job mode (`/labeller?...&job=<id>`, which a `ball_place` job's `url` carries).
- `GET /api/jobs/{id-or-title}/results` -> the canonical results document below.
- `POST /api/jobs/{id-or-title}/done | cancel | reopen`. Statuses: `open`, `in_progress`, `done`, `cancelled`.
  The server moves `open -> in_progress` at the first label and `-> done` when progress reaches 100%
  (never touches a cancelled or manually closed job).
- Open jobs are merged into `GET /api/tasks` (`job: true`, `kind`, `dataset`, a `count_url` for the live "left" chip).
- Storage: one JSON file per job in `$FOOTY_LABELLING_JOBS_DIR` (default `/mnt/storage/footy_data/labelling_jobs`,
  created on the first submit), atomic writes. The label files are only read, never written.

**Results schema `labelling-job-results/v1`** (`GET /api/jobs/{ref}/results`)

```json
{"schema": "labelling-job-results/v1", "generated_at": "2026-11-16T12:00:00Z",
 "job": {"id": "...", "title": "...", "kind": "ball_place", "dataset": "nottinghamforest", "status": "in_progress",
         "requester": {}, "url": "...", "n_items": 24},
 "progress": {"supported": true, "done": 3, "total": 600, "remaining": 597, "fraction": 0.005, "next": {"clip": "...", "frame": 40}},
 "items":  [{"index": 0, "item": {"clip": "...", "frame_start": 10, "frame_end": 259, "step": 10}, "total": 25, "done": 3}],
 "stores": [{"clip": "...", "store": "ball_gt_marks_nottinghamforest/<stem>.jsonl", "store_mtime": "2026-11-16T11:58:00Z"}],
 "labels": [ ...rows, only for labelled units inside the job's items... ]}
```

`labels` rows by kind (every row has `clip`; unlabelled units simply have no row):

- `ball_place`: `{clip, frame_index, status: "ball"|"no_ball"|"not_broadcast", tags, label?, bbox?, center?, provenance?, tier?}`.
  `bbox`/`center` are normalised (0..1) like the sidecar; `provenance` is `labeller` (a human placed it, tier 1) or
  `machine+human_checked` (a machine box a human reviewed, tier 2). Sidecar rows carry **no per-row timestamp**: use
  `stores[].store_mtime` (when the file last changed) as the upper bound.
- `calibration`: `{clip, frame_index, status: "calibrated", correspondences, H, H_convention, pitch_dims, image_size,
  method, inliers, mean_err_m, ts, origin}` or `{status: "not_broadcast", ts, run_id?, note?}`.
- `frame_type`: `{clip, frame_index, status: <type, e.g. "not_broadcast">, ts, run_id?, note?}`.
- `ball_check`: the verdict row: `{clip, frame_index, box_index, source, verdict, bbox, corrected_bbox, label, provenance, confidence, ts}`
  (`ts` = epoch seconds).
- `identity`: the raw pair/review records that mention the clip, plus `clip`.

- **LAB-1400** (MUST) `POST /api/jobs` validates and stores a job (one atomic file) and returns `{id, url, status}`;
  a repeated `key` returns the existing job, even under concurrent submits.
  - Tests: `test_jobs.py::test_submit_returns_id_url_status_and_is_stored_one_file_per_job`,
    `::test_same_key_returns_the_existing_job_not_a_duplicate`, `::test_same_key_concurrent_submits_create_one_job`,
    `::test_validation_errors_are_422`
- **LAB-1401** (MUST) `GET /api/jobs` filters by status, title (exact or substring), kind, dataset, `requester.branch`;
  `GET /api/jobs/{id}` also resolves a unique exact title.
  - Tests: `::test_get_by_id_and_by_exact_title_and_listing_filters`, `::test_title_with_a_slash_and_an_ambiguous_title`
- **LAB-1402** (MUST) Progress is computed from the dataset's actual label store, per kind, never from a counter, and
  ignores other datasets, other clips, other frames and machine-only boxes.
  - Tests: `::test_progress_counts_only_human_labels_in_the_jobs_frames`, `::test_progress_follows_the_label_files_not_a_counter`,
    `::test_labels_in_another_dataset_do_not_count`, `::test_ball_check_progress_and_results_from_the_verdict_log`,
    `::test_calibration_and_frame_type_progress`, `::test_identity_progress`, `::test_window_step_and_start_end_aliases_and_extra_columns`
- **LAB-1403** (MUST) `GET /api/jobs/{id}/results` returns exactly the job's labelled units in the schema above.
  - Tests: `::test_results_contain_exactly_the_jobs_labels_with_provenance`
- **LAB-1404** (MUST) Jobs close via `done`/`cancel`/`reopen`; reaching 100% auto-marks `done`.
  - Tests: `::test_done_cancel_reopen_endpoints`, `::test_auto_done_at_full_progress_and_not_reopened_or_cancel_overridden`
- **LAB-1405** (MUST) Open jobs appear on the task board with a deep link into the right page and dataset; manual
  tasks keep working; a broken job file never breaks the board.
  - Tests: `::test_open_jobs_appear_on_the_task_board_and_manual_tasks_still_work`,
    `::test_board_link_points_at_the_next_unlabelled_frame`, `::test_a_broken_job_file_does_not_break_the_board_or_listing`
- **LAB-1406** (SHOULD) `jobs_client.py` is stdlib-only and standalone (served at `/api/jobs_client.py`): `submit`, `get`,
  `wait(poll)`, `results`; the CLI fills `requester` from git.
  - Tests: `::test_client_submit_get_wait_results_roundtrip`, `::test_client_errors_are_typed`,
    `::test_cli_submits_a_csv_of_windows_and_reports_status`, `::test_client_is_a_standalone_stdlib_only_file_and_served_by_the_server`
- **LAB-1407** (SHOULD) The labeller page deep link `?clip=<stem>&frame=<n>` opens that clip at that frame.
  - UNTESTED-FRONTEND
- **LAB-1408** (SHOULD) `/labeller?job=<id>` is job mode: a bar shows `job frame k / N, clip, stratum`; `J` / the button
  jumps to the next job frame that is not done (wrapping, switching clip), `K` to the previous one. Drawing a ball box,
  or marking no-ball (`N`) / not-broadcast (`B`), saves and advances (checkbox `auto-advance`). No auto-detect (Q and
  the button are off, frame 0 is not detected on load) and no forward propagation, so every mark on a job frame is a
  human `labeller` mark. Without `job=` nothing changes.
  - Tests (server half): `::test_ball_place_url_opens_job_mode_and_other_kinds_do_not`,
    `::test_frames_endpoint_expands_items_in_job_order_with_the_progress_rule`; the page itself is UNTESTED-FRONTEND
    (driven by hand in headless Chromium).

### Prepared Ball Check jobs: inline cards, disk renders, bounds (LAB-1409..1416)

`labelling_jobs submit --kind ball_check --items cards.json --prepare` submits a job **and** queues the
Prefect flow `ballcheck-prepare` (`~/code/jobs/footy_ballcheck.py`), which renders every card's still and
clip to `$FOOTY_BALLCHECK_RENDERS_DIR` (default `/mnt/storage/footy_data/ballcheck_renders`) so the phone only
downloads finished files. `labelling_jobs prepare <job> [--hold] [--where mini|desktop]` does the same for an
existing job (read-only for the job; the renders just land in the cache). `labelling_jobs gc [--dry-run]` is the
housekeeping (it also runs hourly inside the server, `FOOTY_JOBS_GC_INTERVAL_S`).

**Defaults** (all env-overridable; see `jobs.py` / `ball_check_cache.py`):

| thing | default |
|---|---|
| cards per ball_check job | 500; `max_cards` / `--allow-large` raises it to a hard ceiling of **2000** |
| frames per frame-kind job | 5000 (hard 20000) |
| unlabelled cards across jobs that are prepared / being prepared | **1500** (a `--prepare` over it is refused with HTTP 429, nothing is created) |
| render cache size cap | **5 GB**, LRU |
| free-disk floor on `/mnt/storage` | **20 GB** (preparing is refused below it, HTTP 507; a render run stops at it) |
| renders of a done / cancelled job | deleted **24 h** after it closed |
| renders of a live job | deleted after **3 days** with no labelling activity on it |
| jobs and cards | **never expire**; only the renders are temporary. Verdicts are never deleted |

- **LAB-1409** (MUST) A `ball_check` item with a `bbox` (normalised `{x,y,w,h}`) and no `box_index` is an *inline
  card*; the job serves its cards in item order at `/ball_check?dataset=D&source=list&job=<id>` with no review-list file
  and no `labeller_datasets.json` edit. Pointer items (`{clip, frame_index, box_index, source?}`) behave as before.
  - Tests: `test_ball_check_prepare.py::test_inline_cards_are_served_in_item_order_without_a_list_file_or_registry_edit`,
    `::test_pointer_cards_still_work_unchanged`
- **LAB-1410** (MUST) An inline card's `box_index` is `2**30 + hash(job id, item position)` (so a list file can never
  reach it and two jobs on one dataset never share `(clip, frame, box_index)`); its verdict is written with
  `source: "list"` to `<marks_dir>/ball_checks/<clip>.jsonl` exactly like a list-mode verdict, so existing scorers
  and the "list verdicts are not a detector precision sample" rule keep working.
  - Tests: `::test_inline_verdict_lands_in_the_marks_dir_like_a_list_verdict_and_counts_as_progress`,
    `::test_two_jobs_on_one_dataset_never_collide_even_on_the_same_frame`
- **LAB-1411** (MUST) The phone only receives `label`, `provenance`/`role`, `confidence`, `note`, `other_bbox` (the keys the
  page prints or draws) plus geometry; every other key is hidden: kept in the job, returned in `results`
  (`labels[].meta`, `items[].item`), never sent to the page. A blind batch supplies none of the shown keys.
  - Tests: `::test_hidden_keys_are_never_sent_to_the_page_but_pass_through_to_results`
- **LAB-1412** (MUST) Crops and clips are cached on disk (key = clip file identity, frame, box and other-box geometry,
  pad, reticle, `at`, render version and the render constants), checked before rendering, written atomically
  (temp file under `.tmp/`, outside `blobs/`), and concurrent requests for one key share one render. The prepare flow
  fills the cache under the keys the server hands it (`GET /api/jobs/<job>/prepare_plan`), with the same render
  functions, so a prepared card is served from disk, also after a restart.
  - Tests: `::test_disk_cache_miss_then_hit_and_lru_touch`, `::test_disk_write_is_atomic_and_stages_outside_the_served_folder`,
    `::test_concurrent_renders_of_one_key_are_a_single_render`, `::test_key_covers_everything_that_changes_the_bytes`,
    `::test_server_serves_a_prepared_card_from_disk_and_survives_a_restart`
- **LAB-1413** (MUST) Job state: submitted with `--prepare` a job is `preparing` (with `prepare.prepared/total`) until the
  run finishes, then `open`; it is not a board link and the page shows the progress instead of cards. A failed
  run leaves the job `open` (cards render on demand) with `prepare.state = failed` and the reason. Only one prepare
  run per job at a time; a run that stops reporting is declared failed by gc.
  - Tests: `::test_submit_with_prepare_is_preparing_until_the_flow_finishes`, `::test_failed_prepare_releases_the_job_so_it_still_works_on_demand`,
    `::test_only_one_prepare_per_job_and_a_stale_one_can_be_redone`
- **LAB-1414** (MUST) Bounds: card cap (+ hard ceiling), queued-card cap, free-disk floor, cache size cap (see table).
  - Tests: `::test_card_cap_is_rejected_then_allowed_up_to_the_hard_ceiling`, `::test_cli_refuses_over_the_cap_before_posting`,
    `::test_queued_card_cap_refuses_a_prepare_that_would_pass_it`, `::test_disk_floor_refuses_to_prepare`,
    `::test_lru_eviction_order_orphans_then_finished_jobs_then_open_jobs`
- **LAB-1415** (MUST) gc deletes the renders of done/cancelled jobs 24 h after closing and of live jobs 3 days after their last
  labelling activity (job stays open, `prepare.state = evicted`), never touches verdicts or jobs, and creates nothing in the
  registry, clips or marks listings.
  - Tests: `::test_gc_deletes_renders_of_a_done_job_after_the_grace_but_keeps_its_verdicts`, `::test_idle_job_loses_renders_after_3_days_but_stays_open`,
    `::test_housekeeping_does_not_grow_the_clip_or_registry_listings`
- **LAB-1416** (MUST) When labelling resumes on a job whose renders were evicted (the job is opened, or one of its cards is
  requested) a fresh `ballcheck-prepare` run is queued for its remaining unlabelled cards (judged ones are skipped); the
  phone is served by on-demand rendering meanwhile. A failed attempt backs off for 10 minutes.
  - Tests: `::test_labelling_resuming_on_an_evicted_job_reprepares_only_the_remaining_cards`, `::test_first_card_request_of_an_evicted_job_also_reprepares`

## 19b. Broadcast review (job kind `broadcast_review`)

Human yes/no on frames the broadcast classifier (`classifier.py`, argmax of `Yes`/`No`, i.e. `p_yes > 0.5`,
the rule `split_broadcast_segments` cuts with) was unsure about, plus a random sample. Produced per match by the
Prefect flow `broadcast-review` (jobs repo, `footy_broadcast_review.py`). Item:
`{clip: <match key>, frame, ts_s, p_yes, pred, reason: edge_mid|edge_boundary|random, dist_to_transition_s}`.
Frames are `<dataset clips_dir>/<match>/f<frame>.jpg`; verdicts (`yes`/`no`/`unsure`) are written to
`<marks_dir>/broadcast_review/<match>.json` and counted as progress. Page: `/broadcast_review?dataset=..&job=<id>`
(Y / N / U keys; the model's call and `p_yes` are shown while answering). Training: `python -m
footy_track.scripts.export_broadcast_review --dataset broadcast_review --out <dir>` writes `train/{Yes,No}/`
for `scripts/train_classifier.py --local-dataset` (`unsure` skipped; train/valid/test come from the split authority `footy_track.splits`, docs/data.md "Train/eval partitioning"; `--valid-matches` is an explicit whole-match override).

## 19c. Labelling hooks: refit, reorder and kick off work *while* a job is being labelled (LAB-1417..1426)

Any job (every kind) can carry **hooks**: things the server does as labels arrive. Code: `hooks.py` (event point,
triggers, registry, bounds), `job_edit.py` (reorder / append / history), `jobs_routes.py` (API), `broadcast_review.py`
(`broadcast_calibrate`, the first action). Stage 1 of "learn the classifier's real boundary while labelling".

**Event point.** Every label write calls `hooks.emit(kind, clip=, unit=, verdict=)` once, after it is persisted:

| write | where it is emitted | event `kind` / `unit` | job kinds it can answer |
|---|---|---|---|
| Ball Check verdict / undo | `ball_check._append_verdict` | `ball_check` / `(source, frame, box)` | ball_check |
| ball_place: box edit, no-ball, not-broadcast (+ clears) | `server.py` routes (`/edit`, `/no-ball`, `/not-broadcast`, `/*/clear`), `review.py` (`/review/correct`, `/review/delete`) | `ball_place` / frame | ball_place |
| calibration GT save / delete / accepted proposal | `calibration.py` `gt_save`/`gt_delete`, `calibration_proposals.accept` | `calibration_gt` / frame | calibration |
| frame-type flag (not_broadcast, too_few_points) set / clear | `calibration.mark_*` / `clear_*` | `frame_type` / frame | calibration, frame_type |
| identity pair / review | `identity_routes.label_pair` (one event per clip of the pair), `submit_review` | `identity` / - | identity |
| broadcast-review verdict | `broadcast_review.set_verdict` | `broadcast_review` / frame | broadcast_review |

`emit` only queues a small event (bounded queue, `EVENT_QUEUE_MAX`; a full queue drops the event and counts it); one
daemon thread matches it to the live jobs *that contain that unit* (same dataset, canonical clip id for the
clip-aliased kinds) and counts it against their hooks; actions run on a 3-thread pool. Hook errors are logged and
recorded, never raised into the write.
`FOOTY_LABELLING_HOOKS=0` turns the whole thing off.

**Config** (`job["hooks"]`; at submit `hooks: [...]`, later `PUT /api/jobs/<ref>/hooks`, CLI `labelling_jobs hooks REF
--set hooks.json`, `submit --hooks hooks.json`; `GET /api/jobs/<ref>/hooks` shows config, per-hook runtime and history):

```json
[{"name": "calibrate",
  "trigger": {"every_n": 20, "min_interval_s": 60, "on_done": false},
  "action": {"type": "builtin", "name": "broadcast_calibrate", "params": {}}},
 {"name": "rescore",
  "trigger": {"every_n": 50, "min_interval_s": 600, "on_done": true},
  "action": {"type": "prefect", "deployment": "my-flow/my-deployment", "params": {"top": 100}}}]
```

**Actions.** `hooks.register_action(name, fn, kinds=(...))`; `fn(ctx) -> dict` gets `ctx.job` (fresh), `ctx.params`,
`ctx.event`, `ctx.trigger` (`every_n` / `on_done` / `manual`), runs with the job's dataset current, and steers the job
via `ctx.reorder(...)` / `ctx.append(...)`. Built in: `shuffle_unanswered` and `sort_by_field` (`params.key`,
`descending`) for any kind, `broadcast_calibrate` for broadcast_review. A `prefect` action queues `deployment` through
`prefect_client.create_flow_run` with `{job_id, hook, trigger, dataset, labeller_url?, **params}`; the run stays "in
flight" until Prefect reports a terminal state (polled), and the flow talks back through the reorder / append API.
`FOOTY_HOOKS_PREFECT_ALLOW=a/b,c/d` restricts which deployments a job may name.

**Bounds** (per `(job, hook)` unless said): one run in flight (a later trigger waits; `on_done` is skipped if its run is
still in flight, so give it its own hook); `min_interval_s` debounce (default 60; at least 30 for prefect actions);
after a failed run no new run for **10 min** (`BACKOFF_S`, like `maybe_reprepare`; also persisted, so a restart keeps it);
a run that never reports is declared failed after 2 h; at most **8 hooks per job**, 3 prefect runs in flight across
all jobs, 2000 queued events; hook params at most 4000 bytes of JSON. `POST /api/jobs/<ref>/hooks/<name>/run` runs one
now (skips the debounce only).

**Mutation API** (`job_edit.py`; live jobs only):

* `POST /api/jobs/<ref>/reorder {order: [item positions], by?, note?, if_rev?}` - the listed *unanswered* items go first,
  in that order; the other unanswered items follow in their old order; **answered items never move** (unanswered items
  are re-dealt into the positions unanswered items already held), nothing is added or removed. Answered / unknown /
  repeated positions are skipped and counted. "Answered" = every unit of the item has a label in the real store.
* `POST /api/jobs/<ref>/append {items, by?, note?, if_rev?}` - same item shapes as submit; items whose unit is already
  in the job are skipped; bounded by the job's card/frame cap (its stored `max_cards`, else the default; hard ceiling
  as at submit), at most 500 per call, and for a prepared ball_check job the queued-card cap (429) and disk floor (507).
  Appended items get `added_by: {by, ts, note?}` (an item key: it is echoed in results); appended inline cards get
  their `box_index` from the job id and their real position. They render on demand (call `/prepare` to pre-render).
* `rev` on the job counts reorders/appends (`GET /api/jobs/<ref>/rev` is the cheap poll); `hook_history` keeps the last 50
  reorders, appends, hook runs (with the action's summary, e.g. a refit) and errors; `hook_state` keeps last run /
  backoff per hook.

**Pages.** `/broadcast_review?job=`, `/ball_check?...&job=` and `/labeller?job=` poll `rev` (every 10-12 s); on a change
they re-read the job order, keep the frame/card on screen as the user's place and take the *next* one from the new order.

- **LAB-1417** (MUST) Every label write of every kind above emits exactly one event after it is persisted, naming the job it
  belongs to when the unit is in a live hooked job; emitting never blocks, never raises into the write, drops (and counts)
  when the queue is full, and runs hooks off the request thread.
  - Tests: `test_hooks.py::test_event_fires_for_a_non_broadcast_kind_ball_check`, `::test_event_fires_for_broadcast_review_and_only_for_that_jobs_frames`,
    `::test_event_fires_for_ball_place_marks`, `::test_event_fires_for_frame_type_flags_on_calibration_and_frame_type_jobs`,
    `::test_event_fires_for_identity_pairs_and_reviews`, `::test_a_failing_or_full_hook_pipeline_never_breaks_the_write`,
    `::test_emit_is_off_the_request_thread`
- **LAB-1418** (MUST) Hooks are stored in the job, set at submit and replaced through the API / CLI; the config is validated
  (names, triggers, registered action for the job kind, prefect deployment name and debounce floor) and capped at 8 per job.
  - Tests: `::test_hooks_are_set_at_submit_and_updated_through_the_api`, `::test_hook_config_is_validated_and_capped`, `::test_cli_hooks_roundtrip`
- **LAB-1419** (MUST) Triggers and bounds: `every_n` counts the job's own events, `min_interval_s` debounces, at most one run in
  flight per (job, hook), a failure backs off 10 min (persisted), `on_done` runs once at completion, a manual run skips the
  debounce only; a `prefect` action queues the named deployment with `{job_id, ...params}`, stays in flight until Prefect
  says the run ended, backs off when queueing fails or the run fails, and is capped across jobs.
  - Tests: `::test_every_n_triggers_after_n_events_and_debounce_holds_the_rest`, `::test_at_most_one_run_in_flight_per_job_and_hook`,
    `::test_a_failed_run_backs_off_for_ten_minutes_and_is_recorded`, `::test_on_done_runs_once_when_the_job_completes`,
    `::test_manual_run_skips_the_debounce_but_not_the_backoff`, `::test_prefect_action_queues_the_deployment_and_stays_in_flight_until_it_ends`,
    `::test_prefect_failure_backs_off_and_unreachable_prefect_is_recorded`, `::test_prefect_inflight_cap_across_jobs`,
    `::test_generic_builtin_actions_shuffle_and_sort_any_kind`
- **LAB-1420** (MUST) Reordering moves only unanswered items; answered items keep their positions, no item is added, lost or
  duplicated, verdicts and progress are unchanged; the job gets a new `rev` and a history entry; a no-op changes nothing;
  closed jobs are refused; `if_rev` guards against a concurrent change.
  - Tests: `::test_reorder_moves_only_unanswered_items_and_never_answered_ones`, `::test_reorder_is_a_noop_when_nothing_changes_and_honours_if_rev`,
    `::test_reorder_works_for_window_items_of_a_frame_kind`
- **LAB-1421** (MUST) Appending records provenance (`added_by`), skips duplicates, keeps answered items and their verdicts, and
  respects the per-job cap, the per-call cap, and (prepared ball_check) the queued-card cap and disk floor; nothing is written
  when refused.
  - Tests: `::test_append_adds_items_with_provenance_skips_duplicates_and_keeps_answers`, `::test_append_respects_the_per_job_cap_and_per_call_cap`,
    `::test_append_to_a_prepared_ball_check_job_honours_queued_cap_and_disk_floor`, `::test_append_is_refused_for_closed_jobs_and_validates_items`
- **LAB-1422** (MUST) The pages that walk a job in order pick up a new order without losing the user's place: the server serves the
  job in its new order with `rev`; the pages poll `rev`, keep the current frame/card and take the next from the new order.
  - Tests (server half): `::test_broadcast_page_api_serves_the_new_order_and_rev`, `::test_ball_check_queue_serves_the_new_order_and_rev`,
    `::test_labeller_job_frames_carry_the_rev`, `::test_pages_poll_the_rev_and_keep_the_users_place` (text check); the page logic itself is
    UNTESTED-FRONTEND.
- **LAB-1423** (MUST) `broadcast_calibrate` fits Platt scaling `P(yes) = sigmoid(a * logit(p_yes) + b)` (p clipped to [1e-6, 1-1e-6],
  `unsure` excluded, all yes/no verdicts of the dataset) and writes each new fit as an immutable
  `<marks_dir>/broadcast_review/calibration/vNNN.json` (`a, b, n, n_yes, n_no, boundary_p_yes` = the p_yes where calibrated p = 0.5,
  `ts`, `version`, `matches`); the same data again reuses the latest version; fewer than 10 yes or 10 no (with a stored p_yes) is a
  no-op that says why.
  - Tests: `::test_platt_fit_recovers_the_known_scaling_and_boundary`, `::test_broadcast_calibrate_writes_a_versioned_file_and_reorders_by_uncertainty`,
    `::test_broadcast_calibrate_is_a_noop_with_a_reason_when_there_are_too_few_verdicts`, `::test_the_real_dataset_verdicts_fit_when_available`
- **LAB-1424** (MUST) After a fit the job's remaining frames are reordered by `|a * logit(p_yes) + b|` ascending (distance to the fitted
  boundary on the logit scale) plus Gaussian jitter (0.5) and a per-time-bucket penalty (60 s buckets, 0.15 per more-uncertain frame
  already ranked) so no cluster hogs the top; answered frames never move; it fires as verdicts arrive.
  - Tests: `::test_broadcast_calibrate_writes_a_versioned_file_and_reorders_by_uncertainty`, `::test_broadcast_calibrate_fires_as_verdicts_arrive`
- **LAB-1425** (MUST) The broadcast review page shows the calibrated p and the fitted boundary (and version, n) next to the raw p_yes.
  - Tests (server half): `::test_broadcast_calibrate_writes_a_versioned_file_and_reorders_by_uncertainty` (`calibration`, `p_cal` in the job view);
    the page text is UNTESTED-FRONTEND.
- **LAB-1426** (MUST) Actions are registered by name (`hooks.register_action`) so each labelling kind can add its own; the registry
  rejects an action on a job kind it is not for.
  - Tests: `::test_hook_config_is_validated_and_capped`, `::test_generic_builtin_actions_shuffle_and_sort_any_kind`

### How other kinds use it (documentation only)

* **ball_check** (inline cards): attach `{"type": "prefect", "deployment": "ballcheck-rescore/ballcheck-rescore", "params": {...}}`
  with `every_n: 50`. The flow reads `GET /api/jobs/<job_id>/results` (the verdicts so far, hidden keys in `labels[].meta`), refits
  or re-scores the detector candidates, then `POST /api/jobs/<job_id>/reorder` with the positions of the unanswered cards by the new
  score (read the current order with `GET /api/jobs/<job_id>`, pass `if_rev`) and `POST .../append` with fresh near-boundary
  cards (`bbox`, hidden `score` keys; caps, queued-card cap and disk floor apply; follow with `POST .../prepare` to pre-render).
  An in-process equivalent is `register_action("rescore_cards", fn, kinds=("ball_check",))` where `fn` calls `ctx.reorder` /
  `ctx.append`.
* **ball_place**: same shape with window items; `sort_by_field` / `shuffle_unanswered` already work on them, and a prefect action
  can append `{clip, frame}` items around frames the labeller found hard. Items are frames/windows, so reorder moves whole windows.
* Pointer ball_check jobs (no inline cards) are served by the review list, not the job's item order, so reordering them has no
  page effect; use inline cards.

## 19d. Review fixes: the architecture review's bug-fix pass (LAB-1427..1434)

Tests: `test_review_bugfixes.py` (names below are within it).

- **LAB-1427** (MUST) A Ball Check verdict only counts as saved when the HTTP status is 2xx **and** the JSON body has
  `ok: true` (the endpoint answers some validation errors with 200 `{"ok": false}`, and a 500 has no JSON). The card advances
  optimistically and the next ones prefetch as before; when the save did not land (non-2xx, `ok` not true, network error) the
  card goes back to the front of the queue, `judged`/`remaining` are rolled back and the status line says
  `Verdict NOT saved (...)`. `save_box` uses the same check.
  - Tests: `::test_ball_check_judge_keeps_the_card_when_the_save_fails` (the page's own `judge` run under node); the layout of
    the message is UNTESTED-FRONTEND.
- **LAB-1428** (MUST) Blocking cv2 / label-file work never runs on the event loop: `/clips/status` runs in a worker thread and
  the identity data routes are threadpool `def`s, so a slow request does not stall other pages (B1, B8).
  - Tests: `::test_clips_status_does_not_block_the_event_loop`, `::test_identity_routes_do_not_block_the_event_loop`,
    `::test_identity_data_routes_are_not_async_with_blocking_bodies`
- **LAB-1429** (MUST) `jobs.compute` and `broadcast_review.job_view` read each match's store once per call, not once per item (B2).
  - Tests: `::test_jobs_compute_reads_each_broadcast_review_store_once`, `::test_broadcast_job_view_reads_each_store_once`
- **LAB-1430** (MUST) Every calibration-GT read-modify-write (save, delete, accept a proposal, flag/unflag not-broadcast) goes
  through `calibration.update_gt` under one process lock; nothing is written when the document did not change (B5).
  - Tests: `::test_concurrent_calibration_gt_updates_lose_nothing` (with `::test_unlocked_read_modify_write_would_lose_updates`
    proving the race is real), `::test_update_gt_writes_nothing_when_the_document_is_unchanged`
- **LAB-1431** (MUST) A clip name from a request body that becomes part of a path (`/ball_check/verdict`, `/ball_check/undo`,
  `/ball_check/save_box`, `/review/correct`, `/review/delete`) must match `names.safe_clip_name` (`[A-Za-z0-9][A-Za-z0-9._-]*`, at
  most 201 chars, no `..`); anything else is a 422 and nothing is written (B6).
  - Tests: `::test_ball_check_verdict_rejects_unsafe_clip_names`, `::test_review_and_save_box_reject_unsafe_clip_names`,
    `::test_real_clip_names_still_pass`
- **LAB-1432** (MUST) `/review/correct` and `/review/delete` replace the sidecar atomically (temp file + fsync + rename, mode kept):
  a crash mid-write leaves the old file.
  - Tests: `::test_review_correct_and_delete_replace_the_file_atomically`, `::test_a_crash_mid_write_leaves_the_sidecar_intact`
- **LAB-1433** (MUST) `GET /ingest` is a 404, not a 500 page, and the hub has no Ingest card (OPEN-1); the ingest APIs still work.
  - Tests: `::test_ingest_page_route_is_gone_and_the_hub_does_not_link_to_it`
- **LAB-1434** (MUST) The sidecar box scan behind the Ball Check `sidecar` queue and `/review/queue` re-parses only the sidecars
  whose (mtime, size, inode) or video path changed (B13). The deploy script also builds the candidate pool after a restart
  (`warm_ball_check --source candidates`, B12).
  - Tests: `::test_sidecar_scan_reparses_only_changed_files`; `test_task_board.py::test_warm_helper_candidate_pool` (the
    `labeller_deploy.sh` wiring is UNTESTED)


## 15. Traceability summary

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
| 10. Review UI (LAB-9xx) | 7 | 0 | 0 | 7 |
| 11. Ball Check (LAB-10xx) | 12 | 11 | 0 | 1 |
| 12. Confidence sampling (LAB-11xx) | 5 | 5 | 0 | 0 |
| 16. Pitch calibration (LAB-12xx) | 23 | 18 | 0 | 5 |
| 18. Datasets (LAB-13xx) | 6 | 5 | 0 | 1 |
| 19. Labelling jobs (LAB-14xx) | 8 | 7 | 0 | 1 |
| 19c. Labelling hooks (LAB-1417..1426) | 10 | 8 | 0 | 2 |
| 19d. Review fixes (LAB-1427..1434) | 8 | 7 | 1 | 0 |
| **Total** | **156** | **116** | **9** | **31** |

Open items: 4 (OPEN-1..4) + 1 fixed on main (undo provenance, `95b60cc`).
Test suite: 138 tests in `tests/labeller/` (all green), plus the
feature-store round-trip fidelity leg in `tests/feature_store/`.
