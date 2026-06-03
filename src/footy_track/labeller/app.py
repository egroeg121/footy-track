"""Streamlit UI for the SAM3 video labeller.

Workflow:
    1. Point at a local video clip.
    2. Draw bounding boxes on frame 0, one class at a time, and add them.
    3. Run SAM3 propagation across the whole clip.
    4. Preview the propagated boxes and export ``FrameDetections`` JSON.

Launch with::

    uv run streamlit run src/footy_track/scripts/run_labeller.py
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import streamlit as st
from PIL import Image

# Must be imported before streamlit_drawable_canvas: restores image_to_url on
# Streamlit >= 1.40 so the canvas component works.
import footy_track.labeller._canvas_compat  # noqa: F401  # isort: skip
from streamlit_drawable_canvas import st_canvas  # isort: skip

from footy_track.detectors.ultralytics import (
    CURRENT_BEST_DETECTOR_CHECKPOINT,
    get_current_best_detector,
)
from footy_track.detectors.utils import calculate_iou, color_map
from footy_track.labeller.video_utils import (
    BackgroundLabeller,
    LabelledObject,
    _warmup_done,  # noqa: F401 — imported for future sidebar status use
    export_frames_json,
    extract_first_frame,
)
from footy_track.schema import DETECTION_CLASSES, ObjectDetection

CANVAS_DISPLAY_WIDTH = 720


def _rgb_hex(label: str) -> str:
    r, g, b = color_map.get(label.lower(), (255, 0, 0))
    return f"#{r:02x}{g:02x}{b:02x}"


def _load_first_frame_pil(video_path: Path) -> tuple[Image.Image, int, int]:
    frame_bgr = extract_first_frame(video_path)
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    h, w = frame_rgb.shape[:2]
    return Image.fromarray(frame_rgb), w, h


def _init_state() -> None:
    st.session_state.setdefault("objects", [])  # list[LabelledObject]
    st.session_state.setdefault("canvas_key", 0)
    st.session_state.setdefault("frames", None)  # list[FrameDetections] | None
    # Background SAM3 inference (persists across Streamlit reruns).
    st.session_state.setdefault("bg", BackgroundLabeller())
    st.session_state.setdefault("paused_at_frame", None)  # int | None
    st.session_state.setdefault("correcting", False)
    # Frame currently shown in the paused editor (prev/next navigation).
    st.session_state.setdefault("correction_frame", None)  # int | None
    # Edited boxes captured from the editable canvas this run.
    st.session_state.setdefault("edited_objects", [])
    # Previous canvas box centers + the index of the most-recently-moved box,
    # used to highlight the matching row in the side panel (selection heuristic).
    st.session_state.setdefault("prev_centers", [])
    st.session_state.setdefault("active_obj", None)  # int | None
    # canvas_key the list-sync last reconciled against (avoids clobbering a fresh
    # re-detect / class-change before the canvas has re-rendered with it).
    st.session_state.setdefault("_synced_key", None)
    # Path of the video we've already auto-seeded, so it only runs once per video.
    st.session_state.setdefault("autoseeded_video", None)


def _nms_filter(
    detections: list[ObjectDetection], iou_threshold: float = 0.5
) -> list[ObjectDetection]:
    """Greedy IoU NMS — keep highest-confidence box, drop overlaps above threshold."""
    kept: list[ObjectDetection] = []
    for det in sorted(detections, key=lambda d: d.confidence, reverse=True):
        if all(calculate_iou(det, k) <= iou_threshold for k in kept):
            kept.append(det)
    return kept


def _yolo_seed_objects(
    video_path: Path,
    model_path: str,
    min_confidence: float,
    orig_w: int,
    orig_h: int,
    iou_threshold: float = 0.5,
) -> list[LabelledObject]:
    """Run the YOLO detector on frame 0 and return NMS-filtered seed objects.

    Detections come back normalized; we convert to absolute xyxy pixel coords for
    :class:`LabelledObject`, the seed format SAM3 expects.
    """
    import tempfile  # noqa: PLC0415

    frame_bgr = extract_first_frame(video_path)
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
        tmp_path = Path(f.name)
    cv2.imwrite(str(tmp_path), frame_bgr)
    try:
        detector = get_current_best_detector(min_confidence=min_confidence)
        if model_path:
            from footy_track.detectors.ultralytics import (  # noqa: PLC0415
                UltralyticsObjectDetector,
            )

            detector = UltralyticsObjectDetector(
                model_uri=model_path,
                min_confidence=min_confidence,
                use_model_names=True,
            )
        fd = detector.predict_from_path(tmp_path)
    finally:
        tmp_path.unlink(missing_ok=True)

    filtered = _nms_filter(fd.detections, iou_threshold=iou_threshold)
    objects: list[LabelledObject] = []
    for det in filtered:
        x1 = det.x * orig_w
        y1 = det.y * orig_h
        x2 = (det.x + det.w) * orig_w
        y2 = (det.y + det.h) * orig_h
        objects.append(LabelledObject(label=det.label, bbox_xyxy_abs=(x1, y1, x2, y2)))
    return objects


def main() -> None:  # noqa: PLR0912, PLR0915
    st.set_page_config(page_title="SAM3 Video Labeller", layout="wide")
    _init_state()

    st.title("⚽ SAM3 Video Labeller")
    st.caption(
        "Mark players, ball and officials on the first frame; SAM3 propagates "
        "them through the clip."
    )

    # --- Sidebar: all settings ---
    with st.sidebar:
        st.header("1. Video")
        video_path_str = st.text_input("Video path", value="")

        st.header("2. Auto-detect seeds (YOLO)")
        auto_seed = st.checkbox(
            "Auto-seed on load",
            value=True,
            help="Run YOLO on frame 0 automatically when a video is loaded.",
        )
        yolo_model = st.text_input(
            "YOLO checkpoint (blank = current best)",
            value="",
            help=f"Default: {CURRENT_BEST_DETECTOR_CHECKPOINT}",
        )
        yolo_conf = st.slider("YOLO confidence", 0.0, 1.0, 0.35, 0.05)
        yolo_iou = st.slider("YOLO NMS IoU", 0.1, 0.9, 0.5, 0.05)

        st.header("3. Class")
        label = st.selectbox("Class to draw", DETECTION_CLASSES, index=0)
        st.markdown(
            f"<div style='width:100%;height:18px;background:{_rgb_hex(label)};"
            "border-radius:4px'></div>",
            unsafe_allow_html=True,
        )

        st.header("4. Model")
        model_uri = st.text_input("SAM3 checkpoint (blank = default)", value="")
        min_conf = st.slider("Min confidence", 0.0, 1.0, 0.25, 0.05)
        st.divider()
        st.caption(
            "JIT kernels cached in ~/.cache/torch_inductor_sam3 — fast after first run."
        )

    if not video_path_str:
        st.info("Enter a video path in the sidebar to begin.")
        return

    # Drop surrounding quotes (e.g. from drag-and-drop or shell copy-paste) but
    # keep any quotes that are genuinely inside the path.
    video_path_str = video_path_str.strip()
    if len(video_path_str) >= 2 and video_path_str[0] == video_path_str[-1] in (
        "'",
        '"',
    ):
        video_path_str = video_path_str[1:-1]

    video_path = Path(video_path_str).expanduser()
    if not video_path.exists():
        st.error(f"Video not found: {video_path}")
        return

    try:
        _frame0, orig_w, orig_h = _load_first_frame_pil(video_path)
    except Exception as exc:  # noqa: BLE001
        st.error(f"Could not read first frame: {exc}")
        return

    scale = CANVAS_DISPLAY_WIDTH / orig_w

    bg: BackgroundLabeller = st.session_state.bg

    # Auto-seed once per video on load (before the canvas renders, so the seeds
    # prefill it on the same pass). Runs only when enabled and not yet seeded.
    if (
        auto_seed
        and not bg.running
        and st.session_state.autoseeded_video != str(video_path)
    ):
        st.session_state.autoseeded_video = str(video_path)
        with st.spinner("🤖 Auto-detecting objects with YOLO…"):
            try:
                seeds = _yolo_seed_objects(
                    video_path=video_path,
                    model_path=yolo_model,
                    min_confidence=yolo_conf,
                    orig_w=orig_w,
                    orig_h=orig_h,
                    iou_threshold=yolo_iou,
                )
            except Exception as exc:  # noqa: BLE001
                st.exception(exc)
                seeds = []
        if seeds:
            st.session_state.objects = list(seeds)
            st.session_state.canvas_key += 1

    objects: list[LabelledObject] = st.session_state.objects

    # Two-pane layout: video + controls on the left, detected-object list on the
    # right. side_panel is a container we fill in further down.
    main_col, side_col = st.columns([3, 1])
    side_panel = side_col.container()
    main = main_col.container()

    completed = bg.completed_frames()
    paused_frame = st.session_state.paused_at_frame
    correcting = (
        st.session_state.correcting
        and paused_frame is not None
        and 0 <= paused_frame < len(completed)
    )

    # The main window has three modes:
    #   • running    -> latest completed frame with boxes (live, read-only)
    #   • correcting -> editable canvas on the selected paused frame (prev/next)
    #   • seeding    -> editable canvas on frame 0, prefilled with the YOLO seeds
    # edited_objects holds whatever the canvas currently contains, in both
    # editable modes, so Run / Restart read straight from it.
    st.session_state.edited_objects = []

    if bg.running and completed:
        latest_idx = len(completed) - 1
        latest = completed[latest_idx]
        cap = cv2.VideoCapture(str(video_path))
        cap.set(cv2.CAP_PROP_POS_FRAMES, latest_idx)
        ok, frame_bgr = cap.read()
        cap.release()
        if ok:
            live = _draw_boxes_on_array(frame_bgr, latest)
            main.image(live, channels="RGB", width="stretch")
        main.caption(f"🔴 Live — frame {latest_idx} ({len(latest.detections)} objects)")
    elif correcting:
        # Frame being edited (defaults to where we paused); prev/next move it.
        cf = st.session_state.correction_frame
        if cf is None or not (0 <= cf < len(completed)):
            cf = paused_frame
            st.session_state.correction_frame = cf
        tool = main.radio(
            "Tool",
            ["✋ Edit (move/resize/delete)", "✏️ Draw new"],
            horizontal=True,
            key="tool_correct",
        )
        draw_mode = "transform" if tool.startswith("✋") else "rect"
        cap = cv2.VideoCapture(str(video_path))
        cap.set(cv2.CAP_PROP_POS_FRAMES, cf)
        ok, frame_bgr = cap.read()
        cap.release()
        if ok:
            seeds = _detections_to_seeds(completed[cf], orig_w, orig_h)
            with main:
                st.session_state.edited_objects = _render_editable_canvas(
                    frame_bgr,
                    seeds,
                    label,
                    scale,
                    orig_h,
                    key=f"edit_{cf}",
                    drawing_mode=draw_mode,
                )
    else:
        # Seeding: editable canvas on frame 0 prefilled with the current objects
        # (e.g. YOLO auto-seeds). Drag/resize/delete then Run.
        tool = main.radio(
            "Tool",
            ["✋ Edit (move/resize/delete)", "✏️ Draw new"],
            horizontal=True,
            key="tool_seed",
        )
        draw_mode = "transform" if tool.startswith("✋") else "rect"
        frame0_bgr = extract_first_frame(video_path)
        # Key reflects object count + classes + tool so the canvas re-reads its
        # background (with baked-in numbers) whenever the object set changes.
        sig = "-".join(o.label for o in objects)
        with main:
            st.session_state.edited_objects = _render_editable_canvas(
                frame0_bgr,
                _objects_to_seeds(objects),
                label,
                scale,
                orig_h,
                key=f"seed_{st.session_state.canvas_key}_{len(objects)}_{draw_mode}_{hash(sig) & 0xFFFF}",
                drawing_mode=draw_mode,
            )
        main.caption(
            f"**Edit**: click a box to move/resize (Delete key removes). "
            f"**Draw new**: drag a **{label}** box (class from sidebar)."
        )

        # Keep the right-hand list in sync with the canvas WITHOUT rerunning:
        # mirror the canvas's current boxes into `objects` so the list + numbers
        # reflect new/removed boxes on the next natural interaction. No st.rerun()
        # here — that caused a flashing rerun loop.
        edited = st.session_state.edited_objects
        if edited and len(edited) != len(objects):
            st.session_state.objects = list(edited)

    # --- Selection heuristic: highlight the row whose box just moved most ---
    # The canvas can't tell Python which box is selected, so we infer it: the box
    # whose centre shifted most since the last rerun is treated as "active".
    if not bg.running:
        cur_centers = [
            (
                (o.bbox_xyxy_abs[0] + o.bbox_xyxy_abs[2]) / 2,
                (o.bbox_xyxy_abs[1] + o.bbox_xyxy_abs[3]) / 2,
            )
            for o in st.session_state.edited_objects
            if o.bbox_xyxy_abs is not None
        ]
        prev = st.session_state.prev_centers
        if cur_centers and len(cur_centers) == len(prev):
            deltas = [
                (cx - px) ** 2 + (cy - py) ** 2
                for (cx, cy), (px, py) in zip(cur_centers, prev, strict=False)
            ]
            mx = max(range(len(deltas)), key=lambda i: deltas[i])
            # Only treat as a selection if it actually moved a noticeable amount.
            if deltas[mx] > 4.0:
                st.session_state.active_obj = mx
        st.session_state.prev_centers = cur_centers

    # --- Frame navigation (only while paused): slider + prev/next ---
    if correcting:
        cf = st.session_state.correction_frame
        last = len(completed) - 1
        if last > 0:
            slid = main.slider("Frame", 0, last, cf, key=f"scrub_{paused_frame}")
            if slid != cf:
                st.session_state.correction_frame = slid
                st.rerun()
        nav_prev, nav_lbl, nav_next = main.columns([1, 2, 1])
        with nav_prev:
            if st.button("⬅️ Prev", disabled=cf <= 0, use_container_width=True):
                st.session_state.correction_frame = cf - 1
                st.rerun()
        with nav_lbl:
            st.markdown(
                f"<div style='text-align:center'>Frame <b>{cf}</b> / {last}</div>",
                unsafe_allow_html=True,
            )
        with nav_next:
            if st.button("Next ➡️", disabled=cf >= last, use_container_width=True):
                st.session_state.correction_frame = cf + 1
                st.rerun()

    # --- Playback control bar directly under the video window ---
    ctrl_run, ctrl_pause = main.columns(2)
    with ctrl_run:
        if correcting:
            cf = st.session_state.correction_frame
            if st.button(
                f"▶️ Restart from frame {cf}",
                type="primary",
                use_container_width=True,
            ):
                corrected = st.session_state.edited_objects
                if not corrected:
                    main.warning("Draw at least one box before restarting.")
                else:
                    st.session_state.objects = corrected
                    bg.submit(
                        video_path=video_path,
                        objects=corrected,
                        model_uri=model_uri or None,
                        min_confidence=min_conf,
                        start_frame=cf,
                    )
                    st.session_state.correcting = False
                    st.session_state.paused_at_frame = None
                    st.session_state.correction_frame = None
                    st.rerun()
        else:
            run_objs = st.session_state.edited_objects or objects
            if st.button(
                "▶️ Run",
                type="primary",
                disabled=not run_objs or bg.running,
                use_container_width=True,
            ):
                st.session_state.objects = run_objs
                st.session_state.correcting = False
                st.session_state.paused_at_frame = None
                bg.submit(
                    video_path=video_path,
                    objects=run_objs,
                    model_uri=model_uri or None,
                    min_confidence=min_conf,
                    start_frame=0,
                )
                st.rerun()
    with ctrl_pause:
        if st.button("⏸️ Pause", disabled=not bg.running, use_container_width=True):
            bg.pause()
            st.session_state.paused_at_frame = bg.last_completed_frame
            st.session_state.correction_frame = bg.last_completed_frame
            st.session_state.correcting = True
            st.rerun()

    # --- Inference progress bar directly under Run / Pause ---
    if bg.error is not None:
        main.exception(bg.error)

    done, total = bg.progress
    if bg.running:
        frac = min(1.0, done / total) if total else 0.0
        main.progress(frac, text=f"Running… {done}/{total} frames this segment")
        main.caption("Inference runs in the background — Pause to correct boxes.")
    elif bg.last_completed_frame >= 0:
        main.info(f"Completed up to frame {bg.last_completed_frame}.")

    def _run_autoseed() -> None:
        """Run YOLO on frame 0 and append the seeds, with a spinner."""
        with st.spinner("🤖 Auto-detecting objects with YOLO…"):
            try:
                seeds = _yolo_seed_objects(
                    video_path=video_path,
                    model_path=yolo_model,
                    min_confidence=yolo_conf,
                    orig_w=orig_w,
                    orig_h=orig_h,
                    iou_threshold=yolo_iou,
                )
            except Exception as exc:  # noqa: BLE001
                st.exception(exc)
                seeds = []
        if seeds:
            # Replace (not append) so re-detecting doesn't stack duplicate boxes.
            st.session_state.objects = list(seeds)
            st.session_state.canvas_key += 1
            st.success(f"Detected {len(seeds)} YOLO seeds — review and prune.")
        else:
            st.warning("YOLO found no objects on frame 0.")

    if main.button("🤖 Re-detect seeds with YOLO", use_container_width=True):
        _run_autoseed()
        st.rerun()

    # --- Detected-object list in the right-hand side panel ---
    # Class reassignment edits the committed `objects`; in seeding mode the canvas
    # re-prefills from it (showing the new colour). Disabled while correcting,
    # where edits live in the canvas itself.
    with side_panel:
        hdr, clr = st.columns([3, 1])
        hdr.markdown(f"### Detected objects ({len(objects)})")
        if clr.button("🗑️ Clear", use_container_width=True, help="Remove all objects"):
            st.session_state.objects = []
            st.session_state.frames = None
            st.session_state.canvas_key += 1
            st.rerun()
        if not objects:
            st.caption("No objects yet — auto-detect or draw on the frame.")
        active = st.session_state.active_obj

        # Show ball-type detections first; keep each row's original index for
        # the displayed number, edit dropdown and remove button.
        def _ball_first(item: tuple[int, LabelledObject]) -> tuple[int, int]:
            _idx, o = item
            return (0 if "ball" in o.label.lower() else 1, _idx)

        for i, obj in sorted(enumerate(objects), key=_ball_first):
            is_active = i == active
            swatch, c1, c2 = st.columns([0.6, 3.4, 1])
            # Colour swatch matching the box's class; a ▶ marks the active row.
            marker = "▶" if is_active else ""
            swatch.markdown(
                f"<div style='margin-top:6px;white-space:nowrap'>{marker}"
                f"<span style='display:inline-block;width:14px;height:14px;"
                f"border-radius:3px;background:{_rgb_hex(obj.label)};"
                f"color:#000;font-size:10px;font-weight:700;text-align:center;"
                f"line-height:14px;"
                f"outline:{'2px solid #fff' if is_active else 'none'}'>{i}</span>"
                f"</div>",
                unsafe_allow_html=True,
            )
            with c1:
                new_label = st.selectbox(
                    f"obj {i}",
                    DETECTION_CLASSES,
                    index=DETECTION_CLASSES.index(obj.label)
                    if obj.label in DETECTION_CLASSES
                    else 0,
                    key=f"cls_{i}",
                    label_visibility="collapsed",
                )
                if new_label != obj.label and obj.bbox_xyxy_abs is not None:
                    st.session_state.objects[i] = LabelledObject(
                        label=new_label, bbox_xyxy_abs=obj.bbox_xyxy_abs
                    )
                    st.session_state.canvas_key += 1
                    st.rerun()
            if c2.button("✕", key=f"rm_{i}", help="Remove"):
                st.session_state.objects.pop(i)
                st.session_state.canvas_key += 1
                st.rerun()

    # Paused before any frame finished — nothing to correct yet.
    if st.session_state.correcting and paused_frame is not None and not correcting:
        main.warning(
            "Paused before the first frame completed — nothing to correct yet. "
            "Adjust the seeds and hit Run again."
        )

    # Poll for progress while inference is running (keeps the live frame + bar
    # advancing). Skipped when paused so corrections are stable.
    if bg.running:
        import time  # noqa: PLC0415

        time.sleep(1.0)
        st.rerun()

    # --- Step 3: preview + export ---
    if completed:
        st.subheader("Step 3 — Preview & export")
        idx = st.slider("Frame", 0, len(completed) - 1, 0)
        frame_det = completed[idx]

        cap = cv2.VideoCapture(str(video_path))
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame_bgr = cap.read()
        cap.release()
        if ok:
            preview = _draw_boxes_on_array(frame_bgr, frame_det)
            st.image(preview, channels="RGB", width="stretch")
        st.caption(f"{len(frame_det.detections)} detections on frame {idx}")

        default_out = video_path.with_name(f"{video_path.stem}_labels.json")
        out_str = st.text_input("Output JSON path", value=str(default_out))
        if st.button("💾 Save JSON"):
            out_path = export_frames_json(completed, Path(out_str))
            st.success(f"Saved → {out_path}")


def _objects_to_seeds(
    objects: list[LabelledObject],
) -> list[tuple[str, float, float, float, float]]:
    """Convert LabelledObjects (bbox) to (label, x1,y1,x2,y2) abs-pixel tuples."""
    seeds: list[tuple[str, float, float, float, float]] = []
    for o in objects:
        if o.bbox_xyxy_abs is not None:
            x1, y1, x2, y2 = o.bbox_xyxy_abs
            seeds.append((o.label, x1, y1, x2, y2))
    return seeds


def _detections_to_seeds(
    fd, orig_w: int, orig_h: int
) -> list[tuple[str, float, float, float, float]]:
    """Convert a FrameDetections' normalized boxes to abs-pixel seed tuples."""
    seeds: list[tuple[str, float, float, float, float]] = []
    for det in fd.detections:
        x1 = det.x * orig_w
        y1 = det.y * orig_h
        x2 = (det.x + det.w) * orig_w
        y2 = (det.y + det.h) * orig_h
        seeds.append((det.label, x1, y1, x2, y2))
    return seeds


def _seed_boxes_to_canvas_json(
    seeds: list[tuple[str, float, float, float, float]],
    scale: float,
) -> dict:
    """Build a Fabric.js initial_drawing from (label, x1,y1,x2,y2) abs-pixel boxes."""
    objects = []
    for lbl, x1, y1, x2, y2 in seeds:
        objects.append(
            {
                "type": "rect",
                "left": x1 * scale,
                "top": y1 * scale,
                "width": (x2 - x1) * scale,
                "height": (y2 - y1) * scale,
                "fill": "rgba(0,0,0,0.0)",
                "stroke": _rgb_hex(lbl),
                "strokeWidth": 2,
            }
        )
    return {"version": "4.4.0", "objects": objects}


def _hex_to_label() -> dict[str, str]:
    """Reverse map of _rgb_hex(label) -> label for recovering classes from strokes."""
    return {_rgb_hex(lbl): lbl for lbl in DETECTION_CLASSES}


def _render_editable_canvas(
    frame_bgr: np.ndarray,
    seeds: list[tuple[str, float, float, float, float]],
    label: str,
    scale: float,
    orig_h: int,
    key: str,
    drawing_mode: str = "transform",
) -> list[LabelledObject]:
    """Drawable canvas prefilled with *seeds*; returns the edited box objects.

    Each box carries its class as the Fabric stroke colour, so on read we recover
    the original label even after move/resize. New boxes (drawn in "rect" mode)
    take the active sidebar *label*. ``drawing_mode`` is "transform" (select /
    move / resize / delete existing) or "rect" (draw new). Used for both initial
    seeding and paused-frame correction so editing behaves identically.
    """
    disp_h = int(orig_h * scale)
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    disp = cv2.resize(frame_rgb, (CANVAS_DISPLAY_WIDTH, disp_h))

    # Bake each seed's index number near its top-left corner so the numbers line
    # up with the boxes and match the right-hand list, without being editable
    # canvas objects themselves.
    for i, (lbl, x1, y1, _x2, _y2) in enumerate(seeds):
        px, py = int(x1 * scale), int(y1 * scale)
        r, g, b = (int(c) for c in bytes.fromhex(_rgb_hex(lbl)[1:]))
        cv2.putText(
            disp,
            str(i),
            (px + 2, max(14, py - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 0, 0),
            3,
            cv2.LINE_AA,
        )
        cv2.putText(
            disp,
            str(i),
            (px + 2, max(14, py - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (r, g, b),
            1,
            cv2.LINE_AA,
        )
    bg_img = Image.fromarray(disp)

    result = st_canvas(
        fill_color="rgba(0,0,0,0.0)",
        stroke_width=3,
        stroke_color=_rgb_hex(label),
        background_image=bg_img,
        update_streamlit=True,
        height=disp_h,
        width=CANVAS_DISPLAY_WIDTH,
        drawing_mode=drawing_mode,
        initial_drawing=_seed_boxes_to_canvas_json(seeds, scale),
        key=key,
    )

    json_data = result.json_data if result is not None else None
    objs: list[LabelledObject] = []
    if not json_data:
        return objs
    hex_to_label = _hex_to_label()
    for obj in json_data.get("objects", []):
        if obj.get("type") != "rect":
            continue
        left = float(obj["left"]) * obj.get("scaleX", 1.0)
        top = float(obj["top"]) * obj.get("scaleY", 1.0)
        width = float(obj["width"]) * obj.get("scaleX", 1.0)
        height = float(obj["height"]) * obj.get("scaleY", 1.0)
        x1, y1 = left / scale, top / scale
        x2, y2 = (left + width) / scale, (top + height) / scale
        # Recover the class from the stroke colour; fall back to the active label
        # for freshly drawn boxes whose stroke matches the current selection.
        stroke = (obj.get("stroke") or "").lower()
        box_label = hex_to_label.get(stroke, label)
        objs.append(LabelledObject(label=box_label, bbox_xyxy_abs=(x1, y1, x2, y2)))
    return objs


def _draw_boxes_on_array(frame_bgr: np.ndarray, frame_det) -> np.ndarray:
    """Draw normalized detections onto a BGR frame; return an RGB array."""
    img = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]
    for det in frame_det.detections:
        x1 = int(det.x * w)
        y1 = int(det.y * h)
        x2 = int((det.x + det.w) * w)
        y2 = int((det.y + det.h) * h)
        color = color_map.get(det.label.lower(), (255, 0, 0))
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            img,
            det.label,
            (x1, max(0, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
        )
    return img


if __name__ == "__main__":
    main()
