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
    _default_model_uri,
    _warmup_done,  # noqa: F401 — imported for future sidebar status use
    export_frames_json,
    extract_first_frame,
)
from footy_track.schema import DETECTION_CLASSES, ObjectDetection

CANVAS_DISPLAY_WIDTH = 800


def _strip_wrapping_quotes(s: str) -> str:
    """Drop a single pair of surrounding quotes (drag-and-drop / shell paste).

    Keeps quotes that are genuinely inside the path.
    """
    s = s.strip()
    if len(s) >= 2 and s[0] == s[-1] in ("'", '"'):
        return s[1:-1]
    return s


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
    st.session_state.setdefault("correcting", False)
    # Frame currently shown in the paused editor (prev/next navigation).
    st.session_state.setdefault("correction_frame", None)  # int | None
    # Edited boxes captured from the editable canvas this run.
    st.session_state.setdefault("edited_objects", [])
    # Per-frame edit cache {frame_idx: [LabelledObject]} so corrections made on a
    # frame survive scrubbing away and back (instead of reverting to the
    # propagated detections).
    # Previous canvas box centers + the index of the most-recently-moved box,
    # used to highlight the matching row in the side panel (selection heuristic).
    st.session_state.setdefault("prev_centers", [])
    st.session_state.setdefault("active_obj", None)  # int | None
    # Path of the video we've already auto-seeded, so it only runs once per video.
    st.session_state.setdefault("autoseeded_video", None)
    # Path of the currently loaded video; changing it wipes prior results.
    st.session_state.setdefault("loaded_video", None)
    # (frame_idx, objects) submitted to a run, shown during compile before the
    # first new frame arrives.
    st.session_state.setdefault("compile_preview", None)


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
    frame_idx: int = 0,
) -> list[LabelledObject]:
    """Run the YOLO detector on a frame and return NMS-filtered seed objects.

    Detections come back normalized; we convert to absolute xyxy pixel coords for
    :class:`LabelledObject`, the seed format SAM3 expects. ``frame_idx`` selects
    which frame to detect on (0 = first frame).
    """
    import tempfile  # noqa: PLC0415

    if frame_idx <= 0:
        frame_bgr = extract_first_frame(video_path)
    else:
        cap = cv2.VideoCapture(str(video_path))
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame_bgr = cap.read()
        cap.release()
        if not ok:
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
        yolo_model = _strip_wrapping_quotes(
            st.text_input(
                "YOLO checkpoint (blank = current best)",
                value="",
                help=f"Default: {CURRENT_BEST_DETECTOR_CHECKPOINT}",
            )
        )
        yolo_conf = st.slider("YOLO confidence", 0.0, 1.0, 0.35, 0.05)
        yolo_iou = st.slider("YOLO NMS IoU", 0.1, 0.9, 0.5, 0.05)

        st.header("3. Model")
        model_uri = _strip_wrapping_quotes(
            st.text_input("SAM3 checkpoint (blank = default)", value="")
        )
        min_conf = st.slider(
            "Min confidence",
            0.0,
            1.0,
            0.25,
            0.05,
            help="SAM3 detection threshold; raise it to drop weak/snapped boxes.",
        )
        imgsz = st.select_slider(
            "Inference resolution (imgsz)",
            options=[512, 640, 768, 1024],
            value=512,
            help="Higher detects the small ball more reliably (slower/more memory).",
        )

        st.header("4. Display")
        show_boxes = st.checkbox(
            "Show boxes",
            value=True,
            help="Toggle the detection boxes on the live / preview frames.",
        )
        auto_pause_anomaly = st.checkbox(
            "Auto-pause on track anomaly",
            value=True,
            help="Stop and pause when a track box jumps or resizes implausibly "
            "(e.g. the ball snapping to a marking) so you can correct it.",
        )
        st.session_state.bg.anomaly_detection = auto_pause_anomaly
        st.divider()
        st.caption(
            "JIT kernels cached in ~/.cache/torch_inductor_sam3 — fast after first run."
        )

    # Active drawing class — the widget lives next to the Tool selector in the
    # main window; default it here so earlier code (auto-seed, canvas) can read it.
    label = st.session_state.get("draw_class", DETECTION_CLASSES[0])

    if not video_path_str:
        st.info("Enter a video path in the sidebar to begin.")
        return

    video_path_str = _strip_wrapping_quotes(video_path_str)
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

    # Loading a different video wipes the previous run's results and edit state
    # so stale frames/boxes from the old clip don't bleed through.
    if st.session_state.get("loaded_video") != str(video_path):
        bg.pause()
        st.session_state.bg = BackgroundLabeller()
        bg = st.session_state.bg
        st.session_state.objects = []
        st.session_state.edited_objects = []
        st.session_state.frames = None
        st.session_state.correcting = False
        st.session_state.correction_frame = None
        st.session_state.active_obj = None
        st.session_state.prev_centers = []
        st.session_state.compile_preview = None
        st.session_state.autoseeded_video = None
        st.session_state.canvas_key += 1
        st.session_state.loaded_video = str(video_path)

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

    # ------------------------------------------------------------------
    # Unified state model
    # ------------------------------------------------------------------
    # view_mode is derived ONCE and drives every section below:
    #   "running"  – SAM3 is propagating; the window + list are read-only.
    #   "paused"   – propagation stopped; edit the scrubbed frame, then Restart.
    #   "seeding"  – no run yet; edit frame-0 seeds, then Run.
    #
    # `objects` is the single editable working set. It means frame-0 seeds in
    # seeding mode and the scrubbed frame's boxes in paused mode; both are loaded
    # into the SAME list so the canvas, side-panel list and class edits stay in
    # sync. `edited_objects` is the live canvas readback for the current pass.
    completed = bg.completed_frames()

    # Auto-pause on anomaly: if the run stopped because a track box jumped/snapped
    # implausibly, drop into paused mode at that frame so the user can correct it.
    if (
        not bg.running
        and bg.anomaly_frame is not None
        and not st.session_state.correcting
    ):
        n = bg.anomaly_frame
        bg.anomaly_frame = None
        st.session_state.correcting = True
        st.session_state.correction_frame = n
        if 0 <= n < len(completed):
            st.session_state.objects = _frame_dets_to_objects(
                completed[n], orig_w, orig_h
            )
        st.session_state.canvas_key += 1
        st.session_state["anomaly_notice"] = n
        st.rerun()

    if bg.running:
        view_mode = "running"
    elif st.session_state.correcting and st.session_state.correction_frame is not None:
        view_mode = "paused"
    else:
        view_mode = "seeding"

    objects: list[LabelledObject] = st.session_state.objects
    cf = st.session_state.correction_frame  # current paused frame (paused mode)
    main_col, side_col = st.columns([3, 1])
    # Fresh placeholder each run so the previous render's widgets are fully
    # cleared — prevents ghost/doubled rows when switching running<->editable.
    side_panel = side_col.empty().container()
    main = main_col.container()
    st.session_state.edited_objects = []

    def _frame_rgb(frame_bgr: np.ndarray, frame_det) -> np.ndarray:
        """Frame as RGB with boxes drawn, or plain RGB when boxes are hidden."""
        if show_boxes:
            return _draw_boxes_on_array(frame_bgr, frame_det)
        return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

    def _load_frame_objects(frame_idx: int) -> None:
        """Load a frame's propagated detections into the editable working set."""
        if 0 <= frame_idx < len(completed):
            st.session_state.objects = _frame_dets_to_objects(
                completed[frame_idx], orig_w, orig_h
            )
        else:
            st.session_state.objects = []
        st.session_state.canvas_key += 1

    def _run_autoseed() -> None:
        """Run YOLO on the current editable frame and replace the working set."""
        # Detect on the paused frame when correcting, else frame 0 when seeding.
        det_frame = (
            st.session_state.correction_frame
            if st.session_state.correcting
            and st.session_state.correction_frame is not None
            else 0
        )
        with st.spinner("🤖 Auto-detecting objects with YOLO…"):
            try:
                seeds = _yolo_seed_objects(
                    video_path=video_path,
                    model_path=yolo_model,
                    min_confidence=yolo_conf,
                    orig_w=orig_w,
                    orig_h=orig_h,
                    iou_threshold=yolo_iou,
                    frame_idx=det_frame,
                )
            except Exception as exc:  # noqa: BLE001
                st.exception(exc)
                seeds = []
        if seeds:
            st.session_state.objects = list(seeds)
            st.session_state.canvas_key += 1
            st.success(f"Detected {len(seeds)} YOLO seeds — review and prune.")
        else:
            st.warning("YOLO found no objects on frame 0.")

    # ------------------------------------------------------------------
    # Main window
    # ------------------------------------------------------------------
    if view_mode == "running":
        # Show the compile preview (the just-submitted frame + edits) until the
        # new run actually advances past its start frame. progress = (done, total)
        # where done == start_frame right after submit.
        preview = st.session_state.get("compile_preview")
        done_now = bg.progress[0]
        show_preview = preview is not None and done_now <= preview[0]
        latest_idx = -1 if show_preview else (len(completed) - 1 if completed else -1)
        if latest_idx >= 0:
            cap = cv2.VideoCapture(str(video_path))
            cap.set(cv2.CAP_PROP_POS_FRAMES, latest_idx)
            ok, frame_bgr = cap.read()
            cap.release()
            if ok:
                main.image(
                    _frame_rgb(frame_bgr, completed[latest_idx]),
                    channels="RGB",
                    width=CANVAS_DISPLAY_WIDTH,
                )
            main.caption(
                f"🔴 Live — frame {latest_idx} "
                f"({len(completed[latest_idx].detections)} objects)"
            )
            # New run has advanced — drop the compile preview.
            st.session_state.compile_preview = None
        else:
            # Compiling: no NEW frame yet. Show the frame we just submitted from
            # (frame 0 on Run, the edited frame on Restart) with the edits, so
            # the window stays populated and reflects the just-made corrections.
            preview = st.session_state.get("compile_preview")
            pv_idx, pv_objs = preview if preview else (0, objects)
            cap = cv2.VideoCapture(str(video_path))
            cap.set(cv2.CAP_PROP_POS_FRAMES, pv_idx)
            ok, pv_bgr = cap.read()
            cap.release()
            if not ok:
                pv_bgr = extract_first_frame(video_path)
            from types import SimpleNamespace  # noqa: PLC0415

            pv_fd = SimpleNamespace(
                detections=[
                    ObjectDetection(
                        label=o.label,
                        confidence=1.0,
                        x=o.bbox_xyxy_abs[0] / orig_w,
                        y=o.bbox_xyxy_abs[1] / orig_h,
                        w=(o.bbox_xyxy_abs[2] - o.bbox_xyxy_abs[0]) / orig_w,
                        h=(o.bbox_xyxy_abs[3] - o.bbox_xyxy_abs[1]) / orig_h,
                    )
                    for o in pv_objs
                    if o.bbox_xyxy_abs is not None
                ]
            )
            main.image(
                _frame_rgb(pv_bgr, pv_fd),
                channels="RGB",
                width=CANVAS_DISPLAY_WIDTH,
            )
            main.caption(f"⏳ Compiling model… (showing frame {pv_idx} with edits)")
    else:
        # Both editable modes share one canvas + tool selector.
        is_paused = view_mode == "paused"
        if is_paused and not (0 <= cf < len(completed)):
            cf = max(0, len(completed) - 1)
            st.session_state.correction_frame = cf

        tool_col, cls_col, auto_col = main.columns([2.4, 2, 1.2])
        tool = tool_col.radio(
            "Tool",
            ["✋ Edit", "✏️ Draw"],
            horizontal=True,
            key="tool_mode",
            label_visibility="collapsed",
        )
        draw_mode = "transform" if tool.startswith("✋") else "rect"
        # Class for newly drawn boxes — sits next to the Draw tool.
        label = cls_col.selectbox(
            "Class to draw",
            DETECTION_CLASSES,
            index=DETECTION_CLASSES.index(label) if label in DETECTION_CLASSES else 0,
            key="draw_class",
            label_visibility="collapsed",
        )
        # Small auto-detect button — available in both seeding and paused modes.
        if auto_col.button("🤖 Auto-detect", help="Re-run YOLO on this frame"):
            _run_autoseed()
            st.rerun()

        if is_paused:
            cap = cv2.VideoCapture(str(video_path))
            cap.set(cv2.CAP_PROP_POS_FRAMES, cf)
            ok, frame_bgr = cap.read()
            cap.release()
            if not ok:
                frame_bgr = extract_first_frame(video_path)
            canvas_key = f"edit_{cf}_{st.session_state.canvas_key}_{draw_mode}"
        else:
            frame_bgr = extract_first_frame(video_path)
            canvas_key = f"seed_{st.session_state.canvas_key}_{draw_mode}"

        with main:
            st.session_state.edited_objects = _render_editable_canvas(
                frame_bgr,
                _objects_to_seeds(objects),
                label,
                scale,
                orig_h,
                key=canvas_key,
                drawing_mode=draw_mode,
            )
        main.caption(
            f"**✋ Edit**: select to move/resize (Delete removes). "
            f"**✏️ Draw**: drag a **{label}** box."
        )

        # Add-only sync: a newly drawn box appears in the list immediately.
        # Never let a stale/empty readback REDUCE objects (would wipe seeds).
        edited = st.session_state.edited_objects
        if len(edited) > len(objects):
            st.session_state.objects = list(edited)
            st.session_state.canvas_key += 1
            st.rerun()

        # Selection heuristic: highlight the row whose box just moved most.
        cur_centers = [
            (
                (o.bbox_xyxy_abs[0] + o.bbox_xyxy_abs[2]) / 2,
                (o.bbox_xyxy_abs[1] + o.bbox_xyxy_abs[3]) / 2,
            )
            for o in edited
            if o.bbox_xyxy_abs is not None
        ]
        prev = st.session_state.prev_centers
        if cur_centers and len(cur_centers) == len(prev):
            deltas = [
                (cx - px) ** 2 + (cy - py) ** 2
                for (cx, cy), (px, py) in zip(cur_centers, prev, strict=False)
            ]
            mx = max(range(len(deltas)), key=lambda i: deltas[i])
            if deltas[mx] > 4.0:
                st.session_state.active_obj = mx
        st.session_state.prev_centers = cur_centers

    # ------------------------------------------------------------------
    # Frame navigation (paused only): slider + prev/next
    # ------------------------------------------------------------------
    if view_mode == "paused":
        notice_frame = st.session_state.pop("anomaly_notice", None)
        if notice_frame is not None:
            main.warning(
                f"⚠️ Auto-paused at frame {notice_frame}: a track box jumped or "
                "changed size implausibly (likely a lost/snapped object). Correct "
                "the boxes — scroll back if needed — then Restart."
            )
        last = len(completed) - 1
        if last > 0:
            slid = main.slider("Frame", 0, last, cf, key=f"scrub_{cf}")
            if slid != cf:
                st.session_state.correction_frame = slid
                _load_frame_objects(slid)
                st.rerun()
        nav_prev, nav_lbl, nav_next = main.columns([1, 2, 1])
        with nav_prev:
            if st.button("⬅️ Prev", disabled=cf <= 0, use_container_width=True):
                st.session_state.correction_frame = cf - 1
                _load_frame_objects(cf - 1)
                st.rerun()
        with nav_lbl:
            st.markdown(
                f"<div style='text-align:center'>Frame <b>{cf}</b> / {last}</div>",
                unsafe_allow_html=True,
            )
        with nav_next:
            if st.button("Next ➡️", disabled=cf >= last, use_container_width=True):
                st.session_state.correction_frame = cf + 1
                _load_frame_objects(cf + 1)
                st.rerun()

    # ------------------------------------------------------------------
    # Playback controls (Run / Restart / Pause)
    # ------------------------------------------------------------------
    ctrl_run, ctrl_pause = main.columns(2)
    with ctrl_run:
        if view_mode == "paused":
            if st.button(
                f"▶️ Restart from frame {cf}", type="primary", use_container_width=True
            ):
                corrected = st.session_state.edited_objects or objects
                if not corrected:
                    main.warning("Draw at least one box before restarting.")
                else:
                    st.session_state.objects = list(corrected)
                    # Remember the just-edited frame so we can show it (with the
                    # edits) while the model recompiles.
                    st.session_state.compile_preview = (cf, list(corrected))
                    bg.submit(
                        video_path=video_path,
                        objects=corrected,
                        model_uri=model_uri or None,
                        min_confidence=min_conf,
                        start_frame=cf,
                        imgsz=imgsz,
                    )
                    st.session_state.correcting = False
                    st.session_state.correction_frame = None
                    st.rerun()
        elif view_mode == "seeding":
            run_objs = st.session_state.edited_objects or objects
            if st.button(
                "▶️ Run", type="primary", disabled=not run_objs, use_container_width=True
            ):
                st.session_state.objects = list(run_objs)
                st.session_state.compile_preview = (0, list(run_objs))
                bg.submit(
                    video_path=video_path,
                    objects=run_objs,
                    model_uri=model_uri or None,
                    min_confidence=min_conf,
                    start_frame=0,
                    imgsz=imgsz,
                )
                st.rerun()
        else:  # running
            st.button("▶️ Run", type="primary", disabled=True, use_container_width=True)
    with ctrl_pause:
        if st.button(
            "⏸️ Pause", disabled=view_mode != "running", use_container_width=True
        ):
            bg.pause()
            n = bg.last_completed_frame
            st.session_state.correction_frame = n
            st.session_state.correcting = True
            _load_frame_objects(n)
            st.rerun()

    # --- Inference progress bar directly under Run / Pause ---
    if bg.error is not None:
        main.exception(bg.error)
    done, total = bg.progress
    if view_mode == "running":
        if not completed:
            main.progress(0.0, text="⏳ Compiling model…")
        else:
            frac = min(1.0, done / total) if total else 0.0
            main.progress(frac, text=f"Running… {done}/{total} frames")
        model_name = Path(model_uri or _default_model_uri()).name
        main.caption(
            f"Inference runs in the background ({model_name}) — Pause to correct boxes."
        )
    elif bg.last_completed_frame >= 0:
        main.info(f"Completed up to frame {bg.last_completed_frame}.")

    # ------------------------------------------------------------------
    # Side panel: one list, chosen by view_mode
    # ------------------------------------------------------------------
    def _ball_first(item: tuple[int, LabelledObject]) -> tuple[int, int]:
        _idx, o = item
        return (0 if "ball" in o.label.lower() else 1, _idx)

    with side_panel:
        if view_mode == "running":
            live = completed[-1] if completed else None
            live_objs = _frame_dets_to_objects(live, orig_w, orig_h) if live else []
            st.markdown(f"### Detected objects ({len(live_objs)})")
            st.caption("🔴 Live — read-only while running. Pause to edit.")
            rows = "".join(
                f"<div style='display:flex;align-items:center;gap:8px;padding:2px 0'>"
                f"<span style='flex:none;width:16px;height:16px;border-radius:3px;"
                f"background:{_rgb_hex(obj.label)};color:#000;font-size:10px;"
                f"font-weight:700;text-align:center;line-height:16px'>{i}</span>"
                f"<span>{obj.label}</span></div>"
                for i, obj in sorted(enumerate(live_objs), key=_ball_first)
            )
            st.markdown(rows, unsafe_allow_html=True)
        else:
            hdr, clr = st.columns([3, 1])
            hdr.markdown(f"### Detected objects ({len(objects)})")
            if clr.button("🗑️ Clear", use_container_width=True, help="Remove all"):
                st.session_state.objects = []
                st.session_state.canvas_key += 1
                st.rerun()
            if not objects:
                st.caption("No objects yet — auto-detect or draw on the frame.")
            active = st.session_state.active_obj
            # Widget keys namespaced by canvas_key so rows from a previous mount
            # can't linger as ghost widgets when the object set changes.
            ck = st.session_state.canvas_key
            for i, obj in sorted(enumerate(objects), key=_ball_first):
                is_active = i == active
                swatch, c1, c2 = st.columns([0.6, 3.4, 1])
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
                        key=f"cls_{ck}_{i}",
                        label_visibility="collapsed",
                    )
                    if new_label != obj.label and obj.bbox_xyxy_abs is not None:
                        st.session_state.objects[i] = LabelledObject(
                            label=new_label, bbox_xyxy_abs=obj.bbox_xyxy_abs
                        )
                        st.session_state.canvas_key += 1
                        st.rerun()
                if c2.button("✕", key=f"rm_{ck}_{i}", help="Remove"):
                    st.session_state.objects.pop(i)
                    st.session_state.canvas_key += 1
                    st.rerun()

    # Poll for progress while running (advances the live frame + bar).
    if view_mode == "running":
        import time  # noqa: PLC0415

        time.sleep(1.0)
        st.rerun()

    # --- Step 3: preview + export (seeding mode only, once frames exist) ---
    if completed and view_mode == "seeding":
        st.subheader("Step 3 — Preview & export")
        idx = st.slider("Preview frame", 0, len(completed) - 1, 0)
        frame_det = completed[idx]
        cap = cv2.VideoCapture(str(video_path))
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame_bgr = cap.read()
        cap.release()
        if ok:
            st.image(
                _frame_rgb(frame_bgr, frame_det),
                channels="RGB",
                width=CANVAS_DISPLAY_WIDTH,
            )
        st.caption(f"{len(frame_det.detections)} detections on frame {idx}")
        default_out = video_path.with_name(f"{video_path.stem}_labels.json")
        out_str = st.text_input("Output JSON path", value=str(default_out))
        if st.button("💾 Save JSON"):
            out_path = export_frames_json(completed, Path(out_str))
            st.success(f"Saved → {out_path}")


def _frame_dets_to_objects(fd, orig_w: int, orig_h: int) -> list[LabelledObject]:
    """Convert a FrameDetections' normalized boxes to editable LabelledObjects."""
    return [
        LabelledObject(
            label=d.label,
            bbox_xyxy_abs=(
                d.x * orig_w,
                d.y * orig_h,
                (d.x + d.w) * orig_w,
                (d.y + d.h) * orig_h,
            ),
        )
        for d in fd.detections
    ]


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
