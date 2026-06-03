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

import threading
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
from PIL import Image

# Must be imported before streamlit_drawable_canvas: restores image_to_url on
# Streamlit >= 1.40 so the canvas component works.
import footy_track.labeller._canvas_compat  # noqa: F401  # isort: skip
from streamlit_drawable_canvas import st_canvas  # isort: skip

from footy_track.detectors.utils import color_map
from footy_track.labeller.video_utils import (
    LabelledObject,
    Sam3VideoLabeller,
    _warmup_done,  # noqa: F401 — imported for future sidebar status use
    export_frames_json,
    extract_first_frame,
)
from footy_track.schema import DETECTION_CLASSES, ObjectDetection

CANVAS_DISPLAY_WIDTH = 960


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
    st.session_state.setdefault("run_frames", [])  # frames collected so far in current run
    st.session_state.setdefault("run_total", 0)    # total frames expected
    st.session_state.setdefault("run_error", None) # exception if run failed
    st.session_state.setdefault("running", False)  # True while thread is active
    st.session_state.setdefault("stop_flag", False)  # set True to request stop
    # JIT cache (~/.cache/torch_inductor_sam3) persists across restarts instead of warmup.


def _canvas_rects_to_objects(
    json_data: dict | None, label: str, scale: float
) -> list[LabelledObject]:
    """Convert drawable-canvas rect objects (display coords) to LabelledObjects."""
    objects: list[LabelledObject] = []
    if not json_data:
        return objects
    for obj in json_data.get("objects", []):
        if obj.get("type") != "rect":
            continue
        left = float(obj["left"]) * obj.get("scaleX", 1.0)
        top = float(obj["top"]) * obj.get("scaleY", 1.0)
        width = float(obj["width"]) * obj.get("scaleX", 1.0)
        height = float(obj["height"]) * obj.get("scaleY", 1.0)
        # Map from display coords back to original-frame pixels.
        x1, y1 = left / scale, top / scale
        x2, y2 = (left + width) / scale, (top + height) / scale
        objects.append(LabelledObject(label=label, bbox_xyxy_abs=(x1, y1, x2, y2)))
    return objects


def main() -> None:  # noqa: PLR0912, PLR0915
    st.set_page_config(page_title="SAM3 Video Labeller", layout="wide")
    _init_state()

    st.title("⚽ SAM3 Video Labeller")
    st.caption(
        "Mark players, ball and officials on the first frame; SAM3 propagates "
        "them through the clip."
    )

    # --- Sidebar: video + class selection ---
    with st.sidebar:
        st.header("1. Video")
        video_path_str = st.text_input("Video path", value="")
        st.header("2. Class")
        label = st.selectbox("Class to draw", DETECTION_CLASSES, index=0)
        st.markdown(
            f"<div style='width:100%;height:18px;background:{_rgb_hex(label)};"
            "border-radius:4px'></div>",
            unsafe_allow_html=True,
        )
        st.header("3. Model")
        model_uri = st.text_input("SAM3 checkpoint (blank = default)", value="")
        min_conf = st.slider("Min confidence", 0.0, 1.0, 0.25, 0.05)
        st.divider()
        st.caption("JIT kernels cached in ~/.cache/torch_inductor_sam3 — fast after first run.")

    if not video_path_str:
        st.info("Enter a video path in the sidebar to begin.")
        return

    video_path = Path(video_path_str).expanduser()
    if not video_path.exists():
        st.error(f"Video not found: {video_path}")
        return

    try:
        frame_pil, orig_w, orig_h = _load_first_frame_pil(video_path)
    except Exception as exc:  # noqa: BLE001
        st.error(f"Could not read first frame: {exc}")
        return

    scale = CANVAS_DISPLAY_WIDTH / orig_w
    disp_h = int(orig_h * scale)
    disp_frame = frame_pil.resize((CANVAS_DISPLAY_WIDTH, disp_h))

    # If objects are already committed, annotate the background so the canvas
    # shows them while the user draws more. Built here so it's available before
    # the canvas widget is rendered.
    objects_so_far: list[LabelledObject] = st.session_state.objects
    if objects_so_far:
        annotated_bgr = _draw_committed_on_frame(
            cv2.cvtColor(np.array(disp_frame), cv2.COLOR_RGB2BGR),
            objects_so_far, orig_w, orig_h,
        )
        canvas_bg = Image.fromarray(cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB))
    else:
        canvas_bg = disp_frame

    st.subheader("Step 1 — Annotate frame 0")
    prompt_mode = st.radio(
        "Prompt mode",
        ["Box", "Point"],
        horizontal=True,
        help="Box: drag a rectangle. Point: click the centre of the object.",
    )
    if prompt_mode == "Box":
        st.write(f"Draw **{label}** rectangles, then click **Add**.")
        text_hint = None
    else:
        text_hint = st.text_input(
            "Text hint (optional)",
            placeholder='e.g. "soccer ball" — helps SAM3 find the right object',
            help="If set, SAM3Semantic uses this text on frame 0 to find the object nearest your click.",
        ) or None
        st.write(f"Click the centre of each **{label}**, then click **Add**.")

    canvas_mode = "rect" if prompt_mode == "Box" else "point"
    canvas_result = st_canvas(
        fill_color=_rgb_hex(label) + "40",  # translucent fill for points
        stroke_width=3 if prompt_mode == "Box" else 4,
        stroke_color=_rgb_hex(label),
        background_image=canvas_bg,
        update_streamlit=True,
        height=disp_h,
        width=CANVAS_DISPLAY_WIDTH,
        drawing_mode=canvas_mode,
        key=f"canvas_{st.session_state.canvas_key}",
    )

    col_add, col_clear = st.columns(2)
    with col_add:
        btn_label = f"➕ Add {'boxes' if prompt_mode == 'Box' else 'points'} as '{label}'"
        if st.button(btn_label, use_container_width=True):
            if prompt_mode == "Box":
                new_objs = _canvas_rects_to_objects(canvas_result.json_data, label, scale)
            else:
                new_objs = _canvas_points_to_objects(canvas_result.json_data, label, scale, text_hint)
            if new_objs:
                st.session_state.objects.extend(new_objs)
                st.session_state.canvas_key += 1
                st.rerun()
            else:
                st.warning(f"No {'rectangles' if prompt_mode == 'Box' else 'points'} drawn.")
    with col_clear:
        if st.button("🗑️ Clear all objects", use_container_width=True):
            st.session_state.objects = []
            st.session_state.frames = None
            st.session_state.canvas_key += 1
            st.rerun()

    # --- Object list + confirmation overlay ---
    objects: list[LabelledObject] = st.session_state.objects
    if objects:
        st.markdown("**Marked objects:**")
        for i, obj in enumerate(objects):
            c1, c2 = st.columns([5, 1])
            if obj.bbox_xyxy_abs is not None:
                x1, y1, x2, y2 = (round(v) for v in obj.bbox_xyxy_abs)
                c1.write(f"`{i}` **{obj.label}** box — ({x1}, {y1}) → ({x2}, {y2})")
            else:
                px, py = (round(v) for v in obj.point_xy_abs)  # type: ignore[misc]
                hint = f" + text: *{obj.text_hint}*" if obj.text_hint else ""
                c1.write(f"`{i}` **{obj.label}** point — ({px}, {py}){hint}")
            if c2.button("remove", key=f"rm_{i}"):
                st.session_state.objects.pop(i)
                st.rerun()

        # (Committed objects are drawn directly onto the canvas background above.)

    # --- Step 2: run propagation ---
    st.subheader("Step 2 — Propagate through clip")

    running = st.session_state.running
    col_run, col_stop = st.columns([3, 1])
    with col_run:
        clicked = st.button("▶️ Label video", type="primary", disabled=not objects or running)
        if clicked and not st.session_state.running:  # guard double-click
            st.session_state.run_frames = []
            st.session_state.run_total = 0
            st.session_state.run_error = None
            st.session_state.running = True
            st.session_state.stop_flag = False
            st.session_state.frames = None
            running = True  # update local var so spinner shows immediately

            labeller = Sam3VideoLabeller(
                video_path=video_path,
                objects=objects,
                model_uri=model_uri or None,
                min_confidence=min_conf,
            )
            cap_count = cv2.VideoCapture(str(video_path))
            st.session_state.run_total = int(cap_count.get(cv2.CAP_PROP_FRAME_COUNT))
            cap_count.release()

            def _run_thread(lab: Sam3VideoLabeller) -> None:
                try:
                    for frame_det in lab.iter_frames():
                        if st.session_state.stop_flag:
                            break
                        st.session_state.run_frames.append(frame_det)
                    if not st.session_state.stop_flag:
                        st.session_state.frames = list(st.session_state.run_frames)
                except Exception as exc:  # noqa: BLE001
                    st.session_state.run_error = exc
                finally:
                    st.session_state.running = False

            threading.Thread(target=_run_thread, args=(labeller,), daemon=True).start()
            st.rerun()

    with col_stop:
        if st.button("⏹ Stop", disabled=not running):
            st.session_state.stop_flag = True
            st.session_state.frames = list(st.session_state.run_frames)

    if running or st.session_state.run_frames:
        done = len(st.session_state.run_frames)
        total = st.session_state.run_total or 0
        if running and total == 0:
            st.progress(0.0, text="⏳ Loading model…")
        else:
            frac = min(1.0, done / total) if total else 1.0
            label = f"Frame {done}/{total}" if running else f"Done — {done} frames"
            st.progress(frac, text=label)

        if st.session_state.run_error:
            st.exception(st.session_state.run_error)
        elif done > 0:
            # Live preview — show the latest processed frame
            latest = st.session_state.run_frames[-1]
            cap = cv2.VideoCapture(str(video_path))
            cap.set(cv2.CAP_PROP_POS_FRAMES, done - 1)
            ok, frame_bgr = cap.read()
            cap.release()
            if ok:
                st.image(_draw_boxes_on_array(frame_bgr, latest), channels="RGB", width="stretch")
            st.caption(f"Latest: frame {done - 1} — {len(latest.detections)} detections")

        if running:
            st.rerun()
        elif not st.session_state.run_error and done > 0:
            st.success(f"Labelled {done} frames.")

    # --- Step 3: preview + export ---
    frames = st.session_state.frames
    if frames:
        st.subheader("Step 3 — Preview & export")
        idx = st.slider("Frame", 0, len(frames) - 1, 0)
        frame_det = frames[idx]

        # Re-render the chosen frame from the video and draw boxes on it.
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
            out_path = export_frames_json(frames, Path(out_str))
            st.success(f"Saved → {out_path}")


def _draw_committed_on_frame(
    frame_bgr: np.ndarray,
    objects: list[LabelledObject],
    orig_w: int,
    orig_h: int,
) -> np.ndarray:
    """Draw committed objects onto a display-sized BGR frame in-place (copy returned)."""
    img = frame_bgr.copy()
    h, w = img.shape[:2]
    scale_x, scale_y = w / orig_w, h / orig_h
    overlay = img.copy()

    for obj in objects:
        color = color_map.get(obj.label.lower(), (255, 0, 0))
        if obj.bbox_xyxy_abs is not None:
            x1, y1, x2, y2 = obj.bbox_xyxy_abs
            cv2.rectangle(
                img,
                (int(x1 * scale_x), int(y1 * scale_y)),
                (int(x2 * scale_x), int(y2 * scale_y)),
                color, 2,
            )
            cv2.putText(img, obj.label, (int(x1 * scale_x), max(0, int(y1 * scale_y) - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        elif obj.point_xy_abs is not None:
            px, py = int(obj.point_xy_abs[0] * scale_x), int(obj.point_xy_abs[1] * scale_y)
            # Semi-transparent filled circle on overlay, then blend
            cv2.circle(overlay, (px, py), 6, color, -1)
            cv2.circle(img, (px, py), 6, color, 2)  # solid outline on original
            cv2.putText(img, obj.label, (px + 8, py + 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    # Blend overlay (filled circles) at 40% opacity
    cv2.addWeighted(overlay, 0.4, img, 0.6, 0, img)
    return img


def _canvas_points_to_objects(
    json_data: dict | None, label: str, scale: float, text_hint: str | None = None
) -> list[LabelledObject]:
    """Convert drawable-canvas circle/point objects (display coords) to LabelledObjects."""
    objects: list[LabelledObject] = []
    if not json_data:
        return objects
    for obj in json_data.get("objects", []):
        if obj.get("type") not in ("circle", "point"):
            continue
        # Fabric.js circle uses originX/originY="center" so left/top IS the centre.
        cx = float(obj.get("left", 0))
        cy = float(obj.get("top", 0))
        objects.append(LabelledObject(
            label=label,
            point_xy_abs=(cx / scale, cy / scale),
            text_hint=text_hint,
        ))
    return objects


def _fake_frame_det(objects: list[LabelledObject], orig_w: int, orig_h: int, scale: float):
    """Build a SimpleNamespace with detections for the confirmation overlay."""
    from types import SimpleNamespace  # noqa: PLC0415

    detections = []
    for obj in objects:
        if obj.bbox_xyxy_abs is not None:
            x1, y1, x2, y2 = obj.bbox_xyxy_abs
            detections.append(
                ObjectDetection(
                    label=obj.label,
                    confidence=1.0,
                    x=max(0.0, x1 / orig_w),
                    y=max(0.0, y1 / orig_h),
                    w=max(0.0, (x2 - x1) / orig_w),
                    h=max(0.0, (y2 - y1) / orig_h),
                )
            )
        elif obj.point_xy_abs is not None:
            px, py = obj.point_xy_abs
            # Show point as a small box (2% of image) for visibility
            size_x, size_y = 0.02, 0.035
            detections.append(
                ObjectDetection(
                    label=obj.label,
                    confidence=1.0,
                    x=max(0.0, px / orig_w - size_x / 2),
                    y=max(0.0, py / orig_h - size_y / 2),
                    w=size_x,
                    h=size_y,
                )
            )
    return SimpleNamespace(detections=detections)


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
