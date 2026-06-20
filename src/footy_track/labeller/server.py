"""FastAPI server for the SAM3 video labeller (web-app replacement for app.py).

Owns the SAM3 model hot in one long-lived process and exposes the
pause/correct/resume loop over HTTP + WebSocket, so the browser frontend can
hold all interaction state client-side (no Streamlit reruns).

Run with:
    uv run uvicorn footy_track.labeller.server:app --reload

Then open http://localhost:8000

Ball GT marking mode (http://localhost:8000/gt):
    Pick an eval clip from eval_data/clips/, scrub frames, click the ball center.
    Marks save incrementally to eval_data/clips/<clip>.jsonl in FrameLabel format.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import threading
from pathlib import Path

# Persist torch.compile (Inductor) cache before torch loads — see video_utils.
os.environ.setdefault(
    "TORCHINDUCTOR_CACHE_DIR",
    str(Path.home() / ".cache" / "footy_torch_inductor"),
)

import cv2  # noqa: E402
from fastapi import FastAPI, WebSocket, WebSocketDisconnect  # noqa: E402
from fastapi.responses import HTMLResponse, Response  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402

from footy_track.ball_eval.dataset import FrameLabel, write_labels  # noqa: E402
from footy_track.labeller.video_utils import (  # noqa: E402
    BackgroundLabeller,
    LabelledObject,
    yolo_seed_objects,
)
from footy_track.schema import ObjectDetection  # noqa: E402

app = FastAPI(title="SAM3 Video Labeller")

_STATIC_DIR = Path(__file__).parent / "web"

# Eval clips directory: <repo_root>/eval_data/clips/
# Walk up from this file: labeller/ -> footy_track/ -> src/ -> <repo>
_EVAL_CLIPS_DIR = Path(__file__).parent.parent.parent.parent / "eval_data" / "clips"
_VIDEO_SUFFIXES = {".mp4", ".avi", ".mov", ".mkv"}

# Bake-off method identifiers exposed to the client
METHODS = {
    "sam3": "SAM3 (current baseline)",
    "sot": "SOT VitTrack (method A)",
    "sam2": "SAM2 ROI (method B)",
    "roi_yolo": "ROI-YOLO (method C)",
}

# Provenance tags stored in each box's ObjectDetection.model field.
PROV_LABELLER = "labeller"  # manual edit — ground truth, never auto-overwritten
PROV_YOLO = "yolo"
PROV_SAM3 = "sam3"


class Session:
    """Single-session server state with one authoritative per-frame timeline.

    ``timeline[i]`` is the list of boxes (ObjectDetection, normalized) for frame
    i, or None if that frame has never been populated. Every actor — YOLO,
    SAM3, the user — writes here. Box provenance lives in ObjectDetection.model
    (PROV_*). Labeller boxes are ground truth and survive auto re-propagation.
    """

    def __init__(self) -> None:
        self.bg = BackgroundLabeller()
        self.video_path: Path | None = None
        self.fps: float = 25.0
        self.total_frames: int = 0
        self.width: int = 0
        self.height: int = 0
        self.timeline: list[list[ObjectDetection] | None] = []
        self._tl_lock = threading.Lock()

    def load(self, video_path: str) -> dict:
        p = Path(video_path).expanduser()
        if not p.exists():
            raise FileNotFoundError(p)
        self.bg.pause()
        self.bg = BackgroundLabeller()
        self.video_path = p
        cap = cv2.VideoCapture(str(p))
        try:
            self.fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
            self.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        finally:
            cap.release()
        with self._tl_lock:
            self.timeline = [None] * self.total_frames
        return {
            "fps": self.fps,
            "total_frames": self.total_frames,
            "width": self.width,
            "height": self.height,
        }

    # --- timeline access -------------------------------------------------

    def get_frame(self, idx: int) -> list[ObjectDetection]:
        with self._tl_lock:
            if 0 <= idx < len(self.timeline) and self.timeline[idx] is not None:
                return list(self.timeline[idx])
            return []

    def set_frame(self, idx: int, boxes: list[ObjectDetection]) -> None:
        """Overwrite a frame entirely (used by user edits / autodetect)."""
        with self._tl_lock:
            if 0 <= idx < len(self.timeline):
                self.timeline[idx] = list(boxes)

    def merge_propagated(self, idx: int, boxes: list[ObjectDetection]) -> None:
        """Write propagated (sam3/yolo) boxes, KEEPING any labeller ground truth."""
        with self._tl_lock:
            if not (0 <= idx < len(self.timeline)):
                return
            existing = self.timeline[idx] or []
            kept = [b for b in existing if b.model == PROV_LABELLER]
            self.timeline[idx] = kept + list(boxes)

    def seed_objects(self, idx: int) -> list[LabelledObject]:
        """Frame idx's boxes as LabelledObjects (abs xyxy) to seed propagation."""
        objs: list[LabelledObject] = []
        for b in self.get_frame(idx):
            objs.append(
                LabelledObject(
                    label=b.label,
                    bbox_xyxy_abs=(
                        b.x * self.width,
                        b.y * self.height,
                        (b.x + b.w) * self.width,
                        (b.y + b.h) * self.height,
                    ),
                )
            )
        return objs

    def frame_jpeg(self, idx: int, roi_box: tuple | None = None) -> bytes | None:
        """Return JPEG for frame ``idx``.

        If ``roi_box`` is given as (x, y, w, h) normalised, also renders a
        second panel showing the ROI crop alongside the full frame (debug mode).
        """
        if self.video_path is None:
            return None
        cap = cv2.VideoCapture(str(self.video_path))
        try:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, frame = cap.read()
        finally:
            cap.release()
        if not ok:
            return None
        if roi_box is not None:
            frame = _render_debug_frame(frame, roi_box)
        ok, buf = cv2.imencode(".jpg", frame)
        return buf.tobytes() if ok else None


def _render_debug_frame(
    frame: cv2.typing.MatLike,
    roi_box: tuple,
) -> cv2.typing.MatLike:
    """Render full frame + ROI crop side by side for debug/dual-image mode."""
    h, w = frame.shape[:2]
    x, y, bw, bh = roi_box
    x1 = max(0, int(x * w))
    y1 = max(0, int(y * h))
    x2 = min(w, int((x + bw) * w))
    y2 = min(h, int((y + bh) * h))

    # Draw ROI rectangle on full frame
    annotated = frame.copy()
    cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 255), 2)

    # Crop the ROI and scale to a quarter of the full frame height
    crop = frame[y1:y2, x1:x2] if (x2 > x1 and y2 > y1) else frame[:64, :64]
    panel_h = h // 2
    panel_w = max(1, int(crop.shape[1] * panel_h / max(crop.shape[0], 1)))
    crop_resized = cv2.resize(crop, (panel_w, panel_h))

    # Pad crop to same height as full frame
    crop_padded = cv2.copyMakeBorder(
        crop_resized, 0, h - panel_h, 0, 0, cv2.BORDER_CONSTANT, value=(30, 30, 30)
    )
    # Make crop panel same width as needed, pad to w//2
    target_w = w // 2
    if crop_padded.shape[1] < target_w:
        crop_padded = cv2.copyMakeBorder(
            crop_padded,
            0,
            0,
            0,
            target_w - crop_padded.shape[1],
            cv2.BORDER_CONSTANT,
            value=(30, 30, 30),
        )
    else:
        crop_padded = crop_padded[:, :target_w]

    # Shrink full frame to same width
    full_resized = cv2.resize(annotated, (target_w, h))
    combined = cv2.hconcat([full_resized, crop_padded])
    return combined


SESSION = Session()


# ----------------------------------------------------------------------------
# Serialization helpers (normalized boxes over the wire; pixels server-side)
# ----------------------------------------------------------------------------


def _boxes_from_payload(items: list[dict], provenance: str) -> list[ObjectDetection]:
    """Client boxes (normalized xywh + label) -> ObjectDetection with provenance."""
    out: list[ObjectDetection] = []
    for it in items:
        out.append(
            ObjectDetection(
                label=it["label"],
                confidence=float(it.get("conf", 1.0)),
                x=max(0.0, min(1.0, it["x"])),
                y=max(0.0, min(1.0, it["y"])),
                w=max(0.0, min(1.0, it["w"])),
                h=max(0.0, min(1.0, it["h"])),
                model=provenance,
            )
        )
    return out


def _boxes_payload(boxes: list[ObjectDetection]) -> list[dict]:
    """ObjectDetection list -> normalized box dicts for the client (with source)."""
    return [
        {
            "label": b.label,
            "x": b.x,
            "y": b.y,
            "w": b.w,
            "h": b.h,
            "conf": b.confidence,
            "source": b.model,
        }
        for b in boxes
    ]


# ----------------------------------------------------------------------------
# HTTP endpoints
# ----------------------------------------------------------------------------


@app.get("/", response_class=HTMLResponse)
async def index() -> HTMLResponse:
    return HTMLResponse((_STATIC_DIR / "index.html").read_text())


@app.get("/gt", response_class=HTMLResponse)
async def gt_index() -> HTMLResponse:
    return HTMLResponse((_STATIC_DIR / "gt.html").read_text())


@app.post("/session/load")
async def load_session(body: dict) -> dict:
    try:
        return SESSION.load(body["video_path"])
    except FileNotFoundError as exc:
        from fastapi import HTTPException  # noqa: PLC0415

        raise HTTPException(status_code=404, detail=f"Video not found: {exc}") from exc


@app.get("/frame/{idx}.jpg")
async def get_frame(idx: int, debug: int = 0) -> Response:
    # debug=1: ask for dual-image if we have a last-known ROI
    roi = None
    if debug:
        boxes = SESSION.get_frame(idx)
        ball_box = next((b for b in boxes if "ball" in b.label.lower()), None)
        if ball_box is not None:
            roi = (ball_box.x, ball_box.y, ball_box.w, ball_box.h)
    data = SESSION.frame_jpeg(idx, roi_box=roi)
    if data is None:
        return Response(status_code=404)
    return Response(content=data, media_type="image/jpeg")


@app.post("/autodetect")
async def autodetect(body: dict) -> dict:
    """Run YOLO on a frame, WRITE the boxes (yolo provenance) into the timeline.

    Keeps any existing labeller boxes on that frame (merge rule). Returns the
    frame's full box set so the client can render it.
    """
    if SESSION.video_path is None:
        return {"idx": 0, "boxes": []}
    idx = int(body.get("frame_idx", 0))
    seeds = await asyncio.to_thread(
        yolo_seed_objects,
        SESSION.video_path,
        body.get("model_path", "") or "",
        float(body.get("conf", 0.35)),
        SESSION.width,
        SESSION.height,
        float(body.get("iou", 0.5)),
        idx,
    )
    w, h = SESSION.width, SESSION.height
    yolo_boxes = [
        ObjectDetection(
            label=o.label,
            confidence=1.0,
            x=o.bbox_xyxy_abs[0] / w,
            y=o.bbox_xyxy_abs[1] / h,
            w=(o.bbox_xyxy_abs[2] - o.bbox_xyxy_abs[0]) / w,
            h=(o.bbox_xyxy_abs[3] - o.bbox_xyxy_abs[1]) / h,
            model=PROV_YOLO,
        )
        for o in seeds
        if o.bbox_xyxy_abs is not None
    ]
    # Replace this frame's non-labeller boxes with the fresh YOLO set.
    existing = SESSION.get_frame(idx)
    kept = [b for b in existing if b.model == PROV_LABELLER]
    SESSION.set_frame(idx, kept + yolo_boxes)
    return {"idx": idx, "boxes": _boxes_payload(SESSION.get_frame(idx))}


@app.get("/timeline/{idx}")
async def get_timeline(idx: int) -> dict:
    """Return the authoritative boxes (with provenance) for a frame."""
    return {"idx": idx, "boxes": _boxes_payload(SESSION.get_frame(idx))}


@app.post("/edit")
async def edit_frame(body: dict) -> dict:
    """Overwrite a frame with the user's boxes (labeller provenance = ground truth)."""
    idx = int(body["idx"])
    boxes = _boxes_from_payload(body.get("objects", []), PROV_LABELLER)
    SESSION.set_frame(idx, boxes)
    return {"idx": idx, "boxes": _boxes_payload(SESSION.get_frame(idx))}


@app.get("/methods")
async def list_methods() -> dict:
    """Return the available bake-off tracking methods."""
    return {"methods": [{"id": k, "label": v} for k, v in METHODS.items()]}


@app.get("/clips")
async def list_clips() -> dict:
    """List discoverable eval clips for the main labeller."""
    clips: list[dict] = []
    if _EVAL_CLIPS_DIR.exists():
        for p in sorted(_EVAL_CLIPS_DIR.iterdir()):
            if p.suffix.lower() in _VIDEO_SUFFIXES:
                clips.append({"name": p.stem, "path": str(p)})
    return {"clips": clips, "clips_dir": str(_EVAL_CLIPS_DIR)}


@app.post("/session/export")
async def export_session(body: dict) -> dict:
    """Export the current session timeline to a JSONL sidecar next to the video.

    Each populated frame is written as a FrameLabel.  Ball-box center is computed
    from the first 'ball' box on the frame; frames with no boxes are skipped.
    The sidecar path is ``<video_stem>.jsonl`` in the same directory as the video.
    Pass ``{"path": "/custom/out.jsonl"}`` in the body to override the destination.
    """
    if SESSION.video_path is None:
        from fastapi import HTTPException  # noqa: PLC0415

        raise HTTPException(status_code=400, detail="No video loaded")

    out_path = Path(body.get("path") or SESSION.video_path.with_suffix(".jsonl"))

    labels: list[FrameLabel] = []
    with SESSION._tl_lock:
        timeline_copy = list(SESSION.timeline)

    for frame_idx, boxes in enumerate(timeline_copy):
        if not boxes:
            continue
        ball_box = next((b for b in boxes if "ball" in b.label.lower()), None)
        if ball_box is None:
            ball_box = boxes[0]
        cx = ball_box.x + ball_box.w / 2
        cy = ball_box.y + ball_box.h / 2
        labels.append(
            FrameLabel(
                frame_index=frame_idx,
                bbox=(ball_box.x, ball_box.y, ball_box.w, ball_box.h),
                center=(cx, cy),
                tags=(),
            )
        )

    write_labels(labels, out_path)
    return {"saved": str(out_path), "frames": len(labels)}


# ----------------------------------------------------------------------------
# WebSocket: control (run/pause/restart) + live frame stream
# ----------------------------------------------------------------------------


def _ingest_completed_frame(idx: int, fd, start_idx: int) -> list[ObjectDetection]:
    """Write a propagated frame into the timeline and return its boxes to send.

    The seed frame (idx == start_idx) is the ground-truth seed — its timeline
    entry is already correct, so we don't merge SAM3's re-segmentation into it.
    Downstream frames get SAM3 boxes merged in (keeping labeller ground truth).
    """
    if idx == start_idx:
        return SESSION.get_frame(idx)
    sam3_boxes = [
        ObjectDetection(
            label=d.label,
            confidence=d.confidence,
            x=d.x,
            y=d.y,
            w=d.w,
            h=d.h,
            model=PROV_SAM3,
        )
        for d in fd.detections
    ]
    SESSION.merge_propagated(idx, sam3_boxes)
    return SESSION.get_frame(idx)


async def _stream_frames(websocket: WebSocket, start_idx: int) -> None:
    """Push each newly-completed frame to the client until the run stops."""
    sent = start_idx - 1
    await websocket.send_json({"type": "status", "state": "compiling"})
    announced_running = False
    while True:
        cur_bg = SESSION.bg  # may swap on a fresh load
        while sent < cur_bg.last_completed_frame:
            sent += 1
            completed = cur_bg.completed_frames()
            if sent < len(completed):
                if not announced_running:
                    await websocket.send_json({"type": "status", "state": "running"})
                    announced_running = True
                boxes = _ingest_completed_frame(sent, completed[sent], start_idx)
                await websocket.send_json(
                    {
                        "type": "frame",
                        "idx": sent,
                        "boxes": _boxes_payload(boxes),
                    }
                )
        if cur_bg.anomaly_frame is not None:
            await websocket.send_json(
                {
                    "type": "anomaly",
                    "idx": cur_bg.anomaly_frame,
                    "reason": cur_bg.anomaly_reason or "implausible track motion",
                }
            )
            cur_bg.anomaly_frame = None
            await websocket.send_json({"type": "status", "state": "paused"})
            return
        if not cur_bg.running:
            await websocket.send_json(
                {"type": "done", "last_frame": cur_bg.last_completed_frame}
            )
            await websocket.send_json({"type": "status", "state": "idle"})
            return
        await asyncio.sleep(0.1)


@app.websocket("/ws")
async def ws(websocket: WebSocket) -> None:
    await websocket.accept()
    bg = SESSION.bg
    streamer: asyncio.Task | None = None

    async def stream_frames(start_idx: int) -> None:
        await _stream_frames(websocket, start_idx)

    try:
        while True:
            msg = await websocket.receive_json()
            mtype = msg.get("type")

            if mtype in ("run", "restart"):
                if streamer and not streamer.done():
                    SESSION.bg.pause()
                    streamer.cancel()
                    with contextlib.suppress(asyncio.CancelledError):
                        await streamer
                bg = SESSION.bg
                start_frame = int(msg.get("start_frame", 0))
                method = msg.get("method", "sam3")
                # Seed from the TIMELINE at the start frame (the single source of
                # truth) — not from whatever the client happens to send.
                objects = SESSION.seed_objects(start_frame)
                print(
                    f"[ws] {mtype}: start_frame={start_frame} "
                    f"seed_objects={len(objects)} method={method} (from timeline)",
                    flush=True,
                )
                if not objects:
                    await websocket.send_json(
                        {
                            "type": "error",
                            "message": f"No boxes on frame {start_frame} to seed from.",
                        }
                    )
                    continue

                if method == "sam3":
                    await asyncio.to_thread(
                        bg.submit,
                        SESSION.video_path,
                        objects,
                        msg.get("model_uri") or None,
                        float(msg.get("conf", 0.25)),
                        start_frame,
                        int(msg.get("imgsz", 512)),
                    )
                    streamer = asyncio.create_task(stream_frames(start_frame))
                else:
                    # Non-SAM3 bake-off method: run frame-by-frame via helper
                    ball_box = next(
                        (
                            (
                                o.bbox_xyxy_abs[0] / SESSION.width,
                                o.bbox_xyxy_abs[1] / SESSION.height,
                                (o.bbox_xyxy_abs[2] - o.bbox_xyxy_abs[0])
                                / SESSION.width,
                                (o.bbox_xyxy_abs[3] - o.bbox_xyxy_abs[1])
                                / SESSION.height,
                            )
                            for o in objects
                            if "ball" in o.label.lower() and o.bbox_xyxy_abs
                        ),
                        None,
                    )
                    if ball_box is None:
                        # Fall back to first object
                        o = objects[0]
                        if o.bbox_xyxy_abs:
                            x1, y1, x2, y2 = o.bbox_xyxy_abs
                            ball_box = (
                                x1 / SESSION.width,
                                y1 / SESSION.height,
                                (x2 - x1) / SESSION.width,
                                (y2 - y1) / SESSION.height,
                            )
                    streamer = asyncio.create_task(
                        _stream_bakeoff(websocket, method, start_frame, ball_box)
                    )

            elif mtype == "pause":
                SESSION.bg.pause()
                if streamer and not streamer.done():
                    streamer.cancel()
                    with contextlib.suppress(asyncio.CancelledError):
                        await streamer
                await websocket.send_json({"type": "status", "state": "paused"})

    except WebSocketDisconnect:
        SESSION.bg.pause()
        if streamer and not streamer.done():
            streamer.cancel()


async def _stream_bakeoff(
    websocket: WebSocket,
    method: str,
    start_frame: int,
    seed_bbox: tuple | None,
) -> None:
    """Run a non-SAM3 bake-off tracker frame-by-frame and stream results."""
    try:
        tracker = _build_bakeoff_tracker(method)
    except ImportError as exc:
        await websocket.send_json(
            {"type": "error", "message": f"Method {method!r} unavailable: {exc}"}
        )
        return

    await websocket.send_json({"type": "status", "state": "running"})
    total = SESSION.total_frames
    cap = cv2.VideoCapture(str(SESSION.video_path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    prev_bbox = seed_bbox
    abs_idx = start_frame - 1  # guard against empty range in done message
    try:
        for abs_idx in range(start_frame, total):
            ok, bgr = cap.read()
            if not ok:
                break
            frame_rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            try:
                result_bbox = await asyncio.to_thread(
                    tracker.track, prev_bbox, frame_rgb
                )
            except Exception as exc:  # noqa: BLE001
                await websocket.send_json(
                    {
                        "type": "error",
                        "message": f"Tracker error at frame {abs_idx}: {exc}",
                    }
                )
                break
            boxes: list[dict] = []
            if result_bbox is not None:
                x, y, w, h = result_bbox
                boxes = [
                    {
                        "label": "ball",
                        "x": x,
                        "y": y,
                        "w": w,
                        "h": h,
                        "conf": 1.0,
                        "source": method,
                    }
                ]
                prev_bbox = result_bbox
                # Persist to timeline
                SESSION.merge_propagated(
                    abs_idx,
                    [
                        ObjectDetection(
                            label="ball",
                            confidence=1.0,
                            x=x,
                            y=y,
                            w=w,
                            h=h,
                            model=method,
                        )
                    ],
                )
            await websocket.send_json({"type": "frame", "idx": abs_idx, "boxes": boxes})
    finally:
        cap.release()
    await websocket.send_json({"type": "done", "last_frame": abs_idx})
    await websocket.send_json({"type": "status", "state": "idle"})


def _build_bakeoff_tracker(method: str):
    """Instantiate a bake-off tracker by method id. Raises ImportError if unavailable."""
    if method == "sot":
        from footy_track.ball_trackers.sot_vittrack import VitTrackSOT  # noqa: PLC0415

        return VitTrackSOT()
    if method == "sam2":
        from footy_track.ball_tracking.sam2_tracker import SAM2Tracker  # noqa: PLC0415

        return SAM2Tracker()
    if method == "roi_yolo":
        from footy_track.ball_trackers.roi_yolo import ROIYOLOTracker  # noqa: PLC0415

        return ROIYOLOTracker()
    raise ImportError(f"Unknown bake-off method: {method!r}")


# ----------------------------------------------------------------------------
# Ball GT marking mode  (/gt  — browser-based ball-center marking)
# ----------------------------------------------------------------------------


class GTSession:
    """State for the ball-center GT marking session."""

    def __init__(self) -> None:
        self.video_path: Path | None = None
        self.total_frames: int = 0
        self.width: int = 0
        self.height: int = 0
        self.fps: float = 25.0
        self.labels: dict[int, FrameLabel] = {}  # frame_index -> FrameLabel
        self._lock = threading.Lock()

    def load(self, video_path: Path) -> dict:
        if not video_path.exists():
            raise FileNotFoundError(video_path)
        cap = cv2.VideoCapture(str(video_path))
        try:
            self.fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
            self.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        finally:
            cap.release()
        self.video_path = video_path
        # Load existing labels if sidecar exists (resumable)
        jsonl = video_path.with_suffix(".jsonl")
        with self._lock:
            self.labels = {}
            if jsonl.exists():
                with jsonl.open() as f:
                    for raw_line in f:
                        stripped = raw_line.strip()
                        if stripped:
                            d = json.loads(stripped)
                            lbl = FrameLabel.from_dict(d)
                            self.labels[lbl.frame_index] = lbl
        return {
            "fps": self.fps,
            "total_frames": self.total_frames,
            "width": self.width,
            "height": self.height,
            "clip_name": video_path.stem,
            "existing_marks": len(self.labels),
        }

    def mark_center(self, frame_index: int, cx: float, cy: float) -> None:
        """Record a ball center mark and flush to JSONL incrementally."""
        lbl = FrameLabel(frame_index=frame_index, bbox=None, tags=(), center=(cx, cy))
        with self._lock:
            self.labels[frame_index] = lbl
            self._flush()

    def mark_absent(self, frame_index: int) -> None:
        """Record ball-not-visible and flush incrementally."""
        lbl = FrameLabel(
            frame_index=frame_index, bbox=None, tags=("ball_not_visible",), center=None
        )
        with self._lock:
            self.labels[frame_index] = lbl
            self._flush()

    def unmark(self, frame_index: int) -> None:
        """Remove the mark for a frame."""
        with self._lock:
            self.labels.pop(frame_index, None)
            self._flush()

    def _flush(self) -> None:
        """Write all labels to the sidecar JSONL (must hold _lock)."""
        if self.video_path is None:
            return
        out = self.video_path.with_suffix(".jsonl")
        sorted_labels = [self.labels[k] for k in sorted(self.labels)]
        write_labels(sorted_labels, out)

    def get_marks(self) -> list[dict]:
        with self._lock:
            return [lbl.to_dict() for lbl in self.labels.values()]

    def frame_jpeg(self, idx: int) -> bytes | None:
        if self.video_path is None:
            return None
        cap = cv2.VideoCapture(str(self.video_path))
        try:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, frame = cap.read()
        finally:
            cap.release()
        if not ok:
            return None
        ok, buf = cv2.imencode(".jpg", frame)
        return buf.tobytes() if ok else None


GT_SESSION = GTSession()


def _list_eval_clips() -> list[dict]:
    """Return clips in eval_data/clips/ that have a matching video file."""
    if not _EVAL_CLIPS_DIR.exists():
        return []
    clips = []
    for p in sorted(_EVAL_CLIPS_DIR.iterdir()):
        if p.suffix.lower() in _VIDEO_SUFFIXES:
            jsonl = p.with_suffix(".jsonl")
            mark_count = 0
            if jsonl.exists():
                with jsonl.open() as f:
                    mark_count = sum(1 for line in f if line.strip())
            clips.append({"name": p.stem, "path": str(p), "marks": mark_count})
    return clips


@app.get("/gt/clips")
async def gt_list_clips() -> dict:
    clips = await asyncio.to_thread(_list_eval_clips)
    return {"clips": clips, "clips_dir": str(_EVAL_CLIPS_DIR)}


@app.post("/gt/load")
async def gt_load(body: dict) -> dict:
    path = Path(body["path"]).expanduser()
    try:
        result = await asyncio.to_thread(GT_SESSION.load, path)
    except FileNotFoundError as exc:
        from fastapi import HTTPException  # noqa: PLC0415

        raise HTTPException(status_code=404, detail=f"File not found: {path}") from exc
    return result


@app.get("/gt/frame/{idx}.jpg")
async def gt_frame(idx: int) -> Response:
    data = await asyncio.to_thread(GT_SESSION.frame_jpeg, idx)
    if data is None:
        return Response(status_code=404)
    return Response(content=data, media_type="image/jpeg")


@app.get("/gt/marks")
async def gt_get_marks() -> dict:
    return {"marks": GT_SESSION.get_marks()}


@app.post("/gt/mark")
async def gt_mark(body: dict) -> dict:
    """Mark ball center (cx, cy normalised) or mark as absent."""
    idx = int(body["frame_index"])
    if body.get("absent"):
        await asyncio.to_thread(GT_SESSION.mark_absent, idx)
    else:
        cx = float(body["cx"])
        cy = float(body["cy"])
        await asyncio.to_thread(GT_SESSION.mark_center, idx, cx, cy)
    return {"frame_index": idx, "total_marks": len(GT_SESSION.labels)}


@app.post("/gt/unmark")
async def gt_unmark(body: dict) -> dict:
    idx = int(body["frame_index"])
    await asyncio.to_thread(GT_SESSION.unmark, idx)
    return {"frame_index": idx, "total_marks": len(GT_SESSION.labels)}


# Mount static assets (JS/CSS) if any beyond index.html.
if _STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")
