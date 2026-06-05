"""FastAPI server for the SAM3 video labeller (web-app replacement for app.py).

Owns the SAM3 model hot in one long-lived process and exposes the
pause/correct/resume loop over HTTP + WebSocket, so the browser frontend can
hold all interaction state client-side (no Streamlit reruns).

Run with:
    uv run uvicorn footy_track.labeller.server:app --reload

Then open http://localhost:8000
"""

from __future__ import annotations

import asyncio
import contextlib
import os
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

from footy_track.labeller.video_utils import (  # noqa: E402
    BackgroundLabeller,
    LabelledObject,
    yolo_seed_objects,
)

app = FastAPI(title="SAM3 Video Labeller")

_STATIC_DIR = Path(__file__).parent / "web"


class Session:
    """Single-session server state (local, one user/one video at a time)."""

    def __init__(self) -> None:
        self.bg = BackgroundLabeller()
        self.video_path: Path | None = None
        self.fps: float = 25.0
        self.total_frames: int = 0
        self.width: int = 0
        self.height: int = 0

    def load(self, video_path: str) -> dict:
        self.bg.pause()
        self.bg = BackgroundLabeller()
        self.video_path = Path(video_path).expanduser()
        if not self.video_path.exists():
            raise FileNotFoundError(self.video_path)
        cap = cv2.VideoCapture(str(self.video_path))
        try:
            self.fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
            self.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        finally:
            cap.release()
        return {
            "fps": self.fps,
            "total_frames": self.total_frames,
            "width": self.width,
            "height": self.height,
        }

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


SESSION = Session()


# ----------------------------------------------------------------------------
# Serialization helpers (normalized boxes over the wire; pixels server-side)
# ----------------------------------------------------------------------------


def _objects_from_payload(items: list[dict]) -> list[LabelledObject]:
    """Convert client boxes (normalized xywh + label) to LabelledObject (abs xyxy)."""
    objs: list[LabelledObject] = []
    w, h = SESSION.width, SESSION.height
    for it in items:
        x1 = it["x"] * w
        y1 = it["y"] * h
        x2 = (it["x"] + it["w"]) * w
        y2 = (it["y"] + it["h"]) * h
        objs.append(LabelledObject(label=it["label"], bbox_xyxy_abs=(x1, y1, x2, y2)))
    return objs


def _detections_payload(fd) -> list[dict]:
    """FrameDetections -> normalized box dicts for the client."""
    return [
        {"label": d.label, "x": d.x, "y": d.y, "w": d.w, "h": d.h, "conf": d.confidence}
        for d in fd.detections
    ]


# ----------------------------------------------------------------------------
# HTTP endpoints
# ----------------------------------------------------------------------------


@app.get("/", response_class=HTMLResponse)
async def index() -> HTMLResponse:
    return HTMLResponse((_STATIC_DIR / "index.html").read_text())


@app.post("/session/load")
async def load_session(body: dict) -> dict:
    return SESSION.load(body["video_path"])


@app.get("/frame/{idx}.jpg")
async def get_frame(idx: int) -> Response:
    data = SESSION.frame_jpeg(idx)
    if data is None:
        return Response(status_code=404)
    return Response(content=data, media_type="image/jpeg")


@app.post("/autodetect")
async def autodetect(body: dict) -> dict:
    """Run YOLO on a frame; return seed boxes as normalized xywh dicts."""
    if SESSION.video_path is None:
        return {"objects": []}
    seeds = await asyncio.to_thread(
        yolo_seed_objects,
        SESSION.video_path,
        body.get("model_path", "") or "",
        float(body.get("conf", 0.35)),
        SESSION.width,
        SESSION.height,
        float(body.get("iou", 0.5)),
        int(body.get("frame_idx", 0)),
    )
    w, h = SESSION.width, SESSION.height
    objects = [
        {
            "label": o.label,
            "x": o.bbox_xyxy_abs[0] / w,
            "y": o.bbox_xyxy_abs[1] / h,
            "w": (o.bbox_xyxy_abs[2] - o.bbox_xyxy_abs[0]) / w,
            "h": (o.bbox_xyxy_abs[3] - o.bbox_xyxy_abs[1]) / h,
        }
        for o in seeds
        if o.bbox_xyxy_abs is not None
    ]
    return {"objects": objects}


@app.get("/detections/{idx}")
async def get_detections(idx: int) -> dict:
    completed = SESSION.bg.completed_frames()
    if 0 <= idx < len(completed):
        return {"idx": idx, "detections": _detections_payload(completed[idx])}
    return {"idx": idx, "detections": []}


# ----------------------------------------------------------------------------
# WebSocket: control (run/pause/restart) + live frame stream
# ----------------------------------------------------------------------------


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
                await websocket.send_json(
                    {
                        "type": "frame",
                        "idx": sent,
                        "detections": _detections_payload(completed[sent]),
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
                objects = _objects_from_payload(msg.get("objects", []))
                if not objects:
                    await websocket.send_json(
                        {"type": "error", "message": "No boxes to run."}
                    )
                    continue
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


# Mount static assets (JS/CSS) if any beyond index.html.
if _STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")
