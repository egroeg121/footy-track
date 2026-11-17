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
import json
import os
import tempfile
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

from footy_track.labeller.video_utils import (  # noqa: E402
    BackgroundLabeller,
    LabelledObject,
    yolo_seed_objects,
)
from footy_track.pitch_crop import (  # noqa: E402
    CropRect,
    compute_segment_crop,
    write_cropped_video,
)
from footy_track.schema import ObjectDetection  # noqa: E402

app = FastAPI(title="SAM3 Video Labeller")

_STATIC_DIR = Path(__file__).parent / "web"


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
        self.video_path: Path | None = None  # original (full-frame) video
        self.work_path: Path | None = None  # what the labeller/SAM3 actually use
        self.crop: CropRect | None = None  # full <-> crop transform (None = no crop)
        self.crop_corrected: bool = False  # True once the user hand-adjusts framing
        self.fps: float = 25.0
        self.total_frames: int = 0
        # width/height are the WORKING (crop) dims — the timeline lives in this
        # space, so seeds/output/anomaly all normalize consistently. Full-frame
        # dims are kept separately for the export remap + the full-frame toggle.
        self.width: int = 0
        self.height: int = 0
        self.full_width: int = 0
        self.full_height: int = 0
        self.timeline: list[list[ObjectDetection] | None] = []
        self._tl_lock = threading.Lock()

    def load(self, video_path: str, crop_pitch: bool = True) -> dict:
        self.bg.pause()
        self.bg = BackgroundLabeller()
        self.video_path = Path(video_path).expanduser()
        if not self.video_path.exists():
            raise FileNotFoundError(self.video_path)
        cap = cv2.VideoCapture(str(self.video_path))
        try:
            self.fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
            self.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.full_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.full_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        finally:
            cap.release()

        # Pick the crop: a hand-corrected one saved next to the video wins; else
        # the harsh auto-crop (aggressive on purpose — the user can expand it and
        # the correction is persisted). Then bake a cropped working video so the
        # whole labeller (frame serving + SAM3) runs in crop space; full-frame
        # coords are only reconstructed at export.
        saved = self._load_sidecar()
        if saved is not None:
            self.crop, self.crop_corrected = saved, True
        elif crop_pitch:
            self.crop, self.crop_corrected = (
                compute_segment_crop(self.video_path),
                False,
            )
        else:
            self.crop = CropRect.identity(self.full_width, self.full_height)
            self.crop_corrected = False
        self._bake_working_video()

        with self._tl_lock:
            self.timeline = [None] * self.total_frames
        return self._meta()

    # --- crop / framing --------------------------------------------------

    def _meta(self) -> dict:
        """The session descriptor the client needs (dims + current crop)."""
        return {
            "fps": self.fps,
            "total_frames": self.total_frames,
            "width": self.width,
            "height": self.height,
            "full_width": self.full_width,
            "full_height": self.full_height,
            "crop": {
                "x0": self.crop.x0,
                "y0": self.crop.y0,
                "x1": self.crop.x1,
                "y1": self.crop.y1,
                "is_identity": self.crop.is_identity,
            },
            "crop_corrected": self.crop_corrected,
        }

    def _bake_working_video(self) -> None:
        """Materialise the cropped working video (no-op clip for identity crop)."""
        if self.crop.is_identity:
            self.work_path = self.video_path
        else:
            tmp = (
                Path(tempfile.gettempdir())
                / f"_labeller_crop_{self.video_path.stem}.mp4"
            )
            write_cropped_video(self.video_path, tmp, self.crop)
            self.work_path = tmp
        self.width, self.height = self.crop.cw, self.crop.ch

    def _sidecar_path(self) -> Path:
        return self.video_path.with_suffix(self.video_path.suffix + ".pitchcrop.json")

    def _load_sidecar(self) -> CropRect | None:
        p = self._sidecar_path()
        if not p.exists():
            return None
        d = json.loads(p.read_text())
        return CropRect(
            d["x0"], d["y0"], d["x1"], d["y1"], self.full_width, self.full_height
        )

    def _save_sidecar(self) -> None:
        self._sidecar_path().write_text(
            json.dumps(
                {
                    "x0": self.crop.x0,
                    "y0": self.crop.y0,
                    "x1": self.crop.x1,
                    "y1": self.crop.y1,
                    "corrected": True,
                }
            )
        )

    def set_crop(self, new_crop: CropRect, corrected: bool) -> dict:
        """Switch to ``new_crop``, remapping the timeline from the old crop space.

        Existing boxes are crop-normalized to the *old* crop; we round-trip them
        through full-frame coords into the new crop so labels survive a reframe.
        A ``corrected`` crop is persisted to the sidecar as ground-truth framing.
        """
        old = self.crop
        with self._tl_lock:
            for i, boxes in enumerate(self.timeline):
                if not boxes:
                    continue
                remapped = []
                for b in boxes:
                    fx, fy, fw, fh = old.to_full(b.x, b.y, b.w, b.h)
                    nx, ny, nw, nh = new_crop.to_crop(fx, fy, fw, fh)
                    remapped.append(
                        b.model_copy(update={"x": nx, "y": ny, "w": nw, "h": nh})
                    )
                self.timeline[i] = remapped
        self.crop = new_crop
        self._bake_working_video()
        if corrected:
            self.crop_corrected = True
            self._save_sidecar()
        return self._meta()

    def expand_crop(self, pct: float = 0.1) -> dict:
        """Hand-correction: grow the framing by ``pct`` on every side (clamped).

        Clipping is usually at the top (harsh grass line), so this mainly adds
        headroom; sides/bottom are no-ops when already at the frame edge. Marked
        as a manual correction and saved.
        """
        c = self.crop
        dx, dy = int(c.cw * pct), int(c.ch * pct)
        new = CropRect(
            x0=max(0, c.x0 - dx),
            y0=max(0, c.y0 - dy),
            x1=min(self.full_width, c.x1 + dx),
            y1=min(self.full_height, c.y1 + dy),
            W=self.full_width,
            H=self.full_height,
        )
        return self.set_crop(new, corrected=True)

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

    def frame_jpeg(self, idx: int, full: bool = False) -> bytes | None:
        """Encode frame ``idx`` as JPEG.

        Default reads the cropped working video (what the canvas + SAM3 see).
        ``full=True`` reads the original full frame — used by the UI's
        'show full frame' toggle to give context / reveal anything the crop cut.
        """
        path = self.video_path if full else self.work_path
        if path is None:
            return None
        cap = cv2.VideoCapture(str(path))
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


@app.post("/session/load")
async def load_session(body: dict) -> dict:
    return SESSION.load(
        body["video_path"], crop_pitch=bool(body.get("crop_pitch", True))
    )


@app.get("/frame/{idx}.jpg")
async def get_frame(idx: int) -> Response:
    data = SESSION.frame_jpeg(idx)
    if data is None:
        return Response(status_code=404)
    return Response(content=data, media_type="image/jpeg")


@app.get("/frame_full/{idx}.jpg")
async def get_frame_full(idx: int) -> Response:
    """Full (uncropped) frame — for the UI 'show full frame' toggle."""
    data = SESSION.frame_jpeg(idx, full=True)
    if data is None:
        return Response(status_code=404)
    return Response(content=data, media_type="image/jpeg")


@app.post("/crop/expand")
async def crop_expand(body: dict) -> dict:
    """Hand-correct the framing: grow the crop by ``pct`` and persist it.

    Re-bakes the cropped working video and remaps the timeline, so it's a
    blocking op run off the event loop. Returns the updated session meta.
    """
    if SESSION.video_path is None:
        return {}
    pct = float(body.get("pct", 0.1))
    return await asyncio.to_thread(SESSION.expand_crop, pct)


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
        SESSION.work_path,  # cropped working video -> seeds land in crop space
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
                # Seed from the TIMELINE at the start frame (the single source of
                # truth) — not from whatever the client happens to send. This is
                # what makes "restart from frame N" deterministic.
                objects = SESSION.seed_objects(start_frame)
                print(
                    f"[ws] {mtype}: start_frame={start_frame} "
                    f"seed_objects={len(objects)} (from timeline)",
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
                await asyncio.to_thread(
                    bg.submit,
                    SESSION.work_path,  # SAM3 streams the cropped working video
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
