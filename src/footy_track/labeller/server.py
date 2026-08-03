"""FastAPI server for the Footy Track video labeller (web-app replacement for app.py).

Uses VitTrack SOT as the propagation backend behind the Run button.
Exposes the pause/correct/resume loop over HTTP + WebSocket, so the browser
frontend can hold all interaction state client-side (no Streamlit reruns).

Run with:
    uv run uvicorn footy_track.labeller.server:app --reload

Then open http://localhost:8000
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import sys
import tempfile
from pathlib import Path

# Persist torch.compile (Inductor) cache before torch loads — see video_utils.
os.environ.setdefault(
    "TORCHINDUCTOR_CACHE_DIR",
    str(Path.home() / ".cache" / "footy_torch_inductor"),
)

import cv2  # noqa: E402
from fastapi import (  # noqa: E402
    FastAPI,
    File,
    UploadFile,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.responses import HTMLResponse, Response, StreamingResponse  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402

from footy_track.ball_eval.metrics import bbox_iou  # noqa: E402
from footy_track.labeller.constants import (  # noqa: E402
    BALL_LABELS,
    PLAYER_LABELS,
    PROV_LABELLER,
    PROV_SAM3,
    PROV_VITTRACK,
    PROV_YOLO,
)
from footy_track.labeller.review import router as review_router  # noqa: E402
from footy_track.labeller.session import (  # noqa: E402
    Session,
    boxes_from_payload,
    boxes_payload,
)
from footy_track.labeller.video_utils import yolo_seed_objects  # noqa: E402
from footy_track.schema import ObjectDetection  # noqa: E402

__all__ = [
    "PROV_LABELLER",
    "PROV_SAM3",
    "PROV_VITTRACK",
    "PROV_YOLO",
    "SESSION",
    "Session",
    "app",
]

app = FastAPI(title="Footy Track Labeller")

# server.py is the composition root and config surface: extracted modules
# resolve these at call time (and tests monkeypatch them here).
_STATIC_DIR = Path(__file__).parent / "web"
_CLIPS_DIR = Path(__file__).parents[3] / "eval_data" / "clips"
_GT_MARKS_DIR = (
    Path.home()
    / "Library"
    / "Mobile Documents"
    / "com~apple~CloudDocs"
    / "footy_data"
    / "ball_gt_marks"
)


SESSION = Session()


# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------
# HTTP endpoints
# ----------------------------------------------------------------------------


@app.get("/", response_class=HTMLResponse)
@app.get("/main", response_class=HTMLResponse)
async def root() -> HTMLResponse:
    return HTMLResponse((_STATIC_DIR / "main.html").read_text())


@app.get("/labeller", response_class=HTMLResponse)
async def labeller_page() -> HTMLResponse:
    return HTMLResponse((_STATIC_DIR / "index.html").read_text())


@app.get("/object_review", response_class=HTMLResponse)
async def review_page() -> HTMLResponse:
    return HTMLResponse((_STATIC_DIR / "review.html").read_text())


def _clip_completion(stem: str, video_path: Path) -> dict:
    """Return {marked, complete, label_count} for a clip.

    complete = reached near the last frame AND has at least one player mark.
    Ball-only clips (no player labels) are treated as in-progress.
    """
    jsonl = _GT_MARKS_DIR / f"{stem}.jsonl"
    if not jsonl.exists():
        return {"marked": False, "complete": False, "label_count": 0}
    try:
        frame_indices = []
        has_player = False
        with jsonl.open() as f:
            for raw_line in f:
                line = raw_line.strip()
                if not line:
                    continue
                d = json.loads(line)
                frame_indices.append(int(d.get("frame_index", 0)))
                tags = d.get("tags") or []
                if any(t in PLAYER_LABELS for t in tags):
                    has_player = True
        if not frame_indices:
            return {"marked": False, "complete": False, "label_count": 0}
        cap = cv2.VideoCapture(str(video_path))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        reached_end = total > 0 and max(frame_indices) >= total - 15
        complete = reached_end and has_player
        return {"marked": True, "complete": complete, "label_count": len(frame_indices)}
    except Exception:
        return {"marked": True, "complete": False, "label_count": 0}


@app.get("/clips")
async def list_clips() -> dict:
    """Return sorted list of clips immediately — no completion check."""
    if not _CLIPS_DIR.exists():
        return {"clips": []}
    video_suffixes = {".mp4", ".mov", ".avi", ".mkv"}
    paths = sorted(
        p for p in _CLIPS_DIR.iterdir() if p.suffix.lower() in video_suffixes
    )
    # Return names fast; completion checked lazily via /clips/status
    clips = [
        {"name": p.name, "marked": (_GT_MARKS_DIR / f"{p.stem}.jsonl").exists()}
        for p in paths
    ]
    return {"clips": clips, "dir": str(_CLIPS_DIR)}


@app.get("/clips/status")
async def clips_status() -> dict:
    """Return completion status for all clips — may be slow (reads video metadata)."""
    if not _CLIPS_DIR.exists():
        return {"clips": []}
    video_suffixes = {".mp4", ".mov", ".avi", ".mkv"}
    paths = sorted(
        p for p in _CLIPS_DIR.iterdir() if p.suffix.lower() in video_suffixes
    )
    clips = [{"name": p.name, **_clip_completion(p.stem, p)} for p in paths]
    return {"clips": clips}


@app.post("/session/load")
async def load_session(body: dict) -> dict:
    return SESSION.load(body["video_path"])


@app.get("/frame/{idx}.jpg")
async def get_frame(idx: int) -> Response:
    data = SESSION.frame_jpeg(idx)
    if data is None:
        return Response(status_code=404)
    return Response(content=data, media_type="image/jpeg")


@app.get("/marks")
async def get_marks() -> dict:
    """Return no_ball, not_broadcast, ball, and player frame sets for the current session."""
    with SESSION._tl_lock:
        ball_frames = []
        player_frames = []
        for idx, boxes in enumerate(SESSION.timeline):
            if not boxes:
                continue
            has_ball = any(b.label in BALL_LABELS for b in boxes)
            has_player = any(b.label in PLAYER_LABELS for b in boxes)
            if has_ball:
                ball_frames.append(idx)
            if has_player:
                player_frames.append(idx)
    return {
        "no_ball": sorted(SESSION.no_ball_frames),
        "not_broadcast": sorted(SESSION.not_broadcast_frames),
        "ball": ball_frames,
        "player": player_frames,
    }


@app.post("/autodetect")
async def autodetect(body: dict) -> dict:
    """Run YOLO on a frame and merge with what the client currently shows.

    The client sends its current canvas boxes as `current_boxes`. These become
    the labeller ground truth for this frame; YOLO detections are added on top.
    This means autodetect only adds to what you can see — it never pulls in
    stale boxes from earlier sessions.
    """
    if SESSION.video_path is None:
        return {"idx": 0, "boxes": []}
    idx = int(body.get("frame_idx", 0))

    # Treat whatever the client currently shows as the labeller ground truth.
    current = boxes_from_payload(body.get("current_boxes", []), PROV_LABELLER)
    SESSION.set_frame(idx, current)

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

    # Suppress YOLO boxes that overlap heavily with existing labeller boxes.
    def _xywh(b: ObjectDetection) -> tuple:
        return (b.x, b.y, b.w, b.h)

    filtered_yolo = [
        yb
        for yb in yolo_boxes
        if not any(bbox_iou(_xywh(yb), _xywh(cb)) > 0.3 for cb in current)
    ]
    SESSION.set_frame(idx, current + filtered_yolo)
    return {"idx": idx, "boxes": boxes_payload(SESSION.get_frame(idx))}


@app.get("/timeline/{idx}")
async def get_timeline(idx: int) -> dict:
    """Return the authoritative boxes (with provenance) for a frame."""
    return {"idx": idx, "boxes": boxes_payload(SESSION.get_frame(idx))}


@app.get("/next-detection/{from_idx}")
async def next_detection(from_idx: int) -> dict:
    """Return the next frame after from_idx that has at least one box."""
    with SESSION._tl_lock:
        tl = list(SESSION.timeline)
    for idx in range(from_idx + 1, len(tl)):
        if tl[idx]:
            return {"idx": idx}
    return {"idx": None}


@app.post("/edit")
async def edit_frame(body: dict) -> dict:
    """Overwrite a frame with the user's boxes (labeller provenance = ground truth)."""
    idx = int(body["idx"])
    boxes = boxes_from_payload(body.get("objects", []), PROV_LABELLER)
    SESSION.set_frame(idx, boxes)
    if boxes:
        SESSION.no_ball_frames.discard(idx)
        SESSION.not_broadcast_frames.discard(idx)
    SESSION.schedule_flush()
    return {"idx": idx, "boxes": boxes_payload(SESSION.get_frame(idx))}


@app.post("/no-ball")
async def mark_no_ball(body: dict) -> dict:
    """Mark a frame as no-ball-visible; removes any ball boxes from that frame."""
    idx = int(body["idx"])
    SESSION.no_ball_frames.add(idx)
    # Clear ball boxes from this frame so the JSONL is consistent.
    with SESSION._tl_lock:
        if 0 <= idx < len(SESSION.timeline) and SESSION.timeline[idx]:
            SESSION.timeline[idx] = [
                b for b in SESSION.timeline[idx] if b.label not in BALL_LABELS
            ]
    SESSION.schedule_flush()
    return {"idx": idx, "no_ball": True}


@app.post("/not-broadcast")
async def mark_not_broadcast(body: dict) -> dict:
    idx = int(body["idx"])
    SESSION.not_broadcast_frames.add(idx)
    SESSION.schedule_flush()
    return {"idx": idx, "not_broadcast": True}


@app.post("/no-ball/clear")
async def clear_no_ball(body: dict) -> dict:
    idx = int(body["idx"])
    SESSION.no_ball_frames.discard(idx)
    SESSION.schedule_flush()
    return {"idx": idx, "no_ball": False}


@app.post("/not-broadcast/clear")
async def clear_not_broadcast(body: dict) -> dict:
    idx = int(body["idx"])
    SESSION.not_broadcast_frames.discard(idx)
    SESSION.schedule_flush()
    return {"idx": idx, "not_broadcast": False}


@app.post("/propagate")
async def propagate_labels(body: dict) -> dict:
    """Propagate a single labeller-provenance label forward through subsequent frames.

    Matches YOLO-provenance boxes by IoU (threshold 0.3) to propagate a corrected
    label forward frame by frame. Stops when no match found (track lost) or when
    a labeller-provenance box already exists at that location.

    Request body:
    - frame_idx: source frame containing the labeller box
    - box_idx: index of the labeller box within that frame

    Returns:
    - propagated_to: number of frames the label was propagated to
    """
    frame_idx = int(body.get("frame_idx", 0))
    box_idx = int(body.get("box_idx", 0))
    iou_threshold = 0.3

    # Get the labeller box at frame_idx, box_idx
    frame_boxes = SESSION.get_frame(frame_idx)
    if box_idx >= len(frame_boxes):
        return {"propagated_to": 0}

    ref_box = frame_boxes[box_idx]
    if ref_box.model != PROV_LABELLER:
        # Only propagate labeller-provenance boxes
        return {"propagated_to": 0}

    propagated_count = 0
    last_position = (ref_box.x, ref_box.y, ref_box.w, ref_box.h)
    ref_label = ref_box.label

    # Walk forward frame by frame
    for idx in range(frame_idx + 1, SESSION.total_frames):
        frame_boxes = SESSION.get_frame(idx)
        if not frame_boxes:
            continue

        # Stop if labeller-provenance box already exists at this location
        has_labeller = any(b.model == PROV_LABELLER for b in frame_boxes)
        if has_labeller:
            break

        # Find YOLO-provenance box with highest IoU
        best_iou = -1.0
        best_box_idx = -1

        for box_idx_local, b in enumerate(frame_boxes):
            if b.model != PROV_YOLO:
                continue
            iou = bbox_iou(last_position, (b.x, b.y, b.w, b.h))
            if iou > best_iou:
                best_iou = iou
                best_box_idx = box_idx_local

        # Stop if no match
        if best_iou < iou_threshold:
            break

        # Update the matched box's label and provenance
        best_box = frame_boxes[best_box_idx]
        updated_boxes = []
        for j, b in enumerate(frame_boxes):
            if j == best_box_idx:
                # Override label, keep YOLO provenance (it's still YOLO-detected, just corrected)
                updated_boxes.append(
                    ObjectDetection(
                        label=ref_label,
                        confidence=b.confidence,
                        x=b.x,
                        y=b.y,
                        w=b.w,
                        h=b.h,
                        model=PROV_YOLO,
                    )
                )
            else:
                updated_boxes.append(b)

        SESSION.set_frame(idx, updated_boxes)
        last_position = (best_box.x, best_box.y, best_box.w, best_box.h)
        propagated_count += 1

    SESSION.schedule_flush()
    return {"propagated_to": propagated_count}


# ----------------------------------------------------------------------------
# Ingest: upload a match video → split_broadcast_segments → eval_data/clips
# ----------------------------------------------------------------------------

_INGEST_UPLOADS = Path(tempfile.gettempdir()) / "footy_ingest_uploads"
_INGEST_UPLOADS.mkdir(parents=True, exist_ok=True)

# Module-level singleton so the FastAPI param default isn't a call expression (B008).
_UPLOAD_FILE = File(...)


@app.get("/ingest", response_class=HTMLResponse)
async def ingest_page() -> HTMLResponse:
    return HTMLResponse((_STATIC_DIR / "ingest.html").read_text())


@app.post("/ingest/upload")
async def ingest_upload(file: UploadFile = _UPLOAD_FILE) -> dict:
    """Save uploaded video to a temp location and return the path."""
    dest = _INGEST_UPLOADS / (file.filename or "upload.mp4")
    data = await file.read()
    dest.write_bytes(data)
    return {"path": str(dest), "name": dest.name, "size": len(data)}


@app.get("/ingest/run")
async def ingest_run(
    path: str, sample: int = 5, merge_gap_s: float = 0.5, min_seg_s: float = 2.0
) -> StreamingResponse:
    """Stream split_broadcast_segments output as SSE."""
    video_path = Path(path).expanduser()

    async def event_stream():
        if not video_path.exists():
            yield f"data: ERROR: file not found: {video_path}\n\n"
            yield "data: [DONE]\n\n"
            return

        cmd = [
            sys.executable,
            "-m",
            "footy_track.scripts.split_broadcast_segments",
            str(video_path),
            "--outdir",
            str(_CLIPS_DIR),
            "--sample",
            str(sample),
            "--merge-gap-s",
            str(merge_gap_s),
            "--min-seg-s",
            str(min_seg_s),
        ]
        yield f"data: Running: {' '.join(cmd)}\n\n"

        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
        )
        assert proc.stdout is not None
        async for raw in proc.stdout:
            line = raw.decode(errors="replace").rstrip()
            if line:
                yield f"data: {line}\n\n"

        rc = await proc.wait()
        yield f"data: [EXIT {rc}]\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ----------------------------------------------------------------------------
# WebSocket: control (run/pause/restart) + live frame stream
# ----------------------------------------------------------------------------


def _ingest_completed_frame(
    idx: int, fd, start_idx: int
) -> tuple[list[ObjectDetection], bool]:
    """Write a propagated frame into the timeline and return (boxes, gt_kept).

    The seed frame (idx == start_idx) is the ground-truth seed — its timeline
    entry is already correct, so we don't merge SAM3's re-segmentation into it.
    Downstream frames get SAM3 boxes merged in (keeping labeller ground truth);
    ``gt_kept`` is True when existing GT made the frame skip the merge.
    """
    if idx == start_idx:
        return SESSION.get_frame(idx), False
    vittrack_boxes = [
        ObjectDetection(
            label=d.label,
            confidence=d.confidence,
            x=d.x,
            y=d.y,
            w=d.w,
            h=d.h,
            model=PROV_VITTRACK,
        )
        for d in fd.detections
    ]
    gt_kept = SESSION.merge_propagated(idx, vittrack_boxes)
    return SESSION.get_frame(idx), gt_kept


async def _stream_frames(websocket: WebSocket, start_idx: int) -> None:
    """Push each newly-completed frame to the client until the run stops."""
    sent = start_idx - 1
    await websocket.send_json({"type": "status", "state": "compiling"})
    announced_running = False
    while True:
        cur_bg = SESSION.bg  # may swap on a fresh load
        while sent < cur_bg.last_completed_frame:
            sent += 1
            # frame_at handles mid-clip runs: completed_frames() only scans the
            # contiguous run from frame 0, so a run seeded at frame N (with
            # frames 0..N-1 still None) would silently skip every frame —
            # nothing ingested into the timeline, nothing streamed (the bug
            # behind "ran to frame 30 but 28-29 have no boxes").
            fd = cur_bg.frame_at(sent)
            if fd is not None:
                if not announced_running:
                    await websocket.send_json({"type": "status", "state": "running"})
                    announced_running = True
                boxes, gt_kept = _ingest_completed_frame(sent, fd, start_idx)
                await websocket.send_json(
                    {
                        "type": "frame",
                        "idx": sent,
                        "boxes": boxes_payload(boxes),
                        "gt_kept": gt_kept,
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
                # Tell the client we're entering the (potentially slow) model
                # load/warmup phase *before* blocking on it — previously this
                # only happened inside _stream_frames, whose task wasn't
                # created until after bg.submit() had already returned, so the
                # "compiling" status arrived too late to be useful and the
                # frontend had no reliable signal to show a loading overlay
                # for the duration of the actual compile (ft-wkc).
                await websocket.send_json({"type": "status", "state": "compiling"})
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


app.include_router(review_router)

# Mount static assets (JS/CSS) if any beyond index.html.
if _STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")
