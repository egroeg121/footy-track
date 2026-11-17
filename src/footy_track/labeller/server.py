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
import threading
import time
from pathlib import Path

# Persist torch.compile (Inductor) cache before torch loads — see video_utils.
os.environ.setdefault(
    "TORCHINDUCTOR_CACHE_DIR",
    str(Path.home() / ".cache" / "footy_torch_inductor"),
)

import cv2  # noqa: E402
from fastapi import (  # noqa: E402
    FastAPI,
    HTTPException,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.responses import HTMLResponse, Response  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402

from footy_track.ball_eval.metrics import bbox_iou  # noqa: E402
from footy_track.labeller import datasets  # noqa: E402
from footy_track.labeller.ball_check import router as ball_check_router  # noqa: E402
from footy_track.labeller.calibration import router as calibration_router  # noqa: E402
from footy_track.labeller.calibration_proposals import router as calibration_proposals_router  # noqa: E402
from footy_track.labeller.constants import (  # noqa: E402
    BALL_LABELS,
    PLAYER_LABELS,
    PROV_LABELLER,
    PROV_SAM3,
    PROV_VITTRACK,
    PROV_YOLO,
)
from footy_track.labeller.ingest import router as ingest_router  # noqa: E402
from footy_track.labeller.jobs_routes import router as jobs_router  # noqa: E402
from footy_track.labeller.review import router as review_router  # noqa: E402
from footy_track.labeller.tasks import router as tasks_router  # noqa: E402
from footy_track.labeller.run_stream import stream_frames  # noqa: E402
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
# ?dataset=<name> -> the request's dataset (HTTP and WebSocket); unknown name = 404.
app.add_middleware(datasets.DatasetMiddleware)

_STATIC_DIR = Path(__file__).parent / "web"


from footy_track.labeller import identity_routes  # noqa: E402

# Identity review lives on its own page and shares no session state with the
# box labeller: different unit (tracklets, not boxes), different scope (two
# clips at once), and append-only storage rather than whole-file rewrites.
app.include_router(identity_routes.router)

from footy_track.labeller import matches_routes  # noqa: E402

app.include_router(matches_routes.router)

#: Session of the default dataset. Other datasets get their own Session lazily
#: (``_session()``): one shared Session would flush clip A's timeline into the
#: wrong dataset's marks dir the moment two datasets are in use.
SESSION = Session()
_SESSIONS: dict[str, Session] = {}
_SESSIONS_LOCK = threading.Lock()


def _session() -> Session:
    """The Session of the request's dataset (see ``datasets.DatasetMiddleware``)."""
    ds = datasets.current()
    if ds.is_default:
        return SESSION
    with _SESSIONS_LOCK:
        sess = _SESSIONS.get(ds.name)
        if sess is None:
            sess = _SESSIONS[ds.name] = Session(ds.name)
        return sess


# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------
# HTTP endpoints
# ----------------------------------------------------------------------------


@app.middleware("http")
async def _cors_for_read_only_api(request, call_next):
    """Let the home page of one labeller server read ``/api/...`` GETs of another.

    The home page lists every service's ``/api/server_info`` and task counts
    cross-origin (:8011 <-> :8012). Read-only GETs only; no auth exists anyway.
    """
    response = await call_next(request)
    if request.method == "GET" and "/api/" in request.url.path:
        response.headers["Access-Control-Allow-Origin"] = "*"
    return response


@app.get("/api/datasets")
async def api_datasets() -> dict:
    """Registered datasets (see ``datasets.py``); the home page picker reads this."""
    out: dict = {
        "default": datasets.default_name(),
        "current": datasets.current_name(),
        "registry": str(datasets.registry_path()),
        "datasets": [d.to_public() for d in datasets.all_datasets()],
    }
    if datasets.registry_error():
        out["error"] = datasets.registry_error()
    return out


@app.get("/", response_class=HTMLResponse)
@app.get("/main", response_class=HTMLResponse)
async def root() -> HTMLResponse:
    # no-store: the task board must never be a cached copy on a phone.
    return HTMLResponse(
        (_STATIC_DIR / "main.html").read_text(),
        headers={"Cache-Control": "no-store, max-age=0"},
    )


@app.get("/labeller", response_class=HTMLResponse)
async def labeller_page() -> HTMLResponse:
    return HTMLResponse((_STATIC_DIR / "index.html").read_text())


@app.get("/object_review", response_class=HTMLResponse)
async def review_page() -> HTMLResponse:
    return HTMLResponse((_STATIC_DIR / "review.html").read_text())


@app.get("/ball_check", response_class=HTMLResponse)
async def ball_check_page() -> HTMLResponse:
    # no-store: a phone holding a cached copy of this page kept POSTing to an
    # endpoint that had been replaced, which is invisible from the server log
    # because the old route still answers 200.
    return HTMLResponse(
        (_STATIC_DIR / "ball_check.html").read_text(),
        headers={"Cache-Control": "no-store, max-age=0"},
    )


@app.get("/calibration", response_class=HTMLResponse)
async def calibration_page() -> HTMLResponse:
    return HTMLResponse(
        (_STATIC_DIR / "calibration.html").read_text(),
        headers={"Cache-Control": "no-store, max-age=0"},
    )


@app.get("/calibration/review", response_class=HTMLResponse)
async def calibration_review_page() -> HTMLResponse:
    return HTMLResponse(
        (_STATIC_DIR / "calibration_review.html").read_text(),
        headers={"Cache-Control": "no-store, max-age=0"},
    )


@app.get("/ingest", response_class=HTMLResponse)
async def ingest_page() -> HTMLResponse:
    return HTMLResponse((_STATIC_DIR / "ingest.html").read_text())


def _clip_completion(stem: str, video_path: Path) -> dict:
    """Return {marked, complete, label_count} for a clip.

    complete = reached near the last frame AND has at least one player mark.
    Ball-only clips (no player labels) are treated as in-progress.
    """
    jsonl = datasets.current().marks_dir / f"{stem}.jsonl"
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
    if not datasets.current().clips_dir.exists():
        return {"clips": []}
    video_suffixes = {".mp4", ".mov", ".avi", ".mkv"}
    paths = sorted(
        p
        for p in datasets.current().clips_dir.iterdir()
        if p.suffix.lower() in video_suffixes and not p.name.startswith(".")
    )
    # FOOTY_MAX_CLIPS keeps startup cheap on a small development box (--debug).
    max_clips = os.environ.get("FOOTY_MAX_CLIPS")
    if max_clips and max_clips.isdigit() and int(max_clips) > 0:
        paths = paths[: int(max_clips)]
    # Return names fast; completion checked lazily via /clips/status
    clips = [
        {"name": p.name, "marked": (datasets.current().marks_dir / f"{p.stem}.jsonl").exists()}
        for p in paths
    ]
    return {"clips": clips, "dir": str(datasets.current().clips_dir)}


@app.get("/clips/status")
async def clips_status() -> dict:
    """Return completion status for all clips — may be slow (reads video metadata)."""
    if not datasets.current().clips_dir.exists():
        return {"clips": []}
    video_suffixes = {".mp4", ".mov", ".avi", ".mkv"}
    paths = sorted(
        p
        for p in datasets.current().clips_dir.iterdir()
        if p.suffix.lower() in video_suffixes and not p.name.startswith(".")
    )
    clips = [{"name": p.name, **_clip_completion(p.stem, p)} for p in paths]
    return {"clips": clips}


@app.post("/session/load")
async def load_session(body: dict) -> dict:
    session = _session()
    # load() pauses the previous tracker (a thread join): keep it off the loop.
    return await asyncio.to_thread(session.load, body["video_path"])


@app.get("/frame/{idx}.jpg")
async def get_frame(idx: int) -> Response:
    session = _session()
    # Off the event loop: seeking + decoding a 1080p frame takes tens to
    # hundreds of ms (more on a throttled CPU), and a live run fetches one per
    # streamed frame — inline, that queue starves the WebSocket and Pause.
    data = await asyncio.to_thread(session.frame_jpeg, idx)
    if data is None:
        return Response(status_code=404)
    return Response(content=data, media_type="image/jpeg")


@app.get("/marks")
async def get_marks() -> dict:
    """Return no_ball, not_broadcast, ball, and player frame sets for the current session."""
    session = _session()
    with session._tl_lock:
        ball_frames = []
        player_frames = []
        for idx, boxes in enumerate(session.timeline):
            if not boxes:
                continue
            has_ball = any(b.label in BALL_LABELS for b in boxes)
            has_player = any(b.label in PLAYER_LABELS for b in boxes)
            if has_ball:
                ball_frames.append(idx)
            if has_player:
                player_frames.append(idx)
    return {
        "no_ball": sorted(session.no_ball_frames),
        "not_broadcast": sorted(session.not_broadcast_frames),
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
    session = _session()
    if session.video_path is None:
        return {"idx": 0, "boxes": []}
    idx = int(body.get("frame_idx", 0))

    # Treat whatever the client currently shows as the labeller ground truth.
    current = boxes_from_payload(body.get("current_boxes", []), PROV_LABELLER)
    session.set_frame(idx, current)

    try:
        seeds = await asyncio.to_thread(
            yolo_seed_objects,
            session.video_path,
            body.get("model_path", "") or "",
            float(body.get("conf", 0.35)),
            session.width,
            session.height,
            float(body.get("iou", 0.5)),
            idx,
        )
    except FileNotFoundError as exc:
        # Detector weights are not in git; say so instead of an opaque 500.
        raise HTTPException(
            status_code=503, detail=f"detector weights not found: {exc.filename or exc}"
        ) from exc
    w, h = session.width, session.height
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
    session.set_frame(idx, current + filtered_yolo)
    return {"idx": idx, "boxes": boxes_payload(session.get_frame(idx))}


@app.get("/timeline/{idx}")
async def get_timeline(idx: int) -> dict:
    """Return the authoritative boxes (with provenance) for a frame."""
    session = _session()
    return {"idx": idx, "boxes": boxes_payload(session.get_frame(idx))}


@app.get("/next-detection/{from_idx}")
async def next_detection(from_idx: int) -> dict:
    """Return the next frame after from_idx that has at least one box."""
    session = _session()
    with session._tl_lock:
        tl = list(session.timeline)
    for idx in range(from_idx + 1, len(tl)):
        if tl[idx]:
            return {"idx": idx}
    return {"idx": None}


@app.post("/edit")
async def edit_frame(body: dict) -> dict:
    """Overwrite a frame with the user's boxes (labeller provenance = ground truth)."""
    session = _session()
    idx = int(body["idx"])
    boxes = boxes_from_payload(body.get("objects", []), PROV_LABELLER)
    session.set_frame(idx, boxes)
    if boxes:
        session.no_ball_frames.discard(idx)
        session.not_broadcast_frames.discard(idx)
    session.schedule_flush()
    return {"idx": idx, "boxes": boxes_payload(session.get_frame(idx))}


@app.post("/no-ball")
async def mark_no_ball(body: dict) -> dict:
    """Mark a frame as no-ball-visible; removes any ball boxes from that frame."""
    session = _session()
    idx = int(body["idx"])
    session.no_ball_frames.add(idx)
    # Clear ball boxes from this frame so the JSONL is consistent.
    with session._tl_lock:
        if 0 <= idx < len(session.timeline) and session.timeline[idx]:
            session.timeline[idx] = [
                b for b in session.timeline[idx] if b.label not in BALL_LABELS
            ]
    session.schedule_flush()
    return {"idx": idx, "no_ball": True}


@app.post("/not-broadcast")
async def mark_not_broadcast(body: dict) -> dict:
    session = _session()
    idx = int(body["idx"])
    session.not_broadcast_frames.add(idx)
    session.schedule_flush()
    return {"idx": idx, "not_broadcast": True}


@app.post("/no-ball/clear")
async def clear_no_ball(body: dict) -> dict:
    session = _session()
    idx = int(body["idx"])
    session.no_ball_frames.discard(idx)
    session.schedule_flush()
    return {"idx": idx, "no_ball": False}


@app.post("/not-broadcast/clear")
async def clear_not_broadcast(body: dict) -> dict:
    session = _session()
    idx = int(body["idx"])
    session.not_broadcast_frames.discard(idx)
    session.schedule_flush()
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
    session = _session()
    frame_idx = int(body.get("frame_idx", 0))
    box_idx = int(body.get("box_idx", 0))
    iou_threshold = 0.3

    # Get the labeller box at frame_idx, box_idx
    frame_boxes = session.get_frame(frame_idx)
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
    for idx in range(frame_idx + 1, session.total_frames):
        frame_boxes = session.get_frame(idx)
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

        session.set_frame(idx, updated_boxes)
        last_position = (best_box.x, best_box.y, best_box.w, best_box.h)
        propagated_count += 1

    session.schedule_flush()
    return {"propagated_to": propagated_count}


# ----------------------------------------------------------------------------
# WebSocket: control (run/pause/restart) + live frame stream
# ----------------------------------------------------------------------------


def _gt_reseed_frames(session: Session, start_frame: int) -> dict[int, list]:
    """Downstream frames with hand-fixed labels -> their full seed objects.

    A passing run re-seeds at these frames (LAB-714), so human corrections
    steer propagation instead of being displayed-but-ignored.
    """
    out: dict[int, list] = {}
    for idx in range(start_frame + 1, session.total_frames):
        boxes = session.get_frame(idx)
        if any(b.model == PROV_LABELLER for b in boxes):
            objects = session.seed_objects(idx)
            if objects:
                out[idx] = objects
    return out


def _sanitize_thresholds(raw: object) -> dict[str, float]:
    """Client handback thresholds -> clean {label: conf in [0,1]} (LAB-610).

    Non-dict input and non-numeric values are dropped; numerics are clamped.
    """
    if not isinstance(raw, dict):
        return {}
    out: dict[str, float] = {}
    for key, value in raw.items():
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            out[str(key)] = min(1.0, max(0.0, float(value)))
    return out


@app.websocket("/ws")
async def ws(websocket: WebSocket) -> None:  # noqa: PLR0915 — one dispatch loop
    session = _session()
    await websocket.accept()
    run_bg = session.bg  # the tracker this connection's current run drives
    streamer: asyncio.Task | None = None
    starter: asyncio.Task | None = None  # run/restart still submitting

    async def _run_streamer(start_idx: int) -> None:
        await stream_frames(websocket, session, start_idx)

    async def _cancel(task: asyncio.Task | None) -> None:
        if task is not None and not task.done():
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task

    def _request_stop() -> None:
        # Non-blocking: just raises the tracker's stop flag(s). session.bg may
        # have been swapped by a clip load since this run started.
        for bg in {id(b): b for b in (run_bg, session.bg)}.values():
            bg.request_stop()

    async def _start_run(msg: dict, start_frame: int, objects: list) -> None:
        nonlocal streamer
        # Tell the client we're entering the (potentially slow) model
        # load/warmup phase *before* blocking on it — previously this
        # only happened inside _stream_frames, whose task wasn't
        # created until after bg.submit() had already returned, so the
        # "compiling" status arrived too late to be useful and the
        # frontend had no reliable signal to show a loading overlay
        # for the duration of the actual compile (ft-wkc).
        await websocket.send_json({"type": "status", "state": "compiling"})
        await asyncio.to_thread(
            run_bg.submit,
            session.video_path,
            objects,
            msg.get("model_uri") or None,
            float(msg.get("conf", 0.25)),
            start_frame,
            int(msg.get("imgsz", 512)),
            _sanitize_thresholds(msg.get("handback_thresholds")),
            _gt_reseed_frames(session, start_frame),
        )
        streamer = asyncio.create_task(_run_streamer(start_frame))

    try:
        while True:
            msg = await websocket.receive_json()
            mtype = msg.get("type")

            if mtype in ("run", "restart"):
                _request_stop()
                await _cancel(starter)
                await _cancel(streamer)
                run_bg = session.bg
                start_frame = int(msg.get("start_frame", 0))
                # Seed from the TIMELINE at the start frame (the single source of
                # truth) — not from whatever the client happens to send. This is
                # what makes "restart from frame N" deterministic.
                objects = session.seed_objects(start_frame)
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
                # Submit runs as a task so this loop keeps reading: a Pause sent
                # while the run is still starting must not wait behind it.
                starter = asyncio.create_task(_start_run(msg, start_frame, objects))

            elif mtype == "pause":
                t0 = time.monotonic()
                print("[ws] pause: received", flush=True)
                # 1. Raise the stop flag now: the tracker stops within one
                #    object's step, whatever happens next.
                _request_stop()
                # 2. Stop streaming: nothing further reaches the client.
                await _cancel(starter)
                await _cancel(streamer)
                # 3. Wait for the tracker thread to exit, OFF the event loop
                #    (pause() joins for up to 10 s; run inline it froze every
                #    HTTP request and this socket's own reads meanwhile).
                for bg in {id(b): b for b in (run_bg, session.bg)}.values():
                    await asyncio.to_thread(bg.pause)
                await websocket.send_json({"type": "status", "state": "paused"})
                print(
                    f"[ws] pause: stopped in {time.monotonic() - t0:.2f}s",
                    flush=True,
                )

    except WebSocketDisconnect:
        print("[ws] disconnect: stopping run", flush=True)
    finally:
        _request_stop()
        for task in (starter, streamer):
            if task is not None and not task.done():
                task.cancel()


app.include_router(review_router)
app.include_router(ball_check_router)
app.include_router(calibration_router)
app.include_router(calibration_proposals_router)
app.include_router(ingest_router)
app.include_router(tasks_router)
app.include_router(jobs_router)


async def _jobs_gc_loop(interval: float) -> None:
    """Hourly housekeeping (stale prepare runs, idle/finished renders, render-cache cap). Lives in the
    server so it survives restarts with no extra scheduler; ``labelling_jobs gc`` runs the same code."""
    import logging  # noqa: PLC0415

    from footy_track.labeller import jobs as _jobs  # noqa: PLC0415

    await asyncio.sleep(min(60.0, interval))
    while True:
        try:
            await asyncio.to_thread(_jobs.gc)
        except Exception:  # noqa: BLE001 - housekeeping must never take the server down
            logging.getLogger(__name__).exception("jobs gc failed")
        await asyncio.sleep(interval)


@app.on_event("startup")
async def _start_jobs_gc() -> None:
    interval = float(os.environ.get("FOOTY_JOBS_GC_INTERVAL_S", "3600") or 0)
    if interval > 0:
        app.state.jobs_gc_task = asyncio.create_task(_jobs_gc_loop(interval))


@app.on_event("shutdown")
async def _stop_jobs_gc() -> None:
    task = getattr(app.state, "jobs_gc_task", None)
    if task is not None:
        task.cancel()


# Mount static assets (JS/CSS) if any beyond index.html.
if _STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")
