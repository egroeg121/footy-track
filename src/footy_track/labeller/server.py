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
import collections
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
    NO_BALL_TAG,
    NOT_BROADCAST_TAG,
    PLAYER_LABELS,
    PROV_LABELLER,
    PROV_SAM3,
    PROV_VITTRACK,
    PROV_YOLO,
)
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
# Review UI: tinder-style crop correction
# ----------------------------------------------------------------------------

# All classes that can appear in the review picker.
_ALL_LABELS = [
    "player",
    "in_play_ball",
    "out_of_play_ball",
    "referee",
    "coach",
    "player_sub",
    "ball",
]

# LRU crop cache: key = (clip_stem, frame_idx, box_idx), value = JPEG bytes
_CROP_CACHE: collections.OrderedDict[tuple, bytes] = collections.OrderedDict()
_CROP_CACHE_MAX = 200


def _get_cached_crop(key: tuple) -> bytes | None:
    if key in _CROP_CACHE:
        _CROP_CACHE.move_to_end(key)
        return _CROP_CACHE[key]
    return None


def _put_cached_crop(key: tuple, data: bytes) -> None:
    _CROP_CACHE[key] = data
    _CROP_CACHE.move_to_end(key)
    if len(_CROP_CACHE) > _CROP_CACHE_MAX:
        _CROP_CACHE.popitem(last=False)


_ALL_LABELS_SET = set(_ALL_LABELS)
_PROV_TAGS = {PROV_LABELLER, PROV_YOLO, PROV_SAM3}


def _parse_jsonl_box(raw: str, clip_stem: str, video_path: Path | None) -> dict | None:
    """Parse one JSONL line into a box record dict, or return None to skip."""
    try:
        d = json.loads(raw)
    except json.JSONDecodeError:
        return None
    tags = d.get("tags") or []
    if NO_BALL_TAG in tags or NOT_BROADCAST_TAG in tags:
        return None
    bbox = d.get("bbox")
    if bbox is None:
        return None
    frame_idx = int(d.get("frame_index", -1))
    if frame_idx < 0:
        return None
    if isinstance(bbox, dict):
        bx, by, bw, bh = bbox["x"], bbox["y"], bbox["w"], bbox["h"]
    else:
        bx, by, bw, bh = bbox
    label = next((t for t in tags if t in _ALL_LABELS_SET), "player")
    confidence = 1.0 if PROV_LABELLER in tags else 0.5
    provenance = next((t for t in tags if t in _PROV_TAGS), PROV_LABELLER)
    return {
        "clip": clip_stem,
        "video_path": str(video_path) if video_path else None,
        "frame_index": frame_idx,
        "bbox": {"x": bx, "y": by, "w": bw, "h": bh},
        "label": label,
        "confidence": confidence,
        "provenance": provenance,
    }


def _find_video(clip_stem: str) -> Path | None:
    if not _CLIPS_DIR.exists():
        return None
    for ext in (".mp4", ".mov", ".avi", ".mkv"):
        candidate = _CLIPS_DIR / f"{clip_stem}{ext}"
        if candidate.exists():
            return candidate
    return None


def _read_all_boxes() -> list[dict]:
    """Scan all JSONL sidecars and return a flat list of box records."""
    records: list[dict] = []
    if not _GT_MARKS_DIR.exists():
        return records
    for jsonl_path in sorted(_GT_MARKS_DIR.glob("*.jsonl")):
        clip_stem = jsonl_path.stem
        video_path = _find_video(clip_stem)
        try:
            lines = jsonl_path.read_text().splitlines()
        except OSError:
            continue
        frame_boxes: dict[int, list[dict]] = {}
        for raw_line in lines:
            stripped = raw_line.strip()
            if not stripped:
                continue
            rec = _parse_jsonl_box(stripped, clip_stem, video_path)
            if rec is None:
                continue
            frame_boxes.setdefault(rec["frame_index"], []).append(rec)
        for _frame_idx, boxes in sorted(frame_boxes.items()):
            for box_idx, rec in enumerate(boxes):
                rec["box_index"] = box_idx
                records.append(rec)
    return records


def _build_review_queue(records: list[dict]) -> list[dict]:
    """Order and dedup records: low-confidence YOLO first, rare classes weighted up, dedup by IoU."""
    CLASS_WEIGHT = {
        "ball": 3,
        "in_play_ball": 3,
        "out_of_play_ball": 3,
        "referee": 2,
        "coach": 2,
        "player_sub": 2,
    }

    def sort_key(r: dict) -> tuple:
        prov_order = 0 if r["provenance"] != PROV_LABELLER else 1
        weight = -CLASS_WEIGHT.get(r["label"], 1)
        conf = r["confidence"]
        return (prov_order, conf, weight)

    sorted_records = sorted(records, key=sort_key)

    # IoU dedup: skip crops near-identical to an already-queued item from the same frame
    IOU_THRESH = 0.85
    seen: dict[tuple, list[tuple]] = {}  # (clip, frame_index) -> list of (x,y,w,h)
    result: list[dict] = []
    for r in sorted_records:
        key = (r["clip"], r["frame_index"])
        b = r["bbox"]
        bx, by, bw, bh = b["x"], b["y"], b["w"], b["h"]
        existing = seen.get(key, [])
        duplicate = False
        for ex in existing:
            iou = bbox_iou((ex[0], ex[1], ex[2], ex[3]), (bx, by, bw, bh))
            if iou > IOU_THRESH:
                duplicate = True
                break
        if not duplicate:
            seen.setdefault(key, []).append((bx, by, bw, bh))
            result.append(r)
    return result


@app.get("/object_review", response_class=HTMLResponse)
async def review_page() -> HTMLResponse:
    return HTMLResponse((_STATIC_DIR / "review.html").read_text())


@app.get("/review/queue")
async def review_queue() -> dict:
    """Return an ordered list of ReviewItems for the tinder-style review UI."""
    records = await asyncio.to_thread(_read_all_boxes)
    queue = await asyncio.to_thread(_build_review_queue, records)
    # Strip video_path (internal) before sending to client
    items = []
    for r in queue:
        items.append(
            {
                "clip": r["clip"],
                "frame_index": r["frame_index"],
                "box_index": r["box_index"],
                "bbox": r["bbox"],
                "label": r["label"],
                "confidence": r["confidence"],
                "provenance": r["provenance"],
                "image_url": f"/review/crop/{r['clip']}/{r['frame_index']}/{r['box_index']}.jpg",
            }
        )
    return {"total": len(items), "items": items}


def _read_frame_box(
    clip_stem: str, frame_idx: int, box_idx: int
) -> tuple[float, float, float, float] | None:
    """Return (x, y, w, h) for box_idx within frame_idx of clip_stem's JSONL, or None."""
    jsonl_path = _GT_MARKS_DIR / f"{clip_stem}.jsonl"
    if not jsonl_path.exists():
        return None
    frame_boxes: list[dict] = []
    for raw_line in jsonl_path.read_text().splitlines():
        stripped = raw_line.strip()
        if not stripped:
            continue
        try:
            d = json.loads(stripped)
        except json.JSONDecodeError:
            continue
        tags = d.get("tags") or []
        if NO_BALL_TAG in tags or NOT_BROADCAST_TAG in tags:
            continue
        if d.get("bbox") is None:
            continue
        if int(d.get("frame_index", -1)) == frame_idx:
            frame_boxes.append(d)
    if box_idx >= len(frame_boxes):
        return None
    b = frame_boxes[box_idx]["bbox"]
    if isinstance(b, dict):
        return b["x"], b["y"], b["w"], b["h"]
    return tuple(b)  # type: ignore[return-value]


@app.get("/review/crop/{clip_stem}/{frame_idx}/{box_idx}.jpg")
async def review_crop(clip_stem: str, frame_idx: int, box_idx: int) -> Response:
    """Return a JPEG crop of a specific box from a specific frame."""
    cache_key = (clip_stem, frame_idx, box_idx)
    cached = _get_cached_crop(cache_key)
    if cached is not None:
        return Response(content=cached, media_type="image/jpeg")

    video_path = _find_video(clip_stem)
    if video_path is None:
        return Response(status_code=404)

    bbox_raw = _read_frame_box(clip_stem, frame_idx, box_idx)
    if bbox_raw is None:
        return Response(status_code=404)

    bx, by, bw, bh = bbox_raw

    def _crop_frame() -> bytes | None:
        cap = cv2.VideoCapture(str(video_path))
        try:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ok, frame = cap.read()
        finally:
            cap.release()
        if not ok:
            return None
        h_px, w_px = frame.shape[:2]
        pad = 2.0  # large context around box; grid cards will display at small size
        x1 = max(0, int((bx - bw * pad) * w_px))
        y1 = max(0, int((by - bh * pad) * h_px))
        x2 = min(w_px, int((bx + bw + bw * pad) * w_px))
        y2 = min(h_px, int((by + bh + bh * pad) * h_px))
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return None
        ok2, buf = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 85])
        return buf.tobytes() if ok2 else None

    data = await asyncio.to_thread(_crop_frame)
    if data is None:
        return Response(status_code=404)
    _put_cached_crop(cache_key, data)
    return Response(content=data, media_type="image/jpeg")


@app.post("/review/correct")
async def review_correct(body: dict) -> dict:
    """Write a corrected box (PROV_LABELLER) to the clip's JSONL sidecar."""
    clip = body["clip"]
    frame_index = int(body["frame_index"])
    box_index = int(body["box_index"])
    label = body["label"]
    bbox = body["bbox"]  # {x, y, w, h} normalized

    jsonl_path = _GT_MARKS_DIR / f"{clip}.jsonl"
    if not jsonl_path.exists():
        return {"ok": False, "error": "clip not found"}

    lines = jsonl_path.read_text().splitlines()
    frame_lines: list[tuple[int, str]] = []  # (line_index, original_line)
    for i, line in enumerate(lines):
        stripped = line.strip()
        if not stripped:
            continue
        try:
            d = json.loads(stripped)
        except json.JSONDecodeError:
            continue
        tags = d.get("tags") or []
        if NO_BALL_TAG in tags or NOT_BROADCAST_TAG in tags:
            continue
        if d.get("bbox") is None:
            continue
        if int(d.get("frame_index", -1)) == frame_index:
            frame_lines.append((i, stripped))

    if box_index >= len(frame_lines):
        return {"ok": False, "error": "box_index out of range"}

    line_idx, _ = frame_lines[box_index]
    bx = max(0.0, min(1.0, float(bbox["x"])))
    by = max(0.0, min(1.0, float(bbox["y"])))
    bw = max(0.0, min(1.0 - bx, float(bbox["w"])))
    bh = max(0.0, min(1.0 - by, float(bbox["h"])))
    cx, cy = bx + bw / 2, by + bh / 2
    new_record = json.dumps(
        {
            "frame_index": frame_index,
            "bbox": {"x": bx, "y": by, "w": bw, "h": bh},
            "center": {"x": cx, "y": cy},
            "tags": [label, PROV_LABELLER],
        }
    )
    lines[line_idx] = new_record
    _GT_MARKS_DIR.mkdir(parents=True, exist_ok=True)
    jsonl_path.write_text("\n".join(lines) + "\n")
    # Invalidate crop cache for this box
    _CROP_CACHE.pop((clip, frame_index, box_index), None)
    return {"ok": True}


@app.post("/review/delete")
async def review_delete(body: dict) -> dict:
    """Remove a box from the clip's JSONL sidecar."""
    clip = body["clip"]
    frame_index = int(body["frame_index"])
    box_index = int(body["box_index"])

    jsonl_path = _GT_MARKS_DIR / f"{clip}.jsonl"
    if not jsonl_path.exists():
        return {"ok": False, "error": "clip not found"}

    lines = jsonl_path.read_text().splitlines()
    frame_line_indices: list[int] = []
    for i, line in enumerate(lines):
        stripped = line.strip()
        if not stripped:
            continue
        try:
            d = json.loads(stripped)
        except json.JSONDecodeError:
            continue
        tags = d.get("tags") or []
        if NO_BALL_TAG in tags or NOT_BROADCAST_TAG in tags:
            continue
        if d.get("bbox") is None:
            continue
        if int(d.get("frame_index", -1)) == frame_index:
            frame_line_indices.append(i)

    if box_index >= len(frame_line_indices):
        return {"ok": False, "error": "box_index out of range"}

    del lines[frame_line_indices[box_index]]
    jsonl_path.write_text("\n".join(lines) + "\n")
    _CROP_CACHE.pop((clip, frame_index, box_index), None)
    return {"ok": True}


@app.get("/review/frame/{clip_stem}/{frame_idx}.jpg")
async def review_full_frame(clip_stem: str, frame_idx: int) -> Response:
    """Return the full JPEG frame (no crop) for the modal canvas."""
    video_path = _find_video(clip_stem)
    if video_path is None:
        return Response(status_code=404)

    def _read_frame() -> bytes | None:
        cap = cv2.VideoCapture(str(video_path))
        try:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ok, frame = cap.read()
        finally:
            cap.release()
        if not ok:
            return None
        ok2, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        return buf.tobytes() if ok2 else None

    data = await asyncio.to_thread(_read_frame)
    if data is None:
        return Response(status_code=404)
    return Response(content=data, media_type="image/jpeg")


@app.post("/review/yolo")
async def review_yolo(body: dict) -> dict:
    """Re-run YOLO on a specific frame and return detections."""
    clip_stem = body["clip"]
    frame_idx = int(body["frame_index"])
    video_path = _find_video(clip_stem)
    if video_path is None:
        return {"ok": False, "error": "video not found", "boxes": []}

    def _run() -> list[dict]:
        import tempfile as _tf  # noqa: PLC0415

        from footy_track.detectors.ultralytics import (  # noqa: PLC0415
            get_current_best_detector,
        )

        cap = cv2.VideoCapture(str(video_path))
        try:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ok, frame = cap.read()
        finally:
            cap.release()
        if not ok:
            return []
        h_px, w_px = frame.shape[:2]
        with _tf.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
            tmp = Path(f.name)
        cv2.imwrite(str(tmp), frame)
        try:
            detector = get_current_best_detector(min_confidence=0.25)
            fd = detector.predict_from_path(tmp)
        finally:
            tmp.unlink(missing_ok=True)
        return [
            {
                "label": d.label,
                "confidence": round(float(d.confidence), 3),
                "x": round(d.x, 4),
                "y": round(d.y, 4),
                "w": round(d.w, 4),
                "h": round(d.h, 4),
            }
            for d in fd.detections
        ]

    boxes = await asyncio.to_thread(_run)
    return {"ok": True, "boxes": boxes}


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


# Mount static assets (JS/CSS) if any beyond index.html.
if _STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")
