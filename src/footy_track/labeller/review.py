"""Review API: tinder-style crop correction over the JSONL sidecars.

Endpoints operate directly on the sidecar files (not the live Session
timeline). A box's identity is ``(clip, frame_index, box_index)`` where
``box_index`` is the ordinal of the box among that frame's box lines in
current file order (skip-marker and bbox-null lines excluded) — queue, crop,
correct, and delete all share this numbering. See
``src/footy_track/labeller/README.md`` §5 (Review API, LAB-4xx).

``server.py`` is the composition root and config surface: the clips and
GT-marks directories are resolved through it at call time (tests monkeypatch
them there).
"""

from __future__ import annotations

import asyncio
import collections
import json
import tempfile
from pathlib import Path

import cv2
from fastapi import APIRouter
from fastapi.responses import Response

from footy_track.ball_eval.metrics import bbox_iou
from footy_track.labeller.constants import (
    NO_BALL_TAG,
    NOT_BROADCAST_TAG,
    PROV_LABELLER,
    PROV_SAM3,
    PROV_YOLO,
    REVIEW_LABELS_SET,
)

router = APIRouter()


def _gt_marks_dir() -> Path:
    from footy_track.labeller import server  # noqa: PLC0415 — avoid import cycle

    return server._GT_MARKS_DIR


def _clips_dir() -> Path:
    from footy_track.labeller import server  # noqa: PLC0415 — avoid import cycle

    return server._CLIPS_DIR


# Provenance tags recognised when reading sidecar lines for review. NOTE:
# deliberately unchanged from the original implementation — 'vittrack' is
# absent, so vittrack boxes surface with provenance 'labeller' (OPEN-3 in
# src/footy_track/labeller/README.md, pinned by tests).
_PROV_TAGS = {PROV_LABELLER, PROV_YOLO, PROV_SAM3}

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


# ---------------------------------------------------------------------------
# Sidecar reading
# ---------------------------------------------------------------------------


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
    label = next((t for t in tags if t in REVIEW_LABELS_SET), "player")
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
    clips_dir = _clips_dir()
    if not clips_dir.exists():
        return None
    for ext in (".mp4", ".mov", ".avi", ".mkv"):
        candidate = clips_dir / f"{clip_stem}{ext}"
        if candidate.exists():
            return candidate
    return None


def _read_all_boxes() -> list[dict]:
    """Scan all JSONL sidecars and return a flat list of box records."""
    records: list[dict] = []
    marks_dir = _gt_marks_dir()
    if not marks_dir.exists():
        return records
    for jsonl_path in sorted(marks_dir.glob("*.jsonl")):
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


def _iter_frame_box_lines(lines: list[str], frame_index: int) -> list[int]:
    """Indices into ``lines`` of frame_index's box lines, in file order.

    Skip-marker and bbox-null lines are excluded — this is the shared
    ``box_index`` numbering used by queue/crop/correct/delete.
    """
    out: list[int] = []
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
            out.append(i)
    return out


def _read_frame_box(
    clip_stem: str, frame_idx: int, box_idx: int
) -> tuple[float, float, float, float] | None:
    """Return (x, y, w, h) for box_idx within frame_idx of clip_stem's JSONL, or None."""
    jsonl_path = _gt_marks_dir() / f"{clip_stem}.jsonl"
    if not jsonl_path.exists():
        return None
    lines = jsonl_path.read_text().splitlines()
    line_indices = _iter_frame_box_lines(lines, frame_idx)
    if box_idx >= len(line_indices):
        return None
    b = json.loads(lines[line_indices[box_idx]].strip())["bbox"]
    if isinstance(b, dict):
        return b["x"], b["y"], b["w"], b["h"]
    return tuple(b)  # type: ignore[return-value]


def _build_review_queue(records: list[dict]) -> list[dict]:
    """Order and dedup records: low-confidence YOLO first, rare classes weighted up, dedup by IoU."""
    class_weight = {
        "ball": 3,
        "in_play_ball": 3,
        "out_of_play_ball": 3,
        "referee": 2,
        "coach": 2,
        "player_sub": 2,
    }

    def sort_key(r: dict) -> tuple:
        prov_order = 0 if r["provenance"] != PROV_LABELLER else 1
        weight = -class_weight.get(r["label"], 1)
        conf = r["confidence"]
        return (prov_order, conf, weight)

    sorted_records = sorted(records, key=sort_key)

    # IoU dedup: skip crops near-identical to an already-queued item from the same frame
    iou_thresh = 0.85
    seen: dict[tuple, list[tuple]] = {}  # (clip, frame_index) -> list of (x,y,w,h)
    result: list[dict] = []
    for r in sorted_records:
        key = (r["clip"], r["frame_index"])
        b = r["bbox"]
        bx, by, bw, bh = b["x"], b["y"], b["w"], b["h"]
        duplicate = any(
            bbox_iou(ex, (bx, by, bw, bh)) > iou_thresh for ex in seen.get(key, [])
        )
        if not duplicate:
            seen.setdefault(key, []).append((bx, by, bw, bh))
            result.append(r)
    return result


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.get("/review/queue")
async def review_queue() -> dict:
    """Return an ordered list of ReviewItems for the tinder-style review UI."""
    records = await asyncio.to_thread(_read_all_boxes)
    queue = await asyncio.to_thread(_build_review_queue, records)
    # Strip video_path (internal) before sending to client
    items = [
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
        for r in queue
    ]
    return {"total": len(items), "items": items}


@router.get("/review/crop/{clip_stem}/{frame_idx}/{box_idx}.jpg")
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


@router.post("/review/correct")
async def review_correct(body: dict) -> dict:
    """Write a corrected box (PROV_LABELLER) to the clip's JSONL sidecar."""
    clip = body["clip"]
    frame_index = int(body["frame_index"])
    box_index = int(body["box_index"])
    label = body["label"]
    bbox = body["bbox"]  # {x, y, w, h} normalized

    marks_dir = _gt_marks_dir()
    jsonl_path = marks_dir / f"{clip}.jsonl"
    if not jsonl_path.exists():
        return {"ok": False, "error": "clip not found"}

    lines = jsonl_path.read_text().splitlines()
    line_indices = _iter_frame_box_lines(lines, frame_index)
    if box_index >= len(line_indices):
        return {"ok": False, "error": "box_index out of range"}

    bx = max(0.0, min(1.0, float(bbox["x"])))
    by = max(0.0, min(1.0, float(bbox["y"])))
    bw = max(0.0, min(1.0 - bx, float(bbox["w"])))
    bh = max(0.0, min(1.0 - by, float(bbox["h"])))
    lines[line_indices[box_index]] = json.dumps(
        {
            "frame_index": frame_index,
            "bbox": {"x": bx, "y": by, "w": bw, "h": bh},
            "center": {"x": bx + bw / 2, "y": by + bh / 2},
            "tags": [label, PROV_LABELLER],
        }
    )
    marks_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path.write_text("\n".join(lines) + "\n")
    # Invalidate crop cache for this box
    _CROP_CACHE.pop((clip, frame_index, box_index), None)
    return {"ok": True}


@router.post("/review/delete")
async def review_delete(body: dict) -> dict:
    """Remove a box from the clip's JSONL sidecar."""
    clip = body["clip"]
    frame_index = int(body["frame_index"])
    box_index = int(body["box_index"])

    jsonl_path = _gt_marks_dir() / f"{clip}.jsonl"
    if not jsonl_path.exists():
        return {"ok": False, "error": "clip not found"}

    lines = jsonl_path.read_text().splitlines()
    line_indices = _iter_frame_box_lines(lines, frame_index)
    if box_index >= len(line_indices):
        return {"ok": False, "error": "box_index out of range"}

    del lines[line_indices[box_index]]
    jsonl_path.write_text("\n".join(lines) + "\n")
    _CROP_CACHE.pop((clip, frame_index, box_index), None)
    return {"ok": True}


@router.get("/review/frame/{clip_stem}/{frame_idx}.jpg")
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


@router.post("/review/yolo")
async def review_yolo(body: dict) -> dict:
    """Re-run YOLO on a specific frame and return detections."""
    clip_stem = body["clip"]
    frame_idx = int(body["frame_index"])
    video_path = _find_video(clip_stem)
    if video_path is None:
        return {"ok": False, "error": "video not found", "boxes": []}

    def _run() -> list[dict]:
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
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
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
