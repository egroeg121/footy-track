"""Ball Check API: phone-sized yes/no verification of machine ball boxes.

One decision per card — "is the marked object the ball?" — over a zoomed,
reticled crop. The point is precision measurement and hard negatives: the
detector's ball output is the weakest link in the pipeline (see
``docs/design/pipeline_overview.md``), and a swipe on a phone is the cheapest
human signal available for it.

Two deliberate design choices:

* **Nothing is written to the GT sidecars.** Verdicts append to a separate
  log under ``<gt dir>/ball_checks/<clip>.jsonl``. A yes/no answer does not
  check the box *geometry*, so it cannot honestly promote a machine box to
  ``labeller`` GT (README §1, LAB-002) — and an append-only side file cannot
  truncate hand labels the way a sidecar rewrite can.
* **Box identity is review.py's** ``(clip, frame_index, box_index)``, so a
  verdict here refers to the same box the review UI would show.

``server.py`` is the composition root: the clips and GT-marks directories are
resolved through it at call time (tests monkeypatch them there). See
``src/footy_track/labeller/README.md`` §11 (Ball Check, LAB-10xx).
"""

from __future__ import annotations

import asyncio
import collections
import hashlib
import json
import time
from pathlib import Path

import cv2
from fastapi import APIRouter
from fastapi.responses import Response

from footy_track.labeller.constants import BALL_LABELS, PROV_LABELLER
from footy_track.labeller.review import _find_video, _gt_marks_dir, _read_all_boxes

router = APIRouter()

#: Verdicts a card can record. ``unsure`` is kept distinct from ``not_ball``
#: so ambiguous crops never silently become negatives.
VERDICTS = ("ball", "not_ball", "unsure")

#: Context around the box, as a multiple of the *longest* box side. A ball is
#: ~11 px wide, so a proportional pad alone yields a postage stamp; the crop
#: is also floored to _MIN_CONTEXT_PX and upscaled to _DISPLAY_W.
_PAD_FACTOR = 6.0
_MIN_CONTEXT_PX = 160
_DISPLAY_W = 640
_MAX_PAD_FACTOR = 40.0

# LRU crop cache: key = (clip_stem, frame_idx, box_idx, pad), value = JPEG bytes
_BC_CROP_CACHE: collections.OrderedDict[tuple, bytes] = collections.OrderedDict()
_BC_CROP_CACHE_MAX = 200


# ---------------------------------------------------------------------------
# Verdict log
# ---------------------------------------------------------------------------


def _verdicts_dir() -> Path:
    return _gt_marks_dir() / "ball_checks"


def _verdict_path(clip_stem: str) -> Path:
    return _verdicts_dir() / f"{clip_stem}.jsonl"


def _read_verdicts(clip_stem: str | None = None) -> dict[tuple, dict]:
    """Return {(clip, frame_index, box_index): record}, last line winning."""
    out: dict[tuple, dict] = {}
    vdir = _verdicts_dir()
    if not vdir.exists():
        return out
    paths = (
        [_verdict_path(clip_stem)]
        if clip_stem is not None
        else sorted(vdir.glob("*.jsonl"))
    )
    for path in paths:
        if not path.exists():
            continue
        try:
            lines = path.read_text().splitlines()
        except OSError:
            continue
        for raw in lines:
            stripped = raw.strip()
            if not stripped:
                continue
            try:
                rec = json.loads(stripped)
            except json.JSONDecodeError:
                continue
            try:
                key = (
                    rec["clip"],
                    int(rec["frame_index"]),
                    int(rec["box_index"]),
                )
            except (KeyError, TypeError, ValueError):
                continue
            if rec.get("verdict") is None:
                out.pop(key, None)  # undo tombstone
            else:
                out[key] = rec
    return out


def _append_verdict(rec: dict) -> None:
    vdir = _verdicts_dir()
    vdir.mkdir(parents=True, exist_ok=True)
    path = _verdict_path(rec["clip"])
    with path.open("a") as fh:
        fh.write(json.dumps(rec) + "\n")


# ---------------------------------------------------------------------------
# Queue
# ---------------------------------------------------------------------------


def _shuffle_key(rec: dict) -> str:
    """Deterministic spread across clips.

    Precision is a property of the detector, not of one clip, so the queue is
    hash-ordered rather than clip-ordered: a short session still samples the
    whole corpus. Stable across restarts so the card order does not jump.
    """
    raw = f"{rec['clip']}:{rec['frame_index']}:{rec['box_index']}"
    return hashlib.sha1(raw.encode()).hexdigest()


def _build_ball_queue(
    records: list[dict],
    judged: dict[tuple, dict],
    *,
    clip: str | None = None,
    include_gt: bool = False,
) -> list[dict]:
    out = []
    playable: dict[str, bool] = {}
    for r in records:
        if r["label"] not in BALL_LABELS:
            continue
        if not include_gt and r["provenance_tag"] == PROV_LABELLER:
            continue
        if clip is not None and r["clip"] != clip:
            continue
        if (r["clip"], r["frame_index"], r["box_index"]) in judged:
            continue
        # A sidecar stem with no video on disk cannot be cropped (the clip
        # naming schemes diverge), and a card that 404s is worse than no card.
        if r["clip"] not in playable:
            playable[r["clip"]] = _find_video(r["clip"]) is not None
        if not playable[r["clip"]]:
            continue
        out.append(r)
    out.sort(key=_shuffle_key)
    return out


@router.get("/ball_check/queue")
async def ball_check_queue(
    limit: int = 50, clip: str | None = None, include_gt: bool = False
) -> dict:
    """Unjudged machine ball boxes, hash-ordered, newest verdicts excluded."""
    records = await asyncio.to_thread(_read_all_boxes)
    judged = await asyncio.to_thread(_read_verdicts)
    queue = _build_ball_queue(records, judged, clip=clip, include_gt=include_gt)
    limit = max(1, min(int(limit), 500))
    items = [
        {
            "clip": r["clip"],
            "frame_index": r["frame_index"],
            "box_index": r["box_index"],
            "bbox": r["bbox"],
            "label": r["label"],
            "provenance": r["provenance_tag"],
            "image_url": (
                f"/ball_check/crop/{r['clip']}/{r['frame_index']}/{r['box_index']}.jpg"
            ),
        }
        for r in queue[:limit]
    ]
    return {"remaining": len(queue), "judged": len(judged), "items": items}


@router.get("/ball_check/stats")
async def ball_check_stats() -> dict:
    """Verdict tallies plus the precision they imply (unsure excluded)."""
    judged = await asyncio.to_thread(_read_verdicts)
    counts = dict.fromkeys(VERDICTS, 0)
    for rec in judged.values():
        verdict = rec.get("verdict")
        if verdict in counts:
            counts[verdict] += 1
    decided = counts["ball"] + counts["not_ball"]
    return {
        "counts": counts,
        "judged": len(judged),
        "precision": round(counts["ball"] / decided, 4) if decided else None,
    }


# ---------------------------------------------------------------------------
# Crop
# ---------------------------------------------------------------------------


def _crop_window(
    bx: float, by: float, bw: float, bh: float, w_px: int, h_px: int, pad: float
) -> tuple[int, int, int, int]:
    """Square-ish context window around a normalized box, in pixels.

    Floored to ``_MIN_CONTEXT_PX`` so an 11 px ball still lands in a crop the
    eye can judge, and edge-clamped.
    """
    cx = (bx + bw / 2) * w_px
    cy = (by + bh / 2) * h_px
    side = max(bw * w_px, bh * h_px) * (1 + 2 * pad)
    side = max(side, float(_MIN_CONTEXT_PX))
    half = side / 2
    x1 = max(0, int(cx - half))
    y1 = max(0, int(cy - half))
    x2 = min(w_px, int(cx + half))
    y2 = min(h_px, int(cy + half))
    return x1, y1, x2, y2


@router.get("/ball_check/crop/{clip_stem}/{frame_idx}/{box_idx}.jpg")
async def ball_check_crop(
    clip_stem: str, frame_idx: int, box_idx: int, pad: float = _PAD_FACTOR
) -> Response:
    """Zoomed JPEG crop with a reticle drawn on the claimed ball."""
    pad = max(0.0, min(float(pad), _MAX_PAD_FACTOR))
    cache_key = (clip_stem, frame_idx, box_idx, round(pad, 2))
    cached = _BC_CROP_CACHE.get(cache_key)
    if cached is not None:
        _BC_CROP_CACHE.move_to_end(cache_key)
        return Response(content=cached, media_type="image/jpeg")

    video_path = _find_video(clip_stem)
    if video_path is None:
        return Response(status_code=404)

    from footy_track.labeller.review import _read_frame_box  # noqa: PLC0415

    bbox_raw = _read_frame_box(clip_stem, frame_idx, box_idx)
    if bbox_raw is None:
        return Response(status_code=404)
    bx, by, bw, bh = bbox_raw

    def _render() -> bytes | None:
        cap = cv2.VideoCapture(str(video_path))
        try:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ok, frame = cap.read()
        finally:
            cap.release()
        if not ok:
            return None
        h_px, w_px = frame.shape[:2]
        x1, y1, x2, y2 = _crop_window(bx, by, bw, bh, w_px, h_px, pad)
        # Reticle first, on the full frame: without it the card is ambiguous
        # whenever the crop holds more than one round bright thing.
        rx1, ry1 = int(bx * w_px), int(by * h_px)
        rx2, ry2 = int((bx + bw) * w_px), int((by + bh) * h_px)
        margin = max(4, int(max(rx2 - rx1, ry2 - ry1) * 0.6))
        cv2.rectangle(
            frame,
            (rx1 - margin, ry1 - margin),
            (rx2 + margin, ry2 + margin),
            (0, 255, 255),
            2,
        )
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return None
        crop_w = max(1, x2 - x1)
        if crop_w < _DISPLAY_W:
            scale = _DISPLAY_W / crop_w
            crop = cv2.resize(
                crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST
            )
        ok2, buf = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 88])
        return buf.tobytes() if ok2 else None

    data = await asyncio.to_thread(_render)
    if data is None:
        return Response(status_code=404)
    _BC_CROP_CACHE[cache_key] = data
    _BC_CROP_CACHE.move_to_end(cache_key)
    if len(_BC_CROP_CACHE) > _BC_CROP_CACHE_MAX:
        _BC_CROP_CACHE.popitem(last=False)
    return Response(content=data, media_type="image/jpeg")


# ---------------------------------------------------------------------------
# Verdict write / undo
# ---------------------------------------------------------------------------


@router.post("/ball_check/verdict")
async def ball_check_verdict(body: dict) -> dict:
    """Append one verdict. Never touches the clip's GT sidecar."""
    verdict = body.get("verdict")
    if verdict not in VERDICTS:
        return {"ok": False, "error": f"verdict must be one of {list(VERDICTS)}"}
    try:
        clip = str(body["clip"])
        frame_index = int(body["frame_index"])
        box_index = int(body["box_index"])
    except (KeyError, TypeError, ValueError):
        return {"ok": False, "error": "clip, frame_index and box_index are required"}

    rec = {
        "clip": clip,
        "frame_index": frame_index,
        "box_index": box_index,
        "verdict": verdict,
        "bbox": body.get("bbox"),
        "label": body.get("label"),
        "provenance": body.get("provenance"),
        "ts": round(time.time(), 3),
    }
    await asyncio.to_thread(_append_verdict, rec)
    return {"ok": True}


@router.post("/ball_check/undo")
async def ball_check_undo(body: dict) -> dict:
    """Retract a verdict by appending a tombstone (the log stays append-only)."""
    try:
        clip = str(body["clip"])
        frame_index = int(body["frame_index"])
        box_index = int(body["box_index"])
    except (KeyError, TypeError, ValueError):
        return {"ok": False, "error": "clip, frame_index and box_index are required"}
    if not _verdict_path(clip).exists():
        return {"ok": False, "error": "no verdicts for clip"}
    await asyncio.to_thread(
        _append_verdict,
        {
            "clip": clip,
            "frame_index": frame_index,
            "box_index": box_index,
            "verdict": None,
            "ts": round(time.time(), 3),
        },
    )
    return {"ok": True}
