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
import logging
import time
from pathlib import Path

import cv2
from fastapi import APIRouter
from fastapi.responses import Response

from footy_track.labeller import candidates as cand
from footy_track.labeller.constants import (
    BALL_LABELS,
    NO_BALL_TAG,
    NOT_BROADCAST_TAG,
    PROV_LABELLER,
)
from footy_track.labeller.review import _find_video, _gt_marks_dir, _read_all_boxes

LOGGER = logging.getLogger(__name__)

router = APIRouter()

#: Verdicts a card can record. ``unsure`` is kept distinct from ``not_ball``
#: so ambiguous crops never silently become negatives.
VERDICTS = ("ball", "not_ball", "box_off", "corrected", "unsure")

#: Verdicts that confirm a ball is really there. ``box_off`` counts for recall
#: (the detector found it) but not as a clean training box — its geometry is
#: wrong, so it must not be promoted to a reviewed label.
BALL_PRESENT_VERDICTS = ("ball", "box_off", "corrected")

#: Verdicts whose geometry is good enough to train on as-is. ``corrected``
#: qualifies because the human dragged the box themselves — that is hand
#: geometry, written back through ``/review/correct`` as TIER 1 GT.
CLEAN_VERDICTS = ("ball", "corrected")

#: How far a neighbouring-frame box may sit from the anchor centre and still
#: be treated as the same object, in normalized frame units. A ball crosses
#: at most a few percent of the frame per frame at 25 fps; beyond that it is
#: a different detection, not this one moving.
_NEIGHBOUR_MAX_DIST = 0.06

#: Zoom level. The crop window is a FIXED fraction of the frame (below),
#: scaled by ``pad / _PAD_FACTOR`` — deliberately NOT a multiple of the box.
#:
#: Sizing the window off the box was the original design and it hid the very
#: defect this tool exists to catch: the window grew with the box, so a
#: player-sized ball box and a correct 10 px one filled the card identically
#: and looked equally plausible. A fixed window makes an oversized box
#: visibly overflow, and makes every card directly comparable.
_PAD_FACTOR = 6.0

#: Base window as a fraction of frame width: 0.125 -> 160 px at 1280, about
#: 15 ball-widths of context (median hand-labelled ball is 11 px at 1280).
_WINDOW_FRAC = 0.125
_MIN_CONTEXT_PX = 160
_DISPLAY_W = 640
_MAX_PAD_FACTOR = 40.0

# LRU crop cache: key = (clip_stem, frame_idx, box_idx, pad), value = JPEG bytes
_BC_CROP_CACHE: collections.OrderedDict[tuple, bytes] = collections.OrderedDict()
_BC_CROP_CACHE_MAX = 200

#: clip stem -> (width, height); frame size never changes within a clip.
_CLIP_SIZES: dict[str, tuple[int, int]] = {}


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
                    # Records written before candidates existed have no source
                    # and always referred to sidecar boxes.
                    rec.get("source") or "sidecar",
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
        if ("sidecar", r["clip"], r["frame_index"], r["box_index"]) in judged:
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


def _build_candidate_queue(
    records: list[dict],
    judged: dict[tuple, dict],
    *,
    clip: str | None = None,
    tau: float = cand.DEFAULT_TAU,
    sigma: float = cand.DEFAULT_SIGMA,
    focus: float = cand.DEFAULT_FOCUS,
) -> list[dict]:
    pool = [
        r
        for r in records
        if (clip is None or r["clip"] == clip)
        and ("cand", r["clip"], r["frame_index"], r["cand_index"]) not in judged
    ]
    return cand.stratified_order(pool, tau=tau, sigma=sigma, focus=focus)


@router.get("/ball_check/queue")
async def ball_check_queue(
    limit: int = 50,
    clip: str | None = None,
    include_gt: bool = False,
    source: str = "auto",
    tau: float = -1.0,
    sigma: float = cand.DEFAULT_SIGMA,
    focus: float = cand.DEFAULT_FOCUS,
) -> dict:
    """Unjudged ball boxes to review.

    ``source=candidates`` draws from the detector's own output, which carries
    real confidences, and orders them by confidence-stratified sampling.
    ``source=sidecar`` reviews the boxes already in the GT sidecars.
    ``auto`` (default) prefers candidates when any have been pulled down.
    """
    judged = await asyncio.to_thread(_read_verdicts)
    limit = max(1, min(int(limit), 500))

    if source == "auto":
        have_cands = await asyncio.to_thread(
            lambda: cand._candidates_dir().exists()
            and any(cand._candidates_dir().glob("*.jsonl"))
        )
        source = "candidates" if have_cands else "sidecar"

    if source == "candidates":
        pool = await asyncio.to_thread(cand.binned_pool, clip)
        stats = cand.band_stats(list(judged.values()), BALL_PRESENT_VERDICTS)
        # tau < 0 means "measure it": the boundary is re-estimated from the
        # verdicts so far on every fetch, so the queue follows the labeller.
        tau_used = cand.estimate_tau(stats) if tau < 0 else tau

        def _judged(rec: dict) -> bool:
            key = ("cand", rec["clip"], rec["frame_index"], rec["cand_index"])
            return key in judged

        queue = await asyncio.to_thread(
            pool.select,
            lambda c: cand.adaptive_weight(c, tau_used, stats, sigma, focus),
            limit,
            _judged,
        )
        items = [
            {
                "clip": r["clip"],
                "frame_index": r["frame_index"],
                "box_index": r["cand_index"],
                "source": "cand",
                "bbox": r["bbox"],
                "label": r["label"],
                "provenance": r["provenance"],
                "confidence": round(r["confidence"], 4),
                "image_url": (
                    f"/ball_check/crop/{r['clip']}/{r['frame_index']}"
                    f"/{r['cand_index']}.jpg?src=cand"
                ),
            }
            for r in queue[:limit]
        ]
        return {
            "source": "candidates",
            "remaining": pool.total - len(judged),
            "judged": len(judged),
            "items": items,
            "tau": round(tau_used, 4),
            "bands": stats,
            "histogram": [
                {"lo": round(i / 10, 2), "hi": round((i + 1) / 10, 2), "n": n}
                for i, n in enumerate(
                    [
                        sum(len(b) for b in pool.bins[i * 10 : (i + 1) * 10])
                        for i in range(10)
                    ]
                )
            ],
        }

    records = await asyncio.to_thread(_read_all_boxes)
    queue = _build_ball_queue(records, judged, clip=clip, include_gt=include_gt)
    items = [
        {
            "clip": r["clip"],
            "frame_index": r["frame_index"],
            "box_index": r["box_index"],
            "source": "sidecar",
            "bbox": r["bbox"],
            "label": r["label"],
            "provenance": r["provenance_tag"],
            "confidence": None,
            "image_url": (
                f"/ball_check/crop/{r['clip']}/{r['frame_index']}/{r['box_index']}.jpg"
            ),
        }
        for r in queue[:limit]
    ]
    return {
        "source": "sidecar",
        "remaining": len(queue),
        "judged": len(judged),
        "items": items,
    }


@router.get("/ball_check/stats")
async def ball_check_stats() -> dict:
    """Verdict tallies plus the precision they imply (unsure excluded)."""
    judged = await asyncio.to_thread(_read_verdicts)
    counts = dict.fromkeys(VERDICTS, 0)
    for rec in judged.values():
        verdict = rec.get("verdict")
        if verdict in counts:
            counts[verdict] += 1
    present = sum(counts[v] for v in BALL_PRESENT_VERDICTS)
    clean = sum(counts[v] for v in CLEAN_VERDICTS)
    decided = present + counts["not_ball"]
    return {
        "counts": counts,
        "judged": len(judged),
        # A box_off card still found the ball, so it counts as a true positive
        # for detection precision; it is excluded from clean_precision, which
        # is the share good enough to train on as-is.
        "precision": round(present / decided, 4) if decided else None,
        "clean_precision": round(clean / decided, 4) if decided else None,
    }


# ---------------------------------------------------------------------------
# Crop
# ---------------------------------------------------------------------------


def _read_box(
    src: str, clip_stem: str, frame_idx: int, box_idx: int
) -> tuple[float, float, float, float] | None:
    """Resolve a box from whichever source the card came from."""
    if src == "cand":
        return cand.read_candidate_box(clip_stem, frame_idx, box_idx)
    from footy_track.labeller.review import _read_frame_box  # noqa: PLC0415

    return _read_frame_box(clip_stem, frame_idx, box_idx)


def _crop_window(
    bx: float, by: float, bw: float, bh: float, w_px: int, h_px: int, pad: float
) -> tuple[int, int, int, int]:
    """Fixed-scale context window centred on the box, in pixels.

    Floored to ``_MIN_CONTEXT_PX`` so an 11 px ball still lands in a crop the
    eye can judge, widened for outsized boxes, and edge-clamped.
    """
    cx = (bx + bw / 2) * w_px
    cy = (by + bh / 2) * h_px
    side = w_px * _WINDOW_FRAC * (pad / _PAD_FACTOR)
    # A box bigger than the window would be cropped out of its own card, so
    # the window still grows to contain an outlier box (plus a little air).
    side = max(side, float(_MIN_CONTEXT_PX), max(bw * w_px, bh * h_px) * 1.6)
    half = side / 2
    x1 = max(0, int(cx - half))
    y1 = max(0, int(cy - half))
    x2 = min(w_px, int(cx + half))
    y2 = min(h_px, int(cy + half))
    return x1, y1, x2, y2


@router.get("/ball_check/crop_meta/{clip_stem}/{frame_idx}/{box_idx}")
async def ball_check_crop_meta(
    clip_stem: str,
    frame_idx: int,
    box_idx: int,
    pad: float = _PAD_FACTOR,
    src: str = "sidecar",
) -> dict:
    """Geometry the page needs to overlay (and drag) the box on the crop.

    Without this the client cannot map normalized frame coordinates into the
    cropped, upscaled image it is actually showing.
    """
    pad = max(0.0, min(float(pad), _MAX_PAD_FACTOR))
    video_path = _find_video(clip_stem)
    if video_path is None:
        return {"ok": False, "error": "video not found"}

    bbox_raw = await asyncio.to_thread(_read_box, src, clip_stem, frame_idx, box_idx)
    if bbox_raw is None:
        return {"ok": False, "error": "box not found"}
    bx, by, bw, bh = bbox_raw

    size = await asyncio.to_thread(_clip_size, clip_stem, video_path)
    if size is None:
        return {"ok": False, "error": "clip size unreadable"}
    w_px, h_px = size
    x1, y1, x2, y2 = _crop_window(bx, by, bw, bh, w_px, h_px, pad)
    return {
        "ok": True,
        "frame": {"w": w_px, "h": h_px},
        "window": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
        "bbox": {"x": bx, "y": by, "w": bw, "h": bh},
    }


def _clip_size(clip_stem: str, video_path: Path) -> tuple[int, int] | None:
    """(width, height) in pixels, cached per clip — every card needs it."""
    if clip_stem in _CLIP_SIZES:
        return _CLIP_SIZES[clip_stem]
    cap = cv2.VideoCapture(str(video_path))
    try:
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    finally:
        cap.release()
    if w <= 0 or h <= 0:
        return None
    _CLIP_SIZES[clip_stem] = (w, h)
    return w, h


def _encode_window(frame, x1: int, y1: int, x2: int, y2: int) -> bytes | None:
    """Crop to the window, upscale small crops, JPEG-encode."""
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    crop_w = max(1, x2 - x1)
    if crop_w < _DISPLAY_W:
        scale = _DISPLAY_W / crop_w
        crop = cv2.resize(
            crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST
        )
    ok, buf = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 88])
    return buf.tobytes() if ok else None


@router.get("/ball_check/crop/{clip_stem}/{frame_idx}/{box_idx}.jpg")
async def ball_check_crop(
    clip_stem: str,
    frame_idx: int,
    box_idx: int,
    pad: float = _PAD_FACTOR,
    reticle: bool = True,
    src: str = "sidecar",
    at: int = -1,
) -> Response:
    """Zoomed JPEG crop, with an optional reticle drawn on the claimed ball.

    The page passes ``reticle=false`` and draws its own draggable box overlay
    instead, so what you see is the live box you are about to correct.
    """
    pad = max(0.0, min(float(pad), _MAX_PAD_FACTOR))
    # ``at`` renders a neighbouring frame through THIS box's window, so
    # stepping moves the football, not the camera.
    at_frame = frame_idx if at < 0 else at
    cache_key = (
        src,
        clip_stem,
        frame_idx,
        box_idx,
        round(pad, 2),
        bool(reticle),
        at_frame,
    )
    cached = _BC_CROP_CACHE.get(cache_key)
    if cached is not None:
        _BC_CROP_CACHE.move_to_end(cache_key)
        return Response(content=cached, media_type="image/jpeg")

    video_path = _find_video(clip_stem)
    if video_path is None:
        return Response(status_code=404)

    bbox_raw = await asyncio.to_thread(_read_box, src, clip_stem, frame_idx, box_idx)
    if bbox_raw is None:
        return Response(status_code=404)
    bx, by, bw, bh = bbox_raw

    def _render() -> bytes | None:
        cap = cv2.VideoCapture(str(video_path))
        try:
            cap.set(cv2.CAP_PROP_POS_FRAMES, at_frame)
            ok, frame = cap.read()
        finally:
            cap.release()
        if not ok:
            return None
        h_px, w_px = frame.shape[:2]
        x1, y1, x2, y2 = _crop_window(bx, by, bw, bh, w_px, h_px, pad)
        if not reticle:
            return _encode_window(frame, x1, y1, x2, y2)
        # Reticle first, on the full frame: without it the card is ambiguous
        # whenever the crop holds more than one round bright thing.
        #
        # It draws the box *exactly* — an inflated marker would make every box
        # look looser than it is, and judging box tightness is half the point.
        # Visibility comes from corner brackets sitting outside the box rather
        # than from padding the box itself.
        rx1, ry1 = int(bx * w_px), int(by * h_px)
        rx2, ry2 = int((bx + bw) * w_px), int((by + bh) * h_px)
        cv2.rectangle(frame, (rx1, ry1), (rx2, ry2), (0, 255, 255), 1)
        gap = max(3, int(max(rx2 - rx1, ry2 - ry1) * 0.35))
        arm = max(4, int(max(rx2 - rx1, ry2 - ry1) * 0.5))
        for cx_, cy_, sx, sy in (
            (rx1, ry1, -1, -1),
            (rx2, ry1, 1, -1),
            (rx1, ry2, -1, 1),
            (rx2, ry2, 1, 1),
        ):
            ox, oy = cx_ + sx * gap, cy_ + sy * gap
            cv2.line(frame, (ox, oy), (ox + sx * arm, oy), (0, 255, 255), 2)
            cv2.line(frame, (ox, oy), (ox, oy + sy * arm), (0, 255, 255), 2)
        return _encode_window(frame, x1, y1, x2, y2)

    data = await asyncio.to_thread(_render)
    if data is None:
        return Response(status_code=404)
    _BC_CROP_CACHE[cache_key] = data
    _BC_CROP_CACHE.move_to_end(cache_key)
    if len(_BC_CROP_CACHE) > _BC_CROP_CACHE_MAX:
        _BC_CROP_CACHE.popitem(last=False)
    return Response(content=data, media_type="image/jpeg")


def _sidecar_ball_boxes(clip_stem: str, frame_index: int) -> list[dict]:
    """Ball boxes on one frame of a clip's sidecar, with review's box_index."""
    path = _gt_marks_dir() / f"{clip_stem}.jsonl"
    if not path.exists():
        return []
    try:
        lines = path.read_text().splitlines()
    except OSError:
        return []
    out: list[dict] = []
    box_idx = 0
    for raw in lines:
        line = raw.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        tags = rec.get("tags") or []
        if NO_BALL_TAG in tags or NOT_BROADCAST_TAG in tags or rec.get("bbox") is None:
            continue
        if int(rec.get("frame_index", -1)) != frame_index:
            continue
        # box_index counts every box on the frame, not just ball ones, to stay
        # in step with review's numbering.
        label = next((t for t in tags if t in BALL_LABELS), None)
        if label is not None:
            b = rec["bbox"]
            if isinstance(b, dict):
                bbox = {"x": b["x"], "y": b["y"], "w": b["w"], "h": b["h"]}
            else:
                bbox = {"x": b[0], "y": b[1], "w": b[2], "h": b[3]}
            out.append({"bbox": bbox, "label": label, "box_index": box_idx})
        box_idx += 1
    return out


@router.get("/ball_check/step/{clip_stem}/{frame_idx}/{box_idx}")
async def ball_check_step(
    clip_stem: str,
    frame_idx: int,
    box_idx: int,
    delta: int = 0,
    pad: float = _PAD_FACTOR,
    src: str = "sidecar",
) -> dict:
    """The neighbouring frame, viewed through this card's window.

    Returns the box on that frame if the source has one — that is the point:
    a detection that vanishes on the next frame reads very differently from
    one that tracks smoothly, and at 11 px that is often the only way to tell.
    The verdict still belongs to the anchor frame.
    """
    target = max(0, frame_idx + int(delta))
    anchor = await asyncio.to_thread(_read_box, src, clip_stem, frame_idx, box_idx)
    if anchor is None:
        return {"ok": False, "error": "anchor box not found"}

    def _neighbour() -> dict | None:
        ax, ay, aw, ah = anchor
        acx, acy = ax + aw / 2, ay + ah / 2
        if src == "cand":
            rows = [
                r for r in cand.read_candidates(clip_stem) if r["frame_index"] == target
            ]
        else:
            rows = [
                {**r, "confidence": None}
                for r in _sidecar_ball_boxes(clip_stem, target)
            ]
        if not rows:
            return None

        # Nearest to the anchor centre: on a frame with two ball candidates the
        # one being stepped through is the one that stayed put.
        def dist(r: dict) -> float:
            b = r["bbox"]
            return (b["x"] + b["w"] / 2 - acx) ** 2 + (b["y"] + b["h"] / 2 - acy) ** 2

        best = min(rows, key=dist)
        # Only show it if it is plausibly the same object. A detection on the
        # far side of the pitch is not this ball one frame later, and drawing
        # it would suggest a continuity that is not there.
        if dist(best) > _NEIGHBOUR_MAX_DIST**2:
            return None
        return best

    near = await asyncio.to_thread(_neighbour)
    return {
        "ok": True,
        "frame_index": target,
        "delta": target - frame_idx,
        "bbox": near["bbox"] if near else None,
        "label": near.get("label") if near else None,
        "confidence": (near.get("confidence") if near else None),
        "image_url": (
            f"/ball_check/crop/{clip_stem}/{frame_idx}/{box_idx}.jpg"
            f"?src={src}&at={target}&reticle=false"
        ),
    }


@router.post("/ball_check/save_box")
async def ball_check_save_box(body: dict) -> dict:
    """Write a hand-corrected box to the clip's GT sidecar as labeller GT.

    Candidates come from the detector's own output and most of their clips
    have no sidecar at all, so the review endpoint's rewrite-by-index could
    only ever fail with "clip not found". A dragged box is real hand
    geometry, so for a candidate we append a new GT line (creating the file
    if needed); for a sidecar box we rewrite that box in place, which keeps
    review's box_index numbering stable.
    """
    try:
        clip = str(body["clip"])
        frame_index = int(body["frame_index"])
        box_index = int(body["box_index"])
        bbox = body["bbox"]
        bx = max(0.0, min(1.0, float(bbox["x"])))
        by = max(0.0, min(1.0, float(bbox["y"])))
        bw = max(0.0, min(1.0 - bx, float(bbox["w"])))
        bh = max(0.0, min(1.0 - by, float(bbox["h"])))
    except (KeyError, TypeError, ValueError):
        return {"ok": False, "error": "clip, frame_index, box_index and bbox required"}
    if bw <= 0 or bh <= 0:
        return {"ok": False, "error": "box has no area"}

    label = body.get("label") or "in_play_ball"
    if label not in BALL_LABELS:
        label = "in_play_ball"
    src = body.get("source") or "sidecar"

    if src == "sidecar":
        from footy_track.labeller.review import review_correct  # noqa: PLC0415

        result = await review_correct(
            {
                "clip": clip,
                "frame_index": frame_index,
                "box_index": box_index,
                "label": label,
                "bbox": {"x": bx, "y": by, "w": bw, "h": bh},
            }
        )
        if result.get("ok"):
            return {**result, "written": "rewritten"}
        # Rewrite-in-place is only possible when that exact line exists. It
        # does not when the card was really a candidate (a stale page can
        # claim "sidecar", and candidate clips usually have no sidecar), or
        # when the file was renumbered underneath us. Losing a hand-drawn box
        # to a bookkeeping mismatch is the worst outcome available here, so
        # fall through and append it as GT instead of failing.
        LOGGER.info(
            "save_box: rewrite failed for %s f%s box %s (%s) — appending instead",
            clip,
            frame_index,
            box_index,
            result.get("error"),
        )

    def _append_gt() -> None:
        marks_dir = _gt_marks_dir()
        marks_dir.mkdir(parents=True, exist_ok=True)
        path = marks_dir / f"{clip}.jsonl"
        line = json.dumps(
            {
                "frame_index": frame_index,
                "bbox": {"x": bx, "y": by, "w": bw, "h": bh},
                "center": {"x": bx + bw / 2, "y": by + bh / 2},
                "tags": [label, PROV_LABELLER],
            }
        )
        # Append-only, and newline-safe: a sidecar written elsewhere may not
        # end in one, and joining onto it would corrupt the last record.
        existing = path.read_text() if path.exists() else ""
        prefix = "" if (not existing or existing.endswith("\n")) else "\n"
        with path.open("a") as fh:
            fh.write(prefix + line + "\n")

    await asyncio.to_thread(_append_gt)
    return {"ok": True, "written": "appended"}


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
        "source": body.get("source") or "sidecar",
        # The confidence the box carried when judged: without it the running
        # per-band precision estimate has nothing to bin on.
        "confidence": body.get("confidence"),
        "verdict": verdict,
        "bbox": body.get("bbox"),
        "label": body.get("label"),
        "provenance": body.get("provenance"),
        # Present for `corrected`: the box the human actually dragged, so the
        # log records the new geometry as well as the sidecar rewrite.
        "corrected_bbox": body.get("corrected_bbox"),
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
            "source": body.get("source") or "sidecar",
            "verdict": None,
            "ts": round(time.time(), 3),
        },
    )
    return {"ok": True}
