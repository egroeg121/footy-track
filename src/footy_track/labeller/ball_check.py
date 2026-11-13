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
import contextlib
import hashlib
import json
import logging
import time
from pathlib import Path

import cv2
from fastapi import APIRouter, Request
from fastapi.responses import Response

from footy_track.clip_id import canonical_clip_id
from footy_track.labeller import candidates as cand
from footy_track.labeller.constants import (
    BALL_LABELS,
    NO_BALL_TAG,
    NOT_BROADCAST_TAG,
    PROV_LABELLER,
    card_box_index,
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

# Context clip (H.264 MP4, GIF fallback) --------------------------------------
#: Seconds of footage either side of the card's frame.
_CLIP_HALF_S = 1.5
#: Output frame rate cap; sources above it are subsampled by whole frames.
_CLIP_MAX_FPS = 30.0
#: Rendered width in px: the crop is upscaled server-side (Lanczos) to this, so
#: the phone shows a smooth, sharp picture rather than pixelated CSS scaling.
_CLIP_W = 560
#: H.264 quality (lower = better); the clip is ~3 s so size stays small.
_CLIP_CRF = 20
#: The centre frame (the one the box belongs to) is repeated this long, ms, so
#: the box is on screen long enough to see and is exactly right (it is the
#: frame the box was drawn on; neighbouring frames would carry a wrong box).
_CLIP_HOLD_MS = 500
#: GIF fallback (ffmpeg missing): native crop width, ~10 fps, 40 colours.
_GIF_W = 240
_GIF_FPS = 10.0
_BC_CLIP_CACHE: collections.OrderedDict[tuple, tuple[bytes, str, int, int]] = (
    collections.OrderedDict()
)
_BC_CLIP_CACHE_MAX = 40

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
# Review list (explicit cards, source "list")
# ---------------------------------------------------------------------------

_LIST_CACHE: dict[str, tuple[float, list[dict]]] = {}


def _list_path() -> Path | None:
    from footy_track.labeller import server  # noqa: PLC0415 — avoid import cycle

    raw = server._BALL_CHECK_LIST
    return Path(raw).expanduser() if raw else None


def _list_items() -> list[dict]:
    """Cards of the review list, in file order.

    The file is ``{"items": [{"id"?, "clip", "frame_index", "bbox": {x,y,w,h},
    "label", "note", "other_bbox"?, ...}]}`` with normalised boxes. ``box_index`` is
    the list position, or :func:`card_box_index` of ``id`` when the card has one
    (stable under reordering; a repeated id keeps its first card only). A missing or
    unreadable file is an empty pool. Extra keys pass through to the page.
    """
    path = _list_path()
    if path is None or not path.exists():
        return []
    mtime = path.stat().st_mtime
    hit = _LIST_CACHE.get(str(path))
    if hit is not None and hit[0] == mtime:
        return hit[1]
    try:
        raw = json.loads(path.read_text())
        items = raw["items"] if isinstance(raw, dict) else raw
        out = []
        seen: set[str] = set()
        for i, it in enumerate(items):
            cid = it.get("id")
            if cid is not None:
                cid = str(cid)
                if cid in seen:
                    LOGGER.warning("ball_check: duplicate card id %r skipped", cid)
                    continue
                seen.add(cid)
            out.append(
                {
                    **it,
                    "box_index": i if cid is None else card_box_index(cid),
                    "frame_index": int(it["frame_index"]),
                }
            )
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError):
        LOGGER.warning("ball_check: review list %s unreadable", path)
        return []
    _LIST_CACHE[str(path)] = (mtime, out)
    return out


def _list_item(clip: str, frame_index: int, box_index: int) -> dict | None:
    for it in _list_items():
        if (
            it["box_index"] == box_index
            and it["clip"] == clip
            and it["frame_index"] == frame_index
        ):
            return it
    return None


def _list_box(it: dict, key: str = "bbox") -> tuple[float, float, float, float] | None:
    b = it.get(key)
    if not isinstance(b, dict):
        return None
    return float(b["x"]), float(b["y"]), float(b["w"]), float(b["h"])


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


#: List-card fields that reveal where a box came from; withheld when blind.
_BLIND_HIDDEN_KEYS = frozenset({"note", "type", "other_bbox"})


@router.get("/ball_check/queue")
async def ball_check_queue(
    limit: int = 50,
    clip: str | None = None,
    include_gt: bool = False,
    source: str = "auto",
    tau: float = -1.0,
    sigma: float = cand.DEFAULT_SIGMA,
    focus: float = cand.DEFAULT_FOCUS,
    blind: bool = True,
) -> dict:
    """Unjudged ball boxes to review.

    ``source=candidates`` draws from the detector's own output, which carries
    real confidences, and orders them by confidence-stratified sampling.
    ``source=sidecar`` reviews the boxes already in the GT sidecars.
    ``source=list`` serves the explicit cards of the review list
    (``FOOTY_BALL_CHECK_LIST``) in file order, skipping judged ones.
    ``auto`` (default) prefers candidates when any have been pulled down.
    List cards are *blind* by default (``blind=false`` opts out): the
    source-revealing ``note``, ``type`` and ``other_bbox`` are dropped, the
    image URLs ask for a one-box render, and the page shows a neutral prompt.
    """
    judged = await asyncio.to_thread(_read_verdicts)
    limit = max(1, min(int(limit), 500))

    if source == "auto":
        have_cands = await asyncio.to_thread(
            lambda: cand._candidates_dir().exists()
            and any(cand._candidates_dir().glob("*.jsonl"))
        )
        source = "candidates" if have_cands else "sidecar"

    if source == "list":
        all_items = [
            it
            for it in await asyncio.to_thread(_list_items)
            if clip is None or it["clip"] == clip
        ]
        todo = [
            it
            for it in all_items
            if ("list", it["clip"], it["frame_index"], it["box_index"]) not in judged
        ]
        return {
            "source": "list",
            "remaining": len(todo),
            "judged": len(all_items) - len(todo),
            "items": [
                {
                    **{
                        k: v
                        for k, v in it.items()
                        if not (blind and k in _BLIND_HIDDEN_KEYS)
                    },
                    "blind": blind,
                    "source": "list",
                    "label": it.get("label") or "in_play_ball",
                    "provenance": it.get("provenance") or it.get("role") or "list",
                    "confidence": it.get("confidence"),
                    "image_url": (
                        f"/ball_check/crop/{it['clip']}/{it['frame_index']}"
                        f"/{it['box_index']}.jpg?src=list"
                        + ("" if blind else "&blind=false")
                    ),
                }
                for it in todo[:limit]
            ],
        }

    if source == "candidates":
        pool = await asyncio.to_thread(cand.binned_pool, clip)
        n_judged = sum(1 for k in judged if k[0] != "list")
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
            # Events, not raw rows: consecutive detections of one ball are one
            # card, so counting rows overstates the work left by ~2.7x.
            "remaining": max(0, pool.events - n_judged),
            "rows": pool.total,
            "judged": n_judged,
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
    for key, rec in judged.items():
        if key[0] == "list":
            continue  # hand-picked disagreement cards: not a detector precision sample
        verdict = rec.get("verdict")
        if verdict in counts:
            counts[verdict] += 1
    present = sum(counts[v] for v in BALL_PRESENT_VERDICTS)
    clean = sum(counts[v] for v in CLEAN_VERDICTS)
    decided = present + counts["not_ball"]
    return {
        "counts": counts,
        "judged": sum(1 for k in judged if k[0] != "list"),
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
    if src == "list":
        it = _list_item(clip_stem, frame_idx, box_idx)
        return _list_box(it) if it else None
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
    blind: bool = True,
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
        bool(blind),
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
        if src == "list" and at_frame == frame_idx and not blind:
            # The other box of the pair (GT mark vs Claude's box), magenta,
            # so one card shows both.
            it = _list_item(clip_stem, frame_idx, box_idx)
            ob = _list_box(it, "other_bbox") if it else None
            if ob is not None:
                ox1, oy1 = int(ob[0] * w_px), int(ob[1] * h_px)
                ox2, oy2 = int((ob[0] + ob[2]) * w_px), int((ob[1] + ob[3]) * h_px)
                # Inflated a little and 2 px thick: at 1:1 a 1 px box around a
                # 10 px object vanishes once the crop is upscaled and compressed.
                cv2.rectangle(
                    frame, (ox1 - 4, oy1 - 4), (ox2 + 4, oy2 + 4), (255, 0, 255), 2
                )
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


def _draw_box(frame, bx, by, bw, bh, w_px, h_px, colour, thickness=1, bracket=True):
    """The still's reticle: the true box, plus corner brackets outside it."""
    rx1, ry1 = int(bx * w_px), int(by * h_px)
    rx2, ry2 = int((bx + bw) * w_px), int((by + bh) * h_px)
    cv2.rectangle(frame, (rx1, ry1), (rx2, ry2), colour, thickness)
    if not bracket:
        return
    gap = max(3, int(max(rx2 - rx1, ry2 - ry1) * 0.35))
    arm = max(4, int(max(rx2 - rx1, ry2 - ry1) * 0.5))
    for cx_, cy_, sx, sy in (
        (rx1, ry1, -1, -1),
        (rx2, ry1, 1, -1),
        (rx1, ry2, -1, 1),
        (rx2, ry2, 1, 1),
    ):
        ox, oy = cx_ + sx * gap, cy_ + sy * gap
        cv2.line(frame, (ox, oy), (ox + sx * arm, oy), colour, 2)
        cv2.line(frame, (ox, oy), (ox, oy + sy * arm), colour, 2)


def _clip_frame_indices(frame_idx: int, fps: float, total: int) -> tuple[int, int, int]:
    """(first, last, step) source frames for the clip, clamped to the video.

    ``step`` is 1 up to ``_CLIP_MAX_FPS`` and subsamples above it; the centre
    frame is always on the sampling grid because the grid is anchored at
    ``frame_idx``.
    """
    fps = fps if fps and fps > 1 else 25.0
    half = int(round(_CLIP_HALF_S * fps))
    step = max(1, int(round(fps / _CLIP_MAX_FPS)))
    first = max(0, frame_idx - half)
    last = frame_idx + half
    if total > 0:
        last = min(last, total - 1)
    return first, max(first, last), step


def _gif_bytes(frames: list, durations: list[int]) -> bytes:
    """Encode BGR ndarrays as a play-once GIF (no loop extension)."""
    import io  # noqa: PLC0415

    from PIL import Image  # noqa: PLC0415

    imgs = [
        Image.fromarray(f[:, :, ::-1]).quantize(colors=40, dither=Image.Dither.NONE)
        for f in frames
    ]
    buf = io.BytesIO()
    imgs[0].save(
        buf,
        format="GIF",
        save_all=True,
        append_images=imgs[1:],
        duration=durations,
        disposal=1,
        optimize=False,
    )
    return buf.getvalue()


def _have_ffmpeg() -> bool:
    import shutil  # noqa: PLC0415

    return shutil.which("ffmpeg") is not None


class _Mp4Writer:
    """Pipes raw BGR frames into ffmpeg (libx264) and returns the MP4 bytes.

    Output goes to a temp file, not a pipe: ``+faststart`` moves the moov atom
    to the front, which needs a seekable output. iOS Safari needs that plus
    yuv420p / even dimensions / a main-or-baseline profile.
    """

    def __init__(self, width: int, height: int, fps: float) -> None:
        import subprocess  # noqa: PLC0415
        import tempfile  # noqa: PLC0415

        self._tmp = tempfile.TemporaryDirectory(prefix="bc_clip_")
        self._out = Path(self._tmp.name) / "clip.mp4"
        self._err = (Path(self._tmp.name) / "err.txt").open("wb")
        self.proc = subprocess.Popen(  # noqa: S603
            [
                "ffmpeg", "-v", "error", "-y",
                "-f", "rawvideo", "-pix_fmt", "bgr24",
                "-s", f"{width}x{height}", "-framerate", f"{fps:.3f}",
                "-i", "-",
                "-vf", "scale=out_color_matrix=bt709:out_range=tv",
                "-c:v", "libx264", "-profile:v", "main", "-level", "4.0",
                "-preset", "medium", "-crf", str(_CLIP_CRF),
                "-pix_fmt", "yuv420p",
                "-colorspace", "bt709", "-color_primaries", "bt709",
                "-color_trc", "bt709",
                "-movflags", "+faststart", "-an",
                str(self._out),
            ],  # fmt: skip
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=self._err,
        )  # fmt: skip

    def write(self, frame) -> None:
        self.proc.stdin.write(frame.tobytes())

    def finish(self) -> bytes | None:
        try:
            with contextlib.suppress(OSError):
                self.proc.stdin.close()
            rc = self.proc.wait(timeout=60)
            self._err.close()
            if rc != 0 or not self._out.exists():
                LOGGER.warning("ffmpeg clip encode failed (rc=%s)", rc)
                return None
            return self._out.read_bytes()
        finally:
            self._tmp.cleanup()

    def abort(self) -> None:
        self.proc.kill()
        self.proc.wait()
        self._err.close()
        self._tmp.cleanup()


def _box_in_crop(b, w_px, h_px, win, out_wh):
    """Normalised frame box -> normalised box in the upscaled crop."""
    x1, y1, x2, y2 = win
    sx, sy = out_wh[0] / (x2 - x1), out_wh[1] / (y2 - y1)
    return (
        (b[0] * w_px - x1) * sx / out_wh[0],
        (b[1] * h_px - y1) * sy / out_wh[1],
        b[2] * w_px * sx / out_wh[0],
        b[3] * h_px * sy / out_wh[1],
    )


def _draw_centre_boxes(crop, bbox, ob, frame_wh, win, out_wh) -> None:
    """Box (and magenta other box) on the upscaled centre frame."""
    if ob is not None:
        box = _box_in_crop(ob, *frame_wh, win, out_wh)
        _draw_box(crop, *box, *out_wh, (255, 0, 255), 2, bracket=False)
    box = _box_in_crop(bbox, *frame_wh, win, out_wh)
    _draw_box(crop, *box, *out_wh, (0, 255, 255), 2)


def _other_box(
    src: str, clip_stem: str, frame_idx: int, box_idx: int, blind: bool = True
):
    if src != "list" or blind:
        return None
    it = _list_item(clip_stem, frame_idx, box_idx)
    return _list_box(it, "other_bbox") if it else None


def _render_mp4(cap, clip_stem, src, frame_idx, box_idx, bbox, pad, blind=True):
    """Frames -> ffmpeg. Returns (bytes, 'video/mp4', n_frames, ms) or None."""
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    first, last, step = _clip_frame_indices(frame_idx, fps, total)
    out_fps = (fps if fps and fps > 1 else 25.0) / step
    cap.set(cv2.CAP_PROP_POS_FRAMES, first)
    ob = _other_box(src, clip_stem, frame_idx, box_idx, blind)
    writer: _Mp4Writer | None = None
    win = None
    out_wh = (0, 0)
    n_out = 0
    try:
        for idx in range(first, last + 1):
            ok, frame = cap.read()
            if not ok:
                break
            if (idx - frame_idx) % step:
                continue
            h_px, w_px = frame.shape[:2]
            if win is None:
                win = _crop_window(*bbox, w_px, h_px, pad)
            x1, y1, x2, y2 = win
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            cw, ch = x2 - x1, y2 - y1
            if writer is None:
                # even dimensions, same aspect as the window
                ow = max(2, (_CLIP_W // 2) * 2)
                oh = max(2, int(round(ow * ch / max(1, cw) / 2)) * 2)
                out_wh = (ow, oh)
                writer = _Mp4Writer(ow, oh, out_fps)
            interp = (
                getattr(cv2, "INTER_LANCZOS4", 4)
                if out_wh[0] > cw
                else getattr(cv2, "INTER_AREA", 3)
            )
            crop = cv2.resize(crop, out_wh, interpolation=interp)
            reps = 1
            if idx == frame_idx:
                # draw on the upscaled crop so the lines stay crisp
                _draw_centre_boxes(crop, bbox, ob, (w_px, h_px), win, out_wh)
                reps = max(1, int(round(_CLIP_HOLD_MS / 1000 * out_fps)))
            for _ in range(reps):
                writer.write(crop)
            n_out += reps
    except BaseException:
        if writer is not None:
            writer.abort()
        raise
    data = writer.finish() if writer is not None and n_out else None
    if data is None:
        if writer is not None and not n_out:
            writer.abort()
        return None
    return data, "video/mp4", n_out, int(round(n_out * 1000 / out_fps))


def _render_gif(cap, clip_stem, src, frame_idx, box_idx, bbox, pad, blind=True):
    """ffmpeg-less fallback: small play-once GIF (native crop, ~10 fps)."""
    bx, by, bw, bh = bbox
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    first, last, _ = _clip_frame_indices(frame_idx, fps, total)
    step = max(1, int(round((fps if fps and fps > 1 else 25.0) / _GIF_FPS)))
    cap.set(cv2.CAP_PROP_POS_FRAMES, first)
    ob = _other_box(src, clip_stem, frame_idx, box_idx, blind)
    base_ms = int(round(1000 / _GIF_FPS))
    frames: list = []
    durations: list[int] = []
    win = None
    for idx in range(first, last + 1):
        ok, frame = cap.read()
        if not ok:
            break
        if (idx - frame_idx) % step:
            continue
        h_px, w_px = frame.shape[:2]
        if win is None:
            win = _crop_window(bx, by, bw, bh, w_px, h_px, pad)
        x1, y1, x2, y2 = win
        centre = idx == frame_idx
        if centre:
            if ob is not None:
                _draw_box(frame, *ob, w_px, h_px, (255, 0, 255), 2, bracket=False)
            _draw_box(frame, bx, by, bw, bh, w_px, h_px, (0, 255, 255))
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            continue
        scale = _GIF_W / max(1, x2 - x1)
        if scale < 1:
            crop = cv2.resize(
                crop,
                None,
                fx=scale,
                fy=scale,
                interpolation=getattr(cv2, "INTER_AREA", cv2.INTER_NEAREST),
            )
        frames.append(crop)
        durations.append(_CLIP_HOLD_MS if centre else base_ms)
    if not frames:
        return None
    return _gif_bytes(frames, durations), "image/gif", len(frames), sum(durations)


def _render_clip(
    video_path: Path,
    src: str,
    clip_stem: str,
    frame_idx: int,
    box_idx: int,
    bbox: tuple[float, float, float, float],
    pad: float,
    blind: bool = True,
) -> tuple[bytes, str, int, int] | None:
    """Decode once (seek, then sequential reads) and encode.

    Returns ``(bytes, media_type, n_frames, duration_ms)``: MP4 via ffmpeg when
    available (full frame rate, crop upscaled with Lanczos, box on the held
    centre frame only), otherwise the old small GIF.
    """
    render = _render_mp4 if _have_ffmpeg() else _render_gif
    cap = cv2.VideoCapture(str(video_path))
    try:
        return render(cap, clip_stem, src, frame_idx, box_idx, bbox, pad, blind)
    finally:
        cap.release()


@router.get("/ball_check/clip/{clip_stem}/{frame_idx}/{box_idx}.mp4")
async def ball_check_clip(
    clip_stem: str,
    frame_idx: int,
    box_idx: int,
    request: Request,
    pad: float = _PAD_FACTOR,
    src: str = "sidecar",
    blind: bool = True,
) -> Response:
    """~3 s context clip (H.264 MP4) centred on the card's frame.

    Shows whether the marked object is moving like the in-play ball or is a
    spare/crowd ball, which a single still cannot. The crop window is the
    still's window, fixed for every frame, so the view never jumps. The box
    (and, for ``src=list`` with ``blind=false``, the magenta other box) is drawn only on the centre
    frame, which is held ~0.5 s: the ball moves, so a box carried onto other
    frames would lie. Supports HTTP Range (iOS Safari needs 206 for <video>).
    Falls back to a small GIF (same URL, ``image/gif``) if ffmpeg is missing.
    """
    pad = max(0.0, min(float(pad), _MAX_PAD_FACTOR))
    cache_key = (src, clip_stem, frame_idx, box_idx, round(pad, 2), bool(blind))
    hit = _BC_CLIP_CACHE.get(cache_key)
    if hit is not None:
        _BC_CLIP_CACHE.move_to_end(cache_key)
        return _clip_response(request, *hit)

    video_path = _find_video(clip_stem)
    if video_path is None:
        return Response(status_code=404)
    bbox_raw = await asyncio.to_thread(_read_box, src, clip_stem, frame_idx, box_idx)
    if bbox_raw is None:
        return Response(status_code=404)
    out = await asyncio.to_thread(
        _render_clip,
        video_path,
        src,
        clip_stem,
        frame_idx,
        box_idx,
        bbox_raw,
        pad,
        blind,
    )
    if out is None:
        return Response(status_code=404)
    _BC_CLIP_CACHE[cache_key] = out
    _BC_CLIP_CACHE.move_to_end(cache_key)
    if len(_BC_CLIP_CACHE) > _BC_CLIP_CACHE_MAX:
        _BC_CLIP_CACHE.popitem(last=False)
    return _clip_response(request, *out)


def _parse_range(header: str | None, size: int) -> tuple[int, int] | None:
    """Inclusive (start, end) for a single ``bytes=`` range, else None.

    Raises ValueError for an unsatisfiable range (caller answers 416).
    """
    if not header or not header.startswith("bytes=") or "," in header:
        return None
    a, _, b = header[6:].strip().partition("-")
    try:
        if a == "":
            n = int(b)
            if n <= 0:
                raise ValueError
            start, end = max(0, size - n), size - 1
        else:
            start = int(a)
            end = min(int(b), size - 1) if b else size - 1
    except ValueError:
        raise ValueError("bad range") from None
    if start > end or start >= size:
        raise ValueError("unsatisfiable")
    return start, end


def _clip_response(
    request: Request, data: bytes, media_type: str, n_frames: int, duration_ms: int
) -> Response:
    headers = {
        "X-Clip-Frames": str(n_frames),
        "X-Clip-Duration-Ms": str(duration_ms),
        "Access-Control-Expose-Headers": "X-Clip-Frames, X-Clip-Duration-Ms",
        "Cache-Control": "private, max-age=600",
        "Accept-Ranges": "bytes",
    }
    size = len(data)
    try:
        rng = _parse_range(request.headers.get("range"), size)
    except ValueError:
        headers["Content-Range"] = f"bytes */{size}"
        return Response(status_code=416, headers=headers)
    if rng is None:
        return Response(content=data, media_type=media_type, headers=headers)
    start, end = rng
    headers["Content-Range"] = f"bytes {start}-{end}/{size}"
    return Response(
        content=data[start : end + 1],
        status_code=206,
        media_type=media_type,
        headers=headers,
    )


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
        elif src == "list":
            rows = []  # explicit cards carry a box on their own frame only
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

    target: dict[str, str] = {}

    def _append_gt() -> None:
        marks_dir = _gt_marks_dir()
        marks_dir.mkdir(parents=True, exist_ok=True)
        path = marks_dir / f"{_gt_sidecar_stem(marks_dir, clip)}.jsonl"
        target["stem"] = path.stem
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
    return {"ok": True, "written": "appended", "sidecar": target.get("stem")}


def _gt_sidecar_stem(marks_dir: Path, clip: str) -> str:
    """The GT sidecar stem a box drawn on *clip* belongs in (T-03).

    Candidates use the raw clip name (``arsenal_mancity_20250925_seg010``)
    while the clip's GT usually lives under an alias (``arsenal_mancity_seg010``
    or ``mancity_seg010``); all aliases are the same video, so frame indices
    agree. Order: the exact file if it exists, else an existing alias (the one
    named like the canonical id first, then alphabetical), else *clip* as
    given. A clip with GT never gains a second, raw-named sidecar; a clip
    without GT keeps the name its video is served under (renaming it would
    leave the new sidecar without a video for review and ingest).
    """
    if (marks_dir / f"{clip}.jsonl").exists():
        return clip
    cid = canonical_clip_id(clip)
    aliases = sorted(
        p.stem for p in marks_dir.glob("*.jsonl") if canonical_clip_id(p.stem) == cid
    )
    if aliases:
        return cid if cid in aliases else aliases[0]
    return clip


# ---------------------------------------------------------------------------
# Verdict write / undo
# ---------------------------------------------------------------------------


def _card_id_of(body: dict) -> str | None:
    """The ``id`` of the list card a verdict is for, resolved server side."""
    if (body.get("source") or "sidecar") != "list":
        return None
    try:
        it = _list_item(
            str(body["clip"]), int(body["frame_index"]), int(body["box_index"])
        )
    except (KeyError, TypeError, ValueError):
        return None
    return None if it is None or it.get("id") is None else str(it["id"])


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
        # Id of an id-keyed list card (the verdict is keyed by its hash, this
        # lets a reader tie it back to the card after the list is regenerated).
        "card_id": _card_id_of(body),
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
