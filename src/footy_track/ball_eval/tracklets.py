"""Offline tracklets over a window's detector candidates (LAB-1479 roll-forward suggestions).

``add_tracklets`` links the top-k candidates of consecutive sampled frames of ONE window (nearest neighbour with a
speed gate) and writes ``tid`` into each candidate dict. The page then follows the tracklet of the candidate nearest
George's last box with no server round trip. Gate: ``SPEED_PX`` px per video frame at 1080p (our measured human p99),
scaled by the frame gap and by height / 1080. Ids are unique within the file."""

from __future__ import annotations

import math
from pathlib import Path

SPEED_PX = 65.0
DEFAULT_SIZE = (1920, 1080)


def video_size(path: Path) -> tuple[int, int]:
    try:
        import cv2  # noqa: PLC0415

        cap = cv2.VideoCapture(str(path))
        try:
            w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        finally:
            cap.release()
        return (w, h) if w > 0 and h > 0 else DEFAULT_SIZE
    except Exception:  # noqa: BLE001
        return DEFAULT_SIZE


def _centre(c: dict) -> tuple[float, float]:
    return c["x"] + c["w"] / 2, c["y"] + c["h"] / 2


def add_tracklets(boxes: dict, windows: list[list[tuple[str, int]]], size: tuple[int, int], start_id: int = 1) -> int:
    """Set ``tid`` on every candidate (and on the top-1 entry itself) of ``boxes`` (``"clip:frame" -> entry|None``) per
    window (ordered ``(clip, frame)`` list). Returns the next free id."""
    nid = start_id
    w, h = size
    for win in windows:
        active: list[dict] = []  # {"c": (cx, cy) normalised, "f": frame, "tid": id}
        for clip, f in sorted(win, key=lambda t: t[1]):
            entry = boxes.get(f"{clip}:{f}")
            cands = (entry or {}).get("cands") or []
            pairs = []
            for ci, c in enumerate(cands):
                cx, cy = _centre(c)
                for ai, a in enumerate(active):
                    gap = max(f - a["f"], 1)
                    d = math.hypot((cx - a["c"][0]) * w, (cy - a["c"][1]) * h)
                    if d <= SPEED_PX * gap * h / 1080:
                        pairs.append((d, ci, ai))
            used_c, used_a = set(), set()
            for _, ci, ai in sorted(pairs):
                if ci in used_c or ai in used_a:
                    continue
                used_c.add(ci)
                used_a.add(ai)
                cands[ci]["tid"] = active[ai]["tid"]
            for ci, c in enumerate(cands):
                if ci not in used_c:
                    c["tid"] = nid
                    nid += 1
            # tracklets not continued this frame stay active (short gaps), newest position wins
            for ci, c in enumerate(cands):
                hit = next((a for a in active if a["tid"] == c["tid"]), None)
                if hit:
                    hit["c"], hit["f"] = _centre(c), f
                else:
                    active.append({"c": _centre(c), "f": f, "tid": c["tid"]})
            active = [a for a in active if f - a["f"] <= 30]
            if cands:
                entry["tid"] = cands[0]["tid"]
    return nid
