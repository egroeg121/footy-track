"""Identity-review routes: tracklet verification and cross-clip merging.

Mounted on the same FastAPI app as the box labeller, but deliberately sharing
none of its state.

Why a separate page rather than an extension of ``/labeller``:

* **Different unit.** The box labeller's state is ``timeline[frame] -> boxes``.
  Identity works on tracklets — roughly 30 per clip against ~6,500 boxes, a
  250x reduction in decisions. Reviewing tracklets through a per-frame timeline
  would throw that saving away.
* **Different scope.** ``/frame/{idx}.jpg`` serves the single global
  ``SESSION``. Deciding whether two tracklets are the same player needs frames
  from *two different clips at once*, which a one-clip session cannot express.
  Hence the clip-scoped crop route here.
* **Different risk.** The box labeller rewrites whole sidecars on a timer, the
  path that once destroyed 3,348 rows of ground truth. Identity labels are
  append-only (see ``identity/store.py``) and must not share that machinery.

These routes are read-mostly and hold no session: every request names its clip.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import cv2
import numpy as np
from fastapi import APIRouter, Response
from fastapi.responses import HTMLResponse

from footy_track.identity.clusters import build_clusters
from footy_track.identity.labels import (
    CheckedInterval,
    PairLabel,
    TrackletRef,
    TrackletReview,
    Verdict,
)
from footy_track.identity.sampling import FrameRisk, rank_risky_frames
from footy_track.identity.store import (
    append_pair_label,
    append_tracklet_review,
    load_pair_labels,
    load_tracklet_reviews,
)

router = APIRouter(prefix="/identity")

_STATIC_DIR = Path(__file__).parent / "web"
# Clip names come from the URL; keep them to a safe charset so a crafted name
# cannot escape the clips directory.
_SAFE_CLIP = re.compile(r"^[A-Za-z0-9._-]+$")


def _clips_dir() -> Path:
    from footy_track.labeller import server  # noqa: PLC0415 - resolved at call time

    return Path(server._CLIPS_DIR)


def _labels_dir() -> Path:
    from footy_track.labeller import server  # noqa: PLC0415

    return Path(server._GT_MARKS_DIR).parent / "identity_labels"


def _tracklets_dir() -> Path:
    """Where tracker output lives: <clip>.jsonl of detections carrying track_id."""
    from footy_track.labeller import server  # noqa: PLC0415

    return Path(server._GT_MARKS_DIR).parent / "tracklets"


def _safe_clip(clip: str) -> str | None:
    return clip if _SAFE_CLIP.match(clip or "") else None


def _load_tracklet_rows(clip: str) -> list[dict]:
    path = _tracklets_dir() / f"{clip}.jsonl"
    if not path.exists():
        return []
    out = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


@router.get("", response_class=HTMLResponse)
@router.get("/", response_class=HTMLResponse)
async def identity_page() -> HTMLResponse:
    page = _STATIC_DIR / "identity.html"
    if not page.exists():
        return HTMLResponse("<h1>identity.html missing</h1>", status_code=404)
    return HTMLResponse(page.read_text())


@router.get("/merge", response_class=HTMLResponse)
async def merge_page() -> HTMLResponse:
    """Cross-clip merging, deliberately a separate page from tracklet review.

    The two tasks share no state and have different rhythms: review is a fast
    linear pass over one clip, merging is a comparison across clips. Putting
    them side by side split the reviewer's attention between two unrelated
    questions.
    """
    page = _STATIC_DIR / "merge.html"
    if not page.exists():
        return HTMLResponse("<h1>merge.html missing</h1>", status_code=404)
    return HTMLResponse(page.read_text())


@router.get("/tracklets")
async def list_tracklets(clip: str, min_frames: int = 25) -> dict:
    """Summarise the tracklets in one clip, with review state attached.

    ``checked_fraction`` is deliberately surfaced: it turns "I reviewed this"
    into a quantified claim, and stops a tracklet drifting toward looking
    fully-verified after a couple of partial passes.
    """
    safe = _safe_clip(clip)
    if safe is None:
        return {"error": "bad clip name", "tracklets": []}
    rows = _load_tracklet_rows(safe)
    by_track: dict[int, list[dict]] = {}
    for r in rows:
        tid = r.get("track_id")
        if tid is None:
            continue
        by_track.setdefault(int(tid), []).append(r)

    # Drop very short tracklets by default. A clip yields ~1400 tracklets of
    # which ~640 last a second or more; the rest are single-frame detection
    # noise. Offering them all spends the reviewer's attention on fragments
    # that carry no identity information and cannot be judged anyway.
    reviews = {rv.tracklet.key(): rv for rv in load_tracklet_reviews(_labels_dir())}
    out = []
    n_hidden = 0
    for tid, dets in sorted(by_track.items()):
        if len(dets) < max(0, min_frames):
            n_hidden += 1
            continue
        frames = sorted(int(d["frame_index"]) for d in dets)
        ref = TrackletRef(clip=safe, track_id=tid)
        rv = reviews.get(ref.key())
        span = frames[-1] - frames[0] + 1 if frames else 0
        out.append(
            {
                "clip": safe,
                "track_id": tid,
                "label": (dets[0].get("tags") or ["?"])[0],
                "start_frame": frames[0] if frames else None,
                "end_frame": frames[-1] if frames else None,
                "n_detections": len(dets),
                "reviewed": rv is not None,
                "is_pure": rv.is_pure() if rv else None,
                "unsure": rv.unsure if rv else None,
                "jersey": rv.jersey_number if rv else None,
                "checked_fraction": (
                    round(rv.checked_frame_count() / span, 3) if rv and span else 0.0
                ),
            }
        )
    # Longest first: they cover the most detections per decision, and a switch
    # in a long tracklet corrupts far more training pairs than one in a short.
    out.sort(key=lambda t: -t["n_detections"])
    return {"clip": safe, "tracklets": out, "hidden_short": n_hidden,
            "min_frames": min_frames}


@router.get("/risky-frames")
async def risky_frames(clip: str, track_id: int, k: int = 12) -> dict:
    """Which frames of a tracklet a human should actually look at.

    An ID switch happens at one moment; a uniform sample of 12 frames from 288
    usually misses it. Ranking by crowding / low confidence / low association
    margin puts limited attention where the evidence is.
    """
    safe = _safe_clip(clip)
    if safe is None:
        return {"error": "bad clip name", "frames": []}
    rows = [r for r in _load_tracklet_rows(safe) if r.get("track_id") == track_id]
    per_frame_counts: dict[int, int] = {}
    for r in _load_tracklet_rows(safe):
        per_frame_counts[int(r["frame_index"])] = (
            per_frame_counts.get(int(r["frame_index"]), 0) + 1
        )
    risks = [
        FrameRisk(
            frame_index=int(r["frame_index"]),
            crowding=per_frame_counts.get(int(r["frame_index"]), 0),
            confidence=float(r.get("confidence", 1.0)),
            association_margin=float(r.get("association_margin", 1.0)),
        )
        for r in rows
    ]
    ranked = rank_risky_frames(risks, k)
    # Return the box with each frame: without it the client cannot build a crop
    # URL and silently falls back to the full frame, which shows 12 identical
    # wide shots of the pitch instead of one player.
    box_at = {int(r["frame_index"]): r.get("bbox") for r in rows}
    return {
        "clip": safe,
        "track_id": track_id,
        "frames": ranked,
        "boxes": [box_at.get(f) for f in ranked],
    }


@router.get("/best-frame")
async def best_frame(clip: str, track_id: int, n: int = 3) -> dict:
    """Frames most likely to show a READABLE BACK NUMBER.

    Box size alone is the wrong criterion: a large crop of a player facing the
    camera shows no number at all. Orientation is the binding constraint, and
    size only decides whether the digits have enough pixels once they are
    actually pointing at you.

    Without a pose model the usable proxy is motion direction — a player moving
    up-screen is running away from camera, so their back is visible. Two
    corrections make that workable:

    * **Camera compensation.** A pan moves every box together, so raw vertical
      motion mostly measures the camera. The median motion of all tracks in the
      same frame is subtracted, leaving motion relative to the play.
    * **A window, not a frame pair.** Single-frame deltas are dominated by box
      jitter on a ~100px detection; motion is measured over +/-3 frames.

    This is a HEURISTIC and is not validated: players backpedal, sidestep and
    turn, and none of that is captured. So it returns the top ``n`` candidates
    rather than one answer, and the reviewer picks — glancing at three crops is
    fast and does not depend on the heuristic being right, only on it putting a
    good frame somewhere in the shortlist.
    """
    safe = _safe_clip(clip)
    if safe is None:
        return {"error": "bad clip name", "candidates": []}
    all_rows = [r for r in _load_tracklet_rows(safe) if isinstance(r.get("bbox"), dict)]
    rows = [r for r in all_rows if r.get("track_id") == track_id]
    if not rows:
        return {"candidates": []}

    cy = {}
    for r in all_rows:
        b = r["bbox"]
        cy.setdefault(int(r["frame_index"]), []).append(
            (int(r["track_id"]), b["y"] + b["h"] / 2)
        )
    track_cy = {int(r["frame_index"]): r["bbox"]["y"] + r["bbox"]["h"] / 2 for r in rows}

    def camera_dy(f0: int, f1: int) -> float:
        """Median vertical motion of all tracks present in BOTH frames."""
        a = {t: y for t, y in cy.get(f0, [])}
        b = {t: y for t, y in cy.get(f1, [])}
        both = [b[t] - a[t] for t in a.keys() & b.keys()]
        if not both:
            return 0.0
        both.sort()
        return both[len(both) // 2]

    W = 3
    cands = []
    for r in rows:
        f = int(r["frame_index"])
        f0, f1 = f - W, f + W
        if f0 not in track_cy or f1 not in track_cy:
            continue
        rel_dy = (track_cy[f1] - track_cy[f0]) - camera_dy(f0, f1)
        away = max(0.0, -rel_dy)          # negative dy (up-screen) == away from camera
        h_px = r["bbox"]["h"] * 1080
        cands.append((f, r["bbox"], h_px, away))

    if not cands:  # too short to measure motion — fall back to size alone
        best = max(rows, key=lambda r: r["bbox"]["h"])
        return {"clip": safe, "track_id": track_id, "heuristic": "size-only",
                "candidates": [{"frame": int(best["frame_index"]), "bbox": best["bbox"],
                                "height_px": round(best["bbox"]["h"] * 1080),
                                "facing_away": None}]}

    max_h = max(c[2] for c in cands) or 1.0
    max_away = max(c[3] for c in cands) or 1.0
    # Size gates legibility, orientation decides whether there is anything to
    # read. Weighted toward orientation, but never zero on size.
    scored = sorted(
        cands,
        key=lambda c: -((0.35 + 0.65 * (c[3] / max_away)) * (c[2] / max_h)),
    )
    seen, out = set(), []
    for f, bbox, h_px, away in scored:
        if any(abs(f - g) < 15 for g in seen):   # spread candidates out in time
            continue
        seen.add(f)
        out.append({"frame": f, "bbox": bbox, "height_px": round(h_px),
                    "facing_away": round(away / max_away, 2)})
        if len(out) >= max(1, n):
            break
    return {"clip": safe, "track_id": track_id, "heuristic": "orientation+size",
            "candidates": out}


@router.get("/crop/{clip}/{frame}.jpg")
async def crop(
    clip: str, frame: int, x: float = 0.0, y: float = 0.0, w: float = 1.0, h: float = 1.0,
    pad: float = 0.6,
) -> Response:
    """Serve one padded crop. Clip-scoped so two clips can be shown together.

    ``pad`` adds context around the box: a 52x111 px player crop is nearly
    unreadable in isolation, and surrounding pixels are what let a human judge
    identity at all.
    """
    safe = _safe_clip(clip)
    if safe is None:
        return Response(status_code=400)
    path = _clips_dir() / f"{safe}.mp4"
    if not path.exists():
        return Response(status_code=404)
    cap = cv2.VideoCapture(str(path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, int(frame)))
    ok, img = cap.read()
    cap.release()
    if not ok or img is None:
        return Response(status_code=404)
    H, W = img.shape[:2]
    px, py, pw, ph = x * W, y * H, w * W, h * H
    cx, cy = px + pw / 2, py + ph / 2
    half_w, half_h = pw * (1 + pad) / 2, ph * (1 + pad) / 2
    x1, y1 = int(max(0, cx - half_w)), int(max(0, cy - half_h))
    x2, y2 = int(min(W, cx + half_w)), int(min(H, cy + half_h))
    if x2 <= x1 or y2 <= y1:
        return Response(status_code=404)
    crop_img = img[y1:y2, x1:x2].copy()
    # Draw ONLY the reviewed player's box. Padding plus crowding means a crop
    # often contains two or three players, and without this the reviewer cannot
    # tell which one the tracklet is actually following -- they would judge
    # continuity on whichever player is most visible, which is not the question
    # being asked. Skipped when the request is for a full frame (no real box).
    is_full_frame = x <= 0.0 and y <= 0.0 and w >= 1.0 and h >= 1.0
    if not is_full_frame:
        bx1, by1 = int(px - x1), int(py - y1)
        bx2, by2 = int(px + pw - x1), int(py + ph - y1)
        cv2.rectangle(crop_img, (bx1, by1), (bx2, by2), (0, 235, 255), 2)
    # A median player box is ~52x111 px. Even padded that is unreadable at
    # screen size, and the whole point of the review is that a human can SEE
    # the player. Upscale so the short side is at least MIN_SIDE px.
    MIN_SIDE = 220
    ch, cw = crop_img.shape[:2]
    short = min(ch, cw)
    if short and short < MIN_SIDE:
        scale = MIN_SIDE / short
        crop_img = cv2.resize(
            crop_img, (int(cw * scale), int(ch * scale)), interpolation=cv2.INTER_CUBIC
        )
    ok, buf = cv2.imencode(".jpg", crop_img)
    if not ok:
        return Response(status_code=500)
    return Response(content=buf.tobytes(), media_type="image/jpeg")


@router.get("/track-video/{clip}/{track_id}.mp4")
async def track_video(clip: str, track_id: int, max_frames: int = 150) -> Response:
    """Render the tracked player as a short video, box drawn, player centred.

    Stills answer "are these the same person"; a video answers "does the box
    stay on them", which is the actual question and is far faster to judge --
    a switch is obvious as a jump when you watch it and easy to miss across
    twelve thumbnails.

    The window follows the box so the player stays centred, otherwise the
    subject walks out of a fixed crop and the reviewer is tracking the tracker
    by eye. Decoding is SEQUENTIAL over the tracklet span: per-frame seeking on
    a 1080p H.264 file is orders of magnitude slower, and this runs on a
    machine with no GPU. Long tracklets are temporally subsampled to
    ``max_frames`` so cost is bounded by the cap, not the tracklet length.

    Rendered videos are cached on disk: review revisits the same tracklet often
    and re-decoding each time would dominate the interaction.
    """
    safe = _safe_clip(clip)
    if safe is None:
        return Response(status_code=400)
    path = _clips_dir() / f"{safe}.mp4"
    if not path.exists():
        return Response(status_code=404)

    cache_dir = _labels_dir().parent / "track_videos"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cached = cache_dir / f"{safe}__{track_id}.mp4"
    if cached.exists() and cached.stat().st_size > 0:
        return Response(content=cached.read_bytes(), media_type="video/mp4")

    rows = [r for r in _load_tracklet_rows(safe) if r.get("track_id") == track_id]
    boxes = {int(r["frame_index"]): r.get("bbox") for r in rows if r.get("bbox")}
    if not boxes:
        return Response(status_code=404)

    frames = sorted(boxes)
    step = max(1, len(frames) // max_frames)
    wanted = set(frames[::step])

    OUT = 320  # square output; big enough to judge, small enough to stay cheap
    cap = cv2.VideoCapture(str(path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frames[0])
    tmp = cached.with_suffix(".tmp.mp4")
    writer = cv2.VideoWriter(str(tmp), cv2.VideoWriter_fourcc(*"avc1"), 10, (OUT, OUT))
    if not writer.isOpened():  # avc1 is not always available; mp4v always is
        writer = cv2.VideoWriter(str(tmp), cv2.VideoWriter_fourcc(*"mp4v"), 10, (OUT, OUT))

    idx, written = frames[0], 0
    while idx <= frames[-1]:
        ok, img = cap.read()
        if not ok:
            break
        if idx in wanted:
            H, W = img.shape[:2]
            b = boxes[idx]
            cx, cy = (b["x"] + b["w"] / 2) * W, (b["y"] + b["h"] / 2) * H
            half = max(b["w"] * W, b["h"] * H) * 1.6
            half = max(half, 60.0)
            x1, y1 = int(cx - half), int(cy - half)
            x2, y2 = int(cx + half), int(cy + half)
            pad_l, pad_t = max(0, -x1), max(0, -y1)
            pad_r, pad_b = max(0, x2 - W), max(0, y2 - H)
            win = img[max(0, y1):min(H, y2), max(0, x1):min(W, x2)]
            if win.size == 0:
                idx += 1
                continue
            if pad_l or pad_t or pad_r or pad_b:
                # Pad rather than shift the window: shifting would move the
                # player off-centre exactly when they are near the touchline.
                win = cv2.copyMakeBorder(win, pad_t, pad_b, pad_l, pad_r,
                                         cv2.BORDER_CONSTANT, value=(20, 20, 20))
            scale = OUT / max(win.shape[0], win.shape[1])
            win = cv2.resize(win, (int(win.shape[1] * scale), int(win.shape[0] * scale)))
            canvas = np.full((OUT, OUT, 3), 20, dtype=np.uint8)
            oy, ox = (OUT - win.shape[0]) // 2, (OUT - win.shape[1]) // 2
            canvas[oy:oy + win.shape[0], ox:ox + win.shape[1]] = win
            # Box in canvas coords.
            bx1 = int((b["x"] * W - max(0, x1) + pad_l) * scale) + ox
            by1 = int((b["y"] * H - max(0, y1) + pad_t) * scale) + oy
            bx2 = int(((b["x"] + b["w"]) * W - max(0, x1) + pad_l) * scale) + ox
            by2 = int(((b["y"] + b["h"]) * H - max(0, y1) + pad_t) * scale) + oy
            cv2.rectangle(canvas, (bx1, by1), (bx2, by2), (0, 235, 255), 2)
            cv2.putText(canvas, str(idx), (6, OUT - 8), cv2.FONT_HERSHEY_SIMPLEX,
                        0.45, (200, 200, 200), 1, cv2.LINE_AA)
            writer.write(canvas)
            written += 1
        idx += 1
    cap.release()
    writer.release()
    if not written or not tmp.exists():
        return Response(status_code=404)
    tmp.replace(cached)
    return Response(content=cached.read_bytes(), media_type="video/mp4")


@router.post("/pair")
async def label_pair(body: dict) -> dict:
    """Record a same/different/unknown verdict and re-derive the clusters.

    Contradictions are returned rather than resolved: if A~B, B~C but A!=C, one
    human verdict is wrong and only a human can say which. Auto-resolving would
    permanently fuse two players and leave no trace.
    """
    try:
        a = TrackletRef(clip=body["a"]["clip"], track_id=int(body["a"]["track_id"]))
        b = TrackletRef(clip=body["b"]["clip"], track_id=int(body["b"]["track_id"]))
        verdict = Verdict(body["verdict"])
    except (KeyError, ValueError, TypeError) as exc:
        return {"ok": False, "error": f"bad payload: {exc}"}
    append_pair_label(
        _labels_dir(), PairLabel(a=a, b=b, verdict=verdict,
                                 annotator=body.get("annotator", "human"))
    )
    res = build_clusters(load_pair_labels(_labels_dir()))
    return {
        "ok": True,
        "n_clusters": res.n_clusters,
        "n_unknown": res.n_unknown,
        "contradictions": [
            [{"clip": p.clip, "track_id": p.track_id} for p in pair]
            for pair in res.contradictions
        ],
    }


@router.post("/review")
async def submit_review(body: dict) -> dict:
    """Record which intervals of a tracklet a human actually inspected."""
    try:
        tracklet = TrackletRef(clip=body["clip"], track_id=int(body["track_id"]))
        intervals = [CheckedInterval(int(s), int(e)) for s, e in body.get("checked", [])]
    except (KeyError, ValueError, TypeError) as exc:
        return {"ok": False, "error": f"bad payload: {exc}"}
    review = TrackletReview(
        tracklet=tracklet,
        checked_intervals=intervals,
        split_at=[int(f) for f in body.get("split_at", [])],
        unsure=bool(body.get("unsure", False)),
        jersey_number=(str(body.get("jersey_number") or "").strip() or None),
        annotator=body.get("annotator", "human"),
    )
    append_tracklet_review(_labels_dir(), review)
    return {"ok": True, "is_pure": review.is_pure(), "unsure": review.unsure,
            "jersey_number": review.jersey_number,
            "checked_frames": review.checked_frame_count()}


@router.get("/clusters")
async def clusters() -> dict:
    """Current identity clusters, plus any contradictions blocking trust."""
    labels = load_pair_labels(_labels_dir())
    res = build_clusters(labels)
    grouped: dict[int, list[dict]] = {}
    for key, cid in res.clusters.items():
        grouped.setdefault(cid, []).append({"clip": key[0], "track_id": key[1]})
    return {
        "n_labels": len(labels),
        "n_clusters": res.n_clusters,
        "n_unknown": res.n_unknown,
        "is_consistent": res.is_consistent,
        "contradictions": [
            [{"clip": p.clip, "track_id": p.track_id} for p in pair]
            for pair in res.contradictions
        ],
        "clusters": grouped,
    }
