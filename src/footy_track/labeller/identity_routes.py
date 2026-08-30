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


@router.get("/tracklets")
async def list_tracklets(clip: str) -> dict:
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

    reviews = {rv.tracklet.key(): rv for rv in load_tracklet_reviews(_labels_dir())}
    out = []
    for tid, dets in sorted(by_track.items()):
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
                "checked_fraction": (
                    round(rv.checked_frame_count() / span, 3) if rv and span else 0.0
                ),
            }
        )
    return {"clip": safe, "tracklets": out}


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
    return {"clip": safe, "track_id": track_id, "frames": rank_risky_frames(risks, k)}


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
    ok, buf = cv2.imencode(".jpg", img[y1:y2, x1:x2])
    if not ok:
        return Response(status_code=500)
    return Response(content=buf.tobytes(), media_type="image/jpeg")


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
        annotator=body.get("annotator", "human"),
    )
    append_tracklet_review(_labels_dir(), review)
    return {"ok": True, "is_pure": review.is_pure(),
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
