"""Calibration API: manual camera -> pitch calibration (M0 of
``docs/design/calibration.md``) and the calibration ground truth it produces.

An operator pairs points on a video frame with named pitch keypoints (or free
pitch coordinates); the server fits the homography, reprojects the full pitch
model into the image and renders a bird's-eye warp, so the fit can be judged by
eye. Saved pairs are *calibration GT* used to score the automatic calibrator
(M1+).

Storage: ``<gt dir>/calibration_gt/<canonical clip id>.json`` - one JSON file
per clip, ``{"clip", "frames": {"<frame_index>": record}}``, written atomically
(tmp file + rename; the GT sidecars have a history of being truncated). The
clip id is ``footy_track.clip_id.canonical_clip_id`` so every alias of a clip
shares one file. Nothing is written to the GT sidecars or the feature store.

H convention: normalised image coords -> pitch metres (see
``footy_track.calibration``). ``server.py`` is the composition root: clip and
GT directories resolve through it at call time (tests monkeypatch them there).
See ``src/footy_track/labeller/README.md`` §16 (LAB-12xx).
"""

from __future__ import annotations

import asyncio
import collections
import copy
import json
import logging
import os
import tempfile
import threading
import time
from pathlib import Path

import cv2
import numpy as np
from fastapi import APIRouter
from fastapi.responses import JSONResponse, Response

from footy_track.calibration import geometry as geo
from footy_track.calibration import lens
from footy_track.calibration.pitch_model import (
    DEFAULT_LENGTH,
    DEFAULT_WIDTH,
    PitchModel,
)
from footy_track.calibration.frame_type import (
    NOT_BROADCAST,
    TOO_FEW_POINTS,
    VISIBLE_TAGS,
    all_too_few_points,
    not_broadcast_frames,
    too_few_points_frames,
    update_frame_types,
)
from footy_track.calibration.prelabel.proposals import read_proposals, update_proposals
from footy_track.calibration.video import read_frame, read_frame_exact
from footy_track.clip_id import canonical_clip_id
from footy_track.labeller import datasets, hooks, matches_status
from footy_track.labeller.match_segments import SEGMENTS_DIR, find_segment_video
from footy_track.labeller.review import _clips_dir, _gt_marks_dir
from footy_track.labeller.review import _find_video as _find_video_exact

router = APIRouter()
LOGGER = logging.getLogger(__name__)

#: Bird's-eye raster scale and margin; the page draws its diagram with the
#: same extents so the warp lines up with the pitch drawing.
WARP_PX_PER_M = 10.0
WARP_MARGIN_M = 3.0
MAX_WARP_PX_PER_M = 20.0

#: mkstemp gives 0600; other GT files are group-writable.
_NEW_FILE_MODE = 0o664

_VIDEO_SUFFIXES = {".mp4", ".mov", ".avi", ".mkv"}


def _err(status: int, msg: str) -> JSONResponse:
    # Real HTTP status codes: a 200 {"ok": false} makes a failing save look
    # like a good one in the access log (see the KB labeller gotchas).
    return JSONResponse({"ok": False, "error": msg}, status_code=status)


# ---------------------------------------------------------------------------
# Video access (patched in tests)
# ---------------------------------------------------------------------------


def _video_meta(video: Path) -> dict | None:
    cap = cv2.VideoCapture(str(video))
    try:
        if not cap.isOpened():
            return None
        return {
            "frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            "fps": float(cap.get(cv2.CAP_PROP_FPS)),
            "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        }
    finally:
        cap.release()


#: How a saved frame index maps to an image, recorded per frame in the GT sidecar (``frame_index_convention``).
#: ``cv2_seek``: ``cv2`` ``CAP_PROP_POS_FRAMES`` seek (every label made before the field existed; not frame-accurate
#: on long-GOP / variable-frame-rate segments, e.g. Bournemouth seg087 is up to ~170 frames off). ``decoded``: the N-th
#: frame of a sequential decode (``read_frame_exact``), the numbering of the PTZ match files and ``/calibration/match``.
CONV_SEEK = "cv2_seek"
CONV_DECODED = "decoded"
_WARNED_SEEK: set[str] = set()


def _legacy_gt(clip: str) -> bool:
    """Whether ``clip``'s GT sidecar (active dataset) holds a frame NOT marked ``decoded``: a frame without the
    ``frame_index_convention`` field was saved against the cv2 seek, so the whole clip stays on the seek."""
    try:
        frames = _read_gt(clip).get("frames") or {}
    except (OSError, ValueError):
        return False
    return any(not isinstance(r, dict) or r.get("frame_index_convention") != CONV_DECODED for r in frames.values())


def _convention_of_video(video: Path) -> str:
    """Frame-index convention the editor uses for ``video``: ``decoded`` only for a full-match segment (found by
    the fallback of :func:`_find_video`, i.e. in no dataset clip dir) whose clip has no seek-indexed GT."""
    if video.parent.name != SEGMENTS_DIR:
        return CONV_SEEK
    clip = canonical_clip_id(video.stem)
    if _legacy_gt(clip):
        if clip not in _WARNED_SEEK:
            _WARNED_SEEK.add(clip)
            LOGGER.warning(
                "calibration: %s has GT saved without frame_index_convention=decoded; reading its segment video "
                "with the cv2 seek (not the decoded index of /calibration/match)", clip)
        return CONV_SEEK
    return CONV_DECODED


def frame_convention(clip: str) -> str | None:
    """``decoded`` / ``cv2_seek`` for the video the editor would open for ``clip``; ``None`` when it has none."""
    video = _find_video(clip)
    return None if video is None else _convention_of_video(video)


def _read_frame(video: Path, idx: int) -> np.ndarray | None:
    if _convention_of_video(video) == CONV_DECODED:
        return read_frame_exact(video, idx)
    # cv2 seek, with an ffmpeg fallback for clips OpenCV decodes as black.
    return read_frame(video, idx)


def _video_ref(video: Path) -> str:
    """The resolved video path relative to ``footy_data`` (absolute when it lives elsewhere)."""
    try:
        real = video.resolve()
        return str(real.relative_to(matches_status.data_root().resolve()))
    except (OSError, ValueError):
        return str(video)


# ---------------------------------------------------------------------------
# GT store
# ---------------------------------------------------------------------------


def _gt_dir() -> Path:
    return _gt_marks_dir() / "calibration_gt"


def _gt_path(clip: str) -> Path:
    return _gt_dir() / f"{canonical_clip_id(clip)}.json"


def _read_gt(clip: str) -> dict:
    path = _gt_path(clip)
    if not path.exists():
        return {"clip": canonical_clip_id(clip), "frames": {}}
    return json.loads(path.read_text())


def _write_gt(clip: str, doc: dict) -> None:
    path = _gt_path(clip)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        mode = path.stat().st_mode & 0o777  # keep an existing file's mode
    except FileNotFoundError:
        mode = _NEW_FILE_MODE
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=path.name, suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(doc, f, indent=1)
        os.chmod(tmp, mode)  # mkstemp creates 0600; other GT files are 0664
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


#: Serialises every calibration-GT read-modify-write in this process (save, delete, accept a
#: proposal, flag/unflag not-broadcast): two of them racing on one clip would each read the same
#: document and the second write would silently drop the first's frame. Same pattern as
#: ``frame_type.update_frame_types`` / ``proposals.update_proposals``.
_GT_LOCK = threading.Lock()


def update_gt(clip: str, fn):
    """Read-modify-write ``clip``'s GT document under :data:`_GT_LOCK`; returns ``fn(doc)``.

    ``fn`` mutates the document in place; nothing is written when it left the document unchanged.
    Never call another locked store (frame types, proposals) from inside ``fn``: do that after
    this returns.
    """
    with _GT_LOCK:
        doc = _read_gt(clip)
        before = copy.deepcopy(doc)
        out = fn(doc)
        if doc != before:
            _write_gt(clip, doc)
        return out


def _gt_frames(clip: str) -> list[int]:
    try:
        return sorted(int(k) for k in _read_gt(clip)["frames"])
    except (OSError, ValueError, KeyError):
        return []


# ---------------------------------------------------------------------------
# Frame type flag (not a broadcast pitch view; see calibration/frame_type.py)
# ---------------------------------------------------------------------------


def _not_broadcast(clip: str) -> list[int]:
    try:
        return sorted(not_broadcast_frames(_gt_marks_dir(), clip))
    except OSError:
        return []


def mark_not_broadcast(clip: str, idx: int, *, overwrite: bool = False, run_id: str | None = None) -> dict:
    """Flag frame ``idx`` as not a broadcast pitch view.

    Writes the frame-type sidecar and sets the frame's proposals to ``not_broadcast``
    (the named ``run_id``'s whatever its status; other runs' only while still
    ``proposed``/``failed``). A frame that already has calibration GT (hand or accepted)
    is refused with ``{"conflict": <existing GT record>}`` and nothing is written, unless
    ``overwrite``, which moves that GT record into the flag (``displaced_gt``) so
    :func:`clear_not_broadcast` can put it back: GT is never destroyed by this action.
    """
    existing = _read_gt(clip)["frames"].get(str(idx))
    if existing is not None and not overwrite:
        return {"conflict": existing}
    ts = time.strftime("%Y-%m-%dT%H:%M:%S%z")

    def _flag(doc: dict) -> None:
        rec = {"type": NOT_BROADCAST, "ts": ts}
        if run_id:
            rec["run_id"] = run_id
        if existing is not None:
            rec["displaced_gt"] = existing
        doc["frames"][str(idx)] = rec

    update_frame_types(_gt_marks_dir(), clip, _flag)
    removed_gt = False
    if existing is not None:
        removed_gt = update_gt(clip, lambda doc: doc["frames"].pop(str(idx), None) is not None)

    def _props(doc: dict) -> None:
        for rid, run in doc["runs"].items():
            f = run["frames"].get(str(idx))
            if f is None or f["status"] == NOT_BROADCAST:
                continue
            if rid == run_id or f["status"] in ("proposed", "failed"):
                f["prev_status"] = f["status"]
                f["status"] = NOT_BROADCAST
                f["reviewed_at"] = ts

    if read_proposals(_gt_marks_dir(), clip)["runs"]:
        update_proposals(_gt_marks_dir(), clip, _props)
    hooks.emit("frame_type", clip=canonical_clip_id(clip), unit=idx, verdict=NOT_BROADCAST)
    return {"ok": True, "frame_index": idx, "removed_gt": removed_gt, "gt_frames": _gt_frames(clip)}


def clear_not_broadcast(clip: str, idx: int) -> bool:
    """Undo ``mark_not_broadcast``; proposals go back to their previous status and any GT
    the mark displaced is restored (unless the frame has new GT since). ``False`` when the
    frame was not flagged."""
    gone: list[bool] = []
    displaced: list[dict] = []

    def _unflag(doc: dict) -> None:
        rec = doc["frames"].get(str(idx))
        if rec is not None and rec.get("type", NOT_BROADCAST) != NOT_BROADCAST:
            rec = None   # a different flag (too_few_points): not ours to clear
        else:
            doc["frames"].pop(str(idx), None)
        gone.append(rec is not None)
        if rec and rec.get("displaced_gt"):
            displaced.append(rec["displaced_gt"])

    update_frame_types(_gt_marks_dir(), clip, _unflag)
    if displaced:
        update_gt(clip, lambda doc: doc["frames"].setdefault(str(idx), displaced[0]))

    def _props(doc: dict) -> None:
        for run in doc["runs"].values():
            f = run["frames"].get(str(idx))
            if f is not None and f["status"] == NOT_BROADCAST:
                f["status"] = f.pop("prev_status", None) or ("proposed" if f.get("H") is not None else "failed")
                f.pop("reviewed_at", None)

    if read_proposals(_gt_marks_dir(), clip)["runs"]:
        update_proposals(_gt_marks_dir(), clip, _props)
    if gone[0]:
        hooks.emit("frame_type", clip=canonical_clip_id(clip), unit=idx, verdict=f"{NOT_BROADCAST}_cleared")
    return gone[0]


@router.get("/calibration/api/frame_type/{clip}")
async def frame_type_list(clip: str) -> dict:
    return {"clip": clip, "not_broadcast": await asyncio.to_thread(_not_broadcast, clip)}


@router.put("/calibration/api/frame_type/{clip}/{idx}")
async def frame_type_set(clip: str, idx: int, body: dict | None = None):
    """Flag the frame ``not_broadcast`` (body ``{overwrite?, run_id?}``); 409 over existing GT."""
    body = body or {}
    if idx < 0:
        return _err(400, "frame_index must be >= 0")
    if body.get("type", NOT_BROADCAST) != NOT_BROADCAST:
        return _err(400, f"type must be {NOT_BROADCAST!r}")
    res = await asyncio.to_thread(
        lambda: mark_not_broadcast(clip, idx, overwrite=bool(body.get("overwrite")), run_id=body.get("run_id"))
    )
    if "conflict" in res:
        return _not_broadcast_conflict(res["conflict"])
    return res


@router.delete("/calibration/api/frame_type/{clip}/{idx}")
async def frame_type_clear(clip: str, idx: int):
    if not await asyncio.to_thread(clear_not_broadcast, clip, idx):
        return _err(404, "this frame is not marked not_broadcast")
    return {"ok": True, "not_broadcast": await asyncio.to_thread(_not_broadcast, clip)}


def _not_broadcast_conflict(existing: dict) -> JSONResponse:
    human = not existing.get("origin")
    what = "hand-labelled" if human else "accepted machine"
    return JSONResponse(
        {"ok": False, "existing_gt": "human" if human else "machine",
         "error": f"this frame already has {what} calibration GT; marking it not broadcast "
                  "would set that GT aside (Undo restores it). Resend with overwrite=true to do so"},
        status_code=409,
    )


# ---------------------------------------------------------------------------
# Too few points (frame shows < 4 identifiable pitch features; PTZ-solver candidate)
# ---------------------------------------------------------------------------


def _too_few_points(clip: str) -> dict[str, list[str]]:
    """``{"<frame_index>": [tags]}`` of the clip's too-few-points frames (JSON-friendly)."""
    try:
        return {str(k): v["tags"] for k, v in sorted(too_few_points_frames(_gt_marks_dir(), clip).items())}
    except OSError:
        return {}


def _clean_tags(tags: object) -> list[str]:
    if tags is None:
        return []
    if not isinstance(tags, list) or not all(isinstance(t, str) for t in tags):
        raise _BadRequest("tags must be a list of strings")
    bad = [t for t in tags if t not in VISIBLE_TAGS]
    if bad:
        raise _BadRequest(f"unknown tags {bad}; allowed: {list(VISIBLE_TAGS)}")
    return [t for t in VISIBLE_TAGS if t in tags]   # canonical order, de-duplicated


def mark_too_few_points(clip: str, idx: int, *, tags: list[str] | None = None, run_id: str | None = None) -> dict:
    """Flag frame ``idx`` as having too few identifiable pitch points (<4 pairs).

    Writes the frame-type sidecar (``type=too_few_points`` + optional ``tags``) and sets the
    frame's proposals to ``too_few_points`` (the named ``run_id``'s whatever its status;
    other runs' only while still ``proposed``/``failed``), keeping each proposal record and
    remembering ``prev_status``. Marking an already-marked frame just replaces its tags.
    Refused with ``{"conflict": ...}`` when the frame has calibration GT (it evidently has
    enough points) or is flagged not_broadcast; nothing is written then.
    """
    gt = _read_gt(clip)["frames"].get(str(idx))
    if gt is not None:
        return {"conflict": "gt", "existing_gt": "human" if not gt.get("origin") else "machine"}
    if idx in not_broadcast_frames(_gt_marks_dir(), clip):
        return {"conflict": "not_broadcast"}
    ts = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    tags = tags or []

    def _flag(doc: dict) -> None:
        old = doc["frames"].get(str(idx)) or {}
        rec = {"type": TOO_FEW_POINTS, "ts": old.get("ts") or ts, "tags": tags}
        if run_id or old.get("run_id"):
            rec["run_id"] = run_id or old["run_id"]
        doc["frames"][str(idx)] = rec

    update_frame_types(_gt_marks_dir(), clip, _flag)

    def _props(doc: dict) -> None:
        for rid, run in doc["runs"].items():
            f = run["frames"].get(str(idx))
            if f is None or f["status"] == NOT_BROADCAST:
                continue
            if f["status"] == TOO_FEW_POINTS:
                f["tags"] = tags
            elif rid == run_id or f["status"] in ("proposed", "failed"):
                f["prev_status"] = f["status"]
                f["status"] = TOO_FEW_POINTS
                f["reviewed_at"] = ts
                f["tags"] = tags

    if read_proposals(_gt_marks_dir(), clip)["runs"]:
        update_proposals(_gt_marks_dir(), clip, _props)
    hooks.emit("frame_type", clip=canonical_clip_id(clip), unit=idx, verdict=TOO_FEW_POINTS)
    return {"ok": True, "frame_index": idx, "tags": tags, "too_few_points": _too_few_points(clip)}


def clear_too_few_points(clip: str, idx: int) -> bool:
    """Undo ``mark_too_few_points``; proposals go back to their previous status. ``False``
    when the frame was not flagged."""
    gone: list[bool] = []

    def _unflag(doc: dict) -> None:
        rec = doc["frames"].get(str(idx))
        if rec is not None and rec.get("type") == TOO_FEW_POINTS:
            del doc["frames"][str(idx)]
            gone.append(True)
        else:
            gone.append(False)

    update_frame_types(_gt_marks_dir(), clip, _unflag)

    def _props(doc: dict) -> None:
        for run in doc["runs"].values():
            f = run["frames"].get(str(idx))
            if f is not None and f["status"] == TOO_FEW_POINTS:
                f["status"] = f.pop("prev_status", None) or ("proposed" if f.get("H") is not None else "failed")
                f.pop("reviewed_at", None)
                f.pop("tags", None)

    if read_proposals(_gt_marks_dir(), clip)["runs"]:
        update_proposals(_gt_marks_dir(), clip, _props)
    if gone[0]:
        hooks.emit("frame_type", clip=canonical_clip_id(clip), unit=idx, verdict=f"{TOO_FEW_POINTS}_cleared")
    return gone[0]


def _too_few_conflict(res: dict) -> JSONResponse:
    if res["conflict"] == "not_broadcast":
        msg = "this frame is marked not broadcast; clear that mark (Undo) before marking it too few points"
        return JSONResponse({"ok": False, "error": msg}, status_code=409)
    what = "hand-labelled" if res.get("existing_gt") == "human" else "accepted machine"
    return JSONResponse(
        {"ok": False, "existing_gt": res.get("existing_gt"),
         "error": f"this frame already has {what} calibration GT, so it has enough points; "
                  "delete that GT first if you really want to mark it too few points"},
        status_code=409,
    )


def _tags_or_err(body: dict):
    try:
        return _clean_tags(body.get("tags")), None
    except _BadRequest as exc:
        return None, _err(400, str(exc))


@router.get("/calibration/api/too_few_points")
async def too_few_points_all() -> dict:
    """Every too-few-points frame of every clip (canonical ids): for the PTZ proposer."""
    allc = await asyncio.to_thread(lambda: all_too_few_points(_gt_marks_dir()))
    clips = {c: [{"frame_index": i, **v} for i, v in sorted(fr.items())] for c, fr in allc.items()}
    return {"total": sum(len(v) for v in clips.values()), "clips": clips}


@router.get("/calibration/api/too_few_points/{clip}")
async def too_few_points_list(clip: str) -> dict:
    fr = await asyncio.to_thread(lambda: too_few_points_frames(_gt_marks_dir(), clip))
    return {"clip": canonical_clip_id(clip),
            "frames": [{"frame_index": i, **v} for i, v in sorted(fr.items())],
            "too_few_points": await asyncio.to_thread(_too_few_points, clip)}


@router.put("/calibration/api/too_few_points/{clip}/{idx}")
async def too_few_points_set(clip: str, idx: int, body: dict | None = None):
    """Flag the frame ``too_few_points`` (body ``{tags?, run_id?}``); PUT again to change the tags."""
    body = body or {}
    if idx < 0:
        return _err(400, "frame_index must be >= 0")
    tags, err = _tags_or_err(body)
    if err is not None:
        return err
    res = await asyncio.to_thread(lambda: mark_too_few_points(clip, idx, tags=tags, run_id=body.get("run_id")))
    if "conflict" in res:
        return _too_few_conflict(res)
    return res


@router.delete("/calibration/api/too_few_points/{clip}/{idx}")
async def too_few_points_clear(clip: str, idx: int):
    if not await asyncio.to_thread(clear_too_few_points, clip, idx):
        return _err(404, "this frame is not marked too_few_points")
    return {"ok": True, "too_few_points": await asyncio.to_thread(_too_few_points, clip)}



# ---------------------------------------------------------------------------
# Request parsing
# ---------------------------------------------------------------------------


class _BadRequest(Exception):
    pass


def _pitch_from(body: dict) -> PitchModel:
    dims = body.get("pitch_dims") or [DEFAULT_LENGTH, DEFAULT_WIDTH]
    try:
        return PitchModel(float(dims[0]), float(dims[1]))
    except (TypeError, ValueError, IndexError) as exc:
        raise _BadRequest(f"bad pitch_dims: {exc}") from exc


def _parse_correspondences(
    raw: object, pitch: PitchModel
) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    """-> (image norm (N,2), pitch metres (N,2), cleaned records).

    Each item: ``{"image": [u, v]}`` plus ``"pitch_id"`` (a keypoint id) or
    ``"pitch_xy"`` (free metres). A ``pitch_id`` wins and its metres are
    resolved here, from the given dims, so the stored xy is never stale.
    """
    if not isinstance(raw, list):
        raise _BadRequest("correspondences must be a list")
    kps = pitch.keypoints()
    img, ptc, clean = [], [], []
    for i, c in enumerate(raw):
        try:
            u, v = (float(t) for t in c["image"])
            pid = c.get("pitch_id")
            if pid:
                if pid not in kps:
                    raise _BadRequest(f"unknown pitch keypoint {pid!r}")
                x, y = kps[pid]
            else:
                x, y = (float(t) for t in c["pitch_xy"])
        except _BadRequest:
            raise
        except (KeyError, TypeError, ValueError) as exc:
            raise _BadRequest(f"bad correspondence #{i}: {exc}") from exc
        if not all(np.isfinite([u, v, x, y])):
            raise _BadRequest(f"non-finite value in correspondence #{i}")
        img.append([u, v])
        ptc.append([x, y])
        clean.append(
            {"image": [u, v], "pitch_id": pid or None, "pitch_xy": [x, y]}
        )
    return np.array(img).reshape(-1, 2), np.array(ptc).reshape(-1, 2), clean


#: Hartley-normalised condition number of H above which the fit is flagged.
MAX_H_COND = 100.0


def _fit_warnings(img: np.ndarray, ptc: np.ndarray, H: np.ndarray) -> list[str]:
    """Non-fatal notes: H is determined but poorly."""
    out = []
    cfg = geo.analyse_configuration(img, ptc)
    if len(img) > 4 and cfg["n_general_subsets"] == 1:
        out.append(
            "only one set of 4 points is in general position; the rest add no "
            "constraint, so the fit is effectively exact and unchecked - add "
            "points off the shared line"
        )
    cond = geo.conditioning(img, ptc, H)
    if not np.isfinite(cond) or cond > MAX_H_COND:
        out.append(
            f"homography is poorly conditioned (cond {cond:.3g}); the points "
            "are nearly degenerate - spread them out"
        )
    return out


def _distortion_from(body: dict) -> dict | None:
    """The optional lens ``distortion`` of a request (``{model, k1, centre, aspect}``); a missing
    value or k1 == 0 is the plain homography."""
    d = body.get("distortion")
    if not d:
        return None
    try:
        geo.distortion_params(d)
        return geo.make_distortion(float(d["k1"]), float(d.get("aspect") or geo.DEFAULT_ASPECT),
                                   d.get("centre") or (0.5, 0.5))
    except (KeyError, TypeError, ValueError) as exc:
        raise _BadRequest(f"bad distortion: {exc}") from exc


def _json_safe(d: dict) -> dict:
    return {k: (None if isinstance(v, float) and not np.isfinite(v) else v) for k, v in d.items()}


def _aspect(image_size) -> float:
    try:
        return float(image_size[0]) / float(image_size[1])
    except (TypeError, ValueError, IndexError, ZeroDivisionError):
        return geo.DEFAULT_ASPECT


def default_distortion(clip: str, image_size=None) -> dict | None:
    """The ground's *suggested* lens for ``clip`` (division ``k1`` of its camera rig), ``None`` for an unknown ground.

    Only a suggestion: nothing applies it by itself. The editor jumps to it when the operator ticks the lens on,
    and Snap starts its search from it while the lens is off."""
    return lens.default_distortion(clip, _aspect(image_size))


def _fit(body: dict) -> dict:
    """Shared by /fit and /gt save. Raises _BadRequest on bad input.

    Correspondence ``image`` points are observed (distorted) coordinates; with a ``distortion``
    in the body they are undistorted before the DLT, and the overlay is re-distorted."""
    pitch = _pitch_from(body)
    dist = _distortion_from(body)
    img, ptc, clean = _parse_correspondences(body.get("correspondences", []), pitch)
    img_obs = img
    img = geo.undistort_points(img, dist)
    method = body.get("method", "dlt")
    size = body.get("image_size")
    size_t = (float(size[0]), float(size[1])) if size else None
    try:
        H, inliers = geo.fit_homography(
            img,
            ptc,
            method=method,
            ransac_thresh_m=float(body.get("ransac_thresh_m", 1.0)),
        )
    except ValueError as exc:
        raise _BadRequest(str(exc)) from exc
    errs = geo.reprojection_errors(H, img_obs, ptc, size_t, dist)
    warnings = _fit_warnings(img, ptc, H)
    overlay = {
        name: [r.tolist() for r in runs]
        for name, runs in geo.project_pitch_model(H, pitch, distortion=dist).items()
    }

    def _clean(a: np.ndarray | None) -> list | None:
        return None if a is None else [None if np.isnan(t) else float(t) for t in a]

    out = {
        "ok": True,
        "H": H.tolist(),
        "H_flat": H.ravel().tolist(),
        "inliers": inliers.tolist(),
        "err_m": _clean(errs["metres"]),
        "err_px": _clean(errs["px"]),
        "mean_err_m": float(np.mean(errs["metres"])),
        "warnings": warnings,
        "overlay": overlay,
        "correspondences": clean,
        "pitch_dims": [pitch.length, pitch.width],
    }
    if dist is not None:
        out["distortion"] = dist
    return out


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@router.get("/calibration/api/pitch_model")
async def pitch_model(length: float = DEFAULT_LENGTH, width: float = DEFAULT_WIDTH):
    try:
        model = PitchModel(length, width)
    except ValueError as exc:
        return _err(400, str(exc))
    d = model.to_dict()
    d["warp"] = {"px_per_m": WARP_PX_PER_M, "margin_m": WARP_MARGIN_M}
    return d


def _safe_clip(clip: str) -> bool:
    return bool(clip) and not any(c in clip for c in "/\\*?[]\0") and not clip.startswith(".")


def _find_video(clip: str):
    """Video for ``clip``, first hit wins:

    1. the active dataset's clips dir: exact stem first, else any file whose canonical clip id matches (proposal
       sidecars store canonical ids, e.g. ``arsenal_astonvilla_seg110`` for ``astonvilla_seg110.mp4``);
    2. its ``extra_clips_dirs``;
    3. only for a clip that is also a full-match segment: the clips dirs and extra dirs of every OTHER dataset in
       the registry, so a clip that lives in some dataset keeps the cv2-seek frame convention its labels were made
       with whichever dataset is active (dataset isolation is otherwise unchanged: a clip only another dataset has is
       not found);
    4. only then the full-match broadcast segment (``labeller.match_segments``), read by decoded index.
    """
    if not _safe_clip(clip):
        return None
    video = _find_video_exact(clip)
    if video is not None:
        return video
    from footy_track.calibration.prelabel.pipeline import find_video

    video = find_video(_clips_dir(), clip)
    # extra clip dirs the pre-labeller may have read from (os.pathsep-separated)
    for d in datasets.current().extra_clips_dirs:
        if video is None and d.is_dir():
            video = find_video(d, clip)
    if video is None:
        segment = find_segment_video(clip)
        if segment is not None:
            cur = datasets.current_name()
            for ds in datasets.all_datasets():
                if ds.name == cur:
                    continue
                for d in (ds.clips_dir, *ds.extra_clips_dirs):
                    if video is None and d.is_dir():
                        video = find_video(d, clip)
        video = video or segment
    return video


@router.get("/calibration/api/clips")
async def list_clips() -> dict:
    cdir = _clips_dir()
    if not cdir.exists():
        return {"clips": []}
    paths = sorted(
        p
        for p in cdir.iterdir()
        if p.suffix.lower() in _VIDEO_SUFFIXES and not p.name.startswith(".")
    )
    return {
        "clips": [
            {
                "name": p.stem,
                "canonical": canonical_clip_id(p.stem),
                "n_gt": len(_gt_frames(p.stem)),
                "n_not_broadcast": len(_not_broadcast(p.stem)),
                "n_too_few_points": len(_too_few_points(p.stem)),
            }
            for p in paths
        ]
    }


@router.get("/calibration/api/clips/{clip}/meta")
async def clip_meta(clip: str):
    video = _find_video(clip)
    if video is None:
        return _err(404, f"clip not found: {clip}")
    meta = await asyncio.to_thread(_video_meta, video)
    if meta is None:
        return _err(404, f"cannot open video for {clip}")
    return {**meta, "name": clip, "gt_frames": _gt_frames(clip),
            "not_broadcast": _not_broadcast(clip), "too_few_points": _too_few_points(clip),
            "default_distortion": default_distortion(clip, (meta["width"], meta["height"]))}


@router.get("/calibration/api/frame/{clip}/{idx}.jpg")
async def frame_jpg(clip: str, idx: int):
    video = _find_video(clip)
    if video is None:
        return _err(404, f"clip not found: {clip}")
    frame = await asyncio.to_thread(_read_frame, video, idx)
    if frame is None:
        return _err(404, f"cannot read frame {idx}")
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
    if not ok:
        return _err(500, "jpeg encode failed")
    return Response(buf.tobytes(), media_type="image/jpeg")


@router.post("/calibration/api/fit")
async def fit(body: dict):
    try:
        return _fit(body)
    except _BadRequest as exc:
        return _err(400, str(exc))


@router.post("/calibration/api/warp")
async def warp(body: dict):
    """Frame warped into a top-down pitch raster (JPEG).

    Extents: ``(length + 2*margin) x (width + 2*margin)`` metres at
    ``px_per_m``, ``+x`` right, ``+y`` up; margin and scale are reported by
    ``/pitch_model`` so the page draws its diagram over the same box.
    """
    try:
        pitch = _pitch_from(body)
        H = np.array(body["H"], dtype=np.float64)
        if H.size == 9:
            H = H.reshape(3, 3)
        if H.shape != (3, 3) or not np.all(np.isfinite(H)):
            raise _BadRequest("H must be a finite 3x3 matrix")
        clip, idx = str(body["clip"]), int(body["frame_index"])
        dist = _distortion_from(body)
        scale = float(body.get("px_per_m", WARP_PX_PER_M))
        if not 0 < scale <= MAX_WARP_PX_PER_M:
            raise _BadRequest("px_per_m out of range")
    except (KeyError, TypeError, ValueError) as exc:
        return _err(400, f"bad request: {exc}")
    except _BadRequest as exc:
        return _err(400, str(exc))
    video = _find_video(clip)
    if video is None:
        return _err(404, f"clip not found: {clip}")
    frame = await asyncio.to_thread(_read_frame, video, idx)
    if frame is None:
        return _err(404, f"cannot read frame {idx}")
    raster = await asyncio.to_thread(
        geo.warp_to_pitch, frame, H, pitch, scale, WARP_MARGIN_M, dist
    )
    ok, buf = cv2.imencode(".jpg", raster, [cv2.IMWRITE_JPEG_QUALITY, 85])
    if not ok:
        return _err(500, "jpeg encode failed")
    return Response(buf.tobytes(), media_type="image/jpeg")


# ---------------------------------------------------------------------------
# Auto points: two finders, "centre" (circle + halfway line + touchline end) and "d" (penalty area)
# ---------------------------------------------------------------------------

#: ``(dataset, video path, frame, distortion, kind, pitch dims)`` -> ``{kind, confidence,
#: correspondences}`` or ``{reason}``. Abstentions are cached too (the finders are deterministic). Small:
#: one entry is a few hundred bytes, the cost being avoided is a video seek + RANSAC / template search.
_AUTO_CACHE_MAX = 64
_AUTO_CACHE: collections.OrderedDict[tuple, dict] = collections.OrderedDict()
#: One detection at a time: it is CPU work on the machine the operator is labelling on.
_AUTO_LOCK = threading.Lock()
AUTO_KINDS = ("centre", "d")


def _auto_points(
    video: Path, idx: int, dist: dict | None, pitch: PitchModel, kind: str, key: tuple, clip: str = ""
) -> dict | None:
    """Finder verdict for one frame (cached); ``None`` when the frame cannot be read."""
    with _AUTO_LOCK:
        hit = _AUTO_CACHE.get(key)
        if hit is not None:
            _AUTO_CACHE.move_to_end(key)
            return hit
        frame = _read_frame(video, idx)
        if frame is None:
            return None
        from footy_track.calibration.centre_fit import (  # noqa: PLC0415
            rig_for_clip,
            to_norm_H,
        )
        from footy_track.calibration.pitch_finders import find_points  # noqa: PLC0415

        kw = {"rig": rig_for_clip(clip)} if kind == "centre" else {}
        det = find_points(kind, frame, dist, pitch, **kw)
        if det.result is None:
            out = {"reason": det.reason or f"no {kind} structure found"}
        else:
            r = det.result
            out = {"kind": kind, "confidence": float(r.confidence), "correspondences": r.correspondences()}
            shape = r.info.get("shape") or {}
            fitted = det.debug.get("H_px") is not None and (kind == "d" or shape.get("accepted"))
            if fitted:
                # the fitted pitch -> undistorted normalised image homography (apply ``distortion`` to draw it)
                out["H"] = to_norm_H(det.debug["H_px"], det.size).tolist()
                out["H_convention"] = "pitch_m_to_undistorted_normalised_image"
            if kind == "centre":
                if shape.get("accepted") == "intrinsic":
                    out["warning"] = (
                        "approximate: shape fitted from the square-pixel assumption alone (no touchline end, no known "
                        "camera for this ground); left/right and the 45-degree points may be ~10 px off - check them"
                    )
                out["fit"] = {
                    k: (v if not hasattr(v, "item") else v.item())
                    for k, v in shape.items()
                    if k in ("accepted", "precision", "f_over_w", "ground", "ends")
                }
        _AUTO_CACHE[key] = out
        while len(_AUTO_CACHE) > _AUTO_CACHE_MAX:
            _AUTO_CACHE.popitem(last=False)
        return out


@router.post("/calibration/api/auto_points")
async def auto_points(body: dict):
    """Place pitch points the machine can find in a frame.

    Body ``{clip, frame_index, kind, distortion?, pitch_dims?}`` with ``kind`` ``"centre"`` (centre circle x
    halfway line, plus the touchline end(s) of the halfway line and derived circle points when the touchline
    is visible) or ``"d"`` (penalty area: box, 6-yard box and D fitted as one template; falls back to the two
    box-front x D points). ``distortion`` is as in ``/fit``: the returned image points are observed
    coordinates, consistent with it. ``pitch_dims`` are the editor's pitch length x width: the touchline end
    sits at ``(0, +-width/2)`` and the penalty-area corners are placed from the length.

    Answers ``{ok: true, kind, confidence, correspondences: [{image, pitch_id, pitch_xy, confidence,
    derived}], H?, H_convention?, fit?}`` (``pitch_id`` is ``null`` for the free derived circle points,
    ``derived`` marks points synthesised from the fitted geometry rather than seen; ``H`` is the fitted
    pitch-metres -> undistorted normalised image homography every returned point was computed from, present
    when a shape / template fit was accepted; ``fit`` says which source fixed it), or ``{ok: false, reason}`` with HTTP 200 when the
    finder abstains (not an error). Nothing is stored; the page adds the pairs to its working set. Respects
    the request's dataset.
    """
    try:
        clip, idx = str(body["clip"]), int(body["frame_index"])
        kind = body.get("kind")
        if kind not in AUTO_KINDS:
            raise _BadRequest(f"kind must be one of {list(AUTO_KINDS)}")
        pitch = _pitch_from(body)
        dist = _distortion_from(body)
        if idx < 0:
            raise _BadRequest("frame_index must be >= 0")
    except (KeyError, TypeError, ValueError) as exc:
        return _err(400, f"bad request: {exc}")
    except _BadRequest as exc:
        return _err(400, str(exc))
    video = _find_video(clip)
    if video is None:
        return _err(404, f"clip not found: {clip}")
    key = (datasets.current_name(), str(video), idx, json.dumps(dist, sort_keys=True), kind, (pitch.length, pitch.width))
    res = await asyncio.to_thread(_auto_points, video, idx, dist, pitch, kind, key, clip)
    if res is None:
        return _err(404, f"cannot read frame {idx}")
    if "reason" in res:
        return {"ok": False, "reason": res["reason"]}
    return {"ok": True, **{k: v for k, v in res.items() if k != "reason"}}


# ---------------------------------------------------------------------------
# Snap to lines: refine H + k1 of the current fit against the frame's painted lines
# ---------------------------------------------------------------------------

#: ``(dataset, video path, frame)`` -> ``lens.Evidence`` (distance field of the frame's line mask; ~3 s to build)
_SNAP_CACHE_MAX = 4
_SNAP_CACHE: collections.OrderedDict[tuple, lens.Evidence] = collections.OrderedDict()
#: One snap at a time (CPU work on the machine the operator is labelling on)
_SNAP_LOCK = threading.Lock()
SNAP_MODES = ("full", "k1")


def _snap(video: Path, idx: int, key: tuple, H: np.ndarray, dist: dict | None, pitch: PitchModel,
          img: np.ndarray, ptc: np.ndarray, mode: str, aspect: float, suggested=None) -> dict | None:
    with _SNAP_LOCK:
        ev = _SNAP_CACHE.get(key)
        if ev is None:
            frame = _read_frame(video, idx)
            if frame is None:
                return None
            ev = lens.Evidence.from_frame(frame)
            _SNAP_CACHE[key] = ev
            while len(_SNAP_CACHE) > _SNAP_CACHE_MAX:
                _SNAP_CACHE.popitem(last=False)
        else:
            _SNAP_CACHE.move_to_end(key)
        r = lens.snap_best(ev, H, dist, pitch, img, ptc, mode=mode, aspect=aspect, suggested=suggested)
        edge0 = lens.edge_residual(ev, H, dist, pitch)
        edge1 = lens.edge_residual(ev, r.H, r.distortion, pitch) if r.accepted else edge0
        scale = (ev.frame_width or ev.size[0]) / ev.size[0]
        return {"r": r, "edge_before": edge0, "edge_after": edge1, "px_scale": scale}


@router.post("/calibration/api/snap")
async def snap(body: dict):
    """Snap the current fit to the painted lines ("bend" the projected lines onto the pitch markings).

    Body: the ``/fit`` body (``correspondences``, ``distortion?``, ``pitch_dims?``, ``image_size?``) plus
    ``clip``, ``frame_index`` and ``mode`` (``"full"``: refine H and k1, default; ``"k1"``: only k1, H fixed).
    Starts from the fit those points give (with that distortion), then runs a robust least squares over the
    homography (8 parameters) and ``k1`` against the frame's line mask, the clicked points acting as soft
    constraints. Accepted only if the line F1 improves and no clicked point has to move more than a few px.

    Answers ``{ok: true, accepted, reason, f1_before, f1_after, k1_before, k1_after, max_click_shift_px,
    edge_before, edge_after, distortion, correspondences}``: when ``accepted`` the ``correspondences`` carry
    the clicked points moved onto the snapped model (so H refit from them under ``distortion`` reproduces it);
    otherwise they are the input ones and ``distortion`` is the input's, with the ``reason``. Stores nothing.
    """
    try:
        clip, idx = str(body["clip"]), int(body["frame_index"])
        if idx < 0:
            raise _BadRequest("frame_index must be >= 0")
        mode = body.get("mode", "full")
        if mode not in SNAP_MODES:
            raise _BadRequest(f"mode must be one of {list(SNAP_MODES)}")
        res = _fit(body)
        pitch = _pitch_from(body)
        dist = _distortion_from(body)
        if len(res["correspondences"]) < 4:
            raise _BadRequest("need at least 4 point pairs to snap")
    except (KeyError, TypeError, ValueError) as exc:
        return _err(400, f"bad request: {exc}")
    except _BadRequest as exc:
        return _err(400, str(exc))
    video = _find_video(clip)
    if video is None:
        return _err(404, f"clip not found: {clip}")
    corr = res["correspondences"]
    img = np.array([c["image"] for c in corr], dtype=np.float64)
    ptc = np.array([c["pitch_xy"] for c in corr], dtype=np.float64)
    key = (datasets.current_name(), str(video), idx)
    H_cur = np.array(res["H"])
    suggested = None
    sug = default_distortion(clip, body.get("image_size")) if dist is None and mode == "full" else None
    if sug is not None:
        # lens off: also search from the ground's suggested k1 (H refit under it); both are judged against the fit the
        # operator has now, which is what comes back if neither is accepted
        try:
            suggested = (np.array(_fit({**body, "distortion": sug})["H"]), sug)
        except _BadRequest:
            suggested = None
    out = await asyncio.to_thread(
        _snap, video, idx, key, H_cur, dist, pitch, img, ptc, mode, _aspect(body.get("image_size")), suggested
    )
    if out is None:
        return _err(404, f"cannot read frame {idx}")
    r = out["r"]
    new_corr = corr
    if r.accepted and r.points is not None:
        new_corr = [dict(c, image=[float(p[0]), float(p[1])]) for c, p in zip(corr, r.points, strict=True)]
    shift = r.max_click_shift_px
    return {
        "ok": True, "accepted": r.accepted, "reason": r.reason, "mode": mode,
        "f1_before": r.f1_before, "f1_after": r.f1_after,
        "k1_before": r.k1_before, "k1_after": r.k1_after,
        "max_click_shift_px": None if shift is None else shift * out["px_scale"],   # px of the full frame
        "edge_before": _json_safe(out["edge_before"]), "edge_after": _json_safe(out["edge_after"]),
        "distortion": r.distortion, "correspondences": new_corr, "tried_suggested_k1": suggested is not None,
    }


@router.get("/calibration/api/gt/{clip}")
async def gt_list(clip: str) -> dict:
    doc = await asyncio.to_thread(_read_gt, clip)
    return {
        "clip": doc["clip"],
        "not_broadcast": await asyncio.to_thread(_not_broadcast, clip),
        "too_few_points": await asyncio.to_thread(_too_few_points, clip),
        "frames": [
            {
                "frame_index": int(k),
                "n_points": len(v.get("correspondences", [])),
                "mean_err_m": v.get("mean_err_m"),
                "ts": v.get("ts"),
            }
            for k, v in sorted(doc["frames"].items(), key=lambda kv: int(kv[0]))
        ],
    }


@router.get("/calibration/api/gt/{clip}/{idx}")
async def gt_get(clip: str, idx: int):
    doc = await asyncio.to_thread(_read_gt, clip)
    rec = doc["frames"].get(str(idx))
    if rec is None:
        return _err(404, "no calibration GT for this frame")
    return rec


def _gt_record(idx: int, res: dict, body: dict, origin: dict | None = None, clip: str | None = None) -> dict:
    """The GT frame record for a fit result (shared by GT save and proposal accept).

    With ``clip``, also records ``frame_index_convention`` (``decoded`` | ``cv2_seek``) and ``video`` (the resolved
    video path relative to ``footy_data``) so a reader knows which image the index meant."""
    rec = {
        "frame_index": idx,
        "correspondences": res["correspondences"],
        "H": res["H"],  # normalised image -> pitch metres
        "H_convention": "normalised_image_to_pitch_m",
        "pitch_dims": res["pitch_dims"],
        "image_size": body.get("image_size"),
        "method": body.get("method", "dlt"),
        "inliers": res["inliers"],
        "mean_err_m": res["mean_err_m"],
        "err_px": res["err_px"],
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    if res.get("distortion"):  # absent == plain homography (k1 = 0)
        rec["distortion"] = res["distortion"]
    if origin:
        rec["origin"] = origin
    video = _find_video(clip) if clip else None
    if video is not None:
        rec["frame_index_convention"] = _convention_of_video(video)
        rec["video"] = _video_ref(video)
    return rec


@router.put("/calibration/api/gt/{clip}/{idx}")
async def gt_save(clip: str, idx: int, body: dict):
    """Save (replace) the calibration GT of one frame.

    Body: ``{correspondences, pitch_dims?, method?, image_size?}``. H is
    re-fitted server-side from the correspondences, so the stored H can never
    disagree with the stored points.
    """
    if idx < 0:
        return _err(400, "frame_index must be >= 0")
    if idx in await asyncio.to_thread(_not_broadcast, clip):
        return _err(409, "this frame is marked not broadcast; clear the mark before saving GT")
    try:
        res = _fit(body)
    except _BadRequest as exc:
        return _err(400, str(exc))
    rec = _gt_record(idx, res, body, clip=clip)

    def _store() -> None:
        update_gt(clip, lambda doc: doc["frames"].update({str(idx): rec}))
        if idx in too_few_points_frames(_gt_marks_dir(), clip):
            clear_too_few_points(clip, idx)   # labelled after all: it had enough points
        hooks.emit("calibration_gt", clip=canonical_clip_id(clip), unit=idx, verdict="gt")

    await asyncio.to_thread(_store)
    return {"ok": True, "frame_index": idx, "gt_frames": _gt_frames(clip), "record": rec}


@router.delete("/calibration/api/gt/{clip}/{idx}")
async def gt_delete(clip: str, idx: int):
    def _drop() -> bool:
        if not update_gt(clip, lambda doc: doc["frames"].pop(str(idx), None) is not None):
            return False
        hooks.emit("calibration_gt", clip=canonical_clip_id(clip), unit=idx, verdict="gt_deleted")
        return True

    if not await asyncio.to_thread(_drop):
        return _err(404, "no calibration GT for this frame")
    return {"ok": True, "gt_frames": _gt_frames(clip)}
