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
import json
import os
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np
from fastapi import APIRouter
from fastapi.responses import JSONResponse, Response

from footy_track.calibration import geometry as geo
from footy_track.calibration.pitch_model import (
    DEFAULT_LENGTH,
    DEFAULT_WIDTH,
    PitchModel,
)
from footy_track.calibration.video import read_frame
from footy_track.clip_id import canonical_clip_id
from footy_track.labeller.review import _clips_dir, _find_video, _gt_marks_dir

router = APIRouter()

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


def _read_frame(video: Path, idx: int) -> np.ndarray | None:
    # cv2 with an ffmpeg fallback for clips OpenCV decodes as black.
    return read_frame(video, idx)


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


def _gt_frames(clip: str) -> list[int]:
    try:
        return sorted(int(k) for k in _read_gt(clip)["frames"])
    except (OSError, ValueError, KeyError):
        return []


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


def _fit(body: dict) -> dict:
    """Shared by /fit and /gt save. Raises _BadRequest on bad input."""
    pitch = _pitch_from(body)
    img, ptc, clean = _parse_correspondences(body.get("correspondences", []), pitch)
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
    errs = geo.reprojection_errors(H, img, ptc, size_t)
    warnings = _fit_warnings(img, ptc, H)
    overlay = {
        name: [r.tolist() for r in runs]
        for name, runs in geo.project_pitch_model(H, pitch).items()
    }

    def _clean(a: np.ndarray | None) -> list | None:
        return None if a is None else [None if np.isnan(t) else float(t) for t in a]

    return {
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


@router.get("/calibration/api/clips")
async def list_clips() -> dict:
    cdir = _clips_dir()
    if not cdir.exists():
        return {"clips": []}
    paths = sorted(p for p in cdir.iterdir() if p.suffix.lower() in _VIDEO_SUFFIXES)
    return {
        "clips": [
            {"name": p.stem, "n_gt": len(_gt_frames(p.stem))} for p in paths
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
    return {**meta, "name": clip, "gt_frames": _gt_frames(clip)}


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
        geo.warp_to_pitch, frame, H, pitch, scale, WARP_MARGIN_M
    )
    ok, buf = cv2.imencode(".jpg", raster, [cv2.IMWRITE_JPEG_QUALITY, 85])
    if not ok:
        return _err(500, "jpeg encode failed")
    return Response(buf.tobytes(), media_type="image/jpeg")


@router.get("/calibration/api/gt/{clip}")
async def gt_list(clip: str) -> dict:
    doc = await asyncio.to_thread(_read_gt, clip)
    return {
        "clip": doc["clip"],
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


def _gt_record(idx: int, res: dict, body: dict, origin: dict | None = None) -> dict:
    """The GT frame record for a fit result (shared by GT save and proposal accept)."""
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
    if origin:
        rec["origin"] = origin
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
    try:
        res = _fit(body)
    except _BadRequest as exc:
        return _err(400, str(exc))
    rec = _gt_record(idx, res, body)

    def _store() -> None:
        doc = _read_gt(clip)
        doc["frames"][str(idx)] = rec
        _write_gt(clip, doc)

    await asyncio.to_thread(_store)
    return {"ok": True, "frame_index": idx, "gt_frames": _gt_frames(clip), "record": rec}


@router.delete("/calibration/api/gt/{clip}/{idx}")
async def gt_delete(clip: str, idx: int):
    def _drop() -> bool:
        doc = _read_gt(clip)
        if doc["frames"].pop(str(idx), None) is None:
            return False
        _write_gt(clip, doc)
        return True

    if not await asyncio.to_thread(_drop):
        return _err(404, "no calibration GT for this frame")
    return {"ok": True, "gt_frames": _gt_frames(clip)}
