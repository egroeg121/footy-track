"""``/calibration/match``: spot-check a whole match's machine calibration (PTZ V3 match files).

Reads the per-match files written by ``scripts/calibration_match_combine.py``
(``footy_track.calibration.match_calibration``): per frame H, k1, calibrated / accepted flags, line-F1. The page shows a
status strip of the whole match (accepted / calibrated but not accepted / uncalibrated), a viewer with the projected
pitch lines drawn over any frame (division-model k1 distortion, same projection as the editor) and a grid of sampled
frames.

Light on CPU by design: the base frame is a small JPEG (default 960 px, quality 70) decoded by at most
``MAX_DECODERS`` threads at once and kept in an LRU (browsers also cache it: the URL carries the file version);
the pitch lines are drawn client-side as SVG from the overlay JSON (a few KB per frame), so toggling them or
stepping through frames never re-renders anything on the server. Nothing is written.

Frames are addressed by ``(segment, local frame)``: the index into the segment's mp4 under
``<data>/<match>/split_video_broadcast_frames/`` (the same decoded-frame index the calibration editor uses for the clip).
"""

from __future__ import annotations

import asyncio
import collections
import json
import threading
import time
from pathlib import Path

import cv2
import numpy as np
from fastapi import APIRouter
from fastapi.responses import HTMLResponse, JSONResponse, Response

from footy_track.calibration import geometry as geo
from footy_track.calibration import match_calibration as mcal
from footy_track.calibration.pitch_model import PitchModel
from footy_track.calibration.video import read_frame_exact
from footy_track.clip_id import canonical_clip_id
from footy_track.labeller import matches_status

router = APIRouter()
_PAGE = Path(__file__).parent / "web" / "calibration_match.html"

#: status codes of the strip's run-length encoding
CODE_UNCAL, CODE_CAL, CODE_ACC = 0, 1, 2
CODE_NAME = {CODE_UNCAL: "uncalibrated", CODE_CAL: "calibrated", CODE_ACC: "accepted"}
MAX_DECODERS = 2
FRAME_CACHE_MAX = 300
GRID_MAX = 60
OVERLAY_STEP_M = 1.0
_PITCH = PitchModel()

_decode_sem = threading.Semaphore(MAX_DECODERS)
_JPEG_CACHE: collections.OrderedDict[tuple, bytes] = collections.OrderedDict()
_JPEG_LOCK = threading.Lock()
_MC_CACHE: dict[tuple[str, str], tuple[float, mcal.MatchCalibration]] = {}
_MC_LOCK = threading.Lock()
_MC_CACHE_MAX = 2


def _err(status: int, msg: str) -> JSONResponse:
    return JSONResponse({"ok": False, "error": msg}, status_code=status)


# ---------------------------------------------------------------------------
# Access (patched in tests)
# ---------------------------------------------------------------------------


def _read_frame(video: Path, idx: int) -> np.ndarray | None:
    # ``local`` numbers frames as a sequential decode does; a CAP_PROP_POS_FRAMES seek is tens to ~170 frames off
    # on the long-GOP / variable-frame-rate segments (Bournemouth seg087), so seek by pts instead.
    return read_frame_exact(video, idx)


def _video_path(match: str, file: str) -> Path:
    return matches_status.data_root() / match / "split_video_broadcast_frames" / file


_EDITOR_TTL_S = 30.0
_EDITOR_CACHE: dict[tuple[str, str], tuple[float, str | None]] = {}


def _editor_mode(clip: str) -> str | None:
    """Frame convention the calibration editor would use for ``clip``: ``decoded`` (same index as this page),
    ``cv2_seek`` (it opens, but numbers frames by the cv2 seek: a different image on long-GOP / VFR segments) or
    ``None`` (no video).

    Cached for ``_EDITOR_TTL_S`` per (dataset, clip): the lookup scans the clip directories and reads the GT sidecar."""
    from footy_track.labeller import (  # noqa: PLC0415 (avoid an import cycle at module load)
        calibration,
        datasets,
    )

    key = (datasets.current_name(), clip)
    now = time.monotonic()
    hit = _EDITOR_CACHE.get(key)
    if hit and now - hit[0] < _EDITOR_TTL_S:
        return hit[1]
    try:
        mode = calibration.frame_convention(clip)
    except Exception:  # noqa: BLE001 - a missing dir must not break the page
        mode = None
    _EDITOR_CACHE[key] = (now, mode)
    return mode


def _editor_has_clip(clip: str) -> bool:
    """Whether the editor link for ``clip`` lands on the same image as this page: it can open the clip AND numbers
    its frames by decoded index. A clip it can only open with the cv2 seek (dataset clip dirs, or a segment whose
    GT predates the convention marker) is not linked."""
    return _editor_mode(clip) == "decoded"


def _load(match: str, version: str) -> mcal.MatchCalibration | None:
    mf = mcal.calibration_root() / version / f"{match}.manifest.json"
    try:
        mtime = mf.stat().st_mtime
    except OSError:
        return None
    key = (match, version)
    with _MC_LOCK:
        hit = _MC_CACHE.get(key)
        if hit and hit[0] == mtime:
            return hit[1]
    try:
        mc = mcal.load_match_calibration(match, version=version)
    except (FileNotFoundError, ValueError):
        return None
    with _MC_LOCK:
        _MC_CACHE[key] = (mtime, mc)
        while len(_MC_CACHE) > _MC_CACHE_MAX:
            _MC_CACHE.pop(next(iter(_MC_CACHE)))
    return mc


def _valid_name(s: str) -> bool:
    return bool(s) and "/" not in s and "\\" not in s and not s.startswith(".")


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------


def status_codes(mc: mcal.MatchCalibration) -> np.ndarray:
    """Per row: 0 uncalibrated, 1 calibrated but not accepted, 2 accepted (memoised on ``mc``)."""
    codes = getattr(mc, "_status_codes", None)
    if codes is None:
        codes = np.where(mc.accepted, CODE_ACC, np.where(mc.calibrated, CODE_CAL, CODE_UNCAL)).astype(np.int8)
        mc._status_codes = codes
    return codes


def rle(codes: np.ndarray) -> list[list[int]]:
    """Run-length encoding ``[[code, length], ...]``."""
    if len(codes) == 0:
        return []
    change = np.nonzero(np.diff(codes))[0] + 1
    starts = np.concatenate([[0], change])
    ends = np.concatenate([change, [len(codes)]])
    return [[int(codes[a]), int(b - a)] for a, b in zip(starts, ends, strict=False)]


def overlay_runs(H: np.ndarray, k1: float, step_m: float = OVERLAY_STEP_M) -> dict[str, list[list[list[float]]]]:
    """Pitch lines as ``name -> runs`` of normalised OBSERVED image points (k1 distortion applied), 4 decimals."""
    dist = geo.make_distortion(float(k1), geo.DEFAULT_ASPECT) if k1 and np.isfinite(k1) else None
    out = {}
    for name, runs in geo.project_pitch_model(np.asarray(H, float), _PITCH, step_m, distortion=dist).items():
        out[name] = [np.round(r, 4).tolist() for r in runs]
    return out


def _row_of(mc: mcal.MatchCalibration, segment: int, local: int) -> int | None:
    sl = mc.segment_rows(segment)
    if sl.start == sl.stop:
        return None
    k = int(np.searchsorted(mc.local_frame[sl], local))
    if k >= sl.stop - sl.start or mc.local_frame[sl.start + k] != local:
        return None
    return sl.start + k


def _frame_record(mc: mcal.MatchCalibration, row: int, with_overlay: bool = True) -> dict:
    seg, local = int(mc.segment[row]), int(mc.local_frame[row])
    code = CODE_ACC if mc.accepted[row] else CODE_CAL if mc.calibrated[row] else CODE_UNCAL
    ev = float(mc.evidence_f1[row])
    rec = {
        "segment": seg,
        "local": local,
        "frame": int(mc.frame[row]),
        "status": CODE_NAME[code],
        "evidence_f1": None if not np.isfinite(ev) else round(ev, 3),
        "source": str(mc.source[row]),
        "k1": None if not np.isfinite(mc.k1[row]) else float(mc.k1[row]),
        "img": f"/calibration/match/api/frame/{mc.match}/{seg}/{local}.jpg",
    }
    if np.isfinite(mc.logf[row]):
        rec["pan_deg"] = round(float(np.degrees(mc.pan[row])), 2)
        rec["tilt_deg"] = round(float(np.degrees(mc.tilt[row])), 2)
        rec["f_px"] = round(float(np.exp(mc.logf[row])), 1)
    if with_overlay and mc.calibrated[row]:
        rec["overlay"] = overlay_runs(mc.H[row], float(mc.k1[row]) if np.isfinite(mc.k1[row]) else 0.0)
    else:
        rec["overlay"] = {}
    return rec


def _segment_meta(mc: mcal.MatchCalibration) -> dict[int, dict]:
    return {int(s["segment"]): s for s in mc.manifest.get("segments", [])}


def sample_rows(mc: mcal.MatchCalibration, n: int, mode: str, seed: int, segment: int | None) -> list[int]:
    """Row indices for the grid. ``even``: evenly spaced; ``random``; ``uncertain``: random calibrated-not-accepted
    or uncalibrated frames; ``accepted``: random accepted frames. ``segment`` restricts the pool."""
    codes = status_codes(mc)
    pool = np.arange(mc.n)
    if segment is not None:
        pool = pool[mc.segment == segment]
    if mode == "uncertain":
        pool = pool[codes[pool] != CODE_ACC]
    elif mode == "accepted":
        pool = pool[codes[pool] == CODE_ACC]
    if len(pool) == 0:
        return []
    n = min(n, len(pool))
    if mode == "even":
        return [int(pool[int(i)]) for i in np.linspace(0, len(pool) - 1, n)]
    rng = np.random.default_rng(seed)
    pick = np.sort(rng.choice(len(pool), size=n, replace=False))
    return [int(pool[i]) for i in pick]


def next_uncertain(mc: mcal.MatchCalibration, row: int, direction: int) -> int | None:
    """Start of the next run of not-accepted frames after (``direction`` 1) / before (-1) ``row``.

    If ``row`` is itself in such a run the whole run is skipped, so repeated jumps walk through the match."""
    bad = status_codes(mc) != CODE_ACC
    if direction >= 0:
        i = row
        while i < mc.n and bad[i]:
            i += 1
        nxt = np.nonzero(bad[i:])[0]
        return None if len(nxt) == 0 else int(i + nxt[0])
    i = row
    while i >= 0 and bad[i]:
        i -= 1
    prv = np.nonzero(bad[: i + 1])[0]
    if len(prv) == 0:
        return None
    j = int(prv[-1])  # end of the previous bad run: return its start
    while j > 0 and bad[j - 1]:
        j -= 1
    return j


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@router.get("/calibration/match", response_class=HTMLResponse)
async def match_page() -> HTMLResponse:
    return HTMLResponse(_PAGE.read_text(), headers={"Cache-Control": "no-store, max-age=0"})


@router.get("/calibration/match/api/matches")
def list_matches(version: str = mcal.DEFAULT_VERSION):
    if not _valid_name(version):
        return _err(400, "bad version")
    d = mcal.calibration_root() / version
    out = []
    for mf in sorted(d.glob("*.manifest.json")) if d.is_dir() else []:
        try:
            m = json.loads(mf.read_text())
        except (OSError, ValueError):
            continue
        out.append({"match": m.get("match"), "version_id": m.get("version_id"), **(m.get("totals") or {})})
    return {"version": version, "matches": out}


@router.get("/calibration/match/api/summary")
def summary(match: str, version: str = mcal.DEFAULT_VERSION):
    if not (_valid_name(match) and _valid_name(version)):
        return _err(400, "bad match or version")
    mc = _load(match, version)
    if mc is None:
        return _err(404, f"no calibration file for {match} ({version}); run scripts/calibration_match_combine.py")
    codes = status_codes(mc)
    meta = _segment_meta(mc)
    editor_cache: dict[str, bool] = {}
    segs = []
    for seg in sorted(meta):
        info = meta[seg]
        sl = mc.segment_rows(seg)
        clip = canonical_clip_id(Path(info.get("file") or f"{match}_seg{seg:03d}").stem)
        if clip not in editor_cache:
            editor_cache[clip] = _editor_has_clip(clip)
        segs.append({
            "segment": seg,
            "file": info.get("file"),
            "clip": clip,
            "editor_ok": editor_cache[clip],
            "editor_mode": _editor_mode(clip),
            "start_frame": info.get("start_frame"),
            "row0": sl.start,
            "n": sl.stop - sl.start,
            "first_local": int(mc.local_frame[sl.start]) if sl.stop > sl.start else None,
            "source": info.get("source"),
            "accepted_frac": info.get("accepted_frac"),
            "calibrated_frac": info.get("calibrated_frac"),
            "main_accepted_frac": info.get("main_accepted_frac"),
            "rle": rle(codes[sl]),
        })
    m = mc.manifest
    return {
        "match": match, "version": version, "version_id": mc.version_id, "totals": m.get("totals"),
        "code_shas": m.get("code_shas"), "created_at": m.get("created_at"), "n_rows": mc.n, "segments": segs,
    }


@router.get("/calibration/match/api/frame_info")
def frame_info(match: str, segment: int, local: int, version: str = mcal.DEFAULT_VERSION):
    """Status + overlay of one frame. ``segment`` / ``local`` as in the summary."""
    if not (_valid_name(match) and _valid_name(version)):
        return _err(400, "bad match or version")
    mc = _load(match, version)
    if mc is None:
        return _err(404, f"no calibration file for {match}")
    row = _row_of(mc, segment, local)
    if row is None:
        return _err(404, f"no frame {local} in segment {segment}")
    rec = _frame_record(mc, row)
    clip = canonical_clip_id(Path(_segment_meta(mc).get(segment, {}).get("file") or f"{match}_seg{segment:03d}").stem)
    rec["clip"] = clip
    rec["editor_ok"] = _editor_has_clip(clip)
    rec["editor_mode"] = _editor_mode(clip)
    rec["editor_url"] = f"/calibration?clip={clip}&frame={local}"
    rec["row"] = row
    return rec


@router.get("/calibration/match/api/next")
def next_frame(match: str, segment: int, local: int, dir: int = 1, version: str = mcal.DEFAULT_VERSION):
    """The next (``dir`` 1) / previous (-1) 'uncertain' frame: the start of the next run of frames that are not
    accepted (calibrated but unconfirmed, or uncalibrated)."""
    mc = _load(match, version) if _valid_name(match) and _valid_name(version) else None
    if mc is None:
        return _err(404, f"no calibration file for {match}")
    row = _row_of(mc, segment, local)
    if row is None:
        return _err(404, f"no frame {local} in segment {segment}")
    nxt = next_uncertain(mc, row, 1 if dir >= 0 else -1)
    if nxt is None:
        return {"found": False}
    return {"found": True, "segment": int(mc.segment[nxt]), "local": int(mc.local_frame[nxt]),
            "frame": int(mc.frame[nxt]), "status": CODE_NAME[int(status_codes(mc)[nxt])]}


@router.get("/calibration/match/api/grid")
def grid(match: str, n: int = 24, mode: str = "even", seed: int = 0, segment: int | None = None,
         version: str = mcal.DEFAULT_VERSION):
    """``n`` sampled frames with their overlays, for fast eyeballing."""
    if not (_valid_name(match) and _valid_name(version)):
        return _err(400, "bad match or version")
    if mode not in ("even", "random", "uncertain", "accepted"):
        return _err(400, "mode must be even | random | uncertain | accepted")
    mc = _load(match, version)
    if mc is None:
        return _err(404, f"no calibration file for {match}")
    rows = sample_rows(mc, max(1, min(int(n), GRID_MAX)), mode, int(seed), segment)
    return {"match": match, "mode": mode, "seed": seed, "frames": [_frame_record(mc, r) for r in rows]}


def _encode_frame(video: Path, local: int, width: int, quality: int) -> bytes | None:
    with _decode_sem:
        frame = _read_frame(video, local)
    if frame is None:
        return None
    h, w = frame.shape[:2]
    if width and w > width:
        frame = cv2.resize(frame, (width, max(1, round(h * width / w))), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return buf.tobytes() if ok else None


@router.get("/calibration/match/api/frame/{match}/{segment}/{local}.jpg")
async def frame_jpg(match: str, segment: int, local: int, w: int = 960, q: int = 70,
                    version: str = mcal.DEFAULT_VERSION):
    """Small JPEG of a segment frame (no overlay: the page draws it). Long browser cache: the frame never changes."""
    if not (_valid_name(match) and _valid_name(version)):
        return _err(400, "bad match or version")
    mc = _load(match, version)
    if mc is None:
        return _err(404, f"no calibration file for {match}")
    info = _segment_meta(mc).get(segment)
    if info is None or not info.get("file") or _row_of(mc, segment, local) is None:
        return _err(404, f"no frame {local} in segment {segment}")
    w = max(160, min(int(w), 1920))
    q = max(30, min(int(q), 95))
    key = (match, segment, local, w, q)
    with _JPEG_LOCK:
        data = _JPEG_CACHE.get(key)
        if data is not None:
            _JPEG_CACHE.move_to_end(key)
    if data is None:
        video = _video_path(match, info["file"])
        if not video.is_file():
            return _err(404, f"video not found: {video.name}")
        data = await asyncio.to_thread(_encode_frame, video, local, w, q)
        if data is None:
            return _err(404, f"cannot read frame {local}")
        with _JPEG_LOCK:
            _JPEG_CACHE[key] = data
            while len(_JPEG_CACHE) > FRAME_CACHE_MAX:
                _JPEG_CACHE.popitem(last=False)
    return Response(data, media_type="image/jpeg", headers={"Cache-Control": "private, max-age=86400"})
