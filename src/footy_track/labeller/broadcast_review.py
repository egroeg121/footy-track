"""Broadcast-frame review: yes/no on frames the broadcast classifier was unsure about.

The per-match Prefect flow ``broadcast-review`` (jobs repo) runs the frame classifier over a
full match, picks the frames nearest the decision boundary plus a small random sample, writes
them as JPEGs to ``<dataset clips_dir>/<match>/f<frame>.jpg`` and submits a ``broadcast_review``
labelling job (item = ``{clip: <match key>, frame, ts_s, p_yes, pred, reason}``).

Verdicts live in ``<dataset marks_dir>/broadcast_review/<match>.json``::

    {"frames": {"12345": {"verdict": "yes"|"no"|"unsure", "ts": "...", "p_yes": 0.43, "pred": "No",
                          "reason": "edge"}}}

``yes`` = a main-camera broadcast view (what the classifier calls ``Yes``), ``no`` = anything
else (replay, close-up, crowd, graphic ...), ``unsure`` = skip (counts as answered, excluded
from training export). ``scripts/export_broadcast_review.py`` turns these into the
``train/Yes`` ``train/No`` folder layout ``scripts/train_classifier.py --local-dataset`` reads.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import re
import tempfile
import threading
from datetime import UTC, datetime
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse

from footy_track.labeller import datasets, jobs

VERDICTS = ("yes", "no", "unsure")
_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,200}$")
_LOCK = threading.RLock()
_WEB = Path(__file__).parent / "web"

router = APIRouter()


def check_match(match: str) -> str:
    if not _NAME.fullmatch(match or "") or ".." in match:
        raise ValueError(f"bad match key {match!r}")
    return match


def frame_path(ds: datasets.Dataset, match: str, frame: int) -> Path:
    return ds.clips_dir / check_match(match) / f"f{int(frame)}.jpg"


def store_path(ds: datasets.Dataset, match: str) -> Path:
    return ds.marks_dir / "broadcast_review" / f"{check_match(match)}.json"


def read_store(ds: datasets.Dataset, match: str) -> dict[int, dict]:
    """``{frame: {verdict, ts, ...}}`` ({} when nothing is labelled yet)."""
    try:
        doc = json.loads(store_path(ds, match).read_text())
    except (OSError, ValueError):
        return {}
    out: dict[int, dict] = {}
    for k, v in ((doc or {}).get("frames") or {}).items():
        if isinstance(v, dict) and str(k).isdigit() and v.get("verdict") in VERDICTS:
            out[int(k)] = v
    return out


def _write_atomic(path: Path, doc: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(doc, f, indent=1)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        os.chmod(tmp, 0o664)
        os.replace(tmp, path)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp)
        raise


def set_verdict(ds: datasets.Dataset, match: str, frame: int, verdict: str | None, extra: dict | None = None) -> dict:
    """Record (or, with ``verdict=None``, clear) the verdict for one frame."""
    if verdict is not None and verdict not in VERDICTS:
        raise ValueError(f"verdict must be one of {list(VERDICTS)}")
    path = store_path(ds, match)
    with _LOCK:
        try:
            doc = json.loads(path.read_text())
        except (OSError, ValueError):
            doc = {}
        frames = doc.setdefault("frames", {})
        if verdict is None:
            frames.pop(str(int(frame)), None)
        else:
            frames[str(int(frame))] = {
                **{k: v for k, v in (extra or {}).items() if k in ("p_yes", "pred", "reason", "ts_s")},
                "verdict": verdict,
                "ts": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
            }
        _write_atomic(path, doc)
    return frames.get(str(int(frame))) or {}


# ---------------------------------------------------------------------------
# Routes (the dataset comes from ?dataset= via the middleware)
# ---------------------------------------------------------------------------


@router.get("/broadcast_review", response_class=HTMLResponse)
async def page() -> HTMLResponse:
    return HTMLResponse((_WEB / "broadcast_review.html").read_text(), headers={"Cache-Control": "no-store, max-age=0"})


@router.get("/broadcast_review/frame/{match}/{frame}.jpg")
async def frame_jpg(match: str, frame: int):
    try:
        p = frame_path(datasets.current(), match, frame)
    except ValueError:
        raise HTTPException(404) from None
    if not p.is_file():
        raise HTTPException(404)
    return FileResponse(p, media_type="image/jpeg", headers={"Cache-Control": "max-age=3600"})


@router.get("/broadcast_review/api/job/{ref:path}")
async def job_view(ref: str):
    """The job's frames in order with ``done``/``verdict`` (+ the model's call, which the page hides until answered)."""
    def work():
        job = jobs.resolve(ref)
        if job["kind"] != "broadcast_review":
            raise jobs.JobError(422, f"job kind {job['kind']!r} is not broadcast_review")
        ds = datasets.get(job["dataset"])
        stores: dict[str, dict] = {}
        rows = []
        for i, it in enumerate(job["items"]):
            st = stores.setdefault(it["clip"], read_store(ds, it["clip"]))
            v = st.get(it["frame"])
            rows.append({**it, "item": i, "verdict": v["verdict"] if v else None, "done": bool(v)})
        return {"id": job["id"], "title": job["title"], "dataset": job["dataset"], "status": job["status"],
                "total": len(rows), "done": sum(r["done"] for r in rows), "frames": rows}  # fmt: skip

    try:
        return await asyncio.to_thread(work)
    except jobs.JobError as e:
        return JSONResponse({"error": e.detail}, status_code=e.status)


@router.put("/broadcast_review/api/verdict/{match}/{frame}")
async def put_verdict(match: str, frame: int, body: dict):
    try:
        row = await asyncio.to_thread(set_verdict, datasets.current(), match, frame, body.get("verdict"), body)
    except ValueError as e:
        raise HTTPException(422, str(e)) from None
    return {"ok": True, "row": row}


@router.delete("/broadcast_review/api/verdict/{match}/{frame}")
async def delete_verdict(match: str, frame: int):
    try:
        await asyncio.to_thread(set_verdict, datasets.current(), match, frame, None)
    except ValueError as e:
        raise HTTPException(422, str(e)) from None
    return {"ok": True}
