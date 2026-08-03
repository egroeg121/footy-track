"""Ingest API: upload a match video → split_broadcast_segments → clips dir.

See ``src/footy_track/labeller/README.md`` §6 (Ingest API, LAB-5xx). ``server.py`` is the
composition root; the clips output dir is resolved through it at call time.
"""

from __future__ import annotations

import asyncio
import os
import sys
import tempfile
from pathlib import Path

from fastapi import APIRouter, File, UploadFile
from fastapi.responses import StreamingResponse

router = APIRouter()

_INGEST_UPLOADS = Path(tempfile.gettempdir()) / "footy_ingest_uploads"
_INGEST_UPLOADS.mkdir(parents=True, exist_ok=True)

# Module-level singleton so the FastAPI param default isn't a call expression (B008).
_UPLOAD_FILE = File(...)


def _clips_dir() -> Path:
    from footy_track.labeller import server  # noqa: PLC0415 — avoid import cycle

    return server._CLIPS_DIR


@router.post("/ingest/upload")
async def ingest_upload(file: UploadFile = _UPLOAD_FILE) -> dict:
    """Save uploaded video to a temp location and return the path."""
    dest = _INGEST_UPLOADS / (file.filename or "upload.mp4")
    data = await file.read()
    dest.write_bytes(data)
    return {"path": str(dest), "name": dest.name, "size": len(data)}


@router.get("/ingest/run")
async def ingest_run(
    path: str, sample: int = 5, merge_gap_s: float = 0.5, min_seg_s: float = 2.0
) -> StreamingResponse:
    """Stream split_broadcast_segments output as SSE."""
    video_path = Path(path).expanduser()

    async def event_stream():
        if not video_path.exists():
            yield f"data: ERROR: file not found: {video_path}\n\n"
            yield "data: [DONE]\n\n"
            return

        cmd = [
            sys.executable,
            "-m",
            "footy_track.scripts.split_broadcast_segments",
            str(video_path),
            "--outdir",
            str(_clips_dir()),
            "--sample",
            str(sample),
            "--merge-gap-s",
            str(merge_gap_s),
            "--min-seg-s",
            str(min_seg_s),
        ]
        yield f"data: Running: {' '.join(cmd)}\n\n"

        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
        )
        assert proc.stdout is not None
        async for raw in proc.stdout:
            line = raw.decode(errors="replace").rstrip()
            if line:
                yield f"data: {line}\n\n"

        rc = await proc.wait()
        yield f"data: [EXIT {rc}]\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
