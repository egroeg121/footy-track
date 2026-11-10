"""FastAPI app for the inference web UI (see docs/inference-web.md).

A separate small app rather than a router on the labeller: it is meant to run
alone on the GPU machine, and importing the labeller pulls in its whole stack
(SAM/VitTrack labelling, the feature store, ...) that inference does not need.
``create_router`` is still a plain router, so it can be mounted into another
FastAPI app later.

Configuration is environment variables with CLI flags that override them:

``FOOTY_INFERENCE_JOBS_DIR``        job storage, default ``runs/web``
``FOOTY_INFERENCE_WEIGHTS``         detector checkpoint (default: the repo's current best)
``FOOTY_INFERENCE_DEVICE``          cuda / mps / cpu (default: auto)
``FOOTY_INFERENCE_MAX_UPLOAD_MB``   upload size limit, default 8192
``FOOTY_INFERENCE_TOKEN``           optional shared token for the API and downloads
``FOOTY_INFERENCE_DEMO_CLIP``       optional sample video behind the "Try a demo clip" button
``FOOTY_INFERENCE_FAKE``            ``1`` = DEV ONLY fake detector and fixed-length cutter
"""

from __future__ import annotations

import argparse
import hmac
import os
import re
import shutil
from pathlib import Path

from fastapi import APIRouter, Depends, FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

from footy_track.inference.jobs import (
    ALLOWED_EXTENSIONS,
    BadVideo,
    JobBusy,
    JobError,
    JobNotFound,
    JobRunner,
    fake_detector_factory,
    make_chunk_segmenter,
    make_real_detector_factory,
    probe_video,
)

WEB_DIR = Path(__file__).parent
DEFAULT_MAX_UPLOAD_MB = 8192
TOKEN_COOKIE = "footy_token"
_CHUNK = 1024 * 1024


def _token_ok(request: Request, token: str | None) -> bool:
    if not token:
        return True
    auth = request.headers.get("authorization", "")
    given = (
        auth[7:]
        if auth.lower().startswith("bearer ")
        else request.headers.get("x-footy-token")
        or request.query_params.get("token")
        or request.cookies.get(TOKEN_COOKIE)
        or ""
    )
    return hmac.compare_digest(given.encode(), token.encode())


def _safe_ext(name: str) -> str:
    ext = Path(name).suffix.lower()
    return ext if ext in ALLOWED_EXTENSIONS else ".mp4"


def _clean_name(name: str) -> str:
    """Display name only: strip any directory part and control characters."""
    base = re.split(r"[\\/]", name)[-1]
    return re.sub(r"[\x00-\x1f\x7f]", "", base).strip()[:120] or "video"


class _Demo:
    """The optional demo clip. Only its display name and duration leave the server."""

    def __init__(self, path: Path | None) -> None:
        self.path = path
        self._info: dict | None = None

    def available(self) -> bool:
        return self.path is not None and self.path.is_file()

    def config(self) -> dict:
        if not self.available():
            return {"demo": False}
        if self._info is None:  # probe once
            try:
                duration = probe_video(self.path)["duration_s"]
            except BadVideo:
                duration = None
            self._info = {
                "demo_name": _clean_name(self.path.name),
                "demo_duration_s": duration,
            }
        return {"demo": True, **self._info}

    def start(self, runner: JobRunner, options: dict) -> dict:
        """Copy the clip into a new job and queue it, like an upload would."""
        job_id, dest = runner.allocate(
            _clean_name(self.path.name), _safe_ext(self.path.name), options
        )
        try:
            shutil.copyfile(self.path, dest)
            return runner.submit(job_id)
        except BadVideo as exc:
            runner.abort_upload(job_id)
            raise HTTPException(400, str(exc)) from exc
        except BaseException:
            runner.abort_upload(job_id)
            raise


def create_router(  # noqa: PLR0915 - one flat list of routes
    runner: JobRunner,
    *,
    max_upload_bytes: int,
    token: str | None = None,
    fake: bool = False,
    demo_clip: Path | None = None,
) -> APIRouter:
    router = APIRouter(prefix="/infer")

    def require_token(request: Request) -> None:
        if not _token_ok(request, token):
            raise HTTPException(401, "A valid token is required.")

    auth = [Depends(require_token)]
    demo = _Demo(demo_clip)

    def _page(name: str) -> HTMLResponse:
        return HTMLResponse((WEB_DIR / name).read_text())

    @router.get("", response_class=HTMLResponse)
    async def upload_page() -> HTMLResponse:
        return _page("upload.html")

    @router.get("/jobs", response_class=HTMLResponse)
    async def jobs_page() -> HTMLResponse:
        return _page("jobs.html")

    @router.get("/jobs/{job_id}", response_class=HTMLResponse)
    async def job_page(job_id: str) -> HTMLResponse:
        return _page("job.html")

    @router.get("/api/config", dependencies=auth)
    async def config() -> dict:
        return {
            "max_upload_mb": max_upload_bytes // (1024 * 1024),
            "extensions": list(ALLOWED_EXTENSIONS),
            "fake": fake,
            **demo.config(),
        }

    @router.post("/api/jobs", dependencies=auth, status_code=201)
    async def create_job(
        request: Request,
        name: str = "video.mp4",
        max_frames: int | None = Query(None, ge=1, le=10_000_000),
        stride: int = Query(1, ge=1, le=1000),
    ) -> dict:
        """Upload a video as the raw request body (``curl --data-binary @file.mp4 URL``)."""
        declared = request.headers.get("content-length")
        if declared and declared.isdigit() and int(declared) > max_upload_bytes:
            raise HTTPException(
                413, f"File too large (limit {max_upload_bytes >> 20} MB)."
            )
        options = {"max_frames": max_frames, "stride": stride}
        job_id, dest = runner.allocate(_clean_name(name), _safe_ext(name), options)
        size = 0
        try:
            with dest.open("wb") as f:
                async for chunk in request.stream():
                    size += len(chunk)
                    if size > max_upload_bytes:
                        raise HTTPException(
                            413, f"File too large (limit {max_upload_bytes >> 20} MB)."
                        )
                    f.write(chunk)
            if size == 0:
                raise BadVideo("The upload was empty.")
            return runner.submit(job_id)
        except BadVideo as exc:
            runner.abort_upload(job_id)
            raise HTTPException(400, str(exc)) from exc
        except BaseException:
            runner.abort_upload(job_id)
            raise

    @router.post("/api/demo", dependencies=auth, status_code=201)
    async def create_demo_job(
        max_frames: int | None = Query(None, ge=1, le=10_000_000),
        stride: int = Query(1, ge=1, le=1000),
    ) -> dict:
        """Start a normal job from the configured demo clip (no upload)."""
        if not demo.available():
            raise HTTPException(404, "No demo clip is available on this server.")
        return demo.start(runner, {"max_frames": max_frames, "stride": stride})

    @router.get("/api/jobs", dependencies=auth)
    async def list_jobs() -> list[dict]:
        return runner.list()

    def _call(fn, *args):
        try:
            return fn(*args)
        except JobNotFound as exc:
            raise HTTPException(404, "Not found.") from exc
        except JobBusy as exc:
            raise HTTPException(409, str(exc)) from exc
        except JobError as exc:
            raise HTTPException(400, str(exc)) from exc

    @router.get("/api/jobs/{job_id}", dependencies=auth)
    async def get_job(job_id: str) -> dict:
        return _call(runner.get, job_id)

    @router.post("/api/jobs/{job_id}/cancel", dependencies=auth)
    async def cancel_job(job_id: str) -> dict:
        return _call(runner.cancel, job_id)

    @router.delete("/api/jobs/{job_id}", dependencies=auth, status_code=204)
    async def delete_job(job_id: str) -> None:
        _call(runner.delete, job_id)

    @router.get("/api/jobs/{job_id}/clips/{clip_id}/{name}", dependencies=auth)
    async def clip_file(job_id: str, clip_id: str, name: str) -> FileResponse:
        path = _call(runner.clip_file, job_id, clip_id, name)
        media = {".mp4": "video/mp4", ".jsonl": "application/x-ndjson"}.get(
            path.suffix, "application/json"
        )
        return FileResponse(
            path,
            media_type=media,
            filename=f"{clip_id}_{name}"
            if name.endswith((".jsonl", ".json"))
            else None,
        )

    return router


def create_app(
    runner: JobRunner,
    *,
    max_upload_bytes: int = DEFAULT_MAX_UPLOAD_MB * 1024 * 1024,
    token: str | None = None,
    fake: bool = False,
    demo_clip: Path | None = None,
) -> FastAPI:
    app = FastAPI(title="Footy Track Inference")
    app.include_router(
        create_router(
            runner,
            max_upload_bytes=max_upload_bytes,
            token=token,
            fake=fake,
            demo_clip=demo_clip,
        )
    )
    app.mount("/infer/static", StaticFiles(directory=str(WEB_DIR)), name="infer-static")
    app.state.runner = runner

    @app.get("/", include_in_schema=False)
    async def root() -> RedirectResponse:
        return RedirectResponse("/infer")

    return app


def build_from_args(
    argv: list[str] | None = None,
) -> tuple[FastAPI, argparse.Namespace]:
    env = os.environ.get
    ap = argparse.ArgumentParser(description="Footy Track inference web app")
    ap.add_argument(
        "--host", default="127.0.0.1", help="use 0.0.0.0 to serve a private network"
    )
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument(
        "--jobs-dir",
        type=Path,
        default=Path(env("FOOTY_INFERENCE_JOBS_DIR", "runs/web")),
    )
    ap.add_argument("--weights", type=Path, default=env("FOOTY_INFERENCE_WEIGHTS"))
    ap.add_argument("--device", default=env("FOOTY_INFERENCE_DEVICE"))
    ap.add_argument(
        "--max-upload-mb",
        type=int,
        default=int(env("FOOTY_INFERENCE_MAX_UPLOAD_MB", DEFAULT_MAX_UPLOAD_MB)),
    )
    ap.add_argument(
        "--demo-clip",
        type=Path,
        default=Path(env("FOOTY_INFERENCE_DEMO_CLIP"))
        if env("FOOTY_INFERENCE_DEMO_CLIP")
        else None,
        help="sample video offered as a one-click demo on the upload page",
    )
    ap.add_argument(
        "--fake",
        action="store_true",
        default=env("FOOTY_INFERENCE_FAKE") == "1",
        help="DEV ONLY: fake detector and fixed-length cutter; no weights or GPU needed",
    )
    args = ap.parse_args(argv)
    token = env("FOOTY_INFERENCE_TOKEN") or None

    if args.fake:
        runner = JobRunner(
            args.jobs_dir,
            detector_factory=fake_detector_factory,
            segmenter=make_chunk_segmenter(),
        )
    else:
        from footy_track.detectors.ultralytics import (  # noqa: PLC0415
            CURRENT_BEST_DETECTOR_CHECKPOINT,
        )
        from footy_track.utils import get_project_root  # noqa: PLC0415

        weights = args.weights or get_project_root() / CURRENT_BEST_DETECTOR_CHECKPOINT
        if not Path(weights).exists():
            ap.error(
                f"detector weights not found: {weights} "
                "(set FOOTY_INFERENCE_WEIGHTS or pass --weights; or --fake for a dev run)"
            )
        runner = JobRunner(
            args.jobs_dir,
            detector_factory=make_real_detector_factory(Path(weights), args.device),
        )
    app = create_app(
        runner,
        max_upload_bytes=args.max_upload_mb * 1024 * 1024,
        token=token,
        fake=args.fake,
        demo_clip=args.demo_clip,
    )
    return app, args


def main(argv: list[str] | None = None) -> int:
    import uvicorn  # noqa: PLC0415

    app, args = build_from_args(argv)
    if args.demo_clip and not args.demo_clip.is_file():
        print("WARNING: demo clip not found; the demo button will be hidden.")
    if args.host not in ("127.0.0.1", "localhost", "::1"):
        print(
            "WARNING: listening beyond localhost. Uploads run on this machine's GPU; "
            "expose this only on a private network."
        )
    uvicorn.run(app, host=args.host, port=args.port)
    return 0
