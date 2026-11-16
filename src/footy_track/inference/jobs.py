"""Job runner for the inference web app: uploaded video -> clips -> tracking.

One job is one uploaded match video. A single worker thread runs jobs one at a
time (there is one GPU) through these stages:

1. **segment**: cut the video into broadcast-shot clips. The default segmenter
   runs ``scripts/split_broadcast_segments.py`` in a subprocess (it is a CLI that
   loads the broadcast classifier), then reads the ``segments.json`` it writes.
2. **track**: ``run_inference`` on every clip, with an overlay video.
3. **stitch**: build ``full.mp4``, the whole upload in order, with the overlay on
   tracked clips and plain footage for everything the cutter dropped
   (``inference/stitch.py``). The timeline of tracked and skipped spans is kept in
   ``job.json``. A stitch failure does not fail the job.

Everything lives under ``<jobs_dir>/<job_id>/`` and state is a ``job.json`` that
is rewritten atomically, so finished jobs survive a restart::

    <job_id>/job.json          state: status, stages, per-clip results, timings
    <job_id>/input.<ext>       the upload (extension from a fixed allow-list)
    <job_id>/clips/clip.mp4... cut clips, renamed to c000.mp4, c001.mp4, ...
    <job_id>/out/c000/         detections/tracklets/ball_track/summary/overlay
    <job_id>/full.mp4          stitched whole video (when the stitch worked)

Ids and file names on disk are generated here; the user's file name is kept only
as a display name inside ``job.json``.
"""

from __future__ import annotations

import json
import logging
import os
import re
import secrets
import shutil
import subprocess
import sys
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import cv2
import numpy as np

from footy_track.inference.pipeline import (
    FrameDetector,
    InferenceConfig,
    run_inference,
    select_device,
)
from footy_track.inference.preview import make_preview
from footy_track.inference.stitch import (
    StitchCancelled,
    compute_timeline,
    fps_fraction,
    stitch_full_video,
)
from footy_track.schema import FrameDetections

logger = logging.getLogger(__name__)

JOB_ID_RE = re.compile(r"^[0-9a-f]{12}$")
CLIP_ID_RE = re.compile(r"^c\d{3,5}$")
ALLOWED_EXTENSIONS = (".mp4", ".mov", ".mkv", ".ts", ".avi", ".webm", ".m4v")
# Downloadable per-clip files: URL name -> (folder under the job dir, file name).
CLIP_FILES = {
    "clip.mp4": ("clips", None),  # file name is the clip id + .mp4
    "overlay.mp4": ("out", "overlay.mp4"),
    "overlay_preview.mp4": ("out", "overlay_preview.mp4"),
    "detections.jsonl": ("out", "detections.jsonl"),
    "tracklets.jsonl": ("out", "tracklets.jsonl"),
    "ball_track.jsonl": ("out", "ball_track.jsonl"),
    "summary.json": ("out", "summary.json"),
}

QUEUED, RUNNING, DONE, FAILED, CANCELLED = (
    "queued",
    "running",
    "done",
    "failed",
    "cancelled",
)
UPLOADING = "uploading"
FINISHED = (DONE, FAILED, CANCELLED)

NO_CLIPS_MESSAGE = (
    "No broadcast shots were found in this video. The cutter keeps only stretches "
    "the classifier calls broadcast footage (and at least a couple of seconds long), "
    "so replays, crowd shots and non-match video are dropped."
)


class JobError(Exception):
    """Base class for errors the API turns into a 4xx."""


class JobNotFound(JobError):
    pass


class BadVideo(JobError):
    pass


class JobBusy(JobError):
    pass


class Cancelled(Exception):
    """Raised inside the worker when the running job is cancelled."""


@dataclass
class DetectorHandle:
    detector: FrameDetector
    checkpoint: str
    device: str
    source_tag: str = "rtdetr"


DetectorFactory = Callable[[], DetectorHandle]


class Segmenter(Protocol):
    def __call__(
        self, video: Path, outdir: Path, cancel: threading.Event
    ) -> list[dict]:
        """Cut ``video`` into clips under ``outdir``.

        Returns ``[{"file": Path, "start_s": float, "end_s": float,
        "frame_count": int}, ...]`` in time order.
        """


# --------------------------------------------------------------------- segmenters


def subprocess_segmenter(
    video: Path,
    outdir: Path,
    cancel: threading.Event,
    *,
    sample: int = 5,
) -> list[dict]:
    """Run ``split_broadcast_segments`` as a subprocess and read its manifest."""
    if not shutil.which("ffmpeg"):
        raise RuntimeError(
            "ffmpeg is not installed on the server; it is needed to cut clips."
        )
    outdir.mkdir(parents=True, exist_ok=True)
    log_path = outdir.parent / "segment.log"
    cmd = [
        sys.executable,
        "-m",
        "footy_track.scripts.split_broadcast_segments",
        str(video),
        "--outdir",
        str(outdir),
        "--sample",
        str(sample),
    ]
    with log_path.open("wb") as log:
        proc = subprocess.Popen(
            cmd,
            stdout=log,
            stderr=subprocess.STDOUT,
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
        )
        try:
            while True:
                try:
                    rc = proc.wait(timeout=0.5)
                    break
                except subprocess.TimeoutExpired:
                    if cancel.is_set():
                        proc.terminate()
                        try:
                            proc.wait(timeout=10)
                        except subprocess.TimeoutExpired:
                            proc.kill()
                        raise Cancelled from None
        finally:
            if proc.poll() is None:
                proc.kill()
    if rc != 0:
        tail = " | ".join(_log_tail(log_path))
        raise RuntimeError(f"Cutting the video failed (exit code {rc}). {tail}")
    manifest = outdir / "segments.json"
    if not manifest.exists():
        raise RuntimeError("The cutter finished without writing a segment list.")
    segments = json.loads(manifest.read_text()).get("segments", [])
    return [
        {
            "file": outdir / s["file"],
            "start_s": s["start_s"],
            "end_s": s["end_s"],
            "frame_count": s["frame_count"],
        }
        for s in segments
        if (outdir / s["file"]).exists()
    ]


def _log_tail(path: Path, n: int = 4) -> list[str]:
    try:
        lines = [ln.strip() for ln in path.read_text(errors="replace").splitlines()]
    except OSError:
        return []
    return [ln for ln in lines if ln][-n:]


def make_chunk_segmenter(chunk_frames: int = 100, gap_frames: int = 0) -> Segmenter:
    """DEV ONLY: cut into fixed-length chunks with OpenCV, no classifier or ffmpeg.

    With ``gap_frames`` > 0, that many frames after every chunk are left out, like
    the cutaways the real cutter drops, so the stitched video has gaps to fill.

    Used with the fake detector so the whole flow runs on a machine that has
    neither the classifier weights nor a GPU. It does not find broadcast shots.
    """

    def segmenter(video: Path, outdir: Path, cancel: threading.Event) -> list[dict]:
        outdir.mkdir(parents=True, exist_ok=True)
        cap = cv2.VideoCapture(str(video))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out: list[dict] = []
        writer = None
        count = start = 0
        index = 0
        while True:
            if cancel.is_set():
                raise Cancelled
            ok, frame = cap.read()
            if ok and writer is None:
                path = outdir / f"chunk{len(out):03d}.mp4"
                writer = cv2.VideoWriter(
                    str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
                )
                start, count = index, 0
            if ok:
                writer.write(frame)
                count += 1
                index += 1
            if writer is not None and (not ok or count >= chunk_frames):
                writer.release()
                writer = None
                out.append(
                    {
                        "file": path,
                        "start_s": round(start / fps, 3),
                        "end_s": round((start + count) / fps, 3),
                        "frame_count": count,
                    }
                )
                for _ in range(gap_frames if ok else 0):  # leave a gap
                    if not cap.read()[0]:
                        break
                    index += 1
            if not ok:
                break
        cap.release()
        return out

    return segmenter


# --------------------------------------------------------------------- detectors


def make_real_detector_factory(
    weights: Path, device: str | None = None, imgsz: int | None = None
) -> DetectorFactory:
    """Load the RT-DETR detector once, on first use, and reuse it for every clip."""
    lock = threading.Lock()
    cache: list[DetectorHandle] = []

    def factory() -> DetectorHandle:
        with lock:
            if not cache:
                from footy_track.detectors.ultralytics import (  # noqa: PLC0415
                    DEFAULT_IMGSZ,
                    UltralyticsObjectDetector,
                )

                dev = device or select_device()
                detector = UltralyticsObjectDetector(
                    model_uri=str(weights),
                    min_confidence=InferenceConfig().detect_conf,
                    imgsz=imgsz or DEFAULT_IMGSZ,
                    use_model_names=True,
                    device=dev,
                )
                tag = "rtdetr" if "rtdetr" in str(weights).lower() else "yolo"
                cache.append(DetectorHandle(detector, str(weights), dev, tag))
            return cache[0]

    return factory


def fake_detector_factory() -> DetectorHandle:
    """DEV ONLY: a fresh fake detector per clip (it counts frames from zero)."""
    from footy_track.inference.fake_detector import FakeDetector  # noqa: PLC0415

    return DetectorHandle(FakeDetector(), "fake-detector", "none", "fake")


class _ProgressDetector:
    """Wraps a detector to report frames done and to abort on cancel."""

    def __init__(
        self,
        inner: FrameDetector,
        cancel: threading.Event,
        on_frames: Callable[[int], None],
    ) -> None:
        self.inner, self.cancel, self.on_frames = inner, cancel, on_frames
        self.done = 0

    def predict_frames(self, frames: list[np.ndarray]) -> list[FrameDetections]:
        if self.cancel.is_set():
            raise Cancelled
        result = self.inner.predict_frames(frames)
        self.done += len(frames)
        self.on_frames(self.done)
        return result


# ------------------------------------------------------------------ overlay codec


def make_overlay_browser_playable(path: Path) -> dict:
    """Re-encode ``path`` (OpenCV writes mp4v) to H.264/yuv420p for browsers.

    Returns ``{"codec": "h264"|"mp4v", "browser_playable": bool}``. Without ffmpeg,
    or if it fails, the original file is kept and reported as not playable.
    """
    fallback = {"codec": "mp4v", "browser_playable": False}
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg or not path.exists():
        return fallback
    tmp = path.with_name("overlay.h264.tmp.mp4")
    cmd = [
        ffmpeg, "-hide_banner", "-y", "-loglevel", "error", "-i", str(path),
        "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "veryfast",
        "-crf", "23", "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
        "-movflags", "+faststart", str(tmp),
    ]  # fmt: skip
    try:
        proc = subprocess.run(cmd, check=False, capture_output=True, timeout=1800)
    except (OSError, subprocess.TimeoutExpired):
        tmp.unlink(missing_ok=True)
        return fallback
    if proc.returncode != 0 or not tmp.exists() or tmp.stat().st_size == 0:
        tmp.unlink(missing_ok=True)
        logger.warning("overlay re-encode failed: %s", proc.stderr[-300:])
        return fallback
    os.replace(tmp, path)
    return {"codec": "h264", "browser_playable": True}


# ------------------------------------------------------------------------ runner


def _now() -> float:
    return round(time.time(), 2)


def probe_video(path: Path) -> dict:
    """Open ``path`` and read one frame, or raise ``BadVideo``."""
    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            raise BadVideo("This file could not be opened as a video.")
        fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
        frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        ok, _ = cap.read()
        if not ok or width <= 0 or height <= 0:
            raise BadVideo("This video has no readable frames.")
    finally:
        cap.release()
    return {
        "fps": round(fps, 3),
        "frames": frames,
        "width": width,
        "height": height,
        "duration_s": round(frames / fps, 1) if fps else None,
    }


class JobRunner:
    """Owns the jobs directory, the job state and the single worker thread."""

    def __init__(
        self,
        jobs_dir: Path,
        *,
        detector_factory: DetectorFactory,
        segmenter: Segmenter = subprocess_segmenter,
        start_worker: bool = True,
        preview: bool = True,
    ) -> None:
        self.jobs_dir = Path(jobs_dir)
        self.jobs_dir.mkdir(parents=True, exist_ok=True)
        self.detector_factory = detector_factory
        self.segmenter = segmenter
        self.preview = preview  # also make small copies for slow networks
        self._lock = threading.RLock()
        self._jobs: dict[str, dict] = {}
        self._queue: list[str] = []
        self._wake = threading.Condition(self._lock)
        self._running_id: str | None = None
        self._cancel = threading.Event()
        self._stop = False
        self._last_save = 0.0
        self._load_existing()
        self._thread: threading.Thread | None = None
        if start_worker:
            self._thread = threading.Thread(
                target=self._worker, name="inference-worker", daemon=True
            )
            self._thread.start()

    # -- paths ------------------------------------------------------------
    def job_dir(self, job_id: str) -> Path:
        if not JOB_ID_RE.match(job_id):
            raise JobNotFound(job_id)
        return self.jobs_dir / job_id

    def clip_file(self, job_id: str, clip_id: str, name: str) -> Path:
        """Resolve a downloadable clip file, rejecting anything not allow-listed."""
        job = self.get(job_id)
        if not CLIP_ID_RE.match(clip_id) or name not in CLIP_FILES:
            raise JobNotFound(name)
        if clip_id not in {c["id"] for c in job["clips"]}:
            raise JobNotFound(clip_id)
        folder, fname = CLIP_FILES[name]
        base = self.job_dir(job_id)
        path = (
            base / "clips" / f"{clip_id}.mp4"
            if folder == "clips"
            else base / "out" / clip_id / fname
        )
        if not path.is_file():
            raise JobNotFound(name)
        return path

    def full_video_file(self, job_id: str) -> Path:
        """The stitched ``full.mp4`` of a job, or ``JobNotFound``."""
        self.get(job_id)
        path = self.job_dir(job_id) / "full.mp4"
        if not path.is_file():
            raise JobNotFound("full.mp4")
        return path

    def full_preview_file(self, job_id: str) -> Path:
        """The small ``full_preview.mp4`` of a job, or ``JobNotFound``."""
        self.get(job_id)
        path = self.job_dir(job_id) / "full_preview.mp4"
        if not path.is_file():
            raise JobNotFound("full_preview.mp4")
        return path

    # -- persistence ------------------------------------------------------
    def _load_existing(self) -> None:
        for d in sorted(self.jobs_dir.iterdir()):
            if not (d.is_dir() and JOB_ID_RE.match(d.name)):
                continue
            try:
                job = json.loads((d / "job.json").read_text())
            except (OSError, ValueError):
                shutil.rmtree(d, ignore_errors=True)  # half-created, no state
                continue
            if job["status"] == UPLOADING:
                shutil.rmtree(d, ignore_errors=True)
                continue
            if job["status"] == RUNNING:
                job["status"] = FAILED
                job["error"] = "The server restarted while this job was running."
                job["finished"] = _now()
                self._mark_unfinished(job)
            job.setdefault("timeline", None)
            job.setdefault("full_video", None)
            self._jobs[job["id"]] = job
            if job["status"] == QUEUED:
                self._queue.append(job["id"])
            self._save(job, force=True)

    @staticmethod
    def _mark_unfinished(job: dict) -> None:
        for stage in job["stages"].values():
            if stage["status"] == RUNNING:
                stage["status"] = FAILED
        for clip in job["clips"]:
            if clip["status"] in (QUEUED, RUNNING):
                clip["status"] = FAILED
                clip["error"] = "interrupted"

    def _save(self, job: dict, force: bool = True) -> None:
        with self._lock:
            path = self.jobs_dir / job["id"] / "job.json"
            if not path.parent.exists():
                return
            tmp = path.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(job, indent=2))
            os.replace(tmp, path)

    # -- public API -------------------------------------------------------
    def allocate(self, display_name: str, ext: str, options: dict) -> tuple[str, Path]:
        """Create a job in the ``uploading`` state; the caller writes the file."""
        ext = ext.lower() if ext.lower() in ALLOWED_EXTENSIONS else ".mp4"
        with self._lock:
            while True:
                job_id = secrets.token_hex(6)
                if job_id not in self._jobs and not (self.jobs_dir / job_id).exists():
                    break
            (self.jobs_dir / job_id).mkdir()
            job = {
                "id": job_id,
                "name": (display_name or "video")[:120],
                "status": UPLOADING,
                "error": None,
                "created": _now(),
                "started": None,
                "finished": None,
                "options": options,
                "video": None,
                "stages": {
                    "segment": {"status": QUEUED, "seconds": None},
                    "track": {
                        "status": QUEUED,
                        "seconds": None,
                        "clips_done": 0,
                        "clips_total": 0,
                    },
                    "stitch": {"status": QUEUED, "seconds": None},
                },
                "clips": [],
                "timeline": None,
                "full_video": None,
                "full_preview": None,
                "overlay_note": None,
            }
            self._jobs[job_id] = job
            self._save(job)
        return job_id, self.jobs_dir / job_id / f"input{ext}"

    def abort_upload(self, job_id: str) -> None:
        with self._lock:
            self._jobs.pop(job_id, None)
        shutil.rmtree(self.jobs_dir / job_id, ignore_errors=True)

    def submit(self, job_id: str) -> dict:
        """Validate the uploaded file and queue the job. Raises ``BadVideo``."""
        job = self._jobs[job_id]
        inputs = list((self.jobs_dir / job_id).glob("input.*"))
        try:
            if not inputs:
                raise BadVideo("No file was received.")
            video = probe_video(inputs[0])
        except BadVideo:
            self.abort_upload(job_id)
            raise
        with self._lock:
            job["video"] = video
            job["status"] = QUEUED
            self._queue.append(job_id)
            self._save(job)
            self._wake.notify_all()
        return self.get(job_id)

    def get(self, job_id: str) -> dict:
        if not JOB_ID_RE.match(job_id):
            raise JobNotFound(job_id)
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None or job["status"] == UPLOADING:
                raise JobNotFound(job_id)
            return json.loads(json.dumps(job))

    def list(self) -> list[dict]:
        with self._lock:
            jobs = [j for j in self._jobs.values() if j["status"] != UPLOADING]
            jobs.sort(key=lambda j: j["created"], reverse=True)
            return [self._brief(j) for j in jobs]

    @staticmethod
    def _brief(job: dict) -> dict:
        keys = ("id", "name", "status", "error", "created", "finished", "video")
        out = {k: job[k] for k in keys}
        out["clips"] = len(job["clips"])
        out["stages"] = json.loads(json.dumps(job["stages"]))
        return out

    def cancel(self, job_id: str) -> dict:
        with self._lock:
            self.get(job_id)
            job = self._jobs[job_id]
            if job["status"] == QUEUED:
                self._queue.remove(job_id)
                job["status"] = CANCELLED
                job["finished"] = _now()
                self._save(job)
            elif job["status"] == RUNNING and self._running_id == job_id:
                self._cancel.set()
        return self.get(job_id)

    def delete(self, job_id: str) -> None:
        with self._lock:
            self.get(job_id)
            job = self._jobs[job_id]
            if job["status"] in (QUEUED, RUNNING):
                raise JobBusy("Cancel the job before deleting it.")
            del self._jobs[job_id]
        shutil.rmtree(self.jobs_dir / job_id, ignore_errors=True)

    def wait_idle(self, timeout: float = 60.0) -> bool:
        """Block until no job is queued or running (for tests)."""
        end = time.time() + timeout
        while time.time() < end:
            with self._lock:
                if not self._queue and self._running_id is None:
                    return True
            time.sleep(0.02)
        return False

    def shutdown(self) -> None:
        with self._lock:
            self._stop = True
            self._cancel.set()
            self._wake.notify_all()
        if self._thread:
            self._thread.join(timeout=10)

    # -- worker -----------------------------------------------------------
    def _worker(self) -> None:
        while True:
            with self._lock:
                while not self._queue and not self._stop:
                    self._wake.wait()
                if self._stop:
                    return
                job_id = self._queue.pop(0)
                self._running_id = job_id
                self._cancel.clear()
            try:
                self._run_job(self._jobs[job_id])
            except Exception:  # noqa: BLE001 - never let the worker die
                logger.exception("job %s crashed", job_id)
                with self._lock:
                    job = self._jobs[job_id]
                    job["status"] = FAILED
                    job["error"] = "Unexpected server error; see the server log."
                    job["finished"] = _now()
                    self._mark_unfinished(job)
                    self._save(job)
            finally:
                with self._lock:
                    self._running_id = None

    def _update(self, job: dict, fn: Callable[[dict], None]) -> None:
        with self._lock:
            fn(job)
            self._save(job)

    def _fail(self, job: dict, message: str) -> None:
        def f(j: dict) -> None:
            j["status"], j["error"], j["finished"] = FAILED, message, _now()
            self._mark_unfinished(j)

        self._update(job, f)

    def _run_job(self, job: dict) -> None:  # noqa: PLR0915
        job_dir = self.jobs_dir / job["id"]
        video = next(job_dir.glob("input.*"))
        seg, trk = job["stages"]["segment"], job["stages"]["track"]
        t_job = time.time()
        self._update(
            job, lambda j: j.update(status=RUNNING, started=_now(), error=None)
        )
        try:
            # Fail early (and cheaply) if the detector cannot load, e.g. weights missing.
            try:
                self.detector_factory()
            except Exception as exc:
                logger.exception("detector failed to load")
                raise RuntimeError(f"The detector could not be loaded: {exc}") from exc

            # --- stage 1: cut ---
            self._update(job, lambda _j: seg.update(status=RUNNING))
            t0 = time.time()
            clips_dir = job_dir / "clips"
            raw_dir = job_dir / "cut"
            try:
                cuts = self.segmenter(video, raw_dir, self._cancel)
            except Cancelled:
                raise
            except Exception as exc:
                self._update(job, lambda _j: seg.update(status=FAILED))
                raise RuntimeError(str(exc)) from exc
            clips_dir.mkdir(exist_ok=True)
            clips = []
            for i, cut in enumerate(cuts):
                cid = f"c{i:03d}"
                shutil.move(str(cut["file"]), clips_dir / f"{cid}.mp4")
                clips.append(
                    {
                        "id": cid,
                        "status": QUEUED,
                        "error": None,
                        "start_s": cut["start_s"],
                        "end_s": cut["end_s"],
                        "frame_count": cut["frame_count"],
                        "frames_done": 0,
                        "frames_total": self._planned_frames(
                            cut["frame_count"], job["options"]
                        ),
                        "seconds": None,
                        "summary": None,
                        "overlay": None,
                    }
                )
            shutil.rmtree(raw_dir, ignore_errors=True)
            seg_seconds = round(time.time() - t0, 2)

            def after_cut(j: dict) -> None:
                seg.update(status=DONE, seconds=seg_seconds)
                j["clips"] = clips
                trk["clips_total"] = len(clips)

            self._update(job, after_cut)
            if not clips:
                self._update(job, lambda _j: trk.update(status=FAILED))
                raise RuntimeError(NO_CLIPS_MESSAGE)

            # --- stage 2: track each clip ---
            self._update(job, lambda _j: trk.update(status=RUNNING))
            t1 = time.time()
            for clip in clips:
                self._track_clip(job, clip, job_dir)
                self._update(
                    job,
                    lambda _j: trk.update(
                        clips_done=sum(c["status"] in (DONE, FAILED) for c in clips)
                    ),
                )
            ok = sum(c["status"] == DONE for c in clips)
            if ok:
                self._stitch(job, video, job_dir)

            def finish(j: dict) -> None:
                trk.update(
                    status=DONE if ok else FAILED,
                    seconds=round(time.time() - t1, 2),
                )
                j["status"] = DONE if ok else FAILED
                if not ok:
                    j["error"] = "Tracking failed on every clip."
                j["finished"] = _now()
                j["total_seconds"] = round(time.time() - t_job, 2)

            self._update(job, finish)
        except Cancelled:
            self._update(
                job,
                lambda j: (
                    j.update(status=CANCELLED, finished=_now()),
                    self._mark_unfinished(j),
                ),
            )
        except Exception as exc:  # noqa: BLE001
            self._fail(job, str(exc))

    def _make_preview(self, src: Path, dst: Path, what: str) -> dict:
        """Best-effort small copy of ``src``. Never raises except ``Cancelled``.

        Returns ``{"status": "done", "bytes", "seconds"}`` or
        ``{"status": "failed", "error"}``; ``{"status": "off"}`` when disabled.
        """
        if not self.preview:
            return {"status": "off"}
        try:
            info = make_preview(src, dst, self._cancel.is_set)
            return {"status": DONE, **info}
        except StitchCancelled:
            raise Cancelled from None
        except Exception as exc:  # noqa: BLE001 - the full-quality file is still good
            logger.warning("preview of %s failed: %s", what, exc)
            return {"status": FAILED, "error": f"{type(exc).__name__}: {exc}"[:300]}

    def _stitch(self, job: dict, video: Path, job_dir: Path) -> None:
        """Stage 3. Never raises except ``Cancelled``: per-clip results stay usable."""
        stitch = job["stages"]["stitch"]
        t0 = time.time()
        try:
            info = job["video"]
            fps = float(fps_fraction(info["fps"] or 25.0))  # job.json rounds fps
            stride = int(job["options"].get("stride") or 1)
            spans = compute_timeline(job["clips"], int(info["frames"]), fps, stride)
            self._update(
                job,
                lambda j: (
                    j.update(timeline=spans),
                    stitch.update(
                        status=RUNNING, pieces_done=0, pieces_total=len(spans)
                    ),
                ),
            )

            def progress(done: int, total: int) -> None:
                stitch["pieces_done"], stitch["pieces_total"] = done, total
                self._update(job, lambda _j: None)

            stitch_full_video(video, job_dir, spans, fps, self._cancel.is_set, progress)
            total = spans[-1]["end_frame"] if spans else 0
            self._update(
                job,
                lambda j: (
                    stitch.update(status=DONE, seconds=round(time.time() - t0, 2)),
                    j.update(full_video={"frames": total, "fps": fps, "audio": False}),
                ),
            )
        except StitchCancelled:
            raise Cancelled from None
        except Exception as exc:  # noqa: BLE001 - the clips are still good
            logger.exception("stitching job %s failed", job["id"])
            (job_dir / "full.tmp.mp4").unlink(missing_ok=True)
            message = f"{type(exc).__name__}: {exc}"[:300]
            self._update(
                job,
                lambda j: (
                    stitch.update(
                        status=FAILED, error=message, seconds=round(time.time() - t0, 2)
                    ),
                    j.update(full_video=None),
                ),
            )
            return
        # Outside the try above: a cancel here must stay a cancel, not a stitch failure.
        preview = self._make_preview(
            job_dir / "full.mp4", job_dir / "full_preview.mp4", "full video"
        )
        self._update(job, lambda j: j.update(full_preview=preview))

    @staticmethod
    def _planned_frames(frame_count: int, options: dict) -> int:
        n = frame_count
        stride = max(int(options.get("stride") or 1), 1)
        n = -(-n // stride)
        cap = options.get("max_frames")
        return min(n, int(cap)) if cap else n

    def _track_clip(self, job: dict, clip: dict, job_dir: Path) -> None:
        if self._cancel.is_set():
            raise Cancelled
        cid = clip["id"]
        out_dir = job_dir / "out" / cid
        self._update(job, lambda _j: clip.update(status=RUNNING))
        t0 = time.time()
        try:
            handle = self.detector_factory()

            def on_frames(n: int) -> None:
                clip["frames_done"] = n  # persisted with the next state save
                if time.time() - self._last_save > 1.0:
                    self._last_save = time.time()
                    self._update(job, lambda _j: None)

            detector = _ProgressDetector(handle.detector, self._cancel, on_frames)
            opts = job["options"]
            config = InferenceConfig(
                max_frames=opts.get("max_frames"),
                stride=opts.get("stride") or 1,
                overlay=True,
            )
            summary = run_inference(
                job_dir / "clips" / f"{cid}.mp4",
                out_dir,
                detector,  # type: ignore[arg-type]
                config,
                checkpoint=handle.checkpoint,
                device=handle.device,
                source_tag=handle.source_tag,
            )
            overlay = (
                make_overlay_browser_playable(out_dir / "overlay.mp4")
                if (out_dir / "overlay.mp4").exists()
                else None
            )
            preview = (
                self._make_preview(
                    out_dir / "overlay.mp4", out_dir / "overlay_preview.mp4", cid
                )
                if overlay
                else None
            )
            summary["video"] = f"{cid}.mp4"  # do not leak server paths
            (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

            def ok(_j: dict) -> None:
                clip.update(
                    status=DONE,
                    seconds=round(time.time() - t0, 2),
                    summary=summary,
                    overlay=overlay,
                    preview=preview,
                    frames_done=summary["frames_processed"],
                )
                if overlay and not overlay["browser_playable"]:
                    job["overlay_note"] = (
                        "ffmpeg was not available (or failed), so overlay videos "
                        "are in a format many browsers cannot play. Download them "
                        "and play them locally."
                    )

            self._update(job, ok)
        except Cancelled:
            raise
        except Exception as exc:  # noqa: BLE001 - one bad clip must not sink the job
            logger.exception("clip %s failed", cid)
            message = f"{type(exc).__name__}: {exc}"[:300]
            self._update(
                job,
                lambda _j: clip.update(
                    status=FAILED,
                    error=message,
                    seconds=round(time.time() - t0, 2),
                ),
            )


def jobs_dir_from_env() -> Path:
    return Path(os.environ.get("FOOTY_INFERENCE_JOBS_DIR", "runs/web"))
