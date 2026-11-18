"""Stage 1 of the candidate pool: RT-DETR over the main-camera segments of one match, every ``stride``-th frame.

Design ``docs/design/ball_sampling.md`` §3.1 step 1 and §6 (Loop 1, train branch); conclusions of
``experiments/ball_candidate_pool`` (branch ``exp/ball-candidate-pool``): RT-DETR-L with the production weights at
imgsz 1280, conf >= 0.1, is the one detector source worth keeping (the YOLO 3ch / 5ch arms had zero unique hits).

Streaming: frames are decoded sequentially and batched, nothing is held beyond one batch; the skipped frames are only
``grab()``-ed (demuxed and decoded, never converted). Output is one ``<clip>.npz`` per segment plus a ``meta.json`` that is
written LAST, so a directory with ``meta.json`` is a finished stage and a crash leaves nothing that looks complete.

Layout (``<out>`` = ``<FD>/ball_pool/<match>/candidates.<detector_sha8>/``)::

    <clip>.npz   frames int32[n]; ball_ptr int32[n+1], ball float32[m,5]; person_ptr, person float32[k,5];
                 width, height, fps, n_src_frames, stride (0-d)
                 ball / person rows: x, y, w, h (normalised, top-left like every box in footy_track), conf
    meta.json    schema, detector (sha8 / sha256 / weights name), imgsz, conf, stride, per-clip stats, expected clips, complete

The tracker, the motion source and the pool read these; none of them needs the detector or a GPU.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import sys
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

SCHEMA = "ball-candidates/v1"
STRIDE = 5
IMGSZ = 1280
CONF = 0.1  # detector floor; the scorer, not a threshold here, decides what a candidate is worth
PERSON_CONF = 0.3  # person boxes only mask the motion source and feed distance features
BATCH = 8
SEGMENTS_DIR = "split_video_broadcast_frames"
PERSON_TAGS = frozenset({"player", "referee", "coach", "person", "player_sub"})
BALL_TAGS = frozenset({"in_play_ball", "ball"})
META = "meta.json"


# ---------------------------------------------------------------------------------------------- identities / paths
def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while chunk := fh.read(8 << 20):
            h.update(chunk)
    return h.hexdigest()


def sha8(sha256: str) -> str:
    return sha256[:8]


def pool_dir(root: str | Path, match: str) -> Path:
    """``<root>/ball_pool/<match>``: where every stage of one match writes."""
    return Path(root) / "ball_pool" / match


def candidates_dir(root: str | Path, match: str, detector_sha8: str) -> Path:
    return pool_dir(root, match) / f"candidates.{detector_sha8}"


def stage_done(d: str | Path) -> bool:
    """A stage directory is finished when its ``meta.json`` exists and says ``complete``."""
    try:
        return bool(json.loads((Path(d) / META).read_text()).get("complete"))
    except (OSError, ValueError):
        return False


def atomic_write_json(path: Path, doc: Any) -> None:
    tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    tmp.write_text(json.dumps(doc, indent=1, sort_keys=True))
    os.replace(tmp, path)


@dataclass(frozen=True)
class Segment:
    clip: str  # file stem, e.g. ``arsenal-vs-chelsea-2026-27_seg000``
    path: Path
    index: int
    frame_count: int
    fps: float


def segments_of(
    root: str | Path,
    match: str,
    *,
    limit: int | None = None,
    only: list[str] | None = None,
) -> list[Segment]:
    """Main-camera segments of ``match`` (``segments.json`` lists only the broadcast-view cuts) whose file exists.

    ``only`` restricts to clip stems (dry runs); ``limit`` keeps the first N after that."""
    seg_dir = Path(root) / match / SEGMENTS_DIR
    doc = json.loads((seg_dir / "segments.json").read_text())
    fps = float(doc.get("fps") or 25.0)
    out = []
    for s in doc.get("segments") or []:
        f = s.get("file")
        if not f or not (seg_dir / f).is_file():
            continue
        stem = Path(f).stem
        if only and stem not in only:
            continue
        out.append(
            Segment(
                stem,
                seg_dir / f,
                int(s.get("index", len(out))),
                int(s.get("frame_count") or 0),
                fps,
            )
        )
    return out[:limit] if limit else out


# ---------------------------------------------------------------------------------------------- storage
@dataclass
class ClipCandidates:
    """Detector output of one clip at the sampled frames. Boxes are normalised top-left xywh + conf."""

    frames: np.ndarray  # int32 [n]
    ball_ptr: np.ndarray  # int32 [n+1]
    ball: np.ndarray  # float32 [m, 5]
    person_ptr: np.ndarray
    person: np.ndarray
    width: int
    height: int
    fps: float
    n_src_frames: int
    stride: int
    _index: dict[int, int] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        self._index = {int(f): i for i, f in enumerate(self.frames)}

    def has(self, frame: int) -> bool:
        return frame in self._index

    def balls(self, frame: int) -> np.ndarray:
        i = self._index.get(frame)
        return (
            np.zeros((0, 5), np.float32)
            if i is None
            else self.ball[self.ball_ptr[i] : self.ball_ptr[i + 1]]
        )

    def persons(self, frame: int) -> np.ndarray:
        i = self._index.get(frame)
        return (
            np.zeros((0, 5), np.float32)
            if i is None
            else self.person[self.person_ptr[i] : self.person_ptr[i + 1]]
        )

    def persons_px(self, frame: int, w: float = 1280.0, h: float = 720.0) -> np.ndarray:
        """Person boxes as ``(cx, cy, w, h)`` pixels of a ``w x h`` frame (the motion source's convention)."""
        p = self.persons(frame)
        if not len(p):
            return np.zeros((0, 4), np.float32)
        return np.stack(
            [
                (p[:, 0] + p[:, 2] / 2) * w,
                (p[:, 1] + p[:, 3] / 2) * h,
                p[:, 2] * w,
                p[:, 3] * h,
            ],
            1,
        ).astype(np.float32)

    def nearest_sampled(self, frame: int) -> int | None:
        """The sampled frame closest to ``frame`` (ties to the earlier), None for an empty clip."""
        if not len(self.frames):
            return None
        i = int(np.searchsorted(self.frames, frame))
        cands = [int(self.frames[j]) for j in (i - 1, i) if 0 <= j < len(self.frames)]
        return min(cands, key=lambda f: (abs(f - frame), f))


def _csr(rows: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    ptr = np.zeros(len(rows) + 1, np.int32)
    ptr[1:] = np.cumsum([len(r) for r in rows])
    flat = (
        np.concatenate(rows).astype(np.float32)
        if rows and ptr[-1]
        else np.zeros((0, 5), np.float32)
    )
    return ptr, flat


def build_clip(
    frames: list[int],
    balls: list[np.ndarray],
    persons: list[np.ndarray],
    *,
    size: tuple[int, int],
    fps: float,
    n_src_frames: int,
    stride: int,
) -> ClipCandidates:
    bp, b = _csr(balls)
    pp, p = _csr(persons)
    return ClipCandidates(
        np.asarray(frames, np.int32),
        bp,
        b,
        pp,
        p,
        int(size[0]),
        int(size[1]),
        float(fps),
        int(n_src_frames),
        int(stride),
    )


def save_clip(c: ClipCandidates, path: str | Path) -> None:
    path = Path(path)
    tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}.npz")
    np.savez(
        tmp, frames=c.frames, ball_ptr=c.ball_ptr, ball=c.ball, person_ptr=c.person_ptr, person=c.person,
        width=c.width, height=c.height, fps=c.fps, n_src_frames=c.n_src_frames, stride=c.stride,
    )  # fmt: skip
    os.replace(tmp, path)


def load_clip(path: str | Path) -> ClipCandidates:
    z = np.load(path)
    return ClipCandidates(
        z["frames"], z["ball_ptr"], z["ball"], z["person_ptr"], z["person"],
        int(z["width"]), int(z["height"]), float(z["fps"]), int(z["n_src_frames"]), int(z["stride"]),
    )  # fmt: skip


# ---------------------------------------------------------------------------------------------- inference
def iter_sampled_frames(
    path: str | Path, stride: int
) -> Iterator[tuple[int, np.ndarray]]:
    """``(frame_index, bgr)`` for frames ``0, stride, 2*stride...`` of ``path``, decoded sequentially.

    Skipped frames are ``grab()``-ed only (no colour conversion, no copy). Sequential decode is frame-exact; seeking is not."""
    import cv2  # noqa: PLC0415

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(f"cannot open video {path}")
    try:
        i = 0
        while cap.grab():
            if i % stride == 0:
                ok, frame = cap.retrieve()
                if not ok:
                    return
                yield i, frame
            i += 1
    finally:
        cap.release()


def video_info(path: str | Path) -> tuple[tuple[int, int], float, int]:
    import cv2  # noqa: PLC0415

    cap = cv2.VideoCapture(str(path))
    try:
        size = (
            int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        )
        return (
            size,
            float(cap.get(cv2.CAP_PROP_FPS) or 0.0),
            int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        )
    finally:
        cap.release()


def rows_of(
    detections: Any, ball_labels=BALL_TAGS, person_labels=PERSON_TAGS
) -> tuple[np.ndarray, np.ndarray]:
    """``(balls, persons)`` arrays (x, y, w, h, conf) from a ``FrameDetections.detections`` list."""
    b, p = [], []
    for d in detections:
        row = (d.x, d.y, d.w, d.h, d.confidence)
        if d.label in ball_labels:
            b.append(row)
        elif d.label in person_labels and d.confidence >= PERSON_CONF:
            p.append(row)
    f = lambda r: np.asarray(r, np.float32).reshape(-1, 5)  # noqa: E731
    return f(b), f(p)


def detect_clip(
    video: str | Path,
    detector: Any,
    *,
    stride: int = STRIDE,
    batch: int = BATCH,
    on_batch: Callable[[int], None] | None = None,
) -> ClipCandidates:
    """Run ``detector`` (``predict_frames(list[bgr]) -> list[FrameDetections]``) over every ``stride``-th frame of ``video``."""
    size, fps, n_src = video_info(video)
    frames: list[int] = []
    balls: list[np.ndarray] = []
    persons: list[np.ndarray] = []
    pending: list[tuple[int, np.ndarray]] = []
    capped = [False]

    def flush() -> None:
        results = detector.predict_frames([f for _, f in pending])
        if not capped[
            0
        ]:  # ultralytics' first predict() on cpu calls torch.set_num_threads(cpus - 1): re-apply our cap after it
            limit_threads()
            capped[0] = True
        for (index, _), res in zip(pending, results, strict=True):
            b, p = rows_of(res.detections)
            frames.append(index)
            balls.append(b)
            persons.append(p)
        if on_batch:
            on_batch(len(pending))
        pending.clear()

    for item in iter_sampled_frames(video, stride):
        pending.append(item)
        if len(pending) >= batch:
            flush()
    if pending:
        flush()
    return build_clip(
        frames, balls, persons, size=size, fps=fps, n_src_frames=n_src, stride=stride
    )


def thread_cap() -> int:
    """Threads a stage may use: ``$BALL_POOL_THREADS``, else ``$OMP_NUM_THREADS``, else 4 (merlin shares its cores)."""
    for k in ("BALL_POOL_THREADS", "OMP_NUM_THREADS"):
        try:
            if int(os.environ.get(k, "")) > 0:
                return int(os.environ[k])
        except ValueError:
            pass
    return 4


def limit_threads(n: int | None = None) -> int:
    """Cap OpenCV and (if loaded) torch intra-op threads. ultralytics' ``select_device("cpu")`` calls
    ``torch.set_num_threads(min(8, cpus - 1))`` on the first ``predict()``, overriding ``OMP_NUM_THREADS``: a CPU run used 7
    cores of a shared host until the cap was re-applied after that first call (:func:`detect_clip` does)."""
    n = n or thread_cap()
    import cv2  # noqa: PLC0415

    cv2.setNumThreads(n)
    if "torch" in sys.modules:
        with contextlib.suppress(Exception):  # best effort
            sys.modules["torch"].set_num_threads(n)
    return n


def make_detector(
    weights: str | Path,
    *,
    imgsz: int = IMGSZ,
    conf: float = CONF,
    device: str | None = None,
) -> Any:
    """The production detector wrapper (shared with ``datasets.detector_run``); imports ultralytics lazily."""
    from footy_track.datasets.detector_run import make_detector as _make  # noqa: I001, PLC0415

    det = _make(Path(weights), imgsz, conf, device)
    limit_threads()
    return det


def run_match(
    segments: list[Segment],
    out: str | Path,
    *,
    detector: Any,
    weights: str | Path,
    stride: int = STRIDE,
    imgsz: int = IMGSZ,
    conf: float = CONF,
    batch: int = BATCH,
    expected: list[str] | None = None,
    max_seconds: float | None = None,
    code_sha: str | None = None,
    log: Callable[[str], None] = print,
) -> dict:
    """Detect every segment not yet in ``out`` and (re)write ``meta.json`` last.

    Resumable per clip. ``expected`` = every clip stem the stage must cover (default: ``segments``); ``complete`` is True
    only when all of them are present. ``max_seconds`` stops starting new clips after that long (a rerun continues)."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    sha256 = sha256_file(weights)
    want = expected or [s.clip for s in segments]
    prev: dict = {}
    with contextlib.suppress(OSError, ValueError):
        prev = json.loads((out / META).read_text())
    if prev and (
        prev.get("detector", {}).get("sha256") != sha256
        or prev.get("stride") != stride
        or prev.get("imgsz") != imgsz
    ):
        raise RuntimeError(
            f"{out} holds candidates made with other settings ({prev.get('detector')}, stride {prev.get('stride')})"
        )
    clips: dict[str, dict] = dict(prev.get("clips") or {})
    status = {
        "done": [],
        "skipped": [],
        "failed": {},
        "frames": 0,
        "seconds": 0.0,
        "stopped_early": False,
    }
    t_start = time.time()
    for n, seg in enumerate(segments, 1):
        dst = out / f"{seg.clip}.npz"
        if dst.is_file() and seg.clip in clips:
            status["skipped"].append(seg.clip)
            continue
        if max_seconds is not None and time.time() - t_start > max_seconds:
            status["stopped_early"] = True
            break
        t0 = time.time()
        try:
            c = detect_clip(seg.path, detector, stride=stride, batch=batch)
        except Exception as e:  # noqa: BLE001 - one bad clip must not sink the run; the next run retries it
            status["failed"][seg.clip] = f"{type(e).__name__}: {e}"
            log(
                f"[{n}/{len(segments)}] {seg.clip}: FAILED {status['failed'][seg.clip]}"
            )
            continue
        save_clip(c, dst)
        dt = time.time() - t0
        clips[seg.clip] = {
            "n_sampled": int(len(c.frames)),
            "n_balls": int(len(c.ball)),
            "fps": c.fps,
            "size": [c.width, c.height],
            "seconds": round(dt, 2),
        }
        status["done"].append(seg.clip)
        status["frames"] += int(len(c.frames))
        status["seconds"] += dt
        log(
            f"[{n}/{len(segments)}] {seg.clip}: {len(c.frames)} frames, {len(c.frames) / max(dt, 1e-9):.1f} fps"
        )
    complete = all((out / f"{c}.npz").is_file() and c in clips for c in want)
    atomic_write_json(
        out / META,
        {
            "schema": SCHEMA, "complete": complete, "stride": stride, "imgsz": imgsz, "conf": conf, "person_conf": PERSON_CONF,
            "detector": {"sha256": sha256, "sha8": sha8(sha256), "weights": str(weights)},
            "expected": want, "clips": clips, "code_sha": code_sha, "updated": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        },
    )  # fmt: skip
    status["complete"] = complete
    status["wall_seconds"] = round(time.time() - t_start, 1)
    return status
