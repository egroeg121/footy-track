"""Backend for the SAM3 video labeller.

Given a video and a set of bounding boxes drawn on frame 0 (each tagged with a
class label), drive ``SAM3VideoPredictor`` to propagate those boxes through every
frame of the clip and return one :class:`~footy_track.schema.FrameDetections`
per frame.

The Ultralytics SAM3 video predictor seeds one tracked object per frame-0 box
(in the order given) and propagates them. We therefore map ``object index ->
class label`` ourselves, since the predictor only preserves object *order*, not
our semantic labels.
"""

from __future__ import annotations

import subprocess
import tempfile
import threading
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch

from footy_track.detectors.utils import _available_device, mask_poly_to_norm_xywh
from footy_track.schema import FrameDetections, ObjectDetection
from footy_track.utils import get_project_root

MODEL_TAG = "sam3_video"


@dataclass
class LabelledObject:
    """A single object the user marked on frame 0.

    Exactly one of ``bbox_xyxy_abs`` or ``point_xy_abs`` must be set.
    ``text_hint`` is optional — when set alongside ``point_xy_abs``, SAM3Semantic
    runs the text on frame 0 and the candidate mask closest to the point is used
    as the bbox seed for video propagation instead of the raw point.

    Attributes:
        label: Class name, e.g. ``"player"``, ``"ball"``, ``"referee"``.
        bbox_xyxy_abs: Bounding box in absolute pixel coords ``(x1, y1, x2, y2)``.
        point_xy_abs: Single foreground point in absolute pixel coords ``(x, y)``.
        text_hint: Optional text description used with SAM3Semantic to find the
            object on frame 0 (e.g. ``"soccer ball"``).
    """

    label: str
    bbox_xyxy_abs: tuple[float, float, float, float] | None = None
    point_xy_abs: tuple[float, float] | None = None
    text_hint: str | None = None

    def __post_init__(self) -> None:
        if self.bbox_xyxy_abs is None and self.point_xy_abs is None:
            raise ValueError("Either bbox_xyxy_abs or point_xy_abs must be set.")
        if self.bbox_xyxy_abs is not None and self.point_xy_abs is not None:
            raise ValueError("Only one of bbox_xyxy_abs or point_xy_abs may be set.")


def extract_first_frame(video_path: Path) -> np.ndarray:
    """Read and return the first frame of ``video_path`` as a BGR numpy array."""
    cap = cv2.VideoCapture(str(video_path))
    try:
        ok, frame = cap.read()
        if not ok or frame is None:
            raise ValueError(f"Could not read first frame from {video_path}")
        return frame
    finally:
        cap.release()


def video_dimensions(video_path: Path) -> tuple[int, int]:
    """Return ``(width, height)`` in pixels for ``video_path``."""
    cap = cv2.VideoCapture(str(video_path))
    try:
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        return width, height
    finally:
        cap.release()


def _default_model_uri() -> str:
    """Find sam3.pt — checks project-local path then shared footy_data store."""
    candidates = [
        get_project_root() / "model_saves" / "sam3" / "sam3.pt",
        Path.home()
        / "code"
        / "footy"
        / "footy_data"
        / "model_saves"
        / "sam3"
        / "sam3.pt",
        Path.home()
        / "Library"
        / "Mobile Documents"
        / "com~apple~CloudDocs"
        / "footy_data"
        / "model_saves"
        / "sam3"
        / "sam3.1_multiplex.pt",
    ]
    for p in candidates:
        if p.exists():
            return str(p)
    return "sam3.pt"  # fall back to Ultralytics auto-download


_warmup_done = threading.Event()
_warmup_lock = threading.Lock()


def warmup_model(model_uri: str | None = None, imgsz: int = 512) -> None:
    """Load SAM3VideoPredictor and run one dummy frame to trigger JIT compilation.

    Safe to call in a background thread. Sets ``_warmup_done`` when finished so
    the UI can show a ready indicator. Subsequent calls are no-ops (guarded by lock).
    """
    with _warmup_lock:
        if _warmup_done.is_set():
            return
        try:
            from ultralytics.models.sam import SAM3VideoPredictor  # noqa: PLC0415

            uri = model_uri or _default_model_uri()
            dev = _available_device()
            device = dev.type if isinstance(dev, torch.device) else str(dev)

            overrides = {
                "conf": 0.25,
                "task": "segment",
                "mode": "predict",
                "model": uri,
                "imgsz": imgsz,
                "verbose": False,
                "device": device,
                "save": False,
            }
            predictor = SAM3VideoPredictor(overrides=overrides)

            # Write a tiny 3-frame dummy video so the predictor has something to run on.
            with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
                tmp_path = f.name
            h, w = 64, 64
            writer = cv2.VideoWriter(
                tmp_path, cv2.VideoWriter_fourcc(*"mp4v"), 10, (w, h)
            )
            for _ in range(3):
                writer.write(np.zeros((h, w, 3), dtype=np.uint8))
            writer.release()

            predictor.set_prompts({"bboxes": [[8, 8, 24, 24]]})
            for _ in predictor(source=tmp_path, stream=True):
                pass

            Path(tmp_path).unlink(missing_ok=True)
        except Exception:  # noqa: BLE001
            pass  # warmup failure is non-fatal; user will just see JIT on first real run
        finally:
            _warmup_done.set()


def start_warmup_thread(
    model_uri: str | None = None, imgsz: int = 512
) -> threading.Thread:
    """Spawn a daemon thread to warm up SAM3 in the background."""
    t = threading.Thread(
        target=warmup_model, args=(model_uri, imgsz), daemon=True, name="sam3-warmup"
    )
    t.start()
    return t


class Sam3VideoLabeller:
    """Propagate frame-0 box prompts through a clip with ``SAM3VideoPredictor``."""

    def __init__(
        self,
        video_path: str | Path,
        objects: list[LabelledObject],
        model_uri: str | None = None,
        min_confidence: float = 0.25,
        imgsz: int = 512,
        verbose: bool = False,
    ) -> None:
        if not objects:
            raise ValueError("At least one labelled object is required.")
        self.video_path = Path(video_path)
        if not self.video_path.exists():
            raise FileNotFoundError(self.video_path)
        self.objects = objects
        self.model_uri = (
            str(Path(model_uri).expanduser()) if model_uri else _default_model_uri()
        )
        self.min_confidence = min_confidence
        self.imgsz = imgsz
        self.verbose = verbose

        dev = _available_device()
        self.device = dev.type if isinstance(dev, torch.device) else str(dev)
        self.width, self.height = video_dimensions(self.video_path)

    def _build_predictor(self):
        # Imported lazily so importing this module doesn't pull in heavy ML deps.
        from ultralytics.models.sam import SAM3VideoPredictor  # noqa: PLC0415

        overrides = {
            "conf": self.min_confidence,
            "task": "segment",
            "mode": "predict",
            "model": self.model_uri,
            "imgsz": self.imgsz,
            "verbose": self.verbose,
            "device": self.device,
            "save": False,
        }
        return SAM3VideoPredictor(overrides=overrides)

    @torch.no_grad()
    def run(
        self,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> list[FrameDetections]:
        """Run propagation over the whole clip.

        Args:
            progress_callback: Optional ``fn(frame_index, total_frames)`` invoked
                after each frame is processed, for UI progress bars.

        Returns:
            One :class:`FrameDetections` per video frame, in order.
        """
        return list(self.iter_frames(progress_callback=progress_callback))

    def _resolve_text_hints(self) -> list[LabelledObject]:
        """For any object with a text_hint + point, use SAM3Semantic on frame 0
        to find the mask closest to the point and return it as a bbox object.
        Objects without text_hint are returned unchanged.
        Returns immediately if no objects have a text_hint set."""
        needs_text = [o for o in self.objects if o.text_hint and o.point_xy_abs]
        if not needs_text:
            return list(self.objects)
        resolved = []

        from ultralytics.models.sam import SAM3SemanticPredictor  # noqa: PLC0415

        overrides = {
            "conf": 0.1,
            "task": "segment",
            "mode": "predict",
            "model": self.model_uri,
            "imgsz": self.imgsz,
            "verbose": False,
            "device": self.device,
            "save": False,
            "agnostic_nms": True,
            "iou": 0.7,
        }
        sem_predictor = SAM3SemanticPredictor(overrides=overrides)

        first_frame_path = (
            self.video_path.parent / f"_tmp_frame0_{self.video_path.stem}.jpg"
        )
        frame0 = extract_first_frame(self.video_path)
        cv2.imwrite(str(first_frame_path), frame0)
        h0, w0 = frame0.shape[:2]

        try:
            for obj in self.objects:
                if not (obj.text_hint and obj.point_xy_abs):
                    resolved.append(obj)
                    continue

                sem_predictor.set_image(str(first_frame_path))
                results = sem_predictor(text=[obj.text_hint])

                best_bbox = None
                best_dist = float("inf")
                px, py = obj.point_xy_abs

                if results and getattr(results[0], "boxes", None) is not None:
                    boxes = results[0].boxes
                    for box in boxes.xyxy.tolist():
                        x1, y1, x2, y2 = box
                        # Denormalise if needed (xyxy from ultralytics is in pixels)
                        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                        dist = (cx - px) ** 2 + (cy - py) ** 2
                        if dist < best_dist:
                            best_dist = dist
                            best_bbox = (x1, y1, x2, y2)

                if best_bbox is not None:
                    resolved.append(
                        LabelledObject(label=obj.label, bbox_xyxy_abs=best_bbox)
                    )
                else:
                    # Fall back to point if text detection found nothing
                    resolved.append(
                        LabelledObject(label=obj.label, point_xy_abs=obj.point_xy_abs)
                    )
        finally:
            first_frame_path.unlink(missing_ok=True)

        return resolved

    def _build_bboxes(self) -> list[list[float]]:
        """Resolve text hints and convert all objects to seed bboxes in pixels.

        SAM3VideoPredictor handles bboxes and points in separate calls internally,
        so we convert any remaining point-only objects to small bboxes sized
        relative to the video frame (1.5% of width/height).
        """
        objects = self._resolve_text_hints()
        pad_x = self.width * 0.015
        pad_y = self.height * 0.015
        all_bboxes: list[list[float]] = []
        for o in objects:
            if o.bbox_xyxy_abs is not None:
                all_bboxes.append(list(o.bbox_xyxy_abs))
            elif o.point_xy_abs is not None:
                px, py = o.point_xy_abs
                all_bboxes.append([px - pad_x, py - pad_y, px + pad_x, py + pad_y])
        return all_bboxes

    def _total_frames(self) -> int:
        cap = cv2.VideoCapture(str(self.video_path))
        try:
            return int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        finally:
            cap.release()

    def iter_frames(
        self,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> Iterator[FrameDetections]:
        """Lazily yield :class:`FrameDetections` per frame as they are produced."""
        predictor = self._build_predictor()
        predictor.set_prompts({"bboxes": self._build_bboxes()})

        total = self._total_frames()
        results = predictor(source=str(self.video_path), stream=True)
        for frame_idx, result in enumerate(results):
            yield self._result_to_frame(frame_idx, result)
            if progress_callback is not None:
                progress_callback(frame_idx + 1, total)

    @torch.no_grad()
    def iter_frames_from(
        self,
        start_frame: int = 0,
        stop_event: threading.Event | None = None,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> Iterator[FrameDetections]:
        """Propagate from ``start_frame`` onward, yielding one frame at a time.

        SAM3VideoPredictor is stateful and seeds its tracks from the first frame
        it sees, so to restart mid-clip we write frames ``start_frame..end`` to a
        temporary video and stream that with the (corrected) seed bboxes. Yielded
        ``FrameDetections`` carry their absolute ``frame_idx`` so the caller can
        slot them back into the full timeline.

        Args:
            start_frame: Absolute frame index to begin propagation from.
            stop_event: If set during iteration, stops cleanly after the current
                frame (used by :class:`BackgroundLabeller` for pause/restart).
            progress_callback: ``fn(done, total)`` over the *remaining* frames.
        """
        total = self._total_frames()
        if start_frame <= 0:
            # Whole-clip path: reuse the efficient direct-stream iterator.
            for frame_idx, fd in enumerate(self.iter_frames()):
                yield fd
                if progress_callback is not None:
                    progress_callback(frame_idx + 1, total)
                if stop_event is not None and stop_event.is_set():
                    return
            return

        # Write frames [start_frame, total) to a temp clip to re-seed propagation.
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            tmp_path = f.name
        cap = cv2.VideoCapture(str(self.video_path))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        writer = cv2.VideoWriter(
            tmp_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (self.width, self.height)
        )
        try:
            n_written = 0
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                writer.write(frame)
                n_written += 1
            writer.release()
            cap.release()

            predictor = self._build_predictor()
            predictor.set_prompts({"bboxes": self._build_bboxes()})
            results = predictor(source=tmp_path, stream=True)
            for offset, result in enumerate(results):
                abs_idx = start_frame + offset
                yield self._result_to_frame(abs_idx, result)
                if progress_callback is not None:
                    # Report ABSOLUTE position out of the full clip so a restart
                    # at frame N shows N/total, not 0/(total-N).
                    progress_callback(abs_idx + 1, total)
                if stop_event is not None and stop_event.is_set():
                    return
        finally:
            writer.release()
            cap.release()
            Path(tmp_path).unlink(missing_ok=True)

    def _result_to_frame(self, frame_idx: int, result) -> FrameDetections:
        """Convert one Ultralytics ``Results`` to our ``FrameDetections`` schema."""
        uri = self.video_path.parent / f"{self.video_path.stem}_frame_{frame_idx:06d}"

        detections: list[ObjectDetection] = []
        masks = getattr(result, "masks", None)
        polys = masks.xyn if masks is not None else []

        for obj_idx, poly in enumerate(polys):
            box = mask_poly_to_norm_xywh(poly)
            if box is None:
                continue
            x, y, w_n, h_n = box
            label = (
                self.objects[obj_idx].label
                if obj_idx < len(self.objects)
                else "unknown"
            )
            detections.append(
                ObjectDetection(
                    label=label,
                    confidence=1.0,
                    x=x,
                    y=y,
                    w=w_n,
                    h=h_n,
                    model=MODEL_TAG,
                )
            )

        return FrameDetections(
            uri=uri,
            width=self.width,
            height=self.height,
            detections=detections,
        )


class BackgroundLabeller:
    """Run :class:`Sam3VideoLabeller` propagation in a daemon thread.

    Designed for the Streamlit UI: the worker thread fills ``frames`` (indexed by
    absolute frame number) while the UI polls ``progress`` / ``running`` on each
    rerun. Calling :meth:`submit` again — typically after the user corrects boxes
    on a paused frame — cleanly stops the current thread and restarts propagation
    from ``start_frame`` onward, preserving earlier frames.
    """

    def __init__(self) -> None:
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._lock = threading.Lock()
        self.frames: list[FrameDetections | None] = []
        self.progress: tuple[int, int] = (0, 0)  # (done, total) for current job
        self.running: bool = False
        self.error: Exception | None = None
        self.last_completed_frame: int = -1

    def submit(
        self,
        video_path: str | Path,
        objects: list[LabelledObject],
        model_uri: str | None,
        min_confidence: float,
        start_frame: int = 0,
    ) -> None:
        """Stop any running job, then start propagation from ``start_frame``."""
        self.pause()
        labeller = Sam3VideoLabeller(
            video_path=video_path,
            objects=objects,
            model_uri=model_uri,
            min_confidence=min_confidence,
        )
        total = labeller._total_frames()
        with self._lock:
            if len(self.frames) != total:
                # First run (or video changed): allocate the full timeline.
                self.frames = [None] * total
            self.error = None
            # Absolute progress: a restart at frame N shows N/total.
            self.progress = (start_frame, total)
            self.running = True

        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._worker,
            args=(labeller, start_frame),
            daemon=True,
            name="sam3-bg-labeller",
        )
        self._thread.start()

    def pause(self) -> None:
        """Signal the running thread to stop and wait for it to exit."""
        self._stop_event.set()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=10.0)
        self.running = False

    def is_done(self) -> bool:
        """True when no job is running and at least one frame has completed."""
        return not self.running and self.last_completed_frame >= 0

    def completed_frames(self) -> list[FrameDetections]:
        """Return the contiguous run of completed frames from index 0."""
        with self._lock:
            out: list[FrameDetections] = []
            for fd in self.frames:
                if fd is None:
                    break
                out.append(fd)
            return out

    def _worker(self, labeller: Sam3VideoLabeller, start_frame: int) -> None:
        try:
            for fd in labeller.iter_frames_from(
                start_frame=start_frame,
                stop_event=self._stop_event,
                progress_callback=self._on_progress,
            ):
                idx = _frame_index_from_uri(fd, default=start_frame)
                with self._lock:
                    if 0 <= idx < len(self.frames):
                        self.frames[idx] = fd
                    self.last_completed_frame = max(self.last_completed_frame, idx)
        except Exception as exc:  # noqa: BLE001
            with self._lock:
                self.error = exc
        finally:
            self.running = False

    def _on_progress(self, done: int, total: int) -> None:
        with self._lock:
            self.progress = (done, total)


def _frame_index_from_uri(fd: FrameDetections, default: int = 0) -> int:
    """Recover the absolute frame index encoded in a FrameDetections uri stem."""
    stem = Path(fd.uri).name
    marker = "_frame_"
    if marker in stem:
        try:
            return int(stem.rsplit(marker, 1)[1])
        except ValueError:
            return default
    return default


def export_frames_json(frames: list[FrameDetections], out_path: Path) -> Path:
    """Serialise ``frames`` to a single JSON array file via Pydantic."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = "[" + ",".join(f.model_dump_json() for f in frames) + "]"
    out_path.write_text(payload)
    return out_path


def has_ffmpeg() -> bool:
    """Return True if ffmpeg is available (used for optional frame export)."""
    try:
        subprocess.run(
            ["ffmpeg", "-version"],
            check=True,
            capture_output=True,
        )
        return True
    except (OSError, subprocess.CalledProcessError):
        return False
