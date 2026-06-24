"""Backend for the video labeller (VitTrack SOT tracker).

Given a video and a set of bounding boxes drawn on a seed frame (each tagged
with a class label), drive ``VitTrackSOT`` to propagate those boxes through
every subsequent frame of the clip and return one
:class:`~footy_track.schema.FrameDetections` per frame.

Each object gets its own independent ``VitTrackSOT`` instance. Confidence drops
below 0.5 trigger an anomaly handback so the user can correct bad tracks early.

The ``Sam3VideoLabeller`` class is kept for reference but is no longer wired
into ``BackgroundLabeller`` — VitTrack is the active backend.
"""

from __future__ import annotations

import os
import subprocess
import tempfile
import threading
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path

# Persist the torch.compile (Inductor) cache so SAM3's first-run "compiling" cost
# is paid ONCE, not every session. torch's default cache lives in $TMPDIR, which
# macOS periodically purges — forcing a recompile. Pin it to a stable location
# under the user's home. MUST be set before `import torch` so Inductor picks it
# up on init. Override with TORCHINDUCTOR_CACHE_DIR in the environment.
os.environ.setdefault(
    "TORCHINDUCTOR_CACHE_DIR",
    str(Path.home() / ".cache" / "footy_torch_inductor"),
)

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from footy_track.detectors.utils import (  # noqa: E402
    _available_device,
    mask_poly_to_norm_xywh,
)
from footy_track.schema import FrameDetections, ObjectDetection  # noqa: E402
from footy_track.utils import get_project_root  # noqa: E402

MODEL_TAG = "sam3_video"
MODEL_TAG_VITTRACK = "vittrack"

# Confidence threshold for VitTrack anomaly handback — below this, the user
# is asked to correct the box rather than trusting the tracker.
_VITTRACK_HANDBACK_SCORE = 0.5


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
    """Resolve the default SAM3 checkpoint.

    Prefers the base ``sam3.pt``. NOTE: ``sam3.1_multiplex.pt`` was found to
    return whole-frame masks for every seeded object on this footage (verified
    with both box and point prompts), so it is deliberately NOT the default.
    """
    icloud_base = (
        Path.home()
        / "Library"
        / "Mobile Documents"
        / "com~apple~CloudDocs"
        / "footy_data"
        / "model_saves"
        / "sam3"
        / "sam3.pt"
    )
    candidates = [
        icloud_base,
        Path.home() / "Downloads" / "sam3.pt",
        get_project_root() / "model_saves" / "sam3" / "sam3.pt",
        Path.home()
        / "code"
        / "footy"
        / "footy_data"
        / "model_saves"
        / "sam3"
        / "sam3.pt",
    ]
    for p in candidates:
        if p.exists():
            return str(p)
    return "sam3.pt"  # fall back to Ultralytics auto-download


_warmup_done = threading.Event()
_warmup_lock = threading.Lock()

# Module-level cache of a single SAM3VideoPredictor, reused across runs so the
# model stays hot (loaded + JIT-warm) through pause/restart instead of being
# rebuilt — and recompiled — every time. Keyed by the config that affects the
# built predictor. Guarded by a lock since the worker runs in a daemon thread.
_predictor_cache: dict = {"key": None, "predictor": None}
_predictor_lock = threading.Lock()


def get_cached_predictor(model_uri: str, imgsz: int, device: str, conf: float):
    """Return a hot SAM3VideoPredictor, rebuilding only if the config changed.

    SAM3VideoPredictor is stateful across frames, so callers MUST re-seed it
    with ``set_prompts(...)`` before each streaming pass (which the labeller
    already does). Reusing the instance keeps the weights + compiled kernels
    resident, eliminating the "Compiling model…" wait on every restart.
    """
    from ultralytics.models.sam import SAM3VideoPredictor  # noqa: PLC0415

    key = (str(model_uri), int(imgsz), str(device), round(float(conf), 4))
    with _predictor_lock:
        if _predictor_cache["key"] == key and _predictor_cache["predictor"] is not None:
            predictor = _predictor_cache["predictor"]
            # Reset video-tracking state so the previous run's tracks don't bleed
            # into this one (the predictor is stateful across stream() calls).
            import contextlib  # noqa: PLC0415

            for reset_attr in (
                "clear_all_points_in_video",
                "_reset_tracking_results",
                "reset_image",
            ):
                fn = getattr(predictor, reset_attr, None)
                if callable(fn):
                    # Best-effort reset; set_prompts re-seeds regardless.
                    with contextlib.suppress(Exception):
                        fn()
            return predictor
        overrides = {
            "conf": conf,
            "task": "segment",
            "mode": "predict",
            "model": model_uri,
            "imgsz": imgsz,
            "verbose": False,
            "device": device,
            "save": False,
        }
        predictor = SAM3VideoPredictor(overrides=overrides)
        _predictor_cache["key"] = key
        _predictor_cache["predictor"] = predictor
        return predictor


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
        # Reuse a cached predictor so the model stays hot across pause/restart;
        # rebuilt only when the config changes. Callers re-seed via set_prompts.
        return get_cached_predictor(
            self.model_uri, self.imgsz, self.device, self.min_confidence
        )

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
                # The seed frame is GROUND TRUTH: emit the user's marked boxes
                # verbatim instead of SAM3's re-segmentation (which can drop a
                # tiny ball or mangle a corrected box). SAM3's output is only
                # trusted for the propagated frames (offset >= 1).
                if offset == 0:
                    yield self._seed_frame_detections(abs_idx)
                else:
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

    def _seed_frame_detections(self, frame_idx: int) -> FrameDetections:
        """Emit the user's marked objects verbatim as the seed frame's output.

        The frame the user seeded/corrected is ground truth — we keep their exact
        boxes rather than SAM3's re-segmentation of them.
        """
        uri = self.video_path.parent / f"{self.video_path.stem}_frame_{frame_idx:06d}"
        detections: list[ObjectDetection] = []
        for o in self.objects:
            if o.bbox_xyxy_abs is None:
                continue
            x1, y1, x2, y2 = o.bbox_xyxy_abs
            detections.append(
                ObjectDetection(
                    label=o.label,
                    confidence=1.0,
                    x=max(0.0, x1 / self.width),
                    y=max(0.0, y1 / self.height),
                    w=max(0.0, (x2 - x1) / self.width),
                    h=max(0.0, (y2 - y1) / self.height),
                    model=MODEL_TAG,
                )
            )
        return FrameDetections(
            uri=uri, width=self.width, height=self.height, detections=detections
        )

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


class VitTrackVideoLabeller:
    """Propagate seed boxes through a clip using per-object ``VitTrackSOT`` instances.

    Each :class:`LabelledObject` gets its own tracker. Tracking runs frame-by-
    frame from ``start_frame`` onward. A confidence drop below
    ``_VITTRACK_HANDBACK_SCORE`` on any object is surfaced via the yielded
    ``FrameDetections`` confidence field so ``BackgroundLabeller`` can trigger
    an anomaly handback.
    """

    def __init__(
        self,
        video_path: str | Path,
        objects: list[LabelledObject],
        min_confidence: float = 0.25,
        **_kwargs,  # absorb model_uri / imgsz for API compat with Sam3VideoLabeller
    ) -> None:
        if not objects:
            raise ValueError("At least one labelled object is required.")
        self.video_path = Path(video_path)
        if not self.video_path.exists():
            raise FileNotFoundError(self.video_path)
        self.objects = objects
        self.min_confidence = min_confidence
        self.width, self.height = video_dimensions(self.video_path)

    def _total_frames(self) -> int:
        cap = cv2.VideoCapture(str(self.video_path))
        try:
            return int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        finally:
            cap.release()

    def _seed_frame_detections(self, frame_idx: int) -> FrameDetections:
        """Emit the user's marked objects verbatim as the seed frame's output."""
        uri = self.video_path.parent / f"{self.video_path.stem}_frame_{frame_idx:06d}"
        detections: list[ObjectDetection] = []
        for o in self.objects:
            if o.bbox_xyxy_abs is None:
                continue
            x1, y1, x2, y2 = o.bbox_xyxy_abs
            detections.append(
                ObjectDetection(
                    label=o.label,
                    confidence=1.0,
                    x=max(0.0, x1 / self.width),
                    y=max(0.0, y1 / self.height),
                    w=max(0.0, (x2 - x1) / self.width),
                    h=max(0.0, (y2 - y1) / self.height),
                    model=MODEL_TAG_VITTRACK,
                )
            )
        return FrameDetections(
            uri=uri, width=self.width, height=self.height, detections=detections
        )

    def _obj_to_norm_bbox(
        self, o: LabelledObject
    ) -> tuple[float, float, float, float] | None:
        """Convert a LabelledObject's absolute xyxy bbox to normalized xywh."""
        if o.bbox_xyxy_abs is None:
            return None
        x1, y1, x2, y2 = o.bbox_xyxy_abs
        return (
            x1 / self.width,
            y1 / self.height,
            (x2 - x1) / self.width,
            (y2 - y1) / self.height,
        )

    def _detection_from_bbox(
        self,
        label: str,
        bbox: tuple[float, float, float, float],
        score: float,
    ) -> ObjectDetection:
        nx, ny, nw, nh = bbox
        return ObjectDetection(
            label=label,
            confidence=max(score, 0.0),
            x=max(0.0, nx),
            y=max(0.0, ny),
            w=max(0.0, nw),
            h=max(0.0, nh),
            model=MODEL_TAG_VITTRACK,
        )

    def iter_frames_from(
        self,
        start_frame: int = 0,
        stop_event: threading.Event | None = None,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> Iterator[FrameDetections]:
        """Propagate from ``start_frame`` onward, yielding one frame at a time.

        Yields the seed frame first (user boxes verbatim), then tracks forward.
        Confidence drops below ``_VITTRACK_HANDBACK_SCORE`` are encoded as low
        confidence in the yielded detection so ``BackgroundLabeller`` can detect
        and trigger an anomaly handback.
        """
        from footy_track.ball_trackers.sot_vittrack import VitTrackSOT  # noqa: PLC0415

        total = self._total_frames()
        trackers: list[VitTrackSOT] = [VitTrackSOT() for _ in self.objects]
        seed_bboxes: list[tuple[float, float, float, float] | None] = [
            self._obj_to_norm_bbox(o) for o in self.objects
        ]

        yield self._seed_frame_detections(start_frame)
        if progress_callback is not None:
            progress_callback(start_frame + 1, total)
        if stop_event is not None and stop_event.is_set():
            return

        cap = cv2.VideoCapture(str(self.video_path))
        try:
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            ok, seed_bgr = cap.read()
            if ok and seed_bgr is not None:
                seed_rgb = cv2.cvtColor(seed_bgr, cv2.COLOR_BGR2RGB)
                for i, tracker in enumerate(trackers):
                    tracker.reset()
                    tracker.track(seed_bboxes[i], seed_rgb)

            for abs_idx in range(start_frame + 1, total):
                ok, frame_bgr = cap.read()
                if not ok:
                    break
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                uri = (
                    self.video_path.parent
                    / f"{self.video_path.stem}_frame_{abs_idx:06d}"
                )
                detections: list[ObjectDetection] = []

                for i, (tracker, obj) in enumerate(
                    zip(trackers, self.objects, strict=True)
                ):
                    prev_bbox = seed_bboxes[i]
                    new_bbox = tracker.track(prev_bbox, frame_rgb)
                    score = tracker.last_score
                    if new_bbox is not None:
                        seed_bboxes[i] = new_bbox
                        detections.append(
                            self._detection_from_bbox(obj.label, new_bbox, score)
                        )
                    elif prev_bbox is not None:
                        detections.append(
                            self._detection_from_bbox(obj.label, prev_bbox, score)
                        )

                yield FrameDetections(
                    uri=uri, width=self.width, height=self.height, detections=detections
                )
                if progress_callback is not None:
                    progress_callback(abs_idx + 1, total)
                if stop_event is not None and stop_event.is_set():
                    return
        finally:
            cap.release()


class BackgroundLabeller:
    """Run :class:`VitTrackVideoLabeller` propagation in a daemon thread.

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
        # Anomaly auto-stop: when a track box jumps/changes implausibly between
        # frames (e.g. the ball snapping to the penalty spot), stop so the user
        # can correct. anomaly_frame is the index where it was detected, or None.
        self.anomaly_frame: int | None = None
        self.anomaly_reason: str | None = None
        self.anomaly_detection: bool = True

    def submit(
        self,
        video_path: str | Path,
        objects: list[LabelledObject],
        model_uri: str | None,
        min_confidence: float,
        start_frame: int = 0,
        imgsz: int = 512,
    ) -> None:
        """Stop any running job, then start propagation from ``start_frame``."""
        self.pause()
        labeller = VitTrackVideoLabeller(
            video_path=video_path,
            objects=objects,
            min_confidence=min_confidence,
        )
        total = labeller._total_frames()
        with self._lock:
            if len(self.frames) != total:
                # First run (or video changed): allocate the full timeline.
                self.frames = [None] * total
            self.error = None
            self.anomaly_frame = None
            self.anomaly_reason = None
            # Absolute progress: a restart at frame N shows N/total.
            self.progress = (start_frame, total)
            self.running = True

        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._worker,
            args=(labeller, start_frame),
            daemon=True,
            name="vittrack-bg-labeller",
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

    def _worker(self, labeller: VitTrackVideoLabeller, start_frame: int) -> None:
        try:
            prev_fd: FrameDetections | None = None
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

                # Anomaly auto-stop: if a track box did something implausible vs
                # the previous frame, record the frame + reason and stop so the
                # user can correct it (don't let bad tracks propagate further).
                reason = (
                    _track_anomaly_reason(prev_fd, fd)
                    if (self.anomaly_detection and prev_fd is not None)
                    else None
                )
                # Also hand back on confidence drop (VitTrack-specific).
                if reason is None and self.anomaly_detection:
                    low = [
                        d
                        for d in fd.detections
                        if d.confidence < _VITTRACK_HANDBACK_SCORE
                    ]
                    if low:
                        reason = (
                            f"VitTrack confidence dropped to "
                            f"{low[0].confidence:.2f} for '{low[0].label}' "
                            f"(threshold {_VITTRACK_HANDBACK_SCORE})"
                        )
                if reason is not None:
                    with self._lock:
                        self.anomaly_frame = idx
                        self.anomaly_reason = reason
                    self._stop_event.set()
                    break
                prev_fd = fd
        except Exception as exc:  # noqa: BLE001
            with self._lock:
                self.error = exc
        finally:
            self.running = False

    def _on_progress(self, done: int, total: int) -> None:
        with self._lock:
            self.progress = (done, total)


def _track_anomaly_reason(
    prev: FrameDetections,
    cur: FrameDetections,
    jump_frac: float = 0.40,
    area_ratio: float = 8.0,
) -> str | None:
    """Heuristic: did any tracked box move/resize implausibly between frames?

    For each box in ``cur`` we find the nearest same-label box in ``prev`` and
    flag an anomaly if the centre jumped more than ``jump_frac`` of the frame
    diagonal, or the area changed by more than ``area_ratio``x. Catches the
    classic SAM3 failure where a lost ball track snaps onto a distant marking.
    Coordinates are normalized [0,1], so thresholds are resolution-independent.

    Returns a human-readable reason string, or ``None`` if no anomaly.
    """

    def _center(d) -> tuple[float, float]:
        return (d.x + d.w / 2.0, d.y + d.h / 2.0)

    diag = 2.0**0.5  # diagonal of the unit square
    for c in cur.detections:
        same = [p for p in prev.detections if p.label == c.label]
        if not same:
            continue  # a brand-new label this frame isn't an anomaly per se
        cx, cy = _center(c)
        nearest = min(
            same,
            key=lambda p: (_center(p)[0] - cx) ** 2 + (_center(p)[1] - cy) ** 2,
        )
        px, py = _center(nearest)
        dist = ((px - cx) ** 2 + (py - cy) ** 2) ** 0.5
        if dist > jump_frac * diag:
            return (
                f"a '{c.label}' box jumped {dist / diag * 100:.0f}% of the frame "
                f"between frames (threshold {jump_frac * 100:.0f}%)"
            )
        c_area = max(c.w * c.h, 1e-9)
        p_area = max(nearest.w * nearest.h, 1e-9)
        ratio = max(c_area / p_area, p_area / c_area)
        if ratio > area_ratio:
            return (
                f"a '{c.label}' box changed size {ratio:.1f}x between frames "
                f"(threshold {area_ratio:.0f}x)"
            )
    return None


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


def _nms_filter(
    detections: list[ObjectDetection], iou_threshold: float = 0.5
) -> list[ObjectDetection]:
    """Greedy IoU NMS — keep highest-confidence box, drop overlaps above threshold."""
    from footy_track.detectors.utils import calculate_iou  # noqa: PLC0415

    kept: list[ObjectDetection] = []
    for det in sorted(detections, key=lambda d: d.confidence, reverse=True):
        if all(calculate_iou(det, k) <= iou_threshold for k in kept):
            kept.append(det)
    return kept


def yolo_seed_objects(
    video_path: Path,
    model_path: str,
    min_confidence: float,
    orig_w: int,
    orig_h: int,
    iou_threshold: float = 0.5,
    frame_idx: int = 0,
) -> list[LabelledObject]:
    """Run the YOLO detector on a frame and return NMS-filtered seed objects.

    Detections come back normalized; we convert to absolute xyxy pixel coords for
    :class:`LabelledObject`, the seed format SAM3 expects. ``frame_idx`` selects
    which frame to detect on (0 = first frame).
    """
    from footy_track.detectors.ultralytics import (  # noqa: PLC0415
        UltralyticsObjectDetector,
        get_current_best_detector,
    )

    if frame_idx <= 0:
        frame_bgr = extract_first_frame(video_path)
    else:
        cap = cv2.VideoCapture(str(video_path))
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame_bgr = cap.read()
        cap.release()
        if not ok:
            frame_bgr = extract_first_frame(video_path)
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
        tmp_path = Path(f.name)
    cv2.imwrite(str(tmp_path), frame_bgr)
    try:
        detector = get_current_best_detector(min_confidence=min_confidence)
        if model_path:
            detector = UltralyticsObjectDetector(
                model_uri=model_path,
                min_confidence=min_confidence,
                use_model_names=True,
            )
        fd = detector.predict_from_path(tmp_path)
    finally:
        tmp_path.unlink(missing_ok=True)

    filtered = _nms_filter(fd.detections, iou_threshold=iou_threshold)
    objects: list[LabelledObject] = []
    for det in filtered:
        x1 = det.x * orig_w
        y1 = det.y * orig_h
        x2 = (det.x + det.w) * orig_w
        y2 = (det.y + det.h) * orig_h
        objects.append(LabelledObject(label=det.label, bbox_xyxy_abs=(x1, y1, x2, y2)))
    return objects


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
