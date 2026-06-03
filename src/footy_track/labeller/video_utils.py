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
        Path.home() / "code" / "footy" / "footy_data" / "model_saves" / "sam3" / "sam3.pt",
        Path.home() / "Library" / "Mobile Documents" / "com~apple~CloudDocs" / "footy_data" / "model_saves" / "sam3" / "sam3.1_multiplex.pt",
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


def start_warmup_thread(model_uri: str | None = None, imgsz: int = 512) -> threading.Thread:
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
        self.model_uri = str(Path(model_uri).expanduser()) if model_uri else _default_model_uri()
        self.min_confidence = min_confidence
        self.imgsz = imgsz
        self.verbose = verbose

        dev = _available_device()
        self.device = dev.type if isinstance(dev, torch.device) else str(dev)
        self.width, self.height = video_dimensions(self.video_path)

    def _build_predictor(self):
        # Imported lazily so importing this module doesn't pull in heavy ML deps.
        import os  # noqa: PLC0415

        from ultralytics.models.sam import SAM3VideoPredictor  # noqa: PLC0415

        # Key the inductor cache by model stem so sam3 and sam3.1 don't overwrite each other.
        model_stem = Path(self.model_uri).stem
        cache_dir = Path.home() / ".cache" / f"torch_inductor_{model_stem}"
        cache_dir.mkdir(parents=True, exist_ok=True)
        os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(cache_dir)

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

        first_frame_path = self.video_path.parent / f"_tmp_frame0_{self.video_path.stem}.jpg"
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
                    resolved.append(LabelledObject(label=obj.label, bbox_xyxy_abs=best_bbox))
                else:
                    # Fall back to point if text detection found nothing
                    resolved.append(LabelledObject(label=obj.label, point_xy_abs=obj.point_xy_abs))
        finally:
            first_frame_path.unlink(missing_ok=True)

        return resolved

    def iter_frames(
        self,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> Iterator[FrameDetections]:
        """Lazily yield :class:`FrameDetections` per frame as they are produced."""
        predictor = self._build_predictor()

        # Resolve text+point hints into bboxes before seeding the video predictor.
        objects = self._resolve_text_hints()

        # SAM3VideoPredictor handles bboxes and points in separate calls internally.
        # Convert any remaining point-only objects to small bboxes sized relative to
        # the video frame (1.5% of width/height) so they work alongside bbox objects.
        pad_x = self.width * 0.015
        pad_y = self.height * 0.015
        all_bboxes = []
        for o in objects:
            if o.bbox_xyxy_abs is not None:
                all_bboxes.append(list(o.bbox_xyxy_abs))
            elif o.point_xy_abs is not None:
                px, py = o.point_xy_abs
                all_bboxes.append([px - pad_x, py - pad_y, px + pad_x, py + pad_y])

        predictor.set_prompts({"bboxes": all_bboxes})

        total = int(
            cv2.VideoCapture(str(self.video_path)).get(cv2.CAP_PROP_FRAME_COUNT)
        )

        results = predictor(source=str(self.video_path), stream=True)
        for frame_idx, result in enumerate(results):
            yield self._result_to_frame(frame_idx, result)
            if progress_callback is not None:
                progress_callback(frame_idx + 1, total)

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
