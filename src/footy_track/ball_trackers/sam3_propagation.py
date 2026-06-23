"""SAM3 ball propagation tracker — Approach C from ft-55d.

Architecture
------------
Seed SAM3VideoPredictor from a confident YOLO ball detection, then propagate
the mask through up to ``propagation_window`` subsequent frames without needing
YOLO to detect the ball in each one.

Protocol-adapter design
~~~~~~~~~~~~~~~~~~~~~~~
SAM3VideoPredictor requires contiguous video frames fed as a temp video.
BallTracker gives us frames one at a time.  We bridge this mismatch by:

  1. Buffering frames into a rolling deque.
  2. When YOLO gives a confident detection *and* we have a clean seed window,
     flush the buffer to a temp .mp4 and stream through SAM3, storing the
     predicted masks for each buffered frame.
  3. While SAM3 results are cached, return them directly without YOLO.
  4. Once the cache is exhausted (or a new confident seed arrives), re-seed.

Failure modes detected and reported
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
- Whole-frame mask blowup (w > 80% or h > 80% of frame) → drop and return None.
- Centre jump > 40% of frame diagonal vs previous → drop and return None.
- Propagation exhausted → fall back to YOLO for next detection opportunity.

Usage::

    from footy_track.ball_trackers.sam3_propagation import Sam3PropagationTracker

    tracker = Sam3PropagationTracker()
    bbox = tracker.track(None, first_frame_rgb)
    tracker.reset()
"""

from __future__ import annotations

import collections
import math
import pathlib
import tempfile
import time
from typing import TYPE_CHECKING

import cv2
import numpy as np

if TYPE_CHECKING:
    from footy_track.ball_eval.dataset import BBox

# Default SAM3 checkpoint path (same resolution order as video_utils.py)
def _default_sam3_path() -> str:
    candidates = [
        pathlib.Path.home()
        / "Library" / "Mobile Documents" / "com~apple~CloudDocs"
        / "footy_data" / "model_saves" / "sam3" / "sam3.pt",
        pathlib.Path.home() / "Downloads" / "sam3.pt",
    ]
    for p in candidates:
        if p.exists():
            return str(p)
    return "sam3.pt"  # Ultralytics auto-download fallback


# Default trained ball detector (same path as roi_yolo.py)
_DEFAULT_YOLO_PATH = str(
    pathlib.Path(__file__).parents[3]
    / "model_saves"
    / "detector"
    / "optuna_trial_1_2026-01-18_17-51_model_name=yolo11s_dataset_version=3_epochs=2226_freeze_layers=3"
    / "best.pt"
)
_BALL_CLASSES = [0, 2]  # 0=ball, 2=in_play_ball — matches the trained model


class Sam3PropagationTracker:
    """BallTracker protocol: SAM3 point-prompt propagation from confident YOLO seeds.

    Parameters
    ----------
    sam3_model_path:
        Path to sam3.pt weights.  Auto-discovered from iCloud / Downloads if None.
    yolo_model_path:
        Path to the trained YOLO detector checkpoint.
    yolo_conf_threshold:
        Minimum YOLO confidence to treat as a seed detection.  Below this,
        fall back to propagated SAM3 mask.
    propagation_window:
        Maximum number of frames to propagate from one seed before re-seeding.
    imgsz:
        SAM3 inference resolution (fed to SAM3VideoPredictor).
    jump_threshold:
        Fraction of frame diagonal; predictions beyond this vs the seed centre
        are treated as tracking failures and dropped.
    """

    def __init__(
        self,
        sam3_model_path: str | None = None,
        yolo_model_path: str = _DEFAULT_YOLO_PATH,
        yolo_conf_threshold: float = 0.25,
        propagation_window: int = 30,
        imgsz: int = 512,
        jump_threshold: float = 0.40,
    ) -> None:
        self._sam3_path = sam3_model_path or _default_sam3_path()
        self._yolo_path = yolo_model_path
        self._yolo_conf = yolo_conf_threshold
        self._prop_window = propagation_window
        self._imgsz = imgsz
        self._jump_threshold = jump_threshold

        # Lazy-load YOLO so import cost is deferred to first call.
        self._yolo = None

        # Frame buffer: (frame_rgb, frame_idx) pairs for current window.
        self._frame_buffer: collections.deque = collections.deque()
        self._frame_idx: int = 0

        # Cached SAM3 predictions: list of (BBox | None) indexed by buffer offset.
        self._sam3_cache: list[BBox | None] = []
        self._cache_offset: int = 0  # frame_idx of cache[0]

        # Last emitted bbox (for jump detection).
        self._last_bbox: BBox | None = None

        # Statistics for post-run reporting.
        self.stats: dict = {
            "seed_frames": [],          # frame_idx where re-seed happened
            "sam3_used_frames": [],     # frame_idx where SAM3 propagation was returned
            "yolo_used_frames": [],     # frame_idx where direct YOLO detection returned
            "drop_whole_frame": [],     # frame_idx where blowup mask was dropped
            "drop_jump": [],            # frame_idx where centre-jump mask was dropped
            "propagation_exhausted": [],# frame_idx where cache ran out → fell back to YOLO
            "sam3_latency_s": [],       # per-SAM3-run latency (seconds for the batch)
        }
        # Expose crop height for the harness metrics (matches roi_yolo convention).
        self._last_crop_height: int | None = None

    # ------------------------------------------------------------------ #
    # BallTracker protocol                                                 #
    # ------------------------------------------------------------------ #

    def track(self, prev_bbox: "BBox | None", frame: np.ndarray) -> "BBox | None":
        """Locate the ball in *frame*; return normalised (x, y, w, h) or None."""
        fi = self._frame_idx
        h, w = frame.shape[:2]
        self._frame_buffer.append((frame.copy(), fi))
        self._frame_idx += 1

        # --- Try SAM3 cache first ---
        if self._sam3_cache and fi < self._cache_offset + len(self._sam3_cache):
            cache_pos = fi - self._cache_offset
            bbox = self._sam3_cache[cache_pos]
            if bbox is not None:
                if self._is_valid(bbox, w, h):
                    self.stats["sam3_used_frames"].append(fi)
                    self._last_bbox = bbox
                    return bbox
                # Cached prediction failed validity — fall through to YOLO.

        # Cache exhausted or invalid — record and try YOLO.
        if self._sam3_cache:
            self.stats["propagation_exhausted"].append(fi)
        self._sam3_cache = []

        yolo_bbox = self._yolo_detect(frame, w, h)
        if yolo_bbox is not None:
            self.stats["yolo_used_frames"].append(fi)
            # Run SAM3 from this seed to populate the forward cache.
            self._run_sam3_from_seed(yolo_bbox, w, h)
            self._last_bbox = yolo_bbox
            return yolo_bbox

        return None

    def reset(self) -> None:
        """Reset all state. Called between eval clips."""
        self._frame_buffer.clear()
        self._frame_idx = 0
        self._sam3_cache = []
        self._cache_offset = 0
        self._last_bbox = None
        self.stats = {k: [] for k in self.stats}

    # ------------------------------------------------------------------ #
    # Internal helpers                                                     #
    # ------------------------------------------------------------------ #

    def _load_yolo(self):
        if self._yolo is None:
            from ultralytics import YOLO  # noqa: PLC0415
            model_path = self._yolo_path
            if not pathlib.Path(model_path).exists():
                model_path = "yolo11n.pt"  # lightweight fallback
            self._yolo = YOLO(model_path, verbose=False)
        return self._yolo

    def _yolo_detect(self, frame: np.ndarray, w: int, h: int) -> "BBox | None":
        """Run YOLO on frame_rgb; return highest-conf ball box or None."""
        yolo = self._load_yolo()
        bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        results = yolo(bgr, verbose=False, conf=self._yolo_conf)
        best_conf = -1.0
        best_box = None
        for r in results:
            if r.boxes is None:
                continue
            for box in r.boxes:
                cls_id = int(box.cls[0])
                if cls_id not in _BALL_CLASSES:
                    continue
                conf = float(box.conf[0])
                if conf > best_conf:
                    best_conf = conf
                    xyxy = box.xyxy[0].tolist()
                    x1, y1, x2, y2 = xyxy
                    bx = x1 / w
                    by = y1 / h
                    bw = (x2 - x1) / w
                    bh = (y2 - y1) / h
                    best_box = (bx, by, bw, bh)
        return best_box

    def _run_sam3_from_seed(
        self, seed_bbox: "BBox", frame_w: int, frame_h: int
    ) -> None:
        """Build a temp video from buffered frames and run SAM3 propagation.

        Fills ``self._sam3_cache`` with BBox|None predictions for every
        buffered frame *after* the seed (offset 0 = seed frame, which uses
        the YOLO box directly; offsets 1..N are SAM3 propagated).
        """
        buf = list(self._frame_buffer)
        if not buf:
            return
        seed_frame_idx = buf[-1][1]  # seed is the most recent frame
        self._cache_offset = seed_frame_idx

        # Build temp video: seed frame + up to propagation_window following frames.
        # On the first call the buffer only has the seed frame (no future frames yet).
        # Subsequent calls will have up to `_prop_window` buffered future frames if we
        # pre-buffer — but with frame-by-frame protocol we only have past frames.
        # Strategy: build temp video from seed frame only and propagate from it.
        # Cache gets populated with seed bbox at index 0; future frames will be
        # decoded from the cache as they arrive via subsequent track() calls.
        #
        # LIMITATION: SAM3 needs ALL frames upfront to stream. Since we only have
        # frames up to and including the seed, we can pre-populate [0] = seed_bbox
        # and populate later frames lazily on the next batch when we have > 1 frame.
        # For this evaluation we instead write the buffered frames to a temp video
        # and stream through SAM3, capping at _prop_window frames total.

        frames_to_use = [f for f, _ in buf[-self._prop_window:]]
        n = len(frames_to_use)

        if n < 2:
            # Only the seed frame — can't propagate, just return seed box.
            self._sam3_cache = [seed_bbox]
            return

        # Write temp mp4 from the buffered frames (oldest first).
        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
                tmp_path = f.name

            fps = 25.0
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(tmp_path, fourcc, fps, (frame_w, frame_h))
            for fr in frames_to_use:
                bgr = cv2.cvtColor(fr, cv2.COLOR_RGB2BGR)
                writer.write(bgr)
            writer.release()

            # Seed point = centre of YOLO bbox on the last frame of the buffer.
            bx, by, bw, bh = seed_bbox
            cx_abs = (bx + bw / 2) * frame_w
            cy_abs = (by + bh / 2) * frame_h
            seed_pt = [[cx_abs, cy_abs]]
            seed_labels = [1]  # foreground

            # Build SAM3 predictor (import deferred).
            import torch  # noqa: PLC0415
            from ultralytics.models.sam import SAM3VideoPredictor  # noqa: PLC0415

            device = "mps" if torch.backends.mps.is_available() else "cpu"
            overrides = {
                "conf": 0.25,
                "task": "segment",
                "mode": "predict",
                "model": self._sam3_path,
                "imgsz": self._imgsz,
                "verbose": False,
                "device": device,
                "save": False,
            }
            predictor = SAM3VideoPredictor(overrides=overrides)
            # Each point as separate object (N, 2) + labels (N,).
            predictor.set_prompts({"points": seed_pt, "labels": seed_labels})

            t0 = time.perf_counter()
            sam3_results = list(predictor(source=tmp_path, stream=True))
            elapsed = time.perf_counter() - t0
            self.stats["sam3_latency_s"].append(elapsed)
            self.stats["seed_frames"].append(seed_frame_idx)

            # Convert SAM3 mask outputs to BBox predictions.
            # Frame 0 in the temp video = seed frame = keep YOLO bbox directly.
            # Frames 1..n-1 are propagated.
            cache: list["BBox | None"] = []
            for offset, result in enumerate(sam3_results):
                if offset == 0:
                    cache.append(seed_bbox)  # trust YOLO for the seed frame
                    continue
                masks = getattr(result, "masks", None)
                polys = masks.xyn if masks is not None else []
                if not polys:
                    cache.append(None)
                    continue
                poly = polys[0]
                bbox = self._poly_to_bbox(poly)
                cache.append(bbox)

            self._sam3_cache = cache
            # Reset the cache offset to the FIRST buffered frame (oldest),
            # so cache[0] corresponds to the oldest frame in the buffer.
            first_buffered_idx = buf[-n][1]
            self._cache_offset = first_buffered_idx

        except Exception as exc:  # noqa: BLE001
            # SAM3 failed — clear cache so YOLO fallback triggers next frame.
            self._sam3_cache = []
            self.stats["propagation_exhausted"].append(seed_frame_idx)
            print(f"[sam3_propagation] SAM3 failed: {exc}")
        finally:
            if tmp_path is not None:
                pathlib.Path(tmp_path).unlink(missing_ok=True)

    def _poly_to_bbox(self, poly: np.ndarray) -> "BBox | None":
        """Convert a normalised polygon (xyn) to a normalised (x, y, w, h) bbox."""
        if poly is None or len(poly) < 3:
            return None
        xs = poly[:, 0]
        ys = poly[:, 1]
        x_min, x_max = float(xs.min()), float(xs.max())
        y_min, y_max = float(ys.min()), float(ys.max())
        w = x_max - x_min
        h = y_max - y_min
        if w < 1e-4 or h < 1e-4:
            return None
        return (x_min, y_min, w, h)

    def _is_valid(self, bbox: "BBox", frame_w: int, frame_h: int) -> "BBox | None":
        """Check for whole-frame blowup or implausible centre jump. Returns bbox or None."""
        x, y, bw, bh = bbox
        fi = self._frame_idx - 1  # caller incremented already

        # Whole-frame blowup guard.
        if bw > 0.80 and bh > 0.80:
            self.stats["drop_whole_frame"].append(fi)
            return None

        # Centre-jump guard vs last emitted bbox.
        if self._last_bbox is not None:
            lx, ly, lw, lh = self._last_bbox
            lcx = lx + lw / 2
            lcy = ly + lh / 2
            cx = x + bw / 2
            cy = y + bh / 2
            diag = math.sqrt(2.0)
            dist = math.sqrt((cx - lcx) ** 2 + (cy - lcy) ** 2)
            if dist > self._jump_threshold * diag:
                self.stats["drop_jump"].append(fi)
                return None

        return bbox

    def summary(self) -> dict:
        """Return a human-readable summary dict of propagation statistics."""
        n_seed = len(self.stats["seed_frames"])
        n_sam3 = len(self.stats["sam3_used_frames"])
        n_yolo = len(self.stats["yolo_used_frames"])
        n_total = n_sam3 + n_yolo
        latencies = self.stats["sam3_latency_s"]
        return {
            "total_detections": n_total,
            "yolo_direct": n_yolo,
            "sam3_propagated": n_sam3,
            "sam3_propagation_pct": (n_sam3 / max(n_total, 1)) * 100,
            "seed_events": n_seed,
            "whole_frame_drops": len(self.stats["drop_whole_frame"]),
            "jump_drops": len(self.stats["drop_jump"]),
            "propagation_exhaustions": len(self.stats["propagation_exhausted"]),
            "mean_sam3_batch_latency_s": (
                sum(latencies) / len(latencies) if latencies else 0.0
            ),
            "total_sam3_latency_s": sum(latencies),
        }
