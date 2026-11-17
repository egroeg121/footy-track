"""Pitch-crop heuristic: find the playing surface and crop to it before detection.

A broadcast frame is mostly crowd, ad boards and sky; the pitch occupies a band
in the middle. Running detection/SAM3 on just that band is cheaper *and* tends to
be more accurate (players fill more of the model's ``imgsz``).

The crop is a **transparent affine transform**: models run on cropped pixels, but
every detection is mapped back to full-frame-normalized coordinates via
:meth:`CropRect.to_full` before it touches the timeline / schema / parquet. Nothing
downstream needs to know a crop happened.

Design choices (see ``docs`` / discussion):
  * **One rectangle per segment, not per frame.** ``split_broadcast_segments``
    already cuts the broadcast into single-camera clips; within one clip we take a
    single crop (the union of the green region over sampled frames). A constant
    crop keeps the tracker's motion model valid and gives an exact, jitter-free
    remap. Per-frame / polygon cropping is deferred to proper pitch segmentation.
  * **Identity fallback.** If green can't be found (replay, close-up, crowd shot)
    we return a full-frame crop rather than risk clipping players.

The boundary detector (:func:`detect_pitch_crop`) is intentionally swappable: a
learned segmentation model can replace it later behind the same :class:`CropRect`
interface with zero downstream change.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

# HSV bounds for grass green (OpenCV hue is 0-180).
_GREEN_LOWER = np.array([30, 40, 40])
_GREEN_UPPER = np.array([90, 255, 255])


@dataclass(frozen=True)
class CropRect:
    """An axis-aligned crop in original-frame pixels + the affine remap it implies.

    ``(x0, y0)`` is the top-left, ``(x1, y1)`` the bottom-right (exclusive), and
    ``W``/``H`` the full original frame size. All remaps go between *crop-normalized*
    coordinates (what a model sees, [0,1] relative to the crop) and *full-normalized*
    coordinates (what the schema stores, [0,1] relative to the original frame).
    """

    x0: int
    y0: int
    x1: int
    y1: int
    W: int
    H: int

    @property
    def cw(self) -> int:
        return self.x1 - self.x0

    @property
    def ch(self) -> int:
        return self.y1 - self.y0

    @property
    def is_identity(self) -> bool:
        """True when the crop is the whole frame, so callers can skip the work."""
        return self.x0 == 0 and self.y0 == 0 and self.x1 == self.W and self.y1 == self.H

    @property
    def kept_fraction(self) -> float:
        """Fraction of the original frame area this crop retains."""
        if self.W == 0 or self.H == 0:
            return 1.0
        return (self.cw * self.ch) / (self.W * self.H)

    def apply(self, frame: np.ndarray) -> np.ndarray:
        """Return the cropped view of ``frame`` (BGR ndarray)."""
        if self.is_identity:
            return frame
        return frame[self.y0 : self.y1, self.x0 : self.x1]

    def to_full(
        self, x: float, y: float, w: float, h: float
    ) -> tuple[float, float, float, float]:
        """Map a crop-normalized top-left box to full-frame-normalized.

        Use on detection OUTPUT (the model saw the crop; the schema wants full).
        """
        if self.is_identity:
            return x, y, w, h
        x_full = (self.x0 + x * self.cw) / self.W
        y_full = (self.y0 + y * self.ch) / self.H
        w_full = w * self.cw / self.W
        h_full = h * self.ch / self.H
        return x_full, y_full, w_full, h_full

    def to_crop(
        self, x: float, y: float, w: float, h: float
    ) -> tuple[float, float, float, float]:
        """Map a full-frame-normalized top-left box to crop-normalized.

        Use on model INPUT (e.g. SAM3 seed boxes drawn on the full frame).
        """
        if self.is_identity:
            return x, y, w, h
        x_crop = (x * self.W - self.x0) / self.cw
        y_crop = (y * self.H - self.y0) / self.ch
        w_crop = w * self.W / self.cw
        h_crop = h * self.H / self.ch
        return x_crop, y_crop, w_crop, h_crop

    def union(self, other: CropRect) -> CropRect:
        """Smallest crop covering both — used to combine per-frame crops safely."""
        return CropRect(
            x0=min(self.x0, other.x0),
            y0=min(self.y0, other.y0),
            x1=max(self.x1, other.x1),
            y1=max(self.y1, other.y1),
            W=self.W,
            H=self.H,
        )

    @classmethod
    def identity(cls, width: int, height: int) -> CropRect:
        return cls(0, 0, width, height, width, height)


def detect_pitch_crop(
    frame_bgr: np.ndarray, height_pad: float = 0.06
) -> CropRect | None:
    """Find the pitch via a green-colour heuristic and return its crop rectangle.

    Steps: HSV green threshold -> morphological close/open -> largest connected
    component (= the contiguous pitch) -> tight bbox -> pad upward off the grass
    line by ``height_pad`` of frame height (keeps far-touchline players whose
    bodies rise above the visible grass).

    Returns ``None`` if no plausible pitch region is found, so the caller can fall
    back to an identity crop.
    """
    h, w = frame_bgr.shape[:2]
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, _GREEN_LOWER, _GREEN_UPPER)

    # Close gaps inside the pitch (players, lines, on-grass logos), then drop
    # isolated green speckle in the crowd.
    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
    k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k_close)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k_open)

    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n <= 1:
        return None
    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    ys, xs = np.where(labels == largest)
    if xs.size == 0:
        return None
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())

    pad = int(h * height_pad)
    cy0 = max(0, y0 - pad)
    cy1 = min(h, y1 + int(h * 0.02))  # tiny pad below the grass
    return CropRect(x0=x0, y0=cy0, x1=x1 + 1, y1=cy1, W=w, H=h)


def compute_segment_crop(
    video_path: str | Path,
    sample_n: int = 15,
    height_pad: float = 0.03,
    top_pct: float = 0.5,
) -> CropRect:
    """Compute one stable crop for a whole (single-camera) segment.

    Samples ``sample_n`` frames evenly across the clip and detects the pitch in
    each. The crop is aggregated per edge:

    * **Top edge (the grass line)** is the ``top_pct`` percentile of the per-frame
      grass lines — i.e. cut to the *typical* grass line, not the highest one. A
      plain union (min) is dominated by a single panned-up frame and barely crops;
      the median (``top_pct=0.5``) removes the crowd in the common case. Frames
      panned higher than this lose a sliver of upper grass (rarely a player, and
      recoverable via the full-frame view).
    * **Sides + bottom** take the union (widest), since the pitch spans the frame
      there and panning left/right shouldn't clip it.

    Larger ``top_pct`` crops more aggressively; smaller is safer. Falls back to an
    identity crop (full frame) if the pitch is never found.
    """
    video_path = Path(video_path)
    cap = cv2.VideoCapture(str(video_path))
    try:
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        idxs = np.linspace(0, total - 1, num=min(sample_n, total)).astype(int).tolist()
        rects: list[CropRect] = []
        for fi in idxs:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(fi))
            ok, frame = cap.read()
            if not ok or frame is None:
                continue
            rect = detect_pitch_crop(frame, height_pad=height_pad)
            if rect is not None:
                rects.append(rect)
    finally:
        cap.release()

    if not rects:
        return CropRect.identity(w, h)

    x0 = min(r.x0 for r in rects)
    x1 = max(r.x1 for r in rects)
    y1 = max(r.y1 for r in rects)
    y0 = int(np.percentile([r.y0 for r in rects], top_pct * 100))
    return CropRect(x0=x0, y0=y0, x1=x1, y1=y1, W=w, H=h)


def write_cropped_video(src: str | Path, dst: str | Path, crop: CropRect) -> Path:
    """Write a copy of ``src`` cropped to ``crop`` — the labeller's working video.

    Producing one cropped video up front lets the whole labeller (frame serving
    *and* the SAM3 video predictor, which streams from a file) operate in crop
    space with no per-frame remap: the timeline is naturally crop-normalized, and
    full-frame coords are only needed at export. Returns ``dst``.
    """
    src, dst = Path(src), Path(dst)
    cap = cv2.VideoCapture(str(src))
    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        writer = cv2.VideoWriter(
            str(dst),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (crop.cw, crop.ch),
        )
        try:
            while True:
                ok, frame = cap.read()
                if not ok or frame is None:
                    break
                writer.write(crop.apply(frame))
        finally:
            writer.release()
    finally:
        cap.release()
    return dst
