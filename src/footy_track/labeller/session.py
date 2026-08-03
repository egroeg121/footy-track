"""Session state + JSONL-sidecar persistence for the labeller server.

``Session`` holds the single authoritative per-frame timeline; the JSONL
sidecar flush/restore format is specified in ``docs/labeller_requirements.md``
§1–2. ``server.py`` is the composition root and config surface: the GT-marks
directory is resolved through ``server._GT_MARKS_DIR`` at call time (tests
monkeypatch it there), never captured at import time.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path

import cv2

from footy_track.labeller.constants import (
    KNOWN_BOX_LABELS,
    NO_BALL_TAG,
    NOT_BROADCAST_TAG,
    PROV_LABELLER,
    PROVENANCE_TAGS,
)
from footy_track.labeller.video_utils import BackgroundLabeller, LabelledObject
from footy_track.schema import ObjectDetection


def _gt_marks_dir() -> Path:
    """Resolve the GT-marks dir through the server module (config surface)."""
    from footy_track.labeller import server  # noqa: PLC0415 — avoid import cycle

    return server._GT_MARKS_DIR


class Session:
    """Single-session server state with one authoritative per-frame timeline.

    ``timeline[i]`` is the list of boxes (ObjectDetection, normalized) for frame
    i, or None if that frame has never been populated. Every actor — YOLO,
    VitTrack, the user — writes here. Box provenance lives in
    ObjectDetection.model (PROV_*). Labeller boxes are ground truth and survive
    auto re-propagation.
    """

    def __init__(self) -> None:
        self.bg = BackgroundLabeller()
        self.video_path: Path | None = None
        self.fps: float = 25.0
        self.total_frames: int = 0
        self.width: int = 0
        self.height: int = 0
        self.timeline: list[list[ObjectDetection] | None] = []
        self._tl_lock = threading.Lock()
        # no-ball frame set: frames explicitly marked as no-ball-visible
        self.no_ball_frames: set[int] = set()
        self.not_broadcast_frames: set[int] = set()
        # debounced JSONL flush
        self._flush_timer: threading.Timer | None = None
        self._flush_lock = threading.Lock()

    def load(self, video_path: str) -> dict:
        self.bg.pause()
        self.bg = BackgroundLabeller()
        # Flush any pending edits for the previous clip before wiping state.
        self._do_flush()
        with self._flush_lock:
            if self._flush_timer:
                self._flush_timer.cancel()
                self._flush_timer = None
        self.video_path = Path(video_path).expanduser()
        if not self.video_path.exists():
            raise FileNotFoundError(self.video_path)
        cap = cv2.VideoCapture(str(self.video_path))
        try:
            self.fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
            self.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        finally:
            cap.release()
        with self._tl_lock:
            self.timeline = [None] * self.total_frames
        self.no_ball_frames = set()
        self.not_broadcast_frames = set()
        self._load_existing_marks()
        return {
            "fps": self.fps,
            "total_frames": self.total_frames,
            "width": self.width,
            "height": self.height,
        }

    def _load_existing_marks(self) -> None:
        """Populate timeline and no_ball/not_broadcast sets from existing JSONL sidecar."""
        if self.video_path is None:
            return
        jsonl_path = _gt_marks_dir() / f"{self.video_path.stem}.jsonl"
        if not jsonl_path.exists():
            return
        with jsonl_path.open() as f:
            for raw_line in f:
                stripped = raw_line.strip()
                if not stripped:
                    continue
                try:
                    d = json.loads(stripped)
                except json.JSONDecodeError:
                    continue
                idx = int(d.get("frame_index", -1))
                if idx < 0 or idx >= self.total_frames:
                    continue
                tags = d.get("tags") or []
                if NOT_BROADCAST_TAG in tags:
                    self.not_broadcast_frames.add(idx)
                    continue
                if NO_BALL_TAG in tags:
                    self.no_ball_frames.add(idx)
                    continue
                bbox = d.get("bbox")
                if bbox is None:
                    continue
                self._restore_box_line(idx, bbox, tags)

    def _restore_box_line(self, idx: int, bbox: dict | list, tags: list[str]) -> None:
        """Rebuild one sidecar box line into the timeline, provenance intact."""
        if isinstance(bbox, dict):
            x, y, w, h = (
                float(bbox["x"]),
                float(bbox["y"]),
                float(bbox["w"]),
                float(bbox["h"]),
            )
        else:
            x, y, w, h = (float(v) for v in bbox)
        # Label is the first tag that is a known class (flush writes [label, model]).
        label = next((t for t in tags if t in KNOWN_BOX_LABELS), "in_play_ball")
        # Restore the saved provenance rather than promoting every box to
        # labeller GT — otherwise machine boxes (vittrack/yolo/sam3) become
        # "hand marks" after a reload and block re-propagation via the
        # GT-authoritative merge rule.
        provenance = next((t for t in tags if t in PROVENANCE_TAGS), PROV_LABELLER)
        box = ObjectDetection(
            label=label,
            confidence=1.0 if provenance == PROV_LABELLER else 0.5,
            x=x,
            y=y,
            w=w,
            h=h,
            model=provenance,
        )
        with self._tl_lock:
            if self.timeline[idx] is None:
                self.timeline[idx] = []
            self.timeline[idx].append(box)

    # --- timeline access -------------------------------------------------

    def get_frame(self, idx: int) -> list[ObjectDetection]:
        with self._tl_lock:
            if 0 <= idx < len(self.timeline) and self.timeline[idx] is not None:
                return list(self.timeline[idx])
            return []

    def set_frame(self, idx: int, boxes: list[ObjectDetection]) -> None:
        """Overwrite a frame entirely (used by user edits / autodetect)."""
        with self._tl_lock:
            if 0 <= idx < len(self.timeline):
                self.timeline[idx] = list(boxes)

    def merge_propagated(self, idx: int, boxes: list[ObjectDetection]) -> bool:
        """Write propagated (vittrack/yolo) boxes, KEEPING any labeller ground truth.

        If the frame already has GT (labeller) boxes, propagated output is
        ignored entirely — GT is authoritative and should never be replaced or
        augmented. If the frame has no GT, write the propagated boxes.

        Returns True when GT was kept (i.e. the propagated boxes were discarded),
        so callers can surface which frames a run did not touch.
        """
        with self._tl_lock:
            if not (0 <= idx < len(self.timeline)):
                return False
            existing = self.timeline[idx] or []
            kept = [b for b in existing if b.model == PROV_LABELLER]
            if kept:
                # GT boxes exist — do not touch this frame at all.
                return True
            self.timeline[idx] = list(boxes)
            return False

    def seed_objects(self, idx: int) -> list[LabelledObject]:
        """Frame idx's boxes as LabelledObjects (abs xyxy) to seed propagation."""
        objs: list[LabelledObject] = []
        for b in self.get_frame(idx):
            objs.append(
                LabelledObject(
                    label=b.label,
                    bbox_xyxy_abs=(
                        b.x * self.width,
                        b.y * self.height,
                        (b.x + b.w) * self.width,
                        (b.y + b.h) * self.height,
                    ),
                )
            )
        return objs

    def frame_jpeg(self, idx: int) -> bytes | None:
        if self.video_path is None:
            return None
        cap = cv2.VideoCapture(str(self.video_path))
        try:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, frame = cap.read()
        finally:
            cap.release()
        if not ok:
            return None
        ok, buf = cv2.imencode(".jpg", frame)
        return buf.tobytes() if ok else None

    # --- JSONL sidecar flush (debounced, 2 s) ----------------------------

    def schedule_flush(self) -> None:
        with self._flush_lock:
            if self._flush_timer:
                self._flush_timer.cancel()
            self._flush_timer = threading.Timer(2.0, self._do_flush)
            self._flush_timer.daemon = True
            self._flush_timer.start()

    def _do_flush(self) -> None:
        if self.video_path is None:
            return
        out_path = _gt_marks_dir() / f"{self.video_path.stem}.jsonl"
        try:
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with self._tl_lock:
                timeline_snapshot = list(self.timeline)
            no_ball_snapshot = set(self.no_ball_frames)
            not_broadcast_snapshot = set(self.not_broadcast_frames)
            lines: list[str] = []
            for idx, boxes in enumerate(timeline_snapshot):
                if idx in not_broadcast_snapshot:
                    lines.append(_skip_marker_line(idx, NOT_BROADCAST_TAG))
                    continue
                if idx in no_ball_snapshot:
                    lines.append(_skip_marker_line(idx, NO_BALL_TAG))
                    continue
                if boxes is None:
                    continue
                lines.extend(_box_line(idx, b) for b in boxes)
            out_path.write_text("\n".join(lines) + ("\n" if lines else ""))
        except Exception as exc:  # noqa: BLE001 — flush must never kill the server
            print(f"[flush] failed to write {out_path}: {exc}", flush=True)


def _skip_marker_line(idx: int, tag: str) -> str:
    return json.dumps({"frame_index": idx, "bbox": None, "center": None, "tags": [tag]})


def _box_line(idx: int, b: ObjectDetection) -> str:
    return json.dumps(
        {
            "frame_index": idx,
            "bbox": {"x": b.x, "y": b.y, "w": b.w, "h": b.h},
            "center": {"x": b.x + b.w / 2, "y": b.y + b.h / 2},
            "tags": [b.label, b.model],
        }
    )


# ---------------------------------------------------------------------------
# Serialization helpers (normalized boxes over the wire; pixels server-side)
# ---------------------------------------------------------------------------


def boxes_from_payload(items: list[dict], provenance: str) -> list[ObjectDetection]:
    """Client boxes (normalized xywh + label) -> ObjectDetection with provenance.

    ``provenance`` is the fallback for items without a ``model`` field. Items
    that carry one keep it — so saving a frame promotes only the boxes the user
    actually touched (client stamps those "labeller"); untouched machine boxes
    keep their vittrack/yolo/sam3 tag.
    """
    return [
        ObjectDetection(
            label=it["label"],
            confidence=float(it.get("conf", 1.0)),
            x=max(0.0, min(1.0, it["x"])),
            y=max(0.0, min(1.0, it["y"])),
            w=max(0.0, min(1.0, it["w"])),
            h=max(0.0, min(1.0, it["h"])),
            model=it.get("model") or provenance,
        )
        for it in items
    ]


def boxes_payload(boxes: list[ObjectDetection]) -> list[dict]:
    """ObjectDetection list -> normalized box dicts for the client (with source)."""
    return [
        {
            "label": b.label,
            "x": b.x,
            "y": b.y,
            "w": b.w,
            "h": b.h,
            "conf": b.confidence,
            "source": b.model,
        }
        for b in boxes
    ]
