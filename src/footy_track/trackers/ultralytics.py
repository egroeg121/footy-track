"""Ultralytics ByteTrack/BoT-SORT tracker implementing the ObjectTracker protocol.

Wraps YOLO.track(persist=True) so that detection and tracking happen in a single
model call. The tracker state is kept inside the Ultralytics model between frames;
we just harvest box.id from each result.

Usage
-----
tracker = UltralyticsTracker("yolo11n.pt", tracker="bytetrack")
for frame_path, frame_t in frames:
    tracked = tracker.update_from_path(frame_path, frame_idx, frame_t)
    ...
meta = tracker.finalise()
"""

from __future__ import annotations

from pathlib import Path

import torch
from ultralytics import YOLO

from footy_track.detectors.utils import _available_device, _clamp01
from footy_track.schema import FrameDetections
from footy_track.trackers.base import TrackedDetection, TrackMeta


class UltralyticsTracker:
    """YOLO detection + ByteTrack/BoT-SORT tracking in a single model call.

    Implements the ObjectTracker protocol plus an extra ``update_from_path``
    convenience method that accepts a raw image path (avoiding the need to
    pre-build a ``FrameDetections`` object).

    Parameters
    ----------
    model_uri:
        Path or hub tag for the YOLO weights (e.g. ``"yolo11n.pt"``).
    tracker:
        Ultralytics tracker config — ``"bytetrack.yaml"`` or ``"botsort.yaml"``.
    classes:
        Dict mapping YOLO class index → label name. Defaults to COCO person+ball.
    min_confidence:
        Minimum detection confidence passed to the model.
    iou_threshold:
        NMS IoU threshold passed to the model.
    verbose:
        Whether to print Ultralytics progress output.
    """

    def __init__(
        self,
        model_uri: str = "yolo11n.pt",
        tracker: str = "bytetrack.yaml",
        classes: dict[int, str] | None = None,
        min_confidence: float = 0.3,
        iou_threshold: float = 0.9,
        verbose: bool = False,
    ) -> None:
        dev = _available_device()
        self.device = dev.type if isinstance(dev, torch.device) else str(dev)
        self.model = YOLO(model_uri)
        self.model_uri = model_uri
        self.tracker_cfg = tracker
        self.classes: dict[int, str] = classes or {0: "person", 32: "ball"}
        self._predict_kwargs: dict = {
            "conf": min_confidence,
            "iou": iou_threshold,
            "verbose": verbose,
            "persist": True,  # keeps ByteTrack/BoT-SORT state between calls
            "device": self.device,
            # suppress all Ultralytics auto-saving
            "save": False,
            "save_txt": False,
            "save_crop": False,
        }

        self._frame_counter: int = 0
        # track_id -> (label, start_frame, start_time, last_frame, last_time)
        self._track_registry: dict[int, list] = {}

    # ------------------------------------------------------------------
    # Primary entry point: accepts a raw image path
    # ------------------------------------------------------------------

    @torch.no_grad()
    def update_from_path(
        self,
        image_path: Path,
        frame_t: float,
    ) -> list[TrackedDetection]:
        """Run YOLO track on a single frame image and return TrackedDetections."""
        frame_idx = self._frame_counter
        self._frame_counter += 1

        results = self.model.track(str(image_path), **self._predict_kwargs)
        result = results[0]

        h, w = result.orig_shape[:2]
        return self._result_to_tracked(result, frame_idx, frame_t, int(w), int(h))

    # ------------------------------------------------------------------
    # ObjectTracker protocol — accepts a pre-built FrameDetections
    # ------------------------------------------------------------------

    def update(
        self, frame_detections: FrameDetections, frame_t: float
    ) -> list[TrackedDetection]:
        """Satisfy the ObjectTracker protocol by running from the stored image URI."""
        return self.update_from_path(frame_detections.uri, frame_t)

    def finalise(self) -> list[TrackMeta]:
        """Return TrackMeta for every track seen so far."""
        meta: list[TrackMeta] = []
        for track_id, (label, sf, st, ef, et) in self._track_registry.items():
            meta.append(
                TrackMeta(
                    track_id=track_id,
                    label=label,
                    start_frame=sf,
                    end_frame=ef,
                    start_continuous_time_s=st,
                    end_continuous_time_s=et,
                )
            )
        return meta

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _result_to_tracked(
        self,
        result,
        frame_idx: int,
        frame_t: float,
        width: int,
        height: int,
    ) -> list[TrackedDetection]:
        if getattr(result, "boxes", None) is None or result.boxes.id is None:
            return []

        boxes = result.boxes
        track_ids = boxes.id.int().tolist()
        cls_indices = boxes.cls.int().tolist()
        scores = boxes.conf.tolist()
        xyxyn = boxes.xyxyn.tolist()  # normalized x1,y1,x2,y2

        tracked: list[TrackedDetection] = []
        for track_id, cls_idx, score, (x1, y1, x2, y2) in zip(
            track_ids, cls_indices, scores, xyxyn, strict=False
        ):
            label = self.classes.get(int(cls_idx), str(int(cls_idx)))
            x = _clamp01(float(x1))
            y = _clamp01(float(y1))
            w_n = _clamp01(max(0.0, float(x2) - float(x1)))
            h_n = _clamp01(max(0.0, float(y2) - float(y1)))

            self._update_registry(track_id, label, frame_idx, frame_t)

            tracked.append(
                TrackedDetection(
                    label=label,
                    confidence=float(score),
                    x=x,
                    y=y,
                    w=w_n,
                    h=h_n,
                    model=self.model_uri,
                    track_id=int(track_id),
                    frame_index=frame_idx,
                    continuous_time_s=frame_t,
                )
            )
        return tracked

    def _update_registry(
        self, track_id: int, label: str, frame_idx: int, frame_t: float
    ) -> None:
        if track_id not in self._track_registry:
            self._track_registry[track_id] = [
                label,
                frame_idx,
                frame_t,
                frame_idx,
                frame_t,
            ]
        else:
            entry = self._track_registry[track_id]
            entry[3] = frame_idx  # last_frame
            entry[4] = frame_t  # last_time
