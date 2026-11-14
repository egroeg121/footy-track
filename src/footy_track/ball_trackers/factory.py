"""Choose a detection-based ball tracker by name.

``online`` is the streaming ``DetectionBallTracker`` (the default everywhere);
``bidirectional`` and ``viterbi`` are offline: they buffer the clip's candidates
and decide in ``points()``, so they cannot be used for live output.
"""

from __future__ import annotations

from footy_track.ball_trackers.bidirectional import BidirectionalBallTracker
from footy_track.ball_trackers.detection_tracker import DetectionBallTracker
from footy_track.ball_trackers.viterbi import ViterbiBallTracker

TRACKER_KINDS = ("online", "bidirectional", "viterbi")


def make_ball_tracker(kind: str, frame_size: tuple[int, int], **params):
    """Build the tracker ``kind`` for frames of ``frame_size``; ``params`` override defaults."""
    if kind == "online":
        return DetectionBallTracker(frame_size=frame_size, **params)
    if kind == "bidirectional":
        return BidirectionalBallTracker(frame_size=frame_size, **params)
    if kind == "viterbi":
        return ViterbiBallTracker(frame_size=frame_size, **params)
    raise ValueError(f"unknown ball tracker {kind!r}; expected one of {TRACKER_KINDS}")
