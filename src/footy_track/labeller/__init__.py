"""Video labelling tools.

A web UI + backend that lets a human mark objects (players, ball, referee, ...)
on frames of a short clip and propagates those boxes through the clip
(VitTrack SOT; the legacy ``Sam3VideoLabeller`` backend is kept for scripts),
producing JSONL sidecars / ``FrameDetections`` JSON. See ``README.md`` and
``docs/labeller_requirements.md``.
"""

from footy_track.labeller.video_utils import LabelledObject, Sam3VideoLabeller

__all__ = ["LabelledObject", "Sam3VideoLabeller"]
