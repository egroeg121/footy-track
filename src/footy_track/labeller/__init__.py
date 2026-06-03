"""SAM3 video labelling tools.

A small UI + backend that lets a human mark objects (players, ball, referee, ...)
on the first frame of a short clip and uses ``SAM3VideoPredictor`` to propagate
those segmentations through the whole clip, producing ``FrameDetections`` JSON.
"""

from footy_track.labeller.video_utils import LabelledObject, Sam3VideoLabeller

__all__ = ["LabelledObject", "Sam3VideoLabeller"]
