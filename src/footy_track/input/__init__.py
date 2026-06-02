"""Input stage: read video sources and emit timestamped frame records.

See ``docs/design/input_live_stream.md`` for the design contract and
``docs/timings.md`` for ``GameTime`` / ``ContinuousTime`` conventions.
"""

from footy_track.input.streams import (
    FileFrameSource,
    FrameRecord,
    FrameSource,
    GameMetadata,
    GameTime,
    LiveStreamFrameSource,
    ReconnectPolicy,
    StreamBackend,
    to_continuous_time,
)

__all__ = [
    "FileFrameSource",
    "FrameRecord",
    "FrameSource",
    "GameMetadata",
    "GameTime",
    "LiveStreamFrameSource",
    "ReconnectPolicy",
    "StreamBackend",
    "to_continuous_time",
]
