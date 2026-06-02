"""Frame sources for the Input stage.

Defines the ``FrameSource`` protocol and two concrete implementations
(``FileFrameSource`` for on-disk videos and ``LiveStreamFrameSource`` for
HLS / RTMP / RTSP streams), plus the ``FrameRecord`` schema and the
``GameTime`` / ``GameMetadata`` time-conversion helpers used at the input
boundary.

This is the v1 scaffolding for the design described in
``docs/design/input_live_stream.md``. The default network backend for
``LiveStreamFrameSource`` (PyAV) is intentionally not implemented yet —
the class exposes a ``backend`` injection point so the reconnect/sentinel
behaviour can be exercised end-to-end with a mock.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import cv2
import numpy as np
from pydantic import BaseModel, ConfigDict, Field


class GameTime(BaseModel):
    """Time as displayed on the referee/broadcast clock.

    Resets at half-time. ``half`` is 1 or 2; ``seconds`` is the offset within
    that half (including stoppage time).
    """

    model_config = ConfigDict(frozen=True)

    half: int = Field(ge=1, le=2)
    seconds: float = Field(ge=0.0)


class GameMetadata(BaseModel):
    """Mapping required to convert ``GameTime`` to ``ContinuousTime``.

    ``half_start_continuous[h]`` is the ``ContinuousTime`` (seconds from
    kickoff of the first half) at which half ``h`` began. Defaults assume a
    clean first half starting at 0 and a second half offset by 45 minutes;
    the real values come from match metadata when known.
    """

    model_config = ConfigDict(frozen=True)

    half_start_continuous: dict[int, float] = Field(
        default_factory=lambda: {1: 0.0, 2: 45 * 60.0}
    )


def to_continuous_time(game_time: GameTime, metadata: GameMetadata) -> float:
    """Convert ``GameTime`` to ``ContinuousTime`` (see ``docs/timings.md``)."""
    try:
        half_start = metadata.half_start_continuous[game_time.half]
    except KeyError as exc:
        raise ValueError(
            f"GameMetadata has no half_start for half={game_time.half}; "
            f"known halves: {sorted(metadata.half_start_continuous)}"
        ) from exc
    return half_start + game_time.seconds


class FrameRecord(BaseModel):
    """A single decoded frame paired with its timestamps.

    ``frame`` is ``None`` for sentinel records (e.g. a gap in a live stream),
    in which case ``source_metadata['gap']`` is ``True``.
    """

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    frame: np.ndarray | None
    game_time: GameTime
    continuous_time: float
    source_metadata: dict[str, Any] = Field(default_factory=dict)


@runtime_checkable
class FrameSource(Protocol):
    """A source of ``FrameRecord``s. Implementations are iterables."""

    def __iter__(self) -> Iterator[FrameRecord]: ...


@dataclass(frozen=True)
class ReconnectPolicy:
    """Exponential-backoff reconnect policy for ``LiveStreamFrameSource``.

    ``max_retries=None`` means retry forever.
    """

    initial_backoff_s: float = 1.0
    max_backoff_s: float = 30.0
    max_retries: int | None = None

    @classmethod
    def exponential_backoff(
        cls,
        initial: float = 1.0,
        max_s: float = 30.0,
        max_retries: int | None = None,
    ) -> ReconnectPolicy:
        return cls(
            initial_backoff_s=initial,
            max_backoff_s=max_s,
            max_retries=max_retries,
        )


class FileFrameSource:
    """Yield ``FrameRecord``s from a video file on disk (OpenCV-backed)."""

    def __init__(
        self,
        video_path: Path | str,
        kickoff: GameTime,
        metadata: GameMetadata | None = None,
    ):
        self.video_path = Path(video_path)
        if not self.video_path.exists():
            raise FileNotFoundError(self.video_path)
        self.kickoff = kickoff
        self.metadata = metadata or GameMetadata()

    def _open(self) -> tuple[cv2.VideoCapture, float]:
        cap = cv2.VideoCapture(str(self.video_path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        if not fps or fps <= 0:
            cap.release()
            raise ValueError(f"Could not determine FPS for {self.video_path}")
        return cap, fps

    def _record_for(self, frame: np.ndarray, frame_index: int, fps: float) -> FrameRecord:
        offset_s = frame_index / fps
        game_time = GameTime(
            half=self.kickoff.half,
            seconds=self.kickoff.seconds + offset_s,
        )
        kickoff_continuous = to_continuous_time(self.kickoff, self.metadata)
        return FrameRecord(
            frame=frame,
            game_time=game_time,
            continuous_time=kickoff_continuous + offset_s,
            source_metadata={
                "frame_index": frame_index,
                "video_path": str(self.video_path),
            },
        )

    def __iter__(self) -> Iterator[FrameRecord]:
        cap, fps = self._open()
        try:
            i = 0
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                yield self._record_for(frame, i, fps)
                i += 1
        finally:
            cap.release()

    def seek(self, seconds_from_start: float) -> Iterator[FrameRecord]:
        """Resume iteration ``seconds_from_start`` past the start of the video.

        Frame indices and timestamps in emitted records reflect the absolute
        position in the video, not the offset from ``seek``'s start point.
        """
        if seconds_from_start < 0:
            raise ValueError(f"seek requires non-negative offset, got {seconds_from_start}")
        cap, fps = self._open()
        try:
            cap.set(cv2.CAP_PROP_POS_MSEC, seconds_from_start * 1000.0)
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                # ``CAP_PROP_POS_FRAMES`` after a successful read is the index
                # of the *next* frame, so the just-read frame is index-1.
                next_idx = int(round(cap.get(cv2.CAP_PROP_POS_FRAMES)))
                yield self._record_for(frame, max(0, next_idx - 1), fps)
        finally:
            cap.release()


class StreamBackend(Protocol):
    """Pluggable transport used by ``LiveStreamFrameSource``.

    A backend abstracts the network/decoder layer so the reconnect and gap
    behaviour can be exercised in tests with a mock. ``read`` returns the
    next decoded frame paired with its wall-clock offset (seconds since the
    start of the broadcast). It raises ``ConnectionError`` if the underlying
    stream drops, and ``StopIteration`` when the stream ends cleanly.
    """

    def open(self, url: str) -> None: ...

    def read(self) -> tuple[np.ndarray, float]: ...

    def close(self) -> None: ...


class LiveStreamFrameSource:
    """Yield ``FrameRecord``s from a live stream URL with reconnect-on-drop.

    On a ``ConnectionError`` from the backend, the source emits a sentinel
    ``FrameRecord`` with ``frame=None`` and ``source_metadata={'gap': True,
    ...}``, then reconnects with exponential backoff (governed by
    ``reconnect``). On a clean ``StopIteration`` the stream ends.

    The default network backend (PyAV) is not yet implemented; pass
    ``backend=`` to drive the source. ``sleep`` is exposed for test
    injection — by default it is a no-op so policy tests run instantly.
    """

    def __init__(
        self,
        url: str,
        kickoff: GameTime,
        reconnect: ReconnectPolicy | None = None,
        backend: StreamBackend | None = None,
        metadata: GameMetadata | None = None,
        sleep: Callable[[float], None] | None = None,
    ):
        self.url = url
        self.kickoff = kickoff
        self.reconnect = reconnect or ReconnectPolicy.exponential_backoff()
        self.backend = backend
        self.metadata = metadata or GameMetadata()
        self._sleep: Callable[[float], None] = sleep if sleep is not None else (lambda _: None)

    def _record(
        self,
        *,
        frame: np.ndarray | None,
        wallclock_offset_s: float,
        source_metadata: dict[str, Any],
    ) -> FrameRecord:
        kickoff_continuous = to_continuous_time(self.kickoff, self.metadata)
        game_time = GameTime(
            half=self.kickoff.half,
            seconds=self.kickoff.seconds + max(0.0, wallclock_offset_s),
        )
        return FrameRecord(
            frame=frame,
            game_time=game_time,
            continuous_time=kickoff_continuous + max(0.0, wallclock_offset_s),
            source_metadata=source_metadata,
        )

    def __iter__(self) -> Iterator[FrameRecord]:
        if self.backend is None:
            raise NotImplementedError(
                "Default PyAV backend is not implemented yet; pass `backend=` to "
                "drive LiveStreamFrameSource (see StreamBackend protocol)."
            )

        backoff = self.reconnect.initial_backoff_s
        retries = 0
        last_offset = 0.0

        self.backend.open(self.url)
        try:
            while True:
                try:
                    frame, wallclock_offset_s = self.backend.read()
                except StopIteration:
                    return
                except ConnectionError as exc:
                    yield self._record(
                        frame=None,
                        wallclock_offset_s=last_offset,
                        source_metadata={
                            "gap": True,
                            "reason": str(exc) or "connection_error",
                            "url": self.url,
                            "retry": retries,
                        },
                    )
                    if (
                        self.reconnect.max_retries is not None
                        and retries >= self.reconnect.max_retries
                    ):
                        return
                    self._sleep(backoff)
                    backoff = min(backoff * 2.0, self.reconnect.max_backoff_s)
                    retries += 1
                    self.backend.close()
                    self.backend.open(self.url)
                    continue

                # Successful read: reset backoff state.
                backoff = self.reconnect.initial_backoff_s
                retries = 0
                last_offset = wallclock_offset_s
                yield self._record(
                    frame=frame,
                    wallclock_offset_s=wallclock_offset_s,
                    source_metadata={
                        "url": self.url,
                        "wallclock_offset_s": wallclock_offset_s,
                    },
                )
        finally:
            self.backend.close()
