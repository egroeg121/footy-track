"""Pure hand-back trigger logic for VitTrack label interpolation (ft-4e9).

A human hand-labels a frame; VitTrack carries each box forward until it loses
confidence or the track goes implausible, then *hands back* to the human. The
trust-critical part is deciding *when* to hand back. That decision lives here as
side-effect-free functions so it can be unit-tested without a model or a video.

BBox convention matches the rest of the labeller: normalised ``(x, y, w, h)``
top-left, in ``[0, 1]``.

See ``docs/design/vittrack_interpolation.md``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

BBox = tuple[float, float, float, float]


@dataclass(frozen=True)
class HandbackConfig:
    """Thresholds for the hand-back triggers.

    Defaults are starting points tuned from the ft-5hd bake-off. The
    ``/interpolate`` endpoint surfaces them so they can be adjusted per-run
    without a code change.
    """

    min_score: float = 0.30
    """Hand back when VitTrack's confidence drops below this (the 'drops the
    ball' case). Primary trigger."""

    max_center_jump_frac: float = 0.08
    """Hand back when the box centre moves more than this fraction of the frame
    diagonal in a single frame. Catches teleports onto a stale patch."""

    max_size_ratio: float = 2.5
    """Hand back when new_area / old_area leaves
    ``[1 / max_size_ratio, max_size_ratio]``. Catches the box ballooning or
    collapsing as the tracker locks onto background."""


@dataclass(frozen=True)
class HandbackResult:
    """Outcome of a single per-frame hand-back check."""

    stop: bool
    reason: str | None = None


# Reason codes (also surfaced to the UI).
REASON_LOST = "lost"
REASON_CENTER_JUMP = "center_jump"
REASON_SIZE_JUMP = "size_jump"


def _center(bbox: BBox) -> tuple[float, float]:
    x, y, w, h = bbox
    return x + w / 2, y + h / 2


def should_handback(
    prev_bbox: BBox,
    new_bbox: BBox | None,
    score: float,
    cfg: HandbackConfig = HandbackConfig(),
) -> HandbackResult:
    """Decide whether VitTrack should hand back to the human at this frame.

    Args:
        prev_bbox: The box on the previous (accepted) frame, normalised.
        new_bbox: VitTrack's proposed box for this frame, or ``None`` if VitTrack
            lost the object.
        score: VitTrack's confidence for ``new_bbox`` (ignored if ``new_bbox`` is
            ``None``).
        cfg: Trigger thresholds.

    Returns:
        ``HandbackResult(stop=False)`` to continue, or
        ``HandbackResult(stop=True, reason=...)`` to hand back. The reason is one
        of ``REASON_LOST`` / ``REASON_CENTER_JUMP`` / ``REASON_SIZE_JUMP``.

    The triggers are checked in order; the first to fire wins. ``lost`` (None or
    low score) is checked first because a low-confidence box is not worth
    sanity-checking for motion or size.
    """
    # 1. Lost — VitTrack yielded nothing, or its own confidence collapsed.
    if new_bbox is None or score < cfg.min_score:
        return HandbackResult(stop=True, reason=REASON_LOST)

    # 2. Implausible motion — centre teleported.
    px, py = _center(prev_bbox)
    nx, ny = _center(new_bbox)
    # Frame diagonal in normalised space is sqrt(2); compare against it so the
    # threshold is resolution-independent.
    jump = math.hypot(nx - px, ny - py)
    if jump > cfg.max_center_jump_frac * math.sqrt(2.0):
        return HandbackResult(stop=True, reason=REASON_CENTER_JUMP)

    # 3. Implausible size — box ballooned or collapsed.
    prev_area = max(prev_bbox[2] * prev_bbox[3], 1e-9)
    new_area = max(new_bbox[2] * new_bbox[3], 1e-9)
    ratio = new_area / prev_area
    if ratio > cfg.max_size_ratio or ratio < 1.0 / cfg.max_size_ratio:
        return HandbackResult(stop=True, reason=REASON_SIZE_JUMP)

    return HandbackResult(stop=False)
