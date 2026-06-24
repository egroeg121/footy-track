"""Unit tests for labeller/interpolation.py (ft-4e9).

All tests are pure: no model, no video, no I/O.
"""

from footy_track.labeller.interpolation import (
    REASON_CENTER_JUMP,
    REASON_LOST,
    REASON_SIZE_JUMP,
    HandbackConfig,
    should_handback,
)

# A stable reference box used across most tests: centre at (0.5, 0.5), 10% size.
_BOX: tuple[float, float, float, float] = (0.45, 0.45, 0.10, 0.10)


def test_continue_when_all_good() -> None:
    """Trivial happy path: nearby box, healthy score."""
    result = should_handback(_BOX, _BOX, score=0.9)
    assert not result.stop
    assert result.reason is None


def test_lost_when_bbox_is_none() -> None:
    result = should_handback(_BOX, None, score=0.0)
    assert result.stop
    assert result.reason == REASON_LOST


def test_lost_when_score_below_threshold() -> None:
    cfg = HandbackConfig(min_score=0.30)
    result = should_handback(_BOX, _BOX, score=0.29, cfg=cfg)
    assert result.stop
    assert result.reason == REASON_LOST


def test_continue_at_exact_score_threshold() -> None:
    cfg = HandbackConfig(min_score=0.30)
    result = should_handback(_BOX, _BOX, score=0.30, cfg=cfg)
    assert not result.stop


def test_center_jump_fires() -> None:
    """Box centre teleports across the frame — should hand back."""
    far_box = (0.80, 0.80, 0.10, 0.10)
    cfg = HandbackConfig(max_center_jump_frac=0.08)
    result = should_handback(_BOX, far_box, score=0.9, cfg=cfg)
    assert result.stop
    assert result.reason == REASON_CENTER_JUMP


def test_center_jump_does_not_fire_for_small_move() -> None:
    """Tiny nudge should not trigger center_jump."""
    tiny_move = (0.451, 0.451, 0.10, 0.10)
    result = should_handback(_BOX, tiny_move, score=0.9)
    assert not result.stop


def test_size_jump_fires_on_balloon() -> None:
    """Box area triples — balloon case."""
    big_box = (0.40, 0.40, 0.20, 0.20)  # area = 0.04, prev = 0.01 → ratio = 4
    cfg = HandbackConfig(max_size_ratio=2.5)
    result = should_handback(_BOX, big_box, score=0.9, cfg=cfg)
    assert result.stop
    assert result.reason == REASON_SIZE_JUMP


def test_size_jump_fires_on_collapse() -> None:
    """Box area collapses to near zero — collapse case."""
    tiny_box = (0.49, 0.49, 0.01, 0.01)  # area = 0.0001, prev = 0.01 → ratio = 100
    cfg = HandbackConfig(max_size_ratio=2.5)
    result = should_handback(_BOX, tiny_box, score=0.9, cfg=cfg)
    assert result.stop
    assert result.reason == REASON_SIZE_JUMP


def test_size_jump_within_ratio_is_fine() -> None:
    slightly_bigger = (0.44, 0.44, 0.12, 0.12)  # area ≈ 0.0144 vs 0.01 → ratio 1.44
    result = should_handback(_BOX, slightly_bigger, score=0.9)
    assert not result.stop


def test_lost_checked_before_center_jump() -> None:
    """Even if the box teleports, 'lost' fires first when score is below threshold."""
    far_box = (0.80, 0.80, 0.10, 0.10)
    cfg = HandbackConfig(min_score=0.30, max_center_jump_frac=0.08)
    result = should_handback(_BOX, far_box, score=0.10, cfg=cfg)
    assert result.stop
    assert result.reason == REASON_LOST


def test_config_override_raises_threshold() -> None:
    """Raising min_score catches a score that default config would pass."""
    strict = HandbackConfig(min_score=0.80)
    result = should_handback(_BOX, _BOX, score=0.70, cfg=strict)
    assert result.stop
    assert result.reason == REASON_LOST
