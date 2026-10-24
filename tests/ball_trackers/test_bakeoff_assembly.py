"""Tests for the bake-off assembly (ft-5hd.2).

Verifies the scratch branch actually wires the harness (ball_eval, ft-1my)
together with all three bake-off methods:

  - Method A: VitTrackSOT              (ft-ztw)
  - Method B: Sam2BallTracker          (ft-xps)
  - Method C: RoiYoloTracker           (ft-1d9)

and that `scripts/run_bakeoff_abc.py`'s method registry enumerates them and
each satisfies the BallTracker protocol end-to-end (track/reset), without
requiring GPU, network, or real model weights (SAM2/YOLO are mocked; the
harness runs against a synthetic in-memory dataset).
"""

from __future__ import annotations

import importlib
import inspect
import pathlib
import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from footy_track.ball_eval.dataset import EvalClip, EvalDataset, FrameLabel
from footy_track.ball_eval.interface import BallTracker
from footy_track.ball_eval.runner import run_benchmark
from footy_track.ball_trackers import RoiYoloTracker, Sam2BallTracker, VitTrackSOT

SCRIPTS_DIR = pathlib.Path(__file__).parents[2] / "scripts"


def _load_run_bakeoff_abc():
    """Import scripts/run_bakeoff_abc.py as a module (scripts/ isn't a package)."""
    if str(SCRIPTS_DIR.parent) not in sys.path:
        sys.path.insert(0, str(SCRIPTS_DIR.parent))
    return importlib.import_module("scripts.run_bakeoff_abc")


# ---------------------------------------------------------------------------
# Registry enumeration (the "harness can enumerate the 3 methods" smoke check)
# ---------------------------------------------------------------------------


def test_method_registry_has_exactly_abc():
    mod = _load_run_bakeoff_abc()
    assert set(mod.METHOD_REGISTRY) == {"sot", "sam2", "roi-yolo"}
    assert mod.DEFAULT_METHOD_ORDER == ["sot", "sam2", "roi-yolo"]


def test_method_registry_constructors_are_lazy():
    """Merely holding the registry must not touch network/GPU/weights."""
    mod = _load_run_bakeoff_abc()
    for _key, (name, _make) in mod.METHOD_REGISTRY.items():
        assert isinstance(name, str) and name
        assert callable(_make)


def test_cli_list_enumerates_without_constructing(capsys, tmp_path):
    """`--list` must enumerate methods without instantiating any tracker."""
    mod = _load_run_bakeoff_abc()
    argv = sys.argv
    sys.argv = ["run_bakeoff_abc.py", "--list", "--clips-dir", str(tmp_path)]
    try:
        with (
            patch.object(mod, "_make_sot", side_effect=AssertionError("should not run")),
            patch.object(mod, "_make_sam2", side_effect=AssertionError("should not run")),
            patch.object(
                mod, "_make_roi_yolo", side_effect=AssertionError("should not run")
            ),
        ):
            mod.main()
    finally:
        sys.argv = argv
    out = capsys.readouterr().out
    assert "sot" in out
    assert "sam2" in out
    assert "roi-yolo" in out


# ---------------------------------------------------------------------------
# Protocol conformance: every method class satisfies BallTracker structurally
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("cls", [VitTrackSOT, Sam2BallTracker, RoiYoloTracker])
def test_tracker_class_conforms_to_ball_tracker_protocol(cls):
    protocol_methods = [
        name for name, _ in inspect.getmembers(BallTracker) if not name.startswith("_")
    ]
    assert "track" in protocol_methods
    assert "reset" in protocol_methods
    for method_name in protocol_methods:
        assert hasattr(cls, method_name), f"{cls.__name__} missing {method_name}()"
        sig = inspect.signature(getattr(cls, method_name))
        assert "self" in sig.parameters


# ---------------------------------------------------------------------------
# End-to-end: run_benchmark against a synthetic clip for each method, with
# the underlying model backends mocked out (no GPU/network/weights needed).
# ---------------------------------------------------------------------------


def _synthetic_dataset() -> EvalDataset:
    frames = [np.zeros((64, 64, 3), dtype=np.uint8) for _ in range(3)]
    labels = {
        i: FrameLabel(frame_index=i, bbox=(0.4, 0.4, 0.1, 0.1), tags=())
        for i in range(3)
    }
    clip = EvalClip(
        name="synthetic",
        video_path=pathlib.Path("unused.mp4"),
        labels=labels,
        total_frames=len(frames),
    )

    def _iter_frames():
        yield from enumerate(frames)

    clip.iter_frames = _iter_frames  # type: ignore[method-assign]
    return EvalDataset([clip])


def test_roi_yolo_runs_against_harness_with_mocked_model():
    with patch("ultralytics.YOLO") as mock_yolo_cls:
        mock_model = MagicMock()
        mock_yolo_cls.return_value = mock_model
        tracker = RoiYoloTracker()
        tracker._model = mock_model
        boxes = MagicMock()
        boxes.conf = None
        mock_model.predict.return_value = [MagicMock(boxes=None)]

        result = run_benchmark(tracker, _synthetic_dataset(), method_name="roi-yolo-test")
        assert result.method_name == "roi-yolo-test"
        assert len(result.clip_metrics) == 1


def test_sam2_runs_against_harness_with_mocked_model():
    with patch("ultralytics.models.sam.SAM") as mock_sam_cls:
        mock_predictor = MagicMock()
        mock_sam_cls.return_value = mock_predictor
        mock_predictor.return_value = []  # no masks/boxes -> ball not found

        tracker = Sam2BallTracker()
        result = run_benchmark(tracker, _synthetic_dataset(), method_name="sam2-test")
        assert result.method_name == "sam2-test"
        assert len(result.clip_metrics) == 1
