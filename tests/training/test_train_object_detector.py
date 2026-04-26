"""Smoke test for ``train_object_detector.py``.

Runs one epoch of training against a tiny local YOLOv8-format detection
dataset (no Roboflow download), and checks that the script writes ``best.pt``
and returns a finite mAP50-95 metric.
"""

import math
from pathlib import Path

from footy_track.scripts.train_object_detector import train_detector


def test_train_detector_smoke(detection_dataset: Path, staged_workdir: Path) -> None:
    metric, weights_path = train_detector(
        model_name="yolo11n",
        dataset_version=3,
        freeze_layers=9,
        epochs=1,
        local_dataset=str(detection_dataset),
    )
    assert Path(weights_path).is_file(), f"best.pt not written at {weights_path}"
    # One epoch on a single labelled frame is not enough for the head to
    # converge, so mAP50-95 may legitimately be 0. The smoke test only checks
    # that the pipeline returns a finite, non-negative number — i.e. that
    # training, validation, and the metrics dict are all wired up correctly.
    assert isinstance(metric, float) and math.isfinite(metric) and metric >= 0, (
        f"expected finite non-negative mAP50-95, got {metric!r}"
    )
