"""Smoke test for ``train_classifier.py``.

Runs one epoch of training against a tiny local binary classification dataset
(no Roboflow download), and checks that the script writes ``best.pt`` and
returns a non-zero top-1 accuracy.
"""

from pathlib import Path

from footy_track.scripts.train_classifier import train_classifier


def test_train_classifier_smoke(classifier_dataset: Path, staged_workdir: Path) -> None:
    accuracy, weights_path = train_classifier(
        model_name="yolo11n-cls",
        dataset_version=10,
        freeze_layers=9,
        epochs=1,
        local_dataset=str(classifier_dataset),
    )
    assert Path(weights_path).is_file(), f"best.pt not written at {weights_path}"
    assert accuracy > 0, f"expected top1_acc > 0, got {accuracy}"
