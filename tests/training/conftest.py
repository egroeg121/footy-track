"""Shared fixtures for training-script smoke tests.

Each test runs the real training script for one epoch against a tiny
locally-built dataset, so the fixtures here build minimal valid
YOLO-classification and YOLOv8-detection layouts in ``tmp_path`` and stage
the pretrained model checkpoints next to the test's working directory.
"""

import json
import shutil
from pathlib import Path

import numpy as np
import pytest
import yaml
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
TEST_IMAGE = REPO_ROOT / "tests" / "data" / "arsenal_mancity_test_detection.jpg"
TEST_LABELS_JSON = REPO_ROOT / "tests" / "data" / "arsenal_mancity_test_detections.json"

# Roles in the labels JSON we collapse to a single ``person`` class for the
# detector smoke test. The remaining ``ball_*`` roles are dropped.
PERSON_ROLES = {"player_red", "player_blue", "goalkeeper", "referee", "coach"}


@pytest.fixture
def classifier_dataset(tmp_path: Path) -> Path:
    """Build a tiny binary YOLO-classification dataset at ``<tmp>/dataset``.

    Two classes (``No`` / ``Yes``) of solid-coloured 224×224 images. The colour
    contrast is deliberate so that one epoch with a pretrained backbone can
    reach > 0 top-1 accuracy on the val split.
    """
    rng = np.random.default_rng(0)
    root = tmp_path / "classifier_dataset"
    colours = {"No": (0, 0, 0), "Yes": (255, 255, 255)}
    for split in ("train", "val"):
        for cls, base in colours.items():
            d = root / split / cls
            d.mkdir(parents=True, exist_ok=True)
            for i in range(4):
                arr = np.full((224, 224, 3), base, dtype=np.uint8)
                noise = rng.integers(-5, 6, arr.shape, dtype=np.int16)
                arr = np.clip(arr.astype(np.int16) + noise, 0, 255).astype(np.uint8)
                Image.fromarray(arr).save(d / f"{cls}_{i}.jpg")
    return root


@pytest.fixture
def detection_dataset(tmp_path: Path) -> Path:
    """Build a tiny YOLOv8-format detection dataset at ``<tmp>/dataset``.

    Uses the existing LFS-backed ``arsenal_mancity_test_detection.jpg`` and
    its labels JSON, collapsing all person-like roles to a single ``person``
    class. The same image is used for both train and val so the pretrained
    ``yolo11n`` weights can score a non-zero mAP on the val split after one
    epoch.
    """
    if TEST_IMAGE.stat().st_size < 1000 or TEST_LABELS_JSON.stat().st_size < 100:
        pytest.skip("LFS-backed test image/labels not pulled (run 'git lfs pull').")

    with open(TEST_LABELS_JSON) as f:
        objects = json.load(f)["objects"]

    yolo_lines = [
        f"0 {x:.6f} {y:.6f} {w:.6f} {h:.6f}"
        for o in objects
        if o["label"] in PERSON_ROLES
        for x, y, w, h in [o["bbox"]]
    ]
    label_text = "\n".join(yolo_lines) + "\n"

    root = tmp_path / "detection_dataset"
    for split in ("train", "valid"):
        (root / split / "images").mkdir(parents=True, exist_ok=True)
        (root / split / "labels").mkdir(parents=True, exist_ok=True)
        shutil.copyfile(TEST_IMAGE, root / split / "images" / "frame.jpg")
        (root / split / "labels" / "frame.txt").write_text(label_text)

    with open(root / "data.yaml", "w") as f:
        yaml.safe_dump(
            {
                "path": str(root),
                "train": "train/images",
                "val": "valid/images",
                "names": {0: "person"},
            },
            f,
        )
    return root


@pytest.fixture
def staged_workdir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Run the training script from ``tmp_path`` with the pretrained ``.pt``
    checkpoints staged alongside, so ``YOLO("yolo11n[-cls].pt")`` resolves
    locally and outputs are written under ``tmp_path`` instead of polluting
    the repo's ``footy_scan_*`` directories.
    """
    for ckpt in ("yolo11n.pt", "yolo11n-cls.pt"):
        src = REPO_ROOT / ckpt
        if src.exists():
            shutil.copyfile(src, tmp_path / ckpt)
    monkeypatch.chdir(tmp_path)
    return tmp_path
