"""Enumerate the Roboflow v11 export and map every image to its source frame."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from footy_track.multiframe.frames import BALL_ID, read_yolo_labels
from footy_track.multiframe.mapping import FOOTY_DATA, SourceRef, parse_name, resolve

V11_DIR = FOOTY_DATA / "detection_dataset" / "roboflow_dataset_11"
SPLITS = ("train", "valid", "test")


@dataclass
class V11Item:
    split: str
    name: str  # image file name
    prefix: str
    idx: int
    ref: SourceRef
    labels: list = field(default_factory=list)

    @property
    def stem(self) -> str:
        return Path(self.name).stem

    @property
    def n_ball(self) -> int:
        return sum(1 for c, *_ in self.labels if c == BALL_ID)


def load_v11(root: Path = V11_DIR) -> list[V11Item]:
    items = []
    for split in SPLITS:
        for img in sorted((root / split / "images").iterdir()):
            prefix, idx = parse_name(img.name)
            lab = root / split / "labels" / (img.stem + ".txt")
            items.append(
                V11Item(split, img.name, prefix, idx, resolve(prefix, idx),
                        read_yolo_labels(lab) if lab.exists() else [])
            )
    return items
