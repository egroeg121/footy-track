"""Import existing label sets into the feature store.

Two label sources are supported today, each landing as ``detection`` rows tagged
by ``source`` + ``run_id`` so they coexist and stay comparable:

- **Roboflow YOLO detection datasets** (your hand labels) -> ``source='hand_label'``.
  YOLO labels are *centre* xywh class indices; we convert to the store's
  *top-left* xywh and map class indices to names via ``data.yaml``.
- **Web-labeller / SAM3 JSON** (``FrameDetections`` list) -> ``source='sam3'`` by
  default. Boxes are already top-left xywh.

Frame identity ``(game_id, frame_index)`` is recovered from filenames: both
sources ultimately encode ``<video_stem>_<frame_index>`` because
``extract_frames`` names frames ``<video_stem>_%06d``. Two label sets line up
frame-for-frame only if they came from the same video at the same fps; use
:func:`source_overlap` to check empirically after import.

``continuous_time_s`` is derived as ``frame_index / fps + time_offset``. With
the default 1-fps extraction (see ``docs/data_formats.md``) and offset 0 this is
seconds-from-frame-0; supply a real offset once kickoff alignment is known
(``docs/timings.md``).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import yaml

from footy_track.feature_store.schema import (
    DetectionRow,
    FrameRow,
    GameRow,
    RunRow,
    Stage,
)

if TYPE_CHECKING:
    from footy_track.feature_store.store import FeatureStore

# Roboflow appends "_<origext>.rf.<hash>" to the original frame filename.
_ROBOFLOW_SUFFIX = re.compile(r"_(png|jpg|jpeg|webp)\.rf\.[0-9a-f]+$", re.IGNORECASE)
# A trailing "_<frame index>" (zero-padded by extract_frames' %06d).
_TRAILING_INDEX = re.compile(r"^(?P<stem>.+)_(?P<idx>\d+)$")
_LABELLER_FRAME = re.compile(r"^(?P<stem>.+)_frame_(?P<idx>\d+)$")


@dataclass
class ImportReport:
    """Summary of one import call."""

    games: set[str] = field(default_factory=set)
    frames_written: int = 0
    detections_written: int = 0
    sources: set[str] = field(default_factory=set)
    min_frame_index: int | None = None
    max_frame_index: int | None = None

    def _note_frame(self, game_id: str, frame_index: int) -> None:
        self.games.add(game_id)
        self.frames_written += 1
        self.min_frame_index = (
            frame_index
            if self.min_frame_index is None
            else min(self.min_frame_index, frame_index)
        )
        self.max_frame_index = (
            frame_index
            if self.max_frame_index is None
            else max(self.max_frame_index, frame_index)
        )


# --------------------------------------------------------------------------- #
# Filename -> (game_id, frame_index)                                          #
# --------------------------------------------------------------------------- #


def parse_roboflow_stem(filename: str) -> tuple[str, int]:
    """``arsenal_mancity_20250925_002143_png.rf.<hash>.jpg`` -> (stem, 2143)."""
    name = Path(filename).stem  # drop final extension (.jpg/.txt)
    name = _ROBOFLOW_SUFFIX.sub("", name)  # drop _png.rf.<hash> if present
    m = _TRAILING_INDEX.match(name)
    if not m:
        raise ValueError(f"cannot parse frame index from roboflow name: {filename!r}")
    return m.group("stem"), int(m.group("idx"))


def parse_labeller_uri(uri: str) -> tuple[str, int]:
    """``arsenal_mancity_example_video_frame_000000`` -> (stem, 0).

    Falls back to a trailing ``_<index>`` if no ``_frame_`` infix is present.
    """
    name = Path(uri).name
    m = _LABELLER_FRAME.match(name) or _TRAILING_INDEX.match(name)
    if not m:
        raise ValueError(f"cannot parse frame index from labeller uri: {uri!r}")
    return m.group("stem"), int(m.group("idx"))


def _continuous_time(frame_index: int, fps: float, offset: float) -> float:
    return frame_index / fps + offset


# --------------------------------------------------------------------------- #
# Roboflow YOLO detection dataset                                             #
# --------------------------------------------------------------------------- #


def _yolo_centre_to_topleft(
    cx: float, cy: float, w: float, h: float
) -> tuple[float, float, float, float]:
    """YOLO centre xywh -> store top-left xywh, clamped to [0, 1]."""
    x = max(0.0, cx - w / 2)
    y = max(0.0, cy - h / 2)
    return x, y, min(w, 1.0), min(h, 1.0)


def _image_dims(image_path: Path) -> tuple[int, int] | None:
    try:
        from PIL import Image  # noqa: PLC0415
    except ImportError:
        return None
    try:
        with Image.open(image_path) as im:
            return int(im.width), int(im.height)
    except (OSError, ValueError):
        return None


def import_roboflow(
    store: FeatureStore,
    dataset_dir: str | Path,
    *,
    game_id: str | None = None,
    run_id: str | None = None,
    fps: float = 1.0,
    time_offset: float = 0.0,
    splits: tuple[str, ...] = ("train", "valid", "test"),
    default_width: int | None = None,
    default_height: int | None = None,
) -> ImportReport:
    """Import a Roboflow YOLO detection dataset as ``source='hand_label'`` rows.

    ``game_id`` (if ``None``) is taken from each frame's parsed video stem.
    Frame width/height come from the image file; pass ``default_width`` /
    ``default_height`` to skip image reads (e.g. when only labels are present).
    """
    dataset_dir = Path(dataset_dir)
    data_yaml = yaml.safe_load((dataset_dir / "data.yaml").read_text())
    class_names: list[str] = list(data_yaml["names"])
    version = data_yaml.get("roboflow", {}).get("version", "unknown")
    run_id = run_id or f"roboflow_v{version}"

    store.upsert_runs(
        [
            RunRow(
                run_id=run_id,
                stage=Stage.DETECTION,
                source="hand_label",
                model_name="human",
                model_version=f"roboflow_dataset_{version}",
                params_json=json.dumps({"classes": class_names}),
            )
        ]
    )

    report = ImportReport(sources={"hand_label"})
    games_seen: set[str] = set()

    for split in splits:
        labels_dir = dataset_dir / split / "labels"
        images_dir = dataset_dir / split / "images"
        if not labels_dir.is_dir():
            continue
        for label_path in sorted(labels_dir.glob("*.txt")):
            stem, frame_index = parse_roboflow_stem(label_path.name)
            gid = game_id or stem

            # locate the matching image to read dims / record a uri
            image_path = (
                next(images_dir.glob(label_path.stem + ".*"), None)
                if images_dir.is_dir()
                else None
            )
            dims = _image_dims(image_path) if image_path else None
            width, height = dims or (default_width, default_height)
            if width is None or height is None:
                raise ValueError(
                    f"no image dims for {label_path.name}; pass default_width/default_height"
                )

            ct = _continuous_time(frame_index, fps, time_offset)
            if gid not in games_seen:
                store.upsert_games([GameRow(game_id=gid, fps=fps)])
                games_seen.add(gid)

            store.upsert_frames(
                [
                    FrameRow(
                        game_id=gid,
                        frame_index=frame_index,
                        frame_uri=str(image_path) if image_path else label_path.name,
                        width=width,
                        height=height,
                        continuous_time_s=ct,
                    )
                ]
            )
            report._note_frame(gid, frame_index)

            rows: list[DetectionRow] = []
            for i, line in enumerate(label_path.read_text().splitlines()):
                parts = line.split()
                if len(parts) < 5:
                    continue
                cls_idx, cx, cy, w, h = int(parts[0]), *map(float, parts[1:5])
                x, y, bw, bh = _yolo_centre_to_topleft(cx, cy, w, h)
                rows.append(
                    DetectionRow(
                        game_id=gid,
                        frame_index=frame_index,
                        continuous_time_s=ct,
                        detection_id=i,
                        source="hand_label",
                        run_id=run_id,
                        label=class_names[cls_idx],
                        confidence=None,  # hand labels have no model confidence
                        bbox_x=x,
                        bbox_y=y,
                        bbox_w=bw,
                        bbox_h=bh,
                    )
                )
            store.upsert_detections(rows)
            report.detections_written += len(rows)

    return report


# --------------------------------------------------------------------------- #
# Web-labeller / SAM3 JSON                                                    #
# --------------------------------------------------------------------------- #


def import_labeller_json(
    store: FeatureStore,
    json_path: str | Path,
    *,
    run_id: str,
    game_id: str | None = None,
    source: str = "sam3",
    model_name: str = "sam3",
    model_version: str | None = None,
    fps: float = 1.0,
    time_offset: float = 0.0,
) -> ImportReport:
    """Import a web-labeller export (list of ``FrameDetections`` dicts).

    Boxes are already top-left xywh. ``source`` defaults to ``'sam3'`` so the
    labels stay distinct from Roboflow hand labels and can be evaluated against
    them; pass ``source='hand_label'`` to treat them as ground truth.
    """
    records = json.loads(Path(json_path).read_text())

    store.upsert_runs(
        [
            RunRow(
                run_id=run_id,
                stage=Stage.DETECTION,
                source=source,
                model_name=model_name,
                model_version=model_version,
            )
        ]
    )

    report = ImportReport(sources={source})
    games_seen: set[str] = set()

    for rec in records:
        stem, frame_index = parse_labeller_uri(rec["uri"])
        gid = game_id or stem
        ct = _continuous_time(frame_index, fps, time_offset)

        if gid not in games_seen:
            store.upsert_games([GameRow(game_id=gid, fps=fps)])
            games_seen.add(gid)

        store.upsert_frames(
            [
                FrameRow(
                    game_id=gid,
                    frame_index=frame_index,
                    frame_uri=rec["uri"],
                    width=int(rec["width"]),
                    height=int(rec["height"]),
                    continuous_time_s=ct,
                )
            ]
        )
        report._note_frame(gid, frame_index)

        rows: list[DetectionRow] = []
        for i, det in enumerate(rec.get("detections", [])):
            rows.append(
                DetectionRow(
                    game_id=gid,
                    frame_index=frame_index,
                    continuous_time_s=ct,
                    detection_id=i,
                    source=source,
                    run_id=run_id,
                    label=det["label"],
                    confidence=det.get("confidence"),
                    bbox_x=det["x"],
                    bbox_y=det["y"],
                    bbox_w=det["w"],
                    bbox_h=det["h"],
                )
            )
        store.upsert_detections(rows)
        report.detections_written += len(rows)

    return report


# --------------------------------------------------------------------------- #
# Overlap report                                                              #
# --------------------------------------------------------------------------- #


def source_overlap(store: FeatureStore):
    """Per ``(game_id, frame_index)``, which sources have detections and how many.

    Returns a DataFrame with one row per (game_id, frame_index) carrying
    ``n_sources``, a comma-joined ``sources`` list, and per-frame counts. Frames
    where ``n_sources > 1`` are where two label sets overlap and can be compared.
    """
    return store.query(
        """
        SELECT game_id,
               frame_index,
               count(DISTINCT source) AS n_sources,
               string_agg(DISTINCT source, ',' ORDER BY source) AS sources,
               count(*) AS n_detections
        FROM detection
        GROUP BY game_id, frame_index
        ORDER BY n_sources DESC, game_id, frame_index
        """
    )
