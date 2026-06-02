"""MatchExporter — serialises per-match parquet artifacts to FiftyOne / JSON / CSV.

See docs/design/output.md for the full specification.
"""

import json
from pathlib import Path

import fiftyone as fo
import pandas as pd

SCHEMA_VERSION = "1.0.0"

# Expected columns produced by the tracking stage (docs/design/player_tracking_format.md §4.1)
_TRACKS_COLUMNS = [
    "match_id",
    "frame_index",
    "continuous_time_s",
    "track_id",
    "label",
    "confidence",
    "bbox_x",
    "bbox_y",
    "bbox_w",
    "bbox_h",
    "detector_model",
    "tracker",
    "is_interpolated",
]


class MatchExporter:
    """Fan-out serialiser for a completed match's parquet artifacts.

    Reads ``<match_dir>/tracks/tracks.parquet`` and
    ``<match_dir>/tracks/tracks_meta.json`` on construction; all export
    methods are pure with respect to those inputs.

    All exports are idempotent — re-running overwrites; never appends.
    """

    def __init__(self, match_dir: Path) -> None:
        self._match_dir = Path(match_dir)
        self._tracks: pd.DataFrame = pd.read_parquet(
            self._match_dir / "tracks" / "tracks.parquet"
        )
        meta_path = self._match_dir / "tracks" / "tracks_meta.json"
        with meta_path.open() as fh:
            self._meta: dict = json.load(fh)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def to_json(self, out_path: Path) -> None:
        """Write a single JSON file containing all frame records.

        Schema matches docs/design/output.md §3. Frames are ordered by
        ``continuous_time_s``. Missing optional stage data is omitted.
        """
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        frames = self._build_frames()

        payload = {
            "match_id": self._meta.get("match_id", ""),
            "schema_version": SCHEMA_VERSION,
            "frames": frames,
            "tracks_meta": self._build_tracks_meta_list(),
        }

        with out_path.open("w") as fh:
            json.dump(payload, fh, indent=2)

    def to_csv(self, out_dir: Path) -> None:
        """Write per-entity CSV files to *out_dir*.

        Files produced (docs/design/output.md §4):
        - detections.csv  — one row per detection
        - tracks.csv      — one row per (track_id, frame)
        - tracks_meta.csv — one row per track
        - pitch_positions.csv — only when pitch_x / pitch_y columns present
        """
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        self._tracks.to_csv(out_dir / "detections.csv", index=False)

        tracks_cols = [
            c
            for c in [
                "track_id",
                "frame_index",
                "continuous_time_s",
                "label",
                "bbox_x",
                "bbox_y",
                "bbox_w",
                "bbox_h",
            ]
            if c in self._tracks.columns
        ]
        self._tracks[tracks_cols].to_csv(out_dir / "tracks.csv", index=False)

        tracks_meta_rows = self._build_tracks_meta_list()
        if tracks_meta_rows:
            pd.DataFrame(tracks_meta_rows).to_csv(
                out_dir / "tracks_meta.csv", index=False
            )

        if "pitch_x" in self._tracks.columns and "pitch_y" in self._tracks.columns:
            pitch_cols = [
                c
                for c in [
                    "track_id",
                    "frame_index",
                    "continuous_time_s",
                    "pitch_x",
                    "pitch_y",
                ]
                if c in self._tracks.columns
            ]
            self._tracks[pitch_cols].to_csv(
                out_dir / "pitch_positions.csv", index=False
            )

    def to_fiftyone(self, dataset_name: str) -> fo.Dataset:
        """Build a FiftyOne dataset — one sample per unique continuous_time_s.

        The dataset is created with ``overwrite=True`` so re-runs replace
        the existing dataset rather than append to it (idempotent).

        Each sample carries:
        - ``tracks`` (fo.Detections): one Detection per tracked object, with
          ``track_id`` stored as a custom attribute.
        - ``continuous_time`` (float): wall-clock seconds from video start.
        """
        dataset = fo.Dataset(name=dataset_name, overwrite=True)

        samples = []
        sorted_tracks = self._tracks.sort_values("continuous_time_s")
        for ct, group in sorted_tracks.groupby("continuous_time_s", sort=False):
            sample = fo.Sample(filepath=f"frame_{ct:.3f}.jpg")
            sample["continuous_time"] = float(ct)
            sample["tracks"] = fo.Detections(
                detections=[
                    fo.Detection(
                        label=str(row.label),
                        bounding_box=[
                            float(row.bbox_x),
                            float(row.bbox_y),
                            float(row.bbox_w),
                            float(row.bbox_h),
                        ],
                        confidence=float(row.confidence),
                        track_id=int(row.track_id),
                    )
                    for row in group.itertuples(index=False)
                ]
            )
            samples.append(sample)

        dataset.add_samples(samples)
        return dataset

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_frames(self) -> list[dict]:
        frames = []
        sorted_tracks = self._tracks.sort_values("continuous_time_s")
        for ct, group in sorted_tracks.groupby("continuous_time_s", sort=False):
            detections = [
                {
                    "track_id": int(row.track_id),
                    "label": str(row.label),
                    "confidence": float(row.confidence),
                    "bbox": {
                        "x": float(row.bbox_x),
                        "y": float(row.bbox_y),
                        "w": float(row.bbox_w),
                        "h": float(row.bbox_h),
                    },
                }
                for row in group.itertuples(index=False)
            ]
            frame: dict = {
                "continuous_time": float(ct),
                "detections": detections,
                "tracks": [
                    {
                        "track_id": int(row.track_id),
                        "x": float(row.bbox_x),
                        "y": float(row.bbox_y),
                    }
                    for row in group.itertuples(index=False)
                ],
            }
            if "pitch_x" in group.columns and "pitch_y" in group.columns:
                frame["pitch_positions"] = [
                    {
                        "track_id": int(row.track_id),
                        "x_pitch": float(row.pitch_x),
                        "y_pitch": float(row.pitch_y),
                    }
                    for row in group.itertuples(index=False)
                ]
            frames.append(frame)
        return frames

    def _build_tracks_meta_list(self) -> list[dict]:
        raw = self._meta.get("tracks", {})
        return [{"track_id": int(k), **v} for k, v in raw.items()]
