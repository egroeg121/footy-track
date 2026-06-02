"""JSONL streaming writer with Parquet + JSON sidecar finalisation.

Usage
-----
writer = TrackingWriter()
for frame_detections, frame_t in frames:
    tracked = tracker.update(frame_detections, frame_t)
    for td in tracked:
        writer.write(td)
meta = tracker.finalise()
writer.finalise(output_dir=Path("data/match/tracks"), match_id="arsenal_mancity", meta=meta)
"""

from __future__ import annotations

import json
from io import StringIO
from pathlib import Path

from footy_track.trackers.base import TrackMeta, TrackedDetection


class TrackingWriter:
    """Buffers TrackedDetection rows and writes tracks.parquet + tracks_meta.json."""

    def __init__(self) -> None:
        self._rows: list[TrackedDetection] = []

    def write(self, detection: TrackedDetection) -> None:
        """Buffer one detection."""
        self._rows.append(detection)

    def to_jsonl(self) -> str:
        """Return all buffered detections as JSONL string (one JSON object per line)."""
        buf = StringIO()
        for row in self._rows:
            buf.write(row.model_dump_json())
            buf.write("\n")
        return buf.getvalue()

    def finalise(
        self,
        output_dir: Path,
        match_id: str,
        meta: list[TrackMeta],
        *,
        detector: str | None = None,
        tracker_name: str | None = None,
        fps: float | None = None,
        width: int | None = None,
        height: int | None = None,
    ) -> tuple[Path, Path]:
        """Write tracks.parquet and tracks_meta.json to *output_dir*.

        Returns (parquet_path, meta_path).
        """
        import pandas as pd  # noqa: PLC0415

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # -- Parquet ---------------------------------------------------------
        records = []
        for row in self._rows:
            records.append(
                {
                    "match_id": match_id,
                    "frame_index": row.frame_index,
                    "continuous_time_s": row.continuous_time_s,
                    "track_id": row.track_id,
                    "label": row.label,
                    "confidence": row.confidence,
                    "bbox_x": row.x,
                    "bbox_y": row.y,
                    "bbox_w": row.w,
                    "bbox_h": row.h,
                    "detector_model": row.model,
                    "tracker": tracker_name,
                    "is_interpolated": row.is_interpolated,
                }
            )
        df = pd.DataFrame(records)
        parquet_path = output_dir / "tracks.parquet"
        df.to_parquet(parquet_path, index=False)

        # -- JSON sidecar ----------------------------------------------------
        sidecar: dict = {
            "schema_version": "1.0.0",
            "match_id": match_id,
            "produced_by": {
                "detector": detector,
                "tracker": tracker_name,
            },
            "video": {
                "width": width,
                "height": height,
                "fps": fps,
            },
            "tracks": {
                str(m.track_id): m.to_dict() for m in meta
            },
        }
        meta_path = output_dir / "tracks_meta.json"
        meta_path.write_text(json.dumps(sidecar, indent=2))

        return parquet_path, meta_path
