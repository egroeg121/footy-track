"""
Export human-reviewed GT marks from JSONL files to a Roboflow object-detection project.

Only exports annotations where:
  - provenance == 'labeller'  (always exported)
  - provenance in ('yolo', 'sam3') AND the frame has been human-reviewed

Frames marked 'no_ball' or 'not_broadcast' are skipped entirely.

Usage:
    uv run python -m footy_track.scripts.export_gt_to_roboflow \\
        --gt-dir ~/Library/Mobile\\ Documents/com~apple~CloudDocs/footy_data/ball_gt_marks \\
        --clips-dir /path/to/eval_data/clips \\
        --project footy-track-detection \\
        --workspace egroeg121 \\
        [--dry-run] [--batch-name gt-export-YYYY-MM-DD]

API key is read from ROBOFLOW_API_KEY env var or ~/.config/roboflow/config.json.
"""

import argparse
import json
import logging
import os
import tempfile
from collections import defaultdict
from pathlib import Path

import cv2
import requests
from roboflow import Roboflow
from roboflow.config import API_URL

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
_log = logging.getLogger(__name__)

_SKIP_TAGS = {"no_ball", "not_broadcast"}
_LABELLER_PROV = "labeller"
_MACHINE_PROVS = {"yolo", "sam3"}

# Classes present in the GT marks that map directly to Roboflow class names.
_VALID_LABELS = {
    "player",
    "referee",
    "coach",
    "player_sub",
    "in_play_ball",
    "out_of_play_ball",
    "person",
    "ball",
}


def load_api_key() -> str:
    env_key = os.environ.get("ROBOFLOW_API_KEY")
    if env_key:
        return env_key
    config_path = Path.home() / ".config" / "roboflow" / "config.json"
    config = json.loads(config_path.read_text())
    for ws in config.get("workspaces", {}).values():
        if ws.get("url") == "egroeg121":
            return ws["apiKey"]
    raise ValueError("Roboflow API key not found in config or environment")


def fetch_existing_names(api_key: str, workspace: str, project_name: str) -> set[str]:
    """Fetch all image names already in the Roboflow project (for dedup)."""
    existing: set[str] = set()
    batch_size = 200
    offset = 0
    while True:
        r = requests.post(
            f"{API_URL}/{workspace}/{project_name}/search?api_key={api_key}",
            json={"limit": batch_size, "offset": offset, "fields": ["name"]},
        )
        r.raise_for_status()
        data = r.json()
        results = data.get("results", [])
        for img in results:
            if img.get("name"):
                existing.add(img["name"])
        if len(results) < batch_size:
            break
        offset += batch_size
    _log.info("Fetched %d existing image names from Roboflow", len(existing))
    return existing


def parse_jsonl_file(path: Path) -> dict[int, list[dict]]:
    """
    Parse a GT marks JSONL file.

    Returns a mapping of frame_index → list of annotation dicts
    (skipping no_ball/not_broadcast frames and annotations without exportable labels).
    Only includes annotations whose provenance is 'labeller' (unconditional) or
    a machine provenance where the frame was human-reviewed.

    A frame is considered human-reviewed when at least one annotation on that frame
    has provenance == 'labeller'.
    """
    raw_by_frame: dict[int, list[dict]] = defaultdict(list)

    with path.open() as f:
        for raw in f:
            raw = raw.strip()
            if not raw:
                continue
            try:
                d = json.loads(raw)
            except json.JSONDecodeError:
                continue

            tags = d.get("tags") or []
            # Frame-level skip markers — no bbox
            if any(t in _SKIP_TAGS for t in tags):
                continue

            frame_idx = int(d.get("frame_index", -1))
            if frame_idx < 0:
                continue

            raw_by_frame[frame_idx].append({"tags": tags, "bbox": d.get("bbox")})

    # Determine provenance for each frame
    frame_has_labeller: set[int] = set()
    for frame_idx, anns in raw_by_frame.items():
        if any(_LABELLER_PROV in a["tags"] for a in anns):
            frame_has_labeller.add(frame_idx)

    result: dict[int, list[dict]] = {}
    for frame_idx, anns in raw_by_frame.items():
        exportable = []
        for ann in anns:
            tags = ann["tags"]
            prov = next((t for t in tags if t in {_LABELLER_PROV} | _MACHINE_PROVS), None)
            if prov == _LABELLER_PROV:
                exportable.append(ann)
            elif prov in _MACHINE_PROVS and frame_idx in frame_has_labeller:
                exportable.append(ann)

        # Filter to annotations that have a valid label and a bbox
        valid = []
        for ann in exportable:
            bbox = ann.get("bbox")
            if bbox is None:
                continue
            tags = ann["tags"]
            label = next((t for t in tags if t in _VALID_LABELS), None)
            if label is None:
                continue
            valid.append({"label": label, "bbox": bbox})

        if valid:
            result[frame_idx] = valid

    return result


def build_yolo_annotation(annotations: list[dict], class_list: list[str]) -> str:
    """Build YOLO-format annotation string for a single frame.

    Bboxes in JSONL are top-left xywh normalised; YOLO format uses centre xywh normalised.
    """
    lines = []
    for ann in annotations:
        label = ann["label"]
        if label not in class_list:
            _log.warning("Label %r not in class list, skipping", label)
            continue
        class_idx = class_list.index(label)
        bbox = ann["bbox"]
        if isinstance(bbox, dict):
            cx = float(bbox["x"]) + float(bbox["w"]) / 2
            cy = float(bbox["y"]) + float(bbox["h"]) / 2
            w = float(bbox["w"])
            h = float(bbox["h"])
        else:
            x, y, bw, bh = (float(v) for v in bbox)
            cx = x + bw / 2
            cy = y + bh / 2
            w = bw
            h = bh
        lines.append(f"{class_idx} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
    return "\n".join(lines)


def find_clip_video(clips_dir: Path, clip_stem: str) -> Path | None:
    """Find the video file for a given clip stem."""
    for ext in (".mp4", ".mov", ".avi", ".mkv"):
        p = clips_dir / f"{clip_stem}{ext}"
        if p.exists():
            return p
    return None


def extract_frame(video_path: Path, frame_index: int) -> bytes | None:
    """Extract a single frame from a video as JPEG bytes."""
    cap = cv2.VideoCapture(str(video_path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        return None
    ok, buf = cv2.imencode(".jpg", frame)
    if not ok:
        return None
    return bytes(buf)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export GT marks to Roboflow")
    parser.add_argument("--gt-dir", type=Path, help="Directory containing .jsonl GT mark files")
    parser.add_argument("--clips-dir", type=Path, help="Directory containing clip video files")
    parser.add_argument("--project", default="footy-track-detection", help="Roboflow project slug")
    parser.add_argument("--workspace", default="egroeg121", help="Roboflow workspace slug")
    parser.add_argument("--batch-name", default="gt-export", help="Roboflow batch name for upload")
    parser.add_argument("--dry-run", action="store_true", help="Print what would be uploaded without uploading")
    args = parser.parse_args()

    # Defaults
    gt_dir = args.gt_dir or (
        Path.home() / "Library" / "Mobile Documents" / "com~apple~CloudDocs" / "footy_data" / "ball_gt_marks"
    )
    clips_dir = args.clips_dir or (Path(__file__).parents[4] / "eval_data" / "clips")

    if not gt_dir.exists():
        raise FileNotFoundError(f"GT marks directory not found: {gt_dir}")
    if not clips_dir.exists():
        raise FileNotFoundError(f"Clips directory not found: {clips_dir}")

    api_key = load_api_key()

    _log.info("Loading Roboflow project %s/%s", args.workspace, args.project)
    rf = Roboflow(api_key=api_key)
    project = rf.workspace(args.workspace).project(args.project)

    # Fetch project classes in order (needed for YOLO class indices)
    class_list = list(project.classes.keys())
    _log.info("Project classes: %s", class_list)

    # Fetch existing image names for dedup
    existing_names = fetch_existing_names(api_key, args.workspace, args.project)

    jsonl_files = sorted(gt_dir.glob("*.jsonl"))
    _log.info("Found %d JSONL files in %s", len(jsonl_files), gt_dir)

    total_uploaded = 0
    total_skipped_dedup = 0
    total_skipped_no_video = 0
    total_skipped_no_frame = 0

    for jsonl_path in jsonl_files:
        clip_stem = jsonl_path.stem
        frame_annotations = parse_jsonl_file(jsonl_path)

        if not frame_annotations:
            _log.info("[%s] No exportable annotations, skipping", clip_stem)
            continue

        video_path = find_clip_video(clips_dir, clip_stem)
        if video_path is None:
            _log.warning("[%s] No video file found in %s, skipping", clip_stem, clips_dir)
            total_skipped_no_video += len(frame_annotations)
            continue

        _log.info("[%s] %d frames to export (video: %s)", clip_stem, len(frame_annotations), video_path.name)

        for frame_idx, anns in sorted(frame_annotations.items()):
            image_name = f"{clip_stem}_{frame_idx:06d}.jpg"

            if image_name in existing_names:
                _log.debug("[%s] Frame %d already in Roboflow (%s), skipping", clip_stem, frame_idx, image_name)
                total_skipped_dedup += 1
                continue

            if args.dry_run:
                labels_summary = ", ".join(f"{a['label']}" for a in anns)
                _log.info("[DRY RUN] Would upload %s (%d annotations: %s)", image_name, len(anns), labels_summary)
                total_uploaded += 1
                continue

            frame_bytes = extract_frame(video_path, frame_idx)
            if frame_bytes is None:
                _log.warning("[%s] Could not extract frame %d, skipping", clip_stem, frame_idx)
                total_skipped_no_frame += 1
                continue

            yolo_annotation = build_yolo_annotation(anns, class_list)

            with tempfile.TemporaryDirectory() as tmpdir:
                img_path = Path(tmpdir) / image_name
                img_path.write_bytes(frame_bytes)

                ann_path = Path(tmpdir) / f"{clip_stem}_{frame_idx:06d}.txt"
                ann_path.write_text(yolo_annotation)

                try:
                    project.single_upload(
                        image_path=str(img_path),
                        annotation_path=str(ann_path),
                        batch_name=args.batch_name,
                        split="train",
                        num_retry_uploads=2,
                    )
                    _log.info("Uploaded %s (%d annotations)", image_name, len(anns))
                    existing_names.add(image_name)
                    total_uploaded += 1
                except Exception as exc:
                    _log.error("Failed to upload %s: %s", image_name, exc)

    _log.info(
        "Done. uploaded=%d  skipped_dedup=%d  skipped_no_video=%d  skipped_no_frame=%d",
        total_uploaded,
        total_skipped_dedup,
        total_skipped_no_video,
        total_skipped_no_frame,
    )

    if args.dry_run:
        _log.info("[DRY RUN] No changes made.")


if __name__ == "__main__":
    main()
