"""
Script to create Roboflow v11 of the footy-track-broadcast-frame dataset.

Steps:
  1. Search for and delete any images with the 'Unlabeled' class in the train split.
  2. Generate v11 with filter-null preprocessing to exclude any remaining unlabeled images.

Usage:
    uv run python src/footy_track/scripts/create_roboflow_v11.py [--dry-run]

API key is read from ~/.config/roboflow/config.json or ROBOFLOW_API_KEY env var.
"""

import argparse
import json
import os
from pathlib import Path

import requests
from roboflow import Roboflow
from roboflow.config import API_URL


def load_api_key() -> str:
    env_key = os.environ.get("ROBOFLOW_API_KEY")
    if env_key:
        return env_key
    config_path = Path.home() / ".config" / "roboflow" / "config.json"
    config = json.loads(config_path.read_text())
    workspaces = config.get("workspaces", {})
    for ws in workspaces.values():
        if ws.get("url") == "egroeg121":
            return ws["apiKey"]
    raise ValueError("Roboflow API key not found in config or environment")


def find_unlabeled_images(
    api_key: str, workspace: str, project_name: str
) -> list[dict]:
    """Return all images in the train split with 'Unlabeled' class label."""
    unlabeled = []
    offset = 0
    batch_size = 200

    # First try explicit class_name filter
    r = requests.post(
        f"{API_URL}/{workspace}/{project_name}/search?api_key={api_key}",
        json={
            "limit": batch_size,
            "fields": ["id", "name", "split", "labels"],
            "class_name": "Unlabeled",
        },
    )
    r.raise_for_status()
    data = r.json()
    unlabeled.extend(
        [img for img in data.get("results", []) if img.get("split") == "train"]
    )

    if data.get("total", 0) > batch_size:
        # Page through remaining results
        while offset + batch_size < data["total"]:
            offset += batch_size
            r2 = requests.post(
                f"{API_URL}/{workspace}/{project_name}/search?api_key={api_key}",
                json={
                    "limit": batch_size,
                    "offset": offset,
                    "fields": ["id", "name", "split", "labels"],
                    "class_name": "Unlabeled",
                },
            )
            r2.raise_for_status()
            unlabeled.extend(
                [
                    img
                    for img in r2.json().get("results", [])
                    if img.get("split") == "train"
                ]
            )

    return unlabeled


def delete_images(
    api_key: str, workspace: str, project_name: str, image_ids: list[str], dry_run: bool
) -> None:
    """Delete images from the Roboflow dataset by ID."""
    if not image_ids:
        print("No images to delete.")
        return
    print(
        f"{'[DRY RUN] Would delete' if dry_run else 'Deleting'} {len(image_ids)} image(s): {image_ids}"
    )
    if dry_run:
        return
    r = requests.delete(
        f"{API_URL}/{workspace}/{project_name}/images?api_key={api_key}",
        headers={"Content-Type": "application/json"},
        json={"images": image_ids},
    )
    if r.status_code not in (200, 204):
        raise RuntimeError(f"Delete failed ({r.status_code}): {r.text}")
    print(f"Deleted {len(image_ids)} image(s) successfully.")


def generate_v11(
    api_key: str, workspace: str, project_name: str, dry_run: bool
) -> int | None:
    """Generate v11 with auto-orient and filter-null preprocessing."""
    settings = {
        "preprocessing": {
            "auto-orient": True,
            "filter-null": {"percent": 100},
        },
        "augmentation": {},
    }
    print(
        f"{'[DRY RUN] Would generate' if dry_run else 'Generating'} v11 with settings:"
    )
    print(json.dumps(settings, indent=2))
    if dry_run:
        return None

    rf = Roboflow(api_key=api_key)
    project = rf.workspace(workspace).project(project_name)
    version_num = project.generate_version(settings=settings)
    print(f"Version {version_num} generation started.")
    return version_num


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create Roboflow v11 of footy-track-broadcast-frame"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Preview actions without making changes"
    )
    args = parser.parse_args()

    workspace = "egroeg121"
    project_name = "footy-track-broadcast-frame"

    api_key = load_api_key()
    print(f"Workspace: {workspace}, Project: {project_name}")

    # Step 1: find and delete Unlabeled images in train
    print("\n--- Step 1: Find Unlabeled images in train split ---")
    unlabeled = find_unlabeled_images(api_key, workspace, project_name)
    print(f"Found {len(unlabeled)} Unlabeled image(s) in train split.")
    for img in unlabeled:
        print(
            f"  id={img['id']} name={img.get('name', '')} labels={img.get('labels', [])}"
        )

    image_ids = [img["id"] for img in unlabeled]
    delete_images(api_key, workspace, project_name, image_ids, dry_run=args.dry_run)

    # Step 2: generate v11
    print("\n--- Step 2: Generate v11 ---")
    generate_v11(api_key, workspace, project_name, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
