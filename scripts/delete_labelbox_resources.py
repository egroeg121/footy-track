#!/usr/bin/env python3
"""
Delete Labelbox projects and datasets whose name starts with the configured prefixes.
Usage: set LABELBOX_API_KEY (or LB_API_KEY) in environment and run this script.
"""

import os
import sys
from datetime import datetime

try:
    from labelbox import Client
except Exception:
    print("Failed to import Labelbox SDK (labelbox). Install with: pip install labelbox")
    raise

API_KEY = os.getenv("LABELBOX_API_KEY") or os.getenv("LB_API_KEY")
if not API_KEY:
    print("ERROR: Set LABELBOX_API_KEY (or LB_API_KEY) in your environment.")
    sys.exit(1)

client = Client(api_key=API_KEY)
# Prefixes to target. Adjust if you want to match other naming patterns.
# Use a broader match so we remove both pre-annotation projects and datasets
# created as 'GroundingDINO - <something> - <timestamp>'
PROJECT_PREFIX = "GroundingDINO"
DATASET_PREFIX = "GroundingDINO"

print(
    f"Connecting to Labelbox...\nProject prefix: {PROJECT_PREFIX!r}\nDataset prefix: {DATASET_PREFIX!r}"
)

deleted_projects = []
failed_projects = []
found_projects = []

# Projects
try:
    projects = client.get_projects()
except Exception as e:
    print("Failed to fetch projects:", e)
    sys.exit(2)

for p in projects:
    name = getattr(p, "name", None) or getattr(p, "display_name", None)
    uid = getattr(p, "uid", None) or getattr(p, "id", None)
    found_projects.append((uid, name))
    # Match any project that begins with the PROJECT_PREFIX (covers
    # both 'GroundingDINO Pre-annotations ...' and
    # 'GroundingDINO - <dataset> - <timestamp>').
    if name and name.startswith(PROJECT_PREFIX):
        print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] Deleting project: {name!r} ({uid})")
        try:
            if hasattr(p, "delete"):
                p.delete()
            elif hasattr(client, "delete_project"):
                client.delete_project(uid)
            elif hasattr(client, "_api") and hasattr(client._api, "delete"):
                client._api.delete(f"/projects/{uid}")
            else:
                raise RuntimeError("No delete method available on project or client")
            deleted_projects.append((uid, name))
        except Exception as e:
            print(f"  Failed to delete project {uid}: {e}")
            failed_projects.append((uid, name, str(e)))

# Datasets
deleted_datasets = []
failed_datasets = []
found_datasets = []
try:
    datasets = client.get_datasets()
except Exception as e:
    print("Failed to fetch datasets:", e)
    sys.exit(2)

for d in datasets:
    name = getattr(d, "name", None)
    uid = getattr(d, "uid", None) or getattr(d, "id", None)
    found_datasets.append((uid, name))
    if name and name.startswith(DATASET_PREFIX):
        print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] Deleting dataset: {name!r} ({uid})")
        try:
            if hasattr(d, "delete"):
                d.delete()
            elif hasattr(client, "delete_dataset"):
                client.delete_dataset(uid)
            elif hasattr(client, "_api") and hasattr(client._api, "delete"):
                client._api.delete(f"/datasets/{uid}")
            else:
                raise RuntimeError("No delete method available on dataset or client")
            deleted_datasets.append((uid, name))
        except Exception as e:
            print(f"  Failed to delete dataset {uid}: {e}")
            failed_datasets.append((uid, name, str(e)))

# Summary
print("\nSummary:")
print(f"  Projects found: {len(found_projects)}")
print(f"  Projects deleted: {len(deleted_projects)}")
if deleted_projects:
    for uid, name in deleted_projects:
        print(f"    - {name!r} ({uid})")
print(f"  Project failures: {len(failed_projects)}")
if failed_projects:
    for uid, name, err in failed_projects:
        print(f"    - {name!r} ({uid}) -> {err}")

print(f"\n  Datasets found: {len(found_datasets)}")
print(f"  Datasets deleted: {len(deleted_datasets)}")
if deleted_datasets:
    for uid, name in deleted_datasets:
        print(f"    - {name!r} ({uid})")
print(f"  Dataset failures: {len(failed_datasets)}")
if failed_datasets:
    for uid, name, err in failed_datasets:
        print(f"    - {name!r} ({uid}) -> {err}")

if not (deleted_projects or deleted_datasets or failed_projects or failed_datasets):
    print("No matching projects or datasets were found; nothing to delete.")

# Exit codes: 0 success, 3 partial failures
if failed_projects or failed_datasets:
    sys.exit(3)
else:
    sys.exit(0)
