"""Add the ``nottinghamforest_disagree`` dataset to the live labeller registry (stdlib).  DRY RUN unless --apply.

The main labeller (:8011) resolves ``?dataset=<name>`` from /mnt/storage/footy_data/labeller_datasets.json (re-read on change: no restart).  The v2 batch is a Ball Check *list*,
and a list belongs to a dataset, so the batch gets its own dataset entry: the Norwich keyframe-dense clips, the v2 list, and an EMPTY marks dir of its own (verdict logs
and any "corrected" boxes land there, never in ``ball_gt_marks_nottinghamforest`` = George's held-out Norwich GT).

    python register_dataset.py            # prints the entry and the diff
    python register_dataset.py --apply    # atomic write (tmp + rename), other entries untouched
"""
import argparse, json, os, tempfile
from pathlib import Path

REG = Path(os.environ.get("FOOTY_LABELLER_DATASETS", "/mnt/storage/footy_data/labeller_datasets.json"))
D = "/mnt/storage/footy_data"
ap = argparse.ArgumentParser()
ap.add_argument("--apply", action="store_true")
ap.add_argument("--set", default="v2", choices=["v2", "v3"], help="v2: the live disagreement batch; v3: the VitTrack top-up")
a = ap.parse_args()
if a.set == "v2":
    NAME = "nottinghamforest_disagree"
    ENTRY = {"label": "Arsenal v Nottingham Forest: tracker disagreement (blind)", "clips_dir": f"{D}/arsenal_nottinghamforest/split_video_broadcast_frames_kf10",
             "marks_dir": f"{D}/ball_gt_marks_nottinghamforest_disagree", "ball_check_list": f"{D}/nottinghamforest_label_queue/ball_check_list_v2_disagree.json"}
else:
    NAME = "nottinghamforest_vittrack"
    ENTRY = {"label": "Arsenal v Nottingham Forest: VitTrack top-up (blind)", "clips_dir": f"{D}/arsenal_nottinghamforest/split_video_broadcast_frames_kf10",
             "marks_dir": f"{D}/ball_gt_marks_nottinghamforest_vittrack", "ball_check_list": f"{D}/nottinghamforest_label_queue/ball_check_list_v3_vittrack.json"}
doc = json.loads(REG.read_text())
if not Path(ENTRY["ball_check_list"]).exists():
    raise SystemExit("the list does not exist yet: " + ENTRY["ball_check_list"])
print("existing:", NAME in doc["datasets"], "->", json.dumps({NAME: ENTRY}, indent=1))
if not a.apply:
    print("(dry run; --apply to write", REG, ")")
    raise SystemExit(0)
doc["datasets"][NAME] = ENTRY
Path(ENTRY["marks_dir"]).mkdir(parents=True, exist_ok=True)
fd, tmp = tempfile.mkstemp(prefix=REG.name + ".", suffix=".tmp", dir=REG.parent)
with os.fdopen(fd, "w") as f:
    json.dump(doc, f, indent=2)
    f.write("\n")
    f.flush()
    os.fsync(f.fileno())
os.chmod(tmp, 0o600)
os.replace(tmp, REG)
print("registered", NAME)
