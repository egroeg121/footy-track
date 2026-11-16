"""Items file for ``labelling_jobs submit --kind ball_check`` from the v2 list (stdlib).

A ball_check job item is only a pointer ``{clip, frame_index, box_index, source: "list"}``; the card itself (bbox, other_bbox, hidden keys) comes from the dataset's list file, so
the job file carries no stratum / tracker information.  Prints the exact submit command.
"""
import argparse, json
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--list", type=Path, default=Path("/mnt/storage/footy_data/norwich_label_queue/ball_check_list_v2_disagree.json"))
ap.add_argument("--out", type=Path, default=Path("/mnt/storage/exp/disagree/job_items.json"))
a = ap.parse_args()
items = json.loads(a.list.read_text())["items"]
a.out.write_text(json.dumps([{"clip": it["clip"], "frame_index": it["frame_index"], "box_index": i, "source": "list"} for i, it in enumerate(items)]))
print(len(items), "items ->", a.out)
print("cd /home/george/footy-track-main && PYTHONPATH=src python -m footy_track.scripts.labelling_jobs submit \\\n"
      "  --title 'Norwich: three-tracker disagreement (blind, 200 cards)' --kind ball_check --dataset norwich_disagree --device phone \\\n"
      f"  --items {a.out} --key exp/tracker-disagreement/v2-disagree \\\n"
      "  --why 'Blind Ball Check: which of three ball trackers is right where they disagree, plus agreed-frame precision (exp/tracker-disagreement)'")
