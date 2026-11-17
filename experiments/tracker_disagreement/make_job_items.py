"""Items file for ``labelling_jobs submit --kind ball_check`` from the v2 list (stdlib).

A ball_check job item is only a pointer ``{clip, frame_index, box_index, source: "list"}``; the card itself (bbox, other_bbox, hidden keys) comes from the dataset's list file, so
the job file carries no stratum / tracker information.  Prints the exact submit command.
"""
import argparse, json
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--set", default="v2", choices=["v2", "v3"])
ap.add_argument("--list", type=Path)
ap.add_argument("--out", type=Path)
a = ap.parse_args()
V = {"v2": ("ball_check_list_v2_disagree.json", "job_items.json", "nottinghamforest_disagree", "Forest: three-tracker disagreement (blind, {n} cards)", "v2-disagree",
            "Blind Ball Check: which of three ball trackers is right where they disagree, plus agreed-frame precision (exp/tracker-disagreement)"),
     "v3": ("ball_check_list_v3_vittrack.json", "job_items_v3.json", "nottinghamforest_vittrack", "Forest: VitTrack (labeller tracker) top-up (blind, {n} cards)", "v3-vittrack",
            "Blind Ball Check: where does the labeller autodetect + VitTrack go wrong, and can its agreement with the other trackers be trusted (exp/tracker-disagreement)")}[a.set]
a.list = a.list or Path("/mnt/storage/footy_data/nottinghamforest_label_queue") / V[0]
a.out = a.out or Path("/mnt/storage/exp/disagree") / V[1]
items = json.loads(a.list.read_text())["items"]
a.out.write_text(json.dumps([{"clip": it["clip"], "frame_index": it["frame_index"], "box_index": i, "source": "list"} for i, it in enumerate(items)]))
print(len(items), "items ->", a.out)
print("cd /home/george/footy-track-main && PYTHONPATH=src python -m footy_track.scripts.labelling_jobs submit \\\n"
      f"  --title '{V[3].format(n=len(items))}' --kind ball_check --dataset {V[2]} --device phone \\\n"
      f"  --items {a.out} --key exp/tracker-disagreement/{V[4]} \\\n"
      f"  --why '{V[5]}'")
