"""VitTrack positions on every card of a list (mini, stdlib) -> {card id: {arm1: [x, y, score] | null, arm2: [...]}} for ``score_disagreement.py --vit-positions``.

The card id is ``v2_id`` (v2 list) or the v3 index; the position is the raw VitTrack box centre of that frame (a lost track keeps its last box, with its low score), the scorer
applies the 8 px rule.   python make_vit_positions.py --list LIST_V2 --arm1 fused4_arm1.json.gz --arm2 fused4_arm2.json.gz --out v2_vit_positions.json
"""
import argparse, json
from pathlib import Path

import fusion as F

ap = argparse.ArgumentParser()
ap.add_argument("--list", type=Path, required=True)
ap.add_argument("--arm1", type=Path, required=True)
ap.add_argument("--arm2", type=Path, required=True)
ap.add_argument("--out", type=Path, required=True)
a = ap.parse_args()
doc = json.loads(a.list.read_text())
fz = {"arm1": F.load(a.arm1), "arm2": F.load(a.arm2)}
out = {}
for i, it in enumerate(doc["items"]):
    c = it.get("clip_run_id") or it["clip"].replace("arsenal_nottinghamforest", "arsenal_norwich")
    rec = {}
    for arm, d in fz.items():
        t4 = d["clips"][c].get("t4")
        t = it["frame_index"]
        rec[arm] = [t4["raw_x"][t], t4["raw_y"][t], t4["score"][t]] if t4 and t4["raw_x"][t] is not None else None
    out[it.get("v2_id") or str(it.get("v3_index", i))] = rec
a.out.write_text(json.dumps(out))
print(len(out), "cards;", {arm: sum(1 for v in out.values() if v[arm]) for arm in fz}, "with a VitTrack position")
