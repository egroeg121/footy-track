"""Merge the result JSONs of the study into docs/experiments/sam3_offline_labeller.json (mini CPU).

    python assemble.py --out docs/experiments/sam3_offline_labeller.json key=path [key=path ...]
"""
import argparse, json, datetime
ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); ap.add_argument("parts", nargs="+")
a = ap.parse_args()
out = {"generated": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"), "branch": "exp/sam3-offline-labeller",
       "hardware": "desktop-ubuntu RTX 2070 8 GB (sm_75, fp16 autocast, no bf16)", "sam3_src": "facebookresearch/sam3@eef6839", "torch": "2.8.0+cu128"}
for kv in a.parts:
    k, p = kv.split("=", 1)
    out[k] = json.load(open(p))
json.dump(out, open(a.out, "w"), indent=1)
print(a.out, len(json.dumps(out)) // 1024, "KiB")
