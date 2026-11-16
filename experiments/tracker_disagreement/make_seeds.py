"""Shared seed file of the two SAM trackers (mini, stdlib): RT-DETR conf>=0.8 anchors (round-3 B1 anchors) + SAM 3 text seeds ("ball", 384 px tiles).

Text seed = top-1 'ball' detection of a probe frame with score >= --min-score and a plausible size (longest side <= --max-side px); dropped when it duplicates an RT-DETR
anchor (within --dup-frames and --dup-px) -- the anchor then carries ``text_confirms``.  Both trackers read the same file.

    python make_seeds.py --inputs /mnt/storage/exp/disagree/in --seedfind <dir with seedfind__<clip>.jsonl> --out seeds.jsonl
"""
import argparse, glob, json, math
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--inputs", type=Path, required=True)
ap.add_argument("--seedfind", type=Path, required=True)
ap.add_argument("--out", type=Path, required=True)
ap.add_argument("--min-score", type=float, default=0.5)
ap.add_argument("--max-side", type=float, default=50.0)
ap.add_argument("--dup-frames", type=int, default=10)
ap.add_argument("--dup-px", type=float, default=30.0)
a = ap.parse_args()

anchors = [json.loads(l) for l in (a.inputs / "seeds_rtdetr.jsonl").read_text().splitlines() if l.strip()]
text, n_probe, n_none = [], 0, 0
for p in sorted(glob.glob(str(a.seedfind / "seedfind__*.jsonl"))):
    for line in open(p):
        r = json.loads(line)
        n_probe += 1
        ds = r["dets"].get("ball", [])
        if not ds or ds[0][4] < a.min_score or max(ds[0][2] - ds[0][0], ds[0][3] - ds[0][1]) > a.max_side:
            n_none += 1
            continue
        b = ds[0]
        text.append({"clip": r["clip"], "frame": r["frame"], "box": [round(v, 1) for v in b[:4]], "conf": round(b[4], 3), "source": "text", "id": f"t{r['frame']}"})
out, dup = list(anchors), 0
for s in text:
    cx, cy = (s["box"][0] + s["box"][2]) / 2, (s["box"][1] + s["box"][3]) / 2
    hit = None
    for an in anchors:
        if an["clip"] == s["clip"] and abs(an["frame"] - s["frame"]) <= a.dup_frames:
            ax, ay = (an["box"][0] + an["box"][2]) / 2, (an["box"][1] + an["box"][3]) / 2
            if math.hypot(ax - cx, ay - cy) <= a.dup_px:
                hit = an
                break
    if hit:
        hit["text_confirms"] = s["conf"]
        dup += 1
    else:
        out.append(s)
out.sort(key=lambda s: (s["clip"], s["frame"]))
a.out.write_text("".join(json.dumps(s) + "\n" for s in out))
print(f"{n_probe} probes, {len(text)} text seeds >= {a.min_score} ({n_probe - len(text)} below / oversize), {dup} duplicate an anchor; {len(anchors)} anchors; seeds file: {len(out)} "
      f"({sum(s['source'] == 'text' for s in out)} text) in {len({s['clip'] for s in out})} clips")
