"""Pick the ~20 Norwich clips of the tracker-disagreement run and write the shared inputs (mini, stdlib only).

Clips are chosen from those that have NO dense-labelling window (so the +-25 frame exclusion is satisfied trivially; the fusion step still applies the mask),
at random (seeded) until ~``--frames`` frames.  Writes into --out:
  clips.json        {"clips": {clip: {"n_frames", "video"}}, "exclude": {clip: [[lo, hi], ...]}, "judged": {clip: [frame, ...]}}
  seeds_rtdetr.jsonl  RT-DETR conf>=0.8 anchors (the round-3 B1 anchors of <round3>/norwich/spec.json), thinned to >= ``--anchor-gap`` frames
  probes.json       {clip: [frames]} text-seed probe frames (every ``--probe-every``)
"""
import argparse, csv, glob, json, os, random
from pathlib import Path

R3 = Path("/mnt/storage/exp/temporal/round3")
NW = "/mnt/storage/footy_data/arsenal_norwich"
Q = Path("/mnt/storage/footy_data/norwich_label_queue")
GT = Path("/mnt/storage/footy_data/ball_gt_marks_norwich")

ap = argparse.ArgumentParser()
ap.add_argument("--out", type=Path, required=True)
ap.add_argument("--frames", type=int, default=6500)
ap.add_argument("--n-clips", type=int, default=20)
ap.add_argument("--seed", type=int, default=7)
ap.add_argument("--anchor-gap", type=int, default=40)
ap.add_argument("--probe-every", type=int, default=75)
a = ap.parse_args()
a.out.mkdir(parents=True, exist_ok=True)

spec = json.loads((R3 / "norwich" / "spec.json").read_text())
exclude: dict[str, list] = {}
for r in csv.DictReader(open(Q / "windows.csv")):
    exclude.setdefault(r["clip"], []).append([int(r["start_frame"]) - 25, int(r["end_frame"]) + 25])
free = sorted(c for c in spec["clips"] if c not in exclude)
rng = random.Random(a.seed)
rng.shuffle(free)
chosen, tot = [], 0
for c in free:
    if len(chosen) >= a.n_clips or tot >= a.frames:
        break
    chosen.append(c)
    tot += spec["clips"][c]["n_frames"]
chosen.sort()

judged: dict[str, set] = {}
for p in glob.glob(str(GT / "ball_checks" / "*.jsonl")):
    for line in open(p):
        try:
            r = json.loads(line)
        except ValueError:
            continue
        if r.get("verdict") is not None:
            judged.setdefault(r["clip"], set()).add(int(r["frame_index"]))

clips = {c: {"n_frames": spec["clips"][c]["n_frames"], "video": f"{NW}/split_video_broadcast_frames/{c}.mp4"} for c in chosen}
(a.out / "clips.json").write_text(json.dumps({"clips": clips, "exclude": {c: exclude.get(c, []) for c in chosen},
                                              "judged": {c: sorted(judged.get(c, [])) for c in chosen}}, indent=1))
# anchors: round-3 B1 nodes with RT-DETR conf>=0.8 (spec tracks hold one entry per direction)
seeds, probes = [], {}
for c in chosen:
    anc = {}
    for t in spec["clips"][c]["tracks"]:
        anc[t["t"]] = t["box"]
    last = -10**9
    for t in sorted(anc):
        if t - last >= a.anchor_gap:
            seeds.append({"clip": c, "frame": t, "box": [round(v, 1) for v in anc[t]], "source": "rtdetr", "id": f"r{t}"})
            last = t
    n = spec["clips"][c]["n_frames"]
    probes[c] = list(range(a.probe_every // 2, n, a.probe_every))
(a.out / "seeds_rtdetr.jsonl").write_text("".join(json.dumps(s) + "\n" for s in seeds))
(a.out / "probes.json").write_text(json.dumps(probes))
print(f"{len(chosen)} clips, {tot} frames, {len(seeds)} rtdetr anchors, {sum(map(len, probes.values()))} probes; judged frames in these clips: {sum(len(judged.get(c, ())) for c in chosen)}")
print(chosen)

# spec for the sam3-seedfind flow (text-seed probes): probe frames as "gt" keys (values unused), sparse decode
spec = {"clips": {c: {"video": clips[c]["video"], "set": "disagree", "w": 1920, "h": 1080, "stride": 1, "n_frames": clips[c]["n_frames"], "sparse": True,
                      "gt": {str(f): None for f in probes[c]}, "gt_odd": {}, "no_ball": [], "seeds_oracle": [], "seeds_rtdetr": []} for c in chosen}}
Path("/mnt/storage/exp/sam3/spec_disagree.json").write_text(json.dumps(spec))
print("spec written")
