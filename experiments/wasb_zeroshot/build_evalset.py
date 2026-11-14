"""Build the WASB zero-shot eval set from human ball GT (CPU, run on the mini).

Merges every GT sidecar of a physical clip (alias stems share one canonical id, see
``footy_track.clip_id``), keeps only human provenance marks, and writes
``evalset.json``::

    {"clips": {<canonical id>: {"video": <abs path>, "video_rel": <file name>,
                                "stems": [...], "ball": {frame: [[x_px, y_px], ...]},
                                "no_ball": [frame, ...]}}}

Human marks: ``in_play_ball`` with provenance ``labeller`` or ``human_checked``
(machine-only ``yolo``/``rtdetr`` marks are dropped). ``no_ball`` rows (explicit
negatives) are kept; a frame that also has a ball mark counts as a ball frame.
``not_broadcast`` and ``out_of_play_ball`` frames are ignored (neither positive nor
negative).
"""
import argparse
import glob
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
from footy_track.clip_id import canonical_clip_id  # noqa: E402

R = "/mnt/storage/footy_data"
W, H = 1920, 1080


def resolve_video(stem):
    if stem == "arsenal_example":
        return os.path.realpath(f"{R}/arsenal_mancity_example_video.mp4")
    for d in ("ballcheck_clips", "clips", "eval_data/clips"):
        p = f"{R}/{d}/{stem}.mp4"
        if os.path.exists(p):
            return os.path.realpath(p)
    return None


def is_human_ball(tags):
    return bool(tags) and tags[0] == "in_play_ball" and ("labeller" in tags or "human_checked" in tags)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    clips = {}
    for f in sorted(glob.glob(f"{R}/ball_gt_marks/*.jsonl")):
        stem = os.path.basename(f)[:-6]
        cid = canonical_clip_id(stem)
        video = resolve_video(stem)
        if video is None:
            print("NO VIDEO", stem)
            continue
        c = clips.setdefault(cid, {"video": video, "video_rel": os.path.basename(video), "stems": [],
                                   "ball": {}, "no_ball": set()})
        if c["video"] != video:
            print("VIDEO CONFLICT", cid, c["video"], video)
        c["stems"].append(stem)
        for line in open(f):
            d = json.loads(line)
            tags = d.get("tags") or []
            fi = d["frame_index"]
            if is_human_ball(tags) and d.get("center"):
                x, y = d["center"]["x"] * W, d["center"]["y"] * H
                lst = c["ball"].setdefault(fi, [])
                if not any(abs(x - px) < 3 and abs(y - py) < 3 for px, py in lst):
                    lst.append([round(x, 2), round(y, 2)])
            elif tags and tags[0] == "no_ball" and len(tags) == 1:
                c["no_ball"].add(fi)
    out = {"clips": {}}
    for cid, c in sorted(clips.items()):
        nb = sorted(f for f in c["no_ball"] if f not in c["ball"])
        if not c["ball"] and not nb:
            continue
        out["clips"][cid] = {"video": c["video"], "video_rel": c["video_rel"], "stems": c["stems"],
                             "ball": {str(k): v for k, v in sorted(c["ball"].items())}, "no_ball": nb}
    json.dump(out, open(a.out, "w"))
    tb = sum(len(c["ball"]) for c in out["clips"].values())
    tn = sum(len(c["no_ball"]) for c in out["clips"].values())
    print(f"{len(out['clips'])} clips, {tb} ball frames, {tn} no_ball frames")
    for cid, c in out["clips"].items():
        print(f"  {cid:36s} ball={len(c['ball']):4d} no_ball={len(c['no_ball']):3d}  {c['stems']}")


if __name__ == "__main__":
    main()
