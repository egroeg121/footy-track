"""Seeds file for sam3_label.py / the ``sam3-label-match`` flow from RT-DETR machine labels (mini, stdlib only).

One seed per ``--spacing`` original frames: the highest-confidence ball detection with conf >= ``--thr`` inside each window (the realistic seed protocol of the SAM 3
study).  ``--labels`` is a directory of ``<clip>.jsonl`` machine labels (frame_index = original frame; Bournemouth round-3 labels use frame_index = original // 2,
pass ``--index-stride 2``).  Human clicks / overview seeds can simply be appended to the file in the same format.

    python make_seeds.py --labels /mnt/storage/footy_data/machine_labels --video-dir DIR --thr 0.8 --spacing 100 > seeds.jsonl
"""
import argparse, json, subprocess
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--labels", type=Path, required=True)
ap.add_argument("--video-dir", type=Path, required=True, help="to read the frame size of <clip>.mp4 (labels are normalised)")
ap.add_argument("--video-map", type=Path, default=None, help="optional json {clip: video file name} when files are not named <clip>.mp4")
ap.add_argument("--thr", type=float, default=0.8)
ap.add_argument("--spacing", type=int, default=100)
ap.add_argument("--index-stride", type=int, default=1)
ap.add_argument("--clip-prefix", default="", help="only labels files starting with this")
a = ap.parse_args()
for p in sorted(a.labels.glob(f"{a.clip_prefix}*.jsonl")):
    vm = json.loads(a.video_map.read_text()) if a.video_map else {}
    v = a.video_dir / vm.get(p.stem, p.stem + ".mp4")
    if not v.exists():
        continue
    w, h = (int(x) for x in subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height", "-of", "csv=p=0", str(v)],
                                           capture_output=True, text=True).stdout.strip().split(",")[:2])
    best: dict[int, tuple] = {}
    for line in p.read_text().splitlines():
        r = json.loads(line)
        if not any(t in ("in_play_ball", "ball") for t in r["tags"]) or r["confidence"] < a.thr:
            continue
        f = r["frame_index"] * a.index_stride
        k = f // a.spacing
        if k not in best or r["confidence"] > best[k][0]:
            b = r["bbox"]
            best[k] = (r["confidence"], f, [round(b["x"] * w, 1), round(b["y"] * h, 1), round((b["x"] + b["w"]) * w, 1), round((b["y"] + b["h"]) * h, 1)])
    for k, (conf, f, box) in sorted(best.items()):
        print(json.dumps({"clip": p.stem, "frame": f, "box": box, "conf": conf, "source": "rtdetr"}))
