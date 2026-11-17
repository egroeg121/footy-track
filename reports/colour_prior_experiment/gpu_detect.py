"""Run the production RT-DETR detector at LOW confidence over the roboflow v11 full-match frames.

Runs on the desktop (footy-venv). Images: <ds>/{train,valid,test}/images/arsenal_mancity_20250925_NNNNNN_*.jpg
and valid arsenal_example_*. Output JSONL, one row per frame:
 {split, image, frame, w, h, dets: [[cls_id, conf, x, y, w, h(normalised top-left xywh)], ...]}
Settings mirror the machine_labels run: imgsz 1920, iou 0.9, but conf 0.05.
"""
import argparse, json, os, re, sys
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--ds", required=True)
ap.add_argument("--weights", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--conf", type=float, default=0.05)
a = ap.parse_args()

from ultralytics import YOLO  # noqa: E402

model = YOLO(a.weights)
names = dict(model.names)
rows_todo = []
for split in ("train", "valid", "test"):
    for f in sorted(os.listdir(f"{a.ds}/{split}/images")):
        if re.match(r"arsenal_mancity_20250925_\d{6}_", f) or f.startswith("arsenal_example_"):
            rows_todo.append((split, f))
print(f"{len(rows_todo)} images; names={names}", flush=True)
out = Path(a.out)
tmp = out.with_suffix(".tmp")
with open(tmp, "w") as fh:
    fh.write(json.dumps({"names": names, "conf": a.conf}) + "\n")
    for i, (split, f) in enumerate(rows_todo):
        r = model.predict(f"{a.ds}/{split}/images/{f}", conf=a.conf, iou=0.9, imgsz=1920, verbose=False)[0]
        h, w = r.orig_shape[:2]
        dets = []
        for c, s, (x1, y1, x2, y2) in zip(r.boxes.cls.tolist(), r.boxes.conf.tolist(), r.boxes.xyxy.tolist()):
            dets.append([int(c), round(s, 4), x1 / w, y1 / h, (x2 - x1) / w, (y2 - y1) / h])
        fh.write(json.dumps({"split": split, "image": f, "w": w, "h": h, "dets": dets}) + "\n")
        if i % 20 == 0:
            print(i, len(dets), flush=True)
os.replace(tmp, out)
print("done", flush=True)
