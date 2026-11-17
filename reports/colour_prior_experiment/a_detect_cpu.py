"""CPU low-confidence detections WITH the full per-class sigmoid score vector of every RT-DETR query.

This ultralytics version's RT-DETR head does top-k over (query x class), so the same box can be emitted several
times with different classes (e.g. player 0.6 and referee 0.3). We keep every (box, class, conf>=0.05) row; a_eval
regroups rows with identical boxes into one per-class score vector. Boxes: normalised centre-xywh.
Thread-capped. Resumable (appends, skips done frames). One JSON row per frame.
"""
import argparse, json, os, sys, time
import cv2
cv2.setNumThreads(2)
import torch
torch.set_num_threads(2)
from ultralytics import YOLO

ap = argparse.ArgumentParser()
ap.add_argument("--list", required=True, help="text file: one image path per line")
ap.add_argument("--weights", default="/mnt/storage/models/rtdetr_1920_v11/best.pt")
ap.add_argument("--out", required=True)
ap.add_argument("--conf", type=float, default=0.05)
ap.add_argument("--imgsz", type=int, default=1920)
a = ap.parse_args()

paths = [l.strip() for l in open(a.list) if l.strip()]
done = set()
if os.path.exists(a.out):
    for l in open(a.out):
        try: done.add(json.loads(l)["image"])
        except Exception: pass
model = YOLO(a.weights)
print("names", model.names, "todo", len([p for p in paths if os.path.basename(p) not in done]), flush=True)
with open(a.out, "a") as fh:
    for i, p in enumerate(paths):
        b = os.path.basename(p)
        if b in done: continue
        t0 = time.time()
        r = model.predict(p, conf=a.conf, imgsz=a.imgsz, verbose=False, device="cpu")[0]
        bx = r.boxes
        rows = [[round(float(v), 5) for v in xy] + [round(float(c), 4), int(k)] for xy, c, k in zip(bx.xywhn.tolist(), bx.conf.tolist(), bx.cls.tolist())]
        fh.write(json.dumps({"image": b, "w": r.orig_shape[1], "h": r.orig_shape[0], "q": rows}) + "\n"); fh.flush()
        print(i, len(rows), f"{time.time()-t0:.1f}s", flush=True)
