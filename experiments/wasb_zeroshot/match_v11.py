"""Which GT frames are (pixel-)identical to an image in the RT-DETR training data (Roboflow v11)?

Filename matching cannot answer this: v11 images are named either ``<alias>_segNNN_<frame>``
(clip + frame) or ``arsenal_mancity_20250925_<frame>`` (frame index into the full match video,
which cannot be mapped to a segment by name; todo T-33). So we compare content: 64x36 grey
thumbnails of every v11 image (train/valid/test, 1920x1080, no augmentation) against the
thumbnail of every GT frame (decoded from the clip). A near-zero mean abs difference is the
same frame. Writes ``v11_match.json``: {clip: {frame: {"split": train|valid|test|none, "d": min diff,
"image": file}}}.

Run (CPU): uv run --with numpy --with opencv-python-headless python match_v11.py ...
"""
import argparse
import glob
import json
import os

import cv2
import numpy as np

V11 = "/mnt/storage/footy_data/detection_dataset/roboflow_dataset_11"
TW, TH = 64, 36


def thumb(bgr):
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    return cv2.resize(g, (TW, TH), interpolation=cv2.INTER_AREA).astype(np.float32).ravel()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--evalset", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--clip-filter", default="mancity", help="substring of canonical id to check")
    a = ap.parse_args()
    files, names, splits = [], [], []
    for sp in ("train", "valid", "test"):
        for f in sorted(glob.glob(f"{V11}/{sp}/images/*")):
            files.append(f)
            splits.append(sp)
    T = np.zeros((len(files), TW * TH), np.float32)
    for i, f in enumerate(files):
        T[i] = thumb(cv2.imread(f))
    print("v11 thumbs", T.shape)
    ev = json.load(open(a.evalset))["clips"]
    res = {}
    for cid, c in ev.items():
        if a.clip_filter not in cid:
            continue
        want = sorted({int(k) for k in c["ball"]} | set(c["no_ball"]))
        cap = cv2.VideoCapture(c["video"])
        wset, i, last = set(want), 0, max(want)
        res[cid] = {}
        while i <= last:
            ok, bgr = cap.read()
            if not ok:
                break
            if i in wset:
                t = thumb(bgr)
                d = np.abs(T - t).mean(1)
                j = int(d.argmin())
                res[cid][i] = {"split": splits[j], "d": round(float(d[j]), 3), "image": os.path.basename(files[j])}
            i += 1
        cap.release()
        print(cid, len(res[cid]), "frames checked", flush=True)
    json.dump(res, open(a.out, "w"))
    ds = sorted(v["d"] for r in res.values() for v in r.values())
    print("min-diff quantiles", [ds[int(len(ds) * q)] for q in (0.0, 0.05, 0.1, 0.2, 0.5, 0.9)])


if __name__ == "__main__":
    main()
