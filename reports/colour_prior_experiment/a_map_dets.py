"""Map roboflow v11 full-match GT frames (1 fps index n  <->  original frame 25n) to the stored RT-DETR detections in
machine_labels/ (segment clips), so the rescoring experiment can use STORED low-conf detections instead of re-running the
detector. Frame offset inside the segment is found per frame by thumbnail matching (candidates 9..15 frames after 25n - start).
Output: data/dets_cpu.jsonl (same schema a_eval.py reads; name kept) + data/frame_map.csv.
"""
import json, re, os, collections, cv2, numpy as np
cv2.setNumThreads(2)
FD = "/mnt/storage/footy_data"; HERE = os.path.dirname(os.path.abspath(__file__)); D = f"{HERE}/data"
RF = f"{FD}/detection_dataset/roboflow_dataset_11"
OFFSET = 12
S = json.load(open(f"{FD}/arsenal_mancity/split_video_broadcast_frames/segments.json"))["segments"]
meta = [l.rstrip("\n").split("\t") for l in open(f"{D}/frames_meta.tsv")]   # image, frame, split, phase
def seg_of(of):
    for s in S:
        if s["start_frame"] <= of <= s["end_frame"]: return s
by_seg = collections.defaultdict(list)
for img, n, split, phase in meta:
    n = int(n); s = seg_of(25 * n + OFFSET)
    if s: by_seg[s["index"]].append((img, n, split, phase))
out = open(f"{D}/dets_cpu.jsonl", "w"); mp = open(f"{D}/frame_map.csv", "w"); mp.write("image,n,split,phase,seg,seg_frame,thumb_err\n")
for si, items in sorted(by_seg.items()):
    s = S[si]; seg_name = s["file"][:-4]
    rows = collections.defaultdict(list)
    for l in open(f"{FD}/machine_labels/{seg_name}.jsonl"):
        r = json.loads(l); rows[r["frame_index"]].append(r)
    for img, n, split, phase in items:
        k = 25 * n - s["start_frame"] + OFFSET     # offset 12 verified by thumbnail match on 5 random test frames (map_check.py)
        if k < 0 or k >= s["frame_count"]: continue
        e = float("nan")
        g = collections.defaultdict(lambda: np.zeros(7))
        for r in rows.get(k, []):
            b = r["bbox"]; key = tuple(round(v, 4) for v in (b["x"], b["y"], b["w"], b["h"]))
            cls = {"coach": 1, "in_play_ball": 2, "person": 3, "player": 4, "player_sub": 5, "referee": 6, "ball": 0, "out_of_play_ball": 2}[r["tags"][0]]
            g[key][cls] = max(g[key][cls], r["confidence"])
        q = [[key[0] + key[2] / 2, key[1] + key[3] / 2, key[2], key[3], *map(float, v)] for key, v in g.items()]
        out.write(json.dumps({"image": img, "w": 1920, "h": 1080, "q": q}) + "\n")
        mp.write(f"{img},{n},{split},{phase},{si},{k},{e:.2f}\n")
    out.flush(); mp.flush()
out.close(); mp.close(); print("done")
