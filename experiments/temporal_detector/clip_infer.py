"""Full-clip, every-frame ball candidates for the trajectory-solver comparison (GPU + CPU, desktop).

For every frame t of a clip: decode with the ffmpeg CLI (`-fps_mode passthrough`, no cv2 decoding), run the production
RT-DETR-L at 1280 (ball candidates + player boxes), build the 5-channel input exactly like the dataset builder
(LK alignment from `footy_track.calibration.temporal` of frames t-2 / t+2 into frame t, person boxes as masks, neighbour
replaced by gray(t) on clip edge / cut (Bhattacharyya > 0.2) / alignment failure) and run the YOLO arms A and B
(seeds 0, 1). Candidates (ball class, conf >= 0.1, boxes in the 1920x1080 frame) are written per clip to
``<out>/<clip>.npz`` in CSR layout so the harness (`footy_track.ball_offline`) can consume them.

    python clip_infer.py --clips arsenal_example:/path/a.mp4 seg010:/path/b.mp4 --runs ~/exp/temporal/runs/v1 \
        --rtdetr best.pt --out ~/exp/temporal/clips_out/v1 --src ~/exp/temporal/code/src

``--verify`` additionally compares the ffmpeg frame t with the dataset TIFFs of that clip (frame-index check).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np

from common import read_tiff5
from evaluate import YoloRunner

W, H = 1920, 1080
CUT_THRESHOLD = 0.2
STATUS = {"ok": 0, "edge": 1, "cut": 2, "align_fail": 3}
PERSON_IDS = {1, 3, 4, 5, 6}
ARMS = [("A", 0), ("A", 1), ("B", 0), ("B", 1)]


def ffprobe_frames(path: Path) -> int:
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_packets", "-show_entries",
                          "stream=nb_read_packets", "-of", "csv=p=0", str(path)], capture_output=True, text=True, check=True)
    return int(out.stdout.strip().split(",")[0])


def read_frames(path: Path):
    """Every frame, in decode order, with the ffmpeg CLI (no vsync duplication/dropping, no cv2 decoder)."""
    cmd = ["ffmpeg", "-v", "error", "-i", str(path), "-fps_mode", "passthrough", "-f", "rawvideo", "-pix_fmt", "bgr24", "-"]
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, bufsize=1 << 26)
    n = W * H * 3
    try:
        while True:
            buf = p.stdout.read(n)
            if len(buf) < n:
                break
            yield np.frombuffer(buf, np.uint8).reshape(H, W, 3).copy()
    finally:
        p.stdout.close()
        p.wait()


class RtdetrFull:
    """Production RT-DETR-L at 1280, all classes, conf >= 0.1 (as machine_labels)."""

    def __init__(self, weights: str):
        from ultralytics import YOLO

        self.m = YOLO(weights)

    def __call__(self, bgr: np.ndarray):
        r = self.m.predict(bgr, imgsz=1280, conf=0.1, max_det=300, verbose=False, device=0, half=True)[0]
        b = r.boxes
        if b is None or len(b) == 0:
            z = np.zeros((0, 5))
            return z, np.zeros((0, 4)), np.zeros((0, 4))
        xywh, cf, cl = b.xywh.cpu().numpy(), b.conf.cpu().numpy(), b.cls.cpu().numpy().astype(int)
        balls = np.concatenate([xywh[cl == 2], cf[cl == 2, None]], axis=1)
        persons = xywh[np.isin(cl, list(PERSON_IDS)) & (cf >= 0.3)]
        players = xywh[cl == 4]
        return balls, persons, players


def align_neighbour(frames, hd, t, tn, n_frames, boxes_norm, scale=480):
    """frames: dict idx->bgr; hd: dict g->dist(g, g+1). Mirrors build_multiframe_dataset.align_neighbour."""
    from footy_track.multiframe.align import estimate_pair_lk, warp_to_ref

    if tn < 0 or tn >= n_frames:
        return None, "edge"
    lo, hi = sorted((t, tn))
    if max(hd[g] for g in range(lo, hi)) > CUT_THRESHOLD:
        return None, "cut"
    a, b = frames[t], frames[tn]
    pm = estimate_pair_lk(a, b, scale=scale, boxes_a=boxes_norm or None, boxes_b=boxes_norm or None)
    if not pm.ok:
        return None, "align_fail"
    return cv2.cvtColor(warp_to_ref(b, pm.H_px), cv2.COLOR_BGR2GRAY), "ok"


def to_csr(per_frame: list[np.ndarray], ncol: int):
    ptr = np.zeros(len(per_frame) + 1, np.int64)
    ptr[1:] = np.cumsum([len(x) for x in per_frame])
    arr = np.concatenate(per_frame, axis=0) if ptr[-1] else np.zeros((0, ncol))
    return ptr, arr.astype(np.float32)


def process_clip(name: str, video: Path, rt: RtdetrFull, yolos: dict, out: Path, k: int, limit: int):
    from footy_track.multiframe.align import hist_distance

    n = ffprobe_frames(video)
    if limit:
        n = min(n, limit)
    t0 = time.time()
    frames: dict[int, np.ndarray] = {}
    hd: dict[int, float] = {}
    gen = read_frames(video)
    decoded = 0

    def decode_upto(i):
        nonlocal decoded
        while decoded <= i and decoded < n:
            fr = next(gen)
            frames[decoded] = fr
            if decoded > 0:
                hd[decoded - 1] = hist_distance(frames[decoded - 1], fr)
            decoded += 1

    ref_b, players_l, statuses = [], [], np.zeros((n, 2), np.int8)
    cand = {key: [] for key in yolos}
    for t in range(n):
        decode_upto(min(t + k, n - 1))
        a = frames[t]
        balls, persons, players = rt(a)
        ref_b.append(balls[balls[:, 4] >= 0.1] if len(balls) else balls)
        players_l.append(players)
        boxes_norm = [(x / W, y / H, w / W, h / H) for x, y, w, h in persons]
        gray_t = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)
        planes = []
        for j, tn in enumerate((t - k, t + k)):
            g, st = align_neighbour(frames, hd, t, tn, n, boxes_norm)
            statuses[t, j] = STATUS[st]
            planes.append(g if g is not None else gray_t)
        img5 = np.concatenate([planes[0][..., None], a, planes[1][..., None]], axis=2)
        for key, runner in yolos.items():
            dets = runner(img5, conf=0.1)
            cand[key].append(np.array([d for d in dets if d[4] >= 0.1], np.float32).reshape(-1, 5))
        for i in [i for i in frames if i < t - k - 1]:  # keep a window big enough for t-k .. t+k
            del frames[i]
        for i in [i for i in hd if i < t - k - 1]:
            del hd[i]
        if t % 100 == 0:
            print(f"{name} {t}/{n} {time.time() - t0:.0f}s", flush=True)
    arrays = {"n_frames": np.array(n), "status": statuses}
    for key, lst in {"ref": ref_b, **cand}.items():
        ptr, arr = to_csr(lst, 5)
        arrays[f"{key}_ptr"], arrays[f"{key}_det"] = ptr, arr
    ptr, arr = to_csr(players_l, 4)
    arrays["players_ptr"], arrays["players_xywh"] = ptr, arr
    out.mkdir(parents=True, exist_ok=True)
    tmp = out / f"{name}.tmp.npz"
    np.savez_compressed(tmp, **arrays)
    tmp.replace(out / f"{name}.npz")
    print(f"{name}: {n} frames in {time.time() - t0:.0f}s", flush=True)
    return n


def verify_frames(name: str, video: Path, dataset: Path, max_images: int = 6) -> dict:
    """Compare ffmpeg frame t with TIFF BGR(t) of dataset images of this video (frame-index / colour check)."""
    metas = [json.loads(line) for line in open(dataset / "meta.jsonl")]
    sel = [m for m in metas if m.get("clip_id") == name and m.get("frame") is not None][:max_images]  # name = canonical clip id
    res = []
    for m in sel:
        sp = "val" if m["split"] == "valid" else m["split"]
        img = read_tiff5(dataset / "images" / sp / (m["id"] + ".tiff"))[..., 1:4].astype(np.float32)
        t = m["frame"]
        cands = {}
        for g, fr in zip(range(t - 1, t + 2), _frames_range(video, t - 1, 3), strict=False):
            cands[g - t] = float(np.abs(fr.astype(np.float32) - img).mean())
        res.append({"id": m["id"], "frame": t, "mad_by_offset": cands, "best_offset": min(cands, key=cands.get)})
    return {"clip": name, "n": len(res), "all_best_offset_0": all(r["best_offset"] == 0 for r in res), "items": res}


def _frames_range(video: Path, start: int, count: int):
    cmd = ["ffmpeg", "-v", "error", "-i", str(video), "-fps_mode", "passthrough", "-vf", f"select='between(n,{max(start, 0)},{start + count - 1})'",
           "-f", "rawvideo", "-pix_fmt", "bgr24", "-"]
    raw = subprocess.run(cmd, capture_output=True, check=True).stdout
    n = W * H * 3
    fr = [np.frombuffer(raw[i * n:(i + 1) * n], np.uint8).reshape(H, W, 3) for i in range(len(raw) // n)]
    if start < 0:  # pad so offsets line up
        fr = [fr[0]] * (-start) + fr
    return fr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", nargs="+", required=True, help="name:video_path")
    ap.add_argument("--runs", type=Path, required=True, help="dir with A_s0/weights/last.pt ...")
    ap.add_argument("--rtdetr", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--src", type=Path, required=True, help="dir containing footy_track/")
    ap.add_argument("--dataset", type=Path, default=None, help="k2 dataset for --verify")
    ap.add_argument("--keys", nargs="+", default=["A_s0", "A_s1", "B_s0", "B_s1"],
                    help="run names under --runs; the first letter is the arm (A 3ch, B 5ch); round 2: Ap_s0 ...")
    ap.add_argument("--k", type=int, default=2)
    ap.add_argument("--limit", type=int, default=0, help="debug: first N frames per clip")
    ap.add_argument("--verify", action="store_true")
    a = ap.parse_args()
    sys.path.insert(0, str(a.src))
    cv2.setNumThreads(4)
    rt = RtdetrFull(a.rtdetr)
    yolos = {key: YoloRunner(str(a.runs / key / "weights" / "last.pt"), key[0]) for key in a.keys}
    report = []
    for spec in a.clips:
        name, video = spec.split(":", 1)
        video = Path(video)
        if a.verify and a.dataset:
            v = verify_frames(name, video, a.dataset)
            print("VERIFY", json.dumps({k: v[k] for k in ("clip", "n", "all_best_offset_0")}), flush=True)
            report.append(v)
        if (a.out / f"{name}.npz").exists():
            print("skip", name)
            continue
        process_clip(name, video, rt, yolos, a.out, a.k, a.limit)
    if report:
        (a.out / "verify_frames.json").write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
