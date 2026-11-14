"""GPU step of the machine-anchor experiment: SAM2 tracks from detector/B1 anchors.

Reads ``spec.json`` (written by machine_prep.py): per clip a list of
``{t, dir, reach, box}`` (box = detector box at the anchor, px at 1080p). Each track is an
independent inference state (fwd: frames t..t+reach, bwd: t-reach..t, reverse propagation), exactly
as in sam2_tracks.py. Output: ``mtracks__<clip>__<model>__<variant>.json`` with the same per-frame
record format ``[present, cx, cy, area, object_score]``. Resumable.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gt as G  # noqa: E402
import sam2_tracks as T  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", required=True)
    ap.add_argument("--videos", required=True, help="dir with <clip>.mp4")
    ap.add_argument("--out", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--model", default="small")
    ap.add_argument("--variant", default="crop512")
    ap.add_argument("--dtype", default="fp16")
    ap.add_argument("--clips", default="")
    ap.add_argument("--max-tracks", type=int, default=0)
    args = ap.parse_args()

    import torch
    spec = json.loads(Path(args.spec).read_text())
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    only = set(args.clips.split(",")) if args.clips else None
    runner = None
    for clip, cs in sorted(spec["clips"].items()):
        if only and clip not in only:
            continue
        if not cs["tracks"]:
            continue
        dst = out / f"mtracks__{clip}__{args.model}__{args.variant}.json"
        doc = json.loads(dst.read_text()) if dst.exists() else {
            "meta": {"clip": clip, "model": args.model, "variant": args.variant, "dtype": args.dtype,
                     "host": os.uname().nodename, "gpu": torch.cuda.get_device_name(0),
                     "torch": torch.__version__, "n_frames": cs["n_frames"], "full": cs["full"]},
            "tracks": []}
        have = {(t["anchor"], t["dir"]): t for t in doc["tracks"]}
        todo = [t for t in cs["tracks"] if not ((t["t"], t["dir"]) in have and have[(t["t"], t["dir"])]["reach"] >= t["reach"])]
        if args.max_tracks:
            todo = todo[:args.max_tracks]
        if not todo:
            print(f"{clip}: nothing to do", flush=True)
            continue
        if runner is None:
            runner = T.Runner(args.model, args.ckpt, args.variant, args.dtype, "box", None)
            torch.cuda.reset_peak_memory_stats()
        wanted = set()
        for t in todo:
            lo, hi = (t["t"], t["t"] + t["reach"]) if t["dir"] == "fwd" else (t["t"] - t["reach"], t["t"])
            wanted.update(range(max(0, lo), min(cs["n_frames"] - 1, hi) + 1))
        vid = str(Path(args.videos) / f"{clip}.mp4")
        w, h = T.video_size(vid)
        sx, sy = w / 1920.0, h / 1080.0   # spec boxes/positions are 1080p-equivalent px
        t0 = time.time()
        frames = T.read_frames(vid, wanted, w, h)
        print(f"{clip}: decoded {len(wanted)} frames in {time.time()-t0:.1f}s; {len(todo)} tracks", flush=True)
        run = G.Run(clip, 0, cs["n_frames"] - 1, {}, {})
        t_clip = time.time()
        steps = 0
        for i, tr in enumerate(sorted(todo, key=lambda x: (x["t"], x["dir"]))):
            b = tr["box"]
            run.boxes = {tr["t"]: (b[0] * sx, b[1] * sy, b[2] * sx, b[3] * sy)}
            rec = runner.track(frames, run, tr["t"], tr["dir"], tr["reach"])
            if rec is None:
                continue
            rec["run"] = 0
            if (sx, sy) != (1.0, 1.0):
                for fr in rec["frames"].values():
                    if fr[0]:
                        fr[1], fr[2] = round(fr[1] / sx, 2), round(fr[2] / sy, 2)
            doc["tracks"] = [x for x in doc["tracks"] if (x["anchor"], x["dir"]) != (tr["t"], tr["dir"])]
            doc["tracks"].append(rec)
            steps += rec["n_frames"]
            if (i + 1) % 25 == 0 or i + 1 == len(todo):
                tf = sum(x["n_frames"] for x in doc["tracks"])
                ts = sum(x["sec"] for x in doc["tracks"])
                doc["perf"] = {"peak_vram_alloc_mb": round(torch.cuda.max_memory_allocated() / 2**20),
                               "peak_vram_reserved_mb": round(torch.cuda.max_memory_reserved() / 2**20),
                               "frames_propagated": tf, "seconds": round(ts, 1), "sec_per_frame": round(ts / max(tf, 1), 4)}
                tmp = dst.with_name(dst.name + ".tmp")
                tmp.write_text(json.dumps(doc))
                os.replace(tmp, dst)
                print(f"  {clip} {i+1}/{len(todo)} tracks, {steps} steps, {time.time()-t_clip:.0f}s, {doc['perf']}", flush=True)
        del frames


if __name__ == "__main__":
    main()
