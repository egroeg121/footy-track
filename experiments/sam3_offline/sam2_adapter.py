"""SAM2.1-small crop512 baseline on the same spec / seeds / reach as sam3_track.py (runs in ~/sam2-exp/venv, reuses ~/sam2-exp/code/sam2_tracks.py as is).

The only differences to ~/sam2-exp/code/machine_tracks.py (the round-3 S1 tracker): seeds and windows come from the SAM 3 evaluation spec, the frame buffer is
enlarged from 128 to the reach (SAM2's driver stored all window frames), and the output record gets the sam3_track.py layout
[present, cx, cy, area, score, x0, y0, x1, y1] (no box for SAM2: None).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, "/home/george/sam2-exp/code")
import gt as G  # noqa: E402
import sam2_tracks as T  # noqa: E402
from common import decode_window, iter_seeds, load_spec, pick  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", required=True)
    ap.add_argument("--videos", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--config", default="{}")
    ap.add_argument("--clips", default="")
    ap.add_argument("--protocol", default="oracle")
    ap.add_argument("--max-tracks", type=int, default=0)
    ap.add_argument("--max-reach", type=int, default=0, help="override spec max_reach (frames each way)")
    ap.add_argument("--n-seeds", type=int, default=0, help="use only N evenly spaced seeds per clip (0 = all)")
    ap.add_argument("--ckpt", default="/home/george/sam2-exp/ckpt")
    a = ap.parse_args()
    import torch
    cfg = {"model": "sam2.1-small", "prompt": "box", "crop": 512, "dtype": "fp16", **json.loads(a.config)}
    spec = load_spec(a.spec)
    mr = a.max_reach or spec.get("max_reach", 400)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    only = set(a.clips.split(",")) if a.clips else None
    tag = f"model-{cfg['model']}__prompt-{cfg['prompt']}__crop-{cfg['crop']}" + (f"__{cfg['tag']}" if cfg.get("tag") else "")
    runner = None
    for name, clip in sorted(spec["clips"].items()):
        if only and name not in only:
            continue
        seeds = pick(iter_seeds(clip, a.protocol), a.n_seeds)
        if not seeds:
            continue
        dst = out / f"tracks__{name}__{tag}__{a.protocol}.json"
        doc = json.loads(dst.read_text()) if dst.exists() else {"meta": {"clip": name, "cfg": cfg, "protocol": a.protocol, "host": os.uname().nodename,
                                                                          "gpu": torch.cuda.get_device_name(0), "torch": torch.__version__, "w": clip["w"],
                                                                          "h": clip["h"], "n_frames": clip["n_frames"], "stride": clip["stride"]}, "tracks": []}
        have = {(t["seed"], t["dir"]) for t in doc["tracks"]}
        todo = [(s, d) for s in seeds for d in ("fwd", "bwd") if (s["id"], d) not in have]
        if a.max_tracks:
            todo = todo[:a.max_tracks]
        if not todo:
            continue
        if runner is None:
            runner = T.Runner("small", a.ckpt, f"crop{cfg['crop']}", cfg["dtype"], cfg["prompt"], None)
            runner.buf = torch.zeros(2 * mr + 2, 3, T.IMG, T.IMG, dtype=torch.float32)
            torch.cuda.reset_peak_memory_stats()
        lo = max(0, min(s["t"] for s, _ in todo) - mr)
        hi = min(clip["n_frames"] - 1, max(s["t"] for s, _ in todo) + mr)
        frames = decode_window(str(Path(a.videos) / f"{name}.mp4"), clip["w"], clip["h"], clip["stride"], lo, hi)
        for s, d in todo:
            tlo, thi = max(lo, s["t"] - mr), min(hi, s["t"] + mr)
            run = G.Run(name, tlo, thi, {s["t"]: tuple(s["box"])})
            reach = (thi - s["t"]) if d == "fwd" else (s["t"] - tlo)
            t0 = time.time()
            rec = runner.track(frames, run, s["t"], d, reach)
            if rec is None:
                rec = {"frames": {}, "n_frames": 0}
            rec["frames"] = {k: [v[0], v[1], v[2], v[3], v[4], None, None, None, None] for k, v in rec["frames"].items()}
            rec.update({"seed": s["id"], "dir": d, "t": s["t"], "source": s["source"], "sec": round(time.time() - t0, 3)})
            doc["tracks"].append(rec)
            tf = sum(x["n_frames"] for x in doc["tracks"])
            ts = sum(x["sec"] for x in doc["tracks"])
            doc["perf"] = {"peak_vram_alloc_mb": round(torch.cuda.max_memory_allocated() / 2**20), "frames_propagated": tf, "seconds": round(ts, 1),
                           "sec_per_frame": round(ts / max(tf, 1), 4)}
            tmp = dst.with_name(dst.name + ".tmp")
            tmp.write_text(json.dumps(doc))
            os.replace(tmp, dst)
            print(f"  {name} seed {s['id']}@{s['t']} {d}: {rec['n_frames']} fr in {rec['sec']}s", flush=True)
        del frames


if __name__ == "__main__":
    main()
