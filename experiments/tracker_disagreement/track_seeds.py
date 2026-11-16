"""Seed-driven bidirectional ball tracks with SAM 3 or SAM 2.1-small (GPU, desktop) -- one script, two engines, identical seeds / windows / outputs.

Same logic and file layout as ../sam3_offline/sam3_label.py (per seed one forward and one backward crop-following track, up to ``reach`` frames each way, never past the
neighbouring seed + ``overlap``; ``<clip>.labels.jsonl`` one row per processed frame with a position + ``<clip>.raw.json.gz`` every raw track; resumable per track and clip), but
the tracker is chosen with ``--engine``:
  sam3  Sam3Tracker (~/sam3-exp/venv)               config {"model":"sam3","prompt":"box","crop":512}
  sam2  SAM2.1-small crop512 Runner (~/sam2-exp/venv; the round-3 / machine_tracks.py tracker)

Whole clips are decoded once (<= ~630 frames of 1080p RGB = 3.9 GB RAM).  Run inside the ``sam3-label-match`` flow (``script=track_seeds.py``; ``engine`` in the config).

    python track_seeds.py --seeds seeds.jsonl --videos DIR --out DIR --engine sam2 --reach 250 --overlap 15
"""
from __future__ import annotations

import argparse
import gzip
import json
import math
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import decode_window, video_info  # noqa: E402


def seed_box(s: dict, h: int) -> list[float]:
    if "box" in s:
        return [float(v) for v in s["box"]]
    r = 5.0 * h / 1080.0
    x, y = s["xy"]
    return [x - r, y - r, x + r, y + r]


def merge_clip(tracks: list[dict], px: float, stride: int, scale: float) -> list[dict]:
    """One row per processed frame: S1 agreement (forward track of seed a vs backward track of the next seed b within ``px``) else the nearest-seed single track.
    Identical to sam3_label.merge_clip except that the single-track gate is recorded as ``single`` and the score is left to the consumer."""
    by_frame: dict[int, list] = {}
    seed_rows: dict[int, dict] = {}
    for t in tracks:
        r0 = t["frames"].get(str(t["t"]))
        if r0 and r0[0]:
            seed_rows[t["t"]] = {"x": r0[1], "y": r0[2], "box": r0[5:9], "score": 99.0, "seed": t["seed"], "dir": "seed", "dist": 0, "gate": "seed"}
        for k, r in t["frames"].items():
            p = int(k)
            if p == t["t"] or not r[0]:
                continue
            by_frame.setdefault(p, []).append({"x": r[1], "y": r[2], "box": r[5:9], "score": r[4], "seed": t["seed"], "dir": t["dir"], "dist": abs(p - t["t"])})
    rows = []
    for p, cs in sorted(by_frame.items()):
        fwd = [c for c in cs if c["dir"] == "fwd"]
        bwd = [c for c in cs if c["dir"] == "bwd"]
        best = None
        for a in sorted(fwd, key=lambda c: c["dist"]):
            for b in sorted(bwd, key=lambda c: c["dist"]):
                if a["seed"] != b["seed"] and math.hypot(a["x"] - b["x"], a["y"] - b["y"]) * scale <= px:
                    best = {**a, "x": (a["x"] + b["x"]) / 2, "y": (a["y"] + b["y"]) / 2, "score": min(a["score"], b["score"]), "gate": "s1", "seed2": b["seed"]}
                    break
            if best:
                break
        if best is None:
            c = min(cs, key=lambda c: c["dist"])
            best = {**c, "gate": "single"}
        rows.append({"p": p, "frame": p * stride, **best})
    have = {r["p"] for r in rows}
    for p, r in seed_rows.items():
        rows = [x for x in rows if x["p"] != p] if p in have else rows
        rows.append({"p": p, "frame": p * stride, **r})
    return sorted(rows, key=lambda r: r["p"])


class Sam2Engine:
    def __init__(self, reach: int, ckpt: str = "/home/george/sam2-exp/ckpt"):
        sys.path.insert(0, "/home/george/sam2-exp/code")
        import torch
        import gt as G
        import sam2_tracks as T
        self.G, self.torch = G, torch
        self.runner = T.Runner("small", ckpt, "crop512", "fp16", "box", None)
        self.runner.buf = torch.zeros(2 * reach + 2, 3, T.IMG, T.IMG, dtype=torch.float32)
        self.cfg = {"model": "sam2.1-small", "prompt": "box", "crop": 512, "dtype": "fp16"}

    def track(self, get_frame, t, box, d, lo, hi):
        class Frames:
            def __getitem__(s, i):
                return get_frame(i)
        run = self.G.Run("x", lo, hi, {t: tuple(box)})
        reach = (hi - t) if d == "fwd" else (t - lo)
        rec = self.runner.track(Frames(), run, t, d, reach)
        if rec is None:
            return {"frames": {}, "n_frames": 0, "sec": 0.0}
        rec["frames"] = {k: [v[0], v[1], v[2], v[3], v[4], None, None, None, None] for k, v in rec["frames"].items()}
        return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", required=True)
    ap.add_argument("--videos", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--engine", default="sam3", choices=["sam3", "sam2"])
    ap.add_argument("--config", default="{}")
    ap.add_argument("--reach", type=int, default=250)
    ap.add_argument("--overlap", type=int, default=15)
    ap.add_argument("--score-gate", type=float, default=1.0)   # accepted for flow compatibility (unused: the gate is applied by the consumer)
    ap.add_argument("--s1-px", type=float, default=5.0)
    ap.add_argument("--clips", default="")
    ap.add_argument("--max-seeds", type=int, default=0)
    a = ap.parse_args()
    import torch
    cfg = {"prune_age": 300, **json.loads(a.config)}
    out = Path(a.out)
    (out / ".tracks").mkdir(parents=True, exist_ok=True)
    seeds_by_clip: dict[str, list] = {}
    for line in Path(a.seeds).read_text().splitlines():
        if line.strip():
            s = json.loads(line)
            seeds_by_clip.setdefault(s["clip"], []).append(s)
    only = set(a.clips.split(",")) if a.clips else None
    runner = None
    stats = {"clips": 0, "frames": 0, "sec": 0.0}
    for clip, seeds in sorted(seeds_by_clip.items()):
        if only and clip not in only:
            continue
        final = out / f"{clip}.labels.jsonl"
        if final.exists():
            print(f"{clip}: done", flush=True)
            continue
        video = str(Path(a.videos) / f"{clip}.mp4")
        if not Path(video).exists():
            print(f"{clip}: video missing, skipped", flush=True)
            continue
        vi = video_info(video)
        stride = 2 if vi["fps"] > 40 else 1
        n_proc = (vi["n"] + stride - 1) // stride
        scale = 1080.0 / vi["h"]
        sd = sorted(({**s, "t": round(s["frame"] / stride), "box": seed_box(s, vi["h"]), "id": s.get("id", f"s{i}")} for i, s in enumerate(seeds)), key=lambda s: s["t"])
        if a.max_seeds:
            sd = sd[:a.max_seeds]
        if runner is None:
            if a.engine == "sam3":
                from sam3_track import Sam3Tracker
                runner = Sam3Tracker(cfg)
            else:
                runner = Sam2Engine(a.reach)
        frames = decode_window(video, vi["w"], vi["h"], stride, 0, n_proc - 1)
        tracks = []
        for i, s in enumerate(sd):
            prev_t = sd[i - 1]["t"] if i > 0 else None
            next_t = sd[i + 1]["t"] if i + 1 < len(sd) else None
            for d in ("fwd", "bwd"):
                lim = (min(n_proc - 1, s["t"] + a.reach, (next_t + a.overlap) if next_t is not None else 10**9) if d == "fwd" else
                       max(0, s["t"] - a.reach, (prev_t - a.overlap) if prev_t is not None else -10**9))
                lo, hi = (s["t"], lim) if d == "fwd" else (lim, s["t"])
                cache = out / ".tracks" / f"{clip}__{s['id']}__{d}.json"
                if cache.exists():
                    tracks.append(json.loads(cache.read_text()))
                    continue
                rec = runner.track(frames.__getitem__, s["t"], s["box"], d, lo, hi)
                rec.update({"seed": s["id"], "dir": d, "t": s["t"], "source": s.get("source", ""), "seed_conf": s.get("conf")})
                tmp = cache.with_name(cache.name + ".tmp")
                tmp.write_text(json.dumps(rec))
                os.replace(tmp, cache)
                tracks.append(rec)
                stats["frames"] += rec["n_frames"]
                stats["sec"] += rec["sec"]
                print(f"  {clip} {s['id']}@{s['frame']} {d}: {rec['n_frames']} fr in {rec['sec']}s ({rec['n_frames'] / max(rec['sec'], 1e-6):.2f} fps)", flush=True)
        del frames
        rows = merge_clip(tracks, a.s1_px, stride, scale)
        with gzip.open(out / f"{clip}.raw.json.gz", "wt") as fh:
            json.dump({"engine": a.engine, "cfg": cfg if a.engine == "sam3" else runner.cfg, "stride": stride, "n_proc": n_proc, "tracks": tracks}, fh)
        tmp = final.with_name(final.name + ".tmp")
        with open(tmp, "w") as fh:
            for r in rows:
                fh.write(json.dumps({"clip": clip, **{k: (round(v, 2) if isinstance(v, float) else v) for k, v in r.items()}}) + "\n")
        os.replace(tmp, final)
        stats["clips"] += 1
        print(f"{clip}: {len(rows)} labelled frames ({sum(r['gate'] == 's1' for r in rows)} s1)", flush=True)
    print(json.dumps({"stats": stats, "fps": stats["frames"] / max(stats["sec"], 1e-6), "peak_vram_mb": round(torch.cuda.max_memory_allocated() / 2**20)}), flush=True)


if __name__ == "__main__":
    main()
