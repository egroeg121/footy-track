"""PRODUCTION shape: SAM 3 as a slow background ball labeller for whole matches (GPU, desktop venv ~/sam3-exp/venv).

Input  : a seeds file (JSONL, one seed per row)  {"clip": "<segment clip stem>", "frame": <original frame index>, "box": [x0, y0, x1, y1]  |  "xy": [x, y],
         "conf": 0.9, "source": "rtdetr|human|overview"}  and a directory with the clips ``<clip>.mp4`` (1080p broadcast segments; 50 fps clips are processed at 25 fps
         = every 2nd frame, labels are written with the ORIGINAL frame index).
Process: per clip and per seed one forward and one backward SAM 3 track (crop that follows the ball, see sam3_track.py), up to ``reach`` frames each way and
         never past the neighbouring seed (+ ``overlap`` frames, so that two seeds' tracks cover the gap between them twice and the round-3 S1 gate -- forward track
         of seed a and backward track of seed b agree within 5 px -- can be applied).  Resumable per track (``.tracks/``) and per clip (final file).
Output : ``<out>/<clip>.labels.jsonl`` one row per processed frame that has a position: clip, frame (original index), x, y, box, score (SAM object-score logit),
         seed (id), dir, dist (frames from the seed), gate (s1 | score | weak) and a ``label`` = x, y of the best candidate.  ``gate`` is the only trust signal:
         s1 = two tracks agree within 5 px (96-100 % within 10 px in the SAM2 study), score = single track with score logit > ``score_gate``, weak = rest.
         ``<out>/<clip>.raw.json.gz`` keeps every raw track.

    python sam3_label.py --seeds seeds.jsonl --videos DIR --out DIR --config '{"model":"sam3","prompt":"box","crop":512}' [--reach 250] [--overlap 15]
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
from sam3_track import Sam3Tracker, cfg_tag  # noqa: E402

CHUNK = 192


class FrameCache:
    """Sequential-friendly frame provider over decode_window (decodes CHUNK-frame windows on demand, keeps the last two)."""

    def __init__(self, video: str, w: int, h: int, stride: int, n: int):
        self.video, self.w, self.h, self.stride, self.n = video, w, h, stride, n
        self.chunks: dict[int, dict] = {}

    def get(self, p: int):
        c = p // CHUNK
        if c not in self.chunks:
            lo, hi = c * CHUNK, min((c + 1) * CHUNK, self.n) - 1
            self.chunks[c] = decode_window(self.video, self.w, self.h, self.stride, lo, hi)
            for k in [k for k in self.chunks if abs(k - c) > 1]:
                del self.chunks[k]
        return self.chunks[c][p]


def seed_box(s: dict, h: int) -> list[float]:
    if "box" in s:
        return [float(v) for v in s["box"]]
    r = 5.0 * h / 1080.0
    x, y = s["xy"]
    return [x - r, y - r, x + r, y + r]


def merge_clip(tracks: list[dict], score_gate: float, px: float, stride: int, scale: float) -> list[dict]:
    """One row per processed frame: S1 agreement (fwd of seed a vs bwd of the next seed b) else nearest-seed single track + score gate."""
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
            best = {**c, "gate": "score" if c["score"] > score_gate else "weak"}
        rows.append({"p": p, "frame": p * stride, **best})
    have = {r["p"] for r in rows}
    for p, r in seed_rows.items():
        rows = [x for x in rows if x["p"] != p] if p in have else rows
        rows.append({"p": p, "frame": p * stride, **r})
    return sorted(rows, key=lambda r: r["p"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", required=True)
    ap.add_argument("--videos", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--config", default="{}")
    ap.add_argument("--reach", type=int, default=250)
    ap.add_argument("--overlap", type=int, default=15)
    ap.add_argument("--score-gate", type=float, default=1.0)
    ap.add_argument("--s1-px", type=float, default=5.0)
    ap.add_argument("--clips", default="")
    ap.add_argument("--max-seeds", type=int, default=0, help="per clip (demo / smoke)")
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
            runner = Sam3Tracker(cfg)
        fc = FrameCache(video, vi["w"], vi["h"], stride, n_proc)
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
                t0 = time.time()
                rec = runner.track(fc.get, s["t"], s["box"], d, lo, hi)
                rec.update({"seed": s["id"], "dir": d, "t": s["t"], "source": s.get("source", ""), "seed_conf": s.get("conf")})
                tmp = cache.with_name(cache.name + ".tmp")
                tmp.write_text(json.dumps(rec))
                os.replace(tmp, cache)
                tracks.append(rec)
                stats["frames"] += rec["n_frames"]
                stats["sec"] += rec["sec"]
                print(f"  {clip} {s['id']}@{s['frame']} {d}: {rec['n_frames']} fr in {rec['sec']}s ({rec['n_frames'] / max(rec['sec'], 1e-6):.2f} fps)", flush=True)
        rows = merge_clip(tracks, a.score_gate, a.s1_px, stride, scale)
        with gzip.open(out / f"{clip}.raw.json.gz", "wt") as fh:
            json.dump({"cfg": cfg_tag({**cfg}), "stride": stride, "tracks": tracks}, fh)
        tmp = final.with_name(final.name + ".tmp")
        with open(tmp, "w") as fh:
            for r in rows:
                fh.write(json.dumps({"clip": clip, **{k: (round(v, 2) if isinstance(v, float) else v) for k, v in r.items()}}) + "\n")
        os.replace(tmp, final)
        stats["clips"] += 1
        print(f"{clip}: {len(rows)} labelled frames ({sum(r['gate'] == 's1' for r in rows)} s1, {sum(r['gate'] == 'score' for r in rows)} score-gated)", flush=True)
    print(json.dumps({"stats": stats, "fps": stats["frames"] / max(stats["sec"], 1e-6), "peak_vram_mb": round(torch.cuda.max_memory_allocated() / 2**20)}), flush=True)


if __name__ == "__main__":
    main()
