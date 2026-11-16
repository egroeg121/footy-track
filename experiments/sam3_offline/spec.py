"""Mini-side (stdlib only) evaluation spec for the SAM 3 offline-labeller experiment.

Builds ``spec.json``: per clip the video, frame stride (50 fps Bournemouth clips are handled as 25 fps by taking every 2nd frame, as in round 3;
GT indices are halved and odd-index GT frames are dropped), human GT ball boxes / no_ball frames (px of the video, in processed-frame index space),
and tracking seeds (oracle = GT frames; realistic = RT-DETR conf >= 0.8 ball candidates).

    python spec.py --out spec.json [--rtdetr-bm <labels_bm_audit>] [--rtdetr-mc <machine_labels>] [--rtdetr-extra <dir>]
"""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

GT_DIR = Path("/mnt/storage/footy_data/ball_gt_marks")
CB = Path("/mnt/storage/footy_data/ballcheck_clips")
EV = Path("/mnt/storage/footy_data/eval_clips_kf10")
BALL = ("in_play_ball", "out_of_play_ball")

# name -> (video, gt jsonl, set, note)
CLIPS = {
    # final / held-out (round 3 list)
    "arsenal_bournemouth_1st_seg009": (CB / "bournemouth_seg009.mp4", "bournemouth_seg009", "final", "held-out"),
    "arsenal_bournemouth_1st_seg015": (CB / "bournemouth_broadcast_seg015.mp4", "bournemouth_broadcast_seg015", "final", "held-out"),
    "arsenal_bournemouth_1st_seg021": (CB / "bournemouth_seg021.mp4", "bournemouth_seg021", "final", "held-out"),
    "arsenal_bournemouth_1st_seg031": (CB / "bournemouth_seg031.mp4", "bournemouth_seg031", "final", "held-out"),
    "arsenal_bournemouth_1st_seg035": (CB / "bournemouth_seg035.mp4", "bournemouth_seg035", "final", "held-out"),
    "arsenal_bournemouth_1st_seg060": (CB / "bournemouth_1st_seg060.mp4", "bournemouth_1st_seg060", "final", "held-out"),
    "arsenal_mancity_part093": (CB / "mancity_part093.mp4", "mancity_part093", "final", "held-out"),
    "arsenal_mancity_seg101": (CB / "arsenal_mancity_20250925_seg101.mp4", "arsenal_mancity_20250925_seg101", "final", "held-out"),
    "arsenal_mancity_seg134": (CB / "arsenal_mancity_20250925_seg134.mp4", "arsenal_mancity_20250925_seg134", "final", "held-out"),
    "arsenal_mancity_seg135": (CB / "arsenal_mancity_20250925_seg135.mp4", "arsenal_mancity_20250925_seg135", "final", "held-out"),
    "arsenal_example": (EV / "arsenal_example.mp4", "arsenal_example", "example", "every detector scores 0.05 on its 20 held-out frames"),
    # dense long run (final, long-distance curve; Aston Villa, never used for tuning)
    "astonvilla_seg080": (EV / "astonvilla_seg080.mp4", "astonvilla_seg080", "final", "dense 359-frame run"),
    # dev (prompt / trick search)
    "bournemouth_1st_seg070": (CB / "bournemouth_1st_seg070.mp4", "bournemouth_1st_seg070", "dev", "dense"),
    "mancity_seg010": (CB / "mancity_seg010.mp4", "mancity_seg010", "dev", ""),
    "mancity_seg031": (CB / "mancity_seg031.mp4", "mancity_seg031", "dev", ""),
    "mancity_seg050": (CB / "mancity_seg050.mp4", "mancity_seg050", "dev", ""),
}
RT_MAP = {  # clip -> (labels dir key, stem)
    **{f"arsenal_bournemouth_1st_seg{s}": ("bm", f"arsenal_bournemouth_1st_seg{s}") for s in ("009", "015", "021", "031", "035", "060")},
    **{f"arsenal_mancity_seg{s}": ("mc", f"arsenal_mancity_20250925_seg{s}") for s in ("101", "134", "135")},
    "arsenal_mancity_part093": ("extra", "arsenal_mancity_part093"),
    "arsenal_example": ("extra", "arsenal_example"),
}


def probe(video: Path) -> dict:
    o = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height,r_frame_rate,nb_frames", "-of", "csv=p=0",
                        str(video)], capture_output=True, text=True, check=True).stdout.strip().split(",")
    num, den = o[2].split("/")
    return {"w": int(o[0]), "h": int(o[1]), "fps": float(num) / float(den), "n_orig": int(o[3])}


def load_gt(path: Path, w: int, h: int):
    ball, other = {}, {}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        tags, f = r["tags"], r["frame_index"]
        if tags[0] in ("no_ball", "not_broadcast"):
            other[f] = tags[0]
        elif tags[0] in BALL and ("labeller" in tags or "human_checked" in tags) and r.get("bbox"):
            b = r["bbox"]
            box = [b["x"] * w, b["y"] * h, (b["x"] + b["w"]) * w, (b["y"] + b["h"]) * h]
            rank = 0 if tags[0] == "in_play_ball" else 1
            if f not in ball or rank < ball[f][0]:
                ball[f] = (rank, box)
    return {f: v[1] for f, v in ball.items()}, other


def runs_of(frames: list[int]) -> list[tuple[int, int]]:
    out, i = [], 0
    while i < len(frames):
        j = i
        while j + 1 < len(frames) and frames[j + 1] == frames[j] + 1:
            j += 1
        out.append((frames[i], frames[j]))
        i = j + 1
    return out


def oracle_seeds(gt: dict, k: int, dense_min: int = 100) -> list[int]:
    fs = sorted(gt)
    if len(fs) < 2:
        return []
    runs = runs_of(fs)
    longest = max(runs, key=lambda r: r[1] - r[0])
    if longest[1] - longest[0] + 1 >= dense_min:   # dense clip: run start, run middle (both directions get long reach)
        return sorted({longest[0], (longest[0] + longest[1]) // 2})
    if len(fs) <= k:
        return fs
    qs = [0.15 + 0.7 * i / max(k - 1, 1) for i in range(k)]   # avoid the very first GT frame (mancity_seg031 frame 0 is not on a ball)
    return sorted({fs[int(round(q * (len(fs) - 1)))] for q in qs})


def rt_seeds(path: Path, w: int, h: int, stride: int, n_proc: int, thr: float = 0.8, k: int = 3, spacing: int = 40) -> list[dict]:
    cands = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if not any(t in BALL or t == "ball" for t in r["tags"]) or r.get("confidence", 0) < thr:
            continue
        f = r["frame_index"]
        b = r["bbox"]
        cands.append((r["confidence"], f, [b["x"] * w, b["y"] * h, (b["x"] + b["w"]) * w, (b["y"] + b["h"]) * h]))
    cands.sort(reverse=True)
    chosen = []
    for conf, f, box in cands:
        if f < n_proc and all(abs(f - c["t"]) >= spacing for c in chosen):
            chosen.append({"t": f, "box": [round(v, 2) for v in box], "conf": round(conf, 4), "source": "rtdetr"})
        if len(chosen) >= k:
            break
    return sorted(chosen, key=lambda c: c["t"]), len(cands)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--rtdetr-bm", type=Path, default=Path("/mnt/storage/exp/temporal/round3/labels_bm_audit"))
    ap.add_argument("--rtdetr-mc", type=Path, default=Path("/mnt/storage/footy_data/machine_labels"))
    ap.add_argument("--rtdetr-extra", type=Path, default=None, help="dir with <clip>.jsonl RT-DETR labels for part093 / arsenal_example")
    a = ap.parse_args()
    rt_dirs = {"bm": a.rtdetr_bm, "mc": a.rtdetr_mc, "extra": a.rtdetr_extra}
    spec = {"max_reach": 400, "clips": {}}
    for name, (video, gtname, sset, note) in CLIPS.items():
        pr = probe(video)
        stride = 2 if pr["fps"] > 40 else 1
        gt_all, other_all = load_gt(GT_DIR / f"{gtname}.jsonl", pr["w"], pr["h"])
        gt = {f // stride: [round(v, 2) for v in b] for f, b in gt_all.items() if f % stride == 0}
        # odd original frames of 50 fps clips: scored against the mean of the two neighbouring processed frames (key = lower neighbour)
        gt_odd = {(f - 1) // stride: [round(v, 2) for v in b] for f, b in gt_all.items() if f % stride == 1 and (f + 1) // stride < (pr["n_orig"] + stride - 1) // stride}
        other = {f // stride: t for f, t in other_all.items() if f % stride == 0}
        n_proc = (pr["n_orig"] + stride - 1) // stride
        c = {"video": str(video), "set": sset, "note": note, "w": pr["w"], "h": pr["h"], "fps_orig": pr["fps"], "stride": stride, "n_frames": n_proc,
             "n_gt_orig": len(gt_all), "gt": {str(f): b for f, b in sorted(gt.items())}, "gt_odd": {str(f): b for f, b in sorted(gt_odd.items())}, "no_ball": [f for f, t in sorted(other.items()) if t == "no_ball"],
             "not_broadcast": [f for f, t in sorted(other.items()) if t == "not_broadcast"]}
        c["seeds_oracle"] = [{"t": f, "box": gt[f], "source": "oracle"} for f in oracle_seeds(gt, a.k)]
        c["seeds_rtdetr"], c["n_rt_cands"] = [], 0
        if name in RT_MAP and rt_dirs[RT_MAP[name][0]] is not None:
            p = rt_dirs[RT_MAP[name][0]] / f"{RT_MAP[name][1]}.jsonl"
            if p.exists():
                c["seeds_rtdetr"], c["n_rt_cands"] = rt_seeds(p, pr["w"], pr["h"], stride, n_proc)
        spec["clips"][name] = c
        print(f"{name:34s} {sset:7s} {pr['w']}x{pr['h']} {pr['fps']:.0f}fps n={n_proc} gt={len(gt)}+{len(gt_odd)}odd/{len(gt_all)} noball={len(c['no_ball'])} "
              f"oracle={[s['t'] for s in c['seeds_oracle']]} rt={[(s['t'], s['conf']) for s in c['seeds_rtdetr']]}/{c['n_rt_cands']}")
    a.out.write_text(json.dumps(spec))


if __name__ == "__main__":
    main()
