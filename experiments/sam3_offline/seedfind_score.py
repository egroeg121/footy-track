"""Score sam3_seedfind.py output (text prompts x tilings) against GT, next to RT-DETR on the same frames (mini CPU, numpy).

    python seedfind_score.py --spec spec.json --seedfind DIR [DIR..] --rtdetr DIR --out result.json [--exemplar]
"""
import argparse, json, math
from pathlib import Path
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--spec"); ap.add_argument("--seedfind", nargs="+"); ap.add_argument("--rtdetr", default=""); ap.add_argument("--out", default="")
ap.add_argument("--suffix", default="")
a = ap.parse_args()
spec = json.load(open(a.spec))
THR = [0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.8]
rows = []   # frame-level records: clip, frame, kind, gt centre, {name: [(score, cx, cy)]}
for d in a.seedfind:
    for p in sorted(Path(d).glob(f"seedfind__*{a.suffix}.jsonl")):
        if bool(a.suffix) != p.stem.endswith("__ex"):
            continue
        for line in p.read_text().splitlines():
            r = json.loads(line)
            sc = 1080.0 / spec["clips"][r["clip"]]["h"]
            g = r["gt"]
            rows.append((r, sc, None if g is None else ((g[0] + g[2]) / 2, (g[1] + g[3]) / 2)))
by = {}
for r, sc, gc in rows:
    for prompt, dets in r["dets"].items():
        by.setdefault((r["scheme"], prompt), {})[(r["clip"], r["frame"])] = (r["kind"], sc, gc, [((d[0] + d[2]) / 2, (d[1] + d[3]) / 2, d[4]) for d in dets])
# RT-DETR on the same frames
rt = {}
if a.rtdetr:
    for clip in {r["clip"] for r, _, _ in rows}:
        p = Path(a.rtdetr) / f"{clip}.jsonl"
        if not p.exists(): continue
        W, H = spec["clips"][clip]["w"], spec["clips"][clip]["h"]
        fr = {}
        for line in p.read_text().splitlines():
            x = json.loads(line)
            if any(t in ("in_play_ball", "ball") for t in x["tags"]):
                b = x["bbox"]
                fr.setdefault(x["frame_index"], []).append(((b["x"] + b["w"] / 2) * W, (b["y"] + b["h"] / 2) * H, x["confidence"]))
        for (c, f) in {(r["clip"], r["frame"]) for r, _, _ in rows if r["clip"] == clip}:
            kind, sc, gc, _ = next(v for k2, v in by[next(iter(by))].items() if k2 == (c, f))
            rt.setdefault(("RT-DETR", "rtdetr-l v11"), {})[(c, f)] = (kind, sc, gc, fr.get(f, []))
by.update(rt)

SUBSETS = {"held-out all (no arsenal_example)": lambda c: c != "arsenal_example",
           "Bournemouth": lambda c: "bournemouth" in c, "Man City (no arsenal_example)": lambda c: "mancity" in c and c != "arsenal_example",
           "arsenal_example (held-out frames)": lambda c: c == "arsenal_example", "astonvilla_seg080 (dense)": lambda c: c.startswith("astonvilla")}

def metrics(frames, sub):
    ball = [(sc, gc, d) for (c, f), (k, sc, gc, d) in frames.items() if k == "ball" and sub(c)]
    nob = [d for (c, f), (k, sc, gc, d) in frames.items() if k == "noball" and sub(c)]
    out = {"n_ball": len(ball), "n_noball": len(nob)}
    if not ball: return out
    def hit(sc, gc, x, y): return math.hypot(x - gc[0], y - gc[1]) * sc <= 10
    out["top1_hit10"] = float(np.mean([bool(d) and hit(sc, gc, *max(d, key=lambda t: t[2])[:2]) for sc, gc, d in ball]))
    out["any_hit10"] = float(np.mean([any(hit(sc, gc, x, y) for x, y, s in d) for sc, gc, d in ball]))
    out["by_thr"] = {}
    for t in THR:
        emit = [(sc, gc, [e for e in d if e[2] >= t]) for sc, gc, d in ball]
        n_emit = sum(1 for _, _, e in emit if e)
        top1_ok = sum(1 for sc, gc, e in emit if e and hit(sc, gc, *max(e, key=lambda q: q[2])[:2]))
        any_ok = sum(1 for sc, gc, e in emit if any(hit(sc, gc, x, y) for x, y, s in e))
        out["by_thr"][str(t)] = {"frames_with_det": n_emit / len(ball), "top1_precision": (top1_ok / n_emit) if n_emit else None, "top1_recall": top1_ok / len(ball), "any_recall": any_ok / len(ball),
                                 "noball_false_pos": (sum(1 for d in nob if any(e[2] >= t for e in d)) / len(nob)) if nob else None}
    return out

res = {}
for key, frames in sorted(by.items()):
    res[f"{key[0]} | {key[1]}"] = {sn: metrics(frames, f) for sn, f in SUBSETS.items()}
if a.out: Path(a.out).write_text(json.dumps(res, indent=1))
sn = "held-out all (no arsenal_example)"
print(f"{'scheme | prompt':44s} n   top1@10  any@10  | thr 0.3: det-frames top1-prec  | thr 0.5: det-frames top1-prec  | no-ball FP@0.3")
for k, v in res.items():
    m = v[sn]
    if not m.get("n_ball"): continue
    t3, t5 = m["by_thr"]["0.3"], m["by_thr"]["0.5"]
    f = lambda x: "  -  " if x is None else f"{x:.2f}"
    print(f"{k:44s} {m['n_ball']:3d} {m['top1_hit10']:.3f}   {m['any_hit10']:.3f}  | {f(t3['frames_with_det'])} {f(t3['top1_precision'])} | {f(t5['frames_with_det'])} {f(t5['top1_precision'])} | {f(t3['noball_false_pos'])} (n={m['n_noball']})")
