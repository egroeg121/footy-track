"""Score SAM2 anchor tracks against held-out GT (CPU, numpy only).

Input: ``tracks__<clip>__<model>__<variant>__<dtype>__<prompt>.json`` written by sam2_tracks.py.
Output: ``summary.json`` (aggregates) and ``tables.md`` (markdown tables).

Leave-anchors-out: for anchor spacing N the anchors are every N-th labelled frame of a run
(+ the last frame); only the remaining GT ball frames are scored. Per held-out frame:
  fwd     track propagated forward from the previous anchor
  bwd     track propagated backward from the next anchor (reverse=True)
  nearest fwd or bwd, whichever anchor is nearer (tie: fwd)
  scorew  the present prediction with the highest sigmoid(object score) / distance
  interp  linear interpolation of the two bracketing anchor centres (baseline, never absent)
Absent predictions count as misses in every hit rate (denominator = all held-out frames).
"""
from __future__ import annotations

import argparse
import glob
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

BUCKETS = [(1, 5), (6, 12), (13, 25), (26, 50)]
MODES = ["fwd", "bwd", "nearest", "scorew", "hybrid", "interp"]


def bucket_of(d: int) -> int | None:
    for i, (lo, hi) in enumerate(BUCKETS):
        if lo <= d <= hi:
            return i
    return None


def sig(x: float) -> float:
    return 1 / (1 + math.exp(-x)) if x == x else 0.5


def stats(errs: list[float | None]) -> dict:
    """errs: one entry per sample, None = absent (counts as a miss)."""
    n = len(errs)
    if n == 0:
        return {"n": 0}
    pres = np.array([e for e in errs if e is not None], dtype=float)
    out = {"n": n, "absent": round(1 - len(pres) / n, 4)}
    for t in (5, 10, 20):
        out[f"hit{t}"] = round(float((pres <= t).sum()) / n, 4)
    out["wrong50"] = round(float((pres > 50).sum()) / max(len(pres), 1), 4)  # present but >50 px off
    out["med"] = round(float(np.median(pres)), 2) if len(pres) else None
    out["mean"] = round(float(pres.mean()), 2) if len(pres) else None
    return out


def get(tracks, ri, a, d):
    return tracks.get((ri, a, d))


def pred_of(track, f):
    if track is None:
        return None
    return track["frames"].get(str(f))


SCALE = 1.0  # native px -> 1080p-equivalent px (set per file)


def err(p, gc):
    if p is None or not p[0]:
        return None
    return SCALE * math.hypot(p[1] - gc[0], p[2] - gc[1])


def score_file(doc: dict, people: dict | None = None) -> dict:
    global SCALE
    SCALE = doc["meta"].get("scale1080", 1.0)
    tracks = {(t["run"], t["anchor"], t["dir"]): t for t in doc["tracks"]}
    res = {"meta": doc["meta"], "perf": doc.get("perf"), "by_n": {}, "curve": {}, "aux": {}}
    # ------------------------------------------------------------ leave-anchors-out per N
    for n in doc["meta"]["ns"]:
        samples = {m: [] for m in MODES}
        agree = {"n_both": 0, "n_agree": 0, "errs": []}   # forward and backward tracks agree within 5 px
        by_bucket = {m: defaultdict(list) for m in MODES}
        per_run = []
        for ri, run in enumerate(doc["runs"]):
            A = run["anchors"][str(n)]
            boxes = {int(f): b for f, b in run["boxes"].items()}
            cen = {f: ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2) for f, b in boxes.items()}
            held = [f for f in range(run["start"], run["end"] + 1) if f not in set(A)]
            run_e = {m: [] for m in MODES}
            for f in held:
                ap = max(a for a in A if a < f)
                an = min(a for a in A if a > f)
                dp, dn = f - ap, an - f
                pf = pred_of(get(tracks, ri, ap, "fwd"), f)
                pb = pred_of(get(tracks, ri, an, "bwd"), f)
                gc = cen[f]
                t = dp / (dp + dn)
                ip = (cen[ap][0] + t * (cen[an][0] - cen[ap][0]), cen[ap][1] + t * (cen[an][1] - cen[ap][1]))
                e = {"fwd": err(pf, gc), "bwd": err(pb, gc), "interp": SCALE * math.hypot(ip[0] - gc[0], ip[1] - gc[1])}
                e["nearest"] = e["fwd"] if dp <= dn else e["bwd"]
                cand = [(sig(p[4]) / d, x) for p, d, x in ((pf, dp, e["fwd"]), (pb, dn, e["bwd"])) if p is not None and p[0]]
                e["scorew"] = max(cand)[1] if cand else None
                e["hybrid"] = e["scorew"] if e["scorew"] is not None else e["interp"]   # SAM2 else interpolation
                if pf is not None and pb is not None and pf[0] and pb[0]:
                    both = SCALE * math.hypot(pf[1] - pb[1], pf[2] - pb[2])
                    agree["n_both"] += 1
                    if both <= 5:
                        agree["n_agree"] += 1
                        agree["errs"].append(e["scorew"])
                dmin = min(dp, dn)
                for m in MODES:
                    dist = dp if m == "fwd" else dn if m == "bwd" else dmin
                    samples[m].append(e[m])
                    run_e[m].append(e[m])
                    b = bucket_of(dist)
                    if b is not None:
                        by_bucket[m][b].append(e[m])
            per_run.append({"run": [run["start"], run["end"]], "n_held": len(held),
                            **{m: stats(run_e[m]) for m in ("nearest", "interp")}})
        res["by_n"][str(n)] = {
            "overall": {m: stats(samples[m]) for m in MODES},
            "by_dist": {m: {f"{BUCKETS[b][0]}-{BUCKETS[b][1]}": stats(v) for b, v in sorted(by_bucket[m].items())} for m in MODES},
            "per_run": per_run,
            "agree": {"n_held": len(samples["fwd"]), "n_both_present": agree["n_both"], "n_agree_5px": agree["n_agree"],
                      "stats_of_agreeing": stats(agree["errs"])},
        }
    # ------------------------------------------------------------ carry curve (single prompt)
    anchor_union = defaultdict(set)
    for ri, run in enumerate(doc["runs"]):
        for n in doc["meta"]["ns"]:
            anchor_union[ri].update(run["anchors"][str(n)])
    curve = {"fwd": defaultdict(list), "bwd": defaultdict(list), "interp_hold": defaultdict(list)}
    for (ri, a, d), t in tracks.items():
        run = doc["runs"][ri]
        boxes = {int(f): b for f, b in run["boxes"].items()}
        for fs, p in t["frames"].items():
            f = int(fs)
            if f == a or f not in boxes or f in anchor_union[ri]:
                continue
            dist = abs(f - a)
            b = bucket_of(dist)
            if b is None or dist > t["reach"]:
                continue
            b0 = boxes[f]
            gc = ((b0[0] + b0[2]) / 2, (b0[1] + b0[3]) / 2)
            curve[d][b].append(err(p, gc))
            if d == "fwd":  # baseline: hold the anchor position (no motion model)
                ba = boxes[a]
                curve["interp_hold"][b].append(SCALE * math.hypot((ba[0] + ba[2]) / 2 - gc[0], (ba[1] + ba[3]) / 2 - gc[1]))
    res["curve"] = {k: {f"{BUCKETS[b][0]}-{BUCKETS[b][1]}": stats(v) for b, v in sorted(vs.items())} for k, vs in curve.items()}
    # ------------------------------------------------------------ no_ball / not_broadcast frames
    aux = {"no_ball": [], "not_broadcast": []}
    for ri, run in enumerate(doc["runs"]):
        for fs, kind in run["aux"].items():
            f = int(fs)
            if f < run["start"]:
                tr, a = tracks.get((ri, run["start"], "bwd")), run["start"]
            else:
                tr, a = tracks.get((ri, run["end"], "fwd")), run["end"]
            p = pred_of(tr, f)
            if p is not None:
                aux[kind].append({"run": ri, "frame": f, "dist": abs(f - a), "present": bool(p[0]),
                                  "score": p[4], "area": p[3]})
    res["failure"] = {str(n): failure_analysis(doc, people or {}, n) for n in (10, 25, 50) if n in doc["meta"]["ns"]}
    res["aux"] = {k: {"n": len(v), "present": sum(x["present"] for x in v), "items": v} for k, v in aux.items()}
    return res


def load_people(gt_dir: str, clip: str, w: int, h: int) -> dict[int, list[tuple]]:
    """frame -> player / referee / person boxes (native px) from the GT jsonl (any provenance)."""
    p = Path(gt_dir) / f"{clip}.jsonl"
    out: dict[int, list[tuple]] = defaultdict(list)
    if not p.exists():
        return out
    for line in p.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if r["tags"][0] in ("player", "referee", "person") and r.get("bbox"):
            b = r["bbox"]
            out[r["frame_index"]].append((b["x"] * w, b["y"] * h, (b["x"] + b["w"]) * w, (b["y"] + b["h"]) * h))
    return out


def failure_analysis(doc: dict, people: dict, n: int = 25, mode: str = "nearest") -> dict:
    """Categorise every held-out frame of the fused-nearest prediction (anchor spacing n)."""
    tracks = {(t["run"], t["anchor"], t["dir"]): t for t in doc["tracks"]}
    sc = doc["meta"].get("scale1080", 1.0)
    cats = defaultdict(lambda: defaultdict(int))     # speed class -> category -> count
    for ri, run in enumerate(doc["runs"]):
        A = run["anchors"][str(n)]
        boxes = {int(f): b for f, b in run["boxes"].items()}
        cen = {f: ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2) for f, b in boxes.items()}
        for f in range(run["start"], run["end"] + 1):
            if f in set(A):
                continue
            ap = max(a for a in A if a < f)
            an = min(a for a in A if a > f)
            tr, d = ((tracks.get((ri, ap, "fwd")), f - ap) if f - ap <= an - f else (tracks.get((ri, an, "bwd")), an - f))
            p = pred_of(tr, f)
            prev = cen.get(f - 1) or cen.get(f + 1)
            speed = sc * math.hypot(cen[f][0] - prev[0], cen[f][1] - prev[1]) if prev else 0.0
            sp = "slow(<8px/f)" if speed < 8 else "mid(8-20)" if speed < 20 else "fast(>20)"
            if p is None or not p[0]:
                cat = "absent"
            else:
                e = sc * math.hypot(p[1] - cen[f][0], p[2] - cen[f][1])
                if e <= 10:
                    cat = "hit<=10"
                elif e <= 20:
                    cat = "near 10-20"
                else:
                    on_person = any(b[0] <= p[1] <= b[2] and b[1] <= p[2] <= b[3] for b in people.get(f, []))
                    cat = "wrong, on player/ref box" if on_person else "wrong, elsewhere"
            for key in (sp, "ALL"):
                cats[key][cat] += 1
    return {k: dict(v) for k, v in cats.items()}


def pct(x):
    return "-" if x is None else f"{100*x:.0f}%"


def row_stats(s):
    if not s.get("n"):
        return "| 0 | - | - | - | - | - | - |"
    return (f"| {s['n']} | {pct(s['absent'])} | {s['med'] if s['med'] is not None else '-'} | "
            f"{pct(s['hit5'])} | {pct(s['hit10'])} | {pct(s['hit20'])} | {pct(s['wrong50'])} |")


def tag_of(meta):
    return f"{meta['model']}/{meta['variant']}/{meta['dtype']}/{meta['prompt']}"


def build_tables(results: dict[str, dict]) -> str:
    L = []
    hdr = "| n held-out | SAM2 absent | median err px | hit@5 | hit@10 | hit@20 | wrong-object (>50px, of present) |"
    sep = "|---|---|---|---|---|---|---|"
    # overall
    L.append("### Held-out accuracy by anchor spacing\n")
    L.append("| clip | config | N | mode |" + hdr[1:])
    L.append("|---|---|---|---|" + sep[1:])
    for key, r in sorted(results.items()):
        m = r["meta"]
        for n, d in r["by_n"].items():
            for mode in ("interp", "fwd", "bwd", "nearest", "scorew", "hybrid"):
                L.append(f"| {m['clip']} | {tag_of(m)} | {n} | {mode} " + row_stats(d["overall"][mode]))
    L.append("\n### Forward/backward agreement (both tracks present and within 5 px): coverage and precision\n")
    L.append("| clip | config | N | held-out | both present | agree <=5px (coverage) | hit@10 of agreeing | hit@20 of agreeing | median err |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    for key, r in sorted(results.items()):
        m = r["meta"]
        for n, d in r["by_n"].items():
            a = d["agree"]
            s = a["stats_of_agreeing"]
            if not s.get("n"):
                continue
            L.append(f"| {m['clip']} | {tag_of(m)} | {n} | {a['n_held']} | {a['n_both_present']} | {a['n_agree_5px']} ({pct(a['n_agree_5px']/a['n_held'])}) | {pct(s['hit10'])} | {pct(s['hit20'])} | {s['med']} |")
    L.append("\n### Carry curve: single prompt, error vs distance from the anchor frame\n")
    L.append("(pooled over all anchors of all N; frames that are anchors in any N excluded; `hold` = baseline that keeps the anchor position)\n")
    L.append("| clip | config | direction | distance (frames) |" + hdr[1:])
    L.append("|---|---|---|---|" + sep[1:])
    for key, r in sorted(results.items()):
        m = r["meta"]
        for dirn in ("fwd", "bwd", "interp_hold"):
            for b, s in r["curve"].get(dirn, {}).items():
                name = {"interp_hold": "hold-anchor"}.get(dirn, dirn)
                L.append(f"| {m['clip']} | {tag_of(m)} | {name} | {b} " + row_stats(s) if dirn != "interp_hold" else
                         f"| {m['clip']} | {tag_of(m)} | {name} | {b} | {s['n']} | - | {s['med']} | {pct(s['hit5'])} | {pct(s['hit10'])} | {pct(s['hit20'])} | - |")
    L.append("\n### Fused vs interpolation by distance to nearest anchor (N=10 / 25 / 50)\n")
    L.append("| clip | config | N | mode | distance |" + hdr[1:])
    L.append("|---|---|---|---|---|" + sep[1:])
    for key, r in sorted(results.items()):
        m = r["meta"]
        for n, d in r["by_n"].items():
            for mode in ("interp", "nearest"):
                for b, s in d["by_dist"][mode].items():
                    L.append(f"| {m['clip']} | {tag_of(m)} | {n} | {mode} | {b} " + row_stats(s))
    L.append("\n### no_ball frames (false positives) and not_broadcast frames\n")
    L.append("| clip | config | no_ball frames | SAM2 predicts a ball (FP) | not_broadcast frames | SAM2 predicts a ball |")
    L.append("|---|---|---|---|---|---|")
    for key, r in sorted(results.items()):
        m = r["meta"]
        a = r["aux"]
        L.append(f"| {m['clip']} | {tag_of(m)} | {a['no_ball']['n']} | {a['no_ball']['present']} | {a['not_broadcast']['n']} | {a['not_broadcast']['present']} |")
    L.append("\n### Failure modes of the fused-nearest prediction (all held-out frames)\n")
    cats = ["hit<=10", "near 10-20", "wrong, on player/ref box", "wrong, elsewhere", "absent"]
    L.append("| clip | config | N | ball speed | n | " + " | ".join(cats) + " |")
    L.append("|---|---|---|---|---|" + "---|" * len(cats))
    for key, r in sorted(results.items()):
        m = r["meta"]
        for n, fa in r.get("failure", {}).items():
            for sp in ("slow(<8px/f)", "mid(8-20)", "fast(>20)", "ALL"):
                d = fa.get(sp)
                if not d:
                    continue
                tot = sum(d.values())
                L.append(f"| {m['clip']} | {tag_of(m)} | {n} | {sp} | {tot} | " + " | ".join(pct(d.get(k, 0) / tot) for k in cats) + " |")
    L.append("\n### Runtime and memory\n")
    L.append("| clip | config | frames propagated | s/frame | peak VRAM alloc MB | peak VRAM reserved MB | GPU |")
    L.append("|---|---|---|---|---|---|---|")
    for key, r in sorted(results.items()):
        m, p = r["meta"], r["perf"] or {}
        L.append(f"| {m['clip']} | {tag_of(m)} | {p.get('frames_propagated')} | {p.get('sec_per_frame')} | {p.get('peak_vram_alloc_mb')} | {p.get('peak_vram_reserved_mb')} | {m.get('gpu')} |")
    return "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tracks_dir")
    ap.add_argument("--out", default=None)
    ap.add_argument("--glob", default="tracks__*.json")
    ap.add_argument("--gt-dir", default="/mnt/storage/footy_data/ball_gt_marks")
    args = ap.parse_args()
    results = {}
    for p in sorted(glob.glob(str(Path(args.tracks_dir) / args.glob))):
        doc = json.loads(Path(p).read_text())
        w, h = doc["meta"].get("video_wh", (1920, 1080))
        results[Path(p).stem] = score_file(doc, load_people(args.gt_dir, doc["meta"]["clip"], w, h))
    out = Path(args.out or args.tracks_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(results, indent=1))
    (out / "tables.md").write_text(build_tables(results))
    print(build_tables(results))


if __name__ == "__main__":
    main()
