"""Condensed markdown tables from score.py's summary.json (pure stdlib).

usage: report.py summary.json > condensed.md
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict


def p(x):
    return "-" if x is None else f"{100 * x:.0f}"


def cfg(m):
    return f"{m['model']}/{m['variant']}"


def main(path: str):
    res = json.load(open(path))
    items = sorted(res.values(), key=lambda r: (r["meta"]["clip"], r["meta"]["model"], r["meta"]["variant"]))
    clips = sorted({r["meta"]["clip"] for r in items})
    out = []

    out.append("### T1. Leave-anchors-out accuracy (held-out GT frames only)\n")
    out.append("`hit@k` = share of ALL held-out frames with centre error <= k px at 1080p (SAM2 absent counts as a miss); "
               "`abs` = SAM2 reports object absent; `med` = median error over frames where a ball was predicted. "
               "`nearest` = prediction from the nearer anchor; `hybrid` = best-scoring pass, falling back to interpolation when SAM2 is absent.\n")
    out.append("| clip | config | N | n held-out | interp hit@5/10/20 | interp med | SAM2 nearest hit@5/10/20 | SAM2 med | SAM2 abs | hybrid hit@5/10/20 |")
    out.append("|---|---|---|---|---|---|---|---|---|---|")
    for r in items:
        m = r["meta"]
        for n, d in r["by_n"].items():
            o = d["overall"]
            i, s, h = o["interp"], o["nearest"], o["hybrid"]
            out.append(f"| {m['clip']} | {cfg(m)} | {n} | {i['n']} | {p(i['hit5'])}/{p(i['hit10'])}/{p(i['hit20'])} | {i['med']} | "
                       f"{p(s['hit5'])}/{p(s['hit10'])}/{p(s['hit20'])} | {s['med']} | {p(s['absent'])}% | {p(h['hit5'])}/{p(h['hit10'])}/{p(h['hit20'])} |")

    out.append("\n### T2. Forward-only vs backward-only vs fused, by distance to the nearest anchor (hit@10 %, N = 25)\n")
    out.append("| clip | config | mode | 1-5 | 6-12 | 13-25 (n) |")
    out.append("|---|---|---|---|---|---|")
    for r in items:
        m = r["meta"]
        d = r["by_n"].get("25")
        if not d:
            continue
        for mode in ("fwd", "bwd", "nearest", "interp"):
            bd = d["by_dist"][mode]
            cells = []
            for b in ("1-5", "6-12", "13-25"):
                s = bd.get(b)
                cells.append(f"{p(s['hit10'])} (n={s['n']})" if s else "-")
            out.append(f"| {m['clip']} | {cfg(m)} | {mode} | " + " | ".join(cells) + " |")

    out.append("\n### T3. Carry curve of a single prompt: hit@10 % / median error px vs frames from the anchor frame\n")
    out.append("Pooled over every anchor of every N (frames that are anchors in any N excluded). `hold` = keep the anchor position (no motion model). Absent counts as a miss.\n")
    out.append("| clip | config | direction | 1-5 | 6-12 | 13-25 | 26-50 |")
    out.append("|---|---|---|---|---|---|---|")
    for r in items:
        m = r["meta"]
        for dirn in ("fwd", "bwd", "interp_hold"):
            cs = r["curve"].get(dirn, {})
            cells = []
            for b in ("1-5", "6-12", "13-25", "26-50"):
                s = cs.get(b)
                if not s or not s.get("n"):
                    cells.append("-")
                else:
                    cells.append(f"{p(s['hit10'])} / {s['med']} (n={s['n']})")
            out.append(f"| {m['clip']} | {cfg(m)} | {'hold' if dirn == 'interp_hold' else dirn} | " + " | ".join(cells) + " |")

    out.append("\n### T4. Fused (nearest) vs linear interpolation by distance, N = 50 (hit@10 %)\n")
    out.append("| clip | config | mode | 1-5 | 6-12 | 13-25 |")
    out.append("|---|---|---|---|---|---|")
    for r in items:
        m = r["meta"]
        d = r["by_n"].get("50")
        if not d:
            continue
        for mode in ("nearest", "interp"):
            bd = d["by_dist"][mode]
            cells = [f"{p(bd[b]['hit10'])} (n={bd[b]['n']})" if b in bd else "-" for b in ("1-5", "6-12", "13-25")]
            out.append(f"| {m['clip']} | {cfg(m)} | {mode} | " + " | ".join(cells) + " |")

    out.append("\n### T5. Forward/backward agreement gate (both passes present and within 5 px of each other)\n")
    out.append("| clip | config | N | held-out | gate passes (coverage) | hit@10 of gated frames | median err |")
    out.append("|---|---|---|---|---|---|---|")
    for r in items:
        m = r["meta"]
        for n, d in r["by_n"].items():
            a = d["agree"]
            s = a["stats_of_agreeing"]
            if not s.get("n"):
                continue
            out.append(f"| {m['clip']} | {cfg(m)} | {n} | {a['n_held']} | {a['n_agree_5px']} ({p(a['n_agree_5px'] / a['n_held'])}%) | {p(s['hit10'])}% | {s['med']} |")

    out.append("\n### T6. False positives on explicit no_ball frames, and not_broadcast frames\n")
    out.append("Counts are (run, frame) pairs: a frame adjacent to two runs is counted once per run. Only astonvilla_seg080 has no_ball frames.\n")
    out.append("| clip | config | no_ball frames | SAM2 predicts a ball | not_broadcast frames | SAM2 predicts a ball |")
    out.append("|---|---|---|---|---|---|")
    for r in items:
        m, a = r["meta"], r["aux"]
        if a["no_ball"]["n"] or a["not_broadcast"]["n"]:
            out.append(f"| {m['clip']} | {cfg(m)} | {a['no_ball']['n']} | {a['no_ball']['present']} | {a['not_broadcast']['n']} | {a['not_broadcast']['present']} |")

    out.append("\n### T7. Failure categories of the nearest-anchor prediction (N = 25), by ball speed (px/frame at 1080p)\n")
    cats = ["hit<=10", "near 10-20", "wrong, on player/ref box", "wrong, elsewhere", "absent"]
    out.append("| clip | config | speed | n | " + " | ".join(cats) + " |")
    out.append("|---|---|---|---|" + "---|" * len(cats))
    for r in items:
        m = r["meta"]
        fa = r.get("failure", {}).get("25", {})
        for sp in ("slow(<8px/f)", "mid(8-20)", "fast(>20)", "ALL"):
            d = fa.get(sp)
            if not d:
                continue
            t = sum(d.values())
            out.append(f"| {m['clip']} | {cfg(m)} | {sp} | {t} | " + " | ".join(f"{100 * d.get(k, 0) / t:.0f}%" for k in cats) + " |")

    out.append("\n### T8. Runtime and peak VRAM (RTX 2070 8 GB, fp16 autocast, offload_video_to_cpu)\n")
    out.append("`s/frame` = wall time per propagated frame, including image preprocessing and mask post-processing. Peak VRAM is torch max_memory_allocated / reserved over the run (model weights included).\n")
    agg = defaultdict(lambda: [0, 0.0, 0, 0])
    for r in items:
        m, pf = r["meta"], r["perf"] or {}
        a = agg[cfg(m)]
        a[0] += pf.get("frames_propagated", 0)
        a[1] += pf.get("sec_per_frame", 0) * pf.get("frames_propagated", 0)
        a[2] = max(a[2], pf.get("peak_vram_alloc_mb", 0))
        a[3] = max(a[3], pf.get("peak_vram_reserved_mb", 0))
    out.append("| config | frames propagated | s/frame | peak VRAM alloc MB | peak VRAM reserved MB |")
    out.append("|---|---|---|---|---|")
    for k, (n, sec, va, vr) in sorted(agg.items()):
        out.append(f"| {k} | {n} | {sec / max(n, 1):.3f} | {va} | {vr} |")
    print("\n".join(out))


if __name__ == "__main__":
    main(sys.argv[1])
