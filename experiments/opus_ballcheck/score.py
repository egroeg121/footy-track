"""Score a run against George's verdicts. Usage: python score.py RUN [RUN2 ...] [--md] [--split dev|test]"""
from __future__ import annotations

import collections
import json
import math
import statistics
import sys
from pathlib import Path

OUT = Path("/mnt/storage/exp/opus_ballcheck")
CLASSES = ["ball", "not_ball", "box_off", "unsure"]


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def fmt(k: int, n: int) -> str:
    if n == 0:
        return "n/a (0)"
    lo, hi = wilson(k, n)
    return f"{k / n:.3f} [{lo:.2f}-{hi:.2f}] ({k}/{n})"


def kappa(pairs: list[tuple[str, str]]) -> float:
    n = len(pairs)
    if n == 0:
        return float("nan")
    po = sum(a == b for a, b in pairs) / n
    ca = collections.Counter(a for a, _ in pairs)
    cb = collections.Counter(b for _, b in pairs)
    pe = sum(ca[c] * cb[c] for c in set(ca) | set(cb)) / (n * n)
    return (po - pe) / (1 - pe) if pe < 1 else float("nan")


SKIP: set[str] = set()


def load(run: str, split: str | None = None):
    cards = {c["id"]: c for c in json.load(open(OUT / "cards.json"))}
    rows, errors = [], 0
    for line in open(OUT / "runs" / f"{run}.jsonl"):
        r = json.loads(line)
        c = cards[r["id"]]
        if split and c["split"] != split:
            continue
        if r["id"] in SKIP:
            continue
        if "error" in r or not r.get("parsed"):
            errors += 1
            continue
        p = r["parsed"]
        v = p.get("verdict")
        if v not in CLASSES:
            errors += 1
            continue
        try:
            conf = max(0.0, min(1.0, float(p.get("confidence"))))
        except (TypeError, ValueError):
            conf = 0.5
        rows.append({**c, "pred": v, "pconf": conf, "reason": p.get("reason", ""),
                     "which": p.get("which_box"), "cost": r.get("cost_usd") or 0.0,
                     "wall": r.get("wall_s") or 0.0, "usage": r.get("usage") or {}})
    return rows, errors


def report(run: str, split: str | None = None) -> str:
    rows, errors = load(run, split)
    n = len(rows)
    o = [f"## {run}" + (f" ({split})" if split else ""), f"cards scored: {n}, failed/unparseable: {errors}", ""]
    if not n:
        return "\n".join(o)
    acc = sum(r["pred"] == r["george"] for r in rows)
    o.append(f"4-class accuracy: {fmt(acc, n)}   kappa: {kappa([(r['george'], r['pred']) for r in rows]):.3f}")
    # binary: ball-present (ball/box_off) vs not
    def pres(v): return "present" if v in ("ball", "box_off") else ("absent" if v == "not_ball" else "unsure")
    b = [(pres(r["george"]), pres(r["pred"])) for r in rows if r["george"] != "unsure" and r["pred"] != "unsure"]
    o.append(f"binary ball-present (excl. unsure either side, n={len(b)}): acc {fmt(sum(a == c for a, c in b), len(b))}"
             f"   kappa: {kappa(b):.3f}")
    # ball precision / recall (positive = verdict 'ball' exactly)
    tp = sum(r["pred"] == "ball" and r["george"] == "ball" for r in rows)
    pp = sum(r["pred"] == "ball" for r in rows)
    gp = sum(r["george"] == "ball" for r in rows)
    o.append(f"'ball' precision: {fmt(tp, pp)}   recall: {fmt(tp, gp)}")
    # precision where George says ball-present (a box_off is a ball the detector found)
    tp2 = sum(r["pred"] == "ball" and r["george"] in ("ball", "box_off") for r in rows)
    o.append(f"'ball' precision vs George ball-or-box_off: {fmt(tp2, pp)}")
    # not_ball precision/recall (the negative class is the one that matters for pre-filtering)
    tn = sum(r["pred"] == "not_ball" and r["george"] == "not_ball" for r in rows)
    pn = sum(r["pred"] == "not_ball" for r in rows)
    gn = sum(r["george"] == "not_ball" for r in rows)
    o.append(f"'not_ball' precision: {fmt(tn, pn)}   recall: {fmt(tn, gn)}")
    o += ["", "confusion (rows George, cols model):", "| George \\ model | " + " | ".join(CLASSES) + " | total |",
          "|---|" + "---|" * (len(CLASSES) + 1)]
    for g in CLASSES:
        cnt = [sum(1 for r in rows if r["george"] == g and r["pred"] == p) for p in CLASSES]
        o.append(f"| {g} | " + " | ".join(map(str, cnt)) + f" | {sum(cnt)} |")
    # coverage vs agreement on the committed (non-unsure) cards
    o += ["", "coverage vs agreement (model confident >= t and not 'unsure'; agreement = same 4-class verdict as George):",
          "| t | coverage | agreement | 'ball' prec | 'not_ball' prec |", "|---|---|---|---|---|"]
    for t in (0.0, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95):
        s = [r for r in rows if r["pconf"] >= t and r["pred"] != "unsure"]
        ag = sum(r["pred"] == r["george"] for r in s)
        bp = [r for r in s if r["pred"] == "ball"]
        npr = [r for r in s if r["pred"] == "not_ball"]
        o.append(f"| {t} | {len(s)}/{n} = {len(s) / n:.2f} | {fmt(ag, len(s))} | "
                 f"{fmt(sum(r['george'] == 'ball' for r in bp), len(bp))} | "
                 f"{fmt(sum(r['george'] == 'not_ball' for r in npr), len(npr))} |")
    # per set / stratum
    o += ["", "per set (accuracy 4-class | n | 'ball' prec | 'not_ball' rec):", "| set | n | acc | ball prec | not_ball recall |",
          "|---|---|---|---|---|"]
    def groups(key):
        g = collections.defaultdict(list)
        for r in rows:
            g[key(r)].append(r)
        return g
    for key in (lambda r: r["set"], lambda r: r["stratum"]):
        for k, g in sorted(groups(key).items()):
            bp = [r for r in g if r["pred"] == "ball"]
            gn_ = [r for r in g if r["george"] == "not_ball"]
            o.append(f"| {k} | {len(g)} | {fmt(sum(r['pred'] == r['george'] for r in g), len(g))} | "
                     f"{fmt(sum(r['george'] == 'ball' for r in bp), len(bp))} | "
                     f"{fmt(sum(r['pred'] == 'not_ball' for r in gn_), len(gn_))} |")
        o.append("|||||")
    two = [r for r in rows if r.get("other_bbox")]
    if two:
        o.append(f"two-box cards: n={len(two)}, accuracy {fmt(sum(r['pred'] == r['george'] for r in two), len(two))}")
    cost = [r["cost"] for r in rows]
    wall = [r["wall"] for r in rows]
    out_tok = [r["usage"].get("output_tokens", 0) for r in rows]
    o += ["", f"cost/card (list-price equivalent): mean ${statistics.mean(cost):.4f}, median ${statistics.median(cost):.4f}, "
          f"total ${sum(cost):.2f}; per 1,000 cards ${1000 * statistics.mean(cost):.0f}",
          f"latency/card wall: mean {statistics.mean(wall):.1f}s median {statistics.median(wall):.1f}s (includes ~15 s CLI start-up; "
          f"3 in parallel); output tokens mean {statistics.mean(out_tok):.0f}"]
    return "\n".join(o)


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    split = None
    if "--skip" in sys.argv:
        SKIP.update(sys.argv[sys.argv.index("--skip") + 1].split(","))
        args = [a for a in args if a != sys.argv[sys.argv.index("--skip") + 1]]
    if "--split" in sys.argv:
        split = sys.argv[sys.argv.index("--split") + 1]
        args = [a for a in args if a != split]
    for run in args:
        print(report(run, split))
        print()
