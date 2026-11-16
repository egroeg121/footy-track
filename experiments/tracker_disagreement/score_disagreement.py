"""Score George's blind verdicts on ball_check_list_v2_disagree.json (mini; stdlib only).

    python score_disagreement.py --list /mnt/storage/footy_data/norwich_label_queue/ball_check_list_v2_disagree.json \
        --verdicts /mnt/storage/footy_data/ball_gt_marks_norwich_disagree/ball_checks   (or --results job.json) [--out report.md] [--json report.json] [--selftest]

Verdicts are read like the labeller does (``<clip>.jsonl`` append-only, last line per (source, clip, frame_index, box_index) wins, ``verdict: null`` = undo tombstone) and joined to
the list by that key; a verdict is used only when its ``bbox`` equals the list card's bbox (guards against a re-ordered / regenerated list), otherwise it is counted as ``mismatch``.

Truth model (hit radius R = 10 px in 1080p pixels, the project's recall@10 convention):
  ball       the ball is under the shown (bbox) box          truth = centre of bbox
  box_off    ball there but the box is loose                 truth = centre of bbox (lenient; strict scoring drops these)
  corrected  George dragged a box onto the ball              truth = centre of corrected_bbox
  not_ball   no ball under bbox                              truth unknown: bbox tracker WRONG; other_bbox tracker UNRESOLVED (the ball may or may not be there)
  unsure     dropped
A tracker with a position on the card is RIGHT when it is within R of the truth, WRONG when truth is known and it is farther, and on ``not_ball`` WRONG when within R of the
shown bbox.  A tracker with no ball on a truth-known card is a MISS (not in precision).  To mark the magenta box right, George must use "corrected" and drag onto it
(a ``not_ball`` cannot say whether the other box is right: counted as unresolved, reported per tracker).
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

R = 10.0
W, H = 1920.0, 1080.0
NAMES = ("t1", "t2", "t3")
LABEL = {"t1": "A++/B1", "t2": "SAM2.1", "t3": "SAM3"}
MIN_N = 8


def wilson(k: int, n: int, z: float = 1.96):
    if n == 0:
        return (float("nan"), float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, max(0.0, c - h), min(1.0, c + h)


def fmt(k, n):
    if n == 0:
        return "-"
    p, lo, hi = wilson(k, n)
    return f"{k}/{n} = {p:.2f} [{lo:.2f}, {hi:.2f}]"


def centre(b):
    return ((b["x"] + b["w"] / 2) * W, (b["y"] + b["h"] / 2) * H)


def read_verdicts(vdir: Path) -> dict:
    out = {}
    for p in sorted(vdir.glob("*.jsonl")):
        for raw in p.read_text().splitlines():
            if not raw.strip():
                continue
            try:
                r = json.loads(raw)
                key = (r.get("source") or "sidecar", r["clip"], int(r["frame_index"]), int(r["box_index"]))
            except (ValueError, KeyError, TypeError):
                continue
            if r.get("verdict") is None:
                out.pop(key, None)
            else:
                out[key] = r
    return out


def same_box(a, b, tol=1e-4):
    return a is not None and b is not None and all(abs(a[k] - b[k]) <= tol for k in ("x", "y", "w", "h"))


def judge(card: dict, v: dict) -> dict:
    """Per-card outcome: {truth: (x, y) | None, verdict, per_tracker: {t: right|wrong|miss|unresolved|absent}}."""
    verdict = v["verdict"]
    pos = {k: tuple(p) if p else None for k, p in card["positions"].items()}
    shown = centre(card["bbox"])
    truth = None
    if verdict in ("ball", "box_off"):
        truth = shown
    elif verdict == "corrected" and v.get("corrected_bbox"):
        truth = centre(v["corrected_bbox"])
    res = {}
    shown_pts = [shown] + ([centre(card["other_bbox"])] if card.get("other_bbox") else [])
    for k in NAMES:
        p = pos[k]
        if p is not None and not any(math.hypot(p[0] - q[0], p[1] - q[1]) <= 12 for q in shown_pts):
            res[k] = "notshown"      # this tracker's position is not on the card (a far pair is judged by its twin card): not scored here
            continue
        if truth is not None:
            res[k] = "absent" if p is None else ("right" if math.hypot(p[0] - truth[0], p[1] - truth[1]) <= R else "wrong")
            if p is None:
                res[k] = "miss"
        elif verdict == "not_ball":
            if p is None:
                res[k] = "absent"
            elif math.hypot(p[0] - shown[0], p[1] - shown[1]) <= R:
                res[k] = "wrong"
            else:
                res[k] = "unresolved"
        else:
            res[k] = "absent"
    return {"verdict": verdict, "truth": truth, "per_tracker": res}


def situations(card: dict) -> list[str]:
    s = [f"stratum:{card['stratum']}"]
    cls = card["class"]
    s.append(f"class:{cls}" + (f":{card['class_detail']}" if cls in ("two_agree",) and card.get("class_detail") else ""))
    s.append("fast (>12 px/frame)" if card.get("speed_px", 0) > 12 else "slow/medium (<=12 px/frame)")
    crowd = card.get("players_near", 0) >= 2 or card.get("in_player_box") or card.get("crowd_y")
    s.append("crowd / occlusion proxy" if crowd else "open ball")
    if card.get("text_only"):
        s.append("text-seed only")
    srcs = {v for v in (card.get("seed_src") or {}).values() if v}
    if "text" in srcs:
        s.append("a SAM tracker follows a text seed")
    if card.get("stratum") == "loss":
        s.append(f"loss {card.get('loss_side')} ({card.get('loss_tracker')} loses)")
    if card.get("a_train_dist") is not None and card["a_train_dist"] <= 2:
        s.append("A++ trained within 2 frames")
    return s


def build(list_doc: dict, verdicts: dict) -> dict:
    cards = list_doc["items"]
    rows, mismatch, unsure, missing = [], 0, 0, 0
    for i, c in enumerate(cards):
        v = verdicts.get(("list", c["clip"], c["frame_index"], i))
        if v is None:
            missing += 1
            continue
        if not same_box(v.get("bbox"), c["bbox"]):
            mismatch += 1
            continue
        if v["verdict"] == "unsure":
            unsure += 1
            continue
        rows.append((c, judge(c, v)))
    return {"n_cards": len(cards), "judged": len(rows), "unjudged": missing, "unsure": unsure, "mismatch": mismatch, "rows": rows}


def report(list_doc: dict, st: dict) -> tuple[str, dict]:
    rows = st["rows"]
    L, J = [], {"n_cards": st["n_cards"], "judged": st["judged"], "unsure": st["unsure"], "unjudged": st["unjudged"], "mismatch": st["mismatch"]}
    L.append(f"# Tracker disagreement batch: scores\n\n{st['judged']} judged cards of {st['n_cards']} ({st['unjudged']} not yet judged, {st['unsure']} unsure, {st['mismatch']} verdicts whose box did not match the list).\n")
    byst = defaultdict(Counter)
    for c, j in rows:
        byst[c["stratum"]][j["verdict"]] += 1
    L.append("## Verdicts by stratum\n\n| stratum | n | ball | box_off | corrected | not_ball |\n|---|---|---|---|---|---|")
    for s, cn in sorted(byst.items()):
        L.append(f"| {s} | {sum(cn.values())} | {cn['ball']} | {cn['box_off']} | {cn['corrected']} | {cn['not_ball']} |")
    J["verdicts_by_stratum"] = {s: dict(cn) for s, cn in byst.items()}

    # ---- agreed frames
    pop = list_doc.get("population", {})
    L.append("\n## Agreed-frame precision (blind random samples)\n")
    L.append("P(ball under the agreed position). strict = verdict ball (or corrected within 10 px of the shown box); lenient = also box_off and any corrected. Weighted = each card weighted by "
             "(eligible frames of its clip) / (cards of its clip): the sample is spread round-robin over clips, not proportional to frames.\n")
    L.append("| sample | n | strict precision (95% Wilson) | lenient precision | frame-weighted strict |\n|---|---|---|---|---|")
    J["agreed"] = {}
    groups = {"all three agree": lambda c: c["stratum"] == "all3_agree", "pair t1+t2 (A++/B1 + SAM2.1), SAM3 silent": lambda c: c.get("pair") == "t1+t2",
              "pair t1+t3 (A++/B1 + SAM3), SAM2.1 silent": lambda c: c.get("pair") == "t1+t3", "pair t2+t3 (SAM2.1 + SAM3), A++/B1 silent": lambda c: c.get("pair") == "t2+t3",
              "all three agree, A++ trained within 2 frames": lambda c: c["stratum"] == "all3_agree" and c.get("a_train_dist") is not None and c["a_train_dist"] <= 2,
              "all three agree, A++ not trained near": lambda c: c["stratum"] == "all3_agree" and not (c.get("a_train_dist") is not None and c["a_train_dist"] <= 2)}
    popkey = {"all three agree": "all3_agree", "pair t1+t2 (A++/B1 + SAM2.1), SAM3 silent": "pair_agree:t1+t2", "pair t1+t3 (A++/B1 + SAM3), SAM2.1 silent": "pair_agree:t1+t3",
              "pair t2+t3 (SAM2.1 + SAM3), A++/B1 silent": "pair_agree:t2+t3"}
    for name, f in groups.items():
        sub = [(c, j) for c, j in rows if f(c)]
        n = len(sub)
        strict = sum(1 for c, j in sub if j["verdict"] == "ball" or (j["verdict"] == "corrected" and j["truth"] and math.hypot(j["truth"][0] - centre(c["bbox"])[0], j["truth"][1] - centre(c["bbox"])[1]) <= R))
        lenient = sum(1 for c, j in sub if j["verdict"] in ("ball", "box_off", "corrected"))
        w = ""
        if name in popkey and sub:
            per = Counter(c["clip"] for c, _ in sub)
            pc = pop.get(popkey[name], {})
            num = den = 0.0
            for c, j in sub:
                wt = pc.get(c["clip"], 0) / per[c["clip"]]
                den += wt
                num += wt * (1 if (j["verdict"] == "ball" or (j["verdict"] == "corrected" and j["truth"] and math.hypot(j["truth"][0] - centre(c["bbox"])[0], j["truth"][1] - centre(c["bbox"])[1]) <= R)) else 0)
            w = f"{num / den:.2f}" if den else "-"
        L.append(f"| {name} | {n} | {fmt(strict, n)} | {fmt(lenient, n)} | {w or '-'} |")
        J["agreed"][name] = {"n": n, "strict": strict, "lenient": lenient, "strict_ci": wilson(strict, n)[1:] if n else None, "weighted_strict": w}

    # ---- disagreement cards: who was right
    L.append("\n## Disagreement cards: who was right\n")
    dis = [(c, j) for c, j in rows if c["stratum"] == "disagree" or "other_bbox" in c]
    per = {k: Counter() for k in NAMES}
    for c, j in dis:
        for k in NAMES:
            per[k][j["per_tracker"][k]] += 1
    L.append("Per tracker over the disagreement cards where it had a position: right / (right + wrong) among cards with a known truth (ball, box_off, corrected) or wrong on not_ball "
             "(unresolved = not_ball on a card where this tracker was the other box).\n")
    L.append("| tracker | positions on cards | right | wrong | unresolved | misses (no ball, truth known) | precision (right / (right + wrong), 95% Wilson) |\n|---|---|---|---|---|---|---|")
    J["disagree_per_tracker"] = {}
    for k in NAMES:
        r, w_, u, m = per[k]["right"], per[k]["wrong"], per[k]["unresolved"], per[k]["miss"]
        # precision restricted to cards where the tracker was resolved (right or wrong)
        L.append(f"| {LABEL[k]} | {r + w_ + u} | {r} | {w_} | {u} | {m} | {fmt(r, r + w_)} |")
        J["disagree_per_tracker"][k] = {"right": r, "wrong": w_, "unresolved": u, "miss": m}
    L.append("\nHead to head (cards of the disagree stratum where both trackers have positions more than 8 px apart): A right / B right / both wrong / unresolved (not_ball with both positions).\n")
    L.append("| pair | cards | first right | second right | both wrong | unresolved |\n|---|---|---|---|---|---|")
    J["head_to_head"] = {}
    for a_, b_ in (("t1", "t2"), ("t1", "t3"), ("t2", "t3")):
        n = ra = rb = bw = un = 0
        for c, j in dis:
            pa, pb = c["positions"][a_], c["positions"][b_]
            if pa is None or pb is None or math.hypot(pa[0] - pb[0], pa[1] - pb[1]) <= 8:
                continue
            ta, tb = j["per_tracker"][a_], j["per_tracker"][b_]
            n += 1
            if ta == "right" and tb != "right":
                ra += 1
            elif tb == "right" and ta != "right":
                rb += 1
            elif ta == "wrong" and tb == "wrong":
                bw += 1
            else:
                un += 1
        L.append(f"| {LABEL[a_]} vs {LABEL[b_]} | {n} | {ra} | {rb} | {bw} | {un} |")
        J["head_to_head"][f"{a_}-{b_}"] = {"n": n, "first_right": ra, "second_right": rb, "both_wrong": bw, "unresolved": un}

    # ---- trust by situation
    L.append("\n## Which tracker to trust where\n")
    L.append(f"Per situation, precision of each tracker on every judged card (any stratum) where it had a position and the truth is known or it sat under a not_ball box; "
             f"`best` only when n >= {MIN_N} for at least two trackers and the best tracker's lower Wilson bound is above the other's point estimate, otherwise `no clear winner`.\n")
    sits = defaultdict(lambda: {k: [0, 0] for k in NAMES})
    for c, j in rows:
        for s in situations(c):
            for k in NAMES:
                t = j["per_tracker"][k]
                if t == "right":
                    sits[s][k][0] += 1
                    sits[s][k][1] += 1
                elif t == "wrong":
                    sits[s][k][1] += 1
    L.append("| situation | " + " | ".join(LABEL[k] for k in NAMES) + " | verdict |\n|---|---|---|---|---|")
    J["situations"] = {}
    for s in sorted(sits):
        cells = []
        stats = {}
        for k in NAMES:
            r, n = sits[s][k]
            cells.append(fmt(r, n))
            stats[k] = (r, n)
        ok = [k for k in NAMES if stats[k][1] >= MIN_N]
        best = "n < %d" % MIN_N
        if len(ok) >= 2:
            ranked = sorted(ok, key=lambda k: -stats[k][0] / stats[k][1])
            top = ranked[0]
            lo_top = wilson(*stats[top])[1]
            if all(lo_top > stats[o][0] / stats[o][1] for o in ranked[1:]):
                best = f"best: {LABEL[top]}"
            else:
                best = "no clear winner"
        L.append(f"| {s} | " + " | ".join(cells) + f" | {best} |")
        J["situations"][s] = {"stats": stats, "verdict": best}
    return "\n".join(L) + "\n", J


def selftest():
    """Synthetic verdicts (NOT George's) to exercise the join and the truth model."""
    rnd = random.Random(1)
    items = []
    for i in range(60):
        base = (rnd.uniform(200, 1700), rnd.uniform(200, 900))
        pos = {"t1": list(base), "t2": [base[0] + rnd.choice([0, 3, 30]), base[1]], "t3": [base[0] + rnd.choice([0, 3, -40]), base[1] + 2] if i % 4 else None}
        strat = ["disagree", "all3_agree", "single", "pair_agree"][i % 4]
        b = lambda p: {"x": (p[0] - 9) / W, "y": (p[1] - 9) / H, "w": 18 / W, "h": 18 / H}
        it = {"clip": f"c{i % 5}", "frame_index": i, "bbox": b(pos["t1"]), "label": "in_play_ball", "stratum": strat, "class": "two_agree", "class_detail": "t3", "positions": pos,
              "speed_px": 5 + i % 20, "players_near": i % 3, "in_player_box": 0, "crowd_y": False, "a_train_dist": i % 9, "seed_src": {"t2": "rtdetr", "t3": "text"}, "pair": "t1+t2"}
        if strat == "disagree":
            it["other_bbox"] = b(pos["t2"])
        items.append(it)
    doc = {"items": items, "population": {"all3_agree": {f"c{j}": 100 for j in range(5)}, "pair_agree:t1+t2": {f"c{j}": 50 for j in range(5)}}}
    d = Path(tempfile.mkdtemp())
    for i, it in enumerate(items):
        v = rnd.choice(["ball", "ball", "not_ball", "corrected", "box_off", "unsure"])
        rec = {"clip": it["clip"], "frame_index": it["frame_index"], "box_index": i, "source": "list", "verdict": v, "bbox": it["bbox"], "corrected_bbox": None}
        if v == "corrected":
            rec["corrected_bbox"] = {"x": (it["positions"]["t2"][0] - 9) / W, "y": (it["positions"]["t2"][1] - 9) / H, "w": 18 / W, "h": 18 / H}
        with open(d / f"{it['clip']}.jsonl", "a") as fh:
            fh.write(json.dumps(rec) + "\n")
    with open(d / "c0.jsonl", "a") as fh:   # undo tombstone for card 0 and a stale v1-style record
        fh.write(json.dumps({"clip": "c0", "frame_index": 0, "box_index": 0, "source": "list", "verdict": None}) + "\n")
    st = build(doc, read_verdicts(d))
    assert st["unjudged"] >= 1 and st["mismatch"] == 0, st
    text, J = report(doc, st)
    print(text)
    print("selftest ok")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", type=Path, default=Path("/mnt/storage/footy_data/norwich_label_queue/ball_check_list_v2_disagree.json"))
    ap.add_argument("--verdicts", type=Path, default=Path("/mnt/storage/footy_data/ball_gt_marks_norwich_disagree/ball_checks"))
    ap.add_argument("--results", type=Path, help="`labelling_jobs results <job> -o job.json` (labelling-job-results/v1) instead of reading the verdict logs")
    ap.add_argument("--out", type=Path)
    ap.add_argument("--json", type=Path)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        selftest()
        return
    doc = json.loads(a.list.read_text())
    if a.results:
        rows = json.loads(a.results.read_text())["labels"]
        ver = {(r.get("source") or "sidecar", r["clip"], int(r["frame_index"]), int(r["box_index"])): r for r in rows if r.get("verdict")}
    else:
        ver = read_verdicts(a.verdicts)
    st = build(doc, ver)
    text, J = report(doc, st)
    print(text)
    if a.out:
        a.out.write_text(text)
    if a.json:
        a.json.write_text(json.dumps(J, indent=1, default=str))


if __name__ == "__main__":
    main()
