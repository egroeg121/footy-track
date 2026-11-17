"""Top-up list v3: VitTrack-specific Ball Check cards (mini, numpy-free).  Same blinding rules as build_batch.py (v2): fixed 18 x 18 px boxes, seeded coin flip for which tracker is
``bbox``, hash-ordered, no note / provenance / role / confidence keys, hidden keys for everything else.  Far pairs (> 110 px) become two single-box twin cards.

Strata (quota, cards): vit_odd 24 (VitTrack is the odd one out, >= 2 others agree; or any other disagreement involving VitTrack; priority: inside a player box = the labeller's drift
onto a player / boot, fast, text seeds)  |  vit_alone 8 (only VitTrack has a ball)  |  vit_loss 8 (just before VitTrack's score falls below the found threshold while another tracker
still has the ball)  |  vit_agree 24 (blind sample: VitTrack within 8 px of >= 2 other trackers)  |  vit_pair 8 (VitTrack agrees with exactly one other tracker).
Skipped: seed frames, frames within 3 of any v2 card or earlier verdict.   python build_batch_v3.py --fused fused4_arm1.json.gz --v2 LIST_V2 --out LIST_V3
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import fusion as F
import fusion4 as G

BOX, VIEW_PX = 18.0, 110.0
QUOTA = {"vit_alone": 6, "vit_loss": 6, "vit_agree": 20, "vit_pair": 8, "vit_longrun": 10}
OLD = lambda c: c.replace("arsenal_nottinghamforest", "arsenal_norwich")
NEWN = lambda c: c.replace("arsenal_norwich", "arsenal_nottinghamforest")


def h(*parts) -> float:
    return int(hashlib.sha1(":".join(map(str, parts)).encode()).hexdigest()[:12], 16) / 16**12


def box(p):
    return {"x": round((p[0] - BOX / 2) / F.W, 6), "y": round((p[1] - BOX / 2) / F.H, 6), "w": round(BOX / F.W, 6), "h": round(BOX / F.H, 6)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fused", type=Path, required=True)
    ap.add_argument("--fused-other", type=Path, help="the other VitTrack arm, only to record its position as a hidden key")
    ap.add_argument("--v2", type=Path, default=Path("/mnt/storage/footy_data/nottinghamforest_label_queue/ball_check_list_v2_disagree.json"))
    ap.add_argument("--v1", type=Path, default=Path("/mnt/storage/footy_data/nottinghamforest_label_queue/ball_check_list_v1.json"))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--salt", default="vittrack-v3")
    ap.add_argument("--cards", type=int, default=72)
    ap.add_argument("--found", type=float, default=0.20)
    a = ap.parse_args()
    fused = F.load(a.fused)
    other = F.load(a.fused_other) if a.fused_other else None
    tab = G.table(fused)
    skip = defaultdict(set)
    for path in (a.v1, a.v2):
        for it in json.loads(path.read_text())["items"]:
            c = OLD(it["clip"])
            skip[c].update(range(it["frame_index"] - 3, it["frame_index"] + 4))
    for c, fr in fused["judged"].items():
        for f in fr:
            skip[c].update(range(f - 3, f + 4))

    def ok(c, t):
        cl = fused["clips"][c]
        if tab[c][t] is None or t in skip[c]:
            return False
        for k in ("t2", "t3", "t4"):
            if cl.get(k) is not None and cl[k]["is_seed"][t]:
                return False
        return True

    def meta(c, t, stratum, extra=None):
        cl = fused["clips"][c]
        pos = G.pos_at(cl, t)
        kind, info = tab[c][t]
        t4 = cl["t4"]
        return {"stratum": stratum, "class": kind, "class_info": info, "vit_role": G.vit_role(kind, info),
                "positions": {n: ([round(p[0], 1), round(p[1], 1)] if p else None) for n, p in zip(G.NAMES, pos)},
                "t4_raw": [t4["raw_x"][t], t4["raw_y"][t]] if t4 and t4["raw_x"][t] is not None else None, "t4_score": t4["score"][t] if t4 else None,
                "t4_gate": t4["gate"][t] if t4 else None, "t4_would_stop": t4["would_stop"][t] if t4 else None, "t4_in_player_box": t4["in_player_box"][t] if t4 else None,
                "t4_players_near": t4["near"][t] if t4 else None, "vit_arm": fused["vit"]["arm"], "vit_min_score": fused["vit"]["min_score"],
                "t4_other_arm": (([other["clips"][c]["t4"]["raw_x"][t], other["clips"][c]["t4"]["raw_y"][t]] if other and other["clips"][c]["t4"] and other["clips"][c]["t4"]["raw_x"][t] is not None else None)),
                "magnitude_px": round(G.magnitude(pos), 1), **(extra or {})}

    cands = defaultdict(list)       # stratum -> [(prio, clip, t, meta, bbox_src, other_src)]
    # ---- disagreements involving VitTrack (segments)
    for c, cl in fused["clips"].items():
        rows = tab[c]
        flags = [r is not None and G.vit_role(*r) in ("vit_odd_one_out", "vit_split", "vit_all_differ") for r in rows]
        for lo, hi in F.runs(flags, gap=2):
            ts = [t for t in range(lo, hi + 1) if flags[t] and ok(c, t)]
            if not ts:
                continue
            view = [t for t in ts if G.magnitude(G.pos_at(cl, t)) <= 250]
            t = max(view or ts, key=lambda t: G.magnitude(G.pos_at(cl, t)))
            pos = G.pos_at(cl, t)
            kind, info = tab[c][t]
            if kind == "odd_one":
                mates = [k for k in info["agree"]]
                other_src = mates[int(h(a.salt, c, t, "mate") * len(mates))]
                pair = ["t4", other_src]
            elif kind == "split":
                cl2 = info["clusters"]
                mine = [x for x in cl2 if "t4" in x][0]
                theirs = [x for x in cl2 if "t4" not in x][0]
                pair = ["t4", theirs[int(h(a.salt, c, t, "mate") * len(theirs))]]
            else:
                pair = ["t4", [n for n, p in zip(G.NAMES, pos) if p is not None and n != "t4"][0]]
            b, o = (pair[0], pair[1]) if h(a.salt, c, t, "order") < 0.5 else (pair[1], pair[0])
            far = G.dist(pos[G.NAMES.index(b)], pos[G.NAMES.index(o)]) > VIEW_PX
            t4 = cl["t4"]
            speed = 0.0
            pr = 1.0 + (2.0 if t4["in_player_box"][t] else 0) + (1.0 if t4["near"][t] >= 2 else 0) + (0.5 if len(ts) >= 5 else 0) + (1.0 if info and kind == "odd_one" else 0) + h(a.salt, c, t, "prio")
            cands["vit_odd"].append((pr, c, t, meta(c, t, "vit_odd", {"segment": f"{c}:{lo}-{hi}", "segment_frames": len(ts), "pair_far": far}), b, o))
    # ---- unattended long run (arm 1: autodetect on frame 0, one Run to the end of the clip): VitTrack >= 60 frames into the run, off the A++/SAM consensus
    if other is not None:
        for c, cl in fused["clips"].items():
            o4 = other["clips"][c].get("t4")
            if not o4:
                continue
            for t in range(60, cl["n"]):
                if not ok(c, t) or o4["raw_x"][t] is None:
                    continue
                base = [None if p is None else p for p in G.pos_at(cl, t)[:3]]
                agree = [i for i in range(3) if base[i] is not None and any(j != i and base[j] is not None and G.dist(base[i], base[j]) <= G.TOL for j in range(3))]
                if len(agree) < 2:
                    continue
                vp = (o4["raw_x"][t], o4["raw_y"][t])
                if G.dist(vp, base[agree[0]]) <= G.TOL * 2:
                    continue
                pos4 = list(base) + [vp]
                m = meta(c, t, "vit_longrun", {"vit_arm": 1, "t4_score": o4["score"][t], "t4_in_player_box": o4["in_player_box"][t], "t4_players_near": o4["near"][t], "run_frames": t, "t4_gate": o4["gate"][t], "t4_would_stop": None})
                m["positions"]["t4"] = [round(vp[0], 1), round(vp[1], 1)]
                m["class"], m["class_info"] = "odd_one", {"odd": "t4", "n_present": 4, "agree": [G.NAMES[i] for i in agree], "absent": []}
                m["vit_role"] = "vit_odd_one_out"
                m["magnitude_px"] = round(G.dist(vp, base[agree[0]]), 1)
                m["pair_far"] = m["magnitude_px"] > VIEW_PX
                mate = G.NAMES[agree[int(h(a.salt, c, t, "mate") * len(agree))]]
                b, o = ("t4", mate) if h(a.salt, c, t, "order") < 0.5 else (mate, "t4")
                cands["vit_longrun"].append((h(a.salt, c, t, "p"), c, t, m, b, o))
    # ---- single-frame strata
    for c, cl in fused["clips"].items():
        n = cl["n"]
        t4 = cl["t4"]
        if not t4:
            continue
        for t in range(n):
            if not ok(c, t):
                continue
            kind, info = tab[c][t]
            role = G.vit_role(kind, info)
            pos = G.pos_at(cl, t)
            if role == "vit_alone":
                cands["vit_alone"].append((h(a.salt, c, t, "p"), c, t, meta(c, t, "vit_alone"), "t4", None))
            elif role == "vit_agrees_all" and info["n_present"] >= 3:
                names = [nm for nm, p in zip(G.NAMES, pos) if p is not None]
                cands["vit_agree"].append((h(a.salt, c, t, "p"), c, t, meta(c, t, "vit_agree"), names[int(h(a.salt, c, t, "o") * len(names))], None))
            elif role == "vit_agrees_pair":
                names = [nm for nm, p in zip(G.NAMES, pos) if p is not None]
                cands["vit_pair"].append((h(a.salt, c, t, "p"), c, t, meta(c, t, "vit_pair", {"pair": "+".join(names)}), names[int(h(a.salt, c, t, "o") * 2)], None))
            # loss: VitTrack's score is >= found at t and < found at t+1..t+3 while another tracker has the ball at t
            if t4["x"][t] is not None and t + 3 < n and all(t4["x"][t + k] is None for k in (1, 2, 3)) and any(pos[i] is not None for i in range(3)):
                others = [nm for nm, p in zip(G.NAMES[:3], pos[:3]) if p is not None]
                far = G.dist(pos[3], pos[G.NAMES.index(others[0])]) > G.TOL
                cands["vit_loss"].append((h(a.salt, c, t, "p"), c, t, meta(c, t, "vit_loss", {"loss_side": "before"}), "t4", others[0] if far else None))

    def select(pool, n, prefer=False, cost=lambda it: 1):
        by_clip = defaultdict(list)
        for it in pool:
            by_clip[it[1]].append(it)
        for c in by_clip:
            by_clip[c].sort(key=lambda it: -it[0] if prefer else it[0])
        order = sorted(by_clip, key=lambda c: h(a.salt, c, "clip"))
        out, used, spent = [], set(), 0
        while spent < n and any(by_clip.values()):
            for c in order:
                while by_clip[c]:
                    it = by_clip[c].pop(0)
                    if any(abs(it[2] - t2) < 12 for (c2, t2) in used if c2 == c):
                        continue
                    used.add((c, it[2]))
                    out.append(it)
                    spent += cost(it)
                    break
                if spent >= n:
                    break
        return out

    chosen, taken = [], set()
    for st, q in QUOTA.items():
        sel = select([it for it in cands[st] if (it[1], it[2]) not in taken], q, cost=lambda it: 2 if it[3].get("pair_far") else 1)
        chosen += sel
        taken |= {(x[1], x[2]) for x in sel}
    ncards = lambda L: sum(2 if it[3].get("pair_far") else 1 for it in L)
    dsel = select([it for it in cands["vit_odd"] if (it[1], it[2]) not in taken], max(0, a.cards - ncards(chosen)), prefer=True, cost=lambda it: 2 if it[3]["pair_far"] else 1)
    chosen += dsel
    if ncards(chosen) < a.cards:     # top up with more blind agreement
        sel = select([it for it in cands["vit_agree"] if (it[1], it[2]) not in taken | {(x[1], x[2]) for x in chosen}], a.cards - ncards(chosen))
        chosen += sel

    items = []
    for pr, c, t, m, b, o in chosen:
        pos = [tuple(m["positions"][k]) if m["positions"].get(k) else None for k in G.NAMES]
        twins = [(b, None), (o, None)] if m.get("pair_far") else [(b, o)]
        for b_, o_ in twins:
            it = {"clip": c, "frame_index": t, "bbox": box(pos[G.NAMES.index(b_)])}
            if o_ is not None:
                it["other_bbox"] = box(pos[G.NAMES.index(o_)])
            it["label"] = "in_play_ball"
            it.update({"src_bbox": b_, "src_other": o_, **m, "far_twin": bool(m.get("pair_far"))})
            items.append(it)
    items.sort(key=lambda it: h(a.salt, it["clip"], it["frame_index"], it["stratum"], it["src_bbox"], "order"))
    for i, it in enumerate(items):
        it["v3_index"] = i
        it["clip_run_id"] = it["clip"]
        it["clip"] = NEWN(it["clip"])
    pop = {"vit_agree": dict(Counter(it[1] for it in cands["vit_agree"]))}
    pop = {k: {NEWN(c): n for c, n in v.items()} for k, v in pop.items()}
    doc = {"population": pop, "description": "ball_check_list_v3_vittrack: VitTrack-specific top-up (blind); hidden keys: src_bbox/src_other (t1 A++/B1, t2 SAM2.1, t3 SAM3, t4 labeller autodetect+VitTrack), stratum, positions",
           "salt": a.salt, "tolerance_px": F.TOL, "box_px": BOX, "vit": fused["vit"], "items": items}
    a.out.write_text(json.dumps(doc))
    print("cards", len(items), "clips", len({it["clip"] for it in items}), dict(Counter(it["stratum"] for it in items)))
    print("far twins", sum(1 for it in items if it["far_twin"]), "per-clip max", max(Counter(it["clip"] for it in items).values()), {k: len(v) for k, v in cands.items()})


if __name__ == "__main__":
    main()
