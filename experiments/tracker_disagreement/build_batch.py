"""Blind Ball Check list v2 from the fused tracker positions (mini, numpy-free).

Strata (quotas; filled round-robin across clips so no clip dominates):
  disagree     one card per disagreement segment (frame of maximum disagreement); priority: SAM3 vs the others, text-only seeds, fast balls, crowd / occlusion, loss edges
  text_only    segment where only a SAM text seed (no RT-DETR anchor within 40 frames, A++/B1 silent) put a ball on the frame
  loss         just before (t_last) / after (t_last + 2) a tracker loses the ball while another still has it
  single       only one tracker has a ball
  all3_agree   blind random sample of frames where all three agree within 8 px (agreed-frame precision)
  pair_agree   two trackers agree, the third has no ball (8 per pair type: which pairs can be trusted)
Skipped: seed frames, the dense-labelling windows +-25, frames within 3 of a frame George already judged (v1 verdicts).

Card = {clip, frame_index, bbox, other_bbox?, label, <hidden keys>}.  bbox / other_bbox are a FIXED 18 x 18 px box centred on the tracker position (a tracker's own box size would
leak which tracker it is); which tracker goes in bbox is a seeded coin flip, recorded as ``src_bbox`` / ``src_other``.  No ``note`` / ``provenance`` / ``role`` / ``confidence``
keys (the Ball Check page prints those).  Order = sha1 of (salt, clip, frame, stratum): fixed once published.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import fusion as F

BOX = 18.0
VIEW_PX = 110.0      # the magenta box is inside the default (pad 6) 240 px crop when the pair is closer than this
QUOTA = {"text_only": 12, "loss": 14, "single": 20, "all3_agree": 40, "pair_agree": 34}


OLD = lambda c: c.replace("arsenal_nottinghamforest", "arsenal_norwich")
NEWN = lambda c: c.replace("arsenal_norwich", "arsenal_nottinghamforest")


def h(*parts) -> float:
    return int(hashlib.sha1(":".join(map(str, parts)).encode()).hexdigest()[:12], 16) / 16**12


def box(p):
    return {"x": round((p[0] - BOX / 2) / F.W, 6), "y": round((p[1] - BOX / 2) / F.H, 6), "w": round(BOX / F.W, 6), "h": round(BOX / F.H, 6)}


def speed(cl, t):
    best = 0.0
    for k in F.NAMES:
        tr = cl[k]
        if tr is None:
            continue
        for a, b in ((t - 1, t), (t, t + 1)):
            if 0 <= a and b < cl["n"] and tr["x"][a] is not None and tr["x"][b] is not None:
                best = max(best, F.dist((tr["x"][a], tr["y"][a]), (tr["x"][b], tr["y"][b])))
    return round(best, 1)


def ctx(cl, t):
    t1 = cl["t1"]
    ys = [p[1] for p in F.pos_at(cl, t) if p is not None]
    return {"players_near": t1["players_near"][t], "in_player_box": t1["in_player_box"][t], "crowd_y": bool(ys and min(ys) < 0.22 * F.H)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fused", type=Path, required=True)
    ap.add_argument("--seeds", type=Path, required=True)
    ap.add_argument("--train-sel", type=Path, default=Path("/mnt/storage/exp/temporal/round3/norwich_fuse/selection.jsonl"))
    ap.add_argument("--v1", type=Path, default=Path("/mnt/storage/footy_data/nottinghamforest_label_queue/ball_check_list_v1.json"))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--salt", default="disagree-v2")
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--skip-near", type=int, default=3)
    ap.add_argument("--cards", type=int, default=200)
    a = ap.parse_args()
    fused = F.load(a.fused)
    tab = F.frame_table(fused)
    rng = random.Random(a.seed)
    seed_frames = defaultdict(list)       # clip -> [(frame, source)]
    for line in a.seeds.read_text().splitlines():
        if line.strip():
            s = json.loads(line)
            seed_frames[s["clip"]].append((s["frame"], s["source"]))
    trained = defaultdict(list)
    if a.train_sel.exists():
        for line in a.train_sel.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                trained[r["clip"]].append(r.get("t_orig", r.get("t")))
    v1 = json.loads(a.v1.read_text())["items"]
    for it in v1:
        it["clip"] = OLD(it["clip"])      # work in this run's (old) clip ids; the list is written with the new ones
    v1_keys = {(it["clip"], it["frame_index"]) for it in v1}
    skip = defaultdict(set)
    for c, fr in fused["judged"].items():
        for f in fr:
            skip[c].update(range(f - a.skip_near, f + a.skip_near + 1))
    for c, f in v1_keys:
        skip[c].update(range(f - a.skip_near, f + a.skip_near + 1))

    def ok(c, t):
        cl = fused["clips"][c]
        if tab[c][t] is None or t in skip[c]:
            return False
        for k in ("t2", "t3"):
            if cl[k] is not None and cl[k]["is_seed"][t]:
                return False
        return True

    def meta(c, t, extra):
        cl = fused["clips"][c]
        pos = F.pos_at(cl, t)
        kind, det = tab[c][t]
        m = {"stratum": extra.pop("stratum"), "class": kind, "class_detail": det, "positions": {n: ([round(p[0], 1), round(p[1], 1)] if p else None) for n, p in zip(F.NAMES, pos)},
             "scores": {k: (cl[k]["score"][t] if cl[k] is not None else None) for k in ("t2", "t3")},
             "seed_src": {k: (cl[k]["seed_src"][t] if cl[k] is not None else None) for k in ("t2", "t3")},
             "gate": {k: (cl[k]["gate"][t] if cl[k] is not None else None) for k in ("t2", "t3")},
             "t1_node": cl["t1"]["node"][t], "t1_conf": cl["t1"]["conf"][t], "speed_px": speed(cl, t), **ctx(cl, t),
             "a_train_dist": (min((abs(t - x) for x in trained[c]), default=None)), "magnitude_px": round(F.magnitude(pos), 1)}
        m.update(extra)
        return m

    def pick_src(c, t, names, force=None):
        """(src_bbox, src_other) from the candidate tracker names, seeded coin flip for the order."""
        names = list(names)
        a_, b_ = names[0], names[1] if len(names) > 1 else None
        if h(a.salt, c, t, "order") < 0.5 and b_ is not None:
            a_, b_ = b_, a_
        return a_, b_

    cands: dict[str, list] = defaultdict(list)       # stratum -> [(priority, clip, t, meta, bbox_src, other_src)]

    # ---- disagreement segments
    for seg in F.disagreement_segments(fused, tab):
        c = seg["clip"]
        cl = fused["clips"][c]
        ts = [t for t in seg["frames"] if ok(c, t)]
        if not ts:
            continue
        ts_view = [t for t in ts if F.magnitude(F.pos_at(cl, t)) <= 250]
        t = max(ts_view or ts, key=lambda t: F.magnitude(F.pos_at(cl, t)))
        pos = F.pos_at(cl, t)
        kind, det = tab[c][t]
        if kind == "two_agree":
            odd = det
            agree = [n for n in F.NAMES if n != odd]
            mate = agree[0] if h(a.salt, c, t, "mate") < 0.5 else agree[1]
            names = [odd, mate]
        elif kind == "pair_differ":
            names = [n for n, p in zip(F.NAMES, pos) if p is not None]
        else:
            with3 = [n for n in F.NAMES if n != "t3"]
            names = ["t3", with3[0] if h(a.salt, c, t, "mate") < 0.5 else with3[1]]
        bsrc, osrc = pick_src(c, t, names)
        involves3 = "t3" in names
        text_only = any(cl[k] is not None and cl[k]["seed_src"][t] == "text" for k in ("t2", "t3")) and not any(abs(t - f) <= 40 for f, s in seed_frames[c] if s == "rtdetr")
        sp = speed(cl, t)
        cx = ctx(cl, t)
        pr = 1.0 + (2.0 if involves3 else 0) + (1.5 if text_only else 0) + (1.0 if sp > 12 else 0) + (1.0 if (cx["players_near"] >= 2 or cx["in_player_box"] or cx["crowd_y"]) else 0) \
            + (0.5 if len(ts) >= 5 else 0) + h(a.salt, c, t, "prio")
        far = F.dist(pos[F.NAMES.index(bsrc)], pos[F.NAMES.index(osrc)]) > VIEW_PX
        cands["disagree"].append((pr, c, t, meta(c, t, {"stratum": "disagree", "segment": f"{c}:{seg['lo']}-{seg['hi']}", "segment_frames": seg["n_dis"], "segment_main": seg["main"],
                                                       "text_only": text_only, "pair_far": far}), bsrc, osrc))

    # ---- text-only (no disagreement needed): a SAM tracker has a ball from a text seed, nothing from A++/B1 and no RT-DETR anchor near
    for c, cl in fused["clips"].items():
        flags = []
        for t in range(cl["n"]):
            tx = tab[c][t] is not None and cl["t1"]["x"][t] is None and any(cl[k] is not None and cl[k]["x"][t] is not None and cl[k]["seed_src"][t] == "text" for k in ("t2", "t3")) \
                and not any(abs(t - f) <= 40 for f, s in seed_frames[c] if s == "rtdetr")
            flags.append(tx)
        for lo, hi in F.runs(flags):
            ts = [t for t in range(lo, hi + 1) if flags[t] and ok(c, t)]
            if len(ts) < 3:
                continue
            t = ts[len(ts) // 2]
            who = [k for k in ("t3", "t2") if cl[k] is not None and cl[k]["x"][t] is not None]
            cands["text_only"].append((1.0 + h(a.salt, c, t, "prio"), c, t, meta(c, t, {"stratum": "text_only", "segment": f"{c}:{lo}-{hi}", "segment_frames": len(ts)}), who[0], None))

    # ---- loss edges
    for ev in F.presence_events(fused, tab):
        c, nm, tl = ev["clip"], ev["tracker"], ev["t_last"]
        cl = fused["clips"][c]
        before = h(a.salt, c, tl, "loss") < 0.5
        t = tl if before else tl + 2
        if t >= cl["n"] or not ok(c, t):
            continue
        pos = F.pos_at(cl, t)
        if before:
            others = [n for n, p in zip(F.NAMES, pos) if p is not None and n != nm]
            bsrc, osrc = nm, (others[0] if others and F.dist(pos[F.NAMES.index(nm)], pos[F.NAMES.index(others[0])]) > F.TOL else None)
        else:
            others = [n for n, p in zip(F.NAMES, pos) if p is not None]
            if not others:
                continue
            bsrc, osrc = others[int(h(a.salt, c, t, "o") * len(others))], None
        cands["loss"].append((1.0 + h(a.salt, c, t, "prio"), c, t, meta(c, t, {"stratum": "loss", "loss_tracker": nm, "loss_side": "before" if before else "after", "t_last": tl}), bsrc, osrc))

    # ---- single-source runs
    for c, cl in fused["clips"].items():
        for tn in F.NAMES:
            flags = [tab[c][t] is not None and tab[c][t][0] == "only_one" and tab[c][t][1] == tn for t in range(cl["n"])]
            for lo, hi in F.runs(flags, gap=1):
                ts = [t for t in range(lo, hi + 1) if flags[t] and ok(c, t)]
                if len(ts) < 3:
                    continue
                t = ts[len(ts) // 2]
                cands["single"].append((h(a.salt, c, t, "prio"), c, t, meta(c, t, {"stratum": "single", "segment": f"{c}:{lo}-{hi}", "segment_frames": len(ts)}), tn, None))

    # ---- agreed frames: all three, and pairs
    for c, cl in fused["clips"].items():
        for t in range(cl["n"]):
            if not ok(c, t):
                continue
            kind, det = tab[c][t]
            if kind == "all3_agree":
                names = list(F.NAMES)
                bsrc = names[int(h(a.salt, c, t, "o") * 3)]
                cands["all3_agree"].append((h(a.salt, c, t, "prio"), c, t, meta(c, t, {"stratum": "all3_agree"}), bsrc, None))
            elif kind == "pair_agree":
                pair = tuple(n for n in F.NAMES if n != det)
                bsrc = pair[int(h(a.salt, c, t, "o") * 2)]
                cands[f"pair_agree:{'+'.join(pair)}"].append((h(a.salt, c, t, "prio"), c, t, meta(c, t, {"stratum": "pair_agree", "pair": "+".join(pair)}), bsrc, None))

    # ---- select (round-robin across clips, best priority first within a clip; min 2 frames apart in a clip per stratum)
    def select(pool, n, prefer=True, cost=lambda it: 1):
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
                    if (it[1], it[2]) in used or any(abs(it[2] - t2) < 12 for (c2, t2) in used if c2 == c):
                        continue
                    used.add((it[1], it[2]))
                    out.append(it)
                    spent += cost(it)
                    break
                if spent >= n:
                    break
        return out

    chosen = []
    taken = set()
    for stratum, quota in (("text_only", QUOTA["text_only"]), ("loss", QUOTA["loss"]), ("single", QUOTA["single"])):
        pool = [it for it in cands[stratum] if (it[1], it[2]) not in taken]
        sel = select(pool, quota, prefer=False)
        chosen += sel
        taken |= {(it[1], it[2]) for it in sel}
    sel = select([it for it in cands["all3_agree"] if (it[1], it[2]) not in taken], QUOTA["all3_agree"], prefer=False)
    chosen += sel
    taken |= {(it[1], it[2]) for it in sel}
    # pair agreement (two trackers agree, the third has no ball): almost all of it is "SAM2.1 + SAM3 agree, A++/B1 silent" (12% of frames), so that pair gets the quota
    for pn, q in (("t1+t2", 6), ("t1+t3", 6), ("t2+t3", QUOTA["pair_agree"])):
        sel = select([it for it in cands[f"pair_agree:{pn}"] if (it[1], it[2]) not in taken], q, prefer=False)
        chosen += sel
        taken |= {(it[1], it[2]) for it in sel}
    # disagreement segments fill the rest of the card budget; a FAR pair (> VIEW_PX: the magenta box is outside the phone's default crop) costs two single-box cards
    budget = a.cards - len(chosen)
    dsel = select([it for it in cands["disagree"] if (it[1], it[2]) not in taken], budget, cost=lambda it: 2 if it[3]["pair_far"] else 1)
    chosen += dsel
    ncards = lambda L: sum(2 if it[3].get("pair_far") else 1 for it in L)
    if ncards(chosen) < a.cards:     # disagreements are rare (4% of frames): top up with agreed frames (all-three sample to 44, then the SAM-pair-with-A++-silent class)
        have = lambda: taken | {(x[1], x[2]) for x in chosen}
        n3 = sum(1 for x in chosen if x[3]["stratum"] == "all3_agree")
        chosen += select([it for it in cands["all3_agree"] if (it[1], it[2]) not in have()], max(0, 44 - n3), prefer=False)
        chosen += select([it for it in cands["pair_agree:t2+t3"] if (it[1], it[2]) not in have()], a.cards - ncards(chosen), prefer=False)

    items = []
    for pr, c, t, m, bsrc, osrc in chosen:
        pos = F.pos_at(fused["clips"][c], t)
        twins = [(bsrc, None), (osrc, None)] if m.get("pair_far") else [(bsrc, osrc)]
        for k_, (b_, o_) in enumerate(twins):
            it = {"clip": c, "frame_index": t, "bbox": box(pos[F.NAMES.index(b_)])}
            if o_ is not None:
                it["other_bbox"] = box(pos[F.NAMES.index(o_)])
            it["label"] = "in_play_ball"
            it.update({"src_bbox": b_, "src_other": o_, **m, "far_twin": bool(m.get("pair_far"))})
            items.append(it)
    items.sort(key=lambda it: h(a.salt, it["clip"], it["frame_index"], it["stratum"], it["src_bbox"], "order"))
    for i, it in enumerate(items):
        it["v2_index"] = i
        it["v2_id"] = hashlib.sha1(f"{a.salt}:{it['clip']}:{it['frame_index']}:{i}".encode()).hexdigest()[:10]
    # collision check with v1 verdict keys (a judged key is (source, clip, frame_index, box_index))
    v1_index = {(it["clip"], it["frame_index"], i) for i, it in enumerate(v1)}
    clash = [it for it in items if (it["clip"], it["frame_index"], it["v2_index"]) in v1_index]
    assert not clash, clash[:2]
    population = {"all3_agree": dict(Counter(it[1] for it in cands["all3_agree"]))}
    for pn in ("t1+t2", "t1+t3", "t2+t3"):
        population[f"pair_agree:{pn}"] = dict(Counter(it[1] for it in cands[f"pair_agree:{pn}"]))
    for it in items:
        it["clip_run_id"] = it["clip"]       # id used by this run's artifacts (arsenal_norwich_segNNN)
        it["clip"] = NEWN(it["clip"])
    population = {k: {NEWN(c): n for c, n in v.items()} for k, v in population.items()}
    doc = {"population": population, "description": "ball_check_list_v2_disagree: three-tracker disagreement batch (blind); hidden keys: src_bbox/src_other (t1=A++/B1, t2=SAM2.1, t3=SAM3), stratum, positions ...",
           "salt": a.salt, "tolerance_px": F.TOL, "box_px": BOX, "items": items}
    a.out.write_text(json.dumps(doc))
    print("cards", len(items), "clips", len({it["clip"] for it in items}))
    print("by stratum", dict(Counter(it["stratum"] for it in items)))
    print("far pairs (> %d px) among two-box cards:" % VIEW_PX, sum(1 for it in items if it.get("pair_far")), "of", sum(1 for it in items if "other_bbox" in it))
    print("per clip max", max(Counter(it["clip"] for it in items).values()))
    print("candidates:", {k: len(v) for k, v in cands.items()})


if __name__ == "__main__":
    main()
