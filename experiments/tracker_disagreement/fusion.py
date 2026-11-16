"""Per-frame agreement classes, disagreement segments, loss events of the three trackers (pure python / numpy; used by build_batch.py and the report).

Positions are 1080p pixels, ``fused.json.gz`` from fuse.py.  ``TOL`` = 8 px: two trackers agree when their positions are within it.
"""
from __future__ import annotations

import gzip
import json
import math
from collections import Counter
from pathlib import Path

NAMES = ("t1", "t2", "t3")          # A++ + B1, SAM2.1-small crop512, SAM 3 box crop512
LABEL = {"t1": "A++/B1", "t2": "SAM2.1", "t3": "SAM3"}
TOL = 8.0
W, H = 1920, 1080

KINDS = ("none", "only_one", "pair_agree", "pair_differ", "all3_agree", "two_agree", "all_differ")
DISAGREE = ("pair_differ", "two_agree", "all_differ")


def load(path: Path) -> dict:
    with gzip.open(path, "rt") as fh:
        return json.load(fh)


def pos_at(clip: dict, t: int):
    """[p1, p2, p3] each (x, y) or None."""
    out = []
    for k in NAMES:
        tr = clip[k]
        out.append((tr["x"][t], tr["y"][t]) if tr is not None and tr["x"][t] is not None else None)
    return out


def dist(a, b) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def classify(pos, tol: float = TOL):
    """-> (kind, detail).  detail: only_one -> tracker; pair_agree -> absent tracker; pair_differ -> absent tracker; two_agree -> odd tracker; others None."""
    pres = [i for i, p in enumerate(pos) if p is not None]
    if not pres:
        return "none", None
    if len(pres) == 1:
        return "only_one", NAMES[pres[0]]
    if len(pres) == 2:
        absent = NAMES[[i for i in range(3) if i not in pres][0]]
        return ("pair_agree" if dist(pos[pres[0]], pos[pres[1]]) <= tol else "pair_differ"), absent
    d01, d02, d12 = dist(pos[0], pos[1]), dist(pos[0], pos[2]), dist(pos[1], pos[2])
    if max(d01, d02, d12) <= tol:
        return "all3_agree", None
    near = [(d01 <= tol, 2), (d02 <= tol, 1), (d12 <= tol, 0)]   # (pair agrees, index of the third = candidate odd one)
    ok = [odd for agree, odd in near if agree]
    if not ok:
        return "all_differ", None
    if len(ok) == 1:
        return "two_agree", NAMES[ok[0]]
    # chain (a~b, b~c, a!~c): odd one = the one with the largest summed distance
    s = [d01 + d02, d01 + d12, d02 + d12]
    return "two_agree", NAMES[max(range(3), key=lambda i: s[i])]


def magnitude(pos) -> float:
    ds = [dist(pos[i], pos[j]) for i in range(3) for j in range(i + 1, 3) if pos[i] is not None and pos[j] is not None]
    return max(ds) if ds else 0.0


def excluded(meta: dict, clip: str, t: int) -> bool:
    return any(lo <= t <= hi for lo, hi in meta["exclude"].get(clip, []))


def frame_table(fused: dict):
    """{clip: [(kind, detail)] per frame} (excluded frames -> None)."""
    tab = {}
    for c, cl in fused["clips"].items():
        tab[c] = [None if excluded(fused, c, t) else classify(pos_at(cl, t)) for t in range(cl["n"])]
    return tab


def breakdown(fused: dict, tab: dict) -> dict:
    cnt, odd, cov, n = Counter(), Counter(), Counter(), 0
    abs_pair = Counter()
    only = Counter()
    for c, rows in tab.items():
        for t, r in enumerate(rows):
            if r is None:
                continue
            n += 1
            k, d = r
            cnt[k] += 1
            if k == "two_agree":
                odd[d] += 1
            if k == "only_one":
                only[d] += 1
            if k in ("pair_agree", "pair_differ"):
                abs_pair[(k, d)] += 1
            for nm, p in zip(NAMES, pos_at(fused["clips"][c], t)):
                if p is not None:
                    cov[nm] += 1
    return {"frames": n, "kinds": dict(cnt), "odd_one_out": dict(odd), "only_one": dict(only), "pair_by_absent": {f"{k}:{d}": v for (k, d), v in abs_pair.items()},
            "coverage": {nm: cov[nm] for nm in NAMES}}


def runs(flags: list[bool], gap: int = 2) -> list[tuple[int, int]]:
    """Maximal runs of True; runs separated by <= ``gap`` False frames are merged."""
    out = []
    start = last = None
    for t, f in enumerate(flags):
        if f:
            if start is None:
                start = t
            elif t - last > gap + 1:
                out.append((start, last))
                start = t
            last = t
    if start is not None:
        out.append((start, last))
    return out


def disagreement_segments(fused: dict, tab: dict) -> list[dict]:
    segs = []
    for c, rows in tab.items():
        cl = fused["clips"][c]
        for lo, hi in runs([r is not None and r[0] in DISAGREE for r in rows]):
            ts = [t for t in range(lo, hi + 1) if rows[t] is not None and rows[t][0] in DISAGREE]
            kinds = Counter(rows[t][0] + ":" + str(rows[t][1]) for t in ts)
            segs.append({"clip": c, "lo": lo, "hi": hi, "frames": ts, "n_dis": len(ts), "kinds": dict(kinds), "main": kinds.most_common(1)[0][0]})
    return segs


def presence_events(fused: dict, tab: dict) -> list[dict]:
    """A tracker's presence ends (present at t_last, absent at t_last+1) while another tracker still has a ball at t_last+1."""
    ev = []
    for c, rows in tab.items():
        cl = fused["clips"][c]
        n = cl["n"]
        for ti, nm in enumerate(NAMES):
            for t in range(n - 2):
                if rows[t] is None or rows[t + 1] is None:
                    continue
                p0, p1 = pos_at(cl, t), pos_at(cl, t + 1)
                if p0[ti] is not None and p1[ti] is None and any(p1[j] is not None for j in range(3) if j != ti):
                    # require it to stay absent for >= 3 frames (not a one-frame flicker)
                    if all(pos_at(cl, min(t + k, n - 1))[ti] is None for k in (2, 3)):
                        ev.append({"clip": c, "tracker": nm, "t_last": t})
    return ev
