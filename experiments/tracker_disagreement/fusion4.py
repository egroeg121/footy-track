"""Agreement classes for four trackers (t1 A++/B1, t2 SAM2.1, t3 SAM 3, t4 the labeller's autodetect + VitTrack).  Same positions / tolerance as fusion.py (8 px)."""
from __future__ import annotations

import math
from collections import Counter

import fusion as F

NAMES = ("t1", "t2", "t3", "t4")
LABEL = {"t1": "A++/B1", "t2": "SAM2.1", "t3": "SAM3", "t4": "VitTrack"}
TOL = F.TOL


def pos_at(clip: dict, t: int):
    out = []
    for k in NAMES:
        tr = clip.get(k)
        out.append((tr["x"][t], tr["y"][t]) if tr is not None and tr["x"][t] is not None else None)
    return out


def dist(a, b) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def clusters(pos, tol: float = TOL):
    """Complete-linkage clusters of the present trackers: list of lists of tracker indices, largest first."""
    pres = [i for i, p in enumerate(pos) if p is not None]
    cl: list[list[int]] = []
    for i in pres:
        for c in cl:
            if all(dist(pos[i], pos[j]) <= tol for j in c):
                c.append(i)
                break
        else:
            cl.append([i])
    return sorted(cl, key=lambda c: (-len(c), c))


def classify(pos, tol: float = TOL):
    """-> (kind, info).  kinds: none | only_one(t) | agree_all(n_present, absent) | odd_one(t, n_present) | split | all_differ.
    odd_one = exactly one present tracker outside the largest cluster (which has >= 2 members); split = two clusters of >= 2; all_differ = every present tracker alone (>= 2 present)."""
    cl = clusters(pos, tol)
    pres = sum(len(c) for c in cl)
    absent = [NAMES[i] for i, p in enumerate(pos) if p is None]
    if pres == 0:
        return "none", None
    if pres == 1:
        return "only_one", NAMES[cl[0][0]]
    if len(cl) == 1:
        return "agree_all", {"n_present": pres, "absent": absent}
    if len(cl[0]) == 1:
        return "all_differ", {"n_present": pres, "absent": absent}
    if len(cl[1]) >= 2:
        return "split", {"n_present": pres, "clusters": [[NAMES[i] for i in c] for c in cl]}
    out = [NAMES[i] for c in cl[1:] for i in c]
    return "odd_one", {"odd": out[0] if len(out) == 1 else out, "n_present": pres, "agree": [NAMES[i] for i in cl[0]], "absent": absent}


def magnitude(pos) -> float:
    ds = [dist(pos[i], pos[j]) for i in range(len(pos)) for j in range(i + 1, len(pos)) if pos[i] is not None and pos[j] is not None]
    return max(ds) if ds else 0.0


def table(fused: dict):
    tab = {}
    for c, cl in fused["clips"].items():
        tab[c] = [None if F.excluded(fused, c, t) else classify(pos_at(cl, t)) for t in range(cl["n"])]
    return tab


def vit_role(kind, info) -> str:
    """What VitTrack (t4) is doing on this frame."""
    if kind == "none" or (kind == "only_one" and info != "t4") or (kind == "agree_all" and "t4" in info["absent"]) or (kind in ("odd_one", "all_differ") and "t4" in info["absent"]):
        return "vit_silent"
    if kind == "only_one":
        return "vit_alone"
    if kind == "agree_all":
        return "vit_agrees_all" if info["n_present"] >= 3 else "vit_agrees_pair"      # n_present 2: VitTrack + one other
    if kind == "odd_one":
        odd = info["odd"] if isinstance(info["odd"], list) else [info["odd"]]
        return "vit_odd_one_out" if "t4" in odd else "vit_in_majority_other_odd"
    if kind == "split":
        return "vit_split"
    return "vit_all_differ"
