"""Anchor selection shared by machine_prep.py (spec) and machine_score.py (fusion). Stdlib only."""
from __future__ import annotations

# (anchor confidence threshold, minimum anchor spacing in frames)
CONFIGS = [(0.6, 10), (0.7, 10), (0.8, 10), (0.7, 5), (0.7, 25)]
REACH = 50  # max frames a track is propagated from an anchor
MAIN_CLIPS = ("arsenal_mancity_seg010", "arsenal_mancity_seg031", "arsenal_example")  # full propagation (coverage measured)
AGREE_PX = 5.0       # forward/backward agreement gate
SINGLE_PASS_MAX = 12  # a lone pass is accepted only this close to its anchor


def thin(nodes: list[dict], pred_frames: set[int], thr: float, spacing: int) -> list[dict]:
    """Anchors = B1 solver nodes with conf >= thr, thinned to >= spacing frames apart.

    A node that borders a gap (the frame before or after it has no B1 prediction) is always kept:
    those are the prompts that matter for filling."""
    cand = [n for n in nodes if n["conf"] >= thr]
    out, last = [], -10**9
    for n in cand:
        t = n["t"]
        border = (t - 1) not in pred_frames or (t + 1) not in pred_frames
        if border or t - last >= spacing:
            out.append(n)
            last = t
    return out


def tracks_for(anchors: list[dict], n_frames: int) -> list[dict]:
    """Forward/backward reach of each anchor: up to the neighbouring anchor or REACH frames."""
    out = []
    ts = [a["t"] for a in anchors]
    for i, a in enumerate(anchors):
        nxt = ts[i + 1] if i + 1 < len(ts) else None
        prv = ts[i - 1] if i > 0 else None
        f_reach = min(REACH, (nxt - a["t"]) if nxt is not None else REACH, n_frames - 1 - a["t"])
        b_reach = min(REACH, (a["t"] - prv) if prv is not None else REACH, a["t"])
        out.append({"t": a["t"], "box": a["box"], "fwd": max(0, f_reach), "bwd": max(0, b_reach)})
    return out
