"""Ranking of suggestion candidates given the human's previous boxes (LAB-1479). Python mirror of ``rankCandidates`` in
``labeller/web/index.html`` (the page does it in the browser; this copy feeds the report and the tests, keep both in sync).

    score = log(max(conf, 0.05)) - d^2 / (2 sigma^2) (+ 0.5 if the candidate is on the tracklet of the candidate under the last box)

``d`` = px distance (scaled to 1080p) from the predicted position: constant velocity from the last two human boxes, else the
last box, else no motion term (order by conf). ``sigma = SIGMA0 + K * |frame - last_label_frame|`` (K_VEL with a velocity
prediction, K_STATIC without). Tune the constants HERE and in ``SUG_RANK`` of the page."""

from __future__ import annotations

import math

SIGMA0 = 8.0
K_STATIC = 6.0
K_VEL = 4.0
CONF_FLOOR = 0.05
TID_BONUS = 0.5


def rank_candidates(cands: list, anchors: list[dict], frame: int, size: tuple[int, int] = (1920, 1080), anchor_tid: int = 0) -> list[dict]:
    """``cands`` rows ``[x, y, w, h, conf, tid]`` (normalised), ``anchors`` human boxes ``{"f", "cx", "cy"}`` (normalised
    centres) ordered by frame, those BEFORE ``frame`` last. Returns ``[{"row", "score", "tid_bonus"}]`` best first."""
    prev = [a for a in anchors if a["f"] < frame]
    use = prev[-2:] if prev else [min((a for a in anchors if a["f"] > frame), key=lambda a: a["f"], default=None)]
    use = [a for a in use if a]
    scale = 1080.0 / size[1]
    pred, sigma = None, 0.0
    if use:
        last = use[-1]
        k = K_STATIC
        px, py = last["cx"], last["cy"]
        if len(use) == 2 and use[1]["f"] != use[0]["f"] and use[1]["f"] < frame:
            v = (use[1]["f"] - use[0]["f"])
            px += (use[1]["cx"] - use[0]["cx"]) / v * (frame - last["f"])
            py += (use[1]["cy"] - use[0]["cy"]) / v * (frame - last["f"])
            k = K_VEL
        pred, sigma = (px, py), SIGMA0 + k * abs(frame - last["f"])
    out = []
    for r in cands:
        s = math.log(max(float(r[4]), CONF_FLOOR))
        bonus = 0.0
        if pred is not None:
            d = math.hypot((r[0] + r[2] / 2 - pred[0]) * size[0] * scale, (r[1] + r[3] / 2 - pred[1]) * size[1] * scale)
            s -= d * d / (2 * sigma * sigma)
            if anchor_tid and len(r) > 5 and r[5] == anchor_tid:
                bonus = TID_BONUS
                s += bonus
        out.append({"row": r, "score": s, "tid_bonus": bonus})
    return sorted(out, key=lambda o: -o["score"])
