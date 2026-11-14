"""Whole-clip best-path ball tracking that honours human verdicts.

``ConstrainedViterbiTracker`` is the :class:`ViterbiBallTracker` model (same
emissions, transition costs, enter/leave costs; with no constraints it returns the
same path) plus three kinds of constraint, all in pixels of the native frame:

* **anchor** ``(frame, x, y, radius)``: the ball is at that spot in that frame (a
  "ball" verdict, or a dragged position). Only candidates within ``radius`` of it
  survive in that frame, and the path is forced through one of them. If no
  candidate is that close a synthetic full-confidence candidate is injected at
  ``(x, y)``.
* **forbid** ``(frame, x, y, tol)``: a "not a ball" verdict. Candidates within
  ``tol`` px of the spot in that frame are removed.
* **forbid everywhere** ``(x, y, tol)``: the same spot, in every frame (a parked
  spare ball or a fixed overlay seen at one image position).

Besides the path, ``solve`` returns a per-frame *margin*: how much worse (in model
score) the best whole-clip path is if it must use something else at that frame.
It comes from a forward and a backward pass (max-marginals), so it costs two
passes, not one re-run per frame. Small margin = the path is unsure at that frame.
For a frame where the path sits on a candidate the "use something else" options are
the other candidates there, or dropping the ball (approximated by the candidate's own
emission, minus a gap cost, because exact dropping would need a third pass).

Frames are identified by the ``frame_index`` given to ``update``; chaining two
consecutive segments is just feeding the second one with indices continuing the
first (``offset + f``).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from footy_track.ball_trackers.detection_tracker import (
    DETECTED,
    INTERPOLATED,
    BallCandidate,
    BallPoint,
)
from footy_track.ball_trackers.viterbi import ViterbiBallTracker

_ANCHOR_BONUS = 1e5  # makes "pass through an anchor" a hard constraint
_BALL_PX = 11.0  # size of a synthetic anchor box (median ball box at 1280)
# Margins for frames the path says nothing useful about.
INTERP_NO_CAND_MARGIN = 2.0
NONE_NO_CAND_MARGIN = 4.0


@dataclass
class Solution:
    """Result of one constrained solve."""

    points: list[BallPoint]
    margins: dict[int, float]  # frame_index -> margin (inf for anchored frames)
    best_score: float = 0.0
    anchored: set[int] = field(default_factory=set)


class ConstrainedViterbiTracker(ViterbiBallTracker):
    """Viterbi ball tracker with anchors, forbidden spots and per-frame margins."""

    def __init__(self, frame_size: tuple[int, int] = (1920, 1080), **params) -> None:
        super().__init__(frame_size, **params)
        self.rival_px = (
            25.0  # candidates closer than this to the chosen one are duplicates
        )
        self._anchors: dict[int, tuple[float, float, float]] = {}
        self._forbid_at: dict[int, list[tuple[float, float, float]]] = {}
        self._forbid_all: list[tuple[float, float, float]] = []

    # ------------------------------------------------------------ constraints

    def anchor(self, frame: int, x: float, y: float, radius: float = 25.0) -> None:
        """The ball is at pixel ``(x, y)`` in ``frame`` (replaces an earlier anchor)."""
        self._anchors[frame] = (float(x), float(y), float(radius))

    def forbid(self, frame: int, x: float, y: float, tol: float = 3.0) -> None:
        """No ball within ``tol`` px of ``(x, y)`` in ``frame``."""
        self._forbid_at.setdefault(frame, []).append((float(x), float(y), float(tol)))

    def forbid_everywhere(self, x: float, y: float, tol: float = 5.0) -> None:
        """No ball within ``tol`` px of ``(x, y)`` in any frame."""
        self._forbid_all.append((float(x), float(y), float(tol)))

    @property
    def n_constraints(self) -> int:
        return (
            len(self._anchors)
            + sum(len(v) for v in self._forbid_at.values())
            + len(self._forbid_all)
        )

    # ------------------------------------------------------------ preparation

    def _px(self, c: BallCandidate) -> tuple[float, float]:
        return c.center[0] * self._w, c.center[1] * self._h

    def _near(self, c: BallCandidate, spot: tuple[float, float, float]) -> bool:
        x, y = self._px(c)
        return math.hypot(x - spot[0], y - spot[1]) <= spot[2]

    def _prepare(self) -> list[list[BallCandidate]]:
        p = self._p
        kept: list[list[BallCandidate]] = []
        for f, cands in self._frames:
            good = [c for c in cands if c.confidence >= p["min_conf"]]
            if f in self._anchors:
                ax, ay, ar = self._anchors[f]
                near = [c for c in good if self._near(c, (ax, ay, ar))]
                if not near:
                    near = [
                        BallCandidate(
                            ax / self._w - _BALL_PX / 2 / self._w,
                            ay / self._h - _BALL_PX / 2 / self._h,
                            _BALL_PX / self._w,
                            _BALL_PX / self._h,
                            1.0,
                        )
                    ]
                kept.append(near)
                continue
            spots = self._forbid_all + self._forbid_at.get(f, [])
            if spots:
                good = [c for c in good if not any(self._near(c, s) for s in spots)]
            good.sort(key=lambda c: -c.confidence)
            kept.append(good[: p["top_k"]])
        return kept

    # ------------------------------------------------------------------ solve

    def points(self) -> list[BallPoint]:
        return self.solve().points

    def solve(self) -> Solution:  # noqa: PLR0912, PLR0915
        n = len(self._frames)
        if n == 0:
            return Solution([], {})
        p = self._p
        enter, leave = p["enter_cost"], p["leave_cost"]
        cands = self._prepare()
        idx = [f for f, _ in self._frames]
        anchored = {t for t, f in enumerate(idx) if f in self._anchors}
        emis = [
            np.array([self._emission(c) for c in cs])
            + (_ANCHOR_BONUS if t in anchored else 0.0)
            for t, cs in enumerate(cands)
        ]
        pos = [np.array([self._px(c) for c in cs]).reshape(-1, 2) for cs in cands]

        links: dict[tuple[int, int], np.ndarray] = {}

        def cost(s: int, t: int) -> np.ndarray:
            key = (s, t)
            if key not in links:
                g = idx[t] - idx[s]
                d = np.linalg.norm(pos[s][:, None, :] - pos[t][None, :, :], axis=2)
                allowed = p["base_px"] + p["speed_px"] * g
                c = np.minimum(p["jump_cost"], p["move_weight"] * (d / allowed) ** 2)
                links[key] = c + p["gap_cost"] * (g - 1)
            return links[key]

        # ---- forward: fwd[t][i] = best score of a path ending at node (t, i)
        fwd: list[np.ndarray] = [np.empty(0)] * n
        par_t = [np.empty(0, dtype=int)] * n
        par_i = [np.empty(0, dtype=int)] * n
        linked = [np.empty(0, dtype=bool)] * n
        run_best, run_node = 0.0, None
        for t in range(n):
            k = len(cands[t])
            if k == 0:
                continue
            val = emis[t] + max(run_best, 0.0) - enter
            pt = np.full(k, -1, dtype=int)
            pi = np.full(k, -1, dtype=int)
            lk = np.zeros(k, dtype=bool)
            if run_node is not None and run_best > 0.0:
                pt[:], pi[:] = run_node
            for s in range(t - 1, -1, -1):
                if idx[t] - idx[s] > p["max_gap"]:
                    break
                if len(cands[s]) == 0:
                    continue
                cv = fwd[s][:, None] - cost(s, t)
                j = np.argmax(cv, axis=0)
                best = cv[j, np.arange(k)] + emis[t]
                better = best > val
                val = np.where(better, best, val)
                pt[better], pi[better], lk[better] = s, j[better], True
            fwd[t], par_t[t], par_i[t], linked[t] = val, pt, pi, lk
            ib = int(np.argmax(val))
            if val[ib] - leave > run_best:
                run_best, run_node = float(val[ib] - leave), (t, ib)

        # ---- backward: bwd[t][i] = best score of everything after node (t, i)
        bwd: list[np.ndarray] = [np.empty(0)] * n
        suffix = -np.inf  # best (emis + bwd) over nodes in later frames
        for t in range(n - 1, -1, -1):
            k = len(cands[t])
            if k == 0:
                continue
            val = np.full(k, -leave)
            if np.isfinite(suffix):
                val = np.maximum(val, suffix - leave - enter)
            for u in range(t + 1, n):
                if idx[u] - idx[t] > p["max_gap"]:
                    break
                if len(cands[u]) == 0:
                    continue
                cv = -cost(t, u) + (emis[u] + bwd[u])[None, :]
                val = np.maximum(val, cv.max(axis=1))
            bwd[t] = val
            suffix = max(suffix, float((emis[t] + val).max()))

        # ---- path
        points = [BallPoint(f) for f in idx]
        chosen: dict[int, int] = {}
        if run_node is not None and run_best > 0.0:
            chain = self._decode(run_node, par_t, par_i, linked)
            self._assemble_chain(points, cands, idx, chain)
            chosen = {t: i for t, i, _ in chain}
        best = max(run_best, 0.0)

        # ---- margins
        margins: dict[int, float] = {}
        for t, f in enumerate(idx):
            if t in anchored:
                margins[f] = math.inf
                continue
            k = len(cands[t])
            if k == 0:
                src = points[t].source
                margins[f] = (
                    INTERP_NO_CAND_MARGIN
                    if src == INTERPOLATED
                    else NONE_NO_CAND_MARGIN
                )
                continue
            total = fwd[t] + bwd[t]
            if t in chosen:
                i = chosen[t]
                # a rival is a different object: not a duplicate box on the same spot
                far = np.linalg.norm(pos[t] - pos[t][i], axis=1) > self.rival_px
                alt = best - float(total[far].max()) if far.any() else math.inf
                skip = max(float(emis[t][i]) - p["gap_cost"], 0.0)
                margins[f] = min(alt, skip)
            else:
                margins[f] = max(best - float(total.max()), 0.0)
        return Solution(points, margins, best, {idx[t] for t in anchored})

    @staticmethod
    def _decode(run_node, par_t, par_i, linked) -> list[tuple[int, int, bool]]:
        chain: list[tuple[int, int, bool]] = []
        node = run_node
        while node is not None:
            t, i = node
            s, j, lk = int(par_t[t][i]), int(par_i[t][i]), bool(linked[t][i])
            chain.append((t, i, lk))
            node = (s, j) if s >= 0 else None
        chain.reverse()
        return chain

    def _assemble_chain(self, points, cands, idx, chain) -> None:
        track_id = 0
        prev: tuple[int, BallCandidate] | None = None
        for t, i, lk in chain:
            c = cands[t][i]
            if not lk or prev is None:
                track_id += 1
            elif t - prev[0] > 1:
                self._fill(points, cands, idx, prev, (t, c), track_id)
            points[t] = BallPoint(
                idx[t], DETECTED, track_id, c.x, c.y, c.w, c.h, c.confidence
            )
            prev = (t, c)


__all__ = ["ConstrainedViterbiTracker", "Solution"]
