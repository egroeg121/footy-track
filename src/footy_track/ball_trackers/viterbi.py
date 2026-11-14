"""Whole-clip best-path ball tracking (offline dynamic programming).

Instead of committing to a track the moment a confident candidate appears, pick
the single best *path* through all candidates of the clip. A confident false
positive that is not supported by a consistent trajectory loses to the real ball
that is, even when the real ball only shows up later.

Model. Each frame offers up to ``top_k`` candidates (confidence >=
``min_conf``). A path chooses at most one candidate per frame, or none, and its
score is the sum of

* an emission per chosen candidate: ``emission_scale`` times the log-odds that the
  candidate is the ball, read off the Ball Check precision bands (0.04 at
  0.1-0.2, 0.42 at 0.5-0.6, 0.79 at 0.6-0.7, 0.96 at 0.7-0.8, ~1 above)
  interpolated in log-odds, plus ``emission_bias``. A candidate under
  ``top_y`` (the stands, normalised y) loses ``top_penalty`` more.
* minus a transition cost between two consecutive chosen candidates ``g`` frames
  apart and ``d`` px apart: ``move_weight * (d / allowed)^2`` where ``allowed =
  base_px + speed_px * g`` (so a longer gap allows a larger jump), capped at
  ``jump_cost`` (a camera cut is allowed but never free), plus ``gap_cost`` per
  skipped frame.
* minus ``enter_cost`` to start a track and ``leave_cost`` to end one. Candidates
  more than ``max_gap`` frames apart are never linked: a new track has to start.

Frames not on the path are ``none``. Frames skipped *inside* a track (at most
``max_gap - 1`` in a row, never more) are filled by linear interpolation and
marked ``interpolated``; nothing is ever interpolated across a longer gap.

Cost is O(frames * max_gap * top_k^2) with numpy, a fraction of a second per
clip. No pitch mask is used: ``top_penalty`` is a plain image-position prior and
is off (0) by default.
"""

from __future__ import annotations

import math

import numpy as np

from footy_track.ball_trackers.detection_tracker import (
    DETECTED,
    INTERPOLATED,
    BallCandidate,
    BallPoint,
)

# (confidence, precision) mid-points of the Ball Check bands.
_BANDS = ((0.15, 0.04), (0.55, 0.42), (0.65, 0.79), (0.75, 0.96), (0.85, 0.995))
_BAND_X = np.array([c for c, _ in _BANDS])
_BAND_LOGIT = np.array([math.log(p / (1 - p)) for _, p in _BANDS])


def confidence_logodds(conf: np.ndarray | float) -> np.ndarray | float:
    """Log-odds that a candidate of this confidence is the ball (band interpolation)."""
    return np.interp(conf, _BAND_X, _BAND_LOGIT)


class ViterbiBallTracker:
    """Offline drop-in for ``DetectionBallTracker``: ``update`` per frame, ``points`` at the end."""

    def __init__(
        self,
        frame_size: tuple[int, int] = (1920, 1080),
        *,
        min_conf: float = 0.1,
        top_k: int = 8,
        emission_scale: float = 1.0,
        emission_bias: float = 1.0,
        base_px: float = 30.0,
        speed_px: float = 40.0,
        move_weight: float = 3.0,
        jump_cost: float = 8.0,
        gap_cost: float = 0.3,
        enter_cost: float = 4.0,
        leave_cost: float = 0.5,
        max_gap: int = 12,
        top_y: float = 0.12,
        top_penalty: float = 0.0,
    ) -> None:
        self._w, self._h = frame_size
        self._p = {
            "min_conf": min_conf,
            "top_k": int(top_k),
            "emission_scale": emission_scale,
            "emission_bias": emission_bias,
            "base_px": base_px,
            "speed_px": speed_px,
            "move_weight": move_weight,
            "jump_cost": jump_cost,
            "gap_cost": gap_cost,
            "enter_cost": enter_cost,
            "leave_cost": leave_cost,
            "max_gap": int(max_gap),
            "top_y": top_y,
            "top_penalty": top_penalty,
        }
        self._frames: list[tuple[int, list[BallCandidate]]] = []

    def update(self, frame_index: int, candidates: list[BallCandidate]) -> BallPoint:
        self._frames.append((frame_index, list(candidates)))
        return BallPoint(frame_index)  # the real answer needs the future: see points()

    # ------------------------------------------------------------------

    def _prepare(self) -> list[list[BallCandidate]]:
        p = self._p
        kept = []
        for _, cands in self._frames:
            good = [c for c in cands if c.confidence >= p["min_conf"]]
            good.sort(key=lambda c: -c.confidence)
            kept.append(good[: p["top_k"]])
        return kept

    def _emission(self, c: BallCandidate) -> float:
        p = self._p
        e = (
            p["emission_scale"] * float(confidence_logodds(c.confidence))
            + p["emission_bias"]
        )
        if c.center[1] < p["top_y"]:
            e -= p["top_penalty"]
        return e

    def points(self) -> list[BallPoint]:
        n = len(self._frames)
        if n == 0:
            return []
        p = self._p
        cands = self._prepare()
        idx = [f for f, _ in self._frames]
        emis = [np.array([self._emission(c) for c in cs]) for cs in cands]
        pos = [
            np.array(
                [[c.center[0] * self._w, c.center[1] * self._h] for c in cs]
            ).reshape(-1, 2)
            for cs in cands
        ]
        score: list[np.ndarray] = [np.empty(0)] * n
        back: list[np.ndarray] = [
            np.empty((0, 2), dtype=int)
        ] * n  # (t_prev, i_prev); t_prev -1 = new track
        # best_before[t]: best value of "finished everything before frame t" (0 = nothing yet)
        best_end = 0.0
        best_end_node: tuple[int, int] | None = None
        run_best = 0.0  # best score over nodes seen so far, minus leave_cost
        run_node: tuple[int, int] | None = None
        for t in range(n):
            k = len(cands[t])
            if k == 0:
                continue
            # start a new track here (after ending the best earlier path)
            start_val = max(run_best, 0.0) - p["enter_cost"]
            start_from = run_node if run_best > 0.0 else None
            val = emis[t] + start_val
            bk = np.tile(np.array([-1, -1]), (k, 1))
            if start_from is not None:
                bk[:] = (
                    -2 - start_from[0],
                    start_from[1],
                )  # encode "new track after end of start_from"
            for s in range(t - 1, -1, -1):
                g = idx[t] - idx[s]
                if g > p["max_gap"]:
                    break
                if len(cands[s]) == 0:
                    continue
                d = np.linalg.norm(pos[s][:, None, :] - pos[t][None, :, :], axis=2)
                allowed = p["base_px"] + p["speed_px"] * g
                cost = np.minimum(p["jump_cost"], p["move_weight"] * (d / allowed) ** 2)
                cost = cost + p["gap_cost"] * (g - 1)
                cand_val = score[s][:, None] - cost  # (k_s, k_t)
                j = np.argmax(cand_val, axis=0)
                best = cand_val[j, np.arange(k)] + emis[t]
                better = best > val
                val = np.where(better, best, val)
                bk[better, 0] = s
                bk[better, 1] = j[better]
            score[t], back[t] = val, bk
            i_best = int(np.argmax(val))
            if val[i_best] - p["leave_cost"] > run_best:
                run_best, run_node = float(val[i_best] - p["leave_cost"]), (t, i_best)
        best_end, best_end_node = run_best, run_node
        if best_end_node is None or best_end <= 0.0:
            return [BallPoint(f) for f in idx]
        return self._assemble(cands, idx, back, best_end_node)

    def _assemble(
        self,
        cands: list[list[BallCandidate]],
        idx: list[int],
        back: list[np.ndarray],
        end_node: tuple[int, int],
    ) -> list[BallPoint]:
        # walk back: chain of (t, i) with a flag marking track starts
        chain: list[tuple[int, int, bool]] = []  # (t, i, linked_to_previous)
        node: tuple[int, int] | None = end_node
        while node is not None:
            t, i = node
            s, j = back[t][i]
            chain.append((t, i, bool(s >= 0)))
            if s >= 0:
                node = (int(s), int(j))
            elif s == -1:
                node = None
            else:  # a new track that starts after the end of an earlier path
                node = (-2 - int(s), int(j))
        chain.reverse()
        points = [BallPoint(f) for f in idx]
        track_id = 0
        prev: tuple[int, BallCandidate] | None = None
        for t, i, linked in chain:
            c = cands[t][i]
            if not linked or prev is None:
                track_id += 1
            elif t - prev[0] > 1:
                self._fill(points, cands, idx, prev, (t, c), track_id)
            points[t] = BallPoint(
                idx[t], DETECTED, track_id, c.x, c.y, c.w, c.h, c.confidence
            )
            prev = (t, c)
        return points

    @staticmethod
    def _fill(
        points,
        cands,
        idx,
        a: tuple[int, BallCandidate],
        b: tuple[int, BallCandidate],
        tid: int,
    ) -> None:
        (ta, ca), (tb, cb) = a, b
        span = idx[tb] - idx[ta]
        for t in range(ta + 1, tb):
            r = (idx[t] - idx[ta]) / span
            x, y, w, h = (
                u + (v - u) * r
                for u, v in zip(
                    (ca.x, ca.y, ca.w, ca.h), (cb.x, cb.y, cb.w, cb.h), strict=True
                )
            )
            points[t] = BallPoint(idx[t], INTERPOLATED, tid, x, y, w, h)


__all__ = ["ViterbiBallTracker", "confidence_logodds"]
