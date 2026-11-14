"""Canonical football pitch in metres.

Frame: origin at the centre spot, ``+x`` along the long axis, ``+y`` toward the
far touchline. ``L`` / ``R`` name the ``-x`` / ``+x`` halves, ``TOP`` / ``BOTTOM``
the ``+y`` / ``-y`` sides. Regulation dimensions of the penalty/goal areas, the
spot and the circle radii are fixed (IFAB); only length and width vary.

Keypoint ids are stable strings: calibration GT stores them.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

DEFAULT_LENGTH = 105.0
DEFAULT_WIDTH = 68.0

PEN_BOX_DEPTH = 16.5
PEN_BOX_WIDTH = 40.32
GOAL_AREA_DEPTH = 5.5
GOAL_AREA_WIDTH = 18.32
PEN_SPOT_DIST = 11.0
CIRCLE_RADIUS = 9.15
GOAL_WIDTH = 7.32

Polyline = np.ndarray  # (N, 2) float64 metres


def _arc(
    cx: float, cy: float, r: float, a0: float, a1: float, n: int = 48
) -> Polyline:
    t = np.linspace(a0, a1, n)
    return np.stack([cx + r * np.cos(t), cy + r * np.sin(t)], axis=1)


@dataclass(frozen=True)
class PitchModel:
    """A pitch of ``length`` x ``width`` metres."""

    length: float = DEFAULT_LENGTH
    width: float = DEFAULT_WIDTH

    def __post_init__(self) -> None:
        if not (self.length > 2 * PEN_BOX_DEPTH + 2 * CIRCLE_RADIUS):
            raise ValueError(f"pitch length too small: {self.length}")
        if not (self.width > PEN_BOX_WIDTH):
            raise ValueError(f"pitch width too small: {self.width}")

    @property
    def hl(self) -> float:
        return self.length / 2.0

    @property
    def hw(self) -> float:
        return self.width / 2.0

    # ------------------------------------------------------------------
    def segments(self) -> dict[str, np.ndarray]:
        """Straight regulation lines as ``name -> [[x0, y0], [x1, y1]]``."""
        hl, hw = self.hl, self.hw
        seg: dict[str, np.ndarray] = {
            "TOUCHLINE_TOP": [[-hl, hw], [hl, hw]],
            "TOUCHLINE_BOTTOM": [[-hl, -hw], [hl, -hw]],
            "GOAL_LINE_L": [[-hl, -hw], [-hl, hw]],
            "GOAL_LINE_R": [[hl, -hw], [hl, hw]],
            "HALFWAY_LINE": [[0.0, -hw], [0.0, hw]],
        }
        bw, gw = PEN_BOX_WIDTH / 2, GOAL_AREA_WIDTH / 2
        for side, s in (("L", -1.0), ("R", 1.0)):
            gl = s * hl  # goal line x
            xb = s * (hl - PEN_BOX_DEPTH)
            xg = s * (hl - GOAL_AREA_DEPTH)
            seg[f"{side}_PEN_BOX_TOP"] = [[gl, bw], [xb, bw]]
            seg[f"{side}_PEN_BOX_BOTTOM"] = [[gl, -bw], [xb, -bw]]
            seg[f"{side}_PEN_BOX_FRONT"] = [[xb, -bw], [xb, bw]]
            seg[f"{side}_GOAL_AREA_TOP"] = [[gl, gw], [xg, gw]]
            seg[f"{side}_GOAL_AREA_BOTTOM"] = [[gl, -gw], [xg, -gw]]
            seg[f"{side}_GOAL_AREA_FRONT"] = [[xg, -gw], [xg, gw]]
        return {k: np.asarray(v, dtype=np.float64) for k, v in seg.items()}

    def curves(self) -> dict[str, Polyline]:
        """Centre circle and penalty arcs (outside the box) as sampled polylines."""
        hl = self.hl
        r = CIRCLE_RADIUS
        out: dict[str, Polyline] = {
            "CENTRE_CIRCLE": _arc(0.0, 0.0, r, 0.0, 2 * np.pi, 96)
        }
        theta = float(np.arccos((PEN_BOX_DEPTH - PEN_SPOT_DIST) / r))
        out["L_PEN_ARC"] = _arc(-hl + PEN_SPOT_DIST, 0.0, r, -theta, theta)
        out["R_PEN_ARC"] = _arc(hl - PEN_SPOT_DIST, 0.0, r, np.pi - theta, np.pi + theta)
        return out

    def spots(self) -> dict[str, np.ndarray]:
        hl = self.hl
        return {
            "CENTRE_SPOT": np.array([0.0, 0.0]),
            "L_PEN_SPOT": np.array([-hl + PEN_SPOT_DIST, 0.0]),
            "R_PEN_SPOT": np.array([hl - PEN_SPOT_DIST, 0.0]),
        }

    def polylines(self) -> dict[str, Polyline]:
        """Everything drawable: segments (2 points) and curves (sampled)."""
        out: dict[str, Polyline] = dict(self.segments())
        out.update(self.curves())
        return out

    # ------------------------------------------------------------------
    def keypoints(self) -> dict[str, tuple[float, float]]:
        """Named, visually identifiable points ``id -> (x, y)`` in metres."""
        hl, hw = self.hl, self.hw
        r = CIRCLE_RADIUS
        kp: dict[str, tuple[float, float]] = {
            "CENTRE_SPOT": (0.0, 0.0),
            "HALFWAY_TOP": (0.0, hw),
            "HALFWAY_BOTTOM": (0.0, -hw),
            "CENTRE_CIRCLE_TOP": (0.0, r),  # circle ∩ halfway line
            "CENTRE_CIRCLE_BOTTOM": (0.0, -r),
            "CENTRE_CIRCLE_LEFT": (-r, 0.0),  # tangent extremes on the long axis
            "CENTRE_CIRCLE_RIGHT": (r, 0.0),
        }
        dx_arc = PEN_BOX_DEPTH - PEN_SPOT_DIST
        dy_arc = float(np.sqrt(r * r - dx_arc * dx_arc))
        for side, s in (("L", -1.0), ("R", 1.0)):
            gl = s * hl
            xb = s * (hl - PEN_BOX_DEPTH)
            xg = s * (hl - GOAL_AREA_DEPTH)
            bw, gw = PEN_BOX_WIDTH / 2, GOAL_AREA_WIDTH / 2
            for vs, sy in (("TOP", 1.0), ("BOTTOM", -1.0)):
                kp[f"{side}_CORNER_{vs}"] = (gl, sy * hw)
                kp[f"{side}_PEN_BOX_{vs}_OUTER"] = (xb, sy * bw)
                kp[f"{side}_PEN_BOX_{vs}_GOAL"] = (gl, sy * bw)
                kp[f"{side}_GOAL_AREA_{vs}_OUTER"] = (xg, sy * gw)
                kp[f"{side}_GOAL_AREA_{vs}_GOAL"] = (gl, sy * gw)
                kp[f"{side}_GOALPOST_{vs}"] = (gl, sy * GOAL_WIDTH / 2)
                # penalty arc ∩ box front edge
                kp[f"{side}_PEN_ARC_{vs}"] = (xb, sy * dy_arc)
            kp[f"{side}_PEN_SPOT"] = (s * (hl - PEN_SPOT_DIST), 0.0)
            kp[f"{side}_PEN_ARC_MID"] = (s * (hl - PEN_SPOT_DIST - r), 0.0)
        return kp

    def keypoint_array(self) -> tuple[list[str], np.ndarray]:
        kp = self.keypoints()
        ids = list(kp)
        return ids, np.array([kp[i] for i in ids], dtype=np.float64)

    def to_dict(self) -> dict:
        """JSON-able description for the frontend diagram."""
        return {
            "length": self.length,
            "width": self.width,
            "polylines": {k: v.tolist() for k, v in self.polylines().items()},
            "keypoints": {k: list(v) for k, v in self.keypoints().items()},
            "spots": {k: v.tolist() for k, v in self.spots().items()},
        }
