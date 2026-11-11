"""Identify which candidate is which pitch keypoint.

Two implementations of :class:`Identifier`:

* :class:`ClaudeIdentifier` sends the candidate render and a reference pitch
  diagram to Claude and gets structured JSON back (labels, view info,
  unusable flag). ``anthropic`` is imported lazily, so nothing else needs it.
* :class:`MockIdentifier` derives the labels from existing human GT by nearest
  candidate; for end-to-end tests and debugging with no API cost.
"""

from __future__ import annotations

import base64
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import cv2
import numpy as np

from footy_track.calibration.pitch_model import PitchModel
from footy_track.calibration.prelabel.candidates import Detection
from footy_track.calibration.prelabel.fit import Label
from footy_track.clip_id import canonical_clip_id

#: Bump whenever PROMPT / SCHEMA / the diagram change: it is part of the run id.
PROMPT_VERSION = "p1"
DEFAULT_MODEL = "claude-opus-5-5"
DEFAULT_EFFORT = "medium"
FALLBACK_BETA = "server-side-fallback-2026-07-01"

#: An extra (candidate-less) point is snapped to a corner within this many
#: working px (fraction of width) or dropped.
SNAP_RADIUS_FRAC = 0.012

#: Keypoints that are NOT a crossing of two white lines (spots, circle tangent
#: extremes, arc midpoints, goalposts): a corner snap would be meaningless, so
#: candidate-less points of these kinds are dropped rather than snapped.
_NOT_INTERSECTIONS = ("SPOT", "_MID", "CIRCLE_LEFT", "CIRCLE_RIGHT", "GOALPOST")


@dataclass
class IdentifyInput:
    clip: str
    frame_index: int
    pitch: PitchModel
    detection: Detection
    candidates_jpeg: bytes  # frame with numbered candidates
    diagram_jpeg: bytes  # reference pitch diagram with keypoint ids


@dataclass
class IdentifyResult:
    unusable: bool = False
    unusable_reason: str | None = None
    error: str | None = None  # API/parsing failure (not a verdict on the frame)
    view: dict = field(default_factory=dict)  # end, main_camera_wide, reverse_angle_suspected, notes
    labels: list[Label] = field(default_factory=list)
    extra_points: list[dict] = field(default_factory=list)  # kept for the record (pre-snap)
    usage: dict | None = None


class Identifier(Protocol):
    source: str  # "claude" | "classical"

    def run_params(self) -> dict: ...

    def run_prefix(self) -> str: ...

    def identify(self, inp: IdentifyInput) -> IdentifyResult: ...


# --------------------------------------------------------------------------
# snapping extra points
# --------------------------------------------------------------------------


def snap_to_corner(det: Detection, uv: tuple[float, float]) -> tuple[float, float] | None:
    """Nearest white-line corner (two lines meeting) within a small radius of
    ``uv``, or ``None``. Uses the strongest Harris response of the line mask."""
    w, h = det.size
    r = int(max(6, SNAP_RADIUS_FRAC * w))
    cx, cy = uv[0] * w, uv[1] * h
    x0, x1 = max(0, int(cx) - r - 4), min(w, int(cx) + r + 5)
    y0, y1 = max(0, int(cy) - r - 4), min(h, int(cy) + r + 5)
    if x1 - x0 < 8 or y1 - y0 < 8:
        return None
    crop = cv2.dilate(det.lines_mask[y0:y1, x0:x1].astype(np.uint8) * 255, np.ones((3, 3), np.uint8))
    harris = cv2.cornerHarris(np.float32(crop), blockSize=5, ksize=3, k=0.04)
    if harris.max() <= 0:
        return None
    yy, xx = np.mgrid[y0:y1, x0:x1]
    near = np.hypot(xx - cx, yy - cy) <= r
    resp = np.where(near, harris, -np.inf)
    iy, ix = np.unravel_index(int(np.argmax(resp)), resp.shape)
    if resp[iy, ix] < 0.3 * harris.max():
        return None
    return ((x0 + ix + 0.5) / w, (y0 + iy + 0.5) / h)


def resolve_extra_points(res: IdentifyResult, det: Detection) -> None:
    """Turn ``res.extra_points`` into labels where they snap to a corner."""
    for ep in res.extra_points:
        if any(t in ep["keypoint_id"] for t in _NOT_INTERSECTIONS):
            ep["snapped_uv"] = None
            continue
        snapped = snap_to_corner(det, (ep["u"], ep["v"]))
        ep["snapped_uv"] = list(snapped) if snapped else None
        if snapped:
            res.labels.append(
                Label(ep["keypoint_id"], snapped, float(ep.get("confidence", 0.5)) * 0.9, None)
            )


# --------------------------------------------------------------------------
# reference diagram
# --------------------------------------------------------------------------


def render_diagram(pitch: PitchModel, width: int = 1500) -> np.ndarray:
    """Top-down pitch, +x right, +y UP (TOP = far touchline), every keypoint
    numbered with its id. Used as the reference image in the prompt."""
    margin = 70
    s = (width - 2 * margin) / pitch.length
    W = width
    H = int(pitch.width * s + 2 * margin)
    img = np.full((H, W, 3), (40, 90, 40), np.uint8)

    def P(x, y):
        return (int(round(margin + (x + pitch.hl) * s)), int(round(margin + (pitch.hw - y) * s)))

    for _, poly in pitch.polylines().items():
        pts = np.array([P(*p) for p in poly], np.int32)
        cv2.polylines(img, [pts], False, (255, 255, 255), 2, cv2.LINE_AA)
    for _, p in pitch.spots().items():
        cv2.circle(img, P(*p), 3, (255, 255, 255), -1, cv2.LINE_AA)
    font, fs = cv2.FONT_HERSHEY_SIMPLEX, 0.42
    placed: list[tuple[int, int, int, int]] = []
    kps = pitch.keypoints()

    def overlaps(r):
        return any(not (r[2] < o[0] or r[0] > o[2] or r[3] < o[1] or r[1] > o[3]) for o in placed)

    for kid, (x, y) in sorted(kps.items(), key=lambda kv: -abs(kv[1][1])):
        px, py = P(x, y)
        cv2.circle(img, (px, py), 5, (0, 215, 255), -1, cv2.LINE_AA)
        (tw, th), _ = cv2.getTextSize(kid, font, fs, 1)
        offs = [(8, -8), (8, th + 8), (-tw - 8, -8), (-tw - 8, th + 8), (-tw // 2, -14), (-tw // 2, th + 14),
                (8, -26), (8, th + 26), (-tw - 8, -26), (-tw - 8, th + 26), (14, 0), (-tw - 14, 0)]
        for dx, dy in offs:
            r = (px + dx - 2, py + dy - th - 2, px + dx + tw + 2, py + dy + 2)
            if r[0] >= 2 and r[2] < W - 2 and r[1] >= 2 and r[3] < H - 2 and not overlaps(r):
                placed.append(r)
                cv2.putText(img, kid, (px + dx, py + dy), font, fs, (0, 0, 0), 3, cv2.LINE_AA)
                cv2.putText(img, kid, (px + dx, py + dy), font, fs, (0, 255, 255), 1, cv2.LINE_AA)
                break
    big = 0.8
    for txt, org in (
        ("TOP = far touchline (+y)", (W // 2 - 150, 28)),
        ("BOTTOM = near touchline (-y)", (W // 2 - 175, H - 14)),
        ("L goal (-x)", (6, H // 2)),
        ("R goal (+x)", (W - 150, H // 2)),
    ):
        cv2.putText(img, txt, org, font, big, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(img, txt, org, font, big, (255, 255, 255), 1, cv2.LINE_AA)
    return img


def jpeg(img: np.ndarray, q: int = 88) -> bytes:
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, q])
    if not ok:
        raise RuntimeError("jpeg encode failed")
    return buf.tobytes()


# --------------------------------------------------------------------------
# Claude
# --------------------------------------------------------------------------

SYSTEM = """\
You identify football pitch landmarks in broadcast frames so a homography can be fitted.
You get two images: (1) a REFERENCE DIAGRAM of the pitch with every named keypoint id, and \
(2) the broadcast FRAME with numbered candidate markers on detected white-line features \
(white lines are lightly tinted blue-ish; marker colours: amber = line/line intersection, \
magenta = circle-or-arc/line intersection, sky-blue = line end meeting another line, green = small spot).

Pitch frame: origin at the centre spot, +x runs L -> R along the long axis, +y points to TOP. \
In the diagram TOP is the FAR touchline of the standard main-camera view (camera on the near \
touchline at about the halfway line, looking across), L is the left goal end and R the right one. \
So in a standard main-camera view the image-left end is L, the image-right end is R and the \
far touchline is TOP. Reverse-angle shots (camera on the other side, or behind a goal) mirror \
or rotate this: if you suspect one, set reverse_angle_suspected and only label points you can \
place unambiguously from the pitch markings themselves (e.g. which penalty area is which cannot \
be known from a lone box, so be conservative).

Task:
- Set unusable=true for close-ups, replays, graphics/non-pitch shots, or frames with fewer than \
about 4 distinguishable landmarks.
- Otherwise give labels: {candidate_id, keypoint_id, confidence 0..1} for candidates that are \
clearly a specific keypoint. Use ONLY ids from the diagram. Omit anything uncertain; wrong labels are \
worse than missing ones. Need at least 4 points, not all on one line.
- Optionally give extra_points for clearly visible keypoints that have NO candidate marker: \
{keypoint_id, u, v, confidence} with u,v the approximate position as fractions of the frame \
width/height (0..1, origin top-left). They will be snapped to the nearest line intersection or dropped.
- view.end: which end of the pitch is visible (L, R, centre, or unknown). view.main_camera_wide: \
true for a standard wide main-camera view. notes: one short sentence.
"""

REVIEW_SYSTEM = """\
You judge whether a pitch model has been drawn correctly over a broadcast frame. The pink and \
yellow lines are a candidate projection of the pitch markings (pink = straight lines, yellow = \
circle/arcs). Answer match=true only if they sit on the real white markings across the visible \
pitch (small offsets of a few pixels are fine); match=false if lines are misplaced, mirrored, on the \
wrong end, or clearly absent. Give a short reason.
"""


def keypoint_ids(pitch: PitchModel) -> list[str]:
    return list(pitch.keypoints())


def build_schema(pitch: PitchModel) -> dict:
    kids = keypoint_ids(pitch)
    return {
        "type": "object",
        "properties": {
            "unusable": {"type": "boolean"},
            "unusable_reason": {"type": "string"},
            "view": {
                "type": "object",
                "properties": {
                    "end": {"type": "string", "enum": ["L", "R", "centre", "unknown"]},
                    "main_camera_wide": {"type": "boolean"},
                    "reverse_angle_suspected": {"type": "boolean"},
                    "notes": {"type": "string"},
                },
                "required": ["end", "main_camera_wide", "reverse_angle_suspected", "notes"],
                "additionalProperties": False,
            },
            "labels": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "candidate_id": {"type": "integer"},
                        "keypoint_id": {"type": "string", "enum": kids},
                        "confidence": {"type": "number"},
                    },
                    "required": ["candidate_id", "keypoint_id", "confidence"],
                    "additionalProperties": False,
                },
            },
            "extra_points": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "keypoint_id": {"type": "string", "enum": kids},
                        "u": {"type": "number"},
                        "v": {"type": "number"},
                        "confidence": {"type": "number"},
                    },
                    "required": ["keypoint_id", "u", "v", "confidence"],
                    "additionalProperties": False,
                },
            },
        },
        "required": ["unusable", "unusable_reason", "view", "labels", "extra_points"],
        "additionalProperties": False,
    }


REVIEW_SCHEMA = {
    "type": "object",
    "properties": {"match": {"type": "boolean"}, "reason": {"type": "string"}},
    "required": ["match", "reason"],
    "additionalProperties": False,
}


def _img_block(jpg: bytes) -> dict:
    return {
        "type": "image",
        "source": {"type": "base64", "media_type": "image/jpeg", "data": base64.standard_b64encode(jpg).decode()},
    }


def _usage_dict(resp) -> dict:
    u = getattr(resp, "usage", None)
    out = {"model_served": getattr(resp, "model", None)}
    for k in ("input_tokens", "output_tokens", "cache_creation_input_tokens", "cache_read_input_tokens"):
        out[k] = getattr(u, k, None) if u is not None else None
    return out


class ClaudeIdentifier:
    """Claude via the Messages API with structured outputs. Never called in tests
    with a real client: pass ``client=`` (a mock) or mock the module."""

    source = "claude"

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        effort: str = DEFAULT_EFFORT,
        client=None,
        max_tokens: int = 8000,
        fallbacks: bool = True,
    ) -> None:
        self.model, self.effort, self.max_tokens, self.fallbacks = model, effort, max_tokens, fallbacks
        self._client = client

    # --- plumbing ---------------------------------------------------------
    @property
    def client(self):
        if self._client is None:
            try:
                import anthropic  # noqa: PLC0415 - optional dependency
            except ImportError as exc:  # pragma: no cover - environment dependent
                raise RuntimeError(
                    "the anthropic SDK is not installed: `uv sync --extra llm` (and set ANTHROPIC_API_KEY)"
                ) from exc
            self._client = anthropic.Anthropic()
        return self._client

    def run_prefix(self) -> str:
        return f"claude-{self.model}-{PROMPT_VERSION}"

    def run_params(self) -> dict:
        return {
            "model": self.model, "prompt_version": PROMPT_VERSION, "effort": self.effort,
            "fallbacks": "default" if self.fallbacks else None,
        }

    def _call(self, system: str, content: list[dict], schema: dict):
        kw = dict(
            model=self.model, max_tokens=self.max_tokens, system=system,
            messages=[{"role": "user", "content": content}],
            output_config={"effort": self.effort, "format": {"type": "json_schema", "schema": schema}},
        )
        if self.fallbacks:
            return self.client.beta.messages.create(betas=[FALLBACK_BETA], fallbacks="default", **kw)
        return self.client.messages.create(**kw)

    def _parse(self, resp) -> tuple[dict | None, str | None]:
        """-> (json, error)."""
        stop = getattr(resp, "stop_reason", None)
        if stop == "refusal":
            sd = getattr(resp, "stop_details", None)
            return None, f"refusal ({getattr(sd, 'category', None) or 'unspecified'})"
        if stop == "max_tokens":
            return None, "max_tokens reached before the answer finished"
        text = next((b.text for b in resp.content if getattr(b, "type", None) == "text"), None)
        if text is None:
            return None, f"no text block (stop_reason={stop})"
        try:
            return json.loads(text), None
        except json.JSONDecodeError as exc:
            return None, f"unparseable JSON: {exc}"

    # --- identify ---------------------------------------------------------
    def identify(self, inp: IdentifyInput) -> IdentifyResult:
        content = [
            {"type": "text", "text": "Image 1: REFERENCE DIAGRAM (keypoint ids)."},
            _img_block(inp.diagram_jpeg),
            {"type": "text", "text": "Image 2: FRAME with numbered candidates."},
            _img_block(inp.candidates_jpeg),
            {"type": "text", "text": "Identify the candidates."},
        ]
        resp = self._call(SYSTEM, content, build_schema(inp.pitch))
        usage = _usage_dict(resp)
        data, err = self._parse(resp)
        if err:
            return IdentifyResult(error=err, usage=usage)
        valid = {c.id: c for c in inp.detection.candidates}
        labels = []
        for lb in data.get("labels", []):
            c = valid.get(int(lb["candidate_id"]))
            if c is None:
                continue  # hallucinated id
            labels.append(Label(lb["keypoint_id"], c.uv, float(lb["confidence"]), c.id))
        res = IdentifyResult(
            unusable=bool(data.get("unusable")), unusable_reason=data.get("unusable_reason") or None,
            view=data.get("view") or {}, labels=labels, extra_points=list(data.get("extra_points", [])),
            usage=usage,
        )
        return res

    def verify(self, overlay_jpeg: bytes) -> dict:
        """Optional second call: does the projected model sit on the markings?"""
        content = [_img_block(overlay_jpeg), {"type": "text", "text": "Does the drawn pitch model match?"}]
        resp = self._call(REVIEW_SYSTEM, content, REVIEW_SCHEMA)
        data, err = self._parse(resp)
        usage = _usage_dict(resp)
        if err:
            return {"match": None, "reason": err, "usage": usage}
        return {"match": bool(data["match"]), "reason": data["reason"], "usage": usage}


# --------------------------------------------------------------------------
# Mock
# --------------------------------------------------------------------------


class MockIdentifier:
    """Labels = human GT correspondences snapped to the nearest candidate.

    GT points with no candidate within ``match_px`` become ``extra_points``
    (snapped or dropped like Claude's). Frames without GT are ``unusable``.
    ``corrupt=k`` swaps the keypoint ids of k label pairs, to produce bad
    proposals on purpose (tests, review-UI demos).
    """

    source = "classical"

    def __init__(self, gt_dir: Path | str, match_frac: float = 0.012, corrupt: int = 0) -> None:
        self.gt_dir, self.match_frac, self.corrupt = Path(gt_dir), match_frac, corrupt

    def run_prefix(self) -> str:
        return "mock" + (f"-corrupt{self.corrupt}" if self.corrupt else "")

    def run_params(self) -> dict:
        return {"model": None, "prompt_version": None, "effort": None, "mock_gt": True, "corrupt": self.corrupt}

    def _gt_frame(self, clip: str, idx: int) -> dict | None:
        p = self.gt_dir / "calibration_gt" / f"{canonical_clip_id(clip)}.json"
        if not p.exists():
            return None
        return json.loads(p.read_text()).get("frames", {}).get(str(idx))

    def identify(self, inp: IdentifyInput) -> IdentifyResult:
        rec = self._gt_frame(inp.clip, inp.frame_index)
        if rec is None:
            return IdentifyResult(unusable=True, unusable_reason="mock: no human GT for this frame")
        w, h = inp.detection.size
        tol = self.match_frac * w
        cands = inp.detection.candidates
        labels: list[Label] = []
        extras: list[dict] = []
        used: set[int] = set()
        for c in rec["correspondences"]:
            if not c.get("pitch_id"):
                continue
            u, v = c["image"]
            best, bd = None, 1e9
            for cd in cands:
                d = float(np.hypot(cd.uv[0] * w - u * w, cd.uv[1] * h - v * h))
                if d < bd and cd.id not in used:
                    best, bd = cd, d
            if best is not None and bd <= tol:
                used.add(best.id)
                labels.append(Label(c["pitch_id"], best.uv, 1.0, best.id))
            else:
                extras.append({"keypoint_id": c["pitch_id"], "u": u, "v": v, "confidence": 1.0})
        if self.corrupt:
            for i in range(min(self.corrupt, len(labels) // 2)):
                a, b = labels[2 * i], labels[2 * i + 1]
                a.keypoint_id, b.keypoint_id = b.keypoint_id, a.keypoint_id
        return IdentifyResult(
            view={"end": None, "main_camera_wide": None, "reverse_angle_suspected": None,
                  "notes": "mock identifier (human GT snapped to candidates)"},
            labels=labels, extra_points=extras,
        )
