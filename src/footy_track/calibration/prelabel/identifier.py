"""Locate pitch keypoints (and name visible lines) in a broadcast frame.

Two implementations of :class:`Identifier`:

* :class:`ClaudeIdentifier` sends a reference pitch diagram (stable, cached
  prefix) and the frame with a light labelled pixel grid to Claude, which
  localises *any* clearly visible keypoint directly (candidates are optional
  hints it may cite by id) and may name visible lines with two points each.
  Structured JSON comes back. ``anthropic`` is imported lazily.
* :class:`MockIdentifier` takes the human GT points and (optionally) adds
  Gaussian noise / drops / mislabels them, simulating Claude's imprecise
  localisation; for end-to-end tests, the noise sweep and no-cost debugging.

The coarse homography from these points is then refined against the white-line
mask (:mod:`footy_track.calibration.refine`), so the labels only need to be
roughly right.
"""

from __future__ import annotations

import base64
import json
import zlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import cv2
import numpy as np

from footy_track.calibration import geometry as geo
from footy_track.calibration.pitch_model import PitchModel
from footy_track.calibration.prelabel.candidates import Detection
from footy_track.calibration.prelabel.fit import Label
from footy_track.calibration.refine import LineLabel
from footy_track.clip_id import canonical_clip_id

#: Bump whenever PROMPT / SCHEMA / the diagram change: it is part of the run id.
PROMPT_VERSION = "p2"
DEFAULT_MODEL = "claude-opus-5-5"
DEFAULT_EFFORT = "medium"
FALLBACK_BETA = "server-side-fallback-2026-07-01"

#: Grid spacing (px of the image sent to Claude) of the coordinate overlay.
GRID_STEP = 100


@dataclass
class IdentifyInput:
    clip: str
    frame_index: int
    pitch: PitchModel
    detection: Detection
    frame_jpeg: bytes  # frame with the labelled coordinate grid (+ candidate hints)
    diagram_jpeg: bytes  # reference pitch diagram with keypoint and line ids


@dataclass
class IdentifyResult:
    unusable: bool = False
    unusable_reason: str | None = None
    error: str | None = None  # API/parsing failure (not a verdict on the frame)
    view: dict = field(default_factory=dict)  # end, main_camera_wide, reverse_angle_suspected, notes
    labels: list[Label] = field(default_factory=list)  # point labels (normalised image coords)
    line_labels: list[LineLabel] = field(default_factory=list)  # named lines, >= 2 points each
    usage: dict | None = None


class Identifier(Protocol):
    source: str  # "claude" | "classical"

    def run_params(self) -> dict: ...

    def run_prefix(self) -> str: ...

    def identify(self, inp: IdentifyInput) -> IdentifyResult: ...


# --------------------------------------------------------------------------
# the image sent to Claude
# --------------------------------------------------------------------------


def render_prompt_frame(det: Detection, step: int = GRID_STEP, hints: bool = True) -> np.ndarray:
    """The frame as sent: (optionally) the numbered candidate hints, plus a light
    coordinate grid with ticks every ``step`` px labelled in *sent-image pixels*
    (origin top-left, +x right, +y down)."""
    from footy_track.calibration.prelabel.candidates import render_candidates  # noqa: PLC0415

    img = render_candidates(det) if hints else det.work_bgr.copy()
    h, w = img.shape[:2]
    grid = img.copy()
    for x in range(step, w, step):
        cv2.line(grid, (x, 0), (x, h - 1), (255, 255, 255), 1, cv2.LINE_AA)
    for y in range(step, h, step):
        cv2.line(grid, (0, y), (w - 1, y), (255, 255, 255), 1, cv2.LINE_AA)
    img = cv2.addWeighted(grid, 0.22, img, 0.78, 0)
    font, fs = cv2.FONT_HERSHEY_SIMPLEX, 0.42
    for x in range(step, w, step):
        for y in (12, h - 5):
            cv2.putText(img, str(x), (x + 2, y), font, fs, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(img, str(x), (x + 2, y), font, fs, (255, 255, 255), 1, cv2.LINE_AA)
    for y in range(step, h, step):
        for x in (3, w - 36):
            cv2.putText(img, str(y), (x, y - 3), font, fs, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(img, str(y), (x, y - 3), font, fs, (255, 255, 255), 1, cv2.LINE_AA)
    return img


# --------------------------------------------------------------------------
# reference diagram
# --------------------------------------------------------------------------


def render_diagram(pitch: PitchModel, width: int = 1500) -> np.ndarray:
    """Top-down pitch, +x right, +y UP (TOP = far touchline): every keypoint
    marked with its id (yellow) and every straight line with its id (light blue).
    Used as the reference image in the prompt."""
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
    # straight line ids, in light blue, centred on their segment (just off it)
    for lid, seg in pitch.segments().items():
        (x0, y0), (x1, y1) = seg
        mx, my = (x0 + x1) / 2, (y0 + y1) / 2
        px, py = P(mx, my)
        (tw, th), _ = cv2.getTextSize(lid, font, 0.38, 1)
        vertical = abs(x1 - x0) < abs(y1 - y0)
        org = (px + 5, py + th // 2) if vertical else (px - tw // 2, py - 5)
        placed.append((org[0] - 2, org[1] - th - 2, org[0] + tw + 2, org[1] + 2))
        cv2.putText(img, lid, org, font, 0.38, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(img, lid, org, font, 0.38, (255, 200, 120), 1, cv2.LINE_AA)

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
You locate football pitch markings in broadcast frames so a camera-to-pitch homography can be fitted. \
You are shown (1) a REFERENCE DIAGRAM of the pitch with every named keypoint id (yellow) and every \
straight line id (light blue), then (2) one broadcast FRAME.

Pitch frame: origin at the centre spot, +x runs L -> R along the long axis, +y points to TOP. \
In the diagram TOP is the FAR touchline of the standard main-camera view (camera on the near \
touchline at about the halfway line, looking across), L is the left goal end and R the right one. \
So in a standard main-camera view the image-left end is L, the image-right end is R and the \
far touchline is TOP. Reverse-angle shots (camera on the other side, or behind a goal) mirror \
or rotate this: if you suspect one, set reverse_angle_suspected and only label what the markings \
themselves make unambiguous (a lone penalty area cannot tell you L from R, so be conservative).

The FRAME carries a light white grid with ticks every 100 px, labelled in pixels of the image \
you are shown (x along the top and bottom edges, y along the left and right edges; origin top-left, \
x right, y down). Read positions off this grid and give x, y in those pixel coordinates. White pitch \
lines are lightly tinted blue. Small numbered circles are OPTIONAL hints from a classical detector \
(amber = line/line intersection, magenta = circle-or-arc/line intersection, sky-blue = a line end \
meeting another line, green = a spot). They are often missing or misplaced: do not restrict yourself \
to them. If a numbered circle sits exactly on a keypoint you are labelling, you may give its \
candidate_id and its exact position will be used instead of your x, y; otherwise use candidate_id 0.

Task:
- Set unusable=true for close-ups, replays, graphics/non-pitch shots, or frames where fewer than \
about 4 pitch landmarks are distinguishable.
- points: every keypoint from the diagram that you can clearly see AND identify, whether or not a \
hint circle marks it: penalty and centre spots, centre-circle tangent points, penalty-arc/box \
intersections, box and goal-area corners, corner flags, goalposts, halfway-line ends. Give x, y \
(pixels of the shown image, as precise as you can; a few pixels matter) and confidence 0..1. \
Use ONLY ids from the diagram. Omit what you cannot place; a wrong id is worse than a missing one. \
Do not put a point where a keypoint would be if it is hidden or out of frame.
- lines: for each straight white line you can clearly identify (e.g. R_PEN_BOX_FRONT, HALFWAY_LINE, \
TOUCHLINE_TOP), give its id and two points p1, p2 that lie ON the visible line, well apart \
(pixels of the shown image). Name only lines whose identity you are sure of.
- It is fine to give only points, only lines, or both. Points need not be on the lines you name.
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


def line_ids(pitch: PitchModel) -> list[str]:
    return list(pitch.segments())


def build_schema(pitch: PitchModel) -> dict:
    kids = keypoint_ids(pitch)
    lids = line_ids(pitch)
    xy = {
        "type": "object",
        "properties": {"x": {"type": "number"}, "y": {"type": "number"}},
        "required": ["x", "y"],
        "additionalProperties": False,
    }
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
            "points": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "keypoint_id": {"type": "string", "enum": kids},
                        "x": {"type": "number"},
                        "y": {"type": "number"},
                        "confidence": {"type": "number"},
                        "candidate_id": {"type": "integer"},  # 0 = none
                    },
                    "required": ["keypoint_id", "x", "y", "confidence", "candidate_id"],
                    "additionalProperties": False,
                },
            },
            "lines": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "line_id": {"type": "string", "enum": lids},
                        "p1": xy,
                        "p2": xy,
                        "confidence": {"type": "number"},
                    },
                    "required": ["line_id", "p1", "p2", "confidence"],
                    "additionalProperties": False,
                },
            },
        },
        "required": ["unusable", "unusable_reason", "view", "points", "lines"],
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


def build_content(inp: IdentifyInput, cache_breakpoint: bool = True) -> list[dict]:
    """The user-message content blocks (identical for the API and CLI identifiers).

    Stable prefix first (reference diagram) with ONE cache breakpoint at its end;
    everything per-frame (image, its size) comes after it."""
    w, h = inp.detection.size
    blocks = [
        {"type": "text", "text": "REFERENCE DIAGRAM (keypoint ids in yellow, straight line ids in light blue):"},
        _img_block(inp.diagram_jpeg),
        {"type": "text", "text": "End of reference. The frame to analyse follows.",
         "cache_control": {"type": "ephemeral"}},
        {"type": "text", "text": f"FRAME ({w} x {h} px; grid ticks every {GRID_STEP} px):"},
        _img_block(inp.frame_jpeg),
        {"type": "text", "text": "Locate the visible keypoints and name the visible lines."},
    ]
    if not cache_breakpoint:  # the CLI manages its own cache_control; a second one is a 400
        for b in blocks:
            b.pop("cache_control", None)
    return blocks


def result_from_data(data: dict, inp: IdentifyInput, usage: dict | None) -> IdentifyResult:
    """Structured JSON from the model -> labels (normalised image coords)."""
    w, h = inp.detection.size
    valid = {c.id: c for c in inp.detection.candidates}
    kps = inp.pitch.keypoints()
    labels = []
    for pt in data.get("points", []):
        if pt.get("keypoint_id") not in kps:
            continue
        cand = valid.get(int(pt.get("candidate_id") or 0))
        if cand is not None:  # a valid hint id: its exact position beats the estimate
            labels.append(Label(pt["keypoint_id"], cand.uv, float(pt["confidence"]), cand.id))
            continue
        u, v = float(pt["x"]) / w, float(pt["y"]) / h
        if -0.02 <= u <= 1.02 and -0.02 <= v <= 1.02:
            labels.append(Label(pt["keypoint_id"], (u, v), float(pt["confidence"]), None))
    segs = inp.pitch.segments()
    line_labels = []
    for ln in data.get("lines", []):
        if ln.get("line_id") not in segs:
            continue
        pts = [(float(ln[k]["x"]) / w, float(ln[k]["y"]) / h) for k in ("p1", "p2")]
        if np.hypot((pts[0][0] - pts[1][0]) * w, (pts[0][1] - pts[1][1]) * h) < 0.05 * w:
            continue  # too short to define a direction
        line_labels.append(LineLabel(ln["line_id"], pts, float(ln["confidence"])))
    return IdentifyResult(
        unusable=bool(data.get("unusable")), unusable_reason=data.get("unusable_reason") or None,
        view=data.get("view") or {}, labels=labels, line_labels=line_labels, usage=usage,
    )


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
        resp = self._call(SYSTEM, build_content(inp), build_schema(inp.pitch))
        usage = _usage_dict(resp)
        data, err = self._parse(resp)
        if err:
            return IdentifyResult(error=err, usage=usage)
        return result_from_data(data, inp, usage)

    def verify(self, overlay_jpeg: bytes) -> dict:
        """Optional second call: does the projected model sit on the markings?"""
        content = [_img_block(overlay_jpeg), {"type": "text", "text": "Does the drawn pitch model match?"}]
        resp = self._call(REVIEW_SYSTEM, content, REVIEW_SCHEMA)
        data, err = self._parse(resp)
        usage = _usage_dict(resp)
        if err:
            return {"match": None, "reason": err, "usage": usage}
        return {"match": bool(data["match"]), "reason": data["reason"], "usage": usage}


class ClaudeCLIIdentifier:
    """Claude via the Claude Code CLI (``claude -p``), for machines with a claude.ai
    subscription login but no API key. One subprocess per frame, no tools, no
    session persistence, run from a scratch cwd with no settings/MCP/skills so
    no project context leaks in. Same prompt/schema as :class:`ClaudeIdentifier`.

    ``--input-format stream-json`` needs ``--output-format stream-json``; the
    structured answer is ``structured_output`` on the final ``result`` event.
    ``command`` is injectable (tests use a fake executable)."""

    source = "claude"

    def __init__(
        self,
        model: str = "opus",
        effort: str = DEFAULT_EFFORT,
        command: str | list[str] = "claude",
        timeout: float = 300.0,
        retries: int = 1,
        cwd: str | Path | None = None,
    ) -> None:
        self.model, self.effort, self.timeout, self.retries = model, effort, timeout, retries
        self.command = [command] if isinstance(command, str) else list(command)
        self._cwd = Path(cwd) if cwd else None

    def run_prefix(self) -> str:
        return f"claudecli-{self.model}-{PROMPT_VERSION}"

    def run_params(self) -> dict:
        return {"model": self.model, "prompt_version": PROMPT_VERSION, "effort": self.effort, "backend": "claude-cli"}

    def _argv(self, schema: dict) -> list[str]:
        return [
            *self.command, "-p", "--input-format", "stream-json", "--output-format", "stream-json", "--verbose",
            "--json-schema", json.dumps(schema), "--tools", "", "--system-prompt", SYSTEM,
            "--model", self.model, "--effort", self.effort, "--no-session-persistence",
            "--setting-sources", "", "--strict-mcp-config", "--mcp-config", '{"mcpServers":{}}',
            "--disable-slash-commands",
        ]

    def _run_once(self, schema: dict, content: list[dict]) -> tuple[dict | None, dict, str | None]:
        """-> (structured_output, usage, error)."""
        import subprocess  # noqa: PLC0415
        import tempfile  # noqa: PLC0415

        msg = json.dumps({"type": "user", "message": {"role": "user", "content": content}})
        cwd = self._cwd or Path(tempfile.gettempdir())
        usage: dict = {}
        try:
            cp = subprocess.run(
                self._argv(schema), input=msg, capture_output=True, text=True, timeout=self.timeout, cwd=cwd,
            )
        except subprocess.TimeoutExpired:
            return None, usage, f"claude CLI timed out after {self.timeout:.0f}s"
        except OSError as exc:
            return None, usage, f"cannot run claude CLI: {exc}"
        result = None
        for line in cp.stdout.splitlines():
            try:
                ev = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(ev, dict) and ev.get("type") == "result":
                result = ev
        if result is None:
            return None, usage, f"no result event (exit {cp.returncode}): {(cp.stderr or cp.stdout)[-300:].strip()}"
        u = result.get("usage") or {}
        mu = result.get("modelUsage") or {}
        usage = {
            "model_served": next(iter(mu), None),
            **{k: u.get(k) for k in ("input_tokens", "output_tokens", "cache_creation_input_tokens",
                                     "cache_read_input_tokens")},
            "cost_usd": result.get("total_cost_usd"), "duration_ms": result.get("duration_ms"),
            "duration_api_ms": result.get("duration_api_ms"),
        }
        if result.get("is_error") or result.get("subtype") != "success":
            return None, usage, f"claude CLI error ({result.get('subtype')}): {str(result.get('result'))[:300]}"
        so = result.get("structured_output")
        if not isinstance(so, dict):
            return None, usage, f"no structured_output (stop_reason={result.get('stop_reason')})"
        return so, usage, None

    def identify(self, inp: IdentifyInput) -> IdentifyResult:
        content, schema = build_content(inp, cache_breakpoint=False), build_schema(inp.pitch)
        err, usage = None, {}
        for _ in range(self.retries + 1):
            data, usage, err = self._run_once(schema, content)
            if err is None:
                try:
                    return result_from_data(data, inp, usage)
                except (KeyError, TypeError, ValueError) as exc:
                    err = f"malformed structured output: {exc!r}"
        return IdentifyResult(error=err, usage=usage)



# --------------------------------------------------------------------------
# Mock
# --------------------------------------------------------------------------


class MockIdentifier:
    """Labels = the human GT correspondences, optionally degraded to simulate
    Claude's imprecise localisation.

    * ``sigma_px``: Gaussian position noise (std, per axis) in pixels of a
      1920 x 1080 frame; ``0`` reproduces the GT exactly.
    * ``drop_prob``: probability of dropping each point; ``mislabel_prob``:
      probability of giving a point another keypoint's id.
    * ``lines``: also emit line labels (two noisy points on each visible
      straight line, derived from the GT homography).
    * ``corrupt=k``: swap the ids of k label pairs (bad proposals on purpose).
    * ``seed``: noise stream (combined with clip and frame index).

    Frames without GT are ``unusable``.
    """

    source = "classical"

    def __init__(
        self, gt_dir: Path | str, corrupt: int = 0, sigma_px: float = 0.0, drop_prob: float = 0.0,
        mislabel_prob: float = 0.0, lines: bool = False, seed: int = 0,
    ) -> None:
        self.gt_dir, self.corrupt, self.sigma_px = Path(gt_dir), corrupt, sigma_px
        self.drop_prob, self.mislabel_prob, self.lines, self.seed = drop_prob, mislabel_prob, lines, seed

    def run_prefix(self) -> str:
        if self.sigma_px or self.drop_prob or self.mislabel_prob or self.lines:
            return f"mock-s{self.sigma_px:g}"
        return "mock" + (f"-corrupt{self.corrupt}" if self.corrupt else "")

    def run_params(self) -> dict:
        return {
            "model": None, "prompt_version": None, "effort": None, "mock_gt": True, "corrupt": self.corrupt,
            "sigma_px": self.sigma_px, "drop_prob": self.drop_prob, "mislabel_prob": self.mislabel_prob,
            "lines": self.lines, "seed": self.seed,
        }

    def _gt_frame(self, clip: str, idx: int) -> dict | None:
        p = self.gt_dir / "calibration_gt" / f"{canonical_clip_id(clip)}.json"
        if not p.exists():
            return None
        return json.loads(p.read_text()).get("frames", {}).get(str(idx))

    def identify(self, inp: IdentifyInput) -> IdentifyResult:
        rec = self._gt_frame(inp.clip, inp.frame_index)
        if rec is None:
            return IdentifyResult(unusable=True, unusable_reason="mock: no human GT for this frame")
        iw, ih = rec.get("image_size") or (1920, 1080)
        rng = np.random.default_rng([self.seed, int(inp.frame_index), zlib.crc32(canonical_clip_id(inp.clip).encode())])
        sig = np.array([self.sigma_px / 1920.0, self.sigma_px / 1080.0]) * np.array([1920.0 / iw, 1080.0 / ih])
        kids = list(inp.pitch.keypoints())
        labels: list[Label] = []
        for c in rec["correspondences"]:
            if not c.get("pitch_id"):
                continue
            if rng.random() < self.drop_prob:
                continue
            kid = c["pitch_id"]
            if rng.random() < self.mislabel_prob:
                kid = kids[int(rng.integers(len(kids)))]
            u, v = np.array(c["image"], float) + (rng.normal(0, 1, 2) * sig if self.sigma_px else 0.0)
            labels.append(Label(kid, (float(u), float(v)), 1.0, None))
        if self.corrupt:
            for i in range(min(self.corrupt, len(labels) // 2)):
                a, b = labels[2 * i], labels[2 * i + 1]
                a.keypoint_id, b.keypoint_id = b.keypoint_id, a.keypoint_id
        line_labels: list[LineLabel] = []
        if self.lines and rec.get("H"):
            Hgt = np.array(rec["H"], float)
            for lid, seg in inp.pitch.segments().items():
                best = None
                for run in geo.project_polyline_to_image(Hgt, seg, 0.25):
                    inside = run[(run[:, 0] > 0.01) & (run[:, 0] < 0.99) & (run[:, 1] > 0.01) & (run[:, 1] < 0.99)]
                    if len(inside) >= 2 and (best is None or len(inside) > len(best)):
                        best = inside
                if best is None or np.hypot(*((best[-1] - best[0]) * [iw, ih])) < 0.12 * iw:
                    continue
                pts = [best[int(0.25 * (len(best) - 1))], best[int(0.75 * (len(best) - 1))]]
                pts = [tuple((p + rng.normal(0, 1, 2) * sig).tolist()) for p in pts]
                line_labels.append(LineLabel(lid, pts, 1.0))
        return IdentifyResult(
            view={"end": None, "main_camera_wide": None, "reverse_angle_suspected": None,
                  "notes": f"mock identifier (human GT, sigma={self.sigma_px:g}px)"},
            labels=labels, line_labels=line_labels,
        )
