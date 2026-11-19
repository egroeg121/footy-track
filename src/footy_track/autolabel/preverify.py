"""Pre-verify a proposed window: decide ``auto_ok`` vs ``flagged`` per frame
(docs/design/full_frame_labelling.md section 4 stage 3, 4b, 6b).

Inputs: the window's disagreement scores (``proposals/<window_id>.npz`` from
:mod:`footy_track.autolabel.propose`), the merged ball track per frame
(:class:`BallProposal`, ``None`` where no ball) and an injectable :class:`BallVerifier`.

BALL rule (``verdict``): a frame is ``auto_ok`` only when ALL hold, otherwise ``flagged`` with
every failing reason code listed:

* ``no_ball_proposal``     - the track has no ball here (never auto_ok; a human confirms absence)
* ``tracker_disagree``     - one pass has a ball and the other not, or the forward/backward
                             centres are further apart than ``agree_px``
* ``track_not_smooth``     - the ball jumps more than ``max_step_px`` per frame from a
                             neighbouring proposed frame
* ``verifier_not_ball``    - verdict is ``not_ball`` / ``unsure`` / ``invalid`` / missing
* ``verifier_low_conf``    - verdict ``ball`` but confidence < threshold

Threshold precedence: explicit ``conf_threshold`` > ``ball_conf_threshold`` in the calibration
JSON (the audit-derived value, see ``audit_stats.pick_threshold``) > ``DEFAULT_CONF_THRESHOLD``
(0.95; 0.8 measured about 65% precise, so never the default). The verifier is only called for
frames that already pass the tracker checks.

PLAYER rule (``player_verdict``, separate from the ball): ``player_auto_ok_policy="never"``
(default; section 6b, the one-game detector mAP50 is not trusted) flags every frame with
``player_not_trusted``. ``"stable_count"`` is EXPERIMENTAL: auto_ok when the detector/SAM3
person count is stable (``person_flag`` False, zero person disagreement, count > 0), else
``player_count_unstable``.

Ordering / audit / priority: ``flagged`` is ordered by (disagreement score desc, frame_index
asc). ``audit`` is a seeded (``seed`` + window_id) blind sample of ``audit_frac`` (default 10%)
of the ball-``auto_ok`` frames, stratified: auto_ok frames in frame order are cut into
``round(n*frac)`` (at least 1) contiguous strata and one is drawn per stratum. Each frame has
``priority`` = max(disagreement score, 1 - verifier confidence) for the targeting queue.

``preverify/<window_id>.json`` (``version`` 1): ``window_id``, ``verifier`` {model,
prompt_version}, ``params``, ``frames`` [{frame_index, verdict, reasons, player_verdict,
player_reasons, verifier_verdict, verifier_confidence, disagreement, priority}],
``flagged_order`` [frame_index], ``audit`` [frame_index], ``counts``.
"""

from __future__ import annotations

import json
import math
import random
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import numpy as np

PREVERIFY_VERSION = 1
DEFAULT_CONF_THRESHOLD = 0.95
DEFAULT_AGREE_PX = 60.0
DEFAULT_MAX_STEP_PX = 120.0
DEFAULT_AUDIT_FRAC = 0.10
VERIFY_PROMPT_VERSION = "verify-v2.2"
AUTO_OK, FLAGGED = "auto_ok", "flagged"
PLAYER_POLICIES = ("never", "stable_count")


@dataclass(frozen=True)
class BallProposal:
    """Merged-track ball for one frame, in pixels (centre + size)."""

    frame_index: int
    cx: float
    cy: float
    w: float = 0.0
    h: float = 0.0


@dataclass(frozen=True)
class VerifierAnswer:
    verdict: str  # ball | not_ball | unsure | invalid
    confidence: float | None = None


class BallVerifier(Protocol):
    model: str
    prompt_version: str

    def verify(self, proposals: Sequence[BallProposal]) -> Mapping[int, VerifierAnswer]:
        """Answer per frame_index (a missing key counts as invalid)."""


class ClaudeBallVerifier:
    """Thin adapter over the existing Claude verify-v2.2 path (``claude_cli.ask_batch``).

    ``make_item(proposal) -> {"image", "n_candidates"}`` renders the question crop
    (``render_v2.render_question``) and is injected; the claude_cli import is lazy.
    Item ids are ``str(frame_index)``.
    """

    def __init__(
        self,
        make_item: Callable[[BallProposal], dict],
        cwd: Path,
        model: str = "opus",
        prompt_version: str = VERIFY_PROMPT_VERSION,
        batch_size: int = 8,
    ) -> None:
        self.make_item, self.cwd = make_item, Path(cwd)
        self.model, self.prompt_version, self.batch_size = (
            model,
            prompt_version,
            batch_size,
        )

    def verify(self, proposals: Sequence[BallProposal]) -> dict[int, VerifierAnswer]:
        from footy_track.autolabel import claude_cli  # noqa: PLC0415 lazy
        from footy_track.autolabel.verify_v2 import normalise_answer  # noqa: PLC0415

        out: dict[int, VerifierAnswer] = {}
        for i in range(0, len(proposals), self.batch_size):
            chunk = proposals[i : i + self.batch_size]
            items = [{**self.make_item(p), "id": str(p.frame_index)} for p in chunk]
            res = claude_cli.ask_batch(
                items, self.model, self.cwd, self.prompt_version
            )[-1]
            for a in res.answers or []:
                n = normalise_answer(a)
                out[int(a["id"])] = VerifierAnswer(n["verdict"], n["confidence"])
        return out


def load_calibration(path: Path | None) -> dict:
    """Calibration JSON (``{"ball_conf_threshold": 0.97, ...}``); ``{}`` if absent."""
    if path is None or not Path(path).is_file():
        return {}
    return json.loads(Path(path).read_text())


@dataclass
class PreverifyConfig:
    conf_threshold: float | None = None  # explicit override
    calibration_path: Path | None = None
    agree_px: float = DEFAULT_AGREE_PX
    max_step_px: float = DEFAULT_MAX_STEP_PX
    audit_frac: float = DEFAULT_AUDIT_FRAC
    seed: int = 0
    player_auto_ok_policy: str = "never"

    def resolved_threshold(self) -> float:
        if self.conf_threshold is not None:
            return self.conf_threshold
        cal = load_calibration(self.calibration_path).get("ball_conf_threshold")
        return float(cal) if cal is not None else DEFAULT_CONF_THRESHOLD


@dataclass
class FrameVerdict:
    frame_index: int
    verdict: str
    reasons: list[str]
    player_verdict: str
    player_reasons: list[str]
    verifier_verdict: str | None
    verifier_confidence: float | None
    disagreement: float
    priority: float


@dataclass
class PreverifyResult:
    window_id: str
    frames: list[FrameVerdict]
    flagged_order: list[int]
    audit: list[int]
    params: dict
    verifier: dict = field(default_factory=dict)

    def to_json(self) -> dict:
        auto = sum(f.verdict == AUTO_OK for f in self.frames)
        return {
            "version": PREVERIFY_VERSION,
            "window_id": self.window_id,
            "verifier": self.verifier,
            "params": self.params,
            "frames": [f.__dict__ for f in self.frames],
            "flagged_order": self.flagged_order,
            "audit": self.audit,
            "counts": {
                "frames": len(self.frames),
                "auto_ok": auto,
                "flagged": len(self.frames) - auto,
                "audit": len(self.audit),
            },
        }


def audit_sample(
    auto_ok_frames: Sequence[int], frac: float, seed: int, window_id: str
) -> list[int]:
    """Seeded stratified sample: ``round(n*frac)`` (>=1 if n>0) contiguous strata, one each."""
    frames = sorted(auto_ok_frames)
    n = len(frames)
    if n == 0 or frac <= 0:
        return []
    m = min(n, max(1, round(n * frac)))
    rng = random.Random(f"{seed}:{window_id}")
    return [frames[rng.randrange(s * n // m, (s + 1) * n // m)] for s in range(m)]


def _smooth(i: int, props: Sequence[BallProposal | None], max_step: float) -> bool:
    p = props[i]
    assert p is not None
    for j in (i - 1, i + 1):
        q = props[j] if 0 <= j < len(props) else None
        if q is not None and math.dist((p.cx, p.cy), (q.cx, q.cy)) > max_step:
            return False
    return True


def _player_reasons(scores: Mapping[str, np.ndarray], i: int, policy: str) -> list[str]:
    if policy == "never":
        return ["player_not_trusted"]
    stable = (  # EXPERIMENTAL
        not bool(scores["person_flag"][i])
        and float(scores["person_component"][i]) == 0.0
        and int(scores["det_person_count"][i]) > 0
    )
    return [] if stable else ["player_count_unstable"]


def preverify_window(  # noqa: PLR0914
    window_id: str,
    scores: Mapping[str, np.ndarray],
    proposals: Sequence[BallProposal | None],
    verifier: BallVerifier,
    config: PreverifyConfig | None = None,
) -> PreverifyResult:
    """``proposals``: one entry per frame in ``scores['frame_index']`` order (None = no ball)."""
    cfg = config or PreverifyConfig()
    if cfg.player_auto_ok_policy not in PLAYER_POLICIES:
        raise ValueError(f"player_auto_ok_policy must be one of {PLAYER_POLICIES}")
    thr = cfg.resolved_threshold()
    fi = [int(x) for x in scores["frame_index"]]
    if len(proposals) != len(fi):
        raise ValueError("proposals must have one entry per frame")
    reasons: list[list[str]] = [[] for _ in fi]
    candidates: list[BallProposal] = []
    for i, p in enumerate(proposals):
        if p is None:
            reasons[i].append("no_ball_proposal")
            continue
        gap = float(scores["ball_gap_px"][i])
        if bool(scores["ball_presence_mismatch"][i]) or (
            not math.isnan(gap) and gap > cfg.agree_px
        ):
            reasons[i].append("tracker_disagree")
        if not _smooth(i, proposals, cfg.max_step_px):
            reasons[i].append("track_not_smooth")
        if not reasons[i]:
            candidates.append(p)
    answers = dict(verifier.verify(candidates)) if candidates else {}
    asked = {c.frame_index for c in candidates}
    frames: list[FrameVerdict] = []
    for i, f in enumerate(fi):
        ans = answers.get(f) if f in asked else None
        if f in asked:
            if ans is None or ans.verdict != "ball":
                reasons[i].append("verifier_not_ball")
            elif ans.confidence is None or ans.confidence < thr:
                reasons[i].append("verifier_low_conf")
        pr = _player_reasons(scores, i, cfg.player_auto_ok_policy)
        dis = float(scores["score"][i])
        conf = ans.confidence if ans else None
        margin = 1.0 - (conf or 0.0) if f in asked else 0.0
        frames.append(
            FrameVerdict(
                f,
                FLAGGED if reasons[i] else AUTO_OK,
                reasons[i],
                FLAGGED if pr else AUTO_OK,
                pr,
                ans.verdict if ans else None,
                conf,
                dis,
                max(dis, margin),
            )
        )
    order = [
        x.frame_index
        for x in sorted(
            (x for x in frames if x.verdict == FLAGGED),
            key=lambda x: (-x.disagreement, x.frame_index),
        )
    ]
    audit = audit_sample(
        [x.frame_index for x in frames if x.verdict == AUTO_OK],
        cfg.audit_frac,
        cfg.seed,
        window_id,
    )
    params = {
        "conf_threshold": thr,
        "agree_px": cfg.agree_px,
        "max_step_px": cfg.max_step_px,
        "audit_frac": cfg.audit_frac,
        "seed": cfg.seed,
        "player_auto_ok_policy": cfg.player_auto_ok_policy,
    }
    prov = {"model": verifier.model, "prompt_version": verifier.prompt_version}
    return PreverifyResult(window_id, frames, order, audit, params, prov)


def write_preverify(result: PreverifyResult, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{result.window_id}.json"
    path.write_text(json.dumps(result.to_json(), indent=1, sort_keys=True))
    return path
