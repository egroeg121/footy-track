"""Replay evaluation of the ball anchor-iteration loop (design ``ball_anchor_iteration.md`` task 13).

Before any rollout: replay the loop on already-labelled TRAIN clips with a simulated oracle and count the taps the
design needs. The oracle is the human ball GT (tier-1 sidecar marks); the candidate pool is the offline detector's
candidates of the same clip; the generator is the real one (:func:`anchor_search.generate_window`: constrained Viterbi,
frontier ranking, hop cap), run with the unchanged in-memory data model (no job file, no marks dir is written).

Loop per window (the window's selected set F = its human-labelled ball frames):

1. **Card.** The generator is re-run after EVERY verdict (adaptive). Its top-ranked card (hop, margin, confidence) is shown.
2. **Verdict.** The oracle answers ``ball`` when the card centre is within ``tol_px`` of the GT centre of that frame, else
   ``not_ball``. ``ball`` -> anchor at the card centre (hop 0); ``not_ball`` -> forbidden disc on that frame.
3. **Seed / re-seed.** With no anchor yet, or when the generator has no card left (every remaining frame beyond the hop
   cap, unreachable or judged), the next card is the top-1 detection among still-uncovered frames, oracle-judged
   (a "seed card", counted as a tap).
4. **Hand fallback.** When no detection is left to propose, the frame is placed by hand (design 3.4): one tap, anchor
   at the GT centre. Counted separately (``hand_taps``).

Coverage after each tap: a GT frame is *covered* when it carries a correct anchor, or the generator's current card
for it is within tolerance of GT (cards exist only for frames inside the hop cap, so nothing beyond the cap is
credited: a conservative reading of "covered by the path"). Milestones: taps until >= ``target`` (0.9) of the window's
GT frames are covered, and taps until every *reachable* GT frame (one with a detector candidate within tolerance) is
covered. ``unreachable`` = GT frames with no such candidate (detector recall ceiling).

Two policies for a rejected frame: ``design`` (a frame with a verdict is judged and gets no further card, as in design
2.2.5; it ends in the hand fallback) and ``requeue`` (the rejected candidate is forbidden but the frame stays open for
its next candidate, as in the KB simulation).

Everything is train-only: a clip whose ``splits.split_of_name`` is not ``train`` is refused. The oracle may be made noisy
(``error_rate``) to measure wrong-anchor damage. Synthetic helpers build noisy candidates from a GT trajectory for tests
and for proving the harness when real candidates are missing.
"""

from __future__ import annotations

import json
import math
import random
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

from footy_track.ball_eval.panel_score import BALL, load_human_sidecars
from footy_track.ball_mining import anchor_search as A  # noqa: N812
from footy_track.clip_id import canonical_clip_id

BASELINE_TAPS = 13  # KB simulation (footy-claude-autolabel-loop): Viterbi re-track, ~13 taps/clip to 90%
TARGET = 0.9
TOL_PX = 20.0
MAX_CANDS = 6  # top-1 + up to 5, as the job_suggest files
MIN_CONF = 0.1
POLICIES = ("design", "requeue")
ORDERS = ("design", "margin")  # design: (hop, margin, conf) as the generator ranks; margin: smallest Viterbi margin first (KB simulation)
FW, FH = A.FRAME_SIZE


class ReplayError(ValueError):
    pass


# ---------------------------------------------------------------------------
# Inputs: GT (the oracle) and candidates
# ---------------------------------------------------------------------------

GT = dict[int, tuple[float, float]]  # frame -> normalised ball centre


def require_train(clip: str) -> None:
    sp = A.clip_split(clip)
    if sp != "train":
        raise ReplayError(f"clip {clip!r} is split {sp!r}: the replay runs on train clips only")


def load_gt(marks_dirs: Iterable[Path], *, tol_px: float = TOL_PX) -> tuple[dict[str, GT], dict]:
    """Human ball GT per CANONICAL clip from the sidecars of ``marks_dirs`` (every ``<stem>.jsonl``; aliases of one clip
    are merged, first dir / stem wins on a frame). Only tier-1 ``labeller`` ball marks count (``load_human_sidecars``).
    Returns ``(gt, info)``; ``info`` counts alias conflicts (same frame, centres further apart than ``tol_px``)."""
    gt: dict[str, GT] = defaultdict(dict)
    conflicts = 0
    for md_ in marks_dirs:
        md = Path(md_)
        for p in sorted(md.glob("*.jsonl")):
            for (c, f), h in load_human_sidecars(md, [p.stem]).items():
                if h.state != BALL or not h.centres:
                    continue
                d = gt[canonical_clip_id(c)]
                if f in d:
                    conflicts += math.hypot((d[f][0] - h.centres[0][0]) * FW, (d[f][1] - h.centres[0][1]) * FH) > tol_px
                else:
                    d[f] = h.centres[0]
    return dict(gt), {"alias_conflicts": int(conflicts)}


def load_machine_candidates(clip: str, dirs: Iterable[Path]) -> A.CandidateSet | None:
    """Detector candidates of the canonical ``clip`` from the first of ``dirs`` that has a jsonl for it (any alias
    stem), conf >= 0.1, top ``MAX_CANDS`` per frame. ``None`` when no such file exists."""
    from footy_track.ball_mining.schema import read_machine_label_rows  # noqa: PLC0415

    for d in dirs:
        for p in sorted(Path(d).glob("*.jsonl")):
            if canonical_clip_id(p.stem) != clip:
                continue
            cs = A.CandidateSet("machine_labels")
            for r in read_machine_label_rows(p, min_conf=MIN_CONF):
                b = r["bbox"]
                cs.by_frame.setdefault((clip, int(r["frame_index"])), []).append(
                    A.Cand(b["x"] + b["w"] / 2, b["y"] + b["h"] / 2, b["w"], b["h"], float(r["confidence"]))
                )
            cs.by_frame = {k: A._sorted(v)[:MAX_CANDS] for k, v in cs.by_frame.items()}  # noqa: SLF001
            sha = None
            mf = Path(d) / "_manifest.json"
            if mf.is_file():
                try:
                    sha = str(json.loads(mf.read_text()).get("model_sha256_16", ""))[:8] or None
                except ValueError:
                    sha = None
            cs.detector = sha
            return cs
    return None


def _dist_px(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot((a[0] - b[0]) * FW, (a[1] - b[1]) * FH)


def reachable_frames(gt: GT, cands: A.CandidateSet, clip: str, tol_px: float = TOL_PX) -> set[int]:
    """GT frames with a detector candidate within tolerance (what a card can ever be right about)."""
    return {
        f for f, g in gt.items()
        if any(_dist_px((c.cx, c.cy), g) <= tol_px for c in cands.get(clip, f))
    }


# ---------------------------------------------------------------------------
# Synthetic clips (tests, harness proof when real candidates are missing)
# ---------------------------------------------------------------------------


def synthetic_clip(
    clip: str = "synthetic_clip",
    *,
    seed: int = 0,
    n_frames: int = 250,
    label_stride: int = 5,
    recall: float = 0.65,
    n_fp: int = 2,
    jitter_px: float = 3.0,
) -> tuple[GT, A.CandidateSet]:
    """A smooth ball path labelled every ``label_stride`` frames, plus noisy detector candidates on every frame: the true
    ball with probability ``recall`` (jittered, confidence 0.3-0.9), ``n_fp`` random false candidates per frame and one
    parked spare ball (a persistent false positive). Labelled SYNTHETIC: never a measurement of the real detector."""
    rng = random.Random(seed)
    x, y, vx, vy = 0.3, 0.5, 0.002, 0.0005
    path: dict[int, tuple[float, float]] = {}
    for f in range(n_frames):
        vx += rng.gauss(0, 0.0003)
        vy += rng.gauss(0, 0.0003)
        x = min(max(x + vx, 0.05), 0.95)
        y = min(max(y + vy, 0.1), 0.9)
        if x in (0.05, 0.95):
            vx = -vx
        if y in (0.1, 0.9):
            vy = -vy
        path[f] = (x, y)
    spare = (rng.uniform(0.1, 0.9), 0.93)
    cs = A.CandidateSet("synthetic", "synth000")
    for f, (bx, by) in path.items():
        cs_f = []
        if rng.random() < recall:
            cs_f.append(A.Cand(bx + rng.gauss(0, jitter_px / FW), by + rng.gauss(0, jitter_px / FH), 0.008, 0.014, rng.uniform(0.3, 0.9)))
        for _ in range(n_fp):
            cs_f.append(A.Cand(rng.uniform(0.05, 0.95), rng.uniform(0.1, 0.9), 0.008, 0.014, rng.uniform(0.1, 0.5)))
        cs_f.append(A.Cand(spare[0], spare[1], 0.008, 0.014, 0.6))
        cs.by_frame[(clip, f)] = A._sorted(cs_f)  # noqa: SLF001
    gt = {f: path[f] for f in range(0, n_frames, label_stride)}
    return gt, cs


# ---------------------------------------------------------------------------
# The simulator
# ---------------------------------------------------------------------------


@dataclass
class ReplayConfig:
    tol_px: float = TOL_PX
    target: float = TARGET
    policy: str = "design"
    order: str = "design"
    error_rate: float = 0.0  # oracle verdict flipped with this probability (0 = perfect)
    hop_cap: int = A.HOP_CAP
    window_frames: int = A.WINDOW_FRAMES
    max_taps: int = 2000
    bridge_stride: int = 5  # extra tracker frames every N frames of the window (0 = GT frames only)
    seed: int = 0


@dataclass
class WindowReplay:
    window_id: str
    n_gt: int
    n_reachable: int
    taps: int = 0  # all taps until BOTH milestones are met (or the window is exhausted)
    card_taps: int = 0
    seed_taps: int = 0
    hand_taps: int = 0
    taps_to_target: int | None = None  # None: never reached
    taps_to_all_reachable: int | None = None
    hand_to_target: int = 0
    wrong_anchors: int = 0  # ball verdicts given to a candidate that is not within tolerance of GT
    final_covered: int = 0
    unreachable_generator: int = 0  # frames the generator last reported unreachable
    curve: list[int] = field(default_factory=list)  # covered GT frames after each tap


def _margin_key(card: dict) -> tuple:
    m = card["margin"]
    return (math.inf if m is None else m, -card["rank_conf"], card["frame_index"])


def _covered(gt: GT, anchors: A.AnchorSet, cards: list[dict], clip: str, tol_px: float) -> set[int]:
    out = {f for (c, f), a in anchors.anchors.items() if c == clip and f in gt and _dist_px((a.cx, a.cy), gt[f]) <= tol_px}
    for c in cards:
        f = c["frame_index"]
        b = c["bbox"]
        if f in gt and f not in anchors.anchors and _dist_px((b["x"] + b["w"] / 2, b["y"] + b["h"] / 2), gt[f]) <= tol_px:
            out.add(f)
    return out


def simulate_window(  # noqa: PLR0915
    window_id: str, clip: str, start: int, gt: GT, cands: A.CandidateSet, cfg: ReplayConfig,
    *, rng: random.Random | None = None, bridge: Iterable[int] = (),
) -> WindowReplay:
    """Replay the loop on one window; ``gt`` holds exactly the window's selected frames (the ones scored and asked).

    ``bridge`` are extra sampled frames (no GT) fed to the tracker so it can link the sparse GT frames, as the real
    stride-5 sampled grid does; they are never asked and never scored (the oracle cannot judge them)."""
    rng = rng or random.Random(cfg.seed)
    if cfg.policy not in POLICIES or cfg.order not in ORDERS:
        raise ReplayError(f"policy must be one of {POLICIES}, order one of {ORDERS}")
    frames = sorted(gt)
    bridge = sorted(set(bridge) - set(gt))
    grid = sorted(set(frames) | set(bridge))
    reach = reachable_frames(gt, cands, clip, cfg.tol_px)
    out = WindowReplay(window_id, len(frames), len(reach))
    if not frames:
        return out
    need = math.ceil(cfg.target * len(frames) - 1e-9)
    anchors = A.AnchorSet()
    judged: set[A.Key] = {(clip, b) for b in bridge}
    gen = A.generator_version(cands.detector)
    covered: set[int] = set()
    cards: list[dict] = []
    res = None

    def regenerate() -> None:
        nonlocal cards, covered, res
        if anchors.anchors:
            res = A.generate_window(
                window_id, clip, start, grid, cands, anchorset=anchors, judged=frozenset(judged),
                job_id="replay", generator_ver=gen, teacher_version=None, run_id="replay", search_round=out.taps + 1, window_frames=cfg.window_frames,
                hop_cap=cfg.hop_cap,
            )
            cards = res.cards if cfg.order == "design" else sorted(res.cards, key=_margin_key)
            out.unreachable_generator = len(res.unreachable)
        else:
            cards = []
        covered = _covered(gt, anchors, cards, clip, cfg.tol_px)

    def verdict(f: int, cx: float, cy: float, w: float, h: float, *, detected: bool = True) -> None:
        truth = _dist_px((cx, cy), gt[f]) <= cfg.tol_px
        says_ball = truth != (rng.random() < cfg.error_rate)
        if says_ball:
            anchors.anchors[(clip, f)] = A.Anchor(clip, f, cx, cy, "verdict")
            judged.add((clip, f))
            out.wrong_anchors += not truth
        else:
            anchors.forbidden.setdefault((clip, f), []).append(A.Forbidden(clip, f, cx, cy, w, h))
            if cfg.policy == "design" or not detected:  # an interpolated pick has no candidate to forbid
                judged.add((clip, f))

    def seed_card() -> tuple[int, A.Cand] | None:
        """Top-1 detection among uncovered, open frames (not already forbidden / anchored)."""
        best = None
        for f in frames:
            if f in covered or (clip, f) in anchors.anchors or (clip, f) in judged:
                continue
            forb = anchors.forbidden.get((clip, f), [])
            for c in cands.get(clip, f):
                if any(math.hypot((c.cx - d.cx) * FW, (c.cy - d.cy) * FH) <= max(d.w * FW, d.h * FH) / 2 + A.FORBIDDEN_MARGIN_PX for d in forb):
                    continue
                if best is None or c.conf > best[1].conf:
                    best = (f, c)
        return best

    def milestones() -> None:
        if out.taps_to_target is None and len(covered) >= need:
            out.taps_to_target = out.taps
            out.hand_to_target = out.hand_taps
        if out.taps_to_all_reachable is None and reach <= covered:
            out.taps_to_all_reachable = out.taps

    regenerate()
    milestones()
    while out.taps < cfg.max_taps and (out.taps_to_target is None or out.taps_to_all_reachable is None):
        if cards:
            c = cards[0]
            b = c["bbox"]
            f = c["frame_index"]
            out.card_taps += 1
            verdict(f, b["x"] + b["w"] / 2, b["y"] + b["h"] / 2, b["w"], b["h"], detected=c["cand_source"] == "detected")
        elif (s := seed_card()) is not None:
            f, cd = s
            out.seed_taps += 1
            verdict(f, cd.cx, cd.cy, cd.w, cd.h)
        else:
            # hand fallback: reachable frames first, unreachable ones only while the target is still open
            open_ = [f for f in frames if f not in covered]
            pick = [f for f in open_ if f in reach] or ([f for f in open_ if out.taps_to_target is None])
            if not pick:
                break
            f = pick[0]
            out.hand_taps += 1
            anchors.anchors[(clip, f)] = A.Anchor(clip, f, gt[f][0], gt[f][1], "human")
            judged.add((clip, f))
        out.taps += 1
        regenerate()
        out.curve.append(len(covered))
        milestones()
    out.final_covered = len(covered)
    return out


@dataclass
class ClipReplay:
    clip: str
    n_gt: int
    n_reachable: int
    windows: list[WindowReplay]
    candidate_source: str

    def _sum(self, attr: str) -> int | None:
        vals = [getattr(w, attr) for w in self.windows]
        return None if any(v is None for v in vals) else sum(vals)

    @property
    def taps_to_target(self) -> int | None:
        return self._sum("taps_to_target")

    @property
    def taps_to_all_reachable(self) -> int | None:
        return self._sum("taps_to_all_reachable")

    def row(self) -> dict:
        s = lambda a: sum(getattr(w, a) for w in self.windows)  # noqa: E731
        return {
            "clip": self.clip, "source": self.candidate_source, "gt_frames": self.n_gt, "windows": len(self.windows),
            "reachable": self.n_reachable, "unreachable": self.n_gt - self.n_reachable,
            "taps_to_90": self.taps_to_target, "hand_in_90": s("hand_to_target"),
            "taps_to_100_reachable": self.taps_to_all_reachable, "total_taps": s("taps"),
            "card_taps": s("card_taps"), "seed_taps": s("seed_taps"), "hand_taps": s("hand_taps"),
            "wrong_anchors": s("wrong_anchors"), "generator_unreachable_last": s("unreachable_generator"),
            "naive_taps_90": math.ceil(TARGET * self.n_gt - 1e-9),  # one tap per frame, no propagation
            "vs_baseline": None if self.taps_to_target is None else self.taps_to_target - BASELINE_TAPS,
        }


def simulate_clip(clip: str, gt: GT, cands: A.CandidateSet, cfg: ReplayConfig, *, source: str | None = None, synthetic: bool = False) -> ClipReplay:
    """Replay every window of one clip. Train clips only (``synthetic`` clips skip the split register)."""
    if not synthetic:
        require_train(clip)
    rng = random.Random(f"{cfg.seed}:{clip}")
    wins = []
    for wid, (c, start, fs) in A.split_windows([(clip, f) for f in gt], cfg.window_frames).items():
        bridge = (
            [f for f in range(start, start + cfg.window_frames, cfg.bridge_stride) if cands.get(clip, f)] if cfg.bridge_stride > 0 else []
        )
        wins.append(simulate_window(wid, c, start, {f: gt[f] for f in fs}, cands, cfg, rng=rng, bridge=bridge))
    return ClipReplay(clip, len(gt), reachable_n(gt, cands, clip, cfg.tol_px), wins, source or cands.source)


def reachable_n(gt: GT, cands: A.CandidateSet, clip: str, tol_px: float) -> int:
    return len(reachable_frames(gt, cands, clip, tol_px))


def summarize(rows: list[dict]) -> dict:
    """Aggregate of per-clip rows: clips that reached 90%, mean / median taps vs the 13-tap baseline."""
    ok = [r["taps_to_90"] for r in rows if r["taps_to_90"] is not None]
    srt = sorted(ok)
    med = None if not srt else (srt[len(srt) // 2] if len(srt) % 2 else (srt[len(srt) // 2 - 1] + srt[len(srt) // 2]) / 2)
    mean = None if not ok else sum(ok) / len(ok)
    return {
        "clips": len(rows), "reached_90": len(ok), "mean_taps_to_90": mean, "median_taps_to_90": med,
        "baseline": BASELINE_TAPS, "beats_baseline": bool(ok) and len(ok) == len(rows) and mean is not None and mean < BASELINE_TAPS,
        "wrong_anchors": sum(r["wrong_anchors"] for r in rows),
    }


def format_table(rows: list[dict]) -> str:
    cols = ["clip", "gt_frames", "unreachable", "taps_to_90", "hand_in_90", "taps_to_100_reachable", "total_taps", "wrong_anchors", "naive_taps_90", "vs_baseline"]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in rows:
        lines.append("| " + " | ".join("-" if r[c] is None else str(r[c]) for c in cols) + " |")
    return "\n".join(lines)
