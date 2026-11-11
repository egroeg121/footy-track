"""Calibration pilot for the Claude verify step: pure logic, no GPU, no network.

Pipeline (see ``docs/design/claude_autolabel_loop.md``, section 8):

1. :func:`sample_frames`   pick the human-marked frames to ask about.
2. :func:`expected_position`  where to crop, from the tracker only. The GT position
   is never used to place a crop or a candidate; it is only used by the scorer.
3. :func:`crop_origin`, :func:`merge_candidates`  crop geometry and candidate list.
4. ``render.py`` draws the question image; ``claude_cli.py`` asks Claude.
5. :func:`score_question`, :func:`summarise`  compare the answers with the human marks.

Geometry is in native frame pixels throughout. The crop is ``CROP_PX`` square, split
into a ``GRID`` x ``GRID`` grid of cells (``CELL_PX`` = 40 px at native resolution);
cells are named ``A1``..``H8`` (letter = column, digit = row, from the top left).
"""

from __future__ import annotations

import math
import random
import re
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass

CROP_PX = 320
UPSCALE = 3  # crop upscale for the re-detect pass
VIEW_PX = 960  # size of the crop in the question image
GRID = 8
CELL_PX = CROP_PX / GRID
DEDUPE_PX = 6.0
MAX_CANDIDATES = 6
REDETECT_CONF = 0.05
NEAR_FRAME_WINDOW = 12
RADII_PX = (25.0, 10.0)
CELL_RE = re.compile(r"^[A-H][1-8]$")
VERDICTS = ("candidate", "cell", "none", "unsure")


@dataclass(frozen=True)
class Cand:
    """One candidate in native pixels (centre and size)."""

    cx: float
    cy: float
    w: float
    h: float
    conf: float
    source: str  # "orig" (cached detector run) or "redetect" (crop re-run)


@dataclass(frozen=True)
class Sample:
    clip: str
    frame: int
    split: str  # held_out | seen
    kind: str  # ball | no_ball
    stratum: str  # held_out | seen_cand | seen_nocand | seen_no_ball


def dist(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def nearest(p: tuple[float, float], pts: Iterable[tuple[float, float]]) -> float | None:
    ds = [dist(p, q) for q in pts]
    return min(ds) if ds else None


# ------------------------------------------------------------------------ sampling


def sample_frames(
    clips: dict[str, dict],
    split: dict[str, str],
    *,
    seed: int = 7,
    seen_with_cand: int = 50,
    seen_without_cand: int = 50,
    seen_no_ball: int = 20,
    radius: float = 25.0,
) -> list[Sample]:
    """Choose the frames to ask about, in a fixed shuffled order.

    ``clips[clip]`` = ``{"ball": {frame: [(x, y) px]}, "no_ball": set(frames),
    "cands": {frame: [(x, y) px centres]}}``. Only frames present in ``cands`` (the
    detector cache) can be asked. Held-out clips: every human frame. Seen clips:
    a stratified sample (half with a candidate within ``radius`` of the human mark,
    half without) plus no-ball frames. The GT is used here only to stratify.
    """
    rng = random.Random(seed)
    out: list[Sample] = []
    seen_c: list[Sample] = []
    seen_n: list[Sample] = []
    seen_nb: list[Sample] = []
    for clip in sorted(clips):
        sp = split.get(clip)
        if sp not in ("held_out", "seen"):
            continue
        c = clips[clip]
        have = c["cands"]
        for f in sorted(c["ball"]):
            if f not in have:
                continue
            if sp == "held_out":
                out.append(Sample(clip, f, sp, "ball", "held_out"))
                continue
            near = any(
                (d := nearest(g, have[f])) is not None and d <= radius
                for g in c["ball"][f]
            )
            (seen_c if near else seen_n).append(
                Sample(clip, f, sp, "ball", "seen_cand" if near else "seen_nocand")
            )
        for f in sorted(c["no_ball"]):
            if f not in have:
                continue
            s = Sample(
                clip,
                f,
                sp,
                "no_ball",
                "held_out" if sp == "held_out" else "seen_no_ball",
            )
            (out if sp == "held_out" else seen_nb).append(s)
    for pool_, n in (
        (seen_c, seen_with_cand),
        (seen_n, seen_without_cand),
        (seen_nb, seen_no_ball),
    ):
        out += rng.sample(pool_, min(n, len(pool_)))
    rng.shuffle(out)
    return out


# ------------------------------------------------------------ expected position, crop


def expected_position(
    frame: int,
    points: dict[int, tuple[float, float]],
    frame_size: tuple[int, int],
    *,
    window: int = NEAR_FRAME_WINDOW,
) -> tuple[float, float, str]:
    """Crop centre in px from tracker positions only: exact, nearest within window, centre.

    ``points`` maps frame -> (x, y) px for frames where the tracker has a position.
    """
    if frame in points:
        return (*points[frame], "tracker")
    best = None
    for d in range(1, window + 1):
        for f in (frame - d, frame + d):  # earlier frame wins a tie
            if f in points:
                best = f
                break
        if best is not None:
            break
    if best is not None:
        return (*points[best], f"near{best - frame:+d}")
    return frame_size[0] / 2, frame_size[1] / 2, "centre"


def crop_origin(
    cx: float, cy: float, frame_size: tuple[int, int], size: int = CROP_PX
) -> tuple[int, int]:
    """Top-left of a ``size`` square around (cx, cy), clamped inside the frame."""
    x0 = round(min(max(cx - size / 2, 0), frame_size[0] - size))
    y0 = round(min(max(cy - size / 2, 0), frame_size[1] - size))
    return int(x0), int(y0)


def in_crop(p: tuple[float, float], x0: int, y0: int, size: int = CROP_PX) -> bool:
    return x0 <= p[0] < x0 + size and y0 <= p[1] < y0 + size


def merge_candidates(
    orig: Sequence[Cand],
    redetect: Sequence[Cand],
    *,
    dedupe: float = DEDUPE_PX,
    top: int = MAX_CANDIDATES,
) -> list[Cand]:
    """Union of cached and re-detect candidates; originals win duplicates; top by conf."""
    merged: list[Cand] = []
    for c in sorted([*orig, *redetect], key=lambda c: (-c.conf, c.source != "orig")):
        dup = next(
            (m for m in merged if dist((m.cx, m.cy), (c.cx, c.cy)) <= dedupe), None
        )
        if dup is None:
            merged.append(c)
        elif dup.source != "orig" and c.source == "orig":
            merged[merged.index(dup)] = c  # keep provenance of the original run
    return sorted(merged, key=lambda c: -c.conf)[:top]


def cell_label(col: int, row: int) -> str:
    return f"{'ABCDEFGH'[col]}{row + 1}"


def cell_centre(label: str, x0: int, y0: int) -> tuple[float, float]:
    col, row = "ABCDEFGH".index(label[0]), int(label[1]) - 1
    return x0 + (col + 0.5) * CELL_PX, y0 + (row + 0.5) * CELL_PX


# ------------------------------------------------------------------------- scoring


def normalise_answer(raw: object, n_cands: int) -> dict:
    """Validate one answer against the schema; anything off becomes ``invalid``."""
    bad = {
        "verdict": "invalid",
        "candidate": None,
        "cell": None,
        "confidence": None,
        "reason": "",
    }
    if not isinstance(raw, dict):
        return bad
    v = raw.get("verdict")
    cand, cell, conf = raw.get("candidate"), raw.get("cell"), raw.get("confidence")
    ok = v in VERDICTS
    if v == "candidate":
        ok = (
            isinstance(cand, int)
            and not isinstance(cand, bool)
            and 1 <= cand <= n_cands
        )
    elif v == "cell":
        ok = isinstance(cell, str) and bool(CELL_RE.match(cell))
    if not ok or (conf is not None and not isinstance(conf, int | float)):
        return bad
    return {
        "verdict": v,
        "candidate": cand if v == "candidate" else None,
        "cell": cell if v == "cell" else None,
        "confidence": None if conf is None else max(0.0, min(1.0, float(conf))),
        "reason": str(raw.get("reason", ""))[:200],
    }


def score_question(q: dict, truth: dict, answer: dict, radius: float) -> dict:
    """Outcome of one answer. ``q`` has crop origin and candidates; ``truth`` the GT."""
    x0, y0 = q["crop"]
    cands = [Cand(**c) for c in q["candidates"]]
    gts = [tuple(g) for g in truth["gt"]]
    v = answer["verdict"]
    near = [
        i
        for i, c in enumerate(cands, 1)
        if (d := nearest((c.cx, c.cy), gts)) is not None and d <= radius
    ]
    near_redetect_only = bool(near) and all(
        cands[i - 1].source == "redetect" for i in near
    )
    point = None
    if v == "candidate":
        c = cands[answer["candidate"] - 1]
        point = (c.cx, c.cy)
    elif v == "cell":
        point = cell_centre(answer["cell"], x0, y0)
    if truth["kind"] == "ball":
        d = nearest(point, gts) if point else None
        outcome = {
            "candidate": "correct" if d is not None and d <= radius else "wrong_pick",
            "cell": "correct" if d is not None and d <= radius else "wrong_pick",
            "none": "false_none",
            "unsure": "unsure",
            "invalid": "invalid",
        }[v]
    else:
        outcome = {"none": "correct", "unsure": "unsure", "invalid": "invalid"}.get(
            v, "false_ball"
        )
    return {
        "outcome": outcome,
        "cand_within": bool(near),
        "redetect_only": near_redetect_only,
        "picked_near": outcome == "correct" and v == "candidate",
        "gt_in_crop": any(in_crop(g, x0, y0) for g in gts),
    }


def _rate(n: int, d: int) -> float | None:
    return n / d if d else None


def summarise(rows: list[dict]) -> dict:
    """Aggregate scored rows (each: split, kind, outcome, cand_within, redetect_only, ...)."""
    cnt: dict[str, int] = defaultdict(int)
    for r in rows:
        cnt["n"] += 1
        cnt[r["outcome"]] += 1
        cnt[f"kind_{r['kind']}"] += 1
        if r["kind"] == "ball":
            cnt["ball_cand_within" if r["cand_within"] else "ball_no_cand"] += 1
            if r["cand_within"]:
                cnt["cw_correct"] += r["outcome"] == "correct"
                cnt["cw_false_none"] += r["outcome"] == "false_none"
            else:
                cnt["nc_correct"] += r["outcome"] == "correct"
                cnt["nc_cell_correct"] += (
                    r["outcome"] == "correct" and r.get("verdict") == "cell"
                )
                cnt["nc_none"] += r["outcome"] == "false_none"
            cnt["gt_outside_crop"] += not r["gt_in_crop"]
            cnt["cell_answers"] += r.get("verdict") == "cell"
            cnt["cell_correct"] += (
                r.get("verdict") == "cell" and r["outcome"] == "correct"
            )
            # "none" is the right answer for the crop when the human mark lies outside it
            cnt["none_ok_outside"] += (
                r["outcome"] == "false_none" and not r["gt_in_crop"]
            )
            if r["gt_in_crop"]:
                cnt["in_crop_ball"] += 1
                cnt["in_crop_correct"] += r["outcome"] == "correct"
                cnt["in_crop_false_none"] += r["outcome"] == "false_none"
            cnt["redetect_only"] += r["redetect_only"]
            cnt["redetect_only_picked"] += r["redetect_only"] and r["picked_near"]
    n = cnt["n"]
    ball, nb = cnt["kind_ball"], cnt["kind_no_ball"]
    return {
        "n": n,
        "ball_frames": ball,
        "no_ball_frames": nb,
        "agreement": _rate(cnt["correct"], n),
        "agreement_cand_available": _rate(cnt["cw_correct"], cnt["ball_cand_within"]),
        "agreement_no_cand": _rate(cnt["nc_correct"], cnt["ball_no_cand"]),
        "ball_cand_within": cnt["ball_cand_within"],
        "ball_no_cand": cnt["ball_no_cand"],
        "false_none_rate": _rate(cnt["false_none"], ball),
        "false_none_cand_available": _rate(
            cnt["cw_false_none"], cnt["ball_cand_within"]
        ),
        "false_ball_rate": _rate(cnt["false_ball"] + cnt["wrong_pick"], n),
        "false_ball_on_no_ball": _rate(cnt["false_ball"], nb),
        "wrong_pick_rate_on_ball": _rate(cnt["wrong_pick"], ball),
        "unsure_rate": _rate(cnt["unsure"], n),
        "invalid": cnt["invalid"],
        "gt_outside_crop": cnt["gt_outside_crop"],
        "agreement_adjusted": _rate(cnt["correct"] + cnt["none_ok_outside"], n),
        "agreement_gt_in_crop": _rate(cnt["in_crop_correct"], cnt["in_crop_ball"]),
        "false_none_gt_in_crop": _rate(cnt["in_crop_false_none"], cnt["in_crop_ball"]),
        "cell_answers": cnt["cell_answers"],
        "cell_correct": cnt["cell_correct"],
        "redetect_only": cnt["redetect_only"],
        "redetect_only_picked": cnt["redetect_only_picked"],
        "counts": {
            k: v
            for k, v in cnt.items()
            if k
            in (
                "correct",
                "wrong_pick",
                "false_none",
                "false_ball",
                "unsure",
                "invalid",
            )
        },
    }


def cand_to_dict(c: Cand) -> dict:
    return asdict(c)
