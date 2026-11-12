"""Claude verify, method v2: one yes/no question per detector candidate.

Pure logic, no GPU, no network (see ``docs/design/claude_autolabel_loop.md``, section 8).
Differences from v1 (``verify_pilot``):

* The crop is centred on each candidate, never on the tracker's expected position.
* One question per candidate: "is the box on the match ball?" (``ball`` / ``not_ball`` /
  ``unsure``). No grid cells.
* Per frame, the candidates asked about are: the tracker's chosen detection, the top
  few other cached candidates by confidence, and crop-and-re-detect candidates near the
  tracker path where no cached candidate lies within ``NEAR_PX``.
* Scoring is per candidate and per frame. Frames with no candidate within R of the
  human mark cannot be answered by this method and are reported apart.

Geometry is in native frame pixels. The human marks are used only by the scorer.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass

from footy_track.autolabel.prompts_v2 import VERDICTS_V2
from footy_track.autolabel.verify_pilot import Cand, crop_origin, dist, nearest

PATCH_PX = 128  # native region shown zoomed
VIEW_PX = 512  # zoomed size (4x)
CONTEXT_PX = 384  # native region shown at 1x beside it
REDETECT_CROP_PX = 320
NEAR_PX = 25.0
DEDUPE_PX = 6.0
TOP_OTHERS = 2
MAX_REDETECT = 2
RADII_PX = (25.0, 10.0)
SOURCES = ("tracker", "top", "redetect")
CONF_BANDS = ((0.0, 0.5), (0.5, 0.8), (0.8, 0.95), (0.95, 1.01))


@dataclass(frozen=True)
class Candidate:
    """One question: a candidate box in native pixels and where it came from."""

    cx: float
    cy: float
    w: float
    h: float
    conf: float
    source: str  # tracker | top | redetect


def frame_key(clip: str, frame: int) -> str:
    return f"{clip}|{frame}"


def question_id(clip: str, frame: int, k: int) -> str:
    return f"{clip}_{frame:06d}_{k}"


# ---------------------------------------------------------------------- candidates


def select_candidates(
    cached: Sequence[Cand],
    tracker_pt: tuple[float, float] | None,
    redetect: Sequence[Cand] = (),
    *,
    top_others: int = TOP_OTHERS,
    max_redetect: int = MAX_REDETECT,
    near_px: float = NEAR_PX,
    dedupe: float = DEDUPE_PX,
) -> list[Candidate]:
    """The candidates to ask about on one frame.

    ``tracker_pt`` is the centre of the detection the bidirectional tracker chose (None
    when it chose none or only interpolated). It selects the cached candidate nearest
    to it (within ``dedupe``). Then the ``top_others`` highest-confidence remaining
    cached candidates. Then up to ``max_redetect`` re-detect candidates that have no
    cached candidate (any, not only the chosen ones) within ``near_px``. The human marks
    are not an input.
    """
    out: list[Candidate] = []
    pool = sorted(cached, key=lambda c: -c.conf)
    if tracker_pt is not None and pool:
        best = min(pool, key=lambda c: dist((c.cx, c.cy), tracker_pt))
        if dist((best.cx, best.cy), tracker_pt) <= dedupe:
            out.append(
                Candidate(best.cx, best.cy, best.w, best.h, best.conf, "tracker")
            )
            pool.remove(best)
    n_top = 0
    for c in pool:
        if n_top >= top_others:
            break
        if any(dist((c.cx, c.cy), (o.cx, o.cy)) <= dedupe for o in out):
            continue
        out.append(Candidate(c.cx, c.cy, c.w, c.h, c.conf, "top"))
        n_top += 1
    n_re = 0
    for r in sorted(redetect, key=lambda c: -c.conf):
        if n_re >= max_redetect:
            break
        if any(dist((r.cx, r.cy), (c.cx, c.cy)) <= near_px for c in cached):
            continue
        if any(dist((r.cx, r.cy), (o.cx, o.cy)) <= dedupe for o in out):
            continue
        out.append(Candidate(r.cx, r.cy, r.w, r.h, r.conf, "redetect"))
        n_re += 1
    return out


def patch_origin(c: Candidate, size: int = PATCH_PX) -> tuple[int, int]:
    """Top-left of the ``size`` patch centred on the candidate (may leave the frame)."""
    return round(c.cx - size / 2), round(c.cy - size / 2)


def redetect_origin(
    pt: tuple[float, float], frame_size: tuple[int, int]
) -> tuple[int, int]:
    """Top-left of the 320 px re-detect crop around a tracker position, clamped."""
    return crop_origin(pt[0], pt[1], frame_size, REDETECT_CROP_PX)


def tracker_anchor(
    frame: int,
    points: dict[int, tuple[float, float]],
    window: int = 12,
) -> tuple[float, float] | None:
    """Where to run crop-and-re-detect: the tracker point at ``frame`` or the nearest in
    ``window`` frames (earlier wins a tie). None when the tracker has nothing nearby, in
    which case no re-detect is run (a frame-centre crop would be a blind guess)."""
    if frame in points:
        return points[frame]
    for d in range(1, window + 1):
        for f in (frame - d, frame + d):
            if f in points:
                return points[f]
    return None


# ------------------------------------------------------------------------- answers


def normalise_answer(raw: object) -> dict:
    """Validate one answer against the v2 schema; anything off becomes ``invalid``."""
    bad = {"verdict": "invalid", "confidence": None, "reason": ""}
    if not isinstance(raw, dict):
        return bad
    v, conf = raw.get("verdict"), raw.get("confidence")
    if v not in VERDICTS_V2 or (
        conf is not None
        and (isinstance(conf, bool) or not isinstance(conf, int | float))
    ):
        return bad
    return {
        "verdict": v,
        "confidence": None if conf is None else max(0.0, min(1.0, float(conf))),
        "reason": str(raw.get("reason", ""))[:200],
    }


# ------------------------------------------------------------------------- scoring


def _near(c: dict, gts: Sequence[tuple[float, float]], radius: float) -> bool:
    d = nearest((c["cx"], c["cy"]), gts)
    return d is not None and d <= radius


def score_candidates(
    questions: Sequence[dict],
    truth: dict[str, dict],
    answers: dict[str, dict],
    radius: float = NEAR_PX,
) -> list[dict]:
    """One row per answered candidate question."""
    rows = []
    for q in questions:
        a = answers.get(q["id"])
        if a is None:
            continue
        t = truth[q["fkey"]]
        gts = [tuple(g) for g in t["gt"]]
        near = t["kind"] == "ball" and _near(q["cand"], gts, radius)
        rows.append(
            {
                "id": q["id"],
                "fkey": q["fkey"],
                "split": q["split"],
                "kind": t["kind"],
                "source": q["cand"]["source"],
                "det_conf": q["cand"]["conf"],
                "near": near,
                "verdict": a["verdict"],
                "confidence": a["confidence"],
            }
        )
    return rows


def score_frames(
    frames: Sequence[dict], rows: Sequence[dict], questions: Sequence[dict]
) -> list[dict]:
    """One row per frame whose every candidate question has an answer.

    A ball frame is *answerable* when some candidate lies within R of a human mark; it
    is *correct* when exactly the near candidates are answered ``ball``. A no-ball frame
    is correct when no candidate is answered ``ball``.
    """
    n_asked: dict[str, int] = defaultdict(int)
    for q in questions:
        n_asked[q["fkey"]] += 1
    by_frame: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_frame[r["fkey"]].append(r)
    out = []
    for f in frames:
        rs = by_frame.get(f["fkey"], [])
        if len(rs) != n_asked.get(f["fkey"], 0):
            continue  # some questions unanswered: frame not scored
        any_near = any(r["near"] for r in rs)
        said_ball = {r["id"] for r in rs if r["verdict"] == "ball"}
        near_ids = {r["id"] for r in rs if r["near"]}
        if f["kind"] == "ball":
            answerable = any_near
            correct = answerable and said_ball == near_ids
        else:
            answerable = True
            correct = not said_ball
        out.append(
            {
                "fkey": f["fkey"],
                "split": f["split"],
                "kind": f["kind"],
                "n_cands": len(rs),
                "answerable": answerable,
                "correct": bool(correct),
                "has_questions": bool(rs),
                "false_ball_frame": bool(said_ball - near_ids),
                "missed_near": bool(near_ids - said_ball),
            }
        )
    return out


def _rate(n: int | float, d: int | float) -> float | None:
    return n / d if d else None


def summarise(
    frames: Sequence[dict], rows: Sequence[dict], frame_rows: Sequence[dict]
) -> dict:
    """Aggregate. ``rows`` are candidate rows, ``frame_rows`` from :func:`score_frames`."""
    c = defaultdict(int)
    for r in rows:
        c["cands"] += 1
        v = r["verdict"]
        c[f"v_{v}"] += 1
        if r["near"]:
            c["near"] += 1
            c["near_ball"] += v == "ball"
            c["near_not_ball"] += v == "not_ball"
            c["near_unsure"] += v == "unsure"
        else:
            c["far"] += 1
            c["far_ball"] += v == "ball"
            c["far_unsure"] += v == "unsure"
            if r["kind"] == "no_ball":
                c["nb_cands"] += 1
                c["nb_ball"] += v == "ball"
            else:
                c["farb_cands"] += 1
                c["farb_ball"] += v == "ball"
    tp, fp = c["near_ball"], c["far_ball"]
    ball_fr = [f for f in frame_rows if f["kind"] == "ball"]
    ans_fr = [f for f in ball_fr if f["answerable"]]
    nb_fr = [f for f in frame_rows if f["kind"] == "no_ball" and f["has_questions"]]
    n_ball_frames = sum(1 for f in frames if f["kind"] == "ball")
    unans = [f for f in ball_fr if not f["answerable"]]
    unans_ids = {f["fkey"] for f in unans}
    unans_ball = sum(
        1 for r in rows if r["fkey"] in unans_ids and r["verdict"] == "ball"
    )
    bands = {}
    for lo, hi in CONF_BANDS:
        b = [
            r
            for r in rows
            if r["verdict"] == "ball"
            and r["confidence"] is not None
            and lo <= r["confidence"] < hi
        ]
        bands[f"{lo}-{min(hi, 1)}"] = [len(b), sum(1 for r in b if r["near"])]
    by_source = {}
    for s in SOURCES:
        sr = [r for r in rows if r["source"] == s]
        sb = [r for r in sr if r["verdict"] == "ball"]
        by_source[s] = {
            "cands": len(sr),
            "near": sum(r["near"] for r in sr),
            "said_ball": len(sb),
            "said_ball_near": sum(r["near"] for r in sb),
            "precision": _rate(sum(r["near"] for r in sb), len(sb)),
        }
    return {
        "frames_scored": len(frame_rows),
        "ball_frames": n_ball_frames,
        "ball_frames_scored": len(ball_fr),
        "ball_frames_answerable": len(ans_fr),
        "ball_frames_no_cand_within_R": len(unans),
        "ball_frames_no_cand_share": _rate(len(unans), len(ball_fr)),
        "unanswerable_false_ball_cands": unans_ball,
        "no_ball_frames_with_cands": len(nb_fr),
        "no_ball_frames_no_cands": sum(
            1 for f in frame_rows if f["kind"] == "no_ball" and not f["has_questions"]
        ),
        "frame_agreement": _rate(sum(f["correct"] for f in ans_fr), len(ans_fr)),
        "frame_agreement_no_ball": _rate(sum(f["correct"] for f in nb_fr), len(nb_fr)),
        "frame_agreement_all": _rate(
            sum(f["correct"] for f in [*ans_fr, *nb_fr]), len(ans_fr) + len(nb_fr)
        ),
        "false_ball_frames_on_ball_frames": sum(f["false_ball_frame"] for f in ans_fr),
        "cands": c["cands"],
        "near_cands": c["near"],
        "far_cands": c["far"],
        "ball_precision": _rate(tp, tp + fp),
        "ball_recall": _rate(tp, c["near"]),
        "false_not_ball_rate": _rate(c["near_not_ball"], c["near"]),
        "near_unsure_rate": _rate(c["near_unsure"], c["near"]),
        "false_ball_rate": _rate(fp, c["far"]),
        "false_ball_on_far_cands_ball_frames": _rate(c["farb_ball"], c["farb_cands"]),
        "false_ball_on_no_ball_cands": _rate(c["nb_ball"], c["nb_cands"]),
        "false_ball_no_ball_frame_rate": _rate(
            sum(f["false_ball_frame"] for f in nb_fr), len(nb_fr)
        ),
        "unsure_rate": _rate(c["v_unsure"], c["cands"]),
        "invalid": c["v_invalid"],
        "by_source": by_source,
        "ball_verdict_by_conf_band": bands,
    }


def ball_precision_by_det_conf(rows: Iterable[dict]) -> dict[str, list[int]]:
    """For comparison with the detector alone: [n, n near] by detector confidence band."""
    out = {}
    for lo, hi in ((0.0, 0.3), (0.3, 0.6), (0.6, 0.9), (0.9, 1.01)):
        b = [r for r in rows if lo <= r["det_conf"] < hi]
        out[f"{lo}-{min(hi, 1)}"] = [len(b), sum(1 for r in b if r["near"])]
    return out


def cand_to_dict(c: Candidate) -> dict:
    return asdict(c)


# --------------------------------------------------------------------------- recheck
#
# A person re-checks the frames where Claude and the human marks disagree (the marks
# have no second human pass). Each disagreement frame becomes Ball Check "list" cards:
# one per human mark ("is the GT mark the in-play match ball?") and one per disagreeing
# Claude box (same question). Verdicts map onto the existing ones; a spare or crowd ball
# is simply ``not_ball`` for the question asked (the in-play ball), nothing new.

RECHECK_SOURCE = "list"
GT_BOX_PX = 12.0  # the marks store centres only here; drawn as a small square


def find_disagreements(
    questions: Sequence[dict],
    truth: dict[str, dict],
    answers: dict[str, dict],
    radius: float = NEAR_PX,
) -> list[dict]:
    """Candidates where Claude and the human marks disagree.

    ``false_not_ball``: a candidate within R of a mark answered ``not_ball`` or
    ``unsure`` (the ``verdict`` column says which). ``false_ball``: a candidate farther
    than R from every mark answered ``ball``, on a ball frame; ``false_ball_no_ball_frame``
    the same on a frame the human marked as having no ball.
    """
    qs = {q["id"]: q for q in questions}
    out = []
    for r in score_candidates(questions, truth, answers, radius):
        if r["near"] and r["verdict"] != "ball":
            kind = "false_not_ball"
        elif not r["near"] and r["verdict"] == "ball":
            kind = "false_ball" if r["kind"] == "ball" else "false_ball_no_ball_frame"
        else:
            continue
        q = qs[r["id"]]
        gts = [tuple(g) for g in truth[r["fkey"]]["gt"]]
        near = nearest((q["cand"]["cx"], q["cand"]["cy"]), gts)
        out.append(
            {
                "type": kind,
                "qid": r["id"],
                "fkey": r["fkey"],
                "clip": q["clip"],
                "frame": q["frame"],
                "split": q["split"],
                "source": r["source"],
                "verdict": r["verdict"],
                "claude_conf": r["confidence"],
                "reason": answers[r["id"]].get("reason", ""),
                "det_conf": r["det_conf"],
                "cand_cx": q["cand"]["cx"],
                "cand_cy": q["cand"]["cy"],
                "cand_w": q["cand"]["w"],
                "cand_h": q["cand"]["h"],
                "gt": [list(g) for g in gts],
                "dist_to_nearest_gt": None if near is None else round(near, 1),
            }
        )
    out.sort(key=lambda d: (d["clip"], d["frame"], d["qid"]))
    return out


def _box(cx: float, cy: float, w: float, h: float, size: tuple[int, int]) -> dict:
    fw, fh = size
    x = max(0.0, min(1.0, (cx - w / 2) / fw))
    y = max(0.0, min(1.0, (cy - h / 2) / fh))
    return {
        "x": round(x, 5),
        "y": round(y, 5),
        "w": round(min(w / fw, 1 - x), 5),
        "h": round(min(h / fh, 1 - y), 5),
    }


def build_review_items(
    disagreements: Sequence[dict],
    frame_sizes: dict[str, tuple[int, int]],
    video_stems: dict[str, str],
) -> list[dict]:
    """Ball Check cards for ``disagreements``, grouped by frame, human mark first.

    ``video_stems`` maps the run's clip id to the stem the labeller serves the video
    under (``clip`` of a card must resolve to a video). Boxes are normalised.
    """
    by_frame: dict[str, list[dict]] = defaultdict(list)
    for d in disagreements:
        by_frame[d["fkey"]].append(d)
    items: list[dict] = []
    for fkey in sorted(
        by_frame, key=lambda k: (by_frame[k][0]["clip"], by_frame[k][0]["frame"])
    ):
        ds = by_frame[fkey]
        clip, frame = ds[0]["clip"], ds[0]["frame"]
        size = frame_sizes[clip]
        stem = video_stems[clip]
        types = ", ".join(sorted({d["type"] for d in ds}))
        first = ds[0]
        for gi, g in enumerate(first["gt"]):
            others = sorted(
                ds, key=lambda d: dist((d["cand_cx"], d["cand_cy"]), tuple(g))
            )
            o = others[0]
            items.append(
                {
                    "id": f"{fkey}|gt{gi}",
                    "role": "gt",
                    "fkey": fkey,
                    "gt_index": gi,
                    "clip": stem,
                    "run_clip": clip,
                    "frame_size": list(size),
                    "frame_index": frame,
                    "bbox": _box(g[0], g[1], GT_BOX_PX, GT_BOX_PX, size),
                    "other_bbox": _box(
                        o["cand_cx"], o["cand_cy"], o["cand_w"], o["cand_h"], size
                    ),
                    "label": "in_play_ball",
                    "provenance": "gt mark",
                    "confidence": None,
                    "type": types,
                    "note": (
                        f"GT MARK (yellow). Is it the in-play match ball? Magenta = Claude's "
                        f"box ({o['verdict']}, conf {o['claude_conf']}). [{types}]"
                    ),
                }
            )
        for d in ds:
            near = min(
                (tuple(g) for g in d["gt"]),
                key=lambda g: dist((d["cand_cx"], d["cand_cy"]), g),
                default=None,
            )
            items.append(
                {
                    "id": d["qid"],
                    "role": "claude",
                    "fkey": fkey,
                    "qid": d["qid"],
                    "clip": stem,
                    "run_clip": clip,
                    "frame_size": list(size),
                    "frame_index": frame,
                    "bbox": _box(
                        d["cand_cx"], d["cand_cy"], d["cand_w"], d["cand_h"], size
                    ),
                    "other_bbox": None
                    if near is None
                    else _box(near[0], near[1], GT_BOX_PX, GT_BOX_PX, size),
                    "label": "in_play_ball",
                    "provenance": f"claude {d['source']}",
                    "confidence": None,
                    "type": d["type"],
                    "note": (
                        f"CLAUDE'S BOX (yellow), Claude said {d['verdict']} "
                        f"(conf {d['claude_conf']}). Is it the in-play match ball? "
                        f"Spare or crowd ball = Not a ball. Magenta = GT mark. [{d['type']}]"
                    ),
                }
            )
    return items


def apply_decisions(
    truth: dict[str, dict],
    questions: Sequence[dict],
    items: Sequence[dict],
    decisions: dict[str, dict],
) -> tuple[dict[str, dict], dict]:
    """Truth after a person's verdicts on the review cards.

    ``decisions`` maps card ``id`` to the latest live Ball Check record. Rules, per frame:

    * a GT card ``not_ball`` removes that mark; ``corrected`` moves it to the dragged
      box; ``ball`` / ``box_off`` / ``unsure`` / no verdict keep it;
    * a Claude card ``ball`` adds its box (a ``corrected`` adds the dragged box) as a
      mark; ``not_ball`` / ``unsure`` / none add nothing, so a ``ball`` answer from Claude
      on that box scores as a false ``ball``. A spare or crowd ball is ``not_ball``;
    * a frame left with no mark is a no-ball frame; a frame with a mark is a ball frame.

    Frames without decisions are untouched. Returns (new truth, change counts).
    """
    new = {
        k: {"kind": v["kind"], "gt": [list(g) for g in v["gt"]]}
        for k, v in truth.items()
    }
    by_frame: dict[str, list[dict]] = defaultdict(list)
    for it in items:
        by_frame[it["fkey"]].append(it)
    qcand = {q["id"]: q["cand"] for q in questions}
    stats: dict[str, int] = defaultdict(int)
    for fkey, its in by_frame.items():
        decided = [(it, decisions[it["id"]]) for it in its if it["id"] in decisions]
        stats["cards"] += len(its)
        stats["cards_decided"] += len(decided)
        if not decided:
            continue
        gts = [list(g) for g in truth[fkey]["gt"]]
        removed: set[int] = set()
        added: list[list[float]] = []
        for it, rec in decided:
            v = rec["verdict"]
            stats[f"{it['role']}_{v}"] += 1
            cb = rec.get("corrected_bbox")
            pos = None
            if v == "corrected" and isinstance(cb, dict):
                fw, fh = it["frame_size"]
                pos = [(cb["x"] + cb["w"] / 2) * fw, (cb["y"] + cb["h"] / 2) * fh]
            if it["role"] == "gt":
                if v == "not_ball":
                    removed.add(it["gt_index"])
                elif v == "corrected" and pos is not None:
                    removed.add(it["gt_index"])
                    added.append(pos)
            elif v == "ball":
                c = qcand[it["qid"]]
                added.append([c["cx"], c["cy"]])
            elif v == "corrected" and pos is not None:
                added.append(pos)
        kept = [g for i, g in enumerate(gts) if i not in removed] + added
        kind_before = new[fkey]["kind"]
        new[fkey] = {"kind": "ball" if kept else "no_ball", "gt": kept}
        if new[fkey] != {"kind": kind_before, "gt": gts}:
            stats["frames_changed"] += 1
            if kind_before != new[fkey]["kind"]:
                stats[f"frames_{kind_before}_to_{new[fkey]['kind']}"] += 1
    return new, dict(stats)
