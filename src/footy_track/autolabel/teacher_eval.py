"""Score a teacher version against GOLD labels on held-out matches only (design 4b, 6b).

Gold = ``frame_label_state`` methods ``hand_series`` / ``hand_frame`` / ``verdict`` (never
``offline_audited`` silver) plus ``hand_label`` boxes. Scoring refuses (``ScoringRefused``) any frame
whose split is not ``eval_heldout_match`` (train frames are teacher *training* data; scoring on them
is contamination) and any non-gold method. Metrics are computed per match and reported per match;
the headline is the plain mean over matches (every match weighs the same), never a pooled count.

Ball (pixels): a gold-present frame is a TP at radius ``r`` when the prediction is within ``r`` px;
recall = TP / gold-present frames, precision = TP / predicted frames (both over gold-labelled ball
frames; gold ``absent`` frames count against precision). Players: greedy IoU >= 0.5 matching,
split by overlap bucket of the box (max IoU with another box of the SAME set: gold boxes for recall,
predicted boxes for precision): ``isolated`` < 0.1 <= ``touching`` < 0.3 <= ``overlapping``.

Results go to ``eval_result`` as ``ResultRow`` s with ``model_name="teacher"``,
``model_version=<teacher_version>``, ``split="eval_heldout_match"``; per-match metrics are named
``<metric>/<match_key>`` and the match-averaged headline ``<metric>``.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field

from footy_track import splits
from footy_track.datasets.results import ResultRow

GOLD_METHODS = frozenset({"hand_series", "hand_frame", "verdict"})
BALL_RADII_PX = (10.0, 25.0)
PLAYER_IOU = 0.5
BUCKETS = (("isolated", 0.1), ("touching", 0.3), ("overlapping", math.inf))
Box = tuple[float, float, float, float]  # x, y, w, h in px
Point = tuple[float, float]


class ScoringRefused(ValueError):  # noqa: N818
    """The frames are not a legal gold / held-out scoring set."""


@dataclass
class GoldFrame:
    """One gold-labelled frame and the teacher's output on it (pixel coordinates)."""

    match_key: str
    game_id: str
    frame_index: int
    ball_method: str | None = None  # gold method of the ball state (None = ball not gold)
    gold_ball: Point | None = None  # None with ball_method set = ball absent
    pred_ball: Point | None = None
    player_method: str | None = None  # gold method of the player state (complete only)
    gold_players: list[Box] = field(default_factory=list)
    pred_players: list[Box] = field(default_factory=list)


def _iou(a: Box, b: Box) -> float:
    ix = max(0.0, min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = a[2] * a[3] + b[2] * b[3] - inter
    return inter / union if union > 0 else 0.0


def bucket_of(box: Box, others: Sequence[Box]) -> str:
    m = max((_iou(box, o) for o in others if o is not box), default=0.0)
    return next(name for name, hi in BUCKETS if m < hi)


def _ratio(num: int, den: int) -> float | None:
    return None if den == 0 else num / den


def validate(frames: Iterable[GoldFrame], split_of: Callable[[str, str], str]) -> list[GoldFrame]:
    """Raise ``ScoringRefused`` unless every frame is gold and from a held-out match."""
    out = list(frames)
    for f in out:
        for m in (f.ball_method, f.player_method):
            if m is not None and m not in GOLD_METHODS:
                raise ScoringRefused(f"{f.game_id}:{f.frame_index}: method {m!r} is not gold")
        sp = split_of(f.match_key, f.game_id)
        if sp != splits.EVAL_HELDOUT:
            raise ScoringRefused(
                f"{f.game_id}:{f.frame_index}: split {sp!r}; gold scoring needs {splits.EVAL_HELDOUT!r}"
            )
    return out


def score_match(frames: Sequence[GoldFrame]) -> dict[str, float | int | None]:
    """Metrics of ONE match's frames (flat dict, ``None`` where the denominator is 0)."""
    res: dict[str, float | int | None] = {}
    ball = [f for f in frames if f.ball_method is not None]
    n_gold = sum(f.gold_ball is not None for f in ball)
    n_pred = sum(f.pred_ball is not None for f in ball)
    res["n_ball_frames"] = len(ball)
    for r in BALL_RADII_PX:
        tp = sum(
            1 for f in ball
            if f.gold_ball is not None and f.pred_ball is not None
            and math.dist(f.gold_ball, f.pred_ball) <= r
        )  # fmt: skip
        res[f"ball_recall_{int(r)}px"] = _ratio(tp, n_gold)
        res[f"ball_precision_{int(r)}px"] = _ratio(tp, n_pred)
    tp_g: dict[str, int] = defaultdict(int)  # by gold-box bucket
    n_g: dict[str, int] = defaultdict(int)
    tp_p: dict[str, int] = defaultdict(int)  # by predicted-box bucket
    n_p: dict[str, int] = defaultdict(int)
    players = [f for f in frames if f.player_method is not None]
    for f in players:
        pairs = sorted(
            ((_iou(g, p), gi, pi) for gi, g in enumerate(f.gold_players) for pi, p in enumerate(f.pred_players)),
            key=lambda t: (-t[0], t[1], t[2]),
        )
        used_g: set[int] = set()
        used_p: set[int] = set()
        for v, gi, pi in pairs:
            if v < PLAYER_IOU:
                break
            if gi not in used_g and pi not in used_p:
                used_g.add(gi)
                used_p.add(pi)
        for gi, g in enumerate(f.gold_players):
            b = bucket_of(g, f.gold_players)
            n_g[b] += 1
            tp_g[b] += gi in used_g
        for pi, p in enumerate(f.pred_players):
            b = bucket_of(p, f.pred_players)
            n_p[b] += 1
            tp_p[b] += pi in used_p
    res["n_player_frames"] = len(players)
    for name, _ in BUCKETS:
        res[f"person_recall_{name}"] = _ratio(tp_g[name], n_g[name])
        res[f"person_precision_{name}"] = _ratio(tp_p[name], n_p[name])
    res["person_recall_all"] = _ratio(sum(tp_g.values()), sum(n_g.values()))
    res["person_precision_all"] = _ratio(sum(tp_p.values()), sum(n_p.values()))
    return res


@dataclass
class TeacherScore:
    teacher_version: str
    per_match: dict[str, dict[str, float | int | None]]
    headline: dict[str, float | None]  # plain mean over matches that have the metric


def score_teacher(
    teacher_version: str,
    frames: Iterable[GoldFrame],
    split_of: Callable[[str, str], str] | None = None,
) -> TeacherScore:
    """Score ``frames`` (gold, held-out only) per match; headline = mean over matches."""
    split_of = split_of or (lambda mk, gid: str(splits.split_of(mk, gid)))
    ok = validate(frames, split_of)
    by_match: dict[str, list[GoldFrame]] = defaultdict(list)
    for f in ok:
        by_match[f.match_key].append(f)
    per = {mk: score_match(fs) for mk, fs in sorted(by_match.items())}
    metrics = sorted({k for m in per.values() for k in m if not k.startswith("n_")})
    headline: dict[str, float | None] = {}
    for k in metrics:
        vals = [float(m[k]) for m in per.values() if m.get(k) is not None]
        headline[k] = sum(vals) / len(vals) if vals else None
    return TeacherScore(teacher_version, per, headline)


def to_result_rows(
    score: TeacherScore, *, dataset: str = "teacher_gold", dataset_version: int = 1, task: str = "tracking"
) -> list[ResultRow]:
    """``eval_result`` rows: headline ``<metric>`` plus per-match ``<metric>/<match_key>``."""

    def row(metric: str, value: float | None, n: int | None) -> ResultRow:
        return ResultRow(
            task=task, dataset=dataset, dataset_version=dataset_version, split=splits.EVAL_HELDOUT,
            model_name="teacher", model_version=score.teacher_version, metric=metric, value=value,
            n_items=n, contaminated=False, contamination_note="gold labels on held-out matches only",
        )

    rows = [row(k, v, len(score.per_match)) for k, v in sorted(score.headline.items())]
    for mk, m in score.per_match.items():
        for k, v in sorted(m.items()):
            if k.startswith("n_"):
                continue
            n = m["n_ball_frames"] if k.startswith("ball") else m["n_player_frames"]
            rows.append(row(f"{k}/{mk}", None if v is None else float(v), int(n or 0)))
    return rows


# --------------------------------------------------------------------------------------------
# store loader
# --------------------------------------------------------------------------------------------
BALL_LABELS = frozenset({"ball", "in_play_ball", "out_of_play_ball"})


def load_gold_frames(
    store, game_id: str, *, pred_run_prefix: str, width: int, height: int, match_key: str | None = None
) -> list[GoldFrame]:
    """Gold frames of ``game_id`` from the feature store (``store.query`` -> DataFrame).

    Gold boxes: ``source='hand_label'`` detections on frames whose state method is gold. Teacher
    predictions: detection rows whose ``run_id`` starts with ``pred_run_prefix`` and whose
    ``source != 'hand_label'``. Boxes are normalised in the store, scaled to px here.
    """
    mk = match_key or splits.match_key_of(game_id) or game_id
    st = store.query(
        "SELECT frame_index, cls_group, state, method FROM frame_label_state "
        "WHERE game_id = ? AND method IN ('hand_series','hand_frame','verdict') "
        "AND state IN ('complete','absent')", [game_id],
    )  # fmt: skip
    gold = store.query(
        "SELECT frame_index, label, bbox_x, bbox_y, bbox_w, bbox_h FROM detection "
        "WHERE game_id = ? AND source = 'hand_label'", [game_id],
    )  # fmt: skip
    pred = store.query(
        "SELECT frame_index, label, bbox_x, bbox_y, bbox_w, bbox_h FROM detection "
        "WHERE game_id = ? AND source != 'hand_label' AND starts_with(run_id, ?)",
        [game_id, pred_run_prefix],
    )  # fmt: skip

    def boxes(df) -> dict[int, list[tuple[str, Box]]]:  # noqa: ANN001
        out: dict[int, list[tuple[str, Box]]] = defaultdict(list)
        for fi, lab, x, y, w, h in df.itertuples(index=False):
            out[int(fi)].append((str(lab), (x * width, y * height, w * width, h * height)))
        return out

    g, p = boxes(gold), boxes(pred)

    def centre(b: Box) -> Point:
        return (b[0] + b[2] / 2, b[1] + b[3] / 2)

    frames: dict[int, GoldFrame] = {}
    for fi, grp, state, method in st.itertuples(index=False):
        f = frames.setdefault(int(fi), GoldFrame(mk, game_id, int(fi)))
        if grp == "ball":
            f.ball_method = method
            gb = [b for lab, b in g.get(f.frame_index, []) if lab in BALL_LABELS]
            pb = [b for lab, b in p.get(f.frame_index, []) if lab in BALL_LABELS]
            f.gold_ball = centre(gb[0]) if gb and state == "complete" else None
            f.pred_ball = centre(pb[0]) if pb else None
        elif state == "complete":
            f.player_method = method
            f.gold_players = [b for lab, b in g.get(f.frame_index, []) if lab not in BALL_LABELS]
            f.pred_players = [b for lab, b in p.get(f.frame_index, []) if lab not in BALL_LABELS]
    return [frames[k] for k in sorted(frames)]


def gold_by_game(frames: Iterable[GoldFrame]) -> Mapping[str, int]:
    out: dict[str, int] = defaultdict(int)
    for f in frames:
        out[f.game_id] += 1
    return dict(out)
