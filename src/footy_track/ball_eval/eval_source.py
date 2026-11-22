"""Candidate source of the eval card flow (design ``ball_anchor_iteration.md`` task 10; George's decision 2026-11-22).

The eval family shows Ball Check cards pre-filled with a candidate ball. WHICH model supplies those candidates is a decision,
made by one explicit, testable rule and recorded on every card and search document:

    The offline combined framework (the teacher, ``offline_vN``) is the source WHILE it scores better than the best
    student / online ball model on the existing human gold; otherwise the best student. Ties go to the teacher.

"Scores better" = recall within :data:`RECALL_RADIUS_PX` (20 px) of a human ball centre on the SAME gold frames, where a frame
counts as recalled when ANY candidate of the source lies inside the radius (the source's job is to put the ball among the
candidates, not to rank it first). Gold = human ball frames (tier-1 ``labeller`` marks, ``panel_score.Human``) of matches that
are not scored here; the caller chooses them, the selector only compares counts and refuses a comparison that is not
like-for-like (different gold size, or no gold at all).

The model being scored (``scored_model``) is never used silently: when two sources are equally good the one that is NOT the
scored model wins, and when the winner is the scored model anyway the selection says so (``selected_is_scored_model``) and
carries a warning. Scores of every version are kept in the selection record (``scores``) so reports read per version.

Outputs: :func:`select_source` -> ``eval-candidate-source/v1`` record; :func:`candidate_source_block` -> the small block that
rides on cards / search docs; :func:`write_suggest_file` -> a ``job-suggest/v1`` file for the eval job holding the chosen
source's candidates (``candidate_source`` block included) which ``anchor_search.load_candidates`` reads unchanged.

CLI (``python -m footy_track.ball_eval.eval_source select``) compares sources given as ``kind:id:version:path.jsonl`` where the
jsonl rows are ``{clip, frame, x, y, w, h, score}`` (normalised, top-left boxes) against a gold marks directory.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from footy_track.ball_eval import job_suggest as J  # noqa: N812
from footy_track.ball_eval import panel as P  # noqa: N812
from footy_track.ball_eval.panel_score import BALL, Human, load_human_sidecars
from footy_track.clip_id import canonical_clip_id

SCHEMA = "eval-candidate-source/v1"
RULE = "teacher_if_gold_recall>=best_student"
RECALL_RADIUS_PX = 20.0
TEACHER, STUDENT = "teacher", "student"
MIN_CONF = 0.1  # same floor as job_suggest candidates
MAX_CANDS = 6

Key = tuple[str, int]


class SourceSelectionError(ValueError):
    """The comparison cannot be made (no gold, unlike gold sets, no usable source). Never resolved by guessing."""


@dataclass
class Source:
    """One candidate source: its identity, version and per-frame candidate boxes (normalised ``{x, y, w, h, conf}``)."""

    kind: str  # teacher | student
    id: str  # model id / sha8 (what ``job_suggest`` calls the detector)
    version: str  # teacher: offline_vN; student: the model version tag
    candidates: Mapping[Key, Sequence[Mapping]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.kind not in (TEACHER, STUDENT):
            raise SourceSelectionError(f"source kind must be teacher or student, got {self.kind!r}")

    @property
    def ref(self) -> str:
        return f"{self.kind}:{self.id}:{self.version}"


@dataclass(frozen=True)
class SourceScore:
    kind: str
    id: str
    version: str
    n_gold: int
    n_hit: int

    @property
    def recall(self) -> float | None:
        return self.n_hit / self.n_gold if self.n_gold else None

    def to_json(self) -> dict:
        r = self.recall
        return {"kind": self.kind, "id": self.id, "version": self.version, "n_gold": self.n_gold, "n_hit": self.n_hit,
                "recall": None if r is None else round(r, 6)}  # fmt: skip


def gold_ball_frames(human: Mapping[Key, Human]) -> dict[Key, list[tuple[float, float]]]:
    """Human ball frames: ``{(clip, frame): [normalised centres]}`` (no_ball / not_broadcast frames are not gold balls)."""
    return {(canonical_clip_id(c), f): list(h.centres) for (c, f), h in human.items() if h.state == BALL and h.centres}


def score_source(
    source: Source, gold: Mapping[Key, Sequence[tuple[float, float]]], size: tuple[int, int], r_px: float = RECALL_RADIUS_PX
) -> SourceScore:
    """Gold frames that have a candidate of ``source`` within ``r_px`` of a human centre (frames the source has no row for miss)."""
    cands = {(canonical_clip_id(c), int(f)): v for (c, f), v in source.candidates.items()}
    hit = 0
    for k, centres in gold.items():
        for b in cands.get(k, ()):
            bx, by = (b["x"] + b["w"] / 2) * size[0], (b["y"] + b["h"] / 2) * size[1]
            if any(math.hypot(bx - cx * size[0], by - cy * size[1]) <= r_px for cx, cy in centres):
                hit += 1
                break
    return SourceScore(source.kind, source.id, source.version, len(gold), hit)


def _version_key(v: str) -> tuple:
    """``offline_v12`` > ``offline_v3``: numeric parts compare as numbers, the rest as text."""
    import re  # noqa: PLC0415

    return tuple((0, int(t), "") if t.isdigit() else (1, 0, t) for t in re.findall(r"\d+|\D+", v))


def _best(scores: list[SourceScore], scored_model: str | None) -> SourceScore:
    """Most gold hits; ties: not the scored model, then the newest version, then id (deterministic)."""
    return max(scores, key=lambda s: (s.n_hit, s.id != scored_model, _version_key(s.version), s.id))


def select_source(
    scores: Iterable[SourceScore | Mapping], *, scored_model: str | None = None, r_px: float = RECALL_RADIUS_PX
) -> dict:
    """The explicit comparison. Returns the ``eval-candidate-source/v1`` record.

    * best teacher = most gold hits among ``kind == teacher``; best student likewise among ``kind == student``;
    * the teacher wins when ``teacher.n_hit >= best_student.n_hit`` (ties -> teacher), else the best student;
    * only one kind present -> its best; none -> :class:`SourceSelectionError`; no gold, or sources scored on different gold
      sizes -> :class:`SourceSelectionError` (the rule is only meaningful like-for-like).

    Equal hit counts are equal recall because every source is scored on the same gold frames."""
    rows = [s if isinstance(s, SourceScore) else SourceScore(**{k: s[k] for k in ("kind", "id", "version", "n_gold", "n_hit")}) for s in scores]
    if not rows:
        raise SourceSelectionError("no candidate source to choose from")
    golds = {s.n_gold for s in rows}
    if len(golds) != 1:
        raise SourceSelectionError(f"sources were scored on different gold sets ({sorted(golds)}); cannot compare")
    n_gold = golds.pop()
    if n_gold == 0:
        raise SourceSelectionError("no human gold ball frames: the teacher vs student comparison cannot be made")
    teachers = [s for s in rows if s.kind == TEACHER]
    students = [s for s in rows if s.kind == STUDENT]
    bt = _best(teachers, scored_model) if teachers else None
    bs = _best(students, scored_model) if students else None
    if bt and bs:
        chosen, why = (bt, "teacher_recall>=best_student_recall") if bt.n_hit >= bs.n_hit else (bs, "best_student_recall>teacher_recall")
        if bt.n_hit == bs.n_hit:
            why = "tie_goes_to_teacher"
    else:
        chosen, why = (bt or bs), "only_one_kind_available"
    runner = bs if chosen is bt else bt
    is_scored = scored_model is not None and chosen.id == scored_model
    rec = {
        "schema": SCHEMA, "rule": RULE, "radius_px": r_px, "n_gold": n_gold, "reason": why,
        "chosen": chosen.to_json(), "runner_up": runner.to_json() if runner else None,
        "scores": sorted((s.to_json() for s in rows), key=lambda d: (d["kind"], d["version"], d["id"])),
        "scored_model": scored_model, "selected_is_scored_model": is_scored,
    }  # fmt: skip
    if is_scored:
        rec["warning"] = (
            f"the chosen candidate source {chosen.kind}:{chosen.id}:{chosen.version} IS the model being scored: "
            "its own suggestions pre-fill the cards it is evaluated on; no source that scores better was available"
        )
    return rec


def candidate_source_block(selection: Mapping) -> dict:
    """The small block recorded on every eval card and search document: kind, id, version, ``teacher_version`` (teacher source)
    or ``model_version`` (student source), the rule and the gold recall that decided it."""
    c = selection["chosen"]
    blk = {
        "kind": c["kind"], "id": c["id"], "version": c["version"], "rule": selection["rule"], "radius_px": selection["radius_px"],
        "gold_recall": c["recall"], "n_gold": selection["n_gold"], "reason": selection["reason"],
        "selected_is_scored_model": bool(selection.get("selected_is_scored_model")),
    }  # fmt: skip
    blk["teacher_version" if c["kind"] == TEACHER else "model_version"] = c["version"]
    return blk


def choose(
    sources: Sequence[Source],
    human: Mapping[Key, Human],
    size: tuple[int, int],
    *,
    scored_model: str | None = None,
    r_px: float = RECALL_RADIUS_PX,
) -> tuple[dict, Source]:
    """Score every source on the gold balls of ``human`` and select. Returns ``(selection record, chosen Source)``."""
    gold = gold_ball_frames(human)
    sel = select_source([score_source(s, gold, size, r_px) for s in sources], scored_model=scored_model, r_px=r_px)
    c = sel["chosen"]
    return sel, next(s for s in sources if (s.kind, s.id, s.version) == (c["kind"], c["id"], c["version"]))


# ---------------------------------------------------------------------------
# the suggest file of the eval job
# ---------------------------------------------------------------------------


def suggest_doc(job_id: str, source: Source, frames: Iterable[Key], selection: Mapping, size: tuple[int, int]) -> dict:
    """``job-suggest/v1`` document of the chosen source for the eval job's frames (``boxes`` has a key for every frame, null =
    no candidate), plus the ``candidate_source`` block and the whole selection record."""
    boxes: dict[str, dict | None] = {}
    have = {(canonical_clip_id(c), int(f)): v for (c, f), v in source.candidates.items()}
    for clip, f in sorted(set(frames)):
        rows = sorted((r for r in have.get((canonical_clip_id(clip), f), ()) if float(r.get("conf", 0.0)) >= MIN_CONF),
                      key=lambda r: -float(r.get("conf", 0.0)))[:MAX_CANDS]  # fmt: skip
        norm = [{"x": r["x"], "y": r["y"], "w": r["w"], "h": r["h"], "conf": float(r.get("conf", 0.0)), "tid": int(r.get("tid") or 0)}
                for r in rows]  # fmt: skip
        boxes[f"{clip}:{f}"] = {**norm[0], "cands": norm} if norm else None
    return {
        "schema": J.SCHEMA, "job_id": job_id, "created_at": P.now_iso(), "size": list(size),
        "detector": {"sha8": source.id, "kind": source.kind, "version": source.version},
        "candidate_source": candidate_source_block(selection), "selection": dict(selection),
        "pre_labelled": [], "n_frames": len(boxes), "n_with_suggestion": sum(1 for v in boxes.values() if v), "boxes": boxes,
    }  # fmt: skip


def write_suggest_file(job_id: str, source: Source, frames: Iterable[Key], selection: Mapping, size: tuple[int, int], root: Path | None = None) -> Path:
    """Write the eval job's suggest file once per chosen source (a different source writes a different file, the newest is read);
    an existing file of the same source is kept (the document is a function of the inputs)."""
    out = J.suggest_path(job_id, source.id, root)
    out.parent.mkdir(parents=True, exist_ok=True)
    if not out.is_file():
        P._atomic_write(out, suggest_doc(job_id, source, frames, selection, size), exclusive=True, sort_keys=False)  # noqa: SLF001
    return out


# ---------------------------------------------------------------------------
# loading + CLI
# ---------------------------------------------------------------------------


def load_source_jsonl(ref: str) -> Source:
    """``kind:id:version:path`` -> a Source from a jsonl of ``{clip, frame, x, y, w, h, score}`` (normalised, top-left boxes)."""
    kind, sid, ver, path = ref.split(":", 3)
    cands: dict[Key, list[dict]] = {}
    for raw in Path(path).read_text().splitlines():
        if not raw.strip():
            continue
        d = json.loads(raw)
        if d.get("x") is None:
            continue
        cands.setdefault((canonical_clip_id(d["clip"]), int(d["frame"])), []).append(
            {"x": d["x"], "y": d["y"], "w": d["w"], "h": d["h"], "conf": float(d.get("score", d.get("conf", 1.0))), "tid": d.get("tid", 0)}
        )
    return Source(kind, sid, ver, cands)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="eval_source", description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("select", help="score the sources on the human gold and print the selection record")
    s.add_argument("--source", action="append", required=True, help="kind:id:version:path.jsonl (repeat)")
    s.add_argument("--gold-marks", type=Path, required=True, help="marks dir of the gold clips")
    s.add_argument("--gold-clips", nargs="+", required=True, help="sidecar stems of the gold clips")
    s.add_argument("--scored-model", default=None)
    s.add_argument("--size", default="1920x1080")
    s.add_argument("--radius", type=float, default=RECALL_RADIUS_PX)
    a = ap.parse_args(argv)
    w, h = (int(v) for v in a.size.split("x"))
    sel, _ = choose([load_source_jsonl(r) for r in a.source], load_human_sidecars(a.gold_marks, a.gold_clips), (w, h),
                    scored_model=a.scored_model, r_px=a.radius)  # fmt: skip
    print(json.dumps(sel, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
