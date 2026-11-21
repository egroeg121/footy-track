"""Ball anchor iteration (design ``docs/design/ball_anchor_iteration.md``): a confirmed ball anchors the search for the
next candidate cards. This module grows task by task; this part holds the *inputs* only.

1. **Split guard.** One job = one split (``splits.split_of``). :func:`job_split` names it and refuses a job that spans
   several; :func:`check_append_split` is what ``labeller.job_edit.append_items`` calls so an appended card can never
   bring a frame of another split into the job. Eval (heldout / in-match) and train frames are never mixed and no
   label is ever copied across jobs.
2. **Job scope.** :func:`job_scope` fixes the job's selected frame set F (canonical clip id, frame) and its split.
   Every loader below returns only frames of F, so nothing it reads or returns can belong to another job or split.
3. **Candidates.** One uniform :class:`CandidateSet` whether the per-frame candidates come from a ``job_suggest`` file
   (:func:`candidates_from_suggest_doc`) or a pool parquet (:func:`candidates_from_pool`); :func:`load_candidates`
   picks one.
4. **Anchors.** :func:`load_anchors`: human ball labels of F (sidecar marks, same reader as ``job_suggest.job_human`` /
   ``suggestions.candidates_for_job``) and live Ball Check verdicts of F (``ball_mining.clips.read_live_verdicts``,
   the reader ``round`` / ``refit`` use: last line wins, tombstones honoured). ``ball`` / ``corrected`` become anchors,
   ``not_ball`` / ``box_off`` become forbidden discs, ``unsure`` is neither.

All coordinates are normalised (0..1 of the frame); candidate / anchor positions are box **centres**. Keys are
``(canonical clip id, frame)``.
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

from footy_track import splits
from footy_track.ball_eval import job_suggest as J  # noqa: N812
from footy_track.ball_eval import panel_suggest as PS  # noqa: N812
from footy_track.ball_eval.panel_score import BALL
from footy_track.ball_mining import clips as clips_mod
from footy_track.clip_id import canonical_clip_id
from footy_track.labeller import datasets

Key = tuple[str, int]  # (canonical clip id, frame)

ANCHOR_VERDICTS = ("ball", "corrected")
FORBIDDEN_VERDICTS = ("not_ball", "box_off")


# ---------------------------------------------------------------------------
# 1. Split guard
# ---------------------------------------------------------------------------


class SplitError(ValueError):
    """A job (or an append to it) would span more than one split, or the split cannot be determined."""


def clip_split(clip: str) -> str:
    """``splits.split_of`` of one clip name; fails closed when the register is missing or invalid."""
    try:
        return splits.split_of_name(clip)
    except splits.RegisterError as e:
        raise SplitError(f"cannot determine the split of clip {clip!r}: {e}") from e


def _clips_by_split(clip_names: Iterable[str]) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for c in sorted({canonical_clip_id(c) for c in clip_names}):
        out.setdefault(clip_split(c), []).append(c)
    return out


def _describe(by_split: dict[str, list[str]]) -> str:
    return ", ".join(f"{s}: {cs[:2]}{'...' if len(cs) > 2 else ''}" for s, cs in sorted(by_split.items()))


def _item_clips(items: Iterable[dict]) -> list[str]:
    return [it["clip"] for it in items if it.get("clip")]


def job_split(job: dict) -> str | None:
    """The one split of the job's clips (``None`` for a job with no clips). ``SplitError`` when they span several."""
    by_split = _clips_by_split(_item_clips(job.get("items") or []))
    if len(by_split) > 1:
        raise SplitError(f"job {job.get('id')!r} spans more than one split ({_describe(by_split)})")
    return next(iter(by_split), None)


def check_append_split(job: dict, new_items: Iterable[dict]) -> str | None:
    """Raise :class:`SplitError` unless ``new_items`` are all in the job's own split. Returns that split.

    A job that is already mixed (legacy) accepts no appends at all, an empty job takes the split of the new items,
    and the new items must be uniform among themselves."""
    have = _clips_by_split(_item_clips(job.get("items") or []))
    new = _clips_by_split(_item_clips(new_items))
    if len(have) > 1:
        raise SplitError(f"job {job.get('id')!r} already spans more than one split ({_describe(have)}); nothing may be appended")
    if len(new) > 1:
        raise SplitError(f"appended items span more than one split ({_describe(new)})")
    if have and new and next(iter(have)) != next(iter(new)):
        raise SplitError(
            f"job {job.get('id')!r} is {next(iter(have))!r}; refusing to append {next(iter(new))!r} items ({_describe(new)})"
        )
    return next(iter(have or new), None)


# ---------------------------------------------------------------------------
# 2. Job scope: the selected frame set F and the one split
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class JobScope:
    job_id: str
    kind: str
    dataset: str | None
    split: str | None
    frames: frozenset[Key]

    @property
    def clips(self) -> frozenset[str]:
        return frozenset(c for c, _ in self.frames)


def job_scope(job: dict) -> JobScope:
    """F of one job (``job_suggest.job_frames``, canonical clip ids) and its split; ``SplitError`` on a mixed job."""
    split = job_split(job)
    frames = frozenset((canonical_clip_id(c), f) for c, fl in J.job_frames(job).items() for f in fl)
    return JobScope(str(job.get("id")), str(job.get("kind")), job.get("dataset"), split, frames)


# ---------------------------------------------------------------------------
# 3. Candidates (job_suggest cands or pool nodes, one interface)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Cand:
    """One candidate ball position: box centre and size (normalised), confidence, offline tracklet id (0 = none)."""

    cx: float
    cy: float
    w: float
    h: float
    conf: float
    tid: int = 0


@dataclass
class CandidateSet:
    """Candidates per frame of one job, best confidence first. ``source`` is ``job_suggest`` or ``pool``."""

    source: str
    detector: str | None = None
    by_frame: dict[Key, list[Cand]] = field(default_factory=dict)

    def get(self, clip: str, frame: int) -> list[Cand]:
        return self.by_frame.get((canonical_clip_id(clip), int(frame)), [])


def _sorted(cs: list[Cand]) -> list[Cand]:
    return sorted(cs, key=lambda c: -c.conf)


def candidates_from_suggest_doc(doc: dict, scope: JobScope) -> CandidateSet:
    """Candidates of a ``job-suggest/v1`` document restricted to the scope's frames. A document written for another
    job is refused (a job never reads another job's file)."""
    if doc.get("job_id") != scope.job_id:
        raise SplitError(f"suggest file belongs to job {doc.get('job_id')!r}, not {scope.job_id!r}")
    out = CandidateSet("job_suggest", (doc.get("detector") or {}).get("sha8"))
    for key, entry in (doc.get("boxes") or {}).items():
        if not entry:
            continue
        clip, _, frame = key.rpartition(":")
        k = (canonical_clip_id(clip), int(frame))
        if k not in scope.frames:
            continue
        rows = entry.get("cands") or [entry]
        out.by_frame[k] = _sorted([
            Cand(r["x"] + r["w"] / 2, r["y"] + r["h"] / 2, r["w"], r["h"], float(r.get("conf", 0.0)), int(r.get("tid") or 0))
            for r in rows
        ])
    return out


def candidates_from_pool(df, scope: JobScope) -> CandidateSet:
    """Candidates of a pool table (``ball_mining.schema``) restricted to the scope's frames (``p_real`` is the
    confidence, ``track_id`` the tracklet id)."""
    out = CandidateSet("pool")
    shas = set()
    for r in df.itertuples(index=False):
        k = (canonical_clip_id(r.clip), int(r.frame))
        if k not in scope.frames:
            continue
        tid = getattr(r, "track_id", None)
        tid = 0 if tid is None or (isinstance(tid, float) and math.isnan(tid)) else int(tid)
        out.by_frame.setdefault(k, []).append(Cand(float(r.x), float(r.y), float(r.w), float(r.h), float(r.p_real), tid))
        shas.add(str(r.detector_sha))
    out.by_frame = {k: _sorted(v) for k, v in out.by_frame.items()}
    out.detector = next(iter(shas)) if len(shas) == 1 else None
    return out


def load_candidates(job: dict, *, root: Path | None = None, pool_path: Path | None = None) -> CandidateSet:
    """The job's candidates: the pool parquet at ``pool_path`` when given, else the newest ``job_suggest`` file of this
    job (empty ``job_suggest`` set when none exists yet). Only frames of the job's own set and split are returned."""
    scope = job_scope(job)
    if pool_path is not None:
        from footy_track.ball_mining import schema  # noqa: PLC0415

        return candidates_from_pool(schema.read_pool(pool_path), scope)
    f = J.find_suggest_file(scope.job_id, root)
    if f is None:
        return CandidateSet("job_suggest")
    return candidates_from_suggest_doc(json.loads(f.read_text()), scope)


# ---------------------------------------------------------------------------
# 4. Anchors and forbidden positions
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Anchor:
    clip: str
    frame: int
    cx: float
    cy: float
    source: str  # human (sidecar label) | verdict


@dataclass(frozen=True)
class Forbidden:
    clip: str
    frame: int
    cx: float
    cy: float
    w: float
    h: float


@dataclass
class AnchorSet:
    anchors: dict[Key, Anchor] = field(default_factory=dict)
    forbidden: dict[Key, list[Forbidden]] = field(default_factory=dict)


def _box(b) -> tuple[float, float, float, float] | None:
    """``(cx, cy, w, h)`` of a normalised ``{x,y,w,h}`` (top-left) box or ``[x, y, w, h]``; None when unusable."""
    try:
        if isinstance(b, dict):
            x, y, w, h = (float(b[k]) for k in "xywh")
        elif isinstance(b, (list, tuple)) and len(b) == 4:
            x, y, w, h = (float(v) for v in b)
        else:
            return None
    except (KeyError, TypeError, ValueError):
        return None
    return x + w / 2, y + h / 2, w, h


def _item_boxes(job: dict) -> dict[tuple[str, int, int | None], object]:
    """Inline card boxes of a ball_check job, the fallback when a verdict record carries no bbox."""
    out: dict[tuple[str, int, int | None], object] = {}
    if job.get("kind") == "ball_check":
        for it in job.get("items") or []:
            if it.get("bbox") is not None:
                out[(canonical_clip_id(it["clip"]), int(it["frame_index"]), it.get("box_index"))] = it["bbox"]
    return out


def load_anchors(job: dict, *, marks_dir: Path | None = None) -> AnchorSet:
    """Anchors and forbidden positions of the job's own frames, read from the dataset's marks directory.

    * sidecar human ball labels (``<marks>/<clip>.jsonl``) of F -> anchors (win over a verdict on the same frame);
    * live Ball Check verdicts of F (``<marks>/ball_checks/``): ``ball`` -> anchor at the card box centre,
      ``corrected`` -> anchor at the corrected box, ``not_ball`` / ``box_off`` -> forbidden disc, ``unsure`` -> nothing.

    Anything outside F (another job's frames, another clip, another split) is ignored, never an error."""
    scope = job_scope(job)
    marks = Path(marks_dir) if marks_dir else datasets.get(job.get("dataset")).marks_dir
    out = AnchorSet()
    if not scope.frames:
        return out
    for (clip, f), h in J.job_human(job).items():  # honours the job's frames and exact sidecar stems
        k = (canonical_clip_id(clip), f)
        if k in scope.frames and h.state == BALL and h.centres:
            out.anchors[k] = Anchor(k[0], f, h.centres[0][0], h.centres[0][1], "human")
    boxes = _item_boxes(job)
    for rec in sorted(clips_mod.read_live_verdicts(marks), key=lambda r: float(r.get("ts") or 0.0)):
        k = (rec["clip"], int(rec["frame_index"]))
        if k not in scope.frames:
            continue
        v = rec.get("verdict")
        if v in ANCHOR_VERDICTS:
            c = _box(rec.get("corrected_bbox") if v == "corrected" else None) or _box(
                rec.get("bbox") or boxes.get((k[0], k[1], rec.get("box_index"))))
            if c and k not in out.anchors:
                out.anchors[k] = Anchor(k[0], k[1], c[0], c[1], "verdict")
        elif v in FORBIDDEN_VERDICTS:
            c = _box(rec.get("bbox") or boxes.get((k[0], k[1], rec.get("box_index"))))
            if c:
                out.forbidden.setdefault(k, []).append(Forbidden(k[0], k[1], *c))
    return out
