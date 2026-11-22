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

import hashlib
import json
import math
import re
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import date as date_cls
from pathlib import Path

from footy_track import splits
from footy_track.ball_eval import job_suggest as J  # noqa: N812
from footy_track.ball_eval.panel_score import BALL
from footy_track.ball_mining import clips as clips_mod
from footy_track.ball_mining import refit
from footy_track.ball_mining import round as rnd
from footy_track.ball_trackers.detection_tracker import BallCandidate
from footy_track.ball_trackers.viterbi import Anchor as VAnchor
from footy_track.ball_trackers.viterbi import ForbiddenDisc, ViterbiBallTracker
from footy_track.clip_id import canonical_clip_id
from footy_track.labeller import datasets, jobs_client
from footy_track.labeller.job_edit import MAX_APPEND_PER_CALL

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


# ===========================================================================================================
# GENERATOR (design task 3): constrained Viterbi per window -> ranked card specs + unreachable frames,
# stored as an immutable versioned search file. Pure data: appending the cards to a job is task 4.
# ===========================================================================================================

SEARCH_SCHEMA = "anchor-search/v1"
GENERATOR_CODE_VERSION = "anchor-gen/1"
WINDOW_FRAMES = 250  # frames of one clip per window (autolabel.windows.DEFAULT_WINDOW_FRAMES)
HOP_CAP = 3  # sampled steps beyond the nearest confirmation; further frames need a fresh seed
HOP_DECAY = 0.8  # card confidence = p * HOP_DECAY ** hop
TID_BONUS = 0.1  # ranking-confidence bonus for a candidate on the anchor's offline tracklet
FORBIDDEN_MARGIN_PX = 6.0  # forbidden disc radius = half the rejected box + this
TID_MATCH_NORM = 0.03  # anchor <-> candidate distance (normalised) that counts as "the anchor's candidate"
FRAME_SIZE = (1920, 1080)
TRACKER_PARAMS: dict = {}  # overrides of the ViterbiBallTracker defaults (part of generator_version)

REASON_NO_CANDIDATE = "no_candidate_in_gate"
REASON_NO_PATH = "no_path"


def tracker_params() -> dict:
    """The full effective ViterbiBallTracker parameters (defaults + ``TRACKER_PARAMS``) as a plain dict."""
    return dict(ViterbiBallTracker(FRAME_SIZE, **TRACKER_PARAMS)._p)


def generator_version(detector: str | None) -> str:
    """``det:<candidate model sha8>|trk:<hash8 of tracker params>|code:<version string>``."""
    ph = hashlib.sha1(json.dumps(tracker_params(), sort_keys=True).encode()).hexdigest()[:8]
    return f"det:{detector or 'unknown'}|trk:{ph}|code:{GENERATOR_CODE_VERSION}"


def window_id_of(clip: str, start: int, window_frames: int = WINDOW_FRAMES) -> str:
    """Filesystem-safe id of the window ``[start, start + window_frames)`` of one clip."""
    return f"{clip}_f{start:06d}-{start + window_frames:06d}"


def split_windows(frames: Iterable[Key], window_frames: int = WINDOW_FRAMES) -> dict[str, tuple[str, int, list[int]]]:
    """Group frames into windows of ``window_frames`` consecutive frame numbers of ONE clip (never across clips).

    Returns ``{window_id: (clip, start, sorted sampled frames)}``."""
    out: dict[str, tuple[str, int, list[int]]] = {}
    for clip, f in sorted(set(frames)):
        start = (f // window_frames) * window_frames
        out.setdefault(window_id_of(clip, start, window_frames), (clip, start, []))[2].append(f)
    return out


def resolve_selected(job: dict, store=None, *, clip_of_game=lambda g: g) -> frozenset[Key]:
    """The job's selected set F: ``selected_frame`` rows of ``job_key == job id`` when the store has any, else the job's
    item frames. Every selected row must be in the job's one split (``SplitError`` otherwise)."""
    scope = job_scope(job)
    if store is None:
        return scope.frames
    df = store.read_selected_frames(job_key=scope.job_id)
    if df is None or len(df) == 0:
        return scope.frames
    out = set()
    for r in df.itertuples(index=False):
        clip = canonical_clip_id(clip_of_game(r.game_id))
        if scope.split is not None and (r.split != scope.split or clip_split(clip) != scope.split):
            raise SplitError(f"selected_frame {clip}:{r.frame_index} is {r.split!r}, job {scope.job_id!r} is {scope.split!r}")
        out.add((clip, int(r.frame_index)))
    return frozenset(out)


def terminal_frames(store, selected: Iterable[Key], *, game_of_clip=lambda c: c) -> frozenset[Key]:
    """Frames of ``selected`` whose ball ``frame_label_state`` is already terminal (ball / no_ball / unusable / deferred)."""
    by_clip: dict[str, list[int]] = {}
    for c, f in selected:
        by_clip.setdefault(c, []).append(f)
    out = set()
    for c, fs in by_clip.items():
        out.update((c, f) for f in store.read_ball_terminal_states(game_of_clip(c), fs))
    return frozenset(out)


def judged_frames(
    marks_dir: Path | None, *, pending: Iterable[Key] = (), terminal: Iterable[Key] = (), anchors: AnchorSet | None = None
) -> frozenset[Key]:
    """Frames that must not get a card: live verdicts of the dataset, ``pending`` cards of the round job(s)
    (``round.pending_frames``), terminal ``frame_label_state`` rows (``terminal``) and anchors (human labels)."""
    out = {(canonical_clip_id(c), int(f)) for c, f in pending} | {(canonical_clip_id(c), int(f)) for c, f in terminal}
    if marks_dir is not None:
        out |= clips_mod.read_verdict_frames(marks_dir)
    if anchors is not None:
        out |= set(anchors.anchors)
    return frozenset(out)


@dataclass
class SearchResult:
    """Output of :func:`generate_window`: ranked card specs and the frames the search could not reach."""

    window_id: str
    clip: str
    start: int
    end: int
    cards: list[dict] = field(default_factory=list)
    unreachable: list[dict] = field(default_factory=list)  # {clip, frame, window_id, reason}
    beyond_cap: list[dict] = field(default_factory=list)  # {clip, frame, hop}: needs a fresh seed
    n_anchors: int = 0
    n_forbidden: int = 0


def _rank_key(card: dict) -> tuple:
    m = card["margin"]
    return (card["hop"], math.inf if m is None else m, -card["rank_conf"], card["clip"], card["frame_index"])


def _anchor_cand_tid(cands: CandidateSet, a: Anchor) -> int:
    best, bd = 0, TID_MATCH_NORM
    for c in cands.get(a.clip, a.frame):
        d = math.hypot(c.cx - a.cx, c.cy - a.cy)
        if d <= bd and c.tid:
            best, bd = c.tid, d
    return best


def generate_window(
    window_id: str,
    clip: str,
    start: int,
    frames: list[int],
    cands: CandidateSet,
    *,
    anchorset: AnchorSet,
    judged: frozenset[Key],
    job_id: str,
    generator_ver: str,
    teacher_version: str | None,
    run_id: str | None,
    search_round: int,
    cal: dict | None = None,
    window_frames: int = WINDOW_FRAMES,
    hop_cap: int = HOP_CAP,
    decay: float = HOP_DECAY,
) -> SearchResult:
    """Constrained Viterbi over one window (sampled ``frames`` of one clip), ranked frontier cards.

    Cards: unjudged frames with ``hop <= hop_cap`` that have a pick; ordered by (hop, margin ascending, confidence
    descending); ``unreachable``: unjudged frames with no candidate in the motion gate (or no path). A window with
    no anchor yields nothing (a search needs a seed)."""
    res = SearchResult(window_id, clip, start, start + window_frames)
    frames = sorted(frames)
    w_anchors = sorted((a for (c, f), a in anchorset.anchors.items() if c == clip and f in set(frames)), key=lambda a: a.frame)
    if not frames or not w_anchors:
        return res
    fw, fh = FRAME_SIZE
    discs = [
        ForbiddenDisc(f, (d.cx, d.cy), max(d.w * fw, d.h * fh) / 2 + FORBIDDEN_MARGIN_PX)
        for f in frames
        if (clip, f) not in anchorset.anchors
        for d in anchorset.forbidden.get((clip, f), [])
    ]
    res.n_anchors, res.n_forbidden = len(w_anchors), len(discs)
    trk = ViterbiBallTracker(FRAME_SIZE, **TRACKER_PARAMS)
    for f in frames:
        trk.update(f, [BallCandidate(c.cx - c.w / 2, c.cy - c.h / 2, c.w, c.h, c.conf) for c in cands.get(clip, f)])
    out = trk.constrained_points([VAnchor(a.frame, (a.cx, a.cy)) for a in w_anchors], discs)

    pos = {f: i for i, f in enumerate(frames)}
    a_idx = [(pos[a.frame], a) for a in w_anchors]
    tid_of = {a.frame: _anchor_cand_tid(cands, a) for a in w_anchors}
    for i, f in enumerate(frames):
        if (clip, f) in judged:
            continue
        before = [(i - j, a) for j, a in a_idx if j < i]
        after = [(j - i, a) for j, a in a_idx if j > i]
        nb = min(before, key=lambda x: x[0]) if before else None
        na = min(after, key=lambda x: x[0]) if after else None
        near = min([x for x in (nb, na) if x], key=lambda x: (x[0], x[1].frame))
        hop, anchor = near
        if hop > hop_cap:
            res.beyond_cap.append({"clip": clip, "frame": f, "hop": hop})
            continue
        pt = out.points[i]
        if out.unreachable[i] or pt.x is None:
            res.unreachable.append({
                "clip": clip, "frame": f, "window_id": window_id,
                "reason": REASON_NO_CANDIDATE if out.unreachable[i] else REASON_NO_PATH,
            })
            continue
        match = next((c for c in cands.get(clip, f) if abs(c.cx - (pt.x + pt.w / 2)) < 1e-9 and abs(c.cy - (pt.y + pt.h / 2)) < 1e-9), None)
        conf = match.conf if match else None
        p_cal = refit.calibrated(conf, cal) if (match and cal and cal.get("fitted")) else None
        base = p_cal if p_cal is not None else (conf or 0.0)
        tid = match.tid if match else 0
        bonus = TID_BONUS if tid and tid == tid_of.get(anchor.frame) else 0.0
        m = out.margin[i]
        direction = "both" if (nb and na and nb[0] <= hop_cap and na[0] <= hop_cap) else ("fwd" if near is nb else "bwd")
        res.cards.append({
            "clip": clip, "frame_index": f,
            "bbox": {"x": pt.x, "y": pt.y, "w": pt.w, "h": pt.h},
            "conf": None if conf is None else round(conf * decay**hop, 6),
            "raw_conf": conf, "p_cal": p_cal, "tid": tid or None,
            "cand_source": "detected" if match else "interpolated",
            "hop": hop, "margin": None if (m is None or math.isinf(m)) else float(m),
            "direction": direction, "anchor_id": f"{job_id}:{anchor.clip}:{anchor.frame}",
            "source": "search", "generator_version": generator_ver, "teacher_version": teacher_version,
            "run_id": run_id, "search_round": search_round, "window_id": window_id,
            "rank_conf": round(base * decay**hop + bonus, 6),
        })
    res.cards.sort(key=_rank_key)
    for r, c in enumerate(res.cards):
        c["rank"] = r
    return res


def window_digest(
    window_id: str,
    frames: list[int],
    clip: str,
    cands: CandidateSet,
    *,
    anchorset: AnchorSet,
    judged: frozenset[Key],
    generator_ver: str,
    teacher_version: str | None,
    cal: dict | None,
) -> str:
    """Skip key: same window, frames, candidates, anchors, forbidden, judged set and versions -> same output."""
    fs = set(frames)
    doc = {
        "w": window_id, "frames": sorted(fs), "gen": generator_ver, "teacher": teacher_version,
        "hop": [HOP_CAP, HOP_DECAY, TID_BONUS, FORBIDDEN_MARGIN_PX],
        "cal": cal if cal and cal.get("fitted") else None,
        "cands": [[f, [[c.cx, c.cy, c.w, c.h, c.conf, c.tid] for c in cands.get(clip, f)]] for f in sorted(fs)],
        "anchors": sorted((a.frame, a.cx, a.cy) for (c, f), a in anchorset.anchors.items() if c == clip and f in fs),
        "forbidden": sorted(
            (f, d.cx, d.cy, d.w, d.h) for (c, f), ds in anchorset.forbidden.items() if c == clip and f in fs for d in ds
        ),
        "judged": sorted(f for (c, f) in judged if c == clip and f in fs),
    }
    return hashlib.sha1(json.dumps(doc, sort_keys=True).encode()).hexdigest()


_SEARCH_NAME = re.compile(r"^v(\d{3,})\.json$")


def search_dir(marks_dir: str | Path, window_id: str) -> Path:
    return Path(marks_dir) / "search" / window_id


def _search_versions(marks_dir: str | Path, window_id: str) -> list[tuple[int, Path]]:
    d = search_dir(marks_dir, window_id)
    if not d.is_dir():
        return []
    return sorted((int(m.group(1)), p) for p in d.iterdir() if (m := _SEARCH_NAME.match(p.name)))


def latest_search(marks_dir: str | Path, window_id: str) -> dict | None:
    for _, p in reversed(_search_versions(marks_dir, window_id)):
        try:
            doc = json.loads(p.read_text())
        except (OSError, ValueError):
            continue
        if isinstance(doc, dict) and doc.get("schema") == SEARCH_SCHEMA:
            return doc
    return None


def _write_search(marks_dir: str | Path, window_id: str, build) -> dict:
    """Write the next immutable ``vNNN.json`` (``build(version) -> doc``); an exclusive create, retried on a race."""
    from footy_track.labeller import jobs  # noqa: PLC0415

    vs = _search_versions(marks_dir, window_id)
    v = (vs[-1][0] if vs else 0) + 1
    while True:
        doc = build(v)
        if jobs._write_atomic(search_dir(marks_dir, window_id) / f"v{v:03d}.json", doc, exclusive=True):
            return doc
        v += 1


def check_emittable(cards: Iterable[dict], selected: frozenset[Key], split: str | None) -> None:
    """Guard: every emitted frame belongs to the job's selected set F and to the job's one split."""
    for c in cards:
        k = (canonical_clip_id(c["clip"]), int(c["frame_index"]))
        if k not in selected:
            raise SplitError(f"generator would emit {k}, which is not in the job's selected set")
        if split is not None and clip_split(k[0]) != split:
            raise SplitError(f"generator would emit {k} of split {clip_split(k[0])!r} into a {split!r} job")


def run_search(
    job: dict,
    *,
    marks_dir: Path | None = None,
    root: Path | None = None,
    pool_path: Path | None = None,
    store=None,
    clip_of_game=lambda g: g,
    pending: Iterable[Key] = (),
    teacher_version: str | None = None,
    run_id: str | None = None,
    cal: dict | None = None,
    window_frames: int = WINDOW_FRAMES,
    write: bool = True,
) -> list[dict]:
    """Search every window of the job: load candidates + anchors (task 1), constrained Viterbi (task 2), rank, write.

    Returns one search document per window that has an anchor. With ``write`` the document is stored as
    ``<marks>/search/<window_id>/vNNN.json`` (``search_round`` = N, monotonic per window); if the latest version has
    the same input digest nothing is written and that document is returned with ``reused: true``.
    Frames outside the job's selected set F or split are never read nor emitted."""
    scope = job_scope(job)
    marks = Path(marks_dir) if marks_dir else datasets.get(job.get("dataset")).marks_dir
    selected = resolve_selected(job, store, clip_of_game=clip_of_game)
    cands = load_candidates(job, root=root, pool_path=pool_path)
    anchorset = load_anchors(job, marks_dir=marks)
    # the loaders return frames of the job's items; narrow everything to F (selected_frame may be a subset)
    cands.by_frame = {k: v for k, v in cands.by_frame.items() if k in selected}
    anchorset.anchors = {k: v for k, v in anchorset.anchors.items() if k in selected}
    anchorset.forbidden = {k: v for k, v in anchorset.forbidden.items() if k in selected}
    term = terminal_frames(store, selected, game_of_clip=lambda c: c) if store is not None else ()
    judged = judged_frames(marks, pending=pending, terminal=term, anchors=anchorset)
    gen = generator_version(cands.detector)
    docs = []
    for wid, (clip, start, frames) in split_windows(selected, window_frames).items():
        if not any((clip, f) in anchorset.anchors for f in frames):
            continue
        digest = window_digest(
            wid, frames, clip, cands, anchorset=anchorset, judged=judged, generator_ver=gen, teacher_version=teacher_version, cal=cal
        )
        prev = latest_search(marks, wid) if write else None
        if prev is not None and prev.get("digest") == digest:
            docs.append({**prev, "reused": True})
            continue
        n = ((_search_versions(marks, wid) or [(0, None)])[-1][0] + 1) if write else 1
        r = generate_window(
            wid, clip, start, frames, cands, anchorset=anchorset, judged=judged, job_id=scope.job_id, generator_ver=gen,
            teacher_version=teacher_version, run_id=run_id, search_round=n, cal=cal, window_frames=window_frames,
        )
        check_emittable(r.cards, selected, scope.split)

        def build(v, r=r, digest=digest, n=n):
            return {
                "schema": SEARCH_SCHEMA, "version": v, "window_id": r.window_id, "job_id": scope.job_id,
                "dataset": scope.dataset, "split": scope.split, "clip": r.clip, "start": r.start, "end": r.end,
                "digest": digest, "generator_version": gen, "teacher_version": teacher_version, "run_id": run_id,
                "search_round": v, "n_anchors": r.n_anchors, "n_forbidden": r.n_forbidden, "hop_cap": HOP_CAP,
                "decay": HOP_DECAY, "cards": r.cards, "unreachable": r.unreachable, "beyond_cap": r.beyond_cap,
            }

        doc = _write_search(marks, wid, build) if write else build(1)
        docs.append({**doc, "reused": False})
    return docs


# ===========================================================================================================
# CLIENT (design task 4): push the generator's cards into the live round job within the caps, then reorder.
# Nothing above this line depends on it; the generator API is unchanged.
#
# Safety, in the order the flow runs (design 7.3, 7.4, 7.6):
#   * read the job (``results``: rev + every card + answered state) and the job list (caps);
#   * drop cards whose frame already has a card in the job -> a retried append adds 0, a crash between append and
#     reorder just leaves the new cards in place and the rerun goes straight to the reorder;
#   * take at most the headroom of ``round.tick_decision`` (open training cards < 100, today's cards < 600), best first;
#   * ONE ``append`` call with ``if_rev`` (the server writes it as one transaction under the job lock: all or none);
#     a 409 re-reads and retries, bounded;
#   * reorder the unanswered cards so S cards sit in rank order in the slots they occupy (``refit.apply_order``: the
#     same ``if_rev`` retry loop the refit uses).
# ===========================================================================================================

SEARCH_BUCKET = "S"
APPEND_BY = "ball_anchor_search"
APPEND_TRIES = 3  # rev conflicts tolerated on the append before giving up (the next run converges)
LIVE = refit.LIVE
_BBOX_DIGITS = 5  # the server's append dedupe rounds boxes to this many digits


def _bbox_key(bbox: dict) -> tuple[float, ...]:
    return tuple(round(float(bbox.get(k, 0)), _BBOX_DIGITS) for k in "xywh")


def card_key(card: dict) -> tuple[str, int, tuple[float, ...]]:
    """``(clip, frame, rounded bbox)``: the identity of a card spec (what the server dedupes an inline card on)."""
    return (card["clip"], int(card["frame_index"]), _bbox_key(card["bbox"]))


def idempotency_key(digest: str, card: dict) -> str:
    """Stable key of one card of one search file: ``sha1(file digest | clip | frame | bbox)``."""
    clip, frame, box = card_key(card)
    return hashlib.sha1(f"{digest}|{clip}|{frame}|{box}".encode()).hexdigest()[:16]


def item_from_card(card: dict, *, digest: str, split: str | None, file_version: int | None = None) -> dict:
    """The ``ball_check`` item of one card spec: clip, frame, bbox, plus the hidden ``sampling`` (bucket ``S``) and ``search``
    blocks that persist with the verdict."""
    meta = {
        k: card.get(k)
        for k in ("anchor_id", "hop", "direction", "source", "generator_version", "teacher_version", "run_id",
                  "search_round", "margin", "p_cal")
    }  # fmt: skip
    meta["source"] = "search"
    sampling = {"regime": "train" if split == "train" else "eval", "bucket": SEARCH_BUCKET, "split": split, **meta}
    search = {
        **meta, "window_id": card.get("window_id"), "rank": card.get("rank"), "rank_conf": card.get("rank_conf"),
        "raw_conf": card.get("raw_conf"), "cand_source": card.get("cand_source"), "tid": card.get("tid"),
        "digest": digest, "file_version": file_version, "idem": idempotency_key(digest, card),
    }  # fmt: skip
    p = card.get("p_cal") if card.get("p_cal") is not None else card.get("raw_conf")
    item = {
        "clip": card["clip"], "frame_index": int(card["frame_index"]), "bbox": dict(card["bbox"]),
        "sources": ["anchor_search"], "sampling": sampling, "search": search,
    }  # fmt: skip
    if p is not None:
        item["p_real"] = round(float(p), 5)
    return item


def _ranked_cards(docs: Iterable[dict]) -> list[tuple[dict, dict]]:
    """``(card, its search document)`` of every window, best first (hop, margin, confidence; ties by clip / frame)."""
    pairs = [(c, d) for d in docs for c in d.get("cards") or []]
    return sorted(pairs, key=lambda p: _rank_key(p[0]))


def _today_added(items: Iterable[dict], today: str) -> int:
    return sum(1 for it in items if str((it.get("added_by") or {}).get("ts") or "").startswith(today))


def headroom(job_list: list[dict], res: dict, today: str, *, open_max: int, daily_max: int) -> dict:
    """How many cards may still be appended: ``min(open_max - open, daily_max - today)`` (never below 0).

    For a round job (``ball-round/*``) the counts are ``round.tick_decision``'s over every round job; cards this job received
    today although its key is of an earlier day are added to ``today_cards`` (the tick counts a round by its key's date).
    For any other job only the job itself is counted: its open cards and what it was given today."""
    job = res["job"]
    items = [p["item"] for p in res.get("items") or []]
    if refit.is_round_job(job):
        tick = rnd.tick_decision(job_list, today, open_max=open_max, daily_max=daily_max)
        open_cards, today_cards = tick["open_cards"], tick["today_cards"]
        if not str(job["key"]).startswith(f"{rnd.KEY_PREFIX}/{today}/"):
            today_cards += _today_added(items, today)
    else:
        open_cards = sum(1 for p in res.get("items") or [] if not (p["total"] and p["done"] >= p["total"]))
        today_cards = _today_added(items, today)
    room = max(0, min(open_max - open_cards, daily_max - today_cards))
    return {"room": room, "open_cards": open_cards, "today_cards": today_cards, "open_max": open_max, "daily_max": daily_max}


def _search_sort_key(it: dict) -> tuple:
    s = it.get("search") or {}
    m = s.get("margin")
    return (
        s.get("rank") if s.get("rank") is not None else math.inf,
        s.get("hop") if s.get("hop") is not None else math.inf,
        math.inf if m is None else m,
        -(s.get("rank_conf") or 0.0),
        str(s.get("window_id") or ""),
        it["frame_index"],
    )


def search_order(unanswered: list[tuple[int, dict]]) -> list[int]:
    """New order of the unanswered item positions: the S cards are re-dealt in generator-rank order inside the slots S cards
    already hold, every other card keeps its slot (R / A / G / U / M interleaving is the round's, not ours). A permutation."""
    slots = [pos for pos, it in unanswered if (it.get("sampling") or {}).get("bucket") == SEARCH_BUCKET]
    cards = sorted(
        ((pos, it) for pos, it in unanswered if (it.get("sampling") or {}).get("bucket") == SEARCH_BUCKET),
        key=lambda p: (_search_sort_key(p[1]), p[0]),
    )
    fill = dict(zip(slots, (pos for pos, _ in cards), strict=True))
    return [fill.get(pos, pos) for pos, _ in unanswered]


def _is_rev_conflict(e: Exception) -> bool:
    return isinstance(e, jobs_client.JobsError) and e.status == 409 and isinstance(e.body, dict) and "rev" in e.body


def push_search(
    ref: str,
    docs: Iterable[dict],
    *,
    client=None,
    today: str | None = None,
    open_max: int | None = None,
    daily_max: int | None = None,
    tries: int = APPEND_TRIES,
    reorder: bool = True,
    dry_run: bool = False,
) -> dict:
    """Append the cards of ``docs`` (search documents from :func:`run_search`) to job ``ref`` and reorder the unanswered ones.

    * at most the cap headroom (see :func:`headroom`) is appended, best card first across all windows;
    * a card whose frame already has a card in the job is skipped (a retry adds 0);
    * the append carries ``if_rev``; a 409 re-reads the job and re-plans, ``tries`` times, then reports
      ``append: "rev_conflict"`` (nothing was written; the next run converges);
    * then the unanswered cards are reordered (:func:`search_order`) with the same retry loop as the refit; a job that is not
      live is left alone.

    ``client`` is anything with ``results`` / ``list_jobs`` / ``append`` / ``reorder`` (default ``jobs_client``); the server
    errors other than a rev conflict propagate (fail closed, as the refit does). Returns a small JSON-able summary."""
    client = client or jobs_client
    today = today or date_cls.today().isoformat()
    open_max = rnd.OPEN_CARDS_MAX if open_max is None else open_max
    daily_max = rnd.DAILY_CARDS_MAX if daily_max is None else daily_max
    ranked = _ranked_cards(docs)
    out: dict = {"offered": len(ranked), "added": 0, "skipped_duplicates": 0, "append_attempts": 0}
    conflicts = 0
    while True:
        res = client.results(ref)
        job = res["job"]
        out["job"] = job["id"]
        if job.get("status") not in LIVE:
            out["append"] = f"skipped, job is {job.get('status')}"
            return out
        have = {(p["item"]["clip"], int(p["item"]["frame_index"])) for p in res.get("items") or []}
        fresh = [(c, d) for c, d in ranked if (c["clip"], int(c["frame_index"])) not in have]
        out.setdefault("already_in_job", len(ranked) - len(fresh))
        room = headroom(client.list_jobs(kind="ball_check", progress="true"), res, today, open_max=open_max, daily_max=daily_max)
        out.setdefault("headroom", room)  # as found, before this run appended anything
        take = fresh[: min(room["room"], MAX_APPEND_PER_CALL)]
        if not take or dry_run:
            out["would_add"] = len(take) if dry_run else 0
            break
        items = [item_from_card(c, digest=d["digest"], split=d.get("split"), file_version=d.get("version")) for c, d in take]
        out["append_attempts"] += 1
        try:
            r = client.append(ref, items, by=APPEND_BY, note=f"anchor search {len(items)} cards", if_rev=int(job.get("rev") or 0))
        except jobs_client.JobsError as e:
            if not _is_rev_conflict(e):
                raise
            conflicts += 1
            if conflicts >= tries:
                out["append"] = "rev_conflict"
                break
            continue
        out["added"] += int(r["added"])
        out["skipped_duplicates"] += int(r.get("skipped_duplicates") or 0)
        if not r["added"]:
            break  # the server saw them all already: converged
    if reorder and not dry_run and out.get("append") != "rev_conflict":
        doc = client.results(ref)
        out.update(refit.apply_order(client, ref, doc, search_order, by=APPEND_BY, note="anchor search rank"))
    return out
