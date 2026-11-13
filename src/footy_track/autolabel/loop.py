"""The Claude + detector auto-labelling loop, labelling mode: pure logic.

No GPU, no network, no OpenCV (see ``docs/design/claude_autolabel_loop.md``, section 12).
Everything here works on plain dicts and lists so each step is testable on synthetic
data and every output is JSON. The pieces, in order of a round:

1. clip pool: :func:`eval_clip_ids`, :func:`assert_not_eval`, :func:`pick_clips`. A clip
   of the evaluation split (held-out *or* seen), or one with GT marks, is refused.
2. static filter: :func:`static_candidates` drops candidates that sit at one image
   position across many frames while the camera moves (spare balls and logos by the
   hoardings).
3. decision rule: :func:`classify`, thresholds in :class:`Thresholds`.
4. objects: :func:`link_objects` groups the candidates of one clip into objects so a
   person answers once per object; :func:`build_round` makes the blind review cards.
5. ingest: :func:`plan_ingest` turns a person's verdicts into label lines, hard
   negatives and the audit numbers; :func:`breaker_state` is the circuit breaker.

Geometry is native frame pixels; the cards carry normalised boxes.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path

from footy_track.clip_id import canonical_clip_id

VIDEO_SUFFIXES = {".mp4", ".mov", ".avi", ".mkv"}

# provenance tags of the loop (tier 3 and tier 2, see the design page, section 6)
TAG_AUTO = "claude_auto"  # tier 3: Claude + tracker, no person looked
TAG_CLAUDE = "claude"  # with ``human_checked``: tier 2, a person confirmed the object
TAG_CHECKED = "human_checked"
TAG_LABELLER = "labeller"  # tier 1: a person dragged the box
BALL_LABEL = "in_play_ball"

ROLES = ("review", "audit", "control", "spot")


class EvalClipError(RuntimeError):
    """A clip of the evaluation split was passed to the loop."""


# ---------------------------------------------------------------------------------
# 1. clip pool
# ---------------------------------------------------------------------------------


def _stems(directory: Path | None, patterns: Sequence[str]) -> set[str]:
    if directory is None or not Path(directory).is_dir():
        return set()
    out: set[str] = set()
    for pat in patterns:
        out |= {canonical_clip_id(p.name) for p in Path(directory).glob(pat)}
    return out


def eval_clip_ids(
    split_file: Path,
    *,
    gt_dir: Path | None = None,
    eval_dirs: Iterable[Path] = (),
    cache_dir: Path | None = None,
) -> set[str]:
    """Canonical ids of every clip the loop must never touch.

    The union of: every clip in the split file (``held_out`` and ``seen`` both: the
    seen eval clips are still evaluation clips), every clip with a GT sidecar or a Ball
    Check verdict log under ``gt_dir``, every video under the eval directories and every
    cached eval clip. A missing split file is an error, never an empty set.
    """
    split_file = Path(split_file)
    if not split_file.is_file():
        raise EvalClipError(f"eval split file not found: {split_file}")
    ids = {canonical_clip_id(c) for c in json.loads(split_file.read_text())}
    if not ids:
        raise EvalClipError(f"eval split file {split_file} lists no clips")
    ids |= _stems(gt_dir, ["*.jsonl"])
    if gt_dir is not None:
        ids |= _stems(Path(gt_dir) / "ball_checks", ["*.jsonl"])
    for d in eval_dirs:
        ids |= _stems(d, [f"*{s}" for s in sorted(VIDEO_SUFFIXES)])
    ids |= _stems(cache_dir, ["*.jsonl"])
    return ids


def assert_not_eval(clips: Iterable[str], eval_ids: set[str]) -> None:
    """Hard-fail when any clip (any alias) is an evaluation clip."""
    bad = sorted({c for c in clips if canonical_clip_id(c) in eval_ids})
    if bad:
        raise EvalClipError(
            f"refusing to auto-label evaluation clip(s) {bad}: held-out and seen eval "
            "clips stay human-only"
        )


def list_videos(clips_dir: Path) -> dict[str, Path]:
    """``{canonical clip id: video path}`` for a folder (first alias wins)."""
    out: dict[str, Path] = {}
    for p in sorted(Path(clips_dir).iterdir()):
        if p.suffix.lower() in VIDEO_SUFFIXES:
            out.setdefault(canonical_clip_id(p.name), p)
    return out


def pick_clips(
    clips_dir: Path,
    eval_ids: set[str],
    *,
    names: Sequence[str] = (),
    glob: str | None = None,
    n_random: int = 0,
    seed: int = 7,
    exclude: Iterable[str] = (),
) -> list[str]:
    """The clips of a round, as canonical ids, sorted.

    ``names`` (ids or file names) and ``glob`` (matched against file names) are explicit
    and go through :func:`assert_not_eval`. ``n_random`` draws that many more from the
    folder's clips that are neither evaluation clips nor in ``exclude`` (clips of earlier
    rounds), seeded. Every pick is checked again at the end.
    """
    videos = list_videos(clips_dir)
    chosen: dict[str, None] = {}
    for n in names:
        cid = canonical_clip_id(n)
        if cid not in videos:
            raise FileNotFoundError(f"no video for clip {n!r} in {clips_dir}")
        chosen[cid] = None
    if glob:
        for p in sorted(Path(clips_dir).glob(glob)):
            if p.suffix.lower() in VIDEO_SUFFIXES:
                chosen[canonical_clip_id(p.name)] = None
    assert_not_eval(chosen, eval_ids)
    if n_random:
        skip = set(eval_ids) | {canonical_clip_id(e) for e in exclude} | set(chosen)
        pool = sorted(c for c in videos if c not in skip)
        if len(pool) < n_random:
            raise ValueError(
                f"only {len(pool)} unlabelled clips left, wanted {n_random}"
            )
        for c in random.Random(seed).sample(pool, n_random):
            chosen[c] = None
    out = sorted(chosen)
    assert_not_eval(out, eval_ids)
    return out


# ---------------------------------------------------------------------------------
# 2. static filter
# ---------------------------------------------------------------------------------


@dataclass(frozen=True)
class StaticParams:
    """A candidate is static when, within ``window`` frames, at least ``min_frames``
    frames (itself included) have a candidate within ``radius_px`` of its image
    position while the camera moved at least ``min_cam_px`` between those frames."""

    radius_px: float = 6.0
    min_frames: int = 8
    window: int = 75
    min_cam_px: float = 30.0


def static_candidates(
    cands: dict[int, list[tuple[float, float]]],
    cam: dict[int, tuple[float, float]],
    params: StaticParams | None = None,
) -> set[tuple[int, int]]:
    """``(frame, index)`` of the static candidates.

    ``cands`` maps frame to the candidate centres (native px); ``cam`` maps frame to the
    cumulative camera (global image) shift, native px. With no camera motion estimate
    for a frame the candidate is kept: a ball at rest under a still camera is real.
    """
    params = params or StaticParams()
    frames = sorted(cands)
    out: set[tuple[int, int]] = set()
    for f in frames:
        cf = cam.get(f)
        if cf is None:
            continue
        near_frames = [g for g in frames if abs(g - f) <= params.window]
        for i, (x, y) in enumerate(cands[f]):
            n = 0
            travel = 0.0
            for g in near_frames:
                if any(
                    math.hypot(x - u, y - v) <= params.radius_px for u, v in cands[g]
                ):
                    n += 1
                    cg = cam.get(g)
                    if cg is not None:
                        travel = max(travel, math.hypot(cg[0] - cf[0], cg[1] - cf[1]))
            if n >= params.min_frames and travel >= params.min_cam_px:
                out.add((f, i))
    return out


# ---------------------------------------------------------------------------------
# 3. decision rule
# ---------------------------------------------------------------------------------


@dataclass(frozen=True)
class Thresholds:
    """Defaults from the held-out measurement (design page, section 8): Claude ``ball``
    on the tracker's own candidate had precision 61/62, confidence >= 0.95 gave 33/33;
    on the other candidates 10/33."""

    auto_conf: float = 0.95  # tracker candidate + ball + conf >= this: auto-accept
    drop_conf: float = 0.8  # non-tracker candidate + not_ball + conf >= this: dropped
    audit_rate: float = 0.10  # share of auto-accepted objects a person re-checks
    control_rate: float = 0.05  # share of dropped objects a person re-checks
    breaker_precision: float = 0.97
    min_audit_objects: int = 5  # fewer audited objects: the breaker keeps its state
    spot_min_members: int = 5  # objects this big get spot-check cards
    spot_cards: int = 2
    max_gap_cards: int = 40  # missed-ball cards per round (one per gap)


def classify(
    source: str,
    verdict: str | None,
    conf: float | None,
    thr: Thresholds | None = None,
    *,
    breaker_open: bool = False,
) -> tuple[str, bool]:
    """``(class, would_auto)``: class is ``auto``, ``review`` or ``drop``.

    * tracker candidate, ``ball``, conf >= ``auto_conf``: ``auto``. With the breaker
      open it is ``review`` instead and ``would_auto`` stays True, so the round still
      measures what auto-accept would have got right.
    * tracker candidate, ``ball`` below that: ``review`` (second tier: 0.8 to 0.95 is
      right 28 of 43 over all sources, too many wrong to accept unseen). A tracker
      candidate Claude calls ``not_ball`` / ``unsure`` / invalid: ``review``, since the
      tracker and Claude disagree and one is wrong.
    * other candidate (``top``, ``redetect``): ``ball`` of any confidence: ``review``;
      ``not_ball`` with conf >= ``drop_conf``: ``drop``; anything else: ``review``.
    """
    thr = thr or Thresholds()
    conf = 0.0 if conf is None else conf
    if source == "tracker":
        if verdict == "ball" and conf >= thr.auto_conf:
            return ("review", True) if breaker_open else ("auto", True)
        return "review", False
    if verdict == "not_ball" and conf >= thr.drop_conf:
        return "drop", False
    return "review", False


# ---------------------------------------------------------------------------------
# 4. objects and cards
# ---------------------------------------------------------------------------------


def link_objects(
    cands: Sequence[dict],
    *,
    max_gap: int = 12,
    base_px: float = 12.0,
    per_frame_px: float = 12.0,
    tracker_base_px: float = 20.0,
    tracker_per_frame_px: float = 40.0,
) -> list[list[dict]]:
    """Group the candidates of one clip into objects (lists of members by frame).

    Each candidate dict needs ``frame``, ``cx``, ``cy``, ``source`` and ``track_id``
    (the tracker's id for a tracker candidate, else None). Two tracker candidates with
    the same ``track_id`` are one object. Otherwise a candidate joins the object whose
    last member is at most ``max_gap`` frames earlier and within ``base_px +
    per_frame_px * frames`` (a looser bound between two tracker picks), nearest first,
    never two members on one frame. Deterministic.
    """
    objs: list[list[dict]] = []
    by_track: dict[int, int] = {}
    order = sorted(cands, key=lambda c: (c["frame"], c["cx"], c["cy"], c["id"]))
    for c in order:
        tid = c.get("track_id")
        if c["source"] == "tracker" and tid is not None and tid in by_track:
            o = objs[by_track[tid]]
            if o[-1]["frame"] != c["frame"]:
                o.append(c)
                continue
        best, best_d = None, math.inf
        for k, o in enumerate(objs):
            last = o[-1]
            df = c["frame"] - last["frame"]
            if df <= 0 or df > max_gap:
                continue
            both_tracker = c["source"] == "tracker" and last["source"] == "tracker"
            bound = (
                tracker_base_px + tracker_per_frame_px * df
                if both_tracker
                else base_px + per_frame_px * df
            )
            d = math.hypot(c["cx"] - last["cx"], c["cy"] - last["cy"])
            if d <= bound and d < best_d:
                best, best_d = k, d
        if best is None:
            objs.append([c])
            best = len(objs) - 1
        else:
            objs[best].append(c)
        if c["source"] == "tracker" and tid is not None:
            by_track.setdefault(tid, best)
    return objs


def _h(s: str) -> str:
    return hashlib.sha1(s.encode()).hexdigest()


def object_id(clip: str, members: Sequence[dict]) -> str:
    """Stable id from the clip and the object's first member (frame and position)."""
    m = members[0]
    return _h(f"{clip}|{m['frame']}|{round(m['cx'])}|{round(m['cy'])}")[:12]


def _frac(key: str) -> float:
    """Deterministic pseudo-random number in [0, 1) from a key."""
    return int(_h(key)[:8], 16) / 0x100000000


def _box(c: dict, size: Sequence[int]) -> dict:
    fw, fh = size
    w, h = max(c["w"], 10.0), max(c["h"], 10.0)
    x = max(0.0, min(1.0, (c["cx"] - w / 2) / fw))
    y = max(0.0, min(1.0, (c["cy"] - h / 2) / fh))
    return {
        "x": round(x, 5),
        "y": round(y, 5),
        "w": round(min(w / fw, 1 - x), 5),
        "h": round(min(h / fh, 1 - y), 5),
    }


def _rep(members: Sequence[dict], cls: str = "review") -> dict:
    """The member a card shows. Never a run of consecutive frames of one object: a
    gap shows its middle frame (furthest from any detection); an object of 3 or more
    members shows the frame of fastest motion (the most informative one); a shorter one
    its first member."""
    if cls == "gap":
        return members[len(members) // 2]
    if len(members) < 3:
        return members[0]
    best, best_v = members[1], -1.0
    for a, b in zip(members, members[1:], strict=False):
        v = math.hypot(b["cx"] - a["cx"], b["cy"] - a["cy"]) / max(
            b["frame"] - a["frame"], 1
        )
        if v > best_v:
            best, best_v = b, v
    return best


#: queue priority of a card, highest first (lower number): missed balls, then spare balls
#: and false starts, then Claude-uncertain tracker candidates, then the blind audit
#: and controls, then the spot checks.
PRIORITY = {"gap": 1, "other": 2, "tracker": 3, "audit": 4, "control": 4, "spot": 5}


def order_cards(cards: list[dict], meta: dict[str, dict], round_id: str) -> list[dict]:
    """By priority, and inside a priority round-robin over the clips (a hash order inside
    each clip), so no clip floods the front of the queue."""
    rank: dict[tuple, int] = {}
    seq: dict[tuple, int] = defaultdict(int)
    for c in sorted(cards, key=lambda c: _h(f"{round_id}|shuffle|{c['id']}")):
        k = (meta[c["id"]]["priority"], c["run_clip"])
        rank[c["id"]] = seq[k]
        seq[k] += 1
    return sorted(
        cards,
        key=lambda c: (
            meta[c["id"]]["priority"],
            rank[c["id"]],
            _h(f"{round_id}|{c['run_clip']}"),
            c["id"],
        ),
    )


def build_round(  # noqa: PLR0912, PLR0915
    items: Sequence[dict],
    *,
    round_id: str,
    sizes: dict[str, Sequence[int]],
    stems: dict[str, str],
    thr: Thresholds | None = None,
    breaker_open: bool = False,
) -> dict:
    """Decide every answered candidate, group into objects, build the blind cards.

    ``items``: one dict per question with ``id``, ``clip``, ``frame``, ``cx``, ``cy``,
    ``w``, ``h``, ``det_conf``, ``source``, ``track_id``, ``verdict``, ``claude_conf``
    and ``status`` (``asked`` or ``filtered`` for a static-filter drop, never asked).
    ``sizes``/``stems``: per run clip, the frame size and the video stem the labeller
    serves it under. Returns ``{"decisions", "objects", "cards", "meta", "counts"}``:
    ``cards`` is the shuffled blind list (no source, role or hint on a card), ``meta``
    holds card id to role/object/members and is never shown to the reviewer.
    """
    thr = thr or Thresholds()
    decisions: dict[str, dict] = {}
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for it in items:
        if it["status"] == "filtered":
            cls, would = "drop", False
            reason = "static"
        elif it["status"] == "gap":
            cls, would = "gap", False
            reason = "tracker_gap"
        elif it.get("verdict") is None:
            continue  # not answered yet (budget stop): not decided this round
        else:
            cls, would = classify(
                it["source"],
                it["verdict"],
                it["claude_conf"],
                thr,
                breaker_open=breaker_open,
            )
            reason = "claude"
        decisions[it["id"]] = {
            "class": cls,
            "would_auto": would,
            "reason": reason,
            "verdict": it.get("verdict"),
            "claude_conf": it.get("claude_conf"),
            "source": it["source"],
        }
        groups[(it["clip"], cls)].append(it)

    objects: dict[str, dict] = {}
    for (clip, cls), members in sorted(groups.items()):
        gap = cls == "gap"
        linked = (
            link_objects(members, max_gap=15, base_px=80.0, per_frame_px=30.0)
            if gap
            else link_objects(members)
        )
        for ms in linked:
            oid = object_id(clip, ms)
            rep = _rep(ms, cls)
            objects[oid] = {
                "id": oid,
                "clip": clip,
                "class": cls,
                "members": [m["id"] for m in ms],
                "frames": [m["frame"] for m in ms],
                "rep": rep["id"],
                "would_auto": any(decisions[m["id"]]["would_auto"] for m in ms),
                "reason": decisions[ms[0]["id"]]["reason"],
                "sources": sorted({m["source"] for m in ms}),
            }

    byq = {it["id"]: it for it in items}
    cards: list[dict] = []
    meta: dict[str, dict] = {}

    def add_card(
        role: str, obj: dict, member_id: str, tag: str, prio: str | None = None
    ) -> None:
        prio = prio or role
        m = byq[member_id]
        cid = _h(f"{round_id}|{role}|{obj['id']}|{tag}")[:16]
        clip = obj["clip"]
        cards.append(
            {
                "id": cid,
                "clip": stems[clip],
                "run_clip": clip,
                "frame_index": m["frame"],
                "frame_size": list(sizes[clip]),
                "bbox": _box(m, sizes[clip]),
                "label": BALL_LABEL,
                "provenance": "list",
                "confidence": None,
            }
        )
        meta[cid] = {
            "role": role,
            "object": obj["id"],
            "member": member_id,
            "n_members": len(obj["members"]),
            "would_auto": obj["would_auto"],
            "priority": PRIORITY[prio],
        }

    def spots(obj: dict) -> list[str]:
        if len(obj["members"]) < thr.spot_min_members:
            return []
        # the first and the last member (never next to the shown frame by construction)
        ends = [obj["members"][0], obj["members"][-1]]
        return [m for m in dict.fromkeys(ends) if m != obj["rep"]][: thr.spot_cards]

    auto_objs = sorted(
        (o for o in objects.values() if o["class"] == "auto"), key=lambda o: _h(o["id"])
    )
    n_audit = min(len(auto_objs), max(1, round(thr.audit_rate * len(auto_objs))))
    audit_ids = {o["id"] for o in auto_objs[:n_audit]} if auto_objs else set()
    for o in objects.values():
        if o["class"] == "gap":
            continue  # chosen below, capped
        if o["class"] == "review":
            prio = "other" if set(o["sources"]) - {"tracker"} else "tracker"
            add_card("review", o, o["rep"], "rep", prio)
            for m in spots(o):
                add_card("spot", o, m, f"s{m}")
        elif o["class"] == "auto" and o["id"] in audit_ids:
            o["audit"] = True
            add_card("audit", o, o["rep"], "rep")
            for m in spots(o):
                add_card("spot", o, m, f"s{m}")
        elif (
            o["class"] == "drop"
            and _frac(f"{round_id}|ctl|{o['id']}") < thr.control_rate
        ):
            o["control"] = True
            add_card("control", o, o["rep"], "rep")
    gaps = sorted(
        (o for o in objects.values() if o["class"] == "gap"), key=lambda o: _h(o["id"])
    )
    for o in gaps[: thr.max_gap_cards]:
        add_card("gap", o, o["rep"], "rep")
    for o in gaps[thr.max_gap_cards :]:
        o["capped"] = True
    cards = order_cards(cards, meta, round_id)
    counts = round_counts(items, decisions, objects, meta)
    return {
        "decisions": decisions,
        "objects": objects,
        "cards": cards,
        "meta": meta,
        "counts": counts,
    }


def round_counts(items, decisions, objects, meta) -> dict:
    cls = defaultdict(int)
    for d in decisions.values():
        cls[d["class"]] += 1
    roles = defaultdict(int)
    for m in meta.values():
        roles[m["role"]] += 1
    return {
        "candidates": sum(1 for i in items if i["status"] != "gap"),
        "gap_frames": sum(1 for i in items if i["status"] == "gap"),
        "gap_objects": sum(1 for o in objects.values() if o["class"] == "gap"),
        "cards_by_priority": {
            str(p): sum(1 for m in meta.values() if m["priority"] == p)
            for p in sorted(set(PRIORITY.values()))
        },
        "filtered_static": sum(1 for i in items if i["status"] == "filtered"),
        "asked": sum(1 for i in items if i["status"] == "asked"),
        "answered": sum(1 for i in items if i.get("verdict") is not None),
        "auto_candidates": cls["auto"],
        "review_candidates": cls["review"],
        "dropped_candidates": cls["drop"],
        "objects": len(objects),
        "objects_by_class": {
            c: sum(1 for o in objects.values() if o["class"] == c)
            for c in ("auto", "review", "drop", "gap")
        },
        "cards": len(meta),
        "cards_by_role": dict(roles),
        "would_auto_objects": sum(1 for o in objects.values() if o["would_auto"]),
    }


# ---------------------------------------------------------------------------------
# 5. ingest, audit, circuit breaker
# ---------------------------------------------------------------------------------

GOOD = ("ball", "corrected")  # geometry good enough: the object is the ball
PRESENT = ("ball", "box_off", "corrected")  # the object is the ball


@dataclass
class IngestPlan:
    labels: dict[str, list[dict]] = field(default_factory=dict)  # clip -> sidecar lines
    negatives: list[dict] = field(default_factory=list)
    object_verdicts: dict[str, str] = field(default_factory=dict)
    inconsistent: list[str] = field(default_factory=list)
    counts: dict = field(default_factory=dict)
    audit: dict = field(default_factory=dict)
    controls: dict = field(default_factory=dict)


def _line(frame: int, bbox: dict, tags: list[str]) -> dict:
    return {
        "frame_index": frame,
        "bbox": bbox,
        "center": {"x": bbox["x"] + bbox["w"] / 2, "y": bbox["y"] + bbox["h"] / 2},
        "tags": tags,
    }


def _cand_bbox(m: dict, size: Sequence[int]) -> dict:
    return _box(m, size)


def plan_ingest(  # noqa: PLR0912, PLR0915
    items: Sequence[dict],
    built: dict,
    verdicts: dict[str, dict],
    *,
    sizes: dict[str, Sequence[int]],
    eval_ids: set[str],
    thr: Thresholds | None = None,
) -> IngestPlan:
    """Turn a person's verdicts on the round's cards into label lines and numbers.

    ``verdicts`` maps card id to its latest live Ball Check record. Rules (all per
    object; a verdict on the object's card applies to its frames):

    * auto object, audited ``ball``: its frames become ``[label, claude,
      human_checked]`` (tier 2); ``not_ball``: its frames are dropped and written as
      hard negatives; ``unsure`` / no verdict: stays ``claude_auto`` (tier 3);
    * auto object not audited: ``claude_auto``;
    * review (or control) object ``ball``: its review members become ``[label, claude,
      human_checked]``; ``corrected``: only the card's frame, as ``labeller`` with the
      dragged box; ``box_off``: nothing is written (the box is wrong); ``not_ball``:
      hard negatives; ``unsure`` / none: nothing;
    * a missed-ball (gap) object: ``corrected`` writes the dragged box as ``labeller`` (a
      hard positive the detector missed), ``ball`` the predicted box as ``claude`` +
      ``human_checked``, ``not_ball`` (no ball visible) a negative;
    * spot-check cards that contradict the object card (``ball`` against ``not_ball``)
      mark the object inconsistent: nothing is written for it, and it is reported.

    A clip of the evaluation split raises :class:`EvalClipError` before any line is made.
    """
    thr = thr or Thresholds()
    assert_not_eval({o["clip"] for o in built["objects"].values()}, eval_ids)
    byq = {it["id"]: it for it in items}
    meta, objects = built["meta"], built["objects"]
    plan = IngestPlan()
    main_card: dict[str, str] = {}
    spot_cards: dict[str, list[str]] = defaultdict(list)
    for cid, m in meta.items():
        if m["role"] == "spot":
            spot_cards[m["object"]].append(cid)
        else:
            main_card[m["object"]] = cid

    def verdict_of(cid: str) -> str | None:
        rec = verdicts.get(cid)
        return None if rec is None else rec.get("verdict")

    def emit(clip: str, m: dict, tags: list[str], bbox: dict | None = None) -> None:
        plan.labels.setdefault(clip, []).append(
            _line(m["frame"], bbox or _cand_bbox(m, sizes[clip]), tags)
        )

    c = defaultdict(int)
    audit_n = {"ball": 0, "not_ball": 0, "unsure": 0, "frames_ball": 0, "frames_not": 0}
    ctl = {"ball": 0, "not_ball": 0, "unsure": 0}
    for oid in sorted(objects):
        o = objects[oid]
        clip = o["clip"]
        members = [byq[q] for q in o["members"]]
        cid = main_card.get(oid)
        v = verdict_of(cid) if cid else None
        if v is not None:
            plan.object_verdicts[oid] = v
        # spot-check consistency
        spot_vs = [verdict_of(s) for s in spot_cards.get(oid, [])]
        pos = v in PRESENT
        contradict = v in (*PRESENT, "not_ball") and any(
            s is not None and ((s in PRESENT) != pos) and s in (*PRESENT, "not_ball")
            for s in spot_vs
        )
        if contradict:
            plan.inconsistent.append(oid)
            c["objects_inconsistent"] += 1
            continue
        role = meta[cid]["role"] if cid else None
        if o["class"] == "auto":
            if role == "audit":
                if v in PRESENT:
                    audit_n["ball"] += 1
                    audit_n["frames_ball"] += len(members)
                    for m in members:
                        emit(clip, m, [BALL_LABEL, TAG_CLAUDE, TAG_CHECKED])
                    c["labels_tier2_from_audit"] += len(members)
                elif v == "not_ball":
                    audit_n["not_ball"] += 1
                    audit_n["frames_not"] += len(members)
                    for m in members:
                        plan.negatives.append(_neg(m, clip, sizes, "audit_not_ball"))
                    c["auto_rejected_frames"] += len(members)
                else:
                    audit_n["unsure"] += 1
                    for m in members:
                        emit(clip, m, [BALL_LABEL, TAG_AUTO])
                    c["labels_auto"] += len(members)
            else:
                for m in members:
                    emit(clip, m, [BALL_LABEL, TAG_AUTO])
                c["labels_auto"] += len(members)
            continue
        if o["class"] == "gap":
            # a missed ball: the card shows the tracker's predicted position
            if role != "gap" or v is None:
                continue
            m = byq[meta[cid]["member"]]
            c["gap_cards_decided"] += 1
            cb = verdicts[cid].get("corrected_bbox")
            if v == "corrected" and isinstance(cb, dict):
                emit(
                    clip,
                    m,
                    [BALL_LABEL, TAG_LABELLER],
                    {k: float(cb[k]) for k in "xywh"},
                )
                c["gap_hard_positives_tier1"] += 1
            elif v == "ball":  # the predicted position was right
                emit(clip, m, [BALL_LABEL, TAG_CLAUDE, TAG_CHECKED])
                c["gap_confirmed_tier2"] += 1
            elif v == "not_ball":  # no ball visible: a no-ball frame
                plan.negatives.append(_neg(m, clip, sizes, "gap_no_ball"))
                c["gap_no_ball"] += 1
            continue
        if role not in ("review", "control") or v is None:
            continue
        shadow = o["would_auto"] and role == "review"
        if shadow:  # breaker open: a would-be auto object a person judged
            key = (
                "ball" if v in PRESENT else "not_ball" if v == "not_ball" else "unsure"
            )
            audit_n[key] += 1
            audit_n["frames_ball" if key == "ball" else "frames_not"] += (
                len(members) if key != "unsure" else 0
            )
        if role == "control":
            key = (
                "ball" if v in PRESENT else "not_ball" if v == "not_ball" else "unsure"
            )
            ctl[key] += 1
        if v == "ball":
            for m in members:
                emit(clip, m, [BALL_LABEL, TAG_CLAUDE, TAG_CHECKED])
            c["labels_tier2_from_review"] += len(members)
        elif v == "corrected":
            rec = verdicts[cid]
            cb = rec.get("corrected_bbox")
            m = byq[meta[cid]["member"]]
            if isinstance(cb, dict):
                emit(
                    clip,
                    m,
                    [BALL_LABEL, TAG_LABELLER],
                    {k: float(cb[k]) for k in "xywh"},
                )
                c["labels_tier1_corrected"] += 1
        elif v == "not_ball":
            for m in members:
                plan.negatives.append(_neg(m, clip, sizes, role + "_not_ball"))
            c["negative_frames"] += len(members)
    plan.counts = dict(c)
    dec = audit_n["ball"] + audit_n["not_ball"]
    fr = audit_n["frames_ball"] + audit_n["frames_not"]
    plan.audit = {
        **audit_n,
        "objects_decided": dec,
        "precision_objects": audit_n["ball"] / dec if dec else None,
        "precision_frames": audit_n["frames_ball"] / fr if fr else None,
    }
    plan.controls = {
        **ctl,
        "dropped_but_ball": ctl["ball"],
        "dropped_objects_checked": sum(ctl.values()),
    }
    return plan


def _neg(m: dict, clip: str, sizes, reason: str) -> dict:
    return {
        "clip": clip,
        "frame_index": m["frame"],
        "bbox": _box(m, sizes[clip]),
        "source": m["source"],
        "reason": reason,
        "qid": m["id"],
    }


def breaker_state(
    reports: Sequence[dict], thr: Thresholds | None = None
) -> tuple[bool, str]:
    """``(open, why)`` from earlier round reports, oldest first.

    The latest round whose audit judged at least ``min_audit_objects`` objects decides:
    audit precision (frames, else objects) below ``breaker_precision`` opens the breaker,
    so the next round sends everything to the queue. When the breaker is open the round
    still measures the would-be-auto candidates (``would_auto``), so a later round can
    close it. No audited round yet: closed.
    """
    thr = thr or Thresholds()
    for rep in reversed(list(reports)):
        a = rep.get("audit") or {}
        if (a.get("objects_decided") or 0) < thr.min_audit_objects:
            continue
        p = a.get("precision_frames")
        if p is None:
            p = a.get("precision_objects")
        if p is None:
            continue
        ok = p >= thr.breaker_precision
        why = (
            f"round {rep.get('round')}: audit precision {p:.3f} over "
            f"{a['objects_decided']} objects, threshold {thr.breaker_precision}"
        )
        return (not ok), why
    return False, "no audited round yet"


# ---------------------------------------------------------------------------------
# label files: merge and dedupe
# ---------------------------------------------------------------------------------


def tier(tags: Sequence[str]) -> int:
    """1 labeller, 2 claude + human_checked, 3 claude_auto (lower is stronger)."""
    if TAG_LABELLER in tags:
        return 1
    if TAG_CLAUDE in tags and TAG_CHECKED in tags:
        return 2
    return 3


def _iou(a: dict, b: dict) -> float:
    iw = min(a["x"] + a["w"], b["x"] + b["w"]) - max(a["x"], b["x"])
    ih = min(a["y"] + a["h"], b["y"] + b["h"]) - max(a["y"], b["y"])
    if iw <= 0 or ih <= 0:
        return 0.0
    inter = iw * ih
    return inter / (a["w"] * a["h"] + b["w"] * b["h"] - inter)


def merge_label_lines(lines: Iterable[dict], min_iou: float = 0.3) -> list[dict]:
    """Union of sidecar lines with one line per physical box: on one frame, boxes that
    overlap (IoU >= ``min_iou``) are the same box and the strongest tier wins (a person's
    verdict beats ``claude_auto``). Order by frame, then tier."""
    by_frame: dict[int, list[dict]] = defaultdict(list)
    for ln in lines:
        kept = by_frame[ln["frame_index"]]
        for i, k in enumerate(kept):
            if _iou(k["bbox"], ln["bbox"]) >= min_iou:
                if tier(ln["tags"]) < tier(k["tags"]):
                    kept[i] = ln
                break
        else:
            kept.append(ln)
    return [ln for f in sorted(by_frame) for ln in by_frame[f]]


def dedupe_points(
    points: Iterable[Sequence[float]], tol_px: float = 4.0
) -> list[list[float]]:
    """Drop marks within ``tol_px`` of an earlier one (a confirmed box that sat on a
    mark adds a duplicate mark)."""
    out: list[list[float]] = []
    for p in points:
        if not any(math.hypot(p[0] - q[0], p[1] - q[1]) <= tol_px for q in out):
            out.append([float(p[0]), float(p[1])])
    return out


def asdict_thr(thr: Thresholds) -> dict:
    return asdict(thr)
