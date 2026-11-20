"""Player Check API: verify player boxes per *cluster*, not per box (docs/design/full_frame_labelling.md 4c).

Routes (all take the clip as ``clip``; windows are frame ranges ``start``..``end``, default the whole clip):

* ``GET /player-check/queue`` - cluster cards (worst flicker / overlap first), frame-sweep cards and a
  deterministic 10 % blind-audit frame list. Triage ``policy`` defaults to ``audit_all`` (design 6b: nothing
  is auto-accepted for players on detector confidence); ``permissive`` lets isolated, stable, confident
  tracks drop out of the queue as ``auto_ok``.
* ``POST /player-check/verdict`` - ``ok`` | ``count_is_N`` (+ ``start``/``end`` run) | ``fixed`` (+ ``fix``
  boxes, written to the sidecar as ``labeller`` = ``hand_label`` GT) | ``unsure``, for a cluster; or the same
  for a frame sweep (``sweep`` = frame index). A sweep ``ok`` / ``fixed`` with ``scope: "frame"`` also verifies every
  cluster on that frame (whole-frame review).

State: decisions go to the feature store ``player_check`` table. A frame's player ``frame_label_state`` is
``complete`` (method ``hand_frame``) only when EVERY cluster on it is verified AND its sweep is verified;
otherwise ``partial``. ``unsure`` never completes. Cluster ids (``c0``...) are window-local and recomputed from
the sidecar on every call, so completeness is evaluated at write time against the pre-fix boxes.

Split guard: heldout / eval-in-match clips are refused (403) unless the request sets ``eval: true``; such
windows feed the player eval panel and are never training data.

See ``src/footy_track/labeller/README.md`` section 19j (LAB-1470..).
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import re
from collections import defaultdict
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
from fastapi import APIRouter, HTTPException
from fastapi.responses import HTMLResponse

from footy_track import match_splits, splits
from footy_track.autolabel import clusters as cl
from footy_track.feature_store import FeatureStore
from footy_track.feature_store.lock import StoreLocked, default_db_path, store_lock
from footy_track.feature_store.schema import (
    SWEEP_CLUSTER_ID,
    ClsGroup,
    FrameLabelStateRow,
    LabelMethod,
    LabelState,
    PlayerCheckRow,
    PlayerCheckVerdict,
)
from footy_track.labeller.constants import PLAYER_LABELS, PROV_LABELLER
from footy_track.labeller.names import safe_clip_name
from footy_track.labeller.review import (
    _gt_marks_dir,
    _iter_frame_box_lines,
    _sidecar_boxes,
    _write_text_atomic,
)

router = APIRouter()
_PAGE = Path(__file__).parent / "web" / "player_check.html"

DEFAULT_ASPECT = 16 / 9  # x is scaled by this so IoU / distances are isotropic
AUDIT_MOD = 10  # 1 in 10 frames is a blind-audit frame
RUN_ID = "player_check"
_COUNT_IS = re.compile(r"^count_is_(\d+)$")

RowMap = dict[str, tuple[str, int | None]]  # cluster_id -> (verdict, asserted count)


def _db_path() -> Path:  # tests monkeypatch this
    return default_db_path()


@contextlib.contextmanager
def _store() -> Iterator[FeatureStore]:
    db = _db_path()
    try:
        with store_lock(db, exclusive=True, timeout=5.0):
            store = FeatureStore.open(db)
            try:
                yield store
            finally:
                store.close()
    except StoreLocked as exc:
        raise HTTPException(503, "feature store is locked") from exc


@router.get("/player-check", response_class=HTMLResponse)
async def player_check_page() -> HTMLResponse:
    return HTMLResponse(
        _PAGE.read_text(), headers={"Cache-Control": "no-store, max-age=0"}
    )


# ---- split guard ----------------------------------------------------------------------------- #


def guard_split(clip: str, eval_flag: bool) -> str:
    """Return the clip's split; refuse non-train clips unless the request is an explicit eval one."""
    try:
        sp = str(splits.split_of(splits.match_key_of(clip), clip))
    except match_splits.RegisterError as exc:
        raise HTTPException(503, f"split register unavailable: {exc}") from exc
    if sp != splits.TRAIN and not eval_flag:
        raise HTTPException(
            403,
            f"{clip} is {sp}: Player Check on eval data needs an explicit eval flag "
            "(its labels feed the player eval panel, never training)",
        )
    return sp


# ---- window / cluster computation ------------------------------------------------------------ #


def _player_records(clip: str) -> dict[int, list[dict]]:
    path = _gt_marks_dir() / f"{clip}.jsonl"
    recs = _sidecar_boxes(path, None) if path.exists() else None
    if recs is None:
        raise HTTPException(404, "clip not found")
    by_frame: dict[int, list[dict]] = defaultdict(list)
    for r in recs:
        if r["label"] in PLAYER_LABELS:
            by_frame[r["frame_index"]].append(r)
    return by_frame


def _window(
    by_frame: dict[int, list[dict]], start: int | None, end: int | None, aspect: float
) -> tuple[list[int], list[cl.ClusterTrack]]:
    frames = sorted(
        f
        for f in by_frame
        if (start is None or f >= start) and (end is None or f <= end)
    )
    geo = {
        f: [
            (
                r["bbox"]["x"] * aspect,
                r["bbox"]["y"],
                r["bbox"]["w"] * aspect,
                r["bbox"]["h"],
            )
            for r in by_frame[f]
        ]
        for f in frames
    }
    return frames, cl.track_clusters(geo)


def _boxes(recs: list[dict]) -> list[dict]:
    return [
        {
            "box_index": r["box_index"],
            "label": r["label"],
            "bbox": r["bbox"],
            "confidence": r["confidence"],
        }
        for r in recs
    ]


def _crop(boxes: list[dict]) -> dict[str, float]:
    """Normalised union of a cluster's boxes with 15 % padding, clamped to the frame."""
    x0 = min(b["bbox"]["x"] for b in boxes)
    y0 = min(b["bbox"]["y"] for b in boxes)
    x1 = max(b["bbox"]["x"] + b["bbox"]["w"] for b in boxes)
    y1 = max(b["bbox"]["y"] + b["bbox"]["h"] for b in boxes)
    px, py = 0.15 * (x1 - x0), 0.15 * (y1 - y0)
    x0, y0, x1, y1 = (
        max(0.0, x0 - px),
        max(0.0, y0 - py),
        min(1.0, x1 + px),
        min(1.0, y1 + py),
    )
    return {"x": x0, "y": y0, "w": x1 - x0, "h": y1 - y0}


def _strip(frames: list[int], flagged: set[int], size: int = 5) -> list[int]:
    """Up to *size* frames for the card: flagged frames first, then evenly spaced fill."""
    pick = sorted(flagged & set(frames))[:size]
    step = max(1, len(frames) // size)
    for f in frames[::step]:
        if f not in pick and len(pick) < size:
            pick.append(f)
    return sorted(pick)


def is_audit_frame(clip: str, frame_index: int) -> bool:
    """Deterministic blind-audit membership (1 in ``AUDIT_MOD`` frames, stable across calls)."""
    return (
        int(hashlib.sha1(f"{clip}:{frame_index}".encode()).hexdigest(), 16) % AUDIT_MOD
        == 0
    )


def _cluster_verified(verdict: str, count: int | None, n_boxes: int) -> bool:
    if verdict in ("ok", "fixed"):
        return True
    return (
        verdict == "count_n" and count == n_boxes
    )  # a count that disagrees with the boxes needs a fix


def frame_state(expected: dict[str, int], rows: RowMap) -> LabelState:
    """``complete`` iff every expected cluster (id -> box count) is verified and the sweep is verified.

    ``unsure`` (or a count_n that disagrees with the boxes, or a missing row) never completes.
    """
    sweep = rows.get(SWEEP_CLUSTER_ID)
    if sweep is None or sweep[0] not in ("ok", "fixed"):
        return LabelState.PARTIAL
    for cid, n in expected.items():
        row = rows.get(cid)
        if row is None or not _cluster_verified(row[0], row[1], n):
            return LabelState.PARTIAL
    return LabelState.COMPLETE


def _rows_by_frame(df) -> dict[int, RowMap]:  # noqa: ANN001
    out: dict[int, RowMap] = defaultdict(dict)
    for r in df.itertuples():
        out[int(r.frame_index)][r.cluster_id] = (
            r.verdict,
            None if pd.isna(r.count) else int(r.count),
        )
    return out


# ---- queue ----------------------------------------------------------------------------------- #


@router.get("/player-check/queue")
def queue(
    clip: str,
    start: int | None = None,
    end: int | None = None,
    policy: str = cl.POLICY_AUDIT_ALL,
    eval: bool = False,  # noqa: A002 - query parameter name is part of the API
    aspect: float = DEFAULT_ASPECT,
    limit: int = 50,
) -> dict:
    clip = safe_clip_name(clip)
    if policy not in (cl.POLICY_AUDIT_ALL, cl.POLICY_PERMISSIVE):
        raise HTTPException(422, "policy must be audit_all or permissive")
    split = guard_split(clip, eval)
    by_frame = _player_records(clip)
    frames, tracks = _window(by_frame, start, end, aspect)
    with _store() as store:
        rows = _rows_by_frame(store.read_player_checks(clip, frames))

    cards, auto_ok = [], []
    for t in tracks:
        conf = {
            (o.frame, i): by_frame[o.frame][i]["confidence"]
            for o in t.obs
            for i in o.indices
        }
        tri = cl.triage_track(t, policy=policy, confidences=conf)
        if tri.decision == "auto_ok":
            auto_ok.append(t.track_id)
            continue
        if all(
            (row := rows.get(o.frame, {}).get(t.track_id)) is not None
            and _cluster_verified(row[0], row[1], o.count)
            for o in t.obs
        ):
            continue  # fully verified already
        tl = cl.count_timeline(t)
        flick = cl.flicker_frames(tl)
        changes = cl.change_points(tl)
        by_obs = {o.frame: o for o in t.obs}
        shown = _strip(list(t.frames), set(flick) | set(changes))
        strip = [
            {
                "frame_index": f,
                "boxes": _boxes([by_frame[f][i] for i in by_obs[f].indices]),
            }
            for f in shown
        ]
        cards.append(
            {
                "type": "cluster",
                "clip": clip,
                "cluster_id": t.track_id,
                "frames": list(t.frames),
                "timeline": [{"frame_index": f, "count": n} for f, n in tl],
                "runs": [
                    {"start": r.start, "end": r.end, "count": r.count}
                    for r in cl.count_runs(tl)
                ],
                "flicker_frames": flick,
                "change_points": changes,
                "severity": cl.track_severity(t),
                "triage_reason": tri.reason,
                "crop": _crop([b for s in strip for b in s["boxes"]]),
                "strip": strip,
            }
        )
    cards.sort(key=lambda c: (-c["severity"], c["frames"][0], c["cluster_id"]))
    sweeps = [
        {"type": "sweep", "clip": clip, "frame_index": f, "boxes": _boxes(by_frame[f])}
        for f in frames
        if rows.get(f, {}).get(SWEEP_CLUSTER_ID, ("", None))[0] not in ("ok", "fixed")
    ]
    audits = [
        {"type": "audit", "clip": clip, "frame_index": f, "blind": True, "boxes": []}
        for f in frames
        if is_audit_frame(clip, f)
    ]
    return {
        "clip": clip,
        "split": split,
        "eval": eval,
        "policy": policy,
        "frames": len(frames),
        "auto_ok": auto_ok,
        "clusters": cards[:limit],
        "n_clusters_pending": len(cards),
        "sweeps": sweeps[:limit],
        "n_sweeps_pending": len(sweeps),
        "audits": audits,
    }


# ---- verdict --------------------------------------------------------------------------------- #


def _apply_fix(clip: str, fix: list[dict]) -> None:
    """Delete and add player boxes in the clip sidecar (added boxes are ``labeller`` = hand_label GT).

    Same file, per-frame ``box_index`` numbering and atomic-write discipline as ``/review/correct``
    and ``/review/delete``.
    """
    path = _gt_marks_dir() / f"{clip}.jsonl"
    if not path.exists():
        raise HTTPException(404, "clip not found")
    lines = path.read_text().splitlines()
    for item in fix:
        fi = int(item["frame_index"])
        idx = _iter_frame_box_lines(lines, fi)
        for bi in sorted({int(b) for b in item.get("delete", [])}, reverse=True):
            if bi >= len(idx):
                raise HTTPException(422, f"box_index {bi} out of range on frame {fi}")
            del lines[idx[bi]]
        for box in item.get("add", []):
            label = box.get("label", "player")
            if label not in PLAYER_LABELS:
                raise HTTPException(422, f"label {label!r} is not a player class")
            bb = box["bbox"]
            x = max(0.0, min(1.0, float(bb["x"])))
            y = max(0.0, min(1.0, float(bb["y"])))
            w = max(0.0, min(1.0 - x, float(bb["w"])))
            h = max(0.0, min(1.0 - y, float(bb["h"])))
            lines.append(
                json.dumps(
                    {
                        "frame_index": fi,
                        "bbox": {"x": x, "y": y, "w": w, "h": h},
                        "center": {"x": x + w / 2, "y": y + h / 2},
                        "tags": [label, PROV_LABELLER],
                    }
                )
            )
    _write_text_atomic(path, "\n".join(lines) + "\n")


def _parse_verdict(body: dict) -> tuple[PlayerCheckVerdict, int | None]:
    raw = str(body.get("verdict", ""))
    m = _COUNT_IS.match(raw)
    if m or raw == "count_n":
        count = int(m.group(1)) if m else body.get("count")
        if count is None or int(count) < 0:
            raise HTTPException(422, "count_is_N needs a count >= 0")
        return PlayerCheckVerdict.COUNT_N, int(count)
    try:
        return PlayerCheckVerdict(raw), None
    except ValueError as exc:
        raise HTTPException(
            422, "verdict must be ok | count_is_N | fixed | unsure"
        ) from exc


@router.post("/player-check/verdict")
def verdict(body: dict) -> dict:  # noqa: PLR0912, PLR0915
    clip = safe_clip_name(body.get("clip"))
    guard_split(clip, bool(body.get("eval", False)))
    kind, count = _parse_verdict(body)
    fix = body.get("fix") or []
    if kind == PlayerCheckVerdict.FIXED and not fix:
        raise HTTPException(422, "fixed needs a non-empty fix list")
    by = str(body.get("verified_by") or RUN_ID)
    aspect = float(body.get("aspect", DEFAULT_ASPECT))

    window = body.get("window") or [None, None]
    by_frame = _player_records(clip)
    _frames, tracks = _window(by_frame, window[0], window[1], aspect)  # pre-fix boxes
    by_id = {t.track_id: t for t in tracks}

    sweep_frame = body.get("sweep")
    run_id = None
    frame_scope = body.get("scope") == "frame"
    if frame_scope and (sweep_frame is None or kind not in (PlayerCheckVerdict.OK, PlayerCheckVerdict.FIXED)):
        raise HTTPException(422, "scope=frame needs a sweep frame and an ok or fixed verdict")
    if sweep_frame is not None:
        if kind == PlayerCheckVerdict.COUNT_N:
            raise HTTPException(422, "count_is_N does not apply to a frame sweep")
        cluster_id = SWEEP_CLUSTER_ID
        targets = [int(sweep_frame)]
        if targets[0] not in by_frame:
            raise HTTPException(404, "frame has no player boxes")
    else:
        cluster_id = str(body.get("cluster_id", ""))
        track = by_id.get(cluster_id)
        if track is None:
            raise HTTPException(404, f"unknown cluster {cluster_id!r} in this window")
        lo = int(body["start"]) if body.get("start") is not None else track.frames[0]
        hi = int(body["end"]) if body.get("end") is not None else track.frames[-1]
        targets = [f for f in track.frames if lo <= f <= hi]
        if not targets:
            raise HTTPException(422, "start..end contains no frame of this cluster")
        if kind == PlayerCheckVerdict.COUNT_N:
            run_id = f"{cluster_id}:{targets[0]}-{targets[-1]}"

    if fix:
        _apply_fix(clip, fix)

    # scope=frame: the reviewer looked at the whole frame (every box, numbered), so one verdict covers the frame
    # sweep AND every cluster on it. This is the cheap path for windows whose frames are 10 apart: boxes do not
    # overlap between frames, so each person is a one-frame cluster and per-cluster cards would be ~13 per frame.
    cluster_ids = [cluster_id]
    if frame_scope:
        cluster_ids += sorted({t.track_id for t in tracks for o in t.obs if o.frame == targets[0]})

    now = datetime.now(UTC).replace(tzinfo=None)
    new_rows = [
        PlayerCheckRow(
            game_id=clip,
            frame_index=f,
            cluster_id=cid,
            verdict=kind,
            count=count,
            run_id=run_id,
            verified_by=by,
            verified_at=now,
        )
        for f in targets
        for cid in cluster_ids
    ]
    complete: list[int] = []
    partial: list[int] = []
    needs_fix: list[int] = []
    with _store() as store:
        store.upsert_player_checks(new_rows)
        have = _rows_by_frame(store.read_player_checks(clip, targets))
        state_rows = []
        for f in targets:
            expected = {
                t.track_id: o.count for t in tracks for o in t.obs if o.frame == f
            }
            st = frame_state(expected, have[f])
            if (
                kind == PlayerCheckVerdict.COUNT_N
                and expected.get(cluster_id, count) != count
            ):
                needs_fix.append(f)
            (complete if st == LabelState.COMPLETE else partial).append(f)
            state_rows.append(
                FrameLabelStateRow(
                    game_id=clip,
                    frame_index=f,
                    cls_group=ClsGroup.PLAYER,
                    state=st,
                    method=LabelMethod.HAND_FRAME,
                    run_id=RUN_ID,
                    verified_by=by,
                    verified_at=now,
                )
            )
        store.upsert_frame_label_states(state_rows)
    return {
        "ok": True,
        "cluster_id": cluster_id,
        "frames_written": targets,
        "complete_frames": complete,
        "partial_frames": partial,
        "needs_fix": needs_fix,
    }
