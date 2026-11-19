"""Series review: a human accepts a proposed auto-label window in one go (accept-all + exceptions).

Design: ``docs/design/full_frame_labelling.md`` section 4 stages 4-5 (review, commit), 4b (gold/silver),
requirement ids LAB-1451.. in ``labeller/README.md`` section 19i.

Inputs are what ``autolabel/windows.py`` and ``autolabel/propose.py`` produce, found under
``$FOOTY_AUTOLABEL_DIR`` (default ``$FOOTY_DATA/autolabel``)::

    windows.json                    the window list
    proposals/<window_id>.npz       per-frame disagreement scores
    series_review/<window_id>.json  this module's status sidecar (accepted | rejected)
    reproposals.json                windows rejected and queued for re-propose, with corrected anchors

plus the detection rows (``source`` yolo|vittrack|detector_track|sam3, ``reviewed=False``) in the feature
store (``$FOOTY_FEATURE_STORE``), under the per-source runs ``autolabel:<window_id>:<source>``.

Endpoints::

    GET  /series                          the card page
    GET  /series/windows                  windows + status (proposed | unproposed | accepted | rejected)
    GET  /series/window/{id}              frames: boxes, score, flagged-first, ``audit`` frames boxless
    POST /series/window/{id}/accept       {exceptions:[...], audit_results:[...]} -> commit or reject
    POST /series/window/{id}/reopen       clear a rejected status once the window was re-proposed

Windows of a non-train match/clip (heldout or in-match eval, or unregistered) are refused on every
endpoint (rule zero: re-checked with ``splits.split_of`` here, not trusted from windows.json).
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import math
import os
import random
import tempfile
import time
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

import duckdb
import numpy as np
from fastapi import APIRouter, HTTPException
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from footy_track import splits
from footy_track.autolabel.windows import Window, eligible_match_key, read_windows
from footy_track.constants import BALL_TAG, IN_PLAY_BALL_TAG, OUT_OF_PLAY_BALL_TAG
from footy_track.feature_store.lock import StoreLocked, default_db_path, store_lock
from footy_track.feature_store.schema import (
    ClsGroup,
    FrameLabelStateRow,
    LabelMethod,
    LabelState,
    SeriesRow,
)
from footy_track.feature_store.store import FeatureStore
from footy_track.labeller import hooks
from footy_track.labeller.constants import PROV_LABELLER
from footy_track.labeller.names import safe_clip_name
from footy_track.match_splits import RegisterError

router = APIRouter()
_PAGE = Path(__file__).parent / "web" / "series.html"

BALL_LABELS = frozenset({BALL_TAG, IN_PLAY_BALL_TAG, OUT_OF_PLAY_BALL_TAG})
#: Fraction of a window's frames that may be exceptions, and the blind-audit fraction.
EXCEPTION_LIMIT_FRAC = 0.10
AUDIT_FRAC = 0.10
#: A frame is flagged when its disagreement score is at least this (or a person/ball flag is set).
FLAG_SCORE = 0.3
VSOURCE = "series"  # detection_review.vsource of a box rejected here
VERIFIED_BY = "series_review"
STORE_LOCK_TIMEOUT_S = 5.0

ExceptionKind = Literal["fix_box", "ball_not_ball", "ball_absent", "players_incomplete"]


# --------------------------------------------------------------------------------------------
# Request models
# --------------------------------------------------------------------------------------------
class BoxIn(BaseModel):
    x: float
    y: float
    w: float
    h: float


class ExceptionIn(BaseModel):
    frame_index: int
    kind: ExceptionKind
    # fix_box only: the corrected box (normalised top-left x, y, w, h) and its class label.
    label: str | None = None
    bbox: BoxIn | None = None
    box_index: int | None = (
        None  # sidecar box index to overwrite via /review/correct; None = add
    )


class AuditResultIn(BaseModel):
    """The reviewer relabelled the blind frame; ``ok`` = the machine boxes agreed with them."""

    frame_index: int
    ok: bool
    ball_bbox: BoxIn | None = (
        None  # the human's ball box when the machine was wrong (re-track anchor)
    )


class AcceptBody(BaseModel):
    exceptions: list[ExceptionIn] = Field(default_factory=list)
    audit_results: list[AuditResultIn] = Field(default_factory=list)


# --------------------------------------------------------------------------------------------
# Paths and sidecars
# --------------------------------------------------------------------------------------------
def autolabel_dir() -> Path:
    env = os.environ.get("FOOTY_AUTOLABEL_DIR")
    return Path(env) if env else splits.data_root() / "autolabel"


def _status_path(window_id: str) -> Path:
    return autolabel_dir() / "series_review" / f"{window_id}.json"


def _write_json_atomic(path: Path, doc: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(doc, f, indent=1, sort_keys=True)
        Path(tmp).replace(path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def read_status(window_id: str) -> dict | None:
    p = _status_path(window_id)
    return json.loads(p.read_text()) if p.exists() else None


def reproposal_queue() -> list[dict]:
    """Windows rejected and waiting to be re-proposed (with their corrected anchors)."""
    p = autolabel_dir() / "reproposals.json"
    return json.loads(p.read_text()) if p.exists() else []


# --------------------------------------------------------------------------------------------
# Windows, guards
# --------------------------------------------------------------------------------------------
def _windows() -> dict[str, Window]:
    p = autolabel_dir() / "windows.json"
    return {w.window_id: w for w in read_windows(p)} if p.exists() else {}


def _train_only(w: Window) -> None:
    """Refuse a window whose clip is not provably train (heldout / eval-in-match / unregistered)."""
    try:
        ok = eligible_match_key(w.game_id) is not None
    except RegisterError as e:  # fail closed
        raise HTTPException(503, f"split register unavailable: {e}") from e
    if not ok:
        raise HTTPException(
            403,
            f"window {w.window_id}: clip {w.game_id!r} is not a train-split clip (heldout/eval windows are refused)",
        )


def _get_window(window_id: str) -> Window:
    w = _windows().get(window_id)
    if w is None:
        raise HTTPException(404, f"unknown window {window_id!r}")
    _train_only(w)
    return w


def _is_train(w: Window) -> bool:
    try:
        return eligible_match_key(w.game_id) is not None
    except RegisterError:
        return False


# --------------------------------------------------------------------------------------------
# Store access
# --------------------------------------------------------------------------------------------
@contextlib.contextmanager
def _read_store() -> Iterator[FeatureStore | None]:
    """Read-only store under the shared lock; ``None`` when the store file does not exist yet."""
    db = default_db_path()
    if not db.exists():
        yield None
        return
    try:
        with store_lock(db, exclusive=False, timeout=STORE_LOCK_TIMEOUT_S):
            conn = duckdb.connect(str(db), read_only=True)
            try:
                yield FeatureStore(conn)
            finally:
                conn.close()
    except StoreLocked as e:
        raise HTTPException(503, str(e)) from e


@contextlib.contextmanager
def _write_store() -> Iterator[FeatureStore]:
    db = default_db_path()
    try:
        with (
            store_lock(db, exclusive=True, timeout=STORE_LOCK_TIMEOUT_S),
            FeatureStore.open(db) as st,
        ):
            yield st
    except StoreLocked as e:
        raise HTTPException(503, str(e)) from e


def _run_prefix(w: Window) -> str:
    return f"autolabel:{w.window_id}:"


def _proposing_run(w: Window) -> str:
    return f"autolabel:{w.window_id}"


def _window_rows(store: FeatureStore, w: Window):
    return store.query(
        "SELECT frame_index, detection_id, source, run_id, label, confidence, "
        "bbox_x, bbox_y, bbox_w, bbox_h FROM detection "
        "WHERE game_id = ? AND starts_with(run_id, ?) AND frame_index >= ? AND frame_index < ? "
        "ORDER BY frame_index, source, detection_id",
        [w.game_id, _run_prefix(w), w.start_frame, w.end_frame],
    )


def _scores(w: Window) -> dict[int, dict]:
    """``frame_index -> {score, flag}`` from proposals/<window>.npz (empty when absent)."""
    p = autolabel_dir() / "proposals" / f"{w.window_id}.npz"
    if not p.exists():
        return {}
    with np.load(p) as z:
        fi, score = z["frame_index"], z["score"]
        pf = z["person_flag"] if "person_flag" in z else np.zeros(len(fi), bool)
        bm = (
            z["ball_presence_mismatch"]
            if "ball_presence_mismatch" in z
            else np.zeros(len(fi), bool)
        )
        return {
            int(f): {"score": float(s), "flag": bool(s >= FLAG_SCORE or a or b)}
            for f, s, a, b in zip(fi, score, pf, bm, strict=True)
        }


def audit_frame_set(window_id: str, frames: list[int], flagged: set[int]) -> list[int]:
    """The deterministic 10% blind-audit subset, drawn from the unflagged frames.

    Seeded from the window id only (sha256), so the same window always gets the same audit frames,
    independent of the process, request order or scores.
    """
    pool = sorted(f for f in frames if f not in flagged)
    k = min(len(pool), math.ceil(AUDIT_FRAC * len(frames)))
    seed = int.from_bytes(hashlib.sha256(window_id.encode()).digest()[:8], "big")
    return sorted(random.Random(seed).sample(pool, k))


def exception_limit(n_frames: int) -> int:
    """Most exceptions a window may carry; one more rejects it."""
    return math.floor(EXCEPTION_LIMIT_FRAC * n_frames)


def build_frames(w: Window, store: FeatureStore | None) -> list[dict]:
    """Frame cards, flagged first (descending score) then the rest by frame index; audit boxes hidden."""
    scores = _scores(w)
    boxes: dict[int, list[dict]] = {}
    if store is not None:
        df = _window_rows(store, w)
        for r in df.itertuples(index=False):
            boxes.setdefault(int(r.frame_index), []).append(
                {
                    "detection_id": int(r.detection_id),
                    "source": r.source,
                    "label": r.label,
                    "confidence": None if r.confidence is None or r.confidence != r.confidence else float(r.confidence),
                    "x": float(r.bbox_x), "y": float(r.bbox_y), "w": float(r.bbox_w), "h": float(r.bbox_h),
                }
            )  # fmt: skip
    frames = list(range(w.start_frame, w.end_frame))
    flagged = {f for f in frames if scores.get(f, {}).get("flag")}
    audit = set(audit_frame_set(w.window_id, frames, flagged))
    out = [
        {
            "frame_index": f,
            "score": scores.get(f, {}).get("score", 0.0),
            "flagged": f in flagged,
            "audit": f in audit,
            "boxes": [] if f in audit else boxes.get(f, []),
            "image": f"/review/frame/{w.game_id}/{f}.jpg",
        }
        for f in frames
    ]
    out.sort(
        key=lambda c: (
            not c["flagged"],
            -c["score"] if c["flagged"] else 0.0,
            c["frame_index"],
        )
    )
    return out


# --------------------------------------------------------------------------------------------
# Read endpoints
# --------------------------------------------------------------------------------------------
@router.get("/series", response_class=HTMLResponse)
async def series_page() -> HTMLResponse:
    return HTMLResponse(
        _PAGE.read_text(), headers={"Cache-Control": "no-store, max-age=0"}
    )


def _window_meta(w: Window, status: str) -> dict:
    n = w.end_frame - w.start_frame
    return {
        "window_id": w.window_id, "game_id": w.game_id, "clip_id": w.clip_id,
        "start_frame": w.start_frame, "end_frame": w.end_frame, "n_frames": n,
        "status": status, "exception_limit": exception_limit(n),
    }  # fmt: skip


@router.get("/series/windows")
def list_windows() -> dict:
    """Train-split windows with a status: unproposed | proposed | accepted | rejected."""
    wins = [w for w in _windows().values() if _is_train(w)]
    proposed: set[str] = set()
    with _read_store() as st:
        if st is not None:
            df = st.query(
                "SELECT DISTINCT split_part(run_id, ':', 2) AS wid FROM detection WHERE starts_with(run_id, 'autolabel:')"
            )
            proposed = set(df["wid"])
    out = []
    for w in wins:
        sc = read_status(w.window_id)
        status = (
            sc["status"]
            if sc
            else ("proposed" if w.window_id in proposed else "unproposed")
        )
        out.append(_window_meta(w, status))
    return {"windows": out}


@router.get("/series/window/{window_id}")
def get_window(window_id: str) -> dict:
    w = _get_window(window_id)
    sc = read_status(window_id)
    with _read_store() as st:
        frames = build_frames(w, st)
    has_boxes = any(c["boxes"] for c in frames)
    status = sc["status"] if sc else ("proposed" if has_boxes else "unproposed")
    return {
        **_window_meta(w, status),
        "run_id": _proposing_run(w),
        "audit_frames": sorted(c["frame_index"] for c in frames if c["audit"]),
        "frames": frames,
        "result": sc,
    }


@router.post("/series/window/{window_id}/reopen")
def reopen_window(window_id: str) -> dict:
    """Clear a *rejected* status (after the window was re-proposed) so it can be reviewed again."""
    _get_window(window_id)
    sc = read_status(window_id)
    if sc is None or sc["status"] != "rejected":
        raise HTTPException(409, "only a rejected window can be reopened")
    _status_path(window_id).unlink()
    q = [r for r in reproposal_queue() if r["window_id"] != window_id]
    _write_json_atomic(autolabel_dir() / "reproposals.json", q)
    return {"window_id": window_id, "status": "proposed"}


# --------------------------------------------------------------------------------------------
# Accept / reject
# --------------------------------------------------------------------------------------------
def _validate(w: Window, body: AcceptBody, audit: list[int]) -> dict[int, ExceptionIn]:
    in_window = range(w.start_frame, w.end_frame)
    exc: dict[int, ExceptionIn] = {}
    for e in body.exceptions:
        if e.frame_index not in in_window:
            raise HTTPException(
                422, f"exception frame {e.frame_index} is outside the window"
            )
        if e.frame_index in exc:
            raise HTTPException(422, f"two exceptions for frame {e.frame_index}")
        if e.frame_index in audit:
            raise HTTPException(
                422,
                f"frame {e.frame_index} is a blind-audit frame: report it in audit_results",
            )
        if e.kind == "fix_box" and (e.bbox is None or not e.label):
            raise HTTPException(422, "fix_box needs label and bbox")
        exc[e.frame_index] = e
    got = [r.frame_index for r in body.audit_results]
    if sorted(got) != sorted(set(got)) or set(got) != set(audit):
        raise HTTPException(
            422, f"audit_results must cover exactly the audit frames {audit}"
        )
    return exc


def _anchors(body: AcceptBody) -> list[dict]:
    """Corrected anchors the re-propose should use as re-track seeds."""
    out: list[dict] = []
    for e in body.exceptions:
        a: dict = {"frame_index": e.frame_index, "kind": e.kind}
        if e.bbox is not None:
            a |= {"label": e.label, "bbox": e.bbox.model_dump()}
        out.append(a)
    for r in body.audit_results:
        if not r.ok:
            a = {"frame_index": r.frame_index, "kind": "audit_failed"}
            if r.ball_bbox is not None:
                a |= {"label": BALL_TAG, "bbox": r.ball_bbox.model_dump()}
            out.append(a)
    return sorted(out, key=lambda a: a["frame_index"])


def _reject(w: Window, body: AcceptBody, reason: str) -> dict:
    """Record the rejection + corrected anchors and queue the window for re-propose. No state is written."""
    n = w.end_frame - w.start_frame
    doc = {
        "window_id": w.window_id, "game_id": w.game_id, "status": "rejected", "reason": reason,
        "n_frames": n, "n_exceptions": len(body.exceptions),
        "audit_failed": sorted(r.frame_index for r in body.audit_results if not r.ok),
        "anchors": _anchors(body), "rejected_at": datetime.now(UTC).isoformat(),
        "reproposal_queued": True,
    }  # fmt: skip
    _write_json_atomic(_status_path(w.window_id), doc)
    q = [r for r in reproposal_queue() if r["window_id"] != w.window_id]
    q.append({k: doc[k] for k in ("window_id", "game_id", "reason", "anchors")}
             | {"start_frame": w.start_frame, "end_frame": w.end_frame, "queued_at": doc["rejected_at"]})  # fmt: skip
    _write_json_atomic(autolabel_dir() / "reproposals.json", q)
    return doc


async def _write_fix_boxes(w: Window, exc: dict[int, ExceptionIn]) -> None:
    """Fixed boxes go through the existing /review/correct sidecar path (becoming hand_label on ingest)."""
    from footy_track.labeller import review  # noqa: PLC0415

    clip = safe_clip_name(w.game_id)
    for e in exc.values():
        if e.kind != "fix_box":
            continue
        bbox = e.bbox.model_dump()
        if e.box_index is not None:
            res = await review.review_correct(
                {
                    "clip": clip,
                    "frame_index": e.frame_index,
                    "box_index": e.box_index,
                    "label": e.label,
                    "bbox": bbox,
                }
            )
            if not res.get("ok"):
                raise HTTPException(
                    409, f"fix_box frame {e.frame_index}: {res.get('error')}"
                )
        else:
            _append_sidecar_box(clip, e.frame_index, e.label, bbox)
        hooks.emit("ball_place", clip=clip, unit=e.frame_index, verdict="corrected")


def _append_sidecar_box(clip: str, frame_index: int, label: str, bbox: dict) -> None:
    from footy_track.labeller import review  # noqa: PLC0415

    marks = review._gt_marks_dir()  # noqa: SLF001 - same package
    path = marks / f"{clip}.jsonl"
    lines = path.read_text().splitlines() if path.exists() else []
    x = max(0.0, min(1.0, bbox["x"]))
    y = max(0.0, min(1.0, bbox["y"]))
    w = max(0.0, min(1.0 - x, bbox["w"]))
    h = max(0.0, min(1.0 - y, bbox["h"]))
    lines.append(
        json.dumps(
            {
                "frame_index": frame_index,
                "bbox": {"x": x, "y": y, "w": w, "h": h},
                "center": {"x": x + w / 2, "y": y + h / 2},
                "tags": [label, PROV_LABELLER],
            }
        )
    )
    marks.mkdir(parents=True, exist_ok=True)
    review._write_text_atomic(path, "\n".join(lines) + "\n")  # noqa: SLF001


def _commit(w: Window, exc: dict[int, ExceptionIn]) -> dict:
    """Write reviewed flags, rejected-box reviews, frame_label_state and the series row (one transaction)."""
    gid, prefix, run = w.game_id, _run_prefix(w), _proposing_run(w)
    frames = range(w.start_frame, w.end_frame)
    now = datetime.now(UTC)
    with _write_store() as st:
        df = _window_rows(st, w)
        if df.empty:
            raise HTTPException(
                409, f"window {w.window_id} has no proposed detections in the store"
            )
        ball_frames = {
            int(f) for f in df.loc[df["label"].isin(BALL_LABELS), "frame_index"]
        }
        player_frames = {
            int(f) for f in df.loc[~df["label"].isin(BALL_LABELS), "frame_index"]
        }
        conn = st._conn  # noqa: SLF001 - UPDATE / detection_review insert, same package
        rejected_balls: list[tuple] = []
        states: list[FrameLabelStateRow] = []

        def state(fi: int, grp: ClsGroup, s: LabelState, method: LabelMethod) -> None:
            states.append(FrameLabelStateRow(game_id=gid, frame_index=fi, cls_group=grp, state=s,
                                             method=method, run_id=run, verified_by=VERIFIED_BY,
                                             verified_at=now))  # fmt: skip

        with st.transaction():
            # 1. accept every machine box of the window, then take back the ones exceptions touch.
            conn.execute(
                "UPDATE detection SET reviewed = TRUE, needs_review = FALSE WHERE game_id = ? "
                "AND starts_with(run_id, ?) AND frame_index >= ? AND frame_index < ?",
                [gid, prefix, w.start_frame, w.end_frame],
            )
            for fi in frames:
                e = exc.get(fi)
                kind = e.kind if e else None
                m = LabelMethod.HAND_FRAME if e else LabelMethod.HAND_SERIES
                has_ball, has_player = fi in ball_frames, fi in player_frames
                # ball group
                if kind in ("ball_not_ball", "ball_absent") or (
                    kind == "fix_box" and e.label in BALL_LABELS
                ):
                    conn.execute(
                        "UPDATE detection SET reviewed = FALSE, needs_review = TRUE WHERE game_id = ? AND starts_with(run_id, ?) "
                        "AND frame_index = ? AND label IN (SELECT unnest(?))",
                        [gid, prefix, fi, sorted(BALL_LABELS)],
                    )
                if kind in ("ball_not_ball", "ball_absent"):
                    rejected_balls += list(
                        df[
                            (df["frame_index"] == fi) & df["label"].isin(BALL_LABELS)
                        ].itertuples(index=False)
                    )
                    state(
                        fi,
                        ClsGroup.BALL,
                        LabelState.UNREVIEWED
                        if kind == "ball_not_ball"
                        else LabelState.ABSENT,
                        m,
                    )
                elif kind == "fix_box" and e.label in BALL_LABELS:
                    state(
                        fi, ClsGroup.BALL, LabelState.COMPLETE, m
                    )  # the human drew it
                else:
                    state(
                        fi,
                        ClsGroup.BALL,
                        LabelState.COMPLETE if has_ball else LabelState.ABSENT,
                        m,
                    )
                # player group
                if kind == "fix_box" and e.label not in BALL_LABELS:
                    conn.execute(
                        "UPDATE detection SET reviewed = FALSE, needs_review = TRUE WHERE game_id = ? AND starts_with(run_id, ?) "
                        "AND frame_index = ? AND label NOT IN (SELECT unnest(?))",
                        [gid, prefix, fi, sorted(BALL_LABELS)],
                    )
                if kind in ("players_incomplete", "fix_box") and not (
                    kind == "fix_box" and e.label in BALL_LABELS
                ):
                    state(fi, ClsGroup.PLAYER, LabelState.PARTIAL, m)
                else:
                    state(
                        fi,
                        ClsGroup.PLAYER,
                        LabelState.COMPLETE if has_player else LabelState.ABSENT,
                        m,
                    )
            # 2. rejected ball boxes: detection_review semantics of Ball Check (verdict not_ball, matched to the box)
            ts = time.time()
            for r in rejected_balls:
                conn.execute(
                    "INSERT OR REPLACE INTO detection_review VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    [gid, VSOURCE, int(r.frame_index), int(r.detection_id), "not_ball", ts, r.label,
                     float(r.bbox_x), float(r.bbox_y), float(r.bbox_w), float(r.bbox_h),
                     r.source, r.run_id, int(r.detection_id), 1.0],
                )  # fmt: skip
            st.upsert_frame_label_states(states)
            # 3. register as a series only when every frame is complete
            done = int(
                st.query(
                    "SELECT count(*) AS n FROM frame_complete WHERE game_id = ? AND frame_index >= ? AND frame_index < ?",
                    [gid, w.start_frame, w.end_frame],
                )["n"][0]
            )
            series = done == w.end_frame - w.start_frame
            if series:
                st.register_series([SeriesRow(game_id=gid, start_frame=w.start_frame, end_frame=w.end_frame,
                                              run_id=run, verified_at=now)])  # fmt: skip
    return {
        "frames_complete": done,
        "series_registered": series,
        "rejected_ball_boxes": len(rejected_balls),
    }


@router.post("/series/window/{window_id}/accept")
async def accept_window(window_id: str, body: AcceptBody) -> dict:
    w = _get_window(window_id)
    prior = read_status(window_id)
    if prior is not None:
        if prior["status"] == "accepted":  # idempotent re-accept: nothing is rewritten
            return {**prior, "idempotent": True}
        raise HTTPException(
            409,
            f"window {window_id} was rejected and is queued for re-propose (reopen it first)",
        )
    n = w.end_frame - w.start_frame
    with _read_store() as st:
        frames = build_frames(w, st)
    audit = sorted(c["frame_index"] for c in frames if c["audit"])
    exc = _validate(w, body, audit)

    failed = sorted(r.frame_index for r in body.audit_results if not r.ok)
    if len(exc) > exception_limit(n):
        return _reject(
            w,
            body,
            f"{len(exc)} exceptions exceed the {exception_limit(n)}-frame ({EXCEPTION_LIMIT_FRAC:.0%}) limit",
        )
    if failed:
        return _reject(w, body, f"blind audit failed on frames {failed}")

    await _write_fix_boxes(w, exc)
    stats = await run_in_threadpool(_commit, w, exc)
    doc = {
        "window_id": w.window_id, "game_id": w.game_id, "status": "accepted", "run_id": _proposing_run(w),
        "method": LabelMethod.HAND_SERIES.value, "n_frames": n, "n_exceptions": len(exc),
        "exceptions": {str(f): e.kind for f, e in sorted(exc.items())},
        "audit": {"n": len(audit), "failed": 0}, "accepted_at": datetime.now(UTC).isoformat(), **stats,
    }  # fmt: skip
    _write_json_atomic(_status_path(window_id), doc)
    return {**doc, "idempotent": False}
