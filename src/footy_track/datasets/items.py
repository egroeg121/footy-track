"""Canonical item tables: one row per labelled item, assembled from the feature store.

Item granularity (see ``docs/design/datasets_and_progress.md``):

``tracking``     one labelled *frame* of a clip: its boxes (players, referees, ball, ...) with
                 provenance tier, the Ball Check verdicts on that frame, and the sidecar's skip
                 marker (``no_ball`` / ``not_broadcast``). ``item_id = <game_id>:<frame>``.
``broadcast``    one human *frame verdict* (yes / no / unsure) with the classifier's stored call.
                 ``item_id = <video stem>:<frame>``.
``calibration``  one *frame correspondence set*: the human-placed image <-> pitch point pairs and
                 the homography fitted from them, or a human flag that the frame cannot be
                 calibrated (``not_broadcast`` / ``too_few_points``).
                 ``item_id = <canonical clip>:<frame>``.

Every item carries its ``split`` (from :func:`footy_track.splits.split_of`, never stored in the
store: it would go stale as the register grows), ``match_key``, canonical ``clip`` and a
canonical JSON ``payload``. Nothing time- or path-dependent goes into an item, so the same labels
always give the same items and hashes.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from footy_track import splits
from footy_track.clip_id import canonical_clip_id
from footy_track.feature_store.ball_checks import BALL_LABELS
from footy_track.feature_store.ingest_calibration import (
    FRAME_TYPE_DATASET_TAG,
    GT_DATASET_TAG,
)
from footy_track.feature_store.ingest_gt import DATASET_TAG as TRACKING_GT_TAG
from footy_track.feature_store.ingest_gt import GT_RUN_PREFIX
from footy_track.match_splits import RegisterError

DATASETS = ("tracking", "broadcast", "calibration")
#: bump when the item table's shape or the hash canonicalisation changes (forces a new version)
ITEM_SCHEMA = 1
FLOAT_DP = 6


class DatasetBuildError(RuntimeError):
    """The item table cannot be assembled (missing split register, empty store, ...)."""


@dataclass(frozen=True)
class Item:
    item_id: str
    split: str
    match_key: str | None
    clip: str
    game_id: str
    frame_index: int
    label_kind: str
    tier: int | None
    payload: str  # canonical JSON


def canon(obj: Any) -> Any:
    """Round floats and sort dict keys' view of ``obj`` (JSON-ready, deterministic)."""
    if isinstance(obj, float):
        return round(obj, FLOAT_DP)
    if isinstance(obj, dict):
        return {k: canon(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [canon(v) for v in obj]
    return obj


def canonical_json(obj: Any) -> str:
    return json.dumps(canon(obj), sort_keys=True, separators=(",", ":"), ensure_ascii=True)


class _SplitCache:
    """``split_of`` memo over the (game | clip) names of one build."""

    def __init__(self) -> None:
        self._memo: dict[tuple[str, str | None], tuple[str | None, str, str]] = {}

    def __call__(self, name: str, clip: str | None = None) -> tuple[str | None, str, str]:
        key = (name, clip)
        if key not in self._memo:
            try:
                match = splits.match_key_of(name)
                cid = clip or canonical_clip_id(name)
                self._memo[key] = (match, cid, splits.split_of(match, cid))
            except RegisterError as e:
                raise DatasetBuildError(
                    f"{e}. A dataset version needs the real split registers; for a read-only "
                    "look set FOOTY_CLIP_SPLITS_PREVIEW=1 (never bump a version from a preview)."
                ) from e
        return self._memo[key]


def _tier_of(source: str, reviewed: bool) -> int:
    """1 = human-placed, 2 = machine box a human confirmed, 3 = machine only."""
    if source == "hand_label":
        return 1
    return 2 if reviewed else 3


def _tables(conn: Any) -> set[str]:
    return {r[0] for r in conn.execute("SELECT table_name FROM information_schema.tables").fetchall()}


# --------------------------------------------------------------------------------------------
# tracking
# --------------------------------------------------------------------------------------------
def _tracking_items(conn: Any) -> list[Item]:
    have = _tables(conn)
    boxes: dict[tuple[str, int], list[dict]] = {}
    rows = conn.execute(
        """
        SELECT d.game_id, d.frame_index, d.source, d.label, d.reviewed,
               d.bbox_x, d.bbox_y, d.bbox_w, d.bbox_h,
               EXISTS (SELECT 1 FROM detection_review r
                        WHERE r.game_id = d.game_id AND r.verdict = 'not_ball'
                          AND r.matched_source = d.source AND r.matched_run_id = d.run_id
                          AND r.matched_detection_id = d.detection_id
                          AND r.frame_index = d.frame_index) AS rejected
        FROM detection d
        WHERE d.dataset_tag = ? AND starts_with(d.run_id, ?)
        """,
        [TRACKING_GT_TAG, GT_RUN_PREFIX],
    ).fetchall()
    for game, fi, source, label, reviewed, x, y, w, h, rejected in rows:
        boxes.setdefault((game, fi), []).append(
            {
                "label": label,
                "source": source,
                "tier": _tier_of(source, bool(reviewed)),
                "rejected": bool(rejected),
                "x": x,
                "y": y,
                "w": w,
                "h": h,
            }
        )
    checks: dict[tuple[str, int], list[dict]] = {}
    for game, vsource, fi, bi, verdict, x, y, w, h in conn.execute(
        "SELECT game_id, vsource, frame_index, box_index, verdict, bbox_x, bbox_y, bbox_w, bbox_h "
        "FROM detection_review"
    ).fetchall():
        checks.setdefault((game, fi), []).append(
            {"vsource": vsource, "box_index": bi, "verdict": verdict, "x": x, "y": y, "w": w, "h": h}
        )
    markers: dict[tuple[str, int], str | None] = {}
    if "gt_frame" in have:
        markers = {
            (g, f): m for g, f, m in conn.execute("SELECT game_id, frame_index, marker FROM gt_frame").fetchall()
        }
    keys = sorted(set(boxes) | set(checks) | set(markers))
    split_of = _SplitCache()
    items: list[Item] = []
    for game, fi in keys:
        bx = sorted(
            boxes.get((game, fi), []),
            key=lambda b: (b["source"], b["label"], b["x"], b["y"], b["w"], b["h"]),
        )
        ck = sorted(checks.get((game, fi), []), key=lambda c: (c["vsource"], c["box_index"]))
        marker = markers.get((game, fi))
        live_ball = [b for b in bx if b["label"].lower() in BALL_LABELS and not b["rejected"]]
        if marker == "not_broadcast":
            kind = "not_broadcast"
        elif live_ball:
            kind = "ball"
        elif marker == "no_ball":
            kind = "no_ball"
        elif bx:
            kind = "other_boxes"
        else:
            kind = "check_only" if ck else "empty"
        pool = live_ball or bx
        tier = min((b["tier"] for b in pool), default=None)
        match, clip, split = split_of(game)
        items.append(
            Item(
                item_id=f"{game}:{fi}",
                split=split,
                match_key=match,
                clip=clip,
                game_id=game,
                frame_index=int(fi),
                label_kind=kind,
                tier=tier,
                payload=canonical_json({"boxes": bx, "checks": ck, "marker": marker}),
            )
        )
    return items


# --------------------------------------------------------------------------------------------
# broadcast
# --------------------------------------------------------------------------------------------
def _broadcast_items(conn: Any) -> list[Item]:
    if "broadcast_label" not in _tables(conn):
        return []
    split_of = _SplitCache()
    items: list[Item] = []
    for stem, fi, verdict, p_yes, pred, reason, ts_s in conn.execute(
        "SELECT video_stem, frame_index, verdict, p_yes, pred, reason, ts_s "
        "FROM broadcast_label ORDER BY video_stem, frame_index"
    ).fetchall():
        match = splits.match_key_of(stem)
        window = splits.broadcast_window_id(match or stem, fi)
        _, _, split = split_of(match or stem, window)
        items.append(
            Item(
                item_id=f"{stem}:{fi}",
                split=split,
                match_key=match,
                clip=window,
                game_id=stem,
                frame_index=int(fi),
                label_kind=verdict,
                tier=1,
                payload=canonical_json(
                    {"verdict": verdict, "p_yes": p_yes, "pred": pred, "reason": reason, "ts_s": ts_s}
                ),
            )
        )
    return items


# --------------------------------------------------------------------------------------------
# calibration
# --------------------------------------------------------------------------------------------
def _calibration_items(conn: Any) -> list[Item]:
    kps: dict[tuple[str, int], list[dict]] = {}
    for game, fi, kid, u, v, px, py in conn.execute(
        "SELECT game_id, frame_index, keypoint_id, u, v, pitch_x, pitch_y FROM pitch_keypoint "
        "WHERE dataset_tag = ? ORDER BY game_id, frame_index, point_idx",
        [GT_DATASET_TAG],
    ).fetchall():
        kps.setdefault((game, fi), []).append(
            {"pitch_id": kid, "image": [u, v], "pitch_xy": [px, py]}
        )
    rows = conn.execute(
        """
        SELECT game_id, frame_index, dataset_tag, review_status, H, h_convention, distortion_k1,
               image_width, image_height
        FROM calibration WHERE dataset_tag IN (?, ?)
        """,
        [GT_DATASET_TAG, FRAME_TYPE_DATASET_TAG],
    ).fetchall()
    # One item per (canonical clip, frame): the importer copies a file to every alias game_id.
    # A frame that is both calibrated and flagged keeps the calibration (the human GT).
    chosen: dict[tuple[str, int], tuple] = {}
    for row in rows:
        game, fi, tag = row[0], row[1], row[2]
        key = (canonical_clip_id(game), fi)
        prev = chosen.get(key)
        rank = (0 if tag == GT_DATASET_TAG else 1, game)
        if prev is None or rank < (0 if prev[2] == GT_DATASET_TAG else 1, prev[0]):
            chosen[key] = row
    split_of = _SplitCache()
    items: list[Item] = []
    for (clip, fi), row in sorted(chosen.items()):
        game, _, tag, status, H, conv, k1, iw, ih = row
        has_h = H is not None
        payload = {
            "status": status,
            "h_convention": conv if has_h else None,
            "H": list(H) if has_h else None,
            "distortion_k1": k1,
            "image_size": [iw, ih] if iw and ih else None,
            "correspondences": kps.get((game, fi), []) if tag == GT_DATASET_TAG else [],
        }
        match, cid, split = split_of(clip, clip)
        items.append(
            Item(
                item_id=f"{clip}:{fi}",
                split=split,
                match_key=match,
                clip=cid,
                game_id=game,
                frame_index=int(fi),
                label_kind="homography" if has_h else status,
                tier=1,
                payload=canonical_json(payload),
            )
        )
    return items


_BUILDERS: dict[str, Callable[[Any], list[Item]]] = {
    "tracking": _tracking_items,
    "broadcast": _broadcast_items,
    "calibration": _calibration_items,
}


def build_items(dataset: str, conn: Any) -> list[Item]:
    """The full item table of ``dataset`` from an open DuckDB connection, sorted by ``item_id``."""
    if dataset not in _BUILDERS:
        raise DatasetBuildError(f"unknown dataset {dataset!r}; choose from {DATASETS}")
    items = _BUILDERS[dataset](conn)
    ids = [i.item_id for i in items]
    if len(ids) != len(set(ids)):
        raise DatasetBuildError(f"{dataset}: duplicate item ids (builder bug)")
    return sorted(items, key=lambda i: i.item_id)
