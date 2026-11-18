"""The pool table: one row per *candidate node* of a train match (design ``ball_sampling.md`` §3.1).

A node is one place in one frame where the ball might be: a detector box, a SAM2 gap-fill, a tracker
interpolation. The Prefect flows ``ball-candidates`` + ``ball-pool-score`` will write
``ball_pool/<match>/pool.<scorer>.parquet`` in this schema; until they exist,
:func:`pool_from_machine_labels` adapts the RT-DETR jsonl (single source, ``p_real`` = raw confidence).

Coordinates
-----------
``x``, ``y``: ball **centre**, **normalised** to the frame (0..1 of width / height, origin top-left).
``w``, ``h``: box size, same normalisation. Pixel quantities (gold shifts of 15-30 px, static radii) use the
miner's ``frame_wh`` config (default 1920x1080, every broadcast segment so far). ``frame`` is the clip-local
frame index at the clip's native rate ``fps`` (25 or 50 today); time inside the clip is ``frame / fps``.

Columns
-------
=================  ==============  ====================================================================
column             type            meaning
=================  ==============  ====================================================================
match              string          match key as in ``match_splits.json`` (the split gate keys on it)
clip               string          clip stem as the labeller / detector names it (canonicalised on read)
frame              int64           clip-local frame index
fps                float32         clip frame rate (25.0 / 50.0)
x, y, w, h         float32         centre and size, normalised 0..1
track_id           int64, null     tracker track the node sits on
sources            list<string>    candidate sources that proposed it: rtdetr, sam2, tracker_gap, ...
source_scores      list<float32>   one score per source (same length as ``sources``), 0..1
n_sources          int32           ``len(sources)``
solver_post        float32, null   solver (Viterbi / pool) posterior of the node being on the best path
sam2_agree         bool, null      SAM2 propagation agrees with the node
plausible          bool, null      trajectory plausibility (accel / jump limits); False = off any plausible track
static_flag        bool, null      stationary object (spare ball by the hoardings)
gap_frame          bool, null      the tracker interpolated here with no detection (a "missed" candidate)
embedding          list<float32>, null  appearance descriptor (72-d today), for diversity
p_real             float32         P(node is the real ball) from the scorer, 0..1
scorer_version     string          which scorer produced ``p_real`` (``raw_conf`` for the adapter)
detector_sha       string          detector checkpoint hash the candidates came from
=================  ==============  ====================================================================

Optional columns may be absent from a file or all-null. ``null`` booleans mean "unknown", not False.
"""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from footy_track.clip_id import canonical_clip_id

SCHEMA_VERSION = "ball-pool/v1"

#: Detector classes that are the ball in ``machine_labels`` rows.
BALL_TAGS = ("ball", "in_play_ball", "out_of_play_ball")

ARROW_SCHEMA = pa.schema(
    [
        ("match", pa.string()),
        ("clip", pa.string()),
        ("frame", pa.int64()),
        ("fps", pa.float32()),
        ("x", pa.float32()),
        ("y", pa.float32()),
        ("w", pa.float32()),
        ("h", pa.float32()),
        ("track_id", pa.int64()),
        ("sources", pa.list_(pa.string())),
        ("source_scores", pa.list_(pa.float32())),
        ("n_sources", pa.int32()),
        ("solver_post", pa.float32()),
        ("sam2_agree", pa.bool_()),
        ("plausible", pa.bool_()),
        ("static_flag", pa.bool_()),
        ("gap_frame", pa.bool_()),
        ("embedding", pa.list_(pa.float32())),
        ("p_real", pa.float32()),
        ("scorer_version", pa.string()),
        ("detector_sha", pa.string()),
    ],
    metadata={b"schema": SCHEMA_VERSION.encode()},
)
COLUMNS = tuple(ARROW_SCHEMA.names)
REQUIRED = (
    "match", "clip", "frame", "fps", "x", "y", "w", "h", "sources", "source_scores", "n_sources",
    "p_real", "scorer_version", "detector_sha",
)  # fmt: skip
OPTIONAL = tuple(c for c in COLUMNS if c not in REQUIRED)
_UNIT_COLS = ("x", "y", "w", "h", "p_real", "solver_post")
_BOOL_COLS = ("sam2_agree", "plausible", "static_flag", "gap_frame")


class PoolSchemaError(ValueError):
    """The pool table does not match the schema (``problems`` lists every violation found)."""

    def __init__(self, problems: list[str]) -> None:
        self.problems = problems
        super().__init__(
            "; ".join(problems[:8])
            + (f" (+{len(problems) - 8} more)" if len(problems) > 8 else "")
        )


def empty_pool() -> pd.DataFrame:
    return pd.DataFrame({c: pd.Series(dtype=object) for c in COLUMNS})


def problems(df: pd.DataFrame) -> list[str]:  # noqa: PLR0912
    """Every schema violation of ``df`` (empty list = valid). Cheap enough for 1M rows."""
    out: list[str] = []
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        return [f"missing required column(s): {missing}"]
    extra = [c for c in df.columns if c not in COLUMNS]
    if extra:
        out.append(f"unknown column(s): {extra}")
    n = len(df)
    if n == 0:
        return out
    for c in ("match", "clip", "scorer_version", "detector_sha"):
        if df[c].isna().any() or (df[c].astype(str) == "").any():
            out.append(f"{c}: empty or null values")
    for c in ("frame",):
        if df[c].isna().any() or (pd.to_numeric(df[c], errors="coerce") < 0).any():
            out.append(f"{c}: null or negative")
    if not (pd.to_numeric(df["fps"], errors="coerce") > 0).all():
        out.append("fps: must be > 0")
    for c in _UNIT_COLS:
        if c not in df.columns:
            continue
        v = pd.to_numeric(df[c], errors="coerce")
        required = c in REQUIRED
        bad = (v.isna() & required) | (v.notna() & ((v < 0) | (v > 1)))
        if bad.any():
            out.append(
                f"{c}: {int(bad.sum())} value(s) null/outside 0..1 (coordinates are normalised)"
            )
    for c in ("w", "h"):
        if (pd.to_numeric(df[c], errors="coerce") <= 0).any():
            out.append(f"{c}: must be > 0")
    for c in _BOOL_COLS:
        if c in df.columns:
            bad = ~df[c].map(
                lambda v: v is None
                or v is pd.NA
                or isinstance(v, (bool, np.bool_))
                or (isinstance(v, float) and np.isnan(v))
            )
            if bad.any():
                out.append(f"{c}: must be bool or null")
    srcs = df["sources"].map(
        lambda v: len(v) if v is not None and not isinstance(v, float) else -1
    )
    scores = df["source_scores"].map(
        lambda v: len(v) if v is not None and not isinstance(v, float) else -1
    )
    if (srcs < 1).any():
        out.append("sources: every node needs at least one source")
    if (srcs != scores).any():
        out.append("source_scores: must have one score per source")
    if (pd.to_numeric(df["n_sources"], errors="coerce") != srcs).any():
        out.append("n_sources: must equal len(sources)")
    bad_scores = df["source_scores"].map(
        lambda v: v is not None
        and not isinstance(v, float)
        and len(v) > 0
        and (min(v) < 0 or max(v) > 1)
    )
    if bad_scores.any():
        out.append("source_scores: values must be in 0..1")
    if "embedding" in df.columns:
        lens = df["embedding"].map(
            lambda v: -1 if v is None or isinstance(v, float) else len(v)
        )
        known = lens[lens >= 0]
        if len(known) and known.nunique() > 1:
            out.append("embedding: inconsistent dimensions")
    return out


def validate_pool(df: pd.DataFrame) -> pd.DataFrame:
    """Raise :class:`PoolSchemaError` on any violation; return ``df`` with every optional column present."""
    p = problems(df)
    if p:
        raise PoolSchemaError(p)
    df = df.copy()
    for c in OPTIONAL:
        if c not in df.columns:
            df[c] = None
    return df


def _to_arrow(df: pd.DataFrame) -> pa.Table:
    cols = {}
    for f in ARROW_SCHEMA:
        s = df[f.name] if f.name in df.columns else pd.Series([None] * len(df))
        if pa.types.is_list(f.type):
            vals = [
                None
                if (v is None or (isinstance(v, float) and np.isnan(v)))
                else [float(t) if f.name != "sources" else t for t in v]
                for v in s
            ]
            cols[f.name] = pa.array(vals, type=f.type)
        elif pa.types.is_boolean(f.type):
            cols[f.name] = pa.array(
                [
                    None
                    if (
                        v is None
                        or v is pd.NA
                        or (isinstance(v, float) and np.isnan(v))
                    )
                    else bool(v)
                    for v in s
                ],
                type=f.type,
            )
        elif pa.types.is_integer(f.type):
            cols[f.name] = pa.array(
                pd.array(s, dtype="Int64"), type=f.type, from_pandas=True
            )
        else:
            cols[f.name] = pa.array(s, type=f.type, from_pandas=True)
    return pa.table(cols, schema=ARROW_SCHEMA)


def write_pool(df: pd.DataFrame, path: str | Path) -> Path:
    """Validate and write ``df`` as Parquet (``ball_pool/<match>/pool.<scorer>.parquet`` by convention)."""
    df = validate_pool(df)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(_to_arrow(df), path, compression="zstd")
    return path


def read_pool(path: str | Path, *, validate: bool = True) -> pd.DataFrame:
    """Read one Parquet pool, or every ``*.parquet`` below a directory (concatenated)."""
    path = Path(path)
    files = sorted(path.rglob("*.parquet")) if path.is_dir() else [path]
    if not files:
        raise FileNotFoundError(f"no pool parquet under {path}")
    frames = [pq.read_table(f).to_pandas() for f in files]
    df = pd.concat(frames, ignore_index=True) if len(frames) > 1 else frames[0]
    return validate_pool(df) if validate else df


# ---------------------------------------------------------------------------
# Adapter: RT-DETR machine_labels / ball_candidates jsonl -> pool
# ---------------------------------------------------------------------------


def _match_of(clip: str) -> str:
    """Match key of a clip: the split register's answer, else the canonical id minus ``_segNNN`` / ``_partNNN``."""
    from footy_track import splits  # noqa: PLC0415

    key = splits.match_key_of(clip)
    if key:
        return key
    cid = canonical_clip_id(clip)
    return cid.rsplit("_seg", 1)[0].rsplit("_part", 1)[0]


def derive_static_flags(
    df: pd.DataFrame, *, radius: float = 0.004, min_frames: int = 25
) -> pd.Series:
    """Heuristic ``static_flag``: a node whose position (grid cell of ``radius``) is occupied in at least
    ``min_frames`` distinct frames of its clip (a spare ball parked by the hoardings, a corner-flag blob).

    The wiki measured the static-position filter as weak for *dropping* candidates (a resting in-play ball looks
    the same), so it is only used to *select hard-negative cards* (bucket N), never to discard nodes."""
    if df.empty:
        return pd.Series([], dtype=bool, index=df.index)
    gx = np.floor(df["x"].to_numpy(dtype=float) / radius).astype(np.int64)
    gy = np.floor(df["y"].to_numpy(dtype=float) / radius).astype(np.int64)
    key = pd.DataFrame(
        {
            "clip": df["clip"].to_numpy(),
            "gx": gx,
            "gy": gy,
            "frame": df["frame"].to_numpy(),
        }
    )
    # a 3x3 neighbourhood would double count; one cell is coarse but cheap and conservative
    n_frames = key.groupby(["clip", "gx", "gy"])["frame"].transform("nunique")
    return pd.Series(n_frames.to_numpy() >= min_frames, index=df.index)


def read_machine_label_rows(path: Path, *, min_conf: float = 0.0) -> list[dict]:
    """Ball-class rows of one RT-DETR jsonl (``machine_labels`` or the ball-only ``ball_candidates``)."""
    rows = []
    with path.open() as fh:
        for line in fh:
            if (
                '"ball"' not in line and "_ball" not in line
            ):  # cheap pre-filter on the big all-class files
                continue
            try:
                r = json.loads(line)
            except ValueError:
                continue
            tags = r.get("tags") or []
            if not any(t in BALL_TAGS for t in tags):
                continue
            conf = r.get("confidence")
            bb = r.get("bbox")
            if conf is None or not isinstance(bb, dict) or conf < min_conf:
                continue
            rows.append(r)
    return rows


def pool_from_machine_labels(
    paths: Iterable[str | Path],
    *,
    match: str | None = None,
    fps: float = 25.0,
    detector_sha: str | None = None,
    scorer_version: str = "raw_conf",
    min_conf: float = 0.0,
    derive_static: bool = True,
    static_min_frames: int = 25,
) -> pd.DataFrame:
    """Adapt RT-DETR jsonl files (one per clip; a directory means every ``*.jsonl`` in it) to a pool.

    Single source ``rtdetr``; ``p_real`` is the raw detector confidence (``scorer_version="raw_conf"``) until a
    scorer exists; ``gap_frame`` / ``plausible`` / ``solver_post`` stay null. ``static_flag`` is
    :func:`derive_static_flags` when ``derive_static``. ``detector_sha`` defaults to the row's
    ``model_sha`` / ``run_id`` / ``model_id`` (the first one present), else ``"unknown"``.
    """
    files: list[Path] = []
    for entry in paths:
        entry = Path(entry)  # noqa: PLW2901
        files += sorted(entry.glob("*.jsonl")) if entry.is_dir() else [entry]
    recs: dict[str, list] = defaultdict(list)
    for f in files:
        clip = f.stem
        m = match or _match_of(clip)
        for r in read_machine_label_rows(f, min_conf=min_conf):
            bb = r["bbox"]
            w, h = float(bb["w"]), float(bb["h"])
            cx = float(bb["x"]) + w / 2
            cy = float(bb["y"]) + h / 2
            c = r.get("center")
            if isinstance(c, dict):
                cx, cy = float(c["x"]), float(c["y"])
            sha = (
                detector_sha
                or r.get("model_sha")
                or r.get("run_id")
                or r.get("model_id")
                or "unknown"
            )
            recs["match"].append(m)
            recs["clip"].append(clip)
            recs["frame"].append(int(r["frame_index"]))
            recs["x"].append(min(max(cx, 0.0), 1.0))
            recs["y"].append(min(max(cy, 0.0), 1.0))
            recs["w"].append(min(max(w, 1e-4), 1.0))
            recs["h"].append(min(max(h, 1e-4), 1.0))
            recs["conf"].append(min(max(float(r["confidence"]), 0.0), 1.0))
            recs["detector_sha"].append(str(sha))
    if not recs["frame"]:
        return empty_pool()
    n = len(recs["frame"])
    df = pd.DataFrame(
        {
            "match": recs["match"], "clip": recs["clip"], "frame": recs["frame"],
            "fps": np.float32(fps), "x": recs["x"], "y": recs["y"], "w": recs["w"], "h": recs["h"],
            "track_id": pd.array([None] * n, dtype="Int64"),
            "sources": [["rtdetr"]] * n, "source_scores": [[c] for c in recs["conf"]],
            "n_sources": np.int32(1),
            "solver_post": np.float32("nan"), "sam2_agree": None, "plausible": None, "gap_frame": None,
            "embedding": None, "p_real": recs["conf"], "scorer_version": scorer_version,
            "detector_sha": recs["detector_sha"],
        }
    )  # fmt: skip
    df["static_flag"] = (
        derive_static_flags(df, min_frames=static_min_frames).astype(object)
        if derive_static
        else None
    )
    return validate_pool(df)
