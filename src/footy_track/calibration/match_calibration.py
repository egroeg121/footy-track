"""Whole-match camera calibration files: combine per-segment PTZ outputs, load them for tracking / projection.

The PTZ V3 temporal calibrator writes one directory per broadcast segment
(``<temporal>/<match>_segNNN@<rig>/``). This module only *reads* those files (no ptz imports), picks the best
result per segment and writes ONE Parquet + ONE JSON manifest per match::

    <calibration root>/<version>/<match>.parquet         one row per broadcast frame of the match
    <calibration root>/<version>/<match>.manifest.json   version id, content hash, per segment provenance, totals

``<calibration root>`` is ``$FOOTY_CALIBRATION_ROOT`` or ``<$FOOTY_DATA_DIR | /mnt/storage/footy_data>/calibration``.

Conventions
-----------
* ``frame`` is the GLOBAL frame index of the full match video (``segments.json`` ``start_frame`` + the segment-local
  frame); ``segment`` / ``local_frame`` locate it in the segment mp4. Only frames inside a broadcast segment have a
  row (the rest of the match is not broadcast footage).
* ``H`` is row-major 3x3 mapping the UNDISTORTED normalised image (x/1920, y/1080) to pitch metres (origin centre spot,
  +x along the long axis): the same convention as ``calibration.H`` in the feature store and the editor. The lens is
  the division model of :mod:`footy_track.calibration.geometry` with the per-frame ``k1`` (centre (0.5, 0.5), aspect
  16:9), exactly how ``lens.py`` / the editor distort and undistort. ``H`` is NaN where ``calibrated`` is false.
* ``accepted`` = the frame lies in ``accepted_ranges`` of ``line_f1_full.json`` (both neighbouring evidence frames have
  line F1 >= 0.5) and is calibrated. ``calibrated`` = a solution exists, confirmed or not.
* ``source``: ``main`` (the flow run), ``recovered`` (the weak-segment recovery re-run, kept only if it accepts
  more frames than the main run) or ``none`` (segment has no result).

Versioning: a file is immutable by default. ``combine_match`` refuses to replace a file whose content differs
unless ``update=True``, in which case the previous pair is archived under ``<version>/_history/`` first. The manifest
carries ``content_sha256`` (over the data, independent of the parquet writer) and ``version_id`` =
``<version>+<sha8>``.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from footy_track.calibration import geometry as geo

DEFAULT_VERSION = "ptz_v3"
SCHEMA_VERSION = 1
H_CONVENTION = "undistorted_normalised_image_to_pitch_m"
#: the PTZ calibrator works in a 1920x1080 canonical frame
CANON_W, CANON_H = 1920, 1080
STATUS_ACCEPTED, STATUS_CALIBRATED, STATUS_UNCALIBRATED, STATUS_MISSING = (
    "accepted", "calibrated", "uncalibrated", "missing")
SOURCES = ("main", "recovered", "none")

#: known matches -> rig stem of the PTZ outputs. A bare stem also resolves ``<stem>-<hash8>`` dirs (:func:`resolve_rig_tag`);
#: a match outside this table (or a fitted per-match rig) is passed as ``rig_stem=`` / ``--rig``.
KNOWN_MATCHES: dict[str, str] = {
    "arsenal_mancity": "emirates",
    "arsenal_astonvilla": "emirates",
    "arsenal_nottinghamforest": "emirates",
    "arsenal_bournmouth_1st_half": "vitality",
}

ARROW_SCHEMA = pa.schema([
    ("frame", pa.int64()),
    ("segment", pa.int32()),
    ("local_frame", pa.int32()),
    ("calibrated", pa.bool_()),
    ("accepted", pa.bool_()),
    ("evidence_f1", pa.float32()),
    ("source", pa.string()),
    ("pan", pa.float64()),
    ("tilt", pa.float64()),
    ("logf", pa.float64()),
    ("H", pa.list_(pa.float64(), 9)),
    ("k1", pa.float64()),
    ("rig", pa.string()),
    ("camera_variant", pa.string()),
    ("cx", pa.float64()),
    ("cy", pa.float64()),
    ("cz", pa.float64()),
    ("roll", pa.float64()),
])


def data_root() -> Path:
    return Path(os.environ.get("FOOTY_DATA_DIR", "/mnt/storage/footy_data"))


def calibration_root() -> Path:
    env = os.environ.get("FOOTY_CALIBRATION_ROOT")
    return Path(env) if env else data_root() / "calibration"


def temporal_root() -> Path:
    return data_root() / "scratch" / "temporal"


# --------------------------------------------------------------------------
# reading the per-segment PTZ outputs
# --------------------------------------------------------------------------


@dataclass
class SegmentResult:
    """One PTZ output directory, parsed."""

    path: Path
    frame: np.ndarray  # (N,) local frame index
    pan: np.ndarray
    tilt: np.ndarray
    logf: np.ndarray
    H: np.ndarray  # (N, 3, 3)
    rig: np.ndarray  # (5,) Cx Cy Cz roll k1
    evidence: dict[int, float]
    accepted_ranges: list[tuple[int, int]]
    meta: dict
    accepted: np.ndarray = field(init=False)
    calibrated: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        n = len(self.frame)
        self.calibrated = np.isfinite(self.logf) & np.isfinite(self.H).all(axis=(1, 2))
        acc = np.zeros(n, dtype=bool)
        pos = {int(f): i for i, f in enumerate(self.frame)}
        for a, b in self.accepted_ranges:
            for f in range(int(a), int(b) + 1):  # inclusive, a one-frame range is [f, f]
                i = pos.get(f)
                if i is not None:
                    acc[i] = True
        self.accepted = acc & self.calibrated

    @property
    def n(self) -> int:
        return len(self.frame)

    @property
    def accepted_frac(self) -> float:
        """The fraction recorded by the run itself (``meta.json``), else recomputed."""
        v = self.meta.get("accepted_frac")
        return float(v) if v is not None else float(self.accepted.mean()) if self.n else 0.0

    @property
    def calibrated_frac(self) -> float:
        return float(self.calibrated.mean()) if self.n else 0.0


def load_segment_result(path: Path) -> SegmentResult | None:
    """Parse ``<path>/calib_full.npz`` + ``line_f1_full.json`` + ``meta.json``; ``None`` when ``calib_full.npz``
    (the run's done marker) is missing."""
    path = Path(path)
    npz = path / "calib_full.npz"
    if not npz.is_file():
        return None
    z = np.load(npz)
    f1 = json.loads((path / "line_f1_full.json").read_text()) if (path / "line_f1_full.json").is_file() else {}
    meta = json.loads((path / "meta.json").read_text()) if (path / "meta.json").is_file() else {}
    return SegmentResult(
        path=path,
        frame=np.asarray(z["frame"], dtype=np.int64),
        pan=np.asarray(z["pan"], dtype=np.float64),
        tilt=np.asarray(z["tilt"], dtype=np.float64),
        logf=np.asarray(z["logf"], dtype=np.float64),
        H=np.asarray(z["H"], dtype=np.float64).reshape(-1, 3, 3),
        rig=np.asarray(z["rig"], dtype=np.float64),
        evidence={int(k): float(v) for k, v in (f1.get("evidence_f1") or {}).items()},
        accepted_ranges=[(int(a), int(b)) for a, b in (f1.get("accepted_ranges") or [])],
        meta=meta,
    )


def choose_result(
    main: SegmentResult | None, rerun: SegmentResult | None
) -> tuple[SegmentResult | None, str]:
    """The recovery re-run wins only if present and strictly more of the segment is accepted (the rule the
    runner itself uses); a segment with no result at all is ``(None, "none")``."""
    if main is None and rerun is None:
        return None, "none"
    if main is None:
        return rerun, "recovered"
    if rerun is not None and rerun.accepted_frac > main.accepted_frac:
        return rerun, "recovered"
    return main, "main"


def _segment_dirname(match: str, index: int, rig: str) -> str:
    return f"{match}_seg{index:03d}@{rig}"


class AmbiguousRig(RuntimeError):
    """Several rig tags have finished segment dirs for the match and none was named exactly."""


_HASHED = re.compile(r"^(?P<stem>.+)-(?P<hash>[0-9a-f]{8})$")


def done_rig_tags(match: str, temporal: Path) -> dict[str, int]:
    """``{<dir suffix after '@'>: number of finished segment dirs}`` of ``<match>_segNNN@<suffix>`` under ``temporal``
    (finished = ``calib_full.npz`` present)."""
    out: dict[str, int] = {}
    pat = re.compile(rf"^{re.escape(match)}_seg\d+@(?P<tag>.+)$")
    if not temporal.is_dir():
        return out
    for d in temporal.iterdir():
        m = pat.match(d.name)
        if m and (d / "calib_full.npz").is_file():
            out[m["tag"]] = out.get(m["tag"], 0) + 1
    return out


def resolve_rig_tag(match: str, rig_stem: str, temporal: Path) -> str:
    """The ``@``-suffix of the segment dirs of ``match`` for a rig named ``rig_stem``.

    The rig-hash work dirs are ``<stem>-<hash8>`` (the jobs flow names them after the rig file's CONTENT hash), older ones
    are the bare ``<stem>``. ``rig_stem`` may be either form:

    * a name that already has finished dirs is used as it is (an explicit ``emirates-1a2b3c4d`` or an old ``emirates``);
    * a bare stem without bare dirs but with exactly one ``<stem>-<hash8>`` tag resolves to that tag;
    * several hashed tags and no bare dirs raise :class:`AmbiguousRig` listing them (name the hash); no dirs at all
      returns ``rig_stem`` unchanged (the segments then simply have no result).
    """
    tags = done_rig_tags(match, temporal)
    if rig_stem in tags:
        return rig_stem
    hashed = sorted(t for t in tags if (m := _HASHED.match(t)) and m["stem"] == rig_stem)
    if len(hashed) == 1:
        return hashed[0]
    if len(hashed) > 1:
        raise AmbiguousRig(f"{match}: several rig hashes for {rig_stem!r} under {temporal}: {hashed}; pass one of them as the rig")
    return rig_stem


def read_segments_json(match: str, root: Path | None = None) -> dict:
    p = (root or data_root()) / match / "split_video_broadcast_frames" / "segments.json"
    return json.loads(p.read_text())


# --------------------------------------------------------------------------
# combine
# --------------------------------------------------------------------------


def _content_hash(cols: dict[str, np.ndarray]) -> str:
    """sha256 over the column bytes in schema order (independent of the parquet writer / compression)."""
    h = hashlib.sha256()
    h.update(json.dumps({"schema": SCHEMA_VERSION, "columns": [f.name for f in ARROW_SCHEMA]}).encode())
    for f in ARROW_SCHEMA:
        a = cols[f.name]
        if a.dtype == object:
            a = np.array([("" if v is None else str(v)) for v in a], dtype="U")
            h.update("\x1f".join(a.tolist()).encode())
        else:
            h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()


def _nearest_evidence(local: np.ndarray, ev: dict[int, float]) -> np.ndarray:
    out = np.full(len(local), np.nan, dtype=np.float32)
    if not ev:
        return out
    keys = np.array(sorted(ev), dtype=np.int64)
    vals = np.array([ev[int(k)] for k in keys], dtype=np.float32)
    j = np.searchsorted(keys, local)
    lo = np.clip(j - 1, 0, len(keys) - 1)
    hi = np.clip(j, 0, len(keys) - 1)
    pick = np.where(np.abs(keys[lo] - local) <= np.abs(keys[hi] - local), lo, hi)
    return vals[pick]


#: ``(segment dict, number of calibrated frames, segments.json) -> (global frame per local frame, method)``
FrameMap = Callable[[dict, int, dict], "tuple[np.ndarray, str]"]


def pts_frame_map(match: str, data_dir: Path | None = None) -> FrameMap:
    """Global frame of every decoded frame of a segment mp4 from its presentation timestamps.

    ``start_frame + round(pts * fps)``: the segment files are NOT always one frame per source frame. Bournemouth's are
    variable frame rate (gaps where non-broadcast footage was cut, e.g. seg087 holds 1026 of its 1486 source frames) and
    Villa's differ by a frame or two from ``frame_count``; Man City / Forest are exactly ``start_frame + i``. Falls back
    to ``start_frame + i`` (method ``index``) when ffprobe is unavailable or its frame count disagrees with the run's.
    A segment whose run has exactly ``frame_count`` frames is taken as ``start_frame + i`` without reading the file."""
    root = (data_dir or data_root()) / match / "split_video_broadcast_frames"

    def fmap(seg: dict, n: int, meta: dict) -> tuple[np.ndarray, str]:
        start = int(seg["start_frame"])
        fallback = start + np.arange(n, dtype=np.int64), "index"
        if n == int(seg["frame_count"]):  # one frame per source frame: no need to read the (large) file
            return fallback
        try:
            out = subprocess.run(
                ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "packet=pts_time",
                 "-of", "csv=p=0", str(root / seg["file"])],
                capture_output=True, text=True, timeout=120, check=True).stdout
            pts = np.sort(np.array([float(t.strip(",")) for t in out.split()], dtype=np.float64))
        except (OSError, subprocess.SubprocessError, ValueError):
            return fallback
        g = start + np.rint(pts * float(meta["fps"])).astype(np.int64)
        if len(g) != n or len(np.unique(g)) != n:
            return fallback
        return g, "pts"

    return fmap


def _index_frame_map(seg: dict, n: int, meta: dict) -> tuple[np.ndarray, str]:
    return int(seg["start_frame"]) + np.arange(n, dtype=np.int64), "index"


def build_match_table(  # noqa: PLR0915
    match: str,
    *,
    rig_stem: str,
    segments: dict,
    temporal: Path,
    rerun_dir: Path | None,
    frame_map: FrameMap = _index_frame_map,
) -> tuple[dict[str, np.ndarray], dict]:
    """Pick the best result per segment and build the per-frame columns and the manifest body."""
    seg_list = sorted(segments["segments"], key=lambda s: int(s["index"]))
    parts: dict[str, list[np.ndarray]] = {f.name: [] for f in ARROW_SCHEMA}
    seg_rows: list[dict] = []
    warnings: list[str] = []
    code_shas: dict[str, set[str]] = {"main": set(), "recovered": set()}

    for seg in seg_list:
        idx = int(seg["index"])
        name = _segment_dirname(match, idx, rig_stem)
        main = load_segment_result(temporal / name)
        rerun = load_segment_result(rerun_dir / name) if rerun_dir else None
        res, source = choose_result(main, rerun)
        start, n_container = int(seg["start_frame"]), int(seg["frame_count"])
        row = {
            "segment": idx,
            "file": seg.get("file"),
            "start_frame": start,
            "frame_count": n_container,
            "source": source,
            "main_accepted_frac": None if main is None else round(main.accepted_frac, 6),
            "rerun_accepted_frac": None if rerun is None else round(rerun.accepted_frac, 6),
        }
        if res is None:
            # no run for this segment (e.g. unreadable video): no rows, since its frames cannot be mapped
            warnings.append(f"seg{idx:03d}: no result in {temporal}: no rows (frame_count {n_container})")
            row.update(n_frames=0, accepted_frac=0.0, calibrated_frac=0.0, code_sha=None, rig_variant=None,
                       frame_map=None)
            seg_rows.append(row)
            continue
        n = res.n
        gframes, method = frame_map(seg, n, segments)
        if method == "index" and n != n_container:
            warnings.append(f"seg{idx:03d}: {n} frames in the run vs frame_count {n_container} and no pts mapping")
        cal, acc = res.calibrated, res.accepted
        rig = res.rig
        variant_s = res.meta.get("rig_variant")
        sha = res.meta.get("code_sha")
        if sha:
            code_shas["recovered" if source == "recovered" else "main"].add(str(sha)[:12])
        row.update(
            n_frames=n,
            accepted_frac=round(float(acc.mean()) if n else 0.0, 6),
            calibrated_frac=round(float(cal.mean()) if n else 0.0, 6),
            code_sha=str(sha)[:12] if sha else None,
            rig_variant=variant_s,
            frame_map=method,
        )
        if abs(row["accepted_frac"] - res.accepted_frac) > 0.01:
            warnings.append(f"seg{idx:03d}: recomputed accepted {row['accepted_frac']:.4f} vs meta {res.accepted_frac:.4f}")
        seg_rows.append(row)
        parts["frame"].append(np.asarray(gframes, dtype=np.int64)[res.frame])
        parts["segment"].append(np.full(n, idx, dtype=np.int32))
        parts["local_frame"].append(res.frame.astype(np.int32))
        parts["calibrated"].append(cal)
        parts["accepted"].append(acc)
        parts["evidence_f1"].append(_nearest_evidence(res.frame, res.evidence))
        parts["source"].append(np.array([source] * n, dtype=object))
        for k, v in (("pan", res.pan), ("tilt", res.tilt), ("logf", res.logf), ("k1", np.full(n, rig[4])),
                     ("cx", np.full(n, rig[0])), ("cy", np.full(n, rig[1])), ("cz", np.full(n, rig[2])),
                     ("roll", np.full(n, rig[3]))):
            parts[k].append(np.asarray(v, dtype=np.float64))
        parts["H"].append(res.H.reshape(n, 9))
        parts["rig"].append(np.array([rig_stem] * n, dtype=object))
        parts["camera_variant"].append(np.array([variant_s] * n, dtype=object))

    table = {k: (np.concatenate(v) if v else np.zeros(0)) for k, v in parts.items()}
    order = np.argsort(table["frame"], kind="stable")
    table = {k: v[order] for k, v in table.items()}
    if len(table["frame"]) and len(np.unique(table["frame"])) != len(table["frame"]):
        warnings.append("duplicate global frame indices (overlapping segments in segments.json)")

    n_frames = int(len(table["frame"]))
    n_acc, n_cal = int(table["accepted"].sum()), int(table["calibrated"].sum())
    n_missing = int(sum(r["frame_count"] for r in seg_rows if r["source"] == "none"))
    body = {
        "match": match,
        "rig": rig_stem,
        "segments": seg_rows,
        "totals": {
            "segments": len(seg_rows),
            "segments_recovered": sum(r["source"] == "recovered" for r in seg_rows),
            "segments_missing": sum(r["source"] == "none" for r in seg_rows),
            "frames": n_frames,
            "frames_accepted": n_acc,
            "frames_calibrated": n_cal,
            "frames_missing_segments": n_missing,
            "accepted_frac": round(n_acc / n_frames, 6) if n_frames else 0.0,
            "calibrated_frac": round(n_cal / n_frames, 6) if n_frames else 0.0,
        },
        "code_shas": {k: sorted(v) for k, v in code_shas.items()},
        "warnings": warnings,
    }
    return table, body


def _arrow_table(cols: dict[str, np.ndarray]) -> pa.Table:
    arrays = []
    for f in ARROW_SCHEMA:
        a = cols[f.name]
        if f.name == "H":
            flat = pa.array(np.ascontiguousarray(a, dtype=np.float64).ravel(), type=pa.float64())
            arrays.append(pa.FixedSizeListArray.from_arrays(flat, 9))
        elif a.dtype == object:
            arrays.append(pa.array([None if v is None else str(v) for v in a], type=pa.string()))
        else:
            arrays.append(pa.array(a, type=f.type))
    return pa.Table.from_arrays(arrays, schema=ARROW_SCHEMA)


def _atomic_write_bytes(path: Path, data: bytes) -> None:
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=path.name + ".", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
        os.chmod(tmp, 0o664)
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def _atomic_write_parquet(path: Path, table: pa.Table) -> None:
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=path.name + ".", suffix=".tmp")
    os.close(fd)
    try:
        pq.write_table(table, tmp, compression="zstd")
        os.chmod(tmp, 0o664)
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


class WouldOverwrite(RuntimeError):
    """The match file exists with different content and ``update`` was not requested."""


def combine_match(
    match: str,
    *,
    out_root: Path | None = None,
    version: str = DEFAULT_VERSION,
    rig_stem: str | None = None,
    data_dir: Path | None = None,
    temporal: Path | None = None,
    rerun_dir: Path | None = None,
    update: bool = False,
    dry_run: bool = False,
    frame_map: FrameMap | None = None,
    use_rerun: bool = True,
    provenance: dict | None = None,
) -> dict:
    """Write ``<out_root>/<version>/<match>.{parquet,manifest.json}`` and return the manifest.

    Re-running with unchanged inputs is a no-op (same content hash). A changed result raises
    :class:`WouldOverwrite` unless ``update`` (the old pair is then moved to ``_history/``).

    ``rig_stem``: the suffix after ``@`` of the segment dirs, bare (``emirates``) or hashed (``emirates-1a2b3c4d``);
    see :func:`resolve_rig_tag`. ``use_rerun=False`` ignores a legacy ``<temporal>/rerun_weak`` dir (the flow's runs
    include the recovery pass themselves). ``provenance`` is stored in the manifest as is (rig search, per-segment
    diagnostic, code sha) and does not enter the content hash."""
    rig_stem = rig_stem or KNOWN_MATCHES.get(match)
    if rig_stem is None:
        raise ValueError(f"no rig known for {match!r}: pass rig_stem (the suffix after '@' of the segment dirs)")
    temporal = temporal or temporal_root()
    rig_stem = resolve_rig_tag(match, rig_stem, temporal)
    if rerun_dir is None and use_rerun:
        cand = temporal / "rerun_weak"
        rerun_dir = cand if cand.is_dir() else None
    segments = read_segments_json(match, data_dir)
    table, body = build_match_table(
        match, rig_stem=rig_stem, segments=segments, temporal=temporal, rerun_dir=rerun_dir,
        frame_map=frame_map or pts_frame_map(match, data_dir))
    content = _content_hash(table)
    manifest = {
        "schema": SCHEMA_VERSION,
        "dataset": "calibration",
        "version": version,
        "version_id": f"{version}+{content[:8]}",
        "content_sha256": content,
        "h_convention": H_CONVENTION,
        "lens": {"model": "division", "centre": [0.5, 0.5], "aspect": geo.DEFAULT_ASPECT,
                 "k1": "per-frame column, constant within a segment"},
        "columns": [f.name for f in ARROW_SCHEMA],
        "created_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "inputs": {
            "temporal_root": str(temporal),
            "rerun_root": None if rerun_dir is None else str(rerun_dir),
            "segments_json": str((data_dir or data_root()) / match / "split_video_broadcast_frames" / "segments.json"),
            "video_fps": segments.get("fps"),
            "video_total_frames": segments.get("total_frames"),
        },
        **body,
    }
    if provenance:
        manifest["provenance"] = provenance
    out_dir = Path(out_root or calibration_root()) / version
    pq_path, mf_path = out_dir / f"{match}.parquet", out_dir / f"{match}.manifest.json"
    manifest["files"] = {"parquet": pq_path.name, "manifest": mf_path.name}
    manifest["status"] = "dry_run"
    if dry_run:
        return manifest

    out_dir.mkdir(parents=True, exist_ok=True)
    if mf_path.is_file():
        old = json.loads(mf_path.read_text())
        if old.get("content_sha256") == content and pq_path.is_file():
            old["status"] = "unchanged"
            return old
        if not update:
            raise WouldOverwrite(
                f"{mf_path} exists with content {str(old.get('content_sha256'))[:8]} != new {content[:8]}; "
                "pass update=True (--update) to replace it (the old files are archived under _history/)")
        hist = out_dir / "_history"
        hist.mkdir(exist_ok=True)
        tag = str(old.get("content_sha256", "unknown"))[:8]
        for src in (pq_path, mf_path):
            if src.is_file():
                shutil.copy2(src, hist / re.sub(r"(\.parquet|\.manifest\.json)$", rf".{tag}\1", src.name))
    manifest["status"] = "written"
    _atomic_write_parquet(pq_path, _arrow_table(table))
    # the manifest goes last: it is the "done" marker the loader keys on
    _atomic_write_bytes(mf_path, (json.dumps(manifest, indent=1) + "\n").encode())
    return manifest


# --------------------------------------------------------------------------
# loading + projection
# --------------------------------------------------------------------------


def undistorted_to_pitch(H: np.ndarray, uv_und: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Undistorted normalised image points -> pitch metres: ``(xy (N,2), w (N,))``."""
    return geo.apply_homography(np.asarray(H, dtype=np.float64), np.atleast_2d(uv_und))


def project_pitch_to_image(
    H: np.ndarray, pts_m: np.ndarray, k1: float = 0.0, *, aspect: float = geo.DEFAULT_ASPECT
) -> tuple[np.ndarray, np.ndarray]:
    """Pitch metres (N,2) -> OBSERVED (distorted) normalised image points and an in-front/finite mask.

    ``H`` maps undistorted normalised image -> pitch; the lens is applied last with the division model of
    ``geometry`` (same as ``lens.snap_to_lines`` / the editor). Points behind the camera or with no real
    distortion root are ``ok=False``."""
    dist = geo.make_distortion(float(k1), aspect) if k1 and np.isfinite(k1) else None
    xy, w = geo.pitch_to_image(np.asarray(H, dtype=np.float64), np.atleast_2d(pts_m), dist)
    ok = (w > geo.W_EPS) & np.isfinite(xy).all(axis=1)
    return xy, ok


def project_image_to_pitch(
    H: np.ndarray, uv_obs: np.ndarray, k1: float = 0.0, *, aspect: float = geo.DEFAULT_ASPECT
) -> tuple[np.ndarray, np.ndarray]:
    """Observed (distorted) normalised image points -> pitch metres: undistort with ``k1`` then ``H``."""
    dist = geo.make_distortion(float(k1), aspect) if k1 and np.isfinite(k1) else None
    return geo.image_to_pitch(np.asarray(H, dtype=np.float64), np.atleast_2d(uv_obs), dist)


class MatchCalibration:
    """Per-frame calibration of one match, loaded from ``<match>.parquet`` (+ manifest)."""

    def __init__(self, match: str, table: pa.Table, manifest: dict) -> None:
        self.match = match
        self.manifest = manifest
        self.version_id: str = manifest.get("version_id", "")
        self.n = table.num_rows
        c = {name: table.column(name) for name in table.column_names}
        self.frame = c["frame"].to_numpy().astype(np.int64)
        self.segment = c["segment"].to_numpy().astype(np.int32)
        self.local_frame = c["local_frame"].to_numpy().astype(np.int32)
        self.calibrated = c["calibrated"].to_numpy(zero_copy_only=False).astype(bool)
        self.accepted = c["accepted"].to_numpy(zero_copy_only=False).astype(bool)
        self.evidence_f1 = c["evidence_f1"].to_numpy(zero_copy_only=False).astype(np.float32)
        self.source = np.array(c["source"].to_pylist(), dtype=object)
        self.pan, self.tilt, self.logf = (c[k].to_numpy(zero_copy_only=False) for k in ("pan", "tilt", "logf"))
        self.k1 = c["k1"].to_numpy(zero_copy_only=False)
        hflat = c["H"].combine_chunks().values.to_numpy(zero_copy_only=False) if self.n else np.zeros(0)
        self.H = hflat.reshape(self.n, 3, 3)
        order = np.argsort(self.frame, kind="stable")
        self._sorted = bool((order == np.arange(self.n)).all())
        self._order = order

    # -- lookup ------------------------------------------------------------
    def index_of(self, global_frame: int) -> int | None:
        """Row of a global frame, ``None`` when the frame is outside every broadcast segment."""
        i = int(np.searchsorted(self.frame, global_frame)) if self._sorted else -1
        if self._sorted and i < self.n and self.frame[i] == global_frame:
            return i
        return None

    def status(self, global_frame: int) -> str:
        i = self.index_of(global_frame)
        if i is None or self.source[i] == "none":
            return STATUS_MISSING
        if self.accepted[i]:
            return STATUS_ACCEPTED
        return STATUS_CALIBRATED if self.calibrated[i] else STATUS_UNCALIBRATED

    def H_for_frame(self, global_frame: int, *, accepted_only: bool = False) -> np.ndarray | None:
        """3x3 undistorted-normalised-image -> pitch-metres matrix of a frame, ``None`` when the frame is not
        calibrated (or, with ``accepted_only``, not accepted)."""
        i = self.index_of(global_frame)
        if i is None or not self.calibrated[i] or (accepted_only and not self.accepted[i]):
            return None
        return self.H[i]

    def k1_for_frame(self, global_frame: int) -> float | None:
        i = self.index_of(global_frame)
        return None if i is None or not np.isfinite(self.k1[i]) else float(self.k1[i])

    def pitch_to_image(self, global_frame: int, pts_m: np.ndarray, *, accepted_only: bool = False):
        """Pitch metres -> observed normalised image points of a frame (``None`` if no usable calibration)."""
        H = self.H_for_frame(global_frame, accepted_only=accepted_only)
        if H is None:
            return None
        return project_pitch_to_image(H, pts_m, self.k1_for_frame(global_frame) or 0.0)

    def image_to_pitch(self, global_frame: int, uv_obs: np.ndarray, *, accepted_only: bool = False):
        H = self.H_for_frame(global_frame, accepted_only=accepted_only)
        if H is None:
            return None
        return project_image_to_pitch(H, uv_obs, self.k1_for_frame(global_frame) or 0.0)

    def segment_rows(self, segment: int) -> slice:
        """Row range of a segment (rows are sorted by global frame, so a segment is contiguous)."""
        idx = np.nonzero(self.segment == segment)[0]
        return slice(int(idx[0]), int(idx[-1]) + 1) if len(idx) else slice(0, 0)

    def locate(self, global_frame: int) -> tuple[int, int] | None:
        """``(segment, local_frame)`` of a global frame."""
        i = self.index_of(global_frame)
        return None if i is None else (int(self.segment[i]), int(self.local_frame[i]))

    def global_frame(self, segment: int, local_frame: int) -> int | None:
        sel = np.nonzero((self.segment == segment) & (self.local_frame == local_frame))[0]
        return int(self.frame[sel[0]]) if len(sel) else None


def load_match_calibration(
    match: str, *, version: str = DEFAULT_VERSION, root: Path | None = None
) -> MatchCalibration:
    """Load ``<root>/<version>/<match>.parquet`` (``FileNotFoundError`` when it has not been combined yet)."""
    d = Path(root or calibration_root()) / version
    mf, pqf = d / f"{match}.manifest.json", d / f"{match}.parquet"
    if not mf.is_file() or not pqf.is_file():
        raise FileNotFoundError(f"no calibration for {match!r} at {d} (run scripts/calibration_match_combine.py)")
    manifest = json.loads(mf.read_text())
    table = pq.read_table(pqf)
    mc = MatchCalibration(match, table, manifest)
    if manifest.get("totals", {}).get("frames") not in (None, mc.n):
        raise ValueError(f"{pqf}: {mc.n} rows but the manifest says {manifest['totals']['frames']} (torn write?)")
    return mc
