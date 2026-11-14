# ruff: noqa: PLR0912, PLR0915, PLR0124, PLR0917, PLW2901, C408, B007, SIM105
"""Frozen GT snapshots.

Labels keep changing while people label, so a PR's numbers are only comparable
with main's if both were scored against the *same* labels.  A snapshot copies
the GT of an eval set, with everything the metrics need to score it (clip
roles, detector-training contamination, frame counts), into an immutable
directory::

    <root>/<eval_set>-<YYYYMMDD>-<hash12>/
        manifest.json      eval set, full hash, source files (sha256), counts
        gt/<clip>.json     frame -> [x, y, tier] (1920x1080 px), no_ball, role, contaminated

The **hash** covers only the scoring-relevant content (points, no_ball frames,
roles, contamination), not the date or paths, so snapshotting unchanged labels
twice yields the same hash.  ``load_snapshot`` re-derives it and refuses a
tampered directory.  Every report written by the harness records the hash.
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import hashlib
import json
import os
import stat
import subprocess
from pathlib import Path

from footy_track.ball_offline.data import (
    GTFrame,
    GTLoad,
    MarksDirGT,
    contaminated_frames,
    frame_count,
    machine_stem_map,
    roboflow_frames,
)
from footy_track.ball_offline.sources import EvalSetSpec, data_root

SCHEMA = 1
NEAR = 5  # frames either side of a detector-training image that count as contaminated
DEV_MAJORITY = (
    0.5  # clip is "dev" when >= this share of its GT ball frames is contaminated
)
HASH_LEN = 12


def snapshots_root() -> Path:
    return Path(
        os.environ.get("FOOTY_BALL_SNAPSHOTS", data_root() / "ball_eval_snapshots")
    )


@dataclasses.dataclass
class SnapClip:
    clip_id: str
    n_frames: int
    role: str  # "dev" | "test"
    gt: dict[int, GTFrame]
    no_ball: set[int]
    contaminated: set[int]  # GT / no_ball frames overlapping detector training images
    aliases: list[str] = dataclasses.field(default_factory=list)
    conflicts: int = 0

    def to_json(self) -> dict:
        return {
            "clip": self.clip_id,
            "n_frames": self.n_frames,
            "role": self.role,
            "gt": {
                str(t): [round(g.xy[0], 3), round(g.xy[1], 3), g.tier]
                for t, g in sorted(self.gt.items())
            },
            "no_ball": sorted(self.no_ball),
            "contaminated": sorted(self.contaminated),
            "aliases": self.aliases,
            "conflicts": self.conflicts,
        }

    @classmethod
    def from_json(cls, d: dict) -> SnapClip:
        return cls(
            d["clip"],
            d["n_frames"],
            d["role"],
            {int(t): GTFrame((v[0], v[1]), v[2]) for t, v in d["gt"].items()},
            set(d["no_ball"]),
            set(d["contaminated"]),
            d.get("aliases", []),
            d.get("conflicts", 0),
        )

    def gt_load(self) -> GTLoad:
        return self.gt, self.no_ball, self.aliases, self.conflicts

    @property
    def clean_scored(self) -> set[int]:
        return (set(self.gt) | self.no_ball) - self.contaminated


@dataclasses.dataclass
class Snapshot:
    path: Path
    manifest: dict
    clips: dict[str, SnapClip]

    @property
    def hash(self) -> str:
        return self.manifest["hash"]

    @property
    def short(self) -> str:
        return self.hash[:HASH_LEN]

    @property
    def eval_set(self) -> str:
        return self.manifest["eval_set"]

    def label(self) -> str:
        return f"{self.eval_set}@{self.short}"


class SnapshotGT:
    """A :class:`~footy_track.ball_offline.data.GTSource` backed by a snapshot."""

    def __init__(self, snap: Snapshot):
        self.snap = snap
        self.name = snap.label()

    def clip_ids(self) -> list[str]:
        return sorted(self.snap.clips)

    def load(self, clip_id: str) -> GTLoad:
        return self.snap.clips[clip_id].gt_load()


# --------------------------------------------------------------------------
# hashing


def content_hash(eval_set: str, clips: dict[str, SnapClip]) -> str:
    """sha256 over the scoring-relevant content only (stable across dates/paths/aliases)."""
    payload = {
        "schema": SCHEMA,
        "eval_set": eval_set,
        "contamination": {"near": NEAR, "dev_majority": DEV_MAJORITY},
        "clips": {
            cid: {
                k: v
                for k, v in c.to_json().items()
                if k not in ("aliases", "conflicts")
            }
            for cid, c in sorted(clips.items())
        },
    }
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(blob).hexdigest()


def _sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def candidates_fingerprint(directory: Path) -> str:
    """Cheap identity of a candidate directory (names + sizes), recorded in reports."""
    h = hashlib.sha256()
    for p in sorted(directory.glob("*.jsonl")):
        h.update(f"{p.name}:{p.stat().st_size};".encode())
    mp = directory / "_manifest.json"
    if mp.is_file():
        try:
            h.update(str(json.loads(mp.read_text()).get("run_id")).encode())
        except (OSError, ValueError):
            pass
    return h.hexdigest()[:16]


def _git_head() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent,
            timeout=10,
            check=False,
        )
        return out.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


# --------------------------------------------------------------------------
# build / write


def build_snapshot_clips(
    spec: EvalSetSpec, gt: MarksDirGT | None = None, *, rf: dict | None = None
) -> tuple[dict[str, SnapClip], list[str]]:
    """Live GT -> per-clip snapshot content.  Clips with GT but no candidates are
    returned in the second value (they cannot be scored)."""
    gt = gt or MarksDirGT(spec.gt_dir)
    cands = machine_stem_map(spec.candidates_dir)
    rf = rf if rf is not None else roboflow_frames()
    clips: dict[str, SnapClip] = {}
    skipped: list[str] = []
    for cid in gt.clip_ids():
        points, no_ball, aliases, conflicts = gt.load(cid)
        if not points and not no_ball:
            continue
        if cid not in cands:
            skipped.append(cid)
            continue
        stem_path = cands[cid]
        n_frames = frame_count(spec, stem_path)
        points = {t: g for t, g in points.items() if t < n_frames}
        no_ball = {t for t in no_ball if t < n_frames}
        bad = contaminated_frames(cid, rf, near=NEAR)
        scored = set(points) | no_ball
        contam = scored & bad
        n_bad = sum(1 for t in contam if t in points)
        role = "dev" if points and n_bad / len(points) >= DEV_MAJORITY else "test"
        clips[cid] = SnapClip(
            cid, n_frames, role, points, no_ball, contam, aliases, conflicts
        )
    return clips, skipped


def create_snapshot(
    spec: EvalSetSpec,
    out_root: Path | None = None,
    *,
    gt: MarksDirGT | None = None,
    rf: dict | None = None,
    today: dt.date | None = None,
) -> Snapshot:
    """Freeze the current GT of ``spec`` into ``<out_root>/<set>-<date>-<hash12>/``.

    Idempotent: if a snapshot with the same content hash already exists it is
    returned untouched."""
    out_root = Path(out_root) if out_root else snapshots_root()
    gt = gt or MarksDirGT(spec.gt_dir)
    clips, skipped = build_snapshot_clips(spec, gt, rf=rf)
    if not clips:
        raise ValueError(
            f"no scorable GT in {spec.gt_dir} (clips with GT and candidates: 0)"
        )
    h = content_hash(spec.name, clips)
    existing = (
        sorted(out_root.glob(f"{spec.name}-*-{h[:HASH_LEN]}"))
        if out_root.is_dir()
        else []
    )
    if existing:
        return load_snapshot(existing[0])
    date = (today or dt.datetime.now(dt.UTC).date()).strftime("%Y%m%d")
    d = out_root / f"{spec.name}-{date}-{h[:HASH_LEN]}"
    (d / "gt").mkdir(parents=True)
    for cid, c in clips.items():
        _write_ro(d / "gt" / f"{cid}.json", json.dumps(c.to_json(), sort_keys=True))
    sources = {}
    for cid in clips:
        for p in gt.source_files(cid):
            sources[p.name] = {"sha256": _sha256_file(p), "bytes": p.stat().st_size}
    test = [c for c in clips.values() if c.role == "test"]
    manifest = {
        "schema": SCHEMA,
        "hash": h,
        "eval_set": spec.name,
        "created_utc": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        "harness_git": _git_head(),
        "gt_dir": str(spec.gt_dir),
        "candidates_dir": str(spec.candidates_dir),
        "candidates_fingerprint": candidates_fingerprint(spec.candidates_dir),
        "contamination": {"near": NEAR, "dev_majority": DEV_MAJORITY},
        "source_files": sources,
        "skipped_no_candidates": skipped,
        "n_clips": len(clips),
        "n_test_clips": len(test),
        "n_test_clean_ball": sum(
            len(c.gt) - sum(1 for t in c.contaminated if t in c.gt) for c in test
        ),
        "n_test_no_ball": sum(len(c.no_ball - c.contaminated) for c in test),
        "n_dev_ball": sum(len(c.gt) for c in clips.values() if c.role == "dev"),
        "clips": {
            cid: {
                "role": c.role,
                "n_frames": c.n_frames,
                "gt_ball": len(c.gt),
                "no_ball": len(c.no_ball),
                "contaminated": len(c.contaminated),
            }
            for cid, c in sorted(clips.items())
        },
    }
    _write_ro(d / "manifest.json", json.dumps(manifest, indent=1, sort_keys=True))
    return load_snapshot(d)


def _write_ro(path: Path, text: str) -> None:
    path.write_text(text + "\n")
    path.chmod(stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)


# --------------------------------------------------------------------------
# load / verify


def load_snapshot(path: Path | str, *, verify: bool = True) -> Snapshot:
    """Load a snapshot directory; refuses it if the content no longer hashes to the recorded hash."""
    path = Path(path)
    manifest = json.loads((path / "manifest.json").read_text())
    clips = {}
    for p in sorted((path / "gt").glob("*.json")):
        c = SnapClip.from_json(json.loads(p.read_text()))
        clips[c.clip_id] = c
    if verify:
        got = content_hash(manifest["eval_set"], clips)
        if got != manifest["hash"]:
            raise ValueError(
                f"snapshot {path} is corrupt or was edited: hash {got[:HASH_LEN]} != {manifest['hash'][:HASH_LEN]}"
            )
    return Snapshot(path, manifest, clips)


def resolve_snapshot(arg: str | Path, root: Path | None = None) -> Snapshot:
    """``arg`` is a snapshot directory, a directory name under the snapshots root, or a
    (>= 6 char) hash prefix, optionally ``<eval_set>@<hash>``."""
    p = Path(arg)
    if (p / "manifest.json").is_file():
        return load_snapshot(p)
    root = Path(root) if root else snapshots_root()
    key = str(arg).split("@")[-1]
    hits = [
        d
        for d in sorted(root.glob("*"))
        if (d / "manifest.json").is_file()
        and (
            d.name == str(arg)
            or (len(key) >= 6 and d.name.split("-")[-1].startswith(key))
        )
    ]
    if len(hits) != 1:
        raise FileNotFoundError(f"snapshot {arg!r}: {len(hits)} matches under {root}")
    return load_snapshot(hits[0])
