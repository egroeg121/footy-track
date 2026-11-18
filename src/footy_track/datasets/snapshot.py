"""Immutable, content-hashed dataset snapshots.

Layout under the datasets root (default ``/mnt/storage/footy_data``, env ``FOOTY_DATASETS_ROOT``
then ``FOOTY_DATA``)::

    datasets/<dataset>/v<N>/items.parquet   one row per labelled item (+ first_version)
    datasets/<dataset>/v<N>/manifest.json   version, hashes, counts, split registers, diff

A snapshot is written once, atomically (temp dir + rename), then made read-only; writing to an
existing version raises. ``bump`` creates ``v<N+1>`` only when the content hash differs from the
latest version, so re-running it nightly is free.

Content hash
------------
sha256 over a header line (``{"dataset", "schema"}``) followed by one canonical JSON line per
item, sorted by ``item_id``::

    [item_id, split, match_key, clip, game_id, frame_index, label_kind, tier, payload]

Payloads are canonical JSON (sorted keys, no spaces, floats rounded to 6 places). Nothing
time-, path-, version- or git-dependent is hashed, so the same labels and the same split
registers give the same hash on any machine. ``first_version`` is derived (the version an item
first appeared in) and is deliberately outside the hash.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import stat
import subprocess
import tempfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from footy_track import match_splits as ms
from footy_track import splits
from footy_track.datasets.items import DATASETS, ITEM_SCHEMA, DatasetBuildError, Item, build_items
from footy_track.feature_store.lock import default_db_path, store_lock
from footy_track.feature_store.store import FeatureStore

ENV_ROOT = "FOOTY_DATASETS_ROOT"
_VERSION_DIR = re.compile(r"^v(\d+)$")

ITEM_SCHEMA_ARROW = pa.schema(
    [
        ("item_id", pa.string()),
        ("dataset", pa.string()),
        ("split", pa.string()),
        ("match_key", pa.string()),
        ("clip", pa.string()),
        ("game_id", pa.string()),
        ("frame_index", pa.int64()),
        ("label_kind", pa.string()),
        ("tier", pa.int8()),
        ("payload", pa.string()),
        ("item_hash", pa.string()),
        ("first_version", pa.int32()),
    ]
)


class SnapshotError(RuntimeError):
    pass


class SnapshotExists(SnapshotError):
    """Refused to overwrite an immutable snapshot."""


class SnapshotTampered(SnapshotError):
    """A snapshot's items no longer hash to its manifest."""


def datasets_root(root: Path | None = None) -> Path:
    if root:
        return Path(root)
    return Path(os.environ.get(ENV_ROOT) or splits.data_root())


def dataset_dir(dataset: str, root: Path | None = None) -> Path:
    return datasets_root(root) / "datasets" / dataset


def version_dir(dataset: str, version: int, root: Path | None = None) -> Path:
    return dataset_dir(dataset, root) / f"v{version}"


def list_versions(dataset: str, root: Path | None = None) -> list[int]:
    d = dataset_dir(dataset, root)
    if not d.is_dir():
        return []
    return sorted(
        int(m[1])
        for p in d.iterdir()
        if p.is_dir() and (m := _VERSION_DIR.match(p.name)) and (p / "manifest.json").is_file()
    )


def latest_version(dataset: str, root: Path | None = None) -> int | None:
    v = list_versions(dataset, root)
    return v[-1] if v else None


# --------------------------------------------------------------------------------------------
# hashing
# --------------------------------------------------------------------------------------------
def item_line(it: Item | dict) -> str:
    g = (lambda k: getattr(it, k)) if isinstance(it, Item) else (lambda k: it[k])
    return json.dumps(
        [g("item_id"), g("split"), g("match_key"), g("clip"), g("game_id"),
         int(g("frame_index")), g("label_kind"), g("tier"), g("payload")],
        separators=(",", ":"),
        ensure_ascii=True,
    )


def item_hash(it: Item | dict) -> str:
    return hashlib.sha256(item_line(it).encode()).hexdigest()


def content_hash(dataset: str, items: list[Item] | list[dict]) -> str:
    h = hashlib.sha256()
    h.update(json.dumps({"dataset": dataset, "schema": ITEM_SCHEMA}, sort_keys=True).encode() + b"\n")
    key = (lambda i: i.item_id) if items and isinstance(items[0], Item) else (lambda i: i["item_id"])
    for it in sorted(items, key=key):
        h.update(item_line(it).encode() + b"\n")
    return h.hexdigest()


# --------------------------------------------------------------------------------------------
# environment facts recorded in the manifest
# --------------------------------------------------------------------------------------------
def git_state() -> dict[str, Any]:
    here = Path(__file__).resolve().parent
    try:
        sha = subprocess.run(  # noqa: S603
            ["git", "rev-parse", "HEAD"], cwd=here, capture_output=True, text=True, timeout=10, check=True
        ).stdout.strip()
        dirty = bool(
            subprocess.run(  # noqa: S603
                ["git", "status", "--porcelain", "--untracked-files=no"],
                cwd=here, capture_output=True, text=True, timeout=10, check=True,
            ).stdout.strip()
        )
        return {"git_sha": sha, "git_dirty": dirty}
    except (OSError, subprocess.SubprocessError):
        return {"git_sha": None, "git_dirty": None}


def code_sha() -> str | None:
    return git_state()["git_sha"]


def _file_sha(path: Path) -> str | None:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return None


def register_state() -> dict[str, Any]:
    mp, cp = ms.register_path(), splits.clip_register_path()
    return {
        "match_splits_sha256": _file_sha(mp),
        "clip_splits_sha256": _file_sha(cp),
        "match_splits_path": str(mp),
        "clip_splits_path": str(cp),
        "in_match_rule": f"sha1(canonical clip id) % {splits.IN_MATCH_MODULUS} == {splits.IN_MATCH_REMAINDER}",
        "preview": not (mp.is_file() and cp.is_file()),
    }


# --------------------------------------------------------------------------------------------
# counts and diff
# --------------------------------------------------------------------------------------------
def count_summary(items: list[Item] | list[dict]) -> dict[str, Any]:
    def g(i: Any, k: str) -> Any:
        return getattr(i, k) if isinstance(i, Item) else i[k]

    by_split: Counter = Counter(g(i, "split") for i in items)
    by_kind: dict[str, Counter] = defaultdict(Counter)
    by_tier: dict[str, Counter] = defaultdict(Counter)
    clips: dict[str, set] = defaultdict(set)
    for i in items:
        by_kind[g(i, "split")][g(i, "label_kind")] += 1
        by_tier[g(i, "split")][str(g(i, "tier"))] += 1
        clips[g(i, "split")].add(g(i, "clip"))
    return {
        "n_items": len(items),
        "by_split": dict(sorted(by_split.items())),
        "by_split_kind": {s: dict(sorted(c.items())) for s, c in sorted(by_kind.items())},
        "by_split_tier": {s: dict(sorted(c.items())) for s, c in sorted(by_tier.items())},
        "clips_by_split": {s: len(c) for s, c in sorted(clips.items())},
    }


def diff_items(parent: dict[str, dict] | None, new: dict[str, dict]) -> dict[str, Any]:
    """``{added, removed, changed}`` item counts by split between two ``{item_id: row}`` maps.
    A changed item is one whose content hash differs (a label edit, or a split reassignment)."""
    parent = parent or {}
    added = Counter(r["split"] for k, r in new.items() if k not in parent)
    removed = Counter(r["split"] for k, r in parent.items() if k not in new)
    changed = Counter(
        r["split"] for k, r in new.items() if k in parent and parent[k]["item_hash"] != r["item_hash"]
    )
    return {
        "added": sum(added.values()),
        "removed": sum(removed.values()),
        "changed": sum(changed.values()),
        "added_by_split": dict(sorted(added.items())),
        "removed_by_split": dict(sorted(removed.items())),
        "changed_by_split": dict(sorted(changed.items())),
    }


# --------------------------------------------------------------------------------------------
# build / write / load
# --------------------------------------------------------------------------------------------
@dataclass
class Built:
    dataset: str
    items: list[Item]
    content_hash: str

    def rows(self, first_versions: dict[str, int], version: int) -> list[dict]:
        return [
            {
                "item_id": i.item_id, "dataset": self.dataset, "split": i.split,
                "match_key": i.match_key, "clip": i.clip, "game_id": i.game_id,
                "frame_index": i.frame_index, "label_kind": i.label_kind, "tier": i.tier,
                "payload": i.payload, "item_hash": item_hash(i),
                "first_version": first_versions.get(i.item_id, version),
            }
            for i in self.items
        ]


def build(dataset: str, conn: Any) -> Built:
    items = build_items(dataset, conn)
    return Built(dataset, items, content_hash(dataset, items))


@dataclass
class DatasetVersion:
    """A loaded snapshot."""

    dataset: str
    version: int
    manifest: dict
    table: pa.Table
    path: Path

    @property
    def content_hash(self) -> str:
        return self.manifest["content_hash"]

    def rows(self) -> list[dict]:
        return self.table.to_pylist()

    def select(self, split: str | None = None, cohort: int | None = None) -> list[dict]:
        """Items of ``split`` (``eval_all`` = both eval tiers; ``None`` = everything), restricted
        to those first seen at or before version ``cohort``."""
        rows = self.rows()
        if split == "eval_all":
            rows = [r for r in rows if r["split"] != splits.TRAIN]
        elif split:
            rows = [r for r in rows if r["split"] == split]
        if cohort is not None:
            rows = [r for r in rows if r["first_version"] <= cohort]
        return rows


def load_version(
    dataset: str, version: int | None = None, root: Path | None = None, *, verify: bool = True
) -> DatasetVersion:
    """Load ``v<version>`` (default: latest), re-deriving the content hash from the items."""
    if version is None:
        version = latest_version(dataset, root)
        if version is None:
            raise SnapshotError(f"no snapshots of {dataset!r} under {dataset_dir(dataset, root)}")
    vdir = version_dir(dataset, version, root)
    try:
        manifest = json.loads((vdir / "manifest.json").read_text())
        table = pq.read_table(vdir / "items.parquet")
    except OSError as e:
        raise SnapshotError(f"cannot read snapshot {vdir}: {e}") from e
    dv = DatasetVersion(dataset, version, manifest, table, vdir)
    if verify and content_hash(dataset, dv.rows()) != manifest["content_hash"]:
        raise SnapshotTampered(f"{vdir}: items do not hash to the manifest's content_hash")
    return dv


def _readonly(path: Path) -> None:
    mode = stat.S_IMODE(path.stat().st_mode)
    path.chmod(mode & ~(stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH))


def write_snapshot(dataset: str, version: int, rows: list[dict], manifest: dict, root: Path | None = None) -> Path:
    """Write ``v<version>`` once; raises :class:`SnapshotExists` if it is already there."""
    final = version_dir(dataset, version, root)
    if final.exists():
        raise SnapshotExists(f"{final} already exists: snapshots are immutable")
    final.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix=f".v{version}.tmp-", dir=final.parent))
    try:
        cols = {f.name: [r[f.name] for r in rows] for f in ITEM_SCHEMA_ARROW}
        table = pa.table(cols, schema=ITEM_SCHEMA_ARROW)
        table = table.replace_schema_metadata(
            {"dataset": dataset, "version": str(version), "content_hash": manifest["content_hash"]}
        )
        pq.write_table(table, tmp / "items.parquet", compression="zstd")
        (tmp / "manifest.json").write_text(json.dumps(manifest, indent=1, sort_keys=True) + "\n")
        for f in tmp.iterdir():
            _readonly(f)
        os.rename(tmp, final)  # atomic; fails if final appeared meanwhile
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    return final


# --------------------------------------------------------------------------------------------
# bump
# --------------------------------------------------------------------------------------------
@dataclass
class BumpResult:
    dataset: str
    changed: bool
    version: int | None  # the new version, or the unchanged latest
    content_hash: str
    manifest: dict | None = None
    path: Path | None = None


def _check_registers(allow_preview: bool) -> None:
    if allow_preview:
        return
    missing = [str(p) for p in (ms.register_path(), splits.clip_register_path()) if not p.is_file()]
    if missing:
        raise SnapshotError(
            "refusing to bump from split registers that do not exist yet "
            f"({', '.join(missing)}): the split of every item would be a preview. Create them "
            "(`python -m footy_track.splits assign --apply`) or pass --allow-preview for a throwaway version."
        )


def _sync_store_index(conn: Any, dataset: str, root: Path | None) -> None:
    """Upsert every snapshot manifest of ``dataset`` into ``dataset_version`` (files win)."""
    for v in list_versions(dataset, root):
        vdir = version_dir(dataset, v, root)
        m = json.loads((vdir / "manifest.json").read_text())
        conn.execute(
            "INSERT OR REPLACE INTO dataset_version VALUES (?,?,?,?,?,?,?,?,?,?)",
            [dataset, v, m["content_hash"], m["created_at"], m.get("parent_version"),
             m["counts"]["n_items"], m.get("git_sha"), m.get("note"), str(vdir),
             json.dumps(m, sort_keys=True)],
        )


def bump(
    dataset: str,
    db: Path | None = None,
    root: Path | None = None,
    *,
    note: str | None = None,
    allow_preview: bool = False,
    allow_empty: bool = False,
    now: datetime | None = None,
) -> BumpResult:
    """Write ``v<N+1>`` of ``dataset`` iff its content hash differs from the latest version."""
    if dataset not in DATASETS:
        raise DatasetBuildError(f"unknown dataset {dataset!r}; choose from {DATASETS}")
    _check_registers(allow_preview)
    db = Path(db) if db else default_db_path()
    with store_lock(db, exclusive=True):
        store = FeatureStore.open(db)
        try:
            built = build(dataset, store._conn)  # noqa: SLF001
            parent_v = latest_version(dataset, root)
            parent = load_version(dataset, parent_v, root) if parent_v is not None else None
            if parent is not None and parent.content_hash == built.content_hash:
                _sync_store_index(store._conn, dataset, root)  # noqa: SLF001
                return BumpResult(dataset, False, parent_v, built.content_hash, parent.manifest, parent.path)
            if not built.items and not allow_empty:
                raise SnapshotError(f"{dataset}: the item table is empty; ingest first (or --allow-empty)")
            version = (parent_v or 0) + 1
            parent_rows = {r["item_id"]: r for r in parent.rows()} if parent else {}
            first = {k: r["first_version"] for k, r in parent_rows.items()}
            rows = built.rows(first, version)
            new_rows = {r["item_id"]: r for r in rows}
            unreg = sum(1 for i in built.items if splits.unregistered_match(i.match_key, i.clip))
            manifest = {
                "dataset": dataset,
                "version": version,
                "created_at": (now or datetime.now(UTC)).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "content_hash": built.content_hash,
                "item_schema": ITEM_SCHEMA,
                "parent_version": parent_v,
                "parent_content_hash": parent.content_hash if parent else None,
                "note": note,
                "counts": count_summary(rows),
                "n_items_unregistered_match": unreg,
                "split_registers": register_state(),
                **git_state(),
                "diff_vs_parent": diff_items(parent_rows, new_rows) | {"parent_version": parent_v},
            }
            path = write_snapshot(dataset, version, rows, manifest, root)
            _sync_store_index(store._conn, dataset, root)  # noqa: SLF001
            return BumpResult(dataset, True, version, built.content_hash, manifest, path)
        finally:
            store.close()


def diff_versions(dataset: str, a: int, b: int, root: Path | None = None) -> dict[str, Any]:
    """Item-level diff between two snapshots (``a`` older), with the changed item ids."""
    ra = {r["item_id"]: r for r in load_version(dataset, a, root).rows()}
    rb = {r["item_id"]: r for r in load_version(dataset, b, root).rows()}
    out = diff_items(ra, rb) | {"from": a, "to": b}
    out["added_ids"] = sorted(k for k in rb if k not in ra)
    out["removed_ids"] = sorted(k for k in ra if k not in rb)
    out["changed_ids"] = sorted(k for k in rb if k in ra and ra[k]["item_hash"] != rb[k]["item_hash"])
    return out
