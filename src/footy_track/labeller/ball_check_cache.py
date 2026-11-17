"""Disk cache for rendered Ball Check stills and clips (stdlib only).

The phone only ever downloads finished files; this is where they live so a
server restart does not throw the work away and a Prefect run can render a whole
job ahead of time (``scripts/ballcheck_prepare.py``).

Layout under :func:`root` (``$FOOTY_BALLCHECK_RENDERS_DIR``, default
``/mnt/storage/footy_data/ballcheck_renders``; ``off`` disables the disk layer)::

    blobs/<key[:2]>/<key>.jpg | .mp4 | .gif   the rendered bytes (+ <key>.meta.json for clips)
    refs/<job id>.json                         which blobs a job wants (so they die together)
    .tmp/                                      write staging; never listed or served

* **Key** = sha256 over everything that changes the bytes (see ``ball_check.crop_disk_key`` /
  ``clip_disk_key``: clip file identity, frame, box geometry, other box, pad, reticle, ``at``,
  render version). Equal keys mean equal bytes, so blobs are shared between jobs.
* **Atomic**: written to ``.tmp/`` then ``os.replace``d; a half-written file is never visible.
* **De-duplicated**: :class:`Coalescer` makes concurrent renders of one key a single render.
* **Bounded and temporary** (:func:`gc`): a job's renders (its ref) are deleted 24 h after the
  job is done/cancelled, or after ``idle`` seconds with no labelling activity on it; the whole
  cache is capped (default 5 GB) by LRU eviction that takes unreferenced and finished-job renders
  before renders of open jobs; preparing refuses to run below a free-space floor.

Verdicts, jobs and cards are never stored here, so nothing in this directory is precious.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import shutil
import tempfile
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

ENV_DIR = "FOOTY_BALLCHECK_RENDERS_DIR"
ENV_MAX_GB = "FOOTY_BALLCHECK_CACHE_MAX_GB"
ENV_MIN_FREE_GB = "FOOTY_BALLCHECK_MIN_FREE_GB"
DEFAULT_DIR = "/mnt/storage/footy_data/ballcheck_renders"
DEFAULT_MAX_GB = 5.0
DEFAULT_MIN_FREE_GB = 20.0
GB = 1024**3

#: Renders of a done/cancelled job are deleted this long after it closed.
GRACE_S = 24 * 3600
#: Renders of a live job are deleted after this long without labelling activity on it.
IDLE_S = 3 * 24 * 3600
#: A render nobody references (an on-demand render of a non-job card) goes after this long unused.
ORPHAN_TTL_S = 7 * 24 * 3600

EXTS = ("jpg", "mp4", "gif")
ACTIVE = ("preparing", "open", "in_progress")


class DiskFloorError(RuntimeError):
    """Writing renders would leave less than the configured free space."""


def _gb_env(var: str, default: float) -> float:
    try:
        return float(os.environ.get(var, default))
    except ValueError:
        return default


def root() -> Path | None:
    raw = os.environ.get(ENV_DIR, DEFAULT_DIR).strip()
    if not raw or raw.lower() in ("off", "none", "0"):
        return None
    return Path(raw).expanduser()


def max_bytes() -> int:
    return int(_gb_env(ENV_MAX_GB, DEFAULT_MAX_GB) * GB)


def min_free_bytes() -> int:
    return int(_gb_env(ENV_MIN_FREE_GB, DEFAULT_MIN_FREE_GB) * GB)


def make_key(parts: dict) -> str:
    """Stable 40-hex key of a JSON-able dict (floats should already be rounded)."""
    raw = json.dumps(parts, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode()).hexdigest()[:40]


def free_bytes(path: Path) -> int:
    p = path
    while not p.exists() and p != p.parent:
        p = p.parent
    return shutil.disk_usage(p).free


def check_floor(path: Path, need_bytes: int = 0, floor: int | None = None) -> None:
    """Raise :class:`DiskFloorError` unless ``need_bytes`` still leaves ``floor`` free."""
    floor = min_free_bytes() if floor is None else floor
    free = free_bytes(path)
    if free - need_bytes < floor:
        raise DiskFloorError(
            f"only {free / GB:.1f} GB free on the disk holding {path}; need {need_bytes / GB:.2f} GB "
            f"and a floor of {floor / GB:.1f} GB (set {ENV_MIN_FREE_GB} to change)"
        )


# ---------------------------------------------------------------------------
# In-flight de-duplication
# ---------------------------------------------------------------------------


class _Flight:
    __slots__ = ("done", "error", "value")

    def __init__(self) -> None:
        self.done = threading.Event()
        self.value = None
        self.error: BaseException | None = None


class Coalescer:
    """Run ``fn`` once per key at a time; concurrent callers share the leader's result."""

    def __init__(self) -> None:
        self._mu = threading.Lock()
        self._flight: dict[str, _Flight] = {}
        self.leaders = 0
        self.followers = 0

    def run(self, key: str, fn: Callable[[], object]):
        with self._mu:
            ent = self._flight.get(key)
            leader = ent is None
            if leader:
                ent = self._flight[key] = _Flight()
                self.leaders += 1
            else:
                self.followers += 1
        if not leader:
            ent.done.wait()
            if ent.error is not None:
                raise ent.error
            return ent.value
        try:
            ent.value = fn()
        except BaseException as e:
            ent.error = e
            raise
        finally:
            with self._mu:
                self._flight.pop(key, None)
            ent.done.set()
        return ent.value


COALESCER = Coalescer()


# ---------------------------------------------------------------------------
# The cache
# ---------------------------------------------------------------------------


@dataclass
class Blob:
    key: str
    ext: str
    path: Path
    size: int  # bytes of the blob plus its meta file
    mtime: float


class RenderCache:
    def __init__(self, base: Path) -> None:
        self.base = Path(base)
        self.blobs = self.base / "blobs"
        self.refs = self.base / "refs"
        self.tmp = self.base / ".tmp"
        self._puts = 0
        self._put_bytes = 0

    # -- paths ---------------------------------------------------------------
    def blob_path(self, key: str, ext: str) -> Path:
        return self.blobs / key[:2] / f"{key}.{ext}"

    def _meta_path(self, key: str) -> Path:
        return self.blobs / key[:2] / f"{key}.meta.json"

    # -- read / write ----------------------------------------------------------
    def has(self, key: str, ext: str) -> bool:
        return self.blob_path(key, ext).is_file()

    def get(
        self, key: str, ext: str, *, touch: bool = True
    ) -> tuple[bytes, dict] | None:
        """``(bytes, meta)`` of a cached render, or None. A hit refreshes its LRU time."""
        p = self.blob_path(key, ext)
        try:
            data = p.read_bytes()
        except OSError:
            return None
        meta: dict = {}
        with contextlib.suppress(OSError, ValueError):
            meta = json.loads(self._meta_path(key).read_text())
        if touch:
            with contextlib.suppress(OSError):
                os.utime(p, None)
        return data, meta

    def put(self, key: str, ext: str, data: bytes, meta: dict | None = None) -> Path:
        """Atomically store ``data`` (temp file outside ``blobs/``, then rename)."""
        dst = self.blob_path(key, ext)
        dst.parent.mkdir(parents=True, exist_ok=True)
        self.tmp.mkdir(parents=True, exist_ok=True)
        if meta:  # meta first: the blob appearing is the commit
            self._atomic(self._meta_path(key), json.dumps(meta).encode())
        self._atomic(dst, data)
        self._puts += 1
        self._put_bytes += len(data)
        return dst

    def _atomic(self, dst: Path, data: bytes) -> None:
        fd, tmp = tempfile.mkstemp(prefix=dst.name + ".", suffix=".part", dir=self.tmp)
        try:
            with os.fdopen(fd, "wb") as f:
                f.write(data)
                f.flush()
                os.fsync(f.fileno())
            os.chmod(tmp, 0o664)
            os.replace(tmp, dst)
        except BaseException:
            with contextlib.suppress(OSError):
                os.unlink(tmp)
            raise

    def get_or_render(
        self, key: str, ext: str, render: Callable[[], tuple[bytes, dict] | None]
    ) -> tuple[tuple[bytes, dict] | None, str]:
        """Disk first, else ``render()`` once (concurrent callers share it) and store it.

        Returns ``(result, how)`` with ``how`` ``disk`` | ``rendered`` | ``coalesced``.
        A disk write that would break the free-space floor is skipped, not fatal.
        """
        how = ["disk"]

        def work():
            hit = self.get(key, ext)
            if hit is not None:
                return hit
            out = render()
            how[0] = "rendered"
            if out is None:
                return None
            data, meta = out
            try:
                check_floor(self.base, len(data))
                self.put(key, ext, data, meta)
            except (OSError, DiskFloorError):
                pass
            return out

        led = COALESCER.leaders
        res = COALESCER.run(f"{self.base}:{key}", work)
        if COALESCER.leaders == led and how[0] == "disk" and res is not None:
            how[0] = "coalesced"  # we waited on another caller's render
        return res, how[0]

    # -- per-job references ------------------------------------------------------
    def _ref_path(self, job: str) -> Path:
        return self.refs / f"{job}.json"

    def register(
        self, job: str, keys: list[tuple[str, str]], *, now: float | None = None
    ) -> None:
        """Record that ``job`` wants these ``(key, ext)`` blobs (merged with earlier ones)."""
        cur = self.read_ref(job)
        have = {tuple(k) for k in cur.get("keys", [])}
        have.update((k, e) for k, e in keys)
        doc = {"job": job, "updated": now or time.time(), "keys": sorted(have)}
        self.refs.mkdir(parents=True, exist_ok=True)
        self.tmp.mkdir(parents=True, exist_ok=True)
        self._atomic(self._ref_path(job), json.dumps(doc).encode())

    def read_ref(self, job: str) -> dict:
        try:
            doc = json.loads(self._ref_path(job).read_text())
        except (OSError, ValueError):
            return {}
        return doc if isinstance(doc, dict) else {}

    def all_refs(self) -> dict[str, dict]:
        out: dict[str, dict] = {}
        if self.refs.is_dir():
            for p in self.refs.glob("*.json"):
                doc = self.read_ref(p.stem)
                if doc:
                    doc["_mtime"] = p.stat().st_mtime
                    out[p.stem] = doc
        return out

    def drop_ref(self, job: str) -> None:
        with contextlib.suppress(OSError):
            self._ref_path(job).unlink()

    # -- inventory -----------------------------------------------------------------
    def list_blobs(self) -> list[Blob]:
        out: list[Blob] = []
        if not self.blobs.is_dir():
            return out
        for d in self.blobs.iterdir():
            if not d.is_dir():
                continue
            for p in d.iterdir():
                ext = p.suffix.lstrip(".")
                if ext not in EXTS:
                    continue
                try:
                    st = p.stat()
                except OSError:
                    continue
                size = st.st_size
                with contextlib.suppress(OSError):
                    size += self._meta_path(p.stem).stat().st_size
                out.append(Blob(p.stem, ext, p, size, st.st_mtime))
        return out

    def total_bytes(self) -> int:
        return sum(b.size for b in self.list_blobs())

    def delete_blob(self, b: Blob) -> int:
        freed = 0
        for p in (b.path, self._meta_path(b.key)):
            try:
                freed += p.stat().st_size
                p.unlink()
            except OSError:
                continue
        with contextlib.suppress(OSError):
            b.path.parent.rmdir()  # only succeeds when empty
        return freed


_CACHES: dict[str, RenderCache] = {}


def cache() -> RenderCache | None:
    """The configured cache, or None when the disk layer is off."""
    base = root()
    if base is None:
        return None
    key = str(base)
    if key not in _CACHES:
        _CACHES[key] = RenderCache(base)
    return _CACHES[key]


# ---------------------------------------------------------------------------
# Housekeeping
# ---------------------------------------------------------------------------


def gc(  # noqa: PLR0912, PLR0915
    rc: RenderCache,
    jobs: dict[str, dict],
    now: float | None = None,
    *,
    cap: int | None = None,
    grace_s: float = GRACE_S,
    idle_s: float = IDLE_S,
    orphan_ttl_s: float = ORPHAN_TTL_S,
    dry_run: bool = False,
) -> dict:
    """Drop stale job references, delete the renders nobody wants, then enforce the size cap.

    ``jobs``: ``{job_id: {"status", "closed_ts", "last_activity"}}`` (epoch seconds; a job missing
    from the map counts as finished). In order:

    1. a ref goes when its job closed (done/cancelled) ``grace_s`` ago, or is live but idle for
       ``idle_s``, or is gone;
    2. blobs that no remaining ref names go if they were named by a dropped ref, or are orphans
       (on-demand renders of non-job cards) unused for ``orphan_ttl_s``;
    3. over ``cap`` bytes, evict least recently used first: orphans, then renders only finished
       jobs want, and only as a last resort renders of live jobs.

    Never touches verdicts, jobs or cards. Returns a report; ``dry_run`` deletes nothing.
    """
    now = time.time() if now is None else now
    cap = max_bytes() if cap is None else cap
    refs = rc.all_refs()
    dropped: dict[str, str] = {}
    for job, doc in refs.items():
        info = jobs.get(job)
        if info is None:
            if now - doc.get("_mtime", now) >= grace_s:
                dropped[job] = "job gone"
        elif info["status"] in ("done", "cancelled"):
            if now - (info.get("closed_ts") or doc.get("_mtime", now)) >= grace_s:
                dropped[job] = "finished"
        elif now - (info.get("last_activity") or doc.get("_mtime", now)) >= idle_s:
            dropped[job] = "idle"

    kept = {j: d for j, d in refs.items() if j not in dropped}
    wanted = {tuple(k) for d in kept.values() for k in d.get("keys", [])}
    gone_keys = {tuple(k) for j in dropped for k in refs[j].get("keys", [])}
    blobs = rc.list_blobs()
    deleted: list[Blob] = []
    survivors: list[Blob] = []
    for b in blobs:
        k = (b.key, b.ext)
        if k in wanted:
            survivors.append(b)
        elif k in gone_keys or now - b.mtime >= orphan_ttl_s:
            deleted.append(b)
        else:
            survivors.append(b)

    total_before = sum(b.size for b in blobs)
    total = sum(b.size for b in survivors)
    evicted: list[Blob] = []
    if total > cap:

        def keys_of(statuses: tuple[str, ...]) -> set[tuple]:
            return {
                tuple(k)
                for j, d in kept.items()
                if jobs.get(j, {}).get("status") in statuses
                for k in d.get("keys", [])
            }

        active_keys = keys_of(ACTIVE)
        finished_keys = keys_of(("done", "cancelled")) - active_keys

        def klass(b: Blob) -> int:
            k = (b.key, b.ext)
            if k not in wanted:
                return 0  # orphan
            return 1 if k in finished_keys else 2

        for b in sorted(survivors, key=lambda b: (klass(b), b.mtime)):
            if total <= cap:
                break
            evicted.append(b)
            total -= b.size
    freed = 0
    if not dry_run:
        for job in dropped:
            rc.drop_ref(job)
        for b in [*deleted, *evicted]:
            freed += rc.delete_blob(b)
    else:
        freed = sum(b.size for b in [*deleted, *evicted])
    if not dry_run and rc.tmp.is_dir():  # leftovers of a write that died mid-way
        for p in rc.tmp.iterdir():
            with contextlib.suppress(OSError):
                if now - p.stat().st_mtime > 6 * 3600:
                    p.unlink()
    return {
        "dry_run": dry_run,
        "dropped_refs": dropped,
        "deleted_files": len(deleted),
        "evicted_for_cap": len(evicted),
        "freed_bytes": freed,
        "bytes_before": total_before,
        "bytes_after": total_before - freed,
        "cap_bytes": cap,
        "evicted_order": [(b.key, b.ext) for b in evicted],
    }
