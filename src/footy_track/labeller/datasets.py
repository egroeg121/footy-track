"""Dataset registry: one labeller server, many datasets.

A *dataset* is one body of clips plus where its labels live: a clips dir, a
marks (GT) dir, optional machine-candidate dir and ball-check review list, and
optional extra clip dirs for pitch calibration. Every data-touching route
resolves its dataset per request from the ``?dataset=<name>`` query parameter
(default: the registry's default, ``calib``), so existing links and bookmarks
keep working and one server can serve calib, norwich, ... side by side.

Resolution
----------
Datasets come from a JSON registry (``$FOOTY_LABELLER_DATASETS``, default
``/mnt/storage/footy_data/labeller_datasets.json``)::

    {"default": "calib",
     "datasets": {
       "calib": {"label": "Calibration + ball check",
                 "clips_dir": "...", "marks_dir": "...",
                 "candidates_dir": "...",              # optional
                 "ball_check_list": "...",             # optional
                 "extra_clips_dirs": ["..."]}}}        # optional

The registry is re-read when its mtime/size changes, so adding a dataset needs
no restart. With no registry file the single legacy dataset ``calib`` is built
from the environment (``FOOTY_CLIPS_DIR`` ...), exactly as before. The
legacy env vars also override the *default* dataset's fields when set (this is
what ``python -m footy_track.labeller --clips-dir`` relies on).

The per-request choice is carried in a :class:`~contextvars.ContextVar` set by
:class:`DatasetMiddleware` (HTTP and WebSocket), which asyncio tasks and
``to_thread`` workers inherit. Code on plain threads (the debounced sidecar
flush) must not rely on it: ``Session`` binds its dataset name explicitly.
"""

from __future__ import annotations

import contextlib
import contextvars
import json
import logging
import os
import re
import tempfile
import threading
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import parse_qs

LOGGER = logging.getLogger(__name__)

DEFAULT_REGISTRY_FILE = "/mnt/storage/footy_data/labeller_datasets.json"
FALLBACK_DEFAULT = "calib"
_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")


class UnknownDatasetError(KeyError):
    """The requested dataset name is not in the registry."""


@dataclass(frozen=True)
class Dataset:
    name: str
    label: str
    clips_dir: Path
    marks_dir: Path
    candidates_dir: Path
    ball_check_list: Path | None = None
    extra_clips_dirs: tuple[Path, ...] = ()
    identity_labels_dir: Path | None = None
    tracklets_dir: Path | None = None
    embeddings_dir: Path | None = None
    is_default: bool = False
    extra: dict = field(default_factory=dict, compare=False, hash=False)

    def to_public(self) -> dict:
        return {
            "name": self.name,
            "label": self.label,
            "is_default": self.is_default,
            "clips_dir": str(self.clips_dir),
            "marks_dir": str(self.marks_dir),
            "candidates_dir": str(self.candidates_dir),
            "ball_check_list": str(self.ball_check_list)
            if self.ball_check_list
            else None,
            "extra_clips_dirs": [str(p) for p in self.extra_clips_dirs],
        }


# ---------------------------------------------------------------------------
# Legacy (single-dataset, env-driven) configuration
# ---------------------------------------------------------------------------


def _env_dir(var: str, *fallbacks: Path) -> Path:
    """First existing candidate: $var, then each fallback, else fallbacks[0].

    The GT/clip locations differ per machine (macOS iCloud on the laptop, plain
    directories on Linux boxes and rented GPUs). Hardcoding the macOS path made
    Linux hosts resolve to an empty ``~/Library/Mobile Documents/...`` directory,
    so the labeller silently showed no existing labels and there was no way to
    tell what still needed doing. Resolve by environment instead.
    """
    override = os.environ.get(var)
    if override:
        return Path(override).expanduser()
    for cand in fallbacks:
        if cand.is_dir():
            return cand
    return fallbacks[0]


def _legacy_default(name: str) -> Dataset:
    """The pre-registry configuration, resolved from the environment now."""
    clips = _env_dir(
        "FOOTY_CLIPS_DIR",
        Path(__file__).parents[3] / "eval_data" / "clips",
        Path.home() / "footy" / "footy_data" / "clips",
    )
    marks = _env_dir(
        "FOOTY_GT_MARKS_DIR",
        Path.home() / "footy" / "footy_data" / "ball_gt_marks",
        Path.home()
        / "Library"
        / "Mobile Documents"
        / "com~apple~CloudDocs"
        / "footy_data"
        / "ball_gt_marks",
    )
    # Machine detections with real confidences (one JSONL per clip, ball rows
    # only), pulled from s3://.../footy_data/machine_labels.
    cands = _env_dir(
        "FOOTY_BALL_CANDIDATES_DIR",
        Path.home() / "footy" / "footy_data" / "ball_candidates",
    )
    raw_list = os.environ.get("FOOTY_BALL_CHECK_LIST") or None
    extra = tuple(
        Path(d).expanduser()
        for d in filter(
            None, os.environ.get("FOOTY_CALIB_EXTRA_CLIPS_DIRS", "").split(os.pathsep)
        )
    )
    return _finish(
        name,
        label=name,
        clips_dir=clips,
        marks_dir=marks,
        candidates_dir=cands,
        ball_check_list=Path(raw_list).expanduser() if raw_list else None,
        extra_clips_dirs=extra,
        is_default=True,
    )


def _finish(
    name: str,
    *,
    label: str,
    clips_dir: Path,
    marks_dir: Path,
    candidates_dir: Path | None,
    ball_check_list: Path | None,
    extra_clips_dirs: tuple[Path, ...],
    identity_labels_dir: Path | None = None,
    tracklets_dir: Path | None = None,
    embeddings_dir: Path | None = None,
    is_default: bool,
) -> Dataset:
    """Fill the derived defaults.

    The default dataset keeps the historical ``<marks>/../identity_labels`` and
    ``../tracklets`` locations; any other dataset keeps them *inside* its own
    marks dir so identity labels cannot bleed between datasets. Same for the
    candidate crop descriptors (``ball_embeddings``, keyed by clip stem).
    """
    cands = candidates_dir or marks_dir / "ball_candidates"
    emb = embeddings_dir or (
        cands.parent / "ball_embeddings"
        if is_default
        else cands.parent / f"ball_embeddings_{name}"
    )
    if is_default:
        ident = identity_labels_dir or marks_dir.parent / "identity_labels"
        tracks = tracklets_dir or marks_dir.parent / "tracklets"
    else:
        ident = identity_labels_dir or marks_dir / "identity_labels"
        tracks = tracklets_dir or marks_dir / "tracklets"
    return Dataset(
        name=name,
        label=label,
        clips_dir=clips_dir,
        marks_dir=marks_dir,
        candidates_dir=cands,
        ball_check_list=ball_check_list,
        extra_clips_dirs=extra_clips_dirs,
        identity_labels_dir=ident,
        tracklets_dir=tracks,
        embeddings_dir=emb,
        is_default=is_default,
    )


# ---------------------------------------------------------------------------
# Registry file
# ---------------------------------------------------------------------------


def registry_path() -> Path:
    return Path(
        os.environ.get("FOOTY_LABELLER_DATASETS") or DEFAULT_REGISTRY_FILE
    ).expanduser()


@dataclass
class _Loaded:
    sig: tuple | None
    default: str
    datasets: dict[str, Dataset]
    error: str | None = None


_LOCK = threading.Lock()
_CACHE: _Loaded | None = None


def invalidate() -> None:
    """Drop the parsed registry (tests; a rewritten file is also noticed by stat)."""
    global _CACHE  # noqa: PLW0603
    with _LOCK:
        _CACHE = None


def _path_or_none(v: object) -> Path | None:
    return Path(str(v)).expanduser() if v else None


def _parse_entry(name: str, raw: dict, *, is_default: bool) -> Dataset:
    if not _NAME_RE.match(name):
        raise ValueError(f"bad dataset name {name!r}")
    clips = _path_or_none(raw.get("clips_dir"))
    marks = _path_or_none(raw.get("marks_dir"))
    if clips is None or marks is None:
        raise ValueError(f"dataset {name!r} needs clips_dir and marks_dir")
    extra = raw.get("extra_clips_dirs") or []
    if isinstance(extra, str):
        extra = [p for p in extra.split(os.pathsep) if p]
    return _finish(
        name,
        label=str(raw.get("label") or name),
        clips_dir=clips,
        marks_dir=marks,
        candidates_dir=_path_or_none(raw.get("candidates_dir")),
        ball_check_list=_path_or_none(raw.get("ball_check_list")),
        extra_clips_dirs=tuple(Path(str(p)).expanduser() for p in extra),
        identity_labels_dir=_path_or_none(raw.get("identity_labels_dir")),
        tracklets_dir=_path_or_none(raw.get("tracklets_dir")),
        embeddings_dir=_path_or_none(raw.get("embeddings_dir")),
        is_default=is_default,
    )


def _apply_env_overrides(raw: dict) -> dict:
    """Legacy ``FOOTY_*`` env vars win over the registry for the default dataset."""
    env = os.environ
    out = dict(raw)
    for key, var in (
        ("clips_dir", "FOOTY_CLIPS_DIR"),
        ("marks_dir", "FOOTY_GT_MARKS_DIR"),
        ("candidates_dir", "FOOTY_BALL_CANDIDATES_DIR"),
        ("ball_check_list", "FOOTY_BALL_CHECK_LIST"),
    ):
        if env.get(var):
            out[key] = env[var]
    if env.get("FOOTY_CALIB_EXTRA_CLIPS_DIRS"):
        out["extra_clips_dirs"] = env["FOOTY_CALIB_EXTRA_CLIPS_DIRS"]
    return out


def _env_signature() -> tuple:
    return tuple(
        os.environ.get(k)
        for k in (
            "FOOTY_CLIPS_DIR",
            "FOOTY_GT_MARKS_DIR",
            "FOOTY_BALL_CANDIDATES_DIR",
            "FOOTY_BALL_CHECK_LIST",
            "FOOTY_CALIB_EXTRA_CLIPS_DIRS",
        )
    )


def _load() -> _Loaded:
    global _CACHE  # noqa: PLW0603
    path = registry_path()
    try:
        st = path.stat()
        sig: tuple | None = (str(path), st.st_mtime_ns, st.st_size, _env_signature())
    except OSError:
        sig = (str(path), None, None, _env_signature())
    with _LOCK:
        if _CACHE is not None and _CACHE.sig == sig:
            return _CACHE
    error = None
    datasets: dict[str, Dataset] = {}
    default = FALLBACK_DEFAULT
    if sig[1] is not None:
        try:
            doc = json.loads(path.read_text())
            entries = doc["datasets"]
            if not isinstance(entries, dict) or not entries:
                raise ValueError("'datasets' must be a non-empty object")
            default = str(doc.get("default") or next(iter(entries)))
            if default not in entries:
                raise ValueError(f"default dataset {default!r} is not listed")
            for name, raw in entries.items():
                is_default = name == default
                datasets[name] = _parse_entry(
                    name,
                    _apply_env_overrides(raw) if is_default else raw,
                    is_default=is_default,
                )
        except (OSError, ValueError, KeyError, TypeError) as e:
            error = f"{path}: {type(e).__name__}: {e}"
            LOGGER.error(
                "labeller datasets registry unusable, falling back to legacy default: %s",
                error,
            )
            datasets = {}
            default = FALLBACK_DEFAULT
    if not datasets:
        datasets = {default: _legacy_default(default)}
    loaded = _Loaded(sig, default, datasets, error)
    with _LOCK:
        _CACHE = loaded
    return loaded


def default_name() -> str:
    return _load().default


def names() -> list[str]:
    return list(_load().datasets)


def get(name: str | None = None) -> Dataset:
    """Dataset by name (``None``/empty = the default). Raises UnknownDatasetError."""
    loaded = _load()
    key = name or loaded.default
    try:
        return loaded.datasets[key]
    except KeyError:
        raise UnknownDatasetError(key) from None


def all_datasets() -> list[Dataset]:
    return list(_load().datasets.values())


def registry_error() -> str | None:
    return _load().error


def write_registry(path: Path, default: str, entries: dict[str, dict]) -> None:
    """Atomically write a registry file (used by the CLI and tests)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as f:
            json.dump({"default": default, "datasets": entries}, f, indent=2)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp)
        raise
    invalidate()


# ---------------------------------------------------------------------------
# Per-request selection
# ---------------------------------------------------------------------------

_CURRENT: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "footy_dataset", default=None
)


def current() -> Dataset:
    """The dataset of the request being served (default outside a request)."""
    return get(_CURRENT.get())


def current_name() -> str:
    return current().name


@contextlib.contextmanager
def using(name: str | None) -> Iterator[Dataset]:
    """Run a block with ``name`` as the current dataset (tests, scripts)."""
    ds = get(name)
    token = _CURRENT.set(ds.name)
    try:
        yield ds
    finally:
        _CURRENT.reset(token)


class DatasetMiddleware:
    """Pure ASGI middleware: ``?dataset=<name>`` -> the request's current dataset.

    Pure ASGI (not ``BaseHTTPMiddleware``) so it also covers WebSocket
    handshakes. An unknown name is a hard 404 (never a silent fallback to the
    default dataset: that would write labels into the wrong place).
    """

    def __init__(self, app) -> None:
        self.app = app

    async def __call__(self, scope, receive, send) -> None:
        if scope["type"] not in ("http", "websocket"):
            await self.app(scope, receive, send)
            return
        qs = parse_qs(scope.get("query_string", b"").decode("latin-1"))
        name = (qs.get("dataset") or [""])[-1] or None
        if name is not None:
            try:
                get(name)
            except UnknownDatasetError:
                await self._reject(scope, receive, send, name)
                return
        token = _CURRENT.set(name)
        try:
            await self.app(scope, receive, send)
        finally:
            _CURRENT.reset(token)

    @staticmethod
    async def _reject(scope, receive, send, name: str) -> None:
        if scope["type"] == "websocket":
            await receive()  # websocket.connect
            await send({"type": "websocket.close", "code": 4404})
            return
        body = json.dumps(
            {"detail": f"unknown dataset {name!r}", "datasets": names()}
        ).encode()
        await send(
            {
                "type": "http.response.start",
                "status": 404,
                "headers": [
                    (b"content-type", b"application/json"),
                    (b"content-length", str(len(body)).encode()),
                ],
            }
        )
        await send({"type": "http.response.body", "body": body})
