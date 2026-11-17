"""Per-match split register (tier 1 of the split authority, :mod:`footy_track.splits`): which whole matches are held-out (evaluation only) vs training pool.

One persistent JSON file (default ``/mnt/storage/footy_data/match_splits.json``, override with env
``FOOTY_MATCH_SPLITS``)::

    {"<match key>": {"split": "heldout" | "train", "reason": "...", "assigned": "<iso>",
                     "aliases": ["<clip-name prefix>", ...]}}

``heldout`` matches never reach a training export (nor val/test split); ``train`` matches go into
the training pool, whose *clips* are then split train/val by the existing clip-level logic.
New downloaded matches are assigned once (``sha1(key) % 4 == 0`` -> heldout, ~25%) by the jobs repo
(``footy_splits.py``); manual edits always win. The rule here mirrors that file.

Fail closed: a missing or invalid register raises ``RegisterError`` (a host without the file
would otherwise leak newly held-out matches into training). Only with
``FOOTY_MATCH_SPLITS_ALLOW_SEED=1`` (tests/CI) is a *missing* file replaced by the built-in seed
``DEFAULT_REGISTER``, i.e. the old hard-coded tuple; an invalid file always raises.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

from footy_track.clip_id import canonical_clip_id

DEFAULT_PATH = "/mnt/storage/footy_data/match_splits.json"
HELDOUT, TRAIN = "heldout", "train"

#: Fallback register (also the seed written to disk). ``arsenal_norwich`` is the old, mistaken
#: name of the Nottingham Forest match.
DEFAULT_REGISTER: dict[str, dict] = {
    "arsenal_nottinghamforest": {
        "split": HELDOUT,
        "reason": "only match absent from every detector training set (seed)",
        "assigned": "2026-11-17T00:00:00+00:00",
        "aliases": ["arsenal_norwich", "nottinghamforest"],
    },
}

_cache: dict = {"path": None, "mtime": None, "data": None, "t": 0.0}
_CACHE_TTL_S = 5.0


def register_path() -> Path:
    return Path(os.environ.get("FOOTY_MATCH_SPLITS", DEFAULT_PATH))


class RegisterError(RuntimeError):
    """The split register is missing or invalid (fail closed: never guess held-out matches)."""


def _allow_seed() -> bool:
    return os.environ.get("FOOTY_MATCH_SPLITS_ALLOW_SEED", "") not in ("", "0")


def load_register(
    path: Path | None = None, allow_seed: bool | None = None
) -> dict[str, dict]:
    """The register. Raises ``RegisterError`` when the file is invalid (always) or missing
    (unless ``allow_seed`` / env ``FOOTY_MATCH_SPLITS_ALLOW_SEED=1``, which uses the built-in
    ``DEFAULT_REGISTER`` seed - tests/CI only). Cached briefly."""
    p = path or register_path()
    now = time.monotonic()
    if (
        _cache["data"] is not None
        and _cache["path"] == str(p)
        and now - _cache["t"] < _CACHE_TTL_S
    ):
        return _cache["data"]
    try:
        text = p.read_text()
    except FileNotFoundError:
        if allow_seed if allow_seed is not None else _allow_seed():
            data = DEFAULT_REGISTER
            _cache.update(path=str(p), data=data, t=now)
            return data
        raise RegisterError(
            f"match split register not found: {p} (set FOOTY_MATCH_SPLITS, or "
            "FOOTY_MATCH_SPLITS_ALLOW_SEED=1 to use the built-in seed for tests)"
        ) from None
    except OSError as e:
        raise RegisterError(f"cannot read match split register {p}: {e}") from e
    try:
        data = json.loads(text)
        if not isinstance(data, dict) or not all(
            isinstance(v, dict) and v.get("split") in (HELDOUT, TRAIN)
            for v in data.values()
        ):
            raise ValueError("expected {key: {split: heldout|train, ...}}")
    except ValueError as e:
        raise RegisterError(f"invalid match split register {p}: {e}") from e
    _cache.update(path=str(p), data=data, t=now)
    return data


def clear_cache() -> None:
    _cache.update(path=None, data=None, t=0.0)


def _names_of(key: str, entry: dict) -> list[str]:
    return [key, *entry.get("aliases", [])]


def _matches_prefix(clip_id: str, prefix: str) -> bool:
    return clip_id == prefix or clip_id.startswith(prefix + "_")


def split_for_match(key: str, aliases: list[str] | None = None) -> str:
    """``heldout`` / ``train`` for a match key (or any of its aliases), else ``unassigned``."""
    names = {key, *(aliases or [])}
    for k, entry in load_register().items():
        if names & set(_names_of(k, entry)):
            return entry.get("split") or "unassigned"
    return "unassigned"


def heldout_prefixes() -> tuple[str, ...]:
    out: list[str] = []
    for k, entry in load_register().items():
        if entry.get("split") == HELDOUT:
            out += _names_of(k, entry)
    return tuple(out)


def is_heldout_clip(name: str) -> bool:
    """True if ``name`` (any alias of a clip) belongs to a held-out match."""
    cid = canonical_clip_id(name)
    return any(_matches_prefix(cid, p) for p in heldout_prefixes())


def default_split(key: str) -> str:
    """Deterministic automatic assignment: ~25% held-out via a stable hash of the key."""
    h = int(hashlib.sha1(key.encode()).hexdigest(), 16)  # noqa: S324 - not security
    return HELDOUT if h % 4 == 0 else TRAIN
