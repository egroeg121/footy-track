"""Registry of NEW evaluation clips (design page ``claude_autolabel_loop.md``, 12.9).

A clip registered here is an evaluation clip from the moment it is registered, before
it has a GT sidecar or an entry in the split file. Everything that must keep evaluation
clips away from training reads it: the auto-labelling loop's guard
(``autolabel.loop.eval_clip_ids``) and the training export
(``export_training_dataset --eval-registry``). Ids are canonical
(``footy_track.clip_id``), so every alias is caught.

The file is small JSON, ``{"version": 1, "clips": {<canonical id>: {...}}}``, written
atomically. ``FOOTY_EVAL_REGISTRY`` names the default location.
"""

from __future__ import annotations

import json
import os
import time
from collections.abc import Iterable
from pathlib import Path

from footy_track.clip_id import canonical_clip_id

ENV_VAR = "FOOTY_EVAL_REGISTRY"


def default_path() -> Path | None:
    raw = os.environ.get(ENV_VAR)
    return Path(raw).expanduser() if raw else None


def read_registry(path: Path | None) -> dict[str, dict]:
    """``{canonical id: entry}``; a missing file or no path is an empty registry."""
    if path is None or not Path(path).is_file():
        return {}
    data = json.loads(Path(path).read_text())
    return {canonical_clip_id(k): v for k, v in (data.get("clips") or {}).items()}


def registry_ids(path: Path | None) -> set[str]:
    return set(read_registry(path))


def register_clips(
    path: Path, clips: Iterable[str], *, mode: str = "sparse", note: str = ""
) -> list[str]:
    """Add clips (any alias) to the registry; returns the ids newly added."""
    path = Path(path)
    have = read_registry(path)
    added = []
    for c in clips:
        cid = canonical_clip_id(c)
        if cid in have:
            continue
        have[cid] = {"mode": mode, "note": note, "registered_at": time.time()}
        added.append(cid)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps({"version": 1, "clips": have}, indent=1))
    tmp.replace(path)
    return added
