"""Versioned offline-teacher config (docs/design/full_frame_labelling.md section 4b).

A :class:`TeacherConfig` pins every member of the offline pipeline (detector weights, tracker, SAM3,
verifier model + prompt, thresholds). :meth:`TeacherConfig.content_hash` is a stable digest of the
canonical JSON (key order and ``thresholds`` insertion order do not matter). Human-readable labels
``offline_v1``, ``offline_v2`` ... live in an append-only registry JSON (``teachers.json`` under the
autolabel dir: ``$FOOTY_AUTOLABEL_DIR`` or ``<data root>/autolabel``): :func:`register` returns the existing label for a known
hash, else appends the next. The label is the ``teacher_version`` carried by propose/preverify
outputs and ``frame_label_state`` rows.

Registry schema (``version`` 1)::

    {"version": 1, "teachers": [{"label": "offline_v1", "hash": "<12 hex>", "config": {...}}]}
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path

REGISTRY_VERSION = 1
LABEL_PREFIX = "offline_v"
HASH_LEN = 12


@dataclass(frozen=True)
class TeacherConfig:
    detector_weights_id: str
    tracker: str = "detection_tracker_fwd_bwd"
    sam3: bool = True
    verifier_model: str = ""
    verifier_prompt_version: str = "verify-v2.2"
    thresholds: Mapping[str, float] = field(default_factory=dict)

    def to_json(self) -> dict:
        d = asdict(self)
        d["thresholds"] = {k: self.thresholds[k] for k in sorted(self.thresholds)}
        return d

    @classmethod
    def from_json(cls, doc: Mapping) -> TeacherConfig:
        return cls(**{k: doc[k] for k in cls.__dataclass_fields__ if k in doc})

    def content_hash(self) -> str:
        blob = json.dumps(self.to_json(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(blob.encode()).hexdigest()[:HASH_LEN]


def default_registry_path() -> Path:
    from footy_track import splits  # noqa: PLC0415

    root = os.environ.get("FOOTY_AUTOLABEL_DIR")
    return (Path(root) if root else splits.data_root() / "autolabel") / "teachers.json"


def load_registry(path: Path) -> list[dict]:
    path = Path(path)
    if not path.is_file():
        return []
    doc = json.loads(path.read_text())
    if doc.get("version") != REGISTRY_VERSION:
        raise ValueError(f"unsupported teacher registry version {doc.get('version')!r}")
    return list(doc["teachers"])


def register(config: TeacherConfig, path: Path | None = None) -> str:
    """The ``offline_vN`` label of ``config``; appends a new entry when its hash is unseen."""
    path = Path(path) if path else default_registry_path()
    entries = load_registry(path)
    h = config.content_hash()
    for e in entries:
        if e["hash"] == h:
            return e["label"]
    label = f"{LABEL_PREFIX}{len(entries) + 1}"
    entries.append({"label": label, "hash": h, "config": config.to_json()})
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    doc = {"version": REGISTRY_VERSION, "teachers": entries}
    tmp.write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n")
    tmp.replace(path)
    return label


def load_config(path: Path) -> TeacherConfig:
    return TeacherConfig.from_json(json.loads(Path(path).read_text()))


def audit_pass_rates(status_docs: Iterable[Mapping]) -> dict[str, dict]:
    """``{teacher_version: {n, passed, pass_rate}}`` from series-review status docs.

    Only docs carrying ``teacher_version`` and either ``audit`` {n, failed} (accepted windows) or
    ``audit_failed`` (rejected) count; others are ignored (their version is unknown).
    """
    acc: dict[str, list[int]] = {}
    for d in status_docs:
        tv = d.get("teacher_version")
        if not tv:
            continue
        if isinstance(d.get("audit"), Mapping):
            n, failed = int(d["audit"].get("n", 0)), int(d["audit"].get("failed", 0))
        elif "audit_failed" in d:
            n = failed = len(d["audit_failed"])
        else:
            continue
        a = acc.setdefault(tv, [0, 0])
        a[0] += n
        a[1] += n - failed
    return {
        tv: {"n": n, "passed": p, "pass_rate": (p / n if n else None)}
        for tv, (n, p) in sorted(acc.items())
    }
