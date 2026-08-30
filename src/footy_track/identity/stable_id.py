"""Stable, content-addressed detection IDs.

``importers.py`` synthesises ``detection_id`` as the row ordinal ``i``. That is
positional: filter a JSONL, drop a low-confidence row, or re-run inference with
a different threshold, and every subsequent detection silently changes identity.
Anything keyed to a detection — an identity label, a human correction, a review
flag — then rebinds to the WRONG box, with no error and no way to notice.

That is fatal for identity labelling specifically, because identity labels are
expensive to collect and their corruption is invisible: a label still points at
*a* detection, just not the one the human looked at.

The fix is to derive the id from the detection's own content, so it is stable
under reordering, filtering and regeneration:

    detection_id = hash(frame_index, label, rounded bbox)

Rounding matters. Float formatting differs between producers (Python repr,
numpy, JSON round-trips), so the geometry is quantised to a fixed grid before
hashing. ``_QUANT`` = 1e-6 of normalised image width — far finer than any real
box, comfortably coarser than float noise.

Two detections of the same class at the same place in the same frame collide by
construction. That is intentional: they are indistinguishable to a labeller too,
and de-duplicating them is correct.
"""

from __future__ import annotations

import hashlib
from typing import Any

# Quantisation grid for normalised coordinates (1e-6 ~= 0.002 px at 1920 wide).
_QUANT = 1_000_000


def _q(v: float) -> int:
    """Quantise a normalised coordinate to a stable integer grid."""
    return int(round(float(v) * _QUANT))


def stable_detection_id(
    frame_index: int,
    label: str,
    bbox: tuple[float, float, float, float] | dict[str, float],
    *,
    digits: int = 16,
) -> str:
    """Return a stable hex id for one detection.

    Args:
        frame_index: zero-based frame number within the clip.
        label: class name, e.g. ``"player"``. Part of the key so a box
            re-classified between runs does not silently inherit old labels.
        bbox: normalised ``(x, y, w, h)`` top-left, or a dict with those keys.
        digits: hex characters to keep. 16 gives a 64-bit space; with ~3e6
            detections the collision probability is ~1e-7.

    Returns:
        Lower-case hex string, ``digits`` characters long.
    """
    if isinstance(bbox, dict):
        x, y, w, h = bbox["x"], bbox["y"], bbox["w"], bbox["h"]
    else:
        x, y, w, h = bbox
    payload = f"{int(frame_index)}|{label}|{_q(x)}|{_q(y)}|{_q(w)}|{_q(h)}"
    return hashlib.blake2b(payload.encode("utf-8"), digest_size=32).hexdigest()[:digits]


def stable_detection_int(
    frame_index: int,
    label: str,
    bbox: tuple[float, float, float, float] | dict[str, float],
) -> int:
    """Stable id as a non-negative 63-bit int, for the ``BIGINT`` store column.

    ``feature_store`` declares ``detection_id BIGINT`` and includes it in the
    detection primary key, so the content-addressed id has to be an integer to
    be a drop-in for the old row ordinal. 63 bits keeps it positive and inside
    a signed BIGINT.

    MIGRATION: this changes the value of an existing key column. A store
    populated with the old ordinal ids will not recognise re-imported rows as
    the same detections — they will insert alongside rather than upsert over.
    Re-import affected games into a fresh table rather than mixing schemes.
    """
    return int(stable_detection_id(frame_index, label, bbox, digits=16), 16) >> 1


def stable_id_for_row(row: dict[str, Any], *, digits: int = 16) -> str:
    """Convenience wrapper for a labeller/machine-label JSONL row.

    Accepts the on-disk detection format::

        {"frame_index": int, "bbox": {"x","y","w","h"}, "tags": [label, source], ...}

    The label is the first tag that is not a known provenance/source tag, which
    matches how the rest of the codebase recovers a class name from ``tags``.
    """
    from footy_track.labeller.constants import PROVENANCE_TAGS  # noqa: PLC0415

    tags = row.get("tags") or []
    label = next((t for t in tags if t not in PROVENANCE_TAGS), "unknown")
    bbox = row.get("bbox")
    if not isinstance(bbox, dict):
        raise ValueError(f"row has no usable bbox: {row!r}")
    return stable_detection_id(
        int(row["frame_index"]), label, bbox, digits=digits
    )
