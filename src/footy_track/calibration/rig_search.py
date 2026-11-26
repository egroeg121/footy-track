"""Label-free match rig search: pure helpers (design ``calibration_match_flow.md`` 2.2, experiment D, decision E2).

The driver is :mod:`footy_track.scripts.ptz_rig_search` (it needs video); this module holds what can be tested without it:

* :func:`rig_hash` / :func:`rig_tag`: the content hash that names the per-segment result dirs
  ``<match>_segNNN@<stem>-<hash8>``. Must stay identical to ``rig_hash`` in the jobs repo (``footy_ptz_temporal.py``);
  ``tests/calibration/test_rig_search.py`` pins the canonical form.
* :func:`assess_search`: the abort-with-report rule (too few anchored segments, finalists that disagree wildly).
* :func:`rig_document`: the fitted rig as a rig JSON (what ``ptz.Rig.to_json`` writes, plus provenance).

Label-free means no input derived from hand labels. ``camera_rigs/generic_gantry_prior_v2.json`` (round numbers) is the
only rig prior; ``camera_rigs/emirates.json`` is an evaluation reference and is never read here, except that
``carry_pose_prior`` may copy its ``info.pose_prior`` (see :func:`rig_document`).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

RIG_DIR = Path(__file__).parent / "camera_rigs"
GENERIC_PRIOR = RIG_DIR / "generic_gantry_prior_v2.json"

#: the tight Levenberg-Marquardt stopping rule on the finalists (E2) and the iteration budget
REL_TOL = 1e-8
N_ITER = 100
TOP_K = 3
SCREEN_STRIDE = 2

MIN_UNITS = 20  # fewer anchored segments than this: the match is probably another camera / ground
MIN_UNIT_FRACTION = 0.25  # ... or fewer than this share of the segments anchored
MAX_FINALIST_SPREAD_M = 8.0  # finalists whose C differs by more than this "disagree wildly" (the ridge scatter is 0.2-2.9 m)


def rig_hash(rig_doc: dict) -> str:
    """First 8 hex of sha256 over the canonical JSON of the rig: sorted keys, no whitespace, utf-8, non-ASCII kept.
    Content, not filename. The jobs repo's ``footy_ptz_temporal.rig_hash`` is the same function."""
    canon = json.dumps(rig_doc, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canon.encode("utf-8")).hexdigest()[:8]


def rig_tag(stem: str, rig_doc: dict) -> str:
    """The ``@``-suffix of result dirs: ``<stem>-<hash8>``."""
    return f"{stem}-{rig_hash(rig_doc)}"


def assess_search(n_units: int, n_segments: int, finalist_C: list[list[float]]) -> list[str]:
    """Reasons to abort the rig search with a report (empty = go). The cost cannot rank rigs inside ~1-3 m (design 2.2),
    so nothing here gates on small cost differences."""
    why = []
    if n_units < MIN_UNITS or n_units < MIN_UNIT_FRACTION * max(n_segments, 1):
        why.append(f"only {n_units} of {n_segments} segments gave a confident anchor frame "
                   f"(< {MIN_UNITS} or < {MIN_UNIT_FRACTION:.0%}): wrong camera / ground for this prior?")
    c = np.asarray(finalist_C, float).reshape(-1, 3)
    if len(c) > 1:
        spread = max(float(np.linalg.norm(a - b)) for a in c for b in c)
        if spread > MAX_FINALIST_SPREAD_M:
            why.append(f"finalist rigs disagree by {spread:.1f} m (> {MAX_FINALIST_SPREAD_M} m): no stable optimum")
    return why


def finalist_spread(finalist_C: list[list[float]]) -> float:
    c = np.asarray(finalist_C, float).reshape(-1, 3)
    return max((float(np.linalg.norm(a - b)) for a in c for b in c), default=0.0)


def rig_document(C, roll: float, k1: float, prior: dict, *, source: str, pose_prior: dict | None = None,
                 sigma_C=(1.0, 3.0, 1.0)) -> dict:
    """The fitted rig as a rig JSON (``ptz.Rig.from_json`` reads it). Deterministic: no timestamps, so the same fit
    gives the same :func:`rig_hash`. ``pose_prior`` (``info.pose_prior``: the tilt-vs-zoom band the recovery pass searches
    on, and ``side_cx_m``, the camera positions of its ``rig_variants``) is optional; without it a weak segment gets neither
    the band nor the gantry variants."""
    info: dict = {"ground": prior.get("ground") or prior.get("info", {}).get("ground", "unknown"), "source": source}
    if pose_prior:
        info["pose_prior"] = pose_prior
    return {
        "C": [float(x) for x in C],
        "roll": float(roll),
        "k1": float(k1),
        "sigma_C": [float(x) for x in sigma_C],
        "pitch": list(prior["pitch"]),
        "info": info,
    }


def load_json(path: str | Path) -> dict:
    return json.loads(Path(path).read_text())
