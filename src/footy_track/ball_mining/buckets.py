"""Which pool nodes qualify for which bucket (design ``ball_sampling.md`` §3.2).

Every threshold lives in :class:`BucketConfig`; nothing here is a module constant that code reads behind your back.

=======  ===========================================================  ===============================================
bucket   qualifies                                                    rank inside the bucket
=======  ===========================================================  ===============================================
U        not a gap / static node, and any of: ``p_real`` in           ``|p_real - 0.5|`` ascending (most uncertain
         ``[u_p_lo, u_p_hi]``; sources disagree (score spread >=      first), then one card per k-means cluster
         ``u_spread``); solver posterior in ``[u_post_lo, u_post_hi]``;
         SAM2 disagrees while ``p_real >= u_p_lo``
M        ``gap_frame`` and not ``plausible == False`` and             ``p_real`` descending (``p_real`` of a gap
         ``p_real >= m_min_plausibility``                             node is the gap's plausibility)
N        ``static_flag`` or ``plausible == False``, with a source     highest source score first, at most
         score >= ``n_min_score`` (a confident box the scene says is  ``n_per_location`` per (clip, location cell)
         wrong)
A        ``p_real >= a_floor`` (the auto-accept candidate region);    uniform within ``(match, band)``; bands are the
         ``a_band`` is its band in ``a_band_edges``                   intervals between ``a_band_edges``
R        every train node                                             uniform
G        not pool-derived: see :mod:`footy_track.ball_mining.gold`    uniform
=======  ===========================================================  ===============================================

A node can qualify for several buckets; the miner takes them in order U, M, N, A, G, then R so a card is never double
counted, and a 2 s window never holds two cards.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

BUCKETS = ("U", "M", "N", "R", "A", "G")
DEFAULT_QUOTAS: dict[str, float] = {
    "U": 0.35,
    "M": 0.20,
    "N": 0.15,
    "R": 0.15,
    "A": 0.10,
    "G": 0.05,
}


@dataclass(frozen=True)
class BucketConfig:
    # U
    u_p_lo: float = 0.30
    u_p_hi: float = 0.90
    u_spread: float = 0.35
    u_post_lo: float = 0.30
    u_post_hi: float = 0.70
    # M
    m_min_plausibility: float = 0.20
    # N
    n_min_score: float = 0.50
    n_cell: float = 0.05  # location cell (normalised) for the per-location cap
    n_per_location: int = 2
    # A. ``auto_threshold`` is the interim auto-accept rule until LTT finds a valid threshold: 0.95 on the
    # current Claude-verify scale (design §3.3). The audit region reaches below it so LTT can find a lower t.
    auto_threshold: float = 0.95
    a_floor: float = 0.50
    a_band_edges: tuple[float, ...] = (0.50, 0.70, 0.80, 0.90, 0.95, 0.98, 1.0001)
    a_min_per_band: int = 2
    # G
    g_shift_px: tuple[float, float] = (15.0, 30.0)
    # geometry / time
    frame_wh: tuple[int, int] = (1920, 1080)
    window_s: float = 2.0  # at most one card per clip per window
    eval_buffer_s: float = 10.0  # exclusion around in-match eval clips
    # diversity (U, M)
    cluster_overshoot: float = (
        1.5  # k = overshoot * target, so dedupe losses do not leave the bucket short
    )
    cluster_time_weight: float = 1.0
    cluster_candidates: int = 4000  # top-ranked nodes per match that enter k-means
    extra: dict = field(default_factory=dict, compare=False)

    def band_label(self, i: int) -> str:
        lo, hi = self.a_band_edges[i], min(self.a_band_edges[i + 1], 1.0)
        return f"{lo:.2f}-{hi:.2f}"


def _true(s: pd.Series) -> np.ndarray:
    """Null-safe ``is True``."""
    return s.map(
        lambda v: v is True or (isinstance(v, (bool, np.bool_)) and bool(v))
    ).to_numpy(dtype=bool)


def _false(s: pd.Series) -> np.ndarray:
    """Null-safe ``is False`` (unknown is not False)."""
    return s.map(
        lambda v: v is False or (isinstance(v, (bool, np.bool_)) and not bool(v))
    ).to_numpy(dtype=bool)


def bucket_flags(pool: pd.DataFrame, cfg: BucketConfig | None = None) -> pd.DataFrame:
    """Eligibility and rank columns, indexed like ``pool``: bool ``U M N A``, ``a_band`` (-1 outside A),
    ``u_rank`` (smaller = more uncertain), ``m_rank`` and ``n_rank`` (larger = better)."""
    cfg = cfg or BucketConfig()
    p = pool["p_real"].to_numpy(dtype=float)
    gap = _true(pool["gap_frame"]) if "gap_frame" in pool else np.zeros(len(pool), bool)
    static = (
        _true(pool["static_flag"])
        if "static_flag" in pool
        else np.zeros(len(pool), bool)
    )
    implausible = (
        _false(pool["plausible"]) if "plausible" in pool else np.zeros(len(pool), bool)
    )
    sam_no = (
        _false(pool["sam2_agree"])
        if "sam2_agree" in pool
        else np.zeros(len(pool), bool)
    )
    post = (
        pool["solver_post"].to_numpy(dtype=float)
        if "solver_post" in pool
        else np.full(len(pool), np.nan)
    )
    smax = (
        pool["source_scores"]
        .map(lambda v: max(v) if len(v) else 0.0)
        .to_numpy(dtype=float)
    )
    smin = (
        pool["source_scores"]
        .map(lambda v: min(v) if len(v) else 0.0)
        .to_numpy(dtype=float)
    )
    spread = np.where(pool["n_sources"].to_numpy() >= 2, smax - smin, 0.0)

    in_band = (p >= cfg.u_p_lo) & (p <= cfg.u_p_hi)
    in_post = (post >= cfg.u_post_lo) & (post <= cfg.u_post_hi)
    u = (
        ~gap
        & ~static
        & (in_band | (spread >= cfg.u_spread) | in_post | (sam_no & (p >= cfg.u_p_lo)))
    )
    m = gap & ~implausible & (p >= cfg.m_min_plausibility)
    n = (static | implausible) & (smax >= cfg.n_min_score)
    a = p >= cfg.a_floor
    band = np.where(
        a,
        np.clip(np.digitize(p, cfg.a_band_edges) - 1, 0, len(cfg.a_band_edges) - 2),
        -1,
    )
    return pd.DataFrame(
        {
            "U": u,
            "M": m,
            "N": n,
            "A": a,
            "a_band": band,
            "u_rank": np.abs(p - 0.5),
            "m_rank": p,
            "n_rank": smax,
        },
        index=pool.index,
    )
