"""Audit statistics for the ``auto_ok`` rules (docs/design/full_frame_labelling.md 4b/6b).

Inputs are *blind audit outcomes*: for each audited ``auto_ok`` frame, the score the auto-accept
rule used (e.g. verifier confidence) and whether the blind human relabel agreed (``correct``).

Methods (deliberately simple):

* :func:`wilson_lower` - one-sided Wilson score lower bound on precision (default 95%).
* :func:`pick_threshold` - Learn-then-Test style fixed-sequence selection (docs/design/
  ball_sampling.md 3.3). Candidate thresholds are the distinct observed scores, tried from HIGH
  to LOW. For threshold ``t`` the audited frames with ``score >= t`` give ``k`` correct of
  ``n``; the p-value for H0 "precision < target" is the binomial tail ``P(Bin(n, target) >= k)``.
  We keep lowering ``t`` while H0 is rejected at level ``delta`` and stop at the first failure
  (fixed-sequence testing controls the family-wise error at ``delta`` without a correction).
  The answer is the smallest ``t`` reached; ``None`` when even the highest cannot be certified
  (not enough audits yet). Audits must be uniform within the auto-accepted set (exchangeable).
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence

Z_ONE_SIDED_95 = 1.6449


def precision(correct: int, n: int) -> float | None:
    return None if n == 0 else correct / n


def wilson_lower(correct: int, n: int, z: float = Z_ONE_SIDED_95) -> float:
    """Wilson score lower bound for ``correct`` successes of ``n`` (0.0 when n == 0)."""
    if n <= 0:
        return 0.0
    p = correct / n
    z2 = z * z
    centre = p + z2 / (2 * n)
    margin = z * math.sqrt(p * (1 - p) / n + z2 / (4 * n * n))
    return max(0.0, (centre - margin) / (1 + z2 / n))


def binom_tail_ge(k: int, n: int, p: float) -> float:
    """P(X >= k) for X ~ Binomial(n, p)."""
    return min(
        1.0, sum(math.comb(n, i) * p**i * (1 - p) ** (n - i) for i in range(k, n + 1))
    )


def pick_threshold(
    outcomes: Iterable[tuple[float, bool]],
    target: float = 0.95,
    delta: float = 0.1,
    min_audits: int = 1,
) -> float | None:
    """Smallest score threshold whose audited precision is certified >= ``target`` (see module)."""
    data: Sequence[tuple[float, bool]] = list(outcomes)
    chosen: float | None = None
    for t in sorted({s for s, _ in data}, reverse=True):
        sel = [ok for s, ok in data if s >= t]
        n, k = len(sel), sum(sel)
        if n < min_audits or binom_tail_ge(k, n, target) > delta:
            break
        chosen = t
    return chosen
