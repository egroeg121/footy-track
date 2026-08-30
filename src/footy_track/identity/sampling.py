"""Choosing what to label: evaluation holdout, and which frames to show.

Two sampling problems, both of which have already burned this project once.

**Holdout selection.** ``astonvilla_seg080`` scored recall ~1.000 because it was
in the detector's training set. A holdout must be chosen BEFORE results are seen
and then quarantined. It must also be *spread*: consecutive clips share lighting,
camera side and personnel, so ten adjacent clips are closer to one sample than
to ten. ``select_eval_clips`` takes an evenly spaced stride across the ordered
clip list, deterministically, so the same input always yields the same holdout
and it can be committed and cited.

**Frame selection within a tracklet.** An ID switch happens at one moment. A
uniform sample of 12 frames from 288 usually misses it, which makes the review
feel productive while catching nothing. ``rank_risky_frames`` orders frames by
how likely a switch is there, so the human's limited attention lands where the
evidence is.
"""

from __future__ import annotations

from dataclasses import dataclass


def select_eval_clips(clips: list[str], n: int) -> list[str]:
    """Pick ``n`` clips spread evenly across an ordered clip list.

    Deterministic: no RNG, so the holdout is reproducible and reviewable in a
    diff. Sort the input by capture order (the usual ``*_segNNN`` naming does
    this naturally) so "spread" means spread across the match.

    Raises:
        ValueError: if ``n`` exceeds the number of clips available.
    """
    if n <= 0:
        return []
    if n > len(clips):
        raise ValueError(f"asked for {n} eval clips but only {len(clips)} exist")
    stride = len(clips) / n
    picked = [clips[min(len(clips) - 1, int(i * stride + stride / 2))] for i in range(n)]
    # Guard against duplicates from rounding on short lists.
    out: list[str] = []
    for c in picked:
        if c not in out:
            out.append(c)
    for c in clips:  # top up deterministically if rounding collapsed any
        if len(out) == n:
            break
        if c not in out:
            out.append(c)
    return sorted(out)


@dataclass(frozen=True)
class FrameRisk:
    """Per-frame evidence that a tracklet may switch identity here."""

    frame_index: int
    crowding: int = 0
    confidence: float = 1.0
    association_margin: float = 1.0

    def score(self) -> float:
        """Higher means riskier. Weights are heuristics, not measured.

        The association margin dominates deliberately: a tracker that is nearly
        indifferent between two candidates is the clearest switch signal it can
        give, and unlike crowding it is a direct statement about *this*
        decision. Treat the weights as a starting point to be tuned once
        real switch locations are known.
        """
        crowd_term = min(self.crowding, 10) / 10.0
        conf_term = 1.0 - max(0.0, min(1.0, self.confidence))
        margin_term = 1.0 - max(0.0, min(1.0, self.association_margin))
        return 0.25 * crowd_term + 0.25 * conf_term + 0.5 * margin_term


def rank_risky_frames(risks: list[FrameRisk], k: int) -> list[int]:
    """Return the ``k`` riskiest frame indices, highest risk first.

    Ties break on frame index so the output is stable.
    """
    if k <= 0:
        return []
    ordered = sorted(risks, key=lambda r: (-r.score(), r.frame_index))
    return [r.frame_index for r in ordered[:k]]


def review_budget(n_frames: int, *, sampled: int) -> float:
    """Fraction of a tracklet a human actually inspected.

    Exists to keep reviewers honest about coverage: this is the number that
    makes "I reviewed this tracklet" a quantified claim rather than a vibe, and
    it is what ``TrackletReview.checked_intervals`` should reflect.
    """
    if n_frames <= 0:
        return 0.0
    return min(1.0, sampled / n_frames)
