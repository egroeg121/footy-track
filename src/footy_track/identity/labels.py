"""Identity label model: pairwise verdicts, checked intervals, provenance tiers.

Two design positions here differ from how box labels work, and both are
deliberate.

**1. `unknown` is a value, not a missing label.**
A player can be genuinely unidentifiable in a frame — fully occluded, motion
blurred, number never visible. If that is recorded as "no label", it is
indistinguishable from "nobody has looked yet", which corrupts both training
(the pair is silently dropped) and evaluation (the frame is scored as if the
model should have known). Recorded as ``UNKNOWN`` it becomes an explicit
ignore-region, and its *rate* is a measurement in its own right: it is the
human ceiling any accuracy target must be stated against.

**2. Provenance tiers attach to frame intervals, not to whole tracklets.**
A box is either checked or not. A 288-frame tracklet is not: a human reviewing
it sees perhaps 12 sampled frames. Marking the whole tracklet ``human_checked``
claims 288 frames of human attention that never happened. So a review records
which intervals were actually inspected, and only associations inside those
intervals are TIER 2. The rest of the same tracklet stays TIER 3. Tracklets are
routinely mixed-tier, and that is the honest representation.

Tiers follow the project-wide scheme (see the labeller's provenance rules):
    TIER 1  human-created
    TIER 2  machine-proposed, human-checked
    TIER 3  machine only
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

TIER_HUMAN = 1
TIER_HUMAN_CHECKED = 2
TIER_MACHINE = 3


class Verdict(str, Enum):
    """Answer to 'are these two tracklets the same player?'"""

    SAME = "same"
    DIFFERENT = "different"
    UNKNOWN = "unknown"  # a real answer: the human looked and could not tell


@dataclass(frozen=True)
class TrackletRef:
    """Identifies one tracklet: a run's track within a clip."""

    clip: str
    track_id: int
    source: str = "bytetrack"
    run_id: str | None = None

    def key(self) -> tuple[str, int, str, str | None]:
        return (self.clip, self.track_id, self.source, self.run_id)


@dataclass(frozen=True)
class CheckedInterval:
    """A closed frame range a human actually inspected. Both ends inclusive."""

    start_frame: int
    end_frame: int

    def __post_init__(self) -> None:
        if self.end_frame < self.start_frame:
            raise ValueError(
                f"end_frame {self.end_frame} precedes start_frame {self.start_frame}"
            )

    def contains(self, frame: int) -> bool:
        return self.start_frame <= frame <= self.end_frame


@dataclass
class TrackletReview:
    """Result of the 'is this one player throughout?' pass.

    ``checked_intervals`` is the honest record of what was inspected. ``split_at``
    lists frames where the human saw the identity change: the tracklet must be
    cut there before it is used as a bag of same-player pairs, otherwise an ID
    switch is harvested as thousands of false positives.
    """

    tracklet: TrackletRef
    checked_intervals: list[CheckedInterval] = field(default_factory=list)
    split_at: list[int] = field(default_factory=list)
    annotator: str = "human"
    # Some tracklets are obvious; some are genuinely undecidable at 52x111 px.
    # Forcing a binary answer on the hard ones manufactures confident labels out
    # of guesses, and those are worse than no label: they enter the eval set and
    # silently move the purity number. Recorded instead as an ignore-region.
    unsure: bool = False
    # A jersey number read by a HUMAN, not OCR. Measured on this footage, only
    # ~9% of tracklets ever contain a frame >=150px tall (~20px digits), so
    # automated per-frame OCR is not viable. But grounding is per-CLUSTER, not
    # per-tracklet: merging links many fragments into one player, and a single
    # legible number names the whole cluster. A number is also a GLOBAL anchor
    # (same team + same number = same player) so it short-circuits pairwise
    # comparison entirely.
    jersey_number: str | None = None

    def tier_at(self, frame: int) -> int:
        """TIER 2 inside a checked interval, TIER 3 outside it."""
        if any(iv.contains(frame) for iv in self.checked_intervals):
            return TIER_HUMAN_CHECKED
        return TIER_MACHINE

    def checked_frame_count(self) -> int:
        return sum(iv.end_frame - iv.start_frame + 1 for iv in self.checked_intervals)

    def is_pure(self) -> bool:
        """True when the human found no identity change in what they inspected.

        An unsure review is NOT pure: it carries no positive claim at all.
        """
        return not self.split_at and not self.unsure


@dataclass(frozen=True)
class PairLabel:
    """One human same/different/unknown verdict on a tracklet pair.

    ``DIFFERENT`` verdicts are recorded, not discarded. They cost nothing extra
    to collect, they are exactly what a retrieval metric needs, and same-team
    negatives are the hard negatives that actually train a metric-learning
    model. Cross-team negatives are near-worthless by comparison (kit colour
    separates teams at ~1% ambiguity), which is why sampling should constrain
    pairs to within a team.
    """

    a: TrackletRef
    b: TrackletRef
    verdict: Verdict
    annotator: str = "human"
    tier: int = TIER_HUMAN

    def ordered(self) -> tuple[TrackletRef, TrackletRef]:
        """Canonical ordering so (a,b) and (b,a) are the same label."""
        return (self.a, self.b) if self.a.key() <= self.b.key() else (self.b, self.a)

    def is_constraint(self) -> bool:
        """UNKNOWN carries no clustering constraint — it is an ignore-region."""
        return self.verdict in (Verdict.SAME, Verdict.DIFFERENT)


def unknown_rate(labels: list[PairLabel]) -> float:
    """Fraction of verdicts that were UNKNOWN.

    This is the annotator ceiling: a model cannot be meaningfully held to an
    accuracy above the rate at which a human can tell at all. Report it
    alongside any identity metric.
    """
    if not labels:
        return 0.0
    return sum(1 for lbl in labels if lbl.verdict is Verdict.UNKNOWN) / len(labels)
