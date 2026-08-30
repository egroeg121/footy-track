"""Player identity: labelling primitives, clustering and sampling.

Implements the data model and sampling strategy from
``docs/design/player_reid.md``. Deliberately contains no model code: the
bottleneck for player re-identification in this project is measurement and
label collection, not algorithms, and every piece here is CPU-only and
unit-testable so it can be developed without a GPU.
"""

from footy_track.identity.clusters import ClusterResult, build_clusters
from footy_track.identity.labels import (
    TIER_HUMAN,
    TIER_HUMAN_CHECKED,
    TIER_MACHINE,
    CheckedInterval,
    PairLabel,
    TrackletRef,
    TrackletReview,
    Verdict,
    unknown_rate,
)
from footy_track.identity.sampling import (
    FrameRisk,
    rank_risky_frames,
    select_eval_clips,
)
from footy_track.identity.stable_id import stable_detection_id, stable_id_for_row

__all__ = [
    "TIER_HUMAN",
    "TIER_HUMAN_CHECKED",
    "TIER_MACHINE",
    "CheckedInterval",
    "ClusterResult",
    "FrameRisk",
    "PairLabel",
    "TrackletRef",
    "TrackletReview",
    "Verdict",
    "build_clusters",
    "rank_risky_frames",
    "select_eval_clips",
    "stable_detection_id",
    "stable_id_for_row",
    "unknown_rate",
]
