"""Shared label / provenance constants for the labeller server modules.

See ``src/footy_track/labeller/README.md`` §§1–2 (label hierarchy,
sidecar format) for the contract these encode.
"""

from __future__ import annotations

import hashlib

# Provenance tags stored in each box's ObjectDetection.model field.
PROV_LABELLER = "labeller"  # manual edit — ground truth, never auto-overwritten
PROV_YOLO = "yolo"
PROV_SAM3 = "sam3"  # kept for backwards compat with existing JSONL sidecars
PROV_VITTRACK = "vittrack"
PROV_CLAUDE = "claude"  # auto-labelling loop; tier 2 only with the human_checked marker
PROV_CLAUDE_AUTO = "claude_auto"  # auto-labelling loop, unreviewed (tier 3)

#: Every provenance tag that may appear in a sidecar's tags list.
PROVENANCE_TAGS = {
    PROV_LABELLER,
    PROV_YOLO,
    PROV_SAM3,
    PROV_VITTRACK,
    PROV_CLAUDE,
    PROV_CLAUDE_AUTO,
}

# Ball-class labels that appear in the JSONL sidecar.
BALL_LABELS = {"ball", "in_play_ball", "out_of_play_ball"}
PLAYER_LABELS = {"player", "player_sub", "referee", "coach", "person"}

#: Labels recognised when restoring a sidecar box line.
KNOWN_BOX_LABELS = BALL_LABELS | PLAYER_LABELS

# Skip-marker tags (frame recorded, no box).
NO_BALL_TAG = "no_ball"
NOT_BROADCAST_TAG = "not_broadcast"

#: All classes that can appear in the review picker.
REVIEW_LABELS = [
    "player",
    "in_play_ball",
    "out_of_play_ball",
    "referee",
    "coach",
    "player_sub",
    "ball",
]
REVIEW_LABELS_SET = set(REVIEW_LABELS)


def card_box_index(card_id: str) -> int:
    """Stable ``box_index`` of a Ball Check list card that carries an ``id``.

    A card verdict is keyed ``("list", clip, frame_index, box_index)``. Cards without an
    ``id`` use their list position (the original behaviour); cards with one use this
    hash of the id, so the list can be reordered, filtered or regenerated without a
    verdict landing on a different card. 31 bits, far above any list position.
    """
    return int(hashlib.sha1(card_id.encode()).hexdigest()[:8], 16) & 0x7FFFFFFF
