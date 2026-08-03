"""Shared label / provenance constants for the labeller server modules.

See ``src/footy_track/labeller/README.md`` ("Label hierarchy") and
``docs/labeller_requirements.md`` §1 for the contract these encode.
"""

from __future__ import annotations

# Provenance tags stored in each box's ObjectDetection.model field.
PROV_LABELLER = "labeller"  # manual edit — ground truth, never auto-overwritten
PROV_YOLO = "yolo"
PROV_SAM3 = "sam3"  # kept for backwards compat with existing JSONL sidecars
PROV_VITTRACK = "vittrack"

#: Every provenance tag that may appear in a sidecar's tags list.
PROVENANCE_TAGS = {PROV_LABELLER, PROV_YOLO, PROV_SAM3, PROV_VITTRACK}

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
