"""Shared label / provenance constants for the labeller server modules.

See ``src/footy_track/labeller/README.md`` §§1–2 (label hierarchy,
sidecar format) for the contract these encode.
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

#: Marker tags (written after the provenance) of a box accepted as-is from a detector suggestion (eval panel,
#: ``ball_eval/panel_suggest.py``). Audit only: the box is ordinary tier-1 ``labeller`` GT.
SUGGEST_MARKER_TAG = "accepted_suggestion"
SUGGEST_DET_TAG_PREFIX = "suggest_det:"  # + detector sha8


def is_suggest_marker(tag: object) -> bool:
    """Marker tags the page may attach to a box it saves (``/edit`` drops any other client-supplied marker)."""
    return isinstance(tag, str) and (tag == SUGGEST_MARKER_TAG or (tag.startswith(SUGGEST_DET_TAG_PREFIX) and len(tag) <= 40))


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
