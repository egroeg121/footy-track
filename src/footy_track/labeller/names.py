"""One validator for clip / match names that arrive in request bodies and end up in file paths.

``/ball_check/verdict`` builds ``ball_checks/<clip>.jsonl``, ``/review/correct`` rewrites
``<clip>.jsonl`` and so on, so a name with a path separator or ``..`` could reach outside the
marks directory. Real clip stems and ``.mp4`` names are ``[A-Za-z0-9._-]``; anything else is
refused with 422 before any path is built.
"""

from __future__ import annotations

import re

from fastapi import HTTPException

_CLIP_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,200}$")


def safe_clip_name(clip: object) -> str:
    """Return *clip* unchanged when it is a plain file-name-like string, else raise 422."""
    if not isinstance(clip, str) or not _CLIP_NAME.match(clip) or ".." in clip:
        raise HTTPException(status_code=422, detail="invalid clip name")
    return clip
