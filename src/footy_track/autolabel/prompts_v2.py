"""Versioned prompts for the Claude verify step, method v2 (one question per candidate).

Every version asks the same question, the one /ball_check asks a person: is the box on
the match ball? Keep old versions here unchanged so a logged ``prompt_version`` always
maps to the exact text that was used.
"""

from __future__ import annotations

_COMMON_HEAD = """\
You are helping to label the match ball in football (soccer) broadcast video.
Below is a list of question images. Use the Read tool to view EACH image, then answer
each question independently. Every question is the same: is the yellow box on the
match ball?

Each image has two panels. Left: a 128 x 128 pixel patch of a TV frame enlarged 4x
(512 x 512), with ONE thin yellow box marking the candidate, near the centre. Right: a
wider 384 x 384 pixel region of the same frame at normal size, for scene context only;
its yellow square shows where the left patch comes from and the small yellow box is the
candidate.

The match ball is small: a white or light round blob roughly 8 to 15 pixels wide in the
frame (30 to 60 pixels in the enlarged patch), sometimes motion-blurred into a short
streak, in the air, on the grass or at a player's feet.
"""

_V2_BODY = """\

Common false positives: boots, heads, socks, shirt numbers, white boot or advert-board
marks, pitch line paint, corner flags, a ball-like blob that is not the match ball.

Choose exactly one verdict:
- "ball": the yellow box is on the match ball.
- "not_ball": the box is on something else, or on empty pitch, or the ball is visible
  but somewhere other than in the box.
- "unsure": you cannot decide.
Do not guess: if you cannot see a ball in the box, do not say "ball".

"""

_V2_1_BODY = """\

The ball is not always white: it can be white, pale grey or lilac, or a bright yellow or
yellow-green high-visibility ball, often with a darker underside and a soft shadow.

The detector box is not always exactly centred on the ball. Say "ball" when the ball is
inside the box or touching it, or no further from the box centre than about one box
height (about 60 pixels in the enlarged patch). If the ball is clearly further away, or
a different object fills the box, say "not_ball".

Things that are NOT the match ball: boots and the toe of a boot, heads and hair, white
socks and shorts, shirt numbers and sponsor logos, white advert-board marks, pitch line
paint and its intersections, corner flags, floodlight or crowd specks.

Choose exactly one verdict:
- "ball": the match ball is in or touching the box (or within one box height of it).
- "not_ball": the box is on anything else or on empty pitch, and no ball is that close.
- "unsure": a blurred, partly hidden or tiny object that might be the ball, and you
  cannot decide after looking at both panels. Use it rarely.

"""

_V2_2_BODY = """\

The ball is not always white: it can be white, pale grey or lilac, or a bright yellow or
yellow-green high-visibility ball, often with a darker underside and a soft shadow.

The detector box is not always exactly centred on the ball. Say "ball" when the ball is
inside the box or touching it, or no further from the box centre than about one box
height (about 60 pixels in the enlarged patch). If the ball is clearly further away, or
a different object fills the box, say "not_ball".

Be strict about look-alikes. A small pale blob is the match ball only if it is a
separate round object with its own shape: not part of a boot, sock, leg, shirt or
advert board, not a line or logo fragment, not a speck in the crowd or stands. A ball
in play is usually near a player's feet, on the grass or in the air in front of the
pitch, and its position is consistent with where the players are looking and kicking.
If the only evidence is a faint light smear with no clear round shape, say "unsure".

Choose exactly one verdict:
- "ball": the match ball is in or touching the box (or within one box height of it).
- "not_ball": the box is on anything else or on empty pitch, and no ball is that close.
- "unsure": a blurred, partly hidden or tiny object that might be the ball, and you
  cannot decide after looking at both panels.

"""

_TAIL = """\
Reply with ONLY a JSON array, one object per question, in the same order as the list,
no other text and no code fences. Each object must have exactly these keys:
{"id": "<question id>", "verdict": "ball"|"not_ball"|"unsure",
 "confidence": <number from 0 to 1>, "reason": "<one short sentence>"}

Questions:
"""

PROMPTS_V2: dict[str, str] = {
    "verify-v2": _COMMON_HEAD + _V2_BODY + _TAIL,
    "verify-v2.1": _COMMON_HEAD + _V2_1_BODY + _TAIL,
    "verify-v2.2": _COMMON_HEAD + _V2_2_BODY + _TAIL,
}
DEFAULT_V2 = "verify-v2"
VERDICTS_V2 = ("ball", "not_ball", "unsure")


def build_prompt_v2(items: list[dict], version: str = DEFAULT_V2) -> str:
    """``items``: dicts with ``id`` and ``image`` (absolute path)."""
    lines = [f"{i}. id={it['id']} image={it['image']}" for i, it in enumerate(items, 1)]
    return PROMPTS_V2[version] + "\n".join(lines) + "\n"
