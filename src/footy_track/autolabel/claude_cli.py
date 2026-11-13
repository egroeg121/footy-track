"""Ask Claude Code headless (``claude -p``, subscription login) about batches of questions.

Standard library only. One-shot calls, no tools other than Read, no MCP servers. If a
call fails (auth, usage limit, anything), :class:`ClaudeCallError` carries the exact
output and the caller must stop; there is no fallback backend.
"""

from __future__ import annotations

import json
import re
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

PROMPT_VERSION = "verify-v1"

SYSTEM_PROMPT = (
    "You are a careful visual annotator. Follow the user's instructions exactly."
)

PROMPT_HEADER = """\
You are helping to label the match ball in football (soccer) broadcast video.
Below is a list of question images. Use the Read tool to view EACH image, then answer
each question independently.

Each question image shows, on the left, a 320x320 pixel region of a TV frame enlarged
to 960x960. The real match ball is small: usually a white or light round blob roughly
30 to 50 pixels wide in the enlarged view, possibly motion-blurred, in the air, on the
grass or at a player's feet. On the right is a small context picture of the whole frame
with the enlarged region outlined in yellow; use it for scene context only.

Detector candidates are drawn as thin coloured boxes with a large number beside each
box (the number is not part of the scene). A faint 8x8 grid is overlaid, with columns
A-H (left to right) and rows 1-8 (top to bottom) labelled at the edges; a cell is
named column then row, for example "C5".

Common false positives: boots, heads, socks, white boot or advert-board marks, pitch
line paint, corner flags. The ball may be missing from the crop, and none of the boxes
may be on it. Do not guess: if you cannot see a ball, say so.

For each question choose exactly one verdict:
- "candidate": the ball is inside one of the numbered boxes; give its number.
- "cell": the ball is clearly visible but NO box is on it; give the grid cell that
  contains its centre.
- "none": there is no match ball visible in the enlarged region.
- "unsure": you cannot decide.

Reply with ONLY a JSON array, one object per question, in the same order as the list,
no other text and no code fences. Each object must have exactly these keys:
{"id": "<question id>", "verdict": "candidate"|"cell"|"none"|"unsure",
 "candidate": <integer or null>, "cell": <"C5"-style string or null>,
 "confidence": <number from 0 to 1>, "reason": "<one short sentence>"}
Use null for "candidate" unless the verdict is "candidate", and for "cell" unless the
verdict is "cell".

Questions:
"""

_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.M)


class ClaudeCallError(RuntimeError):
    pass


@dataclass
class CallResult:
    model: str
    ids: list[str]
    seconds: float
    attempt: int
    is_error: bool
    cost_usd: float | None
    usage: dict | None
    num_turns: int | None
    session_duration_ms: int | None
    text: str
    answers: list[dict] | None  # None when the output could not be parsed


def build_prompt(items: list[dict]) -> str:
    """``items``: dicts with ``id``, ``image`` (absolute path), ``n_candidates``."""
    lines = [
        f"{i}. id={it['id']} image={it['image']} (boxes numbered 1..{it['n_candidates']})"
        if it["n_candidates"]
        else f"{i}. id={it['id']} image={it['image']} (no boxes drawn)"
        for i, it in enumerate(items, 1)
    ]
    return PROMPT_HEADER + "\n".join(lines) + "\n"


def parse_array(text: str, ids: list[str]) -> list[dict] | None:
    """The JSON array in ``text`` as a list aligned to ``ids``; None if unusable."""
    t = _FENCE.sub("", text.strip())
    lo, hi = t.find("["), t.rfind("]")
    if lo < 0 or hi < lo:
        return None
    try:
        arr = json.loads(t[lo : hi + 1])
    except json.JSONDecodeError:
        return None
    if not isinstance(arr, list) or len(arr) != len(ids):
        return None
    by_id = {a.get("id"): a for a in arr if isinstance(a, dict)}
    if set(by_id) == set(ids):  # tolerate re-ordering
        return [by_id[i] for i in ids]
    return arr


def claude_command(model: str) -> list[str]:
    return [
        "claude", "-p",
        "--model", model,
        "--output-format", "json",
        "--tools", "Read",
        "--allowedTools", "Read",
        "--strict-mcp-config", "--mcp-config", '{"mcpServers":{}}',
        "--disable-slash-commands",
        "--no-session-persistence",
        "--system-prompt", SYSTEM_PROMPT,
    ]  # fmt: skip


def run_once(
    items: list[dict], model: str, cwd: Path, *, attempt: int = 1, timeout: int = 900
) -> CallResult:
    ids = [it["id"] for it in items]
    t0 = time.monotonic()
    proc = subprocess.run(
        claude_command(model),
        input=build_prompt(items),
        capture_output=True,
        text=True,
        cwd=cwd,
        timeout=timeout,
        check=False,
    )
    secs = time.monotonic() - t0
    try:
        out = json.loads(proc.stdout)
    except json.JSONDecodeError:
        out = None
    if proc.returncode != 0 or not isinstance(out, dict) or out.get("is_error"):
        raise ClaudeCallError(
            f"claude -p failed (rc={proc.returncode}, model={model}): "
            f"stdout={proc.stdout[:1500]!r} stderr={proc.stderr[:1500]!r}"
        )
    text = out.get("result") or ""
    return CallResult(
        model=model,
        ids=ids,
        seconds=secs,
        attempt=attempt,
        is_error=bool(out.get("is_error")),
        cost_usd=out.get("total_cost_usd"),
        usage=out.get("usage"),
        num_turns=out.get("num_turns"),
        session_duration_ms=out.get("duration_ms"),
        text=text,
        answers=parse_array(text, ids),
    )


def ask_batch(items: list[dict], model: str, cwd: Path) -> list[CallResult]:
    """One call; a malformed reply is retried once. Returns every attempt made."""
    first = run_once(items, model, cwd)
    if first.answers is not None:
        return [first]
    return [first, run_once(items, model, cwd, attempt=2)]
