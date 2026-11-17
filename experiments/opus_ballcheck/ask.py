"""Ask Claude (headless ``claude -p``, subscription login; no API key exists here) to do
Ball Check on rendered cards. Images go in as base64 content blocks (stream-json input).

Output: one JSON line per card in /mnt/storage/exp/opus_ballcheck/runs/<run>.jsonl.
Usage: python ask.py --run NAME --prompt v2 --images still,wide,strip --split dev
         [--model claude-opus-5-5] [--effort medium] [--shots id:id] [--workers 3] [--limit N]
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from prompts import system_prompt  # noqa: E402

OUT = Path("/mnt/storage/exp/opus_ballcheck")
_DROP_PREFIX = ("CLAUDE_CODE", "CLAUDECODE", "CLAUDE_AGENT", "OMNIGENT", "AI_AGENT", "CLAUDE_ENV", "CLAUDE_PID",
                "CLAUDE_EFFORT")
_DROP = {"ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_BASE_URL"}
LABEL = {"still": "STILL", "wide": "WIDE", "strip": "FILMSTRIP"}


def clean_env() -> dict:
    return {k: v for k, v in os.environ.items()
            if k not in _DROP and not k.startswith(_DROP_PREFIX)}


def img_block(path: Path) -> dict:
    return {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg",
            "data": base64.b64encode(path.read_bytes()).decode()}}


_VS = json.load(open(OUT / "video_sizes.json"))
META = False


def card_blocks(card: dict, images: list[str], *, answer: str | None = None, n: int | None = None) -> list[dict]:
    d = OUT / "renders" / card["id"]
    two = bool(card.get("other_bbox"))
    head = "Card" + (f" {n}" if n else "") + (
        ": TWO boxes (yellow A = judge this one, magenta B = other)." if two
        else ": one box (yellow A).")
    if META:
        w, h = _VS[card["video"]][:2]
        b = card["bbox"]
        head += f" Box A is {b['w'] * w:.0f} x {b['h'] * h:.0f} px in the {w}x{h} frame."
    blocks: list[dict] = [{"type": "text", "text": head}]
    for im in images:
        blocks.append({"type": "text", "text": f"{LABEL[im]}:"})
        blocks.append(img_block(d / f"{im}.jpg"))
    if answer is not None:
        blocks.append({"type": "text", "text": f"Correct answer: {answer}"})
    return blocks


def shot_answer(card: dict, note: str) -> str:
    return json.dumps({"verdict": card["george"], "reason": note})


def build_message(card: dict, images: list[str], shots: list[tuple[dict, str]]) -> dict:
    content: list[dict] = []
    if shots:
        content.append({"type": "text", "text": "Worked examples, each with the reviewer's verdict:"})
        for i, (sc, note) in enumerate(shots, 1):
            content += card_blocks(sc, images, answer=shot_answer(sc, note), n=i)
        # cache breakpoint after the fixed few-shot prefix (the CLI only caches up to the last block otherwise)
        content.append({"type": "text", "text": "Now the card to judge.", "cache_control": {"type": "ephemeral", "ttl": "1h"}})
    content += card_blocks(card, images)
    content.append({"type": "text", "text": "Reply with the JSON object only."})
    return {"type": "user", "message": {"role": "user", "content": content}}


def parse(text: str) -> dict | None:
    t = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.M)
    lo, hi = t.find("{"), t.rfind("}")
    if lo < 0 or hi < lo:
        return None
    try:
        o = json.loads(t[lo:hi + 1])
    except json.JSONDecodeError:
        return None
    return o if isinstance(o, dict) else None


def call(msg: dict, system: str, model: str, effort: str | None, timeout: int = 600) -> dict:
    cmd = ["claude", "-p", "--model", model, "--input-format", "stream-json",
           "--output-format", "stream-json", "--verbose", "--tools", "",
           "--strict-mcp-config", "--mcp-config", '{"mcpServers":{}}',
           "--disable-slash-commands", "--no-session-persistence", "--system-prompt", system]
    if effort:
        cmd += ["--effort", effort]
    t0 = time.monotonic()
    p = subprocess.run(cmd, input=json.dumps(msg) + "\n", capture_output=True, text=True,
                       env=clean_env(), cwd="/tmp", timeout=timeout, check=False)
    wall = time.monotonic() - t0
    res = None
    for line in p.stdout.splitlines():
        try:
            o = json.loads(line)
        except json.JSONDecodeError:
            continue
        if o.get("type") == "result":
            res = o
    if p.returncode != 0 or res is None or res.get("is_error"):
        raise RuntimeError(f"claude failed rc={p.returncode}: {p.stdout[-600:]!r} {p.stderr[-600:]!r}")
    return {"wall_s": wall, "api_ms": res.get("duration_ms"), "cost_usd": res.get("total_cost_usd"),
            "usage": res.get("usage"), "text": res.get("result") or ""}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--prompt", default="v2")
    ap.add_argument("--images", default="still,wide,strip")
    ap.add_argument("--split", default="dev")
    ap.add_argument("--model", default="claude-opus-5-5")
    ap.add_argument("--effort", default=None)
    ap.add_argument("--shots", default="", help="id:id:... dev card ids for few-shot (notes from shots.json)")
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--ids", default="")
    ap.add_argument("--half", action="store_true", help="deterministic hash half of the cards (cheaper-model comparison)")
    ap.add_argument("--subset", default="", help="iter = dev-iter subset; rest = dev minus iter")
    ap.add_argument("--meta", action="store_true", help="tell the model the box size in px")
    a = ap.parse_args()
    global META
    META = a.meta
    cards = json.load(open(OUT / "cards.json"))
    by_id = {c["id"]: c for c in cards}
    todo = [c for c in cards if c["split"] == a.split]
    if a.subset == "iter":
        todo = [c for c in todo if c.get("iter")]
    elif a.subset == "rest":
        todo = [c for c in todo if not c.get("iter")]
    if a.half:
        import hashlib
        todo = [c for c in todo if int(hashlib.sha1(("half" + c["id"]).encode()).hexdigest(), 16) % 2 == 0]
    if a.ids:
        want = set(a.ids.split(","))
        todo = [c for c in todo if c["id"] in want]
    todo = [c for c in todo if (OUT / "renders" / c["id"] / "strip.jpg").exists()]
    images = a.images.split(",")
    notes = json.load(open(Path(__file__).parent / "shots.json")) if a.shots else {}
    shots = [(by_id[i], notes.get(i, "")) for i in a.shots.split(":") if i]
    system = system_prompt(a.prompt)
    (OUT / "runs").mkdir(exist_ok=True)
    path = OUT / "runs" / f"{a.run}.jsonl"
    done = set()
    if path.exists():
        done = {json.loads(l)["id"] for l in open(path) if l.strip()}
    todo = [c for c in todo if c["id"] not in done]
    if a.limit:
        todo = todo[:a.limit]
    print(f"{a.run}: {len(todo)} cards to do ({len(done)} done)", flush=True)

    def work(c):
        msg = build_message(c, images, shots)
        for attempt in (1, 2):
            try:
                r = call(msg, system, a.model, a.effort)
            except Exception as e:  # noqa: BLE001
                err = str(e)
                if attempt == 2 or "usage limit" in err.lower() or "not logged in" in err.lower():
                    return {"id": c["id"], "error": err, "run": a.run}
                continue
            r["parsed"] = parse(r["text"])
            if r["parsed"] is None and attempt == 1:
                continue
            r.update(id=c["id"], run=a.run, model=a.model, attempt=attempt)
            return r
        return {"id": c["id"], "error": "unparseable", "run": a.run}

    n_err = 0
    with ThreadPoolExecutor(a.workers) as ex, open(path, "a") as fh:
        for r in ex.map(work, todo):
            fh.write(json.dumps(r) + "\n")
            fh.flush()
            if "error" in r:
                n_err += 1
                print("ERR", r["id"], r["error"][:300], flush=True)
                if n_err >= 5:
                    print("too many errors, stopping"); os._exit(2)
            else:
                p = r["parsed"]
                print(r["id"], p.get("verdict"), p.get("confidence"), f"${r['cost_usd']:.4f}",
                      f"{r['wall_s']:.0f}s", flush=True)


if __name__ == "__main__":
    main()
