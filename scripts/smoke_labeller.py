"""Smoke test for the SAM3 labeller server — drives the core marking flow.

Starts uvicorn, loads a test clip, posts a ball-center mark via /edit,
verifies the mark is retrievable via /timeline, then shuts down.

Usage::

    uv run python scripts/smoke_labeller.py [VIDEO_PATH]

If VIDEO_PATH is omitted, the script uses tests/data/video/test_tiny.mp4.

Exit codes:
    0  — all checks passed
    1  — a check failed or the server did not start in time
"""

from __future__ import annotations

import argparse
import json
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

_ROOT = Path(__file__).parent.parent
_DEFAULT_VIDEO = _ROOT / "tests" / "data" / "video" / "test_tiny.mp4"


def _wait_for_server(url: str, timeout: float = 15.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            urllib.request.urlopen(url, timeout=1)
            return True
        except Exception:
            time.sleep(0.25)
    return False


def _find_free_port() -> int:
    s = socket.socket()
    s.bind(("", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def main() -> int:  # noqa: PLR0915
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", nargs="?", default=str(_DEFAULT_VIDEO))
    args = parser.parse_args()

    video_path = Path(args.video).expanduser().resolve()
    if not video_path.exists():
        print(f"ERROR: video not found: {video_path}", file=sys.stderr)
        return 1

    port = _find_free_port()
    base = f"http://127.0.0.1:{port}"
    print(f"Starting labeller server on port {port}…")

    proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "footy_track.labeller.server:app",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--log-level",
            "warning",
        ],
    )

    try:
        if not _wait_for_server(base):
            print("ERROR: server did not start within 15s", file=sys.stderr)
            return 1
        print("Server started.")

        def _post(path: str, payload: dict) -> dict:
            data = json.dumps(payload).encode()
            req = urllib.request.Request(
                base + path,
                data=data,
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req) as resp:
                return json.loads(resp.read())

        def _get(path: str) -> dict | bytes:
            with urllib.request.urlopen(base + path) as resp:
                ct = resp.headers.get("content-type", "")
                body = resp.read()
                if "json" in ct:
                    return json.loads(body)
                return body

        # --- Check 1: GET / returns HTML ---
        html = _get("/")
        assert b"SAM3 Video Labeller" in html, "Index page missing expected title"
        print("✓ GET / returns HTML with expected title")

        # --- Check 2: POST /session/load ---
        meta = _post("/session/load", {"video_path": str(video_path)})
        assert meta["total_frames"] > 0, f"total_frames={meta['total_frames']}"
        print(
            f"✓ POST /session/load: {meta['total_frames']} frames @ {meta['fps']} fps"
        )

        # --- Check 3: GET /frame/0.jpg ---
        jpeg = _get("/frame/0.jpg")
        assert isinstance(jpeg, bytes) and len(jpeg) > 100, "Frame 0 JPEG too small"
        print(f"✓ GET /frame/0.jpg: {len(jpeg)} bytes")

        # --- Check 4: GET /timeline/0 initially empty ---
        tl = _get("/timeline/0")
        assert tl["boxes"] == [], f"Expected empty timeline, got {tl['boxes']}"
        print("✓ GET /timeline/0 initially empty")

        # --- Check 5: POST /edit marks a ball center ---
        mark = _post(
            "/edit",
            {
                "idx": 0,
                "objects": [
                    {
                        "label": "ball",
                        "x": 0.45,
                        "y": 0.35,
                        "w": 0.05,
                        "h": 0.05,
                        "conf": 1.0,
                    }
                ],
            },
        )
        assert len(mark["boxes"]) == 1, f"Expected 1 box, got {mark['boxes']}"
        assert mark["boxes"][0]["label"] == "ball"
        assert mark["boxes"][0]["source"] == "labeller"
        print("✓ POST /edit saved ball mark (provenance=labeller)")

        # --- Check 6: GET /timeline/0 now returns the mark ---
        tl2 = _get("/timeline/0")
        assert len(tl2["boxes"]) == 1, f"Expected 1 box in timeline, got {tl2['boxes']}"
        assert abs(tl2["boxes"][0]["x"] - 0.45) < 1e-4, (
            f"x coord wrong: {tl2['boxes'][0]['x']}"
        )
        print("✓ GET /timeline/0 returns persisted mark")

        # --- Check 7: Reload wipes the timeline ---
        _post("/session/load", {"video_path": str(video_path)})
        tl3 = _get("/timeline/0")
        assert tl3["boxes"] == [], "Reload should wipe timeline"
        print("✓ POST /session/load clears timeline")

        print("\n✅ All smoke checks passed.")
        return 0

    except AssertionError as exc:
        print(f"\n❌ FAIL: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"\n❌ ERROR: {exc}", file=sys.stderr)
        return 1
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()


if __name__ == "__main__":
    sys.exit(main())
