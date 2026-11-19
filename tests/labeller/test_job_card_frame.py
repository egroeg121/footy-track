"""Job-mode "Card frame" button and key binding are present in the labeller page (LAB-1442)."""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import pytest

PAGE = Path(__file__).resolve().parents[2] / "src/footy_track/labeller/web/index.html"


def test_card_frame_button_and_key_binding():
    html = PAGE.read_text()
    assert 'id="jobCardBtn"' in html and "Card frame (C)" in html
    assert '$("jobCardBtn").onclick = jobCard' in html
    assert re.search(r'JOB_ID && \(e\.key === "c" \|\| e\.key === "C"\).*jobCard\(\)', html)
    # R must stay Run/Restart; the card key must not shadow it
    assert 'e.key === "r" || e.key === "R"' in html
    # reuses jobGoto and does not mark done / advance
    body = html.split("async function jobCard()")[1].split("\n}\n")[0]
    assert "jobGoto(jobPos)" in body and "jobNext" not in body and "jobDone" not in body


def test_page_script_parses():
    node = shutil.which("node")
    if not node:
        pytest.skip("node not available")
    scripts = re.findall(r"<script>(.*?)</script>", PAGE.read_text(), re.S)
    out = Path(__import__("tempfile").mkdtemp()) / "page.js"
    out.write_text(max(scripts, key=len))
    r = subprocess.run([node, "--check", str(out)], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
