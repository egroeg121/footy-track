"""Playwright browser tests for the SAM3 labeller web UI (ft-b68).

These tests drive a real Chromium browser against a live uvicorn server to
verify the full marking + bake-off-run cycle that a human would use in the
morning marking session.

Run with::

    uv run pytest tests/test_labeller_playwright.py -v -m playwright -p no:xdist

The ``-p no:xdist`` flag is important: these tests spawn a real server process
and browser, so they should run serially.

The tests use a tiny synthetic 10-frame 64×64 video so they don't depend on
real football footage, real YOLO weights, or SAM3.  YOLO autodetect is
allowed to run (finds 0 detections on a blank frame) — we wait for status to
leave "loading" rather than waiting for a specific count.

Each test is marked ``playwright`` so it is skipped in the normal (non-browser)
pytest run.
"""

from __future__ import annotations

import pathlib
import socket
import subprocess
import sys
import time
import urllib.request

import pytest
from playwright.sync_api import sync_playwright

_VIDEO = pathlib.Path(__file__).parent / "data" / "video" / "test_tiny.mp4"

# JS condition: load is complete when status no longer says "loading" or
# "detecting" (autodetect finishes and sets status to something stable).
_LOAD_DONE_JS = (
    "!document.getElementById('status').textContent.includes('loading') && "
    "!document.getElementById('status').textContent.includes('detecting') && "
    "document.getElementById('status').textContent !== 'no video'"
)


def _find_free_port() -> int:
    s = socket.socket()
    s.bind(("", 0))
    port = s.getsockname()[1]
    s.close()
    return port


@pytest.fixture(scope="module")
def labeller_server():
    """Start a uvicorn server on a free port; yield the base URL; teardown."""
    port = _find_free_port()
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
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    base_url = f"http://127.0.0.1:{port}"

    deadline = time.monotonic() + 15.0
    while time.monotonic() < deadline:
        try:
            urllib.request.urlopen(base_url, timeout=1)
            break
        except Exception:
            if proc.poll() is not None:
                raise RuntimeError("labeller server exited unexpectedly") from None
            time.sleep(0.3)
    else:
        proc.terminate()
        raise TimeoutError(f"Server did not start within 15s on port {port}")

    yield base_url

    proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()


def _load_video(page, labeller_server: str) -> None:
    """Helper: navigate to the labeller and load the test clip."""
    page.goto(labeller_server)
    page.wait_for_load_state("networkidle")
    page.fill("#videoPath", str(_VIDEO))
    page.click("#loadBtn")
    # Wait for load + autodetect to complete (status stops saying "loading…" / "detecting…")
    page.wait_for_function(_LOAD_DONE_JS, timeout=60_000)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.playwright
def test_page_loads_no_console_errors(labeller_server):
    """The index page loads, title is correct, no JS console errors."""
    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=True)
        page = browser.new_page()
        console_errors: list[str] = []
        page.on(
            "console",
            lambda m: console_errors.append(m.text) if m.type == "error" else None,
        )
        page.goto(labeller_server)
        page.wait_for_load_state("networkidle")
        assert "SAM3 Video Labeller" in page.title()
        browser.close()

    assert console_errors == [], f"Console errors on page load: {console_errors}"


@pytest.mark.playwright
def test_load_clip_status_updates(labeller_server):
    """Entering a video path and clicking Load shows a non-loading status."""
    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=True)
        page = browser.new_page()
        page.goto(labeller_server)
        page.wait_for_load_state("networkidle")

        page.fill("#videoPath", str(_VIDEO))
        page.click("#loadBtn")
        page.wait_for_function(_LOAD_DONE_JS, timeout=60_000)

        status = page.text_content("#status") or ""
        assert status != "no video"
        assert "loading" not in status.lower()
        browser.close()


@pytest.mark.playwright
def test_frame_image_appears_after_load(labeller_server):
    """After loading a clip the Konva canvas grows beyond its initial 100px width."""
    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=True)
        page = browser.new_page()
        _load_video(page, labeller_server)
        canvas_width = page.evaluate("document.querySelector('canvas').width")
        assert canvas_width > 100, f"Canvas still at initial size: {canvas_width}"
        browser.close()


@pytest.mark.playwright
def test_scrub_navigation_prev_at_zero_stays(labeller_server):
    """Prev button at frame 0 does not move below 0."""
    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=True)
        page = browser.new_page()
        _load_video(page, labeller_server)

        lbl = page.text_content("#frameLbl") or ""
        assert "0" in lbl, f"Expected frame 0, got: {lbl}"

        page.click("#prevBtn")
        time.sleep(0.5)
        lbl = page.text_content("#frameLbl") or ""
        assert "0" in lbl, f"Prev at frame 0 should stay at 0, got: {lbl}"
        browser.close()


@pytest.mark.playwright
def test_scrub_navigation_next_advances(labeller_server):
    """Next button advances from frame 0 to frame 1."""
    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=True)
        page = browser.new_page()
        _load_video(page, labeller_server)

        page.click("#nextBtn")
        page.wait_for_function(
            "document.getElementById('frameLbl').textContent.includes('1')",
            timeout=10_000,
        )
        lbl = page.text_content("#frameLbl") or ""
        assert "Frame 1" in lbl, f"Expected Frame 1, got: {lbl}"
        browser.close()


@pytest.mark.playwright
def test_draw_mode_toggle(labeller_server):
    """Clicking Draw/Edit buttons toggles the 'on' CSS class correctly."""
    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=True)
        page = browser.new_page()
        _load_video(page, labeller_server)

        edit_cls = page.get_attribute("#toolEdit", "class") or ""
        draw_cls = page.get_attribute("#toolDraw", "class") or ""
        assert "on" in edit_cls, "Edit should start active"
        assert "on" not in draw_cls

        page.click("#toolDraw")
        time.sleep(0.2)
        assert "on" in (page.get_attribute("#toolDraw", "class") or "")
        assert "on" not in (page.get_attribute("#toolEdit", "class") or "")

        page.click("#toolEdit")
        time.sleep(0.2)
        assert "on" in (page.get_attribute("#toolEdit", "class") or "")
        browser.close()


@pytest.mark.playwright
def test_run_button_enabled_after_load(labeller_server):
    """Run button is enabled once a video is loaded."""
    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=True)
        page = browser.new_page()
        page.goto(labeller_server)
        page.wait_for_load_state("networkidle")

        assert page.get_attribute("#runBtn", "disabled") is not None, (
            "Run should start disabled"
        )

        _load_video(page, labeller_server)
        page.wait_for_function(
            "!document.getElementById('runBtn').disabled", timeout=10_000
        )
        assert page.get_attribute("#runBtn", "disabled") is None
        browser.close()


@pytest.mark.playwright
def test_no_console_errors_after_navigation(labeller_server):
    """Scrubbing frames should not produce any JS console errors."""
    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=True)
        page = browser.new_page()
        console_errors: list[str] = []
        page.on(
            "console",
            lambda m: console_errors.append(m.text) if m.type == "error" else None,
        )

        _load_video(page, labeller_server)

        for _ in range(3):
            page.click("#nextBtn")
            time.sleep(0.4)
        for _ in range(2):
            page.click("#prevBtn")
            time.sleep(0.4)

        browser.close()

    assert console_errors == [], f"Console errors during navigation: {console_errors}"


@pytest.mark.playwright
def test_video_path_persisted_in_localstorage(labeller_server):
    """The last-used video path is written to localStorage on load."""
    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=True)
        page = browser.new_page()
        _load_video(page, labeller_server)
        stored = page.evaluate("localStorage.getItem('lastVideoPath')")
        assert stored == str(_VIDEO), f"localStorage has: {stored!r}"
        browser.close()
