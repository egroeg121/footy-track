"""Tests for VitTrackSOT's process-wide session cache and model-path memoization
(``src/footy_track/ball_trackers/sot_vittrack.py``).

Per the module docstring: "Session creation is expensive on macOS ... ORT
sessions are thread-safe for run(), so one shared session per model path
serves every tracker instance in the process." These tests verify the cache
sharing and the per-instance independence of tracking state, without ever
instantiating a real ONNX InferenceSession or running inference (both are
slow/fragile and can SIGKILL in this sandboxed shell).

Approach: monkeypatch ``_make_session`` to return a cheap sentinel object, so
``_cached_session`` exercises its real dict-reuse + lock logic against a fast
stand-in. ``_download_model`` is exercised directly with a monkeypatched
``huggingface_hub.hf_hub_download`` to verify the memoization contract (call
count <= 1 after the first successful call).
"""

from __future__ import annotations

import pathlib
import sys
import types

import pytest

from footy_track.ball_trackers import sot_vittrack


@pytest.fixture(autouse=True)
def _reset_module_caches(monkeypatch):
    """Isolate each test from the process-wide caches (they're module globals)."""
    monkeypatch.setattr(sot_vittrack, "_session_cache", {})
    monkeypatch.setattr(sot_vittrack, "_model_path_cache", None)
    yield


class _SentinelSession:
    """Cheap stand-in for ort.InferenceSession — identity is what we assert on."""


def test_cached_session_reuses_same_object_for_same_path(monkeypatch):
    calls = []

    def fake_make_session(model_path):
        calls.append(model_path)
        return _SentinelSession()

    monkeypatch.setattr(sot_vittrack, "_make_session", fake_make_session)

    path = pathlib.Path("/fake/model.onnx")
    s1 = sot_vittrack._cached_session(path)
    s2 = sot_vittrack._cached_session(path)

    assert s1 is s2
    assert len(calls) == 1  # _make_session only invoked once


def test_cached_session_distinguishes_by_path(monkeypatch):
    monkeypatch.setattr(sot_vittrack, "_make_session", lambda p: _SentinelSession())  # noqa: ARG005

    s1 = sot_vittrack._cached_session(pathlib.Path("/fake/a.onnx"))
    s2 = sot_vittrack._cached_session(pathlib.Path("/fake/b.onnx"))

    assert s1 is not s2


def test_two_vittracksot_instances_share_identical_session(monkeypatch):
    """a._session is b._session for two instances constructed with the same model_path."""
    monkeypatch.setattr(sot_vittrack, "_make_session", lambda p: _SentinelSession())  # noqa: ARG005

    path = pathlib.Path("/fake/model.onnx")
    a = sot_vittrack.VitTrackSOT(model_path=path)
    b = sot_vittrack.VitTrackSOT(model_path=path)

    assert a._session is b._session


def test_vittracksot_instances_have_independent_per_instance_state(monkeypatch):
    """_template_blob / _last_bbox_px must not be shared across instances."""
    monkeypatch.setattr(sot_vittrack, "_make_session", lambda p: _SentinelSession())  # noqa: ARG005

    path = pathlib.Path("/fake/model.onnx")
    a = sot_vittrack.VitTrackSOT(model_path=path)
    b = sot_vittrack.VitTrackSOT(model_path=path)

    assert a._session is b._session  # shared session
    # but per-instance tracking state starts independent and stays independent
    assert a._template_blob is None
    assert b._template_blob is None
    a._template_blob = "fake-blob-a"
    a._last_bbox_px = (1.0, 2.0, 3.0, 4.0)

    assert b._template_blob is None
    assert b._last_bbox_px is None
    assert a._template_blob == "fake-blob-a"
    assert a._last_bbox_px == (1.0, 2.0, 3.0, 4.0)


def test_download_model_memoized_across_calls(monkeypatch):
    """_download_model must not re-hit huggingface_hub after the first success."""
    call_count = {"n": 0}

    def fake_hf_hub_download(repo_id, filename):  # noqa: ARG001
        call_count["n"] += 1
        return "/fake/downloaded/model.onnx"

    fake_module = types.SimpleNamespace(hf_hub_download=fake_hf_hub_download)
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_module)

    first = sot_vittrack._download_model()
    second = sot_vittrack._download_model()

    assert first == second
    assert first == pathlib.Path("/fake/downloaded/model.onnx")
    assert call_count["n"] <= 1


def test_download_model_result_is_pathlib_path(monkeypatch):
    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        types.SimpleNamespace(
            hf_hub_download=lambda repo_id, filename: "/fake/model.onnx"  # noqa: ARG005
        ),
    )
    result = sot_vittrack._download_model()
    assert isinstance(result, pathlib.Path)


def test_vittracksot_uses_download_model_when_no_path_given(monkeypatch):
    """When model_path=None, VitTrackSOT should call _download_model (memoized) then _cached_session."""
    monkeypatch.setattr(sot_vittrack, "_make_session", lambda p: _SentinelSession())  # noqa: ARG005

    download_calls = []

    def fake_download_model():
        download_calls.append(1)
        return pathlib.Path("/fake/auto-downloaded.onnx")

    monkeypatch.setattr(sot_vittrack, "_download_model", fake_download_model)

    tracker = sot_vittrack.VitTrackSOT()

    assert len(download_calls) == 1
    assert isinstance(tracker._session, _SentinelSession)
