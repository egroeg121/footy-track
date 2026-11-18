import logging
import types

import pytest
import torch

from footy_track import device as dev


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for k in ("FOOTY_DEVICE", "FOOTY_HALF", "HSA_OVERRIDE_GFX_VERSION"):
        monkeypatch.delenv(k, raising=False)


def _avail(monkeypatch, cuda, mps):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: cuda)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: mps)


@pytest.mark.parametrize(
    "cuda,mps,want", [(True, True, "cuda"), (True, False, "cuda"), (False, True, "mps"), (False, False, "cpu")]
)
def test_auto_order(monkeypatch, cuda, mps, want):
    _avail(monkeypatch, cuda, mps)
    monkeypatch.setattr(torch.version, "hip", None, raising=False)
    assert dev.pick_device() == want


def test_explicit_and_env_override(monkeypatch):
    _avail(monkeypatch, True, True)
    monkeypatch.setenv("FOOTY_DEVICE", "cpu")
    assert dev.pick_device() == "cpu"
    assert dev.pick_device("mps") == "mps"  # explicit beats env


@pytest.mark.parametrize(
    "d,want", [("cuda", True), ("cuda:0", True), ("0", True), ("cpu", False), ("mps", False), (torch.device("cpu"), False)]
)
def test_use_half(d, want):
    assert dev.use_half(d) is want


def test_use_half_env(monkeypatch):
    monkeypatch.setenv("FOOTY_HALF", "0")
    assert dev.use_half("cuda") is False
    monkeypatch.setenv("FOOTY_HALF", "1")
    assert dev.use_half("cpu") is True


def test_rocm_780m_warning(monkeypatch, caplog):
    _avail(monkeypatch, True, False)
    monkeypatch.setattr(torch.version, "hip", "6.2", raising=False)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda i=0: "AMD Radeon 780M")
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda i=0: types.SimpleNamespace(gcnArchName="gfx1103"))
    with caplog.at_level(logging.WARNING, logger="footy_track.device"):
        assert dev.pick_device() == "cuda"
    assert "HSA_OVERRIDE_GFX_VERSION" in caplog.text
    caplog.clear()
    monkeypatch.setenv("HSA_OVERRIDE_GFX_VERSION", "11.0.2")
    with caplog.at_level(logging.WARNING, logger="footy_track.device"):
        dev.pick_device()
    assert not caplog.text
