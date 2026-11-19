"""Shared compute-device selection: cuda (incl. ROCm) -> mps -> cpu.

ROCm PyTorch reports as ``torch.cuda`` (``torch.version.hip`` is set). On the
merlin AMD 780M iGPU the env var ``HSA_OVERRIDE_GFX_VERSION=11.0.2`` must be
exported *before* torch initialises (11.0.0/11.0.1 abort in fp16). We never set
it silently; ``pick_device`` logs a warning when it looks needed but is unset.

Overrides: ``prefer`` argument, then ``FOOTY_DEVICE``; ``FOOTY_HALF=0/1`` forces
:func:`use_half`.
"""

from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)


def _warn_rocm_override(torch) -> None:
    try:
        if not getattr(torch.version, "hip", None) or os.environ.get("HSA_OVERRIDE_GFX_VERSION"):
            return
        name = str(torch.cuda.get_device_name(0)) + str(
            getattr(torch.cuda.get_device_properties(0), "gcnArchName", "")
        )
        if "780M" in name or "gfx1103" in name:
            logger.warning(
                "ROCm 780M iGPU detected but HSA_OVERRIDE_GFX_VERSION is unset; "
                "export HSA_OVERRIDE_GFX_VERSION=11.0.2 before starting Python (fp16 aborts otherwise)"
            )
    except Exception:  # noqa: BLE001 - best-effort diagnostics only
        pass


def pick_device(prefer: str | None = None) -> str:
    """Return ``prefer``, else ``$FOOTY_DEVICE``, else cuda, mps, or cpu (first available)."""
    explicit = prefer or os.environ.get("FOOTY_DEVICE")
    if explicit:
        return explicit
    try:
        import torch  # noqa: PLC0415
    except ImportError:
        return "cpu"
    if torch.cuda.is_available():
        _warn_rocm_override(torch)
        return "cuda"
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return "mps"
    return "cpu"


def use_half(device: object) -> bool:
    """fp16 inference only on cuda/ROCm; ``FOOTY_HALF=0/1`` overrides."""
    env = os.environ.get("FOOTY_HALF")
    d = str(getattr(device, "type", device)).lower()
    on_gpu = d.startswith("cuda") or d.isdigit()
    if env == "0" and on_gpu and _is_rocm():
        # fp32 convs on the merlin iGPU crashed in conv2d and caused GPU resets (MODE2): never run it
        raise RuntimeError("fp32 (FOOTY_HALF=0) on a ROCm GPU is refused: it crashes/resets the iGPU; use fp16")
    if env in ("0", "1"):
        return env == "1"
    return on_gpu


def _is_rocm() -> bool:
    try:
        import torch  # noqa: PLC0415

        return bool(getattr(torch.version, "hip", None))
    except ImportError:
        return False
