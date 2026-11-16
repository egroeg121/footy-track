"""Helpers to train a stock Ultralytics YOLO on the 5-channel multiframe TIFFs.

Channel layout of every TIFF (see docs/experiments/multiframe_dataset.md)::

    index   0           1  2  3        4
            gray(t-k)   B  G  R  (t)   gray(t+k)        (both neighbours aligned into frame t)

Ultralytics 8.3.241 (pinned) loads these as (5, H, W) tensors but

* silently drops the pretrained stem when the input has 5 channels (shape mismatch ->
  ``intersect_dicts``), i.e. the net would train from a random stem;
* keeps the stored B,G,R order for 5 channels (it only reverses channels when there are exactly 3),
  so the pretrained R/G/B stem filters belong on input channels 3/2/1;
* ``RandomPerspective`` warps with ``borderValue=(114, 114, 114)``, which OpenCV expands to a 4-vector
  ``(114, 114, 114, 0)``: channel 3 (R) is padded with 0 instead of 114 in the border of every
  mosaic / affine sample (checked: ``warpAffine`` of a constant 5-channel image gives
  ``[114 114 114 0 114]`` in the border). We patch the warps to pad every channel with 114.

Everything here is stdlib + numpy + torch + ultralytics + cv2.
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import numpy as np

# stored order [g(t-k), B, G, R, g(t+k)]: pretrained filters are in R,G,B order
RGB_TO_INPUT_SLOT = {"R": 3, "G": 2, "B": 1}
N_INPUT_CHANNELS = 5
PAD_VALUE = 114


def inflate_stem_weight(w3, in_ch: int = N_INPUT_CHANNELS, slots=(3, 2, 1)):
    """Pretrained stem weight ``(O, 3, k, k)`` in R,G,B order -> ``(O, in_ch, k, k)``.

    ``slots[i]`` is the input channel receiving pretrained channel ``i`` (R, G, B). All other
    channels (the temporal ones) are zero, so the 5-channel net computes exactly the same function
    as the 3-channel net while the neighbour channels are ignored (step 0 of arm B equals arm A).
    """
    import torch

    out_c, c, kh, kw = w3.shape
    if c != 3:
        raise ValueError(f"expected a 3-channel stem, got {c}")
    w = torch.zeros(out_c, in_ch, kh, kw, dtype=w3.dtype, device=w3.device)
    for src, dst in enumerate(slots):
        w[:, dst] = w3[:, src]
    return w


def make_inflated_checkpoint(src_pt: str | Path, dst_pt: str | Path, in_ch: int = N_INPUT_CHANNELS) -> dict:
    """Write an Ultralytics checkpoint whose first conv takes ``in_ch`` channels, initialised from ``src_pt``.

    Returns a small report (number of tensors copied, stem shape). Everything but the stem is copied
    unchanged (including the 80-class head; the trainer drops mismatched head tensors for nc=7 exactly as
    it does for the stock 3-channel arm).
    """
    import torch
    from ultralytics.nn.tasks import DetectionModel

    ckpt = torch.load(str(src_pt), map_location="cpu", weights_only=False)
    old = (ckpt.get("ema") or ckpt["model"]).float()
    cfg = deepcopy(old.yaml)
    cfg["channels"] = in_ch  # DetectionModel reads channels from the yaml first, ``ch`` is only a fallback
    new = DetectionModel(cfg, ch=in_ch, nc=cfg["nc"], verbose=False)

    old_sd, new_sd = old.state_dict(), new.state_dict()
    stem_key = next(k for k, v in old_sd.items() if v.ndim == 4)  # first conv weight
    n_copied = 0
    for k, v in old_sd.items():
        if k == stem_key:
            new_sd[k] = inflate_stem_weight(v, in_ch)
        else:
            if new_sd[k].shape != v.shape:
                raise RuntimeError(f"unexpected shape mismatch at {k}: {v.shape} vs {new_sd[k].shape}")
            new_sd[k] = v.clone()
        n_copied += 1
    new.load_state_dict(new_sd)
    new.names = old.names
    new.args = getattr(old, "args", {})
    out = {"model": new.half(), "ema": None, "train_args": ckpt.get("train_args", {}), "epoch": -1}
    Path(dst_pt).parent.mkdir(parents=True, exist_ok=True)
    tmp = str(dst_pt) + ".tmp"
    torch.save(out, tmp)
    Path(tmp).replace(dst_pt)
    return {"tensors": n_copied, "stem_key": stem_key, "stem_shape": list(new_sd[stem_key].shape)}


# ------------------------------------------------------------------------------------ cv2 border patch
_PATCHED = False


def _chunked_warp(fn):
    """Wrap cv2.warpAffine / warpPerspective so >4-channel images are padded with PAD_VALUE in every channel."""
    def wrapped(src, M, dsize, *args, **kwargs):
        if isinstance(src, np.ndarray) and src.ndim == 3 and src.shape[2] > 4:
            kwargs["borderValue"] = (PAD_VALUE,) * 4
            parts = []
            for i in range(0, src.shape[2], 4):
                p = fn(np.ascontiguousarray(src[..., i : i + 4]), M, dsize, *args, **kwargs)
                parts.append(p[..., None] if p.ndim == 2 else p)
            return np.concatenate(parts, axis=2)
        return fn(src, M, dsize, *args, **kwargs)

    wrapped.__name__ = getattr(fn, "__name__", "warp")
    wrapped._orig = fn  # type: ignore[attr-defined]
    return wrapped


def patch_ultralytics_multichannel() -> None:
    """Make Ultralytics' augmentation pad all channels of a >4-channel image with 114 (idempotent).

    Must run before the dataloader workers are forked (they inherit the patched module).
    Also disables Albumentations (no-op for 5 channels already; we make it explicit so a 3-channel
    arm cannot pick up augmentations the 5-channel arm does not get).
    """
    global _PATCHED
    if _PATCHED:
        return
    import cv2
    from ultralytics.data import augment

    cv2.warpAffine = _chunked_warp(cv2.warpAffine)
    cv2.warpPerspective = _chunked_warp(cv2.warpPerspective)
    if hasattr(augment, "Albumentations"):
        augment.Albumentations.__call__ = lambda self, labels: labels  # type: ignore[method-assign]
    _PATCHED = True
