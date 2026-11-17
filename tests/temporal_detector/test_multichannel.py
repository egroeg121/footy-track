"""Stem inflation + border-patch tests. Need torch + ultralytics (desktop venv); skipped otherwise."""

import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("ultralytics")
cv2 = pytest.importorskip("cv2")

sys.path.insert(
    0, str(Path(__file__).resolve().parents[2] / "experiments" / "temporal_detector")
)
import multichannel as mc  # noqa: E402


def test_inflate_stem_weight_places_rgb_on_bgr_slots():
    w3 = torch.arange(2 * 3 * 3 * 3, dtype=torch.float32).reshape(
        2, 3, 3, 3
    )  # channels R, G, B
    w5 = mc.inflate_stem_weight(w3)
    assert w5.shape == (2, 5, 3, 3)
    assert torch.equal(w5[:, 3], w3[:, 0])  # R -> input 3
    assert torch.equal(w5[:, 2], w3[:, 1])  # G -> input 2
    assert torch.equal(w5[:, 1], w3[:, 2])  # B -> input 1
    assert torch.count_nonzero(w5[:, 0]) == 0 and torch.count_nonzero(w5[:, 4]) == 0


def test_inflated_net_equals_3ch_net_at_step_0(tmp_path):
    from ultralytics.nn.tasks import DetectionModel

    torch.manual_seed(0)
    m3 = DetectionModel("yolo11n.yaml", ch=3, nc=80, verbose=False).eval()
    src, dst = tmp_path / "m3.pt", tmp_path / "m5.pt"
    torch.save({"model": m3.half(), "ema": None, "train_args": {}}, src)
    m3 = m3.float()
    rep = mc.make_inflated_checkpoint(src, dst)
    assert rep["stem_shape"][1] == 5
    ck = torch.load(dst, weights_only=False)
    m5 = ck["model"].float().eval()
    assert m5.yaml["channels"] == 5

    x_bgr_t = torch.rand(1, 3, 128, 128)  # B,G,R of frame t
    x_rgb = x_bgr_t[:, [2, 1, 0]]
    x5 = torch.cat(
        [torch.rand(1, 1, 128, 128), x_bgr_t, torch.rand(1, 1, 128, 128)], 1
    )  # random neighbours
    with torch.no_grad():
        y3 = m3(x_rgb)
        y5 = m5(x5)
    y3 = y3[0] if isinstance(y3, (tuple, list)) else y3
    y5 = y5[0] if isinstance(y5, (tuple, list)) else y5
    # fp16 round trip of the weights only: tiny differences
    assert torch.allclose(y3, y5, atol=2e-2, rtol=2e-2), float((y3 - y5).abs().max())
    # changing the neighbours must not change the output at step 0
    x5b = x5.clone()
    x5b[:, 0] = torch.rand(1, 128, 128)
    x5b[:, 4] = torch.rand(1, 128, 128)
    with torch.no_grad():
        y5b = m5(x5b)
    y5b = y5b[0] if isinstance(y5b, (tuple, list)) else y5b
    assert torch.equal(y5, y5b)


def test_border_patch_pads_every_channel_with_114():
    mc.patch_ultralytics_multichannel()
    im = np.full((100, 100, 5), 200, np.uint8)
    M = np.float32([[1, 0, 30], [0, 1, 30]])
    out = cv2.warpAffine(im, M, (100, 100), borderValue=(114, 114, 114))
    assert out.shape == (100, 100, 5)
    assert (out[0, 0] == 114).all() and (out[50, 50] == 200).all()
    out3 = cv2.warpPerspective(
        im,
        np.vstack([M, [0, 0, 1]]).astype(np.float32),
        (100, 100),
        borderValue=(114, 114, 114),
    )
    assert (out3[0, 0] == 114).all()
    # 3-channel images keep the stock behaviour
    im3 = np.full((100, 100, 3), 200, np.uint8)
    assert (
        cv2.warpAffine(im3, M, (100, 100), borderValue=(114, 114, 114))[0, 0] == 114
    ).all()


def test_identical_channels_stay_identical_through_geometric_aug():
    """All five channels equal -> after Ultralytics' mosaic/affine/flip they must still be equal."""
    from ultralytics.data.augment import RandomFlip, RandomPerspective
    from ultralytics.utils.instance import Instances

    mc.patch_ultralytics_multichannel()
    rng = np.random.default_rng(0)
    g = cv2.GaussianBlur(rng.integers(0, 255, (320, 320), dtype=np.uint8), (0, 0), 3)
    img = np.repeat(g[..., None], 5, axis=2)
    persp = RandomPerspective(
        degrees=5, translate=0.1, scale=0.5, shear=2, border=(-160, -160)
    )
    flip = RandomFlip(p=1.0, direction="horizontal")
    for _ in range(5):
        inst = Instances(
            np.array([[100.0, 100.0, 120.0, 120.0]], np.float32),
            segments=np.zeros((0, 1000, 2), np.float32),
            bbox_format="xyxy",
            normalized=False,
        )
        labels = {
            "img": img.copy(),
            "cls": np.zeros((1, 1)),
            "instances": inst,
            "resized_shape": (320, 320),
            "ori_shape": (320, 320),
        }
        labels = flip(persp(labels))
        out = labels["img"]
        assert out.shape[2] == 5
        for c in range(1, 5):
            assert np.array_equal(out[..., 0], out[..., c]), f"channel {c} diverged"
