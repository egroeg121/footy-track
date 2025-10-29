from pathlib import Path

import torch
from PIL import Image

from footy_track.object_detections.schema import Detection, FrameDetections
from footy_track.object_detections.utils import (
    draw_bounding_boxes_pil,
    visualise_detections_on_image,
)


def test_draw_bounding_boxes_pil_simple():
    # Create a simple RGB image (solid gray)
    img = Image.new("RGB", (100, 60), color=(128, 128, 128))

    # Single box in pixel coords: (xmin, ymin, xmax, ymax)
    boxes = torch.tensor([[10.0, 5.0, 60.0, 40.0]], dtype=torch.float32)

    # Draw with an explicit RGB color and width
    out = draw_bounding_boxes_pil(img, boxes, colors=[(255, 0, 0)], width=2)

    # Returns a PIL image of the same size
    assert isinstance(out, Image.Image)
    assert out.size == img.size

    # Pixels should differ from the original (box border drawn)
    assert out.tobytes() != img.tobytes()

    # Check specific pixels for correct placement
    red = (255, 0, 0)
    gray = (128, 128, 128)

    # Top-left corner on the top edge
    assert out.getpixel((10, 5)) == red

    # Another point along the top edge (avoid exact xmax to not depend on inclusivity)
    assert out.getpixel((59, 5)) == red

    # A point on the left edge midway down
    assert out.getpixel((10, 20)) == red

    # Inside the box (should remain original gray since only borders are drawn)
    assert out.getpixel((30, 20)) == gray

    # Outside the box
    assert out.getpixel((0, 0)) == gray


def test_visualise_detections_pil(tmp_path: Path):
    # Create a simple RGB image and save
    img_path = tmp_path / "input.png"
    img = Image.new("RGB", (200, 100), color=(128, 128, 128))
    img.save(img_path)

    # One detection centered
    det = Detection(label="player", confidence=0.9, x=0.4, y=0.3, w=0.2, h=0.4)
    frame = FrameDetections(uri=img_path, width=200, height=100, detections=[det])

    out_path = tmp_path / "out.png"

    returned = visualise_detections_on_image(frame, save_path=out_path, show=False)

    # Returned path should be the out_path
    assert returned is not None
    assert Path(returned).exists()

    out_img = Image.open(returned).convert("RGB")
    in_img = Image.open(img_path).convert("RGB")

    # Same size
    assert out_img.size == in_img.size

    # The annotated image should differ from the original (boxes drawn)
    assert out_img.tobytes() != in_img.tobytes()
