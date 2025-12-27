from pathlib import Path
from typing import Any

import fiftyone as fo
import torch
from PIL import Image
from torchvision import transforms, utils

from footy_track.schema import FrameDetections, ObjectDetection


def _available_device():
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    return device


color_map = {
    "person": (70, 130, 180),  # players are all blue-ish
    "player": (30, 144, 255),
    "referee": (65, 105, 225),
    "coach": (100, 149, 237),
    "ball": (255, 200, 0),  # balls are all yellow/orange
    "ball_in_play": (255, 215, 0),
    "ball_out_of_play": (255, 140, 0),
    "unknown": (128, 0, 128),  # purple for unknown
}


def _clamp01(v: float) -> float:
    return float(max(0.0, min(1.0, v)))


def draw_bounding_boxes_pil(
    image: Image.Image,
    boxes: torch.Tensor,
    *args,
    **kwargs,
) -> Image.Image:
    """
    Draw bounding boxes on a Pillow image using torchvision's draw_bounding_boxes.

    Args:
        image (PIL.Image.Image): Input image.
        boxes (torch.Tensor): Tensor of shape [N, 4] in (xmin, ymin, xmax, ymax) format.
        *args: Additional positional arguments for `draw_bounding_boxes`.
        **kwargs: Additional keyword arguments for `draw_bounding_boxes`.
    Returns:
        PIL.Image.Image: Image with drawn boxes.
    """
    # Convert PIL -> Tensor [C, H, W], range [0,255]
    to_tensor = transforms.ToTensor()
    tensor_img = (to_tensor(image) * 255).to(torch.uint8)

    # Draw bounding boxes
    boxed_tensor = utils.draw_bounding_boxes(tensor_img, boxes, *args, **kwargs)

    # Convert Tensor -> PIL
    to_pil = transforms.ToPILImage()
    return to_pil(boxed_tensor)


def visualise_detections_on_image(
    frame_detections: FrameDetections, save_path: Path | None = None, show: bool = True
) -> Path | None:
    """Draw detections on the source image and optionally save/show it.

    - Uses normalized bbox coordinates in `frame_detections.detections`
    - If `save_path` is provided, the annotated image is saved there and the path is returned
    - If `show` is True, the image is opened with the default image viewer
    """
    img_path = Path(frame_detections.uri)
    if not img_path.exists():
        raise FileNotFoundError(f"Image file not found: {img_path}")

    img = Image.open(img_path).convert("RGB")
    w, h = img.size

    thickness = max(2, min(6, int(round(min(w, h) * 0.003))))

    # Build boxes tensor in (xmin, ymin, xmax, ymax) absolute pixels
    boxes_list: list[list[float]] = []
    labels: list[str] = []
    colors: list[tuple[int, int, int]] = []

    for det in frame_detections.detections:
        x1 = float(det.x) * w
        y1 = float(det.y) * h
        x2 = x1 + float(det.w) * w
        y2 = y1 + float(det.h) * h
        boxes_list.append([x1, y1, x2, y2])
        labels.append(f"{det.label} {det.confidence:.2f}")
        colors.append(color_map.get(det.label.lower(), (255, 255, 255)))

    out_img = img
    if boxes_list:
        boxes_tensor = torch.tensor(boxes_list, dtype=torch.float32)

        # torchvision.draw_bounding_boxes supports a single color or list of colors
        out_img = draw_bounding_boxes_pil(
            img,
            boxes_tensor,
            labels=labels,
            colors=colors,
            width=thickness,
        )

    out_path: Path | None = None
    if save_path is not None:
        out_path = Path(save_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_img.save(out_path)

    if show:
        out_img.show()

    return out_path


def ultralytics_result_to_detections(
    result: Any, classes: list[str] | dict[int, str]
) -> list[ObjectDetection]:
    """Convert an Ultralytics YOLO `Results` object to a list of `Detection`.

    Parameters
    ----------
    result: UltralyticsResults-like
        Single-image prediction result from ultralytics
    classes: list[str] | dict[int, str]
        Mapping of class indices to names
    """
    if getattr(result, "boxes", None) is None:
        return []

    labels = result.boxes.cls.int().tolist()
    scores = result.boxes.conf.tolist()
    xyxyn = result.boxes.xyxyn.tolist()  # normalized x1,y1,x2,y2

    out: list[ObjectDetection] = []
    for label_idx, score, (x1, y1, x2, y2) in zip(labels, scores, xyxyn, strict=False):
        x = _clamp01(float(x1))
        y = _clamp01(float(y1))
        w_n = _clamp01(max(0.0, float(x2) - float(x1)))
        h_n = _clamp01(max(0.0, float(y2) - float(y1)))

        try:
            label_name = classes[int(label_idx)]  # type: ignore[index]
        except Exception:
            label_name = str(int(label_idx))

        out.append(
            ObjectDetection(
                label=str(label_name),
                confidence=float(score),
                x=x,
                y=y,
                w=w_n,
                h=h_n,
            )
        )
    return out


# ------------------------------
# Converters to FiftyOne
# ------------------------------


def detection_to_fiftyone(d: ObjectDetection) -> fo.Detection:
    """Convert a single Detection to a FiftyOne Detection.

    Returns
    -------
    fiftyone.core.labels.Detection
        With bounding_box as [x, y, w, h] and confidence
    """
    return fo.Detection(
        label=d.label,
        bounding_box=[float(d.x), float(d.y), float(d.w), float(d.h)],
        confidence=float(d.confidence),
    )


def frame_to_fiftyone_detections(frame: FrameDetections) -> list[fo.Detection]:
    """Convert FrameDetections to a list of FiftyOne Detection objects."""
    return [detection_to_fiftyone(d) for d in frame.detections]


def calculate_iou(box1: ObjectDetection, box2: ObjectDetection) -> float:
    """Calculate the Intersection over Union (IoU) of two bounding boxes."""
    # Convert from (x, y, w, h) to (x1, y1, x2, y2)
    box1_x1, box1_y1, box1_x2, box1_y2 = (
        box1.x,
        box1.y,
        box1.x + box1.w,
        box1.y + box1.h,
    )
    box2_x1, box2_y1, box2_x2, box2_y2 = (
        box2.x,
        box2.y,
        box2.x + box2.w,
        box2.y + box2.h,
    )

    # Calculate the area of intersection
    inter_x1 = max(box1_x1, box2_x1)
    inter_y1 = max(box1_y1, box2_y1)
    inter_x2 = min(box1_x2, box2_x2)
    inter_y2 = min(box1_y2, box2_y2)
    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)

    # Calculate the area of both bounding boxes
    box1_area = box1.w * box1.h
    box2_area = box2.w * box2.h

    # Calculate the IoU
    iou = inter_area / float(box1_area + box2_area - inter_area)
    return iou
