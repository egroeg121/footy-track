from pathlib import Path

import torch
from ultralytics import YOLO, SAM
from ultralytics.engine.results import Results as UltralyticsResults

from footy_track.schema import FrameDetections, ObjectDetection

from .base import ObjectDetector
from .constants import BALL_TAG, PERSON_TAG
from .utils import ultralytics_result_to_detections


class UltralyticsObjectDetector(ObjectDetector):
    """YOLO-based object detector returning Pydantic outputs.

    Uses the ultralytics YOLO models and returns a FrameDetections instance
    with normalized [x, y, w, h] boxes in [0, 1].
    """

    def __init__(
        self,
        model_uri: str = "yolo11n.pt",
        verbose: bool = False,
        compile: bool = False,
        min_confidence: float = 0.3,
        iou_threshold: float = 0.90,
    ):
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.model = YOLO(model_uri)
        self.predict_kwargs = {
            "verbose": verbose,
            "compile": compile,
            "conf": min_confidence,
            "iou": iou_threshold,
        }

    @property
    def classes(self) -> list[str]:
        """Get the list of class names the model can detect."""
        # return self.model.names
        return {
            0: PERSON_TAG,
            32: BALL_TAG,
        }

    @torch.no_grad()
    def predict_from_path(
        self,
        image_path: Path,
    ) -> FrameDetections:
        """Run detection and return FrameDetections."""
        result: UltralyticsResults = self.model.predict(image_path, device=self.device)[
            0
        ]

        # Image size
        h, w = result.orig_shape[:2]

        # Build detections via modular converter
        detections = ultralytics_result_to_detections(result, self.classes)

        return FrameDetections(
            uri=Path(image_path),
            width=int(w),
            height=int(h),
            detections=detections,
        )


class UltralyticsSam3Detector(ObjectDetector):
    """Text-prompted segmentation with SAM 3 returning bounding-box detections.

    This detector uses Ultralytics SAM 3 with text prompts to segment concepts
    and converts the resulting masks' bounding boxes into normalized detections.

    Prompts used:
    - "soccer ball" -> label "ball"
    - "sports player" -> label "person"
    """

    def __init__(
        self,
        model_uri: str = "sam3.pt",
        min_confidence: float = 0.25,
        verbose: bool = False,
    ) -> None:
        self.device = (
            "mps"
            if torch.backends.mps.is_available()
            else "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )
        # Ultralytics will download weights if missing
        self.model = SAM(model_uri)
        self.verbose = verbose
        self.min_confidence = float(min_confidence)

        # Mapping from prompt -> canonical label used in our schema
        self.prompt_label_map: list[tuple[str, str]] = [
            ("soccer ball", BALL_TAG),
            ("sports player", PERSON_TAG),
        ]

    @torch.no_grad()
    def predict_from_path(self, image_path: Path) -> FrameDetections:
        """Run SAM3 with text prompts and return combined FrameDetections.

        We invoke the model once per prompt and merge the results, assigning
        labels based on the prompt used.
        """
        detections: list[ObjectDetection] = []
        img_path = Path(image_path)

        width = height = 0

        for prompt, label in self.prompt_label_map:
            # SAM 3 supports text-based concept segmentation via `prompt`
            results = self.model(
                str(img_path),  # Ultralytics accepts paths/arrays
                prompt=prompt,
                device=self.device,
                verbose=self.verbose,
            )

            # Each call returns a list with one Results for single-image input
            if not results:
                continue

            result = results[0]

            # Cache image size (same for all prompts)
            if width == 0 and height == 0:
                h, w = result.orig_shape[:2]
                width, height = int(w), int(h)

            # If boxes are not present (unlikely), skip
            if getattr(result, "boxes", None) is None:
                continue

            # Normalized xyxy boxes
            xyxyn = (
                result.boxes.xyxyn.tolist() if hasattr(result.boxes, "xyxyn") else []
            )
            scores = (
                result.boxes.conf.tolist()
                if hasattr(result.boxes, "conf") and result.boxes.conf is not None
                else []
            )

            for i, b in enumerate(xyxyn):
                x1, y1, x2, y2 = (float(b[0]), float(b[1]), float(b[2]), float(b[3]))
                # Convert xyxy (normalized) -> top-left wh (normalized)
                x = max(0.0, min(1.0, x1))
                y = max(0.0, min(1.0, y1))
                w_n = max(0.0, min(1.0, x2 - x1))
                h_n = max(0.0, min(1.0, y2 - y1))

                conf = float(scores[i]) if i < len(scores) else 1.0
                if conf < self.min_confidence:
                    continue

                detections.append(
                    ObjectDetection(
                        label=label,
                        confidence=conf,
                        x=x,
                        y=y,
                        w=w_n,
                        h=h_n,
                        model="sam3",
                    )
                )

        # Fallback to image size if not set (e.g., no results); read via ultralytics util by a dry run
        if width == 0 or height == 0:
            # Make a lightweight probe to get size (without prompts)
            probe = self.model(str(img_path), device=self.device, verbose=self.verbose)
            if probe:
                h, w = probe[0].orig_shape[:2]
                width, height = int(w), int(h)

        return FrameDetections(
            uri=img_path, width=width, height=height, detections=detections
        )
