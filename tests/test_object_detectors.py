from pathlib import Path
import json
from footy_track.detectors.ultralytics import UltralyticsObjectDetector
from footy_track.schema import FrameDetections, ObjectDetection
from footy_track.detectors.utils import calculate_iou


def test_ultralytics_object_detector_init():
    """Test that the UltralyticsObjectDetector can be initialized."""
    detector = UltralyticsObjectDetector()
    assert detector is not None


def test_ultralytics_object_detector_predict_from_path():
    """Test that the predict_from_path method returns a FrameDetections object."""
    detector = UltralyticsObjectDetector()
    test_image_path = Path("tests/data/arsenal_mancity_test_detection.jpg")
    frame_detections = detector.predict_from_path(test_image_path)
    assert isinstance(frame_detections, FrameDetections)
    assert frame_detections.uri == test_image_path
    assert len(frame_detections.detections) > 0
    for detection in frame_detections.detections:
        assert detection.label in ["person", "ball"]


def test_ultralytics_object_detector_iou():
    """Test that the detector achieves a minimum IoU for at least half the persons."""
    detector = UltralyticsObjectDetector()
    test_image_path = Path("tests/data/arsenal_mancity_test_detection.jpg")
    frame_detections = detector.predict_from_path(test_image_path)

    # Load ground truth data
    with open("tests/data/arsenal_mancity_test_detections.json") as f:
        ground_truth_data = json.load(f)

    # Remap ground truth labels and create ObjectDetection objects
    ground_truths = []
    for obj in ground_truth_data["objects"]:
        label = obj["label"]
        if (
            "player" in label
            or "referee" in label
            or "coach" in label
            or "goalkeeper" in label
        ):
            label = "person"
        elif "ball" in label:
            label = "ball"

        bbox = obj["bbox"]
        ground_truths.append(
            ObjectDetection(
                label=label, confidence=1.0, x=bbox[0], y=bbox[1], w=bbox[2], h=bbox[3]
            )
        )

    # Filter for persons
    gt_persons = [gt for gt in ground_truths if gt.label == "person"]
    pred_persons = [det for det in frame_detections.detections if det.label == "person"]

    if not gt_persons or not pred_persons:
        assert False, "No persons found in ground truth or predictions"

    # For each ground truth person, find the best matching prediction
    matches = 0
    for gt_person in gt_persons:
        best_iou = 0
        for pred_person in pred_persons:
            iou = calculate_iou(gt_person, pred_person)
            if iou > best_iou:
                best_iou = iou

        if best_iou >= 0.5:
            matches += 1

    # Check if at least half the persons have an IoU of 0.5 or greater
    assert matches / len(gt_persons) >= 0.5
