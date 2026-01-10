import json
from pathlib import Path

from footy_track.constants import COACH_TAG, PLAYER_SUB_TAG, PLAYER_TAG, REFEREE_TAG
from footy_track.detectors.ultralytics import (
    UltralyticsObjectDetector,
    UltralyticsSam3Detector,
    UltralyticsSam3VideoDetector,
)
from footy_track.detectors.utils import calculate_iou
from footy_track.schema import FrameDetections, ObjectDetection


class TestUltralyticsYOLODetector:
    def test_ultralytics_object_detector_init(self):
        """Test that the UltralyticsObjectDetector can be initialized."""
        detector = UltralyticsObjectDetector()
        assert detector is not None

    def test_ultralytics_object_detector_predict_from_path(self):
        """Test that the predict_from_path method returns a FrameDetections object."""
        detector = UltralyticsObjectDetector()
        test_image_path = Path("tests/data/arsenal_mancity_test_detection.jpg")
        frame_detections = detector.predict_from_path(test_image_path)
        assert isinstance(frame_detections, FrameDetections)
        assert frame_detections.uri == test_image_path
        assert len(frame_detections.detections) > 0
        for detection in frame_detections.detections:
            assert detection.label in ["person", "ball"]

    def test_ultralytics_object_detector_iou(self):
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
                or label == "goalkeeper"
                or label == "referee"
                or label == "coach"
            ):
                label = "person"
            elif "ball" in label:
                label = "ball"

            bbox = obj["bbox"]
            ground_truths.append(
                ObjectDetection(
                    label=label,
                    confidence=1.0,
                    x=bbox[0],
                    y=bbox[1],
                    w=bbox[2],
                    h=bbox[3],
                )
            )

        # Filter for persons
        gt_persons = [gt for gt in ground_truths if gt.label == "person"]
        pred_persons = [
            det for det in frame_detections.detections if det.label == "person"
        ]

        if not gt_persons or not pred_persons:
            raise AssertionError("No persons found in ground truth or predictions")

        # For each ground truth person, find the best matching prediction
        matches = 0
        for gt_person in gt_persons:
            best_iou = 0
            for pred_person in pred_persons:
                iou = calculate_iou(gt_person, pred_person)
                best_iou = max(best_iou, iou)

            if best_iou >= 0.5:
                matches += 1

        # Check if at least half the persons have an IoU of 0.5 or greater
        assert matches / len(gt_persons) >= 0.5


class TestUltralyticsSam3Detector:
    def test_ultralytics_sam3_detector_init(self):
        """Test that the UltralyticsSam3Detector can be initialized."""
        detector = UltralyticsSam3Detector()
        assert detector is not None

    def test_ultralytics_sam3_detector_predict_from_path(self):
        """Test that the SAM3 predict_from_path returns a FrameDetections object."""
        detector = UltralyticsSam3Detector()
        test_image_path = Path("tests/data/arsenal_mancity_test_detection.jpg")
        frame_detections = detector.predict_from_path(test_image_path)
        assert isinstance(frame_detections, FrameDetections)
        assert frame_detections.uri == test_image_path
        assert len(frame_detections.detections) > 0
        for detection in frame_detections.detections:
            # assert detection.label in ["person", "ball"]
            assert detection.label in detector.output_classes

    def test_ultralytics_sam3_detector_iou(self):
        """Similar IoU quality check for SAM3 on persons."""
        detector = UltralyticsSam3Detector()
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
                or label == "goalkeeper"
                or label == "referee"
                or label == "coach"
            ):
                label = PLAYER_TAG
            elif "ball" in label:
                label = "ball"

            bbox = obj["bbox"]
            ground_truths.append(
                ObjectDetection(
                    label=label,
                    confidence=1.0,
                    x=bbox[0],
                    y=bbox[1],
                    w=bbox[2],
                    h=bbox[3],
                )
            )

        # Filter for persons
        gt_persons = [gt for gt in ground_truths if gt.label == PLAYER_TAG]
        person_like_labels = {PLAYER_TAG, PLAYER_SUB_TAG, REFEREE_TAG, COACH_TAG}
        pred_persons = [
            det
            for det in frame_detections.detections
            if det.label in person_like_labels
        ]

        if not gt_persons or not pred_persons:
            raise AssertionError("No persons found in ground truth or predictions")

        # For each ground truth person, find the best matching prediction
        matches = 0
        for gt_person in gt_persons:
            best_iou = 0
            for pred_person in pred_persons:
                iou = calculate_iou(gt_person, pred_person)
                best_iou = max(best_iou, iou)

            if best_iou >= 0.5:
                matches += 1

        # Check if at least half the persons have an IoU of 0.5 or greater
        assert matches / len(gt_persons) >= 0.5


class TestUltralyticsSam3VideoDetector:
    def test_ultralytics_sam3_video_detector_init(self):
        """Test that the UltralyticsSam3VideoDetector can be initialized."""
        detector = UltralyticsSam3VideoDetector()
        assert detector is not None

    def test_ultralytics_sam3_video_detector_predict_from_video_path(self):
        """Test that the SAM3 predict_from_video_path returns a list of FrameDetections."""
        detector = UltralyticsSam3VideoDetector()
        test_video_path = Path(
            "tests/data/split_videos/arsenal_mancity_20250925_part192_part000.mp4"
        )
        frame_detections_list = detector.predict_from_video_path(test_video_path)
        assert isinstance(frame_detections_list, list)
        assert len(frame_detections_list) > 0
        for frame_detections in frame_detections_list:
            assert isinstance(frame_detections, FrameDetections)
            # Allow empty detections for now
            for detection in frame_detections.detections:
                assert detection.label in detector.output_classes
