import os
from datetime import datetime

import pytest

from footy_track.classifier import Classifier, get_current_best_guess_classifier
from footy_track.constants import (
    ROBOFLOW_BROADCAST_PROJECT,
    ROBOFLOW_BROADCAST_PROJECT_TEST_PROJECT,
    ROBOFLOW_DETECTION_PROJECT_TEST_PROJECT,
    ROBOFLOW_WORKSPACE,
)
from footy_track.labelling import (
    BaseRoboflowHandler,
    RoboflowClassificationHandler,
    RoboflowObjectDetectionHandler,
)
from footy_track.detectors.ultralytics import UltralyticsObjectDetector

ROBOFLOW_API_KEY = os.getenv("ROBOFLOW_API_KEY")


def test_get_roboflow_broadcast_prod_project():
    """Tests that we can successfully get the footy-track-broadcast-frame production project from roboflow."""
    handler = BaseRoboflowHandler(workspace_name=ROBOFLOW_WORKSPACE)
    project = handler.get_project(ROBOFLOW_BROADCAST_PROJECT)
    assert project is not None
    assert project.name == ROBOFLOW_BROADCAST_PROJECT


def test_get_roboflow_broadcast_test_project(roboflow_test_project_name: str):
    """Tests that we can successfully get the footy-track-broadcast-frame test project from roboflow."""
    handler = BaseRoboflowHandler(workspace_name=ROBOFLOW_WORKSPACE)
    project = handler.get_project(roboflow_test_project_name)
    assert project is not None
    assert project.name == roboflow_test_project_name


@pytest.mark.skipif(not ROBOFLOW_API_KEY, reason="ROBOFLOW_API_KEY not set")
class TestRoboflowClassificationHandlerIntegration:
    WORKSPACE = ROBOFLOW_WORKSPACE
    PROJECT = ROBOFLOW_BROADCAST_PROJECT_TEST_PROJECT

    @pytest.fixture
    def classifier(self) -> Classifier:
        """Returns the best classifier instance."""
        return get_current_best_guess_classifier()

    def test_handler_init(self, classifier):
        """Tests the initialization of the RoboflowClassificationHandler."""
        handler = RoboflowClassificationHandler(
            workspace_name=self.WORKSPACE,
            project_name=self.PROJECT,
            classifier=classifier,
        )
        assert handler.workspace is not None
        assert handler.project is not None
        assert handler.project.id == f"{self.WORKSPACE}/{self.PROJECT}"

    @pytest.mark.slow
    def test_upload_images(self, classifier, extracted_frames):
        """Tests the upload_images method."""
        handler = RoboflowClassificationHandler(
            workspace_name=self.WORKSPACE,
            project_name=self.PROJECT,
            classifier=classifier,
        )
        batch_name = f"sdk-test-batch-images-{datetime.now().strftime('%Y%m%d%H%M%S')}"
        try:
            # Upload a small sample to keep tests fast
            handler.upload_images(extracted_frames[:2], batch_name=batch_name)
        finally:
            # TODO: Implement batch deletion when the SDK supports it
            pass

    @pytest.mark.slow
    def test_upload_dir(self, classifier, frames_path):
        """Tests the upload_dir method."""
        handler = RoboflowClassificationHandler(
            workspace_name=self.WORKSPACE,
            project_name=self.PROJECT,
            classifier=classifier,
        )
        batch_name = f"sdk-test-batch-dir-{datetime.now().strftime('%Y%m%d%H%M%S')}"
        try:
            handler.upload_dir(frames_path, batch_name=batch_name)
        finally:
            # TODO: Implement batch deletion when the SDK supports it
            pass

    @pytest.mark.slow
    def test_upload_dir_with_sampling(self, classifier, frames_path):
        """Tests the upload_dir method with sampling."""
        handler = RoboflowClassificationHandler(
            workspace_name=self.WORKSPACE,
            project_name=self.PROJECT,
            classifier=classifier,
        )
        batch_name = (
            f"sdk-test-batch-dir-sample-{datetime.now().strftime('%Y%m%d%H%M%S')}"
        )
        try:
            handler.upload_dir(frames_path, sample_number=5, batch_name=batch_name)
        finally:
            # TODO: Implement batch deletion when the SDK supports it
            pass


@pytest.mark.skipif(not ROBOFLOW_API_KEY, reason="ROBOFLOW_API_KEY not set")
class TestRoboflowObjectDetectionHandlerIntegration:
    WORKSPACE = ROBOFLOW_WORKSPACE
    PROJECT = ROBOFLOW_DETECTION_PROJECT_TEST_PROJECT

    @pytest.fixture
    def detector(self):
        """Returns an UltralyticsObjectDetector instance."""
        return UltralyticsObjectDetector()

    @pytest.fixture
    def classifier(self) -> Classifier:
        """Returns a RandomClassifier instance."""
        return get_current_best_guess_classifier()

    def test_handler_init(self, detector, classifier):
        """Tests the initialization of the RoboflowObjectDetectionHandler."""
        handler = RoboflowObjectDetectionHandler(
            workspace_name=self.WORKSPACE,
            project_name=self.PROJECT,
            detector=detector,
            classifier=classifier,
        )
        assert handler.workspace is not None
        assert handler.project is not None
        assert handler.project.id == f"{self.WORKSPACE}/{self.PROJECT}"

    @pytest.mark.slow
    def test_upload_images(self, detector, classifier, extracted_frames):
        """Tests the upload_images method."""
        handler = RoboflowObjectDetectionHandler(
            workspace_name=self.WORKSPACE,
            project_name=self.PROJECT,
            detector=detector,
            classifier=classifier,
        )
        batch_name = (
            f"sdk-test-obj-det-batch-images-{datetime.now().strftime('%Y%m%d%H%M%S')}"
        )
        try:
            # Upload a small sample to keep tests fast
            handler.upload_images(extracted_frames[:2], batch_name=batch_name)
        finally:
            # TODO: Implement batch deletion when the SDK supports it
            pass

    @pytest.mark.slow
    def test_upload_dir(self, detector, classifier, frames_path):
        """Tests the upload_dir method."""
        handler = RoboflowObjectDetectionHandler(
            workspace_name=self.WORKSPACE,
            project_name=self.PROJECT,
            detector=detector,
            classifier=classifier,
        )
        batch_name = (
            f"sdk-test-obj-det-batch-dir-{datetime.now().strftime('%Y%m%d%H%M%S')}"
        )
        try:
            handler.upload_dir(frames_path, batch_name=batch_name)
        finally:
            # TODO: Implement batch deletion when the SDK supports it
            pass

    @pytest.mark.slow
    def test_upload_dir_with_sampling(self, detector, classifier, frames_path):
        """Tests the upload_dir method with sampling."""
        handler = RoboflowObjectDetectionHandler(
            workspace_name=self.WORKSPACE,
            project_name=self.PROJECT,
            detector=detector,
            classifier=classifier,
        )
        batch_name = f"sdk-test-obj-det-batch-dir-sample-{datetime.now().strftime('%Y%m%d%H%M%S')}"
        try:
            handler.upload_dir(frames_path, sample_number=5, batch_name=batch_name)
        finally:
            # TODO: Implement batch deletion when the SDK supports it
            pass
