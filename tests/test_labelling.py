import os
from datetime import datetime

import pytest

from footy_track.classifier import RandomClassifier
from footy_track.constants import (
    ROBOFLOW_BROADCAST_PROJECT_TEST_PROJECT,
    ROBOFLOW_WORKSPACE,
)
from footy_track.labelling import RoboflowClassificationHandler

ROBOFLOW_API_KEY = os.getenv("ROBOFLOW_API_KEY")


@pytest.mark.skipif(not ROBOFLOW_API_KEY, reason="ROBOFLOW_API_KEY not set")
class TestRoboflowClassificationHandlerIntegration:
    WORKSPACE = ROBOFLOW_WORKSPACE
    PROJECT = ROBOFLOW_BROADCAST_PROJECT_TEST_PROJECT

    @pytest.fixture
    def classifier(self):
        """Returns a RandomClassifier instance."""
        return RandomClassifier()

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

    def test_upload_dir_with_sampling(self, classifier, frames_path):
        """Tests the upload_dir method with sampling."""
        handler = RoboflowClassificationHandler(
            workspace_name=self.WORKSPACE,
            project_name=self.PROJECT,
            classifier=classifier,
        )
        batch_name = f"sdk-test-batch-dir-sample-{datetime.now().strftime('%Y%m%d%H%M%S')}"
        try:
            handler.upload_dir(frames_path, sample_number=5, batch_name=batch_name)
        finally:
            # TODO: Implement batch deletion when the SDK supports it
            pass
