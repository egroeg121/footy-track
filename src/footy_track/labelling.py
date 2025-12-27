import logging
import os

from roboflow import Roboflow

_logger = logging.getLogger(__name__)


class BaseRoboflowHandler:
    """Base handler for Roboflow API interactions."""

    def __init__(self, workspace_name: str, api_key: str = None):
        self.api_key = api_key or os.getenv("ROBOFLOW_API_KEY")
        if not self.api_key:
            raise ValueError("ROBOFLOW_API_KEY environment variable not set.")

        self.rf = Roboflow(api_key=self.api_key)
        self.workspace = self.rf.workspace(workspace_name)
        self._project = None

    def get_project(self, project_name: str):
        """Get or create a Roboflow project."""

        project = self.workspace.project(project_name)
        _logger.info(f"Using existing project: {project.name}")
        self._project = project
        return project
