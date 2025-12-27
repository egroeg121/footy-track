from footy_track.constants import ROBOFLOW_BROADCAST_PROJECT, ROBOFLOW_WORKSPACE
from footy_track.labelling import BaseRoboflowHandler


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
