from footy_track.constants import ROBOFLOW_BROADCAST_PROJECT, ROBOFLOW_WORKSPACE
from footy_track.labelling import BaseRoboflowHandler


def test_get_roboflow_broadcast_project():
    """Tests that we can successfully get the footy-track-broadcast-frame project from roboflow."""
    handler = BaseRoboflowHandler(workspace_name=ROBOFLOW_WORKSPACE)
    project = handler.get_project(ROBOFLOW_BROADCAST_PROJECT)
    assert project is not None
    assert project.name == "footy-track-broadcast-frame"
