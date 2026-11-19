ROBOFLOW_BROADCAST_PROJECT = "footy-track-broadcast-frame"
ROBOFLOW_BROADCAST_PROJECT_TEST_PROJECT = "footy-track-broadcast-frame-test"
ROBOFLOW_DETECTION_PROJECT = "footy-track-detection"
ROBOFLOW_DETECTION_PROJECT_TEST_PROJECT = "footy-track-detection-test"
ROBOFLOW_WORKSPACE = "egroeg121"


IMAGE_FORMAT = "jpg"
BALL_TAG = "ball"
PERSON_TAG = "person"
OUT_OF_PLAY_BALL_TAG = "out_of_play_ball"
IN_PLAY_BALL_TAG = "in_play_ball"
PLAYER_TAG = "player"
PLAYER_SUB_TAG = "player_sub"
REFEREE_TAG = "referee"
COACH_TAG = "coach"


#: FROZEN training class ids (docs/design/full_frame_labelling.md section 2). The index IS the YOLO class id
#: in a multi-class export. Append only: never reorder or remove, or every trained model and exported dataset
#: silently changes meaning. A test (tests/feature_store/test_export_training_dataset.py) pins this list.
FROZEN_CLASS_NAMES: tuple[str, ...] = (
    PLAYER_TAG,  # 0
    BALL_TAG,  # 1
    REFEREE_TAG,  # 2
    COACH_TAG,  # 3
    PLAYER_SUB_TAG,  # 4
    PERSON_TAG,  # 5
    IN_PLAY_BALL_TAG,  # 6
    OUT_OF_PLAY_BALL_TAG,  # 7
)
FROZEN_CLASS_IDS: dict[str, int] = {n: i for i, n in enumerate(FROZEN_CLASS_NAMES)}
