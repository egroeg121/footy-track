from footy_track.detectors.utils import top_left_wh_to_yolo_xywh


def test_top_left_to_yolo_center_conversion_basic():
    # Example values (normalized)
    x, y, w, h = (
        0.6039308905601501,
        0.647387683391571,
        0.037293076515197754,
        0.10883080959320068,
    )
    xc, yc, wn, hn = top_left_wh_to_yolo_xywh(x, y, w, h)

    # Expected centers
    assert abs(xc - (x + w / 2.0)) < 1e-9
    assert abs(yc - (y + h / 2.0)) < 1e-9
    assert wn == w
    assert hn == h


def test_top_left_to_yolo_clamping():
    # Inputs exceeding [0,1] should be clamped
    xc, yc, wn, hn = top_left_wh_to_yolo_xywh(-0.1, 0.95, 0.3, 0.2)
    assert 0.0 <= xc <= 1.0
    assert 0.0 <= yc <= 1.0
    assert 0.0 <= wn <= 1.0
    assert 0.0 <= hn <= 1.0
