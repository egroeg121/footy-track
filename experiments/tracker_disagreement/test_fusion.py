"""python -m pytest experiments/tracker_disagreement/test_fusion.py -n0 -p no:cacheprovider  (pure python)"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import fusion as F  # noqa: E402


def test_classify():
    a, b, c = (100.0, 100.0), (104.0, 100.0), (400.0, 100.0)
    assert F.classify([None, None, None]) == ("none", None)
    assert F.classify([a, None, None]) == ("only_one", "t1")
    assert F.classify([a, b, None]) == ("pair_agree", "t3")
    assert F.classify([a, c, None]) == ("pair_differ", "t3")
    assert F.classify([a, b, (102.0, 101.0)]) == ("all3_agree", None)
    assert F.classify([a, b, c]) == ("two_agree", "t3")
    assert F.classify([c, a, b]) == ("two_agree", "t1")
    assert F.classify([a, (300.0, 100.0), c]) == ("all_differ", None)


def test_runs_merge_small_gaps():
    flags = [False, True, True, False, True, False, False, False, False, True]
    assert F.runs(flags, gap=1) == [(1, 4), (9, 9)]
    assert F.runs(flags, gap=0) == [(1, 2), (4, 4), (9, 9)]
