"""Player Check geometry: clusters over time, count flicker, triage policy (design section 4c)."""

from __future__ import annotations

from footy_track.autolabel import clusters as cl


def _person(x, y=0.5, w=0.05, h=0.15):
    return (x, y, w, h)


def test_overlapping_boxes_form_one_cluster_isolated_stay_apart():
    boxes = [_person(0.10), _person(0.12), _person(0.80)]
    groups = {c.indices for c in cl.cluster_boxes(boxes)}
    assert groups == {(0, 1), (2,)}


def test_track_follows_a_cluster_across_frames():
    frames = {
        f: [_person(0.10 + 0.001 * f), _person(0.12 + 0.001 * f), _person(0.80)]
        for f in range(4)
    }
    tracks = cl.track_clusters(frames)
    assert sorted(len(t.obs) for t in tracks) == [4, 4]
    pair = next(t for t in tracks if t.obs[0].count == 2)
    assert [n for _, n in cl.count_timeline(pair)] == [2, 2, 2, 2]


def test_flicker_is_flagged_but_a_permanent_change_is_not():
    # 3,3,2,3 flickers; 3,3,4,4 is someone arriving
    assert cl.flicker_frames([(0, 3), (1, 3), (2, 2), (3, 3)]) == [2]
    assert cl.flicker_frames([(0, 3), (1, 3), (2, 4), (3, 4)]) == []
    assert cl.change_points([(0, 3), (1, 3), (2, 4), (3, 4)]) == [2]


def test_flicker_end_to_end_through_tracking():
    trio = [_person(0.10), _person(0.12), _person(0.14)]
    frames = {0: trio, 1: trio, 2: trio[:2], 3: trio}
    (track,) = cl.track_clusters(frames)
    tl = cl.count_timeline(track)
    assert [n for _, n in tl] == [3, 3, 2, 3]
    assert cl.flicker_frames(tl) == [2]
    assert cl.track_severity(track) > 10


def _stable_single_track():
    frames = {f: [_person(0.5 + 0.002 * f)] for f in range(5)}
    (track,) = cl.track_clusters(frames)
    conf = {(f, 0): 0.9 for f in range(5)}
    return track, conf


def test_isolated_stable_box_is_auto_ok_only_under_permissive_policy():
    track, conf = _stable_single_track()
    assert (
        cl.triage_track(track, policy=cl.POLICY_PERMISSIVE, confidences=conf).decision
        == "auto_ok"
    )
    assert (
        cl.triage_track(track, confidences=conf).decision == "review"
    )  # default = audit_all
    assert (
        cl.triage_track(track, policy="audit_all", confidences=conf).reason
        == "audit_all"
    )


def test_permissive_still_reviews_clusters_short_tracks_and_low_confidence():
    track, conf = _stable_single_track()
    low = dict.fromkeys(conf, 0.1)
    assert (
        cl.triage_track(track, policy="permissive", confidences=low).decision
        == "review"
    )
    (pair,) = cl.track_clusters({f: [_person(0.1), _person(0.12)] for f in range(5)})
    assert (
        cl.triage_track(pair, policy="permissive", confidences=conf).decision
        == "review"
    )
    (short,) = cl.track_clusters({0: [_person(0.5)]})
    assert (
        cl.triage_track(short, policy="permissive", confidences={(0, 0): 0.9}).decision
        == "review"
    )
