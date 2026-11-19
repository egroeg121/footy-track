"""Window selection (rule zero, segments, complete frames, determinism) and propose (fakes)."""

from __future__ import annotations

import json

import numpy as np
import pytest

from footy_track import match_splits as ms
from footy_track import splits as sp
from footy_track.autolabel import propose as pr
from footy_track.autolabel import windows as wd
from footy_track.ball_trackers.detection_tracker import BallPoint
from footy_track.feature_store.schema import (
    ClsGroup,
    FrameLabelStateRow,
    LabelMethod,
    LabelState,
)
from footy_track.feature_store.store import FeatureStore
from footy_track.schema import FrameDetections, ObjectDetection

MATCHES = {
    "arsenal_mancity": {"split": "train", "aliases": ["mancity"]},
    "chelsea-vs-hull-city-2026-27": {
        "split": "train",
        "aliases": ["chelsea_vs_hull_city_2026_27"],
    },
    "arsenal_nottinghamforest": {"split": "heldout", "aliases": ["nottinghamforest"]},
}
MC = "arsenal_mancity"
CH = "chelsea_vs_hull_city_2026_27"
HELD = "arsenal_nottinghamforest"


@pytest.fixture(autouse=True)
def regs(tmp_path, monkeypatch):
    mp = tmp_path / "match_splits.json"
    mp.write_text(json.dumps(MATCHES))
    monkeypatch.setenv("FOOTY_MATCH_SPLITS", str(mp))
    monkeypatch.delenv("FOOTY_CLIP_SPLITS", raising=False)
    monkeypatch.delenv("FOOTY_MATCH_SPLITS_ALLOW_SEED")
    monkeypatch.setenv("FOOTY_FULL_MATCHES_MANIFEST", str(tmp_path / "none.jsonl"))
    # explicit clip register: everything train except MC seg009 (in-match eval)
    clips = {
        f"{m}_seg{i:03d}": {"split": "train"} for m in (MC, CH, HELD) for i in range(4)
    }
    clips[f"{MC}_seg009"] = {"split": "eval"}
    sp.clip_register_path().write_text(
        json.dumps(
            {"version": 1, "rule": {"modulus": 5, "eval_remainder": 0}, "clips": clips}
        )
    )
    sp.clear_cache()
    yield
    sp.clear_cache()


def clip(name, n=1000):
    return wd.ClipInfo(name, n, f"/v/{name}.mp4")


ALL = [
    clip(f"{MC}_seg000"),
    clip(f"{MC}_seg001"),
    clip(f"{MC}_seg009"),  # in-match eval
    clip(f"{CH}_seg000"),
    clip(f"{HELD}_seg000"),  # heldout match
    clip(f"{MC}_part050"),  # not a _seg clip
    clip("unknown_match_seg000"),  # unregistered
]


def test_windows_inside_one_segment_and_exact_size():
    ws = wd.select_windows(ALL, count=100)
    assert ws
    for w in ws:
        assert w.end_frame - w.start_frame == 250
        c = next(c for c in ALL if c.game_id == w.game_id)
        assert w.start_frame >= 0 and w.end_frame <= c.n_frames


def test_never_touches_heldout_eval_or_unregistered():
    ws = wd.select_windows(ALL, count=100)
    used = {w.game_id for w in ws}
    assert used == {f"{MC}_seg000", f"{MC}_seg001", f"{CH}_seg000"}
    for w in ws:
        assert sp.split_of(w.match_key, w.clip_id) == sp.TRAIN


def test_fails_closed_without_register(monkeypatch, tmp_path):
    monkeypatch.setenv("FOOTY_MATCH_SPLITS", str(tmp_path / "missing.json"))
    ms.clear_cache()
    sp.clear_cache()
    with pytest.raises(ms.RegisterError):
        wd.select_windows(ALL, count=3)


def test_balanced_across_matches():
    ws = wd.select_windows(ALL, count=4)
    keys = [w.match_key for w in ws]
    assert keys.count(MC) == keys.count("chelsea-vs-hull-city-2026-27") == 2


def test_skips_complete_frames_from_store():
    with FeatureStore.open(":memory:") as st:
        st.upsert_frame_label_states(
            [
                FrameLabelStateRow(
                    game_id=f"{CH}_seg000",
                    frame_index=260,
                    cls_group=g,
                    state=LabelState.COMPLETE,
                    method=LabelMethod.HAND_SERIES,
                )
                for g in (ClsGroup.BALL, ClsGroup.PLAYER)
            ]
        )
        ws = wd.select_windows(
            [clip(f"{CH}_seg000")],
            count=10,
            complete_frames=lambda g: wd.complete_frames_from_store(st, g),
        )
    assert {w.start_frame for w in ws} == {0, 500, 750}


def test_ranks_by_fewest_labels_then_agreement():
    g = f"{CH}_seg000"
    ws = wd.select_windows(
        [clip(g)],
        count=4,
        labelled_frames=lambda _g: set(range(0, 10)),
        agreement={g: {f: (0.2 if f >= 750 else 0.9) for f in range(1000)}},
    )
    assert [w.start_frame for w in ws] == [750, 250, 500, 0]
    assert [w.rank for w in ws] == [0, 1, 2, 3]


def test_deterministic_output(tmp_path):
    a, b = tmp_path / "a.json", tmp_path / "b.json"
    wd.write_windows(a, wd.select_windows(list(reversed(ALL)), count=7), 250)
    wd.write_windows(b, wd.select_windows(ALL, count=7), 250)
    assert a.read_bytes() == b.read_bytes()
    assert wd.read_windows(a) == wd.select_windows(ALL, count=7)


# ---- propose -----------------------------------------------------------------------------------
W, H = 1920, 1080


class FakeReader:
    def fps(self, video):
        return 50.0

    def read(self, video, start, end):
        for i in range(start, end):
            yield i, np.zeros((4, 4, 3), np.uint8)


def person(x=0.1, y=0.1):
    return ObjectDetection(label="player", confidence=0.9, x=x, y=y, w=0.05, h=0.1)


def ball(x, conf=0.9):
    return ObjectDetection(label="ball", confidence=conf, x=x, y=0.5, w=0.01, h=0.01)


class FakeDetector:
    model_tag = "fake-det"

    def __init__(self, drop_person_at=()):
        self.n = 0
        self.drop = set(drop_person_at)

    def detect(self, frames):
        out = []
        for _ in frames:
            i = self.n
            self.n += 1
            people = [person(0.1 + 0.1 * k) for k in range(4)]
            if i in self.drop:
                people = people[:3]
            out.append(
                FrameDetections(
                    uri=f"f{i}",
                    width=W,
                    height=H,
                    detections=[*people, ball(0.3 + 0.001 * i)],
                )
            )
        return out


class FakeSam3:
    model_tag = "fake-sam3"

    def __init__(self):
        self.calls = 0

    def people(self, frame):
        self.calls += 1
        return [person(0.1 + 0.1 * k) for k in range(4)]


def window(start=0):
    g = f"{CH}_seg000"
    key = "chelsea-vs-hull-city-2026-27"
    return wd.Window(
        wd.window_id_of(g, start), g, g, key, "/v/x.mp4", start, start + 20
    )


def test_propose_ingests_unreviewed_and_scores(tmp_path):
    sam = FakeSam3()
    with FeatureStore.open(":memory:") as st:
        res = pr.propose_window(
            window(),
            reader=FakeReader(),
            detector=FakeDetector(drop_person_at={10}),
            person_filler=sam,
            store=st,
            out_dir=tmp_path,
        )
        df = st.query("SELECT * FROM detection")
        assert set(df["source"]) == {"yolo", "vittrack", "sam3"}
        assert not df["reviewed"].any() and df["needs_review"].all()
        assert set(df["run_id"]) == set(res.run_ids.values())
        runs = set(st.query("SELECT run_id FROM run")["run_id"])
        assert res.run_id in runs and set(res.run_ids.values()) <= runs
        # SAM3 ran only on the flagged frame and added only the missing person
        assert sam.calls == 1 and res.sam3_frames == [10]
        s3 = df[df["source"] == "sam3"]
        assert len(s3) == 1 and int(s3["frame_index"].iloc[0]) == 10
        # re-propose replaces rows instead of duplicating
        n = st.count("detection")
        pr.propose_window(
            window(),
            reader=FakeReader(),
            detector=FakeDetector(drop_person_at={10}),
            person_filler=FakeSam3(),
            store=st,
        )
        assert st.count("detection") == n
        # refuses to overwrite once reviewed
        st._conn.execute("UPDATE detection SET reviewed = TRUE")
        with pytest.raises(RuntimeError):
            pr.propose_window(
                window(), reader=FakeReader(), detector=FakeDetector(), store=st
            )
    z = np.load(res.scores_path)
    assert z["score"].shape == (20,) and z["frame_index"][0] == 0
    assert z["person_flag"][10] and z["sam3_person_count"][10] == 4
    assert z["sam3_person_count"][0] == -1 and z["det_person_count"][10] == 3


def test_propose_refuses_heldout_window():
    g = f"{HELD}_seg000"
    w = wd.Window(wd.window_id_of(g, 0), g, g, HELD, "/v/x.mp4", 0, 20)
    with pytest.raises(ValueError, match="refusing"):
        pr.propose_window(w, reader=FakeReader(), detector=FakeDetector())


def bp(i, x=None, src="detected"):
    if x is None:
        return BallPoint(i)
    return BallPoint(i, src, 0, x, 0.5, 0.01, 0.01, 0.9)


def test_disagreement_scores_fixture():
    fwd = [bp(0), bp(1, 0.5), bp(2, 0.5), bp(3, 0.5)]
    bwd = [bp(0), bp(1, 0.5), bp(2, 0.5 + 100 / W), bp(3)]
    z = pr.disagreement_scores([0, 1, 2, 3], fwd, bwd, [5, 5, 5, 5], None, (W, H))
    assert z["ball_component"][0] == 0 and z["ball_component"][1] == 0
    assert np.isnan(z["ball_gap_px"][0])
    assert z["ball_gap_px"][2] == pytest.approx(100, abs=1)
    assert 0 < z["ball_component"][2] < 1
    assert z["ball_presence_mismatch"][3] and z["ball_component"][3] == 1.0
    assert z["person_component"].sum() == 0
    z2 = pr.disagreement_scores(
        [0, 1, 2], fwd[:3], bwd[:3], [5, 5, 5], [-1, 3, -1], (W, H)
    )
    assert z2["person_component"][1] == pytest.approx(1 - 1 / 3)
    assert z2["score"][1] >= z2["person_component"][1]


def test_person_count_flags():
    assert pr.person_count_flags([4, 4, 3, 4, 4], radius=2) == [
        False,
        False,
        True,
        False,
        False,
    ]
