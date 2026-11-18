"""build_broadcast_classifier_dataset: Roboflow train images are filtered by the split authority only."""

from __future__ import annotations

from footy_track import splits as authority
from footy_track.scripts import build_broadcast_classifier_dataset as bb


def _rf(tmp_path, names):
    for cls, name in names:
        d = tmp_path / "rf" / "train" / cls
        d.mkdir(parents=True, exist_ok=True)
        (d / name).write_bytes(b"x")
    return tmp_path / "rf"


def test_roboflow_images_follow_the_authority(monkeypatch, tmp_path):
    rf = _rf(tmp_path, [
        ("Yes", "heldmatch_000001_jpg.rf.aa.jpg"),
        ("No", "trainmatch_000002_jpg.rf.bb.jpg"),
        ("Yes", "mystery_000003_jpg.rf.cc.jpg"),
    ])
    keys = {"heldmatch_000001": "held", "trainmatch_000002": "tr"}

    class Ref:
        def __init__(self, name):
            stem = name.split("_jpg")[0]
            self.match_key, self.clip_id, self.stem = keys.get(stem), None, stem

    monkeypatch.setattr(authority, "parse_image_name", lambda n: Ref(n))
    monkeypatch.setattr(authority, "split_of", lambda key, clip=None: (
        authority.EVAL_HELDOUT if key == "held" else authority.TRAIN))
    out = tmp_path / "out"
    res = bb.add_roboflow_train(rf, out)
    assert res["counts"] == {"heldout": 1, "match_only": 1, "unmapped": 1}
    kept = sorted(p.name for p in (out / "train").rglob("*.jpg"))
    assert kept == ["rf_trainmatch_000002_jpg.rf.bb.jpg"]
    assert not (out / "valid").exists()
