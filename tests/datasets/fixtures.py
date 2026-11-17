"""A synthetic footy data directory for the dataset-version tests (no real data, no videos).

Matches (``match_splits.json``): ``hmatch`` is held out, ``tmatch`` is in the training pool.
Clips of ``tmatch`` are pinned in ``clip_splits.json``: ``tmatch_seg001`` is in-match eval,
``tmatch_seg002`` / ``tmatch_seg003`` are train.

Local runs without the repo's heavy root conftest: ``pytest --noconftest -p tests.datasets.fixtures``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pytest

from footy_track import splits


@dataclass
class World:
    root: Path
    marks: Path
    db: Path
    ds_root: Path

    def sidecar(self, clip: str, lines: list[dict]) -> Path:
        p = self.marks / f"{clip}.jsonl"
        p.write_text("".join(json.dumps(ln) + "\n" for ln in lines))
        return p


def box(frame: int, label: str, prov: str = "labeller", x: float = 0.5, y: float = 0.5,
        extra: list[str] | None = None) -> dict:
    return {
        "frame_index": frame,
        "bbox": {"x": x, "y": y, "w": 0.02, "h": 0.03},
        "center": {"x": x + 0.01, "y": y + 0.015},
        "tags": [label, prov, *(extra or [])],
    }


def skip(frame: int, tag: str) -> dict:
    return {"frame_index": frame, "bbox": None, "center": None, "tags": [tag, "labeller"]}


def calibration_doc(clip: str, frames: dict[int, float]) -> dict:
    """GT file: frame -> x scale; H maps normalised image -> pitch metres (100u-50, 68v-34)."""
    out = {}
    for fi, dx in frames.items():
        pts = [(0.2, 0.3), (0.7, 0.4), (0.5, 0.8), (0.3, 0.6)]
        out[str(fi)] = {
            "frame_index": fi,
            "H": [100.0 + dx, 0, -50, 0, 68, -34, 0, 0, 1],
            "H_convention": "norm_image->pitch_m",
            "image_size": [1920, 1080],
            "correspondences": [
                {"image": [u, v], "pitch_id": f"P{i}", "pitch_xy": [(100.0 + dx) * u - 50, 68 * v - 34]}
                for i, (u, v) in enumerate(pts)
            ],
        }
    return {"clip": clip, "frames": out}


@pytest.fixture
def world(tmp_path, monkeypatch) -> World:
    root = tmp_path / "footy_data"
    marks = root / "ball_gt_marks"
    for sub in ("ball_checks", "calibration_gt", "frame_type", "broadcast_review"):
        (marks / sub).mkdir(parents=True)
    (root / "match_splits.json").write_text(
        json.dumps(
            {
                "hmatch": {"split": "heldout", "aliases": ["hmatch_full"], "reason": "t", "assigned": "2026-01-01T00:00:00+00:00"},
                "tmatch": {"split": "train", "aliases": ["tmatch_full"], "reason": "t", "assigned": "2026-01-01T00:00:00+00:00"},
            }
        )
    )
    (root / "clip_splits.json").write_text(
        json.dumps(
            {
                "version": 1,
                "rule": {"hash": "sha1(canonical_clip_id)", "modulus": 5, "eval_remainder": 0},
                "clips": {
                    "tmatch_seg001": {"split": "eval"},
                    "tmatch_seg002": {"split": "train"},
                    "tmatch_seg003": {"split": "train"},
                    "tmatch_seg004": {"split": "eval"},
                },
            }
        )
    )
    monkeypatch.setenv("FOOTY_DATA", str(root))
    monkeypatch.setenv("FOOTY_MATCH_SPLITS", str(root / "match_splits.json"))
    monkeypatch.setenv("FOOTY_CLIP_SPLITS", str(root / "clip_splits.json"))
    monkeypatch.setenv("FOOTY_FULL_MATCHES_MANIFEST", str(root / "no-manifest.jsonl"))
    monkeypatch.setenv("FOOTY_LABELLER_DATASETS", str(root / "no-registry.json"))
    monkeypatch.setenv("FOOTY_DATASETS_ROOT", str(root))
    monkeypatch.delenv("FOOTY_MATCH_SPLITS_ALLOW_SEED", raising=False)  # exercise the real registers
    monkeypatch.delenv("FOOTY_CLIP_SPLITS_PREVIEW", raising=False)
    splits.clear_cache()
    w = World(root, marks, root / "feature_store" / "fs.duckdb", root)
    populate(w)
    yield w
    splits.clear_cache()


def populate(w: World) -> None:
    """Tracking: 2 held-out clip frames, tmatch eval clip + two train clips. One calibration
    GT per split. Broadcast review of the held-out match's full video."""
    w.sidecar("hmatch_seg001", [box(0, "in_play_ball"), box(0, "player", x=0.2), box(1, "in_play_ball"), skip(2, "no_ball")])
    w.sidecar("tmatch_seg001", [box(0, "in_play_ball"), skip(1, "not_broadcast")])
    w.sidecar("tmatch_seg002", [box(0, "in_play_ball"), box(1, "in_play_ball", prov="vittrack", extra=["human_checked"]),
                                box(2, "in_play_ball", prov="vittrack")])
    w.sidecar("tmatch_seg003", [box(5, "player")])
    (w.marks / "ball_checks" / "tmatch_seg002.jsonl").write_text(
        json.dumps({"clip": "tmatch_seg002", "frame_index": 0, "box_index": 0, "source": "sidecar", "verdict": "ball",
                    "bbox": {"x": 0.5, "y": 0.5, "w": 0.02, "h": 0.03}, "label": "in_play_ball", "ts": 100.0}) + "\n"
    )
    for clip, frames in (("hmatch_seg001", {3: 0.0}), ("tmatch_seg001", {4: 0.0}), ("tmatch_seg002", {6: 0.0, 7: 5.0})):
        (w.marks / "calibration_gt" / f"{clip}.json").write_text(json.dumps(calibration_doc(clip, frames)))
    (w.marks / "frame_type" / "tmatch_seg003.json").write_text(
        json.dumps({"clip": "tmatch_seg003", "frames": {"9": {"type": "not_broadcast", "ts": "2026-11-14T19:15:00+0000"}}})
    )
    write_broadcast(w, "hmatch_full", {10: ("yes", "Yes", "edge"), 11: ("no", "Yes", "random"), 12: ("no", "No", "random"),
                                        13: ("yes", "No", "edge"), 14: ("unsure", "No", "edge")})


def write_broadcast(w: World, stem: str, frames: dict[int, tuple[str, str, str]]) -> None:
    doc = {"frames": {str(f): {"verdict": v, "pred": p, "p_yes": 0.9 if p == "Yes" else 0.1, "reason": r,
                               "ts_s": f / 25, "ts": "2026-11-17T09:00:00Z"} for f, (v, p, r) in frames.items()}}
    (w.marks / "broadcast_review" / f"{stem}.json").write_text(json.dumps(doc))
