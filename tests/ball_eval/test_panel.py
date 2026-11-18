"""Ball evaluation panel sampler (footy_track.ball_eval.panel): determinism, exact inclusion probabilities,
segment boundaries, the heldout gate, append-only storage, QA cards and the blind job body."""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path

import pytest

from footy_track import match_splits as ms
from footy_track import splits as sp
from footy_track.ball_eval import panel as P
from footy_track.feature_store import ingest_all
from footy_track.labeller import jobs
from footy_track.scripts import ball_eval_panel as cli

HELD = "hm_a"
HELD2 = "hm_b"
TRAIN = "tm_c"
REGISTER = {
    HELD: {"split": "heldout", "aliases": []},
    HELD2: {"split": "heldout", "aliases": []},
    TRAIN: {"split": "train", "aliases": []},
}


def write_segments(
    root: Path, match: str, fps: float = 25.0, counts=(400, 130, 1000, 90)
) -> Path:
    d = root / match / "split_video_broadcast_frames"
    d.mkdir(parents=True, exist_ok=True)
    segs, start = [], 100
    for i, n in enumerate(counts):
        segs.append(
            {
                "index": i,
                "file": f"{match}_seg{i:03d}.mp4",
                "start_frame": start,
                "end_frame": start + n - 1,
                "frame_count": n,
            }
        )
        start += n + 50
    (d / "segments.json").write_text(
        json.dumps({"fps": fps, "total_frames": start, "sample": 5, "segments": segs})
    )
    return d / "segments.json"


@pytest.fixture
def fd(tmp_path, monkeypatch):
    (tmp_path / "match_splits.json").write_text(json.dumps(REGISTER))
    monkeypatch.setenv("FOOTY_MATCH_SPLITS", str(tmp_path / "match_splits.json"))
    monkeypatch.setenv("FOOTY_DATA", str(tmp_path))
    monkeypatch.delenv("FOOTY_MATCH_SPLITS_ALLOW_SEED", raising=False)
    ms.clear_cache()
    for m in (HELD, HELD2, TRAIN):
        write_segments(tmp_path, m)
    yield tmp_path
    ms.clear_cache()


def build(fd, match=HELD, **kw):
    kw.setdefault("probe", None)
    kw.setdefault("gold_dirs", [])
    kw.setdefault("sha", "abc123")
    kw.setdefault("created_at", "2026-01-01T00:00:00Z")
    return P.build_panel(match, root=fd, **kw)


# ---- determinism and seed ---------------------------------------------------------------------
def test_seed_is_sha1_of_version_plus_match():
    want = int(hashlib.sha1(b"ball-eval-v1hm_a").hexdigest()[:15], 16)
    assert P.seed_for("ball-eval-v1", "hm_a") == want
    assert P.seed_for("ball-eval-v1", "hm_a") != P.seed_for("ball-eval-v1", "hm_b")
    assert P.seed_for("ball-eval-v2", "hm_a") != want


def test_build_is_deterministic_and_seeded_per_match(fd):
    a, b = build(fd), build(fd)
    assert P.core(a) == P.core(b)
    assert a["seed"] == P.seed_for("ball-eval-v1", HELD)
    other = build(fd, HELD2)
    assert [w["window_id"] for w in other["windows"]] != [
        w["window_id"].replace(HELD, HELD2) for w in a["windows"]
    ]
    # a different panel version is a different (independent) sample
    assert [
        w["window_id"] for w in build(fd, panel_version="ball-eval-v2")["windows"]
    ] != [w["window_id"] for w in a["windows"]]


# ---- population, pi, windows -------------------------------------------------------------------
@pytest.mark.parametrize(("fps", "gap"), [(25.0, 5), (50.0, 10), (60.0, 10), (30.0, 5)])
def test_window_gap_is_on_the_stride_grid_and_about_0_2s(fps, gap):
    assert P.window_gap(fps) == gap and gap % P.STRIDE == 0


@pytest.mark.parametrize("fps", [25.0, 50.0, 60.0])
def test_no_window_crosses_a_segment_and_windows_are_disjoint(fd, fps):
    counts = (400, 130, 1000, 90, 7)
    path = write_segments(fd, HELD, fps=fps, counts=counts)
    windows, info = P.enumerate_windows(HELD, P.load_segments(path))
    n_of = {f"{HELD}_seg{i:03d}": n for i, n in enumerate(counts)}
    gap = P.window_gap(fps)
    seen = set()
    for w in windows:
        assert len(w.frames) == P.WINDOW_FRAMES and w.step == gap
        assert all(b - a == gap for a, b in zip(w.frames, w.frames[1:], strict=False))
        assert (
            w.frames[0] >= 0 and w.frames[-1] <= n_of[w.clip] - 2
        )  # inside the clip, last frame never used
        assert all(f % P.STRIDE == 0 for f in w.frames)  # on the stride-5 grid
        for f in w.frames:
            assert (w.clip, f) not in seen  # a frame is in at most one window
            seen.add((w.clip, f))
    assert sum(s["n_windows"] for s in info["segments"]) == len(windows)
    assert {s["clip"]: s["n_windows"] for s in info["segments"]}[
        f"{HELD}_seg004"
    ] == 0  # too short for a window


def test_probe_caps_the_segment_length(fd):
    doc = P.load_segments(P.segments_path(HELD, fd))
    full, _ = P.enumerate_windows(HELD, doc)
    capped, _ = P.enumerate_windows(HELD, doc, {f"{HELD}_seg002": 100})
    assert len(capped) < len(full)
    assert max(w.frames[-1] for w in capped if w.clip == f"{HELD}_seg002") <= 98


def test_inclusion_probabilities_are_exact(fd):
    d = build(fd, n_windows=12)
    n, big_n = d["n_windows"], d["N_windows"]
    assert n == 12 and d["pi_window"] == n / big_n and d["pi_frame"] == d["pi_window"]
    assert all(
        w["pi_window"] == n / big_n and w["pi_frame"] == n / big_n for w in d["windows"]
    )
    assert len({w["window_id"] for w in d["windows"]}) == n  # without replacement
    assert (
        d["n_frames"] == n * P.WINDOW_FRAMES
        and d["N_frames"] == big_n * P.WINDOW_FRAMES
    )
    # HT identity: the sampled frames, each weighted 1/pi, add up to the population size
    assert d["n_frames"] / d["pi_frame"] == pytest.approx(d["N_frames"])
    assert d["design"]["joint_pi_window"] == pytest.approx(
        n * (n - 1) / (big_n * (big_n - 1))
    )


def test_draw_windows_includes_every_window_with_probability_n_over_N(fd):
    windows, _ = P.enumerate_windows(HELD, P.load_segments(P.segments_path(HELD, fd)))
    big_n, n, reps = len(windows), 10, 4000
    hits = [0] * big_n
    for s in range(reps):
        for w in P.draw_windows(windows, n, seed=s):
            hits[w.index] += 1
    freq = [h / reps for h in hits]
    assert sum(hits) == n * reps
    # every window has pi = n / N: 5 sigma of a binomial proportion
    sigma = (n / big_n * (1 - n / big_n) / reps) ** 0.5
    assert max(abs(f - n / big_n) for f in freq) < 5 * sigma


def test_n_windows_larger_than_population_takes_everything(fd):
    d = build(fd, n_windows=10**6)
    assert d["n_windows"] == d["N_windows"] and d["pi_window"] == 1.0


# ---- heldout gate -------------------------------------------------------------------------------
def test_refuses_train_unregistered_and_missing_register(fd, monkeypatch):
    with pytest.raises(P.PanelError, match="not a heldout match"):
        build(fd, TRAIN)
    write_segments(fd, "unknown_match")
    with pytest.raises(P.PanelError, match="not a heldout match"):
        build(fd, "unknown_match")
    assert not P.panel_path(TRAIN, root=fd).exists()
    (fd / "match_splits.json").unlink()
    ms.clear_cache()
    with pytest.raises(ms.RegisterError):  # fail closed, never guess
        build(fd)


def test_refuses_a_match_without_segments(fd):
    (fd / "nosegs").mkdir()
    reg = json.loads((fd / "match_splits.json").read_text())
    reg["nosegs"] = {"split": "heldout"}
    (fd / "match_splits.json").write_text(json.dumps(reg))
    ms.clear_cache()
    with pytest.raises(P.PanelError, match="no segments.json"):
        build(fd, "nosegs")


def test_segmented_heldout_matches_lists_only_heldout_with_segments(fd):
    assert P.segmented_heldout_matches(fd) == [HELD, HELD2]


def test_job_path_regates_via_cli(fd, capsys):
    assert cli.main(["--root", str(fd), "build", "--match", TRAIN, "--no-probe"]) == 1
    assert json.loads(capsys.readouterr().out.splitlines()[-1])["status"] == "error"


# ---- storage: append-only, idempotent -------------------------------------------------------------
def test_write_is_atomic_idempotent_and_refuses_changes(fd):
    d = build(fd)
    path, status = P.write_panel(d, root=fd)
    assert status == "created" and path == P.panel_path(HELD, root=fd)
    before = path.read_bytes()
    assert (
        P.write_panel(build(fd, created_at="2030-01-01T00:00:00Z", sha="zzz"), root=fd)[
            1
        ]
        == "unchanged"
    )
    assert (
        path.read_bytes() == before
    )  # a no-op build does not touch the file (stamps included)
    with pytest.raises(P.PanelError, match="append-only"):
        P.write_panel(build(fd, n_windows=7), root=fd)
    assert path.read_bytes() == before
    assert not list(path.parent.glob("*.tmp"))
    # --force replaces it and keeps the old one
    p2, status = P.write_panel(build(fd, n_windows=7), root=fd, force=True)
    assert status == "replaced" and P.read_panel(p2)["n_windows"] == 7
    assert [
        json.loads(f.read_text())["n_windows"]
        for f in path.parent.glob("*.json.prev-*")
    ] == [24]


def test_changed_segmentation_is_refused_not_silently_resampled(fd):
    P.write_panel(build(fd), root=fd)
    write_segments(fd, HELD, counts=(400, 130, 1000, 90, 500))  # the segmenter re-ran
    with pytest.raises(P.PanelError, match="append-only"):
        P.write_panel(build(fd), root=fd)


def test_cli_build_twice_is_idempotent(fd, capsys):
    args = ["--root", str(fd), "build", "--match", HELD, "--no-probe"]
    assert cli.main(args) == 0 and cli.main(args) == 0
    out = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert [o["status"] for o in out] == ["created", "unchanged"]
    assert cli.main([*args, "--n-windows", "5"]) == 1  # would change the file


# ---- QA cards ---------------------------------------------------------------------------------
def _gold_dir(tmp_path: Path, rows: dict[str, list[dict]], name="gt") -> Path:
    d = tmp_path / name
    d.mkdir()
    for clip, rs in rows.items():
        (d / f"{clip}.jsonl").write_text("\n".join(json.dumps(r) for r in rs) + "\n")
    return d


def _mark(frame, tags, x=0.4, y=0.5):
    return {
        "frame_index": frame,
        "bbox": {"x": x, "y": y, "w": 0.01, "h": 0.01},
        "center": {"x": x, "y": y},
        "tags": tags,
    }


def test_repeats_are_5_percent_of_panel_frames_on_alias_clips_at_the_end(fd):
    d = build(fd)
    reps = [q for q in d["qa"] if q["qa"] == "repeat"]
    assert len(reps) == round(P.QA_REPEAT_FRAC * d["n_frames"]) == 6
    panel_frames = {(w["clip"], f) for w in d["windows"] for f in w["frames"]}
    assert len({(q["clip"], q["frame"]) for q in reps}) == 6  # distinct
    for q in reps:
        assert q["clip"] == q["of"]["clip"] + P.QA_SUFFIX
        assert (q["of"]["clip"], q["of"]["frame"]) in panel_frames
    items = P.job_items(d)
    assert [i["sampling"]["qa"] for i in items[: d["n_windows"]]] == [None] * d[
        "n_windows"
    ]  # windows first
    assert all(
        i["sampling"]["qa"] in ("repeat", "gold") for i in items[d["n_windows"] :]
    )  # QA last
    assert len(d["qa"]) == len(items) - d["n_windows"]


def test_gold_comes_from_tier1_labeller_marks_of_this_match_only(fd, tmp_path):
    clip = f"{HELD}_seg002"
    gt = _gold_dir(
        tmp_path,
        {
            clip: [
                _mark(f, ["in_play_ball", "labeller"], x=0.1 + f / 5000)
                for f in range(0, 900, 7)
            ]
            + [
                _mark(3, ["in_play_ball", "vittrack"]),
                _mark(4, ["in_play_ball", "vittrack", "human_checked"]),
                {"frame_index": 5, "bbox": None, "center": None, "tags": ["no_ball"]},
            ],
            f"{TRAIN}_seg000": [
                _mark(10, ["in_play_ball", "labeller"])
            ],  # another match: ignored
        },
    )
    d = build(fd, gold_dirs=[gt])
    gold = [q for q in d["qa"] if q["qa"] == "gold"]
    assert (
        len(gold) == round(P.QA_GOLD_FRAC * d["n_frames"]) == 4 and d["n_qa_gold"] == 4
    )
    for q in gold:
        assert q["clip"] == clip + P.QA_SUFFIX and q["of"]["clip"] == clip
        assert q["frame"] % 7 == 0 and q["frame"] not in (
            3,
            4,
            5,
        )  # tier 1 positions only
        assert 0 < q["gold"]["x"] < 1 and q["gold"]["unit"] == "normalised"
    # no overlap between a gold card and a repeat card on the same alias frame
    keys = [(q["clip"], q["frame"]) for q in d["qa"]]
    assert len(keys) == len(set(keys))
    assert P.core(build(fd, gold_dirs=[gt])) == P.core(d)  # deterministic


def test_no_gold_when_the_match_has_no_tier1_marks(fd, tmp_path):
    gt = _gold_dir(
        tmp_path, {f"{HELD}_seg002": [_mark(8, ["in_play_ball", "vittrack"])]}
    )
    assert build(fd, gold_dirs=[gt])["n_qa_gold"] == 0


# ---- blind job ----------------------------------------------------------------------------------
ALLOWED_ITEM_KEYS = {
    "clip",
    "frame_start",
    "frame_end",
    "step",
    "frame",
    "stratum",
    "sampling",
}


def test_job_is_blind_idempotent_keyed_and_carries_sampling(fd):
    d = build(fd)
    body = P.job_body(d)
    assert (
        body["key"] == f"ball-eval-panel/ball-eval-v1/{HELD}"
        and body["kind"] == "ball_place"
    )
    assert body["dataset"] == "ball_eval_v1"
    for it in body["items"]:
        assert (
            set(it) <= ALLOWED_ITEM_KEYS
        )  # no box / candidate / score / reason can reach the labeller
    win = body["items"][0]
    s = win["sampling"]
    assert (
        s["regime"] == "eval"
        and s["panel"] == "ball-eval-v1"
        and s["match"] == HELD
        and s["stratum"] == HELD
    )
    assert (
        s["window_id"] == d["windows"][0]["window_id"]
        and s["pi_window"] == s["pi_frame"] == d["pi_window"]
    )
    assert s["seed"] == d["seed"] and s["qa"] is None and "code_sha" in s
    # the job API accepts it, expands windows to the 5 frames, and the key makes a re-submit a no-op
    items = jobs._normalise_items("ball_place", body["items"])
    assert jobs._item_units("ball_place", items[0]) == d["windows"][0]["frames"]
    assert sum(len(jobs._item_units("ball_place", i)) for i in items) == d[
        "n_frames"
    ] + len(d["qa"])
    assert P.job_body(d)["key"] == body["key"]
    # gold truth never leaves the panel file
    assert "gold" not in json.dumps(body)


def test_clip_links_and_dataset_registration_are_idempotent(fd, tmp_path, monkeypatch):
    d = build(fd)
    seg_dir = Path(d["segments_json"]).parent
    for w in d["windows"]:
        (seg_dir / f"{w['clip']}.mp4").write_bytes(b"x")
    monkeypatch.setenv("FOOTY_LABELLER_DATASETS", str(fd / "labeller_datasets.json"))
    n = P.link_clips(d, fd)
    assert n > 0 and P.link_clips(d, fd) == 0
    qa_alias = (
        fd
        / "ball_eval_panel"
        / "ball-eval-v1"
        / "clips"
        / (d["qa"][0]["clip"] + ".mp4")
    )
    assert (
        qa_alias.is_symlink()
        and qa_alias.resolve()
        == (seg_dir / f"{d['qa'][0]['of']['clip']}.mp4").resolve()
    )
    assert P.register_dataset(root=fd) is True and P.register_dataset(root=fd) is False
    spec = json.loads((fd / "labeller_datasets.json").read_text())["datasets"][
        "ball_eval_v1"
    ]
    assert (
        spec["marks_dir"].endswith("ball_gt_marks_ball_eval_v1")
        and spec["ingest"] is False
    )


def test_ingest_all_skips_registry_entries_marked_ingest_false(fd, monkeypatch):
    gt, panel_marks = fd / "gt", fd / "ball_gt_marks_ball_eval_v1"
    gt.mkdir()
    panel_marks.mkdir()
    (fd / "labeller_datasets.json").write_text(
        json.dumps(
            {
                "default": "calib",
                "datasets": {
                    "calib": {"clips_dir": str(fd / "c"), "marks_dir": str(gt)},
                    "ball_eval_v1": {
                        "clips_dir": str(fd / "c2"),
                        "marks_dir": str(panel_marks),
                        "ingest": False,
                    },
                },
            }
        )
    )
    monkeypatch.setenv("FOOTY_LABELLER_DATASETS", str(fd / "labeller_datasets.json"))
    marks, _ = ingest_all.discover_sources(fd)
    assert marks == [gt.resolve()]


def test_item_sampling_must_be_an_object(fd):
    """A job item keeps its ``sampling`` object; anything else is refused (the echo onto label rows is tested in test_jobs)."""
    items = jobs._normalise_items("ball_place", P.job_items(build(fd))[:1])
    assert items[0]["sampling"]["pi_frame"] > 0
    with pytest.raises(jobs.JobError, match="sampling must be an object"):
        jobs._normalise_items(
            "ball_place", [{"clip": "c", "frame": 1, "sampling": "x"}]
        )


def test_random_sanity(
    fd,
):  # the sampler must not touch the global RNG state (reproducible under any caller)
    state = random.getstate()
    build(fd)
    assert random.getstate() == state
    assert sp.EVAL_HELDOUT == "eval_heldout_match"
