"""Tests for the match-rename migration (scripts/migrate_rename_match.py) on a tmp fixture tree."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import tarfile
from pathlib import Path

import pytest

from footy_track.scripts import migrate_rename_match as M

OLD_AGE = 3600 * 24  # seconds; far outside the 10 minute live window


def _write(
    root: Path, rel: str, content: str | bytes = "x", age: float = OLD_AGE
) -> Path:
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(content if isinstance(content, bytes) else content.encode())
    t = p.stat().st_mtime - age
    os.utime(p, (t, t))
    for d in p.parents:  # directory mtimes count for the live-write guard too
        if d == root.parent:
            break
        old = d.stat().st_mtime - OLD_AGE
        os.utime(d, (old, old))
    return p


def _jsonl(*rows) -> str:
    return "".join(json.dumps(r) + "\n" for r in rows)


@pytest.fixture
def tree(tmp_path: Path) -> Path:
    r = tmp_path / "footy_data"
    # the match footage
    _write(r, "arsenal_norwich/original_video/arsenal_norwich.mkv", b"\x00video")
    _write(r, "arsenal_norwich/proxy_1080p25/arsenal_norwich.mp4", b"\x00proxy")
    for n in (0, 3):
        _write(
            r,
            f"arsenal_norwich/split_video_broadcast_frames_kf10/arsenal_norwich_seg{n:03d}.mp4",
            b"\x00clip",
        )
    _write(
        r, "arsenal_norwich/full_video_frames/arsenal_norwich_000409.jpg", b"\xff\xd8"
    )
    _write(
        r,
        "arsenal_norwich/split_video_broadcast_frames/segments.json",
        json.dumps(
            {
                "source": "/x/arsenal_norwich/proxy/arsenal_norwich.mp4",
                "segments": ["arsenal_norwich_seg000"],
            }
        ),
    )
    # labels
    _write(
        r,
        "ball_gt_marks_norwich/arsenal_norwich_seg003.jsonl",
        _jsonl({"frame_index": 0, "tags": ["labeller"]}),
    )
    _write(
        r,
        "ball_gt_marks_norwich/ball_checks/arsenal_norwich_seg000.jsonl",
        _jsonl(
            {"clip": "arsenal_norwich_seg000", "frame_index": 39, "verdict": "ball"}
        ),
    )
    _write(
        r,
        "ball_candidates_norwich/arsenal_norwich_seg000.jsonl",
        _jsonl({"frame_index": 1}),
    )
    _write(
        r,
        "machine_labels_norwich/_manifest.json",
        json.dumps({"clips": {"arsenal_norwich_seg000": {"frames": 3}}}),
    )
    _write(
        r,
        "machine_labels_norwich/arsenal_norwich_seg000.jsonl",
        _jsonl({"frame_index": 0}),
    )
    _write(
        r,
        "norwich_label_queue/windows.csv",
        "clip,start_frame\narsenal_norwich_seg003,10\n",
    )
    _write(
        r,
        "norwich_label_queue/ball_check_list_v1.json",
        json.dumps({"items": [{"clip": "arsenal_norwich_seg072"}]}),
    )
    _write(
        r,
        "norwich_label_queue/q01_easy_arsenal_norwich_seg003_f0010-0259.png",
        b"\x89PNG",
    )
    # registry, jobs, tasks
    _write(
        r,
        "labeller_datasets.json",
        json.dumps(
            {
                "default": "calib",
                "datasets": {
                    "calib": {
                        "label": "Calibration",
                        "marks_dir": str(r / "ball_gt_marks"),
                    },
                    "norwich": {
                        "label": "Arsenal v Nottingham Forest (held-out; ids say 'norwich')",
                        "clips_dir": str(
                            r / "arsenal_norwich/split_video_broadcast_frames_kf10"
                        ),
                        "marks_dir": str(r / "ball_gt_marks_norwich"),
                        "ball_check_list": str(
                            r / "norwich_label_queue/ball_check_list_v1.json"
                        ),
                    },
                },
            },
            indent=2,
        ),
    )
    _write(
        r,
        "labelling_jobs/j-1.json",
        json.dumps(
            {
                "id": "j-1",
                "title": "Norwich ball placement r1",
                "dataset": "norwich",
                "key": "main/norwich-ball-place-r1",
                "why": "24 held-out Norwich windows",
                "items": [{"clip": "arsenal_norwich_seg003", "frame_start": 10}],
            }
        ),
    )
    _write(
        r,
        "labelling_tasks.json",
        json.dumps({"tasks": [{"id": "t", "title": "Recheck Man City"}]}),
    )
    # untouched stuff
    _write(r, "ball_gt_marks/arsenal_mancity_seg010.jsonl", _jsonl({"frame_index": 5}))
    _write(
        r,
        "labelling_jobs/j-2.json",
        json.dumps({"id": "j-2", "title": "Man City", "dataset": "calib"}),
    )
    # Roboflow exports keep their names (excluded by default)
    _write(
        r,
        "classifier_dataset/roboflow_dataset_9/train/Yes/arsenal_norwich_000009_jpg.rf.abc.jpg",
        b"\xff",
    )
    return r


def _snapshot(root: Path) -> dict[str, str]:
    out = {}
    for p in sorted(root.rglob("*")):
        if p.is_file() and "_backups" not in p.parts:
            out[p.relative_to(root).as_posix()] = hashlib.sha256(
                p.read_bytes()
            ).hexdigest()
        elif p.is_dir() and p.name != "_backups":
            out[p.relative_to(root).as_posix() + "/"] = ""
    return out


def _names_with_old(root: Path) -> list[str]:
    return sorted(
        p.relative_to(root).as_posix()
        for p in root.rglob("*")
        if "norwich" in p.name.lower() and "classifier_dataset" not in p.parts
    )


def _run(root: Path, *args: str, now: float | None = None, runner=None) -> int:
    return M.main(["--root", str(root), *args], now=now, runner=runner)


def test_rules_reject_non_idempotent_names():
    with pytest.raises(M.MigrationError):
        M.Rules(
            old_match="a_b", new_match="a_b_c", old_short="b", new_short="b"
        ).validate()
    M.Rules().validate()


def test_dry_run_is_default_and_writes_nothing(tree, capsys):
    before = _snapshot(tree)
    mtimes = {p: p.stat().st_mtime_ns for p in tree.rglob("*")}
    assert _run(tree) == 0
    out = capsys.readouterr().out
    assert "DRY RUN" in out and "directories to rename: 5" in out
    assert _snapshot(tree) == before
    assert {p: p.stat().st_mtime_ns for p in tree.rglob("*")} == mtimes
    assert not (tree / "_backups").exists()


def test_plan_counts(tree):
    rules = M.Rules()
    plan = M.build_plan(tree, rules)
    # dirs: arsenal_norwich, ball_gt_marks_norwich, ball_candidates_norwich, machine_labels_norwich, norwich_label_queue
    assert plan.n_dirs == 5
    # files: mkv, mp4 proxy, 2 clips, 1 jpg, 1 sidecar, 1 check, 1 candidate, 1 ml jsonl, 1 png = 10
    assert plan.n_files == 10
    rewritten = sorted(M._rel(r.path, tree) for r in plan.rewrites)
    assert rewritten == sorted(
        [
            "arsenal_norwich/split_video_broadcast_frames/segments.json",
            "ball_gt_marks_norwich/ball_checks/arsenal_norwich_seg000.jsonl",
            "machine_labels_norwich/_manifest.json",
            "norwich_label_queue/windows.csv",
            "norwich_label_queue/ball_check_list_v1.json",
            "labeller_datasets.json",
            "labelling_jobs/j-1.json",
        ]
    )
    assert plan.excluded["classifier_dataset"] == 1
    assert not plan.conflicts
    assert all(p.suffix in M.TEXT_SUFFIXES for p in plan.backup_files)


def test_apply_renames_and_rewrites(tree, capsys):
    assert _run(tree, "--apply") == 0
    assert _names_with_old(tree) == []
    nf = "arsenal_nottinghamforest"
    for rel in (
        f"{nf}/original_video/{nf}.mkv",
        f"{nf}/split_video_broadcast_frames_kf10/{nf}_seg003.mp4",
        f"ball_gt_marks_nottinghamforest/{nf}_seg003.jsonl",
        f"ball_gt_marks_nottinghamforest/ball_checks/{nf}_seg000.jsonl",
        f"ball_candidates_nottinghamforest/{nf}_seg000.jsonl",
        f"machine_labels_nottinghamforest/{nf}_seg000.jsonl",
        f"nottinghamforest_label_queue/q01_easy_{nf}_seg003_f0010-0259.png",
    ):
        assert (tree / rel).exists(), rel
    # contents
    row = json.loads(
        (
            tree / f"ball_gt_marks_nottinghamforest/ball_checks/{nf}_seg000.jsonl"
        ).read_text()
    )
    assert row["clip"] == f"{nf}_seg000"
    assert (tree / "nottinghamforest_label_queue/windows.csv").read_text().splitlines()[
        1
    ] == f"{nf}_seg003,10"
    reg = json.loads((tree / "labeller_datasets.json").read_text())
    assert set(reg["datasets"]) == {"calib", "nottinghamforest"}
    ds = reg["datasets"]["nottinghamforest"]
    assert ds["label"] == "Arsenal v Nottingham Forest (held-out)"
    assert ds["marks_dir"].endswith("/ball_gt_marks_nottinghamforest")
    assert ds["clips_dir"].endswith(f"/{nf}/split_video_broadcast_frames_kf10")
    job = json.loads((tree / "labelling_jobs/j-1.json").read_text())
    assert job["title"] == "Nottingham Forest ball placement r1"
    assert (
        job["dataset"] == "nottinghamforest"
        and job["key"] == "main/nottinghamforest-ball-place-r1"
    )
    assert job["items"][0]["clip"] == f"{nf}_seg003"
    assert job["why"] == "24 held-out Nottingham Forest windows"
    # the video bytes are untouched, unrelated files too, Roboflow names kept
    assert (tree / f"{nf}/original_video/{nf}.mkv").read_bytes() == b"\x00video"
    assert (
        json.loads((tree / "labelling_jobs/j-2.json").read_text())["title"]
        == "Man City"
    )
    assert (
        tree
        / "classifier_dataset/roboflow_dataset_9/train/Yes/arsenal_norwich_000009_jpg.rf.abc.jpg"
    ).exists()
    assert (
        "verify: re-planned after apply: 0 operation(s) left (OK)"
        in capsys.readouterr().out
    )
    # no temp files left behind
    assert not list(tree.rglob("*.migrate-tmp"))


def test_backup_and_manifest(tree):
    assert _run(tree, "--apply") == 0
    (tar_path,) = (tree / "_backups").glob("rename_*.tar.gz")
    (manifest_path,) = (tree / "_backups").glob("rename_*.manifest.json")
    assert tar_path.name.replace(".tar.gz", "") == manifest_path.name.replace(
        ".manifest.json", ""
    )
    with tarfile.open(tar_path) as tf:
        names = set(tf.getnames())
        reg = tf.extractfile("labeller_datasets.json").read().decode()
    assert "labeller_datasets.json" in names and "labelling_jobs/j-1.json" in names
    assert (
        "ball_gt_marks_norwich/arsenal_norwich_seg003.jsonl" in names
    )  # renamed-only text files are backed up too
    assert not any(n.endswith((".mkv", ".mp4", ".jpg", ".png")) for n in names)
    assert '"norwich"' in reg  # the *original* registry
    m = json.loads(manifest_path.read_text())
    assert m["status"] == "complete" and len(m["rewrites"]) == 7
    assert any(
        x["src"] == "arsenal_norwich" and x["dst"] == "arsenal_nottinghamforest"
        for x in m["renames"]
    )
    assert all(x["sha_before"] != x["sha_after"] for x in m["rewrites"])


def test_idempotent_rerun(tree, capsys):
    assert _run(tree, "--apply") == 0
    after_first = _snapshot(tree)
    n_backups = len(list((tree / "_backups").glob("*.tar.gz")))
    capsys.readouterr()
    assert _run(tree, "--apply", "--force", now=2e9) == 0
    out = capsys.readouterr().out
    assert "nothing to do" in out
    assert _snapshot(tree) == after_first
    assert (
        len(list((tree / "_backups").glob("*.tar.gz"))) == n_backups
    )  # no empty backup made
    assert _run(tree) == 0 and "PLAN EMPTY" in capsys.readouterr().out


def test_refuses_when_labeller_wrote_recently(tree, capsys):
    _write(
        tree,
        "ball_gt_marks_norwich/arsenal_norwich_seg003.jsonl",
        _jsonl({"frame_index": 9}),
        age=60,
    )  # 1 min ago
    before = _snapshot(tree)
    assert _run(tree, "--apply") == M.EXIT_LIVE
    assert "REFUSING --apply" in capsys.readouterr().err
    assert _snapshot(tree) == before and not (tree / "_backups").exists()
    # the dry run says so too
    assert _run(tree) == 0
    assert "WOULD REFUSE" in capsys.readouterr().out
    # ... and --force overrides
    assert _run(tree, "--apply", "--force") == 0
    assert _names_with_old(tree) == []


def test_live_window_is_configurable(tree):
    _write(
        tree, "ball_gt_marks_norwich/arsenal_norwich_seg003.jsonl", "{}\n", age=300
    )  # 5 min ago
    assert _run(tree, "--apply") == M.EXIT_LIVE
    assert _run(tree, "--apply", "--live-window-min", "2") == 0


def test_rollback_restores_tree_exactly(tree):
    before = _snapshot(tree)
    assert _run(tree, "--apply") == 0
    assert _snapshot(tree) != before
    (manifest,) = (tree / "_backups").glob("rename_*.manifest.json")
    assert _run(tree, "--rollback", str(manifest)) == 0
    assert _snapshot(tree) == before


def test_rollback_keeps_labels_written_after_migration(tree, capsys):
    assert _run(tree, "--apply") == 0
    (manifest,) = (tree / "_backups").glob("rename_*.manifest.json")
    check = (
        tree
        / "ball_gt_marks_nottinghamforest/ball_checks/arsenal_nottinghamforest_seg000.jsonl"
    )
    check.write_text(
        check.read_text()
        + _jsonl({"clip": "arsenal_nottinghamforest_seg000", "frame_index": 7})
    )
    t = check.stat().st_mtime - OLD_AGE
    os.utime(check, (t, t))
    capsys.readouterr()
    assert _run(tree, "--rollback", str(manifest)) == M.EXIT_ERROR  # skipped one file
    assert "changed since the migration" in capsys.readouterr().out
    # directories are back to the old names; the edited file kept its newer (renamed-id) content
    restored = tree / "ball_gt_marks_norwich/ball_checks/arsenal_norwich_seg000.jsonl"
    assert restored.exists() and "nottinghamforest" in restored.read_text()


def test_rollback_refuses_while_live(tree, capsys):
    assert _run(tree, "--apply") == 0
    (manifest,) = (tree / "_backups").glob("rename_*.manifest.json")
    _write(
        tree,
        "ball_gt_marks_nottinghamforest/arsenal_nottinghamforest_seg003.jsonl",
        "{}\n",
        age=5,
    )
    assert _run(tree, "--rollback", str(manifest)) == M.EXIT_LIVE
    assert "REFUSING rollback" in capsys.readouterr().out


def test_conflict_aborts_without_changes(tree, capsys):
    (
        tree / "ball_gt_marks_nottinghamforest"
    ).mkdir()  # a dir with the new name already exists
    before = _snapshot(tree)
    assert _run(tree, "--apply") == M.EXIT_CONFLICT
    assert "CONFLICT" in capsys.readouterr().out
    assert _snapshot(tree) == before


def test_crash_midway_then_rerun_finishes(tree, monkeypatch):
    real = os.rename
    calls = {"n": 0}

    def flaky(src, dst):
        calls["n"] += 1
        if calls["n"] == 4:
            raise OSError("disk fell over")
        return real(src, dst)

    monkeypatch.setattr(os, "rename", flaky)
    with pytest.raises(OSError):
        _run(tree, "--apply")
    monkeypatch.setattr(os, "rename", real)
    assert _names_with_old(tree) != []  # half done
    assert _run(tree, "--apply", "--force", now=2e9) == 0
    assert _names_with_old(tree) == []
    assert (
        json.loads((tree / "labelling_jobs/j-1.json").read_text())["dataset"]
        == "nottinghamforest"
    )


def test_invalid_json_result_is_refused(tmp_path):
    root = tmp_path / "d"
    _write(
        root, "x.json", '{"a": "norwich"'
    )  # already broken JSON: must not be "fixed" silently
    assert _run(root, "--apply") == M.EXIT_ERROR
    assert (root / "x.json").read_text() == '{"a": "norwich"'


def test_other_case_variants_warn(tmp_path, capsys):
    root = tmp_path / "d"
    _write(root, "n.txt", "NORWICH and norwich")
    assert _run(root) == 0
    assert "other-case variant" in capsys.readouterr().out


def test_symlinks_are_renamed_and_repointed(tmp_path):
    root = tmp_path / "d"
    _write(root, "arsenal_norwich/a.mp4", b"v")
    (root / "clips").mkdir()
    os.symlink(
        root / "arsenal_norwich/a.mp4", root / "clips/arsenal_norwich_seg001.mp4"
    )
    assert (
        _run(root, "--apply", "--force") == 0
    )  # (the fresh symlink's own mtime is "now")
    link = root / "clips/arsenal_nottinghamforest_seg001.mp4"
    assert (
        link.is_symlink()
        and link.resolve() == (root / "arsenal_nottinghamforest/a.mp4").resolve()
    )
    assert link.read_bytes() == b"v"


def test_generic_names(tmp_path):
    root = tmp_path / "d"
    _write(root, "arsenal_foo/arsenal_foo_seg001.mp4", b"v")
    _write(
        root,
        "ball_gt_marks_foo/arsenal_foo_seg001.jsonl",
        _jsonl({"clip": "arsenal_foo_seg001"}),
    )
    args = ["--old-match", "arsenal_foo", "--new-match", "arsenal_bar", "--old-short", "foo", "--new-short", "bar",
            "--old-title", "Foo", "--new-title", "Bar", "--no-default-literals"]  # fmt: skip
    assert _run(root, "--apply", *args) == 0
    assert (root / "arsenal_bar/arsenal_bar_seg001.mp4").exists()
    row = json.loads((root / "ball_gt_marks_bar/arsenal_bar_seg001.jsonl").read_text())
    assert row["clip"] == "arsenal_bar_seg001"


def test_extra_exclude(tree):
    assert (
        _run(
            tree,
            "--apply",
            "--exclude",
            "classifier_dataset",
            "--exclude",
            "machine_labels_norwich",
        )
        == 0
    )
    assert (tree / "machine_labels_norwich/arsenal_norwich_seg000.jsonl").exists()
    assert not (tree / "ball_gt_marks_norwich").exists()


# ---------------------------------------------------------------------------
# S3 (fake `aws`)
# ---------------------------------------------------------------------------


class FakeAws:
    """Just enough of `aws s3 ls/cp/rm` over a dict of {key: size}."""

    def __init__(self, bucket: str, objects: dict[str, int]):
        self.bucket, self.objects, self.calls = bucket, dict(objects), []

    def __call__(self, argv):
        self.calls.append(argv)
        cmd = argv[2]
        ok = lambda out="": subprocess.CompletedProcess(argv, 0, out, "")  # noqa: E731
        if cmd == "ls":
            prefix = argv[3].split(f"s3://{self.bucket}/", 1)[1]
            lines = [
                f"2026-11-14 00:00:00 {s:>10} {k}"
                for k, s in sorted(self.objects.items())
                if k.startswith(prefix)
            ]
            return ok("\n".join(lines) + "\n")
        if cmd == "cp":
            src, dst = argv[3], argv[4]
            key = dst.split(f"s3://{self.bucket}/", 1)[1]
            size = (
                self.objects[src.split(f"s3://{self.bucket}/", 1)[1]]
                if src.startswith("s3://")
                else os.path.getsize(src)
            )
            self.objects[key] = size
            return ok()
        if cmd == "rm":
            self.objects.pop(argv[3].split(f"s3://{self.bucket}/", 1)[1])
            return ok()
        raise AssertionError(argv)


@pytest.fixture
def aws():
    return FakeAws(
        "b",
        {
            "footy_data/arsenal_norwich/original_video/arsenal_norwich.mkv": 4_000,
            "footy_data/ball_gt_marks_norwich/ball_checks/arsenal_norwich_seg000.jsonl": 90,
            "footy_data/norwich_label_queue/windows.csv": 40,
            "footy_data/ball_gt_marks/arsenal_mancity_seg010.jsonl": 10,
            "footy_data/classifier_dataset/roboflow_dataset_9/train/Yes/arsenal_norwich_000009_jpg.rf.abc.jpg": 5,
        },
    )


def test_s3_plan_in_dry_run_changes_nothing(tree, aws, capsys):
    assert (
        _run(tree, "--include-s3", "--s3-prefix", "s3://b/footy_data/", runner=aws) == 0
    )
    out = capsys.readouterr().out
    assert "S3 s3://b/footy_data/: 5 keys listed" in out
    assert (
        "1 object(s) to copy server-side" in out and "2 label file(s) to upload" in out
    )
    assert all(c[2] == "ls" for c in aws.calls)
    assert len(aws.objects) == 5


def test_s3_apply_copies_then_deletes_only_on_request(tree, aws):
    args = ["--include-s3", "--s3-prefix", "s3://b/footy_data/"]
    assert _run(tree, "--apply", *args, runner=aws) == 0
    keys = set(aws.objects)
    assert (
        "footy_data/arsenal_nottinghamforest/original_video/arsenal_nottinghamforest.mkv"
        in keys
    )
    assert (
        "footy_data/ball_gt_marks_nottinghamforest/ball_checks/arsenal_nottinghamforest_seg000.jsonl"
        in keys
    )
    assert (
        "footy_data/arsenal_norwich/original_video/arsenal_norwich.mkv" in keys
    )  # old keys kept
    assert (
        "footy_data/classifier_dataset/roboflow_dataset_9/train/Yes/arsenal_norwich_000009_jpg.rf.abc.jpg"
        in keys
    )
    # label files came from the migrated local copy (new size), the video was a server-side copy
    uploaded = [c for c in aws.calls if c[2] == "cp" and not c[3].startswith("s3://")]
    assert len(uploaded) == 2 and all("nottinghamforest" in c[3] for c in uploaded)
    assert not [c for c in aws.calls if c[2] == "rm"]
    # a second run with --s3-delete-old (local already migrated) removes the old keys
    assert _run(tree, "--apply", *args, "--s3-delete-old", "--force", runner=aws) == 0
    assert not any("norwich" in k for k in aws.objects if "classifier_dataset" not in k)
    assert "footy_data/ball_gt_marks/arsenal_mancity_seg010.jsonl" in aws.objects
