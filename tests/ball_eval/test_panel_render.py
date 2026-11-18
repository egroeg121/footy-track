"""Keyframe-dense render of the panel's clips (``panel.render_clips``): same frames and names, atomic swap,
idempotence, alias handling and the symlink fallback. Uses tiny synthetic videos made with ffmpeg."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

from footy_track import match_splits as ms
from footy_track.ball_eval import panel as P
from footy_track.scripts import ball_eval_panel as cli

from .test_panel import HELD, REGISTER, write_segments

pytestmark = pytest.mark.skipif(
    not (shutil.which("ffmpeg") and shutil.which("ffprobe")), reason="needs ffmpeg"
)

N_FRAMES = 60  # long GOP in the source: one keyframe every 50 frames
SIZE = "64x48"


def make_video(
    path: Path, frames: int = N_FRAMES, fps: int = 25, gop: int = 50
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", f"testsrc=size={SIZE}:rate={fps}", "-frames:v", str(frames),
         "-c:v", "libx264", "-preset", "ultrafast", "-g", str(gop), "-keyint_min", str(gop), "-pix_fmt", "yuv420p", str(path)],
        check=True,
    )  # fmt: skip


def keyframes(path: Path) -> list[int]:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-skip_frame", "nokey", "-show_entries", "frame=pts", "-of", "csv=p=0", str(path)],
        capture_output=True, text=True, check=True,
    ).stdout  # fmt: skip
    return [int(x.rstrip(",")) for x in out.split()]


@pytest.fixture
def fd(tmp_path, monkeypatch):
    (tmp_path / "match_splits.json").write_text(json.dumps(REGISTER))
    monkeypatch.setenv("FOOTY_MATCH_SPLITS", str(tmp_path / "match_splits.json"))
    monkeypatch.setenv("FOOTY_DATA", str(tmp_path))
    monkeypatch.delenv("FOOTY_MATCH_SPLITS_ALLOW_SEED", raising=False)
    ms.clear_cache()
    write_segments(tmp_path, HELD)
    yield tmp_path
    ms.clear_cache()


@pytest.fixture
def panel(fd):
    doc = P.build_panel(
        HELD,
        root=fd,
        probe=None,
        gold_dirs=[],
        sha="abc",
        created_at="2026-01-01T00:00:00Z",
    )
    seg_dir = Path(doc["segments_json"]).parent
    clips = sorted(
        {w["clip"] for w in doc["windows"]} | {q["of"]["clip"] for q in doc["qa"]}
    )
    for c in clips:
        make_video(seg_dir / f"{c}.mp4")
    P.link_clips(doc, fd)
    return doc, seg_dir, P.clips_dir(doc["panel_version"], fd)


def test_render_makes_real_keyframe_dense_files_with_identical_frames(panel, fd):
    doc, seg_dir, cdir = panel
    r = P.render_clips(doc, fd)
    assert r["failed"] == [] and r["rendered"]
    for name in r["rendered"]:
        out, src = (
            cdir / f"{name}.mp4",
            seg_dir / f"{name.removesuffix(P.QA_SUFFIX)}.mp4",
        )
        assert out.is_file() and not out.is_symlink()
        assert P.probe_stream(out) == P.probe_stream(src)  # frames, size, fps
        assert P.probe_stream(out)["frames"] == N_FRAMES
        assert len(keyframes(src)) == 2  # source GOP 50
        assert (
            len(keyframes(out)) == N_FRAMES // P.KF_GOP
        )  # a keyframe every KF_GOP frames
    # the source is untouched
    assert all(not p.is_symlink() for p in seg_dir.glob("*.mp4"))
    man = json.loads((cdir / P.RENDER_MANIFEST).read_text())
    ent = man["clips"][f"{r['rendered'][0]}.mp4"]
    assert (
        man["settings"]["gop"] == P.KF_GOP
        and ent["frames"] == N_FRAMES
        and ent["source_size"] > 0
    )


def test_output_is_ios_safe(panel, fd):
    doc, _, cdir = panel
    r = P.render_clips(doc, fd)
    f = cdir / f"{r['rendered'][0]}.mp4"
    st = json.loads(subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "stream=codec_name,profile,pix_fmt,width,height", "-of", "json", str(f)],
        capture_output=True, text=True, check=True).stdout)["streams"][0]  # fmt: skip
    assert (
        st["codec_name"] == "h264"
        and st["pix_fmt"] == "yuv420p"
        and st["profile"] in {"Constrained Baseline", "Baseline", "Main", "High"}
    )
    assert st["width"] % 2 == 0 and st["height"] % 2 == 0
    head = f.read_bytes()[:4096]  # faststart: moov before mdat
    assert head.find(b"moov") != -1 and head.find(b"moov") < (
        head.find(b"mdat") if b"mdat" in head else 10**9
    )


def test_render_is_idempotent(panel, fd):
    doc, _, cdir = panel
    first = P.render_clips(doc, fd)
    mtimes = {
        p.name: p.stat().st_mtime_ns for p in cdir.glob("*.mp4") if not p.is_symlink()
    }
    second = P.render_clips(doc, fd)
    assert (
        second["rendered"] == []
        and sorted(second["skipped"]) == sorted(first["rendered"])
        and second["failed"] == []
    )
    assert {
        p.name: p.stat().st_mtime_ns for p in cdir.glob("*.mp4") if not p.is_symlink()
    } == mtimes
    # a lost manifest falls back to comparing frame counts against the source: still no re-encode
    (cdir / P.RENDER_MANIFEST).unlink()
    third = P.render_clips(doc, fd)
    assert third["rendered"] == [] and sorted(third["skipped"]) == sorted(
        first["rendered"]
    )
    assert (cdir / P.RENDER_MANIFEST).is_file()
    # --force re-renders
    assert P.render_clips(doc, fd, force=True)["rendered"]


def test_aliases_are_symlinks_to_the_rendered_sibling(panel, fd):
    doc, _, cdir = panel
    P.render_clips(doc, fd)
    aliases = {q["clip"] for q in doc["qa"]}
    assert aliases
    window_clips = {w["clip"] for w in doc["windows"]}
    for a in aliases:
        link = cdir / f"{a}.mp4"
        base = a.removesuffix(P.QA_SUFFIX)
        if (
            base in window_clips
        ):  # alias of a clip a window uses: relative link to the rendered sibling
            assert link.is_symlink() and os.readlink(link) == f"{base}.mp4"
            assert (cdir / f"{base}.mp4").is_file() and not (
                cdir / f"{base}.mp4"
            ).is_symlink()
        else:  # alias-only clip: the alias itself is the rendered file
            assert link.is_file()
        assert (
            P.probe_stream(link)["frames"] == N_FRAMES
        )  # resolves to a real video either way
    # link_clips (re-run by `job` on an existing panel) must not clobber the renders
    assert P.link_clips(doc, fd) == 0
    assert all((cdir / f"{a}.mp4").is_file() for a in aliases)


def test_failed_render_keeps_the_symlink_and_never_raises(panel, fd, caplog):
    doc, seg_dir, cdir = panel
    bad = doc["windows"][0]["clip"]
    (seg_dir / f"{bad}.mp4").write_bytes(b"not a video")  # corrupt: ffmpeg fails
    with caplog.at_level("WARNING"):
        r = P.render_clips(doc, fd)
    assert bad in r["failed"]
    link = cdir / f"{bad}.mp4"
    assert link.is_symlink() and link.resolve() == (seg_dir / f"{bad}.mp4").resolve()
    assert any(bad in rec.getMessage() for rec in caplog.records)
    for a in (
        q["clip"] for q in doc["qa"] if q["of"]["clip"] == bad
    ):  # its aliases keep pointing at the source too
        assert (cdir / f"{a}.mp4").is_symlink()
        assert os.readlink(cdir / f"{a}.mp4") == str(seg_dir / f"{bad}.mp4")
    assert (
        not list((cdir.parent / P.RENDER_TMP_DIRNAME).glob("*"))
        if (cdir.parent / P.RENDER_TMP_DIRNAME).exists()
        else True
    )
    others = list(r["rendered"])
    assert others and all(not (cdir / f"{c}.mp4").is_symlink() for c in others)


def test_frame_count_mismatch_keeps_the_symlink(panel, fd):
    doc, seg_dir, cdir = panel

    def lying(
        src, dst, tmp_dir, **kw
    ):  # an encoder that drops a frame: verification must catch it
        info = P.probe_stream(src)
        tmp = tmp_dir / "x.mp4"
        tmp_dir.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [
                "ffmpeg",
                "-v",
                "error",
                "-y",
                "-i",
                str(src),
                "-frames:v",
                str(info["frames"] - 1),
                "-c:v",
                "libx264",
                str(tmp),
            ],
            check=True,
        )
        out = P.probe_stream(tmp)
        tmp.unlink()
        if out != info:
            raise P.PanelError(f"{src.name}: re-encode differs from the source")
        return out

    r = P.render_clips(doc, fd, render=lying)
    assert r["rendered"] == [] and r["failed"]
    assert all(p.is_symlink() for p in cdir.glob("*.mp4"))
    # and the real verification inside render_clip rejects a differing output and leaves dst alone
    src = seg_dir / f"{doc['windows'][0]['clip']}.mp4"
    dst = cdir / src.name
    before = os.readlink(dst)
    real = P.probe_stream
    calls = []

    def fake_probe(path):
        calls.append(path)
        info = real(path)
        return info if len(calls) == 1 else {**info, "frames": info["frames"] - 1}

    P.probe_stream, orig = fake_probe, P.probe_stream
    try:
        with pytest.raises(P.PanelError, match="differs"):
            P.render_clip(src, dst, cdir.parent / P.RENDER_TMP_DIRNAME)
    finally:
        P.probe_stream = orig
    assert dst.is_symlink() and os.readlink(dst) == before
    assert not list((cdir.parent / P.RENDER_TMP_DIRNAME).glob("*.tmp.mp4"))


def test_cli_render_and_job_renders_but_never_fails_on_it(
    panel, fd, monkeypatch, capsys
):
    doc, seg_dir, cdir = panel
    P.write_panel(doc, root=fd)
    monkeypatch.setenv("FOOTY_LABELLER_DATASETS", str(fd / "labeller_datasets.json"))
    assert cli.main(["--root", str(fd), "render", "--match", HELD]) == 0
    out = json.loads(capsys.readouterr().out.splitlines()[-1])
    assert out["status"] == "ok" and out["rendered"] > 0 and out["failed"] == 0
    # default (no --match): every panel of the version; second run is all skips
    assert cli.main(["--root", str(fd), "render"]) == 0
    out = json.loads(capsys.readouterr().out.splitlines()[-1])
    assert out["rendered"] == 0 and out["skipped"] > 0

    # `job` calls render_clips; an exploding render must not fail it
    monkeypatch.setattr(
        P, "render_clips", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom"))
    )
    from footy_track.labeller import jobs_client  # noqa: PLC0415

    monkeypatch.setattr(
        jobs_client,
        "submit",
        lambda *a, **k: {"created": True, "id": "j-x", "url": "/j"},
    )
    assert (
        cli.main(
            [
                "--root",
                str(fd),
                "--panel-version",
                P.PANEL_VERSION,
                "job",
                "--match",
                HELD,
            ]
        )
        == 0
    )
    row = json.loads(capsys.readouterr().out.splitlines()[-1])
    assert row["status"] == "created" and "boom" in row["render"]["error"]


def test_job_renders_new_panels(fd, monkeypatch, capsys):
    doc = P.build_panel(
        HELD,
        root=fd,
        probe=None,
        gold_dirs=[],
        sha="abc",
        created_at="2026-01-01T00:00:00Z",
    )
    P.write_panel(doc, root=fd)
    seg_dir = Path(doc["segments_json"]).parent
    for c in {w["clip"] for w in doc["windows"]} | {q["of"]["clip"] for q in doc["qa"]}:
        make_video(seg_dir / f"{c}.mp4", frames=30)
    monkeypatch.setenv("FOOTY_LABELLER_DATASETS", str(fd / "labeller_datasets.json"))
    from footy_track.labeller import jobs_client  # noqa: PLC0415

    monkeypatch.setattr(
        jobs_client,
        "submit",
        lambda *a, **k: {"created": True, "id": "j-x", "url": "/j"},
    )
    assert cli.main(["--root", str(fd), "job", "--match", HELD]) == 0
    row = json.loads(capsys.readouterr().out.splitlines()[-1])
    assert row["render"]["rendered"] > 0 and row["render"]["failed"] == 0
    cdir = P.clips_dir(P.PANEL_VERSION, fd)
    assert any(p.is_file() and not p.is_symlink() for p in cdir.glob("*.mp4"))
    # job re-run (e.g. a retried sweep): idempotent, nothing re-rendered, nothing clobbered
    assert cli.main(["--root", str(fd), "job", "--match", HELD]) == 0
    row = json.loads(capsys.readouterr().out.splitlines()[-1])
    assert row["render"]["rendered"] == 0
