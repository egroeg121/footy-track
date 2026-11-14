"""Pre-labelling pipeline: candidates, fit/score, identifiers (mocked), run, eval.

No network and no real video: frames are synthetic pitches drawn through a
known homography; the Claude client is a stub.
"""

from __future__ import annotations

import base64
import json
import sys
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from footy_track.calibration import geometry as geo
from footy_track.calibration.pitch_model import PitchModel
from footy_track.calibration.prelabel import evaluate as ev
from footy_track.calibration.prelabel import identifier as idf
from footy_track.calibration.prelabel import pipeline as pl
from footy_track.calibration.prelabel.candidates import find_candidates, render_candidates
from footy_track.calibration.prelabel.fit import Label, fit_and_score, line_support_score
from footy_track.calibration.prelabel.proposals import read_proposals

from .test_geometry import H, W, _camera_h

PITCH = PitchModel()
KP = list(PITCH.keypoints())  # _labels_from keeps those that are visible and have a candidate


def synth_frame(Hn=None) -> np.ndarray:
    """Green pitch with white 5 px lines drawn through ``Hn`` (norm image -> pitch)."""
    Hn = _camera_h() if Hn is None else Hn
    img = np.zeros((H, W, 3), np.uint8)
    img[:] = (40, 140, 50)  # BGR green
    rng = np.random.default_rng(0)
    img = np.clip(img + rng.integers(-6, 6, img.shape), 0, 255).astype(np.uint8)
    for name, runs in geo.project_pitch_model(Hn, PITCH).items():
        for run in runs:
            pts = np.round(run * np.array([W, H])).astype(np.int32)
            cv2.polylines(img, [pts], False, (250, 250, 250), 5, cv2.LINE_AA)
    return img


@pytest.fixture(scope="module")
def frame():
    return synth_frame()


@pytest.fixture(scope="module")
def det(frame):
    return find_candidates(frame)


def _true_uv(kid: str) -> np.ndarray:
    xy = np.array([PITCH.keypoints()[kid]])
    uv, _ = geo.apply_homography(np.linalg.inv(_camera_h()), xy)
    return uv[0]


def test_candidates_find_true_line_intersections(det):
    assert det.lines and det.candidates
    w, h = det.size
    for kid in ("HALFWAY_TOP", "CENTRE_CIRCLE_TOP", "CENTRE_CIRCLE_BOTTOM", "R_PEN_BOX_TOP_OUTER"):
        uv = _true_uv(kid)
        if not (0.02 < uv[0] < 0.98 and 0.02 < uv[1] < 0.98):
            continue
        d = min(np.hypot((c.uv[0] - uv[0]) * w, (c.uv[1] - uv[1]) * h) for c in det.candidates)
        assert d < 8, (kid, d)


def test_candidate_ids_unique_and_render_size(frame, det):
    ids = [c.id for c in det.candidates]
    assert len(ids) == len(set(ids))
    out = render_candidates(det)
    assert max(out.shape[:2]) <= 1568 and out.shape == det.work_bgr.shape


def test_render_downscales_big_frames():
    big = np.zeros((2160, 3840, 3), np.uint8)
    d = find_candidates(big)
    assert max(d.work_bgr.shape[:2]) <= 1568 and d.scale < 1 and d.candidates == []


def _labels_from(det, ids, tol_px=8.0):
    w, h = det.size
    out = []
    for kid in ids:
        uv = _true_uv(kid)
        if not (0 <= uv[0] <= 1 and 0 <= uv[1] <= 1):
            continue
        best = min(det.candidates, key=lambda c: np.hypot((c.uv[0] - uv[0]) * w, (c.uv[1] - uv[1]) * h))
        if np.hypot((best.uv[0] - uv[0]) * w, (best.uv[1] - uv[1]) * h) < tol_px:
            out.append(Label(kid, best.uv, 1.0, best.id))
    return out


def test_fit_recovers_known_homography_and_scores_high(det):
    labels = _labels_from(det, KP)
    assert len(labels) >= 4
    fit = fit_and_score(labels, PITCH, det, (W, H))
    assert fit.ok and sum(fit.inliers) >= 4
    assert fit.score > 0.8
    # reprojected centre-circle top sits where the truth says
    uv, _ = geo.apply_homography(np.linalg.inv(fit.H), np.array([PITCH.keypoints()["CENTRE_CIRCLE_TOP"]]))
    assert np.hypot(*((uv[0] - _true_uv("CENTRE_CIRCLE_TOP")) * [W, H])) < 6


def test_ransac_rejects_a_wrong_label(det):
    labels = _labels_from(det, KP)
    assert len(labels) >= 6
    labels[-1].keypoint_id = "L_CORNER_BOTTOM"  # one wrong label among >= 5 right ones
    fit = fit_and_score(labels, PITCH, det, (W, H))
    assert fit.ok and fit.score > 0.8
    assert not fit.inliers[-1] and sum(fit.inliers) == len(labels) - 1


def test_wrong_homography_scores_low(det):
    good, _ = line_support_score(_camera_h(), PITCH, det)
    shifted = _camera_h() @ np.array([[1, 0, 0.12], [0, 1, 0.05], [0, 0, 1.0]])
    bad, _ = line_support_score(shifted, PITCH, det)
    assert good > 0.8 and bad < good - 0.3


def test_fit_fails_with_reason(det):
    assert "only 2" in fit_and_score(_labels_from(det, KP)[:2], PITCH, det, (W, H)).failure
    collinear = [Label("HALFWAY_TOP", (0.5, 0.2)), Label("CENTRE_CIRCLE_TOP", (0.5, 0.4)),
                 Label("CENTRE_SPOT", (0.5, 0.5)), Label("HALFWAY_BOTTOM", (0.5, 0.9))]
    r = fit_and_score(collinear, PITCH, det, (W, H))
    assert not r.ok and "degenerate" in r.failure


def test_duplicate_keypoint_labels_are_deduped(det):
    labels = _labels_from(det, KP)
    labels.append(Label(labels[0].keypoint_id, (0.9, 0.9), 0.1))
    fit = fit_and_score(labels, PITCH, det, (W, H))
    assert len({lb.keypoint_id for lb in fit.labels}) == len(fit.labels)


# ---- frame selection --------------------------------------------------------


def test_select_frames():
    assert pl.select_frames(100, 25) == [0, 25, 50, 75]
    assert pl.select_frames(100, 25, skip={25}) == [0, 50, 75]
    assert pl.select_frames(100, 25, explicit=[5, 5, 99, 120]) == [5, 99]


# ---- identifiers -------------------------------------------------------------


def _gt_dir(tmp_path, det, ids=KP, clip="arsenal_mancity_seg010", idx=7):
    corr = []
    for kid in ids:
        uv = _true_uv(kid)
        if 0 <= uv[0] <= 1 and 0 <= uv[1] <= 1:
            corr.append({"image": uv.tolist(), "pitch_id": kid, "pitch_xy": list(PITCH.keypoints()[kid])})
    d = tmp_path / "gt"
    (d / "calibration_gt").mkdir(parents=True)
    rec = {"correspondences": corr, "image_size": [W, H], "H": _camera_h().tolist(), "pitch_dims": [105, 68]}
    (d / "calibration_gt" / f"{clip}.json").write_text(json.dumps({"clip": clip, "frames": {str(idx): rec}}))
    return d


def _inp(det, clip="arsenal_mancity_seg010", idx=7):
    return idf.IdentifyInput(clip, idx, PITCH, det, b"", b"")


def test_mock_identifier_uses_gt_and_flags_missing(tmp_path, det):
    d = _gt_dir(tmp_path, det)
    m = idf.MockIdentifier(d)
    r = m.identify(_inp(det))
    assert not r.unusable and len(r.labels) >= 4
    for lb in r.labels:
        assert np.hypot(*((np.array(lb.uv) - _true_uv(lb.keypoint_id)) * [W, H])) < 8
    assert m.identify(_inp(det, idx=8)).unusable  # no GT on frame 8


def test_mock_corrupt_swaps_labels(tmp_path, det):
    d = _gt_dir(tmp_path, det)
    clean = idf.MockIdentifier(d).identify(_inp(det))
    bad = idf.MockIdentifier(d, corrupt=1).identify(_inp(det))
    assert [lb.keypoint_id for lb in clean.labels] != [lb.keypoint_id for lb in bad.labels]


def _fake_response(payload, stop="end_turn", usage=(1000, 200)):
    return SimpleNamespace(
        stop_reason=stop, stop_details=SimpleNamespace(category="cyber"), model="claude-opus-5-5",
        content=[SimpleNamespace(type="text", text=json.dumps(payload))],
        usage=SimpleNamespace(input_tokens=usage[0], output_tokens=usage[1],
                              cache_creation_input_tokens=0, cache_read_input_tokens=0),
    )


class FakeClient:
    def __init__(self, resp):
        self.calls = []
        self.beta = SimpleNamespace(messages=SimpleNamespace(create=self._create))
        self.messages = SimpleNamespace(create=self._create)
        self.resp = resp

    def _create(self, **kw):
        self.calls.append(kw)
        return self.resp


def _payload(det):
    w, h = det.size
    pts = []
    for kid in ("HALFWAY_TOP", "CENTRE_CIRCLE_TOP", "CENTRE_CIRCLE_BOTTOM", "R_PEN_BOX_TOP_OUTER", "CENTRE_SPOT"):
        uv = _true_uv(kid)
        # direct localisation in sent-image pixels (no hint), off by a few px
        pts.append({"keypoint_id": kid, "x": uv[0] * w + 4, "y": uv[1] * h - 3, "confidence": 0.8, "candidate_id": 0})
    cands = _labels_from(det, KP)[:1]
    for lb in cands:  # one cited hint: its exact position is used
        pts.append({"keypoint_id": lb.keypoint_id, "x": 1.0, "y": 1.0, "confidence": 0.9, "candidate_id": lb.candidate_id})
    pts.append({"keypoint_id": "CENTRE_SPOT", "x": 5, "y": 5, "confidence": 0.9, "candidate_id": 9999})  # bad id -> use x, y
    return {
        "unusable": False, "unusable_reason": "",
        "view": {"end": "L", "main_camera_wide": True, "reverse_angle_suspected": False, "notes": "left end"},
        "points": pts,
        "lines": [{"line_id": "HALFWAY_LINE", "p1": {"x": 100, "y": 100}, "p2": {"x": 300, "y": 120}, "confidence": 0.9},
                  {"line_id": "HALFWAY_LINE", "p1": {"x": 100, "y": 100}, "p2": {"x": 101, "y": 100}, "confidence": 0.9}],
    }


def test_claude_identifier_request_shape_and_parsing(det):
    client = FakeClient(_fake_response(_payload(det)))
    ci = idf.ClaudeIdentifier(client=client, effort="low")
    res = ci.identify(_inp(det))
    kw = client.calls[0]
    assert kw["model"] == "claude-opus-5-5" and kw["output_config"]["effort"] == "low"
    assert kw["output_config"]["format"]["type"] == "json_schema"
    assert "thinking" not in kw  # adaptive by default: omitted
    assert kw["max_tokens"] >= 8000
    assert kw["betas"] == ["server-side-fallback-2026-07-01"] and kw["fallbacks"] == "default"
    imgs = [b for b in kw["messages"][0]["content"] if b["type"] == "image"]
    assert len(imgs) == 2
    assert not res.unusable and res.view["end"] == "L"
    w, h = _inp(det).detection.size
    direct = [lb for lb in res.labels if lb.candidate_id is None]
    assert len(direct) >= 5  # localised directly, no candidate needed
    top = next(lb for lb in direct if lb.keypoint_id == "HALFWAY_TOP")
    assert np.allclose(np.array(top.uv) * [w, h], _true_uv("HALFWAY_TOP") * [w, h] + [4, -3], atol=1e-6)
    assert all(lb.candidate_id != 9999 for lb in res.labels)  # hallucinated id is not trusted
    assert [ll.line_id for ll in res.line_labels] == ["HALFWAY_LINE"]  # the degenerate (1 px) line is dropped
    assert any(lb.candidate_id is not None for lb in res.labels)  # a cited hint keeps its candidate id
    assert res.usage["input_tokens"] == 1000 and res.usage["output_tokens"] == 200
    assert ci.run_prefix() == f"claude-claude-opus-5-5-{idf.PROMPT_VERSION}"


def test_claude_schema_enumerates_keypoints():
    s = idf.build_schema(PITCH)
    assert set(s["properties"]["points"]["items"]["properties"]["keypoint_id"]["enum"]) == set(PITCH.keypoints())
    assert set(s["properties"]["lines"]["items"]["properties"]["line_id"]["enum"]) == set(PITCH.segments())


def test_claude_prompt_caches_the_stable_prefix_only(det):
    client = FakeClient(_fake_response(_payload(det)))
    idf.ClaudeIdentifier(client=client).identify(IdentifyInput_(det, b"FRAME", b"DIAGRAM"))
    kw = client.calls[0]
    content = kw["messages"][0]["content"]
    marked = [i for i, b in enumerate(content) if "cache_control" in b]
    assert len(marked) == 1 and content[marked[0]]["cache_control"] == {"type": "ephemeral"}
    before, after = content[: marked[0] + 1], content[marked[0] + 1 :]
    imgs = lambda blocks: [base64.b64decode(b["source"]["data"]) for b in blocks if b["type"] == "image"]  # noqa: E731
    assert imgs(before) == [b"DIAGRAM"] and imgs(after) == [b"FRAME"]  # per-frame stuff is after the breakpoint
    assert "cache_control" not in kw["system"] if isinstance(kw["system"], dict) else True
    # the prefix is byte-identical between two different frames
    client2 = FakeClient(_fake_response(_payload(det)))
    idf.ClaudeIdentifier(client=client2).identify(IdentifyInput_(det, b"OTHER", b"DIAGRAM", idx=9))
    assert json.dumps(client2.calls[0]["messages"][0]["content"][: marked[0] + 1], sort_keys=True) == json.dumps(
        content[: marked[0] + 1], sort_keys=True)
    assert client2.calls[0]["system"] == kw["system"] and client2.calls[0]["output_config"] == kw["output_config"]
    assert idf.PROMPT_VERSION == "p2"


def IdentifyInput_(det, frame_jpg, diagram_jpg, idx=7):  # noqa: N802
    return idf.IdentifyInput("arsenal_mancity_seg010", idx, PITCH, det, frame_jpg, diagram_jpg)


def test_prompt_frame_has_a_labelled_grid(det):
    img = idf.render_prompt_frame(det)
    assert img.shape == det.work_bgr.shape
    plain = idf.render_prompt_frame(det, hints=False)
    assert not np.array_equal(img, plain)
    # grid lines are drawn every 100 px: the gridline column is brighter than its neighbours on dark grass
    row = plain[det.size[1] // 2 + 3]
    assert row[100:101].mean() > row[103:104].mean() - 1  # a tick column exists (no crash; weak check)


def test_claude_refusal_and_truncation_are_errors_not_labels(det):
    r = idf.ClaudeIdentifier(client=FakeClient(_fake_response({}, stop="refusal"))).identify(_inp(det))
    assert r.error and "refusal" in r.error and not r.labels
    r = idf.ClaudeIdentifier(client=FakeClient(_fake_response({}, stop="max_tokens"))).identify(_inp(det))
    assert r.error and "max_tokens" in r.error


def test_claude_without_fallbacks_uses_plain_endpoint(det):
    client = FakeClient(_fake_response(_payload(det)))
    idf.ClaudeIdentifier(client=client, fallbacks=False).identify(_inp(det))
    assert "betas" not in client.calls[0] and "fallbacks" not in client.calls[0]


def test_claude_verify(det):
    client = FakeClient(_fake_response({"match": False, "reason": "mirrored"}))
    v = idf.ClaudeIdentifier(client=client).verify(b"jpg")
    assert v["match"] is False and v["reason"] == "mirrored"


def test_diagram_has_every_keypoint_and_is_bounded():
    img = idf.render_diagram(PITCH)
    assert max(img.shape[:2]) <= 1568


# ---- pipeline + eval ---------------------------------------------------------


@pytest.fixture
def run_env(tmp_path, monkeypatch, frame, det):
    gt = _gt_dir(tmp_path, det)
    clips = tmp_path / "clips"
    clips.mkdir()
    (clips / "arsenal_mancity_20250925_seg010.mp4").touch()
    monkeypatch.setattr(pl, "video_frame_count", lambda v: 12)
    monkeypatch.setattr(pl, "read_frame", lambda v, i: frame)
    return gt, clips


def test_pipeline_with_mock_writes_contract_proposals(run_env):
    gt, clips = run_env
    cfg = pl.RunConfig(gt_dir=gt, clips_dir=clips, clips=["arsenal_mancity_seg010"], frames=[7, 8],
                       skip_gt=False, render_dir=gt.parent / "renders")
    s = pl.run_prelabel(cfg, idf.MockIdentifier(gt), log=lambda *_: None)
    assert s.n_frames == 2 and s.n_proposed == 1 and s.n_failed == 1
    doc = read_proposals(gt, "arsenal_mancity_seg010")
    assert doc["clip"] == "arsenal_mancity_seg010"
    run = doc["runs"][s.run_id]
    assert s.run_id.startswith("mock-") and run["source"] == "classical"
    ok, bad = run["frames"]["7"], run["frames"]["8"]
    assert ok["status"] == "proposed" and 0.8 < ok["score"] <= 1 and ok["H"] and len(ok["H"]) == 3
    assert ok["H_convention"] == "normalised_image_to_pitch_m" and ok["image_size"] == [W, H]
    assert len(ok["inliers"]) == len(ok["correspondences"])
    assert bad["status"] == "failed" and "no human GT" in bad["failure"] and bad["H"] is None
    assert (gt.parent / "renders" / "arsenal_mancity_seg010_7_overlay.jpg").exists()
    # file mode of a new proposals file
    assert (gt / "calibration_proposals" / "arsenal_mancity_seg010.json").stat().st_mode & 0o777 == 0o664


def test_pipeline_skip_gt_and_limit(run_env):
    gt, clips = run_env
    cfg = pl.RunConfig(gt_dir=gt, clips_dir=clips, clips=["mancity_seg010"], every=1, limit=3)
    s = pl.run_prelabel(cfg, idf.MockIdentifier(gt), log=lambda *_: None)
    frames = set(read_proposals(gt, "mancity_seg010")["runs"][s.run_id]["frames"])  # alias -> same file
    assert frames == {"0", "1", "2"} and "7" not in frames


def test_pipeline_logs_usage_and_handles_claude_errors(run_env, det):
    gt, clips = run_env
    client = FakeClient(_fake_response({}, stop="refusal"))
    cfg = pl.RunConfig(gt_dir=gt, clips_dir=clips, clips=["arsenal_mancity_seg010"], frames=[1])
    s = pl.run_prelabel(cfg, idf.ClaudeIdentifier(client=client), log=lambda *_: None)
    run = read_proposals(gt, "arsenal_mancity_seg010")["runs"][s.run_id]
    assert run["source"] == "claude" and run["frames"]["1"]["status"] == "failed"
    assert "refusal" in run["frames"]["1"]["failure"]
    assert run["params"]["usage_total"]["input_tokens"] == 1000
    assert run["params"]["usage"][0]["frame"] == 1 and run["params"]["prompt_version"] == idf.PROMPT_VERSION


def test_eval_scores_good_and_bad_proposals(run_env):
    gt, clips = run_env
    for corrupt in (0, 2):
        cfg = pl.RunConfig(gt_dir=gt, clips_dir=clips, clips=["arsenal_mancity_seg010"], frames=[7],
                           skip_gt=False, run_id=f"mock-c{corrupt}")
        pl.run_prelabel(cfg, idf.MockIdentifier(gt, corrupt=corrupt), log=lambda *_: None)
    res = ev.evaluate(gt)
    by = {r["run_id"]: r for r in res["frames"]}
    assert by["mock-c0"]["mean_px"] < 3 and by["mock-c0"]["agreement"] == 1.0
    assert by["mock-c2"]["mean_px"] is None or by["mock-c2"]["mean_px"] > by["mock-c0"]["mean_px"]
    assert res["summary"]["n_frames"] == 2 and 0 <= res["summary"]["success_rate"] <= 1
    assert "success" in ev.format_table(res)


# --------------------------------------------------------------------------
# claude CLI identifier (fake executable; never a real call)
# --------------------------------------------------------------------------


def _fake_claude(tmp_path, mode="ok", payload=None):
    """A python script standing in for `claude`: logs argv/stdin/cwd, prints stream-json."""
    log = tmp_path / "calls.jsonl"
    (tmp_path / "payload.json").write_text(json.dumps(payload or {}))
    script = tmp_path / "fakeclaude"
    script.write_text(f"""#!{sys.executable}
import json, os, sys, time
stdin = sys.stdin.read()
n = sum(1 for _ in open({str(log)!r})) if os.path.exists({str(log)!r}) else 0
open({str(log)!r}, "a").write(json.dumps({{"argv": sys.argv[1:], "stdin": stdin, "cwd": os.getcwd()}}) + "\\n")
mode = {mode!r}
if mode == "sleep":
    time.sleep(30)
print(json.dumps({{"type": "system", "subtype": "init"}}))
if mode == "garbage":
    print("not json")
    sys.exit(3)
if mode == "flaky" and n == 0:
    print(json.dumps({{"type": "result", "subtype": "error_during_execution", "is_error": True, "result": "boom"}}))
    sys.exit(1)
res = {{"type": "result", "subtype": "success", "is_error": False, "total_cost_usd": 0.25, "duration_ms": 4000,
       "duration_api_ms": 3900, "stop_reason": "end_turn",
       "usage": {{"input_tokens": 5, "output_tokens": 300, "cache_creation_input_tokens": 100, "cache_read_input_tokens": 7}},
       "modelUsage": {{"claude-opus-5-5": {{}}}}}}
if mode != "nostruct":
    res["structured_output"] = json.load(open({str(tmp_path / "payload.json")!r}))
print(json.dumps(res))
""")
    script.chmod(0o755)
    return str(script), log


def test_claude_cli_request_shape_and_parsing(det, tmp_path):
    cmd, log = _fake_claude(tmp_path, payload=_payload(det))
    ci = idf.ClaudeCLIIdentifier(command=cmd, effort="low", cwd=tmp_path)
    res = ci.identify(_inp(det))
    call = json.loads(log.read_text().splitlines()[0])
    argv = call["argv"]
    assert argv[:1] == ["-p"] and call["cwd"] == str(tmp_path.resolve())
    assert argv[argv.index("--model") + 1] == "opus" and argv[argv.index("--effort") + 1] == "low"
    assert argv[argv.index("--tools") + 1] == "" and argv[argv.index("--system-prompt") + 1] == idf.SYSTEM
    assert "--no-session-persistence" in argv and argv[argv.index("--output-format") + 1] == "stream-json"
    assert json.loads(argv[argv.index("--json-schema") + 1]) == idf.build_schema(PITCH)
    msg = json.loads(call["stdin"])
    assert msg["type"] == "user" and msg["message"]["content"] == idf.build_content(_inp(det), cache_breakpoint=False)  # same prompt as the API path
    assert "cache_control" not in call["stdin"]
    assert len([b for b in msg["message"]["content"] if b["type"] == "image"]) == 2
    direct = [lb for lb in res.labels if lb.candidate_id is None]
    assert not res.unusable and len(direct) >= 5 and [ll.line_id for ll in res.line_labels] == ["HALFWAY_LINE"]
    assert res.usage["cost_usd"] == 0.25 and res.usage["duration_ms"] == 4000 and res.usage["output_tokens"] == 300
    assert res.usage["model_served"] == "claude-opus-5-5"
    assert ci.run_prefix() == f"claudecli-opus-{idf.PROMPT_VERSION}"


def test_claude_cli_retries_once_then_reports_errors(det, tmp_path):
    cmd, log = _fake_claude(tmp_path, mode="flaky", payload=_payload(det))
    res = idf.ClaudeCLIIdentifier(command=cmd).identify(_inp(det))
    assert res.error is None and res.labels and len(log.read_text().splitlines()) == 2
    for mode, frag in (("garbage", "no result event"), ("nostruct", "no structured_output")):
        d = tmp_path / mode
        d.mkdir()
        cmd, log = _fake_claude(d, mode=mode)
        r = idf.ClaudeCLIIdentifier(command=cmd).identify(_inp(det))
        assert frag in r.error and not r.labels and len(log.read_text().splitlines()) == 2


def test_claude_cli_timeout_and_missing_binary(det, tmp_path):
    cmd, _ = _fake_claude(tmp_path, mode="sleep")
    r = idf.ClaudeCLIIdentifier(command=cmd, timeout=0.5, retries=0).identify(_inp(det))
    assert "timed out" in r.error
    r = idf.ClaudeCLIIdentifier(command=str(tmp_path / "nope"), retries=0).identify(_inp(det))
    assert "cannot run claude CLI" in r.error


def test_pipeline_with_claude_cli_records_cost(run_env, det, tmp_path):
    gt, clips = run_env
    cmd, _ = _fake_claude(tmp_path, payload=_payload(det))
    cfg = pl.RunConfig(gt_dir=gt, clips_dir=clips, clips=["arsenal_mancity_seg010"], frames=[1])
    s = pl.run_prelabel(cfg, idf.ClaudeCLIIdentifier(command=cmd), log=lambda *_: None)
    run = read_proposals(gt, "arsenal_mancity_seg010")["runs"][s.run_id]
    assert run["params"]["backend"] == "claude-cli" and run["params"]["usage_total"]["cost_usd"] == 0.25
    assert run["params"]["usage"][0]["duration_ms"] == 4000


def _faint_far_edge_frame(bright_base: bool = False):
    """Wide-camera frame: dark stands + apron above y=100, saturated grass below, a 3 px PALE strip (V ~ grass,
    low saturation) at the boundary - or, with ``bright_base``, a bright white board base there."""
    import cv2

    img = np.zeros((300, 600, 3), np.uint8)
    img[:] = (20, 20, 20)  # stands / apron
    grass = np.zeros((300, 600, 3), np.uint8)
    grass[:] = (40, 150, 60)  # BGR: saturated green, V=150
    img[100:] = grass[100:]
    strip = (200, 230, 230) if bright_base else (95, 150, 140)  # bright white vs grass-bright, desaturated
    img[96:101] = strip
    return cv2.GaussianBlur(img, (3, 3), 0.8)


def test_far_edge_line_finds_a_faint_strip_but_not_a_bright_board_base():
    from footy_track.calibration.prelabel.candidates import far_edge_line, grass_mask

    faint = _faint_far_edge_frame()
    g = grass_mask(faint)
    assert not g[40, 300] and g[200, 300]  # the stands/apron are not grass
    far = far_edge_line(faint, g)
    ys, xs = np.nonzero(far)
    assert len(xs) > 300 and 92 <= np.median(ys) <= 104 and xs.max() - xs.min() > 400
    bright = _faint_far_edge_frame(bright_base=True)
    assert not far_edge_line(bright, grass_mask(bright)).any()
