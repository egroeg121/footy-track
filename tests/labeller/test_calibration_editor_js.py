"""Pure JS helpers of the calibration editor page (proposal auto-open), run under node."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

PAGE = Path(__file__).resolve().parents[2] / "src/footy_track/labeller/web/calibration.html"

CHECK = r"""
const assert = require("assert");
const items = [{frame_index: 5, status: "proposed", score: 0.5}, {frame_index: 2, status: "proposed", score: 0.9},
  {frame_index: 9, status: "failed", score: 0}, {frame_index: 1, status: "accepted", score: 0.99},
  {frame_index: 7, status: "proposed", score: 0.9}, {frame_index: 8, status: "proposed", score: null}];
assert.equal(pickBestProposal(items).frame_index, 2);                 // highest score, ties -> lowest frame
assert.equal(pickBestProposal(items, [2]).frame_index, 7);            // frames with GT / not broadcast skipped
assert.equal(pickBestProposal(items, [2, 7, 5]).frame_index, 8);      // null score ranks last but still qualifies
assert.equal(pickBestProposal(items, [2, 7, 5, 8]), null);            // failed/accepted never qualify
assert.equal(pickBestProposal([]), null);
const kps = {A: [1, 2]};
let p = proposalPoints({H: [1], correspondences: [{image: [.1, .2], pitch_id: "A", pitch_xy: [1, 2]},
  {image: [.3, .4], pitch_id: "A", pitch_xy: [1, 2]}], inliers: [true, false]}, kps);
assert.equal(p.length, 1);                                            // outlier dropped when there is an H
p = proposalPoints({H: null, correspondences: [{image: [.1, .2], pitch_id: "A"}], inliers: []}, kps);
assert.deepEqual(p[0].pt, {id: "A", x: 1, y: 2});                     // failed: keep labels, xy from keypoint id
assert.equal(proposalPoints({H: null, correspondences: []}, kps).length, 0);
"""


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
def test_proposal_helpers_under_node(tmp_path):
    html = PAGE.read_text()
    a, b = html.index("// ---- proposals helpers"), html.index("// ---- end proposals helpers")
    f = tmp_path / "helpers.js"
    f.write_text(html[a:b] + CHECK)
    r = subprocess.run(["node", str(f)], capture_output=True, text=True, timeout=30)
    assert r.returncode == 0, r.stderr


AUTO_CHECK = r"""
const assert = require("assert");
const corrs = [{image: [.4, .3], pitch_id: "CENTRE_CIRCLE_TOP", pitch_xy: [0, 9.15]},
               {image: [.5, .6], pitch_id: "CENTRE_CIRCLE_BOTTOM", pitch_xy: [0, -9.15]}];
let a = autoPairsToAdd(corrs, []);
assert.deepEqual(a.img, [{u: .4, v: .3}, {u: .5, v: .6}]);
assert.deepEqual(a.ptc[0], {id: "CENTRE_CIRCLE_TOP", x: 0, y: 9.15});
a = autoPairsToAdd(corrs, [{id: "CENTRE_CIRCLE_TOP", x: 0, y: 9.15}, {id: null, x: 1, y: 2}]);
assert.equal(a.img.length, 1);                                        // placed id skipped, free point ignored
assert.equal(a.ptc[0].id, "CENTRE_CIRCLE_BOTTOM");
assert.equal(autoPairsToAdd(corrs, a.ptc.concat([{id: "CENTRE_CIRCLE_TOP"}])).img.length, 0);
assert.equal(autoPairsToAdd([corrs[0], corrs[0]], []).img.length, 1);  // duplicate id within the answer
assert.equal(autoPairsToAdd(null, []).img.length, 0);
// free (id-less) derived points: added with null id, skipped when the same metres are already placed
const free = [{image: [.2, .2], pitch_id: null, pitch_xy: [-6.47, 6.47]}, {image: [.3, .3], pitch_id: null, pitch_xy: [6.47, 6.47]}];
let f = autoPairsToAdd(free, []);
assert.equal(f.img.length, 2);
assert.deepEqual(f.ptc[0], {id: null, x: -6.47, y: 6.47});
assert.equal(autoPairsToAdd(free, [{id: null, x: -6.47, y: 6.47}]).img.length, 1);
assert.equal(autoPairsToAdd(free.concat(free), []).img.length, 2);
assert.ok(autoStatusText("centre", 0.87, 2, 0, 2, 0).includes("2 points placed — add 2 more (e.g. touchline / halfway line ends) to fit"));
assert.ok(autoStatusText("d", 0.7, 9, 0, 9, 2).includes("penalty area (confidence 70%): added 9 points (2 derived)"));
assert.ok(!autoStatusText("centre", 0.9, 2, 0, 4).includes("placed"));
"""


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
def test_auto_circle_helpers_under_node(tmp_path):
    html = PAGE.read_text()
    a, b = html.index("// ---- auto circle/D helpers"), html.index("// ---- end auto circle/D helpers")
    f = tmp_path / "helpers.js"
    f.write_text(html[a:b] + AUTO_CHECK)
    r = subprocess.run(["node", str(f)], capture_output=True, text=True, timeout=30)
    assert r.returncode == 0, r.stderr


LENS_CHECK = r"""
const assert = require("assert");
assert.equal(makeLens(0, 1.78), null);                                 // k1 = 0 is no lens
assert.equal(makeLens(NaN, 1.78), null);
assert.deepEqual(makeLens(0.0147, 16 / 9), {model: "division", k1: 0.0147, centre: [0.5, 0.5], aspect: 16 / 9});
assert.equal(lensK1Of(null), 0);
assert.equal(lensK1Of({k1: 0.035}), 0.035);
let k = 0;
for (let i = 0; i < 5; i++) k = nudgeK1(k, 1, false);
assert.equal(k, 0.01);                                                 // no float drift from repeated nudges
assert.equal(nudgeK1(0.01, -1, true), 0);                              // Shift = 0.01 steps
assert.equal(nudgeK1(0, -1, false), -0.002);
const ok = {accepted: true, f1_before: 0.86, f1_after: 0.9, k1_before: 0, k1_after: 0.0231, max_click_shift_px: 2.84,
            edge_before: {median_px: 1.5}, edge_after: {median_px: 0.8}};
const t = snapStatusText(ok);
assert.ok(t.includes("0.860") && t.includes("0.900") && t.includes("0.0231") && t.includes("1.5") && t.includes("2.8"));
assert.equal(snapStatusText({accepted: false, reason: "line F1 did not improve"}), "line F1 did not improve");
assert.ok(snapStatusText({accepted: true, f1_before: 0.5, f1_after: 0.6, k1_before: 0, k1_after: 0.01, edge_before: null}).includes("snapped"));
"""


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
def test_lens_helpers_under_node(tmp_path):
    html = PAGE.read_text()
    a, b = html.index("// ---- lens helpers"), html.index("// ---- end lens helpers")
    f = tmp_path / "lens.js"
    f.write_text(html[a:b] + LENS_CHECK)
    r = subprocess.run(["node", str(f)], capture_output=True, text=True, timeout=30)
    assert r.returncode == 0, r.stderr
