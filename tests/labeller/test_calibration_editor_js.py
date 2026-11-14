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
