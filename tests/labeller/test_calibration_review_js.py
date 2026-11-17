"""Warp cache / sequential-prefetch logic of the calibration review page, run under node."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

PAGE = (
    Path(__file__).resolve().parents[2]
    / "src/footy_track/labeller/web/calibration_review.html"
)

CHECK = r"""
const assert = require("assert");
const keyOf = (i) => `${i.clip}|${i.run_id}|${i.frame_index}`;
const it = (n) => ({ clip: "c", run_id: "r", frame_index: n });
const tick = () => new Promise((r) => setTimeout(r, 0));

// one tap always flips what the user asked for; a long-press peek inverts it
assert.equal(wantWarp(true, false), true);
assert.equal(wantWarp(false, false), false);
assert.equal(wantWarp(true, true), false);
assert.equal(wantWarp(false, true), true);

(async () => {
  // fetches are gated so we control completion order and can observe concurrency
  const gates = new Map(), started = []; let active = 0, maxActive = 0;
  const revoked = [];
  const fetchOne = (i) => new Promise((res, rej) => {
    started.push(i.frame_index); active++; maxActive = Math.max(maxActive, active);
    gates.set(i.frame_index, { ok: () => { active--; res(`blob:${i.frame_index}`); }, fail: (m) => { active--; rej(new Error(m)); } });
  });
  const S = makeWarpStore(fetchOne, (u) => revoked.push(u), keyOf);

  // prefetch 2 and 3 behind an in-flight 1; an urgent 5 jumps ahead of the queued ones only
  const p1 = S.request(it(1), true);
  const p2 = S.request(it(2), false), p3 = S.request(it(3), false);
  const p5 = S.request(it(5), true);
  assert.strictEqual(S.request(it(2), false), p2);          // deduped
  await tick();
  assert.deepEqual(started, [1]);                            // sequential: only one request in flight
  gates.get(1).ok(); assert.equal(await p1, "blob:1");
  await tick();
  assert.deepEqual(started, [1, 5]);                         // urgent 5 before 2 and 3
  gates.get(5).ok(); await p5; await tick();
  assert.deepEqual(started, [1, 5, 2]);
  assert.equal(maxActive, 1);

  // retain: drops queued entries outside the window (cancelled), revokes cached URLs outside it
  p3.catch(() => {});
  S.retain(new Set([keyOf(it(2)), keyOf(it(5))]));
  await assert.rejects(p3, /cancelled/);
  assert.deepEqual(revoked, ["blob:1"]);
  assert.equal(S.has(keyOf(it(1))), false); assert.equal(S.has(keyOf(it(5))), true);
  assert.equal(S.get(keyOf(it(5))), "blob:5");
  assert.equal(await S.request(it(5), true), "blob:5");     // cache hit, no new fetch

  // an error does not wedge the queue, and a failed key can be retried
  gates.get(2).fail("boom"); await assert.rejects(p2, /boom/); await tick();
  const p4 = S.request(it(4), false); await tick();
  assert.deepEqual(started.slice(-1), [4]);
  gates.get(4).ok(); await p4;
  const p2b = S.request(it(2), true); await tick();
  assert.deepEqual(started.slice(-1), [2]);
  gates.get(2).ok(); assert.equal(await p2b, "blob:2");
  assert.equal(maxActive, 1);
  console.log("ok");
})().catch((e) => { console.error(e); process.exit(1); });
"""


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
def test_warp_store_under_node(tmp_path):
    html = PAGE.read_text()
    a, b = html.index("// ---- warp helpers"), html.index("// ---- end warp helpers")
    f = tmp_path / "helpers.js"
    f.write_text(html[a:b] + CHECK)
    r = subprocess.run(["node", str(f)], capture_output=True, text=True, timeout=30)
    assert r.returncode == 0, r.stderr
