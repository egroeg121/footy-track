"""PTZ V3 anchor recovery for segments whose lines-only anchors fail (``reports/ptz_v3_weak.md``).

The V3 solve needs one right pose per motion-connected run (an *anchor*); the per-frame lines-only search
(:func:`ptz.search_lines`) provides it. On ~14% of the Man City segments it gave none although the lines are plainly
visible. What this module changes, all of it only for runs the standard pass left uncalibrated:

* **Pose prior** (:func:`pose_prior`, ``rig.info["pose_prior"]``): over the 140 accepted Man City segments the broadcast
  camera keeps ``tilt = a + b ln f`` within 1.1 deg (the operator frames the play at every zoom). The search runs on a
  grid dense along that band (:func:`prior_grid`) instead of a 3-D grid dominated by wrong zoomed-out optima, and a
  hypothesis off the band is refused (:func:`in_prior`).
* **Multi-start refinement** (:func:`search_lines_ms`): the LM refinement of a coarse cell is easily trapped (a
  centre circle 15% too small snaps onto its inner stroke), so every cell is also refined from scaled focal lengths.
* **Camera position** (:func:`rig_variants`): besides the stand's centre position the broadcast uses a gantry camera level
  with each penalty-area line (C_x = +-32.7 m). A weak segment's rig is chosen from a few probe frames.
* **Finder anchors** (:func:`finder_anchor`): the centre-circle / penalty-D finders give image points of known pitch
  positions; their PTZ point solve is a second, independent anchor source.
* **Confirmation** (:func:`confirm_anchors`): a relaxed-F1 candidate only becomes an anchor when another candidate,
  carried along the motion chain, lands on the same camera (or when it is very strong).
* **Trimming** (:func:`trim_runs`, opt-in, off in the runner): evidence frames that are not main-camera views (logo wipe,
  close-up, crowd, scoreboard) cut the run. Measured on the weak segments it did not help: the classifier also rejects
  valid zoomed goal-area views (seg108 f130-174: 100% accepted untrimmed, 75% trimmed), and the F1 acceptance already
  rejects real tails.
"""

# ruff: noqa: PLR0913, PLR0912, PLR0915, PLC0415, N806, E741
from __future__ import annotations

import time

import numpy as np

from footy_track.calibration import ptz
from footy_track.calibration import ptz_temporal as pt
from footy_track.calibration.pitch_model import PitchModel

Array = np.ndarray

F1_CAND = 0.40  # relaxed per-frame F1 for a *candidate* (the standard anchor needs 0.55); confirmed before use
F1_STRONG = 0.70  # a candidate this good needs no second frame
CONFIRM_TOL_M = 1.5  # two candidates carried to the same frame agree within this (pitch-plane metres)
CONFIRM_GAP = 80  # max frame distance between two candidates that confirm each other
PRIOR_NSIG = 4.0
PM_MAIN = 0.4  # an evidence frame below this main-camera probability is not a pitch view


# --------------------------------------------------------------------------
# pose prior
# --------------------------------------------------------------------------


def pose_prior(rig: ptz.Rig) -> dict | None:
    """``rig.info["pose_prior"]`` (``tilt_a``, ``tilt_b``, ``tilt_sigma_deg``, ``pan_deg``, ``f_px``) or None."""
    p = (rig.info or {}).get("pose_prior")
    return dict(p) if p else None


def tilt_resid_deg(prior: dict, tilt: float, logf: float) -> float:
    """Tilt (deg) minus the band's centre at this zoom."""
    return float(np.degrees(tilt) - (prior["tilt_a"] + prior["tilt_b"] * logf))


def in_prior(prior: dict | None, pan: float, tilt: float, logf: float, nsig: float = PRIOR_NSIG) -> bool:
    if prior is None:
        return True
    lo, hi = prior["pan_deg"]
    flo, fhi = prior["f_px"]
    return bool(lo <= np.degrees(pan) <= hi and flo <= np.exp(logf) <= fhi
                and abs(tilt_resid_deg(prior, tilt, logf)) <= nsig * prior["tilt_sigma_deg"])


def prior_grid(prior: dict, n_pan: int = 53, n_f: int = 22, tilt_offsets_deg=(-4.0, -2.0, 0.0, 2.0, 4.0)) -> Array:
    """Pan x log f grid with the tilt on the prior band (+- offsets): ~6k poses, dense where the camera really is.
    ``prior["tilt_offsets_deg"]`` overrides the offsets (a narrower band, e.g. the Vitality's 0.8 deg sigma, needs finer ones)."""
    tilt_offsets_deg = prior.get("tilt_offsets_deg", tilt_offsets_deg)
    pan = np.linspace(np.radians(prior["pan_deg"][0]), np.radians(prior["pan_deg"][1]), n_pan)
    logf = np.linspace(np.log(prior["f_px"][0]), np.log(prior["f_px"][1]), n_f)
    out = []
    for lf in logf:
        c = prior["tilt_a"] + prior["tilt_b"] * lf
        for o in tilt_offsets_deg:
            out.append(np.column_stack([pan, np.full(n_pan, np.radians(c + o)), np.full(n_pan, lf)]))
    return np.concatenate(out)


def rig_variants(rig: ptz.Rig) -> list[tuple[str, ptz.Rig]]:
    """Camera positions on the stand: ``[("centre", rig), ("x+33", ...), ("x-33", ...)]`` from
    ``rig.info["pose_prior"]["side_cx_m"]`` (the rig alone when absent). The broadcast also uses a gantry camera level with
    each penalty-area line: free-C fits of the penalty-D finder points of the weak segments land at C_x = +32.8 / -32.1 m
    with C_y, C_z within 2 m of the centre camera (seg073 / seg108 / seg156); their fixed-C reprojection error is 215 px at the
    centre position and 1-7 px at the side one. An entry is a number (x offset) or ``[dx, dy, dz]`` (the Vitality's side gantries
    sit 0.6-1.7 m further back and 0.8-1.1 m higher than the centre rig: with only dx the line F1 at the side position drops
    from 0.65 to 0.38)."""
    side = ((rig.info or {}).get("pose_prior") or {}).get("side_cx_m")
    out = [("centre", rig)]
    for e in side or []:
        dx, dy, dz = (e, 0.0, 0.0) if np.isscalar(e) else e  # a number = x offset only; [dx, dy, dz] = a full offset from the rig C
        out.append((f"x{dx:+g}", ptz.Rig(np.array([rig.C[0] + dx, rig.C[1] + dy, rig.C[2] + dz]), rig.roll, rig.k1, rig.sigma_C,
                                         rig.pitch, rig.info)))
    return out


# --------------------------------------------------------------------------
# lines-only search, multi-start
# --------------------------------------------------------------------------


def search_lines_ms(ev: ptz.LineEvidence, rig: ptz.Rig, pitch: PitchModel, *, prior: dict | None = None,
                    n_cells: int = 24, n_refine: int = 6, n_ms_cells: int = 10, f_mults=(0.82, 1.0, 1.2),
                    keep: int = 4) -> list[dict]:
    """Lines-only hypotheses of one frame, best F1 first: ``{"cam", "f1", "prior_ok", "tilt_resid"}``.

    Coarse grid (the prior band when ``prior`` is given, else the default grid) scored with the wide capture range,
    NMS cells, then two families of LM starts: the standard local-grid winners of the top ``n_refine`` cells and the
    ``n_ms_cells`` best cells at every focal length multiple of ``f_mults``. Hypotheses are merged when they are the same
    camera in pitch space (< 2.5 m), ranked by F1."""
    grid = prior_grid(prior) if prior else ptz._GRID_LINES
    samples, pa, pb = ptz._samples(pitch, 1.5)
    C0 = np.asarray(rig.C, float)

    def eval_grid(g, tau):
        out = np.empty(len(g))
        for i in range(0, len(g), 400):
            uv, vis = ptz._project_obs(g[i:i + 400], C0, rig.roll, rig.k1, samples, ev)
            out[i:i + 400] = ptz._line_scores(uv, vis, pa, pb, ev, tau)[0]
        return out

    scores = eval_grid(grid, ptz.TAU_COARSE)
    order = np.argsort(scores)
    cells = ptz._nms(grid, order[:1500], max(n_cells, n_ms_cells))
    # standard family: local grid around each cell, tighter capture range
    step = np.array([np.diff(np.unique(grid[:, 0]))[0], max(np.radians(1.0), 0.02), np.diff(np.unique(grid[:, 2]))[0]])
    off = np.stack(np.meshgrid(*[np.arange(-2, 3)] * 3, indexing="ij"), axis=-1).reshape(-1, 3) * step / 2.0
    cl = cells[:n_cells]
    local = np.concatenate([grid[c] + off for c in cl])
    ls = eval_grid(local, 0.4 * ptz.TAU_COARSE)
    starts = []
    for i in range(len(cl)):
        j = int(np.argmin(ls[i * len(off):(i + 1) * len(off)]))
        starts.append((ls[i * len(off) + j], local[i * len(off) + j]))
    starts.sort(key=lambda t: t[0])
    starts = [g for _, g in starts[:n_refine]]
    # multi-start family: top cells at scaled zooms
    for c in cells[:n_ms_cells]:
        for m in f_mults:
            g = grid[c].copy()
            g[2] += np.log(m)
            starts.append(g)
    hyps: list[dict] = []
    for g0 in starts:
        cam = ptz._refine_lines(ptz.Camera(g0[0], g0[1], g0[2], rig.roll, C0.copy(), rig.k1), ev, rig, pitch)
        hyps.append({"cam": cam, "f1": float(ptz.line_support([cam], ev, pitch)[0])})
    hyps.sort(key=lambda h: -h["f1"])
    distinct: list[dict] = []
    for h in hyps:
        if all(ptz._cam_distance(h["cam"], d["cam"]) > 2.5 for d in distinct):
            distinct.append(h)
        if len(distinct) >= keep:
            break
    for h in distinct:
        h["prior_ok"] = in_prior(prior, h["cam"].pan, h["cam"].tilt, h["cam"].logf)
        h["tilt_resid"] = tilt_resid_deg(prior, h["cam"].tilt, h["cam"].logf) if prior else None
    return distinct


def _record(cam: ptz.Camera, ev: ptz.LineEvidence, rig: ptz.Rig, pitch: PitchModel, src: str, f1: float,
            margin: float | None, prior: dict | None, secs: float) -> dict:
    """Anchor record in the driver's format (``ok``, ``f1``, ``pan``, ``tilt``, ``logf``, ``refit``)."""
    fixed = ptz._refine_lines(cam, ev, rig.with_sigma((0.0, 0.0, 0.0)), pitch)
    f1_fixed = float(ptz.line_support([fixed], ev, pitch)[0])
    return {"ok": False, "cand": True, "src": src, "f1": f1, "f1_fixed": f1_fixed, "margin": margin, "secs": secs,
            "pan": cam.pan, "tilt": cam.tilt, "logf": cam.logf,
            "prior_ok": in_prior(prior, fixed.pan, fixed.tilt, fixed.logf),
            "refit": {"free": [cam.pan, cam.tilt, cam.logf, *cam.C], "fixed": [fixed.pan, fixed.tilt, fixed.logf]}}


def lines_candidate(ev: ptz.LineEvidence, rig: ptz.Rig, pitch: PitchModel, prior: dict | None, **kw) -> dict | None:
    t0 = time.time()
    hyps = search_lines_ms(ev, rig, pitch, prior=prior, **kw)
    hyps = [h for h in hyps if h["prior_ok"]] or []
    if not hyps:
        return None
    b = hyps[0]
    margin = b["f1"] - hyps[1]["f1"] if len(hyps) > 1 else 1.0
    return _record(b["cam"], ev, rig, pitch, "lines_ms", b["f1"], float(margin), prior, time.time() - t0)


def finder_anchor(cache: pt.ClipCache, t: int, rig: ptz.Rig, pitch: PitchModel, prior: dict | None) -> dict | None:
    """Centre-circle / penalty-D finders on frame ``t`` -> PTZ point solve with C hard-fixed at the rig -> anchor record.

    The finders' points sit on known pitch features (circle, halfway line, box corners, arc), so the pose they give is not
    refined on the line mask afterwards: on box views the mask refinement slides the box onto stray strokes and *lowers*
    the F1 (measured on Man City seg073 / seg108 / seg156: pts-only 0.56 / 0.48 / 0.46 -> refined 0.37 / 0.32 / 0.35).
    Each finder is solved alone and, when both answered, together; the pose with the best line F1 is kept (``f1`` is that
    F1, ``f1_fixed`` the same: there is only the C-fixed pose)."""
    from footy_track.calibration import geometry as geo
    from footy_track.calibration import pitch_finders as pf

    ev = cache.evidence(t)
    bgr = cache.read_full(t)
    if ev is None or bgr is None:
        return None
    t0 = time.time()
    dist = geo.make_distortion(rig.k1)
    fixed_rig = rig.with_sigma((0.0, 0.0, 0.0))
    found: dict[str, tuple[list, list]] = {}
    for kind in ("centre", "d"):
        try:
            det = pf.find_points(kind, bgr, dist=dist, pitch=pitch)
        except Exception:  # noqa: BLE001  a finder crash is an abstention
            continue
        if det.result is not None and len(det.result.points) >= 2:
            found[kind] = ([p.uv for p in det.result.points], [p.pitch_xy for p in det.result.points])
    if len(found) == 2:
        found["+".join(found)] = (found["centre"][0] + found["d"][0], found["centre"][1] + found["d"][1])
    best = None
    for kind, (uv, xy) in found.items():
        cam = ptz.solve_points(np.array(uv), np.array(xy), fixed_rig)
        if cam is None:
            continue
        f1 = float(ptz.line_support([cam], ev, pitch)[0])
        if best is None or f1 > best[0]:
            best = (f1, cam, kind, len(uv))
    if best is None:
        return None
    f1, cam, kind, n_pts = best
    pose = [cam.pan, cam.tilt, cam.logf]
    return {"ok": False, "cand": True, "src": "finder:" + kind, "f1": f1, "f1_fixed": f1, "margin": None, "secs": time.time() - t0,
            "pan": cam.pan, "tilt": cam.tilt, "logf": cam.logf, "prior_ok": in_prior(prior, *pose), "n_points": n_pts,
            "refit": {"free": [*pose, *map(float, rig.C)], "fixed": pose}}


# --------------------------------------------------------------------------
# trimming
# --------------------------------------------------------------------------


def trim_runs(runs: list[tuple[int, int]], cache: pt.ClipCache, pm: dict[int, float], min_len: int = 8,
              thr: float = PM_MAIN, bad_run: int = 2) -> list[tuple[int, int]]:
    """Cut every run at stretches of >= ``bad_run`` consecutive evidence frames with main-camera probability < ``thr`` (a
    logo wipe, close-up, crowd shot, scoreboard graphic; a stretch touching the run's end counts at any length) and drop
    the stretch. Where the stretch starts, frames from the largest histogram jump in the 5-frame gap before its first
    evidence frame (the true cut) are dropped, or from the gap's start when the transition is smooth; where it ends,
    the run resumes at the largest jump in the gap after it, or at the next good evidence frame. Pieces shorter than
    ``min_len`` are dropped."""
    out: list[tuple[int, int]] = []
    for a, b in _split_hard_cuts(runs, cache, min_len):
        out += _trim_run(a, b, cache, pm, min_len, thr, bad_run)
    return out


def _split_hard_cuts(runs, cache: pt.ClipCache, min_len: int) -> list[tuple[int, int]]:
    """Split every run at a hard histogram cut (``CUT_HARD``). ``segment_shots`` also finds them but merges a piece shorter
    than ``min_len`` back into the previous shot (a 1-3 frame close-up or logo flash at the end of a segment); here such a
    sliver is dropped instead."""
    out = []
    for a, b in runs:
        cuts = [t for t in range(a + 1, b + 1) if cache.hd[t] >= pt.CUT_HARD]
        lo = a
        for c in [*cuts, b + 1]:
            if c - lo >= min_len:
                out.append((lo, c - 1))
            lo = c
    return out


def _trim_run(a: int, b: int, cache: pt.ClipCache, pm: dict[int, float], min_len: int, thr: float, bad_run: int
              ) -> list[tuple[int, int]]:
    e = [t for t in cache.ev_frames if a <= t <= b]
    if not e:  # nothing to judge it by
        return [(a, b)]
    bad = [pm.get(t, float("nan")) < thr for t in e]
    spans = []  # (remove_from, resume_at) in frames
    i = 0
    while i < len(e):
        if not bad[i]:
            i += 1
            continue
        j = i
        while j + 1 < len(e) and bad[j + 1]:
            j += 1
        if j - i + 1 >= bad_run or i == 0 or j == len(e) - 1:
            lo = a if i == 0 else _jump(cache, e[i - 1], e[i], default=e[i - 1] + 1)
            hi = b + 1 if j == len(e) - 1 else _jump(cache, e[j], e[j + 1], default=e[j + 1])
            spans.append((lo, hi))
        i = j + 1
    if e[-1] < b and not bad[-1]:  # frames after the last evidence frame: a jump there starts a different shot
        k = _jump(cache, e[-1], b, default=b + 1)
        if k <= b:
            spans.append((k, b + 1))
    out, cur = [], a
    for lo, hi in spans:
        if lo - cur >= min_len:
            out.append((cur, lo - 1))
        cur = max(cur, hi)
    if b + 1 - cur >= min_len:
        out.append((cur, b))
    return out


def _jump(cache: pt.ClipCache, t_a: int, t_b: int, default: int, min_hd: float = 0.25) -> int:
    """Frame in (t_a, t_b] where the histogram jumps most (the first frame of the new shot), else ``default``."""
    seg = cache.hd[t_a + 1:t_b + 1]
    if len(seg) and float(seg.max()) >= min_hd:
        return t_a + 1 + int(np.argmax(seg))
    return default


# --------------------------------------------------------------------------
# confirmation by motion consensus
# --------------------------------------------------------------------------


def confirm_anchors(cands: dict[int, Array], pairs_by, sh: pt.Shared, tol_m: float = CONFIRM_TOL_M,
                    max_gap: int = CONFIRM_GAP, strong: dict[int, bool] | None = None
                    ) -> tuple[dict[int, Array], dict[int, list[int]]]:
    """Keep the candidate states (``{frame: (pan, tilt, logf)}``) that another candidate confirms: carried along the
    motion chain (forward, exact f) to the other frame, both land within ``tol_m`` (pitch-plane mean displacement) of
    each other. ``strong`` candidates stay without a partner. -> (kept, {frame: partners})."""
    ts = sorted(cands)
    partners: dict[int, list[int]] = {t: [] for t in ts}
    for i, a in enumerate(ts):
        for b in ts[i + 1:]:
            if b - a > max_gap:
                break
            ch = pt.chain(cands[a], pairs_by, a, b, sh)
            if b in ch and pt._state_dist(ch[b], cands[b], sh) < tol_m:
                partners[a].append(b)
                partners[b].append(a)
    kept = {t: cands[t] for t in ts if partners[t] or (strong or {}).get(t)}
    return kept, partners
