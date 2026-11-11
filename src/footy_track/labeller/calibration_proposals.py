"""Review of machine calibration proposals (LAB-125x).

Proposals are written by ``footy_track.scripts.prelabel_calibration`` to
``<gt dir>/calibration_proposals/<canonical clip>.json`` (shared contract in
``docs/design/calibration.md``). These routes list them across clips (filter by
run/status, sort by score), return one with its projected pitch overlay, and
accept (copy into ``calibration_gt`` with ``origin``) or reject. Accepting
re-fits ``H`` server-side from the points so stored H and points agree.
"""

from __future__ import annotations

import asyncio

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from footy_track.calibration import geometry as geo
from footy_track.calibration.pitch_model import PitchModel
from footy_track.calibration.prelabel.proposals import (
    STATUSES,
    list_clips_with_proposals,
    read_proposals,
    update_proposals,
)
from footy_track.labeller import calibration as cal
from footy_track.labeller.review import _gt_marks_dir

router = APIRouter()

#: points closer than this (normalised) count as unchanged by the operator
_SAME_UV = 2e-4


def _summary(clip: str, run_id: str, run: dict, k: str, rec: dict) -> dict:
    corr = rec.get("correspondences", [])
    inl = rec.get("inliers") or [True] * len(corr)
    return {
        "clip": clip, "run_id": run_id, "frame_index": int(k), "status": rec["status"],
        "score": rec.get("score"), "failure": rec.get("failure"), "source": run.get("source"),
        "n_points": len(corr), "n_inliers": int(sum(inl)),
        "view": rec.get("view"), "verify": rec.get("verify"),
    }


def _collect(gt_dir) -> tuple[list[dict], dict]:
    items: list[dict] = []
    runs: dict[str, dict] = {}
    for clip in list_clips_with_proposals(gt_dir):
        for rid, run in read_proposals(gt_dir, clip)["runs"].items():
            info = runs.setdefault(
                rid, {"run_id": rid, "source": run.get("source"), "created": run.get("created"),
                      "params": {k: v for k, v in (run.get("params") or {}).items() if k != "usage"},
                      "counts": dict.fromkeys(STATUSES, 0)},
            )
            for k, rec in run["frames"].items():
                items.append(_summary(clip, rid, run, k, rec))
                info["counts"][rec["status"]] = info["counts"].get(rec["status"], 0) + 1
    return items, runs


@router.get("/calibration/api/proposals")
async def list_proposals(
    run: str | None = None,
    status: str | None = "proposed",
    clip: str | None = None,
    sort: str = "score_asc",
    limit: int = 1000,
):
    """Queue: ``status`` is comma separated (default ``proposed``; ``all`` for every
    status); ``sort`` is ``score_asc`` (worst first, default) or ``score_desc``.
    Failed proposals have score 0 so they lead a worst-first list when included."""
    if sort not in ("score_asc", "score_desc"):
        return cal._err(400, "sort must be score_asc or score_desc")
    want = None if status in (None, "", "all") else set(status.split(","))
    if want and not want <= set(STATUSES):
        return cal._err(400, f"unknown status in {status!r}")
    items, runs = await asyncio.to_thread(_collect, _gt_marks_dir())
    sel = [
        i for i in items
        if (not run or i["run_id"] == run) and (not clip or i["clip"] == clip)
        and (want is None or i["status"] in want)
    ]
    sel.sort(key=lambda i: (i["score"] if i["score"] is not None else 0.0, i["clip"], i["frame_index"]),
             reverse=(sort == "score_desc"))
    return {"items": sel[:limit], "total": len(sel), "runs": sorted(runs.values(), key=lambda r: r["created"] or "")}


def _get(clip: str, run_id: str, idx: int) -> tuple[dict, dict] | None:
    run = read_proposals(_gt_marks_dir(), clip)["runs"].get(run_id)
    if run is None or str(idx) not in run["frames"]:
        return None
    return run, run["frames"][str(idx)]


@router.get("/calibration/api/proposals/{clip}/{run_id}/{idx}")
async def get_proposal(clip: str, run_id: str, idx: int):
    got = await asyncio.to_thread(_get, clip, run_id, idx)
    if got is None:
        return cal._err(404, "no such proposal")
    run, rec = got
    out = dict(rec)
    out.update(clip=clip, run_id=run_id, source=run.get("source"))
    if rec.get("H") is not None:
        pitch = PitchModel(*rec["pitch_dims"])
        import numpy as np  # noqa: PLC0415

        out["overlay"] = {
            n: [r.tolist() for r in runs]
            for n, runs in geo.project_pitch_model(np.array(rec["H"]), pitch).items()
        }
    return out


def _existing_gt(clip: str, idx: int) -> dict | None:
    """The calibration GT record of this frame, or ``None``."""
    try:
        return cal._read_gt(clip)["frames"].get(str(idx))
    except (OSError, ValueError, KeyError):
        return None


def _same_points(a: list[dict], b: list[dict]) -> bool:
    if len(a) != len(b):
        return False
    key = lambda c: (c.get("pitch_id") or tuple(c["pitch_xy"]), round(c["image"][0], 4), round(c["image"][1], 4))  # noqa: E731
    for x, y in zip(sorted(a, key=lambda c: str(key(c))), sorted(b, key=lambda c: str(key(c))), strict=True):
        if (x.get("pitch_id") or None) != (y.get("pitch_id") or None):
            return False
        if abs(x["image"][0] - y["image"][0]) > _SAME_UV or abs(x["image"][1] - y["image"][1]) > _SAME_UV:
            return False
    return True


@router.post("/calibration/api/proposals/{clip}/{run_id}/{idx}/accept")
async def accept_proposal(clip: str, run_id: str, idx: int, body: dict | None = None):
    """Copy the proposal into ``calibration_gt`` with ``origin``.

    No body: accept as proposed (its inlier points). Body ``{correspondences,
    method?, overwrite?}``: the operator's points; status becomes ``edited`` when they
    differ from the proposal's. A frame that already has GT -> 409 unless
    ``overwrite`` (checked first); the 409 says whether that GT is hand-labelled
    (no ``origin``) or from an earlier machine proposal, as ``existing_gt``.
    """
    body = body or {}
    got = await asyncio.to_thread(_get, clip, run_id, idx)
    if got is None:
        return cal._err(404, "no such proposal")
    existing = await asyncio.to_thread(_existing_gt, clip, idx)
    if existing is not None and not body.get("overwrite"):
        human = not existing.get("origin")
        msg = (
            "this frame already has hand-labelled calibration GT; it is not replaced unless you "
            "resend with overwrite=true"
            if human else
            f"this frame already has calibration GT from machine run {existing['origin'].get('run_id')!r}; "
            "resend with overwrite=true to replace it"
        )
        return JSONResponse({"ok": False, "error": msg, "existing_gt": "human" if human else "machine"},
                            status_code=409)
    run, rec = got
    corr = rec.get("correspondences", [])
    inl = rec.get("inliers") or [True] * len(corr)
    proposed_pts = [c for c, ok in zip(corr, inl, strict=False) if ok]
    if "correspondences" in body:
        pts, status = body["correspondences"], None
    elif rec["status"] == "failed" or rec.get("H") is None:
        return cal._err(409, "a failed proposal has no H; edit its points to accept it")
    else:
        pts, status = proposed_pts, "accepted"
    fit_body = {
        "correspondences": pts, "pitch_dims": body.get("pitch_dims") or rec["pitch_dims"],
        "method": body.get("method", "dlt"), "image_size": body.get("image_size") or rec["image_size"],
    }
    try:
        res = cal._fit(fit_body)
    except cal._BadRequest as exc:
        return cal._err(400, str(exc))
    if status is None:
        # compare the operator's cleaned points with what was proposed
        status = "accepted" if _same_points(res["correspondences"], proposed_pts) else "edited"
    origin = {"source": run.get("source") or "claude", "run_id": run_id, "proposal_status": status}
    gt_rec = cal._gt_record(idx, res, fit_body, origin)

    def _store() -> None:
        doc = cal._read_gt(clip)
        doc["frames"][str(idx)] = gt_rec
        cal._write_gt(clip, doc)
        def _flip(d):
            d["runs"][run_id]["frames"][str(idx)]["status"] = status

        update_proposals(_gt_marks_dir(), clip, _flip)

    await asyncio.to_thread(_store)
    return {"ok": True, "status": status, "record": gt_rec, "gt_frames": cal._gt_frames(clip)}


@router.post("/calibration/api/proposals/{clip}/{run_id}/{idx}/reject")
async def reject_proposal(clip: str, run_id: str, idx: int, body: dict | None = None):
    if await asyncio.to_thread(_get, clip, run_id, idx) is None:
        return cal._err(404, "no such proposal")
    reason = (body or {}).get("reason")

    def _flip(d):
        f = d["runs"][run_id]["frames"][str(idx)]
        f["status"] = "rejected"
        if reason:
            f["reject_reason"] = reason

    await asyncio.to_thread(update_proposals, _gt_marks_dir(), clip, _flip)
    return {"ok": True, "status": "rejected"}
