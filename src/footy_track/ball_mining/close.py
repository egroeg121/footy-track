"""Close of a training round (design §6 Loop 2 "Close"): a **library plus CLI that run outside the labeller server**
(``ball_round close --job <ref>``; Prefect flow ``ball-round-close`` in the jobs repo, queued by the job's ``on_done`` hook).

1. a final refit over every verdict (``refit.compute_state(closed=True)``; same verdicts reuse the stored version);
2. the **auto-accept set** for the round's scorer version and threshold: every pool node of that scorer with
   ``p_real >= threshold`` (the Learn-then-Test threshold of the state) that is still train data (split authority, +-10 s
   eval exclusion, not near a human verdict), written to ``<out_root>/<scorer_version>/<match>.jsonl`` with provenance
   ``ball_pool_auto`` and ``tier`` 3. It never writes into GT sidecars or the labeller's marks; the nightly export and
   the feature store read these files as tier-3 positives (loss weight and the 3x-human cap are applied there);
3. ``<marks>/ball_mining/closed/<job_id>.json``: the round is recorded as closed (state version, what was written).
   Closing is idempotent per job: a second run reports ``already_closed`` and writes nothing (``force`` redoes it).

No auto-accept rows are written (and the reason is recorded) when the threshold is only the interim fallback (it is on the
Claude-verify confidence scale, which pool ``p_real`` is not) or when the round's gold check flagged the labeller as
over-accepting (``allow_flagged`` overrides): a threshold learned from suspect verdicts must not admit labels.
"""

from __future__ import annotations

import json
import logging
import os
import re
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from footy_track.ball_mining import miner, refit
from footy_track.ball_mining.buckets import BucketConfig
from footy_track.ball_mining.clips import load_clip_spans, read_verdict_frames
from footy_track.ball_mining.schema import read_pool
from footy_track.clip_id import canonical_clip_id
from footy_track.labeller import jobs, jobs_client

LOGGER = logging.getLogger(__name__)

PROVENANCE = "ball_pool_auto"
TIER = 3
DEFAULT_OUT_ROOT = "/mnt/storage/footy_data/ball_auto"
DEFAULT_POOL = "/mnt/storage/footy_data/ball_pool"
CLOSE_LOCK_WAIT_S = 600.0
DEFAULT_DATA_ROOT = "/mnt/storage/footy_data"
_SAFE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


def out_root() -> Path:
    return Path(os.environ.get("FOOTY_BALL_AUTO_ROOT", DEFAULT_OUT_ROOT))


def closed_dir(marks_dir: str | Path) -> Path:
    return Path(marks_dir) / "ball_mining" / "closed"


def closed_marker(marks_dir: str | Path, job_id: str) -> Path:
    return closed_dir(marks_dir) / f"{job_id}.json"


def auto_accept_set(
    pool: pd.DataFrame,
    *,
    scorer: str,
    threshold: float,
    labelled=(),
    split_of=None,
    match_of=None,
    clip_spans=None,
    eval_clips=None,
    cfg: BucketConfig | None = None,
) -> tuple[dict[str, pd.DataFrame], dict]:
    """``({match: rows}, report)``: the pool nodes of ``scorer`` with ``p_real >= threshold`` that are train data.

    Every match that has train rows of this scorer is a key (possibly with no rows, so a stale file is overwritten empty
    when the threshold rose). ``labelled`` is ``(clip, frame)`` of every live human verdict: their 2 s windows are left to
    the human labels."""
    cfg = cfg or BucketConfig()
    if eval_clips is None and split_of is None:
        eval_clips = miner.default_eval_clips()
    split_of = split_of or miner.default_split_of
    match_of = match_of or miner.default_match_of
    p = pool[pool["scorer_version"] == scorer]
    report = {
        "scorer_version": scorer,
        "threshold": threshold,
        "pool_rows_of_scorer": int(len(p)),
    }
    if p.empty:
        return {}, {**report, "refused": {}, "excluded": {}, "n_accepted": 0}
    gate = miner._Gate(
        split_of=split_of,
        match_of=match_of,
        cfg=cfg,
        clip_spans=clip_spans,
        strict=False,
    )
    gate.register_known_clips(
        set(zip(p["match"], p["clip"], strict=True)) | set(eval_clips or [])
    )
    gate.block_labelled(
        labelled,
        dict(
            zip(p["clip"].map(canonical_clip_id), p["fps"].astype(float), strict=True)
        ),
    )
    clean = gate.apply(p, "auto_accept")
    sel = clean[clean["p_real"] >= threshold]
    out = {m: sel[sel["match"] == m] for m in sorted(clean["match"].unique())}
    return out, {
        **report,
        "refused": gate.report["refused"],
        "excluded": dict(gate.report["excluded"]),
        "n_accepted": int(len(sel)),
    }


def _row(r, *, threshold: float, key: str | None, job_id: str, version: int) -> dict:
    return {
        "match": r["match"], "clip": r["clip"], "frame_index": int(r["frame"]), "fps": float(r["fps"]),
        "bbox": {"x": float(r["x"]), "y": float(r["y"]), "w": float(r["w"]), "h": float(r["h"])},
        "p_real": round(float(r["p_real"]), 6), "scorer_version": r["scorer_version"], "detector_sha": r.get("detector_sha"),
        "provenance": PROVENANCE, "tier": TIER, "threshold": threshold, "round": key, "job": job_id, "state_version": version,
    }  # fmt: skip


def write_auto_accept(
    by_match: dict[str, pd.DataFrame],
    *,
    scorer: str,
    threshold: float,
    root: str | Path,
    key: str | None,
    job_id: str,
    version: int,
) -> dict[str, str]:
    """Write ``<root>/<scorer>/<match>.jsonl`` per match (atomic replace; the same inputs give the same bytes)."""
    if not _SAFE.match(scorer):
        raise ValueError(f"unsafe scorer_version {scorer!r}")
    d = Path(root) / scorer
    paths = {}
    for match, df in by_match.items():
        if not _SAFE.match(str(match)):
            raise ValueError(f"unsafe match key {match!r}")
        rows = [
            _row(r, threshold=threshold, key=key, job_id=job_id, version=version)
            for _, r in df.iterrows()
        ]
        rows.sort(
            key=lambda x: (x["clip"], x["frame_index"], x["bbox"]["x"], x["bbox"]["y"])
        )
        text = "".join(json.dumps(x, sort_keys=True) + "\n" for x in rows)
        path = d / f"{match}.jsonl"
        d.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        tmp.write_text(text)
        os.replace(tmp, path)
        paths[match] = str(path)
    return paths


def _close_round(
    ref: str,
    *,
    client=None,
    pool: str | Path | pd.DataFrame | None = None,
    root: str | Path | None = None,
    marks_dir: str | Path | None = None,
    data_root: str | Path | None = None,
    params: dict | None = None,
    force: bool = False,
    allow_flagged: bool = False,
    split_of=None,
    match_of=None,
    eval_clips=None,
) -> dict:
    """Close the round job ``ref``: final refit, auto-accept files, closed marker (see the module docstring)."""
    client = client or jobs_client
    doc = client.results(ref)
    job = doc["job"]
    marks = refit.resolve_marks(job["dataset"], marks_dir)
    marker = closed_marker(marks, job["id"])
    if marker.is_file() and not force:
        return {
            "job": job["id"],
            "key": job.get("key"),
            "status": "already_closed",
            **json.loads(marker.read_text()),
        }
    if job.get("status") != "done" and not force:
        raise RuntimeError(
            f"job {job['id']} is {job.get('status')}, not done: close runs after the last card is answered (--force to override)"
        )
    rows = refit.collect_rows(client, job["dataset"], current=doc)
    state, cal = refit.compute_state(doc, rows, marks=marks, closed=True, params=params)
    aa = state["auto_accept"]
    scorer = state["scorer_version"]
    out: dict = {
        "threshold": aa["threshold"],
        "scale": aa["scale"],
        "fallback": aa["fallback"],
        "scorer_version": scorer,
        "written": False,
    }
    why = None
    if scorer is None:
        why = "the round has no scorer_version"
    elif aa["fallback"] or aa["scale"] != "p_real":
        why = "no valid Learn-then-Test threshold yet (the interim 0.95 is on the claude_verify_conf scale, not the pool p_real)"
    elif state["gold"]["flagged"] and not allow_flagged:
        why = f"gold check flagged the round (planted-wrong cards accepted {state['gold']['accept_rate']:.0%})"
    if why is None:
        pdf = pool if isinstance(pool, pd.DataFrame) else _read(pool or DEFAULT_POOL)
        if pdf is None:
            why = f"no pool at {pool or DEFAULT_POOL}"
        else:
            root_ = root or out_root()
            dr = str(data_root or os.environ.get("FOOTY_DATA_ROOT", DEFAULT_DATA_ROOT))
            spans = load_clip_spans(dr) if Path(dr).is_dir() else {}
            by_match, rep = auto_accept_set(pdf, scorer=scorer, threshold=float(aa["threshold"]), labelled=read_verdict_frames(marks),
                                            split_of=split_of, match_of=match_of, clip_spans=spans, eval_clips=eval_clips)  # fmt: skip
            paths = write_auto_accept(by_match, scorer=scorer, threshold=float(aa["threshold"]), root=root_, key=job.get("key"),
                                      job_id=job["id"], version=state["version"])  # fmt: skip
            out.update(
                written=True,
                root=str(root_),
                files={
                    m: {"path": p, "n": int(len(by_match[m]))} for m, p in paths.items()
                },
                **rep,
            )
    if why:
        out["reason"] = why
        LOGGER.warning("close %s: no auto-accept written: %s", job["id"], why)
    record = {"closed_ts": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"), "key": job.get("key"), "state_version": state["version"],
              "n_verdicts": state["n_verdicts"], "auto_accept": out}  # fmt: skip
    jobs._write_atomic(marker, record, exclusive=False)
    return {
        "job": job["id"],
        "key": job.get("key"),
        "status": "closed",
        **refit.summary(state, cal),
        "auto_accept": out,
    }


def close_round(ref: str, *, client=None, marks_dir: str | Path | None = None, lock_wait_s: float = CLOSE_LOCK_WAIT_S, **kw) -> dict:
    """Close the round job ``ref`` under the per-job mutex shared with the refit and the anchor search (design 7.4); waits up
    to ``lock_wait_s`` for a run in progress (``refit.JobBusy`` after that: the close is not skipped silently)."""
    client = client or jobs_client
    job = client.results(ref)["job"]
    with refit.job_lock(refit.resolve_marks(job["dataset"], marks_dir), job["id"], wait_s=lock_wait_s):
        return _close_round(ref, client=client, marks_dir=marks_dir, **kw)


def _read(path: str | Path) -> pd.DataFrame | None:
    try:
        return read_pool(path)
    except FileNotFoundError:
        return None
