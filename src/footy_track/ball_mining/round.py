"""One round of the training miner: plan it, create the Ball Check job, report status (design §6 Loop 2).

``plan``   mine the pool and return the cards + report; nothing is written to the labeller.
``create`` tick rule (open training cards < 100 and today's cards < 600), then submit one ``ball_check`` job with
           idempotent key ``ball-round/<date>/<n>`` and two ``prefect`` hooks attached (``ROUND_HOOKS``): every 50 verdicts the
           labeller queues ``ball-refit/ball-refit``, on completion ``ball-round-close/ball-round-close``. The flows (jobs repo
           ``footy_ball_round.py``) run ``ball_round refit`` / ``close`` (``refit.py`` / ``close.py``); the labeller does no work.
``status`` open / today counts, the tick decision and the latest state file.

The labeller is reached through ``jobs_client`` (stdlib HTTP, ``$FOOTY_LABELLER_URL``); pass another object with the
same ``submit`` / ``list_jobs`` / ``get`` functions to test. Marks (verdicts, gold, state) are read from the dataset's
marks directory on disk, which is where the labeller writes them.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
from collections.abc import Iterable
from dataclasses import dataclass, field, replace
from datetime import date as date_cls
from pathlib import Path

import pandas as pd

from footy_track.ball_mining import refit as refit_mod
from footy_track.ball_mining import traindata
from footy_track.ball_mining.buckets import DEFAULT_QUOTAS, BucketConfig
from footy_track.ball_mining.clips import load_clip_spans, read_verdict_frames
from footy_track.ball_mining.gold import load_gold_multi
from footy_track.ball_mining.miner import MineResult, default_match_of, mine_round
from footy_track.ball_mining.schema import pool_from_machine_labels, read_pool

KEY_PREFIX = "ball-round"
OPEN_CARDS_MAX = 100
DAILY_CARDS_MAX = 600
DEFAULT_DATASET = traindata.TRAIN_DATASET  # the round's own dataset: clips linked by ``create_round``, verdicts land in its marks
DEFAULT_DATA_ROOT = "/mnt/storage/footy_data"

#: Hooks attached to every round job (``hooks.validate_hooks`` shape). Both are ``prefect`` actions: the labeller only queues
#: the deployment, with ``{job_id, hook, trigger, dataset, labeller_url?}`` (the job is named ``job_id``, there is no way to
#: add the job ref to ``params`` at submit time since the id does not exist yet); the flows take those and run the CLI.
REFIT_DEPLOYMENT = "ball-refit/ball-refit"
CLOSE_DEPLOYMENT = "ball-round-close/ball-round-close"
ROUND_HOOKS = [
    {"name": "refit", "trigger": {"every_n": 50, "min_interval_s": 60},
     "action": {"type": "prefect", "deployment": REFIT_DEPLOYMENT, "params": {}}},
    {"name": "close", "trigger": {"on_done": True},
     "action": {"type": "prefect", "deployment": CLOSE_DEPLOYMENT, "params": {}}},
]  # fmt: skip


@dataclass
class RoundConfig:
    dataset: str = DEFAULT_DATASET
    budget: int = 300
    quotas: dict = field(default_factory=lambda: dict(DEFAULT_QUOTAS))
    per_match_cap: float = 0.15
    #: ``None`` (default): derived from the round key by :func:`seed_for_round`, so every round draws differently.
    seed: int | None = None
    open_cards_max: int = OPEN_CARDS_MAX
    daily_cards_max: int = DAILY_CARDS_MAX
    data_root: str = field(
        default_factory=lambda: os.environ.get("FOOTY_DATA_ROOT", DEFAULT_DATA_ROOT)
    )
    bucket: BucketConfig = field(default_factory=BucketConfig)
    use_gold: bool = True
    #: Registry datasets the gold GT (human ball boxes, ``not_ball`` verdicts) is read from, and whose clips dirs the gold
    #: cards' videos come from. The round dataset's own marks are always read too.
    gold_datasets: tuple[str, ...] = ("calib",)
    #: Link every card's clip into the dataset's clips dir (and register ``ball_train_v1``) before submitting.
    link_clips: bool = True


def seed_for_round(round_key: str | None) -> int:
    """Deterministic per-round seed: the first 8 hex digits of ``sha1(round_key)`` (0 when there is no key)."""
    if not round_key:
        return 0
    return int(hashlib.sha1(round_key.encode()).hexdigest()[:8], 16)


def marks_dir(dataset: str, root: str | Path | None = None) -> Path:
    """The dataset's marks dir from the registry; the training dataset not registered yet (a ``plan`` before the first
    ``create``) resolves to the path ``create`` will register."""
    from footy_track.labeller import datasets  # noqa: PLC0415

    try:
        return datasets.get(dataset).marks_dir
    except datasets.UnknownDatasetError:
        if dataset != traindata.TRAIN_DATASET:
            raise
        return traindata.marks_dir(root or os.environ.get("FOOTY_DATA_ROOT", DEFAULT_DATA_ROOT), dataset)


def own_clips_dir(cfg: RoundConfig) -> Path:
    from footy_track.labeller import datasets  # noqa: PLC0415

    try:
        return datasets.get(cfg.dataset).clips_dir
    except datasets.UnknownDatasetError:
        if cfg.dataset != traindata.TRAIN_DATASET:
            raise
        return traindata.clips_dir(cfg.data_root, cfg.dataset)


def other_marks_dirs(cfg: RoundConfig, marks: Path) -> list[Path]:
    """Marks dirs of every other registered dataset (calib, nottinghamforest*, ...): verdicts given there, on the same
    physical clip, mean the card was already asked, so it is not asked again."""
    from footy_track.labeller import datasets  # noqa: PLC0415

    own = Path(marks).resolve()
    return [d.marks_dir for d in datasets.all_datasets() if d.name != cfg.dataset and d.marks_dir.resolve() != own]


def gold_resolver(cfg: RoundConfig) -> traindata.ClipResolver:
    """Where a gold card's video is found: the datasets its GT came from first, then the round dataset's own clips dir
    (a ``not_ball`` verdict of an earlier round is on a pool clip, already linked there)."""
    dirs = traindata.gold_dataset_dirs(cfg.gold_datasets)
    with contextlib.suppress(KeyError):  # an unregistered dataset other than the training one: no own clips dir
        dirs.append(own_clips_dir(cfg))
    return traindata.ClipResolver(dirs)


# ---------------------------------------------------------------------------
# Tick rule and bookkeeping (pure functions over the job list)
# ---------------------------------------------------------------------------


_is_round = refit_mod.is_round_job


def _cards(j: dict) -> tuple[int, int]:
    """``(total, remaining)`` cards of a listed job."""
    p = j.get("progress") or {}
    total = p.get("total")
    if total is None:
        total = j.get("n_items") or 0
    done = p.get("done") or 0
    return int(total), int(total) - int(done)


def tick_decision(
    job_list: Iterable[dict],
    today: str,
    *,
    open_max: int = OPEN_CARDS_MAX,
    daily_max: int = DAILY_CARDS_MAX,
) -> dict:
    """Create a round only if open training cards < ``open_max`` and cards created today < ``daily_max``."""
    open_cards = today_cards = 0
    for j in job_list:
        if not _is_round(j) or j.get("status") == "cancelled":
            continue
        total, remaining = _cards(j)
        if j.get("status") in ("open", "in_progress", "preparing"):
            open_cards += remaining
        if str(j["key"]).startswith(f"{KEY_PREFIX}/{today}/"):
            today_cards += total
    reason = None
    if open_cards >= open_max:
        reason = f"{open_cards} open training cards >= {open_max}"
    elif today_cards >= daily_max:
        reason = f"{today_cards} cards already created today >= {daily_max}"
    return {
        "allowed": reason is None,
        "reason": reason,
        "open_cards": open_cards,
        "today_cards": today_cards,
        "date": today,
    }


def next_key(job_list: Iterable[dict], today: str) -> str:
    prefix = f"{KEY_PREFIX}/{today}/"
    ns = [
        int(str(j["key"])[len(prefix) :])
        for j in job_list
        if str(j.get("key") or "").startswith(prefix)
        and str(j["key"])[len(prefix) :].isdigit()
    ]
    return f"{prefix}{max(ns, default=0) + 1}"


# ---------------------------------------------------------------------------
# Plan
# ---------------------------------------------------------------------------


def load_pool(
    pool: str | Path | None = None,
    machine_labels: Iterable[str | Path] = (),
    *,
    match: str | None = None,
    fps: float = 25.0,
    derive_static: bool = True,
) -> pd.DataFrame:
    """A pool from Parquet (``pool``) and / or adapted RT-DETR jsonl (``machine_labels``: files or directories)."""
    frames = []
    if pool:
        frames.append(read_pool(pool))
    ml = list(machine_labels)
    if ml:
        frames.append(
            pool_from_machine_labels(
                ml, match=match, fps=fps, derive_static=derive_static
            )
        )
    if not frames:
        raise ValueError("give a pool Parquet and / or machine_labels jsonl")
    return pd.concat(frames, ignore_index=True) if len(frames) > 1 else frames[0]


def pending_frames(job_list: Iterable[dict], client) -> set[tuple[str, int]]:
    """``(clip, frame)`` of every card of every non-cancelled round job: pending ones (not yet verdicted) and the skipped /
    unanswered cards of finished rounds were all *shown* to the labeller, so their neighbourhood is not mined again either."""
    out: set[tuple[str, int]] = set()
    for j in job_list:
        if _is_round(j) and j.get("status") != "cancelled":
            full = client.get(j["id"])
            out.update(
                (it["clip"], int(it["frame_index"])) for it in full.get("items") or []
            )
    return out


def plan_round(
    pool: pd.DataFrame,
    cfg: RoundConfig,
    *,
    job_list: list[dict] | None = None,
    client=None,
    today: str | None = None,
    round_key: str | None = None,
    marks: Path | None = None,
) -> MineResult:
    """Mine one round for ``cfg.dataset``. Excludes frames with a live verdict (tombstones honoured) and cards pending
    in live round jobs; uses the dataset's tier-1 GT / not_ball verdicts as gold."""
    marks = marks or marks_dir(cfg.dataset, cfg.data_root)
    labelled = read_verdict_frames(marks)
    for d in other_marks_dirs(cfg, marks):
        labelled |= read_verdict_frames(d)
    if job_list is not None and client is not None:
        labelled |= pending_frames(job_list, client)
    gold = None
    gold_unresolved: list[str] = []
    if cfg.use_gold:
        from footy_track.labeller import datasets  # noqa: PLC0415

        gold_marks = [marks]
        for n in cfg.gold_datasets:
            try:
                gold_marks.append(datasets.get(n).marks_dir)
            except datasets.UnknownDatasetError:
                continue
        gold = load_gold_multi(gold_marks, match_of=lambda c: default_match_of(c) or c)
        if cfg.link_clips and len(gold):  # a gold card whose video cannot be found would not render: leave it out
            resolver = gold_resolver(cfg)
            ok = gold["clip"].map(lambda c: resolver.find(c) is not None)
            gold_unresolved = sorted(set(gold.loc[~ok, "clip"]))
            gold = gold[ok].reset_index(drop=True)
    spans = load_clip_spans(cfg.data_root) if Path(cfg.data_root).is_dir() else {}
    seed = cfg.seed if cfg.seed is not None else seed_for_round(round_key)
    res = mine_round(pool, budget=cfg.budget, quotas=cfg.quotas, per_match_cap=cfg.per_match_cap, already_labelled=labelled,
                      seed=seed, cfg=cfg.bucket, gold=gold, clip_spans=spans, round_key=round_key)  # fmt: skip
    if gold_unresolved:
        res.report["gold_unresolved_clips"] = gold_unresolved
    return res


def job_body(result: MineResult, key: str, cfg: RoundConfig) -> dict:
    """The ``POST /api/jobs`` body: inline ball_check cards plus the prefect refit / close hooks."""
    n = len(result.cards)
    body = {
        "title": f"Ball mining {key.split('/', 1)[1]}",
        "kind": "ball_check",
        "dataset": cfg.dataset,
        "items": result.cards,
        "key": key,
        "why": f"Training miner round ({n} cards; quotas {cfg.quotas}); verify cards, see docs/design/ball_sampling.md",
        "hooks": ROUND_HOOKS,
    }
    if n > 500:
        body["max_cards"] = n
    return body


def create_round(
    pool: pd.DataFrame,
    cfg: RoundConfig,
    *,
    client=None,
    today: str | None = None,
    force: bool = False,
    dry_run: bool = False,
    prepare: bool = False,
    marks: Path | None = None,
) -> dict:
    """Apply the tick rule, mine, submit. Returns ``{"created": bool, ...}`` with the decision and the report."""
    if client is None:
        from footy_track.labeller import jobs_client as client  # noqa: PLC0415
    today = today or date_cls.today().isoformat()
    job_list = client.list_jobs(kind="ball_check", progress="true")
    tick = tick_decision(
        job_list, today, open_max=cfg.open_cards_max, daily_max=cfg.daily_cards_max
    )
    if not tick["allowed"] and not force:
        return {"created": False, "tick": tick}
    key = next_key(job_list, today)
    result = plan_round(
        pool,
        cfg,
        job_list=job_list,
        client=client,
        today=today,
        round_key=key,
        marks=marks,
    )
    if not result.cards:
        return {
            "created": False,
            "tick": tick,
            "reason": "no cards to mine",
            "report": result.report,
        }
    link_report = None
    if cfg.link_clips:
        # Registered and linked before the submit (the labeller 404s an unknown dataset and cannot render an unlinked clip).
        # A dry run does both too: they are idempotent, and it is what proves the cards would load.
        if cfg.dataset == traindata.TRAIN_DATASET:
            traindata.register_dataset(cfg.data_root, cfg.dataset)
        cards, link_report = traindata.link_cards(result.cards, own_clips_dir(cfg), data_root=cfg.data_root, resolver=gold_resolver(cfg))
        result = replace(result, cards=cards)
    body = job_body(result, key, cfg)
    if dry_run:
        return {
            "created": False,
            "dry_run": True,
            "tick": tick,
            "key": key,
            "report": result.report,
            "body_n_items": len(result.cards),
            "links": link_report,
            "cards": result.cards,
        }
    job = client.submit(body["title"], "ball_check", body["items"], dataset=cfg.dataset, why=body["why"], key=key, hooks=body["hooks"],
                        max_cards=body.get("max_cards"), prepare=prepare, requester=None, device="phone")  # fmt: skip
    log = (
        (marks or marks_dir(cfg.dataset, cfg.data_root))
        / "ball_mining"
        / "rounds"
        / (key.replace("/", "_") + ".json")
    )
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text(json.dumps({"key": key, "job": job.get("id"), "config": {"budget": cfg.budget, "quotas": cfg.quotas, "seed": result.report.get("seed"),
                                                                "exclude_s": cfg.bucket.exclude_s, "max_per_clip": cfg.bucket.max_per_clip},
                               "report": result.report}, indent=1, default=str))  # fmt: skip
    return {
        "created": bool(job.get("created", True)),
        "job": job,
        "key": key,
        "tick": tick,
        "report": result.report,
        "links": link_report,
        "log": str(log),
    }


def status(cfg: RoundConfig, *, client=None, today: str | None = None) -> dict:
    if client is None:
        from footy_track.labeller import jobs_client as client  # noqa: PLC0415
    today = today or date_cls.today().isoformat()
    job_list = client.list_jobs(kind="ball_check", progress="true")
    rounds = [
        {
            "key": j["key"],
            "id": j["id"],
            "status": j["status"],
            "cards": _cards(j)[0],
            "remaining": _cards(j)[1],
        }
        for j in job_list
        if _is_round(j)
    ]
    return {"tick": tick_decision(job_list, today, open_max=cfg.open_cards_max, daily_max=cfg.daily_cards_max), "rounds": rounds,
            "state": refit_mod.latest_state(marks_dir(cfg.dataset, cfg.data_root))}  # fmt: skip
