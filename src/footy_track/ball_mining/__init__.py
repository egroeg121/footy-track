"""Ball training miner: model-agnostic core (docs/design/ball_sampling.md §3 and §6, Loop 2).

Modules:

``schema``     the candidate *pool table* (one row per candidate node), validator, Parquet I/O and an
               adapter from RT-DETR ``machine_labels`` / ``ball_candidates`` jsonl.
``buckets``    which nodes qualify for the U / M / N / R / A / G buckets (thresholds are config).
``miner``      ``mine_round``: quotas, spill, per-match cap, dedupe, diversity, leakage refusal.
``threshold``  Learn-then-Test auto-accept threshold from A-bucket (audit) verdicts only.
``refit``      the ``ball_mining_refit`` labelling hook: hit rates, Platt fit, LTT, reorder, state file.
``round``      plan / create / status of one round (CLI: ``python -m footy_track.scripts.ball_round``).
"""
