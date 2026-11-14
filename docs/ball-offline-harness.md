# Offline ball-tracking harness

A frozen-label scoreboard for ball trackers that work on whole clips, so a tracker change
can be measured against `main` on the same labels, the same clean frames and the same
metrics. Code: `src/footy_track/ball_offline/`, tests `tests/ball_offline/`. Reproduction
numbers: `docs/experiments/offline_ball_harness_baseline.md`.

Reference trackers (`baselines.py`):

- **b0**: top-1 detector candidate per frame if its confidence is at least 0.4.
- **viterbi**: `ball_trackers/viterbi.py::ViterbiBallTracker` with its default settings, fed
  the clip's candidates exactly as the inference pipeline does.

Not in this harness: the experiment branches' own B1 solver (`exp/offline-ball-tracking`
fc05732, recall 0.759 / precision 0.985 on the same clean test set, statistically
indistinguishable from main's Viterbi), the SAM2 gap fill (`exp/sam2-bidir-ball`) and the
LK+RANSAC camera model (`exp/calibration-temporal`; "no worse, not better" than the player-box
camera on the labelled clips, and main's tracker has no camera input). Each would plug in as
one more tracker jsonl.

## Ground truth, clean frames, roles

GT = `in_play_ball` marks in a labeller sidecar directory that are human-placed (`labeller`
tag) or machine-propagated and human-checked (`vittrack` + `human_checked`). Machine-only
marks, `out_of_play_ball` and `not_broadcast` frames are not scored; `no_ball` frames score a
false positive if a tracker emits a point there. All alias files of a clip are merged
(`clip_id.canonical_clip_id`).

A GT frame is **contaminated** if it is within 5 frames of an image in the Roboflow v11/v3
train or valid split (the detector `rtdetr_1920_v11` has effectively seen it). A clip is
**dev** if at least half of its GT ball frames are contaminated, otherwise **test**. Only
clean frames of test clips are the headline; dev is shown flagged as optimistic.

The GT source is pluggable (`data.GTSource`): `MarksDirGT(dir)` reads any labeller output
directory (the one the labeller was started with, `FOOTY_GT_MARKS_DIR`), and `SnapshotGT`
reads a frozen snapshot. Named eval sets (`sources.py`):

| set | GT dir | candidates | frame counts |
|---|---|---|---|
| `mancity` | `ball_gt_marks/` | `machine_labels/` | its `_manifest.json` |
| `norwich` (held-out) | `ball_gt_marks_norwich/` | `ball_candidates_norwich/` | `machine_labels_norwich/_manifest.json` |

`--gt-dir`, `--candidates-dir`, `--manifest-dir` override any of them (another `--eval-set`
name needs `--gt-dir` and `--candidates-dir`).

## Frozen snapshots

```bash
python -m footy_track.ball_offline snapshot --eval-set mancity      # or norwich, or --gt-dir DIR
# snapshot  /mnt/storage/footy_data/ball_eval_snapshots/mancity-20261114-c9cc63ecbef0
```

This copies the current GT of the eval set into `<root>/<set>-<YYYYMMDD>-<hash12>/`
(`manifest.json` + `gt/<clip>.json` with points, `no_ball` frames, clip role and
contaminated frames; files are read-only). The root is `$FOOTY_BALL_SNAPSHOTS` or
`$FOOTY_DATA/ball_eval_snapshots`. The hash covers only the scoring-relevant content, so
snapshotting unchanged labels again returns the same directory; `load_snapshot` recomputes
it and refuses an edited snapshot. The manifest also records the sha256 of each source
sidecar, the candidate directory and its fingerprint, and the harness commit. Every report
(`score`, `verdicts`) prints and stores the snapshot hash.

Re-snapshot when labels change and say so in the PR; compare a PR and `main` only on the
same hash. Snapshots refer to detector candidates by directory, so a re-run of the detector
(new `candidates_fingerprint`) is a different measurement even on the same hash.

## Scoring any tracker

```bash
python -m footy_track.ball_offline predict --snapshot c9cc63ecbef0 --method viterbi --out viterbi.jsonl
python -m footy_track.ball_offline score --snapshot c9cc63ecbef0 --pred my_tracker.jsonl \
    --baseline viterbi.jsonl [--block 25] [--roles test,dev] [--per-clip] [--out-json r.json]
```

Tracker output is jsonl, one row per frame with a ball:
`{"clip": "arsenal_mancity_seg010", "frame_index": 412, "x": 1031.5, "y": 402.0}`
(pixels at 1920x1080; `--units norm` or a labeller-style `"center": {"x","y"}` for
normalised coordinates; `x: null` or an absent frame = no ball). A snapshot clip with no rows
is listed as a warning and counts as all-missed.

Metrics (pixel radius R = 5/10/20): recall = TP / GT ball frames, precision = TP / (TP + FP)
over scored frames (GT ball frames and `no_ball` frames; unlabelled frames are never
penalised, so absolute precision is optimistic), F1, coverage, median error, and 95% Wilson
intervals. With `--baseline`, a **paired bootstrap** resamples the scored frames (or blocks of
`--block` frames) and reports the difference in recall@10 and precision@10 with 95% CIs and
the number of frames only one tracker hits. Labels are sparse and clustered (two clips hold
163 of the 170 clean test frames), so treat the CIs as optimistic.

`ceiling` prints the detector upper bound (a candidate >= 0.1 within R px of the GT).
`verdicts` cross-checks trackers against Ball Check verdicts (`<gt dir>/ball_checks/`),
candidate-level and independent of the GT frame selection; `predict --include-verdict-clips`
also runs the tracker on clips that only have verdicts.

## Adding a tracker

Write its per-frame output as jsonl and `score` it against the snapshot. To run it
in-process instead, add a `fn(ClipData) -> {frame: (x, y)}` to `baselines.METHODS`.
