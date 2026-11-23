# Full-frame labelling: complete frames, progress, auto-label series

Status: Tasks 1-10 implemented and merged 2026-11-19 (PRs #107, #109-#118); the pieces are untested against real models and the real store. Original proposal 2026-11-19. KB relied on: `footy-claude-autolabel-loop` (Claude verify precision, queue), `footy-datasets-and-progress` (store, splits, `/progress`, Prefect nightly), `footy-labeller-gotchas` (sidecar data loss, `_segNNN` clips not contiguous). Builds on `ball_sampling.md`, `feature_store.md`.

## Decision 2026-11-21: two detectors (supersedes the single multi-class student)

Decision (user, 2026-11-21): train **two separate detectors** instead of one multi-class RT-DETR.
- **Ball detector**: `ball` / `in_play_ball` / `out_of_play_ball`. High-resolution or tiled (can run on crops/tiles), its own hard negatives (Ball Check `not_ball`), evaluated on the ball eval panel.
- **Player-side detector**: `player`, `referee`, `coach`, `player_sub`, `person`.

Why: labelling is already split (Ball Check / ball jobs vs Player Check); the ball is tiny and needs different resolution and evaluation; each model gets its own teacher version, targeted queue and eval; the pipeline is easier to label and train.
Costs: two inference passes (the ball model can run on crops/tiles); ball false positives lose player context, mitigated by Ball Check hard negatives.

Consequences:
1. **`frame_complete` (both groups complete) stops being the training gate.** The ball model trains on frames whose *ball* state is `complete`/`absent`; the player model on frames whose *player* state is `complete`/`absent`. `frame_complete` / `fully_labelled` stay as an informational view. Class ids stay frozen (`constants.py`); the player export simply excludes the ball classes.
2. **Progress** (`/progress/labelling`) should show two independent targets: ball-complete and player-complete; "fully labelled" is informational only.
3. **Auto-label pipeline** (windows, propose, pre-verify, series review, teacher versions, targeting queue; sections 4-4b) runs **per group**: a ball pipeline and a player pipeline, each with its own teacher version and queue.
4. **Eval**: the ball eval panel for the ball model; blind hand-labelled heldout frames (the player eval panel, section 4c.6b) for the player model.

Status:
| Part | State |
|---|---|
| Ball-only export (default, no `--classes`; gate = ball group only; ball classes collapse to one class id 0) | Implemented (`scripts/export_training_dataset.py`) |
| `frame_label_state` per group, per-group writers, frozen class ids | Implemented |
| Player-only export mode (player-group gate, ball classes excluded, frozen ids unchanged) | TODO. Today `--classes` multi-class export still gates on `frame_complete` |
| `/progress/labelling` as two independent targets | Implemented 2026-11-21: `groups.ball` / `groups.player` (section 3), two sections on `/progress`, ball and player badges in the clip list |
| Per-group auto-label pipeline and per-group teacher versions / queues | TODO. Stages in section 4 are built as one pipeline for both groups |
| Ball model training recipe (high-res/tiled) and player model training run | TODO (not covered by this doc) |
| Player eval panel (blind heldout frames) | TODO (see 4c.6b) |

Where sections below say "one detector", "multi-class" or that fully labelled frames are the training target, that text is **superseded** by this decision; it is kept for history and marked inline.

## 1. Fully labelled frame

A frame is **fully labelled** when each class group (`ball`, `player`) has state `complete` or `absent`:
- `ball`: `complete` = a verified ball box exists; `absent` = explicit no_ball (sidecar `no_ball` tag or Ball Check frame verdict). `unsure` and unreviewed are neither.
- `player` (player, player_sub, referee, coach, person): `complete` = a human asserted "all visible people are boxed". A zero-person frame is `absent`.

**Storage: explicit table, not a derived view.** `reviewed` boxes prove only that those boxes were checked, never that no other player is missing, so player completeness cannot be derived. Derivation also makes the ball side depend on JSONL verdict logs the store does not hold.

```
frame_label_state(game_id, frame_index, cls_group, state,  -- complete|absent|partial|unreviewed
                  method, run_id, verified_by, verified_at, PRIMARY KEY(game_id, frame_index, cls_group))
```
- `method`: `hand_series` (accept-all), `hand_frame`, `verdict`, `offline_audited`, `sidecar_marker`, `inferred_gt` (rank 0, lowest).
- `run_id` points at the proposing run (provenance).
- View `frame_complete(game_id, frame_index)` = both groups in (`complete`,`absent`). Absence of a row means `unreviewed`.
- Writers: labeller flush writes `no_ball`, plus `complete` for ball when a human box exists. Ball Check verdicts write ball state. The series review (section 4) writes both groups. Existing GT is NOT back-filled as player-complete, except clips the user flags (`--legacy-complete-clips`) and the interim `inferred_gt` rule (section 1b).
- Missing label never implies background. Only `complete`/`absent` frames reach training. *(Superseded 2026-11-21: per group. Ball-model frames need ball `complete`/`absent`, player-model frames need player `complete`/`absent`; both groups are no longer required.)*


### 1b. Interim rule: inferred player-complete (`inferred_gt`, user decision 2026-11-23)

Pragmatic override of "existing GT is not assumed player-complete": while full-frame player labelling catches up, a frame that already has
>= 1 **human** box (`labeller` provenance or human-checked) of a player-side class (player, referee, coach, player_sub, person) **and** a ball
decision (human ball box, or a `no_ball` mark) gets player `complete` with `method=inferred_gt`, verified_by `inferred_gt`, run_id
`gt_import_inferred_player_complete`. Not marked: players without a ball decision, a ball decision without players, machine-only players,
`not_broadcast` frames, ball/no_ball conflicts. Sparse player labels are likely incomplete, so this is a known approximation.

- `inferred_gt` has the lowest precedence (rank 0, below `sidecar_marker`): any `P` marker, legacy back-fill, verdict, Player Check or hand label on the frame wins, and removing that stronger row lets the inferred row return on the next ingest.
- Re-derived on every replace ingest of a clip (delete this method's rows, then upsert): idempotent; a frame whose player boxes vanish loses its row.
- ON by default in `ingest_all` (nightly); `--no-infer-player-complete` disables it and the next ingest purges the rows. OFF by default in `ingest_gt` (`--infer-player-complete`).
- Revert/filter by method: `DELETE FROM frame_label_state WHERE method='inferred_gt'`, or exclude `method='inferred_gt'` in queries. Progress shows it under `by_method` and in the `other` tier (not gold or silver).
- Existing stores are migrated on open (the `method` CHECK constraint is rebuilt once).

## 2. Exporter (`scripts/export_training_dataset.py`)

- *(Superseded 2026-11-21: the gate is per model group, not `frame_complete`; ball-only mode already does this, player-only mode is TODO.)* Query joins `frame_complete`; only those frames are emitted. Frames with partial state are skipped, with a manifest count.
- Positives are canonical boxes of reviewed sources. Negatives: ball `absent` frames are written with an empty ball label set. Ball Check `not_ball` boxes are kept as `hard_negatives.json` (and optional empty-region crops), never as a class.
- Class ids are stable: a fixed list in `constants.py` (`player=0, ball=1, referee=2, ...`) replaces sorted-label ids. The manifest records the list, and a test fails if ids change.
- `--sources` defaults to `hand_label,reviewed` (machine boxes enter only via `reviewed=true`). `--sources all` remains explicit.
- Train/val comes from `splits.split_of` only (existing guard test applies). Manifest reports frames by state, split and class.

## 3. Progress view

`GET /progress/labelling` presents two independent targets, **Ball** (`ball`, `in_play_ball`, `out_of_play_ball`) and **Player-side** (`player`, `referee`, `coach`, `player_sub`, `person`); their counts are never pooled. Response: `groups.<ball|player>` = `{label, classes, totals, by_split, by_method, tiers, clips}`, where `totals`, each `by_split[split]` and each per-clip row hold `frames`, `complete`, `absent`, `partial`, `unreviewed` (a frame with no row for the group is `unreviewed`), `labelled` (= complete + absent), `clip_frames` and `pct_labelled` (of the clip's video frames when known, else of frames with a state row). `by_split` keys are `train` / `eval_in_match` / `eval_heldout_match`; per-clip `clips` are sorted least-labelled first per group. The legacy keys stay for compatibility: per clip and in `totals`, `frames`, `ball_labelled`, `player_labelled`, `fully_labelled` (both groups; **informational only, not a training gate**), `partial`, plus top-level `by_method`, `by_split`, `tiers`, `teacher_versions`; each legacy clip row also carries `groups.<g>` counts. It reads the store (not the JSONL), so it can run on the mini. The "Labelling coverage" table on `/progress` is two sections (Ball, Player-side) plus an informational fully-labelled line, and the clip list shows separate ball and player badges (the old `_clip_completion` tick remains a sidecar heuristic, not verified progress).

## 4. Auto-label pipeline (priority)

*(2026-11-21: runs per group, one ball pipeline and one player pipeline, each with its own teacher version and queue. The text below describes the combined pipeline as built and is otherwise unchanged. TODO for the split.)*

Stages, each writing a run row and resumable:

1. **Select windows** (mini CPU). Contiguous windows of N=250 frames (5 s at 50 fps) taken inside one `_segNNN` clip, since segments cannot be chained. Choose train-match clips only, balanced across matches, favouring windows with lowest ball-tracker agreement or fewest labels. Heldout matches are excluded (rule zero in `ball_sampling.md`). Output: `windows.json`.
2. **Propose** (desktop GPU, Prefect `gpu` pool, existing `stage-inputs` then GPU then `push-outputs` chain). Per frame: RT-DETR for players and referees and ball candidates (its player mAP50 of ~0.98 comes from the original one-game dataset and says nothing about other matches, camera angles or kits, so player proposals are treated as unmeasured), then ROI-YOLO or ViTTrack bidirectionally for the ball (forward/backward disagreement is a flag). SAM3 is used only to fill missing person boxes when the detector count disagrees with its neighbours. Output: `proposals/<window>.npz` plus `ingest` as `detection` rows (`source=yolo|vittrack|sam3`, `reviewed=false`, `needs_review=true`).
3. **Pre-verify** (mini, Claude verify v2.2 as in the queue). Ball verdict on the tracker candidate (precision 61/62 held-out). Mark frames `auto_ok` when the track is smooth, forward and backward trackers agree, the verdict is `ball` at high confidence, and the person count is stable within the window. Everything else is `flagged`. The auto-accept threshold must come from audits (target 0.95; 0.8 measured at about 65%).
4. **Review** (human, phone/desktop). A new window card shows a 5 s contact strip of every k-th frame with boxes overlaid, with "Accept all", and per-frame exceptions: tap a frame, then one of: fix box (existing `/review/correct`), ball not_ball, ball absent, "players incomplete" (frame is demoted to `partial`). Flagged frames come first, using the Viterbi-margin order. 10% of `auto_ok` frames are shown blind as an audit. If exceptions exceed 10%, the window is rejected and returns to stage 2 with the corrected anchors as re-track seeds.
5. **Commit** (mini). Accept writes `reviewed=true` on accepted machine boxes, `frame_label_state` rows with `method=hand_series` and `run_id` of the proposing run, and the window's audit stats. Boxes the human touched become `hand_label`. Windows with every frame complete are registered as a **series** (`series(game_id,start,end,run_id)`), which the exporter can use for temporal training.

GPU: stage 2 only. CPU: selection, verify, commit, export. The nightly `footy-nightly-datasets` ingests, then bumps dataset versions.

### Ball anchor iteration (design 2026-11-21)

A confirmed ball can anchor a forward/backward search that proposes the next most likely ball cards, checked again in Ball Check, until every selected frame is terminal (`ball`, `no_ball`, `unusable`, explicit `deferred`). The same flow serves the train windows and the ball eval panel (anchoring bias accepted); a frame with no confirmed candidate is never defaulted to `no_ball` but goes to an explicit full-frame human check (laptop job mode or a phone full-frame card), which is how false negatives are found. Train and eval frame sets stay disjoint by `split_of`, eval labels never feed training, anchors propagate only inside one job's selected set, and every card/label records the generator/teacher version. See [ball_anchor_iteration.md](ball_anchor_iteration.md) (design only, nothing built; includes the orchestration/sync rules between labeller jobs, Prefect flows and `frame_label_state`).

## 4b. Offline teacher pipeline (framing for sections 4 and 5)

Two separate systems, deliberately:
- **Offline pipeline ("teacher")**: a versioned set of heavy models with no real-time limit that auto-labels training data. Members: large detector at high resolution plus test-time augmentation, bidirectional ROI-YOLO/ViTTrack ball tracking, SAM3 for missing people, temporal smoothing/Viterbi, Claude verify. It is a pipeline version (`offline_vN`), recorded on every label it writes (`source=offline_vN`, `run_id`).
- **Online detector ("student")**: the RT-DETR that must run fast. *(Superseded 2026-11-21: there are two students, a ball detector and a player-side detector, each trained on frames complete for its own group.)* Originally: one model trained only on fully labelled frames (section 1).

Label tiers: **gold** = human-verified (hand label, accept-all review, exceptions); **silver** = teacher output that passed the `auto_ok` rules and the blind audit (`method=offline_audited`). Each student trains on gold + silver of its own group. The teacher is scored, and the student evaluated, on gold only (held-out matches, never silver).

**Human labelling is targeted, not uniform.** Every labelling request must be one of:
1. *Improves the teacher*: frames where teacher members disagree (forward/backward tracker, detector vs SAM3 person count), flagged frames, audit failures, and hard cases it gets wrong (small/occluded ball, crowded boxes). Corrected frames become teacher training data and gold eval.
2. *Becomes student training data*: windows reviewed as series, where the teacher already does well and human time confirms it.
3. *Measures the teacher*: the gold eval set and the 10% blind audit, which set the `auto_ok` precision.
Anything that is none of these (random frames, clips already well covered) is not queued. The review queue is ranked by expected teacher gain (disagreement, margin), and the progress view reports gold vs silver counts per teacher version and the teacher's measured precision/recall on gold.

Loop: label targeted frames -> retrain/tune teacher -> teacher v(N+1) relabels everything, audit pass rate rises -> more silver per human minute -> retrain student.

## 4c. Player Check (verifying player labels, especially clusters)

Ball Check verifies one box per card. Player errors are different: a missed person, a duplicate box on one person, one box over two overlapping people, a wrong class (player vs referee), a ghost box. Almost all of them occur where boxes overlap. So the unit is the **cluster**, not the box, and the question is "does each person here have exactly one box?"

**Box rule (must be fixed before labelling):** box the *visible* extent of each person (modal box), one box per person, even when boxes overlap heavily. Do not box a person under 12 px tall.

**1. Triage by geometry (mini CPU, no model).** For each frame in a window, group boxes into clusters: connected components where IoU > 0.2 or the centre distance is under 0.6 of the mean box width. Isolated boxes with a stable track and high confidence are `auto_ok`. Clusters, unstable tracks and frames with disagreement go to a human.

**2. Count over time.** Track each cluster through the window. The number of people in a cluster changes only when someone enters or leaves it, so a count that flickers (3, 3, 2, 3) points at the bad frame. The reviewer sees a short strip of 3-5 frames of the cluster with a count timeline and resolves only the change points; "count is N for frames a-b" is one tap and fixes all frames in that run.

**3. Independent count (flag only).** Re-run the cluster crop upscaled 2x with a different model (SAM3 or a pose/head detector) and compare person counts. A disagreement flags the cluster. This is a flag, never an auto-accept, until audits show it is reliable (same rule as Claude verify).

**4. Card types.**
- *Cluster card* (priority): a zoomed crop of the cluster, boxes in distinct colours with index numbers, overlapping boxes drawn with offset outlines so each stays visible, the count timeline underneath. Swipe right = every person has exactly one box; swipe left = opens fix mode: tap an unboxed person to add, tap a box to delete (duplicate/ghost), drag a line across a box to split a merged pair, tap the class chip to change player/referee.
- *Frame sweep card*: whole frame, boxes shown, one question "anyone unboxed?" with tap-to-place. Catches small far players and people at the frame edge that triage never sees.
- *Audit card*: 10% of `auto_ok` frames, shown blind (no boxes) for full relabelling. The disagreement with the machine boxes gives the miss rate per overlap bucket (isolated / touching / heavy overlap) and sets the auto-accept threshold.

**5. Model-assisted check (optional, gated).** Claude gets the numbered cluster crop and lists persons without a box, boxes covering two people, and duplicates. Used first only to order the queue; promoted to a filter if its precision on audited clusters reaches the 0.95 target.

**6. State.** One row per cluster check: `player_check(game_id, frame_index, cluster_id, bbox, n_boxes, verdict, asserted_count, method, verified_by, verified_at)`. A frame's player state is `complete` when every cluster is verified and its sweep is done; any flagged or fixed-but-unreverified cluster makes it `partial`. Fixes write `hand_frame` boxes (gold).

**6b. No trust in the 0.98.** The detector's player mAP50 (0.98-0.99) was measured on the original one-game dataset, so it is in-distribution and does not predict performance on new matches (the same gap seen for the ball: 0.92 in-distribution vs 0.49 on held-out clips). Consequences:
- Nothing is auto-accepted for players on the strength of detector confidence alone. The `auto_ok` rule for isolated boxes starts at "audit everything" and relaxes per match only once that match's blind audits show a low miss rate.
- Player miss rate and box quality are reported **per match and per overlap bucket**, never pooled into one number.
- A **player eval panel** is needed beside the ball one (under the 2026-11-21 decision it is the eval for the player-side detector; TODO): fully hand-labelled frames (blind, all people boxed) from the heldout matches, sampled with logged inclusion probabilities, used to score the teacher and the student. Player mAP/recall/precision on this panel is the headline; the one-game number is retired from reporting.
- Player audits from train matches measure drift across kits, camera zoom and crowd density, and decide how much human review each new match needs.

**7. Cost target.** Under 3 s for a clean cluster card (swipe), under 15 s with fixes. Report cost per card type on `/progress` alongside the miss rate per overlap bucket.

## 5. Task breakdown (priority order)

Additions from 4b: Task 1's `method` also takes `offline_audited` plus a `teacher_version` column; Task 4 emits disagreement scores used by the queue ranking; new Task 9 below.

1. **State table + ingest writers.** Files: `feature_store/schema.py`, `store.py`, `ingest_gt.py`, tests. Writes `no_ball`/ball-box/verdict state. Accept: re-ingest is idempotent; `no_ball` gives ball `absent`; `unsure` stays unreviewed. Deps: none.
2. **Window selector + propose flow.** Files: new `autolabel/windows.py`, `autolabel/propose.py`, a Prefect entry in the jobs repo. Accept: windows stay inside a segment and never touch heldout; proposals ingest as unreviewed rows. Deps: 1 (for exclusion of already-complete frames).
3. **Series review API and card.** Files: `labeller/series_review.py`, `web/series.html`, `server.py` router. Accept: accept-all writes complete state; one exception demotes only that frame; the reject path re-queues. Deps: 1.
4. **Pre-verify + auto_ok rules + audit sampler.** Files: `autolabel/preverify.py`, tests with fixture tracks. Accept: flagged ordering is deterministic; audit fraction is 10% blind. Deps: 2.
5. **Exporter filter.** Files: `scripts/export_training_dataset.py`, `constants.py`, tests. Accept (original; per-group gate after 2026-11-21, player-only mode TODO): no partial frame appears; negatives present; class ids frozen; default sources. Deps: 1.
6. **Progress endpoint and UI.** Files: `labeller/progress.py`, `web/progress*`, clip list badge. Accept: counts equal a fixture store; ball-only clip counts as not fully labelled. Deps: 1.
7. **Series registry + temporal export option.** Files: `schema.py`, exporter. Accept: contiguous series exported in frame order. Deps: 3, 5.
8. **Legacy backfill flag.** Files: `ingest_gt.py` CLI. Accept: only named clips become player-complete. Deps: 1.

9. **Teacher versioning, gold eval and targeting queue.** Files: `autolabel/teacher.py` (pipeline config + version), `autolabel/queue.py` (rank by disagreement/margin, reject untargeted requests), `ball_eval/` gold scoring, progress additions. Accept: every offline label carries `teacher_version`; scoring a teacher version on the gold set outputs precision/recall; queue never returns frames from non-train matches or already-gold frames. Deps: 1, 4.

10. **Player Check.** Files: `autolabel/clusters.py` (geometric clustering, count timeline, triage), `labeller/player_check.py` + `web/player_check.html` (cluster / sweep / audit cards, fix mode), `player_check` table in `feature_store/schema.py`, tests. Accept: isolated stable boxes are `auto_ok`; overlapping boxes form one cluster; a count flicker is flagged; "count is N for a-b" resolves the run; a fixed cluster writes `hand_frame` boxes; frame player state is `complete` only when all clusters and the sweep are verified. Deps: 1, 2.

## 6. Open questions

1. Is a person visible but too small or occluded to box "complete"? Proposal: yes if no human would box it, but we need a minimum size (suggest 12 px height).
2. Should a legacy clip with hand ball marks plus player marks (e.g. the `arsenal_*` clips) be treated as player-complete without recheck?
3. Window length and review budget: is 5 s windows with about 10% exceptions an acceptable human cost, or should reviews be per-clip-segment?
