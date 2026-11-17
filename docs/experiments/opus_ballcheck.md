# Opus as a Ball Check labeller

Status: **partial.** The final prompt was run on 219 of 331 test cards before the Claude subscription's 5-hour usage limit stopped the run (resets 01:30 Europe/London). The 112 missing test cards are all v1 (44) and v2 (68); the **v2 disagreement set has no test numbers yet**. No cheaper-model comparison was run. Resume instructions are at the end.

Question (George): can Opus do the Ball Check job from the card's still and its short clip, scored against his own verdicts?

Short answer: on the cards we have, yes for ball vs not ball. Opus (`claude-opus-5-5`) given a still, a wider still and a 9-frame filmstrip agrees with George on **95.0 %** of 219 test cards (4 verdicts, Wilson 91-97 %), and on **98.2 %** of ball-present vs not-ball (kappa 0.90). When it says 0.8 or more confident it agrees on 98.0 % of 90 % of the cards. The motion filmstrip is the single biggest ingredient. The remaining disagreements are mostly `box_off` / `corrected` / `unsure`, i.e. George's judgement about the geometry, not whether it is the ball.

## How it was run (read this before trusting the numbers)

- **Backend: Claude Code headless (`claude -p`, subscription login), not the API.** There is no `ANTHROPIC_API_KEY`, no `ant` login and no `~/.config/anthropic` on this machine; the earlier autolabel loop (`footy-claude-autolabel-loop` KB page, PRs #43-#45, `src/footy_track/autolabel/claude_cli.py`) uses `claude -p` for the same reason and strips every API-key route from the environment. I reused that. Images go in as base64 content blocks over `--input-format stream-json` (no Read-tool turn), no tools, no MCP, `--system-prompt` carries the instructions, model `claude-opus-5-5`, default effort (medium on Opus 5.5), adaptive thinking.
- **Cost figures are the CLI's list-price equivalent** (`total_cost_usd`, `costBasis: list`), not a charge. Each call also carries about 1.7 K tokens of the CLI's own cached context. A direct API integration would have a smaller overhead and could cache the instructions explicitly; it is not measured here.
- **Caching:** the CLI only caches its own prefix. Few-shot images were cached by putting `cache_control` (1 h) on the last few-shot block (cost fell from $0.14 to $0.02 per card); the final prompt has no few-shot so it does not matter.
- **Renders are offline** (`render.py`, imports `_crop_window`, `_draw_box`, `_box_in_crop` from `footy_track.labeller.ball_check` on `origin/main` `ce91833`). Per card: **STILL** = George's still (`pad=6`, 4x nearest upscale to 640 px, yellow box and brackets drawn like the page overlay, other box magenta); **WIDE** = same frame, `pad=14` (about 2.3x the area), Lanczos; **FILMSTRIP** = 3x3 tiles of the same crop window at offsets -12, -8, -4, -2, 0, +2, +4, +8, +12 frames, box drawn on the t=0 tile only (as in George's clip). No server request was made; the live :8011 server, the other worktrees and every marks dir were untouched (read only).
- **Clip ids match.** All three sets use `arsenal_nottinghamforest_segNNN` (PR #62 rename). The `clip_run_id` hidden key in the v2 list still says `arsenal_norwich_segNNN`; it is a leftover diagnostic key and nothing here uses it.

## Ground truth and split

George's latest verdict per (clip, frame, box_index, source) from `ball_checks/*.jsonl`, same rule as `ball_check._read_verdicts` (tombstones retract). Sets: `sidecar` (generic queue, source `sidecar`), `v1` (Forest blind list, source `list`), `v2` (Forest disagreement list, source `list`; George was still labelling, 100+ judged when the table was frozen). `cand`-source verdicts (the candidate queue) were not used. `corrected` (George dragged the box onto the ball) is scored as `box_off`. Totals at freeze: 473 cards (sidecar 176, v1 200, v2 97), 142 dev / 331 test, split stratified by (set, verdict) 30 / 70 with a seeded shuffle; cards George judged after the first snapshot were assigned by id hash. `build_cards.py` keeps the first assignment sticky. Card table: `/mnt/storage/exp/opus_ballcheck/cards_final.json`.

Opus verdicts only ever went to `/mnt/storage/exp/opus_ballcheck/`.

## Dev iteration (prompts and image formats chosen on dev only)

"dev-iter" = every dev non-ball card plus about half of the dev balls (81 cards; 77 scored after dropping the 4 cards used as few-shot examples; 68 common to all rows below because the ablation was cut short). All dev numbers are therefore small, with wide intervals.

| run | change | 4-class acc | ball-present kappa | not_ball precision | not_ball recall | $/card |
|---|---|---|---|---|---|---|
| A | prompt v1 (task + verdict definitions), STILL + FILMSTRIP | 0.897 | 0.857 | 0.900 | 0.900 | 0.044 |
| B | prompt v2 (ball appearance, false-positive list, "use the motion", calibrated confidence, two-box field), + WIDE | 0.926 | 0.931 | 0.909 | 1.000 | 0.049 |
| C | prompt v3 = v2 + box size in px + judging rules (box_off sizing, occlusion, edge, spare ball, "machines are often wrong") | 0.926 | 0.965 | 0.952 | 1.000 | 0.051 |
| D | C + 4 worked examples (cached) | 0.926 | 0.931 | 0.909 | 1.000 | 0.037 |
| E | C but STILL + WIDE only (no filmstrip, so no motion) | 0.809 | 0.662 | 0.654 | 0.850 | 0.039 |

What helped: (1) the wider context image plus the false-positive list and the instruction to use motion (A to B: `b1_rejected` strata 12/14 to 14/14); (2) the filmstrip is essential: removing it drops accuracy by 12 points and not_ball precision from 0.95 to 0.65 (spare balls, boots and shirt patches look like a ball in a single frame; E is a partial run, 71 of 81 cards, cut short by the usage limit); (3) v3 is a wash on accuracy, slightly better on the binary question. Few-shot examples did not help. The confidence number is usable (see the coverage table). All variants are within noise of each other on accuracy, so the final choice of v3 (no few-shot) is mostly "simplest that is as good as the best"; v3's rules were partly written after reading the dev errors, so dev numbers for it are optimistic, the test numbers are not.

Residual dev errors were identical across B, C and D: a fallen player's white shorts boxed where George said ball (6f01aa4ac7, Opus 0.9 not_ball), a goalkeeper's ball where George said box_off but the box looks fine (8f0739cab2), and a ball at the crop edge where George said not_ball (aadf39cf5b). These are in the dev sheet below.

## Final prompt (verbatim system prompt, v3)

Sent as `--system-prompt`. Per card the user message is: the line `Card: one box (yellow A). Box A is W x H px in the 1920x1080 frame.` (or `Card: TWO boxes (yellow A = judge this one, magenta B = other). ...`), then `STILL:`, image, `WIDE:`, image, `FILMSTRIP:`, image, then `Reply with the JSON object only.`

```
You are doing the "Ball Check" job for a football (soccer) ball-labelling project, standing in
for a human reviewer. Each card is a crop of 1080p TV broadcast footage around a small yellow
box placed by a machine (a detector or tracker). Decide whether the yellow box is on the
in-play match ball.

What the ball looks like here: a small sphere, usually only 6 to 14 pixels wide in the 1080p
frame. White, light grey or pale yellow/pink, sometimes orange; often motion-blurred into a
small smear or ellipse; sometimes partly hidden by a boot or a body. It can be on the grass, at
a player's feet, or in the air.

Common false positives (answer not_ball for these): boots and socks, white or light shirt
parts, heads and hands, white boot or advert-board marks, ball graphics printed on advertising
hoardings, pitch line paint and its intersections, corner flags, camera/lens flares, and spare
balls resting off the pitch or on the hoardings, held by ball boys or sitting still next to the
touchline. A spare ball is not the match ball.

You get, per card:
 1. STILL: the crop for the box's frame (4x enlarged, pixelated), with the yellow box and 4
    yellow corner brackets (outside the box) drawn on it.
 2. WIDE: a wider crop (about 2.3x more area) of the same frame with the same box, for context:
    where on the pitch it is, who is near it, whether it is by the hoardings or the stands.
 3. FILMSTRIP: the same crop window as the STILL over time, 3x3 tiles, frame offsets
    -12, -8, -4, -2, 0, +2, +4, +8, +12 (25 fps). The box is drawn ONLY on the t=0 tile, because the
    ball moves; on the other tiles look for the same object and follow it.
USE THE MOTION. A real in-play ball follows a smooth path, moves differently from the
players around it (it flies, rolls, gets kicked) and does not vanish. A shirt or boot patch moves
with its player; a spare or advert ball is static relative to the pitch or boards.
If the same object is visible in several tiles, say so.

Judging rules learned from the reviewer:
 - The boxes come from machines that are often wrong. Do not assume the box is on the ball; decide from
   the pixels. In some queues roughly a third of the boxes are not the ball.
 - Each card states the box size in 1080p pixels. A ball is typically 8 to 16 px wide, so a box of about
   10 to 24 px on a ball is normal. The box is "ball" if the whole ball sits inside it with at most a modest
   margin. It is "box_off" if the ball is clearly there but the box is more than about 2.5x the ball's
   size (e.g. tall enough to cover the shin as well), is clearly smaller than the ball, or has the ball
   cut by its edge / mostly outside it. Do not call box_off for a small margin.
 - If at t=0 the ball is partly hidden by a boot, sock or leg, but you can see the same ball at the same
   place in the neighbouring tiles (-2, +2), it is still the ball.
 - A ball cut by the edge of the crop is still the ball if you can see that it is a ball.
 - A ball rolling or lying on the grass that does NOT move with the players and is outside the touchline,
   beside hoardings, or in a ball-boy's hands, is a spare ball: not_ball.
 - If you can see the in-play ball elsewhere in the crop and the box is on something else, it is not_ball.

Verdicts (choose exactly one):
 - "ball": the box is on the in-play match ball and reasonably tight around it.
 - "not_ball": the object in the box is not the in-play match ball (see the false positives), or
   there is nothing ball-like there. If you can see a ball clearly somewhere else in the crop
   and the box is not near it, that is also not_ball.
 - "box_off": the ball is clearly visible AND the box is on or touching it, but badly sized or
   placed (far too large, much too small, or only half on it). Use this rarely.
 - "unsure": you cannot tell even after studying all tiles. Use it only when the evidence is
   genuinely ambiguous, not as a hedge.
Be calibrated: confidence is the probability that your verdict matches a careful human
reviewer who watched the clip.
Some cards carry a second, MAGENTA box ("B", a different machine's guess). The yellow box is
"A", and your verdict is ALWAYS about the yellow box A. Also report "which_box": "A", "B",
"both" (A and B are on the ball), or "neither", saying which box, if any, is on the in-play ball.

Reply with ONLY a JSON object, no code fences:
{"verdict": "ball"|"not_ball"|"box_off"|"unsure", "confidence": <0 to 1>, "which_box": "A"|"B"|"both"|"neither" (only on two-box cards, else null), "reason": "<one or two short sentences: what the object is and what the motion shows>"}
```

## Test results (final prompt, 219 of 331 test cards: sidecar 123, v1 96, v2 0)

Not a random subset: the run walked the table in (set, clip) order, so it covers the whole sidecar test set (123 of 123 judged, but sidecar is 95 % `ball` and only 1 `not_ball`) and 96 of 140 v1 cards. **v2 is missing entirely and the 44 unrun v1 cards are 22 `ball` and 22 `not_ball`, so the not_ball side is under-represented** (23 not_balls scored, 44 more to come); treat these numbers as provisional.

- 4-class accuracy: **0.950 [0.91-0.97]** (208/219). Cohen's kappa 0.79.
- Ball-present (`ball`/`box_off`) vs `not_ball`, excluding unsure either side: accuracy 0.982 [0.95-0.99] (214/218), kappa **0.899**.
- Verdict `ball`: precision 0.948 [0.91-0.97] (184/194), recall 0.995 [0.97-1.00] (184/185). Against "George says ball or box_off" the precision is 0.979 [0.95-0.99] (190/194); the 4 errors are 3 not_ball and 1 unsure.
- Verdict `not_ball`: precision 0.952 [0.77-0.99] (20/21), recall 0.870 [0.68-0.95] (20/23). Only 23 George not_balls in the covered cards, so wide intervals.
- Opus `unsure` was never used on test (it was used 0 times on the answered test cards).

Confusion (rows George, columns Opus):

| George \ Opus | ball | not_ball | box_off | unsure | total |
|---|---|---|---|---|---|
| ball | 184 | 1 | 0 | 0 | 185 |
| not_ball | 3 | 20 | 0 | 0 | 23 |
| box_off (incl. corrected) | 6 | 0 | 4 | 0 | 10 |
| unsure | 1 | 0 | 0 | 0 | 1 |

Coverage vs agreement (Opus confidence >= t; agreement = same 4-class verdict as George):

| t | coverage | agreement | `ball` precision (vs George ball) | `not_ball` precision |
|---|---|---|---|---|
| 0.0 | 219/219 = 1.00 | 0.950 [0.91-0.97] | 0.948 [0.91-0.97] | 0.952 [0.77-0.99] |
| 0.7 | 206/219 = 0.94 | 0.961 [0.93-0.98] | 0.957 [0.92-0.98] | 1.000 [0.83-1.00] (19/19) |
| 0.8 | 197/219 = 0.90 | 0.980 [0.95-0.99] | 0.978 [0.94-0.99] | 1.000 [0.83-1.00] (19/19) |
| 0.9 | 148/219 = 0.68 | 0.980 [0.94-0.99] | 0.978 [0.94-0.99] | 1.000 [0.77-1.00] (13/13) |
| 0.95 | 84/219 = 0.38 | 0.988 [0.94-1.00] | 0.988 [0.93-1.00] | 1.000 [0.34-1.00] (2/2) |

Per set: sidecar 0.951 [0.90-0.98] (117/123); v1 0.948 [0.88-0.98] (91/96). v1 strata: `r3_easy` 31/31, `r3_hard_S1_fill` 32/34, `b1_lowp` 15/17, `b1_rejected` 13/14. Two-box cards: 11/12. v2: not run. Model confidence is stated by the model, not a logit; it is coarse (mostly 0.6-0.95).

Cost and latency per card: mean **$0.050** list-price equivalent (about **$50 per 1,000 cards**; about 7.4 K input tokens including the CLI's own, about 220 output tokens). Wall time per card 24 s mean (21.5 s median), but about 15 s of that is `claude -p` start-up and the machine was at load average 45-50; the model's own API time is about 5-8 s. Throughput was about 6 cards/minute with 3 workers.

Cheaper models (`claude-sonnet-5-5`, `claude-haiku-5-5`): **not run** (usage limit).

## Where Opus and George disagree

Eleven test disagreements; none is a clear ball-vs-ball flip in Opus's favour without a look. Seven are `box_off` / `corrected` / `unsure` where Opus says the box is on the ball (geometry or tolerance judgements). Four are ball vs not_ball:

- `ad6845ed0b` (v1 r3_hard_S1_fill): George not_ball, Opus ball 0.80 on a faint fast-moving blur on a smooth kick path.
- `72a6b7fcfc` (v1 b1_lowp): George not_ball, Opus ball 0.58.
- `fbcdad43cd` (v1 b1_rejected): George not_ball, Opus ball 0.75, a ball at the bottom crop edge.
- `647a81b26b` (sidecar): George ball, Opus not_ball 0.60: the ball is a blur just left of the box; box on empty grass.

Review sheets (HTML, images inlined, ball/not_ball flips first): `/mnt/storage/exp/opus_ballcheck/disagreements_test_final_opus_v3.html` (conf >= 0.8: 4 cards, only 1 is a flip) and `disagreements_dev_C_v3_meta.html`. Generate any other with `gen_sheet.py RUN --min-conf 0.6`. Dev disagreements worth a second look: `6f01aa4ac7` (George ball, Opus 0.8-0.9 not_ball: white shorts of a fallen player, the ball is elsewhere), `8f0739cab2` (George box_off, the box looks fine to me), `aadf39cf5b` (George not_ball, a ball at the crop edge). I have not decided who is right on any of them.

## What it means for the pipeline

Provisional, because v2 (the tracker disagreements, the likely main use) is missing and the not_ball side is thin:

- On sidecar and v1 cards Opus is already a good fourth judge: a confident `ball` (>= 0.8) is George's `ball` 97.8 % of the time (174/178) and a confident `not_ball` was right 19/19.
- As a **pre-filter** that spares George cards: auto-accept `ball` at >= 0.8 or 0.9 (90 % / 68 % of cards, about 98 % agreement), and send the rest, every `not_ball`, `box_off` and `unsure` to George. Do not auto-accept `box_off` or geometry: Opus is weak there (4/10).
- Opus's `ball` is the claim to test on the v2 disagreement cards and on SAM/VitTrack pseudo-labels, because those are the cards where the three trackers disagreed and the easy sidecar numbers do not apply.

## Disclaimers

- Dev sets are small (68-81 cards); per-stratum dev and test numbers have very wide intervals, and several strata have 0-2 not_balls.
- George can be wrong and some of the 11 may be his errors; the scoring treats his verdict as truth.
- Prompt v3's rules came partly from reading dev errors; dev scores for v3 are optimistic. Test was run once.
- The list for v1 and v2 was read from the files on disk (position = box_index; every card's clip and frame matched its list entry).

## Files and how to resume

Code in `experiments/opus_ballcheck/`: `build_cards.py`, `render.py`, `prompts.py`, `ask.py`, `score.py`, `compare_dev.py`, `gen_sheet.py`, `shots.json`. Outputs in `/mnt/storage/exp/opus_ballcheck/`: `cards_final.json`, `renders/`, `runs/*.jsonl` (every raw answer, usage and cost; failed calls are in `runs/errors_*.jsonl`), `disagreements_*.html`.

Finish the test run once the limit resets (the script skips answered cards; keep `cards.json` frozen, do not rerun `build_cards.py`):

```
cd experiments/opus_ballcheck
python3 ask.py --run test_final_opus_v3 --prompt v3 --meta --images still,wide,strip --split test --workers 3
python3 score.py test_final_opus_v3        # plus --split test
# cheaper-model comparison on a hash half of test:
python3 ask.py --run test_sonnet_v3_half --model claude-sonnet-5-5 --prompt v3 --meta --images still,wide,strip --split test --half --workers 3
python3 ask.py --run test_haiku_v3_half --model claude-haiku-5-5 --prompt v3 --meta --images still,wide,strip --split test --half --workers 3
```

About 640 card calls were used in total (4 smoke, 81 x 3 for runs A-C, 89 for D including 8 aborted, 76 for E, about 224 test), under the roughly 1,000 budget; the Sonnet/Haiku comparison would add about 330.
