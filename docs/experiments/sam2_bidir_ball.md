# SAM2.1 bidirectional propagation from sparse human ball anchors

Branch `exp/sam2-bidir-ball`. Code: `experiments/sam2_bidir_ball/`. Raw per-frame tracks: `docs/experiments/sam2_bidir_ball_raw/`. Aggregates: `docs/experiments/sam2_bidir_ball.json`.

## Question

Prior research (`research_notes/Offline football ball tracking models/promptable_vos.md`, `docs/design/bidirectional_tracking_research.md` section 3) predicted that SAM2 would be weak on an ~11 px football: the backbone runs at 1024 x 1024 with stride-16 memory features, the one small-fast-object benchmark found (FMOX) gives SAM2-family trackers mean mIoU 0.44-0.51, and nobody has reported SAM2 on football balls. This experiment tests the specific use that matters for labelling: **given a human box on every N-th labelled frame, how well do forward and backward SAM2.1 propagation fill the frames in between, and how far does one prompt carry?**

## Setup

* **GT**: `ball_gt_marks/<clip>.jsonl`. Ball = `in_play_ball` / `out_of_play_ball` with human provenance (`labeller` or `human_checked`); machine-only marks (`yolo`, bare `vittrack`) are dropped. Explicit `no_ball` and `not_broadcast` rows are used for false-positive checks only.
* **Runs** (maximal blocks of consecutive human-labelled ball frames, >= 9 frames): `arsenal_example` six runs of 9/16/31/40/9/50 frames (21 further labelled frames sit in runs shorter than 9 and are skipped), `astonvilla_seg080` 359/104/43/62, `bournemouth_1st_seg070` 63/157. `arsenal_example` is the clip `arsenal_mancity_example_video.mp4` (md5 identical to the desktop copy; canonical id per `src/footy_track/clip_id.py`).
* **Clip facts that differ from the brief**: `astonvilla_seg080` is **1280 x 720** (errors below are scaled to 1080p-equivalent pixels, x1.5), `bournemouth_1st_seg070` is **50 fps** (so "frames" are 20 ms), only `arsenal_example` is 1920 x 1080 / 25 fps. GT `frame_index` is the decoder frame index for all three (verified visually).
* **Anchors**: every N-th frame of a run from its start (N in {10, 25, 50}) plus the run's last frame. Only the remaining GT ball frames are scored (held-out). Prompt = the GT box (`add_new_points_or_box(box=...)`).
* **Propagation**: for each anchor, an independent inference state for the forward window and another for the backward window (`propagate_in_video(reverse=True)`), reach max(N, 25) frames (so tracks never see another anchor). Fusion: **nearest** (the pass from the nearer anchor) and **hybrid** (the pass with the highest sigmoid(object score)/distance, falling back to the interpolation when SAM2 reports the object absent).
* **Variants**: (a) `full` = whole frame squashed to 1024 x 1024 by SAM2; (b) `crop512` = native-resolution 512 x 512 window (no downscale before SAM2's own 2x upscale to 1024), re-centred on the last predicted ball position before every frame (custom driver around `propagate_in_video`: the next frame's crop is injected into the state after each yield); (c) model size tiny / small / base+. fp16 autocast, `offload_video_to_cpu=True`, per-frame mask post-processing = largest connected component of the thresholded mask, centre = its centroid. "Absent" = empty mask (SAM2 object-score head negative).
* **Baseline**: linear interpolation of the two bracketing anchor centres (pure CPU, never "absent").
* **Metrics**: centre error in px at 1080p-equivalent; hit@5/10/20 over **all** held-out frames (an absent prediction is a miss); median error over frames where SAM2 predicted a ball; false-positive rate on `no_ball` frames; absent rate where GT has a ball.
* **Where it ran**: Prefect `sam2-ball/sam2-ball` deployment (gpu pool, desktop RTX 2070 8 GB; flow `sam2_ball.py` + `sam2.prefect.yaml` in `~/code/jobs`, copy in `experiments/sam2_bidir_ball/`) for the `ars1` run (arsenal, astonvilla and an invalidated bournemouth pass, see "Pitfalls"). Two later runs (`bourne`, `sizes2`) sat Late for over an hour behind another session's gpu-pool job whose `rsync` to the mini had hung, so they were run with the same script directly on the desktop in tmux (remote-job style fallback). Results are identical in kind (same script, same data).

## Pitfalls hit (worth knowing)

1. **`cv2.VideoCapture` returns all-black frames** for these clips (swscaler "Cannot convert interlaced to progressive"), in both the mini and desktop wheels. A first smoke run therefore tracked black frames and produced convincing-looking garbage (40-65% "absent", errors growing linearly). Fix: decode with `ffmpeg -f rawvideo -pix_fmt rgb24`.
2. **ffmpeg's default vsync duplicates/drops frames on `bournemouth_1st_seg070`** (start_time 1.18 s, irregular pts), shifting frame indices by up to ~20 frames by the first second; the GT index is the decoder index, so decode with `-fps_mode passthrough`. The first Bournemouth pass (misaligned) looked like total SAM2 failure; it was discarded and re-run. `arsenal_example` and `astonvilla_seg080` have uniform pts and are unaffected.
3. The PyPI `sam2` wheel was built without the CUDA extension (`SAM2_BUILD_CUDA=0`), so SAM2's hole-filling post-process is skipped (warning "cannot import name '_C'"). It fills holes in masks, so it should not matter for a ~100 px^2 ball; not ablated.
4. `GT` caveat: `astonvilla_seg080` (526 of 568 ball marks) and `bournemouth_1st_seg070` (216 of 220) are VitTrack tracks that a human reviewed (`vittrack` + `human_checked`); only `arsenal_example` is roughly half human-drawn (94 `labeller` / 82 reviewed VitTrack). SAM2 agreeing with those GT boxes to ~1 px may partly be two template trackers agreeing, and the reviewed boxes are probably smoother than independent hand labels.


## Headline

SAM2.1 does track the ~11-20 px ball far better than the research notes predicted, **when it is prompted with a correct box**. Median error is ~1 px at 1080p-equivalent, in both directions, in every config, and it does not decay with distance from the anchor on the two easier clips. It beats linear interpolation by a wide margin as soon as anchors are 25+ frames apart. It is weakest on `arsenal_example`, the clip with human-drawn boxes, a kicked ball and a whip-pan.

Held-out denominators at N = 10 / 25 / 50: `arsenal_example` 133 / 140 / 143, `astonvilla_seg080` 505 / 539 / 550, `bournemouth_1st_seg070` 195 / 208 / 212 (T1 gives the exact n per row).

Small model, `crop512`, nearest-anchor fusion, hit@10 over all held-out frames (absent = miss), against interpolation:

| clip | N=10 SAM2 / interp | N=25 SAM2 / interp | N=50 SAM2 / interp | SAM2 absent (N=25) |
|---|---|---|---|---|
| arsenal_example | 82 / 69 | 84 / 51 | 78 / 41 | 11% |
| astonvilla_seg080 | 97 / 89 | 94 / 59 | 92 / 41 | 0% |
| bournemouth_1st_seg070 | 99 / 95 | 100 / 67 | 100 / 18 | 0% |

## Findings

1. **Resolution: the native crop helps only where the ball is hard.** On `arsenal_example` `crop512` beats `full` clearly (N=25 hit@10 84% vs 77%; N=50 78% vs 65%; the backward `full` pass collapses to 19% at 26-50 frames while `crop512` keeps 65%). On `astonvilla_seg080` and `bournemouth_1st_seg070` the two are within a few points (the full-frame variant is even slightly better on Bournemouth, 96-98% hit@5 vs 90-91%, where the ball is larger: median 22 x 19 px). The prediction that an 11 px ball is one stride-16 cell wide and therefore invisible was too pessimistic: median error in `full` is only 1.1-1.6 px (arsenal 1.7-2.4 px for tiny/small).
2. **Model size barely matters.** tiny / small / base+ differ by a few points and in no consistent direction (T1). The only visible gap: `arsenal_example` N=50 `crop512`: tiny 65% / small 78% / base+ 76% hit@10. Tiny is the fastest (0.132 s/frame vs 0.149 small, 0.165 base+) and uses 0.7 GB; base+ 0.96 GB. Nothing here needs more than 1.2 GB reserved, so 8 GB is not a constraint (no need for `offload_state_to_cpu`).
3. **Carry distance (T3).** One prompt carries to the end of the 50-frame window without a measurable decay on `astonvilla_seg080` (fwd 96 / 94 / 90 / 93 % hit@10 for 1-5 / 6-12 / 13-25 / 26-50 frames; bwd 98 / 92 / 85 / 77) and `bournemouth_1st_seg070` (93-100% everywhere). The hold-the-anchor baseline falls to 9% and 0% by 26-50 frames. On `arsenal_example` the forward `crop512` curve is flat (82 / 82 / 87 / 76) while the `full` backward curve breaks after 25 frames (75 / 78 / 68 / 19). Backward is slightly worse than forward at long range on astonvilla (77% vs 93% at 26-50), i.e. fusion by nearest anchor is the right default: nothing is lost by never using a pass beyond N/2.
4. **Fusion.** Nearest-anchor is within 1-2 points of the best single-pass choice everywhere (T1/T2). The score-weighted and hybrid (interpolation fallback when SAM2 says absent) variants help only on `arsenal_example` (N=25: 84 to 90% hit@10; N=50: 78 to 87%) because that is the only clip where SAM2 reports absent.
5. **Forward/backward agreement is an excellent confidence signal (T5).** Where the forward pass from the previous anchor and the backward pass from the next anchor are both present and within 5 px, the fused answer is within 10 px in 96-100% of cases (arsenal 100%, astonvilla 96-99%, Bournemouth 100%) at 64-100% coverage of held-out frames (arsenal small/`crop512`: 77% / 81% / 64% for N = 10 / 25 / 50; astonvilla 86-90%; Bournemouth 87-100%). That is the natural auto-accept rule for a labelling tool; the remainder goes to a human.
6. **Interpolation baseline.** Interpolation is strong at N=10 (median 1.7-4.7 px), because the labelled stretches are smooth camera-tracked ball motion; SAM2 is still ahead on arsenal (82 vs 69% hit@10) and astonvilla (97 vs 89%), and level on Bournemouth (99 vs 95%). At N=25 and especially N=50 interpolation collapses (hit@10 41 / 41 / 18%) while SAM2 stays at 78 / 92 / 100%. SAM2 is worth using when anchors are sparse (>= 25 frames) or the ball changes direction; at 10-frame spacing it is a small win that needs the GPU.
7. **False positives / absence.** `astonvilla_seg080` has 5 (run, frame) no_ball samples (4 distinct frames, 371 counted for both runs touching it): SAM2 invents a ball on 2 of 5 (`crop512`, small and tiny) and 3-4 of 5 (`full`). `bournemouth_1st_seg070` has 4 `not_broadcast` frames (cut-aways): `crop512` reports absent on all 4, `full` reports a ball on all 4 (small; T6). The sample is far too small to quote a rate; the direction (SAM2 hallucinates on no-ball frames, crop is more conservative) is the only thing to take from it. Absent-when-GT-has-ball is 0-1% on astonvilla / Bournemouth and 8-18% (N=10 to 50, `crop512`) on arsenal.

## Failure modes (arsenal_example is where they occur)

T7 splits the nearest-anchor result by ball speed. Everything on `arsenal_example` at < 8 px/frame is right (98% hit@10, `crop512`, N=25); 8-20 px/frame drops to 66% and > 20 px/frame to 0% (10 frames). Looking at the frames (`viz.py`):
* **Kick / whip-pan.** After a kick (frames 146-155, 181-187, 201-206) the camera whip-pans, the ball is a 15-20 px smear against crowd, SAM2 first gives a very low object score (0.2-0.4), then reports absent for 5-9 frames; the re-centred crop loses the ball because the whole scene moves by more than the window within one frame. Errors in the wrong-answer cases are 30-70 px.
* **Recovery**: the pass from the other side often still has the ball (hybrid fusion lifts arsenal N=25 hit@10 from 84% to 90%), but a fast stretch in the middle of both directions is not recovered by either.
* **Drift to players / boots** was rare: "wrong, on player/ref box" is 0% of held-out frames on arsenal and Bournemouth, 1-3% on astonvilla (5-7% of its 8-20 and > 20 px/frame frames, T7). There was no systematic stick-to-a-player failure like those described for template trackers; the failures are loss and fast-motion errors, not distractor capture.
* **Blur**: only the fast frames above; no failure was seen on blurred but slow frames.

## Runtime and memory (T8)

RTX 2070 8 GB, fp16 autocast (bf16 is not natively supported on Turing, not tried), `offload_video_to_cpu=True`, one object. small: 0.15-0.16 s/frame including image preprocessing on the CPU (~6-7 frames/s, so a 25 fps clip is ~4x real time), peak VRAM 0.73 GB allocated / 0.95 GB reserved; tiny 0.13 s/frame, 0.69 GB; base+ 0.17 s/frame, 0.96 GB. The per-frame time includes CPU image preprocessing (PIL resize + normalise to 1024 x 1024) and was not profiled further.

## Verdict

* **(i) Propagation engine for labelling assistance: yes.** Prompt on every 25-50 frames, propagate both ways, auto-accept frames where the forward and backward passes agree within 5 px (96-100% within 10 px, 64-100% coverage), send the rest to the human, and fall back to interpolation when SAM2 says absent. This replaces sparse interpolation (hit@10 18-41% at N=50) with 78-100%. Prefer the native-resolution crop on hard clips; it costs a little custom driver code. A 8 GB GPU is plenty; throughput is ~7 frames/s.
* **(ii) Component of the offline model: conditional, not shown here.** Everything above used **perfect human anchors**. For the offline detector + tracker pipeline the anchors would be detector hits, some wrong; SAM2 will faithfully propagate a wrong box (and SAM2 does not verify what it is given). The result supports using it as a temporal gap-filler / consistency check between *high-confidence* detections (the agreement gate is a natural verifier), not as a detector, and not as a replacement for a learned temporal model on fast kicks, where it failed. Test next: anchors from detector hits at several confidence thresholds, with the identical scoring.
* Neither "SAM2 cannot see an 11 px ball" (the prior notes) nor "SAM2 solves ball tracking" holds: it tracks a visible, slowly-moving ball almost perfectly from one box and fails on fast/blurred frames.

## Limits of this test

* Anchors are perfect, and held-out frames are frames a human could label (contiguous labelled runs), so the hard cases that make labelling expensive (long occlusions, ball out of frame, unlabelled gaps) are not scored; the no_ball sample is 5.
* Two of three clips have mostly VitTrack-then-reviewed GT, so ~1 px agreement is optimistic (point 4 under Pitfalls); `arsenal_example` is the independent check and it is the weaker one.
* One run per config (SAM2 is deterministic for a given input; fp16 autocast not ablated against fp32). Anchor set differs per N, so rows across N are not nested subsets of the same frames.
* bf16, `vos_optimized` torch.compile, SAMURAI/DAM4SAM layers, box+point prompts and `crop640/384` were not run (the `--prompt boxpoint` and `crop<N>` options exist in `sam2_tracks.py`).


## Tables (condensed; full set in `sam2_bidir_ball_tables.md`)

### T1. Leave-anchors-out accuracy (held-out GT frames only)

`hit@k` = share of ALL held-out frames with centre error <= k px at 1080p (SAM2 absent counts as a miss); `abs` = SAM2 reports object absent; `med` = median error over frames where a ball was predicted. `nearest` = prediction from the nearer anchor; `hybrid` = best-scoring pass, falling back to interpolation when SAM2 is absent.

| clip | config | N | n held-out | interp hit@5/10/20 | interp med | SAM2 nearest hit@5/10/20 | SAM2 med | SAM2 abs | hybrid hit@5/10/20 |
|---|---|---|---|---|---|---|---|---|---|
| arsenal_example | base_plus/crop512 | 10 | 133 | 52/69/87 | 4.72 | 81/82/83 | 1.03 | 8% | 82/83/85 |
| arsenal_example | base_plus/crop512 | 25 | 140 | 34/51/62 | 9.74 | 80/81/81 | 1.11 | 14% | 84/86/88 |
| arsenal_example | base_plus/crop512 | 50 | 143 | 22/41/47 | 25.96 | 76/76/76 | 1.0 | 21% | 85/87/88 |
| arsenal_example | base_plus/full | 10 | 133 | 52/69/87 | 4.72 | 76/80/81 | 1.75 | 4% | 76/80/81 |
| arsenal_example | base_plus/full | 25 | 140 | 34/51/62 | 9.74 | 73/76/80 | 1.73 | 8% | 76/80/85 |
| arsenal_example | base_plus/full | 50 | 143 | 22/41/47 | 25.96 | 62/66/66 | 1.9 | 8% | 66/70/71 |
| arsenal_example | small/crop512 | 10 | 133 | 52/69/87 | 4.72 | 81/82/83 | 1.15 | 8% | 82/83/85 |
| arsenal_example | small/crop512 | 25 | 140 | 34/51/62 | 9.74 | 83/84/84 | 1.15 | 11% | 88/90/93 |
| arsenal_example | small/crop512 | 50 | 143 | 22/41/47 | 25.96 | 77/78/78 | 0.99 | 18% | 85/87/88 |
| arsenal_example | small/full | 10 | 133 | 52/69/87 | 4.72 | 74/75/79 | 1.91 | 7% | 75/77/81 |
| arsenal_example | small/full | 25 | 140 | 34/51/62 | 9.74 | 66/77/79 | 2.17 | 9% | 70/83/86 |
| arsenal_example | small/full | 50 | 143 | 22/41/47 | 25.96 | 57/65/66 | 2.42 | 17% | 60/70/72 |
| arsenal_example | tiny/crop512 | 10 | 133 | 52/69/87 | 4.72 | 82/82/83 | 1.12 | 6% | 83/83/85 |
| arsenal_example | tiny/crop512 | 25 | 140 | 34/51/62 | 9.74 | 78/79/79 | 1.1 | 13% | 86/88/89 |
| arsenal_example | tiny/crop512 | 50 | 143 | 22/41/47 | 25.96 | 64/65/66 | 1.19 | 22% | 83/85/86 |
| arsenal_example | tiny/full | 10 | 133 | 52/69/87 | 4.72 | 71/75/80 | 1.99 | 5% | 72/77/81 |
| arsenal_example | tiny/full | 25 | 140 | 34/51/62 | 9.74 | 64/76/80 | 2.74 | 7% | 67/80/84 |
| arsenal_example | tiny/full | 50 | 143 | 22/41/47 | 25.96 | 54/64/66 | 4.09 | 7% | 55/64/66 |
| astonvilla_seg080 | base_plus/crop512 | 10 | 505 | 74/89/96 | 2.01 | 93/97/97 | 1.06 | 1% | 93/98/98 |
| astonvilla_seg080 | base_plus/crop512 | 25 | 539 | 42/59/73 | 7.26 | 87/92/95 | 1.14 | 1% | 88/93/96 |
| astonvilla_seg080 | base_plus/crop512 | 50 | 550 | 31/41/54 | 17.58 | 87/92/95 | 1.14 | 1% | 87/92/95 |
| astonvilla_seg080 | base_plus/full | 10 | 505 | 74/89/96 | 2.01 | 92/97/98 | 1.12 | 0% | 92/97/98 |
| astonvilla_seg080 | base_plus/full | 25 | 539 | 42/59/73 | 7.26 | 89/94/97 | 1.19 | 0% | 89/94/97 |
| astonvilla_seg080 | base_plus/full | 50 | 550 | 31/41/54 | 17.58 | 85/90/94 | 1.18 | 1% | 85/91/95 |
| astonvilla_seg080 | small/crop512 | 10 | 505 | 74/89/96 | 2.01 | 93/97/97 | 1.34 | 1% | 94/98/98 |
| astonvilla_seg080 | small/crop512 | 25 | 539 | 42/59/73 | 7.26 | 91/94/97 | 1.42 | 0% | 91/94/97 |
| astonvilla_seg080 | small/crop512 | 50 | 550 | 31/41/54 | 17.58 | 86/92/95 | 1.45 | 1% | 86/92/95 |
| astonvilla_seg080 | small/full | 10 | 505 | 74/89/96 | 2.01 | 92/97/98 | 1.46 | 0% | 92/97/98 |
| astonvilla_seg080 | small/full | 25 | 539 | 42/59/73 | 7.26 | 88/94/96 | 1.52 | 0% | 88/94/96 |
| astonvilla_seg080 | small/full | 50 | 550 | 31/41/54 | 17.58 | 85/91/94 | 1.56 | 1% | 85/91/95 |
| astonvilla_seg080 | tiny/crop512 | 10 | 505 | 74/89/96 | 2.01 | 93/97/97 | 1.21 | 1% | 94/98/98 |
| astonvilla_seg080 | tiny/crop512 | 25 | 539 | 42/59/73 | 7.26 | 90/94/96 | 1.25 | 1% | 90/94/96 |
| astonvilla_seg080 | tiny/crop512 | 50 | 550 | 31/41/54 | 17.58 | 86/92/95 | 1.28 | 0% | 85/92/95 |
| astonvilla_seg080 | tiny/full | 10 | 505 | 74/89/96 | 2.01 | 93/97/98 | 1.29 | 0% | 93/97/98 |
| astonvilla_seg080 | tiny/full | 25 | 539 | 42/59/73 | 7.26 | 89/95/98 | 1.32 | 0% | 89/95/98 |
| astonvilla_seg080 | tiny/full | 50 | 550 | 31/41/54 | 17.58 | 88/93/97 | 1.36 | 0% | 88/93/97 |
| bournemouth_1st_seg070 | base_plus/crop512 | 10 | 195 | 91/95/97 | 1.68 | 91/100/100 | 1.05 | 0% | 91/100/100 |
| bournemouth_1st_seg070 | base_plus/crop512 | 25 | 208 | 42/67/91 | 6.64 | 90/100/100 | 1.06 | 0% | 90/100/100 |
| bournemouth_1st_seg070 | base_plus/crop512 | 50 | 212 | 11/18/32 | 32.01 | 90/100/100 | 1.06 | 0% | 90/100/100 |
| bournemouth_1st_seg070 | base_plus/full | 10 | 195 | 91/95/97 | 1.68 | 94/99/100 | 1.21 | 0% | 93/99/99 |
| bournemouth_1st_seg070 | base_plus/full | 25 | 208 | 42/67/91 | 6.64 | 94/100/100 | 1.19 | 0% | 94/100/100 |
| bournemouth_1st_seg070 | base_plus/full | 50 | 212 | 11/18/32 | 32.01 | 94/100/100 | 1.23 | 0% | 92/100/100 |
| bournemouth_1st_seg070 | small/crop512 | 10 | 195 | 91/95/97 | 1.68 | 90/99/100 | 1.19 | 0% | 89/100/100 |
| bournemouth_1st_seg070 | small/crop512 | 25 | 208 | 42/67/91 | 6.64 | 91/100/100 | 1.1 | 0% | 91/100/100 |
| bournemouth_1st_seg070 | small/crop512 | 50 | 212 | 11/18/32 | 32.01 | 91/100/100 | 1.08 | 0% | 91/100/100 |
| bournemouth_1st_seg070 | small/full | 10 | 195 | 91/95/97 | 1.68 | 98/99/100 | 1.15 | 0% | 98/99/100 |
| bournemouth_1st_seg070 | small/full | 25 | 208 | 42/67/91 | 6.64 | 96/100/100 | 1.15 | 0% | 96/100/100 |
| bournemouth_1st_seg070 | small/full | 50 | 212 | 11/18/32 | 32.01 | 97/100/100 | 1.14 | 0% | 97/100/100 |
| bournemouth_1st_seg070 | tiny/crop512 | 10 | 195 | 91/95/97 | 1.68 | 91/100/100 | 1.03 | 0% | 91/100/100 |
| bournemouth_1st_seg070 | tiny/crop512 | 25 | 208 | 42/67/91 | 6.64 | 91/100/100 | 1.04 | 0% | 91/100/100 |
| bournemouth_1st_seg070 | tiny/crop512 | 50 | 212 | 11/18/32 | 32.01 | 91/100/100 | 1.06 | 0% | 91/100/100 |
| bournemouth_1st_seg070 | tiny/full | 10 | 195 | 91/95/97 | 1.68 | 95/99/100 | 1.03 | 0% | 96/99/100 |
| bournemouth_1st_seg070 | tiny/full | 25 | 208 | 42/67/91 | 6.64 | 95/100/100 | 0.93 | 0% | 95/100/100 |
| bournemouth_1st_seg070 | tiny/full | 50 | 212 | 11/18/32 | 32.01 | 97/100/100 | 0.99 | 0% | 98/100/100 |

### T2. Forward-only vs backward-only vs fused, by distance to the nearest anchor (hit@10 %, N = 25)

| clip | config | mode | 1-5 | 6-12 | 13-25 (n) |
|---|---|---|---|---|---|
| arsenal_example | base_plus/crop512 | fwd | 70 (n=44) | 72 (n=46) | 98 (n=50) |
| arsenal_example | base_plus/crop512 | bwd | 77 (n=44) | 80 (n=46) | 94 (n=50) |
| arsenal_example | base_plus/crop512 | nearest | 74 (n=78) | 89 (n=62) | - |
| arsenal_example | base_plus/crop512 | interp | 65 (n=78) | 32 (n=62) | - |
| arsenal_example | base_plus/full | fwd | 70 (n=44) | 65 (n=46) | 74 (n=50) |
| arsenal_example | base_plus/full | bwd | 75 (n=44) | 74 (n=46) | 86 (n=50) |
| arsenal_example | base_plus/full | nearest | 73 (n=78) | 79 (n=62) | - |
| arsenal_example | base_plus/full | interp | 65 (n=78) | 32 (n=62) | - |
| arsenal_example | small/crop512 | fwd | 73 (n=44) | 78 (n=46) | 98 (n=50) |
| arsenal_example | small/crop512 | bwd | 77 (n=44) | 80 (n=46) | 94 (n=50) |
| arsenal_example | small/crop512 | nearest | 76 (n=78) | 94 (n=62) | - |
| arsenal_example | small/crop512 | interp | 65 (n=78) | 32 (n=62) | - |
| arsenal_example | small/full | fwd | 70 (n=44) | 65 (n=46) | 74 (n=50) |
| arsenal_example | small/full | bwd | 73 (n=44) | 80 (n=46) | 90 (n=50) |
| arsenal_example | small/full | nearest | 72 (n=78) | 84 (n=62) | - |
| arsenal_example | small/full | interp | 65 (n=78) | 32 (n=62) | - |
| arsenal_example | tiny/crop512 | fwd | 70 (n=44) | 65 (n=46) | 74 (n=50) |
| arsenal_example | tiny/crop512 | bwd | 77 (n=44) | 80 (n=46) | 94 (n=50) |
| arsenal_example | tiny/crop512 | nearest | 74 (n=78) | 84 (n=62) | - |
| arsenal_example | tiny/crop512 | interp | 65 (n=78) | 32 (n=62) | - |
| arsenal_example | tiny/full | fwd | 70 (n=44) | 61 (n=46) | 74 (n=50) |
| arsenal_example | tiny/full | bwd | 73 (n=44) | 80 (n=46) | 86 (n=50) |
| arsenal_example | tiny/full | nearest | 72 (n=78) | 81 (n=62) | - |
| arsenal_example | tiny/full | interp | 65 (n=78) | 32 (n=62) | - |
| astonvilla_seg080 | base_plus/crop512 | fwd | 95 (n=122) | 93 (n=161) | 91 (n=256) |
| astonvilla_seg080 | base_plus/crop512 | bwd | 96 (n=122) | 84 (n=161) | 88 (n=256) |
| astonvilla_seg080 | base_plus/crop512 | nearest | 95 (n=239) | 89 (n=300) | - |
| astonvilla_seg080 | base_plus/crop512 | interp | 66 (n=239) | 54 (n=300) | - |
| astonvilla_seg080 | base_plus/full | fwd | 96 (n=122) | 95 (n=161) | 93 (n=256) |
| astonvilla_seg080 | base_plus/full | bwd | 93 (n=122) | 88 (n=161) | 90 (n=256) |
| astonvilla_seg080 | base_plus/full | nearest | 95 (n=239) | 93 (n=300) | - |
| astonvilla_seg080 | base_plus/full | interp | 66 (n=239) | 54 (n=300) | - |
| astonvilla_seg080 | small/crop512 | fwd | 96 (n=122) | 96 (n=161) | 93 (n=256) |
| astonvilla_seg080 | small/crop512 | bwd | 95 (n=122) | 89 (n=161) | 91 (n=256) |
| astonvilla_seg080 | small/crop512 | nearest | 95 (n=239) | 94 (n=300) | - |
| astonvilla_seg080 | small/crop512 | interp | 66 (n=239) | 54 (n=300) | - |
| astonvilla_seg080 | small/full | fwd | 96 (n=122) | 97 (n=161) | 96 (n=256) |
| astonvilla_seg080 | small/full | bwd | 94 (n=122) | 88 (n=161) | 89 (n=256) |
| astonvilla_seg080 | small/full | nearest | 95 (n=239) | 93 (n=300) | - |
| astonvilla_seg080 | small/full | interp | 66 (n=239) | 54 (n=300) | - |
| astonvilla_seg080 | tiny/crop512 | fwd | 96 (n=122) | 94 (n=161) | 94 (n=256) |
| astonvilla_seg080 | tiny/crop512 | bwd | 97 (n=122) | 87 (n=161) | 89 (n=256) |
| astonvilla_seg080 | tiny/crop512 | nearest | 96 (n=239) | 92 (n=300) | - |
| astonvilla_seg080 | tiny/crop512 | interp | 66 (n=239) | 54 (n=300) | - |
| astonvilla_seg080 | tiny/full | fwd | 95 (n=122) | 96 (n=161) | 95 (n=256) |
| astonvilla_seg080 | tiny/full | bwd | 94 (n=122) | 94 (n=161) | 93 (n=256) |
| astonvilla_seg080 | tiny/full | nearest | 95 (n=239) | 96 (n=300) | - |
| astonvilla_seg080 | tiny/full | interp | 66 (n=239) | 54 (n=300) | - |
| bournemouth_1st_seg070 | base_plus/crop512 | fwd | 100 (n=50) | 100 (n=62) | 100 (n=96) |
| bournemouth_1st_seg070 | base_plus/crop512 | bwd | 100 (n=50) | 100 (n=62) | 100 (n=96) |
| bournemouth_1st_seg070 | base_plus/crop512 | nearest | 100 (n=95) | 100 (n=113) | - |
| bournemouth_1st_seg070 | base_plus/crop512 | interp | 89 (n=95) | 48 (n=113) | - |
| bournemouth_1st_seg070 | base_plus/full | fwd | 100 (n=50) | 100 (n=62) | 100 (n=96) |
| bournemouth_1st_seg070 | base_plus/full | bwd | 100 (n=50) | 100 (n=62) | 100 (n=96) |
| bournemouth_1st_seg070 | base_plus/full | nearest | 100 (n=95) | 100 (n=113) | - |
| bournemouth_1st_seg070 | base_plus/full | interp | 89 (n=95) | 48 (n=113) | - |
| bournemouth_1st_seg070 | small/crop512 | fwd | 100 (n=50) | 100 (n=62) | 100 (n=96) |
| bournemouth_1st_seg070 | small/crop512 | bwd | 100 (n=50) | 100 (n=62) | 100 (n=96) |
| bournemouth_1st_seg070 | small/crop512 | nearest | 100 (n=95) | 100 (n=113) | - |
| bournemouth_1st_seg070 | small/crop512 | interp | 89 (n=95) | 48 (n=113) | - |
| bournemouth_1st_seg070 | small/full | fwd | 100 (n=50) | 100 (n=62) | 100 (n=96) |
| bournemouth_1st_seg070 | small/full | bwd | 100 (n=50) | 100 (n=62) | 100 (n=96) |
| bournemouth_1st_seg070 | small/full | nearest | 100 (n=95) | 100 (n=113) | - |
| bournemouth_1st_seg070 | small/full | interp | 89 (n=95) | 48 (n=113) | - |
| bournemouth_1st_seg070 | tiny/crop512 | fwd | 100 (n=50) | 100 (n=62) | 100 (n=96) |
| bournemouth_1st_seg070 | tiny/crop512 | bwd | 100 (n=50) | 100 (n=62) | 100 (n=96) |
| bournemouth_1st_seg070 | tiny/crop512 | nearest | 100 (n=95) | 100 (n=113) | - |
| bournemouth_1st_seg070 | tiny/crop512 | interp | 89 (n=95) | 48 (n=113) | - |
| bournemouth_1st_seg070 | tiny/full | fwd | 100 (n=50) | 100 (n=62) | 100 (n=96) |
| bournemouth_1st_seg070 | tiny/full | bwd | 100 (n=50) | 100 (n=62) | 100 (n=96) |
| bournemouth_1st_seg070 | tiny/full | nearest | 100 (n=95) | 100 (n=113) | - |
| bournemouth_1st_seg070 | tiny/full | interp | 89 (n=95) | 48 (n=113) | - |

### T3. Carry curve of a single prompt: hit@10 % / median error px vs frames from the anchor frame

Pooled over every anchor of every N (frames that are anchors in any N excluded). `hold` = keep the anchor position (no motion model). Absent counts as a miss.

| clip | config | direction | 1-5 | 6-12 | 13-25 | 26-50 |
|---|---|---|---|---|---|---|
| arsenal_example | base_plus/crop512 | fwd | 81 / 1.09 (n=88) | 80 / 1.18 (n=88) | 87 / 1.15 (n=98) | 76 / 1.18 (n=37) |
| arsenal_example | base_plus/crop512 | bwd | 84 / 1.02 (n=88) | 84 / 1.02 (n=102) | 78 / 0.88 (n=125) | 62 / 1.09 (n=37) |
| arsenal_example | base_plus/crop512 | hold | 47 / 10.62 (n=88) | 15 / 28.54 (n=88) | 3 / 60.41 (n=98) | 0 / 131.76 (n=37) |
| arsenal_example | base_plus/full | fwd | 77 / 1.62 (n=88) | 75 / 1.73 (n=88) | 72 / 1.62 (n=98) | 16 / 509.13 (n=37) |
| arsenal_example | base_plus/full | bwd | 82 / 2.02 (n=88) | 78 / 1.82 (n=102) | 70 / 2.07 (n=125) | 38 / 221.77 (n=37) |
| arsenal_example | base_plus/full | hold | 47 / 10.62 (n=88) | 15 / 28.54 (n=88) | 3 / 60.41 (n=98) | 0 / 131.76 (n=37) |
| arsenal_example | small/crop512 | fwd | 82 / 1.14 (n=88) | 82 / 1.18 (n=88) | 87 / 1.05 (n=98) | 76 / 1.34 (n=37) |
| arsenal_example | small/crop512 | bwd | 84 / 1.23 (n=88) | 84 / 1.15 (n=102) | 78 / 1.23 (n=125) | 65 / 1.22 (n=37) |
| arsenal_example | small/crop512 | hold | 47 / 10.62 (n=88) | 15 / 28.54 (n=88) | 3 / 60.41 (n=98) | 0 / 131.76 (n=37) |
| arsenal_example | small/full | fwd | 77 / 1.9 (n=88) | 73 / 1.82 (n=88) | 72 / 1.68 (n=98) | 65 / 1.46 (n=37) |
| arsenal_example | small/full | bwd | 75 / 1.94 (n=88) | 78 / 1.93 (n=102) | 68 / 4.77 (n=125) | 19 / 135.09 (n=37) |
| arsenal_example | small/full | hold | 47 / 10.62 (n=88) | 15 / 28.54 (n=88) | 3 / 60.41 (n=98) | 0 / 131.76 (n=37) |
| arsenal_example | tiny/crop512 | fwd | 81 / 1.09 (n=88) | 76 / 1.18 (n=88) | 76 / 1.22 (n=98) | 19 / 2.11 (n=37) |
| arsenal_example | tiny/crop512 | bwd | 84 / 1.14 (n=88) | 84 / 1.17 (n=102) | 76 / 1.24 (n=125) | 62 / 1.36 (n=37) |
| arsenal_example | tiny/crop512 | hold | 47 / 10.62 (n=88) | 15 / 28.54 (n=88) | 3 / 60.41 (n=98) | 0 / 131.76 (n=37) |
| arsenal_example | tiny/full | fwd | 77 / 1.98 (n=88) | 72 / 2.2 (n=88) | 73 / 2.79 (n=98) | 16 / 263.25 (n=37) |
| arsenal_example | tiny/full | bwd | 75 / 2.48 (n=88) | 74 / 2.23 (n=102) | 62 / 4.2 (n=125) | 14 / 134.5 (n=37) |
| arsenal_example | tiny/full | hold | 47 / 10.62 (n=88) | 15 / 28.54 (n=88) | 3 / 60.41 (n=98) | 0 / 131.76 (n=37) |
| astonvilla_seg080 | base_plus/crop512 | fwd | 95 / 1.12 (n=316) | 92 / 1.05 (n=406) | 89 / 1.19 (n=696) | 91 / 1.16 (n=234) |
| astonvilla_seg080 | base_plus/crop512 | bwd | 98 / 1.1 (n=325) | 88 / 1.17 (n=424) | 84 / 1.17 (n=738) | 84 / 1.13 (n=300) |
| astonvilla_seg080 | base_plus/crop512 | hold | 64 / 5.12 (n=316) | 40 / 19.36 (n=406) | 21 / 37.53 (n=696) | 9 / 104.75 (n=234) |
| astonvilla_seg080 | base_plus/full | fwd | 97 / 1.18 (n=316) | 94 / 1.11 (n=406) | 90 / 1.23 (n=696) | 88 / 1.16 (n=234) |
| astonvilla_seg080 | base_plus/full | bwd | 97 / 1.24 (n=325) | 90 / 1.25 (n=424) | 85 / 1.22 (n=738) | 78 / 1.29 (n=300) |
| astonvilla_seg080 | base_plus/full | hold | 64 / 5.12 (n=316) | 40 / 19.36 (n=406) | 21 / 37.53 (n=696) | 9 / 104.75 (n=234) |
| astonvilla_seg080 | small/crop512 | fwd | 96 / 1.35 (n=316) | 94 / 1.39 (n=406) | 90 / 1.49 (n=696) | 93 / 1.51 (n=234) |
| astonvilla_seg080 | small/crop512 | bwd | 98 / 1.38 (n=325) | 92 / 1.44 (n=424) | 85 / 1.44 (n=738) | 77 / 1.53 (n=300) |
| astonvilla_seg080 | small/crop512 | hold | 64 / 5.12 (n=316) | 40 / 19.36 (n=406) | 21 / 37.53 (n=696) | 9 / 104.75 (n=234) |
| astonvilla_seg080 | small/full | fwd | 97 / 1.46 (n=316) | 96 / 1.49 (n=406) | 92 / 1.56 (n=696) | 96 / 1.49 (n=234) |
| astonvilla_seg080 | small/full | bwd | 96 / 1.53 (n=325) | 90 / 1.53 (n=424) | 83 / 1.51 (n=738) | 73 / 1.65 (n=300) |
| astonvilla_seg080 | small/full | hold | 64 / 5.12 (n=316) | 40 / 19.36 (n=406) | 21 / 37.53 (n=696) | 9 / 104.75 (n=234) |
| astonvilla_seg080 | tiny/crop512 | fwd | 96 / 1.23 (n=316) | 93 / 1.2 (n=406) | 89 / 1.31 (n=696) | 94 / 1.31 (n=234) |
| astonvilla_seg080 | tiny/crop512 | bwd | 98 / 1.23 (n=325) | 91 / 1.29 (n=424) | 84 / 1.28 (n=738) | 75 / 1.35 (n=300) |
| astonvilla_seg080 | tiny/crop512 | hold | 64 / 5.12 (n=316) | 40 / 19.36 (n=406) | 21 / 37.53 (n=696) | 9 / 104.75 (n=234) |
| astonvilla_seg080 | tiny/full | fwd | 97 / 1.28 (n=316) | 95 / 1.34 (n=406) | 92 / 1.42 (n=696) | 96 / 1.38 (n=234) |
| astonvilla_seg080 | tiny/full | bwd | 97 / 1.34 (n=325) | 92 / 1.43 (n=424) | 86 / 1.35 (n=738) | 68 / 1.62 (n=300) |
| astonvilla_seg080 | tiny/full | hold | 64 / 5.12 (n=316) | 40 / 19.36 (n=406) | 21 / 37.53 (n=696) | 9 / 104.75 (n=234) |
| bournemouth_1st_seg070 | base_plus/crop512 | fwd | 100 / 1.05 (n=123) | 100 / 1.21 (n=153) | 98 / 1.12 (n=263) | 100 / 1.55 (n=88) |
| bournemouth_1st_seg070 | base_plus/crop512 | bwd | 100 / 0.98 (n=126) | 99 / 1.04 (n=162) | 98 / 1.19 (n=278) | 100 / 0.86 (n=131) |
| bournemouth_1st_seg070 | base_plus/crop512 | hold | 55 / 8.54 (n=123) | 6 / 27.17 (n=153) | 5 / 55.94 (n=263) | 0 / 131.81 (n=88) |
| bournemouth_1st_seg070 | base_plus/full | fwd | 99 / 1.13 (n=123) | 98 / 1.33 (n=153) | 98 / 1.34 (n=263) | 100 / 1.53 (n=88) |
| bournemouth_1st_seg070 | base_plus/full | bwd | 100 / 1.16 (n=126) | 100 / 1.05 (n=162) | 100 / 1.28 (n=278) | 100 / 0.96 (n=131) |
| bournemouth_1st_seg070 | base_plus/full | hold | 55 / 8.54 (n=123) | 6 / 27.17 (n=153) | 5 / 55.94 (n=263) | 0 / 131.81 (n=88) |
| bournemouth_1st_seg070 | small/crop512 | fwd | 99 / 1.01 (n=123) | 98 / 1.37 (n=153) | 97 / 1.12 (n=263) | 100 / 1.46 (n=88) |
| bournemouth_1st_seg070 | small/crop512 | bwd | 100 / 1.02 (n=126) | 96 / 1.04 (n=162) | 93 / 1.38 (n=278) | 100 / 0.86 (n=131) |
| bournemouth_1st_seg070 | small/crop512 | hold | 55 / 8.54 (n=123) | 6 / 27.17 (n=153) | 5 / 55.94 (n=263) | 0 / 131.81 (n=88) |
| bournemouth_1st_seg070 | small/full | fwd | 99 / 1.14 (n=123) | 99 / 1.18 (n=153) | 98 / 1.2 (n=263) | 100 / 1.3 (n=88) |
| bournemouth_1st_seg070 | small/full | bwd | 100 / 1.14 (n=126) | 100 / 1.14 (n=162) | 100 / 1.16 (n=278) | 100 / 1.01 (n=131) |
| bournemouth_1st_seg070 | small/full | hold | 55 / 8.54 (n=123) | 6 / 27.17 (n=153) | 5 / 55.94 (n=263) | 0 / 131.81 (n=88) |
| bournemouth_1st_seg070 | tiny/crop512 | fwd | 100 / 1.02 (n=123) | 100 / 1.14 (n=153) | 97 / 1.02 (n=263) | 100 / 1.33 (n=88) |
| bournemouth_1st_seg070 | tiny/crop512 | bwd | 100 / 0.97 (n=126) | 99 / 1.03 (n=162) | 98 / 1.23 (n=278) | 100 / 0.85 (n=131) |
| bournemouth_1st_seg070 | tiny/crop512 | hold | 55 / 8.54 (n=123) | 6 / 27.17 (n=153) | 5 / 55.94 (n=263) | 0 / 131.81 (n=88) |
| bournemouth_1st_seg070 | tiny/full | fwd | 98 / 0.97 (n=123) | 97 / 1.05 (n=153) | 100 / 0.99 (n=263) | 100 / 1.38 (n=88) |
| bournemouth_1st_seg070 | tiny/full | bwd | 100 / 1.01 (n=126) | 100 / 0.89 (n=162) | 97 / 1.08 (n=278) | 100 / 0.85 (n=131) |
| bournemouth_1st_seg070 | tiny/full | hold | 55 / 8.54 (n=123) | 6 / 27.17 (n=153) | 5 / 55.94 (n=263) | 0 / 131.81 (n=88) |

### T4. Fused (nearest) vs linear interpolation by distance, N = 50 (hit@10 %)

| clip | config | mode | 1-5 | 6-12 | 13-25 |
|---|---|---|---|---|---|
| arsenal_example | base_plus/crop512 | nearest | 69 (n=54) | 76 (n=46) | 84 (n=43) |
| arsenal_example | base_plus/crop512 | interp | 65 (n=54) | 39 (n=46) | 14 (n=43) |
| arsenal_example | base_plus/full | nearest | 70 (n=54) | 70 (n=46) | 56 (n=43) |
| arsenal_example | base_plus/full | interp | 65 (n=54) | 39 (n=46) | 14 (n=43) |
| arsenal_example | small/crop512 | nearest | 69 (n=54) | 83 (n=46) | 84 (n=43) |
| arsenal_example | small/crop512 | interp | 65 (n=54) | 39 (n=46) | 14 (n=43) |
| arsenal_example | small/full | nearest | 69 (n=54) | 70 (n=46) | 56 (n=43) |
| arsenal_example | small/full | interp | 65 (n=54) | 39 (n=46) | 14 (n=43) |
| arsenal_example | tiny/crop512 | nearest | 69 (n=54) | 70 (n=46) | 56 (n=43) |
| arsenal_example | tiny/crop512 | interp | 65 (n=54) | 39 (n=46) | 14 (n=43) |
| arsenal_example | tiny/full | nearest | 69 (n=54) | 65 (n=46) | 56 (n=43) |
| arsenal_example | tiny/full | interp | 65 (n=54) | 39 (n=46) | 14 (n=43) |
| astonvilla_seg080 | base_plus/crop512 | nearest | 95 (n=129) | 92 (n=154) | 91 (n=267) |
| astonvilla_seg080 | base_plus/crop512 | interp | 57 (n=129) | 37 (n=154) | 37 (n=267) |
| astonvilla_seg080 | base_plus/full | nearest | 91 (n=129) | 91 (n=154) | 89 (n=267) |
| astonvilla_seg080 | base_plus/full | interp | 57 (n=129) | 37 (n=154) | 37 (n=267) |
| astonvilla_seg080 | small/crop512 | nearest | 95 (n=129) | 92 (n=154) | 90 (n=267) |
| astonvilla_seg080 | small/crop512 | interp | 57 (n=129) | 37 (n=154) | 37 (n=267) |
| astonvilla_seg080 | small/full | nearest | 92 (n=129) | 93 (n=154) | 90 (n=267) |
| astonvilla_seg080 | small/full | interp | 57 (n=129) | 37 (n=154) | 37 (n=267) |
| astonvilla_seg080 | tiny/crop512 | nearest | 95 (n=129) | 93 (n=154) | 91 (n=267) |
| astonvilla_seg080 | tiny/crop512 | interp | 57 (n=129) | 37 (n=154) | 37 (n=267) |
| astonvilla_seg080 | tiny/full | nearest | 91 (n=129) | 95 (n=154) | 93 (n=267) |
| astonvilla_seg080 | tiny/full | interp | 57 (n=129) | 37 (n=154) | 37 (n=267) |
| bournemouth_1st_seg070 | base_plus/crop512 | nearest | 100 (n=55) | 100 (n=57) | 100 (n=100) |
| bournemouth_1st_seg070 | base_plus/crop512 | interp | 65 (n=55) | 4 (n=57) | 0 (n=100) |
| bournemouth_1st_seg070 | base_plus/full | nearest | 100 (n=55) | 100 (n=57) | 100 (n=100) |
| bournemouth_1st_seg070 | base_plus/full | interp | 65 (n=55) | 4 (n=57) | 0 (n=100) |
| bournemouth_1st_seg070 | small/crop512 | nearest | 100 (n=55) | 100 (n=57) | 100 (n=100) |
| bournemouth_1st_seg070 | small/crop512 | interp | 65 (n=55) | 4 (n=57) | 0 (n=100) |
| bournemouth_1st_seg070 | small/full | nearest | 100 (n=55) | 100 (n=57) | 100 (n=100) |
| bournemouth_1st_seg070 | small/full | interp | 65 (n=55) | 4 (n=57) | 0 (n=100) |
| bournemouth_1st_seg070 | tiny/crop512 | nearest | 100 (n=55) | 100 (n=57) | 100 (n=100) |
| bournemouth_1st_seg070 | tiny/crop512 | interp | 65 (n=55) | 4 (n=57) | 0 (n=100) |
| bournemouth_1st_seg070 | tiny/full | nearest | 100 (n=55) | 100 (n=57) | 100 (n=100) |
| bournemouth_1st_seg070 | tiny/full | interp | 65 (n=55) | 4 (n=57) | 0 (n=100) |

### T5. Forward/backward agreement gate (both passes present and within 5 px of each other)

| clip | config | N | held-out | gate passes (coverage) | hit@10 of gated frames | median err |
|---|---|---|---|---|---|---|
| arsenal_example | base_plus/crop512 | 10 | 133 | 100 (75%) | 100% | 0.93 |
| arsenal_example | base_plus/crop512 | 25 | 140 | 109 (78%) | 100% | 1.02 |
| arsenal_example | base_plus/crop512 | 50 | 143 | 88 (62%) | 100% | 0.95 |
| arsenal_example | base_plus/full | 10 | 133 | 94 (71%) | 100% | 1.31 |
| arsenal_example | base_plus/full | 25 | 140 | 94 (67%) | 99% | 1.52 |
| arsenal_example | base_plus/full | 50 | 143 | 50 (35%) | 100% | 1.25 |
| arsenal_example | small/crop512 | 10 | 133 | 102 (77%) | 100% | 0.99 |
| arsenal_example | small/crop512 | 25 | 140 | 113 (81%) | 100% | 1.09 |
| arsenal_example | small/crop512 | 50 | 143 | 92 (64%) | 100% | 0.94 |
| arsenal_example | small/full | 10 | 133 | 83 (62%) | 100% | 1.42 |
| arsenal_example | small/full | 25 | 140 | 91 (65%) | 100% | 1.91 |
| arsenal_example | small/full | 50 | 143 | 69 (48%) | 100% | 1.65 |
| arsenal_example | tiny/crop512 | 10 | 133 | 99 (74%) | 100% | 1.0 |
| arsenal_example | tiny/crop512 | 25 | 140 | 95 (68%) | 100% | 1.04 |
| arsenal_example | tiny/crop512 | 50 | 143 | 50 (35%) | 100% | 0.96 |
| arsenal_example | tiny/full | 10 | 133 | 84 (63%) | 100% | 1.3 |
| arsenal_example | tiny/full | 25 | 140 | 89 (64%) | 99% | 2.12 |
| arsenal_example | tiny/full | 50 | 143 | 45 (31%) | 100% | 2.37 |
| astonvilla_seg080 | base_plus/crop512 | 10 | 505 | 434 (86%) | 99% | 1.02 |
| astonvilla_seg080 | base_plus/crop512 | 25 | 539 | 461 (86%) | 96% | 1.07 |
| astonvilla_seg080 | base_plus/crop512 | 50 | 550 | 473 (86%) | 96% | 1.06 |
| astonvilla_seg080 | base_plus/full | 10 | 505 | 438 (87%) | 98% | 1.06 |
| astonvilla_seg080 | base_plus/full | 25 | 539 | 471 (87%) | 96% | 1.09 |
| astonvilla_seg080 | base_plus/full | 50 | 550 | 444 (81%) | 98% | 1.06 |
| astonvilla_seg080 | small/crop512 | 10 | 505 | 432 (86%) | 99% | 1.3 |
| astonvilla_seg080 | small/crop512 | 25 | 539 | 483 (90%) | 96% | 1.36 |
| astonvilla_seg080 | small/crop512 | 50 | 550 | 472 (86%) | 96% | 1.35 |
| astonvilla_seg080 | small/full | 10 | 505 | 431 (85%) | 99% | 1.39 |
| astonvilla_seg080 | small/full | 25 | 539 | 458 (85%) | 97% | 1.4 |
| astonvilla_seg080 | small/full | 50 | 550 | 450 (82%) | 98% | 1.44 |
| astonvilla_seg080 | tiny/crop512 | 10 | 505 | 436 (86%) | 99% | 1.14 |
| astonvilla_seg080 | tiny/crop512 | 25 | 539 | 467 (87%) | 97% | 1.2 |
| astonvilla_seg080 | tiny/crop512 | 50 | 550 | 471 (86%) | 97% | 1.21 |
| astonvilla_seg080 | tiny/full | 10 | 505 | 439 (87%) | 99% | 1.21 |
| astonvilla_seg080 | tiny/full | 25 | 539 | 481 (89%) | 97% | 1.22 |
| astonvilla_seg080 | tiny/full | 50 | 550 | 434 (79%) | 98% | 1.21 |
| bournemouth_1st_seg070 | base_plus/crop512 | 10 | 195 | 190 (97%) | 100% | 1.04 |
| bournemouth_1st_seg070 | base_plus/crop512 | 25 | 208 | 208 (100%) | 100% | 1.06 |
| bournemouth_1st_seg070 | base_plus/crop512 | 50 | 212 | 212 (100%) | 100% | 1.06 |
| bournemouth_1st_seg070 | base_plus/full | 10 | 195 | 187 (96%) | 100% | 1.13 |
| bournemouth_1st_seg070 | base_plus/full | 25 | 208 | 207 (100%) | 100% | 1.17 |
| bournemouth_1st_seg070 | base_plus/full | 50 | 212 | 209 (99%) | 100% | 1.21 |
| bournemouth_1st_seg070 | small/crop512 | 10 | 195 | 169 (87%) | 100% | 0.94 |
| bournemouth_1st_seg070 | small/crop512 | 25 | 208 | 208 (100%) | 100% | 1.1 |
| bournemouth_1st_seg070 | small/crop512 | 50 | 212 | 212 (100%) | 100% | 1.08 |
| bournemouth_1st_seg070 | small/full | 10 | 195 | 187 (96%) | 100% | 1.12 |
| bournemouth_1st_seg070 | small/full | 25 | 208 | 205 (99%) | 100% | 1.14 |
| bournemouth_1st_seg070 | small/full | 50 | 212 | 209 (99%) | 100% | 1.14 |
| bournemouth_1st_seg070 | tiny/crop512 | 10 | 195 | 188 (96%) | 100% | 1.01 |
| bournemouth_1st_seg070 | tiny/crop512 | 25 | 208 | 208 (100%) | 100% | 1.04 |
| bournemouth_1st_seg070 | tiny/crop512 | 50 | 212 | 212 (100%) | 100% | 1.04 |
| bournemouth_1st_seg070 | tiny/full | 10 | 195 | 180 (92%) | 100% | 0.96 |
| bournemouth_1st_seg070 | tiny/full | 25 | 208 | 206 (99%) | 100% | 0.93 |
| bournemouth_1st_seg070 | tiny/full | 50 | 212 | 209 (99%) | 100% | 0.99 |

### T6. False positives on explicit no_ball frames, and not_broadcast frames

Counts are (run, frame) pairs: a frame adjacent to two runs is counted once per run. Only astonvilla_seg080 has no_ball frames.

| clip | config | no_ball frames | SAM2 predicts a ball | not_broadcast frames | SAM2 predicts a ball |
|---|---|---|---|---|---|
| astonvilla_seg080 | base_plus/crop512 | 5 | 2 | 0 | 0 |
| astonvilla_seg080 | base_plus/full | 5 | 3 | 0 | 0 |
| astonvilla_seg080 | small/crop512 | 5 | 2 | 0 | 0 |
| astonvilla_seg080 | small/full | 5 | 4 | 0 | 0 |
| astonvilla_seg080 | tiny/crop512 | 5 | 2 | 0 | 0 |
| astonvilla_seg080 | tiny/full | 5 | 3 | 0 | 0 |
| bournemouth_1st_seg070 | base_plus/crop512 | 0 | 0 | 4 | 0 |
| bournemouth_1st_seg070 | base_plus/full | 0 | 0 | 4 | 4 |
| bournemouth_1st_seg070 | small/crop512 | 0 | 0 | 4 | 0 |
| bournemouth_1st_seg070 | small/full | 0 | 0 | 4 | 4 |
| bournemouth_1st_seg070 | tiny/crop512 | 0 | 0 | 4 | 0 |
| bournemouth_1st_seg070 | tiny/full | 0 | 0 | 4 | 4 |

### T7. Failure categories of the nearest-anchor prediction (N = 25), by ball speed (px/frame at 1080p)

| clip | config | speed | n | hit<=10 | near 10-20 | wrong, on player/ref box | wrong, elsewhere | absent |
|---|---|---|---|---|---|---|---|---|
| arsenal_example | base_plus/crop512 | slow(<8px/f) | 98 | 96% | 0% | 0% | 0% | 4% |
| arsenal_example | base_plus/crop512 | mid(8-20) | 32 | 59% | 3% | 0% | 9% | 28% |
| arsenal_example | base_plus/crop512 | fast(>20) | 10 | 0% | 0% | 0% | 30% | 70% |
| arsenal_example | base_plus/crop512 | ALL | 140 | 81% | 1% | 0% | 4% | 14% |
| arsenal_example | base_plus/full | slow(<8px/f) | 98 | 89% | 4% | 0% | 1% | 6% |
| arsenal_example | base_plus/full | mid(8-20) | 32 | 59% | 6% | 3% | 19% | 12% |
| arsenal_example | base_plus/full | fast(>20) | 10 | 0% | 0% | 20% | 70% | 10% |
| arsenal_example | base_plus/full | ALL | 140 | 76% | 4% | 2% | 10% | 8% |
| arsenal_example | small/crop512 | slow(<8px/f) | 98 | 98% | 0% | 0% | 0% | 2% |
| arsenal_example | small/crop512 | mid(8-20) | 32 | 66% | 3% | 0% | 12% | 19% |
| arsenal_example | small/crop512 | fast(>20) | 10 | 0% | 0% | 0% | 20% | 80% |
| arsenal_example | small/crop512 | ALL | 140 | 84% | 1% | 0% | 4% | 11% |
| arsenal_example | small/full | slow(<8px/f) | 98 | 91% | 1% | 0% | 0% | 8% |
| arsenal_example | small/full | mid(8-20) | 32 | 59% | 6% | 0% | 22% | 12% |
| arsenal_example | small/full | fast(>20) | 10 | 0% | 0% | 20% | 70% | 10% |
| arsenal_example | small/full | ALL | 140 | 77% | 2% | 1% | 10% | 9% |
| arsenal_example | tiny/crop512 | slow(<8px/f) | 98 | 93% | 0% | 0% | 0% | 7% |
| arsenal_example | tiny/crop512 | mid(8-20) | 32 | 59% | 3% | 3% | 16% | 19% |
| arsenal_example | tiny/crop512 | fast(>20) | 10 | 0% | 0% | 10% | 40% | 50% |
| arsenal_example | tiny/crop512 | ALL | 140 | 79% | 1% | 1% | 6% | 13% |
| arsenal_example | tiny/full | slow(<8px/f) | 98 | 90% | 3% | 2% | 0% | 5% |
| arsenal_example | tiny/full | mid(8-20) | 32 | 56% | 9% | 3% | 19% | 12% |
| arsenal_example | tiny/full | fast(>20) | 10 | 0% | 0% | 20% | 70% | 10% |
| arsenal_example | tiny/full | ALL | 140 | 76% | 4% | 4% | 9% | 7% |
| astonvilla_seg080 | base_plus/crop512 | slow(<8px/f) | 413 | 94% | 3% | 0% | 1% | 1% |
| astonvilla_seg080 | base_plus/crop512 | mid(8-20) | 111 | 85% | 1% | 13% | 0% | 2% |
| astonvilla_seg080 | base_plus/crop512 | fast(>20) | 15 | 93% | 0% | 7% | 0% | 0% |
| astonvilla_seg080 | base_plus/crop512 | ALL | 539 | 92% | 3% | 3% | 1% | 1% |
| astonvilla_seg080 | base_plus/full | slow(<8px/f) | 413 | 95% | 4% | 0% | 1% | 0% |
| astonvilla_seg080 | base_plus/full | mid(8-20) | 111 | 89% | 3% | 8% | 0% | 0% |
| astonvilla_seg080 | base_plus/full | fast(>20) | 15 | 93% | 0% | 7% | 0% | 0% |
| astonvilla_seg080 | base_plus/full | ALL | 539 | 94% | 3% | 2% | 1% | 0% |
| astonvilla_seg080 | small/crop512 | slow(<8px/f) | 413 | 95% | 3% | 0% | 1% | 0% |
| astonvilla_seg080 | small/crop512 | mid(8-20) | 111 | 93% | 1% | 5% | 0% | 1% |
| astonvilla_seg080 | small/crop512 | fast(>20) | 15 | 93% | 0% | 7% | 0% | 0% |
| astonvilla_seg080 | small/crop512 | ALL | 539 | 94% | 3% | 1% | 1% | 0% |
| astonvilla_seg080 | small/full | slow(<8px/f) | 413 | 95% | 2% | 1% | 2% | 0% |
| astonvilla_seg080 | small/full | mid(8-20) | 111 | 91% | 4% | 5% | 0% | 1% |
| astonvilla_seg080 | small/full | fast(>20) | 15 | 93% | 0% | 7% | 0% | 0% |
| astonvilla_seg080 | small/full | ALL | 539 | 94% | 2% | 2% | 1% | 0% |
| astonvilla_seg080 | tiny/crop512 | slow(<8px/f) | 413 | 96% | 2% | 0% | 1% | 1% |
| astonvilla_seg080 | tiny/crop512 | mid(8-20) | 111 | 88% | 1% | 10% | 0% | 1% |
| astonvilla_seg080 | tiny/crop512 | fast(>20) | 15 | 93% | 0% | 7% | 0% | 0% |
| astonvilla_seg080 | tiny/crop512 | ALL | 539 | 94% | 2% | 2% | 1% | 1% |
| astonvilla_seg080 | tiny/full | slow(<8px/f) | 413 | 96% | 2% | 0% | 1% | 0% |
| astonvilla_seg080 | tiny/full | mid(8-20) | 111 | 93% | 4% | 4% | 0% | 0% |
| astonvilla_seg080 | tiny/full | fast(>20) | 15 | 93% | 0% | 7% | 0% | 0% |
| astonvilla_seg080 | tiny/full | ALL | 539 | 95% | 3% | 1% | 1% | 0% |
| bournemouth_1st_seg070 | base_plus/crop512 | slow(<8px/f) | 181 | 100% | 0% | 0% | 0% | 0% |
| bournemouth_1st_seg070 | base_plus/crop512 | mid(8-20) | 27 | 100% | 0% | 0% | 0% | 0% |
| bournemouth_1st_seg070 | base_plus/crop512 | ALL | 208 | 100% | 0% | 0% | 0% | 0% |
| bournemouth_1st_seg070 | base_plus/full | slow(<8px/f) | 181 | 100% | 0% | 0% | 0% | 0% |
| bournemouth_1st_seg070 | base_plus/full | mid(8-20) | 27 | 100% | 0% | 0% | 0% | 0% |
| bournemouth_1st_seg070 | base_plus/full | ALL | 208 | 100% | 0% | 0% | 0% | 0% |
| bournemouth_1st_seg070 | small/crop512 | slow(<8px/f) | 181 | 100% | 0% | 0% | 0% | 0% |
| bournemouth_1st_seg070 | small/crop512 | mid(8-20) | 27 | 100% | 0% | 0% | 0% | 0% |
| bournemouth_1st_seg070 | small/crop512 | ALL | 208 | 100% | 0% | 0% | 0% | 0% |
| bournemouth_1st_seg070 | small/full | slow(<8px/f) | 181 | 100% | 0% | 0% | 0% | 0% |
| bournemouth_1st_seg070 | small/full | mid(8-20) | 27 | 100% | 0% | 0% | 0% | 0% |
| bournemouth_1st_seg070 | small/full | ALL | 208 | 100% | 0% | 0% | 0% | 0% |
| bournemouth_1st_seg070 | tiny/crop512 | slow(<8px/f) | 181 | 100% | 0% | 0% | 0% | 0% |
| bournemouth_1st_seg070 | tiny/crop512 | mid(8-20) | 27 | 100% | 0% | 0% | 0% | 0% |
| bournemouth_1st_seg070 | tiny/crop512 | ALL | 208 | 100% | 0% | 0% | 0% | 0% |
| bournemouth_1st_seg070 | tiny/full | slow(<8px/f) | 181 | 100% | 0% | 0% | 0% | 0% |
| bournemouth_1st_seg070 | tiny/full | mid(8-20) | 27 | 100% | 0% | 0% | 0% | 0% |
| bournemouth_1st_seg070 | tiny/full | ALL | 208 | 100% | 0% | 0% | 0% | 0% |

### T8. Runtime and peak VRAM (RTX 2070 8 GB, fp16 autocast, offload_video_to_cpu)

`s/frame` = wall time per propagated frame, including image preprocessing and mask post-processing. Peak VRAM is torch max_memory_allocated / reserved over the run (model weights included).

| config | frames propagated | s/frame | peak VRAM alloc MB | peak VRAM reserved MB |
|---|---|---|---|---|
| base_plus/crop512 | 6165 | 0.165 | 957 | 1172 |
| base_plus/full | 6165 | 0.173 | 963 | 1154 |
| small/crop512 | 6165 | 0.149 | 731 | 946 |
| small/full | 6165 | 0.163 | 734 | 860 |
| tiny/crop512 | 6165 | 0.132 | 689 | 878 |
| tiny/full | 6165 | 0.138 | 695 | 846 |

## Reproduce

```bash
# stage code snapshot on the mini, then (Prefect, gpu pool)
rsync -a experiments/sam2_bidir_ball/ /mnt/storage/footy/sam2_exp/code/
~/.claude/skills/prefect-jobs/scripts/pf.sh run sam2-ball/sam2-ball run_name=main \
    clips=arsenal_example,astonvilla_seg080,bournemouth_1st_seg070 models=small variants=full,crop512
# direct (what the last two runs used)
python experiments/sam2_bidir_ball/sam2_tracks.py --clips <c> --data <dir with videos/ gt/> --out <dir> --ckpt <dir> --models small --variants full,crop512
# CPU scoring and tables
python experiments/sam2_bidir_ball/score.py <tracks dir> --gt-dir <ball_gt_marks>
python experiments/sam2_bidir_ball/report.py <tracks dir>/summary.json
```

Environment on the desktop: `~/sam2-exp/venv` (torch 2.8.0 + torchvision 0.23.0 from PyPI, `sam2` 1.1.0 from PyPI built with `SAM2_BUILD_CUDA=0`, opencv-headless, ffmpeg CLI), checkpoints `sam2.1_hiera_{tiny,small,base_plus}.pt` from `dl.fbaipublicfiles.com/segment_anything_2/092824/`.


Per-frame predictions: `docs/experiments/sam2_bidir_ball_raw/`; aggregates: `docs/experiments/sam2_bidir_ball.json`.
