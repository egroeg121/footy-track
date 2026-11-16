# SAM 3 / SAM 3.1 as a slow, offline, background ball labeller

Branch `exp/sam3-offline-labeller` (from `exp/temporal-detector` @ `331ae64`).  Code: `experiments/sam3_offline/` (README there), Prefect flows `~/code/jobs/footy_sam3.py` + `sam3.prefect.yaml`,
numbers: `docs/experiments/sam3_offline_labeller.json` (+ `experiments/sam3_offline/results/`), sheets: `docs/experiments/sam3_offline_labeller/`.
Builds on `temporal_detector_round3.md` (held-out list, 50 fps handling, S1 gate, SAM2.1-small crop512 tracker).  All GPU runs were Prefect flow runs on the `gpu` pool (desktop RTX 2070 8 GB), one at a time.

**Question.**  Can SAM 3 / 3.1 label balls offline: one known frame (a confident RT-DETR box, an overview, a human click) -> track forwards and backwards through long stretches -> labels written by a queued background job?
Speed does not matter, offline quality and throughput do.

## Answers in brief

1. **How far does one seed carry?  SAM 3 is no better than the SAM2.1-small crop512 tracker we already have.**  Held-out clips + the 359-frame Aston Villa run, oracle seeds (1 217 scored frames): recall@10 **0.943 (SAM 3) vs 0.945 (SAM2.1)**,
   paired -0.002 [-0.007, +0.004]; on the six sparse held-out clips both are at **1.000** (251 frames, up to 100-250 frames from the seed); with RT-DETR conf >= 0.8 seeds both are **0.996** (228 frames).  The only place SAM 3 separates is
   `arsenal_example` (an out-of-play ball in the crowd, practically invisible): 0.301 vs 0.250 (+0.051 [+0.032, +0.072], one clip; both useless).  The set is saturated, so **it cannot show a gain and does not show one**; SAM 3 costs
   3x the time (2.4 vs 7.0 frames/s) and 6x the memory (4.9 vs 0.8 GB).  SAM 3.1 (multiplex) is equal to SAM 3 here (0.928 vs 0.932 on dev) and slower (2.1 fps, 6.2 GB).
2. **What helps.**  A box seed (box ~ box + point > point-only: -0.020), a **crop that follows the ball at 512 px (round-3 size; 768 is the same, 384 -0.020, 256 -0.244, no crop -0.039)**, fp16 (fp32 gives identical labels at 3.2x the time).  What does not:
   text prompts as the tracker (0.73 / 0.63 vs 0.98, no memory), `frame_step 2` (2x faster but -0.051), the "offload state to CPU" memory setting (-0.28, a bug-like loss and no VRAM saved), crop 256.  Gating: SAM 3's object-score logit is a better gate than SAM2's
   (score > 8: precision 0.996 at 59 % coverage vs 1.0 at 37 %); the S1 gate (forward track of seed a x backward track of seed b within 5 px) was right 14/14 times in the production demo, whereas a score gate cannot catch a **false seed**
   (an RT-DETR conf 0.818 "ball" on a yellow boot was tracked as a boot for ~90 frames with logit 8-10).  Recovery and the rest of the tricks: see the sections below.
3. **SAM 3 as a seed finder is the real finding.**  Text prompt "ball" on 384 px tiles upscaled to 1008: top-1 hit@10 **0.85** vs RT-DETR-L **0.60** on the same 94 held-out GT frames (Bournemouth 0.85 vs 0.51; Man City 0.86 vs 0.90), any-candidate recall 0.96 vs 0.71.
   "yellow ball" is equal on Bournemouth (0.86) and useless on a white ball (0.00): the prompt must follow the match.  Whole-frame text detection (one 1008 px pass) is already 0.76; 540 px tiles are worse (0.71); "white ball" / "small white ball" fail (0.14 / 0.11).
   It costs ~5-6 s per frame at 384 px tiles (15 tiles).  So SAM 3 *can* find its own seeds, slowly.  Caveat: on the 80 dev frames (clips RT-DETR was trained on, many blurred in-play balls) it is *worse* than RT-DETR (0.64 vs 0.85), so the claim holds for footage RT-DETR has not seen.
4. **Throughput / cost on the RTX 2070.**  SAM 3 tracker 0.42 s/frame (2.38 fps) after a bf16 patch (see Setup; unpatched 0.75 s), SAM 3.1 0.48 s, SAM2.1 0.14 s.  A match half (~70 k frames at 25 fps): **8.2 GPU-hours** for one pass over every frame with SAM 3
   (2.7 h with SAM2.1), 16.4 h if every frame is covered twice for the S1 gate, 2.5 h if SAM 3 only fills the ~30 % of frames the detectors miss; 1.1 h of seed finding per half at one 384-tile probe every 100 frames.

**Recommendation** (details at the end): do **not** replace SAM2.1 in the pseudo-label pipeline with SAM 3 / 3.1 as the tracker.  Make SAM 3 the *background seed finder and seed verifier* (text prompt on 384 px tiles, per-match prompt), keep SAM2.1 crop512 as the track engine,
and use S1 (two-seed agreement) rather than scores as the trust signal.  Use the SAM 3 tracker only for stretches where SAM2 is known to fail (crowd / long occlusion), where it showed its only, small, gain.

## Main table: SAM2.1-small crop512 vs SAM 3 vs SAM 3.1 (best settings: box seed, crop 512, fp16)

Oracle seeds (GT frames), reach <= 400 frames each way.  recall@10 by distance of the scored GT frame from its seed (25 fps frames), coverage = fraction of GT frames with a position, no-ball = fraction of GT `no_ball` frames in
the windows where the tracker is silent (n in brackets), first loss = mean / median frames to the first missed GT frame of a (seed, direction) track, *censored* when never lost (fraction in brackets, so the mean is a lower bound).  95 % CIs are bootstrap over frames
(optimistic: frames of one track are correlated); clip-level cluster bootstrap CIs are in the paired lines and in the JSON (6-7 clips: wide).

| method | scored frames | recall@10 [CI] | 0-25 | 25-100 | 100-250 | 250+ | coverage | no-ball silent | first loss | fps | VRAM |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **held-out + Aston Villa dense** (6 sparse held-out clips + `astonvilla_seg080`, no arsenal_example) | | | | | | | | | | | |
| SAM2.1-small crop512 | 1217 | 0.945 [0.932, 0.957] | 1.000 (165) | 0.968 (347) | 0.973 (415) | 0.845 (290) | 0.988 | 0.88 (26) | 76 / 47 (0.94) | 7.05 | 0.8 GB |
| SAM 3 box crop512 | 1217 | 0.943 [0.930, 0.956] | 1.000 (165) | 0.968 (347) | 0.969 (415) | 0.845 (290) | 0.981 | 0.92 (26) | 76 / 47 (0.94) | 2.38 | 4.9 GB |
| SAM 3.1 mask-seed crop512 (dev set only, see below) | 488 (dev) | 0.928 [0.906, 0.949] | 1.000 (167) | 0.878 (238) | 0.928 (83, 100-120) | - | 0.928 | 1.00 (1) | 67 / 70 (0.75) | 2.07 | 6.2 GB |
| *sparse held-out clips only* (Bournemouth 009 / 015 / 021 / 031 / 035, Man City part093) | | | | | | | | | | | |
| SAM2.1-small | 251 | 1.000 [1.000, 1.000] | 1.000 (93) | 1.000 (122) | 1.000 (36) | - | 1.000 | 0.95 (21) | 67 / 42 (1.00) | 7.08 | |
| SAM 3 | 251 | 1.000 [1.000, 1.000] | 1.000 (93) | 1.000 (122) | 1.000 (36) | - | 1.000 | 0.95 (21) | 67 / 42 (1.00) | 2.38 | |
| *astonvilla_seg080* (one 359-frame run, slow dribble: 2.4 px/frame, max 23, camera pan; 720p) | | | | | | | | | | | |
| SAM2.1-small | 966 | 0.931 [0.914, 0.946] | 1.000 (72) | 0.951 (225) | 0.971 (379) | 0.845 (290) | 0.984 | 0.60 (5) | 174 / 179 (0.33) | 6.98 | |
| SAM 3 | 966 | 0.929 [0.912, 0.944] | 1.000 (72) | 0.951 (225) | 0.966 (379) | 0.845 (290) | 0.976 | 0.80 (5) | 175 / 179 (0.33) | 2.39 | |
| ***arsenal_example*** (reported separately; ball in the crowd) | | | | | | | | | | | |
| SAM2.1-small | 525 | 0.250 [0.215, 0.288] | 0.605 (114) | 0.242 (256) | 0.000 (155) | - | 0.701 | - | 32 / 23 (0.33) | 7.09 | |
| SAM 3 | 525 | 0.301 [0.263, 0.341] | 0.605 (114) | 0.246 (256) | 0.168 (155) | - | 0.579 | - | 30 / 14 (0.33) | 2.38 | |

Paired differences (same seeds and frames), SAM 3 - SAM2.1: all (no example) **-0.002** frame CI [-0.007, +0.004] / clip CI [-0.002, +0.000]; sparse held-out +0.000; Aston Villa -0.002 [-0.010, +0.005]; **arsenal_example +0.051 [+0.032, +0.072]** (one clip).

Read it carefully:
- **Saturation.**  Every held-out GT frame is a frame where a human could see the ball, and a clean seed on the same shot is enough for both trackers: 251/251 on the sparse clips.  The "250+" bin is one clip (Aston Villa, a slowly dribbled ball);
  its 0.845 is mostly 10-11 px misses (3 px over the 10 px threshold: box-label noise at 720p).  The rows cannot rank the models; they show that **tracking from one good seed is not where the remaining error is** on this data.
  The ball-in-play motion in the sparse clips is fast (median 20-38 px between consecutive GT frames in Man City) and still 100 %.
- **Drift / loss behaviour** (what the misses look like).  Dev (SAM 3 box crop512): 82 % of misses are "tracker reports absent" (a camera cut ends the track: the crop follows the old position), 9 % drift 10-50 px, 9 % wrong object; SAM2: 85 % absent, 15 % wrong object.
  The drift case in `track_sam3_mancity_seg010_o2_bwd_drift_to_player.jpg`: the crop follows a player's boot for ~30 frames while the ball is occluded (score logit 1-3 vs 7-9 on the ball), then re-acquires the ball.
  The held-out + Aston Villa misses (69 of 1 217): 33 % absent, 67 % within 10-50 px (mostly the 10-11 px near-misses).
- Scene cuts matter: `cuts.py` finds 1-2 cuts in the 16 clips (part093 at frame 37, bournemouth_1st_seg070 at 47, mancity_seg010 at 593).  A track must be stopped at a cut (not done here; the S1 gate would reject the frames after it).

## Setup (desktop, RTX 2070 8 GB)

- Separate venv `~/sam3-exp/venv` (python 3.12, torch 2.8.0+cu128, numpy 1.26.4); upstream `facebookresearch/sam3` @ `eef6839` (2026-11-13) installed editable from `~/sam3-exp/sam3_src`.  Not ultralytics 8.4.213: its `SAM3VideoPredictor`
  is the frame-by-frame predictor pipeline, its 3.1 support is image / point only, whereas the upstream tracker has the SAM2-style `init_state / add_new_points_or_box / propagate_in_video` API the round-3 crop-following driver needs.
  Nothing was installed into `~/sam2-exp/venv`, the jobs venv or the footy venv.
- Weights: `sam3.pt` (sha256 `9999e234...`, identical to the mini copy) and `sam3.1_multiplex.pt` (complete, 3 502 755 717 bytes) pulled from S3 with presigned urls.
- **Licence**: SAM License (Meta, 2025-11-19): royalty-free, worldwide, non-exclusive licence to use / modify / distribute the SAM Materials; redistribution of the model or derivatives only under the same agreement (include a copy);
  publications of research done with it must acknowledge the use; no use for ITAR / military / weapons / espionage or trade-control-prohibited ends; the licence terminates if we sue Meta over the materials; "AS IS".  Fine for an internal
  labelling pipeline; keep the licence text with any redistributed weights.  (`pyproject` says "MIT" in a classifier, the `LICENSE` file is the SAM License.)  Dependencies: timm, ftfy, regex, iopath, huggingface_hub (unused with a local checkpoint), numpy < 2; no flash-attn / triton on sm_75.
- **RTX 2070 (sm_75) patches**, applied as runtime monkeypatches in `sam3_track.patch_turing` (upstream untouched):
  1. `sam3.perflib.fused.addmm_act` (ViT MLP fc1 + GELU) hard-casts to **bfloat16**; sm_75 has no bf16 tensor cores, so cuBLAS ran an emulated `magma_sgemmEx<bf16>` kernel = **59 % of the backbone time**.  Plain fp16 autocast: backbone 0.63 -> 0.29 s per 1008 px frame (**2.1x**), tracker 0.75 -> 0.42 s per frame.
  2. The upstream constructors enter a global bf16 autocast (`bf16_context.__enter__()`); it is exited right after building and runs use `torch.autocast(fp16)`.  No fp16 overflow / NaN observed.
  3. SAM 3.1: `decoder.functional_attention` forces the FlashAttention SDPA backend ("No available kernel" on Turing) -> efficient / math backends.  The standalone 3.1 tracker also needs `tracker.backbone = detector.backbone` and `forward_image(need_sam3_out=False)`;
     the multiplex demo tracker has no box prompt: "box" = the ellipse inscribed in the box as a mask prompt, "point" = the centre point.
- **Precision / memory** (2 dev clips, 1 seed each, 60 frames, 186 frames scored): fp16 autocast 0.995 recall, 2.38 fps, **4.9 GB** peak (fp32 weights resident + fp16 copies cached by autocast); **fp32 identical labels (paired +0.000) at 0.74 fps (3.2x slower), 4.3 GB**;
  memory selection off (`apply_temporal_disambiguation=False`) 1.000 (+0.005 [+0.000, +0.019], n.s.); `offload_state_to_cpu` **0.715 (-0.280)**, VRAM unchanged (4.8 GB: the weights dominate) -> do not use.  Weights are 3.4 GB in fp32; a half-precision copy of the backbone
  would free ~2 GB if more headroom is needed (not needed: 8 GB fit, long clips are fine because frames are injected one at a time and per-frame state is ~1 MB; whole-match tracks prune state older than 300 frames, `prune_age`).

## Data, seeds, metrics

**Clips and GT** (`spec_v2.json`, built by `spec.py` from `/mnt/storage/footy_data/ball_gt_marks`; human `labeller` / `human_checked` ball marks only; error in 1080p-equivalent px, hit = within 10 px):

| set | clips | how used |
|---|---|---|
| final / held-out | Bournemouth seg009, 015, 021, 031, 035, 060 (50 fps clips handled at 25 fps = every 2nd frame, as round 3); Man City part093, seg101, seg134, seg135 | final numbers, never used to choose a setting (the single-GT clips seg060 / 101 / 134 / 135 only enter the realistic-seed protocol) |
| final / dense | `astonvilla_seg080` (568 labelled frames, 4 runs, longest 359 frames, 720p) | the only clip with GT > 250 frames from a seed |
| example | `arsenal_example` | reported separately |
| dev | `mancity_seg010`, `mancity_seg050`, `bournemouth_1st_seg070` (dense, yellow ball); `mancity_seg031` for seed-finder only | every prompt / crop / trick search (2 oracle seeds per clip, 120 frames each way = 488 scored samples) |

- Odd original frames of the 50 fps clips are scored against the mean of the two neighbouring processed frames (GT lies on both parities: seg009 has 2 even + 9 odd GT frames).
- Oracle seeds: GT frames at the 15 / 50 / 85 % quantile of the clip's GT frames (dense runs: run start + run middle); the first GT frame is avoided (`mancity_seg031` frame 0 is **not** on the ball: `gt_mancity_seg031.jpg`).
  Realistic seeds: RT-DETR-L v11 @1280 ball detections with conf >= 0.8, best first, >= 40 frames apart, 3 per clip (Bournemouth from the round-3 `labels_bm_audit`, Man City from `machine_labels`, part093 / arsenal_example from a fresh `r3_detect.py` run).
  RT-DETR has no conf >= 0.8 ball in seg021 / seg031 (no seed there).  A random sample of 64 such candidates over the held-out clips was inspected by eye: **64 / 64 real balls** (`rtdetr_conf80_seed_candidates_64.jpg`; so false seeds are rare, not absent: see the production demo).
- **Norwich**: `ball_gt_marks_norwich` holds only the blind-check verdicts (`ball_checks`), no window labels, so nothing could be scored on Norwich.
- **A sample** = one (seed, GT frame) pair: the forward (g > seed) or backward track of the seed at g; recall@10 counts absent as a miss; precision(emitted) = hits / emitted.  Intervals: 95 % percentile bootstrap over frames and (cluster) over clips; paired differences on identical samples.
- GT sanity (contact sheets): dev and held-out GT are on the ball except `mancity_seg031` frame 0.  **`arsenal_example`'s 20 held-out frames (59, 160-198) are an out-of-play ball inside the crowd** (a white blob among spectators; the labels follow a smooth trajectory, so
  they are not wrong, but the target is practically invisible: `gt_arsenal_example_heldout.jpg`).  RT-DETR (0.05), SAM 3 text detection on full frame / tiles (0.00-0.05 top-1, any prompt) and both trackers fail there: **SAM 3 does not find those balls**.

## Dev sweep: prompts, crops, SAM 3 vs SAM 3.1 (3 dev clips, 2 oracle seeds each, 120 frames each way, 488 samples, fp16)

recall@10 by distance (only 0-25 / 25-100 / 100-120 exist at this reach), coverage, precision of what is emitted; paired differences against SAM 3 box crop512 on identical (clip, seed, frame) samples (95 % CI over frames; clip-level CIs of the 3 dev clips are huge and not shown).

| config | recall@10 [95 % CI] | 0-25 | 25-100 | 100-120 | coverage | prec. emitted | paired vs SAM 3 box crop512 | s/frame | VRAM |
|---|---|---|---|---|---|---|---|---|---|
| SAM2.1-small crop512 (round-3 baseline) | 0.918 [0.891, 0.943] | 1.000 | 0.950 | 0.663 (n=83) | 0.930 | 0.987 | -0.014 [-0.029, +0.000] | 0.14 | 0.8 GB |
| **SAM 3 box, crop512** | **0.932** [0.910, 0.955] | 1.000 | 0.975 | 0.675 | 0.945 | 0.987 | - | 0.42 | 4.9 GB |
| SAM 3 box + centre point, crop512 | 0.934 [0.912, 0.955] | 1.000 | 0.979 | 0.675 | 0.945 | 0.989 | +0.002 [+0.000, +0.006] | 0.42 | 4.9 GB |
| SAM 3 point only, crop512 | 0.912 [0.885, 0.936] | 1.000 | 0.945 | 0.639 | 0.932 | 0.978 | -0.020 [-0.033, -0.008] | 0.42 | 4.9 GB |
| SAM 3 box, crop768 (1.3x up to 1008) | 0.928 [0.906, 0.951] | 1.000 | 0.966 | 0.675 | 0.945 | 0.983 | -0.004 [-0.010, +0.000] | 0.42 | 4.9 GB |
| SAM 3 box, crop384 (2.6x up) | 0.912 [0.885, 0.936] | 1.000 | 0.945 | 0.639 | 0.930 | 0.980 | -0.020 [-0.033, -0.008] | 0.42 | 4.9 GB |
| SAM 3 box, crop256 (3.9x up) | 0.689 [0.648, 0.732] | 1.000 | 0.647 | 0.181 | 0.695 | 0.991 | -0.244 [-0.283, -0.203] | 0.42 | 4.9 GB |
| SAM 3 box, whole 1080p frame squashed to 1008 | 0.893 [0.867, 0.920] | 1.000 | 0.899 | 0.663 | 0.932 | 0.958 | -0.039 [-0.059, -0.020] | 0.42 | 5.0 GB |
| SAM 3.1 mask seed (ellipse in the box), crop512 | 0.928 [0.906, 0.949] | 1.000 | 0.878 | 0.928 | 0.928 | 1.000 | -0.004 [-0.035, +0.027] | 0.49 | 6.2 GB |
| SAM 3.1 point seed, crop512 | 0.928 [0.906, 0.949] | 1.000 | 0.878 | 0.928 | 0.928 | 1.000 | -0.004 [-0.035, +0.027] | 0.49 | 6.2 GB |
| SAM 3 box crop512, `frame_step 2` (track every 2nd frame, interpolate) | 0.881 [0.852, 0.908] | 0.982 | 0.895 | 0.639 | 0.904 | 0.975 | -0.051 [-0.072, -0.033] | 0.21 (eff.) | 4.9 GB |

- **The crop is the biggest lever**, with a plateau at 512-768 px: a 256 px window loses the ball whenever it moves fast (-0.24), 384 -0.02, no crop -0.04 (SAM 3 on the squashed 1080p frame still works: the ball is 5-6 px there).  Compute is identical for every crop (the model always sees
  1008 px), so "upscale the crop to the native input" is free and automatic; **upscaling more (smaller crops) does not help, a larger field of view does**.
- Box seed (with or without the centre point) > point only.  SAM 3.1 behaves like SAM 3 (same recall within noise, higher precision, lower coverage, +20 % time, +1.3 GB): its multiplex advantage is for many simultaneous objects, not for one ball.
- Text prompts as the *tracker* (no seed memory: per frame, text detection on the crop around the last position, nearest candidate, `text:ball` 0.725 [0.679, 0.775], `text:yellow ball` 0.630 (white-ball Man City clips included), reach 80, 346 samples) lose 0.26-0.35 against the box-seeded tracker (0.983):
  the detector alone drops the ball (coverage 0.73) although what it emits is 99.6 % correct.  Text belongs in seed finding / recovery, not in tracking.
- `frame_step 2` doubles throughput (4.75 effective fps) but costs 0.051 recall: linear interpolation over the skipped frame fails on kicks.

## Realistic seeds (RT-DETR conf >= 0.8), held-out clips, reach 250

| method | scored | recall@10 | 0-25 | 25-100 | 100-250 | coverage | prec. emitted | no-ball silent | first loss |
|---|---|---|---|---|---|---|---|---|---|
| SAM2.1-small crop512 | 228 (8 clips, 21 seeds) | 0.996 [0.987, 1.000] | 1.000 (57) | 1.000 (111) | 0.983 (60) | 1.000 | 0.996 | 1.00 (21) | 96 / 74 (0.97) |
| SAM 3 box crop512 | 228 | 0.996 [0.987, 1.000] | 1.000 (57) | 1.000 (111) | 0.983 (60) | 1.000 | 0.996 | 1.00 (21) | 96 / 74 (0.97) |
| arsenal_example, 1 seed: SAM2 / SAM 3 | 175 | 0.263 / 0.274 | 1.000 (26) | 0.328 / 0.361 | 0.000 / 0.000 | 0.714 / 0.537 | 0.368 / 0.511 | | |

The single miss (seg060: the only GT frame is 178 frames from the seed) is the same for both.  Seed verification with SAM 3's own text detector (`seed_check`: best IoU of any "ball" detection on a 512 px crop with the seed box): all 24 seeds had IoU 0.63-0.92 with a detection
(text score 0.08-0.85), i.e. no seed in this set was a false positive, so the check could not be shown to separate anything.

## Gating

- **Object-score logit** (own presence head).  Raising the gate trades coverage for precision; SAM 3's score is the more informative one.  Held-out + Aston Villa, oracle seeds: gate 0 / 4 / 6 / 8: SAM 3 coverage 0.98 / 0.94 / 0.87 / **0.59** with precision 0.961 / 0.964 / 0.974 / **0.996**;
  SAM2 0.99 / 0.91 / 0.84 / 0.37 with 0.957 / 0.969 / 0.972 / 1.000.  On `arsenal_example` gate 8: SAM 3 coverage 0.17, precision 0.966 (recall 0.164) vs SAM2 coverage 0.02: SAM 3's scores separate the few correct frames from the crowd; SAM2's do not.
  (The 0.96 baseline precision is the 10-11 px near-misses of one clip.)
- **S1** (forward track of seed a, backward track of the next seed b, agree within 5 px): with oracle seeds all 217 GT frames between seeds pass and are correct for both models (nothing to reject).  In the production demo with RT-DETR seeds on Bournemouth seg035
  (below) S1-gated frames were **14 / 14** correct while single-track score-gated frames were 24 frames at only **67 %**: the wrong ones come from a false seed.
- **SAM 3 x SAM2 agreement** (within 5 px, oracle seeds, held-out + Aston Villa): coverage 0.974, recall 0.937, no precision gain (both make the same 10-11 px drift errors); it is a useful *independent-model* check against false seeds, not against labelling noise.
- **Lesson (production demo)**: a wrong seed (RT-DETR conf 0.818 "ball" = a **yellow boot** at seg035 frame 58) gives a confident, consistent track (logit 8-10, 90+ frames): `demo_labels_bournemouth_seg035.jpg`.  No score gate can catch it; agreement between *independent seeds* (S1) or a seed check can.

## Recovering lost tracks

Implemented in `sam3_track.py` (config `recover`): after `loss_k = 3` consecutive lost frames (mask empty or score logit < 0), search around the last confident position (radius 150 px, 512 px crop) and re-seed the tracker with the nearest candidate as a new
conditioning box, then continue propagating.  Candidates from (a) SAM 3 text detection `text:ball` on the crop (score >= 0.3) or (b) RT-DETR candidates (conf >= 0.5, `cands`, the "re-seed from detector candidates" variant).  Ablation on the final oracle set, SAM 3 box crop512, reach 400
(astonvilla_seg080, arsenal_example, bournemouth seg035; (b) on the last two): **the recovery never fired (0 re-seeds in every track), results are identical to the baseline** (0.929 / 0.301 / 1.000).  The loss events in these clips are a camera cut or a ball that
is not visible (crowd), where there is no candidate within the search radius; the held-out set has no loss that a nearby detection could repair.  So the benefit of recovery is **unmeasured**, and the re-seed code path was never exercised on real data (treat it as untested);
a test needs footage with occlusions behind players / long kicks that the tracker loses but a detector still sees (the dev drift case `mancity_seg010` o2 backward is such a case, but only 2 of the dev tracks lose the ball).  Text detection as a *seed finder* is the part that works (below).
Not done: RT-DETR / text recovery for SAM 3.1 (its detector API differs), re-seeding on a second SAM2 opinion.

## Text-prompt seed finder (hit rate vs RT-DETR on held-out frames)

SAM 3 image detector, text prompt, boxes in full-frame px, scored on every GT ball frame (original frame index, both parities): "top-1" = the highest-scoring detection is within 10 px, "any" = some detection (score >= 0.02, <= 15 per frame after NMS) is within 10 px.
Tilings: `full` = whole frame squashed to 1008 (ball ~5 px), `t540` = 3x2 tiles of 540 px, `t384` = 5x3 tiles of 384 px (2.6x up).  RT-DETR-L on the same frames (stride-1 labels, conf >= 0.1) for reference.
No prompt / tiling was selected on these frames beyond reading the table; a dev-set check follows.  top-1 hit@10 / any hit@10:

| detector (tiling / prompt) | Bournemouth (73, yellow ball) | Man City (21) | Aston Villa, 39 evenly spaced | all held-out (no arsenal_example) incl. Aston Villa subset |
|---|---|---|---|---|
| RT-DETR-L v11 | 0.51 / 0.63 | 0.90 / 1.00 | - | 0.60 / 0.71 (94 frames, without Aston Villa) |
| full / ball | 0.74 / 0.97 | 0.76 / 1.00 | 0.79 / 0.90 | 0.76 / 0.95 |
| full / football | 0.70 / 0.96 | 0.90 / 1.00 | 0.59 / 0.90 | 0.70 / 0.95 |
| full / soccer ball | 0.67 / 0.97 | 0.76 / 1.00 | 0.51 / 0.92 | 0.64 / 0.96 |
| full / white ball | 0.03 / 0.56 | 0.76 / 1.00 | 0.00 / 0.33 | 0.14 / 0.56 |
| full / small white ball | 0.00 / 0.19 | 0.71 / 1.00 | 0.00 / 0.08 | 0.11 / 0.29 |
| full / yellow ball | 0.78 / 0.99 | 0.00 / 0.00 | 0.82 / 0.92 | 0.67 / 0.81 |
| t540 / ball | 0.67 / 0.78 | 0.57 / 0.71 | 0.85 / 0.95 | 0.71 / 0.82 |
| t540 / yellow ball | 0.75 / 0.79 | 0.14 / 0.14 | 0.87 / 0.92 | 0.69 / 0.73 |
| **t384 / ball** | **0.85 / 0.96** | 0.86 / 0.95 | 0.85 / 0.95 | **0.85 / 0.95** |
| t384 / football | 0.63 / 0.93 | 0.90 / 1.00 | 0.49 / 0.87 | 0.63 / 0.92 |
| t384 / soccer ball | 0.77 / 0.96 | 0.95 / 1.00 | 0.59 / 0.95 | 0.74 / 0.96 |
| t384 / white ball | 0.04 / 0.19 | 0.95 / 1.00 | 0.00 / 0.62 | 0.17 / 0.44 |
| t384 / yellow ball | 0.86 / 0.97 | 0.00 / 0.05 | 0.92 / 0.95 | 0.74 / 0.82 |

(full table with all tilings, thresholds and the other subsets: JSON.)  At a detection-score threshold of 0.5, "t384 / ball" fires on 71 % of the ball frames and its top detection is right 95 % of the time (RT-DETR at 0.5: 73 % of frames, 75 % right; at 0.8: 31 % of frames, 100 % right;
SAM 3's text scores are lower, essentially nothing reaches 0.8).  On the 7-8 GT `no_ball` frames the false-fire rate at 0.5 is 0.12 (SAM 3) vs 0.14 (RT-DETR): too few to separate.  `arsenal_example` held-out frames: top-1 0.05 for RT-DETR and "t384 / ball", 0.00 elsewhere.
Whole-frame detection (0.78 s per frame for 6 prompts) is a cheap first pass (top-1 0.76); tiles cost 4.5 s (540 px) / 11.2 s (384 px) per frame for 6 prompts, ~5.6 s for one prompt at 384 px.  **Yes, SAM 3 can find its own seeds, better than RT-DETR on the held-out Bournemouth frames, at ~5 s per probed frame.**

**Dev check (80 GT frames: 20 each from `mancity_seg010`, `mancity_seg031`, `mancity_seg050`, `bournemouth_1st_seg070`; full / t384 tilings), and an honest limit of the result above.**  On the dev frames SAM 3 text detection is *not* better than RT-DETR:
t384 / ball top-1 0.64 (seg010 0.85, seg050 0.65, seg031 0.60, seg070 0.45), t384 / soccer ball 0.64, full / ball 0.53, vs RT-DETR 0.85 (seg070 1.00, seg010 0.85, seg050 0.90, seg031 0.65).  Two reasons: (1) the dev clips are RT-DETR training material
(`bournemouth_1st_seg070` is memorised: 1.00) while the held-out clips are exactly the ones RT-DETR has not seen, which is where it is weak (0.60); (2) dev GT frames come from dense runs and include many fast, blurred balls (SAM 3 text on seg070: 0.45 vs 0.85 on the held-out
Bournemouth frames), the held-out frames were picked differently.  Prompt ranking agrees between dev and final: "ball" ~ "soccer ball" best on white balls, "yellow ball" only on the yellow ball ("yellow ball" t384: 0.45 on seg070, 0.00-0.05 on the Man City clips), "white ball" /
"small white ball" poor.  So: **for a match RT-DETR has not been trained on, SAM 3 text finds more balls than RT-DETR; on footage RT-DETR was trained on it does not, and in general it is a ~10-50x slower, second, independent detector** (the property that makes it useful as a seed verifier).

**Text + exemplar box** (the API supports it: `add_geometric_prompt` on the same tile; the exemplar here is a rough hint = the GT box inflated 3x, only the tile containing it is searched; dev frames, t384): top-1 **0.72** whatever the prompt ("ball" 0.72, "soccer ball" 0.72,
"yellow ball" on white balls 0.70 vs 0.12 text-only: the exemplar dominates the prompt), emitted on 78 % of frames, vs 0.64 for text-only "ball" on the same frames.  Modest: a coarse click / box is only partly refined into a correct ball box; it also helps simply because the search is restricted to one tile.


## Throughput and cost (RTX 2070, fp16, one process)

| component | s / frame | frames / s | peak VRAM |
|---|---|---|---|
| SAM2.1-small crop512 track (sam2-exp driver) | 0.14 | 7.05 | 0.8-1.0 GB |
| SAM 3 box crop512 track | 0.42 | 2.38 | 4.9 GB |
| SAM 3.1 track | 0.48 | 2.07 | 6.2 GB |
| SAM 3 unpatched (bf16 emulation) | 0.75 | 1.3 | 4.9 GB |
| SAM 3 fp32 | 1.35 | 0.74 | 4.3 GB |
| SAM 3 `frame_step 2` | 0.21 effective | 4.75 | 4.9 GB |
| SAM 3 text probe, 384 px tiles, 1 prompt | ~5.6 per probed frame | | 5.6 GB |

GPU-hours per match half (~70 k frames at 25 fps; Bournemouth 50 fps segments are processed every 2nd frame):
- SAM 3, every frame, single coverage (each seed tracks to the midpoint to its neighbour): **8.2 h**; with S1 double coverage (forward and backward over every gap): **16.4 h**.  SAM2.1: 2.7 h / 5.4 h.
- SAM 3 only on the frames the detector pipeline leaves open (round 3: ~30-50 % of balls had no confident detector candidate): **~2.5-4 h**; with `frame_step 2` half of that at -0.05 recall.
- Seed finding, one 384-tile probe every 100 frames: 700 probes x 5.6 s = **1.1 h** (every 50 frames: 2.2 h).
- Measured end to end: the production demo (3 seeds, ~300 tracked frames incl. overlap) ran at 2.2-2.4 fps, i.e. 0.42 s per tracked frame including decoding.

## Production shape (built and demoed)

### Prefect deployments and how to queue a whole match

Flow module `~/code/jobs/footy_sam3.py`, deployments `~/code/jobs/sam3.prefect.yaml` (pool `gpu`, concurrency 1, desktop RTX 2070).  Experiment code is *not* in the jobs repo: every run rsyncs
`/home/george/footy-track-sam3/experiments/sam3_offline/` (branch `exp/sam3-offline-labeller`) from the mini, so code edits apply to the next queued run without redeploying.  Redeploy only when a flow
signature changes: `cd ~/code/jobs && ~/.claude/skills/prefect-jobs/scripts/sync-desktop.sh && prefect deploy --prefect-file sam3.prefect.yaml --all`.

| deployment | purpose | main parameters |
|---|---|---|
| `sam3-track-eval/sam3-track-eval` | seed-propagated tracks for an evaluation spec; engine `sam3` (model sam3 / sam3.1) or `sam2` (the round-3 SAM2.1-small crop512 baseline through the same seeds) | `spec`, `clips` (dev / final / example / names), `protocol` (oracle / rtdetr), `config` (JSON: model, prompt box / point / boxpoint / text:..., crop, recover, ...), `max_reach`, `n_seeds`, `exp` |
| `sam3-seedfind/sam3-seedfind` | text-prompt (whole frame / 540 px tiles / 384 px tiles) ball detection on GT frames, 6 prompts, optional exemplar | `schemes`, `prompts`, `exemplar`, `only_frames`, `exp` |
| `sam3-rtdetr/sam3-rtdetr` | RT-DETR-L v11 labels (round-3 `r3_detect.py`) for spec clips that have none | `clips`, `exp` |
| `sam3-script/sam3-script` | run one script of the code dir with the sam3 venv (profiling, smoke tests) | `script`, `args` |
| **`sam3-label-match/sam3-label-match`** | **production: queue a whole match (half) for background labelling** | below |

All flows: resumable (per clip file / per track), wait for free VRAM (`min_free_mib`, default 5500) before starting, push outputs to the mini over rsync with an S3 presigned-PUT fallback (`footy_s3hop.py`
tables), `retries=1-2`.  Zombie / crash automations of the gpu pool apply as for every gpu flow.

### Queue a whole match

1. Seeds file on the mini (JSONL, one row per seed; `box` or `xy`, original frame index): from RT-DETR machine labels
   `python experiments/sam3_offline/make_seeds.py --labels <machine_labels dir> --video-dir <clips dir> --thr 0.8 --spacing 100 > /mnt/storage/exp/sam3/<match>/seeds.jsonl`
   (best conf >= 0.8 ball per 100-frame window; Bournemouth round-3 labels: `--index-stride 2`).  Human clicks / overview seeds are appended in the same format (`{"clip": .., "frame": .., "xy": [x, y], "source": "human"}`).
2. Segment clips named `<clip>.mp4` in one directory on the mini (or a `clip_files` map `{clip: file name}`).
3. `$PF run sam3-label-match/sam3-label-match match=<name> seeds=/mnt/storage/exp/sam3/<name>/seeds.jsonl clips_dir=<mini dir> reach=250 overlap=15 'config={"model":"sam3","prompt":"box","crop":512}'`
   - leave it: per clip the video is staged (S3 presigned GET `sam3/<match>/<clip>.mp4` when prepared, else rsync), `sam3_label.py` runs forward + backward tracks per seed (never past the neighbouring seed + `overlap`),
     writes `<clip>.labels.jsonl` + `<clip>.raw.json.gz`, pushes them to `/mnt/storage/exp/sam3/<match>/labels/`, deletes the staged video.  Cancel / retry / crash-resume keeps finished clips and finished tracks.
4. Output rows (JSONL, one per labelled frame): `clip, p, frame` (original index), `x, y, box, score` (SAM object-score logit), `seed, dir, dist` (frames from the seed), `gate`: `seed` (the seed itself) |
   `s1` (forward track of one seed and backward track of the next agree within 5 px: position = mean) | `score` (single track, logit > `score_gate`) | `weak` (the rest: do not use as a label).


**Demo** (`sam3-label-match match=demo`, Bournemouth seg035, 3 RT-DETR seeds at original frames 58, 116, 270, reach 150): 39 GT frames inside the labelled span: **seed 1/1, `s1` 14/14 correct, `score`-gated 24 frames 66.7 % correct** (the errors all come from seed 58, a yellow boot).
`eval_labels.py` reproduces it; `demo_labels_bournemouth_seg035.jpg` shows the labels (magenta = seed, green = s1, yellow = score gate).

## Visual checks (`docs/experiments/sam3_offline_labeller/`)

- `track_sam3_mancity_seg050_o0_fwd_long_good.jpg`: 120-frame forward track through blur, a long kick and into the crowd; green = GT, red cross = SAM 3 (crop centred on its own prediction).
- `track_sam2_astonvilla_o0_fwd_359_frames.jpg`: SAM2 over the 359-frame Aston Villa run (every 16th frame).
- `track_sam3_mancity_seg010_o2_bwd_drift_to_player.jpg`: drift onto a player's boot while the ball is occluded, re-acquisition ~30 frames later.
- `track_sam3_arsenal_example_o1_fwd_crowd_failure.jpg`: lost in the crowd, "absent" with logit -4..-6 (scores say so).
- `demo_labels_bournemouth_seg035.jpg`: production output including the yellow-boot false seed.
- `gt_arsenal_example_heldout.jpg`, `gt_mancity_seg031.jpg`, `rtdetr_conf80_seed_candidates_64.jpg`: GT / seed sanity sheets.

## Caveats

- The held-out GT is saturated for tracking from a good seed: differences between SAM2 / SAM 3 / 3.1 are below what 251-1 217 frames of 7-8 clips can resolve (clip-level CIs are wide; frame-level CIs are optimistic).  A harder, denser GT with fast kicks / occlusions would be needed to rank trackers.
- The 250+ bin is one clip; `arsenal_example` is one clip; no-ball precision has n = 5-26; Norwich could not be scored (no window labels).
- Odd-frame GT of 50 fps clips is scored against the mean of two neighbours (adds up to a few px of interpolation error); 10 px hit threshold equals about the ball diameter.
- Seed-finder prompts were not tuned on a separate set before the held-out numbers were read (a dev check is reported); the "yellow ball" result is match-specific by construction.
- SAM 3.1 was tested with mask / point seeds only (no box prompt in the multiplex demo tracker) and only on the dev set; the text-recovery trick is implemented for SAM 3 only.
- fp16 autocast with fp32 resident weights; no `torch.compile`, no lower input resolution (the tracker is hard-wired to 1008 px / 72x72 features; a 504 px variant would need re-deriving the RoPE / memory sizes and was not tried).

## Recommendation and next step

1. **Keep SAM2.1-small crop512 as the track engine** of the pseudo-label pipeline: same accuracy as SAM 3 / 3.1 wherever the GT can tell (saturated), 3x faster, 6x less memory, 2.7 GPU-hours per half.  SAM 3.1's multiplex gains do not apply to one ball.
2. **Use SAM 3 as the slow background seed finder and verifier**: (a) per match pick the text prompt (white ball: "ball" / "soccer ball"; yellow ball: "ball" / "yellow ball"; check on a few dev frames), (b) probe 384 px tiles upscaled to 1008 every ~50-100 frames (1-2 h per half) and add the top detection
   (score >= 0.5: ~95 % right) as a seed candidate next to RT-DETR conf >= 0.8 seeds, (c) accept a track only where **two independent seeds agree (S1, 5 px)** or a second model (SAM 3 vs SAM2.1) agrees; do not trust a single track's score (false seed = confident wrong track).
3. Use the SAM 3 tracker only on stretches where SAM2 loses the ball (crowd / long occlusion): it was the only place it beat SAM2 (+0.05 on `arsenal_example`, +0.17 at 100-250 frames, one clip) and its logit is the better gate (score > 8: precision 0.97-1.0).
4. **Production**: `sam3-label-match` (built, demoed): ~0.42 s per tracked frame; give it seeds from (2); budget ~8 GPU-hours per half for one SAM 3 pass over every frame, ~2.5-4 h for gap filling, overnight is fine.

Next steps, in order: (i) a harder, denser GT (fast kicks, occlusions, crowd) because the current held-out set cannot rank trackers; (ii) stop tracks at scene cuts (`cuts.py`) and add the SAM 3 text probe as a seed *verifier* in `sam3_label.py`;
(iii) run the seed finder + SAM2 tracks on an unlabelled half (Norwich) and measure the precision of the S1-gated labels with the blind Ball Check (the study could not score Norwich: no window labels); (iv) feed the SAM 3 text detections as extra candidates to the round-3 B1 solver (union with RT-DETR gave +0.03-0.04 recall before);
(v) only if throughput matters, try a 504 px SAM 3 input (4x fewer tokens; needs the RoPE / memory feature sizes re-derived) or half-precision weights.
