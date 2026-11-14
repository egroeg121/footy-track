# WASB-SBDT soccer model, zero-shot on our broadcast ball

Date: 2026-11-14. Branch `exp/wasb-zeroshot`. Question: does the pretrained WASB soccer
heatmap model (3-frame HRNet, 1.48 M params) detect our ~16 px (1080p) broadcast ball without
any training, either on the full frame at its native 288x512 input or on tiles? The answer
decides "fine-tune WASB" versus "build our own heatmap net".

**Result: no.** On 1,495 human-labelled ball frames the model puts its top candidate within
10 px of the ball on 5.2% of frames (full frame) and on 3.5% (best tiling). RT-DETR, on the same
Man City frames that were not in its training set, gets 76%. Details below.

## What ran, where

| step | where | notes |
|---|---|---|
| clone + weights | mini | `github.com/nttcom/WASB-SBDT` at `/mnt/storage/exp/wasb/repo`; soccer checkpoint via `gdown` (MODEL_ZOO.md, Google Drive id `1pg0MpMtKZ6ziYEr4oyfKYPOO3hjLw94l`, 6.1 MB) at `/mnt/storage/exp/wasb/weights/` |
| eval set, v11 overlap check, scoring | mini (CPU) | `experiments/wasb_zeroshot/{build_evalset,match_v11,score,viz_failures}.py` |
| inference | desktop RTX 2070 (8 GB) through Prefect pool `gpu` | deployment `wasb-zeroshot/wasb-zeroshot` (`~/code/jobs/wasb_zeroshot.py`, copy in `experiments/wasb_zeroshot/prefect/`). Runs: `wasb-v1` (51810720, OOM on 24-tile batch), `wasb-v2` (675b087f, all 33 clips, 2,397 s), `wasb-v3-720p` (666abe7c, rerun of the 3 Aston Villa clips) |

Inputs were staged on the mini at `/mnt/storage/exp/wasb/inputs` and rsynced to the desktop by
the flow (same pattern as `sam2_ball.py`). The flow reuses the desktop's shared footy venv
(torch 2.8, opencv 4.11). Prefect was available throughout; the remote-job fallback was not used.
`wasb-v2`'s predictions for the 3 Aston Villa clips were discarded (see Caveats); the
scored set is `wasb-v2` for 30 clips plus `wasb-v3-720p` for the 3 Aston clips.

Knowledge base pages used: `footy-ball-detection-findings` (ball size, v11 training clips,
RT-DETR figures), `prefect-job-system`, `footy-data-s3`, plus the research note
`research_notes/Offline football ball tracking models/sports_ball_trackers.md` (WASB section).

## Setup

* **Model**: `src/models/hrnet.py` from the WASB repo, config of `configs/model/wasb.yaml`
  (frames_in=3, frames_out=3, 288x512, stem strides 1, ~1.48 M params), checkpoint
  `wasb_soccer_best.pth.tar`. Input: ImageNet-normalised RGB, 3 consecutive frames on 9 channels,
  resize by `cv2.warpAffine` bilinear (as in WASB's loader). Output: 3 heatmaps; we use the
  centre frame's (output index 1; near-hit offsets average -0.7/-0.9 px, so the alignment is right).
  Window = (t-1, t, t+1), clamped at clip ends. No tracker (the WASB paper reports it does not help on soccer).
* **Candidates**: local maxima (9x9) of the sigmoid heatmap with value >= 0.1, centroid refined
  over 5x5, mapped back to 1080p coordinates; score = peak value. Tiles are merged with NMS (8 px).
* **Modes** (ball is 12/14/16/20/26 px wide at 1080p, p5/p25/p50/p75/p95; WASB trains at 3.75x down):

| mode | input region -> 512x288 | ball at input | tiles/window |
|---|---|---|---|
| `full_bilinear` (a) | full 1920x1080, 3.75x down, warpAffine | ~4.4 px | 1 |
| `full_area` | same, INTER_AREA (antialiased) | ~4.4 px | 1 |
| `tile_2x` | 1024x576 regions, 2x down, 3x2 grid | ~8 px | 6 |
| `tile_1p5x` | 768x432 regions, 1.5x down, 4x3 grid | ~11 px | 12 |
| `tile_native` (b) | 512x288 regions at 1:1, 6x4 grid | ~16 px | 24 |

* **Eval frames** (`build_evalset.py`): human marks only (`in_play_ball` with provenance
  `labeller` or `human_checked`; `yolo`/`rtdetr` marks dropped), merged across the alias stems of
  a clip by `canonical_clip_id`, de-duplicated within 3 px. 33 clips, **1,495 ball frames and 37
  explicit `no_ball` frames** (`not_broadcast` and `out_of_play_ball` frames ignored).
  Dense clips: `arsenal_example` 176, `astonvilla_seg080` 568, `bournemouth_1st_seg070` 220;
  the 16 Man City clips (`arsenal_mancity_seg*`, `part*`) have 424 ball / 26 no_ball frames.
  WASB never saw our data, so all frames are fair for it.
* **Metrics** (`score.py`, 1080p px, d = 10 unless stated, also 5 and 20): `top1` = share of
  ball frames whose highest-scoring candidate is within d of a GT centre. `oracle` = any candidate
  within d (what a tracker could at best pick). Threshold sweep s = 0.10..0.95: recall = ball
  frames whose top-1 emitted candidate is a hit; `prec_f` = hit frames / frames (ball or no_ball)
  where anything was emitted; `nb_fp` = no_ball frames with any emission; `F1` of the two.
  "Best" = the s with max F1.

## Results: all 1,495 ball + 37 no_ball frames, d = 10 px

| mode | top1 | oracle | best-F1 s | recall | prec_f | F1 | no_ball FP | dets/frame |
|---|---|---|---|---|---|---|---|---|
| a. `full_bilinear` | **77/1495 = 5.2%** | 13.6% | 0.10 | 5.2% | 5.9% | 0.055 | 89% | 5.1 |
| `full_area` | 67 = 4.5% | 11.7% | 0.20 | 4.1% | 6.1% | 0.049 | 78% | 3.1 |
| `tile_2x` | 52 = 3.5% | 21.3% | 0.20 | 3.5% | 3.6% | 0.035 | 81% | 10.5 |
| `tile_1p5x` | 10 = 0.7% | 11.0% | 0.40 | 0.7% | 0.8% | 0.007 | 81% | 9.4 |
| b. `tile_native` | 4 = 0.3% | 6.8% | 0.20 | 0.3% | 0.3% | 0.003 | 100% | 19.8 |

At 5 px / 20 px the top-1 recall of `full_bilinear` is 3.3% / 5.2%, so the misses are not a
localisation tolerance problem. At the WASB default threshold 0.5, `full_bilinear` has recall
2.9%, frame precision 9.6%, per-detection precision 3.6%, and fires on 49% of the no_ball frames.
Per-detection precision only reaches 25% at s = 0.9 where recall is 0.5%.

Dense clips (964 ball frames): `full_bilinear` top1 2.5% (24/964), oracle 10.0%;
`tile_2x` 1.7% (oracle 17.9%). Per clip, top1 for `full_bilinear`: `arsenal_example` 9%,
`astonvilla_seg080` 0% (568 frames, oracle 1%), `bournemouth_1st_seg070` 3%. Man City clips
(424 ball frames): `full_bilinear` 10.8%, `tile_2x` 8.5%, `tile_native` 0.9%. The full per-clip table is in
`wasb_zeroshot_results.json` (`per_clip`).

## Comparison with the RT-DETR machine labels (Man City, same frames)

RT-DETR candidates are the `in_play_ball` rows of `machine_labels/arsenal_mancity_20250925_seg*.jsonl`
(`rtdetr-l_1920_v11`, imgsz 1280, conf >= 0.1), top-1 per frame by confidence. Man City GT frames
on clips with machine labels: 383 ball + 23 no_ball.

**Leakage check.** v11 (the RT-DETR training set, 1,035 train / 215 valid / 268 test images, 1920x1080,
no augmentation) names images either `<clip>_<frame>` or by an index into the full match video, which a
name check cannot map to a segment (todo T-33). So `match_v11.py` compares 64x36 grey thumbnails of every
v11 image against every GT frame; true matches have mean abs diff 0.2-0.3 and everything else is
> 3 (threshold 3.0). Of the 383 ball frames, **181 are pixel-identical to a v11 train image**, 14 to valid,
1 to test, 187 to none (no_ball: 2 train, 5 valid, 16 none). Tables use "clean" = not in v11 train
(valid and test images are not trained on, though valid picks the checkpoint).

| frames (d = 10) | n ball / no_ball | detector | top1 | oracle | best s | recall | prec_f | F1 | no_ball FP |
|---|---|---|---|---|---|---|---|---|---|
| **clean (not in v11 train)** | 202 / 21 | RT-DETR | **154 = 76.2%** | 82.7% | 0.6 | 70.8% | 97.9% | 0.822 | 4.8% |
| | | WASB `full_bilinear` | 31 = 15.3% | 28.2% | 0.1 | 15.3% | 14.6% | 0.150 | 81% |
| | | WASB `full_area` | 25 = 12.4% | 21.3% | 0.1 | 12.4% | 12.0% | 0.122 | 81% |
| | | WASB `tile_2x` | 18 = 8.9% | 27.2% | 0.2 | 8.9% | 8.5% | 0.087 | 76% |
| | | WASB `tile_1p5x` | 4 = 2.0% | 13.9% | 0.4 | 2.0% | 2.0% | 0.020 | 76% |
| | | WASB `tile_native` | 3 = 1.5% | 5.4% | 0.1 | 1.5% | 1.3% | 0.014 | 100% |
| clean, in no v11 split at all | 187 / 16 | RT-DETR | 145 = 77.5% | 84.0% | 0.6 | 72.7% | 98.6% | 0.837 | 6.3% |
| | | WASB `full_bilinear` | 25 = 13.4% | 25.7% | 0.1 | 13.4% | 12.9% | 0.131 | 75% |
| all, incl. trained-on (memorised) | 383 / 23 | RT-DETR | 326 = 85.1% | 90.1% | 0.5 | 81.5% | 95.7% | 0.880 | 17% |
| | | WASB `full_bilinear` | 45 = 11.7% | 23.8% | 0.1 | 11.7% | 11.5% | 0.116 | 83% |

Only the clean rows are a fair RT-DETR number; the "all" row is inflated by memorised frames (86 -> 76%).
Even the clean RT-DETR figure is probably optimistic: GT frames were largely chosen through Ball Check,
whose candidate pool is RT-DETR proposals, so frames where RT-DETR proposes a ball are over-represented,
and "clean" does not exclude frames adjacent to trained ones. WASB is far below it either way.

## Failure modes

Montages (`wasb_zeroshot_failures_full.jpg`, `..._aston.jpg`; 96 px crops at 3x, green = GT, red = top-1
candidate with score >= 0.3):

* **The model mostly does not fire on the ball at all**, including isolated, unblurred balls on plain grass
  (white and yellow). On the Aston Villa clips (a bright yellow ball, 1280x720 source, 579 frames) no frame
  has a candidate within 10 px at 3.75x down, and 79% have no candidate >= 0.3 anywhere.
  15% of `full_bilinear` ball frames have no candidate >= 0.1 at all (35% on the Aston clips, where 79% have none >= 0.3). When a candidate lands on the
  ball it is accurate (nearest-candidate distance p5 = 4 px, mean offset < 1 px), so localisation is fine;
  detection is the problem. Likely causes (not tested): the model is calibrated to a ~3-4 px white ball on a
  fixed ISSIA camera, and our ball is yellow/blurred/on a panning broadcast shot.
* **Misses on hard frames** also include a ball over the crowd, against hoardings, motion-blurred, or held
  by a player (all in the montage), but the easy frames fail too, so this is not the main cause.
* **False positives**: advertising-board text and logos ("Hotels.com", "WEST"), crowd texture, the Premier
  League/score graphics, a broadcast "eye" graphic, white socks and boots, and the edge of the black letterbox
  bars. This is the failure the WASB paper reports on soccer (shoes).
* **Native tiles are the worst** (top1 0.3%, ~20-30 detections per frame): a 16 px ball is ~4x larger than
  what the model learned, and 24 tiles multiply the clutter. Scale is not the only issue, since the 3.75x
  mode that matches the training scale is also at 5%. Tiling raises the oracle only from 14% to 21% (2x), at
  the cost of 10 detections per frame.

## Runtime and VRAM (RTX 2070, torch 2.8 fp32)

Per 3-frame window including CPU resize/normalise (desktop under heavy load from other jobs, load avg ~20,
so CPU prep is inflated). Peak VRAM: 965 MiB for one 288x512 tile; 3,778 MiB at 4 tiles per forward
(about 0.94 GB/tile; a 24-tile batch OOMs on 8 GB, hence the chunking).

| mode | tiles | ms/window (CPU prep + GPU + post) | ms GPU |
|---|---|---|---|
| `full_bilinear` | 1 | 37 | 27 |
| `full_area` | 1 | 44 | 27 |
| `tile_2x` | 6 | 207 | 145 |
| `tile_1p5x` | 12 | 412 | 283 |
| `tile_native` | 24 | 825 | 564 |

Full video at 25 fps would run at ~27 fps (full frame) or ~1.2 fps (native tiles) on this card.
`wasb_zeroshot_runtime.json` has the raw numbers (the 720p rerun is in `..._720p_rerun.json`).

## Licence

* **WASB-SBDT code**: MIT, "Copyright (c) 2023 NTT Communications Corporation" (`LICENSE.md`).
  **BlurBall code** (`cogsys-tuebingen/blurball`): also MIT, same NTT copyright text (`LICENSE.md`).
* **WASB weights** (`wasb_soccer_best.pth.tar`, Google Drive): no licence file or terms accompany them; the repo
  licence covers "the Software" and MODEL_ZOO.md says nothing more. They were trained on the Soccer set of
  D'Orazio et al. (ISSIA-CNR style); that dataset's terms are not stated in the paper or repo and were not
  checked. Treat the weights as research-use until verified. BlurBall's weights (Nextcloud) are likewise
  unlicensed in the README; its table-tennis dataset has no stated licence (videos are described as CC).
* Our own use (zero-shot evaluation) is fine; shipping a product built on these weights is not cleared.

## Is there usable training code?

* **WASB-SBDT (NTT)**: effectively evaluation-only. `runners/train_and_test.py` (a `Trainer` with an epoch
  loop, losses, optimiser and scheduler) is in the tree but its import and registration are commented out in
  `runners/__init__.py`, and `configs/runner/` has only `eval.yaml` and `extract_frame.yaml` (no train config).
  The pieces (losses `hm_*`, optimisers, dataset loaders, heatmap generator) exist, so it could be revived,
  but nothing runs out of the box.
* **BlurBall**: builds on WASB and ships a working training path (`runners/__init__.py` registers `Trainer`,
  `configs/train.yaml`, `train_blur.yaml`, `python src/main.py --config-name=train_blur`), MIT. Datasets are
  per-sport adapters (`datasets/{soccer,tabletennis,...}.py`) reading frame folders plus CSV/annotation files, with
  windows built by sliding over *every* consecutive annotated frame. Dependencies are old (hydra, torch 1.x era
  per the Dockerfile) but it is plain PyTorch.
* **>3-frame windows**: yes, structurally. `frames_in`/`frames_out` are config values; the HRNet stem is
  `Conv2d(3*frames_in, ...)` and the head has `frames_out` channels, and the dataset adapters slice
  `frames_in` consecutive frames. Only the stem and head shapes change, so pretrained weights need conv1/head
  surgery (or are dropped). BlurBall states the "duplicated frames affect BlurBall" caveat (relevant to our
  25 fps vs 50 fps clips).
* **Gaps for our data**: (1) our GT is sparse (many isolated frames), and the loader treats an unlabelled frame
  as "ball not visible" (all-zero heatmap), so we need a loss mask for unlabelled frames or training only on the
  dense runs (`astonvilla_seg080`, `bournemouth_1st_seg070`, `arsenal_example`, ~960 frames, plus the
  `vittrack`-propagated ranges); (2) a native-resolution tile/crop sampler (the stock "crop" transform only
  rescales); (3) a dataset adapter for our JSONL marks; (4) fixed 288x512 assumptions in the configs are plain
  numbers and can change.

## Verdict

**Do not fine-tune WASB as a starting point on the strength of this result; build our own heatmap net.**
The zero-shot result shows the pretrained soccer weights carry almost no usable signal for our ball (5% top-1,
not even for clean yellow-ball frames), so the checkpoint is not a meaningful head start. What *is* reusable
is the MIT code: the 1.5 M-param full-resolution HRNet (27 ms/frame, ~1 GB VRAM per 288x512 tile) and
BlurBall's trainer scaffolding. The practical plan is our own training script that borrows that architecture,
with (a) the loss mask for sparse labels, (b) training at a scale chosen on our data (the ball is 4-16 px
depending on the downscale; 1.5x-2x tiles or a 1280-wide full frame are the candidates; the zero-shot run says
nothing about which is best, since the model fails at every scale), and (c) a window longer than 3 frames. Initialising from the WASB
soccer weights is a free A/B arm (conv1 surgery) but should not be relied on. Caveat: this experiment tests
zero-shot transfer only; it does not show that fine-tuning would fail, only that there is little to inherit.

## Caveats

* The three Aston Villa clips (`astonvilla_seg020/050/080`, 579 of 1,495 ball frames) are **1280x720**, not
  1080p. `wasb-v2` mishandled them; they were rerun (`wasb-v3-720p`) after upscaling each frame to 1920x1080
  (bicubic) so that tiles and scoring share one coordinate space. For these clips the native-tile modes are
  therefore upscaled 720p, not true native pixels, and the ball is ~11 px native.
* `bournemouth_*` clips are 50 fps and the others 25 fps; WASB trains on 25 fps.
* The pip OpenCV 5.0 on the mini decodes `arsenal_example` and `mancity_part050/093` (colour space "unknown")
  as black frames. Inference and the v11 matching of Man City `seg` clips are unaffected (inference used
  desktop OpenCV 4.11, all clips decode non-black; the RT-DETR comparison clips decode fine on the mini);
  the failure montage falls back to ffmpeg for these.
* Only the centre-frame heatmap of each sliding window was scored (the WASB "step=3" pipeline uses all three
  outputs per window); the online tracker was not used. 37 no_ball frames is a small denominator.
* One hit definition (top-1 within d of the GT centre); per-frame precision counts frames, not detections.

## Files

`experiments/wasb_zeroshot/`: `build_evalset.py`, `match_v11.py`, `run_wasb.py`, `score.py`,
`viz_failures.py`, `prefect/` (flow + deployment yaml). `docs/experiments/`: this page,
`wasb_zeroshot_results.json` (all metrics, sweeps, per clip), `wasb_zeroshot_runtime*.json`,
`wasb_zeroshot_pred.tar.gz` (raw per-frame candidates for every mode), the two failure montages.
Working data on the mini: `/mnt/storage/exp/wasb/{inputs,runs,final,repo,weights}`.
