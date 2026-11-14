# Multi-frame (camera-motion-compensated) dataset for a temporal ball detector

Status: dataset built (CPU only, no training). Branch `exp/multiframe-dataset`. Date 2026-11-14.

Goal: give the ball detector frames t-k and t+k aligned into frame t's coordinates, for every
labelled Roboflow v11 image and for human-GT frames of clips the detector never saw, so a
temporal model can be trained and evaluated without any new annotation.

KB pages relied on: `footy-ball-detection-findings` ("The key enabler": neighbours are
recoverable from the file names, cuts are the trap; the dataset notes; the benchmark
trap), `footy-data-s3` / `footy-data-plane-reconciliation` (where the data lives, GT tiers),
`footy-labeller-gotchas`. Research note `research_notes/Offline football ball tracking
models/tiny_object_video_detection.md` (YOLOMG: frame differences from t-k and t+k aligned by
homography, best k=2; Temporal-YOLOv8 channel stacking, untested on moving cameras; "patching
RT-DETR to 9 channels is UNVERIFIED"). Split/exclusion logic read from
`scripts/export_training_dataset.py`, `ball_eval/dataset.py`, `clip_id.py` (T-03).

## 1. Image name to (video, frame) mapping

Roboflow v11 lives at `/mnt/storage/footy_data/detection_dataset/roboflow_dataset_11`
(`train 1035 / valid 215 / test 268`, 1920x1080, class ids `ball 0, coach 1, in_play_ball 2,
person 3, player 4, player_sub 5, referee 6`). Two name families, both mappable
(`multiframe/mapping.py`):

| Name | Meaning | Mapping |
|---|---|---|
| `<clip_stem>_<NNNNNN>_(png\|jpg).rf.<hash>` (stem like `astonvilla_seg080`) | NNNNNN is the frame index **inside the clip** | `ballcheck_clips/<stem>.mp4` (alias symlink), frame = NNNNNN |
| `arsenal_mancity_20250925_<NNNNNN>_png.rf...` (full-match frames, 388 of 1518 images) | NNNNNN is the ordinal of `arsenal_mancity/full_video_frames/*.png`, a **1 fps** extraction | absolute source frame = `25*NNNNNN + 12` (ffmpeg `fps=1` takes t+0.5 s); inside the broadcast segment clip given by `clips/segments.json` (`start_frame`), frame = absolute - start |

Gotchas found: the source `.webm` cannot be used (OpenCV returns the same frame for every
seek), so full-match frames are located in the segment clips; the `+12` offset is not
documented anywhere and was found by pixel search; v11 frames whose absolute frame falls in a
non-broadcast gap have no clip (2 images). `eval_data/clips/` is **empty** on this machine, so
the harness eval-exclusion list the export script builds from it is empty here too.

**Verification:** 60 random mapped images, decoded frame t vs the v11 JPEG: all 60 best-match at
offset 0 (mean abs grey diff 0.3-2.2, i.e. JPEG noise; t+-1 is >10). Includes 9/9 full-match
images. (`multiframe_mapping_report --verify 60`.)

**Coverage** (`mapping_report.json`; ball instance = class `in_play_ball`, the only populated
ball class; the KB's "826 train ball instances" does not match the export on disk, which has 1,010):

| split | images | balls | mapped | k=1 images / balls | k=2 | k=4 |
|---|---|---|---|---|---|---|
| train | 1035 | 1010 | 1034 / 1009 | 1020 / 1000 | 1017 / 997 | 1006 / 987 |
| valid | 215 | 211 | 215 / 211 | 207 / 205 | 204 / 202 | 198 / 196 |
| test | 268 | 243 | 267 / 243 | 265 / 241 | 264 / 241 | 261 / 239 |

"k=n" = frames t-n and t+n both exist in the clip (misses are clip edges). 99.9% map; at k=2
98.2% of train, 94.9% of valid images keep both neighbours. We still emit the edge images
with a duplicated neighbour (below), so the split is identical to v11.

Leakage note: v11's split is **by image, not by clip**. By canonical clip id, 18 clips
appear in both train and valid and 33 in both train and test (mostly full-match frames that
land in the same 101 s segments). The existing val/test numbers therefore reward scene
memorisation (the KB already flags `astonvilla_seg080` / `bournemouth_1st_seg070`).

## 2. Camera-motion estimation

Compared on 30 v11 frames (stratified, <=6 per clip, t vs t-2) at a 480 px working width,
single-thread CPU time per pair (the box was at load 20-50, wall time is meaningless). Quality
= mean abs grey error (blurred, 480 px) over **grass pixels of frame t not under a labelled
person**, after warping t-2 into t. Identity (no alignment) = 1.07 median; 40% of pairs have a
moving camera (identity > 1.5, median 1.98). "worse" = accepted pairs whose aligned residual
exceeds identity.

| method | masking | failed | pitch resid. median | worse than identity | ms / pair (CPU) |
|---|---|---|---|---|---|
| ORB + RANSAC @480 | person boxes | 0% | 1.11 | 50% | 285 |
| ORB + RANSAC @480 | none | 0% | 1.29 | 47% | 308 |
| ORB + RANSAC @960 | person boxes | 0% | 1.39 | 47% | 467 |
| AKAZE + RANSAC @480 | person boxes | 6.7% | 1.10 | 39% | 168 |
| AKAZE + RANSAC @480 | none | 0% | 0.91 | 33% | 154 |
| ECC homography @480 (mine) | person boxes | 0% | 0.62 | 13% | 193 |
| ECC homography @480 (mine) | none | 0-3% | 0.63 | 7% | 195 |
| ORB then ECC @480 | person boxes | 0% | 0.63 | 13% | 455 |
| **LK + RANSAC (`calibration/temporal`, whole frame minus overlay and person boxes)** | person boxes | **0%** | **0.49** | **6.7%** | **57** |
| same, no boxes (whole frame minus overlay) | none | 0% | 0.61 | 10% | 62 |
| same, grass-only mask | none | 0% | 0.81 | 20% | 62 |
| LK then ECC / its own ECC | person boxes | 0% | 0.51 | 3.3% | 520 / 499 |

(`alignment_eval_d2.json`, `compare_temporal_d2.json`; ORB/AKAZE/ECC rows come from an
earlier run of the same sample, n=30 with the identical seed. Small n; differences of
~0.1 are within noise. d=1 and d=4 were not measured.)

What the numbers say:

- Feature matching on ORB/AKAZE is *not* better than doing nothing on broadcast frames: ~1000
  RANSAC inliers, but they sit on the stands and hoardings (different depth/plane), so the
  homography is wrong for the pitch (50% of pairs worse than identity). Intensity-based
  methods (ECC) and tracked-corner LK on the whole frame are clearly better; on moving-camera
  pairs ECC gives 0.85-0.9 vs identity 1.98, on near-static pairs 0.55-0.6 vs 0.92.
- **Chosen: LK + RANSAC from `calibration/temporal.py` (shared module, see below).** Best
  residual, lowest identity-regression rate, 57 ms/pair CPU (~73 ms wall under load), 3.4x
  faster than my ECC and 9x faster than LK+ECC, which gained nothing. This agrees with the
  other session's measurements (LK ~45 ms, ECC 0.6-1.5 s with no accuracy gain, whole-frame
  mask beats grass-only).
- Player masking: for LK, boxes help (0.49 vs 0.61 unmasked); for my ECC they did not
  (0.62 vs 0.63; within noise). Grass-only is clearly worse (0.81, 20% worse than identity).
- Disagreement recorded: the shared module's cut notes say non-cuts have histogram distance
  <= 0.30; on my three 500-1400 frame scans non-cuts were <= 0.09 (p99 0.057), cuts >= 0.37
  (different histogram: 8x4x4 HSV, 160 px, Bhattacharyya). I use 0.2 for the dataset.

Shared-module provenance: `src/footy_track/calibration/{__init__,geometry,pitch_model,
refine,temporal}.py` and `prelabel/{__init__,candidates}.py` are copied **verbatim** from
branch `exp/calibration-temporal` at commit `7988bde` (temporal.py needs those files to
import; identical blobs at identical paths merge cleanly when that branch lands). Use is
through one adapter, `multiframe/align.py::estimate_pair_lk`, and `estimate_pair(a, b,
mask_a, mask_b, scale, method)` returns `PairMotion(H_px, H_norm, ok, inliers, residual,
method, reason)`, a subset of the shared API's `PairMotion` (which adds `n_tracked`, `score`). `H` maps frame a pixels to frame b pixels;
render b into a with `warp_to_ref` (`warpPerspective(..., WARP_INVERSE_MAP)`).

### Full-run alignment stats (k=2, LK, 3,266 neighbour slots)

| split | ok | edge (clip boundary) | cut | alignment failed |
|---|---|---|---|---|
| train (2068) | 2039 | 17 | 11 | 1 |
| valid (430) | 418 | 11 | 1 | 0 |
| test (534) | 526 | 3 | 1 | 4 |
| heldout (234) | 225 | 8 | 1 | 0 |

So 98.0% aligned; 0.4% cuts, 0.15% alignment failures, 1.2% clip edges. Alignment
failures: 2 "too few features", 3 with 17-22 inliers (< 25). Alignment ms per ok pair: median
73 wall (loaded box). Whole build: 4,024 s, single process, 2 threads, for 1,633 images.

## 3. Cuts

Broadcast cuts make t-k a different camera: the diff turns into noise on exactly the frames
that matter. A neighbour is replaced by **a copy of gray(t)** (so |diff| = 0) when, checked
in this order: it is outside the clip (`edge`); the max Bhattacharyya distance between
*consecutive* frames from t to t+-k exceeds 0.2 (`cut`); or alignment fails (`align_fail`:
<25 RANSAC inliers, inlier ratio, degenerate H, residual gate of the shared module).

Calibration (`multiframe_cut_scan.py`, 2,531 frames of three clips): consecutive-frame
distance is bimodal; ordinary frames have median 0.03, p99 0.057, max 0.09; the five cuts
found are 0.37, 0.45, 0.69, 0.72, 0.80 and ECC alignment of those pairs fails or has residual
30-50 (ordinary pairs: < 5). 14 neighbour slots were flagged as cuts in the dataset, one at
0.24 (near threshold; may be a flash rather than a cut). Each stored image records per-side
status, frame number, histogram distance, inliers and residual in `meta.jsonl`.

## 4. Channel format decision and Ultralytics support

**Decision: (A) 5 channels `[gray(t-k) aligned, B, G, R of t, gray(t+k) aligned]`, one
multi-page TIFF per image, 8-bit, LZW, full 1920x1080.** (B) RGB + |aligned diff| is derivable
from A in one line (`abs(g(t) - prev)`, `abs(next - g(t))`) and can be made at load time or in a
custom dataset; A cannot be recovered from B.

Reasoning:
- The viewer shows why signed grey planes matter: a moving ball appears as a bright blob at
  both t-k and t+k positions in A (a dipole after differencing) and the network can use the
  sign and direction; |diff| discards both.
- A **resting ball** (e.g. `train_astonvilla_seg080_000039_ball_static.png`) cancels almost
  completely in any diff, so a pure-diff model loses the ball exactly when it is easiest.
  A keeps the full RGB frame, so the model degrades to the single-frame detector there.
- Extra channels can be initialised so the model starts as the pretrained single-frame
  detector (copy the RGB stem filters, zero the other 2 input channels).
- Cost of A over B: a model must learn differencing; with ~1,000 balls B's explicit prior
  may train faster. Cheap to test both: B is a transform of A.

**TIFF note:** `cv2.imwritemulti` of 2-D pages; Ultralytics' `imread` reads `.tif/.tiff` with
`cv2.imdecodemulti` and stacks 2-D pages on axis 2 -> `(1080, 1920, 5)` uint8 (checked).
Channel order is stored as B, G, R: for 5 channels Ultralytics' `Format` only reverses the
channel order when there are exactly 3, so the tensor keeps this order. **When inflating a
pretrained RGB stem, put R/G/B filters on input channels 3/2/1** (not 1/2/3).

**Ultralytics support, pinned version 8.3.241 (`uv.lock`), source read and exercised on CPU:**

| piece | result |
|---|---|
| dataset yaml `channels: 5` | read in `data/utils.py:431` and passed to `YOLODataset` (`dataset.py:88`) |
| YOLO model (`yolo11n.yaml`, `DetectionModel(ch=5)`) | builds, first conv takes 5 channels, forward OK (checked) |
| RT-DETR (`rtdetr-l.yaml`, `RTDETRDetectionModel(ch=5)`) | builds, first conv takes 5 channels, forward OK (checked); `RTDETRTrainer.get_model` passes `ch=self.data["channels"]` (`models/rtdetr/train.py:55`) |
| data pipeline, `YOLODataset` and `RTDETRDataset`, augment on/off | returns `(5, H, W)` uint8 tensors with labels (checked); `RandomHSV` skips non-3-channel images, `LetterBox` has a multispectral branch, mosaic allocates `img.shape[2]` channels; `cv2.warpAffine/resize/flip` work on 5 channels (border fill is 114 only on the first 3 channels, 0 on the rest) |
| pretrained weights | stem shape mismatch -> `intersect_dicts` silently drops it (random stem). Needs manual stem inflation. |
| full `train()` loop with 5 channels | **UNVERIFIED**: a 1-epoch CPU run of both models was started and killed to shed load before it finished (it did get through model build, dataset build and the first epoch header). Validation, AMP and plotting code paths with 5 channels were not exercised. |

## 5. What the aligned diff images show

Four PNGs in `docs/experiments/multiframe_dataset/` (`visualise_multiframe.py`): top
left frame t (ball boxed), top right the aligned |t - (t-2)| x4 over the whole frame; bottom
a 96 px crop (x3) around the GT ball: aligned prev, RGB t, aligned next, two aligned diffs
and the same diff with **no** alignment.

- **Static structure cancels.** On the pitch, field lines, mowing stripes, hoardings and
  stands, alignment leaves only thin faint edges; players come out as bright edge outlines.
  Crowd texture leaves a noisy residual (high-frequency, sub-pixel misregistration) but it
  is dim and different in character from a ball. The scorebug/logo is bright (screen-static
  graphics move relative to the aligned pitch) but is a fixed-position artefact the model
  can learn to ignore.
- **A moving ball is a dipole / short streak** in the diffs (bright disc at both its t-k
  and t+k positions in the two diffs; `heldout_seg035_000287`, `_000232`: a clear white
  blob under the yellow GT circle in `|t-prev|` and `|next-t|`).
- **But the ball is not isolated in the diff when it is in contact with players**: in the
  scrum (`train_mancity_seg030_000430`) and in `_000232` it sits inside a bright tangle of
  leg and boot motion, so the diff alone does not separate it; appearance (RGB) must still do
  that.
- **A resting ball vanishes**: `train_astonvilla_seg080_000039` (ball stationary on grass,
  player nearby) shows only a faint ring in both diffs, which is why A keeps RGB.
- Alignment helps but is not dramatic when the camera is nearly still (the unaligned crop in
  `_000287` is similar), and the full-frame gap shows up when the camera pans (my 30-pair
  numbers: residual 0.85 aligned vs 1.98 unaligned on moving-camera pairs).

## 6. Dataset

`/mnt/storage/footy_data/multiframe_v1/k2/` (new, nothing overwritten; not uploaded to S3
or the desktop). **5.4 GB, 1,633 TIFFs** (about 3.3 MB each).

```
images/{train,val,test,heldout}/<id>.tiff      5 pages, 1920x1080, uint8
labels/{train,val,test,heldout}/<id>.txt       YOLO, v11 class ids, copied unchanged for v11 images
data.yaml           train/val/test, nc 7, channels 5       (v11 names, same ids as the detector)
data_heldout.yaml   train/images/train, val -> images/heldout
meta.jsonl          per image: source video, frame, per-neighbour status/inliers/residual/hist
manifest.json       counts and options (copy: manifest_k2.json next to this doc)
```

| split | images | ball instances | both neighbours aligned (images / balls) |
|---|---|---|---|
| train (= v11 train, 1 unmappable image dropped) | 1034 | 1009 | 1007 / 988 |
| val (= v11 valid) | 215 | 211 | 203 / 201 |
| test (= v11 test, 1 unmappable dropped) | 267 | 243 | 261 / 239 |
| heldout | 117 | 117 | 108 / 108 |

Image ids are the v11 file stems (so `train/val/test` have the same image ids as the existing
detector). RGB(t) is decoded from the video, not from the v11 JPEG (identical up to JPEG
noise), so t and its neighbours share codec statistics.

**Held-out eval set**: human GT (tier 1 `labeller`, tier 2 `vittrack`+`human_checked`;
tier 3 excluded) frames with a ball, from `ball_gt_marks`, in clips whose **canonical** clip id
(`footy_track.clip_id`) is absent from v11 *train*, minus frames that are already v11 images.
117 frames / 117 balls. Of these **84 balls (6 clips: `arsenal_bournemouth_1st_seg035/031/015/021/060`,
`arsenal_mancity_part093`) are strictly clean** (clip in none of train/valid/test; flag
`strict_heldout` in `meta.jsonl`); the other 33 come from clips present only in v11 valid/test
(`arsenal_example` 20, `bournemouth_seg009` 10, three single frames). Labels for held-out
frames are the tier<=2 GT boxes of that frame in v11 class ids.
**84 strict balls (76 with both neighbours aligned) is a thin benchmark** (the KB's 44-instance warning applies); report
instance counts with any metric, and read the 117 as secondary. Because the repo-wide
split-by-clip rule makes nearly every GT clip a training clip, there is little clean GT left;
more would need new labelling in unseen clips.

Build: `uv run python -m footy_track.scripts.build_multiframe_dataset --k 2 --method lk`
(refuses to write into a non-empty directory). k=2 was chosen on coverage (98% of images
have both neighbours; k=4 gets 97%) and because YOLOMG's best k is 2; k=1 and k=4 need only
a re-run (about 1.1 h on this box).

## 7. Reproduce / code map

- `src/footy_track/multiframe/` `mapping.py` (name -> source), `align.py` (estimate_pair,
  warp, masks, hist distance), `frames.py` (reader, labels, machine-label boxes), `v11.py`.
- `scripts/`: `multiframe_mapping_report.py` (coverage, `--verify`), `multiframe_alignment_eval.py`,
  `multiframe_cut_scan.py`, `build_multiframe_dataset.py`, `visualise_multiframe.py`.
- `docs/experiments/multiframe_dataset/compare_temporal.py` documents how the shared-module
  comparison was run (scratch tree of commit 7988bde; not importable on this branch alone).
- `tests/multiframe/` (5 tests: name parsing, pan recovery for ECC and LK, masked box, cut
  histogram). Run with `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest -o addopts=""` if the hydra
  pytest plugin is broken in the env.

## 8. Before GPU training

1. Run one real `train()` epoch with `channels: 5` for YOLO and RT-DETR on the desktop GPU
   (the part marked UNVERIFIED above), including validation and AMP.
2. Stem inflation: copy pretrained RGB filters (R/G/B to input channels 3/2/1), zero-init
   channels 0 and 4, scale if needed; otherwise the stem is random.
3. Hand the dataset to the desktop (S3 or rsync; 5.4 GB) - not done here.
4. Decide A vs B on held-out strict (84 balls) and v11 val together, always reporting
   the instance count and the by-clip leakage caveat above. Train with `imgsz=1280`.
5. Mosaic/translate augmentation moves the 5 channels together (good); `hsv_*` is skipped for
   5 channels, so colour augmentation is lost.
6. Known limits: stationary balls and balls in player contact carry little motion signal;
   25 fps with the ball moving < 1 diameter/frame gives short streaks (k=2 is 80 ms).
