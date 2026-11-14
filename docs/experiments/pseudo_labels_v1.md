# Pseudo-labelled hard-positive ball set v1 (self-training data for the next detector round)

Branch `exp/pseudo-labels-v1` (cut from `exp/sam2-bidir-ball`). Code: `experiments/pseudo_labels_v1/`. Dataset (v1 multiframe format, 7.8 GB, not in git): `/mnt/storage/footy_data/multiframe_pseudo_v1/`. Provenance and stats: this directory (`meta.jsonl`, `selection_stats.json`, `exclusions.json`, `audit.json`, `audit_ext.json`, `pseudo_labels_v1.json`) and the two contact sheets `contact_sheet_hard.png`, `contact_sheet_easy.png`.

## Summary

* **2,115 images from 62 clips** (the only clips that survive the exclusions): 356 S1 gap fills + 759 B1 detections with det conf < 0.6 (the **hard positives, 1,115**) and 1,000 B1 detections with conf >= 0.8 (easy). Stride >= 5 frames within a clip. **This is below the 3-6k target**; the stride-5 cap of a 21,340-frame pool is about 4.2k and the hard candidates before thinning are 3,020 (954 S1 fills + 2,066 low-confidence nodes); see "Selection".
* Every image has the pseudo ball (class 2) **plus machine player/other-class boxes at conf >= 0.5** (16.2 boxes per image; only 2 of 2,115 images have none, so only 2 would need ball-only training).
* Label quality, measured against human GT on the GT-bearing clips of the previous experiments (the pool itself has only 103 GT frames): B1 conf >= 0.8: 138/138 within 10 px; 0.6-0.8: 229/233 (98.3%); **conf < 0.6: 38/49 (77.6%)**; S1 fills: 19/20 (95%), the one miss an 11.3 px near-miss. Roughly a quarter of the low-confidence hard positives are wrong (boots, crowd, kit), see the contact sheet.
* Format check: RGB pages are bit-identical to the original v1 builder on the same frame; 5 TIFF pages, 1080x1920, uint8, same class ids, same `data.yaml` conventions.

## 1. Clip pool and exclusions

Pool = the Man City broadcast segments that have `machine_labels` (162 clips, canonical ids `arsenal_mancity_segNNN`), minus:

| exclusion | clips (of 162) | rule / source |
|---|---|---|
| contains a v11 **valid**, **test** or **heldout** image | **97** | per-image `clip_id`/`split` in `multiframe_v1/k2/meta.jsonl` (pixel-verified mapping); that file has 107 distinct eval clip ids (valid 35, test 85, heldout 11), 97 of which have machine labels (the rest are Aston Villa / Bournemouth / part clips) |
| the 9 trajectory test clips (`exp/offline-ball-tracking` fc05732, split.test) not already excluded above | **3** | seg010, seg011, seg031 (they are v11 *train* clips; the other 6 test clips are also v11 eval clips) |
| `arsenal_example`, anything `arsenal_norwich` | **0** | neither has machine labels; listed for the record |
| **excluded in total** | **100** | |
| **pool** | **62 clips, 21,340 frames** | |

The v11 *train* clips are **not** excluded (73 clips contain a v11 train image; 91 selected frames lie within 5 frames of one, flagged `near_v11_train_5` in `meta.jsonl`).

Excluded because they contain a v11 valid/test/heldout image (97): seg000, seg003, seg007, seg009, seg013, seg017, seg020, seg022, seg023, seg025, seg026, seg030, seg032, seg033, seg037, seg041, seg043, seg047, seg048, seg049, seg053, seg054, seg058, seg059, seg062, seg063, seg065, seg070, seg072, seg076, seg077, seg078, seg080, seg083, seg084, seg085, seg086, seg087, seg088, seg089, seg091, seg094, seg095, seg096, seg097, seg099, seg100, seg101, seg102, seg105, seg106, seg108, seg110, seg111, seg113, seg114, seg115, seg116, seg117, seg120, seg122, seg124, seg125, seg126, seg127, seg128, seg129, seg130, seg131, seg132, seg133, seg134, seg135, seg136, seg138, seg139, seg140, seg141, seg142, seg143, seg144, seg145, seg146, seg147, seg148, seg149, seg150, seg151, seg152, seg153, seg155, seg156, seg157, seg158, seg159, seg160, seg161 (all `arsenal_mancity_segNNN`; numbers shown).

Excluded as trajectory-test clips only (3): seg010, seg011, seg031. All 9 trajectory test clips: seg010, seg011, seg031, seg094, seg101, seg115, seg134, seg135, seg140.

Pool (62): seg001, seg002, seg004, seg005, seg006, seg008, seg012, seg014, seg015, seg016, seg018, seg019, seg021, seg024, seg027, seg028, seg029, seg034, seg035, seg036, seg038, seg039, seg040, seg042, seg044, seg045, seg046, seg050, seg051, seg052, seg055, seg056, seg057, seg060, seg061, seg064, seg066, seg067, seg068, seg069, seg071, seg073, seg074, seg075, seg079, seg081, seg082, seg090, seg092, seg093, seg098, seg103, seg104, seg107, seg109, seg112, seg118, seg119, seg121, seg123, seg137, seg154.

## 2. Tracking: B1, then S1 (conf >= 0.8, spacing 10)

B1 = the harness solver with the dev-tuned camera-on parameters, unchanged. Anchors = B1 solver nodes with conf >= 0.8 thinned to >= 10 frames apart (nodes bordering a gap always kept); S1 = the existing gate (forward/backward SAM2.1-small/crop512 passes within 5 px, or a lone pass within 12 frames of its anchor) applied to B1 gap frames only, 50 frames maximum per pass. Additionally, a fill whose track crosses a flagged camera cut is dropped (4 fills). Frames within 2 frames of a cut or of a clip end are not eligible.

Pool-wide rows with a pseudo point: **B1 detections 15,747, B1 interpolated 431, S1 fills 962** (of 17,140 rows, 21,340 frames; the other frames have no track and are left out, not used as negatives). Per-frame provenance (`provenance_all.jsonl.gz` in the dataset dir; selected frames in `meta.jsonl` here): source, det conf, solver p (Ball-Check calibration of the conf), forward/backward agreement in px (S1 fills: |fwd - bwd|, null for a lone pass), distance to the nearest anchor, anchor conf, near-GT and near-v11-train flags. For S1 fills the box size is the mean of the bracketing anchors' detector boxes (SAM2 gives a mask centroid, not a box); the median ball box is 18 x 17 px.

## 3. Selection

Greedy random order per category with a per-clip stride >= 5 frames over everything already taken: S1 fills first, then B1 detections with conf < 0.6, then a random sample of conf >= 0.8 detections.

| category | candidates (eligible frames) | selected | notes |
|---|---|---|---|
| hard: S1 gap fill | 954 | **356** | 253 agreement, 103 lone-pass; median |fwd-bwd| 0.39 px (p90 1.5 px); median distance to the anchor 8 frames (max 38) |
| hard: B1 det conf < 0.6, kept by the solver | 2,066 | **759** | conf 0.30-0.60 (median 0.50); median 25 frames from the nearest conf >= 0.8 anchor; 77 are in clips with no conf >= 0.8 node at all |
| easy: B1 det conf >= 0.8 | 6,786 | **1000** | random sample (cap 1,000, kept below the hard count) |
| **total** | | **2,115** | 62 clips, at most 148 images per clip (7%) |

Why not 3-6k: the pool is only 21,340 frames because 100 of 162 Man City clips contain eval images; at stride 5 at most about 4.2k frames fit, and only 3,020 frames are hard candidates before any thinning. Reaching 3k hard frames would need stride 1-2 (near-duplicate neighbours) or a larger pool (the 162 machine-label clips are all the Man City segments that exist). The easy count is a parameter (`--easy`).

## 4. Players and other classes

The detector trains on all seven v11 classes, so each pseudo image carries the machine boxes of **player: 30,660, referee: 2,911, coach: 513, player_sub: 133, person: 61** (v11 class ids, conf >= 0.5, from `machine_labels`; tagged in `meta.jsonl` as `other_boxes_source: machine_labels conf>=0.5`). Mean 16.21 boxes per image; **2 images have none** and would be ball-only. Other ball candidates (class 0/2) from the detector are deliberately not added: the pseudo ball is the only ball label, so a second real ball in frame (rare) would be a missed label. These player boxes are themselves pseudo-labels of the same detector (tier 3).

## 5. Format and build

Exactly the v1 layout of `multiframe_v1/k2` (`docs/experiments/multiframe_dataset.md`, builder `exp/multiframe-dataset` 6dc2606, cherry-picked here as `d7f00d0`): one 5-page 8-bit LZW TIFF per image, full 1920x1080, `[gray(t-2) aligned to t, B(t), G(t), R(t), gray(t+2) aligned to t]`; neighbours that cross a cut (consecutive-frame histogram distance > 0.2) or fail alignment are replaced by gray(t); LK + RANSAC alignment from `calibration/temporal` with person boxes masked; YOLO labels in v11 class ids; `data.yaml` (`channels: 5`, `train: images/train`, `val` pointing at the v1 val images); `meta.jsonl` per image with the per-side alignment status and the pseudo-label provenance. Split name `train` only (this is training data; evaluate on `multiframe_v1/k2` val/test/heldout).

Differences from the original builder (all in `pseudo_build.py`): frames come from one sequential `ffmpeg -fps_mode passthrough` pass per clip with a 5-frame ring buffer (cv2 returned black frames on other clips of this dataset and ffmpeg's default vsync can duplicate frames; passthrough keeps the decoder frame indices); the alignment mask comes from machine person boxes (conf >= 0.3) rather than label boxes. **Check against the original on 2 images that are in both**: the three RGB pages are bit-identical (mean abs diff 0.0), the aligned neighbour pages differ by 0.04-0.74 grey levels on average (alignment mask differs), same shapes/dtype/page count.

Result: 2,115 images, 4,230 neighbour slots: **4,167 aligned (98.5%), 11 cuts, 52 alignment failures (1.2%)** (the v1 build had 0.15% failures). 7.8 GB (3.7 MB per image).

## 6. Audit

### 6a. Pseudo-label hit@10 vs human GT, by source

The pool contains almost no human GT by construction (the GT clips are v11 eval clips): only 103 tracked pool frames have one. Pool-only (n per row in brackets): B1 conf >= 0.8 38/38; conf 0.6-0.8 54/55; conf < 0.6 8/8; S1 fills 1/1 (agreement); B1 interpolated 1/1; all 102/103 = 99.0%, median error 1.3 px. That is too few for fills, so the **same pipeline's** outputs on the 18 GT-bearing harness clips plus `arsenal_example` (from `docs/experiments/sam2_machine_anchors*`, identical config thr 0.8 / spacing 10 and gate) are added. Dev clips are detector-contaminated (B1 confidence there is inflated: treat as an optimistic bound); arsenal_example and the test clips are not.

| source | pool only | test (clean) | dev (contaminated) | arsenal_example (independent) | **all three** |
|---|---|---|---|---|---|
| B1 det conf >= 0.8 | 38/38 | 69/69 | 48/48 | 21/21 | **138/138 (100%)** |
| B1 det conf 0.6-0.8 | 54/55 | 53/53 | 113/116 | 63/64 | **229/233 (98.3%)** |
| **B1 det conf < 0.6 (hard)** | 8/8 | 5/7 | 22/24 | 11/18 | **38/49 (77.6%)** |
| **S1 fill, agreement** | 1/1 | 1/2 | 2/2 | 0/0 | **3/4** |
| **S1 fill, lone pass** | 0/0 | 3/3 | 2/2 | 11/11 | **16/16** |
| B1 interpolated (not selected) | 1/1 | 2/2 | 1/1 | 0/0 | 3/3 |

(hit@10 = pseudo point within 10 px of the human GT centre; medians are 1.0-2.2 px except the S1 agreement fills that are near 3 px and the one 11.3 px near-miss.) Selected frames that sit exactly on a human GT frame: 9 (4 easy, 5 hard low-conf), all 9 within 10 px.

### 6b. Overlap of the selected frames with human GT

**52 of the 2,115 selected frames lie within +-5 frames of a human GT ball frame** (easy 34, B1 low-conf 10, S1 fills 8); 9 are on a GT frame. 91 lie within +-5 frames of a v11 train image. Both are flagged per frame (`near_human_gt_5`, `near_v11_train_5`) and can be dropped with one filter.

### 6c. Contact sheets (for eyeballing)

`contact_sheet_hard.png` (48 random hard crops: S1 fills and low-conf B1) and `contact_sheet_easy.png` (48 random easy crops): 64 px crops of frame t centred on the pseudo ball, shown 3x with the label box drawn and a caption (source and conf, or fill rule and distance to the anchor in frames). On the hard sheet roughly a quarter to a third of the boxes do not contain a clear ball (boots, crowd texture, kit), consistent with 6a; every easy crop shows a ball.

## 7. Limitations

* Hard-positive labels are not clean: the low-confidence band is about 78% correct on the audit sets (22% wrong); the 95% for fills rests on 20 GT points. Using `solver_p`/`det_conf`/`dist_anchor` from `meta.jsonl` to drop the weakest hard positives is likely worthwhile before training (e.g. conf < 0.4 or no anchor within 40 frames).
* Frames without a track are not negatives and are absent; the set contains only positives, plus the machine player/other boxes. Players are pseudo too.
* The pool is 62 clips of one match (one broadcast, one kit colour pair): scene diversity is limited; the trajectory test clips and every v11 valid/test/heldout clip are excluded, so the held-out numbers stay clean, but the v11 train clips overlap this pool (91 near-duplicate frames flagged).
* Boxes for S1 fills copy the anchor box size; boxes for B1 detections are the detector's own (which is also what the detector will be asked to reproduce: confirmation-bias risk in self-training, strongest in the low-confidence band).
* Audit rows for fills are small (20) and the dev rows are contaminated.

## 8. Cost

* **GPU** (desktop RTX 2070, SAM2.1 small crop512 fp16, S1 tracks for the whole pool): 33 min wall (11:56-12:29), 10,428 propagated frames, 1,891 s of track time (0.18 s/frame), peak 0.73 GB allocated / 0.86 GB reserved. The gpu pool was busy with another session's running training (not hung), so the job ran directly in tmux with the >= 2.5 GB VRAM-free guard (3.3 GB free at start); no other flow was touched.
* **CPU** (mini, one thread-capped process each): B1 over 62 clips and the track spec 22 s; fusion and selection about 2 min; **dataset build 988 s for 2,115 images (0.47 s/image)** including video decode (ffmpeg single pass per clip), LK alignment (34 ms per aligned pair) and LZW TIFF writing; the desktop CPU was not used because moving 7.8 GB over the 1.8 MB/s mini link would have taken over an hour.

## 9. Reproduce

```bash
OMP_NUM_THREADS=1 python experiments/pseudo_labels_v1/pseudo_prep.py --out <pp>          # exclusions, B1, anchors, track spec
# GPU (desktop): python experiments/sam2_bidir_ball/machine_tracks.py --spec <pp>/spec_pool.json --videos <dir> --out <tracks> --ckpt <dir> --model small --variant crop512
python experiments/pseudo_labels_v1/pseudo_fuse.py --prep <pp> --tracks <tracks> --out <fz> --easy 1000
python experiments/pseudo_labels_v1/pseudo_build.py --sel <fz>/selection.jsonl --out /mnt/storage/footy_data/multiframe_pseudo_v1   # stems.json next to selection.jsonl
python experiments/pseudo_labels_v1/pseudo_audit.py --prep <pp> --fuse <fz> --dataset <dataset> --out docs/experiments/pseudo_labels_v1
python experiments/pseudo_labels_v1/pseudo_audit_ext.py --prep-dir <mprep> --dec-dir <ms1> --out audit_ext.json
```
