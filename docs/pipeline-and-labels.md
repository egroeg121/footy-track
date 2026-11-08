# Best pipeline and label needs

_Research of 2026-11-08 against `main` @ 26d2506. Evidence tags: **[V]** verified against code, data or run logs; **[H]** stated only in `/mnt/storage/tools/HANDOFF.md` (2026-08-28) or a doc, not re-checked; **[I]** inference._ Context: [Architecture](architecture.md), [Datasets and labels](data.md), [TODO list](todo.md).

## Summary

- The best measured ball detector is **RT-DETR-L trained at 1920 on Roboflow v11: ball mAP50 0.665, recall 0.591** on v11-val [V]. The older ".221 / .209" quoted next to it was yolo11n on v11 (`xeval.log`); it was pasted next to the RT-DETR path by commit `24229bb` (2026-08-29). The comment in `detectors/ultralytics.py` and the docs now say so.
- The number is not a multi-match estimate: **156 of the 211 val balls (74%) come from one 10-second clip** (`arsenal_example`) [V]. The production machine labels were made at **imgsz 1280** while the checkpoint was trained at 1920 [V].
- No ball tracker exists in code. Misses are dominated by long blind spells [H], so the next gains are a tracker for precision and **diversity of labels**, not more frames from the same clips.
- Four label-pipeline bugs ([T-01 to T-04](todo.md)) will corrupt the next training export; fix them before any retrain.

## 1. Best measured pipeline

| Stage | Component | Status |
|---|---|---|
| Shot segmentation | `split_broadcast_segments.py` with yolo11n-cls `0.987.pt` | Works; 0 interior cuts in 3,175 frames [H/doc] |
| Detection (ball, players, refs) | RT-DETR-L (`rtdetr-l.pt` init), fine-tuned on Roboflow v11, 50 epochs, imgsz 1920 rect, batch 4, RTX 5090, 2026-08-28. `rtdetr_1920_v11/best.pt`, 66.8 MB, not in git | Measured [V] |
| Tiling / ROI | None. One full-frame pass. | |
| Batch inference | imgsz 1280, half, conf 0.1 or more; 162 Man City clips, 97,625 frames, 2,767,151 boxes. Script not in repo. | Produced [V] |
| Ball tracker | **None wired.** VitTrack (labeller only) stays on the ball 9/176 frames. | UNSOLVED |
| Confidence threshold | No operating point chosen in code (`get_current_best_detector(min_confidence=0.3)`). | |
| Player tracking | LapTracker over machine labels. Fragmentation measured; purity, IDF1, HOTA unmeasured. | Partial |
| Team, gap bridging, ReID | Team not computed; linker unproven; cross-clip ImageNet embeddings at chance. | Unproven |

Players are the easy part: yolo11n at 1280 reaches player mAP50 0.974 (KB:footy-ball-detection-findings). RT-DETR's per-class player mAP was never logged.

### Ball detector leaderboard

v11-val, 215 images, 211 `in_play_ball`; Ultralytics val [V: `/mnt/storage/backups/vast_run_20260827-*/exp.log`, `vast5090_final_20260828-195228/exp5090.log`, `20260826-214555/trainlogs/{xeval.log,rtdetr_ep9_val.json}`].

| Model, train imgsz, data, epochs | ball mAP50 | mAP50-95 | P | R | Weights exist? |
|---|---|---|---|---|---|
| RT-DETR-L @1280, v11, 50 ep (A4000) | 0.668 | 0.364 | 0.852 | 0.602 | No, logs only |
| **RT-DETR-L @1920 rect, v11, 50 ep (5090) = production** | **0.665** | 0.360 | 0.862 | **0.591** | Yes |
| yolo11m @1280, v11 | 0.451 | 0.241 | 0.775 | 0.327 | No |
| yolo11l @1280, v11 | 0.364 | 0.198 | 0.894 | 0.240 | No |
| RT-DETR-L @1280, **v3**, 50 ep | 0.328 | 0.166 | 0.912 | 0.244 | Yes (S3 `gpu_runs`) |
| RT-DETR-L @1280, v11, 9 ep | 0.314 | 0.176 | 0.759 | 0.283 | ? |
| yolo11n @1280, v3 / v11 trained | 0.250 / 0.221 | | 0.846 / 0.763 | 0.209 / 0.209 | |
| yolo11s v3 @640 (old default) | about 0 | | | 0.000 | in git |

The two top rows differ by noise (about 2 balls of 211). Corrections to the KB: (1) "more data did not help" holds for yolo11n only; with RT-DETR-L, v3 to v11 took mAP50 from 0.328 to 0.665 and recall from 0.244 to 0.59. (2) The capacity confound is tested: yolo11m/l lose clearly to RT-DETR-L. (3) v11 train has 1,010 balls, not 826.

### Held-out recall at the production setting

Machine labels (imgsz 1280) against tier 1/2 GT sidecars, Man City clips; a hit is a machine box within max(12 px, box size) of the GT centre [V, my matching rule]:

| GT clips | n balls | r@0.1 | r@0.3 | r@0.5 | r@0.7 |
|---|---|---|---|---|---|
| Held out of v11 train | 74 | 0.69 | 0.62 | 0.60 | 0.51 |
| Overlap v11 train | 330 | 0.96 | 0.92 | 0.88 | 0.71 |

Held-out recall at 1280 is about 0.6, consistent with 0.59 at 1920, but n=74 is dominated by one clip, and the overlap row shows memorisation.

### Ball Check precision by confidence band

257 decided verdicts on machine candidates (`ball`, `corrected` count as hits) [V]. The sampler oversamples near the boundary, so these are per band, not global precision.

| conf | 0.1-0.2 | 0.2-0.3 | 0.3-0.4 | 0.4-0.5 | 0.5-0.6 | 0.6-0.7 | 0.7-0.8 | 0.8-0.9 |
|---|---|---|---|---|---|---|---|---|
| precision | 0.04 | 0.29 | 0.23 | 0.36 | 0.42 | 0.79 | 0.96 | 1.00 |
| n | 52 | 14 | 26 | 42 | 48 | 38 | 26 | 11 |

Boxes at 0.7 or more are near-certain, 0.6-0.7 mostly right, below 0.6 mostly wrong. A low-confidence second pass must therefore rely on motion gating. Human-checked vittrack ball boxes are about 97% right (81 verdicts: 71 ball, 6 corrected, 2 not_ball, 2 unsure).

### Is the benchmark trustworthy?

Partly. The instance count is reasonable and val clips are disjoint from train at clip level [V], but 74% of val balls are one clip, so HANDOFF's "independent" confirmation (`arsenal_example` 0.585 against val 0.591) is the same clip. The **v11 test split (268 images, 243 balls) has never been evaluated**. Cross-match generalisation is unmeasured. Unverified: whether `arsenal_example` overlaps a Man City training clip in time.

## 2. Target pipeline

1. Per-frame RT-DETR-L at 1280 (keep it).
2. An inference-only ball association layer over the stored conf 0.1 or more detections: ByteTrack/BoT-SORT-style low-confidence second pass, Kalman with high process noise and camera-motion compensation, gap interpolation.
3. Later, a temporal detector: prior-conditioned RT-DETR (CenterTrack, bead ft-1zq), a heatmap over a frame stack (TrackNetV3, WASB), or YOLOFT-style correlation. The segmenter already gives cut-free clips.

What the data says [H + I]: misses are dominated by long blind spells (one 51-frame gap holds 70% of `arsenal_example` misses; only 7% fall in 1-2 frame gaps), so a t-1 prior is estimated to buy about 0.585 to 0.62 recall. The larger lever is diversity; a tracker mainly buys precision. The CenterTrack run reached epoch 11/80 before the desktop reset and **no result is recorded**.

Gaps between today and the target: no committed ball eval; no ball tracker in code; the producers are outside git (training command, batch inference, candidate filter, the temporal package, which exists only in `/mnt/storage/backups/temporal_work.tgz`, and the ft-1zq code, only on desktop-ubuntu); the held-out dense GT is too narrow to see a 0.03 gain; the label-plumbing seams corrupt exports; the 1920-train versus 1280-serve mismatch is unmeasured.

## 3. Label needs and gaps

| Need | Have | Does the gap matter? |
|---|---|---|
| Ball training boxes (tier 1/2) | v11 train 1,010; sidecars 692 T1 + 825 T2 over 45 clips; T3 machine 352k | **Yes, for diversity not count.** 520 of 1,035 v11 train images are one clip (`astonvilla_seg080`) and 212 another (`bournemouth_1st_seg070`). v3 to v11 doubled RT-DETR recall; more frames from the same long clips will not help. |
| Player and referee boxes | v11 15.5k player, 1.7k referee; sidecars 18.7k, 2.1k | No. Players are about 0.97 mAP50. |
| Hard negatives (`not_ball` verdicts, `no_ball` frames) | 144 `not_ball` verdicts; 68 `no_ball`, 138 `not_broadcast` markers | Moderate. Unused today; below 0.6 confidence precision is 0.42 or less. |
| Single-frame eval, 2 or more matches | v11 val 211 (74% one clip); test 243 unused | **Yes, highest.** Every decision depends on it. |
| Dense ball-tracker eval (contiguous frames, explicit `no_ball`) | `arsenal_example` (250 frames, 176 balls, video only as the example mp4), `mancity_seg031` (53). Other long clips are in train. | **Yes.** Needs about 5-10 held-out clips of 250+ frames from 2 or more matches, 1.5-3k balls [I]. |
| Temporal training | About 1,450-1,520 ball frames; 942 in runs of 3 or more (KB) | No new label type; only after the eval exists. |
| Identity (purity, jersey, merges, team) | 9 reviews, 68 proposals, 0 team labels | Defer; not needed for the ball. |

Do not bother: importing external single-frame ball sets at volume, more Man City or Aston Villa seg080-style frames, player boxes, identity labels for the ball pipeline.

## 4. How labels are created today

| Route | Tool | Tier | Path to training | Problems |
|---|---|---|---|---|
| A. Hand labels | `/labeller` draws boxes into `ball_gt_marks/<stem>.jsonl` | T1 (692 ball) | `ingest_gt`, then `export_training_dataset`, then Roboflow or `--local-dataset` | S1 strips `human_checked`; S6 concurrent writers clobber. |
| B. VitTrack propagation | Labeller Run (`VitTrackVideoLabeller`) | T3 when written | as A | Useless for the ball (5% on-ball; Run halts at frame 1). It became T2 only via a one-off backfill on 2026-08-23 (15,386 lines). **No code in the repo writes `human_checked`** [V grep]. |
| C. `/object_review` | Grid crop review | correct gives T1 | as A | S4 ghost rows on re-ingest; OPEN-3. |
| D. `/ball_check` | Phone yes/no on machine candidates or sidecar boxes; dragged box written to the sidecar | `corrected` is T1; `ball` and `not_ball` give nothing | corrected boxes only; **verdicts reach nothing** | S2, S3, S5. Measured 364 verdicts in about 20 active minutes, about 3.3 s per card [V]. |
| E. Roboflow hand labels | Roboflow web annotation | treated as `hand_label` | **The route the best model actually used**: download v11, train directly | v11 was built 2026-08-04 from a store export; the uploader exists only on unmerged branches `feat/roboflow-dataset-export`, `feat/roboflow-upload-script`. |
| F. Machine labels | RT-DETR batch inference (script not in repo) | T3 | Only the Ball Check pool and LapTracker input | A `ball` verdict, the cheapest human signal, is thrown away. |
| G. SAM3 text prompts | `scripts/detect_objects.py --store_path` | T3 | store, then export if selected | No metric; effectively unused. |
| H. Sparse centre GT | `scripts/label_ball_centers.py` | | dropped by `ingest_gt` (`:184`) | Eval only; `mancity_seg104` has no mp4. |
| I. Identity review | `/identity`, `/identity/merge` | T1 | none | No consumer. |

The newest feature store (2026-08-04) predates the `human_checked` ingest fix, the Ball Check corrections and six new sidecars, so store, export and Roboflow is stale end to end ([T-07](todo.md)).

## 5. Recommendation, in order

1. **Fix the record.** Done on this branch: the code comment and docs now cite .665 / .591 and the 74% caveat. The KB findings page still needs updating, with George's approval ([T-15](todo.md)).
2. **One committed ball eval** over the v11 test split plus held-out dense clips (`arsenal_example` after checking it is not a sub-span of a training clip, and `mancity_seg031`), at 1280 and 1920, reporting instance counts and a per-match breakdown ([T-10](todo.md)). The 1280 half needs no GPU for the sidecar clips.
3. **ByteTrack/Kalman ball association over existing machine labels**, with thresholds from the precision bands (0.7 or more near-certain; 0.6-0.7 about 0.79; below 0.6 at most 0.42). Run `analyse_ball_tracker.py` first to split detector misses from tracker misses ([T-20](todo.md)).
4. **Diversity labelling on new matches** (Aston Villa, Bournemouth 2nd half, Norwich) with `/ball_check`, holding one full match out for eval; at about 3.3 s per card, 1,000 cards is about an hour ([T-13](todo.md), [T-21](todo.md)). Look at the 51-frame `arsenal_example` gap first.
5. **Before any retrain through the store**, fix S1 to S4 ([T-01 to T-04](todo.md)) and make the export default to tier 1/2 ([T-22](todo.md)).
6. Commit the missing producers and recover the ft-1zq code from desktop-ubuntu ([T-09](todo.md), [T-23](todo.md)); retrain RT-DETR-L at 1280 on v11 plus new-match labels and compare on the step 2 eval.
7. Only then: finish ft-1zq on a rented GPU, or a heatmap/YOLOFT detector if the tracker and diversity plateau.

Do not bother with: tuning VitTrack thresholds or SAM3 (measured dead ends), 1920 resolution or bigger GPUs (1920 is about 1280), scaling YOLO (yolo11m/l lost), RF-DETR (caps below 1280).

## 6. Uncertainties

- Whether `arsenal_example` overlaps training footage.
- The CenterTrack result is not recorded; the HANDOFF temporal miss analysis was not reproduced.
- The recall matching rule (centre within max(12 px, box size)).
- The held-out GT mixes tier 1 and 2, and tier 2 is vittrack-derived (about 97% precise).
- WSL and desktop data copies were not inspected.
- Per-class player mAP for RT-DETR was never logged.
