# Promptable memory-based VOS/tracking (SAM2 / SAM3 family, trackers-on-top, Cutie/XMem/DEVA) for an ~11 px football, offline and bidirectional

Project context relied on: KB page `~/knowledge/wiki/footy-ball-detection-findings.md` (found via `mcp__knowledge__search "footy ball detection"`). Facts from it used here: ball is ~11 px wide at 1280 (median 10.7-11.5 px, ~25% under 10 px); at 640 px the ball is never detected, at 1280 it is, so the ball does not survive downsampling; ~1,453 ball frames labelled with neighbour frames recoverable by filename; SOT/Siamese trackers are "unvalidated at ball scale" and "fail silently" when the ball exits a 4-5x-target search window; the temporal-heatmap family (TrackNetV3, WASB, YOLOFT) is what the field uses. The 8 GB-OOM note on SAM3 and the reverse-propagation note come from the task brief, not re-verified in the repo.

Researcher caveat: WebFetch summarises pages with a small model. Where a page did not state something, I say so rather than fill it in. Items marked UNVERIFIED are from my own background knowledge or are inferences.

## 1. SAM2 / SAM2.1 / SAM3 / SAM 3.1 / EfficientTAM / EdgeTAM: resolution, small objects, memory under occlusion, bidirectional use, VRAM

### Takeaway
Every model in this family runs its backbone at a fixed square ~1024 px (SAM3: 1008 px), with ~stride-16 memory features, so an 11 px ball at 1280 width becomes roughly one memory-feature cell. No benchmark I found measures SAM2/SAM3 on balls this small, and the one paper that benchmarks SAM2-family trackers on small fast objects (FMOX) finds them weak (mean mIoU 0.44-0.51). Reverse propagation is supported in SAM2 and SAM3, so the bidirectional mechanics are available. As a ball-propagation engine for a ~11 px target the family is unproven, and the base use is likely poor without crops and fine-tuning.

### Cited Findings
**Model sizes and speed (SAM2.1, A100, torch 2.5.1)**
- SAM2.1 checkpoints: tiny 38.9M params / 91.2 FPS; small 46M / 84.8 FPS; base+ 80.8M / 64.1 FPS; large 224.4M / 39.5 FPS — [facebookresearch/sam2](https://github.com/facebookresearch/sam2)
- SAM2 released 29 Jul 2024; SAM2.1 + training code released late Sep 2024; `vos_optimized=True` (torch.compile of whole model) added 11 Dec 2024 — [facebookresearch/sam2](https://github.com/facebookresearch/sam2)
- Config `sam2.1_hiera_t.yaml`: `image_size: 1024`, `num_maskmem: 7`, memory-attention RoPE `feat_sizes: [64, 64]`, `pred_obj_scores: true` (occlusion/no-object head), `fixed_no_obj_ptr: true`, `use_obj_ptrs_in_encoder: true`, memory encoder `out_dim: 64`, Hiera FPN neck — [sam2.1_hiera_t.yaml](https://raw.githubusercontent.com/facebookresearch/sam2/main/sam2/configs/sam2.1/sam2.1_hiera_t.yaml)
- 64x64 memory feature grid over a 1024 input implies stride 16 (my arithmetic from the two config values; see Inferences).

**Reverse / offline use and VRAM knobs (SAM2)**
- `propagate_in_video(self, inference_state, start_frame_idx=None, max_frame_num_to_track=None, reverse=False)`; `init_state(offload_video_to_cpu=False, offload_state_to_cpu=False, async_loading_frames=False)`. Code comments: `offload_video_to_cpu` "saves the GPU memory with only a very small overhead"; `offload_state_to_cpu` "saves the GPU memory at the cost of a lower tracking fps (e.g. ... 768x768 model, fps dropped from 27 to 24 when tracking one object)" — [sam2_video_predictor.py](https://raw.githubusercontent.com/facebookresearch/sam2/main/sam2/sam2_video_predictor.py)
- `SAM2VideoPredictor` (SAM2.1) supports independent per-object inference and adding objects after tracking starts — [facebookresearch/sam2](https://github.com/facebookresearch/sam2)
- SAM2 paper says that when the model loses an object or errs, refinement clicks on additional frames can recover the correct prediction in most cases (secondary summary via search) — [SAM 2 paper](https://arxiv.org/pdf/2408.00714)
- SAM2 paper notes SA-V is challenging for small objects and parts (secondary summary via search) — [SAM 2 paper](https://arxiv.org/pdf/2408.00714)

**SAM3 / SAM 3.1**
- SAM 3: 848M params; detector (DETR-based, conditioned on text / geometry / image exemplars) and tracker sharing one vision encoder; tracker "inherits the SAM 2 transformer encoder-decoder architecture, supporting video segmentation and interactive refinement". Requires Python 3.12+, PyTorch 2.7+, CUDA 12.6+. Checkpoints are gated on Hugging Face — [facebookresearch/sam3](https://github.com/facebookresearch/sam3)
- SAM 3 cgF1 vs human: 55.7 vs 74.0 on SA-Co/Gold, 30.3 vs 53.1 on SA-V — [facebookresearch/sam3](https://github.com/facebookresearch/sam3)
- Hugging Face docs: "The model is meant to be used at 1008px resolution" and "Custom resolutions may degrade accuracy" (example drops to 560 for lower memory). The tracker is a `Sam2VideoConfig` SAM2-style tracker; tracker low-res mask size 288 — [HF Sam3Video docs](https://huggingface.co/docs/transformers/main/en/model_doc/sam3_video)
- HF `Sam3VideoModel.propagate_in_video_iterator(..., reverse=False)` and `forward(..., reverse=...)` both expose reverse propagation. Streaming mode disables "hotstart" heuristics that need future frames; pre-loaded (offline) video gives best results — [HF Sam3Video docs](https://huggingface.co/docs/transformers/main/en/model_doc/sam3_video)
- SAM3 video pipeline defaults that matter for tiny objects: `score_threshold_detection=0.5`, `new_det_thresh=0.7`, `fill_hole_area=16` ("minimum area (in pixels) for filling holes in masks and removing small sprinkles"), `hotstart_delay=15`, `hotstart_unmatch_thresh=8`, `init/max_trk_keep_alive=30`, `recondition_every_nth_frame=16`, `high_conf_thresh=0.8`. `video_storage_device` and `inference_state_device` default to CPU — [HF Sam3Video docs](https://huggingface.co/docs/transformers/main/en/model_doc/sam3_video)
- SAM 3.1 ("Object Multiplex") released 27 Mar 2026; shared-memory joint multi-object tracking, up to 16 objects per forward pass; checkpoints `facebook/sam3.1` on Hugging Face — [facebookresearch/sam3](https://github.com/facebookresearch/sam3), [Labellerr](https://www.labellerr.com/blog/sam-3-vs-sam-3-1-performance-comparison/)
- SAM 3.1 vs SAM 3 (secondary source, not Meta's own): ~16 fps to ~32 fps at medium object counts on one H100, ~7x at 128 objects; +2.0 J&F on MOSEv2; VOS improved on 6 of 7 benchmarks; backbone unchanged (Perception Encoder + DETR-style head); no dedicated paper, technical details in Appendix H of the SAM 3 paper; no VRAM numbers and no small-object remarks — [Labellerr](https://www.labellerr.com/blog/sam-3-vs-sam-3-1-performance-comparison/)
- SAM 3.1 is aimed at multi-object throughput. For a single ball (one tracked object) it is not shown to help; the source gives no single-object benchmark — [Labellerr](https://www.labellerr.com/blog/sam-3-vs-sam-3-1-performance-comparison/)

**EfficientTAM / EdgeTAM**
- EfficientTAM: lightweight ViT image encoder + efficient memory cross-attention, trained on SA-1B and SA-V, Apache-2.0, ">10 FPS on iPhone 15" — [yformer/EfficientTAM](https://github.com/yformer/EfficientTAM)
- EdgeTAM: 16 FPS on iPhone 15 Pro Max, 150.9 FPS on A100 (compiled), SA-V J&F 72.3 val / 71.7 test vs SAM2.1-B+ 76.8 / 77.0; Apache-2.0 — [facebookresearch/EdgeTAM](https://github.com/facebookresearch/EdgeTAM)
- On small fast objects (FMOX, 46 sequences) EfficientTAM was the worst of four: mIoU mean 0.438 (SAM2 0.461, SAMURAI 0.488, DAM4SAM 0.505); authors: "frequent missed detections", and it failed to initialise on a motion-blurred sequence. It was fastest (e.g. FMOv2 410 s vs SAM2 891 s) — [Aktas et al., arXiv 2512.09633](https://arxiv.org/html/2512.09633v1)

**Small-object evidence**
- FMOX benchmark of SAM2, EfficientTAM, DAM4SAM, SAMURAI (all single first-frame template): authors say only FMOv2 and TbD contain extremely tiny to small objects and are harder; "initializing trackers with highly motion-blurred frames can adversely affect their performance". No size-stratified numbers are given — [arXiv 2512.09633](https://arxiv.org/html/2512.09633v1)
- DAM4SAM CVPR 2025 supplement states SAM2.1 improves on SAM2 for small and visually similar objects through extra training augmentations (secondary summary via search) — [DAM4SAM supplement](https://openaccess.thecvf.com/content/CVPR2025/supplemental/Videnovic_A_Distractor-Aware_Memory_CVPR_2025_supplemental.pdf)
- Another paper characterises SAM2 as prone to long-term failure "due to its reliance on visual similarity and limited memory capacity, especially under occlusion, fast motion, or the presence of distractors" (search-result summary) — [arXiv 2507.07603](https://arxiv.org/abs/2507.07603)

### Inferences
- Effective ball size inside SAM2: if SAM2 squashes a 1920x1080 frame to 1024x1024 (UNVERIFIED for this repo; I believe that is the default), an 11 px (at 1280) ball is ~16.5 px at 1920 and becomes ~9 px wide x ~16 px tall at 1024x1024. With stride-16 memory features (64x64 grid from 1024 px) that is about half a cell wide. Memory features therefore barely encode the ball. SAM3 at 1008 is the same situation. Running SAM3 at a reduced 560 px (the memory-saving option in the HF docs) would make the ball ~sub-pixel in feature space, matching the repo's finding that downscaling loses the ball.
- Crops fix this in principle: a 1024x1024 native-resolution crop around the predicted ball path would put the ball at ~16 px (1 feature cell) without any resize. Alternatively upsample a ~512 px crop 2x. This keeps VRAM at the standard 1024 footprint instead of whole-frame cost. Needs a moving crop (camera pans, fast ball) and therefore a crop-follow loop, which breaks "pure offline propagate_in_video" into a custom driver. No source I found has tried this on balls; it is a hypothesis.
- SAM2 tiny/small are small enough (39-46M params) that 8 GB local is not VRAM-limited for inference at 1024 on short clips; the stated "SAM3 borderline OOM on 8 GB" is consistent with 848M params plus a 1008 px backbone. SAM2 memory grows with clip length only through stored memory features; `offload_video_to_cpu` and `offload_state_to_cpu` exist for long clips (small quoted fps penalty). No quantified per-frame VRAM figure was found.
- Reverse propagation exists but is the same model run backward; "bidirectional consensus" (combine forward and reverse masks) is not built in. For SAM2 you would call `propagate_in_video` twice (reverse=False/True) from a prompted frame and fuse yourself.
- EfficientTAM/EdgeTAM trade accuracy for speed; given the FMOX result and that the ball is already at the resolution floor, small/efficient models are the wrong direction for accuracy. They are only useful if throughput matters.
- SAM3's default `fill_hole_area=16` and detection thresholds are tuned for normal-size objects; a ~100-150 px² ball mask is above 16 px² but its detector presence score is untested.

### Gaps
- No size-stratified accuracy for SAM2 / SAM2.1 / SAM3 on objects under ~20 px. The FMOX paper has object-size categories only for dataset description; I did not find a paper that reports error vs object size for SAM2-family trackers.
- No SAM3 tiny-object results; SAM3 tracker small-object behaviour is unverified.
- No quantified VRAM per frame or per clip length for SAM2 tiny/small at 1024, nor SAM3 at 1008 on 8 GB. The task brief's OOM claim is the only data point.
- Exact preprocessing (square resize vs aspect-preserving) in SAM2 was not confirmed from a fetched file.
- SAM2 memory-bank behaviour on occlusion/reappearance was only partially verified: the config has an object-score (no-object) head and 7 memory frames, but I did not read the paper text on how reappearance is handled (the PDF exceeded the fetch size limit).
- SAM 3.1 and EdgeTAM/EfficientTAM param counts and input resolution: not stated on the pages fetched.
- Mid-2026 "SAM2.1++" and other efficient SAM3 variants: not found in the searches I ran; treat as unknown rather than non-existent.

## 2. Tracker-on-top variants (SAMURAI, DAM4SAM, SAM2Long, hierarchical-motion variants, SAMOSA): do they help small fast objects and distractors

### Takeaway
All are training-free layers on SAM2.1 that improve occlusion/distractor handling on general benchmarks; on the one small-fast-object benchmark found (FMOX) DAM4SAM was best on average (mIoU 0.505) but per-dataset rankings flip, and none addresses the ~stride-16 resolution limit. SAMURAI, DAM4SAM have Apache-2.0 licences; SAM2Long code is CC BY-NC 4.0.

### Cited Findings
- SAMURAI: Kalman-filter motion-aware memory on SAM 2.1, zero-shot with SAM 2.1 weights, Apache-2.0, runs on stored videos/frame folders from a first-frame bbox (x,y,w,h), no streaming support; page gives no numbers and says nothing about small objects or reverse tracking — [yangchris11/samurai](https://github.com/yangchris11/samurai)
- DAM4SAM (CVPR 2025; IJCV 2026): plug-in distractor-aware memory on SAM2.1, Apache-2.0. DiDi (Quality/Accuracy/Robustness): DAM4SAM 0.694/0.727/0.944, SAMURAI 0.680/0.722/0.930, SAM2.1 baseline 0.649/0.720/0.887. Claims SOTA on six benchmarks without extra training; evaluated on DiDi, VOT2020/2022, LaSOT, LaSOText, GOT-10k — [jovanavidenovic/DAM4SAM](https://github.com/jovanavidenovic/DAM4SAM)
- SAM2Long (ICCV 2025): training-free memory tree keeping multiple hypotheses and pruning, ~+3 J&F average over 24 comparisons on SA-V and LVOS. Examples: SAM2 Tiny SA-V val 73.5 to 77.0; SAM2.1 Large SA-V val 78.6 to 81.1. Licence: mostly CC BY-NC 4.0 (non-commercial); original SAM2 checkpoints/training code Apache-2.0. No overhead numbers reported — [Mark12Ding/SAM2Long](https://github.com/Mark12Ding/SAM2Long)
- Hierarchical motion + split long/short-term memory SAM2 variant: training-free, +9.6% / +7.2% relative AUC on LaSOT / LaSOText with the large model, "larger relative gains on smaller models"; no small-object results in the abstract — [arXiv 2507.07603](https://arxiv.org/abs/2507.07603)
- SAMOSA (arXiv 2605.22538, May 2026): adds failure detection and recovery to SAM2 with motion, geometry and semantic cues; reports large gains on Anti-UAV datasets (tiny targets in clutter); code at DurYi/SAMOSA. I did not read its numbers or licence — [arXiv 2605.22538](https://arxiv.org/abs/2605.22538)
- FMOX results by dataset (best to worst): Falling Object: DAM4SAM > SAM2 > SAMURAI > EfficientTAM; TbD-3D: SAM2 > DAM4SAM > SAMURAI > EfficientTAM; FMOv2: SAMURAI > DAM4SAM > SAM2 > EfficientTAM; TbD: DAM4SAM > SAMURAI > EfficientTAM > SAM2. Overall mean mIoU: DAM4SAM 0.505, SAMURAI 0.488, SAM2 0.461, EfficientTAM 0.438. Runtime: DAM4SAM 1316.9 s vs SAM2 890.8 s on FMOv2 but 261 s vs 394 s on TbD; SAMURAI 516 s vs SAM2 891 s on FMOv2 — [arXiv 2512.09633](https://arxiv.org/html/2512.09633v1)

### Inferences
- The gains from these layers (about +2-5 points on general benchmarks, +0.03-0.04 mIoU on FMOX) are small next to the resolution problem; they re-rank hypotheses using masks the model has already produced, so they cannot add detail the ball does not have in the 1024 px feature map.
- SAMURAI's Kalman motion prior assumes smooth motion in image coordinates; a panning broadcast camera and ball kicks violate that (same failure the KB documents for SORT-style Kalman defaults). Motion-prior layers would need camera-motion compensation. UNVERIFIED as a failure mode; no source tested it.
- DAM4SAM's distractor memory is conceptually the right fit for white boots / line markings, but there is no evidence it works when the target is 1 feature cell wide.
- SAM2Long is non-commercial for its own code; if the project is commercial, avoid or reimplement.

### Gaps
- No tracker-on-top paper reports results on objects below ~16 px or on soccer/ball sequences.
- Whether SAMURAI/DAM4SAM support `reverse=True` is not stated in their READMEs (they take a first-frame box and run forward). Running them on a reversed frame sequence would be the workaround (my suggestion, not tested).
- SAMOSA's tiny-target numbers and licence not read. SAM2.1++ not found.
- The DAM4SAM runtime comparison is mixed (slower than SAM2 on FMOv2, faster on TbD); I do not have an explanation.

## 3. Cutie, XMem, DEVA: tiny-object performance, bidirectional propagation

### Takeaway
Cutie and XMem are MIT-licensed memory-based VOS models that default to low internal resolution (Cutie example: shorter edge 480) and are not designed for tiny targets; DEVA has genuine bidirectional/consensus propagation but is CC BY-NC-SA 4.0 and its own docs note it can capture smaller objects only with detector help. I found no tiny-object benchmark for any of them.

### Cited Findings
- Cutie: MIT licence (README links MIT LICENSE; `gui/ritm` redistributes RITM code under RITM's licence); follow-up to XMem; example `max_internal_size: 480` (shorter edge) and `-1` for original size; GUI has permanent-memory mode from XMem++ — [hkchengrex/Cutie](https://github.com/hkchengrex/Cutie)
- XMem LICENSE is MIT, Copyright 2024 Ho Kei Cheng — [XMem LICENSE](https://raw.githubusercontent.com/hkchengrex/XMem/main/LICENSE)
- DEVA: "class/task-agnostic bi-directional temporal propagation" plus (semi-)online fusion of segmentation hypotheses from different frames via an integer program (Gurobi recommended, PuLP fallback); pairs with a per-frame image segmenter (SAM, Grounded SAM); licence CC BY-NC-SA 4.0 (non-commercial, share-alike) — [Tracking-Anything-with-DEVA](https://github.com/hkchengrex/Tracking-Anything-with-DEVA), [DEVA LICENSE.md](https://raw.githubusercontent.com/hkchengrex/Tracking-Anything-with-DEVA/main/LICENSE.md)
- DEVA limitations stated by the authors: propagation amplifies positive detections (use a higher threshold / lower false-positive rate detector); memory fills with false positives when objects frequently enter/leave; underperforms end-to-end training when you have enough target data. A Sept 2023 update raised SAM query points to 64/side "to capture smaller objects" — [Tracking-Anything-with-DEVA](https://github.com/hkchengrex/Tracking-Anything-with-DEVA)

### Inferences
- DEVA's design (a task-specific per-frame segmenter/detector + a general propagation model, fused bidirectionally) is structurally close to the project goal (YOLO/RT-DETR ball detections propagated and reconciled in both directions), but its propagation model works at VOS resolution (likely ~480 px short edge by default, as Cutie), and it is non-commercial.
- XMem/Cutie are lightweight and trainable from DAVIS/YouTube-VOS-style masks; they could run on high-res crops but there is no evidence for tiny objects. UNVERIFIED.
- DEVA's consensus idea can be replicated with SAM2 forward+reverse passes plus the detector without taking the DEVA licence.

### Gaps
- No tiny-object results for Cutie, XMem, DEVA found. Cutie VRAM and DEVA's exact training/propagation resolutions not stated on the pages fetched.
- Licence of Cutie's checkpoints and RITM portion not checked beyond the MIT note.

## 4. Benchmarks and papers on small-object / sports-ball VOS or SOT with SAM-family models (with reported numbers)

### Takeaway
There is almost no direct evidence. The FMOX paper (SAM2-family trackers on fast-moving small objects) is the closest, with mean mIoU 0.44-0.51. Practical ball-tracking blog posts use SAM2 as a mask propagator seeded by a detector, and dedicated detectors (TrackNet-family, WASB, BlurBall) remain the established route for tiny balls.

### Cited Findings
- FMOX (46 sequences across Falling Object, TbD-3D, FMOv2, TbD): mIoU mean / median: SAM2 0.461 / 0.591, EfficientTAM 0.438 / 0.548, DAM4SAM 0.505 / 0.605, SAMURAI 0.488 / 0.596; mDice means 0.545 / 0.520 / 0.600 / 0.579. Min per tracker is ~0 (total failures). Authors: only FMOv2 and TbD hold extremely tiny to small objects and are harder — [arXiv 2512.09633](https://arxiv.org/html/2512.09633v1) (Aktas, Markham, McDonald, Dahyot; AICS 2025)
- Roboflow soccer-ball tutorial uses image slicing to improve detection of small objects and a tracker that handles missed/spurious detections, i.e. detector+tracker rather than a VOS model (search-result summary) — [Roboflow blog](https://blog.roboflow.com/tracking-ball-sports-computer-vision/)
- Sieve blog "Building a robust ball tracking system for sports with SAM 2": seeds SAM2 with a ball bounding box, often from a detector, then segments through the clip; mentions handling scene changes and false positives. I could not fetch the page (DNS failure), so this is from a search-result snippet only — [Sieve](https://alpha.sievedata.com/blog/ball-tracking-with-sam2)
- Labellerr markets SAM2 for ball/player annotation ("10x faster"): vendor claim, not a benchmark — [Labellerr docs](https://docs.labellerr.com/features/sam-2-object-tracking)
- Anti-UAV (tiny targets in clutter): SAMOSA reports "substantial gains" for a SAM2-based tracker (no numbers read) — [arXiv 2605.22538](https://arxiv.org/abs/2605.22538). A separate infrared tiny-target paper argues trackers that rely on cropped template regions have limited motion modelling for tiny targets — [arXiv 2505.04917](https://ar5iv.labs.arxiv.org/html/2505.04917). KB (prior in-repo research) puts the closest anti-UAV proxy (target ~0.03% of frame) at ~36% accuracy for SOT trackers.
- Dedicated ball methods referenced for table tennis/racket sports: TrackNet, WASB, BlurBall; RacketVision dataset (1,672 clips, badminton/tennis/table tennis) — [RacketVision arXiv 2511.17045](https://arxiv.org/html/2511.17045v3)

### Inferences
- A VOS model seeded by a detector box is a plausible "gap filler" between confident detections, but the only quantitative evidence on small fast targets (FMOX) shows high failure rates and strong sensitivity to the init frame (motion blur at the prompt frame hurts). For football, prompt only on sharp, human-verified frames.
- The pattern "re-prompt every N frames with a detector hit" is the routine fix when SAM2 drifts (SAM2 paper refinement-clicks point; SAM3 default `recondition_every_nth_frame=16` is the same idea built in).

### Gaps
- No SAM2/SAM3 numbers on soccer/football balls, tennis, or table tennis from any paper. No LaSOT small-subset breakdown found. No anti-UAV numbers for SAM2 variants read.
- Sieve and Roboflow blog contents were not read in full.

## 5. Fine-tuning: official code, label requirements, crops/tiles, LoRA

### Takeaway
SAM2.1 has official training/fine-tuning code (Apache-2.0) with a MOSE fine-tune config and SA-V/DAVIS-style data; it assumes 8x A100-80GB for the reference runs but MedSAM2 shows people fine-tune it from SAM2.1-tiny on single nodes. SAM3 ships training code, but the README documents detection-benchmark fine-tuning (Roboflow100-VL, ODinW) and does not say the video tracker is covered. Sparse box labels are not directly the training format (DAVIS-style masks); you would need to convert boxes to masks or modify the loss. LoRA was not found in official or MedSAM2 docs.

### Cited Findings
- SAM2 training README: start from `sam2.1_hiera_b+_MOSE_finetune.yaml`; DAVIS-style `JPEGImages` + `Annotations` folders, SA-V via `JSONRawDataset` (frames extracted to JPEG, `ann_every: 4`), SA-1B images via `SA1BRawDataset`; `TorchTrainMixedDataset` mixes image and video; 1 frame per image sample and 8 per video sample; reference example uses 8 GPUs, assumes A100 80 GB; `SAM2Train` accepts parameters for simulated prompts (iterative point sampling). Page does not state which components can be frozen or the prompt types — [sam2 training README](https://github.com/facebookresearch/sam2/blob/main/training/README.md)
- SAM2 repo: training code is Apache-2.0 and released with SAM2.1 in `training/` for custom image/video/mixed data — [facebookresearch/sam2](https://github.com/facebookresearch/sam2)
- MedSAM2: fine-tunes from `sam2.1_hiera_tiny.pt` with custom YAMLs (e.g. `sam2.1_hiera_tiny512_FLARE_RECIST.yaml`, note a 512 input variant), batch size set by GPU memory, single-node and Slurm scripts, Apache-2.0; data counts, frozen components, LoRA, bidirectional use not stated on the README — [bowang-lab/MedSAM2](https://github.com/bowang-lab/MedSAM2)
- SAM3 training: `pip install -e ".[train]"`, Hydra YAML configs, `sam3/train/train.py`, examples on 1, 4, and 8 GPUs (2 nodes), local or SLURM; README covers Roboflow100-VL and ODinW; does not state whether detector, tracker or both are fine-tuned; no VRAM figures — [SAM3 README_TRAIN](https://github.com/facebookresearch/sam3/blob/main/README_TRAIN.md)
- SAMURAI, DAM4SAM, SAM2Long and the hierarchical-motion variant are all zero-shot/training-free — [yangchris11/samurai](https://github.com/yangchris11/samurai), [DAM4SAM](https://github.com/jovanavidenovic/DAM4SAM), [SAM2Long](https://github.com/Mark12Ding/SAM2Long), [arXiv 2507.07603](https://arxiv.org/abs/2507.07603)
- HF SAM3: running at a non-native size (e.g. 560) is supported via `config.image_size` but may degrade accuracy; 1008 is the intended size — [HF Sam3Video docs](https://huggingface.co/docs/transformers/main/en/model_doc/sam3_video)

### Inferences
- With ~1,500 sparse human ball boxes (about 1,453 frames with recoverable neighbour frames per the KB), the realistic fine-tune is SAM2.1 tiny/small on native-resolution crops (e.g. 512-1024 px windows around the ball) with the ball at 16+ px, using pseudo-masks: a filled ellipse/box mask of the labelled ball and short clips of 4-8 consecutive frames. MedSAM2's 512-input tiny variant is a precedent for changing input size, but whether 512 or crop-based fine-tuning helps a tiny ball is UNVERIFIED.
- Because SAM2 trains on video with masks, labelled single frames (not clips) cannot supervise the memory pathway. The memory encoder and memory attention only learn from clips with consecutive labelled or pseudo-labelled frames. KB says 942 of the ball frames already sit in consecutive runs of 3+, which is the minimum usable subset.
- 8 GB local is unlikely to train SAM2.1 at 1024 with 8-frame clips beyond tiny/small and tiny batch; a rented 12-16 GB GPU is more realistic (UNVERIFIED; no source gives VRAM for fine-tuning). The official reference is 8x A100-80GB.
- SAM3 fine-tuning for the video tracker is undocumented; since the tracker is SAM2-style, fine-tuning the SAM2.1 tracker directly is the better-documented path.
- Whether crops stay on-distribution for the memory: SAM2 was trained on SA-V full-frame videos (not stated in fetched pages beyond EfficientTAM's note that it trains on SA-1B and SA-V), so a crop-follow wrapper will present the model with camera-motion-like jumps whenever the crop re-centres.

### Gaps
- No documented LoRA recipe for SAM2/SAM3 video from the pages fetched (searches not run specifically for LoRA; absence of mention only).
- Minimum label count needed to adapt SAM2 to a new tiny-object domain: not found.
- SAM2 training on box-only labels: not stated; the README does not say masks are required.
- Which SAM2 submodules can be frozen: not stated in the README.
- No evidence that SAM3's tracker can be fine-tuned with the released code.

## 6. Licences (flag non-commercial)

### Takeaway
SAM2 / SAM2.1, SAMURAI, DAM4SAM, EfficientTAM, EdgeTAM, MedSAM2, Cutie and XMem are permissive (Apache-2.0 or MIT). SAM3/SAM 3.1 uses the custom "SAM License" (not Apache), with export/military and indemnity clauses. SAM2Long (code) and DEVA are non-commercial.

### Cited Findings
- SAM2 / SAM2.1 checkpoints, demo and training code: Apache-2.0 — [facebookresearch/sam2](https://github.com/facebookresearch/sam2)
- SAM3 / SAM 3.1: "SAM License" (last updated 19 Nov 2025). Non-exclusive, worldwide, non-transferable, royalty-free limited licence to use, reproduce, distribute, modify. Does not mention commercial use either way (no explicit NC clause). Restrictions: trade-controls and ITAR, prohibited end uses including "military or warfare purposes", no reverse engineering of the underlying components, research publications must acknowledge use, redistribution must include the licence; user indemnifies Meta for third-party claims; licence terminates if you sue Meta over IP in the materials; Meta may modify the agreement; California law. Weights are gated on Hugging Face (access request) — [SAM3 LICENSE](https://github.com/facebookresearch/sam3/blob/main/LICENSE), [facebookresearch/sam3](https://github.com/facebookresearch/sam3). My summary is not legal advice.
- SAMURAI: Apache-2.0 — [yangchris11/samurai](https://github.com/yangchris11/samurai)
- DAM4SAM: Apache-2.0 — [jovanavidenovic/DAM4SAM](https://github.com/jovanavidenovic/DAM4SAM)
- SAM2Long: most of project CC BY-NC 4.0 (non-commercial); original SAM2 files Apache-2.0 — [Mark12Ding/SAM2Long](https://github.com/Mark12Ding/SAM2Long)
- EfficientTAM: Apache-2.0 (checkpoints and code) — [yformer/EfficientTAM](https://github.com/yformer/EfficientTAM)
- EdgeTAM: Apache-2.0 (code and checkpoints) — [facebookresearch/EdgeTAM](https://github.com/facebookresearch/EdgeTAM)
- MedSAM2: Apache-2.0 — [bowang-lab/MedSAM2](https://github.com/bowang-lab/MedSAM2)
- Cutie: MIT (README links MIT; bundled RITM code has its own licence). XMem: MIT — [Cutie](https://github.com/hkchengrex/Cutie), [XMem LICENSE](https://raw.githubusercontent.com/hkchengrex/XMem/main/LICENSE)
- DEVA: CC BY-NC-SA 4.0 (non-commercial, share-alike) — [DEVA LICENSE.md](https://raw.githubusercontent.com/hkchengrex/Tracking-Anything-with-DEVA/main/LICENSE.md)

### Inferences
- If the project may ever be commercial: SAM2.1 (+SAMURAI/DAM4SAM/EfficientTAM/EdgeTAM) is the clean path; SAM3 needs a legal read of the SAM License; avoid DEVA and SAM2Long code.
- SAMOSA's licence is unchecked.

### Gaps
- Licences of datasets (SA-V, SA-1B) and of specific SAM2 checkpoint hosting not checked. SAM License commercial status is not explicit; treat as needing review rather than as cleared.

## Bottom-line recommendation for the report writer (inference, not sourced)

1. Do not expect whole-frame SAM2/SAM3 propagation to hold an ~11 px ball: input is a fixed 1024/1008 px square with stride-16 memory features, SAM3 at full frame is borderline OOM on 8 GB, and the only small-fast-object benchmark shows mean mIoU ~0.44-0.51 for SAM2-family trackers.
2. The most promising experiment is SAM2.1 tiny/small (Apache-2.0, 39-46M params) on moving native-resolution 512-1024 px crops, seeded by human/high-confidence detections, run forward and reverse with `propagate_in_video(reverse=True)` and fused with the detector, with DAM4SAM/SAMURAI layers as optional add-ons; then fine-tune on pseudo-masks from the ~942 ball frames in runs of 3+.
3. Cheapest first test: take clips with known dense ground truth (the KB says `arsenal_example` is the only clean dense sequential clip), prompt a sharp frame, propagate both ways with SAM2.1-small at native-crop vs full-frame, and measure recall within a few pixels versus the detector baseline.
