# Tiny-object video detection with temporal aggregation (incl. non-causal / future frames) for an ~11 px football

Project context relied on (Step 0, `mcp__knowledge__search "footy ball detection"` then read in full): `~/knowledge/wiki/footy-ball-detection-findings.md` (local KB page, no URL). Facts taken from it: ball ~11 px at 1280 width (~25% under 10 px); 640 px gives zero ball recall and 1280 px is the dominant lever; RT-DETR-L at 1280 fits in 8 GB at batch 2; XS-VID's finding that DETR-family models underperform and YOLOFT wins; the key enabler that multi-frame input needs only a centre-frame label and neighbours are recoverable from filenames; broadcast cuts make frame t-1 a different camera; RF-DETR ruled out (resolution cap). The page also flags two conflicting RT-DETR-L ball-mAP50 figures (.221 vs .665) that were never re-measured, so no baseline number from it is quoted here.

Evidence labels: "(paper)" = read in the paper or repo text; "(aggregator)" = secondary page; "UNVERIFIED" = not confirmed. Today is 2026-11-14.

---

## 1. XS-VID benchmark / YOLOFT: leaderboard, repo, licence, weights, VRAM, frames, numbers on extremely small objects

### Takeaway
XS-VID (HUST, now an IEEE TPAMI 2026 paper with an expanded v2 release) is the closest published benchmark to our regime, and its own baseline YOLOFT (YOLOv8 + RAFT-style multi-scale correlation with the PREVIOUS frame's features) is best on it. But on the final v2 benchmark the winning margin on extremely-small (eS) objects is small (YOLOFT-L APes 11.5 vs YOLOv8-L 10.1 vs D-FINE-L 11.0), absolute eS numbers are low, and YOLOFT is causal (past frame only) and AGPL-3.0. It is a reference design, not a ready answer for an offline, future-frame detector.

### Cited Findings
- XS-VID (arXiv 2407.18137) has been revised: the original v1 (Jul 2024) had 38 segments / ~12k images / 258,944 instances in 8 categories; the current title is "A Large-Scale Benchmark for Small Object Detection and Tracking in Videos" with 374 sequences, 223K frames, 1.4M boxes, 7 categories, listed as IEEE TPAMI 2026 (DOI 10.1109/TPAMI.2026.3741044). Numbers therefore differ between v1 and the current version, so quote the version. — [arXiv abs](https://arxiv.org/abs/2407.18137), [project page](https://gjhhust.github.io/XS-VID/), [v1 HTML](https://arxiv.org/html/2407.18137v1)
- CURRENT (v2/TPAMI) detection leaderboard, AP / APes (extremely small) / latency ms @ batch 1 (paper, Table III): YOLOFT-L 29.3 / 11.5 / 29.3; YOLOFT-S 25.8 / 9.8 / 16.3; D-FINE-L 28.3 / 11.0 / 72.3; CO-DETR 28.3 / 7.8 / 92.1; Unicorn 27.6 / 9.7; YOLOv8-L 26.7 / 10.1 / 26.4; YOLOv8-S 22.7 / 9.0 / 14.6; HyperYOLO-L 26.2 / 11.1; YOLOv8-L-worldv2 FT 27.2 / 11.0; StreamYOLO 21.1 / 7.2 / 47.5; TransVOD 17.4 / 3.8 / 136.4; Deformable-DETR 17.1 / 3.8; YOLO11-L 21.1 / 7.7; YOLOv9-C 22.6 / 7.7; FGFA 11.7 / 0.0 / 151.5; SELSA 13.9 / 0.0; MEGA 9.8 / 0.0; DFF 10.9 / 0.0; TROI 12.3 / 0.0; DiffusionVID 11.5 / 0.0. — [arXiv HTML (latest)](https://arxiv.org/html/2407.18137)
- In that table every classic VOD method (DFF, FGFA, SELSA, TROI, MEGA, DiffusionVID) scores APes = 0.0 and the DETR-style TransVOD / Deformable-DETR score 3.8; D-FINE-L (a DETR-family detector) is competitive at 11.0 APes. YOLOV, RT-DETR and RF-DETR are not in the table (YOLOV is only cited in related work). — [arXiv HTML (latest)](https://arxiv.org/html/2407.18137)
- v1 (2024) numbers, different data protocol, for comparison with the older claims: YOLOFT-S AP 33.2 / APes 21.5; YOLOFT-L 36.4 / 24.7 (Table 2; Table 8 and text say 36.2 / 25.4, an internal inconsistency); YOLOv8-L 33.6 / 21.3; YOLOv8-S 30.0 / 17.8; TransVOD 21.8 / 8.8; Deformable-DETR 21.3 / 11.3; FGFA 12.3 / 0.2; MEGA 7.8 / 1.1; params YOLOFT-S 12.84M, YOLOFT-L 45.16M. — [arXiv v1 HTML](https://arxiv.org/html/2407.18137v1)
- YOLOFT design: YOLOv8 detector plus a Multi-Scale Spatio-Temporal Flow (MSTF) neck that builds multi-scale correlation volumes between current-frame and previous-frame features, uses a RAFT-style lookup with a GRU update, and fuses the resulting motion features into static features. The temporal state carries accumulated context from earlier frames. The paper does not mention future frames or bidirectional use. — [v1 HTML](https://arxiv.org/html/2407.18137v1), [latest HTML](https://arxiv.org/html/2407.18137)
- Cost (latest paper): YOLOFT-S 13.4M params / 36.6 GFLOPs; YOLOFT-L 49.8M params / 194.5 GFLOPs, FLOPs measured at 640x640. Latency 16.3 ms (S) and 29.3 ms (L) at batch 1 (a 3090 is named for testing in v1 appendix). VRAM and training hardware are not stated in either version. — [latest HTML](https://arxiv.org/html/2407.18137), [v1 HTML](https://arxiv.org/html/2407.18137v1)
- Input protocol: v1 trains and tests at 1024x1024 (3840x2160 source split into two 2160x2160 halves with 240 px overlap, then downscaled); the latest version releases 1024x1024 squares, annotates at 3840x2160 and evaluates deployment-style at 1920x1080. — [v1 HTML](https://arxiv.org/html/2407.18137v1), [latest HTML](https://arxiv.org/html/2407.18137)
- The text does not address camera ego-motion compensation (footage is from DJI Air3 drones); it only notes high pixel displacement relative to object size. — [latest HTML](https://arxiv.org/html/2407.18137)
- Repo `gjhhust/YOLOFT`: licence AGPL-3.0 (retains Ultralytics, BoT-SORT and CUDA-operator notices; auxiliary assets keep own licences); 25 stars, 5 forks, single commit on main; vendors its own `ultralytics/` directory (version not stated); needs Python 3.10, PyTorch 2.4.0+cu118, nvcc/CUDA 11.8 and GCC 11 to build CUDA operators (DCNv3-style, credited to InternImage); three pipelines (VID `tools/xsvid/train_vid.py`, unified MOT, unified SOT). README reports VID AP 29.3. Weights `vid.pt`, `unified_mot.pt`, `unified_sot.pt` are hosted on Hugging Face (not in repo). Training recipe: `config/recipes/vid.yaml` + `legacy_yoloft.yaml`, initialised from `vid.pt`; epochs, batch size, VRAM, clip length not documented. — [GitHub](https://github.com/gjhhust/YOLOFT), [REPRODUCTION.md](https://raw.githubusercontent.com/gjhhust/YOLOFT/main/docs/REPRODUCTION.md)
- Release links: dataset [HF XS-VID-v2](https://huggingface.co/datasets/lanlanlan23/XS-VID-v2), weights [HF YOLOFT-XSVID-v2-weights](https://huggingface.co/lanlanlan23/YOLOFT-XSVID-v2-weights). The project page names no licence for data or code ("consult the included licenses before redistribution or commercial use"). — [project page](https://gjhhust.github.io/XS-VID/)
- No separate 2025 follow-up detector paper on XS-VID was found; v2 is an expanded benchmark (adds MOT/SOT tracks) with YOLOFT-L as the released detection baseline. — [project page](https://gjhhust.github.io/XS-VID/)

### Inferences
- Ball size: ~11 px wide at 1280 equals roughly 16 px at 1920 wide, i.e. area ~120-250 px^2. XS-VID's "extremely small" band is reported (one summary, UNVERIFIED) as 0-12^2 px, so our ball sits at or just above the eS/"relatively small" boundary. YOLOFT's APes numbers are the pessimistic end for us; the next bin (APrs: YOLOFT-L 16.2 vs YOLOv8-L 14.4) is probably closer.
- The honest reading of the current leaderboard is "temporal fusion adds ~1.4-2.5 AP points over the same-size YOLOv8 baseline at 29-36 ms", not a step change. Within-XS-VID, higher resolution / better single-frame backbone moves eS about as much. Treat YOLOFT as one option, not as the proven winner. (The project KB page describes YOLOFT as "the winner"; that is true of the v1 table but the margin is thin in the final one.)
- Since the repo bundles a forked ultralytics with custom CUDA ops and a TPAMI-era training stack, porting only MSTF into our ultralytics 8.x/RT-DETR training loop is non-trivial; running the repo as-is would be the lower-effort route for a baseline.

### Gaps
- YOLOFT's number of previous frames / recurrence length, training VRAM, batch size, epochs: not stated anywhere read; would need the repo code or the TPAMI PDF (not fetched).
- Exact XS-VID size-bin definitions (eS/rS/gS) in the final version: not confirmed.
- No YOLOFT run at 1280+ crops or on non-drone footage; behaviour with pan/tilt/zoom broadcast camera unknown.
- The full v2 YOLOFT result at higher input size is only in a chart image on the project page (not machine-readable).

---

## 2. Tiny-object video detection beyond XS-VID: infrared small-target video (IRSTD), satellite video, drone/UAV

### Takeaway
Moving-infrared-small-target (MIRSTD) detection is the most active area and the most transferable: 2024-2026 methods feed 2-5 frames (or a recurrent state / explicit motion map) into a YOLOX- or RT-DETR-style detector, and the strongest recent result (MI-DETR) is an RT-DETR with a hand-crafted motion pathway. YOLOMG (2025) is the single most relevant paper found: 1920x1080 video, objects mostly under 12x12 px, YOLOv5-style detector at 1280, frame differences from t-k AND t+k with homography alignment. Satellite-video work is mostly classical.

### Cited Findings
**MIRSTD (infrared) family**
- Tridos (TGRS 2024, UESTC-nnLab): triple-domain (spatial, temporal, frequency) multi-frame detector with YOLOX head; takes 5 frames at inference; frames reshaped to 512x512; trained on two RTX 3090s; reports mAP50 DAUB-R 85.37, ITSDT-15K 80.41, IRDST-H 38.51 (as re-tabulated in the MI-DETR paper). Repo Apache-2.0, tested on one RTX 3090 (24 GB), no pretrained weights linked. — [arXiv 2406.06949 (PDF text read)](https://arxiv.org/abs/2406.06949), [GitHub](https://github.com/UESTC-nnLab/Tridos)
- DTUM (Direction-coded Temporal U-shape Module, TNNLS 2023): a plug-in temporal module attached to single-frame segmentation nets (ALCNet, ResUNet, DNANet, ISNet, UIUNet); has an `--align` flag for non-aligned backgrounds; datasets NUDT-MIRSDT (synthetic, 120 sequences), IRDST, plus loaders for SatVideoIRSDT; repo licence not stated on the page; only one weights file linked. 5 input frames per the MI-DETR comparison table. — [GitHub](https://github.com/TinaLRJ/Multi-frame-infrared-small-target-detection-DTUM), [MI-DETR PDF](https://arxiv.org/pdf/2603.05071)
- MI-DETR (arXiv 2603.05071, Mar 2026): three stages: RIMCA (Recurrent Interpretable Motion Cue Aggregation, a hand-crafted recurrent motion state built from frame differences / center-surround filtering, computed only from current + EARLIER frames, i.e. causal), PMI (dual appearance/motion ResNet-18 pathways with bidirectional cross-attention) and a standard RT-DETR encoder-decoder. Trained at 512x512 for 600 epochs, batch 32, AdamW, on A100 40 GB; 32.44M params, 93.9 GFLOPs, 34.6 FPS on RTX 3090. mAP50: DAUB-R 98.00, ITSDT-15K 88.30, IRDST-H 70.30 versus the best prior multi-frame method iMoPKL 88.57 / 80.67 / 43.95 (IRDST-H: +26 points). No code link found on the arXiv page; no licence stated. — [arXiv abs](https://arxiv.org/abs/2603.05071), [PDF text read](https://arxiv.org/pdf/2603.05071)
- MI-DETR's own table lists multi-frame methods on IRDST-H (5 frames unless noted): MoPKL 40.66, SSTNet 39.04, TMP 38.93, Tridos 38.51, DTUM 37.98, STME 34.29, iMoPKL (2 frames) 43.95; the best single-frame (L2SKNet) 37.15. I.e. on IRDST-H, fixed-window multi-frame methods barely beat the best single-frame one, while on ITSDT-15K they gain 7-12 points (e.g. Tridos 80.41 vs best single-frame 73.37). — [MI-DETR PDF text read](https://arxiv.org/pdf/2603.05071)
- MI-DETR's ablation (paper text): the recurrent state readout A_t alone gives 95.17 mAP50 on DAUB-R and the full motion representation 98.00. — [MI-DETR PDF](https://arxiv.org/pdf/2603.05071)
- Awesome-list of 2023-2026 multi-frame IRSTD with code: DTUM, 4-D TT/TR, SSTNet (TGRS24), DMIST, Tridos, LMAFormer, MOCID (AAAI25), NeurSTT (PR25), OSFormer (TIP25), LVNet "Low-Level Matters" (JSTARS25), MFE-Net (TGRS25), S2MVP, DQAligner (TGRS26, for large motion), TDCNet (AAAI26), Loddis (AAAI26), CHAL (CVPR26), MIST (TIP26), DeepPro (TPAMI26), MoPKL (AAAI25), SeViL (AAAI26); mostly UESTC-nnLab repos. (Titles/links only; contents not verified.) — [awesome-infrared-small-targets](https://github.com/Tianfang-Zhang/awesome-infrared-small-targets)
- TDCNet (arXiv 2511.09352, AAAI 2026): temporal-difference-convolution re-parameterised blocks plus a parallel 3D backbone with cross-attention; abstract gives no numbers and no code link. — [arXiv abs](https://arxiv.org/abs/2511.09352)
- Benchmarks sizes (from MI-DETR): ITSDT-15K 10,000 train / 5,000 test; IRDST-H 8,725/2,694/5,354; DAUB-R 7,500/2,000/4,277. IR frames are 640x480 (Tridos) and downsized to 512x512. These are static-or-slowly-moving IR cameras with targets of a few pixels; no broadcast pan/tilt/zoom. — [MI-DETR PDF](https://arxiv.org/pdf/2603.05071), [Tridos PDF](https://arxiv.org/abs/2406.06949)

**Drone / UAV**
- YOLOMG (arXiv 2503.07115, 2025): drone-to-drone detection; objects average ~0.01% of the frame, mostly under 12x12 px (a drone at ~100 m is ~10x10 px) in 1920x1080 @ 30 fps video (ARD100: 100 videos, 202,467 frames, 65/35 split). Architecture: shallow YOLOv5s-derived backbone (35 layers, stride 16, four heads incl. an extra small-object head); a pixel-level motion difference map (grayscale three-frame difference from frames t-k and t+k, averaged, then morphological open/close) fused with RGB features via adaptive weights + CBAM. Frames t-k and t+k are ALIGNED to frame t with a homography from grid-sampled keypoints tracked by pyramidal Lucas-Kanade with RANSAC. Best k=2 (AP 0.78 at 640 on ARD100). — [arXiv HTML](https://arxiv.org/html/2503.07115)
- YOLOMG results, ARD100 AP@0.5: YOLOMG-1280 0.85, YOLOMG-640 0.78, YOLOv9 0.64, CFINet 0.63, YOLOv11 0.55, YOLOv5 0.53, YOLOv8 0.53, Dogfight 0.50, MEGA 0.23, LSFTE 0.17, TransVisDrone 0.15. FPS on RTX 2080Ti: YOLOMG-640 133, YOLOMG-1280 35, TransVisDrone 5, MEGA 8. On NPS-Drones: YOLOMG-1280 0.95, -640 0.92, TransVisDrone 0.95. Note: gain from 640 to 1280 (0.78 to 0.85) matches our own finding that resolution matters. — [arXiv HTML](https://arxiv.org/html/2503.07115)
- YOLOMG repo `Irisky123/YOLOMG`: GPL-3.0; YOLOv5-style layout (`--weights yolov5s.pt`; framework not named on the page); no pretrained YOLOMG weights; ARD100 on BaiduYun; 86 stars. The arXiv page carries CC BY-NC-SA 4.0 (paper, not code). — [GitHub](https://github.com/Irisky123/YOLOMG), [arXiv HTML](https://arxiv.org/html/2503.07115)
- TransVisDrone (2022): CSP-DarkNet53 + Video-Swin spatio-temporal transformer; AP50 0.95 NPS, 0.75 FLDrones, 0.80 AOT; scores only 0.15 on ARD100 per the YOLOMG comparison (5 FPS). — [arXiv 2210.08423](https://arxiv.org/pdf/2210.08423.pdf), [YOLOMG HTML](https://arxiv.org/html/2503.07115)
- Temporal-YOLOv8 (Sensors/PMC 2024): standard Ultralytics YOLOv8 with frames from different times packed into input channels (three grayscale frames in the 3 RGB slots, zero extra compute; Color-T-YOLO 9 channels; Manyframe-YOLO 11 grayscale channels); the paper's illustration samples ~15 frames BEFORE and AFTER the current frame (30 fps). On its in-house Nano-VID (stationary cameras; tiny objects) mAP 0.465 default YOLO, 0.583 single-frame with small-object augmentations, 0.839 Temporal-YOLOv8. No code released; explicitly not tested on moving cameras ("motion compensation should be explored"). — [PMC11598073](https://pmc.ncbi.nlm.nih.gov/articles/PMC11598073/)
- Other drone-video items surfaced but not read in full: STARD-Net (spatio-temporal attention for tiny airborne objects from moving drones), C2FDrone (YOLO-based, "16.22% improvement over single-frame", abstract-level only). — [STARD-Net listing](https://scholarsmine.mst.edu/comsci_facwork/2146), search snippets only

**Satellite video**
- VISO benchmark (TGRS 2021; 47 Jilin-1 videos; moving objects typically under 50 px, dense) with a classical Motion Modeling Baseline (multi-frame differencing + matrix completion); a 2024 follow-up HiEUM (unsupervised) notes deep learning for satellite-video moving-object detection is "largely underexplored"; a YOLOv5 study found classical methods beat the trained YOLOv5. No 2025 deep detector was found. — [VISO arXiv 2111.12960](https://arxiv.org/pdf/2111.12960), [HiEUM arXiv 2411.15895](https://arxiv.org/html/2411.15895v1), [VISO repo](https://github.com/qingyonghu/viso)

**Ball-specific heatmap baseline (for context; covered by other researchers)**
- WASB-SBDT (BMVC 2023): HRNet-derived heatmap net, N=3 consecutive frames, 288x512 input, trained from scratch; on its Soccer set (1920x1080, 25 fps, 6 clips re-annotated, ball displacement ~10-16 px/frame) F1 88.3 / AP 83.6 at a 4 px distance threshold, vs TrackNetV2 F1 86.6, DeepBall 44.5. Repo MIT, 212 stars; weights listed in MODEL_ZOO.md. Caveat: the soccer clips are fixed-camera multi-view footage, not a panning broadcast. — [arXiv 2311.05237 (PDF text read)](https://arxiv.org/pdf/2311.05237), [GitHub](https://github.com/nttcom/WASB-SBDT)

### Inferences
- Two designs transfer directly to our setup: (a) MI-DETR's recipe (RT-DETR detector + explicit motion channels from differences) because we already train RT-DETR in ultralytics; (b) YOLOMG's recipe (shallow YOLO at 1280 + aligned t±k difference map), which is already validated at 1080p with sub-12 px objects. Both feed motion as extra input/feature pathway rather than learning flow end-to-end.
- MI-DETR's gain on IRDST-H (70 vs 44 mAP50) is partly dataset-specific (IR sensor, no pan/zoom); do not expect a similar jump on broadcast football, where the background moves with the camera and other moving things (players) are everywhere. Both YOLOMG and MI-DETR rely on "moving small thing on mostly static background", so a ball in flight is aided but a rolling/occluded ball among moving players is not.
- Evidence that fixed-window 5-frame methods give inconsistent gains (IRDST-H) while explicit alignment/motion cues help is mildly supportive of doing camera-motion compensation before differencing.
- I found no MIRSTD method reporting numbers on a panning broadcast; the transfer is architectural only.

### Gaps
- Number of input frames and past-only vs bidirectional for Tridos, SSTNet, DTUM-family: the Tridos text shows 5-frame input, but I did not find a statement on whether windows are centred on the target (non-causal) or trailing; MI-DETR labels only itself as causal.
- MI-DETR and TDCNet code availability: no repo found; the arXiv pages list none. Re-implementation would be needed (RIMCA is hand-crafted and described in the paper, so feasible).
- Licences for most UESTC-nnLab repos other than Tridos (Apache-2.0) were not checked.
- DAUB / ITSDT-15K image-level target sizes (pixels) not extracted.

---

## 3. General VOD (YOLOV/YOLOV++, TransVOD/++, MEGA, SELSA, FGFA, PTSEFormer, newer): future frames? cost at high resolution?

### Takeaway
Several classical VOD methods are non-causal by construction (TransVOD samples reference frames bilaterally; YOLOV++ global sampling draws from the whole video; SELSA is sequence-level), so "offline aggregation" is standard in VOD. But all were designed and tuned on ImageNet VID at ~576-800 px with large objects, and every one scores ~0 on extremely-small objects in XS-VID. They are useful as aggregation ideas (proposal-level attention over many frames), not as drop-in tiny-ball detectors.

### Cited Findings
- YOLOV++ (arXiv 2407.19650, "Practical VOD via Feature Selection and Aggregation"; IJCV version noted Feb 2026): YOLOX-based; selects top candidates per frame (under ~100 proposals/frame on average) and aggregates them across frames with attention. Reference frames: global sampling Fg picked randomly from the whole video, or local sampling Fl of consecutive frames; the paper does not say references are past-only. YOLOX-S ablation AP50 vs global frames Fg = 4/7/15/23/31/39: 75.7/76.9/78.0/78.4/78.7/78.7 (default Fg=31); local Fl = 4..39 reaches only 70.5 to 74.1, so many frames sampled globally beats nearby consecutive frames. ImageNet VID AP50: YOLOX-S 69.5 to YOLOV++-S 78.7 (5.3 ms), YOLOX-L 76.5 to 84.2 (7.6 ms), FocalLarge 92.9 (27.6 ms), on 3090 FP16 at 576x576 (training 352-672). Naive dense attention on YOLOX-L with 4 frames x 8,400 candidates needs 12.1 GB; small objects not specifically evaluated (mAP reported by motion speed). — [arXiv HTML](https://arxiv.org/html/2407.19650v1)
- YOLOV/YOLOV++ repo: Apache-2.0, 440 stars, built on YOLOX (Python 3.7), weights on Google Drive; an online-mode demo (`yolov_demo_online.py`) exists in addition to `vid_demo.py` (the README does not explain the difference; offline vs online semantics UNVERIFIED). — [GitHub](https://github.com/YuHengsss/YOLOV)
- TransVOD / TransVOD++ / TransVOD Lite (TPAMI, arXiv 2201.05047): Deformable-DETR-based; uses 14 reference frames and "bilateral uniform sampling" (reference frames randomly sampled from both sides of the current frame, following MMTracking) at train time, with fixed-stride reference sampling at inference; TransVOD Lite feeds a whole clip in one pass and predicts all frames. ImageNet VID: TransVOD++ ~90.0% mAP (Swin-B), TransVOD Lite 83.7% at ~30 FPS on one V100; code at SJTU-LuHe/TransVOD. In XS-VID: AP 17.4 / APes 3.8 at 136 ms (latest), 21.8 / 8.8 in v1. — [TransVOD PDF text read](https://arxiv.org/pdf/2201.05047), [XS-VID](https://arxiv.org/html/2407.18137)
- SELSA aggregates features from all frames in the video (non-local); FGFA warps neighbouring frames' features by optical flow and averages them (reported ~122M params / ~182 ms in XS-VID v1; 151.5 ms latest). FGFA/SELSA/MEGA/DFF/TROI/DiffusionVID all score APes = 0.0 on the current XS-VID. — [search summary of SELSA/FGFA](https://pmc.ncbi.nlm.nih.gov/articles/PMC9658927/), [XS-VID latest](https://arxiv.org/html/2407.18137)
- StreamYOLO (online/streaming) reaches 21.1 AP / 7.2 APes on XS-VID, the best of the classic VOD group but below plain YOLOv8-L (26.7 / 10.1). — [XS-VID latest](https://arxiv.org/html/2407.18137)
- MEGA scores 0.23 AP50 on ARD100 drone data at 8 FPS (2080Ti), TransVisDrone 0.15 at 5 FPS. — [YOLOMG](https://arxiv.org/html/2503.07115)

### Inferences
- Bilateral reference sampling (TransVOD) and global sampling (YOLOV++) show offline/non-causal aggregation is mainstream and helps; YOLOV++'s finding that many sparse global frames beat dense consecutive ones is about object semantics/appearance (re-identifying a distinct object across a video) and is unlikely to carry to a 11 px ball with little appearance signal, where short-range motion matters most.
- Cost at 1280+ is the blocker: FGFA/SELSA/TransVOD run at 90-230 ms on XS-VID 1024 px, and dense attention over 8,400 candidates x frames already costs 12 GB at 576 px; fitting 1280-1920 px within 8-16 GB would require the proposal-level reduction YOLOV++ uses or short clips.
- PTSEFormer and other "newer" ImageNet-VID methods were not examined (see Gaps); given the XS-VID evidence they are unlikely to help tiny objects.

### Gaps
- PTSEFormer, MEGA/DiffusionVID details, and any 2025-2026 general-VOD papers with future-frame aggregation: not read. A search did not surface a clean 2025-26 non-causal VOD method specialised for tiny objects.
- YOLOV++ number of frames at inference and whether "online" mode restricts to past frames: not confirmed from README or paper text read.
- FGFA's exact +-10 frame window: stated from background knowledge only; not sourced here, UNVERIFIED.

---

## 4. Simple strong baselines: stacking frames / differences as input channels; camera motion; cuts

### Takeaway
Early fusion (channel stacking of 3-7 frames, first-conv weights replicated) is cheap, consistently helps small moving objects in the published ablations (+4 to +13 mAP50 points for tiny YOLO variants, up to +25 mAP in one tiny-object dataset), costs almost no parameters, and is the baseline to beat. The published results are overwhelmingly on static or near-static cameras; the only paper with explicit motion compensation (YOLOMG) uses a homography and symmetric t-k/t+k frames, which is the closest match to our pan/tilt/zoom broadcast.

### Cited Findings
- "Lightweight Multi-Frame Integration for Robust YOLO Object Detection in Videos" (arXiv 2506.20550, Jun 2025): concatenates n consecutive RGB frames along channels (3n channels), repeats and scales the first-conv weights by 1/n, supervises only the latest frame; early fusion beat grouped-conv (per-frame feature groups). YOLOv7-tiny on MOT20Det: mAP50 0.797 to 0.841 (3 adjacent frames), 0.855 (7), best 0.873 (3 frames, step 3); mAP50-95 0.413 to 0.478 (7 frames). YOLOv7 (full) gains are marginal on MOT20Det (0.882 vs 0.879). On BOAT360 (authors' moving-boat fisheye buoy set, small objects): tiny 0.411 to 0.539 (+12.8), full YOLOv7 0.583 to 0.664 (+8.1). More frames can hurt (9 degrades; BOAT360 degrades beyond 5, attributed to rapid scene change). Parameters barely grow (YOLOv7-tiny 6.006M to 6.011M for 7 frames). Past frames only (no future); no explicit motion compensation, augmentations applied consistently across the stack; no code or licence stated; inference measured on Orin AGX (55 to 53-55 FPS). — [arXiv HTML](https://arxiv.org/html/2506.20550)
- Temporal-YOLOv8 (above): channel packing with +-15 frame sampling, 0.583 to 0.839 mAP on a tiny-object static-camera set; no code. — [PMC11598073](https://pmc.ncbi.nlm.nih.gov/articles/PMC11598073/)
- YOLOMG: differences from aligned t-k and t+k frames, k=2 best on ARD100, homography via LK + RANSAC for ego-motion. — [arXiv HTML](https://arxiv.org/html/2503.07115)
- MI-DETR on explicit motion operators: examines two- and three-frame differencing (|I_t - I_{t-1}| and min(|I_t - I_{t-1}|, |I_{t-1} - I_{t-2}|)), Farneback flow and others; says plain frame differences "do not distinguish target changes from background changes", motivating its recurrent filtered state. DTUM's code exposes an `--align` option for non-aligned backgrounds. — [MI-DETR PDF](https://arxiv.org/pdf/2603.05071), [DTUM GitHub](https://github.com/TinaLRJ/Multi-frame-infrared-small-target-detection-DTUM)
- Scene changes degrade distant-frame stacking (the 2506.20550 BOAT360 observation above); none of the sources read describes shot-cut handling. The project KB page notes the cut trap and proposes duplicating frame t on a cut. — [arXiv HTML](https://arxiv.org/html/2506.20550), KB page `footy-ball-detection-findings.md`
- Camera motion: Temporal-YOLOv8 states its setup "will probably not perform as well" on moving platforms; XS-VID's text does not address ego-motion compensation. — [PMC11598073](https://pmc.ncbi.nlm.nih.gov/articles/PMC11598073/), [XS-VID latest](https://arxiv.org/html/2407.18137)

### Inferences
- For our offline use, the natural early-fusion variant is symmetric, e.g. frames {t-2k, t-k, t, t+k, t+2k} (or {t-k, t, t+k} plus aligned difference maps) with short steps; YOLOMG's symmetric differencing is the only evidence found for future-frame use at tiny scale, and Temporal-YOLOv8 used +-15 frames; no paper found ablates causal vs symmetric windows directly. That ablation is cheap for us and should be run.
- Replicating pretrained first-conv weights/n (as in 2506.20550) preserves COCO-pretrained init, which matters with ~1.5k human labels.
- Because the camera pans/zooms, un-aligned differencing will light up the whole pitch; alignment (global homography/affine from feature tracking, plus cut detection, plus masking players) is likely necessary before difference channels help. This is inference; no source tests it on broadcast football.
- Gains reported are mostly on tiny YOLO variants; full-size models showed smaller gains (YOLOv7 on MOT20Det), so expect diminishing returns with RT-DETR-L.

### Gaps
- No ablation found specifically on a ball (table tennis, football) with stacked frames; WASB/TrackNet are heatmap models, not box-detector early fusion.
- No source found quantifying benefit of ego-motion alignment vs none on a moving-camera tiny-object set.
- No source on difference-channel vs raw-frame-stack for RT-DETR.

---

## 5. Ultralytics-compatible or easily adaptable implementations

### Takeaway
Ultralytics supports >3-channel inputs officially (dataset-YAML `channels`, TIFF multichannel loading), so early-fusion experiments are feasible in our existing stack for YOLO models; RT-DETR with a 9-channel stem is not documented and needs a test. YOLOFT is a vendored Ultralytics fork (AGPL, custom CUDA ops); YOLOMG is YOLOv5-style (GPL-3.0); YOLOV/YOLOX-based code is Apache-2.0 but not Ultralytics.

### Cited Findings
- Ultralytics 8.3.112 (Apr 2025) added multispectral/multi-channel training, validation, prediction and export; dataset YAML has `channels: N`; images are N-channel `.tiff`/`.tif` uint8, channel-first when written, read with `cv2.imreadmulti` into HxWxN. Doc example uses 10 channels and YOLO26n. The doc page says nothing about the model-YAML `ch`, HSV/mosaic caveats, or how pretrained first-conv weights are adapted. A July 2025 user comment reports a classification channel-mismatch error. — [Ultralytics coco8-multispectral docs](https://docs.ultralytics.com/datasets/detect/coco8-multispectral), [8.3.112 release](https://newreleases.io/project/pypi/ultralytics/release/8.3.112)
- `DetectionModel` reads `ch` from the model YAML (default 3) in `ultralytics/nn/tasks.py` (seen via mirrored copies, not the official repo); official model-YAML docs state channel dimensions between layers must line up. — [model yaml config docs](https://docs.ultralytics.com/guides/model-yaml-config), mirrored [tasks.py](https://huggingface.co/spaces/llzzyy233/Defect_detection/blob/85fd8d2da310d6a627b57fa1b22e31fcc9eceb96/ultralytics/nn/tasks.py)
- Ultralytics licence: AGPL-3.0, or a separate Ultralytics Enterprise License for closed-source commercial use; README lists YOLOv3/v8/11/26 (YOLO27 "coming soon"); RT-DETR not mentioned on that page. — [GitHub](https://github.com/ultralytics/ultralytics)
- Temporal-YOLOv8 was built on Ultralytics YOLOv8 (citation only; no code). YOLOMG's training commands (`train.py --cfg --weights yolov5s.pt`) suggest a YOLOv5 codebase. — [PMC11598073](https://pmc.ncbi.nlm.nih.gov/articles/PMC11598073/), [YOLOMG repo](https://github.com/Irisky123/YOLOMG)

### Inferences
- Lowest-effort experiment: YOLO11/YOLO26-class model at imgsz=1280 with `channels: 9` (frames t-k,t,t+k) or 5 grayscale/diff channels; custom dataloader to build stacks on the fly (TIFF on disk would multiply storage: 1920x1080 x 9 channels per sample over 356k candidates). Standard colour/HSV and mosaic augmentations need checking for multi-frame consistency (mosaic mixes unrelated clips; per-frame HSV breaks alignment), which the docs do not cover.
- For RT-DETR in ultralytics, the stem is a ResNet/HGNet stem with a 3-channel first conv; patching it to 9 channels (and expanding pretrained weights) is likely possible but UNVERIFIED here.
- MI-DETR-style: precompute motion maps (RIMCA is hand-crafted) as extra input channels and add a second backbone pathway; the paper uses ResNet-18 pathways + RT-DETR, so it maps onto ultralytics RT-DETR only with custom model code.

### Gaps
- Did not verify RT-DETR + `ch=9` in current ultralytics (no doc or issue found).
- Did not check how ultralytics handles pretrained first conv when `ch != 3` (it may silently skip loading that layer).

---

## 6. Licences summary (flag restrictions)

### Takeaway
Almost everything useful is copyleft: YOLOFT (AGPL-3.0, and its Ultralytics lineage), YOLOMG (GPL-3.0), Ultralytics (AGPL-3.0 or paid Enterprise licence). Tridos, YOLOV/YOLOV++ are Apache-2.0; WASB is MIT. Dataset/weights licences are mostly unstated. If the football product is closed-source/commercial, AGPL/GPL matter; for private research use they do not.

### Cited Findings
| Item | Licence found | Source |
|---|---|---|
| Ultralytics (YOLOv8/11/26, RT-DETR wrapper) | AGPL-3.0 or Ultralytics Enterprise License | [GitHub](https://github.com/ultralytics/ultralytics) |
| YOLOFT code (vendors ultralytics, BoT-SORT, InternImage DCNv3 ops) | AGPL-3.0; auxiliary assets keep own licences | [GitHub](https://github.com/gjhhust/YOLOFT) |
| XS-VID-v2 dataset and YOLOFT weights | Not stated on project page ("consult included licenses before redistribution or commercial use") | [project page](https://gjhhust.github.io/XS-VID/) |
| YOLOMG code | GPL-3.0 (paper page CC BY-NC-SA 4.0 applies to the paper, not code) | [GitHub](https://github.com/Irisky123/YOLOMG), [arXiv](https://arxiv.org/html/2503.07115) |
| Tridos | Apache-2.0 | [GitHub](https://github.com/UESTC-nnLab/Tridos) |
| YOLOV / YOLOV++ | Apache-2.0 (YOLOX-based) | [GitHub](https://github.com/YuHengsss/YOLOV) |
| WASB-SBDT | MIT (dataset terms not stated on page) | [GitHub](https://github.com/nttcom/WASB-SBDT) |
| DTUM repo | Not stated on page | [GitHub](https://github.com/TinaLRJ/Multi-frame-infrared-small-target-detection-DTUM) |
| MI-DETR, TDCNet, early-fusion YOLO (2506.20550), Temporal-YOLOv8 | No code found | arXiv/PMC pages above |

### Inferences
- No explicit "non-commercial" code licence was found among the repos read, except the CC BY-NC-SA 4.0 notice on the YOLOMG arXiv page (applies to the paper text). Dataset licences (XS-VID-v2, ARD100, ITSDT-15K) were not established and could carry non-commercial terms.

### Gaps
- Licences of the XS-VID-v2 dataset and YOLOFT weights, ARD100 and ITSDT-15K data, and most other UESTC-nnLab repos: unknown.

---

## Overall pointers for the report writer (kept within the three-section format via inferences)
- Strongest evidence for our regime (1080p, sub-12 px, resolution-sensitive, symmetric t+-k, ego-motion compensation): YOLOMG (AP50 0.53 single-frame YOLOv5/v8 to 0.78 at 640 and 0.85 at 1280 on ARD100). Second: MI-DETR for the RT-DETR + motion-pathway pattern. Third: YOLOFT/XS-VID as the only broad leaderboard, with a thin margin on eS.
- Largest uncertainty: all temporal results come from static or slowly-moving cameras (IR sensors, fixed drones/cameras) except YOLOMG's drone-mounted setup; broadcast pan/tilt/zoom plus shot cuts is untested in the literature read.
- Not covered here (other researchers): TrackNet-family heatmap models in depth, SOT/Siamese trackers, ByteTrack-style post-processing.
- Tool note: several arXiv PDFs returned binary to the fetch tool; MI-DETR, Tridos, WASB and TransVOD figures above were obtained by running pdftotext on the downloaded PDFs. Other claims come from the fetch tool's summarised page text and could contain extraction errors; numbers quoted from tables were checked against the extracted table text only where stated "(PDF text read)".
