# Multi-frame heatmap sports-ball trackers and soccer-specific ball tracking (for offline, tiny-ball broadcast football)

Project context (KB, not web): per-frame RT-DETR-L at 1280, ball ~11 px wide at 1280 (median 10.7-11.5 px, ~25% under 10 px), no tracker; resolution (640 -> 1280) was the lever that mattered; ball-recall figures for the current model are contested (.209 vs .591 for the same weights, unresolved); the KB already lists "heatmap over frame stack" as the intended direction and notes broadcast cuts break frame-difference channels. See ~/knowledge/wiki/footy-ball-detection-findings.md (found via `mcp__knowledge__search "footy ball detection"`).

Research date: 2026-11-14. Everything marked [UNVERIFIED] could not be confirmed from a primary source.

Headline caveat that applies to every model below: **all TrackNet-family and WASB-family models were published at fixed 288x512 input, i.e. a 3.75x downscale from 1080p (WASB, TOTNet, BlurBall, TrackNetV3/V4/V5). None has been shown on a tiny ball in a panning/zooming wide broadcast shot.** The only soccer evidence for these models is the ISSIA-CNR dataset (see Q2), which is not a PTZ broadcast feed.

---

## Q1. TrackNet V1 / V2 / V3 / V4 / V5 (and successors): architecture, window, resolution, rectification, repos, licences, weights, accuracy, fine-tune effort

### Takeaway
TrackNetV2 is the common ancestor (U-Net-like heatmap net on a short stack of consecutive frames at 288x512). V3 adds a median-background input, mixup and a *separate, offline, trajectory-level* inpainting network that uses detections on both sides of a gap; V4/V5 add motion/difference-map attention. Official V3 (MIT, with checkpoints) is the only one with a usable off-the-shelf offline gap-filler, but its checkpoints and its background trick are badminton / fixed-camera specific. V4 and V5 are tennis/badminton papers; V5 had no code/weights found.

### Cited Findings
**TrackNet V1 / V2 (lineage)**
- TrackNetV2 official code/dataset is hosted on the NCTU open-source page, not GitHub; I could not confirm a licence or bundled weights there. Community ports exist (e.g. a PyTorch port listed as MIT, archived; TF2 badminton port stated as not official). — [search result summary](https://relatedrepos.com/gh/ChgygLin/TrackNetV2-pytorch), [TF2 port](https://github.com/Chang-Chia-Chi/TrackNet-Badminton-Tracking-tensorflow2) [licence of official V1/V2 UNVERIFIED]
- TrackNetV2 dataset: badminton, 720x1280, evaluated downsampled to 288x512; 4 px distance threshold for "correct". — [TrackNetV3 paper](https://people.cs.nycu.edu.tw/~yushuen/data/TrackNetV3.pdf)
- TrackNetV2 as re-implemented by the V3 authors: Acc 94.98 / Prec 99.64 / Rec 94.56 / F1 97.03 at 27.70 FPS (the original V2 paper reported 87.72% accuracy). — [TrackNetV3 paper, Table 1 + Sec 4.3](https://people.cs.nycu.edu.tw/~yushuen/data/TrackNetV3.pdf)

**TrackNetV3 (MMAsia 2023, Chen & Wang, NYCU)**
- Two modules: tracking module + rectification module. Tracking module is TrackNetV2 plus (a) an estimated background image concatenated to the frame stack, (b) mixup augmentation. — [paper](https://people.cs.nycu.edu.tw/~yushuen/data/TrackNetV3.pdf), [repo](https://github.com/qaz812345/TrackNetV3)
- Input window: **8 frames** in the paper's implementation (repo training command `--seq_len 8`); overlapping sliding windows, each frame gets up to 8 heatmap predictions that are combined with a Gaussian-weighted average (centre frames weighted most). Evaluated at **288x512**. Batch 10, Adam lr 1e-3, 30 epochs. — [paper Sec 3.4, 4.1](https://people.cs.nycu.edu.tw/~yushuen/data/TrackNetV3.pdf), [repo](https://github.com/qaz812345/TrackNetV3)
- Background estimation = per-pixel **median frame over a fixed-viewpoint court shot** (per rally, or "match median of rally medians"). The paper states it relies on the fixed viewpoint of broadcast court cameras. — [paper Sec 3.1](https://people.cs.nycu.edu.tw/~yushuen/data/TrackNetV3.pdf)
- **Rectification module (InpaintNet):** operates on the *predicted 2D trajectory* (not pixels): U-Net with 1D convolutions over time, input = predicted trajectory + inpainting mask, trained with MSE; trained with ~30% random masking; repo `--seq_len 16`. Mask at inference: for an undetected frame i, track **forward and backward** to the nearest detected frames f and b; the gap [f+1, b-1] is masked only if the shuttle height at both flank frames is below delta=30 px (i.e. it was inside the field of view; a heuristic for "went out of view" vs "occluded"). So it is **offline and bidirectional by construction** (needs a detection on each side of the gap). — [paper Sec 3.3](https://people.cs.nycu.edu.tw/~yushuen/data/TrackNetV3.pdf), [repo](https://github.com/qaz812345/TrackNetV3)
- Reported (badminton, TrackNetV2 test split): Acc 97.51 / Prec 97.79 / Rec 99.33 / F1 98.56 / 25.11 FPS. Ablation (Table 2): V2 94.98 -> +background 95.64 -> +mixup 96.12 -> +rectification 97.51 acc; rectification lifts recall 95.90->99.33 but drops precision 99.62->97.79 (FP1 0.24->1.49%, FP2 0.07->0.43%: inpainted positions can be off, or filled during unlabelled occlusion). Miss rate (FN) 3.56% -> 0.57%. — [paper Table 1, 2](https://people.cs.nycu.edu.tw/~yushuen/data/TrackNetV3.pdf)
- Without the 8-frame ensembling TrackNetV3 runs >75 FPS with ~0.2% accuracy loss. — [paper Sec 4.3](https://people.cs.nycu.edu.tw/~yushuen/data/TrackNetV3.pdf)
- Licence: MIT, stated to cover code and pretrained checkpoints. Weights: `TrackNetV3_ckpts.zip` (Google Drive) containing `TrackNet_best.pt` and `InpaintNet_best.pt` — **badminton (shuttlecock) weights only**. — [repo](https://github.com/qaz812345/TrackNetV3)
- A community fine-tune of TrackNetV3 for beach volleyball exists on Hugging Face (MIT carried forward) — evidence that V3 is fine-tunable to another sport, not an evaluation. — [HF](https://huggingface.co/deadfast/beach-volley-vision-models)
- Note: the metric in V3 deliberately avoids mAP/IoU because of tiny targets ("a 3x3 box drops 50% IoU with a 1 px offset"), using a 4 px centre-distance criterion at 288x512. — [paper Sec 4.2](https://people.cs.nycu.edu.tw/~yushuen/data/TrackNetV3.pdf)

**TrackNetV4 (ICASSP 2025, Raj, Wang, Gedeon, Griffith)**
- "Motion attention maps": frame-difference maps modulated by a learnable motion prompt layer, fused with visual features; a plug-in on top of TrackNetV2 *and* TrackNetV3. — [arXiv 2409.14543](https://arxiv.org/abs/2409.14543), [project page](https://tracknetv4.github.io)
- Repo: TensorFlow/Keras, **MIT** licence, **3-frame sliding window**, default training size 288x512, pretrained `.keras` checkpoints referenced in `docs/RESULT.md` (not read; sport unconfirmed, datasets are tennis and badminton). — [repo](https://github.com/TrackNetV4/TrackNetV4)
- **Dataset licence: tennis dataset is gated by a Research-Only Data Use Agreement (no commercial use, no redistribution).** — [project page](https://tracknetv4.github.io)
- Numbers: the project page and abstract give no figures; the TrackNetV5 paper's table lists V4 at Acc 0.9224 / F1 0.9581 vs V2 0.9396 / 0.9677 on the TrackNetV2 (badminton) dataset, i.e. **V4 below V2 in that (third-party) comparison**. — [TrackNetV5 arXiv HTML](https://arxiv.org/html/2512.02789). An independent reproduction also reported V4+attention underperforming V2+attention (reproducer cautions its eval code may differ). — [search-result summary of hackmd report](https://hackmd.io/@XDU5dRI5RJOkP6SWYyGUQA/HkQA5OZaJg) [secondary source]

**TrackNetV5 (two different papers share the name)**
- (a) arXiv 2512.02789 (Wuhan Univ. of Technology; v4 13 Jan 2026): V2 U-Net + Motion Direction Decoupling (signed positive/negative frame-difference maps) + Residual-Driven Spatio-Temporal Refinement (a small factorised space/time transformer head). Input **3 frames (t-1, t, t+1) -> 13 channels**, 512x288; data: TrackNetV2 badminton set (7:3 split) + a private 1920x1080 "Loveall" set. Result Table 1: V5 Acc 0.9733 / F1 0.9859 (V2 0.9396/0.9677) at ~114 FPS on a T4. Uses frame t+1, so not causal. **No code or weights or licence found on the page.** — [arXiv HTML](https://arxiv.org/html/2512.02789) [code release UNVERIFIED]
- (b) NYCU (Chang, Wang, Huang) "TrackNetV5: Robust Shuttlecock Tracking via Motion Prompts and Spatiotemporal Attentive Fusion", F1 98.47, with a "Lite" variant; found only as a PDF on a university site, arXiv listing not confirmed. — [PDF](https://people.cs.nycu.edu.tw/~yushuen/data/TracknetV5_26.pdf) [details UNVERIFIED]

**Fine-tune effort (inference from the above; no soccer fine-tune of any TrackNet version found)**
- Code in V3 (PyTorch, MIT) trains on ~hundreds of thousands of frames (badminton train set of 78,558 frames per WASB's table) in 30 epochs; training needs per-frame ball *position* labels on consecutive frames and (for V3 as published) a per-video median background. — [paper](https://people.cs.nycu.edu.tw/~yushuen/data/TrackNetV3.pdf), [WASB Table 1](https://arxiv.org/pdf/2311.05237)

### Inferences
- InpaintNet is the only off-the-shelf piece that matches the "hold through occlusion using positions both before and after" requirement, and it is **detached from pixels** (it consumes a coordinate sequence + mask). It could be retrained on our own coordinate tracks (e.g. the human dense runs plus detector output) independent of the heatmap net, and used with any detector. Its fill rule (delta=30 px height test) is shuttlecock/court-specific and would need re-defining for a panning camera (coordinates should be in a camera-compensated frame, otherwise pan velocity dominates).
- V3's median-background input is invalid for a pan/tilt/zoom camera; the "V3 weights" benefit would drop to V2 + mixup. Use the tracking head without background.
- Recall-vs-precision trade of rectification (precision -1.8 pts, recall +3.4 pts on badminton) is a risk when occluded ball positions are not labelled.
- Resolution: at 288x512 a ball of ~11 px at 1280 width becomes ~4-5 px (11 x 512/1280 = 4.4). The TrackNet nets would need to be run at higher input (1280 x ~720 or tiled/cropped) — architecture is fully convolutional so this is possible, but it is an untested change, not a published config. [inference]

### Gaps
- Official V1 / V2 licence and whether weights are bundled: not confirmed.
- TrackNetV4 `docs/RESULT.md` (checkpoint sport and numbers) not read; V4 ICASSP numbers not extracted (PDF too large to fetch).
- TrackNetV5 code/weights: none found for either paper.
- Whether InpaintNet works at long gaps (>1-2 s) — paper does not report gap-length breakdown.

---

## Q2. WASB-SBDT (Widely Applicable Strong Baseline for Sports Ball Detection and Tracking)

### Takeaway
WASB (BMVC 2023, NTT) **does include soccer** and ships soccer weights, reaches F1 ~88 on it — but the soccer set is the ISSIA-CNR-style 1920x1080 25 fps multi-camera dataset, evaluated after resizing to 288x512, and the paper's own ablation says its online tracker does not help on soccer. Code is MIT; best candidate "off-the-shelf" base for fine-tuning because it already supports soccer, the HRNet keeps full resolution, and BlurBall (MIT) is built on it.

### Cited Findings
- Sports: soccer, tennis, badminton, volleyball, basketball. Repo MIT licence; paper arXiv 2311.05237. — [repo](https://github.com/nttcom/WASB-SBDT), [paper](https://arxiv.org/abs/2311.05237)
- Weights: `MODEL_ZOO.md` lists Google Drive weights **per sport including soccer** for DeepBall, DeepBall-Large, BallSeg, TrackNetV2, ResTrackNetV2, MonoTrack and WASB. News item: repo contains "evaluation codes of pretrained WASB models only" (training code was apparently released separately/later; BlurBall repo "adds training scripts ... on top of WASB"). — [MODEL_ZOO](https://raw.githubusercontent.com/nttcom/WASB-SBDT/main/MODEL_ZOO.md), [BlurBall repo](https://github.com/cogsys-tuebingen/blurball) [exact training-code status of the WASB repo UNVERIFIED]
- Architecture: small HRNet, 4 stages, **stem strides removed so features stay at full input resolution**, ~1.5M parameters; input = N=3 consecutive frames concatenated on channels (also a "Step=1" variant that slides by 1 frame; Step=3 is non-overlapping); **input resized to 288x512**; heatmap GT = disk of radius d=2.5 px; Adam, 30 epochs, batch 8; "Hard-to-Localize Sample Mining" from epoch 20. — [WASB paper](https://arxiv.org/pdf/2311.05237)
- Inference: **online (causal) tracker** — predicts next position from previous 3 tracked points with a constant-acceleration model (p̂(t+1)=p(t)+v(t)+a(t)/2), keeps heatmap candidates within a distance threshold of the prediction, picks the highest-confidence. No Kalman/particle filter. Paper: "online tracking does not contribute on the Soccer and Badminton datasets." There is **no bidirectional/offline smoother** in WASB. — [WASB paper](https://arxiv.org/pdf/2311.05237)
- Soccer dataset used: Soccer [19] = D'Orazio et al. 2009 (the "six synchronized videos" ISSIA-CNR style set; link pspagnolo.jimdofree.com), **1920x1080, 25 fps**, 1 game, train 4 clips / 11,994 frames, test 2 clips / 5,999 frames; original ball annotations were "collapsed" so the authors **re-annotated the ball in every frame**. Mean ball displacement 10.4±10.0 (train) / 15.7±13.0 (test) px per frame. **Ball pixel size is not reported for any sport.** — [WASB paper Table 1, Sec 4.1](https://arxiv.org/pdf/2311.05237)
- Soccer results (tau = 4 px, F1 / Acc %): DeepBall 44.5/92.7; DeepBall-Large 44.9/89.5; BallSeg 36.1/92.6; TrackNetV2 86.6/97.7; ResTrackNetV2 84.6/97.4; MonoTrack 85.2/97.4; **WASB Step=3 88.3/97.9; WASB Step=1 88.2/97.9**. Step=1 vs 3 on soccer: AP 86.2 vs 83.6, FPS 23.6 vs 55.7. Other sports for WASB (Step=1): tennis 95.6, badminton 93.1, volleyball 88.0, basketball 82.6 F1. (Note: soccer *accuracy* is dominated by true-negative frames; F1 is the informative number.) — [WASB paper Table 2](https://arxiv.org/pdf/2311.05237)
- Soccer failure case (Appendix B): detections misaligned "due to the noisy background (e.g. player shoes)". — [WASB paper](https://arxiv.org/pdf/2311.05237)
- Compared methods: DeepBall (<0.1 M params; soccer origin), BallSeg, TrackNetV2, ResTrackNetV2, MonoTrack — i.e. single "SBDT" stack baselines, none is bidirectional. — [WASB paper](https://arxiv.org/pdf/2311.05237)
- Derivatives built on WASB: **BlurBall** (Q3). TOTNet uses WASB as a baseline and "WASB official repo" in its code. — [BlurBall repo](https://github.com/cogsys-tuebingen/blurball), [TOTNet repo](https://github.com/AugustRushG/TOTNet)

### Inferences
- WASB at 288x512 mapped onto 1080p soccer means the 1080p ball becomes 3.75x smaller. In our 11 px-at-1280 regime the same net would see a ~4-5 px blob; the paper never states what the ISSIA ball size was, so we cannot tell whether F1 88 transfers. The paper's HRNet is full-resolution so raising input to 720x1280 or 1080x1920 is mechanically straightforward (1.5 M params, small memory) — plausible on 8 GB. [inference, untested]
- Positive for us: 11,994 train frames were enough for F1 ~88 with a 1.5 M-param net; our ~900 dense frames + 356k noisy candidates are the same order of magnitude of *dense consecutive* data only if the noisy candidates can be turned into pseudo-labels.
- WASB's "no benefit from online tracking on soccer" is evidence that for soccer the raw heatmap is already the main signal; the offline requirement we have (bidirectional prior) is something WASB does not provide — we would add our own forward-backward smoother/gap filler.

### Gaps
- Ball pixel size and PTZ/broadcast-ness of the WASB soccer set: not stated in the paper. (Background knowledge, [UNVERIFIED] here: ISSIA-CNR cameras are fixed, stadium-mounted, so no pan/zoom.)
- Training-code availability in the NTT repo itself (README says "evaluation codes of pretrained WASB models only") — check the repo's other branches/sandbox before relying on it.
- Soccer licence of the underlying ISSIA data (redistribution/commercial terms) not checked.

---

## Q3. BlurBall, MonoTrack, TOTNet, BallTrack-style, and other multi-frame ball detectors

### Takeaway
BlurBall (MIT, weights+dataset) and TOTNet (MIT code, research-only data) are both table-tennis-first, 288x512, fixed-camera; TOTNet is the one explicitly designed for *occlusion*, with 5-frame 3D-conv input and a visibility-weighted loss. MonoTrack is badminton-specific 3D reconstruction and the "MonoTrack" baseline inside WASB is just its 2D tracker. For soccer-specific detectors, there is no off-the-shelf pretrained multi-frame tiny-ball model with PTZ broadcast evidence.

### Cited Findings
**BlurBall (arXiv 2509.18387; CVPRW 2026 per repo; Tübingen)**
- Built on WASB (HRNet heatmaps) + Squeeze-and-Excitation attention; jointly estimates ball position and motion-blur length/orientation (PCA on heatmap). ~1.49 M params, 79 FPS (1-step). Input rescaled to 288x512; multi-frame window (1-step vs 3-step compared; exact frame count not stated). — [arXiv HTML v2](https://arxiv.org/html/2509.18387v2)
- Labeling insight: put the ball label at the **centre of the blur streak** rather than leading edge; mid-point labels improved every model tested. — [arXiv HTML](https://arxiv.org/html/2509.18387v2)
- Data/results: 26 online table-tennis recordings, 64,119 frames, **fixed cameras**; test F1: WASB 96.00, TrackNetV2 91.61 (mid labels, 4 px); BlurBall 96.52 (δ=0.5) / 97.17 (δ=0.7). No TrackNetV4 comparison. Blur appears in ~62% of frames; mean inter-frame displacement ~19-20 px. — [arXiv HTML](https://arxiv.org/html/2509.18387v2)
- Stated limitations: "more prone to misdetections with non-static cameras"; false positives on white objects; misses under occlusion; threshold sensitivity (best threshold shifts 0.5->0.7). — [arXiv HTML](https://arxiv.org/html/2509.18387v2)
- Repo: **MIT**, weights for BlurBall/WASB/TrackNetV2/ResTrackNetV2/BallSeg/DeepBall/MonoTrack on Nextcloud, training scripts, dataset (table tennis, 64 k frames) download; paper states source videos chosen under Creative Commons. README: "duplicated frames affect BlurBall". — [repo](https://github.com/cogsys-tuebingen/blurball)
- Relevance: the repo is the most complete *trainable* WASB descendant (training scripts + configs).

**TOTNet (arXiv 2508.09650; Deakin/Paralympics Australia; CVIU 2026 per repo)**
- U-Net with **3D temporal convolutions kept through the encoder** (per-frame 2D conv + 3D conv, temporal dim collapsed at the bottleneck), 1-D x/y heatmaps, visibility-weighted BCE loss, **occlusion augmentation that masks the ball in the target frame**; optional RAFT optical-flow input (OF variant). 8.65 M params, 28 FPS (OF: 12 FPS). — [arXiv HTML](https://arxiv.org/html/2508.09650v1)
- **5 consecutive frames** (ablation 3/5/7/9; no gain past 5; predicting middle vs last frame similar, authors infer past frames suffice); **288x512**; AdamW lr 5e-4, 30 epochs, 2x A100. Authors target offline sports analytics. — [repo](https://github.com/AugustRushG/TOTNet), [arXiv HTML](https://arxiv.org/html/2508.09650v1)
- Results (RMSE px / acc): TTA (table tennis, Paralympic, 25 fps 1080x1920) fully-occluded: TOTNet(OF) 7.19/0.80 vs WASB 37.30/0.63 vs TrackNetV2 100.89/0.51; tennis fully occluded only 5 test samples (RMSE 27.98/0.67 TOTNet; WASB 264.45/0.17); badminton not-visible: TOTNet 44.55/0.84 vs TrackNetV2 32.95/0.88 (TOTNet not uniformly best). Paper text has inconsistencies with its Table 2. 5 px eval threshold "approximates ball's average size". — [arXiv HTML](https://arxiv.org/html/2508.09650v1)
- Licence: code MIT; **TTA dataset research-only, "commercial use and redistribution strictly prohibited"**, gated by a signed access agreement (Hugging Face after approval). Weights: a `weights/` folder exists but README does not document it. — [repo](https://github.com/AugustRushG/TOTNet)

**MonoTrack (CVPRW 2022; Liu & Wang)**
- End-to-end monocular badminton 3D shuttle-trajectory reconstruction (court geometry, pose, hit recognition, physics); a synthetic set of 10 k trajectories. The "MonoTrack" model in WASB's table is only its 2D detector re-implemented; WASB soccer F1 85.2 (Q2). — [paper](https://arxiv.org/abs/2204.01899), [WASB paper](https://arxiv.org/pdf/2311.05237) [official code link UNVERIFIED]

**Other multi-frame / soccer-specific ball work found**
- *Kukleva et al., RoboCup 2019* (arXiv 1909.02406): single-image FCN encoder-decoder whose output feeds TCN / ConvLSTM / ConvGRU temporal models to keep the ball when occluded; evaluated on a new RoboCup dataset (not broadcast). No numbers extracted. — [arXiv](https://arxiv.org/abs/1909.02406)
- *Real-time Localization of a Soccer Ball from a Single Camera* (arXiv 2506.07981, Jun 2025): RTMDet-t 2D detections + multi-mode online filter lifting to 3D by discretised camera rays; ball covers <0.15% of a 6144x3240 frame; ball hidden ~half the time; proprietary data (10 Russian Premier League matches, 25 fps), four-camera triangulated GT (<10 cm); results are 3D Accuracy@threshold not 2D recall; no code/licence mentioned; calibrated per-frame camera required. — [arXiv HTML](https://arxiv.org/html/2506.07981v1)
- *SoccerNet-v3D* (arXiv 2504.10106, 2025): adds 3D ball GT; SoccerNet-v3D 4,051 images (3,240/811), ISSIA-3D 10,544 images; YOLO 2D ball AP@0.5: 0.65 (base) -> 0.81 (optimised) on SoccerNet-v3D; ISSIA-3D 0.04 (base) -> 0.39 (opt) -> 0.65 (fine-tuned on ISSIA). Shows off-the-shelf detectors collapse on the ISSIA set until fine-tuned. Code/pipelines on GitHub (mguti97/SoccerNet-v3D); dataset licence not stated. — [arXiv HTML](https://arxiv.org/html/2504.10106v1)
- *A Computer Vision Framework for Multi-Class Detection and Tracking in Soccer Broadcast Footage* (arXiv 2602.18504, Feb 2026): YOLO + ByteTrack; abstract says ball detection remains the main difficulty (precision/recall/mAP50 quoted only for players/officials). No ball numbers, code not linked. — [arXiv](https://arxiv.org/abs/2602.18504)
- *PLUCC: Ice Hockey Puck Localization Using Contextual Cues* (arXiv 2506.04365; the "player-context" idea closest to the brief): single-frame (no temporal), RGB + player-orientation/position context image at half-res, feature-pyramid + gating decoder, Gaussian heatmap with random context drop-out (p=1%); puck is ~0.005% of frame pixels; reports +12.2% AP over baselines on its PuckDataset (not public detail extracted); explicitly contrasts with a prior 9-frame temporal heatmap method that "struggles with the puck's inherent small size". No code link seen. CC BY 4.0 is the paper's. — [arXiv abs](https://arxiv.org/abs/2506.04365), [PDF](https://arxiv.org/pdf/2506.04365)
- *SoccerEDMF* (Visual Computer 2025, dual-model player tracking + ball detection with interpolation through occlusion) — only the search summary read; no numbers. — [Springer](https://link.springer.com/article/10.1007/s00371-025-04118-x) [UNVERIFIED details]
- *BallTrack-style* (named in the brief): I found no distinct, citable "BallTrack" model; nothing verified under that name. [UNVERIFIED / not found]

### Inferences
- If we want a pretrained, soccer-trained starting point, WASB soccer weights (ISSIA, fixed camera) are the only published one; TOTNet/BlurBall give *code* and training recipes but table-tennis weights.
- TOTNet's occlusion augmentation (mask the ball in the target frame so the net must use neighbours) is directly reusable as a training recipe in any frame-stack net to teach "hold through occlusion"; it is independent of the architecture.
- BlurBall's "label at blur-centre" is relevant when ground-truth boxes are loose — at 25 fps a kicked ball in 1080p broadcast will be blurred/elongated; our human labels should follow a consistent convention. [inference]
- PLUCC suggests the player-context route exists for tiny objects in broadcast, but only in single-frame form and for hockey; no soccer equivalent found.

### Gaps
- TOTNet checkpoint documentation; BlurBall frame-window size; MonoTrack official code.
- Kukleva 2019 numbers and dataset; SoccerEDMF numbers.
- No soccer broadcast heatmap tracker published 2024-2026 found (searches: arXiv, general web) — see Q5.

---

## Q4. SoccerNet challenges: do winners track the ball, and is there a public dense ball dataset?

### Takeaway
Among SoccerNet challenges only **Tracking (2022/2023)** includes the ball in its GT and metric; its top solutions treat the ball as a (poor) single-object detection problem plus smoothing/interpolation. Game State Reconstruction (2024/2025) and Ball Action Spotting concern players/actions, not ball position. SoccerNet-Tracking does contain ball tracks (297 tracklets, 215,156 boxes) but they are bounding boxes interpolated between key frames, 1080p 25 fps, and one source reports label noise.

### Cited Findings
- SoccerNet-Tracking (CVPRW 2022): 200 thirty-second clips from 12 games (Swiss Super League 2019, main camera) + one full half; classes player, goalkeeper, referee, **ball**, other. Ball: 297 unique tracklets, 215,156 bounding boxes (of 3,645,661 total). Boxes are drawn at key frames and **linearly interpolated** (only 11.5% of boxes interpolated overall; stated as dense manual annotation). — [paper](https://arxiv.org/pdf/2204.06918) (Table 2 + Sec 3), [search summary re 12 games Swiss Super League](https://arxiv.org/pdf/2504.10106)
- Baseline observation in the dataset paper: a CrowdHuman-pretrained detector "never detects the ball"; fine-tuning to humans + ball reduces FP/FN and improves HOTA/MOTA. Per-class analysis: errors rise for fast ball/shot events. — [SoccerNet-Tracking paper](https://arxiv.org/pdf/2204.06918)
- Splits in `sn-tracking`: 57 train clips, 49 test, 58 first-challenge (2022); the 2023 challenge set is another 50 clips; the repo says classes are **not taken into account in the challenge/evaluation** and the label CSV has no class column (ball vs player not distinguished in the released GT format). Download via `pip install SoccerNet` (SoccerNetDownloader), tasks `tracking` and `tracking-2023`. Licence not stated on the pages read (SoccerNet video data historically requires an NDA — [UNVERIFIED here]). — [sn-tracking](https://github.com/SoccerNet/sn-tracking), [tracking page](https://www.soccer-net.org/tasks/tracking)
- 2023 Tracking challenge 3rd place (MOT4MOT, HOTA 66.27): ball tracking "formulated as single object detection"; fine-tuned **YOLOv8l at 1080x1920 with confidence 0.05**; detector trained on ball boxes from **2,084 frames** of the training set after filtering label errors with a COCO detector (IoU>0.7 with label); then: take most-confident detection per frame -> smooth with **3rd-order polynomial over a 51-frame window** -> keep detection closest to the curve if < 100 px -> linear interpolation of gaps. Ball detector AP@0.5 0.95, AP@0.5:0.95 0.71 on 200 manually inspected frames; struggles with partial occlusion and ball blending with crowd/white shoes; GT labels contain errors ("labels not modified"). — [MOT4MOT report](https://arxiv.org/pdf/2308.16651)
- Game State Reconstruction 2025 (SoccerNet 2025 report, arXiv 2508.19182): winner GSR-1 GS-HOTA 63.90 (baseline 29.01); pipeline YOLO-X + Deep-EIoU + OSNet ReID, multi-frame keypoint pitch mapping, LLaMA-3.2-Vision for role/jersey/colour. **Outputs are athletes only; no ball.** sn-gamestate roles: player, goalkeeper, referee, other; repo GPL-3.0; 200 annotated clips. — [SoccerNet 2025 results](https://arxiv.org/html/2508.19182v1), [sn-gamestate](https://github.com/SoccerNet/sn-gamestate)
- Team Ball Action Spotting 2025: winner TBAS-1 Team-mAP@1 60.03; 12 action classes; method is T-DEED-style temporal action detector pretrained on 500+ broadcast matches. It predicts *action events*, not ball pixel position. — [SoccerNet 2025 results](https://arxiv.org/html/2508.19182v1). 2026 edition ("Player-Centric Ball Action Spotting") builds on per-player Track-Aware Action Detectors + a sequence transformer, 58.94 Macro-F1 for one submission; ball position not used. — [arXiv 2606.28389](https://arxiv.org/abs/2606.28389)
- SoccerTrack v2 (arXiv 2508.01802): 10 university matches, 932 min, 4K panoramic fixed camera; **ball action events** linked to players; a ball track appears as an input to one baseline but per-frame ball position annotations were not described. — [arXiv](https://arxiv.org/abs/2508.01802) [ball-position labels UNVERIFIED]
- SoccerNet-v3 / v3D: spatial annotations include ball boxes on replays/frames (960x540 to 1920x1080); v3D gives 4,051 images (single images, not dense sequences). — [SoccerNet-v3D](https://arxiv.org/html/2504.10106v1)

### Inferences
- Ball boxes in SoccerNet-Tracking are the nearest public *dense* broadcast-soccer ball data (≈215 k boxes at 1080p, same camera type: main broadcast, PTZ, 25 fps — [inference: "main camera" implies PTZ broadcast; not confirmed in sources read]). The MOT4MOT report shows they are noisy enough that the winning team filtered them with a second detector; only ~2 k frames survived filtering — which is comparable to our own noisy-candidate problem and suggests the same filtering recipe.
- Box-size distribution (hence ball pixel size) for SoccerNet-Tracking not extracted; if ball boxes are ~10-20 px at 1080p they are close to or larger than our 11 px-at-1280 target; check before use.
- The winners' ball recipe (high-resolution single-image detector at low confidence, then trajectory-level polynomial smoothing and interpolation) is essentially the "no heatmap tracker" baseline; no SoccerNet challenge winner found that uses TrackNet/WASB-style multi-frame input for the ball.

### Gaps
- Ball box size statistics, and whether test/challenge ball GT is public (sn-tracking README says classes are ignored in evaluation).
- SoccerNet data licence / NDA terms for Tracking videos.
- 2022 and 2024 Tracking/GSR winner reports' ball handling: not read (only 2023 3rd place read).
- 2026 SoccerNet challenge set (whether a ball-tracking task was added): only the Ball Action Spotting entry found.

---

## Q5. Tiny balls at broadcast-wide-shot scale: who handles them, reported failures

### Takeaway
No model in this set has published evidence at "11 px ball, PTZ broadcast wide shot". The closest published numbers are: WASB soccer F1 ~88 (fixed 1080p cameras, resized to 288x512; ball size unreported), SoccerNet-Tracking YOLOv8l AP50 0.95 / AP50:95 0.71 (full 1080p input), and hockey puck PLUCC (0.005% of frame). Explicit failure statements are about non-static cameras, tiny balls and low-contrast confusers.

### Cited Findings
- BlurBall: "more prone to misdetections with non-static cameras"; fixed-camera datasets only. — [arXiv](https://arxiv.org/html/2509.18387v2)
- TrackNetV3: background estimation depends on fixed viewpoint; evaluated on fixed-camera badminton. — [paper](https://people.cs.nycu.edu.tw/~yushuen/data/TrackNetV3.pdf)
- WASB paper on basketball: "camera frequently moves and zooms in rapidly ... complex ball trajectory"; WASB basketball F1 82.6 (lowest) with mean displacement 33.7 px/frame; soccer F1 88.2. So moving-camera data exists in WASB (basketball) but with lower F1. — [WASB paper](https://arxiv.org/pdf/2311.05237)
- WASB soccer failure example: false positives on player shoes (noisy background). — [WASB paper](https://arxiv.org/pdf/2311.05237)
- MOT4MOT (SoccerNet 2023): ball merges with crowd/white shoes; partial occlusion failures; they needed 1080x1920 input. — [report](https://arxiv.org/pdf/2308.16651)
- PLUCC (hockey): puck ≈0.005% of frame pixels; a prior temporal 9-frame method "struggles with the puck's inherent small size", motivating context cues. — [arXiv](https://arxiv.org/pdf/2506.04365)
- 2506.07981: ball <0.15% of a 6K frame and hidden ~50% of the time; used a plain YOLO-family detector (RTMDet-t) rather than a heatmap tracker, with filter-level reasoning for gaps. — [arXiv HTML](https://arxiv.org/html/2506.07981v1)
- SoccerNet-v3D: YOLO AP@0.5 on ISSIA-3D 0.04 off-the-shelf, 0.65 only after dataset-specific fine-tuning. — [arXiv HTML](https://arxiv.org/html/2504.10106v1)
- Our own KB: XS-VID finds DETR-family detectors under-perform on tiny video objects (0-12² px) and the best is a YOLO with RAFT-style correlation to the previous frame's features. — [KB footy-ball-detection-findings](file:///home/george/knowledge/wiki/footy-ball-detection-findings.md)

### Inferences
- Moving camera: frame-difference channels (V4/V5 motion maps) and median-background (V3) assume a stationary camera or at least small camera motion; on pan/tilt/zoom they encode camera motion rather than ball motion. WASB/TOTNet/BlurBall use raw stacked frames (or flow in TOTNet-OF) and are less dependent on this assumption, though none has been tested with a PTZ camera at soccer scale. WASB's basketball set (moving/zooming camera, F1 82.6) is the nearest evidence.
- Because every candidate model downsamples to 288x512, the effective ball at our scale would be ~4 px at 288x512; recommended experiments are those that raise input resolution (or crop around candidate regions) while retaining the multi-frame stack.

### Gaps
- No paper measured ball-size sensitivity (F1 vs ball px) for TrackNet/WASB.
- No ablation of input resolution above 288x512 for any of these models found.

---

## Q6. Licences summary (flag non-commercial / research-only)

### Takeaway
All code repositories checked are MIT, **but datasets are the traps**: TrackNetV4's tennis set and TOTNet's TTA set are research-only/no redistribution; sn-gamestate is GPL-3.0 (copyleft); SoccerNet video data may be NDA-gated.

### Cited Findings
| Item | Licence / terms | Source |
|---|---|---|
| WASB-SBDT code (+ weights per MODEL_ZOO) | MIT (LICENSE.md) | [repo](https://github.com/nttcom/WASB-SBDT) |
| WASB soccer data (ISSIA-CNR-type set, D'Orazio et al.) | not stated in paper; check original dataset terms | [paper](https://arxiv.org/pdf/2311.05237) [UNVERIFIED] |
| TrackNetV3 code + checkpoints | MIT (stated to cover checkpoints) | [repo](https://github.com/qaz812345/TrackNetV3) |
| TrackNetV4 code | MIT | [repo](https://github.com/TrackNetV4/TrackNetV4) |
| TrackNetV4 tennis dataset | **Research-only data use agreement; no commercial use, no redistribution** | [project page](https://tracknetv4.github.io) |
| TrackNetV5 (arXiv 2512.02789) | no code/licence found | [arXiv HTML](https://arxiv.org/html/2512.02789) |
| TrackNetV1/V2 official | unknown (NCTU site) | [UNVERIFIED] |
| BlurBall code + weights | MIT (README); dataset terms unstated, source videos CC | [repo](https://github.com/cogsys-tuebingen/blurball), [paper](https://arxiv.org/html/2509.18387v2) |
| TOTNet code | MIT | [repo](https://github.com/AugustRushG/TOTNet) |
| TOTNet TTA dataset | **Research only; "Commercial use and redistribution are strictly prohibited"; signed agreement** | [repo](https://github.com/AugustRushG/TOTNet) |
| sn-gamestate / SoccerNet GSR code | **GPL-3.0** | [repo](https://github.com/SoccerNet/sn-gamestate) |
| SoccerNet-Tracking data | not stated on pages read; historically NDA for videos | [sn-tracking](https://github.com/SoccerNet/sn-tracking) [UNVERIFIED] |
| SoccerNet-v3D | code on GitHub; dataset licence not stated | [paper](https://arxiv.org/html/2504.10106v1) |

### Inferences
- If the end use is commercial, the code (MIT) is fine but any pretrained weights trained on research-only datasets (e.g. TOTNet's TTA, V4's tennis set) should be treated as tainted until the dataset terms are checked; WASB's weights derive from the soccer/tennis/badminton/volleyball/basketball datasets above whose terms were not verified.

### Gaps
- Dataset-level licences for WASB's component datasets, SoccerNet-Tracking, and Ultralytics (not in scope but relevant to current stack).

---

## Cross-cutting summary table (for report writer)

| Model | Sport of released weights | Frames in | Input res | Offline/bidirectional gap fill | Code licence | Soccer evidence |
|---|---|---|---|---|---|---|
| TrackNetV2 | tennis/badminton (community) | 3 | 288x512 | no | unknown/MIT ports | WASB-soccer F1 86.6 |
| TrackNetV3 | badminton | 8 | 288x512 | **InpaintNet, trajectory-level, uses both sides of gap** | MIT | none |
| TrackNetV4 | tennis/badminton (.keras) | 3 | 288x512 | no | MIT (data research-only) | none |
| TrackNetV5 (2512.02789) | none found | 3 (t-1,t,t+1) | 512x288 | no | none found | none |
| WASB (HRNet, 1.5M) | **soccer, tennis, badminton, volleyball, basketball** | 3 (step 1 or 3) | 288x512 | no (causal online tracker) | MIT | **F1 88.2-88.3 (ISSIA-type, 1080p)** |
| BlurBall | table tennis | multi (1 or 3 step) | 288x512 | no | MIT | none |
| TOTNet | table tennis (weights undocumented) | 5 | 288x512 | no, but occlusion-trained | MIT (data research-only) | none |
| 2506.07981 (RTMDet-t + filter) | n/a | 1 | not stated | online 3D filter | none | soccer 3D, proprietary data |
| SoccerNet 2023 MOT4MOT | n/a | 1 (+polynomial smoothing) | 1080x1920 | smoothing + linear interpolation | not stated | SoccerNet-Tracking ball AP50 0.95 |
