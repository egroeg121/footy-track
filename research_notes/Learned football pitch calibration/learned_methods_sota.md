# Learned sports-field camera calibration / pitch registration for broadcast football (state as of 2026-11-14)

Scope note: KB search (`mcp__knowledge__search`, "camera calibration pitch homography SoccerNet") returned no calibration pages; nothing from the KB was relied on. All numbers below come from papers/READMEs I opened (full-text PDFs read via pdftotext where marked). Older (2017-2022) results are labelled as older. Benchmark-protocol caveat that applies everywhere: WC14-test has only 186 images and TS-WorldCup test is from the same 2014/2018 World Cup broadcasts, so those numbers are narrow-domain; SoccerNet-calibration (SN) has no ground-truth camera parameters, only annotated field polylines.

Abbreviations: SV = single-view model, MV = multi-view model, CR = completeness rate, FS = CR x JaC@5, SN22-center = main-camera-centre subset (1454 imgs), SN23-test = full 2023 test (all camera types), SN-GSR = SoccerNet Game State Reconstruction video sequences (1080p).

---

## 1. What does each method output (homography vs full camera, lens distortion)?

### Takeaway
Only the newer SoccerNet-era methods (TVCalib, NBJW/PnLCalib, Sportlight, Spiideo, BroadTrack, Theiner 2026) output a full pinhole camera (intrinsics + extrinsics); the 2017-2022 academic line (Homayounfar, Chen & Little, Sha, Jiang, Nie, KpSFR) output a 2D homography or a PTZ pose with principal point at image centre. Lens distortion is explicitly modelled by very few: TVCalib (optional k1,k2, off by default, flaky at low FoV), PnLCalib (added 2026-03, undocumented), BroadTrack (k1, large gain), Theiner CVPRW 2026 (k1, joint), and some GSR entrants (lianyou used an external radial-distortion correction + PnLCalib).

### Cited Findings
- NBJW: heatmap-regression of keypoints and line extremities (HRNetV2-w48), then classical DLT "without further refinement" to a 3D camera; principal point assumed at image centre, distortion neglected; works "as long as ... distortions are not too harsh (i.e. fisheye shots) and a minimum of four keypoints are visible". — [NBJW arXiv v1 (pdf, read)](https://arxiv.org/pdf/2404.08401v1); [NBJW repo](https://github.com/mguti97/No-Bells-Just-Whistles)
- PnLCalib (extended NBJW): adds a non-linear points-and-lines (PnL) refinement on detected field lines; outputs single-view and multi-view 3D camera parameters + homography; paper says distortion is "not modelled" and left as future work; but README update 2026-03-10 says "Lens Distortion optimization added into utils/utils_calib_wp.py" (no benchmark numbers for it found). Published in Computer Vision and Image Understanding 267 (April 2026), 104712. — [PnLCalib arXiv v5 HTML](https://arxiv.org/html/2404.08401v5); [PnLCalib repo](https://github.com/mguti97/PnLCalib); [arXiv abs](https://arxiv.org/abs/2404.08401)
- TVCalib (Theiner & Ewerth, WACV 2023): DeepLabv3-ResNet101 instance segmentation of field segments, then gradient-descent minimisation of a differentiable segment reprojection loss to get 6-DoF pose + focal length; initialised from three camera positions (main tribune only); optional radial lens distortion (k1,k2, kornia) learned jointly — "results can be improved at AC@5 for samples where radial lens distortion is visible" but appendix reports local minima / FoV "explosion" for low-FoV images; the paper lists distortion as "not deeply investigated". — [TVCalib arXiv 2207.11709 (pdf, read)](https://arxiv.org/pdf/2207.11709); [repo](https://github.com/MM4SPA/tvcalib) (has `--distort` evaluation flag, ignored by default)
- SoccerNet challenge task definition allows pinhole + optional tangential/radial/thin-prism distortion; evaluation is reprojection of annotated field polylines, no GT camera parameters. — [SoccerNet 2023 results (pdf, read)](https://arxiv.org/pdf/2309.06006)
- Sportlight (2023 winner): HRNetV2-w48 keypoint heatmaps (57 pts incl. ellipse tangent points) + 23 line detectors (two-peak heatmaps); OpenCV calibration on point subsets, final params by reprojection-error voting; outputs full camera. — [2023 report](https://arxiv.org/pdf/2309.06006); [repo README](https://github.com/NikolasEnt/soccernet-calibration-sportlight)
- Spiideo (2023 #2): DeepLabV3 segmentation into field-part classes, then differentiable rendering (pytorch3d SoftRasterizer, AdamW) initialised from 20 k-means clusters of baseline cameras; full camera; distortion not mentioned. — [2023 report](https://arxiv.org/pdf/2309.06006); [repo](https://github.com/Spiideo/soccersegcal)
- 2022 winner "achengmao" (Baidu): detects/segments lines and conics, computes five landmark types (line-line, conic-line intersections, field centre, vanishing point, curve points), calibrates; full camera. — [SoccerNet 2022 results (pdf, read)](https://arxiv.org/pdf/2210.02365)
- KpSFR (Chu et al., CVPRW 2022): dense uniform grid of keypoints, keypoint detection as instance segmentation with dynamic filters conditioned on image + keypoint identity; homography via NMS -> field-model correspondences -> RANSAC (reproj. thr 10) + DLT. Homography only, no distortion. — [KpSFR paper (pdf, read)](https://openaccess.thecvf.com/content/CVPR2022W/CVSports/papers/Chu_Sports_Field_Registration_via_Keypoints-Aware_Label_Condition_CVPRW_2022_paper.pdf); [repo](https://github.com/ericsujw/KpSFR)
- Nie et al. (WACV 2021, Amazon): multi-task network predicts uniform-grid keypoints + dense field features (distance map to nearest line/region); keypoint homography then optional online optimisation refinement using dense features. Homography only. — [Nie paper (pdf, read)](https://openaccess.thecvf.com/content/WACV2021/papers/Nie_A_Robust_and_Efficient_Framework_for_Sports-Field_Registration_WACV_2021_paper.pdf)
- Chen & Little (CVPRW 2019; arXiv 1810.10658 "Sports Camera Calibration via Synthetic Data"): synthetic camera-pose engine (3 free params f, pan, tilt; principal point at centre, no distortion) generates edge images; siamese net builds a feature-pose database; two-GAN model detects field markings in real images; database retrieval + refinement with distance images. PTZ-style camera, not full intrinsics/distortion. — [arXiv pdf (read)](https://arxiv.org/pdf/1810.10658)
- Jiang et al. (WACV 2020): initial registration net -> homography; a second "registration error" network regresses IoU-type error and the homography is refined by gradient descent through that net. Homography only. — [arXiv 1909.08034 (pdf, read)](https://arxiv.org/pdf/1909.08034)
- Citraro et al. (ECCV 2020 / arXiv 2003.14109, EPFL/SecondSpectrum): fully-convolutional net locates semantic keypoints and player centres; particle-filter pose tracking over time; outputs focal length + extrinsics (camera pose) per frame; needs sequences; separate networks per field side. — [arXiv pdf (read)](https://arxiv.org/pdf/2003.14109)
- Homayounfar et al. (CVPR 2017): deep segmentation -> MRF over field hypotheses solved by branch-and-bound; homography. Older, search-based. — [CVPR 2017 pdf (read)](https://openaccess.thecvf.com/content_cvpr_2017/papers/Homayounfar_Sports_Field_Localization_CVPR_2017_paper.pdf)
- Sha et al. (CVPR 2020): single network with area-based semantic segmentation, siamese-network camera-pose template initialisation, spatial-transformer homography refinement; no lens distortion ("no lens distortion" assumed). — [Sha CVPR 2020 pdf (read)](https://openaccess.thecvf.com/content_CVPR_2020/papers/Sha_End-to-End_Camera_Calibration_for_Broadcast_Videos_CVPR_2020_paper.pdf)
- SoccerNet GSR baseline (sn-gamestate / tracklab): calibration + pitch localisation originally TVCalib (segmentation + camera optimisation); PnLCalib and NBJW added as drop-in options 2024-07-10; the current main-branch `soccernet.yaml` sets both `modules/pitch` and `modules/calibration` to `nbjw_calib`. Baseline calibration therefore inherits NBJW outputs (3D camera -> homography; bbox bottom assumed on ground). — [sn-gamestate repo](https://github.com/SoccerNet/sn-gamestate); [GSR paper (pdf, read)](https://arxiv.org/pdf/2404.11335); config seen via GitHub API at https://github.com/SoccerNet/sn-gamestate/blob/main/sn_gamestate/configs/soccernet.yaml
- BroadTrack (Magera et al., WACV 2025): tracking system on top of existing field detectors; camera model with focal, radial k1, pan/tilt/roll, camera-centre position under a tripod prior; optical-flow tracking; ablation: adding k1 raised JaC5 from 42.09 to 54.1 (no OF, no tripod) on SN-GSR HD. — [BroadTrack arXiv 2412.01721 (pdf, read)](https://arxiv.org/pdf/2412.01721)
- Theiner et al. CVPRW 2026 "Unified Sports Field Registration with Lens Distortion Modeling": joint camera + k1 estimation from pixel-level point annotations per visible segment (no sport-specific keypoints, no synthetic correspondences), single frame / multi-frame / video; "source code will be made available" (not yet confirmed released). — [paper (pdf, read)](https://openaccess.thecvf.com/content/CVPR2026W/CVsports/papers/Theiner_Unified_Sports_Field_Registration_with_Lens_Distortion_Modeling_CVPRW_2026_paper.pdf)
- Constructor Tech (2024 GSR winner): SegFormer-based "Camera Parameters Network" regresses 7 params (x,y,z,pan,roll,tilt,fov) from 512x288 input (5M params), refined by ResNet-18 74-keypoint detector (480x270) with brute-force delta search over camera parameters; Savitzky-Golay smoothing. — [Constructor paper (pdf, read)](https://arxiv.org/pdf/2504.06357); [2024 report](https://arxiv.org/html/2409.10587v1)
- Broadcast2Pitch / KIST (2025 GSR winner, WACV 2026): EfficientNetV2-S attention U-Net predicting 97 keypoint + 18 line heatmaps; homography solved by optimisation over line, circle-conic and keypoint-fallback constraints; homography output. — [Broadcast2Pitch pdf (read)](https://openaccess.thecvf.com/content/WACV2026/papers/Oo_Broadcast2Pitch_Game_State_Reconstruction_from_Unconstrained_Soccer_Videos_WACV_2026_paper.pdf)
- The GSR paper's supplementary describes annotating GSR camera parameters with a pinhole model "augmented with one radial distortion coefficient that may be needed in wide camera shots" (this appears to describe the GSR dataset's ground-truth/annotation procedure, not the baseline model). — [GSR paper](https://arxiv.org/pdf/2404.11335)

### Inferences
- For projecting detections to a 2D pitch you only need a homography (feet on plane); the full camera matters only if you want ball height/offside/3D, or to smooth over time with physically plausible focal/pose.
- Distortion matters on wide shots at HD: BroadTrack and Theiner both report measurable JaC5 gains from k1; NBJW/PnLCalib-without-distortion has a ceiling on wide frames.

### Gaps
- No benchmark numbers found for PnLCalib's new (2026) distortion option.
- Whether Sha 2020 / Citraro 2020 / Homayounfar 2017 / Chen & Little code is public: not found in this pass (see Q5).

---

## 2. Reported numbers (SoccerNet challenges, SN-calibration, WC14, TS-WorldCup)

### Takeaway
On the SoccerNet 2023 challenge set the best entry was Sportlight (combined 0.55, ACC@5 73.22, CR 75.59). The strongest openly-released, reproducible stack in late 2026 is PnLCalib (SN23-test MV+PnL JaC@5 78.7; SN22-center SV+PnL 80.6; WC14 SV+PnL JaC@5 85.9, IoU_whole 93.4, 0.60 m projection error; TSWC 96.3 IoU_whole, 0.23 m). On video sequences (SN-GSR, HD), temporal methods (BroadTrack 56.9 JaC5, Broadcast2Pitch 69.4) clearly beat per-frame NBJW (37.1) -- but protocols differ.

### Cited Findings

**SoccerNet challenge results by year**
- 2022 calibration challenge (first edition, 6 teams, 63 submissions): winner achengmao (Baidu) AC@5 82.38 / AC@10 94.80 / AC@20 96.33 / CR 72.61 / final score 83.96; L3S (= TVCalib, Theiner & Ewerth) 57.83 / 81.42 / 90.74 / CR 69.32 / 66.58; MikeAzatov 62.25 / 84.32 / 90.56 / CR 56.41 / 66.45; baseline 12.94 / 29.14 / 43.48 / CR 58.95 / 21.00. Final score s = (1 - e^(-4 CR)) (0.5 AC@5 + 0.35 AC@10 + 0.15 AC@20). Different metric and test set from 2023: not comparable (older). — [SoccerNet 2022 results (pdf, read)](https://arxiv.org/pdf/2210.02365)
- 2023 calibration challenge (7 teams, 66 submissions; dataset 25,506 images, challenge set 2,690 images; ranking = ACC@5 x completeness): Sportlight 0.55 (ACC@5 73.22, CR 75.59); Spiideo 0.53 (52.95, 99.96); SAIVA_Calibration 0.53 (60.33, 87.22); BPP 0.50 (69.12, 72.54); ikapetan 0.43 (53.78, 79.71); NASK 0.41 (53.01, 77.81); MA&JT 0.41 (58.61, 69.34); baseline 0.08 (13.54, 61.54). — [SoccerNet 2023 results (pdf, read)](https://arxiv.org/pdf/2309.06006)
- 2024: no standalone calibration challenge. The 2024 report lists Ball Action Spotting, Dense Video Captioning, Multi-View Foul Recognition and GSR; calibration appears only inside GSR (winner Constructor.tech, GS-HOTA 63.81, GS-DetA 49.52, GS-AssA 82.23; UPCxMobius GSR-2 swapped in NBJW; GSR-14 patched a TVCalib loss bug and added temporal filtering). — [SoccerNet 2024 results](https://arxiv.org/html/2409.10587v1)
- 2025: also no standalone calibration challenge found; GSR winner KIST (GS-HOTA 63.90; 2025 table also lists Constructor.Tech 63.81 second -- as printed). Calibration approaches named: NBJW, PnLCalib, BroadTrack; new this year: optical-flow temporal smoothing. 3rd-place lianyou: custom RT-DETR detecting 27 line classes x 16 points, radial-distortion correction + PnLCalib, parameters simplified to [pan, tilt, roll, focal] and smoothed. GSR-13 (HTWK Leipzig) tracks NBJW keypoints with Lucas-Kanade flow, initialised from frames where a U-Net pitch segmentation agrees (IoU) with the NBJW-projected pitch; reports "up to 4%" improvement over NBJW baseline. — [SoccerNet 2025 results (pdf, read)](https://arxiv.org/pdf/2508.19182)
- 2024-2025 GSR paper on the baseline: GSR-Baseline GS-HOTA 22.26 (full pipeline), TVCalib module 7.6 FPS (calibration, batch 512) and 2.9 FPS (pitch segmentation, batch 16); oracle-substitution row values 51.39 (calibration) and 49.99 (pitch) GS-HOTA. — [GSR paper](https://arxiv.org/pdf/2404.11335)

**Camera-calibration metrics on SoccerNet (JaC@t = fraction of annotated field polylines whose projection lies within t px of annotation; images 540p for SN22/SN23)** (PnLCalib journal version, Table I, in %)
- SN22-test-center: Chen&Little+HDecomp 34.4/64.6/81.3 (CR 66.6, FS 22.9); TVCalib(tau) 57.6/81.7/93.2 (CR 93.7, FS 53.9); TVCalib 54.8/78.5/90.4 (CR 100); NBJW 63.9/80.7/86.3 (CR 100, FS 63.9; this is PnLCalib's reference [42]); PnLCalib SV 74.4/88.8/92.6 (CR 99.1, FS 73.8); SV+PnL 80.6/91.6/93.7 (CR 98.7, FS 79.5). — [PnLCalib v5 HTML](https://arxiv.org/html/2404.08401v5)
- SN23-test (all camera types): Sportlight [16] 76.6 (CR 73.6, FS 56.3); PnLCalib MV 73.1/86.1/89.6 (CR 80.1, FS 58.6); MV+PnL 78.7/89.6/91.9 (CR 78.4, FS 61.8). — same source
- NBJW original (CVPRW 2024, v1): SN23-test MV 73.7/86.7/90.4 (CR 77.5, FS 57.1) described as the first public multi-view broadcast benchmark; SN22-center SV 75.3/89.4/91.1 (CR 97.8, FS 73.7); WC14-test SV 80.4/91.1/94.2 (CR 99.5, FS 80.0). — [NBJW arXiv v1 (read)](https://arxiv.org/pdf/2404.08401v1)
  - Note: v1 (NBJW) and v5 (PnLCalib) tables differ in the SV rows (e.g. SN22-center 75.3 vs 74.4); the v5 numbers are the later model. Flag when quoting.
- WC14-test (186 imgs), PnLCalib Table II: Chen&Little+HDecomp 32.7/67.3/87.3; Jiang+HDecomp 36.9/66.4/83.9; TVCalib 39.9/71.9/90.5 (tau variant 41.3/73.6/91.4, CR 95.7); NBJW 59.4/82.3/90.9; PnLCalib SV 77.6/89.8/93.7; SV+PnL 85.9/93.7/95.8 (CR 100). — [PnLCalib v5](https://arxiv.org/html/2404.08401v5)
- Other SN23-test entries as quoted in Theiner 2026 (540p): NeRF-based [17] 75.8/86.9 (CR 78.3); lianyou 77.3 (CR 97.2, from benchmark website, unvalidated); sportlight.ai 76.7 (CR 73.4); soccersegcal 54.3 (CR 100); Two-point [Magera] 75.3/86.8 (CR 69.8); NBJW 73.7/86.7 (CR 77.5); Theiner 2026 with NBJWmv localisation 76.0/85.5 (CR 74.1), with GT segments 81.3/88.5 (CR 87.1). — [Theiner CVPRW 2026](https://openaccess.thecvf.com/content/CVPR2026W/CVsports/papers/Theiner_Unified_Sports_Field_Registration_with_Lens_Distortion_Modeling_CVPRW_2026_paper.pdf)
- Theiner 2026 SN22-center with NBJWsv localisation: 80.0/92.4 (CR 96.8) without distortion correction, 82.8/92.5 (CR 98.0) with k1 (vs NBJW 75.3/89.4); WC14: 74.4/92.6 (CR 95.2) -> 82.1/96.1 (CR 93.5) with k1 (NBJW WC14 80.4/91.1). Same paper. — same
- SN-GSR video sequences, 1080p (JaC5/JaC10/CR; harsher in pixels than 540p): TVCalib 19.88/50.42/99.93 (MRE 12.4 px); NBJW 37.14/68.24/93.67 (MRE 10.28); BroadTrack 56.88/79.79/100 (MRE 5.02, MedRE 2.37); Broadcast2Pitch 69.38/92.84/100 (MRE 4.47, MedRE 2.21; their own implementation/evaluation); Theiner 2026 (1 FPS eval) 47.8/77.6 CR 94.0 with NBJWsv localisation, 67.0/91.6 with GT localisation. — [BroadTrack](https://arxiv.org/pdf/2412.01721); [Broadcast2Pitch](https://openaccess.thecvf.com/content/WACV2026/papers/Oo_Broadcast2Pitch_Game_State_Reconstruction_from_Unconstrained_Soccer_Videos_WACV_2026_paper.pdf); [Theiner 2026]
- Two-point reinitialisation (BroadTrack) on SN23 (960x540): 75.25/86.8 CR 69.8 vs TVCalib 52.9/73.4 CR 66.5 and NBJW 73.7/86.7 CR 77.5. — [BroadTrack](https://arxiv.org/pdf/2412.01721)

**Homography metrics: WorldCup 2014 (WC14) and TS-WorldCup (TSWC)** -- IoU_part, IoU_whole mean/median in %, projection error (m) mean/median, reprojection error (normalised) mean/median. Sources: PnLCalib v5 Table III (https://arxiv.org/html/2404.08401v5), KpSFR Tables 1-2 (https://openaccess.thecvf.com/content/CVPR2022W/CVSports/papers/Chu_Sports_Field_Registration_via_Keypoints-Aware_Label_Condition_CVPRW_2022_paper.pdf), NBJW v1 Table 3.

| Method (year) | WC14 IoU_part | WC14 IoU_whole | WC14 proj. m | WC14 reproj | TSWC IoU_part | TSWC IoU_whole | TSWC proj. m |
|---|---|---|---|---|---|---|---|
| Homayounfar 2017 | n/a | 83 / - | n/a | n/a | - | - | - |
| Sha 2020 | 93.2 / 96.1 | 88.3 / 92.1 | n/a | n/a | - | - | - |
| Chen & Little 2019 | 94.5 / 96.1 | 89.4 / 93.8 | n/a | n/a | 96.8/97.4 (ft) | 90.7/94.1 (ft) | 0.54/0.38 (ft) |
| Jiang 2020 | 95.1 / 96.7 | 89.8 / 92.9 | 1.21 / 0.74 | 0.017 / 0.012 | 94.8/95.0 (no ft) | 88.1/90.7 | 1.07/0.91 |
| Citraro 2020 | n/a | 90.5 / 91.8 | n/a | 0.018 / 0.012 | - | - | - |
| Nie 2021 (alignment) | 95.9 / 97.1 | 91.6 / 93.4 | 0.84 / 0.65 | 0.019 / 0.014 | 97.4/97.8 (ft) | 92.5/94.2 (ft) | 0.43/0.38 (ft) |
| KpSFR 2022 | 96.0 / 97.0 | 91.2 / 93.1 | 0.81 / 0.63 | 0.019 / 0.014 | 98.1/98.2 (ft) | 94.8/95.4 (ft) | 0.36/0.33 (ft) |
| Maglo 2022 | 96.3 / 97.4 | 92.0 / 94.1 | 0.74 / 0.55 | 0.018 / 0.014 | 98.3/98.5 (ft) | 95.7/96.2 | 0.26/0.23 |
| Shi 2022 | 96.6 / 97.8 | 93.1 / 94.8 | - | - | - | - | - |
| Oo et al. 2024 (GSR 2024 team) | 96.9 / 97.9 | 92.9 / 94.6 | 0.65 / 0.46 | 0.016 / 0.012 | 98.5/98.7 (ft) | 95.8/96.7 | 0.26/0.21 |
| NBJW SV (v1; trained SN, ft WC14/TSWC) | 96.2 / 97.8 | 92.2 / 94.3 | 0.68 / 0.46 | 0.016 / 0.011 | 98.6/98.8 (ft) | 96.3/96.8 | 0.23/0.20 |
| PnLCalib SV+PnL (v5) | 97.0 / 98.2 | 93.4 / 95.5 | 0.60 / 0.42 | 0.014 / 0.010 | 98.6/98.9 (ft) | 96.3/96.8 | 0.23/0.20 (reproj 0.005) |

  - Notes: (ft) = fine-tuned on TSWC train; Jiang TSWC row is the "trained on WorldCup only" block of KpSFR Table 2; Maglo/Oo/Shi are homography-only (pitch-registration) papers whose code I did not examine. KpSFR Table 2 rows for "Ours" without fine-tuning on TSWC: IoU_whole 93.2/94.3, IoU_part 97.6/97.7, proj. 0.45/0.41 m, reproj 0.012/0.011.
  - NBJW SV trained on SoccerNet only (no WC14 fine-tune), WC14: IoU_part 94.4/96.9, IoU_whole 89.4/93.0, proj. 1.23/0.58 m (v1 Table 3); after fine-tune on WC14's 209 train images: 96.2/97.8, 92.2/94.3, 0.68/0.46 m. — [NBJW v1](https://arxiv.org/pdf/2404.08401v1)
- TVCalib on WC14 without any WC14 training: IoU_part 95.3/96.6 (TVCalib pred. segments), AC@5/10/20 38.8/69.1/89.4 (CR 100) in its own table; with GT segments IoU_part 96.1/97.1. — [TVCalib](https://arxiv.org/pdf/2207.11709)
- Older numbers for context: Chen & Little 2019, Citraro 2020, Jiang 2020, Sha 2020, Homayounfar 2017 are all 2017-2020 results and have been superseded.

### Inferences
- Within each protocol, ranking is: PnLCalib+PnL > NBJW ~ Sportlight > Spiideo/TVCalib on per-frame JaC; on pure homography IoU metrics all methods from 2022-2026 cluster tightly (IoU_whole 91-94 on WC14, 94-96 on TSWC), so the IoU benchmarks no longer discriminate well.
- The three benchmarks measure different things; comparisons across them (e.g. Sportlight 73.2 on the challenge set vs PnLCalib 78.7 on SN23-test) are indicative only because sets, CR and resolution differ. Sportlight's low CR (73-76%) means its JaC is on easy frames.

### Gaps
- I did not find the WACV 2021 Nie paper's own table of absolute fps per method beyond a single sport-wise table (see Q6).
- No independent (non-author) reproduction numbers for PnLCalib/NBJW on a held-out dataset found, except Theiner 2026's re-use of NBJW localisation and Oo 2026's SN-GSR table (Broadcast2Pitch reports NBJW numbers copied from BroadTrack).
- 2022 winner (Baidu) code/weights: not found.

---

## 3. Which metric best predicts player-position accuracy in metres?

### Takeaway
No paper I found directly measures the correlation. In practice: (a) projection error in metres (mean/median over visible-field pixels vs GT homography, as in Nie/KpSFR/NBJW/PnLCalib) is the closest proxy, available only on WC14/TSWC where GT homographies exist; (b) JaC@t / AC@t and MRE are image-plane pixel measures of field-line alignment (resolution-dependent, not normalised, no GT camera) and ignore temporal consistency; (c) GS-HOTA's LocSim is the only end-to-end metric in metres (tolerance tau = 5 m), but mixes tracking/ID. I recommend building an in-house metres-error test set instead of relying on JaC.

### Cited Findings
- SoccerNet-calibration "no camera parameter ground truths are available for our soccer images"; evaluation is reprojection error against manually annotated field lines. — [2023 report](https://arxiv.org/pdf/2309.06006)
- JaC "is not normalized by the image dimensions", so the same pixel threshold is stricter at 1080p than 540p; frames for which a method returns parameters count even if reprojection error is high (hence CR is reported). — [Theiner 2026](https://openaccess.thecvf.com/content/CVPR2026W/CVsports/papers/Theiner_Unified_Sports_Field_Registration_with_Lens_Distortion_Modeling_CVPRW_2026_paper.pdf)
- Projection error is defined as the average distance in metres between points projected with the predicted vs GT homography, over 2,500 pixels sampled uniformly in the visible field area (105 x 68 m field in NBJW; Nie/Chu used 100 x 60). — [NBJW v1](https://arxiv.org/pdf/2404.08401v1); [KpSFR](https://openaccess.thecvf.com/content/CVPR2022W/CVSports/papers/Chu_Sports_Field_Registration_via_Keypoints-Aware_Label_Condition_CVPRW_2022_paper.pdf)
- IoU_whole (warp the template through pred-H^-1 GT-H) was argued to be better than IoU_part because it also scores the invisible area. Jiang found regressing IoU_whole gave best refinement. — [NBJW v1](https://arxiv.org/pdf/2404.08401v1); [Jiang](https://arxiv.org/pdf/1909.08034)
- GS-HOTA LocSim(P,G) = exp(ln(0.05) ||P-G||^2 / tau^2) with Euclidean pitch distance in metres, tau = 5 m (similarity 0.05 at 5 m). — [GSR paper](https://arxiv.org/pdf/2404.11335); [Broadcast2Pitch states 5 m tolerance](https://openaccess.thecvf.com/content/WACV2026/papers/Oo_Broadcast2Pitch_Game_State_Reconstruction_from_Unconstrained_Soccer_Videos_WACV_2026_paper.pdf)
- Calibration quality dominates GSR performance: in the GSR baseline module study (oracle substitution) the full baseline is 22.26 GS-HOTA while calibration/pitch oracle rows are ~50-51 (row semantics slightly ambiguous in my text extraction). Broadcast2Pitch ablation: keypoint-only homography GS-HOTA 48.23, lines-only 56.39, both 58.51 (same detector). — [GSR paper](https://arxiv.org/pdf/2404.11335); [Broadcast2Pitch](https://openaccess.thecvf.com/content/WACV2026/papers/Oo_Broadcast2Pitch_Game_State_Reconstruction_from_Unconstrained_Soccer_Videos_WACV_2026_paper.pdf)
- Reprojection-in-image quality does not guarantee plausible camera pose: BroadTrack notes focal length and position trade off ("vertigo effect") so "more than accurate reprojections" is needed for consistent position/rotation; Magera 2025 says close-up central views admit ambiguous solutions without priors. — [BroadTrack](https://arxiv.org/pdf/2412.01721); [Magera 2025](https://arxiv.org/pdf/2504.20052)
- Typical scale from the homography benchmarks: best published projection error 0.23 m mean / 0.20 m median (TSWC, PnLCalib SV+PnL, fine-tuned) and 0.60 / 0.42 m (WC14); worst-era examples ~1.2 m (NBJW SV without fine-tune, Jiang). — tables in Q2.

### Inferences
- Projection error (m) over the visible field is the best available stand-in for a ground-plane error in metres; IoU_whole correlates with it but saturates. JaC@5 is a good ranking signal on SoccerNet but is in pixels and resolution-dependent; do not use it to estimate metre error on your footage.
- Player feet are typically mid-field far from lines, so metric error there is governed by extrapolation of the homography away from detected lines/keypoints; "visible-area" metrics overweight pixels near lines? (uncertain: not tested in the sources).
- Recommended: evaluate on your own ~50 human-verified frames by projecting human-clicked ground points (e.g. penalty-spot, corner of box, player feet marked against pitch markings) and measuring metres.

### Gaps
- No paper reporting JaC-vs-metre-error correlation, nor per-pixel-location error maps; not found.

---

## 4. Training data needs and synthetic pretraining

### Takeaway
All strong recent methods train on SoccerNet-calibration (22.8k-25.5k annotated frames, polyline annotations) and optionally fine-tune on ~200 images (WC14) up to 3.8k (TSWC). The older retrieval methods (Chen & Little, Sha, Sharma) pretrain on synthetic edge/semantic images; the newer keypoint methods generate keypoints from the geometry of the annotations and do not use GAN/synthetic data (Constructor mixes synthetic + real for its camera-regression network).

### Cited Findings
- NBJW/PnLCalib: SN23-train (SoccerNet-Calibration, 22,816 images in total) for MV and SV (SV uses a subset mostly non-replay frames); fine-tune sets WC14 (209 train / 186 test) and TSWC (3,812 images from 43 videos). Adam, MV: 200 epochs, lr 1e-2, batch 2; SV: 200 epochs, lr 1e-5, batch 1; augment: flip, colour jitter, Gaussian noise. GT homographies are projected to keypoint sets/line extremities. — [PnLCalib v5](https://arxiv.org/html/2404.08401v5); [NBJW v1](https://arxiv.org/pdf/2404.08401v1)
- PnLCalib README: WorldPose dataset also added as training/benchmark source (2025-06-06 update); released weights include SV fine-tuned on WC14, TSWC and WorldPose. — [PnLCalib repo](https://github.com/mguti97/PnLCalib)
- TVCalib: segmentation net trained on SN-Calib train (DeepLabv3-ResNet101, 30 epochs, batch 8, SGD lr 0.01, ImageNet init, images resized to height 256); the camera step needs no training. WC14 was evaluated without any fine-tuning. — [TVCalib](https://arxiv.org/pdf/2207.11709)
- Sportlight: SoccerNet annotations processed geometrically into 57 keypoints; two models (keypoints, lines), hyperparameter-tuned with Optuna. — [repo](https://github.com/NikolasEnt/soccernet-calibration-sportlight)
- Spiideo: segmentation GT generated from SN annotations by flood fill; trains on calibration-2023 splits. — [repo](https://github.com/Spiideo/soccersegcal)
- KpSFR: WorldCup train (209 imgs) for 1500 epochs (lr 1e-4, bs 4, decay x0.1 every 300 epochs) then fine-tune 100 epochs on TS-WorldCup (3,812 images; downloadable). — [KpSFR paper](https://openaccess.thecvf.com/content/CVPR2022W/CVSports/papers/Chu_Sports_Field_Registration_via_Keypoints-Aware_Label_Condition_CVPRW_2022_paper.pdf); [repo](https://github.com/ericsujw/KpSFR)
- Nie 2021: trained on WorldCup (209) and a new "SportsFields" dataset (192 clips, 5 sports); keypoint grid vs local keypoints ablation (WC14 IoU_whole 91.6 vs 81.9, proj. 0.82 vs 3.51 m mean). — [Nie](https://openaccess.thecvf.com/content/WACV2021/papers/Nie_A_Robust_and_Efficient_Framework_for_Sports-Field_Registration_WACV_2021_paper.pdf)
- Chen & Little 2019: synthetic camera poses/edge images generate training data (10,000 synthetic training samples in a robustness experiment); real data for the two-GAN field-marking detection (WC14 209 train images; also evaluated on a volleyball dataset per the abstract). — [arXiv 1810.10658](https://arxiv.org/pdf/1810.10658)
- Sha 2020: pre-trains on synthetic semantic images generated by uniformly sampling pan, tilt, focal length, then end-to-end; basketball dataset 526 train / 114 test images (4-6 clicked point correspondences each), WC14 209 train; authors say WC14 performance suffers from "insufficient training data, particularly for the semantic segmentation". — [Sha](https://openaccess.thecvf.com/content_CVPR_2020/papers/Sha_End-to-End_Camera_Calibration_for_Broadcast_Videos_CVPR_2020_paper.pdf)
- Citraro 2020: per-dataset training; soccer World Cup, basketball, volleyball, MLS soccer datasets; ResNet-50 (ImageNet-pretrained) backbone variant; two networks per field side. — [Citraro](https://arxiv.org/pdf/2003.14109)
- Jiang 2020: two nets trained separately ("decoupled training"); datasets include World Cup 2014 (training-set size not extracted) and hockey — [Jiang](https://arxiv.org/pdf/1909.08034)
- Constructor: camera-parameter regression trained on 62k images with 7 labelled params (Table 1 says "Includes synthetic and real-world data"; text mentions 40,000 images with camera parameters), field keypoints 36k images (74 keypoints); training of the camera net is 400k batches, bs 8, AdamW. — [Constructor](https://arxiv.org/pdf/2504.06357)
- Broadcast2Pitch: detector trained 25 epochs (lr listed in paper), RTX 4090 24 GB; uses SoccerNet-GSR dense keypoint annotations (9.37M pitch keypoints noted for the dataset). — [Broadcast2Pitch](https://openaccess.thecvf.com/content/WACV2026/papers/Oo_Broadcast2Pitch_Game_State_Reconstruction_from_Unconstrained_Soccer_Videos_WACV_2026_paper.pdf)
- Theiner 2026: requires only pixel-level point annotations per visible field segment at train/eval; uses existing pitch localisation models (NBJW) for segments. — [Theiner 2026](https://openaccess.thecvf.com/content/CVPR2026W/CVsports/papers/Theiner_Unified_Sports_Field_Registration_with_Lens_Distortion_Modeling_CVPRW_2026_paper.pdf)

### Inferences
- Fine-tuning the NBJW SV model on ~200 images (WC14) moved WC14 IoU_whole mean from 89.4 to 92.2 and projection error from 1.23 to 0.68 m (v1), which is the best direct evidence that a few hundred frames of in-domain labels help; no evidence found on 50 frames specifically.

### Gaps
- No paper reports an ablation on number of fine-tuning frames (e.g. 50 vs 200).
- Exact real/synthetic split of Constructor's 62k camera-parameter images is not spelled out in what I read.

---

## 5. Code, licences, pretrained weights, framework

### Takeaway
Easiest to adopt legally and technically: PnLCalib (GPL-2.0-or-later code, CC BY 4.0 weights, SV/MV/finetuned weights, PyTorch, active in 2026) or the very similar NBJW (GPL-2.0 code, weights licence unstated). TVCalib and KpSFR (MIT) are clean but older. Sportlight has no licence file. BroadTrack is non-commercial research only. Several older methods have no public code.

### Cited Findings
| Method | Code | Licence | Weights | Framework | Source |
|---|---|---|---|---|---|
| PnLCalib | github.com/mguti97/PnLCalib (pushed 2026-10-26; 113 stars; 8 issues, all closed) | GPL-2.0-or-later (LICENSE file) | GitHub Releases v1.0.0: SV kp/lines base + finetuned WC14, TSWC, WorldPose; MV kp/lines; CC BY 4.0; base models trained on SoccerNet-Calibration under a research agreement that "has since expired", README says weights contain no footage | PyTorch (HRNetv2-w48 configs; HRNet files MIT), conda env / requirements | [repo](https://github.com/mguti97/PnLCalib) (LICENSE read via GitHub API) |
| NBJW | github.com/mguti97/No-Bells-Just-Whistles (last push 2024-10-25; 61 stars) | GPL-2.0 (LICENSE text; GitHub API says NOASSERTION) | GitHub Releases v1.0.0 SV/MV kp+lines + WC14/TSWC finetuned; weights licence not stated | PyTorch | [repo](https://github.com/mguti97/No-Bells-Just-Whistles) |
| TVCalib | github.com/MM4SPA/tvcalib (last push 2024-04-27) | MIT | `train_59.pt` segmentation model hosted on tib.eu; weights licence not stated; depends on `sn_segmentation` submodule with no stated licence | PyTorch, kornia, conda | [repo](https://github.com/MM4SPA/tvcalib) |
| KpSFR | github.com/ericsujw/KpSFR (last push 2022-10-10) | MIT | kpsfr.pth (WorldCup), kpsfr_finetuned.pth (TS-WorldCup), loss-ablation weights; download links in README; weights licence not stated | PyTorch 1.9.0, torchvision 0.9.0, CUDA 11, Python >=3.8, OpenCV 4.5.1.48, Shapely 1.7.1 | [repo](https://github.com/ericsujw/KpSFR) |
| Sportlight (2023 winner) | github.com/NikolasEnt/soccernet-calibration-sportlight (66 stars, last push 2024-12-30) | none (GitHub API licence: null) | README shows no downloadable weights (I saw none) | PyTorch, Docker, Hydra | [repo](https://github.com/NikolasEnt/soccernet-calibration-sportlight); paper arXiv 2410.07401 (MMSports'24) |
| Spiideo soccersegcal (2023 #2) | github.com/Spiideo/soccersegcal (last push 2023-08-03) | MIT | `snapshot.ckpt` (27 epochs) from release SoccerNetChallenge2023 | PyTorch + pytorch3d (example wheel Py3.8/CUDA 11.3/PyTorch 1.12) | [repo](https://github.com/Spiideo/soccersegcal) |
| BroadTrack | github.com/evs-broadcast/BroadTrack (last push 2025-02-17) | EVS licence: personal, non-commercial research use only, non-transferable, no sublicensing | not assessed | - | [repo licence via GitHub API](https://github.com/evs-broadcast/BroadTrack) |
| Jiang 2020 | github.com/vcg-uvic/sportsfield_release (last push 2021-03-28) | "Academic or non-profit organisation noncommercial research use only" | not assessed | - | [repo](https://github.com/vcg-uvic/sportsfield_release); KpSFR paper says Jiang had no official training code in 2022 (conflicting; unchecked) |
| sn-gamestate (baseline) | github.com/SoccerNet/sn-gamestate (471 stars, push 2026-05-02) | GPL-3.0 | TrackLab downloads dataset + all weights on first run (incl. calibration) | PyTorch 1.13.1 / Py3.9, mmcv 2.0.1, tracklab (MIT) | [sn-gamestate](https://github.com/SoccerNet/sn-gamestate); [tracklab](https://github.com/TrackingLaboratory/tracklab) |
| sn-calibration (challenge kit/baseline) | github.com/SoccerNet/sn-calibration (112 stars) | none per GitHub API | baseline only | - | [repo](https://github.com/SoccerNet/sn-calibration) |

- Nie 2021: KpSFR paper states "there are no publicly available codes from the authors" so Chu et al. re-implemented it (their re-implementation scored IoU_whole 90.3 vs 91.6 reported). — [KpSFR](https://openaccess.thecvf.com/content/CVPR2022W/CVSports/papers/Chu_Sports_Field_Registration_via_Keypoints-Aware_Label_Condition_CVPRW_2022_paper.pdf)
- KpSFR: "no publicly available inference codes for some baselines [Homayounfar, Sha, Sharma, Citraro, Cioppa, Xeebra]" -> numbers taken from papers. Same source.
- Constructor Tech's 2024 winning calibration: paper footnote points to a Constructor research platform page; I found no downloadable weights. — [Constructor paper](https://arxiv.org/pdf/2504.06357)
- Theiner 2026: "source code will be made available" (not yet found). Broadcast2Pitch: code/weights link not found in extracted text.

### Inferences
- GPL-2.0 for NBJW and PnLCalib is fine for internal use; it is copyleft if you distribute a derived product. The SoccerNet-derived training data licence (research agreement) is flagged by the PnLCalib authors; they state weights are free of restriction, but a cautious commercial user should note it.
- sn-gamestate (GPL-3.0) is GPL-3.0 vs PnLCalib's GPL-2.0-or-later: compatible via "or later".

### Gaps
- Weight licence for NBJW, TVCalib, KpSFR not stated.
- Code status of Citraro 2020, Sha 2020, Homayounfar 2017, Chen & Little 2019: not found in this pass (not verified either way).
- Whether Sportlight released weights: none seen in README.

---

## 6. GPU memory, training/fine-tune fit on 8 GB RTX 2070, inference FPS

### Takeaway
No source states VRAM for NBJW/PnLCalib. They were trained on a single 12 GB RTX 2080 Ti with batch size 1-2 (SV) which makes 8 GB fine-tuning plausible but unverified. Inference is classical-solver-bound: PnLCalib ~6 FPS (164 ms) minimal, ~2.3 FPS (439 ms) full voting on a Xeon CPU; TVCalib pipeline is slower (module-level 2.9-7.6 FPS in the GSR baseline). Sportlight needs 24 GB for training; KpSFR trained on 32 GB V100.

### Cited Findings
- NBJW/PnLCalib experiments: "a single NVIDIA GeForce RTX 2080 Ti GPU with 12 GB of memory", PyTorch; calibration/optimisation on Intel Xeon Silver 4214 CPU (OpenCV, SciPy); default training batch size 2 (MV) / 1 (SV), 200 epochs; README train defaults batch 2, 4 workers, `cuda:0`; distributed training since 2025-06-06. — [PnLCalib v5](https://arxiv.org/html/2404.08401v5); [repo](https://github.com/mguti97/PnLCalib)
- PnLCalib inference time on SN23-test (ms per frame; FPS derived by me): 1 RANSAC config, 1 keypoint subset, no PnL: 164 ms (~6.1 FPS); with PnL: 178 ms (~5.6 FPS); 6 configs + PnL: 273 ms (~3.7 FPS); 6 configs, 3 keypoint subsets + PnL (full, JaC@5 78.7, CR 78.4): 439 ms (~2.3 FPS); timing "varies across frames". Which GPU was used for the network part at inference is not stated. — [PnLCalib v5](https://arxiv.org/html/2404.08401v5)
- Sportlight: README prerequisites "Nvidia GPU with at least 24GB of memory. Nvidia RTX 3090 and RTX 4090 GPUs were used" (training). — [repo](https://github.com/NikolasEnt/soccernet-calibration-sportlight)
- KpSFR trained on "a single NVIDIA V100 with 32G VRAM", batch 4; speed not reported in the paper. — [KpSFR](https://openaccess.thecvf.com/content/CVPR2022W/CVSports/papers/Chu_Sports_Field_Registration_via_Keypoints-Aware_Label_Condition_CVPRW_2022_paper.pdf)
- Nie 2021 (older): inference fps on SportsFields: soccer 35, American football 25, ice hockey 35, basketball 35, tennis 45 (commodity hardware, HD); online optimisation is 90% of compute; on WC14 Nie's table lists Posenet 19 fps, Eric et al. 5, Chen et al. -, Wei/Jiang 0.6 fps. — [Nie](https://openaccess.thecvf.com/content/WACV2021/papers/Nie_A_Robust_and_Efficient_Framework_for_Sports-Field_Registration_WACV_2021_paper.pdf)
- Citraro 2020 (older): abstract "20-30 frames per second on a desktop computer with an Nvidia TITAN X (Pascal)"; Table 2 on Soccer World Cup lists 8 fps (9 fps without players) -- internal inconsistency in the paper's numbers. — [Citraro](https://arxiv.org/pdf/2003.14109)
- Jiang 2020 (older): 41.76 optimisation iterations/s -> 9.58 s per frame on GTX 1080 Ti (full), or 4.66 it/s -> 1.36 s per frame for another setting. — [Jiang](https://arxiv.org/pdf/1909.08034)
- Sha 2020 (older): 0.004 s per image mean, search space 450 templates (vs 0.44 s Homayounfar, 0.5 s Chen et al.), "2 orders of magnitude faster"; Tensorflow, Titan RTX. — [Sha](https://openaccess.thecvf.com/content_CVPR_2020/papers/Sha_End-to-End_Camera_Calibration_for_Broadcast_Videos_CVPR_2020_paper.pdf)
- Homayounfar 2017 (older): 0.44 s/frame, 3328 mean iterations for soccer; trained ~1 day on one DGX-1 GPU. — [Homayounfar](https://openaccess.thecvf.com/content_cvpr_2017/papers/Homayounfar_Sports_Field_Localization_CVPR_2017_paper.pdf)
- Chen & Little 2019 (older): speed measured on a 3.0 GHz CPU, 16 GB Mac (number not extracted). — [arXiv 1810.10658](https://arxiv.org/pdf/1810.10658)
- TVCalib in the GSR baseline: calibration module 7.6 FPS (batch 512), pitch localisation 2.9 FPS (batch 16); hardware not stated in the extracted text. — [GSR paper](https://arxiv.org/pdf/2404.11335)
- BroadTrack: 16 FPS at 1920x1080 "on a server equipped with two RTX 4090 GPUs, without any optimization". — [BroadTrack](https://arxiv.org/pdf/2412.01721)
- Constructor: camera-parameter net 5M params at 512x288 (very light); full pipeline "real-time ... up to 80 FPS" claimed for tracking (not for calibration alone). — [Constructor](https://arxiv.org/pdf/2504.06357)
- Broadcast2Pitch: all experiments on one RTX 4090 24 GB; no calibration fps given in extracted text. — [Broadcast2Pitch](https://openaccess.thecvf.com/content/WACV2026/papers/Oo_Broadcast2Pitch_Game_State_Reconstruction_from_Unconstrained_Soccer_Videos_WACV_2026_paper.pdf)
- sn-gamestate README advises lowering `batch_size` on memory issues; no VRAM stated. — [sn-gamestate](https://github.com/SoccerNet/sn-gamestate)

### Inferences
- Inference-only use of PnLCalib/NBJW/TVCalib on an RTX 2070 (8 GB) is very likely fine (HRNetV2-w48 at modest resolution, batch 1); fine-tuning with batch 1-2 probably fits in 8 GB, perhaps with mixed precision -- unverified, test before relying.
- For offline processing of broadcast clips at 25 fps, subsample to ~1-5 FPS and use temporal tracking (BroadTrack/optical flow), because per-frame PnL voting is ~2-6 FPS.

### Gaps
- No source gives measured VRAM for NBJW/PnLCalib/TVCalib training or inference.
- No FPS for KpSFR, Spiideo or Sportlight pipelines found.

---

## 7. Known failure modes

### Takeaway
Everything keypoint/line-based fails when fewer than ~4 landmarks are visible (close-ups, central circle shots, tele-shots), on fisheye in-goal cameras, in motion blur, and in the sparse zone between penalty box and halfway line; wide/high cameras need distortion modelling; completeness (not accuracy) is the main cost of the strongest methods on multi-view data.

### Cited Findings
- NBJW/PnLCalib: fail on close-up shots with few or no landmarks and fisheye shots from inside the goals; require at least four visible keypoints. PnLCalib multi-view CR is only ~78-80% on SN23-test (vs 98-100% single-view on main-camera-centre subsets). — [PnLCalib v5](https://arxiv.org/html/2404.08401v5)
- Central close-ups: on central-view subset of SoccerNet test sets (full-HD frames), PnLCalib JaC5 22.4, MRE 12.8 px, CR 79.5; TVCalib JaC5 10.2, MRE 28.8, CR 100 -- these views "challenging for the vast majority of existing methods" because they show only lines and circle(s); the Magera 2025 geometric circle-to-points method complements detectors. — [Magera et al. 2025](https://arxiv.org/pdf/2504.20052)
- Theiner 2026: CR is not 100% even with manually-annotated segments on SN23 due to "tele-shots from the ground, fisheye views, and some other cases like where visible segments are very sparse"; on SN-GSR, "sub-sequences where a low number of visible segments is present (i.e. between the penalty box and pitch center) are not calibrated reliably"; predicted-vs-GT localisation gap due to missed central or side lines from "out-of-distribution inference (motion blur, views differing from the training data)". — [Theiner 2026](https://openaccess.thecvf.com/content/CVPR2026W/CVsports/papers/Theiner_Unified_Sports_Field_Registration_with_Lens_Distortion_Modeling_CVPRW_2026_paper.pdf)
- TVCalib limitations: local minima from gradient-based optimisation; camera initialisation only from main-tribune cameras ("we have not investigated any cameras other than those on the main tribune"); relies on accurate segment localisation without outlier regularisation; joint distortion learning hits trivial local minima on low-FoV images (SN-Calib), not WC14 where FoV is larger; "by design, never signals failure" (no rejection unless tau threshold used). — [TVCalib](https://arxiv.org/pdf/2207.11709); [BroadTrack](https://arxiv.org/pdf/2412.01721)
- Wide-angle broadcast cameras have ~25x zoom lenses and radial distortion at wide FoV; "elevated tripod-mounted cameras around the stadium"; reinitialisation needed after tracking losses (per the BroadTrack text: reinitialisation used 33 times on the sn-gamestate test set, tracking losses lasting 15 frames on average). — [BroadTrack](https://arxiv.org/pdf/2412.01721)
- Nie 2021 failure cases: online optimisation failing on small field areas, heavy motion blur, camera occlusions (e.g. audience raising hands); American football (small FoV, uniform field) triggers optimisation most. — [Nie](https://openaccess.thecvf.com/content/WACV2021/papers/Nie_A_Robust_and_Efficient_Framework_for_Sports-Field_Registration_WACV_2021_paper.pdf)
- Citraro 2020 failure cases: lack of visible lines, clutter, and training-distribution mismatch. — [Citraro](https://arxiv.org/pdf/2003.14109)
- Sha 2020: struggles with insufficient training data for semantic segmentation (WC14); retrieval/template methods limited by template set coverage (Chen & Little). — [Sha](https://openaccess.thecvf.com/content_CVPR_2020/papers/Sha_End-to-End_Camera_Calibration_for_Broadcast_Videos_CVPR_2020_paper.pdf)
- Single-view ambiguity: left/right half ambiguity without further priors is noted generally (Theiner cites [23]); Constructor adds a "Unique Camera Parameters Network"/team-side information to disambiguate. — [Theiner 2026](https://openaccess.thecvf.com/content/CVPR2026W/CVsports/papers/Theiner_Unified_Sports_Field_Registration_with_Lens_Distortion_Modeling_CVPRW_2026_paper.pdf); [Constructor](https://arxiv.org/pdf/2504.06357)
- Scouting-feed cameras (wide FoV, different movement behaviour from PTZ broadcast cameras) are used by Theiner 2026 only as a private set (SF4) to analyse k1 plausibility; no numeric JaC given in the extracted text. — [Theiner 2026](https://openaccess.thecvf.com/content/CVPR2026W/CVsports/papers/Theiner_Unified_Sports_Field_Registration_with_Lens_Distortion_Modeling_CVPRW_2026_paper.pdf)

### Inferences
- "High wide cameras" (tactical/scouting feeds) fall outside the SoccerNet camera distribution; expect degraded localisation and need k1 modelling; fine-tuning on your own frames is advisable.
- "Zoomed-in penalty boxes" are covered by keypoints (penalty-box corners, arcs) and usually work if >=4 landmarks are visible; "reverse angles" (behind-goal/end-zone) are covered only by MV models (CR ~78-80%).

### Gaps
- No quantitative per-camera-type breakdown (reverse angle, high wide, far-side low-texture) found in any source I read; the SoccerNet camera-type annotation exists for SN22/SN23 but breakdowns were not found. "Low-texture far side" is not specifically analysed.

---

## 8. Easiest methods to fine-tune on a small custom set (~50 verified + ~150 proposals)

### Takeaway
PnLCalib (or NBJW) is the easiest: provided SV pretrained weights, `train.py`/`train_l.py` with a `--pretrained` argument, precedent of fine-tuning on only 209 WC14 images with clear gains, and labels derivable from per-frame GT homographies (or SoccerNet-format polylines). KpSFR is next (fine-tune script/weights, but older stack PyTorch 1.9, homography-only, 32 GB GPU used for training). TVCalib has no trainable calibration step; only its segmentation net trains, and the repo gives no fine-tuning guide.

### Cited Findings
- PnLCalib/NBJW authors fine-tune both SV networks on WC14 (209 train images) and TSWC by converting GT homographies into keypoint sets and line extremities, then training the HRNet heatmap networks with lr 1e-5, batch 1. `train.py` (keypoints, `config/hrnetv2_w48.yaml`) and `train_l.py` (lines) accept pretrained weights path and wandb project. Defaults 200 epochs, bs 2. — [PnLCalib repo](https://github.com/mguti97/PnLCalib); [PnLCalib v5](https://arxiv.org/html/2404.08401v5)
- Fine-tuning effect, NBJW SV on WC14 (209 images): IoU_whole 89.4 -> 92.2 mean, proj. error 1.23 -> 0.68 m mean; TSWC 94.4 -> 96.3 (with PnL). — [NBJW v1](https://arxiv.org/pdf/2404.08401v1); [PnLCalib v5](https://arxiv.org/html/2404.08401v5)
- PnLCalib issue tracker has 8 closed issues, none about fine-tuning on custom data (titles: camera parameters, keypoints definition in calibration-2023, wandb key error, missing module, numpy RankWarning, thresholds, focal length, minimap). NBJW has 5 issues (1 open about generating homography/inverse). — [PnLCalib issues](https://github.com/mguti97/PnLCalib/issues); [NBJW issues](https://github.com/mguti97/No-Bells-Just-Whistles/issues) (listed via `gh issue list`)
- KpSFR: `python train_nn.py <param_file>` with `--gpu_ids`, `--train_epochs`, `--step_size`; fine-tuned checkpoint `kpsfr_finetuned.pth`; fine-tune recipe 100 epochs lr decay every 30 epochs on TS-WorldCup. — [KpSFR repo](https://github.com/ericsujw/KpSFR); [KpSFR paper](https://openaccess.thecvf.com/content/CVPR2022W/CVSports/papers/Chu_Sports_Field_Registration_via_Keypoints-Aware_Label_Condition_CVPRW_2022_paper.pdf)
- TVCalib repo: segmentation model reproduction via `run_segmentation.sh`; no fine-tuning instructions; `--distort` evaluation flag. — [TVCalib repo](https://github.com/MM4SPA/tvcalib)
- Sportlight repo: full training pipeline in Docker + Hydra configs but 24 GB GPU prerequisite and no weights; heuristics tuned with Optuna on SN. — [repo](https://github.com/NikolasEnt/soccernet-calibration-sportlight)
- Spiideo: train from scratch pipeline with pytorch3d; weights provided; large dependency stack. — [repo](https://github.com/Spiideo/soccersegcal)
- The PnLCalib refinement step (PnL) and RANSAC/voting are non-learned, so fine-tuning only has to improve keypoint/line heatmaps; the `alpha` parameter (0.6 best on SN23) weights lines vs points in PnL. — [PnLCalib v5](https://arxiv.org/html/2404.08401v5)

### Inferences
- With 50 verified + 150 proposals, fine-tune the PnLCalib SV keypoint + line HRNets from the released SV weights at low lr (1e-5, batch 1-2), hold out the verified frames (or ~half) for testing, weight verified frames higher / use proposals only after filtering by reprojection consistency; keep PnL refinement on.
- Machine proposals risk teaching the model its own biases; keep the human-verified frames as the evaluation set, and the data must be convertible to a per-frame homography/camera + field model so keypoints/line extremities can be generated.
- Complement: add BroadTrack-style temporal tracking or Theiner 2026's k1 option on video to reduce frame-to-frame jitter (BroadTrack is research-only licence).

### Gaps
- No evidence on fine-tuning with 50 images specifically; no custom-data fine-tuning reports found in issues or papers.
- 8 GB feasibility for fine-tuning HRNetV2-w48 untested in sources.

---

## 9. Newer work (2024-2026) and overall recommendation signals

### Takeaway
2024-2026 progress is mostly (a) PnLCalib's refinement + lens distortion, (b) video/temporal methods (BroadTrack, optical-flow tracking in GSR entries, Broadcast2Pitch) that matter most for GSR-style video, (c) geometry-based fixes for hard views (Magera 2025 circle method), (d) distortion-aware, annotation-light calibration (Theiner 2026). No fresh standalone SoccerNet calibration challenge exists after 2023 to rank methods; ranking relies on authors' own benchmarks.

### Cited Findings
- BroadTrack (WACV 2025): halves mean reprojection error and +15% Jaccard vs SoccerNet baseline claim (abstract: "halves the mean reprojection error rate and gains more than 15% in terms of Jaccard index"); SN-GSR JaC5 56.88 vs NBJW 37.14; k1 + optical flow + tripod constraints; 16 FPS on 2x RTX 4090; code under EVS non-commercial licence. — [BroadTrack](https://arxiv.org/pdf/2412.01721)
- Magera et al. 2025 (arXiv 2504.20052): bottom-up geometric derivation of points/lines from circle correspondences for central views; improves over PnLCalib on these views. — [arXiv](https://arxiv.org/pdf/2504.20052)
- Theiner et al. CVPRW 2026: joint k1 estimation, annotation-light, multi-sport (MultiSports-Calib: handball, basketball, ice hockey, volleyball) plus soccer; best SN22-center JaC5 with predicted segments 82.8 (CR 98.0). Code promised. — [paper](https://openaccess.thecvf.com/content/CVPR2026W/CVsports/papers/Theiner_Unified_Sports_Field_Registration_with_Lens_Distortion_Modeling_CVPRW_2026_paper.pdf)
- Broadcast2Pitch (WACV 2026, KIST, 2025 GSR winner): homography module (lines + circle conic + keypoint fallback) JaC5 69.38 / JaC10 92.84 / MRE 4.47 px on SN-GSR test (author-run); GS-HOTA 61.48 with their identity module in its own ablation. — [paper](https://openaccess.thecvf.com/content/WACV2026/papers/Oo_Broadcast2Pitch_Game_State_Reconstruction_from_Unconstrained_Soccer_Videos_WACV_2026_paper.pdf)
- PnLCalib journal version accepted CVIU 2026 with alpha-weighted PnL and distortion option. — [repo](https://github.com/mguti97/PnLCalib)
- Field Converter (arXiv 2609.10498) is not a calibration method; it assumes camera parameters from an upstream registration system such as PnLCalib. — [arXiv listing via search](https://arxiv.org/pdf/2609.10498)

### Inferences
- A pragmatic pipeline for football: PnLCalib/NBJW per frame (or at keyframes) + optical-flow/temporal smoothing for in-between frames; k1 estimation if the footage is wide-angle HD; fine-tune on your frames.

### Gaps
- I did not independently check for further 2026 arXiv papers beyond one extended search; coverage of 2026 is thus partial.
- Official SoccerNet leaderboard pages (EvalAI) were not accessed; numbers for lianyou's 77.3 come only from Theiner's table (taken "from the benchmark website without further information").
