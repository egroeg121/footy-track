# Calibration accuracy needed to project football player detections onto a 2D pitch, and how end-to-end systems do it

(Research date: 2026-11-14. Numbers marked "[computed]" are my own geometry calculation, not from a source; everything else is cited.)

## 1. SoccerNet Game State Reconstruction (GSR): GS-HOTA, baseline and winners, how much calibration costs

### Takeaway
GS-HOTA = HOTA with the localisation similarity replaced by a Gaussian kernel on pitch-plane distance (tau = 5 m) multiplied by an all-or-nothing identity match (role, team, jersey). Baseline and winners all project the bottom-centre of the player box through the ground-plane homography of a per-frame calibrated camera. Calibration is the dominant failure mode (oracle calibration roughly doubles the baseline score), and the 2024-2025 winners differ mainly through better calibration modules plus temporal smoothing.

### Cited Findings
**GS-HOTA definition**
- GS-HOTA is derived from HOTA; Sim_GS-HOTA(P,G) = LocSim(P,G) x IdSim(P,G). LocSim = exp(ln(0.05) * ||P-G||^2 / tau^2), i.e. a Gaussian kernel on the Euclidean distance in pitch coordinates; IdSim = 1 only if role, team and jersey number all match (attributes missing in GT, e.g. referee jersey, are ignored). — [Somers et al., arXiv 2404.11335](https://arxiv.org/pdf/2404.11335)
- HOTA is integrated over alpha in [0.05, 0.95] in 0.05 steps, so pairs with similarity <= 0.05 are never matched; tau therefore equals the maximum match distance in metres, and distances smaller than tau still give higher score. The authors chose tau = 5 m, "a reasonable distance tolerance given the average dimensions of a soccer pitch (68 x 105 m) and the substantial distance between the camera and the athletes". — [Somers et al.](https://arxiv.org/pdf/2404.11335)
- Implication of the formula (my arithmetic from the paper's kernel): LocSim = 0.05 at 5 m, 0.5 at about 2.4 m (d = tau*sqrt(ln0.5/ln0.05) = 0.48*5), 0.95 at about 0.65 m. [computed from the formula]
- Dataset: SoccerNet-GSR, 200 30-s sequences, 2.36 M athlete bounding boxes with pitch positions, 9.37 M pitch line points for calibration. — [Somers et al.](https://arxiv.org/pdf/2404.11335); [Constructor Tech paper, arXiv 2504.06357](https://arxiv.org/pdf/2504.06357)

**Baseline projection method (sn-gamestate / TrackLab)**
- Calibration uses TVCalib: a semantic-segmentation pitch localisation module, then camera parameters estimated by iteratively minimising pitch-segment reprojection error. "Once the camera has been calibrated, its corresponding homography is used to transform image bounding boxes into 2D positions on the pitch. For this purpose, we assume that the bottom of the bounding box lies on the ground field." — [Somers et al.](https://arxiv.org/pdf/2404.11335)
- So: full camera model estimated, but only the ground-plane homography derived from it is used for player projection; foot point = bottom-centre of box.
- sn-gamestate lists TVCalib as default calibration; PnLCalib and "No Bells, Just Whistles" (NBJW) are switchable alternatives (added 2024-07-10); changelog 2026-05-01: "Tracklab updated to 1.3.24 to fix broken GS-HOTA evaluation". — [sn-gamestate GitHub](https://github.com/SoccerNet/sn-gamestate)
- Ground-truth pitch positions in the dataset JSON are stored as `bbox_pitch` with `x_bottom_left/right/middle` fields (i.e. bottom-middle is the GT foot point too). — [Somers et al. supplementary text, via PDF](https://arxiv.org/pdf/2404.11335)

**How much calibration/pitch localisation contributes to the score (2024 paper, test split)**
- Table 1: with pitch localisation disabled (LocSim replaced by image-space box IoU) GS-HOTA = 57.64; enabling pitch/calibration drops it to 42.65 (no identity attributes); all attributes on = 22.26 (baseline). So moving from image-space to pitch-space matching alone costs ~15 GS-HOTA points. — [Somers et al.](https://arxiv.org/pdf/2404.11335)
- Table 2 (oracle ablation, all other modules replaced by ground truth): calibration (TVCalib) module alone predicted = 51.39 GS-HOTA; pitch localisation module alone predicted = 49.99; jersey module alone = 56.75; detector alone = 35.28; vs. full baseline 22.26. I.e. even with perfect everything else, TVCalib calibration caps GS-HOTA near ~50 -- authors say these experiments "show the severe performance impact of enabling calibration and pitch localization". — [Somers et al.](https://arxiv.org/pdf/2404.11335)
- Qualitative failure: a sequence with a GS-HOTA of 0.23 % vs 49.69 % for a good one; "even minor calibration inaccuracies can cause major pitch registration errors", caused by few visible salient pitch points. — [Somers et al.](https://arxiv.org/pdf/2404.11335)
- Baseline calibration is the slowest-but-one stage: calibration 7.6 FPS, pitch 2.9 FPS on A100; 11 min per 30 s sequence for the full pipeline. — [Somers et al.](https://arxiv.org/pdf/2404.11335)

**Leaderboards**
- 2024 challenge: Constructor Tech 63.81 GS-HOTA (GS-DetA 49.52, GS-AssA 82.23), UPCxMobius 43.15, JAM 34.40, baseline 23.36. — [Constructor Tech, arXiv 2504.06357](https://arxiv.org/pdf/2504.06357)
- 2025 challenge (14 teams, 66 submissions): 1. KIST-GSR 63.90, 2. Constructor.Tech 63.81, 3. lianyou 62.76, 4. Playbox & MIXI 61.64, 5. tyler durden 57.08; baseline (improved) 29.01. Top score rose only marginally vs 2024 (+0.09) though the median improved a lot. — [SoccerNet 2025 Challenges Results, arXiv 2508.19182](https://arxiv.org/pdf/2508.19182)
- 2025 report on calibration: popular approaches were landmark detection followed by calibration/homography estimation, with NBJW, PnLCalib and BroadTrack commonly used; "a novel development was the incorporation of optical flow techniques to improve temporal smoothing". — [arXiv 2508.19182](https://arxiv.org/pdf/2508.19182)

**What the winners / top entries used for calibration**
- 2024 winner Constructor Tech: pinhole camera model, zero distortion, principal point fixed at image centre; per-frame initial camera parameters (position x,y,z, pan/tilt/roll, FOV = 7 params) predicted by a SegFormer-based CNN-Transformer; then keypoint-based refinement: ResNet18 detects 74 keypoints (line / grass-line intersections), brute-force search over camera-parameter deltas in steps of 0.05 up to +-0.15 minimising L2 distance between reprojected line sets and the ideal pitch model (outlier keypoints discarded); then a Savitzky-Golay filter over camera parameters (max correction +-2 deg for angles, +-2 m for positions; predictions delayed 15 frames = 0.5 s). Post-processing: tracklet merging and linear interpolation of gaps. — [arXiv 2504.06357](https://arxiv.org/pdf/2504.06357)
- 2025 winner KIST-GSR (GSR-1): players "mapped to pitch coordinates using a multi-frame keypoint detection model" (no further calibration detail in the report). — [arXiv 2508.19182](https://arxiv.org/pdf/2508.19182)
- 2025 3rd (lianyou, GSR-3): 27 field-line classes (16 points each) detected by custom RT-DETR; "radial distortion correction (Wang et al.) combined with PnLCalib; fixed camera parameters simplified to [pan, tilt, roll, focal] and temporally smoothed". — [arXiv 2508.19182](https://arxiv.org/pdf/2508.19182)
- 2025 4th (Playbox & MIXI, GSR-4): "Camera calibration extends BroadTrack by propagating calibration parameters temporally via optical flow, initialised from multiple stable frames"; linear interpolation + Savitzky-Golay filter for trajectories (61.64 GS-HOTA). — [arXiv 2508.19182](https://arxiv.org/pdf/2508.19182)
- 2025 GSR-13 (HTWK Leipzig, calibration-only submission): NBJW homography refined by tracking keypoints with Lucas-Kanade optical flow + RANSAC; initial frames chosen by IoU of U-Net pitch segmentation vs NBJW-warped template; improvement "up to 4 %" over NBJW on GS-HOTA and LocSim, mostly on frames lacking prominent pitch features. — [arXiv 2508.19182](https://arxiv.org/pdf/2508.19182)
- GSR-9 used optical flow for camera-motion compensation in tracking. — [arXiv 2508.19182](https://arxiv.org/pdf/2508.19182)
- SoccerNet 2026: I could not find a 2026 GSR leaderboard; a HF mirror lists 2026 tasks as Player-Centric Ball Action Spotting and Novel View Synthesis with GSR listed as "SN-GSR-2025" and a 2027 "Long Form GSR" entry (unofficial mirror, unverified). — [HF mirror](https://huggingface-gw-kz.just-ai.com/SoccerNet)

### Inferences
- A pitch-position target consistent with GS-HOTA: the metric gives zero credit past 5 m and half-credit around ~2.4 m, so an acceptable calibration + foot-point pipeline should hold typical player error well under ~2 m, and failures should be detected, not silently emitted (a 0.23 % sequence is a total calibration failure, not a gradual degradation).
- Because the 2024 baseline's oracle-calibration ceiling is only ~51 for TVCalib-vs-GT, calibration quality is a first-order lever, but top-2 entries are almost tied (63.9 vs 63.8), suggesting calibration + tracking are saturating in the benchmark and remaining gains lie in identity (jersey) and in sequences with sparse pitch features.
- All published GSR pipelines project the bottom-centre of the box through the ground homography; none (in the sources I could read) uses off-plane cues (head, goalposts) to reduce foot-point bias.

### Gaps
- Could not retrieve the KIST-GSR (2025 winner) technical report; the challenge report gives only a one-sentence description.
- No source gives the per-module GS-HOTA contribution for the 2024/2025 winners (only the 2024 baseline ablation above).
- 2026 SoccerNet GSR challenge results: not found.

---

## 2. Error propagation: pixels to metres, homography vs full camera (PnP-style) model, radial distortion

### Takeaway
On a ground-plane homography the pixel-to-metre scale degrades with depth: for a typical 1080p main-camera wide shot, 1 px of vertical error is ~0.03 m at the near side but ~0.2 m at the far touchline (and ~0.1 m on a tighter zoom), while a 0.1 deg tilt error or 1 % focal error gives 0.5-0.7 m at the far side [computed]. Published calibration benchmarks use pixel thresholds (JaC@5 px at 960x540) that therefore correspond to roughly 1-2 m on the far touchline. Radial distortion modelling is the largest single calibration improvement reported (BroadTrack) and homography-only models are measurably worse than camera models with distortion (ProCC).

### Cited Findings
**Pixel-error geometry (qualitative, cited)**
- "Pixel errors affect real-world distances unevenly. Points near the horizon, such as the far sideline, can produce much larger errors in yards than the same pixel error on the near sideline." — [StatsBomb, How To Map Homography](https://blogarchive.statsbomb.com/articles/football/creating-better-data-how-to-map-homography/)
- From a single image the camera parameters cannot be uniquely recovered, so StatsBomb estimates a homography (8 DoF, minimum four non-collinear points), noting more points make the fit robust to noise. — [StatsBomb](https://blogarchive.statsbomb.com/articles/football/creating-better-data-how-to-map-homography/)
- Theiner et al.: back-projecting known keypoints with registrations of about 95 % IoU gives "about 1 m" mean absolute error in metres; wrong focal length is a cited failure cause. — [Theiner et al., arXiv 2110.11107](https://ar5iv.labs.arxiv.org/html/2110.11107)
- BroadTrack: NBJW's estimated focal point can wander up to ~20 m within one sequence whereas a tripod constraint keeps it near the true point; SoccerNet's 5 px tolerance is at 960x540, a quarter of the HD resolution used in its tracking evaluation. — [BroadTrack, arXiv 2412.01721](https://arxiv.org/html/2412.01721v1)

**Calibration quality numbers (pixels)**
- ProCC protocol / WorldCup 2014 (186 test images): homography (WC14 annotations) JaC5 67.4, reprojection error 3.07; homography (CARWC annotations) JaC5 79.1, error 1.79; pinhole + 1 radial term (Xeebra) JaC5 92.5, error 1.44. On SoccerNet-calibration: simplified pinhole JaC5 78.7 / JaC2 40.2 / error 4.51; pinhole + one radial term JaC5 83.1 / JaC2 54.3 / error 4.01. — [Magera et al., ProCC, arXiv 2404.09807](https://arxiv.org/html/2404.09807v1) (reprojection-error units are not stated in the paper text I read; presumably pixels).
- BroadTrack ablation on sn-gamestate (1920x1080, 49 sequences x 750 frames): no flow/tripod/distortion JaC5 42.09, MRE 5.74 px; adding radial distortion (k1 only, Brown-Conrady) alone JaC5 54.1, MRE 5.04 px ("largest single gain"); full model JaC5 56.88, MRE 5.02 px, median 2.37 px. Compared with NBJW: JaC5 37.14, JaC10 68.24, MRE 10.28 px; TVCalib JaC5 19.88, JaC10 50.42, MRE 12.4 px. — [BroadTrack](https://arxiv.org/html/2412.01721v1)
- BroadTrack camera model: pinhole with single focal length, square pixels, principal point at centre, radial k1 only (higher-order radial and tangential terms discarded as superfluous), plus tripod constraint (pan/tilt axes meet at a fixed point). — [BroadTrack](https://arxiv.org/html/2412.01721v1)
- PnLCalib (SN22-test-center): JaC@5/10/20 = 75.8/89.7/91.9 (SV) and 80.6/89.9/92.4 with the PnL refinement, vs TVCalib 54.8/78.5/90.4. On WC14: PnLCalib+PnL 85.2/94.0/96.1 vs TVCalib 39.9/71.9/90.5. The method assumes zero skew and principal point at centre and neglects distortion; needs >= 4 visible keypoints. — [PnLCalib, arXiv 2404.08401](https://arxiv.org/html/2404.08401v4)
- Homography/IoU: PnLCalib IoU-part 97.0 mean / 98.2 median on WC14 and 98.6/98.9 on TS-WorldCup. — [PnLCalib](https://arxiv.org/html/2404.08401v4)
- ProCC authors conclude homography-based methods "do not accurately represent the field in the image, especially when lens distortion is present", and that camera models with radial distortion give significantly better reprojection accuracy. — [Magera et al., ProCC](https://arxiv.org/html/2404.09807v1) (summary of the paper's conclusion via search snippet and the tables above)
- A homography only holds for planar points; StatsBomb/SoccerNet-v2 pipelines map "the middle of the bottom edge of each bounding box" through the field homography, which "implicitly assumes that point lies on the ground plane"; SoccerNet-v2 does not quantify that error. — [Cioppa et al., SoccerNet-v2 calibration, arXiv 2104.09333](https://ar5iv.arxiv.org/html/2104.09333)
- Full camera calibration (vs homography) can use non-planar points such as goal posts and crossbars (ProCC also annotates goal posts, 53,577 goal-post elements in SoccerNet-calibration). — [PnLCalib](https://arxiv.org/html/2404.08401v4); [ProCC](https://arxiv.org/html/2404.09807v1)

**My own geometry computation [computed, not cited]**
Setup: 1920x1080 frame, main camera 18 m high, 12 m behind the near touchline, centred on the halfway line, on a 105x68 m pitch, pinhole, principal point at centre. Values are ground-plane error at the point, depth direction unless stated.

| Point | 1 px vertical error | 1 px horizontal | 0.1 deg tilt error | 1 % focal error | radial k1=-0.05 ignored |
|---|---|---|---|---|---|
| hFOV 55 deg, 20 m from near touchline (y=-20) | 0.03 m | 0.02 m | 0.10 m | 0.12 m | 0.03 m |
| hFOV 55, far side of centre circle (y=+9) | 0.10 m | 0.03 m | 0.33 m | 0.11 m | 0.00 m |
| hFOV 55, far touchline, x=0 | 0.20 m | 0.04 m | 0.66 m | 0.56 m | 0.07 m |
| hFOV 55, far touchline, x=+25 m (frame edge) | 0.21 m | 0.04 m | 0.69 m | 0.70 m | 0.41 m |
| hFOV 35 (tighter), far touchline x=0 (aim shifted) | 0.12 m | 0.03 m | 0.63 m | 0.46 m | 0.21 m |

- Rule of thumb from this: a 5 px (1080p) vertical error is ~1 m at the far touchline and ~0.15 m at 20 m from the near touchline; a SoccerNet-scale 5 px at 960x540 (= 10 px at 1080p) is ~2 m at the far touchline. Lateral error per px is ~5x smaller than depth error.
- Foot-point height bias: if the "ground" point the detector gives is actually 0.1 m / 0.3 m above the turf (box bottom above the foot, or a jump), the projected position is pushed away from the camera by ~0.15 / 0.44 m at y=-20, 0.26 / 0.78 m at the centre and 0.45 / 1.36 m at the far touchline.
- Radial distortion: a k1 of -0.05 (illustrative value, not from a source) ignored changes projected positions by tens of cm only near frame edges on the far side (up to ~0.4 m here); larger k1 on wide-angle lenses would scale up proportionally.
(Computation scripts are not saved; geometry is a standard pinhole model with back-projection to z = 0.)

### Inferences
- The depth-direction (vertical image) error dominates at the far side; camera tilt/focal-length error matters more than random per-keypoint noise because it acts as a systematic shift of the whole far half of the pitch. A calibration with ~1-2 px reprojection error on lines (e.g. BroadTrack median 2.37 px at 1080p) corresponds to a few tenths of a metre at the far touchline, consistent with the ~1 m target used for "suitable" analytics accuracy (see section 3).
- Foot-point (bounding-box) bias of 0.1-0.3 m of height already matches a 5-10 px calibration error at the far side, so bottom-of-box quality (occlusion, box tightness) can rival calibration error; need an evaluation that measures the combined error.
- For ground points, a homography and the ground-plane homography derived from a full camera model are the same mathematical map; the benefits of the full camera model are (i) constraints from physical parameters (shared focal length, tripod position, temporal consistency), (ii) distortion modelling, (iii) ability to use non-planar landmarks (goalposts) and project off-plane points. The sources show the camera-model-with-distortion wins on the benchmark metrics, but I found no paper that isolates the effect on player foot-point error in metres.

### Gaps
- No published error-vs-distance-from-camera analysis against tracking ground truth; Theiner et al. explicitly do not analyse it.
- No source with typical radial distortion k1 values for broadcast main cameras (BroadTrack states only that k1 alone suffices; value not found).
- TVCalib's paper (projection error in metres) could not be retrieved (HTTP 403); its metre-level results are therefore not reported here.

---

## 3. Reported / validated positional accuracy of broadcast-derived tracking; what is acceptable for analytics

### Takeaway
Independent validation of commercial broadcast tracking against a multi-camera optical system finds position RMSE of 1.68-16.39 m overall, ~0.4-1.1 m when players are actually detected, and 4.6-19 m on frames where the player is off-screen (extrapolated). Vendors (SkillCorner, StatsBomb 360) publish no metre-level position error. Practitioner thresholds used for "suitable" spatial analytics are ~1 m position and ~1 m/s speed.

### Cited Findings
- Crang et al. (J Sports Analytics 2026; arXiv 2508.19477): one 2022 World Cup match, three commercial CV/AI broadcast-tracking providers vs TRACAB Gen 5; position RMSE 1.68-16.39 m, speed RMSE 0.34-2.38 m/s, total match distance mean bias from -1745 m (-21.8 %) to +1945 m (+24.3 %). Best config: Provider 2, 720p tactical feed, RMSE 1.68 m (mean bias 0.68 m, LoA -2.35 to 3.71 m); Provider 1 best 3.14 m (1080p programme feed); Provider 3 best 6.24 m. — [Crang et al.](https://arxiv.org/pdf/2508.19477)
- When players were detected: position RMSE ~0.44-1.14 m (Providers 1 and 2) and speed RMSE < 0.46 m/s; undetected (extrapolated) frames: position RMSE 4.62-19.38 m across providers/feeds. Detection rates: programme feed 36-64 %, tactical feed 89-96 %. — [Crang et al.](https://arxiv.org/pdf/2508.19477)
- Criteria used: position accuracy of ~1 m considered "suitable" for spatial tracking (distances between players, not offside), speed RMSE ~1 m/s (about 10 % of max speed) "aligned with current industry standards"; the authors conclude overall position accuracy "is not currently suitable" for spatial tracking, but is suitable "in isolated moments when players are detected (e.g., set play or goal)". Providers should use a tactical feed. — [Crang et al.](https://arxiv.org/pdf/2508.19477)
- Reference system: TRACAB Gen 5 position RMSE = 0.08 m, speed RMSE 0.08 m/s (as quoted in Crang et al.); original validation Linke et al. 2020: TRACAB Gen4/Gen5 vs VICON: position RMSE ~0.09/0.08 m, speed 0.09/0.08 m/s, total-distance deviation 0.42 %/0.27 %. — [Crang et al.](https://arxiv.org/pdf/2508.19477); [Linke et al., PLOS ONE (PMC7064167)](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7064167/) (position/distance figures via search summary of the paper)
- Theiner et al. (WACV 2022, pitch registration + detection from Bundesliga TV): vs a multi-camera tracking system (itself assumed ~1 m error), after self-verification mean position error 1.81-2.82 m, median 1.14-1.71 m; 55-79 % of positions within 2 m and 71-88 % within 3 m; without self-verification TV12 mean error rises to 3.69 m. Domain expert judged <= 2 m "correct" and < 3 m "potentially useful". — [Theiner et al.](https://ar5iv.labs.arxiv.org/html/2110.11107)
- SkillCorner (vendor study): 10 cameras at Sunderland's Stadium of Light varying height/angle/offset/zoom, compared with the Premier League official tracking; player identification ~90-98 %, ball tracking > 92 % for most camera positions, PSV-99 "closely aligned"; the page reports no metre-level position error and no per-camera numbers. Positional data is unavailable when the director cuts to close-ups or replays; SkillCorner single-camera tracking reportedly got FIFA Basic certification (March 2024). — [SkillCorner article](https://www.skillcorner.com/articles/does-camera-position-and-capture-style-affect-tracking-quality) (certification claim per search summary, not verified in the primary text)
- StatsBomb 360: freeze-frames from broadcast video; many broadcast frames do not show all 22 players; calibrated via annotators labelling pitch lines + a ResNet predicting camera pose, with camera assumptions reducing unknowns; deployed only for shots at the time of that post; no accuracy figures in metres or pixels, validation described only as visual (template projected in red). — [StatsBomb SB Labs: Camera Calibration](https://blogarchive.statsbomb.com/articles/soccer/sb-labs-camera-calibration/); [Hudl/StatsBomb xPass 360 post and freeze-frame descriptions via search](https://www.hudl.com/blog/xpass-360-upgrading-expected-pass-xpass-models)
- Second Spectrum broadcast tracking: I found no accuracy documentation.

### Inferences
- Accuracy targets for analytics, mapped to the evidence: (a) ~1 m / 1 m/s for spatial and physical use (Crang), meaning calibration contribution must be a fraction of that, i.e. a few tenths of a metre at the far side => roughly <= 2-3 px reprojection error at 1080p wide shots (from my geometry table); (b) ~2 m (Theiner expert threshold) is adequate for formations/heatmaps and event context but not for pass-geometry/offside-like calculations; (c) matchwide distance covered is not recoverable from broadcast tracking by calibration alone: biases of +-22-24 % came from undetected frames, not from projection error.
- Position error for undetected frames (4.6-19 m) dwarfs calibration error, so for broadcast tracking the effective accuracy target is "high availability of well-calibrated detected frames" rather than 0.1 m calibration perfection.

### Gaps
- No independent validation of SkillCorner, Second Spectrum or StatsBomb 360 positional accuracy in metres found. Mills et al. (Sports Engineering 2026, automatic event detection from broadcast-derived tracking) was not readable (access blocked).
- No source on tolerances for heatmaps or formations specifically; my statements about their acceptable error are inferences from the 1-2 m thresholds above.
- GPS-vs-broadcast distance-covered validation: not found.

---

## 4. Temporal smoothing, occlusion and non-calibratable shots

### Takeaway
Top systems smooth the camera parameters (Savitzky-Golay, optical-flow propagation, tripod constraints) rather than only the projected player positions, reinitialise from scratch when calibration confidence drops, and fill short gaps by linear interpolation. Replays/close-ups are simply dropped or flagged (vendors) or filtered by a plausibility score / self-verification.

### Cited Findings
- Constructor Tech: Savitzky-Golay on camera parameters with a 15-frame (0.5 s) delay; corrections limited to +-2 deg and +-2 m; tracklet gaps (occlusion or leaving frame) filled by linear interpolation. — [arXiv 2504.06357](https://arxiv.org/pdf/2504.06357)
- Playbox & MIXI: calibration parameters propagated temporally via optical flow, initialised from multiple stable frames; linear interpolation + Savitzky-Golay for trajectories. — [arXiv 2508.19182](https://arxiv.org/pdf/2508.19182)
- lianyou: radial distortion correction + PnLCalib with fixed parameters reduced to [pan, tilt, roll, focal], temporally smoothed. — [arXiv 2508.19182](https://arxiv.org/pdf/2508.19182)
- BroadTrack: Lucas-Kanade optical flow to avoid drift where markings are sparse; reinitialisation when the Jaccard confidence falls below 0.5 during tracking (needs only two point correspondences); on sn-gamestate reinitialised 33 times across 49 sequences with 15-frame average loss; on a 20-min Bundesliga sequence 60 times with 4-frame average loss; tracking drifts when markings become very sparse; speed 16 fps at HD on two RTX 4090. — [BroadTrack](https://arxiv.org/html/2412.01721v1)
- BroadTrack reinit on SoccerNet-calibration (960x540): JaC5 75.25, JaC10 86.8, completeness 69.8 % (NBJW 73.7/86.7, completeness 77.5 %; TVCalib 52.9/73.4, 66.5 %); the paper says this evaluation covers only views with confidence s < 0.2 (as reported; the paper notes it lowers completeness and filters out e.g. fish-eye views). — [BroadTrack](https://arxiv.org/html/2412.01721v1)
- Non-calibratable shots: close-ups with few/no landmarks and fisheye in-goal shots can make calibration impossible; at least four visible keypoints are needed. — [PnLCalib](https://arxiv.org/html/2404.08401v4)
- SoccerNet-v2 pipeline: its student network outputs a binary plausibility score that filters close-up and public views. — [Cioppa et al.](https://ar5iv.arxiv.org/html/2104.09333)
- Theiner et al.: self-verification drops frames where projected players fall well outside the pitch (tolerance rho = 3 m; 2-5 m reasonable), keeping 84-92 % of frames; also notes distant false positives (referees, subs) inflating mean error, so they report mean of the best 80 % of per-frame distances. — [Theiner et al.](https://ar5iv.labs.arxiv.org/html/2110.11107)
- SkillCorner: no positional data when the director shows close-ups or replays; position must be estimated from a single viewpoint (no triangulation). Crang et al.: extrapolating off-screen players gives much worse position error (4.6-19 m). — [SkillCorner](https://www.skillcorner.com/articles/does-camera-position-and-capture-style-affect-tracking-quality) (via search summary of the product article); [Crang et al.](https://arxiv.org/pdf/2508.19477)

### Inferences
- Smooth camera parameters (7 DoF or [pan,tilt,roll,focal]) in a constrained filter rather than smoothing player xy, because camera errors are common-mode across all players and smoothing xy wouldn't remove them. A ~0.5 s non-causal window is the published practice for offline processing.
- Representing per-frame calibration confidence (e.g. Jaccard/IoU of warped template vs segmentation, as BroadTrack and HTWK do) and emitting "no position" instead of extrapolating is supported by Crang's finding that extrapolated positions have 4-19 m errors.

### Gaps
- No numbers on how much smoothing improves position error in metres (papers report only GS-HOTA/JaC deltas, e.g. up to 4 % GS-HOTA/LocSim from HTWK flow refinement).
- No source for handling replays other than dropping them (vendors) or confidence filtering.

---

## 5. Evaluation protocols and how to build a small ground-truth test set

### Takeaway
Three complementary evaluations exist: (1) calibration-level, camera-model-agnostic ProCC/JaC on annotated pitch-element points (5 px at 960x540, 2 px for strict), (2) IoU of the warped template (part-visible and whole-field), (3) metric player-position error against ground-truth pitch coordinates (GSR dataset, or tracking data with ~1 m reference noise). For a small in-house set the best option is to annotate points on pitch markings plus a handful of ground-truth foot positions, and report errors in metres broken down by depth.

### Cited Findings
- ProCC: JaC_tau = TP/(TP+FN+FP) over field elements; an element counts as correct only if all annotated points of the element lie within tau px of its projection; tau = 5 px (JaC5) typical, 2 px (JaC2) for precise methods; ground truth is semantic point annotations along lines, circles, ellipses and goal posts (26 semantic classes), not homographies, so any camera model can be scored; completeness is folded into FN (SoccerNet reports completeness ratio CR separately). — [Magera et al., ProCC](https://arxiv.org/html/2404.09807v1); CR in [PnLCalib](https://arxiv.org/html/2404.08401v4)
- Further thresholds reported in the SoccerNet calibration challenge: JaC@5/10/20 and completeness rate (see PnLCalib tables above); the "5 px" is at 960x540. — [PnLCalib](https://arxiv.org/html/2404.08401v4); [BroadTrack](https://arxiv.org/html/2412.01721v1)
- Homography benchmarks: IoU-part (visible region of the warped template vs ground truth) and IoU-whole, mean and median; best published IoU-part ~97-99 %; WC14 IoU-whole 93-95 %. Projection error (in metres) is also used for homography methods. — [PnLCalib](https://arxiv.org/html/2404.08401v4)
- A ~95 % IoU registration corresponds to ~1 m mean back-projection error of known keypoints; some frames with IoU < 90 % still pass self-verification and give clearly wrong player positions. — [Theiner et al.](https://ar5iv.labs.arxiv.org/html/2110.11107)
- Player-position evaluation in Theiner et al.: per-frame mean distance between matched predicted and ground-truth positions, reported as mean, median and fraction of positions within 2 m / 3 m; matching of detections to GT on the pitch plane; ground truth from a multi-camera tracking system (assumed ~1 m error). — [Theiner et al.](https://ar5iv.labs.arxiv.org/html/2110.11107)
- GSR: GS-HOTA with tau = 5 m on 200 sequences with pitch-coordinate GT for every athlete; datasets and evaluation through TrackLab/sn-trackeval; calibration alone can be scored with the SoccerNet calibration challenge (JaC) on the same annotation conventions. — [sn-gamestate](https://github.com/SoccerNet/sn-gamestate); [Somers et al.](https://arxiv.org/pdf/2404.11335)
- Concurrent validity design to emulate: align providers to reference by phase shift (frame offset) before computing RMSE, report RMSE and mean bias with 95 % limits of agreement, and split into detected vs undetected samples. — [Crang et al.](https://arxiv.org/pdf/2508.19477)

### Inferences
- Suggested small test set (not from a source): (1) 50-100 frames spanning camera positions/zooms and far/near-side play with ProCC-style point annotations of lines and goal posts (to compute JaC2/JaC5, IoU-part and a metric "reprojection error in metres" by back-projecting annotated points to the pitch template); (2) 200-500 player foot points annotated as ground-contact pixel positions with known pitch coordinates by choosing frames where players stand on or next to markings (penalty spot, line intersections, corner flags, set-piece wall lines), or by using a synchronized tracking feed if available; (3) report mean/median/90th-percentile error in metres, fraction within 1 m / 2 m / 5 m, split by distance from camera and by calibration-confidence bin, with extra reporting of availability (fraction of frames with valid calibration).
- Because pixel thresholds hide depth dependence, the metric to prioritise for our use is projected-position error in metres vs depth, with JaC5/JaC2 and IoU as calibration-only diagnostics; use 1 m and 2 m (Crang, Theiner) as target thresholds and the 5 m GS-HOTA tolerance as the failure threshold.
- Also evaluate stability: frame-to-frame jitter of static pitch points (e.g. corner flags, line crossings) reprojected through the sequence calibration (BroadTrack reports focal-point wander of up to 20 m for NBJW, which would show up here).

### Gaps
- Could not verify how TVCalib defines "projection error in metres" (paper blocked), nor find a ready-made public set with GT pitch coordinates other than SoccerNet-GSR and the Bundesliga set of Theiner et al. (not public in sources I read).
- ProCC paper's own recommended thresholds beyond 5 px / 2 px were not found.

---

## Source list (primary documents read)
- Somers et al., SoccerNet GSR / GS-HOTA: https://arxiv.org/pdf/2404.11335
- Constructor Tech, From Broadcast to Minimap: https://arxiv.org/pdf/2504.06357
- SoccerNet 2025 Challenges Results: https://arxiv.org/pdf/2508.19182
- sn-gamestate: https://github.com/SoccerNet/sn-gamestate
- BroadTrack: https://arxiv.org/html/2412.01721v1
- PnLCalib / NBJW: https://arxiv.org/html/2404.08401v4
- ProCC: https://arxiv.org/html/2404.09807v1
- Theiner et al. (positional data from broadcast): https://ar5iv.labs.arxiv.org/html/2110.11107
- Cioppa et al. SoccerNet-v2 calibration + localisation: https://ar5iv.arxiv.org/html/2104.09333
- Crang et al., concurrent validity: https://arxiv.org/pdf/2508.19477
- Linke et al., TRACAB validity: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7064167/
- SkillCorner: https://www.skillcorner.com/articles/does-camera-position-and-capture-style-affect-tracking-quality
- StatsBomb: https://blogarchive.statsbomb.com/articles/soccer/sb-labs-camera-calibration/ and https://blogarchive.statsbomb.com/articles/football/creating-better-data-how-to-map-homography/
