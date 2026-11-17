# Training data for learned football pitch calibration: public datasets, synthetic data, augmentation, self-training

(Researched 2026-11-14. Sources were fetched through web tools; where a page was unreadable or silent, that is stated as a gap. Nothing here is legal advice. "Licence" below means what I could verify from the dataset's own pages, not a legal opinion.)

## Q1. Which public labelled datasets exist, how big, how accessed, and what are the licence terms (research-only vs commercial)?

### Takeaway
The usable open calibration data are SoccerNet-Calibration-2023 (about 22k line-annotated broadcast images), WorldCup 2014 (395 images), TS-WorldCup (3,812 images) and, as an auxiliary, SN-GSR-2025 (GPL-3.0-tagged). None of the calibration datasets carries an explicit data licence I could verify (no licence file, no dataset card, no licence tag), and the underlying pixels are broadcast footage owned by third parties, so treat all of them as research-only unless you obtain written clarification. Only the SoccerNet code repos and a few task datasets carry explicit OSS licences (GPL-3.0 / MIT / CC-BY-SA); the code licence does not clearly cover the broadcast imagery.

### Cited Findings

**SoccerNet-Calibration (a.k.a. "SoccerNet-v3 calibration", `calibration-2023` task)**
- Annotation: 26 semantic pitch classes (e.g. "Big rect. left bottom", "Circle central"); straight lines are given by two extremity 2D points, circles by an unordered list of points tracing them; goal posts are annotated; some images have extra points along lines. Not a homography or a full camera; submissions are per-frame camera JSON (pan/tilt/roll deg, position m, focal lengths, principal point, OpenCV distortion). Metric JaC@5 x Completeness. — [sn-calibration README](https://github.com/SoccerNet/sn-calibration)
- Origin: SoccerNet 400 broadcast videos of whole games (the repo README says 400; the v2 calibration paper says 500 games). Camera types: main and 16m broadcast cameras (bottom side of pitch), plus cameras behind a goal that create left/right ambiguity, and replay frames. No formal taxonomy given. — [sn-calibration README](https://github.com/SoccerNet/sn-calibration)
- SoccerNet-v3 as a whole: 1,324,732 annotations on 33,986 soccer images; lines and goal parts classified into 20 and 6 classes; includes live and replay frames. Source is a third-party summary of the paper. — [PMC9210334 / search result](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9210334/)
- Image counts for the calibration task, conflicting across sources: 21,132 images with 167,589 line/circle elements and 53,577 goal-post elements (ProCC paper); 22,816 images for "SN23" (PnLCalib paper); 14,512 real images used as the fine-tuning set in SoccerSynth-Field. These are probably different splits/versions; the dataset itself gives no per-split counts. — [ProCC arXiv 2404.09807](https://arxiv.org/html/2404.09807v1); [PnLCalib arXiv 2404.08401](https://arxiv.org/html/2404.08401v4); [SoccerSynth-Field arXiv 2503.13969](https://arxiv.org/html/2503.13969)
- Hugging Face repo `SoccerNet/SN-Calibration-2023`: total 3.98 GB (train.zip 2,556,552,506 B; valid.zip 501,188,633 B; test.zip 500,158,845 B; challenge.zip 421,030,309 B). Not gated (`gated=False`), no dataset card, no licence tag. — [HF API](https://huggingface.co/api/datasets/SoccerNet/SN-Calibration-2023?blobs=true) and [HF page](https://huggingface.co/datasets/SoccerNet/SN-Calibration-2023)
- Download: `pip install SoccerNet`; `SoccerNetDownloader(LocalDirectory=...).downloadDataTask(task="calibration-2023", split=["train","valid","test","challenge"])`; the package now pulls migrated datasets from Hugging Face. — [sn-calibration README](https://github.com/SoccerNet/sn-calibration); [soccer-net.org/data](https://www.soccer-net.org/data)
- Licence status: the `SoccerNet/sn-calibration` repo has no LICENSE file (contents listing: .gitattributes, .gitignore, ChallengeRules.md, README.md, doc, images, requirements.txt, resources, src); ChallengeRules.md says nothing about licence, only that participants may use images as input. The generic `SoccerNet/SoccerNet` devkit repo is MIT. The soccer-net.org data page states no research-only / non-commercial wording; it only says "some datasets containing protected video content are gated" and need NDA acceptance via Hugging Face (old NDA form deprecated). Calibration-2023 is not among the gated/NDA repos (those are SoccerNet_raw_HQ, BAS, PCBAS, ActionAnticipation, SN-VQA-2027, challenge raw video). — [GitHub contents API](https://api.github.com/repos/SoccerNet/sn-calibration/contents); [SoccerNet repo README](https://github.com/SoccerNet/SoccerNet); [HF org listing API](https://huggingface.co/api/datasets?author=SoccerNet)
- There is no `SN-Calibration-2024` or `SN-Calibration-2025` dataset in the SoccerNet Hugging Face org as of this check (the 2024/2025 calibration-related work is carried by the GSR task, below). — [HF org listing API](https://huggingface.co/api/datasets?author=SoccerNet)
- Leaderboard baseline numbers in the README (JaC@5 11.7%, completeness 68%) disagree with its own leaderboard row (13.54 / 61.54); not explained. — [sn-calibration README](https://github.com/SoccerNet/sn-calibration)

**SoccerNet-GSR (`SN-GSR-2024` / `SN-GSR-2025`)**
- 2025 repo: 35.1 GB total (train.zip 9,755,097,159 B; valid.zip 11,169,497,921 B; test.zip 8,845,186,210 B; challenge.zip 5,310,696,725 B) in the zip listing; the viewer shows 123,167 image rows (train 42.8k / valid 43.6k / test 36.8k). The HF 2024 repo lists identical file sizes. HF licence tag `gpl-3.0`; not gated; no NDA. — [HF API SN-GSR-2025](https://huggingface.co/api/datasets/SoccerNet/SN-GSR-2025?blobs=true); [HF page](https://huggingface.co/datasets/SoccerNet/SN-GSR-2025)
- The dataset-card text does not describe annotations. Annotation content (per-frame pitch/camera info for GSR, tracking IDs, jersey/team) comes from the GSR paper/README, which I did not fully read; the SoccerMaster paper describes SN-GSR as 42.8K manually annotated frames. — [SoccerMaster arXiv 2512.11016](https://arxiv.org/html/2512.11016)
- The `SoccerNet/sn-gamestate` code repo is GPL-3.0. GPL-3.0 on a dataset of broadcast frames is a licence statement by the SoccerNet team, not by the broadcasters; I would not rely on it for commercial rights over the images. — [GitHub licence API](https://api.github.com/repos/SoccerNet/sn-gamestate/license)
- Winning teams' use: 14 teams, 66 submissions, baseline GS-HOTA 29.01, top 63.90 (SoccerNet 2025 report). Calibration pipelines named: NBJW, PnLCalib, BroadTrack, TVCalib. — [SoccerNet 2025 Challenges Results](https://arxiv.org/html/2508.19182)

**WorldCup 2014 (WC14, Homayounfar et al., CVPR 2017)**
- Table of sizes from ProCC: 395 images with homography annotation, open data; standard split 209 train / 186 test (from 10 + 10 games per Chen & Little). CVPR paper: images from 20 games of the 2014 World Cup; arXiv preprint says 259 images from 12 games (the two descriptions disagree). — [ProCC](https://arxiv.org/html/2404.09807v1); [Chen&Little](https://ar5iv.labs.arxiv.org/html/1810.10658); [Homayounfar CVPR 2017](https://openaccess.thecvf.com/content_cvpr_2017/html/Homayounfar_Sports_Field_Localization_CVPR_2017_paper.html)
- Download: author page lists "Soccer Data" at `data/soccer_data.tar.gz` (nhoma.github.io). No licence text found. — [nhoma.github.io](https://nhoma.github.io/)
- Known quality issue: WC14 homographies are imprecise (ProCC: JaC5 67.4 with original homographies vs 79.1 with the CARWC-corrected ones, vs 92.5 for a pinhole+radial-distortion model). So WC14's homography ground truth under-represents lens distortion; relevant to us because we model radial distortion. — [ProCC](https://arxiv.org/html/2404.09807v1)
- CARWC: a corrected re-annotation of WC14 + TS-WorldCup, 4,207 images (ProCC Table 1, marked open). I did not find its download page or licence. — [ProCC](https://arxiv.org/html/2404.09807v1)

**TS-WorldCup (Chu et al., CVPRW 2022)**
- 3,812 time-sequence images from 43 videos of the 2014 and 2018 World Cups, annotated with detailed field markings and homographies; ~10x WC14. Zip 588 MB at `https://cgv.cs.nthu.edu.tw/KpSFR_data/TS-WorldCup.zip`. Code repo `ericsujw/KpSFR` is MIT. No dataset licence on the project page. — [KpSFR project page](https://ericsujw.github.io/KpSFR); [CVPRW paper page](https://openaccess.thecvf.com/content/CVPR2022W/CVSports/html/Chu_Sports_Field_Registration_via_Keypoints-Aware_Label_Condition_CVPRW_2022_paper.html); [GitHub licence API](https://api.github.com/repos/ericsujw/KpSFR/license)

**Spiideo SoccerNet SynLoc (synthetic, SoccerNet 2026 challenge)**
- Synthetic athletes rendered onto images from real Spiideo camera installations; per-frame full camera calibration (pinhole + radial distortion); COCO-format annotations with `position_on_pitch`. 4K and fullhd versions. Download from research.spiideo.com after registering, or via `downloadDataTask(task="SpiideoSynLoc", split=["train","valid","test","challenge"])`. Devkit `Spiideo/sskit` has no detected licence file (GitHub licence API 404). Dataset licence not found. — [sskit README](https://github.com/Spiideo/sskit); [SCITEPRESS paper](https://www.scitepress.org/Papers/2025/131082)
- Caveat for us: these are fixed, elevated, wide-angle, static installation cameras (panorama-like), not PTZ broadcast main cameras, so only the camera model (radial distortion) transfers, not the viewpoint distribution. (Inference from the descriptions above.)
- Not listed in the HF SoccerNet org dataset browser as a dataset in my check; appears only as a 2026 challenge task. — [HF SoccerNet org](https://huggingface.co/SoccerNet)

**SoccerNet-Depth-2025 (synthetic game-render depth)**
- Licence tag gpl-3.0, not gated. Trained on synthetic frames (1918x1078). Not a calibration set. — [HF org listing](https://huggingface.co/api/datasets?author=SoccerNet); [2025 results](https://arxiv.org/html/2508.19182)

**SoccerTrack / TeamTrack (Tsukuba / Nagoya)**
- SoccerTrack (CVPRW 2022): 8K fisheye + drone video with bounding boxes, GNSS pitch coordinates (STATSPORTS APEX 10 Hz), and camera-calibration code; repo AtomScott/SoccerTrack. Fisheye/drone, not broadcast; licence not stated in sources seen. TeamTrack (2024-04-22) adds basketball/handball. — [CVPRW page](https://openaccess.thecvf.com/content/CVPR2022W/CVSports/html/Scott_SoccerTrack_A_Dataset_and_Tracking_Algorithm_for_Soccer_With_Fish-Eye_CVPRW_2022_paper.html); [TeamTrack](https://atomscott.github.io/TeamTrack)

**Other pitch-calibration code with explicit licences (relevant if reusing models/weights)**
- PnLCalib: GPL-2.0-or-later; No-Bells-Just-Whistles (NBJW): GPL-2.0. GPL weights/code in a commercial product are a licensing question in their own right. — [PnLCalib LICENSE](https://github.com/mguti97/PnLCalib); [NBJW LICENSE](https://github.com/mguti97/No-Bells-Just-Whistles)

**SoccerNet 2026-era tasks of marginal relevance** (listed on HF org page): SN-NVS-2026 (novel view synthesis, no licence tag), SN-VQA-2026 (CC-BY-SA-4.0), SN-PCBAS-2026 (NDA-gated), BannerReplacement (no licence tag). — [HF org listing](https://huggingface.co/api/datasets?author=SoccerNet)

### Inferences
- None of the calibration datasets is demonstrably cleared for commercial training; for a hobby/research project the practical risk is low, but the safe plan is: treat as research-only, do not redistribute images, and prefer using them for pretraining/validation with our own verified labels carrying the final fine-tune.
- Because ~22k SN images lack homographies (only line extremities + points), a model trained on them needs a conversion step (e.g. PnLCalib/TVCalib-style camera fit) or a line/keypoint-heatmap head that consumes those annotations directly (as NBJW does).
- WC14's weak ground truth means it is better used as a test/stress set than as a clean training set unless the CARWC corrections are obtained.

### Gaps
- Exact per-split image counts for SN-Calibration-2023 (zips not downloaded; counts conflict across papers: 21,132 / 22,816 / 14,512).
- Any actual licence text for SN-Calibration-2023, WC14, TS-WorldCup, CARWC, SynLoc. The SoccerNet NDA text was not accessible (personalised copy is emailed on HF access request).
- Whether SN-GSR-2025 includes per-frame full camera parameters (card says nothing; annotation schema not read).
- CARWC download location and licence.
- ISSIA-CNR and any other 2025-2026 pitch-keypoint releases: I did not search ISSIA in this pass and found no new 2025-2026 calibration datasets beyond SynLoc and SoccerMaster's SoccerFactory (announced as "will be publicly available", no release or licence seen).

---

## Q2. What synthetic data have published methods used, how are camera parameters sampled, what domain randomisation is used, and what are the measured gains?

### Takeaway
Three generations of synthetic data exist: (1) edge/line-template renderings from sampled PTZ poses (Chen & Little 2019, used to build a pose-retrieval database, not to train the detector); (2) game-engine simulators (Google Research Football for SoccerNet-GSR 2025 winner; Unreal Engine for SoccerSynth-Field) with randomised textures; (3) synthetic players on real fixed cameras (SynLoc). Quantified evidence that synthetic pretraining helps is thin: the only clean ablation I found (SoccerSynth-Field) shows +0.2 to +0.5 points of line-segmentation accuracy when fine-tuned on top of real data, with larger relative benefit when real data is small; the GSR 2025 winner combined synthetic + real but its paper's ablation numbers were not retrievable.

### Cited Findings

**Chen & Little, "Sports Camera Calibration via Synthetic Data" (CVPRW 2019)**
- Camera pose engine: camera centre drawn from a Gaussian with mean approx [52, -45, 17] m and sigma approx [2, 9, 3] m (estimated from the WorldCup training set; third component is probably height; coordinate frame is the paper's field frame, origin presumably at field centre); pan uniform in [-35, 35] deg; tilt uniform in [-15, -5] deg; focal length uniform in [1000, 6000] px; roll uniform in [-0.1, 0.1] deg. — [Chen&Little (ar5iv)](https://ar5iv.labs.arxiv.org/html/1810.10658)
- Scale: 100,000 sampled poses for the real-data feature-pose database; synthetic experiment: 10,000 train / 1,000 test at 1280x720; edge images resized to 320x180. Paper gives no line width. — same source
- Siamese network: 16-D feature; similar pairs defined by thresholds 1 deg pan, 0.5 deg tilt, 30 px focal length. — same source
- Two chained pix2pix-style conditional GANs (U-Net generators, L1 lambda 100, 200 epochs, batch 1, lr 2e-4; crops 256x256 from 300x300): one segments the playing surface, one detects field markings in the real image (this uses real human-annotated WC14 images, not synthetic images). Segmentation boundary alpha-blend width random in 30-50 px. The GANs, not the synthetic edge images, bridge real-to-template. — same source
- Results on WC14 test (186 images): IoU_part mean 94.5 / median 96.1; IoU_whole mean 89.4 / median 93.8. Ablation: without segmentation GAN 90.2; without LK refinement 91.5; neither 88.9. — same source
- The paper has NO real-only vs synthetic ablation; synthetic data serves only to build the pose database (robustness test: IoU_part >=92% if camera is within 5 m of a database pose, about 70% at 10 m). — same source
- Wider sports-camera-sampling context: a Stats-Perform-style follow-on (Sha et al., CVPR 2020) uses embedded templates + spatial transformer; I could not confirm it uses synthetic training data. — [search result, Sha et al.](https://openaccess.thecvf.com/content_CVPR_2020/papers/Sha_End-to-End_Camera_Calibration_for_Broadcast_Videos_CVPR_2020_paper.pdf)

**Golovkin et al., SoccerNet-GSR 2025 winner, CVPRW 2025 ("From Broadcast to Minimap")**
- Synthetic dataset: 40,000 images with camera parameters (position x,y,z; pan, roll, tilt; field of view), generated from a modified Google Research Football simulator; matches run at 10x real-time with camera position/orientation/zoom varied randomly during play; for every other simulator launch, grass, pitch-line and uniform textures drawn at random from a predefined set. — [Golovkin arXiv 2504.06357](https://arxiv.org/html/2504.06357v1)
- Camera sampling (uniform): x in [-60, 60] m, y in [40, 110] m, z in [-40, -10] m (their world frame has z downward so camera height is 10-40 m); tilt/pan set so the centre ray hits the pitch; roll fixed to 0. Supplement shows the synthetic distribution (position, FoV, pan, tilt) is wider than the real data's, which is more constrained. — same source; [supplemental](https://openaccess.thecvf.com/content/CVPR2025W/CVSPORTS/supplemental/Golovkin_From_Broadcast_to_CVPRW_2025_supplemental.pdf) (via search summary; the supplement text itself was not retrievable)
- Real data: 22,000 images with keypoint annotations (from the Field Keypoints dataset, labels fitted with a modified TVCalib using keypoints, loss in canonical world space). — Golovkin arXiv
- Model: SegFormer encoder, 512x288 input, about 5M params, predicts 7 camera parameters (pinhole, zero distortion, principal point fixed to centre) + auxiliary XY heatmap head; refined by a ResNet18 74-keypoint model with brute-force search over parameter corrections; Savitzky-Golay smoothing. — Golovkin arXiv
- Training schedule: phase 1 200k batches on combined real + synthetic; phase 2 200k batches real only. Batch 8, AdamW, max lr 5e-4, cosine, 20k warm-up. — Golovkin arXiv
- Augmentations (each p=0.5, Kornia): cutouts, flips, noise/blur, colour, shadows; geometric roll rotation, with pan inverted on horizontal flip. — Golovkin arXiv
- No synthetic-vs-real ablation numbers present in the text I retrieved. Whole-system GS-HOTA 63.81 vs 43.15 for 2nd place (secondary summary of the abstract; the SoccerNet 2025 report gives 63.90 top). — [search result](https://openaccess.thecvf.com/content/CVPR2025W/CVSPORTS/html/Golovkin_From_Broadcast_to_Minimap_Achieving_State-of-the-Art_SoccerNet_Game_State_Reconstruction_CVPRW_2025_paper.html); [2025 report](https://arxiv.org/html/2508.19182)

**SoccerSynth-Field (2025), Unreal Engine pretraining for line segmentation**
- 20,000 synthetic images from an Unreal Engine 5 simulator (SoccerSynth-Detection by Qin et al.). Randomised: jersey colours, audience, shades of green grass, ground patterns, lighting colour (white/yellow), multiple cameras, ground colour, and "fake field lines" (white lines between two random perimeter points as distractors). Keypoint instances exported per frame. Camera ranges not given. — [arXiv 2503.13969](https://arxiv.org/html/2503.13969)
- Model: DeepLabV3 (ResNet-101) with TVCalib's sn-segmentation pipeline; fine-tune on 14,512 SoccerNet-Calibration real images; test on 186 WC14 test images. Accuracy: real-only 98.3%; synthetic-only 95.7% (may not be on WC14; unclear); synthetic pretrain + real fine-tune 98.5%; simplest-randomisation variant 97.5%. — same source
- Less real data: 14,000 real frames 98.3 -> 98.5 with pretraining; 9,000: 97.8 -> 98.3; 6,000: 97.8 -> 98.1. i.e. +0.2 to +0.5 points. No results below 6,000 real frames. Task is segmentation accuracy, not homography error. — same source

**Other synthetic-data uses**
- SynLoc (above) synthesises players onto real camera backgrounds, solving the "labels for free" problem for localisation; calibration given. — [sskit](https://github.com/Spiideo/sskit)
- Nie et al. (WACV 2021) and Citraro et al. (2020): their papers' syntheses are not confirmed; Nie only mentions in related work that two prior methods use synthetic template databases; Citraro's abstract page does not mention synthetic data. I did not find evidence these used synthetic images. — [Nie search result](https://openaccess.thecvf.com/content/WACV2021/html/Nie_A_Robust_and_Efficient_Framework_for_Sports-Field_Registration_WACV_2021_paper.html); [Citraro arXiv 2003.14109](https://arxiv.org/abs/2003.14109)
- PnLCalib / NBJW (Gutierrez-Perez & Agudo): no synthetic data; keypoints derived from SoccerNet line annotations by line extraction and ellipse fitting; augmentation = random horizontal flip, colour jitter, Gaussian noise; no augmentation ablation. SV model lr 1e-5 fine-tuned on mostly non-replay frames. Numbers: SN23-test MV model JaC5 72.2 (76.7 with PnL refinement); WC14-test SV JaC5 77.6 (85.2 with PnL); WC14 IoU_part 96.4/97.0 (SV / SV+PnL). — [PnLCalib arXiv](https://arxiv.org/html/2404.08401v4)
- Camera-sampling statement in a CVPRW-adjacent thesis/paper: one source states the recommended camera parameters derived from WC14 are assumed to fit many stadiums for generating synthetic training data (only seen as a search-result snippet; source: arXiv 2110.11107, not verified). — [arXiv 2110.11107](https://arxiv.org/pdf/2110.11107)

**Camera-height / distance parameters for broadcast main cameras (as a prior)**
- From above: Chen&Little fitted a Gaussian on WC14 camera centres, mean approx [52, -45, 17] m, sigma approx [2, 9, 3] m. Their axis convention/origin was not stated in the retrieved text, so which component is height vs distance is uncertain (17 m is plausibly height, ~45 m distance from the pitch centre line); treat as approximate. Golovkin et al. sample camera height 10-40 m and distance from sideline 40-110 m (in their world frame, a wide prior). No source I found measured the PTZ pan/tilt/focal prior directly from SoccerNet; Chen & Little's pan [-35, 35] deg, tilt [-15, -5] deg, focal [1000, 6000] px at 1280x720 are the only published numeric ranges. — sources as above

### Inferences
- A defensible synthetic prior for a "main camera" is: camera height 10-35 m, offset 30-60 m from the near touchline along the pitch normal, pan +/-35-45 deg, tilt -5 to -25 deg, focal ~1000-6000 px at 720p (i.e. roughly 10-65 deg horizontal FoV; widen the zoom range for tight shots), small roll (<0.5 deg), radial distortion k1 within a small range. This blends Chen&Little (measured on WC14) with Golovkin's wider uniform ranges; widen it deliberately because Golovkin's supplement reports synthetic range > real range was helpful for robustness (claim from their supplement summary only).
- Evidence for the benefit of synthetic pretraining is weak and small where real data are plentiful (+0.2 to +0.5 points at 6-14k real images). For our regime (50 verified homographies) the benefit may be much larger, but no paper tests that regime; low-data (<1,000 real) is a gap in the literature.
- Domain randomisation lists that appear in the literature: grass colour/texture, stripe patterns, line colour, fake distractor lines, audience textures, lighting colour, shadows, cutout occluders, noise/blur, colour jitter. Line width, motion blur and lens distortion are not systematically studied in any paper I read.

### Gaps
- Golovkin's synthetic-only vs real-only vs combined ablation was not retrievable; the supplement PDF was not read directly.
- No paper found that studies lens-distortion randomisation or motion blur for pitch registration.
- No published measurement of real broadcast main-camera extrinsic distributions from SoccerNet itself (apart from WC14 stats in Chen & Little).
- Citraro/Sha/Nie synthetic-data usage unconfirmed (could not read full PDFs: pages 403 or compressed).
- Chen & Little's coordinate frame/units for the mean position were not stated in the text I retrieved; I flagged it as approximate.

---

## Q3. Augmenting real labelled frames: homography re-warp, crop/zoom/pan, horizontal flip, copy-paste, label propagation. What works, what are the pitfalls?

### Takeaway
Plane-induced warping (zoom/translate/flip applied with a compensating homography change) is the standard, cheap augmentation and gives small but consistent gains in IoU (about +0.2 to +1.4 points in the one controlled hockey study); horizontal flip needs a left/right label swap (or a template mirror) and, for pose-parametric models, a pan inversion. Non-planar content (stands, players, ads, goal frames) is warped inconsistently, so keep warps moderate and prefer crop/zoom (which is exactly a homography of the whole image) over strong perspective warps.

### Cited Findings
- Hockey rink localisation (Waterloo / Stathletes; 4,501 training images): each training image is randomly zoomed, flipped and colour-jittered, and the homography for the zoom/flip is composed into the ground-truth homography so labels stay consistent. Table 1: IoU_part / IoU_whole of refinement model 96.9 / 86.4 base -> 97.1 / 87.8 with homography+colour augmentation. Authors say it helped most on zoomed-in frames. — [Shang et al.](https://openjournals.uwaterloo.ca/index.php/vsl/article/download/5374/5662/24730)
- Same paper, copy-paste of players from other images onto the training frame to simulate occlusion: initial estimator 96.0/86.2 -> 96.2/86.3; refinement 97.2/87.9 -> 97.3/88.1 (tiny, +0.1 to +0.2). Notes its baseline is not directly comparable with prior work. — same source
- Same paper, synthetic perturbation for the refinement network: pick 4 points on a training image, perturb them, build the homography, warp the *template* edge image (and keep the original frame), regenerating a new perturbation every epoch instead of a fixed 8 per image. This is a self-generated synthetic-supervision trick that uses only real frames and their GT homography. — same source
- Jiang et al. (WACV 2020, "Optimizing Through Learned Errors"): trains a registration-error network on random perturbations of the GT homography (global translation U(-d_g, d_g) plus per-control-point local jitter) with the target template warped accordingly; augmentation = random crop of 90-100% of area, random horizontal flip, simulated shadow (50%-opacity black patch with random translation/rotation 0-45 deg/scale 0.5-2, Gaussian-blurred edges). Works with only 209 WC14 images plus a pseudo-labelled subset; also uses 39 images from the train-valid split as validation. — [Jiang et al.](https://arxiv.org/pdf/1909.08034) (text extracted from the PDF)
- Golovkin et al.: flip handled by "pan inverted on horizontal flip", roll rotation as geometric aug; pixel augs include cutouts, shadows. Same flip handling idea in PnLCalib (random horizontal flip) but no ablation. — [Golovkin](https://arxiv.org/html/2504.06357v1); [PnLCalib](https://arxiv.org/html/2404.08401v4)
- Flip and the SoccerNet class set: SoccerNet class names carry explicit left/right (e.g. "Big rect. left bottom"), and the README warns about the left/right ambiguity for cameras behind the goal. So a flip must swap left<->right classes and mirror the pitch template; for rotationally symmetric pitches it is also an honest relabelling because the pitch is symmetric, but any asymmetric information (e.g. which end the team attacks) is lost. — [sn-calibration README](https://github.com/SoccerNet/sn-calibration)
- Pitfalls reported: KaliCalib (basketball) relies on heavy augmentation plus "perspective-aware keypoint sampling" to deal with a small set. PnLCalib masks non-planar points and lines (goal frames, etc.) out of the loss when projecting ground-truth homographies to keypoint targets. — [KaliCalib](https://ar5iv.arxiv.org/html/2209.07795) (via search snippet); [PnLCalib](https://arxiv.org/html/2404.08401v4)
- Virtual keypoints: methods adding a regular grid of keypoints over the whole plane must infer perspective from texture-less grass and may fail on out-of-domain cameras; central views (only centre circle + halfway line visible) are hard: JaC5 10.2 (TVCalib) and 22.4 (PnLCalib) on 8,565 SoccerNet central views. — [Can Geometry Save Central Views?](https://arxiv.org/html/2504.20052)
- Label propagation through video: Waterloo's temporal method computes the first-frame homography and then uses the refinement model to estimate the frame-to-frame homography, giving smoother output and faster inference; this is exactly a label-propagation primitive. — [Shang et al.](https://openjournals.uwaterloo.ca/index.php/vsl/article/download/5374/5662/24730)
- Jiang's homography approach used a pseudo-labelled sequence dataset: it describes a subset of 1.67M images drawn from World Cup video for training with consecutive-frame splits to avoid game leakage. — [Jiang et al.](https://arxiv.org/pdf/1909.08034)

### Inferences
- With only 3 matches/50 verified homographies, the highest-value cheap augmentations are: (a) crop/zoom/translate and small perspective warps applied through a composed homography; (b) per-epoch re-sampled template perturbation (Waterloo/Jiang trick) if we use a refinement/render-and-compare head; (c) flip with left/right swap and pan sign inversion; (d) colour/shadow/blur/noise; (e) copy-paste occluders (small benefit).
- Pitfall (my reasoning, not directly measured in the sources): warping a labelled frame to a new virtual camera with the plane homography keeps the pitch lines correct, but the stands, advertising boards, goalposts and players are non-planar so they take the wrong parallax; large warps (strong perspective change, e.g. > ~15-20 deg of tilt) will create unrealistic images, and the image borders become empty (needs reflection/black fill, which models can exploit). Radial distortion also moves under a homography warp, so if the label is a camera model (not a plain 3x3 H), the warped label must be re-expressed (zoom/crop fits the camera intrinsics K' = A*K; translation shifts the principal point; pure rotation about the camera centre is also a homography of the image and can be composed into R).
- Rotation-about-centre warps are exactly consistent for the whole scene (including non-planar content), unlike a general plane-induced warp; so restrict virtual camera moves to pan/tilt/roll/zoom about the camera centre, which fits the broadcast PTZ setting and avoids parallax artifacts.

### Gaps
- No study quantifying gains of homography re-warp augmentation specifically for football with radial distortion.
- No source found that characterises when warp augmentation breaks (maximum useful warp magnitude).
- Left/right swap details for flips in the SoccerNet 26-class setting (which classes need to be swapped and how circle/goal-post classes behave) not documented in a source; infer from class names.

---

## Q4. Self-training / pseudo-labelling with a human-in-the-loop verifier for calibration

### Takeaway
There is little academic precedent for the exact loop we run; the closest published artefact is a patent family on video-based sports-field registration (learned model proposes homographies on unlabelled video, a consistency/uncertainty check accepts or rejects, accepted labels go to the labelled set). Large-scale auto-labelling (SoccerMaster) uses the SN-GSR calibration pipeline to label 2.7M frames without human verification, and TVCalib/PnLCalib-style geometric refinement is widely used to turn detections into clean labels.

### Cited Findings
- A patent family "Systems and methods for video-based sports field registration": model predicts homographies on annotated frames, generates pseudo-labels on unlabelled broadcast video, verifies them; verified labels join the labelled database, rejected ones return to the unlabelled pool; one version uses the distance between a directly regressed homography and a keypoint-derived homography as the uncertainty signal (above a first threshold = bad, below a second = good, in between = back to the unlabelled pool); frame selection uses metadata so that teams, fields, weather and seasons vary. Patent claims, not peer-reviewed; assignee and legal status not checked. — [USPTO 11816849 etc. via search result](https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/11816849)
- SoccerMaster / SoccerFactory (2025-12): keypoint and line detectors on main-camera clips from 500 SoccerNet-v2 matches (replays/close-ups excluded), PnL estimates cameras, detections refined by projecting from the canonical pitch; 2.7M auto-generated frames vs 42.8K manual SN-GSR; a quality check via GS-HOTA on the SN-GSR test set (64.1 vs challenge winner 61.5) rather than keypoint accuracy; no human verification described; data "will be publicly available". — [SoccerMaster](https://arxiv.org/html/2512.11016)
- Golovkin et al. use a modified TVCalib to fit cameras to keypoint annotations as the real-data labelling step (22,000 images). — [Golovkin](https://arxiv.org/html/2504.06357v1)
- Self-supervised pretraining (Shi et al., WACV 2022): trains registration with natural images and their edge maps instead of paired labelled data, with a score-regression network for refinement; a search summary states that pretraining for sports field registration "was not studied" prior. — [Shi et al.](https://openaccess.thecvf.com/content/WACV2022/html/Shi_Self-Supervised_Shape_Alignment_for_Sports_Field_Registration_WACV_2022_paper.html) (abstract page 403'd on fetch; taken from search snippet)
- Teacher-student pseudo-labelling exists for football player/ball detection (semi-supervised, same-architecture teacher-student). — [arXiv 2204.06859 via search](https://fugumt.com/fugumt/paper_check/2204.06859v1_enmode)
- Reliability warnings: central views and wide-angle/replay frames are the failure region; generated labels there should be down-weighted or sent to the human (JaC5 10-22% on central views). — [Can Geometry Save Central Views?](https://arxiv.org/html/2504.20052)

### Inferences
- Useful automatic verifier signals consistent with the sources: agreement between two independent estimators (direct regression vs keypoint/line PnP fit; two model checkpoints), reprojection error of detected line pixels, temporal consistency across neighbouring frames (flow/frame-to-frame homography), and a JaC-style line-coverage score at 5 px. Human review should be prioritised for the uncertain middle band and for frame types where the model is known weak (central views, replays, behind-goal cameras).
- Avoid confirmation bias: verified-and-accepted labels are from the model's confident region, so the labelled set gets biased toward easy frames; sample verification frames to cover stadiums, zoom levels and weather (as the patent's metadata-based frame selection suggests), and keep a fixed hand-verified held-out set from different matches/stadiums.

### Gaps
- No peer-reviewed paper found that reports a closed human-verifier loop for calibration with measured gain per round.
- Patent claims are unverified descriptions of method, not results.
- SoccerMaster's appendix on SoccerFactory was truncated in my retrieval.

---

## Practical summary for our situation (for the report writer)
- Open data to use as pretraining/validation: SN-Calibration-2023 (3.98 GB, HF, no gate), TS-WorldCup (588 MB), WC14 (via author page; use as test set because of annotation errors). Treat all as research-only (no explicit licence; broadcast footage).
- Synthetic generator is cheap given our pitch renderer + camera model: sample poses from a Chen&Little/Golovkin-style prior, randomise grass colour/stripes/line colour/width, fake distractor lines, shadows, occluder cutouts, colour jitter, radial distortion; mix with real data in the 1st stage, fine-tune on real only (Golovkin schedule: 200k batches mixed then 200k real-only).
- Expect modest absolute gains where real data is plentiful; no data exists for the 50-image regime, so measure it ourselves with a held-out different-stadium split.

## Source list (key)
- https://github.com/SoccerNet/sn-calibration
- https://huggingface.co/datasets/SoccerNet/SN-Calibration-2023 ; https://huggingface.co/datasets/SoccerNet/SN-GSR-2025
- https://www.soccer-net.org/data
- https://ar5iv.labs.arxiv.org/html/1810.10658 (Chen & Little)
- https://arxiv.org/html/2504.06357v1 (Golovkin)
- https://arxiv.org/html/2503.13969 (SoccerSynth-Field)
- https://arxiv.org/html/2404.08401v4 (PnLCalib/NBJW)
- https://arxiv.org/html/2404.09807v1 (ProCC)
- https://ericsujw.github.io/KpSFR (TS-WorldCup)
- https://openjournals.uwaterloo.ca/index.php/vsl/article/download/5374/5662/24730 (hockey augmentation)
- https://arxiv.org/pdf/1909.08034 (Jiang)
- https://arxiv.org/html/2512.11016 (SoccerMaster)
- https://arxiv.org/html/2508.19182 (SoccerNet 2025 results)
- https://github.com/Spiideo/sskit
