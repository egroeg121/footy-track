# Point trackers as bidirectional propagation engines, and self-/semi-supervised recipes for bootstrapping a temporal ball model

Project context (KB, read at start): footy ball is ~11 px wide at 1280 (25% under 10 px); 640 px never detects it, 1280 does; more single-frame labels did not help; the team already plans heatmap-over-frame-stack models with centre-frame-only labels; SOT/Siamese trackers are unvalidated at ball scale and fail silently when motion exceeds ~2x target size; broadcast cuts break t-1 context; two conflicting ball-mAP figures remain unresolved — [wiki/footy-ball-detection-findings.md](file:///home/george/knowledge/wiki/footy-ball-detection-findings.md) (KB search "footy ball detection" also surfaced `footy-track-overview`, `footy-labeller-gotchas`, `vast-ai-gpu-rental`; only the findings page was read in full).

Conventions: "unverified" = not confirmed from a primary page in this session. All 2026 items checked against pages fetched on 2026-11-14.

---

## A1. Which point trackers are offline/bidirectional, and what do they cost (windows, resolution, VRAM, speed, licence, fine-tune code)?

### Takeaway
Only a few trackers are genuinely bidirectional over a whole window: CoTracker3 (offline mode), BootsTAPIR/TAPIR (offline, whole-video), and AllTracker (16-frame sliding windows with overlap, direction not stated). All are trained and benchmarked at 256–512 px on synthetic or pseudo-labelled real data with texture-like query points; none publishes a 1080p long-clip VRAM figure. CoTracker3 is the only one with an official real-video pseudo-label fine-tuning script, but its repo is CC-BY-NC. Apache-2.0 (TAP/DeepMind, LocoTrack) and MIT (AllTracker) options exist for commercial use.

### Cited Findings

**CoTracker3 / CoTracker2 (Meta)**
- Offline mode: "a single-window model" (whole video is one window, tracking in both directions); evaluation uses window length 60; trained on 60-frame clips. Online mode: sliding window of 16 frames, trained on 64-frame sequences, "can run indefinitely". — [co-tracker README](https://github.com/facebookresearch/co-tracker); [CoTracker3 paper](https://arxiv.org/html/2410.11831)
- Offline is better at tracking occluded points because it sees all frames; offline better on DAVIS/RGB-Stacking, online better on Kinetics/RoboTAP. Offline is "limited by memory in the number of frames". — [CoTracker3 paper](https://arxiv.org/html/2410.11831)
- 25M params (CoTracker 45M, TAPIR 31M, LocoTrack 12M, BootsTAPIR 78M). Time per frame per point: offline 209 µs, online 405 µs, LocoTrack 290 µs. — [CoTracker3 paper](https://arxiv.org/html/2410.11831)
- Inputs resized to 256x256 for TAP-Vid; Dynamic Replica/RoboTAP evaluated at native resolution. — [CoTracker3 paper](https://arxiv.org/html/2410.11831)
- Video-level BootsTAPIR and CoTracker3 ran out of memory on PointOdyssey on a 48 GB A40; AllTracker's authors say CoTracker3 needed a 96 GB H100 in their high-res tests. — [Track-On2](https://arxiv.org/html/2509.19115v1); [AllTracker](https://arxiv.org/html/2506.07310v2)
- Memory rises with grid size (number of tracked points); official remedy for long videos is online mode with `video_chunk`. — [Voxel51/CoTracker docs via search](https://voxel51.com/blog/cotracker3-a-point-tracker-using-real-videos) (secondary source)
- LICENCE: "Most of the project is under CC-BY-NC" (Particle Video Revisited MIT; TAP-Vid and LocoTrack parts Apache-2.0). The README does not give a separate licence for the weights (unverified; assume NC). — [co-tracker README](https://github.com/facebookresearch/co-tracker). **Non-commercial flag.**
- Fine-tune code: `train_on_kubric.py` (paper: 50k iters on 32 GPUs) and `train_on_real_data.py` (starts from a Kubric checkpoint; example 15,000 steps, lr 5e-5; pseudo labels generated inside the script; `real_dataset.py` class for your own videos). Kubric data on Hugging Face (6,000 sequences, 512x512, 120 frames). — [co-tracker README](https://github.com/facebookresearch/co-tracker)
- CoTracker2 jointly tracks up to 265x265 points; CoTracker2.1 released Sep 2024, CoTracker3 Oct 2024. — [co-tracker README](https://github.com/facebookresearch/co-tracker)

**TAPIR / BootsTAPIR / TAPNext / TAPNext++ (Google DeepMind, `tapnet`)**
- Whole-video ("Standard") TAPIR and BootsTAPIR are offline; Online TAPIR/BootsTAPIR and TAPNext/TAPNext++ are causal per-frame. BootsTAPIR works best at 512x512; TAPIR may benefit above 512. — [tapnet repo](https://github.com/google-deepmind/tapnet)
- Repo licence Apache-2.0 (software and checkpoints); TAP-Vid annotations CC-BY 4.0. Jax training code for TAP-Net/TAPIR in `training/`; no training code listed for BootsTAP or TAPNext. PyTorch checkpoints for BootsTAPIR and TAPNext++. — [tapnet repo](https://github.com/google-deepmind/tapnet)
- Speed reference: causal TAPIR ~17 fps at 480x480 on a Quadro RTX 4000; online BootsTAPIR ~30 fps for 400 points at 256x256; ~5.6 s for 10,000 points in a 256x256, 50-frame video on A100. — [tapnet repo](https://github.com/google-deepmind/tapnet); [BootsTAP](https://arxiv.org/html/2402.00847)
- TAPNext++ (arXiv 2604.10582, Apr 2026): online, frame-by-frame, ViT blocks alternating with linear-recurrent (RG-LRU) blocks, no explicit memory. Trained only on synthetic data (PointOdyssey, Kubric-1024 = 10k x 1024-frame videos, DynHumans); no pseudo-labelling; real videos eval only. Effective range extended from ~150 frames (TAPNext) to >4,000 frames. 256x256 base plus a 512x512 variant (fine-tuned 120k steps on 256-frame sequences). 5.18 ms/frame at 256 queries on an H100 (193 FPS); online CoTracker3 ~80 ms (102 FPS). — [TAPNext++ paper](https://arxiv.org/html/2604.10582v1)
- TAPNext++ vs online CoTracker3: DAVIS AJ/δavg/OA 66.6/79.9/92.1 vs 63.8/76.3/90.2; PointOdyssey survival 67.9 vs 56.3; RoboTAP AJ_RD 52.4 (54.6 at 512) vs 51.8. Roll augmentation (wrap-around) trains re-detection; occluded-point position loss weighted 0.2x. — [TAPNext++ paper](https://arxiv.org/html/2604.10582v1); [project page](https://tap-next-plus-plus.github.io)
- TAPNext++ page notes it was the only method to re-detect a ball after a cup was lifted in a cups-and-balls demo (cannot track it while occluded) — a qualitative single example, not a sports result. — [project page](https://tap-next-plus-plus.github.io)
- TAPNext++ code is in the `tapnet` repo (Apache-2.0 repo licence); the project page itself is CC BY-SA 4.0 and states no model-weight licence — treat weight licence as unverified. — [project page](https://tap-next-plus-plus.github.io)

**AllTracker (Harley et al., ICCV 2025)**
- Estimates flow from a query frame to every other frame (output T x H x W x 4: 2 flow channels + visibility + confidence). Sliding window of 16 frames, stride 8, each window initialised from the previous; benchmark videos trimmed to 600 frames max. Bidirectionality not stated; occlusions longer than the window are handled poorly. 16M params. — [AllTracker paper](https://arxiv.org/html/2506.07310v2); [arXiv listing](https://arxiv.org/pdf/2506.07310)
- Resolutions tested 384x512, 448x768, 768x1024; at 768x1024 it produces 786,432 tracks at once on a 40 GB A100. Streaming mode 57.9 FPS at 512x512 (62.6 δavg vs 66.1 standard). TAP-Vid AJ 68.9 vs CoTracker3 63.1; δavg at 768x1024 69.5 vs 66.9. Authors say their protocol likely overestimates CoTracker3. — [AllTracker paper](https://arxiv.org/html/2506.07310v2)
- Training: Kubric 200k steps then 400k steps on point-tracking (Kubric, DynamicReplica, PointOdyssey) plus many optical-flow sets; no pseudo-labelling. Flow accuracy coarser than SEA-RAFT; "appears to be underfitting on short-range motion". — [AllTracker paper](https://arxiv.org/html/2506.07310v2)
- MIT licence. Training code included (`train_stage1.py`, `train_stage2.py`; stage 2 loads stage 1 via `--init_dir`, lr 1e-5, 400k steps); 768x1024 inference gives DA 80.6 / AJ 67.2 / OA 89.7 on their test. No VRAM/runtime numbers for 1080p clips in the README. — [alltracker repo](https://github.com/aharley/alltracker)
- Strong fit for a **dense, high-res, query-frame-to-all-frames** use (e.g. every pixel of a background crop), but at 1080p x hundreds of frames memory is unmeasured (see Gaps).

**LocoTrack (KAIST)**
- Apache-2.0; PyTorch inference/training code released 2024-08-03; trained on panning-MOVi-E (~273 GB); claims "near-dense point tracking in real-time" and 6x faster than prior SOTA. README doesn't specify resolution, memory, or long-video handling. 12M params, 290 µs/frame/point. — [locotrack repo](https://github.com/cvlab-kaist/locotrack); [CoTracker3 paper](https://arxiv.org/html/2410.11831)
- CoTracker3 paper's qualitative comparison: LocoTrack and BootsTAPIR "cannot track through occlusions well". — [CoTracker3 paper](https://arxiv.org/html/2410.11831)

**TAPTR / TAPTRv2 / TAPTRv3 (IDEA Research)**
- TAPTR treats tracking as DETR-style detection with point queries; v2 removes the cost volume; v3 (arXiv 2411.18671, ICLR 2026 poster) adds visibility-aware long-temporal attention, context-aware cross attention and auto-triggered global matching to handle drift and scene cuts. Reported ~57 FPS vs 42 (v2) and 26 (CoTracker). Reported SOTA on most TAP datasets. — [TAPTRv3 listing](https://arxiv.org/pdf/2411.18671); [IDEA-Research/TAPTR](https://github.com/IDEA-Research/TAPTR)
- v3 code released 2026-02-10; README does not state licence text, checkpoint release, training code, or inference resolution (unverified). — [IDEA-Research/TAPTR](https://github.com/IDEA-Research/TAPTR)
- Scene-cut recovery (global matching reset) is directly relevant to broadcast cuts, but only claimed, not evaluated on sports.

**Track-On / Track-On2 (KUIS-AI)**
- Online (causal), DINOv2/DINOv3 encoder, per-query memory; synthetic-only training (Kubric); pseudo-labelling listed as future work. 1024 points ~1.5 GB, >20 FPS on A100. DAVIS AJ 67.0, RoboTAP 68.1. No code repository or code licence stated on the paper page (unverified). — [Track-On2](https://arxiv.org/html/2509.19115v1); [thesis](https://arxiv.org/html/2507.09217)

**Newer items in the search window**
- TAPNext++ (Apr 2026) was the only clearly 2026 point-tracking paper surfaced; others found (TAPVid-360, MV-TAP, TAPIP3D) are 2025 and target different tasks. — [search results](https://www.alphaxiv.org/abs/2604.10582); [TAPVid-360](https://arxiv.org/pdf/2511.21946); [MV-TAP](https://arxiv.org/pdf/2512.02006)

### Inferences
- For the **ball**, the only practical "bidirectional propagation engine" among these is CoTracker3 offline (or AllTracker), run over windows of tens of frames anchored at human-labelled/high-confidence frames, with queries seeded from the ball location. Both would need crops (e.g. a 256–512 px crop tracked around the ball, or around a player region) rather than full 1080p, to keep VRAM within 8 GB.
- CoTracker3's NC licence matters only if the product is commercial; AllTracker (MIT) or TAPIR/BootsTAPIR/LocoTrack (Apache-2.0) are licence-clean alternatives, with CoTracker3 usable as an offline research teacher.
- Online trackers (TAPNext++, Track-On2) are causal by design, so they are poor fits for a non-causal use unless run forward and backward and fused.

### Gaps
- No source gives CoTracker3 offline VRAM vs frames at 1080p or even 720p; the Track-On2 figure plotting it was not extractable. Needs a local measurement.
- AllTracker bidirectionality and 1080p memory not stated.
- CoTracker3 weight licence and TAPNext++ / TAPTRv3 weight licences not confirmed.
- TAPTRv3 checkpoint availability not confirmed.

---

## A2. Do point trackers work on a small, fast, appearance-changing object like a ball?

### Takeaway
No direct evidence found for any point tracker on sports balls or similarly tiny fast objects; the benchmarks (TAP-Vid, RoboTAP, PointOdyssey) are about surface points on larger, slower objects at 256x256, and pseudo-label query points are chosen as SIFT "good to track" texture. Generic small-fast-object literature says texture-correlation trackers fail on blurred, texture-less balls/pucks.

### Cited Findings
- CoTracker3 pseudo-label query points: 8 random frames per video, SIFT on each, 384 points/video biased to "good to track" points; videos with fire, lights, water, CGI/animation filtered out. Teacher set: CoTracker3 online, CoTracker3 offline, CoTracker, TAPIR (all Kubric-only). — [CoTracker3 paper](https://arxiv.org/html/2410.11831)
- BootsTAP's training targets are "solid surfaces move and deform"; its stated goal is tracking surface points. — [BootsTAP abstract](https://arxiv.org/abs/2402.00847)
- At 256x256 downsampling, an 11 px ball at 1280 width becomes ~2 px (arithmetic from KB figure: 1280→256 is 5x) — i.e. below the 8-px patch/stride granularity of these models (inference). Context: [KB findings](file:///home/george/knowledge/wiki/footy-ball-detection-findings.md)
- Hockey puck study: Siamese-FC, DiMP, TaMOs fail on broadcast puck due to size, speed, occlusion and compression. TSFMO benchmark (20 trackers, small fast sports objects): authors conclude much more effort is required. Fast objects can move more than their diameter per frame so inter-frame overlap is negligible; lack of sharp texture makes texture-correlation trackers unusable. — [Waterloo hockey puck study](https://openjournals.uwaterloo.ca/index.php/vsl/article/view/7117); [TSFMO](https://arxiv.org/abs/2209.04284v1) (via search snippets)
- Kalman-filter trackers on fast tiny objects (squash ball) show 3–4x higher error than on standard benchmarks. — [arXiv 2509.18451](https://arxiv.org/html/2509.18451v1) (via snippet)
- TAPNext's evaluation (CoTracker-style) benefits from a local support grid of extra query points when tracking one point; relevant for isolated small targets. — [TAPNext supplement](https://openaccess.thecvf.com/content/ICCV2025/supplemental/Zholus_TAPNext_Tracking_Any_ICCV_2025_supplemental.pdf) (via snippet)
- TAPNext++ page's cups-and-balls qualitative example is the only ball-like example found, and it is for a large ball in a lab scene. — [TAPNext++ page](https://tap-next-plus-plus.github.io)
- Sports ball detection by heatmap: WASB uses a small HRNet on three stacked frames (288x512) and reaches soccer F1 88.3 (τ=4 px) on a 1920x1080 25 fps dataset (4 train clips, 11,994 frames, avg inter-frame displacement 10.4 ± 10.0 px); the paper says soccer false positives come from "player shoes". — [WASB paper](https://arxiv.org/html/2311.05237)

### Inferences
- A point tracker queried at the ball will likely track the *neighbourhood* (players' feet, grass texture) or drift when the ball is thrown/kicked across pitch; no evidence says otherwise. It may work in the short, slow, ball-at-feet segments, and as a **local interpolator between two confident anchors** (where start and end positions are known and only the path needs filling), which a held-out-anchor test could cheaply validate.
- A better use of trackers for the ball is as an *ego-motion* estimator (A3) or to track the **player carrying the ball**, with ball position modelled relative to the player/camera.
- Expect high failure rate during fast passes and shots; the project's own labelled runs (~900 dense frames) are the only usable test bed.

### Gaps
- No paper evaluating CoTracker/TAPIR/TAPNext on balls, pucks, shuttlecocks or tiny objects found; per-object-size breakdowns on TAP-Vid not found. Must be tested locally.
- No published AJ/δ of any tracker at <3 px target sizes.

---

## A3. Camera-motion compensation from pitch/background points

### Takeaway
Classical practice is a RANSAC homography from point correspondences, with PTZ decomposition; learned point trackers have not been shown (in what I found) on sports pan/tilt/zoom, but grid-tracking of background points with an offline tracker plus RANSAC is a plausible, cheap, label-free route; known risk is blur on fast pans.

### Cited Findings
- Homography maps points between views/pitch plane; estimated with RANSAC over noisy correspondences; only 4 correspondences needed; early PTZ work decomposes pan/tilt/zoom from the homography. — [Top-view registration of broadcast football](https://arxiv.org/pdf/1703.01437); [Hayet 2005](https://iis.uibk.ac.at/public/papers/Hayet-2005-AVSS.pdf)
- Without camera-motion compensation, a stationary player appears to move when the camera pans, breaking motion-based tracking. Sports cameras move quickly, causing blur that hurts feature matching. — [Okuma/Ghanem sources](https://migrate2wp.web.illinois.edu/html-files-to-import/publications/ghanem_icassp12.pdf) (via search snippet)
- Online PTZ SLAM with an EKF over pan/tilt/focal length assumes rotation-only motion (a good model for broadcast cameras). — [UBC 2019, arXiv 1907.08816](https://arxiv.org/pdf/1907.08816v1.pdf)
- KB: BoT-SORT's camera-motion compensation was already flagged as relevant because broadcast pans break image-space velocity models. — [KB findings](file:///home/george/knowledge/wiki/footy-ball-detection-findings.md)
- CoTracker (v2/v3) can track "a quasi-dense set" of points; CoTracker2 up to 265x265 points jointly; offline tracking is memory-limited in frames. — [co-tracker README](https://github.com/facebookresearch/co-tracker)
- AllTracker produces dense flow at 768x1024 with MIT licence, so dense background flow per window is feasible. — [AllTracker](https://arxiv.org/html/2506.07310v2)

### Inferences
- Pipeline: track a grid of points (e.g. 20x12) over 16–60 frame windows with CoTracker3 offline or AllTracker; mask out points that land on player boxes; fit a homography/rotation+zoom per frame by RANSAC on visible points; chain through windows. For a PTZ camera the homography between frames is exact for pure rotation (any scene), so grass texture beyond the pitch plane is fine.
- Fallbacks: ECC/ORB homography (BoT-SORT GMC) — much cheaper; benchmark learned tracker against it before adopting.
- Pitch markings (lines) are low-texture and repetitive, which are the failure modes SIFT-seeded training avoids; crowd/advertising boards offer more texture.

### Gaps
- No paper found applying CoTracker/TAPIR/AllTracker to sports broadcast camera estimation (search returned only classical methods).
- Quantitative accuracy of learned-tracker homography under fast pan blur: not found.

---

## B1. Teacher–student / pseudo-label self-training for video tracking and detection

### Takeaway
Two mature recipes exist. BootsTAP: EMA teacher on clean video, student sees heavily augmented view (affine crop/zoom, JPEG), supervised by teacher positions/occlusion with cycle-consistency filtering; works with 1% of data nearly as well (ablation). CoTracker3: frozen Kubric-trained teachers pseudo-label 15k real videos, sampling one teacher per batch, beating BootsTAPIR trained on 15M. Detection-side (Unbiased Teacher, class-specific thresholds) addresses imbalance, which is the ball's situation.

### Cited Findings
**BootsTAP (DeepMind)**
- Teacher = EMA of student, no gradients, both initialised from Kubric-pretrained TAPIR. Student input: smoothly varying frame-wise affine crop/zoom (60–100% of frame) and JPEG; query point transformed identically. Losses: Huber on position weighted by teacher visibility; BCE on teacher occlusion; uncertainty target = student–teacher distance > 6 px. — [BootsTAP](https://arxiv.org/html/2402.00847)
- Cycle-consistency filter: trajectory kept only if student's prediction back at the query frame is visible and within 4 px. — [BootsTAP](https://arxiv.org/html/2402.00847)
- Data: >15M 24-frame clips; shot changes, intros/outros, watermarks removed; 200k iterations on 256 A100s. Gains: DAVIS strided AJ 61.3→67.4, Kinetics 57.2→62.5, RGB-Stacking 62.7→77.4. — [BootsTAP](https://arxiv.org/html/2402.00847)
- Ablations (DAVIS AJ/Kinetics AJ; base 65.8/54.4): no affine transforms 54.4/44.7 (largest drop); no JPEG 65.7/53.5; no occlusion filtering 65.9/54.1; **only 1% of real data 66.2/54.0**; Siamese without EMA/stop-gradient collapses to 49.8/29.6; Kubric-only longer 65.0/52.7. — [BootsTAP](https://arxiv.org/html/2402.00847)

**CoTracker3**
- ~100k internet videos (~30 s, mostly humans/animals); main result uses 15k; 4 frozen teachers, one random teacher per batch (better than mean/median); SIFT queries; track loss only — visibility/confidence head frozen (freezing improved AJ by 0.8, OA by 3.9). 15k iterations, lr 5e-5. Kub+15k online/offline mean AJ 63.8/64.4 vs BootsTAPIR (15M videos) 61.4. Scaling saturates ~30k videos; second self-training round gave negligible change (δavg 76.8→76.9). Teacher quality bounds student. — [CoTracker3](https://arxiv.org/html/2410.11831)
- Offline (bidirectional) tracks used as teachers include CoTracker3 offline — i.e. the recipe already uses a bidirectional smoother as teacher for an online student. — [CoTracker3](https://arxiv.org/html/2410.11831)

**Detection self-training**
- Unbiased Teacher (ICLR 2021): class imbalance makes pseudo-labels biased to majority classes and self-training worsens it; remedy = EMA teacher + focal-style loss that down-weights overly confident pseudo-labels. — [Unbiased Teacher](https://arxiv.org/pdf/2102.09480)
- LabelMatch: global hand-crafted confidence thresholds are the core weakness; proposes class-specific adaptive thresholds based on label distribution; splits pseudo-labels into reliable vs uncertain (soft). — [LabelMatch](https://arxiv.org/pdf/2206.06608)
- Soccer-specific teacher–student: Vandeghen et al. (CVPRW 2022) label unlabeled broadcast video with a teacher, train a same-architecture student with losses adjusted to discount low-confidence teacher predictions; benchmark 52.3% mAP on SoccerNet-v3 detection; code `rvandeghen/SST`; Faster R-CNN; per-frame only. — [paper](https://arxiv.org/abs/2204.06859); [search digest](https://arxiv.org/pdf/2204.06859)

### Inferences
- The project's 356k machine candidates with known precision-by-confidence are already a pseudo-label set. A principled use: confidence-weighted loss with calibrated weights (conf ≥0.7 ≈ 0.96 precision → near-hard; 0.6–0.7 band → soft; ≤0.42 → ignore/low weight rather than negative), plus label smoothing in heatmaps. This mirrors BootsTAP's visibility-weighted loss and LabelMatch's reliable/uncertain split (inference; not tested).
- Offline bidirectional smoothing (forward/backward trackers + Viterbi/Kalman smoother over candidates) can upgrade sub-threshold candidates before they become targets — equivalent to CoTracker3's offline-teacher/online-student pattern.
- BootsTAP's ablation implies strong augmentation (crop/zoom jitter + compression) on the student matters more than volume of unlabeled video; 1% of data gave almost the full gain. The ~33 GB of other-match video is plenty.
- Never treat unlabelled frames as pure negatives: the ball is often visible but unlabelled; ignoring low-confidence regions is safer. Under 1:~50 class skew, Unbiased-Teacher-style down-weighting of confident easy negatives (focal) is consistent with WASB's quality-focal-loss.

### Gaps
- No ball-specific measurement of how teacher confidence correlates with student gains.
- Soft Teacher details not retrieved (only passing mention); mean-teacher detection numbers beyond those above not collected.
- Vandeghen paper's ball-class AP and exact thresholding not extracted (PDF unreadable via fetch).

---

## B2. Cycle-consistency / forward–backward consistency self-supervision

### Takeaway
Cycle consistency is standard and cheap: BootsTAP uses it as a trajectory filter (4 px return error); the contrastive random walk trains correspondences end-to-end from raw video via palindromic walks. For a ball detector, the analogous signal is forward-backward agreement of track/propagation, usable both as a filter on pseudo-labels and as an uncertainty estimate.

### Cited Findings
- Contrastive random walk (Jabri et al., NeurIPS 2020): space-time graph of patches, transition probabilities from learned similarity, trained by walking forward then back (palindrome) and rewarding return to start; outperforms prior self-supervised methods on label propagation of objects, parts and pose; code `ajabri/videowalk` (licence not stated on abstract page). — [arXiv 2006.14613](https://arxiv.org/abs/2006.14613)
- BootsTAP uses cycle-consistency (4 px) as a trajectory filter; effect small (66.1/54.3 vs 65.8/54.4 base). — [BootsTAP](https://arxiv.org/html/2402.00847)
- AllTracker trains confidence to predict whether the estimate is within 12 px of ground truth. — [AllTracker](https://arxiv.org/html/2506.07310v2)
- BootsTAP trains an uncertainty output = whether student/teacher positions differ by >6 px. — [BootsTAP](https://arxiv.org/html/2402.00847)

### Inferences
- For ball interpolation: run propagation forward from anchor A to B and backward from B to A; accept intermediate points only if fwd/bwd disagree by less than ~2–3 px (ball-scale analogue of the 4 px rule); disagreement doubles as a confidence for the trajectory-level label weight.
- CRW works on whole-patch appearance at low res and a ball is ~11 px; I would not expect CRW to be useful for the ball directly; fine for player/background association (unverified).

### Gaps
- No evidence on cycle-consistency losses for tiny objects specifically.
- Temporal-consistency losses for heatmap detectors (smoothness, flicker) not located in sources.

---

## B3. Sparse-label training of multi-frame heatmap detectors (centre-frame-only labels, propagation, interpolation-labelled frames)

### Takeaway
The standard heatmap-over-stack design (WASB: 3 frames in, one heatmap per frame out) uses per-frame labels; I found no source on labelling only the centre frame, but the training mechanics (soft Gaussian-like targets, quality focal loss, hard-example mining) transfer directly. Interpolated frames can be given softer/wider targets with confidence weights.

### Cited Findings
- WASB: 3 consecutive frames concatenated on channels; heatmap for each frame at input resolution (288x512); target = real-valued Gaussian-like blob radius d=2.5, min value 0.7; quality focal loss for foreground-background imbalance; heatmap binarised at 0.5; blob centre-of-heatmap position and summed score as confidence; local motion model (position/velocity/acceleration) picks candidates; oversampling stride 1 improved AP 83.6→86.2. — [WASB](https://arxiv.org/html/2311.05237)
- WASB "Hard-to-Localize Sample Mining": from epoch 20 of 30, run inference on training sequences and use real-valued labels for frames whose predictions are far from ground truth. — [WASB](https://arxiv.org/html/2311.05237)
- WASB soccer data is a fully per-frame, manually re-annotated set because the original annotations were unreliable (12k train frames, 6k test). — [WASB](https://arxiv.org/html/2311.05237)
- WASB repo: MIT; repo currently holds evaluation code for pretrained WASB models only (training code not released per README as read). — [WASB-SBDT](https://github.com/nttcom/WASB-SBDT)
- KB: centre-frame labels + recoverable neighbours give 1,453 ball frames with no new annotation; 942 already in runs of 3+. — [KB findings](file:///home/george/knowledge/wiki/footy-ball-detection-findings.md)
- CoTracker3 and BootsTAP supervise only a sample of points in clips of 8–24 frames, i.e. sparse tracking supervision on real video works, but with teachers, not humans. — [CoTracker3](https://arxiv.org/html/2410.11831); [BootsTAP](https://arxiv.org/html/2402.00847)
- TAPNext++ down-weights occluded-point position loss to 0.2x. — [TAPNext++](https://tap-next-plus-plus.github.io)

### Inferences
- Concrete recipe (not source-validated): stack of N frames → N heatmaps; loss on every frame that has a label of any grade: (i) human labels weight 1.0, narrow blob; (ii) machine candidates conf ≥0.7 weight ~0.9; (iii) interpolated/propagated frames weight by fwd-bwd agreement and with a wider blob (e.g. σ scaled by gap length); (iv) unlabelled and low-confidence regions masked rather than negative. Cuts must be detected so neighbours across cuts are not stacked (KB trap).
- Use WASB-style hard-sample mining on the unlabelled-video pseudo-label rounds.

### Gaps
- No paper describing centre-frame-only labels for heatmap-stack detectors, or weighted interpolation labels, was found.
- WASB training hyperparameters (batch, lr) not extracted.

---

## B4. Sports-specific examples of bootstrapping ball detectors from few labels

### Takeaway
Only one directly relevant published example surfaced: Vandeghen et al. (soccer teacher-student on broadcast video, per-frame Faster R-CNN, public code). Nothing found that bootstraps a *temporal* ball detector from sparse labels plus unlabelled video, or that uses point trackers for it.

### Cited Findings
- Vandeghen, Cioppa, Van Droogenbroeck (CVPRW 2022): small labelled soccer set + unlabelled broadcast video; teacher → surrogate labels → same-architecture student; confidence-aware losses; improvement over labelled-only; 52.3% mAP on SoccerNet-v3; code github.com/rvandeghen/SST. — [arXiv 2204.06859](https://arxiv.org/abs/2204.06859)
- WASB covers soccer, tennis, badminton, volleyball, basketball with one baseline trained per-frame-annotated sets. — [WASB-SBDT](https://github.com/nttcom/WASB-SBDT)
- Single-camera 3D football trajectory reconstruction (2025) cites spline/ballistic-constraint interpolation used in table-tennis and discrete ball-state approaches in football (Strachan et al.). — [arXiv 2506.07981](https://arxiv.org/pdf/2506.07981) (via snippet)
- Sparse broadcast positions interpolated for player tracking using linear and velocity-aware interpolation. — [R. Soc. Open Sci. 2025](https://royalsocietypublishing.org/rsos/article/12/10/251175/236076/Continuous-football-player-tracking-from-discrete) (via snippet)
- Classic broadcast-soccer work fills ball gaps by trajectory extension, interpolation and Kalman filtering when occluded or off-pitch. — [Trajectory-based ball tracking in broadcast soccer](https://www.researchgate.net/publication/221572343_Trajectory-based_ball_detection_and_tracking_with_applications_to_semantic_analysis_of_broadcast_soccer_video) (old, via snippet)

### Inferences
- Ballistic/physical interpolation matters for ball in flight (gravity in world coordinates) but needs camera compensation (A3) to apply; in image space under pan/zoom, straight/parabolic priors are wrong.

### Gaps
- Nothing found on TrackNet/WASB-style self-training with unlabelled video; searches returned only the Vandeghen paper.
- No SoccerNet ball-track bootstrapping results found.

---

## B5. Leave-anchors-out / held-out evaluation of interpolation and propagation quality

### Takeaway
No standard protocol was found in the literature surfaced; the closest analogues are TAP-Vid's query-first/strided protocols (queries sampled from a frame, errors measured at all other frames) and BootsTAP's cycle-consistency threshold. A leave-anchors-out protocol would be project-specific design; the KB's benchmark-trap lesson (denominators, leakage) applies.

### Cited Findings
- TAP-Vid "strided" vs "query-first" protocols; metrics AJ (Average Jaccard), δavg (fraction of points within thresholds), OA (occlusion accuracy). Reported e.g. BootsTAPIR DAVIS strided AJ 67.4, query-first 62.4. — [BootsTAP](https://arxiv.org/html/2402.00847)
- AllTracker reports AJ only on TAP-Vid because other datasets' visibility labels are unreliable; AJ/OA not reliably comparable across datasets. — [AllTracker](https://arxiv.org/html/2506.07310v2); [alltracker repo](https://github.com/aharley/alltracker)
- TAPNext++ introduces AJ_RD, averaging AJ over reappearance events that set a new record for invisibility duration — a re-detection metric. — [TAPNext++](https://tap-next-plus-plus.github.io)
- WASB evaluates with τ=4 px distance threshold (F1/Acc/AP) on held-out clips from a different game. — [WASB](https://arxiv.org/html/2311.05237)
- KB: ball metrics have been inflated before by a tiny 44-instance val set; some clips (`astonvilla_seg080`, `bournemouth_1st_seg070`) are in training and `arsenal_example` is the only clean dense clip; report instance counts. — [KB findings](file:///home/george/knowledge/wiki/footy-ball-detection-findings.md)

### Inferences
- Proposed protocol (not from a source): in the ~900 densely-labelled frames, hide k interior frames of each run (gap lengths 2, 5, 10, 25 frames), propagate from the two flanking human anchors in both directions, and report error in px at tolerance thresholds (e.g. 2/4/8 px, plus normalised by ball diameter) per gap length, with a re-detection metric (AJ_RD analogue) for gaps containing occlusion. Compare: linear interpolation, constant-velocity Kalman/RTS smoother with camera compensation, CoTracker3/AllTracker propagation, and heatmap-model Viterbi. Split by match/clip to avoid leakage; report n per bucket.
- Also hide *anchors* (not just interior): drop each anchor and see if the remaining anchors plus propagation recover it, to measure how far a trajectory can be extrapolated at clip start when the ball is not visible.

### Gaps
- No literature found on gap-length-stratified interpolation benchmarks for ball tracking.
