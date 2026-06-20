# Ball Tracker Failure Analysis

**Date:** 2026-06-21
**Bead:** ft-55d.1
**Clips:** 2 broadcast clips, 500 total sampled frames
**Method:** YOLO detector (best checkpoint) + Hungarian LAP tracker, 250 sampled frames per clip at native 25fps

## TL;DR

**Detection failure is the dominant problem (100% of missed-ball frames).
The tracker is not the bottleneck.**

## Clips Analysed

| Clip | Duration | Frames Sampled | Ball Detected | Ball Tracked | Missed |
|---|---|---|---|---|---|
| arsenal_mancity_20250925_part192.mp4 | 30s | 250 | 90 (36%) | 90 (36%) | 160 (64%) |
| arsenal_mancity_example_video.mp4 | 10s | 250 | 18 (7%) | 18 (7%) | 232 (93%) |
| **Total** | | **500** | **108 (21.6%)** | **108 (21.6%)** | **392 (78.4%)** |

## Failure Characterisation

Of 392 missed-ball frames (frames where the tracker emitted no ball detection):

| Failure Mode | Count | % of Missed |
|---|---|---|
| Detection failure (YOLO conf = 0) | 392 | **100%** |
| Tracker failure (detected ≥ 0.3 but dropped) | 0 | **0%** |
| Low-conf detections dropped (0 < conf < 0.3) | 0 | **0%** |

Every missed-ball frame was a detection failure — YOLO returned no ball box at all.
The LapTracker dropped **zero** ball detections that YOLO found above threshold.

## Track ID Continuity

| Clip | Unique Ball Track IDs | Track ID Switches |
|---|---|---|
| part192 (30s, sampled at 3fps) | 82 unique IDs | 46 switches |
| example_video (10s, sampled at 1fps) | 6 unique IDs | 8 switches |

Track ID fragmentation is severe in part192: 82 unique track IDs across 90 tracked frames means the ball is almost never maintained across gaps. However, this is a **consequence** of detection failures, not an independent tracker failure — when detection resumes after a gap, the tracker correctly spawns a new track (IoU linking can't bridge multi-frame gaps where the ball was never seen).

## Conclusion

**Root cause: YOLO detection.** The ball is simply not detected in ~64–93% of frames depending on the clip. The LapTracker works correctly on every frame where YOLO returns a detection.

## Recommended Remediation Priority

1. **Improve detection (highest priority):**
   - Lower confidence threshold below 0.3 — the analysis used 0.3 but the ball may have sub-threshold confidence in many frames
   - Fine-tune the YOLO checkpoint on harder ball-visible frames (occlusions, motion blur, small ball)
   - Investigate whether the two clips differ so much (36% vs 7% detection rate) due to ball visibility, camera angle, or out-of-play frames

2. **Track interpolation / Kalman smoothing (secondary):**
   - Once detection rate improves, ID switching will reduce naturally
   - A Kalman predictor or ByteTrack-style second-pass on low-conf detections could bridge short gaps where the ball is briefly sub-threshold

3. **Tracker algorithm replacement (lower priority):**
   - The current LapTracker is not the failure point — replacing it with ByteTrack/SORT would not recover missed-ball frames since there are no detections to link

## Output Files

- `runs/ball_tracker_analysis.json` — full per-frame records + summaries
- `scripts/analyse_ball_tracker.py` — rerunnable analysis script

## Example Frames

Detection failure frames from part192: frames 12, 15, 18, 24, 27
Detection failure frames from example_video: frames 0–4 (ball not visible in the opening sequence)
