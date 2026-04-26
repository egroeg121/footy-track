# The Problem footy-track Solves

## Context

Football clubs, broadcasters, and analysts want structured, time-accurate data from match footage: where players were at each moment, when events happened (passes, shots, tackles, substitutions), and how those events connect across the match timeline.

That kind of data is enormously valuable — it drives tactical analysis, scouting reports, injury risk monitoring, fan statistics, and broadcast graphics. But generating it is hard.

## What exists today

Commercial providers (Opta, StatsBomb, Tracab, Second Spectrum) sell this data, but:
- Coverage is limited to top leagues and selected matches.
- The data is expensive and licensed with tight restrictions.
- You get the outputs, not the pipeline — no way to adapt, retrain, or extend.

Manual tagging tools (Wyscout, hudl) let you annotate your own footage, but tagging is slow, inconsistent across operators, and doesn't scale to continuous or real-time use.

## The gap

The raw signal is available — football clubs at every level record their matches. The gap is the pipeline to turn pixels into structured data automatically: who is on screen, where, doing what, and when.

## What footy-track does

footy-track takes a match video and produces a stream of structured, time-stamped records:

- **Object detections** — bounding boxes for players, ball, referee, coach, and substitutes, per frame.
- **Broadcast classifications** — whether a given frame is a pitch view suitable for analysis (vs. a replay, crowd shot, or graphics overlay).
- **Events** (planned) — higher-level inferences: passes, shots, tackles, set-pieces, substitutions, derived from detection trajectories.

Each record carries a **ContinuousTime** timestamp (seconds from first-half kickoff, never resetting across halves) so downstream consumers can reliably align, merge, and resample the data.

## Why it's hard

**Broadcast video is noisy.** A match broadcast is not a clean overhead feed. Camera angles change, the ball disappears under players, directors cut to replays and interviews. A naive detector running on every frame wastes compute on frames that contain no useful pitch information.

**Detection alone isn't enough.** Knowing a bounding box exists in frame 4,231 is not useful without knowing it belongs to the same player who was in frame 4,230. Associating detections across frames (tracking) is a separate, hard problem.

**COCO models don't understand football.** A standard YOLO model trained on COCO can find "person" and "sports ball" but cannot distinguish a goalkeeper from a coach, an in-play ball from a dead ball, or a substitute warming up from a referee. Domain-specific fine-tuning on labelled match footage is required.

**Timing is non-trivial.** Football has two halves with independent clocks plus variable stoppage time. Converting the broadcast clock into a continuous, match-wide time coordinate requires knowing when the second half kicked off — information that must be supplied or inferred.

## Who this is for

- Clubs that record their own matches and want automated analysis without a commercial data subscription.
- Researchers and developers building football analytics tools who need a reproducible, open pipeline.
- Anyone who wants to fine-tune the detection or tracking models on their own footage and integrate the results into a downstream system.
