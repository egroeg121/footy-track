# Time Formats and Conversion

This document explains time formats used by Footy Scan and how to convert between them.

## Time formats

- ContinuousTime
  - Definition: time in seconds from game start (0.0 = kickoff of first half). Continuous through the entire match including stoppage time and both halves. Used as the canonical storage and interchange format for observations and events.
  - Notes: ContinuousTime does not reset at half-time, and it accounts for stoppage time naturally.

- GameTime
  - Definition: time shown by the referee/broadcast clock. Resets at half-time (i.e., second half GameTime starts at 0:00). Includes stoppage time per half.
  - Notes: GameTime is what video feeds and broadcasters display; it is not continuous across the full match.

## Conversion rules

- GameMetadata
  - Contains mapping information required to convert GameTime to ContinuousTime for a given video or broadcast. Fields may include:
    - half (1 or 2)
    - game_start_wallclock (wall-clock time of the kickoff)
    - half_start_continuous (ContinuousTime when the half started)
    - offset_seconds (any manual offset if needed)

- Conversion logic
  - First half: ContinuousTime = GameTime (in seconds).
  - Second half: ContinuousTime = GameTime (in seconds) + half_start_continuous, where half_start_continuous is the ContinuousTime at second-half kickoff. This accounts for any additional stoppage from the first half.

## Usage for video input

- When consuming a broadcast or recorded video, prefer to parse GameTime and use GameMetadata to compute ContinuousTime for each frame.
- For first-half frames, ContinuousTime will match GameTime if half_start_continuous == 0. For second-half frames, apply the half offset so ContinuousTime continues from the end of the first half.

## Examples

- If first half lasted 48.5 minutes (including stoppage), second-half kickoff ContinuousTime = 48.5 * 60 = 2910.0 seconds. A GameTime value of 00:05:00 (5 minutes) in the second half corresponds to ContinuousTime = 5 * 60 + 2910.0 = 3210.0 seconds.
