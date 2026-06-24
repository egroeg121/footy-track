# VitTrack label interpolation in the labeller (ft-4e9)

Human-in-the-loop label assist: a human hand-labels a frame, VitTrack carries each
box forward frame-by-frame until it loses confidence or the track goes implausible,
then **hands back** to the human at the frame where it gave up.

This is the practical realisation of the SAM3-style "place once, propagate" interaction
the user wanted — but real-time. The [ft-5hd bake-off](../bakeoff_results.md) proved
VitTrack SOT is a strong *warm* tracker (≈79.8% centre-accuracy / 0.54 IoU on the ball
at ≈10 FPS on CPU/CoreML, zero VRAM) given a human-placed seed. It cannot cold-start,
but in the labeller the human always supplies the seed, so that limitation never bites.

## Design principles

1. **Never silently overwrite a human mark.** Interpolated boxes carry a distinct
   provenance (`PROV_VITTRACK = "vittrack"`) and are written via
   `Session.merge_propagated`, which always keeps `PROV_LABELLER` ground truth.
2. **Hand back early, not late.** A wrong box that the human must hunt for is worse
   than an early stop. The trigger thresholds are deliberately conservative.
3. **One tracker instance per object.** The ball is single; players are multi. Each
   tracked box gets its own `VitTrackSOT` seeded from the anchor frame and run
   independently — pure interpolation between human anchors needs no cross-object
   association.
4. **Trigger logic is pure and unit-tested.** All hand-back decisions live in
   `labeller/interpolation.py` as side-effect-free functions, so the trust-critical
   part is tested without a model or a video.

## Hand-back triggers

Evaluated per object, per frame, in `interpolation.py:should_handback`. The first
trigger that fires stops *that object's* track and records the reason.

| Trigger | Condition | Why |
|---|---|---|
| `lost` | VitTrack returns `None`, or score `< min_score` | VitTrack "drops the ball" — its own confidence collapsed. Primary trigger. |
| `center_jump` | centre moved `> max_center_jump_frac` of the frame diagonal in one frame | Catches teleports onto a stale patch (bake-off seg100 showed 255 px drift). |
| `size_jump` | new area / old area outside `[1/max_size_ratio, max_size_ratio]` | Catches the box ballooning/collapsing as the tracker locks onto background. |

Defaults (`HandbackConfig`): `min_score=0.30`, `max_center_jump_frac=0.08`,
`max_size_ratio=2.5`. These are starting points tuned from the bake-off; surfaced as
endpoint parameters so they can be adjusted without a code change.

`should_handback(prev_bbox, new_bbox, score, cfg) -> HandbackResult` returns either
`HandbackResult(stop=False)` (continue) or `HandbackResult(stop=True, reason=...)`.
The last *good* frame is the frame before the stop — the interpolated boxes up to and
including the last good frame are persisted; the stop frame itself is where the human
resumes.

## Server: `POST /interpolate`

Decision: **add a new `/interpolate` endpoint** rather than extend `/propagate`.
`/propagate` does IoU re-labelling between boxes that *already exist* in later frames;
this does *visual* tracking that fills frames with **no** detection and stops on
failure. Different inputs, different output, different stop semantics — a separate
endpoint keeps both simple.

Request:
```json
{
  "anchor_idx": 120,
  "end_idx": 400,                 // optional; defaults to last frame
  "labels": ["in_play_ball"],     // optional; which anchor labels to track (default: all)
  "min_score": 0.30,              // optional trigger overrides
  "max_center_jump_frac": 0.08,
  "max_size_ratio": 2.5
}
```

Per anchor box matching `labels`, the server:
1. Seeds a fresh `VitTrackSOT` from the anchor box on the anchor frame.
2. Steps forward frame-by-frame, calling `track_with_score(prev_bbox, frame)`.
3. Runs `should_handback`; on stop, records the handback frame + reason and ends that
   object's track.
4. Writes each accepted interpolated box into the timeline via `merge_propagated`
   with `PROV_VITTRACK` (human GT on those frames is preserved).

Response:
```json
{
  "anchor_idx": 120,
  "objects": [
    {"label": "in_play_ball", "handback_idx": 168, "reason": "center_jump",
     "last_good_idx": 167, "frames_tracked": 47}
  ],
  "handback_idx": 168        // min over objects — the frame the human jumps to
}
```

`handback_idx` is the earliest stop across all objects: that's where the human's
attention is needed first, so the UI jumps there.

## Tracker change

`VitTrackSOT.track_with_score(prev_bbox, frame) -> (bbox | None, score)` exposes the
confidence the existing `track()` already computes internally. `track()` is unchanged
(bake-off harness depends on its `bbox | None` contract); it now delegates to
`track_with_score` and drops the score.

## UI

- **Interpolate (I)** button (and `I` hotkey) — runs `/interpolate` from the current
  frame as the anchor. Disabled while a run is live.
- Interpolated boxes render **dashed** in the object's class colour (vs solid for human
  marks) so the human can never confuse an assist box for a hand label. Driven by box
  `source === "vittrack"` already carried in `/timeline` payloads.
- On completion the UI **jumps to `handback_idx`** and shows a status line with the
  reason (`"VitTrack handed back at 168: center_jump"`), so the human lands exactly
  where correction is needed, fixes the box, and can re-interpolate from there.
- Accept/correct is the normal edit flow: editing any frame writes `PROV_LABELLER`
  ground truth, which from then on survives re-interpolation.

## Persistence

Interpolated boxes flush to the JSONL sidecar like any other box, tagged
`[label, "vittrack"]`. Re-loading a clip restores them as non-GT boxes. The human
promotes one to ground truth simply by editing the frame.

## Tests

`tests/labeller/test_interpolation.py` covers `should_handback`: continue case, each
trigger in isolation (lost / low-score / center jump / size jump), boundary values
around each threshold, and config override behaviour. No model or video required.
