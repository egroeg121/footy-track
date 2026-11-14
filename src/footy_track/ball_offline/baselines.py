"""Reference trackers: B0 (per-frame) and main's ``ViterbiBallTracker``.

``viterbi`` runs ``footy_track.ball_trackers.viterbi.ViterbiBallTracker`` with its
default settings over the clip's candidates, exactly as the inference pipeline does
(candidates in normalised top-left boxes, one ``update`` per frame, ``points()`` at
the end); interpolated points count as predictions.  Any other tracker is scored by
writing its per-frame output as jsonl (see ``score.py``).
"""

from __future__ import annotations

import numpy as np

from footy_track.ball_offline.data import H_PX, W_PX
from footy_track.ball_offline.metrics import Pred

B0_TAU = 0.4  # tau chosen on the dev clips (see the reproduction report)


def b0_per_frame(clip, tau: float = B0_TAU) -> Pred:
    """Top-confidence ball candidate per frame if conf >= tau."""
    pred: Pred = {}
    for t in range(clip.n_frames):
        xy, conf = clip.cands(t)
        if len(xy):
            k = int(np.argmax(conf))
            if conf[k] >= tau:
                pred[t] = (float(xy[k, 0]), float(xy[k, 1]))
    return pred


def viterbi(clip, **params) -> Pred:
    """Main's offline Viterbi ball tracker over the clip's candidates."""
    from footy_track.ball_trackers.detection_tracker import BallCandidate  # noqa: PLC0415
    from footy_track.ball_trackers.viterbi import ViterbiBallTracker  # noqa: PLC0415

    trk = ViterbiBallTracker((int(W_PX), int(H_PX)), **params)
    for t in range(clip.n_frames):
        s, e = clip.ptr[t], clip.ptr[t + 1]
        trk.update(
            t,
            [
                BallCandidate(
                    (x - w / 2) / W_PX,
                    (y - h / 2) / H_PX,
                    w / W_PX,
                    h / H_PX,
                    float(c),
                )
                for (x, y), (w, h), c in zip(
                    clip.xy[s:e], clip.wh[s:e], clip.conf[s:e], strict=True
                )
            ],
        )
    pred: Pred = {}
    for p in trk.points():
        if p.x is not None:
            pred[p.frame_index] = ((p.x + p.w / 2) * W_PX, (p.y + p.h / 2) * H_PX)
    return pred


METHODS = {"b0": b0_per_frame, "viterbi": viterbi}
