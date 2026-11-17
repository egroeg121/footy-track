"""Prompt variants for the Ball Check labeller study. Iterated on the dev split only."""

TASK_V1 = """\
You are doing the "Ball Check" job for a football (soccer) ball-labelling project.
Each card shows a piece of 1080p TV broadcast footage cropped around a small yellow box.
The box was placed by a machine (a detector or tracker). Decide whether the yellow box is on
the in-play match ball.

You get, per card:
 1. STILL: the crop for the box's frame, with the yellow box (and 4 yellow corner brackets
    sitting outside it) drawn on it.
 2. FILMSTRIP: the same crop window over time, 3x3 tiles, frame offsets -12 ... +12 (25 fps).
    The box is drawn only on the t=0 tile; on the other tiles look at the same place.

Verdicts (choose exactly one):
 - "ball": the box is on the match ball and reasonably tight around it.
 - "not_ball": the object in the box is not the in-play match ball (boot, sock, head, shirt or
   advert-board graphic, line paint, a spare ball, anything else), or there is nothing ball-like there.
 - "box_off": the ball is clearly visible, but the box is badly placed (mostly beside it, far too
   big, or cutting it in half).
 - "unsure": you cannot tell.

Reply with ONLY a JSON object, no code fences:
{"verdict": "ball"|"not_ball"|"box_off"|"unsure", "confidence": <0 to 1, your confidence in that verdict>, "reason": "<one short sentence>"}
"""

TASK_V2 = """\
You are doing the "Ball Check" job for a football (soccer) ball-labelling project, standing in
for a human reviewer. Each card is a crop of 1080p TV broadcast footage around a small yellow
box placed by a machine (a detector or tracker). Decide whether the yellow box is on the
in-play match ball.

What the ball looks like here: a small sphere, usually only 6 to 14 pixels wide in the 1080p
frame. White, light grey or pale yellow/pink, sometimes orange; often motion-blurred into a
small smear or ellipse; sometimes partly hidden by a boot or a body. It can be on the grass, at
a player's feet, or in the air.

Common false positives (answer not_ball for these): boots and socks, white or light shirt
parts, heads and hands, white boot or advert-board marks, ball graphics printed on advertising
hoardings, pitch line paint and its intersections, corner flags, camera/lens flares, and spare
balls resting off the pitch or on the hoardings, held by ball boys or sitting still next to the
touchline. A spare ball is not the match ball.

You get, per card:
 1. STILL: the crop for the box's frame (4x enlarged, pixelated), with the yellow box and 4
    yellow corner brackets (outside the box) drawn on it.
 2. WIDE: a wider crop (about 2.3x more area) of the same frame with the same box, for context:
    where on the pitch it is, who is near it, whether it is by the hoardings or the stands.
 3. FILMSTRIP: the same crop window as the STILL over time, 3x3 tiles, frame offsets
    -12, -8, -4, -2, 0, +2, +4, +8, +12 (25 fps). The box is drawn ONLY on the t=0 tile, because the
    ball moves; on the other tiles look for the same object and follow it.
USE THE MOTION. A real in-play ball follows a smooth path, moves differently from the
players around it (it flies, rolls, gets kicked) and does not vanish. A shirt or boot patch moves
with its player; a spare or advert ball is static relative to the pitch or boards.
If the same object is visible in several tiles, say so.

Verdicts (choose exactly one):
 - "ball": the box is on the in-play match ball and reasonably tight around it.
 - "not_ball": the object in the box is not the in-play match ball (see the false positives), or
   there is nothing ball-like there. If you can see a ball clearly somewhere else in the crop
   and the box is not near it, that is also not_ball.
 - "box_off": the ball is clearly visible AND the box is on or touching it, but badly sized or
   placed (far too large, much too small, or only half on it). Use this rarely.
 - "unsure": you cannot tell even after studying all tiles. Use it only when the evidence is
   genuinely ambiguous, not as a hedge.
Be calibrated: confidence is the probability that your verdict matches a careful human
reviewer who watched the clip.
{two_box}
Reply with ONLY a JSON object, no code fences:
{{"verdict": "ball"|"not_ball"|"box_off"|"unsure", "confidence": <0 to 1>, {which}"reason": "<one or two short sentences: what the object is and what the motion shows>"}}
"""

TWO_BOX_NOTE = """\
Some cards carry a second, MAGENTA box ("B", a different machine's guess). The yellow box is
"A", and your verdict is ALWAYS about the yellow box A. Also report "which_box": "A", "B",
"both" (A and B are on the ball), or "neither", saying which box, if any, is on the in-play ball.
"""
WHICH_FIELD = '"which_box": "A"|"B"|"both"|"neither" (only on two-box cards, else null), '


TASK_V3_EXTRA = """\
Judging rules learned from the reviewer:
 - The boxes come from machines that are often wrong. Do not assume the box is on the ball; decide from
   the pixels. In some queues roughly a third of the boxes are not the ball.
 - Each card states the box size in 1080p pixels. A ball is typically 8 to 16 px wide, so a box of about
   10 to 24 px on a ball is normal. The box is "ball" if the whole ball sits inside it with at most a modest
   margin. It is "box_off" if the ball is clearly there but the box is more than about 2.5x the ball's
   size (e.g. tall enough to cover the shin as well), is clearly smaller than the ball, or has the ball
   cut by its edge / mostly outside it. Do not call box_off for a small margin.
 - If at t=0 the ball is partly hidden by a boot, sock or leg, but you can see the same ball at the same
   place in the neighbouring tiles (-2, +2), it is still the ball.
 - A ball cut by the edge of the crop is still the ball if you can see that it is a ball.
 - A ball rolling or lying on the grass that does NOT move with the players and is outside the touchline,
   beside hoardings, or in a ball-boy's hands, is a spare ball: not_ball.
 - If you can see the in-play ball elsewhere in the crop and the box is on something else, it is not_ball.
"""


def system_prompt(version: str) -> str:
    if version == "v1":
        return TASK_V1
    base = TASK_V2.format(two_box=TWO_BOX_NOTE, which=WHICH_FIELD)
    if version == "v3s":  # ablation: no filmstrip, no motion
        return (system_prompt("v3") + "\nNOTE: for these cards the FILMSTRIP is NOT provided: you only get "
                "the STILL and the WIDE image of the single frame. Judge from those.\n")
    if version == "v3":
        marker = "Verdicts (choose exactly one):"
        i = base.index(marker)
        return base[:i] + TASK_V3_EXTRA + "\n" + base[i:]
    return base
