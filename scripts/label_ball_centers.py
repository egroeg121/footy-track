"""Human-anchored sparse ball-center labelling tool.

Usage::

    uv run python scripts/label_ball_centers.py VIDEO.mp4 [--step 5] [--out OUTPUT.jsonl]

Keyboard shortcuts:
  Left / Right arrow   : previous / next frame (step size set by --step)
  Click on frame       : mark ball center at click position
  U                    : undo last label on this frame
  N                    : mark ball as not visible on this frame (writes bbox=null)
  S                    : save progress (also auto-saved on exit)
  Q / Escape           : quit and save

Output: a JSONL file with one entry per labelled frame::

    {"frame_index": 0, "center": [0.451, 0.312], "tags": []}
    {"frame_index": 5, "bbox": null, "tags": ["ball_not_visible"]}

Pass --tags to attach a hard-case tag to ALL frames labelled in this run
(useful when you know the whole clip has a specific challenge)::

    uv run python scripts/label_ball_centers.py clip.mp4 --tags small_ball

The output can be loaded directly by EvalDataset / EvalClip. Frames with only
a center (no bbox) still participate in all center-distance metrics.

Requires: opencv-python with GUI support (cv2.imshow must work).
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys


def _require_cv2_gui():
    try:
        import cv2  # noqa: PLC0415

        # Quick smoke-test: if headless build, createTrackbar will raise
        cv2.namedWindow("__test__")
        cv2.destroyWindow("__test__")
        return cv2
    except Exception as exc:
        sys.exit(
            f"OpenCV GUI not available: {exc}\nInstall opencv-python (not -headless)."
        )


def main():  # noqa: PLR0912, PLR0915
    parser = argparse.ArgumentParser(description="Sparse ball-center GT labeller")
    parser.add_argument("video", type=pathlib.Path, help="Input video file")
    parser.add_argument(
        "--step", type=int, default=5, help="Frame step between labels (default: 5)"
    )
    parser.add_argument(
        "--out",
        type=pathlib.Path,
        default=None,
        help="Output JSONL (default: <video>.jsonl)",
    )
    parser.add_argument(
        "--tags", nargs="*", default=[], help="Hard-case tags to attach to all labels"
    )
    args = parser.parse_args()

    cv2 = _require_cv2_gui()

    video_path: pathlib.Path = args.video
    if not video_path.exists():
        sys.exit(f"Video not found: {video_path}")

    out_path: pathlib.Path = args.out or video_path.with_suffix(".jsonl")
    step: int = max(1, args.step)
    default_tags: list[str] = args.tags or []

    # Load existing labels so re-runs are additive
    labels: dict[int, dict] = {}
    if out_path.exists():
        with out_path.open() as f:
            for raw_line in f:
                stripped = raw_line.strip()
                if stripped:
                    d = json.loads(stripped)
                    labels[d["frame_index"]] = d
        print(f"Loaded {len(labels)} existing labels from {out_path}")

    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    print(
        f"Video: {video_path.name}  {total_frames} frames @ {fps:.1f} fps  ({width}x{height})"
    )
    print(f"Step: every {step} frames  |  Output: {out_path}")
    print(
        "Controls: ←/→ move, click = label center, N = not visible, U = undo, S = save, Q/Esc = quit"
    )

    # Build list of frames to visit
    frame_indices = list(range(0, total_frames, step))
    cursor = 0  # index into frame_indices

    # State: pending label for current frame (before commit)
    pending_center: tuple[float, float] | None = None
    pending_absent: bool = False
    dirty = False

    win = "Ball Center Labeller"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    click_pos: list[tuple[float, float]] = []

    def _on_mouse(event, x, y, flags, param):
        nonlocal pending_center, pending_absent
        if event == cv2.EVENT_LBUTTONDOWN:
            pending_center = (x / width, y / height)
            pending_absent = False
            click_pos.append(pending_center)

    cv2.setMouseCallback(win, _on_mouse)

    def _read_frame(idx: int):
        cap2 = cv2.VideoCapture(str(video_path))
        cap2.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, bgr = cap2.read()
        cap2.release()
        return bgr if ok else None

    def _save():
        with out_path.open("w") as f:
            for fi in sorted(labels):
                f.write(json.dumps(labels[fi]) + "\n")
        print(f"Saved {len(labels)} labels → {out_path}")

    def _draw_frame(bgr, fi: int):
        """Draw labels on frame and show status."""
        cv = cv2
        disp = bgr.copy()
        # Draw existing label for this frame
        lbl = labels.get(fi)
        if lbl:
            if lbl.get("center"):
                cx, cy = lbl["center"]
                px, py = int(cx * width), int(cy * height)
                cv.circle(disp, (px, py), 10, (0, 255, 0), 2)
                cv.circle(disp, (px, py), 2, (0, 255, 0), -1)
            elif lbl.get("bbox") is None:
                cv.putText(
                    disp,
                    "NOT VISIBLE",
                    (10, 60),
                    cv.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (0, 0, 255),
                    2,
                    cv.LINE_AA,
                )
        # Draw pending center
        if pending_center:
            px, py = int(pending_center[0] * width), int(pending_center[1] * height)
            cv.circle(disp, (px, py), 10, (0, 165, 255), 2)
        # Status bar
        n_done = len(labels)
        n_total = len(frame_indices)
        status = (
            f"Frame {fi}/{total_frames - 1}  |  {n_done}/{n_total} labelled  |  "
            f"{'[PENDING]' if pending_center or pending_absent else ''}"
        )
        cv.putText(
            disp,
            status,
            (10, 30),
            cv.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
            cv.LINE_AA,
        )
        cv.imshow(win, disp)

    while True:
        fi = frame_indices[cursor]
        bgr = _read_frame(fi)
        if bgr is None:
            print(f"Could not read frame {fi}, skipping")
            cursor = min(cursor + 1, len(frame_indices) - 1)
            continue

        # Commit any pending label from previous iteration before rendering
        _draw_frame(bgr, fi)
        click_pos.clear()
        pending_center = None
        pending_absent = False

        # Wait for keypress or mouse click
        while True:
            _draw_frame(bgr, fi)
            key = cv2.waitKey(50) & 0xFF

            if click_pos:
                pending_center = click_pos[-1]

            # Commit center label
            if pending_center and key in (ord(" "), 13):  # space or Enter commits
                labels[fi] = {
                    "frame_index": fi,
                    "center": list(pending_center),
                    "tags": default_tags,
                }
                dirty = True
                pending_center = None
                click_pos.clear()
                break  # advance

            # Auto-commit on arrow key — click then arrow moves forward
            if pending_center and key in (
                81,
                83,
                2,
                3,
                0xFF & ord("a"),
                0xFF & ord("d"),
            ):
                labels[fi] = {
                    "frame_index": fi,
                    "center": list(pending_center),
                    "tags": default_tags,
                }
                dirty = True
                pending_center = None
                click_pos.clear()
                # fall through to movement below

            if key in (ord("n"), ord("N")):
                labels[fi] = {
                    "frame_index": fi,
                    "bbox": None,
                    "tags": ["ball_not_visible"] + default_tags,
                }
                dirty = True
                cursor = min(cursor + 1, len(frame_indices) - 1)
                break

            if key in (ord("u"), ord("U")):
                labels.pop(fi, None)
                dirty = True
                pending_center = None
                click_pos.clear()

            if key in (ord("s"), ord("S")):
                _save()
                dirty = False

            if key in (ord("q"), ord("Q"), 27):  # Q or Esc
                if dirty:
                    _save()
                cv2.destroyAllWindows()
                print("Done.")
                return

            # Navigation
            if key in (83, 3, 0xFF & ord("d")):  # right / d
                cursor = min(cursor + 1, len(frame_indices) - 1)
                break
            if key in (81, 2, 0xFF & ord("a")):  # left / a
                cursor = max(cursor - 1, 0)
                break

            # If a center was clicked and no nav key, keep showing the pending dot
            if pending_center:
                continue

    # Shouldn't reach here, but save on exit
    if dirty:
        _save()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
