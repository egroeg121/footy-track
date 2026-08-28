"""Run the labeller server.

    python -m footy_track.labeller                 # normal (needs a GPU/checkpoint)
    python -m footy_track.labeller --debug         # VERY lightweight, CPU-only

``--debug`` exists so the labeller can be developed on a machine with no GPU and
little free RAM. It:

* sets ``FOOTY_DEBUG=1``, which swaps the RT-DETR detector for a stub that loads
  no weights, downloads nothing and never touches CUDA;
* limits how many clips are scanned (``--max-clips``), so startup does not stat
  a large video directory;
* runs a single uvicorn worker with reload disabled.

Point it at local data with ``--clips-dir`` / ``--gt-dir`` (or the
``FOOTY_CLIPS_DIR`` / ``FOOTY_GT_MARKS_DIR`` environment variables) rather than
relying on machine-specific default paths.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m footy_track.labeller")
    parser.add_argument(
        "--debug",
        action="store_true",
        help="lightweight CPU-only mode: no model weights are loaded",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--clips-dir", type=Path, default=None)
    parser.add_argument("--gt-dir", type=Path, default=None)
    parser.add_argument(
        "--max-clips",
        type=int,
        default=None,
        help="only expose the first N clips (default: 25 under --debug, all otherwise)",
    )
    args = parser.parse_args(argv)

    if args.debug:
        os.environ["FOOTY_DEBUG"] = "1"
        if args.max_clips is None:
            args.max_clips = 25
    if args.clips_dir:
        os.environ["FOOTY_CLIPS_DIR"] = str(args.clips_dir.expanduser())
    if args.gt_dir:
        os.environ["FOOTY_GT_MARKS_DIR"] = str(args.gt_dir.expanduser())
    if args.max_clips is not None:
        os.environ["FOOTY_MAX_CLIPS"] = str(args.max_clips)

    # Imported after the environment is set: server.py resolves its directories
    # at import time.
    import uvicorn  # noqa: PLC0415

    from footy_track.labeller import server  # noqa: PLC0415

    if args.debug:
        print(
            "[debug] lightweight mode: stub detector (no weights loaded)\n"
            f"[debug] clips: {server._CLIPS_DIR}\n"
            f"[debug] gt:    {server._GT_MARKS_DIR}\n"
            f"[debug] max clips: {args.max_clips}",
            flush=True,
        )

    uvicorn.run(server.app, host=args.host, port=args.port, reload=False, workers=1)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
