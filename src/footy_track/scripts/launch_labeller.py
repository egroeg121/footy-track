"""Console-script launcher for the SAM3 video labeller.

Sets TORCHINDUCTOR_CACHE_DIR (so compiled MPS kernels persist across restarts)
*before* exec'ing ``streamlit run``, then hands off. torch reads the cache dir
during import, so it must be in the environment before the streamlit process
starts — which is why this can't live inside run_labeller.py itself.

Usage::

    uv run footy-labeller
    uv run footy-labeller -- --server.port 8765   # extra streamlit args after --
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


def main() -> None:
    cache_dir = Path.home() / ".cache" / "torchinductor_sam3labeller"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", str(cache_dir))
    os.environ.setdefault("TORCHINDUCTOR_FX_GRAPH_CACHE", "1")

    app_path = Path(__file__).with_name("run_labeller.py")
    extra_args = sys.argv[1:]

    # Replace this process with streamlit so signals/Ctrl-C behave normally.
    argv = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(app_path),
        "--server.fileWatcherType",
        "none",
        *extra_args,
    ]
    os.execv(sys.executable, argv)


if __name__ == "__main__":
    main()
