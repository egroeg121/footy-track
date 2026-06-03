"""Entry point for the SAM3 video labeller Streamlit app.

Run with::

    uv run streamlit run src/footy_track/scripts/run_labeller.py

Sets TORCHINDUCTOR_CACHE_DIR so compiled MPS kernels are reused across restarts,
then calls the app's ``main()``.
"""

import os
from pathlib import Path

# Persist torch.compile / inductor kernel cache so JIT compilation is only done once.
_cache_dir = Path.home() / ".cache" / "torch_inductor_sam3"
_cache_dir.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", str(_cache_dir))

from footy_track.labeller.app import main  # noqa: E402

main()
