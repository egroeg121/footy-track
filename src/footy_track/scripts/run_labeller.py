"""Entry point for the SAM3 video labeller Streamlit app.

Run with::

    uv run streamlit run src/footy_track/scripts/run_labeller.py

Sets TORCHINDUCTOR_CACHE_DIR so compiled MPS kernels are reused across restarts,
then calls the app's ``main()``.
"""

from footy_track.labeller.app import main

main()
