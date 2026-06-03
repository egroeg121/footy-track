"""Entry point for the SAM3 video labeller Streamlit app.

Do NOT run this file directly with python — launch it via Streamlit::

    uv run footy-labeller          # convenience wrapper (sets cache env vars)
    # or manually:
    TORCHINDUCTOR_CACHE_DIR=~/.cache/torchinductor_sam3labeller \\
        uv run streamlit run src/footy_track/scripts/run_labeller.py

The TORCHINDUCTOR_CACHE_DIR env var MUST be set in the shell before the process
starts — torch reads it during import, so setting it inside this file is too late.
Use the ``footy-labeller`` console script (see launch_labeller.py) which sets it
before exec'ing streamlit.
"""

from footy_track.labeller.app import main

main()
