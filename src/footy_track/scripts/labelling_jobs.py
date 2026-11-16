"""Submit labelling jobs to the main labeller and fetch results.

    python -m footy_track.scripts.labelling_jobs submit --title "Nottingham Forest windows" \\
        --kind ball_place --dataset nottinghamforest --items windows.csv --step 5 --key my-branch/nottinghamforest-1
    python -m footy_track.scripts.labelling_jobs status  "Nottingham Forest windows"
    python -m footy_track.scripts.labelling_jobs wait    "Nottingham Forest windows" --poll 60
    python -m footy_track.scripts.labelling_jobs results "Nottingham Forest windows" -o labels.json

A thin wrapper over ``footy_track/labeller/jobs_client.py`` (which is a standalone single
file you can ``curl`` from the server). ``requester.branch`` / ``commit`` come from git.
"""

from __future__ import annotations

from footy_track.labeller.jobs_client import main

if __name__ == "__main__":
    raise SystemExit(main())
