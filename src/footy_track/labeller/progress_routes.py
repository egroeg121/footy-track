"""``/progress`` page and ``/api/progress`` JSON: dataset growth, model results, split health
(see progress.py). Progress is global: the ``?dataset=`` labelling-dataset parameter is ignored."""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter
from fastapi.responses import HTMLResponse

from footy_track.labeller import labelling_progress, progress

router = APIRouter()
_PAGE = Path(__file__).parent / "web" / "progress.html"


@router.get("/progress", response_class=HTMLResponse)
async def progress_page() -> HTMLResponse:
    return HTMLResponse(_PAGE.read_text(), headers={"Cache-Control": "no-store, max-age=0"})


@router.get("/api/progress")
def progress_api(refresh: bool = False) -> dict:
    # plain def: reads manifests and a DuckDB file, so FastAPI runs it in a worker thread.
    return progress.get_progress(force=refresh)


@router.get("/progress/labelling")
def progress_labelling() -> dict:
    """Ball / player / fully-labelled frame counts per clip, from the feature store."""
    return labelling_progress.get_labelling(audit_rates=_audit_rates())


def _audit_rates() -> dict[str, dict]:
    """Blind-audit pass rate per teacher version from series-review status docs (best effort)."""
    import json  # noqa: PLC0415

    from footy_track.autolabel.teacher import audit_pass_rates  # noqa: PLC0415
    from footy_track.labeller.series_review import autolabel_dir  # noqa: PLC0415

    docs = []
    try:
        for p in sorted((autolabel_dir() / "series_review").glob("*.json")):
            docs.append(json.loads(p.read_text()))
    except (OSError, ValueError):
        return {}
    return audit_pass_rates(docs)
