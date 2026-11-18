"""``/matches`` page and ``/api/matches`` JSON: per-match pipeline status (see matches_status.py)."""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter
from fastapi.responses import HTMLResponse

from footy_track.labeller import matches_status

router = APIRouter()
_PAGE = Path(__file__).parent / "web" / "matches.html"


@router.get("/matches", response_class=HTMLResponse)
async def matches_page() -> HTMLResponse:
    return HTMLResponse(
        _PAGE.read_text(), headers={"Cache-Control": "no-store, max-age=0"}
    )


@router.get("/api/matches")
def matches_api(refresh: bool = False) -> dict:
    # plain def: the scan does blocking file / HTTP IO, so FastAPI runs it in a worker thread.
    return matches_status.get_status(force=refresh)
