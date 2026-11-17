"""Queue a Prefect flow run over the REST API (stdlib only; no ``prefect`` import).

Used by the labelling-jobs server to start ``ballcheck-prepare`` when a job is submitted with
``prepare`` (or when labelling resumes on a job whose renders were evicted), so a client needs no
Prefect credentials of its own. Auth comes from ``PREFECT_API_AUTH_STRING`` (``user:pass``) in the
environment, else from the ``PREFECT_API_AUTH_STRING=`` line of ``$FOOTY_PREFECT_ENV_FILE``
(default ``~/deploy/prefect.env`` on the mini). The secret is never logged or returned.
"""

from __future__ import annotations

import base64
import json
import os
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

DEFAULT_API = "http://127.0.0.1:4200/api"
DEFAULT_ENV_FILE = "~/deploy/prefect.env"
DEPLOYMENT = "ballcheck-prepare/ballcheck-prepare"


class PrefectError(RuntimeError):
    pass


def _file_value(path: Path, name: str) -> str | None:
    try:
        for raw in path.read_text().splitlines():
            k, sep, v = raw.partition("=")
            if sep and k.strip() == name:
                return v.strip().strip("'\"") or None
    except OSError:
        return None
    return None


def api_url() -> str:
    return (os.environ.get("PREFECT_API_URL") or DEFAULT_API).rstrip("/")


def auth_header() -> dict:
    auth = os.environ.get("PREFECT_API_AUTH_STRING")
    if not auth:
        env_file = Path(
            os.environ.get("FOOTY_PREFECT_ENV_FILE") or DEFAULT_ENV_FILE
        ).expanduser()
        auth = _file_value(env_file, "PREFECT_API_AUTH_STRING")
    if not auth:
        return {}
    return {"Authorization": "Basic " + base64.b64encode(auth.encode()).decode()}


def _call(
    method: str, path: str, body: dict | None = None, timeout: float = 15
) -> dict:
    req = urllib.request.Request(  # noqa: S310 (operator-configured http URL)
        api_url() + path,
        data=None if body is None else json.dumps(body).encode(),
        method=method,
        headers={"Content-Type": "application/json", **auth_header()},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:  # noqa: S310
            return json.loads(r.read().decode())
    except urllib.error.HTTPError as e:
        raise PrefectError(f"Prefect API {method} {path}: HTTP {e.code}") from None
    except (urllib.error.URLError, OSError, ValueError) as e:
        raise PrefectError(f"Prefect API {method} {path}: {e}") from None


def create_flow_run(
    parameters: dict, *, deployment: str = DEPLOYMENT, name: str | None = None
) -> str:
    """Queue a run of ``flow/deployment`` and return its flow-run id."""
    dep = _call(
        "GET",
        "/deployments/name/"
        + "/".join(urllib.parse.quote(p, safe="") for p in deployment.split("/")),
    )
    body: dict = {"parameters": parameters}
    if name:
        body["name"] = name
    run = _call("POST", f"/deployments/{dep['id']}/create_flow_run", body)
    return str(run["id"])
