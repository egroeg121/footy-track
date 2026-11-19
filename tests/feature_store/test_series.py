"""``series`` table: registered windows, idempotent on (game, start, end)."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from footy_track.feature_store.schema import SeriesRow
from footy_track.feature_store.store import FeatureStore


@pytest.fixture
def store():
    with FeatureStore.open(":memory:") as s:
        yield s


def test_register_and_read(store):
    now = datetime.now(UTC)
    store.register_series(
        [
            SeriesRow(
                game_id="g",
                start_frame=0,
                end_frame=250,
                run_id="autolabel:g_f000000",
                verified_at=now,
            )
        ]
    )
    df = store.read_series("g")
    assert df[["start_frame", "end_frame", "run_id"]].values.tolist() == [
        [0, 250, "autolabel:g_f000000"]
    ]
    assert store.read_series("other").empty


def test_register_is_idempotent_and_updates_run(store):
    store.register_series(
        [SeriesRow(game_id="g", start_frame=0, end_frame=250, run_id="a")]
    )
    store.register_series(
        [SeriesRow(game_id="g", start_frame=0, end_frame=250, run_id="b")]
    )
    store.register_series(
        [SeriesRow(game_id="g", start_frame=250, end_frame=500, run_id="b")]
    )
    df = store.read_series("g")
    assert len(df) == 2
    assert df["run_id"].tolist() == ["b", "b"]


def test_end_must_follow_start(store):
    with pytest.raises(Exception):  # noqa: B017, PT011 - duckdb CHECK constraint
        store.register_series([SeriesRow(game_id="g", start_frame=10, end_frame=10)])
