from __future__ import annotations

import pandas as pd

from src import pipeline


def test_market_date_guard_skips_a_session_that_is_already_published(monkeypatch) -> None:
    monkeypatch.setattr(
        pipeline,
        "_latest_published_signal_date",
        lambda: pd.Timestamp("2026-08-04"),
    )

    assert pipeline._should_run_for_market_date("2026-08-04") is False
    assert pipeline._should_run_for_market_date("2026-08-05") is True


def test_market_date_guard_allows_first_published_session(monkeypatch) -> None:
    monkeypatch.setattr(pipeline, "_latest_published_signal_date", lambda: None)

    assert pipeline._should_run_for_market_date("2026-08-05") is True
