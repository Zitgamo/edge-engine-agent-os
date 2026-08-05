from __future__ import annotations

import pandas as pd
from datetime import datetime
from zoneinfo import ZoneInfo

from src import pipeline


def test_closed_sessions_excludes_today_before_market_close() -> None:
    dates = pd.to_datetime(["2026-08-04", "2026-08-05"])
    df = pd.DataFrame({"date": dates, "close": [100.0, 101.0]})
    as_of = datetime(2026, 8, 5, 11, 30, tzinfo=ZoneInfo("Asia/Ho_Chi_Minh"))

    result = pipeline._closed_market_sessions(df, as_of)

    assert result["date"].dt.strftime("%Y-%m-%d").tolist() == ["2026-08-04"]


def test_closed_sessions_accepts_today_after_market_close() -> None:
    dates = pd.to_datetime(["2026-08-04", "2026-08-05"])
    df = pd.DataFrame({"date": dates, "close": [100.0, 101.0]})
    as_of = datetime(2026, 8, 5, 16, 0, tzinfo=ZoneInfo("Asia/Ho_Chi_Minh"))

    result = pipeline._closed_market_sessions(df, as_of)

    assert result["date"].dt.strftime("%Y-%m-%d").tolist() == ["2026-08-04", "2026-08-05"]


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
