from __future__ import annotations

from datetime import datetime
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pytest

from src import database, pipeline, supabase_client
from src.model.schema import FEATURE_COLS
from src.pipeline import (
    _enforce_execution_horizon_quality,
    _execution_quality_metrics,
)


def _quality_config() -> SimpleNamespace:
    return SimpleNamespace(
        min_model_roc_auc=0.52,
        min_model_quality_dates=30,
        min_model_top3_excess_return=0.0,
        min_model_top3_spread=0.0,
        stop_loss=-0.03,
        take_profit=0.08,
        round_trip_cost=0.0,
        raw_data_dir="data/raw",
    )


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


def test_execution_quality_gate_allows_low_auc_when_trades_have_edge() -> None:
    config = _quality_config()

    _enforce_execution_horizon_quality(
        {
            "roc_auc": 0.51,
            "execution_evaluation_dates": 40,
            "execution_top3_excess_return": 0.01,
            "execution_top3_spread": 0.02,
        },
        horizon=20,
        config=config,
    )


def test_quality_gate_does_not_block_other_horizons() -> None:
    config = SimpleNamespace(min_model_roc_auc=0.52)

    _enforce_execution_horizon_quality(
        {"roc_auc": 0.40},
        horizon=5,
        config=config,
    )


def test_quality_gate_requires_execution_metrics() -> None:
    config = _quality_config()

    with pytest.raises(RuntimeError, match="execution quality metrics"):
        _enforce_execution_horizon_quality(
            {
                "roc_auc": 0.99,
                "evaluation_dates": 40,
                "top3_excess_return": 0.01,
                "top3_spread": 0.02,
            },
            horizon=20,
            config=config,
        )


def test_quality_gate_uses_execution_ranking_edge() -> None:
    config = _quality_config()

    _enforce_execution_horizon_quality(
        {
            "roc_auc": 0.48,
            "execution_evaluation_dates": 40,
            "execution_top3_excess_return": 0.01,
            "execution_top3_spread": 0.02,
        },
        horizon=20,
        config=config,
    )


def test_quality_gate_blocks_negative_execution_edge() -> None:
    config = _quality_config()

    with pytest.raises(RuntimeError, match="execution top-3 excess return"):
        _enforce_execution_horizon_quality(
            {
                "roc_auc": 0.55,
                "execution_evaluation_dates": 40,
                "execution_top3_excess_return": -0.01,
                "execution_top3_spread": 0.02,
            },
            horizon=20,
            config=config,
        )


def test_execution_quality_metrics_use_next_open_trade_results(monkeypatch) -> None:
    rows = []
    for signal_date in pd.to_datetime(["2026-01-01", "2026-01-02"]):
        for score, ticker in zip([0.9, 0.8, 0.1], ["AAA", "BBB", "CCC"]):
            rows.append({
                "date": signal_date,
                "ticker": ticker,
                **{
                    column: score if column == FEATURE_COLS[0] else 0.0
                    for column in FEATURE_COLS
                },
            })
    df = pd.DataFrame(rows)

    class FakeModel:
        def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
            score = features[FEATURE_COLS[0]].to_numpy()
            return np.column_stack([1.0 - score, score])

    returns = {
        ("2026-01-01", "AAA"): 0.03,
        ("2026-01-01", "BBB"): 0.02,
        ("2026-01-01", "CCC"): -0.02,
        ("2026-01-02", "AAA"): 0.04,
        ("2026-01-02", "BBB"): 0.01,
        ("2026-01-02", "CCC"): -0.03,
    }

    def fake_execution(ticker, signal_date, *args, **kwargs):
        return returns[(signal_date, ticker)]

    monkeypatch.setattr("src.backtest._sltp_excess_return", fake_execution)
    metrics = _execution_quality_metrics(
        FakeModel(),
        df,
        config=_quality_config(),
        holding_period=20,
        top_n=2,
    )

    assert metrics["execution_evaluation_dates"] == 2
    assert metrics["execution_top3_excess_return"] == pytest.approx(0.025)
    assert metrics["execution_top3_spread"] == pytest.approx(1 / 60)


def test_publish_no_trade_clears_local_and_cloud_publications(monkeypatch) -> None:
    events = []

    class FakeStorage:
        def save_processed(self, frame: pd.DataFrame, filename: str):
            events.append(("processed", filename, frame.empty))

    monkeypatch.setattr(
        pipeline,
        "save_pipeline_run",
        lambda metrics, status, run_key: events.append(
            ("run", metrics, status, run_key)
        ),
    )
    monkeypatch.setattr(
        database,
        "clear_publication_for_date",
        lambda signal_date: events.append(("local_clear", signal_date)),
    )
    monkeypatch.setattr(
        database,
        "backfill_actuals",
        lambda holding_period, config=None: events.append(
            ("backfill", holding_period, config)
        ),
    )
    monkeypatch.setattr(
        supabase_client,
        "clear_publication_for_date",
        lambda signal_date: events.append(("cloud_clear", signal_date)),
    )
    monkeypatch.setattr(
        supabase_client,
        "sync_all",
        lambda config=None: events.append(("sync", config)),
    )

    pipeline._publish_no_trade(
        FakeStorage(),
        "2026-08-18",
        {"roc_auc": 0.51},
        status="no_trade",
    )

    assert ("local_clear", "2026-08-18") in events
    assert ("cloud_clear", "2026-08-18") in events
    assert any(
        event[0] == "backfill" and event[1] == pipeline.HOLDING_PERIOD
        for event in events
    )
    assert any(event[0] == "sync" for event in events)
    assert {item[1] for item in events if item[0] == "processed"} == {
        "ranking.parquet",
        "signal.parquet",
    }
    assert all(item[2] for item in events if item[0] == "processed")
