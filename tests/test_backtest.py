from __future__ import annotations

import numpy as np
import pandas as pd

from src import backtest, database
from src.model.schema import FEATURE_COLS, TARGET_COL
from src.backtest import _sltp_excess_return


def test_sltp_backtest_uses_next_open_benchmark_entry() -> None:
    dates = pd.date_range("2026-01-01", periods=4, freq="B")
    stock = pd.DataFrame({
        "date": dates,
        "open": [100.0, 110.0, 120.0, 130.0],
        "high": [101.0, 111.0, 121.0, 131.0],
        "low": [99.0, 109.0, 119.0, 129.0],
        "close": [100.0, 115.0, 125.0, 130.0],
    })
    benchmark = pd.DataFrame({
        "date": dates,
        "open": [1000.0, 2000.0, 2100.0, 2200.0],
        "close": [1000.0, 1900.0, 2000.0, 2100.0],
    })
    cache = {"AAA": stock, "VNINDEX": benchmark}

    result = _sltp_excess_return(
        "AAA",
        "2026-01-01",
        stop_loss=0.0,
        take_profit=0.0,
        holding_period=3,
        prices_cache=cache,
    )

    expected_stock_return = (130.0 - 110.0) / 110.0
    expected_benchmark_return = (2100.0 - 2000.0) / 2000.0
    assert result == expected_stock_return - expected_benchmark_return


def test_sltp_backtest_fills_a_gap_at_the_session_open() -> None:
    dates = pd.date_range("2026-01-01", periods=5, freq="B")
    stock = pd.DataFrame({
        "date": dates,
        "open": [100.0, 100.0, 100.0, 90.0, 100.0],
        "high": [101.0, 101.0, 101.0, 95.0, 101.0],
        "low": [99.0, 99.0, 99.0, 80.0, 99.0],
        "close": [100.0, 100.0, 100.0, 90.0, 100.0],
    })
    benchmark = pd.DataFrame({
        "date": dates,
        "open": [1000.0] * 5,
        "close": [1000.0] * 5,
    })

    result = _sltp_excess_return(
        "AAA",
        "2026-01-01",
        stop_loss=-0.05,
        take_profit=0.10,
        holding_period=4,
        prices_cache={"AAA": stock, "VNINDEX": benchmark},
    )

    assert result == -0.10


def test_auto_retrain_saves_candidate_to_its_horizon_path(monkeypatch, tmp_path) -> None:
    dates = pd.date_range("2026-01-01", periods=30, freq="B")
    features = pd.DataFrame({column: 0.0 for column in FEATURE_COLS}, index=dates)
    features["date"] = dates
    features[TARGET_COL] = np.arange(len(dates)) % 2
    features["label_end_date_5d"] = dates + pd.Timedelta(days=1)

    class FakeConfig:
        model_training_days = 180
        model_path = tmp_path / "xgboost_model_h20.json"

        def model_path_for_horizon(self, horizon: int):
            return tmp_path / f"xgboost_model_h{horizon}.json"

    saved_paths: list[str] = []

    class FakeModel:
        def fit(self, X, y, verbose=False):
            return None

        def predict(self, X):
            return np.arange(len(X)) % 2

        def save_model(self, path):
            saved_paths.append(path)

    monkeypatch.setattr(database, "DB_PATH", tmp_path / "engine.db")
    monkeypatch.setattr(backtest, "Config", FakeConfig)
    monkeypatch.setattr(backtest.pd, "read_parquet", lambda path: features)
    monkeypatch.setattr(backtest.xgb, "XGBClassifier", lambda **kwargs: FakeModel())

    result = backtest.auto_retrain()

    assert result["improvement"] > 0.01
    assert saved_paths == [str(tmp_path / "xgboost_model_h5.json")]
