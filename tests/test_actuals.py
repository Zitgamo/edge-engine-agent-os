from __future__ import annotations

import pandas as pd

from src import actuals, database
from src.config import Config


def _prices(ticker: str, dates: pd.DatetimeIndex, base: float) -> pd.DataFrame:
    close = [base + i for i in range(len(dates))]
    return pd.DataFrame({
        "ticker": ticker,
        "date": dates,
        "open": close,
        "high": [value + 1 for value in close],
        "low": [value - 1 for value in close],
        "close": close,
        "volume": 1_000_000,
    })


def test_calculate_actuals_uses_persisted_raw_prices(tmp_path) -> None:
    dates = pd.date_range("2026-01-01", periods=25, freq="B")
    _prices("AAA", dates, 100).to_parquet(tmp_path / "AAA_raw.parquet")
    _prices("VNINDEX", dates, 1_000).to_parquet(tmp_path / "VNINDEX_raw.parquet")

    config = Config()
    config.raw_data_dir = tmp_path
    signals = pd.DataFrame([{
        "signal_date": "2026-01-01",
        "ticker": "AAA",
        "stop_loss": 0.0,
        "take_profit": 0.0,
    }])

    result = actuals.calculate_actuals(signals, holding_period=20, config=config)

    assert len(result) == 1
    assert result.loc[0, "realized_date"] == "2026-01-29"
    assert result.loc[0, "entry_date"] == "2026-01-02"
    assert result.loc[0, "actual_excess_return_20d"] == result.loc[0, "actual_excess_return"]
    assert result.loc[0, "actual_stock_return"] > 0
    assert result.loc[0, "benchmark_return"] > 0
    assert result.loc[0, "gross_stock_return"] >= result.loc[0, "actual_stock_return"]
    assert result.loc[0, "transaction_cost"] == config.round_trip_cost
    assert result.loc[0, "actual_outperform"] == 1


def test_actuals_table_upsert_is_idempotent(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(database, "DB_PATH", tmp_path / "engine.db")
    database.init_db()
    row = pd.DataFrame([{
        "signal_date": "2026-01-01",
        "ticker": "AAA",
        "actual_excess_return_5d": 0.1,
        "actual_outperform": 1,
        "realized_date": "2026-01-29",
    }])

    database.update_actuals(row)
    database.update_actuals(row)
    conn = database.get_conn()
    count = conn.execute("SELECT COUNT(*) FROM actuals").fetchone()[0]
    excess = conn.execute(
        "SELECT actual_excess_return_5d, actual_excess_return_20d FROM actuals"
    ).fetchone()
    conn.close()

    assert count == 1
    assert excess == (0.1, 0.1)


def test_legacy_actuals_are_migrated_to_canonical_t20_column(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(database, "DB_PATH", tmp_path / "engine.db")
    conn = database.get_conn()
    conn.execute(
        """CREATE TABLE actuals (
               id INTEGER PRIMARY KEY,
               signal_date DATE NOT NULL,
               ticker TEXT NOT NULL,
               actual_excess_return_5d REAL,
               actual_outperform INTEGER,
               realized_date DATE NOT NULL
           )"""
    )
    conn.execute(
        "INSERT INTO actuals VALUES (1, '2026-01-01', 'AAA', 0.25, 1, '2026-01-29')"
    )
    conn.commit()
    conn.close()

    database.init_db()

    conn = database.get_conn()
    migrated = conn.execute(
        "SELECT actual_excess_return_20d FROM actuals WHERE signal_date = '2026-01-01'"
    ).fetchone()[0]
    conn.close()

    assert migrated == 0.25


def test_execution_excess_prefers_t20_and_falls_back_to_legacy() -> None:
    values = pd.DataFrame([
        {"actual_excess_return_20d": 0.2, "actual_excess_return_5d": 0.1},
        {"actual_excess_return_20d": None, "actual_excess_return_5d": -0.1},
    ])

    result = actuals.add_execution_excess_column(values)

    assert result["execution_excess_return"].tolist() == [0.2, -0.1]


def test_actuals_refreshes_persisted_prices_without_full_horizon(
    monkeypatch,
    tmp_path,
) -> None:
    dates = pd.date_range("2026-01-01", periods=25, freq="B")
    full_prices = {
        ticker: _prices(ticker, dates, base)
        for ticker, base in (("AAA", 100), ("VNINDEX", 1_000))
    }
    for ticker, frame in full_prices.items():
        frame.head(5).to_parquet(tmp_path / f"{ticker}_raw.parquet")

    calls: list[str] = []

    class FakeCollector:
        def __init__(self, config):
            self.config = config

        def fetch(self, ticker: str, days: int):
            calls.append(ticker)
            return full_prices[ticker]

    monkeypatch.setattr(actuals, "OHLCVCollector", FakeCollector)
    config = Config()
    config.raw_data_dir = tmp_path

    result = actuals.calculate_actuals(
        pd.DataFrame([{
            "signal_date": "2026-01-01",
            "ticker": "AAA",
            "stop_loss": 0.0,
            "take_profit": 0.0,
        }]),
        holding_period=20,
        config=config,
    )

    assert len(result) == 1
    assert set(calls) == {"AAA", "VNINDEX"}
    assert result.loc[0, "realized_date"] == "2026-01-29"


def test_actuals_merges_recent_refresh_with_persisted_early_exit_data(
    monkeypatch,
    tmp_path,
) -> None:
    persisted_dates = pd.date_range("2026-01-01", periods=4, freq="B")
    fresh_dates = pd.date_range("2026-01-12", periods=2, freq="B")
    persisted_prices = {
        ticker: _prices(ticker, persisted_dates, base)
        for ticker, base in (("AAA", 100), ("VNINDEX", 1_000))
    }
    fresh_prices = {
        ticker: _prices(ticker, fresh_dates, base)
        for ticker, base in (("AAA", 100), ("VNINDEX", 1_000))
    }
    for ticker, frame in persisted_prices.items():
        frame.to_parquet(tmp_path / f"{ticker}_raw.parquet")

    class FakeCollector:
        def __init__(self, config):
            self.config = config

        def fetch(self, ticker: str, days: int):
            return fresh_prices[ticker]

    monkeypatch.setattr(actuals, "OHLCVCollector", FakeCollector)
    config = Config()
    config.raw_data_dir = tmp_path

    result = actuals.calculate_actuals(
        pd.DataFrame([{
            "signal_date": "2026-01-01",
            "ticker": "AAA",
            "stop_loss": 0.0,
            "take_profit": 0.02,
        }]),
        holding_period=20,
        config=config,
    )

    assert len(result) == 1
    assert result.loc[0, "status"] == "HIT_TP"
    assert result.loc[0, "realized_date"] == "2026-01-06"
