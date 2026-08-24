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
