from __future__ import annotations

import pandas as pd
import pytest

from src import database
from src.research.paper_test import load_paper_test_readiness
from src.research.paper_test import summarize_paper_test_readiness


def test_paper_test_counts_only_complete_deduplicated_baskets() -> None:
    frame = pd.DataFrame([
        {"strategy_name": "trend", "signal_date": "2026-01-01", "ticker": "AAA", "return_net": 0.02},
        {"strategy_name": "trend", "signal_date": "2026-01-01", "ticker": "AAA", "return_net": 0.02},
        {"strategy_name": "trend", "signal_date": "2026-01-01", "ticker": "BBB", "return_net": -0.01},
        {"strategy_name": "trend", "signal_date": "2026-01-01", "ticker": "CCC", "return_net": 0.03},
        {"strategy_name": "trend", "signal_date": "2026-01-02", "ticker": "AAA", "return_net": 0.10},
        {"strategy_name": "trend", "signal_date": "2026-01-02", "ticker": "BBB", "return_net": 0.10},
        {"strategy_name": "breakout", "signal_date": "2026-01-01", "ticker": "AAA", "return_net": None},
    ])

    result = summarize_paper_test_readiness(
        frame,
        min_picks=3,
        min_baskets=2,
        target_baskets=4,
    )

    trend = result[result["strategy_name"] == "trend"].iloc[0]
    breakout = result[result["strategy_name"] == "breakout"].iloc[0]
    assert trend["trade_count"] == 3
    assert trend["basket_count"] == 1
    assert trend["avg_return_net"] == pytest.approx((0.02 - 0.01 + 0.03) / 3)
    assert trend["baskets_to_minimum"] == 1
    assert trend["readiness"] == "collecting"
    assert breakout["basket_count"] == 0
    assert breakout["progress_to_target"] == 0.0


def test_paper_test_marks_strategy_ready_at_minimum_baskets() -> None:
    rows = []
    for day in ("2026-01-01", "2026-01-02"):
        for ticker, value in zip(("AAA", "BBB", "CCC"), (0.02, 0.01, -0.01)):
            rows.append({
                "strategy_name": "core",
                "signal_date": day,
                "ticker": ticker,
                "return_net": value,
            })

    result = summarize_paper_test_readiness(
        pd.DataFrame(rows),
        min_picks=3,
        min_baskets=2,
        target_baskets=4,
    )

    row = result.iloc[0]
    assert row["basket_count"] == 2
    assert row["readiness"] == "ready"
    assert row["progress_to_target"] == pytest.approx(0.5)


def test_paper_test_loader_recomputes_net_returns_from_raw_prices(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(database, "DB_PATH", tmp_path / "engine.db")
    dates = pd.date_range("2026-01-01", periods=25, freq="B")
    for ticker, base in (("AAA", 100.0), ("BBB", 110.0), ("CCC", 120.0), ("VNINDEX", 1000.0)):
        close = [base + i for i in range(len(dates))]
        pd.DataFrame({
            "date": dates,
            "open": close,
            "high": [value + 1 for value in close],
            "low": [value - 1 for value in close],
            "close": close,
        }).to_parquet(tmp_path / f"{ticker}_raw.parquet")

    database.init_db()
    conn = database.get_conn()
    rows = [
        ("core", "2026-01-01", ticker, rank, 1.0 / rank)
        for rank, ticker in enumerate(("AAA", "BBB", "CCC"), start=1)
    ]
    conn.executemany(
        """INSERT INTO strategy_performance
           (strategy_name, signal_date, ticker, rank, score)
           VALUES (?, ?, ?, ?, ?)""",
        rows + [rows[0]],
    )
    conn.commit()
    conn.close()

    result = load_paper_test_readiness(
        raw_data_dir=tmp_path,
        stop_loss=0.0,
        take_profit=0.0,
        round_trip_cost=0.003,
        min_baskets=1,
        target_baskets=2,
    )

    row = result.iloc[0]
    assert row["basket_count"] == 1
    assert row["trade_count"] == 3
    assert row["readiness"] == "ready"
    assert pd.notna(row["avg_return_net"])
    assert row["avg_return_net"] > 0.0
