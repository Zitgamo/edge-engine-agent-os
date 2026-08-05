from __future__ import annotations

import pandas as pd

from src.accumulation import simulate_dca, simulate_value_dca


def test_dca_maps_calendar_targets_to_trading_sessions() -> None:
    prices = pd.DataFrame({
        "date": pd.bdate_range("2026-01-01", periods=260),
        "close": 100.0,
    })

    result = simulate_dca(prices, monthly_amount=100.0, frequency="monthly")

    assert result["total_invested"].iloc[-1] == 1_200.0


def test_value_dca_does_not_claim_snapshot_fundamentals_are_historical() -> None:
    prices = pd.DataFrame({
        "date": pd.bdate_range("2026-01-01", periods=40),
        "close": 100.0,
    })
    snapshot = pd.DataFrame([{
        "date": pd.Timestamp("2026-01-01"),
        "pe_ratio": 10.0,
        "pb_ratio": 1.0,
        "pe_ratio_pct": 0.5,
        "pb_ratio_pct": 0.5,
    }])

    result = simulate_value_dca(prices, snapshot, monthly_amount=100.0)

    assert result.empty
