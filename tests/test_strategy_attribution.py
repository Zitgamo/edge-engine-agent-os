from __future__ import annotations

import pandas as pd
import pytest

from src.research.strategy_attribution import summarize_strategy_attribution


def test_strategy_attribution_deduplicates_trades_and_applies_cost() -> None:
    frame = pd.DataFrame([
        # Duplicate persistence row must not count as a second trade.
        {"strategy_name": "trend_following", "signal_date": "2026-01-01", "ticker": "AAA", "actual_excess_return_20d": 0.04, "actual_outperform": 1},
        {"strategy_name": "trend_following", "signal_date": "2026-01-01", "ticker": "AAA", "actual_excess_return_20d": 0.04, "actual_outperform": 1},
        {"strategy_name": "trend_following", "signal_date": "2026-01-01", "ticker": "BBB", "actual_excess_return_20d": -0.02, "actual_outperform": 0},
        {"strategy_name": "trend_following", "signal_date": "2026-01-02", "ticker": "CCC", "actual_excess_return_20d": -0.01, "actual_outperform": 0},
        {"strategy_name": "breakout_volatility", "signal_date": "2026-01-01", "ticker": "AAA", "actual_excess_return_20d": 0.04, "actual_outperform": 1},
    ])

    result = summarize_strategy_attribution(frame, return_col="actual_excess_return_20d", round_trip_cost=0.003)
    trend = result[result["strategy_name"] == "trend_following"].iloc[0]

    assert trend["trade_count"] == 3
    assert trend["signal_dates"] == 2
    assert trend["basket_count"] == 2
    assert trend["win_rate"] == 1 / 3
    assert trend["avg_return_net"] == pytest.approx(
        (0.04 - 0.003 - 0.02 - 0.003 - 0.01 - 0.003) / 3
    )
    assert trend["positive_basket_rate"] == 0.5


def test_strategy_attribution_returns_empty_schema_for_no_realized_rows() -> None:
    result = summarize_strategy_attribution(pd.DataFrame())

    assert result.empty
    assert "strategy_name" in result.columns
    assert "trade_count" in result.columns
