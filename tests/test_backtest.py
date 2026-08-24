from __future__ import annotations

import pandas as pd

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
