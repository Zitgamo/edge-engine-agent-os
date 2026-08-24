from __future__ import annotations

import numpy as np
import pandas as pd

from src.features.strategy import add_strategy_features
from src.strategies.breakout_volatility import BreakoutVolatilityStrategy
from src.strategies.fundamental_value import FundamentalValueStrategy
from src.strategies.trend_following import TrendFollowingStrategy


def _history() -> pd.DataFrame:
    dates = pd.bdate_range("2026-01-01", periods=25)
    rows = []
    for ticker in ("GOOD", "WEAK"):
        for i, date in enumerate(dates):
            close = 100.0 + i if ticker == "GOOD" else 100.0
            high = close + 1.0
            low = close - 1.0
            volume_surge = 2.0 if ticker == "GOOD" else 1.7
            if i == len(dates) - 1 and ticker == "GOOD":
                close, high, low = 126.0, 127.0, 95.0
            rows.append({
                "date": date,
                "ticker": ticker,
                "open": close,
                "high": high,
                "low": low,
                "close": close,
                "volume": 1_000_000,
                "volume_surge": volume_surge,
                "return_20d": 0.25 if ticker == "GOOD" else 0.01,
                "return_60d": 0.35 if ticker == "GOOD" else 0.01,
                "rs_20d": 0.20 if ticker == "GOOD" else 0.01,
                "rs_60d": 0.30 if ticker == "GOOD" else 0.01,
                "atr_pct": 0.03 if ticker == "GOOD" else 0.05,
            })
    return pd.DataFrame(rows)


def test_strategy_features_are_point_in_time_and_grouped_by_ticker() -> None:
    result = add_strategy_features(_history())
    latest = result[result["date"] == result["date"].max()].set_index("ticker")

    assert {"ema_20", "ema_60", "prior_high_20d", "breakout_20d", "close_position"} <= set(result)
    assert latest.loc["GOOD", "prior_high_20d"] < latest.loc["GOOD", "close"]
    assert latest.loc["GOOD", "breakout_20d"] > 0
    assert latest.loc["WEAK", "breakout_20d"] <= 0
    assert np.isfinite(latest.loc["GOOD", "ema_20"])
    assert np.isfinite(latest.loc["GOOD", "ema_60"])


def test_breakout_strategy_requires_breakout_volume_and_close_confirmation() -> None:
    ranked = BreakoutVolatilityStrategy().rank(_history())

    assert ranked["ticker"].tolist() == ["GOOD"]
    assert ranked.iloc[0]["rank"] == 1


def test_trend_strategy_uses_actual_ema_and_atr_features() -> None:
    ranked = TrendFollowingStrategy().rank(_history())

    assert ranked.iloc[0]["ticker"] == "GOOD"
    assert ranked.iloc[0]["score"] > ranked.iloc[-1]["score"]


def test_fundamental_strategy_fails_closed_when_snapshot_is_missing() -> None:
    frame = pd.DataFrame({
        "date": [pd.Timestamp("2026-08-05")] * 3,
        "ticker": ["AAA", "BBB", "CCC"],
        "pe_ratio": [np.nan] * 3,
        "pb_ratio": [np.nan] * 3,
        "roe": [np.nan] * 3,
        "profit_margin": [np.nan] * 3,
    })

    assert FundamentalValueStrategy().rank(frame).empty


def test_fundamental_strategy_accepts_only_snapshot_as_of_signal_date() -> None:
    frame = pd.DataFrame({
        "date": [pd.Timestamp("2026-08-05")] * 3,
        "ticker": ["AAA", "BBB", "CCC"],
        "fundamental_snapshot_date": [pd.Timestamp("2026-08-05")] * 3,
        "pe_ratio": [8.0, 12.0, 20.0],
        "pb_ratio": [1.0, 1.5, 2.0],
        "roe": [0.20, 0.12, 0.05],
        "profit_margin": [0.20, 0.10, 0.03],
    })

    ranked = FundamentalValueStrategy().rank(frame)

    assert ranked.iloc[0]["ticker"] == "AAA"
