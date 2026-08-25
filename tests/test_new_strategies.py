from __future__ import annotations

import pandas as pd

from src.strategies.breakout_volatility import BreakoutVolatilityStrategy
from src.strategies.manager import StrategyManager
from src.strategies.trend_following import TrendFollowingStrategy


def _create_sample_market_data() -> pd.DataFrame:
    dates = pd.date_range("2026-01-01", periods=10, freq="B")
    tickers = ["FPT", "MWG", "HPG", "SSI", "VCB"]
    rows = []
    for d in dates:
        for i, t in enumerate(tickers):
            rows.append({
                "date": d,
                "ticker": t,
                "open": 50.0 + i * 10,
                "high": 55.0 + i * 10,
                "low": 49.0 + i * 10,
                "close": 54.0 + i * 10,
                "volume": 1_000_000 * (i + 1),
                "return_5d": 0.05 * (i + 1),
                "return_20d": 0.10 * (i + 1),
                "return_60d": 0.20 * (i + 1),
                "rs_5d": 60.0 + i * 5,
                "rs_20d": 65.0 + i * 5,
                "rs_60d": 70.0 + i * 5,
                "volume_surge": 1.5 + i * 0.2,
                "atr_14": 1.2 + i * 0.1,
            })
    return pd.DataFrame(rows)


def test_trend_following_strategy_ranks_correctly() -> None:
    df = _create_sample_market_data()
    strat = TrendFollowingStrategy()
    ranked = strat.rank(df)

    assert not ranked.empty
    assert len(ranked) == 5
    assert list(ranked.columns) == ["ticker", "date", "score", "ensemble_score", "rank"]
    assert ranked["rank"].tolist() == [1, 2, 3, 4, 5]
    assert ranked.iloc[0]["ticker"] == "VCB"  # highest momentum and RS


def test_breakout_volatility_strategy_ranks_correctly() -> None:
    df = _create_sample_market_data()
    strat = BreakoutVolatilityStrategy()
    ranked = strat.rank(df)

    # Ten rows are intentionally too short to establish a prior 20-session
    # high; the confirmed-breakout strategy must fail closed.
    assert ranked.empty
    assert list(ranked.columns) == ["ticker", "date", "score", "ensemble_score", "rank"]


def test_strategy_manager_includes_new_strategies() -> None:
    mgr = StrategyManager(include_research=True)
    names = [s.name for s in mgr.strategies]
    assert "trend_following" in names
    assert "breakout_volatility" in names


def test_strategy_manager_keeps_research_modules_out_of_default_ensemble() -> None:
    mgr = StrategyManager(include_research=False)
    names = [s.name for s in mgr.strategies]
    assert "trend_following" not in names
    assert "breakout_volatility" not in names
    assert "fundamental_value" not in names
