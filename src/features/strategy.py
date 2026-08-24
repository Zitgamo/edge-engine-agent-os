"""Point-in-time features used by rule-based strategy ranking."""

from __future__ import annotations

import numpy as np
import pandas as pd


def _group_transform(
    frame: pd.DataFrame,
    column: str,
    function,
) -> pd.Series:
    if "ticker" in frame.columns:
        return frame.groupby("ticker", sort=False)[column].transform(function)
    return frame[column].transform(function)


def add_strategy_features(
    df: pd.DataFrame,
    *,
    breakout_window: int = 20,
) -> pd.DataFrame:
    """Add EMA, prior-high breakout and candle/ATR confirmation features.

    All rolling values are computed from rows strictly before the current
    breakout comparison where appropriate. The function restores the input
    row order and never uses a future row from another ticker.
    """
    if df.empty:
        return df.copy()
    required = {"date", "close"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing strategy-feature columns: {missing}")

    result = df.copy()
    result["_strategy_original_order"] = np.arange(len(result))
    result["_strategy_date"] = pd.to_datetime(result["date"], errors="coerce")
    sort_cols = ["_strategy_date"]
    if "ticker" in result.columns:
        sort_cols.insert(0, "ticker")
    result = result.sort_values(sort_cols, kind="mergesort")

    for span in (20, 60):
        result[f"ema_{span}"] = _group_transform(
            result,
            "close",
            lambda values, span=span: values.ewm(
                span=span,
                adjust=False,
                min_periods=1,
            ).mean(),
        )

    if "high" in result.columns:
        result["prior_high_20d"] = _group_transform(
            result,
            "high",
            lambda values: values.rolling(
                breakout_window,
                min_periods=breakout_window,
            ).max().shift(1),
        )
        result["breakout_20d"] = result["close"] / result["prior_high_20d"] - 1.0
    else:
        result["prior_high_20d"] = np.nan
        result["breakout_20d"] = np.nan

    if {"high", "low"} <= set(result.columns):
        span = (result["high"] - result["low"]).replace(0, np.nan)
        result["close_position"] = ((result["close"] - result["low"]) / span).clip(0.0, 1.0)
    else:
        result["close_position"] = np.nan

    if "atr_pct" in result.columns:
        result["atr_pct_ma20"] = _group_transform(
            result,
            "atr_pct",
            lambda values: values.rolling(20, min_periods=5).mean(),
        )
        result["atr_expansion"] = (
            result["atr_pct"] / result["atr_pct_ma20"].replace(0, np.nan)
        )
    else:
        result["atr_pct_ma20"] = np.nan
        result["atr_expansion"] = np.nan

    return (
        result.sort_values("_strategy_original_order", kind="mergesort")
        .drop(columns=["_strategy_original_order", "_strategy_date"])
        .reset_index(drop=True)
    )
