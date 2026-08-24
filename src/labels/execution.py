"""Execution-aligned forward labels for supervised ranking."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.backtest import _load_backtest_prices, _price_metadata


def _numeric_array(frame: pd.DataFrame, column: str, fallback: str) -> np.ndarray:
    source = frame[column] if column in frame.columns else frame[fallback]
    return pd.to_numeric(source, errors="coerce").to_numpy(dtype=float)


def _execution_outcome_map(
    tickers: set[str],
    *,
    raw_data_dir: str | Path,
    stop_loss: float,
    take_profit: float,
    holding_period: int,
    round_trip_cost: float,
) -> dict[tuple[str, str], tuple[float, str]]:
    """Precompute the same execution outcome once per ticker/session."""
    cache: dict[str, pd.DataFrame] = {}
    benchmark = _load_backtest_prices("VNINDEX", cache, raw_data_dir)
    if benchmark.empty:
        return {}
    benchmark_dates, benchmark_date_to_idx = _price_metadata(benchmark)
    benchmark_close = _numeric_array(benchmark, "close", "close")
    benchmark_entry = _numeric_array(benchmark, "open", "close")

    outcomes: dict[tuple[str, str], tuple[float, str]] = {}
    for ticker in sorted(tickers):
        stock = _load_backtest_prices(ticker, cache, raw_data_dir)
        if stock.empty:
            continue
        stock_dates, stock_date_to_idx = _price_metadata(stock)
        close = _numeric_array(stock, "close", "close")
        open_prices = _numeric_array(stock, "open", "close")
        high = _numeric_array(stock, "high", "close")
        low = _numeric_array(stock, "low", "close")

        for signal_date, signal_idx in stock_date_to_idx.items():
            entry_idx = signal_idx + 1
            end_idx = entry_idx + holding_period
            if end_idx > len(stock):
                continue
            entry = open_prices[entry_idx]
            if not np.isfinite(entry) or entry <= 0:
                entry = close[entry_idx]
            if not np.isfinite(entry) or entry <= 0:
                continue

            exit_idx = end_idx - 1
            exit_price = close[exit_idx]
            for held_idx in range(entry_idx, end_idx):
                held = held_idx - entry_idx + 1
                if held <= 2:
                    continue
                if stop_loss < 0 and low[held_idx] <= entry * (1 + stop_loss):
                    exit_idx = held_idx
                    exit_price = entry * (1 + stop_loss)
                    break
                if take_profit > 0 and high[held_idx] >= entry * (1 + take_profit):
                    exit_idx = held_idx
                    exit_price = entry * (1 + take_profit)
                    break
                exit_idx = held_idx
                exit_price = close[held_idx]

            exit_date = stock_dates[exit_idx]
            benchmark_entry_idx = benchmark_date_to_idx.get(stock_dates[entry_idx])
            benchmark_exit_idx = benchmark_date_to_idx.get(exit_date)
            if benchmark_entry_idx is None or benchmark_exit_idx is None:
                continue
            bm_entry = benchmark_entry[benchmark_entry_idx]
            if not np.isfinite(bm_entry) or bm_entry == 0:
                continue
            if not np.isfinite(exit_price) or not np.isfinite(benchmark_close[benchmark_exit_idx]):
                continue
            stock_return = (exit_price - entry) / entry - round_trip_cost
            benchmark_return = (
                benchmark_close[benchmark_exit_idx] - bm_entry
            ) / bm_entry
            outcomes[(ticker, signal_date)] = (
                float(stock_return - benchmark_return),
                stock_dates[end_idx - 1],
            )
    return outcomes


def add_execution_labels(
    df: pd.DataFrame,
    *,
    raw_data_dir: str | Path = "data/raw",
    stop_loss: float = -0.03,
    take_profit: float = 0.08,
    holding_period: int = 20,
    round_trip_cost: float = 0.0,
) -> pd.DataFrame:
    """Add next-open, settlement-delay and SL/TP labels to a panel.

    The return follows the exact mechanics used by the backtest and actuals
    tracker. ``execution_label_end_date`` deliberately uses the full
    holding-period endpoint, even if a trade exits early, so purged splits
    remain conservative.
    """
    if holding_period <= 0:
        raise ValueError("holding_period must be positive")
    required = {"date", "ticker"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing execution-label columns: {missing}")

    result = df.copy()
    result["date"] = pd.to_datetime(result["date"], errors="coerce").dt.normalize()
    target_col = f"execution_outperform_{holding_period}d"
    return_col = f"execution_excess_return_{holding_period}d"
    end_col = f"execution_label_end_date_{holding_period}d"
    label_cols = [target_col, return_col, end_col]

    keys = (
        result[["date", "ticker"]]
        .dropna(subset=["date", "ticker"])
        .drop_duplicates(["date", "ticker"])
        .sort_values(["date", "ticker"])
    )
    outcomes = _execution_outcome_map(
        set(keys["ticker"].astype(str)),
        raw_data_dir=raw_data_dir,
        stop_loss=stop_loss,
        take_profit=take_profit,
        holding_period=holding_period,
        round_trip_cost=round_trip_cost,
    )
    labels: list[dict[str, object]] = []
    for row in keys.itertuples(index=False):
        signal_date = pd.Timestamp(row.date).date().isoformat()
        ticker = str(row.ticker)
        outcome = outcomes.get((ticker, signal_date))
        valid = outcome is not None
        excess_return = outcome[0] if valid else np.nan
        end_date = outcome[1] if valid else None
        labels.append({
            "date": row.date,
            "ticker": ticker,
            target_col: float(float(excess_return) > 0) if valid else np.nan,
            return_col: float(excess_return) if valid else np.nan,
            end_col: pd.Timestamp(end_date) if valid else pd.NaT,
        })

    label_frame = pd.DataFrame(labels, columns=["date", "ticker", *label_cols])
    result = result.drop(columns=label_cols, errors="ignore")
    return result.merge(label_frame, on=["date", "ticker"], how="left")
