from __future__ import annotations

import glob
import logging
import os
from datetime import date
from typing import Any

import numpy as np
import pandas as pd

from src.config import Config

log = logging.getLogger(__name__)

# Session-level cache: avoids re-fetching the same ticker from yfinance (rate-limit protection)
_stock_cache: dict[tuple[str, str], pd.DataFrame | None] = {}


def _normalize_stock_data(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize market data into the schema expected by the tracker."""
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])
    if df["date"].dt.tz is not None:
        df["date"] = df["date"].dt.tz_localize(None)
    df["date"] = df["date"].dt.normalize()
    return df.sort_values("date").reset_index(drop=True)


def _fetch_yfinance(ticker: str) -> pd.DataFrame | None:
    """Fetch recent OHLCV data from Yahoo Finance."""
    import yfinance as yf

    try:
        t = yf.Ticker(f"{ticker}.VN")
        df = t.history(period="6mo")
        if df.empty:
            return None
        df = df.reset_index()
        df["ticker"] = ticker
        df = df.rename(columns={
            "Date": "date", "Open": "open", "High": "high",
            "Low": "low", "Close": "close", "Volume": "volume",
        })
        return _normalize_stock_data(df)
    except Exception as e:
        log.warning("yfinance fallback failed for %s: %s", ticker, e)
        return None


def load_stock_data(ticker: str, data_dir: str | None = None) -> pd.DataFrame | None:
    """Load raw OHLCV data for a ticker — parquet first, yfinance fallback (cached)."""
    if data_dir is None:
        data_dir = str(Config.raw_data_dir)
    cache_key = (str(data_dir), ticker)
    if cache_key in _stock_cache:
        return _stock_cache[cache_key]

    pattern = os.path.join(data_dir, f"{ticker}_raw.parquet")
    parquet_df: pd.DataFrame | None = None
    files = glob.glob(pattern)
    if files:
        try:
            parquet_df = _normalize_stock_data(pd.read_parquet(files[0]))
            latest_date = parquet_df["date"].max()
            if latest_date >= pd.Timestamp(date.today()):
                _stock_cache[cache_key] = parquet_df
                return parquet_df
            log.info(
                "Refreshing stale parquet for %s (latest=%s)",
                ticker,
                latest_date.date(),
            )
        except Exception as e:
            log.warning("Cannot load parquet %s: %s", ticker, e)

    fresh_df = _fetch_yfinance(ticker)
    if fresh_df is not None and not fresh_df.empty:
        _stock_cache[cache_key] = fresh_df
        return fresh_df

    if parquet_df is not None and not parquet_df.empty:
        log.warning("Using stale parquet for %s because fresh data is unavailable", ticker)
        _stock_cache[cache_key] = parquet_df
        return parquet_df

    _stock_cache[cache_key] = None
    return None


TRADING_DAYS_PER_YEAR = 252


def simulate_holding(
    df: pd.DataFrame,
    signal_date: str,
    entry_price: float,
    stop_loss: float,
    take_profit: float,
    holding_period: int = 20,
    settlement_delay: int = 2,
) -> dict[str, Any]:
    """Simulate the holding period for a signal (VN market rules).

    VN market uses T+2 settlement: shares arrive 2 trading days after buy.
    SL/TP only executable from settlement_delay onwards.

    Returns:
    - status: 'ACTIVE' | 'HIT_SL' | 'HIT_TP' | 'EXPIRED' | 'NO_DATA'
    - exit_price: price at exit (limit price for SL/TP hits, or current close if active)
    - exit_date: when it exited
    - pnl: P&L percentage
    - days_held: number of trading days held
    - high_during_hold: max price during holding
    - low_during_hold: min price during holding
    """
    # SL checked before TP (conservative on wide-range days), exit at limit price not close
    dates = df["date"].dt.strftime("%Y-%m-%d").tolist() if hasattr(df["date"], "dt") else list(df["date"].astype(str))

    if signal_date not in dates:
        return {"status": "NO_DATA", "pnl": 0.0, "days_held": 0}

    idx = dates.index(signal_date)
    if idx + 1 >= len(dates):
        return {"status": "PENDING", "pnl": 0.0, "days_held": 0}

    df_after = df.iloc[idx + 1:].reset_index(drop=True)
    sl_price = entry_price * (1 + stop_loss)
    tp_price = entry_price * (1 + take_profit)

    days_held = 0
    exit_price = None
    exit_date = None
    status = "SETTLING"

    max_high = entry_price
    min_low = entry_price

    for i in range(min(holding_period, len(df_after))):
        row = df_after.iloc[i]
        days_held = i + 1
        high = float(row["high"])
        low = float(row["low"])
        close = float(row["close"])
        current_date = str(row["date"])[:10] if hasattr(row["date"], "strftime") else str(row["date"])[:10]

        max_high = max(max_high, high)
        min_low = min(min_low, low)

        # VN T+2: shares arrive after settlement_delay, can only sell from then
        if days_held < settlement_delay:
            exit_price = close
            exit_date = current_date
            continue

        status = "ACTIVE"

        # Check SL first (conservative: assume worst case on wide-range days)
        if low <= sl_price:
            exit_price = sl_price
            exit_date = current_date
            status = "HIT_SL"
            break

        # Check TP (only if SL not hit)
        if high >= tp_price:
            exit_price = tp_price
            exit_date = current_date
            status = "HIT_TP"
            break

        # Still holding, update current price
        exit_price = close
        exit_date = current_date

    if status == "ACTIVE" and days_held >= holding_period:
        status = "EXPIRED"

    pnl = (exit_price - entry_price) / entry_price if exit_price else 0.0

    return {
        "status": status,
        "entry_price": entry_price,
        "exit_price": exit_price,
        "exit_date": exit_date,
        "pnl": round(pnl, 4),
        "days_held": days_held,
        "high_during_hold": round(max_high, 2),
        "low_during_hold": round(min_low, 2),
        "settlement_delay": settlement_delay,
    }


def track_signal(
    ticker: str,
    signal_date: str,
    entry_price: float | None = None,
    stop_loss: float = -0.03,
    take_profit: float = 0.08,
    holding_period: int = 20,
    settlement_delay: int = 2,
    data_dir: str | None = None,
) -> dict[str, Any]:
    """Track a single signal's realtime P&L.

    VN T+2: settlement_delay=2 → only execute SL/TP from 3rd trading day.
    If entry_price is None, it's inferred from the close on signal_date.
    """
    df = load_stock_data(ticker, data_dir)
    if df is None:
        return {"ticker": ticker, "signal_date": signal_date, "status": "NO_DATA", "pnl": 0.0, "days_held": 0}

    if entry_price is None:
        dates_list = df["date"].dt.strftime("%Y-%m-%d").tolist() if hasattr(df["date"], "dt") else list(df["date"].astype(str))
        if signal_date not in dates_list:
            return {"ticker": ticker, "signal_date": signal_date, "status": "NO_DATA", "pnl": 0.0, "days_held": 0}
        idx = dates_list.index(signal_date)
        entry_price = float(df.iloc[idx]["close"])

    result = simulate_holding(df, signal_date, entry_price, stop_loss, take_profit, holding_period, settlement_delay)
    result["ticker"] = ticker
    result["signal_date"] = signal_date
    result["entry_price"] = round(entry_price, 2)
    return result


def track_signals(
    signals: list[dict[str, Any]],
    holding_period: int = 20,
    data_dir: str | None = None,
) -> list[dict[str, Any]]:
    """Track multiple signals and return their current P&L status."""
    results = []
    for sig in signals:
        ticker = sig.get("ticker", "")
        signal_date = str(sig.get("signal_date", ""))[:10]
        sl = float(sig.get("stop_loss", -0.03))
        tp = float(sig.get("take_profit", 0.08))
        result = track_signal(ticker, signal_date, stop_loss=sl, take_profit=tp, holding_period=holding_period, data_dir=data_dir)
        results.append(result)
    return results


def get_signal_summary(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate tracking results into a portfolio-level summary.

    Equal-weight allocation: each signal gets equal share of capital,
    so daily portfolio return = average return across all signals that day.
    Cumulative portfolio P&L = product of (1 + daily_return) across dates.
    """
    total = len(results)
    settling = sum(1 for r in results if r["status"] == "SETTLING")
    hit_tp = sum(1 for r in results if r["status"] == "HIT_TP")
    hit_sl = sum(1 for r in results if r["status"] == "HIT_SL")
    active = sum(1 for r in results if r["status"] == "ACTIVE")
    expired = sum(1 for r in results if r["status"] == "EXPIRED")

    # Group by signal_date → equal-weight portfolio return
    by_date: dict[str, list[float]] = {}
    for r in results:
        d = str(r.get("signal_date", ""))[:10]
        by_date.setdefault(d, []).append(r["pnl"])

    dates_sorted = sorted(by_date.keys())
    daily_returns = [np.mean(by_date[d]) for d in dates_sorted]
    portfolio_pnl = np.prod([1 + dr for dr in daily_returns]) - 1 if daily_returns else 0.0
    avg_pnl = np.mean([r["pnl"] for r in results]) if results else 0.0

    return {
        "total": total,
        "settling": settling,
        "hit_tp": hit_tp,
        "hit_sl": hit_sl,
        "active": active,
        "expired": expired,
        "win_rate": hit_tp / (hit_tp + hit_sl) * 100 if (hit_tp + hit_sl) > 0 else 0.0,
        "avg_pnl": round(avg_pnl, 4),
        "portfolio_pnl": round(portfolio_pnl, 4),
        "total_pnl": round(portfolio_pnl, 4),
    }
