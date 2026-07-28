from __future__ import annotations

import glob
import logging
import os
from datetime import date, datetime
from typing import Any

import numpy as np
import pandas as pd

from src.config import Config

log = logging.getLogger(__name__)


def load_stock_data(ticker: str, data_dir: str | None = None) -> pd.DataFrame | None:
    """Load raw OHLCV data for a ticker — tries parquet first, then yfinance fallback."""
    if data_dir is None:
        data_dir = str(Config.raw_data_dir)
    pattern = os.path.join(data_dir, f"{ticker}_raw.parquet")
    files = glob.glob(pattern)
    if files:
        try:
            df = pd.read_parquet(files[0])
            df = df.sort_values("date").reset_index(drop=True)
            return df
        except Exception as e:
            log.warning("Cannot load parquet %s: %s", ticker, e)

    # Fallback: fetch from yfinance
    try:
        import yfinance as yf
        stock = yf.Ticker(f"{ticker}.VN" if not ticker.startswith("VN") else ticker)
        df = stock.history(period="1y")
        if df.empty:
            return None
        df = df.reset_index()
        df.columns = [c.lower().replace(" ", "_") for c in df.columns]
        if "date" not in df.columns and "datetime" in df.columns:
            df = df.rename(columns={"datetime": "date"})
        df = df.sort_values("date").reset_index(drop=True)
        log.info("Fetched %s from yfinance (%d rows)", ticker, len(df))
        return df
    except Exception as e:
        log.warning("Cannot fetch %s from yfinance: %s", ticker, e)
        return None


TRADING_DAYS_PER_YEAR = 252


def simulate_holding(
    df: pd.DataFrame,
    signal_date: str,
    entry_price: float,
    stop_loss: float,
    take_profit: float,
    holding_period: int = 20,
) -> dict[str, Any]:
    """Simulate the holding period for a signal.

    Returns:
    - status: 'ACTIVE' | 'HIT_SL' | 'HIT_TP' | 'EXPIRED' | 'NO_DATA'
    - exit_price: price at exit (or current price if active)
    - exit_date: when it exited
    - pnl: P&L percentage
    - days_held: number of trading days held
    - high_during_hold: max price during holding
    - low_during_hold: min price during holding
    """
    dates = df["date"].dt.strftime("%Y-%m-%d").tolist() if hasattr(df["date"], "dt") else list(df["date"].astype(str))

    if signal_date not in dates:
        return {"status": "NO_DATA", "pnl": 0.0, "days_held": 0}

    idx = dates.index(signal_date)
    if idx + 1 >= len(dates):
        return {"status": "NO_DATA", "pnl": 0.0, "days_held": 0}

    df_after = df.iloc[idx + 1:].reset_index(drop=True)
    sl_price = entry_price * (1 + stop_loss)
    tp_price = entry_price * (1 + take_profit)

    days_held = 0
    exit_price = None
    exit_date = None
    status = "ACTIVE"

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

        # Check TP first (price hit target)
        if high >= tp_price:
            exit_price = min(close, high)
            exit_date = current_date
            status = "HIT_TP"
            break

        # Check SL
        if low <= sl_price:
            exit_price = max(close, low)
            exit_date = current_date
            status = "HIT_SL"
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
    }


def track_signal(
    ticker: str,
    signal_date: str,
    entry_price: float | None = None,
    stop_loss: float = -0.03,
    take_profit: float = 0.08,
    holding_period: int = 20,
    data_dir: str | None = None,
) -> dict[str, Any]:
    """Track a single signal's realtime P&L.

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

    result = simulate_holding(df, signal_date, entry_price, stop_loss, take_profit, holding_period)
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
    """Aggregate tracking results into a summary."""
    total = len(results)
    hit_tp = sum(1 for r in results if r["status"] == "HIT_TP")
    hit_sl = sum(1 for r in results if r["status"] == "HIT_SL")
    active = sum(1 for r in results if r["status"] == "ACTIVE")
    expired = sum(1 for r in results if r["status"] == "EXPIRED")

    avg_pnl = np.mean([r["pnl"] for r in results]) if results else 0.0
    total_pnl = sum(r["pnl"] for r in results)

    return {
        "total": total,
        "hit_tp": hit_tp,
        "hit_sl": hit_sl,
        "active": active,
        "expired": expired,
        "win_rate": hit_tp / (hit_tp + hit_sl) * 100 if (hit_tp + hit_sl) > 0 else 0.0,
        "avg_pnl": round(avg_pnl, 4),
        "total_pnl": round(total_pnl, 4),
    }
