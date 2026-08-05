from __future__ import annotations

import glob
import logging
import os
from typing import Any

import numpy as np
import pandas as pd

from src.config import Config
from src.time_utils import today_vn

log = logging.getLogger(__name__)

# Session-level cache: avoids re-fetching the same ticker from yfinance (rate-limit protection)
_stock_cache: dict[tuple[str, str, str], pd.DataFrame | None] = {}


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
    cache_key = (str(data_dir), ticker, today_vn().isoformat())
    if cache_key in _stock_cache:
        return _stock_cache[cache_key]

    pattern = os.path.join(data_dir, f"{ticker}_raw.parquet")
    parquet_df: pd.DataFrame | None = None
    files = glob.glob(pattern)
    if files:
        try:
            parquet_df = _normalize_stock_data(pd.read_parquet(files[0]))
            latest_date = parquet_df["date"].max()
            if latest_date >= pd.Timestamp(today_vn()):
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
    round_trip_cost: float = 0.0,
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
    if holding_period <= 0:
        raise ValueError("holding_period must be positive")

    # The signal is generated from the signal-day close. Trading starts on
    # the next session, using that session's open as the default entry.
    dates = df["date"].dt.strftime("%Y-%m-%d").tolist() if hasattr(df["date"], "dt") else list(df["date"].astype(str))

    if signal_date not in dates:
        return {"status": "NO_DATA", "pnl": 0.0, "days_held": 0}

    idx = dates.index(signal_date)
    if idx + 1 >= len(dates):
        return {"status": "PENDING", "pnl": 0.0, "days_held": 0}

    entry_idx = idx + 1
    df_after = df.iloc[entry_idx: entry_idx + holding_period].reset_index(drop=True)
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
        current_date = str(row["date"])[:10]

        max_high = max(max_high, high)
        min_low = min(min_low, low)

        # VN T+2: shares arrive after settlement_delay, can only sell from then
        if days_held <= settlement_delay:
            exit_price = close
            exit_date = current_date
            continue

        status = "ACTIVE"

        # Check SL first (conservative: assume worst case on wide-range days)
        # A non-negative stop loss disables the stop-loss leg.  This keeps
        # zero-valued test/config overrides from becoming an immediate exit.
        if stop_loss < 0 and low <= sl_price:
            exit_price = sl_price
            exit_date = current_date
            status = "HIT_SL"
            break

        # Check TP (only if SL not hit)
        # A non-positive take profit disables the take-profit leg.
        if take_profit > 0 and high >= tp_price:
            exit_price = tp_price
            exit_date = current_date
            status = "HIT_TP"
            break

        # Still holding, update current price
        exit_price = close
        exit_date = current_date

    if status == "ACTIVE" and days_held >= holding_period:
        status = "EXPIRED"

    gross_pnl = (exit_price - entry_price) / entry_price if exit_price else 0.0
    pnl = gross_pnl - round_trip_cost

    return {
        "status": status,
        "entry_price": entry_price,
        "exit_price": exit_price,
        "exit_date": exit_date,
        "pnl": round(pnl, 4),
        "gross_pnl": round(gross_pnl, 4),
        "transaction_cost": round(round_trip_cost, 4),
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
    round_trip_cost: float = 0.0,
    weight: float | None = None,
    data_dir: str | None = None,
) -> dict[str, Any]:
    """Track a single signal's realtime P&L.

    VN T+2: settlement_delay=2 → only execute SL/TP from 3rd trading day.
    If entry_price is None, it's inferred from the next trading session's open.
    """
    df = load_stock_data(ticker, data_dir)
    if df is None:
        return {
            "ticker": ticker,
            "signal_date": signal_date,
            "status": "NO_DATA",
            "pnl": 0.0,
            "days_held": 0,
            "weight": weight,
        }

    if entry_price is None:
        dates_list = df["date"].dt.strftime("%Y-%m-%d").tolist() if hasattr(df["date"], "dt") else list(df["date"].astype(str))
        if signal_date not in dates_list:
            return {
                "ticker": ticker,
                "signal_date": signal_date,
                "status": "NO_DATA",
                "pnl": 0.0,
                "days_held": 0,
                "weight": weight,
            }
        idx = dates_list.index(signal_date)
        if idx + 1 >= len(df):
            return {
                "ticker": ticker,
                "signal_date": signal_date,
                "status": "PENDING",
                "pnl": 0.0,
                "days_held": 0,
                "weight": weight,
            }
        entry_row = df.iloc[idx + 1]
        entry_price = float(entry_row.get("open", entry_row["close"]))
        if not np.isfinite(entry_price) or entry_price <= 0:
            entry_price = float(entry_row["close"])

    result = simulate_holding(
        df,
        signal_date,
        entry_price,
        stop_loss,
        take_profit,
        holding_period,
        settlement_delay,
        round_trip_cost,
    )
    result["ticker"] = ticker
    result["signal_date"] = signal_date
    result["entry_price"] = round(entry_price, 2)
    result["weight"] = weight
    return result


def track_signals(
    signals: list[dict[str, Any]],
    holding_period: int = 20,
    round_trip_cost: float | None = None,
    data_dir: str | None = None,
) -> list[dict[str, Any]]:
    """Track multiple signals and return their current P&L status."""
    if round_trip_cost is None:
        round_trip_cost = Config.round_trip_cost
    by_date: dict[str, list[dict[str, Any]]] = {}
    for sig in signals:
        by_date.setdefault(str(sig.get("signal_date", ""))[:10], []).append(sig)

    results = []
    for sig in signals:
        ticker = sig.get("ticker", "")
        signal_date = str(sig.get("signal_date", ""))[:10]
        sl = float(sig.get("stop_loss", -0.03))
        tp = float(sig.get("take_profit", 0.08))
        cohort = by_date[signal_date]
        supplied_weight = sig.get("weight")
        if supplied_weight is not None:
            raw_weight = float(supplied_weight)
        else:
            raw_weight = max(float(sig.get("score", 0.0)), 0.0)
        if not raw_weight:
            raw_weight = 1.0
        cohort_total = sum(
            max(float(item.get("weight", item.get("score", 0.0))), 0.0) or 1.0
            for item in cohort
        )
        weight = raw_weight / cohort_total
        result = track_signal(
            ticker,
            signal_date,
            stop_loss=sl,
            take_profit=tp,
            holding_period=holding_period,
            round_trip_cost=round_trip_cost,
            weight=weight,
            data_dir=data_dir,
        )
        results.append(result)
    return results


def get_signal_summary(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate tracking results into a portfolio-level summary.

    The reported basket P&L is a mark-to-market average across signal-date
    cohorts. It deliberately does not compound overlapping cohorts into a
    fictional sequential portfolio.
    """
    total = len(results)
    settling = sum(1 for r in results if r["status"] == "SETTLING")
    pending = sum(1 for r in results if r["status"] == "PENDING")
    no_data = sum(1 for r in results if r["status"] == "NO_DATA")
    hit_tp = sum(1 for r in results if r["status"] == "HIT_TP")
    hit_sl = sum(1 for r in results if r["status"] == "HIT_SL")
    active = sum(1 for r in results if r["status"] == "ACTIVE")
    expired = sum(1 for r in results if r["status"] == "EXPIRED")

    # Each signal date is one equally-weighted cohort; weights within a cohort
    # come from the signal score (or the persisted signal weight).
    by_date: dict[str, list[dict[str, Any]]] = {}
    for r in results:
        d = str(r.get("signal_date", ""))[:10]
        by_date.setdefault(d, []).append(r)

    dates_sorted = sorted(by_date.keys())
    daily_returns = []
    for d in dates_sorted:
        cohort = by_date[d]
        weight_total = sum(float(r.get("weight") or 1.0) for r in cohort)
        daily_returns.append(
            sum(float(r["pnl"]) * float(r.get("weight") or 1.0) for r in cohort) / weight_total
        )
    portfolio_pnl = float(np.mean(daily_returns)) if daily_returns else 0.0
    avg_pnl = np.mean([r["pnl"] for r in results]) if results else 0.0

    return {
        "total": total,
        "settling": settling,
        "pending": pending,
        "no_data": no_data,
        "hit_tp": hit_tp,
        "hit_sl": hit_sl,
        "active": active,
        "expired": expired,
        "win_rate": hit_tp / (hit_tp + hit_sl) * 100 if (hit_tp + hit_sl) > 0 else 0.0,
        "avg_pnl": round(avg_pnl, 4),
        "portfolio_pnl": round(portfolio_pnl, 4),
        "total_pnl": round(portfolio_pnl, 4),
    }
