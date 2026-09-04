"""Calculate realized signal returns from persisted OHLCV data.

The GitHub runner starts with a fresh SQLite database on every run, while
Supabase keeps the signal history.  Keeping the calculation here lets both
local and cloud backfills use the same point-in-time logic.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd

from src.config import Config
from src.data.collector import OHLCVCollector
from src.time_utils import today_vn
from src.tracking.realtime import simulate_holding

log = logging.getLogger(__name__)


def add_execution_excess_column(
    df: pd.DataFrame,
    holding_period: int = 20,
) -> pd.DataFrame:
    """Expose one canonical executable excess-return column.

    Older local/cloud rows use ``actual_excess_return_5d`` even though the
    production holding period is T+20.  Prefer the correctly named field and
    fall back to legacy fields while data is being migrated.
    """
    result = df.copy()
    candidates = [
        "execution_excess_return",
        f"actual_excess_return_{holding_period}d",
        "actual_excess_return",
        "actual_excess_return_5d",
    ]
    combined = pd.Series(pd.NA, index=result.index, dtype="Float64")
    for column in candidates:
        if column in result.columns:
            values = pd.to_numeric(result[column], errors="coerce").astype("Float64")
            combined = combined.combine_first(values)
    result["execution_excess_return"] = combined
    return result


def resolve_execution_exits(
    signals: pd.DataFrame,
    config: Config | None = None,
) -> pd.DataFrame:
    """Materialize configured exits for rows that do not carry per-trade exits.

    Production strategy rankings historically omitted ``stop_loss`` and
    ``take_profit`` while paper candidates persist them explicitly.  Keeping
    the fallback resolution in one place makes both kinds of rows use the
    same execution parameters before outcome keys are built.
    """
    result = signals.copy()
    runtime_config = config or Config()
    defaults = {
        "stop_loss": float(getattr(runtime_config, "stop_loss", Config.stop_loss)),
        "take_profit": float(getattr(runtime_config, "take_profit", Config.take_profit)),
    }
    for column, fallback in defaults.items():
        if column not in result.columns:
            result[column] = fallback
            continue
        result[column] = pd.to_numeric(result[column], errors="coerce").fillna(fallback)
    return result


def _normalise_prices(df: pd.DataFrame) -> pd.DataFrame:
    """Return sorted, de-duplicated prices with normalized dates."""
    if df.empty or "date" not in df.columns or "close" not in df.columns:
        return pd.DataFrame()
    result = df.copy()
    result["date"] = pd.to_datetime(result["date"], errors="coerce")
    result = result.dropna(subset=["date", "close"])
    if result["date"].dt.tz is not None:
        result["date"] = result["date"].dt.tz_localize(None)
    result["date"] = result["date"].dt.normalize()
    return (
        result.drop_duplicates(subset=["date"], keep="last")
        .sort_values("date")
        .reset_index(drop=True)
    )


def _load_prices(
    ticker: str,
    config: Config,
    collector: OHLCVCollector,
    cache: dict[str, pd.DataFrame],
    required_date: str | None = None,
    days: int = 365,
    required_future_bars: int = 0,
) -> pd.DataFrame:
    """Load prices with enough forward coverage for a pending outcome."""

    def covers_required_window(frame: pd.DataFrame) -> bool:
        if frame.empty or required_date is None:
            return not frame.empty
        dates = frame["date"].dt.strftime("%Y-%m-%d").tolist()
        try:
            signal_idx = dates.index(str(required_date)[:10])
        except ValueError:
            return False
        future_bars = len(dates) - signal_idx - 1
        return future_bars >= max(0, int(required_future_bars))

    cached = cache.get(ticker)
    if cached is not None and covers_required_window(cached):
        return cached

    path = Path(config.raw_data_dir) / f"{ticker}_raw.parquet"
    persisted = pd.DataFrame()
    if path.exists():
        try:
            persisted = _normalise_prices(pd.read_parquet(path))
            if covers_required_window(persisted):
                cache[ticker] = persisted
                return persisted
            if required_date is not None:
                log.info(
                    "Persisted prices for %s do not cover %s through %d future bars; refreshing",
                    ticker,
                    required_date,
                    required_future_bars,
                )
        except Exception as exc:
            log.warning("Cannot read persisted prices for %s: %s", ticker, exc)

    try:
        fresh = _normalise_prices(collector.fetch(ticker, days=days))
    except Exception as exc:
        log.warning("Cannot refresh prices for %s: %s", ticker, exc)
        fresh = pd.DataFrame()
    if not fresh.empty:
        # A refresh can be non-empty but still contain only recent rows.  Do
        # not discard persisted signal-day/early-exit data in that case;
        # fresh values win on overlapping dates while the historical window
        # remains available for the calculation.
        combined = (
            fresh
            if persisted.empty
            else _normalise_prices(pd.concat([persisted, fresh], ignore_index=True))
        )
        cache[ticker] = combined
        return combined

    # A transient data-source outage should not discard a persisted early
    # exit that is still independently computable.
    cache[ticker] = persisted
    return persisted


def _history_days(required_date: str | None, holding_period: int) -> int:
    if not required_date:
        return 365
    try:
        age_days = max(0, (today_vn() - pd.Timestamp(required_date).date()).days)
    except (TypeError, ValueError):
        return 365
    return max(365, int(age_days * 1.3) + holding_period + 60)


def calculate_actuals(
    signals: pd.DataFrame | list[dict[str, Any]],
    holding_period: int = 20,
    config: Config | None = None,
    settlement_delay: int = 2,
) -> pd.DataFrame:
    """Calculate executable T+``holding_period`` excess returns for pending signals.

    Signals are generated from a closing price, so entry is the next trading
    session's open.  SL/TP and settlement rules match the realtime tracker.
    Missing future observations stay pending rather than becoming a loss.
    """
    pending = signals.copy() if isinstance(signals, pd.DataFrame) else pd.DataFrame(signals)
    required = {"signal_date", "ticker"}
    if pending.empty or not required.issubset(pending.columns):
        return pd.DataFrame()

    config = config or Config()
    collector = OHLCVCollector(config)
    prices_cache: dict[str, pd.DataFrame] = {}
    signal_dates = pd.to_datetime(pending["signal_date"], errors="coerce").dropna()
    history_days = _history_days(
        signal_dates.min().date().isoformat() if not signal_dates.empty else None,
        holding_period,
    )
    try:
        benchmark = _load_prices(
            "VNINDEX",
            config,
            collector,
            prices_cache,
            required_date=signal_dates.min().date().isoformat() if not signal_dates.empty else None,
            days=history_days,
            required_future_bars=holding_period,
        )
    except Exception as exc:
        log.warning("Cannot load benchmark for actuals: %s", exc)
        return pd.DataFrame()
    if benchmark.empty:
        return pd.DataFrame()

    benchmark_dates = benchmark["date"].dt.strftime("%Y-%m-%d")
    benchmark_close = dict(zip(benchmark_dates, pd.to_numeric(benchmark["close"], errors="coerce")))
    benchmark_open = (
        dict(zip(benchmark_dates, pd.to_numeric(benchmark["open"], errors="coerce")))
        if "open" in benchmark.columns
        else {}
    )

    actuals: list[dict[str, Any]] = []
    for row in pending.to_dict(orient="records"):
        signal_date = str(row.get("signal_date", ""))[:10]
        ticker = str(row.get("ticker", ""))
        try:
            prices = _load_prices(
                ticker,
                config,
                collector,
                prices_cache,
                required_date=signal_date,
                days=history_days,
                required_future_bars=holding_period,
            )
            dates = prices["date"].dt.strftime("%Y-%m-%d").tolist()
            if signal_date not in dates:
                continue
            idx = dates.index(signal_date)
            entry_idx = idx + 1
            if entry_idx >= len(dates):
                continue

            entry_row = prices.iloc[entry_idx]
            entry_date = dates[entry_idx]
            entry_price = float(entry_row.get("open", entry_row["close"]))
            if pd.isna(entry_price) or entry_price <= 0:
                entry_price = float(entry_row["close"])

            stop_loss = row.get("stop_loss", config.stop_loss)
            take_profit = row.get("take_profit", config.take_profit)
            stop_loss = config.stop_loss if stop_loss is None or pd.isna(stop_loss) else float(stop_loss)
            take_profit = config.take_profit if take_profit is None or pd.isna(take_profit) else float(take_profit)

            simulation = simulate_holding(
                prices,
                signal_date,
                entry_price,
                stop_loss,
                take_profit,
                holding_period=holding_period,
                settlement_delay=settlement_delay,
                round_trip_cost=config.round_trip_cost,
            )
            if simulation.get("status") not in {"HIT_SL", "HIT_TP", "EXPIRED"}:
                continue

            exit_date = str(simulation["exit_date"])[:10]
            stock_return = float(simulation["pnl"])
            bm_now = benchmark_open.get(entry_date)
            if bm_now is None or pd.isna(bm_now):
                bm_now = benchmark_close.get(entry_date)
            bm_future = benchmark_close.get(exit_date)
            if (
                bm_now is None
                or bm_future is None
                or pd.isna(bm_now)
                or pd.isna(bm_future)
                or float(bm_now) == 0
            ):
                continue

            benchmark_return = (float(bm_future) - float(bm_now)) / float(bm_now)
            excess = stock_return - benchmark_return
            outcome = {
                "signal_date": signal_date,
                "ticker": ticker,
                "stop_loss": stop_loss,
                "take_profit": take_profit,
                "actual_stock_return": stock_return,
                "benchmark_return": benchmark_return,
                "gross_stock_return": float(simulation.get("gross_pnl", stock_return)),
                "transaction_cost": float(
                    simulation.get("transaction_cost", config.round_trip_cost)
                ),
                "actual_excess_return_5d": excess,
                "actual_excess_return": excess,
                "actual_outperform": int(excess > 0),
                "realized_date": exit_date,
                "entry_date": entry_date,
                "entry_price": entry_price,
                "exit_price": simulation["exit_price"],
                "status": simulation["status"],
            }
            outcome[f"actual_excess_return_{holding_period}d"] = excess
            actuals.append(outcome)
        except Exception as exc:
            log.warning("Actual calculation failed for %s %s: %s", signal_date, ticker, exc)

    return pd.DataFrame(actuals)
