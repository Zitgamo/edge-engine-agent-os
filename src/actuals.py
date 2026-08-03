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

log = logging.getLogger(__name__)


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
) -> pd.DataFrame:
    """Load a ticker from persisted raw data and fetch only when necessary."""
    if ticker in cache:
        return cache[ticker]

    path = Path(config.raw_data_dir) / f"{ticker}_raw.parquet"
    if path.exists():
        try:
            result = _normalise_prices(pd.read_parquet(path))
            if not result.empty:
                cache[ticker] = result
                return result
        except Exception as exc:
            log.warning("Cannot read persisted prices for %s: %s", ticker, exc)

    result = _normalise_prices(collector.fetch(ticker, days=365))
    cache[ticker] = result
    return result


def calculate_actuals(
    signals: pd.DataFrame | list[dict[str, Any]],
    holding_period: int = 20,
    config: Config | None = None,
) -> pd.DataFrame:
    """Calculate T+``holding_period`` excess returns for pending signals.

    A signal is realized only when both the stock and benchmark have a close
    on the stock's future trading date.  Missing future observations stay
    pending rather than being converted into a zero or a loss.
    """
    pending = signals.copy() if isinstance(signals, pd.DataFrame) else pd.DataFrame(signals)
    required = {"signal_date", "ticker"}
    if pending.empty or not required.issubset(pending.columns):
        return pd.DataFrame()

    config = config or Config()
    collector = OHLCVCollector(config)
    prices_cache: dict[str, pd.DataFrame] = {}
    try:
        benchmark = _load_prices("VNINDEX", config, collector, prices_cache)
    except Exception as exc:
        log.warning("Cannot load benchmark for actuals: %s", exc)
        return pd.DataFrame()
    if benchmark.empty:
        return pd.DataFrame()

    benchmark_map = dict(
        zip(benchmark["date"].dt.strftime("%Y-%m-%d"), pd.to_numeric(benchmark["close"], errors="coerce"))
    )

    actuals: list[dict[str, Any]] = []
    for row in pending.itertuples(index=False):
        signal_date = str(getattr(row, "signal_date"))[:10]
        ticker = str(getattr(row, "ticker"))
        try:
            prices = _load_prices(ticker, config, collector, prices_cache)
            dates = prices["date"].dt.strftime("%Y-%m-%d").tolist()
            if signal_date not in dates:
                continue
            idx = dates.index(signal_date)
            future_idx = idx + holding_period
            if future_idx >= len(dates):
                continue

            future_date = dates[future_idx]
            stock_now = float(prices.iloc[idx]["close"])
            stock_future = float(prices.iloc[future_idx]["close"])
            bm_now = benchmark_map.get(signal_date)
            bm_future = benchmark_map.get(future_date)
            if (
                bm_now is None
                or bm_future is None
                or pd.isna(bm_now)
                or pd.isna(bm_future)
                or stock_now == 0
                or float(bm_now) == 0
            ):
                continue

            stock_return = (stock_future - stock_now) / stock_now
            benchmark_return = (float(bm_future) - float(bm_now)) / float(bm_now)
            excess = stock_return - benchmark_return
            actuals.append({
                "signal_date": signal_date,
                "ticker": ticker,
                "actual_excess_return_5d": excess,
                "actual_excess_return": excess,
                "actual_outperform": int(excess > 0),
                "realized_date": future_date,
            })
        except Exception as exc:
            log.warning("Actual calculation failed for %s %s: %s", signal_date, ticker, exc)

    return pd.DataFrame(actuals)
