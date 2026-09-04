"""Build reproducible feature and execution-outcome panels from KBS OHLCV."""

from __future__ import annotations

import logging
from datetime import time
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from src.features.returns import ReturnFeatures
from src.features.rs import RelativeStrength
from src.features.volatility import ATR
from src.time_utils import now_vn

log = logging.getLogger(__name__)

POLICIES = {
    "atr15_tp10": (1.5, 0.10),
    "atr2_tp10": (2.0, 0.10),
}
FEATURE_COLUMNS = [
    "ticker",
    "date",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "return_5d",
    "return_20d",
    "return_60d",
    "rs_5d",
    "rs_20d",
    "rs_60d",
    "atr",
    "atr_pct",
]
OUTCOME_COLUMNS = [
    "ticker",
    "date",
    "entry_date",
    "exit_date",
    "net_return",
    "excess_return",
    "gross_return",
    "reason",
    "label_end_date",
]


def latest_closed_date(as_of: object | None = None) -> pd.Timestamp:
    """Return the latest session that is closed for the Vietnam market."""
    current = as_of or now_vn()
    latest = pd.Timestamp(current.date())
    if current.time() < time(15, 30):
        latest -= pd.Timedelta(days=1)
    return latest.normalize()


def _normalise_prices(frame: pd.DataFrame) -> pd.DataFrame:
    required = {"date", "open", "high", "low", "close", "volume"}
    if frame.empty or not required.issubset(frame.columns):
        return pd.DataFrame()
    result = frame.copy()
    result["date"] = pd.to_datetime(result["date"], errors="coerce")
    if result["date"].dt.tz is not None:
        result["date"] = result["date"].dt.tz_localize(None)
    result["date"] = result["date"].dt.normalize()
    for column in ["open", "high", "low", "close", "volume"]:
        result[column] = pd.to_numeric(result[column], errors="coerce")
    invalid = (
        result["date"].isna()
        | result[["open", "high", "low", "close", "volume"]].isna().any(axis=1)
        | (result[["open", "high", "low", "close"]] <= 0).any(axis=1)
        | (result["volume"] < 0)
        | (result["high"] < result["low"])
        | (result["high"] < result[["open", "close"]].max(axis=1))
        | (result["low"] > result[["open", "close"]].min(axis=1))
    )
    result = result.loc[~invalid].copy()
    if result.empty:
        return pd.DataFrame()
    return (
        result.drop_duplicates("date", keep="last")
        .sort_values("date")
        .reset_index(drop=True)
    )


def _load_prices(raw_dir: Path, ticker: str, closed_date: pd.Timestamp) -> pd.DataFrame:
    path = raw_dir / f"{ticker}_raw.parquet"
    if not path.exists():
        return pd.DataFrame()
    try:
        frame = _normalise_prices(pd.read_parquet(path))
    except Exception as exc:
        log.warning("Cannot read KBS research file %s: %s", path, exc)
        return pd.DataFrame()
    return frame[frame["date"] <= closed_date].reset_index(drop=True)


def research_tickers(raw_dir: str | Path) -> list[str]:
    """List stock tickers represented by a raw directory."""
    root = Path(raw_dir)
    return sorted(
        path.stem.removesuffix("_raw")
        for path in root.glob("*_raw.parquet")
        if path.stem.removesuffix("_raw").upper() != "VNINDEX"
    )


def refresh_research_data(
    research_dir: str | Path,
    *,
    config=None,
    tickers: Iterable[str] | None = None,
    lookback_days: int = 1825,
    refresh_days: int = 120,
    force_full: bool = False,
    as_of: object | None = None,
) -> pd.DataFrame:
    """Refresh a reproducible KBS research cache without deleting old data.

    Existing files are merged with the newest response so a scheduled run can
    update the cache incrementally.  A failed fetch leaves a usable stale file
    in place and records the failure in ``refresh_summary.json``.
    """
    if lookback_days <= 0 or refresh_days <= 0:
        raise ValueError("lookback_days and refresh_days must be positive")

    from src.config import Config
    from src.data.collector import OHLCVCollector
    from src.data.universe import HOSE_TICKERS

    base = config or Config()
    collector_config = Config()
    collector_config.data_source = "kbs"
    collector_config.kbs_base_url = base.kbs_base_url
    collector_config.kbs_timeout_seconds = base.kbs_timeout_seconds
    collector = OHLCVCollector(collector_config)

    root = Path(research_dir)
    raw_dir = root / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    expected_closed_date = latest_closed_date(as_of)
    stale_after_days = 3
    symbols = sorted({
        str(ticker).strip().upper()
        for ticker in [*(tickers or HOSE_TICKERS), "VNINDEX"]
        if str(ticker).strip()
    })
    reports: list[dict[str, object]] = []
    for ticker in symbols:
        path = raw_dir / f"{ticker}_raw.parquet"
        try:
            existing = (
                _normalise_prices(pd.read_parquet(path))
                if path.exists()
                else pd.DataFrame()
            )
        except Exception as exc:
            log.warning("Cannot read existing research cache for %s: %s", ticker, exc)
            existing = pd.DataFrame()
        fetch_days = lookback_days if force_full or existing.empty else refresh_days
        try:
            fresh = _normalise_prices(collector.fetch(ticker, days=fetch_days))
            if fresh.empty:
                if existing.empty:
                    raise RuntimeError("empty normalized response")
                reports.append({
                    "ticker": ticker,
                    "rows": len(existing),
                    "start": existing["date"].min().date().isoformat(),
                    "end": existing["date"].max().date().isoformat(),
                    "fetched_days": fetch_days,
                    "status": "stale",
                    "error": "empty normalized response; retained existing cache",
                })
                continue
            merged = (
                fresh
                if existing.empty
                else _normalise_prices(pd.concat([existing, fresh], ignore_index=True))
            )
            if merged.empty:
                raise RuntimeError("empty normalized response")
            merged.to_parquet(path, index=False)
            closed_rows = merged[merged["date"] <= expected_closed_date]
            closed_latest = (
                pd.Timestamp(closed_rows["date"].max()).normalize()
                if not closed_rows.empty
                else pd.NaT
            )
            cache_is_stale = pd.isna(closed_latest) or closed_latest < (
                expected_closed_date - pd.Timedelta(days=stale_after_days)
            )
            reports.append({
                "ticker": ticker,
                "rows": len(merged),
                "start": merged["date"].min().date().isoformat(),
                "end": (
                    closed_latest.date().isoformat()
                    if pd.notna(closed_latest)
                    else merged["date"].max().date().isoformat()
                ),
                "fetched_days": fetch_days,
                "status": "stale" if cache_is_stale else "ok",
                "error": (
                    f"cache ends {closed_latest.date().isoformat() if pd.notna(closed_latest) else 'before closed market data'}, "
                    f"expected around {expected_closed_date.date().isoformat()}"
                    if cache_is_stale
                    else ""
                ),
            })
        except Exception as exc:
            message = str(exc)
            log.warning("Research refresh failed for %s: %s", ticker, message)
            reports.append({
                "ticker": ticker,
                "rows": len(existing),
                "start": existing["date"].min().date().isoformat() if not existing.empty else "",
                "end": existing["date"].max().date().isoformat() if not existing.empty else "",
                "fetched_days": fetch_days,
                "status": "stale" if not existing.empty else "error",
                "error": message,
            })

    report = pd.DataFrame(reports)
    report["source"] = "kbs"
    report["retrieved_at"] = now_vn().isoformat()
    (root / "download_summary.json").write_text(
        report.to_json(orient="records", force_ascii=False, indent=2),
        encoding="utf-8",
    )
    return report


def build_feature_panel(
    raw_dir: str | Path,
    *,
    output_path: str | Path | None = None,
    as_of: object | None = None,
) -> pd.DataFrame:
    """Build return/RS/ATR features with only point-in-time information."""
    root = Path(raw_dir)
    closed_date = latest_closed_date(as_of)
    benchmark = _load_prices(root, "VNINDEX", closed_date)
    if benchmark.empty:
        raise FileNotFoundError(f"Missing usable VNINDEX data in {root}")

    frames: list[pd.DataFrame] = []
    for ticker in research_tickers(root):
        prices = _load_prices(root, ticker, closed_date)
        if prices.empty:
            continue
        technical = ReturnFeatures().compute(prices)
        technical = RelativeStrength().compute(technical, benchmark)
        technical = ATR().compute(technical)
        technical["ticker"] = ticker
        frames.append(technical[FEATURE_COLUMNS])

    if not frames:
        return pd.DataFrame(columns=FEATURE_COLUMNS)
    result = pd.concat(frames, ignore_index=True)
    result = result.sort_values(["date", "ticker"]).reset_index(drop=True)
    if output_path is not None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        result.to_parquet(path, index=False)
        log.info("Saved %d feature rows to %s", len(result), path)
    return result


def _outcomes_for_ticker(
    ticker: str,
    stock: pd.DataFrame,
    benchmark: pd.DataFrame,
    *,
    atr_multiple: float,
    take_profit: float,
    holding_period: int,
    round_trip_cost: float,
) -> list[dict[str, object]]:
    technical = ATR().compute(stock)
    stock_dates = stock["date"].dt.strftime("%Y-%m-%d").to_numpy()
    opens = pd.to_numeric(stock["open"], errors="coerce").to_numpy(dtype=float)
    highs = pd.to_numeric(stock["high"], errors="coerce").to_numpy(dtype=float)
    lows = pd.to_numeric(stock["low"], errors="coerce").to_numpy(dtype=float)
    closes = pd.to_numeric(stock["close"], errors="coerce").to_numpy(dtype=float)
    atr_values = pd.to_numeric(technical["atr"], errors="coerce").to_numpy(dtype=float)
    benchmark_dates = benchmark["date"].dt.strftime("%Y-%m-%d").to_numpy()
    benchmark_opens = pd.to_numeric(
        benchmark["open"], errors="coerce"
    ).to_numpy(dtype=float)
    benchmark_closes = pd.to_numeric(
        benchmark["close"], errors="coerce"
    ).to_numpy(dtype=float)
    benchmark_by_date = {
        str(date): (float(open_price), float(close_price))
        for date, open_price, close_price in zip(
            benchmark_dates,
            benchmark_opens,
            benchmark_closes,
        )
    }
    outcomes: list[dict[str, object]] = []

    for signal_idx, signal_date_value in enumerate(stock_dates):
        entry_idx = signal_idx + 1
        end_idx = entry_idx + holding_period
        if end_idx > len(stock):
            continue
        signal_date = str(signal_date_value)
        signal_close = closes[signal_idx]
        atr = atr_values[signal_idx]
        if not np.isfinite(signal_close) or signal_close <= 0 or not np.isfinite(atr):
            continue

        entry_date = str(stock_dates[entry_idx])
        entry = opens[entry_idx]
        if not np.isfinite(entry) or entry <= 0:
            entry = closes[entry_idx]
        if not np.isfinite(entry) or entry <= 0 or entry_date not in benchmark_by_date:
            continue
        benchmark_entry = benchmark_by_date[entry_date][0]
        if not np.isfinite(benchmark_entry) or benchmark_entry <= 0:
            benchmark_entry = benchmark_by_date[entry_date][1]
        if not np.isfinite(benchmark_entry) or benchmark_entry <= 0:
            continue

        stop_loss = -(atr_multiple * atr / signal_close)
        exit_idx = end_idx - 1
        exit_price = closes[exit_idx]
        reason = "time"
        stop_price = entry * (1 + stop_loss)
        target_price = entry * (1 + take_profit)
        for held_idx in range(entry_idx, end_idx):
            held = held_idx - entry_idx + 1
            if held <= 2:
                continue
            if lows[held_idx] <= stop_price:
                exit_idx = held_idx
                exit_price = (
                    opens[held_idx]
                    if np.isfinite(opens[held_idx]) and opens[held_idx] <= stop_price
                    else stop_price
                )
                reason = (
                    "stop_gap"
                    if np.isfinite(opens[held_idx]) and opens[held_idx] <= stop_price
                    else "stop"
                )
                break
            if highs[held_idx] >= target_price:
                exit_idx = held_idx
                exit_price = (
                    opens[held_idx]
                    if np.isfinite(opens[held_idx]) and opens[held_idx] >= target_price
                    else target_price
                )
                reason = (
                    "target_gap"
                    if np.isfinite(opens[held_idx]) and opens[held_idx] >= target_price
                    else "target"
                )
                break
            exit_idx = held_idx
            exit_price = closes[held_idx]

        exit_date = str(stock_dates[exit_idx])
        if exit_date not in benchmark_by_date:
            continue
        benchmark_exit = benchmark_by_date[exit_date][1]
        if not np.isfinite(benchmark_exit) or not np.isfinite(exit_price):
            continue
        gross_return = (exit_price - entry) / entry
        net_return = gross_return - round_trip_cost
        benchmark_return = (benchmark_exit - benchmark_entry) / benchmark_entry
        outcomes.append({
            "ticker": ticker,
            "date": signal_date,
            "entry_date": entry_date,
            "exit_date": exit_date,
            "net_return": float(net_return),
            "excess_return": float(net_return - benchmark_return),
            "gross_return": float(gross_return),
            "reason": reason,
            "label_end_date": stock_dates[end_idx - 1],
        })
    return outcomes


def build_execution_outcomes(
    raw_dir: str | Path,
    *,
    output_dir: str | Path | None = None,
    holding_period: int = 20,
    round_trip_cost: float = 0.003,
    as_of: object | None = None,
) -> dict[str, pd.DataFrame]:
    """Build dynamic-ATR, T+2, gap-aware outcomes for declared policies."""
    if holding_period <= 0:
        raise ValueError("holding_period must be positive")
    root = Path(raw_dir)
    closed_date = latest_closed_date(as_of)
    benchmark = _load_prices(root, "VNINDEX", closed_date)
    if benchmark.empty:
        raise FileNotFoundError(f"Missing usable VNINDEX data in {root}")

    ticker_frames: dict[str, pd.DataFrame] = {}
    for ticker in research_tickers(root):
        frame = _load_prices(root, ticker, closed_date)
        if not frame.empty:
            ticker_frames[ticker] = frame

    results: dict[str, pd.DataFrame] = {}
    for policy, (atr_multiple, take_profit) in POLICIES.items():
        rows: list[dict[str, object]] = []
        for ticker, frame in ticker_frames.items():
            rows.extend(_outcomes_for_ticker(
                ticker,
                frame,
                benchmark,
                atr_multiple=atr_multiple,
                take_profit=take_profit,
                holding_period=holding_period,
                round_trip_cost=round_trip_cost,
            ))
        result = pd.DataFrame(rows, columns=OUTCOME_COLUMNS)
        if not result.empty:
            result = result.sort_values(["date", "ticker"]).reset_index(drop=True)
        results[policy] = result
        if output_dir is not None:
            path = Path(output_dir) / f"outcomes_{policy}.parquet"
            path.parent.mkdir(parents=True, exist_ok=True)
            result.to_parquet(path, index=False)
            log.info("Saved %d %s outcome rows to %s", len(result), policy, path)
    return results


def build_research_dataset(
    research_dir: str | Path,
    *,
    holding_period: int = 20,
    round_trip_cost: float = 0.003,
    as_of: object | None = None,
) -> dict[str, object]:
    """Build the panel and policy files consumed by candidate_backtest."""
    root = Path(research_dir)
    raw_dir = root / "raw"
    processed_dir = root / "processed"
    results_dir = root / "research_results"
    features = build_feature_panel(
        raw_dir,
        output_path=processed_dir / "features_exact.parquet",
        as_of=as_of,
    )
    outcomes = build_execution_outcomes(
        raw_dir,
        output_dir=results_dir,
        holding_period=holding_period,
        round_trip_cost=round_trip_cost,
        as_of=as_of,
    )
    return {
        "features": features,
        "outcomes": outcomes,
        "closed_date": latest_closed_date(as_of).date().isoformat(),
    }
