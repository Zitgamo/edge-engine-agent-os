"""The production-safe, non-ML paper candidate.

This module deliberately keeps the forward test independent from the XGBoost
ensemble.  It uses the candidate that survived the current research pass:
VN30, market breadth, cross-sectional RS/momentum, and a volatility-scaled
initial stop.
"""

from __future__ import annotations

import json
import logging
from datetime import time
from pathlib import Path

import pandas as pd

from src.config import Config
from src.data.collector import OHLCVCollector
from src.data.universe import VN30_TICKERS
from src.features.returns import ReturnFeatures
from src.features.rs import RelativeStrength
from src.features.volatility import ATR
from src.time_utils import now_vn

log = logging.getLogger(__name__)

SCORE_COLUMNS = [
    "rs_5d",
    "rs_20d",
    "rs_60d",
    "return_5d",
    "return_20d",
    "return_60d",
]
PAPER_VERSION = "paper_v1_vn30_rs_atr2_tp10"
PAPER_TICKER_EXIT_VERSION = "paper_v2_vn30_rs_ticker_exit_v2"


def _normalise_prices(frame: pd.DataFrame) -> pd.DataFrame:
    """Normalize and conservatively clean one persisted OHLCV frame."""
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


def _read_prices(data_dir: str | Path, ticker: str) -> pd.DataFrame:
    path = Path(data_dir) / f"{ticker}_raw.parquet"
    if not path.exists():
        return pd.DataFrame()
    try:
        return _normalise_prices(pd.read_parquet(path))
    except Exception as exc:
        log.warning("Cannot read paper prices for %s: %s", ticker, exc)
        return pd.DataFrame()


def _latest_closed_date(as_of=None) -> pd.Timestamp:
    current = as_of or now_vn()
    latest = pd.Timestamp(current.date())
    if current.time() < time(15, 30):
        latest -= pd.Timedelta(days=1)
    return latest.normalize()


def paper_execution_config(config: Config | None = None) -> Config:
    """Create an execution config pointing at the isolated paper data store."""
    base = config or Config()
    result = Config()
    result.data_source = "kbs"
    result.raw_data_dir = Path(base.paper_raw_data_dir)
    result.kbs_base_url = base.kbs_base_url
    result.kbs_timeout_seconds = base.kbs_timeout_seconds
    result.round_trip_cost = base.round_trip_cost
    result.stop_loss = base.stop_loss
    result.take_profit = base.ticker_exit_baseline_take_profit
    return result


def refresh_kbs_data(
    config: Config | None = None,
    *,
    force_full: bool = False,
    tickers: list[str] | None = None,
) -> pd.DataFrame:
    """Refresh isolated paper data and return a per-ticker collection report."""
    base = config or Config()
    paper_dir = Path(base.paper_raw_data_dir)
    paper_dir.mkdir(parents=True, exist_ok=True)
    collector = OHLCVCollector(paper_execution_config(base))
    symbols = list(dict.fromkeys([*(tickers or VN30_TICKERS), "VNINDEX"]))
    reports: list[dict[str, object]] = []

    for ticker in symbols:
        path = paper_dir / f"{ticker}_raw.parquet"
        existing = _read_prices(paper_dir, ticker)
        fetch_days = (
            base.paper_lookback_days
            if force_full or existing.empty
            else base.paper_refresh_days
        )
        try:
            fresh = collector.fetch(ticker, days=fetch_days)
            fresh = _normalise_prices(fresh)
            merged = (
                fresh
                if existing.empty
                else _normalise_prices(pd.concat([existing, fresh], ignore_index=True))
            )
            if merged.empty:
                raise RuntimeError("empty normalized response")
            merged.to_parquet(path, index=False)
            reports.append({
                "ticker": ticker,
                "rows": len(merged),
                "start": merged["date"].min().date().isoformat(),
                "end": merged["date"].max().date().isoformat(),
                "fetched_days": fetch_days,
                "status": "ok",
                "error": "",
            })
        except Exception as exc:
            message = str(exc)
            log.warning("Paper refresh failed for %s: %s", ticker, message)
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
    (paper_dir / "refresh_summary.json").write_text(
        json.dumps(report.to_dict(orient="records"), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return report


def build_candidate_signals(
    raw_data_dir: str | Path,
    *,
    signal_date: str | pd.Timestamp | None = None,
    min_breadth: float = 0.60,
    min_universe_count: int = 30,
    top_n: int = 3,
    atr_multiple: float = 2.0,
    take_profit: float = 0.10,
    strategy_name: str = "vn30_rs_atr2_tp10",
    strategy_version: str = PAPER_VERSION,
    exit_profiles: dict[str, object] | None = None,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Build one point-in-time candidate snapshot from raw daily prices."""
    if top_n <= 0:
        raise ValueError("top_n must be positive")
    if not 0 <= min_breadth <= 1:
        raise ValueError("min_breadth must be between 0 and 1")
    if min_universe_count <= 0:
        raise ValueError("min_universe_count must be positive")
    if atr_multiple <= 0:
        raise ValueError("atr_multiple must be positive")

    benchmark = _read_prices(raw_data_dir, "VNINDEX")
    if benchmark.empty:
        return pd.DataFrame(), {
            "status": "blocked",
            "reason": "benchmark data unavailable",
            "signal_date": None,
        }
    closed_date = _latest_closed_date()
    benchmark = benchmark[benchmark["date"] <= closed_date].copy()
    if benchmark.empty:
        return pd.DataFrame(), {
            "status": "blocked",
            "reason": "no closed benchmark session",
            "signal_date": None,
        }

    latest_date = (
        pd.Timestamp(signal_date).normalize()
        if signal_date is not None
        else pd.Timestamp(benchmark["date"].max()).normalize()
    )
    benchmark = benchmark[benchmark["date"] <= latest_date].copy()
    if benchmark.empty or latest_date not in set(benchmark["date"]):
        return pd.DataFrame(), {
            "status": "blocked",
            "reason": "requested signal date is unavailable",
            "signal_date": latest_date.date().isoformat(),
        }

    snapshots: list[pd.DataFrame] = []
    for ticker in VN30_TICKERS:
        prices = _read_prices(raw_data_dir, ticker)
        prices = prices[prices["date"] <= latest_date].copy()
        if len(prices) < 80:
            continue
        technical = ReturnFeatures().compute(prices)
        technical = RelativeStrength().compute(technical, benchmark)
        technical = ATR().compute(technical)
        current = technical[technical["date"] == latest_date].copy()
        if current.empty:
            continue
        current["ticker"] = ticker
        snapshots.append(current.tail(1))

    if not snapshots:
        return pd.DataFrame(), {
            "status": "blocked",
            "reason": "no VN30 point-in-time snapshots",
            "signal_date": latest_date.date().isoformat(),
        }

    snapshot = pd.concat(snapshots, ignore_index=True)
    breadth_values = pd.to_numeric(snapshot["return_20d"], errors="coerce").dropna()
    if breadth_values.empty:
        return pd.DataFrame(), {
            "status": "blocked",
            "reason": "breadth unavailable",
            "signal_date": latest_date.date().isoformat(),
        }
    breadth = float((breadth_values > 0).mean())
    eligible = snapshot.dropna(subset=[*SCORE_COLUMNS, "return_20d", "atr", "close"]).copy()
    eligible = eligible[eligible["atr"] > 0]
    report: dict[str, object] = {
        "status": "blocked",
        "reason": None,
        "signal_date": latest_date.date().isoformat(),
        "market_breadth_20d": breadth,
        "min_market_breadth_20d": min_breadth,
        "universe_count": len(snapshot),
        "min_universe_count": min_universe_count,
        "eligible_count": len(eligible),
        "top_n": top_n,
        "atr_multiple": atr_multiple,
        "take_profit": take_profit,
        "strategy_name": strategy_name,
    }
    if len(snapshot) < min_universe_count:
        report["reason"] = "incomplete VN30 universe"
        return pd.DataFrame(), report
    if breadth < min_breadth:
        report["reason"] = "market breadth below threshold"
        return pd.DataFrame(), report
    if len(eligible) < top_n:
        report["reason"] = "fewer than top_n eligible stocks"
        return pd.DataFrame(), report

    scores = pd.Series(0.0, index=eligible.index)
    for column in SCORE_COLUMNS:
        scores += eligible[column].rank(pct=True)
    eligible["score"] = scores / len(SCORE_COLUMNS)
    eligible = eligible.sort_values(
        ["score", "ticker"],
        ascending=[False, True],
        kind="stable",
    ).head(top_n).copy()
    eligible["rank"] = range(1, len(eligible) + 1)
    eligible["signal_date"] = latest_date.date().isoformat()
    eligible["ensemble_score"] = eligible["score"]
    eligible["weight"] = 1.0 / len(eligible)
    eligible["action"] = "PAPER_BUY"
    eligible["stop_loss"] = -(atr_multiple * eligible["atr"] / eligible["close"])
    eligible["take_profit"] = take_profit
    eligible["exit_profile_used"] = False
    eligible["exit_profile_confidence"] = 0.0
    if exit_profiles is not None:
        from src.research.ticker_exit_optimizer import apply_exit_profiles

        eligible = apply_exit_profiles(
            eligible,
            eligible[["ticker", "atr", "close"]],
            exit_profiles,
            baseline_atr_multiple=atr_multiple,
            baseline_take_profit=take_profit,
        )
    eligible["market_breadth_20d"] = breadth
    eligible["strategy_name"] = strategy_name
    eligible["strategy_version"] = strategy_version
    report["exit_profile_used_count"] = int(
        eligible.get("exit_profile_used", pd.Series(False, index=eligible.index)).sum()
    )
    report.update(status="passed", reason="candidate snapshot created")
    columns = [
        "strategy_name",
        "strategy_version",
        "signal_date",
        "date",
        "rank",
        "ticker",
        "score",
        "ensemble_score",
        "weight",
        "action",
        "stop_loss",
        "take_profit",
        "atr",
        "atr_pct",
        "market_breadth_20d",
        "exit_profile_used",
        "exit_profile_confidence",
    ]
    return eligible[columns].reset_index(drop=True), report


def run_paper_update(
    config: Config | None = None,
    *,
    force_full: bool = False,
) -> dict[str, object]:
    """Refresh KBS, write today's candidate snapshot, and backfill T+20."""
    base = config or Config()
    refresh = refresh_kbs_data(base, force_full=force_full)
    paper_config = paper_execution_config(base)
    exit_profiles = None
    exit_profiles_active = False
    if base.enable_ticker_exit_profiles:
        from src.research.ticker_exit_optimizer import (
            has_approved_profiles,
            load_ticker_exit_profiles,
        )

        exit_profiles = load_ticker_exit_profiles(base.ticker_exit_profile_path)
        exit_profiles_active = has_approved_profiles(
            exit_profiles,
            baseline_atr_multiple=base.ticker_exit_baseline_atr_multiple,
            baseline_take_profit=base.ticker_exit_baseline_take_profit,
            deployment="paper",
        )
        if not exit_profiles_active:
            log.warning(
                "Ticker exit profiles enabled but no approved profile is available; "
                "paper candidate remains on the fixed baseline"
            )
            exit_profiles = None
    strategy_name = (
        base.paper_ticker_exit_strategy_name
        if exit_profiles_active
        else base.paper_strategy_name
    )
    strategy_version = (
        PAPER_TICKER_EXIT_VERSION if exit_profiles_active else PAPER_VERSION
    )
    signals, snapshot_report = build_candidate_signals(
        paper_config.raw_data_dir,
        min_breadth=base.paper_min_breadth_20d,
        min_universe_count=base.paper_min_universe_count,
        top_n=base.paper_top_n,
        atr_multiple=base.ticker_exit_baseline_atr_multiple,
        take_profit=base.ticker_exit_baseline_take_profit,
        strategy_name=strategy_name,
        strategy_version=strategy_version,
        exit_profiles=exit_profiles,
    )
    from src.database import (
        backfill_strategy_performance_actuals,
        save_paper_strategy_signals,
    )

    saved = save_paper_strategy_signals(signals) if not signals.empty else 0
    backfilled = backfill_strategy_performance_actuals(
        strategy_name=strategy_name,
        holding_period=20,
        config=paper_config,
    )
    return {
        "refresh": refresh,
        "snapshot": snapshot_report,
        "signals_saved": saved,
        "actuals_backfilled": backfilled,
        "paper_data_dir": str(paper_config.raw_data_dir),
        "strategy_name": strategy_name,
        "strategy_version": strategy_version,
        "exit_profiles_active": exit_profiles_active,
        "exit_profile_path": str(base.ticker_exit_profile_path),
    }
