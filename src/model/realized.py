"""Realized T+20 validation for the production model family.

This module deliberately separates live evidence from walk-forward evaluation.
The daily candidate can continue collecting evidence while the sample is
immature; once enough complete baskets exist, an underperforming live family
becomes a circuit-breaker for further automatic promotion.
"""

from __future__ import annotations

import logging
import math
from typing import Any

import pandas as pd

log = logging.getLogger(__name__)


def _empty_health(
    *,
    model_family: str,
    min_trades: int,
    min_baskets: int,
    min_avg_excess_return: float,
    min_win_rate: float,
    reason: str = "No realized T+20 evidence is available yet.",
) -> dict[str, Any]:
    return {
        "model_family": model_family,
        "trade_count": 0,
        "basket_count": 0,
        "latest_signal_date": None,
        "avg_excess_return": None,
        "win_rate": None,
        "min_trades": int(min_trades),
        "min_baskets": int(min_baskets),
        "min_avg_excess_return": float(min_avg_excess_return),
        "min_win_rate": float(min_win_rate),
        "trades_to_minimum": int(min_trades),
        "baskets_to_minimum": int(min_baskets),
        "ready": False,
        "status": "collecting",
        "health_status": "pending",
        "reason": reason,
        "model_versions": [],
    }


def _merge_signal_sources(
    local: pd.DataFrame,
    cloud: pd.DataFrame,
    *,
    limit: int,
) -> pd.DataFrame:
    """Merge local/cloud signal rows while preferring complete local values."""
    frames: list[pd.DataFrame] = []
    for source_priority, frame in ((1, local), (0, cloud)):
        if frame is None or frame.empty:
            continue
        current = frame.copy()
        for column in ("signal_date", "ticker", "model_version"):
            if column not in current.columns:
                current[column] = None
        current["signal_date"] = pd.to_datetime(
            current["signal_date"], errors="coerce"
        ).dt.strftime("%Y-%m-%d")
        current["ticker"] = current["ticker"].astype(str)
        current["model_version"] = (
            current["model_version"].fillna("legacy_unknown").astype(str).str.strip()
        )
        current.loc[current["model_version"] == "", "model_version"] = "legacy_unknown"
        completeness_columns = [
            column
            for column in (
                "actual_excess_return_20d",
                "actual_excess_return_5d",
                "actual_excess_return",
                "actual_outperform",
            )
            if column in current.columns
        ]
        current["__completeness"] = (
            current[completeness_columns].notna().sum(axis=1)
            if completeness_columns
            else 0
        )
        current["__source_priority"] = source_priority
        frames.append(current)

    if not frames:
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True, sort=False)
    combined = combined.sort_values(
        ["__completeness", "__source_priority"],
        ascending=[False, False],
        kind="stable",
    )
    combined = combined.drop_duplicates(
        ["signal_date", "ticker"],
        keep="first",
    )
    return combined.drop(
        columns=["__completeness", "__source_priority"],
        errors="ignore",
    ).head(limit)


def load_realized_model_history(
    *,
    limit: int = 2000,
    prefer_cloud: bool = True,
) -> pd.DataFrame:
    """Load signals joined to realized outcomes from local and Supabase data."""
    if limit <= 0:
        raise ValueError("limit must be positive")

    from src.database import get_signals

    local = get_signals(limit=limit)
    cloud = pd.DataFrame()
    if prefer_cloud:
        try:
            from src.supabase_client import get_client

            client = get_client()
            if client is not None:
                cloud = pd.DataFrame(client.get_signals(limit=limit))
        except Exception as exc:  # pragma: no cover - network/config dependent
            log.warning("Could not load realized model history from Supabase: %s", exc)
    return _merge_signal_sources(local, cloud, limit=limit)


def summarize_realized_model_history(
    frame: pd.DataFrame,
    *,
    model_family: str,
    min_trades: int = 30,
    min_baskets: int = 10,
    min_avg_excess_return: float = 0.0,
    min_win_rate: float = 0.45,
    min_picks_per_basket: int = 3,
) -> dict[str, Any]:
    """Summarize current-family actuals and classify live health.

    A basket is counted only when at least ``min_picks_per_basket`` signals
    from the model family on the same signal date have realized outcomes.
    This prevents a few isolated rows from satisfying the live gate.
    """
    if min_trades <= 0 or min_baskets <= 0 or min_picks_per_basket <= 0:
        raise ValueError("Realized validation thresholds must be positive")
    min_avg_excess_return = float(min_avg_excess_return)
    min_win_rate = float(min_win_rate)
    if not math.isfinite(min_avg_excess_return):
        raise ValueError("min_avg_excess_return must be finite")
    if not math.isfinite(min_win_rate):
        raise ValueError("min_win_rate must be finite")

    if frame is None or frame.empty:
        return _empty_health(
            model_family=model_family,
            min_trades=min_trades,
            min_baskets=min_baskets,
            min_avg_excess_return=min_avg_excess_return,
            min_win_rate=min_win_rate,
        )

    data = frame.copy()
    required = {"signal_date", "ticker", "model_version"}
    missing = sorted(required - set(data.columns))
    if missing:
        return _empty_health(
            model_family=model_family,
            min_trades=min_trades,
            min_baskets=min_baskets,
            min_avg_excess_return=min_avg_excess_return,
            min_win_rate=min_win_rate,
            reason=f"Missing realized history columns: {', '.join(missing)}.",
        )

    data["model_version"] = data["model_version"].fillna("").astype(str)
    family_prefix = f"{str(model_family).strip()}_h"
    data = data[data["model_version"].str.startswith(family_prefix)].copy()
    if data.empty:
        return _empty_health(
            model_family=model_family,
            min_trades=min_trades,
            min_baskets=min_baskets,
            min_avg_excess_return=min_avg_excess_return,
            min_win_rate=min_win_rate,
            reason="No realized rows belong to the current model family yet.",
        )

    data["signal_date"] = pd.to_datetime(
        data["signal_date"], errors="coerce"
    ).dt.normalize()
    data["ticker"] = data["ticker"].astype(str)
    execution = pd.Series(pd.NA, index=data.index, dtype="Float64")
    for column in (
        "execution_excess_return",
        "actual_excess_return_20d",
        "actual_excess_return",
        "actual_excess_return_5d",
    ):
        if column in data.columns:
            values = pd.to_numeric(data[column], errors="coerce").astype("Float64")
            execution = execution.combine_first(values)
    data["execution_excess_return"] = execution
    data = data.dropna(subset=["signal_date", "ticker", "execution_excess_return"])
    data = data.drop_duplicates(["signal_date", "ticker"], keep="last")
    if data.empty:
        return _empty_health(
            model_family=model_family,
            min_trades=min_trades,
            min_baskets=min_baskets,
            min_avg_excess_return=min_avg_excess_return,
            min_win_rate=min_win_rate,
            reason="Current model family has no completed T+20 outcomes yet.",
        )

    basket_sizes = data.groupby("signal_date")["ticker"].size()
    complete_dates = set(
        basket_sizes[basket_sizes >= min_picks_per_basket].index.tolist()
    )
    complete = data[data["signal_date"].isin(complete_dates)].copy()
    trade_count = int(len(complete))
    basket_count = int(len(complete_dates))
    avg_return = float(complete["execution_excess_return"].mean())
    win_rate = float((complete["execution_excess_return"] > 0).mean())
    ready = trade_count >= min_trades and basket_count >= min_baskets
    healthy = ready and avg_return >= min_avg_excess_return and win_rate >= min_win_rate
    health_status = "pending" if not ready else ("healthy" if healthy else "underperforming")
    reason = (
        "Enough realized evidence is available and the model family passes the live thresholds."
        if health_status == "healthy"
        else "Enough realized evidence is available, but live thresholds are not met."
        if health_status == "underperforming"
        else "Continue collecting complete realized T+20 baskets before applying the live gate."
    )
    return {
        "model_family": model_family,
        "trade_count": trade_count,
        "basket_count": basket_count,
        "latest_signal_date": complete["signal_date"].max().date().isoformat(),
        "avg_excess_return": avg_return,
        "win_rate": win_rate,
        "min_trades": int(min_trades),
        "min_baskets": int(min_baskets),
        "min_avg_excess_return": float(min_avg_excess_return),
        "min_win_rate": float(min_win_rate),
        "trades_to_minimum": max(0, int(min_trades) - trade_count),
        "baskets_to_minimum": max(0, int(min_baskets) - basket_count),
        "ready": ready,
        "status": "ready" if ready else "collecting",
        "health_status": health_status,
        "reason": reason,
        "model_versions": sorted(data["model_version"].unique().tolist()),
    }


def get_model_live_health(
    *,
    model_family: str,
    min_trades: int = 30,
    min_baskets: int = 10,
    min_avg_excess_return: float = 0.0,
    min_win_rate: float = 0.45,
    prefer_cloud: bool = True,
) -> dict[str, Any]:
    """Load and summarize current-family realized performance."""
    history = load_realized_model_history(prefer_cloud=prefer_cloud)
    return summarize_realized_model_history(
        history,
        model_family=model_family,
        min_trades=min_trades,
        min_baskets=min_baskets,
        min_avg_excess_return=min_avg_excess_return,
        min_win_rate=min_win_rate,
    )
