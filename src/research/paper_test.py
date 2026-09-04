"""Readiness reporting for the live/paper strategy test.

The report counts only complete, independent baskets.  A basket is one
strategy on one signal date with at least ``min_picks`` executable trades.
Returns are recomputed from persisted OHLCV using the configured execution
cost, so legacy SQLite actuals cannot silently make the paper test look
better than it is.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.backtest import _sltp_excess_return
from src.config import Config
from src.database import get_conn, init_db
from src.model.schema import HOLDING_PERIOD, N_PICKS

OUTPUT_COLUMNS = [
    "strategy_name",
    "trade_count",
    "basket_count",
    "signal_dates",
    "latest_signal_date",
    "win_rate",
    "avg_return_net",
    "positive_basket_rate",
    "avg_basket_return_net",
    "min_baskets",
    "target_baskets",
    "baskets_to_minimum",
    "progress_to_target",
    "min_trades",
    "target_trades",
    "trades_to_minimum",
    "progress_to_trade_target",
    "readiness",
]
COHORT_OUTPUT_COLUMNS = [
    "strategy_name",
    "strategy_version",
    *OUTPUT_COLUMNS[1:],
]


def _empty_report(*, include_version: bool = False) -> pd.DataFrame:
    return pd.DataFrame(
        columns=COHORT_OUTPUT_COLUMNS if include_version else OUTPUT_COLUMNS
    )


def _merge_signal_sources(
    remote: pd.DataFrame,
    local: pd.DataFrame,
    *,
    include_version: bool = False,
) -> pd.DataFrame:
    """Merge cloud history with local metadata without losing realized fields."""
    if remote.empty:
        return local
    if local.empty:
        return remote

    combined = pd.concat([remote, local], ignore_index=True, sort=False)
    key_columns = ["strategy_name", "signal_date", "ticker"]
    if include_version:
        if "strategy_version" not in combined.columns:
            combined["strategy_version"] = "legacy_unknown"
        combined["strategy_version"] = (
            combined["strategy_version"].fillna("legacy_unknown").astype(str).str.strip()
        )
        combined.loc[
            combined["strategy_version"] == "", "strategy_version"
        ] = "legacy_unknown"
        key_columns.insert(1, "strategy_version")
    merged_rows: list[dict[str, object]] = []
    for key, group in combined.groupby(key_columns, sort=False, dropna=False):
        row = dict(zip(key_columns, key))
        for column in combined.columns:
            if column in key_columns:
                continue
            values = group[column].dropna()
            row[column] = values.iloc[-1] if not values.empty else np.nan
        merged_rows.append(row)
    return pd.DataFrame(merged_rows)


def summarize_paper_test_readiness(
    frame: pd.DataFrame,
    *,
    return_col: str = "return_net",
    min_picks: int = N_PICKS,
    min_baskets: int = 30,
    target_baskets: int = 50,
    min_trades: int | None = None,
    target_trades: int | None = None,
    include_version: bool = False,
) -> pd.DataFrame:
    """Summarize complete de-duplicated baskets and collection progress."""
    required = {"strategy_name", "signal_date", "ticker", return_col}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Missing paper-test columns: {missing}")
    if min_picks <= 0 or min_baskets <= 0 or target_baskets < min_baskets:
        raise ValueError("Invalid paper-test readiness thresholds")
    if (min_trades is None) != (target_trades is None):
        raise ValueError("min_trades and target_trades must be provided together")
    if min_trades is not None and (min_trades <= 0 or target_trades < min_trades):
        raise ValueError("Invalid paper-test trade thresholds")

    data = frame.copy()
    data["strategy_name"] = data["strategy_name"].astype(str)
    data["ticker"] = data["ticker"].astype(str)
    group_columns = ["strategy_name"]
    if include_version:
        if "strategy_version" not in data.columns:
            data["strategy_version"] = "legacy_unknown"
        data["strategy_version"] = (
            data["strategy_version"].fillna("legacy_unknown").astype(str).str.strip()
        )
        data.loc[data["strategy_version"] == "", "strategy_version"] = "legacy_unknown"
        group_columns.append("strategy_version")
    data["signal_date"] = pd.to_datetime(
        data["signal_date"], errors="coerce"
    ).dt.normalize()
    data[return_col] = pd.to_numeric(data[return_col], errors="coerce")
    data = data.dropna(subset=["signal_date", "ticker"])
    data = data.drop_duplicates([*group_columns, "signal_date", "ticker"])
    if data.empty:
        return _empty_report(include_version=include_version)

    reports: list[dict[str, object]] = []
    for group_key, strategy_rows in data.groupby(group_columns, sort=True, dropna=False):
        if not isinstance(group_key, tuple):
            group_key = (group_key,)
        identity = {"strategy_name": group_key[0]}
        if include_version:
            identity["strategy_version"] = group_key[1]
        realized = strategy_rows.dropna(subset=[return_col]).copy()
        baskets: list[dict[str, object]] = []
        for signal_date, basket in realized.groupby("signal_date", sort=True):
            if len(basket) < min_picks:
                continue
            basket_return = float(basket[return_col].mean())
            baskets.append({
                "signal_date": signal_date,
                "basket_return": basket_return,
                "trade_count": len(basket),
            })

        basket_frame = pd.DataFrame(baskets)
        if basket_frame.empty:
            reports.append({
                **identity,
                "trade_count": 0,
                "basket_count": 0,
                "signal_dates": 0,
                "latest_signal_date": None,
                "win_rate": np.nan,
                "avg_return_net": np.nan,
                "positive_basket_rate": np.nan,
                "avg_basket_return_net": np.nan,
                "min_baskets": min_baskets,
                "target_baskets": target_baskets,
                "baskets_to_minimum": min_baskets,
                "progress_to_target": 0.0,
                "min_trades": min_trades,
                "target_trades": target_trades,
                "trades_to_minimum": min_trades or 0,
                "progress_to_trade_target": 0.0,
                "readiness": "collecting",
            })
            continue

        complete_dates = set(basket_frame["signal_date"])
        complete_trades = realized[realized["signal_date"].isin(complete_dates)]
        basket_count = len(basket_frame)
        trade_count = len(complete_trades)
        trade_ready = min_trades is None or trade_count >= min_trades
        reports.append({
            **identity,
            "trade_count": trade_count,
            "basket_count": basket_count,
            "signal_dates": basket_count,
            "latest_signal_date": basket_frame["signal_date"].max().date().isoformat(),
            "win_rate": float((complete_trades[return_col] > 0).mean()),
            "avg_return_net": float(complete_trades[return_col].mean()),
            "positive_basket_rate": float((basket_frame["basket_return"] > 0).mean()),
            "avg_basket_return_net": float(basket_frame["basket_return"].mean()),
            "min_baskets": min_baskets,
            "target_baskets": target_baskets,
            "baskets_to_minimum": max(0, min_baskets - basket_count),
            "progress_to_target": min(1.0, basket_count / target_baskets),
            "min_trades": min_trades,
            "target_trades": target_trades,
            "trades_to_minimum": (
                max(0, min_trades - trade_count) if min_trades is not None else 0
            ),
            "progress_to_trade_target": (
                min(1.0, trade_count / target_trades)
                if target_trades
                else 0.0
            ),
            "readiness": (
                "ready"
                if basket_count >= min_baskets and trade_ready
                else "collecting"
            ),
        })

    return pd.DataFrame(
        reports,
        columns=COHORT_OUTPUT_COLUMNS if include_version else OUTPUT_COLUMNS,
    )


def load_paper_test_readiness(
    *,
    raw_data_dir: str | Path | None = None,
    paper_raw_data_dir: str | Path | None = None,
    stop_loss: float | None = None,
    take_profit: float | None = None,
    round_trip_cost: float | None = None,
    holding_period: int = HOLDING_PERIOD,
    min_picks: int = N_PICKS,
    min_baskets: int = 30,
    target_baskets: int = 50,
    min_trades: int | None = None,
    target_trades: int | None = None,
    prefer_cloud: bool = False,
    include_version: bool = False,
) -> pd.DataFrame:
    """Recompute strategy signals from OHLCV and return readiness.

    ``prefer_cloud`` is used by the hosted dashboard because GitHub Actions
    does not restore the ignored SQLite database between runs.
    """
    signals = pd.DataFrame()
    if prefer_cloud:
        try:
            from src.supabase_client import get_client

            client = get_client()
            rows = client.get_strategy_performance() if client else []
            signals = pd.DataFrame(rows)
        except Exception:
            signals = pd.DataFrame()

    init_db()
    conn = get_conn()
    local_signals = pd.read_sql_query(
        """SELECT strategy_name, strategy_version, signal_date, ticker, rank, score,
                  stop_loss, take_profit
           FROM strategy_performance
           ORDER BY signal_date, strategy_name, rank""",
        conn,
    )
    conn.close()
    if not signals.empty:
        signals = _merge_signal_sources(
            signals,
            local_signals,
            include_version=include_version,
        )
    else:
        signals = local_signals
    if signals.empty:
        return _empty_report(include_version=include_version)

    config = Config()
    raw_data_dir = raw_data_dir if raw_data_dir is not None else config.raw_data_dir
    paper_raw_data_dir = (
        paper_raw_data_dir
        if paper_raw_data_dir is not None
        else config.paper_raw_data_dir
    )
    stop_loss = config.stop_loss if stop_loss is None else stop_loss
    take_profit = config.take_profit if take_profit is None else take_profit
    round_trip_cost = (
        config.round_trip_cost if round_trip_cost is None else round_trip_cost
    )

    prices_cache: dict[str, pd.DataFrame] = {}
    returns: list[float | None] = []
    for row in signals.itertuples(index=False):
        signal_date = pd.Timestamp(row.signal_date).date().isoformat()
        row_stop_loss = getattr(row, "stop_loss", np.nan)
        row_take_profit = getattr(row, "take_profit", np.nan)
        row_stop_loss = stop_loss if pd.isna(row_stop_loss) else float(row_stop_loss)
        row_take_profit = (
            take_profit if pd.isna(row_take_profit) else float(row_take_profit)
        )
        value = _sltp_excess_return(
            str(row.ticker),
            signal_date,
            row_stop_loss,
            row_take_profit,
            holding_period,
            prices_cache,
            round_trip_cost,
            data_dir=(
                paper_raw_data_dir
                if str(row.strategy_name) in config.paper_strategy_names()
                else raw_data_dir
            ),
        )
        if value is None:
            # Hosted dashboards may not have the ignored paper OHLC cache.
            # Use a previously realized cloud outcome in that case; local
            # OHLC remains authoritative whenever it can be recalculated.
            for column in (
                f"actual_excess_return_{holding_period}d",
                "actual_excess_return_20d",
                "actual_excess_return",
                "actual_excess_return_5d",
            ):
                remote_value = getattr(row, column, np.nan)
                if pd.notna(remote_value):
                    value = float(remote_value)
                    break
        returns.append(float(value) if value is not None and np.isfinite(value) else np.nan)

    signals = signals.copy()
    signals["return_net"] = returns
    return summarize_paper_test_readiness(
        signals,
        return_col="return_net",
        min_picks=min_picks,
        min_baskets=min_baskets,
        target_baskets=target_baskets,
        min_trades=min_trades,
        target_trades=target_trades,
        include_version=include_version,
    )
