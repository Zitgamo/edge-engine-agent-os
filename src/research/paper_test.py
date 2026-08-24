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
    "readiness",
]


def _empty_report() -> pd.DataFrame:
    return pd.DataFrame(columns=OUTPUT_COLUMNS)


def summarize_paper_test_readiness(
    frame: pd.DataFrame,
    *,
    return_col: str = "return_net",
    min_picks: int = N_PICKS,
    min_baskets: int = 30,
    target_baskets: int = 50,
) -> pd.DataFrame:
    """Summarize complete de-duplicated baskets and collection progress."""
    required = {"strategy_name", "signal_date", "ticker", return_col}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Missing paper-test columns: {missing}")
    if min_picks <= 0 or min_baskets <= 0 or target_baskets < min_baskets:
        raise ValueError("Invalid paper-test readiness thresholds")

    data = frame.copy()
    data["strategy_name"] = data["strategy_name"].astype(str)
    data["ticker"] = data["ticker"].astype(str)
    data["signal_date"] = pd.to_datetime(
        data["signal_date"], errors="coerce"
    ).dt.normalize()
    data[return_col] = pd.to_numeric(data[return_col], errors="coerce")
    data = data.dropna(subset=["signal_date", "ticker"])
    data = data.drop_duplicates(["strategy_name", "signal_date", "ticker"])
    if data.empty:
        return _empty_report()

    reports: list[dict[str, object]] = []
    for strategy_name in sorted(data["strategy_name"].unique()):
        strategy_rows = data[data["strategy_name"] == strategy_name]
        realized = strategy_rows.dropna(subset=[return_col]).copy()
        baskets: list[dict[str, object]] = []
        for signal_date, basket in realized.groupby("signal_date", sort=True):
            if len(basket) < min_picks:
                continue
            basket_return = float(basket[return_col].mean())
            baskets.append({
                "signal_date": signal_date,
                "basket_return": basket_return,
                "trade_count": int(len(basket)),
            })

        basket_frame = pd.DataFrame(baskets)
        if basket_frame.empty:
            reports.append({
                "strategy_name": strategy_name,
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
                "readiness": "collecting",
            })
            continue

        complete_dates = set(basket_frame["signal_date"])
        complete_trades = realized[realized["signal_date"].isin(complete_dates)]
        basket_count = int(len(basket_frame))
        reports.append({
            "strategy_name": strategy_name,
            "trade_count": int(len(complete_trades)),
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
            "readiness": "ready" if basket_count >= min_baskets else "collecting",
        })

    return pd.DataFrame(reports, columns=OUTPUT_COLUMNS)


def load_paper_test_readiness(
    *,
    raw_data_dir: str | Path | None = None,
    stop_loss: float | None = None,
    take_profit: float | None = None,
    round_trip_cost: float | None = None,
    holding_period: int = HOLDING_PERIOD,
    min_picks: int = N_PICKS,
    min_baskets: int = 30,
    target_baskets: int = 50,
    prefer_cloud: bool = False,
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

    if signals.empty:
        init_db()
        conn = get_conn()
        signals = pd.read_sql_query(
            """SELECT strategy_name, signal_date, ticker, rank, score
               FROM strategy_performance
               ORDER BY signal_date, strategy_name, rank""",
            conn,
        )
        conn.close()
    if signals.empty:
        return _empty_report()

    config = Config()
    raw_data_dir = raw_data_dir if raw_data_dir is not None else config.raw_data_dir
    stop_loss = config.stop_loss if stop_loss is None else stop_loss
    take_profit = config.take_profit if take_profit is None else take_profit
    round_trip_cost = (
        config.round_trip_cost if round_trip_cost is None else round_trip_cost
    )

    prices_cache: dict[str, pd.DataFrame] = {}
    returns: list[float | None] = []
    for row in signals.itertuples(index=False):
        signal_date = pd.Timestamp(row.signal_date).date().isoformat()
        value = _sltp_excess_return(
            str(row.ticker),
            signal_date,
            stop_loss,
            take_profit,
            holding_period,
            prices_cache,
            round_trip_cost,
            data_dir=raw_data_dir,
        )
        returns.append(float(value) if value is not None and np.isfinite(value) else np.nan)

    signals = signals.copy()
    signals["return_net"] = returns
    return summarize_paper_test_readiness(
        signals,
        return_col="return_net",
        min_picks=min_picks,
        min_baskets=min_baskets,
        target_baskets=target_baskets,
    )
