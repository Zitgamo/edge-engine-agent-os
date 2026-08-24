"""De-duplicated, cost-aware attribution for rule-based strategies."""

from __future__ import annotations

from typing import Final

import pandas as pd

OUTPUT_COLUMNS: Final[list[str]] = [
    "strategy_name",
    "trade_count",
    "signal_dates",
    "basket_count",
    "win_rate",
    "avg_return_net",
    "median_return_net",
    "positive_basket_rate",
    "avg_basket_return_net",
    "cumulative_return_net",
]


def _empty_result() -> pd.DataFrame:
    return pd.DataFrame(columns=OUTPUT_COLUMNS)


def summarize_strategy_attribution(
    frame: pd.DataFrame,
    *,
    return_col: str | None = None,
    round_trip_cost: float = 0.0,
) -> pd.DataFrame:
    """Summarize realized strategy outcomes without counting duplicate trades.

    ``frame[return_col]`` is treated as the gross excess return for one
    realized trade. The configured round-trip cost is subtracted once. Rows
    are de-duplicated by strategy, signal date and ticker; the same ticker
    selected by two different strategies remains a trade in each strategy's
    attribution, but never twice within one strategy.
    """
    if frame.empty:
        return _empty_result()
    required = {"strategy_name", "signal_date", "ticker"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Missing strategy-attribution columns: {missing}")

    candidates = [
        return_col,
        "actual_excess_return_20d",
        "actual_excess_return",
        "actual_excess_return_5d",
        "excess_return",
    ]
    selected_return_col = next(
        (column for column in candidates if column and column in frame.columns),
        None,
    )
    if selected_return_col is None:
        raise ValueError("No realized excess-return column is available")

    data = frame.copy()
    data["signal_date"] = pd.to_datetime(data["signal_date"], errors="coerce").dt.normalize()
    data[selected_return_col] = pd.to_numeric(data[selected_return_col], errors="coerce")
    if "realized" in data.columns:
        realized = pd.to_numeric(data["realized"], errors="coerce")
        data = data[realized.eq(1) | data[selected_return_col].notna()].copy()
    data = data.dropna(subset=["strategy_name", "signal_date", "ticker", selected_return_col])
    if data.empty:
        return _empty_result()
    data = data.drop_duplicates(["strategy_name", "signal_date", "ticker"])
    data["return_net"] = data[selected_return_col] - float(round_trip_cost)

    rows: list[dict[str, object]] = []
    for strategy_name, strategy in data.groupby("strategy_name", sort=True):
        basket = (
            strategy.groupby("signal_date", as_index=False)["return_net"]
            .mean()
            .rename(columns={"return_net": "basket_return_net"})
        )
        trade_returns = strategy["return_net"]
        basket_returns = basket["basket_return_net"]
        rows.append({
            "strategy_name": strategy_name,
            "trade_count": int(len(strategy)),
            "signal_dates": int(strategy["signal_date"].nunique()),
            "basket_count": int(len(basket)),
            "win_rate": float((trade_returns > 0).mean()),
            "avg_return_net": float(trade_returns.mean()),
            "median_return_net": float(trade_returns.median()),
            "positive_basket_rate": float((basket_returns > 0).mean()),
            "avg_basket_return_net": float(basket_returns.mean()),
            "cumulative_return_net": float((1.0 + basket_returns).prod() - 1.0),
        })
    return pd.DataFrame(rows, columns=OUTPUT_COLUMNS)


def load_realized_strategy_attribution(
    *,
    round_trip_cost: float = 0.0,
    prefer_cloud: bool = False,
) -> pd.DataFrame:
    """Load realized strategy outcomes and return the attribution report.

    Streamlit deployments often do not have the GitHub runner's ignored
    SQLite database.  ``prefer_cloud`` reads the persisted Supabase table
    first and falls back to local SQLite for offline use and tests.
    """
    if prefer_cloud:
        try:
            from src.supabase_client import get_client

            client = get_client()
            rows = client.get_strategy_performance() if client else []
            cloud_frame = pd.DataFrame(rows)
            if not cloud_frame.empty:
                return summarize_strategy_attribution(
                    cloud_frame,
                    return_col="actual_excess_return_20d",
                    round_trip_cost=round_trip_cost,
                )
        except Exception:
            pass

    from src.database import get_conn

    conn = get_conn()
    try:
        frame = pd.read_sql_query(
            """
            SELECT
                sp.strategy_name,
                sp.signal_date,
                sp.ticker,
                COALESCE(
                    sp.actual_excess_return_20d,
                    a.actual_excess_return_20d,
                    sp.actual_excess_return_5d,
                    a.actual_excess_return_5d
                ) AS actual_excess_return_20d,
                COALESCE(sp.actual_outperform, a.actual_outperform) AS actual_outperform,
                sp.realized
            FROM strategy_performance sp
            LEFT JOIN actuals a
              ON a.signal_date = sp.signal_date
             AND a.ticker = sp.ticker
            WHERE sp.realized = 1
               OR a.actual_outperform IS NOT NULL
            """,
            conn,
        )
    finally:
        conn.close()
    return summarize_strategy_attribution(
        frame,
        return_col="actual_excess_return_20d",
        round_trip_cost=round_trip_cost,
    )
