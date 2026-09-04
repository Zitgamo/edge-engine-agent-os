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
COHORT_OUTPUT_COLUMNS: Final[list[str]] = [
    "strategy_name",
    "strategy_version",
    *OUTPUT_COLUMNS[1:],
]


def _empty_result(*, include_version: bool = False) -> pd.DataFrame:
    return pd.DataFrame(
        columns=COHORT_OUTPUT_COLUMNS if include_version else OUTPUT_COLUMNS
    )


def summarize_strategy_attribution(
    frame: pd.DataFrame,
    *,
    return_col: str | None = None,
    round_trip_cost: float = 0.0,
    returns_are_net: bool = True,
    include_version: bool = False,
) -> pd.DataFrame:
    """Summarize realized strategy outcomes without counting duplicate trades.

    Persisted ``actual_excess_return_*`` values already include the execution
    round-trip cost.  They are therefore treated as net by default.  For a
    caller supplying a gross return column, pass ``returns_are_net=False`` so
    the configured cost is subtracted exactly once. Rows are de-duplicated by
    strategy, signal date and ticker; the same ticker selected by two
    different strategies remains a trade in each strategy's attribution, but
    never twice within one strategy.
    """
    if frame.empty:
        return _empty_result(include_version=include_version)
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
        return _empty_result(include_version=include_version)
    group_columns = ["strategy_name"]
    if include_version:
        if "strategy_version" not in data.columns:
            data["strategy_version"] = "legacy_unknown"
        data["strategy_version"] = (
            data["strategy_version"].fillna("legacy_unknown").astype(str).str.strip()
        )
        data.loc[data["strategy_version"] == "", "strategy_version"] = "legacy_unknown"
        group_columns.append("strategy_version")
    data = data.drop_duplicates([*group_columns, "signal_date", "ticker"])
    data["return_net"] = data[selected_return_col]
    if not returns_are_net:
        data["return_net"] = data["return_net"] - float(round_trip_cost)

    rows: list[dict[str, object]] = []
    for group_key, strategy in data.groupby(group_columns, sort=True, dropna=False):
        if not isinstance(group_key, tuple):
            group_key = (group_key,)
        basket = (
            strategy.groupby("signal_date", as_index=False)["return_net"]
            .mean()
            .rename(columns={"return_net": "basket_return_net"})
        )
        trade_returns = strategy["return_net"]
        basket_returns = basket["basket_return_net"]
        row: dict[str, object] = {
            "strategy_name": group_key[0],
            "trade_count": len(strategy),
            "signal_dates": int(strategy["signal_date"].nunique()),
            "basket_count": len(basket),
            "win_rate": float((trade_returns > 0).mean()),
            "avg_return_net": float(trade_returns.mean()),
            "median_return_net": float(trade_returns.median()),
            "positive_basket_rate": float((basket_returns > 0).mean()),
            "avg_basket_return_net": float(basket_returns.mean()),
            "cumulative_return_net": float((1.0 + basket_returns).prod() - 1.0),
        }
        if include_version:
            row["strategy_version"] = group_key[1]
        rows.append(row)
    return pd.DataFrame(
        rows,
        columns=COHORT_OUTPUT_COLUMNS if include_version else OUTPUT_COLUMNS,
    )


def load_realized_strategy_attribution(
    *,
    round_trip_cost: float = 0.0,
    prefer_cloud: bool = False,
    include_version: bool = False,
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
                    returns_are_net=True,
                    include_version=include_version,
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
                sp.strategy_version,
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
        returns_are_net=True,
        include_version=include_version,
    )
