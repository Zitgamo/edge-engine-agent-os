"""Reproducible walk-forward reports for the non-ML paper candidate."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from src.data.universe import VN30_TICKERS

log = logging.getLogger(__name__)

SCORE_COLUMNS = [
    "rs_5d",
    "rs_20d",
    "rs_60d",
    "return_5d",
    "return_20d",
    "return_60d",
]
POLICY_FILES = {
    "atr15_tp10": "outcomes_atr15_tp10.parquet",
    "atr2_tp10": "outcomes_atr2_tp10.parquet",
}
REPORT_COLUMNS = [
    "policy",
    "period",
    "start_date",
    "end_date",
    "signal_days",
    "trade_count",
    "trade_win_rate",
    "avg_net_return",
    "avg_excess_return",
    "profit_factor",
    "realized_rr",
    "fixed_exposure_roi",
    "max_drawdown",
    "min_trade_return",
    "promotion_status",
    "promotion_reason",
]


def _score_snapshot(
    features: pd.DataFrame,
    tickers: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Recreate the cross-sectional RS/momentum score point-in-time."""
    result = features.copy()
    result["ticker"] = result["ticker"].astype(str).str.upper()
    allowed_tickers = {
        str(ticker).strip().upper()
        for ticker in (VN30_TICKERS if tickers is None else tickers)
        if str(ticker).strip()
    }
    result = result[result["ticker"].isin(allowed_tickers)].copy()
    result["date"] = pd.to_datetime(result["date"], errors="coerce").dt.normalize()
    result = result.dropna(subset=["date"])
    result = result.dropna(subset=[*SCORE_COLUMNS, "return_20d", "atr", "close"])
    result = result[result["atr"] > 0]
    if result.empty:
        return result

    breadth = result.groupby("date")["return_20d"].apply(
        lambda values: float((values > 0).mean())
    )
    result["market_breadth_20d"] = result["date"].map(breadth)
    scores = pd.Series(0.0, index=result.index)
    for column in SCORE_COLUMNS:
        scores += result.groupby("date")[column].rank(pct=True)
    result["score"] = scores / len(SCORE_COLUMNS)
    return result


def _select_trades(
    features: pd.DataFrame,
    *,
    min_breadth: float,
    top_n: int,
    tickers: Iterable[str] | None = None,
) -> pd.DataFrame:
    scored = _score_snapshot(features, tickers=tickers)
    if scored.empty:
        return pd.DataFrame(columns=["date", "ticker", "score", "market_breadth_20d"])
    scored = scored[scored["market_breadth_20d"] >= min_breadth].copy()
    scored = scored.sort_values(
        ["date", "score", "ticker"],
        ascending=[True, False, True],
        kind="stable",
    )
    selected = scored.groupby("date", group_keys=False).head(top_n).copy()
    selected["rank"] = selected.groupby("date").cumcount() + 1
    return selected[[
        "date",
        "ticker",
        "rank",
        "score",
        "market_breadth_20d",
    ]]


def _periods(dates: pd.Series, holdout_days: int) -> dict[str, tuple[pd.Timestamp, pd.Timestamp]]:
    dates = pd.to_datetime(dates, errors="coerce").dropna().dt.normalize()
    if dates.empty:
        return {}
    end = dates.max()
    holdout_start = end - pd.Timedelta(days=holdout_days)
    return {
        "development": (dates.min(), pd.Timestamp("2023-12-31")),
        "validation": (pd.Timestamp("2024-01-01"), holdout_start - pd.Timedelta(days=1)),
        "locked_holdout": (holdout_start, end),
    }


def _metrics(
    trades: pd.DataFrame,
    *,
    policy: str,
    period: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    top_n: int,
    min_trades: int,
    min_profit_factor: float,
) -> dict[str, object]:
    data = trades[
        (trades["date"] >= start)
        & (trades["date"] <= end)
    ].copy()
    data = data.dropna(subset=["net_return", "excess_return"])
    trade_count = len(data)
    if trade_count:
        wins = data.loc[data["net_return"] > 0, "net_return"]
        losses = data.loc[data["net_return"] < 0, "net_return"]
        profit_factor = (
            float(wins.sum() / abs(losses.sum()))
            if not losses.empty
            else float("inf")
        )
        realized_rr = (
            float(wins.mean() / abs(losses.mean()))
            if not wins.empty and not losses.empty
            else float("nan")
        )
        weight = 1.0 / (20 * top_n)
        event_equity = 1.0 + data.sort_values(["exit_date", "date", "ticker"])[
            "net_return"
        ].cumsum() * weight
        drawdown = event_equity / event_equity.cummax() - 1.0
        avg_net = float(data["net_return"].mean())
        avg_excess = float(data["excess_return"].mean())
        status = (
            "ready"
            if trade_count >= min_trades and profit_factor >= min_profit_factor
            else "reject"
        )
        reason = (
            "minimum trade and PF gates passed"
            if status == "ready"
            else f"trade_count={trade_count}, PF={profit_factor:.2f}"
        )
        return {
            "policy": policy,
            "period": period,
            "start_date": start.date().isoformat(),
            "end_date": end.date().isoformat(),
            "signal_days": int(data["date"].nunique()),
            "trade_count": trade_count,
            "trade_win_rate": float((data["net_return"] > 0).mean()),
            "avg_net_return": avg_net,
            "avg_excess_return": avg_excess,
            "profit_factor": profit_factor,
            "realized_rr": realized_rr,
            "fixed_exposure_roi": float(data["net_return"].sum() * weight),
            "max_drawdown": float(drawdown.min()),
            "min_trade_return": float(data["net_return"].min()),
            "promotion_status": status,
            "promotion_reason": reason,
        }
    return {
        "policy": policy,
        "period": period,
        "start_date": start.date().isoformat(),
        "end_date": end.date().isoformat(),
        "signal_days": 0,
        "trade_count": 0,
        "trade_win_rate": np.nan,
        "avg_net_return": np.nan,
        "avg_excess_return": np.nan,
        "profit_factor": np.nan,
        "realized_rr": np.nan,
        "fixed_exposure_roi": np.nan,
        "max_drawdown": np.nan,
        "min_trade_return": np.nan,
        "promotion_status": "collecting",
        "promotion_reason": "no executable trades",
    }


def run_candidate_research(
    *,
    research_dir: str | Path = "data/research_kbs_5y",
    min_breadth: float = 0.60,
    top_n: int = 3,
    holdout_days: int = 180,
    min_trades: int = 100,
    min_profit_factor: float = 1.15,
    save: bool = True,
) -> pd.DataFrame:
    """Evaluate the predeclared candidate and its volatility challenger."""
    root = Path(research_dir)
    features_path = root / "processed" / "features_exact.parquet"
    if not features_path.exists():
        raise FileNotFoundError(f"Missing research features: {features_path}")
    features = pd.read_parquet(features_path)
    selected = _select_trades(
        features,
        min_breadth=min_breadth,
        top_n=top_n,
    )
    if selected.empty:
        return pd.DataFrame(columns=REPORT_COLUMNS)

    result_rows: list[dict[str, object]] = []
    periods = _periods(selected["date"], holdout_days)
    for policy, filename in POLICY_FILES.items():
        outcome_path = root / "research_results" / filename
        if not outcome_path.exists():
            log.warning("Skipping %s; outcome file is missing", policy)
            continue
        outcomes = pd.read_parquet(outcome_path)
        outcomes["date"] = pd.to_datetime(outcomes["date"], errors="coerce").dt.normalize()
        trades = selected.merge(
            outcomes[[
                "date",
                "ticker",
                "exit_date",
                "net_return",
                "excess_return",
            ]],
            on=["date", "ticker"],
            how="inner",
        )
        for period, (start, end) in periods.items():
            result_rows.append(_metrics(
                trades,
                policy=policy,
                period=period,
                start=start,
                end=end,
                top_n=top_n,
                min_trades=min_trades,
                min_profit_factor=min_profit_factor,
            ))

    report = pd.DataFrame(result_rows, columns=REPORT_COLUMNS)
    if save:
        output_path = root / "research_results" / "candidate_holdout_report.csv"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        report.to_csv(output_path, index=False)
        log.info("Saved candidate holdout report to %s", output_path)
    return report
