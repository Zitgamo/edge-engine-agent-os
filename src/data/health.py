"""Data diagnostics; historical adjustments are warnings, not trade rules."""

from __future__ import annotations

import numpy as np
import pandas as pd


def assess_prices(
    ticker: str,
    frame: pd.DataFrame,
    benchmark_date: object,
    previous: pd.DataFrame | None = None,
) -> dict:
    report = {
        "ticker": ticker,
        "rows": len(frame),
        "latest_date": None,
        "status": "invalid",
        "revised_rows": 0,
        "max_price_revision_pct": 0.0,
    }
    required = {"date", "open", "high", "low", "close", "volume"}
    if frame.empty or not required.issubset(frame.columns):
        report["reason"] = "missing prices or required columns"
        return report
    dates = (
        pd.to_datetime(frame["date"], errors="coerce", utc=True)
        .dt.tz_convert("Asia/Ho_Chi_Minh")
        .dt.tz_localize(None)
        .dt.normalize()
    )
    values = frame[["open", "high", "low", "close", "volume"]].apply(pd.to_numeric, errors="coerce")
    invalid = (
        dates.isna()
        | dates.duplicated()
        | ~np.isfinite(values).all(axis=1)
        | (values[["open", "high", "low", "close"]] <= 0).any(axis=1)
        | (values["volume"] < 0)
        | (values["high"] < values[["open", "close", "low"]].max(axis=1))
        | (values["low"] > values[["open", "close"]].min(axis=1))
    )
    latest = dates.max()
    report["latest_date"] = latest.date().isoformat() if pd.notna(latest) else None
    report["invalid_rows"] = int(invalid.sum())
    benchmark = pd.Timestamp(benchmark_date).normalize()
    if invalid.any() or pd.isna(latest) or latest > benchmark:
        report["reason"] = "invalid prices or dates after benchmark"
        return report
    report["status"] = "stale" if latest < benchmark else "ok"
    report["lag_calendar_days"] = int((benchmark - latest).days)
    if previous is not None and not previous.empty and required.issubset(previous.columns):
        old = previous.copy()
        old["date"] = (
            pd.to_datetime(old["date"], errors="coerce", utc=True)
            .dt.tz_convert("Asia/Ho_Chi_Minh")
            .dt.tz_localize(None)
            .dt.normalize()
        )
        new = frame.assign(date=dates)
        overlap = old.drop_duplicates("date").merge(new, on="date", suffixes=("_old", "_new"))
        changes = []
        for column in ("open", "high", "low", "close"):
            before = pd.to_numeric(overlap[f"{column}_old"], errors="coerce")
            after = pd.to_numeric(overlap[f"{column}_new"], errors="coerce")
            changes.append((after / before.where(before > 0) - 1).abs())
        deltas = pd.concat(changes, axis=1).max(axis=1).replace([np.inf, -np.inf], np.nan)
        report["revised_rows"] = int((deltas > 1e-6).sum())
        report["max_price_revision_pct"] = float(deltas.max()) if deltas.notna().any() else 0.0
        report["large_revision"] = bool((deltas >= 0.05).any())
    return report
