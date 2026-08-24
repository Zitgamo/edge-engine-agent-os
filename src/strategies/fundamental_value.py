from __future__ import annotations

import logging

import pandas as pd

from src.strategies.base import Strategy

log = logging.getLogger(__name__)


class FundamentalValueStrategy(Strategy):
    name = "fundamental_value"
    description = "Value screen: low PE, low PB, high ROE, high profit margin"
    requires_ml = False
    min_coverage = 0.50
    min_metrics_per_stock = 2

    def rank(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return pd.DataFrame(columns=["ticker", "date", "score", "ensemble_score", "rank"])
        latest = df[df["date"] == df["date"].max()].copy()
        if "fundamental_snapshot_date" not in latest.columns:
            log.warning(
                "Fundamental strategy disabled: missing snapshot as-of metadata"
            )
            return pd.DataFrame(columns=["ticker", "date", "score", "ensemble_score", "rank"])
        snapshot_date = pd.to_datetime(
            latest["fundamental_snapshot_date"], errors="coerce"
        ).dt.normalize()
        signal_date = pd.to_datetime(latest["date"], errors="coerce").dt.normalize()
        latest = latest[snapshot_date <= signal_date].copy()
        if latest.empty:
            log.warning(
                "Fundamental strategy disabled: snapshot was retrieved after signal date"
            )
            return pd.DataFrame(columns=["ticker", "date", "score", "ensemble_score", "rank"])
        fundamental_cols = ["pe_ratio", "pb_ratio", "roe", "profit_margin"]
        available = [column for column in fundamental_cols if column in latest.columns]
        if not available:
            log.warning("Fundamental strategy disabled: no point-in-time columns available")
            return pd.DataFrame(columns=["ticker", "date", "score", "ensemble_score", "rank"])

        valid_rows = latest[available].notna().sum(axis=1) >= self.min_metrics_per_stock
        if valid_rows.mean() < self.min_coverage:
            log.warning(
                "Fundamental strategy disabled: only %.1f%% of latest stocks have "
                "%d+ valid metrics",
                valid_rows.mean() * 100,
                self.min_metrics_per_stock,
            )
            return pd.DataFrame(columns=["ticker", "date", "score", "ensemble_score", "rank"])

        scores = pd.Series(0.0, index=latest.index)
        n_metrics = len(available)
        weight = 1.0 / max(n_metrics, 1)

        if "pe_ratio" in available:
            ranked = latest["pe_ratio"].rank(pct=True)
            scores += (1 - ranked.fillna(0.5)) * weight

        if "pb_ratio" in available:
            ranked = latest["pb_ratio"].rank(pct=True)
            scores += (1 - ranked.fillna(0.5)) * weight

        if "roe" in available:
            ranked = latest["roe"].rank(pct=True)
            scores += ranked.fillna(0.5) * weight

        if "profit_margin" in available:
            ranked = latest["profit_margin"].rank(pct=True)
            scores += ranked.fillna(0.5) * weight

        latest["score"] = scores
        latest["ensemble_score"] = scores
        latest = latest.sort_values("score", ascending=False)
        latest["rank"] = range(1, len(latest) + 1)
        return latest[["ticker", "date", "score", "ensemble_score", "rank"]]
