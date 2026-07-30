from __future__ import annotations

import logging

import pandas as pd

from src.strategies.base import Strategy

log = logging.getLogger(__name__)


class FundamentalValueStrategy(Strategy):
    name = "fundamental_value"
    description = "Value screen: low PE, low PB, high ROE, high profit margin"
    requires_ml = False

    def rank(self, df: pd.DataFrame) -> pd.DataFrame:
        latest = df[df["date"] == df["date"].max()].copy()

        scores = pd.Series(0.0, index=latest.index)
        n_metrics = 4
        weight = 1.0 / max(n_metrics, 1)

        if "pe_ratio" in latest.columns:
            ranked = latest["pe_ratio"].rank(pct=True)
            scores += (1 - ranked.fillna(0.5)) * weight

        if "pb_ratio" in latest.columns:
            ranked = latest["pb_ratio"].rank(pct=True)
            scores += (1 - ranked.fillna(0.5)) * weight

        if "roe" in latest.columns:
            ranked = latest["roe"].rank(pct=True)
            scores += ranked.fillna(0.5) * weight

        if "profit_margin" in latest.columns:
            ranked = latest["profit_margin"].rank(pct=True)
            scores += ranked.fillna(0.5) * weight

        latest["score"] = scores
        latest["ensemble_score"] = scores
        latest = latest.sort_values("score", ascending=False)
        latest["rank"] = range(1, len(latest) + 1)
        return latest[["ticker", "date", "score", "ensemble_score", "rank"]]
