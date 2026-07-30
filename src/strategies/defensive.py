from __future__ import annotations

import logging

import pandas as pd

from src.strategies.base import Strategy

log = logging.getLogger(__name__)


class DefensiveStrategy(Strategy):
    name = "defensive"
    description = "Defensive: low volatility + high dividend + stable ROE + low debt"
    requires_ml = False

    def rank(self, df: pd.DataFrame) -> pd.DataFrame:
        latest = df[df["date"] == df["date"].max()].copy()
        scores = pd.Series(0.0, index=latest.index)
        n_metrics = 0

        if "atr_pct" in latest.columns:
            ranked = 1 - latest["atr_pct"].rank(pct=True)
            scores += ranked.fillna(0.5)
            n_metrics += 1

        return_cols = [c for c in latest.columns if c.startswith("return_") and "excess" not in c]
        for c in return_cols:
            ranked = 1 - latest[c].rank(pct=True)
            scores += ranked.fillna(0.5)
            n_metrics += 1

        if "div_yield" in latest.columns:
            ranked = latest["div_yield"].rank(pct=True)
            scores += ranked.fillna(0.5)
            n_metrics += 1

        if "roe" in latest.columns:
            ranked = latest["roe"].rank(pct=True)
            scores += ranked.fillna(0.5)
            n_metrics += 1

        if "debt_equity" in latest.columns:
            ranked = 1 - latest["debt_equity"].rank(pct=True)
            scores += ranked.fillna(0.5)
            n_metrics += 1

        if "profit_margin" in latest.columns:
            ranked = latest["profit_margin"].rank(pct=True)
            scores += ranked.fillna(0.5)
            n_metrics += 1

        if "log_mcap" in latest.columns:
            ranked = latest["log_mcap"].rank(pct=True)
            scores += ranked.fillna(0.5)
            n_metrics += 1

        scores = scores / max(n_metrics, 1)

        latest["score"] = scores
        latest["ensemble_score"] = scores
        latest = latest.sort_values("score", ascending=False)
        latest["rank"] = range(1, len(latest) + 1)
        return latest[["ticker", "date", "score", "ensemble_score", "rank"]]
