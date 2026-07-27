from __future__ import annotations

import logging

import pandas as pd

from src.strategies.base import Strategy

log = logging.getLogger(__name__)


class MeanReversionStrategy(Strategy):
    name = "mean_reversion"
    description = "Oversold bounce: low RS + low returns + high ATR"
    requires_ml = False

    def rank(self, df: pd.DataFrame) -> pd.DataFrame:
        latest = df[df["date"] == df["date"].max()].copy()
        rs_cols = [c for c in latest.columns if c.startswith("rs_")]
        ret_cols = [c for c in latest.columns if c.startswith("return_") and "excess" not in c]
        scores = pd.Series(0.0, index=latest.index)
        n_metrics = len(rs_cols) + len(ret_cols)
        if "atr_pct" in latest.columns:
            n_metrics += 1
        weight = 1.0 / n_metrics

        for c in rs_cols:
            ranked = latest[c].rank(pct=True)
            scores += (1 - ranked.fillna(0.5)) * weight

        for c in ret_cols:
            ranked = latest[c].rank(pct=True)
            scores += (1 - ranked.fillna(0.5)) * weight

        if "atr_pct" in latest.columns:
            ranked_atr = latest["atr_pct"].rank(pct=True)
            scores += ranked_atr.fillna(0.5) * weight

        latest["score"] = scores
        latest["ensemble_score"] = scores
        latest = latest.sort_values("score", ascending=False)
        latest["rank"] = range(1, len(latest) + 1)
        return latest[["ticker", "date", "score", "ensemble_score", "rank"]]
