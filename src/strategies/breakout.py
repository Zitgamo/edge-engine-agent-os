from __future__ import annotations

import logging

import pandas as pd

from src.strategies.base import Strategy

log = logging.getLogger(__name__)


class BreakoutStrategy(Strategy):
    name = "breakout"
    description = "Breakout: high RS + high return_5d + volume surge + close near high"
    requires_ml = False

    def rank(self, df: pd.DataFrame) -> pd.DataFrame:
        latest = df[df["date"] == df["date"].max()].copy()
        scores = pd.Series(0.0, index=latest.index)
        n_metrics = 0

        rs_cols = [c for c in latest.columns if c.startswith("rs_")]
        for c in rs_cols:
            ranked = latest[c].rank(pct=True)
            scores += ranked.fillna(0.5)
            n_metrics += 1

        ret_cols = [c for c in latest.columns if c.startswith("return_") and "excess" not in c]
        for c in ret_cols:
            ranked = latest[c].rank(pct=True)
            scores += ranked.fillna(0.5)
            n_metrics += 1

        if "volume_surge" in latest.columns:
            ranked = latest["volume_surge"].rank(pct=True)
            scores += ranked.fillna(0.5)
            n_metrics += 1

        close_near_high = (latest["close"] / latest["high"]).rank(pct=True) if "close" in latest.columns and "high" in latest.columns else 0
        scores += close_near_high.fillna(0.5)
        n_metrics += 1

        scores = scores / max(n_metrics, 1)

        latest["score"] = scores
        latest["ensemble_score"] = scores
        latest = latest.sort_values("score", ascending=False)
        latest["rank"] = range(1, len(latest) + 1)
        return latest[["ticker", "date", "score", "ensemble_score", "rank"]]
