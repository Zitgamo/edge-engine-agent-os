from __future__ import annotations

import logging

import pandas as pd

from src.strategies.base import Strategy

log = logging.getLogger(__name__)


class MomentumStrategy(Strategy):
    name = "momentum"
    description = "Pure momentum: high returns + volume surge, no RS"
    requires_ml = False

    def rank(self, df: pd.DataFrame) -> pd.DataFrame:
        latest = df[df["date"] == df["date"].max()].copy()

        scores = pd.Series(0.0, index=latest.index)
        ret_cols = [c for c in latest.columns if c.startswith("return_") and "excess" not in c]
        n_metrics = len(ret_cols)
        has_vol = "volume_surge" in latest.columns
        if has_vol:
            n_metrics += 1
        weight = 1.0 / max(n_metrics, 1)

        for c in ret_cols:
            ranked = latest[c].rank(pct=True)
            scores += ranked.fillna(0.5) * weight

        if has_vol:
            ranked_vol = latest["volume_surge"].rank(pct=True)
            scores += ranked_vol.fillna(0.5) * weight

        latest["score"] = scores
        latest["ensemble_score"] = scores
        latest = latest.sort_values("score", ascending=False)
        latest["rank"] = range(1, len(latest) + 1)
        return latest[["ticker", "date", "score", "ensemble_score", "rank"]]
