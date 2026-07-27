from __future__ import annotations

import logging

import pandas as pd

from src.strategies.base import Strategy

log = logging.getLogger(__name__)


class RSMomentumStrategy(Strategy):
    name = "rs_momentum"
    description = "Pure relative strength + momentum (no ML)"
    requires_ml = False

    def rank(self, df: pd.DataFrame) -> pd.DataFrame:
        latest = df[df["date"] == df["date"].max()].copy()
        rs_cols = [c for c in latest.columns if c.startswith("rs_")]
        ret_cols = [c for c in latest.columns if c.startswith("return_") and "excess" not in c]

        scores = pd.Series(0.0, index=latest.index)
        weight = 1.0 / (len(rs_cols) + len(ret_cols))

        for c in rs_cols:
            ranked = latest[c].rank(pct=True)
            scores += ranked.fillna(0.5) * weight

        for c in ret_cols:
            ranked = latest[c].rank(pct=True)
            scores += ranked.fillna(0.5) * weight

        latest["score"] = scores
        latest["ensemble_score"] = scores
        latest = latest.sort_values("score", ascending=False)
        latest["rank"] = range(1, len(latest) + 1)
        return latest[["ticker", "date", "score", "ensemble_score", "rank"]]
