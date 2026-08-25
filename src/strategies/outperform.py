from __future__ import annotations

import logging

import pandas as pd

from src.strategies.base import Strategy

log = logging.getLogger(__name__)


class OutperformStrategy(Strategy):
    name = "outperform"
    description = "Execution-aligned T+20 XGBoost ranking with quality and entry gates"
    requires_ml = True

    def rank(self, df: pd.DataFrame) -> pd.DataFrame:
        result = df.copy()
        if "score" not in result.columns:
            result["score"] = result["ensemble_score"] if "ensemble_score" in result.columns else 0.5
        if "ensemble_score" not in result.columns:
            result["ensemble_score"] = result["score"]

        latest = result[result["date"] == result["date"].max()].copy()
        latest = latest.sort_values("score", ascending=False)
        latest["rank"] = range(1, len(latest) + 1)
        return latest[["ticker", "date", "score", "ensemble_score", "rank"]]
