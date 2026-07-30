from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from src.strategies.base import Strategy

log = logging.getLogger(__name__)


class RSIStrategy(Strategy):
    name = "rsi"
    description = "RSI momentum: oversold bounce + recent upside reversal"
    requires_ml = False

    def rank(self, df: pd.DataFrame) -> pd.DataFrame:
        latest = df[df["date"] == df["date"].max()].copy()
        scores = pd.Series(0.0, index=latest.index)
        n_metrics = 0

        ret_5d = latest["return_5d"].rank(pct=True) if "return_5d" in latest.columns else 0
        ret_20d = latest["return_20d"].rank(pct=True) if "return_20d" in latest.columns else 0

        rsi_5d = 100 - (100 / (1 + np.exp(-ret_5d.fillna(0) * 10)))
        oversold = (rsi_5d < 30).astype(float)
        oversold_score = oversold * (1 - latest["return_5d"].rank(pct=True))
        scores += oversold_score.fillna(0)
        n_metrics += 1

        recent_ret = latest["return_5d"].fillna(0)
        medium_ret = latest["return_20d"].fillna(0)
        reversal = ((recent_ret > medium_ret) & (medium_ret < 0)).astype(float)
        reversal_score = reversal * recent_ret.rank(pct=True)
        scores += reversal_score.fillna(0)
        n_metrics += 1

        if "atr_pct" in latest.columns:
            ranked_atr = latest["atr_pct"].rank(pct=True)
            scores += ranked_atr.fillna(0.5)
            n_metrics += 1

        if "volume_surge" in latest.columns:
            ranked_vol = latest["volume_surge"].rank(pct=True)
            scores += ranked_vol.fillna(0.5)
            n_metrics += 1

        scores = scores / n_metrics

        latest["score"] = scores
        latest["ensemble_score"] = scores
        latest = latest.sort_values("score", ascending=False)
        latest["rank"] = range(1, len(latest) + 1)
        return latest[["ticker", "date", "score", "ensemble_score", "rank"]]
