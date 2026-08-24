from __future__ import annotations

import logging
import pandas as pd

from src.features.strategy import add_strategy_features
from src.strategies.base import Strategy

log = logging.getLogger(__name__)


class TrendFollowingStrategy(Strategy):
    name = "trend_following"
    description = "Trend Following: EMA20/EMA60 alignment + momentum/RS + volume + ATR penalty"
    requires_ml = False

    def rank(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return pd.DataFrame(columns=["ticker", "date", "score", "ensemble_score", "rank"])

        strategy_columns = {"ema_20", "ema_60", "prior_high_20d", "close_position"}
        enriched = df if strategy_columns <= set(df.columns) else add_strategy_features(df)
        latest = enriched[enriched["date"] == enriched["date"].max()].copy()
        if latest.empty:
            return pd.DataFrame(columns=["ticker", "date", "score", "ensemble_score", "rank"])

        scores = pd.Series(0.0, index=latest.index)
        n_metrics = 0

        # 1. EMA alignment blocks sideway/weak trends without lookahead.
        if {"ema_20", "ema_60"} <= set(latest.columns):
            scores += (latest["ema_20"] > latest["ema_60"]).astype(float) * 1.5
            n_metrics += 1.5

        # 2. Medium to long-term return momentum (20d, 60d)
        for ret_col in ["return_20d", "return_60d"]:
            if ret_col in latest.columns:
                scores += latest[ret_col].rank(pct=True).fillna(0.5)
                n_metrics += 1

        # 3. Relative Strength vs Index (rs_20d, rs_60d)
        for rs_col in ["rs_20d", "rs_60d"]:
            if rs_col in latest.columns:
                scores += latest[rs_col].rank(pct=True).fillna(0.5)
                n_metrics += 1

        # 4. Trend strength / Volume confirmation
        if "volume_surge" in latest.columns:
            scores += latest["volume_surge"].rank(pct=True).fillna(0.5)
            n_metrics += 1

        # 5. Low volatility drag (use the actual pipeline feature name).
        volatility_col = "atr_pct" if "atr_pct" in latest.columns else "atr_14"
        if volatility_col in latest.columns:
            scores += (1.0 - latest[volatility_col].rank(pct=True)).fillna(0.5)
            n_metrics += 1

        if n_metrics > 0:
            scores = scores / n_metrics
        else:
            scores = pd.Series(0.5, index=latest.index)

        latest["score"] = scores.round(4)
        latest["ensemble_score"] = latest["score"]
        latest = latest.sort_values("score", ascending=False)
        latest["rank"] = range(1, len(latest) + 1)
        return latest[["ticker", "date", "score", "ensemble_score", "rank"]]
