from __future__ import annotations

import logging
import pandas as pd

from src.features.strategy import add_strategy_features
from src.strategies.base import Strategy

log = logging.getLogger(__name__)


class BreakoutVolatilityStrategy(Strategy):
    name = "breakout_volatility"
    description = "Confirmed prior-20-session high breakout + volume >=1.8x + close position >=70%"
    requires_ml = False

    def rank(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return pd.DataFrame(columns=["ticker", "date", "score", "ensemble_score", "rank"])

        strategy_columns = {"ema_20", "ema_60", "prior_high_20d", "close_position"}
        enriched = df if strategy_columns <= set(df.columns) else add_strategy_features(df)
        latest = enriched[enriched["date"] == enriched["date"].max()].copy()
        if latest.empty:
            return pd.DataFrame(columns=["ticker", "date", "score", "ensemble_score", "rank"])

        required = ["breakout_20d", "close_position", "volume_surge"]
        if any(column not in latest.columns for column in required):
            log.warning("Breakout strategy missing confirmation features")
            return pd.DataFrame(columns=["ticker", "date", "score", "ensemble_score", "rank"])

        # A breakout strategy should not rank ordinary momentum as a breakout.
        # Require prior-high clearance, meaningful volume and a strong close.
        eligible = latest[
            (latest["breakout_20d"] > 0)
            & (latest["volume_surge"] >= 1.8)
            & (latest["close_position"] >= 0.7)
        ].copy()
        if eligible.empty:
            return pd.DataFrame(columns=["ticker", "date", "score", "ensemble_score", "rank"])

        scores = pd.Series(0.0, index=eligible.index)
        n_metrics = 0

        # 1. Short-term return momentum (5d)
        if "return_5d" in eligible.columns:
            scores += eligible["return_5d"].rank(pct=True).fillna(0.5) * 1.5
            n_metrics += 1.5

        # 2. RS 5d and 20d
        for rs_col in ["rs_5d", "rs_20d"]:
            if rs_col in eligible.columns:
                scores += eligible[rs_col].rank(pct=True).fillna(0.5)
                n_metrics += 1.0

        # 3. Volume Surge confirmation (the 1.8x gate is above).
        scores += eligible["volume_surge"].rank(pct=True).fillna(0.5) * 2.0
        n_metrics += 2.0

        # 4. Closeness to high of day (closing on highs confirms breakout).
        scores += eligible["close_position"].rank(pct=True).fillna(0.5)
        n_metrics += 1.0

        # 5. Prefer volatility expansion over a stale high-volatility name.
        if "atr_expansion" in eligible.columns:
            scores += eligible["atr_expansion"].rank(pct=True).fillna(0.5)
            n_metrics += 1.0

        if n_metrics > 0:
            scores = scores / n_metrics
        else:
            scores = pd.Series(0.5, index=eligible.index)

        eligible["score"] = scores.round(4)
        eligible["ensemble_score"] = eligible["score"]
        eligible = eligible.sort_values("score", ascending=False)
        eligible["rank"] = range(1, len(eligible) + 1)
        return eligible[["ticker", "date", "score", "ensemble_score", "rank"]]
