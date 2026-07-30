from __future__ import annotations

import logging

import pandas as pd

from src.strategies.base import Strategy

log = logging.getLogger(__name__)


class AccumulationStrategy(Strategy):
    name = "accumulation"
    description = "Long-term accumulation: high ROE, low PE/PB, stable growth, low volatility"
    requires_ml = False

    def rank(self, df: pd.DataFrame) -> pd.DataFrame:
        latest = df[df["date"] == df["date"].max()].copy()

        scores = pd.Series(0.0, index=latest.index)
        n_metrics = 0

        if "roe" in latest.columns:
            ranked = latest["roe"].rank(pct=True)
            scores += ranked.fillna(0.5)
            n_metrics += 1

        if "profit_margin" in latest.columns:
            ranked = latest["profit_margin"].rank(pct=True)
            scores += ranked.fillna(0.5)
            n_metrics += 1

        if "rev_growth" in latest.columns:
            ranked = latest["rev_growth"].rank(pct=True)
            scores += ranked.fillna(0.5)
            n_metrics += 1

        if "earn_growth" in latest.columns:
            ranked = latest["earn_growth"].rank(pct=True)
            scores += ranked.fillna(0.5)
            n_metrics += 1

        if "pe_ratio" in latest.columns:
            ranked = latest["pe_ratio"].rank(pct=True)
            scores += (1 - ranked.fillna(0.5))
            n_metrics += 1

        if "pb_ratio" in latest.columns:
            ranked = latest["pb_ratio"].rank(pct=True)
            scores += (1 - ranked.fillna(0.5))
            n_metrics += 1

        if "debt_equity" in latest.columns:
            ranked = latest["debt_equity"].rank(pct=True)
            scores += (1 - ranked.fillna(0.5))
            n_metrics += 1

        if "div_yield" in latest.columns:
            ranked = latest["div_yield"].rank(pct=True)
            scores += ranked.fillna(0.5)
            n_metrics += 1

        if "log_mcap" in latest.columns:
            ranked = latest["log_mcap"].rank(pct=True)
            scores += ranked.fillna(0.5)
            n_metrics += 1

        if "return_60d" in df.columns and "ticker" in df.columns:
            vol_by_ticker = df.groupby("ticker")["return_60d"].transform(lambda s: s.rolling(60, min_periods=20).std())
            latest_vol = vol_by_ticker.loc[latest.index]
            if latest_vol.notna().any():
                ranked_vol = latest_vol.rank(pct=True)
                scores += (1 - ranked_vol.fillna(0.5))
                n_metrics += 1

        if n_metrics > 0:
            scores /= n_metrics

        latest["score"] = scores
        latest["ensemble_score"] = scores
        latest = latest.sort_values("score", ascending=False)
        latest["rank"] = range(1, len(latest) + 1)
        return latest[["ticker", "date", "score", "ensemble_score", "rank"]]
