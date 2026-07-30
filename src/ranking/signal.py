from __future__ import annotations

import logging
from datetime import date

import pandas as pd

log = logging.getLogger(__name__)

SL_DEFAULT = -0.03
TP_DEFAULT = 0.08


class SignalGenerator:
    def pick_top_n(
        self,
        ranking: pd.DataFrame,
        n: int = 5,
        weighted: bool = True,
        stop_loss: float = SL_DEFAULT,
        take_profit: float = TP_DEFAULT,
    ) -> pd.DataFrame:
        if "rank" not in ranking.columns:
            raise ValueError("DataFrame must contain 'rank' column")

        latest_date = ranking["date"].max()
        latest = ranking[ranking["date"] == latest_date].copy()
        top_n = latest[latest["rank"] <= n].copy()
        top_n["signal_date"] = date.today().isoformat()
        top_n["action"] = "BUY"

        if weighted:
            total = top_n["score"].sum()
            if total > 0:
                top_n["weight"] = (top_n["score"] / total).round(4)
            else:
                top_n["weight"] = 1.0 / n
        else:
            top_n["weight"] = 1.0 / n

        top_n["stop_loss"] = stop_loss
        top_n["take_profit"] = take_profit

        ens_cols = [c for c in ranking.columns if c.startswith("score_") and c.endswith("d")]
        if ens_cols:
            top_n["ensemble_score"] = top_n[ens_cols].mean(axis=1)
        else:
            top_n["ensemble_score"] = top_n["score"]

        log.info("Top %d picks for %s: %s", n, date.today().isoformat(), list(top_n["ticker"]))
        if weighted:
            log.info("Weights: %s", dict(zip(top_n["ticker"], top_n["weight"])))
        return top_n[["signal_date", "date", "rank", "ticker", "score", "ensemble_score", "weight", "action", "stop_loss", "take_profit"]]
