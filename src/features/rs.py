from __future__ import annotations

import logging

import pandas as pd

log = logging.getLogger(__name__)


class RelativeStrength:
    def compute(self, df: pd.DataFrame, benchmark: pd.DataFrame) -> pd.DataFrame:
        df = df.sort_values("date").copy()
        bm = benchmark.sort_values("date").copy()

        bm_returns = bm["close"].pct_change()
        stock_returns = df["close"].pct_change()

        merged = pd.merge(
            df[["date"]],
            bm[["date", "close"]].rename(columns={"close": "bm_close"}),
            on="date",
            how="left",
        )
        merged["stock_ret"] = stock_returns
        merged["bm_ret"] = bm_returns

        merged["rs_5d"] = merged["stock_ret"].rolling(5).sum() - merged["bm_ret"].rolling(5).sum()
        merged["rs_20d"] = merged["stock_ret"].rolling(20).sum() - merged["bm_ret"].rolling(20).sum()
        merged["rs_60d"] = merged["stock_ret"].rolling(60).sum() - merged["bm_ret"].rolling(60).sum()

        df["rs_5d"] = merged["rs_5d"]
        df["rs_20d"] = merged["rs_20d"]
        df["rs_60d"] = merged["rs_60d"]

        log.info("Computed RS metrics")
        return df
