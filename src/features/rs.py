from __future__ import annotations

import logging

import pandas as pd

log = logging.getLogger(__name__)


class RelativeStrength:
    def compute(self, df: pd.DataFrame, benchmark: pd.DataFrame) -> pd.DataFrame:
        # Returns must be aligned by trading date.  The stock and benchmark
        # series often have different start dates, holidays, or missing
        # sessions; assigning benchmark returns by row position silently
        # shifts the benchmark history in those cases.
        df = (
            df.sort_values("date")
            .drop_duplicates(subset=["date"], keep="last")
            .reset_index(drop=True)
            .copy()
        )
        bm = (
            benchmark[["date", "close"]]
            .sort_values("date")
            .drop_duplicates(subset=["date"], keep="last")
            .reset_index(drop=True)
            .copy()
        )

        stock_returns = df["close"].pct_change()
        bm["bm_ret"] = bm["close"].pct_change()

        merged = df[["date"]].merge(
            bm[["date", "bm_ret"]],
            on="date",
            how="left",
            sort=False,
        )
        merged["stock_ret"] = stock_returns.to_numpy()

        merged["rs_5d"] = merged["stock_ret"].rolling(5).sum() - merged["bm_ret"].rolling(5).sum()
        merged["rs_20d"] = merged["stock_ret"].rolling(20).sum() - merged["bm_ret"].rolling(20).sum()
        merged["rs_60d"] = merged["stock_ret"].rolling(60).sum() - merged["bm_ret"].rolling(60).sum()

        df["rs_5d"] = merged["rs_5d"]
        df["rs_20d"] = merged["rs_20d"]
        df["rs_60d"] = merged["rs_60d"]

        log.info("Computed RS metrics")
        return df
