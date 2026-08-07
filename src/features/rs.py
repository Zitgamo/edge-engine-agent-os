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
        stock = (
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

        # Build returns only on the intersection of stock and benchmark
        # sessions.  Rolling over the stock calendar with missing benchmark
        # rows makes one holiday invalidate the next 60 observations, which
        # can remove almost the entire training window from the pipeline.
        aligned = stock[["date", "close"]].rename(columns={"close": "stock_close"}).merge(
            bm[["date", "close"]].rename(columns={"close": "bm_close"}),
            on="date",
            how="inner",
            sort=True,
        )
        aligned["stock_ret"] = aligned["stock_close"].pct_change()
        aligned["bm_ret"] = aligned["bm_close"].pct_change()

        for window in (5, 20, 60):
            aligned[f"rs_{window}d"] = (
                aligned["stock_ret"].rolling(window).sum()
                - aligned["bm_ret"].rolling(window).sum()
            )

        result = stock.merge(
            aligned[["date", "rs_5d", "rs_20d", "rs_60d"]],
            on="date",
            how="left",
            sort=False,
        )

        log.info("Computed RS metrics")
        return result
