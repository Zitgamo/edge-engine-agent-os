from __future__ import annotations

import logging

import pandas as pd

log = logging.getLogger(__name__)


class OutperformanceLabel:
    def compute(
        self,
        stock_df: pd.DataFrame,
        benchmark_df: pd.DataFrame,
        horizon: int = 5,
    ) -> pd.DataFrame:
        df = stock_df.sort_values("date")[["date", "ticker", "close"]].copy()
        bm = benchmark_df.sort_values("date")[["date", "close"]].copy().rename(columns={"close": "bm_close"})

        merged = df.merge(bm, on="date", how="left")
        merged["stock_future"] = merged["close"].shift(-horizon)
        merged["bm_future"] = merged["bm_close"].shift(-horizon)

        merged["stock_ret"] = (merged["stock_future"] - merged["close"]) / merged["close"]
        merged["bm_ret"] = (merged["bm_future"] - merged["bm_close"]) / merged["bm_close"]

        merged[f"outperform_{horizon}d"] = (merged["stock_ret"] > merged["bm_ret"]).astype(int)
        merged[f"excess_return_{horizon}d"] = merged["stock_ret"] - merged["bm_ret"]

        result = stock_df.merge(
            merged[["date", "ticker", f"outperform_{horizon}d", f"excess_return_{horizon}d"]],
            on=["date", "ticker"],
            how="left",
        )
        log.info("Computed T+%d outperform label", horizon)
        return result
