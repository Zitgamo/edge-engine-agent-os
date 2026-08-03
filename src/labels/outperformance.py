from __future__ import annotations

import logging

import numpy as np
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

        label_col = f"outperform_{horizon}d"
        excess_col = f"excess_return_{horizon}d"
        valid = merged[["close", "bm_close", "stock_future", "bm_future"]].notna().all(axis=1)
        valid &= merged["close"].ne(0) & merged["bm_close"].ne(0)

        merged[label_col] = np.nan
        merged.loc[valid, label_col] = (merged.loc[valid, "stock_ret"] > merged.loc[valid, "bm_ret"]).astype(float)
        merged[f"excess_return_{horizon}d"] = merged["stock_ret"] - merged["bm_ret"]
        merged.loc[~valid, excess_col] = np.nan

        result = stock_df.merge(
            merged[["date", "ticker", label_col, excess_col]],
            on=["date", "ticker"],
            how="left",
        )
        log.info("Computed T+%d outperform label", horizon)
        return result
