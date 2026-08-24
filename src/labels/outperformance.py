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
        if horizon <= 0:
            raise ValueError("horizon must be positive")

        df = stock_df.sort_values("date")[["date", "ticker", "close"]].copy()
        df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.normalize()
        df = df.dropna(subset=["date"]).drop_duplicates(subset=["date"], keep="last")
        bm = benchmark_df.sort_values("date")[["date", "close"]].copy()
        bm["date"] = pd.to_datetime(bm["date"], errors="coerce").dt.normalize()
        bm = (
            bm.dropna(subset=["date"])
            .drop_duplicates(subset=["date"], keep="last")
            .rename(columns={"close": "bm_close"})
        )

        # T+N is N stock trading sessions. Look up the benchmark at the
        # actual future stock date instead of shifting merged rows: the two
        # calendars can have different starts and missing sessions.
        merged = df.copy()
        merged["future_date"] = merged["date"].shift(-horizon)
        merged["stock_future"] = merged["close"].shift(-horizon)
        merged = merged.merge(bm, on="date", how="left")
        future_bm = bm.rename(columns={"date": "future_date", "bm_close": "bm_future"})
        merged = merged.merge(future_bm, on="future_date", how="left")

        merged["stock_ret"] = (merged["stock_future"] - merged["close"]) / merged["close"]
        merged["bm_ret"] = (merged["bm_future"] - merged["bm_close"]) / merged["bm_close"]

        label_col = f"outperform_{horizon}d"
        excess_col = f"excess_return_{horizon}d"
        label_end_col = f"label_end_date_{horizon}d"
        valid = merged[["close", "bm_close", "stock_future", "bm_future"]].notna().all(axis=1)
        valid &= merged["close"].ne(0) & merged["bm_close"].ne(0)

        merged[label_col] = np.nan
        merged.loc[valid, label_col] = (merged.loc[valid, "stock_ret"] > merged.loc[valid, "bm_ret"]).astype(float)
        merged[f"excess_return_{horizon}d"] = merged["stock_ret"] - merged["bm_ret"]
        merged.loc[~valid, excess_col] = np.nan
        # A label is not available until its future stock session has closed.
        # Keep this endpoint so time-based splits can purge labels that would
        # still be using prices from the test period.
        merged[label_end_col] = merged["future_date"]

        result = stock_df.copy()
        result["date"] = pd.to_datetime(result["date"], errors="coerce").dt.normalize()
        result = result.merge(
            merged[["date", "ticker", label_col, excess_col, label_end_col]],
            on=["date", "ticker"],
            how="left",
        )
        log.info("Computed T+%d outperform label", horizon)
        return result
