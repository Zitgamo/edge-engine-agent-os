from __future__ import annotations

import logging

import pandas as pd

log = logging.getLogger(__name__)


def _normalise_dates(values: pd.Series) -> pd.Series:
    dates = pd.to_datetime(values, errors="coerce")
    if dates.dt.tz is not None:
        dates = dates.dt.tz_localize(None)
    return dates.dt.normalize()


class RelativeStrength:
    def compute(self, df: pd.DataFrame, benchmark: pd.DataFrame) -> pd.DataFrame:
        for name, frame in (("stock", df), ("benchmark", benchmark)):
            missing = {"date", "close"} - set(frame.columns)
            if missing:
                raise ValueError(f"{name} data is missing columns: {sorted(missing)}")

        # Returns must be aligned by trading date.  The stock and benchmark
        # series often have different start dates, holidays, or missing
        # sessions; assigning benchmark returns by row position silently
        # shifts the benchmark history in those cases.
        stock_input = df.copy()
        stock_input["date"] = _normalise_dates(stock_input["date"])
        stock = (
            stock_input.dropna(subset=["date", "close"])
            .sort_values("date")
            .drop_duplicates(subset=["date"], keep="last")
            .reset_index(drop=True)
            .copy()
        )
        benchmark_input = benchmark[["date", "close"]].copy()
        benchmark_input["date"] = _normalise_dates(benchmark_input["date"])
        bm = (
            benchmark_input.dropna(subset=["date", "close"])
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
                aligned["stock_ret"].rolling(window, min_periods=window).sum()
                - aligned["bm_ret"].rolling(window, min_periods=window).sum()
            )

        result = stock.merge(
            aligned[["date", "rs_5d", "rs_20d", "rs_60d"]],
            on="date",
            how="left",
            sort=False,
        )

        log.info("Computed RS metrics")
        return result
