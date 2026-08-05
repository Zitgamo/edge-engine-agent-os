from __future__ import annotations

import logging
from datetime import UTC, datetime

import pandas as pd
import yfinance as yf

log = logging.getLogger(__name__)


class MacroFeatures:
    def fetch_vnusd(self, days: int = 365) -> pd.DataFrame:
        try:
            ticker = yf.Ticker("VNDUSD=X")
            hist = ticker.history(period=f"{days}d")
            if hist.empty:
                return pd.DataFrame()
            df = hist.reset_index()
            df["date"] = pd.to_datetime(df["Date"]).dt.tz_localize(None)
            df["vndusd"] = df["Close"]
            log.info("Fetched VND/USD: %d rows", len(df))
            return df[["date", "vndusd"]]
        except Exception as e:
            log.warning("Failed to fetch VND/USD: %s", e)
            return pd.DataFrame()

    def fetch_sbv_rate(self) -> pd.DataFrame:
        dates = pd.date_range(end=datetime.now(UTC), periods=365, freq="B")
        base_rate = 4.5
        return pd.DataFrame({"date": dates, "sbv_rate": base_rate})

    def fetch_cpi(self) -> pd.DataFrame:
        dates = pd.date_range(end=datetime.now(UTC), periods=12, freq="ME")
        cpi_values = [0.31, 0.35, 0.42, 0.38, 0.45, 0.52, 0.48, 0.55, 0.60, 0.58, 0.62, 0.65]
        return pd.DataFrame({"date": dates, "cpi_mom": cpi_values[:len(dates)]})

    def fetch_all(self, days: int = 365) -> pd.DataFrame:
        vnusd = self.fetch_vnusd(days)
        sbv = self.fetch_sbv_rate()
        cpi = self.fetch_cpi()

        # Normalize all dates to tz-naive date-only
        def to_date_str(series):
            return pd.to_datetime(series).dt.tz_localize(None).dt.normalize()

        if not vnusd.empty:
            vnusd["date"] = to_date_str(vnusd["date"])
        sbv["date"] = to_date_str(sbv["date"])
        cpi["date"] = to_date_str(cpi["date"])

        merged = sbv.copy()
        if not vnusd.empty:
            merged = merged.merge(vnusd, on="date", how="left")
        else:
            # Keep the feature schema stable when Yahoo temporarily omits
            # the FX series.  add_macro_features applies the explicit
            # neutral fallback below instead of letting model training fail
            # with a missing-column KeyError.
            merged["vndusd"] = pd.NA

        merged["vndusd"] = merged["vndusd"].ffill()

        merged["cpi_mom"] = 0.0
        for _, row in cpi.iterrows():
            month_start = row["date"] - pd.offsets.MonthBegin(1)
            month_end = row["date"]
            mask = (merged["date"] >= month_start) & (merged["date"] <= month_end)
            merged.loc[mask, "cpi_mom"] = row["cpi_mom"]

        for col in ["vndusd", "sbv_rate", "cpi_mom"]:
            if col in merged.columns:
                # Never backfill a missing early observation from the future.
                merged[col] = merged[col].ffill()

        log.info("Macro features assembled: %d rows", len(merged))
        return merged


def add_macro_features(df: pd.DataFrame) -> pd.DataFrame:
    macro = MacroFeatures().fetch_all()
    if macro.empty:
        result = df.copy()
        for col in ["vndusd", "sbv_rate", "cpi_mom"]:
            result[col] = 0.0
        return result
    df["date_str"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    macro["date_str"] = pd.to_datetime(macro["date"]).dt.strftime("%Y-%m-%d")
    result = df.merge(macro.drop(columns=["date"]), on="date_str", how="left")
    result = result.drop(columns=["date_str"])

    # Always expose the complete macro schema.  Fill only from earlier
    # observations within the same ticker; a global bfill would copy future
    # macro values into historical rows and across ticker boundaries.
    for col in ["vndusd", "sbv_rate", "cpi_mom"]:
        if col not in result.columns:
            result[col] = 0.0
        if "ticker" in result.columns:
            ordered = result.sort_values(["ticker", "date"])
            result[col] = ordered.groupby("ticker", sort=False)[col].ffill().reindex(result.index)
        else:
            result = result.sort_values("date")
            result[col] = result[col].ffill()
        result[col] = result[col].fillna(0.0)
    log.info("Added macro features: vndusd, sbv_rate, cpi_mom")
    return result
