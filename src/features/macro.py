from __future__ import annotations

import logging

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
        """Return an empty frame until an as-of-tagged SBV source is wired.

        A constant placeholder rate is not a market observation and can make
        a historical model appear to have macro coverage it never had.
        """
        log.warning("SBV rate source is unavailable; leaving sbv_rate missing")
        return pd.DataFrame(columns=["date", "sbv_rate"])

    def fetch_cpi(self) -> pd.DataFrame:
        """Return an empty frame until CPI release dates are available."""
        log.warning("CPI source is unavailable; leaving cpi_mom missing")
        return pd.DataFrame(columns=["date", "cpi_mom"])

    def fetch_all(self, days: int = 365) -> pd.DataFrame:
        vnusd = self.fetch_vnusd(days)
        sbv = self.fetch_sbv_rate()
        cpi = self.fetch_cpi()

        # Normalize all dates to tz-naive date-only.
        def to_date_str(series):
            return pd.to_datetime(series).dt.tz_localize(None).dt.normalize()

        frames = []
        for frame in (vnusd, sbv, cpi):
            if frame.empty:
                continue
            frame = frame.copy()
            frame["date"] = to_date_str(frame["date"])
            frames.append(frame)
        if not frames:
            return pd.DataFrame()

        merged = frames[0]
        for frame in frames[1:]:
            merged = merged.merge(frame, on="date", how="outer")
        merged = merged.sort_values("date").drop_duplicates("date")

        for col in ["vndusd", "sbv_rate", "cpi_mom"]:
            if col not in merged.columns:
                merged[col] = pd.NA
            # Forward-fill only observations already released.  There is no
            # backfill from a later macro release into an earlier row.
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
    # macro values into historical rows and across ticker boundaries.  Zero is
    # reserved for the explicit unavailable-data fallback, not a synthetic
    # observation.
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
