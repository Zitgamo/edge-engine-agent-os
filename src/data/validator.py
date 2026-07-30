from __future__ import annotations

import logging

import pandas as pd

log = logging.getLogger(__name__)

REQUIRED_COLUMNS = {"ticker", "date", "open", "high", "low", "close", "volume"}


class DataValidator:
    def validate(self, df: pd.DataFrame) -> list[str]:
        errors: list[str] = []

        if df.empty:
            errors.append("DataFrame is empty")
            return errors

        missing = REQUIRED_COLUMNS - set(df.columns)
        if missing:
            errors.append(f"Missing columns: {missing}")

        if "date" in df.columns:
            if not pd.api.types.is_datetime64_any_dtype(df["date"]):
                errors.append("date column is not datetime")
            if df["date"].isna().any():
                errors.append("date column contains NaT")

        for col in ["open", "high", "low", "close", "volume"]:
            if col in df.columns and df[col].isna().any():
                errors.append(f"{col} column contains NaN")

        for col in ["open", "high", "low", "close"]:
            if col in df.columns and (df[col] <= 0).any():
                non_pos = int((df[col] <= 0).sum())
                errors.append(f"{col} column has {non_pos} non-positive values")

        # OHLC consistency: high >= low, high >= open/close, low <= open/close
        if {"high", "low"}.issubset(df.columns):
            if (df["high"] < df["low"]).any():
                n = int((df["high"] < df["low"]).sum())
                errors.append(f"{n} rows where high < low")
        if {"high", "close"}.issubset(df.columns):
            if (df["high"] < df["close"]).any():
                n = int((df["high"] < df["close"]).sum())
                errors.append(f"{n} rows where high < close")
        if {"low", "close"}.issubset(df.columns):
            if (df["low"] > df["close"]).any():
                n = int((df["low"] > df["close"]).sum())
                errors.append(f"{n} rows where low > close")

        if errors:
            log.warning("Validation errors: %s", errors)
        else:
            log.info("Data validation passed")

        return errors
