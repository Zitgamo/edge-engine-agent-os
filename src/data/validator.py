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
                errors.append(f"{col} column has non-positive values")

        if errors:
            log.warning("Validation errors: %s", errors)
        else:
            log.info("Data validation passed")

        return errors
