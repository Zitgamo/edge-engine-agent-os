from __future__ import annotations

import logging

import pandas as pd

log = logging.getLogger(__name__)


class ATR:
    def compute(self, df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        df = df.sort_values("date").copy()

        high_low = df["high"] - df["low"]
        high_close = (df["high"] - df["close"].shift()).abs()
        low_close = (df["low"] - df["close"].shift()).abs()

        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df["atr"] = tr.rolling(period).mean()
        df["atr_pct"] = df["atr"] / df["close"]

        log.info("Computed ATR (period=%d)", period)
        return df
