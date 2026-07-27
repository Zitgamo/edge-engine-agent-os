from __future__ import annotations

import logging

import pandas as pd

log = logging.getLogger(__name__)


class VolumeSurge:
    def compute(self, df: pd.DataFrame, window: int = 20, threshold: float = 1.5) -> pd.DataFrame:
        df = df.sort_values("date").copy()

        df["volume_ma"] = df["volume"].rolling(window).mean()
        df["volume_surge"] = df["volume"] / df["volume_ma"]
        df["volume_surge_flag"] = (df["volume_surge"] > threshold).astype(int)

        log.info("Computed volume surge (window=%d, threshold=%.1f)", window, threshold)
        return df
