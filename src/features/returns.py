from __future__ import annotations

import logging
from typing import ClassVar

import pandas as pd

log = logging.getLogger(__name__)


class ReturnFeatures:
    WINDOWS: ClassVar[list[int]] = [5, 20, 60]

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.sort_values("date").copy()
        for w in self.WINDOWS:
            df[f"return_{w}d"] = df["close"].pct_change(w)
        log.info("Computed return features: %s", [f"return_{w}d" for w in self.WINDOWS])
        return df
