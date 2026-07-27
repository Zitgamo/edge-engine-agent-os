from __future__ import annotations

import logging

import pandas as pd

log = logging.getLogger(__name__)


class Ranker:
    def generate_top20(self, df: pd.DataFrame) -> pd.DataFrame:
        if "score" not in df.columns:
            raise ValueError("DataFrame must contain 'score' column")

        ranked = (
            df.sort_values("score", ascending=False)
            .groupby("date")
            .head(20)
            .reset_index(drop=True)
        )
        ranked["rank"] = ranked.groupby("date").cumcount() + 1
        log.info("Generated Top20 ranking")
        return ranked
