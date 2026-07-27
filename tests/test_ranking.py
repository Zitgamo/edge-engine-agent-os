from __future__ import annotations

import pandas as pd
import pytest

from src.ranking.ranker import Ranker


class TestRanker:
    def test_returns_top20_per_date(self) -> None:
        dates = pd.date_range("2024-01-01", periods=3, freq="B")
        rows = []
        for d in dates:
            for i in range(30):
                rows.append({"date": d, "ticker": f"STOCK{i:02d}", "score": i / 30.0})
        df = pd.DataFrame(rows)

        result = Ranker().generate_top20(df)
        assert len(result) == 60  # 3 days * 20
        assert "rank" in result.columns
        assert result["rank"].max() == 20

    def test_raises_without_score_column(self) -> None:
        df = pd.DataFrame({"ticker": ["VNM"], "date": pd.to_datetime("2024-01-01")})
        with pytest.raises(ValueError, match="score"):
            Ranker().generate_top20(df)
