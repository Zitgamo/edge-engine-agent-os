from __future__ import annotations

import pandas as pd
import pytest

from src.ranking.signal import SignalGenerator


class TestSignalGenerator:
    def test_picks_top_3(self) -> None:
        dates = pd.date_range("2024-01-01", periods=2, freq="B")
        rows = []
        for d in dates:
            for i in range(20):
                rows.append({"date": d, "ticker": f"S{i:02d}", "score": (20 - i) / 20.0, "rank": i + 1})
        ranking = pd.DataFrame(rows)
        result = SignalGenerator().pick_top_n(ranking, n=3)
        assert len(result) == 3
        assert list(result["rank"]) == [1, 2, 3]
        assert "action" in result.columns
        assert all(result["action"] == "BUY")

    def test_picks_only_latest_date(self) -> None:
        dates = pd.date_range("2024-01-01", periods=3, freq="B")
        rows = []
        for d in dates:
            for i in range(5):
                rows.append({"date": d, "ticker": f"S{i:02d}", "score": (5 - i) / 5.0, "rank": i + 1})
        ranking = pd.DataFrame(rows)
        result = SignalGenerator().pick_top_n(ranking, n=2)
        assert all(result["date"] == dates[-1])

    def test_raises_without_rank_column(self) -> None:
        df = pd.DataFrame({"ticker": ["VNM"], "date": pd.to_datetime("2024-01-01")})
        with pytest.raises(ValueError, match="rank"):
            SignalGenerator().pick_top_n(df)
