from __future__ import annotations

import pandas as pd
import pytest

from src.labels.outperformance import OutperformanceLabel


@pytest.fixture
def stock_df() -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=50, freq="B")
    return pd.DataFrame({
        "ticker": ["VNM"] * 50,
        "date": dates,
        "close": range(100, 150),
    })


@pytest.fixture
def benchmark_df() -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=50, freq="B")
    return pd.DataFrame({
        "date": dates,
        "close": range(1000, 1050),
    })


class TestOutperformanceLabel:
    def test_computes_label_column(self, stock_df: pd.DataFrame, benchmark_df: pd.DataFrame) -> None:
        result = OutperformanceLabel().compute(stock_df, benchmark_df)
        assert "outperform_5d" in result.columns
        assert "excess_return_5d" in result.columns

    def test_label_is_binary(self, stock_df: pd.DataFrame, benchmark_df: pd.DataFrame) -> None:
        result = OutperformanceLabel().compute(stock_df, benchmark_df).dropna()
        assert result["outperform_5d"].isin([0, 1]).all()

    def test_unrealized_tail_is_missing_not_a_loss(
        self, stock_df: pd.DataFrame, benchmark_df: pd.DataFrame
    ) -> None:
        result = OutperformanceLabel().compute(stock_df, benchmark_df, horizon=5)
        assert result["outperform_5d"].tail(5).isna().all()
        assert result["excess_return_5d"].tail(5).isna().all()
