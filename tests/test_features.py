from __future__ import annotations

import pandas as pd
import pytest

from src.features.returns import ReturnFeatures
from src.features.volatility import ATR
from src.features.volume import VolumeSurge


@pytest.fixture
def sample_df() -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=100, freq="B")
    return pd.DataFrame({
        "ticker": ["VNM"] * 100,
        "date": dates,
        "open": range(100, 200),
        "high": range(105, 205),
        "low": range(95, 195),
        "close": range(100, 200),
        "volume": [1_000_000] * 100,
    })


class TestReturnFeatures:
    def test_computes_return_columns(self, sample_df: pd.DataFrame) -> None:
        result = ReturnFeatures().compute(sample_df)
        for w in [5, 20, 60]:
            assert f"return_{w}d" in result.columns

    def test_returns_are_float(self, sample_df: pd.DataFrame) -> None:
        result = ReturnFeatures().compute(sample_df)
        assert result["return_5d"].dtype == "float64"


class TestATR:
    def test_computes_atr(self, sample_df: pd.DataFrame) -> None:
        result = ATR().compute(sample_df)
        assert "atr" in result.columns
        assert "atr_pct" in result.columns

    def test_atr_non_negative(self, sample_df: pd.DataFrame) -> None:
        result = ATR().compute(sample_df).dropna()
        assert (result["atr"] >= 0).all()


class TestVolumeSurge:
    def test_computes_surge(self, sample_df: pd.DataFrame) -> None:
        result = VolumeSurge().compute(sample_df)
        assert "volume_surge" in result.columns
        assert "volume_surge_flag" in result.columns

    def test_flag_is_integer(self, sample_df: pd.DataFrame) -> None:
        result = VolumeSurge().compute(sample_df).dropna()
        assert result["volume_surge_flag"].dtype == "int" or result["volume_surge_flag"].dtype == "int64"
