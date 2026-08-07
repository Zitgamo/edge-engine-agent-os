from __future__ import annotations

import pandas as pd
import pytest

from src.features.returns import ReturnFeatures
from src.features.rs import RelativeStrength
from src.features.volatility import ATR
from src.features.volume import VolumeSurge
from src.features.macro import MacroFeatures, add_macro_features


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


def test_relative_strength_aligns_benchmark_by_date() -> None:
    benchmark_dates = pd.bdate_range("2026-01-01", periods=15)
    stock_dates = benchmark_dates[5:]
    benchmark = pd.DataFrame({
        "date": benchmark_dates,
        "close": [100, 200, 202, 204, 206, 208, 210, 212, 214, 216, 218, 220, 222, 224, 226],
    })
    stock = pd.DataFrame({
        "date": stock_dates,
        "close": [100, 101, 102, 103, 104, 105, 106, 107, 108, 109],
    })

    result = RelativeStrength().compute(stock, benchmark).set_index("date")
    stock_returns = stock.set_index("date")["close"].pct_change().rolling(5).sum()
    benchmark_returns = benchmark.set_index("date")["close"].pct_change().reindex(stock_dates).rolling(5).sum()
    expected = stock_returns - benchmark_returns

    pd.testing.assert_series_equal(
        result["rs_5d"], expected,
        check_names=False,
    )


def test_relative_strength_ignores_non_overlapping_sessions_in_rolling_window() -> None:
    dates = pd.bdate_range("2026-01-01", periods=70)
    benchmark_dates = dates.delete(20)
    benchmark = pd.DataFrame({
        "date": benchmark_dates,
        "close": range(100, 169),
    })
    stock = pd.DataFrame({
        "date": dates,
        "close": range(200, 270),
    })

    result = RelativeStrength().compute(stock, benchmark).set_index("date")

    # The missing benchmark session must not poison the 60-session window
    # after the stock/benchmark calendars re-align.
    assert pd.notna(result.loc[dates[61], "rs_60d"])


def test_macro_features_keep_schema_when_fx_is_unavailable(monkeypatch) -> None:
    monkeypatch.setattr(
        MacroFeatures,
        "fetch_all",
        lambda self: pd.DataFrame({
            "date": pd.to_datetime(["2026-01-02"]),
            "sbv_rate": [4.5],
            "cpi_mom": [0.3],
        }),
    )
    df = pd.DataFrame({
        "ticker": ["AAA"],
        "date": pd.to_datetime(["2026-01-02"]),
        "close": [100.0],
    })

    result = add_macro_features(df)

    assert "vndusd" in result.columns
    assert result.loc[0, "vndusd"] == 0.0
