from __future__ import annotations

import pandas as pd

from src.data.collector import OHLCVCollector
from src.data.universe import get_ticker_universe, filter_quality
from src.data.validator import DataValidator


class TestUniverse:
    def test_returns_list_of_strings(self) -> None:
        universe = get_ticker_universe()
        assert isinstance(universe, list)
        assert len(universe) > 0
        assert all(isinstance(t, str) for t in universe)

    def test_contains_vn30_stocks(self) -> None:
        universe = get_ticker_universe()
        assert "VNM" in universe
        assert "VCB" in universe


class TestCollector:
    def test_mock_data_shape(self) -> None:
        collector = OHLCVCollector()
        df = collector.fetch("VNM", days=60)
        assert isinstance(df, pd.DataFrame)
        assert not df.empty
        assert "close" in df.columns
        assert "volume" in df.columns
        assert df["ticker"].iloc[0] == "VNM"

    def test_mock_data_dates_are_business_days(self) -> None:
        collector = OHLCVCollector()
        df = collector.fetch("ACB", days=30)
        dates = df["date"]
        assert all(d.weekday() < 5 for d in dates)


class TestValidator:
    def test_valid_data_passes(self) -> None:
        df = pd.DataFrame({
            "ticker": ["VNM"],
            "date": pd.to_datetime(["2024-01-02"]),
            "open": [100.0],
            "high": [105.0],
            "low": [99.0],
            "close": [102.0],
            "volume": [1_000_000],
        })
        errors = DataValidator().validate(df)
        assert errors == []

    def test_missing_columns(self) -> None:
        df = pd.DataFrame({"ticker": ["VNM"], "close": [100.0]})
        errors = DataValidator().validate(df)
        assert len(errors) > 0

    def test_empty_dataframe(self) -> None:
        errors = DataValidator().validate(pd.DataFrame())
        assert "DataFrame is empty" in errors

    def test_quality_filter_good_data(self) -> None:
        df = pd.DataFrame({
            "ticker": ["VNM"] * 250,
            "date": pd.date_range("2024-01-01", periods=250, freq="B"),
            "open": [100.0] * 250,
            "high": [105.0] * 250,
            "low": [99.0] * 250,
            "close": [102.0] * 250,
            "volume": [1_000_000] * 250,
        })
        result = filter_quality(df, "VNM")
        assert result is not None

    def test_quality_filter_short_history(self) -> None:
        df = pd.DataFrame({
            "ticker": ["VNM"] * 50,
            "date": pd.date_range("2024-01-01", periods=50, freq="B"),
            "open": [100.0] * 50,
            "high": [105.0] * 50,
            "low": [99.0] * 50,
            "close": [102.0] * 50,
            "volume": [1_000_000] * 50,
        })
        result = filter_quality(df, "VNM")
        assert result is None

    def test_quality_filter_low_volume(self) -> None:
        df = pd.DataFrame({
            "ticker": ["VNM"] * 250,
            "date": pd.date_range("2024-01-01", periods=250, freq="B"),
            "open": [100.0] * 250,
            "high": [105.0] * 250,
            "low": [99.0] * 250,
            "close": [102.0] * 250,
            "volume": [100] * 250,
        })
        result = filter_quality(df, "VNM")
        assert result is None

    def test_negative_price(self) -> None:
        df = pd.DataFrame({
            "ticker": ["VNM"],
            "date": pd.to_datetime(["2024-01-02"]),
            "open": [-100.0],
            "high": [105.0],
            "low": [99.0],
            "close": [102.0],
            "volume": [1_000_000],
        })
        errors = DataValidator().validate(df)
        assert any("open" in e for e in errors)
