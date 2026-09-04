from __future__ import annotations

import pandas as pd

from src.research.kbs_dataset import (
    build_execution_outcomes,
    build_feature_panel,
    refresh_research_data,
)


def _write_prices(path, dates, base) -> None:
    close = [base + index * 0.2 for index in range(len(dates))]
    pd.DataFrame({
        "date": dates,
        "open": close,
        "high": [value + 1.0 for value in close],
        "low": [value - 1.0 for value in close],
        "close": close,
        "volume": [1000] * len(dates),
    }).to_parquet(path, index=False)


def test_kbs_dataset_builder_creates_features_and_gap_aware_outcomes(tmp_path) -> None:
    dates = pd.date_range("2020-01-01", periods=90, freq="B")
    _write_prices(tmp_path / "AAA_raw.parquet", dates, 100.0)
    _write_prices(tmp_path / "VNINDEX_raw.parquet", dates, 1000.0)

    features = build_feature_panel(
        tmp_path,
        output_path=tmp_path / "processed" / "features_exact.parquet",
        as_of=dates.max() + pd.Timedelta(hours=16),
    )
    outcomes = build_execution_outcomes(
        tmp_path,
        output_dir=tmp_path / "research_results",
        as_of=dates.max() + pd.Timedelta(hours=16),
    )

    assert len(features) == len(dates)
    assert {"return_20d", "rs_20d", "atr", "atr_pct"}.issubset(features.columns)
    assert len(outcomes["atr15_tp10"]) > 0
    assert len(outcomes["atr2_tp10"]) > 0
    assert set(outcomes["atr2_tp10"]["reason"]).issubset(
        {"stop", "stop_gap", "target", "target_gap", "time"}
    )
    assert (tmp_path / "processed" / "features_exact.parquet").exists()
    assert (tmp_path / "research_results" / "outcomes_atr2_tp10.parquet").exists()


def test_research_refresh_merges_existing_cache_and_records_fetch_status(
    tmp_path,
    monkeypatch,
) -> None:
    dates = pd.date_range("2026-01-01", periods=3, freq="B")
    (tmp_path / "raw").mkdir()
    _write_prices(tmp_path / "raw" / "AAA_raw.parquet", dates[:2], 100.0)
    fetch_days: dict[str, int] = {}

    class FakeCollector:
        def __init__(self, config) -> None:
            assert config.data_source == "kbs"

        def fetch(self, ticker: str, days: int) -> pd.DataFrame:
            fetch_days[ticker] = days
            return pd.DataFrame({
                "ticker": ticker,
                "date": [dates[-1]],
                "open": [101.0],
                "high": [102.0],
                "low": [100.0],
                "close": [101.5],
                "volume": [1000],
            })

    monkeypatch.setattr("src.data.collector.OHLCVCollector", FakeCollector)
    config = type("ConfigStub", (), {
        "kbs_base_url": "https://example.test",
        "kbs_timeout_seconds": 5,
    })()

    report = refresh_research_data(
        tmp_path,
        config=config,
        tickers=["AAA", "VNINDEX"],
        lookback_days=1825,
        refresh_days=120,
        as_of=dates[-1] + pd.Timedelta(hours=16),
    )

    assert fetch_days == {"AAA": 120, "VNINDEX": 1825}
    assert len(pd.read_parquet(tmp_path / "raw" / "AAA_raw.parquet")) == 3
    assert set(report["status"]) == {"ok"}
    assert (tmp_path / "download_summary.json").exists()


def test_research_refresh_marks_empty_response_stale_and_keeps_existing_cache(
    tmp_path,
    monkeypatch,
) -> None:
    dates = pd.date_range("2026-01-01", periods=2, freq="B")
    (tmp_path / "raw").mkdir()
    _write_prices(tmp_path / "raw" / "AAA_raw.parquet", dates, 100.0)

    class EmptyCollector:
        def __init__(self, config) -> None:
            pass

        def fetch(self, ticker: str, days: int) -> pd.DataFrame:
            return pd.DataFrame()

    monkeypatch.setattr("src.data.collector.OHLCVCollector", EmptyCollector)
    config = type("ConfigStub", (), {
        "kbs_base_url": "https://example.test",
        "kbs_timeout_seconds": 5,
    })()

    report = refresh_research_data(
        tmp_path,
        config=config,
        tickers=["AAA"],
        lookback_days=1825,
        refresh_days=120,
    )

    assert report.loc[0, "status"] == "stale"
    assert len(pd.read_parquet(tmp_path / "raw" / "AAA_raw.parquet")) == 2


def test_research_refresh_uses_latest_closed_date_for_freshness(
    tmp_path,
    monkeypatch,
) -> None:
    (tmp_path / "raw").mkdir()

    class FutureOnlyCollector:
        def __init__(self, config) -> None:
            pass

        def fetch(self, ticker: str, days: int) -> pd.DataFrame:
            return pd.DataFrame({
                "ticker": ticker,
                "date": ["2026-01-10"],
                "open": [101.0],
                "high": [102.0],
                "low": [100.0],
                "close": [101.5],
                "volume": [1000],
            })

    monkeypatch.setattr("src.data.collector.OHLCVCollector", FutureOnlyCollector)
    config = type("ConfigStub", (), {
        "kbs_base_url": "https://example.test",
        "kbs_timeout_seconds": 5,
    })()

    report = refresh_research_data(
        tmp_path,
        config=config,
        tickers=["AAA"],
        lookback_days=1825,
        refresh_days=120,
        as_of=pd.Timestamp("2026-01-10 10:00"),
    )

    assert report.loc[0, "status"] == "stale"
    assert report.loc[0, "end"] == "2026-01-10"
