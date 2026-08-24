from __future__ import annotations

from types import SimpleNamespace

import pandas as pd

from src.filters.entry import apply_entry_filters


def _config(**overrides):
    values = {
        "enable_entry_filters": True,
        "min_market_breadth_20d": 0.50,
        "min_entry_atr_percentile": 0.20,
        "min_entry_trend_percentile": 0.0,
        "min_entry_picks": 3,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _frames(breadth_positive: int = 6):
    tickers = [f"T{i:02d}" for i in range(10)]
    features = pd.DataFrame({
        "date": pd.Timestamp("2026-08-05"),
        "ticker": tickers,
        "return_20d": [1.0] * breadth_positive + [-1.0] * (10 - breadth_positive),
        "atr_pct": [0.01 * (i + 1) for i in range(10)],
        "return_60d": [0.01 * i for i in range(10)],
        "rs_60d": [0.02 * i for i in range(10)],
    })
    ranking = pd.DataFrame({
        "date": pd.Timestamp("2026-08-05"),
        "ticker": tickers[:5],
        "score": [0.9, 0.8, 0.7, 0.6, 0.5],
        "rank": range(1, 6),
    })
    return ranking, features


def test_entry_filters_pass_and_re_rank_eligible_candidates() -> None:
    ranking, features = _frames()
    result, report = apply_entry_filters(ranking, features, _config())

    assert report["status"] == "passed"
    assert len(result) == 4
    assert result["rank"].tolist() == [1, 2, 3, 4]


def test_entry_filters_block_weak_market_breadth() -> None:
    ranking, features = _frames(breadth_positive=4)
    result, report = apply_entry_filters(ranking, features, _config())

    assert result.empty
    assert report["status"] == "blocked"
    assert report["reason"] == "market breadth below threshold"


def test_entry_filters_are_noop_when_feature_flag_is_off() -> None:
    ranking, features = _frames(breadth_positive=0)
    result, report = apply_entry_filters(
        ranking,
        features,
        _config(enable_entry_filters=False),
    )

    assert result["ticker"].tolist() == ranking["ticker"].tolist()
    assert report["status"] == "passed"


def test_entry_filters_block_when_score_margin_is_too_small() -> None:
    ranking = pd.DataFrame({
        "date": pd.Timestamp("2026-01-01"),
        "ticker": ["AAA", "BBB", "CCC", "DDD"],
        "score": [0.40, 0.39, 0.38, 0.379],
    })
    features = pd.DataFrame({
        "date": pd.Timestamp("2026-01-01"),
        "ticker": ["AAA", "BBB", "CCC", "DDD"],
        "return_20d": [0.1, 0.2, 0.3, 0.4],
        "atr_pct": [0.01, 0.02, 0.03, 0.04],
    })

    result, report = apply_entry_filters(
        ranking,
        features,
        _config(
            min_entry_atr_percentile=0.0,
            min_entry_score=0.10,
            min_entry_score_margin=0.02,
        ),
    )

    assert result.empty
    assert report["reason"] == "score margin below threshold"
