from __future__ import annotations

import pandas as pd
import pytest

from src.model.blend import blend_horizon_scores


def test_rank_blend_is_cross_sectional_and_weighted() -> None:
    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-01-01"] * 3),
            "score_5d": [0.1, 0.2, 0.3],
            "score_20d": [0.9, 0.2, 0.1],
        }
    )

    result = blend_horizon_scores(
        frame,
        [5, 20],
        weights={5: 0.75, 20: 0.25},
        mode="rank",
    )

    assert result.iloc[2] > result.iloc[1] > result.iloc[0]
    assert result.between(0.0, 1.0).all()


def test_blend_falls_back_to_available_horizons() -> None:
    frame = pd.DataFrame({"score_5d": [0.2, 0.8]})

    result = blend_horizon_scores(
        frame,
        [5, 20],
        weights={5: 0.0, 20: 1.0},
        mode="raw",
    )

    assert result.tolist() == pytest.approx([0.2, 0.8])


def test_blend_rejects_unknown_mode() -> None:
    with pytest.raises(ValueError, match="mode"):
        blend_horizon_scores(pd.DataFrame({"score_5d": [0.2]}), [5], mode="bad")


def test_max_blend_selects_strongest_available_horizon() -> None:
    frame = pd.DataFrame(
        {
            "score_5d": [0.2, None, 0.4],
            "score_20d": [0.6, 0.3, None],
        }
    )

    result = blend_horizon_scores(
        frame,
        [5, 20],
        weights={5: 0.0, 20: 1.0},
        mode="max",
    )

    assert result.tolist() == pytest.approx([0.6, 0.3, 0.4])
