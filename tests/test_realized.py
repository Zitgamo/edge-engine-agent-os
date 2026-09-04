from __future__ import annotations

import pandas as pd
import pytest

from src.model.realized import summarize_realized_model_history


FAMILY = "test_model"


def _history(
    *,
    baskets: int,
    value: float,
    model_family: str = FAMILY,
) -> pd.DataFrame:
    rows = []
    for day_index in range(baskets):
        signal_date = f"2026-08-{day_index + 1:02d}"
        for rank in range(1, 4):
            rows.append(
                {
                    "signal_date": signal_date,
                    "ticker": f"T{day_index:02d}{rank}",
                    "rank": rank,
                    "model_version": f"{model_family}_h20_2026-09-04",
                    "actual_excess_return_20d": value,
                }
            )
    return pd.DataFrame(rows)


def test_live_validation_stays_pending_until_complete_baskets_exist() -> None:
    frame = pd.concat(
        [
            _history(baskets=2, value=0.02),
            _history(baskets=10, value=0.50, model_family="legacy_model"),
        ],
        ignore_index=True,
    )

    health = summarize_realized_model_history(
        frame,
        model_family=FAMILY,
        min_trades=30,
        min_baskets=10,
    )

    assert health["trade_count"] == 6
    assert health["basket_count"] == 2
    assert health["health_status"] == "pending"
    assert health["ready"] is False
    assert health["model_versions"] == ["test_model_h20_2026-09-04"]


def test_live_validation_marks_a_ready_family_healthy() -> None:
    health = summarize_realized_model_history(
        _history(baskets=10, value=0.01),
        model_family=FAMILY,
        min_trades=30,
        min_baskets=10,
        min_avg_excess_return=0.0,
        min_win_rate=0.45,
    )

    assert health["trade_count"] == 30
    assert health["basket_count"] == 10
    assert health["avg_excess_return"] == pytest.approx(0.01)
    assert health["win_rate"] == pytest.approx(1.0)
    assert health["health_status"] == "healthy"
    assert health["ready"] is True


def test_live_validation_marks_a_ready_underperforming_family() -> None:
    health = summarize_realized_model_history(
        _history(baskets=10, value=-0.01),
        model_family=FAMILY,
        min_trades=30,
        min_baskets=10,
    )

    assert health["health_status"] == "underperforming"
    assert health["ready"] is True
    assert health["win_rate"] == pytest.approx(0.0)


def test_live_validation_handles_missing_columns_and_non_finite_thresholds() -> None:
    health = summarize_realized_model_history(
        pd.DataFrame({"signal_date": ["2026-09-04"]}),
        model_family=FAMILY,
    )

    assert health["health_status"] == "pending"
    assert "Missing realized history columns" in health["reason"]

    with pytest.raises(ValueError, match="finite"):
        summarize_realized_model_history(
            pd.DataFrame(),
            model_family=FAMILY,
            min_win_rate=float("inf"),
        )
