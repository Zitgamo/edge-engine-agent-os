"""Horizon-score blending helpers used by production and backtests."""

from __future__ import annotations

import re

import pandas as pd


_SCORE_PATTERN = re.compile(r"^score_(\d+)d$")


def _horizon_from_column(column: str) -> int | None:
    match = _SCORE_PATTERN.match(column)
    return int(match.group(1)) if match else None


def blend_horizon_scores(
    frame: pd.DataFrame,
    horizons: list[int],
    *,
    weights: dict[int, float] | None = None,
    mode: str = "rank",
) -> pd.Series:
    """Blend per-horizon scores while handling missing models safely.

    Raw classifier probabilities are not necessarily calibrated to the same
    scale across horizons.  ``rank`` mode therefore blends each horizon's
    cross-sectional percentile within a market date.  ``raw`` is retained for
    compatibility and for controlled experiments.
    """
    columns = [f"score_{h}d" for h in horizons if f"score_{h}d" in frame.columns]
    if not columns:
        return pd.Series(float("nan"), index=frame.index, name="ensemble_score")

    scores = frame[columns].apply(pd.to_numeric, errors="coerce")
    normalized = scores.copy()
    if mode == "rank":
        if "date" in frame.columns:
            for column in columns:
                normalized[column] = scores.groupby(frame["date"], sort=False)[column].rank(
                    method="average", pct=True
                )
        else:
            for column in columns:
                normalized[column] = scores[column].rank(method="average", pct=True)
    elif mode != "raw":
        raise ValueError("mode must be 'rank' or 'raw'")

    configured = weights or {}
    column_weights = {
        column: max(0.0, float(configured.get(_horizon_from_column(column) or 0, 0.0)))
        for column in columns
    }
    if not any(column_weights.values()):
        column_weights = {column: 1.0 for column in columns}

    weight_frame = pd.DataFrame(
        {
            column: normalized[column].notna().astype(float) * column_weights[column]
            for column in columns
        },
        index=frame.index,
    )
    denominator = weight_frame.sum(axis=1)
    numerator = sum(
        normalized[column].fillna(0.0) * column_weights[column]
        for column in columns
    )
    result = numerator.div(denominator.where(denominator > 0))
    result.name = "ensemble_score"
    return result
