"""Model target schemas and compatibility resolution."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class TargetSpec:
    """Column names for one horizon's supervised target."""

    horizon: int
    target_col: str
    return_col: str
    label_end_col: str
    execution: bool = False


def target_spec(horizon: int, *, use_execution_target: bool = True) -> TargetSpec:
    """Return the preferred target schema for ``horizon``."""
    if horizon <= 0:
        raise ValueError("horizon must be positive")
    if use_execution_target and horizon == 20:
        return TargetSpec(
            horizon=horizon,
            target_col=f"execution_outperform_{horizon}d",
            return_col=f"execution_excess_return_{horizon}d",
            label_end_col=f"execution_label_end_date_{horizon}d",
            execution=True,
        )
    return TargetSpec(
        horizon=horizon,
        target_col=f"outperform_{horizon}d",
        return_col=f"excess_return_{horizon}d",
        label_end_col=f"label_end_date_{horizon}d",
        execution=False,
    )


def resolve_target_spec(
    df: pd.DataFrame,
    horizon: int,
    *,
    prefer_execution: bool = True,
) -> TargetSpec:
    """Use the execution target when persisted columns are available.

    This compatibility fallback lets historical feature files continue to be
    inspected while ensuring a freshly generated production file uses the
    execution-aligned T+20 target.
    """
    preferred = target_spec(horizon, use_execution_target=prefer_execution)
    if all(column in df.columns for column in (
        preferred.target_col,
        preferred.return_col,
        preferred.label_end_col,
    )):
        return preferred
    return target_spec(horizon, use_execution_target=False)
