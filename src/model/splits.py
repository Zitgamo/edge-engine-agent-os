"""Time-aware, label-purged train/test splits for panel data."""

from __future__ import annotations

import pandas as pd


def _normalise_dates(values: pd.Series) -> pd.Series:
    """Return timezone-naive, date-normalized timestamps."""
    dates = pd.to_datetime(values, errors="coerce")
    if dates.dt.tz is not None:
        dates = dates.dt.tz_localize(None)
    return dates.dt.normalize()


def purged_train_test_split(
    df: pd.DataFrame,
    *,
    test_start: pd.Timestamp | str,
    label_end_col: str,
    date_col: str = "date",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split chronologically without training on labels unavailable at test time.

    A row dated before ``test_start`` is eligible for training only when its
    forward label has fully matured before ``test_start``.  The endpoint is
    supplied per row because a stock may have missing sessions and therefore
    cannot safely use ``signal_date + N business days``.
    """
    missing = [column for column in (date_col, label_end_col) if column not in df.columns]
    if missing:
        raise ValueError(f"Missing purged-split columns: {missing}")

    boundary = pd.Timestamp(test_start)
    if boundary.tzinfo is not None:
        boundary = boundary.tz_localize(None)
    boundary = boundary.normalize()

    dates = _normalise_dates(df[date_col])
    label_end = _normalise_dates(df[label_end_col])
    train_mask = dates.lt(boundary) & label_end.notna() & label_end.lt(boundary)
    test_mask = dates.ge(boundary)

    return df.loc[train_mask].copy(), df.loc[test_mask].copy()


def recent_date_window(
    df: pd.DataFrame,
    max_dates: int | None,
    *,
    date_col: str = "date",
) -> pd.DataFrame:
    """Keep only the most recent ``max_dates`` observations by date.

    The window is measured in observed trading sessions, not calendar days,
    so it remains stable across holidays and missing market sessions.  A
    non-positive or omitted limit means the caller explicitly requested the
    full input frame.
    """
    if max_dates is None or max_dates <= 0:
        return df.copy()
    if date_col not in df.columns:
        raise ValueError(f"Missing rolling-window date column: {date_col}")

    dates = _normalise_dates(df[date_col])
    unique_dates = sorted(dates.dropna().unique())
    if len(unique_dates) <= max_dates:
        return df.copy()
    keep_dates = set(unique_dates[-max_dates:])
    return df.loc[dates.isin(keep_dates)].copy()


def purged_recent_train_window(
    df: pd.DataFrame,
    *,
    test_start: pd.Timestamp | str,
    label_end_col: str,
    max_dates: int | None,
    date_col: str = "date",
) -> pd.DataFrame:
    """Return a label-purged train frame bounded to recent sessions."""
    train, _ = purged_train_test_split(
        df,
        test_start=test_start,
        label_end_col=label_end_col,
        date_col=date_col,
    )
    return recent_date_window(train, max_dates, date_col=date_col)


def require_label_end_columns(df: pd.DataFrame, horizons: list[int]) -> None:
    """Fail closed when a persisted feature file predates purge metadata."""
    require_columns(
        df,
        [f"label_end_date_{h}d" for h in horizons],
        description="label maturity metadata",
    )


def require_columns(
    df: pd.DataFrame,
    columns: list[str],
    *,
    description: str = "required columns",
) -> None:
    """Fail closed with a consistent error for persisted schemas."""
    missing = [column for column in columns if column not in df]
    if missing:
        raise ValueError(
            f"Feature data is missing {description}: "
            f"{missing}. Re-run the pipeline to regenerate features.parquet."
        )
