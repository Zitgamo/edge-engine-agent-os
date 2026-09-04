"""Shared executable-barrier mechanics for backtests and live tracking."""

from __future__ import annotations

from typing import Any

import numpy as np


def _row_value(row: Any, column: str, fallback: float | None = None) -> float:
    """Read a numeric OHLC value from either a Series-like row or namedtuple."""
    try:
        value = row[column]
    except (KeyError, IndexError, TypeError):
        value = getattr(row, column, fallback)
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return result


def barrier_exit(
    entry_price: float,
    row: Any,
    stop_loss: float,
    take_profit: float,
) -> tuple[float, str] | None:
    """Return the executable barrier fill for one OHLC bar.

    A stop/target that is crossed by the session open is filled at that open,
    modelling a gap. Otherwise it is filled at the configured barrier. When
    both barriers are touched on the same bar, the stop is checked first as a
    conservative daily-bar assumption.
    """
    entry = float(entry_price)
    if not np.isfinite(entry) or entry <= 0:
        return None

    open_price = _row_value(row, "open")
    close_price = _row_value(row, "close")
    low = _row_value(row, "low", close_price)
    high = _row_value(row, "high", close_price)
    if not np.isfinite(low) or not np.isfinite(high):
        return None

    sl_price = entry * (1 + float(stop_loss))
    tp_price = entry * (1 + float(take_profit))

    if stop_loss < 0 and low <= sl_price:
        fill = open_price if np.isfinite(open_price) and open_price <= sl_price else sl_price
        return float(fill), "HIT_SL"

    if take_profit > 0 and high >= tp_price:
        fill = open_price if np.isfinite(open_price) and open_price >= tp_price else tp_price
        return float(fill), "HIT_TP"

    return None
