"""Descriptive bottom-to-now diagnostics for every cached stock ticker.

This report is deliberately different from the point-in-time exit optimizer.
It answers a current-state question: after the most recent meaningful swing
low, did a fixed TP/SL leave too much of the move on the table, and does the
price path currently look more suitable for holding or short-term trading?

The bottom is an ex-post diagnostic anchor, not a live entry signal.  A
tradable reference entry is therefore placed at the first open after the
right-side pivot confirmation window.  All current prices are limited to the
latest closed session available in the cache.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from src.features.returns import ReturnFeatures
from src.features.rs import RelativeStrength
from src.features.volatility import ATR
from src.research.kbs_dataset import (
    _load_prices,
    latest_closed_date,
)
from src.research.ticker_exit_optimizer import _resolve_universe
from src.time_utils import now_vn

log = logging.getLogger(__name__)

ANALYSIS_SCHEMA_VERSION = "bottom_to_now_v1"
DEFAULT_BOTTOM_LOOKBACK_BARS = 252
DEFAULT_PIVOT_LEFT_BARS = 5
DEFAULT_PIVOT_RIGHT_BARS = 5
DEFAULT_MINIMUM_REBOUND = 0.05
DEFAULT_FRESHNESS_DAYS = 3
DEFAULT_TARGET_OVERSHOOT_MARGIN = 0.05
DEFAULT_TP_GRID = (0.08, 0.10, 0.12)

REPORT_COLUMNS = [
    "ticker",
    "latest_date",
    "data_age_days",
    "data_status",
    "price_rows",
    "analysis_status",
    "bottom_date",
    "bottom_method",
    "bottom_low",
    "bottom_close",
    "bottom_confirmed_date",
    "entry_date",
    "entry_price",
    "bars_since_entry",
    "bottom_to_now_return_pct",
    "entry_to_now_return_pct",
    "peak_date",
    "peak_high",
    "max_favorable_excursion_pct",
    "max_adverse_excursion_pct",
    "peak_to_current_drawdown_pct",
    "atr_reference",
    "atr_reference_pct",
    "atr2_stop_loss_pct",
    "close_vs_ema20_pct",
    "close_vs_ema50_pct",
    "return_20d_pct",
    "return_60d_pct",
    "rs_20d_pct",
    "rs_60d_pct",
    "trend_score",
    "management_mode",
    "management_reason",
    "fixed_stop_loss_pct",
    "fixed_sl_first_event",
    "fixed_sl_event_date",
    "fixed_sl_before_tp10",
    "fixed_sl_then_tp10",
    "fixed_sl_recovery_to_entry",
    "atr2_sl_first_event",
    "atr2_sl_event_date",
    "atr2_sl_before_tp10",
    "atr2_sl_then_tp10",
    "tp8_hit",
    "tp8_event_date",
    "tp8_bars_from_entry",
    "tp8_peak_after_hit_extra_pct",
    "tp8_current_vs_target_pct",
    "tp8_state",
    "tp10_hit",
    "tp10_event_date",
    "tp10_bars_from_entry",
    "tp10_peak_after_hit_extra_pct",
    "tp10_current_vs_target_pct",
    "tp10_state",
    "tp10_non_current_flag",
    "tp10_non_peak_flag",
    "tp12_hit",
    "tp12_event_date",
    "tp12_bars_from_entry",
    "tp12_peak_after_hit_extra_pct",
    "tp12_current_vs_target_pct",
    "tp12_state",
    "skip_reason",
]


def _float(value: object, default: float = float("nan")) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if np.isfinite(result) else default


def _date_text(value: object) -> str:
    timestamp = pd.Timestamp(value)
    return timestamp.date().isoformat()


def _safe_ratio(numerator: object, denominator: object) -> float:
    top = _float(numerator)
    bottom = _float(denominator)
    if not np.isfinite(top) or not np.isfinite(bottom) or bottom == 0:
        return float("nan")
    return top / bottom


def _find_recent_bottom(
    prices: pd.DataFrame,
    *,
    lookback_bars: int,
    pivot_left_bars: int,
    pivot_right_bars: int,
    minimum_rebound: float,
) -> dict[str, int | str] | None:
    """Find the latest confirmed meaningful low, with a rolling-low fallback."""
    if lookback_bars <= 0:
        raise ValueError("lookback_bars must be positive")
    if pivot_left_bars <= 0 or pivot_right_bars <= 0:
        raise ValueError("pivot confirmation bars must be positive")
    if minimum_rebound < 0:
        raise ValueError("minimum_rebound cannot be negative")

    lows = pd.to_numeric(prices["low"], errors="coerce").to_numpy(dtype=float)
    close_source = prices["close"] if "close" in prices.columns else prices["high"]
    closes = pd.to_numeric(close_source, errors="coerce").to_numpy(dtype=float)
    n_rows = len(prices)
    if n_rows < pivot_left_bars + pivot_right_bars + 2:
        return None

    start = max(pivot_left_bars, n_rows - lookback_bars)
    end = n_rows - pivot_right_bars
    candidates: list[int] = []
    for index in range(start, end):
        low = lows[index]
        if not np.isfinite(low) or low <= 0:
            continue
        local_window = lows[index - pivot_left_bars : index + pivot_right_bars + 1]
        finite_window = local_window[np.isfinite(local_window)]
        if finite_window.size == 0 or low > finite_window.min() * (1.0 + 1e-9):
            continue
        left_window = lows[index - pivot_left_bars : index]
        right_window = lows[index + 1 : index + pivot_right_bars + 1]
        left_min = np.nanmin(left_window) if np.isfinite(left_window).any() else np.nan
        right_min = np.nanmin(right_window) if np.isfinite(right_window).any() else np.nan
        # Do not treat a flat, still-forming base as a series of new bottoms.
        # At least one side must be strictly higher than the pivot low.
        strictly_lower_than_neighbor = (
            (np.isfinite(left_min) and low < left_min * (1.0 - 1e-9))
            or (np.isfinite(right_min) and low < right_min * (1.0 - 1e-9))
        )
        if not strictly_lower_than_neighbor:
            continue
        future_closes = closes[index + 1 :]
        finite_future_closes = future_closes[np.isfinite(future_closes)]
        if (
            finite_future_closes.size > 0
            and finite_future_closes.max() >= low * (1.0 + minimum_rebound)
        ):
            candidates.append(index)

    if candidates:
        return {
            "index": candidates[-1],
            "method": "confirmed_swing_low",
        }

    window_lows = lows[start:]
    finite_indices = np.flatnonzero(np.isfinite(window_lows) & (window_lows > 0))
    if finite_indices.size == 0:
        return None
    relative_index = int(finite_indices[np.argmin(window_lows[finite_indices])])
    return {
        "index": int(start + relative_index),
        "method": "rolling_low_fallback",
    }


def _first_hit(path: pd.DataFrame, *, column: str, level: float, above: bool) -> int | None:
    values = pd.to_numeric(path[column], errors="coerce").to_numpy(dtype=float)
    if above:
        hits = np.flatnonzero(np.isfinite(values) & (values >= level))
    else:
        hits = np.flatnonzero(np.isfinite(values) & (values <= level))
    return int(hits[0]) if hits.size else None


def _first_event(
    path: pd.DataFrame,
    *,
    stop_price: float,
    target_price: float,
) -> tuple[str, int | None, int | None]:
    """Return the first event; a same-day stop/target collision is conservative."""
    stop_index = _first_hit(path, column="low", level=stop_price, above=False)
    target_index = _first_hit(path, column="high", level=target_price, above=True)
    if stop_index is None and target_index is None:
        return "none", stop_index, target_index
    if target_index is None or (
        stop_index is not None and stop_index <= target_index
    ):
        return "stop", stop_index, target_index
    return "target", stop_index, target_index


def _target_metrics(
    path: pd.DataFrame,
    *,
    entry_price: float,
    current_price: float,
    target_pct: float,
    target_index: int | None,
    overshoot_margin: float,
) -> dict[str, object]:
    label = f"tp{int(round(target_pct * 100))}"
    target_price = entry_price * (1.0 + target_pct)
    if target_index is None:
        return {
            f"{label}_hit": False,
            f"{label}_event_date": "",
            f"{label}_bars_from_entry": 0,
            f"{label}_peak_after_hit_extra_pct": float("nan"),
            f"{label}_current_vs_target_pct": float("nan"),
            f"{label}_state": "not_reached",
        }

    after_target = path.iloc[target_index:]
    post_target_peak = pd.to_numeric(after_target["high"], errors="coerce").max()
    peak_extra = _safe_ratio(post_target_peak, target_price) - 1.0
    current_vs_target = _safe_ratio(current_price, target_price) - 1.0
    if peak_extra >= overshoot_margin and current_vs_target >= overshoot_margin:
        state = "extended"
    elif peak_extra >= overshoot_margin:
        state = "extended_then_retraced"
    elif current_vs_target >= 0:
        state = "reached_near_target"
    else:
        state = "reached_then_retraced"
    return {
        f"{label}_hit": True,
        f"{label}_event_date": _date_text(path.iloc[target_index]["date"]),
        f"{label}_bars_from_entry": int(target_index + 1),
        f"{label}_peak_after_hit_extra_pct": float(peak_extra),
        f"{label}_current_vs_target_pct": float(current_vs_target),
        f"{label}_state": state,
    }


def _management_mode(
    latest: pd.Series,
    *,
    drawdown_from_peak: float,
    data_status: str,
) -> tuple[str, int | None, str]:
    """Classify risk management mode from trend health, not entry ranking."""
    if data_status != "fresh":
        return "STALE_DATA", None, "latest OHLCV is outside the freshness window"

    fields = [
        "close",
        "ema20",
        "ema50",
        "return_20d",
        "return_60d",
        "rs_20d",
        "rs_60d",
        "atr_pct",
    ]
    values = [_float(latest.get(field)) for field in fields]
    if not all(np.isfinite(value) for value in values):
        return "INSUFFICIENT_DATA", None, "not enough history for the full trend test"

    close, ema20, ema50, return20, return60, rs20, rs60, atr_pct = values
    conditions = [
        close > ema20,
        ema20 > ema50,
        return20 > 0,
        return60 > 0,
        rs20 > 0,
        rs60 > 0,
    ]
    score = int(sum(conditions))
    pullback_limit = max(0.08, 3.0 * atr_pct)
    extension_limit = max(0.10, 2.0 * atr_pct)
    extension = close / ema20 - 1.0 if ema20 > 0 else float("nan")

    if (
        score >= 5
        and close > ema20
        and drawdown_from_peak >= -pullback_limit
        and extension <= extension_limit
    ):
        return (
            "HOLD",
            score,
            f"trend {score}/6, close above EMA20>EMA50, pullback within {pullback_limit:.1%}",
        )
    if score >= 3 and close > ema50 and return20 > 0:
        if extension > extension_limit:
            reason = (
                f"trend {score}/6 but price is extended {extension:.1%} above EMA20; "
                "prefer partial profit/shorter holding"
            )
        else:
            reason = (
                f"trend {score}/6 is mixed; use a shorter holding window and respect TP"
            )
        return "SCALP", score, reason
    return (
        "WAIT",
        score,
        f"trend {score}/6 or price structure is broken; do not force a hold",
    )


def _empty_row(ticker: str, *, reason: str) -> dict[str, object]:
    row = {column: np.nan for column in REPORT_COLUMNS}
    row.update({
        "ticker": ticker,
        "data_status": "unavailable",
        "analysis_status": "unavailable",
        "management_mode": "INSUFFICIENT_DATA",
        "management_reason": reason,
        "skip_reason": reason,
    })
    return row


def _analyse_ticker(
    ticker: str,
    prices: pd.DataFrame,
    *,
    market_latest_date: pd.Timestamp,
    fixed_stop_loss: float,
    baseline_atr_multiple: float,
    baseline_take_profit: float,
    lookback_bars: int,
    pivot_left_bars: int,
    pivot_right_bars: int,
    minimum_rebound: float,
    freshness_days: int,
    target_overshoot_margin: float,
) -> dict[str, object]:
    ticker = str(ticker).strip().upper()
    if prices.empty:
        return _empty_row(ticker, reason="no usable OHLCV")

    technical = prices.copy().reset_index(drop=True)
    technical = ATR().compute(technical)
    technical = ReturnFeatures().compute(technical)
    technical["ema20"] = technical["close"].ewm(span=20, adjust=False).mean()
    technical["ema50"] = technical["close"].ewm(span=50, adjust=False).mean()
    latest = technical.iloc[-1]
    latest_date = pd.Timestamp(latest["date"]).normalize()
    data_age_days = int((market_latest_date - latest_date).days)
    data_status = "fresh" if data_age_days <= freshness_days else "stale"

    bottom = _find_recent_bottom(
        technical,
        lookback_bars=lookback_bars,
        pivot_left_bars=pivot_left_bars,
        pivot_right_bars=pivot_right_bars,
        minimum_rebound=minimum_rebound,
    )
    if bottom is None:
        return _empty_row(ticker, reason="not enough rows to identify a recent bottom")

    bottom_index = int(bottom["index"])
    confirmation_index = min(bottom_index + pivot_right_bars, len(technical) - 1)
    entry_index = confirmation_index + 1
    bottom_row = technical.iloc[bottom_index]
    current_price = _float(latest.get("close"))
    if entry_index >= len(technical):
        bottom_path = technical.iloc[bottom_index:]
        bottom_peak = pd.to_numeric(bottom_path["high"], errors="coerce").max()
        bottom_drawdown = _safe_ratio(current_price, bottom_peak) - 1.0
        management_mode, trend_score, management_reason = _management_mode(
            latest,
            drawdown_from_peak=bottom_drawdown,
            data_status=data_status,
        )
        row: dict[str, object] = {column: np.nan for column in REPORT_COLUMNS}
        row.update({
            "ticker": ticker,
            "latest_date": _date_text(latest_date),
            "data_age_days": data_age_days,
            "data_status": data_status,
            "price_rows": len(prices),
            "analysis_status": "pending_entry",
            "bottom_date": _date_text(bottom_row["date"]),
            "bottom_method": str(bottom["method"]),
            "bottom_low": _float(bottom_row.get("low")),
            "bottom_close": _float(bottom_row.get("close")),
            "bottom_confirmed_date": _date_text(
                technical.iloc[confirmation_index]["date"]
            ),
            "bottom_to_now_return_pct": (
                _safe_ratio(current_price, bottom_row.get("low")) - 1.0
            ),
            "atr_reference": _float(latest.get("atr")),
            "atr_reference_pct": _safe_ratio(latest.get("atr"), current_price),
            "atr2_stop_loss_pct": (
                -baseline_atr_multiple * _safe_ratio(latest.get("atr"), current_price)
            ),
            "close_vs_ema20_pct": _safe_ratio(
                latest.get("close"), latest.get("ema20")
            ) - 1.0,
            "close_vs_ema50_pct": _safe_ratio(
                latest.get("close"), latest.get("ema50")
            ) - 1.0,
            "return_20d_pct": _float(latest.get("return_20d")),
            "return_60d_pct": _float(latest.get("return_60d")),
            "rs_20d_pct": _float(latest.get("rs_20d")),
            "rs_60d_pct": _float(latest.get("rs_60d")),
            "trend_score": trend_score if trend_score is not None else np.nan,
            "management_mode": management_mode,
            "management_reason": f"{management_reason}; entry reference pending",
            "fixed_stop_loss_pct": fixed_stop_loss,
            "fixed_sl_first_event": "not_evaluable",
            "atr2_sl_first_event": "not_evaluable",
            "tp8_state": "not_evaluable",
            "tp10_state": "not_evaluable",
            "tp12_state": "not_evaluable",
            "skip_reason": "",
        })
        return {column: row.get(column, np.nan) for column in REPORT_COLUMNS}

    entry_row = technical.iloc[entry_index]
    entry_price = _float(entry_row.get("open"), _float(entry_row.get("close")))
    if not np.isfinite(entry_price) or entry_price <= 0 or not np.isfinite(current_price):
        return _empty_row(ticker, reason="invalid entry or current price")

    path = technical.iloc[entry_index:].reset_index(drop=True)
    path_high = pd.to_numeric(path["high"], errors="coerce")
    path_low = pd.to_numeric(path["low"], errors="coerce")
    peak_position = int(path_high.idxmax()) if path_high.notna().any() else 0
    peak_high = _float(path_high.max())
    peak_date = _date_text(path.iloc[peak_position]["date"])
    max_favorable_excursion = _safe_ratio(peak_high, entry_price) - 1.0
    max_adverse_excursion = _safe_ratio(path_low.min(), entry_price) - 1.0
    drawdown_from_peak = _safe_ratio(current_price, peak_high) - 1.0

    atr_reference = _float(technical.iloc[entry_index - 1].get("atr"))
    atr_reference_pct = _safe_ratio(atr_reference, entry_price)
    atr2_stop_loss = (
        -baseline_atr_multiple * atr_reference_pct
        if np.isfinite(atr_reference_pct)
        else float("nan")
    )
    fixed_stop_price = entry_price * (1.0 + fixed_stop_loss)
    atr2_stop_price = (
        entry_price * (1.0 + atr2_stop_loss)
        if np.isfinite(atr2_stop_loss)
        else float("nan")
    )
    fixed_event, fixed_stop_index, _ = _first_event(
        path,
        stop_price=fixed_stop_price,
        target_price=entry_price * (1.0 + baseline_take_profit),
    )
    atr2_event, atr2_stop_index, _ = (
        _first_event(
            path,
            stop_price=atr2_stop_price,
            target_price=entry_price * (1.0 + baseline_take_profit),
        )
        if np.isfinite(atr2_stop_price)
        else ("none", None, None)
    )
    tp10_index = _first_hit(
        path,
        column="high",
        level=entry_price * (1.0 + baseline_take_profit),
        above=True,
    )

    def _stop_flags(stop_index: int | None) -> dict[str, object]:
        if stop_index is None:
            return {
                "before_tp10": False,
                "then_tp10": False,
                "recovery_to_entry": False,
            }
        after_stop = path.iloc[stop_index:]
        after_high = pd.to_numeric(after_stop["high"], errors="coerce")
        before_tp10 = tp10_index is None or stop_index <= tp10_index
        return {
            "before_tp10": before_tp10,
            "then_tp10": bool(
                before_tp10
                and after_high.notna().any()
                and after_high.max() >= entry_price * (1.0 + baseline_take_profit)
            ),
            "recovery_to_entry": bool(
                after_high.notna().any() and after_high.max() >= entry_price
            ),
        }

    fixed_flags = _stop_flags(fixed_stop_index)
    atr2_flags = _stop_flags(atr2_stop_index)
    management_mode, trend_score, management_reason = _management_mode(
        latest,
        drawdown_from_peak=drawdown_from_peak,
        data_status=data_status,
    )

    row: dict[str, object] = {
        "ticker": ticker,
        "latest_date": _date_text(latest_date),
        "data_age_days": data_age_days,
        "data_status": data_status,
        "price_rows": len(prices),
        "analysis_status": "analyzed",
        "bottom_date": _date_text(bottom_row["date"]),
        "bottom_method": str(bottom["method"]),
        "bottom_low": _float(bottom_row.get("low")),
        "bottom_close": _float(bottom_row.get("close")),
        "bottom_confirmed_date": _date_text(technical.iloc[confirmation_index]["date"]),
        "entry_date": _date_text(entry_row["date"]),
        "entry_price": entry_price,
        "bars_since_entry": len(path),
        "bottom_to_now_return_pct": _safe_ratio(current_price, bottom_row.get("low")) - 1.0,
        "entry_to_now_return_pct": _safe_ratio(current_price, entry_price) - 1.0,
        "peak_date": peak_date,
        "peak_high": peak_high,
        "max_favorable_excursion_pct": max_favorable_excursion,
        "max_adverse_excursion_pct": max_adverse_excursion,
        "peak_to_current_drawdown_pct": drawdown_from_peak,
        "atr_reference": atr_reference,
        "atr_reference_pct": atr_reference_pct,
        "atr2_stop_loss_pct": atr2_stop_loss,
        "close_vs_ema20_pct": _safe_ratio(latest.get("close"), latest.get("ema20")) - 1.0,
        "close_vs_ema50_pct": _safe_ratio(latest.get("close"), latest.get("ema50")) - 1.0,
        "return_20d_pct": _float(latest.get("return_20d")),
        "return_60d_pct": _float(latest.get("return_60d")),
        "rs_20d_pct": _float(latest.get("rs_20d")),
        "rs_60d_pct": _float(latest.get("rs_60d")),
        "trend_score": trend_score if trend_score is not None else np.nan,
        "management_mode": management_mode,
        "management_reason": management_reason,
        "fixed_stop_loss_pct": fixed_stop_loss,
        "fixed_sl_first_event": fixed_event,
        "fixed_sl_event_date": (
            _date_text(path.iloc[fixed_stop_index]["date"])
            if fixed_stop_index is not None
            else ""
        ),
        "fixed_sl_before_tp10": bool(fixed_flags["before_tp10"]),
        "fixed_sl_then_tp10": bool(fixed_flags["then_tp10"]),
        "fixed_sl_recovery_to_entry": bool(fixed_flags["recovery_to_entry"]),
        "atr2_sl_first_event": atr2_event,
        "atr2_sl_event_date": (
            _date_text(path.iloc[atr2_stop_index]["date"])
            if atr2_stop_index is not None
            else ""
        ),
        "atr2_sl_before_tp10": bool(atr2_flags["before_tp10"]),
        "atr2_sl_then_tp10": bool(atr2_flags["then_tp10"]),
        "skip_reason": "",
    }

    # RS is computed after the stock-specific technical series is built so
    # that a missing benchmark row cannot shift the stock's return columns.
    for target_pct in DEFAULT_TP_GRID:
        target_index = _first_hit(
            path,
            column="high",
            level=entry_price * (1.0 + target_pct),
            above=True,
        )
        row.update(_target_metrics(
            path,
            entry_price=entry_price,
            current_price=current_price,
            target_pct=target_pct,
            target_index=target_index,
            overshoot_margin=target_overshoot_margin,
        ))

    row["tp10_non_current_flag"] = bool(
        row["tp10_hit"]
        and _float(row["tp10_current_vs_target_pct"]) >= target_overshoot_margin
    )
    row["tp10_non_peak_flag"] = bool(
        row["tp10_hit"]
        and _float(row["tp10_peak_after_hit_extra_pct"]) >= target_overshoot_margin
    )
    return {column: row.get(column, np.nan) for column in REPORT_COLUMNS}


def _with_relative_strength(
    prices: pd.DataFrame,
    benchmark: pd.DataFrame,
) -> pd.DataFrame:
    """Add benchmark-aligned RS features before the ticker technical pass."""
    return RelativeStrength().compute(prices, benchmark)


def _summary_value(series: pd.Series, *, quantile: float | None = None) -> float:
    values = pd.to_numeric(series, errors="coerce").dropna()
    if values.empty:
        return float("nan")
    return float(values.quantile(quantile) if quantile is not None else values.mean())


def _proportion(frame: pd.DataFrame, column: str, *, denominator: int | None = None) -> float:
    if column not in frame.columns:
        return float("nan")
    total = len(frame) if denominator is None else denominator
    flags = frame[column].fillna(False).astype(bool)
    return float(flags.sum() / total) if total else float("nan")


def _event_proportion(
    frame: pd.DataFrame,
    column: str,
    value: str,
    *,
    denominator: int | None = None,
) -> float:
    if column not in frame.columns:
        return float("nan")
    total = len(frame) if denominator is None else denominator
    return float((frame[column] == value).sum() / total) if total else float("nan")


def _json_safe(value: object) -> object:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    return value


def _build_summary(
    report: pd.DataFrame,
    *,
    metadata: dict[str, object],
) -> dict[str, object]:
    usable = report[report["skip_reason"].fillna("").eq("")].copy()
    entry_ready = usable[usable["analysis_status"] == "analyzed"].copy()
    tp10_hit = entry_ready[entry_ready["tp10_hit"].fillna(False).astype(bool)]
    return {
        "schema_version": ANALYSIS_SCHEMA_VERSION,
        "metadata": {key: _json_safe(value) for key, value in metadata.items()},
        "counts": {
            "tickers_requested": int(metadata["tickers_requested"]),
            "tickers_analyzed": int(len(usable)),
            "tickers_entry_ready": int(len(entry_ready)),
            "tickers_pending_entry": int(
                (usable["analysis_status"] == "pending_entry").sum()
            ),
            "tickers_skipped": int(len(report) - len(usable)),
            "fresh": int((usable["data_status"] == "fresh").sum()),
            "stale": int((usable["data_status"] == "stale").sum()),
            "hold": int((usable["management_mode"] == "HOLD").sum()),
            "scalp": int((usable["management_mode"] == "SCALP").sum()),
            "wait": int((usable["management_mode"] == "WAIT").sum()),
            "fixed_sl_stop_first": int(
                (entry_ready["fixed_sl_first_event"] == "stop").sum()
            ),
            "fixed_sl_then_tp10": int(
                entry_ready["fixed_sl_then_tp10"].fillna(False).astype(bool).sum()
            ),
            "atr2_sl_stop_first": int(
                (entry_ready["atr2_sl_first_event"] == "stop").sum()
            ),
            "atr2_sl_then_tp10": int(
                entry_ready["atr2_sl_then_tp10"].fillna(False).astype(bool).sum()
            ),
            "tp8_hit": int(entry_ready["tp8_hit"].fillna(False).astype(bool).sum()),
            "tp10_hit": int(entry_ready["tp10_hit"].fillna(False).astype(bool).sum()),
            "tp12_hit": int(entry_ready["tp12_hit"].fillna(False).astype(bool).sum()),
            "tp10_non_current": int(
                entry_ready["tp10_non_current_flag"].fillna(False).astype(bool).sum()
            ),
            "tp10_non_peak": int(
                entry_ready["tp10_non_peak_flag"].fillna(False).astype(bool).sum()
            ),
        },
        "rates": {
            "fixed_sl_stop_first": _event_proportion(
                entry_ready,
                "fixed_sl_first_event",
                "stop",
            ),
            "fixed_sl_then_tp10": _proportion(entry_ready, "fixed_sl_then_tp10"),
            "atr2_sl_stop_first": _event_proportion(
                entry_ready,
                "atr2_sl_first_event",
                "stop",
            ),
            "atr2_sl_then_tp10": _proportion(entry_ready, "atr2_sl_then_tp10"),
            "tp10_hit": _proportion(entry_ready, "tp10_hit"),
            "tp10_non_current": _proportion(entry_ready, "tp10_non_current_flag"),
            "tp10_non_peak": _proportion(entry_ready, "tp10_non_peak_flag"),
            "tp10_non_peak_given_hit": _proportion(
                tp10_hit,
                "tp10_non_peak_flag",
                denominator=len(tp10_hit),
            ),
        },
        "medians": {
            "bottom_to_now_return_pct": _summary_value(
                usable["bottom_to_now_return_pct"], quantile=0.50
            ),
            "entry_to_now_return_pct": _summary_value(
                entry_ready["entry_to_now_return_pct"], quantile=0.50
            ),
            "max_favorable_excursion_pct": _summary_value(
                entry_ready["max_favorable_excursion_pct"], quantile=0.50
            ),
            "peak_to_current_drawdown_pct": _summary_value(
                entry_ready["peak_to_current_drawdown_pct"], quantile=0.50
            ),
            "atr2_stop_loss_pct": _summary_value(
                entry_ready["atr2_stop_loss_pct"], quantile=0.50
            ),
            "tp10_peak_after_hit_extra_pct": _summary_value(
                tp10_hit["tp10_peak_after_hit_extra_pct"], quantile=0.50
            ),
        },
    }


def run_bottom_to_now_analysis(
    *,
    research_dir: str | Path = "data/research_kbs_5y",
    universe: str | Iterable[str] = "all",
    fixed_stop_loss: float = -0.005,
    baseline_atr_multiple: float = 2.0,
    baseline_take_profit: float = 0.10,
    lookback_bars: int = DEFAULT_BOTTOM_LOOKBACK_BARS,
    pivot_left_bars: int = DEFAULT_PIVOT_LEFT_BARS,
    pivot_right_bars: int = DEFAULT_PIVOT_RIGHT_BARS,
    minimum_rebound: float = DEFAULT_MINIMUM_REBOUND,
    freshness_days: int = DEFAULT_FRESHNESS_DAYS,
    target_overshoot_margin: float = DEFAULT_TARGET_OVERSHOOT_MARGIN,
    as_of: object | None = None,
    output_dir: str | Path | None = None,
    save: bool = True,
) -> dict[str, object]:
    """Analyze every selected ticker from its latest meaningful bottom to now."""
    if baseline_atr_multiple <= 0:
        raise ValueError("baseline_atr_multiple must be positive")
    if baseline_take_profit <= 0:
        raise ValueError("baseline_take_profit must be positive")
    if fixed_stop_loss >= 0:
        raise ValueError("fixed_stop_loss must be negative")
    if freshness_days < 0:
        raise ValueError("freshness_days cannot be negative")
    if target_overshoot_margin < 0:
        raise ValueError("target_overshoot_margin cannot be negative")

    root = Path(research_dir)
    raw_dir = root / "raw"
    resolved_universe = _resolve_universe(raw_dir, universe)
    closed_date = latest_closed_date(as_of)
    benchmark = _load_prices(raw_dir, "VNINDEX", closed_date)
    if benchmark.empty:
        raise FileNotFoundError(f"Missing usable VNINDEX data in {raw_dir}")

    price_frames: dict[str, pd.DataFrame] = {}
    for ticker in resolved_universe:
        prices = _load_prices(raw_dir, ticker, closed_date)
        if not prices.empty:
            price_frames[ticker] = prices
    if not price_frames:
        raise FileNotFoundError(f"No usable ticker OHLCV data in {raw_dir}")

    market_latest_date = max(
        pd.Timestamp(frame["date"].max()).normalize()
        for frame in [*price_frames.values(), benchmark]
    )
    rows: list[dict[str, object]] = []
    for ticker in resolved_universe:
        prices = price_frames.get(ticker)
        if prices is None:
            rows.append(_empty_row(ticker, reason="ticker file is missing or invalid"))
            continue
        try:
            technical = _with_relative_strength(prices, benchmark)
            row = _analyse_ticker(
                ticker,
                technical,
                market_latest_date=market_latest_date,
                fixed_stop_loss=fixed_stop_loss,
                baseline_atr_multiple=baseline_atr_multiple,
                baseline_take_profit=baseline_take_profit,
                lookback_bars=lookback_bars,
                pivot_left_bars=pivot_left_bars,
                pivot_right_bars=pivot_right_bars,
                minimum_rebound=minimum_rebound,
                freshness_days=freshness_days,
                target_overshoot_margin=target_overshoot_margin,
            )
            rows.append(row)
        except (KeyError, TypeError, ValueError) as exc:
            log.warning("Skipping bottom-to-now analysis for %s: %s", ticker, exc)
            rows.append(_empty_row(ticker, reason=f"analysis error: {exc}"))

    report = pd.DataFrame(rows, columns=REPORT_COLUMNS)
    report = report.sort_values(
        ["management_mode", "entry_to_now_return_pct", "ticker"],
        ascending=[True, False, True],
        na_position="last",
        kind="stable",
    ).reset_index(drop=True)
    universe_label = universe.strip().lower() if isinstance(universe, str) else "custom"
    metadata: dict[str, object] = {
        "generated_at": now_vn().isoformat(),
        "closed_date": closed_date.date().isoformat(),
        "market_latest_date": market_latest_date.date().isoformat(),
        "universe": universe_label,
        "tickers_requested": len(resolved_universe),
        "tickers_with_files": len(price_frames),
        "fixed_stop_loss": fixed_stop_loss,
        "baseline_atr_multiple": baseline_atr_multiple,
        "baseline_take_profit": baseline_take_profit,
        "lookback_bars": lookback_bars,
        "pivot_left_bars": pivot_left_bars,
        "pivot_right_bars": pivot_right_bars,
        "minimum_rebound": minimum_rebound,
        "freshness_days": freshness_days,
        "target_overshoot_margin": target_overshoot_margin,
        "bottom_definition": (
            "latest confirmed local low within the lookback whose future close "
            "rebounded by minimum_rebound; otherwise rolling-low fallback"
        ),
        "entry_definition": "first open after the right-side pivot confirmation window",
        "same_day_collision": "conservative stop-first for event labels",
        "selection_note": "descriptive current-state diagnostic; not a live entry backtest",
    }
    summary = _build_summary(report, metadata=metadata)

    result: dict[str, object] = {
        "report": report,
        "summary": summary,
        "metadata": metadata,
    }
    if save:
        destination = Path(output_dir) if output_dir is not None else root / "research_results"
        destination.mkdir(parents=True, exist_ok=True)
        report_path = destination / "bottom_to_now_analysis.csv"
        summary_path = destination / "bottom_to_now_summary.json"
        report.to_csv(report_path, index=False)
        summary_path.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2, default=_json_safe),
            encoding="utf-8",
        )
        result["report_path"] = str(report_path)
        result["summary_path"] = str(summary_path)
        log.info("Saved bottom-to-now report for %d tickers to %s", len(report), report_path)
    return result
