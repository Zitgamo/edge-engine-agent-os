from __future__ import annotations

import logging

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

CEILING_THRESHOLD = 0.065
FLOOR_THRESHOLD = -0.065
LOOKBACK_DAYS = 60
DEEP_DRAWDOWN_THRESHOLD = -0.30
CAPITULATION_VOLUME_THRESHOLD = 3.0


def analyze_stock(df: pd.DataFrame, ticker: str | None = None) -> dict:
    """Analyze a single stock's recent price action for ceiling/floor context.

    Returns a dict with:
    - drawdown_60d: max drawdown in lookback period
    - consecutive_floors: longest consecutive floor run
    - consecutive_ceilings: longest consecutive ceiling run
    - floor_volume_ratio: avg volume on floor days / avg volume overall
    - recovery_pct: % recovery from trough
    - context_label: 'capitulation_bounce' | 'dead_cat_bounce' | 'overbought' | 'normal'
    - score_adjustment: float to add/subtract from ensemble score
    """
    df = df.sort_values("date").tail(LOOKBACK_DAYS).copy()
    closes = df["close"].values
    dates = df["date"].values
    volumes = df["volume"].values if "volume" in df.columns else None

    if len(closes) < 10:
        return {"context_label": "normal", "score_adjustment": 0.0}

    peak = float(np.max(closes))
    trough = float(np.min(closes))
    peak_idx = int(np.argmax(closes))
    trough_idx = int(np.argmin(closes))
    current = float(closes[-1])
    drawdown = (trough - peak) / peak
    recovery = (current - trough) / trough if trough > 0 else 0.0

    pct_changes = np.diff(closes) / closes[:-1]
    is_floor = pct_changes <= FLOOR_THRESHOLD
    is_ceiling = pct_changes >= CEILING_THRESHOLD

    def max_consecutive(arr: np.ndarray) -> int:
        best = 0
        cur = 0
        for v in arr:
            cur = (cur + 1) if v else 0
            best = max(best, cur)
        return best

    max_floor_run = max_consecutive(is_floor)
    max_ceiling_run = max_consecutive(is_ceiling)

    floor_volume_ratio = 1.0
    if volumes is not None and len(volumes) > 1 and is_floor.any():
        avg_vol_all = float(np.mean(volumes[1:]))
        avg_vol_floor = float(np.mean(volumes[1:][is_floor]))
        if avg_vol_all > 0:
            floor_volume_ratio = avg_vol_floor / avg_vol_all

    recent_floor_count = int(is_floor[-10:].sum())
    recent_ceiling_count = int(is_ceiling[-10:].sum())

    context_label = "normal"
    score_adjustment = 0.0

    if drawdown < DEEP_DRAWDOWN_THRESHOLD and max_floor_run >= 3:
        if max_ceiling_run >= 1 and floor_volume_ratio > CAPITULATION_VOLUME_THRESHOLD:
            context_label = "capitulation_bounce"
            score_adjustment = 0.05
        elif max_ceiling_run >= 2:
            context_label = "capitulation_bounce"
            score_adjustment = 0.03
        elif max_ceiling_run == 0 and recent_floor_count >= 3:
            context_label = "free_fall"
            score_adjustment = -0.03
        else:
            context_label = "deep_value"
            score_adjustment = 0.02
    elif max_ceiling_run >= 2 and drawdown > -0.15:
        context_label = "overbought"
        score_adjustment = -0.05
    elif max_ceiling_run >= 1 and recent_floor_count == 0 and drawdown > -0.10:
        context_label = "overbought"
        score_adjustment = -0.03
    elif max_ceiling_run >= 1 and drawdown < DEEP_DRAWDOWN_THRESHOLD:
        context_label = "recovery_bounce"
        score_adjustment = 0.02

    result = {
        "ticker": ticker or df.get("ticker", [""])[0] if isinstance(df.get("ticker"), pd.Series) else "",
        "drawdown_60d": round(drawdown, 4),
        "consecutive_floors": max_floor_run,
        "consecutive_ceilings": max_ceiling_run,
        "floor_volume_ratio": round(floor_volume_ratio, 2),
        "recovery_pct": round(recovery, 4),
        "recent_floor_count": recent_floor_count,
        "recent_ceiling_count": recent_ceiling_count,
        "context_label": context_label,
        "score_adjustment": score_adjustment,
    }

    if ticker:
        log.info("Ceiling context [%s]: %s (adj=%.3f)", ticker, context_label, score_adjustment)

    return result


def adjust_rankings(
    rankings: pd.DataFrame,
    raw_data_dir: str | None = None,
    ticker_col: str = "ticker",
    score_col: str = "score",
) -> pd.DataFrame:
    """Post-process rankings: adjust scores based on ceiling/floor context."""
    import glob
    import os
    from pathlib import Path

    from src.config import Config

    if raw_data_dir is None:
        raw_data_dir = str(Config.raw_data_dir)

    df = rankings.copy()
    adjustments = []

    for _, row in df.iterrows():
        t = row[ticker_col]
        pattern = os.path.join(raw_data_dir, f"{t}_raw.parquet")
        files = glob.glob(pattern)
        if not files:
            adjustments.append(0.0)
            continue
        try:
            stock_df = pd.read_parquet(files[0])
            ctx = analyze_stock(stock_df, ticker=t)
            adjustments.append(ctx["score_adjustment"])
        except Exception:
            adjustments.append(0.0)

    df["ceiling_adjustment"] = adjustments
    df["adjusted_score"] = df[score_col] + df["ceiling_adjustment"]
    df = df.sort_values("adjusted_score", ascending=False).reset_index(drop=True)
    df["adjusted_rank"] = range(1, len(df) + 1)

    log.info("Ceiling adjustment applied — top 3: %s",
             list(df.head(3)["ticker"]))
    return df


def report_ceiling_context(tickers: list[str], raw_data_dir: str | None = None) -> list[dict]:
    """Generate detailed ceiling context report for a list of tickers."""
    import os
    import glob

    if raw_data_dir is None:
        from src.config import Config
        raw_data_dir = str(Config.raw_data_dir)

    results = []
    for t in tickers:
        pattern = os.path.join(raw_data_dir, f"{t}_raw.parquet")
        files = glob.glob(pattern)
        if not files:
            log.warning("No data for %s", t)
            continue
        try:
            stock_df = pd.read_parquet(files[0])
            ctx = analyze_stock(stock_df, ticker=t)
            results.append(ctx)
        except Exception as e:
            log.warning("Error analyzing %s: %s", t, e)

    return results
