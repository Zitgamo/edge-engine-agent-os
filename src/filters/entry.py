"""Configurable, explainable entry eligibility filters."""

from __future__ import annotations

import pandas as pd


def _percentile(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    return values.rank(method="average", pct=True)


def apply_entry_filters(
    ranking: pd.DataFrame,
    features: pd.DataFrame,
    config,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Filter the latest ensemble ranking and return an audit report.

    The filter is deliberately fail-closed when enabled: missing breadth or
    eligibility features means no new signal is published.  The production
    default enables this guard; it can be disabled for a controlled comparison.
    """
    report: dict[str, object] = {
        "enabled": bool(getattr(config, "enable_entry_filters", False)),
        "status": "disabled",
        "reason": None,
        "input_count": int(len(ranking)),
        "output_count": 0,
    }
    if ranking.empty:
        report.update(status="blocked", reason="empty ranking")
        return ranking.copy(), report
    if not report["enabled"]:
        report.update(status="passed", reason="feature flag disabled", output_count=len(ranking))
        return ranking.copy(), report

    required_ranking = {"date", "ticker", "score"}
    missing_ranking = sorted(required_ranking - set(ranking.columns))
    if missing_ranking:
        report.update(status="blocked", reason=f"ranking missing columns: {missing_ranking}")
        return ranking.iloc[0:0].copy(), report

    ranked = ranking.copy()
    ranked["date"] = pd.to_datetime(ranked["date"], errors="coerce").dt.normalize()
    latest_date = ranked["date"].max()
    latest_features = features.copy()
    latest_features["date"] = pd.to_datetime(
        latest_features["date"], errors="coerce"
    ).dt.normalize()
    latest_features = latest_features[latest_features["date"] == latest_date].copy()
    if latest_features.empty:
        report.update(status="blocked", reason="no feature snapshot for ranking date")
        return ranked.iloc[0:0].copy(), report

    if "return_20d" not in latest_features.columns:
        report.update(status="blocked", reason="return_20d is unavailable for breadth")
        return ranked.iloc[0:0].copy(), report
    breadth_values = pd.to_numeric(latest_features["return_20d"], errors="coerce").dropna()
    if breadth_values.empty:
        report.update(status="blocked", reason="market breadth has no valid observations")
        return ranked.iloc[0:0].copy(), report
    breadth = float((breadth_values > 0).mean())
    min_breadth = float(getattr(config, "min_market_breadth_20d", 0.50))
    report["market_breadth_20d"] = breadth
    report["min_market_breadth_20d"] = min_breadth
    if breadth < min_breadth:
        report.update(status="blocked", reason="market breadth below threshold")
        return ranked.iloc[0:0].copy(), report

    eligible = latest_features[["ticker"]].copy()
    eligible["ticker"] = eligible["ticker"].astype(str)
    if "atr_pct" not in latest_features.columns:
        report.update(status="blocked", reason="atr_pct is unavailable for eligibility")
        return ranked.iloc[0:0].copy(), report
    eligible["atr_percentile"] = _percentile(latest_features["atr_pct"])
    min_atr = float(getattr(config, "min_entry_atr_percentile", 0.20))
    eligible = eligible[eligible["atr_percentile"] >= min_atr]

    min_trend = float(getattr(config, "min_entry_trend_percentile", 0.0))
    if min_trend > 0:
        trend_parts = []
        for column in ("return_60d", "rs_60d"):
            if column in latest_features.columns:
                trend_parts.append(_percentile(latest_features[column]).rename(column))
        if not trend_parts:
            report.update(status="blocked", reason="trend features are unavailable")
            return ranked.iloc[0:0].copy(), report
        trend = pd.concat(trend_parts, axis=1).mean(axis=1, skipna=True)
        eligible = eligible.join(trend.rename("trend_percentile"), how="left")
        eligible = eligible[eligible["trend_percentile"] >= min_trend]
    report["min_entry_atr_percentile"] = min_atr
    report["min_entry_trend_percentile"] = min_trend

    allowed = set(eligible["ticker"])
    ranked["ticker"] = ranked["ticker"].astype(str)
    filtered = ranked[ranked["ticker"].isin(allowed)].copy()
    filtered = filtered.sort_values("score", ascending=False).reset_index(drop=True)
    filtered["rank"] = range(1, len(filtered) + 1)
    min_picks = max(1, int(getattr(config, "min_entry_picks", 3)))
    report["eligible_count"] = int(len(allowed))
    report["min_entry_picks"] = min_picks
    report["output_count"] = int(len(filtered))
    if len(filtered) < min_picks:
        report.update(status="blocked", reason="fewer than minimum eligible picks")
        return filtered.iloc[0:0].copy(), report

    min_score = float(getattr(config, "min_entry_score", 0.0))
    top_score = float(filtered["score"].iloc[0])
    report["min_entry_score"] = min_score
    report["top_score"] = top_score
    if top_score < min_score:
        report.update(status="blocked", reason="top score below minimum")
        return filtered.iloc[0:0].copy(), report

    score_margin = float(getattr(config, "min_entry_score_margin", 0.0))
    report["min_entry_score_margin"] = score_margin
    if len(filtered) > min_picks and score_margin > 0:
        cutoff_score = float(filtered["score"].iloc[min_picks - 1])
        next_score = float(filtered["score"].iloc[min_picks])
        margin = cutoff_score - next_score
        report["cutoff_score_margin"] = margin
        if margin < score_margin:
            report.update(status="blocked", reason="score margin below threshold")
            return filtered.iloc[0:0].copy(), report

    report.update(status="passed", reason="all entry filters passed")
    return filtered, report
