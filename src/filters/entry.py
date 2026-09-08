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
        "input_count": len(ranking),
        "output_count": 0,
        "stages": [],
    }
    def finish(result):
        report["published_candidate_count"] = len(result)
        return result, report

    def stage(name, before, after):
        report["stages"].append({"gate": name, "before": int(before),
                                 "after": int(after), "removed": int(before - after)})
    if ranking.empty:
        report.update(status="blocked", reason="empty ranking")
        return finish(ranking.copy())
    if not report["enabled"]:
        report.update(status="passed", reason="feature flag disabled", output_count=len(ranking))
        return finish(ranking.copy())

    required_ranking = {"date", "ticker", "score"}
    missing_ranking = sorted(required_ranking - set(ranking.columns))
    if missing_ranking:
        report.update(status="blocked", reason=f"ranking missing columns: {missing_ranking}")
        return finish(ranking.iloc[0:0].copy())

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
        return finish(ranked.iloc[0:0].copy())

    if "return_20d" not in latest_features.columns:
        report.update(status="blocked", reason="return_20d is unavailable for breadth")
        return finish(ranked.iloc[0:0].copy())
    breadth_values = pd.to_numeric(latest_features["return_20d"], errors="coerce").dropna()
    if breadth_values.empty:
        report.update(status="blocked", reason="market breadth has no valid observations")
        return finish(ranked.iloc[0:0].copy())
    breadth = float((breadth_values > 0).mean())
    min_breadth = float(getattr(config, "min_market_breadth_20d", 0.50))
    report["market_breadth_20d"] = breadth
    report["min_market_breadth_20d"] = min_breadth
    stage("market_breadth", len(ranked), len(ranked) if breadth >= min_breadth else 0)
    if breadth < min_breadth:
        report.update(status="blocked", reason="market breadth below threshold")
        return finish(ranked.iloc[0:0].copy())

    eligible = latest_features[["ticker"]].copy()
    eligible["ticker"] = eligible["ticker"].astype(str)
    if "atr_pct" not in latest_features.columns:
        report.update(status="blocked", reason="atr_pct is unavailable for eligibility")
        return finish(ranked.iloc[0:0].copy())
    eligible["atr_percentile"] = _percentile(latest_features["atr_pct"])
    min_atr = float(getattr(config, "min_entry_atr_percentile", 0.20))
    eligible = eligible[eligible["atr_percentile"] >= min_atr]
    atr_count = int(ranked["ticker"].astype(str).isin(set(eligible["ticker"])).sum())
    stage("atr", len(ranked), atr_count)

    min_trend = float(getattr(config, "min_entry_trend_percentile", 0.0))
    if min_trend > 0:
        trend_parts = []
        for column in ("return_60d", "rs_60d"):
            if column in latest_features.columns:
                trend_parts.append(_percentile(latest_features[column]).rename(column))
        if not trend_parts:
            report.update(status="blocked", reason="trend features are unavailable")
            return finish(ranked.iloc[0:0].copy())
        trend = pd.concat(trend_parts, axis=1).mean(axis=1, skipna=True)
        eligible = eligible.join(trend.rename("trend_percentile"), how="left")
        eligible = eligible[eligible["trend_percentile"] >= min_trend]
        trend_count = int(ranked["ticker"].astype(str).isin(set(eligible["ticker"])).sum())
        stage("trend", atr_count, trend_count)
    report["min_entry_atr_percentile"] = min_atr
    report["min_entry_trend_percentile"] = min_trend

    allowed = set(eligible["ticker"])
    ranked["ticker"] = ranked["ticker"].astype(str)
    filtered = ranked[ranked["ticker"].isin(allowed)].copy()
    filtered = filtered.sort_values("score", ascending=False).reset_index(drop=True)
    filtered["rank"] = range(1, len(filtered) + 1)
    min_picks = max(1, int(getattr(config, "min_entry_picks", 3)))
    report["eligible_count"] = len(allowed)
    report["min_entry_picks"] = min_picks
    report["output_count"] = len(filtered)
    stage("minimum_picks", len(filtered), len(filtered) if len(filtered) >= min_picks else 0)
    if len(filtered) < min_picks:
        report.update(status="blocked", reason="fewer than minimum eligible picks")
        return finish(filtered.iloc[0:0].copy())

    min_score = float(getattr(config, "min_entry_score", 0.0))
    top_score = float(filtered["score"].iloc[0])
    report["min_entry_score"] = min_score
    report["top_score"] = top_score
    stage("top_score", len(filtered), len(filtered) if top_score >= min_score else 0)
    if top_score < min_score:
        report.update(status="blocked", reason="top score below minimum")
        return finish(filtered.iloc[0:0].copy())

    score_margin = float(getattr(config, "min_entry_score_margin", 0.0))
    report["min_entry_score_margin"] = score_margin
    if len(filtered) > min_picks and score_margin > 0:
        cutoff_score = float(filtered["score"].iloc[min_picks - 1])
        next_score = float(filtered["score"].iloc[min_picks])
        margin = cutoff_score - next_score
        report["cutoff_score_margin"] = margin
        stage("score_margin", len(filtered), len(filtered) if margin >= score_margin else 0)
        if margin < score_margin:
            report.update(status="blocked", reason="score margin below threshold")
            return finish(filtered.iloc[0:0].copy())

    report.update(status="passed", reason="all entry filters passed")
    return finish(filtered)
