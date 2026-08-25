from __future__ import annotations

import logging
from datetime import time

import pandas as pd
import xgboost as xgb

from src.config import Config
from src.data.collector import OHLCVCollector
from src.data.storage import PriceStorage
from src.data.universe import filter_quality, get_ticker_universe
from src.data.validator import DataValidator
from src.database import init_db, save_pipeline_run, save_signals
from src.features.fundamental import add_fundamental_features
from src.features.macro import add_macro_features
from src.features.returns import ReturnFeatures
from src.features.rs import RelativeStrength
from src.features.strategy import add_strategy_features
from src.features.volatility import ATR
from src.features.volume import VolumeSurge
from src.labels.execution import add_execution_labels
from src.labels.outperformance import OutperformanceLabel
from src.logging_setup import setup_logging
from src.model.blend import blend_horizon_scores
from src.model.evaluator import ModelEvaluator
from src.model.schema import (
    ENSEMBLE_HORIZONS as _ENSEMBLE_HORIZONS,
)
from src.model.schema import (
    FEATURE_COLS,
    MODEL_VERSION,
    XGBOOST_PARAMS,
)
from src.model.schema import (
    HOLDING_PERIOD as _HOLDING_PERIOD,
)
from src.model.schema import (
    N_PICKS as _N_PICKS,
)
from src.model.splits import purged_recent_train_window, recent_date_window
from src.model.targets import target_spec
from src.model.trainer import ModelTrainer  # noqa: F401  (kept for backwards compat)
from src.ranking.signal import SignalGenerator
from src.time_utils import now_vn, today_vn

log = logging.getLogger(__name__)

# Re-export for any external consumers; canonical values live in src.model.schema
ENSEMBLE_HORIZONS = _ENSEMBLE_HORIZONS
TRAIN_SPLIT = 0.8
N_PICKS = _N_PICKS
HOLDING_PERIOD = _HOLDING_PERIOD
STOP_LOSS = -0.03
TAKE_PROFIT = 0.08


def _publish_no_trade(
    storage: PriceStorage,
    signal_date: str,
    metrics: dict[str, float],
    *,
    status: str,
    config: Config | None = None,
) -> None:
    """Publish an explicit no-trade state without retaining stale picks."""
    save_pipeline_run(metrics, status=status, run_key=signal_date)

    # Clear the complete local publication before backfilling older dates.
    # Otherwise sync_actuals() could upload actuals for signals that the
    # no-trade result just removed.
    from src.database import backfill_actuals, clear_publication_for_date

    clear_publication_for_date(signal_date)
    storage.save_processed(pd.DataFrame(), "ranking.parquet")
    storage.save_processed(pd.DataFrame(), "signal.parquet")
    backfill_actuals(holding_period=HOLDING_PERIOD, config=config)

    from src.supabase_client import (
        clear_publication_for_date as clear_cloud_publication_for_date,
    )
    from src.supabase_client import (
        sync_all,
    )

    clear_cloud_publication_for_date(signal_date)
    sync_all(config=config)


def _closed_market_sessions(
    df: pd.DataFrame,
    as_of=None,
) -> pd.DataFrame:
    """Exclude today's partial bar until the Vietnam session is closed."""
    if df.empty or "date" not in df.columns:
        return df.copy()

    result = df.copy()
    dates = pd.to_datetime(result["date"], errors="coerce")
    if dates.dt.tz is not None:
        dates = dates.dt.tz_localize(None)
    result["date"] = dates.dt.normalize()

    as_of = as_of or now_vn()
    cutoff = pd.Timestamp(as_of.date())
    if as_of.time() >= time(15, 30):
        cutoff += pd.Timedelta(days=1)
    return result[result["date"] < cutoff].reset_index(drop=True)


def _latest_published_signal_date() -> pd.Timestamp | None:
    """Read the latest published signal date from cloud or local storage."""
    candidates: list[pd.Timestamp] = []

    try:
        from src.supabase_client import get_client

        client = get_client()
        if client is not None:
            rows = client.get_signals(limit=1)
            if rows:
                remote_date = pd.to_datetime(rows[0].get("signal_date"), errors="coerce")
                if pd.notna(remote_date):
                    candidates.append(pd.Timestamp(remote_date).normalize())
    except Exception as exc:
        # A cloud read outage must not prevent a local/manual pipeline run.
        log.warning("Cannot read latest cloud signal date: %s", exc)

    try:
        from src.database import get_conn

        conn = get_conn()
        local_date = conn.execute("SELECT MAX(signal_date) FROM signals").fetchone()[0]
        conn.close()
        local_date = pd.to_datetime(local_date, errors="coerce")
        if pd.notna(local_date):
            candidates.append(pd.Timestamp(local_date).normalize())
    except Exception as exc:
        log.warning("Cannot read latest local signal date: %s", exc)

    return max(candidates) if candidates else None


def _should_run_for_market_date(market_date: pd.Timestamp | str) -> bool:
    """Return false when this market session already has a published signal."""
    market_date = pd.Timestamp(market_date).normalize()
    published_date = _latest_published_signal_date()
    if published_date is not None and market_date <= published_date:
        log.info(
            "No new market session: latest market=%s, published signal=%s; skipping",
            market_date.date(),
            published_date.date(),
        )
        return False
    return True


def _enforce_execution_horizon_quality(
    metrics: dict[str, float],
    *,
    horizon: int,
    config: Config,
) -> None:
    """Block publication unless the execution-horizon ranking has real edge."""
    if horizon != HOLDING_PERIOD:
        return

    quality_dates = metrics.get("execution_evaluation_dates")
    top3_return = metrics.get("execution_top3_excess_return")
    top3_spread = metrics.get("execution_top3_spread")
    if quality_dates is None or top3_return is None or top3_spread is None:
        raise RuntimeError(
            f"T+{horizon} model quality gate failed: execution quality metrics "
            "are unavailable. No new signal was published."
        )

    if quality_dates < config.min_model_quality_dates:
        raise RuntimeError(
            f"T+{horizon} model quality gate failed: only {quality_dates:.0f} "
            f"execution ranking dates; minimum is {config.min_model_quality_dates}."
        )
    if top3_return < config.min_model_top3_excess_return:
        raise RuntimeError(
            f"T+{horizon} model quality gate failed: execution top-3 excess return="
            f"{top3_return:.2%} < minimum {config.min_model_top3_excess_return:.2%}."
        )
    if top3_spread < config.min_model_top3_spread:
        raise RuntimeError(
            f"T+{horizon} model quality gate failed: execution top-3 spread="
            f"{top3_spread:.2%} < minimum {config.min_model_top3_spread:.2%}."
        )

    roc_auc = metrics.get("roc_auc")
    if roc_auc is not None and roc_auc < config.min_model_roc_auc:
        log.warning(
            "T+%d class ROC-AUC=%.3f is below %.3f, but execution ranking gates passed "
            "(%d dates; top3 return=%.2f%%; spread=%.2f%%)",
            horizon,
            roc_auc,
            config.min_model_roc_auc,
            quality_dates,
            top3_return * 100,
            top3_spread * 100,
        )


def _execution_quality_metrics(
    model: xgb.XGBClassifier,
    df: pd.DataFrame,
    *,
    config: Config,
    holding_period: int,
    top_n: int = N_PICKS,
) -> dict[str, float]:
    """Measure model ranking edge using executable next-open trade outcomes.

    The publication gate uses the same next-open, settlement-delay, SL/TP and
    benchmark logic used by actuals/backtest, regardless of which supervised
    target is selected. This prevents a model with a good proxy label but no
    executable edge from being published.
    """
    from src.backtest import _sltp_excess_return

    required = [*FEATURE_COLS, "date", "ticker"]
    quality = df.dropna(subset=required).copy()
    if quality.empty:
        return {}
    quality["date"] = pd.to_datetime(quality["date"], errors="coerce").dt.normalize()
    quality = quality.dropna(subset=["date"])
    if quality.empty:
        return {}
    quality["quality_score"] = model.predict_proba(
        quality[FEATURE_COLS].fillna(0)
    )[:, 1]

    prices_cache: dict[str, pd.DataFrame] = {}
    daily: list[dict[str, float]] = []
    for signal_date, day in quality.groupby("date"):
        executable: list[dict[str, float]] = []
        date_key = pd.Timestamp(signal_date).date().isoformat()
        for row in day.itertuples(index=False):
            excess_return = _sltp_excess_return(
                str(row.ticker),
                date_key,
                config.stop_loss,
                config.take_profit,
                holding_period,
                prices_cache,
                config.round_trip_cost,
                data_dir=config.raw_data_dir,
            )
            if excess_return is not None and pd.notna(excess_return):
                executable.append({
                    "score": float(row.quality_score),
                    "excess_return": float(excess_return),
                })
        if len(executable) < top_n:
            continue
        returns = pd.DataFrame(executable)
        top = returns.nlargest(top_n, "score")
        top_return = float(top["excess_return"].mean())
        universe_return = float(returns["excess_return"].mean())
        daily.append({
            "top3_win_rate": float((top["excess_return"] > 0).mean()),
            "top3_excess_return": top_return,
            "universe_excess_return": universe_return,
            "top3_spread": top_return - universe_return,
        })

    if not daily:
        return {}
    ranking_metrics = pd.DataFrame(daily)
    return {
        "execution_evaluation_dates": float(len(ranking_metrics)),
        "execution_top3_win_rate": float(ranking_metrics["top3_win_rate"].mean()),
        "execution_top3_excess_return": float(
            ranking_metrics["top3_excess_return"].mean()
        ),
        "execution_universe_excess_return": float(
            ranking_metrics["universe_excess_return"].mean()
        ),
        "execution_top3_spread": float(ranking_metrics["top3_spread"].mean()),
    }


def run_pipeline(
    config: Config | None = None,
    *,
    force: bool = False,
) -> None:
    setup_logging()
    config = config or Config()
    config.ensure_dirs()
    init_db()

    log.info("=== Pipeline started%s ===", " (forced rebuild)" if force else "")

    collector = OHLCVCollector(config)
    storage = PriceStorage(config)
    validator = DataValidator()
    run_time = now_vn()

    universe = get_ticker_universe()
    bm = _closed_market_sessions(collector.fetch("VNINDEX", days=365), run_time)
    benchmark_errors = validator.validate(bm)
    if benchmark_errors or bm.empty:
        raise RuntimeError(f"Benchmark data failed validation: {benchmark_errors}")
    valid_benchmark_sources = {"yahoo", "vnstock_vci"}
    if str(config.data_source).strip().lower() == "yfinance" and collector.last_benchmark_source not in valid_benchmark_sources:
        raise RuntimeError(
            "Refusing to publish signals without the real VNINDEX benchmark "
            f"(source={collector.last_benchmark_source})"
        )
    storage.save_raw(bm, "VNINDEX_raw.parquet")

    latest_benchmark_date = pd.Timestamp(bm["date"].max()).normalize()
    if not force and not _should_run_for_market_date(latest_benchmark_date):
        return

    all_dfs: list[pd.DataFrame] = []
    collected = 0
    skipped = 0
    for ticker in universe:
        try:
            df = collector.fetch(ticker, days=365)
        except Exception as exc:
            log.exception("Failed to collect %s: %s", ticker, exc)
            skipped += 1
            continue
        df = _closed_market_sessions(df, run_time)
        df = filter_quality(df, ticker)
        if df is None:
            skipped += 1
            continue
        errors = validator.validate(df)
        if errors:
            log.warning("Skipping %s due to validation errors: %s", ticker, errors)
            skipped += 1
            continue
        storage.save_raw(df, f"{ticker}_raw.parquet")
        all_dfs.append(df)
        collected += 1
    log.info("Collected %d/%d tickers (skipped %d)", collected, len(universe), skipped)
    minimum_collected = max(30, int(len(universe) * 0.5))
    if collected < minimum_collected:
        raise RuntimeError(
            f"Only collected {collected}/{len(universe)} tickers; refusing to publish a partial run"
        )

    combined = pd.concat(all_dfs, ignore_index=True)
    storage.save_raw(combined, "all_stocks_raw.parquet")

    log.info("=== Computing features ===")
    return_features = ReturnFeatures()
    rs = RelativeStrength()
    atr = ATR()
    volume_surge = VolumeSurge()

    feature_dfs: list[pd.DataFrame] = []
    for df in all_dfs:
        df = return_features.compute(df)
        df = rs.compute(df, bm)
        df = atr.compute(df)
        df = volume_surge.compute(df)
        feature_dfs.append(df)

    features = add_strategy_features(pd.concat(feature_dfs, ignore_index=True))

    log.info("=== Adding macro features ===")
    features = add_macro_features(features)

    log.info("=== Adding fundamental features ===")
    features = add_fundamental_features(features)

    label_dfs: list[pd.DataFrame] = []
    for df in all_dfs:
        for h in ENSEMBLE_HORIZONS:
            labeler = OutperformanceLabel()
            df = labeler.compute(df, bm, horizon=h)
        label_dfs.append(df)

    labels = pd.concat(label_dfs, ignore_index=True)
    merge_cols = ["date", "ticker"]
    for h in ENSEMBLE_HORIZONS:
        if f"outperform_{h}d" in labels.columns:
            merge_cols += [
                f"outperform_{h}d",
                f"excess_return_{h}d",
                f"label_end_date_{h}d",
            ]
    features = features.merge(labels[merge_cols], on=["date", "ticker"], how="left")

    if config.execution_target_enabled:
        log.info("=== Adding execution-aligned T+%d labels ===", HOLDING_PERIOD)
        execution_keys = pd.concat(
            [df[["date", "ticker"]] for df in all_dfs],
            ignore_index=True,
        )
        execution_labels = add_execution_labels(
            execution_keys,
            raw_data_dir=config.raw_data_dir,
            stop_loss=config.stop_loss,
            take_profit=config.take_profit,
            holding_period=HOLDING_PERIOD,
            round_trip_cost=config.round_trip_cost,
        )
        execution_spec = target_spec(
            HOLDING_PERIOD,
            use_execution_target=True,
        )
        features = features.merge(
            execution_labels[["date", "ticker", execution_spec.target_col,
                              execution_spec.return_col, execution_spec.label_end_col]],
            on=["date", "ticker"],
            how="left",
        )

    storage.save_processed(features, "features.parquet")

    log.info("=== Walk-forward train/test split ===")
    target_specs = {
        h: target_spec(
            h,
            use_execution_target=config.execution_target_enabled,
        )
        for h in ENSEMBLE_HORIZONS
    }
    label_cols = [spec.target_col for spec in target_specs.values()]
    label_end_cols = [spec.label_end_col for spec in target_specs.values()]
    trainable = features.dropna(subset=FEATURE_COLS + label_cols + label_end_cols)
    dates = sorted(trainable["date"].unique())
    if len(dates) < max(30, config.model_quality_test_days + 1):
        raise RuntimeError(f"Only {len(dates)} usable dates remain after feature/label construction")
    train_by_horizon: dict[int, pd.DataFrame] = {}
    test_by_horizon: dict[int, pd.DataFrame] = {}
    for h in ENSEMBLE_HORIZONS:
        spec = target_specs[h]
        target = spec.target_col
        label_end_col = spec.label_end_col
        eligible = features.dropna(subset=FEATURE_COLS + [target, label_end_col])
        horizon_dates = sorted(pd.to_datetime(eligible["date"]).dt.normalize().unique())
        quality_days = min(config.model_quality_test_days, len(horizon_dates) - 1)
        if quality_days <= 0:
            raise RuntimeError(f"No quality-test dates available for T+{h}")
        test_start = pd.Timestamp(horizon_dates[-quality_days]).normalize()
        train_h = purged_recent_train_window(
            eligible,
            test_start=test_start,
            label_end_col=label_end_col,
            max_dates=config.model_training_days,
        )
        test_h = eligible[eligible["date"] >= test_start].copy()
        if train_h.empty or test_h.empty:
            raise RuntimeError(f"Purged split produced no train/test rows for T+{h}")
        if train_h[target].nunique() < 2:
            raise RuntimeError(f"Training target for T+{h} contains only one class")
        train_by_horizon[h] = train_h
        test_by_horizon[h] = test_h
        log.info(
            "T+%d train: %d rows | test: %d rows | rolling window=%d dates | purged before %s",
            h,
            len(train_h),
            len(test_h),
            config.model_training_days,
            test_start.date(),
        )

    log.info("=== Training ensemble models (walk-forward) ===")
    ensemble_models: dict[int, xgb.XGBClassifier] = {}
    for h in ENSEMBLE_HORIZONS:
        target = target_specs[h].target_col
        train_h = train_by_horizon[h]
        X_train = train_h[FEATURE_COLS]
        y_train = train_h[target]
        model = xgb.XGBClassifier(**XGBOOST_PARAMS)
        model.fit(X_train, y_train, verbose=False)
        ensemble_models[h] = model
        log.info("Trained T+%d on %d samples", h, len(X_train))

    log.info("=== Walk-forward evaluation on test set ===")
    evaluator = ModelEvaluator()
    all_metrics = {}
    for h, model in ensemble_models.items():
        spec = target_specs[h]
        target = spec.target_col
        metrics = evaluator.evaluate(model, test_by_horizon[h], target_col=target)
        if h == HOLDING_PERIOD:
            metrics.update(
                _execution_quality_metrics(
                    model,
                    test_by_horizon[h],
                    config=config,
                    holding_period=h,
                )
            )
        all_metrics[f"T+{h}"] = metrics
        log.info(
            "T+%d %s label metrics: %s | positive rate=%.2f%% | majority baseline=%.2f%%",
            h,
            "execution" if spec.execution else "close-to-close",
            metrics,
            metrics["positive_rate"] * 100,
            metrics["majority_baseline_accuracy"] * 100,
        )
        if h == HOLDING_PERIOD:
            log.info(
                "T+%d executable ranking metrics: dates=%s top3_return=%s spread=%s",
                h,
                metrics.get("execution_evaluation_dates", 0),
                f"{metrics['execution_top3_excess_return']:.2%}"
                if "execution_top3_excess_return" in metrics else "N/A",
                f"{metrics['execution_top3_spread']:.2%}"
                if "execution_top3_spread" in metrics else "N/A",
            )
        try:
            _enforce_execution_horizon_quality(metrics, horizon=h, config=config)
        except RuntimeError as exc:
            # A quality failure is an expected market state, not an unhandled
            # runner crash. Persist the diagnosis, keep the previous
            # production model untouched, and publish an explicit no-trade
            # result so scheduled workflows remain observable and idempotent.
            latest_market_date = pd.Timestamp(latest_benchmark_date).date()
            log.error("Model quality gate blocked publication: %s", exc)
            _publish_no_trade(
                storage,
                latest_market_date.isoformat(),
                metrics,
                status="quality_failed",
                config=config,
            )
            log.warning("No trade published because the execution quality gate failed")
            return

    log.info("=== Retrain on full data for production ===")
    for h in ENSEMBLE_HORIZONS:
        spec = target_specs[h]
        target = spec.target_col
        label_end_col = spec.label_end_col
        full = features.dropna(subset=FEATURE_COLS + [target, label_end_col])
        full = recent_date_window(full, config.model_training_days)
        if full.empty or full[target].nunique() < 2:
            raise RuntimeError(f"Full production training data is invalid for T+{h}")
        X_full = full[FEATURE_COLS]
        y_full = full[target]
        model = xgb.XGBClassifier(**XGBOOST_PARAMS)
        model.fit(X_full, y_full, verbose=False)
        ensemble_models[h] = model
        model_path = config.model_path_for_horizon(h)
        model_path.parent.mkdir(parents=True, exist_ok=True)
        model.save_model(str(model_path))
        log.info("Retrained T+%d on full %d samples", h, len(X_full))

    log.info("=== Generating ensemble scores ===")
    df_all = features.copy()
    df_temp = df_all[FEATURE_COLS].fillna(0)
    for h, model in ensemble_models.items():
        df_all[f"score_{h}d"] = model.predict_proba(df_temp)[:, 1]
    df_all["ensemble_score"] = blend_horizon_scores(
        df_all,
        ENSEMBLE_HORIZONS,
        weights=config.ensemble_horizon_weights,
        mode=config.ensemble_blend_mode,
    )
    df_all["score"] = df_all["ensemble_score"]

    log.info("=== Running all strategies ===")
    from src.strategies import StrategyManager
    sm = StrategyManager(holding_period=HOLDING_PERIOD)
    rankings = sm.run_all(df_all)

    primary_name = config.primary_ranking_strategy
    ranking = rankings.get(primary_name)
    if ranking is None or ranking.empty:
        ranking = rankings.get("_ensemble", rankings.get("outperform", pd.DataFrame()))
    log.info("Primary ranking strategy: %s", primary_name)
    from src.filters.entry import apply_entry_filters
    ranking, filter_report = apply_entry_filters(ranking, df_all, config)
    rankings["_ensemble"] = ranking
    log.info("Entry filter report: %s", filter_report)
    log.info("Using ensemble ranking: %s", list(ranking.head(3)["ticker"]) if not ranking.empty else "empty")
    if ranking.empty:
        latest_market_date = pd.Timestamp(df_all["date"].max()).date()
        report_metrics = all_metrics.get(f"T+{HOLDING_PERIOD}") or all_metrics.get("T+5", {})
        _publish_no_trade(
            storage,
            latest_market_date.isoformat(),
            report_metrics,
            status="no_trade",
            config=config,
        )
        log.warning("No eligible ranking passed entry gates; publishing no trade")
        return
    storage.save_processed(ranking, "ranking.parquet")

    latest_market_date = pd.Timestamp(ranking["date"].max()).date()
    data_lag_days = (today_vn() - latest_market_date).days
    if latest_market_date > today_vn() or data_lag_days > 5:
        raise RuntimeError(
            f"Market data is stale or from the future (latest={latest_market_date}, "
            f"today={today_vn()})"
        )
    if latest_market_date != today_vn():
        log.warning(
            "Using latest closed market session %s for the %s run",
            latest_market_date,
            today_vn(),
        )

    sm.save_signals(rankings, n=N_PICKS, signal_date=latest_market_date.isoformat())
    sm.backfill_strategy_actuals(config=config)

    signal = SignalGenerator().pick_top_n(
        ranking, n=N_PICKS,
        stop_loss=config.stop_loss,
        take_profit=config.take_profit,
        signal_date=latest_market_date.isoformat(),
    )
    storage.save_processed(signal, "signal.parquet")
    save_signals(signal, model_version=MODEL_VERSION)
    log.info("Top %d (ensemble): %s", N_PICKS, list(signal["ticker"]))

    log.info("=== Ceiling context analysis (top picks) ===")
    from src.filters.ceiling_context import report_ceiling_context
    pick_tickers = list(signal["ticker"])
    ctx_report = report_ceiling_context(pick_tickers)
    for ctx in ctx_report:
        log.info("  [%s] ctx=%s drawdown=%.1f%% floors=%d ceil=%d adj=%+.3f",
                 ctx["ticker"], ctx["context_label"],
                 ctx["drawdown_60d"] * 100,
                 ctx["consecutive_floors"], ctx["consecutive_ceilings"],
                 ctx["score_adjustment"])

    log.info("=== Telegram notification ===")
    from src.notification.telegram import send_signal
    send_signal(signal, "ensemble")

    report_metrics = all_metrics.get(f"T+{HOLDING_PERIOD}") or all_metrics.get("T+5", {})
    log.info(
        "Persisting %s T+%d label metrics; executable quality metrics remain included",
        "execution" if target_specs[HOLDING_PERIOD].execution else "close-to-close",
        HOLDING_PERIOD,
    )
    save_pipeline_run(report_metrics, run_key=latest_market_date.isoformat())
    log.info("=== Backfilling actuals (T+%d) ===", HOLDING_PERIOD)
    from src.database import backfill_actuals
    bf_count = backfill_actuals(
        holding_period=HOLDING_PERIOD,
        config=config,
    )
    log.info("Backfilled %d actuals", bf_count)

    log.info("=== Syncing to cloud (Supabase) ===")
    from src.supabase_client import sync_all
    sync_all(config=config)

    log.info("=== Pipeline complete ===")


if __name__ == "__main__":
    run_pipeline()
