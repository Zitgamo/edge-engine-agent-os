from __future__ import annotations

import logging

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
from src.features.volatility import ATR
from src.features.volume import VolumeSurge
from src.labels.outperformance import OutperformanceLabel
from src.logging_setup import setup_logging
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
from src.model.trainer import ModelTrainer  # noqa: F401  (kept for backwards compat)
from src.ranking.signal import SignalGenerator
from src.time_utils import today_vn

log = logging.getLogger(__name__)

# Re-export for any external consumers; canonical values live in src.model.schema
ENSEMBLE_HORIZONS = _ENSEMBLE_HORIZONS
TRAIN_SPLIT = 0.8
N_PICKS = _N_PICKS
HOLDING_PERIOD = _HOLDING_PERIOD
STOP_LOSS = -0.03
TAKE_PROFIT = 0.08


def run_pipeline(config: Config | None = None) -> None:
    setup_logging()
    config = config or Config()
    config.ensure_dirs()
    init_db()

    log.info("=== Pipeline started ===")

    collector = OHLCVCollector(config)
    storage = PriceStorage(config)
    validator = DataValidator()

    universe = get_ticker_universe()
    bm = collector.fetch("VNINDEX", days=365)
    benchmark_errors = validator.validate(bm)
    if benchmark_errors or bm.empty:
        raise RuntimeError(f"Benchmark data failed validation: {benchmark_errors}")
    valid_benchmark_sources = {"yahoo", "vnstock_vci"}
    if config.data_source == "yfinance" and collector.last_benchmark_source not in valid_benchmark_sources:
        raise RuntimeError(
            "Refusing to publish signals without the real VNINDEX benchmark "
            f"(source={collector.last_benchmark_source})"
        )
    storage.save_raw(bm, "VNINDEX_raw.parquet")

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

    features = pd.concat(feature_dfs, ignore_index=True)

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
            merge_cols += [f"outperform_{h}d", f"excess_return_{h}d"]
    features = features.merge(labels[merge_cols], on=["date", "ticker"], how="left")

    storage.save_processed(features, "features.parquet")

    log.info("=== Walk-forward train/test split ===")
    trainable = features.dropna(subset=FEATURE_COLS + [f"outperform_{h}d" for h in ENSEMBLE_HORIZONS])
    dates = sorted(trainable["date"].unique())
    if len(dates) < 30:
        raise RuntimeError(f"Only {len(dates)} usable dates remain after feature/label construction")
    cutoff = int(len(dates) * TRAIN_SPLIT)
    train_dates = set(dates[:cutoff])
    test_dates = dates[cutoff:]
    train = trainable[trainable["date"].isin(train_dates)]
    test = trainable[trainable["date"].isin(test_dates)]
    if train.empty or test.empty:
        raise RuntimeError("Walk-forward split produced an empty train or test set")
    log.info("Train: %d dates, %d rows | Test: %d dates, %d rows", len(train_dates), len(train), len(test_dates), len(test))

    log.info("=== Training ensemble models (walk-forward) ===")
    ensemble_models: dict[int, xgb.XGBClassifier] = {}
    for h in ENSEMBLE_HORIZONS:
        target = f"outperform_{h}d"
        X_train = train[FEATURE_COLS]
        y_train = train[target]
        model = xgb.XGBClassifier(**XGBOOST_PARAMS)
        model.fit(X_train, y_train, verbose=False)
        ensemble_models[h] = model
        model.save_model(f"models/xgboost_model_h{h}.json")
        log.info("Trained T+%d on %d samples", h, len(X_train))

    log.info("=== Walk-forward evaluation on test set ===")
    evaluator = ModelEvaluator()
    all_metrics = {}
    for h, model in ensemble_models.items():
        target = f"outperform_{h}d"
        metrics = evaluator.evaluate(model, test, target_col=target)
        all_metrics[f"T+{h}"] = metrics
        log.info("T+%d test metrics: %s", h, metrics)

    log.info("=== Retrain on full data for production ===")
    for h in ENSEMBLE_HORIZONS:
        target = f"outperform_{h}d"
        full = trainable[trainable[target].notna()]
        X_full = full[FEATURE_COLS]
        y_full = full[target]
        model = xgb.XGBClassifier(**XGBOOST_PARAMS)
        model.fit(X_full, y_full, verbose=False)
        ensemble_models[h] = model
        model.save_model(f"models/xgboost_model_h{h}.json")
        log.info("Retrained T+%d on full %d samples", h, len(X_full))

    log.info("=== Generating ensemble scores ===")
    df_all = features.copy()
    df_temp = df_all[FEATURE_COLS].fillna(0)
    for h, model in ensemble_models.items():
        df_all[f"score_{h}d"] = model.predict_proba(df_temp)[:, 1]
    score_cols = [f"score_{h}d" for h in ENSEMBLE_HORIZONS]
    df_all["ensemble_score"] = df_all[score_cols].mean(axis=1)
    df_all["score"] = df_all["ensemble_score"]

    log.info("=== Running all strategies ===")
    from src.strategies import StrategyManager
    sm = StrategyManager(holding_period=HOLDING_PERIOD)
    rankings = sm.run_all(df_all)

    ranking = rankings.get("_ensemble", rankings.get("outperform", pd.DataFrame()))
    log.info("Using ensemble ranking: %s", list(ranking.head(3)["ticker"]) if not ranking.empty else "empty")
    if ranking.empty:
        raise RuntimeError("No ensemble ranking was produced")
    storage.save_processed(ranking, "ranking.parquet")

    latest_market_date = pd.Timestamp(ranking["date"].max()).date()
    if latest_market_date != today_vn():
        log.warning(
            "No fresh market session for %s (latest data=%s); skipping signal publication",
            today_vn(),
            latest_market_date,
        )
        sm.backfill_strategy_actuals()
        report_metrics = all_metrics.get(f"T+{HOLDING_PERIOD}") or all_metrics.get("T+5", {})
        save_pipeline_run(report_metrics)
        from src.database import backfill_actuals
        bf_count = backfill_actuals(holding_period=HOLDING_PERIOD)
        log.info("Backfilled %d actuals", bf_count)
        log.info("=== Syncing to cloud (Supabase) ===")
        from src.supabase_client import sync_all
        sync_all()
        return

    sm.save_signals(rankings, n=N_PICKS, signal_date=latest_market_date.isoformat())
    sm.backfill_strategy_actuals()

    signal = SignalGenerator().pick_top_n(
        ranking, n=N_PICKS,
        stop_loss=STOP_LOSS,
        take_profit=TAKE_PROFIT,
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
    save_pipeline_run(report_metrics)
    log.info("=== Backfilling actuals (T+%d) ===", HOLDING_PERIOD)
    from src.database import backfill_actuals
    bf_count = backfill_actuals(holding_period=HOLDING_PERIOD)
    log.info("Backfilled %d actuals", bf_count)

    log.info("=== Syncing to cloud (Supabase) ===")
    from src.supabase_client import sync_all
    sync_all()

    log.info("=== Pipeline complete ===")


if __name__ == "__main__":
    run_pipeline()
