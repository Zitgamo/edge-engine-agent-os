from __future__ import annotations

import logging
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

from src.config import Config
from src.database import get_conn, init_db
from src.execution import barrier_exit
from src.model.blend import blend_horizon_scores
from src.model.schema import FEATURE_COLS, TARGET_COL, XGBOOST_PARAMS
from src.model.splits import (
    purged_recent_train_window,
    purged_train_test_split,
    recent_date_window,
    require_columns,
    require_label_end_columns,
)
from src.model.targets import resolve_target_spec

log = logging.getLogger(__name__)


def _price_metadata(prices: pd.DataFrame) -> tuple[list[str], dict[str, int]]:
    """Cache normalized date lookups on a loaded price frame."""
    date_keys = prices.attrs.get("_date_keys")
    date_to_idx = prices.attrs.get("_date_to_idx")
    if date_keys is None or date_to_idx is None:
        date_keys = prices["date"].dt.strftime("%Y-%m-%d").tolist()
        date_to_idx = {key: idx for idx, key in enumerate(date_keys)}
        prices.attrs["_date_keys"] = date_keys
        prices.attrs["_date_to_idx"] = date_to_idx
    return date_keys, date_to_idx


def _load_backtest_prices(
    ticker: str,
    cache: dict[str, pd.DataFrame],
    data_dir: str | Path = "data/raw",
) -> pd.DataFrame:
    if ticker in cache:
        return cache[ticker]
    path = Path(data_dir) / f"{ticker}_raw.parquet"
    if not path.exists():
        cache[ticker] = pd.DataFrame()
        return cache[ticker]
    prices = pd.read_parquet(path).copy()
    prices["date"] = pd.to_datetime(prices["date"], errors="coerce").dt.normalize()
    prices = prices.dropna(subset=["date", "close"]).drop_duplicates("date").sort_values("date")
    cache[ticker] = prices.reset_index(drop=True)
    _price_metadata(cache[ticker])
    return cache[ticker]


def _holding_period_end_date(
    ticker: str,
    signal_date: str,
    holding_period: int,
    prices_cache: dict[str, pd.DataFrame],
    *,
    data_dir: str | Path = "data/raw",
) -> str | None:
    """Return the full-horizon stock session used for purging a label."""
    stock = _load_backtest_prices(ticker, prices_cache, data_dir)
    if stock.empty:
        return None
    dates, date_to_idx = _price_metadata(stock)
    if signal_date not in date_to_idx:
        return None
    entry_idx = date_to_idx[signal_date] + 1
    future = stock.iloc[entry_idx: entry_idx + holding_period]
    if len(future) < holding_period:
        return None
    return pd.Timestamp(future.iloc[-1]["date"]).date().isoformat()


def _sltp_excess_return(
    ticker: str,
    signal_date: str,
    stop_loss: float,
    take_profit: float,
    holding_period: int,
    prices_cache: dict[str, pd.DataFrame],
    round_trip_cost: float = 0.0,
    *,
    data_dir: str | Path = "data/raw",
) -> float | None:
    """Apply SL/TP to raw OHLC bars, then subtract benchmark return."""
    stock = _load_backtest_prices(ticker, prices_cache, data_dir)
    benchmark = _load_backtest_prices("VNINDEX", prices_cache, data_dir)
    if stock.empty or benchmark.empty:
        return None
    dates, date_to_idx = _price_metadata(stock)
    if signal_date not in date_to_idx:
        return None
    idx = date_to_idx[signal_date]
    entry_idx = idx + 1
    future = stock.iloc[entry_idx: entry_idx + holding_period]
    if future.empty or len(future) < holding_period:
        return None

    entry_row = stock.iloc[entry_idx]
    entry = float(entry_row.get("open", entry_row["close"]))
    if not np.isfinite(entry) or entry <= 0:
        entry = float(entry_row["close"])
    exit_price = float(future.iloc[-1]["close"])
    exit_date = str(future.iloc[-1]["date"])[:10]
    for held, row in enumerate(future.itertuples(index=False), start=1):
        if held <= 2:
            continue
        barrier = barrier_exit(entry, row, stop_loss, take_profit)
        if barrier is not None:
            exit_price = barrier[0]
            exit_date = str(row.date)[:10]
            break
        exit_price = float(row.close)
        exit_date = str(row.date)[:10]

    benchmark_indexed = benchmark.attrs.get("_benchmark_indexed")
    if benchmark_indexed is None:
        benchmark_indexed = benchmark.set_index(benchmark["date"].dt.strftime("%Y-%m-%d"))
        benchmark.attrs["_benchmark_indexed"] = benchmark_indexed
    benchmark_close = benchmark_indexed["close"]
    benchmark_entry_series = (
        benchmark_indexed["open"]
        if "open" in benchmark_indexed.columns
        else benchmark_close
    )
    benchmark_entry = (
        benchmark_entry_series.loc[dates[entry_idx]]
        if dates[entry_idx] in benchmark_entry_series.index
        else np.nan
    )
    if exit_date not in benchmark_close.index or pd.isna(benchmark_entry) or float(benchmark_entry) == 0:
        return None
    stock_return = (exit_price - entry) / entry - round_trip_cost
    benchmark_return = (float(benchmark_close.loc[exit_date]) - float(benchmark_entry)) / float(benchmark_entry)
    return stock_return - benchmark_return


def backtest_sltp(
    sl_levels: list[float] | None = None,
    tp_levels: list[float] | None = None,
    holding_period: int = 20,
    round_trip_cost: float | None = None,
) -> pd.DataFrame:
    sl_levels = sl_levels or [0.0, -0.005, -0.01, -0.015, -0.02, -0.03, -0.04, -0.05]
    tp_levels = tp_levels or [0.0, 0.03, 0.05, 0.08, 0.10, 0.12]
    if round_trip_cost is None:
        round_trip_cost = Config().round_trip_cost

    df = walk_forward_ensemble_predictions()
    if df.empty:
        log.warning("No out-of-sample predictions available for SL/TP backtest")
        return pd.DataFrame()
    df = df.rename(columns={"ensemble_score": "score"}).dropna(subset=["score"]).sort_values("date")
    dates = sorted(df["date"].unique())
    split_idx = int(len(dates) * 0.6)
    validation_dates = dates[:split_idx]
    test_dates = dates[split_idx:]
    if len(validation_dates) < 10 or len(test_dates) < 10:
        log.warning("Not enough OOS dates for separate SL/TP validation and test periods")
        return pd.DataFrame()
    prices_cache: dict[str, pd.DataFrame] = {}

    def evaluate_dates(level_sl: float, level_tp: float, evaluation_dates: list) -> pd.Series:
        daily_rets = []
        for d in evaluation_dates:
            day = df[df["date"] == d]
            picks = day.sort_values("score", ascending=False).head(3)
            if picks.empty:
                continue
            returns = [
                _sltp_excess_return(
                    str(row.ticker),
                    pd.Timestamp(d).date().isoformat(),
                    level_sl,
                    level_tp,
                    holding_period,
                    prices_cache,
                    round_trip_cost,
                )
                for row in picks.itertuples(index=False)
            ]
            returns = [value for value in returns if value is not None]
            if returns:
                daily_rets.append(float(np.mean(returns)))
        return pd.Series(daily_rets)

    def summarize(returns: pd.Series) -> dict[str, float] | None:
        if len(returns) < 10:
            return None
        avg_ret = float(returns.mean())
        volatility = float(returns.std())
        equity = (1 + returns).cumprod()
        drawdown = equity / equity.cummax() - 1
        return {
            "days": len(returns),
            "win_rate": float((returns > 0).mean()),
            "avg_return": avg_ret,
            "cum_return": float(equity.iloc[-1] - 1),
            "sharpe": (avg_ret / volatility * np.sqrt(252)) if volatility > 0 else 0.0,
            "max_dd": float(drawdown.min()),
        }

    results = []
    for sl, tp in product(sl_levels, tp_levels):
        validation = summarize(evaluate_dates(sl, tp, validation_dates))
        test = summarize(evaluate_dates(sl, tp, test_dates))
        if validation is None or test is None:
            continue

        results.append({
            "stop_loss": sl, "take_profit": tp,
            "validation_sharpe": validation["sharpe"],
            "validation_avg_return": validation["avg_return"],
            "validation_days": validation["days"],
            "days": test["days"],
            "win_rate": test["win_rate"],
            "avg_return": test["avg_return"],
            "cum_return": test["cum_return"],
            "sharpe": test["sharpe"],
            "max_dd": test["max_dd"],
        })

    res = pd.DataFrame(results)
    if res.empty:
        return res
    res = res.sort_values(["validation_sharpe", "sharpe"], ascending=False)
    print(f"{'SL':>7} | {'TP':>7} | {'Days':>5} | {'WinRate':>8} | {'AvgRet':>9} | {'CumRet':>10} | {'TestSharpe':>10} | {'MaxDD':>7}")
    print("=" * 75)
    for _, r in res.iterrows():
        print(f"{r['stop_loss']:>+6.0%} | {r['take_profit']:>+6.0%} | {r['days']:>5} | {r['win_rate']:>7.1%} | {r['avg_return']:>+8.2%} | {r['cum_return']:>+9.2%} | {r['sharpe']:>+6.2f} | {r['max_dd']:>6.2%}")
    return res


def train_ensemble_models() -> dict[int, xgb.XGBClassifier]:
    df = pd.read_parquet("data/processed/features.parquet")
    horizons = [1, 5, 10, 20]
    config = Config()
    specs = {
        h: resolve_target_spec(
            df,
            h,
            prefer_execution=config.execution_target_enabled,
        )
        for h in horizons
    }
    require_columns(
        df,
        [spec.label_end_col for spec in specs.values()],
        description="label maturity metadata",
    )
    models = {}
    for h in horizons:
        spec = specs[h]
        target = spec.target_col
        if target not in df.columns:
            log.warning("Skipping horizon %d: column %s not found", h, target)
            continue
        train = df.dropna(
            subset=FEATURE_COLS + [target, spec.label_end_col]
        ).sort_values("date")
        train = recent_date_window(train, config.model_training_days)
        if len(train) < 100:
            continue
        y = train[target]
        X = train[FEATURE_COLS]
        model = xgb.XGBClassifier(**XGBOOST_PARAMS)
        model.fit(X, y, verbose=False)
        models[h] = model
        model_path = config.model_path_for_horizon(h)
        model_path.parent.mkdir(parents=True, exist_ok=True)
        model.save_model(str(model_path))
        log.info("Trained ensemble model T+%d: %d samples", h, len(train))
    return models


def ensemble_predict(models: dict[int, xgb.XGBClassifier]) -> pd.DataFrame:
    df = pd.read_parquet("data/processed/features.parquet")
    scores = {}
    for h, model in models.items():
        df_temp = df[FEATURE_COLS].fillna(0)
        scores[f"score_{h}d"] = model.predict_proba(df_temp)[:, 1]
    scores_df = pd.DataFrame(scores, index=df.index)
    result = df.copy()
    score_cols = [c for c in scores_df.columns if c.startswith("score_")]
    for c in score_cols:
        result[c] = scores_df[c]
    config = Config()
    result["ensemble_score"] = blend_horizon_scores(
        result,
        [int(c.removeprefix("score_").removesuffix("d")) for c in score_cols],
        weights=config.ensemble_horizon_weights,
        mode=config.ensemble_blend_mode,
    )
    result["ensemble_max"] = result[score_cols].max(axis=1)
    return result


def walk_forward_ensemble_predictions(
    min_train_dates: int = 60,
    max_train_dates: int | None = None,
) -> pd.DataFrame:
    """Generate strictly out-of-sample ensemble scores one date at a time."""
    if max_train_dates is None:
        max_train_dates = Config().model_training_days
    df = pd.read_parquet("data/processed/features.parquet").copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.normalize()
    df = df.dropna(subset=["date"]).sort_values("date")
    config = Config()
    specs = {
        h: resolve_target_spec(
            df,
            h,
            prefer_execution=config.execution_target_enabled,
        )
        for h in [1, 5, 10, 20]
    }
    horizons = [
        h for h in [1, 5, 10, 20]
        if specs[h].target_col in df.columns and specs[h].return_col in df.columns
    ]
    require_columns(
        df,
        [specs[h].label_end_col for h in horizons],
        description="label maturity metadata",
    )
    dates = sorted(df["date"].dropna().unique())
    if len(dates) <= min_train_dates or not horizons:
        return pd.DataFrame()

    predictions: list[pd.DataFrame] = []
    for test_date in dates[min_train_dates:]:
        test_date = pd.Timestamp(test_date).normalize()
        test = df[df["date"] == test_date].copy()
        score_cols: list[str] = []
        for horizon in horizons:
            spec = specs[horizon]
            target = spec.target_col
            label_end_col = spec.label_end_col
            train = purged_recent_train_window(
                df,
                test_start=test_date,
                label_end_col=label_end_col,
                max_dates=max_train_dates,
            )
            train_clean = train.dropna(subset=FEATURE_COLS + [target])
            if len(train_clean) < 100 or train_clean[target].nunique() < 2:
                continue
            model = xgb.XGBClassifier(**XGBOOST_PARAMS)
            try:
                model.fit(train_clean[FEATURE_COLS], train_clean[target], verbose=False)
            except ValueError as exc:
                log.warning("Walk-forward model failed for %s T+%d: %s", test_date, horizon, exc)
                continue
            score_col = f"score_{horizon}d"
            test[score_col] = model.predict_proba(test[FEATURE_COLS].fillna(0))[:, 1]
            score_cols.append(score_col)

        if score_cols:
            test["ensemble_score"] = blend_horizon_scores(
                test,
                [int(c.removeprefix("score_").removesuffix("d")) for c in score_cols],
                weights=config.ensemble_horizon_weights,
                mode=config.ensemble_blend_mode,
            )
            test["ensemble_max"] = test[score_cols].max(axis=1)
            predictions.append(test)

    return pd.concat(predictions, ignore_index=True) if predictions else pd.DataFrame()


def backtest_ensemble(
    min_train_dates: int = 60,
    max_train_dates: int | None = None,
) -> pd.DataFrame:
    """Evaluate the ensemble on dates whose models were trained beforehand."""
    df = walk_forward_ensemble_predictions(
        min_train_dates=min_train_dates,
        max_train_dates=max_train_dates,
    )
    if df.empty:
        log.warning("No out-of-sample ensemble predictions available")
        return pd.DataFrame()
    specs = {
        h: resolve_target_spec(
            df,
            h,
            prefer_execution=Config().execution_target_enabled,
        )
        for h in [1, 5, 10, 20]
    }
    models = {
        h for h in [1, 5, 10, 20]
        if f"score_{h}d" in df.columns and specs[h].return_col in df.columns
    }

    results = []
    configured_method = Config().ensemble_blend_mode
    score_methods = [(configured_method, "ensemble_score")]
    if "ensemble_max" in df.columns and configured_method != "max":
        score_methods.append(("max", "ensemble_max"))
    for method, col in score_methods:
        for h in models:
            for n in [1, 3, 5]:
                daily_rets = []
                return_col = specs[h].return_col
                usable = df.dropna(subset=[col, return_col])
                for d in sorted(usable["date"].unique()):
                    day = usable[usable["date"] == d]
                    picks = day.sort_values(col, ascending=False).head(n)
                    if picks.empty:
                        continue
                    daily_rets.append(picks[return_col].mean())
                rets = pd.Series(daily_rets)
                if len(rets) < 10:
                    continue
                results.append({
                    "method": method, "horizon": h, "n": n,
                    "days": len(rets), "win_rate": (rets > 0).mean(),
                    "avg_return": rets.mean(),
                    "cum_return": (1 + rets).prod() - 1,
                    "sharpe": (rets.mean() / rets.std() * np.sqrt(252 / h)) if rets.std() > 0 else 0.0,
                    "max_dd": (((1 + rets).cumprod() / (1 + rets).cumprod().cummax()) - 1).min(),
                })

    res = pd.DataFrame(results)
    if res.empty:
        return res
    res = res.sort_values("sharpe", ascending=False)
    print(f"{'Method':>6} | {'H':>3} | {'N':>3} | {'Days':>5} | {'WinRate':>8} | {'AvgRet':>9} | {'CumRet':>10} | {'Sharpe':>7}")
    print("=" * 75)
    for _, r in res.iterrows():
        print(f"{r['method']:>6} | T+{r['horizon']:<1} | {r['n']:>3} | {r['days']:>5} | {r['win_rate']:>7.1%} | {r['avg_return']:>+8.2%} | {r['cum_return']:>+9.2%} | {r['sharpe']:>+6.2f}")
    return res


def auto_retrain() -> dict[str, float]:
    """Evaluate and optionally deploy a candidate on a purged time split.

    The production model is trained on the full feature file by the daily
    pipeline, so scoring that model on a slice of the same file is not an
    out-of-sample comparison.  Use a chronological train/test split with a
    label-horizon purge and compare the candidate with a no-skill baseline.
    """
    init_db()
    config = Config()
    df = pd.read_parquet("data/processed/features.parquet")
    horizon = int(TARGET_COL.rsplit("_", 1)[-1].removesuffix("d"))
    label_end_col = f"label_end_date_{horizon}d"
    require_label_end_columns(df, [horizon])
    df = df.dropna(subset=FEATURE_COLS + [TARGET_COL, label_end_col]).copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.normalize()
    df = df.dropna(subset=["date"]).sort_values("date")
    dates = sorted(df["date"].unique())

    conn = get_conn()
    last_run = conn.execute("SELECT MAX(run_date) FROM pipeline_runs").fetchone()[0]
    last_date = conn.execute("SELECT MAX(signal_date) FROM signals").fetchone()[0]
    conn.close()

    split = int(len(dates) * 0.8)
    if split <= 0 or split >= len(dates):
        raise ValueError("Not enough dated rows for purged auto-retrain evaluation")

    test_start = pd.Timestamp(dates[split]).normalize()
    train, test = purged_train_test_split(
        df,
        test_start=test_start,
        label_end_col=label_end_col,
    )
    train = recent_date_window(train, config.model_training_days)
    if train.empty or test.empty or train[TARGET_COL].nunique() < 2:
        raise ValueError("Not enough dated rows or target classes for auto-retrain evaluation")

    model_new = xgb.XGBClassifier(**XGBOOST_PARAMS)
    model_new.fit(train[FEATURE_COLS], train[TARGET_COL], verbose=False)
    y_pred = model_new.predict(test[FEATURE_COLS].fillna(0))
    new_acc = float((y_pred == test[TARGET_COL]).mean())

    # This baseline is also evaluated only on the untouched test dates.  The
    # saved production model is deliberately not scored here because its
    # training coverage is unknown and usually includes these dates.
    positive_rate = float(test[TARGET_COL].mean())
    baseline_acc = max(positive_rate, 1.0 - positive_rate)
    improvement = new_acc - baseline_acc

    print(f"Time-split baseline accuracy: {baseline_acc:.2%}")
    print(f"New model accuracy: {new_acc:.2%}")
    print(f"Improvement vs baseline: {improvement:+.2%}")
    print(f"Last pipeline run: {last_run}")
    print(f"Last signal date: {last_date}")

    if improvement > 0.01:
        model_path = config.model_path_for_horizon(horizon)
        model_path.parent.mkdir(parents=True, exist_ok=True)
        model_new.save_model(str(model_path))
        log.info(
            "New T+%d model saved to %s (improved over baseline by %.2f%%)",
            horizon,
            model_path,
            improvement * 100,
        )
    else:
        log.info("Skipping deploy (improvement %.2f%% < 1%% threshold)", improvement * 100)

    return {
        "old_accuracy": baseline_acc,
        "baseline_accuracy": baseline_acc,
        "new_accuracy": new_acc,
        "improvement": improvement,
    }


def backtest_score_validation(min_train_dates: int = 60) -> None:
    """Walk-forward: compare top 3 vs bottom 3 by score using T+5 excess return."""
    df = pd.read_parquet("data/processed/features.parquet")
    require_label_end_columns(df, [5])
    df = df.dropna(
        subset=FEATURE_COLS + ["outperform_5d", "excess_return_5d", "label_end_date_5d"]
    ).sort_values("date")
    dates = sorted(df["date"].unique())

    if len(dates) < min_train_dates + 10:
        print(f"Not enough dates ({len(dates)}) for walk-forward")
        return

    top_results: list[float] = []
    bot_results: list[float] = []
    spread_results: list[float] = []

    for i in range(min_train_dates, len(dates)):
        test_date = pd.Timestamp(dates[i]).normalize()
        train, _ = purged_train_test_split(
            df,
            test_start=test_date,
            label_end_col="label_end_date_5d",
        )
        test = df[df["date"] == test_date]

        if len(train) < 100 or len(test) < 10:
            continue

        model = xgb.XGBClassifier(
            n_estimators=200, max_depth=5, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, random_state=42, eval_metric="logloss",
        )
        train_clean = train.dropna(subset=FEATURE_COLS + [TARGET_COL])
        if len(train_clean) < 100:
            continue

        try:
            model.fit(train_clean[FEATURE_COLS], train_clean[TARGET_COL], verbose=False)
        except Exception:
            continue

        test_X = test[FEATURE_COLS].fillna(0)
        test = test.copy()
        test["score"] = model.predict_proba(test_X)[:, 1]
        test = test.sort_values("score", ascending=False)

        top3 = test.head(3)
        bot3 = test.tail(3)

        top_ret = top3["excess_return_5d"].mean()
        bot_ret = bot3["excess_return_5d"].mean()

        top_results.append(top_ret)
        bot_results.append(bot_ret)
        spread_results.append(top_ret - bot_ret)

    if len(top_results) < 10:
        print(f"Not enough test days ({len(top_results)})")
        return

    top_ret = pd.Series(top_results)
    bot_ret = pd.Series(bot_results)
    spread = pd.Series(spread_results)

    def _max_drawdown(returns: pd.Series) -> float:
        equity = (1 + returns).cumprod()
        return float((equity / equity.cummax() - 1).min())

    print("\n" + "=" * 75)
    print("  WALK-FORWARD BACKTEST: TOP 3 vs BOTTOM 3 (T+5 excess return)")
    print("=" * 75)
    print(f"  Test days:     {len(top_results)}")
    print(f"  Train window:  rolling, min {min_train_dates} days")
    print()
    print(f"  {'Metric':<25} {'Top 3':>12} {'Bottom 3':>12} {'Spread':>12}")
    print(f"  {'-'*61}")
    print(f"  {'Win Rate':<25} {top_ret.gt(0).mean():>11.1%} {bot_ret.gt(0).mean():>11.1%} {spread.gt(0).mean():>11.1%}")
    print(f"  {'Avg Return':<25} {top_ret.mean():>+11.2%} {bot_ret.mean():>+11.2%} {spread.mean():>+11.2%}")
    print(f"  {'Cum Return':<25} {(1+top_ret).prod()-1:>+11.2%} {(1+bot_ret).prod()-1:>+11.2%} {(1+spread).prod()-1:>+11.2%}")
    print(f"  {'Sharpe (ann)':<25} {top_ret.mean()/top_ret.std()*np.sqrt(252):>+11.2f} {bot_ret.mean()/bot_ret.std()*np.sqrt(252):>+11.2f} {spread.mean()/spread.std()*np.sqrt(252):>+11.2f}")
    print(f"  {'Max Drawdown':<25} {_max_drawdown(top_ret):>11.2%} {_max_drawdown(bot_ret):>11.2%} {_max_drawdown(spread):>11.2%}")
    print()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("\n=== STOP-LOSS / TAKE-PROFIT OPTIMIZATION ===\n")
    backtest_sltp()
    print("\n=== MULTI-MODEL ENSEMBLE ===\n")
    backtest_ensemble()
    print("\n=== AUTO-RETRAIN CHECK ===\n")
    auto_retrain()
