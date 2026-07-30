from __future__ import annotations

import logging
from itertools import product

import numpy as np
import pandas as pd
import xgboost as xgb

from src.config import Config
from src.database import get_conn, init_db
from src.model.inference import ModelInference
from src.model.trainer import FEATURE_COLS, TARGET_COL

log = logging.getLogger(__name__)


def backtest_sltp(sl_levels: list[float] = None, tp_levels: list[float] = None) -> pd.DataFrame:
    sl_levels = sl_levels or [0.0, -0.01, -0.02, -0.03, -0.05]
    tp_levels = tp_levels or [0.0, 0.02, 0.03, 0.05, 0.08, 0.10]

    df = pd.read_parquet("data/processed/features.parquet")
    if "score" not in df.columns:
        inf = ModelInference(Config())
        inf.load()
        df = inf.predict(df)

    df = df.dropna(subset=["score", "excess_return_5d"]).sort_values("date")
    dates = sorted(df["date"].unique())
    split_idx = int(len(dates) * 0.6)
    test_dates = dates[split_idx:]

    results = []
    for sl, tp in product(sl_levels, tp_levels):
        daily_rets = []
        for d in test_dates:
            day = df[df["date"] == d]
            picks = day.sort_values("score", ascending=False).head(3)
            if picks.empty:
                continue
            rets = picks["excess_return_5d"].values
            sl_val = np.where(rets <= sl, sl, rets) if sl < 0 else rets
            tp_val = np.where(sl_val >= tp, tp, sl_val) if tp > 0 else sl_val
            daily_rets.append(tp_val.mean())

        rets = pd.Series(daily_rets)
        if len(rets) < 10:
            continue
        win_rate = (rets > 0).mean()
        avg_ret = rets.mean()
        cum_ret = (1 + rets).prod() - 1
        sharpe = (avg_ret / rets.std() * np.sqrt(252)) if rets.std() > 0 else 0.0
        peak = (1 + rets).cummax()
        dd = ((1 + rets).cumprod() - peak) / peak
        max_dd = dd.min()

        results.append({
            "stop_loss": sl, "take_profit": tp,
            "days": len(rets), "win_rate": win_rate,
            "avg_return": avg_ret, "cum_return": cum_ret,
            "sharpe": sharpe, "max_dd": max_dd,
        })

    res = pd.DataFrame(results).sort_values("sharpe", ascending=False)
    print(f"{'SL':>7} | {'TP':>7} | {'Days':>5} | {'WinRate':>8} | {'AvgRet':>9} | {'CumRet':>10} | {'Sharpe':>7} | {'MaxDD':>7}")
    print("=" * 75)
    for _, r in res.iterrows():
        print(f"{r['stop_loss']:>+6.0%} | {r['take_profit']:>+6.0%} | {r['days']:>5} | {r['win_rate']:>7.1%} | {r['avg_return']:>+8.2%} | {r['cum_return']:>+9.2%} | {r['sharpe']:>+6.2f} | {r['max_dd']:>6.2%}")
    return res


def train_ensemble_models() -> dict[int, xgb.XGBClassifier]:
    df = pd.read_parquet("data/processed/features.parquet")
    horizons = [1, 5, 10, 20]
    models = {}
    for h in horizons:
        target = f"outperform_{h}d"
        if target not in df.columns:
            log.warning("Skipping horizon %d: column %s not found", h, target)
            continue
        train = df.dropna(subset=FEATURE_COLS + [target]).sort_values("date")
        if len(train) < 100:
            continue
        y = train[target]
        X = train[FEATURE_COLS]
        model = xgb.XGBClassifier(n_estimators=200, max_depth=5, learning_rate=0.05,
                                    subsample=0.8, colsample_bytree=0.8, random_state=42, eval_metric="logloss")
        model.fit(X, y, verbose=False)
        models[h] = model
        model.save_model(f"models/xgboost_model_h{h}.json")
        log.info("Trained ensemble model T+%d: %d samples", h, len(train))
    return models


def ensemble_predict(models: dict[int, xgb.XGBClassifier]) -> pd.DataFrame:
    df = pd.read_parquet("data/processed/features.parquet")
    scores = {}
    for h, model in models.items():
        df_temp = df[FEATURE_COLS].fillna(0)
        scores[f"score_{h}d"] = model.predict_proba(df_temp)[:, 1]
    scores_df = pd.DataFrame(scores, index=df.index)
    score_cols = [c for c in scores_df.columns if c.startswith("score_")]
    scores_df["ensemble_score"] = scores_df[score_cols].mean(axis=1)
    scores_df["ensemble_max"] = scores_df[score_cols].max(axis=1)
    result = df.copy()
    for c in score_cols:
        result[c] = scores_df[c]
    result["ensemble_score"] = scores_df["ensemble_score"]
    result["ensemble_max"] = scores_df["ensemble_max"]
    return result


def backtest_ensemble() -> None:
    models = train_ensemble_models()
    df = ensemble_predict(models)
    df = df.dropna(subset=[f"score_{h}d" for h in models] + [f"excess_return_{h}d" for h in models])

    results = []
    for method, col in [("mean", "ensemble_score"), ("max", "ensemble_max")]:
        for h in models:
            for n in [1, 3, 5]:
                daily_rets = []
                for d in sorted(df["date"].unique()):
                    day = df[df["date"] == d]
                    picks = day.sort_values(col, ascending=False).head(n)
                    if picks.empty:
                        continue
                    daily_rets.append(picks[f"excess_return_{h}d"].mean())
                rets = pd.Series(daily_rets)
                if len(rets) < 10:
                    continue
                results.append({
                    "method": method, "horizon": h, "n": n,
                    "days": len(rets), "win_rate": (rets > 0).mean(),
                    "avg_return": rets.mean(),
                    "cum_return": (1 + rets).prod() - 1,
                    "sharpe": (rets.mean() / rets.std() * np.sqrt(252 / h)) if rets.std() > 0 else 0.0,
                })

    res = pd.DataFrame(results).sort_values("sharpe", ascending=False)
    print(f"{'Method':>6} | {'H':>3} | {'N':>3} | {'Days':>5} | {'WinRate':>8} | {'AvgRet':>9} | {'CumRet':>10} | {'Sharpe':>7}")
    print("=" * 75)
    for _, r in res.iterrows():
        print(f"{r['method']:>6} | T+{r['horizon']:<1} | {r['n']:>3} | {r['days']:>5} | {r['win_rate']:>7.1%} | {r['avg_return']:>+8.2%} | {r['cum_return']:>+9.2%} | {r['sharpe']:>+6.2f}")


def auto_retrain() -> dict[str, float]:
    init_db()
    df = pd.read_parquet("data/processed/features.parquet")
    df = df.dropna(subset=FEATURE_COLS + [TARGET_COL]).sort_values("date")
    dates = sorted(df["date"].unique())

    conn = get_conn()
    last_run = conn.execute("SELECT MAX(run_date) FROM pipeline_runs").fetchone()[0]
    last_date = conn.execute("SELECT MAX(signal_date) FROM signals").fetchone()[0]
    conn.close()

    old_model = xgb.XGBClassifier()
    try:
        old_model.load_model(str(Config().model_path))
        X_old = df[FEATURE_COLS].fillna(0)
        y_old = old_model.predict(X_old)
        old_acc = (y_old == df[TARGET_COL]).mean()
    except Exception:
        old_acc = 0.0

    split = int(len(dates) * 0.8)
    train_dates = set(dates[:split])
    test_dates = dates[split:]

    train = df[df["date"].isin(train_dates)]
    test = df[df["date"].isin(test_dates)]

    model_new = xgb.XGBClassifier(n_estimators=200, max_depth=5, learning_rate=0.05,
                                    subsample=0.8, colsample_bytree=0.8, random_state=42, eval_metric="logloss")
    model_new.fit(train[FEATURE_COLS], train[TARGET_COL], verbose=False)

    y_pred = model_new.predict(test[FEATURE_COLS])
    new_acc = (y_pred == test[TARGET_COL]).mean()
    improvement = new_acc - old_acc

    print(f"Old model accuracy: {old_acc:.2%}")
    print(f"New model accuracy: {new_acc:.2%}")
    print(f"Improvement: {improvement:+.2%}")
    print(f"Last pipeline run: {last_run}")
    print(f"Last signal date: {last_date}")

    if improvement > 0.01:
        model_new.save_model(str(Config().model_path))
        log.info("New model saved (improved by %.2f%%)", improvement * 100)
    else:
        log.info("Skipping deploy (improvement %.2f%% < 1%% threshold)", improvement * 100)

    return {"old_accuracy": old_acc, "new_accuracy": new_acc, "improvement": improvement}


def backtest_score_validation(min_train_dates: int = 60) -> None:
    """Walk-forward: compare top 3 vs bottom 3 by score using T+5 excess return."""
    df = pd.read_parquet("data/processed/features.parquet")
    df = df.dropna(subset=FEATURE_COLS + ["excess_return_5d"]).sort_values("date")
    dates = sorted(df["date"].unique())

    if len(dates) < min_train_dates + 10:
        print(f"Not enough dates ({len(dates)}) for walk-forward")
        return

    top_results: list[float] = []
    bot_results: list[float] = []
    spread_results: list[float] = []

    for i in range(min_train_dates, len(dates)):
        train_end = dates[i - 1]
        test_date = dates[i]

        train = df[df["date"] <= train_end]
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
    print(f"  {'Max Drawdown':<25} {top_ret.min():>11.2%} {bot_ret.min():>11.2%} {spread.min():>11.2%}")
    print()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("\n=== STOP-LOSS / TAKE-PROFIT OPTIMIZATION ===\n")
    backtest_sltp()
    print("\n=== MULTI-MODEL ENSEMBLE ===\n")
    backtest_ensemble()
    print("\n=== AUTO-RETRAIN CHECK ===\n")
    auto_retrain()
