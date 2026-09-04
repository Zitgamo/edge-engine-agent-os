from __future__ import annotations

import pandas as pd

from src.research.ticker_exit_optimizer import (
    BASELINE_ATR_MULTIPLE,
    BASELINE_TAKE_PROFIT,
    ExitPolicy,
    PROFILE_SCHEMA_VERSION,
    _date_windows,
    _resolve_universe,
    _select_best_policy,
    _walk_forward_folds,
    apply_exit_profiles,
    build_profile_document,
    has_approved_profiles,
    resolve_exit_parameters,
)


def test_resolver_uses_approved_profile_and_baseline_fallback() -> None:
    document = {
        "schema_version": PROFILE_SCHEMA_VERSION,
        "profiles": {
            "AAA": {
                "approved": True,
                "atr_multiple": 2.5,
                "take_profit": 0.08,
                "confidence": 0.6,
            },
            "BBB": {
                "approved": False,
                "atr_multiple": 1.5,
                "take_profit": 0.12,
                "confidence": 0.9,
            },
        },
    }

    approved = resolve_exit_parameters("AAA", 0.04, document)
    fallback = resolve_exit_parameters("BBB", 0.04, document)
    missing = resolve_exit_parameters("CCC", 0.04, document)

    assert approved["profile_used"] is True
    assert approved["stop_loss"] == -0.10
    assert approved["take_profit"] == 0.08
    assert fallback["profile_used"] is False
    assert fallback["stop_loss"] == -0.08
    assert fallback["take_profit"] == BASELINE_TAKE_PROFIT
    assert missing["atr_multiple"] == BASELINE_ATR_MULTIPLE


def test_apply_exit_profiles_uses_latest_atr_and_explicit_fallback() -> None:
    signals = pd.DataFrame({"ticker": ["AAA", "BBB"], "score": [0.9, 0.8]})
    features = pd.DataFrame({
        "ticker": ["AAA", "AAA", "BBB"],
        "date": pd.to_datetime(["2026-01-01", "2026-01-02", "2026-01-02"]),
        "atr": [4.0, 5.0, 0.0],
        "close": [100.0, 100.0, 100.0],
    })
    document = {
        "schema_version": PROFILE_SCHEMA_VERSION,
        "profiles": {
            "AAA": {
                "approved": True,
                "atr_multiple": 2.5,
                "take_profit": 0.08,
                "confidence": 0.6,
            },
        },
    }

    result = apply_exit_profiles(
        signals,
        features,
        document,
        fallback_stop_loss=-0.03,
        fallback_take_profit=0.10,
    )

    assert result.loc[0, "stop_loss"] == -0.125
    assert result.loc[0, "take_profit"] == 0.08
    assert bool(result.loc[0, "exit_profile_used"]) is True
    assert result.loc[1, "stop_loss"] == -0.03
    assert result.loc[1, "take_profit"] == 0.10
    assert bool(result.loc[1, "exit_profile_used"]) is False


def test_optimizer_selects_candidate_on_train_and_shrinks_document() -> None:
    baseline = ExitPolicy(2.0, 0.10)
    challenger = ExitPolicy(2.5, 0.08)
    rows = []
    for index in range(6):
        date = pd.Timestamp("2023-01-01") + pd.Timedelta(days=index)
        rows.append({
            "ticker": "AAA",
            "date": date,
            "policy": baseline.policy_id,
            "net_return": 0.01,
            "excess_return": 0.005,
        })
        rows.append({
            "ticker": "AAA",
            "date": date,
            "policy": challenger.policy_id,
            "net_return": 0.03,
            "excess_return": 0.025,
        })
    outcomes = pd.DataFrame(rows)

    selected, stats = _select_best_policy(
        outcomes,
        [baseline, challenger],
        minimum_trades=5,
        baseline_id=baseline.policy_id,
    )
    assert selected == challenger
    assert stats["trades"] == 6

    table = pd.DataFrame([{
        "ticker": "AAA",
        "profile_atr_multiple": 2.25,
        "profile_take_profit": 0.09,
        "approved": True,
        "confidence": 0.5,
        "raw_atr_multiple": 2.5,
        "raw_take_profit": 0.08,
        "train_trades": 40,
        "validation_trades": 10,
        "reason": "test",
    }])
    document = build_profile_document(
        table,
        metadata={"status": "ready", "deployment_recommendation": "live"},
    )
    assert has_approved_profiles(document) is True
    assert document["baseline"]["atr_multiple"] == 2.0
    assert document["profiles"]["AAA"]["take_profit"] == 0.09


def test_date_windows_keep_a_locked_holdout_after_validation() -> None:
    dates = pd.Series(pd.date_range("2021-01-01", "2026-08-26", freq="90D"))
    windows = _date_windows(dates, holdout_days=180)
    assert windows["development_end"] < windows["validation_start"]
    assert windows["validation_end"] < windows["holdout_start"]
    assert windows["holdout_end"] == dates.max()


def test_walk_forward_folds_expand_training_and_exclude_locked_holdout() -> None:
    dates = pd.Series(pd.bdate_range("2021-01-01", periods=80))

    folds = _walk_forward_folds(dates, holdout_days=20, n_folds=3)

    assert len(folds) == 3
    assert folds[0]["train_start"] < folds[0]["train_end"]
    assert folds[0]["validation_end"] < folds[1]["validation_start"]
    assert folds[0]["train_end"] < folds[1]["train_end"]
    assert folds[-1]["validation_end"] < dates.max()


def test_universe_resolver_accepts_named_raw_and_explicit_universes(tmp_path) -> None:
    (tmp_path / "AAA_raw.parquet").touch()
    (tmp_path / "VNINDEX_raw.parquet").touch()

    resolved_all = _resolve_universe(tmp_path, "all")
    assert "AAA" in resolved_all
    assert "ACB" in resolved_all
    assert "VNINDEX" not in resolved_all
    assert _resolve_universe(tmp_path, "VHM,vic") == ["VHM", "VIC"]
    assert _resolve_universe(tmp_path, ["aaa", "AAA"]) == ["AAA"]


def test_approved_profiles_are_rejected_when_runtime_baseline_changed() -> None:
    document = {
        "schema_version": PROFILE_SCHEMA_VERSION,
        "baseline": {"atr_multiple": 2.0, "take_profit": 0.10},
        "metadata": {"deployment_recommendation": "live"},
        "profiles": {"AAA": {"approved": True}},
    }

    assert has_approved_profiles(
        document,
        baseline_atr_multiple=2.0,
        baseline_take_profit=0.10,
    ) is True
    assert has_approved_profiles(
        document,
        baseline_atr_multiple=1.5,
        baseline_take_profit=0.10,
    ) is False


def test_approved_profiles_are_rejected_without_live_recommendation() -> None:
    document = {
        "schema_version": PROFILE_SCHEMA_VERSION,
        "baseline": {"atr_multiple": 2.0, "take_profit": 0.10},
        "metadata": {
            "deployment_recommendation": "paper_only_until_more_evidence",
        },
        "profiles": {"AAA": {"approved": True}},
    }

    assert has_approved_profiles(
        document,
        baseline_atr_multiple=2.0,
        baseline_take_profit=0.10,
    ) is False
    assert has_approved_profiles(
        document,
        baseline_atr_multiple=2.0,
        baseline_take_profit=0.10,
        deployment="paper",
    ) is True


def test_approved_profiles_require_explicit_live_recommendation() -> None:
    document = {
        "schema_version": PROFILE_SCHEMA_VERSION,
        "baseline": {"atr_multiple": 2.0, "take_profit": 0.10},
        "metadata": {"deployment_recommendation": "live"},
        "profiles": {"AAA": {"approved": True}},
    }

    assert has_approved_profiles(
        document,
        baseline_atr_multiple=2.0,
        baseline_take_profit=0.10,
    ) is True
