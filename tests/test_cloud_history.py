from __future__ import annotations

from types import SimpleNamespace

import pandas as pd

from src import database, supabase_client
from src.strategies.manager import StrategyManager
from src.supabase_client import SupabaseClient, SupabaseConfig


def test_cli_signal_history_merges_cloud_only_older_rows(monkeypatch) -> None:
    from src import cli

    local = pd.DataFrame([{
        "signal_date": "2026-08-18",
        "ticker": "AAA",
        "rank": 1,
        "score": 0.9,
        "actual_outperform": None,
    }])
    cloud = [{
        "signal_date": "2026-08-18",
        "ticker": "AAA",
        "rank": 1,
        "score": 0.9,
        "actual_excess_return_20d": 0.04,
        "actual_outperform": 1,
    }, {
        "signal_date": "2026-08-15",
        "ticker": "BBB",
        "rank": 1,
        "score": 0.8,
        "actual_excess_return_20d": -0.02,
        "actual_outperform": 0,
    }]

    monkeypatch.setattr(database, "get_signals", lambda limit: local)

    class FakeClient:
        def get_signals(self, limit):
            return cloud

    monkeypatch.setattr(supabase_client, "get_client", lambda: FakeClient())

    result = cli._load_signal_history(limit=20)

    assert result[["signal_date", "ticker"]].to_dict("records") == [
        {"signal_date": "2026-08-18", "ticker": "AAA"},
        {"signal_date": "2026-08-15", "ticker": "BBB"},
    ]
    assert result.iloc[0]["actual_excess_return_20d"] == 0.04


def test_strategy_history_reader_falls_back_to_legacy_schema(monkeypatch) -> None:
    client = SupabaseClient(SupabaseConfig("https://example.test", "anon", "service"))
    calls: list[str] = []

    def query(table, params):
        calls.append(params["select"])
        if "actual_excess_return_20d" in params["select"]:
            return []
        return [{
            "strategy_name": "outperform",
            "signal_date": "2026-01-01",
            "ticker": "AAA",
            "rank": 1,
            "score": 0.9,
            "actual_excess_return_5d": -0.05,
            "actual_outperform": 0,
            "realized": 1,
        }]

    monkeypatch.setattr(client, "_query", query)

    rows = client.get_strategy_performance()

    assert len(rows) == 1
    assert len(calls) == 2
    assert rows[0]["strategy_name"] == "outperform"


def test_strategy_manager_restores_adaptive_weights_from_cloud(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(database, "DB_PATH", tmp_path / "engine.db")
    remote_rows = [
        {
            "strategy_name": "outperform",
            "signal_date": f"2026-01-{day:02d}",
            "ticker": f"T{day:02d}",
            "actual_excess_return_20d": -0.05,
            "actual_outperform": 0,
            "realized": 1,
        }
        for day in range(1, 11)
    ]

    class FakeClient:
        def get_strategy_performance(self):
            return remote_rows

    monkeypatch.setattr(supabase_client, "get_client", lambda: FakeClient())

    manager = StrategyManager(include_research=False)
    weights = manager.get_strategy_weights(min_signals=10)

    assert weights["outperform"] == 0.1
    assert weights["rs_momentum"] == 1.0


def test_strategy_backfill_uses_supplied_config(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(database, "DB_PATH", tmp_path / "engine.db")
    manager = StrategyManager(include_research=False)
    conn = database.get_conn()
    conn.execute(
        """INSERT INTO strategy_performance
           (strategy_name, signal_date, ticker, rank, score)
           VALUES ('outperform', '2026-01-01', 'AAA', 1, 0.9)"""
    )
    conn.commit()
    conn.close()

    config = object()
    captured = {}

    def calculate(signals, holding_period, config=None):
        captured["config"] = config
        return pd.DataFrame()

    monkeypatch.setattr("src.actuals.calculate_actuals", calculate)

    assert manager.backfill_strategy_actuals(config=config) == 0
    assert captured["config"] is config


def test_strategy_backfill_resolves_missing_production_exits(
    monkeypatch,
    tmp_path,
) -> None:
    monkeypatch.setattr(database, "DB_PATH", tmp_path / "engine.db")
    manager = StrategyManager(include_research=False)
    conn = database.get_conn()
    conn.execute(
        """INSERT INTO strategy_performance
           (strategy_name, signal_date, ticker, rank, score)
           VALUES ('outperform', '2026-01-01', 'AAA', 1, 0.9)"""
    )
    conn.commit()
    conn.close()

    config = SimpleNamespace(stop_loss=-0.03, take_profit=0.08)

    def calculate(signals, holding_period, config=None):
        assert signals.loc[0, "stop_loss"] == -0.03
        assert signals.loc[0, "take_profit"] == 0.08
        return pd.DataFrame([{
            "signal_date": "2026-01-01",
            "ticker": "AAA",
            "stop_loss": -0.03,
            "take_profit": 0.08,
            "actual_excess_return": 0.04,
            "actual_outperform": 1,
        }])

    monkeypatch.setattr("src.actuals.calculate_actuals", calculate)

    assert manager.backfill_strategy_actuals(config=config) == 1
    conn = database.get_conn()
    row = conn.execute(
        "SELECT actual_excess_return_20d, actual_outperform, realized "
        "FROM strategy_performance WHERE ticker = 'AAA'"
    ).fetchone()
    conn.close()
    assert row == (0.04, 1, 1)
