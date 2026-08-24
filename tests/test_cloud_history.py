from __future__ import annotations

from src import database, supabase_client
from src.strategies.manager import StrategyManager
from src.supabase_client import SupabaseClient, SupabaseConfig


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
