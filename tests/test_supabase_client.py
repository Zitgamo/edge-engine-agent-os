from __future__ import annotations

import pandas as pd

from src import actuals
from src.supabase_client import SupabaseClient, SupabaseConfig


def test_cloud_performance_summary_matches_dashboard_contract(monkeypatch) -> None:
    client = SupabaseClient(SupabaseConfig("https://example.test", "anon", "service"))
    monkeypatch.setattr(
        client,
        "_query",
        lambda table, params: [
            {
                "signal_date": "2026-07-01",
                "ticker": "AAA",
                "actual_excess_return_5d": 0.10,
                "actual_outperform": 1,
            },
            {
                "signal_date": "2026-07-01",
                "ticker": "BBB",
                "actual_excess_return_5d": -0.02,
                "actual_outperform": 0,
            },
        ],
    )

    result = client.get_performance_summary()

    assert result == [{
        "signal_date": "2026-07-01",
        "total_picks": 2,
        "wins": 1.0,
        "avg_excess_return": 0.04,
        "total_excess_return": 0.08,
        "win_rate": 0.5,
    }]


def test_remote_backfill_upserts_only_unrealized_signals(monkeypatch) -> None:
    client = SupabaseClient(SupabaseConfig("https://example.test", "anon", "service"))

    def query(table, params):
        if table == "signals":
            return [{"signal_date": "2026-07-01", "ticker": "AAA"}]
        return []

    captured = {}
    monkeypatch.setattr(client, "_query", query)
    monkeypatch.setattr(
        actuals,
        "calculate_actuals",
        lambda pending, holding_period: pd.DataFrame([{
            "signal_date": "2026-07-01",
            "ticker": "AAA",
            "actual_excess_return_5d": 0.05,
            "actual_outperform": 1,
            "realized_date": "2026-07-29",
        }]),
    )
    monkeypatch.setattr(
        client,
        "_upsert",
        lambda table, rows, on_conflict=None: captured.update(
            {"table": table, "rows": rows, "on_conflict": on_conflict}
        ) or len(rows),
    )

    result = client.backfill_remote_actuals()

    assert result == 1
    assert captured["table"] == "actuals"
    assert captured["on_conflict"] == "signal_date,ticker"
    assert captured["rows"][0]["actual_outperform"] == 1
