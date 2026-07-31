from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import requests

from src.config import Config

log = logging.getLogger(__name__)


@dataclass
class SupabaseConfig:
    url: str
    anon_key: str
    service_key: str | None = None

    @classmethod
    def from_env(cls) -> SupabaseConfig | None:
        url = Config.supabase_url
        anon = Config.supabase_anon_key
        if not url or not anon:
            return None
        return cls(url=url, anon_key=anon, service_key=Config.supabase_service_key)


class SupabaseClient:
    def __init__(self, cfg: SupabaseConfig) -> None:
        self.cfg = cfg
        self.base = f"{cfg.url.rstrip('/')}/rest/v1"

    def _headers(self, use_service: bool = False) -> dict[str, str]:
        key = self.cfg.service_key if use_service else self.cfg.anon_key
        return {
            "apikey": self.cfg.anon_key,
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

    def _upsert(self, table: str, rows: list[dict], on_conflict: str | None = None) -> int:
        if not rows:
            return 0
        params = {}
        if on_conflict:
            params["on_conflict"] = on_conflict
        resp = requests.post(
            f"{self.base}/{table}",
            headers={**self._headers(use_service=True), "Prefer": "resolution=merge-duplicates"},
            params=params,
            json=rows,
            timeout=15,
        )
        if resp.status_code in (200, 201):
            inserted = len(resp.json()) if resp.content else len(rows)
            log.info("Synced %d rows to supabase.%s", inserted, table)
            return inserted
        log.warning("Supabase upsert %s failed: %s %s", table, resp.status_code, resp.text[:200])
        return 0

    def _exec_sql(self, sql: str) -> bool:
        """Execute raw SQL via Supabase SQL endpoint (needs service_role key)."""
        if not self.cfg.service_key:
            log.warning("Cannot execute SQL: no service_key configured")
            return False
        try:
            resp = requests.post(
                f"{self.cfg.url.rstrip('/')}/rest/v1/rpc/pg_api",
                headers=self._headers(use_service=True),
                json={"query": sql},
                timeout=15,
            )
            if resp.status_code == 200:
                log.info("SQL executed successfully")
                return True
            log.warning("SQL execution failed: %s %s", resp.status_code, resp.text[:200])
            return False
        except requests.RequestException:
            log.warning("SQL execution not supported on this plan, skipping auto-init")
            return False

    def init_tables(self) -> bool:
        sql = """
        CREATE TABLE IF NOT EXISTS signals (
            id BIGSERIAL PRIMARY KEY,
            signal_date DATE NOT NULL,
            ticker TEXT NOT NULL,
            rank INTEGER NOT NULL,
            score REAL NOT NULL,
            ensemble_score REAL,
            stop_loss REAL,
            take_profit REAL,
            model_version TEXT,
            created_at TIMESTAMPTZ DEFAULT NOW(),
            UNIQUE(signal_date, ticker)
        );
        CREATE TABLE IF NOT EXISTS actuals (
            id BIGSERIAL PRIMARY KEY,
            signal_date DATE NOT NULL,
            ticker TEXT NOT NULL,
            actual_excess_return_5d REAL,
            actual_outperform INTEGER,
            realized_date DATE NOT NULL,
            updated_at TIMESTAMPTZ DEFAULT NOW(),
            UNIQUE(signal_date, ticker)
        );
        CREATE TABLE IF NOT EXISTS pipeline_runs (
            id BIGSERIAL PRIMARY KEY,
            run_date TIMESTAMPTZ DEFAULT NOW(),
            accuracy REAL,
            precision REAL,
            recall REAL,
            f1 REAL,
            roc_auc REAL,
            status TEXT
        );
        CREATE TABLE IF NOT EXISTS strategy_performance (
            id BIGSERIAL PRIMARY KEY,
            strategy_name TEXT NOT NULL,
            signal_date DATE NOT NULL,
            ticker TEXT NOT NULL,
            rank INTEGER,
            score REAL,
            actual_excess_return_5d REAL,
            actual_outperform INTEGER,
            realized INTEGER DEFAULT 0,
            created_at TIMESTAMPTZ DEFAULT NOW(),
            UNIQUE(strategy_name, signal_date, ticker)
        );
        CREATE INDEX IF NOT EXISTS idx_signals_date ON signals(signal_date);
        CREATE INDEX IF NOT EXISTS idx_actuals_signal ON actuals(signal_date, ticker);
        CREATE INDEX IF NOT EXISTS idx_strategy_date ON strategy_performance(signal_date);
        """
        return self._exec_sql(sql)

    def sync_signals(self) -> int:
        from src.database import get_conn
        conn = get_conn()
        rows = conn.execute(
            "SELECT signal_date, ticker, rank, score, ensemble_score, stop_loss, take_profit, model_version, created_at "
            "FROM signals ORDER BY id"
        ).fetchall()
        conn.close()
        if not rows:
            return 0
        data = [
            {
                "signal_date": str(r[0]),
                "ticker": r[1],
                "rank": r[2],
                "score": float(r[3]),
                "ensemble_score": float(r[4]) if r[4] is not None else None,
                "stop_loss": float(r[5]) if r[5] is not None else None,
                "take_profit": float(r[6]) if r[6] is not None else None,
                "model_version": r[7],
                "created_at": str(r[8]) if r[8] else None,
            }
            for r in rows
        ]
        return self._upsert("signals", data, on_conflict="signal_date,ticker")

    def sync_actuals(self) -> int:
        from src.database import get_conn
        conn = get_conn()
        rows = conn.execute(
            "SELECT signal_date, ticker, actual_excess_return_5d, actual_outperform, realized_date, updated_at "
            "FROM actuals ORDER BY id"
        ).fetchall()
        conn.close()
        if not rows:
            return 0
        data = [
            {
                "signal_date": str(r[0]),
                "ticker": r[1],
                "actual_excess_return_5d": float(r[2]) if r[2] is not None else None,
                "actual_outperform": int(r[3]) if r[3] is not None else None,
                "realized_date": str(r[4]),
                "updated_at": str(r[5]) if r[5] else None,
            }
            for r in rows
        ]
        return self._upsert("actuals", data, on_conflict="signal_date,ticker")

    def sync_pipeline_runs(self) -> int:
        from src.database import get_conn
        conn = get_conn()
        rows = conn.execute(
            "SELECT run_date, accuracy, precision, recall, f1, roc_auc, status "
            "FROM pipeline_runs ORDER BY id"
        ).fetchall()
        conn.close()
        if not rows:
            return 0
        data = [
            {
                "run_date": str(r[0]) if r[0] else None,
                "accuracy": float(r[1]) if r[1] is not None else None,
                "precision": float(r[2]) if r[2] is not None else None,
                "recall": float(r[3]) if r[3] is not None else None,
                "f1": float(r[4]) if r[4] is not None else None,
                "roc_auc": float(r[5]) if r[5] is not None else None,
                "status": r[6],
            }
            for r in rows
        ]
        return self._upsert("pipeline_runs", data)

    def sync_strategy_performance(self) -> int:
        from src.database import get_conn
        conn = get_conn()
        rows = conn.execute(
            "SELECT strategy_name, signal_date, ticker, rank, score, actual_excess_return_5d, actual_outperform, realized "
            "FROM strategy_performance ORDER BY id"
        ).fetchall()
        conn.close()
        if not rows:
            return 0
        data = [
            {
                "strategy_name": r[0],
                "signal_date": str(r[1]),
                "ticker": r[2],
                "rank": int(r[3]) if r[3] is not None else None,
                "score": float(r[4]) if r[4] is not None else None,
                "actual_excess_return_5d": float(r[5]) if r[5] is not None else None,
                "actual_outperform": int(r[6]) if r[6] is not None else None,
                "realized": int(r[7]) if r[7] is not None else 0,
            }
            for r in rows
        ]
        return self._upsert("strategy_performance", data, on_conflict="strategy_name,signal_date,ticker")

    def sync_all(self) -> dict[str, int]:
        counts = {}
        counts["signals"] = self.sync_signals()
        counts["actuals"] = self.sync_actuals()
        counts["pipeline_runs"] = self.sync_pipeline_runs()
        counts["strategy_performance"] = self.sync_strategy_performance()
        total = sum(counts.values())
        log.info("Supabase sync complete: %d total rows synced", total)
        for table, n in counts.items():
            if n > 0:
                log.info("  %s: %d", table, n)
        return counts

    # --- Read methods for dashboard ---

    def _query(self, table: str, params: dict[str, str] | None = None) -> list[dict[str, Any]]:
        resp = requests.get(
            f"{self.base}/{table}",
            headers=self._headers(),
            params=params,
            timeout=15,
        )
        if resp.status_code == 200:
            return resp.json()
        log.warning("Supabase query %s failed: %s", table, resp.status_code)
        return []

    def get_signals(self, limit: int = 100) -> list[dict[str, Any]]:
        signals = self._query("signals", {
            "select": "*",
            "order": "signal_date.desc,rank.asc",
            "limit": str(limit),
        })
        actuals = self._query("actuals", {
            "select": "*",
            "order": "signal_date.desc",
            "limit": str(limit),
        })
        actual_map: dict[tuple[str, str], dict[str, Any]] = {}
        for a in actuals:
            key = (str(a.get("signal_date", "")), str(a.get("ticker", "")))
            actual_map[key] = a
        for s in signals:
            key = (str(s.get("signal_date", "")), str(s.get("ticker", "")))
            if key in actual_map:
                a = actual_map[key]
                s["actual_excess_return_5d"] = a.get("actual_excess_return_5d")
                s["actual_outperform"] = a.get("actual_outperform")
                s["realized_date"] = a.get("realized_date")
            else:
                s["actual_excess_return_5d"] = None
                s["actual_outperform"] = None
                s["realized_date"] = None
        return signals

    def get_actuals(self, limit: int = 100) -> list[dict[str, Any]]:
        return self._query("actuals", {
            "select": "*",
            "order": "signal_date.desc",
            "limit": str(limit),
        })

    def get_pipeline_summary(self) -> list[dict[str, Any]]:
        return self._query("pipeline_runs", {
            "select": "*",
            "order": "run_date.desc",
            "limit": "50",
        })

    def get_performance_summary(self) -> list[dict[str, Any]]:
        return self._query("actuals", {
            "select": "signal_date,ticker,actual_excess_return_5d,actual_outperform",
            "order": "signal_date.desc",
            "limit": "500",
        })


def get_client() -> SupabaseClient | None:
    cfg = SupabaseConfig.from_env()
    if cfg is None:
        return None
    return SupabaseClient(cfg)


def sync_all() -> dict[str, int] | None:
    client = get_client()
    if client is None:
        log.info("Supabase not configured — skipping cloud sync")
        return None
    return client.sync_all()
