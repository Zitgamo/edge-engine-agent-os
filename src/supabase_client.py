from __future__ import annotations

import logging
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import pandas as pd
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
        url = Config.supabase_url or _streamlit_secret("SUPABASE_URL")
        anon = Config.supabase_anon_key or _streamlit_secret("SUPABASE_ANON_KEY")
        if not url or not anon:
            return None
        service_key = Config.supabase_service_key or _streamlit_secret("SUPABASE_SERVICE_KEY")
        return cls(url=url, anon_key=anon, service_key=service_key)


def _streamlit_secret(name: str) -> str | None:
    """Read a Streamlit secret without making Streamlit a CLI requirement."""
    try:
        import streamlit as st
        value = st.secrets.get(name)
        return str(value) if value else None
    except Exception:
        return None


class SupabaseClient:
    def __init__(self, cfg: SupabaseConfig) -> None:
        self.cfg = cfg
        self.base = f"{cfg.url.rstrip('/')}/rest/v1"
        self._remote_column_cache: dict[tuple[str, str], bool] = {}

    def _headers(self, use_service: bool = False) -> dict[str, str]:
        key = self.cfg.service_key if use_service else self.cfg.anon_key
        if use_service and not key:
            raise RuntimeError("SUPABASE_SERVICE_KEY is required for cloud sync")
        return {
            "apikey": self.cfg.anon_key,
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

    def _remote_column_available(self, table: str, column: str) -> bool:
        """Check and cache whether an optional migration column exists."""
        key = (table, column)
        if key in self._remote_column_cache:
            return self._remote_column_cache[key]
        try:
            resp = requests.get(
                f"{self.base}/{table}",
                headers=self._headers(use_service=True),
                params={"select": column, "limit": "1"},
                timeout=15,
            )
        except requests.RequestException:
            available = False
        else:
            available = resp.status_code == 200
        self._remote_column_cache[key] = available
        if not available:
            log.warning(
                "Supabase %s.%s is unavailable; using legacy schema until supabase_setup.sql is applied",
                table,
                column,
            )
        return available

    def _upsert(self, table: str, rows: list[dict], on_conflict: str | None = None) -> int:
        if not rows:
            return 0
        params = {}
        if on_conflict:
            params["on_conflict"] = on_conflict
        try:
            resp = requests.post(
                f"{self.base}/{table}",
                headers={**self._headers(use_service=True), "Prefer": "resolution=merge-duplicates"},
                params=params,
                json=rows,
                timeout=15,
            )
        except requests.RequestException as exc:
            raise RuntimeError(f"Supabase upsert {table} request failed") from exc
        if resp.status_code in (200, 201, 204):
            inserted = len(resp.json()) if resp.content else len(rows)
            log.info("Synced %d rows to supabase.%s", inserted, table)
            return inserted
        raise RuntimeError(f"Supabase upsert {table} failed: {resp.status_code} {resp.text[:200]}")

    def clear_publication_for_date(
        self,
        signal_date: str,
        *,
        preserve_strategy_names: set[str] | None = None,
    ) -> dict[str, int]:
        """Delete production publication rows while preserving paper cohorts."""
        date_key = str(signal_date)[:10]
        if preserve_strategy_names is None:
            preserve_strategy_names = Config().paper_strategy_names()
        counts: dict[str, int] = {}
        for table in ("signals", "actuals", "strategy_performance"):
            params: dict[str, str] = {"signal_date": f"eq.{date_key}"}
            if table == "strategy_performance" and preserve_strategy_names:
                names = ",".join(sorted(str(name) for name in preserve_strategy_names))
                params["strategy_name"] = f"not.in.({names})"
            try:
                resp = requests.delete(
                    f"{self.base}/{table}",
                    headers=self._headers(use_service=True),
                    params=params,
                    timeout=15,
                )
            except requests.RequestException as exc:
                raise RuntimeError(
                    f"Supabase no-trade cleanup {table} request failed"
                ) from exc
            if resp.status_code not in (200, 204):
                raise RuntimeError(
                    f"Supabase no-trade cleanup {table} failed: "
                    f"{resp.status_code} {resp.text[:200]}"
                )
            counts[table] = len(resp.json()) if resp.content else 0
        log.info("Cleared cloud publication rows for %s: %s", date_key, counts)
        return counts

    def _delete_stale_signal_rows(
        self,
        signal_date: str,
        tickers: list[str],
        actual_tickers: list[str] | None = None,
    ) -> None:
        """Remove replaced signal rows without touching strategy history.

        ``strategy_performance`` contains independent picks for every
        strategy, so the main ensemble ticker set is not a valid cleanup key
        for that table.  Actuals are tied to the main signal publication.  A
        force rerun removes those local actuals first; passing the local
        actual ticker set here then removes the matching cloud outcomes too,
        including a stale outcome for a ticker that remains in the rerun.
        """
        signal_filter = f"not.in.({','.join(tickers)})"
        self._delete_cloud_rows(
            "signals",
            {"signal_date": f"eq.{signal_date}", "ticker": signal_filter},
            error_prefix="Supabase cleanup",
        )

        retained_actual_tickers = [
            ticker
            for ticker in (tickers if actual_tickers is None else actual_tickers)
            if ticker in tickers
        ]
        actual_params = {"signal_date": f"eq.{signal_date}"}
        if retained_actual_tickers:
            actual_params["ticker"] = f"not.in.({','.join(retained_actual_tickers)})"
        self._delete_cloud_rows(
            "actuals",
            actual_params,
            error_prefix="Supabase cleanup",
        )

    def _delete_cloud_rows(
        self,
        table: str,
        params: dict[str, str],
        *,
        error_prefix: str,
    ) -> None:
        try:
            resp = requests.delete(
                f"{self.base}/{table}",
                headers=self._headers(use_service=True),
                params=params,
                timeout=15,
            )
        except requests.RequestException as exc:
            raise RuntimeError(f"{error_prefix} {table} request failed") from exc
        if resp.status_code not in (200, 204):
            raise RuntimeError(
                f"{error_prefix} {table} failed: {resp.status_code} {resp.text[:200]}"
            )

    def init_tables(self) -> bool:
        """Verify that the manually-installed Supabase schema is available.

        Supabase's REST API cannot execute arbitrary DDL with a service key.
        The schema must be installed once from ``supabase_setup.sql``.
        """
        for table in ("signals", "actuals", "pipeline_runs", "strategy_performance"):
            try:
                resp = requests.get(
                    f"{self.base}/{table}",
                    headers=self._headers(use_service=True),
                    params={"select": "*", "limit": "1"},
                    timeout=15,
                )
            except requests.RequestException as exc:
                raise RuntimeError("Supabase schema check request failed") from exc
            if resp.status_code != 200:
                log.warning(
                    "Supabase schema check failed for %s: %s %s",
                    table,
                    resp.status_code,
                    resp.text[:200],
                )
                return False
        return True

    def sync_signals(self) -> int:
        from src.database import get_conn, init_db
        init_db()
        conn = get_conn()
        rows = conn.execute(
            "SELECT signal_date, ticker, rank, score, ensemble_score, stop_loss, take_profit, model_version, created_at "
            "FROM signals ORDER BY id"
        ).fetchall()
        actual_rows = conn.execute(
            "SELECT signal_date, ticker FROM actuals"
        ).fetchall()
        conn.close()
        if not rows:
            return 0
        rows_by_date: dict[str, list[str]] = {}
        for row in rows:
            rows_by_date.setdefault(str(row[0]), []).append(str(row[1]))
        actual_tickers_by_date: dict[str, list[str]] = {}
        for row in actual_rows:
            actual_tickers_by_date.setdefault(str(row[0]), []).append(str(row[1]))
        for signal_date, tickers in rows_by_date.items():
            self._delete_stale_signal_rows(
                signal_date,
                tickers,
                actual_tickers=actual_tickers_by_date.get(signal_date, []),
            )
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
        from src.database import get_conn, init_db
        init_db()
        conn = get_conn()
        rows = conn.execute(
            "SELECT signal_date, ticker, actual_excess_return_5d, actual_excess_return_20d, "
            "actual_stock_return, benchmark_return, gross_stock_return, transaction_cost, "
            "actual_outperform, realized_date, entry_date, entry_price, exit_price, "
            "execution_status, updated_at "
            "FROM actuals ORDER BY id"
        ).fetchall()
        conn.close()
        if not rows:
            return 0
        has_execution_schema = self._remote_column_available(
            "actuals", "actual_excess_return_20d"
        )
        optional_columns = {
            column: self._remote_column_available("actuals", column)
            for column in (
                "actual_stock_return",
                "benchmark_return",
                "gross_stock_return",
                "transaction_cost",
            )
        }
        data = []
        for r in rows:
            row = {
                "signal_date": str(r[0]),
                "ticker": r[1],
                "actual_excess_return_5d": float(r[2]) if r[2] is not None else None,
                "actual_outperform": int(r[8]) if r[8] is not None else None,
                "realized_date": str(r[9]),
                "updated_at": str(r[14]) if r[14] else None,
            }
            if has_execution_schema:
                row.update({
                    "actual_excess_return_20d": float(r[3]) if r[3] is not None else None,
                    "entry_date": str(r[10]) if r[10] else None,
                    "entry_price": float(r[11]) if r[11] is not None else None,
                    "exit_price": float(r[12]) if r[12] is not None else None,
                    "execution_status": r[13],
                })
            optional_values = {
                "actual_stock_return": r[4],
                "benchmark_return": r[5],
                "gross_stock_return": r[6],
                "transaction_cost": r[7],
            }
            for column, available in optional_columns.items():
                if available:
                    value = optional_values[column]
                    row[column] = float(value) if value is not None else None
            data.append(row)
        return self._upsert("actuals", data, on_conflict="signal_date,ticker")

    def backfill_remote_actuals(
        self,
        holding_period: int = 20,
        config: Config | None = None,
    ) -> int:
        """Realize old cloud signals using the current run's raw market data.

        GitHub Actions does not restore the ignored SQLite file between runs,
        so historical signals must be read from Supabase before calculating
        their T+20 outcomes.
        """
        signals = self._query_all("signals", {
            "select": "signal_date,ticker,stop_loss,take_profit",
            "order": "signal_date.asc,ticker.asc",
        })
        if not signals:
            return 0
        actuals = self._query_all("actuals", {
            "select": "signal_date,ticker",
            "order": "signal_date.asc,ticker.asc",
        })
        realized_keys = {
            (str(row.get("signal_date", ""))[:10], str(row.get("ticker", "")))
            for row in actuals
        }
        pending = [
            row for row in signals
            if (str(row.get("signal_date", ""))[:10], str(row.get("ticker", "")))
            not in realized_keys
        ]
        if not pending:
            return 0

        from src.actuals import calculate_actuals
        calculated = calculate_actuals(
            pending,
            holding_period=holding_period,
            config=config,
        )
        if calculated.empty:
            return 0
        has_execution_schema = self._remote_column_available(
            "actuals", "actual_excess_return_20d"
        )
        rows = []
        for row in calculated.to_dict("records"):
            payload = {
                "signal_date": str(row["signal_date"]),
                "ticker": row["ticker"],
                "actual_excess_return_5d": float(row["actual_excess_return_5d"]),
                "actual_outperform": int(row["actual_outperform"]),
                "realized_date": str(row["realized_date"]),
            }
            if has_execution_schema:
                payload.update({
                    "actual_excess_return_20d": float(
                        row.get(
                            f"actual_excess_return_{holding_period}d",
                            row.get("actual_excess_return", row["actual_excess_return_5d"]),
                        )
                    ),
                    "entry_date": str(row.get("entry_date")) if row.get("entry_date") else None,
                    "entry_price": float(row["entry_price"]) if row.get("entry_price") is not None else None,
                    "exit_price": float(row["exit_price"]) if row.get("exit_price") is not None else None,
                    "execution_status": row.get("status"),
                })
            for column in (
                "actual_stock_return",
                "benchmark_return",
                "gross_stock_return",
                "transaction_cost",
            ):
                if self._remote_column_available("actuals", column):
                    value = row.get(column)
                    payload[column] = float(value) if value is not None else None
            rows.append(payload)
        return self._upsert("actuals", rows, on_conflict="signal_date,ticker")

    def sync_pipeline_runs(self) -> int:
        from src.database import get_conn, init_db
        init_db()
        conn = get_conn()
        rows = conn.execute(
            "SELECT run_date, accuracy, precision, recall, f1, roc_auc, "
            "execution_evaluation_dates, execution_top3_win_rate, "
            "execution_top3_excess_return, execution_universe_excess_return, "
            "execution_top3_spread, status, run_key "
            "FROM pipeline_runs ORDER BY id DESC LIMIT 1"
        ).fetchall()
        conn.close()
        if not rows:
            return 0
        row = rows[0]
        has_run_key = self._remote_column_available("pipeline_runs", "run_key")
        run_key = str(row[12]) if row[12] else str(row[0])[:10]
        execution_columns = (
            "execution_evaluation_dates",
            "execution_top3_win_rate",
            "execution_top3_excess_return",
            "execution_universe_excess_return",
            "execution_top3_spread",
        )
        execution_available = {
            column: self._remote_column_available("pipeline_runs", column)
            for column in execution_columns
        }
        payload = {
            "run_date": str(row[0]) if row[0] else None,
            "accuracy": float(row[1]) if row[1] is not None else None,
            "precision": float(row[2]) if row[2] is not None else None,
            "recall": float(row[3]) if row[3] is not None else None,
            "f1": float(row[4]) if row[4] is not None else None,
            "roc_auc": float(row[5]) if row[5] is not None else None,
            "status": row[11],
            **({"run_key": run_key} if has_run_key and run_key else {}),
        }
        for offset, column in enumerate(execution_columns, start=6):
            if execution_available[column]:
                value = row[offset]
                payload[column] = float(value) if value is not None else None
        data = [
            payload,
        ]
        return self._upsert(
            "pipeline_runs",
            data,
            on_conflict="run_key" if has_run_key else None,
        )

    def _prune_strategy_rows_for_complete_dates(
        self,
        complete_dates: set[str],
        local_keys: set[tuple[str, str, str]],
        preserve_strategy_names: set[str] | None = None,
    ) -> None:
        """Remove stale strategy keys only for an explicit full snapshot."""
        preserve_strategy_names = set(preserve_strategy_names or ())
        for raw_date in sorted(complete_dates):
            signal_date = str(raw_date)[:10]
            if not signal_date:
                continue
            remote_rows = self._query_all("strategy_performance", {
                "select": "strategy_name,signal_date,ticker",
                "signal_date": f"eq.{signal_date}",
                "order": "strategy_name.asc,ticker.asc",
            })
            if not remote_rows:
                continue
            for row in remote_rows:
                key = (
                    str(row.get("strategy_name", "")),
                    str(row.get("signal_date", ""))[:10],
                    str(row.get("ticker", "")),
                )
                if key[0] in preserve_strategy_names:
                    continue
                if key not in local_keys:
                    self._delete_cloud_rows(
                        "strategy_performance",
                        {
                            "strategy_name": f"eq.{key[0]}",
                            "signal_date": f"eq.{key[1]}",
                            "ticker": f"eq.{key[2]}",
                        },
                        error_prefix="Supabase strategy cleanup",
                    )

    def sync_strategy_performance(
        self,
        complete_dates: set[str] | None = None,
        *,
        preserve_strategy_names: set[str] | None = None,
    ) -> int:
        from src.database import get_conn, init_db
        init_db()
        if preserve_strategy_names is None:
            preserve_strategy_names = Config().paper_strategy_names()
        conn = get_conn()
        rows = conn.execute(
            "SELECT strategy_name, signal_date, ticker, rank, score, actual_excess_return_5d, "
            "actual_excess_return_20d, actual_outperform, realized, stop_loss, take_profit, "
            "atr, market_breadth_20d, strategy_version "
            "FROM strategy_performance ORDER BY id"
        ).fetchall()
        conn.close()
        local_keys = {
            (str(row[0]), str(row[1])[:10], str(row[2]))
            for row in rows
        }
        if not rows:
            if complete_dates:
                self._prune_strategy_rows_for_complete_dates(
                    complete_dates,
                    local_keys,
                    preserve_strategy_names,
                )
            return 0

        has_execution_schema = self._remote_column_available(
            "strategy_performance", "actual_excess_return_20d"
        )
        optional_columns = {
            column: self._remote_column_available("strategy_performance", column)
            for column in (
                "stop_loss",
                "take_profit",
                "atr",
                "market_breadth_20d",
                "strategy_version",
            )
        }
        paper_strategy_names = Config().paper_strategy_names()
        paper_metadata_available = (
            optional_columns["stop_loss"] and optional_columns["take_profit"]
        )
        data = []
        for r in rows:
            is_paper_candidate = str(r[0]) in paper_strategy_names
            if is_paper_candidate and (
                r[9] is None or r[10] is None or not paper_metadata_available
            ):
                log.warning(
                    "Skipping paper candidate %s/%s/%s: apply supabase_setup.sql "
                    "before syncing dynamic stop/take-profit metadata",
                    r[0],
                    r[1],
                    r[2],
                )
                continue
            payload = {
                "strategy_name": r[0],
                "signal_date": str(r[1]),
                "ticker": r[2],
                "rank": int(r[3]) if r[3] is not None else None,
                "score": float(r[4]) if r[4] is not None else None,
                "actual_excess_return_5d": float(r[5]) if r[5] is not None else None,
                "actual_outperform": int(r[7]) if r[7] is not None else None,
                "realized": int(r[8]) if r[8] is not None else 0,
            }
            if has_execution_schema:
                payload["actual_excess_return_20d"] = (
                    float(r[6]) if r[6] is not None else None
                )
            optional_values = {
                "stop_loss": r[9],
                "take_profit": r[10],
                "atr": r[11],
                "market_breadth_20d": r[12],
                "strategy_version": r[13],
            }
            for column, available in optional_columns.items():
                if available:
                    value = optional_values[column]
                    if column == "strategy_version":
                        payload[column] = str(value) if value is not None else None
                    else:
                        payload[column] = float(value) if value is not None else None
            data.append(payload)
        synced = self._upsert(
            "strategy_performance",
            data,
            on_conflict="strategy_name,signal_date,ticker",
        )
        if complete_dates:
            self._prune_strategy_rows_for_complete_dates(
                complete_dates,
                local_keys,
                preserve_strategy_names,
            )
        return synced

    def backfill_remote_strategy_actuals(
        self,
        holding_period: int = 20,
        config: Config | None = None,
        *,
        paper_config: Config | None = None,
        strategy_names: set[str] | None = None,
    ) -> int:
        """Realize pending strategy rows directly in Supabase.

        A scheduled GitHub runner starts with a fresh SQLite database.  The
        local strategy backfill therefore cannot see older cloud rows.  Read
        pending strategy selections from Supabase, calculate each cohort with
        its own OHLC/config source, and upsert the realized result back onto
        every strategy that selected it.
        """
        production_config = config or Config()
        from src.actuals import calculate_actuals, resolve_execution_exits

        if paper_config is None:
            from src.research.paper_candidate import paper_execution_config

            paper_base_config = (
                production_config if isinstance(production_config, Config) else Config()
            )
            paper_config = paper_execution_config(paper_base_config)
        paper_strategy_names = set(paper_config.paper_strategy_names())
        requested_strategy_names = (
            {str(name).strip() for name in strategy_names}
            if strategy_names is not None
            else None
        )

        strategy_rows = self.get_strategy_performance()
        pending_by_cohort: dict[str, list[dict[str, Any]]] = {
            "production": [],
            "paper": [],
        }
        for row in strategy_rows:
            try:
                realized = int(row.get("realized", 0) or 0)
            except (TypeError, ValueError):
                realized = 0
            if realized == 1:
                continue
            strategy_name = str(row.get("strategy_name", "") or "").strip()
            signal_date = str(row.get("signal_date", "") or "")[:10]
            ticker = str(row.get("ticker", "") or "").strip()
            if not strategy_name or not ticker or not signal_date:
                continue
            if (
                requested_strategy_names is not None
                and strategy_name not in requested_strategy_names
            ):
                continue
            if strategy_name in paper_strategy_names and (
                row.get("stop_loss") is None or row.get("take_profit") is None
            ):
                log.warning(
                    "Skipping paper candidate %s/%s/%s without dynamic stop/take-profit metadata",
                    strategy_name,
                    signal_date,
                    ticker,
                )
                continue
            cohort = "paper" if strategy_name in paper_strategy_names else "production"
            pending_by_cohort[cohort].append({
                **row,
                "strategy_name": strategy_name,
                "signal_date": signal_date,
                "ticker": ticker,
                "stop_loss": row.get("stop_loss"),
                "take_profit": row.get("take_profit"),
            })
        if not any(pending_by_cohort.values()):
            return 0

        def outcome_key(row: dict[str, Any]) -> tuple[str, str, float | None, float | None]:
            def numeric(value: Any) -> float | None:
                try:
                    return round(float(value), 12)
                except (TypeError, ValueError):
                    return None

            return (
                str(row.get("signal_date", ""))[:10],
                str(row.get("ticker", "")),
                numeric(row.get("stop_loss")),
                numeric(row.get("take_profit")),
            )

        def date_ticker_key(row: dict[str, Any]) -> tuple[str, str]:
            return (
                str(row.get("signal_date", ""))[:10],
                str(row.get("ticker", "")),
            )

        has_execution_schema = self._remote_column_available(
            "strategy_performance", "actual_excess_return_20d"
        )
        optional_columns = {
            column: self._remote_column_available("strategy_performance", column)
            for column in ("stop_loss", "take_profit")
        }
        updates: list[dict[str, Any]] = []
        for cohort, pending in pending_by_cohort.items():
            if not pending:
                continue
            cohort_config = paper_config if cohort == "paper" else production_config
            inputs = pd.DataFrame(pending)[
                ["signal_date", "ticker", "stop_loss", "take_profit"]
            ].copy()
            inputs[["stop_loss", "take_profit"]] = resolve_execution_exits(
                inputs[["stop_loss", "take_profit"]],
                cohort_config,
            )[["stop_loss", "take_profit"]]
            calculated = calculate_actuals(
                inputs.drop_duplicates(),
                holding_period=holding_period,
                config=cohort_config,
            )
            if calculated.empty:
                continue
            actual_rows = calculated.to_dict("records")
            actual_map = {outcome_key(row): row for row in actual_rows}
            fallback_actual_map = {date_ticker_key(row): row for row in actual_rows}
            for row, input_row in zip(pending, inputs.to_dict("records")):
                actual = actual_map.get(outcome_key(input_row)) or fallback_actual_map.get(
                    date_ticker_key(input_row)
                )
                if actual is None:
                    continue
                excess_5d = actual.get("actual_excess_return_5d")
                if excess_5d is None:
                    excess_5d = actual.get("actual_excess_return")
                outperform = actual.get("actual_outperform")
                if excess_5d is None or outperform is None:
                    continue
                payload = {
                    "strategy_name": row["strategy_name"],
                    "signal_date": row["signal_date"],
                    "ticker": row["ticker"],
                    "rank": int(row["rank"]) if row.get("rank") is not None else None,
                    "score": float(row["score"]) if row.get("score") is not None else None,
                    "actual_excess_return_5d": float(excess_5d),
                    "actual_outperform": int(outperform),
                    "realized": 1,
                }
                for column in ("stop_loss", "take_profit"):
                    if optional_columns[column] and input_row.get(column) is not None:
                        payload[column] = float(input_row[column])
                if has_execution_schema:
                    execution_return = actual.get(f"actual_excess_return_{holding_period}d")
                    if execution_return is None:
                        execution_return = actual.get("actual_excess_return", excess_5d)
                    payload["actual_excess_return_20d"] = (
                        float(execution_return) if execution_return is not None else None
                    )
                updates.append(payload)
        if not updates:
            return 0
        return self._upsert(
            "strategy_performance",
            updates,
            on_conflict="strategy_name,signal_date,ticker",
        )

    def sync_all(
        self,
        config: Config | None = None,
        strategy_publication_dates: set[str] | None = None,
    ) -> dict[str, int]:
        from src.database import init_db
        init_db()
        execution_config = config or Config()
        from src.research.paper_candidate import paper_execution_config

        paper_config = paper_execution_config(execution_config)
        if not self.init_tables():
            raise RuntimeError(
                "Supabase schema is unavailable; run supabase_setup.sql in the Supabase SQL Editor"
            )
        counts = {}
        counts["signals"] = self.sync_signals()
        counts["remote_actuals"] = self.backfill_remote_actuals(config=config)
        counts["actuals"] = self.sync_actuals()
        counts["pipeline_runs"] = self.sync_pipeline_runs()
        if strategy_publication_dates:
            counts["strategy_performance"] = self.sync_strategy_performance(
                complete_dates=strategy_publication_dates,
                preserve_strategy_names=execution_config.paper_strategy_names(),
            )
        else:
            counts["strategy_performance"] = self.sync_strategy_performance()
        counts["remote_strategy_actuals"] = self.backfill_remote_strategy_actuals(
            config=execution_config,
            paper_config=paper_config,
        )
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

    def _query_all(
        self,
        table: str,
        params: dict[str, str] | None = None,
        *,
        page_size: int = 1000,
    ) -> list[dict[str, Any]]:
        """Read every row by paging through PostgREST ``limit``/``offset``."""
        if page_size <= 0:
            raise ValueError("page_size must be positive")

        base_params = dict(params or {})
        requested_limit: int | None = None
        if "limit" in base_params:
            requested_limit = int(base_params.pop("limit"))
            if requested_limit <= 0:
                return []
        offset = int(base_params.pop("offset", "0"))
        rows: list[dict[str, Any]] = []

        while True:
            remaining = (
                page_size
                if requested_limit is None
                else min(page_size, requested_limit - len(rows))
            )
            if remaining <= 0:
                break
            batch = self._query(
                table,
                {
                    **base_params,
                    "limit": str(remaining),
                    "offset": str(offset),
                },
            )
            if not batch:
                break
            rows.extend(batch[:remaining])
            offset += len(batch)
            if len(batch) < remaining:
                break
        return rows[:requested_limit] if requested_limit is not None else rows

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
                s["actual_excess_return_20d"] = a.get("actual_excess_return_20d")
                s["actual_stock_return"] = a.get("actual_stock_return")
                s["benchmark_return"] = a.get("benchmark_return")
                s["gross_stock_return"] = a.get("gross_stock_return")
                s["transaction_cost"] = a.get("transaction_cost")
                s["actual_outperform"] = a.get("actual_outperform")
                s["realized_date"] = a.get("realized_date")
            else:
                s["actual_excess_return_5d"] = None
                s["actual_excess_return_20d"] = None
                s["actual_stock_return"] = None
                s["benchmark_return"] = None
                s["gross_stock_return"] = None
                s["transaction_cost"] = None
                s["actual_outperform"] = None
                s["realized_date"] = None
        return signals

    def get_actuals(self, limit: int = 100) -> list[dict[str, Any]]:
        return self._query("actuals", {
            "select": "*",
            "order": "signal_date.desc",
            "limit": str(limit),
        })

    def get_strategy_performance(self, limit: int | None = None) -> list[dict[str, Any]]:
        """Read persisted strategy outcomes for cloud runners and dashboards.

        The first query uses the canonical executable T+20 column.  The
        fallback keeps older Supabase projects readable until the one-time
        migration has been applied; legacy ``actual_excess_return_5d`` rows
        are already treated as executable outcomes by the local migration.
        When ``limit`` is omitted, all rows are paged from Supabase so older
        history remains available to adaptive weighting.
        """
        params = {
            "select": (
                "strategy_name,signal_date,ticker,rank,score,"
                "stop_loss,take_profit,atr,market_breadth_20d,strategy_version,"
                "actual_excess_return_5d,actual_excess_return_20d,"
                "actual_outperform,realized"
            ),
            "order": "signal_date.asc,strategy_name.asc,ticker.asc",
        }
        if limit is not None:
            params["limit"] = str(limit)
        rows = self._query_all("strategy_performance", params)
        if rows:
            return rows
        return self._query_all(
            "strategy_performance",
            {
                **params,
                "select": (
                    "strategy_name,signal_date,ticker,rank,score,"
                    "stop_loss,take_profit,atr,market_breadth_20d,strategy_version,"
                    "actual_excess_return_5d,actual_outperform,realized"
                ),
            },
        ) or self._query_all(
            "strategy_performance",
            {
                **params,
                "select": (
                    "strategy_name,signal_date,ticker,rank,score,"
                    "actual_excess_return_5d,actual_outperform,realized"
                ),
            },
        )

    def get_pipeline_summary(self) -> list[dict[str, Any]]:
        return self._query("pipeline_runs", {
            "select": "*",
            "order": "run_date.desc",
            "limit": "50",
        })

    def get_performance_summary(self) -> list[dict[str, Any]]:
        actuals = self._query("actuals", {
            "select": "signal_date,ticker,actual_excess_return_5d,actual_excess_return_20d,"
                       "actual_stock_return,benchmark_return,actual_outperform",
            "order": "signal_date.desc",
            "limit": "500",
        })
        if not actuals:
            actuals = self._query("actuals", {
                "select": "signal_date,ticker,actual_excess_return_5d,actual_outperform",
                "order": "signal_date.desc",
                "limit": "500",
            })
        if not actuals:
            return []

        df = pd.DataFrame(actuals)
        legacy = pd.to_numeric(
            df.get("actual_excess_return_5d", pd.Series(index=df.index)), errors="coerce"
        )
        executable = pd.to_numeric(
            df.get("actual_excess_return_20d", pd.Series(index=df.index)), errors="coerce"
        )
        df["execution_excess_return"] = executable.combine_first(legacy)
        df["actual_outperform"] = pd.to_numeric(df["actual_outperform"], errors="coerce")
        df = df.dropna(subset=["execution_excess_return", "actual_outperform"])
        if df.empty:
            return []
        summary = (
            df.groupby("signal_date", as_index=False)
            .agg(
                total_picks=("ticker", "count"),
                wins=("actual_outperform", "sum"),
                avg_excess_return=("execution_excess_return", "mean"),
                total_excess_return=("execution_excess_return", "sum"),
            )
            .sort_values("signal_date", ascending=False)
        )
        summary["win_rate"] = summary["wins"] / summary["total_picks"]
        if "actual_stock_return" in df.columns:
            df["actual_stock_return"] = pd.to_numeric(
                df["actual_stock_return"], errors="coerce"
            )
            absolute = df.dropna(subset=["actual_stock_return"])
            if not absolute.empty:
                absolute_summary = (
                    absolute.groupby("signal_date", as_index=False)
                    .agg(avg_stock_return=("actual_stock_return", "mean"))
                )
                summary = summary.merge(absolute_summary, on="signal_date", how="left")
        return summary.where(pd.notna(summary), None).to_dict(orient="records")


@lru_cache(maxsize=1)
def get_client() -> SupabaseClient | None:
    cfg = SupabaseConfig.from_env()
    if cfg is None:
        return None
    return SupabaseClient(cfg)


def clear_publication_for_date(
    signal_date: str,
    *,
    preserve_strategy_names: set[str] | None = None,
) -> dict[str, int] | None:
    """Clear one no-trade publication when cloud writes are configured."""
    client = get_client()
    if client is None:
        log.info("Supabase not configured — skipping no-trade cleanup")
        return None
    if not client.cfg.service_key:
        log.warning("SUPABASE_SERVICE_KEY is not configured — skipping no-trade cleanup")
        return None
    return client.clear_publication_for_date(
        signal_date,
        preserve_strategy_names=preserve_strategy_names,
    )


def sync_all(
    config: Config | None = None,
    strategy_publication_dates: set[str] | None = None,
) -> dict[str, int] | None:
    client = get_client()
    if client is None:
        log.info("Supabase not configured — skipping cloud sync")
        return None
    if not client.cfg.service_key:
        log.warning("SUPABASE_SERVICE_KEY is not configured — skipping cloud sync")
        return None
    if strategy_publication_dates:
        return client.sync_all(
            config=config,
            strategy_publication_dates=strategy_publication_dates,
        )
    return client.sync_all(config=config)
