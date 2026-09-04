"""Point-in-time, per-ticker SL/TP research and profile resolution.

The optimizer is intentionally separate from the production model.  It uses
the same cross-sectional RS/momentum selection and the same T+2, gap-aware
execution rules as the paper candidate, then emits a small profile document
that can be consumed at signal time. A profile is only promoted when the same
candidate beats the configured baseline across enough forward folds; otherwise
the resolver returns the baseline for that ticker.
"""

from __future__ import annotations

import json
import logging
from math import ceil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from src.research.candidate_backtest import _select_trades
from src.research.kbs_dataset import (
    OUTCOME_COLUMNS,
    _load_prices,
    _outcomes_for_ticker,
    latest_closed_date,
    research_tickers,
)
from src.data.universe import HOSE_TICKERS, VN30_TICKERS

log = logging.getLogger(__name__)

# v2 marks the rolling walk-forward/consensus contract.  Older single-split
# profiles must not be silently activated after the research methodology
# changes.
PROFILE_SCHEMA_VERSION = "ticker_exit_v2"
BASELINE_ATR_MULTIPLE = 2.0
BASELINE_TAKE_PROFIT = 0.10
DEFAULT_ATR_GRID = (1.5, 1.75, 2.0, 2.25, 2.5)
DEFAULT_TP_GRID = (0.08, 0.10, 0.12)
DEFAULT_WALK_FORWARD_FOLDS = 3


@dataclass(frozen=True)
class ExitPolicy:
    """An ATR-scaled stop and an absolute take-profit target."""

    atr_multiple: float
    take_profit: float

    @property
    def policy_id(self) -> str:
        atr_text = f"{self.atr_multiple:g}".replace(".", "p")
        tp_text = int(round(self.take_profit * 100))
        return f"atr{atr_text}_tp{tp_text}"


def _float(value: object, default: float = float("nan")) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if np.isfinite(result) else default


def _int(value: object, default: int = 0) -> int:
    try:
        result = int(float(value))
    except (TypeError, ValueError):
        return default
    return result


def _valid_policy(policy: ExitPolicy) -> bool:
    return (
        np.isfinite(policy.atr_multiple)
        and 1.0 <= policy.atr_multiple <= 4.0
        and np.isfinite(policy.take_profit)
        and 0.03 <= policy.take_profit <= 0.25
    )


def _normalise_policies(
    atr_grid: Iterable[float],
    tp_grid: Iterable[float],
) -> list[ExitPolicy]:
    policies: list[ExitPolicy] = []
    seen: set[str] = set()
    for atr_multiple in atr_grid:
        for take_profit in tp_grid:
            policy = ExitPolicy(float(atr_multiple), float(take_profit))
            if not _valid_policy(policy):
                raise ValueError(
                    "Exit grid values must satisfy 1 <= ATR multiple <= 4 "
                    "and 3% <= take-profit <= 25%"
                )
            if policy.policy_id not in seen:
                policies.append(policy)
                seen.add(policy.policy_id)
    if not policies:
        raise ValueError("Exit policy grid cannot be empty")
    return policies


def _policy_frame(
    frame: pd.DataFrame,
    *,
    policy_id: str | None = None,
) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    result = frame.copy()
    result["date"] = pd.to_datetime(result["date"], errors="coerce").dt.normalize()
    result = result.dropna(subset=["date", "net_return", "excess_return"])
    if policy_id is not None and "policy" in result.columns:
        result = result[result["policy"] == policy_id]
    return result


def _profit_factor(returns: pd.Series) -> float:
    values = pd.to_numeric(returns, errors="coerce").dropna()
    wins = values[values > 0]
    losses = values[values < 0]
    if losses.empty:
        return float("inf") if not wins.empty else float("nan")
    return float(wins.sum() / abs(losses.sum()))


def _stats(
    frame: pd.DataFrame,
    *,
    policy_id: str | None = None,
) -> dict[str, float | int]:
    """Return stable metrics for one ticker/policy slice."""
    data = _policy_frame(frame, policy_id=policy_id)
    n = len(data)
    if n == 0:
        return {
            "trades": 0,
            "win_rate": float("nan"),
            "avg_net": float("nan"),
            "avg_excess": float("nan"),
            "profit_factor": float("nan"),
            "realized_rr": float("nan"),
            "std_excess": float("nan"),
            "selection_score": float("nan"),
        }

    net = pd.to_numeric(data["net_return"], errors="coerce").dropna()
    excess = pd.to_numeric(data["excess_return"], errors="coerce").dropna()
    wins = net[net > 0]
    losses = net[net < 0]
    std_excess = float(excess.std(ddof=1)) if len(excess) > 1 else 0.0
    avg_excess = float(excess.mean())
    # Penalise noisy/small samples, but keep the objective centred on
    # outperformance rather than absolute stock return.
    selection_score = avg_excess - 0.5 * std_excess / np.sqrt(max(len(excess), 1))
    return {
        "trades": int(len(excess)),
        "win_rate": float((net > 0).mean()),
        "avg_net": float(net.mean()),
        "avg_excess": avg_excess,
        "profit_factor": _profit_factor(net),
        "realized_rr": (
            float(wins.mean() / abs(losses.mean()))
            if not wins.empty and not losses.empty
            else float("nan")
        ),
        "std_excess": std_excess,
        "selection_score": float(selection_score),
    }


def _date_windows(
    dates: pd.Series,
    *,
    holdout_days: int,
    development_end: object | None = None,
) -> dict[str, pd.Timestamp]:
    normalised = pd.to_datetime(dates, errors="coerce").dropna().dt.normalize()
    if normalised.empty:
        raise ValueError("Cannot define exit windows without signal dates")
    if holdout_days <= 0:
        raise ValueError("holdout_days must be positive")

    first = normalised.min()
    last = normalised.max()
    holdout_start = last - pd.Timedelta(days=holdout_days)
    if development_end is None:
        proposed = pd.Timestamp("2023-12-31")
        if proposed <= first or proposed >= holdout_start:
            available_days = max((holdout_start - first).days, 2)
            proposed = first + pd.Timedelta(days=int(available_days * 0.45))
        development_end_ts = proposed
    else:
        development_end_ts = pd.Timestamp(development_end).normalize()
    development_end_ts = min(development_end_ts, holdout_start - pd.Timedelta(days=1))
    validation_start = development_end_ts + pd.Timedelta(days=1)
    validation_end = holdout_start - pd.Timedelta(days=1)
    return {
        "development_start": first,
        "development_end": development_end_ts,
        "validation_start": validation_start,
        "validation_end": validation_end,
        "holdout_start": holdout_start,
        "holdout_end": last,
    }


def _walk_forward_folds(
    dates: pd.Series,
    *,
    holdout_days: int,
    n_folds: int = DEFAULT_WALK_FORWARD_FOLDS,
    development_end: object | None = None,
) -> list[dict[str, pd.Timestamp | int]]:
    """Create expanding-train, forward-only validation folds.

    The locked holdout is deliberately excluded from every fold.  Validation
    boundaries use the actual signal dates instead of arbitrary calendar
    intervals, which keeps each fold populated when the market was closed or
    the candidate was blocked for a period.
    """
    if n_folds <= 0:
        raise ValueError("n_folds must be positive")
    windows = _date_windows(
        dates,
        holdout_days=holdout_days,
        development_end=development_end,
    )
    normalised = (
        pd.to_datetime(dates, errors="coerce")
        .dropna()
        .dt.normalize()
        .drop_duplicates()
        .sort_values()
    )
    validation_dates = normalised[
        (normalised >= windows["validation_start"])
        & (normalised <= windows["validation_end"])
    ]
    if len(validation_dates) < n_folds:
        raise ValueError(
            f"Need at least {n_folds} validation dates for walk-forward folds; "
            f"found {len(validation_dates)}"
        )

    folds: list[dict[str, pd.Timestamp | int]] = []
    for fold_number, chunk in enumerate(
        np.array_split(validation_dates.to_numpy(), n_folds),
        start=1,
    ):
        if len(chunk) == 0:
            continue
        validation_start = pd.Timestamp(chunk[0]).normalize()
        validation_end = pd.Timestamp(chunk[-1]).normalize()
        folds.append({
            "fold": fold_number,
            "train_start": windows["development_start"],
            "train_end": validation_start - pd.Timedelta(days=1),
            "validation_start": validation_start,
            "validation_end": validation_end,
        })
    return folds


def _resolve_universe(
    raw_dir: str | Path,
    universe: str | Iterable[str],
) -> list[str]:
    """Resolve a named or explicit ticker universe for research selection."""
    if isinstance(universe, str):
        selector = universe.strip().lower()
        if selector == "vn30":
            values: Iterable[str] = VN30_TICKERS
        elif selector in {"all", "full", "hose", "research"}:
            # Include the declared HOSE universe even when a cache is partial;
            # this lets the scheduled refresh fill missing symbols instead of
            # silently shrinking "all" to whatever was cached last time.
            values = [*HOSE_TICKERS, *research_tickers(raw_dir)]
        elif "," in selector:
            values = selector.split(",")
        else:
            raise ValueError(
                "universe must be 'vn30', 'all', 'full', 'hose', or a comma-separated ticker list"
            )
    else:
        values = universe

    resolved = sorted({str(value).strip().upper() for value in values if str(value).strip()})
    if not resolved:
        raise ValueError("universe cannot be empty")
    return resolved


def _build_grid_outcomes(
    raw_dir: str | Path,
    tickers: Iterable[str],
    policies: Iterable[ExitPolicy],
    *,
    holding_period: int,
    round_trip_cost: float,
    as_of: object | None,
) -> pd.DataFrame:
    """Build outcomes once per ticker/policy using the canonical simulator."""
    root = Path(raw_dir)
    closed_date = latest_closed_date(as_of)
    benchmark = _load_prices(root, "VNINDEX", closed_date)
    if benchmark.empty:
        raise FileNotFoundError(f"Missing usable VNINDEX data in {root}")

    rows: list[dict[str, object]] = []
    for ticker in sorted({str(value) for value in tickers}):
        stock = _load_prices(root, ticker, closed_date)
        if stock.empty:
            log.warning("Skipping exit optimization for %s: price data unavailable", ticker)
            continue
        for policy in policies:
            outcomes = _outcomes_for_ticker(
                ticker,
                stock,
                benchmark,
                atr_multiple=policy.atr_multiple,
                take_profit=policy.take_profit,
                holding_period=holding_period,
                round_trip_cost=round_trip_cost,
            )
            for outcome in outcomes:
                rows.append({**outcome, "policy": policy.policy_id})

    result = pd.DataFrame(rows, columns=[*OUTCOME_COLUMNS, "policy"])
    if not result.empty:
        result["date"] = pd.to_datetime(result["date"], errors="coerce").dt.normalize()
        result["ticker"] = result["ticker"].astype(str).str.upper()
    return result


def _select_best_policy(
    ticker_outcomes: pd.DataFrame,
    policies: list[ExitPolicy],
    *,
    minimum_trades: int,
    baseline_id: str,
    baseline: ExitPolicy | None = None,
) -> tuple[ExitPolicy, dict[str, float | int]]:
    candidates: list[tuple[ExitPolicy, dict[str, float | int]]] = []
    for policy in policies:
        stats = _stats(ticker_outcomes, policy_id=policy.policy_id)
        if int(stats["trades"]) >= minimum_trades:
            candidates.append((policy, stats))

    baseline = baseline or next(policy for policy in policies if policy.policy_id == baseline_id)
    baseline_stats = _stats(ticker_outcomes, policy_id=baseline_id)
    if not candidates:
        return baseline, baseline_stats

    candidates.sort(
        key=lambda item: (
            _float(item[1]["selection_score"], -np.inf),
            -abs(item[0].atr_multiple - baseline.atr_multiple),
            -abs(item[0].take_profit - baseline.take_profit),
        ),
        reverse=True,
    )
    return candidates[0]


def _validation_gate(
    candidate: ExitPolicy,
    baseline: ExitPolicy,
    candidate_stats: dict[str, float | int],
    baseline_stats: dict[str, float | int],
    *,
    minimum_validation_trades: int,
    minimum_validation_uplift: float,
    minimum_validation_profit_factor: float,
) -> tuple[bool, float, str]:
    """Apply the anti-overfit gate to one forward validation fold."""
    candidate_excess = _float(candidate_stats["avg_excess"])
    baseline_excess = _float(baseline_stats["avg_excess"])
    uplift = (
        candidate_excess - baseline_excess
        if np.isfinite(candidate_excess) and np.isfinite(baseline_excess)
        else float("nan")
    )
    candidate_trades = int(candidate_stats["trades"])
    baseline_trades = int(baseline_stats["trades"])
    candidate_pf = _float(candidate_stats["profit_factor"], float("inf"))

    if candidate.policy_id == baseline.policy_id:
        return False, uplift, "baseline selected on training fold"
    if candidate_trades < minimum_validation_trades:
        return False, uplift, "candidate validation sample below gate"
    if baseline_trades < minimum_validation_trades:
        return False, uplift, "baseline validation sample below gate"
    if not np.isfinite(uplift) or uplift < minimum_validation_uplift:
        return False, uplift, "validation uplift below gate"
    if not np.isfinite(candidate_excess) or candidate_excess <= 0:
        return False, uplift, "candidate validation excess is not positive"
    if candidate_pf < minimum_validation_profit_factor:
        return False, uplift, "validation profit-factor gate failed"
    return True, uplift, "validation gate passed"


def _period_slice(
    frame: pd.DataFrame,
    *,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    data = _policy_frame(frame)
    return data[(data["date"] >= start) & (data["date"] <= end)].copy()


def _summary_row(
    name: str,
    frame: pd.DataFrame,
    *,
    period: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    top_n: int,
) -> dict[str, object]:
    data = _period_slice(frame, start=start, end=end)
    stats = _stats(data)
    ordered = data.sort_values([column for column in ("exit_date", "date", "ticker") if column in data])
    net = pd.to_numeric(ordered.get("net_return", pd.Series(dtype=float)), errors="coerce").dropna()
    if net.empty:
        max_drawdown = float("nan")
    else:
        # The candidate holds up to ``top_n`` names for roughly one holding
        # period; scale each event to the same fixed exposure used elsewhere
        # in the research reports instead of pretending trades are serial.
        equity = (1.0 + net / max(top_n * 20, 1)).cumprod()
        max_drawdown = float((equity / equity.cummax() - 1.0).min())
    return {
        "policy": name,
        "period": period,
        "start_date": start.date().isoformat(),
        "end_date": end.date().isoformat(),
        "signal_days": int(data["date"].nunique()) if not data.empty else 0,
        "trade_count": int(stats["trades"]),
        "trade_win_rate": stats["win_rate"],
        "avg_net_return": stats["avg_net"],
        "avg_excess_return": stats["avg_excess"],
        "profit_factor": stats["profit_factor"],
        "realized_rr": stats["realized_rr"],
        "max_drawdown": max_drawdown,
    }


def build_profile_document(
    profile_table: pd.DataFrame,
    *,
    baseline: ExitPolicy | None = None,
    metadata: dict[str, object] | None = None,
) -> dict[str, object]:
    """Convert an auditable table into the JSON consumed by signal paths."""
    baseline = baseline or ExitPolicy(BASELINE_ATR_MULTIPLE, BASELINE_TAKE_PROFIT)
    profiles: dict[str, dict[str, object]] = {}
    for row in profile_table.to_dict("records"):
        ticker = str(row.get("ticker", "")).strip().upper()
        if not ticker:
            continue
        profiles[ticker] = {
            "atr_multiple": _float(row.get("profile_atr_multiple"), baseline.atr_multiple),
            "take_profit": _float(row.get("profile_take_profit"), baseline.take_profit),
            "approved": bool(row.get("approved", False)),
            "confidence": _float(row.get("confidence"), 0.0),
            "raw_atr_multiple": _float(row.get("raw_atr_multiple"), baseline.atr_multiple),
            "raw_take_profit": _float(row.get("raw_take_profit"), baseline.take_profit),
            "train_trades": _int(row.get("train_trades", 0)),
            "validation_trades": _int(row.get("validation_trades", 0)),
            "folds_total": _int(row.get("folds_total", 0)),
            "folds_passed": _int(row.get("folds_passed", 0)),
            "consensus_required": _int(row.get("consensus_required", 0)),
            "consensus_count": _int(row.get("consensus_count", 0)),
            "consensus_policy": str(row.get("consensus_policy", "")),
            "reason": str(row.get("reason", "")),
        }
    return {
        "schema_version": PROFILE_SCHEMA_VERSION,
        "baseline": {
            "atr_multiple": baseline.atr_multiple,
            "take_profit": baseline.take_profit,
        },
        "metadata": metadata or {},
        "profiles": profiles,
    }


def save_profile_document(document: dict[str, object], path: str | Path) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(document, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    return output


def load_ticker_exit_profiles(path: str | Path) -> dict[str, object] | None:
    """Load a profile document; malformed or missing files fail closed."""
    profile_path = Path(path)
    if not profile_path.exists():
        return None
    try:
        document = json.loads(profile_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        log.warning("Cannot load ticker exit profiles from %s: %s", profile_path, exc)
        return None
    if not isinstance(document, dict) or document.get("schema_version") != PROFILE_SCHEMA_VERSION:
        log.warning("Ignoring unsupported ticker exit profile document: %s", profile_path)
        return None
    if not isinstance(document.get("profiles"), dict):
        log.warning("Ignoring ticker exit profile document without profiles: %s", profile_path)
        return None
    return document


def has_approved_profiles(
    document: dict[str, object] | None,
    *,
    baseline_atr_multiple: float | None = None,
    baseline_take_profit: float | None = None,
    deployment: str = "live",
) -> bool:
    if not document:
        return False
    metadata = document.get("metadata")
    allowed_recommendations = {
        "live",
    } if deployment == "live" else {
        "live",
        "paper_only_until_more_evidence",
    } if deployment == "paper" else set()
    if (
        not isinstance(metadata, dict)
        or metadata.get("deployment_recommendation") not in allowed_recommendations
    ):
        log.warning(
            "Ignoring ticker exit profiles for %s deployment without an allowed recommendation",
            deployment,
        )
        return False
    if baseline_atr_multiple is not None or baseline_take_profit is not None:
        configured = document.get("baseline")
        if not isinstance(configured, dict):
            return False
        configured_atr = _float(configured.get("atr_multiple"))
        configured_tp = _float(configured.get("take_profit"))
        baseline_mismatch = (
            baseline_atr_multiple is not None
            and not np.isclose(configured_atr, float(baseline_atr_multiple))
        ) or (
            baseline_take_profit is not None
            and not np.isclose(configured_tp, float(baseline_take_profit))
        )
        if baseline_mismatch:
            log.warning("Ignoring ticker exit profiles with a mismatched baseline")
            return False
    profiles = document.get("profiles", {})
    return any(
        isinstance(profile, dict) and bool(profile.get("approved", False))
        for profile in profiles.values()
    ) if isinstance(profiles, dict) else False


def resolve_exit_parameters(
    ticker: str,
    atr_pct: float,
    document: dict[str, object] | None,
    *,
    baseline_atr_multiple: float = BASELINE_ATR_MULTIPLE,
    baseline_take_profit: float = BASELINE_TAKE_PROFIT,
) -> dict[str, object]:
    """Resolve one ticker's stop/target, falling back on every invalid input."""
    baseline = ExitPolicy(float(baseline_atr_multiple), float(baseline_take_profit))
    if not _valid_policy(baseline):
        raise ValueError("Invalid baseline exit policy")
    profile: dict[str, object] | None = None
    profiles = document.get("profiles") if isinstance(document, dict) else None
    if isinstance(profiles, dict):
        candidate = profiles.get(str(ticker).strip().upper())
        if isinstance(candidate, dict) and bool(candidate.get("approved", False)):
            profile = candidate

    policy = baseline
    used_profile = False
    if profile is not None:
        candidate_policy = ExitPolicy(
            _float(profile.get("atr_multiple")),
            _float(profile.get("take_profit")),
        )
        if _valid_policy(candidate_policy):
            policy = candidate_policy
            used_profile = True

    atr_fraction = _float(atr_pct)
    stop_loss = (
        -policy.atr_multiple * atr_fraction
        if np.isfinite(atr_fraction) and atr_fraction > 0
        else float("nan")
    )
    return {
        "ticker": str(ticker).strip().upper(),
        "atr_multiple": policy.atr_multiple,
        "take_profit": policy.take_profit,
        "stop_loss": stop_loss,
        "profile_used": used_profile,
        "confidence": _float(profile.get("confidence"), 0.0) if profile else 0.0,
    }


def apply_exit_profiles(
    signals: pd.DataFrame,
    features: pd.DataFrame,
    document: dict[str, object] | None,
    *,
    fallback_stop_loss: float | None = None,
    fallback_take_profit: float | None = None,
    baseline_atr_multiple: float = BASELINE_ATR_MULTIPLE,
    baseline_take_profit: float = BASELINE_TAKE_PROFIT,
) -> pd.DataFrame:
    """Apply profiles to a signal frame using point-in-time ATR/close values."""
    result = signals.copy()
    if result.empty:
        return result
    required = {"ticker", "atr", "close"}
    if features.empty or not required.issubset(features.columns):
        if fallback_stop_loss is not None:
            result["stop_loss"] = float(fallback_stop_loss)
        if fallback_take_profit is not None:
            result["take_profit"] = float(fallback_take_profit)
        result["exit_profile_used"] = False
        return result

    latest = features.copy()
    if "date" in latest.columns:
        latest["date"] = pd.to_datetime(latest["date"], errors="coerce").dt.normalize()
        latest = latest.sort_values("date").drop_duplicates("ticker", keep="last")
    latest["ticker"] = latest["ticker"].astype(str).str.upper()
    atr_pct = pd.to_numeric(latest["atr"], errors="coerce") / pd.to_numeric(
        latest["close"], errors="coerce"
    )
    atr_map = dict(zip(latest["ticker"], atr_pct))

    resolved = [
        resolve_exit_parameters(
            str(ticker),
            _float(atr_map.get(str(ticker).strip().upper())),
            document,
            baseline_atr_multiple=baseline_atr_multiple,
            baseline_take_profit=baseline_take_profit,
        )
        for ticker in result["ticker"]
    ]
    resolved_frame = pd.DataFrame(resolved, index=result.index)
    valid_stop = pd.to_numeric(resolved_frame["stop_loss"], errors="coerce").notna()
    if fallback_stop_loss is not None:
        result["stop_loss"] = pd.to_numeric(
            resolved_frame["stop_loss"], errors="coerce"
        ).where(valid_stop, float(fallback_stop_loss))
    else:
        result["stop_loss"] = resolved_frame["stop_loss"]
    if fallback_take_profit is not None:
        result["take_profit"] = pd.to_numeric(
            resolved_frame["take_profit"], errors="coerce"
        ).fillna(float(fallback_take_profit))
    else:
        result["take_profit"] = resolved_frame["take_profit"]
    result["exit_profile_used"] = resolved_frame["profile_used"].astype(bool)
    result["exit_profile_confidence"] = resolved_frame["confidence"]
    return result



def run_ticker_exit_research(
    *,
    research_dir: str | Path = "data/research_kbs_5y",
    min_breadth: float = 0.60,
    top_n: int = 3,
    holdout_days: int = 180,
    holding_period: int = 20,
    round_trip_cost: float = 0.003,
    minimum_train_trades: int = 20,
    minimum_validation_trades: int = 8,
    shrinkage_trades: int = 40,
    minimum_validation_uplift: float = 0.0025,
    minimum_validation_profit_factor: float = 1.15,
    atr_grid: Iterable[float] = DEFAULT_ATR_GRID,
    tp_grid: Iterable[float] = DEFAULT_TP_GRID,
    baseline_atr_multiple: float = BASELINE_ATR_MULTIPLE,
    baseline_take_profit: float = BASELINE_TAKE_PROFIT,
    universe: str | Iterable[str] = "vn30",
    walk_forward_folds: int = DEFAULT_WALK_FORWARD_FOLDS,
    minimum_consensus_folds: int | None = None,
    development_end: object | None = None,
    as_of: object | None = None,
    profile_path: str | Path | None = None,
    save: bool = True,
) -> dict[str, object]:
    """Fit, validate, and locked-holdout-test per-ticker exit profiles.

    A ticker must pass the forward validation gate in the same policy on a
    configurable number of expanding walk-forward folds before it can be
    promoted.  The final locked holdout is never used for selection.
    """
    if top_n <= 0 or holding_period <= 0:
        raise ValueError("top_n and holding_period must be positive")
    if minimum_train_trades <= 0 or minimum_validation_trades <= 0:
        raise ValueError("minimum trade gates must be positive")
    if shrinkage_trades < 0:
        raise ValueError("shrinkage_trades cannot be negative")
    if walk_forward_folds <= 0:
        raise ValueError("walk_forward_folds must be positive")

    baseline = ExitPolicy(float(baseline_atr_multiple), float(baseline_take_profit))
    if not _valid_policy(baseline):
        raise ValueError("Invalid baseline exit policy")
    configured_consensus = (
        None
        if minimum_consensus_folds is None
        else int(minimum_consensus_folds)
    )
    if configured_consensus in {None, 0}:
        consensus_required = (
            1
            if walk_forward_folds == 1
            else max(2, int(ceil(walk_forward_folds * 2 / 3)))
        )
    else:
        consensus_required = configured_consensus
        if consensus_required <= 0 or consensus_required > walk_forward_folds:
            raise ValueError(
                "minimum_consensus_folds must be between 1 and walk_forward_folds"
            )

    root = Path(research_dir)
    raw_dir = root / "raw"
    resolved_universe = _resolve_universe(raw_dir, universe)
    universe_label = universe.strip().lower() if isinstance(universe, str) else "custom"
    features_path = root / "processed" / "features_exact.parquet"
    if features_path.exists():
        features = pd.read_parquet(features_path)
    else:
        from src.research.kbs_dataset import build_feature_panel

        features = build_feature_panel(
            raw_dir,
            output_path=features_path,
            as_of=as_of,
        )
    if "date" in features.columns:
        features["date"] = pd.to_datetime(features["date"], errors="coerce").dt.normalize()
        if as_of is not None:
            features = features[features["date"] <= latest_closed_date(as_of)].copy()

    selected = _select_trades(
        features,
        min_breadth=min_breadth,
        top_n=top_n,
        tickers=resolved_universe,
    )
    if selected.empty:
        metadata = {
            "status": "blocked",
            "reason": "no selected trades",
            "universe": universe_label,
            "selection_tickers": resolved_universe,
            "baseline": {
                "atr_multiple": baseline.atr_multiple,
                "take_profit": baseline.take_profit,
            },
            "walk_forward_folds": walk_forward_folds,
        }
        return {
            "profiles": build_profile_document(pd.DataFrame(), baseline=baseline, metadata=metadata),
            "profile_table": pd.DataFrame(),
            "fold_table": pd.DataFrame(),
            "holdout_summary": pd.DataFrame(),
            "metadata": metadata,
        }

    policies = _normalise_policies(atr_grid, tp_grid)
    if baseline.policy_id not in {policy.policy_id for policy in policies}:
        policies.append(baseline)
    policy_by_id = {policy.policy_id: policy for policy in policies}
    windows = _date_windows(
        selected["date"],
        holdout_days=holdout_days,
        development_end=development_end,
    )
    folds = _walk_forward_folds(
        selected["date"],
        holdout_days=holdout_days,
        n_folds=walk_forward_folds,
        development_end=development_end,
    )
    selected["ticker"] = selected["ticker"].astype(str).str.upper()
    selected["date"] = pd.to_datetime(selected["date"], errors="coerce").dt.normalize()
    selected_tickers = sorted(selected["ticker"].dropna().unique())
    grid = _build_grid_outcomes(
        raw_dir,
        selected_tickers,
        policies,
        holding_period=holding_period,
        round_trip_cost=round_trip_cost,
        as_of=as_of,
    )
    if grid.empty:
        metadata = {
            "status": "blocked",
            "reason": "no executable outcomes",
            "universe": universe_label,
            "selection_tickers": resolved_universe,
            "baseline": {
                "atr_multiple": baseline.atr_multiple,
                "take_profit": baseline.take_profit,
            },
            "walk_forward_folds": walk_forward_folds,
        }
        return {
            "profiles": build_profile_document(pd.DataFrame(), baseline=baseline, metadata=metadata),
            "profile_table": pd.DataFrame(),
            "fold_table": pd.DataFrame(),
            "holdout_summary": pd.DataFrame(),
            "metadata": metadata,
        }
    grid["ticker"] = grid["ticker"].astype(str).str.upper()
    grid = selected[["date", "ticker", "rank", "score", "market_breadth_20d"]].merge(
        grid,
        on=["date", "ticker"],
        how="inner",
    )

    profile_rows: list[dict[str, object]] = []
    fold_rows: list[dict[str, object]] = []
    for ticker in selected_tickers:
        ticker_grid = grid[grid["ticker"] == ticker].copy()
        ticker_fold_records: list[dict[str, object]] = []
        ticker_fold_frames: dict[int, tuple[pd.DataFrame, pd.DataFrame]] = {}
        for fold in folds:
            fold_number = int(fold["fold"])
            train = _period_slice(
                ticker_grid,
                start=fold["train_start"],
                end=fold["train_end"],
            )
            validation = _period_slice(
                ticker_grid,
                start=fold["validation_start"],
                end=fold["validation_end"],
            )
            raw_policy, train_stats = _select_best_policy(
                train,
                policies,
                minimum_trades=minimum_train_trades,
                baseline_id=baseline.policy_id,
                baseline=baseline,
            )
            raw_validation = _stats(validation, policy_id=raw_policy.policy_id)
            baseline_validation = _stats(validation, policy_id=baseline.policy_id)
            passed, uplift, gate_reason = _validation_gate(
                raw_policy,
                baseline,
                raw_validation,
                baseline_validation,
                minimum_validation_trades=minimum_validation_trades,
                minimum_validation_uplift=minimum_validation_uplift,
                minimum_validation_profit_factor=minimum_validation_profit_factor,
            )
            record: dict[str, object] = {
                "ticker": ticker,
                "fold": fold_number,
                "train_start": fold["train_start"].date().isoformat(),
                "train_end": fold["train_end"].date().isoformat(),
                "validation_start": fold["validation_start"].date().isoformat(),
                "validation_end": fold["validation_end"].date().isoformat(),
                "selection_policy": raw_policy.policy_id,
                "selection_atr_multiple": raw_policy.atr_multiple,
                "selection_take_profit": raw_policy.take_profit,
                "train_trades": int(train_stats["trades"]),
                "train_avg_excess": train_stats["avg_excess"],
                "validation_trades": int(raw_validation["trades"]),
                "validation_avg_excess": raw_validation["avg_excess"],
                "baseline_validation_avg_excess": baseline_validation["avg_excess"],
                "validation_uplift": uplift,
                "validation_profit_factor": raw_validation["profit_factor"],
                "validation_gate": bool(passed),
                "gate_reason": gate_reason,
            }
            ticker_fold_records.append(record)
            ticker_fold_frames[fold_number] = (train, validation)

        fold_rows.extend(ticker_fold_records)
        passed_records = [
            record for record in ticker_fold_records if bool(record["validation_gate"])
        ]
        passed_counts: dict[str, int] = {}
        passed_uplifts: dict[str, list[float]] = {}
        for record in passed_records:
            policy_id = str(record["selection_policy"])
            passed_counts[policy_id] = passed_counts.get(policy_id, 0) + 1
            uplift = _float(record["validation_uplift"])
            if np.isfinite(uplift):
                passed_uplifts.setdefault(policy_id, []).append(uplift)

        consensus_policy_id: str | None = None
        if passed_counts:
            consensus_policy_id = max(
                passed_counts,
                key=lambda policy_id: (
                    passed_counts[policy_id],
                    float(np.mean(passed_uplifts.get(policy_id, [float("-inf")])))
                    if passed_uplifts.get(policy_id)
                    else float("-inf"),
                    policy_id,
                ),
            )

        selected_counts: dict[str, int] = {}
        last_selected_index: dict[str, int] = {}
        for index, record in enumerate(ticker_fold_records):
            policy_id = str(record["selection_policy"])
            selected_counts[policy_id] = selected_counts.get(policy_id, 0) + 1
            last_selected_index[policy_id] = index
        raw_policy_id = consensus_policy_id
        if raw_policy_id is None or passed_counts.get(raw_policy_id, 0) < consensus_required:
            non_baseline = {
                policy_id: count
                for policy_id, count in selected_counts.items()
                if policy_id != baseline.policy_id
            }
            if non_baseline:
                raw_policy_id = max(
                    non_baseline,
                    key=lambda policy_id: (
                        non_baseline[policy_id],
                        last_selected_index[policy_id],
                        policy_id,
                    ),
                )
            else:
                raw_policy_id = baseline.policy_id
        raw_policy = policy_by_id[raw_policy_id]
        consensus_count = passed_counts.get(raw_policy_id, 0)
        approved = (
            raw_policy.policy_id != baseline.policy_id
            and consensus_count >= consensus_required
        )
        last_fold_number = int(folds[-1]["fold"])
        last_train, last_validation = ticker_fold_frames[last_fold_number]
        train_stats = _stats(last_train, policy_id=raw_policy.policy_id)
        validation_stats = _stats(last_validation, policy_id=raw_policy.policy_id)
        relevant_records = [
            record
            for record in passed_records
            if str(record["selection_policy"]) == raw_policy.policy_id
        ] or [
            record
            for record in ticker_fold_records
            if str(record["selection_policy"]) == raw_policy.policy_id
        ]

        def _mean_record_value(key: str) -> float:
            values = [_float(record[key]) for record in relevant_records]
            values = [value for value in values if np.isfinite(value)]
            return float(np.mean(values)) if values else float("nan")

        confidence = (
            int(train_stats["trades"]) / (int(train_stats["trades"]) + shrinkage_trades)
            if int(train_stats["trades"]) > 0
            else 0.0
        )
        profile_atr = (
            baseline.atr_multiple
            + confidence * (raw_policy.atr_multiple - baseline.atr_multiple)
            if approved
            else baseline.atr_multiple
        )
        profile_tp = (
            baseline.take_profit
            + confidence * (raw_policy.take_profit - baseline.take_profit)
            if approved
            else baseline.take_profit
        )
        if approved:
            reason = (
                f"passed {consensus_count}/{len(folds)} walk-forward gates; "
                f"consensus {consensus_count}/{consensus_required}; "
                "shrunk toward configured baseline"
            )
        elif raw_policy.policy_id == baseline.policy_id:
            reason = "baseline selected across walk-forward training folds"
        elif passed_records:
            reason = (
                f"candidate passed {consensus_count}/{len(folds)} folds; "
                f"consensus requires {consensus_required}"
            )
        else:
            reason = "no walk-forward validation gate passed"
        profile_rows.append({
            "ticker": ticker,
            "train_trades": int(train_stats["trades"]),
            "validation_trades": int(validation_stats["trades"]),
            "raw_policy": raw_policy.policy_id,
            "raw_atr_multiple": raw_policy.atr_multiple,
            "raw_take_profit": raw_policy.take_profit,
            "profile_atr_multiple": float(profile_atr),
            "profile_take_profit": float(profile_tp),
            "confidence": float(confidence if approved else 0.0),
            "approved": bool(approved),
            "train_avg_excess": train_stats["avg_excess"],
            "validation_avg_excess": _mean_record_value("validation_avg_excess"),
            "baseline_validation_avg_excess": _mean_record_value(
                "baseline_validation_avg_excess"
            ),
            "validation_uplift": _mean_record_value("validation_uplift"),
            "validation_profit_factor": _mean_record_value("validation_profit_factor"),
            "folds_total": len(folds),
            "folds_passed": len(passed_records),
            "consensus_required": consensus_required,
            "consensus_count": consensus_count,
            "consensus_policy": consensus_policy_id or baseline.policy_id,
            "last_fold_policy": ticker_fold_records[-1]["selection_policy"],
            "reason": reason,
        })

    profile_table = pd.DataFrame(profile_rows)
    fold_table = pd.DataFrame(fold_rows)

    dynamic_rows: list[dict[str, object]] = []
    closed_date = latest_closed_date(as_of)
    benchmark = _load_prices(raw_dir, "VNINDEX", closed_date)
    for row in profile_table.to_dict("records"):
        ticker = str(row["ticker"])
        stock = _load_prices(raw_dir, ticker, closed_date)
        if stock.empty or benchmark.empty:
            continue
        dynamic_rows.extend(_outcomes_for_ticker(
            ticker,
            stock,
            benchmark,
            atr_multiple=float(row["profile_atr_multiple"]),
            take_profit=float(row["profile_take_profit"]),
            holding_period=holding_period,
            round_trip_cost=round_trip_cost,
        ))
    profile_outcomes = pd.DataFrame(dynamic_rows, columns=OUTCOME_COLUMNS)
    if not profile_outcomes.empty:
        profile_outcomes["ticker"] = profile_outcomes["ticker"].astype(str).str.upper()
        profile_outcomes["date"] = pd.to_datetime(
            profile_outcomes["date"], errors="coerce"
        ).dt.normalize()
    baseline_outcomes = grid[grid["policy"] == baseline.policy_id].copy()
    baseline_holdout = _period_slice(
        baseline_outcomes,
        start=windows["holdout_start"],
        end=windows["holdout_end"],
    )
    profile_trades = selected.merge(
        profile_outcomes[["date", "ticker", "exit_date", "net_return", "excess_return"]],
        on=["date", "ticker"],
        how="inner",
    )
    profile_holdout = _period_slice(
        profile_trades,
        start=windows["holdout_start"],
        end=windows["holdout_end"],
    )
    profile_table["holdout_trades"] = 0
    profile_table["holdout_baseline_avg_excess"] = np.nan
    profile_table["holdout_profile_avg_excess"] = np.nan
    profile_table["holdout_uplift"] = np.nan
    for index, row in profile_table.iterrows():
        ticker = str(row["ticker"])
        baseline_ticker = _period_slice(
            baseline_outcomes[baseline_outcomes["ticker"] == ticker],
            start=windows["holdout_start"],
            end=windows["holdout_end"],
        )
        profile_ticker = _period_slice(
            profile_trades[profile_trades["ticker"] == ticker],
            start=windows["holdout_start"],
            end=windows["holdout_end"],
        )
        baseline_stats = _stats(baseline_ticker)
        profile_stats = _stats(profile_ticker)
        baseline_excess = _float(baseline_stats["avg_excess"])
        profile_excess = _float(profile_stats["avg_excess"])
        profile_table.loc[index, "holdout_trades"] = int(profile_stats["trades"])
        profile_table.loc[index, "holdout_baseline_avg_excess"] = baseline_excess
        profile_table.loc[index, "holdout_profile_avg_excess"] = profile_excess
        profile_table.loc[index, "holdout_uplift"] = (
            profile_excess - baseline_excess
            if np.isfinite(profile_excess) and np.isfinite(baseline_excess)
            else np.nan
        )

    holdout_summary = pd.DataFrame([
        _summary_row(
            f"baseline_{baseline.policy_id}",
            baseline_holdout,
            period="locked_holdout",
            start=windows["holdout_start"],
            end=windows["holdout_end"],
            top_n=top_n,
        ),
        _summary_row(
            "ticker_exit_profiles",
            profile_holdout,
            period="locked_holdout",
            start=windows["holdout_start"],
            end=windows["holdout_end"],
            top_n=top_n,
        ),
    ])
    fold_metadata = [
        {
            key: value.date().isoformat() if isinstance(value, pd.Timestamp) else value
            for key, value in fold.items()
        }
        for fold in folds
    ]
    metadata: dict[str, object] = {
        "status": "ready",
        "schema_version": PROFILE_SCHEMA_VERSION,
        "selection": "cross_sectional_rs_momentum_top_n",
        "universe": universe_label,
        "selection_tickers": resolved_universe,
        "selected_tickers": selected_tickers,
        "min_breadth": min_breadth,
        "top_n": top_n,
        "holding_period": holding_period,
        "round_trip_cost": round_trip_cost,
        "baseline": {
            "atr_multiple": baseline.atr_multiple,
            "take_profit": baseline.take_profit,
        },
        "development_start": windows["development_start"].date().isoformat(),
        "development_end": windows["development_end"].date().isoformat(),
        "validation_start": windows["validation_start"].date().isoformat(),
        "validation_end": windows["validation_end"].date().isoformat(),
        "holdout_start": windows["holdout_start"].date().isoformat(),
        "holdout_end": windows["holdout_end"].date().isoformat(),
        "walk_forward_folds": len(folds),
        "walk_forward_windows": fold_metadata,
        "minimum_consensus_folds": consensus_required,
        "minimum_train_trades": minimum_train_trades,
        "minimum_validation_trades": minimum_validation_trades,
        "shrinkage_trades": shrinkage_trades,
        "minimum_validation_uplift": minimum_validation_uplift,
        "minimum_validation_profit_factor": minimum_validation_profit_factor,
        "selected_trades": int(len(selected)),
        "approved_profiles": int(profile_table["approved"].sum()),
    }
    baseline_holdout_excess = _float(_stats(baseline_holdout)["avg_excess"])
    profile_holdout_excess = _float(_stats(profile_holdout)["avg_excess"])
    metadata["locked_holdout_uplift"] = (
        profile_holdout_excess - baseline_holdout_excess
        if np.isfinite(profile_holdout_excess) and np.isfinite(baseline_holdout_excess)
        else None
    )
    metadata["deployment_recommendation"] = (
        "keep_baseline"
        if metadata["locked_holdout_uplift"] is not None
        and metadata["locked_holdout_uplift"] <= 0
        else "paper_only_until_more_evidence"
    )
    document = build_profile_document(profile_table, baseline=baseline, metadata=metadata)
    result_dir = root / "research_results"
    output_profile_path = (
        Path(profile_path)
        if profile_path is not None
        else result_dir / "ticker_exit_profiles.json"
    )
    if save:
        save_profile_document(document, output_profile_path)
        profile_table.to_csv(result_dir / "ticker_exit_profile_table.csv", index=False)
        fold_table.to_csv(result_dir / "ticker_exit_walk_forward_folds.csv", index=False)
        holdout_summary.to_csv(result_dir / "ticker_exit_holdout_summary.csv", index=False)
        log.info("Saved ticker exit profiles to %s", output_profile_path)
    return {
        "profiles": document,
        "profile_table": profile_table,
        "fold_table": fold_table,
        "holdout_summary": holdout_summary,
        "metadata": metadata,
        "profile_path": str(output_profile_path),
    }
