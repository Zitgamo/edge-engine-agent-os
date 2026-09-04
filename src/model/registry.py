"""Persistent champion/challenger decisions for production models.

The daily runner is intentionally ephemeral, so model files alone cannot
provide a reliable memory of which candidate was last accepted.  This small
JSON registry keeps the decision history in the repository while leaving
Supabase as the source of truth for signals and realized trades.

The registry is a publication guard, not an optimizer.  A candidate must
first pass the existing execution quality gate, then it is compared with the
current champion.  Material regressions are rejected and the pipeline can
publish an explicit no-trade state instead of silently replacing production
behavior.
"""

from __future__ import annotations

import json
import logging
import math
import re
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

log = logging.getLogger(__name__)

REGISTRY_SCHEMA_VERSION = 1
DEFAULT_MAX_REGRESSION = 0.002
COMPARISON_METRICS = (
    "execution_top3_excess_return",
    "execution_top3_spread",
)


class ModelRegistryError(RuntimeError):
    """Raised when a registry exists but cannot be trusted or written."""


def _normalise_metrics(metrics: Mapping[str, Any] | None) -> dict[str, float]:
    """Keep only finite numeric metrics so registry writes stay JSON-safe."""
    normalised: dict[str, float] = {}
    for key, value in (metrics or {}).items():
        if not isinstance(key, str) or value is None:
            continue
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(numeric):
            normalised[key] = numeric
    return normalised


def _scope_matches(record: Mapping[str, Any], model_family: str, horizon: int) -> bool:
    """Return whether a registry record belongs to one model scope."""
    try:
        record_horizon = int(record.get("horizon", -1))
    except (TypeError, ValueError):
        return False
    return (
        str(record.get("model_family", "")) == model_family
        and record_horizon == int(horizon)
    )


def _record_sort_key(record: Mapping[str, Any]) -> tuple[str, str, str]:
    return (
        str(record.get("run_key", "")),
        str(record.get("created_at", "")),
        str(record.get("model_version", "")),
    )


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def build_model_version(model_family: str, horizon: int, run_key: str) -> str:
    """Build a stable, human-readable candidate version for one market run."""
    family = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(model_family).strip()).strip("_")
    key = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(run_key).strip()).strip("_")
    return f"{family}_h{int(horizon)}_{key[:10]}"


@dataclass(frozen=True)
class ChallengerAssessment:
    """Pure decision result, which can be recorded after training succeeds."""

    model_family: str
    horizon: int
    model_version: str
    run_key: str
    trained_until: str
    metrics: dict[str, float]
    accepted: bool
    decision: str
    reason: str
    deltas: dict[str, float]
    champion_model_version: str | None
    champion_metrics: dict[str, float]
    quality_passed: bool
    min_quality_dates: int
    max_regression: float

    def to_record(self, *, created_at: str | None = None) -> dict[str, Any]:
        """Convert the assessment to the versioned JSON record shape."""
        return {
            "model_family": self.model_family,
            "horizon": self.horizon,
            "model_version": self.model_version,
            "run_key": self.run_key,
            "trained_until": self.trained_until,
            "status": "champion" if self.accepted else "rejected",
            "decision": self.decision,
            "reason": self.reason,
            "quality_passed": self.quality_passed,
            "min_quality_dates": self.min_quality_dates,
            "max_regression": self.max_regression,
            "metrics": dict(self.metrics),
            "deltas_vs_champion": dict(self.deltas),
            "champion_model_version": self.champion_model_version,
            "champion_metrics": dict(self.champion_metrics),
            "created_at": created_at or _utc_now(),
        }


def assess_challenger(
    candidate_metrics: Mapping[str, Any],
    champion: Mapping[str, Any] | None,
    *,
    model_family: str,
    horizon: int,
    model_version: str,
    run_key: str,
    trained_until: str,
    min_quality_dates: int,
    max_regression: float = DEFAULT_MAX_REGRESSION,
    quality_passed: bool = True,
    quality_reason: str | None = None,
) -> ChallengerAssessment:
    """Assess a candidate against the active champion without writing state.

    ``max_regression`` is an absolute return-point tolerance.  For example,
    ``0.002`` means a candidate may be at most 0.20 percentage points below
    the champion on either execution ranking metric.
    """
    if not math.isfinite(max_regression) or max_regression < 0:
        raise ValueError("max_regression must be a finite non-negative number")

    candidate = _normalise_metrics(candidate_metrics)
    champion_metrics = _normalise_metrics(
        champion.get("metrics") if champion is not None else None
    )
    champion_version = (
        str(champion.get("model_version", "")).strip()
        if champion is not None
        else ""
    ) or None
    min_dates = max(0, int(min_quality_dates))

    def result(
        *,
        accepted: bool,
        decision: str,
        reason: str,
        deltas: dict[str, float] | None = None,
    ) -> ChallengerAssessment:
        return ChallengerAssessment(
            model_family=str(model_family),
            horizon=int(horizon),
            model_version=str(model_version),
            run_key=str(run_key),
            trained_until=str(trained_until),
            metrics=candidate,
            accepted=accepted,
            decision=decision,
            reason=reason,
            deltas=deltas or {},
            champion_model_version=champion_version,
            champion_metrics=champion_metrics,
            quality_passed=quality_passed,
            min_quality_dates=min_dates,
            max_regression=float(max_regression),
        )

    if not quality_passed:
        return result(
            accepted=False,
            decision="quality_gate_failed",
            reason=quality_reason or "Existing execution quality gate failed.",
        )

    missing_candidate = [
        metric for metric in COMPARISON_METRICS if metric not in candidate
    ]
    if missing_candidate:
        return result(
            accepted=False,
            decision="missing_execution_metrics",
            reason=(
                "Candidate is missing comparable execution metrics: "
                + ", ".join(missing_candidate)
            ),
        )

    evaluation_dates = candidate.get("execution_evaluation_dates")
    if evaluation_dates is None or evaluation_dates < min_dates:
        return result(
            accepted=False,
            decision="insufficient_evaluation_dates",
            reason=(
                "Candidate has "
                f"{evaluation_dates or 0:.0f} execution evaluation dates; "
                f"minimum is {min_dates}."
            ),
        )

    if champion is None:
        return result(
            accepted=True,
            decision="bootstrap_champion",
            reason="No previous champion exists for this model scope.",
        )

    missing_champion = [
        metric for metric in COMPARISON_METRICS if metric not in champion_metrics
    ]
    if missing_champion:
        return result(
            accepted=False,
            decision="champion_metrics_unavailable",
            reason=(
                "Current champion is missing comparable execution metrics: "
                + ", ".join(missing_champion)
            ),
        )

    deltas = {
        metric: candidate[metric] - champion_metrics[metric]
        for metric in COMPARISON_METRICS
    }
    regressions = {
        metric: delta
        for metric, delta in deltas.items()
        if delta < -max_regression
    }
    if regressions:
        detail = ", ".join(
            f"{metric} delta={delta:+.2%}"
            for metric, delta in regressions.items()
        )
        return result(
            accepted=False,
            decision="rejected_material_regression",
            reason=(
                f"Candidate regressed beyond {max_regression:.2%} tolerance: "
                + detail
            ),
            deltas=deltas,
        )

    return result(
        accepted=True,
        decision="promoted_no_material_regression",
        reason=(
            "Candidate passed the execution quality gate and stayed within "
            f"{max_regression:.2%} of the current champion."
        ),
        deltas=deltas,
    )


class ModelRegistry:
    """Read, assess, and atomically persist model publication decisions."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def _empty_state(self) -> dict[str, Any]:
        return {
            "schema_version": REGISTRY_SCHEMA_VERSION,
            "records": [],
            "last_decision": None,
            "updated_at": None,
        }

    def load(self) -> dict[str, Any]:
        """Load a trusted registry, treating a missing file as first run."""
        if not self.path.exists():
            return self._empty_state()
        try:
            with self.path.open("r", encoding="utf-8") as handle:
                state = json.load(handle)
        except (OSError, json.JSONDecodeError) as exc:
            raise ModelRegistryError(
                f"Cannot read model registry {self.path}: {exc}"
            ) from exc
        if not isinstance(state, dict):
            raise ModelRegistryError(f"Model registry {self.path} must contain an object")
        if state.get("schema_version") != REGISTRY_SCHEMA_VERSION:
            raise ModelRegistryError(
                f"Unsupported model registry schema: {state.get('schema_version')}"
            )
        records = state.get("records")
        if not isinstance(records, list) or any(not isinstance(item, dict) for item in records):
            raise ModelRegistryError(
                f"Model registry {self.path} has an invalid records collection"
            )
        return state

    def _write(self, state: Mapping[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_name(f".{self.path.name}.{uuid.uuid4().hex}.tmp")
        try:
            with temporary.open("w", encoding="utf-8") as handle:
                json.dump(state, handle, ensure_ascii=False, indent=2, sort_keys=True)
                handle.write("\n")
            temporary.replace(self.path)
        except OSError as exc:
            raise ModelRegistryError(
                f"Cannot write model registry {self.path}: {exc}"
            ) from exc
        finally:
            if temporary.exists():
                temporary.unlink()

    @staticmethod
    def _latest_champion_from_state(
        state: Mapping[str, Any],
        model_family: str,
        horizon: int,
    ) -> dict[str, Any] | None:
        champions = [
            record
            for record in state.get("records", [])
            if _scope_matches(record, model_family, horizon)
            and record.get("status") == "champion"
        ]
        if not champions:
            return None
        return max(champions, key=_record_sort_key)

    def latest_champion(self, model_family: str, horizon: int) -> dict[str, Any] | None:
        """Return the active champion for one model family and horizon."""
        return self._latest_champion_from_state(self.load(), model_family, horizon)

    def assess_candidate(
        self,
        candidate_metrics: Mapping[str, Any],
        *,
        model_family: str,
        horizon: int,
        model_version: str,
        run_key: str,
        trained_until: str,
        min_quality_dates: int,
        max_regression: float = DEFAULT_MAX_REGRESSION,
        quality_passed: bool = True,
        quality_reason: str | None = None,
    ) -> ChallengerAssessment:
        """Assess a candidate using the latest persisted champion."""
        state = self.load()
        champion = self._latest_champion_from_state(state, model_family, horizon)
        return assess_challenger(
            candidate_metrics,
            champion,
            model_family=model_family,
            horizon=horizon,
            model_version=model_version,
            run_key=run_key,
            trained_until=trained_until,
            min_quality_dates=min_quality_dates,
            max_regression=max_regression,
            quality_passed=quality_passed,
            quality_reason=quality_reason,
        )

    def record_assessment(self, assessment: ChallengerAssessment) -> dict[str, Any]:
        """Persist one assessment and retire an older champion on promotion."""
        state = self.load()
        records = list(state["records"])
        matching_indexes = [
            index
            for index, record in enumerate(records)
            if _scope_matches(record, assessment.model_family, assessment.horizon)
            and str(record.get("run_key", "")) == assessment.run_key
        ]
        if matching_indexes:
            existing = records[matching_indexes[-1]]
            if existing.get("status") == "champion" and not assessment.accepted:
                # A rerun of the same already-published candidate must not
                # demote itself merely because an upstream data provider
                # returned a slightly different snapshot.
                return existing
            records = [
                record for index, record in enumerate(records)
                if index not in set(matching_indexes)
            ]

        created_at = _utc_now()
        if assessment.accepted:
            for record in records:
                if (
                    _scope_matches(record, assessment.model_family, assessment.horizon)
                    and record.get("status") == "champion"
                ):
                    record["status"] = "retired"
                    record["retired_at"] = created_at

        record = assessment.to_record(created_at=created_at)
        records.append(record)
        state["records"] = records
        state["last_decision"] = {
            "model_family": assessment.model_family,
            "horizon": assessment.horizon,
            "model_version": assessment.model_version,
            "run_key": assessment.run_key,
            "status": record["status"],
            "decision": assessment.decision,
            "reason": assessment.reason,
            "created_at": created_at,
        }
        state["updated_at"] = created_at
        self._write(state)
        log.info(
            "Model registry: %s %s (%s)",
            record["status"],
            assessment.model_version,
            assessment.decision,
        )
        return record

    def summary(self, model_family: str, horizon: int) -> dict[str, Any]:
        """Return dashboard-safe state for one production model scope."""
        state = self.load()
        scoped = [
            record
            for record in state["records"]
            if _scope_matches(record, model_family, horizon)
        ]
        latest = max(scoped, key=_record_sort_key) if scoped else None
        champion = self._latest_champion_from_state(state, model_family, horizon)
        return {
            "path": str(self.path),
            "record_count": len(scoped),
            "latest": latest,
            "champion": champion,
            "updated_at": state.get("updated_at"),
        }
