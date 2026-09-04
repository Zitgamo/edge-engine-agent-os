from __future__ import annotations

import json

import pytest

from src.model.registry import (
    ModelRegistry,
    ModelRegistryError,
    assess_challenger,
    build_model_version,
)


def _metrics(*, top3_return: float = 0.01, spread: float = 0.02) -> dict[str, float]:
    return {
        "roc_auc": 0.54,
        "execution_evaluation_dates": 40,
        "execution_top3_excess_return": top3_return,
        "execution_top3_spread": spread,
    }


def _assessment(registry: ModelRegistry, metrics: dict[str, float], run_key: str):
    return registry.assess_candidate(
        metrics,
        model_family="test_model",
        horizon=20,
        model_version=build_model_version("test_model", 20, run_key),
        run_key=run_key,
        trained_until=run_key,
        min_quality_dates=30,
        max_regression=0.002,
    )


def test_first_candidate_bootstraps_and_persists_champion(tmp_path) -> None:
    path = tmp_path / "model_registry.json"
    registry = ModelRegistry(path)

    assessment = _assessment(registry, _metrics(), "2026-09-04")
    assert assessment.accepted is True
    assert assessment.decision == "bootstrap_champion"

    record = registry.record_assessment(assessment)

    assert record["status"] == "champion"
    assert registry.latest_champion("test_model", 20)["model_version"] == record[
        "model_version"
    ]
    persisted = json.loads(path.read_text(encoding="utf-8"))
    assert persisted["schema_version"] == 1
    assert len(persisted["records"]) == 1


def test_candidate_within_tolerance_retires_previous_champion(tmp_path) -> None:
    registry = ModelRegistry(tmp_path / "model_registry.json")
    first = _assessment(registry, _metrics(), "2026-09-03")
    registry.record_assessment(first)

    second = _assessment(
        registry,
        _metrics(top3_return=0.009, spread=0.019),
        "2026-09-04",
    )
    assert second.accepted is True
    assert second.decision == "promoted_no_material_regression"
    registry.record_assessment(second)

    state = registry.load()
    statuses = {
        record["run_key"]: record["status"] for record in state["records"]
    }
    assert statuses == {"2026-09-03": "retired", "2026-09-04": "champion"}


def test_material_regression_is_rejected_and_champion_remains(tmp_path) -> None:
    registry = ModelRegistry(tmp_path / "model_registry.json")
    first = _assessment(registry, _metrics(), "2026-09-03")
    registry.record_assessment(first)

    second = _assessment(
        registry,
        _metrics(top3_return=0.007, spread=0.019),
        "2026-09-04",
    )
    assert second.accepted is False
    assert second.decision == "rejected_material_regression"
    registry.record_assessment(second)

    champion = registry.latest_champion("test_model", 20)
    assert champion["run_key"] == "2026-09-03"
    assert champion["status"] == "champion"
    assert any(
        record["run_key"] == "2026-09-04" and record["status"] == "rejected"
        for record in registry.load()["records"]
    )


def test_ready_live_underperformance_blocks_new_promotion(tmp_path) -> None:
    registry = ModelRegistry(tmp_path / "model_registry.json")
    first = _assessment(registry, _metrics(), "2026-09-03")
    registry.record_assessment(first)

    second = registry.assess_candidate(
        _metrics(top3_return=0.02, spread=0.03),
        model_family="test_model",
        horizon=20,
        model_version=build_model_version("test_model", 20, "2026-09-04"),
        run_key="2026-09-04",
        trained_until="2026-09-04",
        min_quality_dates=30,
        live_validation={
            "ready": True,
            "health_status": "underperforming",
            "avg_excess_return": -0.01,
            "win_rate": 0.30,
        },
    )

    assert second.accepted is False
    assert second.decision == "champion_live_validation_failed"
    record = registry.record_assessment(second)
    assert record["status"] == "rejected"
    assert registry.latest_champion("test_model", 20)["run_key"] == "2026-09-03"


def test_quality_failure_is_recorded_without_bootstrapping(tmp_path) -> None:
    registry = ModelRegistry(tmp_path / "model_registry.json")
    assessment = registry.assess_candidate(
        {},
        model_family="test_model",
        horizon=20,
        model_version="test_model_h20_2026-09-04",
        run_key="2026-09-04",
        trained_until="2026-09-04",
        min_quality_dates=30,
        quality_passed=False,
        quality_reason="negative execution spread",
    )

    assert assessment.accepted is False
    assert assessment.decision == "quality_gate_failed"
    record = registry.record_assessment(assessment)
    assert record["status"] == "rejected"
    assert registry.latest_champion("test_model", 20) is None


def test_corrupt_registry_fails_closed(tmp_path) -> None:
    path = tmp_path / "model_registry.json"
    path.write_text("{not-json", encoding="utf-8")

    with pytest.raises(ModelRegistryError, match="Cannot read model registry"):
        ModelRegistry(path).load()


def test_assessment_requires_non_negative_tolerance() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        assess_challenger(
            _metrics(),
            None,
            model_family="test_model",
            horizon=20,
            model_version="v1",
            run_key="2026-09-04",
            trained_until="2026-09-04",
            min_quality_dates=30,
            max_regression=-0.01,
        )
