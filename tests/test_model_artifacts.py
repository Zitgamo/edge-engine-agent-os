from __future__ import annotations

import pytest

from src.model.artifacts import ModelArtifactError, ModelArtifactStore


HORIZONS = (1, 5, 10, 20)


def _write_models(root, version: str) -> dict[int, object]:
    root.mkdir(parents=True, exist_ok=True)
    paths = {}
    for horizon in HORIZONS:
        path = root / f"model_h{horizon}.json"
        path.write_bytes(f"{version}:T+{horizon}".encode("utf-8"))
        paths[horizon] = path
    return paths


def test_publish_and_restore_validates_the_complete_ensemble(tmp_path) -> None:
    source_paths = _write_models(tmp_path / "source", "v1")
    store = ModelArtifactStore(tmp_path / "artifacts")

    manifest = store.publish(
        "v1",
        source_paths,
        trained_until="2026-09-04",
    )

    assert manifest["model_version"] == "v1"
    assert len(manifest["models"]) == len(HORIZONS)
    assert store.current()["archive_sha256"] == manifest["archive_sha256"]

    runtime_paths = _write_models(tmp_path / "runtime", "stale")
    restored = store.restore_current(runtime_paths)

    assert restored["model_version"] == "v1"
    for horizon, path in runtime_paths.items():
        assert path.read_text(encoding="utf-8") == f"v1:T+{horizon}"


def test_publish_keeps_previous_and_rollback_restores_it(tmp_path) -> None:
    source_paths = _write_models(tmp_path / "source", "v1")
    store = ModelArtifactStore(tmp_path / "artifacts")
    store.publish("v1", source_paths, trained_until="2026-09-03")

    _write_models(tmp_path / "source", "v2")
    store.publish("v2", source_paths, trained_until="2026-09-04")
    runtime_paths = _write_models(tmp_path / "runtime", "stale")

    rolled_back = store.restore_previous(runtime_paths)

    assert rolled_back["model_version"] == "v1"
    assert store.current()["model_version"] == "v1"
    for horizon, path in runtime_paths.items():
        assert path.read_text(encoding="utf-8") == f"v1:T+{horizon}"


def test_corrupt_archive_fails_checksum_validation(tmp_path) -> None:
    source_paths = _write_models(tmp_path / "source", "v1")
    store = ModelArtifactStore(tmp_path / "artifacts")
    store.publish("v1", source_paths, trained_until="2026-09-04")
    store.current_archive.write_bytes(store.current_archive.read_bytes() + b"tampered")

    with pytest.raises(ModelArtifactError, match="checksum mismatch"):
        store.restore_current(_write_models(tmp_path / "runtime", "stale"))


def test_publish_rejects_missing_model_file(tmp_path) -> None:
    source_paths = _write_models(tmp_path / "source", "v1")
    source_paths[20].unlink()
    store = ModelArtifactStore(tmp_path / "artifacts")

    with pytest.raises(ModelArtifactError, match=r"missing for T\+20"):
        store.publish("v1", source_paths, trained_until="2026-09-04")
