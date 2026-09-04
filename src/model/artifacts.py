"""Versioned model artifacts with checksum verification and rollback support.

GitHub Actions runners are ephemeral and the runtime ``models/`` directory is
ignored by git.  This store keeps the active and previous ensemble as compact,
deterministic tarballs so a fresh runner can restore a known-good champion
before training, and a failed publication can roll back to the previous one.
"""

from __future__ import annotations

import gzip
import hashlib
import io
import json
import logging
import tarfile
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

log = logging.getLogger(__name__)

ARTIFACT_SCHEMA_VERSION = 1


class ModelArtifactError(RuntimeError):
    """Raised when an artifact is missing, corrupt, or cannot be persisted."""


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_bytes(payload)
        temporary.replace(path)
    except OSError as exc:
        raise ModelArtifactError(f"Cannot write model artifact file {path}: {exc}") from exc
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_copy(source: Path, destination: Path) -> None:
    try:
        _atomic_write_bytes(destination, source.read_bytes())
    except OSError as exc:
        raise ModelArtifactError(
            f"Cannot copy model artifact {source} to {destination}: {exc}"
        ) from exc


class ModelArtifactStore:
    """Store the current and previous complete ensemble atomically."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    @property
    def current_archive(self) -> Path:
        return self.root / "champion.tar.gz"

    @property
    def current_manifest(self) -> Path:
        return self.root / "champion_manifest.json"

    @property
    def previous_archive(self) -> Path:
        return self.root / "previous.tar.gz"

    @property
    def previous_manifest(self) -> Path:
        return self.root / "previous_manifest.json"

    def _read_manifest(self, path: Path) -> dict[str, Any] | None:
        exists = path.exists()
        if not exists:
            return None
        try:
            state = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ModelArtifactError(f"Cannot read model manifest {path}: {exc}") from exc
        if not isinstance(state, dict):
            raise ModelArtifactError(f"Model manifest {path} must contain an object")
        if state.get("schema_version") != ARTIFACT_SCHEMA_VERSION:
            raise ModelArtifactError(
                f"Unsupported model artifact schema in {path}: "
                f"{state.get('schema_version')}"
            )
        models = state.get("models")
        if not isinstance(models, list) or not models:
            raise ModelArtifactError(f"Model manifest {path} has no model files")
        if not state.get("model_version") or not state.get("archive_sha256"):
            raise ModelArtifactError(f"Model manifest {path} is missing identity/checksum")
        return state

    def _validated_manifest(
        self,
        manifest_path: Path,
        archive_path: Path,
    ) -> dict[str, Any] | None:
        manifest = self._read_manifest(manifest_path)
        if manifest is None:
            if archive_path.exists():
                raise ModelArtifactError(
                    f"Model archive exists without its manifest: {archive_path}"
                )
            return None
        if not archive_path.exists():
            raise ModelArtifactError(
                f"Model manifest exists without its archive: {manifest_path}"
            )
        try:
            archive_bytes = archive_path.read_bytes()
        except OSError as exc:
            raise ModelArtifactError(f"Cannot read model archive {archive_path}: {exc}") from exc
        expected = str(manifest["archive_sha256"])
        actual = _sha256(archive_bytes)
        if actual != expected:
            raise ModelArtifactError(
                f"Model archive checksum mismatch for {archive_path}: "
                f"expected {expected}, got {actual}"
            )
        return manifest

    def _create_archive(self, files: Mapping[str, bytes], output: Path) -> None:
        output.parent.mkdir(parents=True, exist_ok=True)
        try:
            with output.open("wb") as raw:
                with gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as compressed:
                    with tarfile.open(fileobj=compressed, mode="w") as archive:
                        for name in sorted(files):
                            info = tarfile.TarInfo(name=name)
                            info.size = len(files[name])
                            info.mtime = 0
                            info.mode = 0o644
                            info.uid = 0
                            info.gid = 0
                            info.uname = ""
                            info.gname = ""
                            archive.addfile(info, io.BytesIO(files[name]))
        except (OSError, tarfile.TarError) as exc:
            raise ModelArtifactError(f"Cannot create model archive {output}: {exc}") from exc

    def _write_manifest(self, path: Path, manifest: Mapping[str, Any]) -> None:
        try:
            payload = json.dumps(
                manifest,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            ).encode("utf-8") + b"\n"
        except (TypeError, ValueError) as exc:
            raise ModelArtifactError(f"Model manifest is not JSON serializable: {exc}") from exc
        _atomic_write_bytes(path, payload)

    def current(self) -> dict[str, Any] | None:
        """Validate and return the active artifact manifest, if present."""
        return self._validated_manifest(self.current_manifest, self.current_archive)

    def previous(self) -> dict[str, Any] | None:
        """Validate and return the rollback artifact manifest, if present."""
        return self._validated_manifest(self.previous_manifest, self.previous_archive)

    def publish(
        self,
        model_version: str,
        model_paths: Mapping[int, str | Path],
        *,
        trained_until: str,
    ) -> dict[str, Any]:
        """Publish a complete ensemble while retaining the old one as backup."""
        if not model_paths:
            raise ModelArtifactError("Cannot publish an empty model ensemble")

        files: dict[str, bytes] = {}
        model_records: list[dict[str, Any]] = []
        for raw_horizon, raw_path in sorted(model_paths.items(), key=lambda item: int(item[0])):
            horizon = int(raw_horizon)
            path = Path(raw_path)
            if not path.is_file():
                raise ModelArtifactError(f"Model file is missing for T+{horizon}: {path}")
            try:
                payload = path.read_bytes()
            except OSError as exc:
                raise ModelArtifactError(f"Cannot read model file {path}: {exc}") from exc
            if not payload:
                raise ModelArtifactError(f"Model file is empty for T+{horizon}: {path}")
            filename = f"xgboost_model_h{horizon}.json"
            files[filename] = payload
            model_records.append({
                "horizon": horizon,
                "filename": filename,
                "sha256": _sha256(payload),
                "size_bytes": len(payload),
            })

        self.root.mkdir(parents=True, exist_ok=True)
        old_manifest = self.current()
        temporary_archive = self.root / f".candidate.{uuid.uuid4().hex}.tar.gz"
        temporary_manifest = self.root / f".candidate.{uuid.uuid4().hex}.json"
        try:
            self._create_archive(files, temporary_archive)
            archive_bytes = temporary_archive.read_bytes()
            manifest = {
                "schema_version": ARTIFACT_SCHEMA_VERSION,
                "model_version": str(model_version),
                "trained_until": str(trained_until),
                "created_at": _utc_now(),
                "archive_sha256": _sha256(archive_bytes),
                "archive_size_bytes": len(archive_bytes),
                "models": model_records,
            }
            self._write_manifest(temporary_manifest, manifest)

            if old_manifest is not None:
                _atomic_copy(self.current_archive, self.previous_archive)
                _atomic_copy(self.current_manifest, self.previous_manifest)

            temporary_archive.replace(self.current_archive)
            temporary_manifest.replace(self.current_manifest)
        except (OSError, ModelArtifactError) as exc:
            # If the new archive replaced the current one but its manifest did
            # not, restore the last validated snapshot before failing closed.
            try:
                if old_manifest is not None and self.previous():
                    _atomic_copy(self.previous_archive, self.current_archive)
                    _atomic_copy(self.previous_manifest, self.current_manifest)
                elif old_manifest is None:
                    for path in (self.current_archive, self.current_manifest):
                        if path.exists():
                            path.unlink()
            except (OSError, ModelArtifactError) as rollback_exc:
                log.error("Artifact publication rollback failed: %s", rollback_exc)
            raise ModelArtifactError(f"Cannot publish model artifact: {exc}") from exc
        finally:
            for path in (temporary_archive, temporary_manifest):
                if path.exists():
                    path.unlink()

        log.info(
            "Published model artifact %s (%d files, %d bytes)",
            model_version,
            len(model_records),
            manifest["archive_size_bytes"],
        )
        return manifest

    def _restore(
        self,
        manifest: Mapping[str, Any],
        archive_path: Path,
        model_paths: Mapping[int, str | Path],
    ) -> dict[str, Any]:
        destinations = {int(horizon): Path(path) for horizon, path in model_paths.items()}
        try:
            archive_bytes = archive_path.read_bytes()
        except OSError as exc:
            raise ModelArtifactError(f"Cannot read model archive {archive_path}: {exc}") from exc

        try:
            models = manifest["models"]
            if not isinstance(models, list) or not models:
                raise ModelArtifactError("Model artifact manifest has no model records")
            expected_horizons = set(destinations)
            restored_horizons: set[int] = set()
            restored_names: set[str] = set()
            staged: list[tuple[Path, bytes]] = []
            with tarfile.open(fileobj=io.BytesIO(archive_bytes), mode="r:gz") as archive:
                for model in models:
                    horizon = int(model["horizon"])
                    if horizon in restored_horizons:
                        raise ModelArtifactError(
                            f"Duplicate T+{horizon} model in artifact manifest"
                        )
                    if horizon not in destinations:
                        raise ModelArtifactError(
                            f"No destination path supplied for T+{horizon} model"
                        )
                    filename = str(model["filename"])
                    if (
                        not filename
                        or Path(filename).name != filename
                        or filename in {".", ".."}
                        or filename in restored_names
                    ):
                        raise ModelArtifactError(
                            f"Unsafe model filename in manifest: {filename}"
                        )
                    member = archive.getmember(filename)
                    if not member.isfile():
                        raise ModelArtifactError(f"Model archive member is not a file: {filename}")
                    extracted = archive.extractfile(member)
                    if extracted is None:
                        raise ModelArtifactError(f"Cannot extract model archive member: {filename}")
                    payload = extracted.read()
                    if _sha256(payload) != str(model["sha256"]):
                        raise ModelArtifactError(
                            f"Model checksum mismatch for T+{horizon} ({filename})"
                        )
                    restored_horizons.add(horizon)
                    restored_names.add(filename)
                    staged.append((destinations[horizon], payload))
                if set(archive.getnames()) != restored_names:
                    raise ModelArtifactError(
                        "Model archive members do not match the manifest"
                    )
            if restored_horizons != expected_horizons:
                missing = sorted(expected_horizons - restored_horizons)
                raise ModelArtifactError(
                    f"Model artifact is missing destinations for horizons: {missing}"
                )
            for destination, payload in staged:
                _atomic_write_bytes(destination, payload)
        except (KeyError, TypeError, ValueError, OSError, tarfile.TarError) as exc:
            raise ModelArtifactError(
                f"Cannot restore model artifact from {archive_path}: {exc}"
            ) from exc
        return dict(manifest)

    def restore_current(self, model_paths: Mapping[int, str | Path]) -> dict[str, Any] | None:
        """Restore the validated active ensemble into runtime model paths."""
        manifest = self.current()
        if manifest is None:
            return None
        restored = self._restore(manifest, self.current_archive, model_paths)
        log.info("Restored model champion %s", restored["model_version"])
        return restored

    def restore_previous(self, model_paths: Mapping[int, str | Path]) -> dict[str, Any] | None:
        """Make the previous artifact active and restore it into runtime paths."""
        manifest = self.previous()
        if manifest is None:
            for path in (self.current_archive, self.current_manifest):
                if path.exists():
                    path.unlink()
            return None
        _atomic_copy(self.previous_archive, self.current_archive)
        _atomic_copy(self.previous_manifest, self.current_manifest)
        restored = self._restore(manifest, self.current_archive, model_paths)
        log.warning("Rolled back model champion to %s", restored["model_version"])
        return restored
