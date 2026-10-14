"""Versioned shared artifact store with optimistic concurrency checks."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Mapping

from work_web.contracts import ArtifactRecord, ArtifactRef, readonly_mapping


class ArtifactNotFound(KeyError):
    pass


class ArtifactConflict(RuntimeError):
    """Raised when a writer's expected artifact version is stale."""


class VersionedWorkspace:
    """Append-only artifact histories, indexed by stable logical key."""

    def __init__(self) -> None:
        self._history: dict[str, list[ArtifactRecord]] = defaultdict(list)

    def write(
        self,
        *,
        key: str,
        payload: Mapping[str, Any],
        producer_task_id: str,
        created_at: int,
        expected_version: int | None = None,
    ) -> ArtifactRecord:
        if not key.strip():
            raise ValueError("artifact key cannot be empty")
        if created_at < 0:
            raise ValueError("created_at must be non-negative")
        history = self._history[key]
        current_version = history[-1].version if history else 0
        if expected_version is not None and expected_version != current_version:
            raise ArtifactConflict(
                f"{key!r} is at version {current_version}; writer expected {expected_version}"
            )
        record = ArtifactRecord(
            key=key,
            version=current_version + 1,
            producer_task_id=producer_task_id,
            created_at=created_at,
            payload=readonly_mapping(payload),
        )
        history.append(record)
        return record

    def get(self, key: str, version: int | None = None) -> ArtifactRecord:
        history = self._history.get(key)
        if not history:
            raise ArtifactNotFound(key)
        if version is None:
            return history[-1]
        if version < 1 or version > len(history):
            raise ArtifactNotFound(f"{key}@{version}")
        return history[version - 1]

    def resolve(self, ref: ArtifactRef) -> ArtifactRecord:
        return self.get(ref.key, ref.version)

    def history(self, key: str) -> tuple[ArtifactRecord, ...]:
        return tuple(self._history.get(key, ()))

    def latest_records(self) -> tuple[ArtifactRecord, ...]:
        return tuple(self._history[key][-1] for key in sorted(self._history))

    def latest_versions(self) -> dict[str, int]:
        return {key: records[-1].version for key, records in sorted(self._history.items())}
