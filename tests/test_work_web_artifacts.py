"""Versioning and optimistic concurrency checks for shared artifacts."""

import pytest

from work_web.workspace import ArtifactConflict, ArtifactNotFound, VersionedWorkspace


def test_artifact_versions_are_append_only_and_resolvable() -> None:
    store = VersionedWorkspace()
    first = store.write(
        key="brief",
        payload={"text": "draft", "claims": [{"source": "log"}]},
        producer_task_id="draft",
        created_at=1,
    )
    second = store.write(
        key="brief",
        payload={"text": "revised"},
        producer_task_id="review",
        created_at=2,
        expected_version=1,
    )

    assert first.version == 1
    assert second.version == 2
    assert store.resolve(first.ref).payload["text"] == "draft"
    assert store.get("brief").payload["text"] == "revised"
    assert len(store.history("brief")) == 2
    with pytest.raises(TypeError):
        first.payload["claims"][0]["source"] = "changed"


def test_stale_write_is_rejected_without_mutating_history() -> None:
    store = VersionedWorkspace()
    store.write(key="evidence", payload={"n": 1}, producer_task_id="a", created_at=0)

    with pytest.raises(ArtifactConflict):
        store.write(
            key="evidence",
            payload={"n": 2},
            producer_task_id="b",
            created_at=1,
            expected_version=0,
        )
    with pytest.raises(ArtifactNotFound):
        store.get("evidence", version=2)
    assert store.get("evidence").payload["n"] == 1
