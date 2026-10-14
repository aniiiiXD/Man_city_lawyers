import sqlite3
from pathlib import Path

import pytest

from dev_web.contracts import AgentTask
from dev_web.events import EventStore
from dev_web.providers import ManualQueueProvider, ProviderUnavailable, UnavailableProvider


def test_event_store_blocks_updates_and_deletes(tmp_path: Path):
    path = tmp_path / "events.sqlite3"
    with EventStore(path) as store:
        event = store.append(run_id="run-1", event_type="started", payload={"ok": True})
        assert store.events("run-1")[0].event_id == event.event_id
        with pytest.raises(sqlite3.IntegrityError, match="append-only"):
            store.connection.execute("UPDATE events SET actor='changed'")
        with pytest.raises(sqlite3.IntegrityError, match="append-only"):
            store.connection.execute("DELETE FROM events")


def sample_task() -> AgentTask:
    return AgentTask(
        task_id="task-1", title="Implement a unit", instructions="Make a small change.",
        file_scopes=("src/",), required_checks=("pytest",),
    )


def test_manual_provider_creates_packet_without_invoking_worker(tmp_path: Path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    result = ManualQueueProvider().dispatch(sample_task(), workspace, tmp_path / "queue")

    assert result.state == "queued"
    assert "no agent was invoked" in result.message
    assert Path(result.packet_path).is_file()
    assert (tmp_path / "queue" / "task.json").is_file()


def test_unavailable_provider_fails_instead_of_claiming_native_subagent(tmp_path: Path):
    with pytest.raises(ProviderUnavailable, match="cannot call native Codex"):
        UnavailableProvider("codex").dispatch(sample_task(), tmp_path, tmp_path)
