"""Source-level checks for append-only work-web traces."""

import pytest

from work_web.event_log import EventLog


def test_event_log_is_append_only_and_supports_cursors() -> None:
    events = EventLog()
    first = events.append(
        tick=0,
        kind="job.started",
        payload={"task_id": "job-1", "details": {"roles": ["researcher"]}},
    )
    second = events.append(tick=2, kind="task.completed", task_id="task-1")

    assert first.sequence == 1
    assert second.sequence == 2
    assert events.since(1) == (second,)
    assert len(events) == 2
    with pytest.raises(TypeError):
        first.payload["task_id"] = "changed"
    with pytest.raises(TypeError):
        first.payload["details"]["roles"] = ("publisher",)
    with pytest.raises(AttributeError):
        first.payload["details"]["roles"].append("publisher")


def test_event_log_rejects_time_travel() -> None:
    events = EventLog()
    with pytest.raises(ValueError):
        events.append(tick=-1, kind="invalid")
