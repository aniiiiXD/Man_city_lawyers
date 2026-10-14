"""Append-only event stream used as the authoritative episode trace."""

from __future__ import annotations

from typing import Any, Mapping

from work_web.contracts import WorkEvent, readonly_mapping


class EventLog:
    """In-memory append-only log with monotonic sequence numbers.

    Readers receive immutable event records.  There is intentionally no edit
    or delete operation; corrections are represented by later events.
    """

    def __init__(self) -> None:
        self._events: list[WorkEvent] = []

    def append(
        self,
        *,
        tick: int,
        kind: str,
        task_id: str | None = None,
        actor_id: str | None = None,
        payload: Mapping[str, Any] | None = None,
    ) -> WorkEvent:
        if tick < 0:
            raise ValueError("event tick must be non-negative")
        if not kind:
            raise ValueError("event kind cannot be empty")
        event = WorkEvent(
            sequence=len(self._events) + 1,
            tick=tick,
            kind=kind,
            task_id=task_id,
            actor_id=actor_id,
            payload=readonly_mapping(payload),
        )
        self._events.append(event)
        return event

    @property
    def events(self) -> tuple[WorkEvent, ...]:
        return tuple(self._events)

    def since(self, sequence: int = 0) -> tuple[WorkEvent, ...]:
        """Return events after a sequence cursor (0 returns the whole log)."""
        if sequence < 0:
            raise ValueError("sequence cursor must be non-negative")
        return tuple(event for event in self._events if event.sequence > sequence)

    def __len__(self) -> int:
        return len(self._events)
