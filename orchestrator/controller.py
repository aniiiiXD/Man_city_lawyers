"""Small deterministic orchestration policies."""

from __future__ import annotations

from typing import Protocol, Sequence

from work_web.contracts import TaskSnapshot


class DispatchController(Protocol):
    def select_ready(
        self, ready: Sequence[TaskSnapshot], *, now: int, slots: int
    ) -> tuple[str, ...]: ...


class RuleBasedController:
    """Fill available slots with highest-priority ready tasks, ties by id."""

    def select_ready(
        self, ready: Sequence[TaskSnapshot], *, now: int, slots: int
    ) -> tuple[str, ...]:
        del now
        if slots <= 0:
            return ()
        ordered = sorted(ready, key=lambda task: (-task.task.priority, task.task.task_id))
        return tuple(task.task.task_id for task in ordered[:slots])
