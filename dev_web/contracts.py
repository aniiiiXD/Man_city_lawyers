"""Validated contracts shared by the task planner, queue, and integrations."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import StrEnum
import re
from typing import Any, Iterable

from .paths import PathSafetyError, PathScope


_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,79}$")


class ContractError(ValueError):
    """Raised when a task contract is malformed or unsafe."""


class TaskStatus(StrEnum):
    BLOCKED = "blocked"
    READY = "ready"
    QUEUED = "queued"
    IN_PROGRESS = "in_progress"
    SUBMITTED = "submitted"
    CHECKS_PASSED = "checks_passed"
    CHECKS_FAILED = "checks_failed"
    REVIEW_APPROVED = "review_approved"
    REVIEW_REJECTED = "review_rejected"
    INTEGRATED = "integrated"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass(frozen=True, slots=True)
class AgentTask:
    """One bounded unit of repository work in a dependency graph."""

    task_id: str
    title: str
    instructions: str
    file_scopes: tuple[str, ...]
    dependencies: tuple[str, ...] = ()
    required_checks: tuple[str, ...] = ()
    max_attempts: int = 1
    author: str = "unassigned"
    expected_outputs: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not _IDENTIFIER.fullmatch(self.task_id):
            raise ContractError(f"Invalid task id: {self.task_id!r}")
        if not self.title.strip() or not self.instructions.strip():
            raise ContractError("Task title and instructions must be non-empty")
        if not self.file_scopes:
            raise ContractError(f"Task {self.task_id!r} needs at least one file scope")
        if not self.required_checks:
            raise ContractError(f"Task {self.task_id!r} needs at least one required check")
        if self.max_attempts < 1 or self.max_attempts > 10:
            raise ContractError("max_attempts must be between 1 and 10")
        if len(set(self.dependencies)) != len(self.dependencies):
            raise ContractError(f"Task {self.task_id!r} repeats a dependency")
        if len(set(self.required_checks)) != len(self.required_checks):
            raise ContractError(f"Task {self.task_id!r} repeats a required check")
        if self.task_id in self.dependencies:
            raise ContractError(f"Task {self.task_id!r} cannot depend on itself")
        for dependency in self.dependencies:
            if not _IDENTIFIER.fullmatch(dependency):
                raise ContractError(f"Invalid dependency id: {dependency!r}")
        try:
            PathScope(self.file_scopes)
        except PathSafetyError as exc:
            raise ContractError(f"Unsafe file scope: {exc}") from exc

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> AgentTask:
        data = dict(raw)
        for key in ("file_scopes", "dependencies", "required_checks", "expected_outputs"):
            data[key] = tuple(data.get(key, ()))
        return cls(**data)


@dataclass(frozen=True, slots=True)
class TaskGraph:
    """A DAG of independently assignable tasks with deterministic validation."""

    tasks: tuple[AgentTask, ...]
    graph_id: str = "run"
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not _IDENTIFIER.fullmatch(self.graph_id):
            raise ContractError(f"Invalid graph id: {self.graph_id!r}")
        if not self.tasks:
            raise ContractError("A task graph must contain at least one task")
        ids = [task.task_id for task in self.tasks]
        if len(set(ids)) != len(ids):
            raise ContractError("Task ids must be unique within a graph")
        known = set(ids)
        for task in self.tasks:
            missing = set(task.dependencies) - known
            if missing:
                raise ContractError(f"Task {task.task_id!r} has unknown dependencies: {sorted(missing)}")
        self.topological_order()  # raises for a cycle

    def topological_order(self) -> tuple[str, ...]:
        by_id = {task.task_id: task for task in self.tasks}
        remaining = {task_id: set(task.dependencies) for task_id, task in by_id.items()}
        result: list[str] = []
        while remaining:
            ready = sorted(task_id for task_id, deps in remaining.items() if not deps)
            if not ready:
                raise ContractError("Task graph contains a dependency cycle")
            for task_id in ready:
                result.append(task_id)
                remaining.pop(task_id)
            for deps in remaining.values():
                deps.difference_update(ready)
        return tuple(result)

    def ready_task_ids(self, completed: Iterable[str]) -> tuple[str, ...]:
        completed_set = set(completed)
        return tuple(
            task.task_id
            for task in self.tasks
            if task.task_id not in completed_set and set(task.dependencies) <= completed_set
        )

    def to_dict(self) -> dict[str, Any]:
        return {"graph_id": self.graph_id, "metadata": self.metadata,
                "tasks": [task.to_dict() for task in self.tasks]}

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> TaskGraph:
        return cls(
            tasks=tuple(AgentTask.from_dict(item) for item in raw["tasks"]),
            graph_id=raw.get("graph_id", "run"),
            metadata=raw.get("metadata", {}),
        )
