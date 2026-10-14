"""Typed contracts for the synthetic, event-sourced agent work web.

The work web deliberately describes jobs and artifacts as data.  It has no
LLM, network, filesystem, or framework dependency, which makes episodes
replayable and suitable for wrapping in an RL environment.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from types import MappingProxyType
from typing import Any, Mapping


class TaskStatus(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    WAITING_APPROVAL = "waiting_approval"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMED_OUT = "timed_out"


class ApprovalStatus(StrEnum):
    PENDING = "pending"
    GRANTED = "granted"
    DENIED = "denied"
    EXPIRED = "expired"
    CANCELLED = "cancelled"


class ApprovalOrigin(StrEnum):
    SYNTHETIC_REVIEW = "synthetic_review"
    GOVERNANCE_GATE = "governance_gate"


TERMINAL_TASK_STATUSES = frozenset(
    {TaskStatus.SUCCEEDED, TaskStatus.FAILED, TaskStatus.CANCELLED, TaskStatus.TIMED_OUT}
)


def freeze_value(value: Any) -> Any:
    """Recursively freeze JSON-like values stored in public records."""
    if isinstance(value, Mapping):
        return MappingProxyType({key: freeze_value(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(freeze_value(item) for item in value)
    if isinstance(value, (set, frozenset)):
        return frozenset(freeze_value(item) for item in value)
    return value


def to_plain_data(value: Any) -> Any:
    """Convert frozen mappings and sequences into JSON-compatible values."""
    if isinstance(value, Mapping):
        return {key: to_plain_data(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_plain_data(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return [to_plain_data(item) for item in sorted(value, key=repr)]
    return value


def readonly_mapping(value: Mapping[str, Any] | None = None) -> Mapping[str, Any]:
    """Return a recursively read-only copy for immutable public records."""
    return freeze_value(dict(value or {}))


@dataclass(frozen=True, slots=True)
class WorkTask:
    """A bounded unit of work in the shared workspace."""

    task_id: str
    role: str
    title: str
    instructions: str
    dependencies: tuple[str, ...] = ()
    output_keys: tuple[str, ...] = ()
    priority: int = 0
    max_attempts: int = 2
    timeout_ticks: int = 8
    approval_timeout_ticks: int = 20
    deadline_tick: int | None = None
    approval_scope: str | None = None
    parent_task_id: str | None = None
    delegation_id: str | None = None

    def __post_init__(self) -> None:
        if not self.task_id.strip():
            raise ValueError("task_id cannot be empty")
        if not self.role.strip():
            raise ValueError("role cannot be empty")
        if self.max_attempts < 1:
            raise ValueError("max_attempts must be at least one")
        if self.timeout_ticks < 1:
            raise ValueError("timeout_ticks must be at least one")
        if self.approval_timeout_ticks < 1:
            raise ValueError("approval_timeout_ticks must be at least one")
        if self.deadline_tick is not None and self.deadline_tick < 0:
            raise ValueError("deadline_tick must be non-negative")
        if len(set(self.dependencies)) != len(self.dependencies):
            raise ValueError("task dependencies must be unique")
        if self.task_id in self.dependencies:
            raise ValueError("a task cannot depend on itself")
        if len(set(self.output_keys)) != len(self.output_keys):
            raise ValueError("task output keys must be unique")


@dataclass(frozen=True, slots=True)
class ArtifactRef:
    key: str
    version: int


@dataclass(frozen=True, slots=True)
class ArtifactDraft:
    key: str
    payload: Mapping[str, Any]
    expected_version: int | None = None

    def __post_init__(self) -> None:
        if not self.key.strip():
            raise ValueError("artifact key cannot be empty")
        object.__setattr__(self, "payload", readonly_mapping(self.payload))


@dataclass(frozen=True, slots=True)
class ArtifactRecord:
    key: str
    version: int
    producer_task_id: str
    created_at: int
    payload: Mapping[str, Any] = field(default_factory=readonly_mapping)

    @property
    def ref(self) -> ArtifactRef:
        return ArtifactRef(self.key, self.version)


@dataclass(frozen=True, slots=True)
class TaskSnapshot:
    task: WorkTask
    status: TaskStatus
    attempt: int
    created_at: int
    started_at: int | None = None
    finished_at: int | None = None
    retry_at: int | None = None
    worker_id: str | None = None
    error: str | None = None
    output_refs: tuple[ArtifactRef, ...] = ()
    approval_request_id: str | None = None


@dataclass(frozen=True, slots=True)
class ApprovalRequest:
    request_id: str
    task_id: str
    scope: str
    reason: str
    created_at: int
    deadline_tick: int | None = None
    status: ApprovalStatus = ApprovalStatus.PENDING
    resolved_at: int | None = None
    note: str | None = None
    origin: ApprovalOrigin = ApprovalOrigin.SYNTHETIC_REVIEW
    locally_resolvable: bool = True
    plan_id: str | None = None
    decision_id: str | None = None
    delegation_id: str | None = None


@dataclass(frozen=True, slots=True)
class WorkEvent:
    sequence: int
    tick: int
    kind: str
    task_id: str | None = None
    actor_id: str | None = None
    payload: Mapping[str, Any] = field(default_factory=readonly_mapping)


@dataclass(frozen=True, slots=True)
class WorkScenario:
    """A reproducible job definition, including its independent verifier."""

    name: str
    tasks: tuple[WorkTask, ...]
    final_task_ids: tuple[str, ...]
    metadata: Mapping[str, Any] = field(default_factory=readonly_mapping)

    def __post_init__(self) -> None:
        object.__setattr__(self, "metadata", readonly_mapping(self.metadata))
