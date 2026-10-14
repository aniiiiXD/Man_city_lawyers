"""Deterministic shared workspace and append-only job state."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Iterable, Mapping

from work_web.contracts import (
    ApprovalRequest,
    ApprovalOrigin,
    ApprovalStatus,
    ArtifactDraft,
    ArtifactRef,
    TaskSnapshot,
    TaskStatus,
    TERMINAL_TASK_STATUSES,
    WorkTask,
)
from work_web.event_log import EventLog
from work_web.workspace import ArtifactConflict, VersionedWorkspace


class InvalidTransition(RuntimeError):
    pass


@dataclass(slots=True)
class _MutableTask:
    task: WorkTask
    status: TaskStatus
    attempt: int = 0
    created_at: int = 0
    started_at: int | None = None
    finished_at: int | None = None
    retry_at: int | None = None
    worker_id: str | None = None
    error: str | None = None
    output_refs: tuple[ArtifactRef, ...] = ()
    approval_request_id: str | None = None

    def snapshot(self) -> TaskSnapshot:
        return TaskSnapshot(
            task=self.task,
            status=self.status,
            attempt=self.attempt,
            created_at=self.created_at,
            started_at=self.started_at,
            finished_at=self.finished_at,
            retry_at=self.retry_at,
            worker_id=self.worker_id,
            error=self.error,
            output_refs=self.output_refs,
            approval_request_id=self.approval_request_id,
        )


_ALLOWED: dict[TaskStatus, frozenset[TaskStatus]] = {
    TaskStatus.QUEUED: frozenset(
        {TaskStatus.RUNNING, TaskStatus.CANCELLED, TaskStatus.TIMED_OUT, TaskStatus.FAILED}
    ),
    TaskStatus.RUNNING: frozenset(
        {
            TaskStatus.WAITING_APPROVAL,
            TaskStatus.SUCCEEDED,
            TaskStatus.FAILED,
            TaskStatus.CANCELLED,
            TaskStatus.TIMED_OUT,
            TaskStatus.QUEUED,
        }
    ),
    TaskStatus.WAITING_APPROVAL: frozenset(
        {
            TaskStatus.QUEUED,
            TaskStatus.FAILED,
            TaskStatus.CANCELLED,
            TaskStatus.TIMED_OUT,
        }
    ),
    TaskStatus.FAILED: frozenset({TaskStatus.QUEUED}),
    TaskStatus.SUCCEEDED: frozenset(),
    TaskStatus.CANCELLED: frozenset(),
    TaskStatus.TIMED_OUT: frozenset(),
}


class WorkWeb:
    """Event-sourced environment for concurrent, artifact-producing agents.

    Time is an integer virtual clock.  This object does not spawn threads or
    make network calls; the orchestrator advances it to deterministic events.
    """

    def __init__(self) -> None:
        self.now = 0
        self.events = EventLog()
        self.workspace = VersionedWorkspace()
        self._tasks: dict[str, _MutableTask] = {}
        self._approvals: dict[str, ApprovalRequest] = {}
        self._approval_counter = 0

    def register_tasks(self, tasks: Iterable[WorkTask]) -> None:
        new_tasks = tuple(tasks)
        ids = [task.task_id for task in new_tasks]
        if len(set(ids)) != len(ids):
            raise ValueError("task ids must be unique")
        if any(task_id in self._tasks for task_id in ids):
            raise ValueError("a task id is already registered")
        known_ids = set(self._tasks) | set(ids)
        for task in new_tasks:
            missing = set(task.dependencies) - known_ids
            if missing:
                raise ValueError(f"{task.task_id} has unknown dependencies: {sorted(missing)}")
        graph = {task_id: set(state.task.dependencies) for task_id, state in self._tasks.items()}
        graph.update({task.task_id: set(task.dependencies) for task in new_tasks})
        self._validate_acyclic(graph)
        for task in new_tasks:
            self._tasks[task.task_id] = _MutableTask(
                task=task, status=TaskStatus.QUEUED, created_at=self.now
            )
            self._emit("task.registered", task_id=task.task_id, role=task.role)

    @staticmethod
    def _validate_acyclic(graph: Mapping[str, set[str]]) -> None:
        visiting: set[str] = set()
        visited: set[str] = set()

        def visit(node: str) -> None:
            if node in visiting:
                raise ValueError("task dependency graph contains a cycle")
            if node in visited:
                return
            visiting.add(node)
            for dependency in graph[node]:
                visit(dependency)
            visiting.remove(node)
            visited.add(node)

        for task_id in graph:
            visit(task_id)

    @property
    def task_ids(self) -> tuple[str, ...]:
        return tuple(self._tasks)

    def task(self, task_id: str) -> TaskSnapshot:
        try:
            return self._tasks[task_id].snapshot()
        except KeyError as error:
            raise KeyError(f"unknown task: {task_id}") from error

    def tasks(self) -> tuple[TaskSnapshot, ...]:
        return tuple(state.snapshot() for state in self._tasks.values())

    def is_terminal(self, task_id: str) -> bool:
        return self.task(task_id).status in TERMINAL_TASK_STATUSES

    def pending_approvals(self) -> tuple[ApprovalRequest, ...]:
        return tuple(
            request
            for request in self._approvals.values()
            if request.status == ApprovalStatus.PENDING
        )

    def approval(self, request_id: str) -> ApprovalRequest:
        try:
            return self._approvals[request_id]
        except KeyError as error:
            raise KeyError(f"unknown approval request: {request_id}") from error

    def start_task(self, task_id: str, worker_id: str) -> TaskSnapshot:
        state = self._state(task_id)
        self._transition(state, TaskStatus.RUNNING, "task.started", worker_id=worker_id)
        state.attempt += 1
        state.started_at = self.now
        state.finished_at = None
        state.retry_at = None
        state.worker_id = worker_id
        state.error = None
        state.approval_request_id = None
        self._emit(
            "task.attempt_started",
            task_id=task_id,
            actor_id=worker_id,
            attempt=state.attempt,
        )
        return state.snapshot()

    def complete_task(
        self,
        task_id: str,
        artifacts: Iterable[ArtifactDraft] = (),
        *,
        worker_id: str | None = None,
    ) -> TaskSnapshot:
        state = self._state(task_id)
        if state.status != TaskStatus.RUNNING:
            raise InvalidTransition(f"cannot complete task in state {state.status}")
        drafts = tuple(artifacts)
        keys = [draft.key for draft in drafts]
        if len(set(keys)) != len(keys):
            raise ValueError("one task attempt cannot write the same artifact key twice")
        unexpected = set(keys) - set(state.task.output_keys)
        if unexpected:
            raise ValueError(f"undeclared output artifacts: {sorted(unexpected)}")
        current = self.workspace.latest_versions()
        for draft in drafts:
            version = current.get(draft.key, 0)
            if draft.expected_version is not None and draft.expected_version != version:
                raise ArtifactConflict(
                    f"{draft.key!r} is at version {version}; writer expected {draft.expected_version}"
                )
        records = tuple(
            self.workspace.write(
                key=draft.key,
                payload=draft.payload,
                producer_task_id=task_id,
                created_at=self.now,
                expected_version=draft.expected_version,
            )
            for draft in drafts
        )
        for record in records:
            self._emit(
                "artifact.written",
                task_id=task_id,
                actor_id=worker_id or state.worker_id,
                key=record.key,
                version=record.version,
                producer_task_id=record.producer_task_id,
            )
        state.output_refs = tuple(record.ref for record in records)
        state.finished_at = self.now
        self._transition(state, TaskStatus.SUCCEEDED, "task.succeeded", worker_id=worker_id)
        return state.snapshot()

    def hold_task(
        self,
        task_id: str,
        *,
        scope: str,
        reason: str,
        origin: ApprovalOrigin = ApprovalOrigin.SYNTHETIC_REVIEW,
        locally_resolvable: bool = True,
        plan_id: str | None = None,
        decision_id: str | None = None,
        delegation_id: str | None = None,
    ) -> ApprovalRequest:
        state = self._state(task_id)
        if state.status != TaskStatus.RUNNING:
            raise InvalidTransition(f"cannot hold task in state {state.status}")
        request_id = f"approval-{self._approval_counter + 1:04d}"
        self._approval_counter += 1
        request = ApprovalRequest(
            request_id=request_id,
            task_id=task_id,
            scope=scope,
            reason=reason,
            created_at=self.now,
            deadline_tick=min(
                self.now + state.task.approval_timeout_ticks,
                state.task.deadline_tick
                if state.task.deadline_tick is not None
                else self.now + state.task.approval_timeout_ticks,
            ),
            origin=origin,
            locally_resolvable=locally_resolvable,
            plan_id=plan_id,
            decision_id=decision_id,
            delegation_id=delegation_id,
        )
        self._approvals[request_id] = request
        state.approval_request_id = request_id
        self._transition(
            state,
            TaskStatus.WAITING_APPROVAL,
            "task.waiting_approval",
            request_id=request_id,
            scope=scope,
        )
        self._emit(
            "approval.requested",
            task_id=task_id,
            request_id=request_id,
            scope=scope,
            reason=reason,
            origin=origin.value,
            locally_resolvable=locally_resolvable,
            plan_id=plan_id,
            decision_id=decision_id,
            delegation_id=delegation_id,
        )
        return request

    def resolve_approval(
        self, request_id: str, *, granted: bool, note: str | None = None
    ) -> ApprovalRequest:
        request = self.approval(request_id)
        if request.status != ApprovalStatus.PENDING:
            raise InvalidTransition(f"approval {request_id} was already resolved")
        if not request.locally_resolvable:
            raise InvalidTransition(
                f"approval {request_id} is controlled by {request.origin.value}; "
                "local approval cannot resolve this hold"
            )
        status = ApprovalStatus.GRANTED if granted else ApprovalStatus.DENIED
        resolved = replace(request, status=status, resolved_at=self.now, note=note)
        self._approvals[request_id] = resolved
        self._emit(
            "approval.resolved",
            task_id=request.task_id,
            request_id=request_id,
            status=status.value,
            note=note,
        )
        return resolved

    def close_approval(self, request_id: str, *, status: ApprovalStatus, note: str) -> ApprovalRequest:
        if status not in {ApprovalStatus.EXPIRED, ApprovalStatus.CANCELLED}:
            raise ValueError("close_approval accepts only expired or cancelled status")
        request = self.approval(request_id)
        if request.status != ApprovalStatus.PENDING:
            return request
        closed = replace(request, status=status, resolved_at=self.now, note=note)
        self._approvals[request_id] = closed
        self._emit(
            "approval.closed",
            task_id=request.task_id,
            request_id=request_id,
            status=status.value,
            note=note,
        )
        return closed

    def resume_approved_task(self, task_id: str, *, request_id: str) -> TaskSnapshot:
        state = self._state(task_id)
        request = self.approval(request_id)
        if request.task_id != task_id or request.status != ApprovalStatus.GRANTED:
            raise InvalidTransition("task can resume only after its own approval is granted")
        if state.status != TaskStatus.WAITING_APPROVAL:
            raise InvalidTransition(f"cannot resume task in state {state.status}")
        self._transition(
            state,
            TaskStatus.QUEUED,
            "task.approval_granted",
            request_id=request_id,
        )
        state.retry_at = self.now
        state.error = None
        state.approval_request_id = None
        return state.snapshot()

    def retry_task(self, task_id: str, *, retry_at: int, reason: str) -> TaskSnapshot:
        state = self._state(task_id)
        if retry_at < self.now:
            raise ValueError("retry_at cannot be earlier than current virtual time")
        self._transition(
            state,
            TaskStatus.QUEUED,
            "task.retry_scheduled",
            retry_at=retry_at,
            reason=reason,
        )
        state.retry_at = retry_at
        state.error = reason
        state.worker_id = None
        state.approval_request_id = None
        return state.snapshot()

    def reopen_failed_task(self, task_id: str, *, reason: str) -> TaskSnapshot:
        state = self._state(task_id)
        if state.status != TaskStatus.FAILED:
            raise InvalidTransition("only failed tasks can be manually retried")
        if state.attempt >= state.task.max_attempts:
            raise InvalidTransition("task has exhausted its attempt limit")
        self._transition(state, TaskStatus.QUEUED, "task.manually_retried", reason=reason)
        state.retry_at = self.now
        state.error = reason
        state.finished_at = None
        state.worker_id = None
        return state.snapshot()

    def fail_task(self, task_id: str, *, reason: str, worker_id: str | None = None) -> TaskSnapshot:
        state = self._state(task_id)
        self._close_task_approval(state, reason=reason)
        state.finished_at = self.now
        state.error = reason
        self._transition(
            state,
            TaskStatus.FAILED,
            "task.failed",
            reason=reason,
            worker_id=worker_id,
        )
        return state.snapshot()

    def timeout_task(self, task_id: str, *, reason: str) -> TaskSnapshot:
        state = self._state(task_id)
        self._close_task_approval(state, reason=reason, status=ApprovalStatus.EXPIRED)
        state.finished_at = self.now
        state.error = reason
        self._transition(state, TaskStatus.TIMED_OUT, "task.timed_out", reason=reason)
        return state.snapshot()

    def cancel_task(self, task_id: str, *, reason: str = "Cancelled by controller") -> TaskSnapshot:
        state = self._state(task_id)
        self._close_task_approval(state, reason=reason, status=ApprovalStatus.CANCELLED)
        state.finished_at = self.now
        state.error = reason
        self._transition(state, TaskStatus.CANCELLED, "task.cancelled", reason=reason)
        return state.snapshot()

    def write_artifact(
        self,
        *,
        key: str,
        payload: Mapping[str, Any],
        producer_task_id: str,
        expected_version: int | None = None,
    ):
        record = self.workspace.write(
            key=key,
            payload=payload,
            producer_task_id=producer_task_id,
            created_at=self.now,
            expected_version=expected_version,
        )
        self._emit(
            "artifact.written",
            task_id=producer_task_id,
            key=record.key,
            version=record.version,
            producer_task_id=producer_task_id,
        )
        return record

    def advance_to(self, tick: int) -> None:
        if tick < self.now:
            raise ValueError("virtual time cannot move backwards")
        if tick == self.now:
            return
        previous = self.now
        self.now = tick
        self._emit("clock.advanced", from_tick=previous, to_tick=tick)

    def _state(self, task_id: str) -> _MutableTask:
        try:
            return self._tasks[task_id]
        except KeyError as error:
            raise KeyError(f"unknown task: {task_id}") from error

    def _close_task_approval(
        self,
        state: _MutableTask,
        *,
        reason: str,
        status: ApprovalStatus = ApprovalStatus.CANCELLED,
    ) -> None:
        if state.approval_request_id is None:
            return
        request = self._approvals.get(state.approval_request_id)
        if request is not None and request.status == ApprovalStatus.PENDING:
            self.close_approval(request.request_id, status=status, note=reason)

    def _transition(
        self,
        state: _MutableTask,
        target: TaskStatus,
        event_kind: str,
        **payload: Any,
    ) -> None:
        if target not in _ALLOWED[state.status]:
            raise InvalidTransition(f"cannot move {state.task.task_id} from {state.status} to {target}")
        prior = state.status
        state.status = target
        self._emit(
            event_kind,
            task_id=state.task.task_id,
            from_status=prior.value,
            status=target.value,
            **payload,
        )

    def _emit(
        self,
        kind: str,
        *,
        task_id: str | None = None,
        actor_id: str | None = None,
        **payload: Any,
    ) -> None:
        self.events.append(
            tick=self.now,
            kind=kind,
            task_id=task_id,
            actor_id=actor_id,
            payload=payload,
        )
